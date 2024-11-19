import os
import pickle
import pandas as pd
import copy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset, IterableDataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch import randperm
# from torch._utils import _accumulate
from collections import defaultdict
from multiprocessing import Pool
from datasets import Dataset as TFDataset
from functools import partial, reduce
from transformers.tokenization_utils_base import BatchEncoding

from collections.abc import Mapping
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import numpy as np
from transformers import (
    DataCollatorForLanguageModeling, 
    DataCollatorForWholeWordMask,
    BertTokenizer, 
    BertTokenizerFast
)
from transformers.data.data_collator import _torch_collate_batch
import operator
import random
from tqdm import tqdm
import warnings

def _accumulate(iterable, fn=lambda x, y: x + y):
    "Return running totals"
    # _accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # _accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    it = iter(iterable)
    try:
        total = next(it)
    except StopIteration:
        return
    yield total
    for element in it:
        total = fn(total, element)
        yield total
 

class CTRDataset(Dataset):
    def __init__(self, dataset, data, trunc=20, tokenizer=None, max_seq_len=512, prefix=''):
        super().__init__()
        self.dataset = dataset
        self.data = data

        self.fuid = dataset.fuid
        self.fiid = dataset.fiid
        self.frating = dataset.frating

        self.item_feat = dataset.item_feat
        self.user_feat = dataset.user_feat  

        self.trunc = trunc                
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len     
        self.prefix = prefix

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data = self.data[index]
        uid, u_bh, iid, y = data
        
        u_bh = u_bh[-self.trunc:]
        ret = {self.frating: torch.tensor(y)}
        if self.user_feat is not None:
            ret = {**ret, **self.user_feat[uid]}
        if self.item_feat is not None:
            ret = {**ret, **self.item_feat[iid]}

        user_behavior_feat = self.item_feat[u_bh]
        for field, value in user_behavior_feat.items():
            ret['in_'+field] = value

        return ret
    
    def get_tokenized_text(self):

        def tokenize_a_serie(df, field, tokenizer, max_seq_len):
            return tokenizer(
                        df[field], 
                        add_special_tokens=True, 
                        truncation=True,
                        max_length=max_seq_len, 
                        return_attention_mask=False,
                        return_token_type_ids=False
                    )
        
        item_text_fields = []
        for f, t in self.dataset.field2type.items():
            if t == 'text' and f in self.dataset.item_feat.fields:
                item_text_fields.append(f)
                item_text_feat = pd.DataFrame(self.dataset.item_feat.get_col(f))
                item_text_feat = self.prefix + item_text_feat
                item_text_feat[f][item_text_feat[f] == 0] = self.tokenizer.unk_token
                item_text_feat = TFDataset.from_pandas(item_text_feat, preserve_index=False)

                item_text_feat = item_text_feat.map(
                            partial(
                                tokenize_a_serie,
                                field=f,
                                tokenizer=self.tokenizer,
                                max_seq_len=self.max_seq_len
                            ),
                            remove_columns=[f],
                            batched=True
                        )
                self.dataset.item_feat.data[f] = item_text_feat['input_ids']
        
        self.dataset.item_feat.data['text'] = [reduce(operator.add, _) 
                                                for _ in zip(*[
                                                                self.dataset.item_feat.data[_] 
                                                                for _ in item_text_fields
                                                            ])
                                                ]
        if 'text' not in self.dataset.item_feat.fields:
            self.dataset.item_feat.fields.append('text')
        self.dataset.field2type['text'] = 'text'

        self.dataset.item_feat.del_fields(keep_fields=
                                          {'text'}.union(
                                              set(self.dataset.item_feat.fields) - \
                                            set(item_text_fields))
                                        )
        


class ContrastiveDataset(Dataset):
    def __init__(self, dataset, tokenizer, loss, max_seq_len=512, pairs=None, negatives=None, negatives_per_pair=5, prefix=''):
        super().__init__()
        self.name = dataset.name
        self.text_fields = [f for f, t in dataset.field2type.items() if t == 'text']
        if 'text' in self.text_fields:
            self.text_fields = ['text']
        self.item_feat = dataset.item_feat
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.loss = loss
        self.prefix = prefix
        if self.loss == ['feature_pair']:
            self.data_index = torch.arange(0, len(self.item_feat))
        else:
            self.pairs = list(pairs)
            self.negatives = negatives
            self.negatives_per_pair = negatives_per_pair
            self.data_index = torch.arange(0, len(pairs))

        self.get_tokenized_text()

    def __iter__(self):
        if self.loss != ['feature_pair']:
            self.pairs = [(y, x) for x, y in self.pairs]
        return self

    def __len__(self):
        return len(self.data_index)
    
    def __getitem__(self, index):
        idx = self.data_index[index]
        if self.loss == ['feature_pair']:
            ret = self.item_feat[idx]
        else:
            item_i, item_j = self.pairs[idx]
            ret_item_i = self.item_feat[item_i]
            ret_item_j = self.item_feat[item_j]
            ret = {
                'anchor': ret_item_i['text'],
                'positive': ret_item_j['text']
            }
            if self.negatives is not None:
                item_i_neg = random.sample(self.negatives[item_i], self.negatives_per_pair)
                ret['negative'] = [self.item_feat.get_col('text')[_] for _ in item_i_neg]

            if 'guided' in self.loss:
                ret = {
                    **ret,
                    'guided_anchor_emb_idx': torch.tensor(item_i),
                    'guided_positive_emb_idx': torch.tensor(item_j),
                }
                if self.negatives is not None:
                    ret['guided_negative_emb_idx'] = torch.tensor(item_i_neg)

            if 'feature_pair' in self.loss:
                for k in self.text_fields:      # except text
                    ret[k] = ret_item_i[k]      # ignore j


        return ret
    
    def build(self, split_ratio):
        lens = [int(len(self) * _) for _ in split_ratio]
        lens[0] = len(self) - sum(lens[1:])
        splits = []
        indices = randperm(len(self), generator=torch.Generator().manual_seed(42))
        for offset, length in zip(_accumulate(lens), lens):
            splits.append(indices[offset - length : offset])
        return [self._copy(_) for _ in splits]
    
    def _copy(self, data_index):
        d = copy.copy(self)
        d.data_index = data_index
        return d
    
    def get_tokenized_text(self):

        def tokenize_a_serie(df, field, tokenizer, max_seq_len):
            return tokenizer(
                        df[field], 
                        add_special_tokens=True, 
                        truncation=True,
                        max_length=max_seq_len, 
                        return_attention_mask=False,
                        return_token_type_ids=False
                    )
        if 'text' in self.item_feat.data:
            f = 'text'
            item_text_feat = pd.DataFrame(self.item_feat.get_col(f))
            item_text_feat = self.prefix + item_text_feat
            item_text_feat[f][item_text_feat[f] == 0] = self.tokenizer.unk_token
            item_text_feat = TFDataset.from_pandas(item_text_feat, preserve_index=False)

            item_text_feat = item_text_feat.map(
                        partial(
                            tokenize_a_serie,
                            field=f,
                            tokenizer=self.tokenizer,
                            max_seq_len=self.max_seq_len
                        ),
                        remove_columns=[f],
                        batched=True
                    )
            self.item_feat.data[f] = [self.tokenizer.truncate_sequences(
                                        _,
                                        truncation_strategy='longest_first',
                                        num_tokens_to_remove=len(_) - self.max_seq_len
                                    )[0]
                                    for _ in item_text_feat['input_ids']]


            self.item_feat.del_fields(keep_fields={'text'})
        else:
            for f in self.text_fields:
                item_text_feat = pd.DataFrame(self.item_feat.get_col(f))
                item_text_feat[f][item_text_feat[f] == 0] = self.tokenizer.unk_token
                item_text_feat = TFDataset.from_pandas(item_text_feat, preserve_index=False)

                item_text_feat = item_text_feat.map(
                            partial(
                                tokenize_a_serie,
                                field=f,
                                tokenizer=self.tokenizer,
                                max_seq_len=self.max_seq_len
                            ),
                            remove_columns=[f],
                            batched=True
                        )
                self.item_feat.data[f] = [self.tokenizer.truncate_sequences(
                                            _,
                                            truncation_strategy='longest_first',
                                            num_tokens_to_remove=len(_) - self.max_seq_len
                                        )[0]
                                        for _ in item_text_feat['input_ids']]

            if self.loss == ['feature_pair']:
                self.item_feat.del_fields(keep_fields=set(self.text_fields))
            else:
                self.item_feat.data['text'] = [reduce(operator.add, _) 
                                                for _ in zip(*[
                                                                self.item_feat.data[_] 
                                                                for _ in self.text_fields
                                                            ])
                                                ]
                
                self.item_feat.data['text'] = [self.tokenizer.truncate_sequences(
                                                _,
                                                truncation_strategy='longest_first',
                                                num_tokens_to_remove=len(_) - self.max_seq_len
                                            )[0]
                                            for _ in self.item_feat.data['text']]

                self.item_feat.del_fields(keep_fields={'text'}.union(set(self.text_fields)))
        
    

class SupervisedDataset_HardPrompt(Dataset):

    def __init__(self, dataset, data, trunc=20, tokenizer=None, max_seq_len=512, logger=None, saved_path=None, sample_ratio=1.0):
        super().__init__()
        self.dataset = dataset

        self.fiid = dataset.fiid
        self.frating = dataset.frating

        self.item_feat = dataset.item_feat 

        self.trunc = trunc                
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len 
        self.logger = logger

        if self.dataset.name == 'MIND':
            self.hardprompt = "The user has clicked the following news: {}."
            self.text_fields = ['title']  

        self._preprocess(data, saved_path, sample_ratio)

    def _preprocess(self, data, saved_path, sample_ratio):
        
        def tokenize_a_serie(df, field, tokenizer, max_seq_len):
            return tokenizer(
                            df[field], 
                            add_special_tokens=True, 
                            truncation=True,
                            max_length=max_seq_len, 
                            return_attention_mask=False,
                            return_token_type_ids=False
                        )
        
        def remove_duplicates(lst):
            seen = set()
            result = []
            for item in lst:
                if item not in seen:
                    seen.add(item)
                    result.append(item)
            return result
        

        self.dataset.item_feat.del_fields(keep_fields=set(self.text_fields))  

        if os.path.exists(saved_path):
            self.data = pickle.load(open(saved_path, 'rb'))
            return

        len_hardprompt = len(self.tokenizer(self.hardprompt)['input_ids']) - 2
        available_tokens = self.max_seq_len - len_hardprompt
        sum_len_u_bh = 0

        self.data = defaultdict(list)
        pre_u_bh = []
        for uid, u_bh, iid, y in tqdm(data):
            uni_u_bh = remove_duplicates(u_bh)
            if y > 0 and len(uni_u_bh) > 10 and random.random() < sample_ratio:
            # if y > 0 and  len(u_bh) > 0
                if uni_u_bh != pre_u_bh:
                    hardprompt_u_bh = []
                    for bh in uni_u_bh[-self.trunc:]:
                        if len(self.text_fields) == 1:
                            hardprompt_u_bh.append(
                                self.item_feat[bh][self.text_fields[0]])
                        else:
                            hardprompt_u_bh.append(
                                ', '.join(k + ': ' + self.item_feat[bh][k] for k in self.text_fields))
                    
                    for i in range(len(hardprompt_u_bh)):
                        if len(' | '.join(hardprompt_u_bh)) > available_tokens:
                            hardprompt_u_bh = hardprompt_u_bh[1:]
                        else:
                            sum_len_u_bh += len(hardprompt_u_bh)
                            hardprompt_u_bh = ' | '.join(hardprompt_u_bh)
                            break

                    hardprompt_u_bh = self.hardprompt.format(hardprompt_u_bh)
                    pre_u_bh = uni_u_bh
                    
                self.data['user_behaviors'].append(hardprompt_u_bh) 
                if len(self.text_fields) == 1:
                    self.data['target_item'].append(self.item_feat[iid][self.text_fields[0]])
                else:
                    self.data['target_item'].append(
                        ', '.join(k + ': ' + self.item_feat[bh][k] for k in self.text_fields))
        
        self.logger.info(f"Average length of user behaviors: {sum_len_u_bh / len(self.data['target_item'])}.")
        self.data = pd.DataFrame(self.data)

        for f in ['user_behaviors', 'target_item']:
            text_feat = pd.DataFrame(self.data[f])
            text_feat[f][text_feat[f] == 0] = self.tokenizer.unk_token
            text_feat = TFDataset.from_pandas(text_feat, preserve_index=False)

            text_feat = text_feat.map(
                        partial(
                            tokenize_a_serie,
                            field=f,
                            tokenizer=self.tokenizer,
                            max_seq_len=self.max_seq_len
                        ),
                        remove_columns=[f],
                        batched=True
                    )
            self.data[f] = text_feat['input_ids']

        with open(saved_path, 'wb') as f:
            pickle.dump(self.data, f)

    def __len__(self):
        return len(self.data['target_item'])

    def __getitem__(self, index):
        ret = {
            'user_behaviors': self.data['user_behaviors'][index],
            'target_item': self.data['target_item'][index],
        }

        return ret
    

class SupervisedDataset_HardPrompt_Expand(SupervisedDataset_HardPrompt, Dataset):
    
    def __init__(self, dataset, data, trunc=20, tokenizer=None, max_seq_len=512, logger=None, saved_path=None, sample_ratio=1.0):
        Dataset.__init__(self)
        self.dataset = dataset

        self.fiid = dataset.fiid
        self.frating = dataset.frating

        self.item_feat = dataset.item_feat 

        self.trunc = trunc                
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len 
        self.logger = logger

        if self.dataset.name == 'MIND':
            self.hardprompt = "The user has clicked the following news: {}, the next clicked news is [TARGET_ITEM]."
            self.text_fields = ['title']  

        self._preprocess(data, saved_path, sample_ratio)


class SupervisedDataset_Pooling(SupervisedDataset_HardPrompt):
    
    def _preprocess(self, data, saved_path=None, sample_ratio=1.0):

        def tokenize_a_serie(df, field, tokenizer, max_seq_len):
            return tokenizer(
                            df[field], 
                            add_special_tokens=True, 
                            truncation=True,
                            max_length=max_seq_len, 
                            return_attention_mask=False,
                            return_token_type_ids=False
                        )
        
        def remove_duplicates(lst):
            seen = set()
            result = []
            for item in lst:
                if item not in seen:
                    seen.add(item)
                    result.append(item)
            return result
        
        
        if os.path.exists(saved_path):
            self.data = pickle.load(open(saved_path, 'rb'))
        else:
            self.data = []
            for uid, u_bh, iid, y in tqdm(data):
                uni_u_bh = remove_duplicates(u_bh)
                if y > 0 and len(uni_u_bh) > 10 and random.random() < sample_ratio:
                    self.data.append((uni_u_bh, iid))

            with open(saved_path, 'wb') as f:
                pickle.dump(self.data, f)

        if self.item_feat.fields != ['text']:
            for f in self.text_fields:
                text_feat = pd.DataFrame(self.item_feat.get_col(f))
                text_feat[f][text_feat[f] == 0] = self.tokenizer.unk_token
                text_feat = TFDataset.from_pandas(text_feat, preserve_index=False)

                text_feat = text_feat.map(
                            partial(
                                tokenize_a_serie,
                                field=f,
                                tokenizer=self.tokenizer,
                                max_seq_len=self.max_seq_len
                            ),
                            remove_columns=[f],
                            batched=True
                        )
                self.item_feat.data[f] = text_feat['input_ids']
        
            self.item_feat.data['text'] = [reduce(operator.add, _) 
                                                    for _ in zip(*[
                                                                    self.item_feat.data[_] 
                                                                    for _ in self.text_fields
                                                                ])
                                                    ]
            self.item_feat.fields.append('text')
            self.item_feat.del_fields(keep_fields={'text'})

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data = self.data[index]
        u_bh, iid = data
        u_bh = u_bh[-self.trunc:]
        ret = {
            'user_behaviors': [self.item_feat[_]['text'] for _ in u_bh],
            'target_item': self.item_feat[iid]['text'],
        }
        return ret


class SupervisedDataset_MLM(Dataset):

    def __init__(self, dataset, data, trunc=20, tokenizer=None, max_seq_len=512, logger=None, saved_path=None, sample_ratio=1.0):
        super().__init__()
        self.dataset = dataset

        self.fiid = dataset.fiid
        self.frating = dataset.frating

        self.item_feat = dataset.item_feat 

        self.trunc = trunc                
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len 
        self.logger = logger

        if self.dataset.name == 'MIND':
            self.hardprompt = "The user has clicked the following news: {}. The next news clicked is: {}."
            self.text_fields = ['title']  

        self._preprocess(data, saved_path, sample_ratio)

    def _preprocess(self, data, saved_path, sample_ratio):
        
        def tokenize_a_serie(df, field, tokenizer, max_seq_len):
            return tokenizer(
                            df[field], 
                            add_special_tokens=True,
                            truncation=True,         
                            max_length=max_seq_len, 
                            return_attention_mask=False,
                            return_token_type_ids=False
                        )

        def find_sublist_end_index(main_list, sublist):
            sublist_len = len(sublist)
            for i in range(len(main_list) - sublist_len + 1):
                if main_list[i:i + sublist_len] == sublist:
                    return i + sublist_len - 1
            return -1
        
        def remove_duplicates(lst):
            seen = set()
            result = []
            for item in lst:
                if item not in seen:
                    seen.add(item)
                    result.append(item)
            return result

        self.dataset.item_feat.del_fields(keep_fields=set(self.text_fields))  

        if os.path.exists(saved_path):
            self.data = pickle.load(open(saved_path, 'rb'))
            return

        len_hardprompt = len(self.tokenizer(self.hardprompt)['input_ids']) - 4
        available_tokens = self.max_seq_len - len_hardprompt
        sum_len_u_bh = 0

        self.data = defaultdict(list)
        pre_u_bh = []
        for uid, u_bh, iid, y in tqdm(data):
            uni_u_bh = remove_duplicates(u_bh)
            if y > 0 and len(uni_u_bh) > 10 and random.random() < sample_ratio:
            # if y > 0 and len(u_bh) > 0:
                if len(self.text_fields) == 1:
                    target_item = self.item_feat[iid][self.text_fields[0]]
                else:
                    target_item = ', '.join(k + ': ' + self.item_feat[bh][k] for k in self.text_fields)
                len_target_item = len(self.tokenizer(target_item)['input_ids'])

                if uni_u_bh != pre_u_bh:
                    hardprompt_u_bh = []
                    for bh in uni_u_bh[-self.trunc:]:
                        if len(self.text_fields) == 1:
                            hardprompt_u_bh.append(
                                self.item_feat[bh][self.text_fields[0]])
                        else:
                            hardprompt_u_bh.append(
                                ', '.join(k + ': ' + self.item_feat[bh][k] for k in self.text_fields))
                    
                    for i in range(len(hardprompt_u_bh)):
                        if len(' | '.join(hardprompt_u_bh)) > available_tokens - len_target_item:
                            hardprompt_u_bh = hardprompt_u_bh[1:]
                        else:
                            break

                    pre_u_bh = uni_u_bh

                sum_len_u_bh += len(hardprompt_u_bh)

                hardprompt = self.hardprompt.format(
                                                    ' | '.join(hardprompt_u_bh), 
                                                    target_item)
                self.data['hardprompt'].append(hardprompt)
                
                
        
        self.logger.info(f"Average length of user behaviors: {sum_len_u_bh / len(self.data['hardprompt'])}.")
        self.data = pd.DataFrame(self.data)

        f = 'hardprompt'
        text_feat = pd.DataFrame(self.data[f])
        text_feat[f][text_feat[f] == 0] = self.tokenizer.unk_token
        text_feat = TFDataset.from_pandas(text_feat, preserve_index=False)
        text_feat = text_feat.map(
                    partial(
                        tokenize_a_serie,
                        field=f,
                        tokenizer=self.tokenizer,
                        max_seq_len=self.max_seq_len
                    ),
                    remove_columns=[f],
                    batched=True
                )
        self.data[f] = text_feat['input_ids']

        pattern = self.tokenizer('The next news clicked is:',  add_special_tokens=False)['input_ids']

        target_item_idx = []
        for hp in self.data['hardprompt']:
            target_item_start_idx = find_sublist_end_index(hp, pattern) + 1
            target_item_end_idx = len(hp) - 2                                                   # [SEP] 
            target_item_idx.append((target_item_start_idx, target_item_end_idx))                # [)
        
        self.data['target_item_idx'] = target_item_idx

        with open(saved_path, 'wb') as f:
            pickle.dump(self.data, f)

    def __len__(self):
        return len(self.data['hardprompt'])

    def __getitem__(self, index):
        return self.data['hardprompt'][index], self.data['target_item_idx'][index]
        # ret = {
        #     'hardprompt': self.data['hardprompt'][index],
        #     'len_target_item': self.data['len_target_item'][index],
        # }

        # return ret
    


class Collator:

    def __init__(self, tokenizer=None, guide_model: Optional[nn.Embedding | Dict] = None):
        self.tokenizer = tokenizer
        self.guide_model = guide_model

    def __call__(self, batch):
        d = defaultdict(list)
        domain = None
        for _ in batch:
            for field, value in _.items():
                if field == 'DOMAIN':
                    assert domain is None or value == domain
                    domain = value
                    continue
                d[field].append(value)
        
        ret = {}
        for field, value in d.items():
            if isinstance(value[0], list):
                if isinstance(value[0][0], list):
                    # assert False, 'waiting to be checked.'
                    tmp = defaultdict(list)
                    max_behavior_len = 0
                    max_seq_len = 0
                    for u_bh_text in value:
                        for k, v in dict(
                                        self.tokenizer.pad(
                                                {'input_ids': u_bh_text},
                                                padding=True,
                                                return_tensors='pt'
                                            )
                                        ).items():
                            max_behavior_len = max(max_behavior_len, v.shape[0])
                            max_seq_len = max(max_seq_len, v.shape[1])
                            tmp[k].append(v)     
                            # tmp: {
                            #           'input_ids': list with a length of bs, each element is a Tensor with shape behavior_len x seq_len
                            #           'attention_mask': list with a length of bs, each element is a Tensor with shape behavior_len x seq_le
                            # }
                    for k, v in tmp.items():
                        for i, _ in enumerate(v):
                            behavior_len, seq_len = _.shape
                            tmp[k][i] = F.pad(_, 
                                              [0, max_seq_len - seq_len, 
                                               0, max_behavior_len - behavior_len])
                        tmp[k] = pad_sequence(v, batch_first=True, padding_value=0)

                    ret[field] = tmp 
                else:
                    ret[field] = dict(
                                    self.tokenizer.pad(
                                        {'input_ids': value},
                                        padding=True,
                                        return_tensors='pt'
                                    )
                                )
            elif value[0].dim() == 0:
                ret[field] = torch.tensor(value)
            else:
                ret[field] = pad_sequence(value, batch_first=True, padding_value=0)

        if self.guide_model is not None and \
            not (isinstance(self.guide_model, dict) and len(self.guide_model) == 0):
            
            if domain is None:
                guide_model = self.guide_model
            else:
                guide_model = self.guide_model[domain]

            ret['guided_anchor_emb'] = guide_model(ret.pop('guided_anchor_emb_idx'))
            ret['guided_positive_emb'] = guide_model(ret.pop('guided_positive_emb_idx'))
            if 'guided_negative_emb_idx' in ret:
                ret['guided_negative_emb'] = guide_model(ret.pop('guided_negative_emb_idx'))

        if 'DOMAIN' in ret:
            ret['DOMAIN'] = domain
            
        return ret
    


class MLMCollator(DataCollatorForWholeWordMask):

    def __init__(self, mask_target_item, **kwargs):
        super().__init__(**kwargs)
        self.mask_target_item = mask_target_item

    def __call__(self, examples):
        input_ids = [e[0] for e in examples]
        batch_input = _torch_collate_batch(input_ids, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)

        mask_labels = []
        for e in examples:
            ref_tokens = [self.tokenizer._convert_id_to_token(id) for id in e[0]]
            mask_labels.append(self._whole_word_mask(ref_tokens, e[1])) # add target item idx

        batch_mask = _torch_collate_batch(mask_labels, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
        inputs, labels = self.torch_mask_tokens(batch_input, batch_mask)
        return {"input_ids": inputs, "labels": labels}


    def _whole_word_mask(self, input_tokens, target_item_idx, max_predictions=512):
        """
        Get 0/1 labels for masked tokens with whole word mask proxy

        if mask_target_item, only mask all target item tokens;
        else random mask tokens
        """
        if not isinstance(self.tokenizer, (BertTokenizer, BertTokenizerFast)):
            warnings.warn(
                "DataCollatorForWholeWordMask is only suitable for BertTokenizer-like tokenizers. "
                "Please refer to the documentation for more information."
            )

        masked_lms = []
        covered_indexes = set()

        cand_indexes = []
        for i, token in enumerate(input_tokens):
            if token == "[CLS]" or token == "[SEP]":
                continue

            if self.mask_target_item and i in range(*target_item_idx):
                covered_indexes.add(i)
                masked_lms.append(i)
            else:
                if len(cand_indexes) >= 1 and token.startswith("##"):
                    cand_indexes[-1].append(i)
                else:
                    cand_indexes.append([i])

        if not self.mask_target_item:
            random.shuffle(cand_indexes)
            num_to_predict = min(max_predictions, max(1, int(round(len(input_tokens) * self.mlm_probability))))
            for index_set in cand_indexes:
                if len(masked_lms) >= num_to_predict:
                    break
                # If adding a whole-word mask would exceed the maximum number of
                # predictions, then just skip this candidate.
                if len(masked_lms) + len(index_set) > num_to_predict:
                    continue
                is_any_index_covered = False
                for index in index_set:
                    if index in covered_indexes:
                        is_any_index_covered = True
                        break
                if is_any_index_covered:
                    continue
                for index in index_set:
                    covered_indexes.add(index)
                    masked_lms.append(index)

        if len(covered_indexes) != len(masked_lms):
            raise ValueError("Length of covered_indexes is not equal to length of masked_lms.")
        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]
        return mask_labels


class ContrastiveDatasetMix(ContrastiveDataset, Dataset):

    def __init__(self, dataset, tokenizer, loss, max_seq_len=512, pairs=None, prefix=''):
        Dataset.__init__(self)
        self.name = dataset.name
        self.text_fields = [f for f, t in dataset.field2type.items() if t == 'text']
        if 'text' in self.text_fields:
            self.text_fields = ['text']
        self.item_feat = dataset.item_feat
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.loss = loss
        self.prefix = prefix
        
        self.pairs = list(pairs)
        self.negatives = None
        self.data_index = torch.arange(0, len(pairs))

        self.get_tokenized_text()

    
    def __getitem__(self, index):
        ret = ContrastiveDataset.__getitem__(self, index)
        ret['DOMAIN'] = self.name
        return ret


    # def get_tokenized_text(self):

    #     def tokenize_a_serie(df, field, tokenizer, max_seq_len):
    #         return tokenizer(
    #                     df[field], 
    #                     add_special_tokens=True, 
    #                     truncation=True,
    #                     max_length=max_seq_len, 
    #                     return_attention_mask=False,
    #                     return_token_type_ids=False
    #                 )
    #     if 'text' in self.item_feat.data:
    #         f = 'text'
    #         item_text_feat = pd.DataFrame(self.item_feat.get_col(f))
    #         item_text_feat[f][item_text_feat[f] == 0] = self.tokenizer.unk_token
    #         item_text_feat = TFDataset.from_pandas(item_text_feat, preserve_index=False)

    #         item_text_feat = item_text_feat.map(
    #                     partial(
    #                         tokenize_a_serie,
    #                         field=f,
    #                         tokenizer=self.tokenizer,
    #                         max_seq_len=self.max_seq_len
    #                     ),
    #                     remove_columns=[f],
    #                     batched=True
    #                 )
    #         self.item_feat.data[f] = [self.tokenizer.truncate_sequences(
    #                                     _,
    #                                     truncation_strategy='longest_first',
    #                                     num_tokens_to_remove=len(_) - self.max_seq_len
    #                                 )[0]
    #                                 for _ in item_text_feat['input_ids']]


    #         self.item_feat.del_fields(keep_fields={'text'})
    #     else:
    #         ContrastiveDataset.get_tokenized_text(self)

        
class DomainConcatDataset(ConcatDataset):

    def __init__(self, datasets, scale_dict: dict = None):
        super().__init__(datasets)
        self.scale = scale_dict
        self.name2idx = {d.name: k for k, d in enumerate(self.datasets)}

    def __len__(self):
        if self.scale is None:
            return self.cumulative_sizes[-1]
        else:
            len_ = 0
            for d in self.datasets:
                len_ += len(d) * self.scale[d.name]
            return len_