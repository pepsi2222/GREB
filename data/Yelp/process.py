import os
import pandas as pd
import pickle
import re
import sys
import numpy as np
from tqdm import tqdm
work_dir = re.search('(.*GRE).*', os.getcwd(), re.IGNORECASE).group(1)
sys.path.append(work_dir)
sys.path.append(os.path.join(work_dir, 'RecStudio'))
from RecStudio.recstudio.data import SeqDataset
from RecStudio.recstudio.data.dataset import TensorFrame
from RecStudio.recstudio.utils import parser_yaml
from RecStudio.recstudio.utils import *
from collections import defaultdict
import random
import json
from datetime import datetime
import csv
csv.field_size_limit(sys.maxsize)

from transformers import set_seed
set_seed(42)


def get_data_from_json_by_line(json_file_path, fields):
    data = defaultdict(list)
    with open(json_file_path, 'r') as rf:
        while True:
            datum = rf.readline()
            if not datum:
                break

            try:
                datum =json.loads(datum)
            except Exception as e:
                print(e, datum)
                continue
                
            if not set(fields).issubset(set(datum.keys())):
                continue
            for f in fields:
                data[f].append(datum[f])
    df = pd.DataFrame(data)
    return df


def process_by_recstudio(dataset, data_config_path):
    data_conf = parser_yaml(data_config_path)
    dataset = SeqDataset(name=dataset, config=data_conf)
    dataset.inter_feat.sort_values(by=[dataset.ftime], inplace=True)
    if data_conf['binarized_rating_thres'] is not None:
        dataset._binarize_rating(data_conf['binarized_rating_thres'])
    return dataset



def negative_sample_and_split(dataset, val=False, max_behavior_len=1e5, neg_ratio=1):

    def sample_a_negative(pos_list, num_iid):
        while True:
            neg_id = random.randint(0, num_iid - 1)
            if neg_id not in pos_list:
                return neg_id
    
    trn_set = []
    val_set = []
    tst_set = []
    num_iid = dataset.num_values(dataset.fiid)
    for uid, hist in tqdm(dataset.inter_feat.groupby(dataset.fuid)):
        pos_list = hist[dataset.fiid].tolist()
        for i in range(1, len(pos_list)):
            if i > max_behavior_len:
                start = i - max_behavior_len
            else:
                start = 0

            u_bh = pos_list[start : i]
            pos_iid = pos_list[i]
            pos_samples = [(uid, u_bh, pos_iid, 1.0)]

            neg_iids = []
            for _ in range(neg_ratio):
                neg_iid = sample_a_negative(pos_list, num_iid)
                neg_iids.append(neg_iid)
            neg_samples = [(uid, u_bh, neg_iid, 0.0) for neg_iid in neg_iids]

            if val:
                if i < len(pos_list) - 2:
                    trn_set += pos_samples
                    trn_set += neg_samples
                elif i == len(pos_list) - 2:
                    val_set += pos_samples
                    val_set += neg_samples
                else:
                    tst_set += pos_samples
                    tst_set += neg_samples
            else:
                if i < len(pos_list) - 1:
                    trn_set += pos_samples
                    trn_set += neg_samples
                else:
                    tst_set += pos_samples
                    tst_set += neg_samples

    random.shuffle(trn_set)
    return trn_set, val_set, tst_set



def no_sample_and_split(dataset, max_behavior_len=1e5, split_ratio=[0.8, 0.0, 0.2]):
    trn_set = []
    val_set = []
    tst_set = []
    for uid, hist in tqdm(dataset.inter_feat.groupby(dataset.fuid)):
        u_trn_set = []
        u_val_set = []
        u_tst_set = []

        all_list = hist[dataset.fiid].tolist()
        all_y = hist[dataset.frating].tolist()
        num_pos_bh = [0] + np.cumsum(all_y[:-1]).tolist()

        num_nonzero = sum([1 if _ > 0 else 0 for _ in num_pos_bh])
        splits = np.outer(
                    num_nonzero, 
                    split_ratio
                ).astype(np.int32).flatten().tolist()
        splits[0] = num_nonzero - sum(splits[1:])
        for i in range(1, len(split_ratio)):
            if (split_ratio[-i] != 0) & (splits[-i] == 0) & (splits[0] > 1):
                splits[-i] += 1
                splits[0] -= 1


        for i, n_p in enumerate(num_pos_bh):
            if n_p == 0:
                continue

            if n_p > max_behavior_len:
                start = num_pos_bh.index(n_p - max_behavior_len)
            else:
                start = 0

            u_bh = []
            for iid, y in zip(all_list[start : i], all_y[start : i]):
                if y == 1.0:
                    u_bh.append(iid)
            iid = all_list[i]
            y = all_y[i]

            if len(u_trn_set) < splits[0]:
                u_trn_set.append((uid, u_bh, iid, y))
            elif len(u_val_set) < splits[1]:
                u_val_set.append((uid, u_bh, iid, y))
            else:
                u_tst_set.append((uid, u_bh, iid, y))

        trn_set += u_trn_set
        val_set += u_val_set
        tst_set += u_tst_set

    random.shuffle(trn_set)
    return trn_set, val_set, tst_set




if __name__ == '__main__':
    # used_features = ['business_id', 'name', 'city', 'state', 'attributes', 'categories']

    # # item metadata
    # item_df = get_data_from_json_by_line(
    #             os.path.join(work_dir, 'data/Yelp/business.json'),
    #             fields=used_features
    #         )
    # item_df['attributes'] = item_df['attributes'].map(lambda x: str(x))
    # print(item_df)
    
    # item_df.to_csv(
    #         os.path.join(work_dir, 'data/Yelp/meta_item.csv'),
    #         sep='\t',
    #         header=True,
    #         index=False
    #     )
    
    # # user metadata
    # user_df = get_data_from_json_by_line(
    #             os.path.join(work_dir, 'data/Yelp/user.json'),
    #             fields=['user_id', 'fans']
    #         )
    # print(user_df)
    
    # user_df.to_csv(
    #         os.path.join(work_dir, 'data/Yelp/meta_user.csv'),
    #         sep='\t',
    #         header=True,
    #         index=False
    #     )
    
    # interaction
    inter_df = get_data_from_json_by_line(
                os.path.join(work_dir, 'data/Yelp/review.json'),
                # fields=['user_id', 'business_id', 'date']
                fields=['user_id', 'business_id', 'date', 'stars']
            )
    inter_df['date'] = inter_df['date'].map(
                                lambda x: int(datetime.strptime(x, '%Y-%m-%d %H:%M:%S').timestamp()))
    
    print(inter_df)
    inter_df.to_csv(
            os.path.join(work_dir, 'data/Yelp/inter_w_rating.csv'),
            sep='\t',
            header=True,
            index=False
        )

    # get recstudio dataset
    dataset = process_by_recstudio(
                'Yelp', 
                os.path.join(work_dir, 'data/Yelp/Yelp.yaml'))
    
    # trn_set, _, tst_set = negative_sample_and_split(
    #                                 dataset, 
    #                                 val=False,
    #                                 max_behavior_len=dataset.config['max_seq_len'])
    
    trn_set, _, tst_set = no_sample_and_split(
                                    dataset, 
                                    max_behavior_len=dataset.config['max_seq_len'])

    with open(os.path.join(work_dir, 'data/Yelp/Yelp_rating.pkl'), 'wb') as f:
        pickle.dump(trn_set, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(tst_set, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
    

    print(
        len(trn_set), 
        len(tst_set), 
        dataset.inter_feat.groupby(dataset.fuid).count()[dataset.fiid].mean(),
        len(dataset.inter_feat),
        len(dataset.item_feat)
    )    
    # 7635874 574232 15.297541760124828 4392169 150347