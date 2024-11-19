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


if __name__ == '__main__':
    category = sys.argv[1]
    neg_ratio = 1 if len(sys.argv) < 3 else int(sys.argv[2])

    # used_features = ['parent_asin', 'categories',
    #                  'title', 'description', 'features']
    # # item metadata
    # item_df = get_data_from_json_by_line(
    #                     json_file_path=os.path.join(work_dir, 'data/AmazonReviews2023/meta_categories', f'{category}.jsonl'),
    #                     fields=used_features)
    # item_df = item_df[~pd.isna(item_df['categories'])].copy()
    # item_df = item_df[~pd.isna(item_df['title'])].copy()
    # item_df = item_df[~pd.isna(item_df['description'])].copy()
    # item_df = item_df[~pd.isna(item_df['features'])].copy()

    # print(item_df)
    # # filter category whose number is greater than 5; pad if no category; join categories with sep
    # all_categories = []
    # for c in item_df['categories']:
    #     all_categories += c
    # print('num_categories: ', len(all_categories))
    # num_cat = defaultdict(int)
    # for c in all_categories:
    #     num_cat[c] += 1
    # categories = item_df['categories'].map(lambda x: [_ for _ in x if num_cat[_] > 5])  
    # categories = categories.map(lambda x: x if len(x) > 0 else ['[PAD]'])  
    # item_df['categories'] = categories.map(lambda x : '|||'.join(x))

    # ## process text
    # item_df['title'] = item_df['title'].map(
    #                             lambda x: x.replace('\t', ' ').replace('\n', '. ').replace('\r', '. ')) 
    # item_df['description'] = item_df['description'].map(
    #                             lambda x: ' '.join(x).replace('\t', ' ').replace('\n', '. ').replace('\r', '. ')) 
    # item_df['features'] = item_df['features'].map(
    #                             lambda x: ' '.join(x).replace('\t', ' ').replace('\n', '. ').replace('\r', '. ')) 
    
    # item_df.to_csv(
    #         os.path.join(work_dir, 'data/AmazonReviews2023/meta_categories', f'{category}.csv'),
    #         sep='\t',
    #         header=True,
    #         index=False
    #     )
    
    # interaction
    inter_df = pd.read_csv(
                os.path.join(work_dir, 'data/AmazonReviews2023/0core_pureID_inter', f'{category}.csv'),
                # os.path.join(work_dir, 'data/AmazonReviews2023/pureID_inter', f'{category}.csv'),
                sep=',',
                header=0
            )
    inter_df.to_csv(
            os.path.join(work_dir, 'data/AmazonReviews2023/0core_pureID_inter', f'new_{category}.csv'),
            # os.path.join(work_dir, 'data/AmazonReviews2023/pureID_inter', f'new_{category}.csv'),
            sep='\t',
            header=True,
            index=False
        )
    

    # get recstudio dataset
    dataset = process_by_recstudio(
                category,
                os.path.join(work_dir, 'data/AmazonReviews2023/0core_config', f'{category}.yaml') 
                # os.path.join(work_dir, 'data/AmazonReviews2023/config', f'{category}.yaml')
            )
    
    trn_set, _, tst_set = negative_sample_and_split(
                                    dataset, 
                                    val=False,
                                    max_behavior_len=dataset.config['max_seq_len'],
                                    neg_ratio=neg_ratio)

    pkl_name = f'{category}.pkl' if neg_ratio == 1 else f'{category}_neg{neg_ratio}.pkl'
    with open(os.path.join(work_dir, 'data/AmazonReviews2023/0core_dataset', pkl_name), 'wb') as f:
        pickle.dump(trn_set, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(tst_set, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
    

        