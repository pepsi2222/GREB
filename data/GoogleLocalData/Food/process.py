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
import operator
from functools import reduce
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
    category = sys.argv[1]
    # neg_ratio = 1 if len(sys.argv) < 3 else int(sys.argv[2])

    # used_features = ['gmap_id', 'name', 'category']
    # # item metadata
    # item_df = pd.read_csv(
    #                 os.path.join(work_dir, 'data/GoogleLocalData/meta', f'{category}.csv'),
    #                 sep='\t',
    #                 header=0
    #             )
    # item_df['category'] = item_df['category'].map(lambda x: x.split('|'))

    # food_words = ['restaurant', 'sandwich', 'salad', 'snack', 'bistro', 
    #               'grill', 'steak', 'noodle', 'food court', 'cafeteria',
    #               'pastry', 'pasta', 'diner', 'gastropub', 'eatery', 
    #               'western food', 'pizza']
    # is_food = item_df['category'].map(
    #             lambda x : reduce(
    #                             operator.or_, 
    #                             [reduce(operator.or_, [food_word in c.lower() for food_word in food_words]) for c in x]
    #                         )
    #             )
    # item_df = item_df[is_food].copy()

    # item_df['category'] = item_df['category'].map(lambda x : '|'.join(x))
    # item_df.to_csv(
    #         os.path.join(work_dir, 'data/GoogleLocalData/Food/meta', f'{category}.csv'),
    #         sep='\t',
    #         header=True,
    #         index=False
    #     )
    
    # food_gmap_ids = set(item_df['gmap_id'].tolist())
    
    # # interaction
    # inter_df = pd.read_csv(
    #                 os.path.join(work_dir, 'data/GoogleLocalData/review', f'{category}.csv'),
    #                 sep='\t',
    #                 header=0
    #             )
    # is_food = inter_df['gmap_id'].map(lambda x: x in food_gmap_ids)
    # inter_df = inter_df[is_food].copy()
    # inter_df.to_csv(
    #         os.path.join(work_dir, 'data/GoogleLocalData/Food/review', f'{category}.csv'),
    #         sep='\t',
    #         header=True,
    #         index=False
    #     )
    

    # get recstudio dataset
    dataset = process_by_recstudio(
                category + '_Food',
                os.path.join(work_dir, 'data/GoogleLocalData/Food/config', f'{category}.yaml') 
            )

    trn_set, _, tst_set = no_sample_and_split(
                                    dataset, 
                                    max_behavior_len=dataset.config['max_seq_len'])
    

    pkl_name = f'{category}.pkl' #if neg_ratio == 1 else f'{category}_neg{neg_ratio}.pkl'
    with open(os.path.join(work_dir, 'data/GoogleLocalData/Food/rating_dataset', pkl_name), 'wb') as f:
        pickle.dump(trn_set, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(tst_set, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
    
    print(
        category,
        len(trn_set), 
        len(tst_set), 
        dataset.inter_feat.groupby(dataset.fuid).count()[dataset.fiid].mean(),
        len(dataset.inter_feat),
        len(dataset.item_feat)
    )  
    # Alabama 2016233 449812 14.517787757440162 2686415 10815  
    # Alaska 161056 35619 11.982221731448764 217022 1603
    # California 16951683 3802576 15.687077317785564 22390840 96744
    # Connecticut 985544 218722 13.449887617490806 1316475 9044
    # Delaware 318966 70364 12.424458347816012 428942 2361
    # District_of_Columbia 254993 57003 10.82256458138811 347729 2821
    # Florida 13222853 2964292 14.91823028433192 17581388 54734
    # Hawaii 451663 100688 10.649303452453058 615370 4481
    # Idaho 704521 155893 13.435589129907036 940854 3970
    # Maine 378300 83769 12.07269192470008 509202 3715
    # Mississippi 774019 171394 13.043500075259645 1039880 6157
    # Texas 16489484 3711669 16.434736082160356 21808336 72065
        