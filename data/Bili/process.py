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



def process_by_recstudio(dataset, data_config_path):
    data_conf = parser_yaml(data_config_path)
    dataset = SeqDataset(name=dataset, config=data_conf)
    if dataset.ftime is not None:
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
    category = sys.argv[1]  # like Source_datasets/Bili_2M
    neg_ratio = 1 if len(sys.argv) < 3 else int(sys.argv[2])

    if not os.path.exists(os.path.join(work_dir, 'data/Bili/dataset')):
        os.makedirs(os.path.join(work_dir, 'data/Bili/dataset'))

    # item metadata
    item_df = pd.read_csv(
                os.path.join(work_dir, 'data/Bili', category, f'{os.path.basename(category)}_item.csv'),
                sep=',',
                header=None,
                names=['iid', 'zh_title', 'en_title'])
    item_df = item_df[['iid', 'en_title']].copy()
    print(item_df)

    item_df.to_csv(
            os.path.join(work_dir, 'data/Bili', category, f'new_item.csv'),
            sep=',',
            header=True,
            index=False
        )
    
    # interaction
    # inter_df = pd.read_csv(
    #             os.path.join(work_dir, 'data/Bili', category, f'{os.path.basename(category)}_pair.csv'),
    #             sep=',',
    #             header=None,
    #             name=['iid', 'uid', 'timestamp']
    #         )
    # inter_df.to_csv(
    #         os.path.join(work_dir, 'data/Bili/inter', f'{os.path.basename(category)}.csv'),
    #         sep='\t',
    #         header=True,
    #         index=False
    #     )
    

    # get recstudio dataset
    dataset = process_by_recstudio(
                os.path.basename(category), 
                os.path.join(work_dir, 'data/Bili/config', f'{os.path.basename(category)}.yaml'))
    
    trn_set, _, tst_set = negative_sample_and_split(
                                    dataset, 
                                    val=False,
                                    max_behavior_len=dataset.config['max_seq_len'],
                                    neg_ratio=neg_ratio)

    pkl_name = f'{os.path.basename(category)}.pkl' if neg_ratio == 1 else f'{os.path.basename(category)}_neg{neg_ratio}.pkl'
    with open(os.path.join(work_dir, 'data/Bili/dataset', pkl_name), 'wb') as f:
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
    # Bili_2M:          40994314 4000000 12.2485785         24497157 144147

    # Bili_Cartoon:     309686  60600   7.1103300330033     215443  4725
    # Bili_Dance:       123924  21430   7.782734484367709   83392   2308 
    # Bili_Food:        53284   13098   6.06810200030539    39740   1580 
    # Bili_Movie:       165052  33050   6.994009077155824   115576  3510 
    # Bili_Music:       517698  101328  7.1091307437233535  360177  6039                                                               

    # DY:               198076  40796   6.855279929404843   139834  8300
    # KU:               28902   4068    9.104719764011799   18519   5371 
    # QB:               196440  35444   7.54226385283828    133664  6122
    # TN:               164308  40422   6.064816189203899   122576  3335
        