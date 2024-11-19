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
    neg_ratio = 1 if len(sys.argv) < 3 else int(sys.argv[2])

    # used_features = ['gmap_id', 'name', 'category']
    # # item metadata
    # item_df = get_data_from_json_by_line(
    #                     json_file_path=os.path.join(work_dir, 'data/GoogleLocalData/meta', f'{category}.json'),
    #                     fields=used_features) 
    # item_df = item_df[~pd.isna(item_df['category'])].copy()
    # item_df['category'] = item_df['category'].map(lambda x : '|'.join(x))
    # item_df.to_csv(
    #         os.path.join(work_dir, 'data/GoogleLocalData/meta', f'{category}.csv'),
    #         sep='\t',
    #         header=True,
    #         index=False
    #     )
    
    # # interaction
    # inter_df = get_data_from_json_by_line(
    #                 json_file_path=os.path.join(work_dir, 'data/GoogleLocalData/review', f'{category}.json'),
    #                 fields=['user_id', 'gmap_id', 'rating', 'time']
    #         )
    # inter_df.to_csv(
    #         os.path.join(work_dir, 'data/GoogleLocalData/review', f'{category}.csv'),
    #         sep='\t',
    #         header=True,
    #         index=False
    #     )
    

    # get recstudio dataset
    dataset = process_by_recstudio(
                category,
                os.path.join(work_dir, 'data/GoogleLocalData/config', f'{category}.yaml') 
            )
    
    # trn_set, _, tst_set = negative_sample_and_split(
    #                                 dataset, 
    #                                 val=False,
    #                                 max_behavior_len=dataset.config['max_seq_len'],
    #                                 neg_ratio=neg_ratio)

    trn_set, _, tst_set = no_sample_and_split(
                                    dataset, 
                                    max_behavior_len=dataset.config['max_seq_len'])
    

    pkl_name = f'{category}.pkl' if neg_ratio == 1 else f'{category}_neg{neg_ratio}.pkl'
    with open(os.path.join(work_dir, 'data/GoogleLocalData/rating_dataset', pkl_name), 'wb') as f:
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
# ----- predict next reviewed -----------------------------------
# Alabama 11311688 667102 18.95645943199091 6322946 74581                                                                                                                                                                                            
# Alaska 1220524 86528 16.105538091715978 696790 12687                                                                                                                                                                                                    
# Arizona 23548488 1383534 19.02053437067683 13157778 108053                                                                                                                                                                                              
# Arkansas 6338256 388938 18.29631457970165 3558066 47071                                                                                                                                                                                                
# California 95644668 5090086 20.790383502361255 52912420 513092                                                                                                                                                                                        
# Colorado 19421692 1193804 18.268744282981125 10904650 106234                                                                
# Connecticut 6008670 378506 17.874702118328376 3382841 48928                                                                 
# Delaware 2060456 143774 16.33121426683545 1174002 14616                                                                       
# District_of_Columbia 1489756 149640 11.955600106923283 894518 11003                                                         
# Florida 78160332 4685986 18.67959144564239 43766152 376168                                                                   
# Georgia 29957166 1781890 18.812017576842567 16760473 165379                                                                  
# Hawaii 3538478 286078 14.368927355476478 2055317 21419                                                                       
# Idaho 4679140 298476 17.67677133169836 2638046 32979    
# Illinois 29078146 1699156 19.113288008870285 16238229 178183                                                                   
# Indiana 16626866 957484 19.365163282101843 9270917 99891 
# Iowa 5975672 374454 17.95836070652203 3362290 47438                                                        
# Kansas 6816352 417096 18.342405585284922 3825272 46029     
# Kentucky 9386514 573352 18.37129372532057 5266609 62853                                        
# Michigan 28415800 1502350 20.914234366159683 15710250 158102  
# Louisiana 8994850 585190 17.370819733761685 5082615 62956
# Maine 2599444 177498 16.64491994275992 1477220 24661                   
# Maryland 12488060 800214 17.605900421637212 7044244 77666
# Massachusetts 12588930 796240 17.81047171707023 7090705 91886  
# Michigan 28415800 1502350 20.914234366159683 15710250 158102                  
# Minnesota 12353158 701740 19.60361102402599 6878319 80581    
# Mississippi 4457086 297734 16.970026936795932 2526277 36930                                                                 
# Missouri 17214150 1012184 19.00693747381899 9619259 98930                                                                    
# Montana 2197214 149346 16.712238694039346 1247953 21528                                                                     
# Nebraska 4033978 243548 18.56337970338496 2260537 29875                                                                     
# Nevada 9602294 684394 16.03035970508216 5485541 47997                                                                       
# New_Hampshire 2976890 207978 16.31348508015271 1696423 24621                                                              
# New_Jersey 18392572 1159566 17.86159994342711 10355852 126543                                                                 
# New_Mexico 5733000 361498 17.859008901847314 3227998 34506                                                                  
# New_York 41377236 2524724 18.388815569543443 23213342 270694                                                             
# North_Carolina 28323394 1658980 19.072776043110828 15820677 165385                                                          
# North_Dakota 1290944 88818 16.534711432367313 734290 11931   
# Ohio 31187026 1668250 20.69445586692642 17261763 172874                                                             
# Oklahoma 10953050 624970 19.52572123461926 6101495 67696                                                                   
# Oregon 13704790 789220 19.36498061377056 7641615 92997                                                                  
# Pennsylvania 28041838 1641982 19.07804226842925 15662901 189818                                                             
# Rhode_Island 2015608 135824 16.839851572623395 1143628 15848                                                               
# South_Carolina 14525556 913274 17.904926670418735 8176052 84535                                                               
# South_Dakota 1587602 117762 15.481445627621813 911563 14164                                                                
# Tennessee 19659886 1222236 18.085179948880576 11052179 110820                                                                
# Texas 88204754 4844612 20.206773628104788 48946989 444922                                                                  
# Utah 10996220 677246 18.236670279337197 6175356 58533 
# Vermont 816172 67714 14.053223853265203 475800 11241                                                                         
# Virginia 19039918 1189680 18.004234752202272 10709639 119017                                                                 
# Washington 22093806 1250854 19.66297745380356 12297757 120626                                                               
# West_Virginia 2478604 171902 16.418703680003723 1411204 23354                                                                 
# Wisconsin 13253182 769770 19.217067435727554 7396361 91513                                                                  
# Wyoming 1069230 90432 13.823580148619957 625047 12006     
        