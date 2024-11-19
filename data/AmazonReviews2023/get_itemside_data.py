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

from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from transformers import set_seed
set_seed(42)



if __name__ == '__main__':
    category = sys.argv[1]
    
    if os.path.exists(f'/data1/home/xingmei/GRE/itemside_data/{category}.csv'):
        sys.exit()
    
    with open(f'dataset/filtered_{category}.pkl', 'rb') as f:
        trn_set = pickle.load(f)
        tst_set = pickle.load(f)
        dataset = pickle.load(f)

    dataset.item_feat.to_csv(f'/data1/home/xingmei/GRE/itemside_data/{category}.csv', 
                            sep='\t',
                            header=True,
                            index=False)
    