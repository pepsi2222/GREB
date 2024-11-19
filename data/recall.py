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
from datetime import datetime
import json
import csv
csv.field_size_limit(sys.maxsize)

from transformers import set_seed
set_seed(42)

with open(os.path.join(work_dir, 'data/Goodreads/filtered_Goodreads_rating.pkl'), 'rb') as f:
        trn_set = pickle.load(f)
        tst_set = pickle.load(f)
        dataset = pickle.load(f)

dataset.config['low_rating_thres'] = 1.0
dataset._filter(
            dataset.config['min_user_inter'],
            dataset.config['min_item_inter'])
dataset.config['low_rating_thres'] = 3.0


with open(os.path.join(work_dir, 'data/Goodreads/filtered_Goodreads_rating.pkl'), 'wb') as f:
        pickle.dump(trn_set, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(tst_set, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)

# -------------------------------------------------------------------------------------------
with open(os.path.join(work_dir, 'data/Yelp/filtered_Yelp_rating.pkl'), 'rb') as f:
        trn_set = pickle.load(f)
        tst_set = pickle.load(f)
        dataset = pickle.load(f)

dataset.config['low_rating_thres'] = 1.0
dataset._filter(
            dataset.config['min_user_inter'],
            dataset.config['min_item_inter'])
dataset.config['low_rating_thres'] = 3.0


with open(os.path.join(work_dir, 'data/Yelp/filtered_Yelp_rating.pkl'), 'wb') as f:
        pickle.dump(trn_set, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(tst_set, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)


# -------------------------------------------------------------------------------------------
with open(os.path.join(work_dir, 'data/GoogleLocalData/Food/rating_dataset/filtered_Maine.pkl'), 'rb') as f:
        trn_set = pickle.load(f)
        tst_set = pickle.load(f)
        dataset = pickle.load(f)

dataset.config['low_rating_thres'] = 1.0
dataset._filter(
            dataset.config['min_user_inter'],
            dataset.config['min_item_inter'])
dataset.config['low_rating_thres'] = 3.0


with open(os.path.join(work_dir, 'data/GoogleLocalData/Food/rating_dataset/filtered_Maine.pkl'), 'wb') as f:
        pickle.dump(trn_set, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(tst_set, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)