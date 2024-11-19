import numpy as np
from scipy.sparse import lil_matrix, save_npz, load_npz
from multiprocessing import Pool, Manager, Value
from tqdm import tqdm

import os
import sys
sys.path.append('../..')
sys.path.append('../../RecStudio')
import pickle
import numpy as np
from data.MIND.process import MINDSeqDataset
from dataset import CTRDataset
from typing import Dict
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from sentence_transformers.util import cos_sim
from collections import defaultdict
import math

BATCH_SIZE = 128
MAX_LEN = 20000
CO_OCCURENCE_THRESHOLD = 0.3
COS_SIM_THRESHOLD = 0.65
MODEL_ID = '../../mxbai-embed-large-v1'

category = os.path.basename(sys.argv[1])

# with open(f'dataset/filtered_{category}.pkl', 'rb') as f:
#     trn_data = pickle.load(f)
#     val_data = pickle.load(f)
#     dataset = pickle.load(f)

# # ----------------- Get co-occurence Matrix ----------------
# bh_data = []
# for uid, hist in tqdm(dataset.inter_feat.groupby(dataset.fuid)):
#     pos_list = hist[dataset.fiid].tolist()
#     bh_data.append(pos_list)

# def process_chunk(chunk, num_items):
#     local_co_occurrence = lil_matrix((num_items, num_items), dtype=float)
#     local_item_cnt = lil_matrix((num_items, 1), dtype=float)

#     for u_bh in tqdm(chunk):
#         u_bh = np.array(u_bh)

#         for i in range(len(u_bh)):
#             iid_i = u_bh[i]
#             local_item_cnt[iid_i, 0] += 1

#             for j in range(i + 1, len(u_bh)):
#                 iid_j = u_bh[j]
#                 local_co_occurrence[iid_i, iid_j] += 1
#                 local_co_occurrence[iid_j, iid_i] += 1
    
#     return local_co_occurrence, local_item_cnt


# num_items = dataset.num_items
# co_occurrence, item_cnt = process_chunk(bh_data, num_items)
# co_occurrence = co_occurrence.tocoo()
# for k in tqdm(range(len(co_occurrence.data))):
#     item_i = co_occurrence.row[k]
#     item_j = co_occurrence.col[k]
#     co_occurrence.data[k] /= math.sqrt(item_cnt[item_i, 0] * item_cnt[item_j, 0])

# if not os.path.exists('co_occurrence/'):
#     os.makedirs('co_occurrence')
# save_npz(f"co_occurrence/{category}.npz", co_occurrence.tocsr())


# # ----------------------Get item text embedding---------------------------------------
# tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# model = AutoModel.from_pretrained(MODEL_ID).cuda()

# dataset.dataframe2tensors()
# trn_dataset = CTRDataset(dataset, trn_data, tokenizer=tokenizer, max_seq_len=MAX_LEN)
# trn_dataset.get_tokenized_text()
# item_text_feat = trn_dataset.dataset.item_feat.data['text']


# text_embeddings = []
# model.eval()
# with torch.no_grad():
#     for i in tqdm(range(0, len(item_text_feat), BATCH_SIZE)):

#         text = [tokenizer.truncate_sequences(
#                                 _,
#                                 truncation_strategy='longest_first',
#                                 num_tokens_to_remove=len(_) - MAX_LEN
#                             )[0]
#                 for _ in item_text_feat[i : i + BATCH_SIZE]]
        
#         text = dict(
#                     tokenizer.pad(
#                         {'input_ids': text},
#                         padding=True,
#                         return_tensors='pt'
#                     )
#                 )
#         text = {k: v.cuda() for k, v in text.items()}
#         text_emb = model(**text, return_dict=True)
#         text_emb = text_emb.last_hidden_state[:, 0, :].to('cpu')
#         text_embeddings.append(text_emb)

#     text_embeddings = torch.cat(text_embeddings, dim=0)     # N x D
#     text_embedding = torch.nn.Embedding.from_pretrained(
#                         text_embeddings,
#                         freeze=True,
#                         padding_idx=0)
#     if not os.path.exists('guide_model_text_embeddings/'):
#         os.makedirs('guide_model_text_embeddings')
#     torch.save(text_embedding, f'guide_model_text_embeddings/{category}.pt')

# ----------------------------Get cosine similarity of co-occurence matrix -------------------------

if not 'guide_model_text_embedding' in locals().keys():
    guide_model_text_embedding = torch.load(f'guide_model_text_embeddings/{category}.pt').cuda()
if not 'co_occurrence' in locals().keys():
    co_occurrence = load_npz(f'co_occurrence/{category}.npz')
    co_occurrence = co_occurrence.tocoo()

sim = []
rows = torch.tensor(co_occurrence.row, dtype=torch.int32)
cols = torch.tensor(co_occurrence.col, dtype=torch.int32)
for start in tqdm(range(0, len(rows), BATCH_SIZE)):
    end = min(start + BATCH_SIZE, len(rows))
    batch_item_i = rows[start:end].cuda()
    batch_item_j = cols[start:end].cuda()
    
    batch_item_i_text_emb = guide_model_text_embedding(batch_item_i)
    batch_item_j_text_emb = guide_model_text_embedding(batch_item_j)
    batch_item_i_text_emb = torch.nn.functional.normalize(batch_item_i_text_emb, p=2, dim=1)
    batch_item_j_text_emb = torch.nn.functional.normalize(batch_item_j_text_emb, p=2, dim=1)
    sim.append((batch_item_i_text_emb * batch_item_j_text_emb).sum(-1))
sim = torch.concat(sim, dim=-1).cpu().tolist()


if not os.path.exists('cos_sim_of_co_occurence/'):
    os.makedirs('cos_sim_of_co_occurence')
with open(f'cos_sim_of_co_occurence/{category}.pkl', 'wb') as f:
    pickle.dump(sim, f)

# ----------------------------------------Filter pairs-----------------------------------------------
# in pair.ipynb