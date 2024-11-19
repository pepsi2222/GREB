import torch
import torch.nn as nn
from recstudio.model.module.ctr import Embeddings
from recstudio.model.module.layers import SeqPoolingLayer

class MyEmbeddings(Embeddings):
    def __init__(self, fields, embed_dim, data, reduction='mean',
                 share_dense_embedding=False, dense_emb_bias=False, dense_emb_norm=True,
                 with_dense_kernel=False):
        super().__init__(fields, embed_dim, data, reduction, 
                         share_dense_embedding, dense_emb_bias, dense_emb_norm,
                         with_dense_kernel)
        self.seq_pooling_layer = SeqPoolingLayer(reduction, keepdim=False)

    def forward(self, batch):
        embs = []
        dense_value_list = []
        for f, t in self.field2types.items():
            if f not in batch and 'in_'+f not in batch:
                continue
            elif 'in_'+f in batch:
                d = batch.pop('in_'+f)
            else:
                d = batch.pop(f)
            
            if t == 'token' or (t == 'float' and not self.with_dense_kernel):
                # shape: [B,] or [B,N]
                e = self.embeddings[f](d)
                embs.append(e)
            elif t == 'float' and self.with_dense_kernel:
                dense_value_list.append(d)
            else:
                # shape: [B, L] or [B,N,L]
                length = (d > 0).float().sum(dim=-1, keepdim=False)
                length = torch.where(length > 0, length, 1)
                seq_emb = self.embeddings[f](d)
                e = self.seq_pooling_layer(seq_emb, length)
                embs.append(e)

        if 'dense_kernel' in self.embeddings:
            dense_emb = self.embeddings['dense_kernel'](
                            torch.vstack(dense_value_list).transpose(0, 1)
                        ).expand(-1, self.embed_dim)
            embs.append(dense_emb)
            
        # [B,num_features,D] or [B,N,num_features,D]
        # emb = torch.stack(embs, dim=-2)
        emb = torch.cat(embs, dim=-1)   # B x (F x D)
        return emb

