import math
import torch
import torch.nn as nn
from collections import OrderedDict
from .baseranker import BaseRanker
from RecStudio.recstudio.model.module.ctr import FMLayer, LinearLayer
from module.ctr import MyEmbeddings


class FM(BaseRanker):


    def __init__(self, config, dataset, ctr_fields, item_text_field=None, **kwargs):
        super().__init__(config, dataset, ctr_fields, item_text_field, **kwargs)

        d = self.embed_dim * len([_ for _ in self.item_fields if dataset.field2type[_] != 'text'])
        if item_text_field is not None:
            d += self.embed_dim

        self.fm = FMLayer(reduction='sum')
        self.linear = LinearLayer(ctr_fields, dataset)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.init_parameters()


    def forward(self, **batch):
        embs = self.get_embedddings(batch)
        embs = embs.reshape(embs.shape[0], -1, self.embed_dim)
        fm_score = self.fm(embs).squeeze(-1)
        lr_score = self.linear(batch)

        score = lr_score + fm_score
        
        loss = self.loss_fn(score, batch[self.frating])
        return {
            'loss': loss,
            'logits': score
        }