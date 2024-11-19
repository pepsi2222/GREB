import math
import torch
import torch.nn as nn
from .baseranker import BaseRanker
from RecStudio.recstudio.model.module.layers import AttentionLayer, MLPModule
from RecStudio.recstudio.model.module import ctr

class DeepFM(BaseRanker):


    def __init__(self, config, dataset, ctr_fields, **kwargs):
        super().__init__(config, dataset, ctr_fields, **kwargs)
        
        d = self.embed_dim * (
                    len([_ for _ in self.item_fields if dataset.field2type[_] != 'text']) + \
                    len([_ for _ in self.behavior_fields if dataset.field2type[_.replace('in_', '')] != 'text']) + \
                    len([_ for _ in self.user_fields if dataset.field2type[_] != 'text']) + \
                    len(self.context_fields)
                )
        if hasattr(self, 'item_text_field') and self.item_text_field is not None:
            d += 2 * self.embed_dim # item text and behavior text


        model_config = self.config_dict['ctr_model']
        self.linear = ctr.LinearLayer(ctr_fields, dataset)
        self.fm = ctr.FMLayer(reduction='sum')
        self.mlp = MLPModule(
                    [d]+model_config['mlp_layer']+[1],
                    model_config['activation'], 
                    model_config['dropout'],
                    batch_norm=model_config['batch_norm'],
                    last_activation=False, 
                    last_bn=False)

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.init_parameters()


    def forward(self, **batch):
        lr_score = self.linear(batch)
        emb = self.get_embedddings(batch)
        emb = emb.reshape(emb.shape[0], -1, self.embed_dim)
        fm_score = self.fm(emb)
        mlp_score = self.mlp(emb.flatten(1)).squeeze(-1)
        score = lr_score + fm_score + mlp_score
        loss = self.loss_fn(score, batch[self.frating])
        return {
            'loss': loss,
            'logits': score
        }