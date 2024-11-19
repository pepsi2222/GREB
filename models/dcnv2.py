import math
import torch
import torch.nn as nn
from .baseranker import BaseRanker
from RecStudio.recstudio.model.module.layers import MLPModule
from RecStudio.recstudio.model.module import ctr

class DCNv2(BaseRanker):


    def __init__(self, config, dataset, ctr_fields, **kwargs):
        super().__init__(config, dataset, ctr_fields, **kwargs)

        d = self.embed_dim * (
                    len([_ for _ in self.item_fields if dataset.field2type[_] != 'text']) + \
                    len([_ for _ in self.user_fields if dataset.field2type[_] != 'text']) + \
                    len(self.context_fields)
                )
        if hasattr(self, 'item_text_field') and self.item_text_field is not None and \
            hasattr(self, 'user_text_field') and self.user_text_field is not None:
            d += 2 * self.embed_dim

        
        model_config = self.config_dict['ctr_model']
        if model_config['low_rank'] is None:
            self.cross_net = ctr.CrossNetworkV2(d, model_config['num_layers'])
        else:
            self.cross_net = ctr.CrossNetworkMix(d, model_config['num_layers'], 
                                                model_config['low_rank'], model_config['num_experts'],
                                                model_config['cross_activation'])
            
        if model_config['combination'].lower() == 'parallel':
            self.mlp = MLPModule(
                        [d] + model_config['mlp_layer'],
                        model_config['activation'],
                        model_config['dropout'],
                        batch_norm=model_config['batch_norm'])
            self.fc = nn.Linear(d + model_config['mlp_layer'][-1], 1)
        elif model_config['combination'].lower() == 'stacked':
            self.mlp = MLPModule(
                        [d] + model_config['mlp_layer'] + [1],
                        model_config['activation'],
                        model_config['dropout'],
                        batch_norm=model_config['batch_norm'],
                        last_activation=False,
                        last_bn=False)
        else:
            raise ValueError(f'Expect combination to be `parallel`|`stacked`, but got {model_config["combination"]}.')

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.init_parameters()

    def get_user_embeddings(self, batch):
        # todo: add user text emb
        user_feat = self._get_user_feat(batch)
        user_embs = self.embedding(user_feat)
        return user_embs
    
    def build_text_embedding(self, batch_size=128, maxlen=512, saved_dir=None, device='cuda:0', grad=False):
        pass
        # todo: add user text embedding


    def get_embedddings(self, batch):
        user_emb = self.get_user_embeddings(batch)
        item_emb = self.get_item_embeddings(batch)
        # pooled_behavior_emb = self.get_pooled_behavior_embeddings(batch)
        embs = [user_emb, item_emb]
        if len(self.context_fields) > 0:
            context_emb = self.get_context_embeddings(batch)
            embs.append(context_emb)
        emb = torch.cat(embs, -1)
        return emb
    

    def forward(self, **batch):
        emb = self.get_embedddings(batch)
        cross_out = self.cross_net(emb)
        if self.config_dict['ctr_model']['combination'].lower() == 'parallel':
            deep_out = self.mlp(emb)
            score = self.fc(torch.cat([deep_out, cross_out], -1)).squeeze(-1)
        else:
            deep_out = self.mlp(cross_out)
            score = deep_out.squeeze(-1)
        loss = self.loss_fn(score, batch[self.frating])
        return {
            'loss': loss,
            'logits': score
        }