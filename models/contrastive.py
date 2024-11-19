import os
import torch
import torch.nn as nn
from module.loss import Loss
from RecStudio.recstudio.model.module.ctr import MLPModule
from RecStudio.recstudio.model import init
from collections import defaultdict
import torch.distributed as dist
from transformers import AutoModel

class Pooler(nn.Module):

    def __init__(self, pooling):
        super().__init__()
        self.pooling = pooling
        if pooling == 'reduce_add_ot':
            self.reduce_layer = nn.Linear(768, 384)
    
    def forward(self, x, attn_mask):
        last_hidden_state = x.hidden_states[-1]
        if self.pooling == 'cls':
            return last_hidden_state[:, 0]
        
        elif self.pooling == 'sum':
            input_mask_expanded = attn_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            return torch.sum(last_hidden_state * input_mask_expanded, 1)
        
        elif self.pooling == 'mean':
            input_mask_expanded = attn_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask
        
        elif self.pooling == 'max':
            input_mask_expanded = attn_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            last_hidden_state[input_mask_expanded == 0] = -1e9
            return torch.max(last_hidden_state, 1)[0]
        
        elif self.pooling == 'reduce_add_ot':
            ot_emb = self.ot_embedding(
                            logits=x.logits[:, 1:],
                            attention_mask=attn_mask[:, 1:]
                        )
            ret = torch.cat([
                    self.reduce_layer(last_hidden_state[:, 0]),
                    ot_emb.topk(k=384).values
                ], dim=1)
            return ret

    def ot_embedding(self, logits, attention_mask):
        mask = (1 - attention_mask.unsqueeze(-1)) * -1000
        reps, _ = torch.max(logits + mask, dim=1)  # B V
        return reps

    

class Contrastive(nn.Module):
    def __init__(self, config, language_model, text_fields, tokenizer):
        super().__init__()
        self.config_dict = config
        self.language_model: AutoModel = language_model
        self.text_fields = text_fields
        self.tokenizer = tokenizer
            
        self.loss_fn = Loss(config['loss'], config['temperature'], config['weight'], config['temperature_dict'])
        self.pooler = Pooler(config['pooling'])
        self.init_parameters()

        if config['negatives_spread'] and dist.is_initialized():
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()


    def init_parameters(self, init_method='xavier_normal'):
        init_methods = {
            'xavier_normal': init.xavier_normal_initialization,
            'xavier_uniform': init.xavier_uniform_initialization,
            'normal': init.normal_initialization(),
        }
        init_method = init_methods[init_method]
        for name, module in self.named_children():
            if 'language_model' not in name:
                module.apply(init_method)

    def gradient_checkpointing_enable(self, **kwargs):
        self.language_model.gradient_checkpointing_enable(**kwargs)


    def forward(self, return_loss=True, **batch):
        if 'feature_pair' in self.config_dict['loss']:
            # batch: {field_0: BxL, field_1: BxL}
            feature_q_emb = self.pooler(
                                self.language_model(**batch[self.text_fields[0]], return_dict=True, output_hidden_states=True), 
                                batch[self.text_fields[0]]['attention_mask'])
            feature_p_emb = self.pooler(
                                self.language_model(**batch[self.text_fields[1]], return_dict=True, output_hidden_states=True), 
                                batch[self.text_fields[1]]['attention_mask'])
        else:
            feature_q_emb = None
            feature_p_emb = None

        if len(
            {'normal', 'improved', 'guided', 'cosine', 'angle'}.intersection(set(self.config_dict['loss']))
        ) > 0:
            # batch: {'anchor': BxL, 'positive': BxL}
            q_emb = self.pooler(
                        self.language_model(**batch['anchor'], return_dict=True, output_hidden_states=True), 
                        batch['anchor']['attention_mask'])
            p_emb = self.pooler(
                        self.language_model(**batch['positive'], return_dict=True, output_hidden_states=True), 
                        batch['positive']['attention_mask'])

            n_emb = self._get_negative_emb(batch['negative']) if 'negative' in batch else None

        if 'guided' in self.config_dict['loss']:
            guided_q_emb = batch['guided_anchor_emb'] 
            guided_p_emb = batch['guided_positive_emb']
            guided_n_emb = batch['guided_negative_emb'] if 'negative' in batch else None
        else:
            guided_q_emb = guided_p_emb = guided_n_emb = None

        if self.config_dict['negatives_spread'] and dist.is_initialized():
            feature_q_emb = self._dist_gather_tensor(feature_q_emb)
            feature_p_emb = self._dist_gather_tensor(feature_p_emb)
            q_emb = self._dist_gather_tensor(q_emb)
            p_emb = self._dist_gather_tensor(p_emb)
            n_emb = self._dist_gather_tensor(n_emb)
            guided_q_emb = self._dist_gather_tensor(guided_q_emb)
            guided_p_emb = self._dist_gather_tensor(guided_p_emb)
            guided_n_emb = self._dist_gather_tensor(guided_n_emb)

        loss = self.loss_fn(
                    feature_q_emb=feature_q_emb,
                    feature_p_emb=feature_p_emb,
                    q_emb=q_emb,
                    p_emb=p_emb,
                    n_emb=n_emb,
                    guided_q_emb=guided_q_emb,
                    guided_p_emb=guided_p_emb,
                    guided_n_emb=guided_n_emb,
                    domain=batch.get('DOMAIN', None)
                )

        return {
            'loss': loss
        }
    
    
    def _get_negative_emb(self, negative):
        # Cancel padding
        mask = negative['attention_mask'].sum(-1)                                       # bs x negative_len
        negative = {k : v.flatten(end_dim=-2) for k, v in negative.items()}             # bsxnegative_len x seq_len 

        non_pad_negative = defaultdict(list)
        nz = mask.flatten().nonzero()
        non_pad_negative['input_ids'] = negative['input_ids'][nz.flatten()]
        non_pad_negative['attention_mask'] = negative['attention_mask'][nz.flatten()]

        non_pad_n_embs = self.pooler(
                            self.language_model(**non_pad_negative, return_dict=True, output_hidden_states=True),
                            non_pad_negative['attention_mask'])

        # Padding
        n_embs = torch.zeros(mask.flatten().shape[0], non_pad_n_embs.shape[-1], device=mask.device)     # bsxnegative_len x D
        # according to index, scatter src to input, given the dim 
        n_embs = n_embs.scatter_(                                                                       # input: bsxnegative_len x D
                                0,                                                                      # dim
                                nz.tile(non_pad_n_embs.shape[-1]),                                      # index: non-pad-num x D
                                non_pad_n_embs)                                                         # src: non-pad-num x D
        n_embs = n_embs.reshape(*mask.shape, -1)   
        return n_embs
    

    def _dist_gather_tensor(self, t: torch.Tensor):
        if t is None:
            return None
        
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors