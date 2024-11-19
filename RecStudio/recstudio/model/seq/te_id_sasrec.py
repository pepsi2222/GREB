import os
import torch
from .te_sasrec import TE_SASRec, TEItemEncoder
from tqdm import tqdm

TEXT_EMBEDDING = 'text_embedding.pt'

class TEItemEncoderWithID(TEItemEncoder):
    def __init__(self, fusion_text_id, **kwargs):
        self.fusion_text_id = fusion_text_id
        if fusion_text_id == 'add':
            embed_dim = kwargs['embed_dim']
        elif fusion_text_id == 'concat':
            assert kwargs['embed_dim'] % 2 == 0, f'Expect embed_dim be divided by 2, but got {kwargs["embed_dim"]}.'
            embed_dim = kwargs['embed_dim'] // 2
        else:
            raise ValueError(f'Expect fusion_text_id be `add` or `concat`, but got {fusion_text_id}.')

        kwargs['embed_dim'] = embed_dim
        super().__init__(**kwargs)
        self.id_embeddings = torch.nn.Embedding(
                                kwargs['train_data'].num_items, 
                                kwargs['embed_dim'], 
                                padding_idx=0)

    def forward(self, batch):
        text_embs = super().forward(batch)
        id_embs = self.id_embeddings(batch)
        if self.fusion_text_id == 'add':
            embs = text_embs + id_embs
        else:
            embs = torch.cat([text_embs, id_embs], dim=-1)
        return embs


class TE_ID_SASRec(TE_SASRec):

    def _get_item_encoder(self, train_data):
        return TEItemEncoderWithID(
            train_data=train_data,
            embed_dim=self.embed_dim,
            model_name_or_path=self.config['text_encoder']['model_name_or_path'],
            activation=self.config['text_encoder']['activation'],
            dropout=self.config['text_encoder']['dropout'],
            batch_norm=self.config['text_encoder']['batch_norm'],
            mlp_hidden_layers=self.config['text_encoder']['mlp_hidden_layers'],
            text_field=self.config['text_encoder']['text_field'],
            pooler=self.config['text_encoder']['pooler'],
            freeze_text_embedding=self.config['text_encoder']['freeze_text_embedding'],
            normalize_embeddings=self.config['text_encoder']['normalize_embeddings'],
            save_dir=os.path.join(self.config['eval']['save_path'], self.__class__.__name__, train_data.name),
            fusion_text_id=self.config['text_encoder']['fusion_text_id'],
            lm_projection_layer_norm=self.config['text_encoder']['lm_projection_layer_norm']
        )

    def _get_item_vector(self, batch_size=512):
        text_embs = self.item_encoder.text_embeddings.weight[1:]
        id_embs = self.item_encoder.id_embeddings.weight[1:]
        
        reprs = []
        for i in tqdm(range(0, len(text_embs), batch_size)):
            cur_text_embs = text_embs[i: i + batch_size]
            if self.config['text_encoder']['fusion_text_id'] == 'add':
                cur_reprs = self.item_encoder.lm_projection(cur_text_embs) + id_embs[i : i + batch_size]
            else:

                cur_reprs = torch.cat(
                    [self.item_encoder.lm_projection(cur_text_embs), id_embs[i : i + batch_size]], 
                    dim=-1
                )
            reprs.append(cur_reprs)
            
        reprs = torch.vstack(reprs)

        return reprs

    # def _get_item_vector(self, batch_size=512):
    #     text_embs = self.item_encoder.text_embeddings.weight[1:]
    #     text_reprs = []
    #     for i in tqdm(range(0, len(text_embs), batch_size)):
    #         embs = text_embs[i: i + batch_size]
    #         text_reprs.append(self.item_encoder.lm_projection(embs))
    #     text_reprs = torch.vstack(text_reprs)

    #     id_reprs = self.item_encoder.id_embeddings.weight[1:]
    #     if self.config['text_encoder']['fusion_text_id'] == 'add':
    #         reprs = text_reprs + id_reprs
    #     else:
    #         reprs = torch.cat([text_reprs, id_reprs], dim=-1)
    #     return reprs