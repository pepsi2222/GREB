import os
import torch
from recstudio.model.seq import SASRec
from recstudio.model.module import MLPModule
from recstudio.utils import get_gpus
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoConfig
from typing import cast
from recstudio.model import init
from recstudio.model.basemodel.recommender import Recommender



TEXT_EMBEDDING = 'text_embedding.pt'

class TEItemEncoder(torch.nn.Module):
    def __init__(
            self, 
            train_data, 
            embed_dim, 
            model_name_or_path, 
            activation, 
            dropout, 
            batch_norm,
            mlp_hidden_layers = [], 
            text_field = 'text',
            pooler = 'cls',
            freeze_text_embedding = True,
            normalize_embeddings = True,
            save_dir = None,
            lm_projection_layer_norm = False 
        ):
        super().__init__()
        self.item_sentences = train_data.item_feat.get_col(text_field).tolist() 
        self.model_name_or_path = model_name_or_path
        self.text_field = text_field 
        self.save_dir = save_dir
        self.freeze_text_embedding = freeze_text_embedding

        text_encoder_config = AutoConfig.from_pretrained(model_name_or_path)
        
        self.lm_projection = torch.nn.Sequential(
                                MLPModule(
                                    mlp_layers=[text_encoder_config.hidden_size] + mlp_hidden_layers + [embed_dim],
                                    activation_func=activation,
                                    dropout=dropout,
                                    batch_norm=batch_norm,
                                )
                            )
        if lm_projection_layer_norm:
            self.lm_projection.append(torch.nn.LayerNorm(embed_dim))

        self.pooler = pooler
        self.normalize_embeddings = normalize_embeddings
        self.text_embeddings = torch.nn.Embedding(
            train_data.num_items, text_encoder_config.hidden_size)

        train_data.item_feat.del_fields(keep_fields=[train_data.fiid])

    
    def get_text_embedding(self, device, batch_size=128, max_length=256):
        # self.text_encoder = AutoModel.from_pretrained(model_name_or_path)
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        save_path = os.path.join(self.save_dir, TEXT_EMBEDDING) if self.save_dir else None
        if save_path is not None:
            if os.path.exists(save_path):
                self.text_embeddings = torch.load(save_path, map_location=device)
            else:
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
        
                text_embeddings = self._get_text_embedding(
                                    sentences=self.item_sentences,
                                    batch_size=batch_size,
                                    max_length=max_length,
                                    device=device 
                                )
                self.text_embeddings = torch.nn.Embedding.from_pretrained(
                                    text_embeddings,
                                    freeze=True,
                                    padding_idx=0).to(device)
                if save_path is not None:
                    if not os.path.exists(self.save_dir):
                        os.makedirs(self.save_dir)
                    torch.save(self.text_embeddings, save_path)

            for p in self.text_embeddings.parameters():
                if not self.freeze_text_embedding:
                    p.requires_grad = True
                else:
                    p.requires_grad = False
                
    def _get_text_embedding(self, sentences, batch_size, max_length, device):
        # reload text_encoder and tokenizer since they are re-inited 
        text_encoder = AutoModel.from_pretrained(self.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        text_encoder = text_encoder.to(device)
        text_encoder.eval()
        with torch.no_grad():
            all_embeddings = []
            for start_index in tqdm(range(0, len(sentences), batch_size), desc="Inference Embeddings",
                                disable=len(sentences) < 256):
                sentences_batch = sentences[start_index:start_index + batch_size]
                sentences_batch = [tokenizer.truncate_sequences(
                                        _,
                                        truncation_strategy='longest_first',
                                        num_tokens_to_remove=len(_) - max_length
                                    )[0]
                                    for _ in sentences_batch]
                inputs = tokenizer(
                            sentences_batch,
                            padding=True,
                            truncation=True,
                            return_tensors='pt',
                            max_length=max_length,
                        ).to(device)
                if not text_encoder.config.is_encoder_decoder:
                    last_hidden_state = text_encoder(**inputs, return_dict=True).last_hidden_state.to('cpu')
                else:
                    last_hidden_state = text_encoder(
                                            input_ids=inputs['input_ids'], 
                                            decoder_input_ids=inputs['input_ids'], 
                                            return_dict=True
                                        ).last_hidden_state.to('cpu')
                embeddings = self._pooling(last_hidden_state, inputs['attention_mask'])
                if self.normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, dim=-1).to('cpu')
                embeddings = cast(torch.Tensor, embeddings)
                all_embeddings.append(embeddings)
            all_embeddings = torch.cat(all_embeddings, dim=0)
        del text_encoder
        return all_embeddings

        
    def _pooling(self, last_hidden_state, attention_mask = None):
        if self.pooler == 'cls':
            return last_hidden_state[:, 0].clone()
        elif self.pooler == 'mean':
            s = torch.sum(last_hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(dim=1, keepdim=True).float()
            return s / d


    def forward(self, batch):
        text_embs = self.text_embeddings(batch)
        text_embs = self.lm_projection(text_embs)
        return text_embs


class TE_SASRec(SASRec):

    def _set_data_field(self, data):
        data.use_field = set([data.fuid, data.fiid, data.frating, self.config['text_encoder']['text_field']])
        if hasattr(self, 'logger'):
            self.logger.info(f"The default fields to be used is set as [user_id, item_id, rating, {self.config['text_encoder']['text_field']}]. "
                             "If more fields are needed, please use `self._set_data_field()` to reset.")
            
    def _get_item_feat(self, data):
        if isinstance(data, dict):  # batch
            return data[self.fiid]
        else:  # neg_item_idx
            return data
            

    def _get_item_encoder(self, train_data):
        return TEItemEncoder(
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
            lm_projection_layer_norm=self.config['text_encoder']['lm_projection_layer_norm']
        )

    def _get_item_vector(self, batch_size=512):
        text_embs = self.item_encoder.text_embeddings.weight[1:]
        text_reprs = []
        for i in tqdm(range(0, len(text_embs), batch_size)):
            embs = text_embs[i: i + batch_size]
            text_reprs.append(self.item_encoder.lm_projection(embs))
        text_reprs = torch.vstack(text_reprs)
        return text_reprs


    def fit_loop(self, val_dataloader=None):
        self.item_encoder.get_text_embedding(self._parameter_device)
        super().fit_loop(val_dataloader)
