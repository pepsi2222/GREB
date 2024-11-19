from collections import OrderedDict

import torch
from recstudio.data import TripletDataset, SeqDataset

from .. import basemodel, loss_func, scorer
from ..module import LambdaLayer, MLPModule, ctr

class DSSMQueryEncoder(torch.nn.Module):
    def __init__(self, fiid, item_encoder, embed_dim, mlp_layer, dropout, activation, batch_norm):
        super().__init__()
        self.fiid = fiid
        self.item_encoder = item_encoder
        self.seq_pooling_layer = ctr.SeqPoolingLayer('mean', keepdim=False)
        self.mlp = MLPModule(
                    [embed_dim] + mlp_layer,
                    dropout=dropout, 
                    activation_func=activation,
                    batch_norm=batch_norm)

    def forward(self, batch):
        user_hist = batch['in_'+self.fiid]
        seq_embs = self.item_encoder(user_hist)
        length = (user_hist > 0).float().sum(dim=-1, keepdim=False)
        length = torch.where(length > 0, length, 1)
        pooled_seq_embs = self.seq_pooling_layer(seq_embs, length)
        output = self.mlp(pooled_seq_embs)
        return output
    
class DSSMItemEncoder(torch.nn.Module):
    def __init__(self, train_data, embed_dim, mlp_layer, dropout, activation, batch_norm):
        super().__init__()
        self.embedding = torch.nn.Embedding(
                                train_data.num_items, 
                                embed_dim, 
                                padding_idx=0)
        
        self.mlp = MLPModule(
                    [embed_dim] + mlp_layer,
                    dropout=dropout, 
                    activation_func=activation,
                    batch_norm=batch_norm)

    def forward(self, batch):
        item_emb = self.embedding(batch)
        output = self.mlp(item_emb)
        return output


class DSSM(basemodel.BaseRetriever):

    # def _set_data_field(self, data):
    #     data.use_field = data.field

    def _get_dataset_class():
        # return TripletDataset
        return SeqDataset

    def _get_query_encoder(self, train_data):
        model_config = self.config['model']
        return DSSMQueryEncoder(
            fiid=self.fiid,
            item_encoder=self.item_encoder.embedding,
            embed_dim=self.embed_dim,
            mlp_layer=model_config['mlp_layer'],
            activation=model_config['activation'],
            dropout=model_config['dropout'],
            batch_norm=model_config['batch_norm']
        )
        # if len(self.query_fields) == 1 and list(self.query_fields)[0] == self.fuid:
        #     embedding = torch.nn.Embedding(train_data.num_users, self.embed_dim, padding_idx=0)
        #     mlp_input_dim = self.embed_dim
        # else:
        #     embedding = ctr.Embeddings(
        #         fields=self.query_fields,
        #         data=train_data,
        #         embed_dim=self.embed_dim)
        #     mlp_input_dim = embedding.num_features * self.embed_dim
        # model_config = self.config['model']
        # mlp = MLPModule(
        #     [mlp_input_dim] + model_config['mlp_layer'],
        #     dropout=model_config['dropout'], activation_func=model_config['activation'],
        #     batch_norm=model_config['batch_norm'])
        # return torch.nn.Sequential(
        #     OrderedDict(
        #         {'embedding': embedding,
        #          'flatten': LambdaLayer(lambda x: x.view(x.size(0), -1)),
        #          'MLP': mlp}))

    def _get_item_encoder(self, train_data):
        model_config = self.config['model']
        return DSSMItemEncoder(
                    train_data=train_data,
                    embed_dim=self.embed_dim, 
                    mlp_layer=model_config['mlp_layer'],
                    activation=model_config['activation'],
                    dropout=model_config['dropout'],
                    batch_norm=model_config['batch_norm']
                )
        # if len(self.item_fields) == 1 and list(self.item_fields)[0] == self.fiid:
        #     embedding = torch.nn.Embedding(train_data.num_items, self.embed_dim, padding_idx=0)
        #     mlp_input_dim = self.embed_dim
        #     flatten_layer = LambdaLayer(lambda x: x)
        # else:
        #     embedding = ctr.Embeddings(
        #         fields=self.item_fields,
        #         data=train_data,
        #         embed_dim=self.embed_dim,
        #     )
        #     mlp_input_dim = embedding.num_features * self.embed_dim
        #     flatten_layer = LambdaLayer(lambda x: x.view(*x.shape[: -2], -1))

        # model_config = self.config['model']
        # mlp = MLPModule(
        #     [mlp_input_dim] + model_config['mlp_layer'],
        #     activation_func = model_config['activation'],
        #     dropout = model_config['dropout'],
        #     batch_norm = model_config['batch_norm'])
        # return torch.nn.Sequential(
        #     OrderedDict(
        #         {'embedding': embedding,
        #          'flatten': flatten_layer,
        #          'MLP': mlp}))

    def _get_score_func(self):
        return scorer.InnerProductScorer()

    def _get_loss_func(self):
        return loss_func.BinaryCrossEntropyLoss()
