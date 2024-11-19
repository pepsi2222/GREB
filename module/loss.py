import torch
import torch.nn as nn
import torch.nn.functional as F
from RecStudio.recstudio.model.scorer import CosineScorer, EuclideanScorer, InnerProductScorer

class Loss(nn.Module):
    def __init__(self, loss, temperature, weight=None, temperature_dict=None):
        super().__init__()
        self.loss = loss
        self.temperature = temperature if temperature is not None else [0.1] * len(self.loss)
        self.weight = weight if weight is not None else [1 / len(self.loss)] * len(self.loss)
        self.loss_module = dict()
        if 'feature_pair' in self.loss:
            idx = self.loss.index('feature_pair')
            self.loss_module['feature_pair'] = InBatchNegativeInfoNCELoss(temperature[idx])
        if 'normal' in self.loss:
            idx = self.loss.index('normal')
            self.loss_module['normal'] = InBatchNegativeInfoNCELoss(temperature[idx])
        if 'improved' in self.loss:
            idx = self.loss.index('improved')
            self.loss_module['improved'] = ImprovedInBatchNegativeInfoNCELoss(temperature[idx])
        if 'guided' in self.loss:
            idx = self.loss.index('guided')
            self.loss_module['guided'] = GuidedImprovedInBatchNegativeInfoNCELoss(temperature[idx])
        if 'cosine' in self.loss:
            idx = self.loss.index('cosine')
            self.loss_module['cosine'] = CosineLoss(temperature[idx])
        if 'angle' in self.loss:
            idx = self.loss.index('angle')
            self.loss_module['angle'] = AngleLoss(temperature[idx])
        self.temperature_dict = temperature_dict

    def _set_temperature(self, temperature):
        assert len(self.loss) == 1
        self.loss_module[self.loss[0]].temperature = temperature



    def forward(self, feature_q_emb=None, feature_p_emb=None,
                q_emb=None, p_emb=None, n_emb=None, 
                guided_q_emb=None, guided_p_emb=None, guided_n_emb=None, domain=None):
        if self.temperature_dict is not None:
            domain_temperature = self.temperature_dict.get(domain, self.temperature[0])
            self._set_temperature(domain_temperature)

        loss = 0
        if 'feature_pair' in self.loss:
            idx = self.loss.index('feature_pair')
            loss += self.weight[idx] * self.loss_module['feature_pair'](feature_q_emb, feature_p_emb)
        if 'normal' in self.loss:
            idx = self.loss.index('norml')
            loss += self.weight[idx] * self.loss_module['normal'](q_emb, p_emb, n_emb)
        if 'improved' in self.loss:
            idx = self.loss.index('improved')
            loss += self.weight[idx] * self.loss_module['improved'](q_emb, p_emb, n_emb)
        if 'guided' in self.loss:
            idx = self.loss.index('guided')
            loss += self.weight[idx] * self.loss_module['guided'](
                                        q_emb, p_emb, 
                                        guided_q_emb, guided_p_emb,
                                        n_emb, guided_n_emb)
        if 'cosine' in self.loss:
            idx = self.loss.index('cosine')
            loss += self.weight[idx] * self.loss_module['cosine'](q_emb, p_emb, n_emb)
        if 'angle' in self.loss:
            idx = self.loss.index('angle')
            loss += self.weight[idx] * self.loss_module['angle'](q_emb, p_emb, n_emb)
        return loss


class BPRLoss(nn.Module):
    def __init__(self, neg_count=1, dns=False):
        super().__init__()
        self.score_fn = CosineScorer()
        self.neg_count = neg_count
        self.dns = dns

    def forward(self, text_emb, ctr_emb):
        pos_score = self.score_fn(text_emb, ctr_emb)
        neg_ctr_emb = self.sample(ctr_emb)
        neg_score = self.score_fn(text_emb, neg_ctr_emb)
        if not self.dns:
            loss = F.logsigmoid(pos_score.view(*pos_score.shape, 1) - neg_score)
            weight = F.softmax(torch.ones_like(neg_score), -1)
            final_loss = -(loss * weight).sum(-1)
        else:
            final_loss = -F.logsigmoid(pos_score - torch.max(neg_score, dim=-1))

        return torch.mean(final_loss)
        
    def sample(self, emb):
        bs = emb.shape[0]
        neg_idx = torch.randint(bs - 1, size=(bs, self.neg_count))
        pos_idx = torch.arange(bs).unsqueeze(-1)
        neg_idx[neg_idx >= pos_idx] += 1
        neg_emb = emb[neg_idx]
        return neg_emb
    
    
class AsymmInBatchNegativeInfoNCELoss(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.score_fn = nn.CosineSimilarity(dim=-1)
        self.temperature = temperature

    def forward(self, q_emb, p_emb):
        sim_qp = self.score_fn(q_emb.unsqueeze(1), p_emb.unsqueeze(0))

        scores_qp = sim_qp / self.temperature
        labels = torch.arange(scores_qp.size(0)).long().to(p_emb.device)
        loss = nn.CrossEntropyLoss()(scores_qp, labels)
        return loss
    

class InBatchNegativeInfoNCELoss(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.score_fn = nn.CosineSimilarity(dim=-1)
        self.temperature = temperature

    def forward(self, q_emb, p_emb, n_emb=None):
        # p_emb: B x D
        # q_emb: B x D
        # n_emb: B x N x D
        sim_qp = self.score_fn(q_emb.unsqueeze(1), p_emb.unsqueeze(0))
        sim_pq = self.score_fn(p_emb.unsqueeze(1), q_emb.unsqueeze(0))
        scores_pq = sim_pq / self.temperature
        if n_emb is None:
            scores_qp = sim_qp / self.temperature
        else:
            sim_qn = self.score_fn(p_emb.unsqueeze(1), n_emb) # B x N  
            scores_qp = torch.cat([sim_qp, sim_qn], dim=1) / self.temperature

        labels = torch.arange(scores_qp.size(0)).long().to(p_emb.device)
        loss = nn.CrossEntropyLoss()(scores_qp, labels) + nn.CrossEntropyLoss()(scores_pq, labels)
        return loss


class ImprovedInBatchNegativeInfoNCELoss(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.score_fn = nn.CosineSimilarity(dim=-1)
        self.temperature = temperature
    
    def forward(self, q_emb, p_emb, n_emb=None):
        sim_qp = self.score_fn(q_emb.unsqueeze(1), p_emb.unsqueeze(0))
        sim_pp = self.score_fn(p_emb.unsqueeze(1), p_emb.unsqueeze(0))
        sim_qq = self.score_fn(q_emb.unsqueeze(1), q_emb.unsqueeze(0))
        sim_pp.fill_diagonal_(-torch.inf)
        sim_qq.fill_diagonal_(-torch.inf)
        if n_emb is None:
            scores = torch.cat([sim_qp, sim_qq, sim_pp], dim=1) / self.temperature
        else:
            sim_qn = self.score_fn(p_emb.unsqueeze(1), n_emb)
            scores = torch.cat([sim_qp, sim_qq, sim_pp, sim_qn], dim=1) / self.temperature

        labels = torch.arange(scores.size(0)).long().to(p_emb.device)
        loss = nn.CrossEntropyLoss()(scores, labels)
        
        return loss
    

class GuidedImprovedInBatchNegativeInfoNCELoss(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.score_fn = nn.CosineSimilarity(dim=-1)
        self.temperature = temperature
    
    def forward(self, q_emb, p_emb, guided_q_emb, guided_p_emb, n_emb=None, guided_n_emb=None):
        sim_qp = self.score_fn(q_emb.unsqueeze(1), p_emb.unsqueeze(0))
        sim_pp = self.score_fn(p_emb.unsqueeze(1), p_emb.unsqueeze(0))
        sim_qq = self.score_fn(q_emb.unsqueeze(1), q_emb.unsqueeze(0))
        
        guided_sim_qp = self.score_fn(guided_q_emb.unsqueeze(1), guided_p_emb.unsqueeze(0))
        guided_sim_pp = self.score_fn(guided_p_emb.unsqueeze(1), guided_p_emb.unsqueeze(0))
        guided_sim_qq = self.score_fn(guided_q_emb.unsqueeze(1), guided_q_emb.unsqueeze(0))

        guided_sim = guided_sim_qp.diagonal().view(-1, 1)

        mask_qp = guided_sim_qp > guided_sim
        mask_pp = guided_sim_pp > guided_sim
        mask_qq = guided_sim_qq > guided_sim

        sim_qp[mask_qp] = -torch.inf
        sim_pp[mask_pp] = -torch.inf
        sim_qq[mask_qq] = -torch.inf

        if n_emb is None or guided_n_emb is None:
           scores = torch.cat([sim_qp, sim_qq, sim_pp], dim=1) / self.temperature
        else:
            sim_qn = self.score_fn(p_emb.unsqueeze(1), n_emb)
            guided_sim_qn = self.score_fn(guided_q_emb.unsqueeze(1), guided_n_emb)
            mask_qn = guided_sim_qn > guided_sim
            sim_qn[mask_qn] = -torch.inf

            scores = torch.cat([sim_qp, sim_qq, sim_pp, sim_qn], dim=1) / self.temperature

        labels = torch.arange(scores.size(0)).long().to(p_emb.device)
        loss = nn.CrossEntropyLoss()(scores, labels)
        
        return loss        
    

class CosineLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()
        self.score_fn = nn.CosineSimilarity(dim=-1)
        self.temperature = temperature

    def forward(self, q_emb, p_emb, n_emb):
        # p_emb: B x D
        # q_emb: B x D
        # n_emb: B x N x D
        bs = n_emb.shape[0]
        sim_qp = self.score_fn(q_emb, p_emb)                            # B
        sim_qn = self.score_fn(p_emb.unsqueeze(1), n_emb).flatten()     # BxN
        sim = torch.cat([sim_qp, sim_qn]) / self.temperature            # B+BxN
        diff = sim[:, None] - sim[None, :]                              # (B+BxN) x (B+BxN)
        diff = diff[bs:, :bs].flatten()
        diff = torch.cat([torch.tensor([0], device=q_emb.device), diff])
        loss = torch.logsumexp(diff, dim=0)
        return loss


class AngleLoss(nn.Module):
    def __init__(self, temperature=0.05, pooling_strategy='sum'):
        super().__init__()
        self.temperature = temperature
        self.pooling_strategy = pooling_strategy

    def forward(self, q_emb, p_emb, n_emb):
        # p_emb: B x D
        # q_emb: B x D
        # n_emb: B x N x D
        bs = n_emb.shape[0]
        q_re, q_im = torch.chunk(q_emb, 2, dim=-1)   # B x D/2
        p_re, p_im = torch.chunk(p_emb, 2, dim=-1)   # B x D/2
        n_re, n_im = torch.chunk(n_emb, 2, dim=-1)   # B x N x D/2
        
        a, b = q_re, q_im                                       # B         x D/2
        c = torch.cat([p_re, n_re.flatten(end_dim=-2)], dim=0)  # (B + BxN) x D/2
        d = torch.cat([p_im, n_im.flatten(end_dim=-2)], dim=0)  # (B + BxN) x D/2

        a = a.unsqueeze(1)                                      # B x 1       x D/2
        b = b.unsqueeze(1)                                      # B x 1       x D/2
        c = c.reshape(bs, -1, c.shape[-1])                      # B x (1 + N) x D/2
        d = d.reshape(bs, -1, d.shape[-1])                      # B x (1 + N) x D/2

        re = a * c + b * d                                      # B x (1 + N) x D/2
        im = b * c - a * d                                      # B x (1 + N) x D/2

        dz = torch.sum(a**2 + b**2, dim=-1, keepdim=True) ** 0.5    # B x 1       x 1
        dw = torch.sum(c**2 + d**2, dim=-1, keepdim=True) ** 0.5    # B x (1 + N) x 1
        re /= (dz * dw)
        im /= (dz * dw)

        sim = torch.concat((re, im), dim=-1)                    # B x (1 + N) x D
        if self.pooling_strategy == 'sum':
            sim = torch.sum(sim, dim=-1)                        # B x (1 + N)
        elif self.pooling_strategy == 'mean':
            sim = torch.mean(sim, dim=-1)
        else:
            raise ValueError(f'Unsupported pooling strategy: {self.pooling_strategy}')
        
        sim = sim.flatten()
        sim = torch.abs(sim) / self.temperature                         # B+BxN
        diff = sim[:, None] - sim[None, :]                              # (B+BxN) x (B+BxN)
        diff = diff[bs:, :bs].flatten()
        diff = torch.cat([torch.tensor([0], device=q_emb.device), diff])
        loss = torch.logsumexp(diff, dim=0)
        return loss



def cosine_loss(y_true: torch.Tensor, y_pred: torch.Tensor, tau: float = 20.0) -> torch.Tensor:
    """
    Compute cosine loss

    :param y_true: torch.Tensor, ground truth.
        The y_true must be zigzag style, such as [x[0][0], x[0][1], x[1][0], x[1][1], ...], where (x[0][0], x[0][1]) stands for a pair.
    :param y_pred: torch.Tensor, model output.
        The y_pred must be zigzag style, such as [o[0][0], o[0][1], o[1][0], o[1][1], ...], where (o[0][0], o[0][1]) stands for a pair.
    :param tau: float, scale factor, default 20

    :return: torch.Tensor, loss value
    """  # NOQA
    # modified from: https://github.com/bojone/CoSENT/blob/124c368efc8a4b179469be99cb6e62e1f2949d39/cosent.py#L79
    y_true = y_true[::2, 0]
    y_true = (y_true[:, None] < y_true[None, :]).float()
    y_pred = F.normalize(y_pred, p=2, dim=1)
    y_pred = torch.sum(y_pred[::2] * y_pred[1::2], dim=1) * tau
    y_pred = y_pred[:, None] - y_pred[None, :]
    y_pred = (y_pred - (1 - y_true) * 1e12).view(-1)
    zero = torch.Tensor([0]).to(y_pred.device)
    y_pred = torch.concat((zero, y_pred), dim=0)
    return torch.logsumexp(y_pred, dim=0)


def angle_loss(y_true: torch.Tensor, y_pred: torch.Tensor, tau: float = 1.0, pooling_strategy: str = 'sum'):
    """
    Compute angle loss

    :param y_true: torch.Tensor, ground truth.
        The y_true must be zigzag style, such as [x[0][0], x[0][1], x[1][0], x[1][1], ...], where (x[0][0], x[0][1]) stands for a pair.
    :param y_pred: torch.Tensor, model output.
        The y_pred must be zigzag style, such as [o[0][0], o[0][1], o[1][0], o[1][1], ...], where (o[0][0], o[0][1]) stands for a pair.
    :param tau: float, scale factor, default 1.0

    :return: torch.Tensor, loss value
    """  # NOQA
    y_true = y_true[::2, 0]
    y_true = (y_true[:, None] < y_true[None, :]).float()

    y_pred_re, y_pred_im = torch.chunk(y_pred, 2, dim=1)
    a = y_pred_re[::2]
    b = y_pred_im[::2]
    c = y_pred_re[1::2]
    d = y_pred_im[1::2]

    # (a+bi) / (c+di)
    # = ((a+bi) * (c-di)) / ((c+di) * (c-di))
    # = ((ac + bd) + i(bc - ad)) / (c^2 + d^2)
    # = (ac + bd) / (c^2 + d^2) + i(bc - ad)/(c^2 + d^2)
    z = torch.sum(c**2 + d**2, dim=1, keepdim=True)
    re = (a * c + b * d) / z
    im = (b * c - a * d) / z

    dz = torch.sum(a**2 + b**2, dim=1, keepdim=True)**0.5
    dw = torch.sum(c**2 + d**2, dim=1, keepdim=True)**0.5
    re /= (dz / dw)
    im /= (dz / dw)

    y_pred = torch.concat((re, im), dim=1)
    if pooling_strategy == 'sum':
        pooling = torch.sum(y_pred, dim=1)
    elif pooling_strategy == 'mean':
        pooling = torch.mean(y_pred, dim=1)
    else:
        raise ValueError(f'Unsupported pooling strategy: {pooling_strategy}')
    
    y_pred = torch.abs(pooling) * tau  # absolute delta angle
    y_pred = y_pred[:, None] - y_pred[None, :]
    y_pred = (y_pred - (1 - y_true) * 1e12).view(-1)
    zero = torch.Tensor([0]).to(y_pred.device)
    y_pred = torch.concat((zero, y_pred), dim=0)
    return torch.logsumexp(y_pred, dim=0)


def guided_in_batch_contrastive_loss(args, guide, model, inputs, return_outputs=False):
    return
    loss = 0

    similarity_fct = nn.CosineSimilarity(dim=-1)
    embeddings_query, embeddings_pos, embeddings_neg = get_embeddings(model, inputs, return_weights=False)

    # Compute the model's similarities
    qp_sim = similarity_fct(embeddings_query.unsqueeze(1), embeddings_pos.unsqueeze(0))
    qn_sim = similarity_fct(embeddings_query.unsqueeze(1), embeddings_neg.unsqueeze(0))
    qq_sim = similarity_fct(embeddings_query.unsqueeze(1), embeddings_query.unsqueeze(0))
    pp_sim = similarity_fct(embeddings_pos.unsqueeze(1), embeddings_pos.unsqueeze(0))

    if guide is None:
        # Adopt the contrastive loss without the guide based on the partition function
        # used in: https://arxiv.org/pdf/2308.03281.pdf
        # Towards General Text Embeddings with Multi-stage Contrastive Learning
        # This loss function is a bit different compared with the bidirectional contrastive loss
        # used in the INSTRUCTOR paper (https://arxiv.org/pdf/2212.09741.pdf), particularly
        # because we don't split the partition function for the query and the document as
        # two separate components of the loss.

        # Mask the diagonal elements for i==j query and document pairs.
        qq_sim.fill_diagonal_(-torch.inf)
        pp_sim.fill_diagonal_(-torch.inf)
    else:
        # Compute the guide's similarities
        guide_embeddings_query, guide_embeddings_pos, guide_embeddings_neg = get_guide_embeddings(guide, inputs)

        # This contains the cosine similarities of the query with respect to hard negatives
        guided_qp_sim = similarity_fct(guide_embeddings_query.unsqueeze(1), guide_embeddings_pos.unsqueeze(0))
        guided_qn_sim = similarity_fct(guide_embeddings_query.unsqueeze(1), guide_embeddings_neg.unsqueeze(0))
        guided_qq_sim = similarity_fct(guide_embeddings_query.unsqueeze(1), guide_embeddings_query.unsqueeze(0))
        guided_pp_sim = similarity_fct(guide_embeddings_pos.unsqueeze(1), guide_embeddings_pos.unsqueeze(0))

        guided_sim = guided_qp_sim.diagonal().view(-1, 1)

        # Find which samples cannot be used as negatives because they are
        # more similar to the query than the assigned positive as deemed by the guide model.
        # For this samples, we mask them with -inf to basically ignore their contribution to
        # the loss.
        qp_mask = guided_qp_sim > guided_sim
        qn_mask = guided_qn_sim > guided_sim
        qq_mask = guided_qq_sim > guided_sim  # This should take care of masking the diagonal.
        pp_mask = guided_pp_sim > guided_sim  # This should take care of masking the diagonal.

        if args.gist_loss_type.startswith("guided-triplet") and args.gist_tl_margin > 0:
            # If we use triplet loss, we go here.
            # We use all the q* matrices as basis for computing
            # the triplet loss.
            qp_tr_mask = guided_qp_sim >= guided_sim

            # From the qp, qn, and qq masks, we find values greater than or equal to guided_sim
            # as the pool of candidate positives. Less than the guided_sim are candidate negatives.

            # For a given query, we find the sample with lowest similarity score among the candidate positives
            # as the triplet positive. We also choose the sample with the highest similarity score among
            # the candidate negatives as the triplet positive.

            # These are "positive" pairs.
            tr_mask = torch.cat([qp_tr_mask, qn_mask, qq_mask], dim=1)
            q = torch.cat([qp_sim, qn_sim, qq_sim], dim=1)

            if args.gist_loss_type == "guided-triplet":
                # Hard
                qp_dist = (q * tr_mask).argmin(dim=1)
                qn_dist = (q * (~tr_mask)).argmax(dim=1)

                # Subtract from 1 to get the cosine distance.
                _p = 1 - q[torch.arange(q.size(0)).long().to(q.device), qp_dist]
                _n = 1 - q[torch.arange(q.size(0)).long().to(q.device), qn_dist]
            elif args.gist_loss_type == "guided-triplet-soft":
                # Soft
                qp_dist = (q * tr_mask)
                qn_dist = (q * (~tr_mask))

                # Note that since qq_mask contains the self similarity of q-q.
                # We should adjust for that below by subtracting 1 in the similarity
                # score and the total components in the denominator.
                # We only do this adjustment to the qp_dist.

                # Convert to cosine distance
                qp_dist[tr_mask] = 1 - qp_dist[tr_mask]
                qn_dist[~tr_mask] = 1 - qn_dist[~tr_mask]

                # Get the mean accounting for q-q in the denominator (-1)
                p_den = tr_mask.sum(dim=-1) - 1
                n_den = (~tr_mask).sum(dim=-1)

                # We exclude in the loss computation samples without negatives
                # or positives.
                sample_mask = (p_den != 0) & (n_den != 0)

                p_den = p_den[sample_mask]
                n_den = n_den[sample_mask]

                _p = qp_dist.sum(dim=-1)[sample_mask] / p_den
                _n = qn_dist.sum(dim=-1)[sample_mask] / n_den
            else:
                raise ValueError(f"Unsupported loss type: {args.gist_loss_type}")

            # Hard triplet margin loss
            tml = (_p - _n + args.gist_tl_margin)
            tml[tml < 0] = 0

            loss += torch.mean(tml)

        if args.gist_negative_mode == "hard":
            # Find the hard negatives defined by examples that the model
            # finds more similar to the query than the assigned positive, but
            # the guide model finds them as negatives.

            model_sim = qp_sim.diagonal().view(-1, 1)

            # We find samples the model already finds as negatives.
            # This means the similarity of the query with respect to the positive
            # is greater than the similarity of the query with respect to the other potential negatives.
            model_qp_mask = qp_sim < model_sim
            model_qn_mask = qn_sim < model_sim
            model_qq_mask = qq_sim < model_sim
            model_pp_mask = pp_sim < model_sim

            qp_mask = qp_mask | model_qp_mask
            qn_mask = qn_mask | model_qn_mask
            qq_mask = qq_mask | model_qq_mask
            pp_mask = pp_mask | model_pp_mask

            # # TODO: We need to handle cases where there are no hard negatives left.
            # mask = torch.cat([qp_mask, qn_mask, qq_mask, pp_mask], dim=1)
            # num_negatives = (~mask).sum(dim=1)

            # # We find samples without hard negatives left.
            # # This means the similarity of the query with respect to the positive
            # # is greater than the similarity of the query with respect to the other potential negatives.

        qp_sim[qp_mask] = -torch.inf
        qn_sim[qn_mask] = -torch.inf
        qq_sim[qq_mask] = -torch.inf
        pp_sim[pp_mask] = -torch.inf

    scores = torch.cat([qp_sim, qn_sim, qq_sim, pp_sim], dim=1) / args.gist_cl_temperature
    labels = torch.arange(scores.size(0)).long().to(embeddings_query.device)
    loss += nn.CrossEntropyLoss()(scores, labels)

    if return_outputs:
        # Note that we only return the contrastive loss scores here.
        outs = (loss, scores)
    else:
        outs = loss

    return outs