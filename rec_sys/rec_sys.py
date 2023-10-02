from functools import partial

import torch
from torch import nn
from feature_extraction.feature_extractors import FeatureExtractor
from utilities.utils import general_weight_init


class RecSysMeTa(nn.Module):

    def __init__(self,
                 user_sim, item_sim, proto_weight,
                 loss_func_name: str, loss_func_aggr: str = 'mean'):
        """
        General Recommender System
        It generates the user/item vectors (given the feature extractors) and computes the similarity by the dot product.
        :param n_users: number of users in the system
        :param n_items: number of items in the system
        :param user_feature_extractor: feature_extractor.FeatureExtractor module that generates user embeddings.
        :param item_feature_extractor: feature_extractor.FeatureExtractor module that generates item embeddings.
        :param loss_func_name: name of the loss function to use for the network.
        :param loss_func_aggr: type of aggregation for the loss function, either 'mean' or 'sum'.
        """

        assert loss_func_aggr in ['mean', 'sum'], f'Loss function aggregators <{loss_func_aggr}> not implemented...yet'

        super().__init__()

        self.user_sim = user_sim
        self.item_sim = item_sim
        self.proto_weight = proto_weight

        self.loss_func_name = loss_func_name
        self.loss_func_aggr = loss_func_aggr

        if self.loss_func_name == 'bce':
            self.rec_loss = partial(bce_loss, aggregator=self.loss_func_aggr)
        elif self.loss_func_name == 'bpr':
            self.rec_loss = partial(bpr_loss, aggregator=self.loss_func_aggr)
        elif self.loss_func_name == 'sampled_softmax':
            self.rec_loss = partial(sampled_softmax_loss, aggregator=self.loss_func_aggr)
        else:
            raise ValueError(f'Recommender System Loss function <{self.rec_loss}> Not Implemented... Yet')

        self.initialized = False

        # print(f'Built RecSys module \n'
        #       f'- loss_func_name: {self.loss_func_name} \n')

    def init_parameters(self):
        """
        Method for initializing the Recommender System Processor
        """
        self.user_sim.init_parameters()
        self.item_sim.init_parameters()

        # if self.user_attention:
        #     self.user_exp_embedding.init_parameters()
        self.initialized = True

    def loss_func(self, uipc_mf_l1_logits, labels):
        """
        Loss function of the Recommender System module. It takes into account eventual feature_extractor loss terms.
        NB. Any feature_extractor loss is pre-weighted.
        :param logits: output of the system.
        :param labels: binary labels
        :return: aggregated loss
        """
        proto_reg = self.user_sim.get_and_reset_loss()+self.item_sim.get_and_reset_loss()
        uipc_mf_l1_loss = self.rec_loss(uipc_mf_l1_logits, labels)
        total_loss = uipc_mf_l1_loss + proto_reg
        return total_loss

    def forward(self, u_idxs, i_idxs):
        """
        Performs the forward pass considering user indexes and the item indexes. Negative Sampling is done automatically
        by the dataloader
        :param u_idxs: User indexes. Shape is (batch_size,)
        :param i_idxs: Item indexes. Shape is (batch_size, n_neg + 1)

        :return: A matrix of logits values. Shape is (batch_size, 1 + n_neg). First column is always associated
                to the positive track.
        """
        assert self.initialized, 'Model initialization has not been called! Please call .init_parameters() ' \
                                 'before using the model'

        # --- User pass ---
        u_embed = self.user_sim(u_idxs)  # [batch,user_n_prototypes]
        i_embed = self.item_sim(i_idxs)  # [batch,neg+1,item_n_prototypes]
        weights = self.proto_weight()  # [user_n_prototypes,item_n_prototypes]]
        u_embed_ext = u_embed.unsqueeze(1).expand(-1, i_embed.shape[1], -1) # [batch,neg+1,user_n_prototypes]
        out = torch.einsum('bij,bik,jk->bi', u_embed_ext, i_embed, weights)
        u_prefer_l1 = torch.linalg.vector_norm(torch.matmul(u_embed,weights),ord=1)/u_embed.shape[0]
        return out,u_prefer_l1


def bce_loss(logits, labels, aggregator='mean'):
    """
    It computes the binary cross entropy loss with negative sampling, expressed by the formula:
                                    -∑_j log(x_ui) + log(1 - x_uj)
    where x_ui and x_uj are the prediction for user u on item i and j, respectively. Item i positive instance while
    Item j is a negative instance. The Sum is carried out across the different negative instances. In other words
    the positive item is weighted as many as negative items are considered.

    :param logits: Logits values from the network. The first column always contain the values of positive instances.
            Shape is (batch_size, 1 + n_neg).
    :param labels: 1-0 Labels. The first column contains 1s while all the others 0s.
    :param aggregator: function to use to aggregate the loss terms. Default to mean

    :return: The binary cross entropy as computed above
    """
    weights = torch.ones_like(logits)
    weights[:, 0] = logits.shape[1] - 1

    loss = nn.BCEWithLogitsLoss(weights.flatten(), reduction=aggregator)(logits.flatten(), labels.flatten())

    return loss


def bpr_loss(logits, labels, aggregator='mean'):
    """
    It computes the Bayesian Personalized Ranking loss (https://arxiv.org/pdf/1205.2618.pdf).

    :param logits: Logits values from the network. The first column always contain the values of positive instances.
            Shape is (batch_size, 1 + n_neg).
    :param labels: 1-0 Labels. The first column contains 1s while all the others 0s.
    :param aggregator: function to use to aggregate the loss terms. Default to mean

    :return: The bayesian personalized ranking loss
    """
    pos_logits = logits[:, 0].unsqueeze(1)  # [batch_size,1]
    neg_logits = logits[:, 1:]  # [batch_size,n_neg]

    labels = labels[:, 0]  # I guess this is just to avoid problems with the device
    labels = torch.repeat_interleave(labels, neg_logits.shape[1])

    diff_logits = pos_logits - neg_logits

    loss = nn.BCEWithLogitsLoss(reduction=aggregator)(diff_logits.flatten(), labels.flatten())

    return loss


def sampled_softmax_loss(logits, labels, aggregator='sum'):
    """
    It computes the (Sampled) Softmax Loss (a.k.a. sampled cross entropy) expressed by the formula:
                        -x_ui +  log( ∑_j e^{x_uj})
    where x_ui and x_uj are the prediction for user u on item i and j, respectively. Item i positive instance while j
    goes over all the sampled items (negatives + the positive).
    :param logits: Logits values from the network. The first column always contain the values of positive instances.
            Shape is (batch_size, 1 + n_neg).
    :param labels: 1-0 Labels. The first column contains 1s while all the others 0s.
    :param aggregator: function to use to aggregate the loss terms. Default to sum
    :return:
    """

    pos_logits_sum = - logits[:, 0]
    log_sum_exp_sum = torch.logsumexp(logits, dim=-1)

    sampled_loss = pos_logits_sum + log_sum_exp_sum

    if aggregator == 'sum':
        return sampled_loss.sum()
    elif aggregator == 'mean':
        return sampled_loss.mean()
    else:
        raise ValueError('Loss aggregator not defined')
