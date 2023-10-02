from typing import Tuple

import torch
from torch import nn

from feature_extraction.feature_extractors import Embedding, PrototypeVectors, \
    PrototypeSimilarity, ProtoWeights


class FeatureExtractorFactory:

    @staticmethod
    def create_meta_models(ft_ext_param: dict, n_users: int, n_items: int) :

        """
        Helper function to create both the user and item feature extractor. It either creates two detached
        FeatureExtractors or a single one shared by users and items.
        :param ft_ext_param: parameters for the user feature extractor model. ft_ext_param.ft_type is used for
            switching between models.
        :param n_users: number of users in the system.
        :param n_items: number of items in the system.
        :return: [user_feature_extractor, item_feature_extractor]
        """
        assert 'ft_type' in ft_ext_param, "Type has not been specified for FeatureExtractor! " \
                                          "FeatureExtractor model not created"
        ft_type = ft_ext_param['ft_type']
        embedding_dim = ft_ext_param['embedding_dim']
        user_n_prototypes = ft_ext_param['user_ft_ext_param']['n_prototypes']
        item_n_prototypes = ft_ext_param['item_ft_ext_param']['n_prototypes']
        if ft_type == 'uipc_mf_l1':

            # create prototype
            user_prototypes = PrototypeVectors(embedding_dim, user_n_prototypes)
            item_prototypes = PrototypeVectors(embedding_dim, item_n_prototypes)


            user_embedding = Embedding(n_users, embedding_dim)
            item_embedding = Embedding(n_items, embedding_dim)

            user_sim = PrototypeSimilarity(user_embedding, user_prototypes,
                                           ft_ext_param['user_ft_ext_param']['sim_proto_weight'],
                                           ft_ext_param['user_ft_ext_param']['sim_batch_weight'],
                                           ft_ext_param['user_ft_ext_param']['proto_reg_weight'],
                                           ft_ext_param['user_ft_ext_param']['reg_proto_type'],
                                           ft_ext_param['user_ft_ext_param']['reg_batch_type'],
                                           ft_ext_param['user_ft_ext_param']['cosine_type'],
                                           using_reg=ft_ext_param['user_ft_ext_param']['using_reg'],
                                           proto_reg=ft_ext_param['user_ft_ext_param']['proto_reg'])
            item_sim = PrototypeSimilarity(item_embedding, item_prototypes,
                                           ft_ext_param['item_ft_ext_param']['sim_proto_weight'],
                                           ft_ext_param['item_ft_ext_param']['sim_batch_weight'],
                                           ft_ext_param['item_ft_ext_param']['proto_reg_weight'],
                                           ft_ext_param['item_ft_ext_param']['reg_proto_type'],
                                           ft_ext_param['item_ft_ext_param']['reg_batch_type'],
                                           ft_ext_param['item_ft_ext_param']['cosine_type'],
                                           using_reg=ft_ext_param['item_ft_ext_param']['using_reg'],
                                           proto_reg=ft_ext_param['item_ft_ext_param']['proto_reg'])
            proto_weight = ProtoWeights(user_n_prototypes, item_n_prototypes)

            return user_sim, item_sim, proto_weight

        else:
            raise ValueError(f'FeatureExtractor <{ft_type}> Not Implemented..yet')
