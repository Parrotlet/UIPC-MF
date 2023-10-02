from abc import abstractmethod, ABC

import torch
import torch.nn as nn
import scipy.special


from utilities.utils import general_weight_init


class FeatureExtractor(nn.Module, ABC):
    """
    Abstract class representing one of the possible FeatureExtractor models. See also FeatureExtractorFactory.
    """

    def __init__(self):
        super().__init__()
        with torch.no_grad():
            self.cumulative_loss = 0.
            self.name = "FeatureExtractor"

    def init_parameters(self):
        """
        Initial the Feature Extractor parameters
        """
        pass

    def get_and_reset_loss(self) -> float:
        """
        Reset the loss of the feature extractor and returns the computed value
        :return: loss of the feature extractor
        """
        with torch.no_grad():
            loss = self.cumulative_loss
            self.cumulative_loss = 0.
        return loss

    @abstractmethod
    def forward(self, o_idxs: torch.Tensor) -> torch.Tensor:
        """
        Performs the feature extraction process of the object.
        """
        pass


class Embedding(FeatureExtractor):
    """
    FeatureExtractor that represents an object (item/user) only given by its embedding.
    """

    def __init__(self, n_objects: int, embedding_dim: int, max_norm: float = None, only_positive: bool = False):
        """
        Standard Embedding Layer
        :param n_objects: number of objects in the system (users or items)
        :param embedding_dim: embedding dimension
        :param max_norm: max norm of the l2 norm of the embeddings.
        :param only_positive: whether the embeddings can be only positive
        """
        super().__init__()
        with torch.no_grad():
            self.n_objects = n_objects
            self.embedding_dim = embedding_dim
            self.max_norm = max_norm
            self.only_positive = only_positive
            self.name = "Embedding"

        self.embedding_layer = nn.Embedding(self.n_objects, self.embedding_dim, max_norm=self.max_norm)
        # print(f'Built Embedding model \n'
        #       f'- n_objects: {self.n_objects} \n'
        #       f'- embedding_dim: {self.embedding_dim} \n'
        #       f'- max_norm: {self.max_norm}\n'
        #       f'- only_positive: {self.only_positive}')

    def init_parameters(self):
        self.embedding_layer.apply(general_weight_init)

    def forward(self, o_idxs: torch.Tensor) -> torch.Tensor:
        assert o_idxs is not None, f"Object Indexes not provided! ({self.name})"
        embeddings = self.embedding_layer(o_idxs)
        if self.only_positive:
            embeddings = torch.absolute(embeddings)
        return embeddings


class PrototypeVectors(nn.Module):
    """
    PrototypeVectors that represents a set of prototype.
    """

    def __init__(self, embedding_dim: int, n_prototypes: int = None):
        """
        :param embedding_dim: embedding dimension
        :param n_prototypes: number of prototypes to consider. If none, is set to be embedding_dim.
        """
        super().__init__()
        with torch.no_grad():
            self.embedding_dim = embedding_dim
            self.n_prototypes = n_prototypes
            self.name = "PrototypeVectors"

        if self.n_prototypes is None:
            self.prototypes = nn.Parameter(torch.randn([self.embedding_dim, self.embedding_dim]))
            self.n_prototypes = self.embedding_dim
        else:
            self.prototypes = nn.Parameter(torch.randn([self.n_prototypes, self.embedding_dim]))
        # print(f'Built PrototypeVectors \n'
        #       f'- embedding_dim: {self.embedding_dim} \n'
        #       f'- n_prototypes: {self.n_prototypes}\n')

    def forward(self) -> torch.Tensor:
        output = self.prototypes
        return output


class ProtoWeights(nn.Module):
    """
    PrototypeVectors that represents a set of prototype.
    """

    def __init__(self, n_user_prototype: int, n_item_prototype: int):
        """
        :param embedding_dim: embedding dimension
        :param n_prototypes: number of prototypes to consider. If none, is set to be embedding_dim.
        """
        super().__init__()
        with torch.no_grad():
            self.n_user_prototype = n_user_prototype
            self.n_item_prototype = n_item_prototype
            self.name = "ProtoWeights"
        self.proto_weight = nn.Parameter(torch.randn([self.n_user_prototype, self.n_item_prototype]))

    def forward(self) -> torch.Tensor:
        output = self.proto_weight
        return output


class PrototypeSimilarity(FeatureExtractor):
    """
    ProtoMF building block. It represents an object (item/user) given the similarity with the prototypes.
    """

    def __init__(self, embeddingw: FeatureExtractor, prototypevectors: nn.Module,
                 sim_proto_weight: float = 1., sim_batch_weight: float = 1., proto_reg_weight: float = 1.,
                 reg_proto_type: str = 'soft', reg_batch_type: str = 'soft', cosine_type: str = 'shifted',
                 using_reg: bool = True, proto_reg: bool = True):
        """
        :param sim_proto_weight: factor multiplied to the regularization loss for prototypes
        :param sim_batch_weight: factor multiplied to the regularization loss for batch
        :param reg_proto_type: type of regularization applied batch-prototype similarity matrix on the prototypes. Possible values are ['max','soft','incl']
        :param reg_batch_type: type of regularization applied batch-prototype similarity matrix on the batch. Possible values are ['max','soft']
        :param cosine_type: type of cosine similarity to apply. Possible values ['shifted','standard','shifted_and_div']

        """

        super().__init__()
        self.embeddingw = embeddingw
        self.prototypevectors = prototypevectors
        with torch.no_grad():
            self.sim_proto_weight = sim_proto_weight
            self.sim_batch_weight = sim_batch_weight
            self.proto_reg_weight = proto_reg_weight
            self.reg_proto_type = reg_proto_type
            self.reg_batch_type = reg_batch_type
            self.cosine_type = cosine_type
            self.using_reg = using_reg
            self.proto_reg = proto_reg
            self.proto_num = self.prototypevectors.n_prototypes

            # Cosine Type
            if self.cosine_type == 'standard':
                self.cosine_sim_func = nn.CosineSimilarity(dim=-1)
            elif self.cosine_type == 'shifted':
                self.cosine_sim_func = lambda x, y: (1 + nn.CosineSimilarity(dim=-1)(x, y))
            elif self.cosine_type == 'shifted_and_div':
                self.cosine_sim_func = lambda x, y: (1 + nn.CosineSimilarity(dim=-1)(x, y)) / 2
            else:
                raise ValueError(f'Cosine type {self.cosine_type} not implemented')

            # Regularization Batch
            if self.reg_batch_type == 'max':
                self.reg_batch_func = lambda x: - x.max(dim=1).values.mean()
            elif self.reg_batch_type == 'soft':
                self.reg_batch_func = lambda x: self._entropy_reg_loss(x, 1)
            else:
                raise ValueError(f'Regularization Type for Batch {self.reg_batch_func} not yet implemented')

            # Regularization Proto
            if self.reg_proto_type == 'max':
                self.reg_proto_func = lambda x: - x.max(dim=0).values.mean()
            elif self.reg_proto_type == 'soft':
                self.reg_proto_func = lambda x: _entropy_reg_loss(x, 0)
            elif self.reg_proto_type == 'incl':
                self.reg_proto_func = lambda x: _inclusiveness_constraint(x)
            else:
                raise ValueError(f'Regularization Type for Proto {self.reg_proto_type} not yet implemented')
            self._acc_r_proto = 0
            self._acc_r_batch = 0
            self._proto_reg_value = 0
            self.name = "PrototypeEmbedding"

        # print(f'Built PrototypeEmbedding model \n'
        #       f'- sim_proto_weight: {self.sim_proto_weight} \n'
        #       f'- sim_batch_weight: {self.sim_batch_weight} \n'
        #       f'- reg_proto_type: {self.reg_proto_type} \n'
        #       f'- reg_batch_type: {self.reg_batch_type} \n'
        #       f'- cosine_type: {self.cosine_type} \n')
    def proto_reg_func(self):
        protos = self.prototypevectors()
        sim = self.cosine_sim_func(protos.unsqueeze(1), protos.unsqueeze(0))
        self._proto_reg_value = ((torch.sum(sim)-torch.trace(sim))/((protos.shape[0]*(protos.shape[0]-1))/2)).item()
        # # _proto_reg_value=0
        # #
        # # for i in range(self.proto_num):
        # #     for j in range(i+1, self.proto_num):
        # #         _proto_reg_value += self.cosine_sim_func(protos[i], protos[j]).item()
        # num_pairs = scipy.special.comb(self.proto_num, 2)
        # # self._proto_reg_value = (_proto_reg_value / num_pairs).item()
        # self._proto_reg_value = sum(self.cosine_sim_func(protos[i], protos[j])for i in range(self.proto_num) for j in range(i+1, self.proto_num))/num_pairs
    def init_parameters(self):
        self.embeddingw.init_parameters()

    def forward(self, o_idxs: torch.Tensor) -> torch.Tensor:
        """
        :param o_idxs: Shape is either [batch_size] or [batch_size,n_neg_p_1]
        :return:
        """
        # assert o_idxs is not None, "Object indexes not provided"
        # assert len(o_idxs.shape) == 2 or len(o_idxs.shape) == 1, \
        #     f'Object indexes have shape that does not match the network ({o_idxs.shape})'

        sim_mtx = self.embeddingw(o_idxs)  # [..., embedding_dim]

        # https://github.com/pytorch/pytorch/issues/48306 sim_mtx
        sim_mtx = self.cosine_sim_func(sim_mtx.unsqueeze(-2), self.prototypevectors())  # [..., n_prototypes]

        output = sim_mtx  # [..., embedding_dim = n_prototypes]

        # Computing additional losses
        batch_proto = sim_mtx.reshape([-1, sim_mtx.shape[-1]])
        if self.using_reg:
            self._acc_r_batch += self.reg_batch_func(batch_proto).item()
            self._acc_r_proto += self.reg_proto_func(batch_proto).item()
        if self.proto_reg:
            self.proto_reg_func()
        return output

    def get_and_reset_loss(self) -> float:
        with torch.no_grad():
            acc_r_proto, acc_r_batch, _proto_reg_value = self._acc_r_proto, self._acc_r_batch, self._proto_reg_value
            self._acc_r_proto = self._acc_r_batch = self._proto_reg_value = 0
        return self.sim_proto_weight * acc_r_proto + self.sim_batch_weight * acc_r_batch + self.proto_reg_weight * _proto_reg_value




@torch.no_grad()
def _entropy_reg_loss(sim_mtx, axis: int):
    o_coeff = nn.Softmax(dim=axis)(sim_mtx)
    entropy = - (o_coeff * torch.log(o_coeff)).sum(axis=axis).mean()
    return entropy


@torch.no_grad()
def _inclusiveness_constraint(sim_mtx):
    '''
    NB. This method is applied only on a square matrix (batch_size,n_prototypes) and it return the negated
    inclusiveness constraints (its minimization brings more equal load sharing among the prototypes)
    '''
    o_coeff = nn.Softmax(dim=1)(sim_mtx)
    q_k = o_coeff.sum(axis=0).div(o_coeff.sum())  # [n_prototypes]
    entropy_q_k = - (q_k * torch.log(q_k)).sum()
    return - entropy_q_k
