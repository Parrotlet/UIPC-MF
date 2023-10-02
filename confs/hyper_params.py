import torch
from ray import tune

base_param = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'n_epochs': 100,
    'eval_neg_strategy': 'uniform',
    'val_batch_size': 256,
}

base_hyper_params = {
    **base_param,
    'neg_train': tune.randint(1, 50),
    'train_neg_strategy': tune.choice(['popular', 'uniform']),
    'loss_func_name': tune.choice(['bce', 'bpr', 'sampled_softmax']),
    'batch_size': tune.lograndint(64, 512, 2),
    'optim_param': {
        'optim': tune.choice(['adam', 'adagrad']),
        'wd': tune.loguniform(1e-4, 1e-2),
        'lr': tune.loguniform(1e-4, 1e-1),
        'u_p_l1': tune.loguniform(1e-4, 1e-1)}
    ,
}

uipc_mf_l1_hyper_params = {
    **base_hyper_params,
    'loss_func_aggr': 'mean',
    'ft_ext_param': {
        "ft_type": "uipc_mf_l1",
        'embedding_dim': tune.randint(10, 100),
        'item_ft_ext_param': {
            'n_prototypes': tune.randint(10, 100),
            'sim_proto_weight': tune.loguniform(1e-3, 10),
            'sim_batch_weight': tune.loguniform(1e-3, 10),
            'proto_reg_weight': 0,
            'cosine_type': 'shifted',
            'reg_proto_type': 'max',
            'reg_batch_type': 'max',
            'using_reg': True,
            'proto_reg': False,
        },
        'user_ft_ext_param': {
            'n_prototypes': tune.randint(10, 100),
            'sim_proto_weight': tune.loguniform(1e-3, 10),
            'sim_batch_weight': tune.loguniform(1e-3, 10),
            'proto_reg_weight': 0,
            'cosine_type': 'shifted',
            'reg_proto_type': 'max',
            'reg_batch_type': 'max',
            'using_reg': True,
            'proto_reg': False,
        }
    }}
