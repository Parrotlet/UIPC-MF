import argparse
import os

from confs.hyper_params import uipc_mf_l1_hyper_params
from experiment_helper import start_hyper, start_multiple_hyper
from utilities.consts import SINGLE_SEED, SEED_LIST

os.environ['WANDB_START_METHOD'] = 'thread'
os.environ['OMP_NUM_THREADS'] = '1'

parser = argparse.ArgumentParser(description='Start an experiment')

parser.add_argument('--model', '-m', type=str, help='Recommender System model',
                    choices=['uipc_mf_l1'], default='uipc_mf_l1')

parser.add_argument('--dataset', '-d', type=str, help='Recommender System Dataset',
                    choices=['amazon2014', 'ml-1m', 'm4_3m'], default='ml-1m')

parser.add_argument('--multiple', '-mp',
                    help='Whether to run the experiment across all seeds (see utilities/consts.py)',
                    action='store_true', default=True, required=False)
parser.add_argument('--seed', '-s', help='Seed to set for the experiments', type=int, default=SEED_LIST,
                    required=False)

args = parser.parse_args()

model = args.model
dataset = args.dataset
multiple = args.multiple
seed = args.seed

conf_dict = None
if model == 'uipc_mf_l1':
    conf_dict = uipc_mf_l1_hyper_params

if multiple:
    start_multiple_hyper(conf_dict, model, dataset)
else:
    start_hyper(conf_dict, model, dataset, seed)
