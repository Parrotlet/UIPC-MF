# --- Experiment Constants --- #
SINGLE_SEED = 38210573
SEED_LIST = [SINGLE_SEED, 9491758, 2931009]
NUM_SAMPLES = 100  # How many hyperparameters samples will be taken into account
DATA_PATH = '/home/adam/PycharmProjects/UIPC-MF_L1/data/'  # Path pointing at the data folder
GPU_PER_TRIAL = 0.5  # Ray Tune parameter,  how many gpus are allocated for a single trial experiment (https://docs.ray.io/en/releases-1.9.0/tune/api_docs/execution.html)
CPU_PER_TRIAL = 3 # Ray Tune parameter, how many cpus are allocated for a single trial experiment
# --- Training Constants --- #
MAX_PATIENCE = 10  # How many epochs without an improvement must pass before stopping the experiment
NUM_WORKERS = CPU_PER_TRIAL*4
# --- Evaluation Constants --- #
K_VALUES = [10,5]  # K value for the evaluation metrics
NEG_VAL = 99  # How many negative samples are considered during negative sampling
OPTIMIZING_METRIC = 'hit_ratio@10'  # Which metric will be used to assess during validation.
# --- Logger Constants --- #
WANDB_API_KEY = '1eccd79f96c5a6499f0b6a3663d333dd029d9e1b'  # Weight & Biases key for logging purposes
PROJECT_NAME = 'uipc_ml_1m_l1_4090_test1'
