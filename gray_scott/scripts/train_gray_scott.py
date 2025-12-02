import torch
from torch.utils.data import dataloader
from einops import rearrange
from tqdm import tqdm

from neuralop.models import FNO
from the_well.data import WellDataset
from the_well.utils.download import well_download
from the_well.benchmark.metrics import VRMSE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_PATH = "./datasets"
DATASET_NAME = "gray_scott_reaction_diffusion"

N_STPES_INPUT = 4
N_STEP_OUTPUT = 1
BATCH_SIZE = 4
LR = 5e-3
EPOCHS = 5

# get the train and validation set
def get_datasets():
    train_data = WellDataset(well_base_path=BASE_PATH, 
                             well_dataset_name=DATASET_NAME, 
                             well_split_name="train",
                             n_steps_input=N_STPES_INPUT,
                             n_steps_output=N_STEP_OUTPUT,
                             use_normalization="False",
                             )
    
    val_data = WellDataset(well_base_path=BASE_PATH,
                           well_dataset_name=DATASET_NAME,
                           well_split_name="valid",
                           n_steps_input=N_STPES_INPUT,
                           n_steps_output=N_STEP_OUTPUT,
                           use_normalization=False,
                           )
    return train_data, val_data

# compute mu and sigma for normalization
def compute_normalization(dataset, F, n_samples=1000):
    xs = []
    steps = max(1, dataset.len // n_samples)

    for i in range(0, dataset.len, steps):
        x = dataset[i]["input_fields"]
        xs.append(x)
        if len(xs) >= n_samples:
            break
    torch.stack(xs)
    xs = xs.reshape(-1, F)

    mu = xs.mean(dim=0).to(DEVICE)
    sigma = xs.std(dim=0).to(DEVICE)

    sigma[sigma == 0] = 1.0

    return mu, sigma

def preprocess(x, mu, sigma):
    return (x - mu) / sigma

def postprocess(x, mu, sigma):
    return x * sigma + mu

