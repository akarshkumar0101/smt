import os
import json
import pickle
import math


def save_json(save_dir, name, item):
    if save_dir is not None:
        os.makedirs(f"{save_dir}/", exist_ok=True)
        with open(f"{save_dir}/{name}.json", "w") as f:
            json.dump(item, f)
            
def load_json(load_dir, name):
    if load_dir is not None:
        with open(f"{load_dir}/{name}.json", "r") as f:
            return json.load(f)
    else:
        return None

def save_pkl(save_dir, name, item):
    if save_dir is not None:
        os.makedirs(f"{save_dir}/", exist_ok=True)
        with open(f"{save_dir}/{name}.pkl", "wb") as f:
            pickle.dump(item, f)

def load_pkl(load_dir, name):
    if load_dir is not None:
        with open(f"{load_dir}/{name}.pkl", "rb") as f:
            return pickle.load(f)
    else:
        return None

def save_pkl_path(save_path, item):
    if save_path is not None:
        # os.makedirs(save_path, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(item, f)

def load_pkl_path(load_path):
    if load_path is not None:
        with open(load_path, "rb") as f:
            return pickle.load(f)
    else:
        return None

def get_lr(itr, n_iters, lr, lr_schedule):
    if lr_schedule == "constant":
        coeff = 1.
    elif lr_schedule == "warmup_constant":
        warmup_iters = n_iters // 100
        if itr < warmup_iters:
            coeff = itr / warmup_iters
        else:
            coeff = 1.
    elif lr_schedule == "warmup_cosine": # Linear warmup followed by cosine decay
        warmup_iters = n_iters // 100
        if itr < warmup_iters:
            coeff = itr / warmup_iters
        else:
            decay_ratio = (itr - warmup_iters) / (n_iters - warmup_iters)
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return coeff * lr
