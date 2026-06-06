import torch
from torch.utils.data import Dataset
import numpy as np

from .tinystories import TinyStoriesTokenizer
from .mnist import MnistTokenizer

class MyDataset(Dataset):
    def __init__(self, dataset_path, tokenizer):
        self.dataset_path = dataset_path
        self.tokenizer = tokenizer
        self.data = np.load(f"{self.dataset_path}") # the dataset is just a long numpy array sequence of tokens.

    def __len__(self, ctx_len: int):
        return len(self.data) - ctx_len

    def __getitem__(self, idx: int, ctx_len: int):
        data = self.data[idx:idx+ctx_len]
        return data
    
    def sample_batch(self, batch_size: int, ctx_len: int):
        """
        Sample a batch of data from the dataset.
        Returns a dictionary with keys 'x' and 'y'.
        x: (batch_size, ctx_len) of input tokens
        y: (batch_size, ctx_len) of output target tokens.
        In this case, y[t] = x[t+1]
        """
        len_ = len(self.data) - ctx_len - 1
        idxs = np.random.randint(0, len_, batch_size)
        batch = np.stack([self.data[i:i+ctx_len+1] for i in idxs])
        batch = torch.from_numpy(batch).long()
        return dict(x=batch[:, :-1], y=batch[:, 1:])

def get_dataset(dataset_name):
    if dataset_name == "tinystories":
        tokenizer = TinyStoriesTokenizer()
        dataset_path = "./data/tinystories/"
        train_ds = MyDataset(f"{dataset_path}/train.npy", tokenizer)
        test_ds = MyDataset(f"{dataset_path}/test.npy", tokenizer)
    elif dataset_name == "mnist":
        tokenizer = MnistTokenizer()
        dataset_path = f"./data/mnist/"
        train_ds = MyDataset(f"{dataset_path}/train.npy", tokenizer)
        test_ds = MyDataset(f"{dataset_path}/test.npy", tokenizer)
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")
    return tokenizer, train_ds, test_ds

def get_vocab_size(dataset_name):
    if dataset_name == "tinystories":
        return 256
    elif dataset_name == "mnist":
        return 256
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")
