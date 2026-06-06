import os
from datasets import load_dataset
import numpy as np

class TinyStoriesTokenizer:
    def __init__(self):
        self.vocab_size = 256

    def encode(self, txt):
        if isinstance(txt, str):
            x = np.frombuffer(txt.encode('ascii', errors="ignore"), dtype=np.uint8)
        elif isinstance(txt, list):
            x = [np.frombuffer(t.encode('ascii', errors="ignore"), dtype=np.uint8) for t in txt]
        return x

    def decode(self, x):
        assert isinstance(x, np.ndarray)
        x = x.astype(np.uint8)
        if x.ndim == 1:
            txt = x.tobytes().decode('ascii', errors="ignore")
        elif x.ndim == 2:
            txt = [t.tobytes().decode('ascii', errors="ignore") for t in x]
        return txt


def prepare_dataset(dataset_path):
    ds_train = load_dataset("roneneldan/TinyStories", split="train", streaming=False)
    ds_test = load_dataset("roneneldan/TinyStories", split="validation", streaming=False)

    tokenizer = TinyStoriesTokenizer()
    x_train = tokenizer.encode(list(ds_train['text']))
    x_test = tokenizer.encode(list(ds_test['text']))
    x_train = np.concatenate(x_train)
    x_test = np.concatenate(x_test)

    os.makedirs(dataset_path, exist_ok=True)
    np.save(f"{dataset_path}/train.npy", x_train)
    np.save(f"{dataset_path}/test.npy", x_test)

if __name__ == "__main__":
    # login(token=os.environ['HF_TOKEN'])
    prepare_dataset(dataset_path="./data/tinystories/")
