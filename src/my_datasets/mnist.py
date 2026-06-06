import os
import numpy as np
import torchvision
from einops import rearrange

class MnistTokenizer:
    def __init__(self):
        self.vocab_size = 256

    def encode(self, imgs):
        x = rearrange(imgs, "... H W -> ... (H W)")
        return x

    def decode(self, x):
        x = rearrange(x, "... (H W) -> ... H W", W=28)
        imgs = x
        return imgs.astype(np.uint8)

def prepare_dataset(dataset_path):
    tokenizer = MnistTokenizer()

    train_ds = torchvision.datasets.MNIST(root=dataset_path, train=True, download=True)
    test_ds = torchvision.datasets.MNIST(root=dataset_path, train=False, download=True)

    train_imgs = np.stack([img for img, _ in train_ds])
    test_imgs = np.stack([img for img, _ in test_ds])
    x_train = tokenizer.encode(train_imgs)
    x_test = tokenizer.encode(test_imgs)
    x_train = rearrange(x_train, "B T -> (B T)")
    x_test = rearrange(x_test, "B T -> (B T)")

    os.makedirs(dataset_path, exist_ok=True)
    np.save(f"{dataset_path}/train.npy", x_train)
    np.save(f"{dataset_path}/test.npy", x_test)

if __name__ == "__main__":
    prepare_dataset(dataset_path="./data/mnist/")
