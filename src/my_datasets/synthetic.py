from torch.utils.data import Dataset
import numpy as np
import torch
from einops import rearrange

"""
Synthetic datasets from the paper.

These won't run out of the box because we use a modified version of the SMT forward pass so that we can train all memories within a sequence.
This code will be published soon.
"""

class ParityDataset(Dataset):
    def __init__(self, ):
        self.vocab_size = 2

    def __len__(self, ctx_len: int):
        raise None

    def __getitem__(self, idx: int, ctx_len: int):
        return None
    
    def sample_batch(self, batch_size: int, ctx_len: int):
        x = torch.randint(0, 2, (batch_size, ctx_len))
        y = x.cumsum(dim=1)%2
        return dict(x=x, y=y)

class StackDataset(Dataset):
    def __init__(self, n_elements: int=4, max_stack_size: int = 10):
        self.n_elements = n_elements
        self.max_stack_size = max_stack_size
        self.vocab_size = n_elements + 1

    def __len__(self, ctx_len: int):
        raise None

    def __getitem__(self, idx: int, ctx_len: int):
        return None
    
    def sample_batch(self, batch_size: int, ctx_len: int):
        x = torch.zeros(batch_size, ctx_len, dtype=torch.long)
        y = torch.zeros(batch_size, ctx_len, dtype=torch.long)
        for b in range(batch_size):
            stack = []
            for t in range(ctx_len):
                can_push = len(stack) < self.max_stack_size
                can_pop = len(stack) > 0
                if can_push and can_pop:
                    if np.random.rand() < 0.5: should_push = True
                    else: should_push = False
                elif can_push: should_push = True
                elif can_pop: should_push = False

                rand_element = np.random.randint(self.n_elements) + 1
                if should_push:
                    stack.append(rand_element)
                    act = rand_element
                    obs = -1
                else:
                    act = 0
                    obs = stack.pop()
                x[b, t] = act
                y[b, t] = obs
        return dict(x=x, y=y)

class StringCopyDataset(Dataset):
    def __init__(self, n_elements:int, noise=0.0, reverse=False):
        self.n_elements = n_elements
        self.vocab_size = n_elements + 1
        self.noise = noise
        self.reverse = True

    def __len__(self, ctx_len: int):
        raise None

    def __getitem__(self, idx: int, ctx_len: int):
        return None
    
    def sample_batch(self, batch_size: int, ctx_len: int):
        x = torch.randint(0, self.n_elements, (batch_size, ctx_len)) + 1
        strlen = ctx_len//2
        s = x[:, :strlen]
        if self.reverse:
            s = s.flip(dims=(1,))
        x[:, strlen] = 0
        x[:, strlen+1:] = s[:, :-1]
        y = torch.zeros_like(x) - 1
        y[:, strlen:] = s
        p = torch.rand(size=(batch_size, strlen))
        corrupt = p < self.noise
        yrand = torch.randint(0, self.n_elements, (batch_size, strlen)) + 1
        y[:, strlen:] = torch.where(corrupt, yrand, y[:, strlen:])
        return dict(x=x, y=y)

class NeedleDataset(Dataset):
    def __init__(self, n_elements, min_pos=0., max_pos=0.5, noise=0.):
        self.n_elements = n_elements
        self.vocab_size = n_elements + 1
        self.min_pos = min_pos
        self.max_pos = max_pos
        self.noise = noise

    def __len__(self, ctx_len: int):
        raise None

    def __getitem__(self, idx: int, ctx_len: int):
        return None
    
    def sample_batch(self, batch_size: int, ctx_len: int):
        x = torch.randint(0, self.n_elements, (batch_size, ctx_len)) + 1

        p = torch.rand((batch_size,)) * (self.max_pos-self.min_pos) + self.min_pos
        needle_pos = (ctx_len * p).to(torch.long).clamp(min=1, max=ctx_len-2)
        
        id_pos = needle_pos - 1
        needle = torch.randint(0, self.n_elements, (batch_size, )) + 1
        x[torch.arange(batch_size), id_pos] = 0
        x[torch.arange(batch_size), needle_pos] = needle

        noisy_needle = torch.randint(0, self.n_elements, (batch_size, )) + 1
        noise = torch.rand((batch_size,))
        needle = torch.where(noise<self.noise, noisy_needle, needle)

        x[:, -1] = 0
        y = torch.zeros_like(x)-1
        y[torch.arange(batch_size), torch.zeros_like(needle)-1] = needle
        return dict(x=x, y=y)

class RecencyPrimacyBiasDataset(Dataset):
    def __init__(self, vocab_size: int, gamma: float = 0.5, primacy = False, second_half: bool = True):
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.primacy = primacy
        self.second_half = second_half
    def __len__(self, ctx_len: int):
        raise None

    def __getitem__(self, idx: int, ctx_len: int):
        return None
    
    def sample_batch(self, batch_size: int, ctx_len: int):
        x = torch.randint(0, self.vocab_size, (batch_size, ctx_len))
        y = torch.zeros_like(x)
        for t in range(ctx_len):
            if self.primacy:
                p = self.gamma ** np.arange(t+1)
            else:
                p = self.gamma ** (np.arange(t+1)[::-1])
            p = p / p.sum()

            idx = np.random.choice(np.arange(t+1), p=p, size=(batch_size,))
            yt = x[torch.arange(batch_size), idx]
            y[:, t] = yt
        if self.second_half:
            # y[:, ctx_len//2:] = -1
            y[:, int(ctx_len*.9):] = -1
        return dict(x=x, y=y)
    
class ICLDataset(Dataset):
    def __init__(self, vocab_size, difficulty=0.2, noise=0.):
        self.vocab_size = vocab_size
        self.d = int(difficulty * vocab_size)
        assert noise==0., "not implemented yet"

    def __len__(self, ctx_len: int):
        raise None

    def __getitem__(self, idx: int, ctx_len: int):
        return None
    
    def sample_batch(self, batch_size: int, ctx_len: int):
        w = torch.randint(0, self.d, (batch_size, 1))
        b = torch.randint(0, self.d, (batch_size, 1))
        noise = torch.randint(0, 1, (batch_size, ctx_len//2))
        xin = torch.randint(0, self.vocab_size, (batch_size, ctx_len//2))
        yout = (w * xin + b + noise) % self.vocab_size
        x = rearrange(torch.stack([xin, yout], dim=-1), "B T two -> B (T two)")
        y = rearrange(torch.stack([yout, torch.zeros_like(yout)-1], dim=-1), "B T two -> B (T two)")
        return dict(x=x, y=y)

class AssociativeRecallDataset(Dataset):
    def __init__(self, V_key, V_val, key_len, n_keys):
        self.V_key = V_key
        self.V_val = V_val
        self.vocab_size = V_key + V_val
        self.key_len = key_len
        self.n_keys = n_keys

    def __len__(self, ctx_len: int):
        raise None

    def __getitem__(self, idx: int, ctx_len: int):
        return None
    
    def sample_batch(self, batch_size: int, ctx_len: int):
        assert ctx_len >= self.key_len * self.n_keys * 2
        keys = torch.randint(0, self.V_key, (batch_size, self.n_keys-1, self.key_len))
        vals = torch.randint(0, self.V_val, (batch_size, self.n_keys-1, self.key_len))+self.V_key
        # pad = torch.zeros((batch_size, self.n_keys))
        idx = torch.randint(0, self.n_keys-1, (batch_size, ))
        query = keys[torch.arange(batch_size), idx]
        queryval = vals[torch.arange(batch_size), idx]

        x = rearrange(torch.stack([keys, vals], dim=-1), "B N K two -> B (N two K)")
        x = torch.cat([x, query, queryval], dim=-1)
        y = torch.zeros_like(x)-1
        y[:, -self.key_len-1:-1] = queryval
        return dict(x=x, y=y)
