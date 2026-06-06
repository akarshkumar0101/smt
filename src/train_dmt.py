import torch
from dataclasses import dataclass, field
import numpy as np
import tyro
from tqdm.auto import tqdm
import util
import pandas as pd
import time

from my_datasets import get_dataset
from model import RNNConfig, TeacherConfig, RNN, Teacher, DMTConfig, forward_dmt

@dataclass
class Config:
    seed: int = 0
    dataset: str = "tinystories"
    n_iters: int = 1500 # number of optimization iterations
    log_every: int = 10 # log every n iterations
    ckpt_every: int | None = None # save checkpoint every n iterations
    save_dir: str | None = None # save directory
    load_rnn_path: str | None = None # load RNN checkpoint
    load_teacher_path: str | None = None # load teacher checkpoint

    rnn: RNNConfig = field(default_factory=RNNConfig) # RNN configuration
    teacher: TeacherConfig = field(default_factory=TeacherConfig) # teacher configuration

    lr: float = 3e-4 # learning rate
    lr_schedule: str = "warmup_constant" # "warmup_constant" | "warmup_cosine"
    weight_decay: float = 1e-2 # weight decay
    adam_betas: tuple = (0.9, 0.95) # Adam beta1 and beta2
    batch_size: int = 32 # batch size
    clip_grad_norm: float = 1. # clip gradient norm

    dmt: DMTConfig = field(default_factory=DMTConfig) # DMT forward pass configuration

class Main:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def init(self):
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        self.amp = torch.amp.autocast("cuda", dtype=torch.bfloat16) # mixed precision training

        self.tokenizer, self.train_ds, self.test_ds = get_dataset(self.cfg.dataset) # get the dataset
        
        self.rnn = RNN(self.cfg.rnn).cuda() # initialize the RNN
        self.teacher = Teacher(self.cfg.teacher).cuda() # initialize the teacher
        if self.cfg.load_rnn_path is not None: # load the RNN checkpoint if provided
            self.rnn.load_state_dict(util.load_pkl_path(self.cfg.load_rnn_path).state_dict())
        if self.cfg.load_teacher_path is not None: # load the teacher checkpoint if provided
            self.teacher.load_state_dict(util.load_pkl_path(self.cfg.load_teacher_path).state_dict())

        parameters = []
        if self.cfg.dmt.hot_rnn: # figure out which parameters to optimize
            parameters += list(self.rnn.parameters())
        if self.cfg.dmt.hot_teacher_encoder:
            parameters += list(self.teacher.encoder.parameters())
            parameters += list(self.teacher.embed.parameters())
            parameters += list(self.teacher.encoder_z_register.parameters())
        if self.cfg.dmt.hot_teacher_decoder:
            parameters += list(self.teacher.decoder.parameters())
            parameters += list(self.teacher.to_logits.parameters())
        self.opt = torch.optim.AdamW(parameters, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay, betas=self.cfg.adam_betas)
        print(f"Number of parameters: {sum(p.numel() for p in self.rnn.parameters()):<10} in RNN")
        print(f"Number of parameters: {sum(p.numel() for p in self.teacher.parameters()):<10} in Teacher")
        print(f"Number of parameters: {sum(p.numel() for p in parameters):<10} being optimized")
        self.logs = []

    def step(self, itr=0):
        logs = dict(itr=itr)
        self.logs.append(logs)

        # TESTING STEP
        if itr % 10 == 0:
            self.rnn.eval(); self.teacher.eval()
            batch = self.test_ds.sample_batch(batch_size=self.cfg.batch_size, ctx_len=self.cfg.dmt.T) # sample test batch
            batch = {k: v.cuda().long() for k, v in batch.items()}
            with torch.no_grad(), self.amp:
                outputs = forward_dmt(self.rnn, self.teacher, batch, self.cfg.dmt) # forward pass with DMTs
            for k, v in outputs.items():
                if v.numel() == 1:
                    logs[f"{k}_test"] = v.item()
    
        # UPDATE LEARNING RATE
        lr = util.get_lr(itr, self.cfg.n_iters, self.cfg.lr, self.cfg.lr_schedule)
        for param_group in self.opt.param_groups:
            param_group['lr'] = lr
        logs['lr'] = lr
        
        # TRAINING STEP
        self.rnn.train(); self.teacher.train()
        self.opt.zero_grad(set_to_none=True)
        batch = self.train_ds.sample_batch(batch_size=self.cfg.batch_size, ctx_len=self.cfg.dmt.T) # sample training batch
        batch = {k: v.cuda().long() for k, v in batch.items()}
        with self.amp:
            outputs = forward_dmt(self.rnn, self.teacher, batch, self.cfg.dmt) # forward pass with DMT
        outputs['loss'].backward() # backward pass
        grad_norm = torch.nn.utils.clip_grad_norm_(self.rnn.parameters(), self.cfg.clip_grad_norm)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.teacher.parameters(), self.cfg.clip_grad_norm)
        logs['grad_norm'] = grad_norm.item()
        self.opt.step() # update parameters
        logs['time'] = time.time()

        for k, v in outputs.items():
            if v.numel() == 1:
                logs[k] = v.item()
        
    def run(self):
        pbar = tqdm(range(self.cfg.n_iters))
        for self.itr in pbar:
            self.step(self.itr) # perform a single optimization step
            
            pbar.set_postfix(loss=f"{self.logs[-1]['loss']:.4f}")
            if self.cfg.ckpt_every is not None and (self.itr+1) % self.cfg.ckpt_every == 0: # save checkpoint
                util.save_pkl(self.cfg.save_dir, f"rnn_{self.itr+1:07d}", self.rnn)
                util.save_pkl(self.cfg.save_dir, f"teacher_{self.itr+1:07d}", self.teacher)
            if (self.itr % self.cfg.log_every == 0 or self.itr==self.cfg.n_iters-1) and self.cfg.save_dir is not None:
                util.save_pkl(self.cfg.save_dir, "cfg", self.cfg) # save config, rnn, teacher, and logs
                util.save_pkl(self.cfg.save_dir, "rnn", self.rnn)
                util.save_pkl(self.cfg.save_dir, "teacher", self.teacher)
                util.save_pkl(self.cfg.save_dir, "logs", pd.DataFrame(self.logs))


if __name__ == "__main__":
    main = Main(tyro.cli(Config))
    main.init()
    main.run()
