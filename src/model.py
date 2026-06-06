from x_transformers import Encoder, Decoder
from dataclasses import dataclass, field
import torch
from tqdm.auto import tqdm
import torch.nn.functional as F
from einops import rearrange, repeat
import torch.nn as nn

from dataclasses import dataclass

import torch
import torch.nn as nn
from einops import rearrange, repeat
from x_transformers import Encoder, Decoder

import torch.nn.functional as F
import torch

from einops import rearrange

def calc_cross_entropy_loss(logits, y):
    """
    Calculate the cross entropy loss between the logits and the target.
    logits: (..., V)
    y: (..., )
    Returns: (..., )
    """
    shape = y.shape
    logits, y = rearrange(logits, '... V -> (...) V'), rearrange(y, '... -> (...)')
    return F.cross_entropy(logits, y, reduction='none').reshape(shape)

def calc_mse_loss(y_pred, y):
    """
    Calculate the mean squared error loss between the predicted and target.
    y_pred: (..., )
    y: (..., )
    Returns: (..., )
    """
    return F.mse_loss(y_pred, y, reduction='none')

def calc_uniformity_loss(z, t=2, sample=512):
    """
    Calculate the uniformity loss between the memory tokens.
    z: (..., D)
    t: float
    sample: int (number of samples to draw for computing the uniformity loss)
    Returns: float
    """
    z = z / (z.norm(dim=-1, keepdim=True) + torch.finfo(z.dtype).eps)
    z = rearrange(z, "... D -> (...) D")
    idx = torch.randperm(len(z))[:sample]
    z = z[idx]
    return torch.pdist(z, p=2).pow(2).mul(-t).exp().mean().log()

def masked_mean(loss, mask):
    """
    Calculate the masked mean of the loss tensor.
    """
    return (loss * mask).sum() / mask.sum()

def masked_cross_entropy_loss(logits, y):
    """
    Masked cross entropy loss that ignores places where y==-1.
    """
    loss_ = calc_cross_entropy_loss(logits, y.clamp(min=0))
    loss_[y==-1] = 0.
    loss = masked_mean(loss_, y!=-1)
    return loss, loss_


def finite_scalar_quantize(a, k=1.0):
    """
    Efficient vector quantization with finite scalar quantization.
    This is NOT used by default.
    """
    quantized = torch.round(a / k) * k
    return a + (quantized - a).detach()

def normalize_mem(z, rms_norm=True, fsq=None):
    """
    Normalize the memory tokens with RMS norm and (optional) FSQ.
    """
    if rms_norm:
        z = torch.nn.functional.rms_norm(z, z.shape[-1:])
    if fsq is not None:
        z = finite_scalar_quantize(z, k=fsq)
    return z

@dataclass
class TeacherConfig:
    vocab_size: int = 256 # vocabulary size
    mem_size: int = 256 # the total memory state size, Z
    mem_tokens: int = 1 # number of memory tokens to factorize memory state into, K
    mem_rms_norm: bool | None = True # whether to do rms norm on the memory vector
    mem_fsq: float | None = None # whether to do vector quantization (finite scalar quantization) on the memory. FSQ is NOT used by default. If you don't know what this is, don't worry about it.

    d_model: int = 256 # the model dimension for encoder and decoder
    encoder_depth: int = 8 # the depth of the encoder
    decoder_depth: int = 4 # the depth of the decoder
    n_heads: int = 8 # the number of attention heads
    rotary_pos_emb: bool | None = True # whether to use rotary position embedding. Keep it True.
    use_rmsnorm: bool | None = True # whether to use rms norm within Transformer blocks. Keep it True.


class Teacher(nn.Module):
    """
    Teacher model consists of an Transformer encoder and decoder.
    """
    def __init__(self, cfg: TeacherConfig):
        super().__init__()
        self.cfg = cfg
        assert cfg.mem_size % cfg.mem_tokens == 0
        assert cfg.d_model * cfg.mem_tokens == cfg.mem_size

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model) # embedding for tokens
        self.encoder = Encoder(dim=cfg.d_model, depth=cfg.encoder_depth, heads=cfg.n_heads, rotary_pos_emb=cfg.rotary_pos_emb, use_rmsnorm=cfg.use_rmsnorm) # encoder

        self.decoder = Decoder(dim=cfg.d_model, depth=cfg.decoder_depth, heads=cfg.n_heads, rotary_pos_emb=cfg.rotary_pos_emb, use_rmsnorm=cfg.use_rmsnorm) # decoder
        self.to_logits = nn.Linear(cfg.d_model, cfg.vocab_size) # linear layer to project decoder output to logits

        self.encoder_mem_register = nn.Embedding(cfg.mem_tokens, cfg.d_model) # the input register tokens for memory tokens for the encoder


    def encode(self, x_ctx): # (B Tc)
        """
        Encode the past context tokens into a memory state, z.

        x_ctx: (B, Tc) tensor of token indices.
        Returns: (B, Z) tensor of memory states.
        """
        B, Tc = x_ctx.shape

        mask_x = (x_ctx != -1) # (B, Tc) boolean mask to ignore padding tokens
        x_ctx = x_ctx.clamp(min=0) # convert padding tokens to 0
        mask_mem = torch.ones((B, self.cfg.mem_tokens), dtype=torch.bool, device=mask_x.device)
        mask = torch.cat([mask_x, mask_mem], dim=1) # (B, Tc+K)
        x_emb = self.embed(x_ctx)
        mem_emb = self.encoder_mem_register(torch.arange(self.cfg.mem_tokens, device=x_ctx.device))
        mem_emb = repeat(mem_emb, "K D -> B K D", B=B)
        encoder_input = torch.cat([x_emb, mem_emb], dim=1) # (B, Tc+K, D)
        z = self.encoder(encoder_input, mask=mask) # (B, Tc+K, D)
        z = z[:, -self.cfg.mem_tokens:, :] # (B, K, D)

        z = normalize_mem(z, rms_norm=self.cfg.mem_rms_norm, fsq=self.cfg.mem_fsq)
        z = rearrange(z, "B K d -> B (K d)") # (B Z)
        return z

    def decode(self, z, x_fut): # (B Z), (B Tf)
        """
        Decode the memory state z into logits for the future tokens.

        z: (B, Z) tensor of memory states.
        x_fut: (B, Tf) tensor of future input tokens.
        Returns: (B, Tf, V) tensor of logits for the future output tokens.
        """
        z = rearrange(z, "B (K d) -> B K d", K=self.cfg.mem_tokens)
        x_fut = x_fut.clamp(min=0)
        logits = self.to_logits(self.decoder(torch.cat([z, self.embed(x_fut)], dim=1)))[:, self.cfg.mem_tokens-1:-1, :]
        return logits # (B, Tf, V)

    def teacher_readout(self, z): # (B Z)
        """
        Get logits for only the next token prediction for a memory state, using the decoder.
        """
        z = rearrange(z, "B (K d) -> B K d", K=self.cfg.mem_tokens)
        logits = self.to_logits(self.decoder(z)[:, -1, :]) # (B, V)
        return logits # (B, V)

@dataclass
class RNNConfig:
    vocab_size: int = 256 # vocabulary size
    mem_size: int = 256 # the total memory state size, Z
    mem_tokens: int = 1 # number of memory tokens to factorize memory state into, K
    mem_rms_norm: bool | None = True # whether to do rms norm on the memory vector
    mem_fsq: float | None = None # whether to do vector quantization (finite scalar quantization) on the memory. FSQ is NOT used by default. If you don't know what this is, don't worry about it.

    d_model: int = 256 # the model dimension for RNN's Transformer blocks
    rnn_depth: int = 8 # the depth of the RNN
    decoder_depth: int = 4 # the depth of the decoder
    n_heads: int = 8 # the number of attention heads
    rotary_pos_emb: bool | None = True # whether to use rotary position embedding. Keep it True.
    use_rmsnorm: bool | None = True # whether to use rms norm within Transformer blocks. Keep it True.

class RNN(nn.Module):
    """
    RNN model consists of a Transformer that maps a memory state (list of tokens) to the memory state of the next timestep (also a list of tokens).
    """
    def __init__(self, cfg: RNNConfig):
        super().__init__()
        self.cfg = cfg
        assert cfg.mem_size % cfg.mem_tokens == 0
        assert cfg.d_model * cfg.mem_tokens == cfg.mem_size

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model) # embedding for tokens
        self.rnn = Encoder(dim=cfg.d_model, depth=cfg.rnn_depth, heads=cfg.n_heads, rotary_pos_emb=cfg.rotary_pos_emb, use_rmsnorm=cfg.use_rmsnorm) # RNN's Transformer blocks

        self.decoder = Decoder(dim=cfg.d_model, depth=cfg.decoder_depth, heads=cfg.n_heads, rotary_pos_emb=cfg.rotary_pos_emb, use_rmsnorm=cfg.use_rmsnorm) # decoder
        self.to_logits = nn.Linear(cfg.d_model, cfg.vocab_size) # linear layer to project decoder output to logits
    
    def rnn_update(self, z, x): # (B Z) and (B 1)
        """
        Update the memory state z with the next token x.

        z: (B, Z) tensor of memory states.
        x: (B, 1) tensor of next token.
        Returns: (B, Z) tensor of updated memory states.
        """
        z = rearrange(z, "B (K d) -> B K d", K=self.cfg.mem_tokens)
        z_res = z
        inputs = torch.cat([z, self.embed(x)], dim=1) # (B, K+T, D)
        z = self.rnn(inputs) # (B, K+T, D)
        z_next = z[:, :self.cfg.mem_tokens, :] # (B, K, D)
        z_next = normalize_mem(z_next, rms_norm=self.cfg.mem_rms_norm, fsq=self.cfg.mem_fsq) # (B Z)
        z_next = rearrange(z_next, "B K d -> B (K d)") # (B Z)
        return z_next

    def rnn_readout(self, z): # (B, Z)
        """
        Read out logits for the next token from the memory state z.

        z: (B, Z) tensor of memory states.
        Returns: (B, V) tensor of logits for the next token.
        """
        z = rearrange(z, "B (K d) -> B K d", K=self.cfg.mem_tokens)
        logits = self.to_logits(self.decoder(z)[:, -1, :]) # (B, V)
        return logits # (B, V)


@dataclass
class BPTTConfig:
    T: int = 64 # sequence length

def forward_bptt(rnn, batch, cfg: BPTTConfig, pbar=False):
    """
    Forward pass for running BPTT .
    """
    x, y = batch['x'], batch['y']
    B, T = x.shape
    assert T==cfg.T
    m_rnn, logits = [], []
    m_t = torch.zeros((B, rnn.cfg.mem_size), device=x.device) # (B Z)
    for t in tqdm(range(cfg.T), disable=not pbar):
        # update memory with input
        m_t = rnn.rnn_update(m_t, x[:, [t]])
        m_rnn.append(m_t)
        # read out logits
        logits_t = rnn.rnn_readout(m_t) # (B, V)
        logits.append(logits_t)
    m_rnn = torch.stack(m_rnn, dim=1) # (B, T+1, Z)
    logits = torch.stack(logits, dim=1) # (B, T, V)
    loss_readout, loss_readout_ = masked_cross_entropy_loss(logits, y)
    loss = loss_readout
    return dict(x=x, y=y, m_rnn=m_rnn, logits=logits, loss_readout_=loss_readout_, loss_readout=loss_readout, loss=loss)

def get_smt_batch(batch, Tc, Tf):
    f"""
    Create a batch for the SMT forward pass.
    batch: dict with keys x, y
        x: (B, T)
        y: (B, T)
    Tc: int (context length)
    Tf: int (future length)
    Returns: dict with keys x_ctx, x_fut, y_fut
        x_ctx: (B, Tc) representing context for m_0, ..., m_{{T-1}}. for m_3 it is [x_0, x_1, x_2, x_3]
        x_fut: (B, Tf) representing future for m_0, ..., m_{{T-1}}. for m_3 it is [x_4, x_5, x_6, x_7]
        y_fut: (B, Tf) representing future for m_0, ..., m_{{T-1}}. for m_3 it is [y_3, y_4, y_5, y_6]
        x_ctx_next: (B, Tc) representing next context for m_0, ..., m_{{T-1}}. for m_3 it is [x_1, x_2, x_3, x_4]
    """
    x, y = batch['x'], batch['y']
    B, T = x.shape
    assert T >= Tc + Tf
    x_ctx = x[:, :Tc]
    x_fut = x[:, Tc:Tc+Tf]
    y_fut = y[:, Tc-1:Tc+Tf-1]
    x_ctx_next = x[:, 1:Tc+1]
    return dict(x_ctx=x_ctx, x_fut=x_fut, y_fut=y_fut, x_ctx_next=x_ctx_next)

@dataclass
class SMTConfig:
    T: int = 256+64 # total sequence length
    Tc: int = 256 # context length
    Tf: int = 64 # future length
    
    gamma: float = 1. # decay factor for the predictive state loss. leave it as 1.
    coef_dec: float = 1. # coefficient for the decoder predictive state loss.
    coef_dyn: float = 0.1 # coefficient for the dynamics loss.
    coef_unif: float = 0.001 # coefficient for the uniformity loss.
    unif_over_tokens: bool | None = True # if True, loss_unif is computed over memory tokens instead of entire memory state. leave it as True.

    detach_rnn: bool | None = False # if True, the gradients from the dynamics loss are not fed to the teacher.
    hot_rnn: bool | None = True # if True, the RNN's parameters are optimized. leave it as True
    hot_teacher: bool | None = True # if True, the teacher's parameters are optimized. leave it as True

def forward_smt(rnn, teacher, batch, cfg: SMTConfig):
    """
    Forward pass for SMT!
    """
    x, y = batch['x'], batch['y']
    B, T = x.shape
    assert x.shape[1] == cfg.T and y.shape[1] == cfg.T

    sb = get_smt_batch(batch, cfg.Tc, cfg.Tf) # get the batch for the SMT forward pass
    x_ctx, x_fut, y_fut, x_ctx_next = sb['x_ctx'], sb['x_fut'], sb['y_fut'], sb['x_ctx_next'] # get the context, future, and next context for the SMT forward pass

    m_now = teacher.encode(x_ctx) # (B, Z) embed context
    m_next = teacher.encode(x_ctx_next) # (B, Z) embed next context
    x_in = x_fut[:, [0]] # (B, 1)
    m = m_now # (B, Z)
    
    # predictive state loss
    logits = teacher.decode(m, x_fut) # (B, Tf, V)
    loss_decoder_ = calc_cross_entropy_loss(logits, y_fut.clamp(min=0)) # (B, Tf)
    loss_decoder_decayed = loss_decoder_ * (cfg.gamma ** torch.arange(cfg.Tf, device=x_ctx.device))
    loss_decoder = masked_mean(loss_decoder_decayed, y_fut != -1)
    loss_decoder0 = masked_mean(loss_decoder_[:, 0], y_fut[:, 0] != -1) # this is purely for tracking purposes, it tells you expected performance of the RNN.
    
    if cfg.unif_over_tokens:
        loss_unif = calc_uniformity_loss(rearrange(m, "B (K d) -> B K d", K=teacher.cfg.mem_tokens))
    else:
        loss_unif = calc_uniformity_loss(m)
    
    m_next_pred = rnn.rnn_update((m_now.detach() if cfg.detach_rnn else m_now), x_in) # (B, Z) # one step prediction of the RNN
    loss_dyn_ = calc_mse_loss(m_next_pred, (m_next.detach() if cfg.detach_rnn else m_next)) # dynamics loss
    loss_dyn = loss_dyn_.mean()
    
    loss = cfg.coef_dec*loss_decoder + cfg.coef_dyn*loss_dyn + cfg.coef_unif*loss_unif # total loss

    return dict(x_ctx=x_ctx, x_fut=x_fut, y_fut=y_fut, x_in=x_in,
                m_now=m_now, m_next=m_next, m_next_pred=m_next_pred, logits=logits,
                loss_decoder_=loss_decoder_, loss_decoder=loss_decoder, loss_decoder0=loss_decoder0,
                loss_dyn_=loss_dyn_, loss_dyn=loss_dyn, loss_unif=loss_unif, loss=loss)

@dataclass
class DMTConfig:
    T: int = 256+64+64 # sequence length
    Td: int = 64 # number of RNN rollout steps for DMT
    Tc: int = 256 # context length
    Tf: int = 64 # future length
    k: int = 1 # subsampling factor for DMT rollout. If Td=16 and k=4, then loss is only calculated at t=0, 4, 8, 12.

    coef_dec: float = 1. # coefficient for the decoder predictive state loss.
    coef_dyn: float = 10. # coefficient for the dynamics loss.
    coef_unif: float = 1e-2 # coefficient for the uniformity loss.
    coef_readout: float = 1. # coefficient for the readout loss.

    detach_readout: bool | None = True # if True, the gradients from the readout loss are not propagated to the RNN. Leave it as True if you don't want BPTT to do credit assignment.
    detach_rnn: bool | None = True # if True, gradients cannot propagate end-to-end through RNN dynamics during RNN rollout.

    hot_rnn: bool | None = True
    hot_teacher_encoder: bool | None = False # leave it as False, keep the encoder frozen during DMT.
    hot_teacher_decoder: bool | None = False # leave it as False, keep the decoder frozen during DMT.

def forward_dmt(rnn, teacher, batch, cfg: DMTConfig, pbar=False):
    x, y = batch['x'], batch['y']
    B, T = x.shape
    assert T == cfg.T
    assert T >= cfg.Tc + cfg.Td + cfg.Tf

    # Run the encoder to get the oracle memory states
    loss_decoder = []
    z_enc = []
    for t in tqdm(range(0, cfg.Td, cfg.k), disable=not pbar):
        x_ctx, x_fut = x[:, t:t+cfg.Tc], x[:, t+cfg.Tc:t+cfg.Tc+cfg.Tf]
        y_fut = y[:, t+cfg.Tc-1:t+cfg.Tc+cfg.Tf-1]
        z = teacher.encode(x_ctx)
        z_enc.append(z)

        logits = teacher.decode(z, x_fut) # (B, Tf, V)
        loss_decoder_t = calc_cross_entropy_loss(logits, y_fut) # (B, Tf)
        loss_decoder.append(loss_decoder_t)
    z_enc = torch.stack(z_enc, dim=1) # (B, Td, Z, D)
    loss_decoder_ = torch.stack(loss_decoder, dim=1) # (B, Td, Tf)
    loss_decoder = loss_decoder_.mean()
    loss_decoder0 = loss_decoder_[:, :, 0].mean()

    # Run the RNN to get the on-policy generated memory states
    loss_readout = []
    z_rnn = []
    z_t = z_enc[:, 0]
    for t in tqdm(range(cfg.Td), disable=not pbar):
        z_rnn.append(z_t)

        logits = rnn.rnn_readout(z_t.detach() if cfg.detach_readout else z_t) # (B, V)
        loss_readout_t = calc_cross_entropy_loss(logits, y[:, t+cfg.Tc-1]) # (B,)
        loss_readout.append(loss_readout_t)

        z_t = rnn.rnn_update((z_t.detach() if not cfg.detach_rnn else z_t), x[:, [cfg.Tc+t]])
    z_rnn = torch.stack(z_rnn, dim=1)[:, ::cfg.k] # (B T Z)
    loss_readout_ = torch.stack(loss_readout, dim=1) # (B T)
    loss_readout = loss_readout_.mean()

    loss_dyn_ = calc_mse_loss(z_rnn, z_enc) # (B T Z)
    loss_dyn = loss_dyn_.mean()

    loss_unif = calc_uniformity_loss(z_enc)

    loss = cfg.coef_dec*loss_decoder + cfg.coef_dyn*loss_dyn + cfg.coef_unif*loss_unif + cfg.coef_readout*loss_readout
    
    return dict(x=x, z_rnn=z_rnn, z_enc=z_enc, #, logits=logits,
                loss_readout_=loss_readout_, loss_readout=loss_readout,
                loss_decoder_=loss_decoder_, loss_decoder=loss_decoder, loss_decoder0=loss_decoder0,
                loss_dyn_=loss_dyn_, loss_dyn=loss_dyn, loss_unif=loss_unif, loss=loss)

def encode_with_rnn(rnn, x, pbar=False):
    """
    Encode sequence x with an RNN.
    """
    B, T = x.shape
    z_t = torch.zeros((B, rnn.cfg.mem_size), device=x.device)
    for t in tqdm(range(T), disable=not pbar):
        z_t = rnn.rnn_update(z_t, x[:, [t]])
    return z_t

@torch.no_grad()
def read_with_teacher_rnn(rnn, teacher, x, read_len, eval_len, pbar=False):
    """
    Read the sequence x with the teacher and the RNN.
    The teacher will encode the first read_len tokens into an initial memory state, then the RNN will be rolled out to read the next eval_len tokens.

    "Reading" means the RNN is unrolled, but the input/output tokens are teacher forced (not autoregressively generated).
    """
    x_read, x_eval = x[:, :read_len], x[:, read_len:read_len+eval_len]
    z = teacher.encode(x_read) if teacher is not None else encode_with_rnn(rnn, x_read)
    logits = []
    for t in tqdm(range(eval_len), disable=not pbar):
        logits.append(rnn.rnn_readout(z)) # (B, V)
        z = rnn.rnn_update(z, x_eval[:, [t]])
    logits = torch.stack(logits, dim=1) # (B, eval_len, V)
    loss_eval_ = calc_cross_entropy_loss(logits, x_eval) # (B, eval_len, V)
    loss_eval = loss_eval_.mean()
    x_all = torch.cat([x_read, x_eval], dim=1)
    return dict(x_read=x_read, x_eval=x_eval, x_all=x_all, logits=logits,
                loss_eval_=loss_eval_, loss_eval=loss_eval)

@torch.no_grad()
def write_with_teacher_rnn(rnn, teacher, x, read_len, write_len, temperature=1.0, pbar=False):
    """
    Generate a sequence with the teacher and the RNN.
    The teacher will encode the first read_len tokens into an initial memory state, then the RNN will be rolled out to generate the next write_len tokens.

    "Writing" means the RNN is unrolled, but the input/output tokens are autoregressively generated.
    """
    x_read = x[:, :read_len]
    z = teacher.encode(x_read) if teacher is not None else encode_with_rnn(rnn, x_read)
    x_write = []
    for _ in tqdm(range(write_len), disable=not pbar):
        logits = rnn.rnn_readout(z) # (B, V)
        probs = F.softmax(logits / temperature, dim=-1) # (B, V)
        token = torch.multinomial(probs, num_samples=1) # (B, 1)
        x_write.append(token)
        z = rnn.rnn_update(z, token)
    x_write = torch.cat(x_write, dim=1)
    x_all = torch.cat([x_read, x_write], dim=1)
    return dict(x_read=x_read, x_write=x_write, x_all=x_all)

@torch.no_grad()
def write_with_teacher(teacher, x, read_len, write_len, temperature=1.0, pbar=False):
    """
    Generate a sequence with the teacher.
    """
    x_read = x[:, :read_len]
    z = teacher.encode(x_read)
    x_write = []
    for _ in tqdm(range(write_len), disable=not pbar):
        logits = teacher.teacher_readout(z) # (B, V)
        probs = F.softmax(logits / temperature, dim=-1) # (B, V)
        token = torch.multinomial(probs, num_samples=1) # (B, 1)
        x_write.append(token)
        x_ctx = torch.cat([x_read, *x_write], dim=1)[:, -read_len:]
        z = teacher.encode(x_ctx)
    x_write = torch.cat(x_write, dim=1)
    x_all = torch.cat([x_read, x_write], dim=1)
    return dict(x_read=x_read, x_write=x_write, x_all=x_all)

@torch.no_grad()
def read_with_rnn(rnn, x, read_len, eval_len, pbar=False):
    """
    Read the sequence x with only the RNN.

    This won't work out of the box because the RNN is only trained to start from an encoder-generated memory state during DMT. This can be modified to work.
    """
    return read_with_teacher_rnn(rnn, None, x, read_len, eval_len, pbar)

@torch.no_grad()
def write_with_rnn(rnn, x, read_len, write_len, temperature=1.0, pbar=False):
    """
    Generate a sequence with only the RNN.

    This won't work out of the box because the RNN is only trained to start from an encoder-generated memory state during DMT. This can be modified to work.
    """
    return write_with_teacher_rnn(rnn, None, x, read_len, write_len, temperature, pbar)
