import torch
import torch.nn.functional as F
import torch.nn as nn
import time
import math

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group

import os
from dataclasses import dataclass

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.c_fc = nn.Linear(config.dim_size, 4 * config.dim_size)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*config.dim_size, config.dim_size)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):

        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        
        return x

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.c_attn = nn.Linear(config.dim_size, 3 * config.dim_size)
        self.c_proj = nn.Linear(config.dim_size, config.dim_size)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.n_head = config.num_heads

        self.register_buffer('bias', torch.tril(torch.ones(1, 1, config.seq_len, config.seq_len)))


    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.shape

        qkv = self.c_attn(x) # [B, T, 3D]
        q, k, v = qkv.split(D, dim=-1) # [B, T, D] each

        q = q.view(B, T, self.n_head, D // self.n_head).transpose(1, 2) # [B, H, T, d] (d = D // H)
        k = k.view(B, T, self.n_head, D // self.n_head).transpose(1, 2) # [B, H, T, d] (d = D // H)
        v = v.view(B, T, self.n_head, D // self.n_head).transpose(1, 2) # [B, H, T, d] (d = D // H)

        # attn = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))
        # attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # attn_scores = F.softmax(attn, dim=-1)
        # out = attn_scores @ v # [B, H, T, d]

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        out = out.transpose(1, 2).contiguous().view(B, T, D)

        out = self.c_proj(out)

        return out


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.dim_size)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.dim_size)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    

@dataclass  
class GPTConfig:
    vocab_size : int = 50257
    seq_len : int = 1024
    dim_size : int = 768
    num_blocks : int = 12
    num_heads : int = 12


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.transformer = nn.ModuleDict({
            'wte' : nn.Embedding(config.vocab_size, config.dim_size),
            'wpe' : nn.Embedding(config.seq_len, config.dim_size),
            'h' : nn.ModuleList([Block(config) for _ in range(config.num_blocks)]),
            'ln_f' : nn.LayerNorm(config.dim_size),
        })

        self.lm_head = nn.Linear(config.dim_size, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.num_blocks) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, inp, targets=None):

        B, T = inp.size()

        pos = torch.arange(0, T, dtype = torch.long, device = inp.device)
        pos_emb = self.transformer.wpe(pos)
        token_emb = self.transformer.wte(inp)

        x = pos_emb + token_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) 

        loss = None

        if targets is not None:
            # logits: BxTxV
            # targets: BXT
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(num_blocks=12, num_heads=12, dim_size=768),  # 124M params
            'gpt2-medium':  dict(num_blocks=24, num_heads=16, dim_size=1024), # 350M params
            'gpt2-large':   dict(num_blocks=36, num_heads=20, dim_size=1280), # 774M params
            'gpt2-xl':      dict(num_blocks=48, num_heads=25, dim_size=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['seq_len'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    def configure_optimizer(self, weight_decay, learning_rate, device):
        
        param_group = {np:p for np, p in self.named_parameters()}
        param_group = {np:p for np, p in param_group.items() if p.requires_grad}

        param_decay = [p for np, p in param_group.items() if p.dim() >= 2]
        param_nodecay = [p for np, p in param_group.items() if p.dim() < 2]

        optimizer_param_groups = [
            {"params": param_decay, 'weight_decay': weight_decay},
            {"params": param_nodecay, 'weight_decay': 0.0}
        ]

        num_decay_params = sum(p.numel() for p in param_decay)
        num_nodecay_params = sum(p.numel() for p in param_nodecay)

        if master_process:
            print(f"# Decayed Param Tensors: {len(param_decay)}, Total Params: {num_decay_params:,}")
            print(f"# Non Decayed Param Tensors: {len(param_nodecay)}, Total Params: {num_nodecay_params:,}")

        optimizer = torch.optim.AdamW(optimizer_param_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=True)
        
        return optimizer
    
# ---------------------------------------------------------------------------------------------------------------------------------------
import tiktoken
import numpy as np

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int64)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.num_processes = num_processes
        self.process_rank = process_rank

        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")

        self.reset()

        # data = open('input.txt').read()
        # enc = tiktoken.get_encoding('gpt2')
        # self.tokens = torch.tensor(enc.encode(data))

        # self.current_position = self.B * self.T * self.process_rank

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        self.current_position += B*T*self.num_processes

        if self.current_position + (B*T*self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

# ---------------------------------------------------------------------------------------------------------------

device = "cuda"

# model = GPT.from_pretrained('gpt2')

ddp = int(os.environ.get('RANK', -1)) != -1

if ddp:
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # Master Process for logging etc.

else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = True

torch.manual_seed(1337)
torch.cuda.manual_seed(1337)

# Larger Batch Size using Gradient Accumulation
batch_size = 524288 # Num Tokens = 2^19
B = 64
T = 1024
num_grad_accu_steps = batch_size // (B * T * ddp_world_size)
train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

if master_process:
    print("Total Batch Size:", batch_size)
    print("Total grad accu iters in each step:", num_grad_accu_steps)

# Enabling tfloat32
torch.set_float32_matmul_precision("high")

model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
model = torch.compile(model)
if (ddp):
    model = DDP(model, device_ids=[ddp_local_rank])

raw_model = model.module if ddp else model

# print(model)
# for k, v in model.state_dict().items():
#     print(k, v.shape)
# exit()

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073

def get_lr(step):

    if step < warmup_steps:
        return (step + 1) * max_lr / warmup_steps
    
    if step > max_steps:
        return min_lr
    
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)
    
# optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), eps=10e-8)
optimizer = raw_model.configure_optimizer(weight_decay=0.1, learning_rate=6e-4, device=device)

for step in range(max_steps):
    t0 = time.time()

    # once in a while evaluate our validation loss
    if step % 100 == 0:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"Validation loss: {val_loss_accum.item():.4f}")

    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0

    for micro_step in range(num_grad_accu_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, loss = model(x, y)
            # import code; code.interact(local=locals()) # To Inspect the precision of variables
        loss /= num_grad_accu_steps
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_step == num_grad_accu_steps -1) # Disable gradient all reduce (sync) until grad accumulation's last step.
            # Wasteful to do grad sync at every micro_step. So collect all grads asynchronously for each full grad accu for each GPU process and then sync collected gradients at the end.
        loss.backward() # Syncs gradients (all reduce) if model.require_backward_grad_sync = True

    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr 

    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000
    tokens_per_sec = (train_loader.B * train_loader.T * num_grad_accu_steps * ddp_world_size) / (t1 - t0)
    if master_process:
        print(f"step {step} | loss : {loss_accum.item()} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f}ms | tok/sec: {tokens_per_sec:.2f}")

if ddp:
    destroy_process_group()

import sys; sys.exit(0)

model.eval()
num_return_sequences = 5
max_length = 30

import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)

x = tokens.to(device)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

while (x.size(-1) < max_length):

    logits = model(x)

    logits = logits[:, -1, :]
    probs = F.softmax(logits, dim=-1)
    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1) # 5x50

    ix = torch.multinomial(topk_probs, 1) # 5x1

    # print(topk_indices.shape, ix.shape)

    xcol = torch.gather(topk_indices, -1, ix)

    x = torch.cat((x, xcol), dim=1)


for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)


'''
# Get a data batch
import tiktoken
enc = tiktoken.get_encoding('gpt2')
data = open('input.txt', 'r').read()[:1000]
tokens = enc.encode(data)
B, T = 4, 32
buf = torch.tensor(tokens[:B*T + 1]).to(device)

x = buf[:-1].view(B, T)
y = buf[1:].view(B, T)
'''