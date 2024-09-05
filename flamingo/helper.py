import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FeedForward(nn.Module):

    def __init__(self, dim, mult=4):
        super().__init__()
        inner_dim = dim * mult
        self.ff = nn.Sequential(nn.LayerNorm(dim),
                                nn.Linear(dim, inner_dim, bias=False),
                                nn.GELU(),
                                nn.Linear(inner_dim, dim, bias=False)
                                )

    def forward(self, x):
        return self.ff(x)


class PerceiverAttention(nn.Module):

    def __init__(self, dim, dim_head, heads):
        super().__init__()

        hidden_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.q = nn.Linear(dim, hidden_dim, bias=False)
        self.kv = nn.Linear(dim, hidden_dim*2, bias=False)
        self.out = nn.Linear(hidden_dim, dim, bias=False)

        self.heads = heads
        self.dim_head = dim_head
        self.scale = math.sqrt(self.d)

    def forward(self, x, latents):

        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        B, n = x.shape[:2]
        n_l = latents.shape[-2]

        x = torch.concat([x, latents], dim=-2)

        query = self.q(latents)
        key_value = self.kv(x)
        key, value = key_value.chunk(2, dim=-1)

        query = query.reshape(B, n_l, self.heads,
                              self.dim_head).permute(0, 2, 1, 3)
        key = key.reshape(B, n + n_l, self.heads,
                          self.dim_head).permute(0, 2, 1, 3)
        value = value.reshape(B, n + n_l, self.heads,
                              self.dim_head).permute(0, 2, 1, 3)

        attn = query @ key.transpose(-1, -2)
        # Softmax Stability
        attn = attn - attn.max(attn, dim=-1, keepdim=True).detach()
        attn_s = F.softmax(attn, dim=-1) / self.scale
        out = attn_s @ value

        print(out.shape, n_l)
        out = out.permute(0, 2, 1, 3).reshape(B, n_l, -1)
        out = self.out(out)

        return out


class PerceiverSampler(nn.Module):

    def __init__(self,
                 dim,
                 depth,
                 dim_head,
                 heads,
                 num_latents,
                 max_num_media,
                 max_num_frames,
                 ff_mult,):

        super().__init__()
        self.depth = depth
        self.learned_latent_queries = nn.Parameter(torch.randn(num_latents, dim))
        
        self.layers = nn.ModuleList([])
        
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PerceiverAttention(dim, dim_head, heads),
                    FeedForward(dim, ff_mult)
                ])
            )
            
        self.norm = nn.LayerNorm(dim)
        
        

    def forward(self, x):
        """
        shape of x : B x T x F x v x D
        """
        
        B, T, F, v, D = x.shape
        
        x = x.reshape(B, -1, D)
        
        latents = self.latents.unsqueeze(0).repeat(B, 1, 1)
        
        for attn, ff in self.layers:
            latents += attn(x, latents)
            latents += ff(latents)
            
        return self.norm(latents)
    

if __name__ == "__main__":
    x = torch.randn((2, 1, 1, 4, 120))
    PS = PerceiverSampler(120, 60, 6, 2)
    PS(x)


# input tensor has shape B x T x f x v x D
