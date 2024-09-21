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
                                nn.Linear(inner_dim, dim, bias=False))

    def forward(self, x):
        return self.ff(x)


class PerceiverAttention(nn.Module):

    def __init__(self, dim, dim_head, heads):
        super().__init__()

        self.heads = heads
        self.dim_head = dim_head
        self.scale = math.sqrt(dim_head)
        hidden_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.q = nn.Linear(dim, hidden_dim, bias=False)
        self.kv = nn.Linear(dim, hidden_dim*2, bias=False)
        self.out = nn.Linear(hidden_dim, dim, bias=False)


    # x: B x n x D (media)
    # latents: B x n_l x D
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
        attn = attn - torch.amax(attn, dim=-1, keepdim=True).detach()
        attn_s = F.softmax(attn, dim=-1) / self.scale
        out = attn_s @ value

        print(out.shape, n_l)
        out = out.permute(0, 2, 1, 3).reshape(B, n_l, -1)
        out = self.out(out)

        return out

# Don't allow frames and time steps
# Current Implementation has frames=1, time steps=1 [Only supports images]

class PerceiverSampler(nn.Module):

    def __init__(self,
                 dim,
                 depth=6,
                 dim_head=64,
                 heads=8,
                 num_latents=64,
                 max_num_media=None,
                 max_num_frames=None,
                 ff_mult=4):

        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        
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
        
        B, *_, D = x.shape
        x = x.reshape(B, -1, D)
        
        latents = self.latents.unsqueeze(0).repeat(B, 1, 1)
        
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
            
        return self.norm(latents)
    

class MaskedCrossAttention(nn.Module):
    
    def __init__(self,
                 dim,
                 dim_media,
                 dim_head=64,
                 heads=8,
                 only_attend_immediate_media=True
                 ):
        
        super().__init__()
        self.dim = dim
        self.dim_head = dim_head
        self.heads = heads
        
        self.norm = nn.LayerNorm(dim)
        
        hidden_dim = dim_head * heads
        self.to_q = nn.Linear(dim, hidden_dim, bias=False)
        self.to_kv = nn.Linear(dim_media, 2 * hidden_dim, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)
        
        self.scale = math.sqrt(dim_head)
    
    # x shape: B x T_text x D_t
    # media shape: B x T_img x n x D_media
    # We use cached media during generation of new tokens:
    # we don't have media tokens during autoregressive generation
    
    def forward(self, x, media, media_locations, use_cached_media=False):
        
        B, T_text, _ = x.shape
        _, T_img, n, D_media = media.shape
        
        x = self.norm(x)
        media = media.reshape(B, -1, D_media)
        
        query = self.to_q(x)
        key_value = self.to_kv(media)
        key, value = key_value.chunk(2, dim=-1)
        
        query = query.reshape(B, T_text, self.heads, self.dim_head).permute(0, 2, 1, 3)
        key = key.reshape(B, T_img * n, self.heads, self.dim_head).permute(0, 2, 1, 3)
        value = value.reshape(B, T_img * n, self.heads, self.dim_head).permute(0, 2, 1, 3)
        
        attn = query @ key.transpose(-2, -1) # B x h x T_text x (T_img * n)
        
        media_time = torch.arange(T_img, device=x.device) + 1 # T_img
        text_time = media_locations.cumsum(dim=1) # B x T_text
        
        # Set all text tokens to attend the latest media
        if (use_cached_media):
            text_time = text_time.max(dim=1, keepdim=True).repeat(1, T_text)
        
        media_time = media_time.repeat_interleave(n).reshape(1, 1, 1, n * T_img)
        text_time = text_time.reshape(B, 1, T_text, 1)
        
        mask_text_img = media_time == text_time
        
        attn.masked_fill(~mask_text_img, -torch.inf)
        attn = attn - attn.amax(dim=-1, keepdim=True).detach()
        attn = F.softmax(attn, dim=-1) / self.scale
        
        # Zeroing out attention for text tokens that have no preceding media
        attn.masked_fill(text_time==0, 0.)
        
        out = attn @ value # B x h x T_text x dim_head
        out = out.permute(0, 2, 1, 3).reshape(B, T_text, -1)
        
        out = self.to_out(out)
        
        return out
    
class GatedCrossAttentionBlock(nn.Module):
    
    def __init__(self,
                 dim,
                 dim_media,
                 dim_head=64,
                 heads=8,
                 ff_mult=4,
                 only_attend_immediate_media=True
                 ):
        super().__init__()
        self.alpha_xattn = nn.Parameter(torch.zeros(1))
        self.alpha_dense = nn.Parameter(torch.zeros(1))
        self.cross_attn = MaskedCrossAttention(dim, dim_media, dim_head, heads, only_attend_immediate_media)
        self.ff = FeedForward(dim, ff_mult)
        
    def forward(self, x, media, media_locations, use_cached_media=False):
        
        x = x + torch.tanh(self.alpha_xattn) * self.cross_attn(x, media, media_locations, use_cached_media)
        x = x + torch.tanh(self.alpha_dense) * self.ff(x)
        
        return x
        

if __name__ == "__main__":
    x = torch.randn((2, 1, 1, 4, 120))
    PS = PerceiverSampler(120, 6, 10, 6, 2, 1, 1, 4)
    PS(x)
    
    x = torch.randn((2, 4, 120))
    y = torch.randn((2, 2, 4, 120))
    z = torch.randn((2, 4))
    MCA = MaskedCrossAttention(120, 120, 10, 6, True)
    MCA(x, y, z)
    
    GCA = GatedCrossAttentionBlock(120, 120, 10, 6, True)
    GCA(x, y, z)


# input tensor has shape B x T x f x v x D
