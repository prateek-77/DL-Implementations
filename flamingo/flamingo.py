import torch.nn as nn
import torch.nn.functional as F
from helper import PerceiverSampler

# TODO: Check Gradient Checkpointing

class Flamingo(nn.Module):
    
    def __init__(self,
                 vision_encoder, # OpenAI CLIP encoder
                 lang_encoder, # Assume already extended with LMMixin
                 media_token_id,
                 eoc_token_id,
                 media_dim,
                 cross_attn_every_n_layers,
                 gradient_checkpointing=False):
        
        self.media_token_id = media_token_id
        self.eoc_token_id = eoc_token_id
        
        self.vision_encoder = vision_encoder.visual
        self.perceiver = PerceiverSampler(dim=media_dim)
        
        self.lang_encoder = lang_encoder
        self.x_dim = lang_encoder.config.hidden_size # Handle MPT
        
        self.lang_encoder.init_flamingo(media_token_id, 
                                        self.x_dim, 
                                        media_dim,
                                        cross_attn_every_n_layers, 
                                        gradient_checkpointing)
        
    def forward(self,
                media,
                x,
                attention_mask=None,
                labels=None,
                clear_conditioned_layers=True,
                past_key_values=None,
                use_cache=False):
        
        if (self.lang_encoder._use_cached_media):
            assert media is None # Use media that is already conditioned
            assert self.lang_encoder.is_conditioned()
        
        else:
            self._encode_and_condition_media(media)
            self._condition_media_locations(x)
            
        output = self.lang_encoder(
            input_ids = x,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache
        )
        
        if clear_conditioned_layers:
            self.lang_encoder.clear_conditioned_layers()
            
        return output
    
    def generate(self, 
                 media, 
                 x, 
                 attention_mask = None, 
                 **kwargs):
        
        # Add beam search support
        
        self._encode_and_condition_media(media)
        self.lang_encoder._use_cached_media = True
        
        eos_token_id = kwargs.pop('eos_token_id', self.eoc_token_id)
        
        output = self.lang_encoder(
            input_ids = x,
            attention_mask = attention_mask,
            eos_token_id = eos_token_id,
            **kwargs
        )
        
        self.lang_encoder.clear_conditioned_layers()
        self.lang_encoder._use_cached_media = False
        
        return output
        
    
    def _encode_and_condition_media(self, media):
        # media shape: B x T x F x C x H x W
        
        B, T, F, C, H, W = media.shape[:-3]
        
        media = media.reshape(-1, C, H, W)
        
        media = self.vision_encoder(media)[1] # (B T F) x v x D
        v, D = media.shape[:-2]
        media = media.reshape(B, T, F, v, D)
        
        media = self.perceiver(media)
        
        for layer in self.lang_encoder._get_decoder_layers:
            layer.condition_media_x(media)

    def _condition_media_locations(self, input_ids):
        
        media_locations = input_ids == self.media_token_id
        
        for layer in self.lang_encoder._get_decoder_layers:
            layer.condition_media_location(media_locations)
        
        
        