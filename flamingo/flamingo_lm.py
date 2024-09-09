import torch.nn as nn
import torch.nn.functional as F
from utils import *
from helper import GatedCrossAttentionBlock

class FlamingoLayer(nn.Module):
    
    def __init__(self,
                 gated_cross_attn_layer,
                 decoder_layer,
                 gradient_checkpointing=False):
        
        super().__init__()
        
        self.gated_cross_attn_layer = gated_cross_attn_layer
        self.decoder_layer = decoder_layer
        
        self.media_x = None
        self.media_locations = None
        
        self.use_cached_media = False
        
    def is_conditioned(self):
        return (self.media_x and self.media_locations)
        
    def condition_media_x(self, media_x):
        self.media_x = media_x
        
    def condition_media_location(self, media_locations):
        self.media_locations = media_locations
        
    def condition_use_cached_media(self, use_cached_media):
        self.use_cached_media = use_cached_media
        
    def forward(self, x, attention_mask=None, **decoder_layer_kwargs):
        
        if (self.gated_cross_attn_layer is not None):
            if (not self.is_conditioned()):
                raise ValueError("Condition the FlamingoLayer with media and media locations first")
            
            x = self.gated_cross_attn_layer(x, self.media_x, self.media_locations, use_cached_media=self.use_cached_media)
            
        x = self.decoder_layer(x, attention_mask, **decoder_layer_kwargs)
        
        return x
    
# Mixin Class that extends Language Encoder's Class with Flamingo capabilities like GCA layer

class FlamingoLMMixin(nn.Module):
    
    def set_decoder_layers_attr_name(self, decoder_layers_attr_name):
        self.decoder_layers_attr_name = decoder_layers_attr_name
    
    # decoder_layers_attr_name = model.decoder.layers
    # return LangEncoder.model.decoder.layers
    def _get_decoder_layers(self):
        return getattr_recursively(self, self.decoder_layers_attr_name)
    
    # set LangEncoder.model.decoder.layers = value (new FlamingoLayers)
    def _set_decoder_layers(self, value):
        setattr_recursively(self, self.decoder_layers_attr_name, value)
    
    # dim_head, heads for GCAB is fixed!!
    def init_flamingo(self, 
                      media_token_id, 
                      x_dim_size, 
                      media_dim_size, 
                      cross_attn_every_n_layer, 
                      gradient_checkpointing=False):
        
        self.original_decoder_layers = self._get_decoder_layers()
        self.gated_cross_attn_layers = nn.ModuleList([])
        
        for layer_idx, _ in enumerate(self.original_decoder_layers):
            self.gated_cross_attn_layers.append(
                GatedCrossAttentionBlock(x_dim_size, media_dim_size)
                if (layer_idx + 1) % cross_attn_every_n_layer == 0
                else None
            )
            
        self.media_token_id = media_token_id
        self.gradient_checkpointing = gradient_checkpointing

        self.init_flamingo_layers()
        self.initialized_flamingo = True
        self._use_cached_media = False
        
    def init_flamingo_layers(self):
        new_flamingo_decoder_layers = nn.ModuleList([])
        
        for layer_idx, layer in enumerate(self.gated_cross_attn_layers):
            new_flamingo_decoder_layers.append([
                FlamingoLayer(layer, self.original_decoder_layers[layer_idx], self.gradient_checkpointing)
            ])
            
        self._set_decoder_layers(new_flamingo_decoder_layers)
        
    def forward(self, input_ids, attention_mask, **kwargs):
        
        media_locations = input_ids == self.media_token_id
        
        use_cached_media_locations = False
        
        if (self._use_cached_media
            and self.is_conditioned() # Important flag for caching decision in subsequent passes of generate().
            and not media_locations.any()):
            use_cached_media_locations = True
        
        for flamingo_layer in self._get_decoder_layers():
            if (not use_cached_media_locations):
                flamingo_layer.condition_media_locations(media_locations)
            flamingo_layer.condition_use_cached_media(use_cached_media_locations)
            
        kwargs['input_ids'] = input_ids
        kwargs['attention_mask'] = attention_mask
        
        return super().forward(**kwargs) # Calls the original LangEncoder class' forward method (Python MRO)
    
    def is_conditioned(self):
        for flamingo_layer in self._get_decoder_layers():
            if (not flamingo_layer.is_conditioned()):
                return False
        return True
    
    def clear_conditioned_layers(self):
        for flamingo_layer in self._get_decoder_layers():
            flamingo_layer.condition_media_x(None)
            flamingo_layer.condition_media_location(None)
            flamingo_layer.condition_use_cached_media(False)