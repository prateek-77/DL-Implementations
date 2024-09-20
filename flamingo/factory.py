import torch
import torch.nn as nn

import open_clip
from transformers import AutoModelForCausalLM, AutoTokenizer

from flamingo import Flamingo
from flamingo_lm import FlamingoLMMixin
from utils import extend_instance

def create_model(
    clip_vision_encoder_path,
    clip_vision_encoder_pretrained,
    lang_encoder_path,
    tokenizer_path,
    cross_attn_every_n_layers,
    use_local_files,
    decoder_layers_attr_name,
    freeze_lm_embeddings,
    cache_dir,
    **flamingo_kwargs):
    
    vision_encoder, _, image_processor = open_clip.create_model_and_transforms(
        clip_vision_encoder_path,
        pretrained=clip_vision_encoder_pretrained,
        cache_dir=cache_dir
    )
    vision_encoder.visual.output_tokens = True
    media_dim = open_clip.get_model_config(clip_vision_encoder_path)['vision_cfg']['width']
    
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=use_local_files,
        trust_remote_code=True,
        cache_dir=cache_dir
        )
    tokenizer.add_special_tokens({'additional_special_tokens': ['<|endofchunk|>', '<media>']})
    
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<PAD>'})
    
    lang_encoder = AutoModelForCausalLM.from_pretrained(
        lang_encoder_path,
        local_files_only=use_local_files,
        trust_remote_code=True,
        cache_dir=cache_dir
        )

    lang_encoder.resize_token_embeddings(len(tokenizer))
    lang_encoder = extend_instance(lang_encoder, FlamingoLMMixin)
    
    if decoder_layers_attr_name is None:
        decoder_layers_attr_name = get_decoder_layers_attr_name(lang_encoder)
    
    lang_encoder.set_decoder_layers_attr_name(decoder_layers_attr_name)
    
    # Why do we take the last token
    media_token_id = tokenizer.encode("<media>")[-1]
    eoc_token_id = tokenizer.encode("<|endofchunk|>")[-1]
    
    flamingo = Flamingo(vision_encoder,
                        lang_encoder,
                        media_token_id,
                        eoc_token_id,
                        media_dim,
                        cross_attn_every_n_layers,
                        gradient_checkpointing=False)
    
    flamingo.requires_grad_(False)
    
    return flamingo, image_processor, tokenizer

def get_decoder_layers_attr_name(lang_encoder):
    lang_encoder_name = lang_encoder.__class__.__name__
    
    for k, v in __KNOWN_DECODER_LAYERS_ATTR_NAMES:
        if k.lower() in lang_encoder_name.lower():
            return v
    
    return None
    
    
__KNOWN_DECODER_LAYERS_ATTR_NAMES = {
    "opt": "model.decoder.layers",
    "gptj": "transformer.h",
    "gpt-j": "transformer.h",
    "pythia": "gpt_neox.layers",
    "llama": "model.layers",
    "gptneoxforcausallm": "gpt_neox.layers",
    "mpt": "transformer.blocks",
    "mosaicgpt": "transformer.blocks",
}
    