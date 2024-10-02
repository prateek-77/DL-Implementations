# Flamingo

This repository contains my implementation of Flamingo, a Vision-Language Model (VLM). Inspired by [OpenFlamingo](OpenFlamingo), this model integrates both visual and language data to perform tasks like image captioning and visual question answering (VQA).

Currently, the repository supports only inference using pretrained weights with image conditioning.

## Example

[inference.ipynb](https://github.com/prateek-77/DL-Implementations/blob/main/flamingo/src/inference.ipynb) consists of a simple example with CLIP vision encoder (ViT-L-14) and RedPajama-INCITE-Base-3B-v1 language encoder.

## Notes
TODO

## References
[Flamingo](https://arxiv.org/abs/2204.14198) \
[OpenFlamingo](https://arxiv.org/abs/2308.01390) \
[OpenFlamingo codebase](https://github.com/mlfoundations/open_flamingo) 