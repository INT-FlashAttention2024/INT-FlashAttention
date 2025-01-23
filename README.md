# INT-FlashAttention: Enabling Flash Attention for INT8 Quantization

This repository is the official implementation of INT-FlashAttention. 



## About this repository

- `flash_atten_*.py` contains the main code for different Flash Attention algorithm in Triton implementation.
- `benchmark.py` contains the performance benchmark for different algorithm above.
- `configs.py` contains the configs you may need to adjust for the Triton autotune.
- `csrc` contains ours algorithm implementation in cuda version. You should reference the official repository for [Flash Attention](https://github.com/Dao-AILab/flash-attention) to compile it.
- More details can be found in the folders.

