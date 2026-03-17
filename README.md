# flash-attention-from-scratch
A ground-up implementation of Flash Attention in CUDA, built as a learning project. Covers the full progression: vector addition → naive matmul → tiled matmul → naive attention → Flash Attention forward pass. Each stage is profiled with ncu to empirically show why tiling reduces HBM traffic. Final stage includes a Triton rewrite for comparison.
