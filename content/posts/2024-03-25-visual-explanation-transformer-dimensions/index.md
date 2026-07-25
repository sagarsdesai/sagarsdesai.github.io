---
title: Visual Explanation of Transformer with Dimensions
date: 2024-03-25
author: Sagar Desai
categories: [LLM]
tags: [prod]
---

## Table of Contents
- [Table of Contents](#table-of-contents)
- [Key points](#key-points)
- [Reference](#reference)


![Multihead Attention Mechanism](\assets_files\blogs\2024-03-25-Visual-Explanation-Transformer-Dimensions\transformer.png)


## Key points
- Batch of 30 sentences 
- Sequence length is 50
- Embedding dimension is 512
- Number of layers 64 decoder is repeated 64 times
  - Gives overlapping learning, makes the embedding more context-aware
- Attention head 8 parallel attention learning
  - Gives parallel independent learning
- Concat
  - Combine the attention layers output, optionally can be passed to a linear layer 
- Add residue
  - To allow gradient flow deep in network, speed up training
- Layer normalization
  - Normalizes across embedding for each token
- Feedforward
  - To add nonlinearity, complex learning
- Linear layer
  - Projects the embedding to vocabulary size
- Decode
  - Decodes to token basis probability from softmax

## Reference
- [code](https://github.com/SDcodehub/LLMs-from-scratch/blob/main/ch03/01_main-chapter-code/multihead-attention.ipynb)
- [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch)
- [CodeEmporium](https://www.youtube.com/watch?v=rPFkX5fJdRY&t=5079s)
- [Umar Jamil](https://www.youtube.com/watch?v=bCz4OMemCcA)
