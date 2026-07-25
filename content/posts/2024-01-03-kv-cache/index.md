---
title: KV Cache in Transformers- Detailed and Simplified Guide
date: 2024-01-03
author: Sagar Desai
categories: [LLM]
tags: [prod]
---

#


## Table of Contents
- [](#)
  - [Table of Contents](#table-of-contents)
    - [Transformers and GPU Memory](#transformers-and-gpu-memory)
    - [Understanding Self-Attention Mechanism](#understanding-self-attention-mechanism)
    - [The Role of KV Cache](#the-role-of-kv-cache)
    - [Pivotal Function of KV Cache within Transformer's Architecture](#pivotal-function-of-kv-cache-within-transformers-architecture)
    - [Memory Usage Calculation](#memory-usage-calculation)
    - [Drawbacks and Constraints of KV Cache](#drawbacks-and-constraints-of-kv-cache)
    - [KVs and Latency](#kvs-and-latency)
    - [Conclusion](#conclusion)
    - [References](#references)


### Transformers and GPU Memory
- OpenAI's GPT-3 charges twice as per input token for longer context models.
- Economic consequences of high memory consumption.
- Most memory, especially with larger context lengths, goes towards the KV cache.

### Understanding Self-Attention Mechanism
- Each token corresponds to an embedding vector X.
- X is multiplied by matrices to form Query (Q), Key (K), and Value (V) vectors.
- Q represents the new token, K and V depict previous contexts.
- Attention mechanism: Softmax((Q.K^T)/sqrt(d))*V.

### The Role of KV Cache
- In autoregressive decoding, Q vector is generated, and cached values of K and V matrices are fetched.
- Model calculates a new column for the K matrix and a new row for the V matrix.

### Pivotal Function of KV Cache within Transformer's Architecture
- KV cache works seamlessly with the self-attention layer.
- Self-attention layer processes previous K and V cache and the embedding for the current token.
- Computes new K and V vectors for the current token, appends them to KV cache.

### Memory Usage Calculation
- Formula: `2 x precision x layers x dimension x sequence_length x batch`
- Elements:
  - 2 for K and V matrices.
  - Precision: number of bytes per parameter.
  - Layers: total number of layers.
  - Dimension: size of embeddings per layer.
  - Sequence_length: length of the sequence to generate.
  - Batch: batch size.

### Drawbacks and Constraints of KV Cache
- Memory allocation breakdown for a 13B-parameter Language Model (LM) on NVIDIA A100 GPU with 40GB RAM.
- Approximately 65% for model weights, 30% for dynamic states (KV cache), and the remaining for other data.

### KVs and Latency
- Higher latency when processing the prompt versus subsequent tokens.
- For subsequent tokens, latency is lower as only cached K and V need to be computed.

### Conclusion
- KV Cache is a core component of Transformer models.
- Demands proper management due to its significant impact on memory usage, especially in large models.
- Balancing powerful NLP models and optimizing them to prevent resource constraints.

### References
- [Vaswani et al. - Attention Is All You Need](#)
- [Kwon et al. - Efficient Memory Management for Large Language Model Serving with PagedAttention](#)
- [The KV Cache: Memory Usage in Transformers](#)
- [Jay Alammar - The Illustrated Transformer](#)
- [YouTube Video - The KV Cache](https://www.youtube.com/watch?v=80bIUggRJf4)
