---
title: Understanding LLM Inferencing Challenges and Tools
date: 2024-01-29
author: Sagar Desai
categories: [LLM, Inferencing]
tags: [prod]
---

## Table of Contents
- [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
- [LLM Inferencing Challenges](#llm-inferencing-challenges)
  - [Algorithmic Challenges](#algorithmic-challenges)
  - [Engineering Challenges](#engineering-challenges)
  - [Software Level](#software-level)
  - [Hardware Level](#hardware-level)
- [Strategies to Speed Up LLM Inference](#strategies-to-speed-up-llm-inference)
  - [Algorithmic Level](#algorithmic-level)
  - [Runtime Level](#runtime-level)
- [LLM Inference Libraries](#llm-inference-libraries)
- [Conclusion](#conclusion)
- [Reference](#reference)

## Introduction
LLM inferencing is a critical aspect of modern language models deployment. As models become increasingly complex and applications more demanding, understanding the challenges and tools available to mitigate these challenges becomes extremely important. This article is a comprehensive guide to LLM inferencing challenges and tools.

While several excellent LLM inference libraries exist, we will avoid focusing too keenly on any one library, aiming for a more balanced overview.

## LLM Inferencing Challenges
### Algorithmic Challenges
- Aggregating tokens in the LLM increases the generation time due to progressively added words requiring attention.
- The unpredictable nature of prompts adds complexity. It is impossible to anticipate the length of a response, causing optimization issues.

### Engineering Challenges
- Most libraries dealing with LLMs are in Python, a high-level language not known for speed, leading to potentially slow generation times.
- Prompts exceeding a predetermined length can cause problems, such as memory allocation issues leading to performance loss.

### Software Level
- LLMs require plenty of logic between forward passes of the model, creating a serial generation process and slowing things down.
- The constant emergence of new algorithms or different attention mechanisms imposes the need to implement CUDA kernels (if working with Nvidia GPUs), which is time-consuming.

### Hardware Level
- Modern NVIDIA GPUs, used widely in AI, can struggle with some tasks due to the discrepancy between compute and data transfer capabilities.
- LLMs have large models with massive parameters, and it might not fit into a single GPU, requiring distributing workload over several GPUs.

## Strategies to Speed Up LLM Inference
### Algorithmic Level
- Altering attention mechanisms— for example, using multi-query or group query versus multi-head attention.
- Efficient model approaches for creating accurate models with fewer layers or layers that are more computationally efficient.
- Quantization— reduce the data needed for weights without major impacts on performance.

### Runtime Level
- Implementing Key-Value (KV) caching, critical for LLMs.
- Improving the efficiency of CUDA kernels or similar hardware-specific optimizations.
- Continuous Batching—constantly turning batches over to reduce the likelihood of downtime.
- Effective Pipeline Orchestration—ensuring all elements of the inference process are efficiently coordinated.

## LLM Inference Libraries
- **vLLM (Berkeley):** 
    - With tensors parallelism support and optimized multi-query attention, vLLM offers a throughput-oriented inference solution. However, it currently does not support encoder-decoder models.
- **TGI (Hugging Face):** 
    - TGI supports various open-source models with GPU kernels optimized for NVIDIA and AMD.
- **TensorRT-LLM + Triron:**
    - info WIP 
- **DeepSpeed:**
    - info WIP
- **Infery:**
    - info WIP

## Conclusion
LLM inferencing challenges span several areas, including algorithm design, engineering complexities, software, and hardware mismatches. Multiple tools and strategies are available to mitigate these challenges, but there is no one-size-fits-all solution—different applications will require different techniques.

Inference libraries like vLLM and TGI have begun to address these challenges, but continual development and improvements are necessary as LLMs become more complex. Strategically addressing these challenges will help realize the full potential of LLMs in a wide range of applications.

## Reference
- [Brousseau, C., & Sharp, M. (2024). LLMs in Production (2nd ed.). Retrieved January 15, 2024](https://www.manning.com/books/llms-in-production)
- [LLMs at Scale: Comparing Top Inference Optimization Libraries](https://www.youtube.com/watch?v=0AkKWIXKQ9s&t=1s)
