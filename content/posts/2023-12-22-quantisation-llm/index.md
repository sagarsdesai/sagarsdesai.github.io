---
title: Quantisation LLM
date: 2023-12-22
author: Sagar Desai
categories: [LLM]
tags: [prod]
---

#


## Table of Contents
- [](#)
  - [Table of Contents](#table-of-contents)
- [Blog Entry for Website (Point-Wise Structure)](#blog-entry-for-website-point-wise-structure)
  - [Title: "LLM Quantisation Unpacked: AWQ v/s GGUF"](#title-llm-quantisation-unpacked-awq-vs-gguf)
    - [The Need for Quantisation:](#the-need-for-quantisation)
    - [Quantisation and Size Reduction](#quantisation-and-size-reduction)
    - [Understanding Quantisation:](#understanding-quantisation)
    - [Which Quantisation Method to Use:](#which-quantisation-method-to-use)
    - [Types of Quantisation:](#types-of-quantisation)
    - [Running Quantisation:](#running-quantisation)
    - [Factors to Compare:](#factors-to-compare)
    - [AWQ Quantisation Steps:](#awq-quantisation-steps)
    - [Methods with respect pre and post training](#methods-with-respect-pre-and-post-training)
    - [Questions, thoughts, or experiences with LLM Quantisation? stay tuned!](#questions-thoughts-or-experiences-with-llm-quantisation-stay-tuned)




# Blog Entry for Website (Point-Wise Structure)
## Title: "LLM Quantisation Unpacked: AWQ v/s GGUF"

### The Need for Quantisation:
- Fit large language models (LLMs) onto smaller devices or GPUs.
- Make LLMs more accessible for smaller companies and individuals doing testing.

### Quantisation and Size Reduction
- A primary advantage of quantisation is significant reductions in the model size. Let's consider our example, a model with a size of 7B (70 billion parameters).
- The size of a model generally corresponds to the number of parameters it has. Each parameter in a floating-point model typically uses 32 bits or 4 bytes in memory. So the size of a model can be approximately calculated as the number of parameters multiplied by the size in memory of each parameter.
- Now, let's consider a model with 7 billion parameters:
- The size of the parameters in a 32-bit floating-point model: 7B (parameters) * 4 (bytes/parameter) = 28GB
- But if we quantise the parameters to 8 bits, each parameter now takes 1 byte in memory. So, the model size after 8-bit quantisation would be: 7B (parameters) * 1 (bytes/parameter) = 7GB
- Similarly, with 4-bit quantisation each parameter now takes 0.5 bytes in memory. So, the model size after 4-bit quantisation would be: 7B (parameters) * 0.5 (bytes/parameter) = 3.5GB
- As you can see, quantisation dramatically reduces the size of the model, making it more manageable to run on smaller devices or GPUs with less memory. However, it's crucial to note that quantisation may also have a slight impact on the model's performance depending on the bit-width used. The choice of quantisation strategy should account for this trade-off between size reduction and performance.

### Understanding Quantisation:
- It simplifies post-training models.
- Approximates 32bit or 16bit numbers by a 4-bit representation.
- Storage space is dramatically reduced.

### Which Quantisation Method to Use:
- For laptops like Macs - GGUF.
- For GPUs - AWQ.

### Types of Quantisation:
- AWQ and GPTQ - rely on data sets for identifying relevant activations.
- GGUF and Bits and Bytes - data set is not required.

### Running Quantisation:
- With Bits and Bytes, you can quantize 'on the fly.'
- For AWQ and GPTQ, a data set selection is necessary before quantization.
- GGUF, in principle, also allows on-the-fly quantization.

### Factors to Compare:
- Speed - Faster on GGUF and AWQ.
- Low Fine Tuning - More straightforward with Bits and Bytes, possible with gptq and GGUF, not yet available with awq.
- Merging Adapters - Challenging in all approaches.
- Saving Model in Quantized Format - Straightforward with GGUF, GPTQ, and AWQ, not feasible with Bits and Bytes.

### AWQ Quantisation Steps:
- Loading the model.
- Installing development version of Transformers.
- Running Auto AWQ for quantization, recommended to use safe tensor format.

### Methods with respect pre and post training
- quantisation aware training QAT
- post-training quantisation PTQ

### Questions, thoughts, or experiences with LLM Quantisation? stay tuned!