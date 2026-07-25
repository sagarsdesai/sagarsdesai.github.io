---
title: LLMOps 1
date: 2023-11-01
author: Sagar Desai
categories: [LLM, production]
tags: [prod]
---

#


## Table of Contents
- [](#)
  - [Table of Contents](#table-of-contents)
  - [Metrics to understand for LLM production](#metrics-to-understand-for-llm-production)
      - [Throughput](#throughput)
      - [Latency](#latency)
      - [Cost](#cost)
  - [What affects the LLM metrics](#what-affects-the-llm-metrics)
  - [GPU memory utilisations](#gpu-memory-utilisations)


## Metrics to understand for LLM production

#### Throughput
  - defined as queries processed per second
  - Maximise Throughput to make best use of GPU resources
#### Latency
  - defined as time per token
  - Minimise to suit the user experiance
#### Cost
  - cost of each token processed
  - Minimise

## What affects the LLM metrics
- Time for computation while inference
- Loading model into memory
- An breakeven exists between the batch size we choose to process in terms of inference and loading model.
- below this breakeven the latency is affected by loading of the model
- Above this breakeven the latency is affected by computation of the tokens 
- Making decision on the batch size is important

## GPU memory utilisations
- Model weights
- space to run calculations
- KV cache


----
References - 
1. [LLMs in Production](https://www.manning.com/books/llms-in-production)
2. [MLOps.community](https://www.youtube.com/@MLOps)