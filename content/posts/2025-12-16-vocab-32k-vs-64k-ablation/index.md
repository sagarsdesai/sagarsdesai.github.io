---
title: "Vocabulary & Context - Ablating 32k vs 64k Tokenizers"
date: 2025-12-16
tags: [LLM, Ablation, Training, Vocabulary, Context-Length, Batch-Size, H200, OpenWebText]
series: ["Training a Language Model"]
draft: true
---

{{< katex >}}

#### Introduction

Following our previous study on context length, we have now introduced a new variable: **Vocabulary Size**. We scaled our tokenizer from 32,000 to 64,000 tokens to see how a richer vocabulary interacts with our context-vs-batch trade-off.

This report compares four distinct runs across two vocabulary regimes using the OpenWebText dataset on an NVIDIA H200.

#### Reference

- Repository: [SDcodehub/LM-training](https://github.com/SDcodehub/LM-training)
- Previous Study: [Context Length vs Batch Size Ablation](/2025/12/14/context-vs-batch-size-ablation)

#### The Core Question

Does doubling the vocabulary size (from 32k to 64k) significantly change the training dynamics when we trade context length for batch size?

#### The Configurations

We have two groups of experiments.

**Group A: 32k Vocabulary (~20M Params)**

| Feature | Baseline (Grey) | Experiment (Red) |
|:---|:---|:---|
| Context Length | 512 | 1024 |
| Batch Size | 256 | 128 |
| Tokens/Step | 131k | 131k |

**Group B: 64k Vocabulary (~37M Params)**

| Feature | Pink | Orange |
|:---|:---|:---|
| Context Length | 512 | 1024 |
| Batch Size | 128 | 64 |
| Tokens/Step | 65k | 65k |

*Note: The 64k runs used half the total throughput (tokens per step) of the 32k runs to manage the increased parameter count.*

#### Model Architecture Comparison

| Feature | 32k Vocab | 64k Vocab |
|:---|:---|:---|
| Vocabulary Size | 32,000 | 64,000 |
| Layers | 4 | 4 |
| Embedding Dim | 256 | 256 |
| Heads | 8 | 8 |
| Total Parameters | ~20.58M | ~37M |

#### Results

##### Loss Dynamics (The "Harder Task" Penalty)



Looking at the `val/loss` graph:
- The **32k runs (Red/Grey)** converge to a lower loss (~4.0)
- The **64k runs (Orange/Pink)** converge to a higher loss (~4.2)

This is intuitive: predicting the correct token out of 64,000 options is statistically harder than 1 out of 32,000. Interestingly, within the 64k group, the curves for `ctx1024_BS64` and `ctx512_BS128` are virtually indistinguishable, reinforcing that context length and batch size are fungible resources at this scale.

##### Perplexity



The perplexity curves mirror the loss dynamics. Both 64k runs track each other closely, as do both 32k runs.

##### Memory & Compute



The cost of the larger vocabulary is evident in the system charts:
- **Memory:** The 64k/1024 run (Orange) used nearly **140 GB** of VRAM, similar to the 32k/1024 run (Red), despite having half the batch size (64 vs 128). This confirms that the larger embedding layer (64k × 256) consumes significant memory.
- **Utilization:** All runs maintained near 100% GPU utilization, proving the H200 was kept busy regardless of the configuration.

#### Qualitative Results

Text generation samples using prompt *"Once upon a time"*:

**64k / Context 1024 (Orange)**
> "Once upon a time when the governor's office is going to approve the debt ceiling, the state administration is going to be able to hold some of the debt ceiling..."

*Analysis:* This model immediately jumped into a complex political topic (debt ceiling/spending bill). The vocabulary seems more specialized ("fiscal year," "projected"), likely a benefit of the 64k tokenizer capturing specific terms as single tokens.

**64k / Context 512 (Pink)**
> "Once upon a time of nearly thirty years, she was forced to take out the E.A. to school... She's a 50-year-old student... Her mother was known for playing an actual student."

*Analysis:* This output is more narrative but slightly incoherent ("playing an actual student"). It successfully tracks the subject ("she/her") but struggles with logical consistency.

#### Key Findings

1. **Stability is Constant:** Doubling the context length while halving the batch size resulted in **no loss of performance** for both 32k and 64k vocabularies
2. **Vocabulary Trade-off:** Increasing the vocabulary to 64k increased the model size by ~80% (20M → 37M params) but resulted in a slightly higher loss, likely because the model capacity (4 layers, 256 dim) is too small to fully exploit such a large vocabulary
3. **Memory Cost:** The larger embedding layer (64k × 256) consumes significant VRAM even with reduced batch sizes

#### Technical Deep Dive: The Cost of Increasing Vocabulary

For "Baby GPTs" (small width/depth), the vocabulary layer is often the most expensive part of the network.

##### A. Model Size Increase (Parameters)

The model size jumped from 20.58M (32k) to 36.96M (64k). Here is why:

In a Transformer, the **Embedding** layer (input) and the **Unembedding/Head** layer (output) scale linearly with vocabulary size ($V$):
- **Input Embedding:** $V \times d_{model}$
- **Output Head:** $V \times d_{model}$ (often tied to input, but your parameter count suggests they are untied)

**Calculation:**

$$
\text{Increase} = (64,000 - 32,000) \times 256 \times 2 \text{ (Input + Output)}
$$

$$
\text{Increase} = 32,000 \times 512 \approx 16,384,000 \text{ parameters}
$$

**Verification:** $20.58M + 16.38M \approx 36.96M$

**Result:** Increasing the vocab to 64k increased the model size by ~80%. The vast majority of the model's "brain" is now just a dictionary lookup table.

##### B. Compute Requirement (FLOPs)

For large models (e.g., Llama-70B), the vocabulary is a tiny fraction of compute. For a 4-layer model, the vocabulary projection is a dominant cost.

The final layer (calculating probabilities for the next token) requires a matrix multiplication of size:

$$
[\text{Batch} \times \text{SeqLen} \times d_{model}] \cdot [d_{model} \times V_{vocab}]
$$

**FLOPs per token for the final layer:**
- **32k Model:** $2 \times 256 \times 32,000 \approx 16.4$ MegaFLOPs
- **64k Model:** $2 \times 256 \times 64,000 \approx 32.8$ MegaFLOPs

**Total Training Impact:**

Since the internal 4 layers are very small (~3 MegaFLOPs per token), the final projection dominates the compute.

**Rough Estimate:** Total FLOPs per step increased by ~40-50%.

This explains why, despite halving the batch size (which usually saves memory), the VRAM usage and compute load remain substantial.

#### Conclusion

For a "Baby GPT" of this size ($d_{model} = 256$), a 32k vocabulary appears more efficient. The 64k vocabulary would likely shine only with a larger model dimension ($d_{model} \ge 512$) that can learn the subtle nuances of so many unique tokens.

The context-vs-batch trade-off remains neutral regardless of vocabulary size, confirming this is a robust finding that generalizes across tokenizer configurations.


