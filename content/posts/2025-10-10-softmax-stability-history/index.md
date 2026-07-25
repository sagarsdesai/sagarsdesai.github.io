---
title: "Softmax: What It Does, Stability Trick, and a Brief History"
date: 2025-10-10
categories: [Deep Learning, Math]
tags: [softmax, attention, stability, numerics, transformers]
description: "Build a numerically stable softmax, understand what it does, and trace its evolution from logistic regression to attention."
---

{{< katex >}}

### TL;DR — What softmax does

- **Normalizes scores to probabilities**: turns arbitrary real-valued logits into values in \(0, 1\) that sum to 1 along a chosen dimension.
- **Amplifies differences**: larger logits get disproportionately higher probabilities; temperature \(T\) controls sharpness.
- **Used everywhere**: output layers for multiclass classification, attention weights, sampling from categorical distributions, and more.

---

### 1) The Big Picture: What is softmax and why does it need a "trick"?

The softmax function takes a vector of real numbers ("logits" or "scores") and transforms them into a probability distribution, where:

1. All outputs are between 0 and 1
2. Outputs sum to 1 along the chosen dimension

The formula is:

\[\mathrm{softmax}(\mathbf{v})_i = \frac{e^{v_i}}{\sum_{j=1}^{n} e^{v_j}}\]

**The Problem:** Computers have limited floating-point range. If you have a large logit, say `v_i = 1000`, then `e^1000` overflows to `inf`. With multiple large values, you can get `inf / inf → NaN`, breaking training or inference.

**The Stability Trick (shift-invariance):** Softmax is invariant to adding a constant `c` to all logits: `softmax(v) = softmax(v - c)`. Choose `c = max(v)`:

1. Compute `v' = v - max(v)`
2. The largest value in `v'` becomes 0 → `e^0 = 1`
3. All other entries are negative → their exponentials lie in `(0, 1]`
4. This avoids overflow and yields numerically stable results

---

### 2) Implementation: a stateless functional kernel

Since softmax is a pure, stateless operation, it belongs in a functional module. The implementation below follows the stability trick.

```python
import torch

def softmax(input: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Applies a numerically stable softmax function.

    Args:
        input (torch.Tensor): The input tensor of logits.
        dim (int): The dimension along which softmax is computed.

    Returns:
        torch.Tensor: Probabilities with the same shape as input.
    """
    # 1) subtract max for numerical stability (keepdim=True for broadcasting)
    max_vals, _ = torch.max(input, dim=dim, keepdim=True)
    shifted_logits = input - max_vals

    # 2) exponentiate
    exps = torch.exp(shifted_logits)

    # 3) sum and normalize
    sum_exps = torch.sum(exps, dim=dim, keepdim=True)
    return exps / sum_exps
```

If you're using PyTorch in practice, `torch.nn.functional.softmax` and `torch.nn.functional.log_softmax` already implement these stability tricks. For loss functions, prefer `torch.nn.CrossEntropyLoss`, which combines `log_softmax` with `NLLLoss` stably and efficiently.

---

### 3) What softmax enables in models

- **Multiclass classification**: final layer over logits to get class probabilities.
- **Attention mechanisms**: converts similarity scores (e.g., query–key dot products) into weights that sum to 1 across tokens; enables weighted sums of values.
- **Sampling**: provides a categorical distribution to sample discrete outcomes; temperature scaling adjusts sharpness.

---

### 4) A brief history and evolution

- **Origins in multinomial logistic regression (a.k.a. softmax regression):** Extends binary logistic regression to multi-class by applying the softmax link function; the associated loss is the cross-entropy.
- **Statistical physics connection:** Softmax mirrors the Boltzmann distribution; with a temperature parameter \(T\), \(\mathrm{softmax}(\mathbf{v} / T)\) sharpens (small \(T\)) or smooths (large \(T\)) probabilities.
- **Neural networks (1980s–2010s):** Became the de facto output activation for multi-class tasks; paired with cross-entropy for maximum-likelihood training.
- **Transformers and attention (2017–):** In "Attention Is All You Need", scaled dot-product attention uses softmax to normalize attention scores. Subsequent work introduced efficient kernels (FlashAttention), fused softmax, masking, and causal variants.
- **Alternatives and variants:** `sparsemax` and `entmax` produce sparse probabilities; temperature scaling calibrates probabilities; label smoothing regularizes training; `log_softmax` improves numerical stability when used with log-likelihood losses.

---

### References and further reading

- Goodfellow, Bengio, Courville. Deep Learning — Chapter on Deep Feedforward Networks (softmax + cross-entropy)
- Vaswani et al., 2017. Attention Is All You Need (scaled dot-product attention with softmax)
- Martins & Astudillo, 2016. From Softmax to Sparsemax: A Sparse Model of Attention
- Peters et al., 2019–2023. Variants on efficient attention and fused softmax kernels


