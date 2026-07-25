---
title: Transformer From Scratch — Causal Multi-Head Self-Attention
date: 2025-10-17
author: Sagar Desai
categories: [LLM]
tags: [transformers, pytorch, cs336, attention]
---

{{< katex >}}

Problem (multihead_self_attention): Implement causal multi-head self-attention (5
points)
Deliverable: Implement causal multi-head self-attention as a torch.nn.Module. Your implemen-
tation should accept (at least) the following parameters:
d_model: int Dimensionality of the Transformer block inputs.
num_heads: int Number of heads to use in multi-head self-attention.
Folllowing Vaswani et al. [2017], set dk = dv = dmodel/h. To test your implementation against our
provided tests, implement the test adapter at [adapters.run_multihead_self_attention]

Of course. We're now at the pinnacle of the Transformer block. Let's build the **Causal Multi-Head Self-Attention** module. This layer is the powerhouse that lets the model look at different parts of the input sequence and decide what's important. We'll be combining our `Linear`, `RotaryPositionalEmbedding`, and `scaled_dot_product_attention` building blocks to make it happen.

-----

### \#\# 1. The Big Picture: Why "Multi-Head"?

Imagine you're reading a sentence. To understand a word, you might look for different kinds of clues simultaneously: "What's the subject of this verb?" (a syntactic clue), or "Which previous word has a similar meaning?" (a semantic clue).

Single-head attention is like having only one way to look back at the sequence. It might get good at one type of relationship but struggle with others. [cite_start]**Multi-head attention** fixes this by running the attention mechanism multiple times in parallel, each with its own set of learned weights[cite: 721]. Each "head" can learn to focus on a different type of relationship in the data. Afterwards, the results from all heads are combined to produce a rich, multi-faceted representation.

### \#\# 2. The Implementation Steps

The process inside the module follows a clear sequence:

1.  [cite_start]**Project Inputs:** We take the input tensor `x` and project it into Query (Q), Key (K), and Value (V) tensors using three independent `Linear` layers ($W_Q, W_K, W_V$)[cite: 727, 729].
2.  **Split Heads:** We reshape the Q, K, and V tensors to split the `d_model` dimension into `num_heads` and a smaller `head_dim`. For example, a tensor of shape `(batch, seq_len, 512)` with 8 heads becomes `(batch, 8, seq_len, 64)`.
3.  [cite_start]**Apply RoPE:** We apply our `RotaryPositionalEmbedding` to the multi-headed Q and K tensors to inject positional information[cite: 739]. [cite_start]The head dimension is treated like a batch dimension, so the same rotation is applied to each head's vectors[cite: 740, 741].
4.  **Apply Attention:** We feed the rotated Q, K, and the original V into our `scaled_dot_product_attention` function. [cite_start]We also provide a **causal mask** to prevent tokens from "peeking" at future tokens in the sequence[cite: 737]. This mask is a lower-triangular matrix of `True` values.
5.  **Combine Heads:** We reverse the split operation, concatenating the outputs from all heads back into a single tensor of shape `(batch, seq_len, d_model)`.
6.  [cite_start]**Final Projection:** Finally, this combined tensor is passed through one last `Linear` layer ($W_O$) to produce the module's final output[cite: 727, 729].

### \#\# 3. The Code: The `MultiHeadSelfAttention` Module

This is a stateful module, as it holds the learnable weight matrices for the projections. It's the most complex module we've built, bringing together many of our previous components.

**File:** `cs336_basics/nn/modules/attention.py` (a new file)

```python
import torch
import torch.nn as nn

from .linear import Linear
from .rope import RotaryPositionalEmbedding
from .. import functional as F

class MultiHeadSelfAttention(nn.Module):
    """
    Implements a stateful Causal Multi-Head Self-Attention module.
    """
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, rope_theta: float = 10000.0, device=None, dtype=None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Linear layers for Q, K, V, and output projections
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.out_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        # Rotary Positional Embedding
        self.rope = RotaryPositionalEmbedding(
            theta=rope_theta,
            d_k=self.head_dim,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype
        )

        # Causal mask buffer
        # The mask is created once and reused, not a learnable parameter.
        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len, device=device, dtype=torch.bool))
        self.register_buffer('causal_mask', causal_mask, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # 1. Project inputs to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. Split heads
        # (B, T, D) -> (B, T, H, h_dim) -> (B, H, T, h_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 3. Apply RoPE to Q and K
        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)
        
        # 4. Apply Scaled Dot-Product Attention with Causal Mask
        # The causal mask is (T, T), we need to slice it for the current seq_len
        mask = self.causal_mask[:seq_len, :seq_len]
        attention_output = F.scaled_dot_product_attention(q, k, v, mask=mask)

        # 5. Combine heads
        # (B, H, T, h_dim) -> (B, T, H, h_dim) -> (B, T, D)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # 6. Final output projection
        return self.out_proj(attention_output)
```

-----

### \#\# 4. Tying It Together

Finally, expose the new module in your `__init__.py` files.

**File:** `cs336_basics/nn/modules/__init__.py`

```python
# ... (previous imports)
from .attention import MultiHeadSelfAttention # Add this line
```

**File:** `cs336_basics/nn/__init__.py`

```python
from . import functional
from .modules import (
    # ... (previous imports)
    MultiHeadSelfAttention # Add this line
)
```

### \#\# Next Steps

1.  **Create/update the files** with the code.
2.  **Implement the test adapter** `adapters.run_multihead_self_attention`.
3.  **Run the tests** with `uv run pytest -k test_multihead_self_attention`.

Congratulations! Once this passes, you have officially built the most complex and important sub-layer of the Transformer. The only things left are to assemble this and the FFN into a full `TransformerBlock`, and then stack those blocks to create the final `TransformerLM`. You're on the home stretch.


