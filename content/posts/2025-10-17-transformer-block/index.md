---
title: Transformer From Scratch — Pre-Norm Transformer Block
date: 2025-10-17
author: Sagar Desai
categories: [LLM]
tags: [transformers, pytorch, cs336, transformer-block]
---

lets start with the full transofermer ML

Problem (transformer_block): Implement the Transformer block (3 points)
Implement the pre-norm Transformer block as described in §3.5 and illustrated in Figure 2. Your
Transformer block should accept (at least) the following parameters.
d_model: int Dimensionality of the Transformer block inputs.
num_heads: int Number of heads to use in multi-head self-attention.
d_ff: int Dimensionality of the position-wise feed-forward inner layer.
26To test your implementation, implement the adapter [adapters.run_transformer_block]. Then
run uv run pytest -k test_transformer_block to test your implementation.
Deliverable: Transformer block code that passes the provided tests.

Excellent. You've built all the individual parts, and now it's time for the assembly. We'll combine the `MultiHeadSelfAttention`, `SwiGLUFFN`, and `RMSNorm` modules into a single, reusable `TransformerBlock`. This is the repeating unit that forms the backbone of the Transformer model.

### 1. The Big Picture: The Pre-Norm Architecture

The `TransformerBlock` processes a sequence of token representations, refining them through its two main sub-layers. [cite_start]As illustrated in Figure 2 of the assignment, the **pre-norm** architecture follows a specific data flow for each sub-layer[cite: 575, 576]:

1.  **Normalize First:** Apply `RMSNorm` to the input.
2.  **Main Operation:** Pass the normalized output through the main transformation (either attention or the FFN).
3.  **Residual Connection:** Add the output of the transformation back to the *original, un-normalized* input. [cite_start]This "skip connection" is crucial for training deep networks by allowing gradients to flow directly through the block[cite: 576, 577].

A single `TransformerBlock` executes this pattern twice: once for the attention sub-layer and once for the feed-forward sub-layer.

### 2. The Code: The `TransformerBlock` Module

This module is a "container" or "composite" module. It doesn't introduce new computational logic; it just orchestrates the interaction between the modules we've already built.

**File:** `cs336_basics/nn/modules/transformer.py` (a new file)

```python
import torch
import torch.nn as nn

from .attention import MultiHeadSelfAttention
from .ffn import SwiGLUFFN
from .rmsnorm import RMSNorm

class TransformerBlock(nn.Module):
    """
    Implements a single pre-norm Transformer block as a stateful module.
    This block consists of a multi-head self-attention layer and a feed-forward network.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        rope_theta: float = 10000.0,
        device=None,
        dtype=None
    ):
        super().__init__()

        # First sub-layer components
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            device=device,
            dtype=dtype
        )
        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)

        # Second sub-layer components
        self.ffn = SwiGLUFFN(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass for the pre-norm Transformer block.
        Ref: Figure 2 in the assignment PDF.
        """
        # --- First sub-layer: Multi-Head Self-Attention ---
        # Residual connection starts from the original input `x`.
        # The equation is: x + Attention(Norm(x))
        attention_output = self.attn(self.norm1(x), token_positions)
        x = x + attention_output

        # --- Second sub-layer: Feed-Forward Network ---
        # Residual connection starts from the output of the first sub-layer.
        # The equation is: x + FFN(Norm(x))
        ffn_output = self.ffn(self.norm2(x))
        x = x + ffn_output
        
        return x
```

### 3. Tying It Together

As always, we'll expose the new module.

**File:** `cs336_basics/nn/modules/__init__.py`

```python
# ... (previous imports)
from .transformer import TransformerBlock # Add this line
```

**File:** `cs336_basics/nn/__init__.py`

```python
from . import functional
from .modules import (
    # ... (previous imports)
    TransformerBlock # Add this line
)
```

### Next Steps

1.  **Create/update the files** with the code.
2.  **Implement the test adapter** `adapters.run_transformer_block`.
3.  **Run the tests** with `uv run pytest -k test_transformer_block`.

You are now just one step away from the complete model. After this, all that's left is to stack these `TransformerBlock`s together, add the initial embedding and the final output layer, and you will have a full `TransformerLM`. Let's finish this!


