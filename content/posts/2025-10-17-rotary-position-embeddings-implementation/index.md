---
title: Transformer From Scratch — Rotary Position Embeddings (RoPE) Implementation
date: 2025-10-17
author: Sagar Desai
categories: [LLM]
tags: [transformers, pytorch, cs336, rope]
---

{{< katex >}}

Problem (rope): Implement RoPE (2 points)
Deliverable: Implement a class RotaryPositionalEmbedding that applies RoPE to the input
tensor. The following interface is recommended:
def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None) Construct
the RoPE module and create buffers if needed.
theta: float Θ value for the RoPE
d_k: int dimension of query and key vectors
max_seq_len: int Maximum sequence length that will be inputted
device: torch.device | None = None Device to store the buffer on
def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor
Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape.
Note that you should tolerate x with an arbitrary number of batch dimensions. You should
assume that the token positions are a tensor of shape (..., seq_len) specifying the token
positions of x along the sequence dimension.
You should use the token positions to slice your (possibly precomputed) cos and sin tensors
along the sequence dimension.
To test your implementation, complete [adapters.run_rope] and make sure it passes uv run
pytest -k test_rope.


Of course. Let's tackle **Rotary Position Embeddings (RoPE)**. This is a clever and elegant way to inform the model about the order of tokens in a sequence.

-----

### \#\# 1. The Big Picture: What is RoPE?

Traditional position embeddings add a unique vector to each token's embedding. RoPE takes a different approach: instead of adding positional information, it **rotates** each token's vector by an amount that depends on its position.

Imagine each token's embedding vector is broken down into pairs of numbers. RoPE treats each pair as a 2D point $(x, y)$ and rotates it on a 2D plane. The key idea is that the **angle of rotation is different for each position** in the sequence.

Why is this a good idea? The dot product between two rotated vectors depends only on their original orientation and the *difference* in their rotation angles. In the context of self-attention, where the model calculates similarity using dot products between query and key vectors, this means that the attention score between two tokens will naturally depend on their **relative positions**. It's a very direct and effective way to inject this crucial information into the model's core mechanism.

[cite_start]The rotation angle for a token at position $i$ and for the $k$-th pair of dimensions is given by the formula[cite: 656]:
$$\theta_{i,k} = \frac{i}{\Theta^{(2k-2)/d}}$$
Here, $\Theta$ is a large constant (typically 10000), and $d$ is the dimension of the vector.

-----

### \#\# 2. The Implementation: Pre-computation and Buffers

[cite_start]Since the rotation angles only depend on the position $i$ and the dimension index $k$, and not on the token's actual content, we can **pre-compute** the sine and cosine values for all possible positions up to `max_seq_len`[cite: 663]. This is a huge efficiency win.

We will store these pre-computed sine and cosine values in **buffers** within our `nn.Module`. [cite_start]A buffer, created with `self.register_buffer()`, is a tensor that is part of the module's state (like parameters) but is **not** considered a learnable parameter by the optimizer[cite: 664]. It gets moved to the correct device (`.to('cuda')`) and saved with the model's state dictionary, which is exactly what we need for our pre-computed values.

-----

### \#\# 3. The Code: Functional Kernel

The stateless function will perform the actual rotation logic. It will take an input tensor `x` and the corresponding pre-computed sine and cosine values. The core trick is to reshape the input tensor so that the last dimension is split into pairs, apply the rotation, and then reshape it back.

**File:** `cs336_basics/nn/functional.py` (add this new function)

```python
import torch

# ... keep existing functions ...

def rope(
    input: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> torch.Tensor:
    """
    Applies Rotary Position Embeddings (RoPE) to the input tensor.
    This is a stateless function.

    Args:
        input (torch.Tensor): Input tensor of shape (..., seq_len, d_k).
        cos (torch.Tensor): Pre-computed cosine values of shape (..., seq_len, d_k).
        sin (torch.Tensor): Pre-computed sine values of shape (..., seq_len, d_k).

    Returns:
        torch.Tensor: Tensor with RoPE applied, of the same shape as input.
    """
    # Reshape input to view the last dimension as pairs of features
    # (..., seq_len, d_k) -> (..., seq_len, d_k/2, 2)
    x_pairs = input.unflatten(-1, (-1, 2))

    # Get the two components of each pair
    x1 = x_pairs[..., 0]
    x2 = x_pairs[..., 1]

    # Reshape sin and cos to match the paired shape
    cos = cos.unflatten(-1, (-1, 2))[..., 0]
    sin = sin.unflatten(-1, (-1, 2))[..., 0]

    # Apply the 2D rotation matrix formula:
    # y1 = x1*cos(theta) - x2*sin(theta)
    # y2 = x1*sin(theta) + x2*cos(theta)
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    
    # Stack the rotated pairs back together
    # (..., seq_len, d_k/2), (..., seq_len, d_k/2) -> (..., seq_len, d_k/2, 2)
    y_pairs = torch.stack((y1, y2), dim=-1)

    # Flatten the last two dimensions to restore the original shape
    # (..., seq_len, d_k/2, 2) -> (..., seq_len, d_k)
    return y_pairs.flatten(-2)
```

-----

### \#\# 4. The Code: Module Wrapper

The `nn.Module` class will handle the pre-computation and buffering of the sine and cosine values. Its `forward` method will then use the provided `token_positions` to slice the correct values from the buffer and pass them to our functional kernel.

**File:** `cs336_basics/nn/modules/rope.py` (a new file)

```python
import torch
import torch.nn as nn

from .. import functional as F

class RotaryPositionalEmbedding(nn.Module):
    """
    A stateful module for Rotary Position Embeddings (RoPE).
    This module pre-computes and buffers the sine and cosine values required for rotation.
    """
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, dtype=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # Pre-compute the inverse frequencies (thetas_k in the formula)
        # Shape: (d_k / 2)
        thetas_k = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device, dtype=dtype) / d_k))
        
        # Pre-compute the position indices
        # Shape: (max_seq_len)
        positions = torch.arange(max_seq_len, device=device, dtype=dtype)
        
        # Calculate the arguments for sin and cos
        # Shape: (max_seq_len, d_k / 2)
        angles = torch.outer(positions, thetas_k)
        
        # Repeat each angle value twice to match the d_k dimension
        # Shape: (max_seq_len, d_k)
        angles_repeated = torch.cat((angles, angles), dim=-1)

        # Register sin and cos values as non-learnable buffers
        self.register_buffer('cos_cached', angles_repeated.cos(), persistent=False)
        self.register_buffer('sin_cached', angles_repeated.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Applies RoPE to the input tensor using the pre-computed values.

        Args:
            x (torch.Tensor): Input tensor of shape (..., seq_len, d_k).
            token_positions (torch.Tensor): A tensor specifying the absolute positions
                                            of tokens in the sequence, shape (..., seq_len).

        Returns:
            torch.Tensor: Tensor with RoPE applied, same shape as x.
        """
        # Slice the pre-computed cos and sin tensors using the provided token positions.
        # This allows handling sequences shorter than max_seq_len and enables features
        # like key-value caching during inference.
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]
        
        # The input x can have arbitrary batch dimensions, so we need to make sure
        # cos and sin are broadcastable to the shape of x.
        # We add unsqueezed dimensions to cos/sin to match the batch dims of x.
        # Example: x shape (B, H, T, D), pos shape (B, T) -> cos/sin shape (B, 1, T, D)
        while cos.dim() < x.dim():
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        return F.rope(x, cos, sin)
```

-----

### \#\# 5. Tying It Together

Finally, update your `__init__.py` files as usual.

**File:** `cs336_basics/nn/modules/__init__.py`

```python
from .linear import Linear
from .embedding import Embedding
from .rmsnorm import RMSNorm
from .ffn import SwiGLUFFN
from .rope import RotaryPositionalEmbedding # Add this line
```

**File:** `cs336_basics/nn/__init__.py`

```python
from . import functional
from .modules import (
    Linear,
    Embedding,
    RMSNorm,
    SwiGLUFFN,
    RotaryPositionalEmbedding # Add this line
)
```

### \#\# Next Steps

1.  **Create/update the files** with the code.
2.  **Implement the test adapter** for `run_rope`.
3.  **Run the tests** with `uv run pytest -k test_rope`.

You've now built all the individual components of the Transformer block. The next step is to assemble them into the **Causal Multi-Head Self-Attention** layer. This will be the most complex part yet, but we have all the pieces ready. Let me know when you're prepared to move on.


