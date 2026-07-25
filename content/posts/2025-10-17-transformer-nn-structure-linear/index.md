---
title: Transformer NN Structure — Functional vs Module (Linear)
date: 2025-10-17
author: Sagar Desai
categories: [LLM]
tags: [transformers, pytorch, architecture]
---

Q - overall folder structure of my repo  tilll the nn module

cs336_basics/
  __init__.py

  nn/
    __init__.py            # expose: Linear, Embedding, functional, etc.
    functional.py          # stateless kernels: linear, softmax, silu, rmsnorm, sdpa, rope, embedding_lookup
    modules/
      __init__.py
      linear.py            # class Linear
  tokenizer/
    __init__.py            # expose: Tokenizer, train_bpe
    bpe/
      __init__.py
      core.py              # Tokenizer: encode/decode/encode_iterable, special tokens
      pretokenize.py       # GPT-2 regex + bytes mapping
      training.py          # train_bpe algorithm (pure functions)
      serialization.py     # load/save vocab & merges
      types.py             # small dataclasses / type aliases
    cli/
      __init__.py
      tokenize_dataset.py
      compute_bytes_per_token.py
  utils/
    __init__.py
    logging_config.py

specially note the linear attention part, the functional and wrapper seperator makes better sense

  nn/
    __init__.py            # expose: Linear, Embedding, functional, etc.
    functional.py          # stateless kernels: linear, softmax, silu, rmsnorm, sdpa, rope, embedding_lookup
    modules/
      __init__.py
      linear.py            # class Linear

can you help me with makeing coe this fasion, ask follow up questions.

Excellent\! That's a fantastic and professional way to structure your code.

Separating the **stateless computational logic** (the "kernel" in `functional.py`) from the **stateful module** (the `class Linear` that holds the parameters) is a hallmark of well-designed ML libraries, including PyTorch itself. It makes your code more modular, reusable, and easier to test. I'm impressed\!

Let's adapt the code to fit this superior structure. I have a couple of quick questions to make sure we're perfectly aligned.

### \#\# Follow-up Questions

1.  **Functional Signature:** For the function in `cs336_basics/nn/functional.py`, I assume you want a pure, stateless function. It would take the input tensor and the weight tensor as direct arguments. Does this signature look right to you?
    ```python
    # In cs336_basics/nn/functional.py
    def linear(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # ... implementation ...
    ```
2.  **Initialization Logic:** The `Linear` class is still responsible for creating and holding the `nn.Parameter` for the weights. Where would you prefer the specific weight initialization logic (the `trunc_normal_` call) to live? Inside the `Linear` class's `__init__` method is the most common pattern, and I think it makes the most sense here.

Assuming the answers are "yes" to both, here is how we can implement this two-part design.

-----

### \#\# 1. The Kernel: `functional.py`

First, we'll create the pure function that just does the math. It doesn't know anything about classes or parameters; it just executes the tensor operation.

**File:** `cs336_basics/nn/functional.py`

```python
import torch

def linear(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Applies a linear transformation to the incoming data using a provided weight matrix.
    This is a stateless function.

    Args:
        input (torch.Tensor): Input tensor of shape (..., in_features).
        weight (torch.Tensor): Weight matrix of shape (out_features, in_features).

    Returns:
        torch.Tensor: Output tensor of shape (..., out_features).
    """
    # Using einsum as it's explicit and handles broadcasting over batch dimensions automatically.
    # This is the core computation of the linear layer.
    return torch.einsum("... d_in, d_out d_in -> ... d_out", input, weight)

# You will add other stateless functions like softmax, rmsnorm, etc., to this file later.
```

This function is now a clean, reusable building block. 🧱

-----

### \#\# 2. The Module: `modules/linear.py`

Next, we'll write the `nn.Module` class. This class acts as a "wrapper" or "container." Its main jobs are to **hold the state** (the learnable `W` parameter) and to call our functional kernel in its `forward` pass.

**File:** `cs336_basics/nn/modules/linear.py`

```python
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

# We import our functional kernel. It's common practice to alias it as F.
from .. import functional as F

class Linear(nn.Module):
    """
    A stateful wrapper for the stateless linear functional kernel.
    This module creates, initializes, and stores the learnable weight parameter.
    """
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Create the weight matrix as a learnable parameter
        self.W = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))

        # Initialize the parameter
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initializes the weight matrix using a truncated normal distribution as
        specified in the assignment (Section 3.4.1).
        """
        std = (2 / (self.in_features + self.out_features))**0.5
        trunc_normal_(self.W, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        In the forward pass, we simply call our stateless functional implementation,
        passing it the input tensor `x` and our stored weight parameter `self.W`.
        """
        return F.linear(x, self.W)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias=False'
```

-----

### \#\# 3. Tying It Together: The `__init__.py` Files

Finally, to make importing clean and easy, you'll update your `__init__.py` files to "expose" the `Linear` class at the `cs336_basics.nn` level.

**File:** `cs336_basics/nn/modules/__init__.py`

```python
from .linear import Linear
# You'll add other modules here later, like Embedding, RMSNorm, etc.
```

**File:** `cs336_basics/nn/__init__.py`

```python
from . import functional
from .modules import Linear, Embedding # Add Embedding later
```

With this setup, you can now import your shiny new module in other files like this:
`from cs336_basics.nn import Linear`

This is a robust and scalable design. How does this look to you? Once you give the green light, we can move on to the `Embedding` module following the same excellent pattern. 👍


