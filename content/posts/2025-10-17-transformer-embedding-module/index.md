---
title: Transformer From Scratch — Embedding Module (Step-by-Step)
date: 2025-10-17
author: Sagar Desai
categories: [LLM]
tags: [transformers, pytorch, cs336]
series: ["Transformer From Scratch"]
---

3.4.3 Embedding Module

implentation now

Excellent. Let's apply the exact same professional design pattern to the `Embedding` module.

-----

### \#\# 1. The Big Picture: What is an Embedding Layer?

An **embedding layer** is essentially a lookup table. Its job is to map discrete, categorical items (like words or tokens) into continuous, dense vectors.

Why do we need this? A language model can't do math on token IDs like `5`, `87`, and `1024`. These are just arbitrary numbers. The model needs a richer, more meaningful representation.

The embedding layer stores a unique vector for every token in our vocabulary. When we feed it a token ID, it simply looks up and returns the corresponding vector. These vectors are **learnable parameters**. During training, the model learns to place tokens with similar meanings closer together in this high-dimensional "embedding space."

  * **Input:** A tensor of integer token IDs, e.g., `[5, 87, 1024]`.
  * **Embedding Matrix:** A large learnable matrix of shape `(vocab_size, d_model)`. Row `i` of this matrix is the embedding vector for token ID `i`.
  * **Output:** A tensor of dense vectors, where each token ID has been replaced by its corresponding vector from the matrix.

-----

### \#\# 2. The Blueprint: Functional vs. Module

Just like with the `Linear` layer, we'll split the implementation into two parts:

1.  **`functional.py`:** A stateless function, `embedding`, that performs the core lookup operation. It will take the input token IDs and the embedding matrix as arguments and return the resulting vectors.
2.  **`modules/embedding.py`:** A stateful `nn.Module` class, `Embedding`, that creates, initializes, and stores the large embedding matrix as an `nn.Parameter`. Its `forward` method will call the functional version.

-----

### \#\# 3. The Code: The Kernel in `functional.py`

The core operation is an indexing or "lookup" into the weight matrix. PyTorch makes this incredibly simple. If you have a matrix `W` and a tensor of indices `ids`, you can just do `W[ids]`.

**File:** `cs336_basics/nn/functional.py` (add this new function)

```python
import torch

# ... keep the existing linear function ...

def embedding(input_ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Performs a lookup in an embedding matrix.
    This is a stateless function.

    Args:
        input_ids (torch.Tensor): A tensor of integer token IDs of shape (...).
        weight (torch.Tensor): The embedding matrix of shape (vocab_size, embedding_dim).

    Returns:
        torch.Tensor: The looked-up embedding vectors, shape will be (..., embedding_dim).
    """
    # PyTorch's indexing is highly optimized for this exact operation.
    return weight[input_ids]
```

-----

### \#\# 4. The Code: The Module in `modules/embedding.py`

Now for the stateful wrapper. It will look very similar to our `Linear` class, but it will create the embedding matrix and use the initialization specified for it in the PDF (truncated normal with $\mu=0, \sigma^2=1$).

**File:** `cs336_basics/nn/modules/embedding.py` (a new file)

```python
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

# Import our functional kernel
from .. import functional as F

class Embedding(nn.Module):
    """
    A stateful wrapper for the stateless embedding functional kernel.
    This module creates, initializes, and stores the learnable embedding matrix.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings # vocab_size
        self.embedding_dim = embedding_dim   # d_model

        # Create the embedding matrix as a learnable parameter
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))

        # Initialize the parameter
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initializes the embedding matrix using a truncated normal distribution
        as specified in the assignment (Section 3.4.1).
        """
        # For embeddings, the spec is N(0, 1) truncated at [-3, 3].
        std = 1.0
        trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        In the forward pass, call the stateless functional implementation,
        passing it the input token IDs and our stored weight parameter.
        """
        return F.embedding(token_ids, self.weight)

    def extra_repr(self) -> str:
        return f'{self.num_embeddings}, {self.embedding_dim}'

```

-----

### \#\# 5. Tying It Together

Finally, update your `__init__.py` files to expose the new `Embedding` module.

**File:** `cs336_basics/nn/modules/__init__.py`

```python
from .linear import Linear
from .embedding import Embedding # Add this line
```

**File:** `cs336_basics/nn/__init__.py`

```python
from . import functional
from .modules import Linear, Embedding # Add Embedding here
```

And you're done!

### \#\# Next Steps

1.  **Create/update the files** with the code above.
2.  **Implement the test adapter** in `adapters.py` for `run_embedding`.
3.  **Run the tests** using `uv run pytest -k test_embedding`.

Once you pass the tests, we'll be ready to tackle the first component of the Transformer block itself: **RMSNorm**. Let me know when you're ready to proceed.


