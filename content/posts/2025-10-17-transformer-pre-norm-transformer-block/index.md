---
date: 2025-10-17
title: "2025-10-17-transformer-pre-norm-transformer-block"
series: ["Transformer From Scratch"]
---

{{< katex >}}

Excellent. We are now diving into the heart of the model: the **Pre-Norm Transformer Block**.

[cite\_start]As the name suggests, the key idea here is that we apply **normalization *before*** each main operation (attention and the feed-forward network)[cite: 575]. [cite\_start]This is a crucial change from the original Transformer paper and is known to make training more stable[cite: 575].

[cite\_start]A single Transformer block has two main sub-layers, as shown in Figure 2 of the assignment[cite: 358]:

1.  Causal Multi-Head Self-Attention
2.  Position-Wise Feed-Forward Network

The very first component we need for both of these sub-layers is the normalization layer itself. [cite\_start]The assignment specifies using **Root Mean Square Layer Normalization (RMSNorm)**[cite: 582].

Let's build it.

-----

### \#\# 1. The Big Picture: What is RMSNorm?

Think of the activations flowing through your model as signals. If the numbers in these signals get too large or too small, it can make training unstable. Normalization acts like a volume control, rescaling the signals to a consistent, manageable range.

[cite\_start]**RMSNorm** is a simple and efficient way to do this[cite: 582]. For a given vector of activations $a$ (representing one token), it does two things:

1.  It calculates the vector's overall magnitude, or "energy," using the Root Mean Square formula.
2.  It divides the vector by this magnitude, effectively normalizing its scale.

[cite\_start]The formula is given by Equation 4 [cite: 584-587]:
$$RMSNorm(a) = \frac{a}{\sqrt{\frac{1}{d}\sum_{i=1}^{d}a_i^2 + \epsilon}} \cdot \gamma$$

  * **The fraction part** does the normalization. It scales the input vector $a$.
  * $\gamma$ (gamma) is a **learnable gain parameter**. After we've rescaled the vector, this learnable parameter allows the network to fine-tune the output's magnitude if needed. It gives the model back some flexibility.
  * [cite\_start]$\epsilon$ (epsilon) is a tiny number added to the denominator to prevent division by zero[cite: 587].

-----

### \#\# 2. The Implementation: Functional Kernel

Following our established pattern, we'll first write the stateless computation. [cite\_start]A critical detail from the PDF is to **upcast the input to `float32`** before squaring to avoid numerical overflow, especially when using lower-precision dtypes like `float16`[cite: 589].

**File:** `cs336_basics/nn/functional.py` (add this new function)

```python
import torch

# ... keep existing linear and embedding functions ...

def rms_norm(input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Applies Root Mean Square Layer Normalization.
    This is a stateless function.

    Args:
        input (torch.Tensor): Input tensor of shape (..., d_model).
        weight (torch.Tensor): Learnable gain parameter (gamma) of shape (d_model,).
        eps (float): A small value added for numerical stability.

    Returns:
        torch.Tensor: Normalized tensor of the same shape as input.
    """
    # Store original dtype to cast back at the end
    input_dtype = input.dtype
    # Upcast to float32 for stable computation of squares
    x = input.to(torch.float32)

    # Calculate the mean of the squares of the input along the last dimension
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    
    # Calculate the reciprocal of the square root (RMS)
    rsqrt = torch.rsqrt(variance + eps)
    
    # Normalize the input and apply the learnable gain (weight)
    normalized_x = x * rsqrt
    
    # Apply the gain and cast back to the original dtype
    return (weight * normalized_x).to(input_dtype)
```

-----

### \#\# 3. The Implementation: Module Wrapper

Now, the stateful module. Its job is to create, store, and initialize the learnable gain parameter `self.weight` ($\gamma$). [cite\_start]The PDF specifies that this gain should be **initialized to ones**[cite: 517].

**File:** `cs336_basics/nn/modules/rmsnorm.py` (a new file)

```python
import torch
import torch.nn as nn

# Import our functional kernel
from .. import functional as F

class RMSNorm(nn.Module):
    """
    A stateful wrapper for the stateless RMSNorm functional kernel.
    This module creates, initializes, and stores the learnable gain parameter.
    """
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        # Create the learnable gain parameter (gamma)
        self.weight = nn.Parameter(torch.empty(d_model, device=device, dtype=dtype))
        
        # Initialize the gain parameter
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initializes the gain parameter to ones, as specified in the PDF (Section 3.4.1).
        """
        nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calls the stateless functional implementation.
        """
        return F.rms_norm(x, self.weight, self.eps)

    def extra_repr(self) -> str:
        return f'{self.d_model}, eps={self.eps}'
```

-----

### \#\# 4. Tying It Together

Don't forget to update your `__init__.py` files to expose the new `RMSNorm` module.

**File:** `cs336_basics/nn/modules/__init__.py`

```python
from .linear import Linear
from .embedding import Embedding
from .rmsnorm import RMSNorm # Add this line
```

**File:** `cs336_basics/nn/__init__.py`

```python
from . import functional
from .modules import Linear, Embedding, RMSNorm # Add RMSNorm here
```

### \#\# Next Steps

1.  **Create/update the files** with the code.
2.  **Implement the test adapter** for `run_rmsnorm`.
3.  **Run the tests** with `uv run pytest -k test_rmsnorm`.

[cite\_start]Once that's working, we'll move on to the more complex part of the Transformer block: the **Position-Wise Feed-Forward Network**, which uses the SwiGLU activation[cite: 627]. Let me know when you're ready.