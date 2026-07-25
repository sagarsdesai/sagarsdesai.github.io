---
title: Transformer From Scratch — Softmax Implementation
date: 2025-10-17
author: Sagar Desai
categories: [LLM]
tags: [transformers, pytorch, cs336, softmax]
---

{{< katex >}}

Problem (softmax): Implement softmax (1 point)
Deliverable: Write a function to apply the softmax operation on a tensor. Your function should
take two parameters: a tensor and a dimension i, and apply softmax to the i-th dimension of the input
tensor. The output tensor should have the same shape as the input tensor, but its i-th dimension will
now have a normalized probability distribution. Use the trick of subtracting the maximum value in
the i-th dimension from all elements of the i-th dimension to avoid numerical stability issues.
To test your implementation, complete [adapters.run_softmax] and make sure it passes uv run
pytest -k test_softmax_matches_pytorch.

Absolutely. Before we can implement the full attention mechanism, we need its final component: the **softmax** function. This is the crucial step that converts the raw attention scores into a valid probability distribution.

Let's build it, focusing on the numerical stability trick mentioned in the assignment.

-----

### \#\# 1. The Big Picture: What is Softmax and Why Does it Need a "Trick"?

The softmax function's job is to take a vector of arbitrary real numbers (called "logits" or "scores") and transform them into a probability distribution, where:

1.  All output values are between 0 and 1.
2.  All output values sum up to 1.

The formula is:
$$softmax(v)_i = \frac{e^{v_i}}{\sum_{j=1}^{n} e^{v_j}}$$

**The Problem:** Computers have a limited range for floating-point numbers. If you have a large logit, say `v_i = 1000`, then `e^1000` is an astronomically huge number that the computer can't store. It results in `inf` (infinity). If your vector has multiple large values, you end up with an `inf / inf` calculation, which results in `NaN` (Not a Number), and your model's training will collapse.

**The Stability Trick:** A beautiful mathematical property of softmax is its invariance to shifting. You can add or subtract any constant `c` from every element in the input vector without changing the result: `softmax(v) = softmax(v - c)`.

To prevent overflow, we can choose a clever `c`: the **maximum value** in the vector `v`.

1.  We calculate `v' = v - max(v)`.
2.  Now, the largest value in `v'` is 0. The largest exponent we'll have to compute is `e^0 = 1`.
3.  All other values in `v'` will be negative, so their exponents `e^{negative_number}` will be small positive numbers between 0 and 1.
4.  This completely avoids the possibility of overflow, making the computation numerically stable.

-----

### \#\# 2. The Implementation: A Stateless Functional Kernel

Since softmax is a pure, stateless operation, it belongs in our `functional.py` file. The implementation will follow the stability trick described above.

**File:** `cs336_basics/nn/functional.py` (add this new function)

```python
import torch

# ... keep existing functions ...

def softmax(input: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Applies a numerically stable softmax function.
    This is a stateless function.

    Args:
        input (torch.Tensor): The input tensor of logits.
        dim (int): The dimension along which softmax is computed.

    Returns:
        torch.Tensor: Tensor of probabilities, same shape as input.
    """
    # 1. Subtract the max for numerical stability
    # We use keepdim=True to ensure the result is broadcastable
    max_vals, _ = torch.max(input, dim=dim, keepdim=True)
    shifted_logits = input - max_vals

    # 2. Exponentiate
    exps = torch.exp(shifted_logits)

    # 3. Sum the exponents and divide
    sum_exps = torch.sum(exps, dim=dim, keepdim=True)
    
    return exps / sum_exps
```

-----

### \#\# 3. Tying It Together

There is no stateful `nn.Module` for softmax, as it has no learnable parameters. It's a pure function. We just need to make it easily accessible. You can optionally expose it from `cs336_basics/nn/__init__.py` if you plan to call it frequently from outside the `nn` package, but it's primarily a building block for other functional kernels like the upcoming scaled dot-product attention.

### \#\# Next Steps

1.  **Add the `softmax` function** to your `cs336_basics/nn/functional.py` file.
2.  **Implement the test adapter** `adapters.run_softmax`.
3.  **Run the tests** with `uv run pytest -k test_softmax_matches_pytorch`.

Once this is done, you are fully equipped to build the **Scaled Dot-Product Attention** function. This will be the first time we combine several of our building blocks (`linear`, `softmax`, `rope`) into a single, powerful operation. Let me know when you're ready!


