---
title: Transformer From Scratch — Scaled Dot-Product Attention
date: 2025-10-17
author: Sagar Desai
categories: [LLM]
tags: [transformers, pytorch, cs336, attention]
---

{{< katex >}}

Problem (scaled_dot_product_attention): Implement scaled dot-product attention
(5 points)
Deliverable: Implement the scaled dot-product attention function. Your implementation should
handle keys and queries of shape (batch_size, ..., seq_len, d_k) and values of shape
(batch_size, ..., seq_len, d_v), where... represents any number of other batch-like
dimensions (if provided). The implementation should return an output with the shape (batch_size,
..., d_v). See section 3.3 for a discussion on batch-like dimensions.
Your implementation should also support an optional user-provided boolean mask of shape (seq_len,
seq_len). The attention probabilities of positions with a mask value of True should collectively sum
to 1, and the attention probabilities of positions with a mask value of False should be zero.
To test your implementation against our provided tests, you will need to implement the test adapter
at [adapters.run_scaled_dot_product_attention].
uv run pytest -k test_scaled_dot_product_attention tests your implementation on third-order
input tensors, while uv run pytest -k test_4d_scaled_dot_product_attention tests your
implementation on fourth-order input tensors.

Of course. We've built all the necessary components, and now it's time to assemble them into the main event: **Scaled Dot-Product Attention**. This is the core mechanism that allows a Transformer to weigh the importance of different tokens in a sequence when producing a representation for a specific token.

-----

### \#\# 1. The Big Picture: Attention as a Retrieval System

You can think of attention as a sophisticated, soft-retrieval system. It operates on three inputs: **Queries (Q)**, **Keys (K)**, and **Values (V)**.

  * **Query:** Represents a question or a "what I'm looking for" vector from a specific token.
  * **Key:** Represents a "label" or "what I contain" vector for every token in the sequence.
  * **Value:** Represents the actual content or information of every token in the sequence.

The attention mechanism works in two main steps:

1.  **Scoring:** For a given **Query**, it compares it to every **Key** in the sequence to compute a similarity score. The primary method for this comparison is the dot product. This tells us how "relevant" each key (and its associated value) is to the query.
2.  **Retrieval:** It uses these scores to compute a weighted average of all the **Value** vectors. Tokens with higher scores get a larger weight, meaning their information contributes more to the final output.

[cite_start]The "Scaled" part of the name comes from a crucial detail: we divide the scores by the square root of the key dimension ($\sqrt{d_k}$) before the final softmax step[cite: 699]. [cite_start]This prevents the dot products from becoming too large, which would lead to tiny gradients and stall the training process[cite: 699].

-----

### \#\# 2. The Implementation: A Stateless Functional Kernel

Scaled dot-product attention is a pure computation, so it belongs in our `functional.py` file. The function will precisely follow the formula from the assignment and handle the optional masking.

[cite_start]**A note on masking**: The assignment states that a `False` value in the mask means the query should *not* attend to the key[cite: 704]. [cite_start]To achieve this, we set the corresponding pre-softmax score to a very large negative number (effectively $-\infty$)[cite: 708]. [cite_start]When the `softmax` function exponentiates this number, it becomes zero, ensuring it gets no probability mass[cite: 715].

**File:** `cs336_basics/nn/functional.py` (add this new function)

```python
import torch
import math

# ... keep existing functions ...

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
):
    """
    Computes scaled dot-product attention as a stateless function.
    Formula: softmax( (Q @ K.T) / sqrt(d_k) ) @ V

    Args:
        query (torch.Tensor): Query tensor of shape (..., seq_len_q, d_k).
        key (torch.Tensor): Key tensor of shape (..., seq_len_k, d_k).
        value (torch.Tensor): Value tensor of shape (..., seq_len_k, d_v).
        mask (torch.Tensor, optional): Boolean mask of shape (..., seq_len_q, seq_len_k).
                                       If a value is False, the corresponding attention
                                       score is set to -inf. Defaults to None.

    Returns:
        torch.Tensor: The output of the attention mechanism, of shape (..., seq_len_q, d_v).
    """
    # d_k is the dimension of the key/query vectors
    d_k = query.size(-1)

    # 1. Compute raw scores with a matrix multiplication (Q @ K.T)
    # einsum is great here for handling arbitrary batch dimensions.
    # "... q d, ... k d -> ... q k"
    scores = torch.einsum("...qd,...kd->...qk", query, key)

    # 2. Scale the scores
    scaled_scores = scores / math.sqrt(d_k)

    # 3. Apply the mask (if provided)
    if mask is not None:
        # We need to ensure the mask can be broadcast to the scores' shape.
        # This is usually handled by how the mask is constructed, but a view can make it robust.
        # For a mask of shape (T, T), we might need to add batch/head dimensions.
        while mask.dim() < scaled_scores.dim():
            mask = mask.unsqueeze(0)
        
        # Set scores to a very large negative number where the mask is False
        scaled_scores = scaled_scores.masked_fill(mask == False, -torch.finfo(scaled_scores.dtype).max)

    # 4. Compute attention weights using softmax
    # The softmax is applied on the last dimension (the keys).
    attention_weights = softmax(scaled_scores, dim=-1)

    # 5. Compute the weighted sum of values
    # "... q k, ... k v -> ... q v"
    output = torch.einsum("...qk,...kv->...qv", attention_weights, value)
    
    return output
```

-----

### \#\# 3. Tying It Together

Like `softmax`, this is a pure function and doesn't require a stateful `nn.Module` wrapper. It will be a key building block for our upcoming `MultiHeadSelfAttention` module.

### \#\# Next Steps

1.  **Add the `scaled_dot_product_attention` function** to your `cs336_basics/nn/functional.py` file.
2.  **Implement the test adapter** `adapters.run_scaled_dot_product_attention`.
3.  **Run the tests**. The assignment mentions two specific tests to check your implementation's ability to handle different numbers of batch dimensions:
      * `uv run pytest -k test_scaled_dot_product_attention`
      * `uv run pytest -k test_4d_scaled_dot_product_attention`

After passing these tests, you will be ready to build the final piece of the attention sub-layer: the **Causal Multi-Head Self-Attention** module. This module will use `Linear` layers to create the Q, K, and V projections, `RoPE` to inject position info, and this `scaled_dot_product_attention` function to do the heavy lifting. Let me know when you're ready!


