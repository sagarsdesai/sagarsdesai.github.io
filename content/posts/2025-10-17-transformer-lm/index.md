---
title: Transformer From Scratch — Transformer Language Model (TransformerLM)
date: 2025-10-17
author: Sagar Desai
categories: [LLM]
tags: [transformers, pytorch, cs336, transformer-lm]
series: ["Transformer From Scratch"]
---

great, lets start wiht transformerLM

Problem (transformer_lm): Implementing the Transformer LM (3 points)
Time to put it all together! Implement the Transformer language model as described in §3.1
and illustrated in Figure 1. At minimum, your implementation should accept all the aforementioned
construction parameters for the Transformer block, as well as these additional parameters:
vocab_size: embedding matrix.
int The size of the vocabulary, necessary for determining the dimensionality of the token
context_length: int The maximum context length, necessary for determining the dimensionality of
the position embedding matrix.
num_layers: int The number of Transformer blocks to use.
To test your implementation against our provided tests, you will first need to implement the test
adapter at [adapters.run_transformer_lm]. Then, run uv run pytest -k test_transformer_lm
to test your implementation.
Deliverable: A Transformer LM module that passes the above tests.

This is it\! Time to put all the components together into the final `TransformerLM`.

[cite_start]This module acts as the master assembler, following the high-level architecture from Figure 1[cite: 355]. It will chain together the `Embedding` layer, our stack of `TransformerBlock`s, and the final `RMSNorm` and `Linear` layers to produce the model's output.

### 1. The Big Picture: The Full Model Architecture

[cite_start]As shown in Figure 1, the data flow is a clean, top-to-bottom pipeline[cite: 355]:

1.  [cite_start]**Token Embedding:** The input token IDs are converted into dense vectors[cite: 369].
2.  [cite_start]**Transformer Blocks:** These vectors are processed by a stack of `num_layers` identical `TransformerBlock`s[cite: 375]. The output of one block becomes the input for the next.
3.  [cite_start]**Final Norm:** After the final block, a single `RMSNorm` is applied[cite: 380].
4.  [cite_start]**LM Head:** A final `Linear` layer projects the refined vectors (of shape `d_model`) up to the full vocabulary size (`vocab_size`)[cite: 366]. This produces the raw, unnormalized scores (logits) for every possible next token.

[cite_start]Note that we **do not apply softmax** in the model's forward pass[cite: 336]. The cross-entropy loss function (which we'll implement later) is much more numerically stable when it receives the raw logits directly.

### 2. The Code: The `TransformerLM` Module

We'll create this in the same file as our `TransformerBlock`, as they are closely related. This module's `__init__` will be responsible for constructing all the sub-modules, including a `torch.nn.ModuleList` to hold all the `TransformerBlock`s.

**File:** `cs336_basics/nn/modules/transformer.py` (Add this class to the file)

```python
import torch
import torch.nn as nn

from .attention import MultiHeadSelfAttention
from .ffn import SwiGLUFFN
from .rmsnorm import RMSNorm
from .embedding import Embedding  # Import Embedding
from .linear import Linear      # Import Linear

# ... (Keep the existing TransformerBlock class here) ...

class TransformerLM(nn.Module):
    """
    A complete Transformer Language Model.
    This module stacks Transformer blocks and adds the embedding layers.
    """
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
        device=None,
        dtype=None
    ):
        super().__init__()
        
        # 1. Token Embedding Layer
        self.token_embedding = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype
        )

        # 2. Stack of Transformer Blocks
        # We use nn.ModuleList to properly register all blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    rope_theta=rope_theta,
                    device=device,
                    dtype=dtype
                )
                for _ in range(num_layers)
            ]
        )

        # 3. Final Normalization Layer
        self.final_norm = RMSNorm(d_model, device=device, dtype=dtype)

        # 4. Output Embedding (LM Head)
        self.lm_head = Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass for the entire Transformer LM.
        Ref: Figure 1 in the assignment PDF.
        """
        batch_size, seq_len = token_ids.shape
        
        # 1. Get token embeddings
        x = self.token_embedding(token_ids)
        
        # We need token positions for RoPE. We can generate them on the fly.
        # Shape: (seq_len,)
        token_positions = torch.arange(seq_len, device=token_ids.device)

        # 2. Pass through all Transformer blocks
        for block in self.blocks:
            x = block(x, token_positions)
            
        # 3. Apply final normalization
        x = self.final_norm(x)
        
        # 4. Apply LM Head to get logits
        logits = self.lm_head(x)
        
        return logits

```

### 3. Tying It Together

Finally, let's expose the new `TransformerLM` module.

**File:** `cs336_basics/nn/modules/__init__.py`

```python
# ... (previous imports)
from .transformer import TransformerBlock, TransformerLM # Add TransformerLM
```

**File:** `cs336_basics/nn/__init__.py`

```python
from . import functional
from .modules import (
    # ... (previous imports)
    TransformerBlock,
    TransformerLM # Add TransformerLM
)
```

### Next Steps

1.  **Update the files** with the new `TransformerLM` class.
2.  **Implement the test adapter** `adapters.run_transformer_lm`.
3.  **Run the tests** with `uv run pytest -k test_transformer_lm`.

Once this test passes, you have successfully built an entire Transformer language model from scratch. 🥳 This is a massive milestone.

The next sections of the assignment involve calculating the parameters and FLOPs for your model, then building the training harness (loss function, optimizer, and training loop). Let me know when you're ready to tackle the "Resource accounting" problems.


