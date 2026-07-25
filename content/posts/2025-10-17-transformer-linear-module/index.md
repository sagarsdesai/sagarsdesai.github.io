---
title: Transformer From Scratch — Linear Module (Step-by-Step)
date: 2025-10-17
author: Sagar Desai
categories: [LLM]
tags: [transformers, pytorch, cs336]
series: ["Transformer From Scratch"]
---

{{< katex >}}

Q -you are expert llm training engineering working at large scale production grade multli node large cluster training org, also you are great teacher to simplyfy things, teach by hands on , top down approach

https://github.com/SDcodehub/assignment1-basics, my repository, get all the files and structure in the file.

till tokeniser everything is done, refer to the files names from the repo.

now I want to move to the transformer language coding.

move step by step, do not give me full code, move as per the outline of the pdf.

I think linear module implentation seems to be good starting point, correct if I am wrong.

lets start with this model, help me understad basics and move to final code of linear module, I have basic understanding of pytorch

Of course\! You're absolutely on the right track. Starting with the `Linear` module is the perfect first step. It's the fundamental building block for almost every part of the Transformer, from the feed-forward networks to the attention mechanism itself.

Let's break it down, top-down, just like you asked.

### \#\# 1. The Big Picture: What is a Linear Layer?

At its core, a **linear layer** (or "fully connected" layer) is one of the simplest and most important components in any neural network. Its job is to perform a linear transformation on the input data.

Think of it like a function $y = Wx$. It takes an input vector $x$ and multiplies it by a weight matrix $W$ to produce an output vector $y$.

  * **Input (`x`):** A vector representing some data. In our Transformer, this will be a vector representing a token at a specific position in the sequence, with dimension `d_model`.
  * **Weights (`W`):** This is a matrix of **learnable parameters**. This is the "brain" of the layer. During training, the optimizer will tweak the values in this matrix to help the model make better predictions.
  * **Output (`y`):** The transformed vector. The layer "projects" the input from its original dimensional space to a new one. For example, it might take a vector of size 512 and project it into a space of size 2048.

[cite_start]The PDF also mentions that we'll be implementing it **without a bias term** ($y = Wx$ instead of $y = Wx + b$)[cite: 523]. This is a common practice in modern large language models (LLMs) like LLaMA, as other layers (like normalization layers) can handle shifting the activations, making the explicit bias term redundant.

### \#\# 2. The Blueprint: Coding a `Linear` Module in PyTorch

To implement this in PyTorch, we'll create a class that inherits from `torch.nn.Module`. This is the standard way to build any custom layer. Inheriting from `torch.nn.Module` gives us a ton of useful functionality for free, like tracking parameters, moving them between CPU/GPU, and saving/loading the model state.

Our class will need two key methods:

1.  `__init__(self, ...)`: The constructor. This is where we'll **define and initialize** our weight matrix `W`.
2.  `forward(self, x)`: This is where the action happens. We'll take an input tensor `x` and perform the actual matrix multiplication using our weight matrix `W`.

-----

### \#\# 3. The Details: `__init__` and Parameter Initialization

Let's look at the constructor first. It needs to create our learnable weight matrix `W`.

  * **Shape:** If the input vector has size `in_features` and we want the output vector to have size `out_features`, our weight matrix `W` needs to have the shape `(out_features, in_features)`.
  * **`torch.nn.Parameter`:** We can't just use a regular `torch.Tensor` for our weights. We need to wrap it in `torch.nn.Parameter`. [cite_start]This is a special wrapper that tells PyTorch, "Hey, this tensor is a learnable parameter. When it's time to train, make sure the optimizer updates this\!"[cite: 541].
  * **Initialization:** How we set the initial values of `W` is surprisingly important. Bad initial values can cause the model to fail to train. [cite_start]The assignment gives us a specific recipe: a truncated normal distribution[cite: 515, 543]. This helps ensure that the signals flowing through our network at the beginning of training are in a healthy range (not too big, not too small). We'll use `torch.nn.init.trunc_normal_` for this.

### \#\# 4. The Details: `forward` and Matrix Multiplication

The `forward` method is where we implement the equation $y=Wx$.

In PyTorch, a standard matrix multiplication is done with the `@` operator or `torch.matmul`. Our input `x` will have a shape like `(batch_size, sequence_length, in_features)`. Our weight matrix `W` has the shape `(out_features, in_features)`.

To make the dimensions line up for multiplication, we need to multiply `x` by the **transpose** of `W`. The operation will be `x @ self.W.T`.

[cite_start]The PDF also strongly encourages using `torch.einsum`[cite: 401]. It's a very powerful and self-documenting way to perform tensor operations. For our linear layer, it would look like this:

`y = torch.einsum("... d_in, d_out d_in -> ... d_out", x, self.W)`

This notation says:

  * Take the input `x`, which has some leading batch dimensions (`...`) and a final dimension `d_in`.
  * Take the weight matrix `W`, with dimensions `d_out` and `d_in`.
  * Multiply them by contracting (summing over) the common `d_in` dimension.
  * The output `y` will have the same leading batch dimensions (`...`) and a final dimension of `d_out`.

It's a great habit to get into, as it makes your code clearer and less prone to dimension-shuffling bugs.

-----

### \#\# 5. The Final Code

Okay, let's put all of that together. Based on your repository structure, a good place for this would be a new file: `cs336_basics/model.py`.

Here is the complete, commented code for the `Linear` module.

```python
# In cs336_basics/model.py

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

class Linear(nn.Module):
    """
    Implementation of a Linear layer from scratch, as required by the assignment.
    This module performs a linear transformation on the input data: y = xW^T.
    It does not include a bias term, following modern LLM practices.
    """
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        """
        Initializes the Linear module.

        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            device (torch.device, optional): The device to store parameters on. Defaults to None.
            dtype (torch.dtype, optional): The data type of the parameters. Defaults to None.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 1. Create the weight matrix as a torch.nn.Parameter.
        # This tells PyTorch that it's a learnable parameter.
        # The shape is (out_features, in_features) which is the standard PyTorch convention.
        self.W = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))

        # 2. Initialize the weights according to the assignment's specification.
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initializes the weight matrix using a truncated normal distribution.
        This specific initialization is crucial for stable training in pre-norm Transformers.
        Ref: Section 3.4.1 in the assignment PDF.
        """
        # The formula for standard deviation is sqrt(2 / (d_in + d_out))
        std = (2 / (self.in_features + self.out_features))**0.5
        # Truncate at 3 standard deviations.
        trunc_normal_(self.W, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the linear transformation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (..., in_features).

        Returns:
            torch.Tensor: Output tensor of shape (..., out_features).
        """
        # The assignment recommends using einsum for clarity and robustness.
        # "... d_in" means the input tensor can have any number of leading (batch) dimensions.
        # "d_out d_in" is the shape of our weight matrix.
        # "-> ... d_out" specifies the output shape, preserving batch dimensions.
        return torch.einsum("... d_in, d_out d_in -> ... d_out", x, self.W)

    def extra_repr(self) -> str:
        """
        Provides a string representation of the module's configuration,
        similar to PyTorch's built-in layers.
        """
        return f'in_features={self.in_features}, out_features={self.out_features}, bias=False'

```

### \#\# Next Steps

1.  **Add the code** to your `cs336_basics/model.py` file.
2.  [cite_start]**Implement the test adapter** in `adapters.py` to hook up your new `Linear` class to the test suite, as described in section 3.4.2 of the PDF[cite: 544].
3.  **Run the tests** using `uv run pytest -k test_linear`.

[cite_start]Once you've passed the tests for the `Linear` module, the next logical step is implementing the **`Embedding` module**[cite: 552]. It's very similar in spirit—it's a simple lookup table, also implemented as a `torch.nn.Module`.

Let me know when you're ready to move on to that\!


