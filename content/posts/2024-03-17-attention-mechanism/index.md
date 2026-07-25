---
title: Attention Mechanism
date: 2024-03-17
author: Sagar Desai
categories: [LLM]
tags: [prod]
---

## Table of Contents
- [Table of Contents](#table-of-contents)
- [](#)
- [Python Code](#python-code)
- [MultiHeadAttention Class Definition](#multiheadattention-class-definition)
  - [Forward Method](#forward-method)
- [Reference](#reference)


---
![Multihead Attention Mechanism](\assets_files\blogs\2024-03-17-Attention-Mechanism\Multi_head_attention.png)
---

## Python Code
```python
1  class MultiHeadAttention(nn.Module):
2      def __init__(self, d_in, d_out, 
3                   block_size, dropout, num_heads, qkv_bias=False):
4          super().__init__()
5          assert d_out % num_heads == 0, "d_out must be divisible by n_heads"
6  
7          self.d_out = d_out
8          self.num_heads = num_heads
9          self.head_dim = d_out // num_heads
10         self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
11         self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
12         self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
13         self.out_proj = nn.Linear(d_out, d_out)
14         self.dropout = nn.Dropout(dropout)
15         self.register_buffer(
16             'mask',
17             torch.triu(torch.ones(block_size, block_size), diagonal=1)
18         )
19 
20     def forward(self, x):
21         b, num_tokens, d_in = x.shape
22         keys = self.W_key(x)
23         queries = self.W_query(x)
24         values = self.W_value(x)
25 
26         keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
27         values = values.view(b, num_tokens, self.num_heads, self.head_dim)
28         queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
29 
30         keys = keys.transpose(1, 2)
31         queries = queries.transpose(1, 2)
32         values = values.transpose(1, 2)
33 
34         attn_scores = queries @ keys.transpose(2, 3)
35         mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
36         mask_unsqueezed = mask_bool.unsqueeze(0).unsqueeze(0)
37         attn_scores.masked_fill_(mask_unsqueezed, -torch.inf)
38 
39         attn_weights = torch.softmax(
40             attn_scores / keys.shape[-1]**0.5, dim=-1)
41         attn_weights = self.dropout(attn_weights)
42 
43         context_vec = (attn_weights @ values).transpose(1, 2)
44 
45         context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
46         context_vec = self.out_proj(context_vec)
47         return context_vec
```


## MultiHeadAttention Class Definition
- **Line 1, Class Definition:**
  - Defines a class `MultiHeadAttention` as a subclass of `nn.Module`.
  - This class implements the multi-head attention mechanism.

- **Line 2, Constructor:**
  - Initializes the multi-head attention layer with specified parameters.
  - Parameters include input dimension (`d_in`), output dimension (`d_out`), block size for masking, dropout rate, number of attention heads (`num_heads`), and an optional bias for query, key, and value projections (`qkv_bias`).

- **Line 3, Superclass Initialization:**
  - Calls the constructor of the superclass `nn.Module` to properly initialize the class.

- **Line 4, Dimension Assertion:**
  - Ensures that the output dimension (`d_out`) is divisible by the number of heads (`num_heads`).
  - This is necessary to evenly distribute the dimensions across the heads.

- **Lines 6-14, Parameter Definitions:**
  - Sets up the internal parameters for the multi-head attention mechanism.
  - `self.d_out`: Output dimension.
  - `self.num_heads`: Number of attention heads.
  - `self.head_dim`: Dimension of each head, calculated as `d_out / num_heads`.
  - `self.W_query`, `self.W_key`, `self.W_value`: Linear layers for projecting inputs to query, key, and value spaces, respectively.
  - `self.out_proj`: Linear layer for projecting concatenated outputs from all heads.
  - `self.dropout`: Dropout layer to prevent overfitting.

- **Line 15-19, Mask Buffer Registration:**
  - Registers a buffer `mask` for the subsequent attention mask, which will be used to ignore future tokens by setting attention scores to `-inf`.
  - The mask is an upper triangular matrix with zeros on the diagonal and ones elsewhere, created using `torch.triu`.

### Forward Method
- **Line 20, Forward Method Definition:**
  - Defines the forward pass of the multi-head attention layer.
  - Takes an input tensor `x` with shape assumptions of `(30, 50, 512)` for batch size, sequence size, and embedding dimension, respectively.

- **Line 21, Input Shape Extraction:**
  - Extracts the batch size (`b`), number of tokens (`num_tokens`), and input dimension (`d_in`) from the input tensor `x`.

- **Lines 22-24, Query/Key/Value Projections:**
  - Applies linear transformations to the input tensor to obtain queries, keys, and values.
  - The resulting tensors have the same shape `(30, 50, 512)`.

- **Lines 26-28, Reshaping for Multi-Head Attention:**
  - Reshapes the query, key, and value tensors to prepare for multi-head attention.
  - The new shape for each tensor is `(30, 50, num_heads, head_dim)`.

- **Lines 30-32, Transposition for Attention Calculation:**
  - Transposes the reshaped tensors to bring the `num_heads` dimension before the `num_tokens` dimension.
  - The new shape for each tensor is `(30, num_heads, 50, head_dim)`.

- **Line 34, Attention Score Calculation:**
  - Computes the attention scores by performing a batched matrix multiplication of queries and keys.
  - The resulting tensor shape is `(30, num_heads, 50, 50)`.

- **Lines 35-37, Attention Mask Application:**
  - Applies the attention mask to the attention scores, setting future tokens' attention scores to `-inf`.
  - This ensures that the model cannot attend to future tokens, maintaining the auto-regressive property.

- **Lines 39-41, Softmax and Dropout on Attention Scores:**
  - Applies the softmax function to the attention scores, normalizing them to probabilities.
  - Applies dropout to the normalized attention scores for regularization.

- **Lines 43, Context Vector Calculation:**
  - Computes the context vectors by performing a weighted sum of the values based on the attention weights.
  - The context vectors are then transposed back to the original token dimension ordering.
  - The resulting tensor shape is `(30, num_heads, 50, head_dim)`.

- **Line 45, Reshaping Context Vectors:**
  - Reshapes the context vectors to concatenate the heads' outputs.
  - The resulting tensor shape is `(30, 50, 512)`.

- **Line 46, Output Projection:**
  - Applies the final linear transformation to the concatenated context vectors.
  - The output tensor has the same shape `(30, 50, 512)` and is returned as the final result of the multi-head attention mechanism.

## Reference
- [code](https://github.com/SDcodehub/LLMs-from-scratch/blob/main/ch03/01_main-chapter-code/multihead-attention.ipynb)
- [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch)
- [CodeEmporium](https://www.youtube.com/watch?v=rPFkX5fJdRY&t=5079s)
- [Umar Jamil](https://www.youtube.com/watch?v=bCz4OMemCcA)
