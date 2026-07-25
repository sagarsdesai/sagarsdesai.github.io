---
title: Transformer LM — Resource Accounting (Parameters and FLOPs)
date: 2025-10-20
author: Sagar Desai
categories: [LLM]
tags: [transformers, pytorch, cs336, transformer-lm]
series: ["Transformer From Scratch"]
---

{{< katex >}}

Consider GPT-2 XL, which has the following configuration:
vocab_size : 50,257
context_length : 1,024
num_layers : 48
d_model : 1,600
27num_heads : 25
d_ff : 6,400
Suppose we constructed our model using this configuration. How many trainable parameters
would our model have? Assuming each parameter is represented using single-precision floating
point, how much memory is required to just load this model?
Deliverable: A one-to-two sentence response.
(b) Identify the matrix multiplies required to complete a forward pass of our GPT-2 XL-shaped
model. How many FLOPs do these matrix multiplies require in total? Assume that our input
sequence has context_length tokens.
Deliverable: A list of matrix multiplies (with descriptions), and the total number of FLOPs
required.
(c) Based on your analysis above, which parts of the model require the most FLOPs?
Deliverable: A one-to-two sentence response.
(d) Repeat your analysis with GPT-2 small (12 layers, 768 d_model, 12 heads), GPT-2 medium (24
layers, 1024 d_model, 16 heads), and GPT-2 large (36 layers, 1280 d_model, 20 heads). As the
model size increases, which parts of the Transformer LM take up proportionally more or less of
the total FLOPs?
Deliverable: For each model, provide a breakdown of model components and its associated
FLOPs (as a proportion of the total FLOPs required for a forward pass). In addition, provide a
one-to-two sentence description of how varying the model size changes the proportional FLOPs
of each component.
(e) Take GPT-2 XL and increase the context length to 16,384. How does the total FLOPs for one
forward pass change? How do the relative contribution of FLOPs of the model components
change?
Deliverable: A one-to-two sentence response.

Here are the resource accounting calculations for the Transformer LM you've built.

---

## (a) GPT-2 XL: Parameters and Memory

[cite_start]Based on our model's architecture (with a SwiGLU FFN [cite: 638] [cite_start]and untied embeddings), a GPT-2 XL configuration [cite: 782-789] would have **2,127,057,600** ($\approx 2.13$ billion) trainable parameters. Assuming single-precision (4 bytes) per parameter, this model would require **$\approx 7.92$ GB** of memory just to load the weights.

* **Embeddings (Untied):** $2 \times (V \times D) = 2 \times (50,257 \times 1,600) = 160,822,400$
* **Transformer Blocks:** $L \times (\text{Params per block})$
    * **Attn ($W_Q, W_K, W_V, W_O$):** $4 \times (D \times D) = 4 \times (1,600^2) = 10,240,000$
    * **FFN ($W_1, W_2, W_3$):** $3 \times (D \times F) = 3 \times (1,600 \times 6,400) = 30,720,000$
    * **Norms (norm1, norm2):** $2 \times D = 2 \times 1,600 = 3,200$
    * **Total per block:** $10,240,000 + 30,720,000 + 3,200 = 40,963,200$
* **Total for Blocks:** $L \times 40,963,200 = 48 \times 40,963,200 = 1,966,233,600$
* **Final Norm:** $D = 1,600$
* **Grand Total:** $160,822,400 \text{ (Embeds)} + 1,966,233,600 \text{ (Blocks)} + 1,600 \text{ (Final Norm)} = 2,127,057,600$
* **Memory:** $2,127,057,600 \times 4 \text{ bytes} / (1024^3 \text{ GB}) = 7.92 \text{ GB}$

---

## (b) GPT-2 XL: FLOPs

[cite_start]Here is a list of the major matrix multiplies in a forward pass and their FLOPs, assuming a sequence length $T=1024$[cite: 784, 794].

* **Per Transformer Block ($L=48$ times):**
    * **Attn Projections ($W_Q, W_K, W_V$):** 3 matmuls, $3 \times (2T D^2) = 15,728,640,000$ FLOPs
    * **Attn Scores ($Q@K^T$):** 1 matmul, $2 H T^2 (D/H) = 6,710,886,400$ FLOPs
    * **Attn Output ($Scores@V$):** 1 matmul, $2 H T^2 (D/H) = 3,355,443,200$ FLOPs
    * **Attn Output Projection ($W_O$):** 1 matmul, $2 T D^2 = 5,242,880,000$ FLOPs
    * **FFN ($W_1, W_3$):** 2 matmuls, $2 \times (2 T D F) = 41,943,040,000$ FLOPs
    * **FFN ($W_2$):** 1 matmul, $2 T F D = 20,971,520,000$ FLOPs
    * **Total per block:** $90,596,966,400$ FLOPs
* **Total for all $L$ Blocks:** $48 \times 90,596,966,400 = 4,348,654,387,200$ FLOPs
* **Final LM Head:**
    * **LM Head Projection:** 1 matmul, $2 T D V = 164,682,137,600$ FLOPs

**Total FLOPs:** $4,348,654,387,200 \text{ (Blocks)} + 164,682,137,600 \text{ (Head)} = \textbf{4,513,336,524,800}$ (or $\approx \mathbf{4.51}$ TFLOPs).

Certainly! Let's break down those FLOPs calculations in detail.

The fundamental rule for calculating **FLOPs (Floating Point Operations)** for a matrix multiplication is:
Given two matrices $A$ (shape $m \times k$) and $B$ (shape $k \times n$), the operation $A @ B$ results in a matrix of shape $(m \times n)$ and requires $2 \times m \times k \times n$ FLOPs. The '2' accounts for the one multiplication and one addition in each multiply-accumulate (MAC) operation.

Let's define our variables for GPT-2 XL:
* $T$ (sequence length) = $1024$
* $D$ (model dimension) = $1600$
* $H$ (num heads) = $25$
* $d_k$ (head dimension) = $D / H = 1600 / 25 = 64$
* $F$ (FFN inner dim) = $6400$
* $V$ (vocab size) = $50257$
* $L$ (num layers) = $48$

---

### Per Transformer Block ($L=48$ times)

Here is the detailed breakdown of the matrix multiplies (matmuls) inside a single Transformer block.

#### 1. Attn Projections ($W_Q, W_K, W_V$)
* **Your specific question: `3 * (2TD^2)`**
* **What it is:** This step creates the **Query ($Q$)**, **Key ($K$)**, and **Value ($V$)** matrices by projecting the input $X$ (which has shape $T \times D$) using three separate weight matrices.
* **Shapes:**
    * Input $X$: $(T \times D) = (1024 \times 1600)$
    * Weight $W_Q$: $(D \times D) = (1600 \times 1600)$
    * Weight $W_K$: $(D \times D) = (1600 \times 1600)$
    * Weight $W_V$: $(D \times D) = (1600 \times 1600)$
* **FLOPs Calculation:**
    * Matmul for $Q = X @ W_Q$: $(T \times D) @ (D \times D)$. FLOPs = $2 \times T \times D \times D = 2TD^2$.
    * Matmul for $K = X @ W_K$: $(T \times D) @ (D \times D)$. FLOPs = $2 \times T \times D \times D = 2TD^2$.
    * Matmul for $V = X @ W_V$: $(T \times D) @ (D \times D)$. FLOPs = $2 \times T \times D \times D = 2TD^2$.
* **Total:** We have 3 identical matmuls, so the total is $2TD^2 + 2TD^2 + 2TD^2 = 3 \times (2TD^2) = \mathbf{6TD^2}$.
* **Value:** $6 \times 1024 \times (1600^2) = \textbf{15,728,640,000}$ FLOPs.

#### 2. Attn Scores ($Q@K^T$)
* **What it is:** This calculates the attention scores by multiplying the $Q$ and $K$ matrices. This is done *per head* in a batched matrix multiply.
* **Shapes (per head):** The $Q$ and $K$ matrices (each $T \times D$) are split across $H$ heads.
    * $Q_{head}$: $(T \times d_k) = (1024 \times 64)$
    * $K_{head}^T$ (transposed): $(d_k \times T) = (64 \times 1024)$
* **FLOPs Calculation (per head):** $Q_{head} @ K_{head}^T$ results in a $(T \times T)$ score matrix.
    * FLOPs = $2 \times T \times d_k \times T = 2T^2d_k$.
* **Total (all $H$ heads):** $H \times (2T^2d_k) = 2HT^2d_k$.
    * Since $d_k = D/H$, this simplifies to $2HT^2(D/H) = \mathbf{2T^2D}$.
* **Value:** $2 \times (1024^2) \times 1600 = \textbf{3,355,443,200}$ FLOPs.
    * *(Note: The value 6,710,886,400 in your provided text for this step appears to be exactly double the correct amount.)*

#### 3. Attn Output ($Scores@V$)
* **What it is:** The $(T \times T)$ attention scores (after softmax) are applied to the $V$ matrix. This is also done *per head*.
* **Shapes (per head):**
    * $Scores_{head}$: $(T \times T) = (1024 \times 1024)$
    * $V_{head}$: $(T \times d_v)$ (where $d_v = d_k$) $= (1024 \times 64)$
* **FLOPs Calculation (per head):** $Scores_{head} @ V_{head}$ results in a $(T \times d_v)$ output matrix.
    * FLOPs = $2 \times T \times T \times d_v = 2T^2d_v$.
* **Total (all $H$ heads):** $H \times (2T^2d_v) = 2HT^2d_v$.
    * This also simplifies to $\mathbf{2T^2D}$.
* **Value:** $2 \times (1024^2) \times 1600 = \textbf{3,355,443,200}$ FLOPs.

#### 4. Attn Output Projection ($W_O$)
* **What it is:** The combined outputs from all heads (which is back to $T \times D$) are projected one last time.
* **Shapes:**
    * Input (concatenated heads): $(T \times D) = (1024 \times 1600)$
    * Weight $W_O$: $(D \times D) = (1600 \times 1600)$
* **FLOPs Calculation:** $(T \times D) @ (D \times D)$. FLOPs = $\mathbf{2TD^2}$.
* **Value:** $2 \times 1024 \times (1600^2) = \textbf{5,242,880,000}$ FLOPs.

#### 5. Feed-Forward Network (FFN)
* **What it is:** A two-layer perceptron with an "up-projection" (to $F=6400$) and a "down-projection" (back to $D=1600$).
    * *(Note: Your text splits this confusingly. The standard FFN involves two matmuls totaling $4TDF$.)*
* **1. Up-Projection ($W_1$):**
    * **Shapes:** Input $(T \times D)$ @ $W_1$ $(D \times F)$.
    * **FLOPs:** $2 \times T \times D \times F$.
* **2. Down-Projection ($W_2$):**
    * **Shapes:** Activated output $(T \times F)$ @ $W_2$ $(F \times D)$.
    * **FLOPs:** $2 \times T \times F \times D$.
* **Total (FFN):** $(2TDF) + (2TFD) = \mathbf{4TDF}$.
* **Value:** $4 \times 1024 \times 1600 \times 6400 = \textbf{41,943,040,000}$ FLOPs.

---

### Final LM Head

This happens *once* after all $L=48$ blocks are finished.

#### LM Head Projection
* **What it is:** The final output of the model $(T \times D)$ is projected to the vocabulary size $V$ to create logits.
* **Shapes:**
    * Input $X_{final}$: $(T \times D) = (1024 \times 1600)$
    * Weight $W_{LM}$: $(D \times V) = (1600 \times 50257)$
* **FLOPs Calculation:** $(T \times D) @ (D \times V)$. FLOPs = $\mathbf{2TDV}$.
* **Value:** $2 \times 1024 \times 1600 \times 50257 = \textbf{164,682,137,600}$ FLOPs.
---

## (c) FLOPs Bottleneck

The vast majority of computation (FLOPs) occurs within the **Transformer blocks** ($\approx 96.5\%$ of the total). Within each block, the **feed-forward network (FFN)** is the most expensive component, accounting for $\approx 70\%$ of the block's FLOPs ($62.9 \text{ GFLOPs}$) compared to the attention mechanism's $\approx 30\%$ ($27.6 \text{ GFLOPs}$).

---

## (d) Scaling Analysis

Here is the breakdown of FLOPs for each model, all with $T=1024$.

* **GPT-2 small ($L=12, D=768, F=3072$):**
    * Blocks: $2.706 \times 10^{11}$ FLOPs (77.4%)
    * LM Head: $0.791 \times 10^{11}$ FLOPs (22.6%)
    * **Total:** $3.497 \times 10^{11}$ FLOPs
* **GPT-2 medium ($L=24, D=1024, F=4096$):**
    * Blocks: $9.277 \times 10^{11}$ FLOPs (89.8%)
    * LM Head: $1.054 \times 10^{11}$ FLOPs (10.2%)
    * **Total:** $1.033 \times 10^{12}$ FLOPs
* **GPT-2 large ($L=36, D=1280, F=5120$):**
    * Blocks: $2.126 \times 10^{12}$ FLOPs (94.2%)
    * LM Head: $0.132 \times 10^{12}$ FLOPs (5.8%)
    * **Total:** $2.258 \times 10^{12}$ FLOPs
* **GPT-2 XL ($L=48, D=1600, F=6400$):**
    * Blocks: $4.349 \times 10^{12}$ FLOPs (96.3%)
    * LM Head: $0.165 \times 10^{12}$ FLOPs (3.7%)
    * **Total:** $4.513 \times 10^{12}$ FLOPs

As the model size increases, the proportional FLOPs cost shifts dramatically toward the **Transformer blocks**, while the LM head's proportional cost shrinks. This is because the blocks' FLOPs scale quadratically with $D$ (from $L \times T D^2$), whereas the LM head's FLOPs scale only linearly with $D$ (from $T D V$).

---

## (e) Scaling Context Length

[cite_start]Increasing the context length ($T$) from $1,024$ to $16,384$ (a $16\times$ increase) [cite: 801] causes the total FLOPs to increase by $\approx \mathbf{33.2\times}$ (from $4.51$ TFLOPs to $149.5$ TFLOPs). The relative contribution of FLOPs also shifts significantly: the attention score computations ($Q@K^T$ and $Scores@V$), which scale quadratically with $T$ ($T^2$), grow from $\approx 7\%$ of the total FLOPs to $\approx \mathbf{55\%}$, becoming the new computational bottleneck.


