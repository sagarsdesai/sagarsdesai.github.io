---
title: "AdamW Accounting Resource and Compute Estimates"
date: 2025-11-30
tags: [AdamW, Resource Accounting, Memory, FLOPs, GPT-2 XL, Training]
---

{{< katex >}}

Here are the answers to the resource accounting problems based on your implementation and the provided configuration.

#### Problem (adamwAccounting): Resource accounting for training with AdamW

**Assuming:**

- **Precision:** Float32 (4 bytes per element) for all tensors.
- **Notation:** $B$ (batch size), $T$ (context length), $D$ (`d_model`), $L$ (`num_layers`), $H$ (`num_heads`), $V$ (`vocab_size`).
- **FFN:** Standard architecture with $d_{ff} = 4D$, consisting of $W_1$ (up-projection) and $W_2$ (down-projection).
- **Architecture:** Untied embeddings (separate input and output embedding matrices).

---

#### (a) Peak Memory Usage

How much peak memory does running AdamW require? Decompose your answer based on the
memory usage of the parameters, activations, gradients, and optimizer state. Express your answer
in terms of the `batch_size` and the model hyperparameters (`vocab_size`, `context_length`,
`num_layers`, `d_model`, `num_heads`). Assume $d_{ff} = 4 d_{model}$.
For simplicity, when calculating memory usage of activations, consider only the following compo-
nents:
- Transformer block
  - RMSNorm(s)
  - Multi-head self-attention sublayer: QKV projections, Q⊤ K matrix multiply, softmax, weighted sum of values, output projection.
  - Position-wise feed-forward: W1 matrix multiply, SiLU, W2 matrix multiply
- final RMSNorm
- output embedding
- cross-entropy on logits
  
Deliverable: An algebraic expression for each of parameters, activations, gradients, and opti-
mizer state, as well as the total.

The peak memory is the sum of memory required for parameters, gradients, optimizer states, and activations.

1. **Parameters ($M_{params}$)**
   The model consists of embeddings, $L$ Transformer blocks, and final layers.
   - **Embeddings:** Input ($V \times D$) + Position ($T \times D$).
   - **Transformer Block (per layer):**
     - Attention: 4 matrices ($W_Q, W_K, W_V, W_O$), each $D \times D$. → $4D^2$.
     - FFN: $W_1$ ($D \times 4D$) + $W_2$ ($4D \times D$). → $8D^2$.
     - Norms: 2 RMSNorms, each has a gain vector of size $D$. → $2D$.
   - **Final Layers:** Final RMSNorm ($D$) + Output Embedding/LM Head ($D \times V$).

$$
N_{params} \approx 2VD + TD + L(12D^2 + 2D) + D
$$

$$
\text{Memory}_{params} = 4 \times N_{params} \text{ bytes}
$$

2. **Gradients ($M_{grads}$)**
   We store one gradient value for every parameter.
$$
\text{Memory}_{grads} = \text{Memory}_{params}
$$

3. **Optimizer State ($M_{opt}$)**
   AdamW maintains two state tensors (\(m\) and \(v\)) for every parameter.
$$
\text{Memory}_{opt} = 2 \times \text{Memory}_{params}
$$

4. **Activations ($M_{act}$)**
   We must store intermediate tensors from the forward pass to compute gradients during the backward pass. Based on the components listed:
   - **Per Layer:**
     - RMSNorm1 Input: $B \times T \times D$
     - QKV Projections (Input/Output): $B \times T \times D$ (Input shared) + $3 \times B \times T \times D$ (Output).
     - $Q K^\top$ (Scores) & Softmax (Probs): $2 \times B \times H \times T^2$
     - Weighted Sum (Context): $B \times T \times D$
     - Output Projection (Input/Output): $B \times T \times D$ (Input shared with Context).
     - RMSNorm2 Input: $B \times T \times D$
     - FFN $W_1$ Out (Hidden): $B \times T \times 4D$
     - FFN SiLU Out (Activated): $B \times T \times 4D$
     - FFN $W_2$ Input: (Shared with SiLU Out)
   - **Per Layer Sum:** Summing the distinct large tensors stored: $16BTD + 2BHT^2$.
   - **Non-Layer:** Final Norm Input ($BTD$) + Logits ($BTV$).

$$
N_{act} = L(16BTD + 2BHT^2) + BTD + BTV
$$

$$
\text{Memory}_{act} = 4 \times N_{act} \text{ bytes}
$$

**Total Peak Memory:**
$$
\text{Total} = 4 \times (4 \cdot N_{params} + N_{act}) \text{ bytes}
$$
*(Note: $4 \cdot N_{params}$ comes from Params + Grads + 2 States)*

---

#### (b) Instantiation for GPT-2 XL

Instantiate your answer for a GPT-2 XL-shaped model to get an expression that only depends on
the `batch_size`. What is the maximum batch size you can use and still fit within 80GB memory?
Deliverable: An expression that looks like a·`batch_size` + b for numerical values a, b, and a
number representing the maximum batch size.

**Configuration:**
- $L=48, D=1600, H=25, T=1024, V=50257$.
- $d_{ff} = 6400$.

**1. Fixed Memory (Params + Grads + Opt):**
- **Parameter Count ($N_{params}$):**
  - Embeddings: $2 \times 50257 \times 1600 + 1024 \times 1600 \approx 162.4 \times 10^6$
  - Layers: $48 \times (12 \times 1600^2 + 2 \times 1600) \approx 48 \times 30.72 \times 10^6 \approx 1.475 \times 10^9$
  - Total $N_{params} \approx 1.637 \times 10^9$ (1.64 Billion parameters).
- **Memory:** 16 bytes/param × $1.637 \times 10^9 \approx 26.2$ GB.

**2. Variable Memory (Activations per batch):**
- **Per Batch ($N_{act}/B$):**
  - Layers: \(48 \times (16 \cdot 1024 \cdot 1600 + 2 \cdot 25 \cdot 1024^2)\)
    $= 48 \times (26.2\,\mathrm{M} + 52.4\,\mathrm{M}) = 48 \times 78.6\,\mathrm{M} \approx 3.77 \times 10^9$ elements.
  - Logits + Misc: $1024 \times 50257 + 1024 \times 1600 \approx 53 \times 10^6$ elements.
  - Total per batch $\,\approx 3.82 \times 10^9$ floats.
  - **Memory:** $4 \times 3.82 \times 10^9 \approx 15.3$ GB.

**Expression:**
$$
\text{Memory (GB)} \approx 15.3 \cdot \text{batch\_size} + 26.2
$$

**Maximum Batch Size:**
We need $15.3 \cdot B + 26.2 \le 80$.
$$
15.3 \cdot B \le 53.8
$$
$$
B \le 3.51
$$
**Answer:** Maximum batch size is **3**.

---

#### (c) FLOPs per AdamW Step

How many FLOPs does running one step of AdamW take?
Deliverable: An algebraic expression, with a brief justification.

**Expression:**
$$
\approx 11 \times N_{params} \text{ FLOPs}
$$

**Justification:**
AdamW performs element-wise operations on the parameters. For each parameter $\theta$, the update involves:
1. Updating first moment $m$: 3 ops (mul, add, mul).
2. Updating second moment $v$: 4 ops (mul, mul, mul, add).
3. Parameter update:

$$
\theta \leftarrow \theta - \eta \frac{m}{\sqrt{v}+\epsilon} - \eta \lambda \theta
$$

≈ 8 ops (sqrt, add, div, mul, sub, mul, mul, sub).

Total is approximately 15 FLOPs per parameter, but typically estimated around **11–12 FLOPs** depending on specific implementation optimizations (e.g., fusing operations).

---

#### (d) Time to Train on A100

Model FLOPs utilization (MFU) is defined as the ratio of observed throughput (tokens per second)
relative to the hardware’s theoretical peak FLOP throughput [Chowdhery et al., 2022]. An
NVIDIA A100 GPU has a theoretical peak of 19.5 teraFLOP/s for float32 operations. Assuming
you are able to get 50% MFU, how long would it take to train a GPT-2 XL for 400K steps and a
batch size of 1024 on a single A100? Following Kaplan et al. [2020] and Hoffmann et al. [2022],
assume that the backward pass has twice the FLOPs of the forward pass.
Deliverable: The number of days training would take, with a brief justification.

**Estimates:**
- **Total Training FLOPs ($C_{train}$):**
  Using the approximation $C \approx 6 \times N_{params} \times D_{tokens}$ (where 6 accounts for 2 Fwd + 4 Bwd FLOPs per token).
  - $N_{params} \approx 1.64 \times 10^9$
  - $D_{tokens} = 400{,}000 \times 1024 \times 1024 \approx 4.19 \times 10^{11}$ tokens.
  - $C_{train} \approx 6 \times 1.64 \times 10^9 \times 4.19 \times 10^{11} \approx 4.12 \times 10^{21}$ FLOPs.

- **Hardware Throughput:**
  - Peak FP32 = 19.5 TFLOPS.
  - Effective MFU (50%) = $0.5 \times 19.5 \times 10^{12} = 9.75 \times 10^{12}$ FLOPs/s.

- **Time:**
$$
\text{Time} = \frac{4.12 \times 10^{21}}{9.75 \times 10^{12}} \approx 422{,}500{,}000 \text{ seconds}
$$
$$
\text{Days} = \frac{4.22 \times 10^{8}}{86{,}400} \approx \mathbf{4{,}880 \text{ days}}
$$

**Answer:** It would take approximately **4,880 days (or ~13 years)**.

**Justification:** Training a 1.5B parameter model on 400 billion tokens is a massive workload (comparable to modern LLM pre-training). Doing this on a single GPU using strictly FP32 cores (without Tensor Cores) is computationally infeasible, necessitating distributed training across hundreds of GPUs.


