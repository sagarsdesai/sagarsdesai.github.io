---
title: AdamW Optimizer Decoupled Weight Decay
date: 2025-11-29
author: Sagar Desai
categories: [Optimization]
tags: [adam, adamw, weight decay, regularization, pytorch]
series: ["Training a Language Model"]
---

{{< katex >}}

```python
import torch
import math
from collections.abc import Callable
from typing import Optional


class AdamW(torch.optim.Optimizer):
    """
    Implements the AdamW optimizer
    """
    def __init__(self, params, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps:float = 1e-8, weight_decay: float = 1e-2):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, beta1=betas[0], beta2=betas[1], eps=eps, weight_decay=weight_decay)

        super().__init__(params, defaults)



    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """
        Performs a single optimization step
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # get hyperparameters for this parameter group
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # lazy state initialization
                if len(state) == 0:
                    # t start at 1 in the algorith
                    state["t"] = 0
                    # m <- 
                    state["m"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # v < - 0
                    state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                m, v = state["m"], state["v"]

                # Increment step t (starts at 1)
                t = state["t"] + 1
                state["t"] = t

                # Decoupled weight decay (AdamW)
                if weight_decay != 0:
                    p.add_(p, alpha=-lr * weight_decay)

                # Update biased first and second moment estimates
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias corrections
                bias_corr1 = 1.0 - beta1**t
                bias_corr2 = 1.0 - beta2**t
                step_size = lr * math.sqrt(bias_corr2) / bias_corr1

                # Denominator: sqrt(v) + eps (PyTorch-style)
                denom = v.sqrt().add_(eps)

                # Parameter update
                p.addcdiv_(m, denom, value=-step_size)

        return loss
```

This is a great question. To understand this code, we first need to understand **why AdamW exists** and how it differs from the standard Adam optimizer.

The "W" in AdamW stands for **Weight Decay**.

#### Part 1: The Concept & The Math

In standard stochastic gradient descent (SGD), "Weight Decay" and "L2 Regularization" are mathematically the same thing. They both try to keep your model weights small to prevent overfitting.

However, in **Adam**, they are *not* the same.

1.  **L2 Regularization:** Adds a penalty to the **Loss function**. The optimizer then calculates gradients based on that modified loss.

2.  **Weight Decay:** Decays the **weights directly** during the update step, independent of the gradient.

**The Problem:** Standard implementations of Adam (like in older PyTorch versions) implemented Weight Decay as L2 Regularization (adding it to the gradient). The AdamW paper proved that this breaks the logic of Adam, because Adam adapts the learning rate based on the gradient size. If you mix the weight decay into the gradient, Adam normalizes it, effectively shrinking the weight decay effect and making it inconsistent.

**The Solution (AdamW):** Decouple the weight decay. Apply the gradient update first, and then apply weight decay directly to the parameters.

#### The Algorithm (Math)

Here are the steps implemented in your code (based on Loshchilov & Hutter, 2017):

Let:

  * $t$: Time step

  * $g_t$: Gradient

  * $\theta$: Parameters (weights)

  * $m$: 1st moment (momentum)

  * $v$: 2nd moment (variance)

  * $\eta$: Learning Rate (`lr`)

  * $\lambda$: Weight Decay (`weight_decay`)

**Step 1: Update Moving Averages (Momentum & Variance)**

$$m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t$$

$$v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2$$

**Step 2: Bias Correction**

Because $m$ and $v$ start at 0, they are biased towards 0 in the beginning. We boost them up:

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$

$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

**Step 3: The Update (Adam Logic)**

$$\theta_t = \theta_{t-1} - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

**Step 4: The Weight Decay (The "W" part)**

$$\theta_t = \theta_t - \eta \cdot \lambda \cdot \theta_{t-1}$$

-----

#### Part 2: Mapping Code to Math

Now, let's look at the specific lines in your Python code.

#### 1. Lazy Initialization

```python
if len(state) == 0:
    state["m"] = torch.zeros_like(p, ...) # m start at 0
    state["v"] = torch.zeros_like(p, ...) # v start at 0
```

This simply creates the buffers to store the history of gradients ($m$) and squared gradients ($v$).

#### 2. Updating Moments (Step 1)

```python
# m <- beta1*m + (1-beta1)*g
m.mul_(beta1).add_(grad, alpha=1 - beta1)
# v <- beta2*v + (1-beta2)*g^2
v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
```

  * `.mul_(beta1)` multiplies the old $m$ by $\beta_1$ (e.g., 0.9).
  * `.add_(grad, alpha=1-beta1)` adds $0.1 \times \text{gradient}$.
  * `.addcmul_` stands for "Add Component-wise Multiplication". It calculates $g^2$ and adds it.

#### 3. Calculating the Adaptive Learning Rate (Step 2 & 3 Combined)

The code uses a computational trick. Instead of dividing $m$ and $v$ by their bias corrections individually, it combines the math into a single multiplier `alpha_t`.

Mathematically, the update term is:

$$\eta \frac{m / (1-\beta_1^t)}{\sqrt{v / (1-\beta_2^t)} + \epsilon} \approx \eta \cdot \underbrace{\frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t}}_{\text{Correction}} \cdot \frac{m}{\sqrt{v} + \epsilon}$$

*Note: The code puts $\epsilon$ inside the square root `(v+eps).sqrt()`, whereas some formulas put it outside. This is an implementation detail.*

**The Code:**

```python
bias_corr1 = 1.0 - beta1 ** t
bias_corr2 = 1.0 - beta2 ** t
# The combined multiplier
alpha_t = lr * (math.sqrt(bias_corr2) / bias_corr1)
denom = (v + eps).sqrt()
# Apply the update to the parameter p
p.addcdiv_(m, denom, value=-alpha_t)
```

`addcdiv` performs: $p = p + (-alpha\_t) \times \frac{m}{denom}$. This updates the weights based on the gradient.

#### 4. Decoupled Weight Decay (Step 4)

**This is the most critical line that makes it AdamW:**

```python
p.add_(p, alpha=-lr * weight_decay)
```

It subtracts a portion of the weight *directly* from itself.

Formula: $p_{new} = p_{old} - \eta \cdot \lambda \cdot p_{old}$.

-----

#### Part 3: A Numerical Example

Let's trace a single parameter to see this in action.

**Scenario:**

  * **Weight ($p$):** 10.0

  * **Gradient ($g$):** 2.0

  * **Learning Rate ($lr$):** 0.1

  * **Weight Decay:** 0.1

  * **Step ($t$):** 1

  * **$\beta_1$:** 0.9, **$\beta_2$:** 0.999

  * **$\epsilon$:** 0 (ignored for simplicity)

**Execution:**

1.  **Update Moments ($m, v$):**

      * $m = 0.9(0) + 0.1(2.0) = \mathbf{0.2}$

      * $v = 0.999(0) + 0.001(2.0^2) = 0.001(4) = \mathbf{0.004}$

2.  **Calculate Correction Factor (`alpha_t`):**

      * `bias_corr1` = $1 - 0.9^1 = 0.1$

      * `bias_corr2` = $1 - 0.999^1 = 0.001$

      * `alpha_t` = $0.1 \times \frac{\sqrt{0.001}}{0.1} = 0.1 \times \frac{0.0316}{0.1} = \mathbf{0.0316}$

3.  **Gradient Update:**

      * `denom` = $\sqrt{v} = \sqrt{0.004} \approx 0.0632$

      * Update term = `alpha_t` $\times (m / \text{denom})$

      * Update term = $0.0316 \times (0.2 / 0.0632) \approx 0.0316 \times 3.16 \approx \mathbf{0.1}$

      * New Weight (intermediate) = $10.0 - 0.1 = \mathbf{9.9}$

4.  **Weight Decay Update (The "W"):**

      * Decay = $lr \times \text{weight\_decay} \times \text{Old Weight}$

      * Decay = $0.1 \times 0.1 \times 10.0 = \mathbf{0.1}$

      * Final Weight = $9.9 - 0.1 = \mathbf{9.8}$

**Comparison:**

In standard Adam (without "W"), the weight decay would have been added to the gradient ($g$) at the very beginning. Because Adam divides by the variance $\sqrt{v}$, that large weight decay ($10 \times 0.1 = 1$) would have been "normalized" down by the optimizer math, resulting in a much smaller effective decay.

By doing it at the end (Step 4), AdamW ensures that exactly $1\%$ (if $lr \times \lambda = 0.01$) of the weight is removed every step, guaranteeing stable regularization.


