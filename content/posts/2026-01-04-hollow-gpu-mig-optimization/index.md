---
title: "The Hollow GPU Problem - Optimizing GenAI Inference with MIG"
date: 2026-01-04
tags: [GenAI, Inference, GPU, MIG, NVIDIA, A100, Kubernetes, Infrastructure]
featured: true
---

{{< katex >}}

#### Introduction

In production GenAI deployments, "Day 1" focuses on functionality: getting the RAG pipeline running, reducing hallucinations, and minimizing latency. "Day 2" reveals the cost reality when you check Grafana dashboards.

This post explores a common infrastructure anti-pattern—the **"Hollow GPU"**—where powerful accelerators are held hostage by lightweight workloads. We detail the technical strategy to solve it using NVIDIA MIG (Multi-Instance GPU).

#### The Scenario - Asymmetric Multimodal AI

Consider a production voice-enabled RAG (Retrieval-Augmented Generation) application running NVIDIA NIMs (NVIDIA Inference Microservices) on **3x NVIDIA A100 80GB** GPUs.

The pipeline consists of three distinct models:

| Component | Model | Role | Size |
|:---|:---|:---|:---|
| **LLM** | `meta-llama3-8b` | Language Understanding | ~16B params (quantized) |
| **ASR** | `parakeet-1.1b` | Automatic Speech Recognition | 1.1B params |
| **TTS** | `magpie` | Text-to-Speech | ~2B params |

#### The Utilization Reality

GPU metrics reveal severe underutilization:

| GPU | Workload | VRAM Used | Utilization |
|:---|:---|:---|:---|
| GPU 0 | LLM | 73.2 GB | 91% ✅ |
| GPU 1 | ASR | 9.6 GB | 12% ❌ |
| GPU 2 | TTS | 13.4 GB | 17% ❌ |

#### The Hollow GPU Problem

Kubernetes treats GPUs as monolithic integer resources. When you request `nvidia.com/gpu: 1`, you get the entire card.

```
BEFORE: The "Hollow GPU" Anti-Pattern
(Monolithic Allocation = Wasted Resources)

     GPU 0 (A100 80GB)                 GPU 1 (A100 80GB)                 GPU 2 (A100 80GB)
  ┌───────────────────────┐         ┌───────────────────────┐         ┌───────────────────────┐
  │ WORKLOAD: LLM (8B)    │         │ WORKLOAD: ASR         │         │ WORKLOAD: TTS         │
  │                       │         │                       │         │                       │
  │  [████████████████]   │         │  [██░░░░░░░░░░░░░░]   │         │  [███░░░░░░░░░░░░░]   │
  │  73GB Used            │         │  9GB Used             │         │  13GB Used            │
  │                       │         │                       │         │                       │
  │    91% EFFICIENT      │         │    12% EFFICIENT      │         │    17% EFFICIENT      │
  │                       │         │    🔴 HUGE WASTE      │         │    🔴 HUGE WASTE      │
  └───────────────────────┘         └───────────────────────┘         └───────────────────────┘
                                                ▲                                 ▲
                                                │                                 │
                                       "The Hollow Space"                "The Hollow Space"
                                   (Expensive silicon doing nothing)
```

**The Math:**
- ASR + TTS combined VRAM usage: **23 GB**
- Allocated capacity (2x A100): **160 GB**
- Wasted HBM2e memory: **137 GB (86%)**
- Total cluster VRAM utilization: **40%**

This configuration wastes sufficient capacity to run a second LLM replica for handling higher concurrency.

---

#### GPU Sharing Strategies - Technical Deep Dive

Moving from **Monolithic Allocation** to **Fractional Allocation** requires understanding three distinct approaches, each operating at different layers of the stack.

##### Option 1 - Time-Slicing (Software Scheduler)

Time-slicing is implemented via the Kubernetes GPU scheduling layer (NVIDIA GPU Operator).

**Mechanism:**
```yaml
# ConfigMap for time-slicing
apiVersion: v1
kind: ConfigMap
metadata:
  name: time-slicing-config
data:
  any: |-
    version: v1
    sharing:
      timeSlicing:
        resources:
        - name: nvidia.com/gpu
          replicas: 4
```

The NVIDIA driver performs rapid context switching between processes. A single physical GPU advertises as multiple "virtual" GPUs.

**Technical Characteristics:**

| Aspect | Behavior |
|:---|:---|
| Context Switch | ~25-50μs per switch (register file + L1 cache flush) |
| Memory Isolation | None (shared address space) |
| Fault Isolation | None (OOM crashes all tenants) |
| Latency Profile | High jitter (10-100ms spikes during switches) |

**Why This Fails for Voice AI:**

For ASR/TTS workloads with real-time constraints:
- Audio generation latency budget: **<50ms** per chunk
- Context switch overhead: **25-50μs** × N processes
- Stutter occurs when TTS is paused mid-generation

##### Option 2 - NVIDIA MPS (Multi-Process Service)

MPS is a CUDA driver feature enabling multiple processes to share GPU resources concurrently.

**Architecture:**

```
┌─────────────────────────────────────────────────────┐
│                    MPS Server                        │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐              │
│  │ Client A │  │ Client B │  │ Client C │  ← Processes│
│  └────┬────┘  └────┬────┘  └────┬────┘              │
│       └───────────┼───────────┘                     │
│                   ▼                                  │
│         ┌─────────────────┐                         │
│         │  MPS Control    │                         │
│         │  Daemon         │                         │
│         └────────┬────────┘                         │
│                  ▼                                   │
│         ┌─────────────────┐                         │
│         │  Unified CUDA   │                         │
│         │  Context        │                         │
│         └─────────────────┘                         │
└─────────────────────────────────────────────────────┘
```

**Technical Characteristics:**

| Aspect | Behavior |
|:---|:---|
| Kernel Execution | Concurrent (if SMs available) |
| Memory Bandwidth | Shared (contention possible) |
| Fault Isolation | Weak (segfault can crash MPS server) |
| QoS | None (no resource guarantees) |

**Why MPS Falls Short:**

- **Blast Radius:** A segfault in one client terminates the MPS daemon, crashing all connected clients
- **Noisy Neighbor:** Memory bandwidth contention causes unpredictable latency spikes
- **No Resource Caps:** A burst of ASR traffic can starve TTS

##### Option 3 - MIG (Multi-Instance GPU) - The Production Choice

MIG (Ampere/Hopper architectures) provides **physical hardware partitioning**, not virtualization.

**A100 Internal Architecture:**

```
┌───────────────────────────────────────────────────────────────┐
│                         A100 80GB                              │
├───────────────────────────────────────────────────────────────┤
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐     │
│  │GPC 0│ │GPC 1│ │GPC 2│ │GPC 3│ │GPC 4│ │GPC 5│ │GPC 6│     │
│  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘     │
│     └───────┴───────┴───────┼───────┴───────┴───────┘         │
│                             │                                  │
│  ┌──────────────────────────┴────────────────────────────┐    │
│  │              L2 Cache (40 MB Total)                    │    │
│  │    [Slice 0][Slice 1][Slice 2][Slice 3][Slice 4]...   │    │
│  └──────────────────────────┬────────────────────────────┘    │
│                             │                                  │
│  ┌──────────────────────────┴────────────────────────────┐    │
│  │           Memory Controllers (8x HBM2e)                │    │
│  │    [MC 0][MC 1][MC 2][MC 3][MC 4][MC 5][MC 6][MC 7]   │    │
│  └───────────────────────────────────────────────────────┘    │
│                             │                                  │
│  ┌──────────────────────────┴────────────────────────────┐    │
│  │                 HBM2e (80 GB Total)                    │    │
│  └───────────────────────────────────────────────────────┘    │
└───────────────────────────────────────────────────────────────┘
```

**MIG Partitioning:**

MIG physically assigns **GPCs**, **L2 Cache slices**, and **Memory Controllers** to each instance.

| Profile | GPCs | Memory | SMs | Compute Capability |
|:---|:---|:---|:---|:---|
| `7g.80gb` | 7 | 80 GB | 98 | Full GPU |
| `4g.40gb` | 4 | 40 GB | 56 | ~57% compute |
| `3g.40gb` | 3 | 40 GB | 42 | ~43% compute |
| `2g.20gb` | 2 | 20 GB | 28 | ~29% compute |
| `1g.10gb` | 1 | 10 GB | 14 | ~14% compute |
| `1g.10gb+me` | 1 | 10 GB | 14 | + Media Engines |

**Hardware Isolation Guarantees:**

| Aspect | Behavior |
|:---|:---|
| Memory Bandwidth | Dedicated (no contention) |
| L2 Cache | Partitioned (no thrashing) |
| Fault Isolation | Hardware-enforced (OOM affects only that instance) |
| Latency | Deterministic (no noisy neighbor) |
| Security | Side-channel attack prevention |

---

#### Implementation - MIG Configuration

Given voice AI latency requirements, **MIG is the only viable production choice**.

##### Target Configuration

We reconfigure **GPU 1** to handle both ASR and TTS, freeing GPU 2.

**Partition Profile: `3g.40gb` × 2**

| Instance | Profile | Compute | Memory | Workload |
|:---|:---|:---|:---|:---|
| GPU 1 - Slice A | `3g.40gb` | 42 SMs | 40 GB | ASR (Parakeet) |
| GPU 1 - Slice B | `3g.40gb` | 42 SMs | 40 GB | TTS (Magpie) |

*Note: The remaining 1/7th of compute (14 SMs) is reserved overhead in this configuration.*

##### MIG Setup Commands

```bash
# Check MIG capability
nvidia-smi -i 1 --query-gpu=mig.mode.current --format=csv

# Enable MIG mode (requires GPU reset)
sudo nvidia-smi -i 1 -mig 1

# Create MIG instances (3g.40gb profile = GPU Instance 19)
sudo nvidia-smi mig -i 1 -cgi 19,19 -C

# Verify instances
nvidia-smi mig -i 1 -lgi
```

**Expected Output:**

```
+-------------------------------------------------------+
| MIG devices:                                          |
+------------------+----------------------+--------------+
| GPU  GI  CI  MIG |         Memory-Usage | SM           |
|      ID  ID  Dev |                      |              |
|==================+======================+==============|
|  1    1   0   0  |      0MiB / 40960MiB | 42           |
+------------------+----------------------+--------------+
|  1    2   0   1  |      0MiB / 40960MiB | 42           |
+------------------+----------------------+--------------+
```

##### Kubernetes Configuration

```yaml
# GPU Operator Helm values for MIG
migStrategy: mixed
devicePlugin:
  config:
    name: mig-config
    default: all-balanced
---
# Pod requesting specific MIG device
apiVersion: v1
kind: Pod
metadata:
  name: asr-parakeet
spec:
  containers:
  - name: asr
    image: nvcr.io/nim/nvidia/parakeet-1.1b
    resources:
      limits:
        nvidia.com/mig-3g.40gb: 1
```

---

#### Resource Accounting

##### Before MIG

| Resource | VRAM Used | VRAM Allocated | Efficiency |
|:---|:---|:---|:---|
| GPU 0 (LLM) | 73.2 GB | 80 GB | 91% |
| GPU 1 (ASR) | 9.6 GB | 80 GB | 12% |
| GPU 2 (TTS) | 13.4 GB | 80 GB | 17% |
| **Total** | **96.2 GB** | **240 GB** | **40%** |

##### After MIG

| Resource | Configuration | VRAM Used | VRAM Allocated | Efficiency |
|:---|:---|:---|:---|:---|
| GPU 0 | Full A100 | 73.2 GB | 80 GB | 91% |
| GPU 1 - Slice A | `3g.40gb` | 9.6 GB | 40 GB | 24% |
| GPU 1 - Slice B | `3g.40gb` | 13.4 GB | 40 GB | 34% |
| GPU 2 | **FREED** | 0 GB | 80 GB | Available |
| **Total Used** | | **96.2 GB** | **160 GB** | **60%** |

```
AFTER: Hardware Partitioning (MIG)
(Physical Isolation = Maximized Density)

     GPU 0 (A100 80GB)                 GPU 1 (A100 80GB)                 GPU 2 (A100 80GB)
                                     (Partitioned Mode)                  (Recaptured Resource)
  ┌───────────────────────┐         ┌───────────────────────┐         ┌───────────────────────┐
  │ WORKLOAD: LLM (8B)    │         │  PARTITION 1 (3g.40)  │         │  🚀 NEW CAPACITY      │
  │                       │         │  ┌─────────────────┐  │         │                       │
  │                       │         │  │ ASR Workload    │  │         │  WORKLOAD: LLM (Replica)
  │  [████████████████]   │         │  │ [███░░░░░░]     │  │         │                       │
  │                       │         │  └─────────────────┘  │         │  [████████████████]   │
  │                       │         │          ||           │         │                       │
  │                       │         │    Hardware Wall      │         │  Doubled Throughput   │
  │                       │         │          ||           │         │                       │
  │                       │         │  PARTITION 2 (3g.40)  │         │                       │
  │                       │         │  ┌─────────────────┐  │         │                       │
  │                       │         │  │ TTS Workload    │  │         │                       │
  │                       │         │  │ [████░░░░░]     │  │         │                       │
  │                       │         │  └─────────────────┘  │         │                       │
  └───────────────────────┘         └───────────────────────┘         └───────────────────────┘
                                                ▲
                                                │
                                       Two models, One Card
                                        Zero Interference
```

##### Cost Impact

| Metric | Before | After | Delta |
|:---|:---|:---|:---|
| GPUs Required | 3 | 2 | -1 GPU |
| Monthly Cost (A100 @ $2/hr) | ~$4,320 | ~$2,880 | **-$1,440/mo** |
| Annualized Savings | | | **~$17,000** |
| Available for LLM Scale-out | 0 replicas | +1 replica | **2× throughput** |

---

#### Memory Bandwidth Analysis

Understanding why MIG provides deterministic performance requires examining HBM2e bandwidth allocation.

##### A100 Memory Subsystem

| Specification | Value |
|:---|:---|
| Total HBM2e Bandwidth | 2,039 GB/s |
| Memory Controllers | 8 |
| Bandwidth per Controller | ~255 GB/s |
| L2 Cache | 40 MB |

##### Bandwidth Partitioning

For `3g.40gb` profile:

$$
\text{Allocated Bandwidth} = \frac{3}{7} \times 2039 \approx 874 \text{ GB/s}
$$

Each MIG instance receives a **guaranteed** slice of memory bandwidth. This eliminates the "noisy neighbor" problem mathematically—there is no shared resource to contend for.

##### Latency Bounds

| Workload | Memory Access Pattern | Expected Latency |
|:---|:---|:---|
| ASR (Parakeet) | Streaming (sequential) | ~2-5ms per chunk |
| TTS (Magpie) | Autoregressive (random) | ~10-20ms per phoneme |

With MIG isolation, these latencies remain bounded regardless of concurrent workload intensity.

---

#### Production Considerations

##### Prerequisites

1. **GPU Operator:** Version ≥1.10 with `mig.strategy=mixed`
2. **Driver:** NVIDIA Driver ≥470.57.02
3. **Architecture:** Ampere (A100, A30) or Hopper (H100)

##### Operational Notes

- **Disruptive Change:** Enabling MIG requires GPU reset or node reboot
- **Profile Selection:** Choose profiles that match model memory footprint + 20% headroom
- **Monitoring:** Use `dcgm-exporter` with MIG-aware configuration

```yaml
# dcgm-exporter config for MIG
serviceMonitor:
  enabled: true
  additionalLabels:
    release: prometheus
env:
  - name: DCGM_EXPORTER_COLLECTORS
    value: "/etc/dcgm-exporter/dcp-metrics-included.csv"
```

##### Failure Modes

| Failure | Impact | Mitigation |
|:---|:---|:---|
| OOM in Slice A | Slice A pod crashes | Slice B unaffected |
| GPU hardware error | Both slices fail | Node-level failover |
| MIG config corruption | GPU requires reset | Store config in ConfigMap |

---

#### Conclusion

Default GPU allocation in Kubernetes treats accelerators as indivisible resources, leading to severe underutilization when deploying heterogeneous workloads.

**Key Takeaways:**

1. **Time-slicing** introduces latency jitter unsuitable for real-time audio
2. **MPS** lacks fault isolation and QoS guarantees
3. **MIG** provides hardware-enforced partitioning with deterministic performance

By consolidating lightweight ASR/TTS models onto a single partitioned GPU, we:
- Reclaimed an entire A100 for LLM scale-out
- Maintained latency SLOs through hardware isolation
- Reduced infrastructure cost by ~33%

In AI infrastructure, the goal is not just "fitting" models onto GPUs—it is managing GPUs as configurable compute fabrics to maximize utilization while preserving performance guarantees.

#### Reference

- [NVIDIA MIG User Guide](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/)
- [NVIDIA GPU Operator Documentation](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/index.html)
- [NVIDIA NIMs Deployment Guide](https://docs.nvidia.com/nim/)

