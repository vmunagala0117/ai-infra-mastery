# Lab 01 — Results

## Run Output

| n | measured (min) | predicted floor | ratio |
|---|---|---|---|
| 1,000,000 | 51.97 us | 37.50 us | 1.39x |
| 10,000,000 | 413.89 us | 375.00 us | 1.10x |
| 100,000,000 | 4031.04 us | 3750.00 us | 1.07x |

## Interpretation

The ratio between measured and predicted time drops from 1.39x → 1.10x → 1.07x
as `n` grows — confirming the fixed-overhead hypothesis. At small sizes,
PyTorch/CUDA kernel launch overhead (dispatcher + driver + queueing, roughly
constant in absolute terms) is a meaningful fraction of total time. As the
workload scales up, that fixed cost gets amortized over more actual
bandwidth-bound work and the measured time converges toward the theoretical
floor.

**Achieved bandwidth** (bytes moved / time taken), computed from the raw
measurements:

| n | achieved bandwidth | % of 320 GB/s peak |
|---|---|---|
| 1,000,000 | ~231 GB/s | ~72% |
| 10,000,000 | ~290 GB/s | ~91% |
| 100,000,000 | ~298 GB/s | ~93% |

**Takeaway:** small workloads rarely reach a GPU's advertised bandwidth or
compute specs, because fixed per-launch overhead dominates at small scale.
Real benchmarking needs enough work to move past that fixed cost before
hardware numbers become a fair comparison. This is also part of the real
justification for batching in production inference — amortizing fixed
overhead across many requests processed together.
