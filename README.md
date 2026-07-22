# AI Infra Mastery Journal

Personal learning journal working toward senior/staff-level AI infrastructure,
distributed training systems, GPU performance engineering, and MLOps
platforms — targeting roles at companies where this work happens at the
hardware-and-scale frontier (NVIDIA-tier infra engineering).

Curriculum adapted from the [ai-infra-curriculum](https://github.com/orgs/ai-infra-curriculum/repositories)
GitHub org — primarily `ai-infra-senior-engineer-learning` as the technical
spine, with reasoning frameworks folded in from `ai-infra-architect-learning`
and `ai-infra-senior-architect-learning` at the points where they naturally
apply.

**Latest status → see [`PROGRESS.md`](./PROGRESS.md)**

---

## How This Roadmap Is Structured

Two things shaped the sequencing, worth stating explicitly since they explain
why the order looks the way it does:

1. **Hardware reality up, abstractions down — never the other way around.**
   GPU/CUDA fundamentals come before Kubernetes, even though Kubernetes is
   listed as "Module 201" in the source curriculum, because Kubernetes in this
   context spends most of its energy orchestrating GPU workloads. Learning the
   orchestration layer before understanding what's being orchestrated leads to
   memorized YAML instead of real judgment.
2. **Depth and altitude on the same subject, not two separate tracks.** Rather
   than doing the full senior-engineer track and then separately doing the
   full architect and senior-architect tracks end to end, the architecture and
   tradeoff-reasoning material gets folded in at the point where each
   technical topic naturally raises a "when is this actually worth it"
   question — e.g., architect-track multi-cloud tradeoff reasoning lands right
   after the technical multi-cloud deployment mechanics, not months later as
   an unrelated unit.

**Pace:** intensive, targeting real depth over 2-4 months. Hands-on labs run
on an Azure T4 GPU instance; results and code live in `labs/`, conceptual
notes live in `notes/`, mirrored phase-by-phase.

---

## Curriculum Overview

| Phase | Topic | Source Module(s) | Status |
|---|---|---|---|
| 0 | GPU & CUDA Foundations | Module 203 (taught as true fundamentals) | 🟡 In progress |
| 1 | Distributed Training Systems | Module 202 | ⬜ Not started |
| 2 | Kubernetes for ML Infra | Module 201 | ⬜ Not started |
| 3 | Model Optimization & High-Performance Inference | Module 204 | ⬜ Not started |
| 4 | Production Systems at Scale | Modules 205-207 | ⬜ Not started |
| 5 | Architecture & Leadership (overlay, ongoing from Phase 1 onward) | Modules 208-210 + architect/senior-architect tracks | ⬜ Not started |

---

## Phase 0 — GPU & CUDA Foundations *(current phase)*

**Goal:** build true first-principles understanding of GPU hardware and the
CUDA execution model — not vocabulary, but the ability to reason about *why*
a kernel is fast or slow before ever profiling it.

**Covered so far** (see `notes/phase0-gpu-cuda/phase0-gpu-cuda-foundations.md`):
- Why GPUs exist — the CPU vs. GPU design-philosophy mental model
- Memory-bound vs. compute-bound as a diagnostic lens
- The full journey of one instruction: host → PCIe → grid scheduler → SM → warp → core → memory
- SIMT execution model, warp scheduling, thread block → warp splitting
- Why SMs are internally partitioned (wiring cost, scheduling throughput, latency hiding)
- CUDA core vs. warp (physical vs. logical), register file vs. instructions
- GPU memory hierarchy (registers → shared memory/L1 → L2 → global memory),
  arithmetic intensity, and calculating a memory-bound performance floor by hand

**Remaining in this phase:**
- Hands-on: real CUDA/PyTorch kernels on the T4, verified against
  hand-calculated performance floors (lab in progress — `labs/phase0-gpu-cuda/01-vector-add-timing/`)
- Multi-GPU communication fundamentals: NCCL, all-reduce, ring vs. tree topologies

---

## Phase 1 — Distributed Training Systems *(Module 202)*

Data parallelism vs. model/tensor/pipeline parallelism, PyTorch DDP and FSDP,
Ray Train, fault-tolerant checkpointing, distributed hyperparameter search.

## Phase 2 — Kubernetes for ML Infra *(Module 201)*

Custom operators for ML jobs, gang scheduling for distributed training, GPU
device plugins and time-slicing, service mesh, multi-cluster federation.
Vanilla Kubernetes concepts move fast (prior general infra experience);
GPU-specific and ML-specific scheduling gets the real depth.

## Phase 3 — Model Optimization & High-Performance Inference *(Module 204)*

Quantization, pruning, distillation, TensorRT, ONNX Runtime, compiler-level
optimization (TVM/XLA). Where training infra and serving infra diverge as
disciplines.

## Phase 4 — Production Systems at Scale *(Modules 205-207)*

Multi-cloud architecture, advanced MLOps (feature stores, model registries,
governance), observability/SRE for ML systems.

## Phase 5 — Architecture & Leadership *(overlay, Modules 208-210 + architect tracks)*

Build-vs-buy and multi-cloud cost/reliability tradeoff frameworks, RFC and
architecture-doc writing, cross-org technical standards, infrastructure-as-code,
security/compliance. Folded in progressively from Phase 1 onward rather than
treated as a single block at the end.

---

## Capstone Projects (from source curriculum)

- Distributed training platform on Ray/Kubernetes
- TensorRT-optimized model serving system
- Multi-region ML platform
- Custom Kubernetes operator (Go)

---

## Repo Structure

```
notes/    — conceptual, phase-by-phase, relatively stable once written
labs/     — hands-on code + results, phase-by-phase, actively iterated
```

Each lab folder includes its own `README.md` (what it demonstrates, how to
run it) and `results.md` (what was actually observed on real hardware —
specific numbers, not just "it worked").
