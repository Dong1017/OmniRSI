# Training vs Inference Optimization: Boundary Map

## Shared capability categories

These apply to both training and inference systems.

### Profiling and metrics collection
- Hardware profiler interpretation: nsys timeline parsing, kernel duration extraction, host-device event correlation.
- Equivalence across vendors: GPU concepts (nsys, ncu, SM SOL%) map to NPU equivalents (msprof, torch_npu.profiler, AI Core utilization, MTE ratios).
- **Source (observed):** `ascend-performance-analysis/SKILL.md` §GPU→NPU concept mapping; `perf-torch-cuda-graphs/SKILL.md` requirements.

### Roofline analysis and bottleneck classification
- Analytical ceiling (FLOPs, memory traffic, arithmetic intensity, ridge point).
- Empirical peak measurement (ERT-style kernel sweeps, L1/L2/DRAM bandwidth per kernel, occupancy metrics).
- Bottleneck classification from profiler counters: memory-bound vs compute-bound vs launch-overhead vs communication-bound vs sync-bound.
- **Source (observed):** `performance-ceiling-analysis/SKILL.md` §L1–L3 ceilings, bottleneck classification table.

### Evidence hierarchy and measurement discipline
- Six evidence levels (correctness, microbench, component profile, isolated scenario, multi-regime, production A/B).
- Claim-evidence matching: what each level proves and cannot prove.
- Negative evidence preservation and reproducibility discipline.
- **Source (observed):** `dense-performance-evidence/SKILL.md` §evidence hierarchy, claim-evidence matching table.

### Kernel launch overhead reduction
- CUDA Graphs and graph capture/replay (API selection: torch.compile, torch.cuda.make_graphed_callables, TE make_graphed_callables, Megatron CudaGraphManager, manual torch.cuda.graph).
- Capture workflows, dynamic pattern handling, stream-based execution.
- **Source (observed):** `perf-torch-cuda-graphs/SKILL.md` §when to use, API selection guide.

### Host-device synchronization elimination
- Detection methods: PyTorch sync debug mode, Nsight Systems profiling.
- False vs true dependencies; sync-free alternatives (.item(), .cpu(), boolean indexing, torch.tensor on CUDA).
- Non-blocking APIs and stream semantics.
- **Source (observed):** `perf-torch-sync-free/SKILL.md` §when to use, workflow.

### Prior-art decision support
- Casebook pattern: queryable record of past optimizations (bottleneck class, signals, mechanism, applies_when, does_not_apply, expected_effect, evidence).
- Query protocol (free-text grep > structured fields).
- Case maintenance and adaptation.
- **Source (observed):** `inference-optimization-casebook/SKILL.md` §casebook pattern, case structure, query protocol.

## Training-specific categories

These do not apply to inference systems.

### Gradient computation and backward pass analysis
- Backward pass profiling: activation recomputation, gradient accumulation, weight gradient aggregation.
- Gradient checkpointing strategies: layer-wise, segment-wise, full-recompute thresholds.
- **Source (inferred):** Training involves backpropagation; inference does not. No matching skill found in existing omnirsi skills or MindIE-SD/TensorRT-LLM training-adjacent skills to cite. This is a gap.

### Optimizer state and parameter updates
- Optimizer memory footprint: momentum buffers, adaptive learning rate states (Adam m/v, AdamW, LAMB).
- Mixed-precision training dynamics: loss scaling, gradient clipping, accumulated gradient precision.
- Parameter update pipelining: distributed optimizer states (ZeRO stages 1–3 offloading behavior).
- **Source (inferred):** Optimizer state is training-only; inference reads frozen weights. No existing skill covers this; inferred from absence in omnirsi baseline.

### Distributed training collective patterns
- DDP/FSDP/ZeRO gradient AllReduce synchronization and timing.
- Activation sharding vs gradient sharding tradeoffs.
- Communication patterns: parameter broadcast, gradient aggregation, optimizer state synchronization.
- **Source (inferred):** Multi-card training has distinct communication phases (gradient sync, parameter update broadcast). `dit-parallel-opt/SKILL.md` addresses inference-specific parallelism (USP, CP, TP) but not training-specific DDP/FSDP patterns. Inferred as training-only.

### Training-specific optimization checkpoints
- Which metrics to checkpoint (loss, gradient norm, parameter statistics, learning rate schedule state).
- Resume from checkpoint: gradient accumulation counter, optimizer state recovery, data loader position.
- **Source (inferred):** Inference does not resume mid-generation from checkpoints; training does. Inferred as training-specific.

## Inference-specific categories

These do not apply to training systems.

### KV cache management
- Cache allocation and eviction policies: token budget, memory-aware scheduling.
- KV cache layout optimizations: paged attention, contiguous vs sparse layouts, offloading to CPU.
- Multi-request cache sharing and reuse.
- **Source (inferred):** KV cache appears in inference only (frozen autoregressive generation). `performance-ceiling-analysis/SKILL.md` §stage breakdown mentions "KV traffic dominates decode-heavy workloads." Explicit in inference context, absent from training.

### Prefill / decode split and continuous batching
- Prefill-heavy vs decode-heavy workload classification.
- Prefill kernel optimizations: batched prompt encoding, attention kernel fusion.
- Decode kernel optimizations: single-token generation, cache-resident kernels.
- Continuous batching and request scheduling: reordering, priority queues, SLO enforcement.
- **Source (observed):** `performance-ceiling-analysis/SKILL.md` §stage breakdown distinguishes prefill and decode stages. Training has no prefill/decode distinction.

### Speculative decoding and sampling optimization
- Draft model selection and verification.
- Acceptance rate tuning and parallelism.
- **Source (inferred):** Speculative decoding is inference serving only. Not present in omnirsi baseline or training skills read.

### Request scheduling and multi-user load patterns
- Throughput-saturating vs latency-sensitive load regimes.
- Batching strategies under variable concurrency.
- Request mix and burstiness handling.
- **Source (observed):** `dense-performance-evidence/SKILL.md` §level 4 (multi-regime serving benchmark) and level 5 (production A/B) distinguish serving-specific concurrency patterns. Training uses fixed batch sizes during an epoch.

## Skill design recommendation

**Recommendation:** Create a `training-systems-optimization` skill as a **peer reference document only**, not as a coordinator or decision router.

**Rationale:**
- OmniRSI's existing skills (`performance-ceiling-analysis`, `dense-performance-evidence`, `ascend-performance-analysis`) are **hardware- and evidence-centric**, not workload-centric. They apply equally to training and inference.
- Training-specific skills needed (gradient checkpointing, DDP/FSDP communication patterns, optimizer state profiling) do not yet exist in omnirsi. They would belong in a **training domain module** (analogous to MindIE-SD's `dit-perf-opt` and `dit-parallel-opt`), not in omnirsi's current inference-focused architecture.
- A training reference document (`training-systems-optimization-map.md` or similar) is valuable as a **capability inventory and decision support**, but should not route tasks the way `inference-optimization-casebook` does. Instead, it should signal what training-specific skills are missing and where training and inference profiling converge.

**Do not create:** A single training-specific skill that tries to cover gradient checkpointing, DDP, and optimizer state all at once. Those are distinct responsibilities requiring separate expertise.

## OmniRSI routing

### Shared capability — which existing skill already covers it

| Category | Covered by | Routing |
|----------|-----------|---------|
| Profiling & metrics | `ascend-performance-analysis` | Use directly for NPU; GPU equivalents in TensorRT-LLM |
| Roofline & bottleneck classification | `performance-ceiling-analysis` | Use directly; extends to training if provided stage breakdown |
| Evidence hierarchy | `dense-performance-evidence` | Use directly; all 6 levels apply to training measurement |
| Kernel launch overhead | `perf-torch-cuda-graphs` (TensorRT-LLM) | Use directly; not in omnirsi but shared library |
| Sync elimination | `perf-torch-sync-free` (TensorRT-LLM) | Use directly; shared library |
| Prior-art decisions | `inference-optimization-casebook` | Extend to training; same pattern, different cases |

### Training-specific gaps to address

1. **Gradient checkpointing & backward pass profiling** — needs new skill or training-domain module reference.
2. **DDP/FSDP/ZeRO communication patterns** — needs new skill; differs structurally from inference parallelism.
3. **Optimizer state and mixed-precision training** — needs new skill; not covered by existing omnirsi baseline.
4. **Training checkpoint management** — needs reference document; no counterpart in inference.

### Recommendation for next phase

If OmniRSI extends to training optimization:
1. Reuse `dense-performance-evidence`, `performance-ceiling-analysis`, and `ascend-performance-analysis` as foundation.
2. Create new training-specific skills in a **training domain module** for gradient, communication, and optimizer patterns.
3. Extend `inference-optimization-casebook` pattern to training cases (separate casebook, same structure).
4. Link both via a high-level `training-systems-optimization-reference.md` (purely informational, not a router).
