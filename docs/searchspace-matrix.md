# OmniRSI SearchSpace Matrix

**Branch:** `opus/meta-skill-bootstrap` | **Date:** 2026-10-15 | **Revision:** `3ec0fbb`

Top-level rows are hardware resource layers.  
Second-level optimization targets (Kernel, Operator, KV Cache, Attention, Runtime, Distributed, Serving, VAE/Codec) live **inside** the row that owns their bottleneck.

**Maturity key:** 🟢 covered (candidate skill + source evidence) · 🟡 partial (source read, no dedicated skill) · 🔴 gap (no source, no skill)

---

## 1 · Graphics Card / Accelerator

The primary compute and HBM resource. Most current work lives here.

| Lifecycle step | What it covers | Candidate Skills | Cases / Experience | Evidence | Maturity |
|---|---|---|---|---|---|
| **Observe / Profile** | System-level GPU/NPU timeline, per-kernel stats, SOL%, AI Core utilization ratios, step-trace four-tuple | `ascend-performance-analysis`, `dense-performance-evidence` (L2) | CANN CSV pipeline, MindIE-SD profiling-analyze lessons | MindIE-SD @ 8481d39b; TRT-LLM nsys/ncu @ f33e0e5f | 🟢 |
| **Observe / Profile** | Single-operator benchmark (msprof op, 8 CSV metrics, per-core occupancy) | `ascend-operator-profiling` | ops-profiling loop | awesome-ascend-skills @ 75ec89ce | 🟢 |
| **Diagnose** | Bottleneck classification: compute-bound vs memory-bound vs AI-CPU fallback vs comm-exposed | `ascend-performance-analysis` (5-layer pipeline), `performance-ceiling-analysis` (L1–L3) | SOL% / mte2-ratio / mac-ratio heuristic | MindIE-SD; TRT-LLM roofline @ f33e0e5f; LLM-Viewer @ 1893e4b5 | 🟢 |
| **Diagnose** | Roofline model, arithmetic intensity, hardware table entry for new accelerator | `performance-ceiling-analysis` (L1) | FLOPs formulas per layer type | LLM-Viewer `roofline_model.py`, `hardware_params.py` | 🟢 |
| **Configure** | Graph mode (npugraph_ex / ACLGraph): capture-replay, FX pass, memory reuse, static kernel compile | `ascend-graph-mode` | TorchAir document-first discipline | awesome-ascend-skills torch-npugraph-ex-knowledge @ 75ec89ce | 🟢 |
| **Configure** | Parallelism strategy (USP/CP/TP/PP), topology-aware card-group selection | — (covered in corpus, not yet its own skill) | dit-parallel-opt HCCL topology rules | MindIE-SD dit-parallel-opt @ 8481d39b | 🟡 |
| **Configure** | Model adapter: FP8 dequantization, KV-head replication, multimodal feature validation order | `ascend-model-adaptation` | vllm-ascend FP8/multimodal lessons | vllm-ascend @ bc8115ed | 🟢 |
| **Optimize** | Kernel-level: Triton/CuTe/TileIR writing, kernel tuning, fusion scope analysis | — (TRT-LLM kernel-* skills not ported to OmniRSI yet; akg-mlir candidate identified) | TRT-LLM kernel casebook | TRT-LLM @ f33e0e5f; akg @ 5aa15f3f | 🟡 |
| **Optimize** | Operator-level: CANN AscendC operator dev, op performance tuning (pypto-op-perf-tune) | — (candidate skills identified in ascend-npu-repo-map.md: `pypto-op-perf-tune`, `triton-op-designer`, `ascendc-precision-debug`) | ops-profiling iterative loop | awesome-ascend-skills; see `docs/ascend-npu-repo-map.md` | 🟡 |
| **Optimize** | Casebook of past optimization decisions (prior-art matching, adapt-not-copy) | `inference-optimization-casebook` | TRT-LLM casebook precedents (CUDA), NPU casebook empty | TRT-LLM perf-optimization-casebook @ f33e0e5f | 🟡 (CUDA populated; NPU empty) |
| **Optimize** | Search policy: phase-locked round, negative memory, cost-aware selection, RRSI constraints, CRISP critical-step | `optimization-search-policy` | RRSI constraint propagation; CRISP critical-step classification | vllm-evolve @ 43cc82c; arXiv 2609.24972; arXiv 2608.01867 | 🟢 |
| **Verify** | Evidence taxonomy (L0 correctness → L5 production), claim-evidence matching | `dense-performance-evidence`, `rsi-evidence`, `claim-contract` | OmniRSI PASS/FAIL/INCONCLUSIVE protocol | OmniRSI @ 4c8e1285 | 🟢 |
| **Verify** | Same-window A/B protocol, noise floor, acceptance-state discipline | — (covered in perf-gate corpus, not yet ported) | MindIE-SD perf-gate protocol | MindIE-SD perf-gate @ 8481d39b | 🟡 |

**Second-level targets in this row:** Kernel, Operator, KV Cache, Attention, Diffusion denoising steps, VAE/Codec, Graph compilation

---

## 2 · RAM / Host Memory

Host-side memory: CPU DRAM, pinned memory, page tables, PyTorch tensor lifecycle.

| Lifecycle step | What it covers | Candidate Skills | Cases / Experience | Evidence | Maturity |
|---|---|---|---|---|---|
| **Observe / Profile** | Host overhead in nsys: GPU idle ratio, host-prep exposed ratio, inter-kernel gaps | — (TRT-LLM perf-host-analysis not ported) | TRT-LLM host-analysis NVTX breakdown | TRT-LLM @ f33e0e5f | 🟡 |
| **Diagnose** | CPU bottleneck detection: does host overhead gate GPU work? | — (TRT-LLM perf-host-analysis DETECT phase not ported) | GPU idle > 30% heuristic | TRT-LLM perf-host-analysis | 🟡 |
| **Configure** | CUDA pinned memory, async H2D transfers, Python GIL avoidance | — | — | — | 🔴 |
| **Optimize** | Host-side optimizations: reduce sync points, optimize PyExecutor scheduling, batch request prep | — (TRT-LLM perf-host-optimization not ported) | `_prepare_tp_inputs` profiling loop | TRT-LLM @ f33e0e5f | 🟡 |
| **Verify** | — | `dense-performance-evidence` (evidence levels apply) | — | — | 🟡 (inherited from accelerator row) |

**Second-level targets:** PyTorch tensor lifecycle, request scheduling, KV cache CPU offload

---

## 3 · SSD / Storage

Storage I/O: model weight loading, checkpoint save/restore, profiling artifact I/O, dataset prefetch.

| Lifecycle step | What it covers | Candidate Skills | Cases / Experience | Evidence | Maturity |
|---|---|---|---|---|---|
| **Observe / Profile** | — | — | — | — | 🔴 |
| **Diagnose** | Slow startup (graph capture + weight loading dominating): distinguish compile warmup from I/O | `ascend-graph-mode` (mentions slow startup), `ascend-model-adaptation` (dummy→real sequence) | ACLGraph startup cost is compile + capture, not I/O bound; FP8 weight loading pitfalls | MindIE-SD; vllm-ascend | 🟡 (incidental coverage only) |
| **Configure** | Checkpoint format for fast loading (FP8 paired tensors, shard loading order) | `ascend-model-adaptation` | FP8 paired-tensor loading pattern | vllm-ascend @ bc8115ed | 🟡 |
| **Optimize** | — | — | — | — | 🔴 |
| **Verify** | — | — | — | — | 🔴 |

**Second-level targets:** Model weight sharding, checkpoint I/O, profiling output archival

---

## 4 · Communication Card / Network

Inter-device and inter-node communication: HCCL (Ascend), NCCL (CUDA), InfiniBand, RoCE.

| Lifecycle step | What it covers | Candidate Skills | Cases / Experience | Evidence | Maturity |
|---|---|---|---|---|---|
| **Observe / Profile** | Communication time in step-trace four-tuple, HCCL per-collective timing, `communication.json` | `ascend-performance-analysis` (Layer 4) | HCCL single-collective latency varies ~2× across devices | MindIE-SD profiling-analyze @ 8481d39b | 🟢 |
| **Diagnose** | Exposed communication: `effective_comm = Communication - Overlapped`; communication-bottleneck class | `ascend-performance-analysis` (L1 step-trace derivation) | Overlapping vs exposed comm in step_trace_time.csv | MindIE-SD | 🟢 |
| **Diagnose** | Topology-aware card group validation: intra-domain (UB/HCCS) vs inter-domain (PCIe/SYS) | `ascend-graph-mode` (topology rules), MindIE-SD dit-parallel-opt (not yet ported) | npu-smi topo output; legal card-group rules | MindIE-SD dit-parallel-opt @ 8481d39b | 🟡 |
| **Configure** | Collective-in-graph capture (HCCL graph mode), comm-stream masking | `ascend-graph-mode` (mentions communication_graph.md) | Masking upper bound: 1–1/n only when truly pipelined | MindIE-SD | 🟡 |
| **Optimize** | Comm-masking stream isolation, USP/CP parallelism choice, AlltoAllV defect workaround | — (dit-parallel-opt not ported) | HCCL same-island rules, AlltoAllV workaround | MindIE-SD dit-parallel-opt @ 8481d39b | 🟡 |
| **Verify** | Masking ratio measurement, before/after comm-fraction comparison | `ascend-performance-analysis`, `dense-performance-evidence` | Step-trace four-tuple as ground truth | MindIE-SD perf-gate | 🟡 |

**Second-level targets:** AllReduce, AllGather, AllToAll, P2P ring; TP/SP/PP/EP collective patterns

---

## 5 · OS / System

System-level: scheduler, DVFS, process isolation, environment variables, driver state.

| Lifecycle step | What it covers | Candidate Skills | Cases / Experience | Evidence | Maturity |
|---|---|---|---|---|---|
| **Observe / Profile** | DVFS effect on first-run timing, process-level NPU ownership (proc-mem via npu-smi) | `ascend-operator-profiling` (--warm-up rationale) | DVFS: always warm up before measuring; skip N iterations | awesome-ascend-skills ops-profiling | 🟡 |
| **Diagnose** | Shared-machine contamination: neighbor load, inter-window drift, temperature variation | `ascend-performance-analysis` (measurement discipline section) | Same-window A/B required; cross-window delta ≈ real gain magnitude | MindIE-SD perf-gate @ 8481d39b | 🟢 |
| **Diagnose** | SIGKILL residual state: per-step time uniformly elevated by ~1 order of magnitude | — (referenced in dit-parallel-opt corpus, not yet its own skill) | npu-smi Health=OK does not prove card is clean; requires smoke run | MindIE-SD | 🟡 |
| **Configure** | Environment variables: `HCCL_OP_EXPANSION_MODE=AIV`, proxy settings, container flags | `ascend-graph-mode` (mentions AIV flag), `ascend-model-adaptation` | HCCL AIV mode recommended with graph capture | MindIE-SD | 🟡 |
| **Configure** | Env-preflight: verify device availability, driver health, Python env before measurement | `performance-ceiling-analysis` (env-preflight references TRT-LLM exec-env-check) | TRT-LLM exec-env-check pattern | TRT-LLM @ f33e0e5f | 🟡 |
| **Optimize** | — | — | — | — | 🔴 |
| **Verify** | Card health check sequence: npu-smi Health, 4-step smoke verify, proc-mem occupancy | — (in dit-parallel-opt corpus, not yet ported) | Health=OK ≠ card usable; smoke run required | MindIE-SD | 🟡 |

**Second-level targets:** NPU driver, CANN runtime, container/process isolation

---

## Gap Summary

| Gap | Row | Priority | Next action |
|-----|-----|----------|-------------|
| NPU optimization casebook (cases for memory-bound / comm-exposed / AI-CPU fallback) | Accelerator / Optimize | High | Mine MindIE-SD `model-auto-optimization` references; port existing perf-gate + dit-perf-opt cases |
| Host overhead profiling + optimization skills | RAM / Observe, Diagnose, Optimize | High | Port TRT-LLM `perf-host-analysis` and `perf-host-optimization` patterns into OmniRSI skills |
| Kernel writing skill (Triton-Ascend / AscendC TileIR) | Accelerator / Optimize | Medium | Candidates identified: `triton-op-designer`, `ascendc-precision-debug` (see ascend-npu-repo-map.md) |
| Same-window A/B acceptance-state skill | Accelerator / Verify | Medium | Port MindIE-SD perf-gate skill content (already in corpus; needs OmniRSI format) |
| Parallelism strategy skill | Accelerator / Configure | Medium | Port dit-parallel-opt skill (already read; substantial content) |
| SIGKILL / card-health recovery procedure | OS / Diagnose, Verify | Medium | Short note skill from dit-parallel-opt § card-group rules |
| Storage I/O profiling | SSD / Observe, Optimize | Low | No source material; needs new investigation |
| Host memory lifecycle (pinned memory, GIL, H2D async) | RAM / Configure, Optimize | Low | No source material |

---

## PR coverage table

Use this table in the PR description for `opus/meta-skill-bootstrap`.

| Search Space | Observe / Profile | Diagnose | Configure | Optimize | Verify |
|---|---|---|---|---|---|
| **Accelerator** | 🟢 `ascend-performance-analysis`, `ascend-operator-profiling`, `dense-performance-evidence` | 🟢 `ascend-performance-analysis`, `performance-ceiling-analysis` | 🟢 `ascend-graph-mode`, `ascend-model-adaptation` | 🟡 casebook (CUDA only), kernel skills gap | 🟢 `dense-performance-evidence`, `rsi-evidence`, `claim-contract` |
| **RAM / Host** | 🟡 (TRT-LLM source read, not ported) | 🟡 | 🔴 | 🟡 | 🟡 (inherited) |
| **SSD / Storage** | 🔴 | 🟡 (startup/loading incidental) | 🟡 `ascend-model-adaptation` (FP8 loading) | 🔴 | 🔴 |
| **Communication** | 🟢 `ascend-performance-analysis` | 🟢 `ascend-performance-analysis` | 🟡 | 🟡 | 🟡 |
| **OS / System** | 🟡 `ascend-operator-profiling` (DVFS) | 🟢 `ascend-performance-analysis` (shared-machine discipline) | 🟡 | 🔴 | 🟡 |

**Optimization targets added in this branch:**
- Kernel profiling (CANN CSV pipeline, 5-layer analysis, operator-level msprof loop)
- Graph compilation (npugraph_ex capture-replay, FX passes, memory reuse)
- Model adapter (FP8, KV replication, multimodal)
- Search policy (phase-locked round, RRSI constraints, CRISP critical-step)
- Skill evolution (Trace2Skill pool consolidation, SkillOpt bounded edit)
- Benchmark design (goodput, three-regime, OmniRSI evidence trail)
- Repository intelligence (LTC temporal split, Magpie patterns, Change2Task, Chronos)

**Skill count:** 12 candidate skills. All `status: candidate`. None promoted.
