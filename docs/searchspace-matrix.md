# OmniRSI SearchSpace Matrix

**Branch:** `opus/meta-skill-bootstrap` | **Date:** 2026-10-15 | **Revision:** `3097788`

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
| **Configure** | Parallelism strategy (USP/CP/TP/PP), topology-aware card-group selection | `ascend-distributed-parallel` | USP activation requirement, masking upper-bound formula, AlltoAllV workaround | MindIE-SD dit-parallel-opt @ 8481d39b (9 reference files) | 🟢 |
| **Configure** | Model adapter: FP8 dequantization, KV-head replication, multimodal feature validation order | `ascend-model-adaptation` | vllm-ascend FP8/multimodal lessons | vllm-ascend @ bc8115ed | 🟢 |
| **Optimize** | Kernel-level: Triton/CuTe/TileIR writing, kernel tuning, fusion scope analysis | `kernel-writing-guide` | Framework routing (Triton vs CuTe vs AscendC), 6 universal correctness rules, verify-before-benchmark workflow | TRT-LLM kernel-triton-writing + kernel-cute-writing @ f33e0e5f | 🟢 |
| **Optimize** | Operator-level: CANN AscendC operator dev, op performance tuning (pypto-op-perf-tune) | — (candidate skills identified in ascend-npu-repo-map.md: `pypto-op-perf-tune`, `triton-op-designer`, `ascendc-precision-debug`) | ops-profiling iterative loop | awesome-ascend-skills; see `docs/ascend-npu-repo-map.md` | 🟡 |
| **Optimize** | Casebook of past NPU optimization decisions: AI-CPU fallback, memory-bound, comm-exposed, graph-compilation, scheduling idle gaps | `ascend-optimization-casebook`, `inference-optimization-casebook` | Five bottleneck classes with symptom→diagnosis→action→evidence schema | MindIE-SD profiling-analyze + dit-perf-opt + perf-gate + dit-parallel-opt @ 8481d39b | 🟢 |
| **Optimize** | Search policy: phase-locked round, negative memory, cost-aware selection, RRSI constraints, CRISP critical-step | `optimization-search-policy` | RRSI constraint propagation; CRISP critical-step classification | vllm-evolve @ 43cc82c; arXiv 2609.24972; arXiv 2608.01867 | 🟢 |
| **Verify** | Evidence taxonomy (L0 correctness → L5 production), claim-evidence matching | `dense-performance-evidence`, `rsi-evidence`, `claim-contract` | OmniRSI PASS/FAIL/INCONCLUSIVE protocol | OmniRSI @ 4c8e1285 | 🟢 |
| **Verify** | Same-window A/B protocol, noise floor, acceptance-state discipline | `ascend-optimization-casebook` (evidence-required section per case), `dense-performance-evidence`, `rsi-evidence` | MindIE-SD perf-gate protocol: exploration vs acceptance state, three-proof requirement | MindIE-SD perf-gate @ 8481d39b | 🟢 |

**Second-level targets in this row:** Kernel, Operator, KV Cache, Attention, Diffusion denoising steps, VAE/Codec, Graph compilation

---

## 2 · RAM / Host Memory

Host-side memory: CPU DRAM, pinned memory, page tables, PyTorch tensor lifecycle.

| Lifecycle step | What it covers | Candidate Skills | Cases / Experience | Evidence | Maturity |
|---|---|---|---|---|---|
| **Observe / Profile** | Host overhead in nsys: GPU idle ratio (>0.30 aggregate/>0.15 generation), host-prep exposed ratio (M3b>0.05), NVTX breakdown, inter-kernel gaps | `host-overhead-profiling` | GPU idle + host-prep verdict metrics; NVTX root-cause breakdown; regression comparison across two traces | TRT-LLM perf-host-analysis @ f33e0e5f | 🟢 |
| **Diagnose** | YES/NO bottleneck verdict; multi-rank straggler detection; NVTX-to-source function mapping | `host-overhead-profiling` | M3b>0.05 AND M3c>0.50 together confirm host-prep bottleneck | TRT-LLM perf-host-analysis @ f33e0e5f | 🟢 |
| **Configure** | Pinned memory for fast H2D (>1MB tensors where transfer is measured bottleneck), async H2D via non_blocking=True + stream + explicit synchronize(), Python GIL avoidance (ProcessPoolExecutor, Python 3.13 free-threading PEP 703), NUMA-aware allocation (numactl) | `host-memory-management` | All APIs verified from PyTorch docs, Python docs, numactl manpage | PyTorch docs, Python docs, PEP 703, numactl manpage | 🟢 |
| **Optimize** | line_profiler loop targeting `_prepare_tp_inputs`; NVTX regression triage to alternative targets (`_fetch_new_requests` +643%, `broadcast_requests`, `_update_requests` +75%) | `host-overhead-optimization` | Iterative profile→identify→optimize→verify loop | TRT-LLM perf-host-optimization @ f33e0e5f | 🟢 |
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
| **Diagnose** | Topology-aware card group validation: intra-domain (UB/HCCS) vs inter-domain (PCIe/SYS) | `ascend-distributed-parallel`, `ascend-device-health` | npu-smi topo output; legal card-group rules; Health=OK not sufficient | MindIE-SD dit-parallel-opt @ 8481d39b | 🟢 |
| **Configure** | Collective-in-graph capture (HCCL graph mode), comm-stream masking; masking formula `1-1/n` when F≥C | `ascend-distributed-parallel` | Masking upper bound derivation; AlltoAllV defect workaround | MindIE-SD @ 8481d39b | 🟢 |
| **Optimize** | USP activation requirement (ParallelConfig wrapping), strategy selection, AlltoAllV workaround | `ascend-distributed-parallel` | USP must set ParallelConfig; verify allToAll appears not just allGather | MindIE-SD dit-parallel-opt @ 8481d39b (9 ref files) | 🟢 |
| **Verify** | Masking ratio measurement, before/after comm-fraction comparison; strategy activation attribution | `ascend-distributed-parallel` (attribution method), `ascend-performance-analysis`, `dense-performance-evidence` | Step-trace four-tuple as ground truth; scope-effectiveness-check.md | MindIE-SD perf-gate + dit-parallel-opt | 🟢 |

**Second-level targets:** AllReduce, AllGather, AllToAll, P2P ring; TP/SP/PP/EP collective patterns

---

## 5 · OS / System

System-level: scheduler, DVFS, process isolation, environment variables, driver state.

| Lifecycle step | What it covers | Candidate Skills | Cases / Experience | Evidence | Maturity |
|---|---|---|---|---|---|
| **Observe / Profile** | DVFS effect on first-run timing, process-level NPU ownership (proc-mem via npu-smi) | `ascend-operator-profiling` (--warm-up rationale), `ascend-device-health` (proc-mem check) | DVFS: always warm up before measuring; proc-mem occupancy check | awesome-ascend-skills ops-profiling; MindIE-SD dit-parallel-opt @ 8481d39b | 🟢 |
| **Diagnose** | Shared-machine contamination: neighbor load, inter-window drift, temperature variation | `ascend-performance-analysis` (measurement discipline section) | Same-window A/B required; cross-window delta ≈ real gain magnitude | MindIE-SD perf-gate @ 8481d39b | 🟢 |
| **Diagnose** | SIGKILL residual state: per-step time uniformly elevated by ~1 order of magnitude | `ascend-device-health` | npu-smi Health=OK does not prove card is clean; 4-step smoke required | MindIE-SD dit-parallel-opt @ 8481d39b | 🟢 |
| **Configure** | Card-group selection, env variables (`HCCL_OP_EXPANSION_MODE=AIV`), proc-mem occupancy check | `ascend-device-health` | Card-group change = contract change; all baselines must be re-measured on new group | MindIE-SD dit-parallel-opt + perf-gate @ 8481d39b | 🟢 |
| **Configure** | Env-preflight: verify device availability, driver health, Python env before measurement | `performance-ceiling-analysis` (env-preflight references TRT-LLM exec-env-check) | TRT-LLM exec-env-check pattern | TRT-LLM @ f33e0e5f | 🟡 |
| **Optimize** | — | — | — | — | 🔴 |
| **Verify** | Card health check sequence: npu-smi Health, 4-step smoke verify, proc-mem occupancy | `ascend-device-health` | Health=OK ≠ card usable; SIGKILL residual state only detectable via smoke run | MindIE-SD @ 8481d39b | 🟢 |

**Second-level targets:** NPU driver, CANN runtime, container/process isolation

---

## Gap Summary

| Gap | Row | Priority | Next action |
|-----|-----|----------|-------------|
| Operator-level tuning (pypto-op-perf-tune, AscendC tiling) | Accelerator / Optimize | Medium | Mine awesome-ascend-skills pypto-op-perf-tune and cannbot/model-infer-fusion; port to OmniRSI format |
| Storage I/O profiling and optimization | SSD / Observe, Optimize | Low | No source material; defer until a strong source appears |
| OS/System Optimize (CPU affinity, NUMA process placement, cgroups) | OS / Optimize | Low | `host-memory-management` covers NUMA allocation; dedicated OS-level scheduling skill needs new sources |

**Resolved gaps since initial matrix (all high/medium priority cleared):**
- ~~Host overhead profiling + optimization skills~~ → `host-overhead-profiling`, `host-overhead-optimization` ✅
- ~~Parallelism strategy skill~~ → `ascend-distributed-parallel` ✅
- ~~NPU optimization casebook~~ → `ascend-optimization-casebook` ✅
- ~~SIGKILL / card-health recovery~~ → `ascend-device-health` ✅
- ~~Same-window A/B acceptance skill~~ → covered in `ascend-optimization-casebook` (per-case evidence schema) ✅
- ~~Kernel writing skill (Triton-Ascend / AscendC TileIR / CuTe)~~ → `kernel-writing-guide` ✅
- ~~Host memory lifecycle (pinned memory / GIL / H2D async)~~ → `host-memory-management` ✅

---

## PR coverage table

Use this table in the PR description for `opus/meta-skill-bootstrap`.

| Search Space | Observe / Profile | Diagnose | Configure | Optimize | Verify |
|---|---|---|---|---|---|
| **Accelerator** | 🟢 `ascend-performance-analysis`, `ascend-operator-profiling`, `dense-performance-evidence` | 🟢 `ascend-performance-analysis`, `performance-ceiling-analysis` | 🟢 `ascend-graph-mode`, `ascend-model-adaptation`, `ascend-distributed-parallel` | 🟢 `ascend-optimization-casebook`, `optimization-search-policy`, `kernel-writing-guide` | 🟢 `dense-performance-evidence`, `rsi-evidence`, `claim-contract` |
| **RAM / Host** | 🟢 `host-overhead-profiling` | 🟢 `host-overhead-profiling` | 🟢 `host-memory-management` | 🟢 `host-overhead-optimization` | 🟡 (inherited from Accelerator row) |
| **SSD / Storage** | 🔴 | 🟡 `ascend-graph-mode` (startup/loading incidental) | 🟡 `ascend-model-adaptation` (FP8 loading) | 🔴 | 🔴 |
| **Communication** | 🟢 `ascend-performance-analysis` | 🟢 `ascend-performance-analysis`, `ascend-distributed-parallel` | 🟢 `ascend-distributed-parallel` | 🟢 `ascend-distributed-parallel` | 🟢 `ascend-distributed-parallel`, `ascend-performance-analysis` |
| **OS / System** | 🟢 `ascend-operator-profiling` (DVFS), `ascend-device-health` (proc-mem) | 🟢 `ascend-device-health`, `ascend-performance-analysis` | 🟢 `ascend-device-health` | 🔴 | 🟢 `ascend-device-health` |

**Optimization targets added in this branch:**
- Kernel profiling (CANN CSV pipeline, 5-layer analysis, operator-level msprof loop)
- Graph compilation (npugraph_ex capture-replay, FX passes, memory reuse)
- Model adapter (FP8, KV replication, multimodal)
- Distributed parallelism (USP/CP strategy, topology rules, masking formula, attribution)
- Host overhead (GPU idle / host-prep verdict, NVTX root-cause, line_profiler loop)
- Host memory (pinned memory, async H2D, GIL avoidance, NUMA allocation)
- Device health (topology validation, SIGKILL detection, 4-step smoke)
- NPU optimization casebook (5 bottleneck classes: AI-CPU fallback, memory-bound, comm-exposed, graph-compile, scheduling)
- Kernel writing (Triton/CuTe/AscendC routing, 6 correctness rules, verify-before-benchmark workflow)
- Search policy (phase-locked round, RRSI constraints, CRISP critical-step)
- Skill evolution (Trace2Skill pool consolidation, SkillOpt bounded edit)
- Benchmark design (goodput, three-regime, OmniRSI evidence trail)
- Repository intelligence (LTC temporal split, Magpie patterns, Change2Task, Chronos)

**Skill count:** 19 candidate skills. All `status: candidate`. None promoted.
