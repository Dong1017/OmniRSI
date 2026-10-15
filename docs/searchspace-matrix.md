# OmniRSI SearchSpace Matrix

**Branch:** `opus/meta-skill-bootstrap` | **Date:** 2026-10-15 | **Revision:** `8a1e999`

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
| **Optimize** | Operator-level: staged accuracy-gated tuning campaign (code-level → scheduling → instruction-level), metric→action routing; AscendC tiling design per operator class (reduction/broadcast/elewise/matmul/conversion) | `ascend-operator-tuning`, `ascendc-tiling-design` | PyPTO frontend/swimlane/incore playbooks; fullload-vs-split, four-element tiling template, chip-generation (DAV_2201/3510) routing | awesome-ascend-skills pypto-op-perf-tune + ascendc-tiling-design @ 75ec89c | 🟢 |
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
| **Observe / Profile** | — | — | — | — | 🔴 (no storage profiling method in any local repo; fio/iostat/page-cache zero hits; logical-vs-physical I/O caveat only in mindie-agent docs) |
| **Diagnose** | Slow startup (graph capture + weight loading dominating): distinguish compile warmup from I/O | `ascend-graph-mode` (mentions slow startup), `ascend-model-adaptation` (dummy→real sequence) | ACLGraph startup cost is compile + capture, not I/O bound; FP8 weight loading pitfalls | MindIE-SD; vllm-ascend | 🟡 (incidental coverage only) |
| **Diagnose** | Load-path judgment (source/format/placement/duplication) and startup fixed-overhead accounting (bytes+counts lines, hit-criteria warmup validation) | `weight-loading-optimization` | Four-dimension load-path judgment; fixed-vs-per-request split; 7 accounting errors | MindIE-SD host-opt + host-overhead-account @ 8481d39b | 🟢 |
| **Configure** | Checkpoint format for fast loading (FP8 paired tensors, shard loading order) | `ascend-model-adaptation` | FP8 paired-tensor loading pattern | vllm-ascend @ bc8115ed | 🟡 |
| **Optimize** | Serving-level load path: device-side weight prefetch (switch-gated, 50%-of-theoretical start), CPU-tier offload ordering by change surface, duplication removal | `weight-loading-optimization` | `npu_prefetch` decision rules; OOM fallback tiers; construction vs inference OOM | cannbot model-infer-prefetch + diffusers memory-optimization @ 75ec89c; MindIE-SD resource-fallback-tiers @ 8481d39b | 🟢 (serving level) |
| **Optimize** | Disk-level I/O mechanics: mmap/pinned staging, shard-parallel loading, fio/page-cache benchmarking, checkpoint save/restore I/O, disk-tier offload | — | Negative finding: 8 repos scanned (queries listed in the skill), zero material; big-3 repos are skill-doc-only sparse checkouts | see `skills/weight-loading-optimization/SKILL.md` negative-findings section | 🔴 (needs upstream vLLM/fastsafetensors/lmcache sources) |
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
| **Optimize** | CPU affinity check-then-pin (accelerator-local NUMA node, cross-round consistency), thread-count env discipline, device-busy preflight with graceful-to-force PID escalation, structured ERROR/WARNING env checks, container flags and ulimits | `os-process-placement` | OMP_NUM_THREADS/PROC_BIND per scenario; `CPU_AFFINITY_CONF`; npu-smi usages preflight; `ASCEND_HOME_PATH` named-var correctness | TRT-LLM perf-host-optimization @ f33e0e5f; awesome-ascend-skills @ 75ec89c | 🟢 (cgroups/cpuset/isolcpus and scheduler priorities explicitly out of scope — no source evidence) |
| **Verify** | Card health check sequence: npu-smi Health, 4-step smoke verify, proc-mem occupancy | `ascend-device-health` | Health=OK ≠ card usable; SIGKILL residual state only detectable via smoke run | MindIE-SD @ 8481d39b | 🟢 |

**Second-level targets:** NPU driver, CANN runtime, container/process isolation

---

## Gap Summary

| Gap | Row | Priority | Next action |
|-----|-----|----------|-------------|
| Storage profiling (Observe) and disk-level I/O mechanics (mmap/pinned, fio/page-cache, checkpoint save/restore I/O, disk-tier offload) | SSD / Observe, Optimize (disk level) | Low | Negative finding recorded 2026-10-15: 8 local repos scanned, zero material; the three big repos are skill-doc-only sparse checkouts. Needs upstream sources (vLLM safetensors loader, fastsafetensors, lmcache) |
| OS/System cgroups/cpuset/isolcpus and scheduler priorities | OS / Optimize | Low | No source evidence in any local repo; `os-process-placement` covers placement/affinity/threads/env/container-flags; the remainder needs external OS documentation |

**Resolved gaps since initial matrix (all high/medium priority cleared):**
- ~~Host overhead profiling + optimization skills~~ → `host-overhead-profiling`, `host-overhead-optimization` ✅
- ~~Parallelism strategy skill~~ → `ascend-distributed-parallel` ✅
- ~~NPU optimization casebook~~ → `ascend-optimization-casebook` ✅
- ~~SIGKILL / card-health recovery~~ → `ascend-device-health` ✅
- ~~Same-window A/B acceptance skill~~ → covered in `ascend-optimization-casebook` (per-case evidence schema) ✅
- ~~Kernel writing skill (Triton-Ascend / AscendC TileIR / CuTe)~~ → `kernel-writing-guide` ✅
- ~~Host memory lifecycle (pinned memory / GIL / H2D async)~~ → `host-memory-management` ✅
- ~~Operator-level tuning (pypto-op-perf-tune, AscendC tiling)~~ → `ascend-operator-tuning`, `ascendc-tiling-design` ✅
- ~~Serving-level weight loading / startup overhead~~ → `weight-loading-optimization` ✅ (disk-level mechanics remain open, see gap table)
- ~~OS/System Optimize (CPU affinity, NUMA process placement, env discipline)~~ → `os-process-placement` ✅ (cgroups/scheduler priorities remain open, see gap table)

---

## PR coverage table

Use this table in the PR description for `opus/meta-skill-bootstrap`.

| Search Space | Observe / Profile | Diagnose | Configure | Optimize | Verify |
|---|---|---|---|---|---|
| **Accelerator** | 🟢 `ascend-performance-analysis`, `ascend-operator-profiling`, `dense-performance-evidence` | 🟢 `ascend-performance-analysis`, `performance-ceiling-analysis` | 🟢 `ascend-graph-mode`, `ascend-model-adaptation`, `ascend-distributed-parallel` | 🟢 `ascend-optimization-casebook`, `optimization-search-policy`, `kernel-writing-guide`, `ascend-operator-tuning`, `ascendc-tiling-design` | 🟢 `dense-performance-evidence`, `rsi-evidence`, `claim-contract` |
| **RAM / Host** | 🟢 `host-overhead-profiling` | 🟢 `host-overhead-profiling` | 🟢 `host-memory-management` | 🟢 `host-overhead-optimization` | 🟡 (inherited from Accelerator row) |
| **SSD / Storage** | 🔴 | 🟢 `weight-loading-optimization` (load-path accounting) + 🟡 `ascend-graph-mode` (startup incidental) | 🟡 `ascend-model-adaptation` (FP8 loading) | 🟢 `weight-loading-optimization` (serving level) · 🔴 disk-level mechanics (negative finding recorded) | 🔴 |
| **Communication** | 🟢 `ascend-performance-analysis` | 🟢 `ascend-performance-analysis`, `ascend-distributed-parallel` | 🟢 `ascend-distributed-parallel` | 🟢 `ascend-distributed-parallel` | 🟢 `ascend-distributed-parallel`, `ascend-performance-analysis` |
| **OS / System** | 🟢 `ascend-operator-profiling` (DVFS), `ascend-device-health` (proc-mem) | 🟢 `ascend-device-health`, `ascend-performance-analysis` | 🟢 `ascend-device-health` | 🟢 `os-process-placement` (affinity/threads/env; cgroups out of scope) | 🟢 `ascend-device-health` |

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
- Operator tuning campaign (accuracy gate, staged escalation, metric contract, routing tables)
- AscendC tiling design (four-element template, fullload-vs-split, per-class routing, chip-generation tables)
- Weight load path and startup overhead (fixed-overhead accounting, sharded verification, prefetch rules, offload tiers)
- OS-level process placement (affinity pinning, thread env discipline, device-busy preflight, env checks)

**Skill count:** 23 candidate skills. All `status: candidate`. None promoted.
