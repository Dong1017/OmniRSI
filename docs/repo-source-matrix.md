# OmniRSI Repo Source Matrix

**Branch:** `opus/meta-skill-bootstrap` | **Date:** 2026-10-15 | **Revision:** `1a9d095`

Companion view to `docs/searchspace-matrix.md`:

- **SearchSpace Matrix** — what optimization intelligence we currently have (lifecycle coverage).
- **Repo Source Matrix** (this file) — where that intelligence comes from, and what remains unread.

Status key (per cell): `mined` (source intelligence extracted into a candidate skill/case — **not** experimentally validated) · `located-not-mined` (relevant material identified, not yet extracted) · `unknown` (not investigated at this revision) · `out-of-scope` (repo does not target that layer).

Reading rules:

- Sparse / skill-doc-only checkouts carry curated methodology, not implementation
  code. **Not present in a sparse checkout does NOT mean absent upstream.**
- A cloned repo is not necessarily a mined repo.
- Negative findings are scoped to the paths and revisions actually inspected
  (see "Negative findings" below).
- Upstream benchmark numbers are author-reported, never local validation.

---

## 1 · Systems sources

| Repo (upstream @ revision) | Checkout coverage | Accelerator | RAM / Host | SSD / Storage | Communication | OS / System |
|---|---|---|---|---|---|---|
| NVIDIA/TensorRT-LLM @ f33e0e5fd | skill-doc-only (`.claude/skills/`) | mined — evidence taxonomy, kernel writing, casebook, roofline refs → `dense-performance-evidence`, `kernel-writing-guide`, `inference-optimization-casebook`, `performance-ceiling-analysis` | mined — host overhead verdict + line-profiling loop → `host-overhead-profiling`, `host-overhead-optimization` | mined — weight-movement/overlap + KV host-cache patterns → `weight-loading-optimization`; disk-level: negative (see below) | mined — critical-stream overlap scheduling pattern → `weight-loading-optimization` | mined — CPU affinity rule + env preflight → `os-process-placement` |
| Ascend/MindIE-SD @ 8481d39 | skill-doc-only (`.agents/skills/`) | mined — 5-layer analysis, bottleneck casebook → `ascend-performance-analysis`, `ascend-optimization-casebook` | mined — host-opt async/pattern sections → `weight-loading-optimization` | mined — load-path judgment, fixed-overhead accounting, offload tiers, weights-prep → `weight-loading-optimization` | mined — USP strategy, masking formula, topology rules → `ascend-distributed-parallel` | mined — device health, card-group contracts, measurement discipline → `ascend-device-health` |
| ascend-ai-coding/awesome-ascend-skills @ 75ec89c | full (`skills/`, `external/cannbot/`) | mined — graph mode, operator profiling/tuning, tiling design → `ascend-graph-mode`, `ascend-operator-profiling`, `ascend-operator-tuning`, `ascendc-tiling-design` | located-not-mined — `external/cannbot/model/model-infer-kvcache` identified, not extracted | mined — device-side prefetch + diffusers CPU-offload playbook → `weight-loading-optimization`; disk-level: negative | located-not-mined — hccl-test / torch-npu-comm-test read for contention preflight only | mined — env checks, device-busy preflight, thread/env tables → `os-process-placement` |
| vllm-project/vllm-ascend @ bc8115ed | skill-doc-only (`.agents/`) | mined — FP8/KV-replication/multimodal adaptation → `ascend-model-adaptation` | unknown — runtime host logic not in sparse checkout | mined — FP8 paired-tensor loading pattern → `ascend-model-adaptation` (Configure cell); runtime loader not in checkout | unknown | mined — Dockerfile thread/env ship defaults → `os-process-placement` |
| mindspore-ai/akg @ 5aa15f3f | full | located-not-mined — Ascend kernel compiler; mapped in `docs/ascend-npu-repo-map.md`, not extracted | unknown | negative — scanned 2026-10-15, no LLM storage material | unknown | negative — OS scan: identifier false-positives only |
| mindie-agent/mindie-agent @ 6a81b41 | full | out-of-scope — agent tooling, not serving performance | out-of-scope | mined (narrow) — logical-vs-physical I/O + cache-flush measurement caveat → `weight-loading-optimization` | out-of-scope | out-of-scope |
| hahnyuan/LLM-Viewer @ 1893e4b | full | mined — roofline model + hardware params → `performance-ceiling-analysis` (L1) | out-of-scope | out-of-scope | out-of-scope | out-of-scope |
| ebugger/Empirical-Roofline-Toolkit @ 0771fa9 | full | mined — empirical roofline method → `performance-ceiling-analysis` (L2) | out-of-scope | out-of-scope | out-of-scope | out-of-scope |
| foundation-model-stack/fastsafetensors @ 219e256 | full (shallow pin, wave 5) | out-of-scope — approved scope was model-loading I/O only | mined (partial) — pinned bounce-buffer staging, reader NUMA binding → `weight-loading-optimization` §7 | mined — loading I/O mechanics (no-mmap design, bounce reader, GDS, readiness overlap, partial reads, static planner) + measurement protocol → `weight-loading-optimization` §7; checkpoint save (`saver.py`) located-not-mined (out of approved scope) | located-not-mined — TP sharding as broadcast/scatter with coalesced runs extracted in evidence, not yet distilled | mined (narrow) — reader worker NUMA binding + O_DIRECT per-filesystem policy → `weight-loading-optimization` §7 |
| LMCache/LMCache @ c3b875a | full (shallow pin, wave 5) | out-of-scope — approved scope was CPU/local-storage KV management | mined — L1 DRAM tier, admission dial, watermark eviction, multi-L1 overflow → `kv-tier-management` | mined — L2 local adapters (fs/nixl/raw_block/dax), serde capacity lever, persist/recover → `kv-tier-management` | out-of-scope — remote adapters noted only | mined (narrow) — MP process/thread concurrency model (isolation, client-affinity pools, TTL locks) → `kv-tier-management` |
| ggml-org/llama.cpp @ 3e390af | full (shallow pin, wave 5) | out-of-scope — approved scope was loading modes and host placement | mined — mmap/mlock residency, lazy tensors, host weight placement (`--cpu-moe` family), KV host offload → `weight-loading-optimization` §6 | mined — loading modes (mmap/mlock/dio/lazy), page-cache interactions, fit-probe bypass → `weight-loading-optimization` §6 | out-of-scope | mined — NUMA modes (distribute/isolate/numactl), gen-vs-batch thread/affinity splits, numa_balancing rule → `os-process-placement` |

Home-repo note: ThinkFlowLab/OmniRSI @ 4c8e128 is the scaffold this branch lives
in (evidence/claim discipline, benchmark hygiene, RunSpec) — not an external
mining source. The local sibling clone at `codex/skill-eval-bootstrap`
(82218b4) is a **different fork lineage**; see §2.

### Negative findings (scoped to inspected paths and revisions)

- **Disk-level storage I/O mechanics** (mmap/pinned staging buffers,
  shard-parallel loading, fio/iostat/page-cache benchmarking, checkpoint
  save/restore I/O, disk-tier offload): zero material in the 8 first-layer
  repos listed above, as visible in their checkout coverage (TRT-LLM
  `.claude/`, MindIE-SD `.agents/`, vllm-ascend `.agents/` are skill-doc-only
  sparse checkouts). Queries used: safetensors, mmap, memory_map, pinned,
  O_DIRECT, posix_fadvise, readahead, page cache, fio, iostat, nvme, startup
  time, load latency, async save. Recorded 2026-10-15; also embedded in
  `skills/weight-loading-optimization/SKILL.md`.
- **cgroups / cpuset / isolcpus / nice / ionice / SCHED_FIFO**: zero hits in
  the same 8 repos and scope. `os-process-placement` covers placement,
  affinity, thread env, and container flags/ulimits only.
- Wave evidence files (grep logs, per-theme verdicts, mechanism extractions)
  are session-local under the research tmp directory and not committed to
  this branch.
- Wave-5 approved mining (2026-10-15) closed part of the first negative
  finding: loading modes and mmap/mlock placement (llama.cpp), pinned-staging
  parallel-read mechanics and a load-path measurement protocol
  (fastsafetensors), runtime KV-to-disk tiering (LMCache). Still open after
  wave 5: checkpoint save/restore I/O (fastsafetensors `saver.py` located,
  out of approved scope) and standalone block-device profiling tooling
  (fio-class). Approval is scoped per `repo × subsystem × SearchSpace gap` —
  wave-5 rows above reflect only the approved scopes.

---

## 2 · Methodology and corpus references

Role key: history learning · corpus · distillation · evolution · search ·
evaluation · reference. Methodology repos are **not** scored against the
SSD/OS systems columns.

| Reference (upstream @ revision) | Role | Used for | Consumed by |
|---|---|---|---|
| LearningToCommit/LearningToCommit @ 7e5a1d7 | history learning | temporal-split method (train/test cutoff discipline) | `repository-evolution-analysis` |
| yetone/magpie @ 77800032 | corpus | production local-LLM-gateway (Go, MIT) evolution history: 3428 commits; commit-size p50, repair-time medians, subsystem coupling | `repository-evolution-analysis` |
| VectorSpaceLab/AREX-Skill @ ac3fe1af | distillation | repo→skill distillation model, artifact lifecycle, refresh/provenance policy | `npu-skill-factory` |
| microsoft/SkillOpt @ 343db22 | evolution | bounded-edit skill evolution | `npu-skill-factory`, `skill-evolution-playbook`, `dense-performance-evidence`, `optimization-search-policy` |
| Qwen-Applications/Trace2Skill @ 3d0b52a | evolution | Trace2Skill pool consolidation | `skill-evolution-playbook` |
| JiusiServe/vllm-evolve @ 43cc82c | search | scheduling-algorithm evolution, search policy | `optimization-search-policy`, `dense-performance-evidence`, `inference-benchmark-design` |
| Dong1017/OmniRSI @ `codex/skill-eval-bootstrap` (82218b4) | evaluation | skill_distill / skill_eval harness, no_skill/human/distilled contract, Tier A pilot | `npu-skill-factory` (evaluation contract) |
| ThinkFlowLab/OmniRSI PR #9 @ 4ad551a9 | reference | `skill_library/` layout + knowledge records | `npu-skill-factory` (bundle layout) |
| arXiv 2609.24972 (RRSI) | search | constraint propagation | `optimization-search-policy` |
| arXiv 2608.01867 (CRISP) | search | critical-step classification | `optimization-search-policy` |

Notes:

- Same-name repos are not the same project: ThinkFlowLab/OmniRSI (upstream
  scaffold) ≠ Dong1017/OmniRSI (sibling evaluation fork). The two local magpie
  checkouts share HEAD 77800032.
- Wave-5 additions (fastsafetensors, LMCache, llama.cpp) are systems sources
  and appear in §1 only, under their approved scopes.
