# OmniRSI Reference Matrix

**Branch:** `opus/meta-skill-bootstrap` | **Date:** 2026-10-15 | **Revision:** `96a93d9`

One of exactly two primary matrices (decision D01, 2026-10-15). The file
keeps its `repo-source-matrix.md` name this round (D02); the official name
is **Reference Matrix**.

- **SearchSpace Matrix** (`docs/searchspace-matrix.md`) — resource ×
  optimization-stage coverage.
- **Reference Matrix** (this file) — sources, inspected scope, provenance,
  and extraction: repositories, papers, official documentation, PRs, and
  expert material; where the intelligence comes from and what remains
  unread. Skills, knowledge, and cases link the two (D03). Methodology
  sources live in §2, not in a third matrix.

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
| axboe/fio @ acb7a302 | pack-mediated (wave 6; HOWTO.rst targeted sections + `examples/aio-read.fio`) | out-of-scope | located-not-mined — cache-state and run-shape material partially feeds the ledger/run-classification rules | mined — workload matching, requested-vs-achieved depth, cache-state ledger, run-shape classification, output schema/units → `storage-io-measurement` | out-of-scope | out-of-scope |
| sysstat/sysstat @ efd0c445 | pack-mediated (wave 6; `man/iostat.in`, `man/pidstat.1`) | out-of-scope | mined (narrow) — per-process I/O/fault attribution caveats → `storage-io-measurement` §8 | mined — per-device attribution, saturation guards (%util/%iowait) → `storage-io-measurement` §6-7 | out-of-scope | mined (narrow) — CPU-side scheduling-pressure vs disk-hypothesis rules (%wait, context switches) → `storage-io-measurement` §8 |
| Linux cgroup v2 docs — kernel.org live 2026-10-15 (revision unidentified) + v6.12 `cgroup-v2.rst` (reproducible baseline) | pack-mediated (wave 6; core/CPU/Memory/IO/Cpuset interface sections) | out-of-scope | mined — memory pressure/events triage, high/low/max semantics → `cgroup-resource-controls` | mined (conditional) — io.max caps gated on confirmed block-backed traffic → `cgroup-resource-controls` | out-of-scope | mined — cpu.max/weight quota-vs-contention (cpu.stat vs .local), requested/effective cpuset, hierarchy/delegation preflight → `cgroup-resource-controls` |
| cann/oam-tools @ `8440935f` (!302 import; later master `f3cf4115` labeled [V2], never merged into !302 claims) | pack-mediated (wave 6; `cann-perf-breakdown`, `cann-multistream-optimize`, `cann-npu-perfanalysis` + selected references/scripts). Lineage: `jinyingqi/*` originals → !302 import — one lineage, not independent corroboration; original SHAs/byte equivalence unverified | mined — representative-step selection, code_ref source ownership, attribution-aware diagnosis, dependency DAG (data/state/event/collective_order), bounded overlap, correctness-first acceptance, metric separation, device-memory domain qualification → `cann-attribution-overlap` | out-of-scope (host RAM not established; device HBM/UB evidence must not fill the RAM row) | out-of-scope | mined — exposed-communication diagnosis, wait pollution, evidence-gated compute/comm overlap → `cann-attribution-overlap` | mined (bridge only) — host-event coverage of device bubbles, uncertainty-aware → `cann-attribution-overlap` |

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
  (fastsafetensors), runtime KV-to-disk tiering (LMCache).
- Wave-6 pack-mediated mining (2026-10-15) closed the rest: checkpoint
  save/restore semantics (`weight-loading-optimization` §8) and
  block-device measurement methodology (`storage-io-measurement`). Pack
  mediation: the gate owners supplied read-only research packs (fio,
  sysstat, kernel cgroup-v2 docs, cann/oam-tools); receiving a pack is not
  mining — extraction into the linked skills happened in this workspace.
  Pack facts are documentation-level; no benchmarks were run, no caches
  manipulated, no settings changed.
- Approval is scoped per `repo × subsystem × SearchSpace gap` — rows above
  reflect only the approved scopes.

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
