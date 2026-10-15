---
name: weight-loading-optimization
description: >
  Optimize the serving-level weight load path from storage to device: judge
  the load path on four dimensions (source, format, placement, duplication),
  account startup fixed overhead separately from per-request cost with
  bytes-and-counts lines and hit-criteria warmup validation, verify sharded
  weight downloads against the safetensors index, apply device-side weight
  prefetch only on profiled memory-bound hot spots, and order CPU-tier
  offload candidates by change surface. Use when startup/loading latency or
  load-path I/O dominates. Does not cover disk-level mechanics (mmap, pinned
  staging, fio/page-cache benchmarking), checkpoint save/restore I/O, or
  disk-tier offload — no source evidence for those.
tags: [storage, weight-loading, startup, prefetch, offload, serving, npu]
status: candidate
---

# Weight Loading and Startup Fixed-Overhead Optimization

Status: candidate. Source-grounded from MindIE-SD `host-opt` +
`env-install/weights-prep` + `dit-perf-opt/resource-fallback-tiers` @ 8481d39b
(Apache-2.0), and awesome-ascend-skills `model-infer-prefetch` +
`diffusers-ascend` memory-optimization @ 75ec89c (license UNKNOWN). Serves the
storage row at the serving level; disk-level I/O mechanics need a second
source (upstream vLLM safetensors loader, fastsafetensors, lmcache).

## SearchSpace mapping

SSD / Storage row → Diagnose + Optimize cells.

Second-level targets: model weight sharding/loading, startup latency,
weight prefetch, CPU-tier offload.

Related skills: `host-overhead-profiling` (request-time host overhead — this
skill covers the once-per-process side), `ascend-device-health` (card state
before launch), `os-process-placement` (process placement before measuring).

---

## 1. Judge the load path on four dimensions

Before optimizing, locate the problem along these axes (MindIE-SD host-opt):

| Dimension | Options | Question |
|---|---|---|
| Source | network disk / local disk / memory cache | Where do the bytes come from? |
| Format | per-tensor reads vs single large file | Read granularity |
| Placement | host-first then to device vs direct device read | Extra hop or not |
| Duplication | multi-process / multi-instance each loading a copy | Redundant reads |

Optimization levers follow directly: fewer reads, fewer copies, parallelize
what remains, reuse what is already resident. Container/image cold-cache
first-read is a cost line of its own — record it separately.

## 2. Account startup fixed overhead separately

The accounting discipline (MindIE-SD host-overhead-account):

- **Fixed vs per-request split**: loading and warmup happen once per process;
  their denominator is startup wall-clock, never amortized into per-request
  e2e. Delivery/transfer work is per-request and accounted separately.
- **No timing point, no account**: if the profiler cannot see a host-side
  segment, add explicit timing points — an unmeasured segment does not exist
  in the ledger.
- **Every line carries**: start/end measurement points + byte count + access
  count + evidence. Bytes and counts distinguish "volume problem" (too many
  bytes) from "path problem" (too many round trips).
- **Warmup is validated by hit criteria, not total time**: cache-entry count,
  graph-hit counters, or expected log lines. Mismatched warmup shapes silently
  warm a different set of cache entries while total time looks fine.
- **Seven accounting errors to avoid** (from source): denominator drift;
  double counting; timing points inside async branches; comparing absolute
  values across windows (same-window A/B only); mixing fixed into per-request;
  trusting unmeasured segments; skipping the warmup hit check.
- Benchmark hygiene (OmniRSI): report preparation/startup/readiness, warmups,
  and measured work as separate numbers.

## 3. Weight acquisition and verification

Sharded downloads (modelscope/HF-style) — operational rules:

- Parallelize with bounded workers (`--max-workers 16` style), select only
  needed shards (`--include '{subset}/**'`).
- `.incomplete` marker files mean in-flight shards, not corruption; wait or
  resume, do not re-download blindly.
- Verify completeness structurally: shard numbering has no gaps **and** the
  set matches `*.safetensors.index.json` `weight_map` keys. On mismatch,
  cross-check weight_map and fetch only the missing shards.
- Measure aggregate rate by `du -sh` differencing on a stable segment (not
  ramp-up); expect single-digit MB/s on slow networks — hours for ~134 GiB is
  normal, not a defect. Triage downloader logs as WARNING (transient) vs
  ERROR (real failure) to avoid false alarms.
- To test a load path without real bytes, generate structure-identical fake
  weights from `model_index.json`/configs (metadata-only download).

## 4. Device-side weight prefetch

Decision rules (cannbot `model-infer-prefetch`, `torch_npu.npu_prefetch`;
methodology transfers, the API is NPU-specific):

- **Only on profiled memory-bound hot spots** — never speculative.
- The **dependency window** (work overlapping the prefetch) must be
  non-memory-bound: use the window after a large MatMul completes, or any
  compute/communication segment whose output is available.
- `max_size` starts at **50% of the theoretical size**; after verified wins,
  step up 0.7 → 1.0 of theoretical. Never start at 1.0.
- Every prefetch sits behind an `enable_prefetch` switch, **default False**.
- Verify a single operator first; then extend position by position. Tuning
  order: position ↔ size, iteratively.
- **When to give up**: no safe dependency window → do not force prefetch.
  If the window regresses after enabling: shrink size or move position; if
  neither works, disable and record.
- Adjacent pattern (TRT-LLM): move periodic background bookkeeping and
  weight-movement off the critical stream; the same shape carries to
  KV-cache defrag/transfer and weight streaming.

## 5. CPU-tier offload ordering

When weights/activations do not fit device memory:

- **Order candidates by change surface** (smallest first): batch size →
  resolution → CFG/feature flags → CPU offload / layered offload →
  tensor parallel degree → activation checkpointing.
- **Construction-phase OOM is not an inference-phase OOM**: for construction
  OOM use meta-device construction then `to_empty` fill; tier-downs
  (resolution, CFG) do not help construction memory.
- **Offload benefit ceiling = the un-overlapped transfer share**: if transfer
  is already hidden behind compute, offload buys nothing; re-estimate after
  adopting each tier.
- Diffusers-on-NPU playbook by free VRAM (≥ 40 GB / 32–40 / < 32 / < 16):
  `enable_model_cpu_offload(device="npu:0")` (do **not** `pipe.to("npu")`
  first; requires `accelerate`); gains bounded by PCIe bandwidth;
  sequential per-layer offload only as the last resort.
- KV-cache host offload (TRT-LLM pattern): fold side caches into the unified
  paged KV manager to inherit host/secondary offload and block reuse;
  `host_cache_size` sizing; host-offloaded blocks are indexed by physical
  memory-pool index, not logical block ID; null Python views before the C++
  free.
- NPU serving shallow knob: `--swap-space 8` (CPU swap when NPU memory is
  full).

---

## Negative findings (searched, not found — keep these gaps open)

Scanned all 8 local repos (TensorRT-LLM, vllm-ascend, MindIE-SD, akg,
mindie-agent, awesome-ascend-skills, LLM-Viewer, OmniRSI; queries included
safetensors, mmap, memory_map, pinned, O_DIRECT, posix_fadvise, readahead,
page cache, fio, iostat, iotop, nvme, startup time, load latency, async
checkpoint save):

- No mmap / pinned staging / shard-parallel loading mechanics anywhere.
- No checkpoint save/restore **I/O optimization** (only format-conversion
  guides).
- No disk/NVMe-tier offload (no weight-spill-to-disk, no KV-to-file cache).
- No fio/iostat/page-cache storage benchmarking methodology.
- Caveat: the three big repos are skill-doc-only sparse checkouts (no runtime
  implementations present), so absence here is not absence upstream.

Measurement caveat when you do get to disk-level work (mindie-agent): disk
read/write counters include cache and pipes — they are **logical I/O, not
physical disk writes**; without dropping the OS file cache, a run is not a
cold-start physical test.

## Known limitations

- MindIE-SD host-opt numbers (134 GiB / hours / single-digit MB/s) are
  author-observed on their network, not benchmarks.
- The prefetch escalation ladder (50% → 0.7 → 1.0) is source guidance, not
  a measured optimum.
- `model-infer-prefetch` API details (`torch_npu.npu_prefetch` signature,
  dependency wiring) are NPU-specific; consult the CANN docs of the target
  version.

## Verification checklist (before claiming a load-path win)

- [ ] Fixed overhead accounted separately from per-request cost
- [ ] Every accounting line has timing points + bytes + counts + evidence
- [ ] Warmup validated by hit criteria (cache entries / graph hits / logs)
- [ ] Before/after compared same-window, same affinity, same device state
- [ ] Prefetch changes verified single-op first, switch defaults recorded
- [ ] Offload tier wins re-estimated against un-overlapped transfer share
