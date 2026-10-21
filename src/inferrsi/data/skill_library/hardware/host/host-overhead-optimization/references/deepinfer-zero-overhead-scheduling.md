# Case: zero-overhead scheduling in DeepInfer

A worked, source-level instance of hiding exposed host prep behind device execution. It is the same trade-off this skill's loop optimizes, applied one layer down: instead of making the Python cheaper, it moves the Python so it runs while the GPU is busy.

Source: `deepseek-ai/DeepInfer` @ `8b16149741a4935e1dec8cea27853f273b374c5b` (Apache-2.0), read statically. Nothing was built or run. **F** is code behaviour at this pin; **P** is proposed.

## What the bubble is

A decode step alternates device work with CPU work, and the CPU work must consume a device-to-host result from the step it post-processes. The naive order is:

```text
forward(N) -> sample(N) -> D2H -> post-process(N) -> schedule(N+1) -> forward(N+1)
```

The D2H wait plus post-process plus schedule sit between two device launches, so the GPU idles for their duration. That is exposed host prep: the same phenomenon this skill measures before deciding whether to optimize or redirect.

## The reorder

```text
forward(N) -> post-process(N-1) -> sample(N) -> [defer post-process(N)]
```

`F`: `deepinfer/worker/worker.py:183-188`. Prompt steps never defer; only decode steps, and only when `DEEPINFER_ZERO_OVERHEAD_SCHEDULING` is on:

```python
if is_prompt or not envs.DEEPINFER_ZERO_OVERHEAD_SCHEDULING:
    request_outputs = np.concatenate([request_outputs, self._post_process_step(curr_step_state)])
    self.prev_step_state = None
else:
    self.prev_step_state = curr_step_state
```

The next step begins by post-processing the retained state (`worker.py:143-144`), so the CPU work overlaps `forward(N)`.

## What becomes stale, and the repair

This is the part worth copying. Three quantities the scheduler reads are now one step behind:

| Quantity | Why it is stale | Repair |
|---|---|---|
| Per-request query length | Under adaptive speculation the row count depends on the *previous* step's acceptance | Recomputed on device from previous-step confidence rates |
| Per-request key length | Rejected drafts mean the previous step accepted fewer tokens than it requested | Recomputed on device from previous-step accepted counts |
| Input token ids | Rows continuing from the previous batch reuse the previous step's sampled ids | Scatter from the previous batch via a remap |

`F`: `worker.py:191-224`. The remap matches current batch slots to previous batch slots by stable sequence id; `-1` marks a fresh row. It short-circuits to the direct path when the batch order matches exactly.

```python
if envs.DEEPINFER_ENABLE_MTP_VARLEN:
    compute_seqlens_q_by_confidence_rates(total_q, remap[:n_seq], prev.sampler_output.confidence_rates_gpu, seqlens_q[:n_seq])
if self.nextn > 0:
    compute_seqlens_k_zos(...)
```

The remap construction itself is `F`: `csrc/engine/zos.cpp:142-160` (`prepare_zos_inputs`), with the fresh/continuing distinction at `:105`.

Note the dependency this creates: the deferred path is only correct because the *next* step repairs what the deferred post-process would have produced. If your engine cannot enumerate the stale quantities, deferring the post-process silently produces wrong metadata rather than a crash.

## The three conditions that make it work

1. **The scheduler must not touch the device.** `Scheduler::_schedule` releases the GIL and returns plain metadata. If scheduling needed a device kernel, deferring it would not overlap anything.
2. **The DP vote must run on the host.** The replicated step-type vote is a `MIN` all-reduce over the data-parallel `ProcessGroup` (`F`: `csrc/engine/scheduler.hpp:316,318,332`, with the group passed in at `:83`). Placing it on a device collective would serialize against the work being overlapped. Which backend that group uses is a deployment decision, not something the call site fixes.
3. **Nothing may synchronize in the launch scope.** The forward and sample launches are wrapped in `forbid_cuda_sync()`, which raises on any torch-visible CPU-GPU sync in scope (`F`: `deepinfer/worker/model_runner.py:321,357`; definition at `deepinfer/utils.py:119-131`). Event-based waits are explicitly exempt, so the deliberate D2H points are unaffected.

## The cost

- One extra step of retained metadata, request outputs, state slots and KV blocks.
- A device-side repair path that must be maintained alongside the scheduler.
- Output latency: callers see results one step late, so the entrypoint needs a drain that flushes the last in-flight step.
- The DP collectives must be on a host-capable backend, which is a deployment constraint, not a code change.

## Relevance to this skill

Use this case to interpret the exposed-prep metrics rather than to replace the loop. A large exposed-prep ratio with a large inter-step gap is the signal; the options are to make the prep cheaper (this skill's line-profiling loop) or to move it (this case). The second is cheaper to implement when the prep is Python-bound and the device has slack, which is exactly the condition a profile should show first.

Do not copy the repair list. It is specific to engines where speculation width and accepted length feed back into the next step's row layout.

## Unresolved

- Preemption of an in-flight decode sequence may free a state slot or KV block that a running kernel is still writing. No guard was found at this pin. Treat the pipeline as unsafe under preemption until that is resolved.
- The default depends on how the engine object was built. `DEEPINFER_ZERO_OVERHEAD_SCHEDULING` is `EnvVar.bool()`, which defaults false (`F`: `deepinfer/environ.py:70-71`). `EngineArgs.__post_init__` sets it with `setdefault` from `engine_type.default_zero_overhead_scheduling`, which is true for `NORMAL` and `COMPLETION` and false for `PROMPT` (`F`: `deepinfer/engine/arg_utils.py:52-55`, `engine_type.py:15-17`). A README claim of "enabled by default" and a false environment-variable default can both be true of different entrypoints: the former describes construction through `EngineArgs`, the latter describes a process that never constructs one.
