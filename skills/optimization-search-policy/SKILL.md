---
name: optimization-search-policy
description: >
  Policy for how an optimization agent should select, order, and stop
  optimization experiments. Covers candidate generation, exploration vs
  exploitation, diversity, negative experiment memory, budget allocation,
  and stopping criteria. Drawn from vllm-evolve (phase-locked optimization
  round), TRT-LLM casebook (prior-art matching), and SkillOpt (bounded edit
  + gate). Use when deciding what to try next and when to stop.
tags:
  - optimization
  - search
  - policy
  - experiment-design
status: candidate
---

# Optimization Search Policy

Status: candidate. Source-grounded from vllm-evolve @ 43cc82c,
TensorRT-LLM @ f33e0e5f, SkillOpt @ 343db229.

## Core problem

An optimization agent without a search policy tends to:
- Re-explore the same configuration families (familiar solution bias)
- Escalate experiment cost prematurely (running E2E benchmarks when profiling
  would answer the question)
- Fail to record negative results (causing the same dead ends to be re-entered)
- Commit to a direction before checking prior art

This skill addresses all four.

## Phase-locked optimization round

An effective optimization round has locked phases that cannot be reordered.
Source: vllm-evolve `ar` CLI + PreToolUse hook (enforces phase order at the
tool call level).

```
Phase 1 — Read context
  Inspect: profiling data, existing config, prior experiment records,
  casebook of known wins and failures.
  Do not skip this because you think you know what to try.

Phase 2 — Design a change
  Propose one candidate change. State the hypothesis explicitly:
  "If I change X, metric Y should improve because Z."
  State the evidence that motivates this hypothesis.

Phase 3 — Edit (or configure)
  Apply the change. Bounded: do not change the evaluation protocol,
  measurement setup, or baseline at the same time.

Phase 4 — Verify (correctness)
  Run correctness check before performance measurement.
  A change that breaks correctness is discarded immediately.
  Do not spend benchmarking budget on a broken candidate.

Phase 5 — Evaluate (performance)
  Run the benchmark at the appropriate evidence level
  (see dense-performance-evidence skill).

Phase 6 — Compare (verdict)
  Verdict: better / worse / inconclusive.
  Record the result unconditionally, including negative results.

Phase 7 — Keep or discard
  On "better": integrate and record the win.
  On "worse" or "inconclusive": discard the change, record the failure.
```

After Phase 7, return to Phase 1 with the updated context.

## Candidate generation

Before generating a new candidate, consult:
1. **Prior art (casebook)**: does a known successful optimization apply to the
   current bottleneck class, model, and hardware? If yes, adapt it rather than
   inventing a new approach. The casebook search is recall-complete via
   free-text grep over the full case file bodies — do not limit by structured
   fields before searching.
   Source: TRT-LLM perf-optimization-casebook @ f33e0e5f.

2. **Negative record**: has this specific candidate class been tried and failed
   in this configuration? If yes, the burden of justification is higher —
   something about the conditions must have changed to try again.

3. **Stage fraction**: is the target stage actually a significant fraction of
   E2E time? A 20% speedup on a stage that is 5% of E2E yields 1% overall.
   Compute the Amdahl contribution before committing to expensive work.
   Source: performance-ceiling-analysis skill.

## Exploration vs exploitation

**Exploit first, explore when stuck.** If prior art exists for the current
bottleneck, try it before generating novel candidates. Novel candidates cost
more to evaluate and have lower prior probability of success.

**Diversify the bottleneck, not the solution.** When the current optimization
family is exhausted (e.g., all memory-bandwidth reductions explored), switch
to investigating the next largest bottleneck stage, not to trying more
variants of the same solution. The classification step (which stage to target)
is where diversity matters most.

**Exploration budget:** reserve a fraction of the experiment budget for
low-cost exploratory runs (single-seed, Tier A, quick profiling) before
committing to full benchmarks. vllm-evolve's architecture uses a "simulation
backend" (not yet implemented at time of writing) for this purpose.
Source: vllm-evolve README @ 43cc82c.

## Negative experiment memory

Every experiment that produces a "worse" or "inconclusive" verdict must be
recorded with:
- The hypothesis that was tested.
- The specific metric that did not improve.
- The hardware/software/workload context.
- The evidence level at which the test was run.
- Whether the failure was due to correctness, no improvement, or high variance.

Do not discard negative results. An optimization that "might work with more
tuning" with no evidence is not a reason to defer recording it as negative.
Source: SkillOpt blog/gating-reflection-safe-updates @ 343db229.

**Negative result → search constraint:** when a hypothesis fails, derive a
constraint from it. Example: "KV cache compression at this sequence length did
not reduce memory traffic (measured: mte2_ratio unchanged at concurrency=4)."
That constraint narrows the remaining search space.

## Cost-aware candidate selection

Not all experiments cost the same. Order candidates by cost-adjusted expected
value:

| Experiment type | Relative cost | When to use |
|-----------------|--------------|-------------|
| Static analysis / config change | Low | To verify correctness, not performance |
| Single-kernel microbench | Low | When the hypothesis is about a specific isolated op |
| Component profiler run | Medium | To identify or confirm a bottleneck class |
| Scenario benchmark (concurrency=1) | Medium | To confirm a win in the simplest case |
| Multi-regime benchmark | High | After a scenario benchmark win, before committing |
| Full production replay | Very high | Only after multi-regime benchmark |

Escalate upward only when a lower cost level returns a positive result that
warrants confirmation at the next level.

## Stopping criteria

Stop local tuning and switch strategy when any of the following is true:

1. **Ridge-point reached:** the dominant bottleneck's SOL% or AI Core
   utilization exceeds 80%. Further optimization within that bottleneck class
   will yield diminishing returns.

2. **Amdahl ceiling reached:** the Amdahl-estimated maximum E2E speedup from
   all remaining stages together is below the target. Local tuning cannot reach
   the goal; the target itself or the approach must change.

3. **Repeated inconclusive results:** if the last N experiments on the same
   hypothesis all return inconclusive due to measurement variance, either the
   effect is below the noise floor or the measurement protocol is wrong. Fix
   the protocol or move to a different hypothesis. N = 3 is a reasonable
   default.

4. **Negative record saturated:** every plausible candidate from prior art and
   novel generation has been tried and recorded as failed. This is a signal to
   escalate the abstraction level (e.g., change parallelism strategy rather
   than tuning an individual op).

## Budget allocation

Heuristic allocation across phases of a campaign:

| Phase | Budget fraction |
|-------|----------------|
| Context reading + profiling (L2 evidence) | 30% |
| Exploration (low-cost candidates, single-regime) | 30% |
| Exploitation (best candidate, multi-regime benchmark) | 30% |
| Recording and analysis | 10% |

Adjust for the specific campaign. If prior art is strong, reduce exploration
and increase exploitation. If the bottleneck is unclear, increase the profiling
fraction before generating any candidates.
