---
name: budget-speculative-verification
description: Decide how many draft tokens to verify per decode step when acceptance varies by request and position. Use when an engine already runs a draft head with multi-token speculation and verification cost grows with query rows. Separates draft cost from verify cost, builds a cumulative-acceptance rate vector, fits a cost model, and allocates a per-step row budget.
---

# Budget speculative verification

## Trigger and boundary

Use when all of the following hold:

- The engine performs multi-token speculative decoding with a draft head.
- Verification is a batched forward whose cost grows with the number of query rows.
- Acceptance probability varies materially by request and by draft position.
- A per-position acceptance signal exists, or can be trained and exported.

Do **not** use when:

- The draft head is a separate model whose forward cost is proportional to the number of drafts requested. Then the budget must also throttle drafting, and many implementations cannot. The procedure below budgets verification only.
- There is no per-step cost model at all, because the budget decision needs a cost-versus-rows estimate.

## Status and source

Status: **candidate**. Empirical validation: **not_run**.

Source: `deepseek-ai/DeepInfer` @ `8b16149741a4935e1dec8cea27853f273b374c5b` (Apache-2.0), read statically. Nothing was built or executed, and no gain was reproduced. Evidence labels follow the readset convention: **F** is a fact observed in the pinned source; **P** is a proposed procedure.

Read [source-notes.md](references/source-notes.md) for the evidence ledger and [validation.md](references/validation.md) for checks that have not been run.

## Required inputs

1. Draft width and how the engine represents a request's rows — typically one fresh row plus `k` verification rows.
2. A per-position acceptance estimate with known semantics: per-position independent, or already cumulative. This distinction decides step 2 and cannot be inferred from a field name.
3. A table of `(batch_rows, step_time)` for the verify forward, or a way to collect it at warmup.
4. A sampler cost estimate per batch, because the total step includes sampling.
5. Which step's acceptance values are available at schedule time. If the scheduler runs ahead of sampling, the current step's values do not exist yet.

## Procedure

1. **Separate the two costs.** Measure the draft forward and the verify forward independently. In the reference source the draft forward is always full width (`F`: `csrc/engine/mtp_varlen.hpp:155` builds `seqlens_q` from `1 + draft_tokens.size()` only on the non-budgeted path), and only verification is budgeted. If your draft cost scales with requests, the budget must cover both or it will not help.

2. **Normalize the acceptance signal to a cumulative rate.** Convert per-position independent probabilities into a non-increasing vector. This is the load-bearing step: a cumulative vector makes any grant of "verify one more draft" a *prefix* grant. Without it a budget could grant position 3 while withholding positions 1 and 2, which has no meaning because verification is sequential.

3. **Fit a cost model.** `time(rows) ≈ baseline(rows) + slope × rows + sampler(requests)`. Collect the curve at warmup under the graph sizes you will actually replay. Clamp the slope at zero so extrapolation cannot produce a negative or non-positive time (`F`: `mtp_varlen.hpp:226`, `std::max(0.0, _forward_time_slope.get())`).

4. **Choose the global row budget.** Maximize `expected_accepted_tokens / time(rows)` over the sorted candidate rows. Seed the expectation with one token per request, because every request always samples at least one (`F`: `mtp_varlen.hpp:231`, `cur_expected = static_cast<float>(num_seqs)`).

5. **Allocate per request.** Walk the cumulative rates in descending order and grant one extra row at a time until the global budget is spent. Break ties by request index so the result is deterministic (`F`: `mtp_varlen.hpp:204-214`).

6. **Enforce the floor.** Never allocate fewer rows than the number of requests; that would be slower than plain decode. Assert it rather than trusting the search (`F`: `mtp_varlen.hpp:260`, `TORCH_CHECK(best_bsz >= num_seqs)`).

7. **Preserve per-position independence when choosing which round's confidence to read.** Verification of draft position `i` must not depend on confidence at positions `i+1 … n`. A scheme that lets a later position's confidence decide whether an earlier draft is verified changes which tokens are accepted, so it changes the output distribution rather than only the cost. The reference states this requirement directly (`F`: `mtp_varlen.hpp:160`, translated from the source comment: "confidence for positions `i+1, …, nextn` must not affect whether `draft_token_i` is verified").

   The two-round protocol exists to satisfy that requirement under a pipelined scheduler, not merely to work around a missing value. It takes the global row budget from round N−2 and the per-request split from round N−1, tracked by per-request counters (`F`: `mtp_varlen.hpp:161-183`). If you substitute a different timing — an earlier round, a current-step estimate, or a same-step recompute — treat it as a new design and show that it preserves the same independence property, because a later position's confidence reaching an earlier decision is a correctness defect rather than a tuning choice.

8. **Propagate the allocation to the metadata.** The verify kernels read per-request query length and the per-position row map. Both must be built from the budget, not from the draft width.

## Expected outputs

- A per-step row budget and a per-request query length in `[1, draft_width + 1]`.
- A cumulative acceptance vector per request.
- Log fields: predicted acceptance, realized acceptance, and their error.

## Evidence to collect

Verification rows and accepted tokens measure different things, so report them separately rather than as one before/after pair:

- **Allocation invariants**: every grant is a prefix over draft positions, and no request receives fewer than one row.
- **Sampling correctness**, against an independently specified oracle: does the accepted-prefix decision match the rejection rule you intend, under explicitly stated assumptions about the draft and target distributions.
- **Cost and throughput**: verification rows per step, accepted tokens per step, total step time, and end-to-end throughput, all with the draft forward included.
- **Realized-versus-predicted acceptance error**, aggregated per step.

A lower accepted length is a legitimate outcome of budgeting, not a failure: the objective is expected accepted tokens per unit time, and verifying fewer drafts can lose accepted tokens while gaining more. Do not use unchanged accepted length as the acceptance criterion. Deterministic token comparison is a valid check only where the decoding path is deterministic and the sampling semantics justify it; under stochastic sampling, two runs that consume random draws in different orders do not have to agree token for token.

## Preconditions and caveats

- **The draft forward is unchanged.** Quoting a throughput gain without saying so overstates the effect: the reference saves verification rows only.
- **Two different temperatures, and neither implies the other.** The confidence-head temperature (`DEEPINFER_DFLASH_CONFIDENCE_HEAD_TEMPERATURE`) scales the confidence head's output; it calibrates the acceptance estimate (`F`: `deepinfer/model_executor/models/mtp/dflash.py:62-64,76`). A request's sampling temperature is a separate per-request parameter that is clamped at the low end with `top_p`/`top_k` reset (`F`: `deepinfer/sampling_params.py:76-79`). Do not infer sampling behaviour from confidence calibration, or the reverse.
- **Rejection is gated on draft logprobs, not on temperature.** The reject path runs when draft logprobs are supplied (`F`: `deepinfer/model_executor/layers/tilelang_ops/fused_sample.py:200`), and request temperature is clamped rather than allowed to reach zero. Together these support the reading that a request configured as greedy still goes through verification; this is a reading of the gating logic, not a measured end-to-end result.
- **Low acceptance still reserves state.** KV blocks are reserved at maximum width and trimmed later, so a request that accepts nothing still pins extra slots until the next scheduling pass.
- **Wide top-k pools scale badly.** Verification logprob pools scale as `rows × topk × positions`; a full-vocabulary top-k can dominate memory before it dominates time.
- **Cross-platform parity is unproven.** The reference has different rejection-comparison arithmetic on its two backends and no cross-platform equality assertion.

## Stop conditions

- Stop if the draft forward cost dominates and cannot be throttled; then the verification saving is a small share of the step.
- Stop if sampler cost dominates verify cost at your occupancy.
- Stop if the acceptance signal cannot be made cumulative; a non-prefix grant breaks the row map.
- Stop if the scheduler can see no acceptance estimate at all, from any round.

Uniform acceptance is **not** by itself a stop condition. A population with near-zero acceptance can still make one row per request the best choice, because it avoids paying for verification that would be discarded. Whether the budget helps depends on the actual cost model and the fixed overhead, not on how uniform the rates are. Measure it rather than excluding the case in advance.

## Configuration in the reference source

| Variable | Default | Effect |
|---|---|---|
| `DEEPINFER_USE_NEXTN_DECODE` | `5` | Maximum speculative width; `0` disables speculation entirely |
| `DEEPINFER_ENABLE_MTP_VARLEN` | unset → false | Enables the budgeted path; without it the draft width is fixed |
| `DEEPINFER_DFLASH_CONFIDENCE_HEAD_TEMPERATURE` | `1.0` | Per-position scale applied inside the confidence head; clamped above zero. It does not set the request sampling temperature. |

The budget is **off by default** in the reference. Do not describe it as the engine's default behaviour.
