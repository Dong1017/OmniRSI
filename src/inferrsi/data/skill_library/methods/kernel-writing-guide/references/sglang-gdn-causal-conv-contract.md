# SGLang GDN and causal-convolution contract

Use this reference to qualify the concrete Qwen3.5 or Qwen3-Next caller before
reusing a Gated DeltaNet (GDN) or causal depthwise convolution operator.
Source: [sgl-project/sglang](https://github.com/sgl-project/sglang) at
`16176278d0180f1a1a578e53e6eef432577876c2`, inspected 2026-10-29.
State: candidate; native execution `not_run`; agent efficacy `not_evaluated`.
The [source inventory](sglang-gdn-source-files.json) records retained file
identities, digests and inspected sections. No upstream package was installed
or imported, and no model, accelerator or benchmark was run for this extraction.

## Start at the model, then resolve the layer and phase

1. The text-only `qwen3_5_text.Qwen3_5ForCausalLM` delegates its body to
   `qwen3_5.Qwen3_5ForCausalLM`. In that implementation,
   `config.layers_block_type[idx]` selects a full-attention or linear decoder.
   The linear decoder constructs `Qwen3_5GatedDeltaNet`, whose projected
   Q/K/V, output gate Z and recurrence gates B/A reach `RadixLinearAttention`.
   Qwen3-Next has its own linear decoder and `Qwen3GatedDeltaNet` caller; do
   not assume that its projection unpacking is identical. [A, B]
2. `hybrid_gdn_config` recognizes the unwrapped text configuration of named
   GDN families. `attn_backend_wrapper` rejects hybrid GDN with MLA, builds a
   GDN backend for the recognized family, stamps the runner's linear backend
   choices, and wraps it with the selected full-attention backend. The hybrid
   wrapper dispatches by full-attention layer IDs and forward mode. A custom
   linear-attention registry entry can supply another backend; being a hybrid
   or registering an architecture name does not make that model GDN. [C, D]
3. Ordinary `DECODE` reaches `forward_decode`; ordinary extend, including a
   mixed extend batch, reaches `forward_extend`. State row IDs and cumulative
   token offsets come from forward metadata. Sequence length alone does not
   identify the phase, prior state or owning request. [D, E]

The CUDA-facing standard path is:

```text
model projections -> RadixLinearAttention -> HybridLinearAttnBackend
  -> GDNAttnBackend -> causal convolution -> GDN recurrence -> output gate/projection
```

Qwen3.5 also has a guarded fused projection/conv decode route and an Intel XPU
whole-GDN hook. Speculative target verification, ReplaySSM and multi-item
scoring have additional state contracts. The standard path above is not a
claim that every backend executes separate kernels. Their implementations and
external package kernels are not qualified here. [A, E]

## Resolve requested and effective GDN backends separately

| Choice | Pinned source behavior |
|---|---|
| Base / decode / prefill flags | `--linear-attn-backend` defaults to `triton`. Decode uses its explicit override or base. Prefill uses its explicit override, then the caller's automatic default, then base. The resolved immutable record belongs to one runner; target and draft runners can differ. It is not a process-wide replacement. |
| Automatic FlashInfer prefill | Requires no explicit prefill override, base `triton`, deterministic inference off, page-major layout off, CUDA capability major 9 or 10, fixed non-dynamic chunking, and key/value head dimensions both 128. Major 9 requires FP32 temporal state and chunk size 1–32768. Major 10 requires BF16 temporal state, CUDA major at least 13 and chunk size 1–8192. The FlashInfer loader must expose a prefill function. Otherwise this helper returns no automatic default. |
| Explicit FlashInfer | CUDA is required; unavailable kernel functions raise during construction. Deterministic inference with effective FlashInfer prefill is rejected. An explicit selection is not established by the automatic-default gates and is not evidence that execution succeeded. |
| CuTe DSL prefill | The dispatcher uses CuTe only when `supports_prefill` is true; otherwise it logs and selects Triton for extend. The inspected CuTe wrapper sets this capability for CUDA major at least 10, and its extend loader additionally requires key head dimension 128. Its decode path is separate. |
| Verify / other choices | Verify has its own flag and dispatcher rules; it is not ordinary prefill. The GDN dispatcher rejects Helion as KDA-only and rejects unsupported choices. Intel XPU has a separate fused hook and dispatcher-level Triton path. NPU selects separate backend classes. |

These rules come from executable predicates, not only CLI help. The help text
mentions an SM100 automatic default while the inspected policy also has an SM90
branch. CUDA capability major 8, including sm89, fails that automatic-default
predicate; this does not qualify Triton, an explicitly requested alternative,
installation or binary compatibility on that hardware. Record the actual
runner stamp, dispatcher classes and executing operator when validating. [C, F–H]

## Keep causal convolution separate from recurrence selection

The GDN module initially imports both conv functions from
`kernels.ops.mamba.causal_conv1d_triton`. On CUDA it rebinds **only the prefill
function** to the SRT conv wrapper. Thus the ordinary GDN decode update remains
the Triton import; the existence of a CUDA update wrapper is not evidence that
this caller uses it. CPU/NPU rebindings are separate and are not native
qualification for either backend. [E, I]

| Call / condition | Convolution behavior at this pin |
|---|---|
| Ordinary prefill | `_convolve_prefill` passes `[channels, total_tokens]` via transpose, weights `[channels, width]`, engine conv state, row IDs, `has_initial_state = extend_prefix_lens > 0`, cumulative offsets and CPU sequence lengths. An optional per-batch conv block table is scheduling metadata, not a persistent state cursor. |
| SRT prefill wrapper | Selects Triton if the device has no CUDA compiled path, or if the input's last stride is not 1 **and** `seq_lens_cpu` is supplied. Otherwise it makes a non-unit-last-stride input contiguous, normalizes row IDs to int32, invokes the CUDA op, and returns that input buffer. |
| Triton prefill wrapper | Allocates a separate `empty_like(x)` output, passes input/state strides, and updates the selected conv-state rows. Return shape equality does not imply input/output aliasing. First-chunk history is controlled by the initial-state mask; padded/null rows and empty spans require the declared packed metadata. |
| Ordinary decode | Calls the imported Triton `causal_conv1d_update` with the selected state rows, then packed or split GDN recurrence. The ordinary call supplies no `cache_seqlens`; its cumulative token offsets are not circular-buffer cursors. |
| Other update APIs | The SRT CUDA update wrapper accepts `cache_seqlens` and mutates its input and conv state; the JIT custom-op registration declares those mutations. The Triton update API additionally has speculative/intermediate-state arguments. These are distinct call schemas; neither is a drop-in standalone Dao or HF binding. |

Do not infer an always-CUDA prefill from `is_cuda()`: the actual transpose/view
strides and supplied CPU lengths can select Triton. Conversely, the selected
GDN recurrence class does not choose the conv implementation. The inspected JIT
loader specializes by dtype and architecture/math mode; no compilation, loaded
binary, launch or cross-implementation numerical equivalence was checked. [E, I, J]

## Preserve state layout, ownership and write-back

For a standard evenly head-sharded CUDA-facing configuration, let `P` be the
linear-attention tensor-parallel size, `Hk/Hv` key/value head counts, `K/V` their
head dimensions and `W` conv width. Per state row:

- Conv history is `[(2*Hk*K + Hv*V)/P, W-1]`.
- GDN temporal state is `[Hv/P, V, K]` (K last).
- The ordinary pool adds leading `[linear_layers, slots+1]` axes, then the
  layer accessor returns the requested layer slice. Conv and temporal state
  have independently selected dtypes; do not infer either from the model's
  activation dtype.

The shape builder can extend key-head groups for non-divisible sharding.
Read the resolved shape instead of applying the simple formula to those
configurations. CPU and page-major layouts need their own orientation/stride
check. The dtype helper defaults conv to BF16 and temporal to FP32, then applies
its declared environment/config precedence; the allocated pool dtype is the
decisive input to the backend policy. [K]

| Owner / metadata | Contract and boundary |
|---|---|
| Hybrid request pool | `alloc` preserves held continuation/cache slots or allocates a Mamba slot, marks a fresh slot as needing clear, and stores request-row → Mamba-row mapping. `get_mamba_indices` reads that mapping. The static translation is identity; metadata calls the pool's translation boundary for other layouts. A batch index is not a stable state identity. |
| Pool versus layer | `mamba2_layer_cache` supplies `conv[0]` and `temporal`; these are conv history and a recurrent matrix, not attention K/V. Pool `clear_slots` clears both and must run on the forward stream. Its complete scheduling, cancellation, radix eviction, copy-on-write and clear-invocation chain are outside this readset. |
| Ordinary extend metadata | Cumulative starts delimit the concatenated sequences, CPU lengths describe each extension, and `extend_prefix_lens > 0` permits prior conv history. Recurrent initialization is a separate pool/lifecycle responsibility: the ordinary Triton extend passes indexed temporal state, rather than using that conv mask to clear it. A standalone one-shot worker must explicitly establish fresh zero state. |
| Strided state pools | For non-CPU, non-target-verify extend, non-contiguous conv or temporal pools are gathered into contiguous per-sequence copies with identity indices. The backend scatters updated copies back into the original selected rows. Removing that write-back loses continuation state. |
| GDN recurrence adapters | Triton ordinary extend passes state, row IDs and cumulative lengths to the chunk rule with in-place update enabled; decode uses indexed recurrent update. FlashInfer extend gathers/remaps rows, prepares Q/K/V and FP32 gates, converts cumulative lengths to int64, calls its prefill function and writes final state back with `index_copy_`. This is caller-level evidence, not inspection of the external kernels. |
| Padding and special modes | Padded request row 0 maps to the reserved pool slot; `PAD_SLOT_ID=-1` is also a conv skip sentinel. FlashInfer prefill remaps negative state IDs differently for its SM90 and state-pool branches. Ordinary extend trims physical token padding while target verify has separate layout/intermediate buffers. Do not transfer a sentinel or aliasing rule across these branches. |

Source: [D, E, H, I, K]. State writes and logical token boundaries are part of
correctness, even when a caller discards its final cache after one request.
Neither this inspection nor the pool's name establishes attention KV, an HF
cache protocol or a network transfer contract.

## Transfer to one-shot classification prefill conditionally

For a worker that evaluates a complete classification input once, the reusable
questions are the concrete model implementation, projection order, Q/K/V and
gate shapes, conv weight layout/activation, sequence boundaries, normalization,
state dtype/layout, zero initial state and operator output/mutation schema.
These can guide a separate binding and correctness plan. Whole-sequence causal
convolution and chunked GDN prefill are the relevant source routes when that
worker really performs a fresh, uncached prefill. This is a conditional transfer
inference, not a delivered binding or measured improvement.

Do not import continuous-batching slot allocation, decode history updates,
speculative rollback, ReplaySSM rings, prefix-cache checkpoint policy or a
service scheduler into that worker merely to reuse an operator. Conversely,
omitting a returned cache in a one-shot task does not prove that a selected
implementation can skip required temporary state writes or preserve logits.
Inspect the actual worker's HF revision and function/instance bindings first;
SGLang source cannot identify them.

Multi-item scoring (MIS) is another distinct workload: this pin requires
prefill-only requests, zero cached prefixes, valid request-local delimiter
segments, Triton prefill and no page-major layout. Its query/item state branching
is not the ordinary independent-input classification contract. A label such as
“scoring” or “prefill-only” is insufficient to adopt it. [E, L]

Before promotion, separately authorize and capture the executing branch, compare
outputs and both state tensors to an independent reference, and cover fresh and
resumed chunks, reordered rows, ragged lengths, padding and mutation behavior
where applicable. Then evaluate classification quality and the complete timed
workload on the required hardware. Static source checks, upstream test intent,
installed resources and green repository CI establish none of those outcomes.
The [four source cases](../cases/sglang-gdn-source-contract.md) are prompts and
rubrics only; they were not executed as agent or model evaluations.

## Source ledger and attribution

All links below use the same full commit. The inventory gives exact file
digests and narrower inspected ranges; retaining a whole file does not imply
that unrelated implementations in it were qualified.

| Key | Pinned source / inspected responsibility |
|---|---|
| A | [qwen3_5_text.py, 29–71](https://github.com/sgl-project/sglang/blob/16176278d0180f1a1a578e53e6eef432577876c2/python/sglang/srt/models/qwen3_5_text.py#L29-L71); [qwen3_5.py, 354–501](https://github.com/sgl-project/sglang/blob/16176278d0180f1a1a578e53e6eef432577876c2/python/sglang/srt/models/qwen3_5.py#L354-L501), [975–1082](https://github.com/sgl-project/sglang/blob/16176278d0180f1a1a578e53e6eef432577876c2/python/sglang/srt/models/qwen3_5.py#L975-L1082), [1122–1214](https://github.com/sgl-project/sglang/blob/16176278d0180f1a1a578e53e6eef432577876c2/python/sglang/srt/models/qwen3_5.py#L1122-L1214), [1628–1751](https://github.com/sgl-project/sglang/blob/16176278d0180f1a1a578e53e6eef432577876c2/python/sglang/srt/models/qwen3_5.py#L1628-L1751) — text delegation, linear-layer construction and projected caller |
| B | [qwen3_next.py, 242–255](https://github.com/sgl-project/sglang/blob/16176278d0180f1a1a578e53e6eef432577876c2/python/sglang/srt/models/qwen3_next.py#L242-L255), [419–475](https://github.com/sgl-project/sglang/blob/16176278d0180f1a1a578e53e6eef432577876c2/python/sglang/srt/models/qwen3_next.py#L419-L475), [503–579](https://github.com/sgl-project/sglang/blob/16176278d0180f1a1a578e53e6eef432577876c2/python/sglang/srt/models/qwen3_next.py#L503-L579), [847–886](https://github.com/sgl-project/sglang/blob/16176278d0180f1a1a578e53e6eef432577876c2/python/sglang/srt/models/qwen3_next.py#L847-L886) — separate Qwen3-Next caller |
| C | [hybrid_arch.py, 35–67](https://github.com/sgl-project/sglang/blob/16176278d0180f1a1a578e53e6eef432577876c2/python/sglang/srt/configs/hybrid_arch.py#L35-L67), [140–155](https://github.com/sgl-project/sglang/blob/16176278d0180f1a1a578e53e6eef432577876c2/python/sglang/srt/configs/hybrid_arch.py#L140-L155); [linear_attn_model_registry.py, 31–85](https://github.com/sgl-project/sglang/blob/16176278d0180f1a1a578e53e6eef432577876c2/python/sglang/srt/configs/linear_attn_model_registry.py#L31-L85); [attention_registry.py, 360–572](https://github.com/sgl-project/sglang/blob/16176278d0180f1a1a578e53e6eef432577876c2/python/sglang/srt/layers/attention/attention_registry.py#L360-L572) — built-in/config-registry boundary and runner/backend construction |
| D | [radix_linear_attention.py, 38–206](https://github.com/sgl-project/sglang/blob/16176278d0180f1a1a578e53e6eef432577876c2/python/sglang/srt/layers/radix_linear_attention.py#L38-L206); [hybrid_linear_attn_backend.py, 166–337](https://github.com/sgl-project/sglang/blob/16176278d0180f1a1a578e53e6eef432577876c2/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#L166-L337), [1181–1193](https://github.com/sgl-project/sglang/blob/16176278d0180f1a1a578e53e6eef432577876c2/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#L1181-L1193), [1397–1504](https://github.com/sgl-project/sglang/blob/16176278d0180f1a1a578e53e6eef432577876c2/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#L1397-L1504) — layer/phase, logical padding and request-row metadata |
| E | [gdn_backend.py, 1–237](https://github.com/sgl-project/sglang/blob/16176278d0180f1a1a578e53e6eef432577876c2/python/sglang/srt/layers/attention/linear/gdn_backend.py#L1-L237), [516–623](https://github.com/sgl-project/sglang/blob/16176278d0180f1a1a578e53e6eef432577876c2/python/sglang/srt/layers/attention/linear/gdn_backend.py#L516-L623), [679–730](https://github.com/sgl-project/sglang/blob/16176278d0180f1a1a578e53e6eef432577876c2/python/sglang/srt/layers/attention/linear/gdn_backend.py#L679-L730), [850–944](https://github.com/sgl-project/sglang/blob/16176278d0180f1a1a578e53e6eef432577876c2/python/sglang/srt/layers/attention/linear/gdn_backend.py#L850-L944), [949–1085](https://github.com/sgl-project/sglang/blob/16176278d0180f1a1a578e53e6eef432577876c2/python/sglang/srt/layers/attention/linear/gdn_backend.py#L949-L1085), [1167–1299](https://github.com/sgl-project/sglang/blob/16176278d0180f1a1a578e53e6eef432577876c2/python/sglang/srt/layers/attention/linear/gdn_backend.py#L1167-L1299) — conv aliasing, standard forward/state writes and special-mode boundaries |
| F | [exec_.py, 480–507](https://github.com/sgl-project/sglang/blob/16176278d0180f1a1a578e53e6eef432577876c2/python/sglang/srt/arg_groups/fields/exec_.py#L480-L507); [linear/utils.py, 20–108](https://github.com/sgl-project/sglang/blob/16176278d0180f1a1a578e53e6eef432577876c2/python/sglang/srt/layers/attention/linear/utils.py#L20-L108) — flags, resolution and per-runner state |
| G | [gdn_backend.py, 191–374](https://github.com/sgl-project/sglang/blob/16176278d0180f1a1a578e53e6eef432577876c2/python/sglang/srt/layers/attention/linear/gdn_backend.py#L191-L374); [gdn_cutedsl.py, 25–69](https://github.com/sgl-project/sglang/blob/16176278d0180f1a1a578e53e6eef432577876c2/python/sglang/srt/layers/attention/linear/kernels/gdn_cutedsl.py#L25-L69) — architecture/default predicates and effective dispatch |
| H | [gdn_triton.py, 40–44](https://github.com/sgl-project/sglang/blob/16176278d0180f1a1a578e53e6eef432577876c2/python/sglang/srt/layers/attention/linear/kernels/gdn_triton.py#L40-L44), [138–215](https://github.com/sgl-project/sglang/blob/16176278d0180f1a1a578e53e6eef432577876c2/python/sglang/srt/layers/attention/linear/kernels/gdn_triton.py#L138-L215); [gdn_flashinfer.py, 167–229](https://github.com/sgl-project/sglang/blob/16176278d0180f1a1a578e53e6eef432577876c2/python/sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py#L167-L229), [481–596](https://github.com/sgl-project/sglang/blob/16176278d0180f1a1a578e53e6eef432577876c2/python/sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py#L481-L596) — recurrence adapter arguments/write-back; external kernel bodies excluded |
| I | [SRT causal_conv1d.py, 1–194](https://github.com/sgl-project/sglang/blob/16176278d0180f1a1a578e53e6eef432577876c2/python/sglang/srt/layers/attention/mamba/causal_conv1d.py#L1-L194); [causal_conv1d_triton.py, 1–15](https://github.com/sgl-project/sglang/blob/16176278d0180f1a1a578e53e6eef432577876c2/python/sglang/kernels/ops/mamba/causal_conv1d_triton.py#L1-L15), [68–146](https://github.com/sgl-project/sglang/blob/16176278d0180f1a1a578e53e6eef432577876c2/python/sglang/kernels/ops/mamba/causal_conv1d_triton.py#L68-L146), [408–589](https://github.com/sgl-project/sglang/blob/16176278d0180f1a1a578e53e6eef432577876c2/python/sglang/kernels/ops/mamba/causal_conv1d_triton.py#L408-L589), [1022–1118](https://github.com/sgl-project/sglang/blob/16176278d0180f1a1a578e53e6eef432577876c2/python/sglang/kernels/ops/mamba/causal_conv1d_triton.py#L1022-L1118) — conv wrappers, layouts and update schema |
| J | [kernel causal_conv1d.py, 1–106](https://github.com/sgl-project/sglang/blob/16176278d0180f1a1a578e53e6eef432577876c2/python/sglang/kernels/ops/mamba/causal_conv1d.py#L1-L106) — JIT builder and mutation registration; CUDA implementation/build not qualified |
| K | [qwen3_5 config, 15–35](https://github.com/sgl-project/sglang/blob/16176278d0180f1a1a578e53e6eef432577876c2/python/sglang/srt/configs/qwen3_5.py#L15-L35), [84–110](https://github.com/sgl-project/sglang/blob/16176278d0180f1a1a578e53e6eef432577876c2/python/sglang/srt/configs/qwen3_5.py#L84-L110); [qwen3_next config, 260–313](https://github.com/sgl-project/sglang/blob/16176278d0180f1a1a578e53e6eef432577876c2/python/sglang/srt/configs/qwen3_next.py#L260-L313); [mamba_utils.py, 42–107](https://github.com/sgl-project/sglang/blob/16176278d0180f1a1a578e53e6eef432577876c2/python/sglang/srt/configs/mamba_utils.py#L42-L107), [195–242](https://github.com/sgl-project/sglang/blob/16176278d0180f1a1a578e53e6eef432577876c2/python/sglang/srt/configs/mamba_utils.py#L195-L242); [memory_pool.py, 519–541](https://github.com/sgl-project/sglang/blob/16176278d0180f1a1a578e53e6eef432577876c2/python/sglang/srt/mem_cache/memory_pool.py#L519-L541), [636–751](https://github.com/sgl-project/sglang/blob/16176278d0180f1a1a578e53e6eef432577876c2/python/sglang/srt/mem_cache/memory_pool.py#L636-L751), [1101–1173](https://github.com/sgl-project/sglang/blob/16176278d0180f1a1a578e53e6eef432577876c2/python/sglang/srt/mem_cache/memory_pool.py#L1101-L1173), [1586–1688](https://github.com/sgl-project/sglang/blob/16176278d0180f1a1a578e53e6eef432577876c2/python/sglang/srt/mem_cache/memory_pool.py#L1586-L1688) — shape/dtype construction, allocation/clear declarations and row mapping only |
| L | [test_gdn_prefill_backend_policy.py](https://github.com/sgl-project/sglang/blob/16176278d0180f1a1a578e53e6eef432577876c2/test/registered/unit/layers/attention/test_gdn_prefill_backend_policy.py), [test_linear_attn_config.py](https://github.com/sgl-project/sglang/blob/16176278d0180f1a1a578e53e6eef432577876c2/test/registered/unit/layers/attention/test_linear_attn_config.py), [test_linear_attn_model_registry.py](https://github.com/sgl-project/sglang/blob/16176278d0180f1a1a578e53e6eef432577876c2/test/registered/unit/configs/test_linear_attn_model_registry.py), [test_gdn_mis_metadata.py](https://github.com/sgl-project/sglang/blob/16176278d0180f1a1a578e53e6eef432577876c2/test/registered/unit/layers/test_gdn_mis_metadata.py), [test_causal_conv1d.py](https://github.com/sgl-project/sglang/blob/16176278d0180f1a1a578e53e6eef432577876c2/test/registered/layers/mamba/test_causal_conv1d.py) — selected upstream test intent only, not executed results |

Root license: [complete retained Apache-2.0 text](licenses/sglang-LICENSE.txt),
also [upstream LICENSE](https://github.com/sgl-project/sglang/blob/16176278d0180f1a1a578e53e6eef432577876c2/LICENSE).
Qwen files credit the Qwen and SGLang teams. The SRT conv file retains its
vLLM adaptation notice, Apache-2.0 identifier, vLLM contributor attribution and
Tri Dao copyright/adaptation source; the Triton conv file also retains the
vLLM/Dao adaptation notices. This addition paraphrases those caller contracts
and does not vendor implementation. FlashInfer, CuTe DSL, FLA, NPU/XPU packages,
model weights and compiled binaries have independent licensing and execution
boundaries; their imported symbols do not extend this readset or its license.
