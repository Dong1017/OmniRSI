# Capability presence audit

A procedure for answering one narrow question: **does the claimed capability actually run at this revision, and through what gate?** It is narrower than a capability-family analysis and wider than reading a README.

Use this reference when a release note, README or vendor document lists features and you must state what the code does. Do not use it for performance claims — this classifies presence and wiring, not speed.

## The four buckets

Every claim lands in exactly one:

| Bucket | Test | Typical evidence |
|---|---|---|
| **Implemented** | A reachable call path from the public entrypoint changes behaviour | A caller that is not gated off, plus a body that is not empty |
| **Hook** | A symbol exists and is called, but its body is empty, or its caller is gated off by a value that never occurs | Empty virtual method with no subclass; a body behind a constant-false gate |
| **Documented only** | No symbol, or symbols only in tests and docs | A tree-wide search that returns nothing in the called graph |
| **Contradicted** | The code does something different from the claim | A default that inverts the claim; a backend that asserts instead of degrading |

A claim that fits two buckets is under-specified. Split it before classifying: "asynchronous scheduling is enabled by default" is two claims, one about existence and one about the default.

## Procedure

1. **Record the claim verbatim.** Do not paraphrase into something easier to confirm. Note whether it is a feature list, a hardware list, or an exclusion list. Exclusion lists are the easiest to audit and the most informative: a README that says "does not ship X" is making a falsifiable claim.

2. **Find the entrypoint that would exercise the claim.** Follow from the public entry, not from the directory that shares the claim's name. Name-adjacency is the most common false positive.

3. **Classify.** Use the table above.

4. **For every hook, find the gate.** Ask what would have to be true for the body to run, then check whether that condition can be true in this build. An empty virtual method plus no subclass is a hook. A real body behind a flag whose value is a constant is also a hook. A flag that defaults false is *not* a hook if the documentation says it is optional.

5. **Check for subclasses and registries.** A base class with virtual methods proves nothing. Search for derived types and for registration. Record the command shape, because "no subclass found" is a claim about a search.

6. **Check defaults, not just existence.** A feature can be present and inert. Read the environment-variable or configuration default and quote it. Then check whether some other construction path overrides it — a `setdefault` in a dataclass `__post_init__` can make a variable default-false in one sense and default-true in another.

7. **Separate the four claim strengths.** Exists, is correct, is fast, is end-to-end usable. Never let one imply another.

8. **State the extraction boundary.** Say explicitly whether the tree is complete. Only a complete tree lets "absent" mean absent; a sparse checkout licenses only "not located in the inspected paths".

## Worked instance

Applied once to `deepseek-ai/DeepInfer` @ `8b16149741a4935e1dec8cea27853f273b374c5b`. That snapshot is the complete parent tree with an untruncated Git tree, which is what allows a strong absence claim.

| Claim (verbatim from README) | Bucket | Deciding evidence |
|---|---|---|
| "Asynchronous scheduling (zero-overhead batch scheduler)" / "Enabled by default." | **Implemented**, but the default claim is **Contradicted** for one path | The pipeline exists and runs (`worker.py:183-188`). The environment variable defaults false (`environ.py:70-71`); `EngineArgs.__post_init__` sets it via `setdefault` from `engine_type.default_zero_overhead_scheduling`, true for `NORMAL`/`COMPLETION` and false for `PROMPT` (`arg_utils.py:52-55`, `engine_type.py:15-17`). Both statements hold for different entrypoints — that is the finding. |
| "Parallelism: Attention DP / balanced CP, MoE EP + EPLB" | EP/CP **Implemented**; EPLB **Hook** by default | The rearrangement interval defaults to `0`, so the rebalancing loop never runs unless configured |
| "A KV cache system" | **Documented only** in this release | Save hooks are empty virtual methods in the block between the "虚函数" markers (`llm_engine.py:800-857`), and no subclass exists |
| "VE/Prefill/Decode disaggregation (most required engine functionality is already implemented)" | **Hook** | `init_pd_split`, `send_processed_prompts`, `post_process_model_outputs` are empty virtuals (`llm_engine.py:801,848,855`); a whole-repo search for `LLMEngine` subclasses returns only the class itself |
| "Load balancing across data-parallel replicas for both prefill and decode" | **Documented only** | No symbol found in the called graph |

Two subtleties the exercise surfaced:

- **`finalize_seq` is not empty.** It is inside the same virtual block but calls `scheduler.free_seq` (`llm_engine.py:804-806`). So the hook block is heterogeneous: an implementation inheriting it would get partial behaviour rather than nothing. A blanket "all these are empty" statement would be wrong.
- **A body can exist and never run.** One state-copy path has a complete implementation (`llm_engine.py:787-796`), but its gate `should_copy_state` returns `False` for every sequence in the base class (`:813-815`) and its buffer is never allocated. Body-present is evidence of intent, not execution.

## Pitfalls

- **Name-adjacency.** A directory or class named after the feature is not evidence that the feature runs. In the audited source, three claimed-absent components were present as named empty methods — the name was *more* misleading than absence would have been.
- **Body-present-is-not-reachable.** See the state-copy case above.
- **Flag defaults.** Two headline features were inert by default; one was on only through a specific constructor path and off if the engine object was built another way. Reading the variable's default alone gives the wrong answer for the entrypoint a user actually takes.
- **Hardware lists.** A supported-device list is not a compatibility guarantee. Check whether unsupported cases assert or degrade silently; in the audited source one property hard-asserted on the non-NVIDIA backend.
- **Documentation that describes the deployment, not the code.** A production Q&A can truthfully describe a shape the open release does not implement. Label it DOC and keep it separate from CODE findings — both statements can be true.
- **Comments as evidence, carefully.** A source comment saying "virtual method, handles finished requests" is the author's description, not the behaviour. Read the body. The same applies in reverse: a comment stating a correctness requirement (as `mtp_varlen.hpp:160` does) is evidence of intent that must then be checked against the code that follows it.
- **Sparse checkouts.** Absence in a curated skill-doc checkout proves nothing. Say so rather than reporting a feature as missing.

## Output

A table with one row per claim: claim verbatim, bucket, deciding evidence, the gate that would enable it, residual uncertainty. Then a short list of findings that change how the release may be described.

## Evidence this instance produced

- `deepinfer/engine/llm_engine.py:800-857` — the virtual-hook block, bounded by its own comment markers at `:799` and `:857`.
- `deepinfer/engine/llm_engine.py:787-796` and the gate at `:813-815` — a complete implementation with a gate that returns `False` in the base class.
- `README.md:12,15,16,26,27,40-42` — the claim list and the exclusion list.
- Command shape for absence: a whole-repo search for `class .*LLMEngine`, one hit, the definition itself.

Full analysis: the DeepInfer distillation package, `notes/07-execution-path-and-scope.md`.
