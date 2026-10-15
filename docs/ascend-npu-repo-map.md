# Ascend NPU Repository Capability Map

## AKG (Ascend Kernel Graph Compiler)

**Purpose** (observed: [mindspore-ai/akg README at the recorded `5aa15f3f` revision](https://github.com/mindspore-ai/akg/blob/5aa15f3f/README.md))

AKG is a deep learning compiler for operator optimization and automatic fusion in DNNs. Two major subprojects:
- **AKG-AGENT**: LLM-driven operator generation framework supporting Triton-Ascend (Atlas A2/A3) and SWFT (300I DUO) DSL generation with integrated AscendKernelBench evaluation platform
- **AKG-MLIR**: MLIR-based compilation pipeline for complete operator compilation on CPU/GPU/Ascend, with MindSpore dialect and enhanced loop fusion scheduling

**Top Capability Categories** (observed)

1. **Operator Code Generation** — Auto-generate kernel code from high-level operator descriptions using LLM-guided AKG-AGENT
2. **Compiler Backends** — Multi-target support (CPU/NVIDIA/Ascend hardware backends including Atlas 800T A2/A3 and 300I DUO)
3. **Operator Fusion** — Automatic pattern-based fusion of computation chains under specific scheduling modes
4. **MLIR Dialect Extensions** — Linalg, Affine, GPU dialects with AscendNPU IR integration for fused operator generation
5. **Operator Benchmark Evaluation** — AscendKernelBench platform for performance assessment

**How an Optimization Agent Would Use It** (observed)

- Input task specifications → AKG-AGENT generates optimized Triton-Ascend kernels → validate on AscendKernelBench
- Or leverage AKG-MLIR's compilation pipeline for operator fusion and schedule exploration across Ascend backends
- Use generated kernels as building blocks for high-level model inference optimization

**Candidate Skill** (inferred)

Yes, **akg-mlir-compile-optimize** would be valuable. AKG operates at the kernel/compiler level, bridging from operator specifications to hardware execution — a gap between high-level model inference and low-level kernel tuning. Currently no OmniRSI skill covers compiler-driven optimization on Ascend.

---

## Ascend PyTorch Operator Development (cannbot/ops)

**Purpose** (observed)

Comprehensive skill suite for designing, implementing, debugging, and optimizing custom operators in AscendC and Triton-Ascend, targeting PyTorch integration on Ascend NPU.

**Top Operational Workflows** (observed: directory listing + representative SKILL.md files)

1. **Environment & Setup** — ascendc-env-check, ascendc-docs-search for configuration and knowledge discovery
2. **API Design & Exploration** — pypto-api-explore, ascendc-api-best-practices for operator interface definition
3. **Direct Kernel Invocation** — ascendc-direct-invoke-template for testing bare kernels before PyTorch binding

**Top Debugging Workflows** (observed: ascendc-crash-debug, pypto-precision-debug)

1. **Crash & Hang Debugging** — ascendc-crash-debug: coredump analysis, plog parsing, buffer/sync issues, AIV/AIC deadlock
2. **Precision & Correctness** — pypto-precision-compare (vs. reference), pypto-precision-debug (detect type/layout mismatches)
3. **Runtime Debugging** — ascendc-runtime-debug (on-device logging with PRINTF/DumpTensor)
4. **Tiling & Layout Issues** — ascendc-tiling-design for memory layout optimization, aiss-tiling-solver for complex tiling patterns

**Top Performance Workflows** (observed: pypto-op-perf-tune and subskills)

1. **Baseline Profiling** — Collect swimlane trace (merged_swimlane.json), measure core utilization, bubble rate, load balance
2. **Frontend Optimization** — tune-frontend: code idiom, TileShape, BLOCK_SIZE for ~10% uplift baseline
3. **Schedule-Level Tuning** — tune-swimlane: Stitch config, kernel fusion, L1 reuse patterns (~60% improvement potential)
4. **Core-Level Tuning** — tune-incore: instruction scheduling, pipeline optimization, special shape handling (~30%)
5. **Operator Latency Optimization** — triton-latency-optimizer for Triton-Ascend kernels

**Candidate Skills** (observed: names extracted from cannbot/ops structure)

Top 3 candidates for inclusion in OmniRSI (not yet present in worktree skills):
1. **pypto-op-perf-tune** — Multi-stage performance tuning framework (frontend → swimlane → incore) with automated profiling and delta measurement
2. **triton-op-designer** — Algorithm sketch generation in UnifiedSketch DSL for Triton-Ascend, matching hardware specs to parallelization strategy
3. **ascendc-precision-debug** — Precision mismatch diagnosis including dtype/layout validation and GDB coredump analysis

---

## Ascend Model Inference (cannbot/model)

**Purpose** (observed)

Integration and optimization layer for full model inference on Ascend NPU, covering operator fusion, parallelization, memory optimization, and runtime tuning for PyTorch models.

**Top Operational Workflows** (observed: cannbot/model skill files)

1. **Model Analysis & Matching** — model-infer-fusion: decompose model into module chain, match to torch_npu fusion operators (Attention/MoE/FFN), validate constraints
2. **Parallelization Planning** — model-infer-parallel-analysis: determine TP/EP/DP/CP split for multi-card inference, generate parallel_config
3. **Memory Optimization** — model-infer-kvcache: KV cache static/PA mode conversion, model-infer-prefetch: speculative prefetch tuning
4. **Framework Migration** — model-infer-migrator: adapt models from other inference frameworks to torch_npu + Ascend

**Top Debugging Workflows** (observed)

1. **Precision Debugging** — model-infer-precision-debug: trace mismatches across module boundaries, isolate fused op precision loss
2. **Runtime Debugging** — model-infer-runtime-debug: kernel hang detection, communication group validation, rank synchronization issues
3. **Profiling & Bottleneck Analysis** — model-infer-parallel-analysis + model-infer-superkernel for identifying memory/compute bound stages

**Top Performance Workflows** (observed)

1. **Fusion Operator Replacement** — model-infer-fusion: systematic module-level operator substitution with pre/post-validation
2. **Parallelization Implementation** — model-infer-parallel-impl: communication group creation, layer replacement (TP/EP/CP/KVP), YAML config generation
3. **Multi-Stream & Prefetch** — model-infer-multi-stream: stream scheduling for overlap, model-infer-prefetch: hidden layers via prefetch
4. **Superkernel Fusion** — model-infer-superkernel: cross-layer fusion for attention/MoE/norm chains

**Candidate Skills** (observed: names from cannbot/model directory)

Top candidates for OmniRSI expansion:
1. **model-infer-fusion** — Full operator replacement workflow: module decomposition → reference matching → API validation → staged replacement with precision/perf checks
2. **model-infer-parallel-analysis** — Parallelization strategy design: model profiling → constraint analysis → multi-scenario config generation
3. **model-infer-prefetch** — KV cache and layer prefetch optimization: hidden layer staging, speculative load balancing

---

## Coverage Gaps

**Capability areas with no dedicated cannbot skill or ascend-* skill in the worktree:**

1. **AKG Compiler Optimization** — No skill bridges operator design ↔ AKG compilation. Gap: users cannot systematically explore AKG-AGENT codegen or AKG-MLIR fusion patterns for new operators.

2. **Operator-to-Fused-Op Mapping** — model-infer-fusion covers module-level matching, but no skill covers operator-level semantic analysis (e.g., "this dense FFN matches fused npu_fused_swiglu pattern"). Blocked by lack of operator knowledge base.

3. **Cross-Model Optimization Recipes** — Each model requires custom parallelization tuning. No skill captures reusable recipes (e.g., "MoE + dynamic batching on 8 NPUs" → template configs, communication patterns, profiling baselines).

4. **Hardware-Specific Tuning** — ascendc-tiling-design exists but is low-level. No skill for mid-level NPU resource allocation (core count, UB partitioning, memory hierarchy choices) given operator type and shape.

5. **Quantization Adapters** — model-infer-fusion checks W8A8/W4A16 compatibility, but no skill automates mixed-precision operator replacement or calibration for Ascend NPU quantization modes.

**Existing ascend-* skills in worktree** (inferred from user context):

- ascend-graph-mode (likely covering npugraph_ex workflows)
- Parallel/TP/EP/DP coordination skills already present
- Basic environment and execution skills

**Not gaps** — already covered by existing skills or out of scope for OmniRSI:
- Low-level AscendC register/memory APIs (ascendc-regbase-best-practice exists)
- Torch-to-AscendC direct binding (torch-ascendc-op-extension exists in cannbot/ops)

