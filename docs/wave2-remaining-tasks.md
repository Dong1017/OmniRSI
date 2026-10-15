# Wave 2 — Remaining Work (Parallel Agent Task Breakdown)

Generated: 2026-10-15. Base branch: `opus/meta-skill-bootstrap` at `4b48b69`.
Historical Wave 2 task snapshot; not the current execution queue or authorization.
Local source paths below use portable placeholders for publication.
Worktree: `<authorized-worktree>`.
All tasks are exploration-mode: produce candidate skills (status: candidate,
source-grounded, clearly marked unverified). No A/B evaluation required.

Every new skill goes in `skills/<name>/SKILL.md` with an optional
`metadata.json`. Commit with `feat: add candidate skill <name>` and the
attribution line:
```
Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
```

---

## TASK A — Training-System Optimization Map (Area 6)

**Output:** `docs/training-systems-optimization-map.md`

**Goal:** Identify what is common between training and inference optimization
skills and what must remain separate. Do not deeply analyze all training repos.

**Sources (already local):**
- `<reference-checkouts>/MindIE-SD/.agents/skills/`
  (look for any training-adjacent skills)
- PDFs in `<local-research-materials>/`:
  - `面向智能体的 CUDA Kernel 开发与训练框架.pdf` — kernel dev + training framework
  - `vLLM‑Evolve · 代码如何演化成调度算法.pdf` — scheduling code evolution

**Questions to answer:**
1. What optimization skill categories apply to both training and inference
   (profiling, roofline, bottleneck classification, communication overlap)?
2. What must remain training-specific (gradient checkpointing, optimizer state,
   mixed-precision training, DDP/FSDP, backward pass analysis)?
3. What must remain inference-specific (KV cache, speculative decoding,
   continuous batching, prefill/decode separation)?
4. Should a `training-systems-optimization` skill exist as a peer to
   `performance-ceiling-analysis`, or as a reference document only?

**Format:** Markdown comparison table + short prose on each boundary.
Mark all claims with their source.

---

## TASK B — Ascend NPU Repository Coverage: pytorch + op-plugin (Area 4)

**Output:** `docs/ascend-npu-repo-map.md` + optionally
`skills/ascend-kernel-development/SKILL.md`

**Goal:** For each of the two priority Ascend NPU repositories, produce a
capability map, top workflows, and a candidate skill list.

**Sources:**
- Do NOT deep-clone Ascend/pytorch or Ascend/op-plugin in this task.
- Use what is already available:
  - `<reference-checkouts>/awesome-ascend-skills/`
    — check for any pytorch or op-plugin related skills
  - `<reference-checkouts>/akg/` — read
    `akg_agents/` (repo workflow skills already indexed), `README.md`,
    `README_en.md`, `docs/` for capability context
  - `<reference-checkouts>/awesome-ascend-skills/external/cannbot/ops/`
    — operator development skills

**For each repo, produce:**
- Capability map: what the repo provides (top-level categories)
- Top operational workflows (how to build, test, profile)
- Top debugging workflows (how to isolate failures)
- Top performance workflows (how to benchmark and optimize)
- Candidate skill list (names + one-line descriptions, not full skills)

**Notes:**
- `akg`'s `akg_agents/` contains only repo-workflow skills (issue/pr/review/
  dev-insight-extractor) — do not expect operator-design/codegen skills there.
- The capability map is a design artifact; do not claim it reflects live
  repository state without citing a specific file.

---

## TASK C — ops-profiling Operator Benchmark Skill (Area 5 supplement)

**Output:** `skills/ascend-operator-profiling/SKILL.md`

**Goal:** Distill the `ops-profiling` skill from awesome-ascend-skills into a
more portable format that works outside the original operator dev workflow.

**Source:**
```
<reference-checkouts>/awesome-ascend-skills/external/cannbot/ops/ops-profiling/SKILL.md
```
Also read (if it exists):
```
<reference-checkouts>/awesome-ascend-skills/external/cannbot/ops/pypto-op-perf-tune/SKILL.md
```

**Extract and generalize:**
1. The `msprof op` collection procedure (command, parameters, output format).
2. The 8 CSV metrics files and what each measures.
3. How to derive a bottleneck verdict from the CSV data.
4. The optimization direction lookup table (which metric → which fix).
5. How the iterative benchmark→archive→analyze→fix loop works.

**License note:** awesome-ascend-skills is UNKNOWN license (synced from GitCode).
Include a license caveat in the skill's metadata.

---

## TASK D — Repository Intelligence: Chronos / Change2Task / Skill Issue (Area 7 supplement)

**Output:** Additions to `docs/repository-intelligence-matrix.md`
(append a new section, do not rewrite)

**Goal:** Add three methods to the existing comparison matrix.

**Sources:** These are NOT local — research from web or arXiv only.
- Chronos (temporal repo knowledge / time-aware RAG for code)
- Change2Task (commit → task construction, similar to LearningToCommit)
- Skill Issue (skill creation from GitHub issues)

**For each method, produce the standard matrix row:**
```
Method | Source snapshot | Git history | Traces | PRs/Issues | Output artifact | OmniRSI implication
```
Plus one paragraph: how it differs from the methods already in the matrix and
whether OmniRSI should prioritize implementing it.

If you cannot find a primary source, write "no primary source found" and skip
that method rather than guessing.

---

## TASK E — RRSI / GLM Infra Agent / AIDE² search policy (Area 1)

**Output:** Additions to `skills/optimization-search-policy/SKILL.md`
(append a "Prior art from related systems" section)

**Goal:** Find and extract concrete search policy mechanisms from RRSI, GLM
Infra Agent, and AIDE² that add something the current skill does not already
cover. The current skill covers: phase-locked round, prior-art check, negative
memory, cost-aware selection, four stopping criteria, Amdahl contribution test.

**Sources:** Web/arXiv research. Do not fabricate citations.
- RRSI: any recursive self-improvement / RSI paper or system
- GLM Infra Agent: Zhipu AI infrastructure agent (if findable)
- AIDE²: AI-driven experiment automation for ML

**Extract only:**
- A concrete mechanism that answers one of the four questions in the task file:
  - How does an agent avoid repeatedly tuning one familiar solution family?
  - How should previous failed experiments constrain the next search?
  - How should experiment cost affect candidate selection?
  - When should the agent stop local tuning and switch strategy?

If none of these systems has a primary source that adds something new, say so
and end the task. Do not pad with summaries of papers that don't answer the
questions.

---

## Agent instructions (all tasks)

1. Every claim cites a specific source file/path/URL. Label each finding
   `observed`, `computed`, or `inferred`.
2. Status: candidate on all new skills. No promotion criteria needed.
3. Use the same `metadata.json` format as existing skills in `skills/`.
4. Commit each task separately. One commit per task.
5. Do not modify existing committed skills or PHASE1_REPORT.md.
6. Do not run GPU/NPU experiments. Do not clone new large repositories
   without stating a specific unanswered question that requires it.
7. If a source file does not exist or a web search returns no primary source,
   report that explicitly rather than inferring.
