# Repository Intelligence Methods: Comparison Matrix

**Date:** 2026-10-15
**Status:** candidate research note (not a SKILL.md — this is a design decision document)

## Purpose

Which data source should feed which OmniRSI subsystem?

This matrix compares six repository intelligence methods by the data source they
consume and the artifact they produce. The goal is to avoid redundant
instrumentation and pick the right source for each need.

---

## Data source × artifact matrix

| Method | Source snapshot | Git history | Execution traces | PRs / Issues / Reviews | Output artifact | OmniRSI implication |
|--------|:--------------:|:-----------:|:----------------:|:----------------------:|-----------------|---------------------|
| **AREX (Repo-To-Skill)** | ✓ primary | — | — | — | Bilingual skill module with capability graph and provenance | `skill_library/` import (PR #9 layout); use for inference-engine knowledge skills |
| **LearningToCommit** | ✓ (base snapshot at cutoff) | ✓ primary (task construction) | — | PR subjects for labeling | Historical task benchmark with temporal split | `benchmarks/skill_eval/` task construction; temporal split in `skill_distill.py` |
| **Trace2Skill** | — | — | ✓ primary (success + error traces) | — | Evolved skill via parallel patch consolidation | Skill evolution pipeline; feeds `skill-evolution-playbook` accept criterion |
| **SkillOpt** | — | — | ✓ (rollout + harvest from agent traces) | — | Evolved skill via bounded edit + held-out gate | `skills/<name>/history/` lineage; accept criterion in `npu-skill-factory` skill |
| **vllm-evolve** | ✓ (policy source code) | — | ✓ (benchmark runs across regimes) | — | Evolved serving policy code + goodput verdict | Policy-level optimization (separate from skill evolution); feeds `optimization-search-policy` |
| **magpie / LTC commit history** | ✓ (at any point) | ✓ primary | — | PR rate in subjects | Capability evolution graph, fast-iteration patterns | `repository-evolution-analysis` skill; future task benchmark for new repos |

---

## By data source

### Source snapshot only
**Use case:** distilling structured knowledge from a repository's current state.
**Best method:** AREX Repo-To-Skill.
**OmniRSI home:** `skill_library/` (imported, unverified status).
**Limitation:** produces a snapshot — does not encode how the system got there or how to navigate common failures.

### Git history + snapshot
**Use case:** building historical tasks with temporal cutoff, understanding capability evolution and fast-iteration patterns.
**Best methods:** LearningToCommit (task construction), magpie/repository-evolution analysis.
**OmniRSI home:** `benchmarks/skill_eval/` for tasks; `repository-evolution-analysis` skill for pattern knowledge.
**Limitation:** task quality depends on commit message quality; works best for repos with clear, atomic commits.

### Execution traces
**Use case:** evolving an existing skill from real agent failures; evolving serving policies from benchmark measurements.
**Best methods:** Trace2Skill (skill evolution), SkillOpt (skill evolution), vllm-evolve (policy evolution).
**OmniRSI home:** `skills/<name>/history/` for skill evolution; separate policy target for serving policy evolution.
**Key distinction:** Trace2Skill and SkillOpt operate on agent execution traces and evolve SKILL.md documents; vllm-evolve operates on serving benchmark traces and evolves Python scheduler code. These are different targets and should not be conflated.

### PRs / Issues / Reviews
**Use case:** labeling tasks by type, understanding intent behind changes, bootstrapping task descriptions.
**Best method:** PR subjects in the LearningToCommit pipeline (used for labeling, not as primary input).
**OmniRSI home:** task construction only; not a primary knowledge source.
**Limitation:** issues are often low-quality or missing for performance-oriented repos (kernel repos, NPU repos); do not depend on issues for task synthesis in these domains.

---

## By output artifact

### Knowledge (imported, unverified)
Source: snapshot → AREX → `skill_library/` entry.
Condition: must carry provenance (source repo, commit, license), `locally_verified: false`, and `skill_documents` pointer.
PR #9 already defines this format for 5 inference engines.

### Task benchmark
Source: git history → LearningToCommit pipeline → task JSONL with temporal split.
Condition: task must be verifiable by a frozen validator (CPU/static for Tier A; device for Tier B).
The `skill_distill.py`/`skill_eval.py` in the sibling workstream implements this contract.

### Skill (candidate, unverified)
Source: snapshot → AREX distillation → candidate bundle → `evidence/<campaign-id>/plan/`.
Not in `skills/` until evaluated.

### Skill (evolved, candidate)
Source: execution traces (agent or benchmark) → Trace2Skill parallel patch or SkillOpt bounded edit → candidate edit to existing skill.
Gate: `task_success(candidate) > task_success(current)` on held-out set.
OmniRSI home: `skills/<name>/history/` with rejected edits preserved in evidence tree.

### Policy code (evolved)
Source: serving benchmark traces across regimes → vllm-evolve phase-locked round → evolved scheduler/policy code.
Distinct from SKILL.md evolution — it produces runnable code, not a document.
Not yet in scope for OmniRSI (no policy evolution loop exists); the `optimization-search-policy` skill describes how to design such a loop manually.

### Capability evolution graph
Source: git history + diff sampling → `repository-evolution-analysis` procedure → JSON graph with commit SHAs.
OmniRSI home: `evidence/` or `tmp/` during analysis; content feeds into skill-conditioning documents.

---

## Recommended routing for OmniRSI subsystems

| OmniRSI need | Recommended data source | Method | Output home |
|-------------|------------------------|--------|-------------|
| Inference-engine knowledge (vLLM, SGLang, etc.) | Source snapshot | AREX | `skill_library/` + knowledge JSON |
| Historical task benchmark for skill eval | Git history | LearningToCommit pipeline | `benchmarks/skill_eval/` |
| Skill distillation (new capability area) | Source snapshot + existing human skill | AREX + `skill_distill.py` | `evidence/<id>/plan/candidates/` |
| Skill evolution (existing skill improvement) | Agent execution traces | Trace2Skill / SkillOpt | `skills/<name>/history/` |
| Fast-iteration patterns for a new repo | Git history | `repository-evolution-analysis` procedure | Skill conditioning doc |
| Serving policy optimization | Benchmark traces across regimes | vllm-evolve pattern (manual) | New policy file (not yet automated) |
| Capability boundary for a new NPU repo | Source snapshot | AREX / manual audit | `repo_packs/<new_repo>/` knowledge records |

---

## Open design question

Should snapshot-derived knowledge (AREX) and trace-derived knowledge (Trace2Skill/SkillOpt) ever merge into the same artifact?

Current answer: no. They have different provenance and different verification requirements.
- Snapshot knowledge is `locally_verified: false` at import; it describes the system as built.
- Trace-evolved knowledge has an evaluation trail (held-out task results); it reflects how the system behaves when an agent uses it.
These are complementary but should not be conflated into a single `SKILL.md` without recording both lineages separately. The `revision.parent` chain captures the evolution; the `sources` list in the knowledge record captures the original import provenance.
