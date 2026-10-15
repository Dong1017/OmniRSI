---
name: skill-evolution-playbook
description: >
  Practical design choices for evolving agent skills from execution traces:
  when one trajectory should update a skill, how conflicting lessons are
  handled, how negative evidence is preserved, how large an edit should be,
  and when skills should split or merge. Drawn from Trace2Skill and SkillOpt.
  Use when designing or operating a skill evolution pipeline.
tags:
  - skill-evolution
  - trace2skill
  - skillopt
  - rsi
status: candidate
---

# Skill Evolution Playbook

Status: candidate. Source-grounded from Trace2Skill @ 7e5a1d77 (arXiv 2603.25158)
and SkillOpt @ 343db229.

## When should one trajectory update a skill?

One trajectory alone should not update a skill. The core risk is that a single
anecdotal success or failure drives a rewrite that degrades performance on cases
not covered by that one trace.

**Trace2Skill's answer:** analyze a pool of traces in parallel, then consolidate
into a conflict-free edit. The minimum pool size that provides meaningful signal
is 10–20 trajectories; single-trajectory updates are explicitly out of scope.
Source: Trace2Skill README @ 7e5a1d77.

**SkillOpt's answer:** gate every proposed edit with a held-out evaluation set
before accepting. The gate criterion: `candidate_accuracy > current_accuracy`
on the held-out set. A single failed trajectory triggers a candidate proposal,
but acceptance requires held-out evidence.
Source: SkillOpt `evaluation/gate.py` @ 343db229 (line ~200: `if cand_score > current_score`).

**Practical minimum:** require at minimum 3 trajectories showing the same pattern
(or 1 trajectory + held-out test pass) before modifying a skill. Record the
trajectory pool alongside the edit so the decision can be audited.

## How are conflicting lessons handled?

Two trajectories can encode contradictory lessons: one shows approach A works,
another shows approach A fails.

**Trace2Skill's approach:** parallel multi-agent patch proposal followed by
hierarchical conflict-free consolidation. The `skill_evolver` agents each
propose section-level patches (`insert_after`, `replace_in_section`,
`add_section`, etc.); the consolidation step identifies conflicting edits to
the same section and resolves them through a separate consolidation model call.
Source: `skill_evolver/skill_evolving_agent.py`, `_SUPPORTED_PATCH_OPS` @ 7e5a1d77.

**Practical resolution strategies:**
1. **Context-condition the lesson.** If A works in context X but fails in
   context Y, the skill should encode both as conditional: "When X, do A.
   When Y, avoid A." This is better than choosing one and losing the other.
2. **Separate sections.** Conflicting lessons often target different conditions.
   Add a "known failure modes" or "does not apply when" section rather than
   overwriting the "procedure" section.
3. **Escalate to human review.** When the conflict cannot be resolved by
   conditioning (the same context genuinely produces different outcomes across
   runs), record it as INCONCLUSIVE and flag for human review. Do not resolve
   by voting on whichever outcome appeared more often in the pool.

## How is negative evidence preserved?

**SkillOpt:** rejected candidate edits are stored in the staging area with
their evaluation result before being discarded. The staging state records
what was tried and why it was rejected.
Source: `skillopt_sleep/staging.py` @ 343db229.

**Trace2Skill:** error traces are analyzed separately from success traces, and
both pools contribute to the consolidated edit. Error-driven evolution produces
failure-mode additions to the skill; success-driven evolution adds positive
procedure refinements. Neither is discarded.
Source: Trace2Skill `run_parallel_skill_evolution.py` (error traces) vs
`run_parallel_combined_skill_evolution.py` (combined success+error traces) @ 7e5a1d77.

**What to preserve in a rejection record:**
```yaml
rejected_edit:
  proposed_change: <what was changed and in which section>
  hypothesis: <why we thought this would help>
  held_out_result: <metric before → metric after>
  verdict: FAIL or INCONCLUSIVE
  evidence_level: <how many trajectories / what held-out set>
  context: <model, hardware, workload, task distribution>
  date: <ISO date>
```

The rejection record is stored alongside the current skill version, not in a
separate archive. It is consulted before proposing the same edit again.

## How large should a skill edit be?

**SkillOpt's bounded edit:** modifications are restricted to specific sections
(`procedure`, `failure_modes`, `knowledge_sources`). The `clip.py` module
enforces a length budget: the candidate must not be substantially longer than
the current skill. The `revision.parent` field tracks the lineage so v0→v1→v2
is reconstructible.
Source: `skillopt/optimizer/clip.py`, `skillopt/optimizer/skill.py` @ 343db229.

**Trace2Skill's section-level patches:** edits target named sections with
specific operations. `replace_in_section` replaces content within a section,
`insert_after` adds content after a named anchor. Protected files (`LICENSE.txt`,
`recalc.py`) cannot be modified regardless of what the patch proposes.
Source: `skill_evolver/skill_evolving_agent.py` PROTECTED_FILES, PATCH_OPS @ 7e5a1d77.

**Practical size rule:**
- Single evolution step: change ≤2 sections, add ≤1 new section.
- Length delta: ±20% of current skill length.
- Never replace the entire `procedure` section in a single edit. If the
  procedure needs a full rewrite, that is a signal to create a new skill
  version (v1) rather than a patch.
- Never change the `capability` or `applicability` fields in an evolution step.
  Those define what the skill is for; changing them means the skill's identity
  has changed, which requires a new name and explicit human decision.

## When should skills split or merge?

**Split a skill when:**
- The `procedure` has grown to cover two distinct workflows that rarely co-occur.
  The signal: most trajectories use only the first half or only the second half,
  rarely both.
- Two distinct `applicability` conditions consistently produce conflicting lesson
  proposals. The conflict is structural, not resolvable by conditioning.
- The skill has grown beyond ~500 words and agents frequently cite only a small
  portion of it.

**Merge skills when:**
- Two skills consistently co-activate (every trajectory that uses skill A also
  uses skill B, with little overlap with other skills).
- One skill's `procedure` duplicates a large portion of another's.
- The two skills' `applicability` conditions are indistinguishable in practice.

**What to version when splitting:**
The original skill becomes deprecated (v_last) with a forward pointer to both
new skills. Both new skills start at v0 with `revision.parent = <original_skill_id>`.

## Skill deepening vs skill creation

Trace2Skill studies two modes:
- **Self-deepen:** start from an existing human-written skill; use traces to
  refine procedure, add failure modes, and extend coverage.
- **Self-create:** start from a minimal draft (e.g., name + description only);
  build up from error traces.

Self-deepening produces better skills faster because the human baseline encodes
structural knowledge that error traces alone cannot reconstruct. For OmniRSI,
prefer self-deepening from a human-authored reference skill whenever one exists.
Self-creation from scratch is appropriate for capability gaps with no prior
human-authored skill.
Source: Trace2Skill paper abstract @ arXiv 2603.25158.

## What should be versioned?

- The skill document itself (`SKILL.md`) at each accepted version.
- The trajectory pool or trace IDs that motivated each edit.
- The held-out evaluation result (metric before → after) for each accepted edit.
- Rejected candidates and their evaluation results.
- The `revision.parent` chain so the full lineage is reconstructible.

What does NOT need to be versioned separately:
- The raw agent trajectories (store a reference/ID, not a full copy).
- Intermediate patch proposals that were not accepted.
- The evaluation harness configuration (that is versioned separately in the
  benchmark/evidence tree).
