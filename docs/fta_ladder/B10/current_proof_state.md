<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
Contributions require CLA — see CONTRIBUTING.md.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# B10 — strict reflection (`a·c < b·c ⟹ a < b`) — current proof state

**Branch:**. **Status 2026-08-12: PROVED AND CLOSED — 44/44, both pipelines airtight after the maintainer-picked Option A walker fix (D-277).**

## Close-out (2026-08-12)

Maintainer picked **Option A**; implemented as `ExpressionAnalyzer::contradictionLbHoldsRecord` guarding both D-51 fallback sites (commit on this branch, unit test `buildstack_contradiction_fallback_requires_twin_record`). All `[B10TRAP]` instrumentation removed; the leftover 2026-05 per-100k `[buildStack]` progress print removed on maintainer flag (the 5M hang-cap tripwire stays).

**Acceptance (production path, one chain):** units **1449/1449**; shortcut **44/44 saved**, verifier **6020 checks / 0 failures — airtight** (B10's chapter exports; `theorem usage termination` covers all 44), method **induction v2**, exe 208.5 s; **full standard pipeline** end-to-end **138972 checks / 0 failures — airtight**, byte-exact match to the D-276 baseline grand total, 1279.5 s (in band). Logs: , .

## The row

Row 44 of `files/shortcut/theorems/conjectures.txt`:

```text
(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10,11](in3[9,10,11,5])(>[12,13](in3[12,10,13,5])(>[](strictOrder[1,4,11,13])(strictOrder[1,4,9,12])))))
```

9=a, 10=c, 11=a·c, 12=b, 13=b·c. Structural mirror of row 11 (additive order reflection) with `*` and `strictOrder`; no `c ≥ 1` guard (a zero `c` falsifies the strict premise).

## The crash

`main.py --shortcut` → prover green ("Saved 44 theorems"), then during **B10's own chapter export** (induction-condition section, successor-case LB `(in2[rec0,10,3])`):

```text
[buildStack] no origin for: (strictOrder[1,4,9,12]) | validity=main | exprKey=__contradiction__!(strictOrder[1,4,9,12])
Assertion failed: false && "buildStack: no origin found", visualizer.cpp (buildStack)
```

Exit `0xC0000409`, deterministic. ~1.3M buildStack calls before the crash (backtracking amplification, a symptom not the defect).

## Diagnosis (three trap runs, commits / / )

Traps are `[B10TRAP]`-marked in `visualizer.cpp`, file-routed to . Runs: .

**Fact 1 — provenance is complete; the prover recorded everything (Rule 16 satisfied).** In `(in2[rec0,10,3])`, `!(=[9,12])@main` holds ~30 **acyclic** `contradiction` origin records — the or0[9,12] cohort's `(=[9,12])` branch died on in-branch facts contradicting main-scope premises (the expected cancellation argument), each record citing the branch-scope/main opposing pair. Alongside them sits ONE cyclic record: `disintegration ← (&(preorder[1,4,9,12])!(=[9,12]))@main`, whose conjunction's only record is `expansion ← (strictOrder[1,4,9,12])@main` (definitional unfolding written after the head existed).

**Fact 2 — every head candidate cites `≠`.** The head `(strictOrder[1,4,9,12])@main` has exactly 4 origin candidates there (three `implication` firings of the `≤ ∧ ≠ ⟹ <` family, one `or convergence` via `(or0[9,12,1,4])`) — all four list `!(=[9,12])@main` as a dependency.

**Fact 3 — the reductio twin `__contradiction__!(strictOrder[1,4,9,12])` never converged.** 4806 origin rows, ZERO `tag=contradiction`, no row for the head at any validity except an internal hypothesis scope. The proof did not go through it (twins prime per unproved head; this one just ran).

**Fact 4 — the death sequence.** Walk: head (frame 1, emits a rule candidate) → dep `!(=[9,12])@main` (frame 2) → D-49 order tries the cyclic `disintegration` candidate first → conjunction (frame 3): only candidate `expansion(head)` is path-cyclic → frame 3 drops to the **last-resort front-emit**, which recurses into deps **without the cycle filter** → re-enters the head (frame 4) with all 4 candidates now path-cyclic (`≠` is on the path) → head's post-loop **D-51 fallback routes into the reductio twin without checking the twin holds a record** → entry-side no-origin assert kills the process. One unwind higher, frame 2's 30 acyclic candidates were waiting; the walk would have completed.

**Exonerated:** the D-276 cross-chapter admissibility check (0 skips involved any cluster expression); the prover's history emission (all needed records exist); the B10 row itself (proves fine).

**Why B10 first:** first campaign whose induction successor-case closes `≠` via an or-branch contradiction while a definitional-unfolding record of the head sits in the same LB — the D-49 order then walks the cyclic record first, and the degraded sub-walk reaches the never-converged twin.

## Proposed fix (maintainer decision required — Rule 8; verifier untouched)

**Option A (minimal, recommended):** guard both D-51 contradiction-fallback sites in `buildStack` — switch into the twin only if the twin's `exprOriginMap` actually holds an origin row for the lifted `(proved, validity)`; otherwise the fallback does not apply (entry side: continue to the assert only when genuinely no route; post-loop side: fall through to the existing last-resort/false return). Degraded sub-walks then unwind as ordinary candidate failures and backtracking reaches the acyclic records. No prover change, no verifier change; chapters that exported before still export byte-identically (the guard only changes runs that today die on the assert).

**Option B (companion, optional):** D-49 ordering — demote a candidate whose sole dependency chain is the proved expression's own definitional compound (disintegration-from-own-compound), so the acyclic records walk first. Performance/robustness only.

**Option C — nothing prover-side:** explicitly not needed; history is complete.

## Open items

None — all `[B10TRAP]` instrumentation was removed before the acceptance chain (see Close-out). The `.debug` trap artifacts remain as untracked run evidence; trap-3's candidates file appended across runs (sections distinguishable by the `SCC sweep` marker).

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
