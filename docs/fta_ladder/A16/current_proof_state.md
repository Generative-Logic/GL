<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# A16 — current proof state (append-only, newest last)

Conjecture (row 17 of `files/shortcut/theorems/conjectures.txt`; human proof
in [`human_proof.md`](human_proof.md)):

```text
(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9](in[9,1])(>[10](in[10,1])(>[]!(strictOrder[1,4,9,10])(>[]!(=[9,10])(strictOrder[1,4,10,9]))))))
```

Goal LB (hashburst dump target since the 2026-08-05 retarget): `!(=[9,10])`
under `!(strictOrder[1,4,9,10])` under `(in[10,1])` under `(in[9,1])` under
`(AnchorFTA[1,2,3,4,5,6,7,8])`; its `main` `toBeProved` carries the head
`(strictOrder[1,4,10,9])`.

## 2026-08-05 — machinery campaign (crash chain) closed

The conjecture initially crashed the shortcut prover three contracts deep;
all three are fixed (commits `c0023c44..8a6c0c47`):
hypothesis constituents now ride `sameIterationInternalMail` and carry a
terminal internal-only `hypothesis` origin (D-259,
I-180). Shortcut run airtight — 1260 checks /
0 failures, all 16 baseline pool rows preserved. A16 itself does not prove.

## 2026-08-05 — stall investigation: the totality or is unreachable

**Evidence files:**  (diagnostic shortcut run),
 (retargeted sacred trace, 15 ENTRY/EXIT
pairs on the goal LB).

**Trace walk against the human proof.**

- The goal LB runs 15 bursts; statements grow 2 → 353 while `toBeProved`
 stays frozen at 1 (the A16 head). From global burst ~13 to the cap at 42
 the whole grid idles (`swept=0 skipped=15` — quiescent-skip).
- Human-proof step 3's ingredients ARE present at `main`: the A14 instance
 `(>[](preorder[1,4,10,9])(>[]!(=[10,9])(strictOrder[1,4,10,9])))`, the
 witness rule for `preorder[1,4,10,9]`, and the integration decomposition
 of the goal (`(&(preorder[1,4,10,9])!(=[10,9]))_integration_goal` — the
 engine correctly reduced `10 < 9` to `10 ≤ 9` plus the premise mirror).
- Human-proof step 1 — the totality case split — NEVER appears:
 `orBookkeeping (0)` / `orDisjunctCount (0)` in every one of the 15
 dumps; no or-compact with `preorder` elements exists anywhere in the
 trace (the compiled map's only ors are the Peano `or0`/`or1`). The A15
 totality reached the LB only in its single-direction implication form
 `¬(a≤b) → b≤a`, which cannot open a case split.

**The frontier mechanism.** The or theorem IS constructed — but only in the
post-run theorem-save path, after `Prover finished.` (run log:
`OR compiled:!(&!(preorder[1,4,9,10])!(preorder[1,4,10,9])) -> (or3[1,4,9,10])`,
after burst 42; likewise `or2[9,10,1,4]` for the A14 pair, both parents
then removed as subsumed). Consequences:

1. **In-run:** no grid can consume an or theorem built from the same run's
 proved implications — construction happens after every LB is done.
2. **Across runs:** the shortcut's externals input
 (`compressed_external_theorems.txt`, 69 rows — derived from the 60-row
 main-run corpus per D-253) contains no A14/A15 rows at all; the
 shortcut's own `theorems.txt` (which carries the or-forms in base form)
 is never fed back as input. So `or3` is unreachable for A16 on ANY
 rerun under the current wiring.

**Classification: architecture / pipeline gap, not a coding bug.** Every
mechanism works as designed; the designs compose so that a shortlist lemma
can never consume an or theorem constructed from earlier shortlist lemmas.
A16 is the first lemma that needs to (its proof is one case split over
A15's or).

**Candidate repair directions (maintainer decision — no code written):**

1. **Feed the shortcut's own proved theorems back as externals.** Union the
 previous run's shortcut `theorems.txt` (already expanded base form —
 D-253-compatible, registry-independent) into the externals input of the
 next `--shortcut` run. `precompileStructuralOperators` (I-1) restores
 the or-compact at load; A16's grid gets the case-split rule from burst
 0. Pipeline-level change (`run_modes.py` shortcut input wiring or a
 corpus-refresh policy extension), prover untouched.
2. **In-run or-construction.** Run the or-construction seam when both
 single directions exist (at the theorem-drain seam) and broadcast the
 or like any proved theorem, enabling one-run closure. Prover change,
 larger surface.
3. No conjecture-ordering workaround exists — construction is post-prover
 regardless of order.

**Next frontier after availability (flagged, untested):** with `or3` loaded
as a corpus rule, step 1 still needs the or-admission demand evidence to
open a cohort at the goal scope (route (a) / I-177) and sequenced release
(I-174); whether the A16 grid produces that demand evidence is the first
thing to check on the next run.

## 2026-08-05 — A16 PROVED (Phase 1: negated-AND De-Morgan door, )

The maintainer's route closed the theorem: the negated premise is itself the
case split. `¬(a<b)` de-Morgans to `¬(a≤b) ∨ (a=b)`; the new door in
`disintegrateExprCore2` decomposes any asserted negated AND-bodied
application at runtime (no operator minted, the negated compound is the
cohort signature) and feeds the shared or downstream
(`consumeOrLeavesCohort`). The K mutual-exclusion rule
`¬(=[9,10]) → ¬(preorder[1,4,9,10])` fired on the standing premise, the A15
implication produced `10≤9`, A14 sharpened it to `10<9`, and the goal
converged — exactly the human proof, no cohort needed (both leaf shapes are
probe-exempt, so no cohort opens; the K rules are the whole consumption
here).

**Results.** Shortcut: pool 16 → 17 (`strictOrder[1,4,10,9]` row present),
verifier 1388 checks / 0 failures after the consented verifier extension
(the expansion + disintegration checkers learned the negated-AND source
shape; 4 correct forcing-function failures before it). The post-run
or-construction then also built `or4` — trichotomy's own or-form — from the
new theorem and its mirror. Full standard run: 137,436 checks /
0 failures; theorems.txt 60 → 62, all baseline rows accounted for (59
verbatim, 1 moved to `compressed_out_theorems.txt` as derivable — an
unguarded interval-exclusion row superseded by three new limitSet-guarded
theorems the De-Morgan door enabled). This run also validates the
hypothesis-mail + `hypothesis`-origin changes across
all batches.

**Evidence files:** ,
, ,
.

**Deferred (planned, approved in outline):** Phase 2 — in-run or-theorem
construction + broadcast (needed by B-block lemmas that case-split on
positive or facts, e.g. B8 over trichotomy's `or4`). Plan preserved in the
session plan file.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
