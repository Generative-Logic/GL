<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# A17 — current proof state (append-only, newest last)

Conjectures (rows 18–19 of `files/shortcut/theorems/conjectures.txt`;
human proof in [`human_proof.md`](human_proof.md)). from main tip.

Goal LB (hashburst dump target since the 2026-08-05 retarget): the
forward grid's innermost premise LB `(strictOrder[1,4,11,10])` under
`(in2[9,10,3])` under `(AnchorFTA[1,2,3,4,5,6,7,8])`; its `main`
`toBeProved` carries the head `(preorder[1,4,11,9])`.

## 2026-08-05 — first run: backward proved, forward stalls

First shortcut run (): airtight, 1474
checks / 0 failures, pool 17 → 18. Backward row present verbatim in
`theorems.txt`; forward row absent. No crash.

## 2026-08-05 — n-ary OR-theorem construction fix (maintainer-ordered, in passing)

The A16 trichotomy row exported incomplete (`¬(a=b) → (a<b ∨ b<a)` —
the second negated premise left behind as a hypothesis).
`constructOrTheorem` now folds EVERY binder-free negated premise into
one n-ary or head and `expandToBaseForm` writes the flat De Morgan form
(D-260). Verified: row 18 of `theorems.txt` is
the full `a<b ∨ a=b ∨ b<a` (`or4` minted with 3 elements), verifier
1474 / 0, unit tests 1377/1377. A latent k≥3 polarity defect in the two
in-run or-EXPANSION twins (span `expandSignature` OR case, verifier
`_build_or_from_elements`) was found, flagged in
`20_core_concepts/07_or_branching.md` Suspected fragility, and left
untouched (verifier half is I-16-gated).

## 2026-08-05 — stall investigation: demand-starved case split (forward)

**Evidence files:**  (diagnostic shortcut
run),  (retargeted sacred trace, 22
ENTRY/EXIT pairs on the forward goal LB).

**What stages correctly (burst 1 already):**

- `strictOrder[1,4,11,10]` expands to
 `(&(preorder[1,4,11,10])!(=[11,10]))` and disintegrates: both
 `!(=[11,10])` and `(preorder[1,4,11,10])` live at `main`.
- The ≤-witness mints: `(in3[11,int_lev_2_1,10,4])` (a + w = s(b)) with
 `(in[int_lev_2_1,1])`; the commuted `(in3[int_lev_2_1,11,10,4])`
 also lands later.
- The goal's integration decomposition is ready: the reformulated intro
 rule `(>[pi](in[pi,1])(>[](in3[11,pi,9,4])(preorder[1,4,11,9])))` —
 one concrete `a + m = b` fact away from closing.
- **The predecessor or IS instantiated on the witness:**
 `(or1[int_lev_2_1,2,1,3])` is a registered statement at `v=main`
 from burst 1, derived by the corpus rule
 `(>[7](in[7,1])(or1[7,2,1,3]))` firing on the witness typing fact.

**The frontier.** The case split (human-proof step 2) never opens:
`orBookkeeping (0)` / `orDisjunctCount (0)` and zero `_ordis_` scopes in
all 22 dumps; `toBeProved` frozen at 1 for all 22 bursts while
statements grow 6 → 447.

**The mechanism.** Cohort opening is two-route (D-250-era design,
I-177/I-178):

- **Route (b)** — products of disintegration open unconditionally.
 `or1[w]` arrived as a plain implication FIRING product, not a
 disintegration product → route (b) does not apply.
- **Route (a)** — demand-driven: some disjunct's product template must
 already be an algebra `admissionMap` key (the probe consults the
 algebra admission map ONLY). The or's only demandable product
 template is `(in2[marker,int_lev_2_1,3])` (the existence disjunct's
 witness fact; the equality disjunct yields no template by design).
 **The algebra `admissionMap` is EMPTY — `(0)` in all 88 dumped
 sections across all 22 bursts.** No demand can ever hit.
- Consequently every or1 instance (`or1[w]`, `or1[11]`, `or1[10]`,
 `or1[9]`) parks forever in `rejectedMapOrdis` under its
 `(in2[marker,·,3])` product template (the I-178 parking is visibly
 correct in the trace), and mail revival never fires because no
 admission key is ever gained.

**Why no admission key can install.** An algebra admission key installs
at ONE-missing-premise state (all other premises of a rule instance
concrete, the marker premise the single gap). The only in2-consuming
rule that could name the pred fact is the successor-transport corpus
row 28 `(>[7,8,9](in3[7,8,9,4])(>[10](in2[8,10,3])(>[11](in2[11,7,3])
(in3[10,11,9,4]))))`. Every instantiation of it that would demand
`(in2[marker,int_lev_2_1,3])` also lacks a SECOND premise — a successor
or predecessor fact for the abstract bound variable 11 (no `s(11)` /
`pred(11)` statement exists, and none can: 11 is the universal `a`).
Two gaps → no admission key → no demand → the or that would fill BOTH
gaps in sequence never opens. Chicken-and-egg by construction.
(The `admissionMapIntegration` keys present at `main` are all
output-slot sum demands `(in3[a,b,marker,4])` — a different key space
the probe deliberately does not consult, and none names the pred
fact.)

**Why the backward direction proved without any of this:** its witness
arithmetic (a + q = b ⊢ a + s(q) = s(b)) runs forward through the `+1` /
associativity integration route — existence-witness minting, no case
split, no or machinery.

**Classification: architecture gap, not a coding bug.** Every observed
mechanism behaves exactly as designed (parking, probe, admission
gates, K-rule suppression for an unopened cohort). The designs compose
so that a case split whose branch products are needed at demand
DEPTH TWO (the goal wants `a+m=b`; deriving `a+m=b` wants the branch
product `s(m)=w`) can never accumulate the one-missing-premise
admission evidence that route (a) requires. A17-forward is the first
shortlist lemma whose proof needs an or opened on a MINTED WITNESS
(rungs 1–2 opened cohorts via route (b) disintegration products;
A16's split came from the De-Morgan door on a negated premise).

**Candidate repair directions (maintainer decision — no code written):**

1. **Let parked-rule marker templates count as demand.** The transport
 rule itself is parked in `rejectedMap` with marker values
 (`(in2[marker,·,3])`-shaped, u-template space, `origImpl` = row 28).
 Matching the or's product templates against REJECTED-map marker
 templates (template-space, instead of / in addition to the concrete
 algebra `admissionMap` keys) would express "some rule wants a pred
 fact, for anything" — transitive demand, one extra probe surface.
 Explosion exposure bounded by the existing sequenced
 one-branch-at-a-time release (I-174) and depth gate.
2. **Relax the ordis-route concreteness gate** so ≥2-missing-premise
 instantiations may still install `ordisOnly`-tagged (invisible to
 Pass B, I-177) demand keys for the missing in-slot premise. Same
 effect as (1) expressed at install time rather than probe time.
3. **Widen route (b)** to open cohorts for or statements whose
 instance argument is a minted witness (`it_` / `int_lev_` name
 shape) at bounded or-depth — "a case split on a witness the grid
 itself created is always on-topic". Narrower than general
 statement-or opening; no admission machinery involved.

**Next frontier after opening (flagged, untested):** branch 1's
reductio (a + 0 = s(b) needs the plus-zero corpus row against
`!(=[11,10])`) and branch 2's transport + injectivity chain both look
statement-available in the trace, but neither has been exercised at a
branch scope in this grid.

**Rule-30 note:** no temporary traps were added in this investigation;
the only instrumentation change is the permanent sacred-dump retarget.

## 2026-08-05 — A17 FORWARD PROVED (helper-lemma route, maintainer-designed)

Resolution came in two maintainer-approved pool additions, no machinery
or config change ("we have an unfair advantage that we can add lemmas"):

1. **A17a (row 20)** — `a < b → (a + k = b → k ≠ 0)`:
 `(>[9,10](strictOrder[1,4,9,10])(>[11](in3[9,11,10,4])!(=[2,11])))`.
 Proved by the reductio machinery first run. In the forward grid it
 fires on the standing premises and registers `!(=[int_lev_2_1,2])`
 (w ≠ 0) at `main` — the trigger the already-emitted K rule
 `¬(w=0) → ∃pred(w)` was waiting for (trace :
 w≠0 from burst 5, `existence11[1,w,3]` deposited, predecessor
 witnesses minted, `m+1=w` and `11+m=y` derived). The disjunctive
 syllogism the repair needed was the K rule all along — the parked
 `rejectedMapOrdis` cohort never had to open.

2. **Second frontier found in the same trace:** the closing
 associativity join (row 27 over {w+11=10, 11+m=y, 1+m=w} → y+1=10)
 binds THREE distinct minted witnesses at `v=main`;
 `requestGatesPass` drops any request over
 `maxNumberSecondaryVariables = 2` outside `_orint_` scopes (D-210).
 All premises present from burst 8, 14 idle bursts, request never
 admitted.

3. **A17b (row 21)** — `a + w = s(b) ∧ w = s(m) → a + m = b`:
 `(>[9,10](in2[9,10,3])(>[11,12](in3[11,12,10,4])(>[13](in2[13,12,3])(in3[11,13,9,4]))))`.
 The below-successor arithmetic core as one transport row (corpus
 row-28 shape family). Its OWN grid proves cap-free (every variable
 conjecture-bound; at most one minted witness per request), and its
 firing in the forward grid binds only w and m — under the cap — with
 head `in3[11,m,9,4]`, directly the goal's staged integration intro
 input. Chosen over widening `maxNumberSecondaryVariables` to 3 for
 the batch (D-210's measured explosions).

**Result:** shortcut run airtight, 1734 checks / 0 failures; pool
19 → 21 rows; A17 forward row present verbatim — **A17 complete, both
directions proved** (backward row 5, forward + A17a + A17b new).
Evidence: , .

**Campaign lesson (for A18/A19):** when a proof needs a k-witness
arithmetic chain, prove the chain as a pool lemma over BOUND variables
(cap-free in its own grid) so its single-rule firing in the consuming
grid stays within the secondary-variable budget. A17b is a natural
library row for A18 (successor reflection) and A19 (no gap).

## 2026-08-05 — full standard-run regression PASSED

The branch's only machinery change (the n-ary or-construction fold)
replicates the last successful full run exactly: verifier 137,436
checks / 0 failures airtight, `files/theorems/theorems.txt` 62 rows
byte-identical to the pre-run baseline
(; run log
). Every standard-batch or chain carries a
single negated premise, where the n-ary fold is byte-identical to the
former two-element pairing. Branch DoD discharged; ready to squash on
maintainer decision.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
