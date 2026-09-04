<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
Contributions require CLA — see CONTRIBUTING.md.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# B3 (unit product) — current proof state

Append-only; newest entries last.

## 2026-08-06 — first shortcut run: both rows unproved (stall)

Rows 33–34 (`(>[9,10](in3[9,10,6,5])(>[](in[10,1])(=[6,9])))` and the `(in[9,1])`/`(=[6,10])` twin, mirroring the additive sum-zero rows 6/7). Run : 32/34 theorems, verifier 3208/0 (exactly the B2 baseline), runtime 85 s. Frontier frozen from burst ~31 to the cap at 42 (`active_bodies=15`, `total_exprs=12398` unchanged) — a deductive stall, not a runtime pathology.

## 2026-08-06 — stall investigation (playbook), dump-backed diagnosis

**Setup.** Sacred dump retargeted (chain-match only) at the B3a goal LB — chain `root → (AnchorFTA[1,2,3,4,5,6,7,8]) → (in3[9,10,6,5]) → (in[10,1])`, goal `(=[6,9])` at main. Diagnosis run  (stall reproduced byte-for-byte: 32/34, 3208/0); trace snapshot  (24 dumps, bursts 1–24 of this LB).

**Observed frontier.**

- Goal count frozen at 1 (`(=[6,9])` at `v=main`) across all 24 dumps; statements plateau at 270, all at `v=main` by the end.
- The goal was attacked via the integration/hypothesis machinery (scope `main_boundary_product_of_hypo_disintegration_of_integration_goal_(=[6,9])`), which opened predecessor case splits on both variables: `_ordis_` cohorts `(or1[10,2,1,3])` and `(or1[9,2,1,3])`, each with the `(existence11[...])` branch released first (I-174 ranks non-equalities first; the equality branches never released).
- In the `or1[10]` existence branch (10 = s(w), w = `it_0_lev_1_76`): the branch derived `(in2[it_0_lev_1_76,10,3])`, witness typing, and the product fact `(in3[9,it_0_lev_1_76,·,5])` (9·w = P) by EXIT #10, and the branch stayed alive with those facts through EXIT #14+ — **four-plus full bursts with every needed premise simultaneously live**.
- The corpus multiplication recursion rules are installed in this LB's `overallHashMemory.originals` — forward `[91]` and the needed inverse `[96]` `(>[1](in[1,u_1])(>[2](in2[1,2,u_3])(>[3,4](in3[3,1,4,u_5])(>[5](in3[3,2,5,u_5])(in3[4,3,5,u_4])))))` — and are marker-indexed (`markerExpr=(in2[1,marker,u_3])` entries present in the final dump, still pending).
- Binding rule [96] with (1:=w, 2:=10, 3:=9, 4:=P, 5:=1): all four premises live at mutually comparable scopes (`in3[9,10,6,5]` at main is an ancestor of the branch scope). Expected head: `(in3[P,9,6,4])` (9·w + 9 = 1).
- **The head never mints: zero `(in3[<witness>,9,6,4])` matches in the whole 402k-line trace.** Without it there is no `9 ≤ 1`, no downstream 0/1 analysis, no branch resolution; the cohort hangs unresolved (neither converged nor refuted), the winnable equality branches never release, and the goal freezes.

**Gates checked and passing (static reading + trace).**

- Iteration cap `maxIterationNumberVariable: 1` — branch witnesses are `it_0`/`it_1` (`extractMaxIterationNumber` reads the `it_<i>` prefix; `int_…` names do not match) and the gates are strict `>`; the firing mints no new witness. Passes.
- Secondary cap `maxNumberSecondaryVariables: 2` (per-statement witness-occurrence count) — every premise and the head carry ≤ 2 witness names. Passes.
- `u_`-literal prune — all `u_` slots bind the anchor digit literals (1, 3, 4, 5). Passes.
- Scope comparability — all premises on one ancestor chain. Passes.
- No burst truncation: `BurstSink::canAccept` carries no cap (bursts run to completion); the LB was never split in this run.
- Attribution cross-check: the 3-premise successor-injectivity corpus rule `(>[1](in[1,u_1])(>[2](in2[2,1,u_3])(>[3](in2[3,1,u_3])(=[2,3]))))` fired 154 times in this same LB — the general corpus-rule path works at 3 premises. **No firing of either 4-premise recursion rule appears anywhere in the trace** (`<- implication | (>[1](in[1,u_1])(>[2](in2[1,2,u_3])` count: 0).

**Classification (per playbook step 7).** The frontier is a non-firing 4-premise corpus rule whose premises, index entries, and every statically checkable gate are in order. The request-generator's own contract (`generateEncodedRequestsStatic` header: grow depth `maxKeyLength − stumpLen`, subkey sets at all four lengths) says the combination should complete — so this is either (a) a **coding bug** in the growth/merge path for 4-element keys, or (b) an **undocumented architecture gate** that in practice ceilings general firings at 3 premises. The dump alone cannot separate (a) from (b); one Rule-12 targeted trap inside the growth DFS (log per-node probe verdicts for the two recursion originals in this LB) would decide it in a single run.

Note the corroborating pattern: B1 and B2 consumed the same recursion equations successfully but **via the induction machinery** (methods `induction v2` / `induction v1`), which does not route through the general request growth. The gap only surfaces when a proof needs recursion facts inside an or-branch, as B3 does.

**Candidate repair directions (maintainer decision, none executed).**

1. **Pin the gate:** one trap run in the growth DFS as above (diagnosis-only, removable per Rule 30).
2. **Pool bypass (unfair-advantage rule):** provide the order–multiplication bridge as pool lemmas proved by induction — B6 (`b≥1 ⟹ a ≤ a·b`) pulled forward, and/or the 2-strength B2 twin (`a≥2 ∧ b≥1 ⟹ a·b≥2`) — so B3's case split consumes 1–2-premise pool rules and never needs the 4-premise firing. Works regardless of what (1) finds.
3. If (1) finds a deliberate ceiling: config/architecture decision on raising it for the shortcut batch (Rule 8).

**Evidence files:** , , . Dump retarget commit (the Rule-14 retarget stays; no temporary traps were added in this investigation).

## 2026-08-06 — trap runs (maintainer option 1): ROOT CAUSE PINNED — the I-6 Pass-B single-input gate

Two `[B3TRAP]` runs (instrumentation commit, five probe points in `generateEncodedRequestsStatic`: filter census, stump report, DFS node verdicts, merge verdicts; all gated on the Rule-12 chain match).

**Trap run 1 — goal LB ().** Decisive negative: across every burst, the goal LB's request universe (`intEncodedStatements`) contains exactly ONE family statement — the main-scope `(in3[9,10,6,5])`. The branch facts seen earlier in the goal LB's dump live ONLY in `exprOriginMap` — history, not statements (statements travel main-only, origins all-scopes: I-26; mail only from ancestors: I-57). Correction to the previous section: the or1 case-split work does NOT run in the goal LB; it runs in the parent.

**Trap run 2 — working LB `(in3[9,10,6,5])` (, dump , retarget ).** The boundary hypothesis scope (`main_boundary_product_of_hypo_disintegration_of_integration_goal_(=[6,9])`) and both or1 cohorts run here, and the engine is exonerated end to end:

- The witness facts `(in[W,1])`, `(in2[W,10,3])` ARE in the registry at the `_ordis_` scopes, survive the filter, and the DFS builds and RECORDS the 3-element base `{(in[W,1]), (in2[W,10,3]), (in3[9,10,6,5])}` — `node n=3 subOk=1 tgtOk=1`. Three of rule [96]'s four premises stand ready. The branches even derive real order facts (`(preorder[1,4,6,10])` etc.).
- The fourth premise — the bare product `(in3[9,W,P,5])` — NEVER exists as a statement (0 registry rows; 0 existence-compound rows over 9·W). No stump can complete the key; the merge never sees a 4-element union.
- Where the product went: **248 `overallHashMemory.rejectedMap` entries** of the form `concrete=(in3[9,W,P,5]) | compound=(existence1[1,9,W,5])` — the disintegration-side admission REJECTED the symbolic-product mints and parked them.
- The rejecting gate is `isAllowedAsOperatorInput` (`prover.hpp`): `cfg->inputIndices.size!= 1 → refuse` — the **Pass-B single-input-operator gate, [I-6](../../agentic_swdd/30_invariants.md#i-6) ("do not widen"; OPEN-3 empirical history)**. The predecessor witness W minted legally (`in2` is single-input); every `+`/`*` application over the symbolic witness is multi-input fan-out and is refused BY DESIGN.

**Classification: architecture gap, not a coding bug.** B3's predecessor-branch route needs the product decomposition `9·b = 9·w + 9`, whose ingredient `9·w = P` requires exactly the symbolic multi-input fan-out that I-6 deliberately forbids. The A-ladder never met this (order witnesses come bound inside the preorder/strictOrder definitional compounds); B1/B2 ran by induction, where the recursion triads supply the products. The general request engine (growth caps, subkey indexing, iteration/secondary gates, merge) is fully exonerated by the trap evidence.

**Repair directions (Rule 8, maintainer decision, none executed):**
1. **Pool bypass (unfair-advantage standing rule; recommended):** add the 2-strength B2 twin `a≥2 ∧ b≥1 ⟹ a·b≥2` (provable by induction like B2, no symbolic products needed); B3 then closes via the 0/1/≥2 case split consuming 1–2-premise pool rows. B6 (`b≥1 ⟹ a ≤ a·b`) remains desirable independently as shortlist inventory.
2. Corpus-level: provide `*`(and/or `+`) totality over symbolic arguments as external rows (`∀a,b ∃c: a·b=c`) — the existence compound then arrives as a fired rule head and disintegrates through the existence door, not Pass-B fan-out. Touches the corpus philosophy.
3. Widen I-6 — historically forbidden (empirical reversion, OPEN-3); last resort.

**Temporary instrumentation still in tree (Rule 30 report):** the five `[B3TRAP]` probes in `memory.cpp` — to be removed when the B3 fix lands; the Rule-14 dump retarget at the working LB stays per standing practice.

## 2026-08-06 — repair option 1 (2-strength B2 twin): twin PROVES, B3 rows still open, NEW verifier failure

Maintainer picked repair direction 1.: the twin `a≥2 ∧ b≥1 ⟹ a·b≥2` added as row 33 (`(>[9](preorder[1,4,7,9])(>[10](preorder[1,4,6,10])(>[11](in3[9,10,11,5])(preorder[1,4,7,11]))))`, byte-mirror of B2 with numeral two); the five `[B3TRAP]` probes removed per Rule 30 (`memory.cpp` verified byte-identical to pre-trap state); dump retarget kept.

Run  (35-row pool):

- **The twin PROVES** (33/35 saved; twin present in artifacts). The bypass lemma itself is fine.
- **The B3 rows are STILL unproved.** The twin alone did not unlock the route — a fresh stall trace is needed to see where the 0/1/≥2 route now stops (candidate suspects, unverified: the or5 case split never opening on the typed variable, or `1≤9` unavailable where the twin's second premise needs it).
- **NEW VERIFIER FAILURE — `anchor handling uniqueness`: 3804 checks, 1 failure** (first non-airtight run of the campaign). Offender: `files/shortcut/processed_proof_graph/50_check_induction_condition.txt` — the TWIN's own induction-condition chapter — carries TWO `anchor handling` rows: `(AnchorFTA[N,i0,s,+,*,i1,2_copy,id])` (slot-7 copy only) and `(AnchorFTA[N,i0,s,+,*,1_copy,2_copy,id])` (both numerals copied). The verifier contract (at most one anchor-handling row per chapter) is presumed right per I-16; the producer emitted two copy-variants into one chapter. Multi-numeral theorems are not new (A20 rows use numerals 0/1/2 and verified airtight), so the trigger is something specific to this proof's shape — uninvestigated.
- Runtime 165 s (up from 85 s — the twin's induction plus the still-stalling B3 grind).

Open items for the maintainer: (1) producer-side investigation of the double anchor-handling emission; (2) fresh stall trace for B3-with-twin. Neither started.

## 2026-08-06 — anchor-handling verdict + verifier revision (maintainer-consented); airtight restored

**Investigation of the double emission (dump-driven, two retargets):** `prehandleAnchor`'s copy set is a function of each LB's upward path — one copied-anchor variant per LB, differing along the premise chain (twin: `(…,6,x7,…)` at `(preorder[1,4,7,9])`, `(…,x6,x7,…)` deeper). The maintainer's decisive question — both anchors live in one LB, or originMap duplication? — was answered at the twin's deepest premise LB (, retarget ): the local `(x6,x7)` variant is the ONLY registered statement (all 32 dumps); the mailed partial variant appears solely in `externalStatements` for one burst and is never registered (the axed-variable deposit gate drops it — copies never breed derived x-facts). Chapter 50's second handling row is therefore **cross-LB aggregation via origin history**, not duplicated state; the producer's real guarantee is per-LB uniqueness, which holds everywhere inspected.

**Resolution (I-16 explicit consent, commit, D-263):** the per-chapter `anchor handling uniqueness` counter is DELETED — its founding assumption (one handling event per proof; a second row = walker duplication) no longer matches the per-LB design — and the `anchor handling trace` check now collects copy variables from the union of ALL handling rows' args (the first-row-only collection had silently left `1_copy` usages untraced; the walk itself was already multi-root-correct). Tests: three uniqueness failure tests + the positive removed; two new trace tests (multi-root positive, second-root orphaned-var negative) — Python suite 374/374.

**Verification run ():** 33/35, verifier **3852 checks / 0 failures — airtight**; `anchor handling trace` now 68/0 (was 20/0 — the second variant's usages are under trace and pass). Runtime 168.7 s. Dump retargeted back to the B3 working LB `(in3[9,10,6,5])`; stall trace with the twin in the pool snapshotted as  — the B3 rows remain the open item.

## 2026-08-06 — stall #2 (twin in pool): the twin's rule is installed but permanently premise-starved

Trace walk of  (48 dumps, working LB):

- **The twin's rule IS in the LB's hash memory** (flat originals row `(preorder[1,4,7,9]) (preorder[1,4,6,10]) (in3[9,10,11,5]) (preorder[1,4,7,11])`), alongside B2's, the A-family, and row 27's `=[6,9]`-producer `(in[9,1])!(=[2,9])!(preorder[1,4,7,9]) (=[6,9])`. In-run rebroadcast of proved pool rows works. (An earlier grep claimed the rule absent — pattern error against the flat originals format; corrected here.)
- **But no `2≤x` fact and no `¬(2≤x)` fact ever exists** — zero `(preorder[1,4,7,…])` and zero `!(preorder[1,4,7,…])` statements in the final registry. The twin needs a `2≤9` hypothesis; row 27 needs `¬(2≤9)`. Neither is ever supplied because:
 - the 0/1/≥2 or-compacts (`or4`/`or5`) are **post-run constructions** (`constructOrTheorem` at export) — in-run there is no 0/1/≥2 split to open a `2≤` branch;
 - the only in-run splits are the corpus predecessor cohorts `or1[9]` / `or1[10]` (confirmed again: the only `_ordis_` scopes in 48 dumps), and a nested second split inside a branch is gated off (`max_or_depth: 1`, D-211);
 - nothing demands a reductio on `2≤9`.
- ~~Deeper structural blocker sighted: cross-cohort release deadlock~~ — **WRONG, corrected by the maintainer same session.** The design is: the TRUE branch closes (reaches the parent-chain goal), then the refutable branch starts and fails (refuted, retired), then reduced-cohort convergence promotes. No cross-branch dependency exists in the design.

**The actual defect (trap-grade, 2026-08-06):** the true branches never close because **the LB that owns both cohorts has an empty goal registry for the entire run — `toBeProved=0` in all 48 dumps.** I-174's resolution predicate ("parent-chain `toBeProved` goal reached, or retired refuted") therefore can never fire by goal in this LB; the existence branches are irrefutable; so they never resolve, the refutable `=[2,n]` branches never release (`orPendingBranches (2)` from release to the burst cap; zero `((=[2,…]))` scopes in the whole trace), no refutation, no promotion — frozen by construction, independent of any pool lemma (the twin, B6, or any other bridge changes nothing while the close-step has no goal to close against).

 Contrast A17, where the design works exactly as stated: its predecessor cohort lived in the innermost premise LB whose main `toBeProved` carried the head — true branch reached the goal, closed; refutable branch started, failed; converged. B3's cohorts were opened by the boundary/integration demand in the goal-LESS premise LB `(in3[9,10,6,5])` — the goal `(=[6,9])` lives in the child `(in[10,1])` LB and in the integration machinery's own containers, invisible to the ordis resolution predicate.

**Classification: mechanism gap** — demand-driven ordis cohorts minted in a goal-less LB cannot satisfy the sequenced-release close condition. Candidate repair directions (Rule 8, maintainer's decision): (a) make the resolution predicate see the driving goal (child-LB / integration-goal awareness for boundary cohorts); (b) open such cohorts in the goal-carrying LB, where the A17 pattern already works; (c) an additional close condition for true branches (saturation) — semantically riskiest.

## 2026-08-06 — B3 PROVED: lemma package closes it with zero splits in B3's grid (38/38, airtight)

Maintainer ruled out nested ordis flatly and picked the split-free lemma path. added rows 34–36 (B3a/B3b shift to 37–38):

- 34/35 — unit-factor nonzero twins: `a·b=1 ∧ b∈N ⟹ a≠0` and the `b≠0` mirror (prove in their own grids via the healthy A17 pattern — single-level cohort in the goal-carrying LB);
- 36 — **B6 growth** `b≥1 ⟹ a ≤ a·b` (the factor-bound the B2/twin constant-bounds could not supply).

Run : **38/38 proved — both B3 rows close** (literal rows in `theorems.txt`; B3a exports as `induction v1` — with the lemmas in the pool GL closed it through induction rather than the hand-traced boundary chain, as is its right). Verifier **4134 checks / 0 failures — airtight**; runtime 89.4 s (stall grind gone; was 165–168 s). Trace snapshot .

**Still open after B3 (unchanged by this close):**
1. The **goal-less-cohort mechanism gap** above (a)/(b)/(c) — B3 stopped depending on it; a future theorem whose route genuinely needs a boundary-scope case split will hit it again.
2. The **I-6 fan-out gate finding** from stall #1 (symbolic products in or-branches) — likewise dormant, not fixed.
3. Rule-14 dump remains targeted at the B3 working LB `(in3[9,10,6,5])` (silent once the campaign branch closes); no temporary traps in tree.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
