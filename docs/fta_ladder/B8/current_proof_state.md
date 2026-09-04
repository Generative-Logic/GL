<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
Contributions require CLA — see CONTRIBUTING.md.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# B8 (multiplicative cancellation) — current proof state

Append-only; newest entries last.

## 2026-08-07 — first shortcut run: unproved (stall)

Row 42 (`(>[9](preorder[1,4,6,9])(>[10,11](in3[10,9,11,5])(>[12](in3[12,9,11,5])(=[10,12]))))` under the AnchorFTA prefix — `c ≥ 1 ∧ a·c = b·c ⟹ a = b`, the two products encoded by a shared result slot as in the additive-cancellation corpus row 12). Run : **41/42 theorems** (every pre-existing row still proves — no regression), verifier 4959 checks / 0 failures (exactly the B7 baseline — B8 contributed nothing), runtime 461.3 s.

**Frontier signature: deductive stall.** `total_exprs` grows normally to burst 24 (peak ~64k mid-run, settling to 14716), then freezes at `active_bodies=12`, `total_exprs=14716` from burst 25 through the iteration cap at 42 — eighteen bursts with zero new expressions. Same signature as B3's stall (frozen ~31→42). Not a runtime pathology.

**Expected human route (shortlist):** A15/A16 + B5 — reductio on `a ≠ b`, trichotomy splits into `a < b` / `b < a`, either side gives `a·c < b·c` (resp. `b·c < a·c`) by B5 (pool row 40), contradicting `a·c = b·c` (both products are the same pool variable, so the strict order lands on one object — A21 irreflexivity, pool row 29, refutes it). The route needs an in-run case split on the trichotomy or an equivalent implication chain under the reductio scope; B3's investigation established that or-compacts are post-run constructions and in-run splits have so far come only from corpus predecessor cohorts.

**No diagnosis yet.** The stall playbook (dump retarget at the B8 working LB, one diagnosis run, trace walk) awaits the maintainer's direction (Rule 14 gates the retarget). No temporary traps in tree; the Rule-14 dump still points at the B3 working LB `(in3[9,10,6,5])`.

**Evidence file:** .

## 2026-08-07 — stall investigation (playbook), dump-backed diagnosis: no demand mechanism ever reaches the head

**Setup.** Two dump retargets under maintainer authorization ( working LB `(in3[10,9,11,5])`, goal LB `(in3[12,9,11,5])`); two diagnosis runs, both reproducing the stall byte-for-byte (41/42, 4959/0). Human proof written first (`proof_b8.md`, playbook step 1). Traces:  (working LB, 48 dumps),  (goal LB, 48 dumps); logs , .

**Working-LB findings (parent, `(in3[10,9,11,5])`).** The LB tree is shared between corpus rows: this LB is row 36's (B6) grid — its own goal `(preorder[1,4,10,11])` (`a ≤ P`) PROVES at ~burst 7 (goal 1→0; the fact then lives at main, 352 dump-rows). The head `(=[10,12])` appears ZERO times in the entire 62 MB trace — no goal, no negation, no boundary or integration scope for it (unlike B3, whose head machinery ran in the parent). The route's rules ARE installed in hash memory (flat-original matches: A15 totality 702, A14 2316, B5 1683).

**Goal-LB findings (`(in3[12,9,11,5])`, `toBeProved = (=[10,12])`).**

- The goal is frozen at 1 across all 48 dumps.
- The integration machinery DID engage: `integration_goal_(=[10,12])` hypothesis-disintegration scopes are minted (96 mentions) — but the boundary scope holds **zero statements anywhere in the trace**. The demand machinery opens and derives nothing for an atomic equality head.
- No reductio: `!(=[10,12])` never exists in any form; no `__contradiction__` LB for this head.
- No order case split on `a`/`b`: the only `_ordis_` cohorts are corpus predecessor `or1[n,2,1,3]` compacts over N-witnesses — `int_lev_1_1` (the witness of `1 ≤ c`'s definitional expansion, i.e. `c−1`) and numerals. No trichotomy/totality-shaped cohort exists anywhere.
- No order fact between `a` and `b` in ANY polarity: `(preorder[1,4,10,12])`'s 1270 mentions are all rule-text collisions (rule-internal variable ids); zero statement rows. `(preorder[1,4,12,10])`, both negations, both strictOrders, and `(strictOrder[1,4,11,11])` (`P<P`): all 0.
- No induction: zero `_induction_`/`recN` markers (contrast B7, which closed `induction v2`).
- Memberships `(in[10,1])` / `(in[12,1])` ARE live at main (129/137 dump-rows) — relevant to any totality-consuming repair.

**Classification: architecture gap (B3 family), not a coding bug.** Every route to the head lacks its enabling in-run mechanism: (1) the trichotomy/antisymmetry route needs an order case split over two bound variables, and the only in-run split producers are corpus predecessor definitional compacts (set-membership splits); the pool's A15/A14/A16 implications are negative-premise rules with no producer for their negated premises; (2) the reductio route has no demand — nothing asserts the negated equality head; (3) the induction route (base `a=0` via B1; step via predecessor split + recursion + additive cancellation + IH) is mathematically clean and its ingredients exist, but induction is never scheduled on this head. The general engine is exonerated: rules installed, memberships live, integration scopes minted, cohort machinery healthy on its usual producers.

**Candidate repair directions (Rule 8, maintainer decision, none executed).**

1. **Unfair-advantage package via the A16 De-Morgan door (strongest lead — reuses shipped machinery).** Add totality in flattened-or form as a pool row, `(>[9](in[9,1])(>[10](in[10,1])!(&!(preorder[1,4,9,10])!(preorder[1,4,10,9]))))` — an A19-shaped negated-AND head (A19 proved such a head first-run; the contradiction machinery primes negated heads at load, I-165). Once proved, its instantiation at `(10,12)` is a negated-AND STATEMENT in the goal LB (memberships are live), and the I-179 De-Morgan door turns exactly that shape into an in-run ordis cohort over `{a≤b, b≤a}` — the missing split producer, landing in the GOAL-CARRYING LB where the I-174 close condition can fire (the healthy A17 pattern, avoiding B3's goal-less-cohort gap). Likely needs a companion consumer lemma (order-guarded half-cancellation, `a≤b ∧ c≥1 ∧ a·c=P ∧ b·c=P ⟹ a=b`) so each branch reaches the goal.
2. **Caveat on direction 1's consumer half:** the half-cancellation lemma's witness route (expand `a≤b` to `a+k=b`, distribute to `a·c + k·c = P`, cancel additively, close `k=0` via B1) needs the symbolic product `k·c` — exactly the multi-input fan-out that I-6 refuses (B3 stall #1's dormant finding, 248 rejectedMap parks). Its own grid may therefore need induction (B7-style) or the corpus product-totality row (B3 repair direction 2) instead. Whether GL finds an induction proof for the consumer lemma unaided is an empirical question — one run answers it.
3. **Induction-scheduling widening** (make `findDigitArgs`-driven scheduling reach equality heads of this shape) — machinery change, Rule 8, last resort per the standing lemma-first method.

**Temporary instrumentation report (Rule 30):** none added — both runs used only the sacred dump (chain-match retargets, ). Dump currently on the B8 goal LB.

## 2026-08-07 — contradiction-LB inspection (maintainer question): both twins exist; the reductio twin pins the frontier to ONE missing link

**Correction to the previous section.** "No reductio" overstated the dump's reach: the dump prints the target LB's own containers, never its children, so the goal-LB trace was structurally blind to contradiction children. Code reading (`addTheoremToMemory`) + two further retargets settled it: **both contradiction twins prime for every unproved head** (`try_contradiction` and `try_contradiction_negated_head`, both `true` in ConfigFTA, both ungated in shortcut mode).

- **Disproof twin `__contradiction__(=[10,12])`** (retarget, trace , 48 dumps): fuel = the POSITIVE head + mirror; a contradiction would *refute* the conjecture (incubator heritage — "can only ever disprove"). Since the head is true, it burns 24 bursts / 515 statements / 5836 origins of equality-class rewrites for nothing. Dead weight in this campaign, not a defect.
- **Reductio twin `__contradiction__!(=[10,12])`** (retarget, trace , 48 dumps): fuel = `!(=[10,12])` + mirror — the human contradiction proof's §1 verbatim, machinery present and running. Interior: 526 statements by EXIT #24, memberships and all pool rules live, **and zero order facts between `a` and `b` in any polarity across the whole run** (`(preorder[1,4,10,12])` / `(preorder[1,4,12,10])` / both negations / both strictOrders: 0 statement rows each; `P<P` never; cohorts = the inherited predecessor `or1`s only). A14 is installed but starved of `a≤b`; A15 is installed but starved of `¬(a≤b)` — negative order facts have no producer.

**Sharpened classification.** The reductio scaffold (§1) WORKS. §5 needs no new mechanism either: A22 (pool row 30, `a<b ⟹ ¬(b<a)`) bound at `(P,P)` turns a converged `P<P` into `¬(P<P)` — the I-78 main/main contradiction pair — so no reflexive equality is required. Every step of the contradiction route except one runs on existing machinery. **The single missing link is the totality case split `a≤b ∨ b≤a` (§2): no in-run producer exists for it.**

**Sharpened repair recommendation (one lemma per frontier, unfair-advantage standing rule):** add flattened totality as a pool row —

```text
(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9](in[9,1])(>[10](in[10,1])!(&!(preorder[1,4,9,10])!(preorder[1,4,10,9])))))
```

- **Provable by existing machinery (A19 precedent — negated-AND head proved first-run):** its own reductio twin assumes the double-negation-cancelled conjunction `¬(a≤b) ∧ ¬(b≤a)`; A15 fires on the first conjunct giving `b≤a`; contradiction pair with the second conjunct; discharge.
- **Consumed by existing machinery:** in B8's reductio LB the lemma-rule fires on the live memberships, minting the negated-AND *statement*; the I-179 De-Morgan door converts exactly that shape into an in-run `_ordis_` cohort over `{a≤b, b≤a}` — in a goal-relevant LB (the healthy A17 pattern); branch 1: A14 (`a≠b` is live) → `a<b` → B5 → `P<P`; branch 2 symmetric → same `P<P`; convergence promotes; A22 refutes; I-78 discharges; the head is proved. No half-cancellation consumer lemma needed (the previous section's direction-1 companion and its I-6 caveat are moot on this route — no symbolic `k·c` is ever built).

**Evidence files:** , , , . Dump currently on the reductio LB. Human proof doc extended with the induction route — still viable as plan B (its scheduler question stands), but the one-lemma contradiction route is strictly smaller.

## 2026-08-07 — in-run or repair landed; acceptance crash diagnosed + fixed; frontier moves to cohort-opening demand

**The repair executed (A16 Phase 2 route, not the pool-row lemma).** In-run OR construction + broadcast on the phase-4 barrier, the n-ary or-expansion polarity fix in both twins, and — after the first acceptance run aborted at burst 15 — the `prepareIntegrationCore` one-entity-per-signature dedup. Crash mechanism (trap-proved, ): inside a scope holding an assumed equality (the disproof twin assumes `a=b`), equivalence-class single-representative collapse (D-106) rewrote a totality instance into the degenerate `(or6[u_1,u_4,X,X])`; the per-occurrence template unfold minted divergent `pi_lev_` placeholders for the two textually identical disjuncts and tripped `flattenOrLeaves`' duplicate-identity assert. Fix: one unfold per distinct instantiated signature ([D-270](../../agentic_swdd/40_decisions.md#d-270), [I-186](../../agentic_swdd/30_invariants.md#i-186)).

**Acceptance result (, 483 s).** Clean full run, verifier 4965/0 airtight, **41/42** — row 42 still open. Working tree byte-identical to the committed baseline artifacts (the in-run ors change no export; the post-run export seam already carried or0–or7). All six or theorems construct in-run (or0 warm-up, or3/or4, or5 positivity, or6 totality, or7 trichotomy).

**Frontier sharpened one more step (, dump still on the reductio twin).** The totality PRODUCER gap of the previous section is closed: the broadcast or6 rule fired in the reductio twin and the statement `(or6[1,4,10,12])` — a≤b ∨ b≤a for exactly the head's a,b — is present (176 dump rows; mirror likewise). But **zero `_(or6` cohort scopes exist; every one of the 17468 `_ordis_` scopes is an inherited predecessor `_(or2` cohort.** The case split now parks at the cohort-opening demand gates: a broadcast-installed rule is not a disintegration product (route (b), D-32 signal), and no disjunct product template is an algebra admission key in the reductio twin (route (a)), so the or6 cohort parks in `rejectedMapOrdis` (I-174/I-177/I-178) and the K mutual-exclusion rules alone cannot split on unproven disjuncts.

**Standing recommendation unchanged.** The previous section's one-lemma route (flattened-totality pool row consumed via the I-179 De-Morgan door) was designed around exactly this gap — the door mints the cohort in a goal-carrying LB (the healthy A17 pattern) without relying on broadcast-head demand. It remains the smallest next step; widening the cohort-opening gates for broadcast-derived or statements is the alternative and is an architectural decision (Rule 8). *(Superseded by the next section: the maintainer chose the three-extension design instead.)*

## 2026-08-07 — maintainer-specified repair design: THREE machinery extensions (Parts 1, 2, 3), to be implemented together in a fresh session

**Status when this section was written.** The `flattenOrLeaves` crash is fixed and verified (, acceptance recorded in the previous section): clean run, 4965/0 airtight, **41/42**. Nothing of the design below is implemented. A fresh session starts here: this section carries the complete evidence inventory, the three-part design with every maintainer decision, the rejected alternatives, and the expected closing flow. Dump evidence is from  (final EXIT #24 of the reductio twin `__contradiction__!(=[10,12])`). Scope-variable map used throughout: 9 = c, 10 = a, 11 = P (the shared product), 12 = b.

### Verified state of the reductio twin (final-EXIT dump inventory)

Present as statement rows: the seed `!(=[10,12])` + mirror (2 rows); B5's other three premises — `(preorder[1,4,6,9])` (1 ≤ c), `(in3[10,9,11,5])` (a·c = P), `(in3[12,9,11,5])` (b·c = P) — and the memberships `(in[10,1])` / `(in[12,1])` (1 row each); **both or-compact statements for the exact (a,b) pair: `(or6[1,4,10,12])` totality AND `(or7[1,4,10,12])` trichotomy** (2 rows each — the in-run broadcast delivers). or7's registry form: `!(&!(strictOrder[1,4,9,10])!(=[9,10])!(strictOrder[1,4,10,9]))` = {a<b, a=b, b<a}.

Absent: order facts between a and b in ANY polarity as statements (`strictOrder`/`preorder`, both orders, both signs: 0 statement rows each; every textual mention in the dump is K-rule body or origin record). The algebra `admissionMap` is `(0)` at all 96 dump points across all 24 bursts — hash-rule hunger is invisible to the route-(a) cohort probe.

Consequences pinned by the inventory: (1) the K full-exclusion rules emitted from or6/or7 each need at least one NEGATIVE order fact besides the seed (`!(=[10,12]) AND!(b<a) -> a<b`); negative order facts have no producer, so the K rules are starved forever — flat consumption alone can never split. (2) Both cohorts park: route (b) fails because the or-producing rules came through the status-3 broadcast door (not disintegration products, no D-32 signal), route (a) fails because the admission maps are empty. (3) B5 (installed; rule text present, 81 rows) is missing EXACTLY one premise: `(strictOrder[1,4,10,12])` = a<b, whose arguments are fully bound by B5's other premises. (4) A14/A15 are installed and equally starved (A14 lacks a≤b, A15 lacks its negative premise) — "a rule waiting to shoot" exists in hash memory, but nothing converts that hunger into demand the cohort probe can read.

### Part 1 — single-exclusion emission with reduced-or heads (new disintegration part)

When a k-ary or cohort is flat-consumed (k ≥ 3 only — at k = 2 the single-exclusion IS the existing K rule), emit IN ADDITION to the K full-exclusion rules the k single-exclusion rules: `!D_i -> or(D_rest)` with a genuine (k−1)-ary or-compact head. For or7 this yields `!(=[9,10]) -> or{a<b, b<a}` — a rule that fires on the reductio seed ALONE, the step no K rule can take. The reduced split is STRICT, so A14 is bypassed in branch 1 (a<b feeds B5 directly).

- **Registry constraint:** the reduced (k−1)-ary or operators are PRE-MINTED at the barrier seam (`constructOrTheorem` / `constructOrTheoremsInRun`, I-23 element-list dedup, deterministic order). Registry writers are single-threaded seams (I-137; the D-76 race is why compaction drains at the barrier) — NEVER mint mid-burst inside disintegration.
- **Polarity per I-175:** elements keep each disjunct's true polarity; the `!D_i` premise negates with double-negation cancellation (`negateScratch`, never a blind `!` prefix).
- **Park-first control (maintainer decision, the load-bearing choice):** these heads must NOT open unconditionally. Although the single-exclusion rule is itself a disintegration product, letting its fired or-head ride route (b) would open a case split wherever ANY negative premise exists — uncontrolled scope spawning, exactly what the D-32 demand discipline prevents. The rule family is tagged at emission (precedent: the `ordisOnly` value tag; the `_orint_`/`_ordis_` marker families), and a head fired by a tagged rule parks in `rejectedMapOrdis` like a broadcast head. Opening comes ONLY from Part 2's demand. Consequence: **Parts 1 and 2 are jointly the closing set for B8; neither suffices alone.**
- Growth is bounded: k reduced ors per k-ary pool or; recursion reaches single-elimination chains only (current pool k ≤ 3 → one 2-ary layer). Degenerate reduced ors (equal disjuncts after equi-class collapse) are safe under the landed dedup fix ([D-270](../../agentic_swdd/40_decisions.md#d-270)).
- Verifier surface: `check_disintegration`'s or branch must learn the single-exclusion shape — I-16 consent required. SwDD: `07_or_branching.md` + tag docs per Rule 10.

### Part 2 — compound-demand admission map (new admission part; pure ordis)

A NEW admission map — not the algebra map, not integration, no machinery reuse (maintainer explicitly rejected the reuse suggestion; the algebra map's keys are marked product templates with Pass-B consumers, this map's keys are fully-ground compound premises with a single consumer; only `rejectedMapOrdis` itself is shared). Registration criterion, all three filters maintainer-specified:

1. **Static qualifying-slot analysis at rule install:** premise slot P qualifies iff args(P) ⊆ args(the rule's other premises) — P is fully argument-decided by the rest. B5's strictOrder slot qualifies (a, b bound by the two product premises); A14's preorder slot qualifies (a, b bound by the `!(=[a,b])` premise).
2. **Compound only** — P must be an operator compact with a compiled body. Excludes atomic function-application facts (`in3`, `in2`).
3. **Minimum 4 arguments** (runtime control): excludes `(=[x,y])` (2 args) and `in2[..]` (3 args) — demand for equalities would be minted by half the pool and sweep constantly. `strictOrder` / `preorder` (4 args) pass.

Runtime semantics: when a rule has matched everything except its qualifying slot, the now-GROUND missing premise — polarity carried verbatim (A15-family rules have a negated compound in the slot; matching against parked disjuncts is polarity-faithful, I-175) — is minted as a demand entry. Mints are STAGED during the burst and drained at the post-fixpoint seam like `admissionKeysAlgebra` (I-68) and the integration preps (I-69); never a mid-burst map write (determinism). **On every map update the drain sweeps ALL parked heads in `rejectedMapOrdis`:** a parked or head with a disjunct exactly equal to a demand entry (ground-to-ground) opens its cohort. Opening consumes both the parked head and the demand entry (I-71-style mutual exclusion keeps the map small; re-sweeps are idempotent).

Branch-side semantics (the second half of the idea): inside the opened cohort, the asserted disjunct is admitted as the COMPACT statement only — its existence body does NOT disintegrate, because no interior admission entry demands it. The ∃k witness of `strictOrder` never materializes, so the I-6 fan-out caveat from the first diagnosis is structurally dodged, not merely avoided. B5 matches the compact directly.

Container discipline: per-LB cold container with the full I-89-family treatment (interned ids, deload facet enrollment, wipe behavior, CE-clone behavior). NOT added to the sacred perform-elem dump (Rule 14) unless the maintainer directs it explicitly.

Known redundancy, accepted by the maintainer's criterion as-is: A14 also qualifies, so once the seed matches, demand for `(preorder[1,4,10,12])` opens the parked or6 (weak split) alongside the reduced-or7 strict split. Both close; extra work, not a defect. The I-174 release ranking serializes within a cohort but not across cohorts.

### Part 3 — `X≠X` recognized as a direct contradiction

The gap is real and structural: I-78 requires BOTH opposing statements at the LB's exact main, and the positive twin `(=[X,X])` of a collapsed negated equality is suppressed as trivial (I-8 family — never deposited), so the pair can never form; a literal self-contradictory `!(=[X,X])` sits inert today. Equi-class collapse in contradiction scopes demonstrably produces such degenerate shapes — the `(or6[..,X,X])` crash was one.

Extension, mirroring the existing pair route: a deposited `!(=[X,X])` with byte-identical arguments at main triggers contradiction handling AS IF the opposing pair were found — same LB flag, same discharge-at-most-once-per-step gating on `isActive`, same `burstDeactivates` mirror (I-78 structure), same subtree suppression (I-173). Soundness: GL equality is reflexive by construction (classes seeded with self), so `!(=[X,X])` is false in every intended model.

Surface cost: the contradiction record today cites a pair; the single-fact case cites one statement — the chapter walker and the verifier's `contradiction` + `contradiction trace` checkers must learn the new record shape (I-16 consent), plus I-78/I-165 doc updates. For B8 the primary close does not need it (the A22 route below); its value is generality — in the disproof twin (which burns 24 bursts of dead weight today) collapsed negated equalities become immediate discharges.

### Expected B8 closing flow once Parts 1+2 land

or7 statement (already deposited) disintegrates → K rules + tagged single-exclusion rules → the seed fires `!(=[10,12]) -> or{a<b, b<a}` → reduced or deposits and PARKS → B5's partial match (two products + the 1 ≤ c guard live, strictOrder slot designated) mints demand `(strictOrder[1,4,10,12])` → post-fixpoint drain sweeps `rejectedMapOrdis`, the reduced or's disjunct matches, cohort opens → branch 1 asserts `a<b` as compact only (no existence disintegration), B5 fires → `(strictOrder[1,4,11,11])` = P<P → branch 2 (`b<a`) symmetric, B5 mirror → the SAME P<P → convergence promotes P<P to the twin's main → A22 bound at (P,P) → `!(strictOrder[1,4,11,11])` → I-78 pair at main → discharge proves the negated seed → `(=[10,12])`, row 42 closes. Every link except the two new mechanisms is dump-verified present.

Why the split content cannot come from the existing machinery (for the fresh session's confidence): the K rules need two negatives and only the seed exists; the De-Morgan door (I-179) needs a negated-AND *statement*, which nothing deposits here; the broadcast or heads carry no route-(b) signal and route (a) reads empty maps. The single-exclusion rule is the unique one-negative-premise consumer of the trichotomy, and the compound-demand map is the unique mechanism that makes B5's hunger visible.

### Rejected and superseded alternatives (do not re-derive)

- **Reduced-or heads inheriting route (b)** (open unconditionally as disintegration products) — REJECTED by the maintainer: uncontrolled case-split spawning; park-first + demand-opening chosen instead.
- **Reusing the I-177 `ordisOnly` / algebra-admission machinery for the demand entries** — REJECTED: incompatible key spaces (marked templates vs ground compounds); only the revival *pattern* (key-gain seam → sweep parked heads) recurs, in a dedicated new map.
- **Widening the cohort-opening gates for broadcast-delivered or heads** (grant route (b) to status-3 installs) — set aside in favor of the demand route; would open every broadcast or everywhere.
- **The one-lemma flattened-totality pool row via the De-Morgan door** (previous sections) — SUPERSEDED as the chosen path by this design; remains a valid fallback if the three extensions stall.
- **Induction route for B8** (`proof_b8.md` plan B) — still viable, still carries the open scheduler question; untouched by this design.

**Branch sequencing (maintainer clarification, 2026-08-07 — not an open point):** dead branches are released upon contradiction. A branch that reaches an in-branch contradiction (branch 1: `a<b` → B5 → P<P → A22 → `!(P<P)`) is DEAD and retires through the existing D-242 / I-172 machinery (`ordisMerge`; cohort runs cleaned of dead entries before any convergence count comparison), which is exactly what releases the next ranked branch (I-174). So the sequential release poses no gap: branch 1 dies by contradiction, branch 2 releases and dies the same way, and **when both branches of an ordis are dead, the whole LB is proved to contradict — the contradiction is PROVED** (maintainer wording). For a contradiction twin that IS the discharge: the twin's theorem (the negated seed, `(=[10,12])`) is proved directly by the all-branches-dead outcome. The convergence-promotion + A22-at-main phrasing in the flow above is an equivalent view of the same close; either way it rides implemented machinery.

**Evidence files for the fresh session:**  (clean acceptance),  (dump target = the reductio twin; Rule-14 retarget still active),  (crash mechanism),  (original crash). Prerequisite already landed: the instruction-table dedup fix — degenerate or instances are safe to disintegrate.

## 2026-08-07 — three-extension implementation, Part 1 LANDED

Part 1 (single-exclusion emission with reduced-or heads, park-first) is implemented and
unit-green across three commits (maintainer-approved plan, three open
design points resolved: keyed revival probe for Part 2's opener; registry-ors-only emission
scope; demand-matched starter):

1. **Reduced-or pre-minting** ([D-268](../../agentic_swdd/40_decisions.md#d-268)):
 `findOrMintOrOperator` (the factored I-23 mint site), `flattenRegistryOrLeaves`(+`Scratch`),
 `renumberULeaves`, `preMintReducedOrs` (worklist to single-elimination closure) at three
 idempotent seams — `proveKernel` entry, `constructOrTheoremsInRun` tail, post-compaction
 drain; `compiledOrByElements` read fence; `kMaxReducedOrLeaves` tripwire.
2. **Park-first tag** ([I-184](../../agentic_swdd/30_invariants.md#i-184)):
 `LocalMemoryValue::singleExclusion` (blob byte @20, full codec/blob-view family shifted),
 `isSingleExclusionInstall` shape detector (D-241 re-detection discipline — mail-recovered
 reinstalls re-mark; polarity-faithful via double-negation cancellation), firing-site strip
 `allowOrDisintegration = productOfDisintegration && !singleExclusion` (`allowOrProbe`
 untouched — fired heads flat-consume and PARK).
3. **Emission + verifier** ([D-269](../../agentic_swdd/40_decisions.md#d-269)):
 `consumeOrLeavesCohort` section 1b emits, for registry-or cohorts with k ≥ 3, the k rules
 `!D_i -> or(D_rest)` (head instance args mapped through parent signature positions in
 first-appearance token order; hard assert on a reduced-registry miss); verifier
 `_check_or_single_exclusion` fallback accepts the shape (one premise, reduced or head,
 parent order preserved). De-Morgan cohorts deliberately keep K-rules-only behavior.

Gates: full rebuild + `--unit-tests` green per commit (1391 → 1393 → 1395); Python verifier
suites 47/47 + 66/66. For B8's reductio twin this means: when `(or7[1,4,10,12])` flat-consumes,
the rule `!(=[10,12]) -> or{a<b, b<a}` now exists and fires on the seed alone — the reduced or
deposits and PARKS. Parts 2 (compound-demand map — the opener) and 3 (`X≠X` direct
contradiction) follow on this branch; Parts 1+2 are jointly the closing set.

## 2026-08-09 — Part-1-only takeover; ordisMerge crash ROOT-CAUSED to the writer (equi-filter suppression overridden by the vacuous full-disintegration marker); fix decision pending

**Takeover.** The three Part-1 commits were cherry-picked verbatim
onto the fresh (tree byte-identical to the _3X Part-1 state;
unit tests 1395/1395). Parts 2 and 3 are NOT in this tree. The first shortcut run
() crashed at burst 16 at `prover.hpp::ordisMerge`'s
known-ambient-negation-without-level-row assert — proving the _3X post-mortem's anomaly #2 is
reachable with Part 1 alone.

**Diagnosis (two trap runs, both deterministic reproductions).**

- **Run A** (, identity print at the assert): the anomalous row is
 `(in2[it_0_lev_4_1816,7,3])` known at `main` — the same statement as the _3X incident — with
 flags `local=1 fullyDisintegrated=1 registered=1 known=1` and no `intStatementLevelsMap`
 row. Crash context: branch `main_boundary_ordis_(or3[2,2,1,4,7])_((preorder[1,4,7,2]))`,
 deposit `!(in2[it_0_lev_4_1816,7,3])`, LB chain `(in2[rec0,10,3])` ← `(in3[12,9,11,5])` ←
 `(in3[10,9,11,5])` ← `(preorder[1,4,6,9])` ← anchor — the B8 grid's own recursion block.
- **Run B** (, writer traps filtered on the exact text): the full
 writer chain, identical in both recursion LBs (`(in2[rec0,10,3])` and `(in2[rec1,12,3])`,
 parent `(in3[12,9,11,5])`):
 1. `addStatement` registration upsert — `registered=1`, no levels (normal two-phase order);
 2. `addStatement` **equivalence-filter suppression** (`filterIterations` via
 `anyFilteredOut`) — the level write AND the `known` bit deliberately withheld: the
 statement is a non-canonical iteration-witness variant. NOT the caps — the post-mortem's
 iteration-cap / secondary-variable-cap hypothesis is refuted (neither cap trap fired);
 3. the **full-disintegration marker upsert** (`addExprToMemoryBlock`, status=1,
 `levelsRowPresent=0` at upsert time) — promotes the same row to
 `known=1 + fullyDisintegrated=1`.

**Root cause.** `disintegrateExpr2`'s `fullDisintegrationHappened` is VACUOUSLY TRUE for an
atomic (no existences → "nothing to witness", by-design comment at the computation), so the
marker upsert runs for plain atomic facts and OVERRIDES the admission decision the
equivalence filter just made: a statement excluded from the level registry becomes `known`.
The known-reads (Site F, negation scans, the ordisMerge refutation probe) then see a fact the
level registry rejected, and the probe's `known ⇒ levels` contract assert fires. The _3X
refutation of the marker attribution ("the trapped statement is a witness-instantiated
product, not a disintegrated compact") missed the vacuous-truth path: both are true — it IS a
product, and the marker processed it anyway. The assert is untouched throughout (Rule 19).

**Fix candidates (Rule 8, maintainer decision, none executed).**

1. **Marker sets `known` only when a levels row exists** (contract-enforcing at the convicted
 writer): admitted statements unchanged (addStatement already set `known`); suppressed
 variants stay `registered + fullyDisintegrated`, invisible to known-reads. cFE suppression
 (I-72) keeps its bit. Deltas: unadmitted rows stop feeding negation scans and ordis
 refutations (arguably correct — the filter rejected them); re-arrivals of never-admitted
 expressions re-process instead of Site-F-skipping (work, not semantics); theorem-set
 preservation is empirical — one run answers it.
2. **Marker stops setting `known` entirely** — same contract, broader delta: every
 fully-disintegrated-but-not-self-returned compound loses its Site-F known-skip.
3. **Move the marker's dedup role off the `known` bit** (Site F reads
 `registered`/`fullyDisintegrated` for these rows) — an I-85 bit-semantics restructuring.

**Temporary instrumentation report (Rule 30).** Two traps live in tree, both committed on
: the run-A identity print at the ordisMerge assert (prover.hpp) and the
run-B writer traps (prover.hpp `addStatement` + equi-class commit, prover.cpp marker upsert).
They are removed once the maintainer's chosen fix is implemented and verified; if the fix is
deferred, they stay reported here.

**Part-1 evidence so far:** pre-minting + emission are alive in the takeover runs (the in-run
or construction log shows the shifted numbering, e.g. trichotomy at `(or8[..])`); firing/park
evidence awaits a completing run, blocked on the fix decision above.

## 2026-08-09 — Levels-contract session DONE: crash gone, both acceptance runs green, theorems byte-identical

The dedicated crash-fix session implemented the maintainer-designed one-door contract
(details in `docs/agentic_swdd/40_decisions.md` D-264 +
`30_invariants.md` I-182): row presence in
`intKnownStatements` = known = admitted, every row paired with a NON-EMPTY levels row,
the `{-1}` non-derived tier for all non-derived facts (transparent to level accounting),
the `known`/`registered` bits retired, the full-disintegration marker demoted to
bookkeeping (never creates rows), uniform admission for not-self-returned
fully-disintegrated deposits, and the CE teardown's keep-loop replaced by a proven-no-op
full reset. All Rule-30 traps from the diagnosis runs are REMOVED (including one
transitional `[TRAP-MIXEDRUN]` used to pin a pre-existing raw level merge in the
equi-class rewrite staging that predated the transparency rule).

**Acceptance:** shortcut run COMPLETES (the burst-16 `ordisMerge` crash is structurally
impossible — any probe-visible row carries non-empty levels by construction): verifier
4969 checks / 0 failures, 41/42 pool rows, `files/shortcut/theorems/theorems.txt`
byte-identical to the committed baseline, RT 500.0 s vs the 461-483 s baseline. Full
Peano+Gauss run: verifier 137444 checks / 0 failures, `files/theorems/theorems.txt`
byte-identical (62 rows), RT 1279.9 s (recent band 1251-1284 s). Unit tests 1399/1399.

**Part-1 acceptance and the subset-exclusion generalization resume on top of this
branch.** The 41/42 result above is with Part 1 in tree; B8's row still stalls, so the
Part-2 (compound-demand wake) work remains the closing candidate.

**Crash-fix-session pointer:** the maintainer's directives (every expression gets a levels set; the new check verifies the levels set is NON-EMPTY, never just the key) are recorded in [`../../agentic_swdd/statement_levels_plan.md`](../../agentic_swdd/statement_levels_plan.md); all implementation design belongs to that session. No code from this direction is implemented on this branch; the Part-1 campaign here resumes after that session lands.

## 2026-08-09 — Part-1 acceptance GREEN on top of the levels contract 

**Run 1 of the Part-1 plan** (, dump snapshot
): full rebuild, unit tests 1399/1399, shortcut pipeline
clean end-to-end in 542.8 s — the ordisMerge crash is structurally gone under the one-door
levels contract. Verifier **4969 checks / 0 failures airtight** (= the 4965 baseline + the 4
single-exclusion checks). `files/shortcut/theorems/theorems.txt`: 41 rows, byte-identical to
the committed baseline; the run left the whole working tree byte-clean — zero artifact drift.

**Or numbering this run:** or7 = totality, or8 = trichotomy, or10 = the reduced
`{a<b, b<a}`.

**Firing evidence (reductio twin, sacred dump):** the single-exclusion family is installed
broadly (witness-instantiated `(>[]!(=[x,y])(or10[u_1,u_4,x,y]))` variants); the trichotomy
statement `(or8[1,4,10,12])` is live (83 rows); and **`(or10[1,4,10,12])` — a<b ∨ b<a for
exactly the head's a, b — is a registered statement at main with an `expansion` origin**: the
reductio seed alone fired the step no K rule can take, now reproduced on this branch with
Part 1 only.

**Park state (recorded per scope, not chased):** the final `rejectedMapOrdis`
(overallHashMemory) holds 132 entries, ALL or7 (totality) instances; **or8 and or10 never
park** — the _3X post-mortem's anomaly #1 reproduces with Part 1 alone, so it is not a
Part-2/3 artifact. It remains the open item for the Part-2 session (whose wake design must be
VALUE-side per the post-mortem).

**Next per the approved plan:** the subset-exclusion generalization (j = 1..k−2 negated
disjuncts; order-free install detector — premise order provably does not survive the
mail-compact round-trip), then run 2 as a pure no-regression gate (pool tops at k = 3, so the
generalization is dormant on this corpus).

## 2026-08-09 — subset-exclusion generalization LANDED and GREEN; Part-1 task complete 

Part 1's rule family is generalized from "one negated disjunct" to "j negated disjuncts",
j = 1..k−2 (j = k−1 stays the K family), in two commits on top of the green takeover:

- **G1 — order-free install detector.** `isSubsetExclusionInstall` (renamed; LMV field
 `subsetExclusion`, blob byte @20 unchanged): ONE registry scan per install, matching each
 same-flat-size or entry by order-preserving subsequence embedding of the head leaves plus
 multiset-match of the premise negations under injective token binding. Order-freedom is
 forced by evidence: the mail-compact round-trip stores only the first-seen body and dedups
 name-sorted permutation families, so reinstalled premises arrive permuted — an
 order-sensitive probe would silently untag exactly the mail-recovered rules. Both shape
 detectors also hoisted above `addToHashMemory`'s permutation loop (loop-invariant).
- **G2 — emission + verifier.** `consumeOrLeavesCohort` section 1b emits per excluded index
 subset (lexicographic odometer; the j = 1 family keeps its former order, so k = 3 cohorts
 emit byte-identically); growth 2^k − k − 2 rules per consumption; the pre-mint closure
 needed NO change (it already registers every leaf subset of size ≥ 2).
 `_check_or_subset_exclusion` accepts j premises as a multiset with an order-preserved head
 embedding.

**Run 2 (, 525.9 s): the strict no-regression gate holds** —
41/42, verifier 4969/0 airtight, and the working tree is BYTE-CLEAN after the run (the
generalization is dormant on this corpus: the pool tops at k = 3, where j = 1 is the whole
family). Gates: unit tests 1401 → 1402; Python suites structural 50/50, or 66/66, meta 56/56,
contradiction 21/21.

**Part-1 task verdict (both runs):** takeover green on the levels contract, the reduced or
`(or10[1,4,10,12])` fires from the reductio seed alone and registers at main, the
generalization is in and dormant-by-construction here. Open for the next sessions, unchanged:
or8/or10 never park (the recorded anomaly — Part-2 blocker), Part 2 (compound-demand opener,
VALUE-side wake per the _3X post-mortem), Part 3 (X≠X direct contradiction).

---

## 2026-08-10 — Part 2 IMPLEMENTED as the ordis2 PAIR; machinery proven live end-to-end; B8 still 41/42 on ONE open question (handoff)

**What this session built (maintainer-directed recovery of the _3X Part-2 work, commits K1-K5 pushed, K6 in tree uncommitted).** The compound-demand opener is now a PAIR that completes each other like algebra's admissionMap/rejectedMap: `admissionMapOrdis2` (the renamed _3X compoundDemandMap — demand side, key = packed (ground missing-premise text, validity)) and the NEW `rejectedMapOrdis2` (park side — the same or heads that park in rejectedMapOrdis dual-file here under each ELIGIBLE disjunct's clean ground text). ONE key language, so the wake is a plain key rendezvous — the _3X key-language mismatch is structurally gone. Full machinery parity per the maintainer's directive: codec families, deload band base+63..71, wipe, CE-clone, BOTH equi hooks at BOTH applyEquiClasses seams, two consented dump sections, I-68 staging + drain (`drainAdmissionKeysOrdis2` → `revisitRejectedOrdis2` — the ONE wake seam; algebra seams never cross-probe), route (c) in consumeOrLeavesCohort (probe precedes the park, demand wins the starter, opening consumes the demand keys), the demand FiringRecord as an OWN kind (isMarker stays false everywhere — maintainer directive: no marker anywhere in the ordis2 machinery; burstDeactivates guarded). Commits: K1 b2859dc52 (LMV bit @21), K2 e1ed6312b (admission map), K3 74a6c0914 (park index + wake), K4 193626264 (ordis2KeyEligible + dual filing; regression 41/42 byte-clean, RT 548.8 s), K5 5924e487a (demand runtime). Unit tests 1417/1417 at the K6-in-progress tip.

**Root causes found and FIXED in the K6-in-progress tree (uncommitted, unit-green, pipeline-clean 41/42 airtight 4969/0):**
1. **The `operators` eligibility defect (the big one, inherited from _3X).** The slot filter gated on `ExpressionAnalyzer::operators` — which holds only input-AND-output core operators — so PREDICATES (`preorder`/`strictOrder`, `output_args: []` in ConfigFTA) never qualified: no demand ever installed, no disjunct ever dual-filed. Masked in every _3X and K-series unit test by synthetic `ea.operators.insert(...)` rig lines. FIX: `ordis2KeyEligible` consults the `compiledEntity` non-atomic fence ONLY (any hit in compiledExpressions is a compiled operator definition by construction; in2/in3/equalities have no entry and stay excluded); tests re-pinned WITHOUT the inserts.
2. **The demand flood.** With eligibility fixed, ~48k demands minted (A15/A16-family: two memberships + one negated-order slot, per member pair, per LB) — RT exploded. FIX (maintainer-set): `kOrdis2DemandMinPremises = 4` non-anchor premises (slot included; install keys carry NO anchor, so nonAnchor == keyN) + the I-171 iteration-depth mirror on the demand text (`extractMaxIterationNumber <= maxIterationNumberVariable`). RT back to 621 s; the or2-branch negated-A22 family (~1.6k mints, all park2=0, harmless) is what remains.
3. **Corrected semantics note:** `maxNumberSecondaryVariables` (= 2) counts distinct ITERATION-WITNESS args (`argIteration > -1`, it_/int_ shapes) in a request — NOT u_ rule variables (maintainer caught the agent's error). The twin's three B5 facts are witness-free, so this cap does NOT block the B5 request.

**PROOF the pair works end-to-end in the pipeline (trap dataset  of the final run; run log ; twin dump snapshot ):** in row 40's OWN grid a real chain fired — demand minted, `[O2TRAP-WAKE] key=(strictOrder[1,4,9,10]) v=main lb=(in3[10,11,13,5])`, and route (c) OPENED cohorts (`[O2TRAP-ROUTEC]... hit=1`, e.g. leaf `(strictOrder[1,4,9,10])`). Dual filing live everywhere (4071 filings). The wake/consume/starter semantics all exercised (plus 20 unit tests incl. both arrival orders).

**THE ONE OPEN QUESTION (why B8 is still 41/42).** In the REDUCTIO twin `__contradiction__!(=[10,12])` the needed positive demand `(strictOrder[1,4,10,12])` never mints. The QUAL ground truth (`[O2TRAP-QUAL]` prints every strictOrder slot entering the demand pass with full key + verdict): B5's CLEAN full form `{(strictOrder[u_1,u_4,u_9,u_10]), (preorder[u_1,u_4,u_6,u_11]), (in3[u_9,u_11,u_12,u_5]), (in3[u_10,u_11,u_13,u_5])}` enters the demand pass ZERO times anywhere in the run — only recursion/literal-contaminated derived forms install variants (verdict=1 at keyN=4/5 with `u_rec` tokens or pre-bound result slots `12`/`13`, e.g. `(in3[u_9,u_rec,12,u_5])`), and none of their keys match the twin's three clean facts. The only pipeline mints of the text are `!(strictOrder[1,4,10,12])` in the wrong-direction disproof twin. So the question for the next session: **why does B5's clean full form never reach makeNormalizedKeysForAdmission's demand pass** (candidates: the broadcast-compact reconstruction pre-instantiates result slots before install; the clean-form install runs on a path that bypasses the pass; or the clean form installs only at LBs the QUAL filter didn't catch — note QUAL has no LB field, correlate via `[O2TRAP-UAM]`, which prints keyN/remN/validity/lb/full key for every strictOrder key entering updateAdmissionMap — 49222 lines, 7994 in contradiction twins, unmined). The maintainer's standing instrument rule applies: retarget the hashburst dump at the LB under scrutiny FIRST (the project conventions as amended today — the target is not sacred; the dump currently targets this reductio twin and both new map sections are wired into it).

**Rule-30 trap inventory left in tree (report, not removed — diagnosis handoff):** header helper `gl::o2trapf` (prover.hpp, writes ); `[O2TRAP-ORARM]` + `[O2TRAP-PARK2]` + `[O2TRAP-ROUTEC]` (prover.cpp consumeOrLeavesCohort), `[O2TRAP-WAKE]`/`[O2TRAP-WAKE2]` (prover.cpp revisitRejectedOrdis2), `[O2TRAP-TWIN]` (prover.cpp addTheoremToMemory priming), `[O2TRAP-DEMAND]` (memory.cpp drainAdmissionKeysOrdis2), `[O2TRAP-MINGATE]` + `[O2TRAP-QUAL]` (memory.cpp demand filter/pass), `[O2TRAP-UAM]` (prover.hpp updateAdmissionMap). All file-routed (main run log stays clean, maintainer directive). Remove per Rule 30 once the open question is fixed.

**Also recorded today (the project conventions + AGENTS.md + ):** the hashburst dump TARGET is retargetable at will and MUST follow the LB under investigation; structure/format/deactivation stay Rule-14-gated. Part 3 (X≠X) deferred to a later session by the maintainer (needed for uniqueness-flavored rungs, not for B8's strict route).

## 2026-08-10 (same day, later) — B8 PROVED: 42/42, verifier airtight; the output-collision variable copy closed the open question

**Root cause of the open question (trap-proven from the fix7 dataset — the prior session's "0 QUAL entries" was a grep artifact).** B5's clean form DOES enter the demand pass — 116 QUAL entries, all verdict=1 — but GROUND, in B5's own bound-variable numerals (bare numerals are the theorem binder convention; `extractRemainingArgs` collects only `u_` tokens, so remN=0). Ground pool rules DO fire under injective renamings (the twin's own antisymmetry-row demand `!(strictOrder[1,4,6,9])` proved it in the same dataset). What can never fire is a NON-INJECTIVE binding: B8 encodes a·c = b·c by fusing both product result binders into ONE name (11), while B5 was written with two distinct result variables (12, 13) — matching would need 12→11 AND 13→11, and partition (variable-equalized) copies come only from `multiplyImplication`, which `ConfigFTA` disables (`allow_multiplication: false`, RT). The end-to-end route-(c) demonstration in row 40's own grid was a name coincidence (identity binding), not generality.

**The fix (maintainer-specified, implemented + squash-ready on this branch).** The output-collision variable copy: at theorem compilation (`addTheoremToMemory`), when the disintegrated chain holds EXACTLY TWO positive premises of the same operator with all non-input/output args byte-equal and the same name r in the single output slot (inputs differing), deposit the dead-end free axiom `(=[r,r_copy])` at the FIRST chain LB carrying r — the fifth emission site of the established variable-copy contract (D-235 is the fourth, same function; same `_copy` name shape, `variableCopy` zero-dep origin, paired mailOut write; deposit at the first carrier, not the innermost, because mail flows only ancestor→descendant). Detection is the pure helper `detectOutputCollisionCopyVar`. SwDD: `D-265`, 04_prover.md; verifier surface unchanged (docstring site-count only). Commits 185a723e7 (feature, 10 unit tests) / e16315eca (verifier docstring).

**Second bug found and fixed on the acceptance run — `fillMailOut` violated I-26.** With the copy live, the strict split opens at the premise-chain LB `(in3[12,9,11,5])`; each branch dies on TWO parallel contradictions (the designed r<r vs irreflexivity, plus r≤r_copy vs ¬(r≤r_copy) from B5 firing in both orientations across the copy pair); `ordisMerge` deposits the refutations `!(strictOrder[1,4,10,12])`/`!(strictOrder[1,4,12,10])` AT MAIN with contradiction records citing the BRANCH-scope antecedents. The main-scope record rides mail; the branch-scope antecedents' origin rows did not — `fillMailOut`'s delta loop gated BOTH the statements channel AND the origin copy behind one main-only `continue`, though I-26 documents the origin channel as ALL-SCOPES. Receivers (the reductio twin) crashed `buildStack` on the missing dep at chapter export. Fix: gate split — statements stay main + `allowedForMail`; the origin copy ships every delta row. Commit f2b57abc2.

**Proof route as run (shorter than designed).** The engine closed B8 at the premise chain without needing the reductio: copy axiom → class variants → B5's demand mints `(strictOrder[1,4,10,12])` → wakes the or10/or8 park → split at `(in3[12,9,11,5])` → both branches die (B5 both orientations + A22 + irreflexivity) → `¬(a<b)`, `¬(b<a)` at main → trichotomy closure → `(=[10,12])`. The reductio twin ran in parallel and consumed the mailed refutations; its chapter is what exposed the I-26 bug.

**Final state.** 42/42 (B8's raw row grep-confirmed in `files/shortcut/theorems/theorems.txt`), `All proof graphs verified. 5127 checks, 0 failures — airtight` (was 4969 at 41/42 — growth is the 42nd theorem's chapters), unit gate 1427/1427, RT 615–710 s (band 461–710). ALL Rule-30 traps removed (K6 O2TRAP inventory + this session's OCTRAP orphan-hunt traps + `o2trapf`); final clean production-path run green. Commits: 185a723e7 / e16315eca / f2b57abc2 / 7441a4e68 + the closing docs commit. Part 3 (X≠X single-fact contradiction) remains deferred by the maintainer (uniqueness-flavored rungs). Next: B9 (shortlist row 43) or squash to main per the maintainer's call.

## 2026-08-10 (same day, latest) — FULL-PIPELINE hardening: first-carrier deposit leak found and fixed (second colliding premise), `tempKept` assert → skip gate; full run green

**The incident.** The first full-pipeline run after B8's shortcut close aborted in the Peano batch on `prepareIntegration`'s `tempKept <= 1` assert (admission template `(in3[marker,marker,marker,4])`, three changeable arguments). An anchor guard in `detectOutputCollisionCopyVar` (shared output must not be an anchor argument — the I-24 analogue; the `s(a)=0 ∧ s(b)=0` family shares output 2 = zero) removed the first crash site (`__contradiction__(=[2,8])`) but the crash moved one conjecture over (`__contradiction__(=[2,9])`) with the identical signature.

**Root cause (trap-proven: deposit inventory + breach chain + breach copy-class,  / `run_B8_full4.log`).** The copy deposit landed at the FIRST chain LB carrying the shared name — in the dense Peano pool an anchor-adjacent chain prefix SHARED by foreign conjectures. The cancellation family's deposit `(=[9,9_copy])` at `(in3[7,8,9,5])` mailed into the subtree of the single-`in3` junk conjecture `(>[7,8,9](in3[7,8,9,5])(>[](=[7,8])(=[2,9])))` (a·b=c ∧ a=b ⟹ c=0, false, routine disproof), whose twin's assumed-head/premise merges fused with the copy into the class `{2,7,8,9,9_copy,…}`; the variant fan then produced multi-marker admission templates. The shortcut's 42-row pool is too sparse for this collision — which is why 42/42 stayed green there.

**Two maintainer decisions, both implemented.** (1) The deposit moved to the SECOND colliding premise LB (`detectOutputCollisionCopyVar` returns index j): every genuine consumer — deeper premise LBs, twins under the innermost premise, recursion auxiliaries — sits at or below it; foreign prefix-sharers never see the copy. (2) `prepareIntegration`'s `tempKept <= 1` assert became a GATE: a template with more than one changeable argument is not integrable and is skipped as a defined result (prepared-marker mint precedes the gate — skip memoized; incubator mode unchanged). SwDD: D-265 updated (title, placement rationale, gate paragraph), 04_prover.md paragraph rewritten; tests re-pinned (deposit at second premise, absent at first).

**Verification.** Unit gate 1428/1428 (anchor-guard negative + re-pinned deposit tests included). Shortcut regression: 42/42, B8 row present, verifier 5127/0 airtight — the moved deposit is B8-safe (B8's route sources the copy at `(in3[12,9,11,5])`, where the split opens). Full pipeline (`run_B8_2ndpremise_full.log`): completes end to end in 1328 s, `All proof graphs verified. 137444 checks, 0 failures — airtight`, and BOTH `files/theorems/theorems.txt` (62 rows) and `files/shortcut/theorems/theorems.txt` (42 rows) byte-identical to the committed baseline. All Rule-30 traps removed before the verification runs (deposit inventory, breach chain/copy-class prints, multi-marker install trap).

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
