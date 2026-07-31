<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
Contributions require CLA — see CONTRIBUTING.md.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Rung 2.1 — disprove `{0,1,2} = [0,1]` — current proof state

Append-only. Newest entry last.

## 2026-07-25 — rung created; work begins in a new session

Rung 2.1 is the disproof of the second false cross-pair, split out of
rung 2 at its closure. Human-notation disproof:
[`proof_02_1_es3_not_interval_0_1.md`](./proof_02_1_es3_not_interval_0_1.md).

### Starting state (inherited from the rung-2 closure)

Evidence: the rung-2 proving run  with the
preserved sacred trace ).

1. The IncubatorGauss3 batch settles everything except this conjecture:
 `EnumerationSet2 ⟹ interval[0,1]` proved, the rung-2 theorem
 `EnumerationSet3 ⟹ interval[0,2]` proved, `existence4/5/6` proved, and
 `EnumerationSet2 ⟹ interval[0,2]` disproved outright.
2. The `EnumerationSet3` LB ends its 15 bursts with exactly the two goals
 of this rung: `(interval[1,4,2,6,10])` at `main` and its stuck forward
 clause `(preorder[1,4,repl_lev_1_1,6])` ("p∈M ⇒ p≤1") at
 `main_boundary_(implication21[10,1,4,6])` — correctly unprovable.
3. The reductio machinery the disproof needs exists and is exercised: the
 `try_contradiction_negated_head` contradiction LBs, main-scope
 contradiction discharge, and the successor-addition descent (unblocked
 on rung 2 by the K-implication ungating plus the `_orint_`-scoped
 secondary budget).

### Anticipated first boundary

The §3 descent runs at the contradiction LB's `main` scope, where the
`maxNumberSecondaryVariablesOrint` widening does NOT apply (its scope
condition is a shared `_orint_` branch). If the descent's firings exceed
the standard distinct-secondary cap of 2 there, the widening's scope
condition needs a maintainer-approved extension — the first thing to check
in the sacred trace of the first directed run.

No prover work on this rung has started; the next session begins here.

## 2026-07-27 — first directed run: the stall's root cause is an architecture gap in the hash-request collision-pattern match, not the anticipated secondary cap

Evidence:  (full Windows pipeline, 1113.21 seconds, 133,410 verifier checks, zero
failures, all `files/` artifacts byte-identical to the committed baseline)
with the preserved sacred trace
 — the dump retargeted
(maintainer-directed, Rule 14) at the reductio LB
`__contradiction__(interval[1,4,2,6,10])`, child of
`(EnumerationSet3[2,6,7,10])` under the incubator anchor. The parent-side
cross-checks below read the preserved rung-2 trace
 (the same pipeline shape with
the dump on the parent LB).

### What the reductio LB has, from the first bursts

1. ENTRY #1: the assumed interval fuel `(interval[1,4,2,6,10])` at `main`,
 already disintegrated into all three clauses; in particular the
 elimination rule the disproof's §3.1 needs — `p∈M ⇒ p≤1` as
 `(>[1](in[1,u_10])(preorder[u_1,u_4,1,u_6]))` — is installed from
 burst 1 and waits the entire run.
2. By EXIT #4 the concrete order facts are live at `main`: `0≤1`, `1≤1`,
 `1≤2`, `2≤2`, `0≤2` (`preorder[1,4,2,6]`, `[1,4,6,6]`, `[1,4,6,7]`,
 `[1,4,7,7]`, `[1,4,2,7]`).
3. The descent support operates at this LB's `main`: the
 successor-addition rule and the `¬(s(x)=0)` rules are installed, 80
 derived negated equalities `!(=[2,it_0_lev_2_*])` (w≠0) and predecessor
 disjunctions `or0[1,w,3,2]` appear — the §3.2–§3.4 machinery is ready.

### The frontier — §3.1's membership fact `2∈M` never exists

`(in[7,10])` appears nowhere in the trace at any scope, in any burst; the
same holds at the parent (zero exact-`main` rows across the full rung-2
trace). With no `2∈M`, the waiting `p∈M ⇒ p≤1` clause never produces
`2≤1`, no witnessed sum `2+k=1` exists, and the descent has nothing to
descend. The statement count freezes at ~1831 from EXIT #10; the only
membership fact ever derived is `(in[6,10])` (1∈M), landing at the final
burst (parent EXIT #14, child EXIT #15) via the sibling rung-2 theorem's
intro clause `implication22[1,4,2,7,10]` firing on `0≤1` and `1≤2`.

### Root cause — the normalized-key match is collision-pattern-exact

The [0,1]-intro rule (installed at the child from ENTRY #1) registers the
two-premise request pattern

```text
key={nExpr=2 len=20 data=[16,0,1,0,2,0,3,0,4,0,16,0,1,0,2,0,4,0,5,0]}
```

— all argument slots distinct, the bound variable shared across premises.
An instance whose second premise is the reflexive endpoint fact `1≤1`
(`preorder[1,4,6,6]`) normalizes with a collapsed tail (`…4,0,4,0`): a
different key, so the exact-match probe misses and no request ever
assembles. The same holds for every endpoint instance: `2∈M` via the
[0,2]-intro needs `2≤2`, `0∈M` needs `0≤0` — all collided. Only interior
elements have all-distinct premises, which is exactly why `1∈M` fired the
moment the [0,2]-intro rule arrived, at both LBs, while nothing else ever
did. Sweep confirmation: the entire child trace contains **zero** hash-rule
firings citing a repeated-argument premise (the 363/407 parent rows citing
`2≤2`/`1≤1` are all `equality1` rewrites); `overallHashMemory.admissionMap`
is empty at EXIT #15, placing the block in request generation, not
admission.

This is an architecture gap, not a coding bug: every mechanism follows its
documented contract. GL already names the concept — the positional
collision pattern (I-36) — and the admission side deliberately chose
"single-representative collapse allows repetitions" (D-106); the
request-match side has no analogue, so a rule instance that needs two
template-distinct slots to take one concrete value is structurally
unreachable. Every route to `2∈M` runs through such an instance:
the [0,2]-intro needs `2≤2`; the assumed [0,1]-intro needs `2≤1` (the
goal itself, circular); the ES3 description's intro direction is not
present as a fact rule in either LB (the mail-absorbed ES3 statement does
not re-expand at the child per the absorb-gate contract) and would need
the reflexive disjunct `2=2` anyway — the same collision class.

### Secondary findings

1. The anticipated wall (the distinct-secondary cap at the contradiction
 LB's `main`, outside the `_orint_` widening) was **never reached** —
 untested, stays on the watchlist. The descent here runs on concrete
 numerals with at most two `it_` witnesses per firing, so the standard
 cap of 2 may not even bind.
2. The sibling-theorem route (M=[0,2] intro) reaches `main` only at burst
 14/15 — even where it works it lands at the budget's edge; a direct
 endpoint-membership producer would start the reductio chain in the
 early bursts, where the 15-burst budget comfortably fits the remaining
 five-link chain.

### Candidate repair directions (maintainer's decision, Rule 8 — no code changed)

- **Matching-side (recommended):** at rule install, also register the
 slot-collapsed variants of a rule's request patterns (the request-match
 analogue of D-106's single-representative collapse; bounded by pairs of
 unifiable slots). Generic — every future rung touching an interval
 endpoint hits this same gap.
- **Producer-side** (the shape of the q≠0 campaign): emit endpoint
 membership through a conjecturer template / external lemma whose
 instances are collision-free. Narrower; leaves the generic match gap in
 place.

### Same-day correction (maintainer question) — the matching-side mechanism already exists; it is config-gated off in this batch

The maintainer asked whether the incubator's implication variable
equalization should consume the reflexive fact. It would:
`multiplyImplication` (applied at rule install, `addToHashMemory` Path 1)
partitions all number-typed variables of an implication — bound and `u_`
alike, banning only classes with two distinct `u_` parameters (I-24) —
and installs each partition copy under its own normalized key. The
`{p, u_endpoint}` class yields exactly the collapsed copy
`…(preorder[u_1,u_4,u_7,u_7])(in[u_7,u_10])` whose request pattern
matches `2≤2`. The "no analogue on the request-match side" sentence above
is therefore wrong as architecture: the analogue exists.

It is disabled here: `ConfigIncubatorGauss3.json` sets
`allow_multiplication: false` — a deliberate choice at the FTA batch's
creation (`D-27`, commit, 2026-04-29: the decoupling exists
precisely to run "Pass B on + multiplyImplication off + incubator_mode
on" as explosion containment in the int16 NameMap era), carried verbatim
through the 2026-05-27 anchor split into IncubatorGauss3 —
while every other incubator batch (IncubatorPeano1/2, IncubatorGauss1/2)
runs `true`. Rungs 1 and 2 both closed with the flag off. The CE filter
multiplies regardless of the flag, but this batch sets
`skip_ce_filter: true`.

**Corrected root-cause statement:** endpoint membership is unreachable in
this batch because implication multiplication — the standard incubator
mechanism whose collapsed copies make repeated-argument facts consumable —
is switched off in the batch config. The open decision is now a config
trial (`allow_multiplication: true` for IncubatorGauss3), weighed against
the batch-wide rule fan-out (partition copies for every installed
implication, `max_partition_size` 5) in the batch whose search already ran
away twice during the rung-2 campaign under cap widenings; the
alternative remains a narrower matching-side or producer-side design from
the list above.

## 2026-07-27 — flat `allow_multiplication: true` trial: mechanism confirmed, cost prohibitive, export crash; rung stays open

Maintainer-directed trial (config commit ). Run:
 with the preserved trace
 (1.65 GB,
all 15 bursts). Earlier launch attempts were killed externally
(Windows session teardown, resolved by reboot); the completed run is
attempt 4.

1. **The mechanism works.** The three [0,1]-interval clauses install as
 7 rules at the reductio LB (ENTRY #1) — the original trio plus the
 slot-collapsed intro/elimination copies for both endpoints. Endpoint
 memberships `0∈M` and `1∈M` fire at EXIT #4 from the assumed
 interval's collapsed intro copies; `2∈M` (`in[7,10]`) and its
 consequence `2≤1` (`preorder[1,4,7,6]`) fire at EXIT #15 — `2∈M`
 cannot come from the assumed [0,1] intro (that instance needs `2≤1`
 itself, circular) and waits for the [0,2]-intro collapsed copy, whose
 source fact (the sibling rung-2 theorem) cascades into this LB only
 at bursts 14–15, exactly as in the baseline. (An earlier draft of
 this entry placed "witnessed sums `2+k=1` at `main` from EXIT #5" —
 that was a substring-match artifact: the 132 matching rows are all
 negations `!(in3[7,it_*,6,4])`, true arithmetic rewritten under a
 `{0, it_*}` witness class; the trace contains zero positive witnessed
 sums.)
2. **The reductio reaches exactly the end of §3.1 at the final burst
 and stops.** The chain state at EXIT #15: `2∈M` ✓, `2≤1` ✓ (both
 that burst), witnessed sum `∃k: 2+k=1` ✗ (needs the existence
 closure of the burst after), descent §3.2–§3.4 ✗ untouched, no
 contradiction pair; the LB ends active with 12,298 statements and
 178,106 origins. The rate limiter is no longer the collision match
 (solved) but the sibling-theorem cascade timing plus the 15-burst
 budget — the chain needs roughly 4–6 bursts beyond where the budget
 ends. The multiplied external arithmetic theorems (the
 6-7-number-variable AnchorPeano/u_-form addition-multiplication
 family, 35-56 partition copies each; hashMem 7 → 4,510 at burst 3)
 drown the bursts in rewrite noise meanwhile. Batch runtime grew from
 79 s (rung-2 closure) to tens of minutes (burst dt up to 445 s).
 Note for any repeat: even scoped (disintegration-only) multiplication
 faces the same cascade timing — `2∈M`'s producer is the [0,2]-intro
 collapsed copy, itself a disintegration product of a fact that
 arrives at burst 14–15; closing the rung this way also needs either
 more bursts, faster propagation of the proved sibling theorem, or
 the flag-off `existence4`-style route.
3. **New defect exposed:** the chapter export aborts —
 `visualizer.cpp` `buildStack: no origin found`
 (`gl_quick.exe IncubatorGauss3` exit 0xC0000409) while exporting the
 batch's re-proved theorems (ES2 disproof, rung 1, rung 2 all
 re-proved and printed; the pipeline died before the verifier). Under
 multiplication some proof step's expression lacks its origin record —
 a Rule-16 maintenance gap on a multiplied-rule path, latent until now
 because the FTA batch never ran flag-on through export.
4. The rung-2.1 negation theorem did not emit.

**Open directions after the trial:** (a) scoped multiplication —
disintegration products only (3 → 7 rules carries the entire fix; the
35-56-copy external-theorem fan-out carries essentially the entire cost)
or a number-typed-variable cap (≤ 3); a Rule-8 design either way;
(b) first pin how `ES2 ⟹ ¬interval[0,2]` closes flag-OFF (its reductio
LB has never been dumped; companion `existence4` = the ¬∀ form) — that
route may be open to this rung without any multiplication cost;
(c) the buildStack origin gap needs its own root-cause pass if
multiplication in any scope is to ship.

## 2026-07-27 — the flag-off ES2 disproof route pinned; the real rung-2.1 gap is the missing fact-side OR-intro, not the collision match

Maintainer-directed run: sacred dump retargeted at the PROVED ES2-pair
reductio LB `__contradiction__(interval[1,4,2,7,10])` under
`(EnumerationSet2[2,6,10])`, `allow_multiplication` restored to `false`.
Run  (1191.21 s, 133,410
checks, zero failures — byte-exact baseline), trace preserved as
. The LB discharges at
burst 8; the exported chapter
`files/incubator/processed_proof_graph/1424_direct_proof.txt` records
the whole proof.

### The route (chapter evidence, i0=0, i1=1, i2=2, V1=M)

1. **Witness-alias laundering beats the collision wall.**
 `existence1[N,2,0,+]` (totality) mints the witness `v2` with
 `2+0=v2`; the add-zero identity gives `v2=2`. The equivalence class
 `{2, v2}` then REWRITES the reflexive anchor fact `2≤2`
 (`preorder[N,+,i2,i2]`) into the all-distinct `v2≤2`, and `0≤2` into
 `0≤v2` — `equality1`, no rule firing, no collision. The ASSUMED
 [0,2]-intro clause fires on `(0≤v2, v2≤2)` → `v2∈M` → equality1 back
 → **`2∈M` at main, flag-off**. The collision-pattern limitation is
 therefore NOT absolute: the equality engine launders repeated
 arguments through witness aliases whenever the needed reflexive fact
 is TRUE and an alias class exists.
2. **Enumeration elimination:** ES2's expansion rule
 (`implication1588`: `w∈M ⇒ or1[0,w,1]`) fires on `2∈M` →
 `(or1[i0,i2,i1])` — "2=0 ∨ 2=1" — at main.
3. **K syllogisms (the D-211 rules) detonate it:** with the anchor
 externals `¬(0=2)` and `¬(1=2)`: ¬A ⇒ B gives `2=1`, ¬B ⇒ A gives
 `2=0`; `equality2` merges them to `1=0`.
4. **The kill:** `equality1` rewrites the true `s(0)=1` under `1=0`
 into `s(0)=0`; the axiom family supplies `¬(s(0)=0)`; the pair at
 main discharges the LB and exports the negation theorem (the ¬∀
 `existence4` rides out as its reformulated statement).

### Why this exact route does not transfer to rung 2.1 — and what does

The ES2 case got its membership from the ASSUMED interval's own intro at
the interval's upper endpoint (2≤2 is TRUE there). Rung 2.1's assumed
interval is [0,1]: its intro can only ever produce 0∈M and 1∈M (both
harmless; the instance for 2 needs the false `2≤1`, circular). The
membership `2∈M` must come from the ENUMERATION side — the human proof's
"2 ∈ {0,1,2} directly from the description" — and the maintainer's
diagnosis is exact: **`2∈M` is not a disintegration product of the ES3
description and is not supposed to appear automatically; that is the GL
gap.** Mechanically: ES3's expansion yields the elimination direction
(`p∈M ⇒ p=0∨p=1∨p=2`) and, per D-211, the K mutual-exclusion rules of
the disjunction — but NO fact-side intro. To fire the membership-intro
(`or(w) ⇒ w∈M`, ES3's `implication1589` analogue) at the concrete
element, the disjunction STATEMENT "2=0 ∨ 2=1 ∨ 2=2" must first exist —
its only true disjunct is the reflexive `2=2`, and GL has no mechanism
that produces a disjunction statement from one true disjunct (the
`_orint_` machinery is goal-side only). The witness-alias trick reduces
the need to "v=0 ∨ v=1 ∨ v=2" from the derived TRUE `v=2` (v:= 2+0) —
still blocked on the same missing OR-intro.

**Sharpened rung-2.1 gap (supersedes the collision framing):** the
missing piece is a fact-side OR-INTRO — the dual of D-211's syllogisms:
alongside `¬A ⇒ B` / `¬B ⇒ A`, the expansion of a disjunction-shaped
description could emit `A ⇒ or` / `B ⇒ or` implications (flat hash
rules, no scopes). With it, `v=2` states the ES3 disjunction at `v`,
the ES3 intro fires `v∈M`, equality1 gives `2∈M`, and the reductio's
§3.1–§3.4 (all machinery verified installed) runs at rung-1-era cost —
no multiplication anywhere. A Rule-8 design decision; no code changed.

## 2026-07-27 — OR-intro rules land (D-237); the first run proves the rung through a deposit-scope LEAK the verifier catches; the leak fix holds and the frontier moves to derived-fact existence closure

Maintainer-approved implementation (commit series): the intro emission in `disintegrateExprCore2`'s
implication branch, the additive verifier acceptance in `_try_expand`,
the dump retarget, and the C++/verifier unit tests — see the decision
entry and 07_or_branching §1c.

**Run 1** (, trace
): the reductio LB
discharges at burst 9 and the negation theorem lands in the incubator
pool — but the incubator verification reports `implication failure 1`
(119,024 checks). The failing chapter row (chapter 1426) exposed a REAL
engine bug the new rules made reachable: `checkLocalEncodedMemoryStatic`
computed the firing deposit scope from the PREMISE validities only; a
rule DEEPER than its premises — the new pattern: a branch-scoped false
equality (`2=3`) spawns, via an intro rule, a fully ground or-statement
(`3=1 ∨ 3=0 ∨ 3=2`) at the branch, whose K rules install at the branch
and fire on main-scope truths (`¬(3=1)`, `¬(3=0)`) — deposited its
conclusion (`2=3`) at MAIN, leaking branch-conditioned knowledge
outward. The history record honestly cited the rule's branch scope,
which is exactly how the verifier's D-35 gate caught it.

**The fix** (maintainer-reviewed post hoc; process note recorded — such
prover-semantics changes are to be discussed BEFORE coding): the per-hit
deposit scope is now `deeperOf(premise consensus, rule scope)` (the
documented I-38 contract), hoisted to cover the marker-staging branch
too, with the closed-scope wipe filter re-run when the hit scope is
deeper. Rule-shallower firings (all pre-existing behavior) are
byte-unchanged. No verifier edit — the checker that fired is untouched.

**Run 2, acceptance** (, trace
): verifier fully
airtight — 133,410 checks, zero failures across both graphs — and the
tracked corpus byte-identical (the checker's historical silence implied
no exported proof ever used a leak; confirmed). As the maintainer
predicted, the leak-dependent rung-2.1 proof VANISHES: the reductio LB
runs all 15 bursts and stays active. The clean all-main chain advances
exactly as designed — `copy∈M` → `copy≤1` (EXIT #8) → `2≤1` and `2∈M`
at main (EXIT #8–#10) — and stops at §3.1's last step: **the existence
witness `∃k: 2+k=1` never mints.** No positive witnessed sum ever forms
at main (the only `…=1` sums are true arithmetic).

**New frontier — derived facts get no existence closure.** The witness
mint lives in the existence branch of disintegration, which only
statements entering `disintegrateExpr2` reach. Derived facts arrive
through the deposit/absorb path whose admission gate accepts only
compact `(implication<N>…)` forms (the ASIC-reshuffle status-3 contract;
the rung-2-stop4 fix widened it once, for negated compact existences as
rule carriers). A DERIVED positive existence-category compact — the
`preorder` fact `2≤1` — therefore never disintegrates and never mints
its witness; assumed (status-0) preorders do, which is why rung-2's
branch descents had their sums. The human §3.1 step "so there is k with
2+k=1" has no producer for derived preorders at main. Candidate
directions (maintainer's decision, no code written): (a) widen the
absorb/deposit admission to positive compact existence-category
statements — globally (explosion risk: every derived existence fact
mints witnesses) or scoped to `primedForContradiction` LBs, where
forward existence closure is precisely what a reductio needs; (b) a
different producer for the witnessed sum. The descent machinery beyond
this single step remains verified installed.

## 2026-07-27 — the 2≤1 reductio in IncubatorGauss2 traced: same wall, purest form; one systemic root cause for the whole family

Maintainer-directed run (dump on `__contradiction__(preorder[1,4,7,6])`
under the AnchorIncubator3 LB; no config or source change):
 (1287.05 s, 133,410 checks,
zero failures), trace .
Background verified beforehand: `¬(2≤1)` exists in no pool (the negated
preorder family covers endpoint 0 only); positive injectivity is the
`NaturalNumbers` AXIOM `implication5[N,s]` (maintainer's correction —
the descent step `s(1+l)=1=s(0) ⇒ 1+l=0` is injectivity, not the
campaign contrapositive, and needs no pool theorem); every incubator
batch runs `try_contradiction: true` and `skip_ce_filter: true`, so no
permission gate is involved.

The trace (12 bursts, LB never discharges, 1,740 statements):

1. The conjecture IS emitted in IncubatorGauss2 and its reductio LB
 exists. The status-0 seed disintegration works perfectly: at
 ENTRY #1 the LB already holds `2+l=1` (`in3[7,int_lev_1_1,6,4]`),
 `l∈N`, and later `¬(s(l)=0)` — the witness mints, as predicted for
 assumed fuel.
2. From burst 2 the totality family sits at `main` as DERIVED compact
 existence statements: `existence1[1,6,int_lev_1_1,4]` ("∃w: 1+l=w")
 and siblings for 0/2/3 and both operators.
3. They are INERT for ten bursts: the trace contains ZERO it_-witnessed
 sums at `main` (the 63 `1+x=y` rows are all concrete numerals). The
 intermediate sum `1+l=w` never exists, so successor-addition can
 never fire, so the descent freezes one step after the seed.

**One systemic wall now explains the entire observed landscape:**
derived compact existence-category statements (preorder facts, totality
products) never disintegrate — witness minting is exclusive to
status-0 fuel. Hence: the ≤0 negation family proved (its reductio needs
only the SEED's witness — one step to the `s≠0` axiom); every
endpoint-≥1 negation unsettled in Gauss1/2 (their descents need derived
intermediate sums); and rung 2.1's direct close blocked (the ES3
reductio's derived `2≤1` never expands, and the would-be supplier
`¬(2≤1)` is blocked by the same wall in Gauss2). `¬(2≤1)` is not proved
by the incubator; the investigation stops here.

## 2026-07-27 — source-audit correction: two distinct derived-existence barriers, not one universal no-disintegration rule

The preceding “derived compact existences never disintegrate” conclusion
is too broad. The current source and the same direct-reductio trace split
the frontier into two mechanisms:

1. A positive compact existence arriving from another LB with status 3
 remains opaque by design. Only compact implications and negated compact
 existences are status-3 rule carriers. Equality-class rewrites also
 insert their rewritten facts directly into the statement registries,
 rather than re-entering `addExprToMemoryBlock`; an equality-derived
 compact existence therefore has no automatic expansion entrance.
2. A compact totality result derived locally by hash firing is absorbed
 with status 1 and does reach `disintegrateExpr2`. In the direct
 `2≤1` reductio, `existence1[1,6,int_lev_1_1,4]` creates expansion
 records for both proposed witness families, but Pass B admits neither
 witness body. In particular the required positive
 `1 + int_lev_1_1 = w` statement never enters `encodedStatements`.

Thus the direct proof of `¬(2≤1)` is presently blocked at the second
barrier: GL has proved the totality existence needed for the intermediate
sum, but its output-position witness has no consumer-side admission key.
Globally opening positive status-3 existences would not solve this direct
reductio and would expose the old totality-closure explosion. The
architecture question is instead how to demand and revive exactly the
proved functional output whose body would complete a live rule firing.

## 2026-07-27 — maintainer strategy: the rung closes through the general antisymmetry theorem; theorem-level work only, no prover extension

Maintainer decision (session dialogue, this date). Both prover-extension
proposals raised during the investigation — the reductio-scoped
existence closure and any status-3 gate widening — are REJECTED. The
scope is the general antisymmetry theorem

```text
a≤b ∧ b≤a ⇒ a=b
```

proved through the machinery as it stands, with `¬(2≤1)` and the rung
falling out as consumers of the proved theorem.

### Inventory (artifact-verified this session)

1. **Zero-sum lemma — PROVED, Peano main.** Byte-exact in
 `files/theorems/theorems.txt`:

 ```text
   (>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8](in3[7,8,2,4])(>[](in[8,1])(=[2,7]))))
   ```

 — ∀k,l: k+l=0 ∧ l∈N ⇒ k=0. Pointed out by the maintainer.
2. **Relational associativity — PROVED, Peano main** (`theorems.txt`,
 the four-`in3` row: `7+8=9 ∧ 8+10=11 ∧ 12+10=7 ⇒ 11+12=9`; a
 multiplicative twin exists).
3. **Cancellation — MISSING everywhere.** No pool contains the
 universal form; the incubator holds only the numeral-ground family
 `!(in3[a,n,a,4])` (117 rows across the table numerals), which can
 never match witness names. The maintainer's GL phrasing of the
 missing lemma: `a+n=b ∧ b=a ⇒ n=0`.
4. **The intermediate witness sum `k+l` is covered by the
 marked-expression contract.** Per the maintainer: a term needed to
 close a rule is permitted to mint — consumer-side admission. In the
 antisymmetry LB both preorder premises are ASSUMED (status-0) fuel,
 so the witnesses k, l mint exactly as today, and associativity —
 installed with two of its three premises live — is the live rule
 firing whose demand covers the `k+l` output slot. This is the
 demanded-output case the preceding entry's closing question names,
 here supplied by an installed rule, not by a new mechanism.

### The proof chain

`a+k=b`, `b+l=a` (assumed witnesses) → `k+l=m` (demand-minted) →
associativity → `m+b=b` → cancellation → `m=0` → the class rewrites
`k+l=m` to `k+l=0` → zero-sum lemma → `k=0` → `a+k=b` rewrites to
`a+0=b` → axiom `a+0=a` → `a=b`.

Cancellation's own induction proof rests on the same mechanism: in the
step case `n+s(a)=s(a)` is assumed fuel, successor-addition's need
demand-mints `n+a=w`, injectivity gives `w=a`, and the induction
hypothesis fires.

### Rung-2.1 closure

`preorder` is a Gauss-side operator, so antisymmetry lands in the Gauss
batch, with cancellation flowing in as a Peano external. The ES3
reductio already holds both `2≤1` (run 2 of the OR-intro entry,
EXIT #8–10) and `1≤2` (EXIT #4) at `main`: antisymmetry fires on the
pair (its two slots bind 2 and 1 — distinct values, no collision
pattern), yields `2=1`; the anchor external `¬(1=2)` completes the
contradiction pair at `main` and the LB discharges. No witness minting
inside the reductio at all.

### Next steps

1. Check the conjecturer's emission space for the two conjectures
 (cancellation at the Peano level; the two-preorder-premise
 equality-head shape at the Gauss level); template if absent — the
 pattern of the rung-2 q≠0 conjecturer-template campaign.
2. One directed run per theorem; read the trace only on a stall.

## 2026-07-27 — conjecturer emission opened for both lemmas (Peano main, maintainer-directed); cancellation PROVED

Maintainer decision (session dialogue, this date): both lemmas are
formulated in the MAIN PEANO batch, not split Peano/Gauss as the
previous entry sketched — `preorder` joins `ConfigPeano.json` (its
definition needs only `in`, `in3`, `N`, `+`, all present under
`AnchorPeano`), limited to co-occurring with `=` and itself.

**Config extension (`ConfigPeano.json`, no Gauss change).** New
`preorder` entry (count cap 2, leaf caps 3/3); `=` count cap raised
1 → 2; `preorder` barred from heads and from co-occurring with
`in`/`in2`/`in3`; pattern pins bar numeral element arguments and the
`*` slot; two-positive-equality chains are confined to the
cancellation shape (variable-variable premise, canonical ascending
`(=[2,…])` head when equality-headed). Full rationale and the
pattern-performance idiom: `D-246` in
the SwDD decision log.

**Companion conjecturer change (maintainer-approved).**
`evaluateOperatorExprs2`'s end-operator dichotomy block asserted every
relation argument is classifiable from operator/property/anchor
membership — impossible to violate at `=` cap 1, violated by
two-equality candidates sharing an argument between the equalities.
The block now classifies such an argument as neither-input-nor-output
and rejects the candidate (assert removed under explicit consent;
trap-verified on 8 offending candidates, all of the
`{=,=,in2,in3}` aliasing class; neither target shape reaches the
block).

**Measured emission.** Peano pool 937 → 1001 rows, zero baseline rows
lost. The +64: the antisymmetry family (4 rows, target byte-exact:
`(>[7,8](preorder[1,4,7,8])(>[](preorder[1,4,8,7])(=[7,8])))` under
the AnchorPeano binder) and the cancellation family (60 rows across
`+`/`*` slots and negated-premise variants). Conjecturer wall time
47 s (first-draft exclusion regexes cost 200 s; rewritten to the
linear-scan idiom — see the `patterns_to_exclude` performance contract
in SwDD `04_configs.md`).

**Full-pipeline run (this date).** Verifier airtight: 134,051 checks,
0 failures. `theorems.txt` 48 → 51 rows, all prior rows intact.

1. **Cancellation PROVED, Peano main** — byte-exact:
 ```text
   (>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,4])(>[](=[8,9])(=[2,7]))))
   ```
 — `a+n=b ∧ n=b ⇒ a=0`, exactly the `m+b=b ⇒ m=0` step the proof
 chain consumes. (The mirror orientation `a+n=b ∧ a=b ⇒ n=0` emitted
 but did not close; the proved twin covers the chain.) Two side
 theorems from the new family also proved (a multiplicative analog
 and a negated variant).
2. **Antisymmetry EMITTED, not yet proved.** Its proof needs the
 cancellation lemma in the loaded theorem context; this run proved
 cancellation only within the same batch. Expected to close on the
 next run, when the freshly proved lemma loads as a prior theorem.

### Next steps

1. One further run with cancellation in the proved-theorem context;
 antisymmetry expected to close. Read the trace only on a stall.
2. On antisymmetry closing: the `¬(2≤1)` consumer and the rung-2.1
 negation theorem per the previous entry's chain.

## 2026-07-27 — correction (broadcast) + determinism run

**Correction to the previous entry's item 2.** "Expected to close on the
next run, when the freshly proved lemma loads as a prior theorem" was
wrong reasoning (maintainer correction, session dialogue): a theorem
proved mid-batch is broadcast to all LBs within the same run, so the
cancellation lemma WAS available to the antisymmetry LB in run 1.
Antisymmetry's non-closure is therefore a stall, not a load-context
sequencing artifact. Candidate factors for the investigation:
broadcast timing versus the antisymmetry LB's remaining iteration
budget (all three chain lemmas — relational associativity, zero-sum,
cancellation — were same-run proofs); the witness minting / demand-mint
steps upstream of the lemma; and whether the proved twin's rule form
(`in3[7,8,9,4] ∧ (=[8,9]) → (=[2,7])`) fires on the LB's actual encoded
fact shape for `m+b=b`.

**Determinism run (this date).** Second full pipeline run, same binary,
clean state: `theorems.txt` byte-identical (51 rows),
`global_theorem_list.txt` byte-identical, verifier identical at
134,051 checks / 0 failures.

### Next steps

1. Stall investigation on the antisymmetry LB per the standing
 playbook (chain-match from the LB-creation site, dump retarget on
 maintainer direction, one run, frontier trace against the proof
 chain; diagnosis only).

## 2026-07-28 — antisymmetry stall traced; the theorem now has its own proof-pair documents

The antisymmetry theorem received the standard FTA proof-pair in this
folder: the human-notation proof
[`proof_02_2_antisymmetry.md`](./proof_02_2_antisymmetry.md) and the
append-only GL state
[`current_proof_state_antisymmetry.md`](./current_proof_state_antisymmetry.md),
where the full investigation record now lives. The sacred dump was
retargeted (maintainer-directed) at the antisymmetry LB — chain
root → `(AnchorPeano[1,2,3,4,5,6])` → `(preorder[1,4,7,8])` →
`(preorder[1,4,8,7])` — and one directed run traced
(, trace
; verifier airtight,
corpus unchanged).

**Result in one paragraph:** the proof works through §3.1 (both
premise witnesses mint) and derives §3.2's totality existence
"∃w: k+l=w" by burst 2, whose witness body is parked in `rejectedMap`
awaiting a demand key — which can never be written: the demanding
associativity marker's completion rides the collision-pattern-exact
normalized-key match, and the fact pair `a+k=b` / `b+l=a` double-links
every additive marker key (two template-distinct slots must bind one
name). The LB reaches a true fixpoint with 19 idle bursts left; all
three chain lemmas verified installed. Classified as the SAME
architecture gap as this rung's first entry, with new reach: the
collision wall gates demand generation too, and the marker
qualification gate (`baselineClassicQualifies`, `minNumOperatorsKey`)
independently excludes the small equality-headed rules that could open
the ES2-style alias route. Candidate directions (collapse-only
install-time key variants; config trials; producer template) are
listed in the antisymmetry state document. Diagnosis only — no code
changed.

## 2026-07-28 — antisymmetry PROVED via the variable-copy trigger; rung closure is next

The maintainer-designed fix — a `(=[x,x_copy])` dead-end axiom
deposited at the innermost premise LB whenever a conjecture has the
antisymmetry shape (`D-235`, commit
) — closed the theorem on the first acceptance run: the LB
discharges at burst 15 and the byte-exact target row lands in
`files/theorems/theorems.txt`. The corpus grows 51 → 71 rows,
including a wave of Gauss interval/limit-set theorems that cite
antisymmetry directly. Full record, trace milestones, and two open
maintainer items (a chapter-167 `contradiction trace` failure from a
premises-inconsistent new theorem whose reductio never used its seed,
and one lost baseline Gauss `existence8`/`sequence` row) in
[`current_proof_state_antisymmetry.md`](./current_proof_state_antisymmetry.md).

Next per the standing chain: the `¬(2≤1)` consumer and the rung-2.1
negation theorem — antisymmetry fires on the reductio's derived `1≤2`
and `2≤1` at `main`, yielding `2=1` against the anchor external
`¬(1=2)`.

## 2026-07-28 — closure run traced: antisymmetry can NEVER fire on (2,1); the element argument collides with an anchor SLOT — the collision wall's third form

Maintainer-directed stall investigation (standing playbook). The sacred
dump was retargeted at this rung's ES3 reductio LB
`__contradiction__(interval[1,4,2,6,10])` →
`(EnumerationSet3[2,6,7,10])` → `(AnchorIncubator3[1,2,3,4,5,6,7,8,9])`
→ root, one full pipeline run:
 (1523.72 s, exit 0, verifier
airtight 138,802 checks / 0 failures, `theorems.txt` unchanged at 68
rows — the negation theorem did NOT emit), trace preserved as
 (30 dumps, all 15
bursts, LB ends ACTIVE at 18,790 statements,
`contradictionIndex=-1`).

### Everything upstream of the kill works

1. `1≤2` (`preorder[1,4,6,7]`) at `main` from the early bursts (anchor
 external); `2≤1` (`preorder[1,4,7,6]`) at `main` from EXIT #8 — via
 the designed copy-alias `equality1` chain AND independently by
 `or convergence` over all three `ordis` branches; `2∈M`
 (`in[7,10]`) at `main` at burst 10 (the [0,1]-intro fired on
 `0≤2` + `2≤1` — proving `2≤1`@main is request-visible); the
 contradiction partners `!(=[6,7])` / `!(=[7,6])` live at `main`.
2. The antisymmetry theorem is installed at this LB from burst 3 and
 FIRES repeatedly — every observed instance pairs (2,3), (2,w),
 (3,w), or (w,w′): `(preorder[1,4,7,9])`+`(preorder[1,4,9,7])` etc.
3. The single missing link: `(=[6,7])`/`(=[7,6])` at `main` never
 exists (it appears only as `ordis` branch ASSUMPTIONS), so the
 contradiction pair never completes. Zero antisymmetry firings on
 the (1,2) instance — at `main` AND at every branch scope, although
 branch copies of both premises exist. Scope-independent absence.

### Root cause — element-argument-vs-anchor-slot collision in the request key

The registry row (EXIT #15 `overallHashMemory.originals`, verbatim):

```text
(AnchorPeano[1,2,3,4,5,6]) (preorder[1,4,7,8]) (preorder[1,4,8,7]) (=[7,8])
```

The external theorem installs in ANCHOR-PREMISE form: the
`AnchorPeano` statement is a key constituent and the anchor slots
1–6 are positional variables of the same normalized key as the
element variables 7, 8. The exact positional-collision match
(I-36 concept) therefore requires an instance's two elements to be
distinct from each other AND from all six anchor values. The rung's
decisive instance binds b:= numeral 1 = id 6 = AnchorPeano's slot 6
(the anchor's "one") — a collision the registered key does not have,
so the normalized-key probe misses at every stage (the two-element
base `{AnchorPeano, 1≤2}` already fails the minus-one map), in every
burst, at every scope. Numerals 2 and 3 (ids 7, 9) are OUTSIDE
AnchorPeano's slots, which is exactly why every observed firing pair
avoids ids 1–6 and why the (1,2) instance alone is structurally
unreachable. The 2026-07-27 closure sketch's "its two slots bind 2
and 1 — distinct values, no collision pattern" checked slot-vs-slot
within the preorders and missed the anchor premise sharing the key.

Contrast: locally expanded rules install in u_-form
(`(preorder[u_1,u_4,u_2,1]) (preorder[u_1,u_4,1,u_6]) → (in[1,u_10])`)
— anchor values are u_ LITERALS matched by the D-120 owner prune, not
positional variables, which is why the intro/K rules fire at anchor
numerals freely. Both forms coexist in this LB; only external
theorems carry the anchor premise.

**Classification: architecture gap** — the third form of the
collision-pattern family (first: premise-slot repetition, rung-2.1
first entry; second: demand-side marker keys, antisymmetry stall;
now: element-vs-anchor-slot on anchor-premise-form externals). Every
mechanism follows its documented contract; nothing miscomputes. No
K-syllogism fallback exists for ES3: its disjunction "2=0 ∨ 2=1 ∨
2=2" contains the true disjunct 2=2, so mutual exclusion cannot force
`2=1` — antisymmetry is the only producer, as designed.

### Secondary observations

1. The distinct-secondary cap was again never reached (the (1,2)
 instance carries zero `it_` names).
2. Real silent caps exist in the late bursts (the grow-universe
 filter buffer 8192 vs 13–18k statements from burst ~11; the
 batch-3 local filter 4096) — truncation is live there but is NOT
 this stall's cause (the key can never match regardless).

### Candidate repair directions (maintainer's decision, Rule 8 — no code changed)

- **Matching-side collapse variants, sharpened:** the standing
 collapse-only install-time direction must include ELEMENT-variable
 × ANCHOR-SLOT fusions for anchor-premise-form rules — fusing b with
 slot 6 yields exactly the key that matches (1≤2, 2≤1). Bounded:
 element-vars × anchor-slots plus element-pair fusions.
- **u_-form installation for cross-anchor externals:** install
 external theorems the way local expansions install (anchor slots as
 u_ literals, no anchor statement premise). Removes this entire
 collision class at the root; a Rule-8 semantics design.
- **Config:** `allow_multiplication` would generate the needed
 partition copies (an element-var + anchor-slot class is a legal
 partition) but is measured prohibitive flat-on in this batch and
 still carries the open buildStack origin defect.
- **Witness-alias laundering** needs an alias class of numeral 1 at
 `main`; its natural producers are parked in `rejectedMap` with no
 demander (the marker-qualification gate excludes small
 equality-headed rules) — the antisymmetry §3.2 wall again.

## 2026-07-28 — maintainer proposal analyzed: the copied-anchor (prehandleAnchor) mechanism is exactly the right tool and a live x-AnchorPeano WOULD fire antisymmetry; the missing piece is the axed-variable filter killing the bridge-derived copy, not parent-to-child inheritance

Maintainer proposal (session dialogue): "for cases like this we have
prepareAnchor; the contradiction LB should inherit such a copied
anchor from its direct parent" plus "check if it would fire
AnchorPeano with similar copies." Analysis only, no code (Rule 8).
Evidence: the closure-run trace
, the preserved
parent-side trace , the
bare-preorder reductio trace
, and
`prover.cpp::prehandleAnchor` / `addExprToMemoryBlock`.

### What the traces show

1. **The contra LB already holds the copied BATCH anchor, live and
 firing.** `(AnchorIncubator3[1,x2,3,4,5,x6,x7,8,9])` is statement
 [6] at ENTRY #1 (deposited locally by `prehandleAnchor` — the
 pre-pass covers dynamically created contradiction LBs), and 5,672
 firings cite it as their anchor premise — including the
 successor-injectivity externals binding ELEMENTS at anchor
 numerals 0, 1, 2 (`in2[7,2,3]`, `in2[it,6,3]`, …). The x-copy
 mechanism demonstrably beats the element-vs-anchor-slot collision
 for batch-anchor-premise rules at this very LB.
2. **Neither the contra LB nor its parent holds a live copied
 EXTERNAL anchor.** `(AnchorPeano[1,x2,3,4,5,x6])` appears at BOTH
 LBs only as a minted name plus an `exprOriginMap` history row —
 never in any statement registry, in any burst. Literal
 parent-to-child inheritance therefore adds nothing here: the
 parent has the same x-batch-anchor the child already has, and no
 live x-AnchorPeano to bequeath.
3. **The copy IS already derived — its deposit is filtered.** The
 history row's producer is a real firing of the bridge rule
 `AnchorIncubator3[1..9] ⟹ AnchorPeano[1,2,3,4,5,6]` on the
 x-copied incubator anchor. The conclusion dies at
 `addExprToMemoryBlock` entry: the axed-variable filter (the
 `intAxedVariables.contains(argFullId)` early return) drops every
 deposit carrying an x-name in any argument slot. Only the history
 line survives (the same only-the-history-travels pattern
 `prehandleAnchor` documents for recursion LBs).
4. **The plain AnchorPeano premise structurally forbids anchor-value
 elements.** Across all 5,807 AnchorPeano-premise firings in the
 trace, the cited preorder premises use only numerals 2, 3, and
 `it_` witnesses — never 0 or 1. The x-anchor enables exactly what
 the raw anchor forbids.

### Would a live x-AnchorPeano fire antisymmetry? Yes — exact key match

Instance {`AnchorPeano[1,x2,3,4,5,x6]`, `preorder[1,4,6,7]`,
`preorder[1,4,7,6]`}, name-sorted anchor-first, normalizes to
AnchorPeano(1,2,3,4,5,6), preorder(1,4,7,8), preorder(1,4,8,7) — with
x6 occupying the sixth normal, raw id 6 (numeral 1) is FRESH and
takes normal 7 — byte-identical to the registered key. Gates: zero
secondaries (x-names carry no iteration marker, same as the 5,672
precedent firings), no u_ literals, all-main comparability, key
length 3, Site F does not dedup (different string from the raw
anchor), dependency skip passes once the copy is registered known.
The head binds only the element variables: `(=[6,7])` — x-free, so
the axed filter passes the CONCLUSION — deposits at `main` against
the live `!(=[6,7])`, completing the contradiction pair at exact
`main` (I-78): discharge, negation theorem exports.

### Where the inheritance idea does have content — the x-set is chain-derived

`prehandleAnchor` picks which slots to copy from the TRACE VARIABLES
of the LB's own premise chain. The bare `¬(2≤1)` reductio directly
under the anchor got only {x6, x7} — no x2, because "0" never occurs
in its chain — while this rung's deeper contra LB got the full
{x2, x6, x7} through the parent's premise variables. A contra LB's
semantic premise set is its parent's plus the seed, so "contra LB
inherits the parent's copied anchor (x-set = parent's ∪ seed
variables)" is the right CONTRACT; today it holds only incidentally.
Adopting it would fix the narrower-x-set sibling case but does not by
itself produce the x-AnchorPeano this rung needs.

### Candidate implementations (maintainer's decision, Rule 8 — no code changed)

- **(a) Anchor-category exception in the axed filter** (surgical):
 let a conclusion through when its operator is an anchor predicate —
 equivalently, let the already-firing bridge rule deposit its
 conclusion. Everything upstream already runs; bounded at one
 statement per external anchor per LB; the origin story is the
 honest implication row already being exported today, so no verifier
 change. Self-maintaining for every future external anchor bridged
 the same way (the `divides` rungs).
- **(b) Extend `prehandleAnchor` to external anchors:** deposit
 x-forms of external-anchor statements in the pre-pass. Requires the
 external-anchor inventory at pre-pass time — note the AnchorPeano
 statement itself is bridge-DERIVED at runtime, so the pre-pass
 would synthesize a statement whose base form arrives only later.
 More reach, same result.
- **(c) The contra-LB x-set inheritance contract** (the literal
 proposal): adopt as the specification for which slots a contra LB
 copies; fixes the sibling gap, orthogonal to the external-anchor
 gap.

## 2026-07-29 — axed-exception acceptance run KILLED (runtime explosion); maintainer proposal analyzed: or-intro subimplications should fire with doNotDisintegrate — sound, and it maps onto the established LMV-bit pattern

The `D-234` acceptance run was killed on
maintainer order after a runtime explosion
(, task stopped, no artifacts
evaluated). Maintainer proposal (session dialogue): the or-INTRO
subimplications that `disintegrateExprCore2` emits for a
disjunction-shaped implication premise (leaf `A → or`, leaf `B → or`;
`D-237`) should fire with
`doNotDisintegrate == true` — a head they fire (the `or<N>` statement)
is an integration instruction and must not re-disintegrate. Analysis
only, no code (Rule 8).

### Today's behavior (closure-trace + source evidence)

1. The intro rules install through the shared `collected.insertImpl` →
 `addToHashMemory` drain with the literal justification
 "implication" — nothing distinguishes them in the rule registry.
2. At a firing, `doNotDisintegrate` resolves FALSE (justification is
 not `integration`; the D-29 premise-locality clause sees the local
 equality leaf) while `allowOrDisintegration` resolves TRUE
 (`lmv.productOfDisintegration` — the intro premise carries `u_`
 args). The fired or-head therefore or-disintegrates into a full
 `_ordis_` branch cohort.
3. The closure trace shows the loop: 47 intro firings (e.g.
 `(>[1](=[u_7,1])(or3[u_6,1,u_2,u_7]))` on the alias equality
 `(=[7,it_0_lev_2_26])`@main), each novel or-tuple spawning three
 branches; branch assumptions breed equalities (ex falso), which
 fire the intro rules again — or-statements INSIDE branch scopes are
 visible throughout. The axed-anchor exception feeds this multiplier
 with many new equality-head firings (the x-anchor instances),
 consistent with the explosion.
4. The designed consumption of an intro-fired or is FLAT: the
 description's membership-intro clause (`or ⇒ w∈M`) fires on the
 compact statement. Branching an intro-derived or is redundant by
 construction — its source leaf is true at the same scope.

### Verdict — sound and precisely scoped

`FiringRecord::doNotDisintegrate = true` for intro-rule firings skips
`disintegrateExpr2` AND forces `allowOrDisintegration = false` at the
deposit (the existing D-32 coupling); statement registration, delta,
hash visibility, and mail are untouched, so every flat consumer keeps
working. Elimination-fired or-heads (premise = membership, not a
leaf) still branch — rung-2's descent machinery (predecessor
disjunctions from non-leaf premises) is unaffected. Status-0 assumed
or-fuel is unaffected.

### Recommended mechanization (maintainer approval pending)

The LMV already carries install-time per-rule bits the firing site
consumes (`isMarker`; `productOfDisintegration`, D-32) — add a third
(`orIntroHead`), consumed as `doNotDisintegrate = (justification ==
integration) || orIntroHead`. Do NOT reuse
`RuleJustification::integration` itself: the justification word is
the firing's exported history tag, and the verifier's integration
checker validates a different row shape — the separate bit keeps
chapters and verifier byte-compatible. Detection at INSTALL, by
shape, inside `addToHashMemory`: single premise + or-category head
(compiled-map probe) + premise equal to one flattened leaf of the
head instance. Shape detection — not a plumbed emission flag — is
load-bearing: intro rules travel cross-LB as compact
`(implication<N>)` statements and re-install receiver-side via
status-3 recovery through the same drain, where an emission flag
would be lost; the shape re-marks them everywhere, deterministically.

### Risks recorded

1. Completeness: proofs that closed only via convergence over
 intro-fired branches; the 68-row corpus was built with intro
 branching active — emission-level diff on the acceptance run
 decides. The rung-2.1 kill chain does not need those branches
 (`2≤1`@main has the flat equality1 route).
2. The FiringRecord sort includes `doNotDisintegrate` — apply order
 shifts for affected records (deterministic, trajectories move).
3. The LMV serialization gains one field (run-scoped; the sacred dump
 format stays untouched per Rule 14).
4. Whether this alone contains the runtime explosion is unproven —
 the elimination-fired branch trees remain (bounded pre-exception at
 ~1524 s); the next acceptance run measures it.

## 2026-07-29 — flag accepted mechanically: THE RUNG-2.1 NEGATION THEOREM PROVES AT BURST 5; the batch then explodes in its post-proof tail — the slow-LB profile names three mechanisms

Two runs, both killed on maintainer order after the same deterministic
explosion; the second carried the sacred dump on the straggler itself.

1.  (flag acceptance; dump still on the
 reductio child, partial trace preserved as
 ).
2.  (dump retargeted at
 `(EnumerationSet3[2,6,7,10])`, commit; trace preserved
 as , 21 dumps through
 ENTRY #11, 1.22 GB).

**The rung's theorem is in hand mechanically.** In BOTH runs the batch
prints, at burst 5:

```text
(>[1,2,3,4,5,6,7,8,9](AnchorIncubator3[1,2,3,4,5,6,7,8,9])(>[10](EnumerationSet3[2,6,7,10])!(interval[1,4,2,6,10])))
```

— the rung-2.1 negation theorem, alongside the re-proved ES2 disproof
and the `existence4`/`existence5` companions. The x-anchor antisymmetry
route plus the intro-branch containment deliver the proof ten bursts
earlier than the sibling-cascade baseline ever could. The theorem does
NOT survive into `files/theorems/theorems.txt` because the batch never
finishes: burst dt runs 9.6 → 9.8 → 25.5 → ~400 s (bursts 4→7), with
`(EnumerationSet3[2,6,7,10])` the runaway (work 538,070 at burst 6 →
2,339,809 at burst 7; 32 buckets ineffective).

### The profile (EXIT #8 → #9 → #10 of the ES3 LB)

Statements 11,175 → 21,142 → 44,550; origins 56,439 → 155,310 →
557,532; nameMap 130,682 ids; `startInt` 135,701 and `startIntPi`
71,025 (seventy-one thousand fresh integration variables);
`orBookkeeping` 30,934. Only 23 distinct scopes, depth ≤ 2 — the
explosion is ARGUMENT-VARIANT count, not scope count: 96% of all
statements live under the goal-clause boundaries
`main_boundary_(implication22[1,4,2,7,10])` / `[1,4,2,6,10]` /
`(implication21[10,1,4,6])` in their `_ordis_` case-split and
`_orint_` element-proof branches.

### Three mechanisms, all evidenced

1. **The witness factory (dominant).** The totality externals
 (`(>[1](in[1,u_1])(>[2](in[2,u_1])(existence1[u_1,1,2,u_4])))` and
 its `*` twin) fire on every membership pair at the branch scopes —
 numeral × witness and witness × witness included — and each
 existence mints its output witness, whose `in[w,1]` typing feeds
 the next round: verified chain
 `in3[it_0_lev_1_11197,int_lev_1_11,it_0_lev_1_48395,4]` (a witness
 plus a witness minting a third), 5,152 statements citing witness
 11197 alone, 8,252 fresh `existence1` variants in burst 10. This is
 the classic totality-closure explosion, running inside the
 `_orint_` branch `(=[7,repl_lev_1_4])` of the [0,2]-intro clause.
2. **Moot goals keep their machinery alive.** `toBeProved` still
 carries `(interval[1,4,2,6,10])` at `main` — the goal whose
 NEGATION proved at burst 5 — plus its clause sub-goals (the
 correctly-unprovable `(preorder[1,4,repl_lev_1_1,6])` under
 `implication21`, the membership goals, the three `_orint_` element
 equalities). The disproof retires nothing; the [0,1] machinery
 grinds to the budget's end.
3. **Ex-falso variant factories in false branches.** Under the
 `_ordis_` assumptions (`(=[2,7])` "0=2", `(=[2,6])` "0=1") the
 equality classes merge numerals and variant generation rewrites
 every statement family through them — 1,049 argument-variants of
 the ES3 description itself, 11,834 or-statement variants (each
 entering cohort bookkeeping). The branch `(=[2,6])` even holds the
 explicit pair `!(=[2,2])` — provably absurd, still producing. The
 never-implemented contradictory-branch closure (OPEN-1) now has its
 concrete acceptance case.

Why the explosion is NEW: the baseline reached `2∈M` at bursts 14–15
with no budget left; the early proof activates the same downstream
machinery at burst 5 with ten bursts of budget. The improvement and
the explosion are the same event — post-proof tail fan-out.

### Candidate directions (maintainer's decision, Rule 8 — no code changed)

- **Goal retirement on disproof:** when a conjecture's negation
 theorem proves, drop the positive goal and wipe its clause/branch
 subtrees (the I-48/I-50 lifecycle machinery exists). Removes the
 whole [0,1] slice at burst 5.
- **Contradictory-branch closure (OPEN-1):** close an `_ordis_`
 branch on a branch-scope E/!E pair; kills the ex-falso factories.
- **Witness-cascade containment:** gate the totality producers on
 witness-typed premises (no witness-on-witness totality), or a
 per-scope witness budget.
- The batch's 15-burst budget is the only current bound on the tail;
 none of the above is a config knob today.

## 2026-07-29 — generation wire shipped and VERIFIED (witness factory contained); the batch now finishes its bursts but still burns ~30 minutes in three ex-falso branch cohorts under the moot goals; a silent export crash is open

The witness-generation wire (`D-233`,
commit; plan-approved) landed with 1336/1336 unit tests.
Acceptance run , trace preserved as
 (1.22 GB, all 15
bursts at the ES3 LB).

1. **The wire works as designed.** Generations are real and capped:
 899,805 `it_1_lev` and 30,486 `it_2_lev` mentions, nothing deeper —
 generation-1 statements fire one final round, generation-2 are
 terminal under `maxIterationNumberVariable = 1`. The burst loop
 that previously exploded at ~400 s/burst COMPLETED all 15 bursts
 (EXIT #15: 56,145 statements, 115,644 origins — versus 44,550
 statements already at EXIT #10 pre-wire with the trajectory still
 accelerating).
2. **The remaining runtime sink is NOT the witness factory.** The
 trajectory jumps +22K statements in burst 10 alone; of those
 23,067 delta rows only 133 carry a generation-1 witness. 22,969 sit
 in exactly THREE `_ordis_` false-assumption branch cohorts
 (assumptions `0=2` / `0=1`) under the moot goal-clause scopes —
 `implication22[1,4,2,6,10]` (the DISPROVED [0,1]-intro),
 `implication22[1,4,2,7,10]` ([0,2]-intro), and
 `implication21[10,1,4,6]` (the correctly-unprovable `p≤1`) — as
 equality-class variant floods: 11,269 or-statement argument
 variants plus ~976 rewritten copies each of the ES3 description /
 `implication1370/1371` / `existence4/5` families. `toBeProved`
 stays frozen at 11 all run — the disproof retires nothing. The
 batch's prover phase now takes ~30 minutes (11:24 → 11:54) against
 the ~80-second rung-2-era baseline.
3. **Conclusion:** of the three profiled mechanisms, one is closed
 (witness factory — by the wire) and the remaining runtime lives
 exactly in the two levers already on the table: GOAL RETIREMENT ON
 DISPROOF (all three producing scopes belong to moot goals) and the
 OPEN-1 CONTRADICTORY-BRANCH CLOSURE (all producing branches are
 provably absurd). Either alone likely restores the batch; both
 together are the principled pair.
4. **Open defect:** the run crashed with a silent 0xC0000409 inside
 `generateRawProofGraph` AFTER the prover phase completed (theorem
 pool + GL_binary written; zero raw chapters emerged; no assert
 message despite unbuffered stderr — a fail-fast, not the buildStack
 no-origin assert). A TEMPORARY flushed per-theorem export trace is
 in the tree (commit, Rule 30 — still present); the
 maintainer stopped the reproduction run since the runtime alone
 already needs the levers above first. The crash reproduces
 deterministically whenever the batch is rerun. (Correction to this
 entry as first written: whether the rung-2.1 negation theorem
 proved in THIS run is unknown — the batch's buffered stdout died
 with the process; the burst-5 print is confirmed only for the
 preceding flag-acceptance run.)

## 2026-07-29 — the flagged casualty is real: the generation wire at cap 1 UNPROVES rungs 1 and 2 (maintainer finding, trace-confirmed)

The maintainer reports the crashed run failed to prove rung 1 and
rung 2. Trace confirmation at the ES3 LB
(`hashburst_generation_wire_20260729.txt`, EXIT #15): the rung-2 goal
`(interval[1,4,2,7,10])` at `main` is STILL OPEN after all 15 bursts —
`toBeProved` frozen at 11 the whole run, whereas the pre-wire runs
closed it (and the flag-acceptance run printed the theorem). Rung 1
(the ES2 LB, not dumped) falls to the same machinery.

**Mechanism — exactly the risk the plan recorded.** The interval-intro
clause proofs close through `_orint_` element case analysis whose
descents chain SPECULATIVE `it_` witnesses (the rung-2 campaign's
D-210 widening existed precisely for firings carrying three distinct
descent witnesses). Under honest generation labels those chains reach
generation 2+; at `maxIterationNumberVariable = 1` their statements
drop out of request building, the descents freeze, the `_orint_`
goals never close, and the rungs stay unproved. The disproof-side
chains (rung 2.1's negation) are generation-shallow and may survive —
unknown for this run.

**Options for the maintainer (recorded, not chosen):**
- **(a) Per-batch cap raise** (`maxIterationNumberVariable` 1 → 2,
 then 3 if needed, for IncubatorGauss3): recovers the descents;
 reopens one more quadratic totality layer per step — cost unknown
 until measured, and the ex-falso cohorts (the current ~30-minute
 burn) are generation-independent and stay either way.
- **(b) Land the moot-goal / absurd-branch levers FIRST** (goal
 retirement on disproof; OPEN-1 contradictory-branch closure), then
 raise the cap into a cleaned batch — the ordering that keeps the
 cap raise's marginal cost small. Recommended by the analysis.

## 2026-07-30 — RUNG 2.1 CLOSED (and with it rung 2), determinism-confirmed

The rung-2.1 negation theorem proves in every acceptance run (axed-anchor activation for AnchorIncubator3 on top
of the dead-branch retirement line):

```text
(>[1,2,3,4,5,6,7,8,9](AnchorIncubator3[1,2,3,4,5,6,7,8,9])(>[10](EnumerationSet3[2,6,7,10])!(interval[1,4,2,6,10])))
```

Closure evidence (run  vs the committed
reference state @ 9ac50948): verifier 138,948 checks / 0 failures —
airtight; `compiled_theorems.txt` byte-identical across the determinism
pair (66 theorems, rungs 1 and 2 included); the sacred ES3 hashburst
trace byte-identical across three consecutive runs
(`hashburst_xed_fix_20260730.txt`); the vacuity classification stable at
exactly the 8 retracted-producer rows in `vacuous_theorems.txt`.

The engine work that carried the rung: dead `_ordis_` branch retirement +
vacuous-theorem reversion ( @ f383ade9), the axed-anchor
deposit exception for AnchorIncubator3, reduced-cohort `or convergence`
rows with reductio ingredients (maintainer-consented verifier extension),
and the post-batch vacuity classification (premise-contradiction pairs +
taint closure). Maintainer declaration: rung 2.1 and rung 2 are
officially closed.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
