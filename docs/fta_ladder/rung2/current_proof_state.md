<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
Contributions require CLA — see CONTRIBUTING.md.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Rung 2 — {0,1,2} = [0,2] — current proof state

Append-only. Newest entry last.

## 2026-07-13 — conjecture formulated; proof blocked on the int16 name-id ceiling

The rung-2 conjecture (set {0,1,2} equals interval [0,2]) is formulated in
`files/config/ConfigIncubatorGauss3.json` (`EnumerationSet3` + `interval`
expressions, anchor-binding filters, `max_number_args_expr` 3). Both target
theorems emit as conjectures at hash burst 2 of the IncubatorGauss3 batch:

```text
(>[1,2,3,4,5,6,7,8,9](AnchorIncubator3[1,2,3,4,5,6,7,8,9])(existence5[2,6,7,1,4]))
(>[1,2,3,4,5,6,7,8,9](AnchorIncubator3[1,2,3,4,5,6,7,8,9])(>[10](EnumerationSet3[2,6,7,10])!(interval[1,4,2,7,10])))
```

The proof search minted past the int16 NameMap ceiling (origins 32,679 by
burst 8) and aborted on the `MAX_NAME_IDS` mint guard.

## 2026-07-14 — id migration complete; proof search is a runaway; proof deferred

The NameId int16→int32 migration (squashed onto ) removed the
ceiling (`MAX_NAME_IDS` = 1,000,000, main pool 12 GiB) and fixed every scaling
seam the deeper search exposed (owner-partition truncation, dump key
truncation, wipe-bitmap sizing, mail-absorb index chunking, page-vid free-list
reuse, equivalence-class run chunking). With ids unbounded, the rung-2 proof
search no longer converges within observed budgets:

- total expressions per burst: 21,601 (burst 6) → 59,600 (12) → 102,563 (13)
 → 197,168 (14) → 408,097 (15) — roughly doubling per burst, no fixpoint in
 sight at abort depth;
- page-vid churn grew ~2.8× per burst (524K vids by burst 13, 4.19M by 15),
 i.e. faster than expression growth.

**Assessment: the proof engine's search on this conjecture is a runaway** —
the exploration expands geometrically instead of converging. This is a
proof-engine / search-control question (what expansion the rung-2 shape
unleashes and why it does not saturate), not a memory-capacity question — the
capacity walls were all removed or made assert-loud.

**Next session:** debug the runaway itself (which rule family fires
combinatorially past burst 13; whether saturation is reachable; whether the
conjecture needs a narrower formulation or the engine a pruning control).
The rung-2 config stays formulated in `ConfigIncubatorGauss3.json`; pipeline validation with `main`'s IncubatorGauss3 config was
airtight (artifacts byte-identical to committed, verifier 0 failures).

## 2026-07-14 — human proof written; first missing inference identified

Human proof: [`proof_02_set_eq_interval.md`](./proof_02_set_eq_interval.md).
Primary evidence: , a 15-burst dump of the
`EnumerationSet3[2,6,7,10]` LB under `AnchorIncubator3`.

### Current target and compiled-name glossary

The current generated conjecture is the positive rung-2 theorem:

```text
(>[1,2,3,4,5,6,7,8,9]
  (AnchorIncubator3[1,2,3,4,5,6,7,8,9])
  (>[10](EnumerationSet3[2,6,7,10])(interval[1,4,2,7,10])))
```

Anchor slots: `1=N, 2=0, 3=s, 4=+, 5=*, 6=1, 7=2, 8=id, 9=3`;
`10=M`.

The ES3 three-way disjunction is compiled as a nested binary shape:

```text
or0[2,p,6]       = (p=0) ∨ (p=1)
or1[2,p,6,7]     = or0[2,p,6] ∨ (p=2)
```

The interval-body roles in this dump are:

| Compact expression | Role | Universal variable |
|---|---|---|
| `implication20[10,1,4,2]` | `p∈M ⇒ 0≤p` | `repl_lev_1_0` |
| `implication21[10,1,4,7]` | `p∈M ⇒ p≤2` | `repl_lev_1_3` |
| `implication22[1,4,2,7,10]` | `0≤p ∧ p≤2 ⇒ p∈M` | `repl_lev_1_4` |

### Covered steps

Every check below uses live `encodedStatements`, not provenance alone.

#### Step 1 — ES3 and both interval descriptions are installed ✓

At ENTRY #1 the LB already contains `EnumerationSet3[2,6,7,10]` at `main`
and the compiled interval bodies for both permitted upper endpoints. The live
goal set contains the true rung-2 goal `interval[1,4,2,7,10]` at `main`.

The same LB also contains `interval[1,4,2,6,10]` as a goal. That is the false
cross-pair `{0,1,2}=[0,1]`, not part of rung 2; see the secondary config issue
below.

#### Step 2 — ES3 membership unfolds to the nested three-way OR ✓

For the lower-bound universal variable, the trace records

```text
!(&!(or0[2,repl_lev_1_0,6])!(=[7,repl_lev_1_0]))
    <- expansion of or1[2,repl_lev_1_0,6,7].
```

The upper-bound universal variable has the identical shape with
`repl_lev_1_3`. Thus the prover has the correct mathematical case statement:
`p=0 ∨ p=1 ∨ p=2`.

#### Step 3 — the outer OR disintegrates ✓

For each forward interval clause the outer `or1` opens two `_ordis_` scopes:

```text
..._boundary_ordis_(or1[2,p,6,7])_((or0[2,p,6]))
..._boundary_ordis_(or1[2,p,6,7])_((=[7,p]))
```

The first scope represents the combined `p=0 ∨ p=1` case. The second is the
atomic `p=2` case.

#### Step 4 — the `p=2` lower bound proves ✓

By EXIT #4, the atomic `p=2` branch contains

```text
(preorder[1,4,2,repl_lev_1_0])
```

at the exact `or1... ((=[7,repl_lev_1_0]))` branch scope. This is the human
proof's witness `2` for `0≤2`.

#### Step 5 — the `p=2` upper bound proves ✓

By EXIT #3, the atomic `p=2` branch contains

```text
(preorder[1,4,repl_lev_1_3,7])
```

at the exact `or1... ((=[7,repl_lev_1_3]))` branch scope. This is reflexivity
`2≤2`, witnessed by `0`.

#### Step 6 — OR bookkeeping sees the atomic branch ✓

At EXIT #15, `orBookkeeping` contains both target preorder expressions, but
each has only

```text
disjuncts={((=[7,p]))}
```

for its outer `or1` signature. The bookkeeper is working: it records the only
outer branch that produces the conclusion.

#### Step 7 — reverse-clause nested OR-integration scopes are constructed ✓

The `implication22` path creates the outer integration goal
`or2[6,repl_lev_1_4,2,7]` and its inner `or0[6,repl_lev_1_4,2]` branches.
The three atomic equality goals `p=0`, `p=1`, and `p=2` are all present in
`toBeProved`. This confirms that nested OR integration is representable; none
of the three target equalities is closed by EXIT #15.

#### Step 8 — no target parent-scope discharge occurs ✗

The lower and upper preorder goals remain at their immediate implication
scopes through EXIT #15. The main target
`interval[1,4,2,7,10]` also remains. `toBeProved` drops from 19 to 15 at
EXIT #9 because four sibling goals close, then stays exactly 15 through
EXIT #15. Those four closures belong to the `[0,1]` cross-pair path; they are
not rung-2 closure.

### First missing inference — recursively open the nested `or0` branch ✗

The first absent result whose premise is already live is the inner case split

```text
or0[2,p,6]  ⟶  branch (p=0) and branch (p=1)
```

under the outer `or1` branch scope.

The trace is decisive:

| EXIT burst | lower `or0` branch statements | lower `p=2` branch statements | upper `or0` branch statements | upper `p=2` branch statements |
|---:|---:|---:|---:|---:|
| 1 | 1 | 14 | 1 | 14 |
| 3 | 1 | 66 | 1 | 70 |
| 5 | 1 | 1,512 | 1 | 1,512 |
| 7 | 1 | 2,130 | 1 | 1,529 |
| 15 | 1 | 2,130 | 1 | 1,529 |

For every burst, the combined `or0` branch contains exactly its seed
`or0[2,p,6]` and nothing else. There are zero descendant scopes of the form

```text
..._boundary_ordis_(or0[2,p,6])_((=[2,p]))
..._boundary_ordis_(or0[2,p,6])_((=[6,p])).
```

Therefore the human `p=0` and `p=1` cases are never instantiated. The outer
OR convergence correctly waits for its second branch; it is not the failing
component.

### Architectural cause

Three current contracts combine to make the missing step impossible:

1. `EnumerationSet3.mpl` is compiled as a nested binary OR, so three atomic
 cases require two disintegration layers.
2. `ConfigIncubatorGauss3.json` sets `max_or_depth` to `1`, explicitly
 permitting only the outer layer.
3. In `ExpressionAnalyzer::disintegrateExprCore2`, admitted OR disjuncts are
 collected in `orBranchStatements`; the caller installs them through
 `addStatement`, not through `addExprToMemoryBlock`. A disjunct that is
 itself an OR therefore never re-enters the disintegration pipeline. The
 D-32 `allowOrDisintegration` authorization belongs to the implication
 firing that produced the outer OR and is not carried as recursive branch
 provenance.

This is not a missing arithmetic theorem and not merely an iteration budget.
The architecture represents a three-way case split as two OR layers while the
execution contract can consume only one layer.

Per Rule 8, the implementation must not choose a replacement design without
the user. The two honest architectural directions are:

- flatten the compiled three-way OR into three atomic `_ordis_` branches at
 one scope; or
- make an OR-valued branch seed recursively disintegrate with inherited,
 auditable D-32 authorization and permit depth `2` for this batch.

They have different proof-graph, scope, convergence, and runtime contracts.
No code change is made here.

### Secondary config error — false cross-pair conjectures

The current `conjectures.txt` contains the full cross product of
`EnumerationSet2` / `EnumerationSet3` with upper endpoints `1` / `2`:

```text
ES2 ⇒ [0,1]   true rung 1
ES2 ⇒ [0,2]   false cross-pair
ES3 ⇒ [0,1]   false cross-pair
ES3 ⇒ [0,2]   true rung 2
```

The endpoint exclusion patterns admit `6` or `7` independently of the
enumeration expression. The false siblings do not logically block the true
rung-2 discharge, but they keep the same LBs searching for impossible goals
and amplify the observed growth. The earlier main-scope contradiction fix
prevents them from being exported as false theorems; it does not remove them
from proof search.

### Why the runaway is a symptom, not the diagnosis

From EXIT #9 through EXIT #15 the goal count is frozen at 15 while the final
dump reaches 55,126 live statements and 426,995 origins. Meanwhile each
target `p=2` branch has already saturated and each nested `or0` branch remains
at one statement. More search cannot create a branch the control-flow contract
never opens.

### Acceptance evidence for a future approved design

A correct rung-2 architecture must show all of the following in the sacred
dump:

1. nested `or0` branch scopes exist below each outer `or1` combined branch;
2. the lower and upper preorder facts are derived in both inner atomic cases;
3. inner convergence promotes each preorder to the combined `or0` branch;
4. outer convergence records both outer disjuncts and promotes each preorder
 to the `implication20` / `implication21` parent scope;
5. `interval[1,4,2,7,10]` discharges at `main` without using either false
 cross-pair conjecture.

---

## 2026-07-14 — flat OR disintegration closes the forward inclusion

This entry supersedes the preceding entry's current-state diagnosis while
preserving it as the record of the gap found in the earlier trace. The frozen
IncubatorGauss3 evidence is
.
The complete Windows pipeline subsequently finished in 1645.82374 seconds
with 132033 verifier checks and zero failures; its log is
.

The approved architecture treats a contiguous nested `_ordis_` expression as
one logical disjunction. One disintegration call flattens the complete OR tree
into ordered, non-OR leaves. All leaves share the original outer OR signature
as their cohort, provenance, and namespace identity. Therefore the configured
`max_or_depth=1` consumes this entire logical disjunction as one level; it does
not create an intermediate branch for the nested `or0` expression.

### Covered steps

1. ES3 unfolds to the outer signature `or1[2,repl_lev_1_0,6,7]`.
2. EXIT #1 contains three sibling equality branches below that signature:
 `p=0`, `p=1`, and `p=2`.
3. No branch whose body is the opaque nested
 `or0[2,repl_lev_1_0,6]` exists in the trace.
4. Each of the three parent implication scopes owns a separate cohort with
 `disjunctCount=3`; the sacred dump identifies the parent on every OR-state
 row, so equal OR signatures cannot be mistaken for one shared cohort.
5. The lower-preorder cohort derives its fact in all three atomic branches and
 promotes it to `main_boundary_(implication20[10,1,4,2])` by EXIT #4.
6. The upper-preorder cohort derives its fact in all three atomic branches and
 promotes it to `main_boundary_(implication21[10,1,4,7])` by EXIT #4.
7. The goal count falls from 19 at ENTRY #1 to 17 at EXIT #4. These are the two
 forward interval goals, so the `_ordis_` portion of rung 2 is closed.
8. The reverse direction constructs its `_orint_` proof shape, but the atomic
 equality goals for `p=0`, `p=1`, and `p=2` remain, as does the resulting
 `in[repl_lev_1_4,10]` membership goal.
9. The goal count reaches 13 at EXIT #9 and remains 13 through EXIT #15. The
 true `interval[1,4,2,7,10]` goal is still present, so rung 2 is not complete.
10. The false cross-pair conjectures remain in the goal set. GL must not prove
 them, and this session neither removes them nor performs the future cleanup
 driven by a proved contradiction sibling.

### Current architectural boundary

Flat `_ordis_` disintegration and its multi-branch bookkeeping are no longer
the failing boundary. The trace proves both required properties: every atomic
alternative appears in the same outer cohort, and cohorts with the same OR
signature remain isolated by their parent validity scope.

The first remaining boundary is now the reverse `[0,2] ⊆ ES3` path. Its
nested `_orint_` structure is present, but the equality alternatives have not
discharged into membership. The trace establishes where progress stops; it
does not yet establish whether the next cause is equality production, nested
OR integration, or the promotion from integrated equality to membership.

### Acceptance evidence for the next investigation

The next trace must show, in order:

1. at least one true reverse equality alternative discharge in its intended
 scope;
2. the inner and outer `_orint_` layers promote those alternatives to the ES3
 membership fact;
3. `in[repl_lev_1_4,10]` and then `interval[1,4,2,7,10]` discharge;
4. the false cross-pair goals remain unproved and present for the separately
 planned contradiction-driven cleanup.

---

## 2026-07-14 — reverse proof stalls inside nested OR integration

This entry replaces the earlier iteration-cap hypothesis. The configured
15-iteration run already contains enough evidence to identify architecture and
code-contract gaps; `maxIterationNumberProof` remains unchanged at 15.

### Exact target frontier

1. The true reverse clause is
 `main_boundary_(implication22[1,4,2,7,10])` and its element under test is
 `p=repl_lev_1_4`.
2. The `p=1` inner branch derives a predecessor `q` of `p`, records `q ∈ N`,
 and reaches `in3[q,int_lev_1_13,6,4]`, meaning `q+l=1`, by EXIT #7.
3. The next human-proof step never appears: there is no `q=0`, `!q=0`, or
 successor row `in2[r,q,3]`. Therefore the second predecessor split has not
 started.
4. The outer `p=2` branch is weaker still. It receives only the opaque premise
 `!(or0[6,p,2])`, meaning `!(p=0 ∨ p=1)`. It receives neither atomic premise
 `!p=0` nor `!p=1`, and no predecessor of `p` is produced in that branch.
5. No positive `p=0`, `p=1`, or `p=2` branch head appears for the true clause.
 Its membership and interval goals consequently remain among the 13 goals at
 EXIT #15.

The global statement count continues to grow, but the target proof frontier
does not move beyond `q+l=1`. Global growth is therefore not evidence that the
required target inference is progressing.

### Architecture gap 1 — `_orint_` still sees syntax levels

Flat disintegration is complete, but integration is not its mirror.
`prepareIntegrationCore2` Case OR iterates only the immediate elements of the
compiled OR. For the three-element set, the immediate outer elements are the
nested `or0[p=0,p=1]` expression and `p=2`. The flat-leaf helper is called only
from `_ordis_` production.

Consequently reverse proof search constructs an outer two-way `_orint_` plus
inner two-way `_orint_` scopes instead of one three-alternative cohort. The
outer `p=2` branch receives one opaque negated inner OR rather than the two
atomic exclusions needed to follow the two predecessor reductions.

### Code-contract gap 2 — ancestor deduplication blocks exact-scope goals

The sibling endpoint-1 path gives a direct reproduction with all dependencies
present:

1. Its direct and nested `p=1` inner branches both emit `or branch proven`.
2. `or0[6,repl_lev_1_2,2]` becomes a known statement at the implication parent
 `main_boundary_(implication22[1,4,2,6,10])`.
3. The identical `or0` goal remains through EXIT #15 at the descendant outer
 branch
 `..._boundary_orint_(or2[6,repl_lev_1_2,2,7])_((or0[6,repl_lev_1_2,2]))`.

`addExprToMemoryBlock` Site F treats an ancestor statement as stronger than
and redundant with the descendant deposit, so it skips that deposit.
`dischargeToBeProved` nevertheless looks up only the exact
`(expression, validity)` goal key. The two contracts disagree: admission says
the ancestor fact is sufficient, while discharge refuses to close the child
goal without a duplicate child statement.

### Code-contract gap 3 — `_orint_` cleanup is not parent-scoped

`cleanUpOrIntegrationBranches` scans every validity and selects victims using
only the OR signature. The same inner `or0` signature exists directly below
the implication and again below the outer OR branch, but cleanup does not
include their different parent validities in the identity. Thus proving one
cohort can schedule branch cleanup in another cohort at a different stack
position.

### Required repair contract for a later approved implementation

1. A contiguous nested OR on the integration side must be one ordered cohort
 of atomic alternatives under the original outer signature.
2. Each chosen branch must receive the negations of every other atomic leaf,
 not a negated intermediate OR node.
3. Integration cohort cleanup must be scoped by both parent validity and outer
 OR signature.
4. Ancestor-visible facts and exact-scope goals must share one contract: an
 ancestor fact must either discharge the descendant goal directly or the
 admission path must still schedule the goal reaction without storing a
 redundant child statement.
5. False conjectures remain present and must remain unproved; no conjecture
 removal belongs to this repair.

---

## 2026-07-23 — reverse OR integration repaired and validated

This entry closes the three reverse-OR defects identified above. The two
complete Windows validation runs are:

-  with the preserved sacred trace
 ;
-  with the preserved sacred trace
 .

Both runs completed the full pipeline. They took 1707.57068 and 1651.54242
seconds respectively, and each finished with 132,033 verifier checks, zero
failures, and `All proof graphs verified`. The SHA-256 hashes of all four
required determinism artifacts were identical between the runs:

| Artifact | SHA-256 |
|---|---|
| `files/incubator/theorems/theorems.txt` | `C50868B01B3F6D6EEFDBA551503526FB27F86E4B01B20121A5BEE7E002C64D95` |
| `files/incubator/processed_proof_graph/global_theorem_list.txt` | `C886D91212ECDBD87EA6973AB7442C5A883A7A14E3907B10AF5E398361E45189` |
| `files/theorems/theorems.txt` | `BC15CB8905F0CCA4BB3E52F5A1795B6850627FBB50EA081BDE48CD85622E737C` |
| `files/processed_proof_graph/global_theorem_list.txt` | `6BFE1F0949A7215705BC377E0AC89BB422848F3B1A63F47AA87375851828951E` |

### Closed reverse-OR boundary

The repaired contracts are visible in the IncubatorGauss3 trace and direct
tests:

1. The compiled three-way reverse target now creates exactly three ordered
 atomic `_orint_` branches for `p=0`, `p=1`, and `p=2`. No opaque inner-OR
 branch exists.
2. Every atomic branch carries the negations of both other atomic leaves. In
 particular, the `p=2` branch now receives `!p=0` and `!p=1`, not
 `!(p=0 ∨ p=1)`.
3. Equal outer signatures below different parent validities remain separate
 cleanup cohorts.
4. On the endpoint-1 sibling path, the known
 `(=[6,repl_lev_1_2])` ancestor fact closes the exact descendant goal and
 emits `or branch proven` without depositing a redundant child fact.
5. The native suite passes 1,268 tests and the verifier suite passes 362 tests,
 including flat-branch rejection cases, parent isolation, ancestor ordering,
 shallowest-source provenance, sibling rejection, and no-child-deposit
 checks.

The old nested reverse-OR boundary is therefore closed. It is not the current
reason the rung remains open.

### Boundary status

Rung 2 remains open. This checkpoint records no diagnosis of the next failing
inference.

---

## 2026-07-23 — diagnosis of the frozen frontier; predecessor OR theorem restored

### Diagnosis of the post-repair stall

Trace analysis of the validated reverse-OR run identified the frozen frontier:
in both live `_orint_` branches of the true reverse clause, `s(q)=p`, `q∈N`,
`q+1=p`, and `q+l=1` are live from EXIT #7, but the second predecessor split on
`q` never starts. The only predecessor mechanism in the batch is the
conditional rule "x∈N ∧ x≠0 ⇒ ∃pred(x)"; the first split fired because `p≠0`
is an assumed `_orint_` peer negation, while `q≠0` must be derived — and no
loaded rule produces a variable-level negated equality, no branch-local
contradiction machinery exists, and no OR-form predecessor theorem was loaded.
Rung 1 is exactly the depth at which every needed negation is an assumption;
rung 2 is the first depth where one must be manufactured. The maintainer chose
the general repair direction: branch-local case analysis (an `_ordis_` split
nested inside `_orint_` branches with contradictory-branch rejection — the
documented contradictory-branch design gap), with rung 2 as its acceptance
case.

### Prerequisite restored — the predecessor OR theorem exists again

The campaign's first work item discovered that the predecessor OR theorem had
vanished from all artifacts (an early-May regression): its companion parent
"no predecessor ⇒ n=0" no longer proved, because the companion's induction
step sub-LB received the inherited premise "n has no predecessor" only as a
mail-absorbed statement whose universal rule was never installed — the
status-3 absorb gate admitted only compact implications to disintegration.
A second, latent defect installed the induction-hypothesis rule with the
premise's negation dropped. Both are fixed on: the
absorb gate admits negated compact existences as the second rule-carrier
shape, and the auxiliary-implication extractor keeps a leading negation.

Validation: the companion closes by vacuous truth at its second burst
(sacred-dump evidence 
broken /  repaired); the
`or theorem` row `(>[N,i0,s,+,*,i1](AnchorPeano[N,i0,s,+,*,i1])(>[v1](in[v1,N])(or0[N,v1,s,i0])))`
is constructed, registered, and exported to the incubator externals; the full
pipeline finishes with 131,827 verifier checks and zero failures
(, 1447.9 seconds); the Gauss summation
theorems are intact; and the batch newly disproves the false cross-pair
`EnumerationSet2 ⇒ interval[0,2]` outright. The true rung-2 conjecture and
the remaining false cross-pair goals stay open, as required — the nested
case-split machinery (authorization, depth accounting, branch rejection) is
the next work.

## 2026-07-25 — the missing negated-equality producer exists (conjecturer-template + contradiction campaign)

The stop4 diagnosis named the reverse clause's blocker: the second
predecessor split needs `q≠0` *derived*, and no loaded rule produced a
variable-level negated equality. That rule now exists. The campaign squashed
onto as (source + `_contradiction` / `_dedup` / `_experiments`
preserved) landed:

- **Conjecturer template addon** — emits the injectivity contrapositive
 `op(a)=b ∧ op(c)=d ∧ b≠d ⇒ a≠c` for every usable single-input operator
 (Peano main and both Peano incubators emit it for `in2`).
- **Reductio contradiction LB** (`try_contradiction_negated_head`) — assumes
 the negated head; a main-scope contradiction proves the conjecture. Proved
 the theorem at **AnchorIncubator3**
 (`(>[1..9](AnchorIncubator3[1..9])(>[10,11](in2[10,11,3])(>[12,13](in2[12,13,3])(>[]!(=[11,13])!(=[10,12])))))`),
 which the rung-2 batch (`IncubatorGauss3`) now loads as a proved external,
 and — after the main-path activation — at **AnchorPeano** as a globally
 essential theorem.
- Support machinery: theorem-sink dedup, contradiction-twin deactivation,
 induction-auxiliary retirement for both head polarities; symmetric
 verifier extension (maintainer-consented).

The claim "no loaded rule produces a variable-level negated equality" is
therefore obsolete for the next session: from `s(q)=p`, `s(0)=1`, `p≠1` the
loaded contrapositive derives `q≠0`, unblocking the second predecessor
split. The nested case-split machinery remains the open alternative track;
whether the reverse clause now closes with the new rule alone is the next
rung-2 investigation. Verified state: byte-determinism double run, verifier
133,104 checks / 0 failures.

---

## 2026-07-25 — reverse clause reaches the second predecessor split; the OR depth gate leaves branch-scope predecessor disjunctions inert

Evidence:  (full Windows pipeline,
1600.57 seconds, 133,104 verifier checks, zero failures) with the preserved
sacred trace  — the dump
retargeted to the rung-2 LB `(EnumerationSet3[2,6,7,10])` under
`(AnchorIncubator3[1,2,3,4,5,6,7,8,9])`.

### Progress attributable to the campaign rules

1. The LB's goal count falls 13 → 11 (EXIT #4) → 8 (EXIT #10) → 7 (EXIT #11)
 and freezes at 7 through EXIT #15; before the campaign it froze at 13.
2. Both forward preorder clauses of the true pair close by EXIT #4.
3. The false cross-pair's reverse inclusion ([0,1] ⊆ {0,1,2}) discharges
 completely: its three atomic `_orint_` equality goals and its membership
 goal `in[repl_lev_1_2,10]` all close. The `_orint_` → membership machinery
 is functional end-to-end at rung-1 depth, where one predecessor split
 suffices and every needed negation is an assumption.
4. In the true reverse clause's `p=2` branch
 (`..._orint_(or3[6,repl_lev_1_4,2,7])_((=[7,repl_lev_1_4]))`) the
 predecessor chain is live at the branch scope: `s(q)=p` as
 `(in2[it_0_lev_1_1028,repl_lev_1_4,3])` and `q∈N` as
 `(in[it_0_lev_1_1028,1])` from EXIT #5, and — via the new injectivity
 contrapositive — the derived `q≠0` as `!(=[2,it_0_lev_1_1028])` from
 EXIT #6. The stop4 blocker ("`q≠0` must be derived and no loaded rule
 derives it") is closed.

### The new frontier — `or0[1,q,3,2]` is inert at OR-branch scopes

The predecessor OR theorem fires on `in[q,1]` and deposits
`(or0[1,it_0_lev_1_1028,3,2])` ("∃pred(q) ∨ q=0") at the same branch scope.
It is never consumed: nine bursts (EXIT #6 through EXIT #15) pass with `q∈N`,
`q≠0`, and the disjunction all live, yet `existence3[1,it_0_lev_1_1028,3]`
never appears, so the second predecessor split never starts. The branch
instead accumulates 17,498 flat statements with zero sub-scopes.

Mechanism, from the trace and `ExpressionAnalyzer::disintegrateExpr2`'s OR
case:

- At non-branch scopes an admitted OR statement receives an expansion record
 plus the K mutual-exclusion implications — for a two-leaf OR the two
 disjunctive-syllogism hash rules (`¬B ⇒ A` and `¬A ⇒ B`). Exactly 17
 variables carry these residual predecessor rules at EXIT #15, all minted at
 implication or hypothesis scopes (OR depth 0). This is precisely how the
 first predecessor split fired.
- At OR-branch scopes the entire OR case is skipped. `disintegrateExpr2`
 counts `_(or` occurrences in the validity name and requires
 `currentOrDepth < max_or_depth` (1 in this batch) before any processing;
 a branch validity already contains one `_(or`, so the or0 statement is
 inserted inert — no expansion record, no K implications, no branch opening.
 Of the 193 live `or0[1,x,3,2]` statements at EXIT #15, 178 sit at
 `_orint_`/`_ordis_` branch scopes in this dead state; the trace contains
 zero or0 expansion events at any branch scope.

### Architectural boundary

`max_or_depth` conflates two different consumptions of an OR statement:
(a) opening nested `_ordis_` branch scopes — the scope explosion the cap
exists to prevent — and (b) emitting the K mutual-exclusion implications,
which are flat hash rules that create no scopes. The human proof's step
"`q≠0` ∧ (∃pred(q) ∨ q=0) ⟹ ∃pred(q)" needs only (b) inside the branch.

Per Rule 8 no code change is made here. Candidate directions for the
maintainer's decision:

- emit the K mutual-exclusion implications (and the matching expansion
 record) regardless of OR depth, gating only branch opening on
 `max_or_depth`; or
- the previously chosen general track: branch-local case analysis (`_ordis_`
 nested inside `_orint_` branches with contradictory-branch rejection),
 which subsumes the syllogism but is a larger machine.

Whether the remainder of the human chain (`s(r)=q`, then `r=0` forced by
`p≤2`, hence `p=2`) closes with flat rules alone is the next empirical
question after either repair.

---

## 2026-07-25 — K implications ungated from OR depth; second predecessor split fires; frontier moves to a non-firing successor-addition instance

The maintainer approved the first candidate direction
(`D-211`, commit): in
`disintegrateExprCore2`'s OR case the expansion record and the K
mutual-exclusion implications now emit at every OR depth; only the
per-branch `_ordis_` case-split remains gated on `max_or_depth` and the
D-32 admission flag.

Validation run:  (full Windows
pipeline, 1637.03 seconds, 133,104 verifier checks, zero failures) with the
preserved sacred trace .
Both theorem artifacts (`files/theorems/theorems.txt`,
`files/incubator/theorems/theorems.txt`) are byte-unchanged against the
committed baseline — no losses (Gauss summation intact), no additions.

### The repair works — acceptance evidence

In the true reverse clause's `p=2` branch:

1. `q≠0` derives at EXIT #6 as before; the previously inert
 `or0[1,q,3,2]` now yields its syllogism rules, and
 `existence3[1,it_0_lev_1_1028,3]` plus the witness `s(r)=q`
 (`in2[it_0_lev_1_2150,it_0_lev_1_1028,3]`) appear at EXIT #7 — one burst
 after the negation, where the previous run sat idle for nine bursts.
 The second predecessor split fires.
2. The chain then advances: `q+l=1` (`in3[it_0_lev_1_1028,int_lev_1_13,6,4]`,
 EXIT #8), `l+r=x` (`in3[int_lev_1_13,it_0_lev_1_2150,it_0_lev_1_2322,4]`,
 EXIT #8), `1+r=q` (EXIT #9), and the q-level identification `x'=1`
 (`=[it_0_lev_1_1170,6]`, EXIT #8) via predecessor uniqueness.
3. The LB's hash memory now grows at branch scopes (1,587 → 2,129 across the
 run; it froze near 1,553 before), confirming rules install inside
 branches.

### The new frontier — one staged firing does not happen

The goal count still freezes at 7 from EXIT #11; the rung does not close in
the 15-burst budget. The single missing link is now precisely staged: the
loaded successor-addition theorem
(`a∈N ∧ s(a)=b ∧ c+a=d ∧ c+b=e ⇒ s(d)=e`) instantiated at the r-level
(`a=r, b=q, c=l, d=x, e=x'`) — every premise live at the branch from
EXIT #8 — would give `s(x)=x'`; with `s(0)=x'` already live, predecessor
uniqueness would then force `x=0`, then `r+l=0`, and the remaining ladder
(`r=0`, `q=1`, `p=2`) would discharge the branch goal. The identical rule
shape fired at the q-level at EXIT #7 (branch ≈5k statements). At the
r-level it does not fire for seven bursts (#9–#15) while the branch grows
from ≈8k to ≈45k statements (whole LB 66,715; heavy `existence1` closure
churn dominates the growth).

Not yet diagnosed: why the staged instance never fires. The rule itself is
verified present in the LB's hash-memory originals at EXIT #15 —
`(in[1,u_1]) (in2[1,2,u_3]) (in3[3,1,4,u_4]) (in3[3,2,5,u_4]) (in2[4,5,u_3])`
— so rule absence or eviction is excluded. Candidate causes for the next
investigation: per-burst work caps / submatch truncation under the branch's
combinatorial load, LB-split request dealing, or an admission-map state (a
consumed or rejected key for this template) — distinguishing them needs a
targeted hash-request trap on that rule and premise set.

---

## 2026-07-25 — root cause: the distinct-secondary-variable cap blocks every request deeper than one predecessor level

A three-layer request trap (temporary, Rule 30; commit) instrumented statement presence, request assembly
(`BurstSink::consume`), and every verdict point of
`checkLocalEncodedMemoryStatic`. The instrumented full pipeline ran clean
(133,104 checks, zero failures; log ,
trap output preserved as ).

### What the trap proved

1. All four r-level premises are present at the branch scope with
 `known=1, registered=1` from their arrival bursts (`in3[l,q,x']` #7,
 `in[r]` + `s(r)=q` #8, `l+r=x` #10). The mandatory filter does NOT drop
 them (iteration cap: `it_0` parses to 0 ≤ `maxIterationNumberVariable`=1).
2. Requests carrying the r-premise ARE built — six of them, all ≤ 3 elements,
 all passing the dependency skip; the 3-element `{in, in2, in3}` shapes hit
 marker keys (admission staging), not the head rule.
3. The q-level instance fired via 4-element requests (18 built) — every one
 pairs a branch-scope `in3` with an ancestor-scope `in3` carrying a numeral
 constant.
4. The exact 4-element normalized key the r-level needs EXISTS in
 `normalizedEncodedKeys`, owned by the pure rule; every intermediate grow
 pattern (the 2-node and the 3-node base) is registered in
 `normalizedEncodedSubkeys` / `...MinusOne` with the pure rule as owner.
 Nothing is missing from the index side.

### The gate

`requestGatesPass` (run per grow-DFS node and per merge) rejects any request
whose DISTINCT secondary variables — args with `argIteration > -1`, i.e.
exactly the `it_*_lev_*` witness variables, products of recursion excluded —
exceed `maxNumberSecondaryVariables` = 2 (`ConfigIncubatorGauss3.json`).

- q-level firing `{in[q], in2[q,p], in3[l,p,7], in3[l,q,x']}`: distinct
 secondaries {q, x'} = 2 — passes exactly at the cap (`p` is `repl_lev`,
 `l` is `int_lev`, both iteration −1).
- r-level base `{in[r], in2[r,q], in3[l,q,x']}`: {r, q, x'} = 3 — rejected
 at the DFS node, so the base is never grown; the full 4-element form
 carries {r, q, x, x'} = 4. Every emitted r-marked request carries ≤ 2
 distinct `it_` variables — full consistency.

Each predecessor split mints one more `it_` witness, so the cap is the
depth limiter of the entire predecessor-descent approach: rung 1 (one
split) fits inside 2; rung 2 (two splits) needs 4 distinct secondaries in
the one successor-addition firing and can never fire under the cap.

### Status

Diagnosis only; no fix chosen and no code or config changed for it. The
trap layers remain in place (their removal accompanies the fix per
Rule 30). The decision on the table: raise `maxNumberSecondaryVariables`
to 4 for the rung-2 batch (search-width blow-up risk — the cap plausibly
contains the documented runaway) versus a structural carve-out (e.g.
predecessor witnesses joining the products-of-recursion exclusion), which
is an architectural change requiring its own design approval.

---

## 2026-07-25 — cap-4 trial: the firing unblocks, the fan-out explodes; run stopped by the maintainer

The maintainer chose the config trial: `maxNumberSecondaryVariables`
2 → 4 in `ConfigIncubatorGauss3.json` (commit;
the request-trap layers were removed in the same commit per Rule 30 —
source byte-identical to the pre-trap state, build clean, 1,287/1,287
unit tests).

Measured outcome (run , stopped on the
maintainer's instruction during IncubatorGauss3 burst 11; partial sacred
trace preserved as ):

1. **The blocked class fires.** The partial trace contains 6,982
 successor-addition firings whose premises include two branch-scope
 `in3` rows — the exact class the cap-2 gate made unbuildable. The
 mechanism verdict of the diagnosis is confirmed end-to-end.
2. **The explosion is real.** Batch burst times: ~5 s (bursts 3–5),
 23 s (burst 6), 457.8 s (burst 7). The target LB reached EXIT #10
 with 48,877 statements and 741,549 origins (cap-2 run at the same
 burst: 7,495 statements, 51,314 origins). Process working set passed
 7 GiB. Goal count 13 → 11 → 8 by EXIT #10 — the same trajectory
 points as cap 2, with no additional closure visible before the stop.
3. The rule fans out over every sum-witness pair in the branch, which is
 precisely the growth the cap contained. Whether bursts 11–15 would
 have converged and closed the rung is unmeasured; each burst was
 taking minutes and growing.

State after the stop: the config keeps `maxNumberSecondaryVariables: 4`
as committed; `files/` artifacts are from the interrupted run and will be
regenerated by the next full pipeline. Open decision for the next
session: pay the cost and measure a full 15-burst run to completion;
constrain the widened budget (for example a scoped grant — extra
secondaries only for requests whose premises include a successor chain,
or predecessor witnesses joining the products-of-recursion exclusion,
both Rule-8 designs); or revisit the depth approach.

---

## 2026-07-25 — RUNG 2 PROVED — the `_orint_`-scoped secondary cap closes it

The maintainer refined the widening twice more. A flat `maxNumberSecondaryVariables: 3`
trial also exploded (198-second burst 8, batch expression count tripling
per burst; run stopped — ). The equality
analysis behind that trial remains the load-bearing insight: the sum
variable x′ carries a constant equivalence (`x′ = 1`), the rewrite
`(in3[int_lev_1_13,it_0_lev_1_1028,6,4])` already exists at the branch, so
the successor-addition instance needs only 3 distinct secondaries — and q,
r, x have no other-shape equivalents (their classes contain only `it_`
siblings), making 3 the hard floor.

The final design (maintainer-specified, `D-210`):
`requestGatesPass` admits a request over the standard cap only when it fits
the new `maxNumberSecondaryVariablesOrint` AND every premise sits at ONE
shared validity scope AND that scope is an `_orint_` branch. The rung-2
batch sets 2/3; the parameter defaults to no widening, so every other batch
is untouched. All four premises of the needed firing sit at the one branch
scope, so the grant reaches exactly the predecessor-descent chains.

**Result** (run , sacred trace preserved
as ):

```text
(>[1,2,3,4,5,6,7,8,9](AnchorIncubator3[1,2,3,4,5,6,7,8,9])(>[10](EnumerationSet3[2,6,7,10])(interval[1,4,2,7,10])))
```

proved at hash burst 11 of IncubatorGauss3, alongside its companion
existence form (`existence6[2,6,7,1,4]`); the batch finished in 79.4
seconds with single-digit-second bursts (compare: 457 s per burst at
global cap 4, 198 s at flat cap 3), expression counts near 25k, and goal
LBs discharging from burst 12. The full pipeline finished in 1515.76
seconds — faster than the ~1600-second baseline — with 133,523 verifier
checks, zero failures, `All proof graphs verified`. The tracked theorem
artifacts are unchanged (Gauss summation intact); the rung-2 theorem
lands in the untracked incubator pool.

**Soundness cross-check.** The only ES3-premised statement in the pool is
the true theorem; the 2026-07-14 false `!(interval[1,4,2,7,10])` proof
does NOT reproduce, and the false cross-pair `ES2 ⇒ [0,2]` is disproved
outright while `ES3 ⇒ [0,1]` stays correctly unproved.

Rung 2 is closed. The trial ladder that got here: contradiction-campaign
rules (q≠0 derivable) → K-implication ungating (second predecessor split
fires) → request-trap root cause (distinct-secondary cap) → scoped grant
(this entry).

**Byte-determinism double run passed.** The second run
(, 1489.50 seconds,
133,523 checks, zero failures) reproduces all four determinism artifacts
byte-identically (SHA-256 equal for both `theorems.txt` files and both
`global_theorem_list.txt` files). The rung-2 state is squash-ready.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
