<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Current Proof State — analysis guide

Append-only **instruction** for how to analyze whatever theorem is
currently being proved on this branch. Each section below = one step
in the expected proof progression: **what should happen**, **why**,
and **how to verify it from the runtime dumps**.

Variable names in the trace are volatile — `repl_lev_*` / `int_lev_*`
numbering shifts after any bug-fix or ordering change in the prover.
So entries below refer to expressions by **role** ("the witness
introduced by the universal clause of interval"), not by the literal
integer suffix seen in any particular hashburst.

**Lifecycle.** When the current theorem closes, rename this file to
`<theorem>_theorem_proof.md` and start a fresh `current_proof_state.md`
for the next rung of the FTA ladder.

## Current theorem

`{0,1} = [0,1]` — human-notation proof in
[`proof_01_set_eq_interval.md`](./proof_01_set_eq_interval.md), §4.

GL stores the set equality as two directed conjectures:
- `theorems.txt:340` — `ES2 ⟹ interval` (forward, §4.1)
- `theorems.txt:342` — `interval ⟹ ES2` (reverse, §4.2)

**Active LB.** Innermost `(EnumerationSet2[a,b,M])` under parent
`(AnchorGauss[…])`. Goal = `(interval[N,+,a,b,M])` at `v=main`.
The §4.2 direction rides on the same LB — it appears as a goal
inside the integration of interval's body, not as a separate LB.

**Primary dump.** . Each `HASHBURST`
section prints LB chain (innermost → root), encodedStatements with
validity scope `v=…`, `toBeProved`, and `exprOriginMap`.

---

**Reading convention.** Each step below is a proof-progress landmark
that has already been **reached**. The step is stated as the ideal
inference ("what needed to fire"), followed by *how to confirm it
fired in the current trace*. Steps further along the roadmap are not
recorded here until they actually land — no pre-guessing.
Trace-interpretation rule: only rows in the `encodedStatements`
section prove live memory. The `exprOriginMap` dump was disabled because its speculative provenance edges were
misread as live facts.

---

## Compiled-name glossary (current Gauss GL binary)

`implication*` / `existence*` / `or*` / `and*` integer suffixes are
assigned by the C++ compiler. As of [D-22](../../agentic_swdd/40_decisions.md#d-22)
they are **persisted across batches** via
`files/GL_binaries/GL_binary_shared.json`: once a structural form is
named, every later batch reuses that name. Names below reflect the
post-cross-batch-registry numbering (Peano allocates first, Gauss
inherits). Names can still shift after a `files/definitions/` edit
(the registry is wiped on a clean run); re-derive suffixes from
`overallHashMemory.originals` rows when in doubt. The **roles**
below are stable.

**Anchor positions (`AnchorGauss[1..8]`).** `1=N, 2=0, 3=s, 4=+,
5=*, 6=1, 7=2, 8=id`. So `(=[2, p])` reads "p = 0" and
`(=[6, p])` reads "p = 1".

**ES2 body** (`EnumerationSet2[a, b, M]`):
- `or2[a, p, b]` — compiled OR: `(=[a, p]) ∨ (=[b, p])`.
 (Was `or0` pre-D-22, then `or1` after D-22 with the original Peano
 OR theorem at `or0`. With D-24 the head-switch block now emits
 *both* orderings of the Peano OR theorem (`or0[1,p,3,2]` and
 `or1[p,2,1,3]`), consuming `or0` and `or1`, so the ES2 disjunct
 shifted again to `or2`.)
- `implication20`, `implication21` — the two ES2 body directions
 (forward `(in[p, M]) ⇒ or2`; backward `or2 ⇒ (in[p, M])`). The
 argument permutation distinguishes them; confirm via the matching
 `overallHashMemory.originals` row.

**`interval` body** (`interval[N, +, lo, hi, M]`):
- `implication22[M, N, +, lo]` — forward conjunct 1:
 `(in[p, M]) ⇒ preorder(N, +, lo, p)`.
- `implication23[M, N, +, hi]` — forward conjunct 2:
 `(in[p, M]) ⇒ preorder(N, +, p, hi)`.
- `implication24[N, +, lo, hi, M]` — backward direction:
 `preorder(N, +, lo, p) ∧ preorder(N, +, p, hi) ⇒ (in[p, M])`.
 **All Steps 1–4 below fire inside this scope.**

**`preorder` body** (`preorder[N, +, n, m]` ≡ `∃k ∈ N. n + k = m`):
- `existence1[N, n, m, +]` — preorder unfolded:
 `∃k. (in[k, N]) ∧ (in3[n, k, m, +])`.

**Peano predecessor existence:**
- `existence0[N, p, s]` ≡ `∃w. (in[w, N]) ∧ (in2[w, p, s])` —
 "p has a predecessor in N". Head of the β template (Step 3),
 body disintegrated in Step 4. (Was `existence3` pre-D-22, then
 `existence2` after D-22; under the current registry Peano allocates
 it first as `existence0`.) The pre-D-22 sibling-unfold-path
 duplicate that the Gauss-only counter previously assigned to a
 separate compact name no longer exists — both unfold paths now
 resolve to the same `existence0` entry via `excludeRepetitions`.

**Atomic operators** (also documented in `the project conventions → ConfigVisu.json
reference`):
- `in[x, S]` — `x ∈ S`.
- `in2[a, b, s]` — `s(a) = b` (single-input operator, slot 0).
- `in3[a, b, c, +]` — `a + b = c`.
- `=[a, b]` — equality.

---

## Step 1 — OR case split mints two branch scopes ✓

**Inference.** implication21 (`(or2[a,p,b]) → (in[p,M])`, the ES2
backward direction) backward-chains on the active goal `(in[p, M])`
at the interval-body integration scope, producing subgoal
`(or2[a,p,b])`. `prepareIntegrationCore2` Case OR
(`prover.hpp:4026-4084`) dispatches two branch validities, each
minted via `NameMap::encodePush(parentValId,
"orint_(or2[…])_((head_k))")`:

- **Branch A** — head `(=[b,p])` at status=2, assumption `!(=[a,p])`
 at status=0 (tag `or branch assumption`).
- **Branch B** — head `(=[a,p])` at status=2, assumption `!(=[b,p])`
 at status=0.

**Why needed.** `(in[p, M])` has no hashMem rule with it as head
other than implication21. The two disjuncts of `or2` are mutually
exclusive at the model, so each branch's head is reachable only
under the negated assumption of the sibling disjunct. Structurally,
each branch scope **is** the contrapositive implication.

**How confirmed.** encodedStatements at each branch validity contain
`!(=[…])` (status=0) and `(=[…])` (status=2). Branch payload
attached via the literal `_boundary_orint_` separator
(scope-rooting invariant). `mb.integrationPrepared` suppresses
re-entry on subsequent `prepareIntegration` calls.

---

## Step 2 — Negated equality mirrored at branch scope ✓

**Inference.** The branch assumption `!(=[a,p])` coexists with its
swapped-arg mirror `!(=[p,a])` at the same branch validity. Mirror
is emitted by the `addNegatedEquality` gateway (`prover.cpp:4163+`)
via the same centralized choke point as positive equality
mirroring, tagged `symmetry of inequality`.

**Why needed.** hashMem unification is argument-order-sensitive. The
Peano template in Step 3 keys on `!(=[u_p, u_i0])`; without the
mirror, unification would fail and Step 3 could not fire.

**How confirmed**. 72 `symmetry of inequality`
rows in current trace, paired under both Branch A and Branch B
validities:

```
!(=[a, p])   | v=…_orint_…_((head_k))
!(=[p, a])   | v=…_orint_…_((head_k))
```

---

## Step 3 — Peano β template fires at Branch A ✓

**Inference.** With the mirrored branch assumption in place, the
Peano β mutual-exclusion template
`!(=[u_p, u_i0]) → (existence0[u_N, u_p, u_s])` (stored in
`overallHashMemory.originals`) unifies against the mirror
`!(=[p, i0])` at Branch A validity. Deposit:
`(existence0[N, p, s])` at Branch A validity. Reading: "p ≠ 0 ⇒ p
has a predecessor in N." Here `existence0` is the Peano compilation
of "∃y. (in[y, N]) ∧ (in2[y, p, s])".

**How confirmed.** `overallHashMemory.originals` shows the β/α
template pair (7 instances — one per universal-p witness):

```
!(=[u_<w>, u_<i0>])                  (existence0[u_<N>, u_<w>, u_<s>])   ← β
!(existence0[u_<N>, u_<w>, u_<s>])   (=[u_<w>, u_<i0>])                  ← α
```

encodedStatements at Branch A validity contains
`(existence0[1, repl_lev_1_2, 3])` at index `[149]` (post-D-22 run).

---

## Step 4 — existence0 disintegrates into its two ingredients at Branch A ✓

**Inference.** `(existence0[1, repl_lev_1_2, 3])` is compiled as
`∃y. (in[y, 1]) ∧ (in2[y, repl_lev_1_2, 3])`. `disintegrateExprCore2`'s
existence case (`prover.cpp:6984-7093`) mints a fresh it_ bound variable
`<w>` via the algebra-case path and registers the two element templates
`(in[<w>, 1])`, `(in2[<w>, repl_lev_1_2, 3])` into `newVarMap`. Pass B
in `disintegrateExpr2` (`prover.cpp:7322-7371`) must then admit `<w>`
for both elements to be deposited into `finalStringStatements` at Branch
A validity.

**Why needed.** Downstream proof steps (witness collapse via α-template,
`(in[p, M])` lift via implication rules) require both ingredients to be
live at Branch A. Before this change, the map-driven `isAdmitted` lookup
missed: no Peano template registered an input-slot-marker key for
`(in2[marker, u_p, u_s])` or `(in[marker, u_N])`, so admission failed
for the algebra-path it_ var and the existence0 body stayed buried.

**How confirmed** (single-input-operator fallback rule,
`prover.hpp:1858+` / `prover.cpp:7342`). The standalone rule
`isAllowedAsOperatorInput` admits `<w>` when the stmt is an operator
invocation with exactly one input slot and `<w>` sits at that slot
(depth/secondary-count bounds applied). `in2`'s input_args is `[0]` and
`in`'s input_args is `[0]` — both single-input — so both elements pass.
encodedStatements at Branch A validity contain, with the same fresh
it_ var (here `it_0_lev_1_86`):

```
[143] (in2[it_0_lev_1_86, repl_lev_1_2, 3])
        | v=…_orint_(or2[6,repl_lev_1_2,2])_((=[6,repl_lev_1_2]))
[144] (in[it_0_lev_1_86, 1])
        | v=…_orint_(or2[6,repl_lev_1_2,2])_((=[6,repl_lev_1_2]))
```

33 occurrences each (one per hashburst). The `((=[6,repl_lev_1_2]))`
payload identifies Branch A (head equality, per Step 1). Step 5
(witness collapse) now has both sides of the existence body at
Branch A validity and can proceed.

---

## Step 5 — p'+l=x admission at Branch A NOT YET REACHED ✗

**Target inference.** With p' (predecessor witness from existence0) and l
(preorder-body disintegration witness from `existence1[N,n,m,+]` →
`(in[k,N]) ∧ (in3[n,k,m,+])`) both live at Branch A validity, the
addition relation `(in3[p',l,x,4])` should be deposited at Branch A
validity — wiring p' into the preorder body so downstream can close
`(in[p,M])`.

**Why needed.** The §4.2 proof of `{0,1}=[0,1]` requires preorder(lo,p)
AND preorder(p,hi) witnesses LIFTED to reference p' (so the Peano
witness-collapse can identify p with i0 or i1 via the α template on
predecessor equality). Without an in3 row pairing p' with a preorder
witness at Branch A, the lift stalls.

**Name fixation (stable across all recent runs):**
- `p` = `repl_lev_1_2` (= anchor slot for the [0,1] inhabitant under test)
- `p'` = `it_0_lev_1_86` (predecessor witness from existence0 at Branch A)
- `l` = `int_lev_1_7` (preorder `p ≤ 1` witness at implication24 scope)
- `k` = `int_lev_1_5` (preorder `0 ≤ p` witness at implication24 scope)
- Branch A validity = `main_boundary_(implication24[1,4,2,6,9])_boundary_orint_(or2[6,repl_lev_1_2,2])_((=[6,repl_lev_1_2]))`

The earlier `_90`-based trail was a red herring — the `_90` rows in
`hashburst_trace.txt` arrive at Branch A via a direct hashmem
implication deposit (not Pass B); the real predecessor witness
entering admission is `_86`.

**BACKWARD P2 auxy IS registered in SE2's encodedMap** (visible after
adding an `encodedMap` filtered dump at hashburst time, commit
). The auxy comes from NaturalNumbers.txt — of the two
directions of `s(a)+b = s(a+b)` stated there (lines 50–57), the
direction with head `(in2[d,e,s])` feeds `makeNormalizedKeysForAdmission`
call #44, whose elem-2 accept produces:

```
markerExpr=(in3[3,1,marker,u_4])
    key={ (in[1,u_1])  (in2[1,2,u_3])  (in3[3,2,4,u_4]) }
    remainingArgs={ 1, 3, 4 }
```

That is: "given `in(a,N)`, `in2(a,a',s)`, `in3(b,a',c,+)`, admit
`(in3[a, b, marker, +])`" — the order the user specified. With
concrete binding {a=p', a'=p, b=l, c=1=anchor 6, marker=0=anchor 2},
all three subkey premises are live in hashmem:

| Auxy element | Concrete | Scope |
|---------------------------|----------------------------------------------------|----------------|
| `in(1, u_1)` | `(in[it_0_lev_1_86, 1])` | Branch A ✓ |
| `in2(1, 2, u_3)` | `(in2[it_0_lev_1_86, repl_lev_1_2, 3])` | Branch A ✓ |
| `in3(3, 2, 4, u_4)` | `(in3[int_lev_1_7, repl_lev_1_2, 6, 4])` (l+p=1) | implication24 ✓ |

Parent/child scopes fire together (by `NameMap::comparable`), so the
implication24-scope statement is usable at Branch A.

**`makeNormalizedKeysForAdmission` replKey bug (fixed, commit
).** The function at `prover.cpp:1532` previously pushed the
marker-source element itself into `replKey`:
```cpp
if (binary[i] || i == index)   // ← bug: `|| i == index`
```
This polluted `lmv.key` with the target expression (the `in3[3,1,4,u_4]`
row being admitted), which downstream consumers `isAdmitted`
(`prover.hpp:2409`) and `admissionMapPropagate` (`prover.cpp:6369`)
would iterate as if a real premise — trivially self-matching.

Fix: drop `|| i == index`. After the fix:
- Verifier: 2224/0 (no regressions).
- encodedMap auxy cleanly stores 3 premises (shown above), no self-row.
- **Step 5 admission still does not fire.** Remaining blocker is
 upstream of the lookup.

**Current blocker: request for the 3-subkey combo is never emitted.**
Trap at `generateEncodedRequestsStatic` emit-site
(`prover.cpp:1840`+) gated on SE2 LB full parent chain + `mc ≥ 3` +
set-membership of the three concrete subkey statements. Target dump:
. Full-run result: **file not
created — zero requests at SE2 LB contain all three statements
simultaneously.** So the encodedMap lookup for the BACKWARD P2 auxy
never runs with the right probe.

The three statements are each individually present in
`body.intEncodedStatements` at comparable scopes (verified in
hashburst trace), but `growBaseCandidates` / mandatory-merge path does
not produce a 3-element request pairing them.

**Next diagnostic.** Widen the trap to `mc ≥ 2` and dump any subset of
the three statements that *does* land in an emitted request. That
will show which of the three gets dropped during candidate growth —
candidates: (a) `filterIntEncodedStatements` excluding a statement
because it doesn't appear in any `normalizedEncodedSubkey`,
(b) `growBaseCandidates` not combining Branch-A pairs with
implication24-scope singles, (c) the specific pair shape never
becoming "mandatory" for the encodedMap entry's first-element
seed.

---

### Step 5 — diagnostic progress (layered trap results)

**Entry-state trigger trap at `performElementaryLogicalStep`** (SE2 LB,
full parent chain): fires exactly once per run.

```
[step5-trigger@entry] #1 mbPtr=0x…12800 intDeltaSize=77 localSize=158 deltaSize=77
  inPrime=1  in2PredSucc=1  in3Preorder l+p=1  in3Preorder p+l=0
```

All three required subkey statements ARE live in both `localEncoded­
Statements` and `localEncodedStatementsDelta` at SE2 LB entry in the
moment the request should fire.

**Batch 2 filter pre/post trap** (around
`makeMandatoryEncodedStatementLists1Static` at `prover.cpp:3614`):

```
[batch2@SE2] #4 mbPtr=0x…12800 intDeltaSize=77 nMsl2=8 preTargets=3 postTargets=3
```

Same SE2 instance (`mbPtr` match), same invocation (`intDeltaSize=77`
matches the entry trigger). **All 3 targets survive
`filterIntEncodedStatements`** and are passed to
`generateEncodedRequestsStatic` as part of an 8-element mandatory list.
Filter is NOT the blocker.

**Decode pitfall encountered.** Earlier runs reported `preTargets=0`
due to the `nameId` vs `originalId` bug: `IntEncodedExpr.nameId`
encodes only the operator name (`"in3"`), not the full expression
(`"(in3[it_0_lev_1_86,int_lev_1_7,6,4])"`). Must decode `originalId`
for full-string comparison. See
the trap-strategy notes.

**Remaining blocker.** Between the 8-element mandatory list (which
contains all 3 of our targets) and the emit-site, no 3-element
request assembling the {inPrime, in2PredSucc, in3Preorder} combo
is produced. The merge path (`growBaseCandidates` + mandatory-seed
cross) is the next layer to instrument.

---

### Step 5 — COMPLETE (commits b6a62e9, 98664d3)

Two changes unblocked Step 5 end-to-end:

1. ** — purity gate relaxed inside OR-branch scopes only.**
 In `checkLocalEncodedMemoryStatic`'s marker branch, `!pure`
 previously short-circuited the admissionMap insert unconditionally.
 Now bypassed when `validityName.find("_boundary_orint_")!=
 std::string::npos` — case-split branches can carry fresh it_/int_
 witnesses that are iteration-bearing but not yet products of
 recursion, which is structurally necessary for §4.2-style ladder
 steps. Everywhere else the purity guard stands to prevent fan-out.
 Also removed the dead non-static `checkLocalEncodedMemory` (269
 LOC).

2. ** — `preEvaluateFromEncoded` dedups secondary vars by
 `argFullId` across the whole request.** Previously the counter
 summed per-element occurrences of iteration-bearing non-prodRec
 args; a single it_ var appearing in 3 premises counted 3 times,
 pushing 4-element implications past the cap of 2 even with only
 2–3 distinct secondaries. Now counts distinct vars — the cap
 reflects unique secondary witnesses, not mentions.

Result: verifier 2234/0 (up from 2224, +10 theorems). Runtime 182s.
The §4.2 Branch A chain fires end-to-end through the P2 step:

- Admission of `(in3[l, p', it_1_lev_1_154, 4])` at Branch A ✓
- Full BACKWARD P2 implication closes with 4-premise request; head
 `(in2[it_1_lev_1_154, 6, 3])` = `s(it_1_lev_1_154) = 1` deposited
 at Branch A ✓
- P3 (s-injectivity) fires at Branch A: `(=[it_1_lev_1_154, 2])`
 and its mirror derived ✓

---

## Step 6 — equality1 substitution across namespace stacks NOT YET REACHED ✗

**Target inference.** From `(=[it_1_lev_1_154, 2])` at Branch A and
`(in3[it_0_lev_1_86, int_lev_1_7, it_1_lev_1_154, 4])` at Branch A,
rewrite the in3 output argument via equality to obtain
`(in3[it_0_lev_1_86, int_lev_1_7, 2, 4])` = `p' + l = 0`. This
feeds P6 (sum-zero split → `p' = 0 ∧ l = 0`), closing the Branch A
case as `p = s(p') = s(0) = 1`.

**Blocker — to be pinpointed.** Both the equality and the target
in3 carry **identical** validity strings (same Branch A scope):

```
main_boundary_(implication24[1,4,2,6,9])_boundary_orint_(or2[6,repl_lev_1_2,2])_((=[6,repl_lev_1_2]))
```

So the known equality-substitution scope rule (same-string namespace
match) should permit firing. And equality1 **is** firing at Branch A
for some rows — new entries in encodedStatements show
`(in3[it_1_lev_1_154, 2, 2, 4])` and `(in3[2, 2, it_1_lev_1_154, 4])`
(products of other substitutions with the same equality). But
`(in3[it_0_lev_1_86, int_lev_1_7, 2, 4])` (the substitution we need)
is not among them.

Candidate reasons to investigate next:
- Secondary-var count on the post-substitution expression
 (`it_0`, `int_7` still present → 2 distinct secondaries, at the
 cap). If the substitution path uses the OLD per-element counter
 (not the dedup'd one that `preEvaluateFromEncoded` now uses), the
 post-substitution expression could be rejected even though its
 distinct count fits.
- Equivalence-class or `equalityClassesMap` filtering at the
 deposit site.
- Duplicate / statementLevelsMap hit that short-circuits re-insert.
- Iteration bounds tied to the in3 arg-slot structure.

**How confirmed.**
- `(=[it_1_lev_1_154, 2])` present at Branch A.
- `(in3[it_0_lev_1_86, int_lev_1_7, it_1_lev_1_154, 4])` present at
 Branch A (identical namespace).
- Search for `(in3[it_0_lev_1_86, int_lev_1_7, 2, 4])` or commuted
 form: **absent**.
- Other substitution products from the same equality DO appear:
 `(in3[it_1_lev_1_154, 2, 2, 4])`, `(in3[2, 2, it_1_lev_1_154, 4])`.
- `(interval[1,4,2,6,9])` at `v=main` still in toBeProved.

**Next move.** Trap the equality1 firing site to see whether the
target substitution is attempted and, if so, which downstream gate
rejects its deposit.

---

### Step 6 — COMPLETE (commit 26e3c45)

**Root cause: Case OR seed deposits omitted `mb.level`** (`prover.hpp:4336,
4341`). Both the negation assumption and the head goal were deposited with
`std::set<int>` (empty), while every other deposit site in the prover
explicitly calls `levels.insert(mb.level)`. Empty seeds meant no
branch-rooted inference ever contributed `{1}` (the LB's level) to
`combinedLevels`. Anchor-scope premises carrying `{0}` (from
`handleAnchor`'s inclusive `0..ky.size` range) won every union, so
every derived Branch-A statement ended up with `statementLevelsMap
entry = {0}`. The equality-substitution gate
`if (maxLevel!= memoryBlock.level) continue;` at
`prover.hpp:5163` rejected the substitution, since the substituted
in3 inherited `{0}` but SE2 LB's `level = 1`.

Diagnostic trail:
- `applyEquivalenceClass` entry trap confirmed both orientations of
 the substituted `(in3[…, 2, 4])` enter `exprLevelsMap`, then are
 dropped with `stage=maxLevel_mismatch maxLevel=0 mb.level=1`.
- `addExprToMemoryBlock` entry trap confirmed all 37 SE2-LB deposits
 for our target exprs carry `involvedLevels={0}` — one
 `disintegration`-tagged root + 36 `implication`-tagged propagations.
- Code audit pinpointed Case OR as the only deposit site omitting
 `mb.level`.

Fix: stamp the two Case OR deposits with `std::set<int>{mb.level}`.

Result:
- Verifier 2234/0 airtight, runtime 183s — no regressions.
- `(in3[it_0_lev_1_86, int_lev_1_7, 2, 4])` and commuted form land
 at Branch A (= `p' + l = 0`).
- `(=[it_1_lev_1_154, 2])` substitution now fires as expected for
 Branch A's target in3.

---

## Step 7 — P6 sum-zero split → `p' = 0, l = 0` ✓

**Inference fired.** With the cancellation theorem
`(>[1,2,4](AnchorPeano[1,2,3,4,5,6])(>[7,8](in3[7,8,2,4])(>[](in[8,1])(=[2,7]))))`
proved on Peano (after the conjecturer relaxations of D-21/D-23 and
the multi-iteration deletion of D-24 made it reachable) and broadcast
into the Gauss prover via the cross-batch shared registry (D-22), the
rule `(in3[a,b,0,+]) ∧ (in[b,N]) ⇒ (=[0,a])` is in `hashMemoryAll`
at v=main as a Peano original. The implication forward chains at
Branch A with concrete binding {a=p'=it_0_lev_1_86,
b=l=int_lev_1_7, 0=anchor 2 ≡ predecessor-of-1 witness
it_0_lev_0_32 at v=main} once `(in3[it_0_lev_1_86, int_lev_1_7,
it_0_lev_0_32, 4])` (= `p' + l = 0`) is at Branch A and
`(in[l, 1])` is live at implication24 scope (parent-of-Branch-A).

**How confirmed.** `hashburst_trace.txt` burst #36 (last burst).
Direct exprOriginMap entry with the cancellation theorem as
antecedent:

```
<- implication
   | (>[1,2,4](AnchorPeano[1,2,3,4,5,6])
       (>[7,8](in3[7,8,2,4])(>[](in[8,1])(=[2,7]))))   (v=main)
   | (AnchorPeano[1,it_0_lev_0_32,3,4,5,6])             (v=main)
   | (in3[it_0_lev_1_86, int_lev_1_7, it_0_lev_0_32, 4])
       (v=…_orint_(or2[6,repl_lev_1_2,2])_((=[6,repl_lev_1_2])))
   | (in[int_lev_1_7, 1])                               (v=…_(implication24[1,4,2,6,9]))
```

`it_0_lev_0_32` is the v=main predecessor-of-1 witness
(`s(it_0_lev_0_32) = 1`, established by the incubator broadcast);
`it_0_lev_0_32 = 0` follows from the α/β template at v=main, so the
cancellation premise reads "`p' + l = 0`" as required.

Step 7 deposits at Branch A (verified in burst #36):
- `(=[2, it_0_lev_1_86])` = `0 = p'` — predecessor witness equals 0 ✓
- `(=[2, int_lev_1_7])` = `0 = l` — preorder upper-bound witness equals 0 ✓
- Plus several derivatives (`(=[2, it_1_lev_1_154/_168/_172/_176/_200/_204])`)
 from downstream equality1 substitutions.

**toBeProved went 6 → 4.** Both OR-branch heads closed (no longer
in `toBeProved`):
- `(=[2, repl_lev_1_2])` (Branch B head, `p = 0` case) — closed.
- `(=[6, repl_lev_1_2])` (Branch A head, `p = 1` case) — closed via
 the chain `p' = 0` (Step 7), `(in2[it_0_lev_1_86, repl_lev_1_2, 3])`
 = `s(p') = p` (Step 4), so `p = s(0) = 1`.

---

## Step 8 — OR-integration: collapse closed branches into `(in[p, 9])` ✓

**Target inference.** With Branch A head `(=[6, p])` closed under its
scope (a single branch firing is sufficient — `cleanUpOrIntegrationBranches`
wipes the sibling), OR-introduction on the disjunction `or2[6, p, 2]`
fires at the OR-branches' parent scope (implication24 in legacy
numbering, implication26 under the post-D-22 cross-batch shared
registry), and implication21
(`or2 ⇒ (in[p, M])`, the ES2 backward direction) closes
`(in[p, M])` at that scope.

**Why this matters for the §4.1 theorem.** `(in[p, M])` at the
parent scope is the conclusion of interval body's *backward*
direction (`preorder(0,p) ∧ preorder(p,1) ⇒ (in[p, M])`). It is one
of the two conjuncts of `(interval[1,4,2,6,…])` at v=main. With Step 8
closed, the backward conjunct of the §4.1 theorem's body is done —
the proof direction the user describes as "x ∈ {0,1} → x ∈ [0,1]"
(meaning: at this LB the theorem `EnumerationSet2 ⟹ interval` is
being proved, and Steps 1–8 are the work for the backward conjunct
of interval's body inside that theorem).

---

### Step 8 — COMPLETE (commit, D-30)

**Root cause: OR emission scope hard-coded to `"main"`.** At
[`prover.cpp:4419`](../../../GL_Quick_VS/GL_Quick/src/prover.cpp), the
`addExprToMemoryBlock` call that emits the wrapping OR after a
branch's head proves passed `"main"` as the destination scope.
Diagnostic trail:

- Trap at `prover.cpp:4404` showed Branch A head `(=[6, repl_lev_1_2])`
 arriving at Branch A scope with `status=1` and `inToBeProvedHere=1`.
 All gate conditions met. The line-4419 emit fired.
- The OR `(or2[6, repl_lev_1_2, 2])` then landed at `v=main` (per
 the hard-coded destination), and `implication21` produced
 `(in[repl_lev_1_2, 15])` at `v=main`.
- The toBeProved goal `(in[repl_lev_1_2, 15])` lived at
 `_boundary_(implication26[1,4,2,6,15])` — the implication's
 hypothetical scope, NOT at `main`. The deposit at `v=main` did
 not match (toBeProved lookup is exact-scope-keyed at
 `prover.cpp:4404`).
- Step 8 stalled at `toBeProved=4` regardless of how many bursts ran.

The hard-coded `"main"` was a latent bug from when ORs lived only at
the top-level scope (Peano OR-branching, `or0[7,2,1,3]` directly at
v=main). Once OR-branching moved into the body of hypothetical /
integration scopes — the FTA-ladder pattern starting with rung 1 —
the destination became wrong.

**Fix.** Compute the OR-emit scope by peeling the last
`_boundary_<payload>` segment off the proving branch's `validityName`:

```cpp
size_t lastBoundary = validityName.rfind(NameMap::BOUNDARY_STR,
                          std::string::npos, NameMap::BOUNDARY_LEN);
const std::string orEmitScope =
    (lastBoundary == std::string::npos)
        ? std::string("main")
        : validityName.substr(0, lastBoundary);
```

For top-level ORs (parent IS main), `orEmitScope == "main"` —
semantics unchanged for Peano OR-branching. For ORs nested under a
hypothetical scope, the OR lands at the wrapping hypothetical scope,
where the implication21-derived `(in[p, M])` then matches the goal.

**Result.**
- Burst #50 toBeProved drops 4 → **3**: the `(in[repl_lev_1_2, 15])`
 entry at implication26 scope is matched and erased.
- Full main.py: 5 batches complete, **35 458 verifier checks /
 0 failures**, runtime 1183 s (vs 1332 s pre-fix — Step-8-closure
 cleanup pays back ~10 % wall-clock).
- IncubatorGauss1 batch still saves 0 theorems because Steps 9, 10
 remain open (forward conjuncts of interval body need
 OR-disintegration mechanism, see Step 9 below).

---

## Step 9 — Forward direction of interval body (`(in[p, M]) ⇒ preorders`)

Split into two sub-steps, one per forward conjunct. Compiled-name
shift: pre-D-22 doc named these implication22/implication23; current
post-cross-batch-registry trace names them implication24/implication25
(with M=15). Roles unchanged: lower-bound forward direction =
`(in[p, M]) ⇒ preorder(0, p)`; upper-bound forward direction =
`(in[p, M]) ⇒ preorder(p, 1)`.

### Step 9a — Lower-bound forward conjunct (impl24) ✓

**Goal.** Establish, in **both** OR-disintegration branches at the
impl24 boundary scope (one per case `=0` and `=1` of the universal
hypothesis), the two preorder bounds on the impl24 replacement
variable `repl_lev_1_0`:

- `(preorder[1,4,2,repl_lev_1_0])` — `0 ≤ repl_lev_1_0`.
- `(preorder[1,4,repl_lev_1_0,6])` — `repl_lev_1_0 ≤ 1`.

Both deposited at the ordis sub-scope
`main_boundary_(implication24[15,1,4,2])_boundary_ordis_(or2[2,repl_lev_1_0,6])_((=[…]))`.
This `0 ≤ x ≤ 1` matrix in every branch is the precondition for
OR-convergence to lift the lower-bound forward conjunct one scope
up to the immediate impl24 boundary, where the toBeProved waits.

**Why this is the §4.1 forward lower conjunct.** Interval body's
forward direction is `(in[p, M]) ⇒ preorder(0,p) ∧ preorder(p,1)`.
The impl24 scope holds the lower-bound forward sub-implication.
`implication20` (ES2-forward, `(in[u_p, u_M]) ⇒ (or2[u_2, u_p, u_6])`)
fires at impl24 on the universal hypothesis `(in[repl_lev_1_0, 15])`,
depositing `(or2[2, repl_lev_1_0, 6])` and minting two
`_boundary_ordis_` branches. Under each branch,
[D-33](../../agentic_swdd/40_decisions.md#d-33)'s descendant-class direction
rewrites main-scope `(preorder[…])` facts using the branch-scope
equivalence class (`{2, repl_lev_1_0}` or `{6, repl_lev_1_0}`), and
[I-25](../../agentic_swdd/30_invariants.md#i-25)'s single-channel routing
threads those cross-scope deposits through the kernel's discharge
loop. See also [D-31](../../agentic_swdd/40_decisions.md#d-31) (initial OR-disint
unblock) and [D-32](../../agentic_swdd/40_decisions.md#d-32) (sharpened gate).

**How confirmed.** Last burst (#50) of the ES2 LB hashburst trace
under the trap chain `(EnumerationSet2[2,6,15]) → (AnchorIncubator
[1..14]) → ""`:

| Expression | `_((=[2,repl_lev_1_0]))` branch | `_((=[6,repl_lev_1_0]))` branch |
|---|---|---|
| `(preorder[1,4,2,repl_lev_1_0])` | ✓ entry [3749] | ✓ entry [3757] |
| `(preorder[1,4,repl_lev_1_0,6])` | ✓ entry [3756] | ✓ entry [3783] |

All four cells deposited. The original main-scope `(preorder[1,4,2,6])`
is left untouched. Verifier 0 failures across all batches.

### Step 9b — Upper-bound forward conjunct (impl25) ✓

**Goal.** Establish, in **both** OR-disintegration branches at the
impl25 boundary scope, the two preorder bounds on the impl25
replacement variable `repl_lev_1_1`:

- `(preorder[1,4,2,repl_lev_1_1])` — `0 ≤ repl_lev_1_1`.
- `(preorder[1,4,repl_lev_1_1,6])` — `repl_lev_1_1 ≤ 1`.

Both deposited at the ordis sub-scope
`main_boundary_(implication25[15,1,4,6])_boundary_ordis_(or2[2,repl_lev_1_1,6])_((=[…]))`.
Same shape as 9a — `0 ≤ x ≤ 1` in every branch is the precondition
for OR-convergence to lift the upper-bound forward conjunct one
scope up to the immediate impl25 boundary, where the toBeProved
waits.

**Why this is the §4.1 forward upper conjunct.** The impl25 scope
holds the upper-bound forward sub-implication. With `repl_lev_1_1`
ranging over M = ES2 = {0, 1}, the witness is non-uniform: `k = 1`
when p = 0 (so `0 + 1 = 1`), `k = 0` when p = 1 (so `1 + 0 = 1`).
Unlike Step 9a's lower bound (where `k = p` works uniformly),
Step 9b structurally requires the case-split. Same mechanism reaches
this state as 9a — `implication20` deposits `(or2[2, repl_lev_1_1, 6])`,
two ordis branches mint, [D-33](../../agentic_swdd/40_decisions.md#d-33)'s
descendant-class direction + [I-25](../../agentic_swdd/30_invariants.md#i-25)'s
single-channel routing produce the per-branch preorder rewrites and
propagate them through the kernel's discharge logic.

**How confirmed.** Last burst (#50) of the ES2 LB hashburst trace:

| Expression | `_((=[2,repl_lev_1_1]))` branch | `_((=[6,repl_lev_1_1]))` branch |
|---|---|---|
| `(preorder[1,4,2,repl_lev_1_1])` | ✓ entry [3752] | ✓ entry [3759] |
| `(preorder[1,4,repl_lev_1_1,6])` | ✓ entry [3758] | ✓ entry [3786] |

All four cells deposited. The original main-scope `(preorder[1,4,2,6])`
is left untouched. Verifier 0 failures across all batches.

---

### Steps 9a + 9b — OR-convergence at impl24 / impl25 boundary ✓ (D-34)

Both 9a and 9b reach the same intermediate state. In every ordis
branch of both impl24 and impl25, the case-split has produced both
bounds on the replacement variable:

| | `_((=[2, x]))` branch (x = 0) | `_((=[6, x]))` branch (x = 1) |
|---|---|---|
| `(preorder[1,4,2, x])` (= `0 ≤ x`) | ✓ | ✓ |
| `(preorder[1,4, x ,6])` (= `x ≤ 1`) | ✓ | ✓ |

— with `x = repl_lev_1_0` for impl24 and `x = repl_lev_1_1` for
impl25. So under both case-splits we have `0 ≤ x ≤ 1` established
in **every** ordis branch of **both** sub-implications.

[D-34](../../agentic_swdd/40_decisions.md#d-34) implements the convergence
mechanism: kernel-level `ordisMerge` per-deposit bookkeeping in
`addExprToMemoryBlockKernel`'s post-`addStatement` loop, with promotion
routed through `internalMailIn` (the `revisitRejectedIntegration2`
revival channel) for the next-hashburst absorb. Per-branch cleanup
of the converged expression via `removeExpressionFromMemoryBlock`
keeps branches alive for further convergences.

**Final post-convergence state at `v=main_boundary_(implicationNN[…])`** in
the latest full-pipeline run trace:

| Expression | impl24 boundary (clean) | impl25 boundary (clean) |
|---|---|---|
| `(preorder[1,4,2,X])` (`0 ≤ X`) | 50 occurrences ✓ | 46 occurrences ✓ |
| `(preorder[1,4,X,6])` (`X ≤ 1`) | 0 — see note | 50 occurrences ✓ |

Three of four lift cleanly to the immediate parent. The fourth
(`(preorder[1,4,repl_lev_1_0,6])` at impl24-down) lands one step
further up at `v=main`: the same orSignature `(or2[2,repl_lev_1_0,6])`
exists at two stack positions (impl24-internal AND top-level), the
top-level OR-convergence promotes to `v=main` first, then the
impl24-internal convergence's deposit gets dedupe'd by Site F's
ancestor scan at [`prover.cpp:4800-4804`](../../../GL_Quick_VS/GL_Quick/src/prover.cpp)
because the fact is already live at the strict ancestor v=main.
Functionally equivalent — comparable-scope inheritance makes the
v=main entry visible at impl24 and at every other descendant.

---

## Step 10 — `(interval[1,4,2,6,15])` at `v=main` ✓ — §4.1 closed

**Closure path.** Once Steps 8, 9a, and 9b complete the body
conjuncts (backward via OR-integration at Step 8; forward lower via
9a; forward upper via 9b — all three OR-convergence lifts firing
under D-34), interval-body integration closes
`(interval[1,4,2,6,15])` at v=main inside the contradiction LB
seeded by the §4.1 forward conjecture. The contradiction discharges,
saving the §4.1 forward direction theorem.

**Verified — full main.py run, exit 0, verifier 2750 / 0 airtight.**
The IncubatorGauss1 batch saves 2 theorems (vs 0 baseline pre-D-34),
including the §4.1 forward direction:

```
(>[1,2,4,6](AnchorIncubator[1,2,3,4,5,6,7,8,9,10,11,12,13,14])
  (>[15](EnumerationSet2[2,6,15])(interval[1,4,2,6,15])))
```

Both Gauss summation copies remain in `proved_theorems.txt` (no main-batch
regression). `or convergence` appears 106 084 times in the hashburst
trace — `ordisMerge` fires extensively across the batch.

---

### §4.2 (reverse) — `interval ⟹ EnumerationSet2` — open

`(interval[1,4,2,6,15]) ⟹ (EnumerationSet2[2,6,15])` is the second
directed conjecture for rung 1 (theorems.txt line 342). It is NOT
yet in `files/incubator/theorems/proved_theorems.txt`. The §4.2
proof plan (per [`proof_01_set_eq_interval.md`](./proof_01_set_eq_interval.md))
involves:

1. Splitting `p ∈ [0,1]` into the `p = 0` case (trivial — `p ∈ {0,1}`
 immediate) and the `p = s(p')` case (via Peano predecessor split).
2. In the `p = s(p')` case, using `p + l = 1 = s(0)` (from `p ≤ 1`),
 `s(p') + l = s(0)`, P2 + P3 give `p' + l = 0`, P6 (sum-zero split)
 forces `p' = 0` and `l = 0`, so `p = s(0) = 1`, hence
 `p ∈ {0,1}`.

This is the next rung-1 frontier. Steps 1–8 (Peano-witness machinery
under contradiction LB, with predecessor-existence + sum-zero split
infrastructure) are already in place from §4.1's chain; the remaining
work is wiring the rung 1 reverse-direction conjecture's contradiction
LB to use them.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
