<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
Contributions require CLA — see CONTRIBUTING.md.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Antisymmetry `a≤b ∧ b≤a ⇒ a=b` (Peano main) — current proof state

Append-only. Newest entry last. Human-notation proof:
[`proof_02_2_antisymmetry.md`](./proof_02_2_antisymmetry.md). Parent
rung state: [`current_proof_state.md`](./current_proof_state.md).

## 2026-07-28 — document created; stall investigation begins

Status inherited from the parent state document's 2026-07-27 entries:
the conjecture is emitted (Peano pool, `D-246`
config extension), all three chain lemmas are proved in the same batch,
two determinism-checked full runs completed at 134,051 checks / 0
failures, and antisymmetry itself did NOT close — a stall, since
mid-batch broadcast makes the same-run cancellation lemma available to
the antisymmetry LB within run 1.

### Target and LB chain

Target theorem (byte-exact, Peano conjecture pool):

```text
(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8](preorder[1,4,7,8])(>[](preorder[1,4,8,7])(=[7,8]))))
```

Premise-LB chain, derived from `addTheoremToMemory`'s chain walk
(`ce::disintegrateImplication` splits the binder chain; plain compact
operators pass `precompileStructuralOperators` untouched):

```text
root (empty exprKey, null parent)
  └─ (AnchorPeano[1,2,3,4,5,6])
       └─ (preorder[1,4,7,8])
            └─ (preorder[1,4,8,7])        ← the antisymmetry LB
```

The innermost LB holds the goal `(=[7,8])` at `main` in `toBeProved`;
the whole four-row antisymmetry conjecture family shares this chain
(same premises, different heads), so all family goals sit on the one
LB. The sacred hashburst dump is retargeted here (maintainer-directed,
Rule 14).

### Compiled-name glossary

Anchor slots: `1=N, 2=0, 3=s, 4=+, 5=*, 6=1`. In this LB's statements
the theorem variables `a`, `b` appear literally as the argument names
`7`, `8`.

| Compact form | Meaning |
|---|---|
| `preorder[1,4,x,y]` | `x ≤ y` (defined: `∃p∈N: x+p=y`) |
| `in3[x,y,z,4]` | `x + y = z` |
| `in3[x,y,z,5]` | `x * y = z` |
| `in2[x,y,3]` | `s(x) = y` |
| `in[x,1]` | `x ∈ N` |
| `it_*_lev_*` / `int_lev_*` | existence-witness names minted by disintegration |
| `repl_lev_*` | universal-instantiation variable names |

Chain lemmas (byte-exact, `files/theorems/theorems.txt`, all proved in
this batch):

```text
(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,4])(>[10,11](in3[8,10,11,4])(>[12](in3[12,10,7,4])(in3[11,12,9,4])))))
(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,4])(>[](=[8,9])(=[2,7]))))
(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8](in3[7,8,2,4])(>[](in[8,1])(=[2,7]))))
```

— relational associativity, cancellation, zero-sum. Totality /
single-valuedness / typing of `+` come from `fXYZ[+,N,N,N]` inside the
`NaturalNumbers` definition; the zero-identity axiom pair closes the
final step.

### Expected trace, step by step (grep targets for the dump walk)

Writing `w1`, `w2` for the two premise witnesses and `w3` for the
demanded sum output (concrete names will be `it_`/`int_`-family):

| § | Expected at the antisymmetry LB's `main` | Producer |
|---|---|---|
| §3.1a | `(in3[7,w1,8,4])`, `(in[w1,1])` | status-0 disintegration of `preorder[1,4,7,8]` at the PARENT LB, arriving via ancestor mail |
| §3.1b | `(in3[8,w2,7,4])`, `(in[w2,1])` | status-0 disintegration of the LB's own `preorder[1,4,8,7]` |
| §3.2 | `(in3[w1,w2,w3,4])` | totality of `fXYZ[+,N,N,N]` firing on `w1,w2 ∈ N`; the output witness `w3` must mint (demanded-output admission) |
| §3.3 | `(in3[w3,8,8,4])` | associativity rule firing on §3.1a + §3.2 + §3.1b |
| §3.4 | `(=[2,w3])` i.e. `w3 = 0` | cancellation rule firing on §3.3 |
| §3.5 | class `{w3, 0}` rewrites §3.2 to `(in3[w1,w2,2,4])`; then `(=[2,w1])` | zero-sum rule firing |
| §3.6 | class `{w1, 0}` rewrites §3.1a to `(in3[7,2,8,4])`; goal `(=[7,8])` closes | zero-identity axiom |

### Watchpoints (candidate stall mechanisms, ordered by position in the chain)

1. **Typing of the theorem variables.** The conjecture has no `in`
 premises (`preorder` is config-barred from co-occurring with `in`),
 so `7 ∈ N` / `8 ∈ N` exist only if the `fXYZ` typing clause fires
 on the witnessed sums. Totality (§3.2) needs only `w1, w2 ∈ N`,
 which the preorder definition supplies directly.
2. **§3.2 demanded-output minting.** The parent state document's
 source-audit entry records that a locally derived compact totality
 existence reaches `disintegrateExpr2` but Pass B admits neither
 witness body (the `¬(2≤1)` direct-reductio barrier). The
 maintainer's strategy expects the antisymmetry case to differ
 because an installed rule (associativity, two premises live)
 demands exactly the output slot. Whether that consumer-side
 admission actually exists on this path is the first thing the trace
 must decide.
3. **§3.3 collision pattern.** Associativity's needed instance binds
 template-distinct slots `9` and `12` both to `8` (= `b`): its
 registered all-distinct normalized request key cannot match the
 collapsed instance (the rung-2.1 first-entry mechanism) unless an
 equality-class alias of `8` launders the repetition (the ES2
 witness-alias route: `8+0=v`, zero identity `v=8`, class `{8,v}`
 rewriting one premise). `allow_multiplication` is `false` in every
 main batch, so no collapsed rule copies exist.
4. **§3.4 double wall.** The cancellation instance repeats `b` the
 same way AND needs the reflexive equality premise `(=[8,8])`,
 which never exists as a statement. Firing therefore requires the
 laundered form `in3[w3,v,8,4]` plus the alias-class equality
 `(=[v,8])` as a live statement.
5. **Broadcast timing.** All three lemmas are same-run proofs; each
 arrives at the antisymmetry LB only after its own LB proves it and
 the commit barrier broadcasts. If the chain's tail lemmas land near
 the batch's final bursts, the remaining budget may not cover
 §3.3–§3.6 even with every mechanism working.

### Next step

One directed full-pipeline run with the dump on the antisymmetry LB;
walk the trace §-by-§ against the table above; find the last § present
and the first § absent; localize the mechanism; classify coding bug vs
architecture gap. Diagnosis only (Rule 8).

## 2026-07-28 — directed run traced: the stall is the collision-pattern gap on the DEMAND side; the parked witness `k+l=m` is never revived because no admission key can ever be written

Evidence:  (full Windows pipeline,
exit 0, verifier airtight 134,051 checks / 0 failures, `theorems.txt`
unchanged at 51 rows) with the preserved sacred trace
 (42 dumps: the LB
runs all 21 Peano bursts, ends ACTIVE at a true fixpoint — EXIT #21
equals ENTRY #21 at 260 statements, 165 rules, `toBeProved` frozen at 1
the entire run).

### §-walk result — frontier at §3.2

1. **§3.1 complete by EXIT #1.** Own premise witness sum
 `(in3[8,int_lev_2_1,7,4])` (b+l=a) at `main` from ENTRY #1; the
 parent's `(in3[7,int_lev_1_1,8,4])` (a+k=b) by EXIT #1; both
 witness typings live; commuted variants and the typings
 `(in[7,1])`, `(in[8,1])` follow.
2. **§3.2's compact existence derives by EXIT #2** —
 `(existence1[1,int_lev_1_1,int_lev_2_1,4])` ("∃w: k+l=w") at
 `main` — and its disintegration DOES propose the witness body: the
 final `rejectedMap` (51 rows) holds
 `key=(in3[int_lev_1_1,int_lev_2_1,marker,4])` with the rejected
 body `(in3[int_lev_1_1,int_lev_2_1,it_0_lev_2_12,4])`, parked for
 revival. **The witness never mints**: no positive `k+l` sum ever
 enters `encodedStatements` in any burst.
3. **§3.3–§3.6 never start.** No `m+b=b` shape, no `(=[2,…])`
 cancellation product, goal `(=[7,8])` never closes. The LB idles
 19 bursts at the fixpoint.

### Mechanism — three interlocking walls, all evidenced

1. **The revival loop's only key-writer is collision-blocked.** Pass B
 (`prover.cpp` `Pass B: New Variable Admission`) rejects the witness
 body because `admissionMap` holds no demand key — and it holds none
 in ALL 42 dumps (`admissionMap (0)` in every burst). The only
 writer of that key is a marker-LMV completion
 (`checkLocalEncodedMemoryStatic`'s marker branch, staged per I-68):
 marker LMVs are installed hash rules and their completion rides the
 SAME normalized-key exact match as any firing. All 129 installed
 markers were enumerated from the dump: every additive
 `in3[…,marker,4]` entry keys on two `in3` premises sharing exactly
 ONE template variable (plus four commuted-pair variants sharing
 two). The live pair `a+k=b`, `b+l=a` chains through BOTH `a` and
 `b`, so any assignment binds two template-distinct slots (the first
 premise's output, the second's operand) to one name — the
 collision-collapsed instance key misses the all-distinct registered
 key, the marker never completes, the admission key is never
 written. This is the rung-2.1 first-entry gap (2026-07-27) verbatim,
 now proven to gate the DEMAND side, not just direct firings. (The
 commuted-pair markers are structurally dead for `+`: their two
 output slots always carry the equal sums.)
2. **The alias escape route has no demander.** The ES2-style
 laundering seed exists (`(existence1[1,8,2,4])` — "∃w: b+0=w" — at
 `main`, would give `w=b` via the zero identity, class `{8,w}`, then
 all-distinct rewritten variants), but its witness body is parked in
 the same `rejectedMap` and its natural demander — the two-premise
 zero-identity rule — registers NO marker: marker registration is
 gated by `baselineClassicQualifies` (`prover.hpp`), which requires
 MORE than `minNumOperatorsKey` (= 2, config default) `in3` premises
 AND an `in3[` head (or a `minLenLongKey`-length key). Equality-headed
 and small rules never demand; the dump confirms zero markers with a
 key of fewer than three premises.
3. **The wall repeats downstream.** Even with `k+l=m` minted, the full
 associativity firing needs the same doubled binding (its instance
 assigns `b` to two template-distinct slots), and §3.4's
 cancellation instance additionally needs the reflexive equality
 premise `(=[8,8])`, which never exists as a statement. Without
 alias laundering, every remaining chain step is
 collision-unreachable.

Watchpoint 5 (lemma broadcast timing) is exonerated: all three chain
lemmas are installed as rules at this LB (verified in
`overallHashMemory.originals` — the 4-in3 associativity family,
cancellation `(in3[7,8,9,4]) (=[8,9]) → (=[2,7])`, zero-sum
`(in3[7,8,2,4]) (in[8,1]) → (=[2,7])`), and 19 idle bursts remained
after the frontier froze.

### Classification — architecture gap

Every mechanism follows its documented contract; nothing miscomputes.
The demand-driven witness-revival design (existence → body parked
under a marker-form key → marker completion writes the admission key →
revival mints) cannot close ANY chain whose live facts double-link the
demanding rule's premises — which antisymmetry's premise pair does
inherently (`a` and `b` each occur in both sums). Same root as the
rung-2.1 collision finding; the new fact is its reach: it gates demand
generation, and the marker qualification gate independently excludes
the small equality-headed rules that could open the alias route.

### Candidate repair directions (maintainer's decision, Rule 8 — no code changed)

- **Matching-side collapse (the first entry's recommended direction,
 now with sharper scope):** register slot-collapsed variants of
 normalized request keys AND marker subkeys at rule install — install-
 time multiplication restricted to collapse-only partitions (fuse
 unifiable slot pairs, drop the then-trivial equality premise, no
 `u_`-parameter fusion). The flat multiplication trial showed the
 collapsed copies carry the entire fix while the full partition
 fan-out of the external-theorem family carries essentially the
 entire cost; a collapse-only variant is the cheap half. Covers §3.2
 demand, §3.3 firing, and §3.4 in one mechanism.
- **Config-side trials (present-but-disabled mechanisms):**
 (a) `allow_multiplication: true` for the Peano batch — proven to
 produce exactly the needed collapsed copies, cost in Peano main
 unmeasured (prohibitive flat-on in IncubatorGauss3; the buildStack
 origin defect on that path is still open); (b) widen the marker
 qualification (lower `minNumOperatorsKey` and/or relax the
 `in3`-head condition) so the zero-identity rule demands `b+0=w`,
 opening the ES2-style alias-laundering route — each later chain step
 then needs re-tracing under the alias classes.
- **Producer-side:** conjecturer template for the collapsed ground
 shape (`x+c=c ⇒ x=0` family) — narrow, leaves the generic gap.

## 2026-07-28 — maintainer proposal analyzed: a variable copy triggered by the antisymmetry conjecture shape; mechanically closes the collision wall end-to-end; second wall = the distinct-secondary cap at the associativity firing

Maintainer proposal (session dialogue): the marked-expression demand
did not fire because `a` occurs twice in the needed instance; so mint a
COPY of one argument, triggered by the conjecture's distinctive form —
two identical relation premises with reversed arguments and an equality
head — copying the argument that comes first left-to-right. Analysis
only; no code changed (Rule 8).

### The mechanism already exists — the proposal is a fourth trigger site

`(=[Y,Y_copy])` dead-end free axioms with the `variableCopy` origin are
an established, verifier-sanctioned pattern: three emission sites
(`checkNecessityForEquality`, `disintegrateExprHypothetically`,
`reactToHypo`), one shape-based checker (`verifier.py`
`check_variable_copy` — structure + empty origin + every use of
`Y_copy` must trace back to the declaration), and the verifier's own
docstring states the sites "differ only in WHEN the prover chose to
introduce the copy." `checkNecessityForEquality` is in fact already a
collision-triggered copy minter — its firing condition is literally
"after substitution the variable occurs twice where it occurred once"
(`countAfter >= 2 && countBefore == 1`) — but its trigger shape
(input statement matching a one-constant rule head) does not cover the
two-premise double-link of the antisymmetry instance. Adding the
conjecture-shape trigger requires NO verifier change (the check is
shape-based; the three-site enumeration is prose).

### Trigger and deposit design (the analyzed variant)

Detection at `addTheoremToMemory`, on the disintegrated chain: two
chain elements with the same operator whose argument lists are equal
except two swapped positions, head an equality over exactly those two
variables. Cheap, static, deterministic — and it generalizes verbatim
to every future antisymmetry conjecture (the `divides` antisymmetry of
the upper FTA rungs has the identical shape and will hit the identical
wall).

Deposit at the innermost premise LB, status 0, scope `main`:

1. `(=[7,7_copy])` — the `variableCopy` dead-end axiom;
2. `(preorder[1,4,8,7_copy])` — the second premise with the copy
 substituted at the recurring argument, provenance `equality1` from
 the original premise plus the copy equality.

Depositing the substituted premise explicitly (not just the bare
equality) is load-bearing: the class `{7, 7_copy}` has canonical `7`
("7" precedes "7_copy" lex), so relying on equivalence-class variant
generation to produce the `7_copy` forms would depend on rewrite
direction semantics; the explicit status-0 deposit instead rides the
proven assumed-fuel disintegration path and mints its own witness
(`8 + l' = 7_copy`) in burst 1.

### The §-walk with the copy — every instance becomes all-distinct

Facts: `in3[7,k,8,4]` (original, k = its witness) and
`in3[8,l,7_copy,4]` (copy premise, l = its witness).

1. **Marker completes.** Canonical associativity key
 `{(AnchorPeano) (in3[7,8,9,4]) (in3[12,10,7,4])}` matches with
 t7=8, t8=l, t9=7_copy, t12=7, t10=k — all distinct; secondaries
 {l,k} = 2 ≤ cap. The demand key `(in3[l,k,marker,4])` is written —
 exactly the parked `rejectedMap` key; the witness body revives and
 `l+k=m'` mints.
2. **Associativity fires** → head `in3[m',7,7_copy,4]` (`m'+a=a_copy`)
 — all-distinct deposit. **But see the second wall below.**
3. **Cancellation fires** on `in3[m',7,7_copy,4]` plus `(=[7,7_copy])`
 — the copy equality does double duty: it launders the collision AND
 is itself the equality premise, dissolving the reflexive-`(=[8,8])`
 problem. Secondaries {m'} = 1. Yields `(=[2,m'])` — m'=0.
4. **Zero-sum:** class `{0,m'}` rewrites `l+k=m'` to `l+k=0` (the
 standard toward-numeral direction, exercised on every prior rung);
 fires with `(in[k,1])` → `(=[2,l])` — l=0. Secondaries {l,k} = 2.
5. **Zero identity:** class `{0,l}` rewrites the copy premise's sum to
 `in3[8,2,7_copy,4]` (`b+0=a_copy`); fires with `(in[8,1])` (live in
 the trace) → `(=[8,7_copy])`. Zero secondaries.
6. **Goal closes:** `{7, 7_copy, 8}` merge; the goal `(=[7,8])` closes
 through the standard equality-class closure.

### The second wall — the distinct-secondary cap at step 2

`requestGatesPass` counts DISTINCT argument names carrying an
iteration number across the whole request; the associativity FIRING
carries {k, l, m'} = 3 > `maxNumberSecondaryVariables` (2), and the
`maxNumberSecondaryVariablesOrint` widening cannot apply at `main`
(its scope condition is an `_orint_` branch). The marker completion
(2), cancellation (1), zero-sum (2), and zero-identity (0) firings all
fit; ONLY step 2 exceeds the cap. The copy name itself does not count
(no iteration number). This is the parent document's original
"anticipated first boundary" resurfacing at the theorem LB instead of
the reductio: the options are the batch-config raise
(`maxNumberSecondaryVariables: 3` for Peano — knob exists; batch-wide
fan-out risk untested in Peano main) or a maintainer-designed scoped
widening (a third scope condition alongside the `_orint_` one).

### Verdict

The proposal is mechanically sound and surgical: one new trigger for
an existing, verifier-accepted mechanism; cost per antisymmetry-shaped
conjecture is one name, two statements, one extra status-0
disintegration; no matching-engine change, no multiplication, no
verifier edit. It closes the collision wall completely for this rung
and for every future antisymmetry conjecture. It does NOT by itself
close the rung: the distinct-secondary cap at the associativity firing
is the measured second wall and needs its own (config-shaped or
scoped) decision. Both walls are now precisely mapped; implementation
awaits maintainer approval (Rule 8).

Open design details for the decision: the exact origin story of the
substituted premise (explicit `equality1` deposit at load versus
letting the class engine derive it), a dedup guard analogous to
`checkNecessityForEquality`'s `hasExistingEquality`, and whether the
trigger also fires for the mirrored family rows (same chain, same LB —
the deposit dedups naturally).

## 2026-07-28 — two corrections to the previous entry (maintainer + source): the bare copy equality suffices, and the second wall does not exist; `force_lev` idea analyzed and parked

### Correction 1 (maintainer): same-tier class members apply both ways

The previous entry claimed the class `{7, 7_copy}` has canonical `7`
and that variant generation toward `7_copy` is therefore not
guaranteed, making an explicit substituted-premise deposit
load-bearing. Wrong: neither name is `it_`/`int_`-form, both sit in
the ordinary tier, and the class engine applies both equally —
substitution variants generate in both directions. The minimal deposit
is therefore the maintainer's original proposal: the bare
`(=[7,7_copy])` variable-copy axiom, nothing else. The needed
all-distinct fact is generated at the SUM level — the class rewrite of
the forced witness sum `in3[8,int_lev_2_1,7,4]` inserts
`in3[8,int_lev_2_1,7_copy,4]` directly into the statement registries
(no re-disintegration, no second witness; the compact `preorder`
variant never needs to exist).

### Correction 2 (source): `int_lev_*` witnesses are already cap-exempt — there is no second wall

The previous entry's cap arithmetic treated k and l (`int_lev_1_1`,
`int_lev_2_1`) as counted secondaries. The encode-time parse
(`memory.hpp`, the `IntEncodedExpr` argument loop) sets
`argIteration` ONLY for names matching `it_<N>_lev_…`; an
`int_lev_*` name fails the prefix test at its second character and
carries `argIteration = -1`, which `requestGatesPass` skips. The
forced-disintegration witnesses are therefore ALREADY excluded from
the distinct-secondary count — exactly the "treated as normal"
semantics the maintainer's `force_lev` idea targets. The associativity
firing counts only the demanded witness m' (`it_0_lev_*`): 1 ≤ 2. The
standard cap holds at every step of the chain; no config raise, no
scoped widening, no new name family is needed for this rung.
(Consistency check: rung 2's `_orint_` widening to 3 was needed for
firings carrying three distinct `it_`-family descent witnesses — all
speculative existence products, none forced — so the historical
evidence matches this reading.)

### The `force_lev` proposal (maintainer, this session) — opinion: principled, currently redundant; park it

The idea: forced (assumed-fuel) disintegration mints its own name
family `force_lev_<level>_<id>`, treated as ordinary everywhere, so
premise witnesses stop being budgeted like speculative search objects.
The taxonomy is right — "7"/"rec"-like task constants versus budgeted
engine speculations, with the givens-versus-speculations line drawn at
exactly the right place — but at the request cap that line is ALREADY
implemented: `int_lev` names are cap-invisible today (Correction 2).
A new family would be a no-op for the cap. What it WOULD change are
the four surfaces where `int_lev` is still witness-like:

1. mail eligibility (`allowedForMail` gates `int_lev`-carrying
 statements on the `canBeSentMarkerIds` set — though this run's
 trace shows the premise witness sums DID mail parent→child through
 the existing `canBeSentSet` registration);
2. Pass-B admission (the `matchIntLevId` branch requires
 `isAdmittedIntegration` for derived arrivals);
3. `reactToHypo`'s all-`int_lev`-class-mates trigger predicate;
4. the equivalence-class name tier (`kindOf`).

Each of those gates exists for containment and would need its own
sizing run if opened. Recommendation: park `force_lev` until a
concrete rung stalls on one of those four surfaces; for this rung the
copy trigger alone should suffice.

### Revised expectation

With the single bare-equality deposit `(=[7,7_copy])` at the
antisymmetry LB's `main` (the conjecture-shape trigger of the previous
entry), the entire chain — marker completion, parked-witness revival,
associativity, cancellation (the copy equality serving as its equality
premise), zero-sum, zero-identity, class-merge goal closure — fits
under every existing gate with no configuration change. Implementation
awaits maintainer approval (Rule 8).

## 2026-07-28 — variable-copy trigger implemented and accepted: ANTISYMMETRY PROVED; two follow-up items for the maintainer (a chapter-167 contradiction-trace failure, one lost Gauss theorem)

Implementation (maintainer-approved plan, commit ): the pure
shape detector `detectAntisymmetryCopyVar` plus the `(=[x,x_copy])`
deposit at the innermost premise LB in `addTheoremToMemory`
(`D-235`); 11 new unit tests; full rebuild
clean, 1323/1323 green. Acceptance run:
, trace preserved as
.

### The theorem is proved

Byte-exact in `files/theorems/theorems.txt`:

```text
(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8](preorder[1,4,7,8])(>[](preorder[1,4,8,7])(=[7,8]))))
```

Trace milestones at the antisymmetry LB (15 bursts, then discharge —
the baseline froze at `toBeProved=1` for all 21): the copy equality is
statement [3] at ENTRY #1; the demanded witness sum
`(in3[int_lev_1_1,int_lev_2_1,it_0_lev_2_12,4])` (k+l=m') is live at
`main` by EXIT #8; the cancellation product `(=[2,it_0_lev_2_12])`
(m'=0) by EXIT #13; the goal closes at EXIT #15 (`toBeProved` 1 → 0,
590 statements). `admissionMap` reads 0 at every dump boundary — the
demand key is written and consumed within a burst window (I-68
staging), consistent with the mint.

### Corpus effect — the unlock is large

`theorems.txt` 51 → 71 rows (40 `AnchorPeano` + 32 `AnchorGauss`, the
Gauss→Peano bridge row counted in both). The gains include the
antisymmetry family and a wave of downstream Gauss interval/limit-set
theorems whose proofs cite antisymmetry directly. Both flagship `fold`
(summation-family) rows intact. Incubator graph airtight: 121,253
checks, 0 failures.

### Item 1 — verifier `contradiction trace` failure, chapter 167 (main graph: 18,381 checks, 1 failed)

The check (verifier `verify_chapter` step 6): a `contradiction` row's
ingredients must trace back through the chapter's origin graph to the
reductio seed's `task formulation` row. Failing chapter
`files/processed_proof_graph/167_direct_proof.txt` — a NEW theorem:
premises `s(v1)=v2`, `limitSet[N,+,V1,i0,V2]` (by the definition:
V2 = {x∈V1: x ≤ 0}), `interval[N,+,i0,v2,V2]`; head
`interval[N,+,i0,v1,V1]`; proved in the negated-head reductio LB.

Root cause of the failure (analysis, no code changed): the premises
are JOINTLY INCONSISTENT — v2 ∈ V2 (interval endpoint membership, a
sibling new theorem) and the limit-set clause give v2 ≤ 0, the
interval's lower bound gives 0 ≤ v2, and the freshly proved
ANTISYMMETRY merges them to v2 = 0, contradicting `¬(s(v1)=0)`. The
reductio LB therefore discharged on a contradiction pair
(`in2[v1,v2,s]` versus `!(in2[v1,v2,s])`) derived from the premises
alone — the seed `!(interval[N,+,i0,v1,V1])` (rest[4], cited by the
record) is never used by either chain, which is exactly what the
trace check demands and flags. The export is mathematically sound
(inconsistent premises prove any head, ex falso), but the RECORD
claims a reductio on a seed the proof never touched. Latent
producer-side record-shape gap, surfaced now because antisymmetry
made a premises-only contradiction derivable for the first time.
Maintainer decision needed — candidate directions: record such
discharges under a vacuous-truth-style tag (the premise-impossible
family) instead of `contradiction`; or export the premises-negation
theorem; or another design. The check itself is correct and stays
(I-16).

### Item 2 — one baseline Gauss theorem lost

Present in the baseline run's output, absent from the new corpus
(byte-exact):

```text
(>[1,2,3,4,5,6,7,8](AnchorGauss[1,2,3,4,5,6,7,8])(>[9,10](in2[9,10,3])(>[11](sequence[1,4,2,10,11])(existence8[1,4,9,11,2]))))
```

The Gauss batch's search trajectory shifted under the 20 new
broadcast rules; this row fell out of the budget window. To weigh
against the +20 gains — maintainer's call whether to chase it.

## 2026-07-28 — vacuous-premise suppression implemented and accepted: BOTH GRAPHS AIRTIGHT (138,802 checks, 0 failures); the vacuous family is gone; antisymmetry unaffected

Maintainer-designed mechanism (`D-236`,
commit ): a normal LB reaching a full E / !E pair at exact
`main` proves its own premise set inconsistent — the LB flags
(`Memory::mainContradiction`), deactivates, its subtree deactivates at
the post-join drain, and every theorem emission whose producer has a
flagged self-or-ancestor is suppressed at the single-threaded seams
(`drainUpdateGlobalDirect` via the producer now carried in the record;
the induction promotion likewise). Genuine reductio parents never flag
(the seed lives only in the child). Anchor-level pairs assert. 1327/1327
unit tests (4 new).

Acceptance run: , trace preserved
as .

1. **Verifier fully airtight on both graphs** — incubator 121,239 / 0,
 main proof graph 17,563 / 0 (total 138,802). The chapter-167
 `contradiction trace` failure is GONE (the counter reads
 success 14 / failure 0; the vacuous chapters were never emitted).
2. **Corpus** — `theorems.txt` 71 → 68 rows. Emission-level delta
 (printed proved-theorem lines of the two runs): 13 dropped,
 0 gained. The four Gauss drops are the vacuous limitSet family —
 chapter 167's row, its same-inconsistent-premise negation sibling,
 and the two `limitSet[..,9,..]` (truncation at v1) vacuous rows.
 The nine Peano-anchored drops are incubator-pool vacuous rows of
 the numeral successor family — including THREE PAIRS that proved
 both a head and its negation from identical premises (the
 definitive inconsistency signature; e.g. `s(x)=0 ∧ s(0)=y ⇒ x=y`
 and `⇒ ¬(x=y)` both exported before this change). The honest
 negation siblings (consistent premise sets) all survive, as do the
 flagship `fold` rows and every chain lemma.
3. **Antisymmetry unaffected** — the LB closes at EXIT #15 exactly as
 in the previous run (`toBeProved` 1 → 0; 581 statements vs 590 —
 the small delta is the cleaner incubator context feeding the Peano
 batch, nine junk rules fewer).
4. **Documented boundary** — a few single-polarity rows with
 impossible premises remain (the `s(x)=0`-premise family): their
 premise LBs never materialized the contradicting STATEMENT pair at
 `main`, and their reductio records genuinely consumed their seeds —
 the verifier's trace check passes them as honest reductio. The
 suppression removes exactly what the parent can see; both walls of
 the original problem (vacuous export + seed-free record) are
 closed.

Rung-2.1 next step unchanged: the `¬(2≤1)` consumer and the ES3
negation theorem via antisymmetry on the reductio's derived pair.

## 2026-07-28 — determinism rerun: byte-identical

Second full pipeline run, same binary, clean state
(): `theorems.txt`,
`global_theorem_list.txt`, and the incubator `global_theorem_list.txt`
all byte-identical by hash to the acceptance run; the sacred trace of
the antisymmetry LB byte-identical to the preserved
; verifier identical
at 121,239 + 17,563 = 138,802 checks, 0 failures on both graphs. The
branch is complete and ready to squash onto.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
