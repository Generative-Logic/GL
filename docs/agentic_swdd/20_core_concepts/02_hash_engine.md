<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Core concept — Hash engine `[DRAFT]`

> The central bet of GL's architecture: **inference is memory access**. Every proof step reduces to a hash lookup on the structural signature of known expressions. This chapter walks the data structures and the query/fire cycle that make that bet real.

---

## The core idea

Traditional theorem provers explore a search tree of tactics, rewrites, or unifications. GL takes the opposite direction: treat inference as a *lookup* problem. Every implication rule `premise_pattern → conclusion_template` is hashed by its premise pattern into a per-LB index. When an LB has a registered statement (an `intEncodedStatements` row) that matches the premise pattern (after parameter binding), the rule fires and the conclusion is emitted.

Two invariants make this work:

1. **Expressions are structurally canonical.** Two theorems that differ only in bound-variable names must hash identically. This is what `reshuffleTheorems`, `precompileStructuralOperators`, and normalisation routines ensure.
2. **Hash keys are integer-encoded.** The string form is canonical but slow; every hot-path lookup uses the `IntNormalizedKey` form.

Result: each step is O(1) amortised. Exploration happens by breadth — multiple LBs fire in the same cycle, and the mail system carries derivations across the grid between cycles.

---

## The main data structure — `HashMemory`

Defined at [`memory.hpp`](../../GL_Quick_VS/GL_Quick/src/memory.hpp). Fields:

| Field | Type | Role |
|---|---|---|
| `encodedMap` | `TypedColdBlobMap<NormKey, LocalMemoryValue>` (cold) | The actual rule registry. Key = premise-pattern's integer-normalised hash; value = the run of rules with that same premise shape. Cold blob map ([I-98](../30_invariants.md#i-98)). |
| `admissionMap` | `TypedColdBlobMap<int32_t, AdmissionMapValue>` (cold) | Populated by `updateAdmissionMap` from consumer-side registrations during Pass B. Consulted by `isAdmitted`. Key shape: **`u_`-prefixed** template; cold-stored ([I-99](../30_invariants.md#i-99)). |
| `admissionMapIntegration` | `TypedColdBlobMap<int32_t, IntegrationEntry>` (cold) | Integration-side analog. Populated at `prover.hpp` from `le.signature`. Consulted by `isAdmittedIntegration`. Key shape: **`u_`-prefixed** template (same as `admissionMap`). Value is a nested `map<IntInstruction, ValueIdSet>` flattened one `IntegrationEntry` blob per inner entry; cold-stored ([I-99](../30_invariants.md#i-99)). |
| `admissionSetIntegration` | bookkeeping | Secondary integration-admission set. Populated at `prover.cpp` via `removeUPrefixFromArguments`. Key shape: **bare concrete** with `repl_` preserved. Consulted by Pass B `int_` fallback and by `applyEquivalenceClassToRejectedMapIntegration`. |
| `rejectedMap` | `TypedColdBlobMap<int32_t, RejectedMapValue>` (cold) | Algebra-side rejection buffer — `it_` vars that failed Pass B admission. Consulted by `revisitRejected2` when a new admission key arrives. Packed `(templateId, validityId)` int32 key, **bare concrete** template form (Pass B's `makeMarkedExpr` output); cold-stored ([I-99](../30_invariants.md#i-99)). |
| `rejectedMapIntegration` | `TypedColdBlobMap<int32_t, RejectedMapIntegrationValue>` (cold) | Integration-side rejection buffer — `int_` vars that failed Pass B admission ([D-19](../40_decisions.md#d-19)). Consulted by `revisitRejectedIntegration2` and rewritten by `applyEquivalenceClassToRejectedMapIntegration`. Key shape: **bare concrete** (same as `rejectedMap`); cold-stored ([I-99](../30_invariants.md#i-99)). |
| `varsInRejectedMapIntegrationKeys` | `TypedColdSet<int16_t>` (cold cache) | Monotonically-growing set of non-marker args appearing in any `rejectedMapIntegration` key. Used by the equi-class-rewrite helper for fast class-overlap short-circuit; cold-stored, never wiped. |
| `normalizedEncodedKeys` | cold owner-set blob map | `NormKey` → exactly ONE `OwnerSet` blob per key (`TypedColdBlobMap<NormKey, OwnerSet>`, run-length-1, whole-value replace — statified onto the cold blob map, [D-140](../40_decisions.md#d-140)). `OwnerSet` = `partitionIds` (the owner record: one packed `(expandedOriginal, scopeVid)` composite per owner, serving [D-72](../40_decisions.md#d-72) ownership + [D-105](../40_decisions.md#d-105) comparability via the low half + [D-119](../40_decisions.md#d-119) split partition — there is NO separate `owners` map, [I-80](../30_invariants.md#i-80)) + `hasLooseOwner` / `uSignatures` (the [D-120](../40_decisions.md#d-120) u_ prune cache). Read on the hot prune path via the zero-allocation byte peek (`ownerKeyAccepts` → `peekRecordBytes` + `OwnerSetBlob`), never a full decode; written read-modify-write (`mergeOwnerRecord`). See [D-71](../40_decisions.md#d-71), [I-100](../30_invariants.md#i-100). |
| `normalizedEncodedSubkeys`, `normalizedEncodedSubkeysMinusOne`, `normalizedEncodedSubkeysMinusTwo` | cold owner-set blob map | Same cold `TypedColdBlobMap<NormKey, OwnerSet>` shape and same peek-read / RMW-write contract as `normalizedEncodedKeys` — fast-rejection subkey indices for the gradient algorithm. Drop a subkey only when its owner-set (`partitionIds`) empties. See [I-49](../30_invariants.md#i-49), [D-120](../40_decisions.md#d-120). |

**Three key shapes, five maps.** Admission MAPS (`admissionMap`, `admissionMapIntegration`) are u_-prefixed templates. The admission SET (`admissionSetIntegration`) and rejection MAPS (`rejectedMap`, `rejectedMapIntegration`) are bare concrete instances. Lookups must transform at the boundary — u_-prefix when querying the maps, strip u_ when querying the set/rejection side. Storage keying: the algebra group (`admissionMap`, `admissionStatusMap`, `consumedAdmissionKeys`, `revisitInProgress`) is packed `(templateId, validityId)` via the dedicated `Memory::templateInterner` ([D-132](../40_decisions.md#d-132)) — write sites `mintTemplateKey`, probes the non-minting `lookupTemplateKey` (a never-interned template is a definitive miss); the key-SHAPE contract above is unchanged, the shapes are the interned strings. The algebra group + `varsInAdmissionMapKeys` AND the integration twins (`admissionMapIntegration` / `rejectedMapIntegration` / `varsIn{Admission,Rejected}MapIntegrationKeys`) are now COLD-stored on the cold-map family (`admissionMap` / `rejectedMap` / `admissionMapIntegration` / `rejectedMapIntegration` → `TypedColdBlobMap`, the satellites → `TypedColdSet` / `TypedColdMap`; [I-99](../30_invariants.md#i-99)). The admission SET side (`admissionSetIntegration` / `triggersForAdmissionSetIntegration`) stays on the heap. See [`10_pipeline/04_prover.md`](../10_pipeline/04_prover.md#admission-rejection-map-key-shapes) for the full table and [G-32](../50_gotchas.md#g-32) for the silent-miss pitfall.

A `LocalMemoryValue` ([`memory.hpp`](../../GL_Quick_VS/GL_Quick/src/memory.hpp)) packages each rule entry. Id form per `D-133`: every string field is an id in the owning LB's `Memory::ruleInterner` (int32), minted at the single-threaded install sites; the parallel hashburst only decodes (array-index const refs).

| Field | Role |
|---|---|
| `valueId` | The normalised conclusion template — the "what to emit". For head-implication LMVs (insert path 1 below) this is the implication's conclusion; for marker LMVs (insert path 2 below) this is the marker-form expression with `"marker"` substituted at the output-arg slot. The firing path substitutes on the decoded template. |
| `levels` | `std::set<int>` — the rule's installation level set: LB depths whose state contributed to the rule being installed here. Propagates into the firing's derived statement via `union(rule.levels, premise.levels)` at fire time. Must be empty for rules deposited through mail (so only the receiver's premise side determines the final union) — see [I-51](../30_invariants.md#i-51) and [`02_glossary.md::levels`](../02_glossary.md#levels). |
| `originalImplicationId` | The full MPL of the original implication — for provenance (decoded into the firing's origin pair). |
| `justification` | The closed `RuleJustification` enum (`ruleJustificationFromString` asserts on an unknown string). `implication` on every installed rule; `none` on marker LMVs. |
| `keyIds` | The marker-key sub-key chain (`vector<int32_t>`, built from premises that don't reference the output arg). Carried ONLY by marker LMVs (insert path 2); head-implication LMVs leave it empty — the premise pattern they key on lives in `IntNormalizedKey`. |
| `remainingArgIds` | Arguments not yet bound — consulted during binding. Decoded-lex sorted id storage (set-iteration encode order). |
| `validityId` | The NameMap id of the scope in which this rule is active, minted at install (the same encode that feeds the owner record); the firing path's scope-comparability check reads it with no per-firing lookup. |
| `isMarker` | Install-time classification: the head template contains `"marker"`. Replaces the per-firing substring scan that used to branch head vs marker firing. |
| `productOfDisintegration` | (D-32, 2026-05-01) Marks rules whose chain has at least one premise with an arg starting with `"u_"` — the reserved bound-variable-placeholder prefix introduced by `prefixArgumentsWithU` during disintegration. Stamped only on head-implication LMVs (insert path 1); marker LMVs default to `false`. Consumed by `checkLocalEncodedMemoryStatic` to gate OR-disintegration via the `allowOrDisintegration` parameter threaded into `disintegrateExpr2`. See [D-32](../40_decisions.md#d-32) and [`07_or_branching.md`](07_or_branching.md#1-or-disintegration). |

One key can point to multiple `LocalMemoryValue`s — several different rules may happen to normalise to the same premise-pattern signature. When an LB matches the key, *all* pointed-to rules are candidate fires.

---

## The query-fire cycle

Per LB, per cycle, the elementary-step hashburst ([`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp), `performElemPhase2` → `performElem2`):

1. **Generate hash requests.** One function does this — `generateEncodedRequestsStatic` ([`memory.cpp`](../../GL_Quick_VS/GL_Quick/src/memory.cpp)) — called with the length of the obligatory stump, the already-known statements every generated request must contain:
 - **1** — mandatory single (batches 1, 2, 4, 5).
 - **2** — mandatory pair (batch 3, local × mail).
 - **0** — CE filter: no element is mandatory, so a grown base candidate is itself the request.

 It emits `IntNormalizedKey` objects that correspond to each candidate match.


 Under the LB split's second dimension the call also carries a **bucket of split stumps** ([D-203](../40_decisions.md#d-203)). The search then runs once per stump in the bucket over one shared filtered statement list, joining the stump to every growing candidate for both owner-set probes — `normalizedEncodedSubkeys` for growth, the target map for recording — and materialising the union into a `BaseCandidate` only where the target map accepts. The stump is never carried by a growing candidate; it is attached per probe and dropped again. The stump alone is probed once before the search, because unsplit that base candidate is recorded in the loop of the candidate one level up, which a stumped sub-part never runs.
2. **Look up.** Query `overallHashMemory.encodedMap[key]`. If the key is in `normalizedEncodedKeys`, there is a hit.

3. **Iterate `LocalMemoryValue` list.** For each pointed-to rule, attempt to bind the remaining arguments. If binding succeeds and admission holds (`isAdmitted` / `isAllowedAsOperatorInput`), fire.

4. **Fire = emit.** Build the conclusion expression by applying the binding to `value`. Compute the derived statement's `levels` as `union(rule.levels, premise[0].levels, premise[1].levels, …)` — every premise that participated in the binding contributes its level set, and the rule itself contributes its installation level set ([`LocalMemoryValue::levels`](#localmemoryvalue)). Call `addStatement` ([`prover.hpp`](../../GL_Quick_VS/GL_Quick/src/prover.hpp)) with the expression, target memory block, computed levels, validity, local flag. Record origin entry into `exprOriginMap`. The derived levels then sit in `body.intStatementLevelsMap` (packed `(originalId, validityId)` key) and are read by the discharge gate `allLevelsInvolved` at `prover.hpp::dischargeToBeProved` whenever the derived expression matches an `intToBeProved` goal at v=main; an oversized level set silently swallows the discharge (see [`02_glossary.md::levels`](../02_glossary.md#levels), [G-46](../50_gotchas.md#g-46)).

5. **Broadcast.** If the new statement is at `validityName == "main"`, also push into `mailOut.statements` for other LBs to consume next cycle.

**Deposits captured + sorted, not applied inline ([D-117](../40_decisions.md#d-117)).** In the current code the fire step does not mutate the deposit containers inline. `checkLocalEncodedMemoryStatic` appends a `FiringRecord` per firing; after the request pass, `applyFiringRecords` sorts the records by a canonical content key and then applies the deposits (statements / origins / `disintegrationSignals` / the two staged admission+integration vectors / `canBeSent` sets). The sort makes the order-sensitive deposits (the origin cap, the `disintegrationSignals` last-write, the staged-vector drain order) independent of request-evaluation order — the property the LB-split rule-partition merge relies on ([I-77](../30_invariants.md#i-77)). The fire step does no deactivation at all — deactivation and discharge run only in phase 3's post-burst `standardProcessing` ([I-66](../30_invariants.md#i-66)).

**Validity-comparability prune at request generation ([D-105](../40_decisions.md#d-105)).** Inside step 1, every site that matches a (growing) request's structural key against an owner-set (`normalizedEncodedKeys` / `normalizedEncodedSubkeys` / `…MinusOne` / `…MinusTwo`) additionally drops the request when its consensus validity is comparable to no owner of the matched key — iterating the `int16` validity id stored with each owner in `OwnerSet::owners` via `ExpressionAnalyzer::ownerSetHasComparable`. This is a pure runtime prune: the firing gate `checkLocalEncodedMemoryStatic` enforces the same comparability downstream, so it can never change which rules fire. A `main`-scope skip (`requestVid == MAIN_ID` keeps the request unconditionally) makes it free on all-`main` workloads — the whole CE filter (every fact at `"main"`) and the bulk of Peano main-prover — so pruning work accrues only in branching proofs (OR theorem, FTA ladder). It is a sound over-approximation: it never drops a request that could fire ([I-70](../30_invariants.md#i-70)).

**Rule-partition filter at request generation ([D-119](../40_decisions.md#d-119)).** AND-combined with the D-105 prune at the same acceptance sites, `partitionAccepts(OwnerSet::partitionIds)` keeps a request only when this LB-split executor (`g_splitProcessID` of `g_splitCount`, thread-local) owns the matched key — some composite id with `id % g_splitCount == g_splitProcessID`. Like the D-105 prune it never changes which rules ultimately fire (the firing gate re-checks scope; an over-claimed key just costs a wasted request), and at `g_splitCount == 1` (the unsplit identity) it short-circuits to accept before touching the set, so it is provably inert until the split drives it.

**u_ literal prune at request generation ([D-120](../40_decisions.md#d-120)).** AND-combined with the D-105 and partition checks at the same acceptance sites, `ExpressionAnalyzer::ownerSetUSatisfied(OwnerSet, exprs, count)` drops a request whose concrete argument values can satisfy no owner's u_ (unchangeable) literals. The four `normalizedEncoded*` maps are `ignoreU=false`, which normalizes every argument to a positional id and so **erases** a rule's u_ literal values — only the repetition pattern survives — so a structurally-matched request may be doomed by a u_ literal it can never match, a mismatch the firing gate (`encodedMap`, `ignoreU=true`) would otherwise catch only later. `recordUSignature` caches each owner's u_ signature — the `(linear-arg-slot, argFullId)` list of its unchangeable args, where `argFullId == NameMap::encode(arg[1])`, exactly the value the firing gate matches against the request's `argFullId` — at insert in `OwnerSet::uSignatures`. An owner with zero u_ args (or, conservatively, a u_ literal not yet interned at insert) sets `OwnerSet::hasLooseOwner`, which short-circuits the check to *keep* (the "0 u_, nothing to check" fast path). At match the request's flattened `argFullId` is compared against the cached signatures and the request is dropped only when no owner can be satisfied. The literal id is read via the non-minting `NameMap::lookup` at insert and never `encode`d at match, so the cache build mints nothing (byte-identical) and the match is read-only (safe on the shared LB the split executors read in parallel). Like D-105 it never changes which rules fire — the firing gate's `encodedMap` lookup enforces the exact u_ literal match — so it is a sound over-approximation ([I-79](../30_invariants.md#i-79)). Cost is O(1) on the common all-loose key; unlike `owners`/`partitionIds` the signatures are NOT maintained on the radical subtree wipe (a stale entry only weakens the prune, never makes it unsound).

---

## Locality semantics: a local rule can fire against entirely non-local premises

A rule that lives in `body.localHashMemory{,Delta}` is "local" — this LB owns it. The premises that fire it can be drawn from anywhere visible at this LB, including parent-scope rows the LB sees only via `nm.comparable`. **A local rule firing on 100 % non-local premises is correct rule semantics**, not a bug.

**Both ends of the firing pipeline use `nm.comparable` for scope inheritance ([D-55](../40_decisions.md#d-55) Part C, 2026-05-09).** Request generation has always honoured ancestor/descendant inheritance via the `pairMap`-backed `nm.comparable` (see the 5-batch fan-out below). The firing site (`checkLocalEncodedMemoryStatic`'s head-LMV scope check) was on a stricter strict-eq + main-mask rule until `D-55`'s Part C: pre-fix, a rule at scope `S` could only fire on facts at scope `S` or at `"main"`, even though the matching request had already been emitted under the broader comparable rule. The mismatch was masked while every load-bearing rule of this shape was a v=main universal (chapter-6 / chapter-11 mirror reformulations). When `D-55` Part B removed those parents from `theorems.txt` and the K-impls took over the role at the OR's non-main scope, the strict-eq mismatch surfaced and rejected legitimate cross-scope firings (rung-1 stalled at burst 5 toBeProved=4). Part C realigned the firing-site check to `nm.comparable`, matching the request-generation rule. Both ends now agree on what scopes are mutually visible.

This is encoded in the 5-batch fan-out at request generation. `performElementaryLogicalStep` ([`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp)) calls `generateEncodedRequestsStatic` (and `…StaticPairs` for Batch 3) in five batches that pair a rule pool with a mandatory-element source. The seed step at [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp) constrains every emitted request to contain ≥ 1 mandatory element, so the mandatory source determines whether the request can be local-free.

| # | Rule pool | Mandatory source | Local mandatory? |
|---|---|---|---|
| 1 | `working.overallHashMemory` | `body.intLocalEncodedStatements` | ✓ |
| 2 | `body.overallHashMemory` | `body.intLocalEncodedStatementsDelta` | ✓ delta |
| 3 | (pairs — `…StaticPairs`) | local × mail pairs | mixed |
| 4 | `body.localHashMemory` | `mailIntEncoded` | ✗ mail (v=main) |
| 5 | `body.localHashMemoryDelta` | `body.intEncodedStatements` (broad) | ✗ |

Batches 4 and 5 are the asymmetric ones — **deliberately** non-local:

- **Batch 5** ([`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp)). When a fresh rule lands in `localHashMemoryDelta` this cycle, it must fire against every pre-existing visible statement, not only against statements that arrived this cycle. Hence `mandatorySrcInt = body.intEncodedStatements.data` (broad view).
- **Batch 4** ([`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp)). Symmetric for the existing `localHashMemory` against fresh mail: every rule already at this LB needs to be tested against new mail rows.

Empirical confirmation: in the `IncubatorGauss1` run, 36 of 6398 firings at the SE2 LB carry **zero** premises in the LB's local statement set (then the string-keyed `localEncodedStatementsSet`; today's packed-key `intLocalEncodedStatementsSet` carries the same membership). The matching Peano universal-x rules have premise `(in[X, 1]) | v=main` among their chain. One example whose mathematical content is uncontroversially sound:

- `(>[1](in[1,u_1])!(in2[1,u_2,u_3]))` — Peano P4 lifted: for any `v ∈ N`, `s(v) ≠ 0` (`0` is not a successor of any natural). One chain element, head is the negation of the in2-successor relation.

The β template that fires at Step 3 of the FTA ladder ([`docs/fta_ladder/rung1/current_proof_state.md`](../../fta_ladder/rung1/current_proof_state.md) §"Step 3 — Peano β template fires at Branch A") has shape `!(=[u_p, u_i0]) → (existence0[u_N, u_p, u_s])` — the negated-equality premise gates predecessor existence. An earlier draft of this paragraph cited a single-`(in[X, N])`-premise version of the β template; that shape, taken literally, would assert *every* `n ∈ N` has a predecessor (false for `n = 0`), so it cannot be the actual installed rule. The original measurement was likely seeing the multi-premise mirror reformulation (anchor + `(in[v, N])` + `!(=[v, i0])`) but only counting the `(in[X, N])` chain element as the "single premise" because the anchor and equality premises had been satisfied via inheritance from v=main / the LB's ancestor scope. The "zero local premises" claim survives that re-reading; the load-bearing point ([D-29 clause 2](../40_decisions.md#d-29) — block disintegration when none of the matched premises is local) does not depend on the exact β-template shape.

The premises are anchor-deposited at v=main; the SE2 LB sees them via `body.intEncodedStatements` but never deposits them into its own `intLocalEncodedStatementsSet`.

**What this implies for disintegration.** Emission of these firings is fine. **Disintegration** of them at a non-anchor LB is what mints fresh `it_*` / `int_*` vars and exhausts the int16_t NameMap in incubator + `!ban_disintegration` mode. The line-2017 gate ([D-29 clause 2](../40_decisions.md#d-29)) is the architectural answer — block disintegration when none of the matched premises is in the LB's `intLocalEncodedStatementsSet` (an O(1) packed-key probe on the request's int rows since `D-129`). The gate is *not* redundant with request generation; it corrects for Batches 4/5 at the disintegration site rather than tightening request emission.

---

## Writing rules — `addToHashMemory`

Defined at [`memory.cpp`](../../GL_Quick_VS/GL_Quick/src/memory.cpp) (extracted from `prover.cpp`, 2026-05-04). Takes a `(chain, head)` pair, normalises, keys by the chain's integer hash, and installs into `encodedMap`. Also invokes `multiplyImplication` (still in `prover.cpp`, called cross-TU) to install Bell-partition equalised copies (see [prover chapter](../10_pipeline/04_prover.md#multiplyimplication-bell-partition-equalisation)).

Every theorem-load path ends at `addToHashMemory`:

- Initial external-theorem load → `addTheoremToMemory` → `addToHashMemory`.
- Per-conjecture registration → `addTheoremToMemory` (with `proved=false`) → `addToHashMemory` for the premises.
- Broadcast receipt → `addStatement` (for a pure statement) or `addTheoremToMemory` (for a derived implication).
- Compressor Phase 1 → direct `addToHashMemory` for every rule in the pool.

### Two `encodedMap` insertion sites — head LMVs vs marker LMVs

`encodedMap` carries entries from **two structurally distinct insertion paths**, both writing into the same map. Distinguishing them matters when you want to thread metadata through the LMV (e.g. `productOfDisintegration` for D-32 — only meaningful on head LMVs).

**Path 1 — head-implication insert** (`memory.cpp`, inside `addToHashMemory`'s per-permutation loop). For each Bell-partition copy of the implication and each permutation of the chain, build an LMV whose `.value` is the head's normalised template (with the chain's normalisation map applied) and push into `encodedMap[intIgnoredKey]`. This is the rule registry entry — the LMV that fires when an LB hash-matches the chain.

**Path 2 — marker LMV insert** (`memory.cpp`, inside `makeNormalizedKeysForAdmission` called from `addToHashMemory`). For each chain element that has an `output_args` slot, replace the output arg with the literal token `"marker"` to build a marker-form expression, then push an LMV whose `.value` is that marker-form expression into `encodedMap[intIgnoredKey]` keyed by the *sub-key* (the chain's other elements that do NOT reference the output arg). This is the consumer-side admission template — the LMV that fires during Pass B's `int_` admission probe via `isAdmittedIntegration`.

The two paths share the map but populate disjoint metadata. Head LMVs carry full `originalImplication` / `justification` / `validityName` for provenance; marker LMVs leave those fields default-empty (the marker entry is structural — its only job is to assert "this output-arg shape is admissible at this premise sub-key"). When adding new fields to `LocalMemoryValue`, ask: do I want this on head firings, on admission probes, or both?

For D-32, `productOfDisintegration` is set only on path 1 (head implications). Marker LMVs (path 2) inherit the default `false` — they don't drive OR-disintegration; only head firings do, and only head firings are the call shape that `checkLocalEncodedMemoryStatic` evaluates.

---

## Integer encoding — the hot-path form

Every string expression has an `EncodedExpression` form ([`memory.hpp`](../../GL_Quick_VS/GL_Quick/src/memory.hpp)):

| Field | Role |
|---|---|
| `name` | The head operator name (e.g. `"in3"`). |
| `negation` | `true` if the expression is `!(...)` at top level. |
| `arguments` | `vector<Argument>` with iteration/level metadata per arg. |
| `maxIterationNumber` | Cap on iteration index used during hash generation. |
| `original` | The original string form — for debug output + round-trip. |
| `validityName` | The scope this expression is asserted in. |

Integer encoding uses `NameMap::encode` to collapse every name into an `int16_t`. The resulting `IntNormalizedKey` is a packed array of (name_id, arg_ids) — cheap to hash, fast to compare.

**The int row is the only stored statement form** ([D-127](../40_decisions.md#d-127)). `encodeExpression` (`memory.hpp`) converts an `EncodedExpression` into the flat all-int16 `IntEncodedExpr` — interning the WHOLE `original` text as `originalId` and the scope name as `validityId` — and `decodeExpression` is its exact inverse: it looks both strings up and lets `EncodedExpression`'s parsing constructor re-derive every other field, so the round trip is lossless (arity above `MAX_ARITY` asserts at encode). `Memory::intEncodedStatements` and the local / delta / external int registries are therefore the single source of truth; `EncodedExpression` values exist only transiently — built at insert sites, decoded on demand at boundaries (diagnostic dump, mail fill, visualizer, equivalence-class rewriting). Boundary reads use the non-minting `NameMap::lookup` and copy decoded references before any mint ([I-3](../30_invariants.md#i-3)).

`precompileStructuralOperators` ([I-1](../30_invariants.md#i-1)) is the transformation that ensures raw `!(&...)` / `!(>...)` *always* become compiled `or<N>` / `existence<N>` names before they hit the encoder. Otherwise the encoded key would carry the structural operator as its name, and the hash index would treat semantically-identical expressions as distinct.

---

## Admission — when does an emission actually happen

Not every hash hit fires. Between the key match and the `addStatement` call, the admission check gates which freshly-minted variables are permitted in the conclusion.

Two paths (see [prover chapter — Disintegration](../10_pipeline/04_prover.md#disintegration)):

1. **`isAdmitted`** ([`prover.hpp`](../../GL_Quick_VS/GL_Quick/src/prover.hpp)) — consults `admissionMap`, the authoritative path.
2. **`isAllowedAsOperatorInput`** ([`prover.hpp`](../../GL_Quick_VS/GL_Quick/src/prover.hpp)) — single-input-operator fallback; guardrail against RT explosion. See [I-6](../30_invariants.md#i-6).

First-match-wins. Rejected variables are buffered in `pendingRejections`; the outer "commit rejections only for never-admitted vars" guard decides whether they stay rejected or get admitted later.

---

## Why hash-based and not SAT / congruence closure

The architectural bet is that the cost of search-based methods (SAT, proof-term reconstruction, tactic scheduling) is avoided if the *definition* of each rule provides its hash key structurally. A SAT solver has to discover what to prove; a GL LB has to discover that it already knows what was asked.

Consequences:

- **Determinism.** The same input always produces the same set of fires in the same order. LBs are stepped across real worker threads, but the result is thread-schedule-independent: no LB observes another's mid-cycle state and every cross-LB effect is drained in content-sorted order after the join ([I-28](../30_invariants.md#i-28)).
- **No backtracking.** Once an expression is admitted, it is never unadmitted. Branches are handled by scope (OR disintegration creates scoped children), not by speculative execution.
- **Latency is memory-access.** Optimising the prover is optimising the hash index's locality. `intNormalizedKey` + `unordered_map` is the current structure; more specialised structures (tries, Bloom filters over premise classes, cache-line-sized buckets) are explored candidates for the next RT campaign.

---

## Weaknesses

### Known & tracked

- **`multiplyImplication` × N install cost.** Every rule installed via `addToHashMemory` triggers `multiplyImplication`, which generates Bell-partition copies and installs each. For implications with 5+ `(1)`-typed bound vars, this is O(B_5) = 52 installs per rule. At FTA scale with larger bound sets, this becomes a measurable percentage of install time.
- **Pre-FTA RT is dominated by CE filtering.** 80% of Gauss wall-clock is CE filtering. The main-prover hash engine itself is not currently the bottleneck.

### Suspected fragility

- **`admissionMap` consistency across nested scopes.** The admission map is per-LB, but nested-scope rules (hypothesis, OR branch) need their own entries. A scoping bug here would silently widen or narrow admission.
- **`IntNormalizedKey` collision assumption.** The normalisation is designed so that distinct semantic expressions produce distinct keys. Not formally proven — if two structurally-distinct expressions normalise to the same key, the hit would fire both rules on the wrong premises. Historically not observed, but the guarantee is implicit, not asserted.
- **`rejectedMap` cleanup.** Entries are added when admission fails; unclear whether all scope-discharge paths clean them up fully. A rejection that persists past its scope would prevent later admission of a legitimate expression.
- **`rejectedMapIntegration` unbounded growth.** See "Scaling pressures on `rejectedMapIntegration`" in [`10_pipeline/04_prover.md`](../10_pipeline/04_prover.md#scaling-pressures-on-rejectedmapintegration-beyond-01-01). Current linear walk per class call is O(|rmi|) despite the short-circuits. Planned mitigation: reverse index `varToKeys` maintained on insert/erase, reducing per-class-call cost to O(|affected|).

### Not exercised by tests

- **Hash-key-collision monitoring.** No instrumentation counts `encodedMap[key].size > 1` events or tracks whether multiple rules share a key. If a batch ever had catastrophic collisions, the symptom would be wall-clock degradation with no clear signal.

---

## See also

- [`10_pipeline/04_prover.md`](../10_pipeline/04_prover.md) — prover operation overview.
- [`20_core_concepts/01_logic_blocks.md`](01_logic_blocks.md) — the LB owning the hash memory.
- [`20_core_concepts/03_mail_system.md`](03_mail_system.md) — cross-LB propagation.
- [I-1](../30_invariants.md#i-1) — precompile before key encoding.
- [I-6](../30_invariants.md#i-6) — Pass B admission gate.
-, — RT campaign history.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
