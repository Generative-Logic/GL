<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Core concept — Hash engine `[DRAFT]`

> The central bet of GL's architecture: **inference is memory access**. Every proof step reduces to a hash lookup on the structural signature of known expressions. This chapter walks the data structures and the query/fire cycle that make that bet real.

---

## The core idea

Traditional theorem provers explore a search tree of tactics, rewrites, or unifications. GL takes the opposite direction: treat inference as a *lookup* problem. Every implication rule `premise_pattern → conclusion_template` is hashed by its premise pattern into a per-LB index. When an LB has an expression in its `wholeExpressions` that matches the premise pattern (after parameter binding), the rule fires and the conclusion is emitted.

Two invariants make this work:

1. **Expressions are structurally canonical.** Two theorems that differ only in bound-variable names must hash identically. This is what `reshuffleTheorems`, `precompileStructuralOperators`, and normalisation routines ensure.
2. **Hash keys are integer-encoded.** The string form is canonical but slow; every hot-path lookup uses the `IntNormalizedKey` form.

Result: each step is O(1) amortised. Exploration happens by breadth — multiple LBs fire in the same cycle, and the mail system carries derivations across the grid between cycles.

---

## The main data structure — `HashMemory`

Defined at [`memory.hpp`](../../GL_Quick_VS/GL_Quick/src/memory.hpp). Fields:

| Field | Type | Role |
|---|---|---|
| `encodedMap` | `unordered_map<IntNormalizedKey, list<LocalMemoryValue>>` | The actual rule registry. Key = premise-pattern's integer-normalised hash; value = list of rules with that same premise shape. |
| `admissionMap` | bookkeeping | Populated by `updateAdmissionMap` from consumer-side registrations during Pass B. Consulted by `isAdmitted`. Key shape: **`u_`-prefixed** template. |
| `admissionMapIntegration` | bookkeeping | Integration-side analog. Populated at `prover.hpp` from `le.signature`. Consulted by `isAdmittedIntegration`. Key shape: **`u_`-prefixed** template (same as `admissionMap`). |
| `admissionSetIntegration` | bookkeeping | Secondary integration-admission set. Populated at `prover.cpp` via `removeUPrefixFromArguments`. Key shape: **bare concrete** with `repl_` preserved. Consulted by Pass B `int_` fallback and by `applyEquivalenceClassToRejectedMapIntegration`. |
| `rejectedMap` | bookkeeping | Algebra-side rejection buffer — `it_` vars that failed Pass B admission. Consulted by `revisitRejected2` when a new admission key arrives. Key shape: **bare concrete** (Pass B's `makeMarkedExpr` output). |
| `rejectedMapIntegration` | bookkeeping | Integration-side rejection buffer — `int_` vars that failed Pass B admission ([D-19](../40_decisions.md#d-19)). Consulted by `revisitRejectedIntegration2` and rewritten by `applyEquivalenceClassToRejectedMapIntegration`. Key shape: **bare concrete** (same as `rejectedMap`). |
| `varsInRejectedMapIntegrationKeys` | cache | Monotonically-growing set of non-marker args appearing in any `rejectedMapIntegration` key. Used by the equi-class-rewrite helper for fast class-overlap short-circuit. |
| `normalizedEncodedKeys` | set | The set of keys currently active in `encodedMap`. Used for fast existence checks without iteration. |

**Three key shapes, five maps.** Admission MAPS (`admissionMap`, `admissionMapIntegration`) are u_-prefixed templates. The admission SET (`admissionSetIntegration`) and rejection MAPS (`rejectedMap`, `rejectedMapIntegration`) are bare concrete instances. Lookups must transform at the boundary — u_-prefix when querying the maps, strip u_ when querying the set/rejection side. See [`10_pipeline/04_prover.md`](../10_pipeline/04_prover.md#admission-rejection-map-key-shapes) for the full table and [G-32](../50_gotchas.md#g-32) for the silent-miss pitfall.

A `LocalMemoryValue` ([`memory.hpp`](../../GL_Quick_VS/GL_Quick/src/memory.hpp)) packages each rule entry:

| Field | Role |
|---|---|
| `value` | The normalised conclusion template — the "what to emit". For head-implication LMVs (insert path 1 below) this is the implication's conclusion; for marker LMVs (insert path 2 below) this is the marker-form expression with `"marker"` substituted at the output-arg slot. |
| `levels` | `(statementLevel, equalityLevel)` at installation time. |
| `originalImplication` | The full MPL of the original implication — for provenance. Empty on marker LMVs. |
| `justification` | Tag category for the proof graph (`"implication"`, `"theorem"`, etc.). Empty on marker LMVs. |
| `key` | The chain (`vector<string>`) that originally keyed this entry. For head-implication LMVs this is the implication's premise list; for marker LMVs this is the marker-key sub-key (built from premises that don't reference the output arg). |
| `remainingArgs` | Arguments not yet bound — consulted during binding. |
| `validityName` | The scope in which this rule is active. |
| `productOfDisintegration` | (D-32, 2026-05-01) Marks rules whose chain has at least one premise with an arg starting with `"u_"` — the reserved bound-variable-placeholder prefix introduced by `prefixArgumentsWithU` during disintegration. Stamped only on head-implication LMVs (insert path 1); marker LMVs default to `false`. Consumed by `checkLocalEncodedMemoryStatic` to gate OR-disintegration via the `allowOrDisintegration` parameter threaded into `disintegrateExpr2`. See [D-32](../40_decisions.md#d-32) and [`07_or_branching.md`](07_or_branching.md#1-or-disintegration). |

One key can point to multiple `LocalMemoryValue`s — several different rules may happen to normalise to the same premise-pattern signature. When an LB matches the key, *all* pointed-to rules are candidate fires.

---

## The query-fire cycle

Per LB, per cycle, `performElementaryLogicalStep` ([`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp)):

1. **Generate hash requests.** Depending on role (main vs. CE filter) and which pattern sizes are active, call:
 - `generateEncodedRequestsStatic` — singleton matches ([`memory.cpp`](../../GL_Quick_VS/GL_Quick/src/memory.cpp); extracted from `prover.cpp`, 2026-05-04).
 - `generateEncodedRequestsStaticPairs` — pair matches ([`memory.cpp`](../../GL_Quick_VS/GL_Quick/src/memory.cpp); same migration).
 - `generateEncodedRequestsStaticCE` — CE mode, no mandatory elements ([`filter.cpp`](../../GL_Quick_VS/GL_Quick/src/filter.cpp); extracted from `prover.cpp`, 2026-05-04).

 These emit `IntNormalizedKey` objects that correspond to each candidate match.

2. **Look up.** Query `overallHashMemory.encodedMap[key]`. If the key is in `normalizedEncodedKeys`, there is a hit.

3. **Iterate `LocalMemoryValue` list.** For each pointed-to rule, attempt to bind the remaining arguments. If binding succeeds and admission holds (`isAdmitted` / `isAllowedAsOperatorInput`), fire.

4. **Fire = emit.** Build the conclusion expression by applying the binding to `value`. Call `addStatement` ([`prover.hpp`](../../GL_Quick_VS/GL_Quick/src/prover.hpp)) with the expression, target memory block, levels, validity, local flag. Record origin entry into `exprOriginMap`.

5. **Broadcast.** If the new statement is at `validityName == "main"`, also push into `mailOut.statements` for other LBs to consume next cycle.

---

## Locality semantics: a local rule can fire against entirely non-local premises

A rule that lives in `body.localHashMemory{,Delta}` is "local" — this LB owns it. The premises that fire it can be drawn from anywhere visible at this LB, including parent-scope rows the LB sees only via `nm.comparable`. **A local rule firing on 100 % non-local premises is correct rule semantics**, not a bug.

**Both ends of the firing pipeline use `nm.comparable` for scope inheritance ([D-55](../40_decisions.md#d-55) Part C, 2026-05-09).** Request generation has always honoured ancestor/descendant inheritance via the `pairMap`-backed `nm.comparable` (see the 5-batch fan-out below). The firing site (`checkLocalEncodedMemoryStatic`'s head-LMV scope check) was on a stricter strict-eq + main-mask rule until `D-55`'s Part C: pre-fix, a rule at scope `S` could only fire on facts at scope `S` or at `"main"`, even though the matching request had already been emitted under the broader comparable rule. The mismatch was masked while every load-bearing rule of this shape was a v=main universal (chapter-6 / chapter-11 mirror reformulations). When `D-55` Part B removed those parents from `proved_theorems.txt` and the K-impls took over the role at the OR's non-main scope, the strict-eq mismatch surfaced and rejected legitimate cross-scope firings (rung-1 stalled at burst 5 toBeProved=4). Part C realigned the firing-site check to `nm.comparable`, matching the request-generation rule. Both ends now agree on what scopes are mutually visible.

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

Empirical confirmation: in the `IncubatorGauss1` run, 36 of 6398 firings at the SE2 LB carry **zero** premises in `memoryBlock.localEncodedStatementsSet`. The matching Peano universal-x rules have premise `(in[X, 1]) | v=main` among their chain. One example whose mathematical content is uncontroversially sound:

- `(>[1](in[1,u_1])!(in2[1,u_2,u_3]))` — Peano P4 lifted: for any `v ∈ N`, `s(v) ≠ 0` (`0` is not a successor of any natural). One chain element, head is the negation of the in2-successor relation.

The β template that fires at Step 3 of the FTA ladder ([`fta_ladder/rung1/current_proof_state.md`](../../fta_ladder/rung1/current_proof_state.md) §"Step 3 — Peano β template fires at Branch A") has shape `!(=[u_p, u_i0]) → (existence0[u_N, u_p, u_s])` — the negated-equality premise gates predecessor existence. An earlier draft of this paragraph cited a single-`(in[X, N])`-premise version of the β template; that shape, taken literally, would assert *every* `n ∈ N` has a predecessor (false for `n = 0`), so it cannot be the actual installed rule. The original measurement was likely seeing the multi-premise mirror reformulation (anchor + `(in[v, N])` + `!(=[v, i0])`) but only counting the `(in[X, N])` chain element as the "single premise" because the anchor and equality premises had been satisfied via inheritance from v=main / the LB's ancestor scope. The "zero local premises" claim survives that re-reading; the load-bearing point ([D-29 clause 2](../40_decisions.md#d-29) — block disintegration when none of the matched premises is local) does not depend on the exact β-template shape.

The premises are anchor-deposited at v=main; the SE2 LB sees them via `body.intEncodedStatements` but never deposits them into its own `localEncodedStatementsSet`.

**What this implies for disintegration.** Emission of these firings is fine. **Disintegration** of them at a non-anchor LB is what mints fresh `it_*` / `int_*` vars and exhausts the int16_t NameMap in incubator + `!ban_disintegration` mode. The line-2017 gate ([D-29 clause 2](../40_decisions.md#d-29)) is the architectural answer — block disintegration when none of the matched premises is in the LB's `localEncodedStatementsSet`. The gate is *not* redundant with request generation; it corrects for Batches 4/5 at the disintegration site rather than tightening request emission.

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

- **Determinism.** The same input always produces the same set of fires in the same order (modulo LB iteration order, which is stable in the single-thread build).
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
