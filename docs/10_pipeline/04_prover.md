<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Pipeline · Stage 5 — Prover `[DRAFT]`

> **Input:** `files/theorems/filtered_conjectures.txt` (from CE filter) + `files/theorems/compressed_external_theorems.txt` (external theorems + mirrors) + compiled `CoreExpressionConfig` map + `files/theorems/proved_theorems.txt` (when running as later batch in a multi-batch pipeline).
> **Output:** `files/theorems/proved_theorems.txt` (append); per-theorem origin maps used by stage 7 (raw proof graph emission).
> **Owner:** `prover.cpp` / `prover.hpp` / `memory.hpp`.
> **Entry:** `ExpressionAnalyzer::analyzeExpressions` at [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp), called from `run_modes::fullRun` at [`run_modes.cpp`](../../GL_Quick_VS/GL_Quick/src/run_modes.cpp).

---

## What this stage does

The prover is the main proof engine. Given:

- a set of candidate conjectures (from the CE filter),
- a set of externally-provided theorems (axioms + previously-proven theorems),
- the compiled expression map (definitions),

it attempts to derive each conjecture by iterated hash-based inference on a grid of logic blocks, with full provenance recording for every emitted fact. Theorems that get proved are written to `proved_theorems.txt`; theorems that fail stay on the candidate list and are discarded.

The prover's central data structure is `ExpressionAnalyzer` — a single per-batch object that owns all proof state. Every per-theorem proof lives in a child `Memory` block parented to `ExpressionAnalyzer::body`.

---

## Chapter map

- [ExpressionAnalyzer — the state container](#expressionanalyzer--the-state-container)
- [The execution model](#the-execution-model)
- [Theorem load path](#theorem-load-path)
- [Core step primitives](#core-step-primitives)
- [Disintegration](#disintegration)
- [Integration](#integration)
- [`multiplyImplication` — Bell-partition equalisation](#multiplyimplication--bell-partition-equalisation)
- [Induction](#induction)
- [Equivalence classes](#equivalence-classes-and-addstatement)
- [OR & contradiction](#or-and-contradiction)
- [Origin tracking](#origin-tracking)
- [Where invariants live](#where-invariants-live)
- [Weaknesses](#weaknesses)
- [Open questions](#open-questions)

---

## ExpressionAnalyzer — the state container

Declared at [`prover.hpp+`](../../GL_Quick_VS/GL_Quick/src/prover.hpp) and implemented across `prover.cpp`. Instantiated once per batch in `run_modes::fullRun`.

Selected top-level fields:

| Field | Purpose |
|---|---|
| `body` | Root `Memory` block — the top of the per-batch LB tree. |
| `anchorInfo` | Resolved anchor metadata (name, slot count, signature). |
| `anchorID_` | String tag of the batch (`"Peano"`, `"Gauss"`, `"IncubatorPeano"`, …). |
| `coreExpressionMap` | `map<string, CoreExpressionConfig>` — compiled expression configs (from stage 1). |
| `compiledExpressions` | `map<string, LogicalEntity>` — per-name metadata (category, elements, signature, definedSet). |
| `operators` | Set of expression names that are "operators" (have `output_args`). Consulted by Pass B's `isAllowedAsOperatorInput`. |
| `globalTheoremList` | The authoritative list of proved theorems + their method + metadata. Written to `proved_theorems.txt` and `global_theorem_list.txt` after the run. |
| `permanentBodies` | `vector<Memory*>` — every LB that is currently alive (target proofs, sub-goals, auxiliary blocks). Retained across iterations. |
| `permanentBodiesCE` | Same, but the CE-filter-specific grid. |
| `box` / `boxes` | Per-core mailbox arrays for the main grid. |
| `contradictionTable` | Per-conjecture contradiction status (set during CE + main prover). |
| `implCounter`, `existenceCounter`, `andCounter`, `orCounter`, `variableCounter` | Monotone counters — name-minting sources for spontaneous compact operators (`implication<N>`, `existence<N>`, `and<N>`, `or<N>`) and freshly-renamed bound variables. The first four are seeded from the loaded `GL_binary_<Tag>.json` at startup so identifiers stay stable across batches; `variableCounter` resets to zero per batch. |
| `parameters` | `ProverParameters` from `parameters.hpp` + config. |
| `globalDependencies` | Per-expression dependency set maintained during main prover runs. |
| `ceFilteringActive` | Flag distinguishing CE-filter mode from main mode; consulted by hash-request generators. |
| `ceBody`, `indexCE`, `boxesCE` | CE-filter grid mirror of the main `body` / `index` / `boxes`. |

The class is large (`prover.hpp` ≈ 5925 lines) because most per-step primitives are inline methods on `ExpressionAnalyzer`. File-level refactor is on the post-FTA plan.

---

## The execution model

A GL proof run is a sequence of **cycles**. Each cycle:

1. For each active LB, call `performElementaryLogicalStep` — produce hash requests from known expressions, look them up in the local `HashMemory`, emit new statements.
2. After every LB has processed its local state for the cycle, drain mail: move `mailOut` contents into recipient `mailIn` buffers.
3. Next cycle: each LB processes its `mailIn`. Loop until no LB produces anything new, or a per-batch iteration cap is reached.

`prove(numberIterations, permanentBodies, index, boxes)` is the outer loop. It is shared between CE mode (via `permanentBodiesCE`, `indexCE`, `boxesCE`) and main mode.

`performElementaryLogicalStep` at [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp) is the inner core. It:

- Generates hash requests via `generateEncodedRequestsStatic` (singles), `generateEncodedRequestsStaticPairs` (pairs), or `generateEncodedRequestsStaticCE` (CE mode) — depending on `ceFilteringActive` and the per-LB role.
- Matches each request against `overallHashMemory.encodedMap`.
- On a match, fires the rule — builds the conclusion instance, runs `addStatement` to install it locally, and optionally routes it upstream via `mailOut`.
- Handles contradiction detection (`primedForContradiction` flag), OR-branch bookkeeping (`classifyOrScope`), and admission-map updates.

**Warm-up + main iterations.** Per the project conventions, every prover run includes warm-up cycles before the main-iteration budget. Warm-up cycles "prime" the hash memory by letting rule-fire cascades reach steady state.

**Single-pass head-switch model.** `analyzeExpressions` runs exactly **one** prove pass per batch. The contrapositive (head-switched) mirrors of every conjecture that structurally qualifies are folded into the conjecture pool **before** that single prove call, so the prover gets the same proof opportunities in one pass that the legacy multi-iteration path reached across many. Immediately after the prove pass, the same `headSwitchOne` walk runs once more on `globalTheoremList` to populate the class member `orPairsFromHeadSwitch` — `run_modes.cpp` consumes those pairs to construct OR theorems (with disjunct-set dedup and parent-removal at construction time per [D-55](../40_decisions.md#d-55); full mechanics in [`20_core_concepts/07_or_branching.md`](../20_core_concepts/07_or_branching.md#or-theorem-construction-run_modescppfullrun)). The compressor is invoked from `run_modes.cpp::fullRun` *after* `analyzeExpressions` returns (gated by `parameters.skipCompression`) — see [D-25](../40_decisions.md#d-25); it is no longer part of `analyzeExpressions` itself. There is no grid teardown / rebuild / re-prove; the multi-iteration loop and its `bigIteration` / `maxBigIterations` / `pre_emit_head_switch` machinery were deleted at [D-24](../40_decisions.md#d-24).

**Contrapositive construction.** `headSwitchOne` ([`prover.cpp+`](../../GL_Quick_VS/GL_Quick/src/prover.cpp)) is purely structural — it takes the implication string, scans the chain for a `prem[0] == '!'` premise with empty inner bound-vars (the rightmost such premise wins), swaps the head with that premise's body, and rebuilds the implication via direct string concatenation. The same helper backs both the pre-emit pass and the post-prove walk; output is byte-identical between the two call sites.

---

## Theorem load path

Every theorem that enters the prover must first be rewritten by `precompileStructuralOperators` (see [I-1](../30_invariants.md#i-1)). There are three distinct load paths:

### 1. External theorems

Loaded at the start of `analyzeExpressions`. Contents of `files/theorems/compressed_external_theorems.txt` are read line by line; each is precompiled, then passed to `addTheoremToMemory`.

In a multi-batch pipeline, theorems proved in an earlier batch are loaded here for the next batch. Example: the branch's `run_modes.cpp` hard-codes a specific Peano theorem to *remove* from the loaded `proved_set` during the Gauss batch, as a mitigation for the Gauss fold/limitSequence cascade regression — visible at [`run_modes.cpp–150`](../../GL_Quick_VS/GL_Quick/src/run_modes.cpp).

### 2. Conjectures to be proved

After CE filtering, each surviving conjecture is precompiled and registered as a proof target via `addTheoremToMemory(conj, this->body, 0, false, globalDependencies)`. The `false` argument indicates "unproved — this is a goal, not a given".

### 3. Broadcast / mail

When an LB derives a theorem that becomes available to siblings, the broadcast path forwards the theorem via `mailOut.statements`. Recipients invoke `addStatement` on receipt, which internally handles re-precompilation if needed.

**Load-path invariant.** Any new load site must call `precompileStructuralOperators` **first**. See [I-1](../30_invariants.md#i-1) and.

---

## Core step primitives

### `addStatement` — central ingestion

Defined at [`prover.hpp`](../../GL_Quick_VS/GL_Quick/src/prover.hpp). Takes:

- an expression (string),
- the target `Memory` block,
- levels `(statementLevel, equalityLevel)`,
- validity name,
- a `local` flag (whether to also register in local-only delta maps).

Does:

1. Equality mirror: if the expression is `(=[a,b])` with `a!= b` **and `local == true`**, push `(=[b,a])` onto `newStatements` (guarded by `args[0]!= args[1]` — see [I-9](../30_invariants.md#i-9)). The `local` gate ([D-50](../40_decisions.md#d-50)) restricts the mirror push to status 0/1 (local additions); status-3 mail-in absorb skips it because `addEquality(allowSymmetry=false)` did not register the mirror in `statementLevelsMap`, and a push without that registration trips the lookup assert in the post-`addStatement` discharge loop. The mirror's appearance in `newStatements` is what triggers `toBeProved` discharge of the mirrored goal in the kernel's post-loop — disabling this gate (the [](../) `&& false` band-aid) cost 366 incubator-Peano theorems before D-50 restored it.
2. Equivalence-class propagation: if the expression is `!(=[a,b])`, apply `applyEquivalenceClassToNegatedEquality` to emit one-sided sibling inequalities ([I-12](../30_invariants.md#i-12)).
3. Deposit into `wholeExpressions`, `statementLevelsMap`, `intKnownStatements`, `encodedStatements`, `intEncodedStatements`.
4. If `local == true`, also deposit into `localEncodedStatements{,Delta}`, `intLocalEncodedStatements{,Delta}`.
5. Append to `newStatements` (consumed by admission-map logic in the next cycle).
6. For `validityName == "main"`, push to `mailOut.statements` for broadcast.

**Return type — `std::vector<ExpressionWithValidity>` ([D-33](../40_decisions.md#d-33), [I-25](../30_invariants.md#i-25)).** Each entry in `newStatements` is an `ExpressionWithValidity` pair carrying the deposit's actual scope. Same-scope deposits, descendant-scope deposits (D-33 new direction — class deeper than the incoming expression), and ancestor-scope rewrites all flow through this single channel. There is no separate "side sink" for cross-scope deposits.

The kernel's post-`addStatement` loop in [`addExprToMemoryBlockKernel`](../../GL_Quick_VS/GL_Quick/src/prover.cpp) iterates the returned pairs and uses each entry's own `validityName` (the local `effectiveValidity`) for:

- `statementLevelsMap` lookup — the same-scope assert at [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp) holds because the lookup is keyed by the deposit's actual scope.
- `updateAdmissionMapIntegration` / `updateAdmissionMapRecursion` calls.
- `toBeProved` discharge for recursion-LB head match, direct-theorem proof, and validity-name promotion (OR-integration / NotOrScope).

This routing closed the cross-scope discharge gap that the WIP D-33 commit had introduced — see [D-33](../40_decisions.md#d-33) for the incident and the fix.

The asserted precondition on negated-equality entry is `isNegatedEquality(expr)` — callers must gate on the same check. See [`applyEquivalenceClassToNegatedEquality`](../../GL_Quick_VS/GL_Quick/src/prover.hpp) at line 4878.

### `addTheoremToMemory`

Defined at [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp). Takes a theorem string and a target memory block; registers the theorem as an encoded statement, builds its disintegration chain, and creates auxiliary implications via `createAuxyImplication`. This is the path for loading an established theorem (external or previously-proven) as an inference rule.

### `addToHashMemory`

Defined at [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp). Stores a normalised implication chain in `HashMemory::encodedMap`, along with its `LocalMemoryValue` metadata (`value`, `levels`, `originalImplication`, `justification`, `key`, `remainingArgs`, `validityName`, and — D-32 — `productOfDisintegration`). This is the actual installation of an implication rule in the hash engine.

`encodedMap` is populated from **two distinct insertion sites**, both reached from this function:

1. **Head-implication insert** at [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp) — the rule registry entry. The LMV's `.value` is the head's normalised template; the LMV carries full provenance (`originalImplication`, `justification`, `validityName`). This is the LMV that fires when an LB hash-matches the chain. **`productOfDisintegration` is stamped here**, true iff at least one premise has an arg starting with `"u_"`.
2. **Marker LMV insert** at [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp), inside `makeNormalizedKeysForAdmission`. The LMV's `.value` is a marker-form expression (the original premise with its output arg replaced by the literal token `"marker"`); provenance fields default-empty. This LMV fires during Pass B's `int_` admission probe via `isAdmittedIntegration`. `productOfDisintegration` defaults to `false` here — marker LMVs never drive OR-disintegration.

When adding metadata to `LocalMemoryValue`, choose explicitly which insertion site receives it. See [`20_core_concepts/02_hash_engine.md`](../20_core_concepts/02_hash_engine.md#two-encodedmap-insertion-sites-head-lmvs-vs-marker-lmvs) for the full discussion.

After installation, `multiplyImplication` is invoked to generate partition-based equalisation copies (see below).

### `createAuxyImplication`

Defined at [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp). Generates auxiliary implications for recursion: separates digit arguments from immutable arguments (via `findDigitArgs` / `findImmutableArgs`), wraps the implication in rec-chain machinery, and registers the result. Called by `addTheoremToMemory` as part of theorem installation.

### `precompileStructuralOperators`

Defined at [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp). In-place rewrite: replaces every raw `!(&...)` with the compiled `or<N>` name and every `!(>...)` with the compiled `existence<N>` name, recursively. Produces a string that contains no uncompiled structural operators.

Invariant [I-1](../30_invariants.md#i-1): **every theorem-load path must call this**. The assert that blows up inside disintegration when this is skipped is the symptom.

### `findDigitArgs` / `findImmutableArgs`

Defined at [`prover.hpp`](../../GL_Quick_VS/GL_Quick/src/prover.hpp) and [`prover.hpp`](../../GL_Quick_VS/GL_Quick/src/prover.hpp). Extract the "flowing" versus "pinned" arguments of an implication chain. Used by auxiliary-implication construction and by `updateAdmissionMap`.

**Verifier parity.** The Python verifier replicates these algorithms — see `_find_digit_args` and `_find_immutable_args` in `verifier.py`. A divergence between C++ and Python is a sound-ness red flag; [I-16](../30_invariants.md#i-16) applies.

### `reconstructImplication` / `reconstructImplicationFullBind`

Defined at [`prover.hpp`](../../GL_Quick_VS/GL_Quick/src/prover.hpp) and [`prover.hpp`](../../GL_Quick_VS/GL_Quick/src/prover.hpp). Rebuild an implication from a chain + head. The two variants differ in quantifier binding — see [I-4](../30_invariants.md#i-4) and [I-5](../30_invariants.md#i-5). Confusing the two corrupts either theorem-level reconstruction or disintegration/integration round-trips.

### `addExprToMemoryBlock` — `status` parameter reference

`addExprToMemoryBlock` at [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp) is the central deposit primitive called from dozens of sites. Its 4th argument `int status` is a mode selector that controls which deposit path runs. The values are not documented inline at the function site — callers learn them by copy-paste from neighbouring call sites, which is error-prone. Reference table:

| `status` | Path | `forceDeep` arg | Disintegration? | Origin tracked? | Typical caller |
|---|---|---|---|---|---|
| `0` | Local derivation, force-deep disintegrate | `true` | yes | yes | Pass B deferred emissions, OR branch seeds, CE filter admission |
| `1` | Local derivation, normal disintegrate | `false` | yes | yes | Canonical "produced and absorbed by LB" — `addStatement` main branch (`:1975`), fixpoint deposits (`:4648`), `addToHashMemory` head writes (`:4956`), **integration-revival absorb** (new — from `internalMailIn`) |
| `2` | Goal / `toBeProved` insert | n/a | skipped — `prepareIntegration` runs instead | yes (goal-origin) | New head registered for proving; OR branch head emission |
| `3` | External mail absorb, no disintegration | n/a | **skipped** — `status!= 3` gate at `prover.cpp` | yes (origin from `mailIn.exprOriginMap`) | Legacy `mailIn` absorb at `prover.cpp, 2566`. Appropriate when the producer has already handled disintegration — recipients shouldn't re-disintegrate a broadcast. |
| `4` | Fast-path fact load | n/a | skipped (no gate, direct push) | no (levels only) | Simple-facts bootstrap `prover.cpp` |

**Choosing the right status.** Ask: has this expression ALREADY been disintegrated by whoever produced it?

- If the producer is another LB's `mailOut → smashMail → mailIn` routing: yes, disintegration happened at the producer. Use `3`.
- If the producer is *this* LB's own body (eq-class rewrite, OR-branch seed, revival absorb): no, or incomplete. Use `1` (normal) or `0` (force deep).
- If inserting a theorem to be proved (vs asserted), use `2` — `prepareIntegration` runs to set up integration rules.
- If reloading simple facts at grid setup, use `4` — no semantic processing needed.

**Silent-failure mode.** Using `status=3` where `1` is needed (e.g. for revival constituents) deposits the expression into `encodedStatements` and `intKnownStatements` but skips `disintegrateExpr2` — the Pass B admission machinery never runs, `int_` vars never mint, no integration progresses. The statement "arrives" but is inert for further derivation. Observed during the `rejectedMapIntegration` development before the `status=1` fix ([D-19](../40_decisions.md#d-19)).

### Admission / rejection-map key shapes

There are FIVE maps in `HashMemory` that key on variations of a marker-form expression. They are NOT interchangeable — each uses a different prefix convention, and a lookup using the wrong form silently misses. Reference table:

| Map | Field in `HashMemory` | Key shape | Produced at |
|---|---|---|---|
| `admissionMap` (algebra) | `.admissionMap` | `u_`-prefixed on all non-marker args — e.g. `(in3[u_a, marker, u_b, u_+])` | `prover.hpp` (in `updateAdmissionMap` path) |
| `admissionMapIntegration` | `.admissionMapIntegration` | `u_`-prefixed on all non-marker args — same as above, integration-side | `prover.hpp` — built from `le.signature` in `prepareIntegrationCore2` Case C |
| `admissionSetIntegration` | `.admissionSetIntegration` | **bare concrete with `repl_` preserved** — e.g. `(in3[repl_lev_2_0, marker, repl_lev_2_3, 4])` | `prover.cpp` — `removeUPrefixFromArguments(mappedElement)` (strips `u_`, keeps `repl_`) |
| `rejectedMap` (algebra) | `.rejectedMap` | **bare concrete** — matches Pass B's `makeMarkedExpr` output | `prover.hpp` (via `updateRejectedMap`) — key built by Pass B at `prover.cpp` |
| `rejectedMapIntegration` | `.rejectedMapIntegration` | **bare concrete** — same shape as `rejectedMap` | `prover.hpp` (via `updateRejectedMapIntegration`) — key built by Pass B at `prover.cpp` |

**Bridging between shapes.** A few gateways exist:

- `isAdmittedIntegration` at `prover.hpp` — converts Pass B's bare marker form to `u_`-prefixed before looking up `admissionMapIntegration`. Mandatory transform for the lookup to hit.
- `removeUPrefixFromArguments` — strips `u_` from args in-place. Used at `prover.cpp` to go from `u_`-prefixed template to `admissionSetIntegration`'s `repl_`-preserving form, and used in the integration-revival path (`applyEquivalenceClassToRejectedMapIntegration` at `prover.hpp`) to strip `u_` off the admissionMap key before probing `rejectedMapIntegration`.

**Silent-failure modes.**

- Revisiting `rejectedMapIntegration` with the `u_`-prefixed key (without stripping): every lookup misses. No rejections get revived. The revival mechanism fires but accomplishes nothing. See [G-32](../50_gotchas.md#g-32-rejectedmapintegration-key-shape-bare-concrete-not-u_-prefixed) for the concrete debug trail.
- Probing `admissionMapIntegration` with a bare key (without `u_`-prefixing): same — every probe misses.
- Probing `admissionSetIntegration` with a `u_`-prefixed key: misses (the set stores bare `repl_`-form).

Rule of thumb: think of `u_` as the "universal template" prefix and the bare form as the "concrete runtime instance." Admission MAPS (map + mapIntegration) store templates (`u_` everywhere). Admission SET (`admissionSetIntegration`) and rejection MAPS (both) store concrete instances (no `u_`). Always transform at the boundary.

---

## Disintegration

Disintegration is the prover's "expand compound expressions into their parts" operation. Two paths:

### Pass A — structural disintegration

The standard path: when an `and` or `existence` node is received, split it into its element expressions via `disintegrateExprCore2` and recurse. Straightforward.

### Pass B — operator-aware disintegration

Defined at [`disintegrateExpr2`](../../GL_Quick_VS/GL_Quick/src/prover.cpp) (`prover.cpp`). Produces sets of new statements and integration expressions with iteration tracking.

Central complexity: freshly-minted `it_…` variables must be *admitted* before they enter `finalStringStatements`. Two admission paths compete:

1. **Map-based admission** — `isAdmitted` at [`prover.hpp`](../../GL_Quick_VS/GL_Quick/src/prover.hpp). Consults `admissionMap` (populated by `updateAdmissionMap` from consumer-side registrations). Authoritative; failure routes to `pendingRejections`.
2. **Single-input-operator standalone** — `isAllowedAsOperatorInput` at [`prover.hpp`](../../GL_Quick_VS/GL_Quick/src/prover.hpp). Fallback; fires only for `reIt`-matching variables. Admits iff:
 - `extractExpression(stmt)` is in `ExpressionAnalyzer::operators`;
 - `cfg.inputIndices.size == 1` — **single-input operators only** (in practice `in`, `in2`);
 - The lone input-arg position holds the variable;
 - Standard guards pass (`maxIterationNumberVariable`, `maxNumberSecondaryVariables`).

First-match-wins — the outer loop `break`s as soon as one path admits. The standalone rule does not consult the admission map at all. This is the guardrail against RT explosion: widening to multi-input operators broke Gauss summation. See [I-6](../30_invariants.md#i-6).

**The `int_…` path** has two admission checks: map-based `isAdmittedIntegration` against `admissionMapIntegration`, then `admissionSetIntegration` fallback. Porting the `isAllowedAsOperatorInput` single-input rule into the `int_` branch is forbidden without careful RT measurement.

### `int_` deferred-rejection buffer (`rejectedMapIntegration`)

As of commit landing [D-19](../40_decisions.md#d-19), the `int_` branch also has a deferred-rejection buffer symmetric to the `it_` branch's `pendingRejections → rejectedMap` flow. When a fresh `int_` variable fails both `isAdmittedIntegration` and `admissionSetIntegration`, the rejection is NOT silently dropped — it is buffered into `pendingRejectionsIntegration` and, if the cascade-admission loop still leaves the var un-admitted, committed to `HashMemory::rejectedMapIntegration` via `updateRejectedMapIntegration`.

Shape:

| Field | Role |
|---|---|
| `rejectedMapIntegration` key | `ExpressionWithValidity(markedConstituent, validityName)` — same shape as algebra's `rejectedMap` key, keyed on the marker form (`int_var → "marker"`) of the body element. Skipped for `(in[...])` typing elements per user spec (they are never a key but still carried in siblings). |
| `rejectedMapIntegration` value (`RejectedMapIntegrationValue`) | `concreteConstituent` (body element with `int_` arg intact) + `siblings` (all OTHER body elements of the compound, incl. `(in[...])` typing) + `compoundExpression` (for origin). No `iteration` field — `int_` mint is level+startInt only. |

Revival paths (once stored):

- **Equivalence-class rewrite.** `applyEquivalenceClassToRejectedMapIntegration` ([eq-classes chapter](../20_core_concepts/05_equivalence_classes.md#extension-rejectedmapintegration)) rewrites keys when the class touches their args; on matching admission, emits rewritten constituents to `internalMailIn`. Uses the full `allMappingsAna` permutation loop (same machinery as the main helper), with per-entry dedup across mappings that collapse to the same rewritten key.
- **New-admission-key trigger.** `revisitRejectedIntegration2` fires at `admissionMapIntegration` insert (`prover.hpp`, with u_-strip — see [G-32](../50_gotchas.md#g-32)) and at `admissionSetIntegration` insert (`prover.cpp`, bare-form). Emits matching `rejectedMapIntegration` entries to `internalMailIn`.

Both revival paths deposit to `internalMailIn` only — no `addExprToMemoryBlock` call. See [D-19](../40_decisions.md#d-19) for why (cyclic-re-entry avoidance). The `internalMailIn` absorb at the top of next hashburst uses `status=1` (full disintegration pipeline), not the legacy mail's `status=3`.

Asymmetries with algebra-side:

- Integration revival does **NOT** call `cleanAdmissionMap` on the admission-map entry after successful revival (see [I-22](../30_invariants.md#i-22)). The admission rule stays live for future matching rejections.
- Integration revival does NOT cascade through `addExprToMemoryBlock`. Linear path only.

### Disintegration gate

Pass B fires when `!parameters.compressor_mode && !parameters.ban_disintegration` (see [I-7](../30_invariants.md#i-7)). `ban_disintegration` also gates back-reformulation, hypothetical disintegration, and necessity-for-equality-hypo — every disintegration-shaped path in the prover. Pre-2026-04-29 the gate was `!parameters.incubator_mode`; that decoupling-from-`incubator_mode` is the lasting change ([D-27](../40_decisions.md#d-27)). A short-lived `allow_disintegration` flag was introduced and then collapsed into `ban_disintegration` later the same day ([D-28](../40_decisions.md#d-28)) once the per-config matrix turned out to be symmetric (`ban_disintegration == !allow_disintegration` in every config). `ConfigIncubatorGauss1.json`'s SE2-migration combination — Pass B on, `multiplyImplication` off, `incubator_mode` on — sets `ban_disintegration=false` to fire Pass B.

### `savedStartInt` freshness

Pass B's freshness check uses `savedStartInt` — a monotonically-increasing counter. A parallel counter breaks the contract; see [I-17](../30_invariants.md#i-17).

### `performDisintegration`

the project conventions lists `performDisintegration` as a key function; the current source search returned "not found" (see OPEN-6 in [`AGENT_SwDD.md`](../AGENT_SwDD.md#open-questions)). The actual current code has `disintegrateExpr2` plus a set of related helpers; `performDisintegration` may be a historical name. Requires clarification on the next expansion pass.

---

## Integration

Integration is the mirror of disintegration: reassemble a compound expression from its constituents so it can be asserted as a conclusion.

Three preparatory tags appear in the processed proof graph:

- `reformulation for integration and` — for `and`-category conclusions.
- `reformulation for integration >[bound]` — for `existence`-category conclusions whose outermost `>[...]` bound-variable list is non-empty; the bound variable is carried through.
- `reformulation for integration >[]` — for `existence`-category conclusions whose outermost `>[...]` list was stripped because the witness slot was already occupied by an external value.

Helpers involved:

- `expandSignatureForIntegration` — uses `reconstructImplicationFullBind` (see [I-4](../30_invariants.md#i-4)).
- `buildIntegrationInstruction` — ditto.
- `updateAdmissionMap` — consumer-side output-slot registration, feeding the disintegration admission map.

Each integration step emits a `validity name` tag row — declaring the scope into which the integrated expression is bound — and then an `expansion for integration` tag row showing the re-expansion into compiled form.

---

## `multiplyImplication` — Bell-partition equalisation

Defined at [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp). Invoked by `addToHashMemory` on every newly-installed implication.

**What it does.** Collects every variable that appears at a `(1)`-typed argument position anywhere in the rule (both bound and free `u_*` anchor parameters), then enumerates Bell partitions of that set. Each partition defines one "equalisation copy" where the variables in the same class are forced equal simultaneously. Each copy is installed as its own rule — broadening the implication's applicability.

**Example.** An implication `(>[v1,v2,v3](P(v1,v2,v3)))` with three bound `(1)`-typed variables has the Bell partitions:

- `{{v1},{v2},{v3}}` — no equalisations.
- `{{v1,v2},{v3}}`, `{{v1,v3},{v2}}`, `{{v2,v3},{v1}}` — one pair equalised.
- `{{v1,v2,v3}}` — all three equal.

For each partition except the original, `multiplyImplication` produces the equalised copy, passes it through `precompileStructuralOperators` (if needed), and registers it as a separate rule.

**Why.** Some theorems hold only when two or more bound variables are identified; `multiplyImplication` surfaces those cases automatically without requiring the conjecturer to have enumerated them.

**Soundness gate (no double-`u_` equivalence classes).** A partition is *skipped* whenever any equivalence class would contain two or more distinct free `u_*` anchor parameters. Free anchors are pre-bound to specific elements of the underlying definition sets — equating two distinct ones would silently rewrite a free slot of the rule body, producing a logically stronger rule than the source. Bound-and-free merging (one `u_*` together with one or more bound vars) is still permitted: that is the documented mechanism for specialising a rule's bound variables to a known constant.

Emitted chapters carry the `multiplied from` tag (see [Proof tags](../20_core_concepts/08_proof_tags.md)).

---

## Induction

Induction is scheduled on a bound variable `n` identified structurally (currently: `(1)`-typed bound variables that appear in certain canonical positions). The prover creates sub-goal LBs for:

1. **Base case** — substitute `i0 = 0` for `n`; prove the head under the substituted premises.
2. **Step case** — assume the head for `n`; prove the head for `s(n)`.

Promotion of an induction theorem to `globalTheoremList` uses `method = induction` with the induction variable recorded in the reference column.

### The typing gap (current debug focus)

See [I-18](../30_invariants.md#i-18) and [`docs/induction_typing_plan.md`](../induction_typing_plan.md). The historical prover scheduled induction on any structurally-typed bound variable without first verifying `(in[n, N])`. For bound variables appearing only in negations, bare equalities, or existence heads, this is unsound — the theorem ranges over all entities, not just over `N`.

The fix is in progress on the current branch:

- A third recursion sub-block `tempMb3` is created at induction setup, with `(in[digitArg, anchor_args[0]])` as its head.
- `parameters.typingProofOnly = true` during the typing proof, forbidding induction re-entry.
- If the typing sub-block does not complete successfully, the induction is silently rejected (no `globalTheoremList` insertion).
- A chapter file `<N>_induction_typing.txt` is emitted with all the usual tags (direct-proof).
- The verifier adds a new `induction typing` checker that walks every `method = induction` row and verifies its accompanying typing chapter.

See the plan file for the full staged implementation.

---

## Equivalence classes and `addStatement`

`Memory::equivalenceClassesMap` is per-validity. Each entry is an `EquivalenceClass` grouping variables known-equal within that validity. When `addStatement` receives `(=[a,b])`, it updates the relevant class; when it receives `!(=[a,b])`, it emits sibling inequalities one-sidedly via `applyEquivalenceClassToNegatedEquality` (see [I-12](../30_invariants.md#i-12)).

The deliberate one-sidedness — not the symmetric cross-product — is the combinatorial containment knob. Two-sided expansion would blow up without clear semantic gain.

### Pair-invariant asserts

`addStatement` carries several pair-invariant asserts (the equality-mirror guard, the negation-shape guard, the validity-namespace guard). These are load-bearing — weakening one to "make a failing test pass" is exactly the [I-19](../30_invariants.md#i-19) anti-pattern. Memory file records the rationale.

---

## OR and contradiction

### OR disintegration / convergence

`classifyOrScope` at [`prover.hpp`](../../GL_Quick_VS/GL_Quick/src/prover.hpp) classifies a validity scope as `NotOrScope` / `Integration` / `Disintegration` based on the NameMap stack payload. Used by:

- `performElementaryLogicalStep` — to decide whether to run OR-aware bookkeeping.
- `ordisMerge` (inline member in [`prover.hpp`](../../GL_Quick_VS/GL_Quick/src/prover.hpp), called from `addExprToMemoryBlockKernel`'s post-`addStatement` loop) — `_ordis_` convergence detection, per-branch cleanup of the converged expression, parent-scope promotion via `internalMailIn`. See [D-34](../40_decisions.md#d-34).
- `cleanUpOrIntegrationBranches` at [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp) — wipes sibling `_orint_` branches' statements when one branch proves the OR-introduction goal's head.

OR disintegration creates a child LB per disjunct, with the other disjuncts' negations seeded as branch-local assumptions. OR convergence fires when every child has independently reached the same conclusion — the conclusion is then promoted to the parent scope.

The OR-disintegration admission gate at `disintegrateExprCore2`'s OR case (per [D-32](../40_decisions.md#d-32)) reads a threaded `allowOrDisintegration` parameter — set by `checkLocalEncodedMemoryStatic` only when the firing implication is a product of disintegration (`LocalMemoryValue::productOfDisintegration`, stamped at install time in `addToHashMemory` based on the presence of a `u_*` arg in any premise). Coupled with `addExprToMemoryBlock`'s `doNotDisintegrate` so OR-disint cannot fire when general disint is forbidden. See [`20_core_concepts/07_or_branching.md`](../20_core_concepts/07_or_branching.md) for the full walkthrough.

### Contradiction discharge

Integrated into `performElementaryLogicalStep` via the `primedForContradiction` flag on the target `Memory`. When a contradicted branch reaches both `X` and `!X`, the prover reconstructs the negated-head implication and stores it in `contradictionTheorem`. The assumption that started the branch is then discharged and the *original* conclusion (the negation of the starting assumption) is emitted into the parent scope.

Memory: `primedForContradiction` keeps contradicted LBs alive in `deactivateRecursively` so the discharge can fire.

### `reactToHypo`

Defined at [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp). Handles the hypothetical-disintegration variable-copy tag path — scopes marked with `_hypo_` payloads. In the current code it handles the book-keeping; the user-facing tag in the processed proof graph is `variable copy` (having subsumed the retired `reaction to hypo` and `necessity for equality (hypo)` tags).

---

## Origin tracking

Every emission records provenance into `exprOriginMap` (field of `Memory`). An entry maps:

```
(expression, levels) → list<origin-citation>
```

where each `origin-citation` names:

- The rule that fired (the encoded implication key).
- The premises in `(expression, levels)` form that matched the rule's chain.
- The justification category (corresponds to the proof-graph tag).

When stage 7 (raw proof graph emission — `visualizer.cpp`) runs, it walks `exprOriginMap` backwards from each proved theorem to build a chapter: `buildStack` at [`visualizer.cpp`](../../GL_Quick_VS/GL_Quick/src/visualizer.cpp).

The origin map is the raw material of every subsequent auditability claim. A missing origin entry means the verifier has to fail the theorem; a *wrong* origin entry (emitting a justification that isn't actually valid) is the silent-unsoundness case verified against.

### Origin map locality (D-51)

Every origin record is **local to the LB that produced it**. Mail transports (parent → children, via `mailOut → sendMail → child.mailIn → child's exprOriginMap on next iteration`) carry origins **downward only**. There is no upward propagation: a child LB's origin records do not reach the parent's `exprOriginMap`. The `pendingAncestorOrigins` queue in `proveKernel` (originally a child-side `parentMemory`-walk that wrote a contradiction LB's `("contradiction", deps)` record into every ancestor's `exprOriginMap`, deferred for thread safety per D-39) is **retired** as of D-51 — the queue is never pushed and the drain iterates an empty container. Cleanup of the dead struct/mutex/loop is deferred.

The principled consequence for chapter emission: a contradiction LB's recipe lives only inside the `__contradiction__(head)` LB. The chapter walker (`buildStack`) enters that LB explicitly via the chapter-boundary `__contradiction__` simpleMap fallback when a negated head has no acyclic direct origin in the LB before the head. This replaces the [D-49](../40_decisions.md#d-49) cap-full preference rule (now superseded — the cap-full preference logic remains in `addOrigin` but is rarely triggered with `max_origin_per_expr = 30` and is no longer load-bearing).

### `buildStack` chapter walker (D-51 algorithm + lifting per `D-56`)

`buildStack` at [`visualizer.cpp`](../../GL_Quick_VS/GL_Quick/src/visualizer.cpp) walks the proof tree starting from the chapter goal and emits one row per visited expression. Four mechanisms work together:

1. **Ancestor lifting.** At entry, `proved` is lifted via `liftToShallowestOriginAncestor(memoryBlock, proved)` — the helper walks the validity-stack ancestor chain of `proved.validityName` (recoverable from the canonical `parent + "_boundary_" + payload` string per [I-2](../30_invariants.md#i-2)) from `main` outward and returns the **closest-to-`main` ancestor** for which `(expr, ancestor)` exists as a key in `memoryBlock.exprOriginMap`. From this point on, `proved` is the lifted form: chapter rows are emitted with the lifted validity, the path-cycle filter inserts/erases the lifted form, and every dependency is independently lifted before being written into a row cell and before recursion. The dep-side lift uses the same helper. Net effect: every chapter cell (`row[1]` and every `row[3+2k+1]`) carries the lifted (closest-to-`main`-with-origin) validity, and each `(expr, lifted_v)` pair has at most one row per chapter via the `covered` dedup set. The pre-lifting fallback ("exact-key try, then `(expr, "main")` shadow lookup") is retired: lifting subsumes it and produces a truthful `row[1]` instead of one that re-tagged the row with the requested deep validity while citing a shallower origin.

 **OR-branch barrier.** The lift may not cross `_boundary_orint_` or `_boundary_ordis_` delimiters. OR-branch scopes are conditional on a disjunct hypothesis; preserving branch-distinct namespaces is required by the OR-family verifier checkers (`check_or_convergence`, `check_or_branch_proven`, `check_or_branch_assumption`, `check_or_disintegration`). The deepest `orint_`/`ordis_` ancestor sets the shallowest allowed lift target. Without the barrier, both branches of an OR converge to identical `(expr, parent_boundary)` cells (the parent boundary holds the post-convergence origin), breaking the verifier's branch-distinctness expectation and producing `origin chain termination` cycles. Boundary detection: payload (segment after `parent + "_boundary_"`) starts with `"orint_"` or `"ordis_"`.
2. **Path-cycle filter.** A `thread_local std::set<ExpressionWithValidity> g_buildStackPath` tracks the recursion path. Before picking an origin candidate, any candidate whose deps (after lifting) include an expression already on the path is rejected (would form a chapter-row cycle the verifier's `origin chain termination` check would flag).
3. **Chapter goal in path.** At `directStack` entry, the chapter goal expression (the wrapped theorem from `theoremList`) is inserted into `g_buildStackPath` and erased on return. The path-cycle filter then rejects the prover's self-applying forward-inference origin (origin tag `implication` with deps `[wrapped theorem, anchor]`) — the chapter walker would otherwise emit a `theorem`-tag leaf row matching the chapter goal, tripping `verifier.py::check_chapter` self-reference.
4. **`__contradiction__` LB fallback.** When all sorted candidates of a negated head fail (rejected by path-cycle filter, or no direct origin existed for the lifted `proved`), `buildStack` walks the LB chain (current → ancestors via `parentMemory`) for a child memory block keyed `"__contradiction__" + positive`. On hit, it switches into that contradiction LB and resolves the head locally — the contradiction record is local there.

`buildStack` performs **no other LB switch**. Each contradiction is an independent LB with its own chapter; nested contradiction LBs are NOT children of an outer contradiction LB. Recursion on a contradiction's three deps stays in the current LB; if a dep itself needs a contradiction proof, the chapter-boundary fallback the next level down enters another sibling `__contradiction__` LB attached to the anchor.

Sorting / backtracking. Per-key origin candidates are ordered by [D-49-style](../40_decisions.md#d-49) preference (non-equality tags first preserving insertion order, then `equality1`/`equality2` preserving insertion order). `buildStack` returns `bool`: `true` on success, `false` to signal the caller (recursive `buildStack`) to backtrack — roll back stack/covered snapshots and try the next candidate. Top-level callers (`directStack`, `checkZeroStack`, `checkInductionConditionStack`) ignore the return.

**TRIPWIRE** at function entry: the hypothetical-disintegration sentinel scope (`_boundary_hypothetical_disintegration`) is checked on the **incoming** `provedIn` before lifting, because the sentinel is a structural probe — it is not an ancestor of anything legitimate and must never reach buildStack regardless of where the consumer might lift it.

**Soundness of lifting.** An origin entry at `(expr, V)` in `exprOriginMap` exists only because the prover fired a rule producing `expr` with every premise available at scope `V`. Per [I-2](../30_invariants.md#i-2), every non-root scope inherits all parent-scope facts. Therefore: lifting from `(expr, V_deep)` to `(expr, V_root)` where `V_root` is the closest-to-`main` ancestor with origin → the rule did fire at `V_root` (origin was recorded there), so its premises were available at `V_root` or shallower → the lifted row truthfully claims `V_root`. The asymmetric direction holds: ancestor facts are universally available, descendant-only facts are not — and lifting only moves toward `main`, never away from it.

---

## Theorem export at end of batch

When `proveKernel` exits its outer iteration loop and the batch terminates, the prover writes the proved-theorem set to disk in two parallel files. Both files carry the same theorem set; the only difference is the form of compiled structural operators (`existence0..N`, `or0..N`, `and0..N`, etc.).

### The two-file dual-form export

[`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp):

```cpp
// Second file: compiled forms (for proof graph pruning — keeps or0, existence2, etc.)
const auto compiledPath = theoremsDir / "compiled_proved_theorems.txt";
std::ofstream ofsCompiled(compiledPath, std::ios::trunc);

int written = 0;
for (const std::string& theorem : essentialTheorems) {
    if (externalTheorems.count(theorem) || mirroredOfExternal.count(theorem)) continue;

    auto it = compactToExpanded.find(theorem);
    std::string toWrite = (it != compactToExpanded.end()) ? it->second : theorem;

    // compiled_proved_theorems.txt: write as-is (keeps compiled heads)
    if (ofsCompiled.is_open()) ofsCompiled << toWrite << "\n";

    // proved_theorems.txt: expand runtime expressions (or0, etc.) to base form
    // so subsequent batches can parse them.
    toWrite = expandToBaseForm(toWrite);
    ofs << toWrite << "\n";
    ++written;
}
```

`essentialTheorems` is the compressor-survivor set (the theorems that remain essential after [`05_compressor.md`](05_compressor.md)). Surviving externals and their mirror variants are skipped — they weren't proved here, they were imported.

For each survivor, the loop emits:

- **`compiled_proved_theorems.txt`** — compact form. Keeps every compiled head (`existence2`, `or0`, …) as the literal token allocated during this batch's run. Truncate-each-batch (`std::ios::trunc`).
- **`proved_theorems.txt`** — same content, but `expandToBaseForm` rewrites every compiled head into its base-form definition. `existence2[N,i,s]` becomes `!(>[8](in[8,N])!(in2[8,i,s]))` (the `¬∀y∈N: ¬in2[y,i,s]` definition); `or0[a,b]` becomes `!(&!(a)!(b))`; etc.

### Why two forms

The expansion at line 7817 exists for one specific reason: a downstream batch (e.g. Gauss main reading Peano main's theorems) might not have the same compact-name dictionary loaded at parse time. Compact heads are batch-local allocations — Peano's `existence2` is *Peano's* compact name for that expansion. If Gauss main saw `existence2` as a raw token without the Peano dictionary, parsing would fail. Writing `proved_theorems.txt` in expanded form gives downstream batches an operator-dictionary-free input.

`compiled_proved_theorems.txt` keeps the compact heads because **proof-graph artifacts also use compact heads**:

- `visualizer.cpp` writes raw chapters with `rest[]` fields citing rules in compact form (whatever the prover's runtime had in `exprOriginMap`).
- `process_proof_graphs.py` propagates these compact citations into the processed chapter rows.
- The verifier's `origin` check (see [`08_verifier.md`](08_verifier.md#origin-meta-check)) looks `rest[0]` up in the registry by alpha-canonical match — without operator expand/compact normalisation. So the registry has to contain the rule in the same form the chapter cites: compact.

This duality is load-bearing. Removing either side breaks something:

- Drop `compiled_proved_theorems.txt` → the Stage-2 verifier-registry route (see [D-40](../40_decisions.md#d-40)) loses its compact-form source; chapter `origin` lookups miss.
- Drop the `expandToBaseForm` step in `proved_theorems.txt` → downstream-batch parsing of non-shared-binary compact heads fails.

### Cross-batch propagation: which file feeds the next batch's externals

The orchestrator (`run_modes.py`) selects between the two files when seeding a downstream batch:

| Channel | Source file | Reader | Form |
|---|---|---|---|
| Inter-batch parser feed (default) | `proved_theorems.txt` | C++ prover loading externals at startup | expanded |
| Externals seed for next-tag incubator (since [D-40](../40_decisions.md#d-40)) | `compiled_proved_theorems.txt` | C++ `--mirror-externals` mode + verifier registry | compact |

The compact-form route is safe because `GL_binary_shared.json` (`run_modes.py:_seed_per_batch_binary` + `_merge_into_shared`) carries the spontaneous-category compact dictionary across batches — see [`09_incubator.md` § Cross-batch externals seed](09_incubator.md#cross-batch-externals-seed).

### Per-batch GL binaries

Alongside the two theorem files, the prover writes `files/GL_binaries/GL_binary_<tag>.json` — the compact-name dictionary specific to this batch. Schema entries are name → `{category, signature, …}`. The write path is project-rooted (computed from `__FILE__` ascent in `visualizer.cpp::generateRawProofGraph`), so every batch — incubator and main alike — writes into the single canonical `files/GL_binaries/` directory; see [D-54](../40_decisions.md#d-54) for the unification of a previously asymmetric writer path.

The orchestrator post-batch step `_merge_into_shared` (`run_modes.py`) selects the spontaneous categories (`implication`, `existence`, `or`, `and`) and unions them into `GL_binary_shared.json`, which the next batch's `_seed_per_batch_binary` (`run_modes.py`) copies back as that batch's per-batch starting dictionary. **Tags whose name begins with `Incubator` are skipped** — incubator-allocated spontaneous names are batch-local; merging them into shared would shift the next main batch's counters and rename its operators, breaking [I-23](../30_invariants.md#i-23) for the main pipeline.

Non-spontaneous categories (anchor entries, atomic entries) stay in the per-batch file only — they are by-design batch-local and never cross.

### File-write summary

| File | Form | Truncate semantics | Audience |
|---|---|---|---|
| `files/theorems/proved_theorems.txt` | expanded | per-batch trunc (`prover.cpp`) | inter-batch parser-input fallback |
| `files/theorems/compiled_proved_theorems.txt` | compact | per-batch trunc | proof-graph pruning + cross-batch externals seed (D-40) |
| `files/theorems/compressed_external_theorems.txt` | rewritten by `--mirror-externals` | per-step write | C++ prover externals load |
| `files/GL_binaries/GL_binary_<tag>.json` | per-batch compact dictionary | per-batch overwrite | next batch's startup load |
| `files/GL_binaries/GL_binary_shared.json` | accumulating spontaneous-category union | append (entries not overwritten) | cross-batch dictionary |

---

## Where invariants live

| Invariant | Code location |
|---|---|
| [I-1](../30_invariants.md#i-1) — precompile before theorem-load | `precompileStructuralOperators` at [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp) |
| [I-2](../30_invariants.md#i-2), [I-3](../30_invariants.md#i-3) — scope mint + ref copy | `NameMap::encodePush` at [`memory.hpp`](../../GL_Quick_VS/GL_Quick/src/memory.hpp) |
| [I-4](../30_invariants.md#i-4), [I-5](../30_invariants.md#i-5) — FullBind vs non-FullBind | `reconstructImplication*` at [`prover.hpp`](../../GL_Quick_VS/GL_Quick/src/prover.hpp), [`prover.hpp`](../../GL_Quick_VS/GL_Quick/src/prover.hpp) |
| [I-6](../30_invariants.md#i-6) — single-input Pass B gate | `isAllowedAsOperatorInput` at [`prover.hpp`](../../GL_Quick_VS/GL_Quick/src/prover.hpp) |
| [I-7](../30_invariants.md#i-7) — Pass B gated on `!incubator_mode` | `disintegrateExpr2` at [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp) |
| [I-12](../30_invariants.md#i-12) — equivalence-class one-sided expansion | `applyEquivalenceClassToNegatedEquality` at [`prover.hpp`](../../GL_Quick_VS/GL_Quick/src/prover.hpp) |
| [I-17](../30_invariants.md#i-17) — `savedStartInt` single-counter | `disintegrateExpr2` at [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp) |
| [I-18](../30_invariants.md#i-18) — induction typing | shipped; auxiliary `(in[digitArg, N])` triad before promotion, `<N>_induction_typing.txt` chapter, verifier `induction typing` checker |

---

## Weaknesses

### Known & tracked

- **Induction-typing soundness — closed.** Historically the prover scheduled induction without first verifying membership in `N`, which was unsound for bound variables appearing only in negations / existence heads / bare equalities. **Closed:** the prover now proves the typing triad before promotion; see [I-18](../30_invariants.md#i-18). Listed here for historical context.
- **Pass B single-input gate is empirical.** [I-6](../30_invariants.md#i-6) is enforced because widening broke Gauss. There is no theoretical argument here — only a measurement-grounded guardrail. A hypothetical operator with single-input semantics but admission-map requirements that the gate does not check would slip through.
- **Hard-coded Peano theorem drop.** On the current branch, `run_modes.cpp–150` hard-codes a filter that removes a specific Peano theorem from the inherited `proved_set` at the start of the Gauss batch. This is a known debug mitigation, not a shipping fix. See commit.

### Suspected fragility

- **`ExpressionAnalyzer` is a monolith.** ≈5925 lines in `prover.hpp` + ≈8770 in `prover.cpp`. Every proof primitive is a method on the same object. Refactor planned post-FTA (see ).
- **`logicalCores = 1` hardcode.** Per: `logicalCores` is fixed to 1; parallel `smashMail` was explicitly rejected in a previous session. This means the mail-batching code paths exist but are never exercised with `batchSize > 1`.
- **`performDisintegration` phantom.** the project conventions names this function, but a grep does not find it — possibly renamed/retired. Stale documentation in the project conventions is a minor fragility but a symptom: doc drift on a core surface. See [OPEN-6](../AGENT_SwDD.md#open-questions).
- **Proof-cycle iteration cap.** `parameters.numberIterationsConjectureFiltering` and the main-mode iteration budget are batch-wide constants. A single unusually-hard conjecture can consume the full budget while the rest starve. No per-conjecture timeout.
- **`savedStartInt` freshness contract.** Any refactor that introduces a parallel counter violates [I-17](../30_invariants.md#i-17) silently.
- **Operator set inference.** `ExpressionAnalyzer::operators` is populated somewhere — OPEN-4 in AGENT_SwDD.md. A mismatch between this set and the conjecturer's notion of "operator" could silently alter Pass B behaviour.

### Worked example — a Gauss revival, end to end

Concrete trace of the mechanism on the first theorem it unblocked: `sequence[1,4,2,12,10] → sequence[1,4,2,9,11]` under `limitSequence[1,4,9,10,11] ∧ in2[9,12,3]` (theorems.txt line 296 on the pre-merge branch). Anchor mapping `AnchorGauss[N, i0, s, +, *, i1, i2, id] = [1..8]`, so `4 = +`, `2 = i0`, `3 = s`.

**Cycle N — Pass B rejection.** Somewhere mid-Gauss, a compound of shape `(sequence[1, 4, 2, X, Y])` lands at some LB at `v=main`. `disintegrateExprCore2` (prover.cpp) unfolds it through `fXY / interval` expansion and mints an integration witness `int_lev_2_42` (level 2, startInt 42) for one of the inner `preorder` existentials. One body element of that existential is `(in3[a, int_lev_2_42, b, 4])` — "a + int_lev_2_42 = b in +", with `a`, `b` concrete from the compound's args.

Pass B's `int_` branch at `prover.cpp`:

1. `isAdmittedIntegration(mb, removedU="(in3[a,int_lev_2_42,b,4])", var="int_lev_2_42", markedExpr="(in3[a,marker,b,4])", …)` u_-prefixes non-marker args → looks up `admissionMapIntegration[(in3[u_a, marker, u_b, u_4])]` → miss (no template registered for this specific concrete `a`/`b`).
2. `admissionSetIntegration.find((in3[a, marker, b, 4]))` → miss.
3. Fall-through. New code kicks in: builds `siblings = [(in[int_lev_2_42, 1])]` (the typing element, preserved for re-emission), packages into `PendingRejectionIntegration{concreteConstituent, markedExpr, siblings}`, buffers. After cascade admission loop (around `prover.cpp`), the var is still un-admitted → `updateRejectedMapIntegration(markedExpr, concreteConstituent, siblings, topLevelExprClean, mb.overallHashMemory, "main")` fires.

State of `mb.overallHashMemory`:

```
rejectedMapIntegration[(in3[a,marker,b,4]) @ main] = {
    { concreteConstituent = "(in3[a,int_lev_2_42,b,4])",
      siblings            = ["(in[int_lev_2_42,1])"],
      compoundExpression  = "(sequence[1,4,2,X,Y])" }
}
varsInRejectedMapIntegrationKeys += {a, b, 4}
```

No visible fact was added to `encodedStatements`; the compound proof is stuck on this rung.

**Cycle N+k — equivalence class propagation.** Some subsequent LB proves `(=[X, 9])` (X can be identified with 9 given the other premises). The equality is absorbed at `v=main`; `updateEquivalenceClasses` creates/merges class `{X, 9}` with canonical `9` (say lex-min). `applyEquivalenceClass` (main helper) rewrites matching `encodedStatements`. Simultaneously, the new `applyEquivalenceClassToRejectedMapIntegration` fires at the same call sites (`prover.hpp` same-NS, `5982-5988` ancestor-NS, fixpoint re-iter at `6049`).

The helper:

1. `varsInRejectedMapIntegrationKeys` contains `{a, b, 4,...}`. Class `{X, 9}` — if `X` or `9` is in the cache, proceed. (One of a/b equals X in this trace → overlap.)
2. Iterate rmi snapshot. Find the `(in3[a,marker,b,4])` entry where `a = X`. Indices of X in args: `[0]`. Look up `allMappingsAna[(1, 2)]` → mappings `[[0], [1]]`.
3. Mapping `[0]`: X → eqList[0] = 9 (assuming `9 < X` lex). Rewritten: `(in3[9, marker, b, 4])`. Non-identity.
4. Mapping `[1]`: X → eqList[1] = X. Identity, skipped.
5. Dedup — one unique rewrite: `(in3[9, marker, b, 4])`.
6. Admission probe: `admissionMapIntegration[(in3[u_9, marker, u_b, u_4]) @ main]` — MATCH (some earlier compound at main had an integration template whose body reformulates "add inside N, output in +" at concrete anchor-slot values; its Case-C insert populated this entry).
7. Match path — emit to `internalMailIn`:
 - Primary: pre = `(in3[a, int_lev_2_42, b, 4])`, post = `(in3[9, int_lev_2_42, b, 4])` (via `substMap = {a→9}` applied to pre).
 - Sibling: pre = `(in[int_lev_2_42, 1])`, post = unchanged (no class member in it).
 - Both pushed to `mb.internalMailIn.statements.insert(tuple(post, lvls, "main"))`.
 - Each gets an origin entry: `origin.first = "equality1"`, `origin.second = [pre, (=[a,9])]`.
8. Entry erased from `rejectedMapIntegration` (rejection resolved). Admission-map entry NOT erased ([I-22](../30_invariants.md#i-22)).

**Cycle N+k+1 — absorb, re-disintegrate, prove.** Top of next hashburst at this LB:

1. `internalMailIn` absorb loop (`prover.cpp+`) iterates `sortedInternal`. For each tuple, calls `addExprToMemoryBlock(stmt, body, -1, status=1, levels, origin, coreId, -1, validityName, false)`. `status=1` ≠ 3, so `disintegrateExpr2` runs (`prover.cpp` gate).
2. For `(in3[9, int_lev_2_42, b, 4])`: disintegrate → atom, goes into `encodedStatements` at main.
3. For `(in[int_lev_2_42, 1])`: atom, goes into `encodedStatements` at main.
4. `updateAdmissionMapIntegration` fires on each deposit (main-only gate at `prover.hpp`) — walks each arg, probes admissionMapIntegration with marker-forms, fires `prepareIntegrationCore2` for any that match. This is where downstream integration rules pick up the new facts and drive toward the compound's proof.
5. Subsequent hash bursts: the compound's enclosing implication (`sequence → sequence`) has its premise chain now satisfied. Head `(sequence[1,4,2,9,11])` is derived. Moves out of `toBeProved` into `proved_theorems.txt`.

**Origin trail as seen by the verifier.** The chapter for the proved theorem has a row for `(in3[9, int_lev_2_42, b, 4])` tagged `equality1`, with `rest = [source="(in3[a,int_lev_2_42,b,4])", ns="main", eq="(=[a,9])", ns="main"]`. Verifier's `check_equality1` at `verifier.py`:
- result_core == source_core (`in3`) ✓
- arity match (4) ✓
- differing args: position 0 — `(a, 9)` ∈ equality set? Yes (the origin provides `(=[a,9])`) ✓ — accepted.

Non-differing positions (`int_lev_2_42`, `b`, `4`) need no equality. Row passes.

**If canonical-only instead of full permutation.** For this trace the class `{X, 9}` has size 2 and only X is in the key, so canonical-only coincides with full permutation — one rewrite, same result. A future rung where two class members land in the same key (e.g. `{X, Y}` with key `(in3[X, marker, Y, 4])`) would have canonical-only produce `(in3[9, marker, 9, 4])` (collapse), while full permutation would also produce `(in3[9, marker, Y, 4])` and `(in3[X, marker, 9, 4])` — three distinct admission-probe chances. The Gauss `{0,1}=[0,1]` rung doesn't hit this, but FTA-ladder downstream almost certainly will.

---

### 3-way admission-map insert fan-out

`prepareIntegrationCore2` Case B at `prover.hpp` calls `addToHashMemory(...)` **three times** — once each with target = `mb.overallHashMemory`, `mb.localHashMemory`, `mb.localHashMemoryDelta`. Each `addToHashMemory` invocation iterates `triggersForAdmissionSetIntegration` and calls `makeAdmissionKeys(..., &mb)`, which — per [D-19](../40_decisions.md#d-19) — includes:

- `admissionMapIntegration[keyString, validity][instructionCopy];` (default-insert of the instructionCopy sub-key into the template map).
- `revisitRejectedIntegration2(bareKey, mb, validityName)` (stripped of u_).
- `admissionSetIntegration.insert(...)` + `revisitRejectedIntegration2(admissionKey, mb, validityName)`.

**Implication:** for one logical "admission template is now registered for compound C" event, `revisitRejectedIntegration2` can fire up to **3× per `makeAdmissionKeys` call** (overall + local + localDelta) **× number of trigger templates** (outer loop in `addToHashMemory`). Each call is idempotent — the second and third calls at the same marker key find `rejectedMapIntegration[(key, validity)]` empty (since the first call already erased it), returns fast. No correctness issue, but a perf hotspot at FTA scale where trigger counts grow.

**Do not** add guards of the form "skip if not overallHashMemory" — the three-way insert is structural (see `prover.hpp` Case B for rationale: local/delta mirrors are needed for static-hotpath request generation). The cheap idempotent revisit is the right design; any "optimization" that conditionally skips the revisit for `local*` targets must first prove that `overallHashMemory` will always have the key available at the revisit moment.

### `rejectedMapIntegration` and ancestor-scope eq classes

`applyEquivalenceClassToRejectedMapIntegration` matches a class against an rmi entry when either:

- `keyEv.validityName == validityName` (class's scope), same-NS case; OR
- `validityName ∈ memoryBlock.nameMap.stringAncestorsOf[keyEv.validityName]`, ancestor-NS case.

This is the same rule the main `applyEquivalenceClass` uses at its ancestor-NS call site (`prover.hpp`). The *class* applies at whichever scope it was registered, the *rewritten entry's validity is the rmi key's original validity*. So a class at `main` validly rewrites a rejected constituent whose compound lived at a deeper scope (e.g. `main_boundary_(implication23[…])_…`) without promoting the rewrite's validity.

**Edge case to watch:** if the rewritten constituent emits to `internalMailIn` with the rmi key's (descendant) validity, the next hashburst's absorb adds it at that validity. But `updateAdmissionMapIntegration` is gated on `validityName == "main"` at `prover.hpp` — so a non-main revival emission does NOT trigger admission-map updates at main, and integration templates that would admit the rewritten form at main never see it. This is a known asymmetry; if FTA rungs need branch-scope revivals to cross-trigger main-scope integrations, the gate needs relaxation (flag as open question, not a 2026-04 fix).

---

### Scaling pressures on `rejectedMapIntegration` beyond `{0,1}=[0,1]`

Current design works for the Gauss `{0,1}=[0,1]` FTA-ladder rung. Three pressures are already visible and expected to bite on subsequent rungs (Euclid, FTA proper):

1. **`rejectedMapIntegration` growth is unbounded.** Pass B keeps emitting fresh entries per rejection; with full-permutation, the no-match-persist path inserts multiple new keys per old entry per class call (up to the number of distinct mappings that produce distinct rewritten keys). Observed ~10^5 entries mid-Gauss; richer fact bases will push this 10–100×. Today's linear walk in `applyEquivalenceClassToRejectedMapIntegration` is O(|rmi|) per class call despite the short-circuits. **Planned mitigation:** reverse index `varToKeys: unordered_map<string, unordered_set<markedKey>>` maintained on insert/erase, so each class call touches only overlap-relevant entries (O(|affected|) instead of O(|rmi|)). Would also subsume the current `varsInRejectedMapIntegrationKeys` cache.

2. **`internalMailIn` is per-LB, unbounded.** Revival emissions accumulate across a hashburst and drain at the top of the next. A runaway revival chain at one LB throttles that LB with no intrinsic cap. Per-hashburst dedup of `(stmt, validity)` emissions would help (partially handled by `std::set` semantics on `statements`, but the equality1-origin trackHistory path still allocates).

3. **`varsInRejectedMapIntegrationKeys` is monotonic.** Grows forever — never shrinks when entries leave rmi via revival. At FTA scale it converges to ~"every arg ever seen" and stops being selective. Subsumed by mitigation (1).

On the current rung these pressures cost ~30% runtime (405s → 525s going canonical-only → full-permutation) but don't regress theorems. On the next rung they may become load-bearing.

### Debug trap patterns for Pass B / admission / revival paths

When diagnosing silent drops or revival misfires, the idiom that worked for the `rejectedMapIntegration` development was a **static `std::ofstream` at the emission site**, gated on a counter:

```cpp
{
    static int __dbgCount = 0;
    static std::ofstream __dbg(".debug/NAME_trace.txt", std::ios::out);
    if (__dbg.is_open()) {
        __dbgCount++;
        if (__dbgCount <= 50 || (__dbgCount % 500) == 0) {
            __dbg << "event #" << __dbgCount
                  << " key=" << markedKey
                  << " v="   << validityName
                  << " LB="  << mb.exprKey
                  << "\n";
            __dbg.flush();
        }
    }
}
```

Properties:

- **Thread-safe enough for single-process `logicalCores=1`** (no mutex — the static is per-thread-init but all access is single-threaded in practice).
- **Self-bounded output** — first 50 events logged verbatim, then every 500th. Catches the start (when the pattern first fires) + periodic samples (to track steady state).
- **Opens once per process** — no file-open-per-emission overhead.
- **Clean removal** — one `git rm` of the block after diagnosis, no header changes, no call-site changes elsewhere.

During this session five such blocks were placed at `updateRejectedMapIntegration` (write count), `emitIntegrationRevivalToInternalMailIn` (emit count), `applyEquivalenceClassToRejectedMapIntegration` (class invocation), `revisitRejectedIntegration2` (find-count with `found=YES/no` tag), and the admissionMapIntegration insert site — then all five removed. Future agents reintroducing them for another silent-drop investigation can `git show 51750a6^:prover.hpp` and copy the exact blocks back.

**Complement this with `git show 51750a6 -- prover.hpp | grep -B2 -A15 "__dbg"`** for the ready-made block template. The counter-gating pattern works for any emission/insert site where you suspect a silent drop or a low-frequency corner case.

### Known silent-drop sites (historical + current)

A "silent drop" is a code path where a statement, admission, or rejection is discarded without a log, counter, or assert — invisible unless you happen to trap at that exact site. Catalog:

- **Pass B `int_` branch — FIXED (now buffers to `rejectedMapIntegration`).** Before [D-19](../40_decisions.md#d-19), the `int_` branch at `prover.cpp` silently dropped any fresh `int_` witness that failed both `isAdmittedIntegration` and the `admissionSetIntegration` fallback. No buffer, no revisit path, no counter. Symptom: theorems requiring deferred admission (after an equality rewrite would admit the int_ witness) never proved. Gauss `sequence / limitSequence / fold` theorems were blocked on this. Fixed by the `rejectedMapIntegration` mechanism.
- **Pass B `it_` branch `else` clause when core is not an operator** (`prover.cpp`). If `extractExpression(removedU)` is not in `operators`, the whole `hasOperator` branch is skipped and no buffering happens. Whether this is a real silent-drop depends on whether non-operator body elements can ever arise for an `it_`-bound stmt — not currently exercised by known theorems, but the path is un-asserted. Tag: suspected fragility, not confirmed.
- **`applyEquivalenceClass` statement-growth cap.** If `countPatternOccurrences > maxNumberSecondaryVariables` the rewritten statement is silently dropped (`prover.hpp`). This is intentional fan-out control, but the drop itself is unlogged — a theorem that depends on the rewrite never proving may look like an unrelated prover issue.
- **`addExprToMemoryBlock` ancestor-scan duplicate suppression (`prover.cpp`).** Redundant deposits are silently ignored — correct behaviour, but a buggy caller that depends on the *second* deposit having different semantics would fail silently.

**Why this matters.** The algebra side (`rejectedMap`) was a fix for a silent-drop pattern years ago; the `rejectedMapIntegration` fix (2026-04) was the equivalent for integration. The pattern repeats: if a new admission category is added later (e.g. contradiction-side, or a new anchor-family), check for its own silent-drop equivalent.

### Not exercised by tests

- **Contradiction-from-multi-hypothesis scope.** Tested implicitly by incubator runs, not by a minimal regression. A refactor of `primedForContradiction` propagation would go unverified.
- **`multiplyImplication` partition correctness for arity > 3.** The Bell numbers grow fast (B₃=5, B₄=15, B₅=52). In practice Peano/Gauss theorems rarely have more than 3 `(1)`-typed bound variables, but a future anchor expansion with more typed slots could expose a partition-generation bug.

---

## Open questions

- **OPEN-3.** The single-input operator gate — theoretical reason or empirical? See [I-6](../30_invariants.md#i-6). Likely requires a concept-level argument about Pass B admission to answer.
- **OPEN-4 — RESOLVED.** `ExpressionAnalyzer::operators` is populated at [`prover.cpp–200`](../../GL_Quick_VS/GL_Quick/src/prover.cpp) immediately after `coreExpressionMap = ce::modifyCoreExpressionMap(anchorID)`. Loop: `for (const auto& kv: coreExpressionMap) { if (!kv.second.inputArgs.empty && !kv.second.outputArgs.empty) operators.insert(kv.first); }`. The invariant *linking it to the conjecturer's notion of "operator"* is structural: both consume the same `coreExpressionMap` from stage 1, and both use the same `inputArgs`/`outputArgs`-non-empty condition. No separate conjecturer-side operator set exists that could drift; the set is canonical.
- **OPEN-6 — RESOLVED.** `performDisintegration` does not exist in the current source tree (grep returns no matches across `GL_Quick_VS/GL_Quick/src/`). the project conventions's naming is stale — it likely refers to a historical name pre-split. The actual disintegration surface today is `ce::disintegrateImplication` in `compiler.hpp` (structural walk, no LB side effects) plus `disintegrateExpr2` in `prover.cpp` (Pass B, with admission gating). the project conventions should be updated to remove the `performDisintegration` reference.
- **OPEN-12 — RESOLVED.** Induction-variable identification happens at [`prover.cpp–5107`](../../GL_Quick_VS/GL_Quick/src/prover.cpp) during auxiliary-implication setup. For each theorem, the prover iterates over `digitArg` values (members of the digit-args set computed by `findDigitArgs`). For each digit-arg, it creates a recursion sub-block (`tempMb2`) and records the choice: `dependencyTable.originalInductionVariableMap[originalIndex] = make_tuple(digitArg, recursionCounter)`. So: **induction is scheduled on every digit-arg as a candidate**; the base case and step case are proved for each; the `digitArg` whose triad succeeds gets promoted with `method = induction` and its name in the reference column of `globalTheoremList`. Multiple digit-args can each yield separate induction theorems from the same chain. With the induction-typing fix ([I-18](../30_invariants.md#i-18)), a third sub-block proves `(in[digitArg, N])` before promotion.

---

## See also

- [`20_core_concepts/01_logic_blocks.md`](../20_core_concepts/01_logic_blocks.md) — the LB grid, `Memory` structure.
- [`20_core_concepts/02_hash_engine.md`](../20_core_concepts/02_hash_engine.md) — hash-based inference detail.
- [`20_core_concepts/03_mail_system.md`](../20_core_concepts/03_mail_system.md) — inter-block comm.
- [`20_core_concepts/04_validity_stack.md`](../20_core_concepts/04_validity_stack.md) — scope names, NameMap.
- [`20_core_concepts/05_equivalence_classes.md`](../20_core_concepts/05_equivalence_classes.md) — equality classes + negated-equality expansion.
- [`20_core_concepts/07_or_branching.md`](../20_core_concepts/07_or_branching.md) — OR handling detail.
- [`30_invariants.md`](../30_invariants.md) — the numbered invariant register.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
