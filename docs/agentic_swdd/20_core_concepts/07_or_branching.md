<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Core concept — OR branching `[DRAFT]`

> Case analysis is the hardest thing GL does, because it requires reasoning to split into scoped sub-proofs, run them independently, and only then recombine. Every FTA-ladder rung depends on some form of OR handling. This chapter describes the OR machinery in operation.

---

## The OR shape

An `or<N>` expression is the compiled name for a disjunction. The expansion follows De Morgan: a two-element OR `E1 ∨ E2` compiles to `!(&!E1!E2)`. Example from `GL_BINARY_MAP` in the generated `index.html`:

```javascript
"or0": {
    "signature": "(or0[x1,x2,x3,x4])",
    "mpl": "!(&!(existence2[x1,x2,x3])!(=[x2,x4]))"
}
```

So `or0[N, n, s, 0]` unpacks as `existence2[N, n, s] ∨ n = 0` — *n has a successor predecessor in N*, OR *n equals 0*. The individual disjuncts of an OR are arbitrary GL expressions (`existence<K>` / equalities / etc.) — there is no requirement that they be implications.

The Python popup builder (`build_gl_binary_map` in [`generate_full_proof_graph.py`](../../generate_full_proof_graph.py)) and the C++ `expandSignature` (`prover.cpp` CASE 4 — OR) both produce the same nested form: two-element OR is `!(&!E1!E2)`; for `n ≥ 3` elements, `!(&current!Ek)` is wrapped around the running expression for each successive `Ek`.

The canonical OR theorem on Peano — `or0[7,2,1,3]` in anchor-argument form — reads *for any n ∈ N: n = 0 ∨ n has a predecessor.* This was the historical branching milestone and the worked example in the present chapter. The current branching milestone is rung 1 of the FTA ladder (`{0,1} = [0,1]`); `or0` remains the natural minimal example for explaining OR disintegration.

---

## Three stages of an OR proof

### 1. OR disintegration

The prover reaches an OR head and decides to split. Two distinct things happen:

**1a. K mutual-exclusion sub-implications at the parent scope.** For each disjunct `d_i`, the prover emits the implication

```
(>[](!d_0) ... (!d_{i-1}) (!d_{i+1}) ... (!d_{N-1}) d_i)
```

— "if every other disjunct is false, then `d_i` must hold". For `K = 2` the chain collapses to a single negated premise, so an OR `A ∨ B` produces the rule pair `!A → B` and `!B → A`. These K rules are inserted into the parent scope's hash memory (no per-branch scope yet) and become available as forward-firing implications.

Tagged `disintegration` in the processed proof graph. Checker: `check_disintegration`'s `or` branch at [`verifier.py`](../../verifier.py) — looks up the OR entry in the GL binary, substitutes the compound's args into the disjunct templates, and verifies that `line.expression` peels into a head matching one disjunct and a premise multiset matching the negations of the other disjuncts. The expansion-row chain (expanded De-Morgan form ↔ compact `(or<N>[…])` name) provides the dispatch path.

The `disintegration` origin is recorded inside [`disintegrateExprCore2`](../../GL_Quick_VS/GL_Quick/src/prover.cpp)'s OR case, at the parent `validityName`, with origin = the expanded De-Morgan form `!(&!d_0…!d_{K-1})` (the same key the matching expansion row records). See [D-55](../40_decisions.md#d-55).

**1b. Per-branch case-split.** For each disjunct `d_i`, open a scope via `NameMap::encodePush(current, "ordis_(or<N>[…])_(d_i)")` and seed the branch with:

- The disjunct `d_i` asserted (as if it held).
- The **negations** of the other disjuncts `!d_j` (`j ≠ i`) asserted as branch-local assumptions.

Each branch now has a different starting context. Each runs the standard prover machinery in its own scope.

Tagged `or disintegration` in the processed proof graph. Checker: `check_or_disintegration` at [`verifier.py`](../../verifier.py).

**1a is unconditional** (subject only to `currentOrDepth < parameters.max_or_depth`). **1b is gated on the OR-admission check** (D-32, see below). When admission fails, the K mutual-exclusion implications still land in hash memory but the per-branch case-split is skipped — auditability for the K rules is preserved either way because the `disintegration` origin record is hoisted above the admission gate.

**Admission gate (D-32, 2026-05-01).** OR-disintegration is admitted in [`disintegrateExprCore2`](../../GL_Quick_VS/GL_Quick/src/prover.cpp)'s OR case only when one of two conditions holds:

1. **Sharper bypass.** `allowOrDisintegration == true` is threaded into `disintegrateExprCore2`. The flag is set by [`checkLocalEncodedMemoryStatic`](../../GL_Quick_VS/GL_Quick/src/prover.cpp) at the head-firing call site, fed from `LocalMemoryValue::productOfDisintegration` (declared in [`memory.hpp`](../../GL_Quick_VS/GL_Quick/src/memory.hpp) line ~84). The flag is stamped at install time inside [`addToHashMemory`](../../GL_Quick_VS/GL_Quick/src/prover.cpp) when at least one premise (chain element) of the implication has an arg starting with `"u_"` — the reserved prefix for bound-variable placeholders introduced by `prefixArgumentsWithU`. Anchor-bound theorems (whose chains carry only concrete integer args) never trigger the bypass; body-element implications produced by disintegrating a compound (e.g. ES2-forward `implication20`) do.
2. **Legacy `orAdmissionSet` gate — REMOVED** (`D-135`). The container had no `.insert` site anywhere ([D-31](../40_decisions.md#d-31)), so the always-empty set made the fallback loop constant-false; the gate is now just the `allowOrDisintegration` flag, behavior identical. The dump prints the literal empty `orAdmissionSet` section header for Rule-14 byte stability.

OR-disintegration is also coupled with general disintegration: if `addExprToMemoryBlock`'s `doNotDisintegrate == true`, `allowOrDisintegration` is forced `false` at function entry. OR-disint cannot fire when general disint is forbidden.

History. Pre-D-31, the gate was structurally dead — zero `_boundary_ordis_` scopes ever minted. D-31 introduced a broad bypass keyed on the deposit scope's last namespace segment matching `(implicationNN[…])`; this caused a runtime explosion (anchor-bound rules' broad fan-out into implication scopes spawned case-split branches everywhere). D-32 narrowed to the sharper "implication itself is a product of disintegration" criterion, preserving the FTA-rung-1 Step 9 lower (impl24) closure path while cutting the impl24 `_ordis_` row count by 88%.

### 2. Per-branch derivation

Each branch LB (child of the current LB) runs independently. It may derive more facts, open further hypothesis or OR scopes, or — if the branch's seeds lead to contradiction — discharge via the contradiction path.

### 3. OR convergence

OR convergence has two distinct mechanical paths, depending on which scope kind the convergence is happening in. Both end with the conclusion landing at the OR-branches' shared parent scope, but the trigger and the cleanup shape differ.

Tagged `or convergence` in the processed proof graph. Checker: `check_or_convergence` at [`verifier.py`](../../verifier.py).

#### 3a. `_orint_` (OR-integration, single-branch-proves)

Used when an OR-introduction goal sits in `toBeProved` and one branch proves the goal's head. The head match closes that branch; the wrapping OR expression is then emitted at the OR-branches' shared parent scope. The remaining sibling branches are wiped wholesale by [`cleanUpOrIntegrationBranches`](../../GL_Quick_VS/GL_Quick/src/prover.cpp) — see [Sibling-branch wipe on convergence](#sibling-branch-wipe-on-convergence) below for the rationale.

**Parent-scope emission detail (D-30, 2026-04-29).** When a single branch's head proves (the `or branch proven` path at [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp)), the wrapping OR expression is added at the OR-branches' shared parent scope — the immediate ancestor of the `_boundary_orint_…_((…))` payload — *not* unconditionally at `main`. The destination is computed by peeling the last `_boundary_<payload>` segment off the proving branch's `validityName`:

```cpp
size_t lastBoundary = validityName.rfind(NameMap::BOUNDARY_STR, std::string::npos,
                                          NameMap::BOUNDARY_LEN);
const std::string orEmitScope =
    (lastBoundary == std::string::npos)
        ? std::string("main")
        : validityName.substr(0, lastBoundary);
```

For top-level ORs whose parent IS `main` (Peano OR-branching), `orEmitScope == "main"` — semantics unchanged. For ORs nested under a hypothetical / integration scope (the FTA-ladder pattern, rung 1 onward), the OR lands at the wrapping hypothetical scope and the `(in[p, M])` derivation that follows via `implication21` matches the goal at *that* scope rather than skipping to main. Pre-D-30 the destination was hard-coded `"main"`, which silently produced scope-mismatched derivations whenever ORs were nested. See [D-30](../40_decisions.md#d-30).

**Goal-flow row shape (D-52, 2026-05-08).** When `prepareIntegrationCore2` Case OR opens the K branches for an `or<N>`, the `_orint_` mental model is "rewrite `(or<N>)_integration_goal` at the parent scope into K sub-implications `(!D_others → D_k)`, one per branch — proving any single one closes the OR". Per branch `k`, the producer emits a parent-scope row:

```text
(>[]<AND-of-negated-others>D_k)_integration_goal  parent_scope
    expansion for integration
    (or<N>[…])_integration_goal  parent_scope
```

Both KEY and ORIGIN sit at the OR's parent scope (the immediate ancestor of `_boundary_orint_<or>_(<chosen>)`); the deeper branch namespace continues to host the `or branch assumption` rows (negated other-disjuncts as `!D_j` premises) and the head-as-toBeProved (chosen disjunct `D_k` as goal). For `K = 2` the AND wrapper collapses to a single negation `!D_other`. For `K >= 3` the AND is left-nested in the same shape `_build_and_from_elements` produces. Empty bound-var list — `_orint_` branches don't introduce new quantifiers.

Pre-D-52 the producer emitted KEY = bare disjunct `D_k` at the deeper branch namespace with ORIGIN = `expansion for integration ← (or<N>)_integration_goal` at parent. That row was structurally absurd (a single disjunct is not the structural expansion of `or<N>`) and namespace-mismatched (line.namespace deeper than right_ns). Verifier rightly rejected it. D-52 also extends `verifier.py::_try_expand` for category `or` to accept any of the K sub-implication forms in addition to the existing De Morgan `!(&!E1!E2…!Ek)` form. See [D-52](../40_decisions.md#d-52).

#### 3b. `_ordis_` (OR-disintegration, all-branches-converge)

Used when an OR-elimination case-split has produced the same conclusion `C` independently in every branch. Convergence detection is performed by [`ordisMerge`](../../GL_Quick_VS/GL_Quick/src/prover.hpp) — an inline member function called from **two** sites: (1) [`addExprToMemoryBlock`](../../GL_Quick_VS/GL_Quick/src/prover.cpp)'s post-`addStatement` loop (once per `(addExpression, effectiveValidity)` pair returned by `addStatement`), sibling to the `toBeProved` discharge logic and to the `_orint_/NotOrScope` block; and (2) [`applyEquiClasses`](../../GL_Quick_VS/GL_Quick/src/prover.hpp)'s `mergeProducts` lambda, run once per equivalence-class **substitution product** in both passes — so a branch disjunct produced by an equi-class rewrite (not by `addStatement`) also registers; see [D-94](../40_decisions.md#d-94). Both call sites observe deposits at their *actual* scope — including [D-33](../40_decisions.md#d-33)'s descendant-direction cross-scope rewrites. (The legacy `trackOrBookkeeping` predecessor lived inside `addStatement` and observed only the caller's scope, missing D-33's cross-scope deposits; it has been deleted. The earlier `addExprToMemoryBlockKernel` that hosted the post-`addStatement` loop was collapsed into `addExprToMemoryBlock` during the `equi_reshuffle` refactor.)

`ordisMerge` records each per-branch deposit in `Memory::orBookkeeping` — keyed by packed `(exprId, sigId)` `lbStateInterner` pairs (`packLbStateKey`), valued as decoded-lex ordered branch-disjunct id sets created only via `orDisjunctsAt` (`D-135`; the decoded order feeds the origin rows and the cleanup loop). When the recorded disjunct count for an `(expr, orSignature)` pair reaches `Memory::orDisjunctCount[sigId]` (registered at OR-disintegration mint time, see [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp)), the convergence path fires:

1. **Promotion via `sameIterationInternalMail`.** `ordisMerge` pushes `(addExpression, addExpressionLevels, parentValidity)` onto `Memory::sameIterationInternalMail.statements` — the same revival channel `revisitRejectedIntegration2` uses, see [`memory.hpp` `Mail` (used as `Memory::sameIterationInternalMail`)](../../GL_Quick_VS/GL_Quick/src/memory.hpp). The drain at the top of the next hashburst body absorbs the tuple via `addExprToMemoryBlock(..., status=1,..., parentValidity,...)`, running the parent-scope deposit through the full kernel pipeline (`toBeProved` discharge of the parent's forward-conjunct goals included). Deferring to `sameIterationInternalMail` instead of calling `addExprToMemoryBlock` directly avoids reentrant kernel invocation in the post-`addStatement` loop.
2. **Per-branch cleanup of the converged expression.** For each branch validity in the recorded disjunct set, `ordisMerge` calls `removeExpressionFromMemoryBlock(EncodedExpression(addExpression, branchValidity), mb, /*state=*/0)` — the standard removal procedure. The N per-branch copies of the converged expression collapse into the single parent-scope copy. The parent-scope copy remains visible to each branch via comparable-scope inheritance, so branches can reuse it. Branches keep all their *other* facts and stay live (unlike the `_orint_` case-split, which wipes the entire branch — see §3a). `exprOriginMap` (history) is not touched; `intStatementLevelsMap` and `intKnownStatements` are also retained, so a re-deposit attempt at a wiped branch scope short-circuits at the standard `intKnownStatements` gate.

History: pre-[D-34](../40_decisions.md#d-34) the bookkeeping path was `trackOrBookkeeping` inside `addStatement`. It was structurally blind to D-33's cross-scope deposits and carried a too-aggressive cleanup (wiped every statement at the branch scopes via prefix-match, plus added the branch validities to `validityNamesToFilter` to block all future inserts). FTA-rung-1 §9a / §9b symptoms: the per-branch preorder facts produced by D-33's descendant-direction rewrite landed at the branch scopes (`current_proof_state.md` confirms all four cells), but `trackOrBookkeeping` never saw them and convergence never fired. D-34 fixes both — the kernel-level placement closes the visibility gap, and the cleanup is narrowed to just the converged expression so branches stay alive for further convergences (e.g. the second preorder of the same impl scope's body).

---

## `classifyOrScope` — the scope-role oracle

Defined at [`prover.hpp`](../../GL_Quick_VS/GL_Quick/src/prover.hpp). Returns one of:

- `NotOrScope` — scope is not related to an OR split.
- `Integration` — scope is part of an OR re-integration (reformulation-for-integration over an OR head).
- `Disintegration` — scope is one of the per-disjunct branches.

Classification is based on the payload prefix of the scope's name. See [`20_core_concepts/04_validity_stack.md`](04_validity_stack.md). The function is consulted by `performElementaryLogicalStep` (to decide OR-aware bookkeeping), `ordisMerge` (to detect `_ordis_` convergence and route the promoted expression through `sameIterationInternalMail`), the kernel's `_orint_/NotOrScope` block in `addExprToMemoryBlockKernel` (to decide OR-integration emission and bare-implication republish), and `cleanUpOrIntegrationBranches` (to find sibling integration branches to wipe after a winner emerges).

---

## The `or theorem` tag

A theorem whose *head* is an `or<N>` expression is tagged `or theorem` in the processed proof graph (replacing the retired `or branch proven` tag). Checker: `check_or_theorem` at [`verifier.py`](../../verifier.py).

Emitted when a goal is of OR shape and has been reached via some combination of branches. The checker validates that the head is indeed an `or<N>` node and that the proof graph properly supports it.

---

## OR theorem construction (run_modes.cpp::fullRun)

OR theorems with head `(or<N>[…])` are not proved directly. They are **constructed** post-prove from pairs of mutually-exclusive parent implications:

```text
parent A:  (>[…premises…]!(D_a)(D_b))    // existence form: from !D_a follows D_b
parent B:  (>[…premises…](!(D_b))(D_a))   // companion form: from !D_b follows D_a
                ⇓
constructed:  (>[…premises…](or<N>[args]))   // disjunct set {D_a, D_b}
```

The construction has three pieces, all in the same emit pass:

### Producer — `headSwitchOne` populates `orPairsFromHeadSwitch`

After the single prove pass, [`analyzeExpressions`](../../GL_Quick_VS/GL_Quick/src/prover.cpp) walks `globalTheoremList` and runs `headSwitchOne` on each theorem. For any theorem with a negated, non-quantified premise, the contrapositive is computed: negate the head, replace the negated premise with the negated head. The (original, contrapositive) pair is stored in `orPairsFromHeadSwitch` (class member of `ExpressionAnalyzer`).

The walk is symmetric — it visits both directions of every mirror pair. So `orPairsFromHeadSwitch` typically contains *both* `(mirror_a, mirror_b)` and `(mirror_b, mirror_a)` for the same disjunct set.

### Constructor — `constructOrTheorem` ([`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp))

Given `(existenceThm, companionThm)`, the constructor:
1. Disintegrates both into chains.
2. Locates the negated non-quantified premise `!D_a` in the existence theorem's chain.
3. Uses the existence theorem's head (which is `!(>[…]!body)` — i.e. the existence form) to recover the second disjunct `D_b`.
4. Builds a fresh `or<N>` LogicalEntity, mints a new compact name (`orCounter` increments), and registers it in `compiledExpressions` + `coreExpressionMap`.
5. Reassembles the OR theorem with the shared premises in front and the compiled `(or<N>[args])` head.

### Caller — `run_modes.cpp::fullRun` post-compression

[`run_modes.cpp::fullRun`](../../GL_Quick_VS/GL_Quick/src/run_modes.cpp) drives the construction loop *after* the compressor runs and *before* `saveProvedTheoremsFiltered`:

1. **Membership filter.** Only construct an OR if both `existenceThm` and `companionThm` survived compression (`provedSet` derived from `globalTheoremList`).
2. **Disjunct-set dedup (post `D-55`).** Canonicalize each `(exist, comp)` pair as `(min, max)` lexicographically. Skip pairs whose canonical form has already produced an OR. Without this, `orPairsFromHeadSwitch`'s symmetric population produces both `or0` (disjuncts in one order) and `or1` (disjuncts swapped) for the same logical OR — pre-fix this is the source of the `or0` / `or1` duplication on Peano.
3. **Construction.** Call `constructOrTheorem(exist, comp)`. Append the result to `globalTheoremList` and `fullTheoremList` with method `or theorem` and references `(exist, comp)`.
4. **Parent-removal (post `D-55`).** Mark `exist` and `comp` as **consumed parents**. After all ORs are constructed, drop consumed parents from `survivingTheorems` before `saveProvedTheoremsFiltered` writes `theorems.txt` / `compiled_theorems.txt`. Append consumed parents to `compressed_out_theorems.txt` — same procedure the compressor applies to redundant theorems, done manually here at OR-creation time so the cleanup lands in the same emit pass.
5. **Append OR theorems** to both proved-theorem files.

The parents are subsumed by the OR via OR-disintegration's K mutual-exclusion implications (§1a above). They survive only as compressed-out artefacts — recoverable from the OR + the [`disintegrateExprCore2`](../../GL_Quick_VS/GL_Quick/src/prover.cpp) machinery, no longer cited as standalone theorems. Pre-fix the parents stayed in `theorems.txt` and downstream chapters (notably FTA-rung-1's `1209_direct_proof.txt`) cited them directly, shadowing the OR theorem they had become equivalent to.

---

## Conjecturer-side OR generation

OR conjectures are not enumerated by the main conjecturer loop. Instead, `generateOrConjectures` at [`conjecturer.cpp`](../../GL_Quick_VS/GL_Quick/src/conjecturer.cpp) reads `files/theorems/or_pairs.txt` — a hand-curated pair-per-line file — and emits OR-shaped conjectures into `conjectures.txt`.

Rationale: OR conjectures are expensive to prove (case-split machinery) and most randomly-generated OR shapes are uninteresting. Hand-curation lets the maintainer target OR enumeration at known-valuable rungs of the FTA ladder.

---

## The FTA ladder

Per `MEMORY.md project_fta_ladder.md`, `docs/fta_ladder/README.md` is the master index of FTA-ladder rungs and `docs/fta_ladder/rung<N>/current_proof_state.md` is the standing entry point for each rung's in-flight state. Each rung climbs another OR-step toward the Fundamental Theorem of Arithmetic. Rung 1 (`{0,1} = [0,1]`) had its §4.1 forward direction close under [D-34](../40_decisions.md#d-34); §4.2 reverse direction is the next rung-1 frontier.

The OR branching machinery carries almost the entire FTA-ladder engineering load. A bug introduced in OR handling is, in effect, a bug in the FTA campaign.

---

## Sibling-branch wipe on convergence

This applies **only to the `_orint_` (OR-integration) path**. For `_ordis_` see [§3b](#3b-_ordis_-or-disintegration-all-branches-converge) — only the converged expression is collapsed across branches; branches stay live for further convergences.

In the `_orint_` case, once a single branch's head proves and the wrapping OR has been emitted at the parent scope, the sibling branches' local statements are wiped. `cleanUpOrIntegrationBranches` does this: it walks the sibling scopes and removes their entries from `intEncodedStatements`, etc., and adds the branch validity ids to `intValidityNamesToFilter` to block future inserts.

Rationale (`_orint_` only): the sibling branches were case-split *to* prove the head; once one branch succeeds, the others are unreachable case structures whose facts were conditional on assumptions the parent now disproves. Keeping them around risks contaminating later reasoning. The `_ordis_` case is structurally different — the case-split is *over* a known disjunction, and every branch contributes facts that are valid in the parent scope under their respective case conditions; wiping siblings would discard usable derivations.

---

## OR slowdown — the Gauss regression

Per `MEMORY.md project_or_slowdown.md`: introducing OR theorems caused a Gauss-batch explosion via branching validity scopes. The symptom is severe wall-clock regression on Gauss when OR handling is in scope.

*Status flagged in memory: verify before citing — the note predates the validity-stack migration, and some of the underlying causes may have been addressed in the migration.* 

The root cause (at time of the memory note) was that OR-branch scopes opened via `encodePush` produced a scope tree deep enough to slow down `comparable` / `deeperOf` operations. The validity-stack migration tightened these operations via `pairMap`-backed ancestor queries; whether the regression persists is a measurement-worthy question.

---

## Variable copy — the `_copy` pattern in OR

When an OR branch duplicates a hypothesis variable to keep it distinct from surrounding scope, the duplication is tagged `variable copy` in the proof graph. This replaces the retired `reaction to hypo` and `necessity for equality (hypo)` tags.

Checker: `check_variable_copy` at [`verifier.py`](../../verifier.py). The checker walks the `_copy` substitution chain (which echoes the Priority-4 renaming rule in stage 6 — see [`10_pipeline/06_process_proof_graph.md`](../10_pipeline/06_process_proof_graph.md)) and verifies each step is consistent.

---

## Weaknesses

### Known & tracked

- **`or_slows_down` regression.** Memory flagged; status ambiguous post-migration. A measurement-worthy open question.
- **OR conjecture curation is manual.** `or_pairs.txt` is hand-maintained. A future FTA rung that needs a new OR shape requires editing this file. No automation.
- **`or branch proven` and `or branch assumption` are first-class verifier tags (D-35, 2026-05-03).** Previously claimed retired/overridden — incorrectly: no override path existed and the prover always emitted the tags live. FTA-rung-1 chapter `1209_direct_proof.txt` carries both. The verifier now registers `check_or_branch_proven` / `check_or_branch_assumption` in `TAG_CHECKERS`, validating: parent/branch scope ancestry, GL-binary disjunct membership (modulo equality symmetry), and the branch-payload `_boundary_orint_<or>_(<disjunct>)` substring pattern. See [`08_proof_tags.md`](08_proof_tags.md#or-branch-proven) for the full row layouts.

### Suspected fragility

- **`or convergence` spec'd row layout — gap closed by D-36.** `ordisMerge` (`prover.hpp`) now emits the spec'd layout `<C> <parent> or convergence <OR> <parent> <C> <branch_D1> <C> <branch_D2> …`. The branch derivations of `C` remain in `exprOriginMap` (untouched by `removeExpressionFromMemoryBlock(state=0)`); `buildStack` renders them automatically by recursion. Side-effect of the fix: 4 previously-hidden `_ordis_` per-branch case-split rows surface in chapter export under `or disintegration` tag — they pass the D-36-extended `check_or_disintegration`. Net incubator-verifier state: 0 failures.

- **Sibling-wipe atomicity.** `cleanUpOrIntegrationBranches` walks and wipes. If an LB were processed in the same cycle its wipe touches, the effect on that LB's iteration state would be ill-defined — but cross-LB writes during the parallel phase are forbidden ([I-28](../30_invariants.md#i-28)), so a worker thread never wipes an LB another worker is processing; the wipe runs in the single-threaded post-join drain.
- **Branch-seed distinctness.** Each branch is seeded with the other disjuncts' negations. If two disjuncts are textually identical (malformed OR conjecture), two branches seed identically — potentially derived-to-completion in parallel with duplicate work. Not currently observed but not excluded.
- **Depth-cost of `comparable`.** OR branching creates scope-tree depth. Post-migration `comparable` uses `pairMap` (O(log N) typical), but with deeply-nested OR chains, log N grows.
- **Cross-stack same-orSignature race in `ordisMerge` bookkeeping.** `Memory::orBookkeeping` is keyed by `(expr, orSignature)` — the parent scope is NOT part of the key. When the same `or<N>` signature is minted at two different stack positions (e.g. `(or2[2,repl_lev_1_0,6])` exists both at `main_boundary_(impl24[…])_boundary_ordis_…` AND at top-level `main_boundary_ordis_…`), ordisMerge's bookkeeping conflates the two contexts. Per-branch deposits from either context insert into the same bucket, the count hits `orDisjunctCount[orSignature]` after seeing N total disjunct payloads regardless of which OR context they belong to, and convergence fires at whatever the triggering deposit's parent scope is. Currently benign because Site F's ancestor-scan dedupe at [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp) (see [I-27](../30_invariants.md#i-27)) absorbs the redundant promotion when the fact is already live at a strict ancestor — exactly the impl24-down case observed in the FTA-rung-1 §4.1 closure trace ([D-34](../40_decisions.md#d-34) verification): the top-level OR-convergence lands the fact at `v=main` first, then the impl24-internal convergence's `sameIterationInternalMail` deposit at `v=impl24` is silently dedupe'd by Site F. The fact is provable at every descendant via comparable-scope inheritance, so the §4.1 forward chain closes regardless. The cleaner fix is to key `orBookkeeping` by `(expr, parentValidity, orSignature)` so each OR context tracks independently, but this is not load-bearing — the current behaviour is correct, just slightly redundant in firing patterns. (The id-form re-key kept the two-component key shape deliberately — same conflation, same Site-F absorption.)

### Not exercised by tests

- **OR-re-integration path.** The `_orint_` payload scope kind exists but is sparsely exercised by the current Peano+Gauss corpus. FTA-ladder rungs may exercise it more; a regression in OR-re-integration could land silently before the ladder climbs.
- **`classifyOrScope` misclassification.** If a payload prefix silently drifts, classification returns the wrong enum. No test verifies the mapping.

---

## Open questions

- **OPEN-1 — PARTIAL.** All-disjuncts-refuted case traced as follows: OR disintegration seeds each branch with (disjunct_i asserted + other disjuncts' negations asserted). If a branch's seeds are internally contradictory, its contradiction LB discharges, proving `!disjunct_i`. If every branch refutes this way, the parent scope has `!disjunct_1 ∧... ∧ !disjunct_N` — i.e. the negation of the OR. `cleanUpOrIntegrationBranches` at [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp) is specifically a *post-convergence* cleanup (wipes losing-branches' statements after a winning branch's conclusion promotes); it is *not* invoked in the all-refuted case. The all-refuted case therefore leaves the OR *unproved* (no convergence fired) while each branch's negation is in the parent scope — useful consequence, but not a vacuous-truth emission. Open: whether any code path specifically detects "all disjuncts refuted → OR is vacuously false" and emits a `vacuous truth` row. Not observed in the Peano/Gauss corpus; may be handled generically by the parent scope's own proof search finding the negations of disjuncts and then deducing `!or<N>`.

---

## See also

- [`20_core_concepts/04_validity_stack.md`](04_validity_stack.md) — scope mechanics.
- [`10_pipeline/04_prover.md`](../10_pipeline/04_prover.md) — `cleanUpOrIntegrationBranches`, `classifyOrScope`, contradiction discharge.
- [`20_core_concepts/08_proof_tags.md`](08_proof_tags.md) — `or disintegration`, `or convergence`, `or theorem`, `variable copy`.
-,,.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
