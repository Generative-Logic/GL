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

The canonical OR theorem on Peano — `or0[7,2,1,3]` in anchor-argument form — reads *for any n ∈ N: n = 0 ∨ n has a predecessor.* This was the historical branching milestone and the worked example in the present chapter. Rung 1 of the FTA ladder (`{0,1} = [0,1]`) established the minimal two-way case split. Rung 2 (`{0,1,2} = [0,2]`) exposed and closed the next boundary: a three-way membership compiled as a nested binary OR is now one flat three-alternative disintegration cohort.

---

## Three stages of an OR proof

### 1. OR disintegration

The prover reaches an OR head and decides to split. Two distinct things happen:

**1a. K mutual-exclusion sub-implications at the parent scope.** Before producing anything, `flattenOrLeaves` recursively expands every contiguous compiled-OR child in the prepared instruction graph. It preserves element order and stops at each non-OR child. For each resulting leaf `d_i`, the prover emits the implication

```
(>[](!d_0) ... (!d_{i-1}) (!d_{i+1}) ... (!d_{N-1}) d_i)
```

— "if every other disjunct is false, then `d_i` must hold". For `K = 2` the chain collapses to a single negated premise, so an OR `A ∨ B` produces the rule pair `!A → B` and `!B → A`. These K rules are inserted into the parent scope's hash memory (no per-branch scope yet) and become available as forward-firing implications.

Tagged `disintegration` in the processed proof graph. Checker: `check_disintegration`'s `or` branch at [`verifier.py`](../../verifier.py) — recursively reconstructs the same substituted non-OR leaf list, then verifies that `line.expression` peels into a head matching one leaf and a premise multiset matching the negations of all other leaves. The expansion-row chain (flat expanded De-Morgan form ↔ original outer compact `(or<N>[…])` name) provides the dispatch path.

The `disintegration` origin is recorded inside [`disintegrateExprCore2`](../../GL_Quick_VS/GL_Quick/src/prover.cpp)'s OR case, at the parent `validityName`, with origin = the expanded De-Morgan form `!(&!d_0…!d_{K-1})` (the same key the matching expansion row records). See [D-55](../40_decisions.md#d-55).

**1b. Per-branch case-split.** For each flattened non-OR leaf `d_i`, open a scope via `NameMap::encodePush(current, "ordis_(outer-or<N>[…])_(d_i)")` and seed that scope with `d_i` asserted. The original outer signature names every sibling branch and the cohort. No intermediate OR-valued branch exists and no later hashburst is needed to expose another syntax level. The negations of the other leaves are not seeded in `_ordis_`; the K mutual-exclusion rules from 1a remain available from the parent scope. Each branch therefore has a different asserted leaf and runs the standard prover machinery in its own scope ([D-218](../40_decisions.md#d-218), [I-166](../30_invariants.md#i-166)).

Tagged `or disintegration` in the processed proof graph. Checker: `check_or_disintegration` at [`verifier.py`](../../verifier.py).

**1a is unconditional at every OR depth** ([D-211](../40_decisions.md#d-211)): the expansion record and the K mutual-exclusion implications are flat hash rules that create no scopes, so they are emitted even when the OR statement lives inside an existing `_ordis_`/`_orint_` branch scope — that is the disjunctive-syllogism consumption of an OR (e.g. deriving `∃pred(q)` from `q≠0` inside an `_orint_` branch, the rung-2 second predecessor split). **1b is gated twice: on `currentOrDepth < parameters.max_or_depth`** — nested `_ordis_` scopes are the scope explosion the cap exists to prevent; one admitted outer OR consumes one depth regardless of how many contiguous nested OR syntax nodes it contains — **and on the OR-admission check** (D-32, see below). When either gate refuses, the K mutual-exclusion implications still land in hash memory but the per-branch case-split is skipped — auditability for the K rules is preserved either way because the `disintegration` origin record is emitted together with them.

**1c. Per-leaf INTRO implications for an or-shaped implication premise** ([D-237](../40_decisions.md)). The K rules are the elimination half of an OR's flat consumption; the introduction half emits in `disintegrateExprCore2`'s IMPLICATION branch: when an installed compiled implication's premise element is a disjunction (compiled-map probe — implication elements have no instruction-graph entities), the prover also emits, per flattened leaf,

```
(>[bound] d_k (or<N>[…]))
```

— "one true disjunct states the disjunction". The leaves come from a compiled-map twin of `flattenOrLeaves` (per-level signature→instance substitution, recursing into or-category elements only), and leaf + head are fed to the reconstruction in RAW element form so `u_` arguments stay free and only the changeable element variable binds. Propositionally valid unconditionally — no soundness surface. This is what lets a single concretely-true disjunct (typically reached through an equality1 witness alias, since the true disjunct at a concrete element is reflexive) fire the implication's own rule: the rung-2.1 chain `2_copy=2 → or(2_copy) → 2_copy∈M → 2∈M`. Each intro rule carries an `expansion` origin whose dependency is the compact implication statement itself — the same chapter-capable antecedent the main rule's expansion record cites; `_try_expand`'s or-aware implication acceptance covers the shape on the verifier side.

An intro-FIRED or head never re-disintegrates ([D-241](../40_decisions.md)): the intro shape is re-detected at every install by `isOrIntroInstall` (`memory.cpp` — single premise byte-equal to one flattened leaf of an or-category head; shape-based so status-3 mail-recovered reinstalls re-mark), the derived per-rule `LocalMemoryValue::disintegrationAllowed` goes false, and the firing site reads it as the rule-intrinsic half of `FiringRecord::doNotDisintegrate`. The stated disjunction stays a flat fact for the membership-intro clause and the K rules — an intro-derived or's source leaf is true at the same scope, so a case split over it is redundant by construction (its `_ordis_` cohorts were a pure fan-out multiplier: branch assumptions breed equalities ex falso, equalities re-fire the intro rules). Elimination-fired or heads (premise not a leaf) and status-0 assumed or fuel branch exactly as before.

**Admission gate (D-32, 2026-05-01).** OR-disintegration is admitted in [`disintegrateExprCore2`](../../GL_Quick_VS/GL_Quick/src/prover.cpp)'s OR case only when one of two conditions holds:

1. **Sharper bypass.** `allowOrDisintegration == true` is threaded into `disintegrateExprCore2`. The flag is set by [`checkLocalEncodedMemoryStatic`](../../GL_Quick_VS/GL_Quick/src/prover.cpp) at the head-firing call site, fed from `LocalMemoryValue::productOfDisintegration` (declared in [`memory.hpp`](../../GL_Quick_VS/GL_Quick/src/memory.hpp)). The flag is stamped at install time inside [`addToHashMemory`](../../GL_Quick_VS/GL_Quick/src/prover.cpp) when at least one premise (chain element) of the implication has an arg starting with `"u_"` — the reserved prefix for bound-variable placeholders introduced by `prefixArgumentsWithU`. Anchor-bound theorems (whose chains carry only concrete integer args) never trigger the bypass; body-element implications produced by disintegrating a compound (e.g. ES2-forward `implication20`) do.
2. **Legacy `orAdmissionSet` gate — REMOVED** (`D-135`). The container had no `.insert` site anywhere ([D-31](../40_decisions.md#d-31)), so the always-empty set made the fallback loop constant-false; the gate is now just the `allowOrDisintegration` flag, behavior identical. The dump prints the literal empty `orAdmissionSet` section header for Rule-14 byte stability.

OR-disintegration is also coupled with general disintegration: if `addExprToMemoryBlock`'s `doNotDisintegrate == true`, `allowOrDisintegration` is forced `false` at function entry. OR-disint cannot fire when general disint is forbidden.

History. Pre-D-31, the gate was structurally dead — zero `_boundary_ordis_` scopes ever minted. D-31 introduced a broad bypass keyed on the deposit scope's last namespace segment matching `(implicationNN[…])`; this caused a runtime explosion (anchor-bound rules' broad fan-out into implication scopes spawned case-split branches everywhere). D-32 narrowed to the sharper "implication itself is a product of disintegration" criterion, preserving the FTA-rung-1 Step 9 lower (impl24) closure path while cutting the impl24 `_ordis_` row count by 88%.

### 2. Per-branch derivation

Each branch scope runs independently inside the current LB. It may derive more facts or open further hypothesis or OR scopes. A pair that is contradictory only because of the branch assertion does **not** discharge the LB: full contradiction requires both expressions at `main` ([D-222](../40_decisions.md#d-222)). The branch itself does not survive such a refutation, though: a branch whose asserted disjunct has its negation known at the branch scope or an ancestor is a dead branch — everything it derives is ex falso — and is retired at the end of the burst; see [Dead-branch retirement](#dead-branch-retirement) ([D-242](../40_decisions.md#d-242)).

### 3. OR convergence

OR convergence has two distinct mechanical paths, depending on which scope kind the convergence is happening in. Both end with the conclusion landing at the OR-branches' shared parent scope, but the trigger and the cleanup shape differ.

Tagged `or convergence` in the processed proof graph. Checker: `check_or_convergence` at [`verifier.py`](../../verifier.py).

#### 3a. `_orint_` (OR-integration, single-branch-proves)

Used when an OR-introduction goal sits in `toBeProved` and one branch proves the goal's head. The head match closes that branch; the wrapping OR expression is then emitted at the OR-branches' shared parent scope. The remaining siblings in that exact `(parent validity, outer OR signature)` cohort are wiped wholesale by [`cleanUpOrIntegrationBranches`](../../GL_Quick_VS/GL_Quick/src/prover.cpp); an equal signature under another parent remains live. The selected branch roots are queued immediately and their descendants are covered by the later `wipeSubtree` drain — see [Sibling-branch wipe on convergence](#sibling-branch-wipe-on-convergence) below for the rationale.

**Ancestor-visible closure (D-221).** A branch goal need not receive a redundant statement at its exact `_orint_` scope. After exact delta reactions, `dischargeToBeProved` snapshots unresolved goals in decoded lexical order and scans each goal's ancestry from `main` toward the goal. The first known matching statement is the strongest proof source: its scope and levels appear in the `or branch proven` dependency, while the exact goal scope still selects the parent emission and `(parent validity, outer signature)` cleanup cohort. Facts in sibling scopes are not visible. A goal beneath a scope already queued for removal is skipped, and no inherited source is inserted into the child ([I-164](../30_invariants.md#i-164)).

**Flat reverse cohort (D-220).** `prepareIntegrationCore2` prepares only maximal contiguous OR roots. It calls the same `flattenOrLeaves` traversal used by `_ordis_`, preserves the ordered non-OR leaves, and runs one renaming pass over the whole cohort. A nested OR entity consumed by that root is skipped by the entity loop, so no opaque intermediate branch family is prepared. Every `_orint_` branch body and every peer negation is atomic; the outer signature still names all branches and their proof rows.

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

`ordisMerge` records each per-branch deposit in `Memory::orBookkeeping`. The cohort is the exact `(parentValidity, originalOrSignature)` pair: both component ids are packed and interned as one `cohortId` by `mintOrCohortId`. Bookkeeping is therefore keyed by packed `(exprId, cohortId)` `lbStateInterner` pairs and valued as decoded-lex ordered branch-disjunct id sets (`D-135`; the decoded order feeds the origin rows and cleanup loop). For a nested OR, the registered structural count is the number of recursively flattened non-OR leaves, while the signature component remains the original outer OR. When the recorded branch count for that exact `(expr, parentValidity, originalOrSignature)` reaches `Memory::orDisjunctCount[cohortId]` (registered at OR-disintegration mint time, see [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp)), the convergence path fires. Equal OR signatures at different stack positions occupy independent rows ([D-218](../40_decisions.md#d-218), [D-219](../40_decisions.md#d-219), [I-167](../30_invariants.md#i-167)).

1. **Promotion via `sameIterationInternalMail`.** `ordisMerge` pushes `(addExpression, addExpressionLevels, parentValidity)` onto `Memory::sameIterationInternalMail.statements` — the same revival channel `revisitRejectedIntegration2` uses, see [`memory.hpp` `Mail` (used as `Memory::sameIterationInternalMail`)](../../GL_Quick_VS/GL_Quick/src/memory.hpp). The drain at the top of the next hashburst body absorbs the tuple via `addExprToMemoryBlock(..., status=1,..., parentValidity,...)`, running the parent-scope deposit through the full kernel pipeline (`toBeProved` discharge of the parent's forward-conjunct goals included). Deferring to `sameIterationInternalMail` instead of calling `addExprToMemoryBlock` directly avoids reentrant kernel invocation in the post-`addStatement` loop.
2. **Per-branch cleanup of the converged expression.** For each branch validity in the recorded disjunct set, `ordisMerge` calls `removeExpressionFromMemoryBlock(EncodedExpression(addExpression, branchValidity), mb, /*state=*/0)` — the standard removal procedure. The N per-branch copies of the converged expression collapse into the single parent-scope copy. The parent-scope copy remains visible to each branch via comparable-scope inheritance, so branches can reuse it. Branches keep all their *other* facts and stay live (unlike the `_orint_` case-split, which wipes the entire branch — see §3a). `exprOriginMap` (history) is not touched; `intStatementLevelsMap` and `intKnownStatements` are also retained, so a re-deposit attempt at a wiped branch scope short-circuits at the standard `intKnownStatements` gate.

#### Dead-branch retirement

A branch whose **asserted disjunct is refuted** — the disjunct's negation is `known` at the branch scope or one of its ancestors — is an ex-falso factory: its equality-class merges and variant rewrites breed unboundedly under an inconsistent premise, and its cohort's convergence count can never be reached. Such a branch is retired ([D-242](../40_decisions.md#d-242), [I-172](../30_invariants.md#i-172)):

- **Detection** is a probe inside `ordisMerge` (so both deposit paths reach it): on each `Disintegration`-classified deposit, the branch payload's wrapped disjunct is negated and probed non-minting against `intKnownStatements` up the `parentOf` chain. A hit stages the branch vid on `Memory::pendingDeadOrBranches` (persistent-arena pod set, the `pendingDisprovedGoals` pattern). `known` is the sound bit — the record is immortal ([I-58](../30_invariants.md#i-58)) and a scope's fact set is monotone.
- **Retirement** runs in [`drainDeadOrBranches`](../../GL_Quick_VS/GL_Quick/src/prover.cpp), end-of-burst, after `drainDisprovedGoals` and before the `pendingWipeScopes` drain: the branch subtree is queued for the radical wipe and inserted into `intValidityNamesToFilter` (permanent — a filtered branch never receives a deposit again, so it never re-stages); the cohort's `orDisjunctCount` drops in place (zero survivors retires the cohort wholesale — the count row leaves the container, every run empties, and the joint contradiction surfaces through the parent-scope K rules + the contradiction machinery; cohort identity — the `(parentValidity, originalOrSignature)` pair, [I-167](../30_invariants.md#i-167) — is unchanged, no successor cohort is minted); every `orBookkeeping` run of the cohort loses the dead branches' disjunct entries **before any count comparison** — a stale dead entry against the reduced count could fire a convergence no surviving branch ever derived; an only-dead run leaves the container. Then convergence is re-checked: a surviving row whose run reaches the reduced count re-fires through `ordisMerge` itself (a missing levels row at the survivor branch means that expression converged earlier — a defined skip).
- **History is kept** ([I-44](../30_invariants.md#i-44)): earlier full-cohort convergences cite dead-branch derivations as `or convergence` ingredients, and the chapter walker must still resolve them. A staged branch whose cohort has no `orDisjunctCount` row was retired wholesale by the same seam's disproof drain — a defined skip.
- **The retirement is artifact-visible** ([D-238](../40_decisions.md#d-238)): the drain records each retired disjunct plus the shallowest scope where its refutation is `known` (`Memory::orRetiredDisjuncts`, persistent arena), and every later convergence of the shrunk cohort emits, per retired branch, the reductio ingredient `(negate(D_i), scope)` alongside the survivors' branch derivations — the exported row accounts for all K disjuncts and `check_or_convergence` validates both entry forms.

History: pre-[D-34](../40_decisions.md#d-34) the bookkeeping path was `trackOrBookkeeping` inside `addStatement`. It was structurally blind to D-33's cross-scope deposits and carried a too-aggressive cleanup (wiped every statement at the branch scopes via prefix-match, plus added the branch validities to `validityNamesToFilter` to block all future inserts). FTA-rung-1 §9a / §9b symptoms: the per-branch preorder facts produced by D-33's descendant-direction rewrite landed at the branch scopes (`current_proof_state.md` confirms all four cells), but `trackOrBookkeeping` never saw them and convergence never fired. D-34 fixes both — the kernel-level placement closes the visibility gap, and the cleanup is narrowed to just the converged expression so branches stay alive for further convergences (e.g. the second preorder of the same impl scope's body).

---

## `classifyOrScope` — the scope-role oracle

Defined at [`prover.hpp`](../../GL_Quick_VS/GL_Quick/src/prover.hpp). Returns one of:

- `NotOrScope` — scope is not related to an OR split.
- `Integration` — scope is part of an OR re-integration (reformulation-for-integration over an OR head).
- `Disintegration` — scope is one of the per-disjunct branches.

Classification is based on the payload prefix of the scope's name. See [`20_core_concepts/04_validity_stack.md`](04_validity_stack.md). The elementary-step phase helpers use it to decide OR-aware bookkeeping; `ordisMerge` uses it to detect `_ordis_` convergence and route the promoted expression through `sameIterationInternalMail`; the post-`addStatement` block in `addExprToMemoryBlock` uses it for OR-integration emission and bare-implication republish; and `cleanUpOrIntegrationBranches` uses it to find sibling integration branches to wipe after a winner emerges.

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

After the single prove pass, [`analyzeExpressions`](../../GL_Quick_VS/GL_Quick/src/prover.cpp) walks `globalTheoremList` and runs `headSwitchOne` on each theorem. For any theorem with a negated, non-quantified premise, the contrapositive is computed: negate the head (an already-negated head loses its `!` instead — double-negation cancellation, [D-217](../40_decisions.md#d-217)), replace the negated premise with the new premise. The (original, contrapositive) pair is stored in `orPairsFromHeadSwitch` (class member of `ExpressionAnalyzer`).

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

`docs/fta_ladder/README.md` is the master index of FTA-ladder rungs and `docs/fta_ladder/rung<N>/current_proof_state.md` is the standing entry point for each rung's in-flight state. Each rung climbs another OR-step toward the Fundamental Theorem of Arithmetic. Rung 1 (`{0,1} = [0,1]`) supplied the initial two-way playbook. Rung 2 (`{0,1,2} = [0,2]`) closed on 2026-07-25: flat atomic disintegration proves the forward containment, while flat atomic integration plus the `_orint_`-scoped secondary-variable budget completes the two-level predecessor descent for the reverse containment. Two independent full-pipeline runs each passed 133,523 verifier checks with zero failures and produced four byte-identical proof artifacts. The trace-backed proof and exact closure ledger live in [`docs/fta_ladder/rung2`](../../fta_ladder/rung2/); the false cross-pair disproof continues separately as rung 2.1.

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
- **Reduced-cohort `or convergence` rows — closed by [D-238](../40_decisions.md#d-238).** The activation run exported the first reduced-cohort convergence row and `check_or_convergence` rightly rejected it (one entry for a three-disjunct OR). The producer now records each retired disjunct with its refutation scope (`Memory::orRetiredDisjuncts`) and the convergence history cites the reductio ingredient per retired branch; the checker (maintainer-consented I-16 change) accepts the two entry forms with exactly-once disjunct coverage.
- **`or branch proven` and `or branch assumption` are first-class verifier tags (D-35, 2026-05-03).** Previously claimed retired/overridden — incorrectly: no override path existed and the prover always emitted the tags live. FTA-rung-1 chapter `1209_direct_proof.txt` carries both. The verifier now registers `check_or_branch_proven` / `check_or_branch_assumption` in `TAG_CHECKERS`, validating: parent/branch scope ancestry, GL-binary disjunct membership (modulo equality symmetry), and the branch-payload `_boundary_orint_<or>_(<disjunct>)` substring pattern. See [`08_proof_tags.md`](08_proof_tags.md#or-branch-proven) for the full row layouts.

### Suspected fragility

- **`or convergence` spec'd row layout — gap closed by D-36.** `ordisMerge` (`prover.hpp`) now emits the spec'd layout `<C> <parent> or convergence <OR> <parent> <C> <branch_D1> <C> <branch_D2> …`. The branch derivations of `C` remain in `exprOriginMap` (untouched by `removeExpressionFromMemoryBlock(state=0)`); `buildStack` renders them automatically by recursion. Side-effect of the fix: 4 previously-hidden `_ordis_` per-branch case-split rows surface in chapter export under `or disintegration` tag — they pass the D-36-extended `check_or_disintegration`. Net incubator-verifier state: 0 failures.

- **Sibling-wipe atomicity.** `cleanUpOrIntegrationBranches` walks and wipes. If an LB were processed in the same cycle its wipe touches, the effect on that LB's iteration state would be ill-defined — but cross-LB writes during the parallel phase are forbidden ([I-28](../30_invariants.md#i-28)), so a worker thread never wipes an LB another worker is processing; the wipe runs in the single-threaded post-join drain.
- **Branch-seed distinctness.** Each `_ordis_` branch is seeded only with its selected disjunct. If two disjuncts are textually identical (malformed OR conjecture), two branch scopes therefore carry identical asserted seeds and can perform duplicate work. Not currently observed but not excluded.
- **Nested OR branch opacity — closed in both directions.** `flattenOrLeaves` turns the whole contiguous OR tree into one ordered leaf cohort before `_ordis_` or `_orint_` production. The outer signature remains the only namespace/provenance identity, and verifier reconstruction mirrors the same recursion ([D-218](../40_decisions.md#d-218), [D-220](../40_decisions.md#d-220)).
- **Depth-cost of `comparable`.** OR branching creates scope-tree depth. Post-migration `comparable` uses `pairMap` (O(log N) typical), but with deeply-nested OR chains, log N grows.
- **Cross-stack same-orSignature race — closed by D-219.** The old `Memory::orBookkeeping[(expr, orSignature)]` and `orDisjunctCount[orSignature]` conflated equal signatures at different stack positions. The observed rung-1 impl24/top-level collision happened to be absorbed later by Site F's ancestor dedup, but mixed cohorts could still reach the structural count. Both containers now share a parent-scoped `cohortId`; the trace prints `parent` and `orSig`, and the direct test holds one parent at one seen branch while completing the other. See [D-219](../40_decisions.md#d-219).

### Not exercised by tests

- **OR-re-integration path.** The `_orint_` payload scope kind exists but is sparsely exercised by the current Peano+Gauss corpus. FTA-ladder rungs may exercise it more; a regression in OR-re-integration could land silently before the ladder climbs.
- **`classifyOrScope` misclassification.** If a payload prefix silently drifts, classification returns the wrong enum. No test verifies the mapping.

---

## Open questions

- **OPEN-1 — contradictory-branch cohort rewrite — RESOLVED (2026-07-30, ).** Full contradiction discharge remains deliberately main/main only ([D-222](../40_decisions.md#d-222)). The branch-level consequence is now implemented as [Dead-branch retirement](#dead-branch-retirement): a branch whose asserted disjunct is refuted is wiped and filtered, its cohort's `orDisjunctCount` drops in place (no successor cohort — cohort identity unchanged, [I-167](../30_invariants.md#i-167)), booked runs are cleaned of dead entries, and convergence re-checks over the survivors. One survivor is the ordinary reduced-count case; zero survivors retires the cohort wholesale, and the joint contradiction surfaces through the parent-scope K mutual-exclusion rules + the contradiction machinery — primed reductio discharge; the vacuous-premise flag with global-theorem reversion for a normal LB at main; or, for a cohort under a deeper hypothetical scope of a normal LB, nothing at the LB level (the refuted disjunction only proves that scope's assumption unsatisfiable — the enclosing clause is vacuously true and the LB's theorems stay valid). The rejection proof needs no new artifact: detection requires the refutation to already be `known` at an ancestor, so the reductio product pre-exists with its own history. The rung-2.1 ES3 numeral cohorts supplied the acceptance case ([D-242](../40_decisions.md#d-242)).

---

## See also

- [`20_core_concepts/04_validity_stack.md`](04_validity_stack.md) — scope mechanics.
- [`10_pipeline/04_prover.md`](../10_pipeline/04_prover.md) — `cleanUpOrIntegrationBranches`, `classifyOrScope`, contradiction discharge.
- [`20_core_concepts/08_proof_tags.md`](08_proof_tags.md) — `or disintegration`, `or convergence`, `or theorem`, `variable copy`.
-,,.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
