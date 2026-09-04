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

**Element polarity.** A disjunct may itself be NEGATED (`(a=b) ∨!(c=d)`, the successor-functionality OR from single-direction licensing), and the registered elements store each disjunct verbatim — sign included. Every consumer that rebuilds the De Morgan conjunct of an element negates it with double-negation cancellation, so a negated element contributes its bare positive core. A blind `!` prefix (or a positive-only store) flips a negated disjunct's sign and mints a semantically FALSE or theorem — see [I-175](../30_invariants.md#i-175) and [D-251](../40_decisions.md#d-251) for the recorded incident (an incubator anchor contradiction).

The Python popup builder (`build_gl_binary_map` in [`generate_full_proof_graph.py`](../../generate_full_proof_graph.py)) and the C++ `expandSignature` (`prover.cpp` CASE 4 — OR) both produce the same nested form: two-element OR is `!(&!E1!E2)`; for `n ≥ 3` elements, `!(&current!Ek)` is wrapped around the running expression for each successive `Ek`.

The canonical OR theorem on Peano — `or0[7,2,1,3]` in anchor-argument form — reads *for any n ∈ N: n = 0 ∨ n has a predecessor.* This was the historical branching milestone and the worked example in the present chapter. Rung 1 of the FTA ladder (`{0,1} = [0,1]`) established the minimal two-way case split. Rung 2 (`{0,1,2} = [0,2]`) exposed and closed the next boundary: a three-way membership compiled as a nested binary OR is now one flat three-alternative disintegration cohort.

---

## Three stages of an OR proof

### 1. OR disintegration

The prover reaches an OR head and decides to split. Since [D-258](../40_decisions.md#d-258) there are two ways to reach this stage: a compiled or-category fact (the classic entry), or the **negated-AND De-Morgan door** — an asserted `!(op[args])` whose compiled definition body is an AND is a disjunction in De-Morgan clothing (`!(& C1.. Cn) = !C1 ∨.. ∨ !Cn`, each disjunct at its true polarity, double negation cancelled), and the door in `disintegrateExprCore2`'s no-instruction region builds those leaves at runtime (no `orN` operator minted; the negated compound itself is the cohort signature) and hands them to the shared downstream `consumeOrLeavesCohort` ([I-179](../30_invariants.md#i-179)). From here on both entries behave identically. Two distinct things happen:

**1a. K mutual-exclusion sub-implications at the parent scope.** Before producing anything, `flattenOrLeaves` recursively expands every contiguous compiled-OR child in the prepared instruction graph. It preserves element order and stops at each non-OR child. For each resulting leaf `d_i`, the prover emits the implication

```
(>[](!d_0) ... (!d_{i-1}) (!d_{i+1}) ... (!d_{N-1}) d_i)
```

— "if every other disjunct is false, then `d_i` must hold". For `K = 2` the chain collapses to a single negated premise, so an OR `A ∨ B` produces the rule pair `!A → B` and `!B → A`. These K rules are inserted into the parent scope's hash memory (no per-branch scope yet) and become available as forward-firing implications.

**The rules are compacts, not on-the-spot text** ([D-309](../40_decisions.md#d-309), [I-208](../30_invariants.md#i-208)). A registry or's K rules — and its subset-exclusion rules below — are compiled ONCE onto the or entity's `implications` list as `(implication<N>[u_…])` instances over the or's own `u_` tokens (`compileOrKRules` at the or's mint, through the mail-broadcast compaction door `compileImplicationToCompact`; the stored argument order is the projection onto the or's tokens), and the GL binary carries the list. At consumption `consumeOrLeavesCohort` substitutes `u_p` → the instance's argument p positionally (repeated arguments are legal — `(implication<N>[9,9,1,4])` is a valid instance), and hands every compact to `prepareIntegrationCore` + `disintegrateExprCore2` as a product statement: a child of the or's key that registers at the parent scope (it rides the main-scope delta like any statement and is rewritten by `applyEquiClasses`), and whose implication branch expands it into exactly the rule text above. Provenance: the compact carries the `disintegration` origin citing the or's expanded form; the rule carries an `expansion` origin citing the compact — one more chapter row than before. The negated-AND De-Morgan door has no or entity and keeps building its K rules on the spot.

Tagged `disintegration` in the processed proof graph. Checker: `check_disintegration`'s `or` branch at [`verifier.py`](../../verifier.py) — recursively reconstructs the same substituted non-OR leaf list, expands a compact row `(implication<N>[…])` through its implication binary entry first (`_expand_implication_compact` — registry ors emit the compact, not the rule text), then verifies that the rule peels into a head matching one leaf and a premise multiset matching the negations of all other leaves (the shared tail `_check_or_disintegration_against`). The expansion-row chain (flat expanded De-Morgan form ↔ the source) provides the dispatch path; the source is either the outer compact `(or<N>[…])` name, or — for the negated-AND door — the negated compound `!(name[args])` itself, whose and-category entry supplies the disjuncts as its negated conjuncts (`check_expansion` validates the matching expansion row the same way).

The `disintegration` origin is recorded inside [`disintegrateExprCore2`](../../GL_Quick_VS/GL_Quick/src/prover.cpp)'s OR case, at the parent `validityName`, with origin = the expanded De-Morgan form `!(&!d_0…!d_{K-1})` (the same key the matching expansion row records). See [D-55](../40_decisions.md#d-55).

**1b. Per-branch case-split — SEQUENCED.** The flattened non-OR leaves form one cohort, but cohort creation mints NO branch scope (D-250, [I-174](../30_invariants.md#i-174)): the bootstrap registers the FULL structural leaf count in `orDisjunctCount` (convergence semantics unchanged), queues EVERY leaf in the deloadable pending queue `Memory::orPendingBranches` (with the cohort's seed level run in `orPendingLevels`), and stages the cohort on `Memory::pendingOrReleases`. The bootstrap runs inside `standardProcessing`'s internal-mail drain, whose step-3 clear would wipe any seed inserted into the channel being drained — so the FIRST release happens at the same end-of-burst drain as every later one. There the leaves are ranked — tier 0: everything that is not a positive equality, by descending token count (most structurally complicated first), byte-lex ties; tier 1: bare equalities; tier 2: equalities carrying an anchor-slot argument, dead last (`pickTopOrDisjunct` / `collectAnchorArgs` in [`prover.hpp`](../../GL_Quick_VS/GL_Quick/src/prover.hpp)); a route-(a) cohort's one-shot `orStarterPick` row overrides the FIRST pick with the admitted disjunct (D-252) — and the top-ranked disjunct's scope `"ordis_(outer-or<N>[…])_(d_i)"` is minted with its seed (the asserted disjunct) on `sameIterationInternalMail` plus an `or disintegration` mail origin ([I-176](../30_invariants.md#i-176)) — the next step's absorb runs it through the full kernel pipeline, so a COMPOUND disjunct (an existence form) decomposes like any deposited statement. The original outer signature names every sibling branch and the cohort; no intermediate OR-valued branch exists. The negations of the other leaves are not seeded in `_ordis_`; the K mutual-exclusion rules from 1a remain available from the parent scope. Each released branch runs the standard prover machinery in its own scope ([D-218](../40_decisions.md#d-218), [I-166](../30_invariants.md#i-166)).

**Release.** The next pending branch mints only when the live branch RESOLVES: (a) it deposits an expression matching a `toBeProved` goal at a scope on its parent chain up to main — `ordisMerge`'s goal probe stages the cohort on the persistent `Memory::pendingOrReleases` inbox; or (b) it retires refuted — `drainDeadOrBranches` stages the cohort ([D-242](../40_decisions.md#d-242) retirement doubles as the release trigger). The end-of-burst `drainPendingOrReleases` ([`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp), between `drainDeadOrBranches` and the `pendingWipeScopes` drain) re-ranks the REMAINING pending disjuncts (the release order is a pure function of the pending set plus the LB's anchor arguments — re-derived, never stored) and mints at most one branch per staged cohort per burst. An undecided live branch stalls its cohort — a defined degradation, no timeouts, no caps. A resolved branch is not left working either: once every `toBeProved` goal on its chain is known at the branch or above, the end-of-burst `freezeResolvedOrBranches` FREEZES it — its subtree leaves the request universe, nothing is wiped (see [Branch freeze](#branch-freeze), [D-306](../40_decisions.md#d-306)). In an LB whose goal registry is empty the release drain releases nothing (the staging drops): no goals, no or branches. A re-deposit of the same or statement at the same parent hits the bootstrap guard (an existing `orDisjunctCount` row) and re-emits only the flat K rules. This one-live-branch discipline is the explosion protection that lets admission be unconditional: the expensive assumed-equality branches (equivalence-class replication near main) open only after every productive branch already resolved, and in descent contexts they refute within a few bursts and retire — one bounded pulse, not a standing population.

Tagged `or disintegration` in the processed proof graph. Checker: `check_or_disintegration` at [`verifier.py`](../../verifier.py).

**1a is unconditional at every OR depth** ([D-211](../40_decisions.md#d-211)): the expansion record and the K mutual-exclusion implications are flat hash rules that create no scopes, so they are emitted even when the OR statement lives inside an existing `_ordis_`/`_orint_` branch scope — that is the disjunctive-syllogism consumption of an OR (e.g. deriving `∃pred(q)` from `q≠0` inside an `_orint_` branch, the rung-2 second predecessor split). **1b is gated three times: on the LB's goal registry being non-empty** (`intToBeProved`, any scope, evaluated at every open attempt — a goal-less LB neither opens nor parks a cohort and releases no pending branch; the `try_contradiction` / `try_contradiction_negated_head` reductio blocks, seeded status 0 only, therefore never split — [D-306](../40_decisions.md#d-306)), **on `currentOrDepth < parameters.max_or_depth`** — nested `_ordis_` scopes are the scope explosion the cap exists to prevent; one admitted outer OR consumes one depth regardless of how many contiguous nested OR syntax nodes it contains — **and on the OR-admission check** (the two-route opening of D-252, see below). When any gate refuses, the K mutual-exclusion implications still land in hash memory but the per-branch case-split is skipped — auditability for the K rules is preserved either way because the `disintegration` origin record is emitted together with them.

**1c. Per-leaf INTRO implications for an or-shaped implication premise** ([D-237](../40_decisions.md)). The K rules are the elimination half of an OR's flat consumption; the introduction half emits in `disintegrateExprCore2`'s IMPLICATION branch: when an installed compiled implication's premise element is a disjunction (compiled-map probe — implication elements have no instruction-graph entities), the prover also emits, per flattened leaf,

```
(>[bound] d_k (or<N>[…]))
```

— "one true disjunct states the disjunction". The leaves come from a compiled-map twin of `flattenOrLeaves` (per-level signature→instance substitution, recursing into or-category elements only), and leaf + head are fed to the reconstruction in RAW element form so `u_` arguments stay free and only the changeable element variable binds. Propositionally valid unconditionally — no soundness surface. This is what lets a single concretely-true disjunct (typically reached through an equality1 witness alias, since the true disjunct at a concrete element is reflexive) fire the implication's own rule: the rung-2.1 chain `2_copy=2 → or(2_copy) → 2_copy∈M → 2∈M`. Each intro rule carries an `expansion` origin whose dependency is the compact implication statement itself — the same chapter-capable antecedent the main rule's expansion record cites; `_try_expand`'s or-aware implication acceptance covers the shape on the verifier side.

An intro-FIRED or head never re-disintegrates ([D-241](../40_decisions.md)): the intro shape is re-detected at every install by `isOrIntroInstall` (`memory.cpp` — single premise byte-equal to one flattened leaf of an or-category head; shape-based so status-3 mail-recovered reinstalls re-mark), the derived per-rule `LocalMemoryValue::disintegrationAllowed` goes false, and the firing site reads it as the rule-intrinsic half of `FiringRecord::doNotDisintegrate`. The stated disjunction stays a flat fact for the membership-intro clause and the K rules — an intro-derived or's source leaf is true at the same scope, so a case split over it is redundant by construction (its `_ordis_` cohorts were a pure fan-out multiplier: branch assumptions breed equalities ex falso, equalities re-fire the intro rules). Elimination-fired or heads (premise not a leaf) and status-0 assumed or fuel branch exactly as before.

**Admission gate (D-252, 2026-08-03 — TWO-ROUTE opening; supersedes D-250's eager door override).** An or head reaching the cohort-mint site opens its cohort on either of two routes, and otherwise PARKS. **Route (b)** is the restored [D-32](../40_decisions.md#d-32) criterion: the firing rule is a product of disintegration (`LocalMemoryValue::productOfDisintegration`, stamped at install time in [`addToHashMemory`](../../GL_Quick_VS/GL_Quick/src/memory.cpp) when at least one premise carries a `u_`-prefixed arg, threaded through `checkLocalEncodedMemoryStatic`'s FiringRecord and the disintegration signals to `addExprToMemoryBlock`'s door as `allowOrDisintegration = allowOrDisintegration && !doNotDisintegrate`) — such heads open unconditionally, as in rungs 1+2. **Route (a)** is demand-driven: some disjunct's PRODUCT TEMPLATE — an existence disjunct's compiled-definition witness fact with the witness slot in `marker` form (the body instantiated in `u_`-form so the witness is the only non-`u_` token), or an operator-application disjunct itself; equalities and negated disjuncts contribute nothing — is already a key of the algebra `admissionMap` (tagged or untagged; the demands arrive through the existing marker machinery plus the ordis-only qualification route, [I-177](../30_invariants.md#i-177)). The probe is `isAdmitted`'s prologue, strictly read-only — the admission maps' single ordis touchpoint — and deliberately NEVER the I-6 shape rule (`isAllowedAsOperatorInput` would admit every exists-predecessor cohort and kill parking). A route-(a) cohort's admitted disjunct becomes the release STARTER (the one-shot `Memory::orStarterPick` row `drainPendingOrReleases` consumes before its ranking). **Neither route:** a real deposit (`allowOrProbe`, true only through the kernel door — the hypothetical path neither opens nor parks) files the cohort in `HashMemory::rejectedMapOrdis` under each operator-based product template at its validity, value = the clean or statement + the seed level run, and `revisitRejectedOrdis` at every admission key-gain seam later re-deposits it on `sameIterationInternalMail` so the absorb re-runs the whole consumption ([I-178](../30_invariants.md#i-178) — including the un-know of the known parked statement that lets the re-deposit pass Site F). D-241's intro-fired suppression and every other `doNotDisintegrate` context (integration-justified rules, the D-29 firing-context clauses) block both routes. The `orAdmissionSet` container stays removed (`D-135`); the dump prints the literal empty section header for Rule-14 byte stability. The explosion protection is now BOTH the demand filter and the sequenced one-branch-at-a-time release (§1b).

**Dual filing (D-267, ).** The parked cohort ADDITIONALLY files in `HashMemory::rejectedMapOrdis2` under each ELIGIBLE disjunct's clean GROUND text (eligibility = `ordis2KeyEligible`: compiled non-atomic entity — predicates included, `operators` not consulted; arity ≥ `kOrdis2DemandMinArity`; polarity verbatim — a negated compound files WITH its `!`). That second index speaks the SAME key language as the `admissionMapOrdis2` demand map, so the demand drain's wake (`revisitRejectedOrdis2`) is a plain key rendezvous — the pair completes each other the way algebra's `admissionMap`/`rejectedMap` do (whichever side arrives second wakes the other; a route-(c) probe of the demand map at deposit time covers the demand-first order because it precedes the park in the same chain). Only the ordis2 drain wakes the second index; the algebra key-gain seams keep waking only the old map ([I-183](../30_invariants.md#i-183)).

**Or-uniqueness gate (I-219, ).** Upstream of everything in this section sits one more gate, at the deposit door itself: an `(or<N>[…])` deposit whose equi class already holds a fully-processed representative at the deposit's scope is a PASSIVE statement — it registers (door-canonical text) and is multiplied by `applyEquiClasses` like any statement, and nothing else happens: no `disintegrateExpr2`, so no K/subset-exclusion compact instantiation, no cohort bootstrap, no park, no dual filing, no `fullyDisintegrated` stamp — on every entry path (local deposits and flag-5 relay arrivals alike). "Fully processed at the scope" is the per-scope `Memory::processedOrLedger` (persistent arena, written when an or-compact deposit reaches `disintegrateExpr2`); each row carries the or's class-canonical id beside its original id (`packInt32Pair(canonical, original)`), re-keyed by `recanonicalizeProcessedOrLedger` at the one class-change seam (`updateEquivalenceClasses`, right after the `changedClassesThisStep` push), so the probe is an exact id membership test on either half — no canonicalization per deposit, and still exactly door-consistent (I-217's walk computes the canonical half; [D-335](../40_decisions.md#d-335)); `checkForEquivalence`'s whole-arg Cartesian probe stays active behind the gate for the ors it does not suppress. Every path that intentionally re-runs an or consumption clears the ledger row first: the parked-or un-know (`resetParkedOrStatementRegistries`) and the two ordis park hooks' re-mail loops; the disproof cleanup sweeps rows at wiped scopes. Why: or proliferation was self-feeding — every equi spelling reaching the door got the full per-statement consumption, minting statements and equalities that minted more spellings.

**Duplicate-cohort retirement ([D-331](../40_decisions.md#d-331)).** The gate stops new duplicates; cohorts opened BEFORE their signatures became class-equal are collapsed at the end-of-burst `retireDuplicateOrCohorts` drain (between the dead-branch retirement and the freeze sweep): ledger entries grouped by their canonical half (current, since the ledger is re-keyed at every class change), a group with two or more OPEN cohorts keeps the one with the most statements under its `_ordis_` branch scopes (lex tie-break) and retires the rest — live branches wiped in the same burst, scheduling rows erased, frozen branches and history kept, the losers' work re-derived under the keeper when needed. Parked-only duplicates need no drain: the ordis hooks re-mail them and the door's gate makes the re-file passive.

History. Pre-D-31, the gate was structurally dead — zero `_boundary_ordis_` scopes ever minted. D-31 introduced a broad bypass keyed on the deposit scope's last namespace segment matching `(implicationNN[…])`; this caused a runtime explosion (anchor-bound rules' broad fan-out into implication scopes spawned case-split branches everywhere). D-32 narrowed to the sharper "implication itself is a product of disintegration" criterion, preserving the FTA-rung-1 Step 9 lower (impl24) closure path while cutting the impl24 `_ordis_` row count by 88% — at the price of never splitting corpus-fired or heads (the A15 `m ≠ 0` stall). D-250 opened admission fully and contained the fan-out by sequencing — proving A15 at 3.2× shortcut runtime. D-252 reversed the polarity to park-by-default, admit-on-demand: A15 intact through the row-34 `ordisOnly` demand, runtime back from ~155 s to ~56 s.

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
2. **Per-branch cleanup of the converged expression.** For each branch validity in the recorded disjunct set, `ordisMerge` calls `removeExpressionFromMemoryBlock(EncodedExpression(addExpression, branchValidity), mb, /*state=*/0)` — the standard removal procedure. Before each removal the branch derivation's history rows are shipped to `mailOut` (`copyOriginRowsToMailOut`): the removal erases the branch row from the per-step delta `fillMailOut` walks, and the last-released branch converges in the step it derives, so a descendant walking the mailed `or convergence` row would otherwise never receive that branch's derivation ([D-311](../40_decisions.md#d-311)). The N per-branch copies of the converged expression collapse into the single parent-scope copy. The parent-scope copy remains visible to each branch via comparable-scope inheritance, so branches can reuse it. Branches keep all their *other* facts and stay live — until they freeze (see [Branch freeze](#branch-freeze): a frozen branch keeps every fact and every bookkeeping row, it just stops feeding request generation) — unlike the `_orint_` case-split, which wipes the entire branch (see §3a). `exprOriginMap` (history) is not touched; `intStatementLevelsMap` and `intKnownStatements` are also retained, so a re-deposit attempt at a wiped branch scope short-circuits at the standard `intKnownStatements` gate.

#### Dead-branch retirement

A branch whose **asserted disjunct is refuted** — the disjunct's negation is `known` at the branch scope or one of its ancestors — is an ex-falso factory: its equality-class merges and variant rewrites breed unboundedly under an inconsistent premise, and its cohort's convergence count can never be reached. Such a branch is retired ([D-242](../40_decisions.md#d-242), [I-172](../30_invariants.md#i-172)):

- **Detection** has two probes running the same predicate. The original sits inside `ordisMerge` (so both deposit paths reach it): on each `Disintegration`-classified deposit, the branch payload's wrapped disjunct is negated and probed non-minting against `intKnownStatements` up the `parentOf` chain. The second is `refutedOrBranchAtOrAbove` ([`prover.hpp`](../../GL_Quick_VS/GL_Quick/src/prover.hpp)), consulted by `applyEquiClasses` Pass 1 before back-applying a delta class — it walks the class validity's ancestor chain, runs the identical probe at each `_ordis_` disintegration scope, and doubles as the equi-apply's dead-scope skip so a refuted branch's classes are never ground across the registry in the burst that would wipe them ([D-262](../40_decisions.md#d-262), [I-181](../30_invariants.md#i-181)). Either probe's hit stages the branch vid on `Memory::pendingDeadOrBranches` (persistent-arena pod set, the `pendingDisprovedGoals` pattern). `known` is the sound bit — the record is immortal ([I-58](../30_invariants.md#i-58)) and a scope's fact set is monotone.
- **Retirement** runs in [`drainDeadOrBranches`](../../GL_Quick_VS/GL_Quick/src/prover.cpp), end-of-burst, after `drainDisprovedGoals` and before the `pendingWipeScopes` drain: the branch subtree is queued for the radical wipe and inserted into `intValidityNamesToFilter` (permanent — a filtered branch never receives a deposit again, so it never re-stages); the cohort's `orDisjunctCount` drops in place (zero survivors retires the cohort wholesale — the count row leaves the container, every run empties, and the joint contradiction surfaces through the parent-scope K rules + the contradiction machinery; cohort identity — the `(parentValidity, originalOrSignature)` pair, [I-167](../30_invariants.md#i-167) — is unchanged, no successor cohort is minted); every `orBookkeeping` run of the cohort loses the dead branches' disjunct entries **before any count comparison** — a stale dead entry against the reduced count could fire a convergence no surviving branch ever derived; an only-dead run leaves the container. Then convergence is re-checked: a surviving row whose run reaches the reduced count re-fires through `ordisMerge` itself (a missing levels row at the survivor branch means that expression converged earlier — a defined skip).
- **History is kept** ([I-44](../30_invariants.md#i-44)): earlier full-cohort convergences cite dead-branch derivations as `or convergence` ingredients, and the chapter walker must still resolve them. A staged branch whose cohort has no `orDisjunctCount` row was retired wholesale by the same seam's disproof drain — a defined skip.
- **The retirement is artifact-visible** ([D-238](../40_decisions.md#d-238)): the drain records each retired disjunct plus the shallowest scope where its refutation is `known` (`Memory::orRetiredDisjuncts`, persistent arena), and every later convergence of the shrunk cohort emits, per retired branch, the reductio ingredient `(negate(D_i), scope)` alongside the survivors' branch derivations — the exported row accounts for all K disjuncts and `check_or_convergence` validates both entry forms.

History: pre-[D-34](../40_decisions.md#d-34) the bookkeeping path was `trackOrBookkeeping` inside `addStatement`. It was structurally blind to D-33's cross-scope deposits and carried a too-aggressive cleanup (wiped every statement at the branch scopes via prefix-match, plus added the branch validities to `validityNamesToFilter` to block all future inserts). FTA-rung-1 §9a / §9b symptoms: the per-branch preorder facts produced by D-33's descendant-direction rewrite landed at the branch scopes (`current_proof_state.md` confirms all four cells), but `trackOrBookkeeping` never saw them and convergence never fired. D-34 fixes both — the kernel-level placement closes the visibility gap, and the cleanup is narrowed to just the converged expression so branches stay alive for further convergences (e.g. the second preorder of the same impl scope's body).

#### Branch freeze

A branch that has done its job is FROZEN, not wiped ([D-306](../40_decisions.md#d-306), [I-206](../30_invariants.md#i-206)):

- **Live registry.** `ordisMerge` registers every branch it sees (the seed deposit is the first sighting) in `Memory::orLiveBranches` — persistent arena, never deloaded; a frozen, dead-staged or filtered branch is never re-registered.
- **Resolution predicate.** A live branch is resolved when EVERY `intToBeProved` key whose scope lies on the branch's chain — the cohort parent or one of its ancestors up to `main` — is `known` at the branch scope or a strict ancestor of it (`ancestorKnown`, self included; the ancestor clause covers goal deposits Site F refused as ancestor-known, [I-187](../30_invariants.md#i-187)). Zero chain goals resolve vacuously. Goals at unrelated scopes (siblings, sub-scopes of other branches) are not the branch's business and are ignored. This is stricter than the release trigger above (one goal match): the successor is released at the first goal, the branch freezes only when all its chain goals are covered.
- **Sweep.** `freezeResolvedOrBranches` ([`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp)) runs at the end of every burst — after `drainDeadOrBranches`, before `drainPendingOrReleases` — over every live branch, so a goal closed by another route in a later burst completes the condition. Resolved branches move to `Memory::frozenOrBranches` (a one-way latch until discharge); retired or wipe-closed branches (`intValidityNamesToFilter`) drop out of the live registry without evaluation.
- **Effect.** The single consumer is `filterIntEncodedStatements` (`memory.cpp`), the request-universe filter shared by the generator and the stump producer: a statement whose scope is a frozen branch or ANY descendant of one (children, grandchildren, every successor scope) is not used to build requests from the next `generateEncodedRequestsStatic` call on. Nothing else changes — no statement row, `known` row, `orBookkeeping` entry or history line is touched; deposits into a frozen scope are still accepted (they can complete a convergence already booked); the reached goal is NOT erased from `toBeProved` — it closes on the normal path when the converged expression lands at the parent scope.
- **Consequence.** A frozen branch derives nothing new, so a convergence on an expression it never deposited cannot happen through it any more: `_ordis_` convergence is over expressions deposited before the freeze — the maintainer's intended trade against the resolved-branch fan-out.

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

**Or-name dedup at registration (I-23).** `constructOrTheorem` builds the u_-canonical elements FIRST and scans `compiledExpressions` for an existing `or`-category entry with the identical ordered element list — a hit reuses the existing name (no `orCounter` bump, no re-registration). Any batch may construct ORs (an incubator batch constructs the successor-functionality OR from its own direct-proved direction), so unconditional minting would give one structure a different `or<N>` per constructing batch; the reloading batch's compile-side dedup then resolves the base-form pool row to a name the producer's `global_theorem_list.txt` never used, and the verifier's origin registry match fails on the operator name ([G-64](../50_gotchas.md#g-64)).

The construction has three pieces, all in the same emit pass:

**Mirror pre-emit and the alpha-variant skip.** At conjecture-load (`analyzeExpressions`' grid setup), every qualifying conjecture's contrapositive (`headSwitchOne`) is folded into the conjecture pool as its own grid so the pair construction below can find both parents proved. A mirror that is the SAME statement up to bound-variable renaming — the totality shape, symmetric binder prefix with identical guards — is recognized by `mirrorIsAlphaVariant` ([`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp): token-aligned head pairing derives the candidate bijection, verified globally against the chain-link multiset) and skipped: scheduling it would prove the identical theorem twice on a duplicate grid. Genuinely different mirrors (the Peano or0 parents — predicate-swapping shapes) keep being scheduled.

### Producer — `headSwitchOne` populates `orPairsFromHeadSwitch`

After the single prove pass, [`analyzeExpressions`](../../GL_Quick_VS/GL_Quick/src/prover.cpp) walks `globalTheoremList` and runs `headSwitchOne` on each theorem. For any theorem with a negated, non-quantified premise, the contrapositive is computed: negate the head (an already-negated head loses its `!` instead — double-negation cancellation, [D-217](../40_decisions.md#d-217)), replace the negated premise with the new premise. The (original, contrapositive) pair is stored in `orPairsFromHeadSwitch` (class member of `ExpressionAnalyzer`).

The walk is symmetric — it visits both directions of every mirror pair. So `orPairsFromHeadSwitch` typically contains *both* `(mirror_a, mirror_b)` and `(mirror_b, mirror_a)` for the same disjunct set.

### Constructor — `constructOrTheorem` ([`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp))

Given `(existenceThm, companionThm)`, the constructor:
1. Disintegrates both into chains (the companion is recorded as parent reference only — nothing is read from its chain).
2. Collects EVERY binder-free negated premise in the existence theorem's chain — classically `!d_1 → (!d_2 → h)` ⟺ `d_1 ∨ d_2 ∨ h`, so the whole hypothesis chain folds into one n-ary OR head ([D-260](../40_decisions.md#d-260); the former single-fold left the A16 trichotomy's second negated premise behind as a hypothesis). A negated premise that binds variables stays a shared premise — folding it would orphan its binder.
3. Builds the disjunct list in chain order, TRUE polarity ([I-175](../30_invariants.md#i-175)): each folded premise un-negated, the head appended VERBATIM last (a negated head stays a negated disjunct).
4. Resolves the full ordered u_-canonical element list through `findOrMintOrOperator` — the ONE or-mint site ([D-268](../40_decisions.md#d-268)): a registry hit on the identical element vector reuses the existing name (I-23); a miss mints `or<orCounter>` and registers it in `compiledExpressions` + `coreExpressionMap`.
5. Reassembles the OR theorem with the surviving (non-folded) premises in front and the compiled `(or<N>[args])` head. `expandToBaseForm`'s or-branch writes the FLAT De Morgan form `!(&!D_1!D_2…!D_k)` for any element count — the same flat n-ary `&` the anchor definition bodies parse.

### In-run seam — `ExpressionAnalyzer::constructOrTheoremsInRun` (D-266)

OR theorems are no longer export-only: `constructOrTheoremsInRun` runs on `proveKernel`'s phase-4 barrier every iteration (after the vacuous-theorem retraction sweep, before the deferred-compaction sort + drain). It walks `globalTheoremList` for rows not yet in `orInRunScannedRows` (string-keyed — vacuous reversion erases rows in place, so an index cursor would slide), licenses each via `headSwitchOne`, folds and registers through the same `constructOrTheorem`, and queues the or theorem on `recordPendingCompaction`, so the same iteration's compaction drain broadcasts its implication compact grid-wide (status-3 rule door, two-iteration delivery). Where the broadcast rule's premises hold it fires, the `(or<N>[…])` head deposits locally, disintegrates, and the cohort machinery below opens under its normal demand gates. Compressor mode is a defined no-op; a premise-free or registers but is not broadcast.

Both seams share the canonical pair ledger `orBuiltByPair` (sorted `(min, max)` pair → constructed or string): mirrored pair orders are DISTINCT registry identities, so without the ledger the mirror row proving later double-mints `or<N>`. The first-scanned side builds (deterministic — `globalTheoremList` append order); the export walk recovers the built string from the ledger instead of reconstructing from its own pair side.

### Subset-exclusion rules — the fewer-negative-premise consumption family (D-269)

Flat consumption of a REGISTRY-OR cohort with k ≥ 3 leaves emits, beside the K mutual-exclusion rules, one subset-exclusion rule `!D_i1 &.. & !D_ij -> or(D_rest)` per excluded index subset with 1 ≤ j ≤ k−2 (`consumeOrLeavesCohort` section 1b; size-j subsets in lexicographic order, 2^k − k − 2 rules total per consumption) — since [D-309](../40_decisions.md#d-309) these rules are the or entity's compacts `implications[k..]`, compiled by `compileOrSubsetExclusions` in that very order at the `preMintReducedOrs` seam (the reduced heads exist only after the closure walk) and instantiated positionally at consumption exactly like the K rules: premises = the excluded disjuncts' negations in parent order (double-negation cancelling, I-175), head = the pre-minted reduced (k−j)-ary or-compact instantiated over the surviving leaves (instance args mapped through the parent signature positions in the reduced list's first-appearance token order). A K rule needs k−1 negative facts; the j = 1 rule fires on ONE — for B8's trichotomy, `!(=[a,b]) -> or{a<b, b<a}` fires on the reductio seed alone. The rules are flat (no scope), emitted at every or depth like the K rules, and install PARK-FIRST via the shape-detected `subsetExclusion` tag ([I-184](../30_invariants.md#i-184) — detection order-free in the premises, since the mail-compact round-trip permutes premise order): their fired heads park in `rejectedMapOrdis` and open only from the compound-demand map (Part 2). De-Morgan negated-AND cohorts are excluded (their signature resolves to no registry or; K-rules-only, the maintainer's approved scope). Verifier acceptance: `_check_or_subset_exclusion` (premises as a multiset, reduced head embedding order-preserved in the parent list).

### Reduced-or pre-minting — `ExpressionAnalyzer::preMintReducedOrs` (I-185)

The single-exclusion rule family (Part 1 of the B8 three-extension design) needs, for every k-ary or with k ≥ 3, the k reduced (k−1)-ary or-operators as genuine registry entries — and the emission runs mid-burst, where registry writes are forbidden ([I-137](../30_invariants.md#i-137)). `preMintReducedOrs` closes that gap at the legal seams: it snapshots the or-category entries of `compiledExpressions` in name order, flattens each to its ordered non-or leaves with `flattenRegistryOrLeaves` (the registry twin of `flattenOrLeaves` — same substitution walk as `isOrIntroInstall`, starting from the entity's own signature), and registers every leaves-minus-one list (u_-renumbered to the canonical first-appearance `u_1..u_m` scheme by `renumberULeaves`) through `findOrMintOrOperator`. Fresh mints join the worklist, so the single-elimination closure lands in one call; `ExecutionParameters::kMaxReducedOrLeaves` asserts on runaway leaf counts. Idempotent; called at `proveKernel` entry, at the tail of `constructOrTheoremsInRun`, and after the phase-4 compaction drain. Consumers resolve a reduced leaf list via the read-only `compiledOrByElements` fence and hard-assert on a miss — the closure is the guarantee behind that assert.

### Shared walk — `ExpressionAnalyzer::constructOrTheoremsFromPairs`

The construction loop lives in [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp)'s `constructOrTheoremsFromPairs`, called from BOTH save branches of [`run_modes.cpp`](../../GL_Quick_VS/GL_Quick/src/run_modes.cpp): after the compressor in the compression path (the proved pool = compression survivors), and directly on the raw batch output in the skip-compression path (incubator / Shortcut batches — the pool = `globalTheoremList` as proved). IDEMPOTENT against the in-run seam: a ledger hit recovers the stored or string and fires only the return vector + parent-subsumption bookkeeping — no re-registration, no reconstruction. Per pair:

1. **Single-direction licensing.** Only the pair's SOURCE theorem must be in the proved pool. Classically `!a → b` is equivalent to `a ∨ b` and the companion `!b → a` is derivable from it, so the companion's own proof is never required — it merely supplies the second disjunct. (The former both-directions membership filter was an engineering artifact of the head-switch grids; a source pruned by compression or retracted as vacuous is still skipped — a defined pipeline state.)
2. **Disjunct-set dedup (post `D-55`).** Canonicalize each `(exist, comp)` pair as `(min, max)` lexicographically. Skip pairs whose canonical form has already produced an OR. Without this, `orPairsFromHeadSwitch`'s symmetric population produces both `or0` (disjuncts in one order) and `or1` (disjuncts swapped) for the same logical OR — pre-fix this is the source of the `or0` / `or1` duplication on Peano.
3. **Construction.** Call `constructOrTheorem(exist, comp)`. Append the result to `globalTheoremList` and `fullTheoremList` with method `or theorem` and references `(exist, comp)`.
4. **Parent-removal (post `D-55`).** Mark `exist` — and `comp` when it is itself a proved row — as **consumed parents**. The compression path drops them from `survivingTheorems` before `saveProvedTheoremsFiltered` writes `theorems.txt` / `compiled_theorems.txt` and appends them to `compressed_out_theorems.txt`; the skip path filters them out of its direct `theorems.txt` append (they stay in `globalTheoremList` so their chapters still render).
5. **Append OR theorems** to the proved-theorem file(s): both files in the compression path; in the skip path the OR rides `globalTheoremList` into the direct save, written through `expandToBaseForm` (no `compiled_theorems.txt` exists in that mode).

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

This applies **only to the `_orint_` (OR-integration) path**. For `_ordis_` see [§3b](#3b-_ordis_-or-disintegration-all-branches-converge) — only the converged expression is collapsed across branches; branches stay live for further convergences until they freeze — a frozen branch (every chain goal reached) keeps everything but no longer feeds request generation; see [Branch freeze](#branch-freeze).

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
- **n-ary or-EXPANSION twins — mirrored polarity defect FIXED (2026-08-07, with the in-run construction seam).** The two expansion builders for a ≥3-element or — the span `expandSignature` OR case (`prover.cpp`) and the verifier's `_build_or_from_elements` (`verifier.py`, edited under explicit [I-16](../30_invariants.md#i-16) consent, plan approval of D-266) — formerly left-nested the or-so-far UN-negated (`¬(D_1∨D_2) ∨ D_3`), byte-mirrored on both sides so the prover-vs-verifier compare passed on the wrong form. Both now nest the or-so-far NEGATED — by double-negation cancellation its `!(&…)` form contributes the bare positive `(&…)`: `!(&(&!D_1!D_2)!D_3)` = `D_1∨D_2∨D_3`. The verifier's inverse `_parse_or_disjuncts` and the test-side heap oracle moved in lockstep; literal-bytes tests on each side (`expand_signature_or_three_disjuncts_nests_negated`; `test_build_or_from_elements_three_disjuncts_nests_negated`, round-trip `test_parse_or_disjuncts_roundtrips_builder`) prevent a shared-error pass from recurring. Made live by in-run construction (3-ary `or5`/`or7` broadcasts); 2-element expansions are byte-unchanged.
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
