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

### Structural introduction from a positive conjunction

`prover.cpp`, `disintegrateExprCoreView`, generates an introduction rule for each ordered OR alternative. A positive compiled conjunction alternative unfolds recursively into successive premises using `reconstructImplicationFullBindScratch`; each signed non-conjunction leaf remains one premise. Head-variable coverage is checked over the whole premise chain. The original compiled alternative remains unchanged for case processing and exclusion; no intermediate positive conjunction statement is required. Nested conjunction cycles and existing capacity limits assert.

The `expansion` origin cites the same compact implication whose definition contains the OR premise. `proof_export/or_introduction.py`, `match_or_introduction`, checks the exact alternative, conjunct order, fixed parameters, complete local binders and binder visibility. The certificate action `structural_or_introduce` records the binary conjunction construction and OR path; Lean introduces the recorded premises, constructs their conjunction and selects that OR alternative explicitly. This is structural implication generation, not a claim that any of its premises is already true. See [I-264](../30_invariants.md#i-264).

### Complete conjunction refutations

An admitted OR fixes every argument of its positive conjunction alternatives. `contraposition.cpp`, `activateDirectOrRefutations`, publishes their prepared direct member-complement rules immediately through the current phase's output mail. `publishConjunctionRefutation` records expansion from the sole OR citation and compilation to each compact; ordinary disintegration installs the rules. Nested positive conjunction members retain their direct structural projections.

Negated conjunction premises of other implications use compiler-prepared guarded support, independently of OR occurrences. For immediate parts `a`, `b`, `c`, the compiler prepares exactly `!a & b & c ->!(a & b & c)`, `a & !b & c ->!(a & b & c)` and `a & b & !c ->!(a & b & c)`. Signed parts are complemented literally. Nested parts stay opaque; helpers produce no further helper or contraposition families. For a unary owning implication, compilation composes each complete guarded helper with that owner. Disintegration installs the direct rules at the owner's scope and retains the original owner; it records the intermediate helper only as a checked proof dependency. Owners with multiple premises retain their ordinary guarded helpers. This is a single compiler pass, not transitive closure.

There is no runtime OR/contrapositive join, descendant replay queue, equality index or matching receipt. Existing compact ownership, ordinary equality rewriting, mail and scope cleanup govern installed support. See [I-259](../30_invariants.md#i-259).


### 1. OR disintegration

**Enclosing existence admission.** An OR encountered inside an existence remains scratch-only until every enclosing witness is admitted. `prover.cpp` / `disintegrateExprCore2` records its raw expression and enclosing witness in the existing collected scratch records; `disintegrateExpr2` applies the ordinary OR consumer only after the witness chain qualifies. Admitting an inner witness coadmits its complete enclosing chain before this check. Rejected chains create no OR statement, source, processed-OR entry, cohort or park. The unconditional OR-opening signal does not override this boundary. Force-deep processing permits only its selected integration-witness chains. See [I-253](../30_invariants.md#i-253).

Only positive logical entities of category `or` enter on-demand OR exclusion and case processing. Negated category-`and` applications use a separate named-predicate rule: a public name generates one flat exclusion implication per conjunct, while generated `and<N>` names stay ordinary negative facts. `isNumberedCompilationName` checks the full category-plus-digits spelling, so a public name that merely starts with `and` is not excluded. Neither kind of negated conjunction registers an OR source or creates a residual, case scope or park.

For a named predicate defined as `P = A and B`, its negation supplies `A => not B` and `B => not A` through ordinary disintegration. These rules remain at the source scope and preserve the expansion/disintegration history in the block and outgoing mail. Ordinary suppression and locality gates still apply. For example, named `strictOrder` expands to preorder and inequality; its negation together with inequality can therefore derive negated preorder. This path does not instantiate a compiled OR exclusion family.

**1a. Exclusion on demand.** An admitted OR records its original expression, scope, ordered flattened leaves, levels, and locality in `orImplications`. `prepareOrImplicationMatches` indexes complementary signed cores and bootstraps all newly watched cores in one scan of `intKnownStatements`. Later fact arrivals and equality merges stage only affected work. `bindOrSourceViews` canonicalizes the whole OR with the existing equality chooser, shares that cached form across its complementary leaf views, and joins exact canonical fact buckets. OR matching opts into standalone equality and inequality canonical views so a refuter receives the same argument substitutions as an equality-valued disjunct. The shared chooser retains its ordinary equality-preservation default for other callers; selected OR refuters use the same opt-in when materializing their `equality1` twins. Descendant equality views are created only where needed; intermediate substitution tokens participate in invalidation.

`evaluateOrImplication` removes currently refuted leaves. One survivor becomes a literal; several survivors become the precompiled residual OR. Every source, selected fact, and applied equality must have comparable scope, and the deepest participating scope receives the internal-mail head. The ordinary implication locality and hypothetical-scope gates still apply. If every leaf is refuted, one selected exclusion produces the opposing literal for the existing contradiction machinery.

`selectOrImplication` chooses refuters and compatible scopes into a bounded stack record of ids and values. It changes no known facts, proof history, mail or source liveness. `materializeOrImplication` consumes that selection immediately, without repeating the match, and retains the ordinary hypothetical-scope gate, rule reconstruction, equality twins, levels, history, delivery and retirement. The `evaluateOrImplication` wrapper preserves the existing timing boundary and calls those two stages consecutively.

Case release retains the exclusion source and its postings. `drainPendingOrReleases` stages only the original cohort's next branch and history; there is no first-release suppression marker. Later complete-conjunction refutations can therefore reduce inherited ORs inside that same case. Source registration, residual continuation and matching reconciliation remain available after release. Exact same-scope canonical duplicates still reconcile through `retireDuplicateOrSources`; source-scope reduction and owning-scope cleanup retain their ordinary retirement. See [I-265](../30_invariants.md#i-265).

A firing at the source scope retires the complete source and its postings. A descendant firing retains the ancestor source for other branches. Residual internal mail carries a sticky source-continuation bit independently of disintegration permission: after the deposit door accepts a known form, that residual becomes a source even when ordinary disintegration is suppressed. Existing known residuals activate the same continuation before source retirement. Refused deposits with no known form do not create sources.

The complete compiled exclusion family remains on `LogicalEntity::implications` for reconstruction. No family compact becomes a known statement and no eager exclusion rule enters hash memory. Only a selected firing creates history: original OR, optional `equality1` bridge, expanded OR, selected compact with `disintegration`, expanded rule with `expansion`, and the head's `implication` row with its selected facts. The unchanged verifier validates these ordinary tags. `drainOrImplications` deduplicates source/scope events within each reconciliation pass. A new canonical fact can requeue a pair already evaluated earlier in the same drain; the drain stops only when staged changes are exhausted. `standardProcessing` drains OR work after canonical watch materialization; the counterexample entry gate feeds completed heads to its existing contradiction check.

**1b. Per-branch case-split — SEQUENCED.** The flattened non-OR leaves form one cohort, but cohort creation mints NO branch scope (D-250, [I-174](../30_invariants.md#i-174)): the bootstrap registers the FULL structural leaf count in `orDisjunctCount` (convergence semantics unchanged), queues EVERY leaf in the deloadable pending queue `Memory::orPendingBranches` (with the cohort's seed level run in `orPendingLevels`), and stages the cohort on `Memory::pendingOrReleases`. The bootstrap runs inside `standardProcessing`'s internal-mail drain, whose step-3 clear would wipe any seed inserted into the channel being drained — so the FIRST release happens at the same end-of-burst drain as every later one. There the leaves are ranked — tier 0: everything that is not a positive equality, by descending token count (most structurally complicated first), byte-lex ties; tier 1: bare equalities; tier 2: equalities carrying an anchor-slot argument, dead last; a positive conjunction whose flattened elements are all positive equalities (e.g. `and27`) takes the highest tier of its elements, one non-equality element keeping it in tier 0 ([D-437](../40_decisions.md#d-437); `pickTopOrDisjunct` / `orDisjunctTier` / `collectAnchorArgs` in [`prover.hpp`](../../GL_Quick_VS/GL_Quick/src/prover.hpp)); a route-(a) cohort's one-shot `orStarterPick` row overrides the FIRST pick with the admitted disjunct (D-252) — and the top-ranked disjunct's scope `"ordis_(outer-or<N>[…])_(d_i)"` is minted with its seed (the asserted disjunct) on `sameIterationInternalMail` plus an `or disintegration` mail origin ([I-176](../30_invariants.md#i-176)) — the next step's absorb runs it through the full kernel pipeline, so a COMPOUND disjunct (an existence form) decomposes like any deposited statement. The original outer signature names every sibling branch and the cohort; no intermediate OR-valued branch exists. The negations of the other leaves are not seeded in `_ordis_`; the original source from 1a remains available for exclusion while the cohort continues its remaining cases. Independent OR sources at other parent scopes retain their own eligibility. Each released branch runs the standard prover machinery in its own scope ([D-218](../40_decisions.md#d-218), [I-166](../30_invariants.md#i-166)).

**Release.** The next pending branch mints only when the live branch RESOLVES: (a) it deposits an expression matching a `toBeProved` goal at a scope on its parent chain up to main — `ordisMerge`'s goal probe stages the cohort on the persistent `Memory::pendingOrReleases` inbox; or (b) it retires refuted — `drainDeadOrBranches` stages the cohort ([D-242](../40_decisions.md#d-242) retirement doubles as the release trigger). The end-of-burst `drainPendingOrReleases` ([`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp), between `drainDeadOrBranches` and the `pendingWipeScopes` drain) re-ranks the REMAINING pending disjuncts (the release order is a pure function of the pending set plus the LB's anchor arguments — re-derived, never stored) and mints at most one branch per staged cohort per burst. An undecided live branch stalls its cohort — a defined degradation, no timeouts, no caps. A resolved branch is not left working either: once every `toBeProved` goal on its chain is known at the branch or above, the end-of-burst `freezeResolvedOrBranches` marks it resolved and queues its subtree for wipeout (see [Branch freeze](#branch-freeze), [D-306](../40_decisions.md#d-306)). In a block without explicit goals or active primed contradiction work the release drain releases nothing. Active primed contradiction blocks retain the ordinary case sequence without synthetic goals ([I-239](../30_invariants.md#i-239)). A re-deposit of the same or statement at the same parent hits the bootstrap guard (an existing `orDisjunctCount` row) and creates no duplicate cohort or eager exclusion rules. This one-live-branch discipline bounds the work after case admission: the expensive assumed-equality branches (equivalence-class replication near main) open only after every productive branch already resolved, and in descent contexts they refute within a few bursts and retire — one bounded pulse, not a standing population.

**Queue rewrite ([D-375](../40_decisions.md#d-375)).** `drainPendingOrReleases` records every released cohort and rewrites `orPendingBranches` / `orPendingLevels` once after its loop — one compaction over the released set, the survivors re-inserted per cohort in release order — instead of one whole-map compaction per release; the resulting key order, runs and level rows are the ones the per-release rewrite produced.

Tagged `or disintegration` in the processed proof graph. Checker: `check_or_disintegration` at [`verifier.py`](../../verifier.py).

**Exclusion sources do not create scopes and remain eligible at every OR depth.** Case opening retains its existing proof-work, depth, demand, and disintegration gates. Refusal of a case gate does not install an exclusion family in hash memory; the independent source supplies exclusion on demand.

**1c. Per-leaf INTRO implications for an or-shaped implication premise** ([D-237](../40_decisions.md)). On-demand exclusion is the elimination half of an OR's flat consumption; the introduction half emits in `disintegrateExprCore2`'s IMPLICATION branch: when an installed compiled implication's premise element is a disjunction (compiled-map probe — implication elements have no instruction-graph entities), the prover also emits, per flattened leaf,

```
(>[bound] d_k (or<N>[…]))
```

— "one true disjunct states the disjunction". The leaves come from a compiled-map twin of `flattenOrLeaves` (per-level signature→instance substitution, recursing into or-category elements only), and leaf + head are fed to the reconstruction in RAW element form so `u_` arguments stay free and only the changeable element variable binds. Propositionally valid unconditionally — no soundness surface. This is what lets a single concretely-true disjunct (typically reached through an equality1 witness alias, since the true disjunct at a concrete element is reflexive) fire the implication's own rule: the rung-2.1 chain `2_copy=2 → or(2_copy) → 2_copy∈M → 2∈M`. Each intro rule carries an `expansion` origin whose dependency is the compact implication statement itself — the same chapter-capable antecedent the main rule's expansion record cites; `_try_expand`'s or-aware implication acceptance covers the shape on the verifier side.

An intro-FIRED or head never re-disintegrates ([D-241](../40_decisions.md)): the intro shape is re-detected at every install by `isOrIntroInstall` (`memory.cpp` — single premise byte-equal to one flattened leaf of an or-category head; shape-based so status-3 mail-recovered reinstalls re-mark), the derived per-rule `LocalMemoryValue::disintegrationAllowed` goes false, and the firing site reads it as the rule-intrinsic half of `FiringRecord::doNotDisintegrate`. The stated disjunction stays a flat fact for ordinary membership-introduction use — an intro-derived or's source leaf is true at the same scope, so a case split over it is redundant by construction (its `_ordis_` cohorts were a pure fan-out multiplier: branch assumptions breed equalities ex falso, equalities re-fire the intro rules). Elimination-fired or heads (premise not a leaf) and status-0 assumed or fuel branch exactly as before.

**Active admission gate: algebraic opening disabled.** With proof work present, `prover.cpp::consumeOrLeavesCohort` opens a cohort only through **route (b)**, disintegration permission, or **route (c)**, a missing whole-compound demand described below. Route (b) carries `LocalMemoryValue::productOfDisintegration` through the firing record and deposit door; its existing suppression checks remain. Algebraic `admissionMap` product-template matches, including `ordisOnly` values, do not authorize cohort creation. This is the maintainer-authorized opening-disable experiment; it introduces no replacement proof-search rule.

**Satisfied-OR gate.** Before either route opens and before the park, `consumeOrLeavesCohort` asks `orDisjunctTrueAt` whether any disjunct already holds at the OR's scope: known there or at an ancestor, literally or up to that scope's complete classes (`elementKnownAtOrAbove`), a reflexive equality, or a positive conjunction whose flattened elements all hold. Such an OR opens no cohort by any route and is not parked, because a disjunct that holds keeps holding (a scope's facts only accumulate and its classes only merge), so a parked satisfied OR could never open ([D-438](../40_decisions.md#d-438)). The branch of a disjunct that already holds is its parent scope, so convergence through it could only conclude what the parent already proves; the one channel not reproduced is the negation a refuted sibling branch would emit at the parent. The route probes run first and keep their own consumption behavior (the D-358 stale-demand erase). The retire-later half, `retireSatisfiedOrCohorts`, runs in the closure sweep after the duplicate-cohort retirement and before the freeze sweep and the release drain: an open cohort whose live disjunct or queued disjunct has come to hold at its parent scope is retired like a duplicate loser (live branch queued for the wipe, scheduling rows erased; frozen branches, bookkeeping and history kept).

The retained route-(a) machinery still constructs and probes disjunct product templates and files parked cohorts in `rejectedMapOrdis`. Algebra key gains can still re-mail a parked OR through `revisitRejectedOrdis`, but its deposit must satisfy route (b) or (c) to open. Ordinary algebra admission and inference are unchanged. Whole-compound demand selects the starter when route (b) is absent; otherwise ordinary ranking applies. Depth, proof-work, duplicate, intro-fired and `doNotDisintegrate` gates remain. The historical route-(a) starter behavior described in §1b is inactive under this opening predicate.



**Dual filing (D-267, ).** The parked cohort ADDITIONALLY files in `HashMemory::rejectedMapOrdis2` under each ELIGIBLE disjunct's clean GROUND text (eligibility = `ordis2KeyEligible`: compiled non-atomic entity — predicates included, `operators` not consulted; arity ≥ `kOrdis2DemandMinArity`; polarity verbatim — a negated compound files WITH its `!`). That second index speaks the SAME key language as the `admissionMapOrdis2` demand map, so the demand drain's wake (`revisitRejectedOrdis2`) is a plain key rendezvous — the pair completes each other the way algebra's `admissionMap`/`rejectedMap` do (whichever side arrives second wakes the other; a route-(c) probe of the demand map at deposit time covers the demand-first order because it precedes the park in the same chain). Only the ordis2 drain wakes the second index; the algebra key-gain seams keep waking only the old map ([I-183](../30_invariants.md#i-183)).

**Shared demand lifetime.** Opening one cohort retains its missing ground demands for every other eligible source, including sources already queued by the same wake. A route-(c) probe removes a demand when its fact is known at that scope or an ancestor; a sibling fact does not satisfy it. The existing parent/source cohort guard prevents duplicate openings. Canonical rekeying and scope cleanup remain unchanged. See [D-393](../40_decisions.md#d-393).

**Or-uniqueness gate (I-219, ).** Upstream of everything in this section sits one more gate, at the deposit door itself: an `(or<N>[…])` deposit whose equality-and-symmetry processing identity already has a fully-processed representative at the deposit's scope is a PASSIVE statement — it registers (door-canonical text) and is multiplied by `applyEquiClasses` like any statement, and nothing else happens: no `disintegrateExpr2`, so no new initial exclusion source, no cohort bootstrap, no park, no dual filing, no `fullyDisintegrated` stamp — on every entry path (local deposits and flag-5 relay arrivals alike). "Fully processed at the scope" is the per-scope `Memory::processedOrLedger` (persistent arena, written when an or-compact deposit reaches `disintegrateExpr2`); each row carries the or's equality-and-symmetry processing id beside its original id (`packInt32Pair(canonical, original)`), re-keyed by `recanonicalizeProcessedOrLedger` at the one class-change seam (`updateEquivalenceClasses`, right after the `changedClassesThisStep` push), so the probe is an exact id membership test on either half — only compiled slot orientation per deposit, and still using the existing equality chooser (I-217's walk computes the canonical half; [D-335](../40_decisions.md#d-335)); `checkForEquivalence`'s whole-arg Cartesian probe stays active behind the gate for the ors it does not suppress. Every path that intentionally re-runs an or consumption clears the ledger row first: the parked-or un-know (`resetParkedOrStatementRegistries`) and the two ordis park hooks' re-mail loops; the disproof cleanup sweeps rows at wiped scopes. Why: or proliferation was self-feeding — every equi spelling reaching the door got the full per-statement consumption, minting statements and equalities that minted more spellings.

**Duplicate-cohort retirement ([D-331](../40_decisions.md#d-331)).** The gate stops new duplicates; cohorts opened BEFORE their signatures became class-equal are collapsed at the end-of-burst `retireDuplicateOrCohorts` drain (between the dead-branch retirement and the freeze sweep): ledger entries grouped by their canonical half (current, since the ledger is re-keyed at every class change), a group with two or more OPEN cohorts keeps the one with the most statements under its `_ordis_` branch scopes (lex tie-break) and retires the rest — live branches wiped in the same burst, scheduling rows erased, frozen branches and history kept, the losers' work re-derived under the keeper when needed. Parked-only duplicates need no drain: the ordis hooks re-mail them and the door's gate makes the re-file passive.

**Compiled two-slot symmetry ([D-385](../40_decisions.md#d-385)).** `compilation_state.cpp::compileOrSymmetry` identifies exactly two digit-valued signature slots from definition types, flattens positive OR nesting, and compares signed leaf multisets before and after their exchange. Equality orientation is normalized; other predicates retain argument order, free slots stay fixed and disjunct multiplicity remains. The derived `LogicalEntity::symmetricDigitSlots` is refreshed at OR compilation and the pre-burst registry seam, including binary loads. The saved D2 totality and trichotomy satisfy this check.

`prover.cpp::canonicalOrProcessingForm` chooses the lexicographically smaller validated orientation after the existing equality chooser. This is only a processing key: the keeper's original carrier, ordered leaves, branch signature and proof history stay unchanged. The deposit gate and nested real OR consumption share the ledger; duplicate spellings remain passive facts for ordinary rules. `or_implications.cpp::retireDuplicateOrSources` reconciles sources before exclusion matching when source or class state changes, including residual continuations. Exact source scope and processing id define a group. An active source of the largest open cohort wins (the same signature tie-break as cohort retirement); without an open cohort, first source order wins. Retirement uses the existing source/posting removal, transferring no leaves, levels, locality or history. The source's existing canonical-id field carries this processing id and follows its existing clone/deload lifecycle.


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

Used when an OR-elimination case split has produced the same conclusion independently in every branch. `ordisMerge` observes ordinary statement-registration deposits, staged and canonicalized equivalence products, the equivalence-pass product loop, and reduced-count retirement rechecks. Every call uses the deposit's actual scope, including descendant-scope rewrites; equal signatures under distinct parents remain distinct cohorts.

**Equality-supported convergence.** The parent conclusion remains fixed when different cases store different canonical forms. `prover.cpp`, `materializeWatchHits`, compares each relevant ancestor goal with visible facts under the live case's complete equality context. `bookOrEqualitySupport` records the actual source, original citation scopes, applied positive equality evidence and combined levels; it adds no target statement. `ordisMerge` books that support under the target's existing expression/cohort key. In a schematic endpoint case, stored E(n) and local i=n contribute to parent target E(i), while the old-index case contributes its literal E(i). No equality context crosses a sibling boundary.

`orBranchGoalsResolved` accepts an exact known goal or booked equality support, so case release and freezing share the same evidence. The support tables and saved levels survive the case's own wipe and deload; closing the cohort parent clears them. Facts-only clones start without convergence support. Extended `or convergence` rows cite the actual source facts and positive equalities at their exact scopes, with duplicate pairs removed. The checker constructs each case's substitution plan independently, and Lean enters the case before applying its equalities. Legacy exact-expression rows keep their existing layout. See [I-263](../30_invariants.md#i-263).

`ordisMerge` records each per-branch deposit in `Memory::orBookkeeping`. The cohort is the exact `(parentValidity, originalOrSignature)` pair: both component ids are packed and interned as one `cohortId` by `mintOrCohortId`. Bookkeeping is therefore keyed by packed `(exprId, cohortId)` `lbStateInterner` pairs and valued as decoded-lex ordered branch-disjunct id sets (`D-135`; the decoded order feeds the origin rows and cleanup loop). For a nested OR, the registered structural count is the number of recursively flattened non-OR leaves, while the signature component remains the original outer OR. When the recorded branch count for that exact `(expr, parentValidity, originalOrSignature)` reaches `Memory::orDisjunctCount[cohortId]` (registered at OR-disintegration mint time, see [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp)), the convergence path fires. Equal OR signatures at different stack positions occupy independent rows ([D-218](../40_decisions.md#d-218), [D-219](../40_decisions.md#d-219), [I-167](../30_invariants.md#i-167)).

1. **Promotion through internal output mail.** During input absorption, `standardProcessing` passes its `internalMailOut` through `addExprToMemoryBlock` to `ordisMerge`; the parent statement and convergence history both use that channel. Pre-burst output is same-iteration and post-burst output is next-iteration. Calls outside absorption retain the ordinary same-iteration destination. The matching subsequent drain registers the parent statement through status 1 and ordinary goal discharge. The existing snapshot/clear order is preserved, and no reentrant deposit is needed. See [D-353](../40_decisions.md#d-353).
2. **Per-branch cleanup of the converged expression.** For each branch validity in the recorded disjunct set, `ordisMerge` calls `removeExpressionFromMemoryBlock(EncodedExpression(addExpression, branchValidity), mb, /*state=*/0)` — the standard removal procedure. Before each removal the branch derivation's history rows are shipped to `mailOut` (`copyOriginRowsToMailOut`): the removal erases the branch row from the per-step delta `fillMailOut` walks, and the last-released branch converges in the step it derives, so a descendant walking the mailed `or convergence` row would otherwise never receive that branch's derivation ([D-311](../40_decisions.md#d-311)). The N per-branch copies of the converged expression collapse into the single parent-scope copy. The parent-scope copy remains visible to each branch via comparable-scope inheritance, so branches can reuse it. Branches keep all their *other* facts and stay live — until they freeze (see [Branch freeze](#branch-freeze): a resolved branch is queued for runtime subtree wipeout while its booked contributions and history remain) — the `_orint_` path instead wipes siblings as soon as a winning branch proves the OR (see §3a). This single-expression cleanup retains `exprOriginMap`, `intStatementLevelsMap`, and `intKnownStatements`. The later successful subtree wipe removes the ordinary statement and level rows, preserves the proof history and saved current-goal levels, and rejects further deposits through the permanent scope filter.

#### Dead-branch retirement

A branch whose **asserted disjunct is refuted** — the disjunct's negation is `known` at the branch scope or one of its ancestors — is an ex-falso factory: its equality-class merges and variant rewrites breed unboundedly under an inconsistent premise, and its cohort's convergence count can never be reached. Such a branch is retired ([D-242](../40_decisions.md#d-242), [I-172](../30_invariants.md#i-172)):

- **Detection** has two probes running the same predicate. The original sits inside `ordisMerge` (so both deposit paths reach it): on each `Disintegration`-classified deposit, the branch payload's wrapped disjunct is negated and probed non-minting against `intKnownStatements` up the `parentOf` chain. The second is `refutedOrBranchAtOrAbove` ([`prover.hpp`](../../GL_Quick_VS/GL_Quick/src/prover.hpp)), consulted by `applyEquiClasses` Pass 1 before back-applying a delta class — it walks the class validity's ancestor chain, runs the identical probe at each `_ordis_` disintegration scope, and doubles as the equi-apply's dead-scope skip so a refuted branch's classes are never ground across the registry in the burst that would wipe them ([D-262](../40_decisions.md#d-262), [I-181](../30_invariants.md#i-181)). Either probe's hit stages the branch vid on `Memory::pendingDeadOrBranches` (persistent-arena pod set, the `pendingDisprovedGoals` pattern). `known` is the sound bit — the record is immortal ([I-58](../30_invariants.md#i-58)) and a scope's fact set is monotone.
- **Retirement** runs in [`drainDeadOrBranches`](../../GL_Quick_VS/GL_Quick/src/prover.cpp), in the shared standard-processing tail in both phases, after `drainDisprovedGoals` and before the `pendingWipeScopes` drain: the branch subtree is queued for the radical wipe and inserted into `intValidityNamesToFilter` (permanent — a filtered branch never receives a deposit again, so it never re-stages); the cohort's `orDisjunctCount` drops in place (zero survivors retires the cohort wholesale — the count row leaves the container, every run empties, and the joint contradiction surfaces through an active parent-scope OR exclusion source or the last case's ordinary convergence, followed by the contradiction machinery; cohort identity — the `(parentValidity, originalOrSignature)` pair, [I-167](../30_invariants.md#i-167) — is unchanged, no successor cohort is minted); every `orBookkeeping` run of the cohort loses the dead branches' disjunct entries **before any count comparison** — a stale dead entry against the reduced count could fire a convergence no surviving branch ever derived; an only-dead run leaves the container. Then convergence is re-checked: a surviving row whose run reaches the reduced count re-fires through `ordisMerge` itself (a missing levels row at the survivor branch means that expression converged earlier — a defined skip).
- **History is kept** ([I-44](../30_invariants.md#i-44)): earlier full-cohort convergences cite dead-branch derivations as `or convergence` ingredients, and the chapter walker must still resolve them. A staged branch whose cohort has no `orDisjunctCount` row was retired wholesale by the same seam's disproof drain — a defined skip.
- **The retirement is artifact-visible** ([D-238](../40_decisions.md#d-238)): the drain records each retired disjunct plus the shallowest scope where its refutation is `known` (`Memory::orRetiredDisjuncts`, persistent arena), and every later convergence of the shrunk cohort emits, per retired branch, the reductio ingredient `(negate(D_i), scope)` alongside the survivors' branch derivations — the exported row accounts for all K disjuncts and `check_or_convergence` validates both entry forms.

History: pre-[D-34](../40_decisions.md#d-34) the bookkeeping path was `trackOrBookkeeping` inside `addStatement`. It was structurally blind to D-33's cross-scope deposits and carried a too-aggressive cleanup (wiped every statement at the branch scopes via prefix-match, plus added the branch validities to `validityNamesToFilter` to block all future inserts). FTA-rung-1 §9a / §9b symptoms: the per-branch preorder facts produced by D-33's descendant-direction rewrite landed at the branch scopes (`current_proof_state.md` confirms all four cells), but `trackOrBookkeeping` never saw them and convergence never fired. D-34 fixes both — the kernel-level placement closes the visibility gap, and the cleanup is narrowed to just the converged expression so branches stay alive for further convergences (e.g. the second preorder of the same impl scope's body).

#### Phase 3 inference exclusion

At Phase 3 entry, after the worker claims and loads the block, `collectPhase3IgnoredOrRoots` selects every existing OR-disintegration scope whose branch-inclusive chain goals are all known; an empty chain-goal set qualifies unless this is an active primed contradiction block. The NameMap scan includes queued seeds not yet recorded in `orLiveBranches`. Sibling and descendant goals do not participate in this decision. The root set remains fixed for the phase, and `Memory::phase3IgnoresScope` covers descendants by walking the current NameMap ancestry, including descendants minted later.

The worker binds this arena-backed context only around inference: mail statements, deposit doors, equivalence-class consumers, admission/rejection rewrites, canonicalization, hypothesis reactions and scoped goal processing. History absorption and outgoing history still run, as do parent-scope operations. The shared standard-processing tail clears the context after goal sanitization and before lifecycle drains; the existing branch-release and wipe order remains. Every live branch selected for exclusion is asserted to have been wiped before the phase finishes. No context state is deloaded, and other logical blocks and phases have no active binding.

See [I-232](../30_invariants.md#i-232), [D-359](../40_decisions.md#d-359), and the [task ledger](../ordis_ignore_plan.md).

#### Branch freeze

Resolved branches now undergo deferred subtree wipeout; `frozenOrBranches` remains the one-way resolved marker ([I-206](../30_invariants.md#i-206), [D-347](../40_decisions.md#d-347)).

- **Predicate.** Every current `intToBeProved` goal at the branch itself or one of its ancestors through main must be known at the branch or above. The shared read-only predicate is `ExpressionAnalyzer::orBranchGoalsResolved`. Unrelated goals are ignored; zero relevant goals qualifies. The successor-release trigger remains the first reached goal, while wipeout requires all relevant goals.
- **Timing.** `freezeResolvedOrBranches` evaluates the live registry after dead-branch and duplicate-cohort retirement, before pending sibling release. Each resolved root enters both `frozenOrBranches` and `pendingWipeScopes`, then leaves `orLiveBranches`. The existing decoded-name-ordered wipe drain removes the union of queued subtrees later in the same call through `Memory::wipeSubtrees`, preserving the former per-root filter insertion order.
- **Effect.** `Memory::wipeSubtree` erases scoped runtime statements, statement levels, classes, local goals, admission and rule state and permanently filters the subtree. The parent-chain goals remain until ordinary convergence deposits close them. Successful closure never decrements the cohort count or records a refuted disjunct.
- **Preserved evidence.** Cohort identity and counts, `orBookkeeping`, and origin history survive. Ordinary later-sibling convergence consumes the retained branch contributions and ships their origin rows. `standardProcessing` fills outgoing history before its scope-cleanup tail. Phase-1 cleanup completes before GPU projection and request construction.
- **Goal derivation levels.** Before wipeout, `freezeResolvedOrBranches` copies only the levels of current chain goals booked at that branch into `LbMemory::closedOrGoalLevels`, keyed by `(expression NameId, branch validity NameId)`. Reduced-count convergence uses the saved run only while that expression remains a goal on the cohort's parent chain. The usual first surviving branch and its exact levels are preserved; non-goal levels are not retained. This map participates in deload, clears at discharge, and loses entries when their cohort parent is wiped. Full-workload validation is tracked in [`ordis_wipeout_closure_plan.md`](../ordis_wipeout_closure_plan.md).

---

## `classifyOrScope` — the scope-role oracle

Defined at [`prover.hpp`](../../GL_Quick_VS/GL_Quick/src/prover.hpp). Returns one of:

- `NotOrScope` — scope is not related to an OR split.
- `Integration` — scope is part of an OR re-integration (reformulation-for-integration over an OR head).
- `Disintegration` — scope is one of the per-disjunct branches.

Classification is based on the payload prefix of the scope's name. See [`20_core_concepts/04_validity_stack.md`](04_validity_stack.md). The elementary-step phase helpers use it to decide OR-aware bookkeeping; `ordisMerge` uses it to detect `_ordis_` convergence and branch refutation, routing both statements and their history through the caller-selected output during absorption, or `sameIterationInternalMail` otherwise; the post-`addStatement` block in `addExprToMemoryBlock` uses it for OR-integration emission and bare-implication republish; and `cleanUpOrIntegrationBranches` uses it to find sibling integration branches to wipe after a winner emerges.

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

The compiled family contains every size-j exclusion for 1 <= j <= k-2, ordered by size then lexicographic excluded positions. `evaluateOrImplication` selects the entry for the currently refuted subset and instantiates only that history path. The residual head preserves survivor order and uses a pre-minted reduced OR. It carries no route-(b) opening signal; ordinary disintegration suppression and demand-driven parking still apply. Its independent source-continuation signal preserves subsequent exclusion. `_check_or_subset_exclusion` in the unchanged verifier checks the selected premise multiset and reduced-head embedding.

### Reduced-or pre-minting — `ExpressionAnalyzer::preMintReducedOrs` (I-185)

The single-exclusion rule family (Part 1 of the B8 three-extension design) needs, for every k-ary or with k ≥ 3, the k reduced (k−1)-ary or-operators as genuine registry entries — and the emission runs mid-burst, where registry writes are forbidden ([I-137](../30_invariants.md#i-137)). `preMintReducedOrs` closes that gap at the legal seams: it snapshots the or-category entries of `compiledExpressions` in name order, flattens each to its ordered non-or leaves with `flattenRegistryOrLeaves` (the registry twin of `flattenOrLeaves` — same substitution walk as `isOrIntroInstall`, starting from the entity's own signature), and registers every leaves-minus-one list (u_-renumbered to the canonical first-appearance `u_1..u_m` scheme by `renumberULeaves`) through `findOrMintOrOperator`. Fresh mints join the worklist, so the single-elimination closure lands in one call; `ExecutionParameters::kMaxReducedOrLeaves` asserts on runaway leaf counts. Idempotent; called at `proveKernel` entry, at the tail of `constructOrTheoremsInRun`, and after the phase-4 compaction drain. Consumers resolve a reduced leaf list via the read-only `compiledOrByElements` fence and hard-assert on a miss — the closure is the guarantee behind that assert.

### Shared walk — `ExpressionAnalyzer::constructOrTheoremsFromPairs`

The construction loop lives in [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp)'s `constructOrTheoremsFromPairs`, called from BOTH save branches of [`run_modes.cpp`](../../GL_Quick_VS/GL_Quick/src/run_modes.cpp): after the compressor in the compression path (the proved pool = compression survivors), and directly on the raw batch output in the skip-compression path (incubator / Shortcut batches — the pool = `globalTheoremList` as proved). IDEMPOTENT against the in-run seam: a ledger hit recovers the stored or string and fires only the return vector + parent-subsumption bookkeeping — no re-registration, no reconstruction. Per pair:

1. **Single-direction licensing.** Only the pair's SOURCE theorem must be in the proved pool. Classically `!a → b` is equivalent to `a ∨ b` and the companion `!b → a` is derivable from it, so the companion's own proof is never required — it merely supplies the second disjunct. (The former both-directions membership filter was an engineering artifact of the head-switch grids; a source pruned by compression or retracted as vacuous is still skipped — a defined pipeline state.)
2. **Disjunct-set dedup (post `D-55`).** Canonicalize each `(exist, comp)` pair as `(min, max)` lexicographically. Skip pairs whose canonical form has already produced an OR. Without this, `orPairsFromHeadSwitch`'s symmetric population produces both `or0` (disjuncts in one order) and `or1` (disjuncts swapped) for the same logical OR — pre-fix this is the source of the `or0` / `or1` duplication on Peano.
3. **Construction.** Call `constructOrTheorem(exist, comp)`. Append the result to `globalTheoremList` and `fullTheoremList` with method `or theorem` and references `(exist, comp)`.
4. **Parent-removal (post `D-55`).** Mark `exist` — and `comp` when it is itself a proved row — as **consumed parents**. The compression path drops them from `survivingTheorems` before `saveProvedTheoremsFiltered` writes `theorems.txt` / `compiled_theorems.txt` and appends them to `compressed_out_theorems.txt`; the skip path filters them out of its direct `theorems.txt` append (they stay in `globalTheoremList` so their chapters still render).
5. **Append OR theorems** to the proved-theorem file(s): both files in the compression path; in the skip path the OR rides `globalTheoremList` into the direct save, written through `expandToBaseForm` (no `compiled_theorems.txt` exists in that mode).

The parents are subsumed by the OR via on-demand OR exclusion (§1a above). They survive only as compressed-out artefacts — recoverable from the OR + the [`disintegrateExprCore2`](../../GL_Quick_VS/GL_Quick/src/prover.cpp) machinery, no longer cited as standalone theorems. Pre-fix the parents stayed in `theorems.txt` and downstream chapters (notably FTA-rung-1's `1209_single_chapter_proof.txt`) cited them directly, shadowing the OR theorem they had become equivalent to.

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

This applies **only to the `_orint_` (OR-integration) path**. For `_ordis_` see [§3b](#3b-_ordis_-or-disintegration-all-branches-converge) — only the converged expression is collapsed across branches; branches stay live for further convergences until every current chain goal is reached, when the branch runtime subtree is wiped while booked contributions and history remain; see [Branch freeze](#branch-freeze).

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

### Convergence during input absorption

`standardProcessing` supplies its existing `internalMailOut` through `addExprToMemoryBlock` to `ordisMerge`. A conclusion completed while input mail is being absorbed and its `or convergence` history are written together to that output channel: same-iteration in the pre-burst call, next-iteration in the post-burst call. The input snapshot and its later clear therefore cannot discard the new parent proof. Calls outside absorption retain their same-iteration destination. Cohort counting, branch-history retention, level accounting and refutation routing are unchanged. See [D-353](../40_decisions.md#d-353).

### Ground demand from integration rules

Integration-installed rules use the same ground demand/park rendezvous as other rules, with six non-anchor premises and preserved integration justification. Their demand firing bypasses the purity gate but retains grounding and iteration limits. A same-scope or ancestor-known fact suppresses demand creation; a stale demand is consumed at route-(c) opening without opening that cohort. Sibling scopes never satisfy this test. No generic integration-template probe opens cohorts. See [D-354](../40_decisions.md#d-354).


## Weaknesses

### Known & tracked

- **Shared-demand loss at opening — corrected.** Opening a cohort retains the ground demand until a fact satisfies it at the demand scope or an ancestor. Direct regressions cover multiple queued and fresh sources, both arrival orders, duplicate deposits and sibling-scope isolation. Whether this correction completes every remaining step of a particular goal is determined separately by its proof run. 

- **Dead-stage omission from phase-3 exclusion.** `prover.cpp::collectPhase3IgnoredOrRoots` selects resolved-goal roots but does not consult `Memory::pendingDeadOrBranches`; `Memory::phase3IgnoresScope` uses only that fixed selection. A multiplication-cancellation trap observes the parent refutation consumed in phase 1 and the branch already marked dead at phase-1 exit, followed by 16,564 branch-door entries during phase-3 absorption before the tail wipe. The later `sameIterationInternalMail` copy is not evidence of delayed first consumption: absorption-time `prover.hpp::ordisMerge` also emitted the refutation through the explicit `nextIterationInternalMail` output, which phase 1 consumed. The sacred snapshot does not expose that next-iteration channel. This is a confirmed performance weakness in exclusion of an already-staged dead branch; any suppression change still requires proof/provenance review. 

- **Persistent exclusion and residual cohorts.** A late refutation after case release may produce a residual while the original cohort continues. Both use the existing admission, case-depth and duplicate-cohort rules. Source-scope reduction retires the source; descendant reduction retains it for other cases. Direct tests cover both first-case choices, late residuals, canonical aliases, original cohort sequencing and parent contradiction discharge. Full proof preservation and runtime comparisons remain acceptance gates for each implementation stage.

- **Algebraic opening requires the accepted bridge input.** The earlier input without the bridge lost totality and its D2 consequences after algebraic case opening was disabled. The accepted bridge restores all original statements, including both D2 conclusions, with checked proof export. Product-template demand alone remains insufficient to open cases. 

- **Numbered negated conjunctions remain opaque.** Their former exclusion implications are absent; the named-predicate exception applies only to public names. An existing positive OR cannot reduce until a complementary fact is known, and case admission remains independently gated. 


- **`or_slows_down` regression.** Memory flagged; status ambiguous post-migration. A measurement-worthy open question.
- **OR conjecture curation is manual.** `or_pairs.txt` is hand-maintained. A future FTA rung that needs a new OR shape requires editing this file. No automation.
- **Reduced-cohort `or convergence` rows — closed by [D-238](../40_decisions.md#d-238).** The activation run exported the first reduced-cohort convergence row and `check_or_convergence` rightly rejected it (one entry for a three-disjunct OR). The producer now records each retired disjunct with its refutation scope (`Memory::orRetiredDisjuncts`) and the convergence history cites the reductio ingredient per retired branch; the checker (maintainer-consented I-16 change) accepts the two entry forms with exactly-once disjunct coverage.
- **`or branch proven` and `or branch assumption` are first-class verifier tags (D-35, 2026-05-03).** Previously claimed retired/overridden — incorrectly: no override path existed and the prover always emitted the tags live. FTA-rung-1 chapter `1209_single_chapter_proof.txt` carries both. The verifier now registers `check_or_branch_proven` / `check_or_branch_assumption` in `TAG_CHECKERS`, validating: parent/branch scope ancestry, GL-binary disjunct membership (modulo equality symmetry), and the branch-payload `_boundary_orint_<or>_(<disjunct>)` substring pattern. See [`08_proof_tags.md`](08_proof_tags.md#or-branch-proven) for the full row layouts.

### Suspected fragility

- **Inherited OR admission is covered by the recipient gate.** Positive compiled ORs arriving by ordinary ancestor mail enter normal disintegration when the recipient has explicit goals or active primed contradiction work. The original D2 barrier and saved evidence remain in [conta_or_plan.md](../conta_or_plan.md). 

- **`or convergence` spec'd row layout — gap closed by D-36.** `ordisMerge` (`prover.hpp`) now emits the spec'd layout `<C> <parent> or convergence <OR> <parent> <C> <branch_D1> <C> <branch_D2> …`. The branch derivations of `C` remain in `exprOriginMap` (untouched by `removeExpressionFromMemoryBlock(state=0)`); `buildStack` renders them automatically by recursion. Side-effect of the fix: 4 previously-hidden `_ordis_` per-branch case-split rows surface in chapter export under `or disintegration` tag — they pass the D-36-extended `check_or_disintegration`. Net incubator-verifier state: 0 failures.

- **Sibling-wipe atomicity.** `cleanUpOrIntegrationBranches` walks and wipes. If an LB were processed in the same cycle its wipe touches, the effect on that LB's iteration state would be ill-defined — but cross-LB writes during the parallel phase are forbidden ([I-28](../30_invariants.md#i-28)), so a worker thread never wipes an LB another worker is processing; the wipe runs in the single-threaded post-join drain.
- **Branch-seed distinctness.** Each `_ordis_` branch is seeded only with its selected disjunct. If two disjuncts are textually identical (malformed OR conjecture), two branch scopes therefore carry identical asserted seeds and can perform duplicate work. Not currently observed but not excluded.
- **Nested OR branch opacity — closed in both directions.** `flattenOrLeaves` turns the whole contiguous OR tree into one ordered leaf cohort before `_ordis_` or `_orint_` production. The outer signature remains the only namespace/provenance identity, and verifier reconstruction mirrors the same recursion ([D-218](../40_decisions.md#d-218), [D-220](../40_decisions.md#d-220)).
- **n-ary or-EXPANSION twins — mirrored polarity defect FIXED (2026-08-07, with the in-run construction seam).** The two expansion builders for a ≥3-element or — the span `expandSignature` OR case (`prover.cpp`) and the verifier's `_build_or_from_elements` (`verifier.py`, edited under explicit [I-16](../30_invariants.md#i-16) consent, plan approval of D-266) — formerly left-nested the or-so-far UN-negated (`¬(D_1∨D_2) ∨ D_3`), byte-mirrored on both sides so the prover-vs-verifier compare passed on the wrong form. Both now nest the or-so-far NEGATED — by double-negation cancellation its `!(&…)` form contributes the bare positive `(&…)`: `!(&(&!D_1!D_2)!D_3)` = `D_1∨D_2∨D_3`. The verifier's inverse `_parse_or_disjuncts` and the test-side heap oracle moved in lockstep; literal-bytes tests on each side (`expand_signature_or_three_disjuncts_nests_negated`; `test_build_or_from_elements_three_disjuncts_nests_negated`, round-trip `test_parse_or_disjuncts_roundtrips_builder`) prevent a shared-error pass from recurring. Made live by in-run construction (3-ary `or5`/`or7` broadcasts); 2-element expansions are byte-unchanged.
- **Depth-cost of `comparable`.** OR branching creates scope-tree depth. `NameMap::comparable` walks the parent-pointer forest; deeper scope chains increase traversal work.
- **Cross-stack same-orSignature race — closed by D-219.** The old `Memory::orBookkeeping[(expr, orSignature)]` and `orDisjunctCount[orSignature]` conflated equal signatures at different stack positions. The observed rung-1 impl24/top-level collision happened to be absorbed later by Site F's ancestor dedup, but mixed cohorts could still reach the structural count. Both containers now share a parent-scoped `cohortId`; the trace prints `parent` and `orSig`, and the direct test holds one parent at one seen branch while completing the other. See [D-219](../40_decisions.md#d-219).

### Not exercised by tests

- **OR-re-integration path.** The `_orint_` payload scope kind exists but is sparsely exercised by the current Peano+Gauss corpus. FTA-ladder rungs may exercise it more; a regression in OR-re-integration could land silently before the ladder climbs.
- **`classifyOrScope` misclassification.** If a payload prefix silently drifts, classification returns the wrong enum. No test verifies the mapping.

---

## Open questions

- **OPEN-1 — contradictory-branch cohort rewrite — RESOLVED (2026-07-30, ).** Full contradiction discharge remains deliberately main/main only ([D-222](../40_decisions.md#d-222)). The branch-level consequence is now implemented as [Dead-branch retirement](#dead-branch-retirement): a branch whose asserted disjunct is refuted is wiped and filtered, its cohort's `orDisjunctCount` drops in place (no successor cohort — cohort identity unchanged, [I-167](../30_invariants.md#i-167)), booked runs are cleaned of dead entries, and convergence re-checks over the survivors. One survivor is the ordinary reduced-count case; zero survivors retires the cohort wholesale, and the joint contradiction surfaces through parent-scope OR exclusion and the contradiction machinery — primed reductio discharge; the vacuous-premise flag with global-theorem reversion for a normal LB at main; or, for a cohort under a deeper hypothetical scope of a normal LB, nothing at the LB level (the refuted disjunction only proves that scope's assumption unsatisfiable — the enclosing clause is vacuously true and the LB's theorems stay valid). The rejection proof needs no new artifact: detection requires the refutation to already be `known` at an ancestor, so the reductio product pre-exists with its own history. The rung-2.1 ES3 numeral cohorts supplied the acceptance case ([D-242](../40_decisions.md#d-242)).

---

## See also

- [`20_core_concepts/04_validity_stack.md`](04_validity_stack.md) — scope mechanics.
- [`10_pipeline/04_prover.md`](../10_pipeline/04_prover.md) — `cleanUpOrIntegrationBranches`, `classifyOrScope`, contradiction discharge.
- [`20_core_concepts/08_proof_tags.md`](08_proof_tags.md) — `or disintegration`, `or convergence`, `or theorem`, `variable copy`.
-,,.


**Inherited positive ORs.** Ordinary status-3 ancestor mail also enters disintegration for a positive compiled `or<N>` when `hasOrProofWork` holds at the recipient. The arrival keeps its external classification; finalized ORs register on-demand exclusion sources. Normal demand, depth, duplicate and suppression checks still control case admission, and both OR parking indexes support later demand. The expansion history of each recovered rule cites its own compact implication at the rule scope, preserving the OR expansion and disintegration intermediates. No new mail flag is required. See [I-240](../30_invariants.md#i-240) and [D-367](../40_decisions.md#d-367).

### Active contradiction blocks and case resolution

`ExpressionAnalyzer::hasOrProofWork` in `prover.cpp` recognizes either explicit goals or an active `primedForContradiction` block. The same predicate controls opening, parking and pending-case release. No new configuration field or stored goal is introduced.

`orBranchGoalsResolved` leaves an empty relevant goal set unresolved in an active primed block. The Phase 3 entry exclusion and end-of-burst freeze both use this predicate, including branches whose seeds are queued. When real relevant goals exist, the ordinary all-known rule remains. Ordinary goal-less blocks retain vacuous resolution.

A registered pair of opposite facts at a case and its ancestors refutes that case through `ordisMerge`. The refutation and its two-dependency history use the same caller-selected output channel as convergence, so clearing an absorbed input cannot lose them. Retirement, next-case release, reduced-cohort convergence and subtree cleanup remain ordinary. A branch contradiction alone never closes the whole block: `dischargeContradiction` still requires a main/main pair. See [I-239](../30_invariants.md#i-239).

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
