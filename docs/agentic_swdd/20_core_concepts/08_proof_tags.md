<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Core concept — Proof tags (full reference) `[DRAFT]`

> Each row in a processed proof-graph chapter carries a **tag** identifying the kind of inference step it represents. The tag determines which verifier checker validates the row and what rest-field shape the row has. This chapter is the single-source-of-truth per-tag reference.

---

## Row shape recap

### Structural conjunction-refutation expansion

A single-member named alias unfolds to its member without adding a projection index. The checker still requires the encountered exact whole expression and complete substitution; an empty member path is valid only after at least one positive definition has unfolded.

Each published compact carries its `compilation` origin in internal mail as well as block and outgoing history. The ordinary absorb door requires that paired origin before installing the compact; structural expansion rows alone do not satisfy the mailed-statement contract.

Ordinary OR exclusion can conclude a compact residual OR. `lean.py`, `_negated_conjunction_projection`, follows only that residual's positive OR definitions and constructs each exact alternative with explicit `Or.inl`/`Or.inr`. Under the residual's counterexample these give its leaf complements, including classical double-negation elimination for a negative alternative. Every original source-conjunction position must be filled exactly once by those complements or the recorded premises. Missing, extra, wrongly signed or differently substituted components still assert; no proof-search fallback is used.

OR expansion replay applies the explicit De Morgan equivalence at every binary node of the definition's left-associated tree, including definitions with more than two alternatives. Double-negation cancellation preserves each alternative's literal polarity. This matches the native complete negated-conjunction expansion used before subset exclusion.

An `expansion` row can publish a fixed direct helper from the exact positive OR exposing its conjunction. Its conclusion has no local binders. Positive conjunction descendants retain explicit projection paths. Saved indirect rows with an additional expanded contrapositive remain checkable, but the runtime no longer generates them.

A guarded supporting helper instead cites exactly one expanded owning implication with the matching negated conjunction premise. `proof_export/conjunction_refutation.py`, `build_guarded_support_plan`, requires all immediate parts as guards, exactly one complemented signed part, complete and bijective binder correspondence, unchanged fixed arguments and exact repeated ports. Nested parts stay opaque. Existing source-presence and comparable-scope checks remain active.

`certificate.py`, `_chapter_certificate`, records `conjunction_refute` without redefining the source. `lean.py`, `_render_chapter_steps`, introduces helper binders and guards in their actual premise-chain order, assumes the whole conjunction, unfolds its immediate definition and contradicts the selected projection. Direct OR helpers retain their explicit positive descendant paths. For a unary owning implication, the guarded rule is a proof-only intermediate: an ordinary `implication composition` row cites that rule and the original owner to establish the direct final-head rule. Only the direct rule's compact is installed; the original owner is retained for independently available negative conjunctions. The existing strict composition checker and Lean replay validate complete binder correspondence and exact scopes. No new tag or verifier exemption is used. Ordinary compilation and implication firing rows keep their normal contracts.


Every chapter row is tab-separated:

```text
expression \t namespace \t tag \t [rest fields...]
```

Rest fields **alternate** `(expression, namespace)` pairs — i.e. `rest[0]` is an expression, `rest[1]` is its namespace, `rest[2]` is the next expression, `rest[3]` its namespace, and so on. The verifier reads them at `rest[i], rest[i+1]` strides (see `verifier.py::check_equality1` for a canonical consumer). Ignoring this alternation when authoring a new origin — e.g. writing `rest=[src, eq1, eq2]` without the interleaved `ns` fields — makes the verifier silently reject every emission with that tag (odd `rest` count → invalid shape). Every tag's checker indexes the rest fields in pairs; the per-tag sections below state which specific `(expr, ns)` slot encodes which role.

---

## Tag index

37 `TAG_CHECKERS` registry entries. `implication grouping` and `contraposition` share the strict auxiliary checker; each retains its own proof contract and counter. `equalize variable` is a retired alias, while `multiplied from` remains active. The verifier iterates in registry order; see [`10_pipeline/08_verifier.md`](../10_pipeline/08_verifier.md).

- [`anchor handling`](#anchor-handling) — pin raw bound-variable index to anchor-slot name.
- [`compilation`](#compilation) — implication compiled to its compact `(implication<N>[…])` named form (ASIC-prep provenance).
- [`contradiction`](#contradiction) — proof by contradiction.
- [`disintegration`](#disintegration) — split compound expression into elements.
- [`definition projection`](#definition-projection) — select a property through positive conjunction definitions, from one same-scope source.
- [`equality1`](#equality1) — argument substitution via equality.
- [`equality2`](#equality2) — transitivity of equality.
- [`equalize variable`](#equalize-variable--multiplied-from) — (shared with `multiplied from`; dead alias).
- [`expansion`](#expansion) — named expression rewritten to compiled form.
- [`expansion for integration`](#expansion-for-integration) — mirror of expansion on integration side.
- [`externally provided theorem`](#externally-provided-theorem) — from `externally_provided_theorems.txt`.
- [`implication`](#implication) — compiled implication rule fired.
- [`implication composition`](#implication-composition) — a producer chain and unary consumer composed with scoped proof support; the result may list the producer premises in any order.
- [`implication grouping`](#implication-grouping) — complete common-premise sources combined into a binder-covering conjunction.
- [`contraposition`](#contraposition) — exact signed reversal of one cited implication.
- [`incubator back reformulation`](#incubator-back-reformulation) — incubator operator-equality rewritten.
- [`multiplied from`](#equalize-variable--multiplied-from) — re-emitted with bound vars identified per Bell partition.
- [`or branch assumption`](#or-branch-assumption) — negated-other-disjunct seeded as an `_orint_` subproof premise.
- [`or branch proven`](#or-branch-proven) — one `_orint_` subproof proved the parent OR goal.
- [`or convergence`](#or-convergence) — all branches reached the same conclusion.
- [`or disintegration`](#or-disintegration) — case split on an OR head.
- [`or theorem`](#or-theorem) — an OR-shaped theorem was reached.
- [`or elimination`](#or-elimination) — a theorem derived by the pre-split merge (guard-variant pair + licensing or theorem).
- [`premise element`](#premise-element) — one specific premise cited during integration.
- [`recursion`](#recursion) — induction-hypothesis step.
- [`reformulated from`](#reformulated-from) — this theorem is a reformulation of the cited source.
- [`reformulation for integration and`](#reformulation-for-integration-and) — reverse-disintegration prep for an `and`-category conclusion.
- [`reformulation for integration flattened`](#reformulation-for-integration-flattened) — an additional root rule with checked multi-level definition construction.
- [`reformulation for integration >[]`](#reformulation-for-integration-) — for an `existence` with empty outer bounds.
- [`reformulation for integration >[bound]`](#reformulation-for-integration-bound) — for an `existence` with non-empty outer bounds.
- [`symmetry of equality`](#symmetry-of-equality) — `(=[b,a])` from `(=[a,b])`.
- [`symmetry of inequality`](#symmetry-of-inequality) — `!(=[b,a])` from `!(=[a,b])`.
- [`task formulation`](#task-formulation) — a root premise of the theorem under proof.
- [`theorem`](#theorem) — a previously-proven theorem fired as an inference rule.
- [`vacuous truth`](#vacuous-truth) — the premise chain is self-contradictory.
- [`validity name`](#validity-name) — declares scope for an integrated expression.
- [`variable copy`](#variable-copy) — fresh `_copy` duplicate of an existing variable.
- [`zoom in`](#zoom-in) — unchanged expression inherited from an exact strict-ancestor scope.

**Non-checker categories** — counted by the verifier, not validated:

- `anchor handling trace` — per-step `_copy` substitution chain from `anchor handling`.
- `origin` — provenance chain for a `contradiction`'s derivation tree.
- `self-reference` — error counter; increments when a chapter cites its own theorem.

**Raw-graph-only tags** — written by the prover, removed by processing, never seen by the verifier:

- [`induction scope`](#induction-scope-raw-graph-only) — a goal reached inside an induction scope, registered at the scope's root in the same absorb.
- [`int scope`](#int-scope-raw-graph-only) — a statement of a finished int scope, registered at the scope's parent when every premise of its goal group has finished.

---


## `zoom in`

**Purpose.** Record an unchanged expression released into a descendant validity scope, keeping its parent expression and derivation intact. The producer is `logical_block/rejected_source_replay.cpp` / `recordZoomInOrigin`, called for the released product and each sibling. It writes one direct dependency from release data and does not read the parent origin map. The normal history writer retains its duplicate and capacity policy.

**Schematic row.** `E child_scope zoom in E parent_scope` — exactly five tab-separated columns and one dependency. The source must be a strict ancestor (not necessarily the immediate parent); expressions must be identical.

**Checker.** `verifier.py` / `check_zoom_in` requires the exact `(expression, parent_scope)` row in the chapter. The child row cannot satisfy its own source requirement merely by sharing expression text. Same-scope, sibling, descendant-source, changed-expression, extra-dependency, and missing-parent cases fail. Ordinary type and origin-chain termination checks remain active; the tag is not origin-exempt.

**Exporter.** [`proof/proof_reconstructor.hpp`](../../../GL_Quick_VS/GL_Quick/src/proof/proof_reconstructor.hpp) / `liftToShallowestOriginAncestor` preserves an exact zoom-in result instead of lifting it away. `ProofReconstructor::buildStack` preserves the recorded source scope while traversing that dependency. Processing preserves exact theorem spelling for the source instead of turning it into a theorem-citation alias. Both endpoint rows remain auditable.

**Lean.** `proof_export/certificate.py` records a `scope_inherit` action with an exact local source step. `proof_export/lean.py` / `_scope_inheritance_proof` enters the target context and applies the cited parent proof to its own parameters, assumptions, and witness guards. Extra child assumptions remain unused. Existing witnesses and existential bundles are inherited, not minted again. No axiom or proof search is introduced.

**Decision.** [D-342](../40_decisions.md#d-342). **Invariant.** [I-223](../30_invariants.md#i-223).

---

## `induction scope` (raw graph only)

**Purpose.** Record that a goal reached inside an induction scope leaves the scope tree: a statement that is an open goal at the tree's root (the nearest scope that is not an induction scope) or above it is registered at the root as well, in the same absorb, so the goal closes in the burst that derived it. The producer is the statement door, `logical_block/statement_memory.hpp` / `StatementMemory::addStatement` (`ScopeSyntax::inductionGoalExit`); no other statement leaves an induction scope.

**Schematic row.** `G root_scope induction scope G leaf_scope` — one dependency, the identical expression at the induction scope it was derived in.

**Never checked.** Induction scopes carry no assumption, and the processed graph deletes them from every validity name (`process_proof_graphs.py` / `strip_induction_scopes`). The row then cites itself and is replaced by the leaf's derivation, so neither the verifier, the Lean export nor the HTML sees the tag; processing asserts that none survives. A `zoom in` of a source into an induction scope becomes an identity the same way and is replaced by its source.

**Exporter.** `ProofReconstructor::buildStack` keeps the row's dependency at the leaf's exact scope, as for `zoom in`: lifting it to the shallowest origin-bearing ancestor would cite the row itself.

**Decision.** [D-519](../40_decisions.md#d-519). **Invariant.** [I-321](../30_invariants.md#i-321).

---

## `int scope` (raw graph only)

**Purpose.** Record that a statement of a finished int scope moves to the scope's parent. An int scope (`<parent>_boundary_int_scope_<premise>_<witness>`) holds one premise-bound integration of a split induction hypothesis; it carries no assumption. When every premise of the goal group has its first finished scope (a statement there is an instance of the integration goal), those scopes are promoted together: every statement registered at each of them and not known at the parent is registered at the parent in the same absorb, with its levels, and the scopes are then wiped. The producer is the statement door, `logical_block/statement_memory.hpp` / `StatementMemory::promoteIntScope`, called by `StatementMemory::finishIntScope`; no other statement leaves an int scope.

**Schematic row.** `S parent_scope int scope S int_scope` — one dependency, the identical expression at the int scope.

**Never checked.** As for `induction scope`: the processed graph deletes the `int_scope_` segments from every validity name (`process_proof_graphs.py` / `strip_induction_scopes`), the row then cites itself and is replaced by the leaf's derivation, and processing asserts that none survives. A `zoom in` of a witness's source into an int scope becomes an identity and is replaced by its source.

**Exporter.** `ProofReconstructor::buildStack` and `DerivationTree::chooseRow` keep the row's dependency at the leaf's exact scope, as for `zoom in` and `induction scope`.

**Decision.** [D-578](../40_decisions.md#d-578). **Invariant.** [I-343](../30_invariants.md#i-343).

---

## `anchor handling`

**Purpose.** Pin a raw bound-variable index in an anchor application to its anchor-slot name. At most one emission per chapter; a trace of subsequent `_copy` substitutions follows.

**Checker.** `check_anchor_handling` at [`verifier.py`](../../verifier.py).

**Row shape.**

```text
<rewritten-anchor-expr>  main  anchor handling  <original-anchor-expr>  main
```

**Example (0_single_chapter_proof.txt):**

```text
(AnchorPeano[N,0_copy,s,+,*,i1])	main	anchor handling	(AnchorPeano[N,i0,s,+,*,i1])	main
```

**What the checker validates:**

- LHS expression is an anchor atom.
- The substitution from RHS to LHS is consistent with the anchor's config slot types.
- Exactly one `anchor handling` row per chapter (uniqueness guard at [`verifier.py–2654`](../../verifier.py)).

---

## `compilation`

**Purpose.** Records the ASIC-0.1-prep step ([D-76](../40_decisions.md#d-76)): an implication entering the mail "implications" channel is *also* compiled to its compact named form and deposited as a `mailOut.statements` expression. The row links the compact form to the original expanded implication it was compiled from. Purely additive provenance — it does not change which theorems prove.

**Checker.** `check_compilation` at [`verifier.py`](../../verifier.py). In `_ORIGIN_EXEMPT_TAGS` — the compact↔original link is definitional (the `implication<N>` name *is* the original by GL-binary construction), so the structural binary-faithfulness check is the whole validation; no generic dependency-origin walk.

**Row shape.**

```text
<compact (implication<N>[args])>  main  compilation  <original expanded implication>  main
```

`line.expression` is the compact form (LEFT); `rest[0]` is the original expanded implication (RIGHT); `rest[1]` its namespace. Both are always `"main"` (the mail implications channel is main-only by [I-26](../30_invariants.md#i-26)).

**What the checker validates:**

- `len(rest) >= 2` and `line.namespace == rest[1]` (compaction stays in scope).
- `line.expression`'s core has a GL-binary entry of `category == "implication"` whose `elements`/`signature`, instantiated with the compact's actual args, reconstruct (via `_try_expand` → `_build_implication_from_elements`, modulo normalize-with-unchangeables) the original implication in `rest[0]`. A miss is a real failure the verifier surfaces — failures are first-class.

**Prover-side emission.** The eight `updateGlobalSingleChapter` / `updateGlobal` broadcast sites (plus the two formerly-backup-less producers — `broadcastTheorems` load-time and `addExprToMemoryBlock` recovered-implication re-broadcast — per [D-83](../40_decisions.md#d-83)) call `recordPendingCompaction`; the single-threaded post-`pool.join` drain compiles each via `StructuralCompiler::compileImplicationToCompact` and emits the `compilation` paired origin (gated on `parameters.trackHistory`; also discharges the receiver's [D-45](../40_decisions.md#d-45) paired-origin requirement). On the legacy `mailOut.implications.insert` sites are retired with the channel itself ([D-78](../40_decisions.md#d-78)); only the compact-form deposit on `mailOut.statements` remains. See [`03_mail_system.md`](03_mail_system.md#additive-compact-implication-deposit-asic-01-prep-d-76) and the [prover chapter](../10_pipeline/04_prover.md#compileimplicationtocompact).

---

## `contradiction`

**Purpose.** Under the contradiction LB's proof assumption, the prover derives both `X` and `!X` at that LB's `main`. The assumption discharges and the original conclusion is emitted. A pair available only in a deeper branch scope does not emit this tag.

**Emitted by.** `dischargeContradiction` ([`logical_block/goal_processor.hpp`](../../../GL_Quick_VS/GL_Quick/src/logical_block/goal_processor.hpp)) — the single per-step sweep over `intEncodedStatements` that detects a main/main contradiction and fires the discharge. (Formerly the "Site G" block inside `addExprToMemoryBlock`; see the [prover chapter](../10_pipeline/04_prover.md#contradiction-discharge).)

**Checker.** `check_contradiction` at [`verifier.py`](../../verifier.py). Accepts the row/seed relation in either polarity: the row's expression and the seed (`cleanOp`, rest[4]) must be exact negations of each other — `"!"+cleanOp` for a verbatim positive-seed LB, `cleanOp == "!"+expression` for a complement negated-seed LB ([D-214](../40_decisions.md#d-214)).

**Scope-carrying reductio variant** (sequenced-ordis campaign; emitted by `dischargeContradictionScopes` and `ordisMerge`'s branch-contradiction refutation): exactly FOUR rest fields — `X` at the marked child scope (rest[0..1]) and `negate(X)` at the scope where it is known (rest[2..3]). The checker validates: the two are exact negations; rest[3] is rest[1] or one of its `_boundary_`-ancestors; `line.namespace` is rest[1] minus its last `_boundary_<payload>` segment; and `line.expression` negates the scope's assumption (the negated seed of a `contradiction_<!S>` payload verbatim, or the disjunct's negation for an `ordis_<sig>_(<disjunct>)` payload). The companion `task formulation` checker accepts the contradiction scope's own positive seed row at that scope; the `contradiction trace` meta-check traces the new shape to the child scope's own seed row.

**Rest fields.** Origin chain — references the two contradicting expressions and the discharged assumption.

**Context.** Each contradiction step has an associated `origin` (non-checker) category entry recording the derivation tree. The verifier's origin-trace loop at [`verifier.py–2706`](../../verifier.py) walks this chain.

---

## `disintegration`

**Purpose.** A compound compiled structure was split into its constituent elements — each conjunct of an `and`, each side of an `existence` after witness binding.

**Checker.** `check_disintegration` at [`verifier.py`](../../verifier.py).

**Scope rule (`D-341`, I-16-consented).** The compound (`rest[0]` at `rest[1]`) lives at the product's scope OR at a strict ancestor of it (`_ns_matches_or_strict_prefix(rest[1], line.namespace)`) — the D-33 / D-58 widening `equality1` and `implication` already carry. The producer is the ancestor-park release: a product parked at `main` and demanded only inside an or branch registers in the branch with its `main`-scope `disintegration` line copied verbatim. Sound: a compound valid at an ancestor holds at every descendant, and an existential instantiated inside a sub-scope with a fresh witness is ordinary ∃-elimination. The expansion row is still looked up at the compound's own scope; existence siblings are collected at the product's scope. A sibling or descendant compound scope stays rejected.

**Generic history transfer.** `copyOriginsToInternalMail` copies this row when present, along with every other stored tag, without changing dependency scopes. This transfer does not validate a proof row or change any checker's scope contract; generic-copy verifier failures are reviewed separately.

**Row shape.**

```text
<element>  main  disintegration  <source-compound>  main
<element>  main_boundary_ordis_…  disintegration  <source-compound>  main
```

**Example (from 12_check_zero.txt):**

```text
(implication4[N,i0,s])	main	disintegration	(&(&(&(&(&(&(&(&(&(&(&(&(in[i0,N])(fXY[s,N,N]))(implication4[N,i0,s]))(implication5[N,s]))(fXYZ[+,N,N,N]))(implication11[N,i0,+]))(implication12[N,i0,+]))(implication13[N,s,+]))(implication14[N,s,+]))(fXYZ[*,N,N,N]))(implication15[N,i0,*]))(implication16[N,s,*,+]))(implication17[N,s,*,+]))	main
```

**Validation.** The LHS must be a constituent of the RHS according to the GL binary's `elements` list for the RHS's named expression.

**Or-cohort rule rows.** For an or-category compact, the row's expression is a flat hash rule over the disjuncts, accepted in either of two shapes (`_check_or_disintegration_implication`): the K mutual-exclusion form — k−1 premises equal to the negations of the other disjuncts, head one disjunct — or the SUBSET-EXCLUSION form ([D-269](../40_decisions.md#d-269), `_check_or_subset_exclusion`): j premises (1 ≤ j ≤ k−2) equal, as a MULTISET, to the negations of the excluded disjuncts, and a reduced (k−j)-ary or-compact head whose flattened disjunct list embeds in the parent's as an order-preserving subsequence (k ≥ 3 parents only; the reduced operator is pre-minted by the producer). Negations use double-negation cancellation throughout (I-175).

---

## `equality1`

**Purpose.** Argument substitution — given `(=[a,b])` and any expression `E(a)`, emit `E(b)`.

**Checker.** `check_equality1` at [`verifier.py`](../../verifier.py).

**Rest layout.** Exactly the interleaved `(expr, ns)` pattern from the row-shape recap:

| Slot | Role | Contents |
|---|---|---|
| `rest[0]` | `source_expr` | The pre-substitution expression: a simple atom (atom, negated atom, equality or negated equality) with the result's polarity, operator and arity. |
| `rest[1]` | `source_ns` | Namespace of the source: the row's own namespace or a strict ancestor ([D-33](../40_decisions.md#d-33) widening, `_ns_matches_or_strict_prefix`). |
| `rest[2i]`, `i ≥ 1` | equality `i` | Exactly one positive equality `(=[a,b])` justifying one substituted arg position (orientation source→result: `(source_arg, result_arg)` must be in the equality set). An inequality or any other relation rejects the row. |
| `rest[2i+1]` | equality `i` namespace | Each equality lives either in the row's namespace OR a strict byte-level-prefix (ancestor) namespace — see `_ns_matches_or_strict_prefix` at `verifier.py`. Branch-scope results can cite ancestor-scope equalities. |

**Minimum `len(rest)` = 4** (source + one equality). Shorter → the checker returns `False`.

**Validation contract.**

1. Source and result parse as simple atoms (`_simple_atom`) with the same polarity, the same operator and the same arity; every cited equality parses as `(=[a,b])` (`_equality_args`).
2. For every arg position `i` where `source_args[i]!= result_args[i]`: the tuple `(source_args[i], result_args[i])` must be present in the assembled equality set.
3. No covering equality found for *any* differing position → row rejected.

**Prover-side emission sites.**

- `EqualityPropagation::drainStagedEquiProducts` at `logical_block/equality_propagation.hpp` — publication of staged statement rewrites. `rest[0]` is the original encodedStatement before args were mapped via the current class.
- `applyEquivalenceClassToRejectedMapIntegration` (integration-revival path, see [`05_equivalence_classes.md`](05_equivalence_classes.md#extension--rejectedmapintegration)) — emits `rest[0]` = the pre-rewrite form of the emitted constituent (NOT the compound, NOT the post-rewrite form). Common authoring mistake: passing the compound in `rest[0]` — the arity check then fails because the constituent and the compound have different cores.

**Common authoring pitfalls.**

- Passing the POST-rewrite form as `rest[0]` — arity matches but substitution check yields `0!= 0`, so no equality is needed, the row passes trivially and the eq is never validated. Subtle soundness leak if the "eq" entries aren't actually equalities. Always pass PRE-rewrite as source.
- Citing a source or an equality from a scope that does not enclose the row's — `_ns_matches_or_strict_prefix` rejects a sibling or descendant scope, and the origin meta-check rejects a citation whose expression is derived only in a nested scope.
- Forgetting the `ns` interleave — `len(rest) < 4` or odd-length `rest` is silent failure.

---

## `equality2`

**Purpose.** Transitivity of equality — from `(=[a,b])` and `(=[b,c])`, emit `(=[a,c])`.

**Checker.** `check_equality2` at [`verifier.py`](../../verifier.py).

**Rest fields.** Exactly the two transitively-related equalities with their scopes. Result and both premises must be exactly positive equalities `(=[x,y])`.

---

## `equalize variable` / `multiplied from`

**Purpose.** A parent theorem is re-emitted with some of its bound variables identified according to a Bell partition — `multiplyImplication` at [`inference/implication_multiplier.cpp`](../../../GL_Quick_VS/GL_Quick/src/inference/implication_multiplier.cpp). Partitions that would merge two distinct free `u_*` anchor parameters into the same equivalence class are skipped (see [prover chapter](../10_pipeline/04_prover.md#multiplyimplication-bell-partition-equalisation) and [I-24](../30_invariants.md#i-24)).

**Checker.** `check_equalize_variable` at [`verifier.py`](../../verifier.py) — **shared** by both tag keys.

In addition to the slot-by-slot consistency check, the checker enforces a free-anchor-merge guard: it parses the bound-variable list from every `>[...]` binder of the source and copy implications, builds the orig→copy arg mapping over the paired premises and head, and rejects any pair where the orig arg is not bound by the source binders, the copy arg is not bound by the copy binders, and the two names differ. Bound→bound (Bell-partition merge) and bound→free (bound-variable specialisation to a known anchor) remain accepted — the rejection fires only on free-to-free remaps with distinct names, which is the chapter-1115 multiplyImplication soundness violation that the prover-side gate also rejects.

A copy binds each name once along each scope chain: a merged class keeps its outermost binder ([I-290](../30_invariants.md#i-290)); the whole-row `bound variable uniqueness` meta-counter rejects a copy, or a rule citation, that rebinds a name.

**Note.** Only `multiplied from` is ever emitted in current code. The `equalize variable` key is a dead alias kept in the registry for symmetry.

**Rest fields.** The source (unmultiplied) theorem + the partition.

---

## `expansion`

**Purpose.** A named compiled expression (`and` / `existence` / `implication` / `or<N>` / `existence<N>`) is rewritten into its GL-binary `elements` form.

**Checker.** `check_expansion` at [`verifier.py`](../../verifier.py).

**Conjunction-alternative introduction.** An implication compact with an OR premise also licenses introduction from one of that OR's alternatives. Positive conjunction alternatives unfold to their complete ordered premise chains; negative conjunctions remain signed leaves. `proof_export/or_introduction.py`, `match_or_introduction`, checks exact fixed parameters, signs, binder visibility and bijective local renaming. The certificate records the conjunction tree and OR position for explicit Lean construction, without redefining the cited compact. See [I-264](../30_invariants.md#i-264).

**Validation.** Ordinary expansion substitutes the compact's actual arguments into its complete category-specific defining form, with the compact source present at the same scope. For a conjunction, the result is the whole defining conjunction; selecting one member is the separate `disintegration` operation. Negative compounds and the explicitly supported auxiliary implication forms have their own structural checks in `check_expansion` and `_try_expand`. The conjunction-refutation subcases above cite either a concrete OR or an expanded owning implication at comparable scopes; they do not redefine the cited expression.

---

## `expansion for integration`

**Purpose.** Mirror of `expansion` on the integration side — a reformulated expression is expanded back into compiled-structure form so its elements can be recombined.

**Checker.** `check_expansion_for_integration` at [`verifier.py`](../../verifier.py).

**Per-category acceptance (`_try_expand`).**
- `and` — LEFT must equal `_build_and_from_elements(elements)` (left-nested `(&E1E2)`, `(&(&E1E2)E3)`, …).
- `implication` — LEFT must equal `_build_implication_from_elements(elements, unchangeables)` (binds first-appearance variables in `>[…]`).
- `existence` — LEFT must equal `_build_existence_from_elements(elements)` (occupied-or-pi-bound) OR `_build_existence_empty_binding(elements)` (occupied bound vars stripped to `>[]`).
- `or` — LEFT must equal **either** `_build_or_from_elements(elements)` (the De Morgan form `!(&!E1!E2…!Ek)`, primary acceptance) **or** any of the K per-branch sub-implication forms `(>[](AND-of-negated-others)(D_k))` produced by `_build_or_subimpls_from_elements(elements)` ([D-52](../40_decisions.md#d-52)). The sub-implication branch corresponds to the `_orint_` (OR-introduction) goal-flow rows the producer emits at the OR's parent scope when opening per-branch sub-proofs. See [`07_or_branching.md` §3a](07_or_branching.md#3a-orint-or-integration-single-branch-proves) for the full mental model.

---

## `externally provided theorem`

**Purpose.** A theorem injected via `files/theorems/externally_provided_theorems.txt` — an axiomatic boundary for the current batch.

**Checker.** `check_externally_provided_theorem` at [`verifier.py`](../../verifier.py).

**Validation:**

- `namespace == "main"`.
- Expression is either a direct member of `state.external_theorems` or a valid mirror of some member.

---

## `implication`

**Purpose.** A compiled implication rule fired — its premise pattern was matched in hash memory at this scope and its conclusion was emitted.

**Checker.** `check_implication` at [`verifier.py`](../../verifier.py) — the most frequently-fired checker.

**Example (0_single_chapter_proof.txt, first row):**

```text
(in3[i0,i1,v1,+])	main	implication	(>[s,+,i1](AnchorPeano[N,i0,s,+,*,i1])(>[v1,v2](in2[v1,v2,s])(in3[v1,i1,v2,+])))	main	(AnchorPeano[N,0_copy,s,+,*,i1])	main	(in2[i0,v1,s])	main
```

Decoded:

- LHS: `(in3[i0,i1,v1,+])` — the emitted conclusion.
- Rule: `(>[s,+,i1](AnchorPeano[…])(>[v1,v2](in2[…])(in3[…])))` — the fired implication.
- Premise 1: `(AnchorPeano[N,0_copy,s,+,*,i1])`.
- Premise 2: `(in2[i0,v1,s])`.

**Validation.** Disintegrate the rule into (chain, head). Verify the cited premises match the chain under some assignment. Verify the LHS is the head under the same assignment.

**Namespace rule (D-35 — comparable-scope premise inheritance).** Every premise namespace and the implication's namespace (`rest[1]`) must be one of: `"main"`, the result's namespace, or a strict ancestor of the result's namespace (`result_ns.startswith(ns + "_boundary_")`). Pre-D-35 the rule was strict "at most one distinct non-main namespace, equal to the result's" — which rejected legitimate FTA-rung-1 firings where the result lands in an OR-branch scope but some premises live at the OR's parent scope. The new rule encodes faithful comparable-scope inheritance: a fact at an ancestor scope is visible at every descendant.

**Validity-stack deposit rule (D-58 — mirrors `nm.deeperOf` accumulation).** On top of D-35:

1. Every PAIR of constituent namespaces (impl + each premise) must be comparable (parent-child via [`comparable / deeperOf`](04_validity_stack.md#comparablea-b) in the validity-stack chapter — one is an ancestor of the other, or they are equal). Sibling scopes are rejected.
2. The row's namespace (`line.namespace`) must EQUAL the deepest constituent namespace. Equivalently, `line.namespace ∈ {impl_ns} ∪ {premise_nss}`.

The C++ prover's hash-kernel ([`generateEncodedRequestsStatic`](../../../GL_Quick_VS/GL_Quick/src/memory.cpp)) accumulates the joined-scope of combined facts via `nm.deeperOf(...)` — the result of an implication firing lives at the deepest scope of its inputs, never at a strictly deeper scope no constituent reaches. The verifier mirrors this. See [I-38](../30_invariants.md#i-38) and [D-58](../40_decisions.md#d-58) for the full rule + sound-under-validity-stack-semantics rationale.

**Concrete failing example (pre-fix, rung-1 incubator branch).**

```text
(=[it_0_lev_0_32,2])	main_boundary_(implication23[2,8,int_lev_4_2365])	implication	(>[1](in[1,u_1])(>[2](in2[2,1,u_3])(>[3](in2[3,1,u_3])(=[2,3]))))	main	(in2[2,6,3])	main	(in2[it_0_lev_0_32,6,3])	main	(in[6,1])	main
```

Every constituent (impl + 3 premises) at `main`; result at `main_boundary_(implication23[2,8,int_lev_4_2365])`. D-35 alone passes — each constituent IS an ancestor of result_ns. The deeperOf-equality rule rejects: no constituent reaches the result's scope, so the prover's hash kernel could not have deposited the row at that scope.

---

## `incubator back reformulation`

**Purpose.** Incubator-only. A proved existence-form theorem `(>[...](Anchor)(>[w](BODY)(=[w,c])))` — "∃w: BODY(w) ∧ w=c" — is back-reformulated into the direct operator form `(>[...](Anchor)(BODY[w:=c]))` by eliminating the existential witness through its equality. E.g. `(>[w1](in3[i1,i1,w1,+])(=[w1,i2]))` → `(in3[i1,i1,i2,+])`.

**Checker.** `check_incubator_back_reformulation` at [`verifier.py`](../../verifier.py). Verifies the rewrite **structurally** via the shared helper `_check_back_reformulation`: the source head must be an equality `(=[w,c])`, the source's last premise is the witness `BODY`, the direct form's premises equal the source's premises minus `BODY`, and the direct head equals `BODY` with the witness token substituted by `c`. The same helper backs the `back_reformulated_statement` chapter-goal check. The tag stays in `_ORIGIN_EXEMPT_TAGS` — this checker certifies the reformulation *step*, not the cited source's own provenance (an incubator-side verification target).

---

## `or branch assumption`

**Purpose.** When an OR goal is rewritten into `_orint_` subproofs, every subproof targeting disjunct `D_i` gets the negation `!D_j` of every other disjunct (`j ≠ i`) as a local premise. Each such premise emits one `or branch assumption` row. Despite the historical tag name, this is not an `_ordis_` case-split event.

**Checker.** `check_or_branch_assumption` at `verifier.py` (D-35).

**Row layout (exactly two rest fields).**

```text
<negated-other-disjunct>  <branch-ns>  or branch assumption  <or-expr>_integration_goal  <parent-ns>
```

**Validation (strict per Codex rounds 2 + 3, 2026-05-03):**

1. `len(rest) == 2` exactly. The tag is in `_ORIGIN_EXEMPT_TAGS`, so any extra rest pairs would be silently accepted by the generic origin check; reject up front so the row's contract stays auditable.
2. `rest[0]` ends with `_integration_goal`; stripping the suffix yields a compiled OR `(or<N>[…])` with a known GL-binary entry whose arity matches the OR's argument count.
3. `line.expression` is `!<disjunct>` where `<disjunct>` is one of the OR's disjuncts modulo equality symmetry.
4. `line.namespace` is **EXACTLY** `parent + "_boundary_orint_" + or_expr + "_(" + <asserted> + ")"` for some disjunct `<asserted>` of the OR — no substring search, no trailing content allowed. The asserted disjunct is parsed by walking one balanced parens group inside the wrapper.
5. The asserted disjunct (parsed in step 4) is DIFFERENT from the negated one (modulo equality symmetry). The row asserts a disjunct's negation only in branches where ANOTHER disjunct is asserted.
6. **A matching `or branch proven` row exists (Codex round-3).** Some chapter row with `tag == "or branch proven"`, `expression == or_expr`, `namespace == parent_ns`, `len(rest) == 2`, `rest[1] == branch_ns`, and `rest[0]` matching the selected disjunct (modulo equality symmetry). Without this check the assumption row could pass structurally even when the corresponding `_orint_` subproof did not prove the OR goal.

History: pre-D-35 the tag was claimed retired (overridden before export). The override never existed; the prover (`logical_block/integration_processor.hpp`, `IntegrationProcessor::prepareIntegrationCore2`) emits this tag live, and FTA-rung-1 chapter `1209_single_chapter_proof.txt` line 43 carries it. Iterations: round-1 added the checker; round-2 tightened to `len(rest) == 2`, exact-equality branch-namespace match, and balanced-parens disjunct parsing; round-3 added the matching-`or branch proven` cross-check. Chapter 1209's `or branch assumption` row continues to pass after round-3 — the matching `or branch proven` row at line 22 satisfies all five sub-conditions.

---

## `or branch proven`

**Purpose.** Records that one `_orint_` subproof has proved its parent OR goal through a selected atomic leaf. Despite the historical tag name, this is not the opening of an `_ordis_` case-split branch.

**Checker.** `check_or_branch_proven` at `verifier.py` (D-35).

**Row layout (exactly two rest fields).**

```text
<or-expr>  <parent-ns>  or branch proven  <asserted-leaf>  <branch-ns>
```

**Validation (strict per Codex round-2; round-3 check #1 dropped per [D-36](../40_decisions.md#d-36)):**

1. `len(rest) == 2` exactly. The tag is in `_ORIGIN_EXEMPT_TAGS`, so any extra rest pairs would be silently accepted by the generic origin check; reject up front so the row's contract stays auditable.
2. `line.expression` is a known compiled OR with ≥2 ordered atomic leaves after recursive substitution against the loaded OR binaries.
3. `rest[0]` is one of those leaves (modulo equality symmetry).
4. `rest[1]` is **EXACTLY** `parent + "_boundary_orint_" + or_expr + "_(" + <leaf> + ")"` for `<leaf>` matching `rest[0]` (modulo equality symmetry). No substring search — the row claims this exact atomic subproof.

**About the missing step 5 ([D-36](../40_decisions.md#d-36)).** An earlier Codex round-3 step required a non-`or branch proven` derivation row for the OR at parent scope. That check was based on a wrong mental model of `_orint_` (treated it as case-split with a separately-derived OR). Correct semantics: `_orint_` rewrites the OR goal into K atomic sub-implications-to-prove; when one fires, the `or branch proven` row IS the OR's derivation by design — there is no separate derivation row to look for. The check was unsatisfiable for legitimate proofs and dropped as a verifier correction, not a relaxation.

**Terminology note ([D-36](../40_decisions.md#d-36)).** "branch" in this tag's name and in `or branch assumption` is historical — these rows record per-SUBPROOF events, not case-split branch events. A "subproof" here is one of the K atomic-leaf implications-to-prove that `_orint_` opens. The renaming has not happened to avoid touching every consumer.

History: pre-D-35 the tag was claimed retired (replaced by `or theorem`). `or theorem` is a different tag — it tags theorems whose head is an OR shape. `or branch proven` is the subproof-firing record, emitted live by the prover (`prover.cpp` `or branch proven` site) and first-class in the verifier since D-35. Iterations: round-1 added the checker; round-2 tightened to `len(rest) == 2` and exact-equality namespace match; round-3 added an OR-origin requirement (later dropped by D-36 as a correction — see above).

---

## `or convergence`

**Equality-supported extension.** A row can cite different actual case facts plus positive equalities. `proof_export/or_convergence.py`, `match_equality_convergence`, checks one source and the necessary equality paths separately under each case scope, using that scope and ancestors only and accounting for every dependency. Retired cases retain explicit refutations. The verifier requires every cited pair to exist; the certificate records dependency indexes; Lean applies equalities inside each case before OR elimination. The exact-expression layout below remains the legacy contract. See [I-263](../30_invariants.md#i-263).

**Purpose (mathematical contract).** An `or convergence` row certifies that the conclusion `line.expression` holds under EVERY branch of the case split on the OR cited in `rest[0]` — derived in each surviving branch, and discharged by reductio in each retired branch (dead-branch retirement: the branch's asserted disjunct is refuted) — and is therefore promoted to the OR's parent scope cited in `rest[1]`.

**Checker.** `check_or_convergence` at [`verifier.py`](../../verifier.py) — validates the spec'd row layout. As of [D-36](../40_decisions.md#d-36) the producer side (`ordisMerge` in `logical_block/case_processor.hpp`) emits the new layout, and chapter `1209_single_chapter_proof.txt`'s convergence rows pass cleanly.

### Row layout (compiled-form OR only)

```text
<C>  <parent>  or convergence  <OR>  <parent>
                                      <C>  <branch_D1>
                                      <C>  <branch_D2>
                                      …
                                      <C>  <branch_DK>
```

For an OR with `K` disintegration leaves, the rest field has `2 + 2*K` slots. `K` is the recursively flattened, ordered non-OR leaf count for a contiguous nested OR; `rest[0]` remains the original outer compact signature:
- `rest[0]` = OR (compiled `(or<N>[…])`)
- `rest[1]` = parent (the OR's parent scope; must equal `line.namespace`)
- one two-field entry per flattened disjunct: `rest[2*i + 2]` (an expression) and `rest[2*i + 3]` (a namespace)

Each entry takes one of two forms ([D-238](../40_decisions.md#d-238)): a **survivor** entry `(C, branch_Di)` — the conclusion at that branch's exact `_ordis_` scope — or a **retired** entry `(negate(D_i), ns)` — the disjunct's negation at a scope the parent inherits from (the parent itself, an ancestor, or the retired branch itself for an ex-falso self-refutation). Either way the row accounts for all K flattened disjuncts.

### Validation contract

For the row to PASS, all of:

1. **Layout.** `len(rest) == 2 + 2*K`; `len(rest)` is even; `len(rest) >= 6` (so `K >= 2`).
2. **Parent-scope match.** `line.namespace == rest[1]`.
3. **OR is real.** `rest[0]` is a compiled `(or<N>[…])` with a known GL-binary entry of `category == "or"`; recursively flattening every substituted OR child yields at least two ordered non-OR leaves, and that leaf count `K` matches the entry count. A negated-conjunction integration premise `!(name[…])` of category `and` is also a source: its `K` leaves are its complemented conjuncts ([D-460](../40_decisions.md#d-460)).
4. **Entry classification + coverage.** Every entry matches one uncovered disjunct — a survivor entry by the disjunct payload embedded in its `_ordis_` branch scope (`parent + "_boundary_ordis_" + OR + "_(" + D_i + ")"`), a retired entry by `negate(D_i)` plus a parent-visible scope — and the K entries cover all K disjuncts exactly once.
5. **Per-entry derivation evidence ("each ingredient has its own line").** For every entry `(expr, ns)`, a chapter row exists with `expression == expr` and `namespace == ns` under any tag — a survivor's branch derivation of `C`, or a retired branch's refutation.

Step 5 is the key strengthening per the user's directive — it ties each cited branch to a real chapter-local derivation of `C`. Without it the rest fields would be unverifiable assertions; with it the verifier reduces convergence-checking to standard chapter-row existence checks (the same pattern used by the inline origin check).

### Status (D-36)

Chapter `1209_single_chapter_proof.txt`'s 2 convergence rows now pass cleanly. The producer side (`ordisMerge` in `logical_block/case_processor.hpp` at the post-D-36 emission site) writes the spec'd layout: `(C, parent, or convergence, OR, parent, C, branch_D1, C, branch_D2)`. Each branch's derivation of `C` remains in `exprOriginMap` (because `removeExpressionFromMemoryBlock(state=0)` touches only `intEncodedStatements`), so chapter export's `buildStack` naturally renders the per-branch rows when it recurses on the new ingredients.

A side-effect of the producer fix: 4 previously-hidden `or disintegration` rows now appear in chapter 1209 (lines 17, 21, 29, 33) — the `_ordis_` per-branch case-split records that were never reached by `buildStack` before. They use the same compiled-OR row layout and were verified by [D-36-extended `check_or_disintegration`](#or-disintegration-d-36-extended-checker).

---

## `or disintegration`

**Purpose.** Case split on an `or` head. Every contiguous nested OR is recursively flattened in the same producer call; each ordered non-OR leaf becomes a sub-goal inside a dedicated branch scope. The branch scope is named with the original outer OR plus the asserted leaf in its payload (`_boundary_ordis_<outer-or>_(<leaf>)`). No intermediate compiled-OR branch is valid. Sibling row to `or branch proven` — both `_ordis_` and `_orint_` use the same ordered atomic leaves, with different convergence contracts.

**Checker.** `check_or_disintegration` at [`verifier.py`](../../verifier.py) — D-36-extended for compiled-OR form + per-branch namespace check + OR-origin requirement.

**Row layout (exactly two rest fields).**

```text
<asserted-disjunct>  <branch-ns>  or disintegration  <OR>  <parent-ns>
```

**Validation (D-36):**

1. `len(rest) == 2` exactly. Same `_ORIGIN_EXEMPT_TAGS` rationale as the other OR-tag checkers.
2. `rest[0]` is a known compiled OR `(or<N>[…])` whose recursively substituted contiguous OR tree yields at least two ordered non-OR leaves (matching arity at every node), or a negated conjunction `!(name[…])` of category `and` — split only as the premise of an implication being integrated — whose leaves are its complemented conjuncts in definition order ([D-460](../40_decisions.md#d-460); `_case_split_disjuncts`).
3. `line.expression` is one of those flattened leaves (modulo equality symmetry); an intermediate compiled OR is not a leaf.
4. `line.namespace` is **EXACTLY** `rest[1] + "_boundary_ordis_" + rest[0] + "_(" + <leaf> + ")"` for `<leaf>` matching `line.expression` (modulo equality symmetry). The outer OR stays in the namespace even when the leaf came from an inner OR. No substring search.
5. **The OR has an independent derivation row at parent scope.** A chapter row exists with `expression == rest[0]`, `namespace == rest[1]`, and `tag!= "or disintegration"` — i.e. the OR was actually derived (via `implication`, `expansion`, `theorem`, …) before being case-split. This check IS well-founded for `_ordis_`: case-split CONSUMES an existing OR. The analogous check on `check_or_branch_proven` was dropped per [D-36](../40_decisions.md#d-36) because `_orint_` PRODUCES an OR — no separate derivation exists by design. This is the exact semantic asymmetry that motivated the round-3 correction.

Pre-D-36 the checker accepted only the expanded `!(&!(…))` form in `rest[0]` and never validated namespace structure or OR-origin. It was effectively dead in the incubator path because chapter export didn't render `_ordis_` per-branch rows. Stage P1's `ordisMerge` extension exposed them; Stage P1b tightened the checker. [D-218](../40_decisions.md#d-218) later made the checker recursively reconstruct the producer's one-go leaf cohort.

---

## `or theorem`

**Purpose.** An OR-shaped theorem was reached as a goal — `line.expression` is the proven theorem and its head is an `or<N>[…]` node. Distinct from [`or branch proven`](#or-branch-proven), which records an `_orint_` subproof discharging its parent OR goal; `or theorem` is the chapter-conclusion record for theorems whose statement IS an OR.

**Checker.** `check_or_theorem` at [`verifier.py`](../../verifier.py).

---

## `or elimination`

**Purpose.** The pre-split merge ([D-289](../40_decisions.md#d-289)): `line.expression` is a DERIVED theorem obtained by classical or-elimination from two previously proved guard variants — rows identical except one binder-free innermost guard premise — under a previously proved or theorem whose two disjuncts are exactly those guards. The row is the whole chapter (`<N>_or_elimination.txt`, fabricated by `generateRawProofGraph` like or-theorem chapters), with three citations: `rest[0]` variant A, `rest[2]` variant B, `rest[4]` the licensing or theorem.

**Checker.** `check_or_elimination` at [`verifier.py`](../../verifier.py) — a full structural check, not a stub: all three citations resolve against the global theorem registry (exact / w-to-v revert / normalized); the variants disintegrate to equal heads and equal chains except the innermost premise (shared-map `_normalize_expr_list` comparison; the guards must differ); the merged claim equals the common chain plus head (normalized); the or theorem's head decodes through the GL binaries (`_or_disjuncts_from_compiled`, flattened) to exactly two leaves aligning with the two guards under ONE variable bijection with literal polarity (`_or_elim_align`, [I-175](../30_invariants.md#i-175)); and every or-theorem side premise is covered by the common premises (byte-present under the bijection, or a typing atom whose mapped variable occurs in a common premise). Origin-exempt (citations are proved theorems, not local rows); the exemption also enrolls all three citation cells in the theorem-usage acyclicity graph.

---

## `premise element`

**Purpose.** During integration, one specific premise of the source implication is called out as a dependency of the integrated step.

**Checker.** `check_premise_element` at [`verifier.py`](../../verifier.py).

---

## `recursion`

**Purpose.** Induction-hypothesis step.

- In a `check_zero` chapter: the induction variable is identified with `i0`.
- In a `check_induction_condition` chapter: the successor form is asserted as the inductive step's premise, and the hypothesis implication is asserted. Under method `induction` the hypothesis is the theorem at the predecessor (other digit arguments fixed); under method `strong induction` it is the theorem at every `m` below the induction variable `n`, bounded by `(in3[m,y,n,+])` and `!(=[m,n])`, every non-anchor variable except `n` universal ([D-500](../40_decisions.md#d-500)).

**Checker.** `check_recursion` at [`verifier.py`](../../verifier.py); the strong hypothesis is rebuilt by `_strong_induction_hypothesis` and compared up to renaming of bound names.

**Example (13_check_induction_condition.txt):**

```text
(in2[v4,v1,s])	main	recursion
```

Read: inside the step chapter, `(in2[v4,v1,s])` — meaning `s(v4) = v1` — is the inductive step's premise, identifying `v1` as the successor-incremented induction variable.

---

## `reformulated from`

**Purpose.** This theorem is a reformulation of the cited source — existence head expanded into left + right elements.

**Checker.** `check_reformulated_from` at [`verifier.py`](../../verifier.py). Helper: `_check_reformulation` at [`verifier.py`](../../verifier.py).

**Binary lookup robustness (D-35).** The helper derives the GL-binary tag from the target's anchor (e.g. `AnchorGauss → "Gauss"`). When the literal anchor-derived tag has no loaded binary entry — the FTA-rung-1 case where `AnchorIncubator → "Incubator"` and the loaded binaries are tagged `IncubatorPeano` / `IncubatorGauss` / `IncubatorGauss1` (no plain `"Incubator"`) — the helper now falls back to scanning every loaded binary for one that defines the head's compiled name as an `existence` entry. Pre-D-35 the missing exact-tag entry caused immediate rejection of legitimate FTA-rung-1 reformulations (chapter `1210_reformulated_statement.txt`).

**Example (103_reformulated_statement.txt):**

```text
(>[N,i0,s,+](AnchorGauss[N,i0,s,+,*,i1,i2,id])(>[v1,v2](in2[v1,v2,s])(>[v3](interval[N,+,i0,v2,v3])(existence4[N,+,v3,v1,i0]))))	main	reformulated from	(>[N,i0,s,+](AnchorGauss[N,i0,s,+,*,i1,i2,id])(>[v1,v2,v3](limitSet[N,+,v1,v2,v3])(>[v4](in2[v2,v4,s])(>[](interval[N,+,i0,v4,v1])(interval[N,+,i0,v2,v3])))))	main
```

---

## `reformulation for integration and`

**Purpose.** Reverse-disintegration prep for an `and`-category conclusion. The `and` node's elements are staged for re-combination.

**Checker.** `check_reformulation_for_integration_and` at [`verifier.py`](../../verifier.py).

---

## `reformulation for integration >[]`

**Purpose.** Reverse-disintegration prep for an `existence` conclusion whose outermost `>[...]` list was stripped (because the witness slot is already occupied by an external value).

**Checker.** `check_reformulation_for_integration_empty` at [`verifier.py`](../../verifier.py).

---

## `reformulation for integration >[bound]`

**Purpose.** Reverse-disintegration prep for an `existence` conclusion with a non-empty outer `>[...]` bound-variable list. The bound variable is carried through the reformulation.

**Checker.** `check_reformulation_for_integration_bound` at [`verifier.py`](../../verifier.py).

---

## `symmetry of equality`

**Purpose.** From `(=[b,a])`, emit `(=[a,b])`.

**Checker.** `check_symmetry_of_equality` at [`verifier.py`](../../verifier.py). Source and result must be exactly positive equalities at the same scope; an inequality or another binary relation is rejected.

---

## `symmetry of inequality`

**Purpose.** From `!(=[b,a])`, emit `!(=[a,b])`.

**Checker.** `check_symmetry_of_inequality` at [`verifier.py`](../../verifier.py). Source and result must be exactly negated equalities `!(=[x,y])` at the same scope.

---

## `task formulation`

**Purpose.** A root premise of the theorem under proof, asserted without justification at the top of the chapter. In contradiction chapters, the assumed seed also carries this tag — always the exact negation of the theorem's head (the un-negated head for a negated-head theorem; the negated head for a positive-head theorem proved by reductio, [D-214](../40_decisions.md#d-214)).

**Checker.** `check_task_formulation` at [`verifier.py`](../../verifier.py).

**Example (end of 0_single_chapter_proof.txt):**

```text
(AnchorPeano[N,i0,s,+,*,i1])	main	task formulation
(in2[i0,v1,s])	main	task formulation
```

Two premises of the theorem `(>[i0,s,+,i1](AnchorPeano[...])(>[v1](in2[i0,v1,s])(in3[i0,i1,v1,+])))`. Both at the root of the chapter — their truth is assumed (the chapter proves the head follows from them).

**Validation.** LHS must disintegrate to one of the theorem's premises (or the anchor, which is always a premise by structure).

---

## `theorem`

**Purpose.** A previously-proven theorem was used as an inference rule in this chapter.

**Checker.** `check_theorem_tag` at [`verifier.py`](../../verifier.py).

**Example (0_single_chapter_proof.txt, row 2):**

```text
(>[s,+,i1](AnchorPeano[N,i0,s,+,*,i1])(>[v1,v2](in2[v1,v2,s])(in3[v1,i1,v2,+])))	main	theorem
```

The cited expression is a proved theorem from `global_theorem_list.txt`. It appears in `theorem` rows whenever it was fired as a rule.

**Validation:**

- LHS expression is present in `state.global_theorems`.
- Self-reference guard: if the cited theorem equals the chapter's own theorem, the `self-reference` counter fires failure ([`verifier.py–2649`](../../verifier.py)).

---

## `vacuous truth`

**Purpose.** The premise chain leading to this implication was shown to be self-contradictory, so the implication head is trivially valid.

**Checker.** `check_vacuous_truth` at [`verifier.py`](../../verifier.py).

**Scope.** Two checked forms: induction at `main`, and the exact head of an ordinary positive implication integration scope.

**Induction rest layout (6 fields).** The existing main-scope contract is unchanged:

| Slot | Role |
|---|---|
| `rest[0..1]` | contradicting expression + namespace |
| `rest[2..3]` | its negation + namespace |
| `rest[4..5]` | the recursion-hypothesis expression + namespace |

**Emitted by.** `dischargeContradiction` ([`logical_block/goal_processor.hpp`](../../../GL_Quick_VS/GL_Quick/src/logical_block/goal_processor.hpp)) — the single per-step sweep over `intEncodedStatements`; the vacuous-truth branch fires on `isPartOfRecursion` plus both contradicting expressions at `main`. See the [prover chapter](../10_pipeline/04_prover.md#contradiction-discharge).

**Induction prover-side: no level gate.** The vacuous-truth discharge has **no** `mb.level` gate. An earlier gate (requiring at least one contradicting ingredient to carry `mb.level` in its per-statement level set) was removed: for theorems whose own premise is impossible (chapter-101-style lemmas) the contradiction is necessarily rooted in the theorem's outer premise, not the recursion step's hypothesis, so an `mb.level` gate would reject those legitimate vacuous-truth cases. Soundness rests on axiom consistency — no chapter-level contradiction can trace only to anchor-level facts.

**Induction verifier-side chapter-local trace (`verifier.py`, counter `vacuous truth trace`).** `verify_chapter` separately walks the chapter-local origin chain from each ingredient; at least one ingredient must trace back to the recursion-hypothesis cited in `rest[4..5]`. Reuses `_trace_back_to` following the contradiction-trace pattern. A separate counter row `vacuous truth trace` reports trace success/failure — when an ingredient fails to reach the hypothesis, the LB's vacuous-truth claim is unsound.

**Ordinary implication rest layout (4 fields).** The first two pairs cite an expression and its exact negation. One fact is at the result's exact namespace; the other is at that namespace or an ancestor. `dischargeVacuousImplicationScopes` emits the scoped goal with the actual level union, then `dischargeToBeProved` emits the ordinary `validity name` closure and schedules scope cleanup. No recursion-hypothesis pair is attached to this form.

`check_scoped_vacuous_truth` requires a registered positive implication compact, a checked expansion at the parent whose exact head is the result, both visible dependency rows, and a dependency path to a checked premise of this same integration expansion. Siblings and descendant-only facts cannot prove the goal. The chapter trace counter applies the same scoped check. Lean enters the implication's complete premise context, specializes each cited fact at its own visible scope, and applies `False.elim` to the opposing pair. If an implication is closed before one of its free parameters receives a chapter witness, Lean keeps that closure universally quantified; later implication applications explicitly specialize the recorded parameter prefix. The six-field induction checks remain unchanged.

**Example (12_check_zero.txt, row 1):**

```text
(in3[i1,v1,i0,*])	main	vacuous truth	(in2[v1,i0,s])	main	!(in2[v1,i0,s])	main	(in[i1,N])	main
```

Read: the head `(in3[i1,v1,i0,*])` is vacuously true because the premise `(in2[v1,i0,s])` and its negation `!(in2[v1,i0,s])` are both derivable under recursion hypothesis `(in[i1,N])` — i.e. the premise is inconsistent with the hypothesis, and the trace chain confirms at least one contradicting side is derived from the hypothesis itself.

---

## `validity name`

**Purpose.** Declares the `validityName` (implication-local scope) to which an integrated expression belongs, so the integrator can bind its conclusion back into the correct hypothetical context.

**Dependency.** The one cited statement is the head of the integration's expansion, the goal the subproof placed at its scope. When a class rewrite respelled that goal before it closed, the head is reached from the closing statement by an `equality1` row at the subproof scope ([D-557](../40_decisions.md#d-557)).

**Scope.** The row lands at the subproof scope's parent, the scope that opened the subproof: `main` for every goal-carrying subproof, the opening scope for a bare subproof opened below `main` (the contradiction-scope demand opens them inside `contradiction_` scopes). The subproof's proof may use every fact of its opening scope, so its compact holds there and below, never above ([D-562](../40_decisions.md#d-562), [I-332](../30_invariants.md#i-332)).

**Checker.** `check_validity_name` at [`verifier.py`](../../verifier.py).

---

## `variable copy`

**Purpose.** A fresh `_copy` duplicate of an existing variable was introduced, to keep a hypothesis variable distinct from its surrounding scope during case analysis or OR branching. Subsumes the retired `reaction to hypo` and `necessity for equality (hypo)` tags. A twin weak step block also uses the tag for `(=[q,rec2])`, which names the twin partner's predecessor q `rec2`; the checker accepts `rec2` as the alias when the chapter declares it for one source only ([D-581](../40_decisions.md#d-581)). Every row holding the copy name traces back to the copy; the walk also ends at a row whose source is an integration goal holding the name, since a goal is filed without a derivation.

**Checker.** `check_variable_copy` at [`verifier.py`](../../verifier.py).

---

## Non-checker categories

### `anchor handling trace`

Per-step chain of `_copy` variable rewrites produced by `anchor handling`. Counted but not validated. Emitted as separate rows following the `anchor handling` row.

### `origin`

Provenance chain for a `contradiction`'s derivation tree. Counted but not validated.

### `self-reference`

Error counter. Increments each time a chapter cites its own theorem as a justification (via a `theorem` tag row whose expression matches the chapter's target). See [`verifier.py–2649`](../../verifier.py).

A non-zero `self-reference` count in the final tally is always a bug — it means a theorem is proving itself in a circle.

---

## Retired tags (kept as `origin.first` labels, not in chapter rows)

- `reaction to hypo`, `necessity for equality (hypo)`, `reformulation for integration` (old umbrella) — retired. Subsumed by `variable copy` and the specific `reformulation for integration …` tags.

## Internal-only labels (live in `exprOriginMap`, never in chapter rows)

- `hypothesis` — terminal provenance (one tag, zero dependencies) of a hypothesis-scope constituent deposited by `disintegrateExprHypothetically` via internal mail, written at the `_hypo_` validity only. Hypothesis constituents steer proof direction but are not part of any proof; a `buildStack` emission assert guarantees the label never reaches a chapter row, so the verifier has no checker for it by design. See [D-259](../40_decisions.md#d-259).

History (D-35, 2026-05-03). The tags `or branch proven` and `or branch assumption` were previously listed here as "retired but still written internally, overridden before export". The override never existed. FTA-rung-1 chapter rows carried the live tags, the verifier did not register them in `TAG_CHECKERS`, and they showed up as `<unknown:…>` failures in incubator runs. D-35 promotes both tags to first-class `TAG_CHECKERS` entries with full structural validation; see the per-tag sections above.

C++ cleanup of the genuinely retired (`reaction to hypo`, `necessity for equality (hypo)`, `reformulation for integration` umbrella) `origin.first` labels is pending. They do not appear in chapter rows so they do not affect verification.

---

## `reformulation for integration flattened`

An additional root integration rule whose definition tree is unfolded through multiple levels. The one dependency is the same-scope compact with its `_integration_goal` suffix. Generated conjunction/existence wrappers dissolve; configured containers may remain intact or expand; all other leaves keep their polarity. `verifier.py` and `check_reformulation_for_integration_flattened` require the exact resulting premise multiset and injective witness renaming; a premise the definition tree reaches twice (an existence's binder premise that is also the first conjunct of the bundle it binds) counts once, as the prover names it once. `proof_export/flat_integration.py` constructs a certificate of premise positions and witnesses for explicit Lean replay. See [D-352](../40_decisions.md#d-352). A disjunct's flat rule ([D-508](../40_decisions.md#d-508)) cites `<disjunct>_integration_goal` and binds the disjunct's open arguments, which stand at argument positions of its head: both checkers accept the head equal to the template up to an injective renaming of those binders onto template arguments the rule does not otherwise use, and the construction builds the head itself.

### Compact-first integration rows

Since [D-503](../40_decisions.md#d-503) every integration rule is installed from a compiled compact `(implication<N>[…])`, and its history states the compact first: the compact takes the integration tag and the citation the rule took, and the rule follows as an `expansion` row citing the compact. After a class merge the canonical compact follows by `equality1` and its rule by `expansion`. An OR introduction serving such a rule states its own compact and cites the rule's compact by `expansion`.

```text
(implication<N>[a,b])	main	reformulation for integration and	(&(p[a,b])(q[b,a]))_integration_goal	main
(>[](p[a,b])(>[](q[b,a])(and<M>[a,b])))	main	expansion	(implication<N>[a,b])	main
```

The verifier checks a compact row on its expansion through the binary entry (`verifier.py::_integration_rule_line`); the certificate proves the expansion in a derived step and folds it into the compact (`proof_export/integration_compact.py`, action `integration_compact`).


## Weaknesses

### Known & tracked

- **the project conventions narrative count was 27; actual is 28 distinct + 29 registry entries.** Documented in [`10_pipeline/08_verifier.md`](../10_pipeline/08_verifier.md) and [`40_decisions.md`](../40_decisions.md).
- **Dead `equalize variable` key.** Kept for symmetry; could be removed, but no one has. Harmless.

### Suspected fragility

- **Tag-string typos.** The prover emits tags as string literals at their emission sites. A typo (e.g. `"implicaton"`) would be unrecognised by the verifier — row would fall through to "anomalous tag" and be counted separately in the final report. No compile-time check prevents this.
- **Per-tag checker lifecycle.** Each checker function is 20–200 lines of Python, independent of the others. Adding a new tag requires (a) emission site in prover, (b) `process_proof_graphs.py` handling, (c) new checker + registry entry, (d) HTML-export rendering. Four coordinated touch points with no umbrella test.

### Not exercised by tests

- **Per-tag regression.** No unit test per `TAG_CHECKERS` entry. A regression in a single checker is caught only by the next full-batch verifier pass on a chapter that exercises it.

---

## See also

- [`10_pipeline/08_verifier.md`](../10_pipeline/08_verifier.md) — verifier flow + TAG_CHECKERS registry.
- [`10_pipeline/06_process_proof_graph.md`](../10_pipeline/06_process_proof_graph.md) — chapter row format.
- [`10_pipeline/07_html_export.md`](../10_pipeline/07_html_export.md) — per-tag rendering.
- [`02_glossary.md`](../02_glossary.md) — each tag has a quick-reference entry.

## `implication composition`

Records an additional implication obtained from a complete producer premise chain and one unary consumer. The result retains all producer premises, in the producer's order or in a permutation of it: the compiler identifies rules up to the order of their premises, so a composed rule can be stored with the chain permuted ([D-509](../40_decisions.md#d-509)). Each producer premise is paired with one result premise of the same sign, operator and arity; the pairings are tried in lexicographic order, the identity first, and the first under which all ports unify is recorded as `premise_order`. The four dependency cells are the producer expression and its validity, followed by the consumer expression and its validity. The intermediate atoms match with a bijection of both complete binder sets; both source binders map onto the complete result binder. Bound-to-fixed specialization and binder collapse are rejected. Repeated arguments, polarity and fixed parameters remain unchanged. The result uses the deeper of the comparable source scopes. Normal origin checking requires both source derivations, either local rows or main-scope citations resolved against the theorem registries. Both registered source citations contribute theorem-usage edges; an unresolved source is an origin failure.

Producers: `logical_block/unary_integration_processor.cpp`, `drainUnaryComposition` (premises always in order), and `logical_block/disintegration_processor.cpp`, `collectContrapositionProducts` (the composed supporting rule of a negated-conjunction premise, expanded from its compiled entity, possibly permuted). Checker: `verifier.py`, `check_implication_composition`. `proof_export/implication_composition.py`, `build_implication_composition_plan`, records explicit source instantiations and `premise_order`; the `implication_compose` action in `proof_export/lean.py`, `_render_chapter_steps`, introduces every result binder group and premise in the result's order, applies each producer premise to the result premise `premise_order` pairs it with, then applies the unary consumer. The composition row is also written to outgoing history through `Memory::addMailOutOrigin`, because the installed rule has no statement delta. Firing the installed result remains an ordinary `implication` row.

## `implication grouping`

The result is a single-premise implication with a conjunction head. Every dependency pair names a contributing unary rule and its scope. All complete binder sets align bijectively through the common signed premise; fixed parameters and repeated ports remain rigid. The head uses each cited conclusion exactly once, following its actual compiled conjunction structure. The result scope is the deepest comparable source scope. This is an auxiliary derived rule, not a supplied conjecture or additional defining element.

Producer: `compiler/supporting_rule_compiler.cpp`, `collectContrapositionProducts`. Checker: `verifier.py`, `check_contraposition_auxiliary`. The certificate records explicit substitutions and a conjunction-construction tree through `proof_export/contraposition.py`, `build_grouping_plan`; Lean applies every cited rule to the same premise and constructs that tree. Ordinary origin checks remain mandatory.

## `contraposition`

The result reverses one cited unary implication and complements both ends, cancelling double negation. Its sole dependency pair names the exact original or grouped source and its scope. The complete binder set must occur at both source ends and map bijectively to the result. Signs, repetitions and fixed parameters are checked. The result retains the source scope.

Producer and checker are the same auxiliary doors as grouping. `proof_export/contraposition.py`, `build_contraposition_plan`, supplies the explicit source arguments; Lean applies that source under the opposite assumption and derives a contradiction. No new provenance exemption, theorem axiom or proof-admission shortcut is introduced.

## `definition projection`
Processed-only row: `target`, namespace, `definition projection`, source, the same namespace. The source must be an actual row of the chapter. `proof_export/definition_projection.py`, `build_definition_projection_plan`, admits only positive named conjunction unfolding and binary conjunction projection. Parameter instantiation is capture-free; lexical bound-name comparison preserves every free argument. Implications, negations, disjunctions and existence definitions are opaque: a whole such proposition can be a conjunct, but this rule never enters it or introduces a witness.

The GL verifier's `check_definition_projection` checks the emitted row against its own chapter registry, preserving all normal origin, cycle and goal checks. The neutral certificate records each definition head, constituent index and primitive projection path. The Lean writer rechecks that plan and emits a direct product-projection term through transparent definitions. The planner is shared structural code; the Lean kernel independently checks the resulting proof term. No discarded original derivation is a proof input. Expanded definition/projection operations are reported alongside physical rows.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
