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

Every chapter row is tab-separated:

```text
expression \t namespace \t tag \t [rest fields...]
```

Rest fields **alternate** `(expression, namespace)` pairs — i.e. `rest[0]` is an expression, `rest[1]` is its namespace, `rest[2]` is the next expression, `rest[3]` its namespace, and so on. The verifier reads them at `rest[i], rest[i+1]` strides (see `verifier.py::check_equality1` at line 1517 for a canonical consumer). Ignoring this alternation when authoring a new origin — e.g. writing `rest=[src, eq1, eq2]` without the interleaved `ns` fields — makes the verifier silently reject every emission with that tag (odd `rest` count → invalid shape). Every tag's checker indexes the rest fields in pairs; the per-tag sections below state which specific `(expr, ns)` slot encodes which role.

---

## Tag index

30 distinct tags + 30 `TAG_CHECKERS` registry entries (one entry per tag — no shared-checker alias; `equalize variable` was removed, only `multiplied from` is emitted). `compilation` was added (briefly 31 + 31); `mirrored from` was removed (D-112) — see [`10_pipeline/08_verifier.md`](../10_pipeline/08_verifier.md). Sorted alphabetically here; the verifier iterates in registry order.

- [`anchor handling`](#anchor-handling) — pin raw bound-variable index to anchor-slot name.
- [`compilation`](#compilation) — implication compiled to its compact `(implication<N>[…])` named form (ASIC-prep provenance).
- [`contradiction`](#contradiction) — proof by contradiction.
- [`disintegration`](#disintegration) — split compound expression into elements.
- [`equality1`](#equality1) — argument substitution via equality.
- [`equality2`](#equality2) — transitivity of equality.
- [`equalize variable`](#equalize-variable--multiplied-from) — (shared with `multiplied from`; dead alias).
- [`expansion`](#expansion) — named expression rewritten to compiled form.
- [`expansion for integration`](#expansion-for-integration) — mirror of expansion on integration side.
- [`externally provided theorem`](#externally-provided-theorem) — from `externally_provided_theorems.txt`.
- [`implication`](#implication) — compiled implication rule fired.
- [`incubator back reformulation`](#incubator-back-reformulation) — incubator operator-equality rewritten.
- [`multiplied from`](#equalize-variable--multiplied-from) — re-emitted with bound vars identified per Bell partition.
- [`or branch assumption`](#or-branch-assumption) — negated-other-disjunct seeded as branch-local fact.
- [`or branch proven`](#or-branch-proven) — OR case-split into a branch carrying the asserted disjunct.
- [`or convergence`](#or-convergence) — all branches reached the same conclusion.
- [`or disintegration`](#or-disintegration) — case split on an OR head.
- [`or theorem`](#or-theorem) — an OR-shaped theorem was reached.
- [`premise element`](#premise-element) — one specific premise cited during integration.
- [`recursion`](#recursion) — induction-hypothesis step.
- [`reformulated from`](#reformulated-from) — this theorem is a reformulation of the cited source.
- [`reformulation for integration and`](#reformulation-for-integration-and) — reverse-disintegration prep for an `and`-category conclusion.
- [`reformulation for integration >[]`](#reformulation-for-integration-) — for an `existence` with empty outer bounds.
- [`reformulation for integration >[bound]`](#reformulation-for-integration-bound) — for an `existence` with non-empty outer bounds.
- [`symmetry of equality`](#symmetry-of-equality) — `(=[b,a])` from `(=[a,b])`.
- [`symmetry of inequality`](#symmetry-of-inequality) — `!(=[b,a])` from `!(=[a,b])`.
- [`task formulation`](#task-formulation) — a root premise of the theorem under proof.
- [`theorem`](#theorem) — a previously-proven theorem fired as an inference rule.
- [`vacuous truth`](#vacuous-truth) — the premise chain is self-contradictory.
- [`validity name`](#validity-name) — declares scope for an integrated expression.
- [`variable copy`](#variable-copy) — fresh `_copy` duplicate of an existing variable.

**Non-checker categories** — counted by the verifier, not validated:

- `anchor handling trace` — per-step `_copy` substitution chain from `anchor handling`.
- `origin` — provenance chain for a `contradiction`'s derivation tree.
- `self-reference` — error counter; increments when a chapter cites its own theorem.

---

## `anchor handling`

**Purpose.** Pin a raw bound-variable index in an anchor application to its anchor-slot name. At most one emission per chapter; a trace of subsequent `_copy` substitutions follows.

**Checker.** `check_anchor_handling` at [`verifier.py`](../../verifier.py).

**Row shape.**

```text
<rewritten-anchor-expr>  main  anchor handling  <original-anchor-expr>  main
```

**Example (0_direct_proof.txt):**

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

**Prover-side emission.** The eight `updateGlobalDirect` / `updateGlobal` broadcast sites (plus the two formerly-backup-less producers — `broadcastTheorems` load-time and `addExprToMemoryBlock` recovered-implication re-broadcast — per [D-83](../40_decisions.md#d-83)) call `recordPendingCompaction`; the single-threaded post-`pool.join` drain compiles each via `ExpressionAnalyzer::compileImplicationToCompact` and emits the `compilation` paired origin (gated on `parameters.trackHistory`; also discharges the receiver's [D-45](../40_decisions.md#d-45) paired-origin requirement). On the legacy `mailOut.implications.insert` sites are retired with the channel itself ([D-78](../40_decisions.md#d-78)); only the compact-form deposit on `mailOut.statements` remains. See [`03_mail_system.md`](03_mail_system.md#additive-compact-implication-deposit-asic-01-prep-d-76) and the [prover chapter](../10_pipeline/04_prover.md#compileimplicationtocompact).

---

## `contradiction`

**Purpose.** Under an assumption of the opposite of the conclusion, the prover derives both `X` and `!X`. The assumption discharges; the original conclusion is emitted.

**Emitted by.** `dischargeContradiction` ([`prover.hpp`](../../GL_Quick_VS/GL_Quick/src/prover.hpp)) — the single per-step sweep over `intEncodedStatements` that detects an in-scope contradiction and fires the discharge. (Formerly the "Site G" block inside `addExprToMemoryBlock`; see the [prover chapter](../10_pipeline/04_prover.md#contradiction-discharge).)

**Checker.** `check_contradiction` at [`verifier.py`](../../verifier.py).

**Rest fields.** Origin chain — references the two contradicting expressions and the discharged assumption.

**Context.** Each contradiction step has an associated `origin` (non-checker) category entry recording the derivation tree. The verifier's origin-trace loop at [`verifier.py–2706`](../../verifier.py) walks this chain.

---

## `disintegration`

**Purpose.** A compound compiled structure was split into its constituent elements — each conjunct of an `and`, each side of an `existence` after witness binding.

**Checker.** `check_disintegration` at [`verifier.py`](../../verifier.py).

**Row shape.**

```text
<element>  main  disintegration  <source-compound>  main
```

**Example (from 12_check_zero.txt):**

```text
(implication4[N,i0,s])	main	disintegration	(&(&(&(&(&(&(&(&(&(&(&(&(in[i0,N])(fXY[s,N,N]))(implication4[N,i0,s]))(implication5[N,s]))(fXYZ[+,N,N,N]))(implication11[N,i0,+]))(implication12[N,i0,+]))(implication13[N,s,+]))(implication14[N,s,+]))(fXYZ[*,N,N,N]))(implication15[N,i0,*]))(implication16[N,s,*,+]))(implication17[N,s,*,+]))	main
```

**Validation.** The LHS must be a constituent of the RHS according to the GL binary's `elements` list for the RHS's named expression.

---

## `equality1`

**Purpose.** Argument substitution — given `(=[a,b])` and any expression `E(a)`, emit `E(b)`.

**Checker.** `check_equality1` at [`verifier.py`](../../verifier.py).

**Rest layout.** Exactly the interleaved `(expr, ns)` pattern from the row-shape recap:

| Slot | Role | Contents |
|---|---|---|
| `rest[0]` | `source_expr` | The pre-substitution expression. Same core + arity as the row's result expression. |
| `rest[1]` | `source_ns` | Namespace of the source. Must equal the row's own namespace — substitution stays in scope. |
| `rest[2i]`, `i ≥ 1` | equality `i` | One `(=[a,b])` justifying one substituted arg position (orientation source→result: `(source_arg, result_arg)` must be in the equality set). |
| `rest[2i+1]` | equality `i` namespace | Each equality lives either in the row's namespace OR a strict byte-level-prefix (ancestor) namespace — see `_ns_matches_or_strict_prefix` at `verifier.py`. Branch-scope results can cite ancestor-scope equalities. |

**Minimum `len(rest)` = 4** (source + one equality). Shorter → the checker returns `False`.

**Validation contract.**

1. `source_core == result_core` AND `arity(source) == arity(result)`.
2. For every arg position `i` where `source_args[i]!= result_args[i]`: the tuple `(source_args[i], result_args[i])` must be present in the assembled equality set.
3. No covering equality found for *any* differing position → row rejected.

**Prover-side emission sites.**

- `applyEquivalenceClass` at `prover.hpp` — the main eq-class rewrite over `intEncodedStatements`. `rest[0]` is the original encodedStatement before args were mapped via the current class.
- `applyEquivalenceClassToRejectedMapIntegration` (integration-revival path, see [`05_equivalence_classes.md`](05_equivalence_classes.md#extension--rejectedmapintegration)) — emits `rest[0]` = the pre-rewrite form of the emitted constituent (NOT the compound, NOT the post-rewrite form). Common authoring mistake: passing the compound in `rest[0]` — the arity check then fails because the constituent and the compound have different cores.

**Common authoring pitfalls.**

- Passing the POST-rewrite form as `rest[0]` — arity matches but substitution check yields `0!= 0`, so no equality is needed, the row passes trivially and the eq is never validated. Subtle soundness leak if the "eq" entries aren't actually equalities. Always pass PRE-rewrite as source.
- Using a different namespace for `source` than for `result` — the very first check in `check_equality1` (`line.namespace!= source_ns → False`) rejects the row. Substitution does not cross scopes.
- Forgetting the `ns` interleave — `len(rest) < 4` or odd-length `rest` is silent failure.

---

## `equality2`

**Purpose.** Transitivity of equality — from `(=[a,b])` and `(=[b,c])`, emit `(=[a,c])`.

**Checker.** `check_equality2` at [`verifier.py`](../../verifier.py).

**Rest fields.** The two transitively-related equalities.

---

## `equalize variable` / `multiplied from`

**Purpose.** A parent theorem is re-emitted with some of its bound variables identified according to a Bell partition — `multiplyImplication` at [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp). Partitions that would merge two distinct free `u_*` anchor parameters into the same equivalence class are skipped (see [prover chapter](../10_pipeline/04_prover.md#multiplyimplication-bell-partition-equalisation) and [I-24](../30_invariants.md#i-24)).

**Checker.** `check_equalize_variable` at [`verifier.py`](../../verifier.py) — **shared** by both tag keys.

In addition to the slot-by-slot consistency check, the checker enforces a free-anchor-merge guard: it parses the bound-variable list from every `>[...]` binder of the source and copy implications, builds the orig→copy arg mapping over the paired premises and head, and rejects any pair where the orig arg is not bound by the source binders, the copy arg is not bound by the copy binders, and the two names differ. Bound→bound (Bell-partition merge) and bound→free (bound-variable specialisation to a known anchor) remain accepted — the rejection fires only on free-to-free remaps with distinct names, which is the chapter-1115 multiplyImplication soundness violation that the prover-side gate also rejects.

**Note.** Only `multiplied from` is ever emitted in current code. The `equalize variable` key is a dead alias kept in the registry for symmetry.

**Rest fields.** The source (unmultiplied) theorem + the partition.

---

## `expansion`

**Purpose.** A named compiled expression (`and` / `existence` / `implication` / `or<N>` / `existence<N>`) is rewritten into its GL-binary `elements` form.

**Checker.** `check_expansion` at [`verifier.py`](../../verifier.py).

**Example (0_direct_proof.txt):**

```text
(NaturalNumbers[N,i0,s,+,*])	main	expansion	(AnchorPeano[N,i0,s,+,*,i1])	main
```

Wait — this row is actually `(NaturalNumbers[...])` being the *result* of expanding `(AnchorPeano[...])`'s `elements` list, per the anchor's body `(&(NaturalNumbers[...])(in2[i0,i1,s]))`. The checker consults the GL binary to verify this expansion.

**Validation.** LHS must appear in RHS's `elements` list per the GL binary.

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

**Example (0_direct_proof.txt, first row):**

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

The C++ prover's hash-kernel ([`generateEncodedRequestsStatic` + `growBaseCandidates`](../../GL_Quick_VS/GL_Quick/src/memory.cpp)) accumulates the joined-scope of combined facts via `nm.deeperOf(...)` — the result of an implication firing lives at the deepest scope of its inputs, never at a strictly deeper scope no constituent reaches. The verifier mirrors this. See [I-38](../30_invariants.md#i-38) and [D-58](../40_decisions.md#d-58) for the full rule + sound-under-validity-stack-semantics rationale.

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

**Purpose.** When an OR is case-split into per-branch scopes, every branch where disjunct `D_i` is asserted gets the negation `!D_j` of every other disjunct (`j ≠ i`) seeded as a branch-local fact. Each such seeding emits one `or branch assumption` row.

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
6. **A matching `or branch proven` row exists (Codex round-3).** Some chapter row with `tag == "or branch proven"`, `expression == or_expr`, `namespace == parent_ns`, `len(rest) == 2`, `rest[1] == branch_ns`, and `rest[0]` matching the asserted disjunct (modulo equality symmetry). Without this check the assumption row could pass structurally even when the corresponding case-split was never opened.

History: pre-D-35 the tag was claimed retired (overridden before export). The override never existed; the prover (`prover.hpp` `or branch assumption` site) emits this tag live, and FTA-rung-1 chapter `1209_direct_proof.txt` line 43 carries it. Iterations: round-1 added the checker; round-2 tightened to `len(rest) == 2`, exact-equality branch-namespace match, and balanced-parens disjunct parsing; round-3 added the matching-`or branch proven` cross-check. Chapter 1209's `or branch assumption` row continues to pass after round-3 — the matching `or branch proven` row at line 22 satisfies all five sub-conditions.

---

## `or branch proven`

**Purpose.** Records the case-split itself: an OR `(or<N>[…])` was opened into a per-branch scope carrying one specific disjunct as its asserted seed.

**Checker.** `check_or_branch_proven` at `verifier.py` (D-35).

**Row layout (exactly two rest fields).**

```text
<or-expr>  <parent-ns>  or branch proven  <asserted-disjunct>  <branch-ns>
```

**Validation (strict per Codex round-2; round-3 check #1 dropped per [D-36](../40_decisions.md#d-36)):**

1. `len(rest) == 2` exactly. The tag is in `_ORIGIN_EXEMPT_TAGS`, so any extra rest pairs would be silently accepted by the generic origin check; reject up front so the row's contract stays auditable.
2. `line.expression` is a known compiled OR with ≥2 disjuncts after `u_i` substitution against the OR's args (and matching arity per the binary's `signature`).
3. `rest[0]` is one of those disjuncts (modulo equality symmetry).
4. `rest[1]` is **EXACTLY** `parent + "_boundary_orint_" + or_expr + "_(" + <disjunct> + ")"` for `<disjunct>` matching `rest[0]` (modulo equality symmetry). No substring search — the row's claim is "this immediate-child subproof", not "some descendant that contains the substring".

**About the missing step 5 ([D-36](../40_decisions.md#d-36)).** An earlier Codex round-3 step required a non-`or branch proven` derivation row for the OR at parent scope. That check was based on a wrong mental model of `_orint_` (treated it as case-split with a separately-derived OR). Correct semantics: `_orint_` rewrites the OR goal `(A ∨ B)` into two sub-implications-to-prove `(!A → B)` and `(!B → A)`; when one fires, the `or branch proven` row IS the OR's derivation by design — there is no separate derivation row to look for. The check was unsatisfiable for legitimate proofs (e.g. chapter `1209_direct_proof.txt` line 22) and dropped as a verifier correction, not a relaxation.

**Terminology note ([D-36](../40_decisions.md#d-36)).** "branch" in this tag's name and in `or branch assumption` is historical — these rows record per-SUBPROOF events, not case-split branch events. A "subproof" here is one of the two `(!A → B)` / `(!B → A)` implications-to-prove that `_orint_` opens. The renaming hasn't happened yet to avoid touching every consumer.

History: pre-D-35 the tag was claimed retired (replaced by `or theorem`). `or theorem` is a different tag — it tags theorems whose head is an OR shape. `or branch proven` is the subproof-firing record, emitted live by the prover (`prover.cpp` `or branch proven` site) and first-class in the verifier since D-35. Iterations: round-1 added the checker; round-2 tightened to `len(rest) == 2` and exact-equality namespace match; round-3 added an OR-origin requirement (later dropped by D-36 as a correction — see above).

---

## `or convergence`

**Purpose (mathematical contract).** An `or convergence` row certifies that the conclusion `line.expression` was independently derived in EVERY branch of the case split on the OR cited in `rest[0]`, and is therefore promoted to the OR's parent scope cited in `rest[1]`.

**Checker.** `check_or_convergence` at [`verifier.py`](../../verifier.py) — validates the spec'd row layout. As of [D-36](../40_decisions.md#d-36) the producer side (`ordisMerge` in `prover.hpp`) emits the new layout, and chapter `1209_direct_proof.txt`'s convergence rows pass cleanly.

### Row layout (compiled-form OR only)

```text
<C>  <parent>  or convergence  <OR>  <parent>
                                      <C>  <branch_D1>
                                      <C>  <branch_D2>
                                      …
                                      <C>  <branch_DK>
```

For an OR with `K` disjuncts, the rest field has `2 + 2*K` slots:
- `rest[0]` = OR (compiled `(or<N>[…])`)
- `rest[1]` = parent (the OR's parent scope; must equal `line.namespace`)
- `rest[2*i + 2]` = `C` (must equal `line.expression` for every `i`)
- `rest[2*i + 3]` = `branch_Di` (the i-th branch's namespace)

### Validation contract

For the row to PASS, all of:

1. **Layout.** `len(rest) == 2 + 2*K`; `len(rest)` is even; `len(rest) >= 6` (so `K >= 2`).
2. **Parent-scope match.** `line.namespace == rest[1]`.
3. **OR is real.** `rest[0]` is a compiled `(or<N>[…])` with a known GL-binary entry of `category == "or"` and `>= 2` elements; the disjunct count `K` matches the number of `(C, branch_Di)` pairs.
4. **Conclusion repetition.** `rest[2*i + 2] == line.expression` for every `i` in `[0, K)`.
5. **Branch-scope ancestry.** Each `branch_Di` is a strict descendant of `parent` (`branch_Di.startswith(parent + "_boundary_")`).
6. **Branch distinctness.** The `K` branch namespaces are pairwise distinct.
7. **Per-branch derivation evidence ("each ingredient has its own line").** For every `(C, branch_Di)` pair, a chapter row exists with `expression == C` and `namespace == branch_Di` under any tag — proves `C` was derived at that branch scope.

Step 7 is the key strengthening per the user's directive — it ties each cited branch to a real chapter-local derivation of `C`. Without it the rest fields would be unverifiable assertions; with it the verifier reduces convergence-checking to standard chapter-row existence checks (the same pattern used by the inline origin check).

### Status (D-36)

Chapter `1209_direct_proof.txt`'s 2 convergence rows now pass cleanly. The producer side (`ordisMerge` in `prover.hpp` at the post-D-36 emission site) writes the spec'd layout: `(C, parent, or convergence, OR, parent, C, branch_D1, C, branch_D2)`. Each branch's derivation of `C` remains in `exprOriginMap` (because `removeExpressionFromMemoryBlock(state=0)` touches only `intEncodedStatements`), so chapter export's `buildStack` naturally renders the per-branch rows when it recurses on the new ingredients.

A side-effect of the producer fix: 4 previously-hidden `or disintegration` rows now appear in chapter 1209 (lines 17, 21, 29, 33) — the `_ordis_` per-branch case-split records that were never reached by `buildStack` before. They use the same compiled-OR row layout and were verified by [D-36-extended `check_or_disintegration`](#or-disintegration-d-36-extended-checker).

---

## `or disintegration`

**Purpose.** Case split on an `or` head. Each disjunct becomes a sub-goal inside a dedicated branch scope. The branch scope is named with the asserted disjunct in its payload (`_boundary_ordis_<or>_(<disjunct>)`). Sibling row to `or branch proven` — the `_ordis_` (case-split-and-converge) counterpart of `_orint_` (sub-implications-to-prove).

**Checker.** `check_or_disintegration` at [`verifier.py`](../../verifier.py) — D-36-extended for compiled-OR form + per-branch namespace check + OR-origin requirement.

**Row layout (exactly two rest fields).**

```text
<asserted-disjunct>  <branch-ns>  or disintegration  <OR>  <parent-ns>
```

**Validation (D-36):**

1. `len(rest) == 2` exactly. Same `_ORIGIN_EXEMPT_TAGS` rationale as the other OR-tag checkers.
2. `rest[0]` is a known compiled OR `(or<N>[…])` with ≥2 disjuncts via GL-binary lookup (matching arity per the binary's `signature`).
3. `line.expression` is one of those disjuncts (modulo equality symmetry).
4. `line.namespace` is **EXACTLY** `rest[1] + "_boundary_ordis_" + rest[0] + "_(" + <disjunct> + ")"` for `<disjunct>` matching `line.expression` (modulo equality symmetry). No substring search.
5. **The OR has an independent derivation row at parent scope.** A chapter row exists with `expression == rest[0]`, `namespace == rest[1]`, and `tag!= "or disintegration"` — i.e. the OR was actually derived (via `implication`, `expansion`, `theorem`, …) before being case-split. This check IS well-founded for `_ordis_`: case-split CONSUMES an existing OR. The analogous check on `check_or_branch_proven` was dropped per [D-36](../40_decisions.md#d-36) because `_orint_` PRODUCES an OR — no separate derivation exists by design. This is the exact semantic asymmetry that motivated the round-3 correction.

Pre-D-36 the checker accepted only the expanded `!(&!(…))` form in `rest[0]` and never validated namespace structure or OR-origin. It was effectively dead in the incubator path because chapter export didn't render `_ordis_` per-branch rows. Stage P1's `ordisMerge` extension exposed them; Stage P1b tightened the checker.

---

## `or theorem`

**Purpose.** An OR-shaped theorem was reached as a goal — `line.expression` is the proven theorem and its head is an `or<N>[…]` node. Distinct from [`or branch proven`](#or-branch-proven), which is the per-branch case-split bookkeeping row inside a proof; `or theorem` is the chapter-conclusion record for theorems whose statement IS an OR.

**Checker.** `check_or_theorem` at [`verifier.py`](../../verifier.py).

---

## `premise element`

**Purpose.** During integration, one specific premise of the source implication is called out as a dependency of the integrated step.

**Checker.** `check_premise_element` at [`verifier.py`](../../verifier.py).

---

## `recursion`

**Purpose.** Induction-hypothesis step.

- In a `check_zero` chapter: the induction variable is identified with `i0`.
- In a `check_induction_condition` chapter: the successor form is asserted as the inductive step's premise.

**Checker.** `check_recursion` at [`verifier.py`](../../verifier.py).

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

**Checker.** `check_symmetry_of_equality` at [`verifier.py`](../../verifier.py).

---

## `symmetry of inequality`

**Purpose.** From `!(=[b,a])`, emit `!(=[a,b])`.

**Checker.** `check_symmetry_of_inequality` at [`verifier.py`](../../verifier.py).

**Note.** Missing from the project conventions's narrative tag list. This is the 28th distinct tag. Confirmed via direct verifier-source inspection.

---

## `task formulation`

**Purpose.** A root premise of the theorem under proof, asserted without justification at the top of the chapter.

**Checker.** `check_task_formulation` at [`verifier.py`](../../verifier.py).

**Example (end of 0_direct_proof.txt):**

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

**Example (0_direct_proof.txt, row 2):**

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

**Scope.** Currently confined to scope `"main"` —.

**Rest layout (6 fields, was 4 before commit ).** As of the vacuous-truth soundness tightening:

| Slot | Role |
|---|---|
| `rest[0..1]` | contradicting expression + namespace |
| `rest[2..3]` | its negation + namespace |
| `rest[4..5]` | the recursion-hypothesis expression + namespace (NEW) |

**Emitted by.** `dischargeContradiction` ([`prover.hpp`](../../GL_Quick_VS/GL_Quick/src/prover.hpp)) — the single per-step sweep over `intEncodedStatements`; the vacuous-truth branch fires on `isPartOfRecursion` + `validityName == "main"` + an ancestor-scope contradiction. See the [prover chapter](../10_pipeline/04_prover.md#contradiction-discharge).

**Prover-side: no level gate.** The vacuous-truth discharge has **no** `mb.level` gate. An earlier gate (requiring at least one contradicting ingredient to carry `mb.level` in its per-statement level set) was removed: for theorems whose own premise is impossible (chapter-101-style lemmas) the contradiction is necessarily rooted in the theorem's outer premise, not the recursion step's hypothesis, so an `mb.level` gate would reject those legitimate vacuous-truth cases. Soundness rests on axiom consistency — no chapter-level contradiction can trace only to anchor-level facts.

**Verifier-side chapter-local trace (commit, `verifier.py` + new counter `vacuous truth trace`).** `check_vacuous_truth` now also walks the chapter-local origin chain from each ingredient; at least one ingredient must trace back to the recursion-hypothesis cited in `rest[4..5]`. Reuses `_trace_back_to` following the contradiction-trace pattern. A separate counter row `vacuous truth trace` reports trace success/failure — when an ingredient fails to reach the hypothesis, the LB's vacuous-truth claim is unsound.

**Example (12_check_zero.txt, row 1):**

```text
(in3[i1,v1,i0,*])	main	vacuous truth	(in2[v1,i0,s])	main	!(in2[v1,i0,s])	main	(in[i1,N])	main
```

Read: the head `(in3[i1,v1,i0,*])` is vacuously true because the premise `(in2[v1,i0,s])` and its negation `!(in2[v1,i0,s])` are both derivable under recursion hypothesis `(in[i1,N])` — i.e. the premise is inconsistent with the hypothesis, and the trace chain confirms at least one contradicting side is derived from the hypothesis itself.

---

## `validity name`

**Purpose.** Declares the `validityName` (implication-local scope) to which an integrated expression belongs, so the integrator can bind its conclusion back into the correct hypothetical context.

**Checker.** `check_validity_name` at [`verifier.py`](../../verifier.py).

---

## `variable copy`

**Purpose.** A fresh `_copy` duplicate of an existing variable was introduced, to keep a hypothesis variable distinct from its surrounding scope during case analysis or OR branching. Subsumes the retired `reaction to hypo` and `necessity for equality (hypo)` tags.

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

- `reaction to hypo`, `necessity for equality (hypo)`, `reformulation for integration` (old umbrella) — retired. Subsumed by `variable copy` and the three `reformulation for integration …` variants.

History (D-35, 2026-05-03). The tags `or branch proven` and `or branch assumption` were previously listed here as "retired but still written internally, overridden before export". The override never existed. FTA-rung-1 chapter rows carried the live tags, the verifier did not register them in `TAG_CHECKERS`, and they showed up as `<unknown:…>` failures in incubator runs. D-35 promotes both tags to first-class `TAG_CHECKERS` entries with full structural validation; see the per-tag sections above.

C++ cleanup of the genuinely retired (`reaction to hypo`, `necessity for equality (hypo)`, `reformulation for integration` umbrella) `origin.first` labels is pending. They do not appear in chapter rows so they do not affect verification.

---

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

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
