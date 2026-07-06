<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Pipeline · Stage 1 — Conjecturer `[DRAFT]`

> **Input:** compiled `CoreExpressionConfig` map (built during `gl_quick.exe` startup; definition compilation is no longer a separate stage), `Config<Tag>.json`.
> **Output:** `files/theorems/conjectures.txt` (and auxiliary output files depending on flags).
> **Owner:** `conjecturer.cpp` / `conjecturer.hpp`.
> **Entry:** `Conjecturer::run` at [`conjecturer.cpp`](../../GL_Quick_VS/GL_Quick/src/conjecturer.cpp). Invoked via `gl_quick.exe --conjecture <tag>`.

---

## What this stage does

The conjecturer enumerates candidate theorems by combinatorial assembly. Given the compiled expressions + per-batch config, it:

1. For each "number of simple expressions" (`nse`) in `[min_nse, max_nse]`, combines predicates into candidate bodies.
2. Attaches an anchor at the outermost position (`AnchorPeano[...]`, `AnchorGauss[...]`, …).
3. Applies a cascade of structural filters (type-set consistency, complexity limits, tautology detection, operator-placement rules, forbidden patterns).
4. Reshuffles arguments to canonical form.
5. Generates the reverse-direction mirror of each conjecture and folds it into the prove pool (`conjectures.txt`) so it is proved for real — see step 9.
6. (Optionally) generates OR conjectures from per-expression `allow_to_constitute_existence` config flags via `generateOrConjectures`; writes the emitted pairs to `or_pairs.txt` as an OUTPUT artefact.
7. Writes the survivors to `files/theorems/conjectures.txt`.

The prover then picks up `conjectures.txt` (via the CE filter, stage 3→4) and tries to prove or refute each entry.

---

## The raw output — what `conjectures.txt` looks like

A representative line, **pre-unification** (sparse anchor binder — see the binder-rule change below; this exact byte form no longer ships):

```text
(>[1,2,3,6](AnchorPeano[1,2,3,4,5,6])(>[7]!(=[2,7])(>[](in[6,1])!(>[8](in[8,1])!(in2[8,7,3])))))
```

Decoded:

- Anchor: `AnchorPeano[1,2,3,4,5,6]` — the 6 Peano slots are pinned to bound-variable indices `1..6`. Slot 1 is `N` (the set), slot 2 is `i0` (zero), slot 3 is `s` (successor), slot 4 is `+`, slot 5 is `*`, slot 6 is `i1` (one).
- Inner quantifier: `>[7]` — a fresh bound variable, index 7.
- Body: `!(=[2,7])` AND `(>[](in[6,1])!(>[8](in[8,1])!(in2[8,7,3])))` — read as "if `6 ∈ 1` (a.k.a. `i1 ∈ N`), then NOT "there exists 8 in 1 such that NOT `in2[8,7,3]`" (a.k.a. "for every 8 ∈ N, `in2[8,7,3]` holds"). Combined: given `i0 ≠ 7` and `i1 ∈ N`, for every natural 8, `s(8) = 7`.

This is a negative conjecture — a claim the prover will try to refute.

**Outer anchor binder — unified rule ([I-4](../30_invariants.md#i-4), [I-11](../30_invariants.md#i-11) inverted, [D-75](../40_decisions.md#d-75)).** The outer `>[...]` binds **every** anchor-atom argument in occurrence order, not only the slots the body references. The pre-unification line above bound `>[1,2,3,6]` (slots 4/5 omitted because the conjecture does not mention `+`/`*`); under the unified rule it binds every anchor slot. Verbatim post-unification line (first row of `files/theorems/conjectures.txt`, Gauss batch):

```text
(>[1,2,3,4,5,6,7,8](AnchorGauss[1,2,3,4,5,6,7,8])(>[9,10,11](in3[9,10,11,4])(>[12,13](fold[1,3,4,8,2,12,13])(>[](in2[9,12,3])(in3[10,13,11,4])))))
```

The outer `>[1,2,3,4,5,6,7,8]` binds all eight `AnchorGauss` slots in anchor-atom order (= occurrence order, since the anchor is the first premise); the inner `>[9,10,11]` / `>[12,13]` / `>[]` quantifiers are unaffected. The equivalent Peano shape is `(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7]…))`. Implemented in the anchor-attach binder of `connectExpressions` / `connectExpressionsInt` (both lanes, gated on `connectToAnchor`); the removable-arg / negation-variant / def-set logic is unchanged — only the emitted binder widens, and `reshuffle`'s `orderByPattern` re-derives the same occurrence order for the reshuffled artefacts.

**Numbered-variable form.** In raw `conjectures.txt`, bound variables are bare integers `1`, `2`, … — no `v1` yet. The renaming from integers to `v1`, `v2`, … happens in `process_proof_graphs.py` after the prover runs. See [I-10](../30_invariants.md#i-10).

**Reshuffled form.** After conjecturer emission, `reshuffled_conjectures.txt` holds the argument-normalised canonical forms. Pre-unification representative line:

```text
(>[1,2,3,4](AnchorPeano[1,2,3,6,7,4])(>[]!(=[2,4])!(>[5](in[5,1])!(in2[5,2,3]))))
```

Pre-unification this bound only the body-referenced slots. Under the unified rule every anchor slot is in the canonical outer `>[...]` in occurrence order; the anchor stays pinned at permutation position 0 so the anchor-atom argument IDs are unchanged — only the binder list widens. Verbatim post-unification line (a compact row of `files/theorems/reshuffled_conjectures.txt`, Gauss batch):

```text
(>[1,2,3,4,5,6,7,8](AnchorGauss[1,2,3,4,5,6,7,8])(>[9](interval[1,4,2,6,9])(in[6,9])))
```

---

## Config switches that shape the output

These live in the top-level `Config<Tag>.json` (not per-expression):

| Field | Default / typical | Effect |
|---|---|---|
| `min_number_simple_expressions` | `2` (Peano/Gauss), `1` (some incubator variants) | When set to `1`, enables a preliminary pass that connects individual expressions directly to the anchor, producing `(>[...](Anchor[...])(single_head))` conjectures. Peano/Gauss default skips this. |
| `max_number_simple_expressions` | `3`–`5` typically | Controls growth — whether combined expressions re-enter `growing_theorems`. When `< 2`, the main combination loop is skipped entirely and only `nse=1` output is produced. |
| `max_values_for_def_sets` | `3` typical | Each `(1)`-typed arg generates mapping tables with size depending on this. |
| `max_values_for_uncomb_def_sets` | `3` typical | Same for "uncombinable" `(1)` args. Together with the previous, the product determines memory footprint of `createMapAnchor` (see [OPEN-8](#open-questions)). |
| `max_iteration_number` | per batch | Controls how deep the int-path permutation iterations go. |

And per-expression (inside `"in"`, `"=`", `"in2"`, `"in3"`, …):

| Field | Effect |
|---|---|
| `max_count_per_conjecture` | Cap on copies of this predicate in one conjecture. |
| `allow_negation` | May this predicate appear as `!(...)`? (`=` yes; most others implicitly yes.) |
| `allow_to_constitute_existence` | Can this predicate head an existence? |
| `existence_variable_position` | Which arg position carries the bound variable in an existence node. |

---

## Code-level flow

The entry is `Conjecturer::run` at [`conjecturer.cpp`](../../GL_Quick_VS/GL_Quick/src/conjecturer.cpp). Pseudo-flow:

1. **Configuration load** — `loadConfiguration(anchorId)` at [`conjecturer.cpp`](../../GL_Quick_VS/GL_Quick/src/conjecturer.cpp). Reads `Config<Tag>.json`, resolves the compiled expression map.
2. **Core-expr-map adapter build** — `buildCoreExprMapAdapter` at [`conjecturer.cpp`](../../GL_Quick_VS/GL_Quick/src/conjecturer.cpp). Imports stage-1 compilation output.
3. **Int encoding** — `buildIntExprConfigs` at [`conjecturer.cpp`](../../GL_Quick_VS/GL_Quick/src/conjecturer.cpp). Converts string-keyed expression maps to int-keyed for the hot-path.
4. **Seed `growing_theorems`** with initial expressions.
5. **For each nse from `min_nse` to `max_nse`:**
 - If `nse == 1` and `min_nse == 1`: run `singleExprAnchorConnection` / `singleExprAnchorConnectionInt` (skips combination-filter subset).
 - Else: for each existing conjecture in `growing_theorems`, combine with each expression via `connectExpressionsInt` ([`conjecturer.cpp`](../../GL_Quick_VS/GL_Quick/src/conjecturer.cpp)) and `makeAllConnectionMapsInt` ([`conjecturer.cpp`](../../GL_Quick_VS/GL_Quick/src/conjecturer.cpp)).
6. **Per-combination filter cascade.** Every new candidate passes through:
 - `checkDefSetsInt` / `checkDefSetsPriorInt` — type-set consistency.
 - `checkComplexityLevelInt` / `checkComplexityPerOpInt` — complexity budget.
 - `numbersGoodInt` — bound-variable numbering sanity.
 - `repetitionsExistInt` — de-dup.
 - `prohibitedHeadsGoodInt` / `onlyInHeadGoodInt` — operator-placement rules.
 - `exprGood2Int` — final structural approval.
7. **String-path filters** (for survivors escalated from int-path or directly constructed in string-path):
 - `checkInputVariablesOrder` ([`conjecturer.cpp`](../../GL_Quick_VS/GL_Quick/src/conjecturer.cpp)) — input-variable-ordering invariant.
 - `checkInputVariablesTheoremOperatorHead` ([`conjecturer.cpp`](../../GL_Quick_VS/GL_Quick/src/conjecturer.cpp)) — operator-head-specific invariant (nse≥2).
 - `evaluateOperatorExprs2` ([`conjecturer.cpp`](../../GL_Quick_VS/GL_Quick/src/conjecturer.cpp)) — operator-expression structural check (nse≥2).
 - `patternInConjecture` ([`conjecturer.cpp`](../../GL_Quick_VS/GL_Quick/src/conjecturer.cpp)) — forbidden-pattern rejection.
 - `controlEquality` ([`conjecturer.cpp`](../../GL_Quick_VS/GL_Quick/src/conjecturer.cpp)) — equality-related guards, including [I-8](../30_invariants.md#i-8).
 - `passesInPremiseFilter` ([`conjecturer.cpp`](../../GL_Quick_VS/GL_Quick/src/conjecturer.cpp)) — in[]-premise shape filter; see the [dedicated section](#passesinpremisefilter) for the three allow-rules (cnt==1 existence, cnt==2 negated, cnt==2 nse=3 neutralisation).
8. **Reshuffle** — `reshuffle(expr, deep)` at [`conjecturer.cpp`](../../GL_Quick_VS/GL_Quick/src/conjecturer.cpp). See [Reshuffle pipeline](#reshuffle-pipeline) below for the canonicalisation stages on the `rt_conjecturer` / `rt_conjecturer2` branches (flat-walk rename, existence-head pinning, contiguous-arg renumber, anchor position-0 pin).
9. **Mirror generation** — `createReshuffledMirrored(expr, anchorFirst)` at [`conjecturer.cpp`](../../GL_Quick_VS/GL_Quick/src/conjecturer.cpp) computes the reverse direction (head swapped with the premise sharing its output variable). `mergeMirrorConjecturesIntoPool` then folds those mirrors into `conjectures.txt` (de-duplicated), so each reverse direction passes through the CE filter and is **genuinely proved** by the prover. The prover no longer fabricates an unproved `mirrored statement` row (D-112).
10. **OR conjectures** — `generateOrConjectures` at [`conjecturer.cpp`](../../GL_Quick_VS/GL_Quick/src/conjecturer.cpp), config-derived from per-expression `allow_to_constitute_existence` flags. The emitted `(existence, companion)` pairs are written to `or_pairs.txt` as an OUTPUT artefact (overwritten each run).
11. **Write output** — `conjectures.txt` (now including the mirror conjectures from step 9), `reshuffled_conjectures.txt`, and the archival `reshuffled_mirrored_conjectures.txt` (the same mirror strings, retained for inspection only).

Two execution lanes exist:

- **Int-path** — functions with `Int` suffix. Hot-path for combinatorial enumeration. Uses `IntConjBuf`, `IntDefSetMap`, `IntConnMap` — integer-encoded forms with no string allocations inside inner loops.
- **String-path** — the older lane. Used for final-stage structural checks where the string form is unavoidable (pattern matching, mirror generation).

Both lanes apply equivalent filters; the int-path is strictly faster. Historically the project had only the string lane; the int lane was added as part of the "100x acceleration" campaign. A second wave of int-story work landed on `rt_conjecturer_session_24042026` (merged): Path B reshuffle permutation loop, thread_local disintegrate cache, and an iterative `disintegrateImplication` walker replacing the `parseExpr`/TreeNode1 tree. The campaign reduced Peano conjecturer wall-time ≈30.5 s → ≈24 s (−20 %) while preserving byte-for-byte output identity on both Peano and Gauss. Full campaign rationale + measurements in [`40_decisions.md#d-20`](../40_decisions.md).

---

## Operator head vs relation head

Expression definitions split into two categories for the conjecturer:

- **Operator.** `output_args` non-empty. Examples: `in2` (output index `2`), `in3` (output `3`), `fold` (output `6`). The output variable needs a companion expression to define it. In the `nse ≥ 2` path, the operator-head-validity checks (`checkInputVariablesTheoremOperatorHead`, `evaluateOperatorExprs2`) ensure the output is properly connected.

- **Relation.** `output_args` empty. Examples: `=`, `in`. No output binding; any placement is valid.

The `nse = 1` path (`singleExprAnchorConnection`) skips the operator-head checks entirely — because with only one expression besides the anchor, there is nothing to connect the operator's output to. This used to produce malformed conjectures; the current code gates the operator-head checks on `nse ≥ 2`, but only operators with `output_args.size > 0` need this gating and the list is implicit in the config — see [Weaknesses](#weaknesses) below.

---

## OR conjectures

`generateOrConjectures` at [`conjecturer.cpp`](../../GL_Quick_VS/GL_Quick/src/conjecturer.cpp) derives `(existence, companion)` pairs directly from per-expression `allow_to_constitute_existence` config flags and generates OR-shaped conjectures of the form:

```text
(>[...](Anchor[...])(or<N>[...]))
```

where `or<N>` is the compiled name for the disjunction expression. Emitted into `conjectures.txt` alongside ordinary conjectures. The prover then treats these specially via `or disintegration` / `or convergence` tags — see [`20_core_concepts/07_or_branching.md`](../20_core_concepts/07_or_branching.md).

The emitted pairs are also recorded in `files/theorems/or_pairs.txt` as an OUTPUT artefact (one `(existence_conjecture\tcompanion_conjecture)` per line). The file is opened with `std::ios::out` and overwritten each successful run; the upstream cleanup pass that removes other stale theorem outputs preserves this filename but the content is replaced. Downstream consumers read `or_pairs.txt` to learn which `conjectures.txt` rows participate in OR-pair semantics. (Historical: an earlier path read `or_pairs.txt` as a hand-curated INPUT; the current implementation does not — it derives pairs from config flags and writes the file as output.)

---

## Filter cascade — details

### `checkDefSets` / `checkDefSetsInt`

Ensures every argument's `definition_sets` type-label is consistent across all occurrences within the candidate. A variable appearing in two positions must have compatible types at both.

*Seen:* [`conjecturer.cpp`](../../GL_Quick_VS/GL_Quick/src/conjecturer.cpp) (string) and [`conjecturer.cpp`](../../GL_Quick_VS/GL_Quick/src/conjecturer.cpp) (int).

### `checkInputVariablesOrder`

Enforces that bound variables appear left-to-right in input-argument positions across the expression. Used to reduce the permutation surface — otherwise trivially-rearranged variants would all pass filtering.

*Seen:* [`conjecturer.cpp`](../../GL_Quick_VS/GL_Quick/src/conjecturer.cpp).

### `controlEquality`

Among other things, implements [I-8](../30_invariants.md#i-8): rejects `(=[x,x])` as a head. Also rejects equalities among un-typed positions.

**Canonicalisation rule.** Rejects descending-ordered equality `(=[a, b])` with `std::stoi(a) > std::stoi(b)` — `=` is symmetric, so keeping only the ascending form eliminates redundant mirror-conjectures at every `nse`. The pair-combination enumeration in `connectExpressionsInt` produces both orderings; dropping the descending one is lossless at conjecture-set level because the ascending sibling is always reached. The cancellation theorem's head shape `=[i0, b]` (e.g. `=[2, 8]` — anchor-slot value first, bound-var second, ascending) survives as the canonical form; its symmetric duplicate `=[8, 2]` is correctly rejected.

**History.** Between [D-21](../40_decisions.md#d-21) (2026-04-24) and [D-23](../40_decisions.md#d-23) (2026-04-28), the function carried an `nse <= 3` exception that admitted descending forms too. The exception was authored when the post-anchor-pinning enumeration was thought to land bound-var IDs in arg-1 (descending) without producing the ascending mirror. Empirical inspection at D-23 time showed the ascending mirror IS produced independently, so the exception was reverted. See D-23 for the trade-off check.

*Seen:* [`conjecturer.cpp`](../../GL_Quick_VS/GL_Quick/src/conjecturer.cpp).

### `patternInConjecture`

Rejects candidates matching known-unproductive patterns. This is the easy-to-extend hook for "this shape has never produced a theorem, don't bother".

*Seen:* [`conjecturer.cpp`](../../GL_Quick_VS/GL_Quick/src/conjecturer.cpp).

### `prohibitedHeadsGood`

Rejects candidates whose head is on an ad-hoc block list. Configurable per batch.

*Seen:* [`conjecturer.cpp`](../../GL_Quick_VS/GL_Quick/src/conjecturer.cpp).

### `passesComplexityAfterExistence`

Per-def-set cap on post-anchor chain length when a chain arg is pinned to an anchor slot of that type. Defined in `parameters.max_complexity_if_anchor_parameter_connected_after_existence` per config — a 2-tuple `[complexity_cap, arity_sum_cap]` per def-set type (with legacy single-int values auto-promoted to `[int, 100]` by the loader). Peano's setting: `(1) → [2, 8]`, `P(x(1)(x(1)(1))) → [10, 100]`, others `[10, 100]` (arity dimension off, complexity cap effectively off).

**Rejection criterion (post-D-23).** A conjecture is rejected when, for some capped type T, **all three** of the following hold:

1. `complexityLevel > complexity_cap_T`
2. `arity_sum > arity_sum_cap_T` (sum of arities across non-anchor leaves)
3. some anchor slot of type T has its value present in non-anchor leaves

If EITHER cap is not exceeded, the conjecture survives via that dimension. Cancellation theorem (`(in[a,N])(in3[a,b,i0,+]) -> (=[b,i0])`) has complexityLevel = 3 AND pins `in3`'s c-arg to anchor slot 2 (i0), so dimension 1 and dimension 3 fire — but `arity_sum = 4 + 2 + 2 = 8 ≤ 8`, so dimension 2 doesn't fire and the conjecture survives. Three-`in3` explosion shapes (e.g. `(>[4,5,6](AnchorPeano…)(in3[…6…])(in3[…6…])(in3[…6…]))` with slot 6 used 3×) have arity_sum = 4 + 4 + 4 = 12 > 8 and are correctly rejected.

**No hard-coded complexity bands.** Every threshold is the config value; the function consults nothing else. (Pre-[D-23](../40_decisions.md#d-23) the function had `if (complexityLevel <= 2) return true;` and a 73-line `if (complexityLevel == 3) { … }` carve-out; both were authored at [D-21](../40_decisions.md#d-21) before the arity-sum dimension existed and were reverted at D-23 because the arity-sum dimension subsumes their job. The two relaxations short-circuited before the per-type cap loop and prevented it from running at small `complexityLevel`. With the relaxations removed, the loop runs uniformly for every conjecture.)

*Seen:* [`conjecturer.cpp`](../../GL_Quick_VS/GL_Quick/src/conjecturer.cpp).

### `passesInPremiseFilter`

Gates conjectures that carry `(in[…])` as a non-anchor chain premise. Conjectures without such a premise pass unconditionally (`hasIn == false` → early return `true`).

**Top-of-function gate — anchor-membership-axiom rejection ([D-23](../40_decisions.md#d-23), commit ).** Before any of the cnt-shape checks, the function walks the chain and rejects any `(in[v, X])` premise (or its negation) where BOTH `v` AND `X` are anchor-slot values. Such a premise is one of the anchor's own axioms — e.g. `(in[2, 1])` reads "i0 ∈ N", which is already entailed by `AnchorPeano`. As a premise it adds no constraint, so the whole conjecture is vacuous in the same sense as a `(=[x, x])` head. Single-anchor-arg shapes like `(in[7, 1])` (bound-var v, anchor-slot N) are NOT rejected — those are genuine typing premises the cancellation family needs.

Historical note: the same check existed inline inside `passesComplexityAfterExistence`'s `complexityLevel == 3` carve-out before D-23. When that carve-out was removed, the check moved here so it now runs for every conjecture, not just nse=3 ones.

**Cnt-shape rules (post-anchor-membership gate).** When `hasIn` is true, accept iff one of:

1. `cnt == 1` AND head is an existence form `!(>[…]…)`. (Existence-head reformulation landed the single in[] premise as typing for the bound variable of the collapsed existence block.)
2. `cnt == 2` AND at least one of the two non-anchor premises is negated (`!(…)`). (The negation carries the semantic weight; the in[] premise is either typing or the negated one.)
3. `cnt == 2` AND there exists a POSITIVE `(in[v,X])` premise whose first argument `v` appears as an argument in the OTHER non-anchor premise OR in the head. (Neutralisation rule, [D-21](../40_decisions.md#d-21) relaxation 1, kept active after D-23 — needed for the additive-cancellation family `(in[a,N]), (in3[a,b,i0,+]) → (=[b,i0])` and its successor/product siblings. Scope is deliberately `cnt == 2` only — i.e. `nse = 3` — to prevent population blow-up at larger `nse`.)

`cnt >= 3` is rejected unconditionally. `nonAnchor` here counts chain premises only (the head is out-of-band); the anchor premise is skipped.

**Neutralisation rule — why the shape.** The first argument of `(in[v,X])` is the element being asserted to live in `X`. Under this filter, "neutralising" means the element `v` actually participates elsewhere in the conjecture — either as an arg of the other chain premise or as an arg of the head. When it does, the `in[v,X]` premise is carrying real typing weight (the cancellation case is the canonical example: without `(in[a,N])`, `a` in `(in3[a,b,i0,+])` is an untyped bound variable). When it doesn't, the `in[v,X]` is vacuous typing of a never-used variable and the original filter keeps rejecting it. A NEGATED `!(in[v,X])` asserts `v ∉ X` and is not a typing premise; such premises do not count for the neutralisation check (but they do count for rule 2).

**Dead config flag.** `parameters.apply_in_premise_filter` (default `true`) is declared at [`conjecturer.hpp`](../../GL_Quick_VS/GL_Quick/src/conjecturer.hpp) and loaded at [`conjecturer.cpp`](../../GL_Quick_VS/GL_Quick/src/conjecturer.cpp), but its value is never consulted inside the filter function or at the callsites. Setting `"apply_in_premise_filter": false` (as `ConfigGauss.json` does) has no effect on conjecturer output. Gauss currently ships zero conjectures with `in[]` as a chain premise anyway — all three `in[]`-containing Gauss theorems place `in[…]` at the head, where `hasIn` stays false and the filter short-circuits. If re-enabling a real batch-level gate is ever needed, add an early `if (!parameters.apply_in_premise_filter) return true;` at the top of the function and update this description. See [OPEN-9](#open-questions) for the historical record.

*Seen:* [`conjecturer.cpp`](../../GL_Quick_VS/GL_Quick/src/conjecturer.cpp).

### `passesMaxDistinctAnchorValuesPerType`

Config-gated post-anchor-attachment filter. For every def-set type `T` listed in `parameters.max_distinct_anchor_values_per_type`, walks the conjecture's non-anchor leaves (descending into nested `!(>[…]…)` existence heads) and counts how many distinct anchor-slot values of type `T` participate. Rejects when the count exceeds the per-type cap. Empty map disables the filter (legacy behaviour preserved).

Applied at three `connExpr2`-producing sites in the cascade:

- [`conjecturer.cpp`](../../GL_Quick_VS/GL_Quick/src/conjecturer.cpp) — int-path inside `singleThreadCalculationInt`.
- [`conjecturer.cpp`](../../GL_Quick_VS/GL_Quick/src/conjecturer.cpp) — `singleExprAnchorConnectionInt`.
- The string-path `singleExprAnchorConnection` (line varies; same gate).

Orthogonal to `passesComplexityAfterExistence` ([see field reference](../04_configs.md#prover_parameters-prover-parameters)): the complexity gate is a binary "type T present at high complexity" check that bypasses existence-head conjectures, while this filter applies uniformly and gates on *count* rather than complexity. Both filters coexist.

Per-tag config rationale captured body: caps were chosen to match per-type maxes observed in `theorems.txt`, so no baseline-proved shape is dropped. Observed effects on the immediate run after introduction:

- Peano: 501 → 501 conjectures (no shapes dropped at the chosen caps).
- Gauss: 395 → 123 conjectures (drops triple-digit `(1)=3` shapes and double-operator `P(x(1)(1))=2` shapes; all 9 proved Gauss theorems preserved).

---

## Reshuffle pipeline (branches `rt_conjecturer` / `rt_conjecturer2`)

`reshuffle(expr, deep)` at `conjecturer.cpp` canonicalises a survived conjecture into its `conjectures.txt` / `reshuffled_conjectures.txt` form. On the `rt_conjecturer` lineage the pipeline is:

1. **Flat-walk rename** (commit, replaces the older `renameVariablesInExpr`). Walks the expression block-by-block (bvs-then-atom-args), emitting new arg-IDs in first-occurrence order. Produces a deterministic numbering independent of recursive tree shape — bv-vs-atom-arg order is what determines final IDs, not depth.

2. **Existence-head pinning**. When the outer head is `!(>[bvs](L)(R))`, the head's bv-list is extracted BEFORE the flat-walk starts and pinned at positions with first-occurrence priority. Without this, the existence bv would end up higher-numbered than chain-introduced bvs that appeared later in walk order — producing "wrong ordering" of existence vs chain vars in the canonical form. Fix: disintegrate the existence head manually, seed its bvs into the renumber map first, then walk the rest.

3. **Contiguous-arg renumber post-connect** (commit, inside `connectExpressionsInt`). After a `connectExpressionsInt` merge via `subMap`, surviving arg-IDs may have holes in the shifted range. A renumber pass collects arg-IDs from atom args in block order, builds a `renumber` map (`oldID → contiguousID` starting at 1), then applies it across the expression. Downstream consumers (`conjectures.txt`, anchor-connect inputs) see a contiguous `1..N` arg-ID space rather than a sparse one.

4. **Anchor position-0 pin**. The anchor call is fixed at permutation position 0 — anchor args are never permuted across reshuffle variants. Ensures the anchor's arg-ID assignments stay stable so that `AnchorGauss[1,2,3,4,5,6,7,8]` reads identically across all conjectures regardless of how the body was walked.

**Cascade order.** Flat-walk → existence-head pinning → connect-with-subMap-and-renumber → anchor-pin. Each stage's output feeds the next's input. Skipping any stage produces drift in the canonical form vs the `_mirrored` companion.

**Disintegrate-cache optimization (`rt_conjecturer_session_24042026`, commit ).** `compiler.hpp::disintegrateImplication` carries a thread_local single-slot cache keyed by the input expression string. At the hot conjecturer call site (`singleThreadCalculationInt`), a single candidate passes through ~10 successive filters that each re-disintegrate the same string; without the cache, that's 10× redundant `parseExpr` + TreeNode1 tree builds per candidate. With the cache, each new candidate misses once and every subsequent filter call on the same string hits (vector copy, no tree build). Cache invariants: keyed by `std::string` content (not pointer), no explicit invalidation — staleness is detected by inequality on the next call. thread_local gives zero-contention per-worker reuse. Measured impact on Peano: `disintegrateImplication` self-time 198 s → 75 s (-62 %), wall 27.2 s → 23.8 s (-12.5 %). The cache also helps the prover's reshuffle-mirror sites in `prover.cpp` via the same shared `compiler.hpp` function, though the prover's access pattern is less dense. Cache ChainT typedef must be kept in sync with the function's output shape — see [G-34](../50_gotchas.md#g-34).

**Iterative disintegrate walker (`rt_conjecturer_session_24042026`, commit ).** The cache-miss path itself was subsequently ported from the `parseExpr` + TreeNode1 tree + `treeToExpr` + `node->arguments` recursive-build pipeline to a direct string walker. The walker peels `(>[bvs]...)` layers one at a time, extracts each premise as a substring (handling both `(prem)` and `!(prem)` shapes), and computes `leftArgs` by walking the premise's `[...]` lists minus bvs declared in any nested `>[bvs]` inside it. This preserves parseExpr's recursive `node->arguments = union(children) - this-level-bvs` semantics for the MPL shapes the conjecturer emits (atoms, `!atom`, `&`, `!&`, `>`, `!>`). Measured impact: disintegrate self-time 75 s → 32 s (-57 %, cache-miss path dominated by tree-allocation cost), per-call avg 4.3 µs → 1.9 µs. Wall was noise-bound at this level (Windows 10-run spread ±3 s), so this commit banks thread-time rather than a measurable wall delta on a single run. Both `parseExpr` and the tree machinery are still used by other `compiler.hpp` functions — only `disintegrateImplication`'s dependency on them was severed.

**Permutation-loop implementation (`rt_conjecturer_session_24042026`, commit ).** The inner loop that evaluates each candidate permutation used to do, per permutation, a full `std::string` rebuild followed by a blanket `replaceKeysInString` pass over the rebuilt string — a sequence of allocator-heavy operations dominated by char-by-char mutation. The current implementation pre-parses each chainEntry + head into an `EntryTemplate` that pinpoints every `[...]`-token byte position with an `isAtom` flag (the flag mirrors OLD's `collectAll` skip-rule for `>[bvs]`). Per permutation: (1) walk atom token positions in permuted byte order to build the first-occurrence rename map into a thread_local `int[]` indexed by dense token id; (2) render the rebuilt string directly into a thread_local `char`-vector with inline rename substitution (no `std::string` temporaries, no `replaceKeysInString`); (3) winner selection uses `memcmp` on the rendered buffer, which is byte-for-byte equivalent to the OLD `std::string::operator<` on renamed rebuilt strings. The external contract (returned rebuilt string + rename map) is preserved identically — byte-for-byte match verified on both Peano and Gauss `conjectures.txt` / `reshuffled_conjectures.txt` / `reshuffled_mirrored_conjectures.txt`. Measured effect: `reshuffle` self-time halved (218 s → 111 s thread-time summed across 32 workers), `reshuffle` per-perm cost 8.4 µs → 3.4 µs, wall 30.2 s → 27.2 s for Peano. See for the allocation pitfalls this avoids.

**Downstream consequence.** `conjectures.txt` / `reshuffled_conjectures.txt` on `rt_conjecturer*` use this pipeline — a conjecture's exact arg-ID layout differs from branches that pre-date these commits (,, etc.). Merging work between branches therefore shifts which specific raw-form string identifies "the same" theorem; use `grep -F` on semantic content + `extractExpression`, not raw-string match. See `reshuffled_conjectures.txt` for byte-accurate examples.

**Verifier view.** The verifier consumes the processed proof graph, not `reshuffled_conjectures.txt`, so the reshuffle pipeline is opaque to it — theorems prove/fail on semantic content, not on arg-ID arrangement.

---

## `createMapAnchor` — the memory sensitive step

Precomputes anchor-to-expression argument permutation tables. For AnchorIncubator with its 7 `(1)`-typed slots, `leftMax = 7`. The `rightMax` parameter = max over def-sets of `(uncomb + comb)` values.

**Critical invariant (observational, not code-enforced):** `rightMax > 3` causes RAM explosion — millions of permutation dicts are materialised. Controlled by `max_values_for_uncomb_def_sets + max_values_for_def_sets` in the config.

*Seen:* `createMapAnchor` at [`conjecturer.cpp`](../../GL_Quick_VS/GL_Quick/src/conjecturer.cpp).

---

## Weaknesses

### Known & tracked

- **`createMapAnchor` RAM explosion.** Mentioned in the project conventions "Key conjecture generator internals". Fix path: keep `max_values_*` config fields small (≤ 3 per arg). Not protected by an assert.
- **Operator-head check gating is `nse`-based.** `checkInputVariablesTheoremOperatorHead` and `evaluateOperatorExprs2` assume `nse ≥ 2` and are skipped on the `nse = 1` path. This is correct for the current operator set (`in2`, `in3`, `fold`, `residual`, `interval`, `preorder`), but adding a new operator that *also* behaves well as a single-expression head would require revisiting — the skip is structural, not operator-specific.

### Suspected fragility

- **Int-path / string-path drift.** Every filter has an int and a string version. If a new filter is added on one lane only, it silently diverges. There is no test asserting int and string produce the same filtered set on a given input. Worth adding.
- **`passesInPremiseFilter` dead flag.** `parameters.apply_in_premise_filter` is loaded from config but never consulted — the filter always runs when `hasIn` is true. Documented under the filter's own section; the SwDD `OPEN-9` entry still states the flag short-circuits the function, which is incorrect until that text is rewritten or the code is reconnected. Until then, do not rely on the flag; all per-batch exemptions happen implicitly because Gauss and the incubators do not emit `in[]` as a chain premise.
- **OR conjecture generation is config-derived, not file-driven.** `generateOrConjectures` derives `(existence, companion)` pairs from per-expression `allow_to_constitute_existence` flags in `config_.data`; it does NOT read `or_pairs.txt` as input (that file is now an OUTPUT artefact recording the emitted pairs). Cross-reference to the FTA-ladder campaign: extending the OR conjecture set requires adding/flipping `allow_to_constitute_existence` on the relevant config expressions, not editing `or_pairs.txt`.

### Not exercised by tests

- **The preliminary-pass code path** (`min_nse = 1`). Peano and Gauss default to `min_nse = 2`, so this code only runs for specific incubator configs. A regression inside the preliminary-pass specifically would go unnoticed on the main batches.
- **Filter-cascade regression tests — partial coverage.** [`test_conjecturer.cpp`](../../GL_Quick_VS/GL_Quick/src/tests/test_conjecturer.cpp) (~373 tests in 13 categories — `helper_*` / `ctor_*` / `intpath_*` / `mappings_*` / `filter_*` / `inputorder_*` / `reshuffle_*` / `reform_*` / `worker_*` / `legacy_*` / `or_pair_*` / `invariant_*`) covers most filters with positive + negative cases, but two are positive-only at the unit level:
 - **Filters with both positive and negative coverage:** `controlEquality` (ascending vs descending; I-8 via `countArgumentsFilter`-forwarding rejecting `(=[x, x])`); `numbersGood` (substring count cap); `prohibitedHeadsGood` (Peano puts `in` on the prohibited list — `in`-headed chains rejected, others pass); `exprGood` / `exprGood2` (rejecting bare-atom inputs); `qualifiedForEquality` (rejecting bare equality); `checkInputVariablesTheoremOperatorHead` (relation passes, operator-headed chain without binding rejected); `passesMaxSizeAfterExistence` (rejects on `leafCount=100`); `passesMaxDistinctAnchorValuesPerType` (rejects when one leaf carries TWO `(1)`-typed anchor slot values like `(in3[7,2,6,4])`, accepts when slots are spread one-per-leaf); `passesInPremiseFilter` — D-23 anchor-membership-axiom REJECTING fixtures using NESTED-quantifier shape `(>[](Anchor)(>[](in[2,1])(=[3,4])))` (plus negated form), plus separate NESTED-quantifier fixtures that exercise each cnt-shape allow-rule individually so deletions are detectable: rule 1 (cnt==1 + existence head) accepts; rule 2 (cnt==2 + at least one negated) accepts; rule 3 (cnt==2 + neutralisation — first arg of in-premise appears elsewhere) accepts; the complementary `cnt==2 / both positive / no neutralisation` case rejects; cnt>=3 rejects unconditionally; head-only-in passes (D-23 only inspects chain premises). Plus the int-path twins (`numbersGoodInt`, `repetitionsExistInt`, etc.).
 - **Smoke-only (positive cases only) — known gap:** `passesComplexityAfterExistence` (the post-D-23 3-condition rule — `complexity > T-cap AND arity_sum > T-cap AND a T-typed anchor slot in non-anchor leaves` — could not be triggered from hand-synthesised inputs in the unit suite; the rejection path is exercised by integration through `main.py` runs where real conjectures from `conjectures.txt` hit the rule). Filter cascade interactions involving the `singleThreadCalculation*` worker drivers also stay smoke at the unit level (the drivers return all-empty `WorkerResult` on hand-built inputs because the cascade-filter caps reject before merge succeeds).
 - **Test infrastructure:** `friend class conj::testing::Friend;` in `Conjecturer`'s private section grants the unit suite access to private methods. Harness chapter: [`_meta/testing.md`](../_meta/testing.md).
- **Int-path / string-path lane drift on real candidates.** The `worker_*` suite runs both lanes on simple inputs but the int-vs-string identity check on a representative production candidate (the highest-leverage drift detector) requires a deeper input fixture and is the obvious next gap to close.

---

## Open questions

- **OPEN-8 — RESOLVED (no assert; one should be added).** No assertion of the form `rightMax ≤ 3` exists anywhere in the codebase; grep across `GL_Quick_VS/GL_Quick/src/` returns nothing matching `assert` + `rightMax` or similar. Misconfiguration (`rightMax > 3`) therefore produces a silent RAM explosion — the process grows unbounded, and the OS kills it at some point with no indication of cause. **Recommendation**: add an assert at the top of `createMapAnchor` ([`conjecturer.cpp`](../../GL_Quick_VS/GL_Quick/src/conjecturer.cpp)) checking `rightMax <= 3` (or a configurable bound), with a message pointing at the responsible config field. Cheap to add; saves hours of bewilderment on a misconfigured run.
- **OPEN-9 — CORRECTED (flag is dead code).** The config flag `apply_in_premise_filter` is declared at [`conjecturer.hpp`](../../GL_Quick_VS/GL_Quick/src/conjecturer.hpp) and loaded at [`conjecturer.cpp`](../../GL_Quick_VS/GL_Quick/src/conjecturer.cpp) with default `true`, but its value is never consulted — no early-return guard exists inside `passesInPremiseFilter` nor at its callsites (`:4378`, `:4398`). `ConfigGauss.json` sets `"apply_in_premise_filter": false`, which has no effect; Gauss escapes the filter only because its `in[]`-containing conjectures place `in` at the head (so `hasIn` stays false and the function short-circuits on line 4994). If a real per-batch gate is desired, add `if (!config_.parameters.apply_in_premise_filter) return true;` at the top of the function. Historical note: the earlier SwDD text claimed the flag short-circuits the filter — that was an unverified assumption, corrected here.

---

## See also

- [`10_pipeline/03_ce_filter.md`](03_ce_filter.md) — downstream consumer of `conjectures.txt`.
- [`10_pipeline/01_mpl_definitions.md`](01_mpl_definitions.md) — upstream producer of the compiled configs.
- [I-8](../30_invariants.md#i-8) — trivial equality forbidden in head.
- [I-9](../30_invariants.md#i-9) — equality mirror guarded by distinctness.
- [I-11](../30_invariants.md#i-11) — (inverted) anchor slots ARE bound in a theorem's `>[...]`; [I-4](../30_invariants.md#i-4) — one binder rule; [D-75](../40_decisions.md#d-75).
- — history of the int-path acceleration.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
