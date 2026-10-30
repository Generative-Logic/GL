<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Glossary `[DRAFT]`

> Every domain term used by the GL codebase, grouped by concern. Each entry gives a one-paragraph definition, a file + symbol or artefact citation, and — where it clarifies — a byte-accurate MPL example. Cross-references use inline links.

When you (future agent) encounter a word you haven't seen before in a code path, this document is the first place to look. If the word *isn't* here, add it.

---

## Table of contents

- [Language & expression shape](#language--expression-shape)
- [Variables](#variables)
- [Anchors](#anchors)
- [Proof structure](#proof-structure)
- [Logic blocks & memory](#logic-blocks--memory)
- [Hash engine & admission](#hash-engine--admission)
- [Scope, validity, & the name map](#scope-validity--the-name-map)
- [Disintegration & integration](#disintegration--integration)
- [OR branching & contradiction](#or-branching--contradiction)
- [Equivalence classes](#equivalence-classes)
- [Conjecturer](#conjecturer)
- [Compressor](#compressor)
- [Incubator](#incubator)
- [Proof tags](#proof-tags)
- [Pipeline artefacts](#pipeline-artefacts)

---

## Language & expression shape

### MPL

**Mathematical Programming Language.** The string grammar in which all GL definitions, conjectures, theorems, and proof-graph lines are written. MPL is S-expression-like but uses square brackets for argument lists and parentheses for node wrapping.

Grammar, minimal:

```
expression    := "(" head-name "[" args "]" ")"                          atomic
              |  "(" "&" expression+ ")"                                  conjunction
              |  "(" ">" "[" bound-vars "]" expression expression ")"    implication
              |  "(" "=" "[" arg1 "," arg2 "]" ")"                       equality
              |  "!" "(" "=" "[" arg1 "," arg2 "]" ")"                   negated equality
              |  "!" expression                                           general negation
```

*Always* byte-exact. **No spaces** anywhere — arguments separated by bare commas. for the hard rule.

Representative real expressions (from `files/processed_proof_graph/`):

```text
(AnchorPeano[N,i0,s,+,*,i1])
(in3[i0,i1,v1,+])
(>[v2,v3](in2[v2,v3,s])(in[v2,N]))
!(in2[v1,i0,s])
```

*Seen:* throughout the codebase; definitions in `files/definitions/*.mpl`; processed chapters in `files/processed_proof_graph/*.txt`.

---

### Expression

A single MPL string. In code, an expression is typically a `std::string` with the literal MPL characters, used as the canonical key for everything (hash lookup, set membership, comparison).

### exprKey

Not a separate type — it is an expression string used as the identifying key for a `Memory` block or for a row in the hash memory. Every `Memory` (per-LB storage; see [Memory](#memory)) has an `exprKey` that is the expression it is responsible for proving or storing. Same bytes as the expression.

### head (of an implication)

The body of a `(>[vars](premise)(body))` — the conclusion that follows from the premise. When the premise chain has multiple layers, the head is the *innermost* body, obtained by repeatedly unwrapping `>[...]` layers. See [`disintegrate_implication_head`](#disintegrateimplicationhead) in the verifier.

### chain (of premises)

The list of premises of an implication, flattened by peeling every `>[...]` layer. For `(>[α,β](P1)(>[γ](P2)(head)))`, the chain is `[P1, P2]` and the head is `head`. See [`disintegrate_implication_full`](#disintegrateimplicationfull).

### atomic expression

An expression whose head name is not one of the structural operators (`&`, `>`, `=`). Atomic expressions are ground predicates applied to arguments — for example `(in[i0,N])`, `(in3[a,b,c,+])`, `(fXY[s,N,N])`. Defined by `category: atomic` in `CoreExpressionConfig`; see [`compiler.hpp`](../../GL_Quick_VS/GL_Quick/src/compiler.hpp).

### and node

A conjunction `(&expr1 expr2...)`. Defined by `category: and`. Expands and disintegrates into its constituent elements. Example: the body of `NaturalNumbers` is a nested `and` tree.

### existence node

An expression with `category: existence`. Defined by a pair of elements (left + right), where one position is marked `definedSet` and holds the bound variable. Expands into left (with new bound var) + right elements when disintegrated. Example: `existence4[N,+,v3,v1,i0]`.

### implication node (as a category)

An expression with `category: implication` — terminal from the perspective of the GL binary. Stored as a hash-table rule: `premise_signature → conclusion_template`. The top-level `(>[...])` of any theorem is an implication by shape, but the *category* term in the GL binary refers to the sub-expressions named `implication0..implicationN` that enumerate the atomic rule shapes carried by each definition.

### structural operator

The three operators that shape expression structure: `&` (conjunction), `>` (implication), and their negations `!(&...)`, `!(>...)`. Every theorem-load path must rewrite raw `!(&...)` / `!(>...)` into compiled `or<N>` / `existence<N>` names via [`precompileStructuralOperators`](#precompilestructuraloperators). See [I-1](30_invariants.md#i-1).

### spontaneous compact operator

A compact operator name allocated *at run time* by the C++ prover from a counter, not declared up-front in the MPL definition file. The four spontaneous categories are `implication<N>`, `existence<N>`, `or<N>`, `and<N>` (set: `_SPONTANEOUS_CATEGORIES = {"implication", "existence", "or", "and"}` in [`run_modes.py`](../run_modes.py)). Examples seen in proof-graph artefacts: `or0[7,2,1,3]`, `existence2[N,n,s]`, `implication26[N,+,i0,i1,V1]`.

Naming is stable across batches via [`GL_binary_shared.json`](#filesgl_binariesgl_binary_sharedjson) — Python copies the shared file into the per-batch `GL_binary_<Tag>.json` before each batch starts and merges newly-allocated spontaneous entries back into shared after each batch ends. A name allocated in batch *N* keeps its definition in batch *N+k* (entries in shared are never overwritten). See [I-23](30_invariants.md#i-23) and [D-22](40_decisions.md#d-22).

Anchor entries (`AnchorPeano`, `AnchorGauss`, `AnchorIncubator`, …) and atomic seeds (`in`, `in2`, `=`, `fXY`, …) are explicitly **not** spontaneous and **not** carried across batches in shared — they are batch-local.

---

## Variables

GL uses several variable namespaces, distinguished by their name prefix.

### argument

A leaf at an `[…]` position inside an expression. An argument can be a variable name, a literal number (which in practice means a bound-variable index in conjecturer output), a function name (`s`, `+`, `*`), or another anchor-defined identifier (`N`, `i0`, `i1`).

### bound variable

A variable listed inside the `>[...]` quantifier marker at the head of an implication. Bound variables have scope limited to the implication body. In the processed proof graph, bound variables are numbered `v1`, `v2`, …; in raw conjecturer output they are numbered `1`, `2`, …

### free variable

A variable name in an expression that is not inside any enclosing `>[...]`. In theorem form these are typically anchor-fixed symbols (`N`, `i0`, `s`, `+`, `*`, `i1`, …) — they refer to specific constants defined by the anchor.

### `u_`-prefixed variable

A formal parameter in a GL binary definition. `u_`-prefixed variables are the definition's "slots" — they must remain free (not quantified) when the definition is instantiated. Example (abstract signature):

```text
(>[u_0,u_1](in2[u_0,u_1,u_2])...)
```

When `reconstructImplicationFullBind` rebuilds an expression, it binds **all** non-`u_` variables universally in the outer `>[…]`. When `reconstructImplication` rebuilds, it binds only variables occurring multiple times. The `u_` prefix is the marker that decides. See [I-4](30_invariants.md#i-4), [I-5](30_invariants.md#i-5).

*Seen:* `CoreExpressionConfig::definition` in [`compiler.hpp`](../../GL_Quick_VS/GL_Quick/src/compiler.hpp); implementation at `compiler/implication_builder.hpp` (`reconstructImplication`) and `compiler/implication_builder.hpp` (`reconstructImplicationFullBind`).

### `v`-variable

A renamed bound variable in the processed proof graph. `process_proof_graphs.py` maps raw bound-variable indices (`1`, `2`, …) to type-cased names via a four-priority scheme seeded from the theorem expression (see [I-10](30_invariants.md#i-10)). Names use **lowercase `v`** (`v1`, `v2`, …) for variables that occupy a `(1)`-typed position (digits / elements of N) and **uppercase `V`** (`V1`, `V2`, …) for variables that occupy a `P(...)`-typed position (sets). The two counters are independent; both start at 1. Anchor-fixed slot names (`N`, `i0`, `s`, etc.) are not renamed. The HTML visualizer further shifts `v→w` / `V→W` when an applied/external theorem is rendered inside a chapter (display-only, to avoid confusion with the chapter's own v/V variables).

Real example — processed chapter line showing `v1`:

```text
(in[v1,N])	main	implication	(>[v2,v3](in2[v2,v3,s])(in[v2,N]))	main	(in2[v1,i0,s])	main
```

### `it_`-prefixed variable

An **iteration variable** minted during Pass B disintegration. Pass B produces fresh variables for each iteration of a disintegration step; these are prefixed `it_` to distinguish them from stable `v-` variables and from integration variables. The Pass B admission paths [`isAdmitted`](#isadmitted) and [`isAllowedAsOperatorInput`](#isallowedasoperatorinput) gate whether an `it_…` variable is committed to `finalStringStatements`.

*Seen:* `disintegrateExpr2` in [`logical_block/disintegration_processor.cpp`](../../GL_Quick_VS/GL_Quick/src/logical_block/disintegration_processor.cpp).

### `int_`-prefixed variable

An **integration variable** used on the integration side (the mirror of disintegration). The `int_` path uses only the map-based [`isAdmitted`](#isadmitted) admission rule — the standalone single-input-operator fallback that applies to `it_…` variables is *not* ported. See [I-6](30_invariants.md#i-6).

### `repl_lev_*_*` variable

Substitution-chain placeholder minted during anchor-handling rewrites. The pattern `repl_lev_<N>_<M>` captures the rewrite level and position. These are internal names that must be renamed consistently when `_copy` suffixes are derived. See renaming Priority 4 in the project conventions.

### `_copy` suffix

A variable ending in `_copy` is a freshly duplicated version of an existing variable, introduced to keep a hypothesis variable distinct from its surrounding scope during case analysis or OR branching. The tag `variable copy` (see [Proof tags](#proof-tags)) records each such duplication.

Raw `X_copy` is renamed in the processed proof graph by attaching the `_copy` suffix to the renamed base variable: if raw `repl_lev_3_2` renames to `v11`, then `repl_lev_3_2_copy` renames to `v11_copy`. The one variable copy whose alias does not end in `_copy` is `(=[q,rec2])` (see `rec2`).

*Seen:* verifier substitution code at `verifier.py`, `verifier.py`.

### digit args

For an implication chain, the set of non-anchor arguments that flow into it from outside — its induction variables. Computed by `ScopeTypeResolver::findDigitArgs` ([`logical_block/scope_type_resolver.hpp`](../../GL_Quick_VS/GL_Quick/src/logical_block/scope_type_resolver.hpp)): collect all `(1)`-typed `inputIndices`-selected arguments from every expression in the chain, subtract anchor args, subtract every argument that also occupies a non-input slot (an output, or a slot that is neither input nor output) of a configured expression: an induction variable is always an input ([D-580](40_decisions.md#d-580), which retired D-565's property arguments). Used to decide which bound variables are "inputs" versus "outputs" of the implication — this shapes `multiplyImplication`'s partition choices and the admission map.

### immutable args

The subset of digit args that are provably not modified by the chain — they propagate from premise to head without being in any output position. Computed by `findImmutableArgs` ([`proof/proof_grid_builder.cpp`](../../GL_Quick_VS/GL_Quick/src/proof/proof_grid_builder.cpp)). Seed = digit args minus the induction variable. Extend through outputs where all inputs are already immutable. Used in induction-step reconstruction.

### twin premises (twin partner)

Two non-anchor premises of a conjecture that are the only two premises of one predicate, where exchanging them leaves the conjecture unchanged after normalization (an equality head read unordered). A digit argument k of the first twin has the twin partner l, the digit argument the second twin holds at k's slot (each absent from the other twin). The conjecture is inducted on k only and the weak hypothesis leaves l universal (`ProofGridBuilder::twinInductionPartners`, `Memory::twinPartnerId`; [I-338](30_invariants.md#i-338)). The full I5 lemma: k = 9, l = 12.

### second twin goal

The second twin premise's integration goal in a twin weak step block: the first premise's two-marker goal with `rec2` in place of `rec`, filed when `(in2[q,l,s])` registers in the successor case of the case experiment (or at `main`), bound to the second premise and grouped with the first goal, so their int scopes are promoted together; `rec2` and l play `rec` and the induction variable for it (`AdmissionMemory::fileTwinSecondGoal`, `inductionRolesOf`; [I-339](30_invariants.md#i-339)).

### rec2

The name of the twin partner's predecessor q in a twin weak step block, as `rec` names the induction variable's predecessor: the second twin goal's filing stages the variable copy `(=[q,rec2])` at its scope, and the class chooser then writes the scope's statements with `rec2`. It counts as a number like `rec` (`isNumberArgument`), except in the role gates of the copies integrating at `rec` ([D-581](40_decisions.md#d-581)).

### case experiment (zero-or-successor experiment)

A twin block's case split of its partner's OR "l = zero or there is a q with s(q) = l", starting with the successor case: a zero block releases only that case, demands there the partner's values of the second premise's functions (Q's last entry for the full I5, `AdmissionMemory::fileTwinValueDemands`), and once the case is refuted the OR's exclusion puts l = zero at `main`; the weak step block proves the step in the successor case and then releases the zero case, which is refuted, and the OR convergence closes the step (`Memory::twinExperiment`, `CaseProcessor::twinExperimentLeaf`, `LbMemory::experimentOrCohorts`; [I-340](30_invariants.md#i-340)).

### output variable

The argument position marked as an output in a `CoreExpressionConfig`. For `in2` (3-arg), `output_args` lists `"2"`, the second name of its `short_mpl` signature, which resolves to 0-indexed position 1 (`CoreExpressionConfig::outputIndices`) — in the expression `(in2[a,b,f])`, `b` is the output of applying `f` to `a`. `output_args` holds signature argument names, not positions; the loader resolves each name to its position in the signature. Listed in `output_args` in `ConfigVisu.json` and per-config files.

### input indices

The `input_args` / `output_args` lists in the config files give argument positions (1-based in MPL notation, 0-based in some code paths). Used everywhere from digit-arg computation to admission-map lookup. `inputIndices` in `CoreExpressionConfig` is the parsed integer form.

---

## Anchors

An **anchor** is the master atomic expression that fixes the axiomatic context of a theorem. Every theorem starts with a single anchor application.

### AnchorPeano

6 arguments: `(AnchorPeano[N,i0,s,+,*,i1])`. Corresponds to:

- `N` — the set of natural numbers
- `i0` — zero (0)
- `s` — the successor function
- `+` — addition (the function symbol)
- `*` — multiplication
- `i1` — one (1, defined as `s(0)`)

*Seen:* `files/definitions/AnchorPeano.mpl`; referenced by every Peano theorem. Signature computed via `makeAnchorSignature` at [`compiler.hpp`](../../GL_Quick_VS/GL_Quick/src/compiler.hpp).

### AnchorGauss

8 arguments: `(AnchorGauss[N,i0,s,+,*,i1,i2,id])`. Extends Peano with `i2` (the constant 2) and `id` (the identity function, used to form sequence maps in the division-free Gauss summation formula).

### AnchorIncubator

14 arguments: `(AnchorIncubator[N,i0,s,+,*,i1,i2,id,i3,i4,i5,i6,i7,i8])`. 9 of the 14 are `(1)`-typed constants (`i0`..`i8` = the elements `0`..`8`). Used by the incubator pipeline to enumerate ground-level facts over a small finite model. See [Incubator](#incubator).

### AnchorFTA

Anchor for the FTA milestone. Shape TBD as the ladder is climbed. *Seen:* `files/definitions/AnchorFTA.mpl` (scaffolded on current branch).

### anchor tag

The user-facing label selecting which anchor to use for a batch. Passed as the CLI argument to `gl_quick.exe`: `Peano`, `Gauss`, `IncubatorPeano`, `IncubatorGauss`. Maps to `ConfigPeano.json`, `ConfigGauss.json`, `ConfigIncubatorPeano.json`, `ConfigIncubatorGauss.json`.

### anchor handling

The prover step that substitutes concrete anchor-slot names (e.g. raw `1` → `N`, raw `2` → `i0`) for their bound-variable indices. Emits the `anchor handling` tag in the processed proof graph. One emission per chapter maximum. Followed by a series of `anchor handling trace` entries recording the `_copy` substitution chain.

### anchor handling trace

A non-checker tag tracked by the verifier. Records the per-step chain of `_copy` variable rewrites produced by `anchor handling`. Counted but not checked. *Seen:* `verifier.py–2689`.

---

## Proof structure

### theorem

A proved `(>[...](premise)(head))` expression. Survives the prover phase and enters `files/theorems/theorems.txt`, then is further pruned by the compressor. Each theorem is emitted into the processed proof graph as one or more chapters (one for a single-chapter proof, three for induction, one each for mirrored/reformulated variants).

### conjecture

An expression that *looks like* a theorem (same shape) but has not yet been proved. The conjecturer (stage 2 of the pipeline) enumerates conjectures. The CE filter (stage 4) prunes those that fail against the simple-facts tables. Survivors go to the prover.

### premise / conclusion

A premise is one entry in the chain; the conclusion is the head. In a multi-premise implication `(>[α](P1)(>[β](P2)(head)))`, there are two premises (`P1`, `P2`) and one head.

### single-chapter proof

A non-induction theorem whose proof occupies one chapter. Emitted as `<N>_single_chapter_proof.txt`; method label in `global_theorem_list.txt`: `single chapter proof`. The final HTML calls the depiction **Contradiction Proof** when the chapter goal row is tagged `contradiction`, and **Direct Proof** otherwise. A contradiction used only by a later dependency does not change the chapter label.

### induction proof

A theorem proved by induction on one bound variable. Emitted as a triad of consecutive chapters:

- `<N>_induction_typing.txt` — proves the induction variable is in `N` (see [I-18](30_invariants.md#i-18))
- `<N+1>_check_zero.txt` — proves the base case (substitute `0` for the induction variable)
- `<N+2>_check_induction_condition.txt` — proves the step case (assume for `k`, conclude for `s(k)`)

Method label: `induction`. Induction variable recorded in column 3 of `global_theorem_list.txt`.

### induction scope

A scope `<parent>_boundary_is_<chain>_<names>` opened by one witness admission made while an induction hypothesis is eaten forward ([D-519](40_decisions.md#d-519)): a route-1 eat of an admission chain flagged "admitted by induction hypothesis" or a route-5 admission through the strong step's demand. `<chain>` is the admitting key's premise chain with the admitted slot written `marker` (`marker1` / `marker2` for the strong demand), `<names>` the one or two admitted names. The admitted witnesses' facts and the advanced chain live there; eating moves forward one order per path (the first eat at a scope fixes its slot, alternatives for that slot open sibling scopes), an admission whose chain already opened a scope on its path is refused, and only a goal visible at the tree's root leaves it, registered at the root in the same absorb (raw-only `induction scope` row). The scopes carry no assumption; the processed proof graph deletes them from every validity name. Recognized by the payload-prefix test `ScopeSyntax::isInductionScopePayload`. See [I-321](30_invariants.md#i-321).

### int scope

A scope `<parent>_boundary_int_scope_<premise>_<witness>` opened by the first fill of a premise-bound integration copy of a split induction hypothesis ([D-578](40_decisions.md#d-578)): `<premise>` is the key of the ancestor block the copy is bound to, `<witness>` the name that filled the marker. The witness's facts and the copy's integration live there, apart from the other premise's integration; one level only, no owner, no assumption. When every premise of the goal group has its first finished scope (a statement there is an instance of the integration goal), those scopes are promoted to the parent together, each statement with a raw-only `int scope` row; then every int scope of the group is wiped and the group opens no further int scope ([D-582](40_decisions.md#d-582)).

### size marker

A value admitted through a connected integration key of a weak or strong induction step block, by a route-3 commit or a statement fill ([D-583](40_decisions.md#d-583)), and in a strong step block each of the two names a strong demand caught, the smaller value and its distance ([D-584](40_decisions.md#d-584)). Every integration key of a weak step block, and every one a strong step block files at an induction scope or below, must be connected to the induction (hold `rec`, the induction variable, the predecessor of `rec`, the successor of the variable, or a size marker), so an admitted value carries a size bound by `rec`, the variable or a caught value. A size marker the scope knows equal to the induction's zero bounds nothing there and connects no key ([D-584](40_decisions.md#d-584)): an interval admitted with size `rec` makes a later key that holds the interval connected. Recorded per block in `LbMemory::sizeMarkers`; never removed by a scope wipe. Outside the step blocks the same record holds the block's connected values: values admitted through a key connected to the accepting instruction's goal, that is a key naming a head variable of the goal (an argument that is neither one of the anchor's arguments nor a label) or a connected value, or repeating an anchor argument exactly as often as the goal does, more than once; only such keys are filed there ([D-593](40_decisions.md#d-593)).

### one-way witness

A witness that a route-3 commit wrote into an integration instruction, recorded in `LbMemory::oneWayWitnesses` ([D-585](40_decisions.md#d-585)). Its admission fixed its role: the statement fill never tries it, or a name of its class at the deposit scope, as a marker again.

### usefulness test (`main_boundary_is_useful`)

The test of a weak induction step block whose hypothesis head is an operator with a `uniquely_definable` output ([D-521](40_decisions.md#d-521)). The grid build plants the hypothesis premises and the head with a test variable in `main_boundary_is_useful`, a child of the block's `main` ([I-322](30_invariants.md#i-322)). `main`'s rules fire on them there (for row 81 the Peano uniqueness of the successor turns the planted s(rec) = m and the seed s(rec) = n into m = n), and after each standard processing the key, the conjecture premise with the head's operator marked at the output slot, is compared with the head through the scope's classes. A caught step block is switched off at the next barrier (`[STEP-USELESS]` in the run diagnostics log); a test uncaught after the batch's third proof burst is closed. The scope is wiped in both cases.

### test variable (`test_level_<level>_<serial>`)

The open output of an induction hypothesis head planted in the usefulness scope `main_boundary_is_useful` (`WitnessNameKind::Test`, last in canonical preference). It never leaves that scope and leaves the block with the scope's wipe.

### settled induction tree

An induction tree (the induction scopes below one root, the nearest scope that is not an induction scope) in which the induction hypothesis has fired in a leaf ([D-520](40_decisions.md#d-520)). In the burst of the first firing every other scope of the tree is wiped; the leaf and its ancestors stay and the work continues in the leaf; afterwards the hypothesis opens no further induction scope in the tree. Its root is recorded in `LbMemory::inductionSettledRoots`; `Memory::inductionTreeSettled` answers for any scope of the tree.

### forced product

A statement a forced disintegration produced in a block, or one a parent shipped with the from-premise mail flag ([D-522](40_decisions.md#d-522)). A forced disintegration is the one the deposit door performs unconditionally: on a status-0 seed (task, subproof, case and OR-integration premises, contradiction heads, variable copies) or on the conclusion of a hypothesis shot. The disintegrated statement itself is not a product. Listed per block in `LbMemory::forcedProducts`, keyed (expression, scope). The witness-history walk ends, without checking it, at a statement listed at its scope or an ancestor scope, or whose canonical form there equals a listed product's: a connection that comes from a forced disintegration does not link a witness to an instruction's path.

### strong induction

An induction proof whose step hypothesis is the theorem at every value `m` below the induction variable `n` (bound `m+x=n`, `m!=n`), with every non-anchor variable except `n` universal, instead of the theorem at the predecessor only. Same triad of chapters and the same zero case; method label `strong induction`. Enabled per batch by `strong_induction` ([D-500](40_decisions.md#d-500)). The step block also holds the multiplication catcher, the same hypothesis behind `x*m=n`, `m!=n`, which catches a cofactor ([D-566](40_decisions.md#d-566)). Built only for an induction variable whose premises are properties, relations to another induction variable or set operators, in a lemma without an operator of non-set output ([D-594](40_decisions.md#d-594)); of such candidates a lemma may have one, or two whose exchange leaves its premises as they are, and none at three or more ([D-595](40_decisions.md#d-595)).

Real example of the triad — chapter 11 (typing) + 12 (zero) + 13 (step) on the current branch:

```text
# 11_induction_typing.txt (first line only)
(in[v1,N])	main	implication	(>[v2,v3](in2[v2,v3,s])(in[v2,N]))	main	(in2[v1,i0,s])	main
```

### reformulated statement

A theorem produced by expanding an `existence`-category head into `left + right` elements, then permuting non-anchor premises. Method label: `reformulated statement`.

*Seen:* `103_reformulated_statement.txt`:

```text
(>[N,i0,s,+](AnchorGauss[N,i0,s,+,*,i1,i2,id])(>[v1,v2](in2[v1,v2,s])(>[v3](interval[N,+,i0,v2,v3])(existence4[N,+,v3,v1,i0]))))	main	reformulated from	(>[N,i0,s,+](AnchorGauss[N,i0,s,+,*,i1,i2,id])(>[v1,v2,v3](limitSet[N,+,v1,v2,v3])(>[v4](in2[v2,v4,s])(>[](interval[N,+,i0,v4,v1])(interval[N,+,i0,v2,v3])))))	main
```

### back-reformulated statement

Incubator-only. An operator-equality theorem (e.g. `(=[+[a,b], c])`) is rewritten into the direct operator form `(in3[a,b,c,+])`. Method label: `incubator back reformulation`.

### OR theorem

A theorem whose head is an `or<N>` node (disjunction). Distinct from [`or branch proven`](#or-branch-proven), which is the per-branch case-split bookkeeping row inside a proof; `or theorem` is the chapter-conclusion record for theorems whose statement IS an OR. Emitted as `<N>_or_theorem.txt`. Method label: `or theorem`.

### pre-split (or elimination)

The maintainer's standing method for a theorem needing a case split on a bound premise variable that the in-prover cohort machinery cannot open: state the two GUARD VARIANTS (the theorem plus one binder-free innermost guard premise each — e.g. `1≤b` and `b=0`) as ordinary flat pool rows, and DERIVE the unguarded theorem by classical or-elimination at the phase-4 drain seam (`constructOrEliminationInRun`), licensed by a proved or theorem whose two disjuncts are exactly the two guards. The derived row carries method label `or elimination` and a fabricated chapter `<N>_or_elimination.txt` citing both variants plus the licensing or theorem. The full theorem is never a pool conjecture (a pool copy would spin an LB the whole run for nothing); the variants are never subsumed. See [D-289](40_decisions.md#d-289) and the third unfair-advantage shape in `docs/fta_ladder/README.md`.

A degenerate guard variant often has a mathematically redundant premise (13b's `a|b`), so its closure trips the D-278 level gate; such variants land in the **proved-not-broadcast tier** — method label `proved not broadcast`, a real single-chapter walk, zero circulation (no reformulation, broadcast, or-seeding, compressor participation, or `theorems.txt` row) — and the merge consumes them as variants like any proved row. See [D-290](40_decisions.md#d-290).

### or branch proven

OR-integration bookkeeping row. Records that one `_orint_` subproof has proved the parent OR goal through its selected disjunct. Row layout: `<or-expr> <parent-ns> or branch proven <selected-disjunct> <subproof-ns>`. The historical tag name says "branch", but this is not an `_ordis_` case-split row. Promoted to a first-class `TAG_CHECKERS` entry by [D-35](40_decisions.md#d-35) (previously claimed retired in the SwDD; the override path never existed and the prover always emitted the tag live). Checker: `check_or_branch_proven`.

### or branch assumption

OR-integration premise row. Records that in an `_orint_` subproof targeting disjunct `D_i`, the negation `!D_j` of every other disjunct (`j ≠ i`) is seeded as a local assumption. Row layout: `<negated-other-disjunct> <subproof-ns> or branch assumption <or-expr>_integration_goal <parent-ns>`. The historical tag name says "branch", but this is not an `_ordis_` case-split row. Promoted to first-class checker by [D-35](40_decisions.md#d-35). Checker: `check_or_branch_assumption`.

### multiplied from

A theorem produced by `multiplyImplication`: a parent theorem with some of its `(1)`-typed bound variables identified (equalised) according to a Bell partition. Each partition of the bound-variable set becomes one "multiplied" copy; a merged class keeps its outermost binder, so the copy binds each name once along each scope chain. The verifier checker is `check_equalize_variable` at [`verifier.py`](../../verifier.py) — the function name reflects the underlying algorithm. Historically a redundant `equalize variable` registry alias also pointed at the same function but was never emitted; it was removed from `TAG_CHECKERS` so the registry now has one entry per unique tag.

### externally provided theorem

A theorem loaded from `files/theorems/externally_provided_theorems.txt` (user-supplied). Counts as an axiomatic boundary within a run — the current prover batch treats it as a given implication rule. On the next batch, theorems proved in the previous batch are also loaded as external theorems.

---

## Logic blocks & memory

### Logic Block (LB)

The unit of execution in the GL grid. Each LB has local memory (the `Memory` instance), its own hash index, its own queue of pending work, and its own mail. LBs communicate between cycles via `mailIn` / `mailOut`, not during them. In the current single-process execution LBs are C++ objects dispatched across real worker threads (`proveKernel`'s barriered phase sweeps over `logicalCores`; a heavy LB's hashburst is itself split into parallel `(LB, part)` tasks); the *architecture* is designed for fuller parallel distribution (ASIC roadmap).

### Memory (class)

The per-LB state object. Formerly called `BodyOfProves`. Defined at [`memory.hpp`](../../GL_Quick_VS/GL_Quick/src/memory.hpp). Key fields:

| Field | Purpose |
|---|---|
| `exprKey` | The expression this LB is responsible for. |
| `intKnownStatements` | Packed-key statement registry; per-row membership bits `registered` / `known`. |
| `intStatementLevelsMap` | `packStatementKey(originalId, validityId) → std::set<int>`. Per-statement set of LB depths (`Memory::level`) whose state contributed to deriving that statement — anchor=0, child=1, grandchild=2, …. Read by the `allLevelsInvolved` registration verdict (goal closure itself is level-free). See [`levels`](#levels). |
| `overallHashMemory` | The local `HashMemory` — indexed implications + statements. |
| `nameMap` | The `NameMap` instance (scope-name encoding). |
| `stackOfValidity` | The scope stack for this LB. |
| `mailIn` / `mailOut` | Per-LB mail **staging buffers** (pull model): `mailOut` is filled by `fillMailOut` and drained by the commit barrier into the LB's [`MailLog`](#maillog) log; `mailIn` is filled by the phase-1 pull from ancestors' logs. No longer routed by `sendMail`/`smashMail` (deleted). |
| `orAdmissionSet` | OR branch admission bookkeeping. |
| `equivalenceClassesMap` | Per-validity equivalence class registry. |
| `parentMemory` / child pointers | Hierarchy (integration, hypothesis, branch). |
| `primedForContradiction` | Keeps LB alive in `deactivateRecursively` for contradiction discharge. |

### LogicalEntity

Metadata struct for an expression at the definition level. Fields: `category` (`atomic` / `and` / `existence` / `implication`), `elements` (constituent sub-expressions), `signature` (canonical form with `u_` args), `arity`, `definedSet`. Defined at [`memory.hpp`](../../GL_Quick_VS/GL_Quick/src/memory.hpp). Stored in `compiledExpressions` (a global map from expression name → `LogicalEntity`).

### LocalMemoryValue

Value type in `HashMemory::encodedMap`. Fields: `value` (normalised conclusion template), `levels` (the rule's installation `std::set<int>` — see [`levels`](#levels); for rules arriving through mail it MUST be empty per [I-51](30_invariants.md#i-51)), `originalImplication`, `justification`, `key`, `remainingArgs`, `validityName`. Defined at [`memory.hpp`](../../GL_Quick_VS/GL_Quick/src/memory.hpp).

### Mail

The inter-LB message buffer. Fields:

- `statements` — list of `(ExpressionWithValidity, std::set<int>)` pairs to be delivered. The deposited level set MUST be empty per [I-51](30_invariants.md#i-51): mail-side levels propagate via the receiver's `addToHashMemory` → `LocalMemoryValue::levels`, and any non-empty deposit pollutes the union that the `allLevelsInvolved` registration verdict reads ([D-77](40_decisions.md#d-77)).
- ~~`implications`~~ — retired ([D-78](40_decisions.md#d-78)). Was a list of tuples (chain, head, args, levels, theorem); levels field always empty. Implications now travel as the compact `(implication<N>[…])` form on `statements`.
- `exprOriginMap` — provenance entries to merge into the recipient's origin map.

Defined at [`memory.hpp`](../../GL_Quick_VS/GL_Quick/src/memory.hpp). (`expandedImplications` and `disintegrationSignals` are two further fields; the cross-LB pull carries only `statements` + `exprOriginMap`.) Each `Memory` has a `mailIn` and `mailOut` — now **per-LB staging buffers** for the pull model ([`MailLog`](#maillog)), not routed outboxes. Statements are main-scope-only by contract (every emission gated on `validityName == "main"`; absorb hard-codes `"main"`).

### MailLog

The pull-model cross-LB mail carrier — a heap `std::` struct on `Prover` ([`mail_log.hpp`](../../GL_Quick_VS/GL_Quick/src/mail_log.hpp)), replacing the retired push routing (`sendMail` / `smashMail` / per-core `boxes`). Two maps: `batches` (`const Memory* → vector<Mail>`, each LB's append-only log of the mail batches it emitted this execution batch — the single stored copy) and `cursor` (`recipient → ancestor → count`, the per-(recipient, ancestor) ingested-batch counter). Each LB is registered at grid build; the commit barrier (`proveKernel` post-join seam) appends each LB's `mailOut` to its log; the phase-1 pull (`MailLog::pull`) merges un-ingested ancestor batches into the receiver's `mailIn`. See [D-137](40_decisions.md#d-137), [I-94](30_invariants.md#i-94), [`03_mail_system.md`](20_core_concepts/03_mail_system.md).

### InternalMail (retired 2026-05-07 — merged into Mail)

`InternalMail` was a separate struct on `Memory` (alongside `Mail mailIn`/`mailOut`) used as the per-LB integration-side revival inbox. Its `statements` was a `set<tuple<string, levels, validityName>>` carrying scope per element. As of 2026-05-07 ([D-53](40_decisions.md#d-53) unification, renumbered from main's D-46), the channel uses the same `Mail` struct as the routing channels — `Memory::sameIterationInternalMail` is now type `Mail` with `statements` element type `pair<ExpressionWithValidity, levels>`, and `struct InternalMail` has been deleted. The variable name `sameIterationInternalMail` and the function name `emitIntegrationRevivalToInternalMailIn` survive. See [`Mail`](#mail), [D-19](40_decisions.md#d-19), [D-53](40_decisions.md#d-53), [I-21](30_invariants.md#i-21).

### parent / child hierarchy

Every non-root `Memory` has a `parentMemory` pointer. Child LBs are created when:

- A new hypothesis is assumed (hypothetical disintegration).
- An OR case split creates one branch per disjunct.
- An integration scope is opened.

The hierarchy structure determines what can be escalated where — a conclusion reached in a child scope cannot be asserted in the parent until integration rebinds its context. See [Scope, validity, & the name map](#scope-validity--the-name-map).

---

## Hash engine & admission

### HashMemory

The per-LB index. Defined at [`memory.hpp`](../../GL_Quick_VS/GL_Quick/src/memory.hpp). Three main maps:

- `encodedMap` — `IntNormalizedKey → list<LocalMemoryValue>`. The actual implication-rule registry: given a premise signature (encoded), look up the conclusion templates.
- `admissionMap` — bookkeeping for Pass B output-slot admission; populated by [`updateAdmissionMap`](#updateadmissionmap).
- `rejectedMap` — expressions that have been rejected for admission (pruning lookahead).

Conceptually a trie keyed by integer-normalised expression hashes.

### encodedMap

Where implications live. When a `(premise → conclusion)` rule enters the system (via `addToHashMemory`), it is normalised, keyed by the premise's normalised integer hash, and stored. When a statement is deposited, its hash is looked up in `encodedMap`; any matching entry fires.

### EncodedExpression

Pre-parsed form of an expression. Fields: `name`, `negation`, `arguments` (with iteration/level), `maxIterationNumber`, `original`, `validityName`. Formerly the key of the goal registry; since the registry re-keyed to packed `(originalId, validityId)` int32 keys (`Memory::intToBeProved`), `EncodedExpression` is a transient parse record at boundaries. Defined at [`memory.hpp`](../../GL_Quick_VS/GL_Quick/src/memory.hpp).

### levels

A `std::set<int>` attached to every deposited statement (run-form on the kernel chain, [I-136](30_invariants.md#i-136)). **Each integer is the depth (`Memory::level`) of an LB whose state contributed to the derivation of the statement.** Level 0 is the anchor LB; level 1 is its direct child; level 2 is a grandchild; and so on. A statement gets level `k` in its set if any premise of its derivation was either installed at, or itself carried level `k`. The levels of a derived statement are computed as the union of every premise's levels (or the union of `(rule's installation levels) ∪ (premise levels)` when a hashmem rule fires).

A statement's stored levels row is NEVER empty ([I-182](30_invariants.md#i-182)): a NON-DERIVED statement — a loaded fact, an anchor, a broadcast theorem compact, an assumed premise element — carries the singleton `{-1}`, the non-derived tier. `-1` is transparent to level accounting (it never enters a derivation union and gates like the former empty run) and singleton-only (a run is exactly `{-1}` or all-values-≥0, never mixed).

Not a monotonic counter. Not an admission ordering. The earlier glossary entry calling this `(newStatementLevel, newEqualityLevel)` was stale — corrected on the branch in the same commit that fixed [D-77](40_decisions.md#d-77).

**Why it matters.** `logical_block/goal_processor.hpp::dischargeToBeProved` reads the derived statement's `levels` when a MAIN statement matches a `toBeProved` goal and computes the registration verdict `allLevelsInvolved`:

```cpp
bool allLevelsInvolved = (lvN == memoryBlock.level + 1);
if (lvN == memoryBlock.level && !has0)
    allLevelsInvolved = true;
```

The primary branch (`lvN == level+1`) requires the derivation to have touched **exactly** every LB from the anchor (level 0) down to the producer (level = `memoryBlock.level`). The alternate branch (`lvN == level`, no level 0) covers derivations that ignore the anchor scope. Since [D-278](40_decisions.md#d-278) the verdict gates ONLY the global registration: goal CLOSURE (goal-row erase, scope wipes, twin retirement, deactivation) is level-free, and the verdict rides the sealed `UpdateGlobalSingleChapterRec` to the post-join drain, which on false runs the lifecycle and refuses `appendGlobalTheorem`. A levels row with even one extra element (e.g. a level the LB has no ancestor for) fails both branches — the goal still closes, but no theorem reaches `globalTheoremList`. The contradiction-twin route seals its own verdict from the colliding pair's level-row union (excluding the twin's own level, the assumed seed's tier). An induction triad's step and base closures do not use this count: each reports its levels as a bit mask through `updateGlobalTuples`, and `TheoremPublisher::updateGlobal` registers the induction only when the union of both masks and the typing fact's level row covers the conjecture's premise levels (`ScopeLifecycle::inductionPremiseLevels`: levels `1..n-1` of the chain, the induction variable's own membership premise included, without the anchor level and the step or base block's own level) — [D-516](40_decisions.md#d-516). See [G-46](50_gotchas.md#g-46) and the fix in [D-77](40_decisions.md#d-77).

**Invariant for producer-side deposits.** Any mail deposit that ships a rule across LBs (`Mail::statements`, retired `Mail::implications`, or any future replacement) **must** use `std::set<int>` (empty) as the deposited level set. The receiver computes the rule's effective levels from its own scope on installation; the mail-side levels are NEVER additive into the derived-statement union. The tuple channel (deleted on this branch) historically used empty sets at every insert site (`prover.cpp` etc.); the D-76 compact-form drain learned this the hard way. See [I-51](30_invariants.md#i-51).

### intStatementLevelsMap

`std::unordered_map<int32_t, std::set<int>>` keyed by `packStatementKey(originalId, validityId)` — one entry per `(expression, validityName)` pair installed in the LB. Holds the per-statement `levels` set described above. Read by `logical_block/goal_processor.hpp::dischargeToBeProved` to compute the `allLevelsInvolved` registration verdict (packed probe when the int ids are in hand, the non-minting `lookupStatementLevels` otherwise); written by every site that constructs or updates a statement's level set (Anchor handling, rule firing, mail absorb, equivalence-class rewrites). Stored on `Memory`; never crosses LBs verbatim (the receiver re-builds its own per-statement entries from premise-side levels). The hashburst dump prints its section by decoding the keys and lex-sorting on `(original, validityName)` — byte-identical to the former `std::map<EncodedExpression, std::set<int>>` iteration order (Rule 14).

### intKnownStatements

The packed-key statement registry of an LB: `packStatementKey(originalId, validityId)` → `StatementFlags`. Carries two membership bits — `registered` (the statement passed an add-path registration door) and `known` (the statement entered the level registry; the Site F dedup record) — plus `local` and `fullyDisintegrated`. Gates test the bit their contract names, never bare presence. See [I-85](30_invariants.md#i-85).

### wholeExpressions

Retired — the former string-keyed statement registry, folded into `intKnownStatements` ([D-128](40_decisions.md#d-128)); the interim `registered` membership bit and the dump's `wholeExpressions` section are retired too ([D-264](40_decisions.md#d-264)): row presence is the one membership.

### admission map

`admissionMap` in `HashMemory`. Populated by [`updateAdmissionMap`](#updateadmissionmap) from consumer-side registrations (output-slot markers). Queried by [`isAdmitted`](#isadmitted) on the producer side to decide whether a freshly-minted `it_…` variable should be committed. **Key shape:** `u_`-prefixed on all non-marker args (template form).

### `admissionMapIntegration`

Integration-side counterpart in `HashMemory`. Populated at `logical_block/integration_processor.hpp` from `le.signature` during `prepareIntegrationCore2` Case C. Queried by [`isAdmittedIntegration`](#isadmittedintegration). **Key shape:** `u_`-prefixed on all non-marker args — same as `admissionMap`, built from `le.signature` which is already in u_ form. The `u_`-transform in `isAdmittedIntegration` and in `applyEquivalenceClassToRejectedMapIntegration` is what bridges Pass B's bare marker form to this template form for lookup. See [G-32](50_gotchas.md#g-32) for the silent-miss pitfall on key-shape mismatch.

### `admissionSetIntegration`

Secondary admission set in `HashMemory`. Populated at `compiler/definition_expansion.cpp` via `removeUPrefixFromArguments(mappedElement)`. **Key shape:** bare concrete with `repl_` preserved (no `u_`). Queried directly by Pass B `int_` branch as a fallback when `admissionMapIntegration` misses. Lookup uses Pass B's bare marker form as-is — no transform needed.

### `admissionMapIntegrationDouble`

Double-marker integration demands in `HashMemory` ([D-492](40_decisions.md#d-492)). One key per OR disjunct with two distinct open arguments of a flattened root integration goal: `marker1` at the earlier open position, `marker2` at the later, every other argument `u_`-prefixed. The value is 0 and carries no information; the key carries no instruction and nothing consumes it. Probed by `isAdmittedIntegrationDouble` (route 5).

### `isAdmittedIntegrationDouble` (admission route 5)

Defined at [`logical_block/admission_memory.hpp`](../../GL_Quick_VS/GL_Quick/src/logical_block/admission_memory.hpp), `AdmissionMemory::isAdmittedIntegrationDouble`. Takes Pass B's bare single-marker constituent, turns its marker and each other argument in turn into `marker1`/`marker2` (the earlier position first), prefixes the remaining arguments with `u_` and matches `admissionMapIntegrationDouble` class-aware at the probe scope and every strict ancestor. A hit admits an ordinary `int_` witness, together with its unadmitted enclosing witnesses when the facts of the path link them to it ([D-501](40_decisions.md#d-501)); the strong step's own key admits only for an original whose arguments are the lemma's input values, sets and operators ([I-285](30_invariants.md#i-285), [D-502](40_decisions.md#d-502)). A `doubleOnly` slot — an integration slot whose source has exactly one algebra-witness input and a constituent with a visible double-marker predicate — is admitted through this route only ([I-302](30_invariants.md#i-302)).

### Wildcard integration key (`pi_`)

An `admissionMapIntegration` key of the weak hypothesis's two-marker start ([D-586](40_decisions.md#d-586)) for a part that holds both open arguments of its goal: `marker` on the path's argument and `pi_` on the other one, for example `(fold[u_1,u_3,u_4,u_5,pi_,u_2,u_rec,marker])`. `pi_` matches any value: a query matches when its variant with `pi_` at one non-marker position equals the key (`AdmissionMemory::makeWildcardQueryScratch`), probed only in a block that filed such a key (`AdmissionMemory::integrationWildcardFiled`). A match fills the marker only; the instruction keeps the other argument open.

### `isAdmittedIntegration`

Defined at [`logical_block/admission_memory.hpp`](../../GL_Quick_VS/GL_Quick/src/logical_block/admission_memory.hpp), `AdmissionMemory::isAdmittedIntegration`. Pass B's integration-admission query. Takes Pass B's bare marker form, adds `u_` prefix to every non-marker arg, then looks up `admissionMapIntegration`. On match, unfolds the admission template via `prepareIntegrationCore2`. Returns `true` iff the var was admitted.

### template key (packed admission / rejection key)

The `int64` key every admission and rejection map is indexed by: `packStatementKey(templateId, validityId)` — the marker-form template interned in the dedicated `Memory::templateInterner` (high half) and the scope interned in `Memory::nameMap` (low half). Minted by `mintTemplateKey` at write sites, probed by the non-minting `lookupTemplateKey` ([I-89](30_invariants.md#i-89)). Because the scope is half of the key, a park and a demand meet only at the SAME scope by key match; the ancestor walk of `D-341` re-probes the same template under each strict ancestor of the demand's scope.

### demand-built rule

A rule recognised by `analyzeDemandBuiltRule` at installation and not written into hash memory. Its instance is built, as the firing would build it, when something needs it. There are two kinds:
- An anchor operator's totality rule: typing premises of distinct bound variables only, and an existence head defined by one application of a configured operator with the witness at the output (`+`, `*`). `RejectedSourceReplay::buildDemandedInstances` builds the instance when a demand for the operator's value appears.
- A typed rule with an OR head: an optional anchor premise, typings of at least two variables, and a positive OR over them (a ≤ b ∨ b ≤ a, trichotomy). The instance is built when a fact refutes a disjunct, a case on a disjunct is demanded, or the instance is a goal.

See [D-596](40_decisions.md#d-596), [D-597](40_decisions.md#d-597), [I-346](30_invariants.md#i-346), [I-347](30_invariants.md#i-347).

### rejected map (algebra)

`rejectedMap` in `HashMemory`. Integration-side buffer for `it_…` vars that failed Pass B admission. Write via [`updateRejectedMap`](#updaterejectedmap) at `logical_block/rejected_source_replay.cpp`. Revisit via [`revisitRejected2`](#revisitrejected2) at `logical_block/rejected_source_replay.cpp` when a new admission key arrives — at the key's scope (erased) or at a strict ancestor of it (kept, released into the key's scope). **Key shape:** bare concrete (matches Pass B's `makeMarkedExpr` output). Revisit stages the stored cohort on `sameIterationInternalMail`; `revisitInProgress` guards the exact-scope path.

### `rejectedMapIntegration`

Integration-side counterpart to `rejectedMap` in `HashMemory`. Added in [D-19](40_decisions.md#d-19). Write via `updateRejectedMapIntegration` at `logical_block/rejected_source_memory.hpp` from Pass B's `int_` branch when admission fails. Value is [`RejectedMapIntegrationValue`](#rejectedmapintegrationvalue). **Key shape:** bare concrete — same as `rejectedMap`. Two revival paths: [`applyEquivalenceClassToRejectedMapIntegration`](#applyequivalenceclasstorejectedmapintegration) (eq-class rewrite) and [`revisitRejectedIntegration2`](#revisitrejectedintegration2) (new-admission-key trigger). Both deposit to `sameIterationInternalMail`, not via direct `addExprToMemoryBlock`. See [I-22](30_invariants.md#i-22): the admission-map entry is **not** erased on revival (asymmetric with algebra).

### `rejectedMapSubproof`

Parked implication subproofs in `HashMemory` ([D-406](40_decisions.md#d-406)). An implication element of an and whose **gate element** — a non-implication element of that and with every argument fixed (atomic, negated or compound) — is not known at the preparing scope or an ancestor does not open its boundary subproof; `openImplicationSubproof` files it here instead (`parkImplicationSubproof`). **Key shape:** the gate element's clean ground text (u_ stripped), normalized through the preparing scope's classes (canonical). **Value:** the `admissionMapIntegration` family (`IntegrationEntry`), in the instruction's own text (never re-canonicalized — the and's integration rule expects that text): entity 0 the implication, entities 1..k its parent ands, `markedGoal` the root goal of a main-level opening. **Revival:** every newly known row (`stageSubproofArrival`) and every scope whose class view changed (`stageSubproofRecheck`) is staged and handled at step 4b by `drainSubproofArrivals`, which first moves parks under changed scopes to their canonical keys (`rekeySubproofPark`); `revisitRejectedSubproof` reopens the records as stored through `openImplicationSubproof`, which parks again under the next unknown gate element. Guard `subproofRevisitInProgress`. Two other record kinds share the map and its revival: a parked negated-premise reductio (category `reductio`, [I-326](30_invariants.md#i-326)) and a parked contradiction-scope demand candidate (category `contradiction_demand`, its elements the candidate's premises; see contradiction-scope demand).

### integration trial scope

A scope `<parent>_boundary_integration_<names>` that holds the witnesses one candidate integration admission mints until its instruction's requirement holds; `<names>` are those witnesses joined by `_` (experiment, [D-481](40_decisions.md#d-481)). When every admitting route of a top-level integration witness is a gated `admissionMapIntegration` instruction, the admission's new names are minted, but their facts are published only there. An instruction is gated when it leaves a **gate element** of one of those names unknown: a non-implication element of the ands at the name's place in the instruction, other than the matched key, naming only fixed values and the name, for example the fold's `in2[u_10,u_14,marker]`. Inside the trial scope the witnesses only fire and join equality classes: nothing is minted there, and no case split, subproof, integration or goal opens. The record (`LbMemory::integrationTrials`) lists the witnesses and the trial's expressions (`LbMemory::integrationTrialContents`). Once one instruction's gate elements are all known there, `drainIntegrationTrials` passes the record. It replays the source at the parent, which admits the listed witnesses under the same names with the listed expressions, and wipes the trial scope. Recognized by the payload-prefix test `ScopeSyntax::isIntegrationTrialPayload`. See [I-298](30_invariants.md#i-298).

### `RejectedMapIntegrationValue`

Value struct for `rejectedMapIntegration`. Fields:

- `concreteConstituent` — body element with `u_` stripped, `int_` arg intact (marker position holds the rejected `int_` var).
- `siblings` — all other body elements of the same compound (including `(in[…])` typing), concrete form.
- `compoundExpression` — original compound for level derivation and origin provenance.

No `iteration` field — `int_` mint uses `level + startInt` only (`memory.hpp`); iteration is an `it_`-only concept.

### `updateRejectedMap`

Defined at [`logical_block/rejected_source_memory.hpp`](../../GL_Quick_VS/GL_Quick/src/logical_block/rejected_source_memory.hpp), `RejectedSourceMemory::updateRejectedMap`. Algebra-side writer. Inserts a `RejectedMapValue` into `rejectedMap` keyed by `(markedExpr, validityName)`. Called from Pass B deferred-commit in `logical_block/disintegration_processor.cpp`, `DisintegrationProcessor::disintegrateExpr2`.

### `updateRejectedMapIntegration`

Defined at [`logical_block/rejected_source_memory.hpp`](../../GL_Quick_VS/GL_Quick/src/logical_block/rejected_source_memory.hpp), `RejectedSourceMemory::updateRejectedMapIntegration`. Integration-side writer. Inserts a `RejectedMapIntegrationValue` into `rejectedMapIntegration` keyed by `(markedKey, validityName)` and maintains the `varsInRejectedMapIntegrationKeys` cache. Called from Pass B deferred-commit in `logical_block/disintegration_processor.cpp`, `DisintegrationProcessor::disintegrateExpr2` (section alongside algebra commit).

### `revisitRejected2`

Defined at [`logical_block/rejected_source_replay.cpp`](../../GL_Quick_VS/GL_Quick/src/logical_block/rejected_source_replay.cpp), `RejectedSourceReplay::revisitRejected2`. Algebra-side revival entry point — called after an admission-map insert to re-process rejected entries at that marker key. Two paths: the exact-scope cohort is snapshotted, erased and mailed on `sameIterationInternalMail` at the key's scope (the `equality1`-door revival; no `addExprToMemoryBlock` call — mail-only since D-19's discipline reached both sides); then every cohort parked at a strict ancestor of the key's scope is released into the key's scope and kept (`releaseAncestorRejected2`, `D-341`). Does not consume the admission key ([I-200](30_invariants.md#i-200)). Cyclic (can re-enter via `updateRejectedMap`); guarded by `revisitInProgress`.

### `revisitRejectedIntegration2`

Defined at [`logical_block/rejected_source_replay.cpp`](../../GL_Quick_VS/GL_Quick/src/logical_block/rejected_source_replay.cpp), `RejectedSourceReplay::revisitRejectedIntegration2` (near `revisitRejected2`). Integration-side counterpart. Called after `admissionMapIntegration` insert at `logical_block/rejected_source_replay.cpp` (with u_-strip on the key) and after `admissionSetIntegration.insert` at `logical_block/statement_memory.cpp` (key already bare). Emits matching `rejectedMapIntegration` entries to `sameIterationInternalMail`. Does **not** call `addExprToMemoryBlock`; leaves the admission entry live ([I-22](30_invariants.md#i-22)).

### `applyEquivalenceClassToRejectedMapIntegration`

Defined at [`logical_block/equivalence_admission_rewriter.hpp`](../../GL_Quick_VS/GL_Quick/src/logical_block/equivalence_admission_rewriter.hpp), `EquivalenceAdmissionRewriter::applyEquivalenceClassToRejectedMapIntegration` (alongside `applyEquivalenceClass`). Equivalence-class-driven revival path. For each rmi entry whose key args overlap the class, iterates `allMappingsAna[(|indices|, |eqList|)]` (full permutation, same as main helper), dedups per-entry, probes each unique rewritten key against `admissionMapIntegration` (u_-transformed) and `admissionSetIntegration` (bare). On match: emit constituents to `sameIterationInternalMail`. On no-match: persist rewritten entry at the new key. Called at three sites mirroring `applyEquivalenceClass`: same-NS, ancestor-NS, fixpoint re-iter. Gated on `!parameters.skip_eq_classes`.

### `varsInRejectedMapIntegrationKeys`

`std::unordered_set<std::string>` in `HashMemory`. Monotonically-growing cache of non-marker args present in any `rejectedMapIntegration` key. Used by `applyEquivalenceClassToRejectedMapIntegration` to short-circuit when the class has zero overlap with any stored key — saves an O(|rmi|) walk per class call. Never shrinks (false positives just cost a walk).

### `allMappingsAna`

`std::map<std::pair<int,int>, std::vector<std::vector<int>>>` in `Prover`. Precomputed permutation tables keyed by `(numberOfIndicesToSubstitute, sizeOfEqClass)` — for `(k, n)` contains all `k`-length sequences over `{0..n-1}` (with repetition). Used by both `applyEquivalenceClass` and `applyEquivalenceClassToRejectedMapIntegration` to enumerate all ways to substitute class args into the positions they occupy in a target expression. Bounded by `parameters.sizeAllPermutationsAna` (default 7).

### `isAdmitted`

Defined at [`logical_block/admission_memory.hpp`](../../GL_Quick_VS/GL_Quick/src/logical_block/admission_memory.hpp), `AdmissionMemory::isAdmitted`. Checks whether a variable is admitted via the admission map, with multi-occurrence and pattern-matching logic. Authoritative admission path.

### `isAllowedAsOperatorInput`

Defined at [`logical_block/admission_policy.hpp`](../../GL_Quick_VS/GL_Quick/src/logical_block/admission_policy.hpp), `AdmissionPolicy::isAllowedAsOperatorInput`. Fallback admission path for `it_…` (Pass B) variables. Fires only when all these hold:

- The expression's head name is in `CompiledDefinitionRegistry::operatorNames`.
- `cfg.inputIndices.size == 1` — single-input operators only.
- The lone input-arg position holds the variable.
- Standard guards (max iteration number, max secondary variables) pass.

**The single-input gate is the guardrail against RT explosion.** Widening to multi-input operators broke Gauss summation. See [I-6](30_invariants.md#i-6).

### `updateAdmissionMap`

Defined at [`logical_block/admission_memory.hpp`](../../GL_Quick_VS/GL_Quick/src/logical_block/admission_memory.hpp), `AdmissionMemory::updateAdmissionMap`. Inserts admission-map entries for operator outputs with renamed keys and remaining-args tracking. Called during Pass B processing on the consumer side.

### `reconstructImplication`

Defined at [`compiler/implication_builder.hpp`](../../GL_Quick_VS/GL_Quick/src/compiler/implication_builder.hpp), `ImplicationBuilder::reconstructImplication`. Rebuilds an implication from a chain + head. Binds **only** variables appearing more than once across chain+head in `>[...]`. Skips `u_`-prefixed variables. Used for theorem-level reconstruction (`addTheoremToMemory`, `updateGlobalSingleChapter`, back-reformulation).

### `reconstructImplicationFullBind`

Defined at [`compiler/implication_builder.hpp`](../../GL_Quick_VS/GL_Quick/src/compiler/implication_builder.hpp), `ImplicationBuilder::reconstructImplicationFullBind`. Like `reconstructImplication` but binds **all** non-`u_` variables universally in `>[...]`. Used only at disintegration/integration sites — where expressions come from GL-binary expansion and all numbered variables must be universally quantified. See [I-4](30_invariants.md#i-4).

---

## Scope, validity, & the name map

### validity name (validityName)

A string identifier for a proof scope. Every deposited statement carries a `validityName`. The root scope is the literal string `"main"`. Non-main scopes are minted by pushing payloads onto a stack rooted at `main`.

### NameMap

The per-LB encoder for validity names. Defined in [`memory.hpp`](../../GL_Quick_VS/GL_Quick/src/memory.hpp). Maps canonical strings to `int16_t` IDs through the cold name table; ancestor queries (`verdict` / `comparable`) derive from the per-scope `ancestorsOf` lists.

### NameMap::encodePush

The authoritative method for minting a new scope. Takes a `parentId` and a `payload`, registers ancestors, returns a fresh `int16_t`. **Every non-`main` validityName must be minted via `encodePush`** — raw string concatenation bypasses `ancestorsOf` registration and produces orphan roots that break `comparable` / `deeperOf`. See [I-2](30_invariants.md#i-2).

### stackOfValidity

The per-scope payload-sub-id stack — derived from the paged parent-pointer forest (`validityNodes`): walk the scope's parent chain, collecting each node's own sub-id. Backs scope-depth comparisons with `ancestorsOf` (`comparable`, `deeperOf`).

### ancestorsOf

The per-scope ancestor-id list (root-first, self at the back) — derived by walking the paged parent-pointer forest (`validityNodes`) from the scope up to its root. `verdict` / `comparable` / `deeperOf` derive from membership in it.

### pairMap

(Retired — superseded by `ancestorsOf`.) Formerly an internal NameMap hash of parent-child links; scope-ancestor queries now derive directly from the per-scope `ancestorsOf` lists. See [D-138](40_decisions.md#d-138).

### encoding payload

The opaque string pushed onto `encodePush`. The prefix of the payload conventionally encodes the scope kind — the *role* of the new scope (hypothesis, integration goal, OR branch, sentinel). The scope kind must live in the payload, not in a Memory side-table..

### "main" scope

The root scope ID. Every theorem's outermost conclusions live in `main`. A well-formed proof graph has every `task formulation` and every promoted conclusion tagged with namespace `main`.

### branch scope

A scope ID minted when an OR disintegration creates a new branch. Each disjunct gets its own scope. Conclusions proved inside a branch scope stay there until OR convergence promotes them to the parent.

### integration scope

A scope ID minted when an integration operation opens. Used by the reformulation-for-integration machinery to stage conclusions before committing them to the parent scope.

---

## Disintegration & integration

### compilation

Proof tag (added, [D-76](40_decisions.md#d-76)). Records conversion of an expanded implication to its compact named form. Implications travel through `Mail::statements` as compact carriers; the separate implications channel is retired ([I-54](30_invariants.md#i-54)). The row's conclusion is the compact expression and its single dependency is the compiled definition's canonical expanded reconstruction ([I-52](30_invariants.md#i-52)). Outgoing implication statements remain main-scope only ([I-26](30_invariants.md#i-26)). Checker: `check_compilation` — structural, in `_ORIGIN_EXEMPT_TAGS` (the compact-to-expanded link is definitional via the GL binary). Purely additive provenance; does not change which theorems prove.

### disintegration

The operation of splitting a compound expression into its elements. Used when:

- An `and` node is received — split into individual conjuncts.
- An `existence` node is received — split into the left-element (with a new bound variable) and the right-element.
- An `implication` node is received — split its premises and head so the premises can be asserted as hypotheses.

Tagged as `disintegration` in the processed proof graph.

### disintegration set / integration set

The two implication sets a logical entity carries, both compiled when the entity is created and both written to the GL binary ([D-503](40_decisions.md#d-503)). The **disintegration set** (`LogicalEntity::implications`) holds the rules deployed when an instance of the entity is disintegrated: an or's K-rules and subset exclusions ([D-309](40_decisions.md#d-309)), an existence's two negation rules ([D-310](40_decisions.md#d-310)). The **integration set** (`LogicalEntity::integrationImplications`) holds the rules deployed when an instance of an and or existence is integrated: its one-level rule, the occupied form of an existence's one-level rule, and, for the entity as integration root, its flat variants, the introductions of their OR premises and the flat rules of those premises' existence disjuncts. Every entry is a compact `(implication<N>[u_…])` over the entity's own tokens.

### disintegrateImplication

Standalone compiler function at [`compiler.hpp`](../../GL_Quick_VS/GL_Quick/src/compiler.hpp). Walks the tree of an implication and produces a chain + head.

### performDisintegration

*Retired / never-existed in the current source tree.* the project conventions lists it as a key function, but a grep across `GL_Quick_VS/GL_Quick/src/` returns no matches. The actual disintegration surface today is [`ce::disintegrateImplication`](../../GL_Quick_VS/GL_Quick/src/compiler.hpp) (structural walk at [`compiler.hpp`](../../GL_Quick_VS/GL_Quick/src/compiler.hpp)) plus [`disintegrateExpr2`](../../GL_Quick_VS/GL_Quick/src/logical_block/disintegration_processor.cpp) (Pass B at [`logical_block/disintegration_processor.cpp`](../../GL_Quick_VS/GL_Quick/src/logical_block/disintegration_processor.cpp)). the project conventions's reference is stale and should be updated. See [OPEN-6 — RESOLVED](10_pipeline/04_prover.md#open-questions).

### disintegrateExpr2

The **Pass B** disintegration path. Defined at [`logical_block/disintegration_processor.cpp`](../../GL_Quick_VS/GL_Quick/src/logical_block/disintegration_processor.cpp) (`DisintegrationProcessor::disintegrateExpr2`). Produces sets of new statements and integration expressions, with iteration tracking. Two admission paths decide whether a freshly-minted `it_…` variable is committed to `finalStringStatements` — [`isAdmitted`](#isadmitted) (map-based, authoritative) and [`isAllowedAsOperatorInput`](#isallowedasoperatorinput) (single-input-operator fallback). First-match-wins.

### disintegration package / shot

The disintegration package of a compiled existence lists its internal universals fired once on its own products ([I-333](30_invariants.md#i-333)). One firing is a **shot**: the universal (a compact implication of the existence's unfolding), the products its premises took, and the instantiated **shot head**. The compiler computes the package with bound variables as placeholders (`SupportingRuleCompiler::computeDisintegrationPackage`); the disintegrator adds the heads to the constituents of the witness whose binder they attach to and repeats the shot on the actual products when it publishes a head (`shootUniversal`, `ShotFrame`). Example: I1's existence (∃D: divisorSet ∧ hasMinimum) shoots divisorSet's membership universals on p ∈ D: 2 ≤ p, p ≤ n and p | n become constituents of p.

### Pass B

The second disintegration pass. Runs after Pass A has populated basic structures. Uses the admission-map gating to decide which newly-introduced variables are allowed to propagate.

### integration

The mirror of disintegration: reassembling expressions from their constituents so that a conclusion involving a compound shape can be asserted. Preparatory tags stage this: `reformulation for integration and` (for conjunction conclusions), `reformulation for integration >[bound]` (for existence conclusions with a non-empty outer bound list), `reformulation for integration >[]` (for existence conclusions whose bound list was already stripped), and `reformulation for integration flattened` (additional root rules joining a permitted multi-level definition unfolding in one firing).

### expansion for integration

Mirror of `expansion` on the integration side: the reformulated expression is expanded back into compiled-structure form so its elements can be recombined.

### premise element (integration)

During integration, one specific premise of the source implication is called out as a dependency of the integrated step. Tagged `premise element`.

### FullBind / non-FullBind

See [`reconstructImplicationFullBind`](#reconstructimplicationfullbind) and [`reconstructImplication`](#reconstructimplication). The distinction is binding behaviour for numbered variables: FullBind binds all non-`u_`; non-FullBind binds only those occurring multiply.

### savedStartInt

A monotonically-increasing integer counter used by `disintegrateExpr2`'s freshness check. Assumes a single global counter; a parallel counter would violate the freshness contract. See [I-17](30_invariants.md#i-17).

---

## OR branching & contradiction

### OR disintegration

The case-split operation on an `or<N>` head. Each disjunct is asserted inside its own dedicated `_ordis_` branch scope; the other disjuncts' negations are not seeded there. Tagged `or disintegration`.

### OR convergence

When all branches of an OR case split independently reach the same conclusion, the conclusion is promoted to the parent scope. Tagged `or convergence`.

### `cleanUpOrIntegrationBranches`

Defined at [`logical_block/case_processor.cpp`](../../GL_Quick_VS/GL_Quick/src/logical_block/case_processor.cpp), `CaseProcessor::cleanUpOrIntegrationBranches`. Wipes sibling OR-integration branch statements when one branch proves via validity-scope classification. Called after `performElementaryLogicalStep` emits a proof.

### `classifyOrScope`

Defined at [`logical_block/scope_syntax.hpp`](../../GL_Quick_VS/GL_Quick/src/logical_block/scope_syntax.hpp), `ScopeSyntax::classifyOrScope`. Classifies a validity scope as `NotOrScope` / `Integration` / `Disintegration` based on the NameMap stack payload. Enables OR-branch-aware bookkeeping without side tables.

### contradiction

Proof by contradiction: under an assumption of the opposite of the conclusion, derive both an expression and its negation; the assumption is then discharged and the original conclusion emitted. Tagged `contradiction`. Implementation is integrated into `performElementaryLogicalStep` via the `primedForContradiction` flag and `contradictionTheoremId` storage — the contradicted LB reconstructs the negated-head implication. See `logical_block/goal_processor.hpp::dischargeContradiction`.

### contradiction-scope demand

The demand an open negated-compound contradiction scope makes for its own contradiction ([D-560](40_decisions.md#d-560)). For a fact the scope sees, a rule whose head is the fact's negation and whose premises the head match binds completely is a candidate; when its one unknown premise is an and compound whose gate holds at the scope itself, the compound is integrated at the scope, the rule can fire, and the scope closes on the contradiction. Candidates waiting for a premise or a gate element park in `rejectedMapSubproof` (category `contradiction_demand`). Drained once per step by `ContradictionDemand::drainContradictionDemands`; registry `LbMemory::contradictionScopes`, derived fact index `ContradictionFactIndex`, chain floor `HashMemory::contradictionChainMark`.

### vacuous truth

The premise chain leading to an implication is shown to be self-contradictory; the implication head is then trivially valid. At `main`, the induction form cites the opposing pair and recursion hypothesis. Inside an ordinary implication integration scope, the scoped form cites two visible opposing facts and closes the exact implication head under its premises. Tagged `vacuous truth`.

### `reactToHypo`

Defined at [`logical_block/logic_block_operations.cpp`](../../GL_Quick_VS/GL_Quick/src/logical_block/logic_block_operations.cpp), `LogicBlockOperations::reactToHypo`. Handles hypothetical-disintegration variable copy tag path (scopes marked with `_hypo_` payloads). Subsumed by the `variable copy` tag path in the processed proof graph.

---

## Equivalence classes

### EquivalenceClass

Struct for an equality grouping. Fields: `variables` (set), `equalityLevelsMap` (var-set → levels), `equalityOriginMap`. Defined at [`memory.hpp`](../../GL_Quick_VS/GL_Quick/src/memory.hpp). Stored per-validity in `Memory::equivalenceClassesMap`.

### negated equality

An expression of shape `!(=[a,b])`. Receives special treatment in `addStatement`.

### `applyEquivalenceClassToNegatedEquality`

Defined at [`logical_block/equality_propagation.hpp`](../../GL_Quick_VS/GL_Quick/src/logical_block/equality_propagation.hpp), `EqualityPropagation::applyEquivalenceClassToNegatedEquality`. On receipt of a negated equality `!(=[a,b])`, emits sibling inequalities:

- For each `c ∈ class(a) \ {a}` — emit `!(=[c, b])`.
- For each `d ∈ class(b) \ {b}` — emit `!(=[a, d])`.

Does **not** emit the symmetric cross-product (both args substituted simultaneously) — that would explode without clear semantic gain. See [I-12](30_invariants.md#i-12).

---

## Conjecturer

### conjecturer

The C++ combinatorial enumerator that produces candidate theorems for a batch. Entry point: `Conjecturer::run` in [`conjecturer.cpp`](../../GL_Quick_VS/GL_Quick/src/conjecturer.cpp). Invoked via `gl_quick.exe --conjecture <tag>`. Writes `files/theorems/conjectures.txt`.

### reshuffled theorems

After the conjecturer emits candidate theorems, a normalisation pass canonicalises their argument order. The normalised forms are written to `files/theorems/reshuffled_conjectures.txt`. Mirrors go to `files/theorems/reshuffled_mirrored_conjectures.txt`.

### make_all_connection_maps

Module-level helper (originally extracted from `single_thread_calculation`). Computes all valid argument mappings between expression `def_sets` and anchor `def_sets`. Used by the nse≥2 combination path and by `single_expr_anchor_connection`.

### create_map_anchor

Precomputes anchor-to-expression arg permutation tables. For AnchorIncubator's 7 `(1)` args, `left = 7`. The `right` parameter = max over def_sets of `(uncomb + comb)` values. **Critical: `right > 3` causes RAM explosion** (millions of dicts). Controlled by `max_values_for_uncomb_def_sets` + `max_values_for_def_sets` in the config.

### single_expr_anchor_connection

The `nse=1` path — connects a single expression directly to the anchor. Skips pre-combination filters (`check_def_sets`, `max_number_args_expr`) and operator-head checks (`check_input_variables_theorem_operator_head`, `evaluate_operator_exprs2`) that assume nse≥2. Keeps `check_input_variables_order`, `pattern_in_conjecture`, `control_equality`.

### `min_number_simple_expressions` / `max_number_simple_expressions`

Config fields controlling conjecture complexity. When `min = 1`, a preliminary pass connects individual expressions directly to the anchor (producing `(>[...](Anchor[...])(single_head))` conjectures). Peano / Gauss use default `min = 2` — preliminary pass does not run. When `max < 2`, the main combination loop is skipped — only `nse=1` output.

### operator head

A conjecturer head expression whose root is an operator (an expression with non-empty `output_args`, e.g. `in2`, `in3`, `fold`). Operator heads need a companion expression in `nse ≥ 2` to bind the output variable. Relations (`=`, `in`) have no output_args and work naturally.

### CE filter (peek-and-prune)

The counterexample-filtering stage. Loads `files/simple_facts/<anchor>_<size>.txt` (ground-level facts), evaluates each conjecture against them, drops any conjecture that fails on any fact. Surviving conjectures go to the prover. C++ implementation; triggered per-batch.

---

## Compressor

### Compressor

Post-proof redundancy-elimination pass. Source: [`compressor.cpp`](../../GL_Quick_VS/GL_Quick/src/compressor.cpp), [`compressor.hpp`](../../GL_Quick_VS/GL_Quick/src/compressor.hpp). Entry: `Compressor::run`. Runs after the main prover.

### Phase 1 (compressor)

Builds per-theorem proof graphs. For each of `N` proved theorems, creates an independent LB with:

- All theorems loaded as hash-memory implication rules.
- The target theorem's premises as fuel, head as proof goal.
- Runs hash bursts to discover derivation paths.
- Extracts a lightweight graph — a `CompressorNode` with `{originalTheorem, head, premises, graph}` where `graph` maps each expression to its dependency lists (from `exprOriginMap`).

### Phase 2 (compressor)

Greedy elimination:

1. Counts per-theorem usage across all proof graphs.
2. Sorts by usage ascending (least-used first).
3. Multi-pass: for each candidate, tentatively adds it to the dead set, checks if all **surviving** theorems remain derivable via [`isDerivable`](#isderivable). If yes, candidate stays dead. If not, candidate is essential.

The "surviving-only" detail is critical: checking dead nodes creates false dependencies where dead twin theorems protect each other from elimination.

### `isDerivable`

Forward reachability from premises + surviving theorems, on a single `CompressorNode`:

1. Seeds the "alive" set with: premises, non-dead theorems, unconditionally-derivable expressions (empty dep lists).
2. Iterates: marks an expression alive if any of its dep lists has all deps alive.
3. Returns whether the head is alive.

### CompressorNode

One per theorem. Fields: `originalTheorem`, `head`, `premises`, `graph`. The `graph` is an adjacency map derived from `exprOriginMap` during Phase 1.

### `multiplyImplication` (partition-based equalization)

Not part of the compressor — but conceptually adjacent because it also produces multiple variant theorems. Defined at [`inference/implication_multiplier.cpp`](../../GL_Quick_VS/GL_Quick/src/inference/implication_multiplier.cpp), `ImplicationMultiplier::multiplyImplication`. Generates Bell partitions of `(1)`-typed bound variables. Each partition = one equalisation copy where groups of variables are set equal simultaneously. Enables cross-expression equalisation (e.g., vars linked transitively through shared expressions). Callers: `addToHashMemory`.

---

## Incubator

### incubator mode

A separate pipeline (`run_modes.incubator_run`) with its own config (`ConfigIncubatorPeano.json`) and theorem storage (`files/theorems_incubator/`). Output never enters the main proof graph — produces proved ground-level facts for future CE-filter use.

### try_contradiction

C++ prover parameter enabling contradiction-attempt LBs for negative conjectures. Set by the incubator config. The LB assumes the registered head verbatim, so it can only ever *disprove* that head.

### try_contradiction_negated_head

C++ prover parameter — complement of `try_contradiction`. Every registered conjecture additionally gets a contradiction LB that assumes the *negation* of its head and, on a main-scope contradiction, emits the conjecture itself as proved (reductio). The polarity that can prove a negated head. Set by the IncubatorPeano2 config and both main configs.

### skip_ce_filter

C++ prover parameter that bypasses the CE filter for the incubator run (since the incubator's purpose is to *produce* facts, not consume them).

---

## Proof tags

A **definition projection** is a processed proof step extracting a constituent through positive conjunction definitions from one explicit fact in the same namespace. Its certificate retains the exact projection path; bound names may differ, but free arguments remain exact. See [definition projection](20_core_concepts/08_proof_tags.md#definition-projection).

37 unique tags dispatched by `TAG_CHECKERS` in [`verifier.py`](../../verifier.py); registry has one entry per tag (the historical `equalize variable` alias was removed). Full details in [`20_core_concepts/08_proof_tags.md`](20_core_concepts/08_proof_tags.md) and [`10_pipeline/08_verifier.md`](10_pipeline/08_verifier.md).

Sorted alphabetically for quick lookup:

| Tag | Checker (verifier.py) | Meaning |
|---|---|---|
| `anchor handling` | `check_anchor_handling` | Pin a raw bound-variable index to its anchor-slot name. |
| `compilation` | `check_compilation` | Records a compiled name whose definition reconstructs its cited implication. |
| `contradiction` | `check_contradiction` | Proof by contradiction — both `X` and `!X` derived at the contradiction LB's `main` under its proof assumption. |
| `contraposition` | `check_contraposition_auxiliary` | Reverses a cited implication while negating both sides and preserving its binders. |
| `definition projection` | `check_definition_projection` | Selects a constituent through positive conjunction definitions from one same-scope fact. |
| `disintegration` | `check_disintegration` | Compound split into its elements. |
| `equality1` | `check_equality1` | Argument substitution via `(=[a,b])`. |
| `equality2` | `check_equality2` | Transitivity of equality. |
| `expansion` | `check_expansion` | Rewrite a named expression into its compiled-structure form. |
| `expansion for integration` | `check_expansion_for_integration` | Mirror of `expansion` on the integration side. |
| `externally provided theorem` | `check_externally_provided_theorem` | A theorem loaded from `externally_provided_theorems.txt`. |
| `implication` | `check_implication` | A compiled implication rule fired in hash memory. |
| `implication composition` | `check_implication_composition` | Composes cited implications with consistent binders, fixed parameters and source scopes. |
| `implication grouping` | `check_contraposition_auxiliary` | Groups conclusions of cited implications that share the same complete premise. |
| `incubator back reformulation` | `check_incubator_back_reformulation` | Incubator operator-equality rewritten to direct operator form. |
| `multiplied from` | `check_equalize_variable` | Re-emitted with bound variables identified per Bell partition. |
| `or branch assumption` | `check_or_branch_assumption` | Negated other-disjunct seeded as an `_orint_` subproof premise (D-35). |
| `or branch proven` | `check_or_branch_proven` | One `_orint_` subproof proved the parent OR goal (D-35). |
| `or convergence` | `check_or_convergence` | All branches converged on the same conclusion. |
| `or disintegration` | `check_or_disintegration` | Case split on an `or` head. |
| `or elimination` | `check_or_elimination` | Derives a common conclusion from a proved disjunction and its cited case implications. |
| `or theorem` | `check_or_theorem` | An OR-shaped theorem was reached as a goal. |
| `premise element` | `check_premise_element` | A specific premise of the source implication is cited. |
| `recursion` | `check_recursion` | Induction-hypothesis step. |
| `reformulated from` | `check_reformulated_from` | This theorem is a reformulation of the cited source. |
| `reformulation for integration >[]` | `check_reformulation_for_integration_empty` | Reverse-disintegration prep for an `existence` with empty outer bounds. |
| `reformulation for integration >[bound]` | `check_reformulation_for_integration_bound` | Reverse-disintegration prep for an `existence` with non-empty outer bounds. |
| `reformulation for integration and` | `check_reformulation_for_integration_and` | Reverse-disintegration prep for an `and`-category conclusion. |
| `reformulation for integration flattened` | `check_reformulation_for_integration_flattened` | Root integration rule with explicitly checked multi-level definition construction. |
| `symmetry of equality` | `check_symmetry_of_equality` | From `(=[b,a])`, emit `(=[a,b])`. |
| `symmetry of inequality` | `check_symmetry_of_inequality` | From `!(=[b,a])`, emit `!(=[a,b])`. |
| `task formulation` | `check_task_formulation` | A root premise of the theorem under proof. |
| `theorem` | `check_theorem_tag` | A previously-proven theorem fired as an inference rule. |
| `vacuous truth` | `check_vacuous_truth` | The premise chain is self-contradictory. |
| `validity name` | `check_validity_name` | Declares the `validityName` for an integrated expression. |
| `variable copy` | `check_variable_copy` | A fresh `_copy` duplicate of an existing variable. |
| `zoom in` | `check_zoom_in` | Imports an identical fact from the explicitly cited strict ancestor scope. |

The verifier additionally validates origin references, origin-chain termination, theorem-use acyclicity, type consistency and chapter coverage. Trace and failure counters are reported alongside the per-tag results.

---

## Pipeline artefacts

### `files/definitions/*.mpl`

User-authored MPL definition files. Input to the compiler. Examples: `NaturalNumbers.mpl`, `AnchorPeano.mpl`, `fold.mpl`, `in.txt`, `limitSet.mpl`.

### `files/config/Config<Tag>.json`

Per-batch configuration. Fields: expression sub-configurations (`in`, `=`, `in2`, `in3`, `fXY`, `fXYZ`, …) with `arity`, `definition_sets`, `full_mpl`, `short_mpl`, `max_count_per_conjecture`, `input_args`, `output_args`, `max_size_expression_before_existence`, `max_size_expression_after_existence`, `allow_negation`, `allow_to_constitute_existence`, `existence_variable_position`, `allowed_for_existence`. The top-level config also carries batch-wide settings (anchor, min/max simple expressions, max iteration number, etc.).

### `files/config/ConfigVisu.json`

Visualisation/verification config. Covers all expression types used in the processed proof graph. Has `input_args`, `output_args`, `definition_sets` for every expression the verifier might encounter. Loaded by `verifier.py`.

### `files/GL_binaries/GL_binary_<Tag>.json`

Compiled definition structure. Generated each run by the compiler; gitignored. Maps expression names to `{arity, category, elements, signature, definedSet}`. Per-batch — covers anchor entries, atomic seeds from `ConfigVisu.json`, and that batch's spontaneously-allocated compact operators.

### `files/GL_binaries/GL_binary_shared.json`

Cross-batch shared registry of *spontaneous compact operators* (categories `implication`, `existence`, `or`, `and` — see `_SPONTANEOUS_CATEGORIES` in [`run_modes.py`](../run_modes.py)). Anchor entries (`AnchorPeano`, `AnchorGauss`, `AnchorIncubator`, …) and atomic entries are excluded — they are batch-local. Lifecycle:

- **Pre-batch.** [`run_modes.py::_seed_per_batch_binary`](../run_modes.py) copies `GL_binary_shared.json` to `GL_binary_<Tag>.json` immediately before invoking `gl_quick.exe <Tag>`. Python is the sole writer of shared and the sole creator of the per-batch file at this step. On a clean run when shared does not exist yet, an empty `{}` is written.
- **During batch.** C++ may append spontaneous operators (`implication<N>`, `existence<N>`, `or<N>`, `and<N>`) to the per-batch file via `compiledExpressions.insert` + `exportCompiledExpressionsJSON` ([`proof/proof_reconstructor.cpp`](../../GL_Quick_VS/GL_Quick/src/proof/proof_reconstructor.cpp)).
- **Post-batch.** [`run_modes.py::_merge_into_shared`](../run_modes.py) reads the per-batch file, filters to entries whose `category` is in `_SPONTANEOUS_CATEGORIES`, and adds any name not already present to `GL_binary_shared.json`. Existing entries are not overwritten — a name allocated by an earlier batch keeps its original definition.

The shared file grows monotonically over a run and is the substrate for cross-batch externals seeding (see [D-40](40_decisions.md#d-40)) and the popup-builder's `build_gl_binary_map` in [`generate_full_proof_graph.py`](../generate_full_proof_graph.py). See also [I-23](30_invariants.md#i-23) and [D-22](40_decisions.md#d-22).

### `files/theorems/conjectures.txt`

Conjecturer output — unfiltered. Raw bound-variable-number form (no `v1` / `v2` renaming yet). Example line from the current branch:

```text
(>[1,2,3,6](AnchorPeano[1,2,3,4,5,6])(>[7]!(=[2,7])(>[](in[6,1])!(>[8](in[8,1])!(in2[8,7,3])))))
```

### `files/theorems/filtered_conjectures.txt`

Conjectures that survived the CE filter.

### `files/theorems/reshuffled_conjectures.txt` / `reshuffled_mirrored_conjectures.txt`

Post-normalisation canonical forms + their mirrors.

### `files/theorems/theorems.txt`

Survivors of the prover phase, after compressor pruning. Input to stage 5 (raw proof graph emission).

### `files/theorems/externally_provided_theorems.txt`

User-supplied external theorems. Durable input — never emptied by a run.

### `files/theorems/compressed_external_theorems.txt`

Originals + mirrors, rebuilt every run, then compressor-pruned. The form the prover actually loads.

### `files/theorems/or_pairs.txt`

OR-shape theorem pair registry. Small file with a pair per line.

### `files/simple_facts/simple_facts_<tag>_<size>.txt`

Arithmetic tables for CE filtering. Line-oriented ground-level facts. Example rows (from `simple_facts_peano_5.txt`):

```text
(AnchorPeano[N,j0,s,+,*,j1])
!(=[i4,i5])
(in2[i0,i1,s])
(>[v1](in2[i0,v1,s])(=[v1,i1]))
```

### `files/raw_proof_graph/*.txt`

Prover output — per-theorem chapters before variable renaming. The `j-copy` and `repl_lev_*` names appear here unchanged.

### `files/processed_proof_graph/*.txt`

Renamed / pruned chapters + `global_theorem_list.txt` + `external_theorems.txt`. Tab-separated line format:

```text
expression \t namespace \t tag \t [ref_expression \t ref_namespace \t ...]
```

### `files/processed_proof_graph/global_theorem_list.txt`

Authoritative theorem registry for verifier. Three columns: `expression`, `method`, `reference`. Live methods are `single chapter proof`, `induction`, `reformulated statement`, `incubator back reformulation`, `or theorem`, `or elimination`, `proved not broadcast`, and the debug-only `debug`. Reference is `-1` for `single chapter proof`, the induction variable for `induction`, and the source-theorem expression for the derived chapter kinds.

### `files/full_proof_graph/*.html`

Generated HTML proof-graph export. `index.html` (chapter list), `tags.html` (tag legend), `chapter<N>.html` (one per theorem, N its position in the theorem list; an induction theorem's three processed chapters share one page, [D-591](40_decisions.md#d-591)).

### zoom in

Proof-history tag for an unchanged expression inherited from a strict ancestor validity scope. The row cites the exact parent expression/scope pair and preserves its derivation there. See [proof tag reference](20_core_concepts/08_proof_tags.md#zoom-in).

### implication composition

Composition of two actual one-premise rules whose intermediate signed atoms match bijectively over the same complete bound-variable set. Construction requires the consumer head to match an existing integration admission demand through the CNFE marker convention, or an open goal through CNFE's candidate test; later demands and goals wake existing rules. The result retains both proof sources, compatible source scope and supporting levels, and can participate in further demand- or goal-qualified composition. `logical_block/unary_integration_processor.cpp`, `drainUnaryComposition`, maintains the incremental closure; the proof tag is `implication composition`. CNFE transfers the integration demand into ordinary algebra witness permission.

### implication grouping

An auxiliary single-premise implication whose conjunction head consists exactly of its cited same-premise source heads. Complete binders align bijectively; every selected source is retained in provenance. It adds no defining element or integration obligation.

### contraposition

The derived implication obtained by exchanging a unary implication's premise and head and complementing both, cancelling double negation. GL prepares binder-covering source families during compilation and activates them during disintegration with exact source scope and checked proof history.

## Permanent witness-name representation

`memory_infra/witness_names.hpp` provides allocation-free classification of `force_lev_<level>_<serial>`, `int_<depth>_lev_<level>_<serial>` and `it_<generation>_lev_<level>_<serial>`. Its canonical comparator prefers ordinary names, forced names, integration names by numeric depth, then algebraic names, with decoded lexical ties. Its outer-expression depth calculation counts integration names only. Forced routes mint forced names. Ordinary integration minting uses the outer expansion assignment under `maxIntegrationDepth`; parked sources preserve that assignment. Cross-block fact and rule mail excludes every ordinary integration witness, while proof-history transport remains available.

## Open questions: older storage descriptions

The admission and equivalence glossary paragraphs still mix historical heap-form
record names with current private, identifier-based storage. Their source links
now identify current operation owners; the older field-shape and per-class
iteration descriptions need a separate semantic reconciliation against
`memory.hpp`, `memory_infra/hash_memory.hpp` and the current equivalence chapter.
In particular, the old `applyEquivalenceClass` helper is not a current declaration;
`EquivalenceProcessor::applyEquiClasses` coordinates the current batch path.
Use the invariant registry and the cited implementation for storage contracts.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
