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

An expression whose head name is not one of the structural operators (`&`, `>`, `=`). Atomic expressions are ground predicates applied to arguments — for example `(in[i0,N])`, `(in3[a,b,c,+])`, `(fXY[s,N,N])`. Defined by `category: atomic` in `CoreExpressionConfig`; see [`compiler.hpp`](../GL_Quick_VS/GL_Quick/src/compiler.hpp).

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

*Seen:* `CoreExpressionConfig::definition` in [`compiler.hpp`](../GL_Quick_VS/GL_Quick/src/compiler.hpp); implementation at `prover.hpp` (`reconstructImplication`) and `prover.hpp` (`reconstructImplicationFullBind`).

### `v`-variable

A renamed bound variable in the processed proof graph. `process_proof_graphs.py` maps raw bound-variable indices (`1`, `2`, …) to type-cased names via a four-priority scheme seeded from the theorem expression (see [I-10](30_invariants.md#i-10)). Names use **lowercase `v`** (`v1`, `v2`, …) for variables that occupy a `(1)`-typed position (digits / elements of N) and **uppercase `V`** (`V1`, `V2`, …) for variables that occupy a `P(...)`-typed position (sets). The two counters are independent; both start at 1. Anchor-fixed slot names (`N`, `i0`, `s`, etc.) are not renamed. The HTML visualizer further shifts `v→w` / `V→W` when an applied/external theorem is rendered inside a chapter (display-only, to avoid confusion with the chapter's own v/V variables).

Real example — processed chapter line showing `v1`:

```text
(in[v1,N])	main	implication	(>[v2,v3](in2[v2,v3,s])(in[v2,N]))	main	(in2[v1,i0,s])	main
```

### `it_`-prefixed variable

An **iteration variable** minted during Pass B disintegration. Pass B produces fresh variables for each iteration of a disintegration step; these are prefixed `it_` to distinguish them from stable `v-` variables and from integration variables. The Pass B admission paths [`isAdmitted`](#isadmitted) and [`isAllowedAsOperatorInput`](#isallowedasoperatorinput) gate whether an `it_…` variable is committed to `finalStringStatements`.

*Seen:* `disintegrateExpr2` in [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp).

### `int_`-prefixed variable

An **integration variable** used on the integration side (the mirror of disintegration). The `int_` path uses only the map-based [`isAdmitted`](#isadmitted) admission rule — the standalone single-input-operator fallback that applies to `it_…` variables is *not* ported. See [I-6](30_invariants.md#i-6).

### `repl_lev_*_*` variable

Substitution-chain placeholder minted during anchor-handling rewrites. The pattern `repl_lev_<N>_<M>` captures the rewrite level and position. These are internal names that must be renamed consistently when `_copy` suffixes are derived. See renaming Priority 4 in the project conventions.

### `_copy` suffix

A variable ending in `_copy` is a freshly duplicated version of an existing variable, introduced to keep a hypothesis variable distinct from its surrounding scope during case analysis or OR branching. The tag `variable copy` (see [Proof tags](#proof-tags)) records each such duplication.

Raw `X_copy` is renamed in the processed proof graph by attaching the `_copy` suffix to the renamed base variable: if raw `repl_lev_3_2` renames to `v11`, then `repl_lev_3_2_copy` renames to `v11_copy`.

*Seen:* verifier substitution code at `verifier.py`, `verifier.py`.

### digit args

For an implication chain, the set of non-anchor arguments that flow into it from outside. Computed by `findDigitArgs` ([`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp)): collect all `inputIndices`-selected arguments from every expression in the chain, subtract anchor args, subtract output args. Used to decide which bound variables are "inputs" versus "outputs" of the implication — this shapes `multiplyImplication`'s partition choices and the admission map.

### immutable args

The subset of digit args that are provably not modified by the chain — they propagate from premise to head without being in any output position. Computed by `findImmutableArgs` ([`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp)). Seed = digit args minus the induction variable. Extend through outputs where all inputs are already immutable. Used in induction-step reconstruction.

### output variable

The argument position marked as an output in a `CoreExpressionConfig`. For `in2` (3-arg), index 2 (0-indexed: position 2) is the output — in the expression `(in2[a,b,f])`, `b` is the output of applying `f` to `a`. Listed in `output_args` in `ConfigVisu.json` and per-config files.

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

*Seen:* `files/definitions/AnchorPeano.mpl`; referenced by every Peano theorem. Signature computed via `makeAnchorSignature` at [`compiler.hpp`](../GL_Quick_VS/GL_Quick/src/compiler.hpp).

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

A proved `(>[...](premise)(head))` expression. Survives the prover phase and enters `files/theorems/theorems.txt`, then is further pruned by the compressor. Each theorem is emitted into the processed proof graph as one or more chapters (one for direct proof, three for induction, one each for mirrored/reformulated variants).

### conjecture

An expression that *looks like* a theorem (same shape) but has not yet been proved. The conjecturer (stage 2 of the pipeline) enumerates conjectures. The CE filter (stage 4) prunes those that fail against the simple-facts tables. Survivors go to the prover.

### premise / conclusion

A premise is one entry in the chain; the conclusion is the head. In a multi-premise implication `(>[α](P1)(>[β](P2)(head)))`, there are two premises (`P1`, `P2`) and one head.

### direct proof

A theorem proved without recourse to induction. Emitted as a single chapter `<N>_direct_proof.txt`. Method label in `global_theorem_list.txt`: `direct`.

### induction proof

A theorem proved by induction on one bound variable. Emitted as a triad of consecutive chapters:

- `<N>_induction_typing.txt` — proves the induction variable is in `N` (see [I-18](30_invariants.md#i-18))
- `<N+1>_check_zero.txt` — proves the base case (substitute `0` for the induction variable)
- `<N+2>_check_induction_condition.txt` — proves the step case (assume for `k`, conclude for `s(k)`)

Method label: `induction`. Induction variable recorded in column 3 of `global_theorem_list.txt`.

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

A degenerate guard variant often has a mathematically redundant premise (13b's `a|b`), so its closure trips the D-278 level gate; such variants land in the **proved-not-broadcast tier** — method label `proved not broadcast`, a real direct-walk chapter, zero circulation (no reformulation, broadcast, or-seeding, compressor participation, or `theorems.txt` row) — and the merge consumes them as variants like any proved row. See [D-290](40_decisions.md#d-290).

### or branch proven

OR-integration bookkeeping row. Records that one `_orint_` subproof has proved the parent OR goal through its selected disjunct. Row layout: `<or-expr> <parent-ns> or branch proven <selected-disjunct> <subproof-ns>`. The historical tag name says "branch", but this is not an `_ordis_` case-split row. Promoted to a first-class `TAG_CHECKERS` entry by [D-35](40_decisions.md#d-35) (previously claimed retired in the SwDD; the override path never existed and the prover always emitted the tag live). Checker: `check_or_branch_proven`.

### or branch assumption

OR-integration premise row. Records that in an `_orint_` subproof targeting disjunct `D_i`, the negation `!D_j` of every other disjunct (`j ≠ i`) is seeded as a local assumption. Row layout: `<negated-other-disjunct> <subproof-ns> or branch assumption <or-expr>_integration_goal <parent-ns>`. The historical tag name says "branch", but this is not an `_ordis_` case-split row. Promoted to first-class checker by [D-35](40_decisions.md#d-35). Checker: `check_or_branch_assumption`.

### multiplied from

A theorem produced by `multiplyImplication`: a parent theorem with some of its `(1)`-typed bound variables identified (equalised) according to a Bell partition. Each partition of the bound-variable set becomes one "multiplied" copy. The verifier checker is `check_equalize_variable` at [`verifier.py`](../verifier.py) — the function name reflects the underlying algorithm. Historically a redundant `equalize variable` registry alias also pointed at the same function but was never emitted; it was removed from `TAG_CHECKERS` so the registry now has one entry per unique tag.

### externally provided theorem

A theorem loaded from `files/theorems/externally_provided_theorems.txt` (user-supplied). Counts as an axiomatic boundary within a run — the current prover batch treats it as a given implication rule. On the next batch, theorems proved in the previous batch are also loaded as external theorems.

---

## Logic blocks & memory

### Logic Block (LB)

The unit of execution in the GL grid. Each LB has local memory (the `Memory` instance), its own hash index, its own queue of pending work, and its own mail. LBs communicate between cycles via `mailIn` / `mailOut`, not during them. In the current single-process execution LBs are C++ objects dispatched across real worker threads (`proveKernel`'s barriered phase sweeps over `logicalCores`; a heavy LB's hashburst is itself split into parallel `(LB, part)` tasks); the *architecture* is designed for fuller parallel distribution (ASIC roadmap).

### Memory (class)

The per-LB state object. Formerly called `BodyOfProves`. Defined at [`memory.hpp`](../GL_Quick_VS/GL_Quick/src/memory.hpp). Key fields:

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

Metadata struct for an expression at the definition level. Fields: `category` (`atomic` / `and` / `existence` / `implication`), `elements` (constituent sub-expressions), `signature` (canonical form with `u_` args), `arity`, `definedSet`. Defined at [`memory.hpp`](../GL_Quick_VS/GL_Quick/src/memory.hpp). Stored in `compiledExpressions` (a global map from expression name → `LogicalEntity`).

### LocalMemoryValue

Value type in `HashMemory::encodedMap`. Fields: `value` (normalised conclusion template), `levels` (the rule's installation `std::set<int>` — see [`levels`](#levels); for rules arriving through mail it MUST be empty per [I-51](30_invariants.md#i-51)), `originalImplication`, `justification`, `key`, `remainingArgs`, `validityName`. Defined at [`memory.hpp`](../GL_Quick_VS/GL_Quick/src/memory.hpp).

### Mail

The inter-LB message buffer. Fields:

- `statements` — list of `(ExpressionWithValidity, std::set<int>)` pairs to be delivered. The deposited level set MUST be empty per [I-51](30_invariants.md#i-51): mail-side levels propagate via the receiver's `addToHashMemory` → `LocalMemoryValue::levels`, and any non-empty deposit pollutes the union that the `allLevelsInvolved` registration verdict reads ([D-77](40_decisions.md#d-77)).
- ~~`implications`~~ — retired ([D-78](40_decisions.md#d-78)). Was a list of tuples (chain, head, args, levels, theorem); levels field always empty. Implications now travel as the compact `(implication<N>[…])` form on `statements`.
- `exprOriginMap` — provenance entries to merge into the recipient's origin map.

Defined at [`memory.hpp`](../GL_Quick_VS/GL_Quick/src/memory.hpp). (`expandedImplications` and `disintegrationSignals` are two further fields; the cross-LB pull carries only `statements` + `exprOriginMap`.) Each `Memory` has a `mailIn` and `mailOut` — now **per-LB staging buffers** for the pull model ([`MailLog`](#maillog)), not routed outboxes. Statements are main-scope-only by contract (every emission gated on `validityName == "main"`; absorb hard-codes `"main"`).

### MailLog

The pull-model cross-LB mail carrier — a heap `std::` struct on `ExpressionAnalyzer` ([`mail_log.hpp`](../GL_Quick_VS/GL_Quick/src/mail_log.hpp)), replacing the retired push routing (`sendMail` / `smashMail` / per-core `boxes`). Two maps: `batches` (`const Memory* → vector<Mail>`, each LB's append-only log of the mail batches it emitted this execution batch — the single stored copy) and `cursor` (`recipient → ancestor → count`, the per-(recipient, ancestor) ingested-batch counter). Each LB is registered at grid build; the commit barrier (`proveKernel` post-join seam) appends each LB's `mailOut` to its log; the phase-1 pull (`MailLog::pull`) merges un-ingested ancestor batches into the receiver's `mailIn`. See [D-137](40_decisions.md#d-137), [I-94](30_invariants.md#i-94), [`03_mail_system.md`](20_core_concepts/03_mail_system.md).

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

The per-LB index. Defined at [`memory.hpp`](../GL_Quick_VS/GL_Quick/src/memory.hpp). Three main maps:

- `encodedMap` — `IntNormalizedKey → list<LocalMemoryValue>`. The actual implication-rule registry: given a premise signature (encoded), look up the conclusion templates.
- `admissionMap` — bookkeeping for Pass B output-slot admission; populated by [`updateAdmissionMap`](#updateadmissionmap).
- `rejectedMap` — expressions that have been rejected for admission (pruning lookahead).

Conceptually a trie keyed by integer-normalised expression hashes.

### encodedMap

Where implications live. When a `(premise → conclusion)` rule enters the system (via `addToHashMemory`), it is normalised, keyed by the premise's normalised integer hash, and stored. When a statement is deposited, its hash is looked up in `encodedMap`; any matching entry fires.

### EncodedExpression

Pre-parsed form of an expression. Fields: `name`, `negation`, `arguments` (with iteration/level), `maxIterationNumber`, `original`, `validityName`. Formerly the key of the goal registry; since the registry re-keyed to packed `(originalId, validityId)` int32 keys (`Memory::intToBeProved`), `EncodedExpression` is a transient parse record at boundaries. Defined at [`memory.hpp`](../GL_Quick_VS/GL_Quick/src/memory.hpp).

### levels

A `std::set<int>` attached to every deposited statement (run-form on the kernel chain, [I-136](30_invariants.md#i-136)). **Each integer is the depth (`Memory::level`) of an LB whose state contributed to the derivation of the statement.** Level 0 is the anchor LB; level 1 is its direct child; level 2 is a grandchild; and so on. A statement gets level `k` in its set if any premise of its derivation was either installed at, or itself carried level `k`. The levels of a derived statement are computed as the union of every premise's levels (or the union of `(rule's installation levels) ∪ (premise levels)` when a hashmem rule fires).

A statement's stored levels row is NEVER empty ([I-182](30_invariants.md#i-182)): a NON-DERIVED statement — a loaded fact, an anchor, a broadcast theorem compact, an assumed premise element — carries the singleton `{-1}`, the non-derived tier. `-1` is transparent to level accounting (it never enters a derivation union and gates like the former empty run) and singleton-only (a run is exactly `{-1}` or all-values-≥0, never mixed).

Not a monotonic counter. Not an admission ordering. The earlier glossary entry calling this `(newStatementLevel, newEqualityLevel)` was stale — corrected on the branch in the same commit that fixed [D-77](40_decisions.md#d-77).

**Why it matters.** `prover.hpp::dischargeToBeProved` reads the derived statement's `levels` when a MAIN statement matches a `toBeProved` goal and computes the registration verdict `allLevelsInvolved`:

```cpp
bool allLevelsInvolved = (lvN == memoryBlock.level + 1);
if (lvN == memoryBlock.level && !has0)
    allLevelsInvolved = true;
```

The primary branch (`lvN == level+1`) requires the derivation to have touched **exactly** every LB from the anchor (level 0) down to the producer (level = `memoryBlock.level`). The alternate branch (`lvN == level`, no level 0) covers derivations that ignore the anchor scope. Since [D-278](40_decisions.md#d-278) the verdict gates ONLY the global registration: goal CLOSURE (goal-row erase, scope wipes, twin retirement, deactivation) is level-free, and the verdict rides the sealed `UpdateGlobalDirectRec` to the post-join drain, which on false runs the lifecycle and refuses `appendGlobalTheorem`. A levels row with even one extra element (e.g. a level the LB has no ancestor for) fails both branches — the goal still closes, but no theorem reaches `globalTheoremList`. The contradiction-twin route seals its own verdict from the colliding pair's level-row union (excluding the twin's own level, the assumed seed's tier). See [G-46](50_gotchas.md#g-46) and the fix in [D-77](40_decisions.md#d-77).

**Invariant for producer-side deposits.** Any mail deposit that ships a rule across LBs (`Mail::statements`, retired `Mail::implications`, or any future replacement) **must** use `std::set<int>` (empty) as the deposited level set. The receiver computes the rule's effective levels from its own scope on installation; the mail-side levels are NEVER additive into the derived-statement union. The tuple channel (deleted on this branch) historically used empty sets at every insert site (`prover.cpp:2149` etc.); the D-76 compact-form drain learned this the hard way. See [I-51](30_invariants.md#i-51).

### intStatementLevelsMap

`std::unordered_map<int32_t, std::set<int>>` keyed by `packStatementKey(originalId, validityId)` — one entry per `(expression, validityName)` pair installed in the LB. Holds the per-statement `levels` set described above. Read by `prover.hpp::dischargeToBeProved` to compute the `allLevelsInvolved` registration verdict (packed probe when the int ids are in hand, the non-minting `lookupStatementLevels` otherwise); written by every site that constructs or updates a statement's level set (Anchor handling, rule firing, mail absorb, equivalence-class rewrites). Stored on `Memory`; never crosses LBs verbatim (the receiver re-builds its own per-statement entries from premise-side levels). The hashburst dump prints its section by decoding the keys and lex-sorting on `(original, validityName)` — byte-identical to the former `std::map<EncodedExpression, std::set<int>>` iteration order (Rule 14).

### intKnownStatements

The packed-key statement registry of an LB: `packStatementKey(originalId, validityId)` → `StatementFlags`. Carries two membership bits — `registered` (the statement passed an add-path registration door) and `known` (the statement entered the level registry; the Site F dedup record) — plus `local` and `fullyDisintegrated`. Gates test the bit their contract names, never bare presence. See [I-85](30_invariants.md#i-85).

### wholeExpressions

Retired — the former string-keyed statement registry, folded into `intKnownStatements` ([D-128](40_decisions.md#d-128)); the interim `registered` membership bit and the dump's `wholeExpressions` section are retired too ([D-264](40_decisions.md#d-264)): row presence is the one membership.

### admission map

`admissionMap` in `HashMemory`. Populated by [`updateAdmissionMap`](#updateadmissionmap) from consumer-side registrations (output-slot markers). Queried by [`isAdmitted`](#isadmitted) on the producer side to decide whether a freshly-minted `it_…` variable should be committed. **Key shape:** `u_`-prefixed on all non-marker args (template form).

### `admissionMapIntegration`

Integration-side counterpart in `HashMemory`. Populated at `prover.hpp` from `le.signature` during `prepareIntegrationCore2` Case C. Queried by [`isAdmittedIntegration`](#isadmittedintegration). **Key shape:** `u_`-prefixed on all non-marker args — same as `admissionMap`, built from `le.signature` which is already in u_ form. The `u_`-transform in `isAdmittedIntegration` and in `applyEquivalenceClassToRejectedMapIntegration` is what bridges Pass B's bare marker form to this template form for lookup. See [G-32](50_gotchas.md#g-32) for the silent-miss pitfall on key-shape mismatch.

### `admissionSetIntegration`

Secondary admission set in `HashMemory`. Populated at `prover.cpp` via `removeUPrefixFromArguments(mappedElement)`. **Key shape:** bare concrete with `repl_` preserved (no `u_`). Queried directly by Pass B `int_` branch as a fallback when `admissionMapIntegration` misses. Lookup uses Pass B's bare marker form as-is — no transform needed.

### `isAdmittedIntegration`

Defined at [`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp). Pass B's integration-admission query. Takes Pass B's bare marker form, adds `u_` prefix to every non-marker arg, then looks up `admissionMapIntegration`. On match, unfolds the admission template via `prepareIntegrationCore2`. Returns `true` iff the var was admitted.

### rejected map (algebra)

`rejectedMap` in `HashMemory`. Integration-side buffer for `it_…` vars that failed Pass B admission. Write via [`updateRejectedMap`](#updaterejectedmap) at `prover.hpp`. Revisit via [`revisitRejected2`](#revisitrejected2) at `prover.cpp` when a new admission key arrives. **Key shape:** bare concrete (matches Pass B's `makeMarkedExpr` output). Revisit path calls `addExprToMemoryBlock` directly — cyclic re-entry guarded by `revisitInProgress`.

### `rejectedMapIntegration`

Integration-side counterpart to `rejectedMap` in `HashMemory`. Added in [D-19](40_decisions.md#d-19). Write via `updateRejectedMapIntegration` at `prover.hpp` from Pass B's `int_` branch when admission fails. Value is [`RejectedMapIntegrationValue`](#rejectedmapintegrationvalue). **Key shape:** bare concrete — same as `rejectedMap`. Two revival paths: [`applyEquivalenceClassToRejectedMapIntegration`](#applyequivalenceclasstorejectedmapintegration) (eq-class rewrite) and [`revisitRejectedIntegration2`](#revisitrejectedintegration2) (new-admission-key trigger). Both deposit to `sameIterationInternalMail`, not via direct `addExprToMemoryBlock`. See [I-22](30_invariants.md#i-22): the admission-map entry is **not** erased on revival (asymmetric with algebra).

### `RejectedMapIntegrationValue`

Value struct for `rejectedMapIntegration`. Fields:

- `concreteConstituent` — body element with `u_` stripped, `int_` arg intact (marker position holds the rejected `int_` var).
- `siblings` — all other body elements of the same compound (including `(in[…])` typing), concrete form.
- `compoundExpression` — original compound for level derivation and origin provenance.

No `iteration` field — `int_` mint uses `level + startInt` only (`prover.cpp`); iteration is an `it_`-only concept.

### `updateRejectedMap`

Defined at [`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp). Algebra-side writer. Inserts a `RejectedMapValue` into `rejectedMap` keyed by `(markedExpr, validityName)`. Called from Pass B deferred-commit at `prover.cpp`.

### `updateRejectedMapIntegration`

Defined at [`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp). Integration-side writer. Inserts a `RejectedMapIntegrationValue` into `rejectedMapIntegration` keyed by `(markedKey, validityName)` and maintains the `varsInRejectedMapIntegrationKeys` cache. Called from Pass B deferred-commit at `prover.cpp` (section alongside algebra commit).

### `revisitRejected2`

Defined at [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp). Algebra-side revival entry point — called after an admission-map insert to re-process rejected entries at that marker key. Snapshots, reconstructs expanded form, deposits body elements via `addExprToMemoryBlock`. Does not consume the admission key ([I-200](30_invariants.md#i-200)). Cyclic (can re-enter via `updateRejectedMap`); guarded by `revisitInProgress`.

### `revisitRejectedIntegration2`

Defined at [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp) (near `revisitRejected2`). Integration-side counterpart. Called after `admissionMapIntegration` insert at `prover.hpp` (with u_-strip on the key) and after `admissionSetIntegration.insert` at `prover.cpp` (key already bare). Emits matching `rejectedMapIntegration` entries to `sameIterationInternalMail`. Does **not** call `addExprToMemoryBlock`; leaves the admission entry live ([I-22](30_invariants.md#i-22)).

### `applyEquivalenceClassToRejectedMapIntegration`

Defined at [`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp) (alongside `applyEquivalenceClass`). Equivalence-class-driven revival path. For each rmi entry whose key args overlap the class, iterates `allMappingsAna[(|indices|, |eqList|)]` (full permutation, same as main helper), dedups per-entry, probes each unique rewritten key against `admissionMapIntegration` (u_-transformed) and `admissionSetIntegration` (bare). On match: emit constituents to `sameIterationInternalMail`. On no-match: persist rewritten entry at the new key. Called at three sites mirroring `applyEquivalenceClass`: same-NS, ancestor-NS, fixpoint re-iter. Gated on `!parameters.skip_eq_classes`.

### `varsInRejectedMapIntegrationKeys`

`std::unordered_set<std::string>` in `HashMemory`. Monotonically-growing cache of non-marker args present in any `rejectedMapIntegration` key. Used by `applyEquivalenceClassToRejectedMapIntegration` to short-circuit when the class has zero overlap with any stored key — saves an O(|rmi|) walk per class call. Never shrinks (false positives just cost a walk).

### `allMappingsAna`

`std::map<std::pair<int,int>, std::vector<std::vector<int>>>` in `ExpressionAnalyzer`. Precomputed permutation tables keyed by `(numberOfIndicesToSubstitute, sizeOfEqClass)` — for `(k, n)` contains all `k`-length sequences over `{0..n-1}` (with repetition). Used by both `applyEquivalenceClass` and `applyEquivalenceClassToRejectedMapIntegration` to enumerate all ways to substitute class args into the positions they occupy in a target expression. Bounded by `parameters.sizeAllPermutationsAna` (default 7).

### `isAdmitted`

Defined at [`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp). Checks whether a variable is admitted via the admission map, with multi-occurrence and pattern-matching logic. Authoritative admission path.

### `isAllowedAsOperatorInput`

Defined at [`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp). Fallback admission path for `it_…` (Pass B) variables. Fires only when all these hold:

- The expression's head name is in `ExpressionAnalyzer::operators`.
- `cfg.inputIndices.size == 1` — single-input operators only.
- The lone input-arg position holds the variable.
- Standard guards (max iteration number, max secondary variables) pass.

**The single-input gate is the guardrail against RT explosion.** Widening to multi-input operators broke Gauss summation. See [I-6](30_invariants.md#i-6).

### `updateAdmissionMap`

Defined at [`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp). Inserts admission-map entries for operator outputs with renamed keys and remaining-args tracking. Called during Pass B processing on the consumer side.

### `reconstructImplication`

Defined at [`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp). Rebuilds an implication from a chain + head. Binds **only** variables appearing more than once across chain+head in `>[...]`. Skips `u_`-prefixed variables. Used for theorem-level reconstruction (`addTheoremToMemory`, `updateGlobalDirect`, back-reformulation).

### `reconstructImplicationFullBind`

Defined at [`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp). Like `reconstructImplication` but binds **all** non-`u_` variables universally in `>[...]`. Used only at disintegration/integration sites — where expressions come from GL-binary expansion and all numbered variables must be universally quantified. See [I-4](30_invariants.md#i-4).

---

## Scope, validity, & the name map

### validity name (validityName)

A string identifier for a proof scope. Every deposited statement carries a `validityName`. The root scope is the literal string `"main"`. Non-main scopes are minted by pushing payloads onto a stack rooted at `main`.

### NameMap

The per-LB encoder for validity names. Defined in [`memory.hpp`](../GL_Quick_VS/GL_Quick/src/memory.hpp). Maps canonical strings to `int16_t` IDs through the cold name table; ancestor queries (`verdict` / `comparable`) derive from the per-scope `ancestorsOf` lists.

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

Proof tag (added, [D-76](40_decisions.md#d-76)). Records the ASIC-prep step where an implication entering the mail "implications" channel is also compiled to its compact named form and deposited as a `mailOut.statements` expression. Row layout: `<compact (implication<N>[args])> main compilation <original expanded implication> main` — left is the compact form, the single dependency is the original it was compiled from, both at `"main"` (the mail implications channel is main-only, [I-26](30_invariants.md#i-26)). Checker: `check_compilation` — structural, in `_ORIGIN_EXEMPT_TAGS` (the compact↔original link is definitional via the GL binary). Purely additive provenance; does not change which theorems prove.

### disintegration

The operation of splitting a compound expression into its elements. Used when:

- An `and` node is received — split into individual conjuncts.
- An `existence` node is received — split into the left-element (with a new bound variable) and the right-element.
- An `implication` node is received — split its premises and head so the premises can be asserted as hypotheses.

Tagged as `disintegration` in the processed proof graph.

### disintegrateImplication

Standalone compiler function at [`compiler.hpp`](../GL_Quick_VS/GL_Quick/src/compiler.hpp). Walks the tree of an implication and produces a chain + head.

### performDisintegration

*Retired / never-existed in the current source tree.* the project conventions lists it as a key function, but a grep across `GL_Quick_VS/GL_Quick/src/` returns no matches. The actual disintegration surface today is [`ce::disintegrateImplication`](../GL_Quick_VS/GL_Quick/src/compiler.hpp) (structural walk at [`compiler.hpp`](../GL_Quick_VS/GL_Quick/src/compiler.hpp)) plus [`disintegrateExpr2`](../GL_Quick_VS/GL_Quick/src/prover.cpp) (Pass B at [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp)). the project conventions's reference is stale and should be updated. See [OPEN-6 — RESOLVED](10_pipeline/04_prover.md#open-questions).

### disintegrateExpr2

The **Pass B** disintegration path. Defined at [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp). Produces sets of new statements and integration expressions, with iteration tracking. Two admission paths decide whether a freshly-minted `it_…` variable is committed to `finalStringStatements` — [`isAdmitted`](#isadmitted) (map-based, authoritative) and [`isAllowedAsOperatorInput`](#isallowedasoperatorinput) (single-input-operator fallback). First-match-wins.

### Pass B

The second disintegration pass. Runs after Pass A has populated basic structures. Uses the admission-map gating to decide which newly-introduced variables are allowed to propagate.

### integration

The mirror of disintegration: reassembling expressions from their constituents so that a conclusion involving a compound shape can be asserted. Two preparatory tags stage this: `reformulation for integration and` (for `and`-category conclusions), `reformulation for integration >[bound]` (for `existence` conclusions with a non-empty outer bound list), `reformulation for integration >[]` (for `existence` conclusions whose bound list was already stripped).

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

Defined at [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp). Wipes sibling OR-integration branch statements when one branch proves via validity-scope classification. Called after `performElementaryLogicalStep` emits a proof.

### `classifyOrScope`

Defined at [`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp). Classifies a validity scope as `NotOrScope` / `Integration` / `Disintegration` based on the NameMap stack payload. Enables OR-branch-aware bookkeeping without side tables.

### contradiction

Proof by contradiction: under an assumption of the opposite of the conclusion, derive both an expression and its negation; the assumption is then discharged and the original conclusion emitted. Tagged `contradiction`. Implementation is integrated into `performElementaryLogicalStep` via the `primedForContradiction` flag and `contradictionTheoremId` storage — the contradicted LB reconstructs the negated-head implication. See `prover.hpp::dischargeContradiction`.

### vacuous truth

The premise chain leading to an implication is shown to be self-contradictory; the implication head is then trivially valid. Currently confined to scope `main`. Tagged `vacuous truth`.

### `reactToHypo`

Defined at [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp). Handles hypothetical-disintegration variable copy tag path (scopes marked with `_hypo_` payloads). Subsumed by the `variable copy` tag path in the processed proof graph.

---

## Equivalence classes

### EquivalenceClass

Struct for an equality grouping. Fields: `variables` (set), `equalityLevelsMap` (var-set → levels), `equalityOriginMap`. Defined at [`memory.hpp`](../GL_Quick_VS/GL_Quick/src/memory.hpp). Stored per-validity in `Memory::equivalenceClassesMap`.

### negated equality

An expression of shape `!(=[a,b])`. Receives special treatment in `addStatement`.

### `applyEquivalenceClassToNegatedEquality`

Defined at [`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp). On receipt of a negated equality `!(=[a,b])`, emits sibling inequalities:

- For each `c ∈ class(a) \ {a}` — emit `!(=[c, b])`.
- For each `d ∈ class(b) \ {b}` — emit `!(=[a, d])`.

Does **not** emit the symmetric cross-product (both args substituted simultaneously) — that would explode without clear semantic gain. See [I-12](30_invariants.md#i-12).

---

## Conjecturer

### conjecturer

The C++ combinatorial enumerator that produces candidate theorems for a batch. Entry point: `Conjecturer::run` in [`conjecturer.cpp`](../GL_Quick_VS/GL_Quick/src/conjecturer.cpp). Invoked via `gl_quick.exe --conjecture <tag>`. Writes `files/theorems/conjectures.txt`.

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

Post-proof redundancy-elimination pass. Source: [`compressor.cpp`](../GL_Quick_VS/GL_Quick/src/compressor.cpp), [`compressor.hpp`](../GL_Quick_VS/GL_Quick/src/compressor.hpp). Entry: `Compressor::run`. Runs after the main prover.

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

Not part of the compressor — but conceptually adjacent because it also produces multiple variant theorems. Defined at [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp). Generates Bell partitions of `(1)`-typed bound variables. Each partition = one equalisation copy where groups of variables are set equal simultaneously. Enables cross-expression equalisation (e.g., vars linked transitively through shared expressions). Callers: `addToHashMemory`.

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

30 unique tags dispatched by `TAG_CHECKERS` in [`verifier.py`](../verifier.py); registry has one entry per tag (the historical `equalize variable` alias was removed). Full details in [`20_core_concepts/08_proof_tags.md`](20_core_concepts/08_proof_tags.md) and [`10_pipeline/08_verifier.md`](10_pipeline/08_verifier.md).

Sorted alphabetically for quick lookup:

| Tag | Checker (verifier.py) | One-line |
|---|---:|---|
| `anchor handling` | `check_anchor_handling` (2167) | Pin a raw bound-variable index to its anchor-slot name. |
| `contradiction` | `check_contradiction` (2396) | Proof by contradiction — both `X` and `!X` derived at the contradiction LB's `main` under its proof assumption. |
| `disintegration` | `check_disintegration` (1417) | Compound split into its elements. |
| `equality1` | `check_equality1` (1517) | Argument substitution via `(=[a,b])`. |
| `equality2` | `check_equality2` (1646) | Transitivity of equality. |
| `expansion` | `check_expansion` (1419) | Rewrite a named expression into its compiled-structure form. |
| `expansion for integration` | `check_expansion_for_integration` (2143) | Mirror of `expansion` on the integration side. |
| `externally provided theorem` | `check_externally_provided_theorem` (2365) | A theorem loaded from `externally_provided_theorems.txt`. |
| `implication` | `check_implication` (1014) | A compiled implication rule fired in hash memory. |
| `incubator back reformulation` | `check_incubator_back_reformulation` (2350) | Incubator operator-equality rewritten to direct operator form. |
| `multiplied from` | `check_equalize_variable` (2464) | Re-emitted with bound variables identified per Bell partition. |
| `or branch assumption` | `check_or_branch_assumption` (2804) | Negated other-disjunct seeded as an `_orint_` subproof premise (D-35). |
| `or branch proven` | `check_or_branch_proven` (2750) | One `_orint_` subproof proved the parent OR goal (D-35). |
| `or convergence` | `check_or_convergence` (2640) | All branches converged on the same conclusion. |
| `or disintegration` | `check_or_disintegration` (2620) | Case split on an `or` head. |
| `or theorem` | `check_or_theorem` (2302) | An OR-shaped theorem was reached as a goal. |
| `premise element` | `check_premise_element` (2178) | A specific premise of the source implication is cited. |
| `recursion` | `check_recursion` (1732) | Induction-hypothesis step. |
| `reformulated from` | `check_reformulated_from` (2334) | This theorem is a reformulation of the cited source. |
| `reformulation for integration and` | `check_reformulation_for_integration_and` (2080) | Reverse-disintegration prep for an `and`-category conclusion. |
| `reformulation for integration >[]` | `check_reformulation_for_integration_empty` (2117) | Reverse-disintegration prep for an `existence` with empty outer bounds. |
| `reformulation for integration >[bound]` | `check_reformulation_for_integration_bound` (2093) | Reverse-disintegration prep for an `existence` with non-empty outer bounds. |
| `symmetry of equality` | `check_symmetry_of_equality` (1680) | From `(=[b,a])`, emit `(=[a,b])`. |
| `symmetry of inequality` | `check_symmetry_of_inequality` (1701) | From `!(=[b,a])`, emit `!(=[a,b])`. |
| `task formulation` | `check_task_formulation` (1552) | A root premise of the theorem under proof. |
| `theorem` | `check_theorem_tag` (1843) | A previously-proven theorem fired as an inference rule. |
| `vacuous truth` | `check_vacuous_truth` (2887) | The premise chain is self-contradictory. |
| `validity name` | `check_validity_name` (2221) | Declares the `validityName` for an integrated expression. |
| `variable copy` | `check_variable_copy` (2388) | A fresh `_copy` duplicate of an existing variable. |

**Non-checker categories** tracked by the verifier but not validated:

- `anchor handling trace` — chain of `_copy` rewrites from `anchor handling`.
- `origin` — provenance chain for a `contradiction`'s derivation tree.
- `self-reference` — error counter; increments when a chapter cites its own theorem.

(the project conventions had enumerated 27 tags in its list with 28 named "in TAG_CHECKERS"; the actual count is 30 distinct tags, 30 registry entries. The missing entries from the project conventions's narrative list are `symmetry of inequality`, `or branch proven`, and `or branch assumption` — the last two added by [D-35](40_decisions.md#d-35). The historical 31st registry entry (`equalize variable`, alias-only) was removed; the 3 reformulation-for-integration sub-variants are first-class registry entries.)

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
- **During batch.** C++ may append spontaneous operators (`implication<N>`, `existence<N>`, `or<N>`, `and<N>`) to the per-batch file via `compiledExpressions.insert` + `exportCompiledExpressionsJSON` ([`visualizer.cpp`](../GL_Quick_VS/GL_Quick/src/visualizer.cpp)).
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

Authoritative theorem registry for verifier. Three columns: `expression`, `method`, `reference`. Method ∈ `direct | induction | reformulated statement`. Reference is `-1` for `direct`, the induction variable for `induction`, and the source-theorem expression for `reformulated statement`.

### `files/full_proof_graph/*.html`

Generated HTML proof-graph export. `index.html` (chapter list), `tags.html` (tag legend), `chapter<N>.html` (one per processed-proof-graph chapter).

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
