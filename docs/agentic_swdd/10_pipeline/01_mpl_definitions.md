<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Pipeline setup — MPL definitions & GL binary `[DRAFT]`

> Definition compilation is folded into every `gl_quick.exe` invocation's startup — not a separately orchestrated pipeline stage. (Previously labelled "Stage 1" before the eight-stage refactor.)

> **Input:** `files/definitions/*.mpl` (user-authored MPL) + `files/config/Config<Tag>.json`.
> **Output:** compiled `CoreExpressionConfig` in-memory; `files/GL_binaries/GL_binary_<Tag>.json` on disk.
> **Owner:** `compiler.cpp` / `compiler.hpp` for configuration input; focused classes under `compiler/` for definition compilation.
> **Pipeline position:** runs once at the start of every `gl_quick.exe` invocation; conjecturer and prover both rely on the compiled output.

---

The component boundary follows [D-474](../40_decisions.md#d-474).

## Expression syntax component

`compiler/expression_parser.hpp` / `ExpressionParser` owns MPL tree parsing,
conjunction-premise normalization, ordered conclusion splitting, binder reading,
and compiler-witness renaming. Its recursive parse and serialization helpers are
private. The compiler and prover call the syntax operations without giving the
parser proof state. `compiler.cpp` / `ce::parseExpr` retains its existing public
entry point and delegates to `ExpressionParser::parse`.

The extraction preserves tree ownership, traversal order, expression spelling,
and the existing caller-visible parse contracts. Compilation registry ownership
and definition expansion remain separate responsibilities.

## Compilation normalization component

`compiler/compilation_normalizer.hpp` / `CompilationNormalizer` owns the
compiler's string-key normalization, unchangeable-argument mapping, retained
argument-list construction, and ordered reuse permutations. It has no prover
state. The Cartesian-product helper is private. Existing key modes, sign bits,
first-occurrence naming and the seven-element permutation limit are unchanged.
These compiler operations remain separate from the heap-free integer request
key builder used by inference.

## Compilation reuse ownership

`compiler/compilation_reuse.hpp` / `CompilationReuse` owns the normalized reuse
maps, public registration order, OR argument mappings, replacement metadata and
public-name freeze state. Its public operations register identities, reuse a
definition, resolve an OR, restore replacements and finalize retirement. Readers
receive const views. Key splitting, identity enumeration and retirement helpers
are private. The component borrows only the definition/configuration stores,
prepared permutations and pending definition-family sets it updates.

The definition stores belong to `CompiledDefinitionRegistry`; its constructor
binding gives `CompilationReuse` only the stores and pending sets it already
uses. Registration, replacement composition and iteration order remain unchanged.

## Definition expansion and binding operations

`compiler/definition_expansion.hpp` / `DefinitionExpansion` borrows read-only
configuration and compiled-definition tables for source and base-form expansion.
Its argument-prefix, negation and expression-side operations are independent of
prover state. Allocation-free counterparts keep their original inline boundaries
and write only into caller-owned scratch arenas. Compiler operations that already
return owning strings retain their existing allocation contract.

`proof/worker_context.hpp` / `WorkerContext` owns the existing thread-local worker
identity behind read and write operations. Workers assign it at the same entry
points; -1 retains the single-threaded seam meaning. Each scratch caller still
selects the reserved slot from the registry it actually indexes.

## Definition ownership and compilation operations

`compiler/compiled_definition_registry.hpp` / `CompiledDefinitionRegistry` owns
source configurations, compiled entities, operator membership, configured-name
membership, typed-equality entities, pending rule-family sets and name counters.
Runtime callers receive const views or use the existing allocation-free point
readers. Only compiler owners can mutate these stores; test-only fixtures can
assemble synthetic registries without adding mutable production accessors.

`compiler/structural_compiler.hpp` / `StructuralCompiler` compiles source structure,
validates registered definitions, derives type metadata and prepares typed equality.
Its `compileExpression` operation owns counter use: callers request a compilation
without passing counter references. Recursive calls preserve their original order.
The configured-name snapshot still occurs before generated configurations are added.

`compiler/or_rule_compiler.hpp` / `OrRuleCompiler` prepares OR closure, symmetry,
mutual exclusions and subset exclusions. `compiler/existence_rule_compiler.hpp` /
`ExistenceRuleCompiler` prepares existence implications. Both request compact
implication compilation from `StructuralCompiler`. `compiler/supporting_rule_compiler.hpp`
/ `SupportingRuleCompiler` prepares contraposition, conjunction-refutation and
supporting families. The mutual component references are bound without invoking
operations during construction; compilation begins after every component exists.

`compiler/compilation_persistence.hpp` / `CompilationPersistence` loads and saves
the existing registry and compilation-state formats. It borrows the batch identity
and preserves name-order serialization and counter restoration. The visualizer
requests persistence instead of implementing it.

`compiler/implication_builder.hpp` / `ImplicationBuilder` owns the shared first-premise
binder-placement rule in owning-string and caller-arena forms. Its inline boundaries,
fixed stack bounds and result bytes remain unchanged.

## Argument-type analysis

`compiler/argument_analyzer.hpp` / `ArgumentAnalyzer` exposes expression analysis,
configuration consistency checking and set-definition recognition. Its structural
comparison and membership-recognition helpers are private. It delegates tree
construction to `compiler/argument_type_parser.hpp` / `ArgumentTypeParser`, which
reads the configured argument types directly and owns recursive parsing, type-map
merging and binder removal. It has no analyzer or prover back-reference.
`compiler/analysis_node.hpp` / `AnalysisNode` retains the existing child ownership
and free-argument metadata. Parsing order, optional-input behavior and all existing
assertions are unchanged.

Every logical entity carries two implication sets, each compiled when the entity
is created: the disintegration set `LogicalEntity::implications` (an or's K-rules
and subset exclusions, an existence's two negation rules) and the integration set
`LogicalEntity::integrationImplications` (every rule `prepareIntegration` installs
for an and or existence instance). `compiler/integration_rule_compiler.cpp` /
`IntegrationRuleCompiler::compileIntegrationImplications` runs at every and and
existence mint of `StructuralCompiler::compileExpression` and for the
extensionality entity in `ensureTypedEqualityEntity`. No pass completes a list
later: an entity loaded from a GL binary carries both sets from the binary
([D-503](../40_decisions.md#d-503)).

## What happens here

Every GL batch starts from a set of user-authored MPL definition files. The compiler:

1. Reads every expression subkey listed in the active `Config<Tag>.json`.
2. If the subkey's `full_mpl` field points to a filename, loads that file from `files/definitions/` and parses its MPL body.
3. If `full_mpl` is itself an inline MPL string, uses that body directly.
4. Builds a `LogicalEntity` metadata record and a `CoreExpressionConfig` for each definition.
5. Serialises the whole map as `files/GL_binaries/GL_binary_<Tag>.json`.

From this point on, the rest of the pipeline refers to definitions by their compiled identity (name + category + signature), not by their source MPL string.

---

## MPL — the string grammar

MPL (Mathematical Programming Language) is the one language the entire GL pipeline speaks. Every definition file, every theorem file, every processed proof-graph line, and every assertion inside the prover is an MPL string. The grammar is compact:

```
expression    := "(" head "[" args "]" ")"                          atomic predicate
              |  "(" "&" expression expression ")"                   binary conjunction
              |  "(" ">" "[" bound-vars "]" expression expression ")"  implication
              |  "(" "=" "[" arg1 "," arg2 "]" ")"                   equality
              |  "!" expression                                       negation

head          := identifier                                           (operator name)
args          := arg ("," arg)*
arg           := identifier | literal-integer
bound-vars    := empty | identifier ("," identifier)*
```

Notes:

- **No spaces.** Ever. Arguments are separated by bare commas. `(in2[a, b, c])` is **invalid** — it must be `(in2[a,b,c])`. Spaces in definition source files (as seen in [NaturalNumbers.mpl](../../files/definitions/NaturalNumbers.mpl)) are purely formatting; the compiler strips them before parsing.
- **Bound variables appear only inside `>[...]`**. The `[...]` after `>` is the quantifier list for the subsequent implication.
- **Negation is a prefix on the whole expression**. `!(=[a,b])` is "a ≠ b", `!(&(p)(q))` is "not (p and q)", `!(>[x](p)(q))` is "there exists x such that p but not q".

Meaningful patterns that recur:

| Pattern | Reads as |
|---|---|
| `(>[x](in[x,N])(P(x)))` | for all x in N, P(x) |
| `!(>[x](in[x,N])!(P(x)))` | there exists x in N with P(x) |
| `(=[a,b])` | a = b |
| `!(=[a,b])` | a ≠ b |
| `(&(P1)(&(P2)(P3)))` | P1 and P2 and P3, with binary nesting |
| `(in[x,X])` | x ∈ X |
| `(in2[x,y,f])` | f(x) = y (single-input operator applied to x gives y) |
| `(in3[x,y,z,f])` | f(x, y) = z (two-input operator) |

The "iff" pattern — since MPL has no native iff — is expressed as a pair of implications wrapped in `(&...)`:

```text
(&
    (>[x](P(x))(Q(x)))
    (>[x](Q(x))(P(x)))
)
```

You see this shape in every set-definition file (`fXY.mpl`, `interval.mpl`, `limitSet.mpl`, …).

---

## Definition files — structure & examples

Each file in `files/definitions/` is one MPL expression — the body of a definition. The definition's *name* (the key under which the body lives) is determined by the `Config<Tag>.json` entry that references the file, **not** by the filename. Multiple configs can reference the same file under different names.

### Anchors — the axiomatic context

An anchor file defines the axiomatic context for a batch. The body is an `and`-tree enumerating the anchoring facts.

#### `AnchorPeano.mpl` — 6 slots

```text
(&
	(NaturalNumbers[N,i0,s,+,*])
	(in2[i0,i1,s])
)
```

Reads as: "the slots `(N, i0, s, +, *)` form a Peano structure, AND `i1 = s(i0)`". In other words, `i1` is defined as `s(0) = 1`. The 6 slots of the anchor are therefore `N`, `i0 = 0`, `s = successor`, `+`, `*`, `i1 = 1`.

#### `AnchorGauss.mpl` — 8 slots

```text
(&
	(NaturalNumbers[N,i0,s,+,*])
	(&
		(in2[i0,i1,s])
		(&
			(in2[i1,i2,s])
			(identity[N,id])
		)
	)
)
```

Extends Peano: `i2 = s(i1) = 2`, and `id` is the identity function on `N`. The 8 slots: `N, i0, s, +, *, i1, i2, id`.

#### `AnchorIncubator.mpl` — 14 slots

14 args: `(AnchorIncubator[N,i0,s,+,*,i1,i2,id,i3,i4,i5,i6,i7,i8])`. The extra slots `i3..i8` pin the constants `3..8`, so the incubator can enumerate ground-level facts over the finite model `{0, 1, 2, 3, 4, 5, 6, 7, 8}`.

### `NaturalNumbers.mpl` — the Peano axioms

The body of this definition spells out every Peano axiom in a single conjunction — rendered here as displayed in source:

```text
(&
    (&
		(in[i0,N])
		(&

			(fXY[s,N,N])
			(&
				(>[n]
					(in[n,N])
					!(in2[n,i0,s])
				)
				(>[m]
					(in[m,N])
					(>[n1,n2]
						(&
							(in2[n1,m,s])
							(in2[n2,m,s])
						)
						(=[n1,n2])
					)
				)
			)
		)
    )
    (&
		(&
			(fXYZ[+,N,N,N])
			(&
				(&
					(>[a]
						(in[a,N])
						(>[b]
							(in3[a,i0,b,+])
							(=[a,b])
						)
					)
					(>[a,b]
						(&(=[a,b])(&(in[a,N])(in[b,N])))
						(in3[a,i0,b,+])
					)
				)
				(>[b]
					(in[b,N])
					(>[a,c,d]
						(&
							(in2[b,c,s])
							(in3[a,b,d,+])
						)
						(&
							(>[e]
								(in3[a,c,e,+])
								(in2[d,e,s])
							)
							(>[e]
								(in2[d,e,s])
								(in3[a,c,e,+])
							)
						)
					)
				)
			)
		)
		(&
			(fXYZ[*,N,N,N])
			(...)
		)
	)
)
```

Each conjunct is a Peano axiom:

- `(in[i0,N])` — 0 is in N.
- `(fXY[s,N,N])` — successor is a function from N to N.
- `(>[n](in[n,N])!(in2[n,i0,s]))` — 0 is not a successor.
- `(>[m](in[m,N])(>[n1,n2](&(in2[n1,m,s])(in2[n2,m,s]))(=[n1,n2])))` — successor is injective.
- `(fXYZ[+,N,N,N])` — `+` is a function N × N → N.
- `(>[a](in[a,N])(>[b](in3[a,i0,b,+])(=[a,b])))` and its converse `(>[a,b](&(=[a,b])(&(in[a,N])(in[b,N])))(in3[a,i0,b,+]))` — `+` has identity `i0 = 0`. The converse builds the fact `a + 0 = a` for every natural `a`.
- `(>[b](in[b,N])(>[a,c,d](...)(...)))` — the successor rule for `+`: if `s(b) = c` and `a + b = d`, then for every `e = s(d)`, `a + c = e` and vice versa.
- `(fXYZ[*,N,N,N])` — `*` is a function N × N → N.
- `(>[a](in[a,N])(>[b](in3[a,i0,b,*])(=[b,i0])))` and its converse `(>[a,b](&(=[b,i0])(&(in[a,N])(in[b,N])))(in3[a,i0,b,*]))` — `0` absorbs under `*`. The converse builds the product fact `a * 0 = 0` for every natural `a`, as the `+` converse builds `a + 0 = a`; without it a product `a * 0` exists only when an admission builds it ([D-505](../40_decisions.md#d-505)).
- The successor rule for `*`: if `s(b) = c` and `a * b = d`, then for every `e`, `d + a = e` exactly when `a * c = e`.

The structure is dense but mechanical. The compiler parses it into a tree of `LogicalEntity` records, flattening the `and` tree into `elements`.

### `fXY.mpl` — functional mapping

```text
(&
	(>[x,y]
		(in2[x,y,f])
		(&
			(in[x,X])
			(in[y,Y])
		)
	)
	(&
		(>[x]
			(in[x,X])
			!(>[y]
				(in[y,Y])
				!(in2[x,y,f])
			)
		)
		(>[x]
			(in[x,X])
			(>[y1,y2]
				(&
					(in2[x,y1,f])
					(in2[x,y2,f])
				)
				(=[y1,y2])
			)
		)	
	)
)
```

Reads as:

1. If `(x, y)` is in `f`, then `x ∈ X` and `y ∈ Y`. (Typing.)
2. For every `x ∈ X`, there exists `y ∈ Y` with `(x, y) ∈ f`. (Totality — expressed via doubly-negated existence: `!(>[y]...!(in2[x,y,f]))`.)
3. For every `x ∈ X`, if `(x, y1) ∈ f` and `(x, y2) ∈ f`, then `y1 = y2`. (Functionality.)

Together: `fXY[f, X, Y]` asserts that `f` is a total function from `X` to `Y`. The compiler gives the whole definition category `and`; its existence conditions remain separate constituent entities (see OPEN-7 below).

### `interval.mpl` — iff-defined set

```text
(&
	(&
		(>[p]
			(in[p,M])
			(&
				(preorder[N,+,n,p])
				(preorder[N,+,p,m])
			)
		)
		(>[p]
			(&
				(preorder[N,+,n,p])
				(preorder[N,+,p,m])
			)
			(in[p,M])
		)
	)
	(&
		(in[n,N])
		(in[m,N])
	)
)
```

Reads as: `interval[N, +, n, m, M]` iff `M = { p: n ≤ p ≤ m }`, with `≤` defined via `preorder`. Plus the typing `n, m ∈ N`. Classic iff form — an `and` of the two implication directions.

### `limitSet.mpl` — nested iff

Same iff-of-iff shape, one layer deeper. Used in the Gauss summation machinery.

### `fold.mpl` — the complex one

```text
!(>[M]
	(interval[N,+,n,m,M])
	(>[g]
		(fXY[g,M,N])
		!(&
			(&
				(>[q](in2[n,q,f])(in2[n,q,g]))
				(in2[m,p,g])
			)
			(>[i]
				(in[i,M])
				(>[j]
					(&(in2[i,j,s])(in[j,M]))
					(>[r](in2[i,r,g])(>[t](in2[j,t,f])(>[q](in3[r,t,q,op])(in2[j,q,g]))))
				)
			)
		)
	)
)
```

Reads as an existence: there exist `M` and a function `g` such that (a) `M = [n, m]` as an interval, (b) `g(n) = f(n)`, (c) `g(m) = p`, and (d) for every `i ∈ M` with successor `j ∈ M`, `op(g(i), f(j)) = g(j)`. In other words, `fold` is "accumulate `f` with `op` along the additive interval [n, m], starting from `f(n)`, and read off `p` at the end". The outermost `!` + inner `!(&...)` encodes "there exists [...]".

The public signature is `fold[N,s,+,op,f,n,m,p]`. Addition defines the interval independently of accumulation: `op=+` is a sum and `op=*` is a product. The seed is the first term, with no empty-interval identity convention. See [I-257](../30_invariants.md#i-257).

---

### Sequence membership definition

FTA registers `inSequence[N,p,P]` from [`inSequence.mpl`](../../files/definitions/inSequence.mpl). It is the typed existence of an index `i` in `N` with `in2[i,p,P]`. Only `i` is existential; `N`, `p` and `P` stay fixed parameters. The definition is category `existence` and exposes no operator output. Public-name promotion uses the ordinary normalized-definition reuse mechanism, so matching anonymous membership existences in FTA become `inSequence` with their argument correspondence preserved; generated compact numbering may change. The sequence property remains a separate hypothesis.

## Compiled form — `CoreExpressionConfig` & `LogicalEntity`

After the compiler reads a definition, it produces two in-memory records per expression name:

### `CoreExpressionConfig` — the configuration surface

Defined at [`compiler.hpp`](../../../GL_Quick_VS/GL_Quick/src/compiler.hpp). Fields:

| Field | Meaning |
|---|---|
| `arity` | Number of arguments the expression takes. |
| `definition` | Parsed MPL body — the canonical expression tree with `u_`-prefixed formal parameters. |
| `signature` | Canonical form `"(name[u_0,u_1,...])"`. |
| `definitionSets` | Per-argument type (`(1)`, `P(1)`, `P(x(1)(1))`, …) + whether the arg is "combinable" (used by the conjecturer). |
| `inputArgs`, `outputArgs` | String-form argument labels from the config. |
| `inputIndices`, `outputIndices` | Parsed integer indices. |

### `LogicalEntity` — the metadata per definition

Defined at [`memory.hpp`](../../../GL_Quick_VS/GL_Quick/src/memory.hpp). Fields:

| Field | Meaning |
|---|---|
| `category` | `atomic` / `and` / `or` / `existence` / `implication`. |
| `elements` | Constituent sub-expressions (with `u_`-prefixed args and `1`, `2`, … for bound vars). |
| `signature` | Same as `CoreExpressionConfig::signature`. |
| `arity` | Same. |
| `definedSet` | For `existence` nodes: which `u_` arg holds the bound variable. |
| `implications` | Compiled exclusion rules for ORs and applicable negated existences; selected rules supply ordinary proof history. |
| `symmetricDigitSlots` | Derived one-based interchangeable positions for a numbered OR with exactly two digit-valued parameters; zeroes mean no validated symmetry. Recomputed after loading, not serialized. |

Both live in global maps keyed by the expression name.

### Categories drive proof behaviour

The `category` field is the switch that determines how an expression is handled downstream:

- `atomic` — ground predicates. Stored as-is; hash-engine keys.
- `or` — ordered signed disjuncts supply exclusion on demand and admitted case branches. Symmetry shares a processing identity while original statements and proof spellings remain distinct.
- `and` — expanded/disintegrated into its elements. `(&(P1)(P2))` admitted ⇒ P1 admitted + P2 admitted.
- `existence` — expanded into left (with new bound variable) + right elements. Requires a fresh variable binding on disintegration.
- `implication` — terminal. Stored as hash-table rules: premise-signature → conclusion-template. A public definition whose whole body is one implication (`primeseq`) is registered under its own name with this category: `StructuralCompiler::compileCoreExpressionMap` forces the definition's signature through `excludeRepetitions`, which returns that full signature, exactly like the single-existence and single-or definitions.

The distinction matters for **every** subsequent stage. A type mismatch in the GL binary (wrong `category` for an expression) silently corrupts the proof graph.

---

## `Config<Tag>.json` — batch configuration

Each batch has its own config file: `ConfigPeano.json`, `ConfigGauss.json`, `ConfigIncubatorPeano.json`, `ConfigIncubatorGauss.json`. The top-level object is a map from expression name to configuration record. Representative entries from `ConfigPeano.json`:

```json
"in": {
    "arity": 2,
    "definition_sets": { "1": ["(1)", true], "2": ["P(1)", false] },
    "full_mpl": "(in[1,2])",
    "short_mpl": "(in[1,2])",
    "max_count_per_conjecture": 2,
    "input_args": ["1"],
    "output_args": [],
    "max_size_expression_before_existence": 5,
    "max_size_expression_after_existence": 5,
    "allow_to_constitute_existence": true,
    "existence_variable_position": 1
},
"in3": {
    "arity": 4,
    "definition_sets": {
        "1": ["(1)", true], "2": ["(1)", true],
        "3": ["(1)", true], "4": ["P(x(1)(x(1)(1)))", false]
    },
    "full_mpl": "(in3[1,2,3,4])",
    "short_mpl": "(in3[1,2,3,4])",
    "max_count_per_conjecture": 5,
    "input_args": ["1", "2"],
    "output_args": ["3"],
    "max_size_expression_before_existence": 5,
    "max_size_expression_after_existence": 5
},
"fXYZ": {
    "arity": 4,
    "definition_sets": {
        "1": ["P(x(1)(x(1)(1)))", false], "2": ["P(1)", false],
        "3": ["P(1)", false], "4": ["P(1)", false]
    },
    "full_mpl": "fXYZ.mpl",
    "short_mpl": "(fXYZ[f,X,Y,Z])",
    "max_count_per_conjecture": 0,
    "input_args": [],
    "output_args": [],
    "max_size_expression_before_existence": 5,
    "max_size_expression_after_existence": 5
}
```

Fields explained:

| Field | Purpose |
|---|---|
| `arity` | Matches the number of `[...]` positions. |
| `definition_sets` | Per-argument: `[<type>, <combinable>]`. Type is a set-theoretic label: `(1)` means elements of `N` (the anchor's first slot); `P(1)` means subsets of `N`; `P(x(1)(1))` means relations on `N × N`; and so on. The boolean flags whether the argument can participate in conjecturer combination. |
| `full_mpl` | Either an inline MPL string (for short definitions like `(in[1,2])`) or a filename under `files/definitions/` (for larger bodies). |
| `short_mpl` | The canonical signature in `u_`-form style — used when the compiler prints the expression in diagnostic output. |
| `max_count_per_conjecture` | Conjecturer limit: how many copies of this predicate can appear in a single conjecture. Relations like `=` usually cap at 1; operators like `in3` can have more. |
| `input_args`, `output_args` | Argument-position labels indicating which args are "inputs" (flow from premise to head) vs "outputs" (defined by the expression). Used by `findDigitArgs` / `findImmutableArgs` + admission-map logic. |
| `max_size_expression_before/after_existence` | Conjecturer size limits. |
| `allow_to_constitute_existence` | Can this predicate head an existence expression? |
| `existence_variable_position` | For existence-constituting predicates: which argument position carries the bound variable. |
| `allow_negation` | Conjecturer — can this expression appear negated? |
| `allowed_for_existence` | For existence nodes: which argument positions can carry the bound variable. |

Batch-level fields (outside per-expression records) control things like the anchor name, `min_number_simple_expressions`, `max_number_simple_expressions`, max iteration counts, `max_values_for_def_sets`, and `max_values_for_uncomb_def_sets` — all in the JSON root.

---

## `files/GL_binaries/GL_binary_<Tag>.json` — the compiled artefact

The compiler's output, and — since [D-22](../40_decisions.md#d-22) — also one of its **inputs**. Maps expression name to `{arity, category, elements, signature, definedSet}` — plus, on **or and existence entries**, `implications`: for an or, its K mutual-exclusion and subset-exclusion rules as `(implication<N>[u_…])` compacts over the or's own `u_` tokens (K-rules in leaf order, then the subset-exclusion rules in the disintegrator's enumeration order; the argument order is the projection onto the or's tokens — [D-309](../40_decisions.md#d-309)); for a two-element existence, its two existence implications `left → !right` then `right → !left` as compacts over the existence's own tokens ([D-310](../40_decisions.md#d-310)). The writer always writes the key for an or or existence (an empty list included) and never for another category. **And and existence entries** also carry `integration_implications`: the entity's integration set as objects `{"compact": "(implication<N>[u_…])", "kind": "one_level" | "occupied" | "flat" | "or_introduction" | "disjunct_flat", "cites": <index or -1>}` in install order ([D-503](../40_decisions.md#d-503)). The loader reads both keys verbatim and asserts each present exactly for its categories; a binary from an older executable is stale and asserts — a full run regenerates every binary. Generated at every run; **gitignored**.

**Cross-batch shared registry.** Spontaneous compact operator names (`implication<N>`, `existence<N>`, `or<N>`, `and<N>`) live in a sibling file `files/GL_binaries/GL_binary_shared.json`. This shared file is the cross-batch source of truth for spontaneous identifiers. Per-batch lifecycle:

1. **Pre-batch (Python).** [`run_modes.py::_seed_per_batch_binary`](../../run_modes.py) copies `GL_binary_shared.json` into `GL_binary_<Tag>.json` immediately before invoking `gl_quick.exe <Tag>`. On a clean run when shared does not exist yet, an empty `{}` is written. Python is the sole writer of shared and the sole creator of the per-batch file at this step.
2. **C++ startup.** The `Prover` constructor calls [`loadGlBinary`](../../../GL_Quick_VS/GL_Quick/src/compiler/compilation_persistence.cpp) (`CompilationPersistence::loadGlBinary`) on the per-batch file its `glBinaryFile` parameter names — the batch driver in [`run_modes.cpp`](../../../GL_Quick_VS/GL_Quick/src/run_modes.cpp) passes `files/GL_binaries/GL_binary_<Tag>.json`; a constructor without a file (every unit test) reads nothing and compiles the registry from the config alone. Every entry is inserted into `compiledExpressions`. The loader first installs every definition, then restores public predicates through `registerLoadedNamedCompilation` and finishes argument-correct generated-name retirement through `finalizeCompilationReplacements`. Public predicates therefore participate in normalized reuse. Numbered spontaneous entries get one row keyed like a fresh registration: the canonical form of their stored elements under their category (`CompilationReuse::registerStoredCompilation` → `canonicalReuseKey`, [I-323](../30_invariants.md#i-323)), so a later body is found whatever the order its elements were stored in. The category is part of the key because two structurally-identical bodies under different categories must allocate distinct names — see [D-60](../40_decisions.md#d-60) for the dormant cross-batch collision this guards against. The four shared counter members (`implCounter`, `existenceCounter`, `andCounter`, `orCounter`) are seeded from `max(N) + 1` so newly-allocated names start above the highest already in use.
3. **C++ shutdown.** `exportCompiledExpressionsJSON` writes the entire `compiledExpressions` map to the same per-batch path. Because shared entries were preloaded, the resulting file contains the inherited shared entries plus this batch's new spontaneous allocations. The path is computed from the running executable through `project_paths.cpp`, `projectRoot` and is identical for every batch (incubator and main alike) — see [D-54](../40_decisions.md#d-54) for the unification of the writer path that previously diverged into `files/incubator/GL_binaries/` for incubator batches.
4. **Post-batch (Python).** [`run_modes.py::_merge_into_shared`](../../run_modes.py) reads the per-batch file and adds any name not already in shared whose `category` is in `{implication, existence, or, and}` (atomic entries are excluded). Applies to EVERY tag — incubator and main alike. The shared file grows monotonically over the run.

**Canonical reuse key.** Every definition, public or numbered, is registered under exactly one key: its body with the premises in canonical order (operator names in string order; inside each same-name group the order whose normalized token sequence is lexicographically smallest, bound variables before `u_` ones, numbers compared numerically; an implication's head last and part of the comparison), normalized in mode 2 (`CompilationNormalizer::canonicalCombination`, `CompilationReuse::canonicalReuseKey`, [D-523](../40_decisions.md#d-523)). A lookup computes the canonical key of the body at hand, so bodies that differ only in premise order and variable names share one name. The stored element order is the one compiled, not the canonical one.

**Named predicate reuse.** Loaded public definitions retain their normalized reuse identities. The first public registration wins an equivalent body; numbered equivalents are retired with argument-correct reference migration before proof consumers start. The counter high-water marks survive retirement. See `compiler/compilation_reuse.cpp` / `CompilationReuse::registerLoadedNamedCompilation` and `CompilationReuse::finalizeCompilationReplacements`.

**`implication<N>` signature shape.** A spontaneous entry's `signature` is `"(name[u_1,…,u_K])"` where `K` is the count of distinct free (unchangeable) arguments. On this branch the mail-broadcast compaction wrapper (`compileImplicationToCompact`, see [prover chapter](04_prover.md#compileimplicationtocompact)) is the first path that compiles a *fully-bound* implication — a proved theorem binds every variable ([D-75](../40_decisions.md#d-75)), so `K = 0` and the signature is `"(implication<N>[])"` (no arguments). An implication carrying genuinely-free arguments yields `"(implication<N>[u_1,…])"`. The `K = 0` form is well-formed only because `excludeRepetitions` no longer unconditionally strips a trailing comma (it would otherwise emit the malformed `"(implication<N>])"`); see the prover chapter Weaknesses entry.

This guarantees [I-23](../30_invariants.md#i-23) for cross-batch allocations across the entire run: the same logical operator carries the same compact name in every batch of a run, so `compiled_theorems.txt`, `raw_proof_graph/global_theorem_list.txt`, and the proof-graph chapters all agree. Pre-2026-05-13 the merge early-returned for incubator-prefixed tags (per the original [D-54](../40_decisions.md#d-54)); that skip is now superseded by [D-61](../40_decisions.md#d-61), enabled by the C++ category-aware `repetitionExclusionMap` key per [D-60](../40_decisions.md#d-60).

Because the binary is regenerated, this document cannot quote a byte-exact example from a fresh checkout. A representative schema (verifiable against [`memory.hpp`](../../../GL_Quick_VS/GL_Quick/src/memory.hpp) and post-run binaries):

```json
{
    "in": {
        "arity": 2,
        "category": "atomic",
        "elements": [],
        "signature": "(in[u_0,u_1])"
    },
    "NaturalNumbers": {
        "arity": 5,
        "category": "and",
        "elements": [
            "(in[u_1,u_0])",
            "(fXY[u_2,u_0,u_0])",
            "...one per Peano axiom..."
        ],
        "signature": "(NaturalNumbers[u_0,u_1,u_2,u_3,u_4])"
    },
    "fXY": {
        "arity": 3,
        "category": "and",
        "elements": ["(...typing...)","(...totality...)","(...functionality...)"],
        "signature": "(fXY[u_0,u_1,u_2])"
    },
    "or6": {
        "arity": 6,
        "category": "or",
        "elements": ["(...)","(...)","(...)"],
        "signature": "(or6[u_1,u_2,u_3,u_4,u_5,u_6])",
        "definedSet": "",
        "implications": [
            "(implication172[u_3,u_2,u_4,u_5,u_6,u_1])",
            "...the 3 K-rules in leaf order, then the 3 subset-exclusion compacts..."
        ]
    }
}
```

The verifier also consults this binary — see [`10_pipeline/08_verifier.md`](08_verifier.md).

---

## Code path — from file to compiled map

| Step | Function | File |
|---|---|---|
| 1. Config read | `ce::modifyCoreExpressionMap` | [`compiler.hpp`](../../../GL_Quick_VS/GL_Quick/src/compiler.hpp) |
| 2. Anchor key discovery | `ce::findAnchorKey` | [`compiler.hpp`](../../../GL_Quick_VS/GL_Quick/src/compiler.hpp) |
| 3. Anchor signature minting | `ce::makeAnchorSignature` | [`compiler.hpp`](../../../GL_Quick_VS/GL_Quick/src/compiler.hpp) |
| 4. Definition-file loading | Inside `modifyCoreExpressionMap` — reads from path under `files/definitions/` if `full_mpl` ends in `.mpl` | |
| 5. Compile into `CoreExpressionConfig` + `LogicalEntity` | `modifyCoreExpressionMap` + populated in per-entity sites | |
| 6. Serialisation to JSON | `CompilationPersistence::exportCompiledExpressionsJSON` | `compiler/compilation_persistence.cpp` |

---

## Worked example — what happens to `(in3[a,b,c,+])`

Take the atomic expression `(in3[a,b,c,+])` — "`a + b = c`".

1. **Definition-file lookup.** The compiler sees `"in3"` in the config, `full_mpl = "(in3[1,2,3,4])"`. Parsed in place; no separate file.
2. **Category determination.** `in3`'s body is an atomic predicate with four typed arg positions. `category = atomic`.
3. **`LogicalEntity` record.** `elements = []`, `arity = 4`, `signature = "(in3[u_0,u_1,u_2,u_3])"`.
4. **Hash indexing.** On every theorem that contains `in3`, the expression becomes a trie key in the LB's `HashMemory`. The `IntNormalizedKey` form strips the specific arg names and keeps only the structural shape + arg positions.
5. **Usage downstream.** When a theorem like `(>[i0,s,+,i1](AnchorPeano[N,i0,s,+,*,i1])(>[v1](in3[i0,v1,i1,+])(in2[i0,v1,s])))` enters the prover, the `(in3[i0,v1,i1,+])` fragment is a hash-key on admission — every LB with an `in3` rule active gets a chance to fire.

---

## Weaknesses

### Known & tracked

- **Generated binary identity is run-specific.** `files/GL_binaries/*` contains ignored generated artifacts; each batch rewrites its registry. Pair saved proofs with the matching archived registry rather than assuming the current generated files match `HEAD`. Examples of binary structure in this document are hand-written and verified against the parsing code.
- **Compiler bug, open: a universal with a nested premise inside a supplied row's existence or conjunction crashes the compile.** The FTA2 I2 draft (2026-10-18) had the head `!(>[p](p|n)!(&(2<=p)(>[d](d|n)(>[](2<=d)(p<=d)))))`. A caller outside [`compiler/structural_compiler.cpp`](../../../GL_Quick_VS/GL_Quick/src/compiler/structural_compiler.cpp), `StructuralCompiler::compileExpression`, compiles the negated body; the negated universal takes the existence branch, its witness `d` is renamed, and the toggled remainder `!(>[](2<=d)(p<=d))` re-enters the existence branch with an empty binder, where the single-witness assertion in [`compiler/expression_parser.cpp`](../../../GL_Quick_VS/GL_Quick/src/compiler/expression_parser.cpp), `ExpressionParser::renameLastRemoved`, fires. The form is valid MPL and must be processed (maintainer, 2026-10-18); the fix is deferred. Mitigation until then: keep such a universal inside a predicate's definition ([MPL cookbook](../03b_mpl_cookbook.md), section 5). Definitions with the same nesting (`prime`, `ordered`) compile.
- **Bug, open: an existence in a supplied row whose body is a conjunction of several predicates crashes the FTA batch's admission pass.** The FTA2 I4 draft (2026-10-18) had the head `∃k∈N ∃P: sequence ∧ primeseq ∧ ordered ∧ fold`. The integration rule of the head's existence carries the conjunction as a negated numbered OR, `!(or17[…])`; `or17` has a compiled entry but no core configuration, so `AdmissionMarkerBuilder::makeNormalizedKeysForAdmission` asserts (`cfg && "Core expression not found"`; trap-confirmed in run `fta_i4_short_02`). Mechanism by code reading: compiled entities get configurations from `StructuralCompiler::extendCoreExpressionMap`, which runs once at the end of definition compilation; `OrRuleCompiler::findOrMintOrOperator` registers a configuration when it mints, but the OR branch of `StructuralCompiler::compileExpression` mints through `CompilationReuse::excludeRepetitions` without one, so an OR first minted while compiling a conjecture stays unconfigured. Mitigation: bundle the properties in a definition (cookbook section 2, the `nontrivialFactor` pattern); I4 states its head through `hasPrimeFactorization`, a typed existence over the bundle `primeFactorSeq` ([configuration reference](../04_configs.md#fta-prime-factorizations)).

### Suspected fragility

- **Hand-written anchor definition files.** `AnchorPeano.mpl`, `AnchorGauss.mpl`, `AnchorIncubator.mpl`, `AnchorFTA.mpl` are each a small `and`-tree. The expected slot count (6 / 8 / 14 / TBD) is implicit in the file and hardcoded elsewhere (conjecturer, verifier, rename logic). Changing the slot count requires coordinated edits across files with no cross-check — the compiler does not reject a mismatch until much later in the pipeline.
- **`full_mpl` string vs file-path duality.** `modifyCoreExpressionMap` decides whether `full_mpl` is inline MPL or a filename based on whether it ends in `.mpl`. A user error (e.g. copying a filename into the inline field without `.mpl` suffix, or an inline string that ends with `.mpl`) can silently corrupt the compilation.
- **`definition_sets` type labels are stringly-typed.** The labels `(1)`, `P(1)`, `P(x(1)(1))`, `P(x(1)(x(1)(1)))`, … are parsed into structured type trees inside the conjecturer, but there is no single schema definition — they're string-match tested at multiple sites. Adding a new type label requires finding every match-site.

### Not exercised by tests

- **Public implication definitions downstream.** `primeseq` is the first registered definition whose body is a single implication; until it, the compiler branch for that shape asserted on every input. Its downstream handling (premise disintegration, goal integration, verifier, Lean export) is exercised only by the FTA F7 rows of the full pipeline, not by a dedicated unit test. Exception: an inherited (status-3) instance is a rule carrier like a numbered compact implication, with its own unit tests (`status_gate` suite; [D-467](../40_decisions.md#d-467)).
- **Malformed MPL error messages.** A definition file with a missing bracket or a typo produces a parse failure deep in the compiler. The error message does not always cite the file or line. A regression test that intentionally breaks each definition file and asserts the specific error is absent.
- **GL-binary schema stability across versions.** No test verifies that `GL_binary_Peano.json` matches a golden reference across commits. Silent field-name drift would not be detected until downstream code broke.

---

## Open questions

- **OPEN-7 — RESOLVED.** `fXY` and `fXYZ` have category `and`. `StructuralCompiler::compileExpression` compiles their conjunctions of defining conditions; existence conditions remain separate constituent entities.

---

## See also

- [`20_core_concepts/06_anchors_and_scopes.md`](../20_core_concepts/06_anchors_and_scopes.md) — anchor roles in the prover.
- [`10_pipeline/02_conjecturer.md`](02_conjecturer.md) — consumer of the compiled `CoreExpressionConfig` for conjecture generation.
- [`02_glossary.md`](../02_glossary.md) — MPL glossary entry.
- [I-1](../30_invariants.md#i-1) — precompile structural operators before passing theorems into the pipeline.

## Compiled contraposition auxiliaries

### Prepared conjunction refutations

`compileConjunctionRefutations` prepares fixed-parameter direct member refutations for concrete AND alternatives of admitted ORs. `LogicalEntity::conjunctionRefutations` retains these member/compact pairs in the versioned `conjunction_refutations` binary object, outside defining elements and integration instructions.

`compileSupportingImplications` separately snapshots implication and prepared contrapositive owners with immediate negated AND premises. For each conjunction it emits exactly one complete premise chain per immediate part, complementing that part and retaining all other signed parts. Nested parts stay opaque, and support auxiliaries never seed further support or contraposition families. Every non-fixed head argument must occur in a premise. Version 2 of `supporting_implications` stores the guarded proof templates, a parallel `composed_rules` list for unary owners, preparation state and auxiliary identity; canonical-name replacement rewrites both lists. Version 1 loads with preparation pending so the normal compiler seam completes direct compositions. A nonempty composed list must align exactly with the helper list. Bound variables use names distinct from the bare fixed-parameter tokens before compact compilation; numeric bound variables must not capture fixed ports. `compilation_state.py`, `rewrite_auxiliary_references`, also migrates all auxiliary metadata and imports its dependency closure across batch registries.

Compilation installs no runtime rules. Ordinary disintegration of a unary implication publishes its prepared direct compositions, with checked helper expansion followed by ordinary implication-composition history and compact compilation. The helper remains proof-only and the original implication remains installed. Owners outside the unary-consumer shape publish their guarded helpers as before. Direct OR admission publishes only the concrete alternative's direct rules. See [I-259](../30_invariants.md#i-259).

**When the passes run ([D-524](../40_decisions.md#d-524)).** `compileContrapositionFamilies` runs at the end of every structural compile (and at the registry-completion seam) and calls the two other passes; every call walks the whole registry. A pass records the entity address of every entry while it collects its owners and, after preparing them, marks the entries it added through `SupportingRuleCompiler::forEachEntryWithAdded` — one walk alongside the recorded addresses, the answer a set of the names present gives, without copying a name. The covered-source set behind the uncovered-source assert is built only when a standalone source is incomplete; refutation helpers are found through the map's transparent comparator.

### Prepared reversals

`compiler/supporting_rule_compiler.cpp`, `compileContrapositionFamilies`, retains the compiler's split forward rules and prepares every binder-covering subset of compatible heads sharing a complete signed premise within one conjunction family. A singleton produces direct contraposition; a larger subset produces an auxiliary conjunction, grouped source implication and its reversal. Definition parameters stay fixed and the complete bound-variable set aligns through the premise.

`LogicalEntity::contrapositives` records the exact source compact instances plus grouped-source and reversed compact templates. The GL binary carries a versioned `contraposition` object with preparation state and source correspondence. These fields are independent of `elements` and definition identity; canonical compilation-name replacement rewrites their references. Missing legacy metadata is completed before workers at the registry-completion seam. Generated auxiliaries do not recursively generate new grouping families. A grouped source carries its prepared direct reversal for independent activation after mail or rewriting. Conjunction compaction returns bare argument names; restore fixed-parameter prefixes before forming the source and reverse binder lists. At disintegration, each auxiliary becomes its own statement compact through `prepareIntegrationCore` and ordinary implication disintegration, following the attached-implication pattern used by existences. See [I-260](../30_invariants.md#i-260).

**D-382 exception.** The maintainer excludes negated-existence body-rule contraposition: a unary source whose head denies membership of its sole bound variable in a fixed set. `compiler/supporting_rule_compiler.cpp`, `isContrapositionTypingExclusion`, recognizes this structural class before grouping or direct reversal. The forward body rule remains; its typing-to-negative-body converse remains absent. Positive typing heads, negative relation heads, fixed-element negatives and multiple-bound-variable rules are not this exception.

**D-468 exception.** A configured definition whose whole body is one implication and whose head does not mention every variable its premise binds (`primeseq`: `P(i)=x` implies `prime(x)`, the index `i` absent from the head) has no binder-covering contrapositive: the reversed rule would leave that variable unbound. `compiler/supporting_rule_compiler.cpp`, `isContrapositionUncoveredDefinition`, recognizes it before grouping or direct reversal; "configured" is `expressionsFromConfig` membership, so generated implication compacts of the same shape keep the coverage assert. The forward rule remains installed; no reversed rule is prepared.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
