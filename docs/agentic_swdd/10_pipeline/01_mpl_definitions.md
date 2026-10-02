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
> **Owner:** `compiler.cpp` / `compiler.hpp`.
> **Pipeline position:** runs once at the start of every `gl_quick.exe` invocation; conjecturer and prover both rely on the compiled output.

---

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
- `(>[a](in[a,N])(>[b](in3[a,i0,b,+])(=[a,b])))` — `+` has identity `i0 = 0` (the "and (=[a,b]) implies in3[a,i0,b,+]" direction is the iff pair).
- `(>[b](in[b,N])(>[a,c,d](...)(...)))` — the successor rule for `+`: if `s(b) = c` and `a + b = d`, then for every `e = s(d)`, `a + c = e` and vice versa.
- Analogous conjuncts for `*`.

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

## Compiled form — `CoreExpressionConfig` & `LogicalEntity`

After the compiler reads a definition, it produces two in-memory records per expression name:

### `CoreExpressionConfig` — the configuration surface

Defined at [`compiler.hpp`](../../GL_Quick_VS/GL_Quick/src/compiler.hpp). Fields:

| Field | Meaning |
|---|---|
| `arity` | Number of arguments the expression takes. |
| `definition` | Parsed MPL body — the canonical expression tree with `u_`-prefixed formal parameters. |
| `signature` | Canonical form `"(name[u_0,u_1,...])"`. |
| `definitionSets` | Per-argument type (`(1)`, `P(1)`, `P(x(1)(1))`, …) + whether the arg is "combinable" (used by the conjecturer). |
| `inputArgs`, `outputArgs` | String-form argument labels from the config. |
| `inputIndices`, `outputIndices` | Parsed integer indices. |

### `LogicalEntity` — the metadata per definition

Defined at [`memory.hpp`](../../GL_Quick_VS/GL_Quick/src/memory.hpp). Fields:

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
- `implication` — terminal. Stored as hash-table rules: premise-signature → conclusion-template.

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

The compiler's output, and — since [D-22](../40_decisions.md#d-22) — also one of its **inputs**. Maps expression name to `{arity, category, elements, signature, definedSet}` — plus, on **or and existence entries**, `implications`: for an or, its K mutual-exclusion and subset-exclusion rules as `(implication<N>[u_…])` compacts over the or's own `u_` tokens (K-rules in leaf order, then the subset-exclusion rules in the disintegrator's enumeration order; the argument order is the projection onto the or's tokens — [D-309](../40_decisions.md#d-309)); for a two-element existence, its two existence implications `left → !right` then `right → !left` as compacts over the existence's own tokens ([D-310](../40_decisions.md#d-310)). The writer omits the key when the list is empty, so every other entry keeps the earlier layout byte-for-byte; an or or existence loaded without it is completed at the next `preMintReducedOrs` seam. Generated at every run; **gitignored**.

**Cross-batch shared registry.** Spontaneous compact operator names (`implication<N>`, `existence<N>`, `or<N>`, `and<N>`) live in a sibling file `files/GL_binaries/GL_binary_shared.json`. This shared file is the cross-batch source of truth for spontaneous identifiers. Per-batch lifecycle:

1. **Pre-batch (Python).** [`run_modes.py::_seed_per_batch_binary`](../../run_modes.py) copies `GL_binary_shared.json` into `GL_binary_<Tag>.json` immediately before invoking `gl_quick.exe <Tag>`. On a clean run when shared does not exist yet, an empty `{}` is written. Python is the sole writer of shared and the sole creator of the per-batch file at this step.
2. **C++ startup.** The `ExpressionAnalyzer` constructor calls [`loadGlBinary`](../../GL_Quick_VS/GL_Quick/src/visualizer.cpp) on the per-batch file its `glBinaryFile` parameter names — the batch driver in [`run_modes.cpp`](../../GL_Quick_VS/GL_Quick/src/run_modes.cpp) passes `files/GL_binaries/GL_binary_<Tag>.json`; a constructor without a file (every unit test) reads nothing and compiles the registry from the config alone. Every entry is inserted into `compiledExpressions`. The current loader registers `repetitionExclusionMap` only after its category-name-prefix and numeric-suffix checks, so named logical predicates such as `preorder` are incorrectly omitted from that reuse cache. Numbered spontaneous entries get a row keyed by `(elements, category)` — both fields drawn from the JSON entry; the elements vector IS the `splitNK` exactly as `excludeRepetitions` stored it at allocation time. The category is part of the key because two structurally-identical bodies under different categories must allocate distinct names — see [D-60](../40_decisions.md#d-60) for the dormant cross-batch collision this guards against. The four shared counter members (`implCounter`, `existenceCounter`, `andCounter`, `orCounter`) are seeded from `max(N) + 1` so newly-allocated names start above the highest already in use.
3. **C++ shutdown.** `exportCompiledExpressionsJSON` writes the entire `compiledExpressions` map to the same per-batch path. Because shared entries were preloaded, the resulting file contains the inherited shared entries plus this batch's new spontaneous allocations. The path is computed from the running executable through `project_paths.cpp`, `projectRoot` and is identical for every batch (incubator and main alike) — see [D-54](../40_decisions.md#d-54) for the unification of the writer path that previously diverged into `files/incubator/GL_binaries/` for incubator batches.
4. **Post-batch (Python).** [`run_modes.py::_merge_into_shared`](../../run_modes.py) reads the per-batch file and adds any name not already in shared whose `category` is in `{implication, existence, or, and}` (atomic entries are excluded). Applies to EVERY tag — incubator and main alike. The shared file grows monotonically over the run.

**Known loader weakness — named predicate reuse.** `loadGlBinary` in `visualizer.cpp`
lets the counter-name filter skip reuse registration as well. The later
`compileCoreExpressionMapCore` returns immediately for an already-loaded named
entity, leaving that entry absent. In the full FTA input this makes strictOrder
compile through a duplicate `existence16` instead of the loaded `preorder`.
Negated strict-order expansion then refutes the duplicate predicate while OR
branch closure probes the public predicate name. The resulting live false
branches trigger the measured burst-16 cost. The permanent loader correction
is pending; diagnostic evidence is in [`rt_restoration_plan.md`](../../GPU/rt_restoration_plan.md).

**`implication<N>` signature shape.** A spontaneous entry's `signature` is `"(name[u_1,…,u_K])"` where `K` is the count of distinct free (unchangeable) arguments. On this branch the mail-broadcast compaction wrapper (`compileImplicationToCompact`, see [prover chapter](04_prover.md#compileimplicationtocompact)) is the first path that compiles a *fully-bound* implication — a proved theorem binds every variable ([D-75](../40_decisions.md#d-75)), so `K = 0` and the signature is `"(implication<N>[])"` (no arguments). An implication carrying genuinely-free arguments yields `"(implication<N>[u_1,…])"`. The `K = 0` form is well-formed only because `excludeRepetitions` no longer unconditionally strips a trailing comma (it would otherwise emit the malformed `"(implication<N>])"`); see the prover chapter Weaknesses entry.

This guarantees [I-23](../30_invariants.md#i-23) for cross-batch allocations across the entire run: the same logical operator carries the same compact name in every batch of a run, so `compiled_theorems.txt`, `raw_proof_graph/global_theorem_list.txt`, and the proof-graph chapters all agree. Pre-2026-05-13 the merge early-returned for incubator-prefixed tags (per the original [D-54](../40_decisions.md#d-54)); that skip is now superseded by [D-61](../40_decisions.md#d-61), enabled by the C++ category-aware `repetitionExclusionMap` key per [D-60](../40_decisions.md#d-60).

Because the binary is regenerated, this document cannot quote a byte-exact example from a fresh checkout. A representative schema (verifiable against [`memory.hpp`](../../GL_Quick_VS/GL_Quick/src/memory.hpp) and post-run binaries):

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
        "category": "existence",
        "elements": ["(...)","(...)"],
        "signature": "(fXY[u_0,u_1,u_2])",
        "definedSet": 0,
        "implications": [
            "(implication34[u_0,u_1,u_2])",
            "...left → !right, then right → !left, over the existence's tokens..."
        ]
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
| 1. Config read | `ce::modifyCoreExpressionMap` | [`compiler.hpp`](../../GL_Quick_VS/GL_Quick/src/compiler.hpp) |
| 2. Anchor key discovery | `ce::findAnchorKey` | [`compiler.hpp`](../../GL_Quick_VS/GL_Quick/src/compiler.hpp) |
| 3. Anchor signature minting | `ce::makeAnchorSignature` | [`compiler.hpp`](../../GL_Quick_VS/GL_Quick/src/compiler.hpp) |
| 4. Definition-file loading | Inside `modifyCoreExpressionMap` — reads from path under `files/definitions/` if `full_mpl` ends in `.mpl` | |
| 5. Compile into `CoreExpressionConfig` + `LogicalEntity` | `modifyCoreExpressionMap` + populated in per-entity sites | |
| 6. Serialisation to JSON | Visualiser — `exportCompiledExpressionsJSON` | [`visualizer.cpp`](../../GL_Quick_VS/GL_Quick/src/visualizer.cpp) |

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

### Suspected fragility

- **Hand-written anchor definition files.** `AnchorPeano.mpl`, `AnchorGauss.mpl`, `AnchorIncubator.mpl`, `AnchorFTA.mpl` are each a small `and`-tree. The expected slot count (6 / 8 / 14 / TBD) is implicit in the file and hardcoded elsewhere (conjecturer, verifier, rename logic). Changing the slot count requires coordinated edits across files with no cross-check — the compiler does not reject a mismatch until much later in the pipeline.
- **`full_mpl` string vs file-path duality.** `modifyCoreExpressionMap` decides whether `full_mpl` is inline MPL or a filename based on whether it ends in `.mpl`. A user error (e.g. copying a filename into the inline field without `.mpl` suffix, or an inline string that ends with `.mpl`) can silently corrupt the compilation.
- **`definition_sets` type labels are stringly-typed.** The labels `(1)`, `P(1)`, `P(x(1)(1))`, `P(x(1)(x(1)(1)))`, … are parsed into structured type trees inside the conjecturer, but there is no single schema definition — they're string-match tested at multiple sites. Adding a new type label requires finding every match-site.

### Not exercised by tests

- **Malformed MPL error messages.** A definition file with a missing bracket or a typo produces a parse failure deep in the compiler. The error message does not always cite the file or line. A regression test that intentionally breaks each definition file and asserts the specific error is absent.
- **GL-binary schema stability across versions.** No test verifies that `GL_binary_Peano.json` matches a golden reference across commits. Silent field-name drift would not be detected until downstream code broke.

---

## Open questions

- **OPEN-7 — RESOLVED.** `fXY` and `fXYZ` have category `and`. `ExpressionAnalyzer::compileCoreExpressionMapCore` compiles their conjunctions of defining conditions; existence conditions remain separate constituent entities.

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

### Prepared reversals

`contraposition.cpp`, `compileContrapositionFamilies`, retains the compiler's split forward rules and prepares every binder-covering subset of compatible heads sharing a complete signed premise within one conjunction family. A singleton produces direct contraposition; a larger subset produces an auxiliary conjunction, grouped source implication and its reversal. Definition parameters stay fixed and the complete bound-variable set aligns through the premise.

`LogicalEntity::contrapositives` records the exact source compact instances plus grouped-source and reversed compact templates. The GL binary carries a versioned `contraposition` object with preparation state and source correspondence. These fields are independent of `elements` and definition identity; canonical compilation-name replacement rewrites their references. Missing legacy metadata is completed before workers at the registry-completion seam. Generated auxiliaries do not recursively generate new grouping families. A grouped source carries its prepared direct reversal for independent activation after mail or rewriting. Conjunction compaction returns bare argument names; restore fixed-parameter prefixes before forming the source and reverse binder lists. At disintegration, each auxiliary becomes its own statement compact through `prepareIntegrationCore` and ordinary implication disintegration, following the attached-implication pattern used by existences. See [I-260](../30_invariants.md#i-260).

**D-382 exception.** The maintainer excludes negated-existence body-rule contraposition: a unary source whose head denies membership of its sole bound variable in a fixed set. `contraposition.cpp`, `isContrapositionTypingExclusion`, recognizes this structural class before grouping or direct reversal. The forward body rule remains; its typing-to-negative-body converse remains absent. Positive typing heads, negative relation heads, fixed-element negatives and multiple-bound-variable rules are not this exception.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
