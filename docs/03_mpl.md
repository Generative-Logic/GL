<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# MPL — Mathematical Programming Language `[DRAFT]`

> MPL is the one language the entire GL pipeline speaks. Every definition file, every theorem, every row of the processed proof graph, every row of a `simple_facts` table, every rule installed in an LB's hash memory — all of it is MPL strings. This chapter is a dedicated reference for MPL: the formal grammar, the variable-naming conventions, the type-label vocabulary, the shape patterns that recur across the codebase, and the common reading exercises.

---

## Chapter map

- [Design goals](#design-goals)
- [Formal grammar](#formal-grammar)
- [Core primitives](#core-primitives)
- [Variable conventions](#variable-conventions)
- [Type-label vocabulary](#type-label-vocabulary)
- [Recurring shape patterns](#recurring-shape-patterns)
- [Pretty-printing](#pretty-printing)
- [Reading exercises](#reading-exercises)
- [Common mistakes](#common-mistakes)
- [Extending MPL](#extending-mpl)
- [Weaknesses](#weaknesses)
- [Open questions](#open-questions)

---

## Design goals

MPL exists to make these claims operationally trivial:

1. **Canonical form.** Two expressions that mean the same thing must have the same bytes. If they don't, the hash engine cannot match them. The grammar is deliberately rigid — no syntactic variation is permitted where none is semantically warranted.
2. **Hashable without parsing.** Every expression's top-level shape is visible in the first three characters (`(<head>[...]`, `(&...`, `(>[...]...`, `(=[...]`, `!...`). Fast dispatch without a full parse.
3. **Single-byte delimiters.** `(`, `)`, `[`, `]`, `,` are the only structural characters. No whitespace, no string-literals-with-escapes, no variable-width separators.
4. **Human-readable enough.** GL developers read MPL directly when inspecting chapters, debug dumps, or theorem files. The grammar is not encrypted.

The result is a notation that looks like S-expressions superficially — but with a strict argument-list shape (`[...]` with commas) and explicit implication/quantifier markers (`>` with its bound-variable list).

---

## Formal grammar

```
expression    := atomic
              |  conjunction
              |  implication
              |  equality
              |  negation

atomic        := "(" name "[" args "]" ")"
conjunction   := "(" "&" expression+ ")"
implication   := "(" ">" "[" bound-vars "]" expression expression ")"
equality      := "(" "=" "[" arg "," arg "]" ")"
negation      := "!" expression

name          := identifier                       -- operator / predicate / compound-expression name
args          := /empty/ | arg ("," arg)*
arg           := identifier | integer-literal     -- no spaces; commas only
bound-vars    := /empty/ | identifier ("," identifier)*

identifier    := [A-Za-z_][A-Za-z0-9_]*           -- including prefix conventions below
```

No whitespace is permitted **anywhere** inside an expression. `(in3[a, b, c, +])` is invalid — `(in3[a,b,c,+])` is the only acceptable form. Whitespace in definition source files (visible as indentation in `NaturalNumbers.txt` for example) is stripped by the compiler before parsing and carries zero semantic weight.

The grammar is context-free enough to parse left-to-right with a small state machine. `compiler.hpp` implements the parse; `process_proof_graphs.py` and `verifier.py` each have their own independent parse (the verifier by design — see [I-16](30_invariants.md#i-16)).

---

## Core primitives

### Atomic — `(name[args])`

The leaf expression shape. `name` is an operator or predicate; `args` are its arguments (comma-separated). The interpretation depends on whether `name` is defined with `category: atomic` in the GL binary (then it's a ground predicate) or with a compound category (then it expands).

Examples:

```text
(in[i0,N])          -- i0 is in N
(=[a,b])            -- a equals b
(in2[x,y,f])        -- f(x) = y (single-input operator applied)
(in3[a,b,c,+])      -- a + b = c (two-input operator applied)
(fXY[s,N,N])        -- s is a function from N to N
(AnchorPeano[N,i0,s,+,*,i1])  -- the Peano anchor, pinning the 6 slots
```

### Conjunction — `(&expr1 expr2...)`

N-ary conjunction. Two or more sub-expressions, no comma separators (only the `&` marker distinguishes the shape from an atomic). Most definition bodies are deeply-nested conjunctions.

```text
(&
    (in[i0,N])
    (fXY[s,N,N])
)
```

Note that `(&)` with zero sub-expressions is not grammatically valid — there must be at least two elements. `(&X)` with exactly one is also atypical; the prover compiles it away at definition time.

### Implication — `(>[bound-vars](premise)(body))`

Universally-quantified implication. `bound-vars` is the list of identifiers universally quantified over the body (may be empty `>[]`). `premise` is the antecedent; `body` is the consequent.

```text
(>[x](in[x,N])(P(x)))           -- for all x in N, P(x)
(>[](Q)(R))                      -- Q implies R (no quantifier)
(>[a,b,c](P1)(>[d](P2)(P3)))    -- nested — forall a,b,c,d: P1, P2 ⇒ P3
```

The nested form is how multi-premise rules are encoded: each `>` takes exactly one premise + one body, so three premises mean three nested `>`s.

Implication with an empty bound-list `>[]` is distinct from omitting the `>` — it marks the expression as a *conditional* with no additional quantification. Used when both sides already contain their variables from an outer scope.

### Equality — `(=[a,b])`

Ground equality of two arguments. Special-cased throughout the prover: equivalence-class registration, automatic mirror emission `(=[b,a])` (guarded by `a!= b` — see [I-9](30_invariants.md#i-9)), dedicated verifier tags.

### Negation — `!expr`

Prefix negation on any expression. Not parenthesised — `!` directly precedes the expression it negates:

```text
!(=[a,b])                       -- a ≠ b
!(in2[x,y,f])                   -- f(x) ≠ y (or: (x,y) is not in the f relation)
!(&(P1)(P2))                    -- not (P1 and P2)
!(>[x](P)(Q))                   -- there exists x such that P but not Q
```

Negation composes arbitrarily. Double negation `!!X` is grammatical but not produced by the conjecturer (redundant); the prover, verifier, and compiler handle it if encountered.

The `!(&...)` and `!(>...)` forms are **structural operators** — shape-patterns that the compiler rewrites into named compiled expressions (`or<N>` / `existence<N>`) during `precompileStructuralOperators`. See [I-1](30_invariants.md#i-1).

---

## Variable conventions

MPL variable names follow prefix conventions that encode their role. The prefix alone carries information about how the variable will be handled downstream.

### Raw numeric indices — conjecturer output

The conjecturer emits bound variables as bare integers `1`, `2`, `3`, …:

```text
(>[2,3,4,6](AnchorPeano[1,2,3,4,5,6])(>[7](in2[2,7,3])(in3[2,6,7,4])))
```

Raw form. Each integer is a fresh bound variable within the theorem's scope. Anchor slots are also given integer placeholders — the outer `(AnchorPeano[1,2,3,4,5,6])` pins them to indices, which the renamer later translates.

### `v<k>` — renamed bound variables

After `process_proof_graphs.py` runs its four-priority renaming scheme (see [`10_pipeline/06_process_proof_graph.md`](10_pipeline/06_process_proof_graph.md)), non-anchor bound variables become `v1`, `v2`, `v3`, …:

```text
(>[i0,s,+,i1](AnchorPeano[N,i0,s,+,*,i1])(>[v1](in2[i0,v1,s])(in3[i0,i1,v1,+])))
```

The `v<k>` numbering is seeded from the theorem expression's left-to-right scan, so chapters and `global_theorem_list.txt` share a v-numbering (see [I-10](30_invariants.md#i-10)).

### Anchor-slot names — `N`, `i0`, `s`, `+`, `*`, `i1`, `i2`, `id`, `i3`..`i8`

Canonical names for anchor slots. Set by the anchor's definition and preserved through renaming. The digit-indexed `i<k>` naming denotes the natural number `k` (`i0 = 0`, `i1 = 1`, etc.). Symbolic names (`N`, `s`, `+`, `*`, `id`) denote sets / functions.

### `u_<k>` — formal parameters in GL-binary definitions

Inside a compiled expression's body, the formal parameters are `u_0`, `u_1`, … These are the slots an instantiation fills. They must remain *free* (not quantified) when a definition is instantiated. `reconstructImplicationFullBind` ([I-4](30_invariants.md#i-4)) uses the `u_` prefix as the gate — it binds every non-`u_` variable universally.

### `x<k>` — prover-internal anchor-slot copies

During anchor handling, the prover mints fresh copies of anchor-slot `(1)`-typed variables to keep them distinct from their originals. The minted names are `x` + the original slot name. See [`prover.cpp–3449`](../GL_Quick_VS/GL_Quick/src/prover.cpp). Priority 2.5 of the processor renaming (see [`10_pipeline/06_process_proof_graph.md`](10_pipeline/06_process_proof_graph.md)) rewrites these `x<N>` to `<digit>_copy` in the processed proof graph.

### `it_<k>` — Pass B iteration variables

Freshly-minted inside `disintegrateExpr2` (Pass B). Subject to the admission-map gate + the single-input-operator fallback. See [I-6](30_invariants.md#i-6).

### `int_<k>` — integration variables

Integration-side mirror of `it_<k>`. Uses only map-based admission — the single-input fallback does not apply.

### `repl_lev_<N>_<M>` — substitution-chain placeholders

Minted during anchor-handling and integration rewrites. The numbers encode the rewrite level and position. Internal names; rarely seen in final output but visible in debug dumps.

### `<base>_copy` — duplicated variables

A variable with `_copy` appended is a fresh duplicate of `<base>`, introduced during hypothesis handling or OR branching to keep the hypothesis variable distinct from its surrounding scope. See the `variable copy` tag in [`20_core_concepts/08_proof_tags.md`](20_core_concepts/08_proof_tags.md).

---

## Type-label vocabulary

Every argument position in a config's `definition_sets` carries a type label — a compact string describing the set the argument ranges over. These labels drive the conjecturer's type-consistent combination logic and the compiler's anchor-slot typing.

| Label | Meaning | Example slot |
|---|---|---|
| `(1)` | An element of `N` (the base set) | `i0`, `i1`, `v1` |
| `P(1)` | A subset of `N` (a set) | `N`, `M` |
| `P(x(1)(1))` | A relation on `N × N` — used as a function on `N` | `s` (successor), `id` (identity) |
| `P(x(1)(x(1)(1)))` | A relation on `N × N × N` — used as a binary function on `N` | `+`, `*` |

The syntax reads recursively: `P(X)` is "power set of X" (i.e. subsets of X), and `x(A)(B)` is "Cartesian product of A and B". So:

- `P(1)` = subsets of the domain = sets.
- `x(1)(1)` = `1 × 1` = pairs of domain elements.
- `P(x(1)(1))` = subsets of pairs = relations.
- `x(1)(x(1)(1))` = `1 × (1 × 1)` = triples.
- `P(x(1)(x(1)(1)))` = subsets of triples = ternary relations (binary functions).

New type labels can be constructed using the same operators. `P(x(x(1)(1))(1))` would be "a relation between pairs and single elements" — not currently used but grammatically valid.

The `[<label>, <combinable>]` pair in `definition_sets` has the label first and a boolean second. The boolean decides whether the argument can participate in conjecturer combination (some arguments are "fixed" and should not be generalised).

---

## Recurring shape patterns

MPL expressions fall into a handful of shape patterns that recur across the codebase. Recognising them speeds up reading.

### Universal quantifier — `(>[x](typing)(body))`

*"For all x in T, body(x)."* Typing is usually `(in[x,T])` restricting `x` to a set.

```text
(>[v1](in[v1,N])(P(v1)))        -- for all v1 in N, P(v1)
```

### Existential quantifier via double negation — `!(>[x](typing)!(body))`

*"There exists x in T such that body(x)."*

```text
!(>[v1](in[v1,N])!(P(v1)))       -- there exists v1 in N with P(v1)
```

Rationale: MPL has no native existential quantifier. The double-negated universal encodes it cleanly, and the compiler rewrites it to `existence<N>` compiled names during `precompileStructuralOperators`.

### Iff via conjunction of two implications — `(&(>[x](P)(Q))(>[x](Q)(P)))`

*"P iff Q."*

```text
(&
    (>[x](P(x))(Q(x)))
    (>[x](Q(x))(P(x)))
)
```

Seen throughout definition files whenever a set is defined by a characteristic predicate — `fXY`, `interval`, `limitSet`, etc.

### Function typing — `(fXY[f,X,Y])`

*"f is a total function from X to Y."*

The body of `fXY` unpacks to three clauses: typing (the graph of f is contained in X × Y), totality (every x in X has *some* y in Y), and functionality (every x in X has *at most one* y). See [`files/definitions/fXY.txt`](../files/definitions/fXY.txt).

### Anchor application — `(Anchor<Tag>[slot1,slot2,...])`

Every theorem's outermost atomic expression. Pins the axiomatic context. See [`20_core_concepts/06_anchors_and_scopes.md`](20_core_concepts/06_anchors_and_scopes.md).

### Multi-premise implication — nested `>`

*"For all x,y: P1(x,y) ∧ P2(x,y) ⇒ R(x,y)"* becomes:

```text
(>[x,y](P1(x,y))(>[](P2(x,y))(R(x,y))))
```

The outer `>` carries the quantifier. Each inner `>[]` (with empty bound-list) adds one more premise. Final body is the conclusion.

### Vacuous implication — `(>[](P)(R))`

*"P implies R"* without any additional quantifier. Used when both P and R already reference variables bound by an outer scope.

### Operator equality (negated) — `!(in3[a,b,c,+])`

*"a + b ≠ c."* Negated atomics of `in2` / `in3` encode disequalities between function applications.

---

## Pretty-printing

Source definition files are indented for readability (e.g. [`files/definitions/NaturalNumbers.txt`](../files/definitions/NaturalNumbers.txt)). The compiler strips whitespace before parsing, so the indentation serves maintainers only.

Generated HTML chapter pages (stage 9, [`10_pipeline/07_html_export.md`](10_pipeline/07_html_export.md)) apply their own pretty-printing via the `processText` JavaScript function embedded in every page: click an expression to expand it to indented form.

Canonical indentation convention (as seen in definition files):

- Each new level of nesting indents by one tab.
- `&` opens on its own line; sub-expressions follow one per line.
- `>[...]` stays on the opening line with the bound-var list; the two body-slots go to separate lines (or stay inline if short).

There is no reformat tool in the repo today — indentation is hand-maintained.

---

## Reading exercises

Four real MPL expressions decoded step-by-step.

### Exercise 1 — a simple anchor application

```text
(AnchorPeano[N,0_copy,s,+,*,i1])
```

Read right-to-left through slots:

- Slot 6 = `i1` (the successor of zero).
- Slot 5 = `*` (multiplication).
- Slot 4 = `+` (addition).
- Slot 3 = `s` (successor function).
- Slot 2 = `0_copy` (a fresh copy of `i0` — post-anchor-handling). Originally `i0`.
- Slot 1 = `N` (the base set).

Semantic: the Peano anchor with the zero slot duplicated into a `_copy` variable. Produced by the anchor-handling step of some chapter.

### Exercise 2 — a typing predicate

```text
(>[v2,v3](in2[v2,v3,s])(in[v2,N]))
```

Parse:

- Outer: `(>[v2,v3](A)(B))` — for all `v2, v3`: A implies B.
- A = `(in2[v2,v3,s])` — `s(v2) = v3`, i.e. v3 is the successor of v2.
- B = `(in[v2,N])` — v2 is in N.

Reading: "For all v2, v3: if s(v2) = v3, then v2 ∈ N." This is a consequence of `s: N → N` — every element that has a successor is itself in N. It appears as the cited rule in `11_induction_typing.txt` on the current branch.

### Exercise 3 — Peano's successor injectivity

```text
(>[m](in[m,N])(>[n1,n2](&(in2[n1,m,s])(in2[n2,m,s]))(=[n1,n2])))
```

Parse:

- Outer: `(>[m](A)(B))` — for all m: A implies B.
- A = `(in[m,N])` — m ∈ N.
- B = `(>[n1,n2](C)(D))` — for all n1, n2: C implies D.
- C = `(&(in2[n1,m,s])(in2[n2,m,s]))` — s(n1) = m AND s(n2) = m.
- D = `(=[n1,n2])` — n1 = n2.

Reading: "For all m ∈ N, for all n1 and n2: if s(n1) = m and s(n2) = m, then n1 = n2." This is the standard statement that `s` is injective on its fibres — one of the Peano axioms, part of `NaturalNumbers`.

### Exercise 4 — a Gauss reformulated statement

```text
(>[N,i0,s,+](AnchorGauss[N,i0,s,+,*,i1,i2,id])(>[v1,v2](in2[v1,v2,s])(>[v3](interval[N,+,i0,v2,v3])(existence4[N,+,v3,v1,i0]))))
```

Parse:

- Outer: `(>[N,i0,s,+](A)(B))` — for all N, i0, s, +: A implies B.
- A = `(AnchorGauss[...])` — the Gauss anchor context.
- B = `(>[v1,v2](C)(D))` — for all v1, v2: C implies D.
- C = `(in2[v1,v2,s])` — s(v1) = v2 (v2 is the successor of v1).
- D = `(>[v3](E)(F))` — for all v3: E implies F.
- E = `(interval[N,+,i0,v2,v3])` — v3 is the interval [i0, v2] = [0, v2].
- F = `(existence4[N,+,v3,v1,i0])` — a compiled existence claim; expands to "there exists a function g on the interval v3 such that g(0) = 0 and g satisfies the fold recurrence with f = +, producing v1 at the right endpoint".

Reading: "In the Gauss context, if s(v1) = v2 and v3 is the interval [0, v2], then there exists a fold-valid function on v3 that accumulates to v1 at v2." A piece of the Gauss summation machinery.

---

## Common mistakes

- **Inserting spaces after commas.** `(in3[a, b, c, +])` — invalid. Use `(in3[a,b,c,+])`. Parser rejects.
- **Omitting the `!` for structural operators.** `(&...` is a conjunction; `!(&...)` is "not (conjunction)" which the compiler rewrites to `or<N>`. Omitting the `!` silently changes the semantics.
- **Binding anchor-slot names in inner `>[...]`.** Forbidden by [I-11](30_invariants.md#i-11). The outer anchor application supplies them.
- **Using `(=[x,x])` in a head.** Trivial equality is forbidden as a conclusion but allowed as a premise ([I-8](30_invariants.md#i-8)). Conjecturer filters for this.
- **Miscounting parentheses.** Each opening `(` must match a closing `)`. No auto-closing. When a MPL string gets corrupted, it's usually because a `)` was lost during manual editing. `verifier.py`'s `_find_matching_paren` at [`verifier.py`-ish](../verifier.py) is the reference parenthesis-matcher; use it as a mental model.
- **Forgetting that `!` is a prefix, not a wrap.** `!(P)` is correct; `(!(P))` is not a valid MPL expression (the extra parens surround the negation itself, turning it into a malformed expression).

---

## Extending MPL

Adding a new operator or predicate:

1. Add a per-expression entry to every relevant `Config<Tag>.json` — see [`04_configs.md`](04_configs.md).
2. Add a definition file to `files/definitions/` (or use inline `full_mpl` if the body is small).
3. If the operator has `output_args`, it joins the prover's `operators` set at [`prover.cpp–200`](../GL_Quick_VS/GL_Quick/src/prover.cpp) automatically (the set is recomputed from `coreExpressionMap` every run).
4. Ensure the GL binary's `category` is correctly inferred — `atomic` for leaves, `and` / `existence` / `implication` / `or` for compound. See `compiler.cpp` loading logic.
5. Verify that `process_proof_graphs.py`'s anchor-mapping and HTML-export's `rename_expr_peano` / `rename_expr_gauss` handle the new name (they may default-through, which is OK for auxiliary operators).

Adding a new structural operator (a new shape like `@`, `|`,...):

- Touch the grammar itself — not recommended without discussion. The current grammar is deliberately minimal. New shapes would require parallel updates to every MPL parser (C++ compiler, C++ conjecturer, `process_proof_graphs.py`, `verifier.py`) without introducing divergence.

---

## Weaknesses

### Known & tracked

- **Grammar not formally specified.** The BNF above is my reconstruction, not an authoritative reference. A drift between what the C++ parser accepts and what `verifier.py` accepts would be hard to spot. A published formal grammar (possibly ANTLR-compatible) is on the long-term list.
- **No canonical reformatter.** Definition files are hand-indented. A bulk reformat would improve readability; no tool exists.

### Suspected fragility

- **Three independent MPL parsers.** C++ compiler, C++ conjecturer, Python processor, Python verifier — that's actually four. A bug in one that the others don't have creates silent divergence. The verifier's independence is a feature (see [I-16](30_invariants.md#i-16)) but any update to MPL's shape must update all four lock-step.
- **Whitespace handling.** The C++ compiler strips whitespace before parsing; the conjecturer never emits whitespace; the verifier assumes zero whitespace. A chapter row with inadvertent whitespace (from a hand edit) would confuse the verifier but not the generator's output — asymmetric failure mode.
- **Structural-operator detection is string-level.** `precompileStructuralOperators` finds `!(&` and `!(>` by substring search. A pathological expression where `!(&` appears inside a string literal (there are no string literals in MPL currently, but a future extension) would break the rewrite silently.

### Not exercised by tests

- **Full grammar coverage.** No test suite enumerates every valid MPL shape + every invalid shape with expected error. Adding this would require the formal grammar first.
- **Round-trip parse/serialise.** No test asserts `serialise(parse(s)) == s` for representative expressions. The de-facto invariant holds because the parse is destructive-free, but no regression catches a break.

---

## Open questions

- **OPEN-MPL-1 — RESOLVED (partial normalisation).** Double-negation `!!X` is cancelled at specific prover sites, not globally. Explicit cancellation:
 - [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp) — during `existence2` rewriting: *"`!(existence2[args]) =!!(>[bound](left)!(right)) = (>[bound](left)!(right))`"* — a structural-operator simplification where the `!` on existence cancels with the `!` inside the compiled definition.
 - [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp) — explicit `expanded = expanded.substr(1); // double negation cancels`.
 
 There is **no universal normaliser** that scans an arbitrary expression and collapses every `!!X` to `X`. The conjecturer does not emit `!!` by construction; the verifier does not specifically test for it. A manually-authored external theorem containing raw `!!X` would be passed through unchanged in most paths and would hash distinctly from `X` — potentially preventing hash-matches that should succeed. Hardening: either canonicalise at the load boundary (precompile phase) or forbid `!!` in external theorems.
- **OPEN-MPL-2.** `(&X)` (single-element conjunction) and `(>[])` with no sub-expressions — parser behaviour not tested. Neither is emitted by the conjecturer, and neither appears in any checked-in definition file. Whether the C++ compiler rejects them or silently accepts them has not been investigated for this chapter. Low priority — if the shape never enters the pipeline, behaviour is moot.

---

## See also

- [`02_glossary.md`](02_glossary.md) — every term used in MPL.
- [`04_configs.md`](04_configs.md) — config files that describe MPL-expression metadata.
- [`10_pipeline/01_mpl_definitions.md`](10_pipeline/01_mpl_definitions.md) — how definition files are compiled.
- [`20_core_concepts/06_anchors_and_scopes.md`](20_core_concepts/06_anchors_and_scopes.md) — anchor applications.
- [I-1](30_invariants.md#i-1), [I-8](30_invariants.md#i-8), [I-9](30_invariants.md#i-9), [I-11](30_invariants.md#i-11) — MPL-shape invariants.
-.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
