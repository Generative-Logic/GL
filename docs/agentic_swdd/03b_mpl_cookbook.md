<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
Contributions require CLA — see CONTRIBUTING.md.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# MPL cookbook — writing definitions and conjectures `[FULL]`

Read this chapter before writing or editing MPL, including MPL strings inside configuration, source code, tests and documentation. The [language reference](03_mpl.md) describes the notation; this cookbook describes how to turn a mathematical statement into the intended GL input. Inline examples using `P`, `Q` or `R` are schematic predicates, not registered project definitions.

## 1. Split an AND head into separate conjectures

**GL does not use an AND as a conjecture head. A conclusion with two conjuncts becomes two conjectures.** Copy the same premise chain into both rows and put one conclusion at each head. Split recursively if there are more than two conclusions. Do not replace the conjunction with an `and<N>` compact to evade this rule.

Schematic translation: `premises => (P and Q)` becomes `premises => P` and `premises => Q`. The conjunction is a mathematical summary of the two results, not a final integration goal. An equivalence to be proved likewise supplies its two directions as separate conjectures.

The supplied-input path does not do this authoring step for you: [`run_modes.py`](../../run_modes.py), `_prepare_batch_conjectures`, copies each supplied row into the batch input. [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp), `Prover::analyzeExpressions`, registers each conjecture; [`compiler.hpp`](../../GL_Quick_VS/GL_Quick/src/compiler.hpp), `ce::disintegrateImplication`, peels the premise chain and returns its single terminal head. Supply the separate rows explicitly.

### Worked example: D2

For natural numbers, `q*d+r = q'*d+r'`, `r<d`, and `r'<d` imply both `q=q'` and `r=r'`. The original expanded D2 input used these two FTA conjectures:

```text
(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10,11](in3[9,10,11,5])(>[12,13](in3[11,12,13,4])(>[14,15](in3[14,10,15,5])(>[16](in3[15,16,13,4])(>[](strictOrder[1,4,12,10])(>[](strictOrder[1,4,16,10])(=[9,14]))))))))
(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10,11](in3[9,10,11,5])(>[12,13](in3[11,12,13,4])(>[14,15](in3[14,10,15,5])(>[16](in3[15,16,13,4])(>[](strictOrder[1,4,12,10])(>[](strictOrder[1,4,16,10])(=[12,16]))))))))
```

Source: the historical expanded statement in [D2's proof state](../fta_ladder/D2/current_proof_state.md). The current supplied input replaces these two rows with one remainder-predicate uniqueness conjecture. Track the requested results independently; remainder uniqueness alone is not completion of the original two-conclusion D2 statement — the quotient half is row 68 on the one-output `quotient` predicate (see the one-output rule in section 2).

## 2. Fix the meaning and argument map before writing parentheses

Write the mathematical hypotheses and each separate conclusion first. Identify the batch's actual anchor and read its definition, the referenced predicate definitions, and the corresponding `Config<Tag>.json` entries. Record arity, slot order, types, inputs and outputs. Names alone do not establish any of these.

For the D2 rows above:

| Slots | Meaning |
|---|---|
| `1,2,3,4,5,6,7,8` | `N,zero,successor,addition,multiplication,one,two,identity` |
| `9,10,11,12,13` | `q,d,q*d,r,a` |
| `14,15,16` | `q',q'*d,r'` |

The numeric tokens are variable names, not numeral literals. In this anchor, token `2` denotes zero, `6` denotes one, and `7` denotes two. Copying another batch's slot order without checking its anchor changes the statement.

Do not silently strengthen a conjecture by adding a convenient hypothesis. D2's strict remainder bounds already imply a positive divisor over natural numbers; its source rows preserve the stated hypotheses. A predicate that hides a needed witness may be unsuitable: [`remainder.mpl`](../../files/definitions/remainder.mpl) exposes the remainder but quantifies the quotient internally, so two remainder predicates alone do not expose the two quotient variables that D2 must equate. [`quotient.mpl`](../../files/definitions/quotient.mpl) is the twin that exposes the quotient and binds the remainder; quotient uniqueness is supplied on it as row 68.

**An operator has at most one output slot.** `output_args` names the single argument position the admission and integration machinery treats as the operator's result — the marker slot for premise-completion demands and for integration preparation. Nothing is built for a second one. A relation with two results, such as Euclidean division's quotient and remainder, is not one operator with two outputs: write one predicate per result and bind the other result as a witness (`remainder.mpl` exposes `r` and binds `q`; `quotient.mpl` exposes `q` and binds `r`). The seven-argument, two-output `euclideanDivision` attempt of 2026-09-11 added output-slot marker preparations to every block that unfolds `remainder` and crashed the D1 contradiction block ([D-372](40_decisions.md#d-372), `ledger/d2_quotient_uniqueness_ledger.md`).

An existential conclusion can require several properties of the same witnesses. Preserve that shared scope: for E6, [`nontrivialFactor.mpl`](../../files/definitions/nontrivialFactor.mpl) exposes factor `a` and binds factor `b` internally, requiring `a*b=n`, `2<=a<n` and `2<=b<n` together. The supplied theorem existentially quantifies `a`. This preserves the complete two-factor statement with one output slot; splitting the properties into independent existence claims would not preserve their common witnesses. Writing the properties as a raw conjunction inside the supplied existence instead currently crashes the FTA batch's admission pass (open bug, [definition-compilation weaknesses](10_pipeline/01_mpl_definitions.md#known-tracked)); I4 therefore states its factorization through the predicate [`hasPrimeFactorization.mpl`](../../files/definitions/hasPrimeFactorization.mpl), a typed existence over the bundle [`primeFactorSeq.mpl`](../../files/definitions/primeFactorSeq.mpl) (the `hasMinimum` pattern), so the supplied row's head is a single predicate with no existence.

**A new public predicate replaces a spontaneous existence with the same body everywhere.** The compiler mints a numbered `existence<N>` for each inner existence of a definition; registering a public predicate whose body equals one of them makes the public name win in every definition that contains that body ([D-349](40_decisions.md#d-349)). `euclideanDivision` was byte for byte the inner ∃t of `remainder`, so every `remainder` unfolding in every block, D1's included, changed shape. Before adding a predicate that mirrors part of an existing definition, check the compiled registry (`compiledExpressions` in the sacred dump, or the `GL_binary_*.json` files) for the spontaneous entity it would replace.

## 3. Express arithmetic through relations and explicit intermediate values

| Human notation | MPL shape (schematic) |
|---|---|
| `x` belongs to `N` | `(in[x,N])` |
| `f(x)=y` | `(in2[x,y,f])` |
| `a+b=c` | `(in3[a,b,c,+])` |
| `a*b=c` | `(in3[a,b,c,*])` |
| `a=b` | `(=[a,b])` |
| `a<=b` over the additive natural order | `(preorder[N,+,a,b])` |
| `a<b` over the additive natural order | `(strictOrder[N,+,a,b])` |

Supply the configured relation in the last operator slot. `preorder[N,*,a,b]` has a different meaning from `preorder[N,+,a,b]`; inspect the definition rather than reading every `preorder` as numeric order.

Do not insert arithmetic terms such as `q*d+r` or function notation such as `P(x)` into an argument slot. D2 introduces product values `11` and `15`, then uses two addition relations with the same result `13`. This expresses equality of the sums with the intended shared output. Separate result variables would require an explicit equality between them.

## 4. Bind every theorem variable and preserve premise order

Every implication has one bound-variable list, one premise and one body: schematic `(>[x](P[x])(Q[x]))`. Multiple premises form nested implications; an inner `>[]` adds a premise when its variables are already bound outside it.

A theorem binds all its variables, including every referenced anchor slot. Bind a new variable at a scope enclosing all of its uses. Do not reuse a name for an unrelated witness or leave it free accidentally. Definition parameters are different: the declared external parameters remain free in the definition body; locally quantified witnesses are bound. Never copy internal `u_`, `it_`, `int_`, `repl_lev_` or validity-scope names into a new conjecture as user variables.

Keep the selected premise order deliberate. [`proof/proof_grid_builder.hpp`](../../GL_Quick_VS/GL_Quick/src/proof/proof_grid_builder.hpp), `ProofGridBuilder::addTheoremToMemory`, builds one child logic block for each premise in that order. The final premise's block owns the goal; ancestor premises reach it through the existing mail rules. Reordering logically equivalent hypotheses can change the proof search and the target's full parent chain.

### Keep premise chains short; a long chain signals a missing definition

**No supplied row should approach the batch's premise-permutation bound (`sizeAllPermutationsAna`, FTA 8).** [`inference/hash_rule_installer.cpp`](../../GL_Quick_VS/GL_Quick/src/inference/hash_rule_installer.cpp), `HashRuleInstaller::addToHashMemory`, installs no head record for a rule copy with more premises than that table holds, so such a theorem can never fire as a rule. The first G2 draft (2026-10-06) supplied two helper rows with nine arithmetic premises each; the full run crashed in the FTA batch two bursts after proving them, on the `premisePatternsOfCopy` assertion "a premise of an installed copy was never encoded" (cause not trap-confirmed). A run of premises that names every intermediate product and sum of a definition's body means that a definition is missing or unused: fold the repeated pattern into a predicate. G2's helpers spelled out both orientations of the Bezout equation; the predicate `bezEq` reduced each row to five premises and states `bez1`'s cases, so the cases and the helpers share one form.

### Neutralize an introduced index in another expression

**A quantifier list alone is not enough. An introduced index must be constrained (neutralized) by another expression.** Writing `>[i,p]` before `(in2[i,p,P])` syntactically binds the names, but does not satisfy this authoring rule for the index. Do not confuse absence of free variables, configured port types, or typing that could later be deduced from a sequence with an explicit constraining expression.

For the premise "p belongs to sequence P", use `(inSequence[N,p,P])` and keep the index inside its typed existence. The definition in [`inSequence.mpl`](../../files/definitions/inSequence.mpl) is:

```text
!(>[i]
    (in[i,N])
    !(in2[i,p,P])
)
```

Here `in[i,N]` constrains the index used by `in2[i,p,P]`. The existential binder keeps `i` local to the definition; its occurrence in the separate `in[i,N]` expression supplies the required constraint. `N`, `p` and `P` are the external parameters. If the index itself is part of the intended theorem, expose it only together with another expression constraining it.

**F8 example, symbolic schema.** P is a sequence on `[i0,n]`, q is its Fold product, and p belongs to P; therefore p divides q. This is the symbolic form of supplied row 106, with the index absent from the theorem's parameters:

```mpl-schema
(>[N,i0,s,+,*,i1,i2,id]
    (AnchorFTA[N,i0,s,+,*,i1,i2,id])
    (>[n,P]
        (sequence[N,+,i0,n,P])
        (>[q]
            (fold[N,s,+,*,P,i0,n,q])
            (>[p]
                (inSequence[N,p,P])
                (preorder[N,*,p,q])
            )
        )
    )
)
```

`preorder[N,*,p,q]` means divisibility because its operation is multiplication. The Fold premise represents an existing product; this statement does not assert general product existence.

Check types from predicate ports, not from spelling or a successful text parse. If a free mathematical variable has no typing supplied by the chosen relations or context, express the required typing hypothesis. Do not add redundant typing premises without checking the intended statement and the conjecture conventions.

## 5. Use binary conjunctions inside definitions and preserve negation scope

**Prefer existence statements over implications in definitions when both are equivalent formulations of the intended relation.** Choose the witness-based formulation when it expresses the same mathematical contract; preserve its typing, polarity and quantifier scope. This preference never permits replacing a universal claim with a weaker existence claim, adding hypotheses, or changing the defined relation.

Definitions can contain conjunctions. Each `&` node has exactly two children; represent three conjuncts by nesting, for example schematic `(&(P[x])(&(Q[x])(R[x])))`. Both [`compiler.cpp`](../../GL_Quick_VS/GL_Quick/src/compiler.cpp), `ce::parseExpr`, and [`expression_utils.py`](../../expression_utils.py), `parse_expr`, parse a conjunction as a left and a right child. Do not write a flat three-child node or an empty or unary conjunction.

Negation is a prefix: schematic `!(P[x])`, not `(!(P[x]))`. The schematic OR `P or Q` can be represented as `!(&!(P[x])!(Q[x]))`; negating a conjunction of positive `P` and `Q` means `not P or not Q`, a different statement. Keep every leaf's polarity exact. Cancel an intended double negation in the authored expression rather than assuming a global normalizer will do it.

A typed universal has schematic form `(>[x](in[x,N])(P[x]))`. A typed existential has schematic form `!(>[x](in[x,N])!(P[x]))`. With several existential variables, preserve the entire quantified scope under the outer negation. Do not put an existential witness into the theorem's outer universal binder, which changes existence into a claim about every value.

**Compiled existences use one referenced witness per implication binder.** Nest single-variable bindings under the same outer negation when an existence has several witnesses; keep their common body and original quantifier scope. A grouped outer binder such as schematic `!(>[x,y](P[x,y])!(Q[x,y]))` reaches the single-witness assertion in [`compiler/expression_parser.cpp`](../../GL_Quick_VS/GL_Quick/src/compiler/expression_parser.cpp), `ExpressionParser::renameLastRemoved`. This is a compilation constraint on existences; ordinary universally quantified implications may bind several variables.

**Known compiler bug (open): a universal with a nested premise inside a supplied row's existence or conjunction crashes the compile.** The form is valid MPL and is to be supported (maintainer, 2026-10-18); the fix is deferred. Written directly in a conjecture, schematic `!(>[p](P[p])!(&(Q[p])(>[d](R[d])(>[](S[d])(T[p,d])))))` makes the compiler negate the universal: its witness `d` is renamed, and the toggled remainder `!(>[](S[d])(T[p,d]))` reaches the existence branch of [`compiler/structural_compiler.cpp`](../../GL_Quick_VS/GL_Quick/src/compiler/structural_compiler.cpp), `StructuralCompiler::compileExpression`, with an empty binder, so the single-witness assertion in `ExpressionParser::renameLastRemoved` fires. The FTA2 I2 draft of 2026-10-18 crashed this way (`ledger/fta_i2_ledger.md`; tracked in the [definition-compilation weaknesses](10_pipeline/01_mpl_definitions.md#known-tracked)). Definitions with the same nesting (`prime`, `ordered`) compile. Until the fix lands, keep such a universal inside a predicate's definition, as I1 does through `hasMinimum` and `minimum`, and prefer existing predicates before adding one.

Use the source-level definition or configured public predicate. Numbered `or<N>`, `existence<N>` and `implication<N>` names belong to the current compiler registry; never guess or hard-code a newly numbered predicate when authoring fresh MPL. Decode an existing artifact against its matching saved registry when investigating it.

### Set-definition implications can produce existence statements

**Do not demand an additional existence conjecture before checking GL's set-definition reformulation.** A proved implication whose premise defines a set or relation can supply a witness together with the proved property. This is an existing construction mechanism, used by the main Gauss lemmas; it is not an inference from an arbitrary implication.

The schematic transformation is `H => (D(X) => Q(X))` into `H => exists X: D(X) and Q(X)`. Its MPL existential tail is `!(>[X](D[X])!(Q[X]))`. The definition and property share the same witness. Here `H` denotes the remaining premise chain, and `D` and `Q` are explanatory placeholders.

The exact gates are in [`compiler.hpp`](../../GL_Quick_VS/GL_Quick/src/compiler.hpp), `ArgumentAnalyzer::isSetDefinition`, and [`proof/theorem_publisher.hpp`](../../GL_Quick_VS/GL_Quick/src/proof/theorem_publisher.hpp), `TheoremPublisher::reformulateTheorem`:

- The definition has two converse implications, `membership => body` and `body => membership`, joined by binary conjunction. Membership may be `in`, `in2`, or `in3`; its set or relation argument has a powerset type. The defining body must not contain that argument. `compileCoreExpressionMap` records the recognized argument in `LogicalEntity::definedSet`.
- Reformulation requires disintegration to be enabled. It moves the candidate definition to the end of the premise chain while preserving the other premises' relative order.
- The defined object must be the sole candidate argument occurring at least twice across premises and head while absent from every other premise. The existential binder contains exactly that object; the remaining premises retain their bindings.
- The definition premise must bind the defined object alone. A variable that only the definition premise mentions, such as an appended value bound together with the defined sequence in `(>[a,Q](appendToSequence[n,a,P,Q])…)`, would lose its binder in the generated statement, so no reformulation is formed ([I-4](30_invariants.md#i-4), [D-462](40_decisions.md#d-462)). The implication itself is still proved and broadcast.
- The emitted existence keeps both the definition and the original head, including the head's polarity. Its provenance cites the proved implication. Check the resulting `reformulated statement` and its proof; merely writing a definition does not certify a conjecture.

**Gauss example: restrict an interval.** These are verbatim source and result expressions from the processed theorem list, with the result decoded using the matching Gauss registry. The source proves that restricting `[0,s(v1)]` to values at most `v1` gives `[0,v1]`:

```text
(>[N,i0,s,+,*,i1,i2,id](AnchorGauss[N,i0,s,+,*,i1,i2,id])(>[v1,v2](in2[v1,v2,s])(>[V1,V2](limitSet[N,+,V1,v1,V2])(>[](interval[N,+,i0,v2,V1])(interval[N,+,i0,v1,V2])))))
(>[N,i0,s,+,*,i1,i2,id](AnchorGauss[N,i0,s,+,*,i1,i2,id])(>[v1,v2](in2[v1,v2,s])(>[V1](interval[N,+,i0,v2,V1])(existence8[N,+,V1,v1,i0]))))
```

In that registry, `existence8[N,+,V1,v1,i0]` means `exists V2: limitSet(N,+,V1,v1,V2) and interval(N,+,i0,v1,V2)`. The restricted interval is constructed by reformulation of the proved implication; no separate supplied existence conjecture is needed for this transformation.

**Gauss example: restrict a sequence.** The relation-valued definition follows the same mechanism:

```text
(>[N,i0,s,+,*,i1,i2,id](AnchorGauss[N,i0,s,+,*,i1,i2,id])(>[v1,v2](in2[v1,v2,s])(>[V1,V2](limitSequence[N,+,v1,V1,V2])(>[](sequence[N,+,i0,v2,V1])(sequence[N,+,i0,v1,V2])))))
(>[N,i0,s,+,*,i1,i2,id](AnchorGauss[N,i0,s,+,*,i1,i2,id])(>[v1,v2](in2[v1,v2,s])(>[V1](sequence[N,+,i0,v2,V1])(existence15[N,+,v1,V1,i0]))))
```

Here `existence15[N,+,v1,V1,i0]` means `exists V2: limitSequence(N,+,v1,V1,V2) and sequence(N,+,i0,v1,V2)`. Both definitions record their fifth argument as `definedSet`. The examples are evidenced by [`global_theorem_list.txt`](../../files/processed_proof_graph/global_theorem_list.txt), its cited proof chapters, and [`GL_binary_Gauss.json`](../../files/GL_binaries/GL_binary_Gauss.json). Numbered compact names are snapshot-specific: author fresh conjectures with the public definitions, never these numbers.

For singleton construction, a membership characterization of a singleton set and an `in2` characterization of a singleton binary relation can qualify in the same way. Prove the corresponding interval or sequence implication, then verify its generated existence result. Do not add separate witness-existence goals solely because the supplied row is an implication.

## 6. Write the durable input, then refresh the correct pipeline state

For supplied conjectures, read `external_conjectures_file` in the batch configuration. FTA currently uses [`files/shortcut/theorems/conjectures.txt`](../../files/shortcut/theorems/conjectures.txt). Keep one complete expression per line with no comments or internal whitespace. Definition `.mpl` files may be indented; the definition compiler strips that formatting.

Do not add an unproved goal to `externally_provided_theorems.txt` or to a proved-theorem artifact. Do not edit generated `replay_conjectures.txt` as the durable input. [`fta_shortcut.py`](../../fta_shortcut.py), `source_fingerprints` and `restore`, require a new full GPU capture after changing definitions, configurations or supplied conjectures. Run `main.py --GPU` to refresh the capture, then `main.py --shortcut --GPU` for replay. Preserve the fingerprint assertion and use the same source inputs for both.

## 7. Validate the expression separately from its proof

1. Read it back into ordinary mathematics using the slot table. Check each hypothesis, polarity, shared output and separate head.
2. Parse with the existing `expression_utils.parse_expr`; for a complete theorem, inspect `arguments` for unintended free variables. Check delimiter balance, row uniqueness and exact agreement between the documented statement and durable input. This is a syntax and binding check, not proof or complete type validation.
3. Let the ordinary compiler and pipeline validate the configured predicates and types. Use the full rebuild, GPU flags, dedicated logs and completion watcher required by the operational rules.
4. Inspect each exact theorem result and its proof evidence. A zero-failure verifier report validates exported proofs; it does not mean every supplied conjecture was proved. The shortcut omits connected Lean certification; that evidence comes from the full run.
5. If an obligation remains unproved, use its human proof and live trace to find the first missing step. Do not change its hypotheses or conclusion to make the result appear successful.

## Pitfalls

- A mathematical conjunction in the summary is not permission to author an AND head: apply rule 1 first.
- Shared numeric tokens represent shared values. An accidental reuse changes the theorem; an accidental fresh name removes an intended equality.
- A syntactically correct existential can still bind the witness at the wrong scope.
- A compiled name from a previous run can denote a different definition in the current registry.
- An identical-looking premise can occur in several logic blocks. Debug targets require the full parent chain to the empty root sentinel.
- A theorem list or proof count alone cannot establish that both halves of a split statement succeeded.
- `output_args` with two entries is not a two-result operator; it is an unsupported shape (one output slot maximum, section 2).

## See also

- [MPL language reference](03_mpl.md) and [configuration reference](04_configs.md).
- [Definition compilation](10_pipeline/01_mpl_definitions.md), [conjecture preparation](10_pipeline/02_conjecturer.md), and [prover execution](10_pipeline/04_prover.md).
- [D2's proof state](../fta_ladder/D2/current_proof_state.md) for the running example and its measured result.
- [I-4](30_invariants.md#i-4), [I-8](30_invariants.md#i-8), [I-11](30_invariants.md#i-11) and [I-29](30_invariants.md#i-29) for binding, nontrivial heads and type consistency.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
