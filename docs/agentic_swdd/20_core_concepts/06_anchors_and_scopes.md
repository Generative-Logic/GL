<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Core concept — Anchors and anchor-handling `[DRAFT]`

> Every theorem in a GL batch starts with one anchor atom that pins the axiomatic context. The anchor supplies the fixed names (`N`, `i0`, `s`, `+`, `*`, `i1`, …) that subsequent arguments refer to. Anchor-handling is the prover step that resolves raw bound-variable indices (`1`, `2`, …) to these anchor-slot names; the proof graph records it as the `anchor handling` tag with a per-step substitution trace.

---

## What an anchor does

An anchor is a single atomic expression that the conjecturer places at the outermost position of every theorem:

```text
(AnchorPeano[N,i0,s,+,*,i1])        — 6 slots
(AnchorGauss[N,i0,s,+,*,i1,i2,id])  — 8 slots
(AnchorIncubator3[N,i0,s,+,*,i1,i2,id,i3]) — 9 slots
(AnchorIncubator8[N,i0,s,+,*,i1,i2,id,i3,i4,i5,i6,i7,i8]) — 14 slots
(AnchorFTA[...])                     — (TBD per current FTA ladder)
```

Every anchor's body is a small MPL file in `files/definitions/Anchor*.mpl`. The body unpacks the anchor's claim about its arguments. For `AnchorPeano.mpl`:

```text
(&
    (NaturalNumbers[N,i0,s,+,*])
    (in2[i0,i1,s])
)
```

The anchor, in other words, claims that "(N, i0, s, +, *) satisfy the Peano axioms, and i1 is the successor of i0". Any theorem whose body follows the outer `(AnchorPeano[...])` application can freely use `N`, `i0`, etc. as fixed symbols.

---

## Free vs bound anchor slots

An anchor slot can appear in a theorem's outer bound-variable list (the `>[...]` at the very outside) or not:

```text
(>[i0,s,+,i1](AnchorPeano[N,i0,s,+,*,i1])(>[v1](in2[i0,v1,s])(in3[i0,i1,v1,+])))
```

Here `i0`, `s`, `+`, `i1` are bound; `N`, `*` are free. This means: "for any i0, s, +, i1 satisfying the Peano axioms (with N and * left as given global symbols), …". In practice the conjecturer binds only those anchor slots that are actually *used* somewhere in the theorem body; slots unused in the body stay free.

**Invariant [I-11](../30_invariants.md#i-11):** anchor-slot names must never appear in *inner* `>[...]` lists. Binding them inner-level would turn a theorem about Peano arithmetic into a theorem about arbitrary structures — silently changing the meaning.

---

## Anchor-slot typing and the config

Each anchor's slots have types fixed by the config (`Config<Tag>.json`):

- `N` — type `P(1)` (a set over the base domain).
- `i0`, `i1`, `i2`, `i3`, … — type `(1)` (elements of `N`).
- `s` — type `P(x(1)(1))` (a relation over pairs, used as a function).
- `+`, `*` — type `P(x(1)(x(1)(1)))` (a relation over triples, used as a binary function).
- `id` — type `P(x(1)(1))` (the identity function; Gauss anchor).

The conjecturer uses these types to decide which slots can participate in which expressions. An expression like `(in3[a,b,c,+])` wants a `P(x(1)(x(1)(1)))` in position 4 — so the conjecturer tries `+` and `*` (the two `P(x(1)(x(1)(1)))` slots on Peano), not `N` or `i0`.

---

## The `_copy` machinery

When the prover works with an anchor slot, it often needs a "fresh copy" of the slot variable — distinct in identity but constrained to be equal to the original. The prover mints `x`-prefixed names for these copies during its internal execution (e.g. raw `x3` might be a fresh copy of slot 3).

The renaming stage (stage 6, `process_proof_graphs.py`) then rewrites `x<N>` names to `<digit>_copy` where `<digit>` is the anchor-slot index that the `x` was a copy of:

```text
raw: (AnchorPeano[N,x1,x2,x3,x4,x5])
renamed: (AnchorPeano[N,0_copy,s,+,*,1_copy])      (example — actual slot pinning depends on chapter)
```

Priority 2.5 of the renaming scheme (see [`10_pipeline/06_process_proof_graph.md`](../10_pipeline/06_process_proof_graph.md)) handles this. Priority 4 then propagates the `_copy` suffix through any raw variable whose base was renamed.

---

## The `anchor handling` tag

Each chapter has at most one `anchor handling` row. Example from `0_direct_proof.txt`:

```text
(AnchorPeano[N,0_copy,s,+,*,i1])	main	anchor handling	(AnchorPeano[N,i0,s,+,*,i1])	main
```

Read: the expression on the left (`AnchorPeano[N,0_copy,s,+,*,i1]`) is the anchor-handling result — `0_copy` has been substituted into slot 1 (i0's slot). The right-hand citation is the original anchor expression.

The verifier's `check_anchor_handling` ([`verifier.py`](../../verifier.py)) validates:

- The expression is an anchor atom.
- The substitution is consistent with the anchor's config slot types.
- Only one `anchor handling` row per chapter.

Following the `anchor handling` row are zero or more `anchor handling trace` rows — each recording one step of the `_copy` substitution chain. The verifier counts these but does not validate them (non-checker category).

---

## Anchor handling in code

Resolving raw bound-variable indices to anchor-slot names happens in stage 6 (the processor), not in the prover. The prover emits raw-index chapters via `visualizer.cpp`'s `buildStack`/`generateRawProofGraph` ([`visualizer.cpp`](../../GL_Quick_VS/GL_Quick/src/visualizer.cpp), [`visualizer.cpp`](../../GL_Quick_VS/GL_Quick/src/visualizer.cpp)), and the processor applies Priority 1 (anchor-slot mapping from the theorem expression) as the first renaming pass.

This means: the `anchor handling` tag row exists as an explicit row in the raw proof graph too — stage 6 just renames its arguments. The row is emitted by the prover when the anchor-handling step occurs during proof.

### `prehandleAnchor` — runtime axed-anchor registration

The prover-side function `ExpressionAnalyzer::prehandleAnchor` ([`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp)) walks every memory block once per grid build (called from the `buildGrid` lambda inside `analyzeExpressions`, after every LB — induction sub-LBs included — exists) and, for each non-anchor LB whose ancestry reaches an `Anchor*` LB, computes the axed-anchor expression `(AnchorXxx[…, x<i>, …])` for slots whose `definitionSets[i] == "(1)"` and whose original arg appears in the chain of LB exprKeys leading down from the anchor LB (the `traceVariables` guard — the chain INCLUDES the LB's own exprKey). State written per touched LB:

1. **Axed names (every LB)** — each substituted slot's x-name is minted into the LB's `intAxedVariables`, arming the deposit filter (below).
2. **Statement (outside recursion subtrees only)** — `replacedAnchor` is appended to `intEncodedStatements`, `intKnownStatements` (both membership bits), `intLocalEncodedStatements{,Delta}` + `intLocalEncodedStatementsSet`, and `intStatementLevelsMap` (the `{-1}` non-derived tier) so the axed form is a live fact at this LB. The write is direct — deliberately bypassing the `addExprToMemoryBlock` door, which would refuse the x-citing arguments.
3. **History line (with the statement)** — a single `addOriginEncoded(mb->exprOriginMap, …, replacedAnchor, ("anchor handling", [originalAnchor]))`, written locally on the same LB.

#### Set-only containment in recursion subtrees, and the anchor-numeral exception

Recursion subtrees take one of two modes, decided at the subtree root and threaded down the walk (`axedMode` parameter — only subtree roots carry `isPartOfRecursion`; [D-272](../40_decisions.md), [I-188](../30_invariants.md#i-188)):

- **Set-only (the default):** the subtree mints the axed set but **never registers the axed-anchor statement or history line**. Because each LB's trace accumulates over its whole ancestor chain, a descendant's axed set is always a **superset** of its ancestors' — the parent's axed-anchor statement arriving by statement mail is refused at the armed door, and the induction sub-blocks stay entirely x-free: no x-citing statement, rule, or firing exists there at all. Block-#2 `_induction_` side-chain roots always take this mode (their synthetic `(=[digitArg,zero])` goal cites the zero numeral by construction).
- **Anchor-numeral exception:** a block-#1 root whose goal — the theorem head, queued verbatim at recursion-block creation — cites a `(1)`-typed anchor slot value as a whole token leaves its subtree **entirely untouched** (no mint, no registration): the ancestor's mailed axed-anchor statement registers at the inert door and the historical premise-completion machinery runs inside that grid's induction blocks. Anchor-numeral theorems need it — the unit-product row un-proves under pure set-only even though its chapters cite zero x-content (the machinery is search-dynamics scaffolding, which is also why a statement-with-armed-door middle ground is near-inert: every x-anchor firing's head cites an x-slot and dies at the armed door).

Historically the walk early-returned on `isPartOfRecursion` LBs ABOVE both the mint and the child recursion, leaving induction subtrees with permanently empty axed sets. The intended contract was "no statement on recursion stubs, only the mailed history line travels" — but the parent's axed-anchor statement rode `intLocalEncodedStatementsDelta` → `fillMailOut` → statement mail into the stub anyway, registered unchecked at the inert door, and ordinary anchor rules bred derived x-facts and x-instantiated rule twins from it (the x-copy leak: 553/7 700 statements and 282/1 884 rules citing `x6` in B5's rec2 sub-LB, single external mail seed + pure internal breeding per the 2026-08-11 trap run — see `docs/fta_ladder/B_RT_investigation.md`). Set-only containment enforces the original contract's intent at the door instead of assuming it.

#### The axed-variable deposit filter and its positive-anchor exception

The x-copy names minted into `intAxedVariables` are contained at THREE set-keyed checks ( closed the two bypass gaps): the `addExprToMemoryBlock` prologue drops any deposit carrying an axed name in an argument slot before registration; the `addStatement` door repeats the check for registrations that never pass the prologue (disintegration products, equalities); and the `applyEquivalenceClass` orbit commit refuses a rewritten product citing an axed name (the door-bypassing path). The copies therefore serve as anchor-premise completions without breeding derived x-facts; all three checks are inert where the set is empty (the mode-2 anchor-numeral subtrees, by design). One flag-gated exception (`D-234`, `parameters.axed_anchor_exception`, default OFF — no shipped config enables it): a POSITIVE anchor-category expression (`isAnchor` by name prefix, not negated) deposits normally. This is what extends the copied-anchor mechanism to EXTERNAL anchors: an anchor-bridge rule (batch anchor ⟹ external anchor) firing on the x-copied batch anchor derives the external anchor's x-form — e.g. `(AnchorPeano[1,x2,3,4,5,x6])` from `(AnchorIncubator3[1,x2,3,4,5,x6,x7,8,9])` in the incubator batches — as a live statement, so external-anchor-premise rules can bind element arguments at anchor slot values just as batch-anchor-premise rules do. Without it an instance binding an element to an anchor value cannot match the rule's normalized request key (the slots are positional variables of one shared key — the element-vs-anchor-slot collision, third form of the I-36 family). Negated anchor forms stay filtered even when enabled: a derived negated x-anchor at `main` would pair against the positive form as a spurious premise inconsistency under the vacuous-premise suppression. The flag defaults off because the live x-anchor opens the external anchor's whole rule universe at x-arguments — an x-ed anchor triggering another anchor — which exploded the IncubatorGauss3 runtime in the rung-2.1 acceptance runs.

---

## Anchor rewriting — `makeAnchorSignature` and `findAnchorKey`

Two compiler-side helpers:

### `findAnchorKey`

Defined at [`compiler.hpp`](../../GL_Quick_VS/GL_Quick/src/compiler.hpp). Scans the `coreExpressionMap` for the first key beginning with `"Anchor"`. Returns its name. This is how `run_modes::fullRun` knows the name of the anchor for the current batch without requiring it to be passed explicitly.

### `makeAnchorSignature`

Defined at [`compiler.hpp`](../../GL_Quick_VS/GL_Quick/src/compiler.hpp). Given the anchor's name and arity, constructs its canonical signature string: `"(AnchorName[1,2,...,arity])"`. Used when the prover needs to construct an anchor-shaped expression for matching.

## Anchors in derivative admission keys

`ExpressionAnalyzer::makeNormalizedKeysForAdmission` in [`memory.cpp`](../../GL_Quick_VS/GL_Quick/src/memory.cpp) extracts derivative rules that can fire when one non-anchor premise is missing. If the actual rule contains an `Anchor*` premise, the anchor remains in every derivative key produced by the regular, `ordisOnly`, and `ordis2Demand` routes. It is context, never the missing premise. The regular route retains it even when it contains the selected premise's output argument; the route's existing non-anchor eligibility count remains unchanged.

No special anchor normalization exists. The ordinary whole-key normalization already preserves equality classes. Without an anchor, `(preorder[1,4,6,9])` and `(preorder[1,4,2,9])` normalize to the same one-expression shape. With `(AnchorFTA[1,2,3,4,5,6,7,8])` retained, the former ties the third `preorder` argument to anchor slot 6 and the latter ties it to slot 2, so their normalized keys differ. Renaming all variables consistently still produces the same key. See [I-193](../30_invariants.md#i-193) and [D-281](../40_decisions.md#d-281).

---

## Scope vs anchor — not the same thing

It's worth drawing a sharp line:

- **Anchor** — a *content* commitment. It is one of the theorem's premises (conjoined with the rest of the premises and the head). Different theorems within the same batch use the same anchor.
- **Scope (validityName)** — a *provenance* marker. It identifies *where* a statement is asserted — at root, under a hypothesis, in an OR branch, etc. Independent of the anchor.

A theorem under `AnchorGauss` is still proved at scope `"main"` by default — the anchor does not open a scope. Scopes open for hypothetical reasoning (see [`20_core_concepts/04_validity_stack.md`](04_validity_stack.md)); anchors fix axiomatic content.

---

## Cross-anchor theorems

Some theorems reference two anchors. In the current codebase, this appears as:

```text
(>[N,i0,s,+,*,i1](AnchorGauss[N,i0,s,+,*,i1,i2,id])(AnchorPeano[N,i0,s,+,*,i1]))    direct    -1
```

— the trivial claim that AnchorGauss implies AnchorPeano (via the shared first-6 slots). These serve as the bridge between batches. Peano theorems proved in a prior batch can be cited against a Gauss goal via this bridge.

---

## Weaknesses

### Known & tracked

- **Anchor-slot count is implicit across files.** AnchorPeano has 6 slots; AnchorGauss has 8; AnchorIncubator has 14. These counts are hardcoded in conjecturer filters, verifier logic, rename priorities, and several other places without a single source of truth. Adding a new anchor (e.g. AnchorFTA) requires coordinating all these sites manually.
- **[I-11](../30_invariants.md#i-11) enforcement is structural, not asserted.** The invariant "anchor vars not in inner `>[...]`" holds by construction in the current conjecturer. A regression in conjecturer filters could silently emit violating theorems.

### Suspected fragility

- **Substring-match anchor detection.** Verifier, stage 6, and several prover sites detect "which anchor is in use" by substring-matching `"Anchor"` in the theorem expression. A theorem that happened to contain a literal `"Anchor"` substring in some non-anchor position would confuse detection. Unlikely in the current definition set but not excluded.
- **`_copy` + `_copy` composition.** Can a raw variable have `_copy_copy`? If so, the rename priorities' suffix stripping (`[:-5]`) would handle only one layer. Not observed; not tested.
- **Anchor-slot typing table is not validated at load — partially addressed.** Originally: if `AnchorPeano.mpl` ever defined types inconsistent with `ConfigPeano.json`'s `definition_sets`, the mismatch would not be caught at load. As of [D-41](../40_decisions.md#d-41) the verifier-side `definition set consistency` meta-check ([`08_verifier.md`](../10_pipeline/08_verifier.md#definition-set-consistency-meta-check)) catches the in-flight artefact half: any chapter row whose variable connections respect the configured `definition_sets` will pass; any row whose variable types disagree across operator-call positions fails. The load-time half (config↔MPL-definition cross-validation at config-load time) remains open — but the in-flight check makes the runtime contract observable.

### Not exercised by tests

- **Cross-anchor theorem handling.** The AnchorGauss → AnchorPeano bridge theorem exists. Tests for chapter rendering, verifier checks on it — assumed correct but not targeted.
- **Per-anchor `rename_expr_*` symmetry.** HTML-export's `rename_expr_peano` and `rename_expr_gauss` ([`generate_full_proof_graph.py–753`](../../generate_full_proof_graph.py)) are separate functions. A divergence in conventions (e.g. Peano renames `i0` to `0` but Gauss to `zero`) would silently produce inconsistent HTML output.

---

## See also

- [`10_pipeline/01_mpl_definitions.md`](../10_pipeline/01_mpl_definitions.md) — anchor files + config.
- [`10_pipeline/06_process_proof_graph.md`](../10_pipeline/06_process_proof_graph.md) — renaming priorities, including anchor handling.
- [`20_core_concepts/04_validity_stack.md`](04_validity_stack.md) — scopes as distinct from anchors.
- [`20_core_concepts/08_proof_tags.md`](08_proof_tags.md) — `anchor handling` tag.
- [I-11](../30_invariants.md#i-11).
- — step-by-step anchor expansion recipe.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
