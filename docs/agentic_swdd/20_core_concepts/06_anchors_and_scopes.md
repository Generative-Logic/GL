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

The prover-side function `ExpressionAnalyzer::prehandleAnchor` ([`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp)) walks every memory block once at the top of `analyzeExpressions` and, for each non-anchor LB whose ancestry reaches an `Anchor*` LB, computes the axed-anchor expression `(AnchorXxx[…, x<i>, …])` for slots whose `definitionSets[i] == "(1)"` and whose original arg appears in the chain of LB exprKeys leading down from the anchor LB. Two pieces of state are written per touched LB:

1. **Statement** — `replacedAnchor` is appended to `intEncodedStatements`, `intKnownStatements` (both membership bits), `intLocalEncodedStatements{,Delta}` + `intLocalEncodedStatementsSet`, and `intStatementLevelsMap` so the axed form is treated as a derivable fact at this LB.
2. **History line** — a single `addOriginEncoded(mb->exprOriginMap, …, replacedAnchor, ("anchor handling", [originalAnchor]))`. There is no separate `mailOut` origin write — the entry reaches descendant LBs through the normal mail merge (see [Recursion-LB history line](#recursion-lb-history-line)).

#### Recursion-LB history line

`prehandleAnchor` early-returns on LBs flagged `isPartOfRecursion` (induction-hypothesis stubs created by `prover.cpp`). The early return is **correct** for the statement registration — recursion LBs must not carry axed variables, and the axed-anchor expression must not become a derivable statement on a recursion stub — but the **history line MUST still travel** to the recursion LB anyway. Reason: the hash engine binds implication templates against the axed form during ordinary inference; an implication firing inside a recursion LB can therefore cite `(AnchorXxx[…, x<i>, …])` as an antecedent even though the recursion LB never registered it as a statement. Without an `exprOriginMap` entry for that ingredient, `buildStack` ([`visualizer.cpp`](../../GL_Quick_VS/GL_Quick/src/visualizer.cpp)) finds no origin and asserts at line 88.

Mechanism: `prehandleAnchor` writes the history line to `mb->exprOriginMap` on the non-recursion LB where the axed anchor is generated — a single write, with no separate `mailOut` origin write. The pull-model mail merge then carries the entry to descendant LBs: the producing LB's origin entries reach the LBs that pull from it, where they merge into the local `exprOriginMap` (see [`03_mail_system.md`](03_mail_system.md)). The recursion-LB descendant ends up with the origin entry — but no statement, no `intAxedVariables` insert — exactly the state buildStack needs to resolve cross-scope citations.

This propagation pattern was added by commit after a buildStack assert at validity=main exprKey=`(in2[rec0,7,3])` could not find origin for `(AnchorPeano[1,x2,3,4,5,6])`. Pre-fix, dropping certain double-digit baseline shapes from a Peano (1)-cap config punctured the cross-scope chain because the axed anchor never reached recursion LBs.

---

## Anchor rewriting — `makeAnchorSignature` and `findAnchorKey`

Two compiler-side helpers:

### `findAnchorKey`

Defined at [`compiler.hpp`](../../GL_Quick_VS/GL_Quick/src/compiler.hpp). Scans the `coreExpressionMap` for the first key beginning with `"Anchor"`. Returns its name. This is how `run_modes::fullRun` knows the name of the anchor for the current batch without requiring it to be passed explicitly.

### `makeAnchorSignature`

Defined at [`compiler.hpp`](../../GL_Quick_VS/GL_Quick/src/compiler.hpp). Given the anchor's name and arity, constructs its canonical signature string: `"(AnchorName[1,2,...,arity])"`. Used when the prover needs to construct an anchor-shaped expression for matching.

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
