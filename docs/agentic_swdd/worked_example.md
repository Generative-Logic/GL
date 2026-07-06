<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Worked example — one theorem, all eight stages `[DRAFT]`

> This chapter traces a single Peano theorem from its conjecturer emission to its verifier row. Every intermediate artefact is a byte-accurate quote from a real pipeline run of the current branch. The purpose is pedagogical: reading top-to-bottom gives you a concrete mental model of what `full_run` actually does, and every subsequent chapter in this document becomes less abstract.

---

## The theorem

We follow the simplest theorem in `global_theorem_list.txt`:

```text
(>[i0,s,+,i1](AnchorPeano[N,i0,s,+,*,i1])(>[v1](in2[i0,v1,s])(in3[i0,i1,v1,+])))
```

Reads literally: *"for any instantiation of the Peano anchor, and for any `v1` such that `s(i0) = v1`, we have `i0 + i1 = v1`."*

In natural-number terms: *"0 + 1 = 1"*. The premise `s(i0) = v1` pins `v1 = 1` (since `i1 = s(i0) = 1` by the anchor, and `s` is injective), so the conclusion `i0 + i1 = v1` reads `0 + 1 = 1`.

**Method:** direct (not induction, not mirror, not reformulation). Row 1 of `global_theorem_list.txt` on the current branch:

```text
(>[i0,s,+,i1](AnchorPeano[N,i0,s,+,*,i1])(>[v1](in2[i0,v1,s])(in3[i0,i1,v1,+])))	direct	-1
```

The `-1` in column 3 signals "no induction variable" — direct proofs use `-1` there.

---

## Setup — Definitions & anchor (folded into the prover's startup, not a separate pipeline stage)

The anchor `AnchorPeano` is defined in `files/definitions/AnchorPeano.mpl` as:

```text
(&
	(NaturalNumbers[N,i0,s,+,*])
	(in2[i0,i1,s])
)
```

Reads as: "(N, i0, s, +, *) satisfy the Peano axioms, AND i1 is the successor of i0" — i.e. `i1 = 1`. When our theorem's outer `(AnchorPeano[N,i0,s,+,*,i1])` application unpacks, these two facts become available:

- `(NaturalNumbers[N,i0,s,+,*])` — the Peano structure.
- `(in2[i0,i1,s])` — pins `i1 = s(i0) = 1`.

`NaturalNumbers` in turn expands (from `files/definitions/NaturalNumbers.mpl`) into a long conjunction enumerating every Peano axiom — including the ones we'll need for the proof.

The relevant axiom for our theorem is the **identity rule for `+`**: every natural `a` has `a + 0 = a`, and additionally the **successor rule for `+`**: if `s(b) = c` and `a + b = d`, then `a + c = s(d)`. The combination of these, specialised to `a = 0`, yields `0 + 1 = 1` — which is exactly what we want to prove.

---

## Stage 1 — Conjecturer emits it

The conjecturer enumerates candidates by combining expressions under the anchor. Our theorem appears in `files/theorems/conjectures.txt` in raw bound-variable-index form. **Pre-unification** byte form (sparse anchor binder — no longer the shipped form, see the binder-rule note below):

```text
(>[2,3,4,6](AnchorPeano[1,2,3,4,5,6])(>[7](in2[2,7,3])(in3[2,6,7,4])))
```

Decoded with the anchor's slot map `1 → N, 2 → i0, 3 → s, 4 → +, 5 → *, 6 → i1`:

- Anchor: `(AnchorPeano[1,2,3,4,5,6])` = `(AnchorPeano[N,i0,s,+,*,i1])`.
- Inner bound: `>[7]` — introduces a fresh variable `v1`.
- Body: `(>[v1](in2[i0,v1,s])(in3[i0,i1,v1,+]))` — "for every `v1` such that `s(i0) = v1`, `i0 + i1 = v1`".

**Outer anchor binder — unified rule ([I-4](30_invariants.md#i-4), [I-11](30_invariants.md#i-11) inverted, [D-75](40_decisions.md#d-75)).** The outer `>[...]` binds **every** anchor-atom argument in occurrence order (the anchor atom is the first premise, so occurrence order = anchor-atom order), not only the body-referenced subset. So this theorem's outer binder is the full `>[1,2,3,4,5,6]` (= `N,i0,s,+,*,i1`) — slots `N` and `*` are now bound too. The earlier rationale ("not binding unused slots avoids widening the quantifier universe") was retired: the free/bound status of single-occurrence anchor symbols was never load-bearing for the hash engine, and a theorem is by definition an implication with zero `u_` (unchangeable) parameters, so every variable in it is bound. The byte-accurate post-unification artefacts here will be refreshed from disk after the next full pipeline run.

The conjecturer also produces reshuffled / mirrored variants (canonicalised argument orders + the mirror with the output premise swapped to head). Those go to `files/theorems/reshuffled_conjectures.txt` and `files/theorems/reshuffled_mirrored_conjectures.txt`, and the mirror is **also folded into `conjectures.txt`** so the prover proves it for real (D-112) instead of fabricating a `mirrored statement` row. Our theorem's mirror then appears in the global theorem list as a genuinely-proved theorem (row refreshed from disk after the next full run):

```text
(>[i0,s,+,i1](AnchorPeano[N,i0,s,+,*,i1])(>[v1](in3[i0,i1,v1,+])(in2[i0,v1,s])))	direct	-1
```

— "if `i0 + i1 = v1`, then `s(i0) = v1`". As a directly-proved theorem its reference column is `-1`.

---

## Stage 2 — CE filter (normalisation/reshuffle is a workaround inside the conjecturer's emit logic, not a stage)

Normalisation renames argument positions to canonical order. Since our theorem is already canonical, reshuffling produces essentially the same string.

CE filtering tests the candidate against `files/simple_facts/simple_facts_peano_5.txt` (and `_6.txt`). The candidate reads as "for every `v1` with `s(0) = v1`, `0 + 1 = v1`". The CE filter asks: *can this ever be false against the fact table?*

The fact table contains `(in2[i0,i1,s])` (i.e. `s(0) = 1`) and `(in3[i0,i1,i1,+])` (i.e. `0 + 1 = 1`). Under these, our candidate specialises to `1 = 1` — true. No contradiction, so the candidate survives and is written to `files/theorems/filtered_conjectures.txt`.

---

## Stage 3 — Prover proves it

The prover loads the conjecture as a goal on an LB parented to `body`. The LB's `exprKey` is the theorem. Its `toBeProved` entry is the theorem's head after disintegrating the outer implication layers.

Hash-burst iterations fire. The key inference step is:

The **successor rule for `+`** — part of the `NaturalNumbers[N,i0,s,+,*]` expansion — is a compiled implication whose relevant form is:

```text
(>[s,+,i1](AnchorPeano[N,i0,s,+,*,i1])(>[v1,v2](in2[v1,v2,s])(in3[v1,i1,v2,+])))
```

Reads: "for Peano-anchor context, for any `v1, v2`: if `s(v1) = v2`, then `v1 + i1 = v2`".

Our theorem's premises match this rule with the binding `v1 → i0, v2 → v1` (task variable):

- Rule's `(in2[v1,v2,s])` unifies with our task premise `(in2[i0,v1,s])` under `v1 → i0, v2 → v1`.
- Rule's anchor premise is already satisfied (our theorem's anchor is pinned).

Under this unification, the rule's conclusion `(in3[v1,i1,v2,+])` becomes `(in3[i0,i1,v1,+])` — exactly the head we want.

The prover emits the conclusion via `addStatement`, records origin entries into `exprOriginMap`, and marks the goal as proved. The theorem enters `globalTheoremList` with method `"direct"` and reference `-1`.

---

## Stage 4 — Compressor keeps it

The compressor's Phase 1 builds a per-theorem LB for every proved theorem. For our theorem, the Phase 1 LB finds a one-step derivation (the same step the prover found): the successor rule for `+` applied with `v1 → i0`.

Phase 2 then considers eliminating our theorem. For it to be dead, every surviving theorem must remain derivable without it. Since our theorem is used as an input to other derivations elsewhere (mirrored variants, associativity chains), it stays essential.

No elimination. Theorem survives into the output `theorems.txt`.

---

## Stage 5 — Raw proof graph

The visualiser walks `exprOriginMap` backwards from the theorem's head. The result is a chapter file in `files/raw_proof_graph/` with raw names (prover-internal form: `x`-prefixed anchor copies, numeric-indexed bound variables, no `v1` yet).

The chapter has ≈5 rows, each a proof step leading from the task-formulation premises up to the head.

---

## Stage 6 — Process proof graph — the rename

The processor applies the four-priority renaming scheme. For our theorem:

- **Priority 1.** Anchor mapping from theorem expression: `1 → N, 2 → i0, 3 → s, 4 → +, 5 → *, 6 → i1`.
- **Priority 2.** Left-to-right v-numbering of the theorem's non-anchor args: `7 → v1` (only one non-anchor var here).
- **Priority 2.5.** Anchor-handling `x`-prefix rewrites: the prover's internal `x2` (a copy of slot 2, which is `i0`) renames to `0_copy` via the `i2 → 2_copy` style rule (slot 1 contains `i0`, so `x2` → `0_copy` reflecting the underlying digit value).
- **Priority 3.** No additional non-anchor vars found in chapter lines.
- **Priority 4.** `_copy` suffix derivation: `0_copy` already resolved by Priority 2.5.

Output: `files/processed_proof_graph/0_direct_proof.txt`:

```text
(in3[i0,i1,v1,+])	main	implication	(>[s,+,i1](AnchorPeano[N,i0,s,+,*,i1])(>[v1,v2](in2[v1,v2,s])(in3[v1,i1,v2,+])))	main	(AnchorPeano[N,0_copy,s,+,*,i1])	main	(in2[i0,v1,s])	main
(>[s,+,i1](AnchorPeano[N,i0,s,+,*,i1])(>[v1,v2](in2[v1,v2,s])(in3[v1,i1,v2,+])))	main	theorem
(AnchorPeano[N,0_copy,s,+,*,i1])	main	anchor handling	(AnchorPeano[N,i0,s,+,*,i1])	main
(AnchorPeano[N,i0,s,+,*,i1])	main	task formulation
(in2[i0,v1,s])	main	task formulation
```

Five rows, reading bottom-up (the proof direction):

1. **Task formulations** (rows 4 & 5). Assumed: the Peano anchor holds, and `s(i0) = v1`.
2. **Anchor handling** (row 3). The prover produced a `_copy` of slot `i0` for its internal working; this row records the substitution.
3. **Theorem citation** (row 2). The successor rule for `+` is cited as an available rule.
4. **Implication fire** (row 1). The rule fires with the task premises matching its antecedent; the head `(in3[i0,i1,v1,+])` is emitted.

The theorem's head `(in3[i0,i1,v1,+])` in row 1 is the chapter's "goal reached" — `state.goal_reached` becomes `True` when the verifier processes this chapter.

### Simultaneous rename in `global_theorem_list.txt`

Row 1 of the list (rendered form):

```text
(>[i0,s,+,i1](AnchorPeano[N,i0,s,+,*,i1])(>[v1](in2[i0,v1,s])(in3[i0,i1,v1,+])))	direct	-1
```

Priority 1+2 seeded the chapter's v-numbering from this expression, so the chapter's `v1` matches the theorem's `v1`. This is the [I-10](30_invariants.md#i-10) invariant in action.

---

## Stage 7 — HTML export

The HTML generator renders `files/processed_proof_graph/0_direct_proof.txt` into `files/full_proof_graph/chapter<N>.html`. For each row:

- The `<title>` tag shows the theorem's MPL expression and a human-readable caption (e.g. *"from v1 is a successor of 0, we conclude 0 + 1 = v1"*).
- The proof-line `<div>` for row 1 renders:
 - Step badge: `implication` (clickable to `tags.html#implication`).
 - Content: the derived expression `(in3[i0,i1,v1,+])` rendered with `i` stripped (displays as `(in3[0,1,v1,+])` in the readable form).
 - Rule citation: the successor rule for `+`, clickable — a click expands it into indented MPL via `processText`.
 - Premise citations: the two `main`-scope premises, each clickable back to their task-formulation rows.
- Dep-glow highlighting on hover traces dependencies visually.

The embedded `GL_BINARY_MAP` includes the entries for `AnchorPeano` and `NaturalNumbers` (and all the implications it depends on), letting the user expand any named expression to its compiled form without leaving the page.

---

## Stage 8 — Verifier validates

`verify_chapter` runs on `0_direct_proof.txt`. For each row, the row's tag dispatches to the appropriate checker.

### Row 1 — `implication`

`check_implication` at [`verifier.py`](../verifier.py):

1. Disintegrate the cited rule `(>[s,+,i1](AnchorPeano[…])(>[v1,v2](in2[v1,v2,s])(in3[v1,i1,v2,+])))` into `(chain = [AnchorPeano..., in2[v1,v2,s]], head = (in3[v1,i1,v2,+]))`.
2. Match the cited premises `(AnchorPeano[N,0_copy,s,+,*,i1])` and `(in2[i0,v1,s])` against the chain under some assignment.
3. Under that assignment, verify the head equals the row's LHS `(in3[i0,i1,v1,+])`.

All three hold ⇒ counter for `implication` gets `.record(True)`.

### Row 2 — `theorem`

`check_theorem_tag` at [`verifier.py`](../verifier.py):

- LHS is the successor rule for `+`. Is it in `state.global_theorems`?
- Yes — it is theorem #28 in the list (`(>[s,+,i1](AnchorPeano[…])(>[v1,v2](in2[v1,v2,s])(in3[v1,i1,v2,+])))	direct	-1` or similar entry). 
- **Self-reference guard:** is it the same as the chapter's own theorem `(>[i0,s,+,i1](AnchorPeano[…])(>[v1](in2[i0,v1,s])(in3[i0,i1,v1,+])))`? No — they differ in shape. Self-reference counter not triggered.

✓ counter for `theorem` records success.

### Row 3 — `anchor handling`

`check_anchor_handling` at [`verifier.py`](../verifier.py):

- LHS is an anchor atom `(AnchorPeano[N,0_copy,s,+,*,i1])`.
- RHS is the original anchor `(AnchorPeano[N,i0,s,+,*,i1])`.
- Substitution: slot 2 (originally `i0`) replaced with `0_copy`. Consistent with the anchor's config slot type `(1)`.

Uniqueness: only one `anchor handling` row in this chapter. The uniqueness counter (non-checker) records success.

✓ counter for `anchor handling` records success.

### Rows 4 & 5 — `task formulation`

`check_task_formulation` at [`verifier.py`](../verifier.py):

- Row 4 LHS `(AnchorPeano[N,i0,s,+,*,i1])` — this is the theorem's anchor premise.
- Row 5 LHS `(in2[i0,v1,s])` — disintegrate the theorem and confirm this is in its premise chain.

Both hold ⇒ two successes.

### Theorem goal reached

The verifier's `state.goal_reached` flag is set to `True` at [`verifier.py`](../verifier.py) when the chapter's row 1 LHS matches the theorem's head (after disintegration). `(in3[i0,i1,v1,+])` is the head of our theorem. ✓

---

## The tally — what the verifier prints for our chapter

If this were the only chapter in the run, the final report would look like (format: `<tag>:<W=44>success N, failure M`):

```
theorem goal reached                         success 1, failure 0
implication                                  success 1, failure 0
expansion                                    success 0, failure 0
disintegration                               success 0, failure 0
task formulation                             success 2, failure 0
equality1                                    success 0, failure 0
equality2                                    success 0, failure 0
symmetry of equality                         success 0, failure 0
symmetry of inequality                       success 0, failure 0
recursion                                    success 0, failure 0
theorem                                      success 1, failure 0
reformulation for integration and            success 0, failure 0
...
anchor handling                              success 1, failure 0
...

5 checks, 0 failures — airtight.
```

Five checker rows succeed, all others at zero, no failures — airtight.

---

## What this example does NOT show

Intentionally simple. It does not exercise:

- **Induction.** See chapter 11/12/13 (induction triad) or induction chapters in the current global theorem list for a three-chapter theorem.
- **Disintegration.** See any chapter that expands an `implication<K>` or `and` node and disintegrates its elements (typical of chapters rooted in Peano-axiom unpacking — chapters 12/13 are good examples).
- **Reformulation.** See `103_reformulated_statement.txt`.
- **OR branching.** See any `or_theorem.txt` chapter — requires the OR machinery + branch convergence.
- **Contradiction / vacuous truth.** See `12_check_zero.txt` row 1 (`vacuous truth`) for a live example.
- **Equivalence classes.** See any chapter that applies `equality1` — the substitution reflects the class's membership.

Each of those concepts has its own chapter in [`20_core_concepts/`](20_core_concepts/).

---

## See also

- [`10_pipeline/`](10_pipeline/) — each pipeline stage chapter, each referencing this worked example.
- [`20_core_concepts/08_proof_tags.md`](20_core_concepts/08_proof_tags.md) — per-tag details.
- [I-10](30_invariants.md#i-10), [I-11](30_invariants.md#i-11) — the invariants this example makes concrete.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
