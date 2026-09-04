<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
Contributions require CLA — see CONTRIBUTING.md.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Pre-split merge (or elimination) — architecture plan

> **Status:** [IN CONSTRUCTION] — maintainer-approved design; prover-side seam + chapter writer + unit tests landed (`constructOrEliminationInRun`); Python pipeline, verifier checker, Lean export, and the C13 campaign rows follow per the commit plan below. First target: full C13 (divisibility antisymmetry), derived.

## Goal

Prove a theorem that needs a case split on one of its bound premise variables — without opening an in-prover case-split cohort. Instead: **pre-split** the lemma into guard variants, prove each flat, and merge them back by a construction step licensed by a proved or theorem.

For C13 (`a|b ∧ b|a ⟹ a=b`): prove `…∧ 1≤b ⟹ a=b` and `…∧ b=0 ⟹ a=b` as ordinary pool rows, then merge into the unguarded theorem because `b∈N ⟹ (0=b ∨ 1≤b)` is a proved or theorem.

## Method (the maintainer's standing branching method)

The merge is classical or-elimination. Given proved theorems `P ∧ p₂ ⟹ G` and `P ∧ p₁ ⟹ G` that are identical except for the one premise, and a proved anchor-conditioned or theorem whose disjuncts are exactly `p₁` and `p₂`, conclude `P ⟹ G`. The or theorem is the coverage certificate — a merge without it is unsound and must be refused.

This is the dual of the existing in-run or construction: `constructOrTheoremsInRun` derives an or theorem from a proved implication with a negated premise; the merge consumes an or theorem plus two implications to derive the premise-weakened implication. Same seam, same determinism model.

## Design decisions (all discussed 2026-08-19)

- **Pool model (maintainer correction, 2026-08-19).** The pool holds ONLY the guard variants. The full theorem is NEVER a pool conjecture — the merge DERIVES it, exactly as or theorems are derived rows (a pool copy would spin an LB the whole run for nothing, since the merge cannot fire before both variants prove). Consequently there is no standing LB to close and the seam touches no LB state. Unlike or theorems, the variants are NOT subsumed from `theorems.txt`; they stay first-class pool rows.
- **The merge row is the derived theorem's chapter**, with the new method `or elimination`, citing the two variants' and the or theorem's chapters. The merged theorem is never routed into any LB as a closure device (the D-51 self-citation trap does not arise — nothing stands to close); it is broadcast normally as a proved rule for downstream consumers.
- **Authoring convention (load-bearing).** Each variant = the full theorem plus ONE guard premise appended INNERMOST, binder-free (its variable is bound at an earlier premise). Detection is then byte-level — no alpha machinery in the first build.
- **Pending retry.** A pair whose licensing or theorem has not been minted yet parks and is re-probed at every seam invocation, in insertion order.
- **Proved-not-broadcast tier (maintainer, 2026-08-19 — scope: everywhere).** A degenerate guard variant's proof often skips a redundant premise's level, so the D-278 gate refuses its registration; instead of vanishing, every level-refused closure — in any batch, any mode — now registers as a `proved not broadcast` row with a real chapter and ZERO circulation, and the merge consumes such rows as variants. See [`40_decisions.md` D-290](40_decisions.md#d-290).
- **Seam placement.** The merge scan runs in-run at the phase-4 theorem-drain seam, ordered AFTER the or construction in the same barrier window — a freshly minted or can license a merge in the same drain, and downstream lemmas can consume the merged theorem in the same run.
- **Comparison is canonical, never raw text.** "Differ by one premise" is decided on the disintegrated premise multisets after canonical variable renumbering (`disintegrateImplication` / `reconstructImplicationFullBind`, single binder rule I-4). Raw-text comparison is defeated by alpha-variants with reordered premises (the I-52 lesson).
- **Polarity is literal (I-175).** A negated disjunct matches a negated premise as written; no blind `!`-prefixing anywhere.
- **Or side-premises checked by entailment, not textual subset.** The minted or carries typing premises (`in[b,1]`) that the variants hold only implicitly through their relation premises; the check uses the definition-set machinery (D-41), not string containment.
- **Determinism.** The pair scan walks proved theorems in `globalTheoremList` append order; pair identity and or lookup use canonical normalized forms.
- **Scope of first build: two disjuncts.** k-ary generalization (k variants pairwise differing in the same slot, k-ary or) stays open.
- **Variant 2 (autonomy layer) is deferred.** GL detecting the one-premise pair at conjecture arrival and minting the bridging conjecture `typing ∧ ¬p₁ ⟹ p₂` itself belongs to the auto-discovery phase. In the shortcut campaign, a missing bridge is one authored pool row (C13's bridge is pool row 24, already proved; D1's step split gets its or from row 15).
- **Zero prover hot-path or gate changes.** Cohort opening, demand keys, and premise-count gates stay untouched; `_ordis_` cohorts remain the machinery for splits on mid-proof witnesses that are not head-bound.

## Constraints (one-line rules)

- Merge allowed only when the or theorem over exactly the two differing premises is proved — never on premise-pair heuristics alone.
- Merge allowed only when everything except the one premise is canonically identical — binder structure, other premises, head.
- The merged theorem is a derived row — it is never a pool conjecture and never closes any LB.
- A refused merge is silent non-action, not an error — the pair parks and the derived theorem simply does not appear.

## Pipeline surface (new-proof-tag checklist)

1. Prover: merge constructor at the drain seam; emits the merged theorem with method `or elimination` and the three-citation chapter row.
2. `process_proof_graphs.py`: handle the new method label and citations.
3. `verifier.py` (I-16 — maintainer consent given via this plan): new `TAG_CHECKERS` entry verifying the pair difference on canonical forms, disjunct match with polarity, side-premise entailment, and that the merged head/premises equal the common part.
4. `generate_full_proof_graph.py` + `tag_descriptions.json`: render the new rows (reuse the or-theorem chapter pattern).
5. SwDD: `20_core_concepts/08_proof_tags.md`, `20_core_concepts/07_or_branching.md`, new D/I entries as `D-pending-<slug>` / `I-pending-<slug>` per Rule 15.
6. `docs/fta_ladder/README.md`: pre-split recorded in the unfair-advantage chapter (done alongside this plan).

## Commit plan

1. Merge constructor in the prover + method emission, unit-tested (Rule 18).
2. Python pipeline handling + verifier checker + tag description.
3. Pool rows: 13a and 13b only (full C13 is derived by the merge); shortcut run; diagnose.
4. SwDD chapters + pending D/I entries.
5. Full-pipeline gate; squash on maintainer instruction.

## Verification

```bash
/c/Users/nikol/anaconda3/python.exe main.py --shortcut   # full C13 proved via `or elimination`, verifier airtight
/c/Users/nikol/anaconda3/python.exe main.py              # full gate: 0 failures, theorems.txt preserved
GL_Quick_VS/GL_Quick/gl_quick.exe --unit-tests           # all pass
```

Pass criteria: the full C13 row appears in `files/shortcut/theorems/theorems.txt` (base form) while absent from `conjectures.txt` — a derived row; its chapter carries method `or elimination` citing 13a, 13b, and the or theorem; every verifier category reports zero failures; the full standard run stays category-identical with the theorem pool byte-preserved (modulo the one new always-printed `or elimination` report line).

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
