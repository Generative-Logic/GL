<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# A16 — human-notation proof of trichotomy

Conjecture row (`files/shortcut/theorems/conjectures.txt`, row 17):

```text
(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9](in[9,1])(>[10](in[10,1])(>[]!(strictOrder[1,4,9,10])(>[]!(=[9,10])(strictOrder[1,4,10,9]))))))
```

Human notation: **∀a, b ∈ ℕ: ¬(a < b) ⟹ (¬(a = b) ⟹ b < a)** — the
flattened implication form of shortlist A16 (trichotomy,
`a < b ∨ a = b ∨ b < a`). The three-way OR form follows mechanically from
this row and its permuted siblings via OR construction, exactly as A15's OR
form followed from its implication row.

Operator glossary (slots: `1` = ℕ, `4` = `+`): `preorder[1,4,a,b]` is
`a ≤ b` (witness form `∃p ∈ ℕ: a + p = b`); `strictOrder[1,4,a,b]` is
`a < b` with definition body "witness-form `≤` inlined AND `!(=[a,b])`"
(`files/definitions/strictOrder.mpl`).

## Proof — case split on totality (no induction needed)

Premises live in one LB chain: `a, b ∈ ℕ` (typing), `¬(a < b)`, `¬(a = b)`.
Goal in the innermost premise scope: `b < a`.

1. **Totality case split.** A15 (in the pool in its or-form,
 `a ≤ b ∨ b ≤ a`) instantiates at `(a, b)` — both typing premises are
 present, so the or fact lands at the innermost scope and the
 or-disintegration machinery opens a two-branch cohort.
2. **Branch `a ≤ b` — refuted.** The pool's A14 contrapositive
 (`a ≤ b ∧ ¬(a < b) ⟹ a = b`, autonomously derived during the A14
 campaign) fires on the branch assumption plus the standing premise
 `¬(a < b)`, yielding `a = b` — a direct contradiction with the standing
 premise `¬(a = b)`. The branch retires as a refuted disjunct
 (dead-branch retirement, D-242 / I-172).
3. **Branch `b ≤ a` — closes the goal.** A14
 (`x ≤ y ∧ ¬(x = y) ⟹ x < y`) instantiates at `(b, a)`: the branch
 assumption supplies `b ≤ a`, and the standing premise `¬(a = b)` supplies
 `¬(b = a)` through the negated-equality mirror (registered pairwise by
 `addNegatedEquality`). The head `b < a` is the goal. Convergence at the
 reduced cohort count (single surviving branch) lifts it to the parent
 scope. ∎

## What each step needs from the engine

- **Step 1** needs the corpus totality row to fire as an **or head** at the
 innermost premise scope, and the cohort to open there (or-admission demand
 evidence per I-177, cohort identity per I-167, sequenced release per
 I-174). The base-form corpus row is `!(&!(preorder[1,4,9,10])!(preorder[1,4,10,9]))`
 under the two typing premises; `precompileStructuralOperators` (I-1)
 restores the compact or form at load.
- **Step 2** needs the branch assumption to be consumable at the branch
 scope together with the two standing premises from the ancestor chain,
 the A14-contrapositive rule to fire there, and the branch-scope
 contradiction (`a = b` against `¬(a = b)`) to retire the branch
 (I-172: cleaned before any convergence count comparison).
- **Step 3** needs the mirror `¬(b = a)` to be visible at the branch scope
 (ancestor fact), the A14 rule to fire, and single-survivor convergence to
 emit the goal at the parent.

No induction, no witness algebra beyond what the fired rules carry — the
entire proof is one case split plus two A14-family rule firings. If the
engine stalls, the frontier is in the or-cohort mechanics or premise
visibility at branch scopes, not in arithmetic.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
