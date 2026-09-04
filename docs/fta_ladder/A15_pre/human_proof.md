<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# A15_pre — human-notation proof of the totality implication

Conjecture row (`files/shortcut/theorems/conjectures.txt`):

```text
(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9](in[9,1])(>[10](in[10,1])(>[]!(preorder[1,4,9,10])(preorder[1,4,10,9])))))
```

Human notation: **∀a, b ∈ ℕ: ¬(a ≤ b) ⟹ b ≤ a** — the implication half of
shortlist A15 (totality). The OR form `a ≤ b ∨ b ≤ a` follows mechanically
from this row and its head-switched mirror via OR construction.

Witness form throughout: `a ≤ b:⟺ ∃p ∈ ℕ: a + p = b`. In MPL this is
`preorder[N,+,a,b]` with definition body `!(>[p](in[p,N])!(in3[a,p,b,+]))` —
the standard negated-implication existence encoding. Consequently the negated
premise `!(preorder[1,4,9,10])` is a **double negation**: substituting the
definition body and cancelling `!!` yields the positive universally
quantified rule

```text
(>[p](in[p,1])!(in3[9,p,10,+]))        — "∀p ∈ ℕ: a + p ≠ b"
```

## Proof — GL-style induction on b (digit arg 10)

GL's recursion triad works at `b` with `rec` the **predecessor**: the triad
hypothesis is `(in2[rec0,10,3])`, i.e. `b = s(rec)` (rec plays the role of
b − 1), and the induction hypothesis is the whole theorem instantiated at
`rec`. Available in the triad LB: the asserted premise `¬(a ≤ b)` (as the
expanded rule `∀q ∈ ℕ: a + q ≠ b`), the hypothesis `b = s(rec)`, and the IH
rule `¬(a ≤ rec) ⟹ rec ≤ a`. Goal: `b ≤ a`.

**Zero block (b = 0).** Goal `¬(a ≤ 0) ⟹ 0 ≤ a`: the head `0 ≤ a` holds
unconditionally with witness `a`, since `0 + a = a` (corpus); the premise is
unused. (This is A1, already in the pool.)

**Step block (b = s(rec)).**

1. **`¬(a ≤ rec)`** — reductio: assume `a ≤ rec`, witness `k` with
 `a + k = rec`. Successor law: `a + s(k) = s(a + k) = s(rec) = b` — fires
 the premise rule at `q = s(k)` — contradiction. Hence `¬(a ≤ rec)`.
 [This is exactly the fact the IH rule demands: `!(preorder[1,4,9,rec0])`.]
2. **`rec ≤ a`** — the IH rule fires on step 1's fact; witness `m` with
 `rec + m = a`.
3. **Case split on m** (the corpus or0 theorem: `m = 0 ∨ ∃m′: m = s(m′)`):
 - **`m = 0`:** then `rec = a` (`rec + 0 = rec`), so `b = s(a)`; but
 `a + s(0) = s(a + 0) = s(a) = b` fires the premise rule — contradiction;
 the branch retires as a refuted disjunct.
 - **`m = s(m′)`:** `b + m′ = s(rec) + m′ = s(rec + m′) = rec + s(m′) = a`
 (successor laws + commutativity) — witness `m′` closes the goal:
 `b ≤ a`. ∎

## What each step needs from the engine

- **Step 1 — SOLVED by contradiction-based integration (landed).** The IH's negated premise `!(preorder[1,4,9,rec])`
 is integrated by reductio: a `contradiction_`-marked child scope assumes
 the positive compact, standard consumption decomposes it, and a
 scope-level contradiction discharges the negated compact at the parent
 (`prepareNegatedCompoundContradiction` +
 `dischargeContradictionScopes`). Verified in-run: the discharge fired and
 the IH produced `(preorder[1,4,rec,9])` at main. (An earlier
 dual-implication subproof design — prove `A → !B` / `B → !A` as goals —
 was implemented and abandoned: its subproof goal is a negated atom that
 itself needs contradiction machinery, and the two directions collide on
 one scope payload.)
- **Step 2** is a plain rule firing once step 1's fact exists — confirmed
 firing in-run.
- **Step 3 is the OPEN piece.** The or0 predecessor split on rec≤a's witness
 is currently consumed FLAT (the D-32 or-admission gate blocks branch
 minting for corpus-fired or heads), so neither branch runs. The agreed
 fix is the sequenced or-disintegration architecture change — full design
 and session handoff in [`current_proof_state.md`](current_proof_state.md).
 Under it, the `m = 0` branch dies on the premise rule and retires via the
 refuted-disjunct retirement (D-242), and the `m = s(m′)` branch is pure
 corpus algebra (successor laws, commutativity).

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
