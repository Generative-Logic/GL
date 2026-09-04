<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
Contributions require CLA — see CONTRIBUTING.md.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# B8 (multiplicative cancellation) — human-notation proof

Target (row 42): `c ≥ 1 ∧ a·c = b·c ⟹ a = b`.

Pool-variable mapping (row 42's ladder): `9 = c`, `10 = a`, `11 = P` (the shared product — `a·c = b·c` is encoded by both `in3` atoms writing the same result slot), `12 = b`. Premise LBs: `(preorder[1,4,6,9])` → `(in3[10,9,11,5])` (the working LB) → `(in3[12,9,11,5])` (the goal LB, `toBeProved` = `(=[10,12])`). Anchor ids: `1 = N`, `2 = 0`, `3 = s`, `4 = +`, `5 = ·`, `6 = 1`, `7 = 2`.

## Proof

- **§1 (reductio).** Assume `a ≠ b` — expected in-run shape: `!(=[10,12])` asserted in a contradiction / hypothesis scope under the goal LB's chain (I-165 pattern: `__contradiction__` LB primed with the negated head, or a hypothesis-disintegration boundary scope).
- **§2 (case split via trichotomy).** From `a, b ∈ N` and `a ≠ b`, exactly one of `a < b` / `b < a` holds. Pool inventory for this step (all implication-form, not or-form):
 - row 16 (A15 totality): `¬(a ≤ b) ⟹ b ≤ a` — `(>[]!(preorder[1,4,9,10])(preorder[1,4,10,9]))` under `in[9,1]`/`in[10,1]`;
 - row 15 (A14): `a ≤ b ∧ a ≠ b ⟹ a < b` — `(>[9,10](preorder[1,4,9,10])(>[]!(=[9,10])(strictOrder[1,4,9,10])))`;
 - row 17 (A16 trichotomy, implication form): `¬(a < b) ∧ a ≠ b ⟹ b < a`.
 The split needs either an in-run `_ordis_` cohort over `a ≤ b ∨ b ≤ a` (B3 precedent: or-compacts are post-run constructions; in-run cohorts have only ever come from corpus predecessor definitional compounds) or a double-reductio chain through rows 16/15/17 (assume `¬(a<b)`, derive `b<a`, refute; conclude `a<b`, refute; close).
- **§3 (case `a < b`).** `(strictOrder[1,4,10,12])`. By B5 (row 40) with `a < b`, `1 ≤ c`, `a·c = P`, `b·c = P`: `P < P` — expected head `(strictOrder[1,4,11,11])`.
- **§4 (case `b < a`).** Symmetric: B5 with `(strictOrder[1,4,12,10])` and the two products in swapped roles gives the same `(strictOrder[1,4,11,11])`.
- **§5 (absurdity).** `P < P` is false. Pool inventory: row 29 (A21 irreflexivity, form `a = b ⟹ ¬(a < b)`) needs a live reflexive equality `(=[11,11])` to bind — which the pipeline never mints as a statement (I-8 bans trivial equalities in heads; no reflexive-equality emitter exists at runtime). The alternative refutation is definitional: `strictOrder` unfolds to a witness form (`∃k: P + s(k) = P`-shape), whose additive consequences contradict the A-family (cancellation / `a + k = a ⟹ k = 0` route, rows 5–7/12). Which mechanism GL would actually use is exactly what the trace must show.
- **§6 (close).** Both cases refuted ⟹ `¬(a ≠ b)` discharges the reductio ⟹ `(=[10,12])` at the goal LB's main; the goal closes.

## Alternative route — induction (no trichotomy, no reductio on the head)

Induction on `a`, `b` universally quantified; only mechanisms GL already exercises elsewhere.

- **§I1 (base `a = 0`).** `0·c = 0` (recursion base), so `b·c = 0`. From `1 ≤ c`: `c ≠ 0` (row 25). No zero divisors (B1, row 31 form: `x·y = 0 ∧ x ∈ N ∧ x ≠ 0 ⟹ y = 0`, bound `x:=c, y:=b` after commutativity) gives `b = 0 = a`.
- **§I2 (step, `a → s(a)`).** Given `s(a)·c = b·c`. First `b ≠ 0`: else `s(a)·c = 0`, and B1 with `s(a) ≠ 0` (successor row 39) and `c ≠ 0` refutes. So `b = s(m)` — exactly the corpus predecessor `or1` split that demonstrably opens in-run.
- **§I3.** Recursion `s(x)·c = x·c + c` on both sides: `a·c + c = m·c + c`. Additive cancellation (row 12) gives `a·c = m·c`.
- **§I4.** Induction hypothesis (`a` vs `m` with the same `c`): `a = m`, hence `s(a) = s(m) = b`. Close.

Expected in-run shape: an induction triad on the head with the predecessor cohort inside the successor branch — the machinery B7 (`induction v2`) and B1 already used. The goal-LB trace shows zero induction markers, so this route was never attempted; whether the scheduler CAN reach an equality head of this premise shape is one of the investigation's open questions.

## Mechanism-sensitive steps (what the trace walk checks)

1. **Does §1's reductio scope open at all** (a `__contradiction__(=[10,12])`-suffixed LB or a boundary hypothesis scope carrying `!(=[10,12])`)?
2. **Does §2's case split ever open in-run** — any `_ordis_` cohort, or the totality row 16 firing under the reductio scope? This is the prime suspect: B3 established that demand-driven splits outside corpus predecessor cohorts have no in-run producer.
3. **If a case opens, does B5's rule fire** (4-premise rule: `strictOrder + preorder + in3 + in3` — watch the I-6 single-input fan-out gate from B3 stall #1: the products here are PREMISE-bound (`11` is a bound pool variable), not symbolic witness products, so the B3 fan-out refusal should NOT apply — verify).
4. **Does the `P < P` refutation have any producer** (§5's two candidate mechanisms)?

The frontier is the first § step with no trace evidence; the mechanism gap or bug lives there.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
