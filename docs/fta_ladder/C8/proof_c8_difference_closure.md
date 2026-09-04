<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
Contributions require CLA — see CONTRIBUTING.md.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.



# C8 — difference closure of divisibility: `d | a ∧ a + b = c ∧ d | c ⟹ d | b`

Shortlist row C8 (41); pool row 54 of `files/shortcut/theorems/conjectures.txt`:

```text
(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](preorder[1,5,9,10])(>[11,12](in3[10,11,12,4])(>[](preorder[1,5,9,12])(preorder[1,5,9,11])))))
```

Argument ids: 1=N, 2=0, 3=s, 4=+, 5=·, 6=1, 7=2; 9=d, 10=a, 11=b, 12=c (the sum `a + b`). `preorder[1,5,x,y]` is divisibility `x | y` (∃k: x·k = y); `preorder[1,4,x,y]` is `x ≤ y` (∃k: x + k = y); `in3[x,y,z,4]` is `x + y = z`; `in3[x,y,z,5]` is `x·y = z`.

This is the division-free "subtract a multiple" step: it is what replaces `b = c − a` in a ring-free setting, and it is load-bearing for D2, G1 and G5 further up the ladder.

## Theorem

For all `d, a, b, c ∈ N`: if `d | a`, `a + b = c` and `d | c`, then `d | b`.

## Background facts

- **(BG1) Multiplicative order reflection (B9)** — pool row 42:
 ```text
  (>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9](preorder[1,4,6,9])(>[10,11](in3[10,9,11,5])(>[12,13](in3[12,9,13,5])(>[](preorder[1,4,11,13])(preorder[1,4,10,12]))))))
  ```
 Reading: `1 ≤ d ∧ x·d = p ∧ y·d = q ∧ p ≤ q ⟹ x ≤ y`. Note the common factor sits in the **second** argument slot, while divisibility hands it over in the **first** — one commutativity rewrite (BG5) bridges the two.
- **(BG2) Distributivity, decomposition direction** — `externally_provided_theorems.txt` line 33:
 ```text
  (>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,5])(>[10,11](in3[7,10,11,5])(>[12,13](in3[7,12,13,5])(>[](in3[8,10,12,4])(in3[9,11,13,4]))))))
  ```
 Reading: `d·x = p ∧ d·z = q ∧ d·y = r ∧ x + z = y ⟹ p + q = r`.
- **(BG3) Additive cancellation** — pool row 12: `x + w = t ∧ v + w = t ⟹ x = v` (cancels the **second** summand; the first-summand form needs one addition-commutativity rewrite, corpus line for `+`).
- **(BG4) Zero product / zero sum facts** — `0·x = 0` and `x + y = 0 ⟹ x = 0 ∧ y = 0` (pool rows 6/7), plus `n | 0` (pool row 49).
- **(BG5) Multiplication commutativity** — `externally_provided_theorems.txt` line 35: `x·y = z ⟹ y·x = z`.
- **(BG6) Multiplication totality** — for any `x, y` there is a `w` with `x·y = w`; in GL this existence arrives through witness generation (minted `it_*_lev_*` names).
- **(BG7) Positivity** — pool row 24: `n ∈ N ∧ ¬(n = 0) ⟹ 1 ≤ n`.

## Proof

- **§1.** From premise `d | a`, take the witness `x`: `d·x = a`, `x ∈ N`.
- **§2.** From premise `d | c`, take the witness `y`: `d·y = c`, `y ∈ N`.
- **§3.** Premise `a + b = c` is a fact at the innermost premise block.
- **§4.** Substituting §1 and §2 into §3: `d·x + b = d·y`. By the definition of `≤` with the **already present** witness `b`, this is `d·x ≤ d·y` — no minting required.
- **§5 (case `d = 0`).** Then `a = 0·x = 0` and `c = 0·y = 0` (BG4). With §3, `0 + b = 0`, so `b = 0` (BG4), and `0 | 0` (BG4, row 49). Head holds.
- **§6 (case `1 ≤ d`).** Commuting §1, §2 (BG5) gives `x·d = a`, `y·d = c`; with §4 and BG1 this yields `x ≤ y`.
- **§7.** From `x ≤ y` take the witness `z`: `x + z = y`, `z ∈ N`.
- **§8.** By multiplication totality (BG6), obtain `q` with `d·z = q`.
- **§9.** BG2 with `p = a`, `r = c`: from `d·x = a`, `d·z = q`, `d·y = c`, `x + z = y` conclude `a + q = c`.
- **§10.** From §3 (`a + b = c`) and §9 (`a + q = c`), additive cancellation (BG3, after one commutativity rewrite) gives `b = q`.
- **§11.** So `d·z = b` with `z ∈ N`, i.e. `∃k: d·k = b` — the head `(preorder[1,5,9,11])`. ∎

## Structure GL needs to reconstruct

| § | Expected compact at the innermost premise LB (or its ancestors) | Mechanism |
|---|---|---|
| §1 | `(in3[9,it_<x>,10,5])` + `(in[it_<x>,1])` at/below `(preorder[1,5,9,10])` | premise disintegration (Pass B) |
| §2 | `(in3[9,it_<y>,12,5])` + `(in[it_<y>,1])` at/below `(preorder[1,5,9,12])` | premise disintegration (Pass B) |
| §3 | `(in3[10,11,12,4])` at the innermost LB | premise registration |
| §4 | `(preorder[1,4,10,12])` — i.e. `a ≤ c` with witness `b` | existence integration over the present witness `b` |
| §5 | branch on `(=[2,9])` | case split over the or `d = 0 ∨ 1 ≤ d`, minted in-run from pool row 24 |
| §6 | `(preorder[1,4,it_<x>,it_<y>])` | BG5 rewrites + BG1 firing (4 premises, one scope) |
| §7 | `(in3[it_<x>,it_<z>,it_<y>,4])` + `(in[it_<z>,1])` | witness generation over a minted pair |
| §8 | `(in3[9,it_<z>,it_<q>,5])` | multiplication totality on a minted pair |
| §9 | `(in3[10,it_<q>,12,4])` | BG2 firing (4 premises, one scope) |
| §10 | `(=[11,it_<q>])` | BG3 firing + commutativity rewrite |
| §11 | goal `(preorder[1,5,9,11])` closes | existence integration over `it_<z>` |

## Why this row is structurally harder than C5–C7

C5, C6 and C7 each closed by **building one product or sum witness out of premise witnesses and firing a single corpus rule**. C8 is the first divisibility row that reverses the direction: the head's witness `z` is not constructed from the premise witnesses by an operation, it is **extracted from an order fact**. That forces three things none of the earlier rows needed:

1. **A case split on `d = 0`** (§5) — nothing in the premises rules out `d = 0`, and BG1 is false without `1 ≤ d`. The split itself is available: ors are minted and broadcast in-run (`constructOrTheoremsInRun`), and pool row 24 (`n ∈ N ∧ ¬(n = 0) ⟹ 1 ≤ n`) is exactly the negated-premise implication that yields `d = 0 ∨ 1 ≤ d`; the C8 trace shows `or0` and `or10` rules registered at the premise LB. What the split costs is a cohort whose every branch has to carry the rest of the chain.
2. **Four minted witnesses in one chain** — `x`, `y` (premise existentials), `z` (from the order fact), `q` (multiplication totality on a minted pair) — with rule firings that must pair minted names across two different scopes.
3. **Two orientation rewrites** (BG5 for the factor order BG1 expects, and addition commutativity for BG3's cancellation slot), each multiplying the request space at exactly the point where the chain is deepest.

That combination is the "deep manipulation with multiple `it_` witnesses" this engine is not expected to perform unaided; the ladder's standing repair for it is a helper pool lemma stated over **bound** variables, proved cap-free in its own grid, and fired in the C8 grid as one rule.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
