<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
Contributions require CLA — see CONTRIBUTING.md.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# C6 — sum closure of divisibility: `d | a ∧ d | b ⟹ d | (a + b)`

Shortlist row C6 (39); pool row 52 of `files/shortcut/theorems/conjectures.txt`:

```text
(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](preorder[1,5,9,10])(>[11](preorder[1,5,9,11])(>[12](in3[10,11,12,4])(preorder[1,5,9,12])))))
```

Argument ids: 1=N, 2=0, 3=s, 4=+, 5=·, 6=1, 7=2; 9=d, 10=a, 11=b, 12=a+b (called `s` below). `preorder[1,5,x,y]` is divisibility `x | y` (∃k: x·k = y); `in3[x,y,z,4]` is `x + y = z`; `in3[x,y,z,5]` is `x·y = z`.

## Theorem

For all `d, a, b, s ∈ N`: if `d | a`, `d | b`, and `a + b = s`, then `d | s`.

## Background facts (all externally provided or premise-local)

- **(BG1) Distributivity, composition direction** — `externally_provided_theorems.txt` line 25:
 ```text
  (>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,4])(>[10,11,12](in3[10,11,12,4])(>[13](in3[13,10,7,5])(>[](in3[13,11,8,5])(in3[13,12,9,5]))))))
  ```
 Reading: `x' + y' = s' ∧ x + y = w ∧ d·x = x' ∧ d·y = y' ⟹ d·w = s'`. Exactly the C6 step: from the two divisibility witnesses and the sum decomposition, the product of `d` with the witness sum equals `a + b`.
- **(BG2) Addition totality** — for any two naturals `x, y` there is a `w` with `x + y = w`. Not a corpus row; in GL this existence arrives through witness generation (minted `it_*_lev_*` names), the same mechanism every preceding ladder row relied on for compound witnesses.

## Proof

- **§1.** From premise `d | a` (`(preorder[1,5,9,10])`), take the witness `x`: `d·x = a` and `x ∈ N`.
- **§2.** From premise `d | b` (`(preorder[1,5,9,11])`), take the witness `y`: `d·y = b` and `y ∈ N`.
- **§3.** Premise `a + b = s` (`(in3[10,11,12,4])`) is a fact at the innermost premise block.
- **§4.** By addition totality (BG2), obtain `w` with `x + y = w`.
- **§5.** Instantiate BG1 with `x' = a, y' = b, s' = s, x, y, w, d`: premises `a + b = s` (§3), `x + y = w` (§4), `d·x = a` (§1), `d·y = b` (§2) give `d·w = s`.
- **§6.** From `w ∈ N` and `d·w = s`, conclude `∃k: d·k = s`, i.e. `d | s` — the head `(preorder[1,5,9,12])`. ∎

## Structure GL needs to reconstruct

| § | Expected compact at the innermost premise LB (or its ancestors) | Mechanism |
|---|---|---|
| §1 | `(in3[9,it_<x>,10,5])` + `(in[it_<x>,1])` at/below `(preorder[1,5,9,10])` | premise disintegration (Pass B) |
| §2 | `(in3[9,it_<y>,11,5])` + `(in[it_<y>,1])` at/below `(preorder[1,5,9,11])` | premise disintegration (Pass B) |
| §3 | `(in3[10,11,12,4])` at the innermost LB | premise registration |
| §4 | `(in3[it_<x>,it_<y>,it_<w>,4])` + `(in[it_<w>,1])` — a minted sum witness for the PAIR (x, y) | witness generation |
| §5 | `(in3[9,it_<w>,12,5])` | BG1 rule firing (4 premises, one scope) |
| §6 | goal `(preorder[1,5,9,12])` closes | existence integration over `it_<w>` |

The proof has no case split, no induction, no negation — it is a flat four-premise rule firing plus one witness mint and one integration. Every ingredient except §4 is premise-local or a single corpus rule.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
