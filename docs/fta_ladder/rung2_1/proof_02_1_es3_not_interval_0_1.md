<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
Contributions require CLA — see CONTRIBUTING.md.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Disproof of `{0, 1, 2} = [0, 1]`

## 1. Statement

Work inside the Peano structure carried by `AnchorIncubator3`:

- `N` is the set of natural numbers;
- `0 ∈ N`;
- `1 = s(0)` and `2 = s(1)`;
- `s: N → N` is injective and never has value `0`;
- `+: N × N → N` is Peano addition.

Define additive order by

```text
a ≤ b  :⇔  there is k ∈ N with a + k = b.
```

The claim to be **disproved** is the false cross-pair left open by rung 2:

```text
{0, 1, 2} = [0, 1].
```

Equivalently, GL must **prove the negation**. With the anchor positions
`1=N, 2=0, 3=s, 4=+, 5=*, 6=1, 7=2, 8=id, 9=3` and `10=M`, the target
theorem is

```text
(>[1,2,3,4,5,6,7,8,9]
  (AnchorIncubator3[1,2,3,4,5,6,7,8,9])
  (>[10](EnumerationSet3[2,6,7,10])!(interval[1,4,2,6,10])))
```

— the exact analogue of the already-proved disproof of the sibling
cross-pair `EnumerationSet2 ⟹ ¬interval[0, 2]`.

---

## 2. Background facts

1. `2 ∈ {0, 1, 2}` — directly from the enumerated-set description.
2. `p ∈ [0, 1] ⟹ p ≤ 1` — the interval's upper-bound clause.
3. `s(a) + k = s(a + k)` — successor-compatibility of addition.
4. `s` is injective: `s(a) = s(b) ⟹ a = b`.
5. `s` never has value `0`: `¬(s(a) = 0)` for every `a ∈ N`.
6. `1 = s(0)`, `2 = s(1)`.

---

## 3. Disproof (reductio)

Assume, toward a contradiction, that `M = {0, 1, 2}` is the interval
`[0, 1]`.

**§3.1.** By fact 1, `2 ∈ M`. By the assumption and fact 2, `2 ≤ 1`,
so there is `k ∈ N` with

```text
2 + k = 1.
```

**§3.2.** Rewrite the left side with fact 6 and fact 3:

```text
2 + k = s(1) + k = s(1 + k),
```

so `s(1 + k) = 1 = s(0)`.

**§3.3.** By injectivity (fact 4),

```text
1 + k = 0.
```

**§3.4.** Rewrite the left side again with fact 6 and fact 3:

```text
1 + k = s(0) + k = s(0 + k),
```

so `s(0 + k) = 0` — a successor with value `0`, contradicting fact 5. ∎

---

## 4. Conclusion

`{0, 1, 2} ≠ [0, 1]`; GL exports the negation theorem stated in §1.
Together with rung 2's positive theorem and the already-proved disproof of
`{0, 1} = [0, 2]`, this settles the entire `EnumerationSet × interval`
cross product of the IncubatorGauss3 batch.

---

## 5. Structure of the proof GL needs to reconstruct

1. The reductio arrives through the `try_contradiction_negated_head`
 machinery: the `__contradiction__` LB assumes the head
 `interval[1,4,2,6,10]` alongside the `EnumerationSet3` premise and must
 derive a full contradiction at its `main` scope (both opposing
 expressions at `main`, per the main-scope contradiction contract).
2. §3.1 is ordinary interval disintegration plus membership: the upper
 bound instantiated at the concrete element `2` yields the witnessed sum
 `2 + k = 1`.
3. §3.2–§3.4 are two rounds of the successor-addition descent — the same
 rule family that closed rung 2's reverse clause — finishing at a
 positive `s(·) = 0` fact whose negation `¬(s(·) = 0)` the batch already
 produces. The pair at `main` discharges the LB and emits the negation
 theorem.
4. **Anticipated wall:** the descent's firings carry several distinct
 `it_`-witness variables, and they run at the contradiction LB's `main`
 scope — NOT inside an `_orint_` branch — so the rung-2
 `maxNumberSecondaryVariablesOrint` widening does not apply there. If
 the descent stalls at the standard cap of 2, the scope condition of the
 widened budget needs a maintainer-approved extension (for example:
 contradiction-LB scopes), measured against the same explosion risk
 that forced the `_orint_` scoping in the first place.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
