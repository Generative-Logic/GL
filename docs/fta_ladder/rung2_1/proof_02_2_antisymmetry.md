<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
Contributions require CLA — see CONTRIBUTING.md.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Antisymmetry of the additive order: `a ≤ b ∧ b ≤ a ⇒ a = b`

## 1. Statement

Work inside the Peano structure carried by `AnchorPeano`:

- `N` is the set of natural numbers;
- `0 ∈ N`;
- `1 = s(0)`;
- `s: N → N` is injective and never has value `0`;
- `+: N × N → N` is Peano addition (a total, single-valued mapping);
- `*: N × N → N` is Peano multiplication.

Define additive order by

```text
a ≤ b  :⇔  there is k ∈ N with a + k = b.
```

The claim to be **proved** is antisymmetry of this order:

```text
a ≤ b  ∧  b ≤ a   ⇒   a = b.
```

With the anchor positions `1=N, 2=0, 3=s, 4=+, 5=*, 6=1`, the target
theorem is

```text
(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8](preorder[1,4,7,8])(>[](preorder[1,4,8,7])(=[7,8]))))
```

— emitted into the Peano conjecture pool by the 2026-07-27 config
extension, not yet proved. Its role in the FTA ladder: applied to the
derived facts `1 ≤ 2` and `2 ≤ 1` inside the rung-2.1 reductio it yields
`2 = 1`, whose negation the anchor supplies, closing the rung without
any witness minting inside the reductio.

---

## 2. Background facts

All four lemmas are already proved in the same Peano batch (byte-exact
rows in `files/theorems/theorems.txt`); the remaining facts are axioms
of the anchor's definition set.

1. **Relational associativity** — `x+y=z ∧ y+u=v ∧ w+u=x ⇒ v+w=z`:

 ```text
   (>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,4])(>[10,11](in3[8,10,11,4])(>[12](in3[12,10,7,4])(in3[11,12,9,4])))))
   ```

2. **Cancellation** — `x+n=y ∧ n=y ⇒ x=0` (equivalently `x+n=n ⇒ x=0`):

 ```text
   (>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,4])(>[](=[8,9])(=[2,7]))))
   ```

3. **Zero-sum lemma** — `x+y=0 ∧ y∈N ⇒ x=0`:

 ```text
   (>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8](in3[7,8,2,4])(>[](in[8,1])(=[2,7]))))
   ```

4. **Mapping property of `+`** (`fXYZ[+,N,N,N]` inside the
 `NaturalNumbers` definition): typing (`x+y=z ⇒ x,y,z∈N`), totality
 (`x,y∈N ⇒ ∃z∈N: x+y=z`), and single-valuedness
 (`x+y=z1 ∧ x+y=z2 ⇒ z1=z2`).
5. **Zero identity** (axiom pair in `NaturalNumbers`): `a∈N ⇒
 (a+0=b ⇒ a=b)` and its converse.

---

## 3. Proof

Assume `a ≤ b` and `b ≤ a`.

**§3.1.** Unfold both order facts: there are `k, l ∈ N` with

```text
a + k = b        and        b + l = a.
```

**§3.2.** By fact 4 (totality; `k, l ∈ N`), there is `m ∈ N` with

```text
k + l = m.
```

**§3.3.** Apply fact 1 (associativity) to the triple
`a+k=b`, `k+l=m`, `b+l=a` (substitution `x=a, y=k, z=b, u=l, v=m, w=b`):

```text
m + b = b.
```

**§3.4.** By fact 2 (cancellation, instance `x=m, n=b, y=b`):

```text
m = 0.
```

**§3.5.** Rewrite §3.2 under `m = 0` to `k + l = 0`; by fact 3
(zero-sum, with `l ∈ N`):

```text
k = 0.
```

**§3.6.** Rewrite §3.1's first sum under `k = 0` to `a + 0 = b`; by
fact 5 (zero identity, with `a ∈ N` from fact 4's typing):

```text
a = b.   ∎
```

---

## 4. Conclusion

The additive order on `N` is antisymmetric. Downstream, the theorem is
the rung-2.1 closer: inside the ES3 reductio the derived pair `1 ≤ 2`
and `2 ≤ 1` at `main` yields `2 = 1`, the anchor external `¬(1 = 2)`
completes the contradiction pair, and the negation theorem
`EnumerationSet3 ⇒ ¬interval[0,1]` exports.

---

## 5. Structure of the proof GL needs to reconstruct

1. The conjecture builds the premise-LB chain root →
 `(AnchorPeano[1,2,3,4,5,6])` → `(preorder[1,4,7,8])` →
 `(preorder[1,4,8,7])`; the goal `(=[7,8])` lives at the innermost
 LB's `main`.
2. §3.1 is status-0 fuel disintegration: each assumed `preorder`
 premise mints its existence witness (`a+k=b` with an `it_`/`int_`
 witness name plus the witness's `N`-membership) at its own LB; the
 outer premise's products reach the innermost LB through ancestor
 mail.
3. §3.2 is the demanded-output step: the totality clause of
 `fXYZ[+,N,N,N]` fires on the two witnesses, and the output-slot
 witness `m` must actually mint — the consumer-side admission the
 maintainer's strategy relies on (associativity stands installed
 with two of its three premises live, demanding exactly this
 output).
4. §3.3–§3.5 are plain hash-rule firings of the three same-batch
 lemmas plus one equivalence-class rewrite each; §3.6 is the zero
 identity axiom closing the goal.
5. **Watchpoints** (why this proof may stall; see the companion state
 document): the §3.3 instance repeats `b` across two
 template-distinct slots and the §3.4 instance additionally needs
 the reflexive equality `b = b` — both are collision-pattern
 instances of the kind rung 2.1's first investigation showed the
 request match cannot assemble unless an equality-class alias
 launders the repetition; and all three lemmas are same-run proofs,
 so their broadcast timing competes with the antisymmetry LB's
 iteration budget.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
