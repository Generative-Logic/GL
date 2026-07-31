<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
Contributions require CLA — see CONTRIBUTING.md.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Proof of `{0, 1, 2} = [0, 2]`

## 1. Theorem

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

The claim is

```text
{0, 1, 2} = [0, 2].
```

The left side is the set whose elements are exactly `0`, `1`, and `2`.
The anchor axioms make the three numerals pairwise distinct. The right side is
the closed interval of natural numbers `p` satisfying
`0 ≤ p ≤ 2`.

The current GL conjecture is the direction that assumes the enumerated-set
description and asks GL to integrate the interval description:

```text
(>[1,2,3,4,5,6,7,8,9]
  (AnchorIncubator3[1,2,3,4,5,6,7,8,9])
  (>[10]
    (EnumerationSet3[2,6,7,10])
    (interval[1,4,2,7,10])))
```

The anchor positions are
`1=N, 2=0, 3=s, 4=+, 5=*, 6=1, 7=2, 8=id, 9=3`, and `10=M` is the
set being characterised.

---

## 2. Translation from the GL definitions

### `EnumerationSet3[a,b,c,M]`

[`EnumerationSet3.mpl`](../../../files/definitions/EnumerationSet3.mpl)
states both directions of

```text
x ∈ M  ⇔  x = a ∨ x = b ∨ x = c.
```

The MPL body writes the three-way disjunction as

```text
!(&(&!(=[a,x])!(=[b,x]))!(=[c,x])).
```

For the rung-2 instance this is

```text
x ∈ M  ⇔  x = 0 ∨ x = 1 ∨ x = 2.
```

The compiler represents that left-nested disjunction as two compact ORs.
In the current trace:

```text
or0[2,x,6]       means  x = 0 ∨ x = 1
or1[2,x,6,7]     means  or0[2,x,6] ∨ x = 2.
```

### `interval[N,+,n,m,M]`

[`interval.mpl`](../../../files/definitions/interval.mpl) states

```text
(∀ p. p ∈ M ⇒ preorder(N,+,n,p) ∧ preorder(N,+,p,m))
∧
(∀ p. preorder(N,+,n,p) ∧ preorder(N,+,p,m) ⇒ p ∈ M)
∧
n ∈ N ∧ m ∈ N.
```

[`preorder.mpl`](../../../files/definitions/preorder.mpl) defines

```text
preorder(N,+,a,b)  ⇔  ∃ k ∈ N. a + k = b.
```

Consequently `interval[1,4,2,7,10]` says that `M` contains exactly the
natural numbers between anchor `0` and anchor `2`, inclusive.

---

## 3. Background Peano facts used

The proof uses the following facts. Each is either an axiom in
[`NaturalNumbers.mpl`](../../../files/definitions/NaturalNumbers.mpl) or a
standard immediate Peano consequence already needed on rung 1.

- **P1 — right identity.** `a + 0 = a`.
- **P2 — recursion in the second argument.** `a + s(b) = s(a + b)`.
- **P3 — left identity.** `0 + a = a`. This follows by induction from P1 and
 P2.
- **P4 — successor on the left.** `s(a) + b = s(a + b)`. This follows from
 the Peano addition recursion, or from commutativity plus P2.
- **P5 — successor injectivity.** `s(a) = s(b) ⇒ a = b`.
- **P6 — zero is not a successor.** `s(a) ≠ 0`.
- **P7 — predecessor split.** Every `x ∈ N` is either `0` or `s(y)` for some
 `y ∈ N`.
- **P8 — sum-zero split.** If `a,b ∈ N` and `a+b=0`, then `a=0` and `b=0`.
- **P9 — order reflexivity.** `a ≤ a`, witnessed by `0` through P1.

The small additions required by the forward inclusion are therefore

```text
0+0=0,  0+1=1,  0+2=2,  1+1=2,  2+0=2.
```

For example,
`1+1 = 1+s(0) = s(1+0) = s(1) = 2` by P2 and P1.

---

## 4. Proof

The interval definition contains both membership directions, so the GL
conjecture must reconstruct both set inclusions even though the outer
conjecture is written only as `EnumerationSet3 ⇒ interval`.

### 4.1 `{0,1,2} ⊆ [0,2]`

Take `x ∈ {0,1,2}`. By `EnumerationSet3`, exactly the following three cases
have to be covered.

#### Case `x = 0`

- `0 ≤ x`, witnessed by `0`, because `0+0=0=x`.
- `x ≤ 2`, witnessed by `2`, because `x+2=0+2=2`.

Thus `x ∈ [0,2]`.

#### Case `x = 1`

- `0 ≤ x`, witnessed by `1`, because `0+1=1=x`.
- `x ≤ 2`, witnessed by `1`, because `1+1=2`.

Thus `x ∈ [0,2]`.

#### Case `x = 2`

- `0 ≤ x`, witnessed by `2`, because `0+2=2=x`.
- `x ≤ 2`, witnessed by `0`, because `2+0=2=x`.

Thus `x ∈ [0,2]` in every enumerated case, proving
`{0,1,2} ⊆ [0,2]`.

### 4.2 `[0,2] ⊆ {0,1,2}`

Take `p ∈ [0,2]`. The upper bound `p ≤ 2` supplies an `l ∈ N` with

```text
p + l = 2 = s(1).
```

Split `p` using P7.

#### Case `p = 0`

Then `p ∈ {0,1,2}` directly.

#### Case `p = s(q)` for some `q ∈ N`

Substitute `p=s(q)` into the upper-bound equation:

```text
s(q) + l = s(1).
```

By P4, the left side is `s(q+l)`. P5 then gives

```text
q + l = 1 = s(0).
```

Now split `q` using P7.

##### Subcase `q = 0`

Then

```text
p = s(q) = s(0) = 1,
```

so `p ∈ {0,1,2}`.

##### Subcase `q = s(r)` for some `r ∈ N`

Substitute into `q+l=1`:

```text
s(r) + l = s(0).
```

P4 and P5 yield `r+l=0`. P8 forces `r=0` and `l=0`. Hence

```text
q = s(r) = s(0) = 1
p = s(q) = s(1) = 2.
```

Again `p ∈ {0,1,2}`.

All cases give membership in the enumerated set, proving
`[0,2] ⊆ {0,1,2}`.

### 4.3 Conclusion

The two inclusions give

```text
{0,1,2} = [0,2].
```

No arithmetic beyond the Peano structure is needed. Rung 2 adds one genuine
proof layer over rung 1: after the first predecessor reduction, the proof must
perform a second case split to distinguish predecessor `0` from predecessor
`1`.

---

## 5. Structure GL must reconstruct

### Forward interval clauses

For a universal element `x ∈ M`, the ES3 forward clause produces

```text
or1[0,x,1,2] = (or0[0,x,1]) ∨ (x=2).
```

GL must reason through all three atomic cases:

1. flatten the contiguous `or1` / `or0` tree into the sibling branches
 `x=0`, `x=1`, and `x=2` in one `_ordis_` operation; all branches retain
 the original outer `or1` signature;
2. derive both `0≤x` and `x≤2` in every atomic branch;
3. converge each preorder once across its three-branch cohort at the immediate
 implication scope;
4. discharge the two forward preorder goals.

### Backward interval clause

For `0≤p≤2`, GL must:

1. open the witness `l` with `p+l=2`;
2. use the first predecessor split on `p`;
3. reduce the successor branch to `q+l=1`;
4. use a second predecessor split on `q`;
5. close `p=0`, `p=1`, or `p=2` as appropriate;
6. integrate the nested OR and then derive `p ∈ M`.

### Current architectural boundary

The frozen IncubatorGauss3 traces confirm that both OR directions now use one
ordered cohort of the three atomic alternatives. Forward `_ordis_` closes both
preorder goals. Reverse `_orint_` gives each branch both peer negations, keeps
equal signatures below different parents isolated during cleanup, and closes a
descendant goal directly from its strongest visible ancestor without storing a
duplicate child fact.

Rung 2 remains open. No diagnosis of the next failing inference is recorded
here. The trace evidence and complete covered-step ledger are in
[`current_proof_state.md`](./current_proof_state.md).

---

## 6. Validation evidence — 2026-07-23

Two complete 15-iteration Windows pipelines reproduced the same frontier. Both
finished with 132,033 verifier checks and zero failures, and the four generated
theorem and global-theorem-list artifacts were byte-identical by SHA-256.

The preserved IncubatorGauss3 traces show:

1. exactly three atomic reverse-integration branches and no intermediate OR
 branch;
2. both peer negations in every branch;
3. successful ancestor-visible closure on the endpoint-1 sibling;
4. the false endpoint-1 cross-pair still unproved.

Thus the reverse OR-integration repair is validated and its former boundary is
closed. Rung 2 remains open without a recorded diagnosis of its next boundary.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
