<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-->

# Proof of `{0, 1} = [0, 1]`

**Filename note.** The requested name `proof_{0,1}=[0,1].md` contains characters
(`{`, `}`, `=`) that collide with bash brace-expansion and are awkward to pass
to `git add` / shells without quoting. This file uses a safer name;
the theorem itself is stated in full mathematical notation below.

---

## 1. Theorem

Working inside a Peano structure `(N, 0, 1, s, +)` where
- `N` is the set of natural numbers,
- `0 ∈ N`, `1 := s(0) ∈ N`,
- `s : N → N` is the successor,
- `+ : N × N → N` is the addition induced by `s`,

**Claim.**
```
{0, 1}  =  [0, 1]
```
where the two sides are:

- **Left (enumerated set):**
  `{0, 1} := { x ∈ N : x = 0 ∨ x = 1 }`
- **Right (closed interval):**
  `[0, 1] := { p ∈ N : 0 ≤ p ≤ 1 }`,
  with the order defined additively, `a ≤ b  :⇔  ∃ k ∈ N. a + k = b`.

The claim is a set equality — both sides denote the same subset of `N`.

---

## 2. Translation from the GL definitions

The two sides are compiled by GL from the MPL files
`files/definitions/EnumerationSet2.txt` and `files/definitions/interval.txt`.
Restating them in standard notation:

**`EnumerationSet2[a, b, M]`**
A predicate saying "M is the set `{a, b}`". Unfolded:
```
(∀ x. x ∈ M  ⟹  x = a ∨ x = b)
∧
(∀ x. x = a ∨ x = b  ⟹  x ∈ M)
```
Note GL stores both directions as the double implication
`¬(¬(a=x) ∧ ¬(b=x))`, which is logically equivalent to `a = x ∨ b = x`.

**`interval[N, +, n, m, M]`**
A predicate saying "M is the closed interval `[n, m]` on `N`". Unfolded:
```
(∀ p. p ∈ M  ⟹  preorder(N, +, n, p) ∧ preorder(N, +, p, m))
∧
(∀ p. preorder(N, +, n, p) ∧ preorder(N, +, p, m)  ⟹  p ∈ M)
∧
n ∈ N  ∧  m ∈ N
```

**`preorder[N, +, n, m]`** (`files/definitions/preorder.txt`)
```
preorder(N, +, n, m)  :⇔  ∃ p ∈ N.  in3(n, p, m, +)
                     ⇔  ∃ p ∈ N.  n + p = m
```
That is, `preorder(n, m)` is the standard additive order `n ≤ m` on `N`.

With these translations the theorem instance we want is

> For the AnchorGauss constants `a = 0`, `b = 1`, `n = 0`, `m = 1`,
> the ES2 witness set and the interval witness set coincide:
> a set `M` satisfies `EnumerationSet2[0, 1, M]` iff it satisfies
> `interval[N, +, 0, 1, M]`.

In the raw theorems file (`files/theorems/theorems.txt` line 340):
```
(>[1,2,4,6] (AnchorGauss[1,2,3,4,5,6,7,8])
  (>[9] (EnumerationSet2[2,6,9])
        (interval[1,4,2,6,9])))
```
with the anchor arg map `[N=1, i0=2, s=3, +=4, *=5, i1=6, i2=7, id=8]`,
i.e. variable `2` plays the role of `0` and variable `6` plays the role
of `1`, and variable `9` is the shared set `M`.

---

## 3. Background Peano facts used

All of the following are immediate from the Peano axioms and the recursive
definition of `+` (`a + 0 = a`, `a + s(b) = s(a + b)`):

- (P1) **Left identity of addition.** `0 + k = k` for all `k ∈ N`.
- (P2) **Successor pulls out of `+`.** `s(a) + b = s(a + b)`.
- (P3) **`s` is injective.** `s(a) = s(b) ⟹ a = b`.
- (P4) **`0` is not a successor.** `s(a) ≠ 0` for all `a ∈ N`.
- (P5) **Case split on inhabitants of `N`.** Every `x ∈ N` is either
  `0` or `s(y)` for some `y ∈ N`.
- (P6) **Sum-zero split.** `a + b = 0 ⟹ a = 0 ∧ b = 0`.
  *Proof:* if `a = s(a')`, then by (P2) `a + b = s(a' + b)`, and by (P4)
  `s(a' + b) ≠ 0`, contradicting the hypothesis. So `a = 0`; then by
  (P1) `a + b = 0 + b = b`, hence `b = 0`. ∎
- (P7) **Reflexivity of `≤`.** `a ≤ a`, witnessed by `k = 0` and
  `a + 0 = a`.

These are exactly the pieces that GL's prover must chain through the
hash-memory during the proof search of this theorem.

---

## 4. Proof

We prove the two inclusions separately. No auxiliary "set variable" is
introduced — the argument is directly between the two defined sets
`{0, 1}` and `[0, 1]`. The side conditions `0 ∈ N` and `1 = s(0) ∈ N`
are immediate from the Peano axioms and are used implicitly below.

### 4.1  `{0, 1} ⊆ [0, 1]`

Take any `x ∈ {0, 1}`. By definition `x = 0` or `x = 1`.

- *Case `x = 0`.*
  - `0 ≤ x`: witness `k = 0`, since `0 + 0 = 0 = x` by (P1). ✓
  - `x ≤ 1`: witness `l = 1`, since `x + 1 = 0 + 1 = 1` by (P1). ✓
  So `0 ≤ x ≤ 1`, hence `x ∈ [0, 1]`.

- *Case `x = 1`.*
  - `0 ≤ x`: witness `k = 1`, since `0 + 1 = 1 = x` by (P1). ✓
  - `x ≤ 1`: witness `l = 0`, since `x + 0 = 1 + 0 = 1 = x` by the
    recursive definition of `+` at the second argument. ✓
  So `0 ≤ x ≤ 1`, hence `x ∈ [0, 1]`.

### 4.2  `[0, 1] ⊆ {0, 1}`

Take any `p ∈ [0, 1]`, i.e. assume `0 ≤ p` and `p ≤ 1`. From `p ≤ 1`
there exists `l ∈ N` with `p + l = 1 = s(0)`.

Split on `p` via (P5):

- *Case `p = 0`.* Then `p ∈ {0, 1}` trivially.

- *Case `p = s(p')` for some `p' ∈ N`.* Substituting in `p + l = s(0)`:
  ```
  s(p') + l = s(0)
  ```
  By (P2), `s(p') + l = s(p' + l)`, so `s(p' + l) = s(0)`.
  By (P3), `p' + l = 0`.
  By (P6), `p' = 0` and `l = 0`.
  Therefore `p = s(p') = s(0) = 1`, hence `p ∈ {0, 1}`.

Both cases yield `p ∈ {0, 1}`, so `[0, 1] ⊆ {0, 1}`.

Combining §4.1 and §4.2, `{0, 1} = [0, 1]`.

### 4.3  Relation to the two GL conjectures

GL stores the set equality as two *directed* conjectures at
`theorems.txt:340` (ES2 ⟹ interval) and `theorems.txt:342`
(interval ⟹ ES2). Each one, for a single variable `M` assumed to
satisfy one characterisation, demands `M` also satisfies the other.
Because `{0, 1}` and `[0, 1]` are the *unique* sets satisfying the
respective characterisations, both conjectures reduce to the single
set equality proved in §4.1 + §4.2 — the forward GL direction uses
§4.1 to feed `⊆` and §4.2 to feed `⊇`, and the reverse GL direction
uses the same two inclusions with the roles swapped.

---

## 5. Conclusion

`{0, 1} = [0, 1]` as subsets of `N`, with the equality provable from the
Peano axioms alone — no further structure on `(N, +)` is required.

**Structure of the proof GL needs to reconstruct.**

1. Two instances of reflexivity of `≤` via `k = 0` in `a + 0 = a`
   (one for `x = 0 ≤ 0` and one for `x = 1 ≤ 1`).
2. Two instances of the fact `0 + k = k` (P1), used both to check
   `0 ≤ x` for `x ∈ {0, 1}` and to collapse the premise
   `0 + k = p` into `p = k` in the reverse direction.
3. One use of the successor split (P5) on `p`, giving the base case
   `p = 0` directly and reducing the recursive case to `p' + l = 0`.
4. One use of the sum-zero split (P6), which closes the recursive case
   by forcing `p' = 0` and hence `p = s(0) = 1`.

On the prover side these show up as:

- expansions of `EnumerationSet2`, `interval`, `preorder` into their
  body definitions;
- disintegrations of the two universally-quantified bodies of
  `EnumerationSet2` and `interval` under the hypothetical validity
  introduced by the head;
- `in3` / `preorder` chasing through `overallHashMemory` for the
  `0 + 0 = 0`, `0 + 1 = 1`, `1 + 0 = 1` additive facts;
- one case-split / induction step for the successor decomposition of
  `p` in §4.1 (⊇).

Steps 1–3 above should terminate inside the direct-proof phase.
Step 4 — the `a + b = 0 ⟹ a = 0 ∧ b = 0` lemma — is the subgoal the
`(EnumerationSet2[2,6,9])` LB has to close in the `hashburst_trace.txt`
dump, and it is the candidate handle for the current investigation of
why the `(interval[1,4,2,6,9])` goal in `toBeProved` does not get
retired.
