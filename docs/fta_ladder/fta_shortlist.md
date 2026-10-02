<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# FTA shortlist — nonempty ordered factorization via the least divisor

> **Status:** CORRECTED MATHEMATICAL SHORTLIST; MPL formalization and new GL
> proof runs remain separate. All entries A1–E6 are preserved verbatim.
> E6, F1, F2, F5 and the conditional form of F3 are proved and certified, with
> unique Fold values; E1 is retained for consistency, with its ground instances
> deferred. The other retained entries are established.
> Every sequence in the revised route is nonempty. Proofs stop at singletons.
> Existence selects the least nontrivial divisor and appends it to construct
> a nonincreasing factorization directly. No empty product, recursive definition,
> general concatenation, rearrangement relation, or sorting theorem is required.

**Target.** Every natural number `n ≥ 2` has exactly one nonincreasing,
nonempty sequence of primes with product `n`, where uniqueness means the
same length and the same entry at each position. No factorization of `1`
is constructed or required.

## Conventions and representation

Scalar variables range over ℕ (Peano). Sequence variables range over the
canonical nonempty finite sequences specified below. No subtraction or
numerical-division operator occurs in theorem statements: `a ≤ b` means
`∃k: a + k = b`, and `d | n` means `∃k: d·k = n`. The symbol `s` is successor;
`a < b` abbreviates `s(a) ≤ b`; `n ≥ 2` abbreviates `2 ≤ n`; and
`p ∤ a` abbreviates `¬(p | a)`.

**Canonical indexing.** A sequence `P` has positive length `|P|` and is a
finite function on exactly the positions `0,…,m`, with values in ℕ, where
`m+1=|P|`. Thus `P(0)` is always defined; the singleton has endpoint zero
and length one. Write `last(P)=P(m)` for the entry at that endpoint. This is
an abbreviation for the existing indexed value, not a new primitive operation;
MPL carries the endpoint and value as explicit variables. A proper prefix is
formed only at length at least two and retains the original zero-based indices.
Length changes use witness form, for example `|R| + 1 = |P|`.

The existing representation is `sequence[N,+,zero,m,P]` with `m∈N`. Here
`zero` denotes the theory's actual object for zero. These MPL-shaped
expressions are schematic, not ready-to-copy theorem rows. Numeric MPL
tokens are variable names, not automatically numeral literals. The endpoint
parameter is the last index, not the length; no primitive length operation
is assumed. All FTA sequences use this nonempty zero-based domain.

Membership is the direct, nonrecursive property

```text
x ∈ P :⟺ ∃i: 0 ≤ i < |P| ∧ P(i) = x.
```

Singleton `(a)` has domain `[0,0]` and value `a`. For nonempty `P`,
`append(P,a)` has length `|P|+1`, retains `P(i)` at every position
`0≤i<|P|`, and has value `a` at the new last position `|P|`. These equations
specify a finite graph directly; they do not refer to `append` in its own
definition.
F1 and F5 certify that the corresponding constructions exist. There is no binary
concatenation operation in the required library, and no empty operand.

**Sequence agreement is pointwise.** Use a separate mathematical notation:

```text
P ≡ Q :⟺ |P| = |Q| ∧ (∀i: 0 ≤ i < |P| ⟹ P(i) = Q(i)).
```

Scalar `=` continues to mean equality of natural-number values. Sequence
agreement `≡` is expanded into the displayed property in MPL, or given a
nonrecursive public predicate with precisely that meaning. It is not
translated as scalar `(=[P,Q])`: the current FTA equality ports have type
`(1)`, whereas sequence objects have type `P(x(1)(1))`.

No equality of allocation or witness identifiers, or general function-object
equality theorem, is required. Singleton and append notation identify their
pointwise sequence, independently of the particular graph witness. F13
supplies the agreement and property-transfer bridges; substitution into
sequence properties must use those bridges, not assume scalar equality.

**Existing product notation.** Keep `Π(P)` and the existing Fold relation.
For endpoint `m` with `m+1=|P|`, the schematic interpretation of `Π(P)=z` is
`fold[N,s,+,*,P,zero,m,z]`, with the sequence and nonempty-domain guards above.
Addition constructs the index interval, multiplication accumulates the
entries, and the first entry supplies the initial value. The Fold signature
is unchanged. No second product predicate or replacement accumulator
definition is introduced.

The total-operation notation presupposes existence and unique value on this
domain. The following obligations are explicit work in the existing library,
even though they are not additional numbered FTA entries:

| Required library result | Role and evidence boundary |
|---|---|
| Singleton, append, and nonempty proper-prefix graph construction | F1 and F5 are certified, including exact domains, totality and unique entry values. The append definition supplies the indexed bridges, and set-definition reformulation supplies the shared graph witness. F4 reuses the proved Gauss `limitSequence` restriction to the preceding endpoint. |
| Product existence for every guarded nonempty sequence | Must be supplied by an established theorem or proved using the existing Fold relation. The saved conditional product tests do not establish this general result. |
| Unique Fold value for a fixed guarded sequence | Needed to read Fold as `Π`. Proved and certified: two Fold values of one sequence over one range are equal (supplied row 89). |
| Fold transfer between pointwise agreeing sequences | Needed for the product clause of F13; retain both sequences' domain guards. |
| Bounded induction statements sufficient for G3 and I4 | Must be expressed and proved from the existing natural-number induction principle, rather than assumed to be automatically available. |

The current Fold ledger records the tested product statements as conditional
value results with Fold in the premise. They are useful evidence for the
representation, not a certificate of all the library results above.

**A noncircular product-library proof order.** Construct singleton graphs
and their Fold witnesses first. From an existing Fold witness `g` for `P`,
let `m` be its endpoint, `L=m+1=|P|`, and `z=g(m)` its final value.
Construct a witness `h` for `append(P,a)` by the direct graph equations
`h(i)=g(i)` for `0≤i≤m` and `h(L)=z·a`. All previous successor equations
are retained. The sole new transition is from m to L, and its equation is
exactly `h(L)=g(m)·a`. The starting value is unchanged. This extends an
existing witness; it neither shifts its indices nor recursively defines a
new product. State the construction conditionally on the supplied Fold value z.
Prove unique Fold values by induction along the guarded interval, and prove
Fold transfer using pointwise agreement. Positive-length induction with F4,
the singleton witness and this conditional witness-extension construction
then establishes product existence for every sequence. Only afterwards use
the unconditional function-style equations F2/F3 throughout the ladder.
The existence proof must use the conditional Fold construction, not assume
the totality already implicit in a displayed `Π` equation.

**Induction is a proof principle, not a recursive definition.** The native
prover supplies zero and successor induction. Sequence arguments can use
the statement universal over all sequences of length `k+1`; their meaningful
base is a singleton and their step has a nonempty proper prefix of length `k+1`.
All auxiliary parameters stay universally quantified where the induction
hypothesis must change them.

For strong-induction sketches, prove the bounded formula
`U(k):⟺ ∀m: m≤k ⟹ T(m)` by ordinary induction. Its successor step uses
`U(k)` at every `m<k+1`. For I4 take `T(m)` to be its implication with guard
`m≥2`; for G3 take `T(b)` to be its statement universally in `a`. The zero
cases outside the guards are vacuous propositions, not empty sequences or
products. I5's length induction must remain universal in the competing
sequence `Q`. These are explicit mathematical formulations to encode in MPL;
the document does not claim the current scheduler infers the strengthening.

**Given externally:** the proved Peano corpus — commutativity and
associativity of `+` and `·`, distributivity, the successor and zero laws
(`a+0=a`, `a+s(b)=s(a+b)`, `a·0=0`, `a·s(b)=a·b+a`), injectivity of `s`,
`s(a)≠0`, and the established incubator facts. E1's retained ground instances
are not silently added to that proved input set.

There are **87 numbered entries**: the 57 unchanged entries A1–E6 and 30
entries F1–I6. F2 remains statement 59; E6 remains statement 57. The catalog
order is not a proof order, and one entry may require several MPL conjectures.

---

## Part A — zero, successor, and the order relation

- **A1 (1).** `0 ≤ a`. *(Witness `a`.)*
- **A2 (2) (reflexivity).** `a ≤ a`. *(Witness `0`.)*
- **A3 (3).** `a ≤ s(a)`. *(Witness `1`.)*
- **A4 (4).** `s(a) = a + 1`.
- **A5 (5).** `a + b = 0 ⟹ a = 0 ∧ b = 0`.
- **A6 (6).** `a ≤ 0 ⟹ a = 0`. *(A5.)*
- **A7 (7) (transitivity).** `a ≤ b ∧ b ≤ c ⟹ a ≤ c`. *(Witness addition; associativity.)*
- **A8 (8) (antisymmetry).** `a ≤ b ∧ b ≤ a ⟹ a = b`. *(A5 on the two witnesses; already exercised by rung 2.1's machinery.)*
- **A9 (9) (monotone `+`).** `a ≤ b ⟹ a + c ≤ b + c`.
- **A10 (10) (additive order reflection).** `a + c ≤ b + c ⟹ a ≤ b`.
- **A11 (11) (additive cancellation).** `a + c = b + c ⟹ a = b`. *(Induction on `c`; `s` injective.)*
- **A12 (12) (strict implies weak).** `a < b ⟹ a ≤ b`. *(A3 + A7.)*
- **A13 (13) (strict implies unequal).** `a < b ⟹ a ≠ b`.
- **A14 (14) (weak + unequal implies strict).** `a ≤ b ∧ a ≠ b ⟹ a < b`.
- **A15 (15) (totality).** `a ≤ b ∨ b ≤ a`. *(Induction + case split.)*
- **A16 (16) (trichotomy).** `a < b ∨ a = b ∨ b < a`. *(A14 + A15.)*
- **A17 (17) (below successor).** `a < s(b) ⟺ a ≤ b`.
- **A18 (18) (successor reflection).** `s(a) ≤ s(b) ⟹ a ≤ b`.
- **A19 (19) (no gap).** `¬(a < k ∧ k < s(a))`. *(A17 + A8/A13.)*
- **A20 (20) (positivity forms).** `a ≠ 0 ⟺ 1 ≤ a`; `a ≠ 0 ∧ a ≠ 1 ⟺ 2 ≤ a`.
- **A21 (21) (irreflexivity of `<`).** `¬(a < a)`.
- **A22 (22) (asymmetry of `<`).** `a < b ⟹ ¬(b < a)`.

## Part B — multiplication and order

- **B1 (23) (no zero divisors).** `a·b = 0 ⟹ a = 0 ∨ b = 0`. *(Case split zero/successor.)*
- **B2 (24) (positive product).** `a ≥ 1 ∧ b ≥ 1 ⟹ a·b ≥ 1`.
- **B3 (25) (unit product).** `a·b = 1 ⟹ a = 1 ∧ b = 1`. *(B1 + case analysis.)*
- **B4 (26) (monotone `·`).** `a ≤ b ⟹ a·c ≤ b·c`. *(Distributivity over the witness.)*
- **B5 (27) (strict monotone `·`).** `a < b ∧ c ≥ 1 ⟹ a·c < b·c`.
- **B6 (28) (growth).** `b ≥ 1 ⟹ a ≤ a·b`.
- **B7 (29) (strict growth).** `a ≥ 1 ∧ b ≥ 2 ⟹ a < a·b`.
- **B8 (30) (multiplicative cancellation).** `c ≥ 1 ∧ a·c = b·c ⟹ a = b`.
 *(A15/A16 + B5. The division-free substitute for dividing — load-bearing
 for the whole uniqueness half.)*
- **B9 (31) (multiplicative order reflection).** `c ≥ 1 ∧ a·c ≤ b·c ⟹ a ≤ b`.
 *(Companion of B8; needed wherever a multiple is "subtracted".)*
- **B10 (32) (strict reflection).** `a·c < b·c ⟹ a < b`.
- **B11 (33) (product ≥ 2 split).** `a·b ≥ 2 ⟹ a ≥ 2 ∨ b ≥ 2`.

## Part C — divisibility

Definition: `d | n:⟺ ∃k: d·k = n`.

- **C1 (34) (one divides).** `1 | n`. *(Witness `n`.)*
- **C2 (35) (self divides).** `n | n`. *(Witness `1`.)*
- **C3 (36) (everything divides zero).** `n | 0`. *(Witness `0`.)*
- **C4 (37) (zero divides only zero).** `0 | n ⟹ n = 0`.
- **C5 (38) (transitivity).** `d | a ∧ a | b ⟹ d | b`. *(Witness product; associativity.)*
- **C6 (39) (sum closure).** `d | a ∧ d | b ⟹ d | (a + b)`. *(Distributivity.)*
- **C7 (40) (multiple closure).** `d | a ⟹ d | (a·b)` and `d | (b·a)`. *(Associativity, commutativity.)*
- **C8 (41) (difference closure, witness form).** `d | a ∧ d | (a + b) ⟹ d | b`.
 *(From `d·x = a`, `d·y = a + b`: B9 gives `x ≤ y`; the witness `z` with
 `x + z = y` yields `b = d·z` by distributivity + A11. The division-free
 "subtract a multiple" step — load-bearing for D2, G1, G5.)*
- **C9 (42) (divisor bound).** `d | n ∧ n ≥ 1 ⟹ d ≤ n`. *(B6 on the witness.)*
- **C10 (43) (divisor positive).** `d | n ∧ n ≥ 1 ⟹ d ≥ 1`. *(C4 contrapositive.)*
- **C11 (44) (cofactor divides).** `n = d·k ⟹ k | n`. *(Commutativity.)*
- **C12 (45) (cofactor bound).** `n = d·k ∧ d ≥ 2 ∧ n ≥ 1 ⟹ k < n`. *(B7. Feeds every descent.)*
- **C13 (46) (divisibility antisymmetry).** `a | b ∧ b | a ⟹ a = b`. *(C9 + A8, zero cases via C3/C4.)*
- **C14 (47) (scaling).** `c ≥ 1 ⟹ (d | n ⟺ c·d | c·n)`. *(B8 for the reverse direction.)*
- **C15 (48) (product of divisors).** `d | a ∧ e | b ⟹ d·e | a·b`.

## Part D — division with remainder

- **D1 (49) (existence).** `∀a ∀d ≥ 1 ∃q, r: a = q·d + r ∧ r < d`.
 *(Induction on `a`; the step case-splits on `s(r) = d` — remainder rolls
 over to `(s(q), 0)` — versus `s(r) < d`.)*
- **D2 (50) (uniqueness).** `q·d + r = q'·d + r' ∧ r < d ∧ r' < d ⟹ q = q' ∧ r = r'`.
 *(A15 on `q, q'`; C8-style witness arithmetic; A11.)*
- **D3 (51) (zero remainder is divisibility).** `a = q·d + r ∧ r < d ⟹ (d | a ⟺ r = 0)`.
 *(⟸ trivial; ⟹ via D2 against the divisibility witness.)*

## Part E — primes

Definition: `prime(p):⟺ p ≥ 2 ∧ ∀d: (d | p ⟹ d = 1 ∨ d = p)`.

- **E1 (52) (ground instances).** `prime(2)`, `prime(3)`, `prime(5)`;
 `¬prime(0)`, `¬prime(1)`, `¬prime(4)`, `¬prime(6)`.
 *(Incubator-style concrete anchors for the definition.)*
- **E2 (53) (primes are ≥ 2).** `prime(p) ⟹ p ≥ 2`. *(Definitional projection.)*
- **E3 (54) (nontrivial divisor is the prime).** `prime(p) ∧ d | p ∧ d ≥ 2 ⟹ d = p`.
- **E4 (55) (prime divides prime).** `prime(p) ∧ prime(q) ∧ p | q ⟹ p = q`. *(E3 + E2.)*
- **E5 (56) (prime coprimality).** `prime(p) ∧ p ∤ a ∧ d | p ∧ d | a ⟹ d = 1`.
 *(E3: `d ≥ 2` would force `d = p`, contradicting `p ∤ a`.)*
- **E6 (57) (composite split).** `n ≥ 2 ∧ ¬prime(n) ⟹ ∃a, b: n = a·b ∧ 2 ≤ a < n ∧ 2 ≤ b < n`.
 *(Negate the definition; C9/C12 give the bounds, C11 the cofactor.)*



## Part F — nonempty finite sequences and products

Every sequence variable below has the canonical nonempty zero-based domain from the
conventions. Define only the ordinary property

```text
primeseq(P) :⟺ ∀x: x ∈ P ⟹ prime(x).
```

- **F1 (58) (singleton construction and indexing).**
 `∀a ∃P: |P|=1 ∧ P(0)=a`.
 Write its pointwise sequence `(a)`.
 *(Construct the graph consisting of the pair `(0,a)` and prove its domain,
 totality and unique value. The singleton-to-interval and singleton-relation-to-sequence implications generate existence statements through set-definition reformulation. Any two such sequences agree by F13.)*

- **F2 (59) (singleton product).** `Π((a))=a`.
 *(The singleton's own value graph supplies a Fold witness. Its first and
 last positions coincide; there is no successor pair inside the interval.
 Unique Fold value gives the displayed equation. No empty base is used.)*

- **F3 (60) (append product).**
 `Π(append(P,a))=Π(P)·a`.
 *(Use the existing Fold witness for P and extend it by one pair as described
 in the conventions. Previous indices and accumulator values are unchanged;
 the new final value is the old final value multiplied by a.
 Unique Fold value gives the equality. The conditional witness construction
 is available before general Fold totality, so the library proof is not
 circular. No general concatenation-product theorem is required.)*
 **Proved and Lean-certified in conditional form.** For a sequence R on
 `[0,s(m)]`, a Fold value p of R up to m gives the Fold value `p·R(s(m))` up
 to `s(m)` (supplied row 81); conversely, Fold values p up to m and w up to
 `s(m)` satisfy `w = p·R(s(m))` (row 87). Two Fold values of one sequence over
 one range are equal (row 89), and restricting a sequence keeps its Fold value
 over the shorter range (rows 88 and 90). That every nonempty sequence has a
 Fold value (general product existence) is not yet proved.

- **F4 (61) (singleton-or-longer decomposition).**
 `|P|=1 ⟹ P ≡ (last(P))`;
 `|P|≥2 ⟹ ∃R:`
 ` P ≡ append(R,last(P)) ∧ |R|+1=|P| ∧ 1≤|R|<|P|`.
 *(In the longer case the nonzero endpoint has the form s(k). Reuse the
 proved Gauss `limitSequence` construction to restrict P to `[0,k]`, obtaining
 the nonempty prefix R with unchanged indices and values. F5's indexed
 equations recover pointwise agreement with P. Retain exact domains and
 unique entry values. In the singleton case `last(P)=P(0)`; no proper prefix
 is constructed.)*

- **F5 (62) (append construction and indexed bridges).**
 **Proved and Lean-certified.** The `appendToSequence` graph is a sequence
 on the extended interval, and constructive reformulation supplies its
 existence. Its defining equations are the indexed bridges below.
 For every natural number `a` and nonempty sequence `P`, there is `R` with
 `|R|=|P|+1`, `R(|P|)=a`, and
 `∀i: 0≤i<|P| ⟹ R(i)=P(i)`.
 Write its pointwise sequence `append(P,a)`; in particular its length is
 at least two, and `last(append(P,a))=a`.
 *(Adjoin the pair `(|P|,a)` to P's graph. Its new position is disjoint from
 the old domain `[0,m]`, where `m+1=|P|`. A14/A17 separate every position in
 the extended interval into an old position or the new endpoint. Establish
 the exact domain, totality and unique values, and then pointwise uniqueness.
 This constructs only a one-element append, not arbitrary concatenation.)*

- **F6 (63) (membership bridges).**
 `x ∈ append(P,a) ⟺ x∈P ∨ x=a`;
 `x ∈ (a) ⟺ x=a`;
 `last(P) ∈ P`.
 *(Use the indexed membership definition, F1 and F5. The last-position
 witness is valid because every sequence here has positive length.)*

- **F7 (64) (prime singleton and append closure).**
 `primeseq((p)) ⟺ prime(p)`;
 `primeseq(append(P,p)) ⟺ primeseq(P) ∧ prime(p)`.
 *(Unfold primeseq and use F6. No empty-sequence clause is present.)*

- **F8 (65) (a member divides the sequence product).**
 `x∈P ⟹ x | Π(P)`.
 *(Induct on positive length, universally in the sequence and member.
 F4 separates singleton and longer cases; F13 transfers properties across
 pointwise agreement. The singleton uses F6/F2/C2. In the longer case F6
 gives membership in the prefix or `x=last(P)`. The induction hypothesis
 and C7 handle the prefix case; C2/C7 handle the last entry. F10 supplies
 the product decomposition in both cases.)*

- **F9 (66) (nonempty prime-product lower bound).**
 `primeseq(P) ⟹ Π(P)≥2`.
 *(Induct on positive length. A singleton uses F7/E2/F2. In the longer
 case F4/F11 give a nonempty prime prefix R and prime last entry p. The
 induction hypothesis gives `Π(R)≥2`; E2 gives `p≥1`. B6 gives
 `Π(R)≤Π(R)·p`; combine F10 and A7. The product is neither zero nor one.)*

- **F10 (67) (product of an existing prefix/last-entry decomposition).**
 `P ≡ append(R,a) ⟹ Π(P)=Π(R)·a`.
 *(F13's product transfer followed by F3. This directly matches the
 decomposition returned by F4, with R nonempty.)*

- **F11 (68) (primality of an existing prefix/last-entry decomposition).**
 `P ≡ append(R,p) ⟹`
 ` (primeseq(P) ⟺ primeseq(R) ∧ prime(p))`.
 *(F13's primality transfer and F7. This is the longer-sequence bridge;
 the singleton uses F7 directly.)*

- **F12 (69) (prime-sequence members are at least two).**
 `primeseq(P) ∧ x∈P ⟹ x≥2`.
 *(Unfold primeseq and use E2.)*

- **F13 (70) (pointwise agreement and property transfer).**
 Agreement `≡` is reflexive, symmetric and transitive. In addition:
 `a=b ⟹ (a) ≡ (b)`;
 `a=b ∧ P≡Q ⟹ append(P,a) ≡ append(Q,b)`;
 `P≡Q ⟹ last(P)=last(Q)`;
 `P≡Q ⟹ Π(P)=Π(Q)`;
 `P≡Q ⟹ (x∈P ⟺ x∈Q)`;
 `P≡Q ⟹ (primeseq(P) ⟺ primeseq(Q))`;
 `P≡Q ⟹ (ordered(P) ⟺ ordered(Q))`.
 *(Use the explicit same-length/entry formula and the graphs from F1/F5.
 Equal lengths give equal endpoints by A11, so entry agreement gives equal
 last values. Product transfer uses the Fold congruence result established
 in the foundation stage; the other properties transfer by their definitions,
 with ordered defined in Part H.
 Catalog position is not proof order. Each direction needed in MPL is a
 separate conjecture. No scalar equality between sequence objects is used.)*

- **F14 (71) (a longer prime sequence exceeds its last entry).**
 `primeseq(P) ∧ |P|≥2 ⟹ last(P)<Π(P)`.
 *(F4/F11 give nonempty prime prefix R and prime last entry `p=last(P)`.
 E2 gives `p≥1`, F9 gives `Π(R)≥2`, and B7 gives `p<p·Π(R)`.
 Commutativity and F10 give the displayed strict inequality.
 This closes the singleton-versus-longer uniqueness cases directly.)*

## Part G — Euclid's lemma via symmetric Bézout

Define the ordinary properties

```text
coprime(a,b) :⟺ ∀d: d | a ∧ d | b ⟹ d=1.
bez1(a,b) :⟺ ∃x,y: (a·x=b·y+1) ∨ (b·y=a·x+1).
```

Neither is recursive. The second is the subtraction-free natural-number
form of a Bézout identity with coefficient difference ±1. Zero coefficients
remain allowed.

- **G1 (72) (coprimality survives a Euclidean step).**
 `coprime(a,b) ∧ a=q·b+r ⟹ coprime(b,r)`.
 *(If `d|b` and `d|r`, C7/C6 give `d|a`; the original coprimality gives
 `d=1`. No new greatest-common-divisor construction is needed.)*

- **G2 (73) (Bézout transport through a Euclidean step).**
 `a=q·b+r ∧ bez1(b,r) ⟹ bez1(a,b)`.
 First prove the two arithmetic implications separately:
 `a=q·b+r ∧ b·x=r·y+1 ⟹ b·(x+q·y)=a·y+1`;
 `a=q·b+r ∧ r·y=b·x+1 ⟹ a·y=b·(q·y+x)+1`.
 *(Distributivity, associativity and commutativity prove both identities.
 Then discharge the two bez1 cases. The first uses output witnesses
 `(y,x+q·y)` in the second orientation; the second uses `(y,q·y+x)` in the
 first orientation. Both witnesses of each identity stay in one scope.)*

- **G3 (74) (Bézout for positive coprime naturals).**
 `coprime(a,b) ∧ a≥1 ∧ b≥1 ⟹ bez1(a,b)`.
 *(Strong induction on b, universally in a, implemented by the bounded
 statement in the conventions. D1 gives `a=q·b+r`, `r<b`. If `r=0`,
 then `b|a` and `b|b`, so coprimality gives `b=1`; use
 `1·1=a·0+1`. If `r≠0`, then `r≥1`; G1 gives coprimality of `(b,r)`.
 Apply the induction hypothesis at r with first argument b, and lift by G2.
 The changed first argument must not be fixed outside the hypothesis.)*

- **G4 (75) (Euclid's lemma).**
 `prime(p) ∧ p | a·b ⟹ p | a ∨ p | b`.
 *(If `p∤a`, E5 gives `coprime(p,a)`. C3 shows `a≠0`, hence `a≥1`;
 E2 gives `p≥1`. Apply G3. Multiply either Bézout orientation by b;
 `a·b` and the explicit p-multiple are divisible by p, and C8 isolates b.)*

- **G5 (76) (prime divides a nonempty sequence product).**
 `prime(p) ∧ p | Π(P) ⟹ ∃x: x∈P ∧ p|x`.
 *(Induct on positive length, universally in p and P. The singleton case
 uses F2/F6 with its sole entry. In the longer case use F4/F10 and G4:
 p divides the prefix product or the last entry. Use the induction
 hypothesis in the first case; choose the last entry in the second.
 F6/F13 lift membership to P.)*

- **G6 (77) (prime occurs in a prime sequence it divides).**
 `prime(p) ∧ primeseq(P) ∧ p | Π(P) ⟹ p∈P`.
 *(G5 gives a member x divisible by p. That member is prime; E4 gives
 `p=x`. Multiplicities are handled later by pointwise sequence agreement,
 not by treating membership as sequence equality.)*

## Part H — order directly on indexed positions

Define the ordinary property

```text
ordered(P) :⟺
    ∀i,j: 0≤i ∧ i<j ∧ j<|P| ⟹ P(j)≤P(i).
```

Thus entries are nonincreasing; equal adjacent primes are allowed. The last
entry is a minimum. This uses only indexed entries. No self-reference,
sequence splitting around an arbitrary member, or sorting construction occurs.

- **H1 (78) (a singleton is ordered).** `ordered((a))`.
 *(There are no indices with `0≤i<j<1`.)*

- **H2 (79) (a nonempty prefix stays ordered).**
 `ordered(append(P,a)) ⟹ ordered(P)`.
 *(Use F5 at the same indices. For an existing decomposition
 `Q≡append(P,a)`, first use F13 to transfer orderedness.)*

- **H3 (80) (the last entry bounds every member).**
 `ordered(P) ∧ x∈P ⟹ last(P)≤x`.
 *(Write m for the endpoint and choose a membership index i. A17 gives
 `i≤m`. If `i=m`, use reflexivity. Otherwise A14 gives `i<m`, and the
 nonincreasing order property gives `P(m)≤P(i)`. This includes singletons.)*

- **H4 (81) (append below the last entry preserves order).**
 `ordered(P) ∧ a≤last(P) ⟹ ordered(append(P,a))`.
 *(For positions `i<j` in the new sequence, either both are old positions
 and ordered(P) applies, or j is the new endpoint. In the latter case H3
 and transitivity give `a≤last(P)≤P(i)`. F5 supplies the entry values.
 Callers supply just one scalar comparison, not a universal lower-bound
 premise. The prefix is nonempty.)*

## Part I — least-divisor construction of the canonical factorization

The construction chooses the least nontrivial divisor at each descent.
It appends that divisor after the cofactor's nonincreasing factorization.

- **I1 (82) (bounded least divisor).**
 `(∃d: 2≤d ∧ d≤k ∧ d|n) ⟹`
 `∃p: 2≤p ∧ p≤k ∧ p|n ∧`
 ` (∀d: 2≤d ∧ d≤k ∧ d|n ⟹ p≤d)`.
 *(Ordinary induction on k. At k=0 no qualifying divisor exists. At the
 successor step, use the old minimum if there is a candidate at or below k;
 every additional candidate is k+1 and lies above it. Otherwise the premise
 forces the candidate to be k+1. A14/A17 rule out a smaller candidate.
 No recursively defined least-divisor function is introduced.)*

- **I2 (83) (least nontrivial divisor exists).**
 `n≥2 ⟹ ∃p: p|n ∧ p≥2 ∧ (∀d: d|n ∧ d≥2 ⟹ p≤d)`.
 *(Apply I1 with k=n. C2 provides candidate n. C9 bounds every divisor
 of the positive n by n, so the bounded minimum is the required minimum.)*

- **I3 (84) (the least nontrivial divisor is prime).**
 `p|n ∧ p≥2 ∧ (∀d: d|n ∧ d≥2 ⟹ p≤d) ⟹ prime(p)`.
 *(Retain the already proved E6 route. If p is not prime, E6 gives
 `p=a·b` with `2≤a<p`. Thus `a|p`; C5 gives `a|n`. Minimality gives
 `p≤a`, contradicting `a<p` by A12/A8/A13. The cofactor b and its bounds
 remain in E6's shared witness bundle, although this consumer needs only a.
 Removing the E6 dependency alone is not a simplification of GL's work;
 no alternative primality proof is required by this revision.)*

- **I4 (85) (nonempty ordered prime-factorization existence).**
 `n≥2 ⟹ ∃P: |P|≥1 ∧ primeseq(P) ∧ ordered(P) ∧ Π(P)=n`.
 *(Strong induction on n, implemented with the bounded statement from
 the conventions. I2 chooses the least nontrivial divisor p and I3 gives
 primality. Choose m with `n=p·m`. C11/C10 give `m|n` and `m≥1`.

 **Singleton case, m=1:** then n=p. F1 constructs `(p)`; F7/H1/F2 give
 its primality, order and product n. No factorization of one is requested.

 **Longer case, m≠1:** A20 gives `m≥2` and C12 gives `m<n`. The induction
 hypothesis supplies a nonempty ordered prime sequence Q with product m.
 Put `q=last(Q)`. F6 gives `q∈Q`; F12 gives `q≥2`; F8 gives `q|m`.
 C11/C5 give `q|n`, so minimality gives the single comparison `p≤q`.
 Construct `append(Q,p)` by F5. H4 gives order from `p≤last(Q)`, F7 gives
 primality, and F3 gives product `m·p=p·m=n` by commutativity. F5 gives
 positive length. No universal comparison with all prefix members is needed.)*

- **I5 (86) (uniqueness of nonempty ordered prime factorization).**
 `primeseq(P) ∧ ordered(P) ∧ primeseq(Q) ∧ ordered(Q) ∧ Π(P)=Π(Q)`
 `⟹ P≡Q`.

 First establish and reuse the separate scalar bridge with the same
 premises and conclusion `last(P)=last(Q)`. It needs no induction or
 decomposition: put `p=last(P)`, `q=last(Q)`; F6 gives membership and hence
 primality.
 F8 and product equality give `p|Π(Q)`; G6/H3 give `q≤p`.
 Symmetrically `p≤q`; A8 gives `p=q`.

 Now induct on the positive length of P, universally in Q. Do not assume
 the two lengths equal. Apply the scalar bridge before any decomposition.

 **Both singleton:** F4 gives `P≡(p)` and `Q≡(q)`; `p=q` and F13 give
 `P≡Q`.

 **Exactly one singleton:** if P is the singleton, F2/F13 give `Π(P)=p`.
 The longer Q satisfies `q<Π(Q)` by F14. Product equality and `p=q`
 contradict A21. Exchange the roles for the other case. No singleton prefix
 is constructed and no product of an empty sequence occurs.

 **Both longer:** F4 gives `P≡append(P′,p)` and `Q≡append(Q′,q)`, with
 nonempty prefixes and `|P′|+1=|P|`. H2/F11/F13 give ordered prime prefixes.
 F10 and `p=q` give `Π(P′)·p=Π(Q′)·p`. E2 and B8 cancel the common final
 factor p.
 The induction hypothesis gives `P′≡Q′`. Append congruence and transitivity
 from F13 give `P≡Q`.

- **I6 (87) (Fundamental Theorem of Arithmetic — nonempty canonical form).**
 `∀n≥2 ∃P:`
 ` |P|≥1 ∧ primeseq(P) ∧ ordered(P) ∧ Π(P)=n ∧`
 ` (∀Q: |Q|≥1 ∧ primeseq(Q) ∧ ordered(Q) ∧ Π(Q)=n ⟹ Q≡P)`.
 *(Choose P by I4 and use I5 for each competing Q with the same product.
 Agreement fixes length, each indexed prime, and every multiplicity.
 Positive-length conditions are displayed to emphasize the target, although
 the sequence convention already supplies them.)*

---

## Dependency guide and proof order

1. **Existing arithmetic:** retain A1–E6 unchanged, including the proved E6.
 E1 remains in the catalog for consistency and is not a prerequisite of
 the general factorization argument.
2. **Sequence and Fold foundations:** construct the finite graphs needed by
 F1/F5/F4 and establish pointwise agreement properties. Establish the
 singleton and conditional append Fold witnesses, unique Fold value,
 transfer under agreement and general positive-length product existence
 in the order described in the conventions. F2/F3 then apply as equations.
3. **Small sequence bridges:** F5 gives F6; F6 gives F7; Fold transfer/F3 give
 F10; agreement/F7 give F11; primality gives F12. F13's order clause uses
 the direct definition in Part H and does not depend on an order theorem.
4. **Inductive sequence facts:** F4/F6/F10/C2/C7 give F8; F4/F7/F11/F10/E2/B6
 give F9; F4/F9/F11/F10/E2/B7 and commutativity give F14.
5. **Ordering:** prove H1–H3 from the indexed definitions and F5/F6, then H4.
6. **Least divisor and existence:** prove I1 by ordinary induction, I2 using
 C2/C9, and I3 using the established E6/C5. Encode the bounded induction
 statement for I4 and use only the prefix's last entry in its append step.
7. **Euclid chain:** G1, the two G2 arithmetic implications, and D1 give G3
 under the explicit bounded induction formulation. E5/G3/C7/C8 give G4;
 G4/F2/F4/F6/F10 give G5; G5/E4 give G6. This chain is independent of I4.
8. **Uniqueness:** prove I5's scalar equal-last-entry bridge from F6/F8/G6/H3/A8;
 handle boundaries with F2/F4/F14; cancel and reconstruct the longer case
 with F10/B8/H2/F11/F13. Finally combine I4 and I5 into I6.

## MPL authoring obligations

- Preserve exact zero-based sequence-domain guards, the natural endpoint
 (which may be zero), positive length and the
 existing eight-slot Fold signature. Values `P(i)` are represented by
 `in2` with explicit value variables, not inserted as terms in MPL slots.
- Expand pointwise agreement into scalar length/entry facts, or use an
 explicitly defined nonrecursive property. Do not use scalar equality on
 function objects. Carry the relevant lengths and endpoints as explicit
 parameters. Expand `last(P)` using the endpoint and an `in2` value variable.
- Prove graph existence, totality and unique entry values for the needed
 constructions. A well-defined graph body is not by itself all those proofs.
- Separate equivalence directions and independent conjunctive conclusions
 into their own conjectures. Do not split an existential witness bundle
 into unrelated existences: the same P must satisfy domain, primality,
 order and product, and the same Bézout witnesses must satisfy its identity.
- Use at most one configured output slot per public predicate. Additional
 witnesses remain bound in the common property as needed. A shorter
 mathematical notation is not a claim that its exact MPL form is available.
- Record ordinary-induction versions of the bounded G3/I4 statements and
 retain changing parameters inside the induction hypothesis. Product,
 member and uniqueness inductions likewise quantify over all appropriate
 sequences of the chosen positive length.
- Track the proof of each required MPL row independently. The 87 catalog
 entries, supplied conjecture count and verified theorem count need not
 coincide. No new proof or runtime claim follows from this document edit.

## Clarifications for the unchanged arithmetic ladder

All bytes of the A1–E6 section are retained, including the existing proof
notes. One proof-note clarification is recorded here rather than changing
that established section: C8's sketch uses B9 on multiplication by d, which
requires `d≥1`. If `d=0`, C4 applied to `d|a` and `d|(a+b)` gives
`a=0` and `a+b=0`; A5 gives `b=0`, and C3 gives `d|b`. If `d≠0`, A20
gives the positivity required by B9 and the retained argument applies.
The C8 statement is unchanged.

E6 is established, not an outstanding prerequisite to prove again. I3 keeps
its proof through E6. A direct divisor-by-divisor proof is mathematically
possible, but removing that dependency alone does not justify replacing an
available theorem with another proof route. The simplifications adopted here
instead remove general concatenation and reduce I4's order obligation to one
comparison. Their runtime effect has not been measured.

## Changes in this correction

| Previous formulation | Corrected formulation |
|---|---|
| Blanket proved status for A1–E6 | E6 and the established ladder are acknowledged; E1 stays unchanged for consistency with its deferred status explicit. |
| General nonempty concatenation and its product law | Singleton, one-element append and nonempty proper prefix suffice; F3/F5 concern only append. |
| Shifted sequence and accumulator graphs | Append preserves all previous indices and Fold values, adding one final pair to each graph. |
| Ascending factors with the least divisor inserted first | Nonincreasing factors with the least divisor appended last; F1 remains zero-based. |
| Optional interpretation of sequence object equality | Explicit pointwise agreement `≡`, with F13 transfer bridges and scalar `=` reserved for values. |
| Product-library prerequisites stated abstractly | Exact existing Fold interface, explicit obligations, evidence boundary and a noncircular witness-based proof order. |
| Strong induction only named in proof sketches | Bounded statements explain its derivation from ordinary successor induction. |
| H4 requires a bound for every prefix member | H4 requires only `a≤last(P)` for an already ordered nonempty prefix. |
| I4 compares p with every prefix member | I4 compares p only with the last prefix prime. |
| G2 mixes arithmetic rearrangement with its disjunction | Two separate oriented arithmetic implications feed the shared Bézout transport. |
| I5 repeats its distinguished-entry comparison inside its proof | A separate reusable scalar equal-last-entry bridge precedes the boundary split. |
| F14 stated only for a constructed sequence | F14 compares the last entry of any prime sequence of length at least two with its product. |

The labels and global numbers A1–I6 are retained. F2 remains statement 59,
and I6 remains statement 87. Entries F10/F11 match an existing decomposition
directly and are small consumers of the basic construction/transfer lemmas.

## Scope and source references

The theorem is the canonical nonincreasing-sequence form of FTA for `n≥2`.
It fixes multiplicities through same-length pointwise agreement; ordinary
membership is never used as a replacement for sequence agreement. A separate
theorem about permuting an arbitrarily ordered factorization is outside this
route and is not a prerequisite.

No zero-length sequence, empty product, empty prefix, arbitrary member removal,
general concatenation, sorting construction, or factorization of one is used.
Arithmetic zero and one remain available, including zero remainders and
zero Bézout coefficients. The existing Fold uses successor equations for a
witness, but none of the new graph or property definitions is recursive.

Local sources checked for this revision (paths relative to the GL repository):

- `files/definitions/sequence.mpl`, `interval.mpl`, `fXY.mpl`, and `fold.mpl`:
 actual sequence/domain and product interfaces.
- `files/definitions/limitSequence.mpl` and the proved Gauss restriction
 implication with its generated existence: nonempty prefix restriction
 at the preceding endpoint, retaining the original indices.
- `files/config/ConfigFTA.json`, entries `=`, `sequence`, and `fold`:
 scalar equality types, sequence relation types and Fold slot order.
- `docs/agentic_swdd/03b_mpl_cookbook.md`: separate conjecture heads,
 shared existential scopes, explicit values and one-output predicates.
- `docs/agentic_swdd/10_pipeline/04_prover.md`, Induction, and
 `GL_Quick_VS/GL_Quick/src/prover.cpp`, `addTheoremToMemory`:
 ordinary zero/successor induction.
- `proof_export/certificate.py`, `_defined_output_totality`:
 constructive graph witnesses require an acyclic definition closure.


F1, F2, F5 and the conditional form of F3 are implemented and certified,
together with unique Fold values. General product existence, F4 and the rest
of the append-based route through I6 remain a mathematical and formalization
proposal. New conjectures, general Fold
prerequisite proofs and performance gains are not claimed to have been
implemented or verified by this document revision.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
