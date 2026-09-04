<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# FTA shortlist — the exhaustive lemma ladder for the shortcut

> **Status:** DRAFT for maintainer review. Human-language lemma list; MPL
> formalization follows after review. This document implements the strategy
> announced in the blog post *To Be or Not to Be* (2026-07-31): GL receives
> its previously proved Peano theorems externally in MPL, plus the list below
> as true conjectures, and must assemble the proofs autonomously in an order
> of its own making — up to and including FTA. Auto-discovery is deferred.
> The list is deliberately EXHAUSTIVE: every lemma a careful human proof
> walks through, at the granularity a human would actually cite.

**Conventions.** All variables range over ℕ (Peano). No subtraction and no
division anywhere — every statement is in witness form (`a ≤ b` means
`∃k: a + k = b`; `d | n` means `∃k: d·k = n`). `s` is successor; `a < b`
abbreviates `s(a) ≤ b`; `n ≥ 2` abbreviates `2 ≤ n`; `p ∤ a` abbreviates
`¬(p | a)`. `Π(P)` is the product of a finite sequence `P` (fold with `·`);
`⧺` is concatenation; `|P|` is length; `x ∈ P` is membership; `(a)` is the
singleton sequence.

**Given externally (not on the list):** the proved Peano corpus —
commutativity / associativity of `+` and `·`, distributivity, the successor
and zero laws (`a + 0 = a`, `a + s(b) = s(a + b)`, `a·0 = 0`,
`a·s(b) = a·b + a`), injectivity of `s`, `s(a) ≠ 0`, plus the incubator
ground facts. Overlaps between Part A/B/C below and the corpus move from
"conjecture" to "externally provided" at MPL-formalization time (Open
decision 4). New *definitions* (`|`, `prime`, sequences, `Π`, `∈`, `≈`)
enter as MPL definitions, not as conjectures.

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
- **E7 (58) (prime divisor exists).** `n ≥ 2 ⟹ ∃p: prime(p) ∧ p | n`.
 *(STRONG induction on `n`: prime → itself via C2; composite → E6, recurse
 on the smaller factor, lift with C5.)*
- **E8 (59) (least nontrivial divisor is prime — alternative route to E7).**
 `n ≥ 2 ∧ d | n ∧ d ≥ 2 ∧ (∀e: e | n ∧ e ≥ 2 ⟹ d ≤ e) ⟹ prime(d)`.
 *(Any proper divisor of `d` divides `n` (C5) and undercuts minimality.
 Listed as the well-ordering variant; E7's strong-induction route is
 primary.)*

## Part F — finite sequences and products

Definitions: finite sequences over ℕ; `Π` by fold with `·` and `Π(∅) = 1`
(Open decision 2); `|P|`; `x ∈ P`; `primeseq(P):⟺ ∀x ∈ P: prime(x)`.

- **F1 (60) (singleton length).** `|(a)| = 1`.
- **F2 (61) (concat length).** `|P ⧺ Q| = |P| + |Q|`.
- **F3 (62) (empty product).** `Π(∅) = 1`.
- **F4 (63) (singleton product).** `Π((a)) = a`.
- **F5 (64) (head product).** `Π((a) ⧺ P) = a · Π(P)`. *(Fold step.)*
- **F6 (65) (concatenation product).** `Π(P ⧺ Q) = Π(P) · Π(Q)`.
 *(Induction on `|P|` via F5; associativity.)*
- **F7 (66) (singleton membership).** `x ∈ (a) ⟺ x = a`.
- **F8 (67) (concat membership).** `x ∈ P ⧺ Q ⟺ x ∈ P ∨ x ∈ Q`.
- **F9 (68) (membership split).** `x ∈ P ⟹ ∃P₁, P₂: P = P₁ ⧺ (x) ⧺ P₂`.
 *(Induction on `|P|`.)*
- **F10 (69) (primeseq closure).** `primeseq(∅)`; `primeseq((p)) ⟺ prime(p)`;
 `primeseq(P ⧺ Q) ⟺ primeseq(P) ∧ primeseq(Q)`.
- **F11 (70) (element divides product).** `x ∈ P ⟹ x | Π(P)`.
 *(F9 + F6/F4 + C7.)*
- **F12 (71) (prime sequence product positive).** `primeseq(P) ⟹ Π(P) ≥ 1`.
 *(Induction; E2 + B2.)*
- **F13 (72) (nonempty prime product is ≥ 2).** `primeseq(P) ∧ |P| ≥ 1 ⟹ Π(P) ≥ 2`.
 *(F5 head form; E2; B6.)*
- **F14 (73) (trivial product forces empty).** `primeseq(P) ∧ Π(P) = 1 ⟹ P = ∅`.
 *(Contrapositive of F13.)*

## Part G — Euclid's lemma (division-free route)

- **G1 (74) (remainder keeps non-divisibility).**
 `prime(p) ∧ p ∤ a ∧ a = q·p + r ∧ r < p ⟹ r ≥ 1 ∧ p ∤ r`.
 *(`r = 0` would give `p | a` via D3; `p | r` would give `p | a` via C6/C7.)*
- **G2 (75) (Bézout base).** `∃x, y: 1·x = p·y + 1`. *(Witnesses `x = 1, y = 0`.)*
- **G3 (76) (Bézout descent step).**
 `a = q·p + r ∧ r·x = p·y + 1 ⟹ a·x = p·(q·x + y) + 1`.
 *(Pure ring rewriting: distributivity, associativity, commutativity.)*
- **G4 (77) (one-sided Bézout for a prime).**
 `prime(p) ∧ p ∤ a ⟹ ∃x, y: a·x = p·y + 1`.
 *(STRONG induction on `a`: D1 splits `a = q·p + r`; G1 gives `r ≥ 1`,
 `p ∤ r`, and `r < p ≤ a` unless `a < p` already — descent to `r`, lift
 with G3; base via G2 when `a = 1`. The heaviest lemma on the list. Over ℕ
 the identity is one-sided by design: no integers, no subtraction.)*
- **G5 (78) (Euclid's lemma).** `prime(p) ∧ p | a·b ⟹ p | a ∨ p | b`.
 *(If `p ∤ a`: G4 gives `a·x = p·y + 1`; multiply by `b`:
 `a·b·x = p·b·y + b`; `p` divides the left side (C7) and `p·b·y`, so C8
 gives `p | b`.)*
- **G6 (79) (prime divides a sequence product).**
 `prime(p) ∧ p | Π(P) ⟹ ∃x ∈ P: p | x`.
 *(Induction on `|P|`: head form F5, G5 at each step; empty case
 impossible via F3 and `p ≥ 2`.)*
- **G7 (80) (prime occurs in a prime sequence it divides).**
 `prime(p) ∧ primeseq(P) ∧ p | Π(P) ⟹ p ∈ P`.
 *(G6 gives an element `p` divides; E4 makes it equal to `p`.)*

## Part H — rearrangement (equality up to order)

Definition (Open decision 1): `P ≈ Q`, inductively — `∅ ≈ ∅`; and
`(p) ⧺ P′ ≈ Q:⟺ ∃Q₁, Q₂: Q = Q₁ ⧺ (p) ⧺ Q₂ ∧ P′ ≈ Q₁ ⧺ Q₂`.

- **H1 (81) (reflexivity).** `P ≈ P`. *(Induction on `|P|`; split with `Q₁ = ∅`.)*
- **H2 (82) (length invariance).** `P ≈ Q ⟹ |P| = |Q|`. *(Induction; F2.)*
- **H3 (83) (symmetry).** `P ≈ Q ⟹ Q ≈ P`. *(Induction on `|P|`.)*
- **H4 (84) (transitivity).** `P ≈ Q ∧ Q ≈ R ⟹ P ≈ R`. *(Induction.)*
- **H5 (85) (product invariance).** `P ≈ Q ⟹ Π(P) = Π(Q)`.
 *(Induction; F6/F5 + commutativity/associativity.)*
- **H6 (86) (primeseq invariance).** `P ≈ Q ∧ primeseq(P) ⟹ primeseq(Q)`.
 *(Induction; F10.)*
- **H7 (87) (membership invariance).** `P ≈ Q ∧ x ∈ P ⟹ x ∈ Q`.
- **H8 (88) (removal product).** `Q = Q₁ ⧺ (p) ⧺ Q₂ ⟹ Π(Q) = p · Π(Q₁ ⧺ Q₂)`.
 *(F6/F4 + commutativity/associativity.)*
- **H9 (89) (removal keeps primeseq).** `primeseq(Q₁ ⧺ (p) ⧺ Q₂) ⟹ primeseq(Q₁ ⧺ Q₂)`. *(F10.)*
- **H10 (90) (insertion congruence).** `P ≈ Q₁ ⧺ Q₂ ⟹ (p) ⧺ P ≈ Q₁ ⧺ (p) ⧺ Q₂`.
 *(Definitional split.)*

## Part I — uniqueness and the final theorems

- **I1 (91) (uniqueness core).**
 `primeseq(P) ∧ primeseq(Q) ∧ Π(P) = Π(Q) ⟹ P ≈ Q`.
 *(Induction on `|P|`. Empty case: F3 + F14 force `Q = ∅`. Step: head `p`
 divides `Π(P) = Π(Q)` (F11), G7 puts `p ∈ Q`, F9 splits `Q`, H8 + B8
 cancel `p` on both sides, H9 keeps primeseq, induction on the tails,
 close with H10.)*
- **I2 (92) (FTA, existence half).**
 `n ≥ 2 ⟹ ∃P: primeseq(P) ∧ |P| ≥ 1 ∧ Π(P) = n`.
 *(STRONG induction on `n`: prime → singleton (F4, F10); composite → E6
 splits `n = a·b` with both factors in `[2, n)`, induction gives prime
 sequences for `a` and `b`, concatenate (F6, F10, F2).)*
- **I3 (93) (FTA, uniqueness half).**
 `n ≥ 2 ∧ primeseq(P) ∧ primeseq(Q) ∧ Π(P) = n ∧ Π(Q) = n ⟹ P ≈ Q`.
 *(I1 at `n`.)*
- **I4 (94) (FTA, combined statement).**
 `∀n ≥ 2 ∃P: primeseq(P) ∧ |P| ≥ 1 ∧ Π(P) = n ∧ (∀Q: primeseq(Q) ∧ Π(Q) = n ⟹ Q ≈ P)`.
 *(I2 + I3.)*

## Part J — strong-induction packaging (superseded — native mechanism)

**Decided (2026-07-31):** strong induction enters GL as an internal proof
mechanism, introduced the same way weak induction is — a native method the
prover schedules itself, not assembled helper machinery. The former
packaging lemmas J1–J4 (package base / step / unwrap over
`Q(n):⟺ ∀m ≤ n: P(m)`, plus the per-predicate course-of-values scheme)
are no longer needed as lemmas and are dropped from the list; the shortlist
ends at I4 (94).

---

## Missing capabilities (separate review part)

What the current engine (v0.10.0, rungs 1–2 machinery) does not yet have,
mapped to the lemmas that need it:

1. **Strong induction (course-of-values).** Needed by E7, G4, I1, I2. The
 prover schedules structural zero/successor induction on a digit argument;
 the ladder needs `(∀m < n: P(m)) ⟹ P(n)`. **Decided (2026-07-31):**
 ships as a native internal proof mechanism, introduced the same way weak
 induction is; Part J stays off the conjecture list.
2. **Descent on a measure / two-component descent.** G4's Euclidean descent
 walks remainder pairs `(p, a) → (p, r)` — termination is by the strictly
 decreasing remainder, not structural recursion. Either a
 termination-measure capability or the single-variable strong-induction
 reformulation (as G4 is phrased above).
3. **Finite sequences as first-class proof objects.** Sequences of unbounded
 length with `Π`, `⧺`, `|·|`, `∈`, and *induction over sequence length*
 (F6, F9, G6, H1–H10, I1). The Gauss fold machinery covers fixed-bound
 sums; the ladder needs general finite sequences with a length-induction
 principle and the Part-F algebra.
4. **Rearrangement (multiset equality).** The inductive `≈` with its closure
 lemmas (Part H). Alternatives at review: canonical *sorted* sequences
 (needs sortedness + a least-prime-factor argument) or per-prime
 occurrence counting (needs counting machinery). The inductive `≈` is the
 lightest.
5. **Witness-generation budgets.** G4's Bézout witnesses `x, y` are not
 bounded by the operands, and D1's `q` grows with `a`. The per-batch
 generation cap (`maxIterationNumberVariable`, D-232) bounds witness depth
 tightly today; the ladder needs either principled per-conjecture budgets
 or witnesses arriving lemma-level via the conjecture list rather than
 search-level.
6. **Nested case analysis at scale.** Already present (rungs 1–2): B1, A15,
 D1's rollover split, E6/I2's prime-vs-composite, G5's disjunctive
 conclusion. Deeper and more frequent cohorts than rung 2 — expected to
 stress the existing OR machinery, not to require new kinds of it.
7. **Definitional intake.** `|` (divisibility), `prime`, sequences with
 `Π`/`⧺`/`∈`/length, and `≈` as new MPL definitions plus their incubator
 ground facts (E1-style instances). Also the intake path itself: the full
 Peano corpus as externally provided theorems is the blog post's rule of
 engagement and needs to stay a first-class, verifier-covered channel.
8. **Negative statements as first-class conjectures.** `p ∤ a` premises and
 conclusions (G1, G4, E1's `¬prime` instances) ride the
 reductio/contradiction machinery from rung 2.1 — expected to work,
 listed because the ladder uses negation far more densely than the rungs
 did.

## Open decisions for review

1. **Uniqueness formalization:** inductive rearrangement `≈` (recommended)
 vs sorted canonical sequences vs occurrence counting.
2. **Empty product:** `Π(∅) = 1` as used above (cleaner algebra, F6
 unconditional) vs nonempty-only sequences (closer to current fold usage;
 would condition F6 and complicate I1's base case).
3. **Granularity of G4:** keep as one heavy lemma with G1–G3 as supports (as
 listed), or split the descent into an explicit gcd-style chain (more
 conjectures, shallower steps).
4. **Corpus overlap:** which Part A/B/C items are already in the proved
 Peano corpus (move from "conjecture" to "externally provided") — check
 against `files/theorems/theorems.txt` at MPL-formalization time.
5. **Part J — DECIDED (2026-07-31):** strong induction is implemented
 natively as a GL internal mechanism (like weak induction); the Part-J
 packaging lemmas stay off the conjecture list.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
