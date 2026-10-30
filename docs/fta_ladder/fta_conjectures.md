<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# FTA conjectures — what GL is given to prove the Fundamental Theorem of Arithmetic

GL's input for the Fundamental Theorem of Arithmetic is the Peano axioms with the definitions built on them, plus the 160 conjectures of [`fta_conjectures.txt`](fta_conjectures.txt), one per line in GL's input form. GL proves every one of them, ending with I4 (every n ≥ 2 is a product of primes in descending order) and I5 (two such factorizations of the same number are equal). No external theorem enters any proof, and the Lean 4 kernel checks every proof. The FTA conjectures of the list are exactly the rows of GL's supplied conjecture files.

| Corpus | Conjectures |
|---|---|
| Peano | 48 |
| Gauss | 9 |
| FTA | 103, I4 and I5 included |
| **Total** | **160** |

How the list was made, from the proof graph of the full run of 2026-10-26 that proves I5:

1. Start at I4 and I5 and add every theorem their proofs cite, then every theorem those proofs cite, down to the axioms: 181 theorems (51 Peano, 11 Gauss, 119 FTA).
2. Leave out what GL writes by itself from a conjecture it is given: restated theorems and OR theorems (each replaced by the conjecture it comes from) and mirrored conjectures (contrapositives of another conjecture on the list). 158 remain.
3. Write each in the form GL is given. FTA conjectures are exactly the rows of GL's supplied conjecture files; 8 of them appear in the proof graph in compiled form (a defined existence such as `inSequence`, or an operator such as `existence16`) and are listed as the row GL compiled them from. Peano and Gauss conjectures, which GL normally conjectures by itself, are written in the same numbered input form.
4. Add the two supplied conjectures no proof cites but GL's search depends on (marked below): GL proves a conjecture not only with the lemmas a proof cites — a lemma's facts can also make GL request the expansions another proof starts from, or show GL that two witnesses are one object, so that it does not work through both.

Every other supplied conjecture was removed, and a full run confirmed that GL still proves I4 and I5. Order: proof order. Notation: v₁, v₂, … are numbers; V₁, V₂, … are sets, sequences or relations; s is the successor.

## Peano (48)

1. from v₁ = v₃ follows s(v₁) = s(v₃) 
 `(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8](in2[7,8,3])(>[9,10](in2[9,10,3])(>[](=[7,9])(=[8,10])))))`

2. from s(v₃) = s(v₁) follows v₁ = v₃ 
 `(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8](in2[7,8,3])(>[9](in2[9,8,3])(=[7,9]))))`

3. from v₁ = v₁ · 0 follows 0 = v₁ 
 `(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8](=[7,8])(>[](in3[7,2,8,5])(=[2,7]))))`

4. from v₁ ∈ ℕ follows v₁ + 0 = v₁ 
 `(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8](in3[7,2,8,4])(>[](in[7,1])(=[7,8]))))`

5. from v₁ + v₂ = 0, v₂ ∈ ℕ follows 0 = v₁ 
 `(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8](in3[7,8,2,4])(>[](in[8,1])(=[2,7]))))`

6. from v₄ + v₂ = v₁ + v₂ follows v₁ = v₄ 
 `(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,4])(>[10](in3[10,8,9,4])(=[7,10]))))`

7. v₁ · s(v₄) = v₁ · v₄ + v₁ 
 `(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,5])(>[10,11](in3[7,10,11,5])(>[](in2[10,8,3])(in3[11,7,9,4])))))`

8. v₁ · 0 = 0 · v₁ 
 `(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8](in3[7,2,8,5])(in3[2,7,8,5])))`

9. s(v₁ + v₂) = v₁ + s(v₂) 
 `(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,4])(>[10](in2[8,10,3])(>[11](in2[9,11,3])(in3[7,10,11,4])))))`

10. s(v₁ + v₂) = s(v₁) + v₂ 
 `(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,4])(>[10](in2[7,10,3])(>[11](in2[9,11,3])(in3[10,8,11,4])))))`

11. s(v₄) + v₂ = v₄ + s(v₂) 
 `(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,4])(>[10](in2[10,7,3])(>[11](in2[8,11,3])(in3[10,11,9,4])))))`

12. v₁ + s(v₄) = s(v₁ + v₄) 
 `(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,4])(>[10,11](in3[7,10,11,4])(>[](in2[10,8,3])(in2[11,9,3])))))`

13. v₁ + (v₂ + v₆) = (v₁ + v₂) + v₆ 
 `(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,4])(>[10,11](in3[7,10,11,4])(>[12](in3[8,12,10,4])(in3[9,12,11,4])))))`

14. v₁ + s(v₅) = s(v₁) + v₅ 
 `(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,4])(>[10](in2[7,10,3])(>[11](in2[11,8,3])(in3[10,11,9,4])))))`

15. (v₄ + v₅) + v₂ = (v₄ + v₂) + v₅ 
 `(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,4])(>[10,11](in3[10,11,7,4])(>[12](in3[10,8,12,4])(in3[12,11,9,4])))))`

16. (v₄ + v₅) + v₂ = v₄ + (v₅ + v₂) 
 `(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,4])(>[10,11](in3[10,11,7,4])(>[12](in3[11,8,12,4])(in3[10,12,9,4])))))`

17. v₁ + 1 = s(v₁) 
 `(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8](in3[7,6,8,4])(in2[7,8,3])))`

18. from v₁ ∈ ℕ follows 1 · v₁ = v₁ 
 `(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7](in[7,1])(>[8](in3[6,7,8,5])(=[7,8]))))`

19. from v₁ ∈ ℕ, 1 · v₂ = v₁ follows v₁ = v₂ 
 `(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7](in[7,1])(>[8](in3[6,8,7,5])(=[7,8]))))`

20. v₂ · v₄ + v₂ = v₂ · s(v₄) 
 `(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,4])(>[10,11](in2[10,11,3])(>[](in3[8,10,7,5])(in3[8,11,9,5])))))`

21. from v₁ ∈ ℕ follows v₁ · 0 = 0 
 `(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8](in3[7,2,8,5])(>[](in[7,1])(=[2,8]))))`

22. from v₁ ∈ ℕ follows 0 · v₁ = 0 
 `(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7](in[7,1])(>[8](in3[2,7,8,5])(=[2,8]))))`

23. from v₁ + v₂ = 0, v₁ ∈ ℕ follows 0 = v₂ 
 `(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8](in3[7,8,2,4])(>[](in[7,1])(=[2,8]))))`

24. from v₁ ∈ ℕ follows 0 + v₁ = v₁ 
 `(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7](in[7,1])(>[8](in3[2,7,8,4])(=[7,8]))))`

25. 1 · v₁ = v₁ · 1 
 `(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8](in3[6,7,8,5])(in3[7,6,8,5])))`

26. s(v₁) = 1 + v₁ 
 `(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8](in2[7,8,3])(in3[6,7,8,4])))`

27. s(v₁) = v₁ + 1 
 `(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8](in2[7,8,3])(in3[7,6,8,4])))`

28. s(v₄) + v₂ = s(v₂) + v₄ 
 `(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,4])(>[10](in2[10,7,3])(>[11](in2[8,11,3])(in3[11,10,9,4])))))`

29. s(v₄) · v₂ = v₂ + v₄ · v₂ 
 `(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,5])(>[10,11](in3[10,8,11,5])(>[](in2[10,7,3])(in3[8,11,9,4])))))`

30. v₁ · (v₂ + v₄) = v₁ · v₂ + v₁ · v₄ 
 `(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,5])(>[10,11](in3[7,10,11,5])(>[12,13](in3[7,12,13,5])(>[](in3[8,10,12,4])(in3[9,11,13,4]))))))`

31. v₇ · v₄ + v₇ · v₅ = v₇ · (v₄ + v₅) 
 `(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,4])(>[10,11,12](in3[10,11,12,4])(>[13](in3[13,10,7,5])(>[](in3[13,11,8,5])(in3[13,12,9,5]))))))`

32. v₁ + 0 = 0 + v₁ 
 `(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8](in3[7,2,8,4])(in3[2,7,8,4])))`

33. v₁ + v₂ = v₂ + v₁ 
 `(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,4])(in3[8,7,9,4])))`

34. s(v₂) · v₁ = v₁ + v₁ · v₂ 
 `(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,5])(>[10,11](in3[10,7,11,5])(>[](in2[8,10,3])(in3[7,9,11,4])))))`

35. v₁ + v₁ · v₄ = v₁ · s(v₄) 
 `(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,4])(>[10,11](in2[10,11,3])(>[](in3[7,10,8,5])(in3[7,11,9,5])))))`

36. (v₁ + v₂) · v₄ = v₁ · v₄ + v₂ · v₄ 
 `(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,4])(>[10,11](in3[7,10,11,5])(>[12](in3[8,10,12,5])(>[13](in3[9,10,13,5])(in3[11,12,13,4]))))))`

37. (v₄ · v₅) · v₂ = (v₄ · v₂) · v₅ 
 `(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,5])(>[10,11](in3[10,11,7,5])(>[12](in3[10,8,12,5])(in3[12,11,9,5])))))`

38. v₁ · v₂ = v₂ · v₁ 
 `(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,5])(in3[8,7,9,5])))`

39. from v₂ = v₁ + v₂ follows 0 = v₁ 
 `(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,4])(>[](=[8,9])(=[2,7]))))`

40. from v₁ ∈ ℕ, v₁ is not a value of s follows v₁ = 0 
 `(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7](in[7,1])(>[]!(inSequence[1,7,3])(=[7,2]))))`

41. from v₁ + v₄ = v₁ + v₂ follows v₂ = v₄ 
 `(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,4])(>[10](in3[7,10,9,4])(=[8,10]))))`

42. 1 is a value of s 
 `(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(inSequence[1,6,3]))`

43. 1 + v₁ = v₁ + 1 
 `(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8](in3[6,7,8,4])(in3[7,6,8,4])))`

44. from v₁ ≤ v₂, v₂ ≤ v₁ follows v₁ = v₂ 
 `(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8](preorder[1,4,7,8])(>[](preorder[1,4,8,7])(=[7,8]))))`

45. (v₄ + v₅) + v₂ = (v₂ + v₅) + v₄ 
 `(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,4])(>[10,11](in3[10,11,7,4])(>[12](in3[8,11,12,4])(in3[12,10,9,4])))))`

46. (v₄ · v₅) · v₂ = (v₂ · v₅) · v₄ 
 `(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,5])(>[10,11](in3[10,11,7,5])(>[12](in3[8,11,12,5])(in3[12,10,9,5])))))`

47. from v₅ + s(v₁) = s(v₃) follows v₅ + v₁ = v₃ 
 `(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8](in2[7,8,3])(>[9,10](in2[9,10,3])(>[11](in3[11,8,10,4])(in3[11,7,9,4])))))`

48. from v₁ = v₂, v₂ · v₁ = 0 follows 0 = v₂ 
 `(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8](=[7,8])(>[](in3[8,7,2,5])(=[2,8]))))`

## Gauss (9)

49. from v₃ ≤ v₁ follows v₃ ≤ s(v₁) 
 `(>[1,2,3,4,5,6,7,8](AnchorGauss[1,2,3,4,5,6,7,8])(>[9,10](in2[9,10,3])(>[11](preorder[1,4,11,9])(preorder[1,4,11,10]))))`

50. from V₂ = V₁ ∩ [0, v₁], V₁ = [0, s(v₁)] follows V₂ = [0, v₁] 
 `(>[1,2,3,4,5,6,7,8](AnchorGauss[1,2,3,4,5,6,7,8])(>[9,10](in2[9,10,3])(>[11,12](limitSet[1,4,11,9,12])(>[](interval[1,4,2,10,11])(interval[1,4,2,9,12])))))`

51. from V₂ = V₁|≤ v₁, V₁: [0, s(v₁)] → ℕ follows V₂: [0, v₁] → ℕ 
 `(>[1,2,3,4,5,6,7,8](AnchorGauss[1,2,3,4,5,6,7,8])(>[9,10](in2[9,10,3])(>[11,12](limitSequence[1,4,9,11,12])(>[](sequence[1,4,2,10,11])(sequence[1,4,2,9,12])))))`

52. v₁ + v₂ = v₂ + v₁ 
 `(>[1,2,3,4,5,6,7,8](AnchorGauss[1,2,3,4,5,6,7,8])(>[9,10,11](in3[9,10,11,4])(in3[10,9,11,4])))`

53. from V₁ = [0, v₁], V₂: [0, v₁] → ℕ follows V₂: V₁ → ℕ 
 `(>[1,2,3,4,5,6,7,8](AnchorGauss[1,2,3,4,5,6,7,8])(>[9,10](interval[1,4,2,9,10])(>[11](sequence[1,4,2,9,11])(fXY[11,10,1]))))`

54. from V₁: V₂ → ℕ, V₂ = [0, v₁] follows V₁: [0, v₁] → ℕ 
 `(>[1,2,3,4,5,6,7,8](AnchorGauss[1,2,3,4,5,6,7,8])(>[9,10](fXY[9,10,1])(>[11](interval[1,4,2,11,10])(sequence[1,4,2,11,9]))))`

55. from V₁ = [0, v₁] follows v₁ ∈ V₁ 
 `(>[1,2,3,4,5,6,7,8](AnchorGauss[1,2,3,4,5,6,7,8])(>[9,10](interval[1,4,2,9,10])(in[9,10])))`

56. v₁ ≤ s(v₁) 
 `(>[1,2,3,4,5,6,7,8](AnchorGauss[1,2,3,4,5,6,7,8])(>[9,10](in2[9,10,3])(preorder[1,4,9,10])))`

57. from v₁ ≤ v₂, v₂ ≤ v₃ follows v₁ ≤ v₃ 
 `(>[1,2,3,4,5,6,7,8](AnchorGauss[1,2,3,4,5,6,7,8])(>[9,10](preorder[1,4,9,10])(>[11](preorder[1,4,10,11])(preorder[1,4,9,11]))))`

## FTA (103)

58. from V₁ = {0 ↦ v₁}, v₁ = v₂ follows v₂ is a value of V₁ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](EnumerationRelationSet1[2,9,10])(>[11](=[9,11])!(>[12](in[12,1])!(in2[12,11,10])))))`

59. from V₂ = V₁ ∪ {v₁ ↦ v₂}, V₃ = V₁ ∪ {v₁ ↦ v₂} follows V₂ = V₃ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10,11,12](appendToSequence[9,10,11,12])(>[13](appendToSequence[9,10,11,13])(=[12,13]))))`

60. from V₁ = {0 ↦ v₁}, V₂ = {0 ↦ v₁} follows V₁ = V₂ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](EnumerationRelationSet1[2,9,10])(>[11](EnumerationRelationSet1[2,9,11])(=[10,11]))))`

61. from v₁ ∈ ℕ, v₁ = v₂ follows 1 · v₁ = v₂ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9](in[9,1])(>[10](=[9,10])(in3[6,9,10,5]))))`

62. from v₁ ∈ ℕ follows 1 ∣ v₁ — *no proof cites it, yet GL needs it: its facts 1 ∣ 0, 1 ∣ v₁, 1 ∣ v₂ let another conjecture request the expansion of 0 ≤ v that the base case of Bezout's induction is built on* 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9](in[9,1])(preorder[1,5,6,9])))`

63. from v₁ ∈ ℕ, v₁ = v₂ follows v₁ ∣ v₂ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9](in[9,1])(>[10](=[9,10])(preorder[1,5,9,10]))))`

64. from v₁ · (v₂ + v₇) = v₁ · v₂ + v₄ follows v₁ · v₇ = v₄ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10,11](in3[9,10,11,5])(>[12,13](in3[11,12,13,4])(>[14](in3[9,14,13,5])(>[15](in3[10,15,14,4])(in3[9,15,12,5]))))))`

65. from v₄ · v₆ = v₁ · v₇ + 1 follows ∃ Y₁: v₂ · v₆ + v₇ = Y₁ ∧ (v₁ · v₂ + v₄) · v₆ = v₁ · Y₁ + 1 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10,11](in3[9,10,11,5])(>[12,13](in3[11,12,13,4])(>[14,15](bezEq[1,4,5,6,12,15,9,14])(>[16](in3[10,15,16,5])!(>[17](in3[16,14,17,4])!(bezEq[1,4,5,6,13,15,9,17])))))))`

66. from v₁ · v₆ = v₄ · v₇ + 1 follows ∃ Y₁: v₆ + v₂ · v₇ = Y₁ ∧ v₁ · Y₁ = (v₁ · v₂ + v₄) · v₇ + 1 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10,11](in3[9,10,11,5])(>[12,13](in3[11,12,13,4])(>[14,15](bezEq[1,4,5,6,9,14,12,15])(>[16](in3[10,15,16,5])!(>[17](in3[14,16,17,4])!(bezEq[1,4,5,6,9,17,13,15])))))))`

67. from v₁ ∣ v₂, (∃ m, n ∈ ℕ: v₁ · m = v₃ · n ± 1) follows ∃ m, n ∈ ℕ: (v₂ + v₃) · m = v₁ · n ± 1 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](preorder[1,5,9,10])(>[11,12](in3[10,11,12,4])(>[](bez1[1,4,5,6,9,11])(bez1[1,4,5,6,12,9])))))`

68. from v₁ ∈ ℕ, 0 ≠ v₁, 1 ≠ v₁ follows 2 ≤ v₁ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9](in[9,1])(>[]!(=[2,9])(>[]!(=[6,9])(preorder[1,4,7,9])))))`

69. from gcd(v₃ · v₂ + v₅, v₂) = 1 follows gcd(v₂, v₅) = 1 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](coprime[1,2,3,5,9,10])(>[11,12](in3[11,10,12,5])(>[13](in3[12,13,9,4])(coprime[1,2,3,5,10,13])))))`

70. from v₁ ∣ v₂, 1 ≤ v₂ follows 1 ≤ v₁ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](preorder[1,5,9,10])(>[](preorder[1,4,6,10])(preorder[1,4,6,9]))))`

71. from s(v₁) ≤ v₃ follows v₁ ≠ v₃ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](in2[9,10,3])(>[11](preorder[1,4,10,11])!(=[9,11]))))`

72. from v₁ ∈ ℕ follows 0 ≤ v₁ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9](in[9,1])(preorder[1,4,2,9])))`

73. from s(v₁) ≤ v₃ follows v₁ ≤ v₃ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](in2[9,10,3])(>[11](preorder[1,4,10,11])(preorder[1,4,9,11]))))`

74. from V₁ = [0, v₁], V₂ = [0, v₁] follows V₁ = V₂ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](interval[1,4,2,9,10])(>[11](interval[1,4,2,9,11])(=[10,11]))))`

75. from V₂ = V₁|≤ v₁ follows ∏(i=0..v₁) V₂(i) = ∏(i=0..v₁) V₁(i) 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10,11](limitSequence[1,4,9,10,11])(>[12](fold[1,3,4,5,11,2,9,12])(fold[1,3,4,5,10,2,9,12]))))`

76. v₂ ∣ v₁ · v₂ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10,11](in3[9,10,11,5])(preorder[1,5,10,11])))`

77. from v₁ ≤ v₂ follows v₁ + v₃ ≤ v₂ + v₃ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](preorder[1,4,9,10])(>[11,12](in3[9,11,12,4])(>[13](in3[10,11,13,4])(preorder[1,4,12,13])))))`

78. v₁ ≤ v₁ + v₂ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10,11](in3[9,10,11,4])(preorder[1,4,9,11])))`

79. from 1 ≤ v₁, v₂ · v₁ ≤ v₄ · v₁ follows v₂ ≤ v₄ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9](preorder[1,4,6,9])(>[10,11](in3[10,9,11,5])(>[12,13](in3[12,9,13,5])(>[](preorder[1,4,11,13])(preorder[1,4,10,12]))))))`

80. from V₁: [0, v₁] → ℕ, V₁(v₂) = v₃, V₁(v₄) = v₅, v₃ ≠ v₅ follows v₂ ≠ v₄ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](sequence[1,4,2,9,10])(>[11,12](in2[11,12,10])(>[13,14](in2[13,14,10])(>[]!(=[12,14])!(=[11,13]))))))`

81. from v₁ < v₂ follows v₂ ≮ v₁ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](strictOrder[1,4,9,10])!(strictOrder[1,4,10,9])))`

82. from v₃ + s(v₅) = s(v₁) follows v₃ + v₅ = v₁ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](in2[9,10,3])(>[11,12](in3[11,12,10,4])(>[13](in2[13,12,3])(in3[11,13,9,4])))))`

83. from v₁ < v₁ + v₃ follows 0 ≠ v₃ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](strictOrder[1,4,9,10])(>[11](in3[9,11,10,4])!(=[2,11]))))`

84. from v₁ < v₃ follows v₃ ≮ s(v₁) 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](in2[9,10,3])(>[11](strictOrder[1,4,9,11])!(strictOrder[1,4,11,10]))))`

85. from v₃ < s(v₁) follows v₃ ≤ v₁ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](in2[9,10,3])(>[11](strictOrder[1,4,11,10])(preorder[1,4,11,9]))))`

86. from v₁ ≤ v₂, v₁ ≠ v₂ follows v₁ < v₂ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](preorder[1,4,9,10])(>[]!(=[9,10])(strictOrder[1,4,9,10]))))`

87. from V₁: [0, s(v₁)] → ℕ, V₂ = V₁|≤ v₁ follows V₁ = V₂ ∪ {s(v₁) ↦ V₁(s(v₁))} 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](in2[9,10,3])(>[11](sequence[1,4,2,10,11])(>[12](in2[10,12,11])(>[13](limitSequence[1,4,9,11,13])(appendToSequence[10,12,13,11]))))))`

88. from v₁ ∈ ℕ, 0 ≠ v₁ follows 1 ≤ v₁ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9](in[9,1])(>[]!(=[2,9])(preorder[1,4,6,9]))))`

89. from v₁ ≤ v₂, v₃ ≰ v₂ follows v₁ ≠ v₃ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](preorder[1,4,9,10])(>[11]!(preorder[1,4,11,10])!(=[9,11]))))`

90. from v₁ ∈ ℕ, v₂ ∈ ℕ, v₁ ≰ v₂ follows v₂ ≤ v₁ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9](in[9,1])(>[10](in[10,1])(>[]!(preorder[1,4,9,10])(preorder[1,4,10,9])))))`

91. from v₁ ∈ ℕ, v₂ ∈ ℕ, v₁ ≮ v₂, v₁ ≠ v₂ follows v₂ < v₁ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9](in[9,1])(>[10](in[10,1])(>[]!(strictOrder[1,4,9,10])(>[]!(=[9,10])(strictOrder[1,4,10,9]))))))`

92. from 1 ≤ v₁ follows 0 ≠ v₁ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9](preorder[1,4,6,9])!(=[2,9])))`

93. from V₁ = [0, v₁], V₂ = V₁ ∪ {s(v₁)} follows V₂ = [0, s(v₁)] 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](in2[9,10,3])(>[11](interval[1,4,2,9,11])(>[12](addToSet[11,10,12])(interval[1,4,2,10,12])))))`

94. from v₁ ∈ ℕ, v₁ = v₂ follows v₁ ≤ v₂ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9](in[9,1])(>[10](=[9,10])(preorder[1,4,9,10]))))`

95. from V₁: [0, s(v₁)] → ℕ follows ∏(i=0..s(v₁)) V₁(i) = (∏(i=0..v₁) V₁(i)) · V₁(s(v₁)) 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](in2[9,10,3])(>[11](sequence[1,4,2,10,11])(>[12](in2[10,12,11])(>[13](fold[1,3,4,5,11,2,9,13])(>[14](fold[1,3,4,5,11,2,10,14])(in3[13,12,14,5])))))))`

96. from v₁ ∣ v₂ follows v₁ ∣ v₂ · v₃ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](preorder[1,5,9,10])(>[11,12](in3[10,11,12,5])(preorder[1,5,9,12]))))`

97. from V₁: [0, v₁] → ℕ, v₃ is a value of V₁ follows v₃ ∣ ∏(i=0..v₁) V₁(i) 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](sequence[1,4,2,9,10])(>[11](fold[1,3,4,5,10,2,9,11])(>[12](inSequence[1,12,10])(preorder[1,5,12,11])))))`

98. from v₃ ≤ v₁ follows v₃ < s(v₁) 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](in2[9,10,3])(>[11](preorder[1,4,11,9])(strictOrder[1,4,11,10]))))`

99. from V₂ = V₁ ∪ {v₁ ↦ v₂} follows V₂(v₁) = v₂ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10,11,12](appendToSequence[9,10,11,12])(in2[9,10,12])))`

100. from v₁ = v₂ follows v₁ ≮ v₂ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](=[9,10])!(strictOrder[1,4,9,10])))`

101. from V₁: [0, v₁] → ℕ, v₃ ∈ ℕ, V₂ = V₁ ∪ {s(v₁) ↦ v₃} follows V₂: [0, s(v₁)] → ℕ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](in2[9,10,3])(>[11](sequence[1,4,2,9,11])(>[12](in[12,1])(>[13](appendToSequence[10,12,11,13])(sequence[1,4,2,10,13]))))))`

102. from V₁: [0, v₁] → ℕ, V₂ = V₁ ∪ {s(v₁) ↦ v₃}, V₃ = V₂|≤ v₁ follows V₃ = V₁ — *no proof cites it, yet it shortens GL's search: it shows GL that the sequence cut from V₂ at v₁ and the sequence V₂ extends are one object, which GL would otherwise work through twice* 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](in2[9,10,3])(>[11](sequence[1,4,2,9,11])(>[12,13](appendToSequence[10,12,11,13])(>[14](limitSequence[1,4,9,13,14])(=[14,11]))))))`

103. from V₁: [0, s(v₁)] → ℕ follows (∏(i=0..v₁) V₁(i)) · V₁(s(v₁)) = ∏(i=0..s(v₁)) V₁(i) 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](in2[9,10,3])(>[11](sequence[1,4,2,10,11])(>[12](in2[10,12,11])(>[13](fold[1,3,4,5,11,2,9,13])(>[14](in3[13,12,14,5])(fold[1,3,4,5,11,2,10,14])))))))`

104. from V₁: [0, v₁] → ℕ, V₂ = V₁ ∪ {s(v₁) ↦ v₃} follows ∏(i=0..v₁) V₁(i) = ∏(i=0..v₁) V₂(i) 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](in2[9,10,3])(>[11](sequence[1,4,2,9,11])(>[12,13](appendToSequence[10,12,11,13])(>[14](fold[1,3,4,5,11,2,9,14])(fold[1,3,4,5,13,2,9,14]))))))`

105. from V₁: [0, v₁] → ℕ, V₂ = V₁ ∪ {s(v₁) ↦ v₃} follows (∏(i=0..v₁) V₁(i)) · v₃ = ∏(i=0..s(v₁)) V₂(i) 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](in2[9,10,3])(>[11](sequence[1,4,2,9,11])(>[12,13](appendToSequence[10,12,11,13])(>[14](fold[1,3,4,5,11,2,9,14])(>[15](in3[14,12,15,5])(fold[1,3,4,5,13,2,10,15])))))))`

106. from V₁: [0, v₁] → ℕ, v₃ ∈ ℕ, V₂ = V₁ ∪ {s(v₁) ↦ v₃} follows ∏(i=0..s(v₁)) V₂(i) = (∏(i=0..v₁) V₁(i)) · v₃ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](in2[9,10,3])(>[11](sequence[1,4,2,9,11])(>[12](in[12,1])(>[13](appendToSequence[10,12,11,13])(>[14](fold[1,3,4,5,11,2,9,14])(>[15](fold[1,3,4,5,13,2,10,15])(in3[14,12,15,5]))))))))`

107. from v₁ · v₂ < v₄ · v₂ follows v₁ < v₄ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10,11](in3[9,10,11,5])(>[12,13](in3[12,10,13,5])(>[](strictOrder[1,4,11,13])(strictOrder[1,4,9,12])))))`

108. from v₁ ∣ v₂, v₁ ∣ v₂ + v₃ follows v₁ ∣ v₃ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](preorder[1,5,9,10])(>[11,12](in3[10,11,12,4])(>[](preorder[1,5,9,12])(preorder[1,5,9,11])))))`

109. v₃ · (v₁ + 1) = v₃ · v₁ + v₃ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](in3[9,6,10,4])(>[11,12](in3[11,9,12,5])(>[13](in3[11,10,13,5])(in3[12,11,13,4])))))`

110. from v₅ ∣ v₁ + 1, v₅ ∣ v₃ · v₁ follows v₅ ∣ v₃ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](in3[9,6,10,4])(>[11,12](in3[11,9,12,5])(>[13](preorder[1,5,13,10])(>[](preorder[1,5,13,12])(preorder[1,5,13,11]))))))`

111. from v₅ ∣ v₁, v₅ ∣ v₃ · (v₁ + 1) follows v₅ ∣ v₃ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](in3[9,6,10,4])(>[11,12](in3[11,10,12,5])(>[13](preorder[1,5,13,9])(>[](preorder[1,5,13,12])(preorder[1,5,13,11]))))))`

112. from (∃ m, n ∈ ℕ: v₁ · m = v₂ · n ± 1), v₁ ∣ v₂ · v₃ follows v₁ ∣ v₃ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](bez1[1,4,5,6,9,10])(>[11,12](in3[10,11,12,5])(>[](preorder[1,5,9,12])(preorder[1,5,9,11])))))`

113. from 1 ≤ v₁ follows v₂ ≤ v₂ · v₁ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9](preorder[1,4,6,9])(>[10,11](in3[10,9,11,5])(preorder[1,4,10,11]))))`

114. from v₁ < v₂, 1 ≤ v₃ follows v₁ · v₃ < v₂ · v₃ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](strictOrder[1,4,9,10])(>[11](preorder[1,4,6,11])(>[12](in3[9,11,12,5])(>[13](in3[10,11,13,5])(strictOrder[1,4,12,13]))))))`

115. from v₁ ∣ v₂, 1 ≤ v₂ follows v₁ ≤ v₂ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](preorder[1,5,9,10])(>[](preorder[1,4,6,10])(preorder[1,4,9,10]))))`

116. from v₁ ∣ v₂, v₂ ∣ v₃ follows v₁ ∣ v₃ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](preorder[1,5,9,10])(>[11](preorder[1,5,10,11])(preorder[1,5,9,11]))))`

117. from 2 ≤ v₁, V₁ = {d ∈ ℕ: 2 ≤ d ≤ v₁, d ∣ v₁}, v₂ = min V₁ follows v₂ is prime 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9](preorder[1,4,7,9])(>[10](divisorSet[1,4,5,7,9,10])(>[11](minimum[1,4,10,11])(prime[1,4,5,6,7,11])))))`

118. from 2 ≤ v₁ follows 1 ≠ v₁ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9](preorder[1,4,7,9])!(=[6,9])))`

119. from 1 ≤ v₁, 2 ≤ v₂ follows v₁ < v₁ · v₂ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9](preorder[1,4,6,9])(>[10](preorder[1,4,7,10])(>[11](in3[9,10,11,5])(strictOrder[1,4,9,11])))))`

120. from s(v₁) ≤ s(v₃) follows v₁ ≤ v₃ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](in2[9,10,3])(>[11,12](in2[11,12,3])(>[](preorder[1,4,10,12])(preorder[1,4,9,11])))))`

121. from 0 < v₁, V₁: [0, v₁] → ℕ, all values of V₁ are prime follows V₁(v₁) < ∏(i=0..v₁) V₁(i) 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9](strictOrder[1,4,2,9])(>[10](sequence[1,4,2,9,10])(>[11](in2[9,11,10])(>[12](fold[1,3,4,5,10,2,9,12])(>[](primeseq[1,4,5,6,7,10])(strictOrder[1,4,11,12])))))))`

122. from 2 ≤ v₁, 1 ≤ v₂ follows 2 ≤ v₁ · v₂ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9](preorder[1,4,7,9])(>[10](preorder[1,4,6,10])(>[11](in3[9,10,11,5])(preorder[1,4,7,11])))))`

123. from V₁: [0, v₁] → ℕ, all values of V₁ are prime follows 2 ≤ ∏(i=0..v₁) V₁(i) 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](sequence[1,4,2,9,10])(>[11](fold[1,3,4,5,10,2,9,11])(>[](primeseq[1,4,5,6,7,10])(preorder[1,4,7,11])))))`

124. from v₁ ∈ ℕ, 1 ≤ v₂ follows ∃ y₁ ∈ ℕ: v₁ mod v₂ = y₁ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9](in[9,1])(>[10](preorder[1,4,6,10])!(>[11](in[11,1])!(remainder[1,4,5,9,10,11])))))`

125. Peano anchor (ℕ, 0, s, +, ·, 1) 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(AnchorPeano[1,2,3,4,5,6]))`

126. from v₁ ∈ ℕ, v₂ ∈ ℕ, gcd(v₁, v₂) = 1 follows ∃ m, n ∈ ℕ: v₁ · m = v₂ · n ± 1 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9](in[9,1])(>[10](in[10,1])(>[](coprime[1,2,3,5,9,10])(bez1[1,4,5,6,9,10])))))`

127. from 2 ≤ v₁ follows 0 ≠ v₁ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9](preorder[1,4,7,9])!(=[2,9])))`

128. from v₁ is prime, v₁ ∤ v₂, v₃ ∣ v₁, v₃ ∣ v₂ follows v₃ = 1 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9](prime[1,4,5,6,7,9])(>[10]!(preorder[1,5,9,10])(>[11](preorder[1,5,11,9])(>[](preorder[1,5,11,10])(=[11,6]))))))`

129. from v₁ is prime, v₁ ∤ v₂ follows gcd(v₁, v₂) = 1 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9](prime[1,4,5,6,7,9])(>[10]!(preorder[1,5,9,10])(coprime[1,2,3,5,9,10]))))`

130. from v₁ ∈ ℕ, V₁ ⊆ ℕ, V₁ has no minimum, v₂ ∈ V₁ follows v₂ ≰ v₁ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9](in[9,1])(>[10](subset[10,1])(>[]!(hasMinimum[1,4,10])(>[11](in[11,10])!(preorder[1,4,11,9]))))))`

131. from 1 ≤ v₁, 1 ≤ v₂ follows 1 ≤ v₁ · v₂ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9](preorder[1,4,6,9])(>[10](preorder[1,4,6,10])(>[11](in3[9,10,11,5])(preorder[1,4,6,11])))))`

132. from V₁: [0, v₁] → ℕ follows V₁(v₁) is a value of V₁ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](sequence[1,4,2,9,10])(>[11](in2[9,11,10])!(>[12](in[12,1])!(in2[12,11,10])))))`

133. from v₂ = ∏(i=0..v₁) V₁(i) with primes V₁(0) ≥ V₁(1) ≥ … ≥ V₁(v₁), V₂ = {d ∈ ℕ: 2 ≤ d ≤ v₃ · v₂, d ∣ v₃ · v₂}, v₃ = min V₂ follows v₃ ≤ V₁(v₁) 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10,11](primeFactorSeq[1,3,4,5,2,6,7,9,10,11])(>[12,13](in3[12,11,13,5])(>[14](divisorSet[1,4,5,7,13,14])(>[](minimum[1,4,14,12])(>[15](in2[9,15,10])(preorder[1,4,12,15])))))))`

134. from 2 ≤ v₁, 1 ≤ v₁ · v₂ follows v₂ < v₁ · v₂ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10,11](in3[9,10,11,5])(>[](preorder[1,4,7,9])(>[](preorder[1,4,6,11])(strictOrder[1,4,10,11])))))`

135. from V₁ = {0} follows V₁ = [0, 0] 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9](EnumerationSet1[2,9])(interval[1,4,2,2,9])))`

136. from V₁ = {0 ↦ v₁}, V₁(v₂) = v₃ follows v₃ = v₁ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](EnumerationRelationSet1[2,9,10])(>[11,12](in2[11,12,10])(=[12,9]))))`

137. from v₁ ∈ ℕ, V₁ = {0 ↦ v₁} follows V₁: [0, 0] → ℕ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9](in[9,1])(>[10](EnumerationRelationSet1[2,9,10])(sequence[1,4,2,2,10]))))`

138. from v₁ ∈ ℕ, V₁ = {0 ↦ v₁} follows ∏(i=0..0) V₁(i) = v₁ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9](in[9,1])(>[10](EnumerationRelationSet1[2,9,10])(fold[1,3,4,5,10,2,2,9]))))`

139. Gauss anchor (ℕ, 0, s, +, ·, 1, 2, id) 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(AnchorGauss[1,2,3,4,5,6,7,8]))`

140. from V₁: [0, v₁] → ℕ, V₁ is non-increasing, v₃ ≤ V₁(v₁), V₁(v₄) = v₅ ∨ (v₄ = v₆ ∧ v₅ = v₃) follows v₃ ≤ v₅ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](sequence[1,4,2,9,10])(>[11](in2[9,11,10])(>[](ordered[1,4,10])(>[12](preorder[1,4,12,11])(>[13,14,15]!(&!(in2[14,15,10])!(&(=[14,13])(=[15,12])))(preorder[1,4,12,15])))))))`

141. from V₁: [0, v₁] → ℕ, V₁ is non-increasing, v₄ ≤ V₁(v₁), V₂ = V₁ ∪ {s(v₁) ↦ v₄} follows V₂ is non-increasing 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](in2[9,10,3])(>[11](sequence[1,4,2,9,11])(>[12](in2[9,12,11])(>[](ordered[1,4,11])(>[13](preorder[1,4,13,12])(>[14](appendToSequence[10,13,11,14])(ordered[1,4,14]))))))))`

142. from V₁ = {d ∈ ℕ: 2 ≤ d ≤ v₁, d ∣ v₁} follows V₁ ⊆ ℕ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](divisorSet[1,4,5,7,9,10])(subset[10,1])))`

143. from 2 ≤ v₁, V₁ = {d ∈ ℕ: 2 ≤ d ≤ v₁, d ∣ v₁} follows V₁ has a minimum 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9](preorder[1,4,7,9])(>[10](divisorSet[1,4,5,7,9,10])(hasMinimum[1,4,10]))))`

144. from v₁ is prime, V₁ = {0 ↦ v₁} follows all values of V₁ are prime 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9](prime[1,4,5,6,7,9])(>[10](EnumerationRelationSet1[2,9,10])(primeseq[1,4,5,6,7,10]))))`

145. from v₁ ∈ ℕ, V₁ = {0 ↦ v₁} follows V₁ is non-increasing 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9](in[9,1])(>[10](EnumerationRelationSet1[2,9,10])(ordered[1,4,10]))))`

146. from all values of V₁ are prime, v₁ is prime, V₂ = V₁ ∪ {v₂ ↦ v₁} follows all values of V₂ are prime 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9](primeseq[1,4,5,6,7,9])(>[10](prime[1,4,5,6,7,10])(>[11,12](appendToSequence[11,10,9,12])(primeseq[1,4,5,6,7,12])))))`

147. from 2 ≤ v₁ follows ∃ primes p₀ ≥ p₁ ≥ … ≥ pₖ: v₁ = ∏(i=0..k) pᵢ — **I4** 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9](preorder[1,4,7,9])(hasPrimeFactorization[1,3,4,5,2,6,7,9])))`

148. from v₁ is prime, v₁ ∣ v₂ · v₃, v₁ ∤ v₂ follows v₁ ∣ v₃ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9](prime[1,4,5,6,7,9])(>[10,11,12](in3[10,11,12,5])(>[](preorder[1,5,9,12])(>[]!(preorder[1,5,9,10])(preorder[1,5,9,11]))))))`

149. from v₁ is prime, v₂ ∣ v₁, 2 ≤ v₂ follows v₂ = v₁ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9](prime[1,4,5,6,7,9])(>[10](preorder[1,5,10,9])(>[](preorder[1,4,7,10])(=[10,9])))))`

150. from V₁: [0, s(v₁)] → ℕ follows ∃ Y₁: (Y₁: [0, v₁] → ℕ) ∧ V₁ = Y₁ ∪ {s(v₁) ↦ V₁(s(v₁))} 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](in2[9,10,3])(>[11](sequence[1,4,2,10,11])(>[12](in2[10,12,11])!(>[13](sequence[1,4,2,9,13])!(appendToSequence[10,12,13,11]))))))`

151. from V₁: [0, v₁] → ℕ, V₂ = V₁ ∪ {s(v₁) ↦ v₃}, V₁(v₄) = v₅ follows v₅ is a value of V₂ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](in2[9,10,3])(>[11](sequence[1,4,2,9,11])(>[12,13](appendToSequence[10,12,11,13])(>[14,15](in2[14,15,11])!(>[16](in[16,1])!(in2[16,15,13])))))))`

152. from V₁: [0, v₁] → ℕ, all values of V₁ are prime, v₃ is prime, v₃ ∣ ∏(i=0..v₁) V₁(i) follows v₃ is a value of V₁ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](sequence[1,4,2,9,10])(>[11](fold[1,3,4,5,10,2,9,11])(>[](primeseq[1,4,5,6,7,10])(>[12](prime[1,4,5,6,7,12])(>[](preorder[1,5,12,11])(inSequence[1,12,10])))))))`

153. from V₁ = [0, v₁], V₁ = [0, v₂] follows v₁ = v₂ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](interval[1,4,2,9,10])(>[11](interval[1,4,2,11,10])(=[9,11]))))`

154. from V₁: V₂ → ℕ, V₁: V₃ → ℕ follows V₂ = V₃ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](fXY[9,10,1])(>[11](fXY[9,11,1])(=[10,11]))))`

155. from V₁: [0, v₁] → ℕ, V₁: [0, v₂] → ℕ follows v₁ = v₂ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](sequence[1,4,2,9,10])(>[11](sequence[1,4,2,11,10])(=[9,11]))))`

156. from 1 ≤ v₁, v₄ · v₁ = v₂ · v₁ follows v₂ = v₄ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9](preorder[1,4,6,9])(>[10,11](in3[10,9,11,5])(>[12](in3[12,9,11,5])(=[10,12])))))`

157. from V₂ = V₁ ∪ {v₁ ↦ v₂}, all values of V₂ are prime follows all values of V₁ are prime 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10,11,12](appendToSequence[9,10,11,12])(>[](primeseq[1,4,5,6,7,12])(primeseq[1,4,5,6,7,11]))))`

158. from V₁: [0, v₁] → ℕ, V₁ is non-increasing, v₃ is a value of V₁ follows V₁(v₁) ≤ v₃ 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](sequence[1,4,2,9,10])(>[11](in2[9,11,10])(>[](ordered[1,4,10])(>[12](inSequence[1,12,10])(preorder[1,4,11,12]))))))`

159. from V₂ = V₁ ∪ {v₁ ↦ v₂}, V₂ is non-increasing follows V₁ is non-increasing 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10,11,12](appendToSequence[9,10,11,12])(>[](ordered[1,4,12])(ordered[1,4,11]))))`

160. from v₂ = ∏(i=0..v₁) V₁(i) with primes V₁(0) ≥ V₁(1) ≥ … ≥ V₁(v₁), v₂ = ∏(i=0..v₃) V₂(i) with primes V₂(0) ≥ V₂(1) ≥ … ≥ V₂(v₃) follows V₁ = V₂ — **I5** 
 `(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10,11](primeFactorSeq[1,3,4,5,2,6,7,9,10,11])(>[12,13](primeFactorSeq[1,3,4,5,2,6,7,12,13,11])(=[10,13]))))`

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
