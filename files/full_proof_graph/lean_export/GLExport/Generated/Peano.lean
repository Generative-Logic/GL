/-
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-/

import GLExport.Generated.Definitions

set_option linter.unusedVariables false

namespace GLExport

universe u

theorem peano_source_003
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α) (v5 : α), ((succ v4 v5) → ((mul v2 v4 v1) → (mul v2 v5 v3)))))) := by
  intro v1
  intro v2
  intro v3
  intro premise_1
  intro v4
  intro v5
  intro premise_2
  intro premise_3
  -- chapter_9_line_10: GL tag task formulation.
  have row_10 : (mul v2 v4 v1) := by
    exact premise_3
  -- chapter_9_line_9: GL tag task formulation.
  have row_9 : (add v1 v2 v3) := by
    exact premise_1
  -- chapter_9_line_8: GL tag task formulation.
  have row_8 : (succ v4 v5) := by
    exact premise_2
  -- chapter_9_line_7: GL tag task formulation.
  have row_7 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_9_line_6: GL tag expansion.
  have row_6 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_7
  -- chapter_9_line_5: GL tag disintegration.
  have row_5 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_6.1
  -- chapter_9_line_4: GL tag expansion.
  have row_4 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_5
  -- chapter_9_line_15: GL tag disintegration.
  have row_15 : (gl_fXYZ mul N N N) := by
    exact row_4.1.1.1.2
  -- chapter_9_line_14: GL tag expansion.
  have row_14 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_15
  -- chapter_9_line_13: GL tag disintegration.
  have row_13 : (gl_implication9 mul N) := by
    exact row_14.1.1.1.2
  -- chapter_9_line_12: GL tag expansion.
  have row_12 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication9] using row_13
  -- chapter_9_line_11: GL tag implication.
  have row_11 : (N v4) := by
    apply row_12
    exact row_10
  -- chapter_9_line_3: GL tag disintegration.
  have row_3 : (gl_implication20 N succ mul add) := by
    exact row_4.1.2
  -- chapter_9_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((mul w3 w1 w4) → (∀ (w5 : α), ((add w4 w3 w5) → (mul w3 w2 w5))))))))) := by
    simpa only [gl_implication20] using row_3
  -- chapter_9_line_1: GL tag implication.
  have row_1 : (mul v2 v5 v3) := by
    apply row_2
    exact row_11
    exact row_8
    exact row_10
    exact row_9
  exact row_1

theorem peano_source_006
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α) (v5 : α), ((add v1 v4 v5) → ((succ v4 v2) → (succ v5 v3)))))) := by
  intro v1
  intro v2
  intro v3
  intro premise_1
  intro v4
  intro v5
  intro premise_2
  intro premise_3
  -- chapter_16_line_10: GL tag task formulation.
  have row_10 : (add v1 v2 v3) := by
    exact premise_1
  -- chapter_16_line_9: GL tag task formulation.
  have row_9 : (add v1 v4 v5) := by
    exact premise_2
  -- chapter_16_line_8: GL tag task formulation.
  have row_8 : (succ v4 v2) := by
    exact premise_3
  -- chapter_16_line_7: GL tag task formulation.
  have row_7 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_16_line_6: GL tag expansion.
  have row_6 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_7
  -- chapter_16_line_5: GL tag disintegration.
  have row_5 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_6.1
  -- chapter_16_line_4: GL tag expansion.
  have row_4 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_5
  -- chapter_16_line_15: GL tag disintegration.
  have row_15 : (gl_fXY succ N N) := by
    exact row_4.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_16_line_14: GL tag expansion.
  have row_14 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_15
  -- chapter_16_line_13: GL tag disintegration.
  have row_13 : (gl_implication0 succ N) := by
    exact row_14.1.1.1
  -- chapter_16_line_12: GL tag expansion.
  have row_12 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_13
  -- chapter_16_line_11: GL tag implication.
  have row_11 : (N v4) := by
    apply row_12
    exact row_8
  -- chapter_16_line_3: GL tag disintegration.
  have row_3 : (gl_implication17 N succ add) := by
    exact row_4.1.1.1.1.1.2
  -- chapter_16_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication17] using row_3
  -- chapter_16_line_1: GL tag implication.
  have row_1 : (succ v5 v3) := by
    apply row_2
    exact row_11
    exact row_8
    exact row_9
    exact row_10
  exact row_1

theorem peano_source_011
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α), ((succ v2 v4) → (∀ (v5 : α), ((succ v3 v5) → (add v1 v4 v5))))))) := by
  intro v1
  intro v2
  intro v3
  intro premise_1
  intro v4
  intro premise_2
  intro v5
  intro premise_3
  -- chapter_29_line_10: GL tag task formulation.
  have row_10 : (add v1 v2 v3) := by
    exact premise_1
  -- chapter_29_line_9: GL tag task formulation.
  have row_9 : (succ v3 v5) := by
    exact premise_3
  -- chapter_29_line_8: GL tag task formulation.
  have row_8 : (succ v2 v4) := by
    exact premise_2
  -- chapter_29_line_7: GL tag task formulation.
  have row_7 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_29_line_6: GL tag expansion.
  have row_6 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_7
  -- chapter_29_line_5: GL tag disintegration.
  have row_5 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_6.1
  -- chapter_29_line_4: GL tag expansion.
  have row_4 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_5
  -- chapter_29_line_15: GL tag disintegration.
  have row_15 : (gl_fXYZ add N N N) := by
    exact row_4.1.1.1.1.1.1.1.1.2
  -- chapter_29_line_14: GL tag expansion.
  have row_14 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_15
  -- chapter_29_line_13: GL tag disintegration.
  have row_13 : (gl_implication9 add N) := by
    exact row_14.1.1.1.2
  -- chapter_29_line_12: GL tag expansion.
  have row_12 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication9] using row_13
  -- chapter_29_line_11: GL tag implication.
  have row_11 : (N v2) := by
    apply row_12
    exact row_10
  -- chapter_29_line_3: GL tag disintegration.
  have row_3 : (gl_implication18 N succ add) := by
    exact row_4.1.1.1.1.2
  -- chapter_29_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((succ w4 w5) → (add w3 w2 w5))))))))) := by
    simpa only [gl_implication18] using row_3
  -- chapter_29_line_1: GL tag implication.
  have row_1 : (add v1 v4 v5) := by
    apply row_2
    exact row_11
    exact row_8
    exact row_10
    exact row_9
  exact row_1

theorem peano_source_020
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α) (v3 : α), ((mul v1 v2 v3) → (∀ (v4 : α) (v5 : α), ((mul v1 v4 v5) → ((succ v4 v2) → (add v5 v1 v3)))))) := by
  intro v1
  intro v2
  intro v3
  intro premise_1
  intro v4
  intro v5
  intro premise_2
  intro premise_3
  -- chapter_52_line_10: GL tag task formulation.
  have row_10 : (mul v1 v2 v3) := by
    exact premise_1
  -- chapter_52_line_9: GL tag task formulation.
  have row_9 : (mul v1 v4 v5) := by
    exact premise_2
  -- chapter_52_line_8: GL tag task formulation.
  have row_8 : (succ v4 v2) := by
    exact premise_3
  -- chapter_52_line_7: GL tag task formulation.
  have row_7 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_52_line_6: GL tag expansion.
  have row_6 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_7
  -- chapter_52_line_5: GL tag disintegration.
  have row_5 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_6.1
  -- chapter_52_line_4: GL tag expansion.
  have row_4 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_5
  -- chapter_52_line_15: GL tag disintegration.
  have row_15 : (gl_fXY succ N N) := by
    exact row_4.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_52_line_14: GL tag expansion.
  have row_14 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_15
  -- chapter_52_line_13: GL tag disintegration.
  have row_13 : (gl_implication0 succ N) := by
    exact row_14.1.1.1
  -- chapter_52_line_12: GL tag expansion.
  have row_12 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_13
  -- chapter_52_line_11: GL tag implication.
  have row_11 : (N v4) := by
    apply row_12
    exact row_8
  -- chapter_52_line_3: GL tag disintegration.
  have row_3 : (gl_implication21 N succ mul add) := by
    exact row_4.2
  -- chapter_52_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((mul w3 w1 w4) → (∀ (w5 : α), ((mul w3 w2 w5) → (add w4 w3 w5))))))))) := by
    simpa only [gl_implication21] using row_3
  -- chapter_52_line_1: GL tag implication.
  have row_1 : (add v5 v1 v3) := by
    apply row_2
    exact row_11
    exact row_8
    exact row_9
    exact row_10
  exact row_1

theorem peano_source_023
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((¬ (v1 = v2)) → ((add v2 v1 zero) → (¬ (zero = v1))))) := by
  intro v1
  intro v2
  intro premise_1
  intro premise_2
  intro reductio
  -- chapter_57_line_16: GL tag task formulation.
  have row_16 : (zero = v1) := by
    exact reductio
  -- chapter_57_line_15: GL tag symmetry of equality.
  have row_15 : (v1 = zero) := by
    exact Eq.symm row_16
  -- chapter_57_line_14: GL tag task formulation.
  have row_14 : (add v2 v1 zero) := by
    exact premise_2
  -- chapter_57_line_13: GL tag equality1.
  have row_13 : (add v2 zero zero) := by
    have equality_source := row_14
    have equality_step_1 := row_15
    cases equality_step_1
    exact equality_source
  -- chapter_57_line_12: GL tag equality1.
  have row_12 : (add v2 zero v1) := by
    have equality_source := row_13
    have equality_step_1 := row_16
    cases equality_step_1
    exact equality_source
  -- chapter_57_line_11: GL tag task formulation.
  have row_11 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_57_line_10: GL tag expansion.
  have row_10 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_11
  -- chapter_57_line_9: GL tag disintegration.
  have row_9 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_10.1
  -- chapter_57_line_8: GL tag expansion.
  have row_8 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_9
  -- chapter_57_line_21: GL tag disintegration.
  have row_21 : (gl_fXYZ add N N N) := by
    exact row_8.1.1.1.1.1.1.1.1.2
  -- chapter_57_line_20: GL tag expansion.
  have row_20 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_21
  -- chapter_57_line_19: GL tag disintegration.
  have row_19 : (gl_implication8 add N) := by
    exact row_20.1.1.1.1
  -- chapter_57_line_18: GL tag expansion.
  have row_18 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_19
  -- chapter_57_line_17: GL tag implication.
  have row_17 : (N v2) := by
    apply row_18
    exact row_12
  -- chapter_57_line_7: GL tag disintegration.
  have row_7 : (gl_implication15 N zero add) := by
    exact row_8.1.1.1.1.1.1.1.2
  -- chapter_57_line_6: GL tag expansion.
  have row_6 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((add w1 zero w2) → (w1 = w2))))) := by
    simpa only [gl_implication15] using row_7
  -- chapter_57_line_5: GL tag implication.
  have row_5 : (v2 = v1) := by
    apply row_6
    exact row_17
    exact row_12
  -- chapter_57_line_4: GL tag equality2.
  have row_4 : (v2 = zero) := by
    exact Eq.trans row_5 row_15
  -- chapter_57_line_3: GL tag task formulation.
  have row_3 : (¬ (v1 = v2)) := by
    exact premise_1
  -- chapter_57_line_2: GL tag equality1.
  have row_2 : (¬ (v1 = zero)) := by
    have equality_source := row_3
    have equality_step_1 := row_4
    cases equality_step_1
    exact equality_source
  -- chapter_57_line_1: GL tag symmetry of inequality.
  have row_1 : (¬ (zero = v1)) := by
    exact fun equality => row_2 (Eq.symm equality)
  exact row_1 reductio

private theorem peano_source_024_induction_typing
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (assumption_10 : (add v1 v2 zero))
    : (N v1) := by
  -- chapter_58_line_10: GL tag task formulation.
  have row_10 : (add v1 v2 zero) := by
    exact assumption_10
  -- chapter_58_line_9: GL tag task formulation.
  have row_9 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_58_line_8: GL tag expansion.
  have row_8 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_9
  -- chapter_58_line_7: GL tag disintegration.
  have row_7 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_8.1
  -- chapter_58_line_6: GL tag expansion.
  have row_6 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_7
  -- chapter_58_line_5: GL tag disintegration.
  have row_5 : (gl_fXYZ add N N N) := by
    exact row_6.1.1.1.1.1.1.1.1.2
  -- chapter_58_line_4: GL tag expansion.
  have row_4 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_5
  -- chapter_58_line_3: GL tag disintegration.
  have row_3 : (gl_implication8 add N) := by
    exact row_4.1.1.1.1
  -- chapter_58_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_3
  -- chapter_58_line_1: GL tag implication.
  have row_1 : (N v1) := by
    apply row_2
    exact row_10
  exact row_1

private theorem peano_source_024_check_zero
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (assumption_2 : (v1 = zero))
    : (zero = v1) := by
  -- chapter_59_line_2: GL tag recursion.
  have row_2 : (v1 = zero) := by
    exact assumption_2
  -- chapter_59_line_1: GL tag symmetry of equality.
  have row_1 : (zero = v1) := by
    exact Eq.symm row_2
  exact row_1

private theorem peano_source_024_check_induction_condition
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (assumption_32 : (v1 = v2))
    (assumption_25 : (succ previous v1))
    (assumption_19 : (add v1 v2 zero))
    : (zero = v1) := by
  -- chapter_60_line_32: GL tag task formulation.
  have row_32 : (v1 = v2) := by
    exact assumption_32
  -- chapter_60_line_31: GL tag symmetry of equality.
  have row_31 : (v2 = v1) := by
    exact Eq.symm row_32
  -- chapter_60_line_25: GL tag recursion.
  have row_25 : (succ previous v1) := by
    exact assumption_25
  -- chapter_60_line_19: GL tag task formulation.
  have row_19 : (add v1 v2 zero) := by
    exact assumption_19
  -- chapter_60_line_30: GL tag equality1.
  have row_30 : (add v1 v1 zero) := by
    have equality_source := row_19
    have equality_step_1 := row_31
    cases equality_step_1
    exact equality_source
  -- chapter_60_line_29: GL tag equality1.
  have row_29 : (add v2 v1 zero) := by
    have equality_source := row_30
    have equality_step_1 := row_32
    cases equality_step_1
    exact equality_source
  -- chapter_60_line_8: GL tag task formulation.
  have row_8 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_60_line_7: GL tag expansion.
  have row_7 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_8
  -- chapter_60_line_6: GL tag disintegration.
  have row_6 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_7.1
  -- chapter_60_line_5: GL tag expansion.
  have row_5 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_6
  -- chapter_60_line_28: GL tag disintegration.
  have row_28 : (gl_implication17 N succ add) := by
    exact row_5.1.1.1.1.1.2
  -- chapter_60_line_27: GL tag expansion.
  have row_27 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication17] using row_28
  -- chapter_60_line_24: GL tag disintegration.
  have row_24 : (gl_fXY succ N N) := by
    exact row_5.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_60_line_23: GL tag expansion.
  have row_23 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_24
  -- chapter_60_line_22: GL tag disintegration.
  have row_22 : (gl_implication0 succ N) := by
    exact row_23.1.1.1
  -- chapter_60_line_21: GL tag expansion.
  have row_21 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_22
  -- chapter_60_line_20: GL tag implication.
  have row_20 : (N previous) := by
    apply row_21
    exact row_25
  -- chapter_60_line_15: GL tag disintegration.
  have row_15 : (gl_fXYZ add N N N) := by
    exact row_5.1.1.1.1.1.1.1.1.2
  -- chapter_60_line_14: GL tag expansion.
  have row_14 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_15
  -- chapter_60_line_18: GL tag disintegration.
  have row_18 : (gl_implication8 add N) := by
    exact row_14.1.1.1.1
  -- chapter_60_line_17: GL tag expansion.
  have row_17 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_18
  -- chapter_60_line_16: GL tag implication.
  have row_16 : (N v1) := by
    apply row_17
    exact row_19
  -- chapter_60_line_13: GL tag disintegration.
  have row_13 : (gl_implication13 N N N add) := by
    exact row_14.1.2
  -- chapter_60_line_12: GL tag expansion.
  have row_12 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication13] using row_13
  -- chapter_60_line_11: GL tag implication.
  have row_11 : (gl_existence1 N v1 previous add) := by
    apply row_12
    exact row_16
    exact row_20
  -- chapter_60_line_10: GL tag expansion.
  have row_10 : (¬ (∀ (v3 : α), ((N v3) → (¬ (add v1 previous v3))))) := by
    simpa only [gl_existence1] using row_11
  have exists_row_10 : ∃ (v3 : α), ((N v3) ∧ (add v1 previous v3)) := existsAndOfNotForallImpNot row_10
  obtain ⟨v3, witness_row_10⟩ := exists_row_10
  -- chapter_60_line_34: GL tag disintegration.
  have row_34 : (add v1 previous v3) := by
    exact witness_row_10.2
  -- chapter_60_line_33: GL tag equality1.
  have row_33 : (add v2 previous v3) := by
    have equality_source := row_34
    have equality_step_1 := row_32
    cases equality_step_1
    exact equality_source
  -- chapter_60_line_26: GL tag implication.
  have row_26 : (succ v3 zero) := by
    apply row_27
    exact row_20
    exact row_25
    exact row_33
    exact row_29
  -- chapter_60_line_9: GL tag disintegration.
  have row_9 : (N v3) := by
    exact witness_row_10.1
  -- chapter_60_line_4: GL tag disintegration.
  have row_4 : (gl_implication6 N zero succ) := by
    exact row_5.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_60_line_3: GL tag expansion.
  have row_3 : (∀ (w1 : α), ((N w1) → (¬ (succ w1 zero)))) := by
    simpa only [gl_implication6] using row_4
  -- chapter_60_line_2: GL tag implication.
  have row_2 : (¬ (succ v3 zero)) := by
    apply row_3
    exact row_9
  -- chapter_60_line_1: GL tag vacuous truth.
  have row_1 : (zero = v1) := by
    exact False.elim (row_2 row_26)
  exact row_1

theorem peano_source_024
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((v1 = v2) → ((add v1 v2 zero) → (zero = v1)))) := by
  intro v1
  intro v2
  intro premise_1
  intro premise_2
  have inductionMember : N v1 := by
    have typingRule := peano_source_024_induction_typing N zero succ add mul one anchor relationalInduction
    exact typingRule v1 v2 premise_2
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v2 : α), ((zero = v2) → ((add zero v2 zero) → (zero = zero)))) := by
    intro v2
    intro base_premise_1
    intro base_premise_2
    have zeroRule := peano_source_024_check_zero N zero succ add mul one anchor relationalInduction
    exact zeroRule zero rfl
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v2 : α), ((induction_n = v2) → ((add induction_n v2 zero) → (zero = induction_n)))) → ∀ induction_m, succ induction_n induction_m → (∀ (v2 : α), ((induction_m = v2) → ((add induction_m v2 zero) → (zero = induction_m)))) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro v2
    intro step_premise_1
    intro step_premise_2
    have step_induction_assumption_1 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have stepRule := peano_source_024_check_induction_condition N zero succ add mul one anchor relationalInduction
    exact stepRule induction_n induction_m v2 step_premise_1 step_induction_assumption_1 step_premise_2
  have inductionProperty : (∀ (v2 : α), ((v1 = v2) → ((add v1 v2 zero) → (zero = v1)))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v2 : α), ((induction_value = v2) → ((add induction_value v2 zero) → (zero = induction_value)))))
      v1
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v2 premise_1 premise_2

theorem peano_source_026
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (v3 : α) (v4 : α), ((succ v3 v4) → ((¬ (v2 = v4)) → (¬ (v1 = v3))))))) := by
  intro v1
  intro v2
  intro premise_1
  intro v3
  intro v4
  intro premise_2
  intro premise_3
  intro reductio
  -- chapter_64_line_15: GL tag task formulation.
  have row_15 : (v1 = v3) := by
    exact reductio
  -- chapter_64_line_14: GL tag task formulation.
  have row_14 : (succ v1 v2) := by
    exact premise_1
  -- chapter_64_line_13: GL tag equality1.
  have row_13 : (succ v3 v2) := by
    have equality_source := row_14
    have equality_step_1 := row_15
    cases equality_step_1
    exact equality_source
  -- chapter_64_line_12: GL tag task formulation.
  have row_12 : (succ v3 v4) := by
    exact premise_2
  -- chapter_64_line_11: GL tag task formulation.
  have row_11 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_64_line_10: GL tag expansion.
  have row_10 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_11
  -- chapter_64_line_9: GL tag disintegration.
  have row_9 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_10.1
  -- chapter_64_line_8: GL tag expansion.
  have row_8 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_9
  -- chapter_64_line_7: GL tag disintegration.
  have row_7 : (gl_fXY succ N N) := by
    exact row_8.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_64_line_6: GL tag expansion.
  have row_6 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_7
  -- chapter_64_line_18: GL tag disintegration.
  have row_18 : (gl_implication0 succ N) := by
    exact row_6.1.1.1
  -- chapter_64_line_17: GL tag expansion.
  have row_17 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_18
  -- chapter_64_line_16: GL tag implication.
  have row_16 : (N v3) := by
    apply row_17
    exact row_12
  -- chapter_64_line_5: GL tag disintegration.
  have row_5 : (gl_implication5 N succ) := by
    exact row_6.2
  -- chapter_64_line_4: GL tag expansion.
  have row_4 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α), ((succ w1 w3) → (w2 = w3))))))) := by
    simpa only [gl_implication5] using row_5
  -- chapter_64_line_3: GL tag implication.
  have row_3 : (v2 = v4) := by
    apply row_4
    exact row_16
    exact row_13
    exact row_12
  -- chapter_64_line_2: GL tag task formulation.
  have row_2 : (¬ (v2 = v4)) := by
    exact premise_3
  -- chapter_64_line_1: GL tag contradiction.
  have row_1 : (¬ (v1 = v3)) := by
    exact False.elim (row_2 row_3)
  exact row_1 reductio

theorem peano_source_027
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (v3 : α) (v4 : α), ((succ v3 v4) → ((v1 = v3) → (v2 = v4)))))) := by
  intro v1
  intro v2
  intro premise_1
  intro v3
  intro v4
  intro premise_2
  intro premise_3
  -- chapter_65_line_13: GL tag task formulation.
  have row_13 : (v1 = v3) := by
    exact premise_3
  -- chapter_65_line_12: GL tag task formulation.
  have row_12 : (succ v1 v2) := by
    exact premise_1
  -- chapter_65_line_11: GL tag equality1.
  have row_11 : (succ v3 v2) := by
    have equality_source := row_12
    have equality_step_1 := row_13
    cases equality_step_1
    exact equality_source
  -- chapter_65_line_10: GL tag task formulation.
  have row_10 : (succ v3 v4) := by
    exact premise_2
  -- chapter_65_line_9: GL tag task formulation.
  have row_9 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_65_line_8: GL tag expansion.
  have row_8 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_9
  -- chapter_65_line_7: GL tag disintegration.
  have row_7 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_8.1
  -- chapter_65_line_6: GL tag expansion.
  have row_6 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_7
  -- chapter_65_line_5: GL tag disintegration.
  have row_5 : (gl_fXY succ N N) := by
    exact row_6.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_65_line_4: GL tag expansion.
  have row_4 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_5
  -- chapter_65_line_16: GL tag disintegration.
  have row_16 : (gl_implication0 succ N) := by
    exact row_4.1.1.1
  -- chapter_65_line_15: GL tag expansion.
  have row_15 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_16
  -- chapter_65_line_14: GL tag implication.
  have row_14 : (N v3) := by
    apply row_15
    exact row_10
  -- chapter_65_line_3: GL tag disintegration.
  have row_3 : (gl_implication5 N succ) := by
    exact row_4.2
  -- chapter_65_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α), ((succ w1 w3) → (w2 = w3))))))) := by
    simpa only [gl_implication5] using row_3
  -- chapter_65_line_1: GL tag implication.
  have row_1 : (v2 = v4) := by
    apply row_2
    exact row_14
    exact row_11
    exact row_10
  exact row_1

theorem peano_source_030
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (v3 : α), ((succ v3 v2) → (v1 = v3))))) := by
  intro v1
  intro v2
  intro premise_1
  intro v3
  intro premise_2
  -- chapter_68_line_12: GL tag task formulation.
  have row_12 : (succ v3 v2) := by
    exact premise_2
  -- chapter_68_line_10: GL tag variable copy.
  have row_10 : (v2 = v2) := by
    rfl
  -- chapter_68_line_11: GL tag equality1.
  have row_11 : (succ v3 v2) := by
    have equality_source := row_12
    have equality_step_1 := row_10
    cases equality_step_1
    exact equality_source
  -- chapter_68_line_9: GL tag task formulation.
  have row_9 : (succ v1 v2) := by
    exact premise_1
  -- chapter_68_line_8: GL tag equality1.
  have row_8 : (succ v1 v2) := by
    have equality_source := row_9
    have equality_step_1 := row_10
    cases equality_step_1
    exact equality_source
  -- chapter_68_line_7: GL tag task formulation.
  have row_7 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_68_line_6: GL tag expansion.
  have row_6 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_7
  -- chapter_68_line_5: GL tag disintegration.
  have row_5 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_6.1
  -- chapter_68_line_4: GL tag expansion.
  have row_4 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_5
  -- chapter_68_line_17: GL tag disintegration.
  have row_17 : (gl_fXY succ N N) := by
    exact row_4.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_68_line_16: GL tag expansion.
  have row_16 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_17
  -- chapter_68_line_15: GL tag disintegration.
  have row_15 : (gl_implication1 succ N) := by
    exact row_16.1.1.2
  -- chapter_68_line_14: GL tag expansion.
  have row_14 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using row_15
  -- chapter_68_line_13: GL tag implication.
  have row_13 : (N v2) := by
    apply row_14
    exact row_8
  -- chapter_68_line_3: GL tag disintegration.
  have row_3 : (gl_implication7 N succ) := by
    exact row_4.1.1.1.1.1.1.1.1.1.2
  -- chapter_68_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w2 w1) → (∀ (w3 : α), ((succ w3 w1) → (w2 = w3))))))) := by
    simpa only [gl_implication7] using row_3
  -- chapter_68_line_1: GL tag implication.
  have row_1 : (v1 = v3) := by
    apply row_2
    exact row_13
    exact row_8
    exact row_11
  exact row_1

theorem peano_source_031
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → ((succ zero v2) → (zero = v1)))) := by
  intro v1
  intro v2
  intro premise_1
  intro premise_2
  -- chapter_69_line_12: GL tag task formulation.
  have row_12 : (succ v1 v2) := by
    exact premise_1
  -- chapter_69_line_10: GL tag variable copy.
  have row_10 : (v2 = v2) := by
    rfl
  -- chapter_69_line_11: GL tag equality1.
  have row_11 : (succ v1 v2) := by
    have equality_source := row_12
    have equality_step_1 := row_10
    cases equality_step_1
    exact equality_source
  -- chapter_69_line_9: GL tag task formulation.
  have row_9 : (succ zero v2) := by
    exact premise_2
  -- chapter_69_line_8: GL tag equality1.
  have row_8 : (succ zero v2) := by
    have equality_source := row_9
    have equality_step_1 := row_10
    cases equality_step_1
    exact equality_source
  -- chapter_69_line_7: GL tag task formulation.
  have row_7 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_69_line_6: GL tag expansion.
  have row_6 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_7
  -- chapter_69_line_5: GL tag disintegration.
  have row_5 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_6.1
  -- chapter_69_line_4: GL tag expansion.
  have row_4 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_5
  -- chapter_69_line_17: GL tag disintegration.
  have row_17 : (gl_fXY succ N N) := by
    exact row_4.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_69_line_16: GL tag expansion.
  have row_16 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_17
  -- chapter_69_line_15: GL tag disintegration.
  have row_15 : (gl_implication1 succ N) := by
    exact row_16.1.1.2
  -- chapter_69_line_14: GL tag expansion.
  have row_14 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using row_15
  -- chapter_69_line_13: GL tag implication.
  have row_13 : (N v2) := by
    apply row_14
    exact row_8
  -- chapter_69_line_3: GL tag disintegration.
  have row_3 : (gl_implication7 N succ) := by
    exact row_4.1.1.1.1.1.1.1.1.1.2
  -- chapter_69_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w2 w1) → (∀ (w3 : α), ((succ w3 w1) → (w2 = w3))))))) := by
    simpa only [gl_implication7] using row_3
  -- chapter_69_line_1: GL tag implication.
  have row_1 : (zero = v1) := by
    apply row_2
    exact row_13
    exact row_8
    exact row_11
  exact row_1

theorem peano_source_032
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → ((succ v2 one) → (¬ (one = v1))))) := by
  intro v1
  intro v2
  intro premise_1
  intro premise_2
  intro reductio
  -- chapter_70_line_27: GL tag task formulation.
  have row_27 : (succ v2 one) := by
    exact premise_2
  -- chapter_70_line_14: GL tag task formulation.
  have row_14 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_70_line_13: GL tag expansion.
  have row_13 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_14
  -- chapter_70_line_21: GL tag disintegration.
  have row_21 : (succ zero one) := by
    exact row_13.2
  -- chapter_70_line_12: GL tag disintegration.
  have row_12 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_13.1
  -- chapter_70_line_11: GL tag expansion.
  have row_11 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_12
  -- chapter_70_line_24: GL tag disintegration.
  have row_24 : (gl_implication7 N succ) := by
    exact row_11.1.1.1.1.1.1.1.1.1.2
  -- chapter_70_line_23: GL tag expansion.
  have row_23 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w2 w1) → (∀ (w3 : α), ((succ w3 w1) → (w2 = w3))))))) := by
    simpa only [gl_implication7] using row_24
  -- chapter_70_line_20: GL tag disintegration.
  have row_20 : (gl_fXY succ N N) := by
    exact row_11.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_70_line_19: GL tag expansion.
  have row_19 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_20
  -- chapter_70_line_18: GL tag disintegration.
  have row_18 : (gl_implication1 succ N) := by
    exact row_19.1.1.2
  -- chapter_70_line_17: GL tag expansion.
  have row_17 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using row_18
  -- chapter_70_line_16: GL tag implication.
  have row_16 : (N one) := by
    apply row_17
    exact row_21
  -- chapter_70_line_10: GL tag disintegration.
  have row_10 : (gl_implication6 N zero succ) := by
    exact row_11.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_70_line_9: GL tag expansion.
  have row_9 : (∀ (w1 : α), ((N w1) → (¬ (succ w1 zero)))) := by
    simpa only [gl_implication6] using row_10
  -- chapter_70_line_5: GL tag task formulation.
  have row_5 : (one = v1) := by
    exact reductio
  -- chapter_70_line_26: GL tag equality1.
  have row_26 : (succ v2 v1) := by
    have equality_source := row_27
    have equality_step_1 := row_5
    cases equality_step_1
    exact equality_source
  -- chapter_70_line_25: GL tag equality1.
  have row_25 : (succ zero v1) := by
    have equality_source := row_21
    have equality_step_1 := row_5
    cases equality_step_1
    exact equality_source
  -- chapter_70_line_15: GL tag equality1.
  have row_15 : (N v1) := by
    have equality_source := row_16
    have equality_step_1 := row_5
    cases equality_step_1
    exact equality_source
  -- chapter_70_line_22: GL tag implication.
  have row_22 : (zero = v2) := by
    apply row_23
    exact row_15
    exact row_25
    exact row_26
  -- chapter_70_line_8: GL tag implication.
  have row_8 : (¬ (succ v1 zero)) := by
    apply row_9
    exact row_15
  -- chapter_70_line_4: GL tag symmetry of equality.
  have row_4 : (v1 = one) := by
    exact Eq.symm row_5
  -- chapter_70_line_7: GL tag equality1.
  have row_7 : (¬ (succ one zero)) := by
    have equality_source := row_8
    have equality_step_1 := row_4
    cases equality_step_1
    exact equality_source
  -- chapter_70_line_6: GL tag equality1.
  have row_6 : (¬ (succ one v2)) := by
    have equality_source := row_7
    have equality_step_1 := row_22
    cases equality_step_1
    exact equality_source
  -- chapter_70_line_3: GL tag task formulation.
  have row_3 : (succ v1 v2) := by
    exact premise_1
  -- chapter_70_line_2: GL tag equality1.
  have row_2 : (succ one v2) := by
    have equality_source := row_3
    have equality_step_1 := row_4
    cases equality_step_1
    exact equality_source
  -- chapter_70_line_1: GL tag contradiction.
  have row_1 : (¬ (one = v1)) := by
    exact False.elim (row_6 row_2)
  exact row_1 reductio

private theorem peano_source_033_induction_typing
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (assumption_10 : (succ v1 v2))
    : (N v1) := by
  -- chapter_71_line_10: GL tag task formulation.
  have row_10 : (succ v1 v2) := by
    exact assumption_10
  -- chapter_71_line_9: GL tag task formulation.
  have row_9 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_71_line_8: GL tag expansion.
  have row_8 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_9
  -- chapter_71_line_7: GL tag disintegration.
  have row_7 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_8.1
  -- chapter_71_line_6: GL tag expansion.
  have row_6 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_7
  -- chapter_71_line_5: GL tag disintegration.
  have row_5 : (gl_fXY succ N N) := by
    exact row_6.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_71_line_4: GL tag expansion.
  have row_4 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_5
  -- chapter_71_line_3: GL tag disintegration.
  have row_3 : (gl_implication0 succ N) := by
    exact row_4.1.1.1
  -- chapter_71_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_3
  -- chapter_71_line_1: GL tag implication.
  have row_1 : (N v1) := by
    apply row_2
    exact row_10
  exact row_1

private theorem peano_source_033_check_zero
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (assumption_25 : (v1 = zero))
    (assumption_24 : (succ v1 v2))
    : (add one v1 v2) := by
  -- chapter_72_line_25: GL tag recursion.
  have row_25 : (v1 = zero) := by
    exact assumption_25
  -- chapter_72_line_36: GL tag symmetry of equality.
  have row_36 : (zero = v1) := by
    exact Eq.symm row_25
  -- chapter_72_line_24: GL tag task formulation.
  have row_24 : (succ v1 v2) := by
    exact assumption_24
  -- chapter_72_line_34: GL tag equality1.
  have row_34 : (succ zero v2) := by
    have equality_source := row_24
    have equality_step_1 := row_25
    cases equality_step_1
    exact equality_source
  -- chapter_72_line_14: GL tag task formulation.
  have row_14 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_72_line_13: GL tag expansion.
  have row_13 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_14
  -- chapter_72_line_20: GL tag disintegration.
  have row_20 : (succ zero one) := by
    exact row_13.2
  -- chapter_72_line_12: GL tag disintegration.
  have row_12 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_13.1
  -- chapter_72_line_11: GL tag expansion.
  have row_11 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_12
  -- chapter_72_line_35: GL tag disintegration.
  have row_35 : (N zero) := by
    exact row_11.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_72_line_30: GL tag disintegration.
  have row_30 : (gl_implication15 N zero add) := by
    exact row_11.1.1.1.1.1.1.1.2
  -- chapter_72_line_29: GL tag expansion.
  have row_29 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((add w1 zero w2) → (w1 = w2))))) := by
    simpa only [gl_implication15] using row_30
  -- chapter_72_line_19: GL tag disintegration.
  have row_19 : (gl_fXY succ N N) := by
    exact row_11.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_72_line_18: GL tag expansion.
  have row_18 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_19
  -- chapter_72_line_33: GL tag disintegration.
  have row_33 : (gl_implication5 N succ) := by
    exact row_18.2
  -- chapter_72_line_32: GL tag expansion.
  have row_32 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α), ((succ w1 w3) → (w2 = w3))))))) := by
    simpa only [gl_implication5] using row_33
  -- chapter_72_line_31: GL tag implication.
  have row_31 : (one = v2) := by
    apply row_32
    exact row_35
    exact row_20
    exact row_34
  -- chapter_72_line_23: GL tag disintegration.
  have row_23 : (gl_implication0 succ N) := by
    exact row_18.1.1.1
  -- chapter_72_line_22: GL tag expansion.
  have row_22 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_23
  -- chapter_72_line_21: GL tag implication.
  have row_21 : (N v1) := by
    apply row_22
    exact row_24
  -- chapter_72_line_17: GL tag disintegration.
  have row_17 : (gl_implication1 succ N) := by
    exact row_18.1.1.2
  -- chapter_72_line_16: GL tag expansion.
  have row_16 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using row_17
  -- chapter_72_line_15: GL tag implication.
  have row_15 : (N one) := by
    apply row_16
    exact row_20
  -- chapter_72_line_10: GL tag disintegration.
  have row_10 : (gl_fXYZ add N N N) := by
    exact row_11.1.1.1.1.1.1.1.1.2
  -- chapter_72_line_9: GL tag expansion.
  have row_9 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_10
  -- chapter_72_line_8: GL tag disintegration.
  have row_8 : (gl_implication13 N N N add) := by
    exact row_9.1.2
  -- chapter_72_line_7: GL tag expansion.
  have row_7 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication13] using row_8
  -- chapter_72_line_6: GL tag implication.
  have row_6 : (gl_existence1 N one v1 add) := by
    apply row_7
    exact row_15
    exact row_21
  -- chapter_72_line_5: GL tag expansion.
  have row_5 : (¬ (∀ (v3 : α), ((N v3) → (¬ (add one v1 v3))))) := by
    simpa only [gl_existence1] using row_6
  have exists_row_5 : ∃ (v3 : α), ((N v3) ∧ (add one v1 v3)) := existsAndOfNotForallImpNot row_5
  obtain ⟨v3, witness_row_5⟩ := exists_row_5
  -- chapter_72_line_4: GL tag disintegration.
  have row_4 : (add one v1 v3) := by
    exact witness_row_5.2
  -- chapter_72_line_3: GL tag equality1.
  have row_3 : (add one zero v3) := by
    have equality_source := row_4
    have equality_step_1 := row_25
    cases equality_step_1
    exact equality_source
  -- chapter_72_line_28: GL tag implication.
  have row_28 : (one = v3) := by
    apply row_29
    exact row_15
    exact row_3
  -- chapter_72_line_27: GL tag symmetry of equality.
  have row_27 : (v3 = one) := by
    exact Eq.symm row_28
  -- chapter_72_line_26: GL tag equality2.
  have row_26 : (v3 = v2) := by
    exact Eq.trans row_27 row_31
  -- chapter_72_line_2: GL tag equality1.
  have row_2 : (add one zero v2) := by
    have equality_source := row_3
    have equality_step_1 := row_26
    cases equality_step_1
    exact equality_source
  -- chapter_72_line_1: GL tag equality1.
  have row_1 : (add one v1 v2) := by
    have equality_source := row_2
    have equality_step_1 := row_36
    cases equality_step_1
    exact equality_source
  exact row_1

private theorem peano_source_033_check_induction_condition
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (assumption_24 : (succ previous v1))
    (assumption_23 : (∀ (w1 : α), ((succ previous w1) → (add one previous w1))))
    (assumption_9 : (succ v1 v2))
    : (add one v1 v2) := by
  -- chapter_73_line_24: GL tag recursion.
  have row_24 : (succ previous v1) := by
    exact assumption_24
  -- chapter_73_line_23: GL tag recursion.
  have row_23 : (∀ (w1 : α), ((succ previous w1) → (add one previous w1))) := by
    exact assumption_23
  -- chapter_73_line_22: GL tag implication.
  have row_22 : (add one previous v1) := by
    apply row_23
    exact row_24
  -- chapter_73_line_9: GL tag task formulation.
  have row_9 : (succ v1 v2) := by
    exact assumption_9
  -- chapter_73_line_8: GL tag task formulation.
  have row_8 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_73_line_7: GL tag expansion.
  have row_7 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_8
  -- chapter_73_line_6: GL tag disintegration.
  have row_6 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_7.1
  -- chapter_73_line_5: GL tag expansion.
  have row_5 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_6
  -- chapter_73_line_21: GL tag disintegration.
  have row_21 : (gl_fXYZ add N N N) := by
    exact row_5.1.1.1.1.1.1.1.1.2
  -- chapter_73_line_20: GL tag expansion.
  have row_20 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_21
  -- chapter_73_line_19: GL tag disintegration.
  have row_19 : (gl_implication9 add N) := by
    exact row_20.1.1.1.2
  -- chapter_73_line_18: GL tag expansion.
  have row_18 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication9] using row_19
  -- chapter_73_line_17: GL tag implication.
  have row_17 : (N previous) := by
    apply row_18
    exact row_22
  -- chapter_73_line_16: GL tag disintegration.
  have row_16 : (gl_fXY succ N N) := by
    exact row_5.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_73_line_15: GL tag expansion.
  have row_15 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_16
  -- chapter_73_line_27: GL tag disintegration.
  have row_27 : (gl_implication5 N succ) := by
    exact row_15.2
  -- chapter_73_line_26: GL tag expansion.
  have row_26 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α), ((succ w1 w3) → (w2 = w3))))))) := by
    simpa only [gl_implication5] using row_27
  -- chapter_73_line_14: GL tag disintegration.
  have row_14 : (gl_implication4 N N succ) := by
    exact row_15.1.2
  -- chapter_73_line_13: GL tag expansion.
  have row_13 : (∀ (w1 : α), ((N w1) → (gl_existence0 N w1 succ))) := by
    simpa only [gl_implication4] using row_14
  -- chapter_73_line_12: GL tag implication.
  have row_12 : (gl_existence0 N previous succ) := by
    apply row_13
    exact row_17
  -- chapter_73_line_11: GL tag expansion.
  have row_11 : (¬ (∀ (v3 : α), ((N v3) → (¬ (succ previous v3))))) := by
    simpa only [gl_existence0] using row_12
  have exists_row_11 : ∃ (v3 : α), ((N v3) ∧ (succ previous v3)) := existsAndOfNotForallImpNot row_11
  obtain ⟨v3, witness_row_11⟩ := exists_row_11
  -- chapter_73_line_10: GL tag disintegration.
  have row_10 : (succ previous v3) := by
    exact witness_row_11.2
  -- chapter_73_line_25: GL tag implication.
  have row_25 : (v3 = v1) := by
    apply row_26
    exact row_17
    exact row_10
    exact row_24
  -- chapter_73_line_4: GL tag disintegration.
  have row_4 : (gl_implication18 N succ add) := by
    exact row_5.1.1.1.1.2
  -- chapter_73_line_3: GL tag expansion.
  have row_3 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((succ w4 w5) → (add w3 w2 w5))))))))) := by
    simpa only [gl_implication18] using row_4
  -- chapter_73_line_2: GL tag implication.
  have row_2 : (add one v3 v2) := by
    apply row_3
    exact row_17
    exact row_10
    exact row_22
    exact row_9
  -- chapter_73_line_1: GL tag equality1.
  have row_1 : (add one v1 v2) := by
    have equality_source := row_2
    have equality_step_1 := row_25
    cases equality_step_1
    exact equality_source
  exact row_1

theorem peano_source_033
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (add one v1 v2))) := by
  intro v1
  intro v2
  intro premise_1
  have inductionMember : N v1 := by
    have typingRule := peano_source_033_induction_typing N zero succ add mul one anchor relationalInduction
    exact typingRule v1 v2 premise_1
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v2 : α), ((succ zero v2) → (add one zero v2))) := by
    intro v2
    intro base_premise_1
    have zeroRule := peano_source_033_check_zero N zero succ add mul one anchor relationalInduction
    exact zeroRule zero v2 rfl base_premise_1
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v2 : α), ((succ induction_n v2) → (add one induction_n v2))) → ∀ induction_m, succ induction_n induction_m → (∀ (v2 : α), ((succ induction_m v2) → (add one induction_m v2))) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro v2
    intro step_premise_1
    have step_induction_assumption_1 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have step_induction_assumption_2 :
        (∀ (w1 : α), ((succ induction_n w1) → (add one induction_n w1))) := by
      intro w1
      intro step_induction_assumption_2_premise_1
      apply induction_hypothesis
      all_goals assumption
    have stepRule := peano_source_033_check_induction_condition N zero succ add mul one anchor relationalInduction
    exact stepRule induction_n induction_m v2 step_induction_assumption_1 step_induction_assumption_2 step_premise_1
  have inductionProperty : (∀ (v2 : α), ((succ v1 v2) → (add one v1 v2))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v2 : α), ((succ induction_value v2) → (add one induction_value v2))))
      v1
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v2 premise_1

theorem peano_source_039
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((add v1 zero v2) → ((N v1) → (v1 = v2)))) := by
  intro v1
  intro v2
  intro premise_1
  intro premise_2
  -- chapter_85_line_9: GL tag task formulation.
  have row_9 : (N v1) := by
    exact premise_2
  -- chapter_85_line_8: GL tag task formulation.
  have row_8 : (add v1 zero v2) := by
    exact premise_1
  -- chapter_85_line_7: GL tag task formulation.
  have row_7 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_85_line_6: GL tag expansion.
  have row_6 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_7
  -- chapter_85_line_5: GL tag disintegration.
  have row_5 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_6.1
  -- chapter_85_line_4: GL tag expansion.
  have row_4 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_5
  -- chapter_85_line_3: GL tag disintegration.
  have row_3 : (gl_implication15 N zero add) := by
    exact row_4.1.1.1.1.1.1.1.2
  -- chapter_85_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((add w1 zero w2) → (w1 = w2))))) := by
    simpa only [gl_implication15] using row_3
  -- chapter_85_line_1: GL tag implication.
  have row_1 : (v1 = v2) := by
    apply row_2
    exact row_9
    exact row_8
  exact row_1

theorem peano_source_041
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((mul v1 zero v2) → ((N v1) → (zero = v2)))) := by
  intro v1
  intro v2
  intro premise_1
  intro premise_2
  -- chapter_89_line_10: GL tag task formulation.
  have row_10 : (N v1) := by
    exact premise_2
  -- chapter_89_line_9: GL tag task formulation.
  have row_9 : (mul v1 zero v2) := by
    exact premise_1
  -- chapter_89_line_8: GL tag task formulation.
  have row_8 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_89_line_7: GL tag expansion.
  have row_7 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_8
  -- chapter_89_line_6: GL tag disintegration.
  have row_6 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_7.1
  -- chapter_89_line_5: GL tag expansion.
  have row_5 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_6
  -- chapter_89_line_4: GL tag disintegration.
  have row_4 : (gl_implication19 N zero mul) := by
    exact row_5.1.1.2
  -- chapter_89_line_3: GL tag expansion.
  have row_3 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((mul w1 zero w2) → (w2 = zero))))) := by
    simpa only [gl_implication19] using row_4
  -- chapter_89_line_2: GL tag implication.
  have row_2 : (v2 = zero) := by
    apply row_3
    exact row_10
    exact row_9
  -- chapter_89_line_1: GL tag symmetry of equality.
  have row_1 : (zero = v2) := by
    exact Eq.symm row_2
  exact row_1

private theorem peano_source_045_induction_typing
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (assumption_10 : (add v1 v2 zero))
    : (N v2) := by
  -- chapter_97_line_10: GL tag task formulation.
  have row_10 : (add v1 v2 zero) := by
    exact assumption_10
  -- chapter_97_line_9: GL tag task formulation.
  have row_9 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_97_line_8: GL tag expansion.
  have row_8 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_9
  -- chapter_97_line_7: GL tag disintegration.
  have row_7 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_8.1
  -- chapter_97_line_6: GL tag expansion.
  have row_6 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_7
  -- chapter_97_line_5: GL tag disintegration.
  have row_5 : (gl_fXYZ add N N N) := by
    exact row_6.1.1.1.1.1.1.1.1.2
  -- chapter_97_line_4: GL tag expansion.
  have row_4 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_5
  -- chapter_97_line_3: GL tag disintegration.
  have row_3 : (gl_implication9 add N) := by
    exact row_4.1.1.1.2
  -- chapter_97_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication9] using row_3
  -- chapter_97_line_1: GL tag implication.
  have row_1 : (N v2) := by
    apply row_2
    exact row_10
  exact row_1

private theorem peano_source_045_check_zero
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v2 : α)
    (assumption_2 : (v2 = zero))
    : (zero = v2) := by
  -- chapter_98_line_2: GL tag recursion.
  have row_2 : (v2 = zero) := by
    exact assumption_2
  -- chapter_98_line_1: GL tag symmetry of equality.
  have row_1 : (zero = v2) := by
    exact Eq.symm row_2
  exact row_1

private theorem peano_source_045_check_induction_condition
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (assumption_26 : (add v1 v2 zero))
    (assumption_22 : (succ previous v2))
    (assumption_16 : (N v1))
    : (zero = v2) := by
  -- chapter_99_line_26: GL tag task formulation.
  have row_26 : (add v1 v2 zero) := by
    exact assumption_26
  -- chapter_99_line_22: GL tag recursion.
  have row_22 : (succ previous v2) := by
    exact assumption_22
  -- chapter_99_line_16: GL tag task formulation.
  have row_16 : (N v1) := by
    exact assumption_16
  -- chapter_99_line_8: GL tag task formulation.
  have row_8 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_99_line_7: GL tag expansion.
  have row_7 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_8
  -- chapter_99_line_6: GL tag disintegration.
  have row_6 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_7.1
  -- chapter_99_line_5: GL tag expansion.
  have row_5 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_6
  -- chapter_99_line_25: GL tag disintegration.
  have row_25 : (gl_implication17 N succ add) := by
    exact row_5.1.1.1.1.1.2
  -- chapter_99_line_24: GL tag expansion.
  have row_24 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication17] using row_25
  -- chapter_99_line_21: GL tag disintegration.
  have row_21 : (gl_fXY succ N N) := by
    exact row_5.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_99_line_20: GL tag expansion.
  have row_20 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_21
  -- chapter_99_line_19: GL tag disintegration.
  have row_19 : (gl_implication0 succ N) := by
    exact row_20.1.1.1
  -- chapter_99_line_18: GL tag expansion.
  have row_18 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_19
  -- chapter_99_line_17: GL tag implication.
  have row_17 : (N previous) := by
    apply row_18
    exact row_22
  -- chapter_99_line_15: GL tag disintegration.
  have row_15 : (gl_fXYZ add N N N) := by
    exact row_5.1.1.1.1.1.1.1.1.2
  -- chapter_99_line_14: GL tag expansion.
  have row_14 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_15
  -- chapter_99_line_13: GL tag disintegration.
  have row_13 : (gl_implication13 N N N add) := by
    exact row_14.1.2
  -- chapter_99_line_12: GL tag expansion.
  have row_12 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication13] using row_13
  -- chapter_99_line_11: GL tag implication.
  have row_11 : (gl_existence1 N v1 previous add) := by
    apply row_12
    exact row_16
    exact row_17
  -- chapter_99_line_10: GL tag expansion.
  have row_10 : (¬ (∀ (v3 : α), ((N v3) → (¬ (add v1 previous v3))))) := by
    simpa only [gl_existence1] using row_11
  have exists_row_10 : ∃ (v3 : α), ((N v3) ∧ (add v1 previous v3)) := existsAndOfNotForallImpNot row_10
  obtain ⟨v3, witness_row_10⟩ := exists_row_10
  -- chapter_99_line_27: GL tag disintegration.
  have row_27 : (add v1 previous v3) := by
    exact witness_row_10.2
  -- chapter_99_line_23: GL tag implication.
  have row_23 : (succ v3 zero) := by
    apply row_24
    exact row_17
    exact row_22
    exact row_27
    exact row_26
  -- chapter_99_line_9: GL tag disintegration.
  have row_9 : (N v3) := by
    exact witness_row_10.1
  -- chapter_99_line_4: GL tag disintegration.
  have row_4 : (gl_implication6 N zero succ) := by
    exact row_5.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_99_line_3: GL tag expansion.
  have row_3 : (∀ (w1 : α), ((N w1) → (¬ (succ w1 zero)))) := by
    simpa only [gl_implication6] using row_4
  -- chapter_99_line_2: GL tag implication.
  have row_2 : (¬ (succ v3 zero)) := by
    apply row_3
    exact row_9
  -- chapter_99_line_1: GL tag vacuous truth.
  have row_1 : (zero = v2) := by
    exact False.elim (row_2 row_23)
  exact row_1

theorem peano_source_045
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((add v1 v2 zero) → ((N v1) → (zero = v2)))) := by
  intro v1
  intro v2
  intro premise_1
  intro premise_2
  have inductionMember : N v2 := by
    have typingRule := peano_source_045_induction_typing N zero succ add mul one anchor relationalInduction
    exact typingRule v1 v2 premise_1
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v1 : α), ((add v1 zero zero) → ((N v1) → (zero = zero)))) := by
    intro v1
    intro base_premise_1
    intro base_premise_2
    have zeroRule := peano_source_045_check_zero N zero succ add mul one anchor relationalInduction
    exact zeroRule zero rfl
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v1 : α), ((add v1 induction_n zero) → ((N v1) → (zero = induction_n)))) → ∀ induction_m, succ induction_n induction_m → (∀ (v1 : α), ((add v1 induction_m zero) → ((N v1) → (zero = induction_m)))) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro v1
    intro step_premise_1
    intro step_premise_2
    have step_induction_assumption_1 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have stepRule := peano_source_045_check_induction_condition N zero succ add mul one anchor relationalInduction
    exact stepRule induction_n v1 induction_m step_premise_1 step_induction_assumption_1 step_premise_2
  have inductionProperty : (∀ (v1 : α), ((add v1 v2 zero) → ((N v1) → (zero = v2)))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v1 : α), ((add v1 induction_value zero) → ((N v1) → (zero = induction_value)))))
      v2
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v1 premise_1 premise_2

private theorem peano_source_046_induction_typing
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (assumption_10 : (add v1 v2 zero))
    : (N v2) := by
  -- chapter_100_line_10: GL tag task formulation.
  have row_10 : (add v1 v2 zero) := by
    exact assumption_10
  -- chapter_100_line_9: GL tag task formulation.
  have row_9 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_100_line_8: GL tag expansion.
  have row_8 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_9
  -- chapter_100_line_7: GL tag disintegration.
  have row_7 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_8.1
  -- chapter_100_line_6: GL tag expansion.
  have row_6 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_7
  -- chapter_100_line_5: GL tag disintegration.
  have row_5 : (gl_fXYZ add N N N) := by
    exact row_6.1.1.1.1.1.1.1.1.2
  -- chapter_100_line_4: GL tag expansion.
  have row_4 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_5
  -- chapter_100_line_3: GL tag disintegration.
  have row_3 : (gl_implication9 add N) := by
    exact row_4.1.1.1.2
  -- chapter_100_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication9] using row_3
  -- chapter_100_line_1: GL tag implication.
  have row_1 : (N v2) := by
    apply row_2
    exact row_10
  exact row_1

private theorem peano_source_046_check_zero
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (assumption_11 : (v2 = zero))
    (assumption_10 : (add v1 v2 zero))
    : (v1 = v2) := by
  -- chapter_101_line_11: GL tag recursion.
  have row_11 : (v2 = zero) := by
    exact assumption_11
  -- chapter_101_line_12: GL tag symmetry of equality.
  have row_12 : (zero = v2) := by
    exact Eq.symm row_11
  -- chapter_101_line_10: GL tag task formulation.
  have row_10 : (add v1 v2 zero) := by
    exact assumption_10
  -- chapter_101_line_9: GL tag equality1.
  have row_9 : (add v1 zero zero) := by
    have equality_source := row_10
    have equality_step_1 := row_11
    cases equality_step_1
    exact equality_source
  -- chapter_101_line_8: GL tag equality1.
  have row_8 : (add v1 zero v2) := by
    have equality_source := row_9
    have equality_step_1 := row_12
    cases equality_step_1
    exact equality_source
  -- chapter_101_line_7: GL tag task formulation.
  have row_7 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_101_line_6: GL tag expansion.
  have row_6 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_7
  -- chapter_101_line_5: GL tag disintegration.
  have row_5 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_6.1
  -- chapter_101_line_4: GL tag expansion.
  have row_4 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_5
  -- chapter_101_line_17: GL tag disintegration.
  have row_17 : (gl_fXYZ add N N N) := by
    exact row_4.1.1.1.1.1.1.1.1.2
  -- chapter_101_line_16: GL tag expansion.
  have row_16 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_17
  -- chapter_101_line_15: GL tag disintegration.
  have row_15 : (gl_implication8 add N) := by
    exact row_16.1.1.1.1
  -- chapter_101_line_14: GL tag expansion.
  have row_14 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_15
  -- chapter_101_line_13: GL tag implication.
  have row_13 : (N v1) := by
    apply row_14
    exact row_10
  -- chapter_101_line_3: GL tag disintegration.
  have row_3 : (gl_implication15 N zero add) := by
    exact row_4.1.1.1.1.1.1.1.2
  -- chapter_101_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((add w1 zero w2) → (w1 = w2))))) := by
    simpa only [gl_implication15] using row_3
  -- chapter_101_line_1: GL tag implication.
  have row_1 : (v1 = v2) := by
    apply row_2
    exact row_13
    exact row_8
  exact row_1

private theorem peano_source_046_check_induction_condition
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (assumption_26 : (add v1 v2 zero))
    (assumption_22 : (succ previous v2))
    (assumption_16 : (N v1))
    : (v1 = v2) := by
  -- chapter_102_line_26: GL tag task formulation.
  have row_26 : (add v1 v2 zero) := by
    exact assumption_26
  -- chapter_102_line_22: GL tag recursion.
  have row_22 : (succ previous v2) := by
    exact assumption_22
  -- chapter_102_line_16: GL tag task formulation.
  have row_16 : (N v1) := by
    exact assumption_16
  -- chapter_102_line_8: GL tag task formulation.
  have row_8 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_102_line_7: GL tag expansion.
  have row_7 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_8
  -- chapter_102_line_6: GL tag disintegration.
  have row_6 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_7.1
  -- chapter_102_line_5: GL tag expansion.
  have row_5 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_6
  -- chapter_102_line_25: GL tag disintegration.
  have row_25 : (gl_implication17 N succ add) := by
    exact row_5.1.1.1.1.1.2
  -- chapter_102_line_24: GL tag expansion.
  have row_24 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication17] using row_25
  -- chapter_102_line_21: GL tag disintegration.
  have row_21 : (gl_fXY succ N N) := by
    exact row_5.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_102_line_20: GL tag expansion.
  have row_20 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_21
  -- chapter_102_line_19: GL tag disintegration.
  have row_19 : (gl_implication0 succ N) := by
    exact row_20.1.1.1
  -- chapter_102_line_18: GL tag expansion.
  have row_18 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_19
  -- chapter_102_line_17: GL tag implication.
  have row_17 : (N previous) := by
    apply row_18
    exact row_22
  -- chapter_102_line_15: GL tag disintegration.
  have row_15 : (gl_fXYZ add N N N) := by
    exact row_5.1.1.1.1.1.1.1.1.2
  -- chapter_102_line_14: GL tag expansion.
  have row_14 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_15
  -- chapter_102_line_13: GL tag disintegration.
  have row_13 : (gl_implication13 N N N add) := by
    exact row_14.1.2
  -- chapter_102_line_12: GL tag expansion.
  have row_12 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication13] using row_13
  -- chapter_102_line_11: GL tag implication.
  have row_11 : (gl_existence1 N v1 previous add) := by
    apply row_12
    exact row_16
    exact row_17
  -- chapter_102_line_10: GL tag expansion.
  have row_10 : (¬ (∀ (v3 : α), ((N v3) → (¬ (add v1 previous v3))))) := by
    simpa only [gl_existence1] using row_11
  have exists_row_10 : ∃ (v3 : α), ((N v3) ∧ (add v1 previous v3)) := existsAndOfNotForallImpNot row_10
  obtain ⟨v3, witness_row_10⟩ := exists_row_10
  -- chapter_102_line_27: GL tag disintegration.
  have row_27 : (add v1 previous v3) := by
    exact witness_row_10.2
  -- chapter_102_line_23: GL tag implication.
  have row_23 : (succ v3 zero) := by
    apply row_24
    exact row_17
    exact row_22
    exact row_27
    exact row_26
  -- chapter_102_line_9: GL tag disintegration.
  have row_9 : (N v3) := by
    exact witness_row_10.1
  -- chapter_102_line_4: GL tag disintegration.
  have row_4 : (gl_implication6 N zero succ) := by
    exact row_5.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_102_line_3: GL tag expansion.
  have row_3 : (∀ (w1 : α), ((N w1) → (¬ (succ w1 zero)))) := by
    simpa only [gl_implication6] using row_4
  -- chapter_102_line_2: GL tag implication.
  have row_2 : (¬ (succ v3 zero)) := by
    apply row_3
    exact row_9
  -- chapter_102_line_1: GL tag vacuous truth.
  have row_1 : (v1 = v2) := by
    exact False.elim (row_2 row_23)
  exact row_1

theorem peano_source_046
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((add v1 v2 zero) → ((N v1) → (v1 = v2)))) := by
  intro v1
  intro v2
  intro premise_1
  intro premise_2
  have inductionMember : N v2 := by
    have typingRule := peano_source_046_induction_typing N zero succ add mul one anchor relationalInduction
    exact typingRule v1 v2 premise_1
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v1 : α), ((add v1 zero zero) → ((N v1) → (v1 = zero)))) := by
    intro v1
    intro base_premise_1
    intro base_premise_2
    have zeroRule := peano_source_046_check_zero N zero succ add mul one anchor relationalInduction
    exact zeroRule v1 zero rfl base_premise_1
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v1 : α), ((add v1 induction_n zero) → ((N v1) → (v1 = induction_n)))) → ∀ induction_m, succ induction_n induction_m → (∀ (v1 : α), ((add v1 induction_m zero) → ((N v1) → (v1 = induction_m)))) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro v1
    intro step_premise_1
    intro step_premise_2
    have step_induction_assumption_1 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have stepRule := peano_source_046_check_induction_condition N zero succ add mul one anchor relationalInduction
    exact stepRule induction_n v1 induction_m step_premise_1 step_induction_assumption_1 step_premise_2
  have inductionProperty : (∀ (v1 : α), ((add v1 v2 zero) → ((N v1) → (v1 = v2)))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v1 : α), ((add v1 induction_value zero) → ((N v1) → (v1 = induction_value)))))
      v2
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v1 premise_1 premise_2

theorem peano_source_053
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α), ((one = v1) → (gl_existence3 N v1 succ))) := by
  intro v1
  intro premise_1
  -- chapter_117_line_11: GL tag task formulation.
  have row_11 : (one = v1) := by
    exact premise_1
  -- chapter_117_line_7: GL tag task formulation.
  have row_7 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_117_line_6: GL tag expansion.
  have row_6 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_7
  -- chapter_117_line_10: GL tag disintegration.
  have row_10 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_6.1
  -- chapter_117_line_9: GL tag expansion.
  have row_9 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_10
  -- chapter_117_line_8: GL tag disintegration.
  have row_8 : (N zero) := by
    exact row_9.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_117_line_5: GL tag disintegration.
  have row_5 : (succ zero one) := by
    exact row_6.2
  -- chapter_117_line_4: GL tag expansion for integration.
  have row_4 : ((gl_existence3 N one succ) ↔ (¬ (∀ (v3 : α), ((N v3) → (¬ (succ v3 one)))))) := by
    exact Iff.rfl
  -- chapter_117_line_3: GL tag reformulation for integration >[bound].
  have row_3 : (∀ (v2 : α), ((N v2) → ((succ v2 one) → (gl_existence3 N one succ)))) := by
    intro v2
    intro integration_premise_1
    intro integration_premise_2
    apply (row_4).2
    intro universal_counterexample
    exact universal_counterexample v2 integration_premise_1 integration_premise_2
  -- chapter_117_line_2: GL tag implication.
  have row_2 : (gl_existence3 N one succ) := by
    apply row_3
    exact row_8
    exact row_5
  -- chapter_117_line_1: GL tag equality1.
  have row_1 : (gl_existence3 N v1 succ) := by
    have equality_source := row_2
    have equality_step_1 := row_11
    cases equality_step_1
    exact equality_source
  exact row_1

theorem peano_source_054
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α), ((succ v1 one) → (∀ (v2 : α), ((succ v2 one) → (v1 = v2))))) := by
  intro v1
  intro premise_1
  intro v2
  intro premise_2
  -- chapter_118_line_9: GL tag task formulation.
  have row_9 : (succ v2 one) := by
    exact premise_2
  -- chapter_118_line_8: GL tag task formulation.
  have row_8 : (succ v1 one) := by
    exact premise_1
  -- chapter_118_line_7: GL tag task formulation.
  have row_7 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_118_line_6: GL tag expansion.
  have row_6 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_7
  -- chapter_118_line_15: GL tag disintegration.
  have row_15 : (succ zero one) := by
    exact row_6.2
  -- chapter_118_line_5: GL tag disintegration.
  have row_5 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_6.1
  -- chapter_118_line_4: GL tag expansion.
  have row_4 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_5
  -- chapter_118_line_14: GL tag disintegration.
  have row_14 : (gl_fXY succ N N) := by
    exact row_4.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_118_line_13: GL tag expansion.
  have row_13 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_14
  -- chapter_118_line_12: GL tag disintegration.
  have row_12 : (gl_implication1 succ N) := by
    exact row_13.1.1.2
  -- chapter_118_line_11: GL tag expansion.
  have row_11 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using row_12
  -- chapter_118_line_10: GL tag implication.
  have row_10 : (N one) := by
    apply row_11
    exact row_15
  -- chapter_118_line_3: GL tag disintegration.
  have row_3 : (gl_implication7 N succ) := by
    exact row_4.1.1.1.1.1.1.1.1.1.2
  -- chapter_118_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w2 w1) → (∀ (w3 : α), ((succ w3 w1) → (w2 = w3))))))) := by
    simpa only [gl_implication7] using row_3
  -- chapter_118_line_1: GL tag implication.
  have row_1 : (v1 = v2) := by
    apply row_2
    exact row_10
    exact row_8
    exact row_9
  exact row_1

private theorem peano_source_059_induction_typing
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (assumption_1 : (N v1))
    : (N v1) := by
  -- chapter_131_line_1: GL tag task formulation.
  have row_1 : (N v1) := by
    exact assumption_1
  exact row_1

private theorem peano_source_059_check_zero
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (assumption_3 : (¬ (v1 = zero)))
    (assumption_2 : (v1 = zero))
    : (gl_existence3 N v1 succ) := by
  -- chapter_132_line_3: GL tag task formulation.
  have row_3 : (¬ (v1 = zero)) := by
    exact assumption_3
  -- chapter_132_line_2: GL tag recursion.
  have row_2 : (v1 = zero) := by
    exact assumption_2
  -- chapter_132_line_1: GL tag vacuous truth.
  have row_1 : (gl_existence3 N v1 succ) := by
    exact False.elim (row_3 row_2)
  exact row_1

private theorem peano_source_059_check_induction_condition
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (assumption_4 : (succ previous v1))
    : (gl_existence3 N v1 succ) := by
  -- chapter_133_line_13: GL tag task formulation.
  have row_13 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_133_line_12: GL tag expansion.
  have row_12 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_13
  -- chapter_133_line_11: GL tag disintegration.
  have row_11 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_12.1
  -- chapter_133_line_10: GL tag expansion.
  have row_10 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_11
  -- chapter_133_line_9: GL tag disintegration.
  have row_9 : (gl_fXY succ N N) := by
    exact row_10.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_133_line_8: GL tag expansion.
  have row_8 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_9
  -- chapter_133_line_7: GL tag disintegration.
  have row_7 : (gl_implication0 succ N) := by
    exact row_8.1.1.1
  -- chapter_133_line_6: GL tag expansion.
  have row_6 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_7
  -- chapter_133_line_4: GL tag recursion.
  have row_4 : (succ previous v1) := by
    exact assumption_4
  -- chapter_133_line_5: GL tag implication.
  have row_5 : (N previous) := by
    apply row_6
    exact row_4
  -- chapter_133_line_3: GL tag expansion for integration.
  have row_3 : ((gl_existence3 N v1 succ) ↔ (¬ (∀ (v3 : α), ((N v3) → (¬ (succ v3 v1)))))) := by
    exact Iff.rfl
  -- chapter_133_line_2: GL tag reformulation for integration >[bound].
  have row_2 : (∀ (v2 : α), ((N v2) → ((succ v2 v1) → (gl_existence3 N v1 succ)))) := by
    intro v2
    intro integration_premise_1
    intro integration_premise_2
    apply (row_3).2
    intro universal_counterexample
    exact universal_counterexample v2 integration_premise_1 integration_premise_2
  -- chapter_133_line_1: GL tag implication.
  have row_1 : (gl_existence3 N v1 succ) := by
    apply row_2
    exact row_5
    exact row_4
  exact row_1

theorem peano_source_059
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α), ((N v1) → ((¬ (v1 = zero)) → (gl_existence3 N v1 succ)))) := by
  intro v1
  intro premise_1
  intro premise_2
  have inductionMember : N v1 := by
    have typingRule := peano_source_059_induction_typing N zero succ add mul one anchor relationalInduction
    exact typingRule v1 premise_1
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : ((¬ (zero = zero)) → (gl_existence3 N zero succ)) := by
    intro base_premise_1
    have zeroRule := peano_source_059_check_zero N zero succ add mul one anchor relationalInduction
    exact zeroRule zero base_premise_1 rfl
  have inductionStep :
      ∀ induction_n, N induction_n → ((¬ (induction_n = zero)) → (gl_existence3 N induction_n succ)) → ∀ induction_m, succ induction_n induction_m → ((¬ (induction_m = zero)) → (gl_existence3 N induction_m succ)) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro step_premise_1
    have step_induction_assumption_1 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have stepRule := peano_source_059_check_induction_condition N zero succ add mul one anchor relationalInduction
    exact stepRule induction_n induction_m step_induction_assumption_1
  have inductionProperty : ((¬ (v1 = zero)) → (gl_existence3 N v1 succ)) := by
    exact relationalInduction
      (fun induction_value => ((¬ (induction_value = zero)) → (gl_existence3 N induction_value succ)))
      v1
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty premise_2

private theorem peano_source_060_induction_typing
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (assumption_1 : (N v1))
    : (N v1) := by
  -- chapter_134_line_1: GL tag task formulation.
  have row_1 : (N v1) := by
    exact assumption_1
  exact row_1

private theorem peano_source_060_check_zero
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (assumption_1 : (v1 = zero))
    : (v1 = zero) := by
  -- chapter_135_line_1: GL tag recursion.
  have row_1 : (v1 = zero) := by
    exact assumption_1
  exact row_1

private theorem peano_source_060_check_induction_condition
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (assumption_6 : (succ previous v1))
    (assumption_5 : (¬ (gl_existence3 N v1 succ)))
    : (v1 = zero) := by
  -- chapter_136_line_15: GL tag task formulation.
  have row_15 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_136_line_14: GL tag expansion.
  have row_14 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_15
  -- chapter_136_line_13: GL tag disintegration.
  have row_13 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_14.1
  -- chapter_136_line_12: GL tag expansion.
  have row_12 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_13
  -- chapter_136_line_11: GL tag disintegration.
  have row_11 : (gl_fXY succ N N) := by
    exact row_12.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_136_line_10: GL tag expansion.
  have row_10 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_11
  -- chapter_136_line_9: GL tag disintegration.
  have row_9 : (gl_implication0 succ N) := by
    exact row_10.1.1.1
  -- chapter_136_line_8: GL tag expansion.
  have row_8 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_9
  -- chapter_136_line_6: GL tag recursion.
  have row_6 : (succ previous v1) := by
    exact assumption_6
  -- chapter_136_line_7: GL tag implication.
  have row_7 : (N previous) := by
    apply row_8
    exact row_6
  -- chapter_136_line_5: GL tag task formulation.
  have row_5 : (¬ (gl_existence3 N v1 succ)) := by
    exact assumption_5
  -- chapter_136_line_4: GL tag expansion.
  have row_4 : (gl_implication1247 v1 succ N) := by
    simp only [gl_implication1247]
    intro existence_witness_1 compact_premise compact_negated
    have unfolded_row_5 := row_5
    simp only [gl_existence3] at unfolded_row_5
    apply unfolded_row_5
    intro positive_row_5
    exact positive_row_5 existence_witness_1 compact_negated compact_premise
  -- chapter_136_line_3: GL tag expansion.
  have row_3 : (∀ (w1 : α), ((succ w1 v1) → (¬ (N w1)))) := by
    simpa only [gl_implication1247] using row_4
  -- chapter_136_line_2: GL tag implication.
  have row_2 : (¬ (N previous)) := by
    apply row_3
    exact row_6
  -- chapter_136_line_1: GL tag vacuous truth.
  have row_1 : (v1 = zero) := by
    exact False.elim (row_2 row_7)
  exact row_1

theorem peano_source_060
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α), ((N v1) → ((¬ (gl_existence3 N v1 succ)) → (v1 = zero)))) := by
  intro v1
  intro premise_1
  intro premise_2
  have inductionMember : N v1 := by
    have typingRule := peano_source_060_induction_typing N zero succ add mul one anchor relationalInduction
    exact typingRule v1 premise_1
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : ((¬ (gl_existence3 N zero succ)) → (zero = zero)) := by
    intro base_premise_1
    have zeroRule := peano_source_060_check_zero N zero succ add mul one anchor relationalInduction
    exact zeroRule zero rfl
  have inductionStep :
      ∀ induction_n, N induction_n → ((¬ (gl_existence3 N induction_n succ)) → (induction_n = zero)) → ∀ induction_m, succ induction_n induction_m → ((¬ (gl_existence3 N induction_m succ)) → (induction_m = zero)) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro step_premise_1
    have step_induction_assumption_1 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have stepRule := peano_source_060_check_induction_condition N zero succ add mul one anchor relationalInduction
    exact stepRule induction_n induction_m step_induction_assumption_1 step_premise_1
  have inductionProperty : ((¬ (gl_existence3 N v1 succ)) → (v1 = zero)) := by
    exact relationalInduction
      (fun induction_value => ((¬ (gl_existence3 N induction_value succ)) → (induction_value = zero)))
      v1
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty premise_2

theorem peano_source_063
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (gl_existence3 N one succ) := by
  -- chapter_139_line_6: GL tag task formulation.
  have row_6 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_139_line_5: GL tag expansion.
  have row_5 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_6
  -- chapter_139_line_9: GL tag disintegration.
  have row_9 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_5.1
  -- chapter_139_line_8: GL tag expansion.
  have row_8 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_9
  -- chapter_139_line_7: GL tag disintegration.
  have row_7 : (N zero) := by
    exact row_8.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_139_line_4: GL tag disintegration.
  have row_4 : (succ zero one) := by
    exact row_5.2
  -- chapter_139_line_3: GL tag expansion for integration.
  have row_3 : ((gl_existence3 N one succ) ↔ (¬ (∀ (v2 : α), ((N v2) → (¬ (succ v2 one)))))) := by
    exact Iff.rfl
  -- chapter_139_line_2: GL tag reformulation for integration >[bound].
  have row_2 : (∀ (v1 : α), ((N v1) → ((succ v1 one) → (gl_existence3 N one succ)))) := by
    intro v1
    intro integration_premise_1
    intro integration_premise_2
    apply (row_3).2
    intro universal_counterexample
    exact universal_counterexample v1 integration_premise_1 integration_premise_2
  -- chapter_139_line_1: GL tag implication.
  have row_1 : (gl_existence3 N one succ) := by
    apply row_2
    exact row_7
    exact row_4
  exact row_1

private theorem peano_source_004_induction_typing
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v2 : α)
    (v5 : α)
    (v6 : α)
    (assumption_10 : (add v5 v2 v6))
    : (N v2) := by
  -- chapter_10_line_10: GL tag task formulation.
  have row_10 : (add v5 v2 v6) := by
    exact assumption_10
  -- chapter_10_line_9: GL tag task formulation.
  have row_9 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_10_line_8: GL tag expansion.
  have row_8 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_9
  -- chapter_10_line_7: GL tag disintegration.
  have row_7 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_8.1
  -- chapter_10_line_6: GL tag expansion.
  have row_6 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_7
  -- chapter_10_line_5: GL tag disintegration.
  have row_5 : (gl_fXYZ add N N N) := by
    exact row_6.1.1.1.1.1.1.1.1.2
  -- chapter_10_line_4: GL tag expansion.
  have row_4 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_5
  -- chapter_10_line_3: GL tag disintegration.
  have row_3 : (gl_implication9 add N) := by
    exact row_4.1.1.1.2
  -- chapter_10_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication9] using row_3
  -- chapter_10_line_1: GL tag implication.
  have row_1 : (N v2) := by
    apply row_2
    exact row_10
  exact row_1

private theorem peano_source_004_check_zero
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (v4 : α)
    (v5 : α)
    (v6 : α)
    (assumption_20 : (add v1 v2 v3))
    (assumption_9 : (v2 = zero))
    (assumption_8 : (add v5 v2 v6))
    (assumption_3 : (add v4 v5 v1))
    : (add v4 v6 v3) := by
  -- chapter_11_line_20: GL tag task formulation.
  have row_20 : (add v1 v2 v3) := by
    exact assumption_20
  -- chapter_11_line_9: GL tag recursion.
  have row_9 : (v2 = zero) := by
    exact assumption_9
  -- chapter_11_line_19: GL tag equality1.
  have row_19 : (add v1 zero v3) := by
    have equality_source := row_20
    have equality_step_1 := row_9
    cases equality_step_1
    exact equality_source
  -- chapter_11_line_8: GL tag task formulation.
  have row_8 : (add v5 v2 v6) := by
    exact assumption_8
  -- chapter_11_line_7: GL tag equality1.
  have row_7 : (add v5 zero v6) := by
    have equality_source := row_8
    have equality_step_1 := row_9
    cases equality_step_1
    exact equality_source
  -- chapter_11_line_6: GL tag task formulation.
  have row_6 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_11_line_17: GL tag expansion.
  have row_17 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_6
  -- chapter_11_line_16: GL tag disintegration.
  have row_16 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_17.1
  -- chapter_11_line_15: GL tag expansion.
  have row_15 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_16
  -- chapter_11_line_14: GL tag disintegration.
  have row_14 : (gl_fXYZ add N N N) := by
    exact row_15.1.1.1.1.1.1.1.1.2
  -- chapter_11_line_13: GL tag expansion.
  have row_13 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_14
  -- chapter_11_line_12: GL tag disintegration.
  have row_12 : (gl_implication8 add N) := by
    exact row_13.1.1.1.1
  -- chapter_11_line_11: GL tag expansion.
  have row_11 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_12
  -- chapter_11_line_21: GL tag implication.
  have row_21 : (N v1) := by
    apply row_11
    exact row_20
  have rule_row_18 := peano_source_039 N zero succ add mul one anchor relationalInduction
  -- chapter_11_line_18: GL tag implication.
  have row_18 : (v1 = v3) := by
    apply rule_row_18
    exact row_19
    exact row_21
  -- chapter_11_line_10: GL tag implication.
  have row_10 : (N v5) := by
    apply row_11
    exact row_8
  -- chapter_11_line_5: GL tag theorem.
  have row_5 := peano_source_039 N zero succ add mul one anchor relationalInduction
  -- chapter_11_line_4: GL tag implication.
  have row_4 : (v5 = v6) := by
    apply row_5
    exact row_7
    exact row_10
  -- chapter_11_line_3: GL tag task formulation.
  have row_3 : (add v4 v5 v1) := by
    exact assumption_3
  -- chapter_11_line_2: GL tag equality1.
  have row_2 : (add v4 v6 v1) := by
    have equality_source := row_3
    have equality_step_1 := row_4
    cases equality_step_1
    exact equality_source
  -- chapter_11_line_1: GL tag equality1.
  have row_1 : (add v4 v6 v3) := by
    have equality_source := row_2
    have equality_step_1 := row_18
    cases equality_step_1
    exact equality_source
  exact row_1

private theorem peano_source_004_check_induction_condition
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (v4 : α)
    (v5 : α)
    (v6 : α)
    (assumption_35 : (add v4 v5 v1))
    (assumption_34 : (∀ (w1 : α), ((add v1 previous w1) → ((add v4 v5 v1) → (∀ (w2 : α), ((add v5 previous w2) → (add v4 w2 w1)))))))
    (assumption_28 : (add v1 v2 v3))
    (assumption_11 : (add v5 v2 v6))
    (assumption_10 : (succ previous v2))
    : (add v4 v6 v3) := by
  -- chapter_12_line_35: GL tag task formulation.
  have row_35 : (add v4 v5 v1) := by
    exact assumption_35
  -- chapter_12_line_34: GL tag recursion.
  have row_34 : (∀ (w1 : α), ((add v1 previous w1) → ((add v4 v5 v1) → (∀ (w2 : α), ((add v5 previous w2) → (add v4 w2 w1)))))) := by
    exact assumption_34
  -- chapter_12_line_28: GL tag task formulation.
  have row_28 : (add v1 v2 v3) := by
    exact assumption_28
  -- chapter_12_line_11: GL tag task formulation.
  have row_11 : (add v5 v2 v6) := by
    exact assumption_11
  -- chapter_12_line_10: GL tag recursion.
  have row_10 : (succ previous v2) := by
    exact assumption_10
  -- chapter_12_line_3: GL tag task formulation.
  have row_3 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_12_line_9: GL tag expansion.
  have row_9 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_3
  -- chapter_12_line_8: GL tag disintegration.
  have row_8 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_9.1
  -- chapter_12_line_7: GL tag expansion.
  have row_7 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_8
  -- chapter_12_line_26: GL tag disintegration.
  have row_26 : (gl_fXY succ N N) := by
    exact row_7.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_12_line_25: GL tag expansion.
  have row_25 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_26
  -- chapter_12_line_24: GL tag disintegration.
  have row_24 : (gl_implication0 succ N) := by
    exact row_25.1.1.1
  -- chapter_12_line_23: GL tag expansion.
  have row_23 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_24
  -- chapter_12_line_22: GL tag implication.
  have row_22 : (N previous) := by
    apply row_23
    exact row_10
  -- chapter_12_line_18: GL tag disintegration.
  have row_18 : (gl_fXYZ add N N N) := by
    exact row_7.1.1.1.1.1.1.1.1.2
  -- chapter_12_line_17: GL tag expansion.
  have row_17 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_18
  -- chapter_12_line_21: GL tag disintegration.
  have row_21 : (gl_implication8 add N) := by
    exact row_17.1.1.1.1
  -- chapter_12_line_20: GL tag expansion.
  have row_20 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_21
  -- chapter_12_line_32: GL tag implication.
  have row_32 : (N v1) := by
    apply row_20
    exact row_28
  -- chapter_12_line_19: GL tag implication.
  have row_19 : (N v5) := by
    apply row_20
    exact row_11
  -- chapter_12_line_16: GL tag disintegration.
  have row_16 : (gl_implication13 N N N add) := by
    exact row_17.1.2
  -- chapter_12_line_15: GL tag expansion.
  have row_15 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication13] using row_16
  -- chapter_12_line_31: GL tag implication.
  have row_31 : (gl_existence1 N v1 previous add) := by
    apply row_15
    exact row_32
    exact row_22
  -- chapter_12_line_30: GL tag expansion.
  have row_30 : (¬ (∀ (v8 : α), ((N v8) → (¬ (add v1 previous v8))))) := by
    simpa only [gl_existence1] using row_31
  have exists_row_30 : ∃ (v8 : α), ((N v8) ∧ (add v1 previous v8)) := existsAndOfNotForallImpNot row_30
  obtain ⟨v8, witness_row_30⟩ := exists_row_30
  -- chapter_12_line_29: GL tag disintegration.
  have row_29 : (add v1 previous v8) := by
    exact witness_row_30.2
  -- chapter_12_line_14: GL tag implication.
  have row_14 : (gl_existence1 N v5 previous add) := by
    apply row_15
    exact row_19
    exact row_22
  -- chapter_12_line_13: GL tag expansion.
  have row_13 : (¬ (∀ (v7 : α), ((N v7) → (¬ (add v5 previous v7))))) := by
    simpa only [gl_existence1] using row_14
  have exists_row_13 : ∃ (v7 : α), ((N v7) ∧ (add v5 previous v7)) := existsAndOfNotForallImpNot row_13
  obtain ⟨v7, witness_row_13⟩ := exists_row_13
  -- chapter_12_line_12: GL tag disintegration.
  have row_12 : (add v5 previous v7) := by
    exact witness_row_13.2
  -- chapter_12_line_33: GL tag implication.
  have row_33 : (add v4 v7 v8) := by
    apply row_34
    exact row_29
    exact row_35
    exact row_12
  -- chapter_12_line_6: GL tag disintegration.
  have row_6 : (gl_implication17 N succ add) := by
    exact row_7.1.1.1.1.1.2
  -- chapter_12_line_5: GL tag expansion.
  have row_5 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication17] using row_6
  -- chapter_12_line_27: GL tag implication.
  have row_27 : (succ v8 v3) := by
    apply row_5
    exact row_22
    exact row_10
    exact row_29
    exact row_28
  -- chapter_12_line_4: GL tag implication.
  have row_4 : (succ v7 v6) := by
    apply row_5
    exact row_22
    exact row_10
    exact row_12
    exact row_11
  -- chapter_12_line_2: GL tag theorem.
  have row_2 := peano_source_011 N zero succ add mul one anchor relationalInduction
  -- chapter_12_line_1: GL tag implication.
  have row_1 : (add v4 v6 v3) := by
    apply row_2
    exact row_33
    exact row_4
    exact row_27
  exact row_1

theorem peano_source_004
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α) (v5 : α), ((add v4 v5 v1) → (∀ (v6 : α), ((add v5 v2 v6) → (add v4 v6 v3))))))) := by
  intro v1
  intro v2
  intro v3
  intro premise_1
  intro v4
  intro v5
  intro premise_2
  intro v6
  intro premise_3
  have inductionMember : N v2 := by
    have typingRule := peano_source_004_induction_typing N zero succ add mul one anchor relationalInduction
    exact typingRule v2 v5 v6 premise_3
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v1 : α) (v3 : α), ((add v1 zero v3) → (∀ (v4 : α) (v5 : α), ((add v4 v5 v1) → (∀ (v6 : α), ((add v5 zero v6) → (add v4 v6 v3))))))) := by
    intro v1
    intro v3
    intro base_premise_1
    intro v4
    intro v5
    intro base_premise_2
    intro v6
    intro base_premise_3
    have zeroRule := peano_source_004_check_zero N zero succ add mul one anchor relationalInduction
    exact zeroRule v1 zero v3 v4 v5 v6 base_premise_1 rfl base_premise_3 base_premise_2
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v1 : α) (v3 : α), ((add v1 induction_n v3) → (∀ (v4 : α) (v5 : α), ((add v4 v5 v1) → (∀ (v6 : α), ((add v5 induction_n v6) → (add v4 v6 v3))))))) → ∀ induction_m, succ induction_n induction_m → (∀ (v1 : α) (v3 : α), ((add v1 induction_m v3) → (∀ (v4 : α) (v5 : α), ((add v4 v5 v1) → (∀ (v6 : α), ((add v5 induction_m v6) → (add v4 v6 v3))))))) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro v1
    intro v3
    intro step_premise_1
    intro v4
    intro v5
    intro step_premise_2
    intro v6
    intro step_premise_3
    have step_induction_assumption_1 :
        (∀ (w1 : α), ((add v1 induction_n w1) → ((add v4 v5 v1) → (∀ (w2 : α), ((add v5 induction_n w2) → (add v4 w2 w1)))))) := by
      intro w1
      intro step_induction_assumption_1_premise_1
      intro step_induction_assumption_1_premise_2
      intro w2
      intro step_induction_assumption_1_premise_3
      apply induction_hypothesis
      all_goals assumption
    have step_induction_assumption_2 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have stepRule := peano_source_004_check_induction_condition N zero succ add mul one anchor relationalInduction
    exact stepRule induction_n v1 induction_m v3 v4 v5 v6 step_premise_2 step_induction_assumption_1 step_premise_1 step_premise_3 step_induction_assumption_2
  have inductionProperty : (∀ (v1 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α) (v5 : α), ((add v4 v5 v1) → (∀ (v6 : α), ((add v5 v2 v6) → (add v4 v6 v3))))))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v1 : α) (v3 : α), ((add v1 induction_value v3) → (∀ (v4 : α) (v5 : α), ((add v4 v5 v1) → (∀ (v6 : α), ((add v5 induction_value v6) → (add v4 v6 v3))))))))
      v2
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v1 v3 premise_1 v4 v5 premise_2 v6 premise_3

private theorem peano_source_005_induction_typing
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v2 : α)
    (v4 : α)
    (v5 : α)
    (assumption_10 : (add v4 v2 v5))
    : (N v2) := by
  -- chapter_13_line_10: GL tag task formulation.
  have row_10 : (add v4 v2 v5) := by
    exact assumption_10
  -- chapter_13_line_9: GL tag task formulation.
  have row_9 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_13_line_8: GL tag expansion.
  have row_8 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_9
  -- chapter_13_line_7: GL tag disintegration.
  have row_7 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_8.1
  -- chapter_13_line_6: GL tag expansion.
  have row_6 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_7
  -- chapter_13_line_5: GL tag disintegration.
  have row_5 : (gl_fXYZ add N N N) := by
    exact row_6.1.1.1.1.1.1.1.1.2
  -- chapter_13_line_4: GL tag expansion.
  have row_4 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_5
  -- chapter_13_line_3: GL tag disintegration.
  have row_3 : (gl_implication9 add N) := by
    exact row_4.1.1.1.2
  -- chapter_13_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication9] using row_3
  -- chapter_13_line_1: GL tag implication.
  have row_1 : (N v2) := by
    apply row_2
    exact row_10
  exact row_1

private theorem peano_source_005_check_zero
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (v4 : α)
    (v5 : α)
    (assumption_20 : (add v1 v2 v3))
    (assumption_9 : (v2 = zero))
    (assumption_8 : (add v4 v2 v5))
    (assumption_3 : (succ v4 v1))
    : (succ v5 v3) := by
  -- chapter_14_line_20: GL tag task formulation.
  have row_20 : (add v1 v2 v3) := by
    exact assumption_20
  -- chapter_14_line_9: GL tag recursion.
  have row_9 : (v2 = zero) := by
    exact assumption_9
  -- chapter_14_line_19: GL tag equality1.
  have row_19 : (add v1 zero v3) := by
    have equality_source := row_20
    have equality_step_1 := row_9
    cases equality_step_1
    exact equality_source
  -- chapter_14_line_8: GL tag task formulation.
  have row_8 : (add v4 v2 v5) := by
    exact assumption_8
  -- chapter_14_line_7: GL tag equality1.
  have row_7 : (add v4 zero v5) := by
    have equality_source := row_8
    have equality_step_1 := row_9
    cases equality_step_1
    exact equality_source
  -- chapter_14_line_6: GL tag task formulation.
  have row_6 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_14_line_17: GL tag expansion.
  have row_17 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_6
  -- chapter_14_line_16: GL tag disintegration.
  have row_16 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_17.1
  -- chapter_14_line_15: GL tag expansion.
  have row_15 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_16
  -- chapter_14_line_14: GL tag disintegration.
  have row_14 : (gl_fXYZ add N N N) := by
    exact row_15.1.1.1.1.1.1.1.1.2
  -- chapter_14_line_13: GL tag expansion.
  have row_13 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_14
  -- chapter_14_line_12: GL tag disintegration.
  have row_12 : (gl_implication8 add N) := by
    exact row_13.1.1.1.1
  -- chapter_14_line_11: GL tag expansion.
  have row_11 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_12
  -- chapter_14_line_21: GL tag implication.
  have row_21 : (N v1) := by
    apply row_11
    exact row_20
  have rule_row_18 := peano_source_039 N zero succ add mul one anchor relationalInduction
  -- chapter_14_line_18: GL tag implication.
  have row_18 : (v1 = v3) := by
    apply rule_row_18
    exact row_19
    exact row_21
  -- chapter_14_line_10: GL tag implication.
  have row_10 : (N v4) := by
    apply row_11
    exact row_8
  -- chapter_14_line_5: GL tag theorem.
  have row_5 := peano_source_039 N zero succ add mul one anchor relationalInduction
  -- chapter_14_line_4: GL tag implication.
  have row_4 : (v4 = v5) := by
    apply row_5
    exact row_7
    exact row_10
  -- chapter_14_line_3: GL tag task formulation.
  have row_3 : (succ v4 v1) := by
    exact assumption_3
  -- chapter_14_line_2: GL tag equality1.
  have row_2 : (succ v5 v1) := by
    have equality_source := row_3
    have equality_step_1 := row_4
    cases equality_step_1
    exact equality_source
  -- chapter_14_line_1: GL tag equality1.
  have row_1 : (succ v5 v3) := by
    have equality_source := row_2
    have equality_step_1 := row_18
    cases equality_step_1
    exact equality_source
  exact row_1

private theorem peano_source_005_check_induction_condition
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (v4 : α)
    (v5 : α)
    (assumption_37 : (succ v4 v1))
    (assumption_36 : (∀ (w1 : α), ((add v1 previous w1) → (∀ (w2 : α), ((add v4 previous w2) → ((succ v4 v1) → (succ w2 w1)))))))
    (assumption_30 : (add v4 v2 v5))
    (assumption_10 : (add v1 v2 v3))
    (assumption_9 : (succ previous v2))
    : (succ v5 v3) := by
  -- chapter_15_line_37: GL tag task formulation.
  have row_37 : (succ v4 v1) := by
    exact assumption_37
  -- chapter_15_line_36: GL tag recursion.
  have row_36 : (∀ (w1 : α), ((add v1 previous w1) → (∀ (w2 : α), ((add v4 previous w2) → ((succ v4 v1) → (succ w2 w1)))))) := by
    exact assumption_36
  -- chapter_15_line_30: GL tag task formulation.
  have row_30 : (add v4 v2 v5) := by
    exact assumption_30
  -- chapter_15_line_10: GL tag task formulation.
  have row_10 : (add v1 v2 v3) := by
    exact assumption_10
  -- chapter_15_line_9: GL tag recursion.
  have row_9 : (succ previous v2) := by
    exact assumption_9
  -- chapter_15_line_8: GL tag task formulation.
  have row_8 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_15_line_7: GL tag expansion.
  have row_7 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_8
  -- chapter_15_line_6: GL tag disintegration.
  have row_6 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_7.1
  -- chapter_15_line_5: GL tag expansion.
  have row_5 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_6
  -- chapter_15_line_25: GL tag disintegration.
  have row_25 : (gl_fXY succ N N) := by
    exact row_5.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_15_line_24: GL tag expansion.
  have row_24 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_25
  -- chapter_15_line_28: GL tag disintegration.
  have row_28 : (gl_implication5 N succ) := by
    exact row_24.2
  -- chapter_15_line_27: GL tag expansion.
  have row_27 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α), ((succ w1 w3) → (w2 = w3))))))) := by
    simpa only [gl_implication5] using row_28
  -- chapter_15_line_23: GL tag disintegration.
  have row_23 : (gl_implication0 succ N) := by
    exact row_24.1.1.1
  -- chapter_15_line_22: GL tag expansion.
  have row_22 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_23
  -- chapter_15_line_21: GL tag implication.
  have row_21 : (N previous) := by
    apply row_22
    exact row_9
  -- chapter_15_line_17: GL tag disintegration.
  have row_17 : (gl_fXYZ add N N N) := by
    exact row_5.1.1.1.1.1.1.1.1.2
  -- chapter_15_line_16: GL tag expansion.
  have row_16 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_17
  -- chapter_15_line_20: GL tag disintegration.
  have row_20 : (gl_implication8 add N) := by
    exact row_16.1.1.1.1
  -- chapter_15_line_19: GL tag expansion.
  have row_19 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_20
  -- chapter_15_line_34: GL tag implication.
  have row_34 : (N v4) := by
    apply row_19
    exact row_30
  -- chapter_15_line_18: GL tag implication.
  have row_18 : (N v1) := by
    apply row_19
    exact row_10
  -- chapter_15_line_15: GL tag disintegration.
  have row_15 : (gl_implication13 N N N add) := by
    exact row_16.1.2
  -- chapter_15_line_14: GL tag expansion.
  have row_14 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication13] using row_15
  -- chapter_15_line_33: GL tag implication.
  have row_33 : (gl_existence1 N v4 previous add) := by
    apply row_14
    exact row_34
    exact row_21
  -- chapter_15_line_32: GL tag expansion.
  have row_32 : (¬ (∀ (v12 : α), ((N v12) → (¬ (add v4 previous v12))))) := by
    simpa only [gl_existence1] using row_33
  have exists_row_32 : ∃ (v12 : α), ((N v12) ∧ (add v4 previous v12)) := existsAndOfNotForallImpNot row_32
  obtain ⟨v12, witness_row_32⟩ := exists_row_32
  -- chapter_15_line_38: GL tag disintegration.
  have row_38 : (N v12) := by
    exact witness_row_32.1
  -- chapter_15_line_31: GL tag disintegration.
  have row_31 : (add v4 previous v12) := by
    exact witness_row_32.2
  -- chapter_15_line_13: GL tag implication.
  have row_13 : (gl_existence1 N v1 previous add) := by
    apply row_14
    exact row_18
    exact row_21
  -- chapter_15_line_12: GL tag expansion.
  have row_12 : (¬ (∀ (v6 : α), ((N v6) → (¬ (add v1 previous v6))))) := by
    simpa only [gl_existence1] using row_13
  have exists_row_12 : ∃ (v6 : α), ((N v6) ∧ (add v1 previous v6)) := existsAndOfNotForallImpNot row_12
  obtain ⟨v6, witness_row_12⟩ := exists_row_12
  -- chapter_15_line_11: GL tag disintegration.
  have row_11 : (add v1 previous v6) := by
    exact witness_row_12.2
  -- chapter_15_line_35: GL tag implication.
  have row_35 : (succ v12 v6) := by
    apply row_36
    exact row_11
    exact row_31
    exact row_37
  -- chapter_15_line_4: GL tag disintegration.
  have row_4 : (gl_implication17 N succ add) := by
    exact row_5.1.1.1.1.1.2
  -- chapter_15_line_3: GL tag expansion.
  have row_3 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication17] using row_4
  -- chapter_15_line_29: GL tag implication.
  have row_29 : (succ v12 v5) := by
    apply row_3
    exact row_21
    exact row_9
    exact row_31
    exact row_30
  -- chapter_15_line_26: GL tag implication.
  have row_26 : (v6 = v5) := by
    apply row_27
    exact row_38
    exact row_35
    exact row_29
  -- chapter_15_line_2: GL tag implication.
  have row_2 : (succ v6 v3) := by
    apply row_3
    exact row_21
    exact row_9
    exact row_11
    exact row_10
  -- chapter_15_line_1: GL tag equality1.
  have row_1 : (succ v5 v3) := by
    have equality_source := row_2
    have equality_step_1 := row_26
    cases equality_step_1
    exact equality_source
  exact row_1

theorem peano_source_005
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α) (v5 : α), ((add v4 v2 v5) → ((succ v4 v1) → (succ v5 v3)))))) := by
  intro v1
  intro v2
  intro v3
  intro premise_1
  intro v4
  intro v5
  intro premise_2
  intro premise_3
  have inductionMember : N v2 := by
    have typingRule := peano_source_005_induction_typing N zero succ add mul one anchor relationalInduction
    exact typingRule v2 v4 v5 premise_2
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v1 : α) (v3 : α), ((add v1 zero v3) → (∀ (v4 : α) (v5 : α), ((add v4 zero v5) → ((succ v4 v1) → (succ v5 v3)))))) := by
    intro v1
    intro v3
    intro base_premise_1
    intro v4
    intro v5
    intro base_premise_2
    intro base_premise_3
    have zeroRule := peano_source_005_check_zero N zero succ add mul one anchor relationalInduction
    exact zeroRule v1 zero v3 v4 v5 base_premise_1 rfl base_premise_2 base_premise_3
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v1 : α) (v3 : α), ((add v1 induction_n v3) → (∀ (v4 : α) (v5 : α), ((add v4 induction_n v5) → ((succ v4 v1) → (succ v5 v3)))))) → ∀ induction_m, succ induction_n induction_m → (∀ (v1 : α) (v3 : α), ((add v1 induction_m v3) → (∀ (v4 : α) (v5 : α), ((add v4 induction_m v5) → ((succ v4 v1) → (succ v5 v3)))))) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro v1
    intro v3
    intro step_premise_1
    intro v4
    intro v5
    intro step_premise_2
    intro step_premise_3
    have step_induction_assumption_1 :
        (∀ (w1 : α), ((add v1 induction_n w1) → (∀ (w2 : α), ((add v4 induction_n w2) → ((succ v4 v1) → (succ w2 w1)))))) := by
      intro w1
      intro step_induction_assumption_1_premise_1
      intro w2
      intro step_induction_assumption_1_premise_2
      intro step_induction_assumption_1_premise_3
      apply induction_hypothesis
      all_goals assumption
    have step_induction_assumption_2 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have stepRule := peano_source_005_check_induction_condition N zero succ add mul one anchor relationalInduction
    exact stepRule induction_n v1 induction_m v3 v4 v5 step_premise_3 step_induction_assumption_1 step_premise_2 step_premise_1 step_induction_assumption_2
  have inductionProperty : (∀ (v1 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α) (v5 : α), ((add v4 v2 v5) → ((succ v4 v1) → (succ v5 v3)))))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v1 : α) (v3 : α), ((add v1 induction_value v3) → (∀ (v4 : α) (v5 : α), ((add v4 induction_value v5) → ((succ v4 v1) → (succ v5 v3)))))))
      v2
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v1 v3 premise_1 v4 v5 premise_2 premise_3

private theorem peano_source_008_induction_typing
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v2 : α)
    (v5 : α)
    (assumption_10 : (succ v2 v5))
    : (N v2) := by
  -- chapter_20_line_10: GL tag task formulation.
  have row_10 : (succ v2 v5) := by
    exact assumption_10
  -- chapter_20_line_9: GL tag task formulation.
  have row_9 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_20_line_8: GL tag expansion.
  have row_8 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_9
  -- chapter_20_line_7: GL tag disintegration.
  have row_7 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_8.1
  -- chapter_20_line_6: GL tag expansion.
  have row_6 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_7
  -- chapter_20_line_5: GL tag disintegration.
  have row_5 : (gl_fXY succ N N) := by
    exact row_6.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_20_line_4: GL tag expansion.
  have row_4 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_5
  -- chapter_20_line_3: GL tag disintegration.
  have row_3 : (gl_implication0 succ N) := by
    exact row_4.1.1.1
  -- chapter_20_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_3
  -- chapter_20_line_1: GL tag implication.
  have row_1 : (N v2) := by
    apply row_2
    exact row_10
  exact row_1

private theorem peano_source_008_check_zero
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (v4 : α)
    (v5 : α)
    (assumption_31 : (v2 = zero))
    (assumption_30 : (add v1 v2 v3))
    (assumption_11 : (succ v4 v1))
    (assumption_8 : (succ v2 v5))
    : (add v4 v5 v3) := by
  -- chapter_21_line_31: GL tag recursion.
  have row_31 : (v2 = zero) := by
    exact assumption_31
  -- chapter_21_line_30: GL tag task formulation.
  have row_30 : (add v1 v2 v3) := by
    exact assumption_30
  -- chapter_21_line_33: GL tag equality1.
  have row_33 : (add v1 zero v3) := by
    have equality_source := row_30
    have equality_step_1 := row_31
    cases equality_step_1
    exact equality_source
  -- chapter_21_line_13: GL tag theorem.
  have row_13 := peano_source_039 N zero succ add mul one anchor relationalInduction
  -- chapter_21_line_11: GL tag task formulation.
  have row_11 : (succ v4 v1) := by
    exact assumption_11
  -- chapter_21_line_8: GL tag task formulation.
  have row_8 : (succ v2 v5) := by
    exact assumption_8
  -- chapter_21_line_7: GL tag task formulation.
  have row_7 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_21_line_6: GL tag expansion.
  have row_6 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_7
  -- chapter_21_line_5: GL tag disintegration.
  have row_5 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_6.1
  -- chapter_21_line_4: GL tag expansion.
  have row_4 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_5
  -- chapter_21_line_26: GL tag disintegration.
  have row_26 : (gl_fXY succ N N) := by
    exact row_4.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_21_line_25: GL tag expansion.
  have row_25 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_26
  -- chapter_21_line_24: GL tag disintegration.
  have row_24 : (gl_implication0 succ N) := by
    exact row_25.1.1.1
  -- chapter_21_line_23: GL tag expansion.
  have row_23 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_24
  -- chapter_21_line_22: GL tag implication.
  have row_22 : (N v4) := by
    apply row_23
    exact row_11
  -- chapter_21_line_21: GL tag disintegration.
  have row_21 : (gl_fXYZ add N N N) := by
    exact row_4.1.1.1.1.1.1.1.1.2
  -- chapter_21_line_20: GL tag expansion.
  have row_20 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_21
  -- chapter_21_line_36: GL tag disintegration.
  have row_36 : (gl_implication8 add N) := by
    exact row_20.1.1.1.1
  -- chapter_21_line_35: GL tag expansion.
  have row_35 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_36
  -- chapter_21_line_34: GL tag implication.
  have row_34 : (N v1) := by
    apply row_35
    exact row_30
  -- chapter_21_line_32: GL tag implication.
  have row_32 : (v1 = v3) := by
    apply row_13
    exact row_33
    exact row_34
  -- chapter_21_line_29: GL tag disintegration.
  have row_29 : (gl_implication9 add N) := by
    exact row_20.1.1.1.2
  -- chapter_21_line_28: GL tag expansion.
  have row_28 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication9] using row_29
  -- chapter_21_line_27: GL tag implication.
  have row_27 : (N v2) := by
    apply row_28
    exact row_30
  -- chapter_21_line_19: GL tag disintegration.
  have row_19 : (gl_implication13 N N N add) := by
    exact row_20.1.2
  -- chapter_21_line_18: GL tag expansion.
  have row_18 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication13] using row_19
  -- chapter_21_line_17: GL tag implication.
  have row_17 : (gl_existence1 N v4 v2 add) := by
    apply row_18
    exact row_22
    exact row_27
  -- chapter_21_line_16: GL tag expansion.
  have row_16 : (¬ (∀ (v11 : α), ((N v11) → (¬ (add v4 v2 v11))))) := by
    simpa only [gl_existence1] using row_17
  have exists_row_16 : ∃ (v11 : α), ((N v11) ∧ (add v4 v2 v11)) := existsAndOfNotForallImpNot row_16
  obtain ⟨v11, witness_row_16⟩ := exists_row_16
  -- chapter_21_line_15: GL tag disintegration.
  have row_15 : (add v4 v2 v11) := by
    exact witness_row_16.2
  -- chapter_21_line_14: GL tag equality1.
  have row_14 : (add v4 zero v11) := by
    have equality_source := row_15
    have equality_step_1 := row_31
    cases equality_step_1
    exact equality_source
  -- chapter_21_line_12: GL tag implication.
  have row_12 : (v4 = v11) := by
    apply row_13
    exact row_14
    exact row_22
  -- chapter_21_line_10: GL tag equality1.
  have row_10 : (succ v11 v1) := by
    have equality_source := row_11
    have equality_step_1 := row_12
    cases equality_step_1
    exact equality_source
  -- chapter_21_line_9: GL tag equality1.
  have row_9 : (succ v11 v3) := by
    have equality_source := row_10
    have equality_step_1 := row_32
    cases equality_step_1
    exact equality_source
  -- chapter_21_line_3: GL tag disintegration.
  have row_3 : (gl_implication18 N succ add) := by
    exact row_4.1.1.1.1.2
  -- chapter_21_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((succ w4 w5) → (add w3 w2 w5))))))))) := by
    simpa only [gl_implication18] using row_3
  -- chapter_21_line_1: GL tag implication.
  have row_1 : (add v4 v5 v3) := by
    apply row_2
    exact row_27
    exact row_8
    exact row_15
    exact row_9
  exact row_1

private theorem peano_source_008_check_induction_condition
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (v4 : α)
    (v5 : α)
    (assumption_39 : (succ v4 v1))
    (assumption_38 : (∀ (w1 : α), ((add v1 previous w1) → ((succ v4 v1) → (∀ (w2 : α), ((succ previous w2) → (add v4 w2 w1)))))))
    (assumption_26 : (add v1 v2 v3))
    (assumption_14 : (succ previous v2))
    (assumption_5 : (succ v2 v5))
    : (add v4 v5 v3) := by
  -- chapter_22_line_39: GL tag task formulation.
  have row_39 : (succ v4 v1) := by
    exact assumption_39
  -- chapter_22_line_38: GL tag recursion.
  have row_38 : (∀ (w1 : α), ((add v1 previous w1) → ((succ v4 v1) → (∀ (w2 : α), ((succ previous w2) → (add v4 w2 w1)))))) := by
    exact assumption_38
  -- chapter_22_line_26: GL tag task formulation.
  have row_26 : (add v1 v2 v3) := by
    exact assumption_26
  -- chapter_22_line_14: GL tag recursion.
  have row_14 : (succ previous v2) := by
    exact assumption_14
  -- chapter_22_line_5: GL tag task formulation.
  have row_5 : (succ v2 v5) := by
    exact assumption_5
  -- chapter_22_line_3: GL tag task formulation.
  have row_3 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_22_line_13: GL tag expansion.
  have row_13 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_3
  -- chapter_22_line_12: GL tag disintegration.
  have row_12 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_13.1
  -- chapter_22_line_11: GL tag expansion.
  have row_11 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_12
  -- chapter_22_line_33: GL tag disintegration.
  have row_33 : (gl_fXYZ add N N N) := by
    exact row_11.1.1.1.1.1.1.1.1.2
  -- chapter_22_line_32: GL tag expansion.
  have row_32 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_33
  -- chapter_22_line_36: GL tag disintegration.
  have row_36 : (gl_implication8 add N) := by
    exact row_32.1.1.1.1
  -- chapter_22_line_35: GL tag expansion.
  have row_35 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_36
  -- chapter_22_line_34: GL tag implication.
  have row_34 : (N v1) := by
    apply row_35
    exact row_26
  -- chapter_22_line_31: GL tag disintegration.
  have row_31 : (gl_implication13 N N N add) := by
    exact row_32.1.2
  -- chapter_22_line_30: GL tag expansion.
  have row_30 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication13] using row_31
  -- chapter_22_line_25: GL tag disintegration.
  have row_25 : (gl_implication17 N succ add) := by
    exact row_11.1.1.1.1.1.2
  -- chapter_22_line_24: GL tag expansion.
  have row_24 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication17] using row_25
  -- chapter_22_line_10: GL tag disintegration.
  have row_10 : (gl_fXY succ N N) := by
    exact row_11.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_22_line_9: GL tag expansion.
  have row_9 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_10
  -- chapter_22_line_22: GL tag disintegration.
  have row_22 : (gl_implication0 succ N) := by
    exact row_9.1.1.1
  -- chapter_22_line_21: GL tag expansion.
  have row_21 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_22
  -- chapter_22_line_20: GL tag implication.
  have row_20 : (N previous) := by
    apply row_21
    exact row_14
  -- chapter_22_line_29: GL tag implication.
  have row_29 : (gl_existence1 N v1 previous add) := by
    apply row_30
    exact row_34
    exact row_20
  -- chapter_22_line_28: GL tag expansion.
  have row_28 : (¬ (∀ (v7 : α), ((N v7) → (¬ (add v1 previous v7))))) := by
    simpa only [gl_existence1] using row_29
  have exists_row_28 : ∃ (v7 : α), ((N v7) ∧ (add v1 previous v7)) := existsAndOfNotForallImpNot row_28
  obtain ⟨v7, witness_row_28⟩ := exists_row_28
  -- chapter_22_line_27: GL tag disintegration.
  have row_27 : (add v1 previous v7) := by
    exact witness_row_28.2
  -- chapter_22_line_23: GL tag implication.
  have row_23 : (succ v7 v3) := by
    apply row_24
    exact row_20
    exact row_14
    exact row_27
    exact row_26
  -- chapter_22_line_19: GL tag disintegration.
  have row_19 : (gl_implication4 N N succ) := by
    exact row_9.1.2
  -- chapter_22_line_18: GL tag expansion.
  have row_18 : (∀ (w1 : α), ((N w1) → (gl_existence0 N w1 succ))) := by
    simpa only [gl_implication4] using row_19
  -- chapter_22_line_17: GL tag implication.
  have row_17 : (gl_existence0 N previous succ) := by
    apply row_18
    exact row_20
  -- chapter_22_line_16: GL tag expansion.
  have row_16 : (¬ (∀ (v6 : α), ((N v6) → (¬ (succ previous v6))))) := by
    simpa only [gl_existence0] using row_17
  have exists_row_16 : ∃ (v6 : α), ((N v6) ∧ (succ previous v6)) := existsAndOfNotForallImpNot row_16
  obtain ⟨v6, witness_row_16⟩ := exists_row_16
  -- chapter_22_line_15: GL tag disintegration.
  have row_15 : (succ previous v6) := by
    exact witness_row_16.2
  -- chapter_22_line_37: GL tag implication.
  have row_37 : (add v4 v6 v7) := by
    apply row_38
    exact row_27
    exact row_39
    exact row_15
  -- chapter_22_line_8: GL tag disintegration.
  have row_8 : (gl_implication5 N succ) := by
    exact row_9.2
  -- chapter_22_line_7: GL tag expansion.
  have row_7 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α), ((succ w1 w3) → (w2 = w3))))))) := by
    simpa only [gl_implication5] using row_8
  -- chapter_22_line_6: GL tag implication.
  have row_6 : (v2 = v6) := by
    apply row_7
    exact row_20
    exact row_14
    exact row_15
  -- chapter_22_line_4: GL tag equality1.
  have row_4 : (succ v6 v5) := by
    have equality_source := row_5
    have equality_step_1 := row_6
    cases equality_step_1
    exact equality_source
  -- chapter_22_line_2: GL tag theorem.
  have row_2 := peano_source_011 N zero succ add mul one anchor relationalInduction
  -- chapter_22_line_1: GL tag implication.
  have row_1 : (add v4 v5 v3) := by
    apply row_2
    exact row_37
    exact row_4
    exact row_23
  exact row_1

theorem peano_source_008
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α), ((succ v4 v1) → (∀ (v5 : α), ((succ v2 v5) → (add v4 v5 v3))))))) := by
  intro v1
  intro v2
  intro v3
  intro premise_1
  intro v4
  intro premise_2
  intro v5
  intro premise_3
  have inductionMember : N v2 := by
    have typingRule := peano_source_008_induction_typing N zero succ add mul one anchor relationalInduction
    exact typingRule v2 v5 premise_3
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v1 : α) (v3 : α), ((add v1 zero v3) → (∀ (v4 : α), ((succ v4 v1) → (∀ (v5 : α), ((succ zero v5) → (add v4 v5 v3))))))) := by
    intro v1
    intro v3
    intro base_premise_1
    intro v4
    intro base_premise_2
    intro v5
    intro base_premise_3
    have zeroRule := peano_source_008_check_zero N zero succ add mul one anchor relationalInduction
    exact zeroRule v1 zero v3 v4 v5 rfl base_premise_1 base_premise_2 base_premise_3
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v1 : α) (v3 : α), ((add v1 induction_n v3) → (∀ (v4 : α), ((succ v4 v1) → (∀ (v5 : α), ((succ induction_n v5) → (add v4 v5 v3))))))) → ∀ induction_m, succ induction_n induction_m → (∀ (v1 : α) (v3 : α), ((add v1 induction_m v3) → (∀ (v4 : α), ((succ v4 v1) → (∀ (v5 : α), ((succ induction_m v5) → (add v4 v5 v3))))))) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro v1
    intro v3
    intro step_premise_1
    intro v4
    intro step_premise_2
    intro v5
    intro step_premise_3
    have step_induction_assumption_1 :
        (∀ (w1 : α), ((add v1 induction_n w1) → ((succ v4 v1) → (∀ (w2 : α), ((succ induction_n w2) → (add v4 w2 w1)))))) := by
      intro w1
      intro step_induction_assumption_1_premise_1
      intro step_induction_assumption_1_premise_2
      intro w2
      intro step_induction_assumption_1_premise_3
      apply induction_hypothesis
      all_goals assumption
    have step_induction_assumption_2 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have stepRule := peano_source_008_check_induction_condition N zero succ add mul one anchor relationalInduction
    exact stepRule induction_n v1 induction_m v3 v4 v5 step_premise_2 step_induction_assumption_1 step_premise_1 step_induction_assumption_2 step_premise_3
  have inductionProperty : (∀ (v1 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α), ((succ v4 v1) → (∀ (v5 : α), ((succ v2 v5) → (add v4 v5 v3))))))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v1 : α) (v3 : α), ((add v1 induction_value v3) → (∀ (v4 : α), ((succ v4 v1) → (∀ (v5 : α), ((succ induction_value v5) → (add v4 v5 v3))))))))
      v2
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v1 v3 premise_1 v4 premise_2 v5 premise_3

private theorem peano_source_009_induction_typing
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v2 : α)
    (v5 : α)
    (assumption_10 : (succ v5 v2))
    : (N v5) := by
  -- chapter_23_line_10: GL tag task formulation.
  have row_10 : (succ v5 v2) := by
    exact assumption_10
  -- chapter_23_line_9: GL tag task formulation.
  have row_9 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_23_line_8: GL tag expansion.
  have row_8 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_9
  -- chapter_23_line_7: GL tag disintegration.
  have row_7 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_8.1
  -- chapter_23_line_6: GL tag expansion.
  have row_6 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_7
  -- chapter_23_line_5: GL tag disintegration.
  have row_5 : (gl_fXY succ N N) := by
    exact row_6.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_23_line_4: GL tag expansion.
  have row_4 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_5
  -- chapter_23_line_3: GL tag disintegration.
  have row_3 : (gl_implication0 succ N) := by
    exact row_4.1.1.1
  -- chapter_23_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_3
  -- chapter_23_line_1: GL tag implication.
  have row_1 : (N v5) := by
    apply row_2
    exact row_10
  exact row_1

private theorem peano_source_009_check_zero
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (v4 : α)
    (v5 : α)
    (assumption_35 : (v5 = zero))
    (assumption_34 : (add v1 v2 v3))
    (assumption_22 : (succ v5 v2))
    (assumption_18 : (succ v1 v4))
    : (add v4 v5 v3) := by
  -- chapter_24_line_35: GL tag recursion.
  have row_35 : (v5 = zero) := by
    exact assumption_35
  -- chapter_24_line_34: GL tag task formulation.
  have row_34 : (add v1 v2 v3) := by
    exact assumption_34
  -- chapter_24_line_26: GL tag theorem.
  have row_26 := peano_source_039 N zero succ add mul one anchor relationalInduction
  -- chapter_24_line_24: GL tag theorem.
  have row_24 := peano_source_027 N zero succ add mul one anchor relationalInduction
  -- chapter_24_line_22: GL tag task formulation.
  have row_22 : (succ v5 v2) := by
    exact assumption_22
  -- chapter_24_line_18: GL tag task formulation.
  have row_18 : (succ v1 v4) := by
    exact assumption_18
  -- chapter_24_line_12: GL tag task formulation.
  have row_12 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_24_line_11: GL tag expansion.
  have row_11 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_12
  -- chapter_24_line_10: GL tag disintegration.
  have row_10 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_11.1
  -- chapter_24_line_9: GL tag expansion.
  have row_9 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_10
  -- chapter_24_line_41: GL tag disintegration.
  have row_41 : (gl_implication17 N succ add) := by
    exact row_9.1.1.1.1.1.2
  -- chapter_24_line_40: GL tag expansion.
  have row_40 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication17] using row_41
  -- chapter_24_line_17: GL tag disintegration.
  have row_17 : (gl_fXY succ N N) := by
    exact row_9.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_24_line_16: GL tag expansion.
  have row_16 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_17
  -- chapter_24_line_21: GL tag disintegration.
  have row_21 : (gl_implication0 succ N) := by
    exact row_16.1.1.1
  -- chapter_24_line_20: GL tag expansion.
  have row_20 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_21
  -- chapter_24_line_19: GL tag implication.
  have row_19 : (N v5) := by
    apply row_20
    exact row_22
  -- chapter_24_line_15: GL tag disintegration.
  have row_15 : (gl_implication1 succ N) := by
    exact row_16.1.1.2
  -- chapter_24_line_14: GL tag expansion.
  have row_14 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using row_15
  -- chapter_24_line_13: GL tag implication.
  have row_13 : (N v4) := by
    apply row_14
    exact row_18
  -- chapter_24_line_8: GL tag disintegration.
  have row_8 : (gl_fXYZ add N N N) := by
    exact row_9.1.1.1.1.1.1.1.1.2
  -- chapter_24_line_7: GL tag expansion.
  have row_7 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_8
  -- chapter_24_line_33: GL tag disintegration.
  have row_33 : (gl_implication8 add N) := by
    exact row_7.1.1.1.1
  -- chapter_24_line_32: GL tag expansion.
  have row_32 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_33
  -- chapter_24_line_31: GL tag implication.
  have row_31 : (N v1) := by
    apply row_32
    exact row_34
  -- chapter_24_line_6: GL tag disintegration.
  have row_6 : (gl_implication13 N N N add) := by
    exact row_7.1.2
  -- chapter_24_line_5: GL tag expansion.
  have row_5 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication13] using row_6
  -- chapter_24_line_30: GL tag implication.
  have row_30 : (gl_existence1 N v1 v5 add) := by
    apply row_5
    exact row_31
    exact row_19
  -- chapter_24_line_29: GL tag expansion.
  have row_29 : (¬ (∀ (v9 : α), ((N v9) → (¬ (add v1 v5 v9))))) := by
    simpa only [gl_existence1] using row_30
  have exists_row_29 : ∃ (v9 : α), ((N v9) ∧ (add v1 v5 v9)) := existsAndOfNotForallImpNot row_29
  obtain ⟨v9, witness_row_29⟩ := exists_row_29
  -- chapter_24_line_28: GL tag disintegration.
  have row_28 : (add v1 v5 v9) := by
    exact witness_row_29.2
  -- chapter_24_line_39: GL tag implication.
  have row_39 : (succ v9 v3) := by
    apply row_40
    exact row_19
    exact row_22
    exact row_28
    exact row_34
  -- chapter_24_line_27: GL tag equality1.
  have row_27 : (add v1 zero v9) := by
    have equality_source := row_28
    have equality_step_1 := row_35
    cases equality_step_1
    exact equality_source
  -- chapter_24_line_25: GL tag implication.
  have row_25 : (v1 = v9) := by
    apply row_26
    exact row_27
    exact row_31
  -- chapter_24_line_4: GL tag implication.
  have row_4 : (gl_existence1 N v4 v5 add) := by
    apply row_5
    exact row_13
    exact row_19
  -- chapter_24_line_3: GL tag expansion.
  have row_3 : (¬ (∀ (v6 : α), ((N v6) → (¬ (add v4 v5 v6))))) := by
    simpa only [gl_existence1] using row_4
  have exists_row_3 : ∃ (v6 : α), ((N v6) ∧ (add v4 v5 v6)) := existsAndOfNotForallImpNot row_3
  obtain ⟨v6, witness_row_3⟩ := exists_row_3
  -- chapter_24_line_2: GL tag disintegration.
  have row_2 : (add v4 v5 v6) := by
    exact witness_row_3.2
  -- chapter_24_line_38: GL tag equality1.
  have row_38 : (add v4 zero v6) := by
    have equality_source := row_2
    have equality_step_1 := row_35
    cases equality_step_1
    exact equality_source
  -- chapter_24_line_37: GL tag implication.
  have row_37 : (v4 = v6) := by
    apply row_26
    exact row_38
    exact row_13
  -- chapter_24_line_36: GL tag equality1.
  have row_36 : (succ v1 v6) := by
    have equality_source := row_18
    have equality_step_1 := row_37
    cases equality_step_1
    exact equality_source
  -- chapter_24_line_23: GL tag implication.
  have row_23 : (v6 = v3) := by
    apply row_24
    exact row_36
    exact row_39
    exact row_25
  -- chapter_24_line_1: GL tag equality1.
  have row_1 : (add v4 v5 v3) := by
    have equality_source := row_2
    have equality_step_1 := row_23
    cases equality_step_1
    exact equality_source
  exact row_1

private theorem peano_source_009_check_induction_condition
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (v4 : α)
    (v5 : α)
    (assumption_30 : (succ v1 v4))
    (assumption_29 : (∀ (w1 : α) (w2 : α), ((add v1 w1 w2) → ((succ v1 v4) → ((succ previous w1) → (add v4 previous w2))))))
    (assumption_27 : (add v1 v2 v3))
    (assumption_23 : (succ previous v5))
    (assumption_10 : (succ v5 v2))
    : (add v4 v5 v3) := by
  -- chapter_25_line_30: GL tag task formulation.
  have row_30 : (succ v1 v4) := by
    exact assumption_30
  -- chapter_25_line_29: GL tag recursion.
  have row_29 : (∀ (w1 : α) (w2 : α), ((add v1 w1 w2) → ((succ v1 v4) → ((succ previous w1) → (add v4 previous w2))))) := by
    exact assumption_29
  -- chapter_25_line_27: GL tag task formulation.
  have row_27 : (add v1 v2 v3) := by
    exact assumption_27
  -- chapter_25_line_23: GL tag recursion.
  have row_23 : (succ previous v5) := by
    exact assumption_23
  -- chapter_25_line_10: GL tag task formulation.
  have row_10 : (succ v5 v2) := by
    exact assumption_10
  -- chapter_25_line_3: GL tag task formulation.
  have row_3 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_25_line_9: GL tag expansion.
  have row_9 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_3
  -- chapter_25_line_8: GL tag disintegration.
  have row_8 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_9.1
  -- chapter_25_line_7: GL tag expansion.
  have row_7 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_8
  -- chapter_25_line_22: GL tag disintegration.
  have row_22 : (gl_fXY succ N N) := by
    exact row_7.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_25_line_21: GL tag expansion.
  have row_21 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_22
  -- chapter_25_line_20: GL tag disintegration.
  have row_20 : (gl_implication1 succ N) := by
    exact row_21.1.1.2
  -- chapter_25_line_19: GL tag expansion.
  have row_19 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using row_20
  -- chapter_25_line_18: GL tag implication.
  have row_18 : (N v5) := by
    apply row_19
    exact row_23
  -- chapter_25_line_17: GL tag disintegration.
  have row_17 : (gl_fXYZ add N N N) := by
    exact row_7.1.1.1.1.1.1.1.1.2
  -- chapter_25_line_16: GL tag expansion.
  have row_16 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_17
  -- chapter_25_line_26: GL tag disintegration.
  have row_26 : (gl_implication8 add N) := by
    exact row_16.1.1.1.1
  -- chapter_25_line_25: GL tag expansion.
  have row_25 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_26
  -- chapter_25_line_24: GL tag implication.
  have row_24 : (N v1) := by
    apply row_25
    exact row_27
  -- chapter_25_line_15: GL tag disintegration.
  have row_15 : (gl_implication13 N N N add) := by
    exact row_16.1.2
  -- chapter_25_line_14: GL tag expansion.
  have row_14 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication13] using row_15
  -- chapter_25_line_13: GL tag implication.
  have row_13 : (gl_existence1 N v1 v5 add) := by
    apply row_14
    exact row_24
    exact row_18
  -- chapter_25_line_12: GL tag expansion.
  have row_12 : (¬ (∀ (v6 : α), ((N v6) → (¬ (add v1 v5 v6))))) := by
    simpa only [gl_existence1] using row_13
  have exists_row_12 : ∃ (v6 : α), ((N v6) ∧ (add v1 v5 v6)) := existsAndOfNotForallImpNot row_12
  obtain ⟨v6, witness_row_12⟩ := exists_row_12
  -- chapter_25_line_11: GL tag disintegration.
  have row_11 : (add v1 v5 v6) := by
    exact witness_row_12.2
  -- chapter_25_line_28: GL tag implication.
  have row_28 : (add v4 previous v6) := by
    apply row_29
    exact row_11
    exact row_30
    exact row_23
  -- chapter_25_line_6: GL tag disintegration.
  have row_6 : (gl_implication17 N succ add) := by
    exact row_7.1.1.1.1.1.2
  -- chapter_25_line_5: GL tag expansion.
  have row_5 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication17] using row_6
  -- chapter_25_line_4: GL tag implication.
  have row_4 : (succ v6 v3) := by
    apply row_5
    exact row_18
    exact row_10
    exact row_11
    exact row_27
  -- chapter_25_line_2: GL tag theorem.
  have row_2 := peano_source_011 N zero succ add mul one anchor relationalInduction
  -- chapter_25_line_1: GL tag implication.
  have row_1 : (add v4 v5 v3) := by
    apply row_2
    exact row_28
    exact row_23
    exact row_4
  exact row_1

theorem peano_source_009
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α), ((succ v1 v4) → (∀ (v5 : α), ((succ v5 v2) → (add v4 v5 v3))))))) := by
  intro v1
  intro v2
  intro v3
  intro premise_1
  intro v4
  intro premise_2
  intro v5
  intro premise_3
  have inductionMember : N v5 := by
    have typingRule := peano_source_009_induction_typing N zero succ add mul one anchor relationalInduction
    exact typingRule v2 v5 premise_3
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α), ((succ v1 v4) → ((succ zero v2) → (add v4 zero v3)))))) := by
    intro v1
    intro v2
    intro v3
    intro base_premise_1
    intro v4
    intro base_premise_2
    intro base_premise_3
    have zeroRule := peano_source_009_check_zero N zero succ add mul one anchor relationalInduction
    exact zeroRule v1 v2 v3 v4 zero rfl base_premise_1 base_premise_3 base_premise_2
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α), ((succ v1 v4) → ((succ induction_n v2) → (add v4 induction_n v3)))))) → ∀ induction_m, succ induction_n induction_m → (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α), ((succ v1 v4) → ((succ induction_m v2) → (add v4 induction_m v3)))))) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro v1
    intro v2
    intro v3
    intro step_premise_1
    intro v4
    intro step_premise_2
    intro step_premise_3
    have step_induction_assumption_1 :
        (∀ (w1 : α) (w2 : α), ((add v1 w1 w2) → ((succ v1 v4) → ((succ induction_n w1) → (add v4 induction_n w2))))) := by
      intro w1
      intro w2
      intro step_induction_assumption_1_premise_1
      intro step_induction_assumption_1_premise_2
      intro step_induction_assumption_1_premise_3
      apply induction_hypothesis
      all_goals assumption
    have step_induction_assumption_2 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have stepRule := peano_source_009_check_induction_condition N zero succ add mul one anchor relationalInduction
    exact stepRule induction_n v1 v2 v3 v4 induction_m step_premise_2 step_induction_assumption_1 step_premise_1 step_induction_assumption_2 step_premise_3
  have inductionProperty : (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α), ((succ v1 v4) → ((succ v5 v2) → (add v4 v5 v3)))))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α), ((succ v1 v4) → ((succ induction_value v2) → (add v4 induction_value v3)))))))
      v5
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v1 v2 v3 premise_1 v4 premise_2 premise_3

private theorem peano_source_012_induction_typing
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v2 : α)
    (v3 : α)
    (v4 : α)
    (assumption_10 : (add v4 v2 v3))
    : (N v2) := by
  -- chapter_30_line_10: GL tag task formulation.
  have row_10 : (add v4 v2 v3) := by
    exact assumption_10
  -- chapter_30_line_9: GL tag task formulation.
  have row_9 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_30_line_8: GL tag expansion.
  have row_8 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_9
  -- chapter_30_line_7: GL tag disintegration.
  have row_7 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_8.1
  -- chapter_30_line_6: GL tag expansion.
  have row_6 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_7
  -- chapter_30_line_5: GL tag disintegration.
  have row_5 : (gl_fXYZ add N N N) := by
    exact row_6.1.1.1.1.1.1.1.1.2
  -- chapter_30_line_4: GL tag expansion.
  have row_4 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_5
  -- chapter_30_line_3: GL tag disintegration.
  have row_3 : (gl_implication9 add N) := by
    exact row_4.1.1.1.2
  -- chapter_30_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication9] using row_3
  -- chapter_30_line_1: GL tag implication.
  have row_1 : (N v2) := by
    apply row_2
    exact row_10
  exact row_1

private theorem peano_source_012_check_zero
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (v4 : α)
    (assumption_30 : (add v1 v2 v3))
    (assumption_14 : (v2 = zero))
    (assumption_13 : (add v4 v2 v3))
    : (v1 = v4) := by
  -- chapter_31_line_30: GL tag task formulation.
  have row_30 : (add v1 v2 v3) := by
    exact assumption_30
  -- chapter_31_line_18: GL tag variable copy.
  have row_18 : (v3 = v3) := by
    rfl
  -- chapter_31_line_16: GL tag theorem.
  have row_16 := peano_source_039 N zero succ add mul one anchor relationalInduction
  -- chapter_31_line_14: GL tag recursion.
  have row_14 : (v2 = zero) := by
    exact assumption_14
  -- chapter_31_line_29: GL tag equality1.
  have row_29 : (add v1 zero v3) := by
    have equality_source := row_30
    have equality_step_1 := row_14
    cases equality_step_1
    exact equality_source
  -- chapter_31_line_24: GL tag symmetry of equality.
  have row_24 : (zero = v2) := by
    exact Eq.symm row_14
  -- chapter_31_line_13: GL tag task formulation.
  have row_13 : (add v4 v2 v3) := by
    exact assumption_13
  -- chapter_31_line_38: GL tag equality1.
  have row_38 : (add v4 v2 v3) := by
    have equality_source := row_13
    have equality_step_1 := row_18
    cases equality_step_1
    exact equality_source
  -- chapter_31_line_12: GL tag equality1.
  have row_12 : (add v4 zero v3) := by
    have equality_source := row_13
    have equality_step_1 := row_14
    cases equality_step_1
    exact equality_source
  -- chapter_31_line_17: GL tag equality1.
  have row_17 : (add v4 zero v3) := by
    have equality_source := row_12
    have equality_step_1 := row_18
    cases equality_step_1
    exact equality_source
  -- chapter_31_line_9: GL tag task formulation.
  have row_9 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_31_line_8: GL tag expansion.
  have row_8 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_9
  -- chapter_31_line_7: GL tag disintegration.
  have row_7 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_8.1
  -- chapter_31_line_6: GL tag expansion.
  have row_6 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_7
  -- chapter_31_line_5: GL tag disintegration.
  have row_5 : (gl_fXYZ add N N N) := by
    exact row_6.1.1.1.1.1.1.1.1.2
  -- chapter_31_line_4: GL tag expansion.
  have row_4 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_5
  -- chapter_31_line_37: GL tag disintegration.
  have row_37 : (gl_implication10 add N) := by
    exact row_4.1.1.2
  -- chapter_31_line_36: GL tag expansion.
  have row_36 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication10] using row_37
  -- chapter_31_line_35: GL tag implication.
  have row_35 : (N v3) := by
    apply row_36
    exact row_38
  -- chapter_31_line_34: GL tag disintegration.
  have row_34 : (gl_implication9 add N) := by
    exact row_4.1.1.1.2
  -- chapter_31_line_33: GL tag expansion.
  have row_33 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication9] using row_34
  -- chapter_31_line_32: GL tag implication.
  have row_32 : (N v2) := by
    apply row_33
    exact row_13
  -- chapter_31_line_21: GL tag disintegration.
  have row_21 : (gl_implication8 add N) := by
    exact row_4.1.1.1.1
  -- chapter_31_line_20: GL tag expansion.
  have row_20 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_21
  -- chapter_31_line_31: GL tag implication.
  have row_31 : (N v1) := by
    apply row_20
    exact row_30
  -- chapter_31_line_28: GL tag implication.
  have row_28 : (v1 = v3) := by
    apply row_16
    exact row_29
    exact row_31
  -- chapter_31_line_27: GL tag symmetry of equality.
  have row_27 : (v3 = v1) := by
    exact Eq.symm row_28
  -- chapter_31_line_19: GL tag implication.
  have row_19 : (N v4) := by
    apply row_20
    exact row_13
  -- chapter_31_line_23: GL tag implication.
  have row_23 : (v4 = v3) := by
    apply row_16
    exact row_12
    exact row_19
  -- chapter_31_line_22: GL tag symmetry of equality.
  have row_22 : (v3 = v4) := by
    exact Eq.symm row_23
  -- chapter_31_line_15: GL tag implication.
  have row_15 : (v4 = v3) := by
    apply row_16
    exact row_17
    exact row_19
  -- chapter_31_line_26: GL tag equality1.
  have row_26 : (add v3 zero v1) := by
    have equality_source := row_12
    have equality_step_1 := row_15
    cases equality_step_1
    have equality_step_2 := row_27
    cases equality_step_2
    exact equality_source
  -- chapter_31_line_25: GL tag equality1.
  have row_25 : (add v3 v2 v1) := by
    have equality_source := row_26
    have equality_step_1 := row_24
    cases equality_step_1
    exact equality_source
  -- chapter_31_line_11: GL tag equality1.
  have row_11 : (add v3 zero v4) := by
    have equality_source := row_12
    have equality_step_1 := row_15
    cases equality_step_1
    have equality_step_2 := row_22
    cases equality_step_2
    exact equality_source
  -- chapter_31_line_10: GL tag equality1.
  have row_10 : (add v3 v2 v4) := by
    have equality_source := row_11
    have equality_step_1 := row_24
    cases equality_step_1
    exact equality_source
  -- chapter_31_line_3: GL tag disintegration.
  have row_3 : (gl_implication14 N N add) := by
    exact row_4.2
  -- chapter_31_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (∀ (w3 : α), ((add w1 w2 w3) → (∀ (w4 : α), ((add w1 w2 w4) → (w3 = w4))))))))) := by
    simpa only [gl_implication14] using row_3
  -- chapter_31_line_1: GL tag implication.
  have row_1 : (v1 = v4) := by
    apply row_2
    exact row_35
    exact row_32
    exact row_25
    exact row_10
  exact row_1

private theorem peano_source_012_check_induction_condition
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (v4 : α)
    (assumption_32 : (add v1 v2 v3))
    (assumption_26 : (succ previous v2))
    (assumption_19 : (add v4 v2 v3))
    (assumption_2 : (∀ (w1 : α), ((add v1 previous w1) → ((add v4 previous w1) → (v1 = v4)))))
    : (v1 = v4) := by
  -- chapter_32_line_32: GL tag task formulation.
  have row_32 : (add v1 v2 v3) := by
    exact assumption_32
  -- chapter_32_line_28: GL tag theorem.
  have row_28 := peano_source_030 N zero succ add mul one anchor relationalInduction
  -- chapter_32_line_26: GL tag recursion.
  have row_26 : (succ previous v2) := by
    exact assumption_26
  -- chapter_32_line_20: GL tag variable copy.
  have row_20 : (v3 = v3) := by
    rfl
  -- chapter_32_line_37: GL tag equality1.
  have row_37 : (add v1 v2 v3) := by
    have equality_source := row_32
    have equality_step_1 := row_20
    cases equality_step_1
    exact equality_source
  -- chapter_32_line_19: GL tag task formulation.
  have row_19 : (add v4 v2 v3) := by
    exact assumption_19
  -- chapter_32_line_18: GL tag equality1.
  have row_18 : (add v4 v2 v3) := by
    have equality_source := row_19
    have equality_step_1 := row_20
    cases equality_step_1
    exact equality_source
  -- chapter_32_line_14: GL tag task formulation.
  have row_14 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_32_line_13: GL tag expansion.
  have row_13 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_14
  -- chapter_32_line_12: GL tag disintegration.
  have row_12 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_13.1
  -- chapter_32_line_11: GL tag expansion.
  have row_11 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_12
  -- chapter_32_line_31: GL tag disintegration.
  have row_31 : (gl_implication17 N succ add) := by
    exact row_11.1.1.1.1.1.2
  -- chapter_32_line_30: GL tag expansion.
  have row_30 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication17] using row_31
  -- chapter_32_line_25: GL tag disintegration.
  have row_25 : (gl_fXY succ N N) := by
    exact row_11.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_32_line_24: GL tag expansion.
  have row_24 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_25
  -- chapter_32_line_23: GL tag disintegration.
  have row_23 : (gl_implication0 succ N) := by
    exact row_24.1.1.1
  -- chapter_32_line_22: GL tag expansion.
  have row_22 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_23
  -- chapter_32_line_21: GL tag implication.
  have row_21 : (N previous) := by
    apply row_22
    exact row_26
  -- chapter_32_line_10: GL tag disintegration.
  have row_10 : (gl_fXYZ add N N N) := by
    exact row_11.1.1.1.1.1.1.1.1.2
  -- chapter_32_line_9: GL tag expansion.
  have row_9 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_10
  -- chapter_32_line_17: GL tag disintegration.
  have row_17 : (gl_implication8 add N) := by
    exact row_9.1.1.1.1
  -- chapter_32_line_16: GL tag expansion.
  have row_16 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_17
  -- chapter_32_line_36: GL tag implication.
  have row_36 : (N v1) := by
    apply row_16
    exact row_37
  -- chapter_32_line_15: GL tag implication.
  have row_15 : (N v4) := by
    apply row_16
    exact row_18
  -- chapter_32_line_8: GL tag disintegration.
  have row_8 : (gl_implication13 N N N add) := by
    exact row_9.1.2
  -- chapter_32_line_7: GL tag expansion.
  have row_7 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication13] using row_8
  -- chapter_32_line_35: GL tag implication.
  have row_35 : (gl_existence1 N v1 previous add) := by
    apply row_7
    exact row_36
    exact row_21
  -- chapter_32_line_34: GL tag expansion.
  have row_34 : (¬ (∀ (v6 : α), ((N v6) → (¬ (add v1 previous v6))))) := by
    simpa only [gl_existence1] using row_35
  have exists_row_34 : ∃ (v6 : α), ((N v6) ∧ (add v1 previous v6)) := existsAndOfNotForallImpNot row_34
  obtain ⟨v6, witness_row_34⟩ := exists_row_34
  -- chapter_32_line_33: GL tag disintegration.
  have row_33 : (add v1 previous v6) := by
    exact witness_row_34.2
  -- chapter_32_line_29: GL tag implication.
  have row_29 : (succ v6 v3) := by
    apply row_30
    exact row_21
    exact row_26
    exact row_33
    exact row_32
  -- chapter_32_line_6: GL tag implication.
  have row_6 : (gl_existence1 N v4 previous add) := by
    apply row_7
    exact row_15
    exact row_21
  -- chapter_32_line_5: GL tag expansion.
  have row_5 : (¬ (∀ (v7 : α), ((N v7) → (¬ (add v4 previous v7))))) := by
    simpa only [gl_existence1] using row_6
  have exists_row_5 : ∃ (v7 : α), ((N v7) ∧ (add v4 previous v7)) := existsAndOfNotForallImpNot row_5
  obtain ⟨v7, witness_row_5⟩ := exists_row_5
  -- chapter_32_line_4: GL tag disintegration.
  have row_4 : (add v4 previous v7) := by
    exact witness_row_5.2
  -- chapter_32_line_38: GL tag implication.
  have row_38 : (succ v7 v3) := by
    apply row_30
    exact row_21
    exact row_26
    exact row_4
    exact row_19
  -- chapter_32_line_27: GL tag implication.
  have row_27 : (v7 = v6) := by
    apply row_28
    exact row_38
    exact row_29
  -- chapter_32_line_3: GL tag equality1.
  have row_3 : (add v4 previous v6) := by
    have equality_source := row_4
    have equality_step_1 := row_27
    cases equality_step_1
    exact equality_source
  -- chapter_32_line_2: GL tag recursion.
  have row_2 : (∀ (w1 : α), ((add v1 previous w1) → ((add v4 previous w1) → (v1 = v4)))) := by
    exact assumption_2
  -- chapter_32_line_1: GL tag implication.
  have row_1 : (v1 = v4) := by
    apply row_2
    exact row_33
    exact row_3
  exact row_1

theorem peano_source_012
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α), ((add v4 v2 v3) → (v1 = v4))))) := by
  intro v1
  intro v2
  intro v3
  intro premise_1
  intro v4
  intro premise_2
  have inductionMember : N v2 := by
    have typingRule := peano_source_012_induction_typing N zero succ add mul one anchor relationalInduction
    exact typingRule v2 v3 v4 premise_2
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v1 : α) (v3 : α), ((add v1 zero v3) → (∀ (v4 : α), ((add v4 zero v3) → (v1 = v4))))) := by
    intro v1
    intro v3
    intro base_premise_1
    intro v4
    intro base_premise_2
    have zeroRule := peano_source_012_check_zero N zero succ add mul one anchor relationalInduction
    exact zeroRule v1 zero v3 v4 base_premise_1 rfl base_premise_2
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v1 : α) (v3 : α), ((add v1 induction_n v3) → (∀ (v4 : α), ((add v4 induction_n v3) → (v1 = v4))))) → ∀ induction_m, succ induction_n induction_m → (∀ (v1 : α) (v3 : α), ((add v1 induction_m v3) → (∀ (v4 : α), ((add v4 induction_m v3) → (v1 = v4))))) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro v1
    intro v3
    intro step_premise_1
    intro v4
    intro step_premise_2
    have step_induction_assumption_1 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have step_induction_assumption_2 :
        (∀ (w1 : α), ((add v1 induction_n w1) → ((add v4 induction_n w1) → (v1 = v4)))) := by
      intro w1
      intro step_induction_assumption_2_premise_1
      intro step_induction_assumption_2_premise_2
      apply induction_hypothesis
      all_goals assumption
    have stepRule := peano_source_012_check_induction_condition N zero succ add mul one anchor relationalInduction
    exact stepRule induction_n v1 induction_m v3 v4 step_premise_1 step_induction_assumption_1 step_premise_2 step_induction_assumption_2
  have inductionProperty : (∀ (v1 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α), ((add v4 v2 v3) → (v1 = v4))))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v1 : α) (v3 : α), ((add v1 induction_value v3) → (∀ (v4 : α), ((add v4 induction_value v3) → (v1 = v4))))))
      v2
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v1 v3 premise_1 v4 premise_2

private theorem peano_source_025_induction_typing
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (assumption_12 : (v1 = v2))
    (assumption_11 : (mul v2 v1 zero))
    : (N v1) := by
  -- chapter_61_line_12: GL tag task formulation.
  have row_12 : (v1 = v2) := by
    exact assumption_12
  -- chapter_61_line_13: GL tag symmetry of equality.
  have row_13 : (v2 = v1) := by
    exact Eq.symm row_12
  -- chapter_61_line_11: GL tag task formulation.
  have row_11 : (mul v2 v1 zero) := by
    exact assumption_11
  -- chapter_61_line_10: GL tag equality1.
  have row_10 : (mul v1 v2 zero) := by
    have equality_source := row_11
    have equality_step_1 := row_12
    cases equality_step_1
    have equality_step_2 := row_13
    cases equality_step_2
    exact equality_source
  -- chapter_61_line_9: GL tag task formulation.
  have row_9 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_61_line_8: GL tag expansion.
  have row_8 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_9
  -- chapter_61_line_7: GL tag disintegration.
  have row_7 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_8.1
  -- chapter_61_line_6: GL tag expansion.
  have row_6 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_7
  -- chapter_61_line_5: GL tag disintegration.
  have row_5 : (gl_fXYZ mul N N N) := by
    exact row_6.1.1.1.2
  -- chapter_61_line_4: GL tag expansion.
  have row_4 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_5
  -- chapter_61_line_3: GL tag disintegration.
  have row_3 : (gl_implication8 mul N) := by
    exact row_4.1.1.1.1
  -- chapter_61_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_3
  -- chapter_61_line_1: GL tag implication.
  have row_1 : (N v1) := by
    apply row_2
    exact row_10
  exact row_1

private theorem peano_source_025_check_zero
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (assumption_4 : (v1 = v2))
    (assumption_3 : (v1 = zero))
    : (zero = v2) := by
  -- chapter_62_line_4: GL tag task formulation.
  have row_4 : (v1 = v2) := by
    exact assumption_4
  -- chapter_62_line_3: GL tag recursion.
  have row_3 : (v1 = zero) := by
    exact assumption_3
  -- chapter_62_line_2: GL tag symmetry of equality.
  have row_2 : (zero = v1) := by
    exact Eq.symm row_3
  -- chapter_62_line_1: GL tag equality2.
  have row_1 : (zero = v2) := by
    exact Eq.trans row_2 row_4
  exact row_1

private theorem peano_source_025_check_induction_condition
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (assumption_26 : (v1 = v2))
    (assumption_11 : (mul v2 v1 zero))
    (assumption_10 : (succ previous v1))
    : (zero = v2) := by
  -- chapter_63_line_26: GL tag task formulation.
  have row_26 : (v1 = v2) := by
    exact assumption_26
  -- chapter_63_line_25: GL tag symmetry of equality.
  have row_25 : (v2 = v1) := by
    exact Eq.symm row_26
  -- chapter_63_line_11: GL tag task formulation.
  have row_11 : (mul v2 v1 zero) := by
    exact assumption_11
  -- chapter_63_line_24: GL tag equality1.
  have row_24 : (mul v1 v1 zero) := by
    have equality_source := row_11
    have equality_step_1 := row_25
    cases equality_step_1
    exact equality_source
  -- chapter_63_line_23: GL tag equality1.
  have row_23 : (mul v1 v2 zero) := by
    have equality_source := row_24
    have equality_step_1 := row_26
    cases equality_step_1
    exact equality_source
  -- chapter_63_line_10: GL tag recursion.
  have row_10 : (succ previous v1) := by
    exact assumption_10
  -- chapter_63_line_3: GL tag task formulation.
  have row_3 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_63_line_9: GL tag expansion.
  have row_9 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_3
  -- chapter_63_line_8: GL tag disintegration.
  have row_8 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_9.1
  -- chapter_63_line_7: GL tag expansion.
  have row_7 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_8
  -- chapter_63_line_31: GL tag disintegration.
  have row_31 : (gl_fXY succ N N) := by
    exact row_7.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_63_line_30: GL tag expansion.
  have row_30 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_31
  -- chapter_63_line_29: GL tag disintegration.
  have row_29 : (gl_implication0 succ N) := by
    exact row_30.1.1.1
  -- chapter_63_line_28: GL tag expansion.
  have row_28 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_29
  -- chapter_63_line_27: GL tag implication.
  have row_27 : (N previous) := by
    apply row_28
    exact row_10
  -- chapter_63_line_19: GL tag disintegration.
  have row_19 : (gl_fXYZ mul N N N) := by
    exact row_7.1.1.1.2
  -- chapter_63_line_18: GL tag expansion.
  have row_18 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_19
  -- chapter_63_line_22: GL tag disintegration.
  have row_22 : (gl_implication8 mul N) := by
    exact row_18.1.1.1.1
  -- chapter_63_line_21: GL tag expansion.
  have row_21 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_22
  -- chapter_63_line_20: GL tag implication.
  have row_20 : (N v1) := by
    apply row_21
    exact row_23
  -- chapter_63_line_17: GL tag disintegration.
  have row_17 : (gl_implication13 N N N mul) := by
    exact row_18.1.2
  -- chapter_63_line_16: GL tag expansion.
  have row_16 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication13] using row_17
  -- chapter_63_line_15: GL tag implication.
  have row_15 : (gl_existence1 N v1 previous mul) := by
    apply row_16
    exact row_20
    exact row_27
  -- chapter_63_line_14: GL tag expansion.
  have row_14 : (¬ (∀ (v3 : α), ((N v3) → (¬ (mul v1 previous v3))))) := by
    simpa only [gl_existence1] using row_15
  have exists_row_14 : ∃ (v3 : α), ((N v3) ∧ (mul v1 previous v3)) := existsAndOfNotForallImpNot row_14
  obtain ⟨v3, witness_row_14⟩ := exists_row_14
  -- chapter_63_line_32: GL tag disintegration.
  have row_32 : (N v3) := by
    exact witness_row_14.1
  -- chapter_63_line_13: GL tag disintegration.
  have row_13 : (mul v1 previous v3) := by
    exact witness_row_14.2
  -- chapter_63_line_12: GL tag equality1.
  have row_12 : (mul v2 previous v3) := by
    have equality_source := row_13
    have equality_step_1 := row_26
    cases equality_step_1
    exact equality_source
  -- chapter_63_line_6: GL tag disintegration.
  have row_6 : (gl_implication21 N succ mul add) := by
    exact row_7.2
  -- chapter_63_line_5: GL tag expansion.
  have row_5 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((mul w3 w1 w4) → (∀ (w5 : α), ((mul w3 w2 w5) → (add w4 w3 w5))))))))) := by
    simpa only [gl_implication21] using row_6
  -- chapter_63_line_4: GL tag implication.
  have row_4 : (add v3 v2 zero) := by
    apply row_5
    exact row_27
    exact row_10
    exact row_12
    exact row_11
  -- chapter_63_line_2: GL tag theorem.
  have row_2 := peano_source_045 N zero succ add mul one anchor relationalInduction
  -- chapter_63_line_1: GL tag implication.
  have row_1 : (zero = v2) := by
    apply row_2
    exact row_4
    exact row_32
  exact row_1

theorem peano_source_025
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((v1 = v2) → ((mul v2 v1 zero) → (zero = v2)))) := by
  intro v1
  intro v2
  intro premise_1
  intro premise_2
  have inductionMember : N v1 := by
    have typingRule := peano_source_025_induction_typing N zero succ add mul one anchor relationalInduction
    exact typingRule v1 v2 premise_1 premise_2
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v2 : α), ((zero = v2) → ((mul v2 zero zero) → (zero = v2)))) := by
    intro v2
    intro base_premise_1
    intro base_premise_2
    have zeroRule := peano_source_025_check_zero N zero succ add mul one anchor relationalInduction
    exact zeroRule zero v2 base_premise_1 rfl
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v2 : α), ((induction_n = v2) → ((mul v2 induction_n zero) → (zero = v2)))) → ∀ induction_m, succ induction_n induction_m → (∀ (v2 : α), ((induction_m = v2) → ((mul v2 induction_m zero) → (zero = v2)))) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro v2
    intro step_premise_1
    intro step_premise_2
    have step_induction_assumption_1 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have stepRule := peano_source_025_check_induction_condition N zero succ add mul one anchor relationalInduction
    exact stepRule induction_n induction_m v2 step_premise_1 step_premise_2 step_induction_assumption_1
  have inductionProperty : (∀ (v2 : α), ((v1 = v2) → ((mul v2 v1 zero) → (zero = v2)))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v2 : α), ((induction_value = v2) → ((mul v2 induction_value zero) → (zero = v2)))))
      v1
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v2 premise_1 premise_2

theorem peano_source_028
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (v3 : α) (v4 : α), ((succ v3 v4) → (gl_or0 v2 v4 v1 v3))))) := by
  intro v1
  intro v2
  intro premise_1
  intro v3
  intro v4
  intro premise_2
  have or_parent_1 := peano_source_026 N zero succ add mul one anchor relationalInduction
  have or_parent_2 := peano_source_027 N zero succ add mul one anchor relationalInduction
  -- chapter_66_line_1: GL tag or theorem.
  have row_1 : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (v3 : α) (v4 : α), ((succ v3 v4) → (gl_or0 v2 v4 v1 v3))))) := by
    classical
    intro v1
    intro v2
    intro or_parent_premise_1
    intro v3
    intro v4
    intro or_parent_premise_2
    simp only [gl_or0]
    by_cases or_case_1 : (v2 = v4)
    · exact Or.inl (or_case_1)
    ·
      exact Or.inr ((or_parent_1 v1 v2 or_parent_premise_1 v3 v4 or_parent_premise_2 or_case_1))
  solve_by_elim [row_1]

theorem peano_source_029
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (v3 : α) (v4 : α), ((succ v3 v4) → (gl_or0 v2 v4 v1 v3))))) := by
  intro v1
  intro v2
  intro premise_1
  intro v3
  intro v4
  intro premise_2
  have or_parent_1 := peano_source_026 N zero succ add mul one anchor relationalInduction
  have or_parent_2 := peano_source_027 N zero succ add mul one anchor relationalInduction
  -- chapter_67_line_1: GL tag or theorem.
  have row_1 : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (v3 : α) (v4 : α), ((succ v3 v4) → (gl_or0 v2 v4 v1 v3))))) := by
    classical
    intro v1
    intro v2
    intro or_parent_premise_1
    intro v3
    intro v4
    intro or_parent_premise_2
    simp only [gl_or0]
    by_cases or_case_1 : (v2 = v4)
    · exact Or.inl (or_case_1)
    ·
      exact Or.inr ((or_parent_1 v1 v2 or_parent_premise_1 v3 v4 or_parent_premise_2 or_case_1))
  solve_by_elim [row_1]

theorem peano_source_034
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (add v1 one v2))) := by
  intro v1
  intro v2
  intro premise_1
  -- chapter_74_line_12: GL tag theorem.
  have row_12 := peano_source_039 N zero succ add mul one anchor relationalInduction
  -- chapter_74_line_10: GL tag task formulation.
  have row_10 : (succ v1 v2) := by
    exact premise_1
  -- chapter_74_line_7: GL tag task formulation.
  have row_7 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_74_line_6: GL tag expansion.
  have row_6 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_7
  -- chapter_74_line_8: GL tag disintegration.
  have row_8 : (succ zero one) := by
    exact row_6.2
  -- chapter_74_line_5: GL tag disintegration.
  have row_5 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_6.1
  -- chapter_74_line_4: GL tag expansion.
  have row_4 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_5
  -- chapter_74_line_25: GL tag disintegration.
  have row_25 : (gl_fXY succ N N) := by
    exact row_4.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_74_line_24: GL tag expansion.
  have row_24 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_25
  -- chapter_74_line_23: GL tag disintegration.
  have row_23 : (gl_implication0 succ N) := by
    exact row_24.1.1.1
  -- chapter_74_line_22: GL tag expansion.
  have row_22 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_23
  -- chapter_74_line_21: GL tag implication.
  have row_21 : (N v1) := by
    apply row_22
    exact row_10
  -- chapter_74_line_20: GL tag disintegration.
  have row_20 : (N zero) := by
    exact row_4.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_74_line_19: GL tag disintegration.
  have row_19 : (gl_fXYZ add N N N) := by
    exact row_4.1.1.1.1.1.1.1.1.2
  -- chapter_74_line_18: GL tag expansion.
  have row_18 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_19
  -- chapter_74_line_17: GL tag disintegration.
  have row_17 : (gl_implication13 N N N add) := by
    exact row_18.1.2
  -- chapter_74_line_16: GL tag expansion.
  have row_16 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication13] using row_17
  -- chapter_74_line_15: GL tag implication.
  have row_15 : (gl_existence1 N v1 zero add) := by
    apply row_16
    exact row_21
    exact row_20
  -- chapter_74_line_14: GL tag expansion.
  have row_14 : (¬ (∀ (v8 : α), ((N v8) → (¬ (add v1 zero v8))))) := by
    simpa only [gl_existence1] using row_15
  have exists_row_14 : ∃ (v8 : α), ((N v8) ∧ (add v1 zero v8)) := existsAndOfNotForallImpNot row_14
  obtain ⟨v8, witness_row_14⟩ := exists_row_14
  -- chapter_74_line_13: GL tag disintegration.
  have row_13 : (add v1 zero v8) := by
    exact witness_row_14.2
  -- chapter_74_line_11: GL tag implication.
  have row_11 : (v1 = v8) := by
    apply row_12
    exact row_13
    exact row_21
  -- chapter_74_line_9: GL tag equality1.
  have row_9 : (succ v8 v2) := by
    have equality_source := row_10
    have equality_step_1 := row_11
    cases equality_step_1
    exact equality_source
  -- chapter_74_line_3: GL tag disintegration.
  have row_3 : (gl_implication18 N succ add) := by
    exact row_4.1.1.1.1.2
  -- chapter_74_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((succ w4 w5) → (add w3 w2 w5))))))))) := by
    simpa only [gl_implication18] using row_3
  -- chapter_74_line_1: GL tag implication.
  have row_1 : (add v1 one v2) := by
    apply row_2
    exact row_20
    exact row_8
    exact row_13
    exact row_9
  exact row_1

private theorem peano_source_035_induction_typing
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (assumption_10 : (add zero v1 v2))
    : (N v1) := by
  -- chapter_75_line_10: GL tag task formulation.
  have row_10 : (add zero v1 v2) := by
    exact assumption_10
  -- chapter_75_line_9: GL tag task formulation.
  have row_9 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_75_line_8: GL tag expansion.
  have row_8 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_9
  -- chapter_75_line_7: GL tag disintegration.
  have row_7 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_8.1
  -- chapter_75_line_6: GL tag expansion.
  have row_6 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_7
  -- chapter_75_line_5: GL tag disintegration.
  have row_5 : (gl_fXYZ add N N N) := by
    exact row_6.1.1.1.1.1.1.1.1.2
  -- chapter_75_line_4: GL tag expansion.
  have row_4 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_5
  -- chapter_75_line_3: GL tag disintegration.
  have row_3 : (gl_implication9 add N) := by
    exact row_4.1.1.1.2
  -- chapter_75_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication9] using row_3
  -- chapter_75_line_1: GL tag implication.
  have row_1 : (N v1) := by
    apply row_2
    exact row_10
  exact row_1

private theorem peano_source_035_check_zero
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (assumption_4 : (v1 = zero))
    (assumption_3 : (add zero v1 v2))
    : (add v1 zero v2) := by
  -- chapter_76_line_4: GL tag recursion.
  have row_4 : (v1 = zero) := by
    exact assumption_4
  -- chapter_76_line_5: GL tag symmetry of equality.
  have row_5 : (zero = v1) := by
    exact Eq.symm row_4
  -- chapter_76_line_3: GL tag task formulation.
  have row_3 : (add zero v1 v2) := by
    exact assumption_3
  -- chapter_76_line_2: GL tag equality1.
  have row_2 : (add zero zero v2) := by
    have equality_source := row_3
    have equality_step_1 := row_4
    cases equality_step_1
    exact equality_source
  -- chapter_76_line_1: GL tag equality1.
  have row_1 : (add v1 zero v2) := by
    have equality_source := row_2
    have equality_step_1 := row_5
    cases equality_step_1
    exact equality_source
  exact row_1

private theorem peano_source_035_check_induction_condition
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (assumption_50 : (add zero v1 v2))
    (assumption_27 : (∀ (w1 : α), ((add zero previous w1) → (add previous zero w1))))
    (assumption_19 : (succ previous v1))
    : (add v1 zero v2) := by
  -- chapter_77_line_50: GL tag task formulation.
  have row_50 : (add zero v1 v2) := by
    exact assumption_50
  -- chapter_77_line_35: GL tag theorem.
  have row_35 := peano_source_039 N zero succ add mul one anchor relationalInduction
  -- chapter_77_line_27: GL tag recursion.
  have row_27 : (∀ (w1 : α), ((add zero previous w1) → (add previous zero w1))) := by
    exact assumption_27
  -- chapter_77_line_21: GL tag theorem.
  have row_21 := peano_source_027 N zero succ add mul one anchor relationalInduction
  -- chapter_77_line_19: GL tag recursion.
  have row_19 : (succ previous v1) := by
    exact assumption_19
  -- chapter_77_line_12: GL tag task formulation.
  have row_12 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_77_line_11: GL tag expansion.
  have row_11 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_12
  -- chapter_77_line_10: GL tag disintegration.
  have row_10 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_11.1
  -- chapter_77_line_9: GL tag expansion.
  have row_9 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_10
  -- chapter_77_line_49: GL tag disintegration.
  have row_49 : (gl_implication17 N succ add) := by
    exact row_9.1.1.1.1.1.2
  -- chapter_77_line_48: GL tag expansion.
  have row_48 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication17] using row_49
  -- chapter_77_line_18: GL tag disintegration.
  have row_18 : (gl_fXY succ N N) := by
    exact row_9.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_77_line_17: GL tag expansion.
  have row_17 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_18
  -- chapter_77_line_33: GL tag disintegration.
  have row_33 : (gl_implication0 succ N) := by
    exact row_17.1.1.1
  -- chapter_77_line_32: GL tag expansion.
  have row_32 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_33
  -- chapter_77_line_31: GL tag implication.
  have row_31 : (N previous) := by
    apply row_32
    exact row_19
  -- chapter_77_line_16: GL tag disintegration.
  have row_16 : (gl_implication1 succ N) := by
    exact row_17.1.1.2
  -- chapter_77_line_15: GL tag expansion.
  have row_15 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using row_16
  -- chapter_77_line_14: GL tag implication.
  have row_14 : (N v1) := by
    apply row_15
    exact row_19
  -- chapter_77_line_13: GL tag disintegration.
  have row_13 : (N zero) := by
    exact row_9.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_77_line_8: GL tag disintegration.
  have row_8 : (gl_fXYZ add N N N) := by
    exact row_9.1.1.1.1.1.1.1.1.2
  -- chapter_77_line_7: GL tag expansion.
  have row_7 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_8
  -- chapter_77_line_24: GL tag disintegration.
  have row_24 : (gl_implication14 N N add) := by
    exact row_7.2
  -- chapter_77_line_23: GL tag expansion.
  have row_23 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (∀ (w3 : α), ((add w1 w2 w3) → (∀ (w4 : α), ((add w1 w2 w4) → (w3 = w4))))))))) := by
    simpa only [gl_implication14] using row_24
  -- chapter_77_line_6: GL tag disintegration.
  have row_6 : (gl_implication13 N N N add) := by
    exact row_7.1.2
  -- chapter_77_line_5: GL tag expansion.
  have row_5 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication13] using row_6
  -- chapter_77_line_38: GL tag implication.
  have row_38 : (gl_existence1 N previous zero add) := by
    apply row_5
    exact row_31
    exact row_13
  -- chapter_77_line_37: GL tag expansion.
  have row_37 : (¬ (∀ (v9 : α), ((N v9) → (¬ (add previous zero v9))))) := by
    simpa only [gl_existence1] using row_38
  have exists_row_37 : ∃ (v9 : α), ((N v9) ∧ (add previous zero v9)) := existsAndOfNotForallImpNot row_37
  obtain ⟨v9, witness_row_37⟩ := exists_row_37
  -- chapter_77_line_41: GL tag disintegration.
  have row_41 : (N v9) := by
    exact witness_row_37.1
  -- chapter_77_line_36: GL tag disintegration.
  have row_36 : (add previous zero v9) := by
    exact witness_row_37.2
  -- chapter_77_line_34: GL tag implication.
  have row_34 : (previous = v9) := by
    apply row_35
    exact row_36
    exact row_31
  -- chapter_77_line_40: GL tag symmetry of equality.
  have row_40 : (v9 = previous) := by
    exact Eq.symm row_34
  -- chapter_77_line_39: GL tag equality1.
  have row_39 : (add v9 zero previous) := by
    have equality_source := row_36
    have equality_step_1 := row_40
    cases equality_step_1
    have equality_step_2 := row_34
    cases equality_step_2
    exact equality_source
  -- chapter_77_line_30: GL tag implication.
  have row_30 : (gl_existence1 N zero previous add) := by
    apply row_5
    exact row_13
    exact row_31
  -- chapter_77_line_29: GL tag expansion.
  have row_29 : (¬ (∀ (v6 : α), ((N v6) → (¬ (add zero previous v6))))) := by
    simpa only [gl_existence1] using row_30
  have exists_row_29 : ∃ (v6 : α), ((N v6) ∧ (add zero previous v6)) := existsAndOfNotForallImpNot row_29
  obtain ⟨v6, witness_row_29⟩ := exists_row_29
  -- chapter_77_line_28: GL tag disintegration.
  have row_28 : (add zero previous v6) := by
    exact witness_row_29.2
  -- chapter_77_line_47: GL tag implication.
  have row_47 : (succ v6 v2) := by
    apply row_48
    exact row_31
    exact row_19
    exact row_28
    exact row_50
  -- chapter_77_line_26: GL tag implication.
  have row_26 : (add previous zero v6) := by
    apply row_27
    exact row_28
  -- chapter_77_line_44: GL tag implication.
  have row_44 : (previous = v6) := by
    apply row_35
    exact row_26
    exact row_31
  -- chapter_77_line_43: GL tag equality1.
  have row_43 : (succ v6 v1) := by
    have equality_source := row_19
    have equality_step_1 := row_44
    cases equality_step_1
    exact equality_source
  -- chapter_77_line_25: GL tag equality1.
  have row_25 : (add v9 zero v6) := by
    have equality_source := row_26
    have equality_step_1 := row_34
    cases equality_step_1
    exact equality_source
  -- chapter_77_line_22: GL tag implication.
  have row_22 : (v6 = previous) := by
    apply row_23
    exact row_41
    exact row_13
    exact row_25
    exact row_39
  -- chapter_77_line_46: GL tag equality1.
  have row_46 : (succ previous v2) := by
    have equality_source := row_47
    have equality_step_1 := row_22
    cases equality_step_1
    exact equality_source
  -- chapter_77_line_4: GL tag implication.
  have row_4 : (gl_existence1 N v1 zero add) := by
    apply row_5
    exact row_14
    exact row_13
  -- chapter_77_line_3: GL tag expansion.
  have row_3 : (¬ (∀ (v3 : α), ((N v3) → (¬ (add v1 zero v3))))) := by
    simpa only [gl_existence1] using row_4
  have exists_row_3 : ∃ (v3 : α), ((N v3) ∧ (add v1 zero v3)) := existsAndOfNotForallImpNot row_3
  obtain ⟨v3, witness_row_3⟩ := exists_row_3
  -- chapter_77_line_2: GL tag disintegration.
  have row_2 : (add v1 zero v3) := by
    exact witness_row_3.2
  -- chapter_77_line_45: GL tag implication.
  have row_45 : (v1 = v3) := by
    apply row_35
    exact row_2
    exact row_14
  -- chapter_77_line_42: GL tag equality1.
  have row_42 : (succ v6 v3) := by
    have equality_source := row_43
    have equality_step_1 := row_45
    cases equality_step_1
    exact equality_source
  -- chapter_77_line_20: GL tag implication.
  have row_20 : (v3 = v2) := by
    apply row_21
    exact row_42
    exact row_46
    exact row_22
  -- chapter_77_line_1: GL tag equality1.
  have row_1 : (add v1 zero v2) := by
    have equality_source := row_2
    have equality_step_1 := row_20
    cases equality_step_1
    exact equality_source
  exact row_1

theorem peano_source_035
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((add zero v1 v2) → (add v1 zero v2))) := by
  intro v1
  intro v2
  intro premise_1
  have inductionMember : N v1 := by
    have typingRule := peano_source_035_induction_typing N zero succ add mul one anchor relationalInduction
    exact typingRule v1 v2 premise_1
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v2 : α), ((add zero zero v2) → (add zero zero v2))) := by
    intro v2
    intro base_premise_1
    have zeroRule := peano_source_035_check_zero N zero succ add mul one anchor relationalInduction
    exact zeroRule zero v2 rfl base_premise_1
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v2 : α), ((add zero induction_n v2) → (add induction_n zero v2))) → ∀ induction_m, succ induction_n induction_m → (∀ (v2 : α), ((add zero induction_m v2) → (add induction_m zero v2))) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro v2
    intro step_premise_1
    have step_induction_assumption_1 :
        (∀ (w1 : α), ((add zero induction_n w1) → (add induction_n zero w1))) := by
      intro w1
      intro step_induction_assumption_1_premise_1
      apply induction_hypothesis
      all_goals assumption
    have step_induction_assumption_2 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have stepRule := peano_source_035_check_induction_condition N zero succ add mul one anchor relationalInduction
    exact stepRule induction_n induction_m v2 step_premise_1 step_induction_assumption_1 step_induction_assumption_2
  have inductionProperty : (∀ (v2 : α), ((add zero v1 v2) → (add v1 zero v2))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v2 : α), ((add zero induction_value v2) → (add induction_value zero v2))))
      v1
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v2 premise_1

private theorem peano_source_036_induction_typing
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (assumption_10 : (mul zero v1 v2))
    : (N v1) := by
  -- chapter_78_line_10: GL tag task formulation.
  have row_10 : (mul zero v1 v2) := by
    exact assumption_10
  -- chapter_78_line_9: GL tag task formulation.
  have row_9 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_78_line_8: GL tag expansion.
  have row_8 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_9
  -- chapter_78_line_7: GL tag disintegration.
  have row_7 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_8.1
  -- chapter_78_line_6: GL tag expansion.
  have row_6 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_7
  -- chapter_78_line_5: GL tag disintegration.
  have row_5 : (gl_fXYZ mul N N N) := by
    exact row_6.1.1.1.2
  -- chapter_78_line_4: GL tag expansion.
  have row_4 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_5
  -- chapter_78_line_3: GL tag disintegration.
  have row_3 : (gl_implication9 mul N) := by
    exact row_4.1.1.1.2
  -- chapter_78_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication9] using row_3
  -- chapter_78_line_1: GL tag implication.
  have row_1 : (N v1) := by
    apply row_2
    exact row_10
  exact row_1

private theorem peano_source_036_check_zero
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (assumption_4 : (v1 = zero))
    (assumption_3 : (mul zero v1 v2))
    : (mul v1 zero v2) := by
  -- chapter_79_line_4: GL tag recursion.
  have row_4 : (v1 = zero) := by
    exact assumption_4
  -- chapter_79_line_5: GL tag symmetry of equality.
  have row_5 : (zero = v1) := by
    exact Eq.symm row_4
  -- chapter_79_line_3: GL tag task formulation.
  have row_3 : (mul zero v1 v2) := by
    exact assumption_3
  -- chapter_79_line_2: GL tag equality1.
  have row_2 : (mul zero zero v2) := by
    have equality_source := row_3
    have equality_step_1 := row_4
    cases equality_step_1
    exact equality_source
  -- chapter_79_line_1: GL tag equality1.
  have row_1 : (mul v1 zero v2) := by
    have equality_source := row_2
    have equality_step_1 := row_5
    cases equality_step_1
    exact equality_source
  exact row_1

private theorem peano_source_036_check_induction_condition
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (assumption_47 : (mul zero v1 v2))
    (assumption_38 : (∀ (w1 : α), ((mul zero previous w1) → (mul previous zero w1))))
    (assumption_19 : (succ previous v1))
    : (mul v1 zero v2) := by
  -- chapter_80_line_47: GL tag task formulation.
  have row_47 : (mul zero v1 v2) := by
    exact assumption_47
  -- chapter_80_line_43: GL tag theorem.
  have row_43 := peano_source_039 N zero succ add mul one anchor relationalInduction
  -- chapter_80_line_38: GL tag recursion.
  have row_38 : (∀ (w1 : α), ((mul zero previous w1) → (mul previous zero w1))) := by
    exact assumption_38
  -- chapter_80_line_36: GL tag theorem.
  have row_36 := peano_source_041 N zero succ add mul one anchor relationalInduction
  -- chapter_80_line_19: GL tag recursion.
  have row_19 : (succ previous v1) := by
    exact assumption_19
  -- chapter_80_line_12: GL tag task formulation.
  have row_12 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_80_line_11: GL tag expansion.
  have row_11 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_12
  -- chapter_80_line_10: GL tag disintegration.
  have row_10 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_11.1
  -- chapter_80_line_9: GL tag expansion.
  have row_9 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_10
  -- chapter_80_line_46: GL tag disintegration.
  have row_46 : (gl_implication21 N succ mul add) := by
    exact row_9.2
  -- chapter_80_line_45: GL tag expansion.
  have row_45 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((mul w3 w1 w4) → (∀ (w5 : α), ((mul w3 w2 w5) → (add w4 w3 w5))))))))) := by
    simpa only [gl_implication21] using row_46
  -- chapter_80_line_34: GL tag disintegration.
  have row_34 : (gl_implication19 N zero mul) := by
    exact row_9.1.1.2
  -- chapter_80_line_33: GL tag expansion.
  have row_33 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((mul w1 zero w2) → (w2 = zero))))) := by
    simpa only [gl_implication19] using row_34
  -- chapter_80_line_18: GL tag disintegration.
  have row_18 : (gl_fXY succ N N) := by
    exact row_9.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_80_line_17: GL tag expansion.
  have row_17 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_18
  -- chapter_80_line_30: GL tag disintegration.
  have row_30 : (gl_implication0 succ N) := by
    exact row_17.1.1.1
  -- chapter_80_line_29: GL tag expansion.
  have row_29 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_30
  -- chapter_80_line_28: GL tag implication.
  have row_28 : (N previous) := by
    apply row_29
    exact row_19
  -- chapter_80_line_16: GL tag disintegration.
  have row_16 : (gl_implication1 succ N) := by
    exact row_17.1.1.2
  -- chapter_80_line_15: GL tag expansion.
  have row_15 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using row_16
  -- chapter_80_line_14: GL tag implication.
  have row_14 : (N v1) := by
    apply row_15
    exact row_19
  -- chapter_80_line_13: GL tag disintegration.
  have row_13 : (N zero) := by
    exact row_9.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_80_line_8: GL tag disintegration.
  have row_8 : (gl_fXYZ mul N N N) := by
    exact row_9.1.1.1.2
  -- chapter_80_line_7: GL tag expansion.
  have row_7 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_8
  -- chapter_80_line_23: GL tag disintegration.
  have row_23 : (gl_implication14 N N mul) := by
    exact row_7.2
  -- chapter_80_line_22: GL tag expansion.
  have row_22 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (∀ (w3 : α), ((mul w1 w2 w3) → (∀ (w4 : α), ((mul w1 w2 w4) → (w3 = w4))))))))) := by
    simpa only [gl_implication14] using row_23
  -- chapter_80_line_6: GL tag disintegration.
  have row_6 : (gl_implication13 N N N mul) := by
    exact row_7.1.2
  -- chapter_80_line_5: GL tag expansion.
  have row_5 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication13] using row_6
  -- chapter_80_line_41: GL tag implication.
  have row_41 : (gl_existence1 N zero previous mul) := by
    apply row_5
    exact row_13
    exact row_28
  -- chapter_80_line_40: GL tag expansion.
  have row_40 : (¬ (∀ (v6 : α), ((N v6) → (¬ (mul zero previous v6))))) := by
    simpa only [gl_existence1] using row_41
  have exists_row_40 : ∃ (v6 : α), ((N v6) ∧ (mul zero previous v6)) := existsAndOfNotForallImpNot row_40
  obtain ⟨v6, witness_row_40⟩ := exists_row_40
  -- chapter_80_line_48: GL tag disintegration.
  have row_48 : (N v6) := by
    exact witness_row_40.1
  -- chapter_80_line_39: GL tag disintegration.
  have row_39 : (mul zero previous v6) := by
    exact witness_row_40.2
  -- chapter_80_line_44: GL tag implication.
  have row_44 : (add v6 zero v2) := by
    apply row_45
    exact row_28
    exact row_19
    exact row_39
    exact row_47
  -- chapter_80_line_42: GL tag implication.
  have row_42 : (v6 = v2) := by
    apply row_43
    exact row_44
    exact row_48
  -- chapter_80_line_37: GL tag implication.
  have row_37 : (mul previous zero v6) := by
    apply row_38
    exact row_39
  -- chapter_80_line_27: GL tag implication.
  have row_27 : (gl_existence1 N previous zero mul) := by
    apply row_5
    exact row_28
    exact row_13
  -- chapter_80_line_26: GL tag expansion.
  have row_26 : (¬ (∀ (v9 : α), ((N v9) → (¬ (mul previous zero v9))))) := by
    simpa only [gl_existence1] using row_27
  have exists_row_26 : ∃ (v9 : α), ((N v9) ∧ (mul previous zero v9)) := existsAndOfNotForallImpNot row_26
  obtain ⟨v9, witness_row_26⟩ := exists_row_26
  -- chapter_80_line_25: GL tag disintegration.
  have row_25 : (mul previous zero v9) := by
    exact witness_row_26.2
  -- chapter_80_line_32: GL tag implication.
  have row_32 : (v9 = zero) := by
    apply row_33
    exact row_28
    exact row_25
  -- chapter_80_line_4: GL tag implication.
  have row_4 : (gl_existence1 N v1 zero mul) := by
    apply row_5
    exact row_14
    exact row_13
  -- chapter_80_line_3: GL tag expansion.
  have row_3 : (¬ (∀ (v3 : α), ((N v3) → (¬ (mul v1 zero v3))))) := by
    simpa only [gl_existence1] using row_4
  have exists_row_3 : ∃ (v3 : α), ((N v3) ∧ (mul v1 zero v3)) := existsAndOfNotForallImpNot row_3
  obtain ⟨v3, witness_row_3⟩ := exists_row_3
  -- chapter_80_line_2: GL tag disintegration.
  have row_2 : (mul v1 zero v3) := by
    exact witness_row_3.2
  -- chapter_80_line_35: GL tag implication.
  have row_35 : (zero = v3) := by
    apply row_36
    exact row_2
    exact row_14
  -- chapter_80_line_31: GL tag equality2.
  have row_31 : (v9 = v3) := by
    exact Eq.trans row_32 row_35
  -- chapter_80_line_24: GL tag equality1.
  have row_24 : (mul previous zero v3) := by
    have equality_source := row_25
    have equality_step_1 := row_31
    cases equality_step_1
    exact equality_source
  -- chapter_80_line_21: GL tag implication.
  have row_21 : (v3 = v6) := by
    apply row_22
    exact row_28
    exact row_13
    exact row_24
    exact row_37
  -- chapter_80_line_20: GL tag equality2.
  have row_20 : (v3 = v2) := by
    exact Eq.trans row_21 row_42
  -- chapter_80_line_1: GL tag equality1.
  have row_1 : (mul v1 zero v2) := by
    have equality_source := row_2
    have equality_step_1 := row_20
    cases equality_step_1
    exact equality_source
  exact row_1

theorem peano_source_036
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((mul zero v1 v2) → (mul v1 zero v2))) := by
  intro v1
  intro v2
  intro premise_1
  have inductionMember : N v1 := by
    have typingRule := peano_source_036_induction_typing N zero succ add mul one anchor relationalInduction
    exact typingRule v1 v2 premise_1
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v2 : α), ((mul zero zero v2) → (mul zero zero v2))) := by
    intro v2
    intro base_premise_1
    have zeroRule := peano_source_036_check_zero N zero succ add mul one anchor relationalInduction
    exact zeroRule zero v2 rfl base_premise_1
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v2 : α), ((mul zero induction_n v2) → (mul induction_n zero v2))) → ∀ induction_m, succ induction_n induction_m → (∀ (v2 : α), ((mul zero induction_m v2) → (mul induction_m zero v2))) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro v2
    intro step_premise_1
    have step_induction_assumption_1 :
        (∀ (w1 : α), ((mul zero induction_n w1) → (mul induction_n zero w1))) := by
      intro w1
      intro step_induction_assumption_1_premise_1
      apply induction_hypothesis
      all_goals assumption
    have step_induction_assumption_2 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have stepRule := peano_source_036_check_induction_condition N zero succ add mul one anchor relationalInduction
    exact stepRule induction_n induction_m v2 step_premise_1 step_induction_assumption_1 step_induction_assumption_2
  have inductionProperty : (∀ (v2 : α), ((mul zero v1 v2) → (mul v1 zero v2))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v2 : α), ((mul zero induction_value v2) → (mul induction_value zero v2))))
      v1
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v2 premise_1

theorem peano_source_037
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((add one v1 v2) → (add v1 one v2))) := by
  intro v1
  intro v2
  intro premise_1
  -- chapter_81_line_36: GL tag theorem.
  have row_36 := peano_source_033 N zero succ add mul one anchor relationalInduction
  -- chapter_81_line_25: GL tag theorem.
  have row_25 := peano_source_039 N zero succ add mul one anchor relationalInduction
  -- chapter_81_line_23: GL tag task formulation.
  have row_23 : (add one v1 v2) := by
    exact premise_1
  -- chapter_81_line_8: GL tag task formulation.
  have row_8 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_81_line_7: GL tag expansion.
  have row_7 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_8
  -- chapter_81_line_9: GL tag disintegration.
  have row_9 : (succ zero one) := by
    exact row_7.2
  -- chapter_81_line_6: GL tag disintegration.
  have row_6 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_7.1
  -- chapter_81_line_5: GL tag expansion.
  have row_5 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_6
  -- chapter_81_line_31: GL tag disintegration.
  have row_31 : (N zero) := by
    exact row_5.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_81_line_22: GL tag disintegration.
  have row_22 : (gl_fXYZ add N N N) := by
    exact row_5.1.1.1.1.1.1.1.1.2
  -- chapter_81_line_21: GL tag expansion.
  have row_21 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_22
  -- chapter_81_line_34: GL tag disintegration.
  have row_34 : (gl_implication14 N N add) := by
    exact row_21.2
  -- chapter_81_line_33: GL tag expansion.
  have row_33 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (∀ (w3 : α), ((add w1 w2 w3) → (∀ (w4 : α), ((add w1 w2 w4) → (w3 = w4))))))))) := by
    simpa only [gl_implication14] using row_34
  -- chapter_81_line_30: GL tag disintegration.
  have row_30 : (gl_implication13 N N N add) := by
    exact row_21.1.2
  -- chapter_81_line_29: GL tag expansion.
  have row_29 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication13] using row_30
  -- chapter_81_line_20: GL tag disintegration.
  have row_20 : (gl_implication9 add N) := by
    exact row_21.1.1.1.2
  -- chapter_81_line_19: GL tag expansion.
  have row_19 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication9] using row_20
  -- chapter_81_line_18: GL tag implication.
  have row_18 : (N v1) := by
    apply row_19
    exact row_23
  -- chapter_81_line_28: GL tag implication.
  have row_28 : (gl_existence1 N v1 zero add) := by
    apply row_29
    exact row_18
    exact row_31
  -- chapter_81_line_27: GL tag expansion.
  have row_27 : (¬ (∀ (v9 : α), ((N v9) → (¬ (add v1 zero v9))))) := by
    simpa only [gl_existence1] using row_28
  have exists_row_27 : ∃ (v9 : α), ((N v9) ∧ (add v1 zero v9)) := existsAndOfNotForallImpNot row_27
  obtain ⟨v9, witness_row_27⟩ := exists_row_27
  -- chapter_81_line_26: GL tag disintegration.
  have row_26 : (add v1 zero v9) := by
    exact witness_row_27.2
  -- chapter_81_line_24: GL tag implication.
  have row_24 : (v1 = v9) := by
    apply row_25
    exact row_26
    exact row_18
  -- chapter_81_line_17: GL tag disintegration.
  have row_17 : (gl_fXY succ N N) := by
    exact row_5.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_81_line_16: GL tag expansion.
  have row_16 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_17
  -- chapter_81_line_39: GL tag disintegration.
  have row_39 : (gl_implication1 succ N) := by
    exact row_16.1.1.2
  -- chapter_81_line_38: GL tag expansion.
  have row_38 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using row_39
  -- chapter_81_line_37: GL tag implication.
  have row_37 : (N one) := by
    apply row_38
    exact row_9
  -- chapter_81_line_15: GL tag disintegration.
  have row_15 : (gl_implication4 N N succ) := by
    exact row_16.1.2
  -- chapter_81_line_14: GL tag expansion.
  have row_14 : (∀ (w1 : α), ((N w1) → (gl_existence0 N w1 succ))) := by
    simpa only [gl_implication4] using row_15
  -- chapter_81_line_13: GL tag implication.
  have row_13 : (gl_existence0 N v1 succ) := by
    apply row_14
    exact row_18
  -- chapter_81_line_12: GL tag expansion.
  have row_12 : (¬ (∀ (v3 : α), ((N v3) → (¬ (succ v1 v3))))) := by
    simpa only [gl_existence0] using row_13
  have exists_row_12 : ∃ (v3 : α), ((N v3) ∧ (succ v1 v3)) := existsAndOfNotForallImpNot row_12
  obtain ⟨v3, witness_row_12⟩ := exists_row_12
  -- chapter_81_line_11: GL tag disintegration.
  have row_11 : (succ v1 v3) := by
    exact witness_row_12.2
  -- chapter_81_line_35: GL tag implication.
  have row_35 : (add one v1 v3) := by
    apply row_36
    exact row_11
  -- chapter_81_line_32: GL tag implication.
  have row_32 : (v3 = v2) := by
    apply row_33
    exact row_37
    exact row_18
    exact row_35
    exact row_23
  -- chapter_81_line_10: GL tag equality1.
  have row_10 : (succ v9 v3) := by
    have equality_source := row_11
    have equality_step_1 := row_24
    cases equality_step_1
    exact equality_source
  -- chapter_81_line_4: GL tag disintegration.
  have row_4 : (gl_implication18 N succ add) := by
    exact row_5.1.1.1.1.2
  -- chapter_81_line_3: GL tag expansion.
  have row_3 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((succ w4 w5) → (add w3 w2 w5))))))))) := by
    simpa only [gl_implication18] using row_4
  -- chapter_81_line_2: GL tag implication.
  have row_2 : (add v1 one v3) := by
    apply row_3
    exact row_31
    exact row_9
    exact row_26
    exact row_10
  -- chapter_81_line_1: GL tag equality1.
  have row_1 : (add v1 one v2) := by
    have equality_source := row_2
    have equality_step_1 := row_32
    cases equality_step_1
    exact equality_source
  exact row_1

private theorem peano_source_040_induction_typing
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (assumption_10 : (add v1 zero v2))
    : (N v1) := by
  -- chapter_86_line_10: GL tag task formulation.
  have row_10 : (add v1 zero v2) := by
    exact assumption_10
  -- chapter_86_line_9: GL tag task formulation.
  have row_9 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_86_line_8: GL tag expansion.
  have row_8 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_9
  -- chapter_86_line_7: GL tag disintegration.
  have row_7 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_8.1
  -- chapter_86_line_6: GL tag expansion.
  have row_6 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_7
  -- chapter_86_line_5: GL tag disintegration.
  have row_5 : (gl_fXYZ add N N N) := by
    exact row_6.1.1.1.1.1.1.1.1.2
  -- chapter_86_line_4: GL tag expansion.
  have row_4 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_5
  -- chapter_86_line_3: GL tag disintegration.
  have row_3 : (gl_implication8 add N) := by
    exact row_4.1.1.1.1
  -- chapter_86_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_3
  -- chapter_86_line_1: GL tag implication.
  have row_1 : (N v1) := by
    apply row_2
    exact row_10
  exact row_1

private theorem peano_source_040_check_zero
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (assumption_18 : (v1 = zero))
    (assumption_4 : (add v1 zero v2))
    : (add zero v1 v2) := by
  -- chapter_87_line_18: GL tag recursion.
  have row_18 : (v1 = zero) := by
    exact assumption_18
  -- chapter_87_line_19: GL tag symmetry of equality.
  have row_19 : (zero = v1) := by
    exact Eq.symm row_18
  -- chapter_87_line_12: GL tag task formulation.
  have row_12 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_87_line_11: GL tag expansion.
  have row_11 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_12
  -- chapter_87_line_10: GL tag disintegration.
  have row_10 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_11.1
  -- chapter_87_line_9: GL tag expansion.
  have row_9 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_10
  -- chapter_87_line_17: GL tag disintegration.
  have row_17 : (gl_fXYZ add N N N) := by
    exact row_9.1.1.1.1.1.1.1.1.2
  -- chapter_87_line_16: GL tag expansion.
  have row_16 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_17
  -- chapter_87_line_15: GL tag disintegration.
  have row_15 : (gl_implication8 add N) := by
    exact row_16.1.1.1.1
  -- chapter_87_line_14: GL tag expansion.
  have row_14 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_15
  -- chapter_87_line_8: GL tag disintegration.
  have row_8 : (gl_implication15 N zero add) := by
    exact row_9.1.1.1.1.1.1.1.2
  -- chapter_87_line_7: GL tag expansion.
  have row_7 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((add w1 zero w2) → (w1 = w2))))) := by
    simpa only [gl_implication15] using row_8
  -- chapter_87_line_4: GL tag task formulation.
  have row_4 : (add v1 zero v2) := by
    exact assumption_4
  -- chapter_87_line_13: GL tag implication.
  have row_13 : (N v1) := by
    apply row_14
    exact row_4
  -- chapter_87_line_6: GL tag implication.
  have row_6 : (v1 = v2) := by
    apply row_7
    exact row_13
    exact row_4
  -- chapter_87_line_20: GL tag equality2.
  have row_20 : (zero = v2) := by
    exact Eq.trans row_19 row_6
  -- chapter_87_line_5: GL tag symmetry of equality.
  have row_5 : (v2 = v1) := by
    exact Eq.symm row_6
  -- chapter_87_line_3: GL tag equality1.
  have row_3 : (add v1 zero v1) := by
    have equality_source := row_4
    have equality_step_1 := row_5
    cases equality_step_1
    exact equality_source
  -- chapter_87_line_2: GL tag equality1.
  have row_2 : (add zero zero zero) := by
    have equality_source := row_3
    have equality_step_1 := row_18
    cases equality_step_1
    exact equality_source
  -- chapter_87_line_1: GL tag equality1.
  have row_1 : (add zero v1 v2) := by
    have equality_source := row_2
    have equality_step_1 := row_19
    cases equality_step_1
    have equality_step_2 := row_20
    cases equality_step_2
    exact equality_source
  exact row_1

private theorem peano_source_040_check_induction_condition
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (assumption_36 : (∀ (w1 : α), ((add previous zero w1) → (add zero previous w1))))
    (assumption_29 : (add v1 zero v2))
    (assumption_6 : (succ previous v1))
    : (add zero v1 v2) := by
  -- chapter_88_line_36: GL tag recursion.
  have row_36 : (∀ (w1 : α), ((add previous zero w1) → (add zero previous w1))) := by
    exact assumption_36
  -- chapter_88_line_29: GL tag task formulation.
  have row_29 : (add v1 zero v2) := by
    exact assumption_29
  -- chapter_88_line_8: GL tag theorem.
  have row_8 := peano_source_039 N zero succ add mul one anchor relationalInduction
  -- chapter_88_line_6: GL tag recursion.
  have row_6 : (succ previous v1) := by
    exact assumption_6
  -- chapter_88_line_4: GL tag task formulation.
  have row_4 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_88_line_18: GL tag expansion.
  have row_18 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_4
  -- chapter_88_line_17: GL tag disintegration.
  have row_17 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_18.1
  -- chapter_88_line_16: GL tag expansion.
  have row_16 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_17
  -- chapter_88_line_28: GL tag disintegration.
  have row_28 : (gl_implication15 N zero add) := by
    exact row_16.1.1.1.1.1.1.1.2
  -- chapter_88_line_27: GL tag expansion.
  have row_27 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((add w1 zero w2) → (w1 = w2))))) := by
    simpa only [gl_implication15] using row_28
  -- chapter_88_line_24: GL tag disintegration.
  have row_24 : (gl_fXY succ N N) := by
    exact row_16.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_88_line_23: GL tag expansion.
  have row_23 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_24
  -- chapter_88_line_32: GL tag disintegration.
  have row_32 : (gl_implication1 succ N) := by
    exact row_23.1.1.2
  -- chapter_88_line_31: GL tag expansion.
  have row_31 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using row_32
  -- chapter_88_line_30: GL tag implication.
  have row_30 : (N v1) := by
    apply row_31
    exact row_6
  -- chapter_88_line_26: GL tag implication.
  have row_26 : (v1 = v2) := by
    apply row_27
    exact row_30
    exact row_29
  -- chapter_88_line_25: GL tag equality1.
  have row_25 : (succ previous v2) := by
    have equality_source := row_6
    have equality_step_1 := row_26
    cases equality_step_1
    exact equality_source
  -- chapter_88_line_22: GL tag disintegration.
  have row_22 : (gl_implication0 succ N) := by
    exact row_23.1.1.1
  -- chapter_88_line_21: GL tag expansion.
  have row_21 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_22
  -- chapter_88_line_20: GL tag implication.
  have row_20 : (N previous) := by
    apply row_21
    exact row_6
  -- chapter_88_line_19: GL tag disintegration.
  have row_19 : (N zero) := by
    exact row_16.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_88_line_15: GL tag disintegration.
  have row_15 : (gl_fXYZ add N N N) := by
    exact row_16.1.1.1.1.1.1.1.1.2
  -- chapter_88_line_14: GL tag expansion.
  have row_14 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_15
  -- chapter_88_line_13: GL tag disintegration.
  have row_13 : (gl_implication13 N N N add) := by
    exact row_14.1.2
  -- chapter_88_line_12: GL tag expansion.
  have row_12 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication13] using row_13
  -- chapter_88_line_11: GL tag implication.
  have row_11 : (gl_existence1 N previous zero add) := by
    apply row_12
    exact row_20
    exact row_19
  -- chapter_88_line_10: GL tag expansion.
  have row_10 : (¬ (∀ (v3 : α), ((N v3) → (¬ (add previous zero v3))))) := by
    simpa only [gl_existence1] using row_11
  have exists_row_10 : ∃ (v3 : α), ((N v3) ∧ (add previous zero v3)) := existsAndOfNotForallImpNot row_10
  obtain ⟨v3, witness_row_10⟩ := exists_row_10
  -- chapter_88_line_9: GL tag disintegration.
  have row_9 : (add previous zero v3) := by
    exact witness_row_10.2
  -- chapter_88_line_35: GL tag implication.
  have row_35 : (add zero previous v3) := by
    apply row_36
    exact row_9
  -- chapter_88_line_7: GL tag implication.
  have row_7 : (previous = v3) := by
    apply row_8
    exact row_9
    exact row_20
  -- chapter_88_line_37: GL tag symmetry of equality.
  have row_37 : (v3 = previous) := by
    exact Eq.symm row_7
  -- chapter_88_line_34: GL tag equality1.
  have row_34 : (add zero previous previous) := by
    have equality_source := row_35
    have equality_step_1 := row_37
    cases equality_step_1
    exact equality_source
  -- chapter_88_line_33: GL tag equality1.
  have row_33 : (add zero v3 previous) := by
    have equality_source := row_34
    have equality_step_1 := row_7
    cases equality_step_1
    exact equality_source
  -- chapter_88_line_5: GL tag equality1.
  have row_5 : (succ v3 v1) := by
    have equality_source := row_6
    have equality_step_1 := row_7
    cases equality_step_1
    exact equality_source
  -- chapter_88_line_3: GL tag anchor handling.
  have row_3 : (gl_AnchorPeano N zero succ add mul one) := by
    exact row_4
  -- chapter_88_line_2: GL tag theorem.
  have row_2 := peano_source_011 N zero succ add mul one anchor relationalInduction
  -- chapter_88_line_1: GL tag implication.
  have row_1 : (add zero v1 v2) := by
    apply row_2
    exact row_33
    exact row_5
    exact row_25
  exact row_1

theorem peano_source_040
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((add v1 zero v2) → (add zero v1 v2))) := by
  intro v1
  intro v2
  intro premise_1
  have inductionMember : N v1 := by
    have typingRule := peano_source_040_induction_typing N zero succ add mul one anchor relationalInduction
    exact typingRule v1 v2 premise_1
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v2 : α), ((add zero zero v2) → (add zero zero v2))) := by
    intro v2
    intro base_premise_1
    have zeroRule := peano_source_040_check_zero N zero succ add mul one anchor relationalInduction
    exact zeroRule zero v2 rfl base_premise_1
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v2 : α), ((add induction_n zero v2) → (add zero induction_n v2))) → ∀ induction_m, succ induction_n induction_m → (∀ (v2 : α), ((add induction_m zero v2) → (add zero induction_m v2))) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro v2
    intro step_premise_1
    have step_induction_assumption_1 :
        (∀ (w1 : α), ((add induction_n zero w1) → (add zero induction_n w1))) := by
      intro w1
      intro step_induction_assumption_1_premise_1
      apply induction_hypothesis
      all_goals assumption
    have step_induction_assumption_2 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have stepRule := peano_source_040_check_induction_condition N zero succ add mul one anchor relationalInduction
    exact stepRule induction_n induction_m v2 step_induction_assumption_1 step_premise_1 step_induction_assumption_2
  have inductionProperty : (∀ (v2 : α), ((add v1 zero v2) → (add zero v1 v2))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v2 : α), ((add induction_value zero v2) → (add zero induction_value v2))))
      v1
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v2 premise_1

private theorem peano_source_042_induction_typing
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (assumption_10 : (mul v1 zero v2))
    : (N v1) := by
  -- chapter_90_line_10: GL tag task formulation.
  have row_10 : (mul v1 zero v2) := by
    exact assumption_10
  -- chapter_90_line_9: GL tag task formulation.
  have row_9 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_90_line_8: GL tag expansion.
  have row_8 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_9
  -- chapter_90_line_7: GL tag disintegration.
  have row_7 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_8.1
  -- chapter_90_line_6: GL tag expansion.
  have row_6 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_7
  -- chapter_90_line_5: GL tag disintegration.
  have row_5 : (gl_fXYZ mul N N N) := by
    exact row_6.1.1.1.2
  -- chapter_90_line_4: GL tag expansion.
  have row_4 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_5
  -- chapter_90_line_3: GL tag disintegration.
  have row_3 : (gl_implication8 mul N) := by
    exact row_4.1.1.1.1
  -- chapter_90_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_3
  -- chapter_90_line_1: GL tag implication.
  have row_1 : (N v1) := by
    apply row_2
    exact row_10
  exact row_1

private theorem peano_source_042_check_zero
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (assumption_17 : (v1 = zero))
    (assumption_4 : (mul v1 zero v2))
    : (mul zero v1 v2) := by
  -- chapter_91_line_17: GL tag recursion.
  have row_17 : (v1 = zero) := by
    exact assumption_17
  -- chapter_91_line_18: GL tag symmetry of equality.
  have row_18 : (zero = v1) := by
    exact Eq.symm row_17
  -- chapter_91_line_11: GL tag task formulation.
  have row_11 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_91_line_10: GL tag expansion.
  have row_10 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_11
  -- chapter_91_line_9: GL tag disintegration.
  have row_9 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_10.1
  -- chapter_91_line_8: GL tag expansion.
  have row_8 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_9
  -- chapter_91_line_16: GL tag disintegration.
  have row_16 : (gl_fXYZ mul N N N) := by
    exact row_8.1.1.1.2
  -- chapter_91_line_15: GL tag expansion.
  have row_15 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_16
  -- chapter_91_line_14: GL tag disintegration.
  have row_14 : (gl_implication8 mul N) := by
    exact row_15.1.1.1.1
  -- chapter_91_line_13: GL tag expansion.
  have row_13 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_14
  -- chapter_91_line_7: GL tag disintegration.
  have row_7 : (gl_implication19 N zero mul) := by
    exact row_8.1.1.2
  -- chapter_91_line_6: GL tag expansion.
  have row_6 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((mul w1 zero w2) → (w2 = zero))))) := by
    simpa only [gl_implication19] using row_7
  -- chapter_91_line_4: GL tag task formulation.
  have row_4 : (mul v1 zero v2) := by
    exact assumption_4
  -- chapter_91_line_12: GL tag implication.
  have row_12 : (N v1) := by
    apply row_13
    exact row_4
  -- chapter_91_line_5: GL tag implication.
  have row_5 : (v2 = zero) := by
    apply row_6
    exact row_12
    exact row_4
  -- chapter_91_line_19: GL tag symmetry of equality.
  have row_19 : (zero = v2) := by
    exact Eq.symm row_5
  -- chapter_91_line_3: GL tag equality1.
  have row_3 : (mul v1 zero zero) := by
    have equality_source := row_4
    have equality_step_1 := row_5
    cases equality_step_1
    exact equality_source
  -- chapter_91_line_2: GL tag equality1.
  have row_2 : (mul zero zero zero) := by
    have equality_source := row_3
    have equality_step_1 := row_17
    cases equality_step_1
    exact equality_source
  -- chapter_91_line_1: GL tag equality1.
  have row_1 : (mul zero v1 v2) := by
    have equality_source := row_2
    have equality_step_1 := row_18
    cases equality_step_1
    have equality_step_2 := row_19
    cases equality_step_2
    exact equality_source
  exact row_1

private theorem peano_source_042_check_induction_condition
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (assumption_33 : (mul v1 zero v2))
    (assumption_7 : (∀ (w1 : α), ((mul previous zero w1) → (mul zero previous w1))))
    (assumption_5 : (succ previous v1))
    : (mul zero v1 v2) := by
  -- chapter_92_line_41: GL tag theorem.
  have row_41 := peano_source_041 N zero succ add mul one anchor relationalInduction
  -- chapter_92_line_33: GL tag task formulation.
  have row_33 : (mul v1 zero v2) := by
    exact assumption_33
  -- chapter_92_line_7: GL tag recursion.
  have row_7 : (∀ (w1 : α), ((mul previous zero w1) → (mul zero previous w1))) := by
    exact assumption_7
  -- chapter_92_line_5: GL tag recursion.
  have row_5 : (succ previous v1) := by
    exact assumption_5
  -- chapter_92_line_4: GL tag task formulation.
  have row_4 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_92_line_17: GL tag expansion.
  have row_17 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_4
  -- chapter_92_line_16: GL tag disintegration.
  have row_16 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_17.1
  -- chapter_92_line_15: GL tag expansion.
  have row_15 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_16
  -- chapter_92_line_32: GL tag disintegration.
  have row_32 : (gl_implication19 N zero mul) := by
    exact row_15.1.1.2
  -- chapter_92_line_31: GL tag expansion.
  have row_31 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((mul w1 zero w2) → (w2 = zero))))) := by
    simpa only [gl_implication19] using row_32
  -- chapter_92_line_28: GL tag disintegration.
  have row_28 : (gl_implication16 N zero add) := by
    exact row_15.1.1.1.1.1.1.2
  -- chapter_92_line_27: GL tag expansion.
  have row_27 : (∀ (w1 : α) (w2 : α), ((w1 = w2) → ((N w1) → ((N w2) → (add w1 zero w2))))) := by
    simpa only [gl_implication16] using row_28
  -- chapter_92_line_23: GL tag disintegration.
  have row_23 : (gl_fXY succ N N) := by
    exact row_15.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_92_line_22: GL tag expansion.
  have row_22 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_23
  -- chapter_92_line_36: GL tag disintegration.
  have row_36 : (gl_implication1 succ N) := by
    exact row_22.1.1.2
  -- chapter_92_line_35: GL tag expansion.
  have row_35 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using row_36
  -- chapter_92_line_34: GL tag implication.
  have row_34 : (N v1) := by
    apply row_35
    exact row_5
  -- chapter_92_line_30: GL tag implication.
  have row_30 : (v2 = zero) := by
    apply row_31
    exact row_34
    exact row_33
  -- chapter_92_line_29: GL tag symmetry of equality.
  have row_29 : (zero = v2) := by
    exact Eq.symm row_30
  -- chapter_92_line_21: GL tag disintegration.
  have row_21 : (gl_implication0 succ N) := by
    exact row_22.1.1.1
  -- chapter_92_line_20: GL tag expansion.
  have row_20 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_21
  -- chapter_92_line_19: GL tag implication.
  have row_19 : (N previous) := by
    apply row_20
    exact row_5
  -- chapter_92_line_18: GL tag disintegration.
  have row_18 : (N zero) := by
    exact row_15.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_92_line_14: GL tag disintegration.
  have row_14 : (gl_fXYZ mul N N N) := by
    exact row_15.1.1.1.2
  -- chapter_92_line_13: GL tag expansion.
  have row_13 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_14
  -- chapter_92_line_39: GL tag disintegration.
  have row_39 : (gl_implication10 mul N) := by
    exact row_13.1.1.2
  -- chapter_92_line_38: GL tag expansion.
  have row_38 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication10] using row_39
  -- chapter_92_line_37: GL tag implication.
  have row_37 : (N v2) := by
    apply row_38
    exact row_33
  -- chapter_92_line_26: GL tag implication.
  have row_26 : (add zero zero v2) := by
    apply row_27
    exact row_29
    exact row_18
    exact row_37
  -- chapter_92_line_25: GL tag equality1.
  have row_25 : (add zero zero zero) := by
    have equality_source := row_26
    have equality_step_1 := row_30
    cases equality_step_1
    exact equality_source
  -- chapter_92_line_12: GL tag disintegration.
  have row_12 : (gl_implication13 N N N mul) := by
    exact row_13.1.2
  -- chapter_92_line_11: GL tag expansion.
  have row_11 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication13] using row_12
  -- chapter_92_line_10: GL tag implication.
  have row_10 : (gl_existence1 N previous zero mul) := by
    apply row_11
    exact row_19
    exact row_18
  -- chapter_92_line_9: GL tag expansion.
  have row_9 : (¬ (∀ (v3 : α), ((N v3) → (¬ (mul previous zero v3))))) := by
    simpa only [gl_existence1] using row_10
  have exists_row_9 : ∃ (v3 : α), ((N v3) ∧ (mul previous zero v3)) := existsAndOfNotForallImpNot row_9
  obtain ⟨v3, witness_row_9⟩ := exists_row_9
  -- chapter_92_line_8: GL tag disintegration.
  have row_8 : (mul previous zero v3) := by
    exact witness_row_9.2
  -- chapter_92_line_40: GL tag implication.
  have row_40 : (zero = v3) := by
    apply row_41
    exact row_8
    exact row_19
  -- chapter_92_line_24: GL tag equality1.
  have row_24 : (add v3 zero v2) := by
    have equality_source := row_25
    have equality_step_1 := row_29
    cases equality_step_1
    have equality_step_2 := row_40
    cases equality_step_2
    exact equality_source
  -- chapter_92_line_6: GL tag implication.
  have row_6 : (mul zero previous v3) := by
    apply row_7
    exact row_8
  -- chapter_92_line_3: GL tag anchor handling.
  have row_3 : (gl_AnchorPeano N zero succ add mul one) := by
    exact row_4
  -- chapter_92_line_2: GL tag theorem.
  have row_2 := peano_source_003 N zero succ add mul one anchor relationalInduction
  -- chapter_92_line_1: GL tag implication.
  have row_1 : (mul zero v1 v2) := by
    apply row_2
    exact row_24
    exact row_5
    exact row_6
  exact row_1

theorem peano_source_042
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((mul v1 zero v2) → (mul zero v1 v2))) := by
  intro v1
  intro v2
  intro premise_1
  have inductionMember : N v1 := by
    have typingRule := peano_source_042_induction_typing N zero succ add mul one anchor relationalInduction
    exact typingRule v1 v2 premise_1
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v2 : α), ((mul zero zero v2) → (mul zero zero v2))) := by
    intro v2
    intro base_premise_1
    have zeroRule := peano_source_042_check_zero N zero succ add mul one anchor relationalInduction
    exact zeroRule zero v2 rfl base_premise_1
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v2 : α), ((mul induction_n zero v2) → (mul zero induction_n v2))) → ∀ induction_m, succ induction_n induction_m → (∀ (v2 : α), ((mul induction_m zero v2) → (mul zero induction_m v2))) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro v2
    intro step_premise_1
    have step_induction_assumption_1 :
        (∀ (w1 : α), ((mul induction_n zero w1) → (mul zero induction_n w1))) := by
      intro w1
      intro step_induction_assumption_1_premise_1
      apply induction_hypothesis
      all_goals assumption
    have step_induction_assumption_2 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have stepRule := peano_source_042_check_induction_condition N zero succ add mul one anchor relationalInduction
    exact stepRule induction_n induction_m v2 step_premise_1 step_induction_assumption_1 step_induction_assumption_2
  have inductionProperty : (∀ (v2 : α), ((mul v1 zero v2) → (mul zero v1 v2))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v2 : α), ((mul induction_value zero v2) → (mul zero induction_value v2))))
      v1
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v2 premise_1

theorem peano_source_043
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((add v1 one v2) → (succ v1 v2))) := by
  intro v1
  intro v2
  intro premise_1
  -- chapter_93_line_24: GL tag theorem.
  have row_24 := peano_source_039 N zero succ add mul one anchor relationalInduction
  -- chapter_93_line_21: GL tag task formulation.
  have row_21 : (add v1 one v2) := by
    exact premise_1
  -- chapter_93_line_8: GL tag task formulation.
  have row_8 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_93_line_7: GL tag expansion.
  have row_7 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_8
  -- chapter_93_line_9: GL tag disintegration.
  have row_9 : (succ zero one) := by
    exact row_7.2
  -- chapter_93_line_6: GL tag disintegration.
  have row_6 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_7.1
  -- chapter_93_line_5: GL tag expansion.
  have row_5 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_6
  -- chapter_93_line_17: GL tag disintegration.
  have row_17 : (N zero) := by
    exact row_5.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_93_line_16: GL tag disintegration.
  have row_16 : (gl_fXYZ add N N N) := by
    exact row_5.1.1.1.1.1.1.1.1.2
  -- chapter_93_line_15: GL tag expansion.
  have row_15 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_16
  -- chapter_93_line_20: GL tag disintegration.
  have row_20 : (gl_implication8 add N) := by
    exact row_15.1.1.1.1
  -- chapter_93_line_19: GL tag expansion.
  have row_19 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_20
  -- chapter_93_line_18: GL tag implication.
  have row_18 : (N v1) := by
    apply row_19
    exact row_21
  -- chapter_93_line_14: GL tag disintegration.
  have row_14 : (gl_implication13 N N N add) := by
    exact row_15.1.2
  -- chapter_93_line_13: GL tag expansion.
  have row_13 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication13] using row_14
  -- chapter_93_line_12: GL tag implication.
  have row_12 : (gl_existence1 N v1 zero add) := by
    apply row_13
    exact row_18
    exact row_17
  -- chapter_93_line_11: GL tag expansion.
  have row_11 : (¬ (∀ (v3 : α), ((N v3) → (¬ (add v1 zero v3))))) := by
    simpa only [gl_existence1] using row_12
  have exists_row_11 : ∃ (v3 : α), ((N v3) ∧ (add v1 zero v3)) := existsAndOfNotForallImpNot row_11
  obtain ⟨v3, witness_row_11⟩ := exists_row_11
  -- chapter_93_line_10: GL tag disintegration.
  have row_10 : (add v1 zero v3) := by
    exact witness_row_11.2
  -- chapter_93_line_23: GL tag implication.
  have row_23 : (v1 = v3) := by
    apply row_24
    exact row_10
    exact row_18
  -- chapter_93_line_22: GL tag symmetry of equality.
  have row_22 : (v3 = v1) := by
    exact Eq.symm row_23
  -- chapter_93_line_4: GL tag disintegration.
  have row_4 : (gl_implication17 N succ add) := by
    exact row_5.1.1.1.1.1.2
  -- chapter_93_line_3: GL tag expansion.
  have row_3 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication17] using row_4
  -- chapter_93_line_2: GL tag implication.
  have row_2 : (succ v3 v2) := by
    apply row_3
    exact row_17
    exact row_9
    exact row_10
    exact row_21
  -- chapter_93_line_1: GL tag equality1.
  have row_1 : (succ v1 v2) := by
    have equality_source := row_2
    have equality_step_1 := row_22
    cases equality_step_1
    exact equality_source
  exact row_1

private theorem peano_source_047_induction_typing
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v2 : α)
    (assumption_1 : (N v2))
    : (N v2) := by
  -- chapter_103_line_1: GL tag task formulation.
  have row_1 : (N v2) := by
    exact assumption_1
  exact row_1

private theorem peano_source_047_check_zero
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (assumption_10 : (add v1 v2 zero))
    (assumption_3 : (v2 = zero))
    : (zero = v1) := by
  -- chapter_104_line_10: GL tag task formulation.
  have row_10 : (add v1 v2 zero) := by
    exact assumption_10
  -- chapter_104_line_7: GL tag task formulation.
  have row_7 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_104_line_18: GL tag expansion.
  have row_18 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_7
  -- chapter_104_line_17: GL tag disintegration.
  have row_17 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_18.1
  -- chapter_104_line_16: GL tag expansion.
  have row_16 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_17
  -- chapter_104_line_15: GL tag disintegration.
  have row_15 : (gl_fXYZ add N N N) := by
    exact row_16.1.1.1.1.1.1.1.1.2
  -- chapter_104_line_14: GL tag expansion.
  have row_14 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_15
  -- chapter_104_line_13: GL tag disintegration.
  have row_13 : (gl_implication8 add N) := by
    exact row_14.1.1.1.1
  -- chapter_104_line_12: GL tag expansion.
  have row_12 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_13
  -- chapter_104_line_11: GL tag implication.
  have row_11 : (N v1) := by
    apply row_12
    exact row_10
  -- chapter_104_line_6: GL tag theorem.
  have row_6 := peano_source_039 N zero succ add mul one anchor relationalInduction
  -- chapter_104_line_3: GL tag recursion.
  have row_3 : (v2 = zero) := by
    exact assumption_3
  -- chapter_104_line_9: GL tag equality1.
  have row_9 : (add v1 zero zero) := by
    have equality_source := row_10
    have equality_step_1 := row_3
    cases equality_step_1
    exact equality_source
  -- chapter_104_line_2: GL tag symmetry of equality.
  have row_2 : (zero = v2) := by
    exact Eq.symm row_3
  -- chapter_104_line_8: GL tag equality1.
  have row_8 : (add v1 zero v2) := by
    have equality_source := row_9
    have equality_step_1 := row_2
    cases equality_step_1
    exact equality_source
  -- chapter_104_line_5: GL tag implication.
  have row_5 : (v1 = v2) := by
    apply row_6
    exact row_8
    exact row_11
  -- chapter_104_line_4: GL tag symmetry of equality.
  have row_4 : (v2 = v1) := by
    exact Eq.symm row_5
  -- chapter_104_line_1: GL tag equality2.
  have row_1 : (zero = v1) := by
    exact Eq.trans row_2 row_4
  exact row_1

private theorem peano_source_047_check_induction_condition
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (assumption_25 : (succ previous v2))
    (assumption_19 : (add v1 v2 zero))
    : (zero = v1) := by
  -- chapter_105_line_25: GL tag recursion.
  have row_25 : (succ previous v2) := by
    exact assumption_25
  -- chapter_105_line_19: GL tag task formulation.
  have row_19 : (add v1 v2 zero) := by
    exact assumption_19
  -- chapter_105_line_8: GL tag task formulation.
  have row_8 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_105_line_7: GL tag expansion.
  have row_7 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_8
  -- chapter_105_line_6: GL tag disintegration.
  have row_6 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_7.1
  -- chapter_105_line_5: GL tag expansion.
  have row_5 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_6
  -- chapter_105_line_28: GL tag disintegration.
  have row_28 : (gl_implication17 N succ add) := by
    exact row_5.1.1.1.1.1.2
  -- chapter_105_line_27: GL tag expansion.
  have row_27 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication17] using row_28
  -- chapter_105_line_24: GL tag disintegration.
  have row_24 : (gl_fXY succ N N) := by
    exact row_5.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_105_line_23: GL tag expansion.
  have row_23 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_24
  -- chapter_105_line_22: GL tag disintegration.
  have row_22 : (gl_implication0 succ N) := by
    exact row_23.1.1.1
  -- chapter_105_line_21: GL tag expansion.
  have row_21 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_22
  -- chapter_105_line_20: GL tag implication.
  have row_20 : (N previous) := by
    apply row_21
    exact row_25
  -- chapter_105_line_15: GL tag disintegration.
  have row_15 : (gl_fXYZ add N N N) := by
    exact row_5.1.1.1.1.1.1.1.1.2
  -- chapter_105_line_14: GL tag expansion.
  have row_14 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_15
  -- chapter_105_line_18: GL tag disintegration.
  have row_18 : (gl_implication8 add N) := by
    exact row_14.1.1.1.1
  -- chapter_105_line_17: GL tag expansion.
  have row_17 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_18
  -- chapter_105_line_16: GL tag implication.
  have row_16 : (N v1) := by
    apply row_17
    exact row_19
  -- chapter_105_line_13: GL tag disintegration.
  have row_13 : (gl_implication13 N N N add) := by
    exact row_14.1.2
  -- chapter_105_line_12: GL tag expansion.
  have row_12 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication13] using row_13
  -- chapter_105_line_11: GL tag implication.
  have row_11 : (gl_existence1 N v1 previous add) := by
    apply row_12
    exact row_16
    exact row_20
  -- chapter_105_line_10: GL tag expansion.
  have row_10 : (¬ (∀ (v3 : α), ((N v3) → (¬ (add v1 previous v3))))) := by
    simpa only [gl_existence1] using row_11
  have exists_row_10 : ∃ (v3 : α), ((N v3) ∧ (add v1 previous v3)) := existsAndOfNotForallImpNot row_10
  obtain ⟨v3, witness_row_10⟩ := exists_row_10
  -- chapter_105_line_29: GL tag disintegration.
  have row_29 : (add v1 previous v3) := by
    exact witness_row_10.2
  -- chapter_105_line_26: GL tag implication.
  have row_26 : (succ v3 zero) := by
    apply row_27
    exact row_20
    exact row_25
    exact row_29
    exact row_19
  -- chapter_105_line_9: GL tag disintegration.
  have row_9 : (N v3) := by
    exact witness_row_10.1
  -- chapter_105_line_4: GL tag disintegration.
  have row_4 : (gl_implication6 N zero succ) := by
    exact row_5.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_105_line_3: GL tag expansion.
  have row_3 : (∀ (w1 : α), ((N w1) → (¬ (succ w1 zero)))) := by
    simpa only [gl_implication6] using row_4
  -- chapter_105_line_2: GL tag implication.
  have row_2 : (¬ (succ v3 zero)) := by
    apply row_3
    exact row_9
  -- chapter_105_line_1: GL tag vacuous truth.
  have row_1 : (zero = v1) := by
    exact False.elim (row_2 row_26)
  exact row_1

theorem peano_source_047
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((add v1 v2 zero) → ((N v2) → (zero = v1)))) := by
  intro v1
  intro v2
  intro premise_1
  intro premise_2
  have inductionMember : N v2 := by
    have typingRule := peano_source_047_induction_typing N zero succ add mul one anchor relationalInduction
    exact typingRule v2 premise_2
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v1 : α), ((add v1 zero zero) → ((N zero) → (zero = v1)))) := by
    intro v1
    intro base_premise_1
    intro base_premise_2
    have zeroRule := peano_source_047_check_zero N zero succ add mul one anchor relationalInduction
    exact zeroRule v1 zero base_premise_1 rfl
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v1 : α), ((add v1 induction_n zero) → ((N induction_n) → (zero = v1)))) → ∀ induction_m, succ induction_n induction_m → (∀ (v1 : α), ((add v1 induction_m zero) → ((N induction_m) → (zero = v1)))) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro v1
    intro step_premise_1
    intro step_premise_2
    have step_induction_assumption_1 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have stepRule := peano_source_047_check_induction_condition N zero succ add mul one anchor relationalInduction
    exact stepRule induction_n v1 induction_m step_induction_assumption_1 step_premise_1
  have inductionProperty : (∀ (v1 : α), ((add v1 v2 zero) → ((N v2) → (zero = v1)))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v1 : α), ((add v1 induction_value zero) → ((N induction_value) → (zero = v1)))))
      v2
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v1 premise_1 premise_2

private theorem peano_source_048_induction_typing
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (assumption_10 : (add v1 v2 zero))
    : (N v2) := by
  -- chapter_106_line_10: GL tag task formulation.
  have row_10 : (add v1 v2 zero) := by
    exact assumption_10
  -- chapter_106_line_9: GL tag task formulation.
  have row_9 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_106_line_8: GL tag expansion.
  have row_8 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_9
  -- chapter_106_line_7: GL tag disintegration.
  have row_7 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_8.1
  -- chapter_106_line_6: GL tag expansion.
  have row_6 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_7
  -- chapter_106_line_5: GL tag disintegration.
  have row_5 : (gl_fXYZ add N N N) := by
    exact row_6.1.1.1.1.1.1.1.1.2
  -- chapter_106_line_4: GL tag expansion.
  have row_4 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_5
  -- chapter_106_line_3: GL tag disintegration.
  have row_3 : (gl_implication9 add N) := by
    exact row_4.1.1.1.2
  -- chapter_106_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication9] using row_3
  -- chapter_106_line_1: GL tag implication.
  have row_1 : (N v2) := by
    apply row_2
    exact row_10
  exact row_1

private theorem peano_source_048_check_zero
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (assumption_4 : (v2 = zero))
    (assumption_3 : (add v1 v2 zero))
    : (add v2 v1 zero) := by
  -- chapter_107_line_10: GL tag task formulation.
  have row_10 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_107_line_19: GL tag expansion.
  have row_19 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_10
  -- chapter_107_line_18: GL tag disintegration.
  have row_18 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_19.1
  -- chapter_107_line_17: GL tag expansion.
  have row_17 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_18
  -- chapter_107_line_16: GL tag disintegration.
  have row_16 : (gl_fXYZ add N N N) := by
    exact row_17.1.1.1.1.1.1.1.1.2
  -- chapter_107_line_15: GL tag expansion.
  have row_15 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_16
  -- chapter_107_line_14: GL tag disintegration.
  have row_14 : (gl_implication8 add N) := by
    exact row_15.1.1.1.1
  -- chapter_107_line_13: GL tag expansion.
  have row_13 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_14
  -- chapter_107_line_9: GL tag theorem.
  have row_9 := peano_source_039 N zero succ add mul one anchor relationalInduction
  -- chapter_107_line_4: GL tag recursion.
  have row_4 : (v2 = zero) := by
    exact assumption_4
  -- chapter_107_line_6: GL tag symmetry of equality.
  have row_6 : (zero = v2) := by
    exact Eq.symm row_4
  -- chapter_107_line_3: GL tag task formulation.
  have row_3 : (add v1 v2 zero) := by
    exact assumption_3
  -- chapter_107_line_12: GL tag implication.
  have row_12 : (N v1) := by
    apply row_13
    exact row_3
  -- chapter_107_line_2: GL tag equality1.
  have row_2 : (add v1 zero zero) := by
    have equality_source := row_3
    have equality_step_1 := row_4
    cases equality_step_1
    exact equality_source
  -- chapter_107_line_11: GL tag equality1.
  have row_11 : (add v1 zero v2) := by
    have equality_source := row_2
    have equality_step_1 := row_6
    cases equality_step_1
    exact equality_source
  -- chapter_107_line_8: GL tag implication.
  have row_8 : (v1 = v2) := by
    apply row_9
    exact row_11
    exact row_12
  -- chapter_107_line_7: GL tag symmetry of equality.
  have row_7 : (v2 = v1) := by
    exact Eq.symm row_8
  -- chapter_107_line_5: GL tag equality2.
  have row_5 : (zero = v1) := by
    exact Eq.trans row_6 row_7
  -- chapter_107_line_1: GL tag equality1.
  have row_1 : (add v2 v1 zero) := by
    have equality_source := row_2
    have equality_step_1 := row_5
    cases equality_step_1
    have equality_step_2 := row_8
    cases equality_step_2
    exact equality_source
  exact row_1

private theorem peano_source_048_check_induction_condition
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (assumption_25 : (succ previous v2))
    (assumption_19 : (add v1 v2 zero))
    : (add v2 v1 zero) := by
  -- chapter_108_line_25: GL tag recursion.
  have row_25 : (succ previous v2) := by
    exact assumption_25
  -- chapter_108_line_19: GL tag task formulation.
  have row_19 : (add v1 v2 zero) := by
    exact assumption_19
  -- chapter_108_line_8: GL tag task formulation.
  have row_8 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_108_line_7: GL tag expansion.
  have row_7 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_8
  -- chapter_108_line_6: GL tag disintegration.
  have row_6 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_7.1
  -- chapter_108_line_5: GL tag expansion.
  have row_5 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_6
  -- chapter_108_line_28: GL tag disintegration.
  have row_28 : (gl_implication17 N succ add) := by
    exact row_5.1.1.1.1.1.2
  -- chapter_108_line_27: GL tag expansion.
  have row_27 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication17] using row_28
  -- chapter_108_line_24: GL tag disintegration.
  have row_24 : (gl_fXY succ N N) := by
    exact row_5.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_108_line_23: GL tag expansion.
  have row_23 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_24
  -- chapter_108_line_22: GL tag disintegration.
  have row_22 : (gl_implication0 succ N) := by
    exact row_23.1.1.1
  -- chapter_108_line_21: GL tag expansion.
  have row_21 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_22
  -- chapter_108_line_20: GL tag implication.
  have row_20 : (N previous) := by
    apply row_21
    exact row_25
  -- chapter_108_line_15: GL tag disintegration.
  have row_15 : (gl_fXYZ add N N N) := by
    exact row_5.1.1.1.1.1.1.1.1.2
  -- chapter_108_line_14: GL tag expansion.
  have row_14 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_15
  -- chapter_108_line_18: GL tag disintegration.
  have row_18 : (gl_implication8 add N) := by
    exact row_14.1.1.1.1
  -- chapter_108_line_17: GL tag expansion.
  have row_17 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_18
  -- chapter_108_line_16: GL tag implication.
  have row_16 : (N v1) := by
    apply row_17
    exact row_19
  -- chapter_108_line_13: GL tag disintegration.
  have row_13 : (gl_implication13 N N N add) := by
    exact row_14.1.2
  -- chapter_108_line_12: GL tag expansion.
  have row_12 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication13] using row_13
  -- chapter_108_line_11: GL tag implication.
  have row_11 : (gl_existence1 N v1 previous add) := by
    apply row_12
    exact row_16
    exact row_20
  -- chapter_108_line_10: GL tag expansion.
  have row_10 : (¬ (∀ (v3 : α), ((N v3) → (¬ (add v1 previous v3))))) := by
    simpa only [gl_existence1] using row_11
  have exists_row_10 : ∃ (v3 : α), ((N v3) ∧ (add v1 previous v3)) := existsAndOfNotForallImpNot row_10
  obtain ⟨v3, witness_row_10⟩ := exists_row_10
  -- chapter_108_line_29: GL tag disintegration.
  have row_29 : (add v1 previous v3) := by
    exact witness_row_10.2
  -- chapter_108_line_26: GL tag implication.
  have row_26 : (succ v3 zero) := by
    apply row_27
    exact row_20
    exact row_25
    exact row_29
    exact row_19
  -- chapter_108_line_9: GL tag disintegration.
  have row_9 : (N v3) := by
    exact witness_row_10.1
  -- chapter_108_line_4: GL tag disintegration.
  have row_4 : (gl_implication6 N zero succ) := by
    exact row_5.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_108_line_3: GL tag expansion.
  have row_3 : (∀ (w1 : α), ((N w1) → (¬ (succ w1 zero)))) := by
    simpa only [gl_implication6] using row_4
  -- chapter_108_line_2: GL tag implication.
  have row_2 : (¬ (succ v3 zero)) := by
    apply row_3
    exact row_9
  -- chapter_108_line_1: GL tag vacuous truth.
  have row_1 : (add v2 v1 zero) := by
    exact False.elim (row_2 row_26)
  exact row_1

theorem peano_source_048
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((add v1 v2 zero) → (add v2 v1 zero))) := by
  intro v1
  intro v2
  intro premise_1
  have inductionMember : N v2 := by
    have typingRule := peano_source_048_induction_typing N zero succ add mul one anchor relationalInduction
    exact typingRule v1 v2 premise_1
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v1 : α), ((add v1 zero zero) → (add zero v1 zero))) := by
    intro v1
    intro base_premise_1
    have zeroRule := peano_source_048_check_zero N zero succ add mul one anchor relationalInduction
    exact zeroRule v1 zero rfl base_premise_1
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v1 : α), ((add v1 induction_n zero) → (add induction_n v1 zero))) → ∀ induction_m, succ induction_n induction_m → (∀ (v1 : α), ((add v1 induction_m zero) → (add induction_m v1 zero))) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro v1
    intro step_premise_1
    have step_induction_assumption_1 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have stepRule := peano_source_048_check_induction_condition N zero succ add mul one anchor relationalInduction
    exact stepRule induction_n v1 induction_m step_induction_assumption_1 step_premise_1
  have inductionProperty : (∀ (v1 : α), ((add v1 v2 zero) → (add v2 v1 zero))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v1 : α), ((add v1 induction_value zero) → (add induction_value v1 zero))))
      v2
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v1 premise_1

private theorem peano_source_055_induction_typing
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (assumption_1 : (N v1))
    : (N v1) := by
  -- chapter_119_line_1: GL tag task formulation.
  have row_1 : (N v1) := by
    exact assumption_1
  exact row_1

private theorem peano_source_055_check_zero
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (assumption_13 : (N v1))
    (assumption_11 : (v1 = zero))
    (assumption_10 : (add zero v1 v2))
    : (v1 = v2) := by
  -- chapter_120_line_13: GL tag task formulation.
  have row_13 : (N v1) := by
    exact assumption_13
  -- chapter_120_line_11: GL tag recursion.
  have row_11 : (v1 = zero) := by
    exact assumption_11
  -- chapter_120_line_12: GL tag symmetry of equality.
  have row_12 : (zero = v1) := by
    exact Eq.symm row_11
  -- chapter_120_line_10: GL tag task formulation.
  have row_10 : (add zero v1 v2) := by
    exact assumption_10
  -- chapter_120_line_9: GL tag equality1.
  have row_9 : (add zero zero v2) := by
    have equality_source := row_10
    have equality_step_1 := row_11
    cases equality_step_1
    exact equality_source
  -- chapter_120_line_8: GL tag equality1.
  have row_8 : (add v1 zero v2) := by
    have equality_source := row_9
    have equality_step_1 := row_12
    cases equality_step_1
    exact equality_source
  -- chapter_120_line_7: GL tag task formulation.
  have row_7 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_120_line_6: GL tag expansion.
  have row_6 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_7
  -- chapter_120_line_5: GL tag disintegration.
  have row_5 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_6.1
  -- chapter_120_line_4: GL tag expansion.
  have row_4 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_5
  -- chapter_120_line_3: GL tag disintegration.
  have row_3 : (gl_implication15 N zero add) := by
    exact row_4.1.1.1.1.1.1.1.2
  -- chapter_120_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((add w1 zero w2) → (w1 = w2))))) := by
    simpa only [gl_implication15] using row_3
  -- chapter_120_line_1: GL tag implication.
  have row_1 : (v1 = v2) := by
    apply row_2
    exact row_13
    exact row_8
  exact row_1

private theorem peano_source_055_check_induction_condition
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (assumption_29 : (add zero v1 v2))
    (assumption_23 : (succ previous v1))
    (assumption_5 : ((N previous) → (∀ (w1 : α), ((add zero previous w1) → (previous = w1)))))
    : (v1 = v2) := by
  -- chapter_121_line_29: GL tag task formulation.
  have row_29 : (add zero v1 v2) := by
    exact assumption_29
  -- chapter_121_line_23: GL tag recursion.
  have row_23 : (succ previous v1) := by
    exact assumption_23
  -- chapter_121_line_16: GL tag task formulation.
  have row_16 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_121_line_15: GL tag expansion.
  have row_15 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_16
  -- chapter_121_line_14: GL tag disintegration.
  have row_14 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_15.1
  -- chapter_121_line_13: GL tag expansion.
  have row_13 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_14
  -- chapter_121_line_28: GL tag disintegration.
  have row_28 : (gl_implication17 N succ add) := by
    exact row_13.1.1.1.1.1.2
  -- chapter_121_line_27: GL tag expansion.
  have row_27 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication17] using row_28
  -- chapter_121_line_22: GL tag disintegration.
  have row_22 : (gl_fXY succ N N) := by
    exact row_13.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_121_line_21: GL tag expansion.
  have row_21 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_22
  -- chapter_121_line_20: GL tag disintegration.
  have row_20 : (gl_implication0 succ N) := by
    exact row_21.1.1.1
  -- chapter_121_line_19: GL tag expansion.
  have row_19 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_20
  -- chapter_121_line_18: GL tag implication.
  have row_18 : (N previous) := by
    apply row_19
    exact row_23
  -- chapter_121_line_17: GL tag disintegration.
  have row_17 : (N zero) := by
    exact row_13.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_121_line_12: GL tag disintegration.
  have row_12 : (gl_fXYZ add N N N) := by
    exact row_13.1.1.1.1.1.1.1.1.2
  -- chapter_121_line_11: GL tag expansion.
  have row_11 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_12
  -- chapter_121_line_10: GL tag disintegration.
  have row_10 : (gl_implication13 N N N add) := by
    exact row_11.1.2
  -- chapter_121_line_9: GL tag expansion.
  have row_9 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication13] using row_10
  -- chapter_121_line_8: GL tag implication.
  have row_8 : (gl_existence1 N zero previous add) := by
    apply row_9
    exact row_17
    exact row_18
  -- chapter_121_line_7: GL tag expansion.
  have row_7 : (¬ (∀ (v3 : α), ((N v3) → (¬ (add zero previous v3))))) := by
    simpa only [gl_existence1] using row_8
  have exists_row_7 : ∃ (v3 : α), ((N v3) ∧ (add zero previous v3)) := existsAndOfNotForallImpNot row_7
  obtain ⟨v3, witness_row_7⟩ := exists_row_7
  -- chapter_121_line_6: GL tag disintegration.
  have row_6 : (add zero previous v3) := by
    exact witness_row_7.2
  -- chapter_121_line_26: GL tag implication.
  have row_26 : (succ v3 v2) := by
    apply row_27
    exact row_18
    exact row_23
    exact row_6
    exact row_29
  -- chapter_121_line_5: GL tag recursion.
  have row_5 : ((N previous) → (∀ (w1 : α), ((add zero previous w1) → (previous = w1)))) := by
    exact assumption_5
  -- chapter_121_line_4: GL tag implication.
  have row_4 : (previous = v3) := by
    apply row_5
    exact row_18
    exact row_6
  -- chapter_121_line_24: GL tag equality1.
  have row_24 : (succ v3 v1) := by
    have equality_source := row_23
    have equality_step_1 := row_4
    cases equality_step_1
    exact equality_source
  -- chapter_121_line_3: GL tag symmetry of equality.
  have row_3 : (v3 = previous) := by
    exact Eq.symm row_4
  -- chapter_121_line_25: GL tag equality1.
  have row_25 : (succ previous v2) := by
    have equality_source := row_26
    have equality_step_1 := row_3
    cases equality_step_1
    exact equality_source
  -- chapter_121_line_2: GL tag theorem.
  have row_2 := peano_source_027 N zero succ add mul one anchor relationalInduction
  -- chapter_121_line_1: GL tag implication.
  have row_1 : (v1 = v2) := by
    apply row_2
    exact row_24
    exact row_25
    exact row_3
  exact row_1

theorem peano_source_055
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α), ((N v1) → (∀ (v2 : α), ((add zero v1 v2) → (v1 = v2))))) := by
  intro v1
  intro premise_1
  intro v2
  intro premise_2
  have inductionMember : N v1 := by
    have typingRule := peano_source_055_induction_typing N zero succ add mul one anchor relationalInduction
    exact typingRule v1 premise_1
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v2 : α), ((add zero zero v2) → (zero = v2))) := by
    intro v2
    intro base_premise_1
    have zeroRule := peano_source_055_check_zero N zero succ add mul one anchor relationalInduction
    exact zeroRule zero v2 inductionZeroMember rfl base_premise_1
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v2 : α), ((add zero induction_n v2) → (induction_n = v2))) → ∀ induction_m, succ induction_n induction_m → (∀ (v2 : α), ((add zero induction_m v2) → (induction_m = v2))) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro v2
    intro step_premise_1
    have step_induction_assumption_1 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have step_induction_assumption_2 :
        ((N induction_n) → (∀ (w1 : α), ((add zero induction_n w1) → (induction_n = w1)))) := by
      intro step_induction_assumption_2_premise_1
      intro w1
      intro step_induction_assumption_2_premise_2
      apply induction_hypothesis
      all_goals assumption
    have stepRule := peano_source_055_check_induction_condition N zero succ add mul one anchor relationalInduction
    exact stepRule induction_n induction_m v2 step_premise_1 step_induction_assumption_1 step_induction_assumption_2
  have inductionProperty : (∀ (v2 : α), ((add zero v1 v2) → (v1 = v2))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v2 : α), ((add zero induction_value v2) → (induction_value = v2))))
      v1
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v2 premise_2

private theorem peano_source_056_induction_typing
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (assumption_1 : (N v1))
    : (N v1) := by
  -- chapter_122_line_1: GL tag task formulation.
  have row_1 : (N v1) := by
    exact assumption_1
  exact row_1

private theorem peano_source_056_check_zero
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (assumption_9 : (N v1))
    (assumption_7 : (v1 = zero))
    (assumption_6 : (mul zero v1 v2))
    : (zero = v2) := by
  -- chapter_123_line_9: GL tag task formulation.
  have row_9 : (N v1) := by
    exact assumption_9
  -- chapter_123_line_7: GL tag recursion.
  have row_7 : (v1 = zero) := by
    exact assumption_7
  -- chapter_123_line_8: GL tag symmetry of equality.
  have row_8 : (zero = v1) := by
    exact Eq.symm row_7
  -- chapter_123_line_6: GL tag task formulation.
  have row_6 : (mul zero v1 v2) := by
    exact assumption_6
  -- chapter_123_line_5: GL tag equality1.
  have row_5 : (mul zero zero v2) := by
    have equality_source := row_6
    have equality_step_1 := row_7
    cases equality_step_1
    exact equality_source
  -- chapter_123_line_4: GL tag equality1.
  have row_4 : (mul v1 zero v2) := by
    have equality_source := row_5
    have equality_step_1 := row_8
    cases equality_step_1
    exact equality_source
  -- chapter_123_line_3: GL tag task formulation.
  have row_3 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_123_line_2: GL tag theorem.
  have row_2 := peano_source_041 N zero succ add mul one anchor relationalInduction
  -- chapter_123_line_1: GL tag implication.
  have row_1 : (zero = v2) := by
    apply row_2
    exact row_4
    exact row_9
  exact row_1

private theorem peano_source_056_check_induction_condition
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (assumption_27 : (mul zero v1 v2))
    (assumption_21 : (succ previous v1))
    (assumption_2 : ((N previous) → (∀ (w1 : α), ((mul zero previous w1) → (zero = w1)))))
    : (zero = v2) := by
  -- chapter_124_line_27: GL tag task formulation.
  have row_27 : (mul zero v1 v2) := by
    exact assumption_27
  -- chapter_124_line_23: GL tag theorem.
  have row_23 := peano_source_039 N zero succ add mul one anchor relationalInduction
  -- chapter_124_line_21: GL tag recursion.
  have row_21 : (succ previous v1) := by
    exact assumption_21
  -- chapter_124_line_14: GL tag task formulation.
  have row_14 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_124_line_13: GL tag expansion.
  have row_13 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_14
  -- chapter_124_line_12: GL tag disintegration.
  have row_12 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_13.1
  -- chapter_124_line_11: GL tag expansion.
  have row_11 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_12
  -- chapter_124_line_26: GL tag disintegration.
  have row_26 : (gl_implication21 N succ mul add) := by
    exact row_11.2
  -- chapter_124_line_25: GL tag expansion.
  have row_25 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((mul w3 w1 w4) → (∀ (w5 : α), ((mul w3 w2 w5) → (add w4 w3 w5))))))))) := by
    simpa only [gl_implication21] using row_26
  -- chapter_124_line_20: GL tag disintegration.
  have row_20 : (gl_fXY succ N N) := by
    exact row_11.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_124_line_19: GL tag expansion.
  have row_19 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_20
  -- chapter_124_line_18: GL tag disintegration.
  have row_18 : (gl_implication0 succ N) := by
    exact row_19.1.1.1
  -- chapter_124_line_17: GL tag expansion.
  have row_17 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_18
  -- chapter_124_line_16: GL tag implication.
  have row_16 : (N previous) := by
    apply row_17
    exact row_21
  -- chapter_124_line_15: GL tag disintegration.
  have row_15 : (N zero) := by
    exact row_11.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_124_line_10: GL tag disintegration.
  have row_10 : (gl_fXYZ mul N N N) := by
    exact row_11.1.1.1.2
  -- chapter_124_line_9: GL tag expansion.
  have row_9 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_10
  -- chapter_124_line_8: GL tag disintegration.
  have row_8 : (gl_implication13 N N N mul) := by
    exact row_9.1.2
  -- chapter_124_line_7: GL tag expansion.
  have row_7 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication13] using row_8
  -- chapter_124_line_6: GL tag implication.
  have row_6 : (gl_existence1 N zero previous mul) := by
    apply row_7
    exact row_15
    exact row_16
  -- chapter_124_line_5: GL tag expansion.
  have row_5 : (¬ (∀ (v4 : α), ((N v4) → (¬ (mul zero previous v4))))) := by
    simpa only [gl_existence1] using row_6
  have exists_row_5 : ∃ (v4 : α), ((N v4) ∧ (mul zero previous v4)) := existsAndOfNotForallImpNot row_5
  obtain ⟨v4, witness_row_5⟩ := exists_row_5
  -- chapter_124_line_28: GL tag disintegration.
  have row_28 : (N v4) := by
    exact witness_row_5.1
  -- chapter_124_line_4: GL tag disintegration.
  have row_4 : (mul zero previous v4) := by
    exact witness_row_5.2
  -- chapter_124_line_24: GL tag implication.
  have row_24 : (add v4 zero v2) := by
    apply row_25
    exact row_16
    exact row_21
    exact row_4
    exact row_27
  -- chapter_124_line_22: GL tag implication.
  have row_22 : (v4 = v2) := by
    apply row_23
    exact row_24
    exact row_28
  -- chapter_124_line_3: GL tag equality1.
  have row_3 : (mul zero previous v2) := by
    have equality_source := row_4
    have equality_step_1 := row_22
    cases equality_step_1
    exact equality_source
  -- chapter_124_line_2: GL tag recursion.
  have row_2 : ((N previous) → (∀ (w1 : α), ((mul zero previous w1) → (zero = w1)))) := by
    exact assumption_2
  -- chapter_124_line_1: GL tag implication.
  have row_1 : (zero = v2) := by
    apply row_2
    exact row_16
    exact row_3
  exact row_1

theorem peano_source_056
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α), ((N v1) → (∀ (v2 : α), ((mul zero v1 v2) → (zero = v2))))) := by
  intro v1
  intro premise_1
  intro v2
  intro premise_2
  have inductionMember : N v1 := by
    have typingRule := peano_source_056_induction_typing N zero succ add mul one anchor relationalInduction
    exact typingRule v1 premise_1
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v2 : α), ((mul zero zero v2) → (zero = v2))) := by
    intro v2
    intro base_premise_1
    have zeroRule := peano_source_056_check_zero N zero succ add mul one anchor relationalInduction
    exact zeroRule zero v2 inductionZeroMember rfl base_premise_1
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v2 : α), ((mul zero induction_n v2) → (zero = v2))) → ∀ induction_m, succ induction_n induction_m → (∀ (v2 : α), ((mul zero induction_m v2) → (zero = v2))) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro v2
    intro step_premise_1
    have step_induction_assumption_1 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have step_induction_assumption_2 :
        ((N induction_n) → (∀ (w1 : α), ((mul zero induction_n w1) → (zero = w1)))) := by
      intro step_induction_assumption_2_premise_1
      intro w1
      intro step_induction_assumption_2_premise_2
      apply induction_hypothesis
      all_goals assumption
    have stepRule := peano_source_056_check_induction_condition N zero succ add mul one anchor relationalInduction
    exact stepRule induction_n induction_m v2 step_premise_1 step_induction_assumption_1 step_induction_assumption_2
  have inductionProperty : (∀ (v2 : α), ((mul zero v1 v2) → (zero = v2))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v2 : α), ((mul zero induction_value v2) → (zero = v2))))
      v1
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v2 premise_2

theorem peano_source_061
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α), ((N v1) → (gl_or1 N v1 succ zero))) := by
  intro v1
  intro premise_1
  have or_parent_1 := peano_source_060 N zero succ add mul one anchor relationalInduction
  have or_parent_2 := peano_source_059 N zero succ add mul one anchor relationalInduction
  -- chapter_137_line_1: GL tag or theorem.
  have row_1 : (∀ (v1 : α), ((N v1) → (gl_or1 N v1 succ zero))) := by
    classical
    intro v1
    intro or_parent_premise_1
    simp only [gl_or1]
    by_cases or_case_1 : (gl_existence3 N v1 succ)
    · exact Or.inl (or_case_1)
    ·
      exact Or.inr ((or_parent_1 v1 or_parent_premise_1 or_case_1))
  solve_by_elim [row_1]

theorem peano_source_062
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α), ((N v1) → (gl_or1 N v1 succ zero))) := by
  intro v1
  intro premise_1
  have or_parent_1 := peano_source_060 N zero succ add mul one anchor relationalInduction
  have or_parent_2 := peano_source_059 N zero succ add mul one anchor relationalInduction
  -- chapter_138_line_1: GL tag or theorem.
  have row_1 : (∀ (v1 : α), ((N v1) → (gl_or1 N v1 succ zero))) := by
    classical
    intro v1
    intro or_parent_premise_1
    simp only [gl_or1]
    by_cases or_case_1 : (gl_existence3 N v1 succ)
    · exact Or.inl (or_case_1)
    ·
      exact Or.inr ((or_parent_1 v1 or_parent_premise_1 or_case_1))
  solve_by_elim [row_1]

private theorem peano_source_000_induction_typing
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v2 : α)
    (v5 : α)
    (v7 : α)
    (assumption_10 : (mul v7 v5 v2))
    : (N v5) := by
  -- chapter_0_line_10: GL tag task formulation.
  have row_10 : (mul v7 v5 v2) := by
    exact assumption_10
  -- chapter_0_line_9: GL tag task formulation.
  have row_9 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_0_line_8: GL tag expansion.
  have row_8 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_9
  -- chapter_0_line_7: GL tag disintegration.
  have row_7 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_8.1
  -- chapter_0_line_6: GL tag expansion.
  have row_6 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_7
  -- chapter_0_line_5: GL tag disintegration.
  have row_5 : (gl_fXYZ mul N N N) := by
    exact row_6.1.1.1.2
  -- chapter_0_line_4: GL tag expansion.
  have row_4 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_5
  -- chapter_0_line_3: GL tag disintegration.
  have row_3 : (gl_implication9 mul N) := by
    exact row_4.1.1.1.2
  -- chapter_0_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication9] using row_3
  -- chapter_0_line_1: GL tag implication.
  have row_1 : (N v5) := by
    apply row_2
    exact row_10
  exact row_1

private theorem peano_source_000_check_zero
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (v4 : α)
    (v5 : α)
    (v6 : α)
    (v7 : α)
    (assumption_76 : (mul v7 v5 v2))
    (assumption_33 : (v5 = zero))
    (assumption_32 : (add v4 v5 v6))
    (assumption_9 : (add v1 v2 v3))
    (assumption_3 : (mul v7 v4 v1))
    : (mul v7 v6 v3) := by
  -- chapter_1_line_85: GL tag theorem.
  have row_85 := peano_source_055 N zero succ add mul one anchor relationalInduction
  -- chapter_1_line_76: GL tag task formulation.
  have row_76 : (mul v7 v5 v2) := by
    exact assumption_76
  -- chapter_1_line_74: GL tag theorem.
  have row_74 := peano_source_041 N zero succ add mul one anchor relationalInduction
  -- chapter_1_line_57: GL tag theorem.
  have row_57 := peano_source_034 N zero succ add mul one anchor relationalInduction
  -- chapter_1_line_42: GL tag theorem.
  have row_42 := peano_source_030 N zero succ add mul one anchor relationalInduction
  -- chapter_1_line_33: GL tag recursion.
  have row_33 : (v5 = zero) := by
    exact assumption_33
  -- chapter_1_line_75: GL tag equality1.
  have row_75 : (mul v7 zero v2) := by
    have equality_source := row_76
    have equality_step_1 := row_33
    cases equality_step_1
    exact equality_source
  -- chapter_1_line_39: GL tag symmetry of equality.
  have row_39 : (zero = v5) := by
    exact Eq.symm row_33
  -- chapter_1_line_32: GL tag task formulation.
  have row_32 : (add v4 v5 v6) := by
    exact assumption_32
  -- chapter_1_line_31: GL tag equality1.
  have row_31 : (add v4 zero v6) := by
    have equality_source := row_32
    have equality_step_1 := row_33
    cases equality_step_1
    exact equality_source
  -- chapter_1_line_30: GL tag theorem.
  have row_30 := peano_source_040 N zero succ add mul one anchor relationalInduction
  -- chapter_1_line_11: GL tag theorem.
  have row_11 := peano_source_039 N zero succ add mul one anchor relationalInduction
  -- chapter_1_line_9: GL tag task formulation.
  have row_9 : (add v1 v2 v3) := by
    exact assumption_9
  -- chapter_1_line_6: GL tag task formulation.
  have row_6 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_1_line_29: GL tag implication.
  have row_29 : (add zero v4 v6) := by
    apply row_30
    exact row_31
  -- chapter_1_line_21: GL tag expansion.
  have row_21 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_6
  -- chapter_1_line_48: GL tag disintegration.
  have row_48 : (succ zero one) := by
    exact row_21.2
  -- chapter_1_line_47: GL tag equality1.
  have row_47 : (succ v5 one) := by
    have equality_source := row_48
    have equality_step_1 := row_39
    cases equality_step_1
    exact equality_source
  -- chapter_1_line_20: GL tag disintegration.
  have row_20 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_21.1
  -- chapter_1_line_19: GL tag expansion.
  have row_19 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_20
  -- chapter_1_line_81: GL tag disintegration.
  have row_81 : (gl_fXYZ mul N N N) := by
    exact row_19.1.1.1.2
  -- chapter_1_line_80: GL tag expansion.
  have row_80 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_81
  -- chapter_1_line_79: GL tag disintegration.
  have row_79 : (gl_implication8 mul N) := by
    exact row_80.1.1.1.1
  -- chapter_1_line_78: GL tag expansion.
  have row_78 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_79
  -- chapter_1_line_64: GL tag disintegration.
  have row_64 : (gl_fXY succ N N) := by
    exact row_19.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_1_line_63: GL tag expansion.
  have row_63 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_64
  -- chapter_1_line_62: GL tag disintegration.
  have row_62 : (gl_implication4 N N succ) := by
    exact row_63.1.2
  -- chapter_1_line_61: GL tag expansion.
  have row_61 : (∀ (w1 : α), ((N w1) → (gl_existence0 N w1 succ))) := by
    simpa only [gl_implication4] using row_62
  -- chapter_1_line_46: GL tag disintegration.
  have row_46 : (gl_implication17 N succ add) := by
    exact row_19.1.1.1.1.1.2
  -- chapter_1_line_45: GL tag expansion.
  have row_45 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication17] using row_46
  -- chapter_1_line_22: GL tag disintegration.
  have row_22 : (N zero) := by
    exact row_19.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_1_line_18: GL tag disintegration.
  have row_18 : (gl_fXYZ add N N N) := by
    exact row_19.1.1.1.1.1.1.1.1.2
  -- chapter_1_line_17: GL tag expansion.
  have row_17 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_18
  -- chapter_1_line_54: GL tag disintegration.
  have row_54 : (gl_implication9 add N) := by
    exact row_17.1.1.1.2
  -- chapter_1_line_53: GL tag expansion.
  have row_53 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication9] using row_54
  -- chapter_1_line_55: GL tag implication.
  have row_55 : (N v2) := by
    apply row_53
    exact row_9
  -- chapter_1_line_60: GL tag implication.
  have row_60 : (gl_existence0 N v2 succ) := by
    apply row_61
    exact row_55
  -- chapter_1_line_59: GL tag expansion.
  have row_59 : (¬ (∀ (v12 : α), ((N v12) → (¬ (succ v2 v12))))) := by
    simpa only [gl_existence0] using row_60
  have exists_row_59 : ∃ (v12 : α), ((N v12) ∧ (succ v2 v12)) := existsAndOfNotForallImpNot row_59
  obtain ⟨v12, witness_row_59⟩ := exists_row_59
  -- chapter_1_line_58: GL tag disintegration.
  have row_58 : (succ v2 v12) := by
    exact witness_row_59.2
  -- chapter_1_line_56: GL tag implication.
  have row_56 : (add v2 one v12) := by
    apply row_57
    exact row_58
  -- chapter_1_line_52: GL tag implication.
  have row_52 : (N v5) := by
    apply row_53
    exact row_32
  -- chapter_1_line_38: GL tag disintegration.
  have row_38 : (gl_implication8 add N) := by
    exact row_17.1.1.1.1
  -- chapter_1_line_37: GL tag expansion.
  have row_37 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_38
  -- chapter_1_line_36: GL tag implication.
  have row_36 : (N v4) := by
    apply row_37
    exact row_32
  -- chapter_1_line_35: GL tag implication.
  have row_35 : (v4 = v6) := by
    apply row_11
    exact row_31
    exact row_36
  -- chapter_1_line_34: GL tag symmetry of equality.
  have row_34 : (v6 = v4) := by
    exact Eq.symm row_35
  -- chapter_1_line_28: GL tag equality1.
  have row_28 : (add v5 v4 v4) := by
    have equality_source := row_29
    have equality_step_1 := row_34
    cases equality_step_1
    have equality_step_2 := row_39
    cases equality_step_2
    exact equality_source
  -- chapter_1_line_27: GL tag equality1.
  have row_27 : (add v5 v4 v6) := by
    have equality_source := row_28
    have equality_step_1 := row_35
    cases equality_step_1
    exact equality_source
  -- chapter_1_line_25: GL tag disintegration.
  have row_25 : (gl_implication10 add N) := by
    exact row_17.1.1.2
  -- chapter_1_line_24: GL tag expansion.
  have row_24 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication10] using row_25
  -- chapter_1_line_23: GL tag implication.
  have row_23 : (N v3) := by
    apply row_24
    exact row_9
  -- chapter_1_line_16: GL tag disintegration.
  have row_16 : (gl_implication13 N N N add) := by
    exact row_17.1.2
  -- chapter_1_line_15: GL tag expansion.
  have row_15 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication13] using row_16
  -- chapter_1_line_71: GL tag implication.
  have row_71 : (gl_existence1 N v5 v2 add) := by
    apply row_15
    exact row_52
    exact row_55
  -- chapter_1_line_70: GL tag expansion.
  have row_70 : (¬ (∀ (v16 : α), ((N v16) → (¬ (add v5 v2 v16))))) := by
    simpa only [gl_existence1] using row_71
  have exists_row_70 : ∃ (v16 : α), ((N v16) ∧ (add v5 v2 v16)) := existsAndOfNotForallImpNot row_70
  obtain ⟨v16, witness_row_70⟩ := exists_row_70
  -- chapter_1_line_69: GL tag disintegration.
  have row_69 : (add v5 v2 v16) := by
    exact witness_row_70.2
  -- chapter_1_line_86: GL tag equality1.
  have row_86 : (add zero v2 v16) := by
    have equality_source := row_69
    have equality_step_1 := row_33
    cases equality_step_1
    exact equality_source
  -- chapter_1_line_84: GL tag implication.
  have row_84 : (v2 = v16) := by
    apply row_85
    exact row_55
    exact row_86
  -- chapter_1_line_83: GL tag symmetry of equality.
  have row_83 : (v16 = v2) := by
    exact Eq.symm row_84
  -- chapter_1_line_51: GL tag implication.
  have row_51 : (gl_existence1 N v2 v5 add) := by
    apply row_15
    exact row_55
    exact row_52
  -- chapter_1_line_50: GL tag expansion.
  have row_50 : (¬ (∀ (v13 : α), ((N v13) → (¬ (add v2 v5 v13))))) := by
    simpa only [gl_existence1] using row_51
  have exists_row_50 : ∃ (v13 : α), ((N v13) ∧ (add v2 v5 v13)) := existsAndOfNotForallImpNot row_50
  obtain ⟨v13, witness_row_50⟩ := exists_row_50
  -- chapter_1_line_49: GL tag disintegration.
  have row_49 : (add v2 v5 v13) := by
    exact witness_row_50.2
  -- chapter_1_line_88: GL tag equality1.
  have row_88 : (add v2 zero v13) := by
    have equality_source := row_49
    have equality_step_1 := row_33
    cases equality_step_1
    exact equality_source
  -- chapter_1_line_87: GL tag implication.
  have row_87 : (v2 = v13) := by
    apply row_11
    exact row_88
    exact row_55
  -- chapter_1_line_82: GL tag equality2.
  have row_82 : (v16 = v13) := by
    exact Eq.trans row_83 row_87
  -- chapter_1_line_44: GL tag implication.
  have row_44 : (succ v13 v12) := by
    apply row_45
    exact row_52
    exact row_47
    exact row_49
    exact row_56
  -- chapter_1_line_14: GL tag implication.
  have row_14 : (gl_existence1 N v3 zero add) := by
    apply row_15
    exact row_23
    exact row_22
  -- chapter_1_line_13: GL tag expansion.
  have row_13 : (¬ (∀ (v8 : α), ((N v8) → (¬ (add v3 zero v8))))) := by
    simpa only [gl_existence1] using row_14
  have exists_row_13 : ∃ (v8 : α), ((N v8) ∧ (add v3 zero v8)) := existsAndOfNotForallImpNot row_13
  obtain ⟨v8, witness_row_13⟩ := exists_row_13
  -- chapter_1_line_12: GL tag disintegration.
  have row_12 : (add v3 zero v8) := by
    exact witness_row_13.2
  -- chapter_1_line_89: GL tag equality1.
  have row_89 : (add v3 v5 v8) := by
    have equality_source := row_12
    have equality_step_1 := row_39
    cases equality_step_1
    exact equality_source
  -- chapter_1_line_10: GL tag implication.
  have row_10 : (v3 = v8) := by
    apply row_11
    exact row_12
    exact row_23
  -- chapter_1_line_8: GL tag equality1.
  have row_8 : (add v1 v2 v8) := by
    have equality_source := row_9
    have equality_step_1 := row_10
    cases equality_step_1
    exact equality_source
  -- chapter_1_line_5: GL tag theorem.
  have row_5 := peano_source_012 N zero succ add mul one anchor relationalInduction
  -- chapter_1_line_3: GL tag task formulation.
  have row_3 : (mul v7 v4 v1) := by
    exact assumption_3
  -- chapter_1_line_77: GL tag implication.
  have row_77 : (N v7) := by
    apply row_78
    exact row_3
  -- chapter_1_line_73: GL tag implication.
  have row_73 : (zero = v2) := by
    apply row_74
    exact row_75
    exact row_77
  -- chapter_1_line_72: GL tag symmetry of equality.
  have row_72 : (v2 = zero) := by
    exact Eq.symm row_73
  -- chapter_1_line_68: GL tag equality1.
  have row_68 : (add v5 zero v13) := by
    have equality_source := row_69
    have equality_step_1 := row_72
    cases equality_step_1
    have equality_step_2 := row_82
    cases equality_step_2
    exact equality_source
  -- chapter_1_line_67: GL tag implication.
  have row_67 : (v5 = v13) := by
    apply row_11
    exact row_68
    exact row_52
  -- chapter_1_line_66: GL tag equality1.
  have row_66 : (add v13 v4 v6) := by
    have equality_source := row_27
    have equality_step_1 := row_67
    cases equality_step_1
    exact equality_source
  -- chapter_1_line_65: GL tag implication.
  have row_65 : (v13 = v5) := by
    apply row_5
    exact row_66
    exact row_27
  -- chapter_1_line_43: GL tag equality1.
  have row_43 : (succ v5 v12) := by
    have equality_source := row_44
    have equality_step_1 := row_65
    cases equality_step_1
    exact equality_source
  -- chapter_1_line_41: GL tag implication.
  have row_41 : (v5 = v2) := by
    apply row_42
    exact row_43
    exact row_58
  -- chapter_1_line_40: GL tag equality1.
  have row_40 : (add v2 v4 v6) := by
    have equality_source := row_27
    have equality_step_1 := row_41
    cases equality_step_1
    exact equality_source
  -- chapter_1_line_26: GL tag implication.
  have row_26 : (v2 = v5) := by
    apply row_5
    exact row_40
    exact row_27
  -- chapter_1_line_7: GL tag equality1.
  have row_7 : (add v1 v5 v8) := by
    have equality_source := row_8
    have equality_step_1 := row_26
    cases equality_step_1
    exact equality_source
  -- chapter_1_line_4: GL tag implication.
  have row_4 : (v1 = v3) := by
    apply row_5
    exact row_7
    exact row_89
  -- chapter_1_line_2: GL tag equality1.
  have row_2 : (mul v7 v4 v3) := by
    have equality_source := row_3
    have equality_step_1 := row_4
    cases equality_step_1
    exact equality_source
  -- chapter_1_line_1: GL tag equality1.
  have row_1 : (mul v7 v6 v3) := by
    have equality_source := row_2
    have equality_step_1 := row_35
    cases equality_step_1
    exact equality_source
  exact row_1

private theorem peano_source_000_check_induction_condition
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (v4 : α)
    (v5 : α)
    (v6 : α)
    (v7 : α)
    (assumption_64 : (mul v7 v5 v2))
    (assumption_58 : (∀ (w1 : α) (w2 : α), ((add v1 w1 w2) → (∀ (w3 : α), ((add v4 previous w3) → ((mul v7 v4 v1) → ((mul v7 previous w1) → (mul v7 w3 w2))))))))
    (assumption_46 : (succ previous v5))
    (assumption_26 : (add v1 v2 v3))
    (assumption_22 : (mul v7 v4 v1))
    (assumption_18 : (add v4 v5 v6))
    : (mul v7 v6 v3) := by
  -- chapter_2_line_66: GL tag theorem.
  have row_66 := peano_source_020 N zero succ add mul one anchor relationalInduction
  -- chapter_2_line_64: GL tag task formulation.
  have row_64 : (mul v7 v5 v2) := by
    exact assumption_64
  -- chapter_2_line_58: GL tag recursion.
  have row_58 : (∀ (w1 : α) (w2 : α), ((add v1 w1 w2) → (∀ (w3 : α), ((add v4 previous w3) → ((mul v7 v4 v1) → ((mul v7 previous w1) → (mul v7 w3 w2))))))) := by
    exact assumption_58
  -- chapter_2_line_46: GL tag recursion.
  have row_46 : (succ previous v5) := by
    exact assumption_46
  -- chapter_2_line_28: GL tag theorem.
  have row_28 := peano_source_004 N zero succ add mul one anchor relationalInduction
  -- chapter_2_line_26: GL tag task formulation.
  have row_26 : (add v1 v2 v3) := by
    exact assumption_26
  -- chapter_2_line_22: GL tag task formulation.
  have row_22 : (mul v7 v4 v1) := by
    exact assumption_22
  -- chapter_2_line_18: GL tag task formulation.
  have row_18 : (add v4 v5 v6) := by
    exact assumption_18
  -- chapter_2_line_12: GL tag task formulation.
  have row_12 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_2_line_11: GL tag expansion.
  have row_11 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_12
  -- chapter_2_line_10: GL tag disintegration.
  have row_10 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_11.1
  -- chapter_2_line_9: GL tag expansion.
  have row_9 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_10
  -- chapter_2_line_69: GL tag disintegration.
  have row_69 : (gl_implication17 N succ add) := by
    exact row_9.1.1.1.1.1.2
  -- chapter_2_line_68: GL tag expansion.
  have row_68 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication17] using row_69
  -- chapter_2_line_63: GL tag disintegration.
  have row_63 : (gl_implication21 N succ mul add) := by
    exact row_9.2
  -- chapter_2_line_62: GL tag expansion.
  have row_62 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((mul w3 w1 w4) → (∀ (w5 : α), ((mul w3 w2 w5) → (add w4 w3 w5))))))))) := by
    simpa only [gl_implication21] using row_63
  -- chapter_2_line_45: GL tag disintegration.
  have row_45 : (gl_fXY succ N N) := by
    exact row_9.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_2_line_44: GL tag expansion.
  have row_44 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_45
  -- chapter_2_line_43: GL tag disintegration.
  have row_43 : (gl_implication0 succ N) := by
    exact row_44.1.1.1
  -- chapter_2_line_42: GL tag expansion.
  have row_42 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_43
  -- chapter_2_line_41: GL tag implication.
  have row_41 : (N previous) := by
    apply row_42
    exact row_46
  -- chapter_2_line_17: GL tag disintegration.
  have row_17 : (gl_fXYZ add N N N) := by
    exact row_9.1.1.1.1.1.1.1.1.2
  -- chapter_2_line_16: GL tag expansion.
  have row_16 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_17
  -- chapter_2_line_72: GL tag disintegration.
  have row_72 : (gl_implication9 add N) := by
    exact row_16.1.1.1.2
  -- chapter_2_line_71: GL tag expansion.
  have row_71 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication9] using row_72
  -- chapter_2_line_70: GL tag implication.
  have row_70 : (N v2) := by
    apply row_71
    exact row_26
  -- chapter_2_line_37: GL tag disintegration.
  have row_37 : (gl_implication8 add N) := by
    exact row_16.1.1.1.1
  -- chapter_2_line_36: GL tag expansion.
  have row_36 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_37
  -- chapter_2_line_56: GL tag implication.
  have row_56 : (N v4) := by
    apply row_36
    exact row_18
  -- chapter_2_line_35: GL tag implication.
  have row_35 : (N v1) := by
    apply row_36
    exact row_26
  -- chapter_2_line_34: GL tag disintegration.
  have row_34 : (gl_implication13 N N N add) := by
    exact row_16.1.2
  -- chapter_2_line_33: GL tag expansion.
  have row_33 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication13] using row_34
  -- chapter_2_line_55: GL tag implication.
  have row_55 : (gl_existence1 N v4 previous add) := by
    apply row_33
    exact row_56
    exact row_41
  -- chapter_2_line_54: GL tag expansion.
  have row_54 : (¬ (∀ (v16 : α), ((N v16) → (¬ (add v4 previous v16))))) := by
    simpa only [gl_existence1] using row_55
  have exists_row_54 : ∃ (v16 : α), ((N v16) ∧ (add v4 previous v16)) := existsAndOfNotForallImpNot row_54
  obtain ⟨v16, witness_row_54⟩ := exists_row_54
  -- chapter_2_line_59: GL tag disintegration.
  have row_59 : (add v4 previous v16) := by
    exact witness_row_54.2
  -- chapter_2_line_67: GL tag implication.
  have row_67 : (succ v16 v6) := by
    apply row_68
    exact row_41
    exact row_46
    exact row_59
    exact row_18
  -- chapter_2_line_53: GL tag disintegration.
  have row_53 : (N v16) := by
    exact witness_row_54.1
  -- chapter_2_line_25: GL tag disintegration.
  have row_25 : (gl_implication14 N N add) := by
    exact row_16.2
  -- chapter_2_line_24: GL tag expansion.
  have row_24 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (∀ (w3 : α), ((add w1 w2 w3) → (∀ (w4 : α), ((add w1 w2 w4) → (w3 = w4))))))))) := by
    simpa only [gl_implication14] using row_25
  -- chapter_2_line_15: GL tag disintegration.
  have row_15 : (gl_implication10 add N) := by
    exact row_16.1.1.2
  -- chapter_2_line_14: GL tag expansion.
  have row_14 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication10] using row_15
  -- chapter_2_line_13: GL tag implication.
  have row_13 : (N v6) := by
    apply row_14
    exact row_18
  -- chapter_2_line_8: GL tag disintegration.
  have row_8 : (gl_fXYZ mul N N N) := by
    exact row_9.1.1.1.2
  -- chapter_2_line_7: GL tag expansion.
  have row_7 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_8
  -- chapter_2_line_49: GL tag disintegration.
  have row_49 : (gl_implication14 N N mul) := by
    exact row_7.2
  -- chapter_2_line_48: GL tag expansion.
  have row_48 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (∀ (w3 : α), ((mul w1 w2 w3) → (∀ (w4 : α), ((mul w1 w2 w4) → (w3 = w4))))))))) := by
    simpa only [gl_implication14] using row_49
  -- chapter_2_line_21: GL tag disintegration.
  have row_21 : (gl_implication8 mul N) := by
    exact row_7.1.1.1.1
  -- chapter_2_line_20: GL tag expansion.
  have row_20 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_21
  -- chapter_2_line_19: GL tag implication.
  have row_19 : (N v7) := by
    apply row_20
    exact row_22
  -- chapter_2_line_6: GL tag disintegration.
  have row_6 : (gl_implication13 N N N mul) := by
    exact row_7.1.2
  -- chapter_2_line_5: GL tag expansion.
  have row_5 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication13] using row_6
  -- chapter_2_line_52: GL tag implication.
  have row_52 : (gl_existence1 N v7 v16 mul) := by
    apply row_5
    exact row_19
    exact row_53
  -- chapter_2_line_51: GL tag expansion.
  have row_51 : (¬ (∀ (v14 : α), ((N v14) → (¬ (mul v7 v16 v14))))) := by
    simpa only [gl_existence1] using row_52
  have exists_row_51 : ∃ (v14 : α), ((N v14) ∧ (mul v7 v16 v14)) := existsAndOfNotForallImpNot row_51
  obtain ⟨v14, witness_row_51⟩ := exists_row_51
  -- chapter_2_line_50: GL tag disintegration.
  have row_50 : (mul v7 v16 v14) := by
    exact witness_row_51.2
  -- chapter_2_line_40: GL tag implication.
  have row_40 : (gl_existence1 N v7 previous mul) := by
    apply row_5
    exact row_19
    exact row_41
  -- chapter_2_line_39: GL tag expansion.
  have row_39 : (¬ (∀ (v13 : α), ((N v13) → (¬ (mul v7 previous v13))))) := by
    simpa only [gl_existence1] using row_40
  have exists_row_39 : ∃ (v13 : α), ((N v13) ∧ (mul v7 previous v13)) := existsAndOfNotForallImpNot row_39
  obtain ⟨v13, witness_row_39⟩ := exists_row_39
  -- chapter_2_line_60: GL tag disintegration.
  have row_60 : (mul v7 previous v13) := by
    exact witness_row_39.2
  -- chapter_2_line_61: GL tag implication.
  have row_61 : (add v13 v7 v2) := by
    apply row_62
    exact row_41
    exact row_46
    exact row_60
    exact row_64
  -- chapter_2_line_38: GL tag disintegration.
  have row_38 : (N v13) := by
    exact witness_row_39.1
  -- chapter_2_line_32: GL tag implication.
  have row_32 : (gl_existence1 N v1 v13 add) := by
    apply row_33
    exact row_35
    exact row_38
  -- chapter_2_line_31: GL tag expansion.
  have row_31 : (¬ (∀ (v15 : α), ((N v15) → (¬ (add v1 v13 v15))))) := by
    simpa only [gl_existence1] using row_32
  have exists_row_31 : ∃ (v15 : α), ((N v15) ∧ (add v1 v13 v15)) := existsAndOfNotForallImpNot row_31
  obtain ⟨v15, witness_row_31⟩ := exists_row_31
  -- chapter_2_line_30: GL tag disintegration.
  have row_30 : (add v1 v13 v15) := by
    exact witness_row_31.2
  -- chapter_2_line_57: GL tag implication.
  have row_57 : (mul v7 v16 v15) := by
    apply row_58
    exact row_30
    exact row_59
    exact row_22
    exact row_60
  -- chapter_2_line_47: GL tag implication.
  have row_47 : (v15 = v14) := by
    apply row_48
    exact row_19
    exact row_53
    exact row_57
    exact row_50
  -- chapter_2_line_29: GL tag equality1.
  have row_29 : (add v1 v13 v14) := by
    have equality_source := row_30
    have equality_step_1 := row_47
    cases equality_step_1
    exact equality_source
  -- chapter_2_line_4: GL tag implication.
  have row_4 : (gl_existence1 N v7 v6 mul) := by
    apply row_5
    exact row_19
    exact row_13
  -- chapter_2_line_3: GL tag expansion.
  have row_3 : (¬ (∀ (v8 : α), ((N v8) → (¬ (mul v7 v6 v8))))) := by
    simpa only [gl_existence1] using row_4
  have exists_row_3 : ∃ (v8 : α), ((N v8) ∧ (mul v7 v6 v8)) := existsAndOfNotForallImpNot row_3
  obtain ⟨v8, witness_row_3⟩ := exists_row_3
  -- chapter_2_line_2: GL tag disintegration.
  have row_2 : (mul v7 v6 v8) := by
    exact witness_row_3.2
  -- chapter_2_line_65: GL tag implication.
  have row_65 : (add v14 v7 v8) := by
    apply row_66
    exact row_2
    exact row_50
    exact row_67
  -- chapter_2_line_27: GL tag implication.
  have row_27 : (add v1 v2 v8) := by
    apply row_28
    exact row_65
    exact row_29
    exact row_61
  -- chapter_2_line_23: GL tag implication.
  have row_23 : (v8 = v3) := by
    apply row_24
    exact row_35
    exact row_70
    exact row_27
    exact row_26
  -- chapter_2_line_1: GL tag equality1.
  have row_1 : (mul v7 v6 v3) := by
    have equality_source := row_2
    have equality_step_1 := row_23
    cases equality_step_1
    exact equality_source
  exact row_1

theorem peano_source_000
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α) (v5 : α) (v6 : α), ((add v4 v5 v6) → (∀ (v7 : α), ((mul v7 v4 v1) → ((mul v7 v5 v2) → (mul v7 v6 v3)))))))) := by
  intro v1
  intro v2
  intro v3
  intro premise_1
  intro v4
  intro v5
  intro v6
  intro premise_2
  intro v7
  intro premise_3
  intro premise_4
  have inductionMember : N v5 := by
    have typingRule := peano_source_000_induction_typing N zero succ add mul one anchor relationalInduction
    exact typingRule v2 v5 v7 premise_4
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α) (v6 : α), ((add v4 zero v6) → (∀ (v7 : α), ((mul v7 v4 v1) → ((mul v7 zero v2) → (mul v7 v6 v3)))))))) := by
    intro v1
    intro v2
    intro v3
    intro base_premise_1
    intro v4
    intro v6
    intro base_premise_2
    intro v7
    intro base_premise_3
    intro base_premise_4
    have zeroRule := peano_source_000_check_zero N zero succ add mul one anchor relationalInduction
    exact zeroRule v1 v2 v3 v4 zero v6 v7 base_premise_4 rfl base_premise_2 base_premise_1 base_premise_3
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α) (v6 : α), ((add v4 induction_n v6) → (∀ (v7 : α), ((mul v7 v4 v1) → ((mul v7 induction_n v2) → (mul v7 v6 v3)))))))) → ∀ induction_m, succ induction_n induction_m → (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α) (v6 : α), ((add v4 induction_m v6) → (∀ (v7 : α), ((mul v7 v4 v1) → ((mul v7 induction_m v2) → (mul v7 v6 v3)))))))) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro v1
    intro v2
    intro v3
    intro step_premise_1
    intro v4
    intro v6
    intro step_premise_2
    intro v7
    intro step_premise_3
    intro step_premise_4
    have step_induction_assumption_1 :
        (∀ (w1 : α) (w2 : α), ((add v1 w1 w2) → (∀ (w3 : α), ((add v4 induction_n w3) → ((mul v7 v4 v1) → ((mul v7 induction_n w1) → (mul v7 w3 w2))))))) := by
      intro w1
      intro w2
      intro step_induction_assumption_1_premise_1
      intro w3
      intro step_induction_assumption_1_premise_2
      intro step_induction_assumption_1_premise_3
      intro step_induction_assumption_1_premise_4
      apply induction_hypothesis
      all_goals assumption
    have step_induction_assumption_2 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have stepRule := peano_source_000_check_induction_condition N zero succ add mul one anchor relationalInduction
    exact stepRule induction_n v1 v2 v3 v4 induction_m v6 v7 step_premise_4 step_induction_assumption_1 step_induction_assumption_2 step_premise_1 step_premise_3 step_premise_2
  have inductionProperty : (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α) (v6 : α), ((add v4 v5 v6) → (∀ (v7 : α), ((mul v7 v4 v1) → ((mul v7 v5 v2) → (mul v7 v6 v3)))))))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α) (v6 : α), ((add v4 induction_value v6) → (∀ (v7 : α), ((mul v7 v4 v1) → ((mul v7 induction_value v2) → (mul v7 v6 v3)))))))))
      v5
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v1 v2 v3 premise_1 v4 v6 premise_2 v7 premise_3 premise_4

private theorem peano_source_001_induction_typing
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (v5 : α)
    (assumption_10 : (add v2 v1 v5))
    : (N v1) := by
  -- chapter_3_line_10: GL tag task formulation.
  have row_10 : (add v2 v1 v5) := by
    exact assumption_10
  -- chapter_3_line_9: GL tag task formulation.
  have row_9 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_3_line_8: GL tag expansion.
  have row_8 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_9
  -- chapter_3_line_7: GL tag disintegration.
  have row_7 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_8.1
  -- chapter_3_line_6: GL tag expansion.
  have row_6 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_7
  -- chapter_3_line_5: GL tag disintegration.
  have row_5 : (gl_fXYZ add N N N) := by
    exact row_6.1.1.1.1.1.1.1.1.2
  -- chapter_3_line_4: GL tag expansion.
  have row_4 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_5
  -- chapter_3_line_3: GL tag disintegration.
  have row_3 : (gl_implication9 add N) := by
    exact row_4.1.1.1.2
  -- chapter_3_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication9] using row_3
  -- chapter_3_line_1: GL tag implication.
  have row_1 : (N v1) := by
    apply row_2
    exact row_10
  exact row_1

private theorem peano_source_001_check_zero
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (v4 : α)
    (v5 : α)
    (v6 : α)
    (assumption_38 : (v1 = zero))
    (assumption_30 : (add v1 v2 v3))
    (assumption_21 : (add v2 v1 v5))
    (assumption_2 : (add v4 v5 v6))
    : (add v4 v3 v6) := by
  -- chapter_4_line_40: GL tag theorem.
  have row_40 := peano_source_055 N zero succ add mul one anchor relationalInduction
  -- chapter_4_line_38: GL tag recursion.
  have row_38 : (v1 = zero) := by
    exact assumption_38
  -- chapter_4_line_37: GL tag symmetry of equality.
  have row_37 : (zero = v1) := by
    exact Eq.symm row_38
  -- chapter_4_line_30: GL tag task formulation.
  have row_30 : (add v1 v2 v3) := by
    exact assumption_30
  -- chapter_4_line_41: GL tag equality1.
  have row_41 : (add zero v2 v3) := by
    have equality_source := row_30
    have equality_step_1 := row_38
    cases equality_step_1
    exact equality_source
  -- chapter_4_line_21: GL tag task formulation.
  have row_21 : (add v2 v1 v5) := by
    exact assumption_21
  -- chapter_4_line_6: GL tag task formulation.
  have row_6 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_4_line_15: GL tag expansion.
  have row_15 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_6
  -- chapter_4_line_14: GL tag disintegration.
  have row_14 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_15.1
  -- chapter_4_line_13: GL tag expansion.
  have row_13 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_14
  -- chapter_4_line_20: GL tag disintegration.
  have row_20 : (gl_fXYZ add N N N) := by
    exact row_13.1.1.1.1.1.1.1.1.2
  -- chapter_4_line_19: GL tag expansion.
  have row_19 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_20
  -- chapter_4_line_34: GL tag disintegration.
  have row_34 : (gl_implication9 add N) := by
    exact row_19.1.1.1.2
  -- chapter_4_line_33: GL tag expansion.
  have row_33 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication9] using row_34
  -- chapter_4_line_29: GL tag disintegration.
  have row_29 : (gl_implication8 add N) := by
    exact row_19.1.1.1.1
  -- chapter_4_line_28: GL tag expansion.
  have row_28 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_29
  -- chapter_4_line_31: GL tag implication.
  have row_31 : (N v2) := by
    apply row_28
    exact row_21
  -- chapter_4_line_39: GL tag implication.
  have row_39 : (v2 = v3) := by
    apply row_40
    exact row_31
    exact row_41
  -- chapter_4_line_27: GL tag implication.
  have row_27 : (N v1) := by
    apply row_28
    exact row_30
  -- chapter_4_line_26: GL tag disintegration.
  have row_26 : (gl_implication13 N N N add) := by
    exact row_19.1.2
  -- chapter_4_line_25: GL tag expansion.
  have row_25 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication13] using row_26
  -- chapter_4_line_24: GL tag implication.
  have row_24 : (gl_existence1 N v2 v1 add) := by
    apply row_25
    exact row_31
    exact row_27
  -- chapter_4_line_23: GL tag expansion.
  have row_23 : (¬ (∀ (v7 : α), ((N v7) → (¬ (add v2 v1 v7))))) := by
    simpa only [gl_existence1] using row_24
  have exists_row_23 : ∃ (v7 : α), ((N v7) ∧ (add v2 v1 v7)) := existsAndOfNotForallImpNot row_23
  obtain ⟨v7, witness_row_23⟩ := exists_row_23
  -- chapter_4_line_35: GL tag disintegration.
  have row_35 : (N v7) := by
    exact witness_row_23.1
  -- chapter_4_line_22: GL tag disintegration.
  have row_22 : (add v2 v1 v7) := by
    exact witness_row_23.2
  -- chapter_4_line_18: GL tag disintegration.
  have row_18 : (gl_implication14 N N add) := by
    exact row_19.2
  -- chapter_4_line_17: GL tag expansion.
  have row_17 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (∀ (w3 : α), ((add w1 w2 w3) → (∀ (w4 : α), ((add w1 w2 w4) → (w3 = w4))))))))) := by
    simpa only [gl_implication14] using row_18
  -- chapter_4_line_36: GL tag implication.
  have row_36 : (v7 = v5) := by
    apply row_17
    exact row_31
    exact row_27
    exact row_22
    exact row_21
  -- chapter_4_line_16: GL tag implication.
  have row_16 : (v5 = v7) := by
    apply row_17
    exact row_31
    exact row_27
    exact row_21
    exact row_22
  -- chapter_4_line_12: GL tag disintegration.
  have row_12 : (gl_implication16 N zero add) := by
    exact row_13.1.1.1.1.1.1.2
  -- chapter_4_line_11: GL tag expansion.
  have row_11 : (∀ (w1 : α) (w2 : α), ((w1 = w2) → ((N w1) → ((N w2) → (add w1 zero w2))))) := by
    simpa only [gl_implication16] using row_12
  -- chapter_4_line_5: GL tag theorem.
  have row_5 := peano_source_012 N zero succ add mul one anchor relationalInduction
  -- chapter_4_line_2: GL tag task formulation.
  have row_2 : (add v4 v5 v6) := by
    exact assumption_2
  -- chapter_4_line_32: GL tag implication.
  have row_32 : (N v5) := by
    apply row_33
    exact row_2
  -- chapter_4_line_10: GL tag implication.
  have row_10 : (add v5 zero v7) := by
    apply row_11
    exact row_16
    exact row_32
    exact row_35
  -- chapter_4_line_9: GL tag equality1.
  have row_9 : (add v5 zero v5) := by
    have equality_source := row_10
    have equality_step_1 := row_36
    cases equality_step_1
    exact equality_source
  -- chapter_4_line_8: GL tag equality1.
  have row_8 : (add v5 v1 v5) := by
    have equality_source := row_9
    have equality_step_1 := row_37
    cases equality_step_1
    exact equality_source
  -- chapter_4_line_7: GL tag equality1.
  have row_7 : (add v5 v1 v7) := by
    have equality_source := row_8
    have equality_step_1 := row_16
    cases equality_step_1
    exact equality_source
  -- chapter_4_line_4: GL tag implication.
  have row_4 : (v5 = v2) := by
    apply row_5
    exact row_7
    exact row_22
  -- chapter_4_line_3: GL tag equality2.
  have row_3 : (v5 = v3) := by
    exact Eq.trans row_4 row_39
  -- chapter_4_line_1: GL tag equality1.
  have row_1 : (add v4 v3 v6) := by
    have equality_source := row_2
    have equality_step_1 := row_3
    cases equality_step_1
    exact equality_source
  exact row_1

private theorem peano_source_001_check_induction_condition
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (v4 : α)
    (v5 : α)
    (v6 : α)
    (assumption_51 : (∀ (w1 : α), ((add previous v2 w1) → (∀ (w2 : α) (w3 : α), ((add v4 w2 w3) → ((add v2 previous w2) → (add v4 w1 w3)))))))
    (assumption_37 : (add v4 v5 v6))
    (assumption_22 : (add v2 v1 v5))
    (assumption_11 : (add v1 v2 v3))
    (assumption_10 : (succ previous v1))
    : (add v4 v3 v6) := by
  -- chapter_5_line_51: GL tag recursion.
  have row_51 : (∀ (w1 : α), ((add previous v2 w1) → (∀ (w2 : α) (w3 : α), ((add v4 w2 w3) → ((add v2 previous w2) → (add v4 w1 w3)))))) := by
    exact assumption_51
  -- chapter_5_line_37: GL tag task formulation.
  have row_37 : (add v4 v5 v6) := by
    exact assumption_37
  -- chapter_5_line_30: GL tag theorem.
  have row_30 := peano_source_006 N zero succ add mul one anchor relationalInduction
  -- chapter_5_line_22: GL tag task formulation.
  have row_22 : (add v2 v1 v5) := by
    exact assumption_22
  -- chapter_5_line_11: GL tag task formulation.
  have row_11 : (add v1 v2 v3) := by
    exact assumption_11
  -- chapter_5_line_10: GL tag recursion.
  have row_10 : (succ previous v1) := by
    exact assumption_10
  -- chapter_5_line_9: GL tag theorem.
  have row_9 := peano_source_005 N zero succ add mul one anchor relationalInduction
  -- chapter_5_line_7: GL tag task formulation.
  have row_7 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_5_line_6: GL tag expansion.
  have row_6 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_7
  -- chapter_5_line_5: GL tag disintegration.
  have row_5 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_6.1
  -- chapter_5_line_4: GL tag expansion.
  have row_4 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_5
  -- chapter_5_line_33: GL tag disintegration.
  have row_33 : (gl_implication17 N succ add) := by
    exact row_4.1.1.1.1.1.2
  -- chapter_5_line_32: GL tag expansion.
  have row_32 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication17] using row_33
  -- chapter_5_line_27: GL tag disintegration.
  have row_27 : (gl_fXY succ N N) := by
    exact row_4.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_5_line_26: GL tag expansion.
  have row_26 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_27
  -- chapter_5_line_25: GL tag disintegration.
  have row_25 : (gl_implication0 succ N) := by
    exact row_26.1.1.1
  -- chapter_5_line_24: GL tag expansion.
  have row_24 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_25
  -- chapter_5_line_23: GL tag implication.
  have row_23 : (N previous) := by
    apply row_24
    exact row_10
  -- chapter_5_line_18: GL tag disintegration.
  have row_18 : (gl_fXYZ add N N N) := by
    exact row_4.1.1.1.1.1.1.1.1.2
  -- chapter_5_line_17: GL tag expansion.
  have row_17 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_18
  -- chapter_5_line_45: GL tag disintegration.
  have row_45 : (gl_implication14 N N add) := by
    exact row_17.2
  -- chapter_5_line_44: GL tag expansion.
  have row_44 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (∀ (w3 : α), ((add w1 w2 w3) → (∀ (w4 : α), ((add w1 w2 w4) → (w3 = w4))))))))) := by
    simpa only [gl_implication14] using row_45
  -- chapter_5_line_21: GL tag disintegration.
  have row_21 : (gl_implication8 add N) := by
    exact row_17.1.1.1.1
  -- chapter_5_line_20: GL tag expansion.
  have row_20 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_21
  -- chapter_5_line_41: GL tag implication.
  have row_41 : (N v4) := by
    apply row_20
    exact row_37
  -- chapter_5_line_19: GL tag implication.
  have row_19 : (N v2) := by
    apply row_20
    exact row_22
  -- chapter_5_line_16: GL tag disintegration.
  have row_16 : (gl_implication13 N N N add) := by
    exact row_17.1.2
  -- chapter_5_line_15: GL tag expansion.
  have row_15 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication13] using row_16
  -- chapter_5_line_36: GL tag implication.
  have row_36 : (gl_existence1 N v2 previous add) := by
    apply row_15
    exact row_19
    exact row_23
  -- chapter_5_line_35: GL tag expansion.
  have row_35 : (¬ (∀ (v15 : α), ((N v15) → (¬ (add v2 previous v15))))) := by
    simpa only [gl_existence1] using row_36
  have exists_row_35 : ∃ (v15 : α), ((N v15) ∧ (add v2 previous v15)) := existsAndOfNotForallImpNot row_35
  obtain ⟨v15, witness_row_35⟩ := exists_row_35
  -- chapter_5_line_42: GL tag disintegration.
  have row_42 : (N v15) := by
    exact witness_row_35.1
  -- chapter_5_line_40: GL tag implication.
  have row_40 : (gl_existence1 N v4 v15 add) := by
    apply row_15
    exact row_41
    exact row_42
  -- chapter_5_line_39: GL tag expansion.
  have row_39 : (¬ (∀ (v14 : α), ((N v14) → (¬ (add v4 v15 v14))))) := by
    simpa only [gl_existence1] using row_40
  have exists_row_39 : ∃ (v14 : α), ((N v14) ∧ (add v4 v15 v14)) := existsAndOfNotForallImpNot row_39
  obtain ⟨v14, witness_row_39⟩ := exists_row_39
  -- chapter_5_line_38: GL tag disintegration.
  have row_38 : (add v4 v15 v14) := by
    exact witness_row_39.2
  -- chapter_5_line_34: GL tag disintegration.
  have row_34 : (add v2 previous v15) := by
    exact witness_row_35.2
  -- chapter_5_line_31: GL tag implication.
  have row_31 : (succ v15 v5) := by
    apply row_32
    exact row_23
    exact row_10
    exact row_34
    exact row_22
  -- chapter_5_line_29: GL tag implication.
  have row_29 : (succ v14 v6) := by
    apply row_30
    exact row_37
    exact row_38
    exact row_31
  -- chapter_5_line_14: GL tag implication.
  have row_14 : (gl_existence1 N previous v2 add) := by
    apply row_15
    exact row_23
    exact row_19
  -- chapter_5_line_13: GL tag expansion.
  have row_13 : (¬ (∀ (v12 : α), ((N v12) → (¬ (add previous v2 v12))))) := by
    simpa only [gl_existence1] using row_14
  have exists_row_13 : ∃ (v12 : α), ((N v12) ∧ (add previous v2 v12)) := existsAndOfNotForallImpNot row_13
  obtain ⟨v12, witness_row_13⟩ := exists_row_13
  -- chapter_5_line_49: GL tag disintegration.
  have row_49 : (N v12) := by
    exact witness_row_13.1
  -- chapter_5_line_48: GL tag implication.
  have row_48 : (gl_existence1 N v4 v12 add) := by
    apply row_15
    exact row_41
    exact row_49
  -- chapter_5_line_47: GL tag expansion.
  have row_47 : (¬ (∀ (v13 : α), ((N v13) → (¬ (add v4 v12 v13))))) := by
    simpa only [gl_existence1] using row_48
  have exists_row_47 : ∃ (v13 : α), ((N v13) ∧ (add v4 v12 v13)) := existsAndOfNotForallImpNot row_47
  obtain ⟨v13, witness_row_47⟩ := exists_row_47
  -- chapter_5_line_46: GL tag disintegration.
  have row_46 : (add v4 v12 v13) := by
    exact witness_row_47.2
  -- chapter_5_line_12: GL tag disintegration.
  have row_12 : (add previous v2 v12) := by
    exact witness_row_13.2
  -- chapter_5_line_50: GL tag implication.
  have row_50 : (add v4 v12 v14) := by
    apply row_51
    exact row_12
    exact row_38
    exact row_34
  -- chapter_5_line_43: GL tag implication.
  have row_43 : (v14 = v13) := by
    apply row_44
    exact row_41
    exact row_49
    exact row_50
    exact row_46
  -- chapter_5_line_28: GL tag equality1.
  have row_28 : (succ v13 v6) := by
    have equality_source := row_29
    have equality_step_1 := row_43
    cases equality_step_1
    exact equality_source
  -- chapter_5_line_8: GL tag implication.
  have row_8 : (succ v12 v3) := by
    apply row_9
    exact row_11
    exact row_12
    exact row_10
  -- chapter_5_line_3: GL tag disintegration.
  have row_3 : (gl_implication18 N succ add) := by
    exact row_4.1.1.1.1.2
  -- chapter_5_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((succ w4 w5) → (add w3 w2 w5))))))))) := by
    simpa only [gl_implication18] using row_3
  -- chapter_5_line_1: GL tag implication.
  have row_1 : (add v4 v3 v6) := by
    apply row_2
    exact row_49
    exact row_8
    exact row_46
    exact row_28
  exact row_1

theorem peano_source_001
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α) (v5 : α) (v6 : α), ((add v4 v5 v6) → ((add v2 v1 v5) → (add v4 v3 v6)))))) := by
  intro v1
  intro v2
  intro v3
  intro premise_1
  intro v4
  intro v5
  intro v6
  intro premise_2
  intro premise_3
  have inductionMember : N v1 := by
    have typingRule := peano_source_001_induction_typing N zero succ add mul one anchor relationalInduction
    exact typingRule v1 v2 v5 premise_3
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v2 : α) (v3 : α), ((add zero v2 v3) → (∀ (v4 : α) (v5 : α) (v6 : α), ((add v4 v5 v6) → ((add v2 zero v5) → (add v4 v3 v6)))))) := by
    intro v2
    intro v3
    intro base_premise_1
    intro v4
    intro v5
    intro v6
    intro base_premise_2
    intro base_premise_3
    have zeroRule := peano_source_001_check_zero N zero succ add mul one anchor relationalInduction
    exact zeroRule zero v2 v3 v4 v5 v6 rfl base_premise_1 base_premise_3 base_premise_2
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v2 : α) (v3 : α), ((add induction_n v2 v3) → (∀ (v4 : α) (v5 : α) (v6 : α), ((add v4 v5 v6) → ((add v2 induction_n v5) → (add v4 v3 v6)))))) → ∀ induction_m, succ induction_n induction_m → (∀ (v2 : α) (v3 : α), ((add induction_m v2 v3) → (∀ (v4 : α) (v5 : α) (v6 : α), ((add v4 v5 v6) → ((add v2 induction_m v5) → (add v4 v3 v6)))))) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro v2
    intro v3
    intro step_premise_1
    intro v4
    intro v5
    intro v6
    intro step_premise_2
    intro step_premise_3
    have step_induction_assumption_1 :
        (∀ (w1 : α), ((add induction_n v2 w1) → (∀ (w2 : α) (w3 : α), ((add v4 w2 w3) → ((add v2 induction_n w2) → (add v4 w1 w3)))))) := by
      intro w1
      intro step_induction_assumption_1_premise_1
      intro w2
      intro w3
      intro step_induction_assumption_1_premise_2
      intro step_induction_assumption_1_premise_3
      apply induction_hypothesis
      all_goals assumption
    have step_induction_assumption_2 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have stepRule := peano_source_001_check_induction_condition N zero succ add mul one anchor relationalInduction
    exact stepRule induction_n induction_m v2 v3 v4 v5 v6 step_induction_assumption_1 step_premise_2 step_premise_3 step_premise_1 step_induction_assumption_2
  have inductionProperty : (∀ (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α) (v5 : α) (v6 : α), ((add v4 v5 v6) → ((add v2 v1 v5) → (add v4 v3 v6)))))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v2 : α) (v3 : α), ((add induction_value v2 v3) → (∀ (v4 : α) (v5 : α) (v6 : α), ((add v4 v5 v6) → ((add v2 induction_value v5) → (add v4 v3 v6)))))))
      v1
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v2 v3 premise_1 v4 v5 v6 premise_2 premise_3

private theorem peano_source_002_induction_typing
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (v4 : α)
    (assumption_10 : (mul v4 v1 v2))
    : (N v1) := by
  -- chapter_6_line_10: GL tag task formulation.
  have row_10 : (mul v4 v1 v2) := by
    exact assumption_10
  -- chapter_6_line_9: GL tag task formulation.
  have row_9 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_6_line_8: GL tag expansion.
  have row_8 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_9
  -- chapter_6_line_7: GL tag disintegration.
  have row_7 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_8.1
  -- chapter_6_line_6: GL tag expansion.
  have row_6 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_7
  -- chapter_6_line_5: GL tag disintegration.
  have row_5 : (gl_fXYZ mul N N N) := by
    exact row_6.1.1.1.2
  -- chapter_6_line_4: GL tag expansion.
  have row_4 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_5
  -- chapter_6_line_3: GL tag disintegration.
  have row_3 : (gl_implication9 mul N) := by
    exact row_4.1.1.1.2
  -- chapter_6_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication9] using row_3
  -- chapter_6_line_1: GL tag implication.
  have row_1 : (N v1) := by
    apply row_2
    exact row_10
  exact row_1

private theorem peano_source_002_check_zero
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (v4 : α)
    (v5 : α)
    (assumption_36 : (mul v4 v1 v2))
    (assumption_26 : (v1 = zero))
    (assumption_25 : (add v1 v2 v3))
    (assumption_19 : (succ v4 v5))
    : (mul v5 v1 v3) := by
  -- chapter_7_line_41: GL tag theorem.
  have row_41 := peano_source_055 N zero succ add mul one anchor relationalInduction
  -- chapter_7_line_36: GL tag task formulation.
  have row_36 : (mul v4 v1 v2) := by
    exact assumption_36
  -- chapter_7_line_34: GL tag theorem.
  have row_34 := peano_source_041 N zero succ add mul one anchor relationalInduction
  -- chapter_7_line_26: GL tag recursion.
  have row_26 : (v1 = zero) := by
    exact assumption_26
  -- chapter_7_line_35: GL tag equality1.
  have row_35 : (mul v4 zero v2) := by
    have equality_source := row_36
    have equality_step_1 := row_26
    cases equality_step_1
    exact equality_source
  -- chapter_7_line_27: GL tag symmetry of equality.
  have row_27 : (zero = v1) := by
    exact Eq.symm row_26
  -- chapter_7_line_25: GL tag task formulation.
  have row_25 : (add v1 v2 v3) := by
    exact assumption_25
  -- chapter_7_line_42: GL tag equality1.
  have row_42 : (add zero v2 v3) := by
    have equality_source := row_25
    have equality_step_1 := row_26
    cases equality_step_1
    exact equality_source
  -- chapter_7_line_19: GL tag task formulation.
  have row_19 : (succ v4 v5) := by
    exact assumption_19
  -- chapter_7_line_13: GL tag task formulation.
  have row_13 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_7_line_12: GL tag expansion.
  have row_12 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_13
  -- chapter_7_line_11: GL tag disintegration.
  have row_11 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_12.1
  -- chapter_7_line_10: GL tag expansion.
  have row_10 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_11
  -- chapter_7_line_32: GL tag disintegration.
  have row_32 : (gl_implication19 N zero mul) := by
    exact row_10.1.1.2
  -- chapter_7_line_31: GL tag expansion.
  have row_31 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((mul w1 zero w2) → (w2 = zero))))) := by
    simpa only [gl_implication19] using row_32
  -- chapter_7_line_24: GL tag disintegration.
  have row_24 : (gl_fXYZ add N N N) := by
    exact row_10.1.1.1.1.1.1.1.1.2
  -- chapter_7_line_23: GL tag expansion.
  have row_23 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_24
  -- chapter_7_line_45: GL tag disintegration.
  have row_45 : (gl_implication9 add N) := by
    exact row_23.1.1.1.2
  -- chapter_7_line_44: GL tag expansion.
  have row_44 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication9] using row_45
  -- chapter_7_line_43: GL tag implication.
  have row_43 : (N v2) := by
    apply row_44
    exact row_25
  -- chapter_7_line_40: GL tag implication.
  have row_40 : (v2 = v3) := by
    apply row_41
    exact row_43
    exact row_42
  -- chapter_7_line_22: GL tag disintegration.
  have row_22 : (gl_implication8 add N) := by
    exact row_23.1.1.1.1
  -- chapter_7_line_21: GL tag expansion.
  have row_21 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_22
  -- chapter_7_line_20: GL tag implication.
  have row_20 : (N v1) := by
    apply row_21
    exact row_25
  -- chapter_7_line_18: GL tag disintegration.
  have row_18 : (gl_fXY succ N N) := by
    exact row_10.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_7_line_17: GL tag expansion.
  have row_17 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_18
  -- chapter_7_line_16: GL tag disintegration.
  have row_16 : (gl_implication1 succ N) := by
    exact row_17.1.1.2
  -- chapter_7_line_15: GL tag expansion.
  have row_15 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using row_16
  -- chapter_7_line_14: GL tag implication.
  have row_14 : (N v5) := by
    apply row_15
    exact row_19
  -- chapter_7_line_9: GL tag disintegration.
  have row_9 : (gl_fXYZ mul N N N) := by
    exact row_10.1.1.1.2
  -- chapter_7_line_8: GL tag expansion.
  have row_8 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_9
  -- chapter_7_line_39: GL tag disintegration.
  have row_39 : (gl_implication8 mul N) := by
    exact row_8.1.1.1.1
  -- chapter_7_line_38: GL tag expansion.
  have row_38 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_39
  -- chapter_7_line_37: GL tag implication.
  have row_37 : (N v4) := by
    apply row_38
    exact row_36
  -- chapter_7_line_33: GL tag implication.
  have row_33 : (zero = v2) := by
    apply row_34
    exact row_35
    exact row_37
  -- chapter_7_line_7: GL tag disintegration.
  have row_7 : (gl_implication13 N N N mul) := by
    exact row_8.1.2
  -- chapter_7_line_6: GL tag expansion.
  have row_6 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication13] using row_7
  -- chapter_7_line_5: GL tag implication.
  have row_5 : (gl_existence1 N v5 v1 mul) := by
    apply row_6
    exact row_14
    exact row_20
  -- chapter_7_line_4: GL tag expansion.
  have row_4 : (¬ (∀ (v6 : α), ((N v6) → (¬ (mul v5 v1 v6))))) := by
    simpa only [gl_existence1] using row_5
  have exists_row_4 : ∃ (v6 : α), ((N v6) ∧ (mul v5 v1 v6)) := existsAndOfNotForallImpNot row_4
  obtain ⟨v6, witness_row_4⟩ := exists_row_4
  -- chapter_7_line_3: GL tag disintegration.
  have row_3 : (mul v5 v1 v6) := by
    exact witness_row_4.2
  -- chapter_7_line_2: GL tag equality1.
  have row_2 : (mul v5 zero v6) := by
    have equality_source := row_3
    have equality_step_1 := row_26
    cases equality_step_1
    exact equality_source
  -- chapter_7_line_30: GL tag implication.
  have row_30 : (v6 = zero) := by
    apply row_31
    exact row_14
    exact row_2
  -- chapter_7_line_29: GL tag equality2.
  have row_29 : (v6 = v2) := by
    exact Eq.trans row_30 row_33
  -- chapter_7_line_28: GL tag equality2.
  have row_28 : (v6 = v3) := by
    exact Eq.trans row_29 row_40
  -- chapter_7_line_1: GL tag equality1.
  have row_1 : (mul v5 v1 v3) := by
    have equality_source := row_2
    have equality_step_1 := row_27
    cases equality_step_1
    have equality_step_2 := row_28
    cases equality_step_2
    exact equality_source
  exact row_1

private theorem peano_source_002_check_induction_condition
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (v4 : α)
    (v5 : α)
    (assumption_75 : (∀ (w1 : α) (w2 : α), ((add previous w1 w2) → ((succ v4 v5) → ((mul v4 previous w1) → (mul v5 previous w2))))))
    (assumption_50 : (mul v4 v1 v2))
    (assumption_33 : (add v1 v2 v3))
    (assumption_20 : (succ previous v1))
    (assumption_18 : (succ v4 v5))
    : (mul v5 v1 v3) := by
  -- chapter_8_line_75: GL tag recursion.
  have row_75 : (∀ (w1 : α) (w2 : α), ((add previous w1 w2) → ((succ v4 v5) → ((mul v4 previous w1) → (mul v5 previous w2))))) := by
    exact assumption_75
  -- chapter_8_line_50: GL tag task formulation.
  have row_50 : (mul v4 v1 v2) := by
    exact assumption_50
  -- chapter_8_line_45: GL tag theorem.
  have row_45 := peano_source_004 N zero succ add mul one anchor relationalInduction
  -- chapter_8_line_37: GL tag theorem.
  have row_37 := peano_source_011 N zero succ add mul one anchor relationalInduction
  -- chapter_8_line_35: GL tag theorem.
  have row_35 := peano_source_005 N zero succ add mul one anchor relationalInduction
  -- chapter_8_line_33: GL tag task formulation.
  have row_33 : (add v1 v2 v3) := by
    exact assumption_33
  -- chapter_8_line_22: GL tag theorem.
  have row_22 := peano_source_030 N zero succ add mul one anchor relationalInduction
  -- chapter_8_line_20: GL tag recursion.
  have row_20 : (succ previous v1) := by
    exact assumption_20
  -- chapter_8_line_18: GL tag task formulation.
  have row_18 : (succ v4 v5) := by
    exact assumption_18
  -- chapter_8_line_12: GL tag task formulation.
  have row_12 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_8_line_11: GL tag expansion.
  have row_11 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_12
  -- chapter_8_line_10: GL tag disintegration.
  have row_10 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_11.1
  -- chapter_8_line_9: GL tag expansion.
  have row_9 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_10
  -- chapter_8_line_49: GL tag disintegration.
  have row_49 : (gl_implication21 N succ mul add) := by
    exact row_9.2
  -- chapter_8_line_48: GL tag expansion.
  have row_48 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((mul w3 w1 w4) → (∀ (w5 : α), ((mul w3 w2 w5) → (add w4 w3 w5))))))))) := by
    simpa only [gl_implication21] using row_49
  -- chapter_8_line_32: GL tag disintegration.
  have row_32 : (gl_fXYZ add N N N) := by
    exact row_9.1.1.1.1.1.1.1.1.2
  -- chapter_8_line_31: GL tag expansion.
  have row_31 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_32
  -- chapter_8_line_69: GL tag disintegration.
  have row_69 : (gl_implication13 N N N add) := by
    exact row_31.1.2
  -- chapter_8_line_68: GL tag expansion.
  have row_68 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication13] using row_69
  -- chapter_8_line_43: GL tag disintegration.
  have row_43 : (gl_implication9 add N) := by
    exact row_31.1.1.1.2
  -- chapter_8_line_42: GL tag expansion.
  have row_42 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication9] using row_43
  -- chapter_8_line_41: GL tag implication.
  have row_41 : (N v2) := by
    apply row_42
    exact row_33
  -- chapter_8_line_30: GL tag disintegration.
  have row_30 : (gl_implication10 add N) := by
    exact row_31.1.1.2
  -- chapter_8_line_29: GL tag expansion.
  have row_29 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication10] using row_30
  -- chapter_8_line_28: GL tag implication.
  have row_28 : (N v3) := by
    apply row_29
    exact row_33
  -- chapter_8_line_17: GL tag disintegration.
  have row_17 : (gl_fXY succ N N) := by
    exact row_9.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_8_line_16: GL tag expansion.
  have row_16 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_17
  -- chapter_8_line_59: GL tag disintegration.
  have row_59 : (gl_implication0 succ N) := by
    exact row_16.1.1.1
  -- chapter_8_line_58: GL tag expansion.
  have row_58 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_59
  -- chapter_8_line_57: GL tag implication.
  have row_57 : (N previous) := by
    apply row_58
    exact row_20
  -- chapter_8_line_27: GL tag disintegration.
  have row_27 : (gl_implication4 N N succ) := by
    exact row_16.1.2
  -- chapter_8_line_26: GL tag expansion.
  have row_26 : (∀ (w1 : α), ((N w1) → (gl_existence0 N w1 succ))) := by
    simpa only [gl_implication4] using row_27
  -- chapter_8_line_40: GL tag implication.
  have row_40 : (gl_existence0 N v2 succ) := by
    apply row_26
    exact row_41
  -- chapter_8_line_39: GL tag expansion.
  have row_39 : (¬ (∀ (v11 : α), ((N v11) → (¬ (succ v2 v11))))) := by
    simpa only [gl_existence0] using row_40
  have exists_row_39 : ∃ (v11 : α), ((N v11) ∧ (succ v2 v11)) := existsAndOfNotForallImpNot row_39
  obtain ⟨v11, witness_row_39⟩ := exists_row_39
  -- chapter_8_line_38: GL tag disintegration.
  have row_38 : (succ v2 v11) := by
    exact witness_row_39.2
  -- chapter_8_line_25: GL tag implication.
  have row_25 : (gl_existence0 N v3 succ) := by
    apply row_26
    exact row_28
  -- chapter_8_line_24: GL tag expansion.
  have row_24 : (¬ (∀ (v9 : α), ((N v9) → (¬ (succ v3 v9))))) := by
    simpa only [gl_existence0] using row_25
  have exists_row_24 : ∃ (v9 : α), ((N v9) ∧ (succ v3 v9)) := existsAndOfNotForallImpNot row_24
  obtain ⟨v9, witness_row_24⟩ := exists_row_24
  -- chapter_8_line_23: GL tag disintegration.
  have row_23 : (succ v3 v9) := by
    exact witness_row_24.2
  -- chapter_8_line_36: GL tag implication.
  have row_36 : (add v1 v11 v9) := by
    apply row_37
    exact row_33
    exact row_38
    exact row_23
  -- chapter_8_line_15: GL tag disintegration.
  have row_15 : (gl_implication1 succ N) := by
    exact row_16.1.1.2
  -- chapter_8_line_14: GL tag expansion.
  have row_14 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using row_15
  -- chapter_8_line_19: GL tag implication.
  have row_19 : (N v1) := by
    apply row_14
    exact row_20
  -- chapter_8_line_13: GL tag implication.
  have row_13 : (N v5) := by
    apply row_14
    exact row_18
  -- chapter_8_line_8: GL tag disintegration.
  have row_8 : (gl_fXYZ mul N N N) := by
    exact row_9.1.1.1.2
  -- chapter_8_line_7: GL tag expansion.
  have row_7 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_8
  -- chapter_8_line_73: GL tag disintegration.
  have row_73 : (gl_implication14 N N mul) := by
    exact row_7.2
  -- chapter_8_line_72: GL tag expansion.
  have row_72 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (∀ (w3 : α), ((mul w1 w2 w3) → (∀ (w4 : α), ((mul w1 w2 w4) → (w3 = w4))))))))) := by
    simpa only [gl_implication14] using row_73
  -- chapter_8_line_56: GL tag disintegration.
  have row_56 : (gl_implication8 mul N) := by
    exact row_7.1.1.1.1
  -- chapter_8_line_55: GL tag expansion.
  have row_55 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_56
  -- chapter_8_line_54: GL tag implication.
  have row_54 : (N v4) := by
    apply row_55
    exact row_50
  -- chapter_8_line_6: GL tag disintegration.
  have row_6 : (gl_implication13 N N N mul) := by
    exact row_7.1.2
  -- chapter_8_line_5: GL tag expansion.
  have row_5 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication13] using row_6
  -- chapter_8_line_63: GL tag implication.
  have row_63 : (gl_existence1 N v5 previous mul) := by
    apply row_5
    exact row_13
    exact row_57
  -- chapter_8_line_62: GL tag expansion.
  have row_62 : (¬ (∀ (v13 : α), ((N v13) → (¬ (mul v5 previous v13))))) := by
    simpa only [gl_existence1] using row_63
  have exists_row_62 : ∃ (v13 : α), ((N v13) ∧ (mul v5 previous v13)) := existsAndOfNotForallImpNot row_62
  obtain ⟨v13, witness_row_62⟩ := exists_row_62
  -- chapter_8_line_61: GL tag disintegration.
  have row_61 : (mul v5 previous v13) := by
    exact witness_row_62.2
  -- chapter_8_line_53: GL tag implication.
  have row_53 : (gl_existence1 N v4 previous mul) := by
    apply row_5
    exact row_54
    exact row_57
  -- chapter_8_line_52: GL tag expansion.
  have row_52 : (¬ (∀ (v12 : α), ((N v12) → (¬ (mul v4 previous v12))))) := by
    simpa only [gl_existence1] using row_53
  have exists_row_52 : ∃ (v12 : α), ((N v12) ∧ (mul v4 previous v12)) := existsAndOfNotForallImpNot row_52
  obtain ⟨v12, witness_row_52⟩ := exists_row_52
  -- chapter_8_line_70: GL tag disintegration.
  have row_70 : (N v12) := by
    exact witness_row_52.1
  -- chapter_8_line_67: GL tag implication.
  have row_67 : (gl_existence1 N previous v12 add) := by
    apply row_68
    exact row_57
    exact row_70
  -- chapter_8_line_66: GL tag expansion.
  have row_66 : (¬ (∀ (v16 : α), ((N v16) → (¬ (add previous v12 v16))))) := by
    simpa only [gl_existence1] using row_67
  have exists_row_66 : ∃ (v16 : α), ((N v16) ∧ (add previous v12 v16)) := existsAndOfNotForallImpNot row_66
  obtain ⟨v16, witness_row_66⟩ := exists_row_66
  -- chapter_8_line_65: GL tag disintegration.
  have row_65 : (add previous v12 v16) := by
    exact witness_row_66.2
  -- chapter_8_line_51: GL tag disintegration.
  have row_51 : (mul v4 previous v12) := by
    exact witness_row_52.2
  -- chapter_8_line_74: GL tag implication.
  have row_74 : (mul v5 previous v16) := by
    apply row_75
    exact row_65
    exact row_18
    exact row_51
  -- chapter_8_line_71: GL tag implication.
  have row_71 : (v16 = v13) := by
    apply row_72
    exact row_13
    exact row_57
    exact row_74
    exact row_61
  -- chapter_8_line_64: GL tag equality1.
  have row_64 : (add previous v12 v13) := by
    have equality_source := row_65
    have equality_step_1 := row_71
    cases equality_step_1
    exact equality_source
  -- chapter_8_line_47: GL tag implication.
  have row_47 : (add v12 v4 v2) := by
    apply row_48
    exact row_57
    exact row_20
    exact row_51
    exact row_50
  -- chapter_8_line_46: GL tag implication.
  have row_46 : (add v12 v5 v11) := by
    apply row_37
    exact row_47
    exact row_18
    exact row_38
  -- chapter_8_line_4: GL tag implication.
  have row_4 : (gl_existence1 N v5 v1 mul) := by
    apply row_5
    exact row_13
    exact row_19
  -- chapter_8_line_3: GL tag expansion.
  have row_3 : (¬ (∀ (v6 : α), ((N v6) → (¬ (mul v5 v1 v6))))) := by
    simpa only [gl_existence1] using row_4
  have exists_row_3 : ∃ (v6 : α), ((N v6) ∧ (mul v5 v1 v6)) := existsAndOfNotForallImpNot row_3
  obtain ⟨v6, witness_row_3⟩ := exists_row_3
  -- chapter_8_line_2: GL tag disintegration.
  have row_2 : (mul v5 v1 v6) := by
    exact witness_row_3.2
  -- chapter_8_line_60: GL tag implication.
  have row_60 : (add v13 v5 v6) := by
    apply row_48
    exact row_57
    exact row_20
    exact row_61
    exact row_2
  -- chapter_8_line_44: GL tag implication.
  have row_44 : (add previous v11 v6) := by
    apply row_45
    exact row_60
    exact row_64
    exact row_46
  -- chapter_8_line_34: GL tag implication.
  have row_34 : (succ v6 v9) := by
    apply row_35
    exact row_36
    exact row_44
    exact row_20
  -- chapter_8_line_21: GL tag implication.
  have row_21 : (v6 = v3) := by
    apply row_22
    exact row_34
    exact row_23
  -- chapter_8_line_1: GL tag equality1.
  have row_1 : (mul v5 v1 v3) := by
    have equality_source := row_2
    have equality_step_1 := row_21
    cases equality_step_1
    exact equality_source
  exact row_1

theorem peano_source_002
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α) (v5 : α), ((succ v4 v5) → ((mul v4 v1 v2) → (mul v5 v1 v3)))))) := by
  intro v1
  intro v2
  intro v3
  intro premise_1
  intro v4
  intro v5
  intro premise_2
  intro premise_3
  have inductionMember : N v1 := by
    have typingRule := peano_source_002_induction_typing N zero succ add mul one anchor relationalInduction
    exact typingRule v1 v2 v4 premise_3
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v2 : α) (v3 : α), ((add zero v2 v3) → (∀ (v4 : α) (v5 : α), ((succ v4 v5) → ((mul v4 zero v2) → (mul v5 zero v3)))))) := by
    intro v2
    intro v3
    intro base_premise_1
    intro v4
    intro v5
    intro base_premise_2
    intro base_premise_3
    have zeroRule := peano_source_002_check_zero N zero succ add mul one anchor relationalInduction
    exact zeroRule zero v2 v3 v4 v5 base_premise_3 rfl base_premise_1 base_premise_2
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v2 : α) (v3 : α), ((add induction_n v2 v3) → (∀ (v4 : α) (v5 : α), ((succ v4 v5) → ((mul v4 induction_n v2) → (mul v5 induction_n v3)))))) → ∀ induction_m, succ induction_n induction_m → (∀ (v2 : α) (v3 : α), ((add induction_m v2 v3) → (∀ (v4 : α) (v5 : α), ((succ v4 v5) → ((mul v4 induction_m v2) → (mul v5 induction_m v3)))))) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro v2
    intro v3
    intro step_premise_1
    intro v4
    intro v5
    intro step_premise_2
    intro step_premise_3
    have step_induction_assumption_1 :
        (∀ (w1 : α) (w2 : α), ((add induction_n w1 w2) → ((succ v4 v5) → ((mul v4 induction_n w1) → (mul v5 induction_n w2))))) := by
      intro w1
      intro w2
      intro step_induction_assumption_1_premise_1
      intro step_induction_assumption_1_premise_2
      intro step_induction_assumption_1_premise_3
      apply induction_hypothesis
      all_goals assumption
    have step_induction_assumption_2 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have stepRule := peano_source_002_check_induction_condition N zero succ add mul one anchor relationalInduction
    exact stepRule induction_n induction_m v2 v3 v4 v5 step_induction_assumption_1 step_premise_3 step_premise_1 step_induction_assumption_2 step_premise_2
  have inductionProperty : (∀ (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α) (v5 : α), ((succ v4 v5) → ((mul v4 v1 v2) → (mul v5 v1 v3)))))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v2 : α) (v3 : α), ((add induction_value v2 v3) → (∀ (v4 : α) (v5 : α), ((succ v4 v5) → ((mul v4 induction_value v2) → (mul v5 induction_value v3)))))))
      v1
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v2 v3 premise_1 v4 v5 premise_2 premise_3

private theorem peano_source_010_induction_typing
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (v5 : α)
    (assumption_10 : (succ v5 v1))
    : (N v5) := by
  -- chapter_26_line_10: GL tag task formulation.
  have row_10 : (succ v5 v1) := by
    exact assumption_10
  -- chapter_26_line_9: GL tag task formulation.
  have row_9 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_26_line_8: GL tag expansion.
  have row_8 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_9
  -- chapter_26_line_7: GL tag disintegration.
  have row_7 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_8.1
  -- chapter_26_line_6: GL tag expansion.
  have row_6 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_7
  -- chapter_26_line_5: GL tag disintegration.
  have row_5 : (gl_fXY succ N N) := by
    exact row_6.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_26_line_4: GL tag expansion.
  have row_4 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_5
  -- chapter_26_line_3: GL tag disintegration.
  have row_3 : (gl_implication0 succ N) := by
    exact row_4.1.1.1
  -- chapter_26_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_3
  -- chapter_26_line_1: GL tag implication.
  have row_1 : (N v5) := by
    apply row_2
    exact row_10
  exact row_1

private theorem peano_source_010_check_zero
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (v4 : α)
    (v5 : α)
    (assumption_50 : (v5 = zero))
    (assumption_20 : (add v1 v2 v3))
    (assumption_6 : (succ v2 v4))
    (assumption_5 : (succ v5 v1))
    : (add v4 v5 v3) := by
  -- chapter_27_line_59: GL tag theorem.
  have row_59 := peano_source_008 N zero succ add mul one anchor relationalInduction
  -- chapter_27_line_56: GL tag theorem.
  have row_56 := peano_source_055 N zero succ add mul one anchor relationalInduction
  -- chapter_27_line_50: GL tag recursion.
  have row_50 : (v5 = zero) := by
    exact assumption_50
  -- chapter_27_line_49: GL tag symmetry of equality.
  have row_49 : (zero = v5) := by
    exact Eq.symm row_50
  -- chapter_27_line_20: GL tag task formulation.
  have row_20 : (add v1 v2 v3) := by
    exact assumption_20
  -- chapter_27_line_6: GL tag task formulation.
  have row_6 : (succ v2 v4) := by
    exact assumption_6
  -- chapter_27_line_5: GL tag task formulation.
  have row_5 : (succ v5 v1) := by
    exact assumption_5
  -- chapter_27_line_4: GL tag task formulation.
  have row_4 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_27_line_58: GL tag implication.
  have row_58 : (add v5 v4 v3) := by
    apply row_59
    exact row_20
    exact row_5
    exact row_6
  -- chapter_27_line_57: GL tag equality1.
  have row_57 : (add zero v4 v3) := by
    have equality_source := row_58
    have equality_step_1 := row_50
    cases equality_step_1
    exact equality_source
  -- chapter_27_line_16: GL tag expansion.
  have row_16 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_4
  -- chapter_27_line_15: GL tag disintegration.
  have row_15 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_16.1
  -- chapter_27_line_14: GL tag expansion.
  have row_14 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_15
  -- chapter_27_line_38: GL tag disintegration.
  have row_38 : (gl_fXY succ N N) := by
    exact row_14.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_27_line_37: GL tag expansion.
  have row_37 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_38
  -- chapter_27_line_54: GL tag disintegration.
  have row_54 : (gl_implication0 succ N) := by
    exact row_37.1.1.1
  -- chapter_27_line_53: GL tag expansion.
  have row_53 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_54
  -- chapter_27_line_52: GL tag implication.
  have row_52 : (N v5) := by
    apply row_53
    exact row_5
  -- chapter_27_line_46: GL tag disintegration.
  have row_46 : (gl_implication1 succ N) := by
    exact row_37.1.1.2
  -- chapter_27_line_45: GL tag expansion.
  have row_45 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using row_46
  -- chapter_27_line_44: GL tag implication.
  have row_44 : (N v4) := by
    apply row_45
    exact row_6
  -- chapter_27_line_55: GL tag implication.
  have row_55 : (v4 = v3) := by
    apply row_56
    exact row_44
    exact row_57
  -- chapter_27_line_43: GL tag disintegration.
  have row_43 : (gl_implication4 N N succ) := by
    exact row_37.1.2
  -- chapter_27_line_42: GL tag expansion.
  have row_42 : (∀ (w1 : α), ((N w1) → (gl_existence0 N w1 succ))) := by
    simpa only [gl_implication4] using row_43
  -- chapter_27_line_36: GL tag disintegration.
  have row_36 : (gl_implication5 N succ) := by
    exact row_37.2
  -- chapter_27_line_35: GL tag expansion.
  have row_35 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α), ((succ w1 w3) → (w2 = w3))))))) := by
    simpa only [gl_implication5] using row_36
  -- chapter_27_line_33: GL tag disintegration.
  have row_33 : (gl_implication16 N zero add) := by
    exact row_14.1.1.1.1.1.1.2
  -- chapter_27_line_32: GL tag expansion.
  have row_32 : (∀ (w1 : α) (w2 : α), ((w1 = w2) → ((N w1) → ((N w2) → (add w1 zero w2))))) := by
    simpa only [gl_implication16] using row_33
  -- chapter_27_line_13: GL tag disintegration.
  have row_13 : (gl_fXYZ add N N N) := by
    exact row_14.1.1.1.1.1.1.1.1.2
  -- chapter_27_line_12: GL tag expansion.
  have row_12 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_13
  -- chapter_27_line_27: GL tag disintegration.
  have row_27 : (gl_implication14 N N add) := by
    exact row_12.2
  -- chapter_27_line_26: GL tag expansion.
  have row_26 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (∀ (w3 : α), ((add w1 w2 w3) → (∀ (w4 : α), ((add w1 w2 w4) → (w3 = w4))))))))) := by
    simpa only [gl_implication14] using row_27
  -- chapter_27_line_23: GL tag disintegration.
  have row_23 : (gl_implication9 add N) := by
    exact row_12.1.1.1.2
  -- chapter_27_line_22: GL tag expansion.
  have row_22 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication9] using row_23
  -- chapter_27_line_21: GL tag implication.
  have row_21 : (N v2) := by
    apply row_22
    exact row_20
  -- chapter_27_line_41: GL tag implication.
  have row_41 : (gl_existence0 N v2 succ) := by
    apply row_42
    exact row_21
  -- chapter_27_line_40: GL tag expansion.
  have row_40 : (¬ (∀ (v11 : α), ((N v11) → (¬ (succ v2 v11))))) := by
    simpa only [gl_existence0] using row_41
  have exists_row_40 : ∃ (v11 : α), ((N v11) ∧ (succ v2 v11)) := existsAndOfNotForallImpNot row_40
  obtain ⟨v11, witness_row_40⟩ := exists_row_40
  -- chapter_27_line_47: GL tag disintegration.
  have row_47 : (N v11) := by
    exact witness_row_40.1
  -- chapter_27_line_39: GL tag disintegration.
  have row_39 : (succ v2 v11) := by
    exact witness_row_40.2
  -- chapter_27_line_48: GL tag implication.
  have row_48 : (v11 = v4) := by
    apply row_35
    exact row_21
    exact row_39
    exact row_6
  -- chapter_27_line_34: GL tag implication.
  have row_34 : (v4 = v11) := by
    apply row_35
    exact row_21
    exact row_6
    exact row_39
  -- chapter_27_line_31: GL tag implication.
  have row_31 : (add v4 zero v11) := by
    apply row_32
    exact row_34
    exact row_44
    exact row_47
  -- chapter_27_line_30: GL tag equality1.
  have row_30 : (add v4 zero v4) := by
    have equality_source := row_31
    have equality_step_1 := row_48
    cases equality_step_1
    exact equality_source
  -- chapter_27_line_29: GL tag equality1.
  have row_29 : (add v4 v5 v4) := by
    have equality_source := row_30
    have equality_step_1 := row_49
    cases equality_step_1
    exact equality_source
  -- chapter_27_line_28: GL tag equality1.
  have row_28 : (add v11 v5 v4) := by
    have equality_source := row_29
    have equality_step_1 := row_34
    cases equality_step_1
    exact equality_source
  -- chapter_27_line_19: GL tag disintegration.
  have row_19 : (gl_implication8 add N) := by
    exact row_12.1.1.1.1
  -- chapter_27_line_18: GL tag expansion.
  have row_18 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_19
  -- chapter_27_line_17: GL tag implication.
  have row_17 : (N v1) := by
    apply row_18
    exact row_20
  -- chapter_27_line_11: GL tag disintegration.
  have row_11 : (gl_implication13 N N N add) := by
    exact row_12.1.2
  -- chapter_27_line_10: GL tag expansion.
  have row_10 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication13] using row_11
  -- chapter_27_line_9: GL tag implication.
  have row_9 : (gl_existence1 N v2 v1 add) := by
    apply row_10
    exact row_21
    exact row_17
  -- chapter_27_line_8: GL tag expansion.
  have row_8 : (¬ (∀ (v6 : α), ((N v6) → (¬ (add v2 v1 v6))))) := by
    simpa only [gl_existence1] using row_9
  have exists_row_8 : ∃ (v6 : α), ((N v6) ∧ (add v2 v1 v6)) := existsAndOfNotForallImpNot row_8
  obtain ⟨v6, witness_row_8⟩ := exists_row_8
  -- chapter_27_line_7: GL tag disintegration.
  have row_7 : (add v2 v1 v6) := by
    exact witness_row_8.2
  have rule_row_51 := peano_source_009 N zero succ add mul one anchor relationalInduction
  -- chapter_27_line_51: GL tag implication.
  have row_51 : (add v11 v5 v6) := by
    apply rule_row_51
    exact row_7
    exact row_39
    exact row_5
  -- chapter_27_line_25: GL tag implication.
  have row_25 : (v6 = v4) := by
    apply row_26
    exact row_47
    exact row_52
    exact row_51
    exact row_28
  -- chapter_27_line_24: GL tag equality2.
  have row_24 : (v6 = v3) := by
    exact Eq.trans row_25 row_55
  -- chapter_27_line_3: GL tag theorem.
  have row_3 := peano_source_009 N zero succ add mul one anchor relationalInduction
  -- chapter_27_line_2: GL tag implication.
  have row_2 : (add v4 v5 v6) := by
    apply row_3
    exact row_7
    exact row_6
    exact row_5
  -- chapter_27_line_1: GL tag equality1.
  have row_1 : (add v4 v5 v3) := by
    have equality_source := row_2
    have equality_step_1 := row_24
    cases equality_step_1
    exact equality_source
  exact row_1

private theorem peano_source_010_check_induction_condition
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (v4 : α)
    (v5 : α)
    (assumption_41 : (∀ (w1 : α) (w2 : α), ((add w1 v2 w2) → ((succ v2 v4) → ((succ previous w1) → (add v4 previous w2))))))
    (assumption_36 : (succ v2 v4))
    (assumption_28 : (add v1 v2 v3))
    (assumption_24 : (succ previous v5))
    (assumption_10 : (succ v5 v1))
    : (add v4 v5 v3) := by
  -- chapter_28_line_41: GL tag recursion.
  have row_41 : (∀ (w1 : α) (w2 : α), ((add w1 v2 w2) → ((succ v2 v4) → ((succ previous w1) → (add v4 previous w2))))) := by
    exact assumption_41
  -- chapter_28_line_36: GL tag task formulation.
  have row_36 : (succ v2 v4) := by
    exact assumption_36
  -- chapter_28_line_28: GL tag task formulation.
  have row_28 : (add v1 v2 v3) := by
    exact assumption_28
  -- chapter_28_line_24: GL tag recursion.
  have row_24 : (succ previous v5) := by
    exact assumption_24
  -- chapter_28_line_10: GL tag task formulation.
  have row_10 : (succ v5 v1) := by
    exact assumption_10
  -- chapter_28_line_9: GL tag theorem.
  have row_9 := peano_source_005 N zero succ add mul one anchor relationalInduction
  -- chapter_28_line_7: GL tag task formulation.
  have row_7 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_28_line_6: GL tag expansion.
  have row_6 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_7
  -- chapter_28_line_5: GL tag disintegration.
  have row_5 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_6.1
  -- chapter_28_line_4: GL tag expansion.
  have row_4 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_5
  -- chapter_28_line_23: GL tag disintegration.
  have row_23 : (gl_fXY succ N N) := by
    exact row_4.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_28_line_22: GL tag expansion.
  have row_22 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_23
  -- chapter_28_line_39: GL tag disintegration.
  have row_39 : (gl_implication0 succ N) := by
    exact row_22.1.1.1
  -- chapter_28_line_38: GL tag expansion.
  have row_38 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_39
  -- chapter_28_line_37: GL tag implication.
  have row_37 : (N previous) := by
    apply row_38
    exact row_24
  -- chapter_28_line_21: GL tag disintegration.
  have row_21 : (gl_implication1 succ N) := by
    exact row_22.1.1.2
  -- chapter_28_line_20: GL tag expansion.
  have row_20 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using row_21
  -- chapter_28_line_35: GL tag implication.
  have row_35 : (N v4) := by
    apply row_20
    exact row_36
  -- chapter_28_line_19: GL tag implication.
  have row_19 : (N v5) := by
    apply row_20
    exact row_24
  -- chapter_28_line_18: GL tag disintegration.
  have row_18 : (gl_fXYZ add N N N) := by
    exact row_4.1.1.1.1.1.1.1.1.2
  -- chapter_28_line_17: GL tag expansion.
  have row_17 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_18
  -- chapter_28_line_31: GL tag disintegration.
  have row_31 : (gl_implication14 N N add) := by
    exact row_17.2
  -- chapter_28_line_30: GL tag expansion.
  have row_30 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (∀ (w3 : α), ((add w1 w2 w3) → (∀ (w4 : α), ((add w1 w2 w4) → (w3 = w4))))))))) := by
    simpa only [gl_implication14] using row_31
  -- chapter_28_line_27: GL tag disintegration.
  have row_27 : (gl_implication9 add N) := by
    exact row_17.1.1.1.2
  -- chapter_28_line_26: GL tag expansion.
  have row_26 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication9] using row_27
  -- chapter_28_line_25: GL tag implication.
  have row_25 : (N v2) := by
    apply row_26
    exact row_28
  -- chapter_28_line_16: GL tag disintegration.
  have row_16 : (gl_implication13 N N N add) := by
    exact row_17.1.2
  -- chapter_28_line_15: GL tag expansion.
  have row_15 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication13] using row_16
  -- chapter_28_line_34: GL tag implication.
  have row_34 : (gl_existence1 N v4 previous add) := by
    apply row_15
    exact row_35
    exact row_37
  -- chapter_28_line_33: GL tag expansion.
  have row_33 : (¬ (∀ (v11 : α), ((N v11) → (¬ (add v4 previous v11))))) := by
    simpa only [gl_existence1] using row_34
  have exists_row_33 : ∃ (v11 : α), ((N v11) ∧ (add v4 previous v11)) := existsAndOfNotForallImpNot row_33
  obtain ⟨v11, witness_row_33⟩ := exists_row_33
  -- chapter_28_line_32: GL tag disintegration.
  have row_32 : (add v4 previous v11) := by
    exact witness_row_33.2
  -- chapter_28_line_14: GL tag implication.
  have row_14 : (gl_existence1 N v5 v2 add) := by
    apply row_15
    exact row_19
    exact row_25
  -- chapter_28_line_13: GL tag expansion.
  have row_13 : (¬ (∀ (v12 : α), ((N v12) → (¬ (add v5 v2 v12))))) := by
    simpa only [gl_existence1] using row_14
  have exists_row_13 : ∃ (v12 : α), ((N v12) ∧ (add v5 v2 v12)) := existsAndOfNotForallImpNot row_13
  obtain ⟨v12, witness_row_13⟩ := exists_row_13
  -- chapter_28_line_12: GL tag disintegration.
  have row_12 : (add v5 v2 v12) := by
    exact witness_row_13.2
  -- chapter_28_line_40: GL tag implication.
  have row_40 : (add v4 previous v12) := by
    apply row_41
    exact row_12
    exact row_36
    exact row_24
  -- chapter_28_line_29: GL tag implication.
  have row_29 : (v12 = v11) := by
    apply row_30
    exact row_35
    exact row_37
    exact row_40
    exact row_32
  -- chapter_28_line_11: GL tag equality1.
  have row_11 : (add v5 v2 v11) := by
    have equality_source := row_12
    have equality_step_1 := row_29
    cases equality_step_1
    exact equality_source
  -- chapter_28_line_8: GL tag implication.
  have row_8 : (succ v11 v3) := by
    apply row_9
    exact row_28
    exact row_11
    exact row_10
  -- chapter_28_line_3: GL tag disintegration.
  have row_3 : (gl_implication18 N succ add) := by
    exact row_4.1.1.1.1.2
  -- chapter_28_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((succ w4 w5) → (add w3 w2 w5))))))))) := by
    simpa only [gl_implication18] using row_3
  -- chapter_28_line_1: GL tag implication.
  have row_1 : (add v4 v5 v3) := by
    apply row_2
    exact row_37
    exact row_24
    exact row_32
    exact row_8
  exact row_1

theorem peano_source_010
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α), ((succ v2 v4) → (∀ (v5 : α), ((succ v5 v1) → (add v4 v5 v3))))))) := by
  intro v1
  intro v2
  intro v3
  intro premise_1
  intro v4
  intro premise_2
  intro v5
  intro premise_3
  have inductionMember : N v5 := by
    have typingRule := peano_source_010_induction_typing N zero succ add mul one anchor relationalInduction
    exact typingRule v1 v5 premise_3
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α), ((succ v2 v4) → ((succ zero v1) → (add v4 zero v3)))))) := by
    intro v1
    intro v2
    intro v3
    intro base_premise_1
    intro v4
    intro base_premise_2
    intro base_premise_3
    have zeroRule := peano_source_010_check_zero N zero succ add mul one anchor relationalInduction
    exact zeroRule v1 v2 v3 v4 zero rfl base_premise_1 base_premise_2 base_premise_3
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α), ((succ v2 v4) → ((succ induction_n v1) → (add v4 induction_n v3)))))) → ∀ induction_m, succ induction_n induction_m → (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α), ((succ v2 v4) → ((succ induction_m v1) → (add v4 induction_m v3)))))) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro v1
    intro v2
    intro v3
    intro step_premise_1
    intro v4
    intro step_premise_2
    intro step_premise_3
    have step_induction_assumption_1 :
        (∀ (w1 : α) (w2 : α), ((add w1 v2 w2) → ((succ v2 v4) → ((succ induction_n w1) → (add v4 induction_n w2))))) := by
      intro w1
      intro w2
      intro step_induction_assumption_1_premise_1
      intro step_induction_assumption_1_premise_2
      intro step_induction_assumption_1_premise_3
      apply induction_hypothesis
      all_goals assumption
    have step_induction_assumption_2 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have stepRule := peano_source_010_check_induction_condition N zero succ add mul one anchor relationalInduction
    exact stepRule induction_n v1 v2 v3 v4 induction_m step_induction_assumption_1 step_premise_2 step_premise_1 step_induction_assumption_2 step_premise_3
  have inductionProperty : (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α), ((succ v2 v4) → ((succ v5 v1) → (add v4 v5 v3)))))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α), ((succ v2 v4) → ((succ induction_value v1) → (add v4 induction_value v3)))))))
      v5
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v1 v2 v3 premise_1 v4 premise_2 premise_3

private theorem peano_source_013_induction_typing
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (v3 : α)
    (v4 : α)
    (assumption_10 : (add v1 v4 v3))
    : (N v1) := by
  -- chapter_33_line_10: GL tag task formulation.
  have row_10 : (add v1 v4 v3) := by
    exact assumption_10
  -- chapter_33_line_9: GL tag task formulation.
  have row_9 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_33_line_8: GL tag expansion.
  have row_8 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_9
  -- chapter_33_line_7: GL tag disintegration.
  have row_7 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_8.1
  -- chapter_33_line_6: GL tag expansion.
  have row_6 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_7
  -- chapter_33_line_5: GL tag disintegration.
  have row_5 : (gl_fXYZ add N N N) := by
    exact row_6.1.1.1.1.1.1.1.1.2
  -- chapter_33_line_4: GL tag expansion.
  have row_4 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_5
  -- chapter_33_line_3: GL tag disintegration.
  have row_3 : (gl_implication8 add N) := by
    exact row_4.1.1.1.1
  -- chapter_33_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_3
  -- chapter_33_line_1: GL tag implication.
  have row_1 : (N v1) := by
    apply row_2
    exact row_10
  exact row_1

private theorem peano_source_013_check_zero
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (v4 : α)
    (assumption_20 : (add v1 v2 v3))
    (assumption_8 : (v1 = zero))
    (assumption_7 : (add v1 v4 v3))
    : (v2 = v4) := by
  -- chapter_34_line_20: GL tag task formulation.
  have row_20 : (add v1 v2 v3) := by
    exact assumption_20
  -- chapter_34_line_8: GL tag recursion.
  have row_8 : (v1 = zero) := by
    exact assumption_8
  -- chapter_34_line_19: GL tag equality1.
  have row_19 : (add zero v2 v3) := by
    have equality_source := row_20
    have equality_step_1 := row_8
    cases equality_step_1
    exact equality_source
  -- chapter_34_line_7: GL tag task formulation.
  have row_7 : (add v1 v4 v3) := by
    exact assumption_7
  -- chapter_34_line_6: GL tag equality1.
  have row_6 : (add zero v4 v3) := by
    have equality_source := row_7
    have equality_step_1 := row_8
    cases equality_step_1
    exact equality_source
  -- chapter_34_line_5: GL tag task formulation.
  have row_5 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_34_line_16: GL tag expansion.
  have row_16 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_5
  -- chapter_34_line_15: GL tag disintegration.
  have row_15 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_16.1
  -- chapter_34_line_14: GL tag expansion.
  have row_14 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_15
  -- chapter_34_line_13: GL tag disintegration.
  have row_13 : (gl_fXYZ add N N N) := by
    exact row_14.1.1.1.1.1.1.1.1.2
  -- chapter_34_line_12: GL tag expansion.
  have row_12 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_13
  -- chapter_34_line_11: GL tag disintegration.
  have row_11 : (gl_implication9 add N) := by
    exact row_12.1.1.1.2
  -- chapter_34_line_10: GL tag expansion.
  have row_10 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication9] using row_11
  -- chapter_34_line_21: GL tag implication.
  have row_21 : (N v2) := by
    apply row_10
    exact row_20
  have rule_row_18 := peano_source_055 N zero succ add mul one anchor relationalInduction
  -- chapter_34_line_18: GL tag implication.
  have row_18 : (v2 = v3) := by
    apply rule_row_18
    exact row_21
    exact row_19
  -- chapter_34_line_17: GL tag symmetry of equality.
  have row_17 : (v3 = v2) := by
    exact Eq.symm row_18
  -- chapter_34_line_9: GL tag implication.
  have row_9 : (N v4) := by
    apply row_10
    exact row_7
  -- chapter_34_line_4: GL tag theorem.
  have row_4 := peano_source_055 N zero succ add mul one anchor relationalInduction
  -- chapter_34_line_3: GL tag implication.
  have row_3 : (v4 = v3) := by
    apply row_4
    exact row_9
    exact row_6
  -- chapter_34_line_2: GL tag equality2.
  have row_2 : (v4 = v2) := by
    exact Eq.trans row_3 row_17
  -- chapter_34_line_1: GL tag symmetry of equality.
  have row_1 : (v2 = v4) := by
    exact Eq.symm row_2
  exact row_1

private theorem peano_source_013_check_induction_condition
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (v4 : α)
    (assumption_32 : (add v1 v2 v3))
    (assumption_25 : (succ previous v1))
    (assumption_18 : (add v1 v4 v3))
    (assumption_2 : (∀ (w1 : α), ((add previous v2 w1) → ((add previous v4 w1) → (v2 = v4)))))
    : (v2 = v4) := by
  -- chapter_35_line_36: GL tag theorem.
  have row_36 := peano_source_005 N zero succ add mul one anchor relationalInduction
  -- chapter_35_line_34: GL tag theorem.
  have row_34 := peano_source_030 N zero succ add mul one anchor relationalInduction
  -- chapter_35_line_32: GL tag task formulation.
  have row_32 : (add v1 v2 v3) := by
    exact assumption_32
  -- chapter_35_line_25: GL tag recursion.
  have row_25 : (succ previous v1) := by
    exact assumption_25
  -- chapter_35_line_19: GL tag variable copy.
  have row_19 : (v3 = v3) := by
    rfl
  -- chapter_35_line_31: GL tag equality1.
  have row_31 : (add v1 v2 v3) := by
    have equality_source := row_32
    have equality_step_1 := row_19
    cases equality_step_1
    exact equality_source
  -- chapter_35_line_18: GL tag task formulation.
  have row_18 : (add v1 v4 v3) := by
    exact assumption_18
  -- chapter_35_line_17: GL tag equality1.
  have row_17 : (add v1 v4 v3) := by
    have equality_source := row_18
    have equality_step_1 := row_19
    cases equality_step_1
    exact equality_source
  -- chapter_35_line_13: GL tag task formulation.
  have row_13 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_35_line_12: GL tag expansion.
  have row_12 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_13
  -- chapter_35_line_11: GL tag disintegration.
  have row_11 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_12.1
  -- chapter_35_line_10: GL tag expansion.
  have row_10 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_11
  -- chapter_35_line_24: GL tag disintegration.
  have row_24 : (gl_fXY succ N N) := by
    exact row_10.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_35_line_23: GL tag expansion.
  have row_23 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_24
  -- chapter_35_line_22: GL tag disintegration.
  have row_22 : (gl_implication0 succ N) := by
    exact row_23.1.1.1
  -- chapter_35_line_21: GL tag expansion.
  have row_21 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_22
  -- chapter_35_line_20: GL tag implication.
  have row_20 : (N previous) := by
    apply row_21
    exact row_25
  -- chapter_35_line_9: GL tag disintegration.
  have row_9 : (gl_fXYZ add N N N) := by
    exact row_10.1.1.1.1.1.1.1.1.2
  -- chapter_35_line_8: GL tag expansion.
  have row_8 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_9
  -- chapter_35_line_16: GL tag disintegration.
  have row_16 : (gl_implication9 add N) := by
    exact row_8.1.1.1.2
  -- chapter_35_line_15: GL tag expansion.
  have row_15 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication9] using row_16
  -- chapter_35_line_30: GL tag implication.
  have row_30 : (N v2) := by
    apply row_15
    exact row_31
  -- chapter_35_line_14: GL tag implication.
  have row_14 : (N v4) := by
    apply row_15
    exact row_17
  -- chapter_35_line_7: GL tag disintegration.
  have row_7 : (gl_implication13 N N N add) := by
    exact row_8.1.2
  -- chapter_35_line_6: GL tag expansion.
  have row_6 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication13] using row_7
  -- chapter_35_line_29: GL tag implication.
  have row_29 : (gl_existence1 N previous v2 add) := by
    apply row_6
    exact row_20
    exact row_30
  -- chapter_35_line_28: GL tag expansion.
  have row_28 : (¬ (∀ (v11 : α), ((N v11) → (¬ (add previous v2 v11))))) := by
    simpa only [gl_existence1] using row_29
  have exists_row_28 : ∃ (v11 : α), ((N v11) ∧ (add previous v2 v11)) := existsAndOfNotForallImpNot row_28
  obtain ⟨v11, witness_row_28⟩ := exists_row_28
  -- chapter_35_line_27: GL tag disintegration.
  have row_27 : (add previous v2 v11) := by
    exact witness_row_28.2
  -- chapter_35_line_37: GL tag implication.
  have row_37 : (succ v11 v3) := by
    apply row_36
    exact row_32
    exact row_27
    exact row_25
  -- chapter_35_line_5: GL tag implication.
  have row_5 : (gl_existence1 N previous v4 add) := by
    apply row_6
    exact row_20
    exact row_14
  -- chapter_35_line_4: GL tag expansion.
  have row_4 : (¬ (∀ (v6 : α), ((N v6) → (¬ (add previous v4 v6))))) := by
    simpa only [gl_existence1] using row_5
  have exists_row_4 : ∃ (v6 : α), ((N v6) ∧ (add previous v4 v6)) := existsAndOfNotForallImpNot row_4
  obtain ⟨v6, witness_row_4⟩ := exists_row_4
  -- chapter_35_line_3: GL tag disintegration.
  have row_3 : (add previous v4 v6) := by
    exact witness_row_4.2
  -- chapter_35_line_35: GL tag implication.
  have row_35 : (succ v6 v3) := by
    apply row_36
    exact row_18
    exact row_3
    exact row_25
  -- chapter_35_line_33: GL tag implication.
  have row_33 : (v11 = v6) := by
    apply row_34
    exact row_37
    exact row_35
  -- chapter_35_line_26: GL tag equality1.
  have row_26 : (add previous v2 v6) := by
    have equality_source := row_27
    have equality_step_1 := row_33
    cases equality_step_1
    exact equality_source
  -- chapter_35_line_2: GL tag recursion.
  have row_2 : (∀ (w1 : α), ((add previous v2 w1) → ((add previous v4 w1) → (v2 = v4)))) := by
    exact assumption_2
  -- chapter_35_line_1: GL tag implication.
  have row_1 : (v2 = v4) := by
    apply row_2
    exact row_26
    exact row_3
  exact row_1

theorem peano_source_013
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α), ((add v1 v4 v3) → (v2 = v4))))) := by
  intro v1
  intro v2
  intro v3
  intro premise_1
  intro v4
  intro premise_2
  have inductionMember : N v1 := by
    have typingRule := peano_source_013_induction_typing N zero succ add mul one anchor relationalInduction
    exact typingRule v1 v3 v4 premise_2
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v2 : α) (v3 : α), ((add zero v2 v3) → (∀ (v4 : α), ((add zero v4 v3) → (v2 = v4))))) := by
    intro v2
    intro v3
    intro base_premise_1
    intro v4
    intro base_premise_2
    have zeroRule := peano_source_013_check_zero N zero succ add mul one anchor relationalInduction
    exact zeroRule zero v2 v3 v4 base_premise_1 rfl base_premise_2
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v2 : α) (v3 : α), ((add induction_n v2 v3) → (∀ (v4 : α), ((add induction_n v4 v3) → (v2 = v4))))) → ∀ induction_m, succ induction_n induction_m → (∀ (v2 : α) (v3 : α), ((add induction_m v2 v3) → (∀ (v4 : α), ((add induction_m v4 v3) → (v2 = v4))))) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro v2
    intro v3
    intro step_premise_1
    intro v4
    intro step_premise_2
    have step_induction_assumption_1 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have step_induction_assumption_2 :
        (∀ (w1 : α), ((add induction_n v2 w1) → ((add induction_n v4 w1) → (v2 = v4)))) := by
      intro w1
      intro step_induction_assumption_2_premise_1
      intro step_induction_assumption_2_premise_2
      apply induction_hypothesis
      all_goals assumption
    have stepRule := peano_source_013_check_induction_condition N zero succ add mul one anchor relationalInduction
    exact stepRule induction_n induction_m v2 v3 v4 step_premise_1 step_induction_assumption_1 step_premise_2 step_induction_assumption_2
  have inductionProperty : (∀ (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α), ((add v1 v4 v3) → (v2 = v4))))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v2 : α) (v3 : α), ((add induction_value v2 v3) → (∀ (v4 : α), ((add induction_value v4 v3) → (v2 = v4))))))
      v1
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v2 v3 premise_1 v4 premise_2

private theorem peano_source_014_induction_typing
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (assumption_10 : (add v1 v2 v3))
    : (N v1) := by
  -- chapter_36_line_10: GL tag task formulation.
  have row_10 : (add v1 v2 v3) := by
    exact assumption_10
  -- chapter_36_line_9: GL tag task formulation.
  have row_9 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_36_line_8: GL tag expansion.
  have row_8 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_9
  -- chapter_36_line_7: GL tag disintegration.
  have row_7 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_8.1
  -- chapter_36_line_6: GL tag expansion.
  have row_6 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_7
  -- chapter_36_line_5: GL tag disintegration.
  have row_5 : (gl_fXYZ add N N N) := by
    exact row_6.1.1.1.1.1.1.1.1.2
  -- chapter_36_line_4: GL tag expansion.
  have row_4 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_5
  -- chapter_36_line_3: GL tag disintegration.
  have row_3 : (gl_implication8 add N) := by
    exact row_4.1.1.1.1
  -- chapter_36_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_3
  -- chapter_36_line_1: GL tag implication.
  have row_1 : (N v1) := by
    apply row_2
    exact row_10
  exact row_1

private theorem peano_source_014_check_zero
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (assumption_10 : (v1 = zero))
    (assumption_9 : (v1 = v3))
    (assumption_7 : (add v1 v2 v3))
    : (zero = v2) := by
  -- chapter_37_line_10: GL tag recursion.
  have row_10 : (v1 = zero) := by
    exact assumption_10
  -- chapter_37_line_11: GL tag symmetry of equality.
  have row_11 : (zero = v1) := by
    exact Eq.symm row_10
  -- chapter_37_line_9: GL tag task formulation.
  have row_9 : (v1 = v3) := by
    exact assumption_9
  -- chapter_37_line_8: GL tag symmetry of equality.
  have row_8 : (v3 = v1) := by
    exact Eq.symm row_9
  -- chapter_37_line_7: GL tag task formulation.
  have row_7 : (add v1 v2 v3) := by
    exact assumption_7
  -- chapter_37_line_6: GL tag equality1.
  have row_6 : (add v1 v2 v1) := by
    have equality_source := row_7
    have equality_step_1 := row_8
    cases equality_step_1
    exact equality_source
  -- chapter_37_line_5: GL tag equality1.
  have row_5 : (add zero v2 zero) := by
    have equality_source := row_6
    have equality_step_1 := row_10
    cases equality_step_1
    exact equality_source
  -- chapter_37_line_4: GL tag equality1.
  have row_4 : (add v1 v2 zero) := by
    have equality_source := row_5
    have equality_step_1 := row_11
    cases equality_step_1
    exact equality_source
  -- chapter_37_line_3: GL tag task formulation.
  have row_3 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_37_line_19: GL tag expansion.
  have row_19 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_3
  -- chapter_37_line_18: GL tag disintegration.
  have row_18 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_19.1
  -- chapter_37_line_17: GL tag expansion.
  have row_17 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_18
  -- chapter_37_line_16: GL tag disintegration.
  have row_16 : (gl_fXYZ add N N N) := by
    exact row_17.1.1.1.1.1.1.1.1.2
  -- chapter_37_line_15: GL tag expansion.
  have row_15 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_16
  -- chapter_37_line_14: GL tag disintegration.
  have row_14 : (gl_implication8 add N) := by
    exact row_15.1.1.1.1
  -- chapter_37_line_13: GL tag expansion.
  have row_13 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_14
  -- chapter_37_line_12: GL tag implication.
  have row_12 : (N v1) := by
    apply row_13
    exact row_7
  -- chapter_37_line_2: GL tag theorem.
  have row_2 := peano_source_045 N zero succ add mul one anchor relationalInduction
  -- chapter_37_line_1: GL tag implication.
  have row_1 : (zero = v2) := by
    apply row_2
    exact row_4
    exact row_12
  exact row_1

private theorem peano_source_014_check_induction_condition
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (assumption_13 : (add v1 v2 v3))
    (assumption_10 : (v1 = v3))
    (assumption_9 : (succ previous v1))
    (assumption_2 : (∀ (w1 : α), ((add previous v2 w1) → ((previous = w1) → (zero = v2)))))
    : (zero = v2) := by
  -- chapter_38_line_13: GL tag task formulation.
  have row_13 : (add v1 v2 v3) := by
    exact assumption_13
  -- chapter_38_line_10: GL tag task formulation.
  have row_10 : (v1 = v3) := by
    exact assumption_10
  -- chapter_38_line_14: GL tag symmetry of equality.
  have row_14 : (v3 = v1) := by
    exact Eq.symm row_10
  -- chapter_38_line_12: GL tag equality1.
  have row_12 : (add v1 v2 v1) := by
    have equality_source := row_13
    have equality_step_1 := row_14
    cases equality_step_1
    exact equality_source
  -- chapter_38_line_11: GL tag equality1.
  have row_11 : (add v3 v2 v1) := by
    have equality_source := row_12
    have equality_step_1 := row_10
    cases equality_step_1
    exact equality_source
  -- chapter_38_line_9: GL tag recursion.
  have row_9 : (succ previous v1) := by
    exact assumption_9
  -- chapter_38_line_8: GL tag equality1.
  have row_8 : (succ previous v3) := by
    have equality_source := row_9
    have equality_step_1 := row_10
    cases equality_step_1
    exact equality_source
  -- chapter_38_line_7: GL tag theorem.
  have row_7 := peano_source_005 N zero succ add mul one anchor relationalInduction
  -- chapter_38_line_5: GL tag task formulation.
  have row_5 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_38_line_24: GL tag expansion.
  have row_24 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_5
  -- chapter_38_line_23: GL tag disintegration.
  have row_23 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_24.1
  -- chapter_38_line_22: GL tag expansion.
  have row_22 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_23
  -- chapter_38_line_32: GL tag disintegration.
  have row_32 : (gl_fXY succ N N) := by
    exact row_22.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_38_line_31: GL tag expansion.
  have row_31 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_32
  -- chapter_38_line_30: GL tag disintegration.
  have row_30 : (gl_implication0 succ N) := by
    exact row_31.1.1.1
  -- chapter_38_line_29: GL tag expansion.
  have row_29 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_30
  -- chapter_38_line_28: GL tag implication.
  have row_28 : (N previous) := by
    apply row_29
    exact row_9
  -- chapter_38_line_21: GL tag disintegration.
  have row_21 : (gl_fXYZ add N N N) := by
    exact row_22.1.1.1.1.1.1.1.1.2
  -- chapter_38_line_20: GL tag expansion.
  have row_20 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_21
  -- chapter_38_line_27: GL tag disintegration.
  have row_27 : (gl_implication9 add N) := by
    exact row_20.1.1.1.2
  -- chapter_38_line_26: GL tag expansion.
  have row_26 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication9] using row_27
  -- chapter_38_line_25: GL tag implication.
  have row_25 : (N v2) := by
    apply row_26
    exact row_13
  -- chapter_38_line_19: GL tag disintegration.
  have row_19 : (gl_implication13 N N N add) := by
    exact row_20.1.2
  -- chapter_38_line_18: GL tag expansion.
  have row_18 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication13] using row_19
  -- chapter_38_line_17: GL tag implication.
  have row_17 : (gl_existence1 N previous v2 add) := by
    apply row_18
    exact row_28
    exact row_25
  -- chapter_38_line_16: GL tag expansion.
  have row_16 : (¬ (∀ (v5 : α), ((N v5) → (¬ (add previous v2 v5))))) := by
    simpa only [gl_existence1] using row_17
  have exists_row_16 : ∃ (v5 : α), ((N v5) ∧ (add previous v2 v5)) := existsAndOfNotForallImpNot row_16
  obtain ⟨v5, witness_row_16⟩ := exists_row_16
  -- chapter_38_line_15: GL tag disintegration.
  have row_15 : (add previous v2 v5) := by
    exact witness_row_16.2
  -- chapter_38_line_6: GL tag implication.
  have row_6 : (succ v5 v1) := by
    apply row_7
    exact row_11
    exact row_15
    exact row_8
  -- chapter_38_line_4: GL tag theorem.
  have row_4 := peano_source_030 N zero succ add mul one anchor relationalInduction
  -- chapter_38_line_3: GL tag implication.
  have row_3 : (previous = v5) := by
    apply row_4
    exact row_9
    exact row_6
  -- chapter_38_line_2: GL tag recursion.
  have row_2 : (∀ (w1 : α), ((add previous v2 w1) → ((previous = w1) → (zero = v2)))) := by
    exact assumption_2
  -- chapter_38_line_1: GL tag implication.
  have row_1 : (zero = v2) := by
    apply row_2
    exact row_15
    exact row_3
  exact row_1

theorem peano_source_014
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → ((v1 = v3) → (zero = v2)))) := by
  intro v1
  intro v2
  intro v3
  intro premise_1
  intro premise_2
  have inductionMember : N v1 := by
    have typingRule := peano_source_014_induction_typing N zero succ add mul one anchor relationalInduction
    exact typingRule v1 v2 v3 premise_1
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v2 : α) (v3 : α), ((add zero v2 v3) → ((zero = v3) → (zero = v2)))) := by
    intro v2
    intro v3
    intro base_premise_1
    intro base_premise_2
    have zeroRule := peano_source_014_check_zero N zero succ add mul one anchor relationalInduction
    exact zeroRule zero v2 v3 rfl base_premise_2 base_premise_1
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v2 : α) (v3 : α), ((add induction_n v2 v3) → ((induction_n = v3) → (zero = v2)))) → ∀ induction_m, succ induction_n induction_m → (∀ (v2 : α) (v3 : α), ((add induction_m v2 v3) → ((induction_m = v3) → (zero = v2)))) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro v2
    intro v3
    intro step_premise_1
    intro step_premise_2
    have step_induction_assumption_1 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have step_induction_assumption_2 :
        (∀ (w1 : α), ((add induction_n v2 w1) → ((induction_n = w1) → (zero = v2)))) := by
      intro w1
      intro step_induction_assumption_2_premise_1
      intro step_induction_assumption_2_premise_2
      apply induction_hypothesis
      all_goals assumption
    have stepRule := peano_source_014_check_induction_condition N zero succ add mul one anchor relationalInduction
    exact stepRule induction_n induction_m v2 v3 step_premise_1 step_premise_2 step_induction_assumption_1 step_induction_assumption_2
  have inductionProperty : (∀ (v2 : α) (v3 : α), ((add v1 v2 v3) → ((v1 = v3) → (zero = v2)))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v2 : α) (v3 : α), ((add induction_value v2 v3) → ((induction_value = v3) → (zero = v2)))))
      v1
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v2 v3 premise_1 premise_2

private theorem peano_source_015_induction_typing
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (assumption_10 : (add v1 v2 v3))
    : (N v2) := by
  -- chapter_39_line_10: GL tag task formulation.
  have row_10 : (add v1 v2 v3) := by
    exact assumption_10
  -- chapter_39_line_9: GL tag task formulation.
  have row_9 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_39_line_8: GL tag expansion.
  have row_8 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_9
  -- chapter_39_line_7: GL tag disintegration.
  have row_7 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_8.1
  -- chapter_39_line_6: GL tag expansion.
  have row_6 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_7
  -- chapter_39_line_5: GL tag disintegration.
  have row_5 : (gl_fXYZ add N N N) := by
    exact row_6.1.1.1.1.1.1.1.1.2
  -- chapter_39_line_4: GL tag expansion.
  have row_4 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_5
  -- chapter_39_line_3: GL tag disintegration.
  have row_3 : (gl_implication9 add N) := by
    exact row_4.1.1.1.2
  -- chapter_39_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication9] using row_3
  -- chapter_39_line_1: GL tag implication.
  have row_1 : (N v2) := by
    apply row_2
    exact row_10
  exact row_1

private theorem peano_source_015_check_zero
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (assumption_10 : (v2 = zero))
    (assumption_9 : (v2 = v3))
    (assumption_7 : (add v1 v2 v3))
    : (zero = v1) := by
  -- chapter_40_line_12: GL tag theorem.
  have row_12 := peano_source_031 N zero succ add mul one anchor relationalInduction
  -- chapter_40_line_10: GL tag recursion.
  have row_10 : (v2 = zero) := by
    exact assumption_10
  -- chapter_40_line_9: GL tag task formulation.
  have row_9 : (v2 = v3) := by
    exact assumption_9
  -- chapter_40_line_8: GL tag symmetry of equality.
  have row_8 : (v3 = v2) := by
    exact Eq.symm row_9
  -- chapter_40_line_7: GL tag task formulation.
  have row_7 : (add v1 v2 v3) := by
    exact assumption_7
  -- chapter_40_line_6: GL tag equality1.
  have row_6 : (add v1 v2 v2) := by
    have equality_source := row_7
    have equality_step_1 := row_8
    cases equality_step_1
    exact equality_source
  -- chapter_40_line_31: GL tag equality1.
  have row_31 : (add v1 v3 v2) := by
    have equality_source := row_6
    have equality_step_1 := row_9
    cases equality_step_1
    exact equality_source
  -- chapter_40_line_5: GL tag equality1.
  have row_5 : (add v1 zero zero) := by
    have equality_source := row_6
    have equality_step_1 := row_10
    cases equality_step_1
    exact equality_source
  -- chapter_40_line_3: GL tag task formulation.
  have row_3 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_40_line_23: GL tag expansion.
  have row_23 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_3
  -- chapter_40_line_22: GL tag disintegration.
  have row_22 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_23.1
  -- chapter_40_line_21: GL tag expansion.
  have row_21 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_22
  -- chapter_40_line_28: GL tag disintegration.
  have row_28 : (gl_fXYZ add N N N) := by
    exact row_21.1.1.1.1.1.1.1.1.2
  -- chapter_40_line_27: GL tag expansion.
  have row_27 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_28
  -- chapter_40_line_26: GL tag disintegration.
  have row_26 : (gl_implication9 add N) := by
    exact row_27.1.1.1.2
  -- chapter_40_line_25: GL tag expansion.
  have row_25 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication9] using row_26
  -- chapter_40_line_30: GL tag implication.
  have row_30 : (N v3) := by
    apply row_25
    exact row_31
  -- chapter_40_line_24: GL tag implication.
  have row_24 : (N v2) := by
    apply row_25
    exact row_7
  -- chapter_40_line_20: GL tag disintegration.
  have row_20 : (gl_fXY succ N N) := by
    exact row_21.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_40_line_19: GL tag expansion.
  have row_19 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_20
  -- chapter_40_line_18: GL tag disintegration.
  have row_18 : (gl_implication4 N N succ) := by
    exact row_19.1.2
  -- chapter_40_line_17: GL tag expansion.
  have row_17 : (∀ (w1 : α), ((N w1) → (gl_existence0 N w1 succ))) := by
    simpa only [gl_implication4] using row_18
  -- chapter_40_line_16: GL tag implication.
  have row_16 : (gl_existence0 N v2 succ) := by
    apply row_17
    exact row_24
  -- chapter_40_line_15: GL tag expansion.
  have row_15 : (¬ (∀ (v4 : α), ((N v4) → (¬ (succ v2 v4))))) := by
    simpa only [gl_existence0] using row_16
  have exists_row_15 : ∃ (v4 : α), ((N v4) ∧ (succ v2 v4)) := existsAndOfNotForallImpNot row_15
  obtain ⟨v4, witness_row_15⟩ := exists_row_15
  -- chapter_40_line_14: GL tag disintegration.
  have row_14 : (succ v2 v4) := by
    exact witness_row_15.2
  -- chapter_40_line_29: GL tag equality1.
  have row_29 : (succ v3 v4) := by
    have equality_source := row_14
    have equality_step_1 := row_9
    cases equality_step_1
    exact equality_source
  -- chapter_40_line_13: GL tag equality1.
  have row_13 : (succ zero v4) := by
    have equality_source := row_14
    have equality_step_1 := row_10
    cases equality_step_1
    exact equality_source
  -- chapter_40_line_11: GL tag implication.
  have row_11 : (zero = v3) := by
    apply row_12
    exact row_29
    exact row_13
  -- chapter_40_line_4: GL tag equality1.
  have row_4 : (add v1 v3 zero) := by
    have equality_source := row_5
    have equality_step_1 := row_11
    cases equality_step_1
    exact equality_source
  -- chapter_40_line_2: GL tag theorem.
  have row_2 := peano_source_047 N zero succ add mul one anchor relationalInduction
  -- chapter_40_line_1: GL tag implication.
  have row_1 : (zero = v1) := by
    apply row_2
    exact row_4
    exact row_30
  exact row_1

private theorem peano_source_015_check_induction_condition
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (assumption_17 : (add v1 v2 v3))
    (assumption_14 : (v2 = v3))
    (assumption_13 : (succ previous v2))
    (assumption_2 : (∀ (w1 : α), ((add v1 previous w1) → ((previous = w1) → (zero = v1)))))
    : (zero = v1) := by
  -- chapter_41_line_17: GL tag task formulation.
  have row_17 : (add v1 v2 v3) := by
    exact assumption_17
  -- chapter_41_line_14: GL tag task formulation.
  have row_14 : (v2 = v3) := by
    exact assumption_14
  -- chapter_41_line_18: GL tag symmetry of equality.
  have row_18 : (v3 = v2) := by
    exact Eq.symm row_14
  -- chapter_41_line_16: GL tag equality1.
  have row_16 : (add v1 v2 v2) := by
    have equality_source := row_17
    have equality_step_1 := row_18
    cases equality_step_1
    exact equality_source
  -- chapter_41_line_15: GL tag equality1.
  have row_15 : (add v1 v3 v2) := by
    have equality_source := row_16
    have equality_step_1 := row_14
    cases equality_step_1
    exact equality_source
  -- chapter_41_line_13: GL tag recursion.
  have row_13 : (succ previous v2) := by
    exact assumption_13
  -- chapter_41_line_12: GL tag equality1.
  have row_12 : (succ previous v3) := by
    have equality_source := row_13
    have equality_step_1 := row_14
    cases equality_step_1
    exact equality_source
  -- chapter_41_line_5: GL tag task formulation.
  have row_5 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_41_line_11: GL tag expansion.
  have row_11 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_5
  -- chapter_41_line_10: GL tag disintegration.
  have row_10 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_11.1
  -- chapter_41_line_9: GL tag expansion.
  have row_9 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_10
  -- chapter_41_line_33: GL tag disintegration.
  have row_33 : (gl_fXY succ N N) := by
    exact row_9.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_41_line_32: GL tag expansion.
  have row_32 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_33
  -- chapter_41_line_31: GL tag disintegration.
  have row_31 : (gl_implication0 succ N) := by
    exact row_32.1.1.1
  -- chapter_41_line_30: GL tag expansion.
  have row_30 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_31
  -- chapter_41_line_29: GL tag implication.
  have row_29 : (N previous) := by
    apply row_30
    exact row_13
  -- chapter_41_line_25: GL tag disintegration.
  have row_25 : (gl_fXYZ add N N N) := by
    exact row_9.1.1.1.1.1.1.1.1.2
  -- chapter_41_line_24: GL tag expansion.
  have row_24 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_25
  -- chapter_41_line_28: GL tag disintegration.
  have row_28 : (gl_implication8 add N) := by
    exact row_24.1.1.1.1
  -- chapter_41_line_27: GL tag expansion.
  have row_27 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_28
  -- chapter_41_line_26: GL tag implication.
  have row_26 : (N v1) := by
    apply row_27
    exact row_17
  -- chapter_41_line_23: GL tag disintegration.
  have row_23 : (gl_implication13 N N N add) := by
    exact row_24.1.2
  -- chapter_41_line_22: GL tag expansion.
  have row_22 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication13] using row_23
  -- chapter_41_line_21: GL tag implication.
  have row_21 : (gl_existence1 N v1 previous add) := by
    apply row_22
    exact row_26
    exact row_29
  -- chapter_41_line_20: GL tag expansion.
  have row_20 : (¬ (∀ (v5 : α), ((N v5) → (¬ (add v1 previous v5))))) := by
    simpa only [gl_existence1] using row_21
  have exists_row_20 : ∃ (v5 : α), ((N v5) ∧ (add v1 previous v5)) := existsAndOfNotForallImpNot row_20
  obtain ⟨v5, witness_row_20⟩ := exists_row_20
  -- chapter_41_line_19: GL tag disintegration.
  have row_19 : (add v1 previous v5) := by
    exact witness_row_20.2
  -- chapter_41_line_8: GL tag disintegration.
  have row_8 : (gl_implication17 N succ add) := by
    exact row_9.1.1.1.1.1.2
  -- chapter_41_line_7: GL tag expansion.
  have row_7 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication17] using row_8
  -- chapter_41_line_6: GL tag implication.
  have row_6 : (succ v5 v2) := by
    apply row_7
    exact row_29
    exact row_12
    exact row_19
    exact row_15
  -- chapter_41_line_4: GL tag theorem.
  have row_4 := peano_source_030 N zero succ add mul one anchor relationalInduction
  -- chapter_41_line_3: GL tag implication.
  have row_3 : (previous = v5) := by
    apply row_4
    exact row_13
    exact row_6
  -- chapter_41_line_2: GL tag recursion.
  have row_2 : (∀ (w1 : α), ((add v1 previous w1) → ((previous = w1) → (zero = v1)))) := by
    exact assumption_2
  -- chapter_41_line_1: GL tag implication.
  have row_1 : (zero = v1) := by
    apply row_2
    exact row_19
    exact row_3
  exact row_1

theorem peano_source_015
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → ((v2 = v3) → (zero = v1)))) := by
  intro v1
  intro v2
  intro v3
  intro premise_1
  intro premise_2
  have inductionMember : N v2 := by
    have typingRule := peano_source_015_induction_typing N zero succ add mul one anchor relationalInduction
    exact typingRule v1 v2 v3 premise_1
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v1 : α) (v3 : α), ((add v1 zero v3) → ((zero = v3) → (zero = v1)))) := by
    intro v1
    intro v3
    intro base_premise_1
    intro base_premise_2
    have zeroRule := peano_source_015_check_zero N zero succ add mul one anchor relationalInduction
    exact zeroRule v1 zero v3 rfl base_premise_2 base_premise_1
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v1 : α) (v3 : α), ((add v1 induction_n v3) → ((induction_n = v3) → (zero = v1)))) → ∀ induction_m, succ induction_n induction_m → (∀ (v1 : α) (v3 : α), ((add v1 induction_m v3) → ((induction_m = v3) → (zero = v1)))) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro v1
    intro v3
    intro step_premise_1
    intro step_premise_2
    have step_induction_assumption_1 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have step_induction_assumption_2 :
        (∀ (w1 : α), ((add v1 induction_n w1) → ((induction_n = w1) → (zero = v1)))) := by
      intro w1
      intro step_induction_assumption_2_premise_1
      intro step_induction_assumption_2_premise_2
      apply induction_hypothesis
      all_goals assumption
    have stepRule := peano_source_015_check_induction_condition N zero succ add mul one anchor relationalInduction
    exact stepRule induction_n v1 induction_m v3 step_premise_1 step_premise_2 step_induction_assumption_1 step_induction_assumption_2
  have inductionProperty : (∀ (v1 : α) (v3 : α), ((add v1 v2 v3) → ((v2 = v3) → (zero = v1)))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v1 : α) (v3 : α), ((add v1 induction_value v3) → ((induction_value = v3) → (zero = v1)))))
      v2
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v1 v3 premise_1 premise_2

private theorem peano_source_016_induction_typing
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (assumption_10 : (add v1 v2 v3))
    : (N v1) := by
  -- chapter_42_line_10: GL tag task formulation.
  have row_10 : (add v1 v2 v3) := by
    exact assumption_10
  -- chapter_42_line_9: GL tag task formulation.
  have row_9 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_42_line_8: GL tag expansion.
  have row_8 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_9
  -- chapter_42_line_7: GL tag disintegration.
  have row_7 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_8.1
  -- chapter_42_line_6: GL tag expansion.
  have row_6 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_7
  -- chapter_42_line_5: GL tag disintegration.
  have row_5 : (gl_fXYZ add N N N) := by
    exact row_6.1.1.1.1.1.1.1.1.2
  -- chapter_42_line_4: GL tag expansion.
  have row_4 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_5
  -- chapter_42_line_3: GL tag disintegration.
  have row_3 : (gl_implication8 add N) := by
    exact row_4.1.1.1.1
  -- chapter_42_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_3
  -- chapter_42_line_1: GL tag implication.
  have row_1 : (N v1) := by
    apply row_2
    exact row_10
  exact row_1

private theorem peano_source_016_check_zero
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (assumption_7 : (v1 = zero))
    (assumption_6 : (add v1 v2 v3))
    : (add v2 v1 v3) := by
  -- chapter_43_line_7: GL tag recursion.
  have row_7 : (v1 = zero) := by
    exact assumption_7
  -- chapter_43_line_8: GL tag symmetry of equality.
  have row_8 : (zero = v1) := by
    exact Eq.symm row_7
  -- chapter_43_line_6: GL tag task formulation.
  have row_6 : (add v1 v2 v3) := by
    exact assumption_6
  -- chapter_43_line_5: GL tag equality1.
  have row_5 : (add zero v2 v3) := by
    have equality_source := row_6
    have equality_step_1 := row_7
    cases equality_step_1
    exact equality_source
  -- chapter_43_line_4: GL tag task formulation.
  have row_4 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_43_line_3: GL tag theorem.
  have row_3 := peano_source_035 N zero succ add mul one anchor relationalInduction
  -- chapter_43_line_2: GL tag implication.
  have row_2 : (add v2 zero v3) := by
    apply row_3
    exact row_5
  -- chapter_43_line_1: GL tag equality1.
  have row_1 : (add v2 v1 v3) := by
    have equality_source := row_2
    have equality_step_1 := row_8
    cases equality_step_1
    exact equality_source
  exact row_1

private theorem peano_source_016_check_induction_condition
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (assumption_35 : (∀ (w1 : α), ((add previous v2 w1) → (add v2 previous w1))))
    (assumption_11 : (add v1 v2 v3))
    (assumption_10 : (succ previous v1))
    : (add v2 v1 v3) := by
  -- chapter_44_line_35: GL tag recursion.
  have row_35 : (∀ (w1 : α), ((add previous v2 w1) → (add v2 previous w1))) := by
    exact assumption_35
  -- chapter_44_line_11: GL tag task formulation.
  have row_11 : (add v1 v2 v3) := by
    exact assumption_11
  -- chapter_44_line_10: GL tag recursion.
  have row_10 : (succ previous v1) := by
    exact assumption_10
  -- chapter_44_line_9: GL tag theorem.
  have row_9 := peano_source_005 N zero succ add mul one anchor relationalInduction
  -- chapter_44_line_7: GL tag task formulation.
  have row_7 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_44_line_6: GL tag expansion.
  have row_6 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_7
  -- chapter_44_line_5: GL tag disintegration.
  have row_5 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_6.1
  -- chapter_44_line_4: GL tag expansion.
  have row_4 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_5
  -- chapter_44_line_27: GL tag disintegration.
  have row_27 : (gl_fXY succ N N) := by
    exact row_4.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_44_line_26: GL tag expansion.
  have row_26 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_27
  -- chapter_44_line_25: GL tag disintegration.
  have row_25 : (gl_implication0 succ N) := by
    exact row_26.1.1.1
  -- chapter_44_line_24: GL tag expansion.
  have row_24 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_25
  -- chapter_44_line_23: GL tag implication.
  have row_23 : (N previous) := by
    apply row_24
    exact row_10
  -- chapter_44_line_19: GL tag disintegration.
  have row_19 : (gl_fXYZ add N N N) := by
    exact row_4.1.1.1.1.1.1.1.1.2
  -- chapter_44_line_18: GL tag expansion.
  have row_18 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_19
  -- chapter_44_line_30: GL tag disintegration.
  have row_30 : (gl_implication14 N N add) := by
    exact row_18.2
  -- chapter_44_line_29: GL tag expansion.
  have row_29 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (∀ (w3 : α), ((add w1 w2 w3) → (∀ (w4 : α), ((add w1 w2 w4) → (w3 = w4))))))))) := by
    simpa only [gl_implication14] using row_30
  -- chapter_44_line_22: GL tag disintegration.
  have row_22 : (gl_implication9 add N) := by
    exact row_18.1.1.1.2
  -- chapter_44_line_21: GL tag expansion.
  have row_21 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication9] using row_22
  -- chapter_44_line_20: GL tag implication.
  have row_20 : (N v2) := by
    apply row_21
    exact row_11
  -- chapter_44_line_17: GL tag disintegration.
  have row_17 : (gl_implication13 N N N add) := by
    exact row_18.1.2
  -- chapter_44_line_16: GL tag expansion.
  have row_16 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication13] using row_17
  -- chapter_44_line_33: GL tag implication.
  have row_33 : (gl_existence1 N v2 previous add) := by
    apply row_16
    exact row_20
    exact row_23
  -- chapter_44_line_32: GL tag expansion.
  have row_32 : (¬ (∀ (v9 : α), ((N v9) → (¬ (add v2 previous v9))))) := by
    simpa only [gl_existence1] using row_33
  have exists_row_32 : ∃ (v9 : α), ((N v9) ∧ (add v2 previous v9)) := existsAndOfNotForallImpNot row_32
  obtain ⟨v9, witness_row_32⟩ := exists_row_32
  -- chapter_44_line_31: GL tag disintegration.
  have row_31 : (add v2 previous v9) := by
    exact witness_row_32.2
  -- chapter_44_line_15: GL tag implication.
  have row_15 : (gl_existence1 N previous v2 add) := by
    apply row_16
    exact row_23
    exact row_20
  -- chapter_44_line_14: GL tag expansion.
  have row_14 : (¬ (∀ (v10 : α), ((N v10) → (¬ (add previous v2 v10))))) := by
    simpa only [gl_existence1] using row_15
  have exists_row_14 : ∃ (v10 : α), ((N v10) ∧ (add previous v2 v10)) := existsAndOfNotForallImpNot row_14
  obtain ⟨v10, witness_row_14⟩ := exists_row_14
  -- chapter_44_line_13: GL tag disintegration.
  have row_13 : (add previous v2 v10) := by
    exact witness_row_14.2
  -- chapter_44_line_34: GL tag implication.
  have row_34 : (add v2 previous v10) := by
    apply row_35
    exact row_13
  -- chapter_44_line_28: GL tag implication.
  have row_28 : (v10 = v9) := by
    apply row_29
    exact row_20
    exact row_23
    exact row_34
    exact row_31
  -- chapter_44_line_12: GL tag equality1.
  have row_12 : (add previous v2 v9) := by
    have equality_source := row_13
    have equality_step_1 := row_28
    cases equality_step_1
    exact equality_source
  -- chapter_44_line_8: GL tag implication.
  have row_8 : (succ v9 v3) := by
    apply row_9
    exact row_11
    exact row_12
    exact row_10
  -- chapter_44_line_3: GL tag disintegration.
  have row_3 : (gl_implication18 N succ add) := by
    exact row_4.1.1.1.1.2
  -- chapter_44_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((succ w4 w5) → (add w3 w2 w5))))))))) := by
    simpa only [gl_implication18] using row_3
  -- chapter_44_line_1: GL tag implication.
  have row_1 : (add v2 v1 v3) := by
    apply row_2
    exact row_23
    exact row_10
    exact row_31
    exact row_8
  exact row_1

theorem peano_source_016
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (add v2 v1 v3))) := by
  intro v1
  intro v2
  intro v3
  intro premise_1
  have inductionMember : N v1 := by
    have typingRule := peano_source_016_induction_typing N zero succ add mul one anchor relationalInduction
    exact typingRule v1 v2 v3 premise_1
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v2 : α) (v3 : α), ((add zero v2 v3) → (add v2 zero v3))) := by
    intro v2
    intro v3
    intro base_premise_1
    have zeroRule := peano_source_016_check_zero N zero succ add mul one anchor relationalInduction
    exact zeroRule zero v2 v3 rfl base_premise_1
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v2 : α) (v3 : α), ((add induction_n v2 v3) → (add v2 induction_n v3))) → ∀ induction_m, succ induction_n induction_m → (∀ (v2 : α) (v3 : α), ((add induction_m v2 v3) → (add v2 induction_m v3))) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro v2
    intro v3
    intro step_premise_1
    have step_induction_assumption_1 :
        (∀ (w1 : α), ((add induction_n v2 w1) → (add v2 induction_n w1))) := by
      intro w1
      intro step_induction_assumption_1_premise_1
      apply induction_hypothesis
      all_goals assumption
    have step_induction_assumption_2 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have stepRule := peano_source_016_check_induction_condition N zero succ add mul one anchor relationalInduction
    exact stepRule induction_n induction_m v2 v3 step_induction_assumption_1 step_premise_1 step_induction_assumption_2
  have inductionProperty : (∀ (v2 : α) (v3 : α), ((add v1 v2 v3) → (add v2 v1 v3))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v2 : α) (v3 : α), ((add induction_value v2 v3) → (add v2 induction_value v3))))
      v1
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v2 v3 premise_1

private theorem peano_source_018_induction_typing
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v2 : α)
    (v4 : α)
    (v6 : α)
    (assumption_10 : (add v2 v4 v6))
    : (N v4) := by
  -- chapter_46_line_10: GL tag task formulation.
  have row_10 : (add v2 v4 v6) := by
    exact assumption_10
  -- chapter_46_line_9: GL tag task formulation.
  have row_9 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_46_line_8: GL tag expansion.
  have row_8 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_9
  -- chapter_46_line_7: GL tag disintegration.
  have row_7 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_8.1
  -- chapter_46_line_6: GL tag expansion.
  have row_6 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_7
  -- chapter_46_line_5: GL tag disintegration.
  have row_5 : (gl_fXYZ add N N N) := by
    exact row_6.1.1.1.1.1.1.1.1.2
  -- chapter_46_line_4: GL tag expansion.
  have row_4 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_5
  -- chapter_46_line_3: GL tag disintegration.
  have row_3 : (gl_implication9 add N) := by
    exact row_4.1.1.1.2
  -- chapter_46_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication9] using row_3
  -- chapter_46_line_1: GL tag implication.
  have row_1 : (N v4) := by
    apply row_2
    exact row_10
  exact row_1

private theorem peano_source_018_check_zero
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (v4 : α)
    (v5 : α)
    (v6 : α)
    (v7 : α)
    (assumption_41 : (add v2 v4 v6))
    (assumption_37 : (mul v1 v6 v7))
    (assumption_30 : (v4 = zero))
    (assumption_20 : (mul v1 v2 v3))
    (assumption_18 : (mul v1 v4 v5))
    : (add v3 v5 v7) := by
  -- chapter_47_line_41: GL tag task formulation.
  have row_41 : (add v2 v4 v6) := by
    exact assumption_41
  -- chapter_47_line_37: GL tag task formulation.
  have row_37 : (mul v1 v6 v7) := by
    exact assumption_37
  -- chapter_47_line_30: GL tag recursion.
  have row_30 : (v4 = zero) := by
    exact assumption_30
  -- chapter_47_line_40: GL tag equality1.
  have row_40 : (add v2 zero v6) := by
    have equality_source := row_41
    have equality_step_1 := row_30
    cases equality_step_1
    exact equality_source
  -- chapter_47_line_28: GL tag theorem.
  have row_28 := peano_source_041 N zero succ add mul one anchor relationalInduction
  -- chapter_47_line_24: GL tag theorem.
  have row_24 := peano_source_039 N zero succ add mul one anchor relationalInduction
  -- chapter_47_line_20: GL tag task formulation.
  have row_20 : (mul v1 v2 v3) := by
    exact assumption_20
  -- chapter_47_line_18: GL tag task formulation.
  have row_18 : (mul v1 v4 v5) := by
    exact assumption_18
  -- chapter_47_line_29: GL tag equality1.
  have row_29 : (mul v1 zero v5) := by
    have equality_source := row_18
    have equality_step_1 := row_30
    cases equality_step_1
    exact equality_source
  -- chapter_47_line_12: GL tag task formulation.
  have row_12 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_47_line_11: GL tag expansion.
  have row_11 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_12
  -- chapter_47_line_10: GL tag disintegration.
  have row_10 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_11.1
  -- chapter_47_line_9: GL tag expansion.
  have row_9 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_10
  -- chapter_47_line_17: GL tag disintegration.
  have row_17 : (gl_fXYZ mul N N N) := by
    exact row_9.1.1.1.2
  -- chapter_47_line_16: GL tag expansion.
  have row_16 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_17
  -- chapter_47_line_36: GL tag disintegration.
  have row_36 : (gl_implication14 N N mul) := by
    exact row_16.2
  -- chapter_47_line_35: GL tag expansion.
  have row_35 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (∀ (w3 : α), ((mul w1 w2 w3) → (∀ (w4 : α), ((mul w1 w2 w4) → (w3 = w4))))))))) := by
    simpa only [gl_implication14] using row_36
  -- chapter_47_line_33: GL tag disintegration.
  have row_33 : (gl_implication8 mul N) := by
    exact row_16.1.1.1.1
  -- chapter_47_line_32: GL tag expansion.
  have row_32 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_33
  -- chapter_47_line_31: GL tag implication.
  have row_31 : (N v1) := by
    apply row_32
    exact row_18
  -- chapter_47_line_27: GL tag implication.
  have row_27 : (zero = v5) := by
    apply row_28
    exact row_29
    exact row_31
  -- chapter_47_line_26: GL tag symmetry of equality.
  have row_26 : (v5 = zero) := by
    exact Eq.symm row_27
  -- chapter_47_line_15: GL tag disintegration.
  have row_15 : (gl_implication10 mul N) := by
    exact row_16.1.1.2
  -- chapter_47_line_14: GL tag expansion.
  have row_14 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication10] using row_15
  -- chapter_47_line_19: GL tag implication.
  have row_19 : (N v3) := by
    apply row_14
    exact row_20
  -- chapter_47_line_13: GL tag implication.
  have row_13 : (N v5) := by
    apply row_14
    exact row_18
  -- chapter_47_line_8: GL tag disintegration.
  have row_8 : (gl_fXYZ add N N N) := by
    exact row_9.1.1.1.1.1.1.1.1.2
  -- chapter_47_line_7: GL tag expansion.
  have row_7 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_8
  -- chapter_47_line_47: GL tag disintegration.
  have row_47 : (gl_implication10 add N) := by
    exact row_7.1.1.2
  -- chapter_47_line_46: GL tag expansion.
  have row_46 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication10] using row_47
  -- chapter_47_line_45: GL tag implication.
  have row_45 : (N v6) := by
    apply row_46
    exact row_41
  -- chapter_47_line_44: GL tag disintegration.
  have row_44 : (gl_implication8 add N) := by
    exact row_7.1.1.1.1
  -- chapter_47_line_43: GL tag expansion.
  have row_43 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_44
  -- chapter_47_line_42: GL tag implication.
  have row_42 : (N v2) := by
    apply row_43
    exact row_41
  -- chapter_47_line_39: GL tag implication.
  have row_39 : (v2 = v6) := by
    apply row_24
    exact row_40
    exact row_42
  -- chapter_47_line_38: GL tag equality1.
  have row_38 : (mul v1 v6 v3) := by
    have equality_source := row_20
    have equality_step_1 := row_39
    cases equality_step_1
    exact equality_source
  -- chapter_47_line_34: GL tag implication.
  have row_34 : (v3 = v7) := by
    apply row_35
    exact row_31
    exact row_45
    exact row_38
    exact row_37
  -- chapter_47_line_6: GL tag disintegration.
  have row_6 : (gl_implication13 N N N add) := by
    exact row_7.1.2
  -- chapter_47_line_5: GL tag expansion.
  have row_5 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication13] using row_6
  -- chapter_47_line_4: GL tag implication.
  have row_4 : (gl_existence1 N v3 v5 add) := by
    apply row_5
    exact row_19
    exact row_13
  -- chapter_47_line_3: GL tag expansion.
  have row_3 : (¬ (∀ (v8 : α), ((N v8) → (¬ (add v3 v5 v8))))) := by
    simpa only [gl_existence1] using row_4
  have exists_row_3 : ∃ (v8 : α), ((N v8) ∧ (add v3 v5 v8)) := existsAndOfNotForallImpNot row_3
  obtain ⟨v8, witness_row_3⟩ := exists_row_3
  -- chapter_47_line_2: GL tag disintegration.
  have row_2 : (add v3 v5 v8) := by
    exact witness_row_3.2
  -- chapter_47_line_25: GL tag equality1.
  have row_25 : (add v3 zero v8) := by
    have equality_source := row_2
    have equality_step_1 := row_26
    cases equality_step_1
    exact equality_source
  -- chapter_47_line_23: GL tag implication.
  have row_23 : (v3 = v8) := by
    apply row_24
    exact row_25
    exact row_19
  -- chapter_47_line_22: GL tag symmetry of equality.
  have row_22 : (v8 = v3) := by
    exact Eq.symm row_23
  -- chapter_47_line_21: GL tag equality2.
  have row_21 : (v8 = v7) := by
    exact Eq.trans row_22 row_34
  -- chapter_47_line_1: GL tag equality1.
  have row_1 : (add v3 v5 v7) := by
    have equality_source := row_2
    have equality_step_1 := row_21
    cases equality_step_1
    exact equality_source
  exact row_1

private theorem peano_source_018_check_induction_condition
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (v4 : α)
    (v5 : α)
    (v6 : α)
    (v7 : α)
    (assumption_50 : (mul v1 v6 v7))
    (assumption_37 : (succ previous v4))
    (assumption_31 : (add v2 v4 v6))
    (assumption_20 : (mul v1 v4 v5))
    (assumption_6 : (mul v1 v2 v3))
    (assumption_5 : ((mul v1 v2 v3) → (∀ (w1 : α), ((mul v1 previous w1) → (∀ (w2 : α) (w3 : α), ((mul v1 w2 w3) → ((add v2 previous w2) → (add v3 w1 w3))))))))
    : (add v3 v5 v7) := by
  -- chapter_48_line_50: GL tag task formulation.
  have row_50 : (mul v1 v6 v7) := by
    exact assumption_50
  -- chapter_48_line_46: GL tag theorem.
  have row_46 := peano_source_020 N zero succ add mul one anchor relationalInduction
  -- chapter_48_line_37: GL tag recursion.
  have row_37 : (succ previous v4) := by
    exact assumption_37
  -- chapter_48_line_31: GL tag task formulation.
  have row_31 : (add v2 v4 v6) := by
    exact assumption_31
  -- chapter_48_line_20: GL tag task formulation.
  have row_20 : (mul v1 v4 v5) := by
    exact assumption_20
  -- chapter_48_line_6: GL tag task formulation.
  have row_6 : (mul v1 v2 v3) := by
    exact assumption_6
  -- chapter_48_line_5: GL tag recursion.
  have row_5 : ((mul v1 v2 v3) → (∀ (w1 : α), ((mul v1 previous w1) → (∀ (w2 : α) (w3 : α), ((mul v1 w2 w3) → ((add v2 previous w2) → (add v3 w1 w3))))))) := by
    exact assumption_5
  -- chapter_48_line_3: GL tag task formulation.
  have row_3 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_48_line_16: GL tag expansion.
  have row_16 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_3
  -- chapter_48_line_15: GL tag disintegration.
  have row_15 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_16.1
  -- chapter_48_line_14: GL tag expansion.
  have row_14 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_15
  -- chapter_48_line_49: GL tag disintegration.
  have row_49 : (gl_implication17 N succ add) := by
    exact row_14.1.1.1.1.1.2
  -- chapter_48_line_48: GL tag expansion.
  have row_48 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication17] using row_49
  -- chapter_48_line_44: GL tag disintegration.
  have row_44 : (gl_implication21 N succ mul add) := by
    exact row_14.2
  -- chapter_48_line_43: GL tag expansion.
  have row_43 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((mul w3 w1 w4) → (∀ (w5 : α), ((mul w3 w2 w5) → (add w4 w3 w5))))))))) := by
    simpa only [gl_implication21] using row_44
  -- chapter_48_line_36: GL tag disintegration.
  have row_36 : (gl_fXY succ N N) := by
    exact row_14.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_48_line_35: GL tag expansion.
  have row_35 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_36
  -- chapter_48_line_34: GL tag disintegration.
  have row_34 : (gl_implication0 succ N) := by
    exact row_35.1.1.1
  -- chapter_48_line_33: GL tag expansion.
  have row_33 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_34
  -- chapter_48_line_32: GL tag implication.
  have row_32 : (N previous) := by
    apply row_33
    exact row_37
  -- chapter_48_line_27: GL tag disintegration.
  have row_27 : (gl_fXYZ add N N N) := by
    exact row_14.1.1.1.1.1.1.1.1.2
  -- chapter_48_line_26: GL tag expansion.
  have row_26 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_27
  -- chapter_48_line_30: GL tag disintegration.
  have row_30 : (gl_implication8 add N) := by
    exact row_26.1.1.1.1
  -- chapter_48_line_29: GL tag expansion.
  have row_29 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_30
  -- chapter_48_line_28: GL tag implication.
  have row_28 : (N v2) := by
    apply row_29
    exact row_31
  -- chapter_48_line_25: GL tag disintegration.
  have row_25 : (gl_implication13 N N N add) := by
    exact row_26.1.2
  -- chapter_48_line_24: GL tag expansion.
  have row_24 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication13] using row_25
  -- chapter_48_line_23: GL tag implication.
  have row_23 : (gl_existence1 N v2 previous add) := by
    apply row_24
    exact row_28
    exact row_32
  -- chapter_48_line_22: GL tag expansion.
  have row_22 : (¬ (∀ (v13 : α), ((N v13) → (¬ (add v2 previous v13))))) := by
    simpa only [gl_existence1] using row_23
  have exists_row_22 : ∃ (v13 : α), ((N v13) ∧ (add v2 previous v13)) := existsAndOfNotForallImpNot row_22
  obtain ⟨v13, witness_row_22⟩ := exists_row_22
  -- chapter_48_line_41: GL tag disintegration.
  have row_41 : (add v2 previous v13) := by
    exact witness_row_22.2
  -- chapter_48_line_47: GL tag implication.
  have row_47 : (succ v13 v6) := by
    apply row_48
    exact row_32
    exact row_37
    exact row_41
    exact row_31
  -- chapter_48_line_21: GL tag disintegration.
  have row_21 : (N v13) := by
    exact witness_row_22.1
  -- chapter_48_line_13: GL tag disintegration.
  have row_13 : (gl_fXYZ mul N N N) := by
    exact row_14.1.1.1.2
  -- chapter_48_line_12: GL tag expansion.
  have row_12 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_13
  -- chapter_48_line_19: GL tag disintegration.
  have row_19 : (gl_implication8 mul N) := by
    exact row_12.1.1.1.1
  -- chapter_48_line_18: GL tag expansion.
  have row_18 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_19
  -- chapter_48_line_17: GL tag implication.
  have row_17 : (N v1) := by
    apply row_18
    exact row_20
  -- chapter_48_line_11: GL tag disintegration.
  have row_11 : (gl_implication13 N N N mul) := by
    exact row_12.1.2
  -- chapter_48_line_10: GL tag expansion.
  have row_10 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication13] using row_11
  -- chapter_48_line_40: GL tag implication.
  have row_40 : (gl_existence1 N v1 previous mul) := by
    apply row_10
    exact row_17
    exact row_32
  -- chapter_48_line_39: GL tag expansion.
  have row_39 : (¬ (∀ (v8 : α), ((N v8) → (¬ (mul v1 previous v8))))) := by
    simpa only [gl_existence1] using row_40
  have exists_row_39 : ∃ (v8 : α), ((N v8) ∧ (mul v1 previous v8)) := existsAndOfNotForallImpNot row_39
  obtain ⟨v8, witness_row_39⟩ := exists_row_39
  -- chapter_48_line_38: GL tag disintegration.
  have row_38 : (mul v1 previous v8) := by
    exact witness_row_39.2
  -- chapter_48_line_42: GL tag implication.
  have row_42 : (add v8 v1 v5) := by
    apply row_43
    exact row_32
    exact row_37
    exact row_38
    exact row_20
  -- chapter_48_line_9: GL tag implication.
  have row_9 : (gl_existence1 N v1 v13 mul) := by
    apply row_10
    exact row_17
    exact row_21
  -- chapter_48_line_8: GL tag expansion.
  have row_8 : (¬ (∀ (v9 : α), ((N v9) → (¬ (mul v1 v13 v9))))) := by
    simpa only [gl_existence1] using row_9
  have exists_row_8 : ∃ (v9 : α), ((N v9) ∧ (mul v1 v13 v9)) := existsAndOfNotForallImpNot row_8
  obtain ⟨v9, witness_row_8⟩ := exists_row_8
  -- chapter_48_line_7: GL tag disintegration.
  have row_7 : (mul v1 v13 v9) := by
    exact witness_row_8.2
  -- chapter_48_line_45: GL tag implication.
  have row_45 : (add v9 v1 v7) := by
    apply row_46
    exact row_50
    exact row_7
    exact row_47
  -- chapter_48_line_4: GL tag implication.
  have row_4 : (add v3 v8 v9) := by
    apply row_5
    exact row_6
    exact row_38
    exact row_7
    exact row_41
  -- chapter_48_line_2: GL tag theorem.
  have row_2 := peano_source_004 N zero succ add mul one anchor relationalInduction
  -- chapter_48_line_1: GL tag implication.
  have row_1 : (add v3 v5 v7) := by
    apply row_2
    exact row_45
    exact row_4
    exact row_42
  exact row_1

theorem peano_source_018
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α) (v3 : α), ((mul v1 v2 v3) → (∀ (v4 : α) (v5 : α), ((mul v1 v4 v5) → (∀ (v6 : α) (v7 : α), ((mul v1 v6 v7) → ((add v2 v4 v6) → (add v3 v5 v7)))))))) := by
  intro v1
  intro v2
  intro v3
  intro premise_1
  intro v4
  intro v5
  intro premise_2
  intro v6
  intro v7
  intro premise_3
  intro premise_4
  have inductionMember : N v4 := by
    have typingRule := peano_source_018_induction_typing N zero succ add mul one anchor relationalInduction
    exact typingRule v2 v4 v6 premise_4
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v1 : α) (v2 : α) (v3 : α), ((mul v1 v2 v3) → (∀ (v5 : α), ((mul v1 zero v5) → (∀ (v6 : α) (v7 : α), ((mul v1 v6 v7) → ((add v2 zero v6) → (add v3 v5 v7)))))))) := by
    intro v1
    intro v2
    intro v3
    intro base_premise_1
    intro v5
    intro base_premise_2
    intro v6
    intro v7
    intro base_premise_3
    intro base_premise_4
    have zeroRule := peano_source_018_check_zero N zero succ add mul one anchor relationalInduction
    exact zeroRule v1 v2 v3 zero v5 v6 v7 base_premise_4 base_premise_3 rfl base_premise_1 base_premise_2
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v1 : α) (v2 : α) (v3 : α), ((mul v1 v2 v3) → (∀ (v5 : α), ((mul v1 induction_n v5) → (∀ (v6 : α) (v7 : α), ((mul v1 v6 v7) → ((add v2 induction_n v6) → (add v3 v5 v7)))))))) → ∀ induction_m, succ induction_n induction_m → (∀ (v1 : α) (v2 : α) (v3 : α), ((mul v1 v2 v3) → (∀ (v5 : α), ((mul v1 induction_m v5) → (∀ (v6 : α) (v7 : α), ((mul v1 v6 v7) → ((add v2 induction_m v6) → (add v3 v5 v7)))))))) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro v1
    intro v2
    intro v3
    intro step_premise_1
    intro v5
    intro step_premise_2
    intro v6
    intro v7
    intro step_premise_3
    intro step_premise_4
    have step_induction_assumption_1 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have step_induction_assumption_2 :
        ((mul v1 v2 v3) → (∀ (w1 : α), ((mul v1 induction_n w1) → (∀ (w2 : α) (w3 : α), ((mul v1 w2 w3) → ((add v2 induction_n w2) → (add v3 w1 w3))))))) := by
      intro step_induction_assumption_2_premise_1
      intro w1
      intro step_induction_assumption_2_premise_2
      intro w2
      intro w3
      intro step_induction_assumption_2_premise_3
      intro step_induction_assumption_2_premise_4
      apply induction_hypothesis
      all_goals assumption
    have stepRule := peano_source_018_check_induction_condition N zero succ add mul one anchor relationalInduction
    exact stepRule induction_n v1 v2 v3 induction_m v5 v6 v7 step_premise_3 step_induction_assumption_1 step_premise_4 step_premise_2 step_premise_1 step_induction_assumption_2
  have inductionProperty : (∀ (v1 : α) (v2 : α) (v3 : α), ((mul v1 v2 v3) → (∀ (v5 : α), ((mul v1 v4 v5) → (∀ (v6 : α) (v7 : α), ((mul v1 v6 v7) → ((add v2 v4 v6) → (add v3 v5 v7)))))))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v1 : α) (v2 : α) (v3 : α), ((mul v1 v2 v3) → (∀ (v5 : α), ((mul v1 induction_value v5) → (∀ (v6 : α) (v7 : α), ((mul v1 v6 v7) → ((add v2 induction_value v6) → (add v3 v5 v7)))))))))
      v4
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v1 v2 v3 premise_1 v5 premise_2 v6 v7 premise_3 premise_4

private theorem peano_source_038_induction_typing
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (assumption_10 : (mul one v1 v2))
    : (N v1) := by
  -- chapter_82_line_10: GL tag task formulation.
  have row_10 : (mul one v1 v2) := by
    exact assumption_10
  -- chapter_82_line_9: GL tag task formulation.
  have row_9 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_82_line_8: GL tag expansion.
  have row_8 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_9
  -- chapter_82_line_7: GL tag disintegration.
  have row_7 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_8.1
  -- chapter_82_line_6: GL tag expansion.
  have row_6 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_7
  -- chapter_82_line_5: GL tag disintegration.
  have row_5 : (gl_fXYZ mul N N N) := by
    exact row_6.1.1.1.2
  -- chapter_82_line_4: GL tag expansion.
  have row_4 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_5
  -- chapter_82_line_3: GL tag disintegration.
  have row_3 : (gl_implication9 mul N) := by
    exact row_4.1.1.1.2
  -- chapter_82_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication9] using row_3
  -- chapter_82_line_1: GL tag implication.
  have row_1 : (N v1) := by
    apply row_2
    exact row_10
  exact row_1

private theorem peano_source_038_check_zero
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (assumption_24 : (v1 = zero))
    (assumption_23 : (mul one v1 v2))
    : (mul v1 one v2) := by
  -- chapter_83_line_38: GL tag theorem.
  have row_38 := peano_source_041 N zero succ add mul one anchor relationalInduction
  -- chapter_83_line_24: GL tag recursion.
  have row_24 : (v1 = zero) := by
    exact assumption_24
  -- chapter_83_line_25: GL tag symmetry of equality.
  have row_25 : (zero = v1) := by
    exact Eq.symm row_24
  -- chapter_83_line_23: GL tag task formulation.
  have row_23 : (mul one v1 v2) := by
    exact assumption_23
  -- chapter_83_line_55: GL tag equality1.
  have row_55 : (mul one zero v2) := by
    have equality_source := row_23
    have equality_step_1 := row_24
    cases equality_step_1
    exact equality_source
  -- chapter_83_line_13: GL tag task formulation.
  have row_13 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_83_line_12: GL tag expansion.
  have row_12 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_13
  -- chapter_83_line_19: GL tag disintegration.
  have row_19 : (succ zero one) := by
    exact row_12.2
  -- chapter_83_line_11: GL tag disintegration.
  have row_11 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_12.1
  -- chapter_83_line_10: GL tag expansion.
  have row_10 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_11
  -- chapter_83_line_52: GL tag disintegration.
  have row_52 : (gl_implication21 N succ mul add) := by
    exact row_10.2
  -- chapter_83_line_51: GL tag expansion.
  have row_51 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((mul w3 w1 w4) → (∀ (w5 : α), ((mul w3 w2 w5) → (add w4 w3 w5))))))))) := by
    simpa only [gl_implication21] using row_52
  -- chapter_83_line_46: GL tag disintegration.
  have row_46 : (gl_implication19 N zero mul) := by
    exact row_10.1.1.2
  -- chapter_83_line_45: GL tag expansion.
  have row_45 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((mul w1 zero w2) → (w2 = zero))))) := by
    simpa only [gl_implication19] using row_46
  -- chapter_83_line_42: GL tag disintegration.
  have row_42 : (N zero) := by
    exact row_10.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_83_line_36: GL tag disintegration.
  have row_36 : (gl_implication16 N zero add) := by
    exact row_10.1.1.1.1.1.1.2
  -- chapter_83_line_35: GL tag expansion.
  have row_35 : (∀ (w1 : α) (w2 : α), ((w1 = w2) → ((N w1) → ((N w2) → (add w1 zero w2))))) := by
    simpa only [gl_implication16] using row_36
  -- chapter_83_line_31: GL tag disintegration.
  have row_31 : (gl_fXYZ add N N N) := by
    exact row_10.1.1.1.1.1.1.1.1.2
  -- chapter_83_line_30: GL tag expansion.
  have row_30 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_31
  -- chapter_83_line_29: GL tag disintegration.
  have row_29 : (gl_implication14 N N add) := by
    exact row_30.2
  -- chapter_83_line_28: GL tag expansion.
  have row_28 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (∀ (w3 : α), ((add w1 w2 w3) → (∀ (w4 : α), ((add w1 w2 w4) → (w3 = w4))))))))) := by
    simpa only [gl_implication14] using row_29
  -- chapter_83_line_18: GL tag disintegration.
  have row_18 : (gl_fXY succ N N) := by
    exact row_10.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_83_line_17: GL tag expansion.
  have row_17 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_18
  -- chapter_83_line_16: GL tag disintegration.
  have row_16 : (gl_implication1 succ N) := by
    exact row_17.1.1.2
  -- chapter_83_line_15: GL tag expansion.
  have row_15 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using row_16
  -- chapter_83_line_14: GL tag implication.
  have row_14 : (N one) := by
    apply row_15
    exact row_19
  -- chapter_83_line_54: GL tag implication.
  have row_54 : (v2 = zero) := by
    apply row_45
    exact row_14
    exact row_55
  -- chapter_83_line_53: GL tag symmetry of equality.
  have row_53 : (zero = v2) := by
    exact Eq.symm row_54
  -- chapter_83_line_9: GL tag disintegration.
  have row_9 : (gl_fXYZ mul N N N) := by
    exact row_10.1.1.1.2
  -- chapter_83_line_8: GL tag expansion.
  have row_8 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_9
  -- chapter_83_line_22: GL tag disintegration.
  have row_22 : (gl_implication9 mul N) := by
    exact row_8.1.1.1.2
  -- chapter_83_line_21: GL tag expansion.
  have row_21 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication9] using row_22
  -- chapter_83_line_20: GL tag implication.
  have row_20 : (N v1) := by
    apply row_21
    exact row_23
  -- chapter_83_line_7: GL tag disintegration.
  have row_7 : (gl_implication13 N N N mul) := by
    exact row_8.1.2
  -- chapter_83_line_6: GL tag expansion.
  have row_6 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication13] using row_7
  -- chapter_83_line_41: GL tag implication.
  have row_41 : (gl_existence1 N v1 zero mul) := by
    apply row_6
    exact row_20
    exact row_42
  -- chapter_83_line_40: GL tag expansion.
  have row_40 : (¬ (∀ (v8 : α), ((N v8) → (¬ (mul v1 zero v8))))) := by
    simpa only [gl_existence1] using row_41
  have exists_row_40 : ∃ (v8 : α), ((N v8) ∧ (mul v1 zero v8)) := existsAndOfNotForallImpNot row_40
  obtain ⟨v8, witness_row_40⟩ := exists_row_40
  -- chapter_83_line_43: GL tag disintegration.
  have row_43 : (N v8) := by
    exact witness_row_40.1
  -- chapter_83_line_39: GL tag disintegration.
  have row_39 : (mul v1 zero v8) := by
    exact witness_row_40.2
  -- chapter_83_line_44: GL tag implication.
  have row_44 : (v8 = zero) := by
    apply row_45
    exact row_20
    exact row_39
  -- chapter_83_line_37: GL tag implication.
  have row_37 : (zero = v8) := by
    apply row_38
    exact row_39
    exact row_20
  -- chapter_83_line_34: GL tag implication.
  have row_34 : (add zero zero v8) := by
    apply row_35
    exact row_37
    exact row_42
    exact row_43
  -- chapter_83_line_33: GL tag equality1.
  have row_33 : (add zero zero zero) := by
    have equality_source := row_34
    have equality_step_1 := row_44
    cases equality_step_1
    exact equality_source
  -- chapter_83_line_32: GL tag equality1.
  have row_32 : (add v1 v8 zero) := by
    have equality_source := row_33
    have equality_step_1 := row_25
    cases equality_step_1
    have equality_step_2 := row_37
    cases equality_step_2
    exact equality_source
  -- chapter_83_line_5: GL tag implication.
  have row_5 : (gl_existence1 N v1 one mul) := by
    apply row_6
    exact row_20
    exact row_14
  -- chapter_83_line_4: GL tag expansion.
  have row_4 : (¬ (∀ (v3 : α), ((N v3) → (¬ (mul v1 one v3))))) := by
    simpa only [gl_existence1] using row_5
  have exists_row_4 : ∃ (v3 : α), ((N v3) ∧ (mul v1 one v3)) := existsAndOfNotForallImpNot row_4
  obtain ⟨v3, witness_row_4⟩ := exists_row_4
  -- chapter_83_line_3: GL tag disintegration.
  have row_3 : (mul v1 one v3) := by
    exact witness_row_4.2
  -- chapter_83_line_50: GL tag implication.
  have row_50 : (add v8 v1 v3) := by
    apply row_51
    exact row_42
    exact row_19
    exact row_39
    exact row_3
  -- chapter_83_line_49: GL tag equality1.
  have row_49 : (add zero v1 v3) := by
    have equality_source := row_50
    have equality_step_1 := row_44
    cases equality_step_1
    exact equality_source
  -- chapter_83_line_48: GL tag equality1.
  have row_48 : (add zero zero v3) := by
    have equality_source := row_49
    have equality_step_1 := row_24
    cases equality_step_1
    exact equality_source
  -- chapter_83_line_47: GL tag equality1.
  have row_47 : (add v1 v8 v3) := by
    have equality_source := row_48
    have equality_step_1 := row_25
    cases equality_step_1
    have equality_step_2 := row_37
    cases equality_step_2
    exact equality_source
  -- chapter_83_line_27: GL tag implication.
  have row_27 : (v3 = zero) := by
    apply row_28
    exact row_20
    exact row_43
    exact row_47
    exact row_32
  -- chapter_83_line_26: GL tag equality2.
  have row_26 : (v3 = v2) := by
    exact Eq.trans row_27 row_53
  -- chapter_83_line_2: GL tag equality1.
  have row_2 : (mul zero one v3) := by
    have equality_source := row_3
    have equality_step_1 := row_24
    cases equality_step_1
    exact equality_source
  -- chapter_83_line_1: GL tag equality1.
  have row_1 : (mul v1 one v2) := by
    have equality_source := row_2
    have equality_step_1 := row_25
    cases equality_step_1
    have equality_step_2 := row_26
    cases equality_step_2
    exact equality_source
  exact row_1

private theorem peano_source_038_check_induction_condition
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (assumption_50 : (∀ (w1 : α), ((mul one previous w1) → (mul previous one w1))))
    (assumption_29 : (mul one v1 v2))
    (assumption_20 : (succ previous v1))
    : (mul v1 one v2) := by
  -- chapter_84_line_55: GL tag theorem.
  have row_55 := peano_source_041 N zero succ add mul one anchor relationalInduction
  -- chapter_84_line_50: GL tag recursion.
  have row_50 : (∀ (w1 : α), ((mul one previous w1) → (mul previous one w1))) := by
    exact assumption_50
  -- chapter_84_line_37: GL tag theorem.
  have row_37 := peano_source_006 N zero succ add mul one anchor relationalInduction
  -- chapter_84_line_29: GL tag task formulation.
  have row_29 : (mul one v1 v2) := by
    exact assumption_29
  -- chapter_84_line_25: GL tag theorem.
  have row_25 := peano_source_043 N zero succ add mul one anchor relationalInduction
  -- chapter_84_line_20: GL tag recursion.
  have row_20 : (succ previous v1) := by
    exact assumption_20
  -- chapter_84_line_12: GL tag task formulation.
  have row_12 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_84_line_11: GL tag expansion.
  have row_11 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_12
  -- chapter_84_line_18: GL tag disintegration.
  have row_18 : (succ zero one) := by
    exact row_11.2
  -- chapter_84_line_10: GL tag disintegration.
  have row_10 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_11.1
  -- chapter_84_line_9: GL tag expansion.
  have row_9 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_10
  -- chapter_84_line_53: GL tag disintegration.
  have row_53 : (gl_implication19 N zero mul) := by
    exact row_9.1.1.2
  -- chapter_84_line_52: GL tag expansion.
  have row_52 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((mul w1 zero w2) → (w2 = zero))))) := by
    simpa only [gl_implication19] using row_53
  -- chapter_84_line_42: GL tag disintegration.
  have row_42 : (N zero) := by
    exact row_9.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_84_line_28: GL tag disintegration.
  have row_28 : (gl_implication21 N succ mul add) := by
    exact row_9.2
  -- chapter_84_line_27: GL tag expansion.
  have row_27 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((mul w3 w1 w4) → (∀ (w5 : α), ((mul w3 w2 w5) → (add w4 w3 w5))))))))) := by
    simpa only [gl_implication21] using row_28
  -- chapter_84_line_17: GL tag disintegration.
  have row_17 : (gl_fXY succ N N) := by
    exact row_9.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_84_line_16: GL tag expansion.
  have row_16 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_17
  -- chapter_84_line_35: GL tag disintegration.
  have row_35 : (gl_implication0 succ N) := by
    exact row_16.1.1.1
  -- chapter_84_line_34: GL tag expansion.
  have row_34 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_35
  -- chapter_84_line_33: GL tag implication.
  have row_33 : (N previous) := by
    apply row_34
    exact row_20
  -- chapter_84_line_23: GL tag disintegration.
  have row_23 : (gl_implication5 N succ) := by
    exact row_16.2
  -- chapter_84_line_22: GL tag expansion.
  have row_22 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α), ((succ w1 w3) → (w2 = w3))))))) := by
    simpa only [gl_implication5] using row_23
  -- chapter_84_line_15: GL tag disintegration.
  have row_15 : (gl_implication1 succ N) := by
    exact row_16.1.1.2
  -- chapter_84_line_14: GL tag expansion.
  have row_14 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using row_15
  -- chapter_84_line_19: GL tag implication.
  have row_19 : (N v1) := by
    apply row_14
    exact row_20
  -- chapter_84_line_13: GL tag implication.
  have row_13 : (N one) := by
    apply row_14
    exact row_18
  -- chapter_84_line_8: GL tag disintegration.
  have row_8 : (gl_fXYZ mul N N N) := by
    exact row_9.1.1.1.2
  -- chapter_84_line_7: GL tag expansion.
  have row_7 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_8
  -- chapter_84_line_6: GL tag disintegration.
  have row_6 : (gl_implication13 N N N mul) := by
    exact row_7.1.2
  -- chapter_84_line_5: GL tag expansion.
  have row_5 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication13] using row_6
  -- chapter_84_line_48: GL tag implication.
  have row_48 : (gl_existence1 N previous zero mul) := by
    apply row_5
    exact row_33
    exact row_42
  -- chapter_84_line_47: GL tag expansion.
  have row_47 : (¬ (∀ (v11 : α), ((N v11) → (¬ (mul previous zero v11))))) := by
    simpa only [gl_existence1] using row_48
  have exists_row_47 : ∃ (v11 : α), ((N v11) ∧ (mul previous zero v11)) := existsAndOfNotForallImpNot row_47
  obtain ⟨v11, witness_row_47⟩ := exists_row_47
  -- chapter_84_line_46: GL tag disintegration.
  have row_46 : (mul previous zero v11) := by
    exact witness_row_47.2
  -- chapter_84_line_51: GL tag implication.
  have row_51 : (v11 = zero) := by
    apply row_52
    exact row_33
    exact row_46
  -- chapter_84_line_41: GL tag implication.
  have row_41 : (gl_existence1 N v1 zero mul) := by
    apply row_5
    exact row_19
    exact row_42
  -- chapter_84_line_40: GL tag expansion.
  have row_40 : (¬ (∀ (v10 : α), ((N v10) → (¬ (mul v1 zero v10))))) := by
    simpa only [gl_existence1] using row_41
  have exists_row_40 : ∃ (v10 : α), ((N v10) ∧ (mul v1 zero v10)) := existsAndOfNotForallImpNot row_40
  obtain ⟨v10, witness_row_40⟩ := exists_row_40
  -- chapter_84_line_39: GL tag disintegration.
  have row_39 : (mul v1 zero v10) := by
    exact witness_row_40.2
  -- chapter_84_line_54: GL tag implication.
  have row_54 : (zero = v10) := by
    apply row_55
    exact row_39
    exact row_19
  -- chapter_84_line_32: GL tag implication.
  have row_32 : (gl_existence1 N one previous mul) := by
    apply row_5
    exact row_13
    exact row_33
  -- chapter_84_line_31: GL tag expansion.
  have row_31 : (¬ (∀ (v7 : α), ((N v7) → (¬ (mul one previous v7))))) := by
    simpa only [gl_existence1] using row_32
  have exists_row_31 : ∃ (v7 : α), ((N v7) ∧ (mul one previous v7)) := existsAndOfNotForallImpNot row_31
  obtain ⟨v7, witness_row_31⟩ := exists_row_31
  -- chapter_84_line_56: GL tag disintegration.
  have row_56 : (N v7) := by
    exact witness_row_31.1
  -- chapter_84_line_30: GL tag disintegration.
  have row_30 : (mul one previous v7) := by
    exact witness_row_31.2
  -- chapter_84_line_49: GL tag implication.
  have row_49 : (mul previous one v7) := by
    apply row_50
    exact row_30
  -- chapter_84_line_45: GL tag implication.
  have row_45 : (add v11 previous v7) := by
    apply row_27
    exact row_42
    exact row_18
    exact row_46
    exact row_49
  -- chapter_84_line_44: GL tag equality1.
  have row_44 : (add zero previous v7) := by
    have equality_source := row_45
    have equality_step_1 := row_51
    cases equality_step_1
    exact equality_source
  -- chapter_84_line_43: GL tag equality1.
  have row_43 : (add v10 previous v7) := by
    have equality_source := row_44
    have equality_step_1 := row_54
    cases equality_step_1
    exact equality_source
  -- chapter_84_line_26: GL tag implication.
  have row_26 : (add v7 one v2) := by
    apply row_27
    exact row_33
    exact row_20
    exact row_30
    exact row_29
  -- chapter_84_line_24: GL tag implication.
  have row_24 : (succ v7 v2) := by
    apply row_25
    exact row_26
  -- chapter_84_line_4: GL tag implication.
  have row_4 : (gl_existence1 N v1 one mul) := by
    apply row_5
    exact row_19
    exact row_13
  -- chapter_84_line_3: GL tag expansion.
  have row_3 : (¬ (∀ (v3 : α), ((N v3) → (¬ (mul v1 one v3))))) := by
    simpa only [gl_existence1] using row_4
  have exists_row_3 : ∃ (v3 : α), ((N v3) ∧ (mul v1 one v3)) := existsAndOfNotForallImpNot row_3
  obtain ⟨v3, witness_row_3⟩ := exists_row_3
  -- chapter_84_line_2: GL tag disintegration.
  have row_2 : (mul v1 one v3) := by
    exact witness_row_3.2
  -- chapter_84_line_38: GL tag implication.
  have row_38 : (add v10 v1 v3) := by
    apply row_27
    exact row_42
    exact row_18
    exact row_39
    exact row_2
  -- chapter_84_line_36: GL tag implication.
  have row_36 : (succ v7 v3) := by
    apply row_37
    exact row_38
    exact row_43
    exact row_20
  -- chapter_84_line_21: GL tag implication.
  have row_21 : (v3 = v2) := by
    apply row_22
    exact row_56
    exact row_36
    exact row_24
  -- chapter_84_line_1: GL tag equality1.
  have row_1 : (mul v1 one v2) := by
    have equality_source := row_2
    have equality_step_1 := row_21
    cases equality_step_1
    exact equality_source
  exact row_1

theorem peano_source_038
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((mul one v1 v2) → (mul v1 one v2))) := by
  intro v1
  intro v2
  intro premise_1
  have inductionMember : N v1 := by
    have typingRule := peano_source_038_induction_typing N zero succ add mul one anchor relationalInduction
    exact typingRule v1 v2 premise_1
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v2 : α), ((mul one zero v2) → (mul zero one v2))) := by
    intro v2
    intro base_premise_1
    have zeroRule := peano_source_038_check_zero N zero succ add mul one anchor relationalInduction
    exact zeroRule zero v2 rfl base_premise_1
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v2 : α), ((mul one induction_n v2) → (mul induction_n one v2))) → ∀ induction_m, succ induction_n induction_m → (∀ (v2 : α), ((mul one induction_m v2) → (mul induction_m one v2))) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro v2
    intro step_premise_1
    have step_induction_assumption_1 :
        (∀ (w1 : α), ((mul one induction_n w1) → (mul induction_n one w1))) := by
      intro w1
      intro step_induction_assumption_1_premise_1
      apply induction_hypothesis
      all_goals assumption
    have step_induction_assumption_2 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have stepRule := peano_source_038_check_induction_condition N zero succ add mul one anchor relationalInduction
    exact stepRule induction_n induction_m v2 step_induction_assumption_1 step_premise_1 step_induction_assumption_2
  have inductionProperty : (∀ (v2 : α), ((mul one v1 v2) → (mul v1 one v2))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v2 : α), ((mul one induction_value v2) → (mul induction_value one v2))))
      v1
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v2 premise_1

private theorem peano_source_044_induction_typing
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (assumption_10 : (mul v1 one v2))
    : (N v1) := by
  -- chapter_94_line_10: GL tag task formulation.
  have row_10 : (mul v1 one v2) := by
    exact assumption_10
  -- chapter_94_line_9: GL tag task formulation.
  have row_9 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_94_line_8: GL tag expansion.
  have row_8 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_9
  -- chapter_94_line_7: GL tag disintegration.
  have row_7 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_8.1
  -- chapter_94_line_6: GL tag expansion.
  have row_6 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_7
  -- chapter_94_line_5: GL tag disintegration.
  have row_5 : (gl_fXYZ mul N N N) := by
    exact row_6.1.1.1.2
  -- chapter_94_line_4: GL tag expansion.
  have row_4 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_5
  -- chapter_94_line_3: GL tag disintegration.
  have row_3 : (gl_implication8 mul N) := by
    exact row_4.1.1.1.1
  -- chapter_94_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_3
  -- chapter_94_line_1: GL tag implication.
  have row_1 : (N v1) := by
    apply row_2
    exact row_10
  exact row_1

private theorem peano_source_044_check_zero
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (assumption_24 : (v1 = zero))
    (assumption_23 : (mul v1 one v2))
    : (mul one v1 v2) := by
  -- chapter_95_line_41: GL tag theorem.
  have row_41 := peano_source_041 N zero succ add mul one anchor relationalInduction
  -- chapter_95_line_24: GL tag recursion.
  have row_24 : (v1 = zero) := by
    exact assumption_24
  -- chapter_95_line_25: GL tag symmetry of equality.
  have row_25 : (zero = v1) := by
    exact Eq.symm row_24
  -- chapter_95_line_23: GL tag task formulation.
  have row_23 : (mul v1 one v2) := by
    exact assumption_23
  -- chapter_95_line_13: GL tag task formulation.
  have row_13 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_95_line_12: GL tag expansion.
  have row_12 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_13
  -- chapter_95_line_19: GL tag disintegration.
  have row_19 : (succ zero one) := by
    exact row_12.2
  -- chapter_95_line_11: GL tag disintegration.
  have row_11 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_12.1
  -- chapter_95_line_10: GL tag expansion.
  have row_10 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_11
  -- chapter_95_line_53: GL tag disintegration.
  have row_53 : (gl_implication21 N succ mul add) := by
    exact row_10.2
  -- chapter_95_line_52: GL tag expansion.
  have row_52 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((mul w3 w1 w4) → (∀ (w5 : α), ((mul w3 w2 w5) → (add w4 w3 w5))))))))) := by
    simpa only [gl_implication21] using row_53
  -- chapter_95_line_45: GL tag disintegration.
  have row_45 : (N zero) := by
    exact row_10.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_95_line_39: GL tag disintegration.
  have row_39 : (gl_implication16 N zero add) := by
    exact row_10.1.1.1.1.1.1.2
  -- chapter_95_line_38: GL tag expansion.
  have row_38 : (∀ (w1 : α) (w2 : α), ((w1 = w2) → ((N w1) → ((N w2) → (add w1 zero w2))))) := by
    simpa only [gl_implication16] using row_39
  -- chapter_95_line_34: GL tag disintegration.
  have row_34 : (gl_fXYZ add N N N) := by
    exact row_10.1.1.1.1.1.1.1.1.2
  -- chapter_95_line_33: GL tag expansion.
  have row_33 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_34
  -- chapter_95_line_32: GL tag disintegration.
  have row_32 : (gl_implication14 N N add) := by
    exact row_33.2
  -- chapter_95_line_31: GL tag expansion.
  have row_31 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (∀ (w3 : α), ((add w1 w2 w3) → (∀ (w4 : α), ((add w1 w2 w4) → (w3 = w4))))))))) := by
    simpa only [gl_implication14] using row_32
  -- chapter_95_line_29: GL tag disintegration.
  have row_29 : (gl_implication19 N zero mul) := by
    exact row_10.1.1.2
  -- chapter_95_line_28: GL tag expansion.
  have row_28 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((mul w1 zero w2) → (w2 = zero))))) := by
    simpa only [gl_implication19] using row_29
  -- chapter_95_line_18: GL tag disintegration.
  have row_18 : (gl_fXY succ N N) := by
    exact row_10.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_95_line_17: GL tag expansion.
  have row_17 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_18
  -- chapter_95_line_16: GL tag disintegration.
  have row_16 : (gl_implication1 succ N) := by
    exact row_17.1.1.2
  -- chapter_95_line_15: GL tag expansion.
  have row_15 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using row_16
  -- chapter_95_line_14: GL tag implication.
  have row_14 : (N one) := by
    apply row_15
    exact row_19
  -- chapter_95_line_9: GL tag disintegration.
  have row_9 : (gl_fXYZ mul N N N) := by
    exact row_10.1.1.1.2
  -- chapter_95_line_8: GL tag expansion.
  have row_8 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_9
  -- chapter_95_line_22: GL tag disintegration.
  have row_22 : (gl_implication8 mul N) := by
    exact row_8.1.1.1.1
  -- chapter_95_line_21: GL tag expansion.
  have row_21 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_22
  -- chapter_95_line_20: GL tag implication.
  have row_20 : (N v1) := by
    apply row_21
    exact row_23
  -- chapter_95_line_7: GL tag disintegration.
  have row_7 : (gl_implication13 N N N mul) := by
    exact row_8.1.2
  -- chapter_95_line_6: GL tag expansion.
  have row_6 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication13] using row_7
  -- chapter_95_line_44: GL tag implication.
  have row_44 : (gl_existence1 N v1 zero mul) := by
    apply row_6
    exact row_20
    exact row_45
  -- chapter_95_line_43: GL tag expansion.
  have row_43 : (¬ (∀ (v8 : α), ((N v8) → (¬ (mul v1 zero v8))))) := by
    simpa only [gl_existence1] using row_44
  have exists_row_43 : ∃ (v8 : α), ((N v8) ∧ (mul v1 zero v8)) := existsAndOfNotForallImpNot row_43
  obtain ⟨v8, witness_row_43⟩ := exists_row_43
  -- chapter_95_line_46: GL tag disintegration.
  have row_46 : (N v8) := by
    exact witness_row_43.1
  -- chapter_95_line_42: GL tag disintegration.
  have row_42 : (mul v1 zero v8) := by
    exact witness_row_43.2
  -- chapter_95_line_51: GL tag implication.
  have row_51 : (add v8 v1 v2) := by
    apply row_52
    exact row_45
    exact row_19
    exact row_42
    exact row_23
  -- chapter_95_line_47: GL tag implication.
  have row_47 : (v8 = zero) := by
    apply row_28
    exact row_20
    exact row_42
  -- chapter_95_line_50: GL tag equality1.
  have row_50 : (add zero v1 v2) := by
    have equality_source := row_51
    have equality_step_1 := row_47
    cases equality_step_1
    exact equality_source
  -- chapter_95_line_49: GL tag equality1.
  have row_49 : (add zero zero v2) := by
    have equality_source := row_50
    have equality_step_1 := row_24
    cases equality_step_1
    exact equality_source
  -- chapter_95_line_40: GL tag implication.
  have row_40 : (zero = v8) := by
    apply row_41
    exact row_42
    exact row_20
  -- chapter_95_line_48: GL tag equality1.
  have row_48 : (add v1 v8 v2) := by
    have equality_source := row_49
    have equality_step_1 := row_25
    cases equality_step_1
    have equality_step_2 := row_40
    cases equality_step_2
    exact equality_source
  -- chapter_95_line_37: GL tag implication.
  have row_37 : (add zero zero v8) := by
    apply row_38
    exact row_40
    exact row_45
    exact row_46
  -- chapter_95_line_36: GL tag equality1.
  have row_36 : (add zero zero zero) := by
    have equality_source := row_37
    have equality_step_1 := row_47
    cases equality_step_1
    exact equality_source
  -- chapter_95_line_35: GL tag equality1.
  have row_35 : (add v1 v8 zero) := by
    have equality_source := row_36
    have equality_step_1 := row_25
    cases equality_step_1
    have equality_step_2 := row_40
    cases equality_step_2
    exact equality_source
  -- chapter_95_line_30: GL tag implication.
  have row_30 : (zero = v2) := by
    apply row_31
    exact row_20
    exact row_46
    exact row_35
    exact row_48
  -- chapter_95_line_5: GL tag implication.
  have row_5 : (gl_existence1 N one v1 mul) := by
    apply row_6
    exact row_14
    exact row_20
  -- chapter_95_line_4: GL tag expansion.
  have row_4 : (¬ (∀ (v3 : α), ((N v3) → (¬ (mul one v1 v3))))) := by
    simpa only [gl_existence1] using row_5
  have exists_row_4 : ∃ (v3 : α), ((N v3) ∧ (mul one v1 v3)) := existsAndOfNotForallImpNot row_4
  obtain ⟨v3, witness_row_4⟩ := exists_row_4
  -- chapter_95_line_3: GL tag disintegration.
  have row_3 : (mul one v1 v3) := by
    exact witness_row_4.2
  -- chapter_95_line_2: GL tag equality1.
  have row_2 : (mul one zero v3) := by
    have equality_source := row_3
    have equality_step_1 := row_24
    cases equality_step_1
    exact equality_source
  -- chapter_95_line_27: GL tag implication.
  have row_27 : (v3 = zero) := by
    apply row_28
    exact row_14
    exact row_2
  -- chapter_95_line_26: GL tag equality2.
  have row_26 : (v3 = v2) := by
    exact Eq.trans row_27 row_30
  -- chapter_95_line_1: GL tag equality1.
  have row_1 : (mul one v1 v2) := by
    have equality_source := row_2
    have equality_step_1 := row_25
    cases equality_step_1
    have equality_step_2 := row_26
    cases equality_step_2
    exact equality_source
  exact row_1

private theorem peano_source_044_check_induction_condition
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (assumption_76 : (∀ (w1 : α), ((mul previous one w1) → (mul one previous w1))))
    (assumption_33 : (mul v1 one v2))
    (assumption_20 : (succ previous v1))
    : (mul one v1 v2) := by
  -- chapter_96_line_76: GL tag recursion.
  have row_76 : (∀ (w1 : α), ((mul previous one w1) → (mul one previous w1))) := by
    exact assumption_76
  -- chapter_96_line_59: GL tag theorem.
  have row_59 := peano_source_020 N zero succ add mul one anchor relationalInduction
  -- chapter_96_line_49: GL tag theorem.
  have row_49 := peano_source_043 N zero succ add mul one anchor relationalInduction
  -- chapter_96_line_46: GL tag theorem.
  have row_46 := peano_source_041 N zero succ add mul one anchor relationalInduction
  -- chapter_96_line_33: GL tag task formulation.
  have row_33 : (mul v1 one v2) := by
    exact assumption_33
  -- chapter_96_line_25: GL tag theorem.
  have row_25 := peano_source_006 N zero succ add mul one anchor relationalInduction
  -- chapter_96_line_20: GL tag recursion.
  have row_20 : (succ previous v1) := by
    exact assumption_20
  -- chapter_96_line_12: GL tag task formulation.
  have row_12 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_96_line_60: GL tag anchor handling.
  have row_60 : (gl_AnchorPeano N zero succ add mul one) := by
    exact row_12
  -- chapter_96_line_11: GL tag expansion.
  have row_11 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_12
  -- chapter_96_line_18: GL tag disintegration.
  have row_18 : (succ zero one) := by
    exact row_11.2
  -- chapter_96_line_10: GL tag disintegration.
  have row_10 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_11.1
  -- chapter_96_line_9: GL tag expansion.
  have row_9 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_10
  -- chapter_96_line_71: GL tag disintegration.
  have row_71 : (gl_implication19 N zero mul) := by
    exact row_9.1.1.2
  -- chapter_96_line_70: GL tag expansion.
  have row_70 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((mul w1 zero w2) → (w2 = zero))))) := by
    simpa only [gl_implication19] using row_71
  -- chapter_96_line_41: GL tag disintegration.
  have row_41 : (gl_fXYZ add N N N) := by
    exact row_9.1.1.1.1.1.1.1.1.2
  -- chapter_96_line_40: GL tag expansion.
  have row_40 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_41
  -- chapter_96_line_56: GL tag disintegration.
  have row_56 : (gl_implication14 N N add) := by
    exact row_40.2
  -- chapter_96_line_55: GL tag expansion.
  have row_55 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (∀ (w3 : α), ((add w1 w2 w3) → (∀ (w4 : α), ((add w1 w2 w4) → (w3 = w4))))))))) := by
    simpa only [gl_implication14] using row_56
  -- chapter_96_line_39: GL tag disintegration.
  have row_39 : (gl_implication13 N N N add) := by
    exact row_40.1.2
  -- chapter_96_line_38: GL tag expansion.
  have row_38 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication13] using row_39
  -- chapter_96_line_32: GL tag disintegration.
  have row_32 : (N zero) := by
    exact row_9.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_96_line_28: GL tag disintegration.
  have row_28 : (gl_implication21 N succ mul add) := by
    exact row_9.2
  -- chapter_96_line_27: GL tag expansion.
  have row_27 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((mul w3 w1 w4) → (∀ (w5 : α), ((mul w3 w2 w5) → (add w4 w3 w5))))))))) := by
    simpa only [gl_implication21] using row_28
  -- chapter_96_line_17: GL tag disintegration.
  have row_17 : (gl_fXY succ N N) := by
    exact row_9.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_96_line_16: GL tag expansion.
  have row_16 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_17
  -- chapter_96_line_44: GL tag disintegration.
  have row_44 : (gl_implication0 succ N) := by
    exact row_16.1.1.1
  -- chapter_96_line_43: GL tag expansion.
  have row_43 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_44
  -- chapter_96_line_42: GL tag implication.
  have row_42 : (N previous) := by
    apply row_43
    exact row_20
  -- chapter_96_line_37: GL tag implication.
  have row_37 : (gl_existence1 N zero previous add) := by
    apply row_38
    exact row_32
    exact row_42
  -- chapter_96_line_36: GL tag expansion.
  have row_36 : (¬ (∀ (v7 : α), ((N v7) → (¬ (add zero previous v7))))) := by
    simpa only [gl_existence1] using row_37
  have exists_row_36 : ∃ (v7 : α), ((N v7) ∧ (add zero previous v7)) := existsAndOfNotForallImpNot row_36
  obtain ⟨v7, witness_row_36⟩ := exists_row_36
  -- chapter_96_line_77: GL tag disintegration.
  have row_77 : (N v7) := by
    exact witness_row_36.1
  -- chapter_96_line_35: GL tag disintegration.
  have row_35 : (add zero previous v7) := by
    exact witness_row_36.2
  -- chapter_96_line_23: GL tag disintegration.
  have row_23 : (gl_implication5 N succ) := by
    exact row_16.2
  -- chapter_96_line_22: GL tag expansion.
  have row_22 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α), ((succ w1 w3) → (w2 = w3))))))) := by
    simpa only [gl_implication5] using row_23
  -- chapter_96_line_15: GL tag disintegration.
  have row_15 : (gl_implication1 succ N) := by
    exact row_16.1.1.2
  -- chapter_96_line_14: GL tag expansion.
  have row_14 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using row_15
  -- chapter_96_line_19: GL tag implication.
  have row_19 : (N v1) := by
    apply row_14
    exact row_20
  -- chapter_96_line_13: GL tag implication.
  have row_13 : (N one) := by
    apply row_14
    exact row_18
  -- chapter_96_line_8: GL tag disintegration.
  have row_8 : (gl_fXYZ mul N N N) := by
    exact row_9.1.1.1.2
  -- chapter_96_line_7: GL tag expansion.
  have row_7 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_8
  -- chapter_96_line_74: GL tag disintegration.
  have row_74 : (gl_implication14 N N mul) := by
    exact row_7.2
  -- chapter_96_line_73: GL tag expansion.
  have row_73 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (∀ (w3 : α), ((mul w1 w2 w3) → (∀ (w4 : α), ((mul w1 w2 w4) → (w3 = w4))))))))) := by
    simpa only [gl_implication14] using row_74
  -- chapter_96_line_6: GL tag disintegration.
  have row_6 : (gl_implication13 N N N mul) := by
    exact row_7.1.2
  -- chapter_96_line_5: GL tag expansion.
  have row_5 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication13] using row_6
  -- chapter_96_line_68: GL tag implication.
  have row_68 : (gl_existence1 N previous zero mul) := by
    apply row_5
    exact row_42
    exact row_32
  -- chapter_96_line_67: GL tag expansion.
  have row_67 : (¬ (∀ (v12 : α), ((N v12) → (¬ (mul previous zero v12))))) := by
    simpa only [gl_existence1] using row_68
  have exists_row_67 : ∃ (v12 : α), ((N v12) ∧ (mul previous zero v12)) := existsAndOfNotForallImpNot row_67
  obtain ⟨v12, witness_row_67⟩ := exists_row_67
  -- chapter_96_line_66: GL tag disintegration.
  have row_66 : (mul previous zero v12) := by
    exact witness_row_67.2
  -- chapter_96_line_69: GL tag implication.
  have row_69 : (v12 = zero) := by
    apply row_70
    exact row_42
    exact row_66
  -- chapter_96_line_64: GL tag implication.
  have row_64 : (gl_existence1 N previous one mul) := by
    apply row_5
    exact row_42
    exact row_13
  -- chapter_96_line_63: GL tag expansion.
  have row_63 : (¬ (∀ (v13 : α), ((N v13) → (¬ (mul previous one v13))))) := by
    simpa only [gl_existence1] using row_64
  have exists_row_63 : ∃ (v13 : α), ((N v13) ∧ (mul previous one v13)) := existsAndOfNotForallImpNot row_63
  obtain ⟨v13, witness_row_63⟩ := exists_row_63
  -- chapter_96_line_62: GL tag disintegration.
  have row_62 : (mul previous one v13) := by
    exact witness_row_63.2
  -- chapter_96_line_75: GL tag implication.
  have row_75 : (mul one previous v13) := by
    apply row_76
    exact row_62
  -- chapter_96_line_53: GL tag implication.
  have row_53 : (gl_existence1 N one previous mul) := by
    apply row_5
    exact row_13
    exact row_42
  -- chapter_96_line_52: GL tag expansion.
  have row_52 : (¬ (∀ (v11 : α), ((N v11) → (¬ (mul one previous v11))))) := by
    simpa only [gl_existence1] using row_53
  have exists_row_52 : ∃ (v11 : α), ((N v11) ∧ (mul one previous v11)) := existsAndOfNotForallImpNot row_52
  obtain ⟨v11, witness_row_52⟩ := exists_row_52
  -- chapter_96_line_51: GL tag disintegration.
  have row_51 : (mul one previous v11) := by
    exact witness_row_52.2
  -- chapter_96_line_72: GL tag implication.
  have row_72 : (v13 = v11) := by
    apply row_73
    exact row_13
    exact row_42
    exact row_75
    exact row_51
  -- chapter_96_line_31: GL tag implication.
  have row_31 : (gl_existence1 N v1 zero mul) := by
    apply row_5
    exact row_19
    exact row_32
  -- chapter_96_line_30: GL tag expansion.
  have row_30 : (¬ (∀ (v8 : α), ((N v8) → (¬ (mul v1 zero v8))))) := by
    simpa only [gl_existence1] using row_31
  have exists_row_30 : ∃ (v8 : α), ((N v8) ∧ (mul v1 zero v8)) := existsAndOfNotForallImpNot row_30
  obtain ⟨v8, witness_row_30⟩ := exists_row_30
  -- chapter_96_line_29: GL tag disintegration.
  have row_29 : (mul v1 zero v8) := by
    exact witness_row_30.2
  -- chapter_96_line_45: GL tag implication.
  have row_45 : (zero = v8) := by
    apply row_46
    exact row_29
    exact row_19
  -- chapter_96_line_65: GL tag equality1.
  have row_65 : (mul previous v8 v12) := by
    have equality_source := row_66
    have equality_step_1 := row_45
    cases equality_step_1
    exact equality_source
  -- chapter_96_line_61: GL tag equality1.
  have row_61 : (succ v8 one) := by
    have equality_source := row_18
    have equality_step_1 := row_45
    cases equality_step_1
    exact equality_source
  -- chapter_96_line_58: GL tag implication.
  have row_58 : (add v12 previous v13) := by
    apply row_59
    exact row_62
    exact row_65
    exact row_61
  -- chapter_96_line_57: GL tag equality1.
  have row_57 : (add zero previous v11) := by
    have equality_source := row_58
    have equality_step_1 := row_69
    cases equality_step_1
    have equality_step_2 := row_72
    cases equality_step_2
    exact equality_source
  -- chapter_96_line_54: GL tag implication.
  have row_54 : (v11 = v7) := by
    apply row_55
    exact row_32
    exact row_42
    exact row_57
    exact row_35
  -- chapter_96_line_34: GL tag equality1.
  have row_34 : (add v8 previous v7) := by
    have equality_source := row_35
    have equality_step_1 := row_45
    cases equality_step_1
    exact equality_source
  -- chapter_96_line_26: GL tag implication.
  have row_26 : (add v8 v1 v2) := by
    apply row_27
    exact row_32
    exact row_18
    exact row_29
    exact row_33
  -- chapter_96_line_24: GL tag implication.
  have row_24 : (succ v7 v2) := by
    apply row_25
    exact row_26
    exact row_34
    exact row_20
  -- chapter_96_line_4: GL tag implication.
  have row_4 : (gl_existence1 N one v1 mul) := by
    apply row_5
    exact row_13
    exact row_19
  -- chapter_96_line_3: GL tag expansion.
  have row_3 : (¬ (∀ (v3 : α), ((N v3) → (¬ (mul one v1 v3))))) := by
    simpa only [gl_existence1] using row_4
  have exists_row_3 : ∃ (v3 : α), ((N v3) ∧ (mul one v1 v3)) := existsAndOfNotForallImpNot row_3
  obtain ⟨v3, witness_row_3⟩ := exists_row_3
  -- chapter_96_line_2: GL tag disintegration.
  have row_2 : (mul one v1 v3) := by
    exact witness_row_3.2
  -- chapter_96_line_50: GL tag implication.
  have row_50 : (add v11 one v3) := by
    apply row_27
    exact row_42
    exact row_20
    exact row_51
    exact row_2
  -- chapter_96_line_48: GL tag implication.
  have row_48 : (succ v11 v3) := by
    apply row_49
    exact row_50
  -- chapter_96_line_47: GL tag equality1.
  have row_47 : (succ v7 v3) := by
    have equality_source := row_48
    have equality_step_1 := row_54
    cases equality_step_1
    exact equality_source
  -- chapter_96_line_21: GL tag implication.
  have row_21 : (v3 = v2) := by
    apply row_22
    exact row_77
    exact row_47
    exact row_24
  -- chapter_96_line_1: GL tag equality1.
  have row_1 : (mul one v1 v2) := by
    have equality_source := row_2
    have equality_step_1 := row_21
    cases equality_step_1
    exact equality_source
  exact row_1

theorem peano_source_044
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((mul v1 one v2) → (mul one v1 v2))) := by
  intro v1
  intro v2
  intro premise_1
  have inductionMember : N v1 := by
    have typingRule := peano_source_044_induction_typing N zero succ add mul one anchor relationalInduction
    exact typingRule v1 v2 premise_1
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v2 : α), ((mul zero one v2) → (mul one zero v2))) := by
    intro v2
    intro base_premise_1
    have zeroRule := peano_source_044_check_zero N zero succ add mul one anchor relationalInduction
    exact zeroRule zero v2 rfl base_premise_1
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v2 : α), ((mul induction_n one v2) → (mul one induction_n v2))) → ∀ induction_m, succ induction_n induction_m → (∀ (v2 : α), ((mul induction_m one v2) → (mul one induction_m v2))) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro v2
    intro step_premise_1
    have step_induction_assumption_1 :
        (∀ (w1 : α), ((mul induction_n one w1) → (mul one induction_n w1))) := by
      intro w1
      intro step_induction_assumption_1_premise_1
      apply induction_hypothesis
      all_goals assumption
    have step_induction_assumption_2 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have stepRule := peano_source_044_check_induction_condition N zero succ add mul one anchor relationalInduction
    exact stepRule induction_n induction_m v2 step_induction_assumption_1 step_premise_1 step_induction_assumption_2
  have inductionProperty : (∀ (v2 : α), ((mul v1 one v2) → (mul one v1 v2))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v2 : α), ((mul induction_value one v2) → (mul one induction_value v2))))
      v1
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v2 premise_1

private theorem peano_source_049_induction_typing
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (assumption_10 : (mul v1 v2 zero))
    : (N v2) := by
  -- chapter_109_line_10: GL tag task formulation.
  have row_10 : (mul v1 v2 zero) := by
    exact assumption_10
  -- chapter_109_line_9: GL tag task formulation.
  have row_9 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_109_line_8: GL tag expansion.
  have row_8 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_9
  -- chapter_109_line_7: GL tag disintegration.
  have row_7 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_8.1
  -- chapter_109_line_6: GL tag expansion.
  have row_6 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_7
  -- chapter_109_line_5: GL tag disintegration.
  have row_5 : (gl_fXYZ mul N N N) := by
    exact row_6.1.1.1.2
  -- chapter_109_line_4: GL tag expansion.
  have row_4 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_5
  -- chapter_109_line_3: GL tag disintegration.
  have row_3 : (gl_implication9 mul N) := by
    exact row_4.1.1.1.2
  -- chapter_109_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication9] using row_3
  -- chapter_109_line_1: GL tag implication.
  have row_1 : (N v2) := by
    apply row_2
    exact row_10
  exact row_1

private theorem peano_source_049_check_zero
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (assumption_21 : (v2 = zero))
    (assumption_17 : (mul v1 v2 zero))
    : (mul v2 v1 zero) := by
  -- chapter_110_line_25: GL tag theorem.
  have row_25 := peano_source_056 N zero succ add mul one anchor relationalInduction
  -- chapter_110_line_21: GL tag recursion.
  have row_21 : (v2 = zero) := by
    exact assumption_21
  -- chapter_110_line_22: GL tag symmetry of equality.
  have row_22 : (zero = v2) := by
    exact Eq.symm row_21
  -- chapter_110_line_17: GL tag task formulation.
  have row_17 : (mul v1 v2 zero) := by
    exact assumption_17
  -- chapter_110_line_13: GL tag task formulation.
  have row_13 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_110_line_12: GL tag expansion.
  have row_12 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_13
  -- chapter_110_line_11: GL tag disintegration.
  have row_11 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_12.1
  -- chapter_110_line_10: GL tag expansion.
  have row_10 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_11
  -- chapter_110_line_9: GL tag disintegration.
  have row_9 : (gl_fXYZ mul N N N) := by
    exact row_10.1.1.1.2
  -- chapter_110_line_8: GL tag expansion.
  have row_8 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_9
  -- chapter_110_line_20: GL tag disintegration.
  have row_20 : (gl_implication9 mul N) := by
    exact row_8.1.1.1.2
  -- chapter_110_line_19: GL tag expansion.
  have row_19 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication9] using row_20
  -- chapter_110_line_18: GL tag implication.
  have row_18 : (N v2) := by
    apply row_19
    exact row_17
  -- chapter_110_line_16: GL tag disintegration.
  have row_16 : (gl_implication8 mul N) := by
    exact row_8.1.1.1.1
  -- chapter_110_line_15: GL tag expansion.
  have row_15 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_16
  -- chapter_110_line_14: GL tag implication.
  have row_14 : (N v1) := by
    apply row_15
    exact row_17
  -- chapter_110_line_7: GL tag disintegration.
  have row_7 : (gl_implication13 N N N mul) := by
    exact row_8.1.2
  -- chapter_110_line_6: GL tag expansion.
  have row_6 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication13] using row_7
  -- chapter_110_line_5: GL tag implication.
  have row_5 : (gl_existence1 N v2 v1 mul) := by
    apply row_6
    exact row_18
    exact row_14
  -- chapter_110_line_4: GL tag expansion.
  have row_4 : (¬ (∀ (v3 : α), ((N v3) → (¬ (mul v2 v1 v3))))) := by
    simpa only [gl_existence1] using row_5
  have exists_row_4 : ∃ (v3 : α), ((N v3) ∧ (mul v2 v1 v3)) := existsAndOfNotForallImpNot row_4
  obtain ⟨v3, witness_row_4⟩ := exists_row_4
  -- chapter_110_line_3: GL tag disintegration.
  have row_3 : (mul v2 v1 v3) := by
    exact witness_row_4.2
  -- chapter_110_line_2: GL tag equality1.
  have row_2 : (mul zero v1 v3) := by
    have equality_source := row_3
    have equality_step_1 := row_21
    cases equality_step_1
    exact equality_source
  -- chapter_110_line_24: GL tag implication.
  have row_24 : (zero = v3) := by
    apply row_25
    exact row_14
    exact row_2
  -- chapter_110_line_23: GL tag symmetry of equality.
  have row_23 : (v3 = zero) := by
    exact Eq.symm row_24
  -- chapter_110_line_1: GL tag equality1.
  have row_1 : (mul v2 v1 zero) := by
    have equality_source := row_2
    have equality_step_1 := row_22
    cases equality_step_1
    have equality_step_2 := row_23
    cases equality_step_2
    exact equality_source
  exact row_1

private theorem peano_source_049_check_induction_condition
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (assumption_22 : (succ previous v2))
    (assumption_16 : (mul v1 v2 zero))
    : (mul v2 v1 zero) := by
  -- chapter_111_line_29: GL tag theorem.
  have row_29 := peano_source_045 N zero succ add mul one anchor relationalInduction
  -- chapter_111_line_22: GL tag recursion.
  have row_22 : (succ previous v2) := by
    exact assumption_22
  -- chapter_111_line_16: GL tag task formulation.
  have row_16 : (mul v1 v2 zero) := by
    exact assumption_16
  -- chapter_111_line_12: GL tag task formulation.
  have row_12 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_111_line_11: GL tag expansion.
  have row_11 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_12
  -- chapter_111_line_10: GL tag disintegration.
  have row_10 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_11.1
  -- chapter_111_line_9: GL tag expansion.
  have row_9 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_10
  -- chapter_111_line_32: GL tag disintegration.
  have row_32 : (gl_implication21 N succ mul add) := by
    exact row_9.2
  -- chapter_111_line_31: GL tag expansion.
  have row_31 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((mul w3 w1 w4) → (∀ (w5 : α), ((mul w3 w2 w5) → (add w4 w3 w5))))))))) := by
    simpa only [gl_implication21] using row_32
  -- chapter_111_line_25: GL tag disintegration.
  have row_25 : (gl_implication19 N zero mul) := by
    exact row_9.1.1.2
  -- chapter_111_line_24: GL tag expansion.
  have row_24 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((mul w1 zero w2) → (w2 = zero))))) := by
    simpa only [gl_implication19] using row_25
  -- chapter_111_line_21: GL tag disintegration.
  have row_21 : (gl_fXY succ N N) := by
    exact row_9.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_111_line_20: GL tag expansion.
  have row_20 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_21
  -- chapter_111_line_38: GL tag disintegration.
  have row_38 : (gl_implication0 succ N) := by
    exact row_20.1.1.1
  -- chapter_111_line_37: GL tag expansion.
  have row_37 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_38
  -- chapter_111_line_36: GL tag implication.
  have row_36 : (N previous) := by
    apply row_37
    exact row_22
  -- chapter_111_line_19: GL tag disintegration.
  have row_19 : (gl_implication1 succ N) := by
    exact row_20.1.1.2
  -- chapter_111_line_18: GL tag expansion.
  have row_18 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using row_19
  -- chapter_111_line_17: GL tag implication.
  have row_17 : (N v2) := by
    apply row_18
    exact row_22
  -- chapter_111_line_8: GL tag disintegration.
  have row_8 : (gl_fXYZ mul N N N) := by
    exact row_9.1.1.1.2
  -- chapter_111_line_7: GL tag expansion.
  have row_7 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_8
  -- chapter_111_line_15: GL tag disintegration.
  have row_15 : (gl_implication8 mul N) := by
    exact row_7.1.1.1.1
  -- chapter_111_line_14: GL tag expansion.
  have row_14 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_15
  -- chapter_111_line_13: GL tag implication.
  have row_13 : (N v1) := by
    apply row_14
    exact row_16
  -- chapter_111_line_6: GL tag disintegration.
  have row_6 : (gl_implication13 N N N mul) := by
    exact row_7.1.2
  -- chapter_111_line_5: GL tag expansion.
  have row_5 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication13] using row_6
  -- chapter_111_line_35: GL tag implication.
  have row_35 : (gl_existence1 N v1 previous mul) := by
    apply row_5
    exact row_13
    exact row_36
  -- chapter_111_line_34: GL tag expansion.
  have row_34 : (¬ (∀ (v7 : α), ((N v7) → (¬ (mul v1 previous v7))))) := by
    simpa only [gl_existence1] using row_35
  have exists_row_34 : ∃ (v7 : α), ((N v7) ∧ (mul v1 previous v7)) := existsAndOfNotForallImpNot row_34
  obtain ⟨v7, witness_row_34⟩ := exists_row_34
  -- chapter_111_line_39: GL tag disintegration.
  have row_39 : (N v7) := by
    exact witness_row_34.1
  -- chapter_111_line_33: GL tag disintegration.
  have row_33 : (mul v1 previous v7) := by
    exact witness_row_34.2
  -- chapter_111_line_30: GL tag implication.
  have row_30 : (add v7 v1 zero) := by
    apply row_31
    exact row_36
    exact row_22
    exact row_33
    exact row_16
  -- chapter_111_line_28: GL tag implication.
  have row_28 : (zero = v1) := by
    apply row_29
    exact row_30
    exact row_39
  -- chapter_111_line_27: GL tag symmetry of equality.
  have row_27 : (v1 = zero) := by
    exact Eq.symm row_28
  -- chapter_111_line_4: GL tag implication.
  have row_4 : (gl_existence1 N v2 v1 mul) := by
    apply row_5
    exact row_17
    exact row_13
  -- chapter_111_line_3: GL tag expansion.
  have row_3 : (¬ (∀ (v3 : α), ((N v3) → (¬ (mul v2 v1 v3))))) := by
    simpa only [gl_existence1] using row_4
  have exists_row_3 : ∃ (v3 : α), ((N v3) ∧ (mul v2 v1 v3)) := existsAndOfNotForallImpNot row_3
  obtain ⟨v3, witness_row_3⟩ := exists_row_3
  -- chapter_111_line_2: GL tag disintegration.
  have row_2 : (mul v2 v1 v3) := by
    exact witness_row_3.2
  -- chapter_111_line_26: GL tag equality1.
  have row_26 : (mul v2 zero v3) := by
    have equality_source := row_2
    have equality_step_1 := row_27
    cases equality_step_1
    exact equality_source
  -- chapter_111_line_23: GL tag implication.
  have row_23 : (v3 = zero) := by
    apply row_24
    exact row_17
    exact row_26
  -- chapter_111_line_1: GL tag equality1.
  have row_1 : (mul v2 v1 zero) := by
    have equality_source := row_2
    have equality_step_1 := row_23
    cases equality_step_1
    exact equality_source
  exact row_1

theorem peano_source_049
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((mul v1 v2 zero) → (mul v2 v1 zero))) := by
  intro v1
  intro v2
  intro premise_1
  have inductionMember : N v2 := by
    have typingRule := peano_source_049_induction_typing N zero succ add mul one anchor relationalInduction
    exact typingRule v1 v2 premise_1
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v1 : α), ((mul v1 zero zero) → (mul zero v1 zero))) := by
    intro v1
    intro base_premise_1
    have zeroRule := peano_source_049_check_zero N zero succ add mul one anchor relationalInduction
    exact zeroRule v1 zero rfl base_premise_1
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v1 : α), ((mul v1 induction_n zero) → (mul induction_n v1 zero))) → ∀ induction_m, succ induction_n induction_m → (∀ (v1 : α), ((mul v1 induction_m zero) → (mul induction_m v1 zero))) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro v1
    intro step_premise_1
    have step_induction_assumption_1 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have stepRule := peano_source_049_check_induction_condition N zero succ add mul one anchor relationalInduction
    exact stepRule induction_n v1 induction_m step_induction_assumption_1 step_premise_1
  have inductionProperty : (∀ (v1 : α), ((mul v1 v2 zero) → (mul v2 v1 zero))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v1 : α), ((mul v1 induction_value zero) → (mul induction_value v1 zero))))
      v2
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v1 premise_1

private theorem peano_source_050_induction_typing
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (assumption_10 : (add v1 v2 one))
    : (N v2) := by
  -- chapter_112_line_10: GL tag task formulation.
  have row_10 : (add v1 v2 one) := by
    exact assumption_10
  -- chapter_112_line_9: GL tag task formulation.
  have row_9 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_112_line_8: GL tag expansion.
  have row_8 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_9
  -- chapter_112_line_7: GL tag disintegration.
  have row_7 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_8.1
  -- chapter_112_line_6: GL tag expansion.
  have row_6 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_7
  -- chapter_112_line_5: GL tag disintegration.
  have row_5 : (gl_fXYZ add N N N) := by
    exact row_6.1.1.1.1.1.1.1.1.2
  -- chapter_112_line_4: GL tag expansion.
  have row_4 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_5
  -- chapter_112_line_3: GL tag disintegration.
  have row_3 : (gl_implication9 add N) := by
    exact row_4.1.1.1.2
  -- chapter_112_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication9] using row_3
  -- chapter_112_line_1: GL tag implication.
  have row_1 : (N v2) := by
    apply row_2
    exact row_10
  exact row_1

private theorem peano_source_050_check_zero
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (assumption_22 : (v2 = zero))
    (assumption_18 : (add v1 v2 one))
    : (add v2 v1 one) := by
  -- chapter_113_line_26: GL tag theorem.
  have row_26 := peano_source_055 N zero succ add mul one anchor relationalInduction
  -- chapter_113_line_22: GL tag recursion.
  have row_22 : (v2 = zero) := by
    exact assumption_22
  -- chapter_113_line_31: GL tag symmetry of equality.
  have row_31 : (zero = v2) := by
    exact Eq.symm row_22
  -- chapter_113_line_18: GL tag task formulation.
  have row_18 : (add v1 v2 one) := by
    exact assumption_18
  -- chapter_113_line_30: GL tag equality1.
  have row_30 : (add v1 zero one) := by
    have equality_source := row_18
    have equality_step_1 := row_22
    cases equality_step_1
    exact equality_source
  -- chapter_113_line_14: GL tag task formulation.
  have row_14 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_113_line_13: GL tag expansion.
  have row_13 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_14
  -- chapter_113_line_12: GL tag disintegration.
  have row_12 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_13.1
  -- chapter_113_line_11: GL tag expansion.
  have row_11 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_12
  -- chapter_113_line_29: GL tag disintegration.
  have row_29 : (gl_implication15 N zero add) := by
    exact row_11.1.1.1.1.1.1.1.2
  -- chapter_113_line_28: GL tag expansion.
  have row_28 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((add w1 zero w2) → (w1 = w2))))) := by
    simpa only [gl_implication15] using row_29
  -- chapter_113_line_10: GL tag disintegration.
  have row_10 : (gl_fXYZ add N N N) := by
    exact row_11.1.1.1.1.1.1.1.1.2
  -- chapter_113_line_9: GL tag expansion.
  have row_9 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_10
  -- chapter_113_line_21: GL tag disintegration.
  have row_21 : (gl_implication9 add N) := by
    exact row_9.1.1.1.2
  -- chapter_113_line_20: GL tag expansion.
  have row_20 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication9] using row_21
  -- chapter_113_line_19: GL tag implication.
  have row_19 : (N v2) := by
    apply row_20
    exact row_18
  -- chapter_113_line_17: GL tag disintegration.
  have row_17 : (gl_implication8 add N) := by
    exact row_9.1.1.1.1
  -- chapter_113_line_16: GL tag expansion.
  have row_16 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_17
  -- chapter_113_line_15: GL tag implication.
  have row_15 : (N v1) := by
    apply row_16
    exact row_18
  -- chapter_113_line_27: GL tag implication.
  have row_27 : (v1 = one) := by
    apply row_28
    exact row_15
    exact row_30
  -- chapter_113_line_8: GL tag disintegration.
  have row_8 : (gl_implication13 N N N add) := by
    exact row_9.1.2
  -- chapter_113_line_7: GL tag expansion.
  have row_7 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication13] using row_8
  -- chapter_113_line_6: GL tag implication.
  have row_6 : (gl_existence1 N v2 v1 add) := by
    apply row_7
    exact row_19
    exact row_15
  -- chapter_113_line_5: GL tag expansion.
  have row_5 : (¬ (∀ (v3 : α), ((N v3) → (¬ (add v2 v1 v3))))) := by
    simpa only [gl_existence1] using row_6
  have exists_row_5 : ∃ (v3 : α), ((N v3) ∧ (add v2 v1 v3)) := existsAndOfNotForallImpNot row_5
  obtain ⟨v3, witness_row_5⟩ := exists_row_5
  -- chapter_113_line_4: GL tag disintegration.
  have row_4 : (add v2 v1 v3) := by
    exact witness_row_5.2
  -- chapter_113_line_3: GL tag equality1.
  have row_3 : (add zero v1 v3) := by
    have equality_source := row_4
    have equality_step_1 := row_22
    cases equality_step_1
    exact equality_source
  -- chapter_113_line_25: GL tag implication.
  have row_25 : (v1 = v3) := by
    apply row_26
    exact row_15
    exact row_3
  -- chapter_113_line_24: GL tag symmetry of equality.
  have row_24 : (v3 = v1) := by
    exact Eq.symm row_25
  -- chapter_113_line_23: GL tag equality2.
  have row_23 : (v3 = one) := by
    exact Eq.trans row_24 row_27
  -- chapter_113_line_2: GL tag equality1.
  have row_2 : (add zero v1 one) := by
    have equality_source := row_3
    have equality_step_1 := row_23
    cases equality_step_1
    exact equality_source
  -- chapter_113_line_1: GL tag equality1.
  have row_1 : (add v2 v1 one) := by
    have equality_source := row_2
    have equality_step_1 := row_31
    cases equality_step_1
    exact equality_source
  exact row_1

private theorem peano_source_050_check_induction_condition
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (assumption_22 : (succ previous v2))
    (assumption_16 : (add v1 v2 one))
    : (add v2 v1 one) := by
  -- chapter_114_line_56: GL tag theorem.
  have row_56 := peano_source_045 N zero succ add mul one anchor relationalInduction
  -- chapter_114_line_50: GL tag theorem.
  have row_50 := peano_source_027 N zero succ add mul one anchor relationalInduction
  -- chapter_114_line_36: GL tag theorem.
  have row_36 := peano_source_046 N zero succ add mul one anchor relationalInduction
  -- chapter_114_line_34: GL tag theorem.
  have row_34 := peano_source_024 N zero succ add mul one anchor relationalInduction
  -- chapter_114_line_26: GL tag theorem.
  have row_26 := peano_source_039 N zero succ add mul one anchor relationalInduction
  -- chapter_114_line_22: GL tag recursion.
  have row_22 : (succ previous v2) := by
    exact assumption_22
  -- chapter_114_line_16: GL tag task formulation.
  have row_16 : (add v1 v2 one) := by
    exact assumption_16
  -- chapter_114_line_12: GL tag task formulation.
  have row_12 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_114_line_51: GL tag anchor handling.
  have row_51 : (gl_AnchorPeano N zero succ add mul one) := by
    exact row_12
  -- chapter_114_line_11: GL tag expansion.
  have row_11 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_12
  -- chapter_114_line_31: GL tag disintegration.
  have row_31 : (succ zero one) := by
    exact row_11.2
  -- chapter_114_line_10: GL tag disintegration.
  have row_10 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_11.1
  -- chapter_114_line_9: GL tag expansion.
  have row_9 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_10
  -- chapter_114_line_47: GL tag disintegration.
  have row_47 : (gl_implication17 N succ add) := by
    exact row_9.1.1.1.1.1.2
  -- chapter_114_line_46: GL tag expansion.
  have row_46 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication17] using row_47
  -- chapter_114_line_30: GL tag disintegration.
  have row_30 : (gl_implication7 N succ) := by
    exact row_9.1.1.1.1.1.1.1.1.1.2
  -- chapter_114_line_29: GL tag expansion.
  have row_29 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w2 w1) → (∀ (w3 : α), ((succ w3 w1) → (w2 = w3))))))) := by
    simpa only [gl_implication7] using row_30
  -- chapter_114_line_21: GL tag disintegration.
  have row_21 : (gl_fXY succ N N) := by
    exact row_9.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_114_line_20: GL tag expansion.
  have row_20 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_21
  -- chapter_114_line_43: GL tag disintegration.
  have row_43 : (gl_implication0 succ N) := by
    exact row_20.1.1.1
  -- chapter_114_line_42: GL tag expansion.
  have row_42 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_43
  -- chapter_114_line_41: GL tag implication.
  have row_41 : (N previous) := by
    apply row_42
    exact row_22
  -- chapter_114_line_19: GL tag disintegration.
  have row_19 : (gl_implication1 succ N) := by
    exact row_20.1.1.2
  -- chapter_114_line_18: GL tag expansion.
  have row_18 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using row_19
  -- chapter_114_line_48: GL tag implication.
  have row_48 : (N one) := by
    apply row_18
    exact row_31
  -- chapter_114_line_17: GL tag implication.
  have row_17 : (N v2) := by
    apply row_18
    exact row_22
  -- chapter_114_line_8: GL tag disintegration.
  have row_8 : (gl_fXYZ add N N N) := by
    exact row_9.1.1.1.1.1.1.1.1.2
  -- chapter_114_line_7: GL tag expansion.
  have row_7 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_8
  -- chapter_114_line_15: GL tag disintegration.
  have row_15 : (gl_implication8 add N) := by
    exact row_7.1.1.1.1
  -- chapter_114_line_14: GL tag expansion.
  have row_14 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_15
  -- chapter_114_line_13: GL tag implication.
  have row_13 : (N v1) := by
    apply row_14
    exact row_16
  -- chapter_114_line_6: GL tag disintegration.
  have row_6 : (gl_implication13 N N N add) := by
    exact row_7.1.2
  -- chapter_114_line_5: GL tag expansion.
  have row_5 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication13] using row_6
  -- chapter_114_line_40: GL tag implication.
  have row_40 : (gl_existence1 N v1 previous add) := by
    apply row_5
    exact row_13
    exact row_41
  -- chapter_114_line_39: GL tag expansion.
  have row_39 : (¬ (∀ (v7 : α), ((N v7) → (¬ (add v1 previous v7))))) := by
    simpa only [gl_existence1] using row_40
  have exists_row_39 : ∃ (v7 : α), ((N v7) ∧ (add v1 previous v7)) := existsAndOfNotForallImpNot row_39
  obtain ⟨v7, witness_row_39⟩ := exists_row_39
  -- chapter_114_line_38: GL tag disintegration.
  have row_38 : (add v1 previous v7) := by
    exact witness_row_39.2
  -- chapter_114_line_45: GL tag implication.
  have row_45 : (succ v7 one) := by
    apply row_46
    exact row_41
    exact row_22
    exact row_38
    exact row_16
  -- chapter_114_line_44: GL tag implication.
  have row_44 : (v7 = zero) := by
    apply row_29
    exact row_48
    exact row_45
    exact row_31
  -- chapter_114_line_37: GL tag equality1.
  have row_37 : (add v1 previous zero) := by
    have equality_source := row_38
    have equality_step_1 := row_44
    cases equality_step_1
    exact equality_source
  -- chapter_114_line_55: GL tag implication.
  have row_55 : (zero = previous) := by
    apply row_56
    exact row_37
    exact row_13
  -- chapter_114_line_54: GL tag equality1.
  have row_54 : (succ previous one) := by
    have equality_source := row_31
    have equality_step_1 := row_55
    cases equality_step_1
    exact equality_source
  -- chapter_114_line_35: GL tag implication.
  have row_35 : (v1 = previous) := by
    apply row_36
    exact row_37
    exact row_13
  -- chapter_114_line_53: GL tag symmetry of equality.
  have row_53 : (previous = v1) := by
    exact Eq.symm row_35
  -- chapter_114_line_52: GL tag equality1.
  have row_52 : (succ v1 v2) := by
    have equality_source := row_22
    have equality_step_1 := row_53
    cases equality_step_1
    exact equality_source
  -- chapter_114_line_49: GL tag implication.
  have row_49 : (v2 = one) := by
    apply row_50
    exact row_52
    exact row_54
    exact row_35
  -- chapter_114_line_33: GL tag implication.
  have row_33 : (zero = v1) := by
    apply row_34
    exact row_35
    exact row_37
  -- chapter_114_line_32: GL tag equality1.
  have row_32 : (succ v1 one) := by
    have equality_source := row_31
    have equality_step_1 := row_33
    cases equality_step_1
    exact equality_source
  -- chapter_114_line_28: GL tag implication.
  have row_28 : (v1 = zero) := by
    apply row_29
    exact row_48
    exact row_32
    exact row_31
  -- chapter_114_line_4: GL tag implication.
  have row_4 : (gl_existence1 N v2 v1 add) := by
    apply row_5
    exact row_17
    exact row_13
  -- chapter_114_line_3: GL tag expansion.
  have row_3 : (¬ (∀ (v3 : α), ((N v3) → (¬ (add v2 v1 v3))))) := by
    simpa only [gl_existence1] using row_4
  have exists_row_3 : ∃ (v3 : α), ((N v3) ∧ (add v2 v1 v3)) := existsAndOfNotForallImpNot row_3
  obtain ⟨v3, witness_row_3⟩ := exists_row_3
  -- chapter_114_line_2: GL tag disintegration.
  have row_2 : (add v2 v1 v3) := by
    exact witness_row_3.2
  -- chapter_114_line_27: GL tag equality1.
  have row_27 : (add v2 zero v3) := by
    have equality_source := row_2
    have equality_step_1 := row_28
    cases equality_step_1
    exact equality_source
  -- chapter_114_line_25: GL tag implication.
  have row_25 : (v2 = v3) := by
    apply row_26
    exact row_27
    exact row_17
  -- chapter_114_line_24: GL tag symmetry of equality.
  have row_24 : (v3 = v2) := by
    exact Eq.symm row_25
  -- chapter_114_line_23: GL tag equality2.
  have row_23 : (v3 = one) := by
    exact Eq.trans row_24 row_49
  -- chapter_114_line_1: GL tag equality1.
  have row_1 : (add v2 v1 one) := by
    have equality_source := row_2
    have equality_step_1 := row_23
    cases equality_step_1
    exact equality_source
  exact row_1

theorem peano_source_050
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((add v1 v2 one) → (add v2 v1 one))) := by
  intro v1
  intro v2
  intro premise_1
  have inductionMember : N v2 := by
    have typingRule := peano_source_050_induction_typing N zero succ add mul one anchor relationalInduction
    exact typingRule v1 v2 premise_1
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v1 : α), ((add v1 zero one) → (add zero v1 one))) := by
    intro v1
    intro base_premise_1
    have zeroRule := peano_source_050_check_zero N zero succ add mul one anchor relationalInduction
    exact zeroRule v1 zero rfl base_premise_1
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v1 : α), ((add v1 induction_n one) → (add induction_n v1 one))) → ∀ induction_m, succ induction_n induction_m → (∀ (v1 : α), ((add v1 induction_m one) → (add induction_m v1 one))) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro v1
    intro step_premise_1
    have step_induction_assumption_1 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have stepRule := peano_source_050_check_induction_condition N zero succ add mul one anchor relationalInduction
    exact stepRule induction_n v1 induction_m step_induction_assumption_1 step_premise_1
  have inductionProperty : (∀ (v1 : α), ((add v1 v2 one) → (add v2 v1 one))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v1 : α), ((add v1 induction_value one) → (add induction_value v1 one))))
      v2
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v1 premise_1

private theorem peano_source_057_induction_typing
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (assumption_1 : (N v1))
    : (N v1) := by
  -- chapter_125_line_1: GL tag task formulation.
  have row_1 : (N v1) := by
    exact assumption_1
  exact row_1

private theorem peano_source_057_check_zero
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (assumption_22 : (mul one v1 v2))
    (assumption_16 : (N v1))
    (assumption_14 : (v1 = zero))
    : (v1 = v2) := by
  -- chapter_126_line_22: GL tag task formulation.
  have row_22 : (mul one v1 v2) := by
    exact assumption_22
  -- chapter_126_line_16: GL tag task formulation.
  have row_16 : (N v1) := by
    exact assumption_16
  -- chapter_126_line_14: GL tag recursion.
  have row_14 : (v1 = zero) := by
    exact assumption_14
  -- chapter_126_line_21: GL tag equality1.
  have row_21 : (mul one zero v2) := by
    have equality_source := row_22
    have equality_step_1 := row_14
    cases equality_step_1
    exact equality_source
  -- chapter_126_line_13: GL tag symmetry of equality.
  have row_13 : (zero = v1) := by
    exact Eq.symm row_14
  -- chapter_126_line_7: GL tag task formulation.
  have row_7 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_126_line_6: GL tag expansion.
  have row_6 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_7
  -- chapter_126_line_28: GL tag disintegration.
  have row_28 : (succ zero one) := by
    exact row_6.2
  -- chapter_126_line_5: GL tag disintegration.
  have row_5 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_6.1
  -- chapter_126_line_4: GL tag expansion.
  have row_4 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_5
  -- chapter_126_line_27: GL tag disintegration.
  have row_27 : (gl_fXY succ N N) := by
    exact row_4.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_126_line_26: GL tag expansion.
  have row_26 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_27
  -- chapter_126_line_25: GL tag disintegration.
  have row_25 : (gl_implication1 succ N) := by
    exact row_26.1.1.2
  -- chapter_126_line_24: GL tag expansion.
  have row_24 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using row_25
  -- chapter_126_line_23: GL tag implication.
  have row_23 : (N one) := by
    apply row_24
    exact row_28
  -- chapter_126_line_20: GL tag disintegration.
  have row_20 : (gl_implication19 N zero mul) := by
    exact row_4.1.1.2
  -- chapter_126_line_19: GL tag expansion.
  have row_19 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((mul w1 zero w2) → (w2 = zero))))) := by
    simpa only [gl_implication19] using row_20
  -- chapter_126_line_18: GL tag implication.
  have row_18 : (v2 = zero) := by
    apply row_19
    exact row_23
    exact row_21
  -- chapter_126_line_17: GL tag symmetry of equality.
  have row_17 : (zero = v2) := by
    exact Eq.symm row_18
  -- chapter_126_line_15: GL tag disintegration.
  have row_15 : (N zero) := by
    exact row_4.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_126_line_12: GL tag disintegration.
  have row_12 : (gl_implication16 N zero add) := by
    exact row_4.1.1.1.1.1.1.2
  -- chapter_126_line_11: GL tag expansion.
  have row_11 : (∀ (w1 : α) (w2 : α), ((w1 = w2) → ((N w1) → ((N w2) → (add w1 zero w2))))) := by
    simpa only [gl_implication16] using row_12
  -- chapter_126_line_10: GL tag implication.
  have row_10 : (add zero zero v1) := by
    apply row_11
    exact row_13
    exact row_15
    exact row_16
  -- chapter_126_line_9: GL tag equality1.
  have row_9 : (add zero zero zero) := by
    have equality_source := row_10
    have equality_step_1 := row_14
    cases equality_step_1
    exact equality_source
  -- chapter_126_line_8: GL tag equality1.
  have row_8 : (add v1 zero v2) := by
    have equality_source := row_9
    have equality_step_1 := row_13
    cases equality_step_1
    have equality_step_2 := row_17
    cases equality_step_2
    exact equality_source
  -- chapter_126_line_3: GL tag disintegration.
  have row_3 : (gl_implication15 N zero add) := by
    exact row_4.1.1.1.1.1.1.1.2
  -- chapter_126_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((add w1 zero w2) → (w1 = w2))))) := by
    simpa only [gl_implication15] using row_3
  -- chapter_126_line_1: GL tag implication.
  have row_1 : (v1 = v2) := by
    apply row_2
    exact row_16
    exact row_8
  exact row_1

private theorem peano_source_057_check_induction_condition
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (assumption_33 : (mul one v1 v2))
    (assumption_13 : ((N previous) → (∀ (w1 : α), ((mul one previous w1) → (previous = w1)))))
    (assumption_11 : (succ previous v1))
    : (v1 = v2) := by
  -- chapter_127_line_33: GL tag task formulation.
  have row_33 : (mul one v1 v2) := by
    exact assumption_33
  -- chapter_127_line_29: GL tag theorem.
  have row_29 := peano_source_043 N zero succ add mul one anchor relationalInduction
  -- chapter_127_line_13: GL tag recursion.
  have row_13 : ((N previous) → (∀ (w1 : α), ((mul one previous w1) → (previous = w1)))) := by
    exact assumption_13
  -- chapter_127_line_11: GL tag recursion.
  have row_11 : (succ previous v1) := by
    exact assumption_11
  -- chapter_127_line_9: GL tag task formulation.
  have row_9 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_127_line_8: GL tag expansion.
  have row_8 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_9
  -- chapter_127_line_24: GL tag disintegration.
  have row_24 : (succ zero one) := by
    exact row_8.2
  -- chapter_127_line_7: GL tag disintegration.
  have row_7 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_8.1
  -- chapter_127_line_6: GL tag expansion.
  have row_6 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_7
  -- chapter_127_line_32: GL tag disintegration.
  have row_32 : (gl_implication21 N succ mul add) := by
    exact row_6.2
  -- chapter_127_line_31: GL tag expansion.
  have row_31 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((mul w3 w1 w4) → (∀ (w5 : α), ((mul w3 w2 w5) → (add w4 w3 w5))))))))) := by
    simpa only [gl_implication21] using row_32
  -- chapter_127_line_20: GL tag disintegration.
  have row_20 : (gl_fXYZ mul N N N) := by
    exact row_6.1.1.1.2
  -- chapter_127_line_19: GL tag expansion.
  have row_19 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_20
  -- chapter_127_line_18: GL tag disintegration.
  have row_18 : (gl_implication13 N N N mul) := by
    exact row_19.1.2
  -- chapter_127_line_17: GL tag expansion.
  have row_17 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication13] using row_18
  -- chapter_127_line_5: GL tag disintegration.
  have row_5 : (gl_fXY succ N N) := by
    exact row_6.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_127_line_4: GL tag expansion.
  have row_4 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_5
  -- chapter_127_line_27: GL tag disintegration.
  have row_27 : (gl_implication0 succ N) := by
    exact row_4.1.1.1
  -- chapter_127_line_26: GL tag expansion.
  have row_26 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_27
  -- chapter_127_line_25: GL tag implication.
  have row_25 : (N previous) := by
    apply row_26
    exact row_11
  -- chapter_127_line_23: GL tag disintegration.
  have row_23 : (gl_implication1 succ N) := by
    exact row_4.1.1.2
  -- chapter_127_line_22: GL tag expansion.
  have row_22 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using row_23
  -- chapter_127_line_21: GL tag implication.
  have row_21 : (N one) := by
    apply row_22
    exact row_24
  -- chapter_127_line_16: GL tag implication.
  have row_16 : (gl_existence1 N one previous mul) := by
    apply row_17
    exact row_21
    exact row_25
  -- chapter_127_line_15: GL tag expansion.
  have row_15 : (¬ (∀ (v6 : α), ((N v6) → (¬ (mul one previous v6))))) := by
    simpa only [gl_existence1] using row_16
  have exists_row_15 : ∃ (v6 : α), ((N v6) ∧ (mul one previous v6)) := existsAndOfNotForallImpNot row_15
  obtain ⟨v6, witness_row_15⟩ := exists_row_15
  -- chapter_127_line_34: GL tag disintegration.
  have row_34 : (N v6) := by
    exact witness_row_15.1
  -- chapter_127_line_14: GL tag disintegration.
  have row_14 : (mul one previous v6) := by
    exact witness_row_15.2
  -- chapter_127_line_30: GL tag implication.
  have row_30 : (add v6 one v2) := by
    apply row_31
    exact row_25
    exact row_11
    exact row_14
    exact row_33
  -- chapter_127_line_28: GL tag implication.
  have row_28 : (succ v6 v2) := by
    apply row_29
    exact row_30
  -- chapter_127_line_12: GL tag implication.
  have row_12 : (previous = v6) := by
    apply row_13
    exact row_25
    exact row_14
  -- chapter_127_line_10: GL tag equality1.
  have row_10 : (succ v6 v1) := by
    have equality_source := row_11
    have equality_step_1 := row_12
    cases equality_step_1
    exact equality_source
  -- chapter_127_line_3: GL tag disintegration.
  have row_3 : (gl_implication5 N succ) := by
    exact row_4.2
  -- chapter_127_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α), ((succ w1 w3) → (w2 = w3))))))) := by
    simpa only [gl_implication5] using row_3
  -- chapter_127_line_1: GL tag implication.
  have row_1 : (v1 = v2) := by
    apply row_2
    exact row_34
    exact row_10
    exact row_28
  exact row_1

theorem peano_source_057
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α), ((N v1) → (∀ (v2 : α), ((mul one v1 v2) → (v1 = v2))))) := by
  intro v1
  intro premise_1
  intro v2
  intro premise_2
  have inductionMember : N v1 := by
    have typingRule := peano_source_057_induction_typing N zero succ add mul one anchor relationalInduction
    exact typingRule v1 premise_1
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v2 : α), ((mul one zero v2) → (zero = v2))) := by
    intro v2
    intro base_premise_1
    have zeroRule := peano_source_057_check_zero N zero succ add mul one anchor relationalInduction
    exact zeroRule zero v2 base_premise_1 inductionZeroMember rfl
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v2 : α), ((mul one induction_n v2) → (induction_n = v2))) → ∀ induction_m, succ induction_n induction_m → (∀ (v2 : α), ((mul one induction_m v2) → (induction_m = v2))) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro v2
    intro step_premise_1
    have step_induction_assumption_1 :
        ((N induction_n) → (∀ (w1 : α), ((mul one induction_n w1) → (induction_n = w1)))) := by
      intro step_induction_assumption_1_premise_1
      intro w1
      intro step_induction_assumption_1_premise_2
      apply induction_hypothesis
      all_goals assumption
    have step_induction_assumption_2 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have stepRule := peano_source_057_check_induction_condition N zero succ add mul one anchor relationalInduction
    exact stepRule induction_n induction_m v2 step_premise_1 step_induction_assumption_1 step_induction_assumption_2
  have inductionProperty : (∀ (v2 : α), ((mul one v1 v2) → (v1 = v2))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v2 : α), ((mul one induction_value v2) → (induction_value = v2))))
      v1
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v2 premise_2

private theorem peano_source_058_induction_typing
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (assumption_10 : (mul one v2 v1))
    : (N v2) := by
  -- chapter_128_line_10: GL tag task formulation.
  have row_10 : (mul one v2 v1) := by
    exact assumption_10
  -- chapter_128_line_9: GL tag task formulation.
  have row_9 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_128_line_8: GL tag expansion.
  have row_8 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_9
  -- chapter_128_line_7: GL tag disintegration.
  have row_7 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_8.1
  -- chapter_128_line_6: GL tag expansion.
  have row_6 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_7
  -- chapter_128_line_5: GL tag disintegration.
  have row_5 : (gl_fXYZ mul N N N) := by
    exact row_6.1.1.1.2
  -- chapter_128_line_4: GL tag expansion.
  have row_4 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_5
  -- chapter_128_line_3: GL tag disintegration.
  have row_3 : (gl_implication9 mul N) := by
    exact row_4.1.1.1.2
  -- chapter_128_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication9] using row_3
  -- chapter_128_line_1: GL tag implication.
  have row_1 : (N v2) := by
    apply row_2
    exact row_10
  exact row_1

private theorem peano_source_058_check_zero
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (assumption_33 : (N v1))
    (assumption_21 : (mul one v2 v1))
    (assumption_14 : (v2 = zero))
    : (v1 = v2) := by
  -- chapter_129_line_33: GL tag task formulation.
  have row_33 : (N v1) := by
    exact assumption_33
  -- chapter_129_line_21: GL tag task formulation.
  have row_21 : (mul one v2 v1) := by
    exact assumption_21
  -- chapter_129_line_14: GL tag recursion.
  have row_14 : (v2 = zero) := by
    exact assumption_14
  -- chapter_129_line_26: GL tag equality1.
  have row_26 : (mul one zero v1) := by
    have equality_source := row_21
    have equality_step_1 := row_14
    cases equality_step_1
    exact equality_source
  -- chapter_129_line_13: GL tag symmetry of equality.
  have row_13 : (zero = v2) := by
    exact Eq.symm row_14
  -- chapter_129_line_7: GL tag task formulation.
  have row_7 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_129_line_6: GL tag expansion.
  have row_6 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_7
  -- chapter_129_line_32: GL tag disintegration.
  have row_32 : (succ zero one) := by
    exact row_6.2
  -- chapter_129_line_5: GL tag disintegration.
  have row_5 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_6.1
  -- chapter_129_line_4: GL tag expansion.
  have row_4 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_5
  -- chapter_129_line_31: GL tag disintegration.
  have row_31 : (gl_fXY succ N N) := by
    exact row_4.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_129_line_30: GL tag expansion.
  have row_30 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_31
  -- chapter_129_line_29: GL tag disintegration.
  have row_29 : (gl_implication1 succ N) := by
    exact row_30.1.1.2
  -- chapter_129_line_28: GL tag expansion.
  have row_28 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using row_29
  -- chapter_129_line_27: GL tag implication.
  have row_27 : (N one) := by
    apply row_28
    exact row_32
  -- chapter_129_line_25: GL tag disintegration.
  have row_25 : (gl_implication19 N zero mul) := by
    exact row_4.1.1.2
  -- chapter_129_line_24: GL tag expansion.
  have row_24 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((mul w1 zero w2) → (w2 = zero))))) := by
    simpa only [gl_implication19] using row_25
  -- chapter_129_line_23: GL tag implication.
  have row_23 : (v1 = zero) := by
    apply row_24
    exact row_27
    exact row_26
  -- chapter_129_line_22: GL tag symmetry of equality.
  have row_22 : (zero = v1) := by
    exact Eq.symm row_23
  -- chapter_129_line_20: GL tag disintegration.
  have row_20 : (gl_fXYZ mul N N N) := by
    exact row_4.1.1.1.2
  -- chapter_129_line_19: GL tag expansion.
  have row_19 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_20
  -- chapter_129_line_18: GL tag disintegration.
  have row_18 : (gl_implication9 mul N) := by
    exact row_19.1.1.1.2
  -- chapter_129_line_17: GL tag expansion.
  have row_17 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication9] using row_18
  -- chapter_129_line_16: GL tag implication.
  have row_16 : (N v2) := by
    apply row_17
    exact row_21
  -- chapter_129_line_15: GL tag disintegration.
  have row_15 : (N zero) := by
    exact row_4.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_129_line_12: GL tag disintegration.
  have row_12 : (gl_implication16 N zero add) := by
    exact row_4.1.1.1.1.1.1.2
  -- chapter_129_line_11: GL tag expansion.
  have row_11 : (∀ (w1 : α) (w2 : α), ((w1 = w2) → ((N w1) → ((N w2) → (add w1 zero w2))))) := by
    simpa only [gl_implication16] using row_12
  -- chapter_129_line_10: GL tag implication.
  have row_10 : (add zero zero v2) := by
    apply row_11
    exact row_13
    exact row_15
    exact row_16
  -- chapter_129_line_9: GL tag equality1.
  have row_9 : (add zero zero zero) := by
    have equality_source := row_10
    have equality_step_1 := row_14
    cases equality_step_1
    exact equality_source
  -- chapter_129_line_8: GL tag equality1.
  have row_8 : (add v1 zero v2) := by
    have equality_source := row_9
    have equality_step_1 := row_22
    cases equality_step_1
    have equality_step_2 := row_13
    cases equality_step_2
    exact equality_source
  -- chapter_129_line_3: GL tag disintegration.
  have row_3 : (gl_implication15 N zero add) := by
    exact row_4.1.1.1.1.1.1.1.2
  -- chapter_129_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((add w1 zero w2) → (w1 = w2))))) := by
    simpa only [gl_implication15] using row_3
  -- chapter_129_line_1: GL tag implication.
  have row_1 : (v1 = v2) := by
    apply row_2
    exact row_33
    exact row_8
  exact row_1

private theorem peano_source_058_check_induction_condition
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (assumption_34 : (∀ (w1 : α), ((N w1) → ((mul one previous w1) → (w1 = previous)))))
    (assumption_16 : (mul one v2 v1))
    (assumption_15 : (succ previous v2))
    : (v1 = v2) := by
  -- chapter_130_line_34: GL tag recursion.
  have row_34 : (∀ (w1 : α), ((N w1) → ((mul one previous w1) → (w1 = previous)))) := by
    exact assumption_34
  -- chapter_130_line_16: GL tag task formulation.
  have row_16 : (mul one v2 v1) := by
    exact assumption_16
  -- chapter_130_line_15: GL tag recursion.
  have row_15 : (succ previous v2) := by
    exact assumption_15
  -- chapter_130_line_11: GL tag theorem.
  have row_11 := peano_source_043 N zero succ add mul one anchor relationalInduction
  -- chapter_130_line_9: GL tag task formulation.
  have row_9 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_130_line_8: GL tag expansion.
  have row_8 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_9
  -- chapter_130_line_27: GL tag disintegration.
  have row_27 : (succ zero one) := by
    exact row_8.2
  -- chapter_130_line_7: GL tag disintegration.
  have row_7 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_8.1
  -- chapter_130_line_6: GL tag expansion.
  have row_6 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_7
  -- chapter_130_line_23: GL tag disintegration.
  have row_23 : (gl_fXYZ mul N N N) := by
    exact row_6.1.1.1.2
  -- chapter_130_line_22: GL tag expansion.
  have row_22 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_23
  -- chapter_130_line_21: GL tag disintegration.
  have row_21 : (gl_implication13 N N N mul) := by
    exact row_22.1.2
  -- chapter_130_line_20: GL tag expansion.
  have row_20 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication13] using row_21
  -- chapter_130_line_14: GL tag disintegration.
  have row_14 : (gl_implication21 N succ mul add) := by
    exact row_6.2
  -- chapter_130_line_13: GL tag expansion.
  have row_13 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((mul w3 w1 w4) → (∀ (w5 : α), ((mul w3 w2 w5) → (add w4 w3 w5))))))))) := by
    simpa only [gl_implication21] using row_14
  -- chapter_130_line_5: GL tag disintegration.
  have row_5 : (gl_fXY succ N N) := by
    exact row_6.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_130_line_4: GL tag expansion.
  have row_4 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_5
  -- chapter_130_line_30: GL tag disintegration.
  have row_30 : (gl_implication0 succ N) := by
    exact row_4.1.1.1
  -- chapter_130_line_29: GL tag expansion.
  have row_29 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_30
  -- chapter_130_line_28: GL tag implication.
  have row_28 : (N previous) := by
    apply row_29
    exact row_15
  -- chapter_130_line_26: GL tag disintegration.
  have row_26 : (gl_implication1 succ N) := by
    exact row_4.1.1.2
  -- chapter_130_line_25: GL tag expansion.
  have row_25 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using row_26
  -- chapter_130_line_24: GL tag implication.
  have row_24 : (N one) := by
    apply row_25
    exact row_27
  -- chapter_130_line_19: GL tag implication.
  have row_19 : (gl_existence1 N one previous mul) := by
    apply row_20
    exact row_24
    exact row_28
  -- chapter_130_line_18: GL tag expansion.
  have row_18 : (¬ (∀ (v6 : α), ((N v6) → (¬ (mul one previous v6))))) := by
    simpa only [gl_existence1] using row_19
  have exists_row_18 : ∃ (v6 : α), ((N v6) ∧ (mul one previous v6)) := existsAndOfNotForallImpNot row_18
  obtain ⟨v6, witness_row_18⟩ := exists_row_18
  -- chapter_130_line_35: GL tag disintegration.
  have row_35 : (N v6) := by
    exact witness_row_18.1
  -- chapter_130_line_17: GL tag disintegration.
  have row_17 : (mul one previous v6) := by
    exact witness_row_18.2
  -- chapter_130_line_33: GL tag implication.
  have row_33 : (v6 = previous) := by
    apply row_34
    exact row_35
    exact row_17
  -- chapter_130_line_32: GL tag symmetry of equality.
  have row_32 : (previous = v6) := by
    exact Eq.symm row_33
  -- chapter_130_line_31: GL tag equality1.
  have row_31 : (succ v6 v2) := by
    have equality_source := row_15
    have equality_step_1 := row_32
    cases equality_step_1
    exact equality_source
  -- chapter_130_line_12: GL tag implication.
  have row_12 : (add v6 one v1) := by
    apply row_13
    exact row_28
    exact row_15
    exact row_17
    exact row_16
  -- chapter_130_line_10: GL tag implication.
  have row_10 : (succ v6 v1) := by
    apply row_11
    exact row_12
  -- chapter_130_line_3: GL tag disintegration.
  have row_3 : (gl_implication5 N succ) := by
    exact row_4.2
  -- chapter_130_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α), ((succ w1 w3) → (w2 = w3))))))) := by
    simpa only [gl_implication5] using row_3
  -- chapter_130_line_1: GL tag implication.
  have row_1 : (v1 = v2) := by
    apply row_2
    exact row_35
    exact row_10
    exact row_31
  exact row_1

theorem peano_source_058
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α), ((N v1) → (∀ (v2 : α), ((mul one v2 v1) → (v1 = v2))))) := by
  intro v1
  intro premise_1
  intro v2
  intro premise_2
  have inductionMember : N v2 := by
    have typingRule := peano_source_058_induction_typing N zero succ add mul one anchor relationalInduction
    exact typingRule v1 v2 premise_2
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v1 : α), ((N v1) → ((mul one zero v1) → (v1 = zero)))) := by
    intro v1
    intro base_premise_1
    intro base_premise_2
    have zeroRule := peano_source_058_check_zero N zero succ add mul one anchor relationalInduction
    exact zeroRule v1 zero base_premise_1 base_premise_2 rfl
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v1 : α), ((N v1) → ((mul one induction_n v1) → (v1 = induction_n)))) → ∀ induction_m, succ induction_n induction_m → (∀ (v1 : α), ((N v1) → ((mul one induction_m v1) → (v1 = induction_m)))) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro v1
    intro step_premise_1
    intro step_premise_2
    have step_induction_assumption_1 :
        (∀ (w1 : α), ((N w1) → ((mul one induction_n w1) → (w1 = induction_n)))) := by
      intro w1
      intro step_induction_assumption_1_premise_1
      intro step_induction_assumption_1_premise_2
      apply induction_hypothesis
      all_goals assumption
    have step_induction_assumption_2 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have stepRule := peano_source_058_check_induction_condition N zero succ add mul one anchor relationalInduction
    exact stepRule induction_n v1 induction_m step_induction_assumption_1 step_premise_2 step_induction_assumption_2
  have inductionProperty : (∀ (v1 : α), ((N v1) → ((mul one v2 v1) → (v1 = v2)))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v1 : α), ((N v1) → ((mul one induction_value v1) → (v1 = induction_value)))))
      v2
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v1 premise_1 premise_2

private theorem peano_source_007_induction_typing
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (v4 : α)
    (v6 : α)
    (assumption_10 : (add v6 v4 v1))
    : (N v4) := by
  -- chapter_17_line_10: GL tag task formulation.
  have row_10 : (add v6 v4 v1) := by
    exact assumption_10
  -- chapter_17_line_9: GL tag task formulation.
  have row_9 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_17_line_8: GL tag expansion.
  have row_8 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_9
  -- chapter_17_line_7: GL tag disintegration.
  have row_7 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_8.1
  -- chapter_17_line_6: GL tag expansion.
  have row_6 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_7
  -- chapter_17_line_5: GL tag disintegration.
  have row_5 : (gl_fXYZ add N N N) := by
    exact row_6.1.1.1.1.1.1.1.1.2
  -- chapter_17_line_4: GL tag expansion.
  have row_4 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_5
  -- chapter_17_line_3: GL tag disintegration.
  have row_3 : (gl_implication9 add N) := by
    exact row_4.1.1.1.2
  -- chapter_17_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication9] using row_3
  -- chapter_17_line_1: GL tag implication.
  have row_1 : (N v4) := by
    apply row_2
    exact row_10
  exact row_1

private theorem peano_source_007_check_zero
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (v4 : α)
    (v5 : α)
    (v6 : α)
    (assumption_30 : (add v6 v4 v1))
    (assumption_25 : (v4 = zero))
    (assumption_8 : (add v2 v4 v5))
    (assumption_7 : (add v1 v2 v3))
    : (add v5 v6 v3) := by
  -- chapter_18_line_30: GL tag task formulation.
  have row_30 : (add v6 v4 v1) := by
    exact assumption_30
  -- chapter_18_line_28: GL tag theorem.
  have row_28 := peano_source_039 N zero succ add mul one anchor relationalInduction
  -- chapter_18_line_25: GL tag recursion.
  have row_25 : (v4 = zero) := by
    exact assumption_25
  -- chapter_18_line_29: GL tag equality1.
  have row_29 : (add v6 zero v1) := by
    have equality_source := row_30
    have equality_step_1 := row_25
    cases equality_step_1
    exact equality_source
  -- chapter_18_line_24: GL tag symmetry of equality.
  have row_24 : (zero = v4) := by
    exact Eq.symm row_25
  -- chapter_18_line_8: GL tag task formulation.
  have row_8 : (add v2 v4 v5) := by
    exact assumption_8
  -- chapter_18_line_7: GL tag task formulation.
  have row_7 : (add v1 v2 v3) := by
    exact assumption_7
  -- chapter_18_line_6: GL tag theorem.
  have row_6 := peano_source_004 N zero succ add mul one anchor relationalInduction
  -- chapter_18_line_3: GL tag task formulation.
  have row_3 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_18_line_19: GL tag expansion.
  have row_19 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_3
  -- chapter_18_line_18: GL tag disintegration.
  have row_18 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_19.1
  -- chapter_18_line_17: GL tag expansion.
  have row_17 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_18
  -- chapter_18_line_20: GL tag disintegration.
  have row_20 : (N zero) := by
    exact row_17.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_18_line_16: GL tag disintegration.
  have row_16 : (gl_fXYZ add N N N) := by
    exact row_17.1.1.1.1.1.1.1.1.2
  -- chapter_18_line_15: GL tag expansion.
  have row_15 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_16
  -- chapter_18_line_33: GL tag disintegration.
  have row_33 : (gl_implication8 add N) := by
    exact row_15.1.1.1.1
  -- chapter_18_line_32: GL tag expansion.
  have row_32 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_33
  -- chapter_18_line_31: GL tag implication.
  have row_31 : (N v6) := by
    apply row_32
    exact row_30
  -- chapter_18_line_27: GL tag implication.
  have row_27 : (v6 = v1) := by
    apply row_28
    exact row_29
    exact row_31
  -- chapter_18_line_26: GL tag symmetry of equality.
  have row_26 : (v1 = v6) := by
    exact Eq.symm row_27
  -- chapter_18_line_23: GL tag disintegration.
  have row_23 : (gl_implication10 add N) := by
    exact row_15.1.1.2
  -- chapter_18_line_22: GL tag expansion.
  have row_22 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication10] using row_23
  -- chapter_18_line_21: GL tag implication.
  have row_21 : (N v3) := by
    apply row_22
    exact row_7
  -- chapter_18_line_14: GL tag disintegration.
  have row_14 : (gl_implication13 N N N add) := by
    exact row_15.1.2
  -- chapter_18_line_13: GL tag expansion.
  have row_13 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication13] using row_14
  -- chapter_18_line_12: GL tag implication.
  have row_12 : (gl_existence1 N v3 zero add) := by
    apply row_13
    exact row_21
    exact row_20
  -- chapter_18_line_11: GL tag expansion.
  have row_11 : (¬ (∀ (v7 : α), ((N v7) → (¬ (add v3 zero v7))))) := by
    simpa only [gl_existence1] using row_12
  have exists_row_11 : ∃ (v7 : α), ((N v7) ∧ (add v3 zero v7)) := existsAndOfNotForallImpNot row_11
  obtain ⟨v7, witness_row_11⟩ := exists_row_11
  -- chapter_18_line_10: GL tag disintegration.
  have row_10 : (add v3 zero v7) := by
    exact witness_row_11.2
  -- chapter_18_line_35: GL tag implication.
  have row_35 : (v3 = v7) := by
    apply row_28
    exact row_10
    exact row_21
  -- chapter_18_line_34: GL tag symmetry of equality.
  have row_34 : (v7 = v3) := by
    exact Eq.symm row_35
  -- chapter_18_line_9: GL tag equality1.
  have row_9 : (add v3 v4 v7) := by
    have equality_source := row_10
    have equality_step_1 := row_24
    cases equality_step_1
    exact equality_source
  -- chapter_18_line_5: GL tag implication.
  have row_5 : (add v1 v5 v7) := by
    apply row_6
    exact row_9
    exact row_7
    exact row_8
  -- chapter_18_line_4: GL tag equality1.
  have row_4 : (add v6 v5 v3) := by
    have equality_source := row_5
    have equality_step_1 := row_26
    cases equality_step_1
    have equality_step_2 := row_34
    cases equality_step_2
    exact equality_source
  -- chapter_18_line_2: GL tag theorem.
  have row_2 := peano_source_016 N zero succ add mul one anchor relationalInduction
  -- chapter_18_line_1: GL tag implication.
  have row_1 : (add v5 v6 v3) := by
    apply row_2
    exact row_4
  exact row_1

private theorem peano_source_007_check_induction_condition
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (v4 : α)
    (v5 : α)
    (v6 : α)
    (assumption_50 : (∀ (w1 : α) (w2 : α), ((add w1 v2 w2) → (∀ (w3 : α), ((add v2 previous w3) → ((add v6 previous w1) → (add w3 v6 w2)))))))
    (assumption_38 : (add v1 v2 v3))
    (assumption_31 : (succ previous v4))
    (assumption_20 : (add v6 v4 v1))
    (assumption_16 : (add v2 v4 v5))
    : (add v5 v6 v3) := by
  -- chapter_19_line_50: GL tag recursion.
  have row_50 : (∀ (w1 : α) (w2 : α), ((add w1 v2 w2) → (∀ (w3 : α), ((add v2 previous w3) → ((add v6 previous w1) → (add w3 v6 w2)))))) := by
    exact assumption_50
  -- chapter_19_line_38: GL tag task formulation.
  have row_38 : (add v1 v2 v3) := by
    exact assumption_38
  -- chapter_19_line_31: GL tag recursion.
  have row_31 : (succ previous v4) := by
    exact assumption_31
  -- chapter_19_line_27: GL tag theorem.
  have row_27 := peano_source_005 N zero succ add mul one anchor relationalInduction
  -- chapter_19_line_20: GL tag task formulation.
  have row_20 : (add v6 v4 v1) := by
    exact assumption_20
  -- chapter_19_line_16: GL tag task formulation.
  have row_16 : (add v2 v4 v5) := by
    exact assumption_16
  -- chapter_19_line_12: GL tag task formulation.
  have row_12 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_19_line_11: GL tag expansion.
  have row_11 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_12
  -- chapter_19_line_10: GL tag disintegration.
  have row_10 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_11.1
  -- chapter_19_line_9: GL tag expansion.
  have row_9 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_10
  -- chapter_19_line_30: GL tag disintegration.
  have row_30 : (gl_implication17 N succ add) := by
    exact row_9.1.1.1.1.1.2
  -- chapter_19_line_29: GL tag expansion.
  have row_29 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication17] using row_30
  -- chapter_19_line_25: GL tag disintegration.
  have row_25 : (gl_fXY succ N N) := by
    exact row_9.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_19_line_24: GL tag expansion.
  have row_24 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_25
  -- chapter_19_line_37: GL tag disintegration.
  have row_37 : (gl_implication0 succ N) := by
    exact row_24.1.1.1
  -- chapter_19_line_36: GL tag expansion.
  have row_36 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_37
  -- chapter_19_line_35: GL tag implication.
  have row_35 : (N previous) := by
    apply row_36
    exact row_31
  -- chapter_19_line_23: GL tag disintegration.
  have row_23 : (gl_implication5 N succ) := by
    exact row_24.2
  -- chapter_19_line_22: GL tag expansion.
  have row_22 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α), ((succ w1 w3) → (w2 = w3))))))) := by
    simpa only [gl_implication5] using row_23
  -- chapter_19_line_8: GL tag disintegration.
  have row_8 : (gl_fXYZ add N N N) := by
    exact row_9.1.1.1.1.1.1.1.1.2
  -- chapter_19_line_7: GL tag expansion.
  have row_7 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_8
  -- chapter_19_line_19: GL tag disintegration.
  have row_19 : (gl_implication8 add N) := by
    exact row_7.1.1.1.1
  -- chapter_19_line_18: GL tag expansion.
  have row_18 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_19
  -- chapter_19_line_42: GL tag implication.
  have row_42 : (N v2) := by
    apply row_18
    exact row_16
  -- chapter_19_line_17: GL tag implication.
  have row_17 : (N v6) := by
    apply row_18
    exact row_20
  -- chapter_19_line_15: GL tag disintegration.
  have row_15 : (gl_implication10 add N) := by
    exact row_7.1.1.2
  -- chapter_19_line_14: GL tag expansion.
  have row_14 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication10] using row_15
  -- chapter_19_line_13: GL tag implication.
  have row_13 : (N v5) := by
    apply row_14
    exact row_16
  -- chapter_19_line_6: GL tag disintegration.
  have row_6 : (gl_implication13 N N N add) := by
    exact row_7.1.2
  -- chapter_19_line_5: GL tag expansion.
  have row_5 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication13] using row_6
  -- chapter_19_line_48: GL tag implication.
  have row_48 : (gl_existence1 N v2 previous add) := by
    apply row_5
    exact row_42
    exact row_35
  -- chapter_19_line_47: GL tag expansion.
  have row_47 : (¬ (∀ (v15 : α), ((N v15) → (¬ (add v2 previous v15))))) := by
    simpa only [gl_existence1] using row_48
  have exists_row_47 : ∃ (v15 : α), ((N v15) ∧ (add v2 previous v15)) := existsAndOfNotForallImpNot row_47
  obtain ⟨v15, witness_row_47⟩ := exists_row_47
  -- chapter_19_line_46: GL tag disintegration.
  have row_46 : (add v2 previous v15) := by
    exact witness_row_47.2
  -- chapter_19_line_45: GL tag implication.
  have row_45 : (succ v15 v5) := by
    apply row_29
    exact row_35
    exact row_31
    exact row_46
    exact row_16
  -- chapter_19_line_34: GL tag implication.
  have row_34 : (gl_existence1 N v6 previous add) := by
    apply row_5
    exact row_17
    exact row_35
  -- chapter_19_line_33: GL tag expansion.
  have row_33 : (¬ (∀ (v12 : α), ((N v12) → (¬ (add v6 previous v12))))) := by
    simpa only [gl_existence1] using row_34
  have exists_row_33 : ∃ (v12 : α), ((N v12) ∧ (add v6 previous v12)) := existsAndOfNotForallImpNot row_33
  obtain ⟨v12, witness_row_33⟩ := exists_row_33
  -- chapter_19_line_43: GL tag disintegration.
  have row_43 : (N v12) := by
    exact witness_row_33.1
  -- chapter_19_line_41: GL tag implication.
  have row_41 : (gl_existence1 N v12 v2 add) := by
    apply row_5
    exact row_43
    exact row_42
  -- chapter_19_line_40: GL tag expansion.
  have row_40 : (¬ (∀ (v11 : α), ((N v11) → (¬ (add v12 v2 v11))))) := by
    simpa only [gl_existence1] using row_41
  have exists_row_40 : ∃ (v11 : α), ((N v11) ∧ (add v12 v2 v11)) := existsAndOfNotForallImpNot row_40
  obtain ⟨v11, witness_row_40⟩ := exists_row_40
  -- chapter_19_line_51: GL tag disintegration.
  have row_51 : (N v11) := by
    exact witness_row_40.1
  -- chapter_19_line_39: GL tag disintegration.
  have row_39 : (add v12 v2 v11) := by
    exact witness_row_40.2
  -- chapter_19_line_32: GL tag disintegration.
  have row_32 : (add v6 previous v12) := by
    exact witness_row_33.2
  -- chapter_19_line_49: GL tag implication.
  have row_49 : (add v15 v6 v11) := by
    apply row_50
    exact row_39
    exact row_46
    exact row_32
  -- chapter_19_line_28: GL tag implication.
  have row_28 : (succ v12 v1) := by
    apply row_29
    exact row_35
    exact row_31
    exact row_32
    exact row_20
  -- chapter_19_line_26: GL tag implication.
  have row_26 : (succ v11 v3) := by
    apply row_27
    exact row_38
    exact row_39
    exact row_28
  -- chapter_19_line_4: GL tag implication.
  have row_4 : (gl_existence1 N v5 v6 add) := by
    apply row_5
    exact row_13
    exact row_17
  -- chapter_19_line_3: GL tag expansion.
  have row_3 : (¬ (∀ (v7 : α), ((N v7) → (¬ (add v5 v6 v7))))) := by
    simpa only [gl_existence1] using row_4
  have exists_row_3 : ∃ (v7 : α), ((N v7) ∧ (add v5 v6 v7)) := existsAndOfNotForallImpNot row_3
  obtain ⟨v7, witness_row_3⟩ := exists_row_3
  -- chapter_19_line_2: GL tag disintegration.
  have row_2 : (add v5 v6 v7) := by
    exact witness_row_3.2
  -- chapter_19_line_44: GL tag implication.
  have row_44 : (succ v11 v7) := by
    apply row_27
    exact row_2
    exact row_49
    exact row_45
  -- chapter_19_line_21: GL tag implication.
  have row_21 : (v7 = v3) := by
    apply row_22
    exact row_51
    exact row_44
    exact row_26
  -- chapter_19_line_1: GL tag equality1.
  have row_1 : (add v5 v6 v3) := by
    have equality_source := row_2
    have equality_step_1 := row_21
    cases equality_step_1
    exact equality_source
  exact row_1

theorem peano_source_007
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α) (v5 : α), ((add v2 v4 v5) → (∀ (v6 : α), ((add v6 v4 v1) → (add v5 v6 v3))))))) := by
  intro v1
  intro v2
  intro v3
  intro premise_1
  intro v4
  intro v5
  intro premise_2
  intro v6
  intro premise_3
  have inductionMember : N v4 := by
    have typingRule := peano_source_007_induction_typing N zero succ add mul one anchor relationalInduction
    exact typingRule v1 v4 v6 premise_3
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v5 : α), ((add v2 zero v5) → (∀ (v6 : α), ((add v6 zero v1) → (add v5 v6 v3))))))) := by
    intro v1
    intro v2
    intro v3
    intro base_premise_1
    intro v5
    intro base_premise_2
    intro v6
    intro base_premise_3
    have zeroRule := peano_source_007_check_zero N zero succ add mul one anchor relationalInduction
    exact zeroRule v1 v2 v3 zero v5 v6 base_premise_3 rfl base_premise_2 base_premise_1
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v5 : α), ((add v2 induction_n v5) → (∀ (v6 : α), ((add v6 induction_n v1) → (add v5 v6 v3))))))) → ∀ induction_m, succ induction_n induction_m → (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v5 : α), ((add v2 induction_m v5) → (∀ (v6 : α), ((add v6 induction_m v1) → (add v5 v6 v3))))))) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro v1
    intro v2
    intro v3
    intro step_premise_1
    intro v5
    intro step_premise_2
    intro v6
    intro step_premise_3
    have step_induction_assumption_1 :
        (∀ (w1 : α) (w2 : α), ((add w1 v2 w2) → (∀ (w3 : α), ((add v2 induction_n w3) → ((add v6 induction_n w1) → (add w3 v6 w2)))))) := by
      intro w1
      intro w2
      intro step_induction_assumption_1_premise_1
      intro w3
      intro step_induction_assumption_1_premise_2
      intro step_induction_assumption_1_premise_3
      apply induction_hypothesis
      all_goals assumption
    have step_induction_assumption_2 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have stepRule := peano_source_007_check_induction_condition N zero succ add mul one anchor relationalInduction
    exact stepRule induction_n v1 v2 v3 induction_m v5 v6 step_induction_assumption_1 step_premise_1 step_induction_assumption_2 step_premise_3 step_premise_2
  have inductionProperty : (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v5 : α), ((add v2 v4 v5) → (∀ (v6 : α), ((add v6 v4 v1) → (add v5 v6 v3))))))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v5 : α), ((add v2 induction_value v5) → (∀ (v6 : α), ((add v6 induction_value v1) → (add v5 v6 v3))))))))
      v4
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v1 v2 v3 premise_1 v5 premise_2 v6 premise_3

private theorem peano_source_019_induction_typing
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v2 : α)
    (v4 : α)
    (v6 : α)
    (assumption_10 : (mul v2 v6 v4))
    : (N v6) := by
  -- chapter_49_line_10: GL tag task formulation.
  have row_10 : (mul v2 v6 v4) := by
    exact assumption_10
  -- chapter_49_line_9: GL tag task formulation.
  have row_9 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_49_line_8: GL tag expansion.
  have row_8 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_9
  -- chapter_49_line_7: GL tag disintegration.
  have row_7 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_8.1
  -- chapter_49_line_6: GL tag expansion.
  have row_6 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_7
  -- chapter_49_line_5: GL tag disintegration.
  have row_5 : (gl_fXYZ mul N N N) := by
    exact row_6.1.1.1.2
  -- chapter_49_line_4: GL tag expansion.
  have row_4 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_5
  -- chapter_49_line_3: GL tag disintegration.
  have row_3 : (gl_implication9 mul N) := by
    exact row_4.1.1.1.2
  -- chapter_49_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication9] using row_3
  -- chapter_49_line_1: GL tag implication.
  have row_1 : (N v6) := by
    apply row_2
    exact row_10
  exact row_1

private theorem peano_source_019_check_zero
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (v4 : α)
    (v5 : α)
    (v6 : α)
    (assumption_45 : (v6 = zero))
    (assumption_34 : (mul v1 v4 v5))
    (assumption_20 : (mul v1 v2 v3))
    (assumption_16 : (mul v2 v6 v4))
    : (mul v3 v6 v5) := by
  -- chapter_50_line_51: GL tag theorem.
  have row_51 := peano_source_041 N zero succ add mul one anchor relationalInduction
  -- chapter_50_line_45: GL tag recursion.
  have row_45 : (v6 = zero) := by
    exact assumption_45
  -- chapter_50_line_44: GL tag symmetry of equality.
  have row_44 : (zero = v6) := by
    exact Eq.symm row_45
  -- chapter_50_line_40: GL tag theorem.
  have row_40 := peano_source_054 N zero succ add mul one anchor relationalInduction
  -- chapter_50_line_34: GL tag task formulation.
  have row_34 : (mul v1 v4 v5) := by
    exact assumption_34
  -- chapter_50_line_25: GL tag theorem.
  have row_25 := peano_source_016 N zero succ add mul one anchor relationalInduction
  -- chapter_50_line_23: GL tag theorem.
  have row_23 := peano_source_012 N zero succ add mul one anchor relationalInduction
  -- chapter_50_line_20: GL tag task formulation.
  have row_20 : (mul v1 v2 v3) := by
    exact assumption_20
  -- chapter_50_line_16: GL tag task formulation.
  have row_16 : (mul v2 v6 v4) := by
    exact assumption_16
  -- chapter_50_line_52: GL tag equality1.
  have row_52 : (mul v2 zero v4) := by
    have equality_source := row_16
    have equality_step_1 := row_45
    cases equality_step_1
    exact equality_source
  -- chapter_50_line_12: GL tag task formulation.
  have row_12 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_50_line_11: GL tag expansion.
  have row_11 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_12
  -- chapter_50_line_43: GL tag disintegration.
  have row_43 : (succ zero one) := by
    exact row_11.2
  -- chapter_50_line_42: GL tag equality1.
  have row_42 : (succ v6 one) := by
    have equality_source := row_43
    have equality_step_1 := row_44
    cases equality_step_1
    exact equality_source
  -- chapter_50_line_10: GL tag disintegration.
  have row_10 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_11.1
  -- chapter_50_line_9: GL tag expansion.
  have row_9 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_10
  -- chapter_50_line_68: GL tag disintegration.
  have row_68 : (gl_implication19 N zero mul) := by
    exact row_9.1.1.2
  -- chapter_50_line_67: GL tag equality1.
  have row_67 : (gl_implication19 N v6 mul) := by
    have equality_source := row_68
    have equality_step_1 := row_44
    cases equality_step_1
    exact equality_source
  -- chapter_50_line_32: GL tag disintegration.
  have row_32 : (gl_fXYZ add N N N) := by
    exact row_9.1.1.1.1.1.1.1.1.2
  -- chapter_50_line_31: GL tag expansion.
  have row_31 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_32
  -- chapter_50_line_30: GL tag disintegration.
  have row_30 : (gl_implication13 N N N add) := by
    exact row_31.1.2
  -- chapter_50_line_29: GL tag expansion.
  have row_29 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication13] using row_30
  -- chapter_50_line_8: GL tag disintegration.
  have row_8 : (gl_fXYZ mul N N N) := by
    exact row_9.1.1.1.2
  -- chapter_50_line_7: GL tag expansion.
  have row_7 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_8
  -- chapter_50_line_56: GL tag disintegration.
  have row_56 : (gl_implication14 N N mul) := by
    exact row_7.2
  -- chapter_50_line_55: GL tag expansion.
  have row_55 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (∀ (w3 : α), ((mul w1 w2 w3) → (∀ (w4 : α), ((mul w1 w2 w4) → (w3 = w4))))))))) := by
    simpa only [gl_implication14] using row_56
  -- chapter_50_line_37: GL tag disintegration.
  have row_37 : (gl_implication8 mul N) := by
    exact row_7.1.1.1.1
  -- chapter_50_line_36: GL tag expansion.
  have row_36 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_37
  -- chapter_50_line_69: GL tag implication.
  have row_69 : (N v1) := by
    apply row_36
    exact row_34
  -- chapter_50_line_35: GL tag implication.
  have row_35 : (N v2) := by
    apply row_36
    exact row_16
  -- chapter_50_line_50: GL tag implication.
  have row_50 : (zero = v4) := by
    apply row_51
    exact row_52
    exact row_35
  -- chapter_50_line_49: GL tag symmetry of equality.
  have row_49 : (v4 = zero) := by
    exact Eq.symm row_50
  -- chapter_50_line_48: GL tag equality2.
  have row_48 : (v4 = v6) := by
    exact Eq.trans row_49 row_44
  -- chapter_50_line_19: GL tag disintegration.
  have row_19 : (gl_implication10 mul N) := by
    exact row_7.1.1.2
  -- chapter_50_line_18: GL tag expansion.
  have row_18 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication10] using row_19
  -- chapter_50_line_17: GL tag implication.
  have row_17 : (N v3) := by
    apply row_18
    exact row_20
  -- chapter_50_line_15: GL tag disintegration.
  have row_15 : (gl_implication9 mul N) := by
    exact row_7.1.1.1.2
  -- chapter_50_line_14: GL tag expansion.
  have row_14 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication9] using row_15
  -- chapter_50_line_33: GL tag implication.
  have row_33 : (N v4) := by
    apply row_14
    exact row_34
  -- chapter_50_line_28: GL tag implication.
  have row_28 : (gl_existence1 N v2 v4 add) := by
    apply row_29
    exact row_35
    exact row_33
  -- chapter_50_line_27: GL tag expansion.
  have row_27 : (¬ (∀ (v11 : α), ((N v11) → (¬ (add v2 v4 v11))))) := by
    simpa only [gl_existence1] using row_28
  have exists_row_27 : ∃ (v11 : α), ((N v11) ∧ (add v2 v4 v11)) := existsAndOfNotForallImpNot row_27
  obtain ⟨v11, witness_row_27⟩ := exists_row_27
  -- chapter_50_line_26: GL tag disintegration.
  have row_26 : (add v2 v4 v11) := by
    exact witness_row_27.2
  -- chapter_50_line_24: GL tag implication.
  have row_24 : (add v4 v2 v11) := by
    apply row_25
    exact row_26
  -- chapter_50_line_47: GL tag equality1.
  have row_47 : (add v6 v2 v11) := by
    have equality_source := row_24
    have equality_step_1 := row_48
    cases equality_step_1
    exact equality_source
  -- chapter_50_line_46: GL tag implication.
  have row_46 : (v6 = v4) := by
    apply row_23
    exact row_47
    exact row_24
  -- chapter_50_line_66: GL tag equality1.
  have row_66 : (gl_implication19 N v4 mul) := by
    have equality_source := row_67
    have equality_step_1 := row_46
    cases equality_step_1
    exact equality_source
  -- chapter_50_line_65: GL tag expansion.
  have row_65 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((mul w1 v4 w2) → (w2 = v4))))) := by
    simpa only [gl_implication19] using row_66
  -- chapter_50_line_64: GL tag implication.
  have row_64 : (v5 = v4) := by
    apply row_65
    exact row_69
    exact row_34
  -- chapter_50_line_63: GL tag symmetry of equality.
  have row_63 : (v4 = v5) := by
    exact Eq.symm row_64
  -- chapter_50_line_41: GL tag equality1.
  have row_41 : (succ v4 one) := by
    have equality_source := row_42
    have equality_step_1 := row_46
    cases equality_step_1
    exact equality_source
  -- chapter_50_line_13: GL tag implication.
  have row_13 : (N v6) := by
    apply row_14
    exact row_16
  -- chapter_50_line_6: GL tag disintegration.
  have row_6 : (gl_implication13 N N N mul) := by
    exact row_7.1.2
  -- chapter_50_line_5: GL tag expansion.
  have row_5 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication13] using row_6
  -- chapter_50_line_4: GL tag implication.
  have row_4 : (gl_existence1 N v3 v6 mul) := by
    apply row_5
    exact row_17
    exact row_13
  -- chapter_50_line_3: GL tag expansion.
  have row_3 : (¬ (∀ (v7 : α), ((N v7) → (¬ (mul v3 v6 v7))))) := by
    simpa only [gl_existence1] using row_4
  have exists_row_3 : ∃ (v7 : α), ((N v7) ∧ (mul v3 v6 v7)) := existsAndOfNotForallImpNot row_3
  obtain ⟨v7, witness_row_3⟩ := exists_row_3
  -- chapter_50_line_2: GL tag disintegration.
  have row_2 : (mul v3 v6 v7) := by
    exact witness_row_3.2
  -- chapter_50_line_62: GL tag equality1.
  have row_62 : (mul v3 v4 v7) := by
    have equality_source := row_2
    have equality_step_1 := row_46
    cases equality_step_1
    exact equality_source
  -- chapter_50_line_61: GL tag equality1.
  have row_61 : (mul v3 zero v7) := by
    have equality_source := row_2
    have equality_step_1 := row_45
    cases equality_step_1
    exact equality_source
  -- chapter_50_line_60: GL tag implication.
  have row_60 : (zero = v7) := by
    apply row_51
    exact row_61
    exact row_17
  -- chapter_50_line_59: GL tag symmetry of equality.
  have row_59 : (v7 = zero) := by
    exact Eq.symm row_60
  -- chapter_50_line_58: GL tag equality2.
  have row_58 : (v7 = v6) := by
    exact Eq.trans row_59 row_44
  -- chapter_50_line_57: GL tag equality1.
  have row_57 : (mul v3 v4 v6) := by
    have equality_source := row_2
    have equality_step_1 := row_46
    cases equality_step_1
    have equality_step_2 := row_58
    cases equality_step_2
    exact equality_source
  -- chapter_50_line_54: GL tag implication.
  have row_54 : (v6 = v7) := by
    apply row_55
    exact row_17
    exact row_33
    exact row_57
    exact row_62
  -- chapter_50_line_53: GL tag equality1.
  have row_53 : (succ v7 one) := by
    have equality_source := row_42
    have equality_step_1 := row_54
    cases equality_step_1
    exact equality_source
  -- chapter_50_line_39: GL tag implication.
  have row_39 : (v4 = v7) := by
    apply row_40
    exact row_41
    exact row_53
  -- chapter_50_line_38: GL tag equality1.
  have row_38 : (add v7 v2 v11) := by
    have equality_source := row_24
    have equality_step_1 := row_39
    cases equality_step_1
    exact equality_source
  -- chapter_50_line_22: GL tag implication.
  have row_22 : (v7 = v4) := by
    apply row_23
    exact row_38
    exact row_24
  -- chapter_50_line_21: GL tag equality2.
  have row_21 : (v7 = v5) := by
    exact Eq.trans row_22 row_63
  -- chapter_50_line_1: GL tag equality1.
  have row_1 : (mul v3 v6 v5) := by
    have equality_source := row_2
    have equality_step_1 := row_21
    cases equality_step_1
    exact equality_source
  exact row_1

private theorem peano_source_019_check_induction_condition
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (v4 : α)
    (v5 : α)
    (v6 : α)
    (assumption_61 : ((mul v1 v2 v3) → (∀ (w1 : α) (w2 : α), ((mul v1 w1 w2) → ((mul v2 previous w1) → (mul v3 previous w2))))))
    (assumption_56 : (mul v2 v6 v4))
    (assumption_44 : (mul v1 v4 v5))
    (assumption_39 : (mul v1 v2 v3))
    (assumption_25 : (succ previous v6))
    : (mul v3 v6 v5) := by
  -- chapter_51_line_61: GL tag recursion.
  have row_61 : ((mul v1 v2 v3) → (∀ (w1 : α) (w2 : α), ((mul v1 w1 w2) → ((mul v2 previous w1) → (mul v3 previous w2))))) := by
    exact assumption_61
  -- chapter_51_line_56: GL tag task formulation.
  have row_56 : (mul v2 v6 v4) := by
    exact assumption_56
  -- chapter_51_line_44: GL tag task formulation.
  have row_44 : (mul v1 v4 v5) := by
    exact assumption_44
  -- chapter_51_line_43: GL tag theorem.
  have row_43 := peano_source_018 N zero succ add mul one anchor relationalInduction
  -- chapter_51_line_41: GL tag theorem.
  have row_41 := peano_source_039 N zero succ add mul one anchor relationalInduction
  -- chapter_51_line_39: GL tag task formulation.
  have row_39 : (mul v1 v2 v3) := by
    exact assumption_39
  -- chapter_51_line_27: GL tag theorem.
  have row_27 := peano_source_034 N zero succ add mul one anchor relationalInduction
  -- chapter_51_line_25: GL tag recursion.
  have row_25 : (succ previous v6) := by
    exact assumption_25
  -- chapter_51_line_7: GL tag task formulation.
  have row_7 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_51_line_26: GL tag implication.
  have row_26 : (add previous one v6) := by
    apply row_27
    exact row_25
  -- chapter_51_line_6: GL tag expansion.
  have row_6 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_7
  -- chapter_51_line_11: GL tag disintegration.
  have row_11 : (succ zero one) := by
    exact row_6.2
  -- chapter_51_line_5: GL tag disintegration.
  have row_5 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_6.1
  -- chapter_51_line_4: GL tag expansion.
  have row_4 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_5
  -- chapter_51_line_65: GL tag disintegration.
  have row_65 : (gl_implication21 N succ mul add) := by
    exact row_4.2
  -- chapter_51_line_64: GL tag expansion.
  have row_64 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((mul w3 w1 w4) → (∀ (w5 : α), ((mul w3 w2 w5) → (add w4 w3 w5))))))))) := by
    simpa only [gl_implication21] using row_65
  -- chapter_51_line_35: GL tag disintegration.
  have row_35 : (gl_fXYZ mul N N N) := by
    exact row_4.1.1.1.2
  -- chapter_51_line_34: GL tag expansion.
  have row_34 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_35
  -- chapter_51_line_59: GL tag disintegration.
  have row_59 : (gl_implication14 N N mul) := by
    exact row_34.2
  -- chapter_51_line_58: GL tag expansion.
  have row_58 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (∀ (w3 : α), ((mul w1 w2 w3) → (∀ (w4 : α), ((mul w1 w2 w4) → (w3 = w4))))))))) := by
    simpa only [gl_implication14] using row_59
  -- chapter_51_line_51: GL tag disintegration.
  have row_51 : (gl_implication8 mul N) := by
    exact row_34.1.1.1.1
  -- chapter_51_line_50: GL tag expansion.
  have row_50 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_51
  -- chapter_51_line_55: GL tag implication.
  have row_55 : (N v2) := by
    apply row_50
    exact row_56
  -- chapter_51_line_49: GL tag implication.
  have row_49 : (N v1) := by
    apply row_50
    exact row_44
  -- chapter_51_line_38: GL tag disintegration.
  have row_38 : (gl_implication10 mul N) := by
    exact row_34.1.1.2
  -- chapter_51_line_37: GL tag expansion.
  have row_37 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication10] using row_38
  -- chapter_51_line_36: GL tag implication.
  have row_36 : (N v3) := by
    apply row_37
    exact row_39
  -- chapter_51_line_33: GL tag disintegration.
  have row_33 : (gl_implication13 N N N mul) := by
    exact row_34.1.2
  -- chapter_51_line_32: GL tag expansion.
  have row_32 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication13] using row_33
  -- chapter_51_line_24: GL tag disintegration.
  have row_24 : (gl_fXY succ N N) := by
    exact row_4.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_51_line_23: GL tag expansion.
  have row_23 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_24
  -- chapter_51_line_22: GL tag disintegration.
  have row_22 : (gl_implication0 succ N) := by
    exact row_23.1.1.1
  -- chapter_51_line_21: GL tag expansion.
  have row_21 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_22
  -- chapter_51_line_20: GL tag implication.
  have row_20 : (N previous) := by
    apply row_21
    exact row_25
  -- chapter_51_line_54: GL tag implication.
  have row_54 : (gl_existence1 N v2 previous mul) := by
    apply row_32
    exact row_55
    exact row_20
  -- chapter_51_line_53: GL tag expansion.
  have row_53 : (¬ (∀ (v14 : α), ((N v14) → (¬ (mul v2 previous v14))))) := by
    simpa only [gl_existence1] using row_54
  have exists_row_53 : ∃ (v14 : α), ((N v14) ∧ (mul v2 previous v14)) := existsAndOfNotForallImpNot row_53
  obtain ⟨v14, witness_row_53⟩ := exists_row_53
  -- chapter_51_line_62: GL tag disintegration.
  have row_62 : (mul v2 previous v14) := by
    exact witness_row_53.2
  -- chapter_51_line_63: GL tag implication.
  have row_63 : (add v14 v2 v4) := by
    apply row_64
    exact row_20
    exact row_25
    exact row_62
    exact row_56
  -- chapter_51_line_52: GL tag disintegration.
  have row_52 : (N v14) := by
    exact witness_row_53.1
  -- chapter_51_line_48: GL tag implication.
  have row_48 : (gl_existence1 N v1 v14 mul) := by
    apply row_32
    exact row_49
    exact row_52
  -- chapter_51_line_47: GL tag expansion.
  have row_47 : (¬ (∀ (v15 : α), ((N v15) → (¬ (mul v1 v14 v15))))) := by
    simpa only [gl_existence1] using row_48
  have exists_row_47 : ∃ (v15 : α), ((N v15) ∧ (mul v1 v14 v15)) := existsAndOfNotForallImpNot row_47
  obtain ⟨v15, witness_row_47⟩ := exists_row_47
  -- chapter_51_line_46: GL tag disintegration.
  have row_46 : (mul v1 v14 v15) := by
    exact witness_row_47.2
  -- chapter_51_line_60: GL tag implication.
  have row_60 : (mul v3 previous v15) := by
    apply row_61
    exact row_39
    exact row_46
    exact row_62
  -- chapter_51_line_31: GL tag implication.
  have row_31 : (gl_existence1 N v3 previous mul) := by
    apply row_32
    exact row_36
    exact row_20
  -- chapter_51_line_30: GL tag expansion.
  have row_30 : (¬ (∀ (v13 : α), ((N v13) → (¬ (mul v3 previous v13))))) := by
    simpa only [gl_existence1] using row_31
  have exists_row_30 : ∃ (v13 : α), ((N v13) ∧ (mul v3 previous v13)) := existsAndOfNotForallImpNot row_30
  obtain ⟨v13, witness_row_30⟩ := exists_row_30
  -- chapter_51_line_29: GL tag disintegration.
  have row_29 : (mul v3 previous v13) := by
    exact witness_row_30.2
  -- chapter_51_line_57: GL tag implication.
  have row_57 : (v15 = v13) := by
    apply row_58
    exact row_36
    exact row_20
    exact row_60
    exact row_29
  -- chapter_51_line_45: GL tag equality1.
  have row_45 : (mul v1 v14 v13) := by
    have equality_source := row_46
    have equality_step_1 := row_57
    cases equality_step_1
    exact equality_source
  -- chapter_51_line_42: GL tag implication.
  have row_42 : (add v13 v3 v5) := by
    apply row_43
    exact row_45
    exact row_39
    exact row_44
    exact row_63
  -- chapter_51_line_19: GL tag disintegration.
  have row_19 : (N zero) := by
    exact row_4.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_51_line_18: GL tag disintegration.
  have row_18 : (gl_fXYZ add N N N) := by
    exact row_4.1.1.1.1.1.1.1.1.2
  -- chapter_51_line_17: GL tag expansion.
  have row_17 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_18
  -- chapter_51_line_16: GL tag disintegration.
  have row_16 : (gl_implication13 N N N add) := by
    exact row_17.1.2
  -- chapter_51_line_15: GL tag expansion.
  have row_15 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication13] using row_16
  -- chapter_51_line_14: GL tag implication.
  have row_14 : (gl_existence1 N previous zero add) := by
    apply row_15
    exact row_20
    exact row_19
  -- chapter_51_line_13: GL tag expansion.
  have row_13 : (¬ (∀ (v12 : α), ((N v12) → (¬ (add previous zero v12))))) := by
    simpa only [gl_existence1] using row_14
  have exists_row_13 : ∃ (v12 : α), ((N v12) ∧ (add previous zero v12)) := existsAndOfNotForallImpNot row_13
  obtain ⟨v12, witness_row_13⟩ := exists_row_13
  -- chapter_51_line_66: GL tag disintegration.
  have row_66 : (N v12) := by
    exact witness_row_13.1
  -- chapter_51_line_12: GL tag disintegration.
  have row_12 : (add previous zero v12) := by
    exact witness_row_13.2
  -- chapter_51_line_40: GL tag implication.
  have row_40 : (previous = v12) := by
    apply row_41
    exact row_12
    exact row_20
  -- chapter_51_line_28: GL tag equality1.
  have row_28 : (mul v3 v12 v13) := by
    have equality_source := row_29
    have equality_step_1 := row_40
    cases equality_step_1
    exact equality_source
  -- chapter_51_line_10: GL tag disintegration.
  have row_10 : (gl_implication17 N succ add) := by
    exact row_4.1.1.1.1.1.2
  -- chapter_51_line_9: GL tag expansion.
  have row_9 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication17] using row_10
  -- chapter_51_line_8: GL tag implication.
  have row_8 : (succ v12 v6) := by
    apply row_9
    exact row_19
    exact row_11
    exact row_12
    exact row_26
  -- chapter_51_line_3: GL tag disintegration.
  have row_3 : (gl_implication20 N succ mul add) := by
    exact row_4.1.2
  -- chapter_51_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((mul w3 w1 w4) → (∀ (w5 : α), ((add w4 w3 w5) → (mul w3 w2 w5))))))))) := by
    simpa only [gl_implication20] using row_3
  -- chapter_51_line_1: GL tag implication.
  have row_1 : (mul v3 v6 v5) := by
    apply row_2
    exact row_66
    exact row_8
    exact row_28
    exact row_42
  exact row_1

theorem peano_source_019
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α) (v3 : α), ((mul v1 v2 v3) → (∀ (v4 : α) (v5 : α), ((mul v1 v4 v5) → (∀ (v6 : α), ((mul v2 v6 v4) → (mul v3 v6 v5))))))) := by
  intro v1
  intro v2
  intro v3
  intro premise_1
  intro v4
  intro v5
  intro premise_2
  intro v6
  intro premise_3
  have inductionMember : N v6 := by
    have typingRule := peano_source_019_induction_typing N zero succ add mul one anchor relationalInduction
    exact typingRule v2 v4 v6 premise_3
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v1 : α) (v2 : α) (v3 : α), ((mul v1 v2 v3) → (∀ (v4 : α) (v5 : α), ((mul v1 v4 v5) → ((mul v2 zero v4) → (mul v3 zero v5)))))) := by
    intro v1
    intro v2
    intro v3
    intro base_premise_1
    intro v4
    intro v5
    intro base_premise_2
    intro base_premise_3
    have zeroRule := peano_source_019_check_zero N zero succ add mul one anchor relationalInduction
    exact zeroRule v1 v2 v3 v4 v5 zero rfl base_premise_2 base_premise_1 base_premise_3
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v1 : α) (v2 : α) (v3 : α), ((mul v1 v2 v3) → (∀ (v4 : α) (v5 : α), ((mul v1 v4 v5) → ((mul v2 induction_n v4) → (mul v3 induction_n v5)))))) → ∀ induction_m, succ induction_n induction_m → (∀ (v1 : α) (v2 : α) (v3 : α), ((mul v1 v2 v3) → (∀ (v4 : α) (v5 : α), ((mul v1 v4 v5) → ((mul v2 induction_m v4) → (mul v3 induction_m v5)))))) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro v1
    intro v2
    intro v3
    intro step_premise_1
    intro v4
    intro v5
    intro step_premise_2
    intro step_premise_3
    have step_induction_assumption_1 :
        ((mul v1 v2 v3) → (∀ (w1 : α) (w2 : α), ((mul v1 w1 w2) → ((mul v2 induction_n w1) → (mul v3 induction_n w2))))) := by
      intro step_induction_assumption_1_premise_1
      intro w1
      intro w2
      intro step_induction_assumption_1_premise_2
      intro step_induction_assumption_1_premise_3
      apply induction_hypothesis
      all_goals assumption
    have step_induction_assumption_2 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have stepRule := peano_source_019_check_induction_condition N zero succ add mul one anchor relationalInduction
    exact stepRule induction_n v1 v2 v3 v4 v5 induction_m step_induction_assumption_1 step_premise_3 step_premise_2 step_premise_1 step_induction_assumption_2
  have inductionProperty : (∀ (v1 : α) (v2 : α) (v3 : α), ((mul v1 v2 v3) → (∀ (v4 : α) (v5 : α), ((mul v1 v4 v5) → ((mul v2 v6 v4) → (mul v3 v6 v5)))))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v1 : α) (v2 : α) (v3 : α), ((mul v1 v2 v3) → (∀ (v4 : α) (v5 : α), ((mul v1 v4 v5) → ((mul v2 induction_value v4) → (mul v3 induction_value v5)))))))
      v6
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v1 v2 v3 premise_1 v4 v5 premise_2 premise_3

private theorem peano_source_022_induction_typing
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (assumption_10 : (mul v1 v2 v3))
    : (N v2) := by
  -- chapter_54_line_10: GL tag task formulation.
  have row_10 : (mul v1 v2 v3) := by
    exact assumption_10
  -- chapter_54_line_9: GL tag task formulation.
  have row_9 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_54_line_8: GL tag expansion.
  have row_8 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_9
  -- chapter_54_line_7: GL tag disintegration.
  have row_7 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_8.1
  -- chapter_54_line_6: GL tag expansion.
  have row_6 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_7
  -- chapter_54_line_5: GL tag disintegration.
  have row_5 : (gl_fXYZ mul N N N) := by
    exact row_6.1.1.1.2
  -- chapter_54_line_4: GL tag expansion.
  have row_4 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_5
  -- chapter_54_line_3: GL tag disintegration.
  have row_3 : (gl_implication9 mul N) := by
    exact row_4.1.1.1.2
  -- chapter_54_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication9] using row_3
  -- chapter_54_line_1: GL tag implication.
  have row_1 : (N v2) := by
    apply row_2
    exact row_10
  exact row_1

private theorem peano_source_022_check_zero
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (assumption_21 : (v2 = zero))
    (assumption_17 : (mul v1 v2 v3))
    : (mul v2 v1 v3) := by
  -- chapter_55_line_28: GL tag theorem.
  have row_28 := peano_source_041 N zero succ add mul one anchor relationalInduction
  -- chapter_55_line_26: GL tag theorem.
  have row_26 := peano_source_056 N zero succ add mul one anchor relationalInduction
  -- chapter_55_line_21: GL tag recursion.
  have row_21 : (v2 = zero) := by
    exact assumption_21
  -- chapter_55_line_22: GL tag symmetry of equality.
  have row_22 : (zero = v2) := by
    exact Eq.symm row_21
  -- chapter_55_line_17: GL tag task formulation.
  have row_17 : (mul v1 v2 v3) := by
    exact assumption_17
  -- chapter_55_line_29: GL tag equality1.
  have row_29 : (mul v1 zero v3) := by
    have equality_source := row_17
    have equality_step_1 := row_21
    cases equality_step_1
    exact equality_source
  -- chapter_55_line_13: GL tag task formulation.
  have row_13 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_55_line_12: GL tag expansion.
  have row_12 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_13
  -- chapter_55_line_11: GL tag disintegration.
  have row_11 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_12.1
  -- chapter_55_line_10: GL tag expansion.
  have row_10 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_11
  -- chapter_55_line_9: GL tag disintegration.
  have row_9 : (gl_fXYZ mul N N N) := by
    exact row_10.1.1.1.2
  -- chapter_55_line_8: GL tag expansion.
  have row_8 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_9
  -- chapter_55_line_20: GL tag disintegration.
  have row_20 : (gl_implication9 mul N) := by
    exact row_8.1.1.1.2
  -- chapter_55_line_19: GL tag expansion.
  have row_19 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication9] using row_20
  -- chapter_55_line_18: GL tag implication.
  have row_18 : (N v2) := by
    apply row_19
    exact row_17
  -- chapter_55_line_16: GL tag disintegration.
  have row_16 : (gl_implication8 mul N) := by
    exact row_8.1.1.1.1
  -- chapter_55_line_15: GL tag expansion.
  have row_15 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_16
  -- chapter_55_line_14: GL tag implication.
  have row_14 : (N v1) := by
    apply row_15
    exact row_17
  -- chapter_55_line_27: GL tag implication.
  have row_27 : (zero = v3) := by
    apply row_28
    exact row_29
    exact row_14
  -- chapter_55_line_7: GL tag disintegration.
  have row_7 : (gl_implication13 N N N mul) := by
    exact row_8.1.2
  -- chapter_55_line_6: GL tag expansion.
  have row_6 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication13] using row_7
  -- chapter_55_line_5: GL tag implication.
  have row_5 : (gl_existence1 N v2 v1 mul) := by
    apply row_6
    exact row_18
    exact row_14
  -- chapter_55_line_4: GL tag expansion.
  have row_4 : (¬ (∀ (v4 : α), ((N v4) → (¬ (mul v2 v1 v4))))) := by
    simpa only [gl_existence1] using row_5
  have exists_row_4 : ∃ (v4 : α), ((N v4) ∧ (mul v2 v1 v4)) := existsAndOfNotForallImpNot row_4
  obtain ⟨v4, witness_row_4⟩ := exists_row_4
  -- chapter_55_line_3: GL tag disintegration.
  have row_3 : (mul v2 v1 v4) := by
    exact witness_row_4.2
  -- chapter_55_line_2: GL tag equality1.
  have row_2 : (mul zero v1 v4) := by
    have equality_source := row_3
    have equality_step_1 := row_21
    cases equality_step_1
    exact equality_source
  -- chapter_55_line_25: GL tag implication.
  have row_25 : (zero = v4) := by
    apply row_26
    exact row_14
    exact row_2
  -- chapter_55_line_24: GL tag symmetry of equality.
  have row_24 : (v4 = zero) := by
    exact Eq.symm row_25
  -- chapter_55_line_23: GL tag equality2.
  have row_23 : (v4 = v3) := by
    exact Eq.trans row_24 row_27
  -- chapter_55_line_1: GL tag equality1.
  have row_1 : (mul v2 v1 v3) := by
    have equality_source := row_2
    have equality_step_1 := row_22
    cases equality_step_1
    have equality_step_2 := row_23
    cases equality_step_2
    exact equality_source
  exact row_1

private theorem peano_source_022_check_induction_condition
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (assumption_45 : (∀ (w1 : α), ((mul v1 previous w1) → (mul previous v1 w1))))
    (assumption_32 : (mul v1 v2 v3))
    (assumption_24 : (succ previous v2))
    : (mul v2 v1 v3) := by
  -- chapter_56_line_47: GL tag theorem.
  have row_47 := peano_source_039 N zero succ add mul one anchor relationalInduction
  -- chapter_56_line_45: GL tag recursion.
  have row_45 : (∀ (w1 : α), ((mul v1 previous w1) → (mul previous v1 w1))) := by
    exact assumption_45
  -- chapter_56_line_32: GL tag task formulation.
  have row_32 : (mul v1 v2 v3) := by
    exact assumption_32
  -- chapter_56_line_28: GL tag theorem.
  have row_28 := peano_source_016 N zero succ add mul one anchor relationalInduction
  -- chapter_56_line_26: GL tag theorem.
  have row_26 := peano_source_034 N zero succ add mul one anchor relationalInduction
  -- chapter_56_line_24: GL tag recursion.
  have row_24 : (succ previous v2) := by
    exact assumption_24
  -- chapter_56_line_3: GL tag task formulation.
  have row_3 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_56_line_25: GL tag implication.
  have row_25 : (add previous one v2) := by
    apply row_26
    exact row_24
  -- chapter_56_line_9: GL tag expansion.
  have row_9 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_3
  -- chapter_56_line_10: GL tag disintegration.
  have row_10 : (succ zero one) := by
    exact row_9.2
  -- chapter_56_line_8: GL tag disintegration.
  have row_8 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_9.1
  -- chapter_56_line_7: GL tag expansion.
  have row_7 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_8
  -- chapter_56_line_39: GL tag disintegration.
  have row_39 : (gl_fXYZ mul N N N) := by
    exact row_7.1.1.1.2
  -- chapter_56_line_38: GL tag expansion.
  have row_38 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_39
  -- chapter_56_line_42: GL tag disintegration.
  have row_42 : (gl_implication8 mul N) := by
    exact row_38.1.1.1.1
  -- chapter_56_line_41: GL tag expansion.
  have row_41 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_42
  -- chapter_56_line_40: GL tag implication.
  have row_40 : (N v1) := by
    apply row_41
    exact row_32
  -- chapter_56_line_37: GL tag disintegration.
  have row_37 : (gl_implication13 N N N mul) := by
    exact row_38.1.2
  -- chapter_56_line_36: GL tag expansion.
  have row_36 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication13] using row_37
  -- chapter_56_line_31: GL tag disintegration.
  have row_31 : (gl_implication21 N succ mul add) := by
    exact row_7.2
  -- chapter_56_line_30: GL tag expansion.
  have row_30 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((mul w3 w1 w4) → (∀ (w5 : α), ((mul w3 w2 w5) → (add w4 w3 w5))))))))) := by
    simpa only [gl_implication21] using row_31
  -- chapter_56_line_23: GL tag disintegration.
  have row_23 : (gl_fXY succ N N) := by
    exact row_7.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_56_line_22: GL tag expansion.
  have row_22 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_23
  -- chapter_56_line_21: GL tag disintegration.
  have row_21 : (gl_implication0 succ N) := by
    exact row_22.1.1.1
  -- chapter_56_line_20: GL tag expansion.
  have row_20 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_21
  -- chapter_56_line_19: GL tag implication.
  have row_19 : (N previous) := by
    apply row_20
    exact row_24
  -- chapter_56_line_35: GL tag implication.
  have row_35 : (gl_existence1 N v1 previous mul) := by
    apply row_36
    exact row_40
    exact row_19
  -- chapter_56_line_34: GL tag expansion.
  have row_34 : (¬ (∀ (v5 : α), ((N v5) → (¬ (mul v1 previous v5))))) := by
    simpa only [gl_existence1] using row_35
  have exists_row_34 : ∃ (v5 : α), ((N v5) ∧ (mul v1 previous v5)) := existsAndOfNotForallImpNot row_34
  obtain ⟨v5, witness_row_34⟩ := exists_row_34
  -- chapter_56_line_33: GL tag disintegration.
  have row_33 : (mul v1 previous v5) := by
    exact witness_row_34.2
  -- chapter_56_line_44: GL tag implication.
  have row_44 : (mul previous v1 v5) := by
    apply row_45
    exact row_33
  -- chapter_56_line_29: GL tag implication.
  have row_29 : (add v5 v1 v3) := by
    apply row_30
    exact row_19
    exact row_24
    exact row_33
    exact row_32
  -- chapter_56_line_27: GL tag implication.
  have row_27 : (add v1 v5 v3) := by
    apply row_28
    exact row_29
  -- chapter_56_line_18: GL tag disintegration.
  have row_18 : (N zero) := by
    exact row_7.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_56_line_17: GL tag disintegration.
  have row_17 : (gl_fXYZ add N N N) := by
    exact row_7.1.1.1.1.1.1.1.1.2
  -- chapter_56_line_16: GL tag expansion.
  have row_16 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_17
  -- chapter_56_line_15: GL tag disintegration.
  have row_15 : (gl_implication13 N N N add) := by
    exact row_16.1.2
  -- chapter_56_line_14: GL tag expansion.
  have row_14 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication13] using row_15
  -- chapter_56_line_13: GL tag implication.
  have row_13 : (gl_existence1 N previous zero add) := by
    apply row_14
    exact row_19
    exact row_18
  -- chapter_56_line_12: GL tag expansion.
  have row_12 : (¬ (∀ (v4 : α), ((N v4) → (¬ (add previous zero v4))))) := by
    simpa only [gl_existence1] using row_13
  have exists_row_12 : ∃ (v4 : α), ((N v4) ∧ (add previous zero v4)) := existsAndOfNotForallImpNot row_12
  obtain ⟨v4, witness_row_12⟩ := exists_row_12
  -- chapter_56_line_11: GL tag disintegration.
  have row_11 : (add previous zero v4) := by
    exact witness_row_12.2
  -- chapter_56_line_46: GL tag implication.
  have row_46 : (previous = v4) := by
    apply row_47
    exact row_11
    exact row_19
  -- chapter_56_line_43: GL tag equality1.
  have row_43 : (mul v4 v1 v5) := by
    have equality_source := row_44
    have equality_step_1 := row_46
    cases equality_step_1
    exact equality_source
  -- chapter_56_line_6: GL tag disintegration.
  have row_6 : (gl_implication17 N succ add) := by
    exact row_7.1.1.1.1.1.2
  -- chapter_56_line_5: GL tag expansion.
  have row_5 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication17] using row_6
  -- chapter_56_line_4: GL tag implication.
  have row_4 : (succ v4 v2) := by
    apply row_5
    exact row_18
    exact row_10
    exact row_11
    exact row_25
  -- chapter_56_line_2: GL tag theorem.
  have row_2 := peano_source_002 N zero succ add mul one anchor relationalInduction
  -- chapter_56_line_1: GL tag implication.
  have row_1 : (mul v2 v1 v3) := by
    apply row_2
    exact row_27
    exact row_4
    exact row_43
  exact row_1

theorem peano_source_022
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α) (v3 : α), ((mul v1 v2 v3) → (mul v2 v1 v3))) := by
  intro v1
  intro v2
  intro v3
  intro premise_1
  have inductionMember : N v2 := by
    have typingRule := peano_source_022_induction_typing N zero succ add mul one anchor relationalInduction
    exact typingRule v1 v2 v3 premise_1
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v1 : α) (v3 : α), ((mul v1 zero v3) → (mul zero v1 v3))) := by
    intro v1
    intro v3
    intro base_premise_1
    have zeroRule := peano_source_022_check_zero N zero succ add mul one anchor relationalInduction
    exact zeroRule v1 zero v3 rfl base_premise_1
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v1 : α) (v3 : α), ((mul v1 induction_n v3) → (mul induction_n v1 v3))) → ∀ induction_m, succ induction_n induction_m → (∀ (v1 : α) (v3 : α), ((mul v1 induction_m v3) → (mul induction_m v1 v3))) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro v1
    intro v3
    intro step_premise_1
    have step_induction_assumption_1 :
        (∀ (w1 : α), ((mul v1 induction_n w1) → (mul induction_n v1 w1))) := by
      intro w1
      intro step_induction_assumption_1_premise_1
      apply induction_hypothesis
      all_goals assumption
    have step_induction_assumption_2 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have stepRule := peano_source_022_check_induction_condition N zero succ add mul one anchor relationalInduction
    exact stepRule induction_n v1 induction_m v3 step_induction_assumption_1 step_premise_1 step_induction_assumption_2
  have inductionProperty : (∀ (v1 : α) (v3 : α), ((mul v1 v2 v3) → (mul v2 v1 v3))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v1 : α) (v3 : α), ((mul v1 induction_value v3) → (mul induction_value v1 v3))))
      v2
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v1 v3 premise_1

theorem peano_source_052
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((gl_preorder N add v1 v2) → ((gl_preorder N add v2 v1) → (v1 = v2)))) := by
  intro v1
  intro v2
  intro premise_1
  intro premise_2
  -- chapter_116_line_41: GL tag theorem.
  have row_41 := peano_source_004 N zero succ add mul one anchor relationalInduction
  -- chapter_116_line_39: GL tag variable copy.
  have row_39 : (v1 = v1) := by
    rfl
  -- chapter_116_line_38: GL tag theorem.
  have row_38 := peano_source_014 N zero succ add mul one anchor relationalInduction
  -- chapter_116_line_29: GL tag theorem.
  have row_29 := peano_source_045 N zero succ add mul one anchor relationalInduction
  -- chapter_116_line_26: GL tag task formulation.
  have row_26 : (gl_preorder N add v1 v2) := by
    exact premise_1
  -- chapter_116_line_25: GL tag expansion.
  have row_25 : (¬ (∀ (v5 : α), ((N v5) → (¬ (add v1 v5 v2))))) := by
    simpa only [gl_preorder] using row_26
  have exists_row_25 : ∃ (v5 : α), ((N v5) ∧ (add v1 v5 v2)) := existsAndOfNotForallImpNot row_25
  obtain ⟨v5, witness_row_25⟩ := exists_row_25
  -- chapter_116_line_42: GL tag disintegration.
  have row_42 : (add v1 v5 v2) := by
    exact witness_row_25.2
  -- chapter_116_line_24: GL tag disintegration.
  have row_24 : (N v5) := by
    exact witness_row_25.1
  -- chapter_116_line_23: GL tag task formulation.
  have row_23 : (gl_preorder N add v2 v1) := by
    exact premise_2
  -- chapter_116_line_22: GL tag expansion.
  have row_22 : (¬ (∀ (v8 : α), ((N v8) → (¬ (add v2 v8 v1))))) := by
    simpa only [gl_preorder] using row_23
  have exists_row_22 : ∃ (v8 : α), ((N v8) ∧ (add v2 v8 v1)) := existsAndOfNotForallImpNot row_22
  obtain ⟨v8, witness_row_22⟩ := exists_row_22
  -- chapter_116_line_35: GL tag disintegration.
  have row_35 : (N v8) := by
    exact witness_row_22.1
  -- chapter_116_line_21: GL tag disintegration.
  have row_21 : (add v2 v8 v1) := by
    exact witness_row_22.2
  -- chapter_116_line_43: GL tag equality1.
  have row_43 : (add v2 v8 v1) := by
    have equality_source := row_21
    have equality_step_1 := row_39
    cases equality_step_1
    exact equality_source
  -- chapter_116_line_10: GL tag theorem.
  have row_10 := peano_source_016 N zero succ add mul one anchor relationalInduction
  -- chapter_116_line_7: GL tag task formulation.
  have row_7 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_116_line_47: GL tag implication.
  have row_47 : (add v5 v1 v2) := by
    apply row_10
    exact row_42
  -- chapter_116_line_6: GL tag expansion.
  have row_6 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_7
  -- chapter_116_line_5: GL tag disintegration.
  have row_5 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_6.1
  -- chapter_116_line_4: GL tag expansion.
  have row_4 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_5
  -- chapter_116_line_17: GL tag disintegration.
  have row_17 : (gl_fXYZ add N N N) := by
    exact row_4.1.1.1.1.1.1.1.1.2
  -- chapter_116_line_16: GL tag expansion.
  have row_16 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_17
  -- chapter_116_line_46: GL tag disintegration.
  have row_46 : (gl_implication14 N N add) := by
    exact row_16.2
  -- chapter_116_line_45: GL tag expansion.
  have row_45 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (∀ (w3 : α), ((add w1 w2 w3) → (∀ (w4 : α), ((add w1 w2 w4) → (w3 = w4))))))))) := by
    simpa only [gl_implication14] using row_46
  -- chapter_116_line_20: GL tag disintegration.
  have row_20 : (gl_implication10 add N) := by
    exact row_16.1.1.2
  -- chapter_116_line_19: GL tag expansion.
  have row_19 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication10] using row_20
  -- chapter_116_line_18: GL tag implication.
  have row_18 : (N v1) := by
    apply row_19
    exact row_21
  -- chapter_116_line_15: GL tag disintegration.
  have row_15 : (gl_implication13 N N N add) := by
    exact row_16.1.2
  -- chapter_116_line_14: GL tag expansion.
  have row_14 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication13] using row_15
  -- chapter_116_line_34: GL tag implication.
  have row_34 : (gl_existence1 N v5 v8 add) := by
    apply row_14
    exact row_24
    exact row_35
  -- chapter_116_line_33: GL tag expansion.
  have row_33 : (¬ (∀ (v9 : α), ((N v9) → (¬ (add v5 v8 v9))))) := by
    simpa only [gl_existence1] using row_34
  have exists_row_33 : ∃ (v9 : α), ((N v9) ∧ (add v5 v8 v9)) := existsAndOfNotForallImpNot row_33
  obtain ⟨v9, witness_row_33⟩ := exists_row_33
  -- chapter_116_line_32: GL tag disintegration.
  have row_32 : (add v5 v8 v9) := by
    exact witness_row_33.2
  -- chapter_116_line_40: GL tag implication.
  have row_40 : (add v1 v9 v1) := by
    apply row_41
    exact row_43
    exact row_42
    exact row_32
  -- chapter_116_line_37: GL tag implication.
  have row_37 : (zero = v9) := by
    apply row_38
    exact row_40
    exact row_39
  -- chapter_116_line_36: GL tag symmetry of equality.
  have row_36 : (v9 = zero) := by
    exact Eq.symm row_37
  -- chapter_116_line_31: GL tag implication.
  have row_31 : (add v8 v5 v9) := by
    apply row_10
    exact row_32
  -- chapter_116_line_30: GL tag equality1.
  have row_30 : (add v8 v5 zero) := by
    have equality_source := row_31
    have equality_step_1 := row_36
    cases equality_step_1
    exact equality_source
  -- chapter_116_line_28: GL tag implication.
  have row_28 : (zero = v5) := by
    apply row_29
    exact row_30
    exact row_35
  -- chapter_116_line_27: GL tag symmetry of equality.
  have row_27 : (v5 = zero) := by
    exact Eq.symm row_28
  -- chapter_116_line_13: GL tag implication.
  have row_13 : (gl_existence1 N v5 v1 add) := by
    apply row_14
    exact row_24
    exact row_18
  -- chapter_116_line_12: GL tag expansion.
  have row_12 : (¬ (∀ (v6 : α), ((N v6) → (¬ (add v5 v1 v6))))) := by
    simpa only [gl_existence1] using row_13
  have exists_row_12 : ∃ (v6 : α), ((N v6) ∧ (add v5 v1 v6)) := existsAndOfNotForallImpNot row_12
  obtain ⟨v6, witness_row_12⟩ := exists_row_12
  -- chapter_116_line_11: GL tag disintegration.
  have row_11 : (add v5 v1 v6) := by
    exact witness_row_12.2
  -- chapter_116_line_44: GL tag implication.
  have row_44 : (v6 = v2) := by
    apply row_45
    exact row_24
    exact row_18
    exact row_11
    exact row_47
  -- chapter_116_line_9: GL tag implication.
  have row_9 : (add v1 v5 v6) := by
    apply row_10
    exact row_11
  -- chapter_116_line_8: GL tag equality1.
  have row_8 : (add v1 zero v2) := by
    have equality_source := row_9
    have equality_step_1 := row_27
    cases equality_step_1
    have equality_step_2 := row_44
    cases equality_step_2
    exact equality_source
  -- chapter_116_line_3: GL tag disintegration.
  have row_3 : (gl_implication15 N zero add) := by
    exact row_4.1.1.1.1.1.1.1.2
  -- chapter_116_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((add w1 zero w2) → (w1 = w2))))) := by
    simpa only [gl_implication15] using row_3
  -- chapter_116_line_1: GL tag implication.
  have row_1 : (v1 = v2) := by
    apply row_2
    exact row_18
    exact row_8
  exact row_1

theorem peano_source_017
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α) (v3 : α), ((mul v1 v2 v3) → (∀ (v4 : α) (v5 : α) (v6 : α), ((mul v4 v5 v6) → ((mul v2 v1 v5) → (mul v4 v3 v6)))))) := by
  intro v1
  intro v2
  intro v3
  intro premise_1
  intro v4
  intro v5
  intro v6
  intro premise_2
  intro premise_3
  -- chapter_45_line_15: GL tag task formulation.
  have row_15 : (mul v1 v2 v3) := by
    exact premise_1
  -- chapter_45_line_14: GL tag task formulation.
  have row_14 : (mul v2 v1 v5) := by
    exact premise_3
  -- chapter_45_line_13: GL tag theorem.
  have row_13 := peano_source_022 N zero succ add mul one anchor relationalInduction
  -- chapter_45_line_11: GL tag task formulation.
  have row_11 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_45_line_12: GL tag implication.
  have row_12 : (mul v1 v2 v5) := by
    apply row_13
    exact row_14
  -- chapter_45_line_10: GL tag expansion.
  have row_10 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using row_11
  -- chapter_45_line_9: GL tag disintegration.
  have row_9 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_10.1
  -- chapter_45_line_8: GL tag expansion.
  have row_8 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_9
  -- chapter_45_line_7: GL tag disintegration.
  have row_7 : (gl_fXYZ mul N N N) := by
    exact row_8.1.1.1.2
  -- chapter_45_line_6: GL tag expansion.
  have row_6 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_7
  -- chapter_45_line_21: GL tag disintegration.
  have row_21 : (gl_implication8 mul N) := by
    exact row_6.1.1.1.1
  -- chapter_45_line_20: GL tag expansion.
  have row_20 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_21
  -- chapter_45_line_19: GL tag implication.
  have row_19 : (N v2) := by
    apply row_20
    exact row_14
  -- chapter_45_line_18: GL tag disintegration.
  have row_18 : (gl_implication9 mul N) := by
    exact row_6.1.1.1.2
  -- chapter_45_line_17: GL tag expansion.
  have row_17 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication9] using row_18
  -- chapter_45_line_16: GL tag implication.
  have row_16 : (N v1) := by
    apply row_17
    exact row_14
  -- chapter_45_line_5: GL tag disintegration.
  have row_5 : (gl_implication14 N N mul) := by
    exact row_6.2
  -- chapter_45_line_4: GL tag expansion.
  have row_4 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (∀ (w3 : α), ((mul w1 w2 w3) → (∀ (w4 : α), ((mul w1 w2 w4) → (w3 = w4))))))))) := by
    simpa only [gl_implication14] using row_5
  -- chapter_45_line_3: GL tag implication.
  have row_3 : (v5 = v3) := by
    apply row_4
    exact row_16
    exact row_19
    exact row_12
    exact row_15
  -- chapter_45_line_2: GL tag task formulation.
  have row_2 : (mul v4 v5 v6) := by
    exact premise_2
  -- chapter_45_line_1: GL tag equality1.
  have row_1 : (mul v4 v3 v6) := by
    have equality_source := row_2
    have equality_step_1 := row_3
    cases equality_step_1
    exact equality_source
  exact row_1

theorem peano_source_021
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α) (v3 : α), ((mul v1 v2 v3) → (∀ (v4 : α) (v5 : α), ((mul v2 v4 v5) → (∀ (v6 : α), ((mul v6 v4 v1) → (mul v5 v6 v3))))))) := by
  intro v1
  intro v2
  intro v3
  intro premise_1
  intro v4
  intro v5
  intro premise_2
  intro v6
  intro premise_3
  -- chapter_53_line_9: GL tag task formulation.
  have row_9 : (mul v1 v2 v3) := by
    exact premise_1
  -- chapter_53_line_7: GL tag task formulation.
  have row_7 : (mul v2 v4 v5) := by
    exact premise_2
  -- chapter_53_line_6: GL tag task formulation.
  have row_6 : (mul v6 v4 v1) := by
    exact premise_3
  -- chapter_53_line_5: GL tag theorem.
  have row_5 := peano_source_022 N zero succ add mul one anchor relationalInduction
  -- chapter_53_line_3: GL tag task formulation.
  have row_3 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_53_line_8: GL tag implication.
  have row_8 : (mul v2 v1 v3) := by
    apply row_5
    exact row_9
  -- chapter_53_line_4: GL tag implication.
  have row_4 : (mul v4 v6 v1) := by
    apply row_5
    exact row_6
  -- chapter_53_line_2: GL tag theorem.
  have row_2 := peano_source_019 N zero succ add mul one anchor relationalInduction
  -- chapter_53_line_1: GL tag implication.
  have row_1 : (mul v5 v6 v3) := by
    apply row_2
    exact row_7
    exact row_8
    exact row_4
  exact row_1

theorem peano_source_051
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (anchor : gl_AnchorPeano N zero succ add mul one)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((mul v1 v2 one) → (mul v2 v1 one))) := by
  intro v1
  intro v2
  intro premise_1
  -- chapter_115_line_5: GL tag task formulation.
  have row_5 : (mul v1 v2 one) := by
    exact premise_1
  -- chapter_115_line_4: GL tag task formulation.
  have row_4 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_115_line_3: GL tag anchor handling.
  have row_3 : (gl_AnchorPeano N zero succ add mul one) := by
    exact row_4
  -- chapter_115_line_2: GL tag theorem.
  have row_2 := peano_source_022 N zero succ add mul one anchor relationalInduction
  -- chapter_115_line_1: GL tag implication.
  have row_1 : (mul v2 v1 one) := by
    apply row_2
    exact row_5
  exact row_1

end GLExport
