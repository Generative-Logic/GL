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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((¬ (v1 = v2)) → ((add v2 v1 zero) → (¬ (zero = v1))))) := by
  intro v1
  intro v2
  intro premise_1
  intro premise_2
  intro reductio
  -- chapter_0_line_20: GL tag task formulation.
  have row_20 : (zero = v1) := by
    exact reductio
  -- chapter_0_line_19: GL tag symmetry of equality.
  have row_19 : (v1 = zero) := by
    exact Eq.symm row_20
  -- chapter_0_line_18: GL tag task formulation.
  have row_18 : (add v2 v1 zero) := by
    exact premise_2
  -- chapter_0_line_17: GL tag equality1.
  have row_17 : (add v2 zero v1) := by
    have equality_source := row_18
    have equality_step_1 := row_19
    cases equality_step_1
    have equality_step_2 := row_20
    cases equality_step_2
    exact equality_source
  -- chapter_0_line_16: GL tag task formulation.
  have row_16 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_0_line_15: GL tag expansion.
  have row_15 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_16)
  -- chapter_0_line_14: GL tag disintegration.
  have row_14 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_15.1
  -- chapter_0_line_13: GL tag expansion.
  have row_13 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_14)
  -- chapter_0_line_12: GL tag disintegration.
  have row_12 : (gl_fXYZ add N N N) := by
    exact row_13.1.1.1.1.1.1.1.1.1.2
  -- chapter_0_line_11: GL tag expansion.
  have row_11 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_12)
  -- chapter_0_line_10: GL tag disintegration.
  have row_10 : (gl_implication11 add N) := by
    exact row_11.1.1.1.1
  -- chapter_0_line_9: GL tag expansion.
  have row_9 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_10)
  -- chapter_0_line_8: GL tag disintegration.
  have row_8 : (gl_implication21 N zero add) := by
    exact row_13.1.1.1.1.1.1.1.1.2
  -- chapter_0_line_7: GL tag expansion.
  have row_7 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((add w1 zero w2) → (w1 = w2))))) := by
    simpa only [gl_implication21] using (row_8)
  -- chapter_0_line_6: GL tag implication.
  have row_6 : (N v2) := by
    apply row_9
    exact row_17
  -- chapter_0_line_5: GL tag implication.
  have row_5 : (v2 = v1) := by
    apply row_7
    exact row_6
    exact row_17
  -- chapter_0_line_4: GL tag equality2.
  have row_4 : (v2 = zero) := by
    exact Eq.trans row_5 row_19
  -- chapter_0_line_3: GL tag task formulation.
  have row_3 : (¬ (v1 = v2)) := by
    exact premise_1
  -- chapter_0_line_2: GL tag equality1.
  have row_2 : (¬ (v1 = zero)) := by
    have equality_source := row_3
    have equality_step_1 := row_4
    cases equality_step_1
    exact equality_source
  -- chapter_0_line_1: GL tag symmetry of inequality.
  have row_1 : (¬ (zero = v1)) := by
    exact fun equality => row_2 (Eq.symm equality)
  exact row_1 reductio

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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α), ((one = v1) → (gl_inSequence N v1 succ))) := by
  intro v1
  intro premise_1
  -- chapter_1_line_12: GL tag task formulation.
  have row_12 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_1_line_11: GL tag expansion.
  have row_11 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_12)
  -- chapter_1_line_10: GL tag disintegration.
  have row_10 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_11.1
  -- chapter_1_line_9: GL tag expansion.
  have row_9 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_10)
  -- chapter_1_line_8: GL tag disintegration.
  have row_8 : (N zero) := by
    exact row_9.1.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_1_line_7: GL tag disintegration.
  have row_7 : (succ zero one) := by
    exact row_11.2
  -- chapter_1_line_6: GL tag expansion for integration.
  have row_6 : ((gl_inSequence N one succ) ↔ (¬ (∀ (v3 : α), ((N v3) → (¬ (succ v3 one)))))) := by
    exact Iff.rfl
  -- chapter_1_line_5_rule: GL tag reformulation for integration >[bound].
  have row_5_rule : (∀ (cr5_1 : α), ((N cr5_1) → ((succ cr5_1 one) → (gl_inSequence N one succ)))) := by
    intro cr5_1
    intro integration_premise_1
    intro integration_premise_2
    apply (row_6).2
    intro universal_counterexample
    exact universal_counterexample cr5_1 integration_premise_1 integration_premise_2
  -- chapter_1_line_5: GL tag reformulation for integration >[bound].
  have row_5 : (gl_implication1873 N one succ) := by
    simpa only [gl_implication1873] using row_5_rule
  -- chapter_1_line_4: GL tag expansion.
  have row_4 : (∀ (v2 : α), ((N v2) → ((succ v2 one) → (gl_inSequence N one succ)))) := by
    simpa only [gl_implication1873] using (row_5)
  -- chapter_1_line_3: GL tag implication.
  have row_3 : (gl_inSequence N one succ) := by
    apply row_4
    exact row_8
    exact row_7
  -- chapter_1_line_2: GL tag task formulation.
  have row_2 : (one = v1) := by
    exact premise_1
  -- chapter_1_line_1: GL tag equality1.
  have row_1 : (gl_inSequence N v1 succ) := by
    have equality_source := row_3
    have equality_step_1 := row_2
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
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
  -- chapter_2_line_17: GL tag task formulation.
  have row_17 : (v1 = v3) := by
    exact premise_3
  -- chapter_2_line_16: GL tag symmetry of equality.
  have row_16 : (v3 = v1) := by
    exact Eq.symm row_17
  -- chapter_2_line_15: GL tag task formulation.
  have row_15 : (succ v3 v4) := by
    exact premise_2
  -- chapter_2_line_14: GL tag equality1.
  have row_14 : (succ v1 v4) := by
    have equality_source := row_15
    have equality_step_1 := row_16
    cases equality_step_1
    exact equality_source
  -- chapter_2_line_13: GL tag task formulation.
  have row_13 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_2_line_12: GL tag expansion.
  have row_12 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_13)
  -- chapter_2_line_11: GL tag disintegration.
  have row_11 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_12.1
  -- chapter_2_line_10: GL tag expansion.
  have row_10 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_11)
  -- chapter_2_line_9: GL tag disintegration.
  have row_9 : (gl_fXY succ N N) := by
    exact row_10.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_2_line_8: GL tag expansion.
  have row_8 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_9)
  -- chapter_2_line_7: GL tag disintegration.
  have row_7 : (gl_implication0 succ N) := by
    exact row_8.1.1.1
  -- chapter_2_line_6: GL tag expansion.
  have row_6 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_7)
  -- chapter_2_line_5: GL tag disintegration.
  have row_5 : (gl_implication7 N succ) := by
    exact row_8.2
  -- chapter_2_line_4: GL tag expansion.
  have row_4 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α), ((succ w1 w3) → (w2 = w3))))))) := by
    simpa only [gl_implication7] using (row_5)
  -- chapter_2_line_3: GL tag implication.
  have row_3 : (N v1) := by
    apply row_6
    exact row_14
  -- chapter_2_line_2: GL tag task formulation.
  have row_2 : (succ v1 v2) := by
    exact premise_1
  -- chapter_2_line_1: GL tag implication.
  have row_1 : (v2 = v4) := by
    apply row_4
    exact row_3
    exact row_2
    exact row_14
  exact row_1

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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
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
  -- chapter_3_line_19: GL tag task formulation.
  have row_19 : (v1 = v3) := by
    exact reductio
  -- chapter_3_line_18: GL tag symmetry of equality.
  have row_18 : (v3 = v1) := by
    exact Eq.symm row_19
  -- chapter_3_line_17: GL tag task formulation.
  have row_17 : (succ v3 v4) := by
    exact premise_2
  -- chapter_3_line_16: GL tag equality1.
  have row_16 : (succ v1 v4) := by
    have equality_source := row_17
    have equality_step_1 := row_18
    cases equality_step_1
    exact equality_source
  -- chapter_3_line_15: GL tag task formulation.
  have row_15 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_3_line_14: GL tag expansion.
  have row_14 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_15)
  -- chapter_3_line_13: GL tag disintegration.
  have row_13 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_14.1
  -- chapter_3_line_12: GL tag expansion.
  have row_12 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_13)
  -- chapter_3_line_11: GL tag disintegration.
  have row_11 : (gl_fXY succ N N) := by
    exact row_12.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_3_line_10: GL tag expansion.
  have row_10 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_11)
  -- chapter_3_line_9: GL tag disintegration.
  have row_9 : (gl_implication0 succ N) := by
    exact row_10.1.1.1
  -- chapter_3_line_8: GL tag expansion.
  have row_8 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_9)
  -- chapter_3_line_7: GL tag disintegration.
  have row_7 : (gl_implication7 N succ) := by
    exact row_10.2
  -- chapter_3_line_6: GL tag expansion.
  have row_6 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α), ((succ w1 w3) → (w2 = w3))))))) := by
    simpa only [gl_implication7] using (row_7)
  -- chapter_3_line_5: GL tag implication.
  have row_5 : (N v1) := by
    apply row_8
    exact row_16
  -- chapter_3_line_4: GL tag task formulation.
  have row_4 : (succ v1 v2) := by
    exact premise_1
  -- chapter_3_line_3: GL tag implication.
  have row_3 : (v2 = v4) := by
    apply row_6
    exact row_5
    exact row_4
    exact row_16
  -- chapter_3_line_2: GL tag task formulation.
  have row_2 : (¬ (v2 = v4)) := by
    exact premise_3
  -- chapter_3_line_1: GL tag contradiction.
  have row_1 : (¬ (v1 = v3)) := by
    exact False.elim (row_2 row_3)
  exact row_1 reductio

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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (v3 : α), ((succ v3 v2) → (v1 = v3))))) := by
  intro v1
  intro v2
  intro premise_1
  intro v3
  intro premise_2
  -- chapter_6_line_14: GL tag task formulation.
  have row_14 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_6_line_13: GL tag expansion.
  have row_13 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_14)
  -- chapter_6_line_12: GL tag disintegration.
  have row_12 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_13.1
  -- chapter_6_line_11: GL tag expansion.
  have row_11 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_12)
  -- chapter_6_line_10: GL tag disintegration.
  have row_10 : (gl_fXY succ N N) := by
    exact row_11.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_6_line_9: GL tag expansion.
  have row_9 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_10)
  -- chapter_6_line_8: GL tag disintegration.
  have row_8 : (gl_implication1 succ N) := by
    exact row_9.1.1.2
  -- chapter_6_line_7: GL tag expansion.
  have row_7 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using (row_8)
  -- chapter_6_line_6: GL tag disintegration.
  have row_6 : (gl_implication10 N succ) := by
    exact row_11.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_6_line_5: GL tag expansion.
  have row_5 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w2 w1) → (∀ (w3 : α), ((succ w3 w1) → (w2 = w3))))))) := by
    simpa only [gl_implication10] using (row_6)
  -- chapter_6_line_4: GL tag task formulation.
  have row_4 : (succ v3 v2) := by
    exact premise_2
  -- chapter_6_line_3: GL tag implication.
  have row_3 : (N v2) := by
    apply row_7
    exact row_4
  -- chapter_6_line_2: GL tag task formulation.
  have row_2 : (succ v1 v2) := by
    exact premise_1
  -- chapter_6_line_1: GL tag implication.
  have row_1 : (v1 = v3) := by
    apply row_5
    exact row_3
    exact row_2
    exact row_4
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
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
  -- chapter_8_line_15: GL tag task formulation.
  have row_15 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_8_line_14: GL tag expansion.
  have row_14 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_15)
  -- chapter_8_line_13: GL tag disintegration.
  have row_13 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_14.1
  -- chapter_8_line_12: GL tag expansion.
  have row_12 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_13)
  -- chapter_8_line_11: GL tag disintegration.
  have row_11 : (gl_fXY succ N N) := by
    exact row_12.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_8_line_10: GL tag expansion.
  have row_10 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_11)
  -- chapter_8_line_9: GL tag disintegration.
  have row_9 : (gl_implication0 succ N) := by
    exact row_10.1.1.1
  -- chapter_8_line_8: GL tag expansion.
  have row_8 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_9)
  -- chapter_8_line_7: GL tag disintegration.
  have row_7 : (gl_implication28 N succ mul add) := by
    exact row_12.2
  -- chapter_8_line_6: GL tag expansion.
  have row_6 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((mul w3 w1 w4) → (∀ (w5 : α), ((mul w3 w2 w5) → (add w4 w3 w5))))))))) := by
    simpa only [gl_implication28] using (row_7)
  -- chapter_8_line_5: GL tag task formulation.
  have row_5 : (succ v4 v2) := by
    exact premise_3
  -- chapter_8_line_4: GL tag implication.
  have row_4 : (N v4) := by
    apply row_8
    exact row_5
  -- chapter_8_line_3: GL tag task formulation.
  have row_3 : (mul v1 v2 v3) := by
    exact premise_1
  -- chapter_8_line_2: GL tag task formulation.
  have row_2 : (mul v1 v4 v5) := by
    exact premise_2
  -- chapter_8_line_1: GL tag implication.
  have row_1 : (add v5 v1 v3) := by
    apply row_6
    exact row_4
    exact row_5
    exact row_2
    exact row_3
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((v1 = v2) → ((mul v1 zero v2) → (zero = v1)))) := by
  intro v1
  intro v2
  intro premise_1
  intro premise_2
  -- chapter_9_line_17: GL tag task formulation.
  have row_17 : (v1 = v2) := by
    exact premise_1
  -- chapter_9_line_16: GL tag symmetry of equality.
  have row_16 : (v2 = v1) := by
    exact Eq.symm row_17
  -- chapter_9_line_15: GL tag task formulation.
  have row_15 : (mul v1 zero v2) := by
    exact premise_2
  -- chapter_9_line_14: GL tag equality1.
  have row_14 : (mul v2 zero v1) := by
    have equality_source := row_15
    have equality_step_1 := row_17
    cases equality_step_1
    have equality_step_2 := row_16
    cases equality_step_2
    exact equality_source
  -- chapter_9_line_13: GL tag task formulation.
  have row_13 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_9_line_12: GL tag expansion.
  have row_12 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_13)
  -- chapter_9_line_11: GL tag disintegration.
  have row_11 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_12.1
  -- chapter_9_line_10: GL tag expansion.
  have row_10 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_11)
  -- chapter_9_line_9: GL tag disintegration.
  have row_9 : (gl_fXYZ mul N N N) := by
    exact row_10.1.1.1.1.2
  -- chapter_9_line_8: GL tag expansion.
  have row_8 : (((((gl_implication11 mul N) ∧ (gl_implication12 mul N)) ∧ (gl_implication13 mul N)) ∧ (gl_implication18 N N N mul)) ∧ (gl_implication19 N N mul)) := by
    simpa only [gl_fXYZ] using (row_9)
  -- chapter_9_line_7: GL tag disintegration.
  have row_7 : (gl_implication11 mul N) := by
    exact row_8.1.1.1.1
  -- chapter_9_line_6: GL tag expansion.
  have row_6 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_7)
  -- chapter_9_line_5: GL tag disintegration.
  have row_5 : (gl_implication25 N zero mul) := by
    exact row_10.1.1.1.2
  -- chapter_9_line_4: GL tag expansion.
  have row_4 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((mul w1 zero w2) → (w2 = zero))))) := by
    simpa only [gl_implication25] using (row_5)
  -- chapter_9_line_3: GL tag implication.
  have row_3 : (N v2) := by
    apply row_6
    exact row_14
  -- chapter_9_line_2: GL tag implication.
  have row_2 : (v1 = zero) := by
    apply row_4
    exact row_3
    exact row_14
  -- chapter_9_line_1: GL tag symmetry of equality.
  have row_1 : (zero = v1) := by
    exact Eq.symm row_2
  exact row_1

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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (assumption_2 : (succ v1 v2))
    : (N v1) := by
  -- chapter_10_line_10: GL tag task formulation.
  have row_10 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_10_line_9: GL tag expansion.
  have row_9 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_10)
  -- chapter_10_line_8: GL tag disintegration.
  have row_8 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_9.1
  -- chapter_10_line_7: GL tag expansion.
  have row_7 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_8)
  -- chapter_10_line_6: GL tag disintegration.
  have row_6 : (gl_fXY succ N N) := by
    exact row_7.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_10_line_5: GL tag expansion.
  have row_5 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_6)
  -- chapter_10_line_4: GL tag disintegration.
  have row_4 : (gl_implication0 succ N) := by
    exact row_5.1.1.1
  -- chapter_10_line_3: GL tag expansion.
  have row_3 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_4)
  -- chapter_10_line_2: GL tag task formulation.
  have row_2 : (succ v1 v2) := by
    exact assumption_2
  -- chapter_10_line_1: GL tag implication.
  have row_1 : (N v1) := by
    apply row_3
    exact row_2
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (assumption_40 : (v1 = zero))
    (assumption_39 : (succ v1 v2))
    : (add one v1 v2) := by
  -- chapter_11_line_40: GL tag recursion.
  have row_40 : (v1 = zero) := by
    exact assumption_40
  -- chapter_11_line_39: GL tag task formulation.
  have row_39 : (succ v1 v2) := by
    exact assumption_39
  -- chapter_11_line_38: GL tag equality1.
  have row_38 : (succ zero v2) := by
    have equality_source := row_39
    have equality_step_1 := row_40
    cases equality_step_1
    exact equality_source
  -- chapter_11_line_37: GL tag task formulation.
  have row_37 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_11_line_36: GL tag expansion.
  have row_36 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_37)
  -- chapter_11_line_35: GL tag disintegration.
  have row_35 : (succ zero one) := by
    exact row_36.2
  -- chapter_11_line_34: GL tag disintegration.
  have row_34 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_36.1
  -- chapter_11_line_33: GL tag expansion.
  have row_33 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_34)
  -- chapter_11_line_32: GL tag disintegration.
  have row_32 : (gl_fXY succ N N) := by
    exact row_33.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_11_line_31: GL tag expansion.
  have row_31 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_32)
  -- chapter_11_line_30: GL tag disintegration.
  have row_30 : (gl_implication7 N succ) := by
    exact row_31.2
  -- chapter_11_line_29: GL tag expansion.
  have row_29 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α), ((succ w1 w3) → (w2 = w3))))))) := by
    simpa only [gl_implication7] using (row_30)
  -- chapter_11_line_28: GL tag disintegration.
  have row_28 : (N zero) := by
    exact row_33.1.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_11_line_27: GL tag disintegration.
  have row_27 : (gl_implication1 succ N) := by
    exact row_31.1.1.2
  -- chapter_11_line_26: GL tag expansion.
  have row_26 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using (row_27)
  -- chapter_11_line_25: GL tag disintegration.
  have row_25 : (gl_implication0 succ N) := by
    exact row_31.1.1.1
  -- chapter_11_line_24: GL tag expansion.
  have row_24 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_25)
  -- chapter_11_line_23: GL tag disintegration.
  have row_23 : (gl_fXYZ add N N N) := by
    exact row_33.1.1.1.1.1.1.1.1.1.2
  -- chapter_11_line_22: GL tag expansion.
  have row_22 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_23)
  -- chapter_11_line_21: GL tag disintegration.
  have row_21 : (gl_implication18 N N N add) := by
    exact row_22.1.2
  -- chapter_11_line_20: GL tag expansion.
  have row_20 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication18] using (row_21)
  -- chapter_11_line_19: GL tag disintegration.
  have row_19 : (gl_implication21 N zero add) := by
    exact row_33.1.1.1.1.1.1.1.1.2
  -- chapter_11_line_18: GL tag expansion.
  have row_18 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((add w1 zero w2) → (w1 = w2))))) := by
    simpa only [gl_implication21] using (row_19)
  -- chapter_11_line_17: GL tag implication.
  have row_17 : (one = v2) := by
    apply row_29
    exact row_28
    exact row_35
    exact row_38
  -- chapter_11_line_16: GL tag implication grouping.
  have row_16 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (gl_and20 w1 N w2 N))) := by
    have auxiliary_source_1 := row_24
    have auxiliary_source_2 := row_26
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_premise
    exact ⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_premise)⟩
  -- chapter_11_line_15: GL tag implication.
  have row_15 : (gl_and20 v1 N v2 N) := by
    apply row_16
    exact row_39
  -- chapter_11_line_14: GL tag expansion.
  have row_14 : ((N v1) ∧ (N v2)) := by
    simpa only [gl_and20] using (row_15)
  -- chapter_11_line_13: GL tag disintegration.
  have row_13 : (N v1) := by
    exact row_14.1
  -- chapter_11_line_12: GL tag implication.
  have row_12 : (gl_and20 zero N one N) := by
    apply row_16
    exact row_35
  -- chapter_11_line_11: GL tag expansion.
  have row_11 : ((N zero) ∧ (N one)) := by
    simpa only [gl_and20] using (row_12)
  -- chapter_11_line_10: GL tag disintegration.
  have row_10 : (N one) := by
    exact row_11.2
  -- chapter_11_line_9: GL tag implication.
  have row_9 : (gl_existence1 N one v1 add) := by
    apply row_20
    exact row_10
    exact row_13
  -- chapter_11_line_8: GL tag expansion.
  have row_8 : (¬ (∀ (v3 : α), ((N v3) → (¬ (add one v1 v3))))) := by
    simpa only [gl_existence1] using (row_9)
  have exists_row_8 : ∃ (v3 : α), ((N v3) ∧ (add one v1 v3)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_8
      intro v3 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v3, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v3, witness_row_8⟩ := exists_row_8
  -- chapter_11_line_7: GL tag disintegration.
  have row_7 : (add one v1 v3) := by
    exact witness_row_8.2
  -- chapter_11_line_6: GL tag equality1.
  have row_6 : (add one zero v3) := by
    have equality_source := row_7
    have equality_step_1 := row_40
    cases equality_step_1
    exact equality_source
  -- chapter_11_line_5: GL tag implication.
  have row_5 : (one = v3) := by
    apply row_18
    exact row_10
    exact row_6
  -- chapter_11_line_4: GL tag symmetry of equality.
  have row_4 : (v3 = one) := by
    exact Eq.symm row_5
  -- chapter_11_line_3: GL tag equality2.
  have row_3 : (v3 = v2) := by
    exact Eq.trans row_4 row_17
  -- chapter_11_line_2: GL tag symmetry of equality.
  have row_2 : (zero = v1) := by
    exact Eq.symm row_40
  -- chapter_11_line_1: GL tag equality1.
  have row_1 : (add one v1 v2) := by
    have equality_source := row_6
    have equality_step_1 := row_2
    cases equality_step_1
    have equality_step_2 := row_3
    cases equality_step_2
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (assumption_27 : (succ previous v1))
    (assumption_26 : (∀ (w1 : α), ((succ previous w1) → (add one previous w1))))
    (assumption_3 : (succ v1 v2))
    : (add one v1 v2) := by
  -- chapter_12_line_27: GL tag recursion.
  have row_27 : (succ previous v1) := by
    exact assumption_27
  -- chapter_12_line_26: GL tag recursion.
  have row_26 : (∀ (w1 : α), ((succ previous w1) → (add one previous w1))) := by
    exact assumption_26
  -- chapter_12_line_25: GL tag implication.
  have row_25 : (add one previous v1) := by
    apply row_26
    exact row_27
  -- chapter_12_line_24: GL tag task formulation.
  have row_24 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_12_line_23: GL tag expansion.
  have row_23 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_24)
  -- chapter_12_line_22: GL tag disintegration.
  have row_22 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_23.1
  -- chapter_12_line_21: GL tag expansion.
  have row_21 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_22)
  -- chapter_12_line_20: GL tag disintegration.
  have row_20 : (gl_fXYZ add N N N) := by
    exact row_21.1.1.1.1.1.1.1.1.1.2
  -- chapter_12_line_19: GL tag expansion.
  have row_19 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_20)
  -- chapter_12_line_18: GL tag disintegration.
  have row_18 : (gl_implication12 add N) := by
    exact row_19.1.1.1.2
  -- chapter_12_line_17: GL tag expansion.
  have row_17 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_18)
  -- chapter_12_line_16: GL tag disintegration.
  have row_16 : (gl_fXY succ N N) := by
    exact row_21.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_12_line_15: GL tag expansion.
  have row_15 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_16)
  -- chapter_12_line_14: GL tag disintegration.
  have row_14 : (gl_implication6 N N succ) := by
    exact row_15.1.2
  -- chapter_12_line_13: GL tag expansion.
  have row_13 : (∀ (w1 : α), ((N w1) → (gl_existence0 N w1 succ))) := by
    simpa only [gl_implication6] using (row_14)
  -- chapter_12_line_12: GL tag disintegration.
  have row_12 : (gl_implication7 N succ) := by
    exact row_15.2
  -- chapter_12_line_11: GL tag expansion.
  have row_11 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α), ((succ w1 w3) → (w2 = w3))))))) := by
    simpa only [gl_implication7] using (row_12)
  -- chapter_12_line_10: GL tag disintegration.
  have row_10 : (gl_implication24 N succ add) := by
    exact row_21.1.1.1.1.1.2
  -- chapter_12_line_9: GL tag expansion.
  have row_9 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((succ w4 w5) → (add w3 w2 w5))))))))) := by
    simpa only [gl_implication24] using (row_10)
  -- chapter_12_line_8: GL tag implication.
  have row_8 : (N previous) := by
    apply row_17
    exact row_25
  -- chapter_12_line_7: GL tag implication.
  have row_7 : (gl_existence0 N previous succ) := by
    apply row_13
    exact row_8
  -- chapter_12_line_6: GL tag expansion.
  have row_6 : (¬ (∀ (v3 : α), ((N v3) → (¬ (succ previous v3))))) := by
    simpa only [gl_existence0] using (row_7)
  have exists_row_6 : ∃ (v3 : α), ((N v3) ∧ (succ previous v3)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_6
      intro v3 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v3, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v3, witness_row_6⟩ := exists_row_6
  -- chapter_12_line_5: GL tag disintegration.
  have row_5 : (succ previous v3) := by
    exact witness_row_6.2
  -- chapter_12_line_4: GL tag implication.
  have row_4 : (v3 = v1) := by
    apply row_11
    exact row_8
    exact row_5
    exact row_27
  -- chapter_12_line_3: GL tag task formulation.
  have row_3 : (succ v1 v2) := by
    exact assumption_3
  -- chapter_12_line_2: GL tag implication.
  have row_2 : (add one v3 v2) := by
    apply row_9
    exact row_8
    exact row_5
    exact row_25
    exact row_3
  -- chapter_12_line_1: GL tag equality1.
  have row_1 : (add one v1 v2) := by
    have equality_source := row_2
    have equality_step_1 := row_4
    cases equality_step_1
    exact equality_source
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (add one v1 v2))) := by
  intro v1
  intro v2
  intro premise_1
  have inductionMember : N v1 := by
    have typingRule := peano_source_010_induction_typing N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact typingRule v1 v2 premise_1
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v2 : α), ((succ zero v2) → (add one zero v2))) := by
    intro v2
    intro base_premise_1
    have zeroRule := peano_source_010_check_zero N zero succ add mul one anchor relationalInduction strongRelationalInduction
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
    have stepRule := peano_source_010_check_induction_condition N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact stepRule induction_n induction_m v2 step_induction_assumption_1 step_induction_assumption_2 step_premise_1
  have inductionProperty : (∀ (v2 : α), ((succ v1 v2) → (add one v1 v2))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v2 : α), ((succ induction_value v2) → (add one induction_value v2))))
      v1
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v2 premise_1

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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((add v1 zero v2) → ((N v1) → (v1 = v2)))) := by
  intro v1
  intro v2
  intro premise_1
  intro premise_2
  -- chapter_13_line_9: GL tag task formulation.
  have row_9 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_13_line_8: GL tag expansion.
  have row_8 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_9)
  -- chapter_13_line_7: GL tag disintegration.
  have row_7 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_8.1
  -- chapter_13_line_6: GL tag expansion.
  have row_6 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_7)
  -- chapter_13_line_5: GL tag disintegration.
  have row_5 : (gl_implication21 N zero add) := by
    exact row_6.1.1.1.1.1.1.1.1.2
  -- chapter_13_line_4: GL tag expansion.
  have row_4 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((add w1 zero w2) → (w1 = w2))))) := by
    simpa only [gl_implication21] using (row_5)
  -- chapter_13_line_3: GL tag task formulation.
  have row_3 : (add v1 zero v2) := by
    exact premise_1
  -- chapter_13_line_2: GL tag task formulation.
  have row_2 : (N v1) := by
    exact premise_2
  -- chapter_13_line_1: GL tag implication.
  have row_1 : (v1 = v2) := by
    apply row_4
    exact row_2
    exact row_3
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
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
  -- chapter_25_line_15: GL tag task formulation.
  have row_15 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_25_line_14: GL tag expansion.
  have row_14 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_15)
  -- chapter_25_line_13: GL tag disintegration.
  have row_13 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_14.1
  -- chapter_25_line_12: GL tag expansion.
  have row_12 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_13)
  -- chapter_25_line_11: GL tag disintegration.
  have row_11 : (gl_fXY succ N N) := by
    exact row_12.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_25_line_10: GL tag expansion.
  have row_10 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_11)
  -- chapter_25_line_9: GL tag disintegration.
  have row_9 : (gl_implication0 succ N) := by
    exact row_10.1.1.1
  -- chapter_25_line_8: GL tag expansion.
  have row_8 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_9)
  -- chapter_25_line_7: GL tag disintegration.
  have row_7 : (gl_implication23 N succ add) := by
    exact row_12.1.1.1.1.1.1.2
  -- chapter_25_line_6: GL tag expansion.
  have row_6 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication23] using (row_7)
  -- chapter_25_line_5: GL tag task formulation.
  have row_5 : (succ v4 v2) := by
    exact premise_3
  -- chapter_25_line_4: GL tag implication.
  have row_4 : (N v4) := by
    apply row_8
    exact row_5
  -- chapter_25_line_3: GL tag task formulation.
  have row_3 : (add v1 v2 v3) := by
    exact premise_1
  -- chapter_25_line_2: GL tag task formulation.
  have row_2 : (add v1 v4 v5) := by
    exact premise_2
  -- chapter_25_line_1: GL tag implication.
  have row_1 : (succ v5 v3) := by
    apply row_6
    exact row_4
    exact row_5
    exact row_2
    exact row_3
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
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
  -- chapter_26_line_22: GL tag task formulation.
  have row_22 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_26_line_21: GL tag expansion.
  have row_21 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_22)
  -- chapter_26_line_20: GL tag disintegration.
  have row_20 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_21.1
  -- chapter_26_line_19: GL tag expansion.
  have row_19 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_20)
  -- chapter_26_line_18: GL tag disintegration.
  have row_18 : (gl_fXYZ add N N N) := by
    exact row_19.1.1.1.1.1.1.1.1.1.2
  -- chapter_26_line_17: GL tag expansion.
  have row_17 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_18)
  -- chapter_26_line_16: GL tag disintegration.
  have row_16 : (gl_implication13 add N) := by
    exact row_17.1.1.2
  -- chapter_26_line_15: GL tag expansion.
  have row_15 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_16)
  -- chapter_26_line_14: GL tag disintegration.
  have row_14 : (gl_implication12 add N) := by
    exact row_17.1.1.1.2
  -- chapter_26_line_13: GL tag expansion.
  have row_13 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_14)
  -- chapter_26_line_12: GL tag disintegration.
  have row_12 : (gl_implication11 add N) := by
    exact row_17.1.1.1.1
  -- chapter_26_line_11: GL tag expansion.
  have row_11 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_12)
  -- chapter_26_line_10: GL tag disintegration.
  have row_10 : (gl_implication24 N succ add) := by
    exact row_19.1.1.1.1.1.2
  -- chapter_26_line_9: GL tag expansion.
  have row_9 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((succ w4 w5) → (add w3 w2 w5))))))))) := by
    simpa only [gl_implication24] using (row_10)
  -- chapter_26_line_8: GL tag implication grouping.
  have row_8 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_11
    have auxiliary_source_2 := row_13
    have auxiliary_source_3 := row_15
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_26_line_7: GL tag task formulation.
  have row_7 : (add v1 v2 v3) := by
    exact premise_1
  -- chapter_26_line_6: GL tag implication.
  have row_6 : (gl_and21 v1 N v2 N v3 N) := by
    apply row_8
    exact row_7
  -- chapter_26_line_5: GL tag expansion.
  have row_5 : (((N v1) ∧ (N v2)) ∧ (N v3)) := by
    simpa only [gl_and21] using (row_6)
  -- chapter_26_line_4: GL tag disintegration.
  have row_4 : (N v2) := by
    exact row_5.1.2
  -- chapter_26_line_3: GL tag task formulation.
  have row_3 : (succ v2 v4) := by
    exact premise_2
  -- chapter_26_line_2: GL tag task formulation.
  have row_2 : (succ v3 v5) := by
    exact premise_3
  -- chapter_26_line_1: GL tag implication.
  have row_1 : (add v1 v4 v5) := by
    apply row_9
    exact row_4
    exact row_3
    exact row_7
    exact row_2
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
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
  -- chapter_40_line_15: GL tag task formulation.
  have row_15 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_40_line_14: GL tag expansion.
  have row_14 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_15)
  -- chapter_40_line_13: GL tag disintegration.
  have row_13 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_14.1
  -- chapter_40_line_12: GL tag expansion.
  have row_12 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_13)
  -- chapter_40_line_11: GL tag disintegration.
  have row_11 : (gl_fXYZ mul N N N) := by
    exact row_12.1.1.1.1.2
  -- chapter_40_line_10: GL tag expansion.
  have row_10 : (((((gl_implication11 mul N) ∧ (gl_implication12 mul N)) ∧ (gl_implication13 mul N)) ∧ (gl_implication18 N N N mul)) ∧ (gl_implication19 N N mul)) := by
    simpa only [gl_fXYZ] using (row_11)
  -- chapter_40_line_9: GL tag disintegration.
  have row_9 : (gl_implication12 mul N) := by
    exact row_10.1.1.1.2
  -- chapter_40_line_8: GL tag expansion.
  have row_8 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_9)
  -- chapter_40_line_7: GL tag disintegration.
  have row_7 : (gl_implication27 N succ mul add) := by
    exact row_12.1.2
  -- chapter_40_line_6: GL tag expansion.
  have row_6 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((mul w3 w1 w4) → (∀ (w5 : α), ((add w4 w3 w5) → (mul w3 w2 w5))))))))) := by
    simpa only [gl_implication27] using (row_7)
  -- chapter_40_line_5: GL tag task formulation.
  have row_5 : (mul v2 v4 v1) := by
    exact premise_3
  -- chapter_40_line_4: GL tag implication.
  have row_4 : (N v4) := by
    apply row_8
    exact row_5
  -- chapter_40_line_3: GL tag task formulation.
  have row_3 : (add v1 v2 v3) := by
    exact premise_1
  -- chapter_40_line_2: GL tag task formulation.
  have row_2 : (succ v4 v5) := by
    exact premise_2
  -- chapter_40_line_1: GL tag implication.
  have row_1 : (mul v2 v5 v3) := by
    apply row_6
    exact row_4
    exact row_2
    exact row_5
    exact row_3
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((mul v1 zero v2) → ((N v1) → (zero = v2)))) := by
  intro v1
  intro v2
  intro premise_1
  intro premise_2
  -- chapter_41_line_10: GL tag task formulation.
  have row_10 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_41_line_9: GL tag expansion.
  have row_9 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_10)
  -- chapter_41_line_8: GL tag disintegration.
  have row_8 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_9.1
  -- chapter_41_line_7: GL tag expansion.
  have row_7 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_8)
  -- chapter_41_line_6: GL tag disintegration.
  have row_6 : (gl_implication25 N zero mul) := by
    exact row_7.1.1.1.2
  -- chapter_41_line_5: GL tag expansion.
  have row_5 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((mul w1 zero w2) → (w2 = zero))))) := by
    simpa only [gl_implication25] using (row_6)
  -- chapter_41_line_4: GL tag task formulation.
  have row_4 : (mul v1 zero v2) := by
    exact premise_1
  -- chapter_41_line_3: GL tag task formulation.
  have row_3 : (N v1) := by
    exact premise_2
  -- chapter_41_line_2: GL tag implication.
  have row_2 : (v2 = zero) := by
    apply row_5
    exact row_3
    exact row_4
  -- chapter_41_line_1: GL tag symmetry of equality.
  have row_1 : (zero = v2) := by
    exact Eq.symm row_2
  exact row_1

private theorem peano_source_029_induction_typing
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (assumption_4 : (add v1 v2 zero))
    : (N v2) := by
  -- chapter_49_line_17: GL tag task formulation.
  have row_17 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_49_line_16: GL tag expansion.
  have row_16 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_17)
  -- chapter_49_line_15: GL tag disintegration.
  have row_15 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_16.1
  -- chapter_49_line_14: GL tag expansion.
  have row_14 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_15)
  -- chapter_49_line_13: GL tag disintegration.
  have row_13 : (gl_fXYZ add N N N) := by
    exact row_14.1.1.1.1.1.1.1.1.1.2
  -- chapter_49_line_12: GL tag expansion.
  have row_12 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_13)
  -- chapter_49_line_11: GL tag disintegration.
  have row_11 : (gl_implication13 add N) := by
    exact row_12.1.1.2
  -- chapter_49_line_10: GL tag expansion.
  have row_10 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_11)
  -- chapter_49_line_9: GL tag disintegration.
  have row_9 : (gl_implication12 add N) := by
    exact row_12.1.1.1.2
  -- chapter_49_line_8: GL tag expansion.
  have row_8 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_9)
  -- chapter_49_line_7: GL tag disintegration.
  have row_7 : (gl_implication11 add N) := by
    exact row_12.1.1.1.1
  -- chapter_49_line_6: GL tag expansion.
  have row_6 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_7)
  -- chapter_49_line_5: GL tag implication grouping.
  have row_5 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_6
    have auxiliary_source_2 := row_8
    have auxiliary_source_3 := row_10
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_49_line_4: GL tag task formulation.
  have row_4 : (add v1 v2 zero) := by
    exact assumption_4
  -- chapter_49_line_3: GL tag implication.
  have row_3 : (gl_and21 v1 N v2 N zero N) := by
    apply row_5
    exact row_4
  -- chapter_49_line_2: GL tag expansion.
  have row_2 : (((N v1) ∧ (N v2)) ∧ (N zero)) := by
    simpa only [gl_and21] using (row_3)
  -- chapter_49_line_1: GL tag disintegration.
  have row_1 : (N v2) := by
    exact row_2.1.2
  exact row_1

private theorem peano_source_029_check_zero
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v2 : α)
    (assumption_2 : (v2 = zero))
    : (zero = v2) := by
  -- chapter_50_line_2: GL tag recursion.
  have row_2 : (v2 = zero) := by
    exact assumption_2
  -- chapter_50_line_1: GL tag symmetry of equality.
  have row_1 : (zero = v2) := by
    exact Eq.symm row_2
  exact row_1

private theorem peano_source_029_check_induction_condition
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (assumption_11 : (succ previous v2))
    (assumption_9 : (N v1))
    (assumption_4 : (add v1 v2 zero))
    : (zero = v2) := by
  -- chapter_51_line_27: GL tag task formulation.
  have row_27 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_51_line_26: GL tag expansion.
  have row_26 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_27)
  -- chapter_51_line_25: GL tag disintegration.
  have row_25 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_26.1
  -- chapter_51_line_24: GL tag expansion.
  have row_24 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_25)
  -- chapter_51_line_23: GL tag disintegration.
  have row_23 : (gl_implication23 N succ add) := by
    exact row_24.1.1.1.1.1.1.2
  -- chapter_51_line_22: GL tag expansion.
  have row_22 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication23] using (row_23)
  -- chapter_51_line_21: GL tag disintegration.
  have row_21 : (gl_fXY succ N N) := by
    exact row_24.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_51_line_20: GL tag expansion.
  have row_20 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_21)
  -- chapter_51_line_19: GL tag disintegration.
  have row_19 : (gl_implication0 succ N) := by
    exact row_20.1.1.1
  -- chapter_51_line_18: GL tag expansion.
  have row_18 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_19)
  -- chapter_51_line_17: GL tag disintegration.
  have row_17 : (gl_fXYZ add N N N) := by
    exact row_24.1.1.1.1.1.1.1.1.1.2
  -- chapter_51_line_16: GL tag expansion.
  have row_16 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_17)
  -- chapter_51_line_15: GL tag disintegration.
  have row_15 : (gl_implication18 N N N add) := by
    exact row_16.1.2
  -- chapter_51_line_14: GL tag expansion.
  have row_14 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication18] using (row_15)
  -- chapter_51_line_13: GL tag disintegration.
  have row_13 : (gl_implication9 N zero succ) := by
    exact row_24.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_51_line_12: GL tag expansion.
  have row_12 : (∀ (w1 : α), ((N w1) → (¬ (succ w1 zero)))) := by
    simpa only [gl_implication9] using (row_13)
  -- chapter_51_line_11: GL tag recursion.
  have row_11 : (succ previous v2) := by
    exact assumption_11
  -- chapter_51_line_10: GL tag implication.
  have row_10 : (N previous) := by
    apply row_18
    exact row_11
  -- chapter_51_line_9: GL tag task formulation.
  have row_9 : (N v1) := by
    exact assumption_9
  -- chapter_51_line_8: GL tag implication.
  have row_8 : (gl_existence1 N v1 previous add) := by
    apply row_14
    exact row_9
    exact row_10
  -- chapter_51_line_7: GL tag expansion.
  have row_7 : (¬ (∀ (v3 : α), ((N v3) → (¬ (add v1 previous v3))))) := by
    simpa only [gl_existence1] using (row_8)
  have exists_row_7 : ∃ (v3 : α), ((N v3) ∧ (add v1 previous v3)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_7
      intro v3 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v3, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v3, witness_row_7⟩ := exists_row_7
  -- chapter_51_line_6: GL tag disintegration.
  have row_6 : (add v1 previous v3) := by
    exact witness_row_7.2
  -- chapter_51_line_5: GL tag disintegration.
  have row_5 : (N v3) := by
    exact witness_row_7.1
  -- chapter_51_line_4: GL tag task formulation.
  have row_4 : (add v1 v2 zero) := by
    exact assumption_4
  -- chapter_51_line_3: GL tag implication.
  have row_3 : (succ v3 zero) := by
    apply row_22
    exact row_10
    exact row_11
    exact row_6
    exact row_4
  -- chapter_51_line_2: GL tag implication.
  have row_2 : (¬ (succ v3 zero)) := by
    apply row_12
    exact row_5
  -- chapter_51_line_1: GL tag vacuous truth.
  have row_1 : (zero = v2) := by
    exact False.elim (row_2 row_3)
  exact row_1

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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((add v1 v2 zero) → ((N v1) → (zero = v2)))) := by
  intro v1
  intro v2
  intro premise_1
  intro premise_2
  have inductionMember : N v2 := by
    have typingRule := peano_source_029_induction_typing N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact typingRule v1 v2 premise_1
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v1 : α), ((add v1 zero zero) → ((N v1) → (zero = zero)))) := by
    intro v1
    intro base_premise_1
    intro base_premise_2
    have zeroRule := peano_source_029_check_zero N zero succ add mul one anchor relationalInduction strongRelationalInduction
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
    have stepRule := peano_source_029_check_induction_condition N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact stepRule induction_n v1 induction_m step_induction_assumption_1 step_premise_2 step_premise_1
  have inductionProperty : (∀ (v1 : α), ((add v1 v2 zero) → ((N v1) → (zero = v2)))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v1 : α), ((add v1 induction_value zero) → ((N v1) → (zero = induction_value)))))
      v2
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v1 premise_1 premise_2

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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → ((succ zero v2) → (zero = v1)))) := by
  intro v1
  intro v2
  intro premise_1
  intro premise_2
  -- chapter_86_line_25: GL tag task formulation.
  have row_25 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_86_line_24: GL tag expansion.
  have row_24 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_25)
  -- chapter_86_line_23: GL tag disintegration.
  have row_23 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_24.1
  -- chapter_86_line_22: GL tag expansion.
  have row_22 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_23)
  -- chapter_86_line_21: GL tag disintegration.
  have row_21 : (gl_fXY succ N N) := by
    exact row_22.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_86_line_20: GL tag expansion.
  have row_20 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_21)
  -- chapter_86_line_19: GL tag disintegration.
  have row_19 : (gl_implication1 succ N) := by
    exact row_20.1.1.2
  -- chapter_86_line_18: GL tag expansion.
  have row_18 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using (row_19)
  -- chapter_86_line_17: GL tag disintegration.
  have row_17 : (gl_implication0 succ N) := by
    exact row_20.1.1.1
  -- chapter_86_line_16: GL tag expansion.
  have row_16 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_17)
  -- chapter_86_line_15: GL tag disintegration.
  have row_15 : (succ zero one) := by
    exact row_24.2
  -- chapter_86_line_14: GL tag disintegration.
  have row_14 : (gl_implication7 N succ) := by
    exact row_20.2
  -- chapter_86_line_13: GL tag expansion.
  have row_13 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α), ((succ w1 w3) → (w2 = w3))))))) := by
    simpa only [gl_implication7] using (row_14)
  -- chapter_86_line_12: GL tag disintegration.
  have row_12 : (N zero) := by
    exact row_22.1.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_86_line_11: GL tag disintegration.
  have row_11 : (gl_implication10 N succ) := by
    exact row_22.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_86_line_10: GL tag expansion.
  have row_10 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w2 w1) → (∀ (w3 : α), ((succ w3 w1) → (w2 = w3))))))) := by
    simpa only [gl_implication10] using (row_11)
  -- chapter_86_line_9: GL tag implication grouping.
  have row_9 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (gl_and20 w1 N w2 N))) := by
    have auxiliary_source_1 := row_16
    have auxiliary_source_2 := row_18
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_premise
    exact ⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_premise)⟩
  -- chapter_86_line_8: GL tag implication.
  have row_8 : (gl_and20 zero N one N) := by
    apply row_9
    exact row_15
  -- chapter_86_line_7: GL tag task formulation.
  have row_7 : (succ zero v2) := by
    exact premise_2
  -- chapter_86_line_6: GL tag implication.
  have row_6 : (v2 = one) := by
    apply row_13
    exact row_12
    exact row_7
    exact row_15
  -- chapter_86_line_5: GL tag task formulation.
  have row_5 : (succ v1 v2) := by
    exact premise_1
  -- chapter_86_line_4: GL tag equality1.
  have row_4 : (succ v1 one) := by
    have equality_source := row_5
    have equality_step_1 := row_6
    cases equality_step_1
    exact equality_source
  -- chapter_86_line_3: GL tag expansion.
  have row_3 : ((N zero) ∧ (N one)) := by
    simpa only [gl_and20] using (row_8)
  -- chapter_86_line_2: GL tag disintegration.
  have row_2 : (N one) := by
    exact row_3.2
  -- chapter_86_line_1: GL tag implication.
  have row_1 : (zero = v1) := by
    apply row_10
    exact row_2
    exact row_15
    exact row_4
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (assumption_4 : (add v1 v2 zero))
    : (N v2) := by
  -- chapter_87_line_17: GL tag task formulation.
  have row_17 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_87_line_16: GL tag expansion.
  have row_16 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_17)
  -- chapter_87_line_15: GL tag disintegration.
  have row_15 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_16.1
  -- chapter_87_line_14: GL tag expansion.
  have row_14 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_15)
  -- chapter_87_line_13: GL tag disintegration.
  have row_13 : (gl_fXYZ add N N N) := by
    exact row_14.1.1.1.1.1.1.1.1.1.2
  -- chapter_87_line_12: GL tag expansion.
  have row_12 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_13)
  -- chapter_87_line_11: GL tag disintegration.
  have row_11 : (gl_implication13 add N) := by
    exact row_12.1.1.2
  -- chapter_87_line_10: GL tag expansion.
  have row_10 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_11)
  -- chapter_87_line_9: GL tag disintegration.
  have row_9 : (gl_implication12 add N) := by
    exact row_12.1.1.1.2
  -- chapter_87_line_8: GL tag expansion.
  have row_8 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_9)
  -- chapter_87_line_7: GL tag disintegration.
  have row_7 : (gl_implication11 add N) := by
    exact row_12.1.1.1.1
  -- chapter_87_line_6: GL tag expansion.
  have row_6 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_7)
  -- chapter_87_line_5: GL tag implication grouping.
  have row_5 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_6
    have auxiliary_source_2 := row_8
    have auxiliary_source_3 := row_10
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_87_line_4: GL tag task formulation.
  have row_4 : (add v1 v2 zero) := by
    exact assumption_4
  -- chapter_87_line_3: GL tag implication.
  have row_3 : (gl_and21 v1 N v2 N zero N) := by
    apply row_5
    exact row_4
  -- chapter_87_line_2: GL tag expansion.
  have row_2 : (((N v1) ∧ (N v2)) ∧ (N zero)) := by
    simpa only [gl_and21] using (row_3)
  -- chapter_87_line_1: GL tag disintegration.
  have row_1 : (N v2) := by
    exact row_2.1.2
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (assumption_23 : (add v1 v2 zero))
    (assumption_22 : (v2 = zero))
    : (v1 = v2) := by
  -- chapter_88_line_23: GL tag task formulation.
  have row_23 : (add v1 v2 zero) := by
    exact assumption_23
  -- chapter_88_line_22: GL tag recursion.
  have row_22 : (v2 = zero) := by
    exact assumption_22
  -- chapter_88_line_21: GL tag symmetry of equality.
  have row_21 : (zero = v2) := by
    exact Eq.symm row_22
  -- chapter_88_line_20: GL tag equality1.
  have row_20 : (add v1 zero v2) := by
    have equality_source := row_23
    have equality_step_1 := row_22
    cases equality_step_1
    have equality_step_2 := row_21
    cases equality_step_2
    exact equality_source
  -- chapter_88_line_19: GL tag task formulation.
  have row_19 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_88_line_18: GL tag expansion.
  have row_18 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_19)
  -- chapter_88_line_17: GL tag disintegration.
  have row_17 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_18.1
  -- chapter_88_line_16: GL tag expansion.
  have row_16 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_17)
  -- chapter_88_line_15: GL tag disintegration.
  have row_15 : (gl_fXYZ add N N N) := by
    exact row_16.1.1.1.1.1.1.1.1.1.2
  -- chapter_88_line_14: GL tag expansion.
  have row_14 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_15)
  -- chapter_88_line_13: GL tag disintegration.
  have row_13 : (gl_implication13 add N) := by
    exact row_14.1.1.2
  -- chapter_88_line_12: GL tag expansion.
  have row_12 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_13)
  -- chapter_88_line_11: GL tag disintegration.
  have row_11 : (gl_implication12 add N) := by
    exact row_14.1.1.1.2
  -- chapter_88_line_10: GL tag expansion.
  have row_10 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_11)
  -- chapter_88_line_9: GL tag disintegration.
  have row_9 : (gl_implication11 add N) := by
    exact row_14.1.1.1.1
  -- chapter_88_line_8: GL tag expansion.
  have row_8 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_9)
  -- chapter_88_line_7: GL tag disintegration.
  have row_7 : (gl_implication21 N zero add) := by
    exact row_16.1.1.1.1.1.1.1.1.2
  -- chapter_88_line_6: GL tag expansion.
  have row_6 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((add w1 zero w2) → (w1 = w2))))) := by
    simpa only [gl_implication21] using (row_7)
  -- chapter_88_line_5: GL tag implication grouping.
  have row_5 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_8
    have auxiliary_source_2 := row_10
    have auxiliary_source_3 := row_12
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_88_line_4: GL tag implication.
  have row_4 : (gl_and21 v1 N v2 N zero N) := by
    apply row_5
    exact row_23
  -- chapter_88_line_3: GL tag expansion.
  have row_3 : (((N v1) ∧ (N v2)) ∧ (N zero)) := by
    simpa only [gl_and21] using (row_4)
  -- chapter_88_line_2: GL tag disintegration.
  have row_2 : (N v1) := by
    exact row_3.1.1
  -- chapter_88_line_1: GL tag implication.
  have row_1 : (v1 = v2) := by
    apply row_6
    exact row_2
    exact row_20
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (assumption_11 : (succ previous v2))
    (assumption_9 : (N v1))
    (assumption_4 : (add v1 v2 zero))
    : (v1 = v2) := by
  -- chapter_89_line_27: GL tag task formulation.
  have row_27 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_89_line_26: GL tag expansion.
  have row_26 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_27)
  -- chapter_89_line_25: GL tag disintegration.
  have row_25 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_26.1
  -- chapter_89_line_24: GL tag expansion.
  have row_24 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_25)
  -- chapter_89_line_23: GL tag disintegration.
  have row_23 : (gl_implication23 N succ add) := by
    exact row_24.1.1.1.1.1.1.2
  -- chapter_89_line_22: GL tag expansion.
  have row_22 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication23] using (row_23)
  -- chapter_89_line_21: GL tag disintegration.
  have row_21 : (gl_fXY succ N N) := by
    exact row_24.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_89_line_20: GL tag expansion.
  have row_20 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_21)
  -- chapter_89_line_19: GL tag disintegration.
  have row_19 : (gl_implication0 succ N) := by
    exact row_20.1.1.1
  -- chapter_89_line_18: GL tag expansion.
  have row_18 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_19)
  -- chapter_89_line_17: GL tag disintegration.
  have row_17 : (gl_fXYZ add N N N) := by
    exact row_24.1.1.1.1.1.1.1.1.1.2
  -- chapter_89_line_16: GL tag expansion.
  have row_16 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_17)
  -- chapter_89_line_15: GL tag disintegration.
  have row_15 : (gl_implication18 N N N add) := by
    exact row_16.1.2
  -- chapter_89_line_14: GL tag expansion.
  have row_14 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication18] using (row_15)
  -- chapter_89_line_13: GL tag disintegration.
  have row_13 : (gl_implication9 N zero succ) := by
    exact row_24.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_89_line_12: GL tag expansion.
  have row_12 : (∀ (w1 : α), ((N w1) → (¬ (succ w1 zero)))) := by
    simpa only [gl_implication9] using (row_13)
  -- chapter_89_line_11: GL tag recursion.
  have row_11 : (succ previous v2) := by
    exact assumption_11
  -- chapter_89_line_10: GL tag implication.
  have row_10 : (N previous) := by
    apply row_18
    exact row_11
  -- chapter_89_line_9: GL tag task formulation.
  have row_9 : (N v1) := by
    exact assumption_9
  -- chapter_89_line_8: GL tag implication.
  have row_8 : (gl_existence1 N v1 previous add) := by
    apply row_14
    exact row_9
    exact row_10
  -- chapter_89_line_7: GL tag expansion.
  have row_7 : (¬ (∀ (v3 : α), ((N v3) → (¬ (add v1 previous v3))))) := by
    simpa only [gl_existence1] using (row_8)
  have exists_row_7 : ∃ (v3 : α), ((N v3) ∧ (add v1 previous v3)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_7
      intro v3 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v3, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v3, witness_row_7⟩ := exists_row_7
  -- chapter_89_line_6: GL tag disintegration.
  have row_6 : (add v1 previous v3) := by
    exact witness_row_7.2
  -- chapter_89_line_5: GL tag disintegration.
  have row_5 : (N v3) := by
    exact witness_row_7.1
  -- chapter_89_line_4: GL tag task formulation.
  have row_4 : (add v1 v2 zero) := by
    exact assumption_4
  -- chapter_89_line_3: GL tag implication.
  have row_3 : (succ v3 zero) := by
    apply row_22
    exact row_10
    exact row_11
    exact row_6
    exact row_4
  -- chapter_89_line_2: GL tag implication.
  have row_2 : (¬ (succ v3 zero)) := by
    apply row_12
    exact row_5
  -- chapter_89_line_1: GL tag vacuous truth.
  have row_1 : (v1 = v2) := by
    exact False.elim (row_2 row_3)
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((add v1 v2 zero) → ((N v1) → (v1 = v2)))) := by
  intro v1
  intro v2
  intro premise_1
  intro premise_2
  have inductionMember : N v2 := by
    have typingRule := peano_source_045_induction_typing N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact typingRule v1 v2 premise_1
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v1 : α), ((add v1 zero zero) → ((N v1) → (v1 = zero)))) := by
    intro v1
    intro base_premise_1
    intro base_premise_2
    have zeroRule := peano_source_045_check_zero N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact zeroRule v1 zero base_premise_1 rfl
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
    have stepRule := peano_source_045_check_induction_condition N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact stepRule induction_n v1 induction_m step_induction_assumption_1 step_premise_2 step_premise_1
  have inductionProperty : (∀ (v1 : α), ((add v1 v2 zero) → ((N v1) → (v1 = v2)))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v1 : α), ((add v1 induction_value zero) → ((N v1) → (v1 = induction_value)))))
      v2
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v1 premise_1 premise_2

private theorem peano_source_062_induction_typing
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v1 : α)
    (assumption_1 : (N v1))
    : (N v1) := by
  -- chapter_130_line_1: GL tag task formulation.
  have row_1 : (N v1) := by
    exact assumption_1
  exact row_1

private theorem peano_source_062_check_zero
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v1 : α)
    (assumption_1 : (v1 = zero))
    : (v1 = zero) := by
  -- chapter_131_line_1: GL tag recursion.
  have row_1 : (v1 = zero) := by
    exact assumption_1
  exact row_1

private theorem peano_source_062_check_induction_condition
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (assumption_7 : (succ previous v1))
    (assumption_5 : (¬ (gl_inSequence N v1 succ)))
    : (v1 = zero) := by
  -- chapter_132_line_15: GL tag task formulation.
  have row_15 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_132_line_14: GL tag expansion.
  have row_14 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_15)
  -- chapter_132_line_13: GL tag disintegration.
  have row_13 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_14.1
  -- chapter_132_line_12: GL tag expansion.
  have row_12 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_13)
  -- chapter_132_line_11: GL tag disintegration.
  have row_11 : (gl_fXY succ N N) := by
    exact row_12.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_132_line_10: GL tag expansion.
  have row_10 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_11)
  -- chapter_132_line_9: GL tag disintegration.
  have row_9 : (gl_implication0 succ N) := by
    exact row_10.1.1.1
  -- chapter_132_line_8: GL tag expansion.
  have row_8 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_9)
  -- chapter_132_line_7: GL tag recursion.
  have row_7 : (succ previous v1) := by
    exact assumption_7
  -- chapter_132_line_6: GL tag implication.
  have row_6 : (N previous) := by
    apply row_8
    exact row_7
  -- chapter_132_line_5: GL tag task formulation.
  have row_5 : (¬ (gl_inSequence N v1 succ)) := by
    exact assumption_5
  -- chapter_132_line_4: GL tag expansion.
  have row_4 : (gl_implication165 v1 succ N) := by
    simp only [gl_implication165]
    intro existence_witness_1 compact_premise compact_negated
    have unfolded_row_5 := row_5
    simp only [gl_inSequence] at unfolded_row_5
    apply unfolded_row_5
    intro positive_row_5
    exact positive_row_5 existence_witness_1 compact_negated compact_premise
  -- chapter_132_line_3: GL tag expansion.
  have row_3 : (∀ (w1 : α), ((succ w1 v1) → (¬ (N w1)))) := by
    simpa only [gl_implication165] using (row_4)
  -- chapter_132_line_2: GL tag implication.
  have row_2 : (¬ (N previous)) := by
    apply row_3
    exact row_7
  -- chapter_132_line_1: GL tag vacuous truth.
  have row_1 : (v1 = zero) := by
    exact False.elim (row_2 row_6)
  exact row_1

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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α), ((N v1) → ((¬ (gl_inSequence N v1 succ)) → (v1 = zero)))) := by
  intro v1
  intro premise_1
  intro premise_2
  have inductionMember : N v1 := by
    have typingRule := peano_source_062_induction_typing N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact typingRule v1 premise_1
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : ((¬ (gl_inSequence N zero succ)) → (zero = zero)) := by
    intro base_premise_1
    have zeroRule := peano_source_062_check_zero N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact zeroRule zero rfl
  have inductionStep :
      ∀ induction_n, N induction_n → ((¬ (gl_inSequence N induction_n succ)) → (induction_n = zero)) → ∀ induction_m, succ induction_n induction_m → ((¬ (gl_inSequence N induction_m succ)) → (induction_m = zero)) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro step_premise_1
    have step_induction_assumption_1 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have stepRule := peano_source_062_check_induction_condition N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact stepRule induction_n induction_m step_induction_assumption_1 step_premise_1
  have inductionProperty : ((¬ (gl_inSequence N v1 succ)) → (v1 = zero)) := by
    exact relationalInduction
      (fun induction_value => ((¬ (gl_inSequence N induction_value succ)) → (induction_value = zero)))
      v1
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty premise_2

theorem peano_source_065
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (gl_inSequence N one succ) := by
  -- chapter_135_line_10: GL tag task formulation.
  have row_10 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_135_line_9: GL tag expansion.
  have row_9 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_10)
  -- chapter_135_line_8: GL tag disintegration.
  have row_8 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_9.1
  -- chapter_135_line_7: GL tag expansion.
  have row_7 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_8)
  -- chapter_135_line_6: GL tag disintegration.
  have row_6 : (N zero) := by
    exact row_7.1.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_135_line_5: GL tag disintegration.
  have row_5 : (succ zero one) := by
    exact row_9.2
  -- chapter_135_line_4: GL tag expansion for integration.
  have row_4 : ((gl_inSequence N one succ) ↔ (¬ (∀ (v2 : α), ((N v2) → (¬ (succ v2 one)))))) := by
    exact Iff.rfl
  -- chapter_135_line_3_rule: GL tag reformulation for integration >[bound].
  have row_3_rule : (∀ (cr3_1 : α), ((N cr3_1) → ((succ cr3_1 one) → (gl_inSequence N one succ)))) := by
    intro cr3_1
    intro integration_premise_1
    intro integration_premise_2
    apply (row_4).2
    intro universal_counterexample
    exact universal_counterexample cr3_1 integration_premise_1 integration_premise_2
  -- chapter_135_line_3: GL tag reformulation for integration >[bound].
  have row_3 : (gl_implication1873 N one succ) := by
    simpa only [gl_implication1873] using row_3_rule
  -- chapter_135_line_2: GL tag expansion.
  have row_2 : (∀ (v1 : α), ((N v1) → ((succ v1 one) → (gl_inSequence N one succ)))) := by
    simpa only [gl_implication1873] using (row_3)
  -- chapter_135_line_1: GL tag implication.
  have row_1 : (gl_inSequence N one succ) := by
    apply row_2
    exact row_6
    exact row_5
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (v3 : α) (v4 : α), ((succ v3 v4) → (gl_or0 v2 v4 v1 v3))))) := by
  intro v1
  intro v2
  intro premise_1
  intro v3
  intro v4
  intro premise_2
  have or_parent_1 := peano_source_003 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  have or_parent_2 := peano_source_002 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_4_line_1: GL tag or theorem.
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (v3 : α) (v4 : α), ((succ v3 v4) → (gl_or0 v2 v4 v1 v3))))) := by
  intro v1
  intro v2
  intro premise_1
  intro v3
  intro v4
  intro premise_2
  have or_parent_1 := peano_source_003 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  have or_parent_2 := peano_source_002 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_5_line_1: GL tag or theorem.
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (v3 : α) (v4 : α), ((succ v3 v4) → (∀ (v5 : α), ((add v5 v2 v4) → (add v5 v1 v3))))))) := by
  intro v1
  intro v2
  intro premise_1
  intro v3
  intro v4
  intro premise_2
  intro v5
  intro premise_3
  -- chapter_7_line_33: GL tag task formulation.
  have row_33 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_7_line_32: GL tag expansion.
  have row_32 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_33)
  -- chapter_7_line_31: GL tag disintegration.
  have row_31 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_32.1
  -- chapter_7_line_30: GL tag expansion.
  have row_30 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_31)
  -- chapter_7_line_29: GL tag disintegration.
  have row_29 : (gl_fXY succ N N) := by
    exact row_30.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_7_line_28: GL tag expansion.
  have row_28 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_29)
  -- chapter_7_line_27: GL tag disintegration.
  have row_27 : (gl_implication1 succ N) := by
    exact row_28.1.1.2
  -- chapter_7_line_26: GL tag expansion.
  have row_26 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using (row_27)
  -- chapter_7_line_25: GL tag disintegration.
  have row_25 : (gl_implication0 succ N) := by
    exact row_28.1.1.1
  -- chapter_7_line_24: GL tag expansion.
  have row_24 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_25)
  -- chapter_7_line_23: GL tag disintegration.
  have row_23 : (gl_fXYZ add N N N) := by
    exact row_30.1.1.1.1.1.1.1.1.1.2
  -- chapter_7_line_22: GL tag expansion.
  have row_22 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_23)
  -- chapter_7_line_21: GL tag disintegration.
  have row_21 : (gl_implication11 add N) := by
    exact row_22.1.1.1.1
  -- chapter_7_line_20: GL tag expansion.
  have row_20 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_21)
  -- chapter_7_line_19: GL tag disintegration.
  have row_19 : (gl_implication18 N N N add) := by
    exact row_22.1.2
  -- chapter_7_line_18: GL tag expansion.
  have row_18 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication18] using (row_19)
  -- chapter_7_line_17: GL tag disintegration.
  have row_17 : (gl_implication23 N succ add) := by
    exact row_30.1.1.1.1.1.1.2
  -- chapter_7_line_16: GL tag expansion.
  have row_16 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication23] using (row_17)
  -- chapter_7_line_15: GL tag implication grouping.
  have row_15 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (gl_and20 w1 N w2 N))) := by
    have auxiliary_source_1 := row_24
    have auxiliary_source_2 := row_26
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_premise
    exact ⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_premise)⟩
  -- chapter_7_line_14: GL tag task formulation.
  have row_14 : (succ v1 v2) := by
    exact premise_1
  -- chapter_7_line_13: GL tag implication.
  have row_13 : (gl_and20 v1 N v2 N) := by
    apply row_15
    exact row_14
  -- chapter_7_line_12: GL tag expansion.
  have row_12 : ((N v1) ∧ (N v2)) := by
    simpa only [gl_and20] using (row_13)
  -- chapter_7_line_11: GL tag disintegration.
  have row_11 : (N v1) := by
    exact row_12.1
  -- chapter_7_line_10: GL tag task formulation.
  have row_10 : (add v5 v2 v4) := by
    exact premise_3
  -- chapter_7_line_9: GL tag implication.
  have row_9 : (N v5) := by
    apply row_20
    exact row_10
  -- chapter_7_line_8: GL tag implication.
  have row_8 : (gl_existence1 N v5 v1 add) := by
    apply row_18
    exact row_9
    exact row_11
  -- chapter_7_line_7: GL tag expansion.
  have row_7 : (¬ (∀ (v6 : α), ((N v6) → (¬ (add v5 v1 v6))))) := by
    simpa only [gl_existence1] using (row_8)
  have exists_row_7 : ∃ (v6 : α), ((N v6) ∧ (add v5 v1 v6)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_7
      intro v6 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v6, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v6, witness_row_7⟩ := exists_row_7
  -- chapter_7_line_6: GL tag disintegration.
  have row_6 : (add v5 v1 v6) := by
    exact witness_row_7.2
  -- chapter_7_line_5: GL tag implication.
  have row_5 : (succ v6 v4) := by
    apply row_16
    exact row_11
    exact row_14
    exact row_6
    exact row_10
  -- chapter_7_line_4: GL tag task formulation.
  have row_4 : (succ v3 v4) := by
    exact premise_2
  -- chapter_7_line_3: GL tag theorem.
  have row_3 := peano_source_006 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_7_line_2: GL tag implication.
  have row_2 : (v6 = v3) := by
    apply row_3
    exact row_5
    exact row_4
  -- chapter_7_line_1: GL tag equality1.
  have row_1 : (add v5 v1 v3) := by
    have equality_source := row_6
    have equality_step_1 := row_2
    cases equality_step_1
    exact equality_source
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((add one v1 v2) → (add v1 one v2))) := by
  intro v1
  intro v2
  intro premise_1
  -- chapter_14_line_44: GL tag task formulation.
  have row_44 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_14_line_43: GL tag expansion.
  have row_43 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_44)
  -- chapter_14_line_42: GL tag disintegration.
  have row_42 : (succ zero one) := by
    exact row_43.2
  -- chapter_14_line_41: GL tag disintegration.
  have row_41 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_43.1
  -- chapter_14_line_40: GL tag expansion.
  have row_40 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_41)
  -- chapter_14_line_39: GL tag disintegration.
  have row_39 : (N zero) := by
    exact row_40.1.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_14_line_38: GL tag disintegration.
  have row_38 : (gl_fXYZ add N N N) := by
    exact row_40.1.1.1.1.1.1.1.1.1.2
  -- chapter_14_line_37: GL tag expansion.
  have row_37 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_38)
  -- chapter_14_line_36: GL tag disintegration.
  have row_36 : (gl_implication19 N N add) := by
    exact row_37.2
  -- chapter_14_line_35: GL tag expansion.
  have row_35 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (∀ (w3 : α), ((add w1 w2 w3) → (∀ (w4 : α), ((add w1 w2 w4) → (w3 = w4))))))))) := by
    simpa only [gl_implication19] using (row_36)
  -- chapter_14_line_34: GL tag disintegration.
  have row_34 : (gl_implication12 add N) := by
    exact row_37.1.1.1.2
  -- chapter_14_line_33: GL tag expansion.
  have row_33 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_34)
  -- chapter_14_line_32: GL tag disintegration.
  have row_32 : (gl_implication18 N N N add) := by
    exact row_37.1.2
  -- chapter_14_line_31: GL tag expansion.
  have row_31 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication18] using (row_32)
  -- chapter_14_line_30: GL tag disintegration.
  have row_30 : (gl_fXY succ N N) := by
    exact row_40.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_14_line_29: GL tag expansion.
  have row_29 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_30)
  -- chapter_14_line_28: GL tag disintegration.
  have row_28 : (gl_implication1 succ N) := by
    exact row_29.1.1.2
  -- chapter_14_line_27: GL tag expansion.
  have row_27 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using (row_28)
  -- chapter_14_line_26: GL tag disintegration.
  have row_26 : (gl_implication0 succ N) := by
    exact row_29.1.1.1
  -- chapter_14_line_25: GL tag expansion.
  have row_25 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_26)
  -- chapter_14_line_24: GL tag disintegration.
  have row_24 : (gl_implication6 N N succ) := by
    exact row_29.1.2
  -- chapter_14_line_23: GL tag expansion.
  have row_23 : (∀ (w1 : α), ((N w1) → (gl_existence0 N w1 succ))) := by
    simpa only [gl_implication6] using (row_24)
  -- chapter_14_line_22: GL tag disintegration.
  have row_22 : (gl_implication24 N succ add) := by
    exact row_40.1.1.1.1.1.2
  -- chapter_14_line_21: GL tag expansion.
  have row_21 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((succ w4 w5) → (add w3 w2 w5))))))))) := by
    simpa only [gl_implication24] using (row_22)
  -- chapter_14_line_20: GL tag task formulation.
  have row_20 : (add one v1 v2) := by
    exact premise_1
  -- chapter_14_line_19: GL tag implication.
  have row_19 : (N v1) := by
    apply row_33
    exact row_20
  -- chapter_14_line_18: GL tag implication.
  have row_18 : (gl_existence1 N v1 zero add) := by
    apply row_31
    exact row_19
    exact row_39
  -- chapter_14_line_17: GL tag expansion.
  have row_17 : (¬ (∀ (v9 : α), ((N v9) → (¬ (add v1 zero v9))))) := by
    simpa only [gl_existence1] using (row_18)
  have exists_row_17 : ∃ (v9 : α), ((N v9) ∧ (add v1 zero v9)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_17
      intro v9 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v9, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v9, witness_row_17⟩ := exists_row_17
  -- chapter_14_line_16: GL tag disintegration.
  have row_16 : (add v1 zero v9) := by
    exact witness_row_17.2
  -- chapter_14_line_15: GL tag implication grouping.
  have row_15 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (gl_and20 w1 N w2 N))) := by
    have auxiliary_source_1 := row_25
    have auxiliary_source_2 := row_27
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_premise
    exact ⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_premise)⟩
  -- chapter_14_line_14: GL tag implication.
  have row_14 : (gl_and20 zero N one N) := by
    apply row_15
    exact row_42
  -- chapter_14_line_13: GL tag expansion.
  have row_13 : ((N zero) ∧ (N one)) := by
    simpa only [gl_and20] using (row_14)
  -- chapter_14_line_12: GL tag disintegration.
  have row_12 : (N one) := by
    exact row_13.2
  -- chapter_14_line_11: GL tag implication.
  have row_11 : (gl_existence0 N v1 succ) := by
    apply row_23
    exact row_19
  -- chapter_14_line_10: GL tag expansion.
  have row_10 : (¬ (∀ (v3 : α), ((N v3) → (¬ (succ v1 v3))))) := by
    simpa only [gl_existence0] using (row_11)
  have exists_row_10 : ∃ (v3 : α), ((N v3) ∧ (succ v1 v3)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_10
      intro v3 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v3, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v3, witness_row_10⟩ := exists_row_10
  -- chapter_14_line_9: GL tag disintegration.
  have row_9 : (succ v1 v3) := by
    exact witness_row_10.2
  -- chapter_14_line_8: GL tag theorem.
  have row_8 := peano_source_010 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_14_line_7: GL tag implication.
  have row_7 : (add one v1 v3) := by
    apply row_8
    exact row_9
  -- chapter_14_line_6: GL tag implication.
  have row_6 : (v3 = v2) := by
    apply row_35
    exact row_12
    exact row_19
    exact row_7
    exact row_20
  -- chapter_14_line_5: GL tag theorem.
  have row_5 := peano_source_011 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_14_line_4: GL tag implication.
  have row_4 : (v1 = v9) := by
    apply row_5
    exact row_16
    exact row_19
  -- chapter_14_line_3: GL tag equality1.
  have row_3 : (succ v9 v3) := by
    have equality_source := row_9
    have equality_step_1 := row_4
    cases equality_step_1
    exact equality_source
  -- chapter_14_line_2: GL tag implication.
  have row_2 : (add v1 one v3) := by
    apply row_21
    exact row_39
    exact row_42
    exact row_16
    exact row_3
  -- chapter_14_line_1: GL tag equality1.
  have row_1 : (add v1 one v2) := by
    have equality_source := row_2
    have equality_step_1 := row_6
    cases equality_step_1
    exact equality_source
  exact row_1

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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v2 : α)
    (assumption_1 : (N v2))
    : (N v2) := by
  -- chapter_15_line_1: GL tag task formulation.
  have row_1 : (N v2) := by
    exact assumption_1
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (assumption_28 : (add v1 v2 zero))
    (assumption_27 : (v2 = zero))
    : (zero = v1) := by
  -- chapter_16_line_28: GL tag task formulation.
  have row_28 : (add v1 v2 zero) := by
    exact assumption_28
  -- chapter_16_line_27: GL tag recursion.
  have row_27 : (v2 = zero) := by
    exact assumption_27
  -- chapter_16_line_26: GL tag symmetry of equality.
  have row_26 : (zero = v2) := by
    exact Eq.symm row_27
  -- chapter_16_line_25: GL tag equality1.
  have row_25 : (add v1 zero v2) := by
    have equality_source := row_28
    have equality_step_1 := row_27
    cases equality_step_1
    have equality_step_2 := row_26
    cases equality_step_2
    exact equality_source
  -- chapter_16_line_24: GL tag task formulation.
  have row_24 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_16_line_23: GL tag expansion.
  have row_23 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_24)
  -- chapter_16_line_22: GL tag disintegration.
  have row_22 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_23.1
  -- chapter_16_line_21: GL tag expansion.
  have row_21 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_22)
  -- chapter_16_line_20: GL tag disintegration.
  have row_20 : (gl_implication26 zero N mul) := by
    exact row_21.1.1.2
  -- chapter_16_line_19: GL tag expansion.
  have row_19 : (∀ (w1 : α), ((w1 = zero) → (∀ (w2 : α), ((N w2) → ((N w1) → (mul w2 zero w1)))))) := by
    simpa only [gl_implication26] using (row_20)
  -- chapter_16_line_18: GL tag disintegration.
  have row_18 : (gl_fXYZ add N N N) := by
    exact row_21.1.1.1.1.1.1.1.1.1.2
  -- chapter_16_line_17: GL tag expansion.
  have row_17 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_18)
  -- chapter_16_line_16: GL tag disintegration.
  have row_16 : (gl_implication13 add N) := by
    exact row_17.1.1.2
  -- chapter_16_line_15: GL tag expansion.
  have row_15 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_16)
  -- chapter_16_line_14: GL tag disintegration.
  have row_14 : (gl_implication12 add N) := by
    exact row_17.1.1.1.2
  -- chapter_16_line_13: GL tag expansion.
  have row_13 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_14)
  -- chapter_16_line_12: GL tag disintegration.
  have row_12 : (gl_implication11 add N) := by
    exact row_17.1.1.1.1
  -- chapter_16_line_11: GL tag expansion.
  have row_11 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_12)
  -- chapter_16_line_10: GL tag implication grouping.
  have row_10 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_11
    have auxiliary_source_2 := row_13
    have auxiliary_source_3 := row_15
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_16_line_9: GL tag implication.
  have row_9 : (gl_and21 v1 N v2 N zero N) := by
    apply row_10
    exact row_28
  -- chapter_16_line_8: GL tag expansion.
  have row_8 : (((N v1) ∧ (N v2)) ∧ (N zero)) := by
    simpa only [gl_and21] using (row_9)
  -- chapter_16_line_7: GL tag disintegration.
  have row_7 : (N v1) := by
    exact row_8.1.1
  -- chapter_16_line_6: GL tag disintegration.
  have row_6 : (N v2) := by
    exact row_8.1.2
  -- chapter_16_line_5: GL tag implication.
  have row_5 : (mul v1 zero v2) := by
    apply row_19
    exact row_27
    exact row_7
    exact row_6
  -- chapter_16_line_4: GL tag theorem.
  have row_4 := peano_source_011 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_16_line_3: GL tag implication.
  have row_3 : (v1 = v2) := by
    apply row_4
    exact row_25
    exact row_7
  -- chapter_16_line_2: GL tag theorem.
  have row_2 := peano_source_009 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_16_line_1: GL tag implication.
  have row_1 : (zero = v1) := by
    apply row_2
    exact row_3
    exact row_5
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (assumption_14 : (succ previous v2))
    (assumption_11 : (add v1 v2 zero))
    : (zero = v1) := by
  -- chapter_17_line_36: GL tag task formulation.
  have row_36 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_17_line_35: GL tag expansion.
  have row_35 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_36)
  -- chapter_17_line_34: GL tag disintegration.
  have row_34 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_35.1
  -- chapter_17_line_33: GL tag expansion.
  have row_33 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_34)
  -- chapter_17_line_32: GL tag disintegration.
  have row_32 : (gl_implication23 N succ add) := by
    exact row_33.1.1.1.1.1.1.2
  -- chapter_17_line_31: GL tag expansion.
  have row_31 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication23] using (row_32)
  -- chapter_17_line_30: GL tag disintegration.
  have row_30 : (gl_fXY succ N N) := by
    exact row_33.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_17_line_29: GL tag expansion.
  have row_29 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_30)
  -- chapter_17_line_28: GL tag disintegration.
  have row_28 : (gl_implication0 succ N) := by
    exact row_29.1.1.1
  -- chapter_17_line_27: GL tag expansion.
  have row_27 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_28)
  -- chapter_17_line_26: GL tag disintegration.
  have row_26 : (gl_fXYZ add N N N) := by
    exact row_33.1.1.1.1.1.1.1.1.1.2
  -- chapter_17_line_25: GL tag expansion.
  have row_25 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_26)
  -- chapter_17_line_24: GL tag disintegration.
  have row_24 : (gl_implication13 add N) := by
    exact row_25.1.1.2
  -- chapter_17_line_23: GL tag expansion.
  have row_23 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_24)
  -- chapter_17_line_22: GL tag disintegration.
  have row_22 : (gl_implication12 add N) := by
    exact row_25.1.1.1.2
  -- chapter_17_line_21: GL tag expansion.
  have row_21 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_22)
  -- chapter_17_line_20: GL tag disintegration.
  have row_20 : (gl_implication11 add N) := by
    exact row_25.1.1.1.1
  -- chapter_17_line_19: GL tag expansion.
  have row_19 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_20)
  -- chapter_17_line_18: GL tag disintegration.
  have row_18 : (gl_implication18 N N N add) := by
    exact row_25.1.2
  -- chapter_17_line_17: GL tag expansion.
  have row_17 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication18] using (row_18)
  -- chapter_17_line_16: GL tag disintegration.
  have row_16 : (gl_implication9 N zero succ) := by
    exact row_33.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_17_line_15: GL tag expansion.
  have row_15 : (∀ (w1 : α), ((N w1) → (¬ (succ w1 zero)))) := by
    simpa only [gl_implication9] using (row_16)
  -- chapter_17_line_14: GL tag recursion.
  have row_14 : (succ previous v2) := by
    exact assumption_14
  -- chapter_17_line_13: GL tag implication.
  have row_13 : (N previous) := by
    apply row_27
    exact row_14
  -- chapter_17_line_12: GL tag implication grouping.
  have row_12 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_19
    have auxiliary_source_2 := row_21
    have auxiliary_source_3 := row_23
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_17_line_11: GL tag task formulation.
  have row_11 : (add v1 v2 zero) := by
    exact assumption_11
  -- chapter_17_line_10: GL tag implication.
  have row_10 : (gl_and21 v1 N v2 N zero N) := by
    apply row_12
    exact row_11
  -- chapter_17_line_9: GL tag expansion.
  have row_9 : (((N v1) ∧ (N v2)) ∧ (N zero)) := by
    simpa only [gl_and21] using (row_10)
  -- chapter_17_line_8: GL tag disintegration.
  have row_8 : (N v1) := by
    exact row_9.1.1
  -- chapter_17_line_7: GL tag implication.
  have row_7 : (gl_existence1 N v1 previous add) := by
    apply row_17
    exact row_8
    exact row_13
  -- chapter_17_line_6: GL tag expansion.
  have row_6 : (¬ (∀ (v3 : α), ((N v3) → (¬ (add v1 previous v3))))) := by
    simpa only [gl_existence1] using (row_7)
  have exists_row_6 : ∃ (v3 : α), ((N v3) ∧ (add v1 previous v3)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_6
      intro v3 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v3, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v3, witness_row_6⟩ := exists_row_6
  -- chapter_17_line_5: GL tag disintegration.
  have row_5 : (add v1 previous v3) := by
    exact witness_row_6.2
  -- chapter_17_line_4: GL tag disintegration.
  have row_4 : (N v3) := by
    exact witness_row_6.1
  -- chapter_17_line_3: GL tag implication.
  have row_3 : (succ v3 zero) := by
    apply row_31
    exact row_13
    exact row_14
    exact row_5
    exact row_11
  -- chapter_17_line_2: GL tag implication.
  have row_2 : (¬ (succ v3 zero)) := by
    apply row_15
    exact row_4
  -- chapter_17_line_1: GL tag vacuous truth.
  have row_1 : (zero = v1) := by
    exact False.elim (row_2 row_3)
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((add v1 v2 zero) → ((N v2) → (zero = v1)))) := by
  intro v1
  intro v2
  intro premise_1
  intro premise_2
  have inductionMember : N v2 := by
    have typingRule := peano_source_013_induction_typing N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact typingRule v2 premise_2
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v1 : α), ((add v1 zero zero) → ((N zero) → (zero = v1)))) := by
    intro v1
    intro base_premise_1
    intro base_premise_2
    have zeroRule := peano_source_013_check_zero N zero succ add mul one anchor relationalInduction strongRelationalInduction
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
    have stepRule := peano_source_013_check_induction_condition N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact stepRule induction_n v1 induction_m step_induction_assumption_1 step_premise_1
  have inductionProperty : (∀ (v1 : α), ((add v1 v2 zero) → ((N v2) → (zero = v1)))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v1 : α), ((add v1 induction_value zero) → ((N induction_value) → (zero = v1)))))
      v2
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v1 premise_1 premise_2

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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (assumption_2 : (add v1 v2 zero))
    : (N v2) := by
  -- chapter_18_line_10: GL tag task formulation.
  have row_10 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_18_line_9: GL tag expansion.
  have row_9 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_10)
  -- chapter_18_line_8: GL tag disintegration.
  have row_8 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_9.1
  -- chapter_18_line_7: GL tag expansion.
  have row_7 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_8)
  -- chapter_18_line_6: GL tag disintegration.
  have row_6 : (gl_fXYZ add N N N) := by
    exact row_7.1.1.1.1.1.1.1.1.1.2
  -- chapter_18_line_5: GL tag expansion.
  have row_5 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_6)
  -- chapter_18_line_4: GL tag disintegration.
  have row_4 : (gl_implication12 add N) := by
    exact row_5.1.1.1.2
  -- chapter_18_line_3: GL tag expansion.
  have row_3 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_4)
  -- chapter_18_line_2: GL tag task formulation.
  have row_2 : (add v1 v2 zero) := by
    exact assumption_2
  -- chapter_18_line_1: GL tag implication.
  have row_1 : (N v2) := by
    apply row_3
    exact row_2
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (assumption_13 : (add v1 v2 zero))
    (assumption_9 : (v2 = zero))
    : (add v2 v1 zero) := by
  -- chapter_19_line_26: GL tag task formulation.
  have row_26 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_19_line_25: GL tag expansion.
  have row_25 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_26)
  -- chapter_19_line_24: GL tag disintegration.
  have row_24 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_25.1
  -- chapter_19_line_23: GL tag expansion.
  have row_23 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_24)
  -- chapter_19_line_22: GL tag disintegration.
  have row_22 : (gl_fXYZ add N N N) := by
    exact row_23.1.1.1.1.1.1.1.1.1.2
  -- chapter_19_line_21: GL tag expansion.
  have row_21 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_22)
  -- chapter_19_line_20: GL tag disintegration.
  have row_20 : (gl_implication13 add N) := by
    exact row_21.1.1.2
  -- chapter_19_line_19: GL tag expansion.
  have row_19 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_20)
  -- chapter_19_line_18: GL tag disintegration.
  have row_18 : (gl_implication12 add N) := by
    exact row_21.1.1.1.2
  -- chapter_19_line_17: GL tag expansion.
  have row_17 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_18)
  -- chapter_19_line_16: GL tag disintegration.
  have row_16 : (gl_implication11 add N) := by
    exact row_21.1.1.1.1
  -- chapter_19_line_15: GL tag expansion.
  have row_15 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_16)
  -- chapter_19_line_14: GL tag implication grouping.
  have row_14 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_15
    have auxiliary_source_2 := row_17
    have auxiliary_source_3 := row_19
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_19_line_13: GL tag task formulation.
  have row_13 : (add v1 v2 zero) := by
    exact assumption_13
  -- chapter_19_line_12: GL tag implication.
  have row_12 : (gl_and21 v1 N v2 N zero N) := by
    apply row_14
    exact row_13
  -- chapter_19_line_11: GL tag expansion.
  have row_11 : (((N v1) ∧ (N v2)) ∧ (N zero)) := by
    simpa only [gl_and21] using (row_12)
  -- chapter_19_line_10: GL tag disintegration.
  have row_10 : (N v1) := by
    exact row_11.1.1
  -- chapter_19_line_9: GL tag recursion.
  have row_9 : (v2 = zero) := by
    exact assumption_9
  -- chapter_19_line_8: GL tag equality1.
  have row_8 : (add v1 zero zero) := by
    have equality_source := row_13
    have equality_step_1 := row_9
    cases equality_step_1
    exact equality_source
  -- chapter_19_line_7: GL tag symmetry of equality.
  have row_7 : (zero = v2) := by
    exact Eq.symm row_9
  -- chapter_19_line_6: GL tag equality1.
  have row_6 : (add v1 zero v2) := by
    have equality_source := row_8
    have equality_step_1 := row_7
    cases equality_step_1
    exact equality_source
  -- chapter_19_line_5: GL tag theorem.
  have row_5 := peano_source_011 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_19_line_4: GL tag implication.
  have row_4 : (v1 = v2) := by
    apply row_5
    exact row_6
    exact row_10
  -- chapter_19_line_3: GL tag symmetry of equality.
  have row_3 : (v2 = v1) := by
    exact Eq.symm row_4
  -- chapter_19_line_2: GL tag equality2.
  have row_2 : (zero = v1) := by
    exact Eq.trans row_7 row_3
  -- chapter_19_line_1: GL tag equality1.
  have row_1 : (add v2 v1 zero) := by
    have equality_source := row_8
    have equality_step_1 := row_2
    cases equality_step_1
    have equality_step_2 := row_4
    cases equality_step_2
    exact equality_source
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (assumption_14 : (succ previous v2))
    (assumption_11 : (add v1 v2 zero))
    : (add v2 v1 zero) := by
  -- chapter_20_line_36: GL tag task formulation.
  have row_36 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_20_line_35: GL tag expansion.
  have row_35 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_36)
  -- chapter_20_line_34: GL tag disintegration.
  have row_34 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_35.1
  -- chapter_20_line_33: GL tag expansion.
  have row_33 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_34)
  -- chapter_20_line_32: GL tag disintegration.
  have row_32 : (gl_implication23 N succ add) := by
    exact row_33.1.1.1.1.1.1.2
  -- chapter_20_line_31: GL tag expansion.
  have row_31 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication23] using (row_32)
  -- chapter_20_line_30: GL tag disintegration.
  have row_30 : (gl_fXY succ N N) := by
    exact row_33.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_20_line_29: GL tag expansion.
  have row_29 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_30)
  -- chapter_20_line_28: GL tag disintegration.
  have row_28 : (gl_implication0 succ N) := by
    exact row_29.1.1.1
  -- chapter_20_line_27: GL tag expansion.
  have row_27 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_28)
  -- chapter_20_line_26: GL tag disintegration.
  have row_26 : (gl_fXYZ add N N N) := by
    exact row_33.1.1.1.1.1.1.1.1.1.2
  -- chapter_20_line_25: GL tag expansion.
  have row_25 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_26)
  -- chapter_20_line_24: GL tag disintegration.
  have row_24 : (gl_implication13 add N) := by
    exact row_25.1.1.2
  -- chapter_20_line_23: GL tag expansion.
  have row_23 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_24)
  -- chapter_20_line_22: GL tag disintegration.
  have row_22 : (gl_implication12 add N) := by
    exact row_25.1.1.1.2
  -- chapter_20_line_21: GL tag expansion.
  have row_21 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_22)
  -- chapter_20_line_20: GL tag disintegration.
  have row_20 : (gl_implication11 add N) := by
    exact row_25.1.1.1.1
  -- chapter_20_line_19: GL tag expansion.
  have row_19 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_20)
  -- chapter_20_line_18: GL tag disintegration.
  have row_18 : (gl_implication18 N N N add) := by
    exact row_25.1.2
  -- chapter_20_line_17: GL tag expansion.
  have row_17 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication18] using (row_18)
  -- chapter_20_line_16: GL tag disintegration.
  have row_16 : (gl_implication9 N zero succ) := by
    exact row_33.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_20_line_15: GL tag expansion.
  have row_15 : (∀ (w1 : α), ((N w1) → (¬ (succ w1 zero)))) := by
    simpa only [gl_implication9] using (row_16)
  -- chapter_20_line_14: GL tag recursion.
  have row_14 : (succ previous v2) := by
    exact assumption_14
  -- chapter_20_line_13: GL tag implication.
  have row_13 : (N previous) := by
    apply row_27
    exact row_14
  -- chapter_20_line_12: GL tag implication grouping.
  have row_12 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_19
    have auxiliary_source_2 := row_21
    have auxiliary_source_3 := row_23
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_20_line_11: GL tag task formulation.
  have row_11 : (add v1 v2 zero) := by
    exact assumption_11
  -- chapter_20_line_10: GL tag implication.
  have row_10 : (gl_and21 v1 N v2 N zero N) := by
    apply row_12
    exact row_11
  -- chapter_20_line_9: GL tag expansion.
  have row_9 : (((N v1) ∧ (N v2)) ∧ (N zero)) := by
    simpa only [gl_and21] using (row_10)
  -- chapter_20_line_8: GL tag disintegration.
  have row_8 : (N v1) := by
    exact row_9.1.1
  -- chapter_20_line_7: GL tag implication.
  have row_7 : (gl_existence1 N v1 previous add) := by
    apply row_17
    exact row_8
    exact row_13
  -- chapter_20_line_6: GL tag expansion.
  have row_6 : (¬ (∀ (v3 : α), ((N v3) → (¬ (add v1 previous v3))))) := by
    simpa only [gl_existence1] using (row_7)
  have exists_row_6 : ∃ (v3 : α), ((N v3) ∧ (add v1 previous v3)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_6
      intro v3 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v3, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v3, witness_row_6⟩ := exists_row_6
  -- chapter_20_line_5: GL tag disintegration.
  have row_5 : (add v1 previous v3) := by
    exact witness_row_6.2
  -- chapter_20_line_4: GL tag disintegration.
  have row_4 : (N v3) := by
    exact witness_row_6.1
  -- chapter_20_line_3: GL tag implication.
  have row_3 : (succ v3 zero) := by
    apply row_31
    exact row_13
    exact row_14
    exact row_5
    exact row_11
  -- chapter_20_line_2: GL tag implication.
  have row_2 : (¬ (succ v3 zero)) := by
    apply row_15
    exact row_4
  -- chapter_20_line_1: GL tag vacuous truth.
  have row_1 : (add v2 v1 zero) := by
    exact False.elim (row_2 row_3)
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((add v1 v2 zero) → (add v2 v1 zero))) := by
  intro v1
  intro v2
  intro premise_1
  have inductionMember : N v2 := by
    have typingRule := peano_source_014_induction_typing N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact typingRule v1 v2 premise_1
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v1 : α), ((add v1 zero zero) → (add zero v1 zero))) := by
    intro v1
    intro base_premise_1
    have zeroRule := peano_source_014_check_zero N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact zeroRule v1 zero base_premise_1 rfl
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v1 : α), ((add v1 induction_n zero) → (add induction_n v1 zero))) → ∀ induction_m, succ induction_n induction_m → (∀ (v1 : α), ((add v1 induction_m zero) → (add induction_m v1 zero))) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro v1
    intro step_premise_1
    have step_induction_assumption_1 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have stepRule := peano_source_014_check_induction_condition N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact stepRule induction_n v1 induction_m step_induction_assumption_1 step_premise_1
  have inductionProperty : (∀ (v1 : α), ((add v1 v2 zero) → (add v2 v1 zero))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v1 : α), ((add v1 induction_value zero) → (add induction_value v1 zero))))
      v2
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v1 premise_1

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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((add v1 one v2) → (succ v1 v2))) := by
  intro v1
  intro v2
  intro premise_1
  -- chapter_21_line_24: GL tag task formulation.
  have row_24 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_21_line_23: GL tag expansion.
  have row_23 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_24)
  -- chapter_21_line_22: GL tag disintegration.
  have row_22 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_23.1
  -- chapter_21_line_21: GL tag expansion.
  have row_21 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_22)
  -- chapter_21_line_20: GL tag disintegration.
  have row_20 : (gl_fXYZ add N N N) := by
    exact row_21.1.1.1.1.1.1.1.1.1.2
  -- chapter_21_line_19: GL tag expansion.
  have row_19 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_20)
  -- chapter_21_line_18: GL tag disintegration.
  have row_18 : (gl_implication23 N succ add) := by
    exact row_21.1.1.1.1.1.1.2
  -- chapter_21_line_17: GL tag expansion.
  have row_17 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication23] using (row_18)
  -- chapter_21_line_16: GL tag disintegration.
  have row_16 : (gl_implication11 add N) := by
    exact row_19.1.1.1.1
  -- chapter_21_line_15: GL tag expansion.
  have row_15 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_16)
  -- chapter_21_line_14: GL tag disintegration.
  have row_14 : (gl_implication18 N N N add) := by
    exact row_19.1.2
  -- chapter_21_line_13: GL tag expansion.
  have row_13 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication18] using (row_14)
  -- chapter_21_line_12: GL tag disintegration.
  have row_12 : (N zero) := by
    exact row_21.1.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_21_line_11: GL tag disintegration.
  have row_11 : (succ zero one) := by
    exact row_23.2
  -- chapter_21_line_10: GL tag task formulation.
  have row_10 : (add v1 one v2) := by
    exact premise_1
  -- chapter_21_line_9: GL tag implication.
  have row_9 : (N v1) := by
    apply row_15
    exact row_10
  -- chapter_21_line_8: GL tag implication.
  have row_8 : (gl_existence1 N v1 zero add) := by
    apply row_13
    exact row_9
    exact row_12
  -- chapter_21_line_7: GL tag expansion.
  have row_7 : (¬ (∀ (v3 : α), ((N v3) → (¬ (add v1 zero v3))))) := by
    simpa only [gl_existence1] using (row_8)
  have exists_row_7 : ∃ (v3 : α), ((N v3) ∧ (add v1 zero v3)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_7
      intro v3 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v3, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v3, witness_row_7⟩ := exists_row_7
  -- chapter_21_line_6: GL tag disintegration.
  have row_6 : (add v1 zero v3) := by
    exact witness_row_7.2
  -- chapter_21_line_5: GL tag implication.
  have row_5 : (succ v3 v2) := by
    apply row_17
    exact row_12
    exact row_11
    exact row_6
    exact row_10
  -- chapter_21_line_4: GL tag theorem.
  have row_4 := peano_source_011 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_21_line_3: GL tag implication.
  have row_3 : (v1 = v3) := by
    apply row_4
    exact row_6
    exact row_9
  -- chapter_21_line_2: GL tag symmetry of equality.
  have row_2 : (v3 = v1) := by
    exact Eq.symm row_3
  -- chapter_21_line_1: GL tag equality1.
  have row_1 : (succ v1 v2) := by
    have equality_source := row_5
    have equality_step_1 := row_2
    cases equality_step_1
    exact equality_source
  exact row_1

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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v2 : α)
    (v4 : α)
    (v6 : α)
    (assumption_2 : (add v2 v6 v4))
    : (N v6) := by
  -- chapter_27_line_10: GL tag task formulation.
  have row_10 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_27_line_9: GL tag expansion.
  have row_9 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_10)
  -- chapter_27_line_8: GL tag disintegration.
  have row_8 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_9.1
  -- chapter_27_line_7: GL tag expansion.
  have row_7 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_8)
  -- chapter_27_line_6: GL tag disintegration.
  have row_6 : (gl_fXYZ add N N N) := by
    exact row_7.1.1.1.1.1.1.1.1.1.2
  -- chapter_27_line_5: GL tag expansion.
  have row_5 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_6)
  -- chapter_27_line_4: GL tag disintegration.
  have row_4 : (gl_implication12 add N) := by
    exact row_5.1.1.1.2
  -- chapter_27_line_3: GL tag expansion.
  have row_3 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_4)
  -- chapter_27_line_2: GL tag task formulation.
  have row_2 : (add v2 v6 v4) := by
    exact assumption_2
  -- chapter_27_line_1: GL tag implication.
  have row_1 : (N v6) := by
    apply row_3
    exact row_2
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (v4 : α)
    (v5 : α)
    (v6 : α)
    (assumption_43 : (v6 = zero))
    (assumption_42 : (add v2 v6 v4))
    (assumption_20 : (add v1 v2 v3))
    (assumption_7 : (add v1 v4 v5))
    : (add v3 v6 v5) := by
  -- chapter_28_line_43: GL tag recursion.
  have row_43 : (v6 = zero) := by
    exact assumption_43
  -- chapter_28_line_42: GL tag task formulation.
  have row_42 : (add v2 v6 v4) := by
    exact assumption_42
  -- chapter_28_line_41: GL tag equality1.
  have row_41 : (add v2 zero v4) := by
    have equality_source := row_42
    have equality_step_1 := row_43
    cases equality_step_1
    exact equality_source
  -- chapter_28_line_40: GL tag task formulation.
  have row_40 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_28_line_39: GL tag expansion.
  have row_39 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_40)
  -- chapter_28_line_38: GL tag disintegration.
  have row_38 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_39.1
  -- chapter_28_line_37: GL tag expansion.
  have row_37 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_38)
  -- chapter_28_line_36: GL tag disintegration.
  have row_36 : (gl_fXYZ add N N N) := by
    exact row_37.1.1.1.1.1.1.1.1.1.2
  -- chapter_28_line_35: GL tag expansion.
  have row_35 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_36)
  -- chapter_28_line_34: GL tag disintegration.
  have row_34 : (gl_implication13 add N) := by
    exact row_35.1.1.2
  -- chapter_28_line_33: GL tag expansion.
  have row_33 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_34)
  -- chapter_28_line_32: GL tag disintegration.
  have row_32 : (gl_implication12 add N) := by
    exact row_35.1.1.1.2
  -- chapter_28_line_31: GL tag expansion.
  have row_31 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_32)
  -- chapter_28_line_30: GL tag disintegration.
  have row_30 : (gl_implication11 add N) := by
    exact row_35.1.1.1.1
  -- chapter_28_line_29: GL tag expansion.
  have row_29 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_30)
  -- chapter_28_line_28: GL tag disintegration.
  have row_28 : (gl_implication18 N N N add) := by
    exact row_35.1.2
  -- chapter_28_line_27: GL tag expansion.
  have row_27 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication18] using (row_28)
  -- chapter_28_line_26: GL tag disintegration.
  have row_26 : (gl_implication19 N N add) := by
    exact row_35.2
  -- chapter_28_line_25: GL tag expansion.
  have row_25 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (∀ (w3 : α), ((add w1 w2 w3) → (∀ (w4 : α), ((add w1 w2 w4) → (w3 = w4))))))))) := by
    simpa only [gl_implication19] using (row_26)
  -- chapter_28_line_24: GL tag implication grouping.
  have row_24 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_29
    have auxiliary_source_2 := row_31
    have auxiliary_source_3 := row_33
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_28_line_23: GL tag implication.
  have row_23 : (gl_and21 v2 N v6 N v4 N) := by
    apply row_24
    exact row_42
  -- chapter_28_line_22: GL tag expansion.
  have row_22 : (((N v2) ∧ (N v6)) ∧ (N v4)) := by
    simpa only [gl_and21] using (row_23)
  -- chapter_28_line_21: GL tag disintegration.
  have row_21 : (N v6) := by
    exact row_22.1.2
  -- chapter_28_line_20: GL tag task formulation.
  have row_20 : (add v1 v2 v3) := by
    exact assumption_20
  -- chapter_28_line_19: GL tag implication.
  have row_19 : (gl_and21 v1 N v2 N v3 N) := by
    apply row_24
    exact row_20
  -- chapter_28_line_18: GL tag expansion.
  have row_18 : (((N v1) ∧ (N v2)) ∧ (N v3)) := by
    simpa only [gl_and21] using (row_19)
  -- chapter_28_line_17: GL tag disintegration.
  have row_17 : (N v2) := by
    exact row_18.1.2
  -- chapter_28_line_16: GL tag disintegration.
  have row_16 : (N v3) := by
    exact row_18.2
  -- chapter_28_line_15: GL tag theorem.
  have row_15 := peano_source_011 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_28_line_14: GL tag implication.
  have row_14 : (v2 = v4) := by
    apply row_15
    exact row_41
    exact row_17
  -- chapter_28_line_13: GL tag implication.
  have row_13 : (gl_existence1 N v3 v6 add) := by
    apply row_27
    exact row_16
    exact row_21
  -- chapter_28_line_12: GL tag expansion.
  have row_12 : (¬ (∀ (v7 : α), ((N v7) → (¬ (add v3 v6 v7))))) := by
    simpa only [gl_existence1] using (row_13)
  have exists_row_12 : ∃ (v7 : α), ((N v7) ∧ (add v3 v6 v7)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_12
      intro v7 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v7, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v7, witness_row_12⟩ := exists_row_12
  -- chapter_28_line_11: GL tag disintegration.
  have row_11 : (add v3 v6 v7) := by
    exact witness_row_12.2
  -- chapter_28_line_10: GL tag equality1.
  have row_10 : (add v3 zero v7) := by
    have equality_source := row_11
    have equality_step_1 := row_43
    cases equality_step_1
    exact equality_source
  -- chapter_28_line_9: GL tag implication.
  have row_9 : (v3 = v7) := by
    apply row_15
    exact row_10
    exact row_16
  -- chapter_28_line_8: GL tag equality1.
  have row_8 : (add v1 v4 v7) := by
    have equality_source := row_20
    have equality_step_1 := row_14
    cases equality_step_1
    have equality_step_2 := row_9
    cases equality_step_2
    exact equality_source
  -- chapter_28_line_7: GL tag task formulation.
  have row_7 : (add v1 v4 v5) := by
    exact assumption_7
  -- chapter_28_line_6: GL tag implication.
  have row_6 : (gl_and21 v1 N v4 N v5 N) := by
    apply row_24
    exact row_7
  -- chapter_28_line_5: GL tag expansion.
  have row_5 : (((N v1) ∧ (N v4)) ∧ (N v5)) := by
    simpa only [gl_and21] using (row_6)
  -- chapter_28_line_4: GL tag disintegration.
  have row_4 : (N v1) := by
    exact row_5.1.1
  -- chapter_28_line_3: GL tag disintegration.
  have row_3 : (N v4) := by
    exact row_5.1.2
  -- chapter_28_line_2: GL tag implication.
  have row_2 : (v7 = v5) := by
    apply row_25
    exact row_4
    exact row_3
    exact row_8
    exact row_7
  -- chapter_28_line_1: GL tag equality1.
  have row_1 : (add v3 v6 v5) := by
    have equality_source := row_11
    have equality_step_1 := row_2
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (v4 : α)
    (v5 : α)
    (v6 : α)
    (assumption_26 : (succ previous v6))
    (assumption_23 : (add v1 v4 v5))
    (assumption_19 : (add v1 v2 v3))
    (assumption_8 : ((add v1 v2 v3) → (∀ (w1 : α) (w2 : α), ((add v1 w1 w2) → ((add v2 previous w1) → (add v3 previous w2))))))
    (assumption_6 : (add v2 v6 v4))
    : (add v3 v6 v5) := by
  -- chapter_29_line_46: GL tag task formulation.
  have row_46 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_29_line_45: GL tag expansion.
  have row_45 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_46)
  -- chapter_29_line_44: GL tag disintegration.
  have row_44 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_45.1
  -- chapter_29_line_43: GL tag expansion.
  have row_43 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_44)
  -- chapter_29_line_42: GL tag disintegration.
  have row_42 : (gl_fXY succ N N) := by
    exact row_43.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_29_line_41: GL tag expansion.
  have row_41 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_42)
  -- chapter_29_line_40: GL tag disintegration.
  have row_40 : (gl_implication0 succ N) := by
    exact row_41.1.1.1
  -- chapter_29_line_39: GL tag expansion.
  have row_39 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_40)
  -- chapter_29_line_38: GL tag disintegration.
  have row_38 : (gl_fXYZ add N N N) := by
    exact row_43.1.1.1.1.1.1.1.1.1.2
  -- chapter_29_line_37: GL tag expansion.
  have row_37 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_38)
  -- chapter_29_line_36: GL tag disintegration.
  have row_36 : (gl_implication13 add N) := by
    exact row_37.1.1.2
  -- chapter_29_line_35: GL tag expansion.
  have row_35 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_36)
  -- chapter_29_line_34: GL tag disintegration.
  have row_34 : (gl_implication12 add N) := by
    exact row_37.1.1.1.2
  -- chapter_29_line_33: GL tag expansion.
  have row_33 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_34)
  -- chapter_29_line_32: GL tag disintegration.
  have row_32 : (gl_implication11 add N) := by
    exact row_37.1.1.1.1
  -- chapter_29_line_31: GL tag expansion.
  have row_31 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_32)
  -- chapter_29_line_30: GL tag disintegration.
  have row_30 : (gl_implication18 N N N add) := by
    exact row_37.1.2
  -- chapter_29_line_29: GL tag expansion.
  have row_29 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication18] using (row_30)
  -- chapter_29_line_28: GL tag disintegration.
  have row_28 : (gl_implication23 N succ add) := by
    exact row_43.1.1.1.1.1.1.2
  -- chapter_29_line_27: GL tag expansion.
  have row_27 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication23] using (row_28)
  -- chapter_29_line_26: GL tag recursion.
  have row_26 : (succ previous v6) := by
    exact assumption_26
  -- chapter_29_line_25: GL tag implication.
  have row_25 : (N previous) := by
    apply row_39
    exact row_26
  -- chapter_29_line_24: GL tag implication grouping.
  have row_24 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_31
    have auxiliary_source_2 := row_33
    have auxiliary_source_3 := row_35
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_29_line_23: GL tag task formulation.
  have row_23 : (add v1 v4 v5) := by
    exact assumption_23
  -- chapter_29_line_22: GL tag implication.
  have row_22 : (gl_and21 v1 N v4 N v5 N) := by
    apply row_24
    exact row_23
  -- chapter_29_line_21: GL tag expansion.
  have row_21 : (((N v1) ∧ (N v4)) ∧ (N v5)) := by
    simpa only [gl_and21] using (row_22)
  -- chapter_29_line_20: GL tag disintegration.
  have row_20 : (N v1) := by
    exact row_21.1.1
  -- chapter_29_line_19: GL tag task formulation.
  have row_19 : (add v1 v2 v3) := by
    exact assumption_19
  -- chapter_29_line_18: GL tag implication.
  have row_18 : (gl_and21 v1 N v2 N v3 N) := by
    apply row_24
    exact row_19
  -- chapter_29_line_17: GL tag expansion.
  have row_17 : (((N v1) ∧ (N v2)) ∧ (N v3)) := by
    simpa only [gl_and21] using (row_18)
  -- chapter_29_line_16: GL tag disintegration.
  have row_16 : (N v2) := by
    exact row_17.1.2
  -- chapter_29_line_15: GL tag implication.
  have row_15 : (gl_existence1 N v2 previous add) := by
    apply row_29
    exact row_16
    exact row_25
  -- chapter_29_line_14: GL tag expansion.
  have row_14 : (¬ (∀ (v8 : α), ((N v8) → (¬ (add v2 previous v8))))) := by
    simpa only [gl_existence1] using (row_15)
  have exists_row_14 : ∃ (v8 : α), ((N v8) ∧ (add v2 previous v8)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_14
      intro v8 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v8, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v8, witness_row_14⟩ := exists_row_14
  -- chapter_29_line_13: GL tag disintegration.
  have row_13 : (N v8) := by
    exact witness_row_14.1
  -- chapter_29_line_12: GL tag disintegration.
  have row_12 : (add v2 previous v8) := by
    exact witness_row_14.2
  -- chapter_29_line_11: GL tag implication.
  have row_11 : (gl_existence1 N v1 v8 add) := by
    apply row_29
    exact row_20
    exact row_13
  -- chapter_29_line_10: GL tag expansion.
  have row_10 : (¬ (∀ (v7 : α), ((N v7) → (¬ (add v1 v8 v7))))) := by
    simpa only [gl_existence1] using (row_11)
  have exists_row_10 : ∃ (v7 : α), ((N v7) ∧ (add v1 v8 v7)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_10
      intro v7 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v7, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v7, witness_row_10⟩ := exists_row_10
  -- chapter_29_line_9: GL tag disintegration.
  have row_9 : (add v1 v8 v7) := by
    exact witness_row_10.2
  -- chapter_29_line_8: GL tag recursion.
  have row_8 : ((add v1 v2 v3) → (∀ (w1 : α) (w2 : α), ((add v1 w1 w2) → ((add v2 previous w1) → (add v3 previous w2))))) := by
    exact assumption_8
  -- chapter_29_line_7: GL tag implication.
  have row_7 : (add v3 previous v7) := by
    apply row_8
    exact row_19
    exact row_9
    exact row_12
  -- chapter_29_line_6: GL tag task formulation.
  have row_6 : (add v2 v6 v4) := by
    exact assumption_6
  -- chapter_29_line_5: GL tag implication.
  have row_5 : (succ v8 v4) := by
    apply row_27
    exact row_25
    exact row_26
    exact row_12
    exact row_6
  -- chapter_29_line_4: GL tag theorem.
  have row_4 := peano_source_017 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_29_line_3: GL tag implication.
  have row_3 : (succ v7 v5) := by
    apply row_4
    exact row_23
    exact row_9
    exact row_5
  -- chapter_29_line_2: GL tag theorem.
  have row_2 := peano_source_018 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_29_line_1: GL tag implication.
  have row_1 : (add v3 v6 v5) := by
    apply row_2
    exact row_7
    exact row_26
    exact row_3
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α) (v5 : α), ((add v1 v4 v5) → (∀ (v6 : α), ((add v2 v6 v4) → (add v3 v6 v5))))))) := by
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
    have typingRule := peano_source_019_induction_typing N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact typingRule v2 v4 v6 premise_3
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α) (v5 : α), ((add v1 v4 v5) → ((add v2 zero v4) → (add v3 zero v5)))))) := by
    intro v1
    intro v2
    intro v3
    intro base_premise_1
    intro v4
    intro v5
    intro base_premise_2
    intro base_premise_3
    have zeroRule := peano_source_019_check_zero N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact zeroRule v1 v2 v3 v4 v5 zero rfl base_premise_3 base_premise_1 base_premise_2
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α) (v5 : α), ((add v1 v4 v5) → ((add v2 induction_n v4) → (add v3 induction_n v5)))))) → ∀ induction_m, succ induction_n induction_m → (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α) (v5 : α), ((add v1 v4 v5) → ((add v2 induction_m v4) → (add v3 induction_m v5)))))) := by
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
        (succ induction_n induction_m) := by
      exact induction_successor
    have step_induction_assumption_2 :
        ((add v1 v2 v3) → (∀ (w1 : α) (w2 : α), ((add v1 w1 w2) → ((add v2 induction_n w1) → (add v3 induction_n w2))))) := by
      intro step_induction_assumption_2_premise_1
      intro w1
      intro w2
      intro step_induction_assumption_2_premise_2
      intro step_induction_assumption_2_premise_3
      apply induction_hypothesis
      all_goals assumption
    have stepRule := peano_source_019_check_induction_condition N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact stepRule induction_n v1 v2 v3 v4 v5 induction_m step_induction_assumption_1 step_premise_2 step_premise_1 step_induction_assumption_2 step_premise_3
  have inductionProperty : (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α) (v5 : α), ((add v1 v4 v5) → ((add v2 v6 v4) → (add v3 v6 v5)))))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α) (v5 : α), ((add v1 v4 v5) → ((add v2 induction_value v4) → (add v3 induction_value v5)))))))
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v2 : α)
    (v5 : α)
    (v6 : α)
    (assumption_2 : (add v5 v2 v6))
    : (N v2) := by
  -- chapter_36_line_10: GL tag task formulation.
  have row_10 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_36_line_9: GL tag expansion.
  have row_9 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_10)
  -- chapter_36_line_8: GL tag disintegration.
  have row_8 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_9.1
  -- chapter_36_line_7: GL tag expansion.
  have row_7 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_8)
  -- chapter_36_line_6: GL tag disintegration.
  have row_6 : (gl_fXYZ add N N N) := by
    exact row_7.1.1.1.1.1.1.1.1.1.2
  -- chapter_36_line_5: GL tag expansion.
  have row_5 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_6)
  -- chapter_36_line_4: GL tag disintegration.
  have row_4 : (gl_implication12 add N) := by
    exact row_5.1.1.1.2
  -- chapter_36_line_3: GL tag expansion.
  have row_3 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_4)
  -- chapter_36_line_2: GL tag task formulation.
  have row_2 : (add v5 v2 v6) := by
    exact assumption_2
  -- chapter_36_line_1: GL tag implication.
  have row_1 : (N v2) := by
    apply row_3
    exact row_2
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (v4 : α)
    (v5 : α)
    (v6 : α)
    (assumption_27 : (v2 = zero))
    (assumption_26 : (add v1 v2 v3))
    (assumption_24 : (add v5 v2 v6))
    (assumption_9 : (add v4 v5 v1))
    : (add v4 v6 v3) := by
  -- chapter_37_line_27: GL tag recursion.
  have row_27 : (v2 = zero) := by
    exact assumption_27
  -- chapter_37_line_26: GL tag task formulation.
  have row_26 : (add v1 v2 v3) := by
    exact assumption_26
  -- chapter_37_line_25: GL tag equality1.
  have row_25 : (add v1 zero v3) := by
    have equality_source := row_26
    have equality_step_1 := row_27
    cases equality_step_1
    exact equality_source
  -- chapter_37_line_24: GL tag task formulation.
  have row_24 : (add v5 v2 v6) := by
    exact assumption_24
  -- chapter_37_line_23: GL tag equality1.
  have row_23 : (add v5 zero v6) := by
    have equality_source := row_24
    have equality_step_1 := row_27
    cases equality_step_1
    exact equality_source
  -- chapter_37_line_22: GL tag task formulation.
  have row_22 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_37_line_21: GL tag expansion.
  have row_21 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_22)
  -- chapter_37_line_20: GL tag disintegration.
  have row_20 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_21.1
  -- chapter_37_line_19: GL tag expansion.
  have row_19 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_20)
  -- chapter_37_line_18: GL tag disintegration.
  have row_18 : (gl_fXYZ add N N N) := by
    exact row_19.1.1.1.1.1.1.1.1.1.2
  -- chapter_37_line_17: GL tag expansion.
  have row_17 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_18)
  -- chapter_37_line_16: GL tag disintegration.
  have row_16 : (gl_implication13 add N) := by
    exact row_17.1.1.2
  -- chapter_37_line_15: GL tag expansion.
  have row_15 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_16)
  -- chapter_37_line_14: GL tag disintegration.
  have row_14 : (gl_implication12 add N) := by
    exact row_17.1.1.1.2
  -- chapter_37_line_13: GL tag expansion.
  have row_13 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_14)
  -- chapter_37_line_12: GL tag disintegration.
  have row_12 : (gl_implication11 add N) := by
    exact row_17.1.1.1.1
  -- chapter_37_line_11: GL tag expansion.
  have row_11 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_12)
  -- chapter_37_line_10: GL tag implication grouping.
  have row_10 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_11
    have auxiliary_source_2 := row_13
    have auxiliary_source_3 := row_15
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_37_line_9: GL tag task formulation.
  have row_9 : (add v4 v5 v1) := by
    exact assumption_9
  -- chapter_37_line_8: GL tag implication.
  have row_8 : (gl_and21 v4 N v5 N v1 N) := by
    apply row_10
    exact row_9
  -- chapter_37_line_7: GL tag expansion.
  have row_7 : (((N v4) ∧ (N v5)) ∧ (N v1)) := by
    simpa only [gl_and21] using (row_8)
  -- chapter_37_line_6: GL tag disintegration.
  have row_6 : (N v1) := by
    exact row_7.2
  -- chapter_37_line_5: GL tag disintegration.
  have row_5 : (N v5) := by
    exact row_7.1.2
  -- chapter_37_line_4: GL tag theorem.
  have row_4 := peano_source_011 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_37_line_3: GL tag implication.
  have row_3 : (v1 = v3) := by
    apply row_4
    exact row_25
    exact row_6
  -- chapter_37_line_2: GL tag implication.
  have row_2 : (v5 = v6) := by
    apply row_4
    exact row_23
    exact row_5
  -- chapter_37_line_1: GL tag equality1.
  have row_1 : (add v4 v6 v3) := by
    have equality_source := row_9
    have equality_step_1 := row_2
    cases equality_step_1
    have equality_step_2 := row_3
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (v4 : α)
    (v5 : α)
    (v6 : α)
    (assumption_22 : (succ previous v2))
    (assumption_19 : (add v4 v5 v1))
    (assumption_8 : (∀ (w1 : α), ((add v1 previous w1) → ((add v4 v5 v1) → (∀ (w2 : α), ((add v5 previous w2) → (add v4 w2 w1)))))))
    (assumption_6 : (add v1 v2 v3))
    (assumption_4 : (add v5 v2 v6))
    : (add v4 v6 v3) := by
  -- chapter_38_line_42: GL tag task formulation.
  have row_42 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_38_line_41: GL tag expansion.
  have row_41 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_42)
  -- chapter_38_line_40: GL tag disintegration.
  have row_40 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_41.1
  -- chapter_38_line_39: GL tag expansion.
  have row_39 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_40)
  -- chapter_38_line_38: GL tag disintegration.
  have row_38 : (gl_fXY succ N N) := by
    exact row_39.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_38_line_37: GL tag expansion.
  have row_37 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_38)
  -- chapter_38_line_36: GL tag disintegration.
  have row_36 : (gl_implication0 succ N) := by
    exact row_37.1.1.1
  -- chapter_38_line_35: GL tag expansion.
  have row_35 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_36)
  -- chapter_38_line_34: GL tag disintegration.
  have row_34 : (gl_fXYZ add N N N) := by
    exact row_39.1.1.1.1.1.1.1.1.1.2
  -- chapter_38_line_33: GL tag expansion.
  have row_33 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_34)
  -- chapter_38_line_32: GL tag disintegration.
  have row_32 : (gl_implication13 add N) := by
    exact row_33.1.1.2
  -- chapter_38_line_31: GL tag expansion.
  have row_31 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_32)
  -- chapter_38_line_30: GL tag disintegration.
  have row_30 : (gl_implication12 add N) := by
    exact row_33.1.1.1.2
  -- chapter_38_line_29: GL tag expansion.
  have row_29 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_30)
  -- chapter_38_line_28: GL tag disintegration.
  have row_28 : (gl_implication11 add N) := by
    exact row_33.1.1.1.1
  -- chapter_38_line_27: GL tag expansion.
  have row_27 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_28)
  -- chapter_38_line_26: GL tag disintegration.
  have row_26 : (gl_implication18 N N N add) := by
    exact row_33.1.2
  -- chapter_38_line_25: GL tag expansion.
  have row_25 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication18] using (row_26)
  -- chapter_38_line_24: GL tag disintegration.
  have row_24 : (gl_implication23 N succ add) := by
    exact row_39.1.1.1.1.1.1.2
  -- chapter_38_line_23: GL tag expansion.
  have row_23 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication23] using (row_24)
  -- chapter_38_line_22: GL tag recursion.
  have row_22 : (succ previous v2) := by
    exact assumption_22
  -- chapter_38_line_21: GL tag implication.
  have row_21 : (N previous) := by
    apply row_35
    exact row_22
  -- chapter_38_line_20: GL tag implication grouping.
  have row_20 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_27
    have auxiliary_source_2 := row_29
    have auxiliary_source_3 := row_31
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_38_line_19: GL tag task formulation.
  have row_19 : (add v4 v5 v1) := by
    exact assumption_19
  -- chapter_38_line_18: GL tag implication.
  have row_18 : (gl_and21 v4 N v5 N v1 N) := by
    apply row_20
    exact row_19
  -- chapter_38_line_17: GL tag expansion.
  have row_17 : (((N v4) ∧ (N v5)) ∧ (N v1)) := by
    simpa only [gl_and21] using (row_18)
  -- chapter_38_line_16: GL tag disintegration.
  have row_16 : (N v1) := by
    exact row_17.2
  -- chapter_38_line_15: GL tag disintegration.
  have row_15 : (N v5) := by
    exact row_17.1.2
  -- chapter_38_line_14: GL tag implication.
  have row_14 : (gl_existence1 N v1 previous add) := by
    apply row_25
    exact row_16
    exact row_21
  -- chapter_38_line_13: GL tag expansion.
  have row_13 : (¬ (∀ (v8 : α), ((N v8) → (¬ (add v1 previous v8))))) := by
    simpa only [gl_existence1] using (row_14)
  have exists_row_13 : ∃ (v8 : α), ((N v8) ∧ (add v1 previous v8)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_13
      intro v8 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v8, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v8, witness_row_13⟩ := exists_row_13
  -- chapter_38_line_12: GL tag disintegration.
  have row_12 : (add v1 previous v8) := by
    exact witness_row_13.2
  -- chapter_38_line_11: GL tag implication.
  have row_11 : (gl_existence1 N v5 previous add) := by
    apply row_25
    exact row_15
    exact row_21
  -- chapter_38_line_10: GL tag expansion.
  have row_10 : (¬ (∀ (v7 : α), ((N v7) → (¬ (add v5 previous v7))))) := by
    simpa only [gl_existence1] using (row_11)
  have exists_row_10 : ∃ (v7 : α), ((N v7) ∧ (add v5 previous v7)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_10
      intro v7 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v7, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v7, witness_row_10⟩ := exists_row_10
  -- chapter_38_line_9: GL tag disintegration.
  have row_9 : (add v5 previous v7) := by
    exact witness_row_10.2
  -- chapter_38_line_8: GL tag recursion.
  have row_8 : (∀ (w1 : α), ((add v1 previous w1) → ((add v4 v5 v1) → (∀ (w2 : α), ((add v5 previous w2) → (add v4 w2 w1)))))) := by
    exact assumption_8
  -- chapter_38_line_7: GL tag implication.
  have row_7 : (add v4 v7 v8) := by
    apply row_8
    exact row_12
    exact row_19
    exact row_9
  -- chapter_38_line_6: GL tag task formulation.
  have row_6 : (add v1 v2 v3) := by
    exact assumption_6
  -- chapter_38_line_5: GL tag implication.
  have row_5 : (succ v8 v3) := by
    apply row_23
    exact row_21
    exact row_22
    exact row_12
    exact row_6
  -- chapter_38_line_4: GL tag task formulation.
  have row_4 : (add v5 v2 v6) := by
    exact assumption_4
  -- chapter_38_line_3: GL tag implication.
  have row_3 : (succ v7 v6) := by
    apply row_23
    exact row_21
    exact row_22
    exact row_9
    exact row_4
  -- chapter_38_line_2: GL tag theorem.
  have row_2 := peano_source_018 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_38_line_1: GL tag implication.
  have row_1 : (add v4 v6 v3) := by
    apply row_2
    exact row_7
    exact row_3
    exact row_5
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
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
    have typingRule := peano_source_022_induction_typing N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact typingRule v2 v5 v6 premise_3
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v1 : α) (v3 : α), ((add v1 zero v3) → (∀ (v4 : α) (v5 : α), ((add v4 v5 v1) → (∀ (v6 : α), ((add v5 zero v6) → (add v4 v6 v3))))))) := by
    intro v1
    intro v3
    intro base_premise_1
    intro v4
    intro v5
    intro base_premise_2
    intro v6
    intro base_premise_3
    have zeroRule := peano_source_022_check_zero N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact zeroRule v1 zero v3 v4 v5 v6 rfl base_premise_1 base_premise_3 base_premise_2
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
        (succ induction_n induction_m) := by
      exact induction_successor
    have step_induction_assumption_2 :
        (∀ (w1 : α), ((add v1 induction_n w1) → ((add v4 v5 v1) → (∀ (w2 : α), ((add v5 induction_n w2) → (add v4 w2 w1)))))) := by
      intro w1
      intro step_induction_assumption_2_premise_1
      intro step_induction_assumption_2_premise_2
      intro w2
      intro step_induction_assumption_2_premise_3
      apply induction_hypothesis
      all_goals assumption
    have stepRule := peano_source_022_check_induction_condition N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact stepRule induction_n v1 induction_m v3 v4 v5 v6 step_induction_assumption_1 step_premise_2 step_induction_assumption_2 step_premise_1 step_premise_3
  have inductionProperty : (∀ (v1 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α) (v5 : α), ((add v4 v5 v1) → (∀ (v6 : α), ((add v5 v2 v6) → (add v4 v6 v3))))))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v1 : α) (v3 : α), ((add v1 induction_value v3) → (∀ (v4 : α) (v5 : α), ((add v4 v5 v1) → (∀ (v6 : α), ((add v5 induction_value v6) → (add v4 v6 v3))))))))
      v2
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v1 v3 premise_1 v4 v5 premise_2 v6 premise_3

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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (add v1 one v2))) := by
  intro v1
  intro v2
  intro premise_1
  -- chapter_45_line_25: GL tag task formulation.
  have row_25 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_45_line_24: GL tag expansion.
  have row_24 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_25)
  -- chapter_45_line_23: GL tag disintegration.
  have row_23 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_24.1
  -- chapter_45_line_22: GL tag expansion.
  have row_22 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_23)
  -- chapter_45_line_21: GL tag disintegration.
  have row_21 : (gl_fXY succ N N) := by
    exact row_22.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_45_line_20: GL tag expansion.
  have row_20 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_21)
  -- chapter_45_line_19: GL tag disintegration.
  have row_19 : (gl_implication0 succ N) := by
    exact row_20.1.1.1
  -- chapter_45_line_18: GL tag expansion.
  have row_18 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_19)
  -- chapter_45_line_17: GL tag disintegration.
  have row_17 : (gl_fXYZ add N N N) := by
    exact row_22.1.1.1.1.1.1.1.1.1.2
  -- chapter_45_line_16: GL tag expansion.
  have row_16 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_17)
  -- chapter_45_line_15: GL tag disintegration.
  have row_15 : (gl_implication18 N N N add) := by
    exact row_16.1.2
  -- chapter_45_line_14: GL tag expansion.
  have row_14 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication18] using (row_15)
  -- chapter_45_line_13: GL tag disintegration.
  have row_13 : (N zero) := by
    exact row_22.1.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_45_line_12: GL tag disintegration.
  have row_12 : (gl_implication24 N succ add) := by
    exact row_22.1.1.1.1.1.2
  -- chapter_45_line_11: GL tag expansion.
  have row_11 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((succ w4 w5) → (add w3 w2 w5))))))))) := by
    simpa only [gl_implication24] using (row_12)
  -- chapter_45_line_10: GL tag disintegration.
  have row_10 : (succ zero one) := by
    exact row_24.2
  -- chapter_45_line_9: GL tag task formulation.
  have row_9 : (succ v1 v2) := by
    exact premise_1
  -- chapter_45_line_8: GL tag implication.
  have row_8 : (N v1) := by
    apply row_18
    exact row_9
  -- chapter_45_line_7: GL tag implication.
  have row_7 : (gl_existence1 N v1 zero add) := by
    apply row_14
    exact row_8
    exact row_13
  -- chapter_45_line_6: GL tag expansion.
  have row_6 : (¬ (∀ (v8 : α), ((N v8) → (¬ (add v1 zero v8))))) := by
    simpa only [gl_existence1] using (row_7)
  have exists_row_6 : ∃ (v8 : α), ((N v8) ∧ (add v1 zero v8)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_6
      intro v8 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v8, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v8, witness_row_6⟩ := exists_row_6
  -- chapter_45_line_5: GL tag disintegration.
  have row_5 : (add v1 zero v8) := by
    exact witness_row_6.2
  -- chapter_45_line_4: GL tag theorem.
  have row_4 := peano_source_011 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_45_line_3: GL tag implication.
  have row_3 : (v1 = v8) := by
    apply row_4
    exact row_5
    exact row_8
  -- chapter_45_line_2: GL tag equality1.
  have row_2 : (succ v8 v2) := by
    have equality_source := row_9
    have equality_step_1 := row_3
    cases equality_step_1
    exact equality_source
  -- chapter_45_line_1: GL tag implication.
  have row_1 : (add v1 one v2) := by
    apply row_11
    exact row_13
    exact row_10
    exact row_5
    exact row_2
  exact row_1

private theorem peano_source_030_induction_typing
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (assumption_13 : (v1 = v2))
    (assumption_11 : (mul v2 v1 zero))
    : (N v1) := by
  -- chapter_52_line_13: GL tag task formulation.
  have row_13 : (v1 = v2) := by
    exact assumption_13
  -- chapter_52_line_12: GL tag symmetry of equality.
  have row_12 : (v2 = v1) := by
    exact Eq.symm row_13
  -- chapter_52_line_11: GL tag task formulation.
  have row_11 : (mul v2 v1 zero) := by
    exact assumption_11
  -- chapter_52_line_10: GL tag equality1.
  have row_10 : (mul v1 v2 zero) := by
    have equality_source := row_11
    have equality_step_1 := row_13
    cases equality_step_1
    have equality_step_2 := row_12
    cases equality_step_2
    exact equality_source
  -- chapter_52_line_9: GL tag task formulation.
  have row_9 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_52_line_8: GL tag expansion.
  have row_8 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_9)
  -- chapter_52_line_7: GL tag disintegration.
  have row_7 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_8.1
  -- chapter_52_line_6: GL tag expansion.
  have row_6 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_7)
  -- chapter_52_line_5: GL tag disintegration.
  have row_5 : (gl_fXYZ mul N N N) := by
    exact row_6.1.1.1.1.2
  -- chapter_52_line_4: GL tag expansion.
  have row_4 : (((((gl_implication11 mul N) ∧ (gl_implication12 mul N)) ∧ (gl_implication13 mul N)) ∧ (gl_implication18 N N N mul)) ∧ (gl_implication19 N N mul)) := by
    simpa only [gl_fXYZ] using (row_5)
  -- chapter_52_line_3: GL tag disintegration.
  have row_3 : (gl_implication11 mul N) := by
    exact row_4.1.1.1.1
  -- chapter_52_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_3)
  -- chapter_52_line_1: GL tag implication.
  have row_1 : (N v1) := by
    apply row_2
    exact row_10
  exact row_1

private theorem peano_source_030_check_zero
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (assumption_4 : (v1 = zero))
    (assumption_2 : (v1 = v2))
    : (zero = v2) := by
  -- chapter_53_line_4: GL tag recursion.
  have row_4 : (v1 = zero) := by
    exact assumption_4
  -- chapter_53_line_3: GL tag symmetry of equality.
  have row_3 : (zero = v1) := by
    exact Eq.symm row_4
  -- chapter_53_line_2: GL tag task formulation.
  have row_2 : (v1 = v2) := by
    exact assumption_2
  -- chapter_53_line_1: GL tag equality2.
  have row_1 : (zero = v2) := by
    exact Eq.trans row_3 row_2
  exact row_1

private theorem peano_source_030_check_induction_condition
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (assumption_31 : (mul v2 v1 zero))
    (assumption_30 : (v1 = v2))
    (assumption_11 : (succ previous v1))
    : (zero = v2) := by
  -- chapter_54_line_31: GL tag task formulation.
  have row_31 : (mul v2 v1 zero) := by
    exact assumption_31
  -- chapter_54_line_30: GL tag task formulation.
  have row_30 : (v1 = v2) := by
    exact assumption_30
  -- chapter_54_line_29: GL tag symmetry of equality.
  have row_29 : (v2 = v1) := by
    exact Eq.symm row_30
  -- chapter_54_line_28: GL tag equality1.
  have row_28 : (mul v1 v2 zero) := by
    have equality_source := row_31
    have equality_step_1 := row_29
    cases equality_step_1
    have equality_step_2 := row_30
    cases equality_step_2
    exact equality_source
  -- chapter_54_line_27: GL tag task formulation.
  have row_27 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_54_line_26: GL tag expansion.
  have row_26 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_27)
  -- chapter_54_line_25: GL tag disintegration.
  have row_25 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_26.1
  -- chapter_54_line_24: GL tag expansion.
  have row_24 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_25)
  -- chapter_54_line_23: GL tag disintegration.
  have row_23 : (gl_fXY succ N N) := by
    exact row_24.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_54_line_22: GL tag expansion.
  have row_22 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_23)
  -- chapter_54_line_21: GL tag disintegration.
  have row_21 : (gl_implication0 succ N) := by
    exact row_22.1.1.1
  -- chapter_54_line_20: GL tag expansion.
  have row_20 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_21)
  -- chapter_54_line_19: GL tag disintegration.
  have row_19 : (gl_fXYZ mul N N N) := by
    exact row_24.1.1.1.1.2
  -- chapter_54_line_18: GL tag expansion.
  have row_18 : (((((gl_implication11 mul N) ∧ (gl_implication12 mul N)) ∧ (gl_implication13 mul N)) ∧ (gl_implication18 N N N mul)) ∧ (gl_implication19 N N mul)) := by
    simpa only [gl_fXYZ] using (row_19)
  -- chapter_54_line_17: GL tag disintegration.
  have row_17 : (gl_implication11 mul N) := by
    exact row_18.1.1.1.1
  -- chapter_54_line_16: GL tag expansion.
  have row_16 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_17)
  -- chapter_54_line_15: GL tag disintegration.
  have row_15 : (gl_implication18 N N N mul) := by
    exact row_18.1.2
  -- chapter_54_line_14: GL tag expansion.
  have row_14 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication18] using (row_15)
  -- chapter_54_line_13: GL tag disintegration.
  have row_13 : (gl_implication28 N succ mul add) := by
    exact row_24.2
  -- chapter_54_line_12: GL tag expansion.
  have row_12 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((mul w3 w1 w4) → (∀ (w5 : α), ((mul w3 w2 w5) → (add w4 w3 w5))))))))) := by
    simpa only [gl_implication28] using (row_13)
  -- chapter_54_line_11: GL tag recursion.
  have row_11 : (succ previous v1) := by
    exact assumption_11
  -- chapter_54_line_10: GL tag implication.
  have row_10 : (N previous) := by
    apply row_20
    exact row_11
  -- chapter_54_line_9: GL tag implication.
  have row_9 : (N v1) := by
    apply row_16
    exact row_28
  -- chapter_54_line_8: GL tag implication.
  have row_8 : (gl_existence1 N v1 previous mul) := by
    apply row_14
    exact row_9
    exact row_10
  -- chapter_54_line_7: GL tag expansion.
  have row_7 : (¬ (∀ (v3 : α), ((N v3) → (¬ (mul v1 previous v3))))) := by
    simpa only [gl_existence1] using (row_8)
  have exists_row_7 : ∃ (v3 : α), ((N v3) ∧ (mul v1 previous v3)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_7
      intro v3 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v3, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v3, witness_row_7⟩ := exists_row_7
  -- chapter_54_line_6: GL tag disintegration.
  have row_6 : (mul v1 previous v3) := by
    exact witness_row_7.2
  -- chapter_54_line_5: GL tag disintegration.
  have row_5 : (N v3) := by
    exact witness_row_7.1
  -- chapter_54_line_4: GL tag equality1.
  have row_4 : (mul v2 previous v3) := by
    have equality_source := row_6
    have equality_step_1 := row_30
    cases equality_step_1
    exact equality_source
  -- chapter_54_line_3: GL tag implication.
  have row_3 : (add v3 v2 zero) := by
    apply row_12
    exact row_10
    exact row_11
    exact row_4
    exact row_31
  -- chapter_54_line_2: GL tag theorem.
  have row_2 := peano_source_029 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_54_line_1: GL tag implication.
  have row_1 : (zero = v2) := by
    apply row_2
    exact row_3
    exact row_5
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((v1 = v2) → ((mul v2 v1 zero) → (zero = v2)))) := by
  intro v1
  intro v2
  intro premise_1
  intro premise_2
  have inductionMember : N v1 := by
    have typingRule := peano_source_030_induction_typing N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact typingRule v1 v2 premise_1 premise_2
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v2 : α), ((zero = v2) → ((mul v2 zero zero) → (zero = v2)))) := by
    intro v2
    intro base_premise_1
    intro base_premise_2
    have zeroRule := peano_source_030_check_zero N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact zeroRule zero v2 rfl base_premise_1
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
    have stepRule := peano_source_030_check_induction_condition N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact stepRule induction_n induction_m v2 step_premise_2 step_premise_1 step_induction_assumption_1
  have inductionProperty : (∀ (v2 : α), ((v1 = v2) → ((mul v2 v1 zero) → (zero = v2)))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v2 : α), ((induction_value = v2) → ((mul v2 induction_value zero) → (zero = v2)))))
      v1
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v2 premise_1 premise_2

private theorem peano_source_031_induction_typing
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v2 : α)
    (v3 : α)
    (v4 : α)
    (assumption_2 : (add v4 v2 v3))
    : (N v2) := by
  -- chapter_55_line_10: GL tag task formulation.
  have row_10 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_55_line_9: GL tag expansion.
  have row_9 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_10)
  -- chapter_55_line_8: GL tag disintegration.
  have row_8 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_9.1
  -- chapter_55_line_7: GL tag expansion.
  have row_7 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_8)
  -- chapter_55_line_6: GL tag disintegration.
  have row_6 : (gl_fXYZ add N N N) := by
    exact row_7.1.1.1.1.1.1.1.1.1.2
  -- chapter_55_line_5: GL tag expansion.
  have row_5 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_6)
  -- chapter_55_line_4: GL tag disintegration.
  have row_4 : (gl_implication12 add N) := by
    exact row_5.1.1.1.2
  -- chapter_55_line_3: GL tag expansion.
  have row_3 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_4)
  -- chapter_55_line_2: GL tag task formulation.
  have row_2 : (add v4 v2 v3) := by
    exact assumption_2
  -- chapter_55_line_1: GL tag implication.
  have row_1 : (N v2) := by
    apply row_3
    exact row_2
  exact row_1

private theorem peano_source_031_check_zero
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (v4 : α)
    (assumption_20 : (add v4 v2 v3))
    (assumption_16 : (add v1 v2 v3))
    (assumption_12 : (v2 = zero))
    : (v1 = v4) := by
  -- chapter_56_line_35: GL tag task formulation.
  have row_35 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_56_line_34: GL tag expansion.
  have row_34 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_35)
  -- chapter_56_line_33: GL tag disintegration.
  have row_33 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_34.1
  -- chapter_56_line_32: GL tag expansion.
  have row_32 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_33)
  -- chapter_56_line_31: GL tag disintegration.
  have row_31 : (gl_fXYZ add N N N) := by
    exact row_32.1.1.1.1.1.1.1.1.1.2
  -- chapter_56_line_30: GL tag expansion.
  have row_30 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_31)
  -- chapter_56_line_29: GL tag disintegration.
  have row_29 : (gl_implication13 add N) := by
    exact row_30.1.1.2
  -- chapter_56_line_28: GL tag expansion.
  have row_28 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_29)
  -- chapter_56_line_27: GL tag disintegration.
  have row_27 : (gl_implication12 add N) := by
    exact row_30.1.1.1.2
  -- chapter_56_line_26: GL tag expansion.
  have row_26 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_27)
  -- chapter_56_line_25: GL tag disintegration.
  have row_25 : (gl_implication11 add N) := by
    exact row_30.1.1.1.1
  -- chapter_56_line_24: GL tag expansion.
  have row_24 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_25)
  -- chapter_56_line_23: GL tag disintegration.
  have row_23 : (gl_implication21 N zero add) := by
    exact row_32.1.1.1.1.1.1.1.1.2
  -- chapter_56_line_22: GL tag expansion.
  have row_22 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((add w1 zero w2) → (w1 = w2))))) := by
    simpa only [gl_implication21] using (row_23)
  -- chapter_56_line_21: GL tag implication grouping.
  have row_21 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_24
    have auxiliary_source_2 := row_26
    have auxiliary_source_3 := row_28
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_56_line_20: GL tag task formulation.
  have row_20 : (add v4 v2 v3) := by
    exact assumption_20
  -- chapter_56_line_19: GL tag implication.
  have row_19 : (gl_and21 v4 N v2 N v3 N) := by
    apply row_21
    exact row_20
  -- chapter_56_line_18: GL tag expansion.
  have row_18 : (((N v4) ∧ (N v2)) ∧ (N v3)) := by
    simpa only [gl_and21] using (row_19)
  -- chapter_56_line_17: GL tag disintegration.
  have row_17 : (N v4) := by
    exact row_18.1.1
  -- chapter_56_line_16: GL tag task formulation.
  have row_16 : (add v1 v2 v3) := by
    exact assumption_16
  -- chapter_56_line_15: GL tag implication.
  have row_15 : (gl_and21 v1 N v2 N v3 N) := by
    apply row_21
    exact row_16
  -- chapter_56_line_14: GL tag expansion.
  have row_14 : (((N v1) ∧ (N v2)) ∧ (N v3)) := by
    simpa only [gl_and21] using (row_15)
  -- chapter_56_line_13: GL tag disintegration.
  have row_13 : (N v1) := by
    exact row_14.1.1
  -- chapter_56_line_12: GL tag recursion.
  have row_12 : (v2 = zero) := by
    exact assumption_12
  -- chapter_56_line_11: GL tag equality1.
  have row_11 : (add v1 zero v3) := by
    have equality_source := row_16
    have equality_step_1 := row_12
    cases equality_step_1
    exact equality_source
  -- chapter_56_line_10: GL tag equality1.
  have row_10 : (add v4 zero v3) := by
    have equality_source := row_20
    have equality_step_1 := row_12
    cases equality_step_1
    exact equality_source
  -- chapter_56_line_9: GL tag theorem.
  have row_9 := peano_source_011 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_56_line_8: GL tag implication.
  have row_8 : (v1 = v3) := by
    apply row_9
    exact row_11
    exact row_13
  -- chapter_56_line_7: GL tag symmetry of equality.
  have row_7 : (v3 = v1) := by
    exact Eq.symm row_8
  -- chapter_56_line_6: GL tag equality1.
  have row_6 : (add v4 zero v1) := by
    have equality_source := row_10
    have equality_step_1 := row_7
    cases equality_step_1
    exact equality_source
  -- chapter_56_line_5: GL tag implication.
  have row_5 : (v4 = v3) := by
    apply row_9
    exact row_10
    exact row_17
  -- chapter_56_line_4: GL tag symmetry of equality.
  have row_4 : (v3 = v4) := by
    exact Eq.symm row_5
  -- chapter_56_line_3: GL tag implication.
  have row_3 : (v4 = v1) := by
    apply row_22
    exact row_17
    exact row_6
  -- chapter_56_line_2: GL tag equality1.
  have row_2 : (add v1 zero v4) := by
    have equality_source := row_10
    have equality_step_1 := row_3
    cases equality_step_1
    have equality_step_2 := row_4
    cases equality_step_2
    exact equality_source
  -- chapter_56_line_1: GL tag implication.
  have row_1 : (v1 = v4) := by
    apply row_22
    exact row_13
    exact row_2
  exact row_1

private theorem peano_source_031_check_induction_condition
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (v4 : α)
    (assumption_38 : (add v4 v2 v3))
    (assumption_37 : (add v1 v2 v3))
    (assumption_17 : (succ previous v2))
    (assumption_2 : (∀ (w1 : α), ((add v1 previous w1) → ((add v4 previous w1) → (v1 = v4)))))
    : (v1 = v4) := by
  -- chapter_57_line_38: GL tag task formulation.
  have row_38 : (add v4 v2 v3) := by
    exact assumption_38
  -- chapter_57_line_37: GL tag task formulation.
  have row_37 : (add v1 v2 v3) := by
    exact assumption_37
  -- chapter_57_line_36: GL tag variable copy.
  have row_36 : (v3 = v3) := by
    rfl
  -- chapter_57_line_35: GL tag equality1.
  have row_35 : (add v1 v2 v3) := by
    have equality_source := row_37
    have equality_step_1 := row_36
    cases equality_step_1
    exact equality_source
  -- chapter_57_line_34: GL tag equality1.
  have row_34 : (add v4 v2 v3) := by
    have equality_source := row_38
    have equality_step_1 := row_36
    cases equality_step_1
    exact equality_source
  -- chapter_57_line_33: GL tag task formulation.
  have row_33 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_57_line_32: GL tag expansion.
  have row_32 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_33)
  -- chapter_57_line_31: GL tag disintegration.
  have row_31 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_32.1
  -- chapter_57_line_30: GL tag expansion.
  have row_30 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_31)
  -- chapter_57_line_29: GL tag disintegration.
  have row_29 : (gl_implication23 N succ add) := by
    exact row_30.1.1.1.1.1.1.2
  -- chapter_57_line_28: GL tag expansion.
  have row_28 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication23] using (row_29)
  -- chapter_57_line_27: GL tag disintegration.
  have row_27 : (gl_fXY succ N N) := by
    exact row_30.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_57_line_26: GL tag expansion.
  have row_26 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_27)
  -- chapter_57_line_25: GL tag disintegration.
  have row_25 : (gl_implication0 succ N) := by
    exact row_26.1.1.1
  -- chapter_57_line_24: GL tag expansion.
  have row_24 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_25)
  -- chapter_57_line_23: GL tag disintegration.
  have row_23 : (gl_fXYZ add N N N) := by
    exact row_30.1.1.1.1.1.1.1.1.1.2
  -- chapter_57_line_22: GL tag expansion.
  have row_22 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_23)
  -- chapter_57_line_21: GL tag disintegration.
  have row_21 : (gl_implication11 add N) := by
    exact row_22.1.1.1.1
  -- chapter_57_line_20: GL tag expansion.
  have row_20 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_21)
  -- chapter_57_line_19: GL tag disintegration.
  have row_19 : (gl_implication18 N N N add) := by
    exact row_22.1.2
  -- chapter_57_line_18: GL tag expansion.
  have row_18 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication18] using (row_19)
  -- chapter_57_line_17: GL tag recursion.
  have row_17 : (succ previous v2) := by
    exact assumption_17
  -- chapter_57_line_16: GL tag implication.
  have row_16 : (N previous) := by
    apply row_24
    exact row_17
  -- chapter_57_line_15: GL tag implication.
  have row_15 : (N v4) := by
    apply row_20
    exact row_34
  -- chapter_57_line_14: GL tag implication.
  have row_14 : (N v1) := by
    apply row_20
    exact row_35
  -- chapter_57_line_13: GL tag implication.
  have row_13 : (gl_existence1 N v1 previous add) := by
    apply row_18
    exact row_14
    exact row_16
  -- chapter_57_line_12: GL tag implication.
  have row_12 : (gl_existence1 N v4 previous add) := by
    apply row_18
    exact row_15
    exact row_16
  -- chapter_57_line_11: GL tag expansion.
  have row_11 : (¬ (∀ (v11 : α), ((N v11) → (¬ (add v1 previous v11))))) := by
    simpa only [gl_existence1] using (row_13)
  have exists_row_11 : ∃ (v11 : α), ((N v11) ∧ (add v1 previous v11)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_11
      intro v11 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v11, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v11, witness_row_11⟩ := exists_row_11
  -- chapter_57_line_10: GL tag disintegration.
  have row_10 : (add v1 previous v11) := by
    exact witness_row_11.2
  -- chapter_57_line_9: GL tag implication.
  have row_9 : (succ v11 v3) := by
    apply row_28
    exact row_16
    exact row_17
    exact row_10
    exact row_37
  -- chapter_57_line_8: GL tag expansion.
  have row_8 : (¬ (∀ (v6 : α), ((N v6) → (¬ (add v4 previous v6))))) := by
    simpa only [gl_existence1] using (row_12)
  have exists_row_8 : ∃ (v6 : α), ((N v6) ∧ (add v4 previous v6)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_8
      intro v6 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v6, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v6, witness_row_8⟩ := exists_row_8
  -- chapter_57_line_7: GL tag disintegration.
  have row_7 : (add v4 previous v6) := by
    exact witness_row_8.2
  -- chapter_57_line_6: GL tag implication.
  have row_6 : (succ v6 v3) := by
    apply row_28
    exact row_16
    exact row_17
    exact row_7
    exact row_38
  -- chapter_57_line_5: GL tag theorem.
  have row_5 := peano_source_006 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_57_line_4: GL tag implication.
  have row_4 : (v11 = v6) := by
    apply row_5
    exact row_9
    exact row_6
  -- chapter_57_line_3: GL tag equality1.
  have row_3 : (add v1 previous v6) := by
    have equality_source := row_10
    have equality_step_1 := row_4
    cases equality_step_1
    exact equality_source
  -- chapter_57_line_2: GL tag recursion.
  have row_2 : (∀ (w1 : α), ((add v1 previous w1) → ((add v4 previous w1) → (v1 = v4)))) := by
    exact assumption_2
  -- chapter_57_line_1: GL tag implication.
  have row_1 : (v1 = v4) := by
    apply row_2
    exact row_3
    exact row_7
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
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
    have typingRule := peano_source_031_induction_typing N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact typingRule v2 v3 v4 premise_2
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v1 : α) (v3 : α), ((add v1 zero v3) → (∀ (v4 : α), ((add v4 zero v3) → (v1 = v4))))) := by
    intro v1
    intro v3
    intro base_premise_1
    intro v4
    intro base_premise_2
    have zeroRule := peano_source_031_check_zero N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact zeroRule v1 zero v3 v4 base_premise_2 base_premise_1 rfl
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
    have stepRule := peano_source_031_check_induction_condition N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact stepRule induction_n v1 induction_m v3 v4 step_premise_2 step_premise_1 step_induction_assumption_1 step_induction_assumption_2
  have inductionProperty : (∀ (v1 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α), ((add v4 v2 v3) → (v1 = v4))))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v1 : α) (v3 : α), ((add v1 induction_value v3) → (∀ (v4 : α), ((add v4 induction_value v3) → (v1 = v4))))))
      v2
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v1 v3 premise_1 v4 premise_2

private theorem peano_source_032_induction_typing
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v1 : α)
    (assumption_1 : (N v1))
    : (N v1) := by
  -- chapter_58_line_1: GL tag task formulation.
  have row_1 : (N v1) := by
    exact assumption_1
  exact row_1

private theorem peano_source_032_check_zero
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (assumption_23 : (add zero v1 v2))
    (assumption_22 : (v1 = zero))
    : (v1 = v2) := by
  -- chapter_59_line_23: GL tag task formulation.
  have row_23 : (add zero v1 v2) := by
    exact assumption_23
  -- chapter_59_line_22: GL tag recursion.
  have row_22 : (v1 = zero) := by
    exact assumption_22
  -- chapter_59_line_21: GL tag symmetry of equality.
  have row_21 : (zero = v1) := by
    exact Eq.symm row_22
  -- chapter_59_line_20: GL tag equality1.
  have row_20 : (add v1 zero v2) := by
    have equality_source := row_23
    have equality_step_1 := row_21
    cases equality_step_1
    have equality_step_2 := row_22
    cases equality_step_2
    exact equality_source
  -- chapter_59_line_19: GL tag task formulation.
  have row_19 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_59_line_18: GL tag expansion.
  have row_18 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_19)
  -- chapter_59_line_17: GL tag disintegration.
  have row_17 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_18.1
  -- chapter_59_line_16: GL tag expansion.
  have row_16 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_17)
  -- chapter_59_line_15: GL tag disintegration.
  have row_15 : (gl_fXYZ add N N N) := by
    exact row_16.1.1.1.1.1.1.1.1.1.2
  -- chapter_59_line_14: GL tag expansion.
  have row_14 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_15)
  -- chapter_59_line_13: GL tag disintegration.
  have row_13 : (gl_implication13 add N) := by
    exact row_14.1.1.2
  -- chapter_59_line_12: GL tag expansion.
  have row_12 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_13)
  -- chapter_59_line_11: GL tag disintegration.
  have row_11 : (gl_implication12 add N) := by
    exact row_14.1.1.1.2
  -- chapter_59_line_10: GL tag expansion.
  have row_10 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_11)
  -- chapter_59_line_9: GL tag disintegration.
  have row_9 : (gl_implication11 add N) := by
    exact row_14.1.1.1.1
  -- chapter_59_line_8: GL tag expansion.
  have row_8 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_9)
  -- chapter_59_line_7: GL tag disintegration.
  have row_7 : (gl_implication21 N zero add) := by
    exact row_16.1.1.1.1.1.1.1.1.2
  -- chapter_59_line_6: GL tag expansion.
  have row_6 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((add w1 zero w2) → (w1 = w2))))) := by
    simpa only [gl_implication21] using (row_7)
  -- chapter_59_line_5: GL tag implication grouping.
  have row_5 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_8
    have auxiliary_source_2 := row_10
    have auxiliary_source_3 := row_12
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_59_line_4: GL tag implication.
  have row_4 : (gl_and21 zero N v1 N v2 N) := by
    apply row_5
    exact row_23
  -- chapter_59_line_3: GL tag expansion.
  have row_3 : (((N zero) ∧ (N v1)) ∧ (N v2)) := by
    simpa only [gl_and21] using (row_4)
  -- chapter_59_line_2: GL tag disintegration.
  have row_2 : (N v1) := by
    exact row_3.1.2
  -- chapter_59_line_1: GL tag implication.
  have row_1 : (v1 = v2) := by
    apply row_6
    exact row_2
    exact row_20
  exact row_1

private theorem peano_source_032_check_induction_condition
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (assumption_11 : (succ previous v1))
    (assumption_6 : (add zero v1 v2))
    (assumption_4 : ((N previous) → (∀ (w1 : α), ((add zero previous w1) → (previous = w1)))))
    : (v1 = v2) := by
  -- chapter_60_line_26: GL tag task formulation.
  have row_26 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_60_line_25: GL tag expansion.
  have row_25 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_26)
  -- chapter_60_line_24: GL tag disintegration.
  have row_24 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_25.1
  -- chapter_60_line_23: GL tag expansion.
  have row_23 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_24)
  -- chapter_60_line_22: GL tag disintegration.
  have row_22 : (gl_fXY succ N N) := by
    exact row_23.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_60_line_21: GL tag expansion.
  have row_21 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_22)
  -- chapter_60_line_20: GL tag disintegration.
  have row_20 : (gl_implication0 succ N) := by
    exact row_21.1.1.1
  -- chapter_60_line_19: GL tag expansion.
  have row_19 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_20)
  -- chapter_60_line_18: GL tag disintegration.
  have row_18 : (gl_fXYZ add N N N) := by
    exact row_23.1.1.1.1.1.1.1.1.1.2
  -- chapter_60_line_17: GL tag expansion.
  have row_17 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_18)
  -- chapter_60_line_16: GL tag disintegration.
  have row_16 : (gl_implication18 N N N add) := by
    exact row_17.1.2
  -- chapter_60_line_15: GL tag expansion.
  have row_15 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication18] using (row_16)
  -- chapter_60_line_14: GL tag disintegration.
  have row_14 : (N zero) := by
    exact row_23.1.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_60_line_13: GL tag disintegration.
  have row_13 : (gl_implication23 N succ add) := by
    exact row_23.1.1.1.1.1.1.2
  -- chapter_60_line_12: GL tag expansion.
  have row_12 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication23] using (row_13)
  -- chapter_60_line_11: GL tag recursion.
  have row_11 : (succ previous v1) := by
    exact assumption_11
  -- chapter_60_line_10: GL tag implication.
  have row_10 : (N previous) := by
    apply row_19
    exact row_11
  -- chapter_60_line_9: GL tag implication.
  have row_9 : (gl_existence1 N zero previous add) := by
    apply row_15
    exact row_14
    exact row_10
  -- chapter_60_line_8: GL tag expansion.
  have row_8 : (¬ (∀ (v3 : α), ((N v3) → (¬ (add zero previous v3))))) := by
    simpa only [gl_existence1] using (row_9)
  have exists_row_8 : ∃ (v3 : α), ((N v3) ∧ (add zero previous v3)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_8
      intro v3 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v3, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v3, witness_row_8⟩ := exists_row_8
  -- chapter_60_line_7: GL tag disintegration.
  have row_7 : (add zero previous v3) := by
    exact witness_row_8.2
  -- chapter_60_line_6: GL tag task formulation.
  have row_6 : (add zero v1 v2) := by
    exact assumption_6
  -- chapter_60_line_5: GL tag implication.
  have row_5 : (succ v3 v2) := by
    apply row_12
    exact row_10
    exact row_11
    exact row_7
    exact row_6
  -- chapter_60_line_4: GL tag recursion.
  have row_4 : ((N previous) → (∀ (w1 : α), ((add zero previous w1) → (previous = w1)))) := by
    exact assumption_4
  -- chapter_60_line_3: GL tag implication.
  have row_3 : (previous = v3) := by
    apply row_4
    exact row_10
    exact row_7
  -- chapter_60_line_2: GL tag theorem.
  have row_2 := peano_source_002 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_60_line_1: GL tag implication.
  have row_1 : (v1 = v2) := by
    apply row_2
    exact row_11
    exact row_5
    exact row_3
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α), ((N v1) → (∀ (v2 : α), ((add zero v1 v2) → (v1 = v2))))) := by
  intro v1
  intro premise_1
  intro v2
  intro premise_2
  have inductionMember : N v1 := by
    have typingRule := peano_source_032_induction_typing N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact typingRule v1 premise_1
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v2 : α), ((add zero zero v2) → (zero = v2))) := by
    intro v2
    intro base_premise_1
    have zeroRule := peano_source_032_check_zero N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact zeroRule zero v2 base_premise_1 rfl
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
    have stepRule := peano_source_032_check_induction_condition N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact stepRule induction_n induction_m v2 step_induction_assumption_1 step_premise_1 step_induction_assumption_2
  have inductionProperty : (∀ (v2 : α), ((add zero v1 v2) → (v1 = v2))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v2 : α), ((add zero induction_value v2) → (induction_value = v2))))
      v1
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v2 premise_2

private theorem peano_source_041_induction_typing
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v2 : α)
    (v5 : α)
    (assumption_2 : (succ v2 v5))
    : (N v2) := by
  -- chapter_77_line_10: GL tag task formulation.
  have row_10 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_77_line_9: GL tag expansion.
  have row_9 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_10)
  -- chapter_77_line_8: GL tag disintegration.
  have row_8 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_9.1
  -- chapter_77_line_7: GL tag expansion.
  have row_7 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_8)
  -- chapter_77_line_6: GL tag disintegration.
  have row_6 : (gl_fXY succ N N) := by
    exact row_7.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_77_line_5: GL tag expansion.
  have row_5 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_6)
  -- chapter_77_line_4: GL tag disintegration.
  have row_4 : (gl_implication0 succ N) := by
    exact row_5.1.1.1
  -- chapter_77_line_3: GL tag expansion.
  have row_3 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_4)
  -- chapter_77_line_2: GL tag task formulation.
  have row_2 : (succ v2 v5) := by
    exact assumption_2
  -- chapter_77_line_1: GL tag implication.
  have row_1 : (N v2) := by
    apply row_3
    exact row_2
  exact row_1

private theorem peano_source_041_check_zero
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (v4 : α)
    (v5 : α)
    (assumption_32 : (succ v4 v1))
    (assumption_27 : (add v1 v2 v3))
    (assumption_17 : (succ v2 v5))
    (assumption_12 : (v2 = zero))
    : (add v4 v5 v3) := by
  -- chapter_78_line_60: GL tag task formulation.
  have row_60 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_78_line_59: GL tag expansion.
  have row_59 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_60)
  -- chapter_78_line_58: GL tag disintegration.
  have row_58 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_59.1
  -- chapter_78_line_57: GL tag expansion.
  have row_57 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_58)
  -- chapter_78_line_56: GL tag disintegration.
  have row_56 : (gl_implication9 N zero succ) := by
    exact row_57.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_78_line_55: GL tag expansion.
  have row_55 : (∀ (w1 : α), ((N w1) → (¬ (succ w1 zero)))) := by
    simpa only [gl_implication9] using (row_56)
  -- chapter_78_line_54: GL tag disintegration.
  have row_54 : (gl_implication24 N succ add) := by
    exact row_57.1.1.1.1.1.2
  -- chapter_78_line_53: GL tag expansion.
  have row_53 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((succ w4 w5) → (add w3 w2 w5))))))))) := by
    simpa only [gl_implication24] using (row_54)
  -- chapter_78_line_52: GL tag disintegration.
  have row_52 : (gl_fXY succ N N) := by
    exact row_57.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_78_line_51: GL tag expansion.
  have row_51 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_52)
  -- chapter_78_line_50: GL tag disintegration.
  have row_50 : (gl_implication6 N N succ) := by
    exact row_51.1.2
  -- chapter_78_line_49: GL tag expansion.
  have row_49 : (∀ (w1 : α), ((N w1) → (gl_existence0 N w1 succ))) := by
    simpa only [gl_implication6] using (row_50)
  -- chapter_78_line_48: GL tag disintegration.
  have row_48 : (gl_implication1 succ N) := by
    exact row_51.1.1.2
  -- chapter_78_line_47: GL tag expansion.
  have row_47 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using (row_48)
  -- chapter_78_line_46: GL tag disintegration.
  have row_46 : (gl_implication0 succ N) := by
    exact row_51.1.1.1
  -- chapter_78_line_45: GL tag expansion.
  have row_45 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_46)
  -- chapter_78_line_44: GL tag disintegration.
  have row_44 : (gl_fXYZ add N N N) := by
    exact row_57.1.1.1.1.1.1.1.1.1.2
  -- chapter_78_line_43: GL tag expansion.
  have row_43 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_44)
  -- chapter_78_line_42: GL tag disintegration.
  have row_42 : (gl_implication13 add N) := by
    exact row_43.1.1.2
  -- chapter_78_line_41: GL tag expansion.
  have row_41 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_42)
  -- chapter_78_line_40: GL tag disintegration.
  have row_40 : (gl_implication12 add N) := by
    exact row_43.1.1.1.2
  -- chapter_78_line_39: GL tag expansion.
  have row_39 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_40)
  -- chapter_78_line_38: GL tag disintegration.
  have row_38 : (gl_implication11 add N) := by
    exact row_43.1.1.1.1
  -- chapter_78_line_37: GL tag expansion.
  have row_37 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_38)
  -- chapter_78_line_36: GL tag disintegration.
  have row_36 : (gl_implication18 N N N add) := by
    exact row_43.1.2
  -- chapter_78_line_35: GL tag expansion.
  have row_35 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication18] using (row_36)
  -- chapter_78_line_34: GL tag implication grouping.
  have row_34 : (∀ (w1 : α), ((N w1) → (gl_and3 N w1 succ zero))) := by
    have auxiliary_source_1 := row_49
    have auxiliary_source_2 := row_55
    intro auxiliary_argument_1
    intro auxiliary_premise
    exact ⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_premise)⟩
  -- chapter_78_line_33: GL tag implication grouping.
  have row_33 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (gl_and20 w1 N w2 N))) := by
    have auxiliary_source_1 := row_45
    have auxiliary_source_2 := row_47
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_premise
    exact ⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_premise)⟩
  -- chapter_78_line_32: GL tag task formulation.
  have row_32 : (succ v4 v1) := by
    exact assumption_32
  -- chapter_78_line_31: GL tag implication.
  have row_31 : (gl_and20 v4 N v1 N) := by
    apply row_33
    exact row_32
  -- chapter_78_line_30: GL tag expansion.
  have row_30 : ((N v4) ∧ (N v1)) := by
    simpa only [gl_and20] using (row_31)
  -- chapter_78_line_29: GL tag disintegration.
  have row_29 : (N v4) := by
    exact row_30.1
  -- chapter_78_line_28: GL tag implication grouping.
  have row_28 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_37
    have auxiliary_source_2 := row_39
    have auxiliary_source_3 := row_41
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_78_line_27: GL tag task formulation.
  have row_27 : (add v1 v2 v3) := by
    exact assumption_27
  -- chapter_78_line_26: GL tag implication.
  have row_26 : (gl_and21 v1 N v2 N v3 N) := by
    apply row_28
    exact row_27
  -- chapter_78_line_25: GL tag expansion.
  have row_25 : (((N v1) ∧ (N v2)) ∧ (N v3)) := by
    simpa only [gl_and21] using (row_26)
  -- chapter_78_line_24: GL tag disintegration.
  have row_24 : (N v3) := by
    exact row_25.2
  -- chapter_78_line_23: GL tag disintegration.
  have row_23 : (N v2) := by
    exact row_25.1.2
  -- chapter_78_line_22: GL tag implication.
  have row_22 : (gl_and3 N v3 succ zero) := by
    apply row_34
    exact row_24
  -- chapter_78_line_21: GL tag expansion.
  have row_21 : ((gl_existence0 N v3 succ) ∧ (¬ (succ v3 zero))) := by
    simpa only [gl_and3] using (row_22)
  -- chapter_78_line_20: GL tag disintegration.
  have row_20 : (gl_existence0 N v3 succ) := by
    exact row_21.1
  -- chapter_78_line_19: GL tag expansion.
  have row_19 : (¬ (∀ (v10 : α), ((N v10) → (¬ (succ v3 v10))))) := by
    simpa only [gl_existence0] using (row_20)
  have exists_row_19 : ∃ (v10 : α), ((N v10) ∧ (succ v3 v10)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_19
      intro v10 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v10, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v10, witness_row_19⟩ := exists_row_19
  -- chapter_78_line_18: GL tag disintegration.
  have row_18 : (succ v3 v10) := by
    exact witness_row_19.2
  -- chapter_78_line_17: GL tag task formulation.
  have row_17 : (succ v2 v5) := by
    exact assumption_17
  -- chapter_78_line_16: GL tag implication.
  have row_16 : (add v1 v5 v10) := by
    apply row_53
    exact row_23
    exact row_17
    exact row_27
    exact row_18
  -- chapter_78_line_15: GL tag implication.
  have row_15 : (gl_and21 v1 N v5 N v10 N) := by
    apply row_28
    exact row_16
  -- chapter_78_line_14: GL tag expansion.
  have row_14 : (((N v1) ∧ (N v5)) ∧ (N v10)) := by
    simpa only [gl_and21] using (row_15)
  -- chapter_78_line_13: GL tag disintegration.
  have row_13 : (N v1) := by
    exact row_14.1.1
  -- chapter_78_line_12: GL tag recursion.
  have row_12 : (v2 = zero) := by
    exact assumption_12
  -- chapter_78_line_11: GL tag equality1.
  have row_11 : (add v1 zero v3) := by
    have equality_source := row_27
    have equality_step_1 := row_12
    cases equality_step_1
    exact equality_source
  -- chapter_78_line_10: GL tag theorem.
  have row_10 := peano_source_011 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_78_line_9: GL tag implication.
  have row_9 : (v1 = v3) := by
    apply row_10
    exact row_11
    exact row_13
  -- chapter_78_line_8: GL tag implication.
  have row_8 : (gl_existence1 N v4 v2 add) := by
    apply row_35
    exact row_29
    exact row_23
  -- chapter_78_line_7: GL tag expansion.
  have row_7 : (¬ (∀ (v6 : α), ((N v6) → (¬ (add v4 v2 v6))))) := by
    simpa only [gl_existence1] using (row_8)
  have exists_row_7 : ∃ (v6 : α), ((N v6) ∧ (add v4 v2 v6)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_7
      intro v6 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v6, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v6, witness_row_7⟩ := exists_row_7
  -- chapter_78_line_6: GL tag disintegration.
  have row_6 : (add v4 v2 v6) := by
    exact witness_row_7.2
  -- chapter_78_line_5: GL tag equality1.
  have row_5 : (add v4 zero v6) := by
    have equality_source := row_6
    have equality_step_1 := row_12
    cases equality_step_1
    exact equality_source
  -- chapter_78_line_4: GL tag implication.
  have row_4 : (v4 = v6) := by
    apply row_10
    exact row_5
    exact row_29
  -- chapter_78_line_3: GL tag equality1.
  have row_3 : (succ v6 v3) := by
    have equality_source := row_32
    have equality_step_1 := row_4
    cases equality_step_1
    have equality_step_2 := row_9
    cases equality_step_2
    exact equality_source
  -- chapter_78_line_2: GL tag theorem.
  have row_2 := peano_source_018 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_78_line_1: GL tag implication.
  have row_1 : (add v4 v5 v3) := by
    apply row_2
    exact row_6
    exact row_17
    exact row_3
  exact row_1

private theorem peano_source_041_check_induction_condition
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (v4 : α)
    (v5 : α)
    (assumption_17 : (succ previous v2))
    (assumption_14 : (add v1 v2 v3))
    (assumption_7 : (succ v4 v1))
    (assumption_6 : (∀ (w1 : α), ((add v1 previous w1) → ((succ v4 v1) → (∀ (w2 : α), ((succ previous w2) → (add v4 w2 w1)))))))
    (assumption_3 : (succ v2 v5))
    : (add v4 v5 v3) := by
  -- chapter_79_line_37: GL tag task formulation.
  have row_37 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_79_line_36: GL tag expansion.
  have row_36 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_37)
  -- chapter_79_line_35: GL tag disintegration.
  have row_35 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_36.1
  -- chapter_79_line_34: GL tag expansion.
  have row_34 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_35)
  -- chapter_79_line_33: GL tag disintegration.
  have row_33 : (gl_fXY succ N N) := by
    exact row_34.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_79_line_32: GL tag expansion.
  have row_32 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_33)
  -- chapter_79_line_31: GL tag disintegration.
  have row_31 : (gl_implication0 succ N) := by
    exact row_32.1.1.1
  -- chapter_79_line_30: GL tag expansion.
  have row_30 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_31)
  -- chapter_79_line_29: GL tag disintegration.
  have row_29 : (gl_fXYZ add N N N) := by
    exact row_34.1.1.1.1.1.1.1.1.1.2
  -- chapter_79_line_28: GL tag expansion.
  have row_28 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_29)
  -- chapter_79_line_27: GL tag disintegration.
  have row_27 : (gl_implication13 add N) := by
    exact row_28.1.1.2
  -- chapter_79_line_26: GL tag expansion.
  have row_26 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_27)
  -- chapter_79_line_25: GL tag disintegration.
  have row_25 : (gl_implication12 add N) := by
    exact row_28.1.1.1.2
  -- chapter_79_line_24: GL tag expansion.
  have row_24 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_25)
  -- chapter_79_line_23: GL tag disintegration.
  have row_23 : (gl_implication11 add N) := by
    exact row_28.1.1.1.1
  -- chapter_79_line_22: GL tag expansion.
  have row_22 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_23)
  -- chapter_79_line_21: GL tag disintegration.
  have row_21 : (gl_implication18 N N N add) := by
    exact row_28.1.2
  -- chapter_79_line_20: GL tag expansion.
  have row_20 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication18] using (row_21)
  -- chapter_79_line_19: GL tag disintegration.
  have row_19 : (gl_implication23 N succ add) := by
    exact row_34.1.1.1.1.1.1.2
  -- chapter_79_line_18: GL tag expansion.
  have row_18 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication23] using (row_19)
  -- chapter_79_line_17: GL tag recursion.
  have row_17 : (succ previous v2) := by
    exact assumption_17
  -- chapter_79_line_16: GL tag implication.
  have row_16 : (N previous) := by
    apply row_30
    exact row_17
  -- chapter_79_line_15: GL tag implication grouping.
  have row_15 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_22
    have auxiliary_source_2 := row_24
    have auxiliary_source_3 := row_26
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_79_line_14: GL tag task formulation.
  have row_14 : (add v1 v2 v3) := by
    exact assumption_14
  -- chapter_79_line_13: GL tag implication.
  have row_13 : (gl_and21 v1 N v2 N v3 N) := by
    apply row_15
    exact row_14
  -- chapter_79_line_12: GL tag expansion.
  have row_12 : (((N v1) ∧ (N v2)) ∧ (N v3)) := by
    simpa only [gl_and21] using (row_13)
  -- chapter_79_line_11: GL tag disintegration.
  have row_11 : (N v1) := by
    exact row_12.1.1
  -- chapter_79_line_10: GL tag implication.
  have row_10 : (gl_existence1 N v1 previous add) := by
    apply row_20
    exact row_11
    exact row_16
  -- chapter_79_line_9: GL tag expansion.
  have row_9 : (¬ (∀ (v6 : α), ((N v6) → (¬ (add v1 previous v6))))) := by
    simpa only [gl_existence1] using (row_10)
  have exists_row_9 : ∃ (v6 : α), ((N v6) ∧ (add v1 previous v6)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_9
      intro v6 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v6, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v6, witness_row_9⟩ := exists_row_9
  -- chapter_79_line_8: GL tag disintegration.
  have row_8 : (add v1 previous v6) := by
    exact witness_row_9.2
  -- chapter_79_line_7: GL tag task formulation.
  have row_7 : (succ v4 v1) := by
    exact assumption_7
  -- chapter_79_line_6: GL tag recursion.
  have row_6 : (∀ (w1 : α), ((add v1 previous w1) → ((succ v4 v1) → (∀ (w2 : α), ((succ previous w2) → (add v4 w2 w1)))))) := by
    exact assumption_6
  -- chapter_79_line_5: GL tag implication.
  have row_5 : (add v4 v2 v6) := by
    apply row_6
    exact row_8
    exact row_7
    exact row_17
  -- chapter_79_line_4: GL tag implication.
  have row_4 : (succ v6 v3) := by
    apply row_18
    exact row_16
    exact row_17
    exact row_8
    exact row_14
  -- chapter_79_line_3: GL tag task formulation.
  have row_3 : (succ v2 v5) := by
    exact assumption_3
  -- chapter_79_line_2: GL tag theorem.
  have row_2 := peano_source_018 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_79_line_1: GL tag implication.
  have row_1 : (add v4 v5 v3) := by
    apply row_2
    exact row_5
    exact row_3
    exact row_4
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
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
    have typingRule := peano_source_041_induction_typing N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact typingRule v2 v5 premise_3
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v1 : α) (v3 : α), ((add v1 zero v3) → (∀ (v4 : α), ((succ v4 v1) → (∀ (v5 : α), ((succ zero v5) → (add v4 v5 v3))))))) := by
    intro v1
    intro v3
    intro base_premise_1
    intro v4
    intro base_premise_2
    intro v5
    intro base_premise_3
    have zeroRule := peano_source_041_check_zero N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact zeroRule v1 zero v3 v4 v5 base_premise_2 base_premise_1 base_premise_3 rfl
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
        (succ induction_n induction_m) := by
      exact induction_successor
    have step_induction_assumption_2 :
        (∀ (w1 : α), ((add v1 induction_n w1) → ((succ v4 v1) → (∀ (w2 : α), ((succ induction_n w2) → (add v4 w2 w1)))))) := by
      intro w1
      intro step_induction_assumption_2_premise_1
      intro step_induction_assumption_2_premise_2
      intro w2
      intro step_induction_assumption_2_premise_3
      apply induction_hypothesis
      all_goals assumption
    have stepRule := peano_source_041_check_induction_condition N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact stepRule induction_n v1 induction_m v3 v4 v5 step_induction_assumption_1 step_premise_1 step_premise_2 step_induction_assumption_2 step_premise_3
  have inductionProperty : (∀ (v1 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α), ((succ v4 v1) → (∀ (v5 : α), ((succ v2 v5) → (add v4 v5 v3))))))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v1 : α) (v3 : α), ((add v1 induction_value v3) → (∀ (v4 : α), ((succ v4 v1) → (∀ (v5 : α), ((succ induction_value v5) → (add v4 v5 v3))))))))
      v2
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v1 v3 premise_1 v4 premise_2 v5 premise_3

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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v1 : α)
    (assumption_1 : (N v1))
    : (N v1) := by
  -- chapter_96_line_1: GL tag task formulation.
  have row_1 : (N v1) := by
    exact assumption_1
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (assumption_22 : (mul zero v1 v2))
    (assumption_21 : (v1 = zero))
    : (zero = v2) := by
  -- chapter_97_line_22: GL tag task formulation.
  have row_22 : (mul zero v1 v2) := by
    exact assumption_22
  -- chapter_97_line_21: GL tag recursion.
  have row_21 : (v1 = zero) := by
    exact assumption_21
  -- chapter_97_line_20: GL tag symmetry of equality.
  have row_20 : (zero = v1) := by
    exact Eq.symm row_21
  -- chapter_97_line_19: GL tag equality1.
  have row_19 : (mul v1 zero v2) := by
    have equality_source := row_22
    have equality_step_1 := row_20
    cases equality_step_1
    have equality_step_2 := row_21
    cases equality_step_2
    exact equality_source
  -- chapter_97_line_18: GL tag task formulation.
  have row_18 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_97_line_17: GL tag expansion.
  have row_17 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_18)
  -- chapter_97_line_16: GL tag disintegration.
  have row_16 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_17.1
  -- chapter_97_line_15: GL tag expansion.
  have row_15 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_16)
  -- chapter_97_line_14: GL tag disintegration.
  have row_14 : (gl_fXYZ mul N N N) := by
    exact row_15.1.1.1.1.2
  -- chapter_97_line_13: GL tag expansion.
  have row_13 : (((((gl_implication11 mul N) ∧ (gl_implication12 mul N)) ∧ (gl_implication13 mul N)) ∧ (gl_implication18 N N N mul)) ∧ (gl_implication19 N N mul)) := by
    simpa only [gl_fXYZ] using (row_14)
  -- chapter_97_line_12: GL tag disintegration.
  have row_12 : (gl_implication13 mul N) := by
    exact row_13.1.1.2
  -- chapter_97_line_11: GL tag expansion.
  have row_11 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_12)
  -- chapter_97_line_10: GL tag disintegration.
  have row_10 : (gl_implication12 mul N) := by
    exact row_13.1.1.1.2
  -- chapter_97_line_9: GL tag expansion.
  have row_9 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_10)
  -- chapter_97_line_8: GL tag disintegration.
  have row_8 : (gl_implication11 mul N) := by
    exact row_13.1.1.1.1
  -- chapter_97_line_7: GL tag expansion.
  have row_7 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_8)
  -- chapter_97_line_6: GL tag implication grouping.
  have row_6 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_7
    have auxiliary_source_2 := row_9
    have auxiliary_source_3 := row_11
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_97_line_5: GL tag implication.
  have row_5 : (gl_and21 zero N v1 N v2 N) := by
    apply row_6
    exact row_22
  -- chapter_97_line_4: GL tag expansion.
  have row_4 : (((N zero) ∧ (N v1)) ∧ (N v2)) := by
    simpa only [gl_and21] using (row_5)
  -- chapter_97_line_3: GL tag disintegration.
  have row_3 : (N v1) := by
    exact row_4.1.2
  -- chapter_97_line_2: GL tag theorem.
  have row_2 := peano_source_025 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_97_line_1: GL tag implication.
  have row_1 : (zero = v2) := by
    apply row_2
    exact row_19
    exact row_3
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (assumption_21 : (succ previous v1))
    (assumption_18 : (mul zero v1 v2))
    (assumption_9 : ((N previous) → (∀ (w1 : α), ((mul zero previous w1) → (zero = w1)))))
    : (zero = v2) := by
  -- chapter_98_line_44: GL tag task formulation.
  have row_44 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_98_line_43: GL tag expansion.
  have row_43 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_44)
  -- chapter_98_line_42: GL tag disintegration.
  have row_42 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_43.1
  -- chapter_98_line_41: GL tag expansion.
  have row_41 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_42)
  -- chapter_98_line_40: GL tag disintegration.
  have row_40 : (gl_implication26 zero N mul) := by
    exact row_41.1.1.2
  -- chapter_98_line_39: GL tag expansion.
  have row_39 : (∀ (w1 : α), ((w1 = zero) → (∀ (w2 : α), ((N w2) → ((N w1) → (mul w2 zero w1)))))) := by
    simpa only [gl_implication26] using (row_40)
  -- chapter_98_line_38: GL tag disintegration.
  have row_38 : (gl_fXY succ N N) := by
    exact row_41.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_98_line_37: GL tag expansion.
  have row_37 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_38)
  -- chapter_98_line_36: GL tag disintegration.
  have row_36 : (gl_implication0 succ N) := by
    exact row_37.1.1.1
  -- chapter_98_line_35: GL tag expansion.
  have row_35 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_36)
  -- chapter_98_line_34: GL tag disintegration.
  have row_34 : (gl_fXYZ mul N N N) := by
    exact row_41.1.1.1.1.2
  -- chapter_98_line_33: GL tag expansion.
  have row_33 : (((((gl_implication11 mul N) ∧ (gl_implication12 mul N)) ∧ (gl_implication13 mul N)) ∧ (gl_implication18 N N N mul)) ∧ (gl_implication19 N N mul)) := by
    simpa only [gl_fXYZ] using (row_34)
  -- chapter_98_line_32: GL tag disintegration.
  have row_32 : (gl_implication13 mul N) := by
    exact row_33.1.1.2
  -- chapter_98_line_31: GL tag expansion.
  have row_31 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_32)
  -- chapter_98_line_30: GL tag disintegration.
  have row_30 : (gl_implication12 mul N) := by
    exact row_33.1.1.1.2
  -- chapter_98_line_29: GL tag expansion.
  have row_29 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_30)
  -- chapter_98_line_28: GL tag disintegration.
  have row_28 : (gl_implication11 mul N) := by
    exact row_33.1.1.1.1
  -- chapter_98_line_27: GL tag expansion.
  have row_27 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_28)
  -- chapter_98_line_26: GL tag disintegration.
  have row_26 : (gl_implication18 N N N mul) := by
    exact row_33.1.2
  -- chapter_98_line_25: GL tag expansion.
  have row_25 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication18] using (row_26)
  -- chapter_98_line_24: GL tag disintegration.
  have row_24 : (N zero) := by
    exact row_41.1.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_98_line_23: GL tag disintegration.
  have row_23 : (gl_implication28 N succ mul add) := by
    exact row_41.2
  -- chapter_98_line_22: GL tag expansion.
  have row_22 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((mul w3 w1 w4) → (∀ (w5 : α), ((mul w3 w2 w5) → (add w4 w3 w5))))))))) := by
    simpa only [gl_implication28] using (row_23)
  -- chapter_98_line_21: GL tag recursion.
  have row_21 : (succ previous v1) := by
    exact assumption_21
  -- chapter_98_line_20: GL tag implication.
  have row_20 : (N previous) := by
    apply row_35
    exact row_21
  -- chapter_98_line_19: GL tag implication grouping.
  have row_19 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_27
    have auxiliary_source_2 := row_29
    have auxiliary_source_3 := row_31
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_98_line_18: GL tag task formulation.
  have row_18 : (mul zero v1 v2) := by
    exact assumption_18
  -- chapter_98_line_17: GL tag implication.
  have row_17 : (gl_and21 zero N v1 N v2 N) := by
    apply row_19
    exact row_18
  -- chapter_98_line_16: GL tag expansion.
  have row_16 : (((N zero) ∧ (N v1)) ∧ (N v2)) := by
    simpa only [gl_and21] using (row_17)
  -- chapter_98_line_15: GL tag disintegration.
  have row_15 : (N v2) := by
    exact row_16.2
  -- chapter_98_line_14: GL tag implication.
  have row_14 : (gl_existence1 N zero previous mul) := by
    apply row_25
    exact row_24
    exact row_20
  -- chapter_98_line_13: GL tag expansion.
  have row_13 : (¬ (∀ (v3 : α), ((N v3) → (¬ (mul zero previous v3))))) := by
    simpa only [gl_existence1] using (row_14)
  have exists_row_13 : ∃ (v3 : α), ((N v3) ∧ (mul zero previous v3)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_13
      intro v3 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v3, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v3, witness_row_13⟩ := exists_row_13
  -- chapter_98_line_12: GL tag disintegration.
  have row_12 : (mul zero previous v3) := by
    exact witness_row_13.2
  -- chapter_98_line_11: GL tag disintegration.
  have row_11 : (N v3) := by
    exact witness_row_13.1
  -- chapter_98_line_10: GL tag implication.
  have row_10 : (add v3 zero v2) := by
    apply row_22
    exact row_20
    exact row_21
    exact row_12
    exact row_18
  -- chapter_98_line_9: GL tag recursion.
  have row_9 : ((N previous) → (∀ (w1 : α), ((mul zero previous w1) → (zero = w1)))) := by
    exact assumption_9
  -- chapter_98_line_8: GL tag implication.
  have row_8 : (zero = v3) := by
    apply row_9
    exact row_20
    exact row_12
  -- chapter_98_line_7: GL tag symmetry of equality.
  have row_7 : (v3 = zero) := by
    exact Eq.symm row_8
  -- chapter_98_line_6: GL tag implication.
  have row_6 : (mul v2 zero v3) := by
    apply row_39
    exact row_7
    exact row_15
    exact row_11
  -- chapter_98_line_5: GL tag theorem.
  have row_5 := peano_source_011 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_98_line_4: GL tag implication.
  have row_4 : (v3 = v2) := by
    apply row_5
    exact row_10
    exact row_11
  -- chapter_98_line_3: GL tag symmetry of equality.
  have row_3 : (v2 = v3) := by
    exact Eq.symm row_4
  -- chapter_98_line_2: GL tag theorem.
  have row_2 := peano_source_009 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_98_line_1: GL tag implication.
  have row_1 : (zero = v2) := by
    apply row_2
    exact row_3
    exact row_6
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α), ((N v1) → (∀ (v2 : α), ((mul zero v1 v2) → (zero = v2))))) := by
  intro v1
  intro premise_1
  intro v2
  intro premise_2
  have inductionMember : N v1 := by
    have typingRule := peano_source_048_induction_typing N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact typingRule v1 premise_1
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v2 : α), ((mul zero zero v2) → (zero = v2))) := by
    intro v2
    intro base_premise_1
    have zeroRule := peano_source_048_check_zero N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact zeroRule zero v2 base_premise_1 rfl
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
    have stepRule := peano_source_048_check_induction_condition N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact stepRule induction_n induction_m v2 step_induction_assumption_1 step_premise_1 step_induction_assumption_2
  have inductionProperty : (∀ (v2 : α), ((mul zero v1 v2) → (zero = v2))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v2 : α), ((mul zero induction_value v2) → (zero = v2))))
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (assumption_2 : (mul v1 zero v2))
    : (N v1) := by
  -- chapter_120_line_10: GL tag task formulation.
  have row_10 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_120_line_9: GL tag expansion.
  have row_9 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_10)
  -- chapter_120_line_8: GL tag disintegration.
  have row_8 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_9.1
  -- chapter_120_line_7: GL tag expansion.
  have row_7 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_8)
  -- chapter_120_line_6: GL tag disintegration.
  have row_6 : (gl_fXYZ mul N N N) := by
    exact row_7.1.1.1.1.2
  -- chapter_120_line_5: GL tag expansion.
  have row_5 : (((((gl_implication11 mul N) ∧ (gl_implication12 mul N)) ∧ (gl_implication13 mul N)) ∧ (gl_implication18 N N N mul)) ∧ (gl_implication19 N N mul)) := by
    simpa only [gl_fXYZ] using (row_6)
  -- chapter_120_line_4: GL tag disintegration.
  have row_4 : (gl_implication11 mul N) := by
    exact row_5.1.1.1.1
  -- chapter_120_line_3: GL tag expansion.
  have row_3 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_4)
  -- chapter_120_line_2: GL tag task formulation.
  have row_2 : (mul v1 zero v2) := by
    exact assumption_2
  -- chapter_120_line_1: GL tag implication.
  have row_1 : (N v1) := by
    apply row_3
    exact row_2
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (assumption_7 : (mul v1 zero v2))
    (assumption_3 : (v1 = zero))
    : (mul zero v1 v2) := by
  -- chapter_121_line_22: GL tag task formulation.
  have row_22 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_121_line_21: GL tag expansion.
  have row_21 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_22)
  -- chapter_121_line_20: GL tag disintegration.
  have row_20 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_21.1
  -- chapter_121_line_19: GL tag expansion.
  have row_19 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_20)
  -- chapter_121_line_18: GL tag disintegration.
  have row_18 : (gl_implication25 N zero mul) := by
    exact row_19.1.1.1.2
  -- chapter_121_line_17: GL tag expansion.
  have row_17 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((mul w1 zero w2) → (w2 = zero))))) := by
    simpa only [gl_implication25] using (row_18)
  -- chapter_121_line_16: GL tag disintegration.
  have row_16 : (gl_fXYZ mul N N N) := by
    exact row_19.1.1.1.1.2
  -- chapter_121_line_15: GL tag expansion.
  have row_15 : (((((gl_implication11 mul N) ∧ (gl_implication12 mul N)) ∧ (gl_implication13 mul N)) ∧ (gl_implication18 N N N mul)) ∧ (gl_implication19 N N mul)) := by
    simpa only [gl_fXYZ] using (row_16)
  -- chapter_121_line_14: GL tag disintegration.
  have row_14 : (gl_implication13 mul N) := by
    exact row_15.1.1.2
  -- chapter_121_line_13: GL tag expansion.
  have row_13 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_14)
  -- chapter_121_line_12: GL tag disintegration.
  have row_12 : (gl_implication12 mul N) := by
    exact row_15.1.1.1.2
  -- chapter_121_line_11: GL tag expansion.
  have row_11 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_12)
  -- chapter_121_line_10: GL tag disintegration.
  have row_10 : (gl_implication11 mul N) := by
    exact row_15.1.1.1.1
  -- chapter_121_line_9: GL tag expansion.
  have row_9 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_10)
  -- chapter_121_line_8: GL tag implication grouping.
  have row_8 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_9
    have auxiliary_source_2 := row_11
    have auxiliary_source_3 := row_13
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_121_line_7: GL tag task formulation.
  have row_7 : (mul v1 zero v2) := by
    exact assumption_7
  -- chapter_121_line_6: GL tag implication.
  have row_6 : (gl_and21 v1 N zero N v2 N) := by
    apply row_8
    exact row_7
  -- chapter_121_line_5: GL tag expansion.
  have row_5 : (((N v1) ∧ (N zero)) ∧ (N v2)) := by
    simpa only [gl_and21] using (row_6)
  -- chapter_121_line_4: GL tag disintegration.
  have row_4 : (N v1) := by
    exact row_5.1.1
  -- chapter_121_line_3: GL tag recursion.
  have row_3 : (v1 = zero) := by
    exact assumption_3
  -- chapter_121_line_2: GL tag symmetry of equality.
  have row_2 : (zero = v1) := by
    exact Eq.symm row_3
  -- chapter_121_line_1: GL tag equality1.
  have row_1 : (mul zero v1 v2) := by
    have equality_source := row_7
    have equality_step_1 := row_3
    cases equality_step_1
    have equality_step_2 := row_2
    cases equality_step_2
    exact equality_source
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (assumption_28 : (succ previous v1))
    (assumption_25 : (mul v1 zero v2))
    (assumption_18 : (∀ (w1 : α), ((mul previous zero w1) → (mul zero previous w1))))
    : (mul zero v1 v2) := by
  -- chapter_122_line_55: GL tag task formulation.
  have row_55 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_122_line_54: GL tag expansion.
  have row_54 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_55)
  -- chapter_122_line_53: GL tag disintegration.
  have row_53 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_54.1
  -- chapter_122_line_52: GL tag expansion.
  have row_52 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_53)
  -- chapter_122_line_51: GL tag disintegration.
  have row_51 : (gl_implication26 zero N mul) := by
    exact row_52.1.1.2
  -- chapter_122_line_50: GL tag expansion.
  have row_50 : (∀ (w1 : α), ((w1 = zero) → (∀ (w2 : α), ((N w2) → ((N w1) → (mul w2 zero w1)))))) := by
    simpa only [gl_implication26] using (row_51)
  -- chapter_122_line_49: GL tag disintegration.
  have row_49 : (gl_fXY succ N N) := by
    exact row_52.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_122_line_48: GL tag expansion.
  have row_48 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_49)
  -- chapter_122_line_47: GL tag disintegration.
  have row_47 : (gl_implication0 succ N) := by
    exact row_48.1.1.1
  -- chapter_122_line_46: GL tag expansion.
  have row_46 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_47)
  -- chapter_122_line_45: GL tag disintegration.
  have row_45 : (gl_implication1 succ N) := by
    exact row_48.1.1.2
  -- chapter_122_line_44: GL tag expansion.
  have row_44 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using (row_45)
  -- chapter_122_line_43: GL tag disintegration.
  have row_43 : (gl_implication25 N zero mul) := by
    exact row_52.1.1.1.2
  -- chapter_122_line_42: GL tag expansion.
  have row_42 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((mul w1 zero w2) → (w2 = zero))))) := by
    simpa only [gl_implication25] using (row_43)
  -- chapter_122_line_41: GL tag disintegration.
  have row_41 : (gl_fXYZ mul N N N) := by
    exact row_52.1.1.1.1.2
  -- chapter_122_line_40: GL tag expansion.
  have row_40 : (((((gl_implication11 mul N) ∧ (gl_implication12 mul N)) ∧ (gl_implication13 mul N)) ∧ (gl_implication18 N N N mul)) ∧ (gl_implication19 N N mul)) := by
    simpa only [gl_fXYZ] using (row_41)
  -- chapter_122_line_39: GL tag disintegration.
  have row_39 : (gl_implication13 mul N) := by
    exact row_40.1.1.2
  -- chapter_122_line_38: GL tag expansion.
  have row_38 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_39)
  -- chapter_122_line_37: GL tag disintegration.
  have row_37 : (gl_implication12 mul N) := by
    exact row_40.1.1.1.2
  -- chapter_122_line_36: GL tag expansion.
  have row_36 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_37)
  -- chapter_122_line_35: GL tag disintegration.
  have row_35 : (gl_implication11 mul N) := by
    exact row_40.1.1.1.1
  -- chapter_122_line_34: GL tag expansion.
  have row_34 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_35)
  -- chapter_122_line_33: GL tag disintegration.
  have row_33 : (gl_implication18 N N N mul) := by
    exact row_40.1.2
  -- chapter_122_line_32: GL tag expansion.
  have row_32 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication18] using (row_33)
  -- chapter_122_line_31: GL tag disintegration.
  have row_31 : (N zero) := by
    exact row_52.1.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_122_line_30: GL tag disintegration.
  have row_30 : (gl_implication19 N N mul) := by
    exact row_40.2
  -- chapter_122_line_29: GL tag expansion.
  have row_29 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (∀ (w3 : α), ((mul w1 w2 w3) → (∀ (w4 : α), ((mul w1 w2 w4) → (w3 = w4))))))))) := by
    simpa only [gl_implication19] using (row_30)
  -- chapter_122_line_28: GL tag recursion.
  have row_28 : (succ previous v1) := by
    exact assumption_28
  -- chapter_122_line_27: GL tag implication.
  have row_27 : (N v1) := by
    apply row_44
    exact row_28
  -- chapter_122_line_26: GL tag implication.
  have row_26 : (N previous) := by
    apply row_46
    exact row_28
  -- chapter_122_line_25: GL tag task formulation.
  have row_25 : (mul v1 zero v2) := by
    exact assumption_25
  -- chapter_122_line_24: GL tag implication.
  have row_24 : (v2 = zero) := by
    apply row_42
    exact row_27
    exact row_25
  -- chapter_122_line_23: GL tag implication grouping.
  have row_23 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_34
    have auxiliary_source_2 := row_36
    have auxiliary_source_3 := row_38
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_122_line_22: GL tag implication.
  have row_22 : (gl_and21 v1 N zero N v2 N) := by
    apply row_23
    exact row_25
  -- chapter_122_line_21: GL tag expansion.
  have row_21 : (((N v1) ∧ (N zero)) ∧ (N v2)) := by
    simpa only [gl_and21] using (row_22)
  -- chapter_122_line_20: GL tag disintegration.
  have row_20 : (N v2) := by
    exact row_21.2
  -- chapter_122_line_19: GL tag implication.
  have row_19 : (mul previous zero v2) := by
    apply row_50
    exact row_24
    exact row_26
    exact row_20
  -- chapter_122_line_18: GL tag recursion.
  have row_18 : (∀ (w1 : α), ((mul previous zero w1) → (mul zero previous w1))) := by
    exact assumption_18
  -- chapter_122_line_17: GL tag implication.
  have row_17 : (mul zero previous v2) := by
    apply row_18
    exact row_19
  -- chapter_122_line_16: GL tag implication.
  have row_16 : (gl_existence1 N zero previous mul) := by
    apply row_32
    exact row_31
    exact row_26
  -- chapter_122_line_15: GL tag expansion.
  have row_15 : (¬ (∀ (v6 : α), ((N v6) → (¬ (mul zero previous v6))))) := by
    simpa only [gl_existence1] using (row_16)
  have exists_row_15 : ∃ (v6 : α), ((N v6) ∧ (mul zero previous v6)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_15
      intro v6 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v6, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v6, witness_row_15⟩ := exists_row_15
  -- chapter_122_line_14: GL tag disintegration.
  have row_14 : (mul zero previous v6) := by
    exact witness_row_15.2
  -- chapter_122_line_13: GL tag disintegration.
  have row_13 : (N v6) := by
    exact witness_row_15.1
  -- chapter_122_line_12: GL tag implication.
  have row_12 : (v6 = v2) := by
    apply row_29
    exact row_31
    exact row_26
    exact row_14
    exact row_17
  -- chapter_122_line_11: GL tag implication.
  have row_11 : (gl_existence1 N zero v1 mul) := by
    apply row_32
    exact row_31
    exact row_27
  -- chapter_122_line_10: GL tag expansion.
  have row_10 : (¬ (∀ (v3 : α), ((N v3) → (¬ (mul zero v1 v3))))) := by
    simpa only [gl_existence1] using (row_11)
  have exists_row_10 : ∃ (v3 : α), ((N v3) ∧ (mul zero v1 v3)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_10
      intro v3 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v3, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v3, witness_row_10⟩ := exists_row_10
  -- chapter_122_line_9: GL tag disintegration.
  have row_9 : (mul zero v1 v3) := by
    exact witness_row_10.2
  -- chapter_122_line_8: GL tag anchor handling.
  have row_8 : (gl_AnchorPeano N zero succ add mul one) := by
    exact row_55
  -- chapter_122_line_7: GL tag theorem.
  have row_7 := peano_source_008 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_122_line_6: GL tag implication.
  have row_6 : (add v6 zero v3) := by
    apply row_7
    exact row_9
    exact row_14
    exact row_28
  -- chapter_122_line_5: GL tag theorem.
  have row_5 := peano_source_011 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_122_line_4: GL tag implication.
  have row_4 : (v6 = v3) := by
    apply row_5
    exact row_6
    exact row_13
  -- chapter_122_line_3: GL tag symmetry of equality.
  have row_3 : (v3 = v6) := by
    exact Eq.symm row_4
  -- chapter_122_line_2: GL tag equality2.
  have row_2 : (v3 = v2) := by
    exact Eq.trans row_3 row_12
  -- chapter_122_line_1: GL tag equality1.
  have row_1 : (mul zero v1 v2) := by
    have equality_source := row_9
    have equality_step_1 := row_2
    cases equality_step_1
    exact equality_source
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((mul v1 zero v2) → (mul zero v1 v2))) := by
  intro v1
  intro v2
  intro premise_1
  have inductionMember : N v1 := by
    have typingRule := peano_source_058_induction_typing N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact typingRule v1 v2 premise_1
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v2 : α), ((mul zero zero v2) → (mul zero zero v2))) := by
    intro v2
    intro base_premise_1
    have zeroRule := peano_source_058_check_zero N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact zeroRule zero v2 base_premise_1 rfl
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v2 : α), ((mul induction_n zero v2) → (mul zero induction_n v2))) → ∀ induction_m, succ induction_n induction_m → (∀ (v2 : α), ((mul induction_m zero v2) → (mul zero induction_m v2))) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro v2
    intro step_premise_1
    have step_induction_assumption_1 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have step_induction_assumption_2 :
        (∀ (w1 : α), ((mul induction_n zero w1) → (mul zero induction_n w1))) := by
      intro w1
      intro step_induction_assumption_2_premise_1
      apply induction_hypothesis
      all_goals assumption
    have stepRule := peano_source_058_check_induction_condition N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact stepRule induction_n induction_m v2 step_induction_assumption_1 step_premise_1 step_induction_assumption_2
  have inductionProperty : (∀ (v2 : α), ((mul v1 zero v2) → (mul zero v1 v2))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v2 : α), ((mul induction_value zero v2) → (mul zero induction_value v2))))
      v1
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v2 premise_1

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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α), ((N v1) → (gl_or1 N v1 succ zero))) := by
  intro v1
  intro premise_1
  have or_parent_1 := peano_source_062 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_133_line_1: GL tag or theorem.
  have row_1 : (∀ (v1 : α), ((N v1) → (gl_or1 N v1 succ zero))) := by
    classical
    intro v1
    intro or_parent_premise_1
    simp only [gl_or1]
    by_cases or_case_1 : (gl_inSequence N v1 succ)
    · exact Or.inl (or_case_1)
    ·
      exact Or.inr ((or_parent_1 v1 or_parent_premise_1 or_case_1))
  solve_by_elim [row_1]

theorem peano_source_064
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α), ((N v1) → (gl_or1 N v1 succ zero))) := by
  intro v1
  intro premise_1
  have or_parent_1 := peano_source_062 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_134_line_1: GL tag or theorem.
  have row_1 : (∀ (v1 : α), ((N v1) → (gl_or1 N v1 succ zero))) := by
    classical
    intro v1
    intro or_parent_premise_1
    simp only [gl_or1]
    by_cases or_case_1 : (gl_inSequence N v1 succ)
    · exact Or.inl (or_case_1)
    ·
      exact Or.inr ((or_parent_1 v1 or_parent_premise_1 or_case_1))
  solve_by_elim [row_1]

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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (assumption_2 : (mul one v2 v1))
    : (N v2) := by
  -- chapter_22_line_10: GL tag task formulation.
  have row_10 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_22_line_9: GL tag expansion.
  have row_9 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_10)
  -- chapter_22_line_8: GL tag disintegration.
  have row_8 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_9.1
  -- chapter_22_line_7: GL tag expansion.
  have row_7 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_8)
  -- chapter_22_line_6: GL tag disintegration.
  have row_6 : (gl_fXYZ mul N N N) := by
    exact row_7.1.1.1.1.2
  -- chapter_22_line_5: GL tag expansion.
  have row_5 : (((((gl_implication11 mul N) ∧ (gl_implication12 mul N)) ∧ (gl_implication13 mul N)) ∧ (gl_implication18 N N N mul)) ∧ (gl_implication19 N N mul)) := by
    simpa only [gl_fXYZ] using (row_6)
  -- chapter_22_line_4: GL tag disintegration.
  have row_4 : (gl_implication12 mul N) := by
    exact row_5.1.1.1.2
  -- chapter_22_line_3: GL tag expansion.
  have row_3 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_4)
  -- chapter_22_line_2: GL tag task formulation.
  have row_2 : (mul one v2 v1) := by
    exact assumption_2
  -- chapter_22_line_1: GL tag implication.
  have row_1 : (N v2) := by
    apply row_3
    exact row_2
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (assumption_14 : (mul one v2 v1))
    (assumption_8 : (v2 = zero))
    : (v1 = v2) := by
  -- chapter_23_line_34: GL tag task formulation.
  have row_34 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_23_line_33: GL tag expansion.
  have row_33 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_34)
  -- chapter_23_line_32: GL tag disintegration.
  have row_32 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_33.1
  -- chapter_23_line_31: GL tag expansion.
  have row_31 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_32)
  -- chapter_23_line_30: GL tag disintegration.
  have row_30 : (gl_implication25 N zero mul) := by
    exact row_31.1.1.1.2
  -- chapter_23_line_29: GL tag expansion.
  have row_29 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((mul w1 zero w2) → (w2 = zero))))) := by
    simpa only [gl_implication25] using (row_30)
  -- chapter_23_line_28: GL tag disintegration.
  have row_28 : (gl_fXYZ mul N N N) := by
    exact row_31.1.1.1.1.2
  -- chapter_23_line_27: GL tag expansion.
  have row_27 : (((((gl_implication11 mul N) ∧ (gl_implication12 mul N)) ∧ (gl_implication13 mul N)) ∧ (gl_implication18 N N N mul)) ∧ (gl_implication19 N N mul)) := by
    simpa only [gl_fXYZ] using (row_28)
  -- chapter_23_line_26: GL tag disintegration.
  have row_26 : (gl_implication13 mul N) := by
    exact row_27.1.1.2
  -- chapter_23_line_25: GL tag expansion.
  have row_25 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_26)
  -- chapter_23_line_24: GL tag disintegration.
  have row_24 : (gl_implication12 mul N) := by
    exact row_27.1.1.1.2
  -- chapter_23_line_23: GL tag expansion.
  have row_23 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_24)
  -- chapter_23_line_22: GL tag disintegration.
  have row_22 : (gl_implication11 mul N) := by
    exact row_27.1.1.1.1
  -- chapter_23_line_21: GL tag expansion.
  have row_21 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_22)
  -- chapter_23_line_20: GL tag disintegration.
  have row_20 : (gl_implication22 N zero add) := by
    exact row_31.1.1.1.1.1.1.1.2
  -- chapter_23_line_19: GL tag expansion.
  have row_19 : (∀ (w1 : α) (w2 : α), ((w1 = w2) → ((N w1) → ((N w2) → (add w1 zero w2))))) := by
    simpa only [gl_implication22] using (row_20)
  -- chapter_23_line_18: GL tag disintegration.
  have row_18 : (N zero) := by
    exact row_31.1.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_23_line_17: GL tag disintegration.
  have row_17 : (gl_implication21 N zero add) := by
    exact row_31.1.1.1.1.1.1.1.1.2
  -- chapter_23_line_16: GL tag expansion.
  have row_16 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((add w1 zero w2) → (w1 = w2))))) := by
    simpa only [gl_implication21] using (row_17)
  -- chapter_23_line_15: GL tag implication grouping.
  have row_15 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_21
    have auxiliary_source_2 := row_23
    have auxiliary_source_3 := row_25
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_23_line_14: GL tag task formulation.
  have row_14 : (mul one v2 v1) := by
    exact assumption_14
  -- chapter_23_line_13: GL tag implication.
  have row_13 : (gl_and21 one N v2 N v1 N) := by
    apply row_15
    exact row_14
  -- chapter_23_line_12: GL tag expansion.
  have row_12 : (((N one) ∧ (N v2)) ∧ (N v1)) := by
    simpa only [gl_and21] using (row_13)
  -- chapter_23_line_11: GL tag disintegration.
  have row_11 : (N one) := by
    exact row_12.1.1
  -- chapter_23_line_10: GL tag disintegration.
  have row_10 : (N v2) := by
    exact row_12.1.2
  -- chapter_23_line_9: GL tag disintegration.
  have row_9 : (N v1) := by
    exact row_12.2
  -- chapter_23_line_8: GL tag recursion.
  have row_8 : (v2 = zero) := by
    exact assumption_8
  -- chapter_23_line_7: GL tag equality1.
  have row_7 : (mul one zero v1) := by
    have equality_source := row_14
    have equality_step_1 := row_8
    cases equality_step_1
    exact equality_source
  -- chapter_23_line_6: GL tag implication.
  have row_6 : (v1 = zero) := by
    apply row_29
    exact row_11
    exact row_7
  -- chapter_23_line_5: GL tag symmetry of equality.
  have row_5 : (zero = v1) := by
    exact Eq.symm row_6
  -- chapter_23_line_4: GL tag symmetry of equality.
  have row_4 : (zero = v2) := by
    exact Eq.symm row_8
  -- chapter_23_line_3: GL tag implication.
  have row_3 : (add zero zero v2) := by
    apply row_19
    exact row_4
    exact row_18
    exact row_10
  -- chapter_23_line_2: GL tag equality1.
  have row_2 : (add v1 zero v2) := by
    have equality_source := row_3
    have equality_step_1 := row_5
    cases equality_step_1
    exact equality_source
  -- chapter_23_line_1: GL tag implication.
  have row_1 : (v1 = v2) := by
    apply row_16
    exact row_9
    exact row_2
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (assumption_14 : (succ previous v2))
    (assumption_8 : (mul one v2 v1))
    (assumption_4 : (∀ (w1 : α), ((N w1) → ((mul one previous w1) → (w1 = previous)))))
    : (v1 = v2) := by
  -- chapter_24_line_35: GL tag task formulation.
  have row_35 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_24_line_34: GL tag expansion.
  have row_34 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_35)
  -- chapter_24_line_33: GL tag disintegration.
  have row_33 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_34.1
  -- chapter_24_line_32: GL tag expansion.
  have row_32 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_33)
  -- chapter_24_line_31: GL tag disintegration.
  have row_31 : (gl_fXY succ N N) := by
    exact row_32.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_24_line_30: GL tag expansion.
  have row_30 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_31)
  -- chapter_24_line_29: GL tag disintegration.
  have row_29 : (gl_implication0 succ N) := by
    exact row_30.1.1.1
  -- chapter_24_line_28: GL tag expansion.
  have row_28 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_29)
  -- chapter_24_line_27: GL tag disintegration.
  have row_27 : (gl_implication1 succ N) := by
    exact row_30.1.1.2
  -- chapter_24_line_26: GL tag expansion.
  have row_26 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using (row_27)
  -- chapter_24_line_25: GL tag disintegration.
  have row_25 : (succ zero one) := by
    exact row_34.2
  -- chapter_24_line_24: GL tag disintegration.
  have row_24 : (gl_fXYZ mul N N N) := by
    exact row_32.1.1.1.1.2
  -- chapter_24_line_23: GL tag expansion.
  have row_23 : (((((gl_implication11 mul N) ∧ (gl_implication12 mul N)) ∧ (gl_implication13 mul N)) ∧ (gl_implication18 N N N mul)) ∧ (gl_implication19 N N mul)) := by
    simpa only [gl_fXYZ] using (row_24)
  -- chapter_24_line_22: GL tag disintegration.
  have row_22 : (gl_implication18 N N N mul) := by
    exact row_23.1.2
  -- chapter_24_line_21: GL tag expansion.
  have row_21 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication18] using (row_22)
  -- chapter_24_line_20: GL tag disintegration.
  have row_20 : (gl_implication28 N succ mul add) := by
    exact row_32.2
  -- chapter_24_line_19: GL tag expansion.
  have row_19 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((mul w3 w1 w4) → (∀ (w5 : α), ((mul w3 w2 w5) → (add w4 w3 w5))))))))) := by
    simpa only [gl_implication28] using (row_20)
  -- chapter_24_line_18: GL tag implication grouping.
  have row_18 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (gl_and20 w1 N w2 N))) := by
    have auxiliary_source_1 := row_28
    have auxiliary_source_2 := row_26
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_premise
    exact ⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_premise)⟩
  -- chapter_24_line_17: GL tag implication.
  have row_17 : (gl_and20 zero N one N) := by
    apply row_18
    exact row_25
  -- chapter_24_line_16: GL tag expansion.
  have row_16 : ((N zero) ∧ (N one)) := by
    simpa only [gl_and20] using (row_17)
  -- chapter_24_line_15: GL tag disintegration.
  have row_15 : (N one) := by
    exact row_16.2
  -- chapter_24_line_14: GL tag recursion.
  have row_14 : (succ previous v2) := by
    exact assumption_14
  -- chapter_24_line_13: GL tag implication.
  have row_13 : (N previous) := by
    apply row_28
    exact row_14
  -- chapter_24_line_12: GL tag implication.
  have row_12 : (gl_existence1 N one previous mul) := by
    apply row_21
    exact row_15
    exact row_13
  -- chapter_24_line_11: GL tag expansion.
  have row_11 : (¬ (∀ (v3 : α), ((N v3) → (¬ (mul one previous v3))))) := by
    simpa only [gl_existence1] using (row_12)
  have exists_row_11 : ∃ (v3 : α), ((N v3) ∧ (mul one previous v3)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_11
      intro v3 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v3, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v3, witness_row_11⟩ := exists_row_11
  -- chapter_24_line_10: GL tag disintegration.
  have row_10 : (mul one previous v3) := by
    exact witness_row_11.2
  -- chapter_24_line_9: GL tag disintegration.
  have row_9 : (N v3) := by
    exact witness_row_11.1
  -- chapter_24_line_8: GL tag task formulation.
  have row_8 : (mul one v2 v1) := by
    exact assumption_8
  -- chapter_24_line_7: GL tag implication.
  have row_7 : (add v3 one v1) := by
    apply row_19
    exact row_13
    exact row_14
    exact row_10
    exact row_8
  -- chapter_24_line_6: GL tag theorem.
  have row_6 := peano_source_015 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_24_line_5: GL tag implication.
  have row_5 : (succ v3 v1) := by
    apply row_6
    exact row_7
  -- chapter_24_line_4: GL tag recursion.
  have row_4 : (∀ (w1 : α), ((N w1) → ((mul one previous w1) → (w1 = previous)))) := by
    exact assumption_4
  -- chapter_24_line_3: GL tag implication.
  have row_3 : (v3 = previous) := by
    apply row_4
    exact row_9
    exact row_10
  -- chapter_24_line_2: GL tag theorem.
  have row_2 := peano_source_002 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_24_line_1: GL tag implication.
  have row_1 : (v1 = v2) := by
    apply row_2
    exact row_5
    exact row_14
    exact row_3
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α), ((N v1) → (∀ (v2 : α), ((mul one v2 v1) → (v1 = v2))))) := by
  intro v1
  intro premise_1
  intro v2
  intro premise_2
  have inductionMember : N v2 := by
    have typingRule := peano_source_016_induction_typing N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact typingRule v1 v2 premise_2
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v1 : α), ((N v1) → ((mul one zero v1) → (v1 = zero)))) := by
    intro v1
    intro base_premise_1
    intro base_premise_2
    have zeroRule := peano_source_016_check_zero N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact zeroRule v1 zero base_premise_2 rfl
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v1 : α), ((N v1) → ((mul one induction_n v1) → (v1 = induction_n)))) → ∀ induction_m, succ induction_n induction_m → (∀ (v1 : α), ((N v1) → ((mul one induction_m v1) → (v1 = induction_m)))) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro v1
    intro step_premise_1
    intro step_premise_2
    have step_induction_assumption_1 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have step_induction_assumption_2 :
        (∀ (w1 : α), ((N w1) → ((mul one induction_n w1) → (w1 = induction_n)))) := by
      intro w1
      intro step_induction_assumption_2_premise_1
      intro step_induction_assumption_2_premise_2
      apply induction_hypothesis
      all_goals assumption
    have stepRule := peano_source_016_check_induction_condition N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact stepRule induction_n v1 induction_m step_induction_assumption_1 step_premise_2 step_induction_assumption_2
  have inductionProperty : (∀ (v1 : α), ((N v1) → ((mul one v2 v1) → (v1 = v2)))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v1 : α), ((N v1) → ((mul one induction_value v1) → (v1 = induction_value)))))
      v2
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v1 premise_1 premise_2

private theorem peano_source_020_induction_typing
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v2 : α)
    (v5 : α)
    (assumption_2 : (succ v5 v2))
    : (N v5) := by
  -- chapter_30_line_10: GL tag task formulation.
  have row_10 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_30_line_9: GL tag expansion.
  have row_9 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_10)
  -- chapter_30_line_8: GL tag disintegration.
  have row_8 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_9.1
  -- chapter_30_line_7: GL tag expansion.
  have row_7 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_8)
  -- chapter_30_line_6: GL tag disintegration.
  have row_6 : (gl_fXY succ N N) := by
    exact row_7.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_30_line_5: GL tag expansion.
  have row_5 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_6)
  -- chapter_30_line_4: GL tag disintegration.
  have row_4 : (gl_implication0 succ N) := by
    exact row_5.1.1.1
  -- chapter_30_line_3: GL tag expansion.
  have row_3 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_4)
  -- chapter_30_line_2: GL tag task formulation.
  have row_2 : (succ v5 v2) := by
    exact assumption_2
  -- chapter_30_line_1: GL tag implication.
  have row_1 : (N v5) := by
    apply row_3
    exact row_2
  exact row_1

private theorem peano_source_020_check_zero
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (v4 : α)
    (v5 : α)
    (assumption_38 : (succ v5 v2))
    (assumption_34 : (succ v1 v4))
    (assumption_29 : (add v1 v2 v3))
    (assumption_18 : (v5 = zero))
    : (add v4 v5 v3) := by
  -- chapter_31_line_61: GL tag task formulation.
  have row_61 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_31_line_60: GL tag expansion.
  have row_60 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_61)
  -- chapter_31_line_59: GL tag disintegration.
  have row_59 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_60.1
  -- chapter_31_line_58: GL tag expansion.
  have row_58 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_59)
  -- chapter_31_line_57: GL tag disintegration.
  have row_57 : (gl_fXY succ N N) := by
    exact row_58.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_31_line_56: GL tag expansion.
  have row_56 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_57)
  -- chapter_31_line_55: GL tag disintegration.
  have row_55 : (gl_implication1 succ N) := by
    exact row_56.1.1.2
  -- chapter_31_line_54: GL tag expansion.
  have row_54 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using (row_55)
  -- chapter_31_line_53: GL tag disintegration.
  have row_53 : (gl_implication0 succ N) := by
    exact row_56.1.1.1
  -- chapter_31_line_52: GL tag expansion.
  have row_52 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_53)
  -- chapter_31_line_51: GL tag disintegration.
  have row_51 : (gl_implication23 N succ add) := by
    exact row_58.1.1.1.1.1.1.2
  -- chapter_31_line_50: GL tag expansion.
  have row_50 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication23] using (row_51)
  -- chapter_31_line_49: GL tag disintegration.
  have row_49 : (gl_fXYZ add N N N) := by
    exact row_58.1.1.1.1.1.1.1.1.1.2
  -- chapter_31_line_48: GL tag expansion.
  have row_48 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_49)
  -- chapter_31_line_47: GL tag disintegration.
  have row_47 : (gl_implication13 add N) := by
    exact row_48.1.1.2
  -- chapter_31_line_46: GL tag expansion.
  have row_46 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_47)
  -- chapter_31_line_45: GL tag disintegration.
  have row_45 : (gl_implication12 add N) := by
    exact row_48.1.1.1.2
  -- chapter_31_line_44: GL tag expansion.
  have row_44 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_45)
  -- chapter_31_line_43: GL tag disintegration.
  have row_43 : (gl_implication11 add N) := by
    exact row_48.1.1.1.1
  -- chapter_31_line_42: GL tag expansion.
  have row_42 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_43)
  -- chapter_31_line_41: GL tag disintegration.
  have row_41 : (gl_implication18 N N N add) := by
    exact row_48.1.2
  -- chapter_31_line_40: GL tag expansion.
  have row_40 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication18] using (row_41)
  -- chapter_31_line_39: GL tag implication grouping.
  have row_39 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (gl_and20 w1 N w2 N))) := by
    have auxiliary_source_1 := row_52
    have auxiliary_source_2 := row_54
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_premise
    exact ⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_premise)⟩
  -- chapter_31_line_38: GL tag task formulation.
  have row_38 : (succ v5 v2) := by
    exact assumption_38
  -- chapter_31_line_37: GL tag implication.
  have row_37 : (gl_and20 v5 N v2 N) := by
    apply row_39
    exact row_38
  -- chapter_31_line_36: GL tag expansion.
  have row_36 : ((N v5) ∧ (N v2)) := by
    simpa only [gl_and20] using (row_37)
  -- chapter_31_line_35: GL tag disintegration.
  have row_35 : (N v5) := by
    exact row_36.1
  -- chapter_31_line_34: GL tag task formulation.
  have row_34 : (succ v1 v4) := by
    exact assumption_34
  -- chapter_31_line_33: GL tag implication.
  have row_33 : (gl_and20 v1 N v4 N) := by
    apply row_39
    exact row_34
  -- chapter_31_line_32: GL tag expansion.
  have row_32 : ((N v1) ∧ (N v4)) := by
    simpa only [gl_and20] using (row_33)
  -- chapter_31_line_31: GL tag disintegration.
  have row_31 : (N v4) := by
    exact row_32.2
  -- chapter_31_line_30: GL tag implication grouping.
  have row_30 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_42
    have auxiliary_source_2 := row_44
    have auxiliary_source_3 := row_46
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_31_line_29: GL tag task formulation.
  have row_29 : (add v1 v2 v3) := by
    exact assumption_29
  -- chapter_31_line_28: GL tag implication.
  have row_28 : (gl_and21 v1 N v2 N v3 N) := by
    apply row_30
    exact row_29
  -- chapter_31_line_27: GL tag expansion.
  have row_27 : (((N v1) ∧ (N v2)) ∧ (N v3)) := by
    simpa only [gl_and21] using (row_28)
  -- chapter_31_line_26: GL tag disintegration.
  have row_26 : (N v1) := by
    exact row_27.1.1
  -- chapter_31_line_25: GL tag implication.
  have row_25 : (gl_existence1 N v1 v5 add) := by
    apply row_40
    exact row_26
    exact row_35
  -- chapter_31_line_24: GL tag expansion.
  have row_24 : (¬ (∀ (v9 : α), ((N v9) → (¬ (add v1 v5 v9))))) := by
    simpa only [gl_existence1] using (row_25)
  have exists_row_24 : ∃ (v9 : α), ((N v9) ∧ (add v1 v5 v9)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_24
      intro v9 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v9, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v9, witness_row_24⟩ := exists_row_24
  -- chapter_31_line_23: GL tag disintegration.
  have row_23 : (add v1 v5 v9) := by
    exact witness_row_24.2
  -- chapter_31_line_22: GL tag implication.
  have row_22 : (succ v9 v3) := by
    apply row_50
    exact row_35
    exact row_38
    exact row_23
    exact row_29
  -- chapter_31_line_21: GL tag implication.
  have row_21 : (gl_existence1 N v4 v5 add) := by
    apply row_40
    exact row_31
    exact row_35
  -- chapter_31_line_20: GL tag expansion.
  have row_20 : (¬ (∀ (v6 : α), ((N v6) → (¬ (add v4 v5 v6))))) := by
    simpa only [gl_existence1] using (row_21)
  have exists_row_20 : ∃ (v6 : α), ((N v6) ∧ (add v4 v5 v6)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_20
      intro v6 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v6, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v6, witness_row_20⟩ := exists_row_20
  -- chapter_31_line_19: GL tag disintegration.
  have row_19 : (add v4 v5 v6) := by
    exact witness_row_20.2
  -- chapter_31_line_18: GL tag recursion.
  have row_18 : (v5 = zero) := by
    exact assumption_18
  -- chapter_31_line_17: GL tag equality1.
  have row_17 : (add v4 zero v6) := by
    have equality_source := row_19
    have equality_step_1 := row_18
    cases equality_step_1
    exact equality_source
  -- chapter_31_line_16: GL tag theorem.
  have row_16 := peano_source_011 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_31_line_15: GL tag implication.
  have row_15 : (v4 = v6) := by
    apply row_16
    exact row_17
    exact row_31
  -- chapter_31_line_14: GL tag equality1.
  have row_14 : (add v1 zero v9) := by
    have equality_source := row_23
    have equality_step_1 := row_18
    cases equality_step_1
    exact equality_source
  -- chapter_31_line_13: GL tag implication.
  have row_13 : (v1 = v9) := by
    apply row_16
    exact row_14
    exact row_26
  -- chapter_31_line_12: GL tag theorem.
  have row_12 := peano_source_004 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_31_line_11: GL tag implication.
  have row_11 : (gl_or0 v4 v3 v1 v9) := by
    apply row_12
    exact row_34
    exact row_22
  -- chapter_31_line_10: GL tag symmetry of equality.
  have row_10 : (v9 = v1) := by
    exact Eq.symm row_13
  -- chapter_31_line_9: GL tag equality1.
  have row_9 : (gl_or0 v4 v3 v1 v1) := by
    have equality_source := row_11
    have equality_step_1 := row_10
    cases equality_step_1
    exact equality_source
  -- chapter_31_line_8: GL tag expansion.
  have row_8 : (¬ ((¬ (v4 = v3)) ∧ (v1 = v1))) := by
    simpa only [gl_or0, GLExport.orIffNotAndNot, Classical.not_not] using (row_9)
  -- chapter_31_line_7: GL tag disintegration.
  have row_7 : (gl_implication1756 v1 v1 v4 v3) := by
    change ((v1 = v1) → (v4 = v3))
    classical
    intro projection_premise
    apply Classical.byContradiction
    intro projection_counterexample
    exact (row_8) ⟨projection_counterexample, projection_premise⟩
  -- chapter_31_line_6: GL tag expansion.
  have row_6 : ((v1 = v1) → (v4 = v3)) := by
    simpa only [gl_implication1756] using (row_7)
  -- chapter_31_line_5: GL tag equality1.
  have row_5 : (v1 = v1) := by
    have equality_source := row_13
    have equality_step_1 := row_10
    cases equality_step_1
    exact equality_source
  -- chapter_31_line_4: GL tag implication.
  have row_4 : (v4 = v3) := by
    apply row_6
    exact row_5
  -- chapter_31_line_3: GL tag symmetry of equality.
  have row_3 : (v6 = v4) := by
    exact Eq.symm row_15
  -- chapter_31_line_2: GL tag equality2.
  have row_2 : (v6 = v3) := by
    exact Eq.trans row_3 row_4
  -- chapter_31_line_1: GL tag equality1.
  have row_1 : (add v4 v5 v3) := by
    have equality_source := row_19
    have equality_step_1 := row_2
    cases equality_step_1
    exact equality_source
  exact row_1

private theorem peano_source_020_check_induction_condition
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (v4 : α)
    (v5 : α)
    (assumption_17 : (succ previous v5))
    (assumption_14 : (add v1 v2 v3))
    (assumption_7 : (succ v1 v4))
    (assumption_6 : (∀ (w1 : α) (w2 : α), ((add v1 w1 w2) → ((succ v1 v4) → ((succ previous w1) → (add v4 previous w2))))))
    (assumption_4 : (succ v5 v2))
    : (add v4 v5 v3) := by
  -- chapter_32_line_37: GL tag task formulation.
  have row_37 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_32_line_36: GL tag expansion.
  have row_36 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_37)
  -- chapter_32_line_35: GL tag disintegration.
  have row_35 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_36.1
  -- chapter_32_line_34: GL tag expansion.
  have row_34 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_35)
  -- chapter_32_line_33: GL tag disintegration.
  have row_33 : (gl_fXY succ N N) := by
    exact row_34.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_32_line_32: GL tag expansion.
  have row_32 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_33)
  -- chapter_32_line_31: GL tag disintegration.
  have row_31 : (gl_implication1 succ N) := by
    exact row_32.1.1.2
  -- chapter_32_line_30: GL tag expansion.
  have row_30 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using (row_31)
  -- chapter_32_line_29: GL tag disintegration.
  have row_29 : (gl_fXYZ add N N N) := by
    exact row_34.1.1.1.1.1.1.1.1.1.2
  -- chapter_32_line_28: GL tag expansion.
  have row_28 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_29)
  -- chapter_32_line_27: GL tag disintegration.
  have row_27 : (gl_implication13 add N) := by
    exact row_28.1.1.2
  -- chapter_32_line_26: GL tag expansion.
  have row_26 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_27)
  -- chapter_32_line_25: GL tag disintegration.
  have row_25 : (gl_implication12 add N) := by
    exact row_28.1.1.1.2
  -- chapter_32_line_24: GL tag expansion.
  have row_24 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_25)
  -- chapter_32_line_23: GL tag disintegration.
  have row_23 : (gl_implication11 add N) := by
    exact row_28.1.1.1.1
  -- chapter_32_line_22: GL tag expansion.
  have row_22 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_23)
  -- chapter_32_line_21: GL tag disintegration.
  have row_21 : (gl_implication18 N N N add) := by
    exact row_28.1.2
  -- chapter_32_line_20: GL tag expansion.
  have row_20 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication18] using (row_21)
  -- chapter_32_line_19: GL tag disintegration.
  have row_19 : (gl_implication23 N succ add) := by
    exact row_34.1.1.1.1.1.1.2
  -- chapter_32_line_18: GL tag expansion.
  have row_18 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication23] using (row_19)
  -- chapter_32_line_17: GL tag recursion.
  have row_17 : (succ previous v5) := by
    exact assumption_17
  -- chapter_32_line_16: GL tag implication.
  have row_16 : (N v5) := by
    apply row_30
    exact row_17
  -- chapter_32_line_15: GL tag implication grouping.
  have row_15 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_22
    have auxiliary_source_2 := row_24
    have auxiliary_source_3 := row_26
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_32_line_14: GL tag task formulation.
  have row_14 : (add v1 v2 v3) := by
    exact assumption_14
  -- chapter_32_line_13: GL tag implication.
  have row_13 : (gl_and21 v1 N v2 N v3 N) := by
    apply row_15
    exact row_14
  -- chapter_32_line_12: GL tag expansion.
  have row_12 : (((N v1) ∧ (N v2)) ∧ (N v3)) := by
    simpa only [gl_and21] using (row_13)
  -- chapter_32_line_11: GL tag disintegration.
  have row_11 : (N v1) := by
    exact row_12.1.1
  -- chapter_32_line_10: GL tag implication.
  have row_10 : (gl_existence1 N v1 v5 add) := by
    apply row_20
    exact row_11
    exact row_16
  -- chapter_32_line_9: GL tag expansion.
  have row_9 : (¬ (∀ (v6 : α), ((N v6) → (¬ (add v1 v5 v6))))) := by
    simpa only [gl_existence1] using (row_10)
  have exists_row_9 : ∃ (v6 : α), ((N v6) ∧ (add v1 v5 v6)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_9
      intro v6 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v6, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v6, witness_row_9⟩ := exists_row_9
  -- chapter_32_line_8: GL tag disintegration.
  have row_8 : (add v1 v5 v6) := by
    exact witness_row_9.2
  -- chapter_32_line_7: GL tag task formulation.
  have row_7 : (succ v1 v4) := by
    exact assumption_7
  -- chapter_32_line_6: GL tag recursion.
  have row_6 : (∀ (w1 : α) (w2 : α), ((add v1 w1 w2) → ((succ v1 v4) → ((succ previous w1) → (add v4 previous w2))))) := by
    exact assumption_6
  -- chapter_32_line_5: GL tag implication.
  have row_5 : (add v4 previous v6) := by
    apply row_6
    exact row_8
    exact row_7
    exact row_17
  -- chapter_32_line_4: GL tag task formulation.
  have row_4 : (succ v5 v2) := by
    exact assumption_4
  -- chapter_32_line_3: GL tag implication.
  have row_3 : (succ v6 v3) := by
    apply row_18
    exact row_16
    exact row_4
    exact row_8
    exact row_14
  -- chapter_32_line_2: GL tag theorem.
  have row_2 := peano_source_018 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_32_line_1: GL tag implication.
  have row_1 : (add v4 v5 v3) := by
    apply row_2
    exact row_5
    exact row_17
    exact row_3
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
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
    have typingRule := peano_source_020_induction_typing N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact typingRule v2 v5 premise_3
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α), ((succ v1 v4) → ((succ zero v2) → (add v4 zero v3)))))) := by
    intro v1
    intro v2
    intro v3
    intro base_premise_1
    intro v4
    intro base_premise_2
    intro base_premise_3
    have zeroRule := peano_source_020_check_zero N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact zeroRule v1 v2 v3 v4 zero base_premise_3 base_premise_2 base_premise_1 rfl
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
        (succ induction_n induction_m) := by
      exact induction_successor
    have step_induction_assumption_2 :
        (∀ (w1 : α) (w2 : α), ((add v1 w1 w2) → ((succ v1 v4) → ((succ induction_n w1) → (add v4 induction_n w2))))) := by
      intro w1
      intro w2
      intro step_induction_assumption_2_premise_1
      intro step_induction_assumption_2_premise_2
      intro step_induction_assumption_2_premise_3
      apply induction_hypothesis
      all_goals assumption
    have stepRule := peano_source_020_check_induction_condition N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact stepRule induction_n v1 v2 v3 v4 induction_m step_induction_assumption_1 step_premise_1 step_premise_2 step_induction_assumption_2 step_premise_3
  have inductionProperty : (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α), ((succ v1 v4) → ((succ v5 v2) → (add v4 v5 v3)))))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α), ((succ v1 v4) → ((succ induction_value v2) → (add v4 induction_value v3)))))))
      v5
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v1 v2 v3 premise_1 v4 premise_2 premise_3

private theorem peano_source_026_induction_typing
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (assumption_2 : (mul v1 one v2))
    : (N v1) := by
  -- chapter_42_line_10: GL tag task formulation.
  have row_10 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_42_line_9: GL tag expansion.
  have row_9 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_10)
  -- chapter_42_line_8: GL tag disintegration.
  have row_8 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_9.1
  -- chapter_42_line_7: GL tag expansion.
  have row_7 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_8)
  -- chapter_42_line_6: GL tag disintegration.
  have row_6 : (gl_fXYZ mul N N N) := by
    exact row_7.1.1.1.1.2
  -- chapter_42_line_5: GL tag expansion.
  have row_5 : (((((gl_implication11 mul N) ∧ (gl_implication12 mul N)) ∧ (gl_implication13 mul N)) ∧ (gl_implication18 N N N mul)) ∧ (gl_implication19 N N mul)) := by
    simpa only [gl_fXYZ] using (row_6)
  -- chapter_42_line_4: GL tag disintegration.
  have row_4 : (gl_implication11 mul N) := by
    exact row_5.1.1.1.1
  -- chapter_42_line_3: GL tag expansion.
  have row_3 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_4)
  -- chapter_42_line_2: GL tag task formulation.
  have row_2 : (mul v1 one v2) := by
    exact assumption_2
  -- chapter_42_line_1: GL tag implication.
  have row_1 : (N v1) := by
    apply row_3
    exact row_2
  exact row_1

private theorem peano_source_026_check_zero
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (assumption_23 : (mul v1 one v2))
    (assumption_10 : (v1 = zero))
    : (mul one v1 v2) := by
  -- chapter_43_line_52: GL tag task formulation.
  have row_52 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_43_line_51: GL tag expansion.
  have row_51 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_52)
  -- chapter_43_line_50: GL tag disintegration.
  have row_50 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_51.1
  -- chapter_43_line_49: GL tag expansion.
  have row_49 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_50)
  -- chapter_43_line_48: GL tag disintegration.
  have row_48 : (gl_fXYZ add N N N) := by
    exact row_49.1.1.1.1.1.1.1.1.1.2
  -- chapter_43_line_47: GL tag expansion.
  have row_47 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_48)
  -- chapter_43_line_46: GL tag disintegration.
  have row_46 : (gl_implication19 N N add) := by
    exact row_47.2
  -- chapter_43_line_45: GL tag expansion.
  have row_45 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (∀ (w3 : α), ((add w1 w2 w3) → (∀ (w4 : α), ((add w1 w2 w4) → (w3 = w4))))))))) := by
    simpa only [gl_implication19] using (row_46)
  -- chapter_43_line_44: GL tag disintegration.
  have row_44 : (gl_fXYZ mul N N N) := by
    exact row_49.1.1.1.1.2
  -- chapter_43_line_43: GL tag expansion.
  have row_43 : (((((gl_implication11 mul N) ∧ (gl_implication12 mul N)) ∧ (gl_implication13 mul N)) ∧ (gl_implication18 N N N mul)) ∧ (gl_implication19 N N mul)) := by
    simpa only [gl_fXYZ] using (row_44)
  -- chapter_43_line_42: GL tag disintegration.
  have row_42 : (gl_implication13 mul N) := by
    exact row_43.1.1.2
  -- chapter_43_line_41: GL tag expansion.
  have row_41 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_42)
  -- chapter_43_line_40: GL tag disintegration.
  have row_40 : (gl_implication12 mul N) := by
    exact row_43.1.1.1.2
  -- chapter_43_line_39: GL tag expansion.
  have row_39 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_40)
  -- chapter_43_line_38: GL tag disintegration.
  have row_38 : (gl_implication11 mul N) := by
    exact row_43.1.1.1.1
  -- chapter_43_line_37: GL tag expansion.
  have row_37 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_38)
  -- chapter_43_line_36: GL tag disintegration.
  have row_36 : (gl_implication18 N N N mul) := by
    exact row_43.1.2
  -- chapter_43_line_35: GL tag expansion.
  have row_35 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication18] using (row_36)
  -- chapter_43_line_34: GL tag disintegration.
  have row_34 : (N zero) := by
    exact row_49.1.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_43_line_33: GL tag disintegration.
  have row_33 : (gl_implication25 N zero mul) := by
    exact row_49.1.1.1.2
  -- chapter_43_line_32: GL tag expansion.
  have row_32 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((mul w1 zero w2) → (w2 = zero))))) := by
    simpa only [gl_implication25] using (row_33)
  -- chapter_43_line_31: GL tag disintegration.
  have row_31 : (gl_implication28 N succ mul add) := by
    exact row_49.2
  -- chapter_43_line_30: GL tag expansion.
  have row_30 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((mul w3 w1 w4) → (∀ (w5 : α), ((mul w3 w2 w5) → (add w4 w3 w5))))))))) := by
    simpa only [gl_implication28] using (row_31)
  -- chapter_43_line_29: GL tag disintegration.
  have row_29 : (succ zero one) := by
    exact row_51.2
  -- chapter_43_line_28: GL tag disintegration.
  have row_28 : (gl_implication22 N zero add) := by
    exact row_49.1.1.1.1.1.1.1.2
  -- chapter_43_line_27: GL tag expansion.
  have row_27 : (∀ (w1 : α) (w2 : α), ((w1 = w2) → ((N w1) → ((N w2) → (add w1 zero w2))))) := by
    simpa only [gl_implication22] using (row_28)
  -- chapter_43_line_26: GL tag disintegration.
  have row_26 : (gl_implication26 zero N mul) := by
    exact row_49.1.1.2
  -- chapter_43_line_25: GL tag expansion.
  have row_25 : (∀ (w1 : α), ((w1 = zero) → (∀ (w2 : α), ((N w2) → ((N w1) → (mul w2 zero w1)))))) := by
    simpa only [gl_implication26] using (row_26)
  -- chapter_43_line_24: GL tag implication grouping.
  have row_24 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_37
    have auxiliary_source_2 := row_39
    have auxiliary_source_3 := row_41
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_43_line_23: GL tag task formulation.
  have row_23 : (mul v1 one v2) := by
    exact assumption_23
  -- chapter_43_line_22: GL tag implication.
  have row_22 : (gl_and21 v1 N one N v2 N) := by
    apply row_24
    exact row_23
  -- chapter_43_line_21: GL tag expansion.
  have row_21 : (((N v1) ∧ (N one)) ∧ (N v2)) := by
    simpa only [gl_and21] using (row_22)
  -- chapter_43_line_20: GL tag disintegration.
  have row_20 : (N v1) := by
    exact row_21.1.1
  -- chapter_43_line_19: GL tag disintegration.
  have row_19 : (N one) := by
    exact row_21.1.2
  -- chapter_43_line_18: GL tag implication.
  have row_18 : (gl_existence1 N v1 zero mul) := by
    apply row_35
    exact row_20
    exact row_34
  -- chapter_43_line_17: GL tag expansion.
  have row_17 : (¬ (∀ (v3 : α), ((N v3) → (¬ (mul v1 zero v3))))) := by
    simpa only [gl_existence1] using (row_18)
  have exists_row_17 : ∃ (v3 : α), ((N v3) ∧ (mul v1 zero v3)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_17
      intro v3 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v3, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v3, witness_row_17⟩ := exists_row_17
  -- chapter_43_line_16: GL tag disintegration.
  have row_16 : (mul v1 zero v3) := by
    exact witness_row_17.2
  -- chapter_43_line_15: GL tag disintegration.
  have row_15 : (N v3) := by
    exact witness_row_17.1
  -- chapter_43_line_14: GL tag theorem.
  have row_14 := peano_source_025 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_43_line_13: GL tag implication.
  have row_13 : (zero = v3) := by
    apply row_14
    exact row_16
    exact row_20
  -- chapter_43_line_12: GL tag implication.
  have row_12 : (v3 = zero) := by
    apply row_32
    exact row_20
    exact row_16
  -- chapter_43_line_11: GL tag implication.
  have row_11 : (add v3 v1 v2) := by
    apply row_30
    exact row_34
    exact row_29
    exact row_16
    exact row_23
  -- chapter_43_line_10: GL tag recursion.
  have row_10 : (v1 = zero) := by
    exact assumption_10
  -- chapter_43_line_9: GL tag equality1.
  have row_9 : (add zero zero v2) := by
    have equality_source := row_11
    have equality_step_1 := row_12
    cases equality_step_1
    have equality_step_2 := row_10
    cases equality_step_2
    exact equality_source
  -- chapter_43_line_8: GL tag symmetry of equality.
  have row_8 : (zero = v1) := by
    exact Eq.symm row_10
  -- chapter_43_line_7: GL tag equality1.
  have row_7 : (add v1 v3 v2) := by
    have equality_source := row_9
    have equality_step_1 := row_8
    cases equality_step_1
    have equality_step_2 := row_13
    cases equality_step_2
    exact equality_source
  -- chapter_43_line_6: GL tag implication.
  have row_6 : (add zero zero v3) := by
    apply row_27
    exact row_13
    exact row_34
    exact row_15
  -- chapter_43_line_5: GL tag equality1.
  have row_5 : (add v1 v3 zero) := by
    have equality_source := row_6
    have equality_step_1 := row_8
    cases equality_step_1
    have equality_step_2 := row_13
    cases equality_step_2
    have equality_step_3 := row_12
    cases equality_step_3
    exact equality_source
  -- chapter_43_line_4: GL tag implication.
  have row_4 : (zero = v2) := by
    apply row_45
    exact row_20
    exact row_15
    exact row_5
    exact row_7
  -- chapter_43_line_3: GL tag implication.
  have row_3 : (mul one zero v3) := by
    apply row_25
    exact row_12
    exact row_19
    exact row_15
  -- chapter_43_line_2: GL tag equality1.
  have row_2 : (mul one zero zero) := by
    have equality_source := row_3
    have equality_step_1 := row_12
    cases equality_step_1
    exact equality_source
  -- chapter_43_line_1: GL tag equality1.
  have row_1 : (mul one v1 v2) := by
    have equality_source := row_2
    have equality_step_1 := row_8
    cases equality_step_1
    have equality_step_2 := row_4
    cases equality_step_2
    exact equality_source
  exact row_1

private theorem peano_source_026_check_induction_condition
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (assumption_45 : (succ previous v1))
    (assumption_38 : (mul v1 one v2))
    (assumption_16 : (∀ (w1 : α), ((mul previous one w1) → (mul one previous w1))))
    : (mul one v1 v2) := by
  -- chapter_44_line_69: GL tag task formulation.
  have row_69 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_44_line_68: GL tag expansion.
  have row_68 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_69)
  -- chapter_44_line_67: GL tag disintegration.
  have row_67 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_68.1
  -- chapter_44_line_66: GL tag expansion.
  have row_66 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_67)
  -- chapter_44_line_65: GL tag disintegration.
  have row_65 : (gl_fXYZ mul N N N) := by
    exact row_66.1.1.1.1.2
  -- chapter_44_line_64: GL tag expansion.
  have row_64 : (((((gl_implication11 mul N) ∧ (gl_implication12 mul N)) ∧ (gl_implication13 mul N)) ∧ (gl_implication18 N N N mul)) ∧ (gl_implication19 N N mul)) := by
    simpa only [gl_fXYZ] using (row_65)
  -- chapter_44_line_63: GL tag disintegration.
  have row_63 : (gl_implication18 N N N mul) := by
    exact row_64.1.2
  -- chapter_44_line_62: GL tag expansion.
  have row_62 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication18] using (row_63)
  -- chapter_44_line_61: GL tag disintegration.
  have row_61 : (gl_fXY succ N N) := by
    exact row_66.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_44_line_60: GL tag expansion.
  have row_60 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_61)
  -- chapter_44_line_59: GL tag disintegration.
  have row_59 : (gl_implication7 N succ) := by
    exact row_60.2
  -- chapter_44_line_58: GL tag expansion.
  have row_58 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α), ((succ w1 w3) → (w2 = w3))))))) := by
    simpa only [gl_implication7] using (row_59)
  -- chapter_44_line_57: GL tag disintegration.
  have row_57 : (gl_implication1 succ N) := by
    exact row_60.1.1.2
  -- chapter_44_line_56: GL tag expansion.
  have row_56 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using (row_57)
  -- chapter_44_line_55: GL tag disintegration.
  have row_55 : (N zero) := by
    exact row_66.1.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_44_line_54: GL tag disintegration.
  have row_54 : (gl_implication28 N succ mul add) := by
    exact row_66.2
  -- chapter_44_line_53: GL tag expansion.
  have row_53 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((mul w3 w1 w4) → (∀ (w5 : α), ((mul w3 w2 w5) → (add w4 w3 w5))))))))) := by
    simpa only [gl_implication28] using (row_54)
  -- chapter_44_line_52: GL tag disintegration.
  have row_52 : (succ zero one) := by
    exact row_68.2
  -- chapter_44_line_51: GL tag disintegration.
  have row_51 : (gl_implication25 N zero mul) := by
    exact row_66.1.1.1.2
  -- chapter_44_line_50: GL tag expansion.
  have row_50 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((mul w1 zero w2) → (w2 = zero))))) := by
    simpa only [gl_implication25] using (row_51)
  -- chapter_44_line_49: GL tag disintegration.
  have row_49 : (gl_implication0 succ N) := by
    exact row_60.1.1.1
  -- chapter_44_line_48: GL tag expansion.
  have row_48 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_49)
  -- chapter_44_line_47: GL tag disintegration.
  have row_47 : (gl_implication6 N N succ) := by
    exact row_60.1.2
  -- chapter_44_line_46: GL tag expansion.
  have row_46 : (∀ (w1 : α), ((N w1) → (gl_existence0 N w1 succ))) := by
    simpa only [gl_implication6] using (row_47)
  -- chapter_44_line_45: GL tag recursion.
  have row_45 : (succ previous v1) := by
    exact assumption_45
  -- chapter_44_line_44: GL tag implication.
  have row_44 : (N v1) := by
    apply row_56
    exact row_45
  -- chapter_44_line_43: GL tag implication.
  have row_43 : (gl_existence1 N v1 zero mul) := by
    apply row_62
    exact row_44
    exact row_55
  -- chapter_44_line_42: GL tag expansion.
  have row_42 : (¬ (∀ (v10 : α), ((N v10) → (¬ (mul v1 zero v10))))) := by
    simpa only [gl_existence1] using (row_43)
  have exists_row_42 : ∃ (v10 : α), ((N v10) ∧ (mul v1 zero v10)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_42
      intro v10 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v10, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v10, witness_row_42⟩ := exists_row_42
  -- chapter_44_line_41: GL tag disintegration.
  have row_41 : (mul v1 zero v10) := by
    exact witness_row_42.2
  -- chapter_44_line_40: GL tag theorem.
  have row_40 := peano_source_025 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_44_line_39: GL tag implication.
  have row_39 : (zero = v10) := by
    apply row_40
    exact row_41
    exact row_44
  -- chapter_44_line_38: GL tag task formulation.
  have row_38 : (mul v1 one v2) := by
    exact assumption_38
  -- chapter_44_line_37: GL tag implication.
  have row_37 : (add v10 v1 v2) := by
    apply row_53
    exact row_55
    exact row_52
    exact row_41
    exact row_38
  -- chapter_44_line_36: GL tag implication grouping.
  have row_36 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (gl_and20 w1 N w2 N))) := by
    have auxiliary_source_1 := row_48
    have auxiliary_source_2 := row_56
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_premise
    exact ⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_premise)⟩
  -- chapter_44_line_35: GL tag implication.
  have row_35 : (gl_and20 zero N one N) := by
    apply row_36
    exact row_52
  -- chapter_44_line_34: GL tag expansion.
  have row_34 : ((N zero) ∧ (N one)) := by
    simpa only [gl_and20] using (row_35)
  -- chapter_44_line_33: GL tag disintegration.
  have row_33 : (N one) := by
    exact row_34.2
  -- chapter_44_line_32: GL tag implication.
  have row_32 : (N previous) := by
    apply row_48
    exact row_45
  -- chapter_44_line_31: GL tag implication.
  have row_31 : (gl_existence1 N previous zero mul) := by
    apply row_62
    exact row_32
    exact row_55
  -- chapter_44_line_30: GL tag expansion.
  have row_30 : (¬ (∀ (v13 : α), ((N v13) → (¬ (mul previous zero v13))))) := by
    simpa only [gl_existence1] using (row_31)
  have exists_row_30 : ∃ (v13 : α), ((N v13) ∧ (mul previous zero v13)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_30
      intro v13 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v13, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v13, witness_row_30⟩ := exists_row_30
  -- chapter_44_line_29: GL tag disintegration.
  have row_29 : (mul previous zero v13) := by
    exact witness_row_30.2
  -- chapter_44_line_28: GL tag implication.
  have row_28 : (v13 = zero) := by
    apply row_50
    exact row_32
    exact row_29
  -- chapter_44_line_27: GL tag implication.
  have row_27 : (gl_existence0 N previous succ) := by
    apply row_46
    exact row_32
  -- chapter_44_line_26: GL tag expansion.
  have row_26 : (¬ (∀ (v3 : α), ((N v3) → (¬ (succ previous v3))))) := by
    simpa only [gl_existence0] using (row_27)
  have exists_row_26 : ∃ (v3 : α), ((N v3) ∧ (succ previous v3)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_26
      intro v3 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v3, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v3, witness_row_26⟩ := exists_row_26
  -- chapter_44_line_25: GL tag disintegration.
  have row_25 : (succ previous v3) := by
    exact witness_row_26.2
  -- chapter_44_line_24: GL tag implication.
  have row_24 : (v3 = v1) := by
    apply row_58
    exact row_32
    exact row_25
    exact row_45
  -- chapter_44_line_23: GL tag implication.
  have row_23 : (gl_existence1 N previous one mul) := by
    apply row_62
    exact row_32
    exact row_33
  -- chapter_44_line_22: GL tag expansion.
  have row_22 : (¬ (∀ (v5 : α), ((N v5) → (¬ (mul previous one v5))))) := by
    simpa only [gl_existence1] using (row_23)
  have exists_row_22 : ∃ (v5 : α), ((N v5) ∧ (mul previous one v5)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_22
      intro v5 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v5, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v5, witness_row_22⟩ := exists_row_22
  -- chapter_44_line_21: GL tag disintegration.
  have row_21 : (mul previous one v5) := by
    exact witness_row_22.2
  -- chapter_44_line_20: GL tag disintegration.
  have row_20 : (N v5) := by
    exact witness_row_22.1
  -- chapter_44_line_19: GL tag implication.
  have row_19 : (gl_existence1 N one v1 mul) := by
    apply row_62
    exact row_33
    exact row_44
  -- chapter_44_line_18: GL tag expansion.
  have row_18 : (¬ (∀ (v4 : α), ((N v4) → (¬ (mul one v1 v4))))) := by
    simpa only [gl_existence1] using (row_19)
  have exists_row_18 : ∃ (v4 : α), ((N v4) ∧ (mul one v1 v4)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_18
      intro v4 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v4, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v4, witness_row_18⟩ := exists_row_18
  -- chapter_44_line_17: GL tag disintegration.
  have row_17 : (mul one v1 v4) := by
    exact witness_row_18.2
  -- chapter_44_line_16: GL tag recursion.
  have row_16 : (∀ (w1 : α), ((mul previous one w1) → (mul one previous w1))) := by
    exact assumption_16
  -- chapter_44_line_15: GL tag implication.
  have row_15 : (mul one previous v5) := by
    apply row_16
    exact row_21
  -- chapter_44_line_14: GL tag anchor handling.
  have row_14 : (gl_AnchorPeano N zero succ add mul one) := by
    exact row_69
  -- chapter_44_line_13: GL tag theorem.
  have row_13 := peano_source_008 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_44_line_12: GL tag implication.
  have row_12 : (add v5 one v4) := by
    apply row_13
    exact row_17
    exact row_15
    exact row_45
  -- chapter_44_line_11: GL tag implication.
  have row_11 : (add v13 previous v5) := by
    apply row_13
    exact row_21
    exact row_29
    exact row_52
  -- chapter_44_line_10: GL tag equality1.
  have row_10 : (add zero previous v5) := by
    have equality_source := row_11
    have equality_step_1 := row_28
    cases equality_step_1
    exact equality_source
  -- chapter_44_line_9: GL tag equality1.
  have row_9 : (add v10 previous v5) := by
    have equality_source := row_10
    have equality_step_1 := row_39
    cases equality_step_1
    exact equality_source
  -- chapter_44_line_8: GL tag theorem.
  have row_8 := peano_source_017 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_44_line_7: GL tag implication.
  have row_7 : (succ v5 v2) := by
    apply row_8
    exact row_37
    exact row_9
    exact row_45
  -- chapter_44_line_6: GL tag theorem.
  have row_6 := peano_source_015 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_44_line_5: GL tag implication.
  have row_5 : (succ v5 v4) := by
    apply row_6
    exact row_12
  -- chapter_44_line_4: GL tag implication.
  have row_4 : (v4 = v2) := by
    apply row_58
    exact row_20
    exact row_5
    exact row_7
  -- chapter_44_line_3: GL tag theorem.
  have row_3 := peano_source_024 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_44_line_2: GL tag implication.
  have row_2 : (mul one v3 v4) := by
    apply row_3
    exact row_12
    exact row_25
    exact row_15
  -- chapter_44_line_1: GL tag equality1.
  have row_1 : (mul one v1 v2) := by
    have equality_source := row_2
    have equality_step_1 := row_4
    cases equality_step_1
    have equality_step_2 := row_24
    cases equality_step_2
    exact equality_source
  exact row_1

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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((mul v1 one v2) → (mul one v1 v2))) := by
  intro v1
  intro v2
  intro premise_1
  have inductionMember : N v1 := by
    have typingRule := peano_source_026_induction_typing N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact typingRule v1 v2 premise_1
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v2 : α), ((mul zero one v2) → (mul one zero v2))) := by
    intro v2
    intro base_premise_1
    have zeroRule := peano_source_026_check_zero N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact zeroRule zero v2 base_premise_1 rfl
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v2 : α), ((mul induction_n one v2) → (mul one induction_n v2))) → ∀ induction_m, succ induction_n induction_m → (∀ (v2 : α), ((mul induction_m one v2) → (mul one induction_m v2))) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro v2
    intro step_premise_1
    have step_induction_assumption_1 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have step_induction_assumption_2 :
        (∀ (w1 : α), ((mul induction_n one w1) → (mul one induction_n w1))) := by
      intro w1
      intro step_induction_assumption_2_premise_1
      apply induction_hypothesis
      all_goals assumption
    have stepRule := peano_source_026_check_induction_condition N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact stepRule induction_n induction_m v2 step_induction_assumption_1 step_premise_1 step_induction_assumption_2
  have inductionProperty : (∀ (v2 : α), ((mul v1 one v2) → (mul one v1 v2))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v2 : α), ((mul induction_value one v2) → (mul one induction_value v2))))
      v1
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v2 premise_1

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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((add v1 zero v2) → (add zero v1 v2))) := by
  intro v1
  intro v2
  intro premise_1
  -- chapter_67_line_27: GL tag task formulation.
  have row_27 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_67_line_26: GL tag expansion.
  have row_26 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_27)
  -- chapter_67_line_25: GL tag disintegration.
  have row_25 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_26.1
  -- chapter_67_line_24: GL tag expansion.
  have row_24 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_25)
  -- chapter_67_line_23: GL tag disintegration.
  have row_23 : (gl_fXYZ add N N N) := by
    exact row_24.1.1.1.1.1.1.1.1.1.2
  -- chapter_67_line_22: GL tag expansion.
  have row_22 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_23)
  -- chapter_67_line_21: GL tag disintegration.
  have row_21 : (gl_implication13 add N) := by
    exact row_22.1.1.2
  -- chapter_67_line_20: GL tag expansion.
  have row_20 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_21)
  -- chapter_67_line_19: GL tag disintegration.
  have row_19 : (gl_implication11 add N) := by
    exact row_22.1.1.1.1
  -- chapter_67_line_18: GL tag expansion.
  have row_18 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_19)
  -- chapter_67_line_17: GL tag disintegration.
  have row_17 : (gl_implication18 N N N add) := by
    exact row_22.1.2
  -- chapter_67_line_16: GL tag expansion.
  have row_16 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication18] using (row_17)
  -- chapter_67_line_15: GL tag disintegration.
  have row_15 : (N zero) := by
    exact row_24.1.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_67_line_14: GL tag disintegration.
  have row_14 : (gl_implication21 N zero add) := by
    exact row_24.1.1.1.1.1.1.1.1.2
  -- chapter_67_line_13: GL tag expansion.
  have row_13 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((add w1 zero w2) → (w1 = w2))))) := by
    simpa only [gl_implication21] using (row_14)
  -- chapter_67_line_12: GL tag task formulation.
  have row_12 : (add v1 zero v2) := by
    exact premise_1
  -- chapter_67_line_11: GL tag implication.
  have row_11 : (N v2) := by
    apply row_20
    exact row_12
  -- chapter_67_line_10: GL tag implication.
  have row_10 : (N v1) := by
    apply row_18
    exact row_12
  -- chapter_67_line_9: GL tag implication.
  have row_9 : (gl_existence1 N zero v1 add) := by
    apply row_16
    exact row_15
    exact row_10
  -- chapter_67_line_8: GL tag expansion.
  have row_8 : (¬ (∀ (v3 : α), ((N v3) → (¬ (add zero v1 v3))))) := by
    simpa only [gl_existence1] using (row_9)
  have exists_row_8 : ∃ (v3 : α), ((N v3) ∧ (add zero v1 v3)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_8
      intro v3 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v3, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v3, witness_row_8⟩ := exists_row_8
  -- chapter_67_line_7: GL tag disintegration.
  have row_7 : (add zero v1 v3) := by
    exact witness_row_8.2
  -- chapter_67_line_6: GL tag implication.
  have row_6 : (v1 = v2) := by
    apply row_13
    exact row_10
    exact row_12
  -- chapter_67_line_5: GL tag equality1.
  have row_5 : (add zero v2 v3) := by
    have equality_source := row_7
    have equality_step_1 := row_6
    cases equality_step_1
    exact equality_source
  -- chapter_67_line_4: GL tag theorem.
  have row_4 := peano_source_032 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_67_line_3: GL tag implication.
  have row_3 : (v2 = v3) := by
    apply row_4
    exact row_11
    exact row_5
  -- chapter_67_line_2: GL tag symmetry of equality.
  have row_2 : (v3 = v2) := by
    exact Eq.symm row_3
  -- chapter_67_line_1: GL tag equality1.
  have row_1 : (add zero v1 v2) := by
    have equality_source := row_7
    have equality_step_1 := row_2
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (assumption_4 : (add v1 v2 v3))
    : (N v2) := by
  -- chapter_74_line_17: GL tag task formulation.
  have row_17 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_74_line_16: GL tag expansion.
  have row_16 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_17)
  -- chapter_74_line_15: GL tag disintegration.
  have row_15 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_16.1
  -- chapter_74_line_14: GL tag expansion.
  have row_14 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_15)
  -- chapter_74_line_13: GL tag disintegration.
  have row_13 : (gl_fXYZ add N N N) := by
    exact row_14.1.1.1.1.1.1.1.1.1.2
  -- chapter_74_line_12: GL tag expansion.
  have row_12 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_13)
  -- chapter_74_line_11: GL tag disintegration.
  have row_11 : (gl_implication13 add N) := by
    exact row_12.1.1.2
  -- chapter_74_line_10: GL tag expansion.
  have row_10 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_11)
  -- chapter_74_line_9: GL tag disintegration.
  have row_9 : (gl_implication12 add N) := by
    exact row_12.1.1.1.2
  -- chapter_74_line_8: GL tag expansion.
  have row_8 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_9)
  -- chapter_74_line_7: GL tag disintegration.
  have row_7 : (gl_implication11 add N) := by
    exact row_12.1.1.1.1
  -- chapter_74_line_6: GL tag expansion.
  have row_6 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_7)
  -- chapter_74_line_5: GL tag implication grouping.
  have row_5 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_6
    have auxiliary_source_2 := row_8
    have auxiliary_source_3 := row_10
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_74_line_4: GL tag task formulation.
  have row_4 : (add v1 v2 v3) := by
    exact assumption_4
  -- chapter_74_line_3: GL tag implication.
  have row_3 : (gl_and21 v1 N v2 N v3 N) := by
    apply row_5
    exact row_4
  -- chapter_74_line_2: GL tag expansion.
  have row_2 : (((N v1) ∧ (N v2)) ∧ (N v3)) := by
    simpa only [gl_and21] using (row_3)
  -- chapter_74_line_1: GL tag disintegration.
  have row_1 : (N v2) := by
    exact row_2.1.2
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (v4 : α)
    (v5 : α)
    (assumption_68 : (v2 = zero))
    (assumption_67 : (add v1 v2 v3))
    (assumption_38 : (succ v1 v4))
    (assumption_20 : (succ v3 v5))
    : (add v4 v2 v5) := by
  -- chapter_75_line_68: GL tag recursion.
  have row_68 : (v2 = zero) := by
    exact assumption_68
  -- chapter_75_line_67: GL tag task formulation.
  have row_67 : (add v1 v2 v3) := by
    exact assumption_67
  -- chapter_75_line_66: GL tag equality1.
  have row_66 : (add v1 zero v3) := by
    have equality_source := row_67
    have equality_step_1 := row_68
    cases equality_step_1
    exact equality_source
  -- chapter_75_line_65: GL tag task formulation.
  have row_65 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_75_line_64: GL tag expansion.
  have row_64 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_65)
  -- chapter_75_line_63: GL tag disintegration.
  have row_63 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_64.1
  -- chapter_75_line_62: GL tag expansion.
  have row_62 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_63)
  -- chapter_75_line_61: GL tag disintegration.
  have row_61 : (gl_implication9 N zero succ) := by
    exact row_62.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_75_line_60: GL tag expansion.
  have row_60 : (∀ (w1 : α), ((N w1) → (¬ (succ w1 zero)))) := by
    simpa only [gl_implication9] using (row_61)
  -- chapter_75_line_59: GL tag disintegration.
  have row_59 : (gl_fXY succ N N) := by
    exact row_62.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_75_line_58: GL tag expansion.
  have row_58 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_59)
  -- chapter_75_line_57: GL tag disintegration.
  have row_57 : (gl_implication1 succ N) := by
    exact row_58.1.1.2
  -- chapter_75_line_56: GL tag expansion.
  have row_56 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using (row_57)
  -- chapter_75_line_55: GL tag disintegration.
  have row_55 : (gl_implication0 succ N) := by
    exact row_58.1.1.1
  -- chapter_75_line_54: GL tag expansion.
  have row_54 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_55)
  -- chapter_75_line_53: GL tag disintegration.
  have row_53 : (gl_fXYZ add N N N) := by
    exact row_62.1.1.1.1.1.1.1.1.1.2
  -- chapter_75_line_52: GL tag expansion.
  have row_52 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_53)
  -- chapter_75_line_51: GL tag disintegration.
  have row_51 : (gl_implication13 add N) := by
    exact row_52.1.1.2
  -- chapter_75_line_50: GL tag expansion.
  have row_50 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_51)
  -- chapter_75_line_49: GL tag disintegration.
  have row_49 : (gl_implication12 add N) := by
    exact row_52.1.1.1.2
  -- chapter_75_line_48: GL tag expansion.
  have row_48 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_49)
  -- chapter_75_line_47: GL tag disintegration.
  have row_47 : (gl_implication11 add N) := by
    exact row_52.1.1.1.1
  -- chapter_75_line_46: GL tag expansion.
  have row_46 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_47)
  -- chapter_75_line_45: GL tag disintegration.
  have row_45 : (gl_implication6 N N succ) := by
    exact row_58.1.2
  -- chapter_75_line_44: GL tag expansion.
  have row_44 : (∀ (w1 : α), ((N w1) → (gl_existence0 N w1 succ))) := by
    simpa only [gl_implication6] using (row_45)
  -- chapter_75_line_43: GL tag disintegration.
  have row_43 : (gl_implication7 N succ) := by
    exact row_58.2
  -- chapter_75_line_42: GL tag expansion.
  have row_42 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α), ((succ w1 w3) → (w2 = w3))))))) := by
    simpa only [gl_implication7] using (row_43)
  -- chapter_75_line_41: GL tag disintegration.
  have row_41 : (gl_implication18 N N N add) := by
    exact row_52.1.2
  -- chapter_75_line_40: GL tag expansion.
  have row_40 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication18] using (row_41)
  -- chapter_75_line_39: GL tag implication grouping.
  have row_39 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (gl_and20 w1 N w2 N))) := by
    have auxiliary_source_1 := row_54
    have auxiliary_source_2 := row_56
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_premise
    exact ⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_premise)⟩
  -- chapter_75_line_38: GL tag task formulation.
  have row_38 : (succ v1 v4) := by
    exact assumption_38
  -- chapter_75_line_37: GL tag implication.
  have row_37 : (gl_and20 v1 N v4 N) := by
    apply row_39
    exact row_38
  -- chapter_75_line_36: GL tag expansion.
  have row_36 : ((N v1) ∧ (N v4)) := by
    simpa only [gl_and20] using (row_37)
  -- chapter_75_line_35: GL tag disintegration.
  have row_35 : (N v4) := by
    exact row_36.2
  -- chapter_75_line_34: GL tag implication grouping.
  have row_34 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_46
    have auxiliary_source_2 := row_48
    have auxiliary_source_3 := row_50
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_75_line_33: GL tag implication.
  have row_33 : (gl_and21 v1 N v2 N v3 N) := by
    apply row_34
    exact row_67
  -- chapter_75_line_32: GL tag expansion.
  have row_32 : (((N v1) ∧ (N v2)) ∧ (N v3)) := by
    simpa only [gl_and21] using (row_33)
  -- chapter_75_line_31: GL tag disintegration.
  have row_31 : (N v1) := by
    exact row_32.1.1
  -- chapter_75_line_30: GL tag disintegration.
  have row_30 : (N v3) := by
    exact row_32.2
  -- chapter_75_line_29: GL tag disintegration.
  have row_29 : (N v2) := by
    exact row_32.1.2
  -- chapter_75_line_28: GL tag theorem.
  have row_28 := peano_source_011 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_75_line_27: GL tag implication.
  have row_27 : (v1 = v3) := by
    apply row_28
    exact row_66
    exact row_31
  -- chapter_75_line_26: GL tag implication grouping.
  have row_26 : (∀ (w1 : α), ((N w1) → (gl_and3 N w1 succ zero))) := by
    have auxiliary_source_1 := row_44
    have auxiliary_source_2 := row_60
    intro auxiliary_argument_1
    intro auxiliary_premise
    exact ⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_premise)⟩
  -- chapter_75_line_25: GL tag implication.
  have row_25 : (gl_and3 N v3 succ zero) := by
    apply row_26
    exact row_30
  -- chapter_75_line_24: GL tag expansion.
  have row_24 : ((gl_existence0 N v3 succ) ∧ (¬ (succ v3 zero))) := by
    simpa only [gl_and3] using (row_25)
  -- chapter_75_line_23: GL tag disintegration.
  have row_23 : (gl_existence0 N v3 succ) := by
    exact row_24.1
  -- chapter_75_line_22: GL tag expansion.
  have row_22 : (¬ (∀ (v10 : α), ((N v10) → (¬ (succ v3 v10))))) := by
    simpa only [gl_existence0] using (row_23)
  have exists_row_22 : ∃ (v10 : α), ((N v10) ∧ (succ v3 v10)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_22
      intro v10 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v10, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v10, witness_row_22⟩ := exists_row_22
  -- chapter_75_line_21: GL tag disintegration.
  have row_21 : (succ v3 v10) := by
    exact witness_row_22.2
  -- chapter_75_line_20: GL tag task formulation.
  have row_20 : (succ v3 v5) := by
    exact assumption_20
  -- chapter_75_line_19: GL tag implication.
  have row_19 : (v10 = v5) := by
    apply row_42
    exact row_30
    exact row_21
    exact row_20
  -- chapter_75_line_18: GL tag theorem.
  have row_18 := peano_source_004 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_75_line_17: GL tag implication.
  have row_17 : (gl_or0 v4 v10 v1 v3) := by
    apply row_18
    exact row_38
    exact row_21
  -- chapter_75_line_16: GL tag symmetry of equality.
  have row_16 : (v3 = v1) := by
    exact Eq.symm row_27
  -- chapter_75_line_15: GL tag equality1.
  have row_15 : (gl_or0 v4 v5 v1 v1) := by
    have equality_source := row_17
    have equality_step_1 := row_16
    cases equality_step_1
    have equality_step_2 := row_19
    cases equality_step_2
    exact equality_source
  -- chapter_75_line_14: GL tag expansion.
  have row_14 : (¬ ((¬ (v4 = v5)) ∧ (v1 = v1))) := by
    simpa only [gl_or0, GLExport.orIffNotAndNot, Classical.not_not] using (row_15)
  -- chapter_75_line_13: GL tag disintegration.
  have row_13 : (gl_implication1756 v1 v1 v4 v5) := by
    change ((v1 = v1) → (v4 = v5))
    classical
    intro projection_premise
    apply Classical.byContradiction
    intro projection_counterexample
    exact (row_14) ⟨projection_counterexample, projection_premise⟩
  -- chapter_75_line_12: GL tag expansion.
  have row_12 : ((v1 = v1) → (v4 = v5)) := by
    simpa only [gl_implication1756] using (row_13)
  -- chapter_75_line_11: GL tag equality1.
  have row_11 : (v1 = v1) := by
    have equality_source := row_27
    have equality_step_1 := row_16
    cases equality_step_1
    exact equality_source
  -- chapter_75_line_10: GL tag implication.
  have row_10 : (v4 = v5) := by
    apply row_12
    exact row_11
  -- chapter_75_line_9: GL tag implication.
  have row_9 : (gl_existence1 N v4 v2 add) := by
    apply row_40
    exact row_35
    exact row_29
  -- chapter_75_line_8: GL tag expansion.
  have row_8 : (¬ (∀ (v6 : α), ((N v6) → (¬ (add v4 v2 v6))))) := by
    simpa only [gl_existence1] using (row_9)
  have exists_row_8 : ∃ (v6 : α), ((N v6) ∧ (add v4 v2 v6)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_8
      intro v6 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v6, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v6, witness_row_8⟩ := exists_row_8
  -- chapter_75_line_7: GL tag disintegration.
  have row_7 : (add v4 v2 v6) := by
    exact witness_row_8.2
  -- chapter_75_line_6: GL tag equality1.
  have row_6 : (add v4 zero v6) := by
    have equality_source := row_7
    have equality_step_1 := row_68
    cases equality_step_1
    exact equality_source
  -- chapter_75_line_5: GL tag implication.
  have row_5 : (v4 = v6) := by
    apply row_28
    exact row_6
    exact row_35
  -- chapter_75_line_4: GL tag symmetry of equality.
  have row_4 : (v6 = v4) := by
    exact Eq.symm row_5
  -- chapter_75_line_3: GL tag equality2.
  have row_3 : (v6 = v5) := by
    exact Eq.trans row_4 row_10
  -- chapter_75_line_2: GL tag symmetry of equality.
  have row_2 : (zero = v2) := by
    exact Eq.symm row_68
  -- chapter_75_line_1: GL tag equality1.
  have row_1 : (add v4 v2 v5) := by
    have equality_source := row_6
    have equality_step_1 := row_2
    cases equality_step_1
    have equality_step_2 := row_3
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (v4 : α)
    (v5 : α)
    (assumption_17 : (succ previous v2))
    (assumption_14 : (add v1 v2 v3))
    (assumption_6 : (∀ (w1 : α), ((add v1 previous w1) → ((succ v1 v4) → (∀ (w2 : α), ((succ w1 w2) → (add v4 previous w2)))))))
    (assumption_5 : (succ v1 v4))
    (assumption_3 : (succ v3 v5))
    : (add v4 v2 v5) := by
  -- chapter_76_line_37: GL tag task formulation.
  have row_37 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_76_line_36: GL tag expansion.
  have row_36 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_37)
  -- chapter_76_line_35: GL tag disintegration.
  have row_35 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_36.1
  -- chapter_76_line_34: GL tag expansion.
  have row_34 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_35)
  -- chapter_76_line_33: GL tag disintegration.
  have row_33 : (gl_fXY succ N N) := by
    exact row_34.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_76_line_32: GL tag expansion.
  have row_32 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_33)
  -- chapter_76_line_31: GL tag disintegration.
  have row_31 : (gl_implication0 succ N) := by
    exact row_32.1.1.1
  -- chapter_76_line_30: GL tag expansion.
  have row_30 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_31)
  -- chapter_76_line_29: GL tag disintegration.
  have row_29 : (gl_fXYZ add N N N) := by
    exact row_34.1.1.1.1.1.1.1.1.1.2
  -- chapter_76_line_28: GL tag expansion.
  have row_28 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_29)
  -- chapter_76_line_27: GL tag disintegration.
  have row_27 : (gl_implication13 add N) := by
    exact row_28.1.1.2
  -- chapter_76_line_26: GL tag expansion.
  have row_26 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_27)
  -- chapter_76_line_25: GL tag disintegration.
  have row_25 : (gl_implication12 add N) := by
    exact row_28.1.1.1.2
  -- chapter_76_line_24: GL tag expansion.
  have row_24 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_25)
  -- chapter_76_line_23: GL tag disintegration.
  have row_23 : (gl_implication11 add N) := by
    exact row_28.1.1.1.1
  -- chapter_76_line_22: GL tag expansion.
  have row_22 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_23)
  -- chapter_76_line_21: GL tag disintegration.
  have row_21 : (gl_implication18 N N N add) := by
    exact row_28.1.2
  -- chapter_76_line_20: GL tag expansion.
  have row_20 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication18] using (row_21)
  -- chapter_76_line_19: GL tag disintegration.
  have row_19 : (gl_implication23 N succ add) := by
    exact row_34.1.1.1.1.1.1.2
  -- chapter_76_line_18: GL tag expansion.
  have row_18 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication23] using (row_19)
  -- chapter_76_line_17: GL tag recursion.
  have row_17 : (succ previous v2) := by
    exact assumption_17
  -- chapter_76_line_16: GL tag implication.
  have row_16 : (N previous) := by
    apply row_30
    exact row_17
  -- chapter_76_line_15: GL tag implication grouping.
  have row_15 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_22
    have auxiliary_source_2 := row_24
    have auxiliary_source_3 := row_26
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_76_line_14: GL tag task formulation.
  have row_14 : (add v1 v2 v3) := by
    exact assumption_14
  -- chapter_76_line_13: GL tag implication.
  have row_13 : (gl_and21 v1 N v2 N v3 N) := by
    apply row_15
    exact row_14
  -- chapter_76_line_12: GL tag expansion.
  have row_12 : (((N v1) ∧ (N v2)) ∧ (N v3)) := by
    simpa only [gl_and21] using (row_13)
  -- chapter_76_line_11: GL tag disintegration.
  have row_11 : (N v1) := by
    exact row_12.1.1
  -- chapter_76_line_10: GL tag implication.
  have row_10 : (gl_existence1 N v1 previous add) := by
    apply row_20
    exact row_11
    exact row_16
  -- chapter_76_line_9: GL tag expansion.
  have row_9 : (¬ (∀ (v8 : α), ((N v8) → (¬ (add v1 previous v8))))) := by
    simpa only [gl_existence1] using (row_10)
  have exists_row_9 : ∃ (v8 : α), ((N v8) ∧ (add v1 previous v8)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_9
      intro v8 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v8, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v8, witness_row_9⟩ := exists_row_9
  -- chapter_76_line_8: GL tag disintegration.
  have row_8 : (add v1 previous v8) := by
    exact witness_row_9.2
  -- chapter_76_line_7: GL tag implication.
  have row_7 : (succ v8 v3) := by
    apply row_18
    exact row_16
    exact row_17
    exact row_8
    exact row_14
  -- chapter_76_line_6: GL tag recursion.
  have row_6 : (∀ (w1 : α), ((add v1 previous w1) → ((succ v1 v4) → (∀ (w2 : α), ((succ w1 w2) → (add v4 previous w2)))))) := by
    exact assumption_6
  -- chapter_76_line_5: GL tag task formulation.
  have row_5 : (succ v1 v4) := by
    exact assumption_5
  -- chapter_76_line_4: GL tag implication.
  have row_4 : (add v4 previous v3) := by
    apply row_6
    exact row_8
    exact row_5
    exact row_7
  -- chapter_76_line_3: GL tag task formulation.
  have row_3 : (succ v3 v5) := by
    exact assumption_3
  -- chapter_76_line_2: GL tag theorem.
  have row_2 := peano_source_018 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_76_line_1: GL tag implication.
  have row_1 : (add v4 v2 v5) := by
    apply row_2
    exact row_4
    exact row_17
    exact row_3
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α), ((succ v1 v4) → (∀ (v5 : α), ((succ v3 v5) → (add v4 v2 v5))))))) := by
  intro v1
  intro v2
  intro v3
  intro premise_1
  intro v4
  intro premise_2
  intro v5
  intro premise_3
  have inductionMember : N v2 := by
    have typingRule := peano_source_040_induction_typing N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact typingRule v1 v2 v3 premise_1
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v1 : α) (v3 : α), ((add v1 zero v3) → (∀ (v4 : α), ((succ v1 v4) → (∀ (v5 : α), ((succ v3 v5) → (add v4 zero v5))))))) := by
    intro v1
    intro v3
    intro base_premise_1
    intro v4
    intro base_premise_2
    intro v5
    intro base_premise_3
    have zeroRule := peano_source_040_check_zero N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact zeroRule v1 zero v3 v4 v5 rfl base_premise_1 base_premise_2 base_premise_3
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v1 : α) (v3 : α), ((add v1 induction_n v3) → (∀ (v4 : α), ((succ v1 v4) → (∀ (v5 : α), ((succ v3 v5) → (add v4 induction_n v5))))))) → ∀ induction_m, succ induction_n induction_m → (∀ (v1 : α) (v3 : α), ((add v1 induction_m v3) → (∀ (v4 : α), ((succ v1 v4) → (∀ (v5 : α), ((succ v3 v5) → (add v4 induction_m v5))))))) := by
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
        (succ induction_n induction_m) := by
      exact induction_successor
    have step_induction_assumption_2 :
        (∀ (w1 : α), ((add v1 induction_n w1) → ((succ v1 v4) → (∀ (w2 : α), ((succ w1 w2) → (add v4 induction_n w2)))))) := by
      intro w1
      intro step_induction_assumption_2_premise_1
      intro step_induction_assumption_2_premise_2
      intro w2
      intro step_induction_assumption_2_premise_3
      apply induction_hypothesis
      all_goals assumption
    have stepRule := peano_source_040_check_induction_condition N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact stepRule induction_n v1 induction_m v3 v4 v5 step_induction_assumption_1 step_premise_1 step_induction_assumption_2 step_premise_2 step_premise_3
  have inductionProperty : (∀ (v1 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α), ((succ v1 v4) → (∀ (v5 : α), ((succ v3 v5) → (add v4 v2 v5))))))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v1 : α) (v3 : α), ((add v1 induction_value v3) → (∀ (v4 : α), ((succ v1 v4) → (∀ (v5 : α), ((succ v3 v5) → (add v4 induction_value v5))))))))
      v2
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v1 v3 premise_1 v4 premise_2 v5 premise_3

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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (assumption_2 : (add v1 v2 one))
    : (N v2) := by
  -- chapter_90_line_10: GL tag task formulation.
  have row_10 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_90_line_9: GL tag expansion.
  have row_9 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_10)
  -- chapter_90_line_8: GL tag disintegration.
  have row_8 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_9.1
  -- chapter_90_line_7: GL tag expansion.
  have row_7 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_8)
  -- chapter_90_line_6: GL tag disintegration.
  have row_6 : (gl_fXYZ add N N N) := by
    exact row_7.1.1.1.1.1.1.1.1.1.2
  -- chapter_90_line_5: GL tag expansion.
  have row_5 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_6)
  -- chapter_90_line_4: GL tag disintegration.
  have row_4 : (gl_implication12 add N) := by
    exact row_5.1.1.1.2
  -- chapter_90_line_3: GL tag expansion.
  have row_3 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_4)
  -- chapter_90_line_2: GL tag task formulation.
  have row_2 : (add v1 v2 one) := by
    exact assumption_2
  -- chapter_90_line_1: GL tag implication.
  have row_1 : (N v2) := by
    apply row_3
    exact row_2
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (assumption_18 : (add v1 v2 one))
    (assumption_13 : (v2 = zero))
    : (add v2 v1 one) := by
  -- chapter_91_line_35: GL tag task formulation.
  have row_35 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_91_line_34: GL tag expansion.
  have row_34 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_35)
  -- chapter_91_line_33: GL tag disintegration.
  have row_33 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_34.1
  -- chapter_91_line_32: GL tag expansion.
  have row_32 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_33)
  -- chapter_91_line_31: GL tag disintegration.
  have row_31 : (gl_fXYZ add N N N) := by
    exact row_32.1.1.1.1.1.1.1.1.1.2
  -- chapter_91_line_30: GL tag expansion.
  have row_30 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_31)
  -- chapter_91_line_29: GL tag disintegration.
  have row_29 : (gl_implication13 add N) := by
    exact row_30.1.1.2
  -- chapter_91_line_28: GL tag expansion.
  have row_28 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_29)
  -- chapter_91_line_27: GL tag disintegration.
  have row_27 : (gl_implication12 add N) := by
    exact row_30.1.1.1.2
  -- chapter_91_line_26: GL tag expansion.
  have row_26 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_27)
  -- chapter_91_line_25: GL tag disintegration.
  have row_25 : (gl_implication11 add N) := by
    exact row_30.1.1.1.1
  -- chapter_91_line_24: GL tag expansion.
  have row_24 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_25)
  -- chapter_91_line_23: GL tag disintegration.
  have row_23 : (gl_implication21 N zero add) := by
    exact row_32.1.1.1.1.1.1.1.1.2
  -- chapter_91_line_22: GL tag expansion.
  have row_22 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((add w1 zero w2) → (w1 = w2))))) := by
    simpa only [gl_implication21] using (row_23)
  -- chapter_91_line_21: GL tag disintegration.
  have row_21 : (gl_implication18 N N N add) := by
    exact row_30.1.2
  -- chapter_91_line_20: GL tag expansion.
  have row_20 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication18] using (row_21)
  -- chapter_91_line_19: GL tag implication grouping.
  have row_19 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_24
    have auxiliary_source_2 := row_26
    have auxiliary_source_3 := row_28
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_91_line_18: GL tag task formulation.
  have row_18 : (add v1 v2 one) := by
    exact assumption_18
  -- chapter_91_line_17: GL tag implication.
  have row_17 : (gl_and21 v1 N v2 N one N) := by
    apply row_19
    exact row_18
  -- chapter_91_line_16: GL tag expansion.
  have row_16 : (((N v1) ∧ (N v2)) ∧ (N one)) := by
    simpa only [gl_and21] using (row_17)
  -- chapter_91_line_15: GL tag disintegration.
  have row_15 : (N v1) := by
    exact row_16.1.1
  -- chapter_91_line_14: GL tag disintegration.
  have row_14 : (N v2) := by
    exact row_16.1.2
  -- chapter_91_line_13: GL tag recursion.
  have row_13 : (v2 = zero) := by
    exact assumption_13
  -- chapter_91_line_12: GL tag equality1.
  have row_12 : (add v1 zero one) := by
    have equality_source := row_18
    have equality_step_1 := row_13
    cases equality_step_1
    exact equality_source
  -- chapter_91_line_11: GL tag implication.
  have row_11 : (v1 = one) := by
    apply row_22
    exact row_15
    exact row_12
  -- chapter_91_line_10: GL tag implication.
  have row_10 : (gl_existence1 N v2 v1 add) := by
    apply row_20
    exact row_14
    exact row_15
  -- chapter_91_line_9: GL tag expansion.
  have row_9 : (¬ (∀ (v3 : α), ((N v3) → (¬ (add v2 v1 v3))))) := by
    simpa only [gl_existence1] using (row_10)
  have exists_row_9 : ∃ (v3 : α), ((N v3) ∧ (add v2 v1 v3)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_9
      intro v3 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v3, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v3, witness_row_9⟩ := exists_row_9
  -- chapter_91_line_8: GL tag disintegration.
  have row_8 : (add v2 v1 v3) := by
    exact witness_row_9.2
  -- chapter_91_line_7: GL tag equality1.
  have row_7 : (add zero v1 v3) := by
    have equality_source := row_8
    have equality_step_1 := row_13
    cases equality_step_1
    exact equality_source
  -- chapter_91_line_6: GL tag theorem.
  have row_6 := peano_source_032 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_91_line_5: GL tag implication.
  have row_5 : (v1 = v3) := by
    apply row_6
    exact row_15
    exact row_7
  -- chapter_91_line_4: GL tag symmetry of equality.
  have row_4 : (v3 = v1) := by
    exact Eq.symm row_5
  -- chapter_91_line_3: GL tag equality2.
  have row_3 : (v3 = one) := by
    exact Eq.trans row_4 row_11
  -- chapter_91_line_2: GL tag symmetry of equality.
  have row_2 : (zero = v2) := by
    exact Eq.symm row_13
  -- chapter_91_line_1: GL tag equality1.
  have row_1 : (add v2 v1 one) := by
    have equality_source := row_7
    have equality_step_1 := row_2
    cases equality_step_1
    have equality_step_2 := row_3
    cases equality_step_2
    exact equality_source
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (assumption_39 : (succ previous v2))
    (assumption_36 : (add v1 v2 one))
    : (add v2 v1 one) := by
  -- chapter_92_line_68: GL tag task formulation.
  have row_68 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_92_line_67: GL tag expansion.
  have row_67 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_68)
  -- chapter_92_line_66: GL tag disintegration.
  have row_66 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_67.1
  -- chapter_92_line_65: GL tag expansion.
  have row_65 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_66)
  -- chapter_92_line_64: GL tag disintegration.
  have row_64 : (gl_implication10 N succ) := by
    exact row_65.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_92_line_63: GL tag expansion.
  have row_63 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w2 w1) → (∀ (w3 : α), ((succ w3 w1) → (w2 = w3))))))) := by
    simpa only [gl_implication10] using (row_64)
  -- chapter_92_line_62: GL tag disintegration.
  have row_62 : (gl_fXY succ N N) := by
    exact row_65.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_92_line_61: GL tag expansion.
  have row_61 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_62)
  -- chapter_92_line_60: GL tag disintegration.
  have row_60 : (gl_implication0 succ N) := by
    exact row_61.1.1.1
  -- chapter_92_line_59: GL tag expansion.
  have row_59 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_60)
  -- chapter_92_line_58: GL tag disintegration.
  have row_58 : (gl_implication1 succ N) := by
    exact row_61.1.1.2
  -- chapter_92_line_57: GL tag expansion.
  have row_57 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using (row_58)
  -- chapter_92_line_56: GL tag disintegration.
  have row_56 : (succ zero one) := by
    exact row_67.2
  -- chapter_92_line_55: GL tag disintegration.
  have row_55 : (gl_implication23 N succ add) := by
    exact row_65.1.1.1.1.1.1.2
  -- chapter_92_line_54: GL tag expansion.
  have row_54 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication23] using (row_55)
  -- chapter_92_line_53: GL tag disintegration.
  have row_53 : (gl_fXYZ add N N N) := by
    exact row_65.1.1.1.1.1.1.1.1.1.2
  -- chapter_92_line_52: GL tag expansion.
  have row_52 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_53)
  -- chapter_92_line_51: GL tag disintegration.
  have row_51 : (gl_implication13 add N) := by
    exact row_52.1.1.2
  -- chapter_92_line_50: GL tag expansion.
  have row_50 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_51)
  -- chapter_92_line_49: GL tag disintegration.
  have row_49 : (gl_implication12 add N) := by
    exact row_52.1.1.1.2
  -- chapter_92_line_48: GL tag expansion.
  have row_48 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_49)
  -- chapter_92_line_47: GL tag disintegration.
  have row_47 : (gl_implication11 add N) := by
    exact row_52.1.1.1.1
  -- chapter_92_line_46: GL tag expansion.
  have row_46 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_47)
  -- chapter_92_line_45: GL tag disintegration.
  have row_45 : (gl_implication18 N N N add) := by
    exact row_52.1.2
  -- chapter_92_line_44: GL tag expansion.
  have row_44 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication18] using (row_45)
  -- chapter_92_line_43: GL tag implication grouping.
  have row_43 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (gl_and20 w1 N w2 N))) := by
    have auxiliary_source_1 := row_59
    have auxiliary_source_2 := row_57
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_premise
    exact ⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_premise)⟩
  -- chapter_92_line_42: GL tag implication.
  have row_42 : (gl_and20 zero N one N) := by
    apply row_43
    exact row_56
  -- chapter_92_line_41: GL tag expansion.
  have row_41 : ((N zero) ∧ (N one)) := by
    simpa only [gl_and20] using (row_42)
  -- chapter_92_line_40: GL tag disintegration.
  have row_40 : (N one) := by
    exact row_41.2
  -- chapter_92_line_39: GL tag recursion.
  have row_39 : (succ previous v2) := by
    exact assumption_39
  -- chapter_92_line_38: GL tag implication.
  have row_38 : (N previous) := by
    apply row_59
    exact row_39
  -- chapter_92_line_37: GL tag implication grouping.
  have row_37 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_46
    have auxiliary_source_2 := row_48
    have auxiliary_source_3 := row_50
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_92_line_36: GL tag task formulation.
  have row_36 : (add v1 v2 one) := by
    exact assumption_36
  -- chapter_92_line_35: GL tag implication.
  have row_35 : (gl_and21 v1 N v2 N one N) := by
    apply row_37
    exact row_36
  -- chapter_92_line_34: GL tag expansion.
  have row_34 : (((N v1) ∧ (N v2)) ∧ (N one)) := by
    simpa only [gl_and21] using (row_35)
  -- chapter_92_line_33: GL tag disintegration.
  have row_33 : (N v1) := by
    exact row_34.1.1
  -- chapter_92_line_32: GL tag implication.
  have row_32 : (gl_existence1 N v1 previous add) := by
    apply row_44
    exact row_33
    exact row_38
  -- chapter_92_line_31: GL tag expansion.
  have row_31 : (¬ (∀ (v3 : α), ((N v3) → (¬ (add v1 previous v3))))) := by
    simpa only [gl_existence1] using (row_32)
  have exists_row_31 : ∃ (v3 : α), ((N v3) ∧ (add v1 previous v3)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_31
      intro v3 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v3, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v3, witness_row_31⟩ := exists_row_31
  -- chapter_92_line_30: GL tag disintegration.
  have row_30 : (add v1 previous v3) := by
    exact witness_row_31.2
  -- chapter_92_line_29: GL tag implication.
  have row_29 : (succ v3 one) := by
    apply row_54
    exact row_38
    exact row_39
    exact row_30
    exact row_36
  -- chapter_92_line_28: GL tag implication.
  have row_28 : (v3 = zero) := by
    apply row_63
    exact row_40
    exact row_29
    exact row_56
  -- chapter_92_line_27: GL tag anchor handling.
  have row_27 : (gl_AnchorPeano N zero succ add mul one) := by
    exact row_68
  -- chapter_92_line_26: GL tag theorem.
  have row_26 := peano_source_044 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_92_line_25: GL tag implication.
  have row_25 : (zero = v3) := by
    apply row_26
    exact row_29
    exact row_56
  -- chapter_92_line_24: GL tag equality1.
  have row_24 : (zero = zero) := by
    have equality_source := row_25
    have equality_step_1 := row_28
    cases equality_step_1
    exact equality_source
  -- chapter_92_line_23: GL tag equality1.
  have row_23 : (add v1 previous zero) := by
    have equality_source := row_30
    have equality_step_1 := row_28
    cases equality_step_1
    exact equality_source
  -- chapter_92_line_22: GL tag theorem.
  have row_22 := peano_source_029 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_92_line_21: GL tag implication.
  have row_21 : (zero = previous) := by
    apply row_22
    exact row_23
    exact row_33
  -- chapter_92_line_20: GL tag theorem.
  have row_20 := peano_source_045 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_92_line_19: GL tag implication.
  have row_19 : (v1 = previous) := by
    apply row_20
    exact row_23
    exact row_33
  -- chapter_92_line_18: GL tag symmetry of equality.
  have row_18 : (previous = zero) := by
    exact Eq.symm row_21
  -- chapter_92_line_17: GL tag theorem.
  have row_17 := peano_source_004 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_92_line_16: GL tag implication.
  have row_16 : (gl_or0 one v2 v3 previous) := by
    apply row_17
    exact row_29
    exact row_39
  -- chapter_92_line_15: GL tag equality1.
  have row_15 : (gl_or0 one v2 zero zero) := by
    have equality_source := row_16
    have equality_step_1 := row_28
    cases equality_step_1
    have equality_step_2 := row_18
    cases equality_step_2
    exact equality_source
  -- chapter_92_line_14: GL tag implication.
  have row_14 : (gl_or0 v2 one previous v3) := by
    apply row_17
    exact row_39
    exact row_29
  -- chapter_92_line_13: GL tag equality1.
  have row_13 : (gl_or0 v2 one zero zero) := by
    have equality_source := row_14
    have equality_step_1 := row_28
    cases equality_step_1
    have equality_step_2 := row_18
    cases equality_step_2
    exact equality_source
  -- chapter_92_line_12: GL tag expansion.
  have row_12 : (¬ ((¬ (v2 = one)) ∧ (zero = zero))) := by
    simpa only [gl_or0, GLExport.orIffNotAndNot, Classical.not_not] using (row_13)
  -- chapter_92_line_11: GL tag disintegration.
  have row_11 : (gl_implication1756 zero zero v2 one) := by
    change ((zero = zero) → (v2 = one))
    classical
    intro projection_premise
    apply Classical.byContradiction
    intro projection_counterexample
    exact (row_12) ⟨projection_counterexample, projection_premise⟩
  -- chapter_92_line_10: GL tag expansion.
  have row_10 : ((zero = zero) → (v2 = one)) := by
    simpa only [gl_implication1756] using (row_11)
  -- chapter_92_line_9: GL tag implication.
  have row_9 : (v2 = one) := by
    apply row_10
    exact row_24
  -- chapter_92_line_8: GL tag expansion.
  have row_8 : (¬ ((¬ (one = v2)) ∧ (zero = zero))) := by
    simpa only [gl_or0, GLExport.orIffNotAndNot, Classical.not_not] using (row_15)
  -- chapter_92_line_7: GL tag disintegration.
  have row_7 : (gl_implication1756 zero zero one v2) := by
    change ((zero = zero) → (one = v2))
    classical
    intro projection_premise
    apply Classical.byContradiction
    intro projection_counterexample
    exact (row_8) ⟨projection_counterexample, projection_premise⟩
  -- chapter_92_line_6: GL tag expansion.
  have row_6 : ((zero = zero) → (one = v2)) := by
    simpa only [gl_implication1756] using (row_7)
  -- chapter_92_line_5: GL tag implication.
  have row_5 : (one = v2) := by
    apply row_6
    exact row_24
  -- chapter_92_line_4: GL tag symmetry of equality.
  have row_4 : (previous = v1) := by
    exact Eq.symm row_19
  -- chapter_92_line_3: GL tag theorem.
  have row_3 := peano_source_010 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_92_line_2: GL tag implication.
  have row_2 : (add one previous v2) := by
    apply row_3
    exact row_39
  -- chapter_92_line_1: GL tag equality1.
  have row_1 : (add v2 v1 one) := by
    have equality_source := row_2
    have equality_step_1 := row_5
    cases equality_step_1
    have equality_step_2 := row_4
    cases equality_step_2
    have equality_step_3 := row_9
    cases equality_step_3
    exact equality_source
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((add v1 v2 one) → (add v2 v1 one))) := by
  intro v1
  intro v2
  intro premise_1
  have inductionMember : N v2 := by
    have typingRule := peano_source_046_induction_typing N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact typingRule v1 v2 premise_1
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v1 : α), ((add v1 zero one) → (add zero v1 one))) := by
    intro v1
    intro base_premise_1
    have zeroRule := peano_source_046_check_zero N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact zeroRule v1 zero base_premise_1 rfl
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v1 : α), ((add v1 induction_n one) → (add induction_n v1 one))) → ∀ induction_m, succ induction_n induction_m → (∀ (v1 : α), ((add v1 induction_m one) → (add induction_m v1 one))) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro v1
    intro step_premise_1
    have step_induction_assumption_1 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have stepRule := peano_source_046_check_induction_condition N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact stepRule induction_n v1 induction_m step_induction_assumption_1 step_premise_1
  have inductionProperty : (∀ (v1 : α), ((add v1 v2 one) → (add v2 v1 one))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v1 : α), ((add v1 induction_value one) → (add induction_value v1 one))))
      v2
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v1 premise_1

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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v2 : α)
    (v4 : α)
    (v6 : α)
    (assumption_2 : (add v2 v4 v6))
    : (N v4) := by
  -- chapter_93_line_10: GL tag task formulation.
  have row_10 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_93_line_9: GL tag expansion.
  have row_9 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_10)
  -- chapter_93_line_8: GL tag disintegration.
  have row_8 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_9.1
  -- chapter_93_line_7: GL tag expansion.
  have row_7 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_8)
  -- chapter_93_line_6: GL tag disintegration.
  have row_6 : (gl_fXYZ add N N N) := by
    exact row_7.1.1.1.1.1.1.1.1.1.2
  -- chapter_93_line_5: GL tag expansion.
  have row_5 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_6)
  -- chapter_93_line_4: GL tag disintegration.
  have row_4 : (gl_implication12 add N) := by
    exact row_5.1.1.1.2
  -- chapter_93_line_3: GL tag expansion.
  have row_3 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_4)
  -- chapter_93_line_2: GL tag task formulation.
  have row_2 : (add v2 v4 v6) := by
    exact assumption_2
  -- chapter_93_line_1: GL tag implication.
  have row_1 : (N v4) := by
    apply row_3
    exact row_2
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (v4 : α)
    (v5 : α)
    (v6 : α)
    (v7 : α)
    (assumption_52 : (v4 = zero))
    (assumption_51 : (add v2 v4 v6))
    (assumption_49 : (mul v1 v4 v5))
    (assumption_28 : (mul v1 v6 v7))
    (assumption_20 : (mul v1 v2 v3))
    : (add v3 v5 v7) := by
  -- chapter_94_line_52: GL tag recursion.
  have row_52 : (v4 = zero) := by
    exact assumption_52
  -- chapter_94_line_51: GL tag task formulation.
  have row_51 : (add v2 v4 v6) := by
    exact assumption_51
  -- chapter_94_line_50: GL tag equality1.
  have row_50 : (add v2 zero v6) := by
    have equality_source := row_51
    have equality_step_1 := row_52
    cases equality_step_1
    exact equality_source
  -- chapter_94_line_49: GL tag task formulation.
  have row_49 : (mul v1 v4 v5) := by
    exact assumption_49
  -- chapter_94_line_48: GL tag equality1.
  have row_48 : (mul v1 zero v5) := by
    have equality_source := row_49
    have equality_step_1 := row_52
    cases equality_step_1
    exact equality_source
  -- chapter_94_line_47: GL tag task formulation.
  have row_47 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_94_line_46: GL tag expansion.
  have row_46 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_47)
  -- chapter_94_line_45: GL tag disintegration.
  have row_45 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_46.1
  -- chapter_94_line_44: GL tag expansion.
  have row_44 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_45)
  -- chapter_94_line_43: GL tag disintegration.
  have row_43 : (gl_fXYZ mul N N N) := by
    exact row_44.1.1.1.1.2
  -- chapter_94_line_42: GL tag expansion.
  have row_42 : (((((gl_implication11 mul N) ∧ (gl_implication12 mul N)) ∧ (gl_implication13 mul N)) ∧ (gl_implication18 N N N mul)) ∧ (gl_implication19 N N mul)) := by
    simpa only [gl_fXYZ] using (row_43)
  -- chapter_94_line_41: GL tag disintegration.
  have row_41 : (gl_implication19 N N mul) := by
    exact row_42.2
  -- chapter_94_line_40: GL tag expansion.
  have row_40 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (∀ (w3 : α), ((mul w1 w2 w3) → (∀ (w4 : α), ((mul w1 w2 w4) → (w3 = w4))))))))) := by
    simpa only [gl_implication19] using (row_41)
  -- chapter_94_line_39: GL tag disintegration.
  have row_39 : (gl_implication13 mul N) := by
    exact row_42.1.1.2
  -- chapter_94_line_38: GL tag expansion.
  have row_38 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_39)
  -- chapter_94_line_37: GL tag disintegration.
  have row_37 : (gl_implication12 mul N) := by
    exact row_42.1.1.1.2
  -- chapter_94_line_36: GL tag expansion.
  have row_36 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_37)
  -- chapter_94_line_35: GL tag disintegration.
  have row_35 : (gl_implication11 mul N) := by
    exact row_42.1.1.1.1
  -- chapter_94_line_34: GL tag expansion.
  have row_34 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_35)
  -- chapter_94_line_33: GL tag disintegration.
  have row_33 : (gl_fXYZ add N N N) := by
    exact row_44.1.1.1.1.1.1.1.1.1.2
  -- chapter_94_line_32: GL tag expansion.
  have row_32 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_33)
  -- chapter_94_line_31: GL tag disintegration.
  have row_31 : (gl_implication18 N N N add) := by
    exact row_32.1.2
  -- chapter_94_line_30: GL tag expansion.
  have row_30 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication18] using (row_31)
  -- chapter_94_line_29: GL tag implication grouping.
  have row_29 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_34
    have auxiliary_source_2 := row_36
    have auxiliary_source_3 := row_38
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_94_line_28: GL tag task formulation.
  have row_28 : (mul v1 v6 v7) := by
    exact assumption_28
  -- chapter_94_line_27: GL tag implication.
  have row_27 : (gl_and21 v1 N v6 N v7 N) := by
    apply row_29
    exact row_28
  -- chapter_94_line_26: GL tag expansion.
  have row_26 : (((N v1) ∧ (N v6)) ∧ (N v7)) := by
    simpa only [gl_and21] using (row_27)
  -- chapter_94_line_25: GL tag disintegration.
  have row_25 : (N v6) := by
    exact row_26.1.2
  -- chapter_94_line_24: GL tag implication.
  have row_24 : (gl_and21 v1 N v4 N v5 N) := by
    apply row_29
    exact row_49
  -- chapter_94_line_23: GL tag expansion.
  have row_23 : (((N v1) ∧ (N v4)) ∧ (N v5)) := by
    simpa only [gl_and21] using (row_24)
  -- chapter_94_line_22: GL tag disintegration.
  have row_22 : (N v1) := by
    exact row_23.1.1
  -- chapter_94_line_21: GL tag disintegration.
  have row_21 : (N v5) := by
    exact row_23.2
  -- chapter_94_line_20: GL tag task formulation.
  have row_20 : (mul v1 v2 v3) := by
    exact assumption_20
  -- chapter_94_line_19: GL tag implication.
  have row_19 : (gl_and21 v1 N v2 N v3 N) := by
    apply row_29
    exact row_20
  -- chapter_94_line_18: GL tag expansion.
  have row_18 : (((N v1) ∧ (N v2)) ∧ (N v3)) := by
    simpa only [gl_and21] using (row_19)
  -- chapter_94_line_17: GL tag disintegration.
  have row_17 : (N v2) := by
    exact row_18.1.2
  -- chapter_94_line_16: GL tag disintegration.
  have row_16 : (N v3) := by
    exact row_18.2
  -- chapter_94_line_15: GL tag theorem.
  have row_15 := peano_source_011 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_94_line_14: GL tag implication.
  have row_14 : (v2 = v6) := by
    apply row_15
    exact row_50
    exact row_17
  -- chapter_94_line_13: GL tag equality1.
  have row_13 : (mul v1 v6 v3) := by
    have equality_source := row_20
    have equality_step_1 := row_14
    cases equality_step_1
    exact equality_source
  -- chapter_94_line_12: GL tag implication.
  have row_12 : (v3 = v7) := by
    apply row_40
    exact row_22
    exact row_25
    exact row_13
    exact row_28
  -- chapter_94_line_11: GL tag theorem.
  have row_11 := peano_source_025 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_94_line_10: GL tag implication.
  have row_10 : (zero = v5) := by
    apply row_11
    exact row_48
    exact row_22
  -- chapter_94_line_9: GL tag symmetry of equality.
  have row_9 : (v5 = zero) := by
    exact Eq.symm row_10
  -- chapter_94_line_8: GL tag implication.
  have row_8 : (gl_existence1 N v3 v5 add) := by
    apply row_30
    exact row_16
    exact row_21
  -- chapter_94_line_7: GL tag expansion.
  have row_7 : (¬ (∀ (v8 : α), ((N v8) → (¬ (add v3 v5 v8))))) := by
    simpa only [gl_existence1] using (row_8)
  have exists_row_7 : ∃ (v8 : α), ((N v8) ∧ (add v3 v5 v8)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_7
      intro v8 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v8, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v8, witness_row_7⟩ := exists_row_7
  -- chapter_94_line_6: GL tag disintegration.
  have row_6 : (add v3 v5 v8) := by
    exact witness_row_7.2
  -- chapter_94_line_5: GL tag equality1.
  have row_5 : (add v3 zero v8) := by
    have equality_source := row_6
    have equality_step_1 := row_9
    cases equality_step_1
    exact equality_source
  -- chapter_94_line_4: GL tag implication.
  have row_4 : (v3 = v8) := by
    apply row_15
    exact row_5
    exact row_16
  -- chapter_94_line_3: GL tag symmetry of equality.
  have row_3 : (v8 = v3) := by
    exact Eq.symm row_4
  -- chapter_94_line_2: GL tag equality2.
  have row_2 : (v8 = v7) := by
    exact Eq.trans row_3 row_12
  -- chapter_94_line_1: GL tag equality1.
  have row_1 : (add v3 v5 v7) := by
    have equality_source := row_6
    have equality_step_1 := row_2
    cases equality_step_1
    exact equality_source
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
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
    (assumption_31 : (succ previous v4))
    (assumption_28 : (mul v1 v2 v3))
    (assumption_20 : (mul v1 v4 v5))
    (assumption_11 : (add v2 v4 v6))
    (assumption_6 : (mul v1 v6 v7))
    (assumption_4 : ((mul v1 v2 v3) → (∀ (w1 : α), ((mul v1 previous w1) → (∀ (w2 : α) (w3 : α), ((mul v1 w2 w3) → ((add v2 previous w2) → (add v3 w1 w3))))))))
    : (add v3 v5 v7) := by
  -- chapter_95_line_57: GL tag task formulation.
  have row_57 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_95_line_56: GL tag expansion.
  have row_56 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_57)
  -- chapter_95_line_55: GL tag disintegration.
  have row_55 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_56.1
  -- chapter_95_line_54: GL tag expansion.
  have row_54 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_55)
  -- chapter_95_line_53: GL tag disintegration.
  have row_53 : (gl_fXY succ N N) := by
    exact row_54.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_95_line_52: GL tag expansion.
  have row_52 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_53)
  -- chapter_95_line_51: GL tag disintegration.
  have row_51 : (gl_implication0 succ N) := by
    exact row_52.1.1.1
  -- chapter_95_line_50: GL tag expansion.
  have row_50 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_51)
  -- chapter_95_line_49: GL tag disintegration.
  have row_49 : (gl_fXYZ add N N N) := by
    exact row_54.1.1.1.1.1.1.1.1.1.2
  -- chapter_95_line_48: GL tag expansion.
  have row_48 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_49)
  -- chapter_95_line_47: GL tag disintegration.
  have row_47 : (gl_implication18 N N N add) := by
    exact row_48.1.2
  -- chapter_95_line_46: GL tag expansion.
  have row_46 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication18] using (row_47)
  -- chapter_95_line_45: GL tag disintegration.
  have row_45 : (gl_fXYZ mul N N N) := by
    exact row_54.1.1.1.1.2
  -- chapter_95_line_44: GL tag expansion.
  have row_44 : (((((gl_implication11 mul N) ∧ (gl_implication12 mul N)) ∧ (gl_implication13 mul N)) ∧ (gl_implication18 N N N mul)) ∧ (gl_implication19 N N mul)) := by
    simpa only [gl_fXYZ] using (row_45)
  -- chapter_95_line_43: GL tag disintegration.
  have row_43 : (gl_implication13 mul N) := by
    exact row_44.1.1.2
  -- chapter_95_line_42: GL tag expansion.
  have row_42 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_43)
  -- chapter_95_line_41: GL tag disintegration.
  have row_41 : (gl_implication12 mul N) := by
    exact row_44.1.1.1.2
  -- chapter_95_line_40: GL tag expansion.
  have row_40 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_41)
  -- chapter_95_line_39: GL tag disintegration.
  have row_39 : (gl_implication11 mul N) := by
    exact row_44.1.1.1.1
  -- chapter_95_line_38: GL tag expansion.
  have row_38 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_39)
  -- chapter_95_line_37: GL tag disintegration.
  have row_37 : (gl_implication18 N N N mul) := by
    exact row_44.1.2
  -- chapter_95_line_36: GL tag expansion.
  have row_36 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication18] using (row_37)
  -- chapter_95_line_35: GL tag disintegration.
  have row_35 : (gl_implication28 N succ mul add) := by
    exact row_54.2
  -- chapter_95_line_34: GL tag expansion.
  have row_34 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((mul w3 w1 w4) → (∀ (w5 : α), ((mul w3 w2 w5) → (add w4 w3 w5))))))))) := by
    simpa only [gl_implication28] using (row_35)
  -- chapter_95_line_33: GL tag disintegration.
  have row_33 : (gl_implication23 N succ add) := by
    exact row_54.1.1.1.1.1.1.2
  -- chapter_95_line_32: GL tag expansion.
  have row_32 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication23] using (row_33)
  -- chapter_95_line_31: GL tag recursion.
  have row_31 : (succ previous v4) := by
    exact assumption_31
  -- chapter_95_line_30: GL tag implication.
  have row_30 : (N previous) := by
    apply row_50
    exact row_31
  -- chapter_95_line_29: GL tag implication grouping.
  have row_29 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_38
    have auxiliary_source_2 := row_40
    have auxiliary_source_3 := row_42
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_95_line_28: GL tag task formulation.
  have row_28 : (mul v1 v2 v3) := by
    exact assumption_28
  -- chapter_95_line_27: GL tag implication.
  have row_27 : (gl_and21 v1 N v2 N v3 N) := by
    apply row_29
    exact row_28
  -- chapter_95_line_26: GL tag expansion.
  have row_26 : (((N v1) ∧ (N v2)) ∧ (N v3)) := by
    simpa only [gl_and21] using (row_27)
  -- chapter_95_line_25: GL tag disintegration.
  have row_25 : (N v2) := by
    exact row_26.1.2
  -- chapter_95_line_24: GL tag implication.
  have row_24 : (gl_existence1 N v2 previous add) := by
    apply row_46
    exact row_25
    exact row_30
  -- chapter_95_line_23: GL tag expansion.
  have row_23 : (¬ (∀ (v13 : α), ((N v13) → (¬ (add v2 previous v13))))) := by
    simpa only [gl_existence1] using (row_24)
  have exists_row_23 : ∃ (v13 : α), ((N v13) ∧ (add v2 previous v13)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_23
      intro v13 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v13, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v13, witness_row_23⟩ := exists_row_23
  -- chapter_95_line_22: GL tag disintegration.
  have row_22 : (N v13) := by
    exact witness_row_23.1
  -- chapter_95_line_21: GL tag disintegration.
  have row_21 : (add v2 previous v13) := by
    exact witness_row_23.2
  -- chapter_95_line_20: GL tag task formulation.
  have row_20 : (mul v1 v4 v5) := by
    exact assumption_20
  -- chapter_95_line_19: GL tag implication.
  have row_19 : (gl_and21 v1 N v4 N v5 N) := by
    apply row_29
    exact row_20
  -- chapter_95_line_18: GL tag expansion.
  have row_18 : (((N v1) ∧ (N v4)) ∧ (N v5)) := by
    simpa only [gl_and21] using (row_19)
  -- chapter_95_line_17: GL tag disintegration.
  have row_17 : (N v1) := by
    exact row_18.1.1
  -- chapter_95_line_16: GL tag implication.
  have row_16 : (gl_existence1 N v1 v13 mul) := by
    apply row_36
    exact row_17
    exact row_22
  -- chapter_95_line_15: GL tag implication.
  have row_15 : (gl_existence1 N v1 previous mul) := by
    apply row_36
    exact row_17
    exact row_30
  -- chapter_95_line_14: GL tag expansion.
  have row_14 : (¬ (∀ (v8 : α), ((N v8) → (¬ (mul v1 previous v8))))) := by
    simpa only [gl_existence1] using (row_15)
  have exists_row_14 : ∃ (v8 : α), ((N v8) ∧ (mul v1 previous v8)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_14
      intro v8 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v8, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v8, witness_row_14⟩ := exists_row_14
  -- chapter_95_line_13: GL tag disintegration.
  have row_13 : (mul v1 previous v8) := by
    exact witness_row_14.2
  -- chapter_95_line_12: GL tag implication.
  have row_12 : (add v8 v1 v5) := by
    apply row_34
    exact row_30
    exact row_31
    exact row_13
    exact row_20
  -- chapter_95_line_11: GL tag task formulation.
  have row_11 : (add v2 v4 v6) := by
    exact assumption_11
  -- chapter_95_line_10: GL tag implication.
  have row_10 : (succ v13 v6) := by
    apply row_32
    exact row_30
    exact row_31
    exact row_21
    exact row_11
  -- chapter_95_line_9: GL tag expansion.
  have row_9 : (¬ (∀ (v9 : α), ((N v9) → (¬ (mul v1 v13 v9))))) := by
    simpa only [gl_existence1] using (row_16)
  have exists_row_9 : ∃ (v9 : α), ((N v9) ∧ (mul v1 v13 v9)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_9
      intro v9 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v9, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v9, witness_row_9⟩ := exists_row_9
  -- chapter_95_line_8: GL tag disintegration.
  have row_8 : (mul v1 v13 v9) := by
    exact witness_row_9.2
  -- chapter_95_line_7: GL tag theorem.
  have row_7 := peano_source_008 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_95_line_6: GL tag task formulation.
  have row_6 : (mul v1 v6 v7) := by
    exact assumption_6
  -- chapter_95_line_5: GL tag implication.
  have row_5 : (add v9 v1 v7) := by
    apply row_7
    exact row_6
    exact row_8
    exact row_10
  -- chapter_95_line_4: GL tag recursion.
  have row_4 : ((mul v1 v2 v3) → (∀ (w1 : α), ((mul v1 previous w1) → (∀ (w2 : α) (w3 : α), ((mul v1 w2 w3) → ((add v2 previous w2) → (add v3 w1 w3))))))) := by
    exact assumption_4
  -- chapter_95_line_3: GL tag implication.
  have row_3 : (add v3 v8 v9) := by
    apply row_4
    exact row_28
    exact row_13
    exact row_8
    exact row_21
  -- chapter_95_line_2: GL tag theorem.
  have row_2 := peano_source_022 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_95_line_1: GL tag implication.
  have row_1 : (add v3 v5 v7) := by
    apply row_2
    exact row_5
    exact row_3
    exact row_12
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
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
    have typingRule := peano_source_047_induction_typing N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact typingRule v2 v4 v6 premise_4
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1.1
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
    have zeroRule := peano_source_047_check_zero N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact zeroRule v1 v2 v3 zero v5 v6 v7 rfl base_premise_4 base_premise_2 base_premise_3 base_premise_1
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
    have stepRule := peano_source_047_check_induction_condition N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact stepRule induction_n v1 v2 v3 induction_m v5 v6 v7 step_induction_assumption_1 step_premise_1 step_premise_2 step_premise_4 step_premise_3 step_induction_assumption_2
  have inductionProperty : (∀ (v1 : α) (v2 : α) (v3 : α), ((mul v1 v2 v3) → (∀ (v5 : α), ((mul v1 v4 v5) → (∀ (v6 : α) (v7 : α), ((mul v1 v6 v7) → ((add v2 v4 v6) → (add v3 v5 v7)))))))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v1 : α) (v2 : α) (v3 : α), ((mul v1 v2 v3) → (∀ (v5 : α), ((mul v1 induction_value v5) → (∀ (v6 : α) (v7 : α), ((mul v1 v6 v7) → ((add v2 induction_value v6) → (add v3 v5 v7)))))))))
      v4
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v1 v2 v3 premise_1 v5 premise_2 v6 v7 premise_3 premise_4

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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (assumption_2 : (mul v1 v2 zero))
    : (N v2) := by
  -- chapter_99_line_10: GL tag task formulation.
  have row_10 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_99_line_9: GL tag expansion.
  have row_9 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_10)
  -- chapter_99_line_8: GL tag disintegration.
  have row_8 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_9.1
  -- chapter_99_line_7: GL tag expansion.
  have row_7 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_8)
  -- chapter_99_line_6: GL tag disintegration.
  have row_6 : (gl_fXYZ mul N N N) := by
    exact row_7.1.1.1.1.2
  -- chapter_99_line_5: GL tag expansion.
  have row_5 : (((((gl_implication11 mul N) ∧ (gl_implication12 mul N)) ∧ (gl_implication13 mul N)) ∧ (gl_implication18 N N N mul)) ∧ (gl_implication19 N N mul)) := by
    simpa only [gl_fXYZ] using (row_6)
  -- chapter_99_line_4: GL tag disintegration.
  have row_4 : (gl_implication12 mul N) := by
    exact row_5.1.1.1.2
  -- chapter_99_line_3: GL tag expansion.
  have row_3 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_4)
  -- chapter_99_line_2: GL tag task formulation.
  have row_2 : (mul v1 v2 zero) := by
    exact assumption_2
  -- chapter_99_line_1: GL tag implication.
  have row_1 : (N v2) := by
    apply row_3
    exact row_2
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (assumption_15 : (mul v1 v2 zero))
    (assumption_7 : (v2 = zero))
    : (mul v2 v1 zero) := by
  -- chapter_100_line_30: GL tag task formulation.
  have row_30 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_100_line_29: GL tag expansion.
  have row_29 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_30)
  -- chapter_100_line_28: GL tag disintegration.
  have row_28 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_29.1
  -- chapter_100_line_27: GL tag expansion.
  have row_27 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_28)
  -- chapter_100_line_26: GL tag disintegration.
  have row_26 : (gl_fXYZ mul N N N) := by
    exact row_27.1.1.1.1.2
  -- chapter_100_line_25: GL tag expansion.
  have row_25 : (((((gl_implication11 mul N) ∧ (gl_implication12 mul N)) ∧ (gl_implication13 mul N)) ∧ (gl_implication18 N N N mul)) ∧ (gl_implication19 N N mul)) := by
    simpa only [gl_fXYZ] using (row_26)
  -- chapter_100_line_24: GL tag disintegration.
  have row_24 : (gl_implication13 mul N) := by
    exact row_25.1.1.2
  -- chapter_100_line_23: GL tag expansion.
  have row_23 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_24)
  -- chapter_100_line_22: GL tag disintegration.
  have row_22 : (gl_implication12 mul N) := by
    exact row_25.1.1.1.2
  -- chapter_100_line_21: GL tag expansion.
  have row_21 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_22)
  -- chapter_100_line_20: GL tag disintegration.
  have row_20 : (gl_implication11 mul N) := by
    exact row_25.1.1.1.1
  -- chapter_100_line_19: GL tag expansion.
  have row_19 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_20)
  -- chapter_100_line_18: GL tag disintegration.
  have row_18 : (gl_implication18 N N N mul) := by
    exact row_25.1.2
  -- chapter_100_line_17: GL tag expansion.
  have row_17 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication18] using (row_18)
  -- chapter_100_line_16: GL tag implication grouping.
  have row_16 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_19
    have auxiliary_source_2 := row_21
    have auxiliary_source_3 := row_23
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_100_line_15: GL tag task formulation.
  have row_15 : (mul v1 v2 zero) := by
    exact assumption_15
  -- chapter_100_line_14: GL tag implication.
  have row_14 : (gl_and21 v1 N v2 N zero N) := by
    apply row_16
    exact row_15
  -- chapter_100_line_13: GL tag expansion.
  have row_13 : (((N v1) ∧ (N v2)) ∧ (N zero)) := by
    simpa only [gl_and21] using (row_14)
  -- chapter_100_line_12: GL tag disintegration.
  have row_12 : (N v1) := by
    exact row_13.1.1
  -- chapter_100_line_11: GL tag disintegration.
  have row_11 : (N v2) := by
    exact row_13.1.2
  -- chapter_100_line_10: GL tag implication.
  have row_10 : (gl_existence1 N v2 v1 mul) := by
    apply row_17
    exact row_11
    exact row_12
  -- chapter_100_line_9: GL tag expansion.
  have row_9 : (¬ (∀ (v3 : α), ((N v3) → (¬ (mul v2 v1 v3))))) := by
    simpa only [gl_existence1] using (row_10)
  have exists_row_9 : ∃ (v3 : α), ((N v3) ∧ (mul v2 v1 v3)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_9
      intro v3 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v3, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v3, witness_row_9⟩ := exists_row_9
  -- chapter_100_line_8: GL tag disintegration.
  have row_8 : (mul v2 v1 v3) := by
    exact witness_row_9.2
  -- chapter_100_line_7: GL tag recursion.
  have row_7 : (v2 = zero) := by
    exact assumption_7
  -- chapter_100_line_6: GL tag equality1.
  have row_6 : (mul zero v1 v3) := by
    have equality_source := row_8
    have equality_step_1 := row_7
    cases equality_step_1
    exact equality_source
  -- chapter_100_line_5: GL tag theorem.
  have row_5 := peano_source_048 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_100_line_4: GL tag implication.
  have row_4 : (zero = v3) := by
    apply row_5
    exact row_12
    exact row_6
  -- chapter_100_line_3: GL tag symmetry of equality.
  have row_3 : (v3 = zero) := by
    exact Eq.symm row_4
  -- chapter_100_line_2: GL tag symmetry of equality.
  have row_2 : (zero = v2) := by
    exact Eq.symm row_7
  -- chapter_100_line_1: GL tag equality1.
  have row_1 : (mul v2 v1 zero) := by
    have equality_source := row_6
    have equality_step_1 := row_2
    cases equality_step_1
    have equality_step_2 := row_3
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (assumption_22 : (succ previous v2))
    (assumption_18 : (mul v1 v2 zero))
    : (mul v2 v1 zero) := by
  -- chapter_101_line_46: GL tag task formulation.
  have row_46 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_101_line_45: GL tag expansion.
  have row_45 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_46)
  -- chapter_101_line_44: GL tag disintegration.
  have row_44 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_45.1
  -- chapter_101_line_43: GL tag expansion.
  have row_43 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_44)
  -- chapter_101_line_42: GL tag disintegration.
  have row_42 : (gl_implication28 N succ mul add) := by
    exact row_43.2
  -- chapter_101_line_41: GL tag expansion.
  have row_41 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((mul w3 w1 w4) → (∀ (w5 : α), ((mul w3 w2 w5) → (add w4 w3 w5))))))))) := by
    simpa only [gl_implication28] using (row_42)
  -- chapter_101_line_40: GL tag disintegration.
  have row_40 : (gl_implication25 N zero mul) := by
    exact row_43.1.1.1.2
  -- chapter_101_line_39: GL tag expansion.
  have row_39 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((mul w1 zero w2) → (w2 = zero))))) := by
    simpa only [gl_implication25] using (row_40)
  -- chapter_101_line_38: GL tag disintegration.
  have row_38 : (gl_fXY succ N N) := by
    exact row_43.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_101_line_37: GL tag expansion.
  have row_37 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_38)
  -- chapter_101_line_36: GL tag disintegration.
  have row_36 : (gl_implication0 succ N) := by
    exact row_37.1.1.1
  -- chapter_101_line_35: GL tag expansion.
  have row_35 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_36)
  -- chapter_101_line_34: GL tag disintegration.
  have row_34 : (gl_implication1 succ N) := by
    exact row_37.1.1.2
  -- chapter_101_line_33: GL tag expansion.
  have row_33 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using (row_34)
  -- chapter_101_line_32: GL tag disintegration.
  have row_32 : (gl_fXYZ mul N N N) := by
    exact row_43.1.1.1.1.2
  -- chapter_101_line_31: GL tag expansion.
  have row_31 : (((((gl_implication11 mul N) ∧ (gl_implication12 mul N)) ∧ (gl_implication13 mul N)) ∧ (gl_implication18 N N N mul)) ∧ (gl_implication19 N N mul)) := by
    simpa only [gl_fXYZ] using (row_32)
  -- chapter_101_line_30: GL tag disintegration.
  have row_30 : (gl_implication13 mul N) := by
    exact row_31.1.1.2
  -- chapter_101_line_29: GL tag expansion.
  have row_29 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_30)
  -- chapter_101_line_28: GL tag disintegration.
  have row_28 : (gl_implication12 mul N) := by
    exact row_31.1.1.1.2
  -- chapter_101_line_27: GL tag expansion.
  have row_27 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_28)
  -- chapter_101_line_26: GL tag disintegration.
  have row_26 : (gl_implication11 mul N) := by
    exact row_31.1.1.1.1
  -- chapter_101_line_25: GL tag expansion.
  have row_25 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_26)
  -- chapter_101_line_24: GL tag disintegration.
  have row_24 : (gl_implication18 N N N mul) := by
    exact row_31.1.2
  -- chapter_101_line_23: GL tag expansion.
  have row_23 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication18] using (row_24)
  -- chapter_101_line_22: GL tag recursion.
  have row_22 : (succ previous v2) := by
    exact assumption_22
  -- chapter_101_line_21: GL tag implication.
  have row_21 : (N previous) := by
    apply row_35
    exact row_22
  -- chapter_101_line_20: GL tag implication.
  have row_20 : (N v2) := by
    apply row_33
    exact row_22
  -- chapter_101_line_19: GL tag implication grouping.
  have row_19 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_25
    have auxiliary_source_2 := row_27
    have auxiliary_source_3 := row_29
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_101_line_18: GL tag task formulation.
  have row_18 : (mul v1 v2 zero) := by
    exact assumption_18
  -- chapter_101_line_17: GL tag implication.
  have row_17 : (gl_and21 v1 N v2 N zero N) := by
    apply row_19
    exact row_18
  -- chapter_101_line_16: GL tag expansion.
  have row_16 : (((N v1) ∧ (N v2)) ∧ (N zero)) := by
    simpa only [gl_and21] using (row_17)
  -- chapter_101_line_15: GL tag disintegration.
  have row_15 : (N v1) := by
    exact row_16.1.1
  -- chapter_101_line_14: GL tag implication.
  have row_14 : (gl_existence1 N v2 v1 mul) := by
    apply row_23
    exact row_20
    exact row_15
  -- chapter_101_line_13: GL tag implication.
  have row_13 : (gl_existence1 N v1 previous mul) := by
    apply row_23
    exact row_15
    exact row_21
  -- chapter_101_line_12: GL tag expansion.
  have row_12 : (¬ (∀ (v7 : α), ((N v7) → (¬ (mul v1 previous v7))))) := by
    simpa only [gl_existence1] using (row_13)
  have exists_row_12 : ∃ (v7 : α), ((N v7) ∧ (mul v1 previous v7)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_12
      intro v7 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v7, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v7, witness_row_12⟩ := exists_row_12
  -- chapter_101_line_11: GL tag disintegration.
  have row_11 : (mul v1 previous v7) := by
    exact witness_row_12.2
  -- chapter_101_line_10: GL tag disintegration.
  have row_10 : (N v7) := by
    exact witness_row_12.1
  -- chapter_101_line_9: GL tag implication.
  have row_9 : (add v7 v1 zero) := by
    apply row_41
    exact row_21
    exact row_22
    exact row_11
    exact row_18
  -- chapter_101_line_8: GL tag theorem.
  have row_8 := peano_source_029 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_101_line_7: GL tag implication.
  have row_7 : (zero = v1) := by
    apply row_8
    exact row_9
    exact row_10
  -- chapter_101_line_6: GL tag symmetry of equality.
  have row_6 : (v1 = zero) := by
    exact Eq.symm row_7
  -- chapter_101_line_5: GL tag expansion.
  have row_5 : (¬ (∀ (v3 : α), ((N v3) → (¬ (mul v2 v1 v3))))) := by
    simpa only [gl_existence1] using (row_14)
  have exists_row_5 : ∃ (v3 : α), ((N v3) ∧ (mul v2 v1 v3)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_5
      intro v3 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v3, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v3, witness_row_5⟩ := exists_row_5
  -- chapter_101_line_4: GL tag disintegration.
  have row_4 : (mul v2 v1 v3) := by
    exact witness_row_5.2
  -- chapter_101_line_3: GL tag equality1.
  have row_3 : (mul v2 zero v3) := by
    have equality_source := row_4
    have equality_step_1 := row_6
    cases equality_step_1
    exact equality_source
  -- chapter_101_line_2: GL tag implication.
  have row_2 : (v3 = zero) := by
    apply row_39
    exact row_20
    exact row_3
  -- chapter_101_line_1: GL tag equality1.
  have row_1 : (mul v2 v1 zero) := by
    have equality_source := row_4
    have equality_step_1 := row_2
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((mul v1 v2 zero) → (mul v2 v1 zero))) := by
  intro v1
  intro v2
  intro premise_1
  have inductionMember : N v2 := by
    have typingRule := peano_source_049_induction_typing N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact typingRule v1 v2 premise_1
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v1 : α), ((mul v1 zero zero) → (mul zero v1 zero))) := by
    intro v1
    intro base_premise_1
    have zeroRule := peano_source_049_check_zero N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact zeroRule v1 zero base_premise_1 rfl
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v1 : α), ((mul v1 induction_n zero) → (mul induction_n v1 zero))) → ∀ induction_m, succ induction_n induction_m → (∀ (v1 : α), ((mul v1 induction_m zero) → (mul induction_m v1 zero))) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro v1
    intro step_premise_1
    have step_induction_assumption_1 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have stepRule := peano_source_049_check_induction_condition N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact stepRule induction_n v1 induction_m step_induction_assumption_1 step_premise_1
  have inductionProperty : (∀ (v1 : α), ((mul v1 v2 zero) → (mul v2 v1 zero))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v1 : α), ((mul v1 induction_value zero) → (mul induction_value v1 zero))))
      v2
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v1 premise_1

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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v1 : α)
    (assumption_1 : (N v1))
    : (N v1) := by
  -- chapter_126_line_1: GL tag task formulation.
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (assumption_14 : (mul one v1 v2))
    (assumption_9 : (v1 = zero))
    : (v1 = v2) := by
  -- chapter_127_line_34: GL tag task formulation.
  have row_34 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_127_line_33: GL tag expansion.
  have row_33 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_34)
  -- chapter_127_line_32: GL tag disintegration.
  have row_32 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_33.1
  -- chapter_127_line_31: GL tag expansion.
  have row_31 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_32)
  -- chapter_127_line_30: GL tag disintegration.
  have row_30 : (gl_implication25 N zero mul) := by
    exact row_31.1.1.1.2
  -- chapter_127_line_29: GL tag expansion.
  have row_29 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((mul w1 zero w2) → (w2 = zero))))) := by
    simpa only [gl_implication25] using (row_30)
  -- chapter_127_line_28: GL tag disintegration.
  have row_28 : (gl_fXYZ mul N N N) := by
    exact row_31.1.1.1.1.2
  -- chapter_127_line_27: GL tag expansion.
  have row_27 : (((((gl_implication11 mul N) ∧ (gl_implication12 mul N)) ∧ (gl_implication13 mul N)) ∧ (gl_implication18 N N N mul)) ∧ (gl_implication19 N N mul)) := by
    simpa only [gl_fXYZ] using (row_28)
  -- chapter_127_line_26: GL tag disintegration.
  have row_26 : (gl_implication13 mul N) := by
    exact row_27.1.1.2
  -- chapter_127_line_25: GL tag expansion.
  have row_25 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_26)
  -- chapter_127_line_24: GL tag disintegration.
  have row_24 : (gl_implication12 mul N) := by
    exact row_27.1.1.1.2
  -- chapter_127_line_23: GL tag expansion.
  have row_23 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_24)
  -- chapter_127_line_22: GL tag disintegration.
  have row_22 : (gl_implication11 mul N) := by
    exact row_27.1.1.1.1
  -- chapter_127_line_21: GL tag expansion.
  have row_21 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_22)
  -- chapter_127_line_20: GL tag disintegration.
  have row_20 : (gl_implication22 N zero add) := by
    exact row_31.1.1.1.1.1.1.1.2
  -- chapter_127_line_19: GL tag expansion.
  have row_19 : (∀ (w1 : α) (w2 : α), ((w1 = w2) → ((N w1) → ((N w2) → (add w1 zero w2))))) := by
    simpa only [gl_implication22] using (row_20)
  -- chapter_127_line_18: GL tag disintegration.
  have row_18 : (N zero) := by
    exact row_31.1.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_127_line_17: GL tag disintegration.
  have row_17 : (gl_implication21 N zero add) := by
    exact row_31.1.1.1.1.1.1.1.1.2
  -- chapter_127_line_16: GL tag expansion.
  have row_16 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((add w1 zero w2) → (w1 = w2))))) := by
    simpa only [gl_implication21] using (row_17)
  -- chapter_127_line_15: GL tag implication grouping.
  have row_15 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_21
    have auxiliary_source_2 := row_23
    have auxiliary_source_3 := row_25
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_127_line_14: GL tag task formulation.
  have row_14 : (mul one v1 v2) := by
    exact assumption_14
  -- chapter_127_line_13: GL tag implication.
  have row_13 : (gl_and21 one N v1 N v2 N) := by
    apply row_15
    exact row_14
  -- chapter_127_line_12: GL tag expansion.
  have row_12 : (((N one) ∧ (N v1)) ∧ (N v2)) := by
    simpa only [gl_and21] using (row_13)
  -- chapter_127_line_11: GL tag disintegration.
  have row_11 : (N one) := by
    exact row_12.1.1
  -- chapter_127_line_10: GL tag disintegration.
  have row_10 : (N v1) := by
    exact row_12.1.2
  -- chapter_127_line_9: GL tag recursion.
  have row_9 : (v1 = zero) := by
    exact assumption_9
  -- chapter_127_line_8: GL tag equality1.
  have row_8 : (mul one zero v2) := by
    have equality_source := row_14
    have equality_step_1 := row_9
    cases equality_step_1
    exact equality_source
  -- chapter_127_line_7: GL tag implication.
  have row_7 : (v2 = zero) := by
    apply row_29
    exact row_11
    exact row_8
  -- chapter_127_line_6: GL tag symmetry of equality.
  have row_6 : (zero = v2) := by
    exact Eq.symm row_7
  -- chapter_127_line_5: GL tag symmetry of equality.
  have row_5 : (zero = v1) := by
    exact Eq.symm row_9
  -- chapter_127_line_4: GL tag implication.
  have row_4 : (add zero zero v1) := by
    apply row_19
    exact row_5
    exact row_18
    exact row_10
  -- chapter_127_line_3: GL tag equality1.
  have row_3 : (add zero zero zero) := by
    have equality_source := row_4
    have equality_step_1 := row_9
    cases equality_step_1
    exact equality_source
  -- chapter_127_line_2: GL tag equality1.
  have row_2 : (add v1 zero v2) := by
    have equality_source := row_3
    have equality_step_1 := row_5
    cases equality_step_1
    have equality_step_2 := row_6
    cases equality_step_2
    exact equality_source
  -- chapter_127_line_1: GL tag implication.
  have row_1 : (v1 = v2) := by
    apply row_16
    exact row_10
    exact row_2
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (assumption_13 : (succ previous v1))
    (assumption_8 : (mul one v1 v2))
    (assumption_4 : ((N previous) → (∀ (w1 : α), ((mul one previous w1) → (previous = w1)))))
    : (v1 = v2) := by
  -- chapter_128_line_34: GL tag task formulation.
  have row_34 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_128_line_33: GL tag expansion.
  have row_33 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_34)
  -- chapter_128_line_32: GL tag disintegration.
  have row_32 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_33.1
  -- chapter_128_line_31: GL tag expansion.
  have row_31 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_32)
  -- chapter_128_line_30: GL tag disintegration.
  have row_30 : (gl_fXY succ N N) := by
    exact row_31.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_128_line_29: GL tag expansion.
  have row_29 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_30)
  -- chapter_128_line_28: GL tag disintegration.
  have row_28 : (gl_implication0 succ N) := by
    exact row_29.1.1.1
  -- chapter_128_line_27: GL tag expansion.
  have row_27 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_28)
  -- chapter_128_line_26: GL tag disintegration.
  have row_26 : (gl_implication1 succ N) := by
    exact row_29.1.1.2
  -- chapter_128_line_25: GL tag expansion.
  have row_25 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using (row_26)
  -- chapter_128_line_24: GL tag disintegration.
  have row_24 : (succ zero one) := by
    exact row_33.2
  -- chapter_128_line_23: GL tag disintegration.
  have row_23 : (gl_fXYZ mul N N N) := by
    exact row_31.1.1.1.1.2
  -- chapter_128_line_22: GL tag expansion.
  have row_22 : (((((gl_implication11 mul N) ∧ (gl_implication12 mul N)) ∧ (gl_implication13 mul N)) ∧ (gl_implication18 N N N mul)) ∧ (gl_implication19 N N mul)) := by
    simpa only [gl_fXYZ] using (row_23)
  -- chapter_128_line_21: GL tag disintegration.
  have row_21 : (gl_implication18 N N N mul) := by
    exact row_22.1.2
  -- chapter_128_line_20: GL tag expansion.
  have row_20 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication18] using (row_21)
  -- chapter_128_line_19: GL tag disintegration.
  have row_19 : (gl_implication28 N succ mul add) := by
    exact row_31.2
  -- chapter_128_line_18: GL tag expansion.
  have row_18 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((mul w3 w1 w4) → (∀ (w5 : α), ((mul w3 w2 w5) → (add w4 w3 w5))))))))) := by
    simpa only [gl_implication28] using (row_19)
  -- chapter_128_line_17: GL tag implication grouping.
  have row_17 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (gl_and20 w1 N w2 N))) := by
    have auxiliary_source_1 := row_27
    have auxiliary_source_2 := row_25
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_premise
    exact ⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_premise)⟩
  -- chapter_128_line_16: GL tag implication.
  have row_16 : (gl_and20 zero N one N) := by
    apply row_17
    exact row_24
  -- chapter_128_line_15: GL tag expansion.
  have row_15 : ((N zero) ∧ (N one)) := by
    simpa only [gl_and20] using (row_16)
  -- chapter_128_line_14: GL tag disintegration.
  have row_14 : (N one) := by
    exact row_15.2
  -- chapter_128_line_13: GL tag recursion.
  have row_13 : (succ previous v1) := by
    exact assumption_13
  -- chapter_128_line_12: GL tag implication.
  have row_12 : (N previous) := by
    apply row_27
    exact row_13
  -- chapter_128_line_11: GL tag implication.
  have row_11 : (gl_existence1 N one previous mul) := by
    apply row_20
    exact row_14
    exact row_12
  -- chapter_128_line_10: GL tag expansion.
  have row_10 : (¬ (∀ (v3 : α), ((N v3) → (¬ (mul one previous v3))))) := by
    simpa only [gl_existence1] using (row_11)
  have exists_row_10 : ∃ (v3 : α), ((N v3) ∧ (mul one previous v3)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_10
      intro v3 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v3, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v3, witness_row_10⟩ := exists_row_10
  -- chapter_128_line_9: GL tag disintegration.
  have row_9 : (mul one previous v3) := by
    exact witness_row_10.2
  -- chapter_128_line_8: GL tag task formulation.
  have row_8 : (mul one v1 v2) := by
    exact assumption_8
  -- chapter_128_line_7: GL tag implication.
  have row_7 : (add v3 one v2) := by
    apply row_18
    exact row_12
    exact row_13
    exact row_9
    exact row_8
  -- chapter_128_line_6: GL tag theorem.
  have row_6 := peano_source_015 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_128_line_5: GL tag implication.
  have row_5 : (succ v3 v2) := by
    apply row_6
    exact row_7
  -- chapter_128_line_4: GL tag recursion.
  have row_4 : ((N previous) → (∀ (w1 : α), ((mul one previous w1) → (previous = w1)))) := by
    exact assumption_4
  -- chapter_128_line_3: GL tag implication.
  have row_3 : (previous = v3) := by
    apply row_4
    exact row_12
    exact row_9
  -- chapter_128_line_2: GL tag theorem.
  have row_2 := peano_source_002 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_128_line_1: GL tag implication.
  have row_1 : (v1 = v2) := by
    apply row_2
    exact row_13
    exact row_5
    exact row_3
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α), ((N v1) → (∀ (v2 : α), ((mul one v1 v2) → (v1 = v2))))) := by
  intro v1
  intro premise_1
  intro v2
  intro premise_2
  have inductionMember : N v1 := by
    have typingRule := peano_source_060_induction_typing N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact typingRule v1 premise_1
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v2 : α), ((mul one zero v2) → (zero = v2))) := by
    intro v2
    intro base_premise_1
    have zeroRule := peano_source_060_check_zero N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact zeroRule zero v2 base_premise_1 rfl
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v2 : α), ((mul one induction_n v2) → (induction_n = v2))) → ∀ induction_m, succ induction_n induction_m → (∀ (v2 : α), ((mul one induction_m v2) → (induction_m = v2))) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro v2
    intro step_premise_1
    have step_induction_assumption_1 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have step_induction_assumption_2 :
        ((N induction_n) → (∀ (w1 : α), ((mul one induction_n w1) → (induction_n = w1)))) := by
      intro step_induction_assumption_2_premise_1
      intro w1
      intro step_induction_assumption_2_premise_2
      apply induction_hypothesis
      all_goals assumption
    have stepRule := peano_source_060_check_induction_condition N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact stepRule induction_n induction_m v2 step_induction_assumption_1 step_premise_1 step_induction_assumption_2
  have inductionProperty : (∀ (v2 : α), ((mul one v1 v2) → (v1 = v2))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v2 : α), ((mul one induction_value v2) → (induction_value = v2))))
      v1
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v2 premise_2

private theorem peano_source_021_induction_typing
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v2 : α)
    (v4 : α)
    (v6 : α)
    (assumption_2 : (add v4 v2 v6))
    : (N v2) := by
  -- chapter_33_line_10: GL tag task formulation.
  have row_10 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_33_line_9: GL tag expansion.
  have row_9 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_10)
  -- chapter_33_line_8: GL tag disintegration.
  have row_8 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_9.1
  -- chapter_33_line_7: GL tag expansion.
  have row_7 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_8)
  -- chapter_33_line_6: GL tag disintegration.
  have row_6 : (gl_fXYZ add N N N) := by
    exact row_7.1.1.1.1.1.1.1.1.1.2
  -- chapter_33_line_5: GL tag expansion.
  have row_5 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_6)
  -- chapter_33_line_4: GL tag disintegration.
  have row_4 : (gl_implication12 add N) := by
    exact row_5.1.1.1.2
  -- chapter_33_line_3: GL tag expansion.
  have row_3 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_4)
  -- chapter_33_line_2: GL tag task formulation.
  have row_2 : (add v4 v2 v6) := by
    exact assumption_2
  -- chapter_33_line_1: GL tag implication.
  have row_1 : (N v2) := by
    apply row_3
    exact row_2
  exact row_1

private theorem peano_source_021_check_zero
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (v4 : α)
    (v5 : α)
    (v6 : α)
    (assumption_27 : (v2 = zero))
    (assumption_26 : (add v1 v2 v3))
    (assumption_24 : (add v4 v2 v6))
    (assumption_9 : (add v4 v5 v1))
    : (add v6 v5 v3) := by
  -- chapter_34_line_27: GL tag recursion.
  have row_27 : (v2 = zero) := by
    exact assumption_27
  -- chapter_34_line_26: GL tag task formulation.
  have row_26 : (add v1 v2 v3) := by
    exact assumption_26
  -- chapter_34_line_25: GL tag equality1.
  have row_25 : (add v1 zero v3) := by
    have equality_source := row_26
    have equality_step_1 := row_27
    cases equality_step_1
    exact equality_source
  -- chapter_34_line_24: GL tag task formulation.
  have row_24 : (add v4 v2 v6) := by
    exact assumption_24
  -- chapter_34_line_23: GL tag equality1.
  have row_23 : (add v4 zero v6) := by
    have equality_source := row_24
    have equality_step_1 := row_27
    cases equality_step_1
    exact equality_source
  -- chapter_34_line_22: GL tag task formulation.
  have row_22 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_34_line_21: GL tag expansion.
  have row_21 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_22)
  -- chapter_34_line_20: GL tag disintegration.
  have row_20 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_21.1
  -- chapter_34_line_19: GL tag expansion.
  have row_19 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_20)
  -- chapter_34_line_18: GL tag disintegration.
  have row_18 : (gl_fXYZ add N N N) := by
    exact row_19.1.1.1.1.1.1.1.1.1.2
  -- chapter_34_line_17: GL tag expansion.
  have row_17 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_18)
  -- chapter_34_line_16: GL tag disintegration.
  have row_16 : (gl_implication13 add N) := by
    exact row_17.1.1.2
  -- chapter_34_line_15: GL tag expansion.
  have row_15 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_16)
  -- chapter_34_line_14: GL tag disintegration.
  have row_14 : (gl_implication12 add N) := by
    exact row_17.1.1.1.2
  -- chapter_34_line_13: GL tag expansion.
  have row_13 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_14)
  -- chapter_34_line_12: GL tag disintegration.
  have row_12 : (gl_implication11 add N) := by
    exact row_17.1.1.1.1
  -- chapter_34_line_11: GL tag expansion.
  have row_11 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_12)
  -- chapter_34_line_10: GL tag implication grouping.
  have row_10 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_11
    have auxiliary_source_2 := row_13
    have auxiliary_source_3 := row_15
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_34_line_9: GL tag task formulation.
  have row_9 : (add v4 v5 v1) := by
    exact assumption_9
  -- chapter_34_line_8: GL tag implication.
  have row_8 : (gl_and21 v4 N v5 N v1 N) := by
    apply row_10
    exact row_9
  -- chapter_34_line_7: GL tag expansion.
  have row_7 : (((N v4) ∧ (N v5)) ∧ (N v1)) := by
    simpa only [gl_and21] using (row_8)
  -- chapter_34_line_6: GL tag disintegration.
  have row_6 : (N v1) := by
    exact row_7.2
  -- chapter_34_line_5: GL tag disintegration.
  have row_5 : (N v4) := by
    exact row_7.1.1
  -- chapter_34_line_4: GL tag theorem.
  have row_4 := peano_source_011 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_34_line_3: GL tag implication.
  have row_3 : (v1 = v3) := by
    apply row_4
    exact row_25
    exact row_6
  -- chapter_34_line_2: GL tag implication.
  have row_2 : (v4 = v6) := by
    apply row_4
    exact row_23
    exact row_5
  -- chapter_34_line_1: GL tag equality1.
  have row_1 : (add v6 v5 v3) := by
    have equality_source := row_9
    have equality_step_1 := row_2
    cases equality_step_1
    have equality_step_2 := row_3
    cases equality_step_2
    exact equality_source
  exact row_1

private theorem peano_source_021_check_induction_condition
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (v4 : α)
    (v5 : α)
    (v6 : α)
    (assumption_30 : (add v4 v5 v1))
    (assumption_18 : (succ previous v2))
    (assumption_12 : (add v1 v2 v3))
    (assumption_8 : (add v4 v2 v6))
    (assumption_6 : (∀ (w1 : α), ((add v1 previous w1) → ((add v4 v5 v1) → (∀ (w2 : α), ((add v4 previous w2) → (add w2 v5 w1)))))))
    : (add v6 v5 v3) := by
  -- chapter_35_line_55: GL tag task formulation.
  have row_55 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_35_line_54: GL tag expansion.
  have row_54 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_55)
  -- chapter_35_line_53: GL tag disintegration.
  have row_53 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_54.1
  -- chapter_35_line_52: GL tag expansion.
  have row_52 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_53)
  -- chapter_35_line_51: GL tag disintegration.
  have row_51 : (gl_implication23 N succ add) := by
    exact row_52.1.1.1.1.1.1.2
  -- chapter_35_line_50: GL tag expansion.
  have row_50 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication23] using (row_51)
  -- chapter_35_line_49: GL tag disintegration.
  have row_49 : (gl_fXYZ add N N N) := by
    exact row_52.1.1.1.1.1.1.1.1.1.2
  -- chapter_35_line_48: GL tag expansion.
  have row_48 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_49)
  -- chapter_35_line_47: GL tag disintegration.
  have row_47 : (gl_implication18 N N N add) := by
    exact row_48.1.2
  -- chapter_35_line_46: GL tag expansion.
  have row_46 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication18] using (row_47)
  -- chapter_35_line_45: GL tag disintegration.
  have row_45 : (gl_implication13 add N) := by
    exact row_48.1.1.2
  -- chapter_35_line_44: GL tag expansion.
  have row_44 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_45)
  -- chapter_35_line_43: GL tag disintegration.
  have row_43 : (gl_implication12 add N) := by
    exact row_48.1.1.1.2
  -- chapter_35_line_42: GL tag expansion.
  have row_42 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_43)
  -- chapter_35_line_41: GL tag disintegration.
  have row_41 : (gl_implication11 add N) := by
    exact row_48.1.1.1.1
  -- chapter_35_line_40: GL tag expansion.
  have row_40 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_41)
  -- chapter_35_line_39: GL tag disintegration.
  have row_39 : (gl_implication9 N zero succ) := by
    exact row_52.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_35_line_38: GL tag expansion.
  have row_38 : (∀ (w1 : α), ((N w1) → (¬ (succ w1 zero)))) := by
    simpa only [gl_implication9] using (row_39)
  -- chapter_35_line_37: GL tag disintegration.
  have row_37 : (gl_fXY succ N N) := by
    exact row_52.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_35_line_36: GL tag expansion.
  have row_36 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_37)
  -- chapter_35_line_35: GL tag disintegration.
  have row_35 : (gl_implication6 N N succ) := by
    exact row_36.1.2
  -- chapter_35_line_34: GL tag expansion.
  have row_34 : (∀ (w1 : α), ((N w1) → (gl_existence0 N w1 succ))) := by
    simpa only [gl_implication6] using (row_35)
  -- chapter_35_line_33: GL tag disintegration.
  have row_33 : (gl_implication0 succ N) := by
    exact row_36.1.1.1
  -- chapter_35_line_32: GL tag expansion.
  have row_32 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_33)
  -- chapter_35_line_31: GL tag implication grouping.
  have row_31 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_40
    have auxiliary_source_2 := row_42
    have auxiliary_source_3 := row_44
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_35_line_30: GL tag task formulation.
  have row_30 : (add v4 v5 v1) := by
    exact assumption_30
  -- chapter_35_line_29: GL tag implication.
  have row_29 : (gl_and21 v4 N v5 N v1 N) := by
    apply row_31
    exact row_30
  -- chapter_35_line_28: GL tag expansion.
  have row_28 : (((N v4) ∧ (N v5)) ∧ (N v1)) := by
    simpa only [gl_and21] using (row_29)
  -- chapter_35_line_27: GL tag disintegration.
  have row_27 : (N v5) := by
    exact row_28.1.2
  -- chapter_35_line_26: GL tag disintegration.
  have row_26 : (N v1) := by
    exact row_28.2
  -- chapter_35_line_25: GL tag disintegration.
  have row_25 : (N v4) := by
    exact row_28.1.1
  -- chapter_35_line_24: GL tag implication grouping.
  have row_24 : (∀ (w1 : α), ((N w1) → (gl_and3 N w1 succ zero))) := by
    have auxiliary_source_1 := row_34
    have auxiliary_source_2 := row_38
    intro auxiliary_argument_1
    intro auxiliary_premise
    exact ⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_premise)⟩
  -- chapter_35_line_23: GL tag implication.
  have row_23 : (gl_and3 N v5 succ zero) := by
    apply row_24
    exact row_27
  -- chapter_35_line_22: GL tag expansion.
  have row_22 : ((gl_existence0 N v5 succ) ∧ (¬ (succ v5 zero))) := by
    simpa only [gl_and3] using (row_23)
  -- chapter_35_line_21: GL tag disintegration.
  have row_21 : (gl_existence0 N v5 succ) := by
    exact row_22.1
  -- chapter_35_line_20: GL tag expansion.
  have row_20 : (¬ (∀ (v7 : α), ((N v7) → (¬ (succ v5 v7))))) := by
    simpa only [gl_existence0] using (row_21)
  have exists_row_20 : ∃ (v7 : α), ((N v7) ∧ (succ v5 v7)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_20
      intro v7 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v7, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v7, witness_row_20⟩ := exists_row_20
  -- chapter_35_line_19: GL tag disintegration.
  have row_19 : (succ v5 v7) := by
    exact witness_row_20.2
  -- chapter_35_line_18: GL tag recursion.
  have row_18 : (succ previous v2) := by
    exact assumption_18
  -- chapter_35_line_17: GL tag implication.
  have row_17 : (N previous) := by
    apply row_32
    exact row_18
  -- chapter_35_line_16: GL tag implication.
  have row_16 : (gl_existence1 N v1 previous add) := by
    apply row_46
    exact row_26
    exact row_17
  -- chapter_35_line_15: GL tag implication.
  have row_15 : (gl_existence1 N v4 previous add) := by
    apply row_46
    exact row_25
    exact row_17
  -- chapter_35_line_14: GL tag expansion.
  have row_14 : (¬ (∀ (v14 : α), ((N v14) → (¬ (add v1 previous v14))))) := by
    simpa only [gl_existence1] using (row_16)
  have exists_row_14 : ∃ (v14 : α), ((N v14) ∧ (add v1 previous v14)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_14
      intro v14 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v14, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v14, witness_row_14⟩ := exists_row_14
  -- chapter_35_line_13: GL tag disintegration.
  have row_13 : (add v1 previous v14) := by
    exact witness_row_14.2
  -- chapter_35_line_12: GL tag task formulation.
  have row_12 : (add v1 v2 v3) := by
    exact assumption_12
  -- chapter_35_line_11: GL tag implication.
  have row_11 : (succ v14 v3) := by
    apply row_50
    exact row_17
    exact row_18
    exact row_13
    exact row_12
  -- chapter_35_line_10: GL tag expansion.
  have row_10 : (¬ (∀ (v8 : α), ((N v8) → (¬ (add v4 previous v8))))) := by
    simpa only [gl_existence1] using (row_15)
  have exists_row_10 : ∃ (v8 : α), ((N v8) ∧ (add v4 previous v8)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_10
      intro v8 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v8, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v8, witness_row_10⟩ := exists_row_10
  -- chapter_35_line_9: GL tag disintegration.
  have row_9 : (add v4 previous v8) := by
    exact witness_row_10.2
  -- chapter_35_line_8: GL tag task formulation.
  have row_8 : (add v4 v2 v6) := by
    exact assumption_8
  -- chapter_35_line_7: GL tag implication.
  have row_7 : (succ v8 v6) := by
    apply row_50
    exact row_17
    exact row_18
    exact row_9
    exact row_8
  -- chapter_35_line_6: GL tag recursion.
  have row_6 : (∀ (w1 : α), ((add v1 previous w1) → ((add v4 v5 v1) → (∀ (w2 : α), ((add v4 previous w2) → (add w2 v5 w1)))))) := by
    exact assumption_6
  -- chapter_35_line_5: GL tag implication.
  have row_5 : (add v8 v5 v14) := by
    apply row_6
    exact row_13
    exact row_30
    exact row_9
  -- chapter_35_line_4: GL tag theorem.
  have row_4 := peano_source_018 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_35_line_3: GL tag implication.
  have row_3 : (add v8 v7 v3) := by
    apply row_4
    exact row_5
    exact row_19
    exact row_11
  -- chapter_35_line_2: GL tag theorem.
  have row_2 := peano_source_020 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_35_line_1: GL tag implication.
  have row_1 : (add v6 v5 v3) := by
    apply row_2
    exact row_3
    exact row_7
    exact row_19
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α) (v5 : α), ((add v4 v5 v1) → (∀ (v6 : α), ((add v4 v2 v6) → (add v6 v5 v3))))))) := by
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
    have typingRule := peano_source_021_induction_typing N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact typingRule v2 v4 v6 premise_3
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v1 : α) (v3 : α), ((add v1 zero v3) → (∀ (v4 : α) (v5 : α), ((add v4 v5 v1) → (∀ (v6 : α), ((add v4 zero v6) → (add v6 v5 v3))))))) := by
    intro v1
    intro v3
    intro base_premise_1
    intro v4
    intro v5
    intro base_premise_2
    intro v6
    intro base_premise_3
    have zeroRule := peano_source_021_check_zero N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact zeroRule v1 zero v3 v4 v5 v6 rfl base_premise_1 base_premise_3 base_premise_2
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v1 : α) (v3 : α), ((add v1 induction_n v3) → (∀ (v4 : α) (v5 : α), ((add v4 v5 v1) → (∀ (v6 : α), ((add v4 induction_n v6) → (add v6 v5 v3))))))) → ∀ induction_m, succ induction_n induction_m → (∀ (v1 : α) (v3 : α), ((add v1 induction_m v3) → (∀ (v4 : α) (v5 : α), ((add v4 v5 v1) → (∀ (v6 : α), ((add v4 induction_m v6) → (add v6 v5 v3))))))) := by
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
        (succ induction_n induction_m) := by
      exact induction_successor
    have step_induction_assumption_2 :
        (∀ (w1 : α), ((add v1 induction_n w1) → ((add v4 v5 v1) → (∀ (w2 : α), ((add v4 induction_n w2) → (add w2 v5 w1)))))) := by
      intro w1
      intro step_induction_assumption_2_premise_1
      intro step_induction_assumption_2_premise_2
      intro w2
      intro step_induction_assumption_2_premise_3
      apply induction_hypothesis
      all_goals assumption
    have stepRule := peano_source_021_check_induction_condition N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact stepRule induction_n v1 induction_m v3 v4 v5 v6 step_premise_2 step_induction_assumption_1 step_premise_1 step_premise_3 step_induction_assumption_2
  have inductionProperty : (∀ (v1 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α) (v5 : α), ((add v4 v5 v1) → (∀ (v6 : α), ((add v4 v2 v6) → (add v6 v5 v3))))))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v1 : α) (v3 : α), ((add v1 induction_value v3) → (∀ (v4 : α) (v5 : α), ((add v4 v5 v1) → (∀ (v6 : α), ((add v4 induction_value v6) → (add v6 v5 v3))))))))
      v2
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v1 v3 premise_1 v4 v5 premise_2 v6 premise_3

private theorem peano_source_028_induction_typing
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v2 : α)
    (v4 : α)
    (v5 : α)
    (assumption_4 : (mul v4 v2 v5))
    : (N v2) := by
  -- chapter_46_line_17: GL tag task formulation.
  have row_17 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_46_line_16: GL tag expansion.
  have row_16 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_17)
  -- chapter_46_line_15: GL tag disintegration.
  have row_15 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_16.1
  -- chapter_46_line_14: GL tag expansion.
  have row_14 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_15)
  -- chapter_46_line_13: GL tag disintegration.
  have row_13 : (gl_fXYZ mul N N N) := by
    exact row_14.1.1.1.1.2
  -- chapter_46_line_12: GL tag expansion.
  have row_12 : (((((gl_implication11 mul N) ∧ (gl_implication12 mul N)) ∧ (gl_implication13 mul N)) ∧ (gl_implication18 N N N mul)) ∧ (gl_implication19 N N mul)) := by
    simpa only [gl_fXYZ] using (row_13)
  -- chapter_46_line_11: GL tag disintegration.
  have row_11 : (gl_implication13 mul N) := by
    exact row_12.1.1.2
  -- chapter_46_line_10: GL tag expansion.
  have row_10 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_11)
  -- chapter_46_line_9: GL tag disintegration.
  have row_9 : (gl_implication12 mul N) := by
    exact row_12.1.1.1.2
  -- chapter_46_line_8: GL tag expansion.
  have row_8 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_9)
  -- chapter_46_line_7: GL tag disintegration.
  have row_7 : (gl_implication11 mul N) := by
    exact row_12.1.1.1.1
  -- chapter_46_line_6: GL tag expansion.
  have row_6 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_7)
  -- chapter_46_line_5: GL tag implication grouping.
  have row_5 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_6
    have auxiliary_source_2 := row_8
    have auxiliary_source_3 := row_10
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_46_line_4: GL tag task formulation.
  have row_4 : (mul v4 v2 v5) := by
    exact assumption_4
  -- chapter_46_line_3: GL tag implication.
  have row_3 : (gl_and21 v4 N v2 N v5 N) := by
    apply row_5
    exact row_4
  -- chapter_46_line_2: GL tag expansion.
  have row_2 : (((N v4) ∧ (N v2)) ∧ (N v5)) := by
    simpa only [gl_and21] using (row_3)
  -- chapter_46_line_1: GL tag disintegration.
  have row_1 : (N v2) := by
    exact row_2.1.2
  exact row_1

private theorem peano_source_028_check_zero
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (v4 : α)
    (v5 : α)
    (assumption_58 : (v2 = zero))
    (assumption_57 : (mul v1 v2 v3))
    (assumption_55 : (mul v4 v2 v5))
    (assumption_52 : (succ v4 v1))
    : (add v2 v5 v3) := by
  -- chapter_47_line_58: GL tag recursion.
  have row_58 : (v2 = zero) := by
    exact assumption_58
  -- chapter_47_line_57: GL tag task formulation.
  have row_57 : (mul v1 v2 v3) := by
    exact assumption_57
  -- chapter_47_line_56: GL tag equality1.
  have row_56 : (mul v1 zero v3) := by
    have equality_source := row_57
    have equality_step_1 := row_58
    cases equality_step_1
    exact equality_source
  -- chapter_47_line_55: GL tag task formulation.
  have row_55 : (mul v4 v2 v5) := by
    exact assumption_55
  -- chapter_47_line_54: GL tag equality1.
  have row_54 : (mul v4 zero v5) := by
    have equality_source := row_55
    have equality_step_1 := row_58
    cases equality_step_1
    exact equality_source
  -- chapter_47_line_53: GL tag task formulation.
  have row_53 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_47_line_52: GL tag task formulation.
  have row_52 : (succ v4 v1) := by
    exact assumption_52
  -- chapter_47_line_51: GL tag theorem.
  have row_51 := peano_source_027 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_47_line_50: GL tag implication.
  have row_50 : (add v4 one v1) := by
    apply row_51
    exact row_52
  -- chapter_47_line_49: GL tag expansion.
  have row_49 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_53)
  -- chapter_47_line_48: GL tag disintegration.
  have row_48 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_49.1
  -- chapter_47_line_47: GL tag expansion.
  have row_47 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_48)
  -- chapter_47_line_46: GL tag disintegration.
  have row_46 : (gl_fXYZ mul N N N) := by
    exact row_47.1.1.1.1.2
  -- chapter_47_line_45: GL tag expansion.
  have row_45 : (((((gl_implication11 mul N) ∧ (gl_implication12 mul N)) ∧ (gl_implication13 mul N)) ∧ (gl_implication18 N N N mul)) ∧ (gl_implication19 N N mul)) := by
    simpa only [gl_fXYZ] using (row_46)
  -- chapter_47_line_44: GL tag disintegration.
  have row_44 : (gl_implication13 mul N) := by
    exact row_45.1.1.2
  -- chapter_47_line_43: GL tag expansion.
  have row_43 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_44)
  -- chapter_47_line_42: GL tag disintegration.
  have row_42 : (gl_implication12 mul N) := by
    exact row_45.1.1.1.2
  -- chapter_47_line_41: GL tag expansion.
  have row_41 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_42)
  -- chapter_47_line_40: GL tag disintegration.
  have row_40 : (gl_implication11 mul N) := by
    exact row_45.1.1.1.1
  -- chapter_47_line_39: GL tag expansion.
  have row_39 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_40)
  -- chapter_47_line_38: GL tag disintegration.
  have row_38 : (gl_fXYZ add N N N) := by
    exact row_47.1.1.1.1.1.1.1.1.1.2
  -- chapter_47_line_37: GL tag expansion.
  have row_37 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_38)
  -- chapter_47_line_36: GL tag disintegration.
  have row_36 : (gl_implication13 add N) := by
    exact row_37.1.1.2
  -- chapter_47_line_35: GL tag expansion.
  have row_35 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_36)
  -- chapter_47_line_34: GL tag disintegration.
  have row_34 : (gl_implication12 add N) := by
    exact row_37.1.1.1.2
  -- chapter_47_line_33: GL tag expansion.
  have row_33 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_34)
  -- chapter_47_line_32: GL tag disintegration.
  have row_32 : (gl_implication11 add N) := by
    exact row_37.1.1.1.1
  -- chapter_47_line_31: GL tag expansion.
  have row_31 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_32)
  -- chapter_47_line_30: GL tag disintegration.
  have row_30 : (gl_implication25 N zero mul) := by
    exact row_47.1.1.1.2
  -- chapter_47_line_29: GL tag expansion.
  have row_29 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((mul w1 zero w2) → (w2 = zero))))) := by
    simpa only [gl_implication25] using (row_30)
  -- chapter_47_line_28: GL tag disintegration.
  have row_28 : (gl_implication18 N N N add) := by
    exact row_37.1.2
  -- chapter_47_line_27: GL tag expansion.
  have row_27 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication18] using (row_28)
  -- chapter_47_line_26: GL tag implication grouping.
  have row_26 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_39
    have auxiliary_source_2 := row_41
    have auxiliary_source_3 := row_43
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_47_line_25: GL tag implication.
  have row_25 : (gl_and21 v4 N v2 N v5 N) := by
    apply row_26
    exact row_55
  -- chapter_47_line_24: GL tag expansion.
  have row_24 : (((N v4) ∧ (N v2)) ∧ (N v5)) := by
    simpa only [gl_and21] using (row_25)
  -- chapter_47_line_23: GL tag disintegration.
  have row_23 : (N v5) := by
    exact row_24.2
  -- chapter_47_line_22: GL tag disintegration.
  have row_22 : (N v2) := by
    exact row_24.1.2
  -- chapter_47_line_21: GL tag implication grouping.
  have row_21 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_31
    have auxiliary_source_2 := row_33
    have auxiliary_source_3 := row_35
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_47_line_20: GL tag implication.
  have row_20 : (gl_and21 v4 N one N v1 N) := by
    apply row_21
    exact row_50
  -- chapter_47_line_19: GL tag expansion.
  have row_19 : (((N v4) ∧ (N one)) ∧ (N v1)) := by
    simpa only [gl_and21] using (row_20)
  -- chapter_47_line_18: GL tag disintegration.
  have row_18 : (N v1) := by
    exact row_19.2
  -- chapter_47_line_17: GL tag disintegration.
  have row_17 : (N v4) := by
    exact row_19.1.1
  -- chapter_47_line_16: GL tag theorem.
  have row_16 := peano_source_025 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_47_line_15: GL tag implication.
  have row_15 : (zero = v3) := by
    apply row_16
    exact row_56
    exact row_18
  -- chapter_47_line_14: GL tag implication.
  have row_14 : (zero = v5) := by
    apply row_16
    exact row_54
    exact row_17
  -- chapter_47_line_13: GL tag implication.
  have row_13 : (v5 = zero) := by
    apply row_29
    exact row_17
    exact row_54
  -- chapter_47_line_12: GL tag symmetry of equality.
  have row_12 : (zero = v2) := by
    exact Eq.symm row_58
  -- chapter_47_line_11: GL tag equality2.
  have row_11 : (v5 = v2) := by
    exact Eq.trans row_13 row_12
  -- chapter_47_line_10: GL tag implication.
  have row_10 : (gl_existence1 N v5 v2 add) := by
    apply row_27
    exact row_23
    exact row_22
  -- chapter_47_line_9: GL tag expansion.
  have row_9 : (¬ (∀ (v6 : α), ((N v6) → (¬ (add v5 v2 v6))))) := by
    simpa only [gl_existence1] using (row_10)
  have exists_row_9 : ∃ (v6 : α), ((N v6) ∧ (add v5 v2 v6)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_9
      intro v6 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v6, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v6, witness_row_9⟩ := exists_row_9
  -- chapter_47_line_8: GL tag disintegration.
  have row_8 : (add v5 v2 v6) := by
    exact witness_row_9.2
  -- chapter_47_line_7: GL tag equality1.
  have row_7 : (add v5 zero v6) := by
    have equality_source := row_8
    have equality_step_1 := row_58
    cases equality_step_1
    exact equality_source
  -- chapter_47_line_6: GL tag theorem.
  have row_6 := peano_source_011 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_47_line_5: GL tag implication.
  have row_5 : (v5 = v6) := by
    apply row_6
    exact row_7
    exact row_23
  -- chapter_47_line_4: GL tag symmetry of equality.
  have row_4 : (v6 = v5) := by
    exact Eq.symm row_5
  -- chapter_47_line_3: GL tag equality2.
  have row_3 : (v6 = zero) := by
    exact Eq.trans row_4 row_13
  -- chapter_47_line_2: GL tag equality2.
  have row_2 : (v6 = v3) := by
    exact Eq.trans row_3 row_15
  -- chapter_47_line_1: GL tag equality1.
  have row_1 : (add v2 v5 v3) := by
    have equality_source := row_7
    have equality_step_1 := row_11
    cases equality_step_1
    have equality_step_2 := row_14
    cases equality_step_2
    have equality_step_3 := row_2
    cases equality_step_3
    exact equality_source
  exact row_1

private theorem peano_source_028_check_induction_condition
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (v4 : α)
    (v5 : α)
    (assumption_32 : (succ previous v2))
    (assumption_29 : (mul v1 v2 v3))
    (assumption_21 : (mul v4 v2 v5))
    (assumption_9 : (succ v4 v1))
    (assumption_8 : (∀ (w1 : α), ((mul v1 previous w1) → (∀ (w2 : α), ((mul v4 previous w2) → ((succ v4 v1) → (add previous w2 w1)))))))
    : (add v2 v5 v3) := by
  -- chapter_48_line_56: GL tag task formulation.
  have row_56 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_48_line_55: GL tag expansion.
  have row_55 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_56)
  -- chapter_48_line_54: GL tag disintegration.
  have row_54 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_55.1
  -- chapter_48_line_53: GL tag expansion.
  have row_53 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_54)
  -- chapter_48_line_52: GL tag disintegration.
  have row_52 : (gl_fXYZ add N N N) := by
    exact row_53.1.1.1.1.1.1.1.1.1.2
  -- chapter_48_line_51: GL tag expansion.
  have row_51 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_52)
  -- chapter_48_line_50: GL tag disintegration.
  have row_50 : (gl_implication18 N N N add) := by
    exact row_51.1.2
  -- chapter_48_line_49: GL tag expansion.
  have row_49 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication18] using (row_50)
  -- chapter_48_line_48: GL tag disintegration.
  have row_48 : (gl_fXY succ N N) := by
    exact row_53.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_48_line_47: GL tag expansion.
  have row_47 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_48)
  -- chapter_48_line_46: GL tag disintegration.
  have row_46 : (gl_implication0 succ N) := by
    exact row_47.1.1.1
  -- chapter_48_line_45: GL tag expansion.
  have row_45 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_46)
  -- chapter_48_line_44: GL tag disintegration.
  have row_44 : (gl_implication28 N succ mul add) := by
    exact row_53.2
  -- chapter_48_line_43: GL tag expansion.
  have row_43 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((mul w3 w1 w4) → (∀ (w5 : α), ((mul w3 w2 w5) → (add w4 w3 w5))))))))) := by
    simpa only [gl_implication28] using (row_44)
  -- chapter_48_line_42: GL tag disintegration.
  have row_42 : (gl_fXYZ mul N N N) := by
    exact row_53.1.1.1.1.2
  -- chapter_48_line_41: GL tag expansion.
  have row_41 : (((((gl_implication11 mul N) ∧ (gl_implication12 mul N)) ∧ (gl_implication13 mul N)) ∧ (gl_implication18 N N N mul)) ∧ (gl_implication19 N N mul)) := by
    simpa only [gl_fXYZ] using (row_42)
  -- chapter_48_line_40: GL tag disintegration.
  have row_40 : (gl_implication13 mul N) := by
    exact row_41.1.1.2
  -- chapter_48_line_39: GL tag expansion.
  have row_39 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_40)
  -- chapter_48_line_38: GL tag disintegration.
  have row_38 : (gl_implication12 mul N) := by
    exact row_41.1.1.1.2
  -- chapter_48_line_37: GL tag expansion.
  have row_37 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_38)
  -- chapter_48_line_36: GL tag disintegration.
  have row_36 : (gl_implication11 mul N) := by
    exact row_41.1.1.1.1
  -- chapter_48_line_35: GL tag expansion.
  have row_35 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_36)
  -- chapter_48_line_34: GL tag disintegration.
  have row_34 : (gl_implication18 N N N mul) := by
    exact row_41.1.2
  -- chapter_48_line_33: GL tag expansion.
  have row_33 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication18] using (row_34)
  -- chapter_48_line_32: GL tag recursion.
  have row_32 : (succ previous v2) := by
    exact assumption_32
  -- chapter_48_line_31: GL tag implication.
  have row_31 : (N previous) := by
    apply row_45
    exact row_32
  -- chapter_48_line_30: GL tag implication grouping.
  have row_30 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_35
    have auxiliary_source_2 := row_37
    have auxiliary_source_3 := row_39
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_48_line_29: GL tag task formulation.
  have row_29 : (mul v1 v2 v3) := by
    exact assumption_29
  -- chapter_48_line_28: GL tag implication.
  have row_28 : (gl_and21 v1 N v2 N v3 N) := by
    apply row_30
    exact row_29
  -- chapter_48_line_27: GL tag expansion.
  have row_27 : (((N v1) ∧ (N v2)) ∧ (N v3)) := by
    simpa only [gl_and21] using (row_28)
  -- chapter_48_line_26: GL tag disintegration.
  have row_26 : (N v1) := by
    exact row_27.1.1
  -- chapter_48_line_25: GL tag implication.
  have row_25 : (gl_existence1 N v1 previous mul) := by
    apply row_33
    exact row_26
    exact row_31
  -- chapter_48_line_24: GL tag expansion.
  have row_24 : (¬ (∀ (v13 : α), ((N v13) → (¬ (mul v1 previous v13))))) := by
    simpa only [gl_existence1] using (row_25)
  have exists_row_24 : ∃ (v13 : α), ((N v13) ∧ (mul v1 previous v13)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_24
      intro v13 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v13, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v13, witness_row_24⟩ := exists_row_24
  -- chapter_48_line_23: GL tag disintegration.
  have row_23 : (mul v1 previous v13) := by
    exact witness_row_24.2
  -- chapter_48_line_22: GL tag implication.
  have row_22 : (add v13 v1 v3) := by
    apply row_43
    exact row_31
    exact row_32
    exact row_23
    exact row_29
  -- chapter_48_line_21: GL tag task formulation.
  have row_21 : (mul v4 v2 v5) := by
    exact assumption_21
  -- chapter_48_line_20: GL tag implication.
  have row_20 : (gl_and21 v4 N v2 N v5 N) := by
    apply row_30
    exact row_21
  -- chapter_48_line_19: GL tag expansion.
  have row_19 : (((N v4) ∧ (N v2)) ∧ (N v5)) := by
    simpa only [gl_and21] using (row_20)
  -- chapter_48_line_18: GL tag disintegration.
  have row_18 : (N v4) := by
    exact row_19.1.1
  -- chapter_48_line_17: GL tag implication.
  have row_17 : (gl_existence1 N v4 previous mul) := by
    apply row_33
    exact row_18
    exact row_31
  -- chapter_48_line_16: GL tag expansion.
  have row_16 : (¬ (∀ (v7 : α), ((N v7) → (¬ (mul v4 previous v7))))) := by
    simpa only [gl_existence1] using (row_17)
  have exists_row_16 : ∃ (v7 : α), ((N v7) ∧ (mul v4 previous v7)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_16
      intro v7 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v7, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v7, witness_row_16⟩ := exists_row_16
  -- chapter_48_line_15: GL tag disintegration.
  have row_15 : (N v7) := by
    exact witness_row_16.1
  -- chapter_48_line_14: GL tag disintegration.
  have row_14 : (mul v4 previous v7) := by
    exact witness_row_16.2
  -- chapter_48_line_13: GL tag implication.
  have row_13 : (gl_existence1 N v7 v1 add) := by
    apply row_49
    exact row_15
    exact row_26
  -- chapter_48_line_12: GL tag implication.
  have row_12 : (add v7 v4 v5) := by
    apply row_43
    exact row_31
    exact row_32
    exact row_14
    exact row_21
  -- chapter_48_line_11: GL tag expansion.
  have row_11 : (¬ (∀ (v6 : α), ((N v6) → (¬ (add v7 v1 v6))))) := by
    simpa only [gl_existence1] using (row_13)
  have exists_row_11 : ∃ (v6 : α), ((N v6) ∧ (add v7 v1 v6)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_11
      intro v6 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v6, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v6, witness_row_11⟩ := exists_row_11
  -- chapter_48_line_10: GL tag disintegration.
  have row_10 : (add v7 v1 v6) := by
    exact witness_row_11.2
  -- chapter_48_line_9: GL tag task formulation.
  have row_9 : (succ v4 v1) := by
    exact assumption_9
  -- chapter_48_line_8: GL tag recursion.
  have row_8 : (∀ (w1 : α), ((mul v1 previous w1) → (∀ (w2 : α), ((mul v4 previous w2) → ((succ v4 v1) → (add previous w2 w1)))))) := by
    exact assumption_8
  -- chapter_48_line_7: GL tag implication.
  have row_7 : (add previous v7 v13) := by
    apply row_8
    exact row_23
    exact row_14
    exact row_9
  -- chapter_48_line_6: GL tag theorem.
  have row_6 := peano_source_022 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_48_line_5: GL tag implication.
  have row_5 : (add previous v6 v3) := by
    apply row_6
    exact row_22
    exact row_7
    exact row_10
  -- chapter_48_line_4: GL tag theorem.
  have row_4 := peano_source_017 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_48_line_3: GL tag implication.
  have row_3 : (succ v5 v6) := by
    apply row_4
    exact row_10
    exact row_12
    exact row_9
  -- chapter_48_line_2: GL tag theorem.
  have row_2 := peano_source_020 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_48_line_1: GL tag implication.
  have row_1 : (add v2 v5 v3) := by
    apply row_2
    exact row_5
    exact row_32
    exact row_3
  exact row_1

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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α) (v3 : α), ((mul v1 v2 v3) → (∀ (v4 : α) (v5 : α), ((mul v4 v2 v5) → ((succ v4 v1) → (add v2 v5 v3)))))) := by
  intro v1
  intro v2
  intro v3
  intro premise_1
  intro v4
  intro v5
  intro premise_2
  intro premise_3
  have inductionMember : N v2 := by
    have typingRule := peano_source_028_induction_typing N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact typingRule v2 v4 v5 premise_2
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v1 : α) (v3 : α), ((mul v1 zero v3) → (∀ (v4 : α) (v5 : α), ((mul v4 zero v5) → ((succ v4 v1) → (add zero v5 v3)))))) := by
    intro v1
    intro v3
    intro base_premise_1
    intro v4
    intro v5
    intro base_premise_2
    intro base_premise_3
    have zeroRule := peano_source_028_check_zero N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact zeroRule v1 zero v3 v4 v5 rfl base_premise_1 base_premise_2 base_premise_3
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v1 : α) (v3 : α), ((mul v1 induction_n v3) → (∀ (v4 : α) (v5 : α), ((mul v4 induction_n v5) → ((succ v4 v1) → (add induction_n v5 v3)))))) → ∀ induction_m, succ induction_n induction_m → (∀ (v1 : α) (v3 : α), ((mul v1 induction_m v3) → (∀ (v4 : α) (v5 : α), ((mul v4 induction_m v5) → ((succ v4 v1) → (add induction_m v5 v3)))))) := by
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
        (succ induction_n induction_m) := by
      exact induction_successor
    have step_induction_assumption_2 :
        (∀ (w1 : α), ((mul v1 induction_n w1) → (∀ (w2 : α), ((mul v4 induction_n w2) → ((succ v4 v1) → (add induction_n w2 w1)))))) := by
      intro w1
      intro step_induction_assumption_2_premise_1
      intro w2
      intro step_induction_assumption_2_premise_2
      intro step_induction_assumption_2_premise_3
      apply induction_hypothesis
      all_goals assumption
    have stepRule := peano_source_028_check_induction_condition N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact stepRule induction_n v1 induction_m v3 v4 v5 step_induction_assumption_1 step_premise_1 step_premise_2 step_premise_3 step_induction_assumption_2
  have inductionProperty : (∀ (v1 : α) (v3 : α), ((mul v1 v2 v3) → (∀ (v4 : α) (v5 : α), ((mul v4 v2 v5) → ((succ v4 v1) → (add v2 v5 v3)))))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v1 : α) (v3 : α), ((mul v1 induction_value v3) → (∀ (v4 : α) (v5 : α), ((mul v4 induction_value v5) → ((succ v4 v1) → (add induction_value v5 v3)))))))
      v2
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v1 v3 premise_1 v4 v5 premise_2 premise_3

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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (assumption_4 : (add v1 v2 v3))
    : (N v2) := by
  -- chapter_61_line_17: GL tag task formulation.
  have row_17 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_61_line_16: GL tag expansion.
  have row_16 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_17)
  -- chapter_61_line_15: GL tag disintegration.
  have row_15 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_16.1
  -- chapter_61_line_14: GL tag expansion.
  have row_14 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_15)
  -- chapter_61_line_13: GL tag disintegration.
  have row_13 : (gl_fXYZ add N N N) := by
    exact row_14.1.1.1.1.1.1.1.1.1.2
  -- chapter_61_line_12: GL tag expansion.
  have row_12 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_13)
  -- chapter_61_line_11: GL tag disintegration.
  have row_11 : (gl_implication13 add N) := by
    exact row_12.1.1.2
  -- chapter_61_line_10: GL tag expansion.
  have row_10 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_11)
  -- chapter_61_line_9: GL tag disintegration.
  have row_9 : (gl_implication12 add N) := by
    exact row_12.1.1.1.2
  -- chapter_61_line_8: GL tag expansion.
  have row_8 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_9)
  -- chapter_61_line_7: GL tag disintegration.
  have row_7 : (gl_implication11 add N) := by
    exact row_12.1.1.1.1
  -- chapter_61_line_6: GL tag expansion.
  have row_6 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_7)
  -- chapter_61_line_5: GL tag implication grouping.
  have row_5 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_6
    have auxiliary_source_2 := row_8
    have auxiliary_source_3 := row_10
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_61_line_4: GL tag task formulation.
  have row_4 : (add v1 v2 v3) := by
    exact assumption_4
  -- chapter_61_line_3: GL tag implication.
  have row_3 : (gl_and21 v1 N v2 N v3 N) := by
    apply row_5
    exact row_4
  -- chapter_61_line_2: GL tag expansion.
  have row_2 : (((N v1) ∧ (N v2)) ∧ (N v3)) := by
    simpa only [gl_and21] using (row_3)
  -- chapter_61_line_1: GL tag disintegration.
  have row_1 : (N v2) := by
    exact row_2.1.2
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (v4 : α)
    (v5 : α)
    (assumption_38 : (add v1 v2 v3))
    (assumption_34 : (succ v5 v1))
    (assumption_28 : (succ v2 v4))
    (assumption_15 : (v2 = zero))
    : (add v4 v5 v3) := by
  -- chapter_62_line_57: GL tag task formulation.
  have row_57 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_62_line_56: GL tag expansion.
  have row_56 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_57)
  -- chapter_62_line_55: GL tag disintegration.
  have row_55 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_56.1
  -- chapter_62_line_54: GL tag expansion.
  have row_54 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_55)
  -- chapter_62_line_53: GL tag disintegration.
  have row_53 : (gl_implication22 N zero add) := by
    exact row_54.1.1.1.1.1.1.1.2
  -- chapter_62_line_52: GL tag expansion.
  have row_52 : (∀ (w1 : α) (w2 : α), ((w1 = w2) → ((N w1) → ((N w2) → (add w1 zero w2))))) := by
    simpa only [gl_implication22] using (row_53)
  -- chapter_62_line_51: GL tag disintegration.
  have row_51 : (gl_fXYZ add N N N) := by
    exact row_54.1.1.1.1.1.1.1.1.1.2
  -- chapter_62_line_50: GL tag expansion.
  have row_50 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_51)
  -- chapter_62_line_49: GL tag disintegration.
  have row_49 : (gl_implication13 add N) := by
    exact row_50.1.1.2
  -- chapter_62_line_48: GL tag expansion.
  have row_48 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_49)
  -- chapter_62_line_47: GL tag disintegration.
  have row_47 : (gl_implication12 add N) := by
    exact row_50.1.1.1.2
  -- chapter_62_line_46: GL tag expansion.
  have row_46 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_47)
  -- chapter_62_line_45: GL tag disintegration.
  have row_45 : (gl_implication11 add N) := by
    exact row_50.1.1.1.1
  -- chapter_62_line_44: GL tag expansion.
  have row_44 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_45)
  -- chapter_62_line_43: GL tag disintegration.
  have row_43 : (gl_implication18 N N N add) := by
    exact row_50.1.2
  -- chapter_62_line_42: GL tag expansion.
  have row_42 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication18] using (row_43)
  -- chapter_62_line_41: GL tag disintegration.
  have row_41 : (gl_implication19 N N add) := by
    exact row_50.2
  -- chapter_62_line_40: GL tag expansion.
  have row_40 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (∀ (w3 : α), ((add w1 w2 w3) → (∀ (w4 : α), ((add w1 w2 w4) → (w3 = w4))))))))) := by
    simpa only [gl_implication19] using (row_41)
  -- chapter_62_line_39: GL tag implication grouping.
  have row_39 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_44
    have auxiliary_source_2 := row_46
    have auxiliary_source_3 := row_48
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_62_line_38: GL tag task formulation.
  have row_38 : (add v1 v2 v3) := by
    exact assumption_38
  -- chapter_62_line_37: GL tag implication.
  have row_37 : (gl_and21 v1 N v2 N v3 N) := by
    apply row_39
    exact row_38
  -- chapter_62_line_36: GL tag expansion.
  have row_36 : (((N v1) ∧ (N v2)) ∧ (N v3)) := by
    simpa only [gl_and21] using (row_37)
  -- chapter_62_line_35: GL tag disintegration.
  have row_35 : (N v3) := by
    exact row_36.2
  -- chapter_62_line_34: GL tag task formulation.
  have row_34 : (succ v5 v1) := by
    exact assumption_34
  -- chapter_62_line_33: GL tag theorem.
  have row_33 := peano_source_027 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_62_line_32: GL tag implication.
  have row_32 : (add v5 one v1) := by
    apply row_33
    exact row_34
  -- chapter_62_line_31: GL tag implication.
  have row_31 : (gl_and21 v5 N one N v1 N) := by
    apply row_39
    exact row_32
  -- chapter_62_line_30: GL tag expansion.
  have row_30 : (((N v5) ∧ (N one)) ∧ (N v1)) := by
    simpa only [gl_and21] using (row_31)
  -- chapter_62_line_29: GL tag disintegration.
  have row_29 : (N v1) := by
    exact row_30.2
  -- chapter_62_line_28: GL tag task formulation.
  have row_28 : (succ v2 v4) := by
    exact assumption_28
  -- chapter_62_line_27: GL tag theorem.
  have row_27 := peano_source_010 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_62_line_26: GL tag implication.
  have row_26 : (add one v2 v4) := by
    apply row_27
    exact row_28
  -- chapter_62_line_25: GL tag implication.
  have row_25 : (gl_and21 one N v2 N v4 N) := by
    apply row_39
    exact row_26
  -- chapter_62_line_24: GL tag expansion.
  have row_24 : (((N one) ∧ (N v2)) ∧ (N v4)) := by
    simpa only [gl_and21] using (row_25)
  -- chapter_62_line_23: GL tag disintegration.
  have row_23 : (N v2) := by
    exact row_24.1.2
  -- chapter_62_line_22: GL tag implication.
  have row_22 : (gl_existence1 N v2 v1 add) := by
    apply row_42
    exact row_23
    exact row_29
  -- chapter_62_line_21: GL tag implication.
  have row_21 : (gl_existence1 N v1 v2 add) := by
    apply row_42
    exact row_29
    exact row_23
  -- chapter_62_line_20: GL tag expansion.
  have row_20 : (¬ (∀ (v10 : α), ((N v10) → (¬ (add v1 v2 v10))))) := by
    simpa only [gl_existence1] using (row_21)
  have exists_row_20 : ∃ (v10 : α), ((N v10) ∧ (add v1 v2 v10)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_20
      intro v10 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v10, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v10, witness_row_20⟩ := exists_row_20
  -- chapter_62_line_19: GL tag disintegration.
  have row_19 : (add v1 v2 v10) := by
    exact witness_row_20.2
  -- chapter_62_line_18: GL tag disintegration.
  have row_18 : (N v10) := by
    exact witness_row_20.1
  -- chapter_62_line_17: GL tag implication.
  have row_17 : (v3 = v10) := by
    apply row_40
    exact row_29
    exact row_23
    exact row_38
    exact row_19
  -- chapter_62_line_16: GL tag implication.
  have row_16 : (add v3 zero v10) := by
    apply row_52
    exact row_17
    exact row_35
    exact row_18
  -- chapter_62_line_15: GL tag recursion.
  have row_15 : (v2 = zero) := by
    exact assumption_15
  -- chapter_62_line_14: GL tag symmetry of equality.
  have row_14 : (zero = v2) := by
    exact Eq.symm row_15
  -- chapter_62_line_13: GL tag equality1.
  have row_13 : (add v3 v2 v10) := by
    have equality_source := row_16
    have equality_step_1 := row_14
    cases equality_step_1
    exact equality_source
  -- chapter_62_line_12: GL tag theorem.
  have row_12 := peano_source_031 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_62_line_11: GL tag implication.
  have row_11 : (v1 = v3) := by
    apply row_12
    exact row_19
    exact row_13
  -- chapter_62_line_10: GL tag expansion.
  have row_10 : (¬ (∀ (v6 : α), ((N v6) → (¬ (add v2 v1 v6))))) := by
    simpa only [gl_existence1] using (row_22)
  have exists_row_10 : ∃ (v6 : α), ((N v6) ∧ (add v2 v1 v6)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_10
      intro v6 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v6, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v6, witness_row_10⟩ := exists_row_10
  -- chapter_62_line_9: GL tag disintegration.
  have row_9 : (add v2 v1 v6) := by
    exact witness_row_10.2
  -- chapter_62_line_8: GL tag equality1.
  have row_8 : (add zero v1 v6) := by
    have equality_source := row_9
    have equality_step_1 := row_15
    cases equality_step_1
    exact equality_source
  -- chapter_62_line_7: GL tag theorem.
  have row_7 := peano_source_032 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_62_line_6: GL tag implication.
  have row_6 : (v1 = v6) := by
    apply row_7
    exact row_29
    exact row_8
  -- chapter_62_line_5: GL tag symmetry of equality.
  have row_5 : (v6 = v1) := by
    exact Eq.symm row_6
  -- chapter_62_line_4: GL tag equality2.
  have row_4 : (v6 = v3) := by
    exact Eq.trans row_5 row_11
  -- chapter_62_line_3: GL tag theorem.
  have row_3 := peano_source_020 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_62_line_2: GL tag implication.
  have row_2 : (add v4 v5 v6) := by
    apply row_3
    exact row_9
    exact row_28
    exact row_34
  -- chapter_62_line_1: GL tag equality1.
  have row_1 : (add v4 v5 v3) := by
    have equality_source := row_2
    have equality_step_1 := row_4
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (v4 : α)
    (v5 : α)
    (assumption_19 : (succ previous v2))
    (assumption_16 : (add v1 v2 v3))
    (assumption_9 : (succ v5 v1))
    (assumption_8 : (∀ (w1 : α), ((add v1 previous w1) → (∀ (w2 : α), ((succ previous w2) → ((succ v5 v1) → (add w2 v5 w1)))))))
    (assumption_3 : (succ v2 v4))
    : (add v4 v5 v3) := by
  -- chapter_63_line_39: GL tag task formulation.
  have row_39 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_63_line_38: GL tag expansion.
  have row_38 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_39)
  -- chapter_63_line_37: GL tag disintegration.
  have row_37 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_38.1
  -- chapter_63_line_36: GL tag expansion.
  have row_36 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_37)
  -- chapter_63_line_35: GL tag disintegration.
  have row_35 : (gl_fXY succ N N) := by
    exact row_36.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_63_line_34: GL tag expansion.
  have row_34 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_35)
  -- chapter_63_line_33: GL tag disintegration.
  have row_33 : (gl_implication0 succ N) := by
    exact row_34.1.1.1
  -- chapter_63_line_32: GL tag expansion.
  have row_32 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_33)
  -- chapter_63_line_31: GL tag disintegration.
  have row_31 : (gl_fXYZ add N N N) := by
    exact row_36.1.1.1.1.1.1.1.1.1.2
  -- chapter_63_line_30: GL tag expansion.
  have row_30 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_31)
  -- chapter_63_line_29: GL tag disintegration.
  have row_29 : (gl_implication13 add N) := by
    exact row_30.1.1.2
  -- chapter_63_line_28: GL tag expansion.
  have row_28 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_29)
  -- chapter_63_line_27: GL tag disintegration.
  have row_27 : (gl_implication12 add N) := by
    exact row_30.1.1.1.2
  -- chapter_63_line_26: GL tag expansion.
  have row_26 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_27)
  -- chapter_63_line_25: GL tag disintegration.
  have row_25 : (gl_implication11 add N) := by
    exact row_30.1.1.1.1
  -- chapter_63_line_24: GL tag expansion.
  have row_24 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_25)
  -- chapter_63_line_23: GL tag disintegration.
  have row_23 : (gl_implication18 N N N add) := by
    exact row_30.1.2
  -- chapter_63_line_22: GL tag expansion.
  have row_22 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication18] using (row_23)
  -- chapter_63_line_21: GL tag disintegration.
  have row_21 : (gl_implication23 N succ add) := by
    exact row_36.1.1.1.1.1.1.2
  -- chapter_63_line_20: GL tag expansion.
  have row_20 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication23] using (row_21)
  -- chapter_63_line_19: GL tag recursion.
  have row_19 : (succ previous v2) := by
    exact assumption_19
  -- chapter_63_line_18: GL tag implication.
  have row_18 : (N previous) := by
    apply row_32
    exact row_19
  -- chapter_63_line_17: GL tag implication grouping.
  have row_17 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_24
    have auxiliary_source_2 := row_26
    have auxiliary_source_3 := row_28
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_63_line_16: GL tag task formulation.
  have row_16 : (add v1 v2 v3) := by
    exact assumption_16
  -- chapter_63_line_15: GL tag implication.
  have row_15 : (gl_and21 v1 N v2 N v3 N) := by
    apply row_17
    exact row_16
  -- chapter_63_line_14: GL tag expansion.
  have row_14 : (((N v1) ∧ (N v2)) ∧ (N v3)) := by
    simpa only [gl_and21] using (row_15)
  -- chapter_63_line_13: GL tag disintegration.
  have row_13 : (N v1) := by
    exact row_14.1.1
  -- chapter_63_line_12: GL tag implication.
  have row_12 : (gl_existence1 N v1 previous add) := by
    apply row_22
    exact row_13
    exact row_18
  -- chapter_63_line_11: GL tag expansion.
  have row_11 : (¬ (∀ (v6 : α), ((N v6) → (¬ (add v1 previous v6))))) := by
    simpa only [gl_existence1] using (row_12)
  have exists_row_11 : ∃ (v6 : α), ((N v6) ∧ (add v1 previous v6)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_11
      intro v6 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v6, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v6, witness_row_11⟩ := exists_row_11
  -- chapter_63_line_10: GL tag disintegration.
  have row_10 : (add v1 previous v6) := by
    exact witness_row_11.2
  -- chapter_63_line_9: GL tag task formulation.
  have row_9 : (succ v5 v1) := by
    exact assumption_9
  -- chapter_63_line_8: GL tag recursion.
  have row_8 : (∀ (w1 : α), ((add v1 previous w1) → (∀ (w2 : α), ((succ previous w2) → ((succ v5 v1) → (add w2 v5 w1)))))) := by
    exact assumption_8
  -- chapter_63_line_7: GL tag implication.
  have row_7 : (add v2 v5 v6) := by
    apply row_8
    exact row_10
    exact row_19
    exact row_9
  -- chapter_63_line_6: GL tag implication.
  have row_6 : (succ v6 v3) := by
    apply row_20
    exact row_18
    exact row_19
    exact row_10
    exact row_16
  -- chapter_63_line_5: GL tag theorem.
  have row_5 := peano_source_018 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_63_line_4: GL tag implication.
  have row_4 : (add v2 v1 v3) := by
    apply row_5
    exact row_7
    exact row_9
    exact row_6
  -- chapter_63_line_3: GL tag task formulation.
  have row_3 : (succ v2 v4) := by
    exact assumption_3
  -- chapter_63_line_2: GL tag theorem.
  have row_2 := peano_source_020 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_63_line_1: GL tag implication.
  have row_1 : (add v4 v5 v3) := by
    apply row_2
    exact row_4
    exact row_3
    exact row_9
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
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
  have inductionMember : N v2 := by
    have typingRule := peano_source_033_induction_typing N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact typingRule v1 v2 v3 premise_1
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v1 : α) (v3 : α), ((add v1 zero v3) → (∀ (v4 : α), ((succ zero v4) → (∀ (v5 : α), ((succ v5 v1) → (add v4 v5 v3))))))) := by
    intro v1
    intro v3
    intro base_premise_1
    intro v4
    intro base_premise_2
    intro v5
    intro base_premise_3
    have zeroRule := peano_source_033_check_zero N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact zeroRule v1 zero v3 v4 v5 base_premise_1 base_premise_3 base_premise_2 rfl
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v1 : α) (v3 : α), ((add v1 induction_n v3) → (∀ (v4 : α), ((succ induction_n v4) → (∀ (v5 : α), ((succ v5 v1) → (add v4 v5 v3))))))) → ∀ induction_m, succ induction_n induction_m → (∀ (v1 : α) (v3 : α), ((add v1 induction_m v3) → (∀ (v4 : α), ((succ induction_m v4) → (∀ (v5 : α), ((succ v5 v1) → (add v4 v5 v3))))))) := by
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
        (succ induction_n induction_m) := by
      exact induction_successor
    have step_induction_assumption_2 :
        (∀ (w1 : α), ((add v1 induction_n w1) → (∀ (w2 : α), ((succ induction_n w2) → ((succ v5 v1) → (add w2 v5 w1)))))) := by
      intro w1
      intro step_induction_assumption_2_premise_1
      intro w2
      intro step_induction_assumption_2_premise_2
      intro step_induction_assumption_2_premise_3
      apply induction_hypothesis
      all_goals assumption
    have stepRule := peano_source_033_check_induction_condition N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact stepRule induction_n v1 induction_m v3 v4 v5 step_induction_assumption_1 step_premise_1 step_premise_3 step_induction_assumption_2 step_premise_2
  have inductionProperty : (∀ (v1 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α), ((succ v2 v4) → (∀ (v5 : α), ((succ v5 v1) → (add v4 v5 v3))))))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v1 : α) (v3 : α), ((add v1 induction_value v3) → (∀ (v4 : α), ((succ induction_value v4) → (∀ (v5 : α), ((succ v5 v1) → (add v4 v5 v3))))))))
      v2
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v1 v3 premise_1 v4 premise_2 v5 premise_3

private theorem peano_source_034_induction_typing
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v2 : α)
    (v5 : α)
    (assumption_2 : (succ v2 v5))
    : (N v2) := by
  -- chapter_64_line_10: GL tag task formulation.
  have row_10 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_64_line_9: GL tag expansion.
  have row_9 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_10)
  -- chapter_64_line_8: GL tag disintegration.
  have row_8 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_9.1
  -- chapter_64_line_7: GL tag expansion.
  have row_7 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_8)
  -- chapter_64_line_6: GL tag disintegration.
  have row_6 : (gl_fXY succ N N) := by
    exact row_7.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_64_line_5: GL tag expansion.
  have row_5 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_6)
  -- chapter_64_line_4: GL tag disintegration.
  have row_4 : (gl_implication0 succ N) := by
    exact row_5.1.1.1
  -- chapter_64_line_3: GL tag expansion.
  have row_3 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_4)
  -- chapter_64_line_2: GL tag task formulation.
  have row_2 : (succ v2 v5) := by
    exact assumption_2
  -- chapter_64_line_1: GL tag implication.
  have row_1 : (N v2) := by
    apply row_3
    exact row_2
  exact row_1

private theorem peano_source_034_check_zero
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (v4 : α)
    (v5 : α)
    (assumption_28 : (succ v2 v5))
    (assumption_22 : (succ v4 v1))
    (assumption_13 : (v2 = zero))
    (assumption_12 : (add v1 v2 v3))
    : (add v5 v4 v3) := by
  -- chapter_65_line_43: GL tag task formulation.
  have row_43 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_65_line_42: GL tag expansion.
  have row_42 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_43)
  -- chapter_65_line_41: GL tag disintegration.
  have row_41 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_42.1
  -- chapter_65_line_40: GL tag expansion.
  have row_40 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_41)
  -- chapter_65_line_39: GL tag disintegration.
  have row_39 : (gl_fXYZ add N N N) := by
    exact row_40.1.1.1.1.1.1.1.1.1.2
  -- chapter_65_line_38: GL tag expansion.
  have row_38 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_39)
  -- chapter_65_line_37: GL tag disintegration.
  have row_37 : (gl_implication13 add N) := by
    exact row_38.1.1.2
  -- chapter_65_line_36: GL tag expansion.
  have row_36 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_37)
  -- chapter_65_line_35: GL tag disintegration.
  have row_35 : (gl_implication12 add N) := by
    exact row_38.1.1.1.2
  -- chapter_65_line_34: GL tag expansion.
  have row_34 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_35)
  -- chapter_65_line_33: GL tag disintegration.
  have row_33 : (gl_implication11 add N) := by
    exact row_38.1.1.1.1
  -- chapter_65_line_32: GL tag expansion.
  have row_32 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_33)
  -- chapter_65_line_31: GL tag disintegration.
  have row_31 : (gl_implication18 N N N add) := by
    exact row_38.1.2
  -- chapter_65_line_30: GL tag expansion.
  have row_30 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication18] using (row_31)
  -- chapter_65_line_29: GL tag implication grouping.
  have row_29 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_32
    have auxiliary_source_2 := row_34
    have auxiliary_source_3 := row_36
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_65_line_28: GL tag task formulation.
  have row_28 : (succ v2 v5) := by
    exact assumption_28
  -- chapter_65_line_27: GL tag theorem.
  have row_27 := peano_source_010 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_65_line_26: GL tag implication.
  have row_26 : (add one v2 v5) := by
    apply row_27
    exact row_28
  -- chapter_65_line_25: GL tag implication.
  have row_25 : (gl_and21 one N v2 N v5 N) := by
    apply row_29
    exact row_26
  -- chapter_65_line_24: GL tag expansion.
  have row_24 : (((N one) ∧ (N v2)) ∧ (N v5)) := by
    simpa only [gl_and21] using (row_25)
  -- chapter_65_line_23: GL tag disintegration.
  have row_23 : (N v2) := by
    exact row_24.1.2
  -- chapter_65_line_22: GL tag task formulation.
  have row_22 : (succ v4 v1) := by
    exact assumption_22
  -- chapter_65_line_21: GL tag theorem.
  have row_21 := peano_source_027 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_65_line_20: GL tag implication.
  have row_20 : (add v4 one v1) := by
    apply row_21
    exact row_22
  -- chapter_65_line_19: GL tag implication.
  have row_19 : (gl_and21 v4 N one N v1 N) := by
    apply row_29
    exact row_20
  -- chapter_65_line_18: GL tag expansion.
  have row_18 : (((N v4) ∧ (N one)) ∧ (N v1)) := by
    simpa only [gl_and21] using (row_19)
  -- chapter_65_line_17: GL tag disintegration.
  have row_17 : (N v1) := by
    exact row_18.2
  -- chapter_65_line_16: GL tag implication.
  have row_16 : (gl_existence1 N v2 v1 add) := by
    apply row_30
    exact row_23
    exact row_17
  -- chapter_65_line_15: GL tag expansion.
  have row_15 : (¬ (∀ (v6 : α), ((N v6) → (¬ (add v2 v1 v6))))) := by
    simpa only [gl_existence1] using (row_16)
  have exists_row_15 : ∃ (v6 : α), ((N v6) ∧ (add v2 v1 v6)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_15
      intro v6 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v6, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v6, witness_row_15⟩ := exists_row_15
  -- chapter_65_line_14: GL tag disintegration.
  have row_14 : (add v2 v1 v6) := by
    exact witness_row_15.2
  -- chapter_65_line_13: GL tag recursion.
  have row_13 : (v2 = zero) := by
    exact assumption_13
  -- chapter_65_line_12: GL tag task formulation.
  have row_12 : (add v1 v2 v3) := by
    exact assumption_12
  -- chapter_65_line_11: GL tag equality1.
  have row_11 : (add v1 zero v3) := by
    have equality_source := row_12
    have equality_step_1 := row_13
    cases equality_step_1
    exact equality_source
  -- chapter_65_line_10: GL tag theorem.
  have row_10 := peano_source_011 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_65_line_9: GL tag implication.
  have row_9 : (v1 = v3) := by
    apply row_10
    exact row_11
    exact row_17
  -- chapter_65_line_8: GL tag equality1.
  have row_8 : (add zero v1 v6) := by
    have equality_source := row_14
    have equality_step_1 := row_13
    cases equality_step_1
    exact equality_source
  -- chapter_65_line_7: GL tag theorem.
  have row_7 := peano_source_032 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_65_line_6: GL tag implication.
  have row_6 : (v1 = v6) := by
    apply row_7
    exact row_17
    exact row_8
  -- chapter_65_line_5: GL tag symmetry of equality.
  have row_5 : (v6 = v1) := by
    exact Eq.symm row_6
  -- chapter_65_line_4: GL tag equality2.
  have row_4 : (v6 = v3) := by
    exact Eq.trans row_5 row_9
  -- chapter_65_line_3: GL tag theorem.
  have row_3 := peano_source_020 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_65_line_2: GL tag implication.
  have row_2 : (add v5 v4 v6) := by
    apply row_3
    exact row_14
    exact row_28
    exact row_22
  -- chapter_65_line_1: GL tag equality1.
  have row_1 : (add v5 v4 v3) := by
    have equality_source := row_2
    have equality_step_1 := row_4
    cases equality_step_1
    exact equality_source
  exact row_1

private theorem peano_source_034_check_induction_condition
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (v4 : α)
    (v5 : α)
    (assumption_19 : (succ previous v2))
    (assumption_16 : (add v1 v2 v3))
    (assumption_9 : (succ v4 v1))
    (assumption_8 : (∀ (w1 : α), ((add v1 previous w1) → ((succ v4 v1) → (∀ (w2 : α), ((succ previous w2) → (add w2 v4 w1)))))))
    (assumption_3 : (succ v2 v5))
    : (add v5 v4 v3) := by
  -- chapter_66_line_39: GL tag task formulation.
  have row_39 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_66_line_38: GL tag expansion.
  have row_38 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_39)
  -- chapter_66_line_37: GL tag disintegration.
  have row_37 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_38.1
  -- chapter_66_line_36: GL tag expansion.
  have row_36 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_37)
  -- chapter_66_line_35: GL tag disintegration.
  have row_35 : (gl_fXY succ N N) := by
    exact row_36.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_66_line_34: GL tag expansion.
  have row_34 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_35)
  -- chapter_66_line_33: GL tag disintegration.
  have row_33 : (gl_implication0 succ N) := by
    exact row_34.1.1.1
  -- chapter_66_line_32: GL tag expansion.
  have row_32 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_33)
  -- chapter_66_line_31: GL tag disintegration.
  have row_31 : (gl_fXYZ add N N N) := by
    exact row_36.1.1.1.1.1.1.1.1.1.2
  -- chapter_66_line_30: GL tag expansion.
  have row_30 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_31)
  -- chapter_66_line_29: GL tag disintegration.
  have row_29 : (gl_implication13 add N) := by
    exact row_30.1.1.2
  -- chapter_66_line_28: GL tag expansion.
  have row_28 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_29)
  -- chapter_66_line_27: GL tag disintegration.
  have row_27 : (gl_implication12 add N) := by
    exact row_30.1.1.1.2
  -- chapter_66_line_26: GL tag expansion.
  have row_26 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_27)
  -- chapter_66_line_25: GL tag disintegration.
  have row_25 : (gl_implication11 add N) := by
    exact row_30.1.1.1.1
  -- chapter_66_line_24: GL tag expansion.
  have row_24 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_25)
  -- chapter_66_line_23: GL tag disintegration.
  have row_23 : (gl_implication18 N N N add) := by
    exact row_30.1.2
  -- chapter_66_line_22: GL tag expansion.
  have row_22 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication18] using (row_23)
  -- chapter_66_line_21: GL tag disintegration.
  have row_21 : (gl_implication23 N succ add) := by
    exact row_36.1.1.1.1.1.1.2
  -- chapter_66_line_20: GL tag expansion.
  have row_20 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication23] using (row_21)
  -- chapter_66_line_19: GL tag recursion.
  have row_19 : (succ previous v2) := by
    exact assumption_19
  -- chapter_66_line_18: GL tag implication.
  have row_18 : (N previous) := by
    apply row_32
    exact row_19
  -- chapter_66_line_17: GL tag implication grouping.
  have row_17 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_24
    have auxiliary_source_2 := row_26
    have auxiliary_source_3 := row_28
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_66_line_16: GL tag task formulation.
  have row_16 : (add v1 v2 v3) := by
    exact assumption_16
  -- chapter_66_line_15: GL tag implication.
  have row_15 : (gl_and21 v1 N v2 N v3 N) := by
    apply row_17
    exact row_16
  -- chapter_66_line_14: GL tag expansion.
  have row_14 : (((N v1) ∧ (N v2)) ∧ (N v3)) := by
    simpa only [gl_and21] using (row_15)
  -- chapter_66_line_13: GL tag disintegration.
  have row_13 : (N v1) := by
    exact row_14.1.1
  -- chapter_66_line_12: GL tag implication.
  have row_12 : (gl_existence1 N v1 previous add) := by
    apply row_22
    exact row_13
    exact row_18
  -- chapter_66_line_11: GL tag expansion.
  have row_11 : (¬ (∀ (v6 : α), ((N v6) → (¬ (add v1 previous v6))))) := by
    simpa only [gl_existence1] using (row_12)
  have exists_row_11 : ∃ (v6 : α), ((N v6) ∧ (add v1 previous v6)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_11
      intro v6 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v6, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v6, witness_row_11⟩ := exists_row_11
  -- chapter_66_line_10: GL tag disintegration.
  have row_10 : (add v1 previous v6) := by
    exact witness_row_11.2
  -- chapter_66_line_9: GL tag task formulation.
  have row_9 : (succ v4 v1) := by
    exact assumption_9
  -- chapter_66_line_8: GL tag recursion.
  have row_8 : (∀ (w1 : α), ((add v1 previous w1) → ((succ v4 v1) → (∀ (w2 : α), ((succ previous w2) → (add w2 v4 w1)))))) := by
    exact assumption_8
  -- chapter_66_line_7: GL tag implication.
  have row_7 : (add v2 v4 v6) := by
    apply row_8
    exact row_10
    exact row_9
    exact row_19
  -- chapter_66_line_6: GL tag implication.
  have row_6 : (succ v6 v3) := by
    apply row_20
    exact row_18
    exact row_19
    exact row_10
    exact row_16
  -- chapter_66_line_5: GL tag theorem.
  have row_5 := peano_source_018 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_66_line_4: GL tag implication.
  have row_4 : (add v2 v1 v3) := by
    apply row_5
    exact row_7
    exact row_9
    exact row_6
  -- chapter_66_line_3: GL tag task formulation.
  have row_3 : (succ v2 v5) := by
    exact assumption_3
  -- chapter_66_line_2: GL tag theorem.
  have row_2 := peano_source_020 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_66_line_1: GL tag implication.
  have row_1 : (add v5 v4 v3) := by
    apply row_2
    exact row_4
    exact row_3
    exact row_9
  exact row_1

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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α), ((succ v4 v1) → (∀ (v5 : α), ((succ v2 v5) → (add v5 v4 v3))))))) := by
  intro v1
  intro v2
  intro v3
  intro premise_1
  intro v4
  intro premise_2
  intro v5
  intro premise_3
  have inductionMember : N v2 := by
    have typingRule := peano_source_034_induction_typing N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact typingRule v2 v5 premise_3
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v1 : α) (v3 : α), ((add v1 zero v3) → (∀ (v4 : α), ((succ v4 v1) → (∀ (v5 : α), ((succ zero v5) → (add v5 v4 v3))))))) := by
    intro v1
    intro v3
    intro base_premise_1
    intro v4
    intro base_premise_2
    intro v5
    intro base_premise_3
    have zeroRule := peano_source_034_check_zero N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact zeroRule v1 zero v3 v4 v5 base_premise_3 base_premise_2 rfl base_premise_1
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v1 : α) (v3 : α), ((add v1 induction_n v3) → (∀ (v4 : α), ((succ v4 v1) → (∀ (v5 : α), ((succ induction_n v5) → (add v5 v4 v3))))))) → ∀ induction_m, succ induction_n induction_m → (∀ (v1 : α) (v3 : α), ((add v1 induction_m v3) → (∀ (v4 : α), ((succ v4 v1) → (∀ (v5 : α), ((succ induction_m v5) → (add v5 v4 v3))))))) := by
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
        (succ induction_n induction_m) := by
      exact induction_successor
    have step_induction_assumption_2 :
        (∀ (w1 : α), ((add v1 induction_n w1) → ((succ v4 v1) → (∀ (w2 : α), ((succ induction_n w2) → (add w2 v4 w1)))))) := by
      intro w1
      intro step_induction_assumption_2_premise_1
      intro step_induction_assumption_2_premise_2
      intro w2
      intro step_induction_assumption_2_premise_3
      apply induction_hypothesis
      all_goals assumption
    have stepRule := peano_source_034_check_induction_condition N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact stepRule induction_n v1 induction_m v3 v4 v5 step_induction_assumption_1 step_premise_1 step_premise_2 step_induction_assumption_2 step_premise_3
  have inductionProperty : (∀ (v1 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α), ((succ v4 v1) → (∀ (v5 : α), ((succ v2 v5) → (add v5 v4 v3))))))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v1 : α) (v3 : α), ((add v1 induction_value v3) → (∀ (v4 : α), ((succ v4 v1) → (∀ (v5 : α), ((succ induction_value v5) → (add v5 v4 v3))))))))
      v2
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v1 v3 premise_1 v4 premise_2 v5 premise_3

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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (assumption_2 : (add v1 v2 v3))
    : (N v2) := by
  -- chapter_68_line_10: GL tag task formulation.
  have row_10 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_68_line_9: GL tag expansion.
  have row_9 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_10)
  -- chapter_68_line_8: GL tag disintegration.
  have row_8 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_9.1
  -- chapter_68_line_7: GL tag expansion.
  have row_7 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_8)
  -- chapter_68_line_6: GL tag disintegration.
  have row_6 : (gl_fXYZ add N N N) := by
    exact row_7.1.1.1.1.1.1.1.1.1.2
  -- chapter_68_line_5: GL tag expansion.
  have row_5 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_6)
  -- chapter_68_line_4: GL tag disintegration.
  have row_4 : (gl_implication12 add N) := by
    exact row_5.1.1.1.2
  -- chapter_68_line_3: GL tag expansion.
  have row_3 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_4)
  -- chapter_68_line_2: GL tag task formulation.
  have row_2 : (add v1 v2 v3) := by
    exact assumption_2
  -- chapter_68_line_1: GL tag implication.
  have row_1 : (N v2) := by
    apply row_3
    exact row_2
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (assumption_8 : (v2 = zero))
    (assumption_7 : (add v1 v2 v3))
    : (add v2 v1 v3) := by
  -- chapter_69_line_8: GL tag recursion.
  have row_8 : (v2 = zero) := by
    exact assumption_8
  -- chapter_69_line_7: GL tag task formulation.
  have row_7 : (add v1 v2 v3) := by
    exact assumption_7
  -- chapter_69_line_6: GL tag equality1.
  have row_6 : (add v1 zero v3) := by
    have equality_source := row_7
    have equality_step_1 := row_8
    cases equality_step_1
    exact equality_source
  -- chapter_69_line_5: GL tag task formulation.
  have row_5 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_69_line_4: GL tag theorem.
  have row_4 := peano_source_035 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_69_line_3: GL tag implication.
  have row_3 : (add zero v1 v3) := by
    apply row_4
    exact row_6
  -- chapter_69_line_2: GL tag symmetry of equality.
  have row_2 : (zero = v2) := by
    exact Eq.symm row_8
  -- chapter_69_line_1: GL tag equality1.
  have row_1 : (add v2 v1 v3) := by
    have equality_source := row_3
    have equality_step_1 := row_2
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (assumption_22 : (add v1 v2 v3))
    (assumption_16 : (succ previous v2))
    (assumption_6 : (∀ (w1 : α), ((add v1 previous w1) → (add previous v1 w1))))
    : (add v2 v1 v3) := by
  -- chapter_70_line_47: GL tag task formulation.
  have row_47 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_70_line_46: GL tag expansion.
  have row_46 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_47)
  -- chapter_70_line_45: GL tag disintegration.
  have row_45 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_46.1
  -- chapter_70_line_44: GL tag expansion.
  have row_44 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_45)
  -- chapter_70_line_43: GL tag disintegration.
  have row_43 : (gl_fXYZ add N N N) := by
    exact row_44.1.1.1.1.1.1.1.1.1.2
  -- chapter_70_line_42: GL tag expansion.
  have row_42 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_43)
  -- chapter_70_line_41: GL tag disintegration.
  have row_41 : (gl_implication13 add N) := by
    exact row_42.1.1.2
  -- chapter_70_line_40: GL tag expansion.
  have row_40 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_41)
  -- chapter_70_line_39: GL tag disintegration.
  have row_39 : (gl_implication12 add N) := by
    exact row_42.1.1.1.2
  -- chapter_70_line_38: GL tag expansion.
  have row_38 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_39)
  -- chapter_70_line_37: GL tag disintegration.
  have row_37 : (gl_implication11 add N) := by
    exact row_42.1.1.1.1
  -- chapter_70_line_36: GL tag expansion.
  have row_36 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_37)
  -- chapter_70_line_35: GL tag disintegration.
  have row_35 : (gl_implication18 N N N add) := by
    exact row_42.1.2
  -- chapter_70_line_34: GL tag expansion.
  have row_34 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication18] using (row_35)
  -- chapter_70_line_33: GL tag disintegration.
  have row_33 : (gl_implication9 N zero succ) := by
    exact row_44.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_70_line_32: GL tag expansion.
  have row_32 : (∀ (w1 : α), ((N w1) → (¬ (succ w1 zero)))) := by
    simpa only [gl_implication9] using (row_33)
  -- chapter_70_line_31: GL tag disintegration.
  have row_31 : (gl_fXY succ N N) := by
    exact row_44.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_70_line_30: GL tag expansion.
  have row_30 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_31)
  -- chapter_70_line_29: GL tag disintegration.
  have row_29 : (gl_implication6 N N succ) := by
    exact row_30.1.2
  -- chapter_70_line_28: GL tag expansion.
  have row_28 : (∀ (w1 : α), ((N w1) → (gl_existence0 N w1 succ))) := by
    simpa only [gl_implication6] using (row_29)
  -- chapter_70_line_27: GL tag disintegration.
  have row_27 : (gl_implication23 N succ add) := by
    exact row_44.1.1.1.1.1.1.2
  -- chapter_70_line_26: GL tag expansion.
  have row_26 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication23] using (row_27)
  -- chapter_70_line_25: GL tag disintegration.
  have row_25 : (gl_implication0 succ N) := by
    exact row_30.1.1.1
  -- chapter_70_line_24: GL tag expansion.
  have row_24 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_25)
  -- chapter_70_line_23: GL tag implication grouping.
  have row_23 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_36
    have auxiliary_source_2 := row_38
    have auxiliary_source_3 := row_40
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_70_line_22: GL tag task formulation.
  have row_22 : (add v1 v2 v3) := by
    exact assumption_22
  -- chapter_70_line_21: GL tag implication.
  have row_21 : (gl_and21 v1 N v2 N v3 N) := by
    apply row_23
    exact row_22
  -- chapter_70_line_20: GL tag expansion.
  have row_20 : (((N v1) ∧ (N v2)) ∧ (N v3)) := by
    simpa only [gl_and21] using (row_21)
  -- chapter_70_line_19: GL tag disintegration.
  have row_19 : (N v1) := by
    exact row_20.1.1
  -- chapter_70_line_18: GL tag implication grouping.
  have row_18 : (∀ (w1 : α), ((N w1) → (gl_and3 N w1 succ zero))) := by
    have auxiliary_source_1 := row_28
    have auxiliary_source_2 := row_32
    intro auxiliary_argument_1
    intro auxiliary_premise
    exact ⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_premise)⟩
  -- chapter_70_line_17: GL tag implication.
  have row_17 : (gl_and3 N v1 succ zero) := by
    apply row_18
    exact row_19
  -- chapter_70_line_16: GL tag recursion.
  have row_16 : (succ previous v2) := by
    exact assumption_16
  -- chapter_70_line_15: GL tag implication.
  have row_15 : (N previous) := by
    apply row_24
    exact row_16
  -- chapter_70_line_14: GL tag implication.
  have row_14 : (gl_existence1 N v1 previous add) := by
    apply row_34
    exact row_19
    exact row_15
  -- chapter_70_line_13: GL tag expansion.
  have row_13 : (¬ (∀ (v8 : α), ((N v8) → (¬ (add v1 previous v8))))) := by
    simpa only [gl_existence1] using (row_14)
  have exists_row_13 : ∃ (v8 : α), ((N v8) ∧ (add v1 previous v8)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_13
      intro v8 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v8, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v8, witness_row_13⟩ := exists_row_13
  -- chapter_70_line_12: GL tag disintegration.
  have row_12 : (add v1 previous v8) := by
    exact witness_row_13.2
  -- chapter_70_line_11: GL tag implication.
  have row_11 : (succ v8 v3) := by
    apply row_26
    exact row_15
    exact row_16
    exact row_12
    exact row_22
  -- chapter_70_line_10: GL tag expansion.
  have row_10 : ((gl_existence0 N v1 succ) ∧ (¬ (succ v1 zero))) := by
    simpa only [gl_and3] using (row_17)
  -- chapter_70_line_9: GL tag disintegration.
  have row_9 : (gl_existence0 N v1 succ) := by
    exact row_10.1
  -- chapter_70_line_8: GL tag expansion.
  have row_8 : (¬ (∀ (v4 : α), ((N v4) → (¬ (succ v1 v4))))) := by
    simpa only [gl_existence0] using (row_9)
  have exists_row_8 : ∃ (v4 : α), ((N v4) ∧ (succ v1 v4)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_8
      intro v4 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v4, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v4, witness_row_8⟩ := exists_row_8
  -- chapter_70_line_7: GL tag disintegration.
  have row_7 : (succ v1 v4) := by
    exact witness_row_8.2
  -- chapter_70_line_6: GL tag recursion.
  have row_6 : (∀ (w1 : α), ((add v1 previous w1) → (add previous v1 w1))) := by
    exact assumption_6
  -- chapter_70_line_5: GL tag implication.
  have row_5 : (add previous v1 v8) := by
    apply row_6
    exact row_12
  -- chapter_70_line_4: GL tag theorem.
  have row_4 := peano_source_018 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_70_line_3: GL tag implication.
  have row_3 : (add previous v4 v3) := by
    apply row_4
    exact row_5
    exact row_7
    exact row_11
  -- chapter_70_line_2: GL tag theorem.
  have row_2 := peano_source_020 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_70_line_1: GL tag implication.
  have row_1 : (add v2 v1 v3) := by
    apply row_2
    exact row_3
    exact row_16
    exact row_7
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (add v2 v1 v3))) := by
  intro v1
  intro v2
  intro v3
  intro premise_1
  have inductionMember : N v2 := by
    have typingRule := peano_source_036_induction_typing N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact typingRule v1 v2 v3 premise_1
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v1 : α) (v3 : α), ((add v1 zero v3) → (add zero v1 v3))) := by
    intro v1
    intro v3
    intro base_premise_1
    have zeroRule := peano_source_036_check_zero N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact zeroRule v1 zero v3 rfl base_premise_1
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v1 : α) (v3 : α), ((add v1 induction_n v3) → (add induction_n v1 v3))) → ∀ induction_m, succ induction_n induction_m → (∀ (v1 : α) (v3 : α), ((add v1 induction_m v3) → (add induction_m v1 v3))) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro v1
    intro v3
    intro step_premise_1
    have step_induction_assumption_1 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have step_induction_assumption_2 :
        (∀ (w1 : α), ((add v1 induction_n w1) → (add induction_n v1 w1))) := by
      intro w1
      intro step_induction_assumption_2_premise_1
      apply induction_hypothesis
      all_goals assumption
    have stepRule := peano_source_036_check_induction_condition N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact stepRule induction_n v1 induction_m v3 step_premise_1 step_induction_assumption_1 step_induction_assumption_2
  have inductionProperty : (∀ (v1 : α) (v3 : α), ((add v1 v2 v3) → (add v2 v1 v3))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v1 : α) (v3 : α), ((add v1 induction_value v3) → (add induction_value v1 v3))))
      v2
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v1 v3 premise_1

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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → ((v2 = v3) → (zero = v1)))) := by
  intro v1
  intro v2
  intro v3
  intro premise_1
  intro premise_2
  -- chapter_72_line_22: GL tag task formulation.
  have row_22 : (add v1 v2 v3) := by
    exact premise_1
  -- chapter_72_line_21: GL tag task formulation.
  have row_21 : (v2 = v3) := by
    exact premise_2
  -- chapter_72_line_20: GL tag symmetry of equality.
  have row_20 : (v3 = v2) := by
    exact Eq.symm row_21
  -- chapter_72_line_19: GL tag equality1.
  have row_19 : (add v1 v3 v2) := by
    have equality_source := row_22
    have equality_step_1 := row_21
    cases equality_step_1
    have equality_step_2 := row_20
    cases equality_step_2
    exact equality_source
  -- chapter_72_line_18: GL tag task formulation.
  have row_18 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_72_line_17: GL tag expansion.
  have row_17 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_18)
  -- chapter_72_line_16: GL tag disintegration.
  have row_16 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_17.1
  -- chapter_72_line_15: GL tag expansion.
  have row_15 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_16)
  -- chapter_72_line_14: GL tag disintegration.
  have row_14 : (gl_fXYZ add N N N) := by
    exact row_15.1.1.1.1.1.1.1.1.1.2
  -- chapter_72_line_13: GL tag expansion.
  have row_13 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_14)
  -- chapter_72_line_12: GL tag disintegration.
  have row_12 : (gl_implication12 add N) := by
    exact row_13.1.1.1.2
  -- chapter_72_line_11: GL tag expansion.
  have row_11 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_12)
  -- chapter_72_line_10: GL tag disintegration.
  have row_10 : (gl_implication22 N zero add) := by
    exact row_15.1.1.1.1.1.1.1.2
  -- chapter_72_line_9: GL tag expansion.
  have row_9 : (∀ (w1 : α) (w2 : α), ((w1 = w2) → ((N w1) → ((N w2) → (add w1 zero w2))))) := by
    simpa only [gl_implication22] using (row_10)
  -- chapter_72_line_8: GL tag implication.
  have row_8 : (N v3) := by
    apply row_11
    exact row_19
  -- chapter_72_line_7: GL tag implication.
  have row_7 : (N v2) := by
    apply row_11
    exact row_22
  -- chapter_72_line_6: GL tag implication.
  have row_6 : (add v2 zero v3) := by
    apply row_9
    exact row_21
    exact row_7
    exact row_8
  -- chapter_72_line_5: GL tag theorem.
  have row_5 := peano_source_035 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_72_line_4: GL tag implication.
  have row_4 : (add zero v2 v3) := by
    apply row_5
    exact row_6
  -- chapter_72_line_3: GL tag anchor handling.
  have row_3 : (gl_AnchorPeano N zero succ add mul one) := by
    exact row_18
  -- chapter_72_line_2: GL tag theorem.
  have row_2 := peano_source_031 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_72_line_1: GL tag implication.
  have row_1 : (zero = v1) := by
    apply row_2
    exact row_4
    exact row_22
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
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
  -- chapter_102_line_45: GL tag task formulation.
  have row_45 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_102_line_44: GL tag expansion.
  have row_44 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_45)
  -- chapter_102_line_43: GL tag disintegration.
  have row_43 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_44.1
  -- chapter_102_line_42: GL tag expansion.
  have row_42 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_43)
  -- chapter_102_line_41: GL tag disintegration.
  have row_41 : (gl_fXYZ add N N N) := by
    exact row_42.1.1.1.1.1.1.1.1.1.2
  -- chapter_102_line_40: GL tag expansion.
  have row_40 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_41)
  -- chapter_102_line_39: GL tag disintegration.
  have row_39 : (gl_implication19 N N add) := by
    exact row_40.2
  -- chapter_102_line_38: GL tag expansion.
  have row_38 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (∀ (w3 : α), ((add w1 w2 w3) → (∀ (w4 : α), ((add w1 w2 w4) → (w3 = w4))))))))) := by
    simpa only [gl_implication19] using (row_39)
  -- chapter_102_line_37: GL tag disintegration.
  have row_37 : (gl_implication13 add N) := by
    exact row_40.1.1.2
  -- chapter_102_line_36: GL tag expansion.
  have row_36 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_37)
  -- chapter_102_line_35: GL tag disintegration.
  have row_35 : (gl_implication12 add N) := by
    exact row_40.1.1.1.2
  -- chapter_102_line_34: GL tag expansion.
  have row_34 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_35)
  -- chapter_102_line_33: GL tag disintegration.
  have row_33 : (gl_implication11 add N) := by
    exact row_40.1.1.1.1
  -- chapter_102_line_32: GL tag expansion.
  have row_32 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_33)
  -- chapter_102_line_31: GL tag disintegration.
  have row_31 : (gl_fXYZ mul N N N) := by
    exact row_42.1.1.1.1.2
  -- chapter_102_line_30: GL tag expansion.
  have row_30 : (((((gl_implication11 mul N) ∧ (gl_implication12 mul N)) ∧ (gl_implication13 mul N)) ∧ (gl_implication18 N N N mul)) ∧ (gl_implication19 N N mul)) := by
    simpa only [gl_fXYZ] using (row_31)
  -- chapter_102_line_29: GL tag disintegration.
  have row_29 : (gl_implication13 mul N) := by
    exact row_30.1.1.2
  -- chapter_102_line_28: GL tag expansion.
  have row_28 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_29)
  -- chapter_102_line_27: GL tag disintegration.
  have row_27 : (gl_implication11 mul N) := by
    exact row_30.1.1.1.1
  -- chapter_102_line_26: GL tag expansion.
  have row_26 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_27)
  -- chapter_102_line_25: GL tag disintegration.
  have row_25 : (gl_implication12 mul N) := by
    exact row_30.1.1.1.2
  -- chapter_102_line_24: GL tag expansion.
  have row_24 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_25)
  -- chapter_102_line_23: GL tag disintegration.
  have row_23 : (gl_implication18 N N N mul) := by
    exact row_30.1.2
  -- chapter_102_line_22: GL tag expansion.
  have row_22 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication18] using (row_23)
  -- chapter_102_line_21: GL tag implication grouping.
  have row_21 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_32
    have auxiliary_source_2 := row_34
    have auxiliary_source_3 := row_36
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_102_line_20: GL tag task formulation.
  have row_20 : (add v4 v5 v6) := by
    exact premise_2
  -- chapter_102_line_19: GL tag implication.
  have row_19 : (gl_and21 v4 N v5 N v6 N) := by
    apply row_21
    exact row_20
  -- chapter_102_line_18: GL tag expansion.
  have row_18 : (((N v4) ∧ (N v5)) ∧ (N v6)) := by
    simpa only [gl_and21] using (row_19)
  -- chapter_102_line_17: GL tag disintegration.
  have row_17 : (N v6) := by
    exact row_18.2
  -- chapter_102_line_16: GL tag implication grouping.
  have row_16 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_26
    have auxiliary_source_2 := row_24
    have auxiliary_source_3 := row_28
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_102_line_15: GL tag task formulation.
  have row_15 : (mul v7 v4 v1) := by
    exact premise_3
  -- chapter_102_line_14: GL tag implication.
  have row_14 : (gl_and21 v7 N v4 N v1 N) := by
    apply row_16
    exact row_15
  -- chapter_102_line_13: GL tag task formulation.
  have row_13 : (mul v7 v5 v2) := by
    exact premise_4
  -- chapter_102_line_12: GL tag implication.
  have row_12 : (N v7) := by
    apply row_26
    exact row_13
  -- chapter_102_line_11: GL tag implication.
  have row_11 : (gl_existence1 N v7 v6 mul) := by
    apply row_22
    exact row_12
    exact row_17
  -- chapter_102_line_10: GL tag expansion.
  have row_10 : (¬ (∀ (v8 : α), ((N v8) → (¬ (mul v7 v6 v8))))) := by
    simpa only [gl_existence1] using (row_11)
  have exists_row_10 : ∃ (v8 : α), ((N v8) ∧ (mul v7 v6 v8)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_10
      intro v8 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v8, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v8, witness_row_10⟩ := exists_row_10
  -- chapter_102_line_9: GL tag disintegration.
  have row_9 : (mul v7 v6 v8) := by
    exact witness_row_10.2
  -- chapter_102_line_8: GL tag theorem.
  have row_8 := peano_source_047 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_102_line_7: GL tag implication.
  have row_7 : (add v1 v2 v8) := by
    apply row_8
    exact row_15
    exact row_13
    exact row_9
    exact row_20
  -- chapter_102_line_6: GL tag expansion.
  have row_6 : (((N v7) ∧ (N v4)) ∧ (N v1)) := by
    simpa only [gl_and21] using (row_14)
  -- chapter_102_line_5: GL tag disintegration.
  have row_5 : (N v1) := by
    exact row_6.2
  -- chapter_102_line_4: GL tag implication.
  have row_4 : (N v2) := by
    apply row_28
    exact row_13
  -- chapter_102_line_3: GL tag task formulation.
  have row_3 : (add v1 v2 v3) := by
    exact premise_1
  -- chapter_102_line_2: GL tag implication.
  have row_2 : (v8 = v3) := by
    apply row_38
    exact row_5
    exact row_4
    exact row_7
    exact row_3
  -- chapter_102_line_1: GL tag equality1.
  have row_1 : (mul v7 v6 v3) := by
    have equality_source := row_9
    have equality_step_1 := row_2
    cases equality_step_1
    exact equality_source
  exact row_1

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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((mul one v1 v2) → (mul v1 one v2))) := by
  intro v1
  intro v2
  intro premise_1
  -- chapter_129_line_44: GL tag task formulation.
  have row_44 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_129_line_43: GL tag expansion.
  have row_43 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_44)
  -- chapter_129_line_42: GL tag disintegration.
  have row_42 : (succ zero one) := by
    exact row_43.2
  -- chapter_129_line_41: GL tag disintegration.
  have row_41 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_43.1
  -- chapter_129_line_40: GL tag expansion.
  have row_40 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_41)
  -- chapter_129_line_39: GL tag disintegration.
  have row_39 : (gl_fXY succ N N) := by
    exact row_40.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_129_line_38: GL tag expansion.
  have row_38 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_39)
  -- chapter_129_line_37: GL tag disintegration.
  have row_37 : (gl_implication1 succ N) := by
    exact row_38.1.1.2
  -- chapter_129_line_36: GL tag expansion.
  have row_36 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using (row_37)
  -- chapter_129_line_35: GL tag disintegration.
  have row_35 : (gl_implication0 succ N) := by
    exact row_38.1.1.1
  -- chapter_129_line_34: GL tag expansion.
  have row_34 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_35)
  -- chapter_129_line_33: GL tag disintegration.
  have row_33 : (gl_fXYZ mul N N N) := by
    exact row_40.1.1.1.1.2
  -- chapter_129_line_32: GL tag expansion.
  have row_32 : (((((gl_implication11 mul N) ∧ (gl_implication12 mul N)) ∧ (gl_implication13 mul N)) ∧ (gl_implication18 N N N mul)) ∧ (gl_implication19 N N mul)) := by
    simpa only [gl_fXYZ] using (row_33)
  -- chapter_129_line_31: GL tag disintegration.
  have row_31 : (gl_implication12 mul N) := by
    exact row_32.1.1.1.2
  -- chapter_129_line_30: GL tag expansion.
  have row_30 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_31)
  -- chapter_129_line_29: GL tag disintegration.
  have row_29 : (gl_implication18 N N N mul) := by
    exact row_32.1.2
  -- chapter_129_line_28: GL tag expansion.
  have row_28 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication18] using (row_29)
  -- chapter_129_line_27: GL tag disintegration.
  have row_27 : (N zero) := by
    exact row_40.1.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_129_line_26: GL tag disintegration.
  have row_26 : (gl_implication25 N zero mul) := by
    exact row_40.1.1.1.2
  -- chapter_129_line_25: GL tag expansion.
  have row_25 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((mul w1 zero w2) → (w2 = zero))))) := by
    simpa only [gl_implication25] using (row_26)
  -- chapter_129_line_24: GL tag disintegration.
  have row_24 : (gl_implication28 N succ mul add) := by
    exact row_40.2
  -- chapter_129_line_23: GL tag expansion.
  have row_23 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((mul w3 w1 w4) → (∀ (w5 : α), ((mul w3 w2 w5) → (add w4 w3 w5))))))))) := by
    simpa only [gl_implication28] using (row_24)
  -- chapter_129_line_22: GL tag implication grouping.
  have row_22 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (gl_and20 w1 N w2 N))) := by
    have auxiliary_source_1 := row_34
    have auxiliary_source_2 := row_36
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_premise
    exact ⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_premise)⟩
  -- chapter_129_line_21: GL tag implication.
  have row_21 : (gl_and20 zero N one N) := by
    apply row_22
    exact row_42
  -- chapter_129_line_20: GL tag expansion.
  have row_20 : ((N zero) ∧ (N one)) := by
    simpa only [gl_and20] using (row_21)
  -- chapter_129_line_19: GL tag disintegration.
  have row_19 : (N one) := by
    exact row_20.2
  -- chapter_129_line_18: GL tag task formulation.
  have row_18 : (mul one v1 v2) := by
    exact premise_1
  -- chapter_129_line_17: GL tag implication.
  have row_17 : (N v1) := by
    apply row_30
    exact row_18
  -- chapter_129_line_16: GL tag implication.
  have row_16 : (gl_existence1 N v1 one mul) := by
    apply row_28
    exact row_17
    exact row_19
  -- chapter_129_line_15: GL tag expansion.
  have row_15 : (¬ (∀ (v3 : α), ((N v3) → (¬ (mul v1 one v3))))) := by
    simpa only [gl_existence1] using (row_16)
  have exists_row_15 : ∃ (v3 : α), ((N v3) ∧ (mul v1 one v3)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_15
      intro v3 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v3, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v3, witness_row_15⟩ := exists_row_15
  -- chapter_129_line_14: GL tag disintegration.
  have row_14 : (mul v1 one v3) := by
    exact witness_row_15.2
  -- chapter_129_line_13: GL tag disintegration.
  have row_13 : (N v3) := by
    exact witness_row_15.1
  -- chapter_129_line_12: GL tag implication.
  have row_12 : (gl_existence1 N v1 zero mul) := by
    apply row_28
    exact row_17
    exact row_27
  -- chapter_129_line_11: GL tag expansion.
  have row_11 : (¬ (∀ (v7 : α), ((N v7) → (¬ (mul v1 zero v7))))) := by
    simpa only [gl_existence1] using (row_12)
  have exists_row_11 : ∃ (v7 : α), ((N v7) ∧ (mul v1 zero v7)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_11
      intro v7 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v7, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v7, witness_row_11⟩ := exists_row_11
  -- chapter_129_line_10: GL tag disintegration.
  have row_10 : (mul v1 zero v7) := by
    exact witness_row_11.2
  -- chapter_129_line_9: GL tag implication.
  have row_9 : (v7 = zero) := by
    apply row_25
    exact row_17
    exact row_10
  -- chapter_129_line_8: GL tag implication.
  have row_8 : (add v7 v1 v3) := by
    apply row_23
    exact row_27
    exact row_42
    exact row_10
    exact row_14
  -- chapter_129_line_7: GL tag equality1.
  have row_7 : (add zero v1 v3) := by
    have equality_source := row_8
    have equality_step_1 := row_9
    cases equality_step_1
    exact equality_source
  -- chapter_129_line_6: GL tag theorem.
  have row_6 := peano_source_032 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_129_line_5: GL tag implication.
  have row_5 : (v1 = v3) := by
    apply row_6
    exact row_17
    exact row_7
  -- chapter_129_line_4: GL tag equality1.
  have row_4 : (mul one v3 v2) := by
    have equality_source := row_18
    have equality_step_1 := row_5
    cases equality_step_1
    exact equality_source
  -- chapter_129_line_3: GL tag theorem.
  have row_3 := peano_source_060 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_129_line_2: GL tag implication.
  have row_2 : (v3 = v2) := by
    apply row_3
    exact row_13
    exact row_4
  -- chapter_129_line_1: GL tag equality1.
  have row_1 : (mul v1 one v2) := by
    have equality_source := row_14
    have equality_step_1 := row_2
    cases equality_step_1
    exact equality_source
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
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
  -- chapter_39_line_35: GL tag task formulation.
  have row_35 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_39_line_34: GL tag expansion.
  have row_34 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_35)
  -- chapter_39_line_33: GL tag disintegration.
  have row_33 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_34.1
  -- chapter_39_line_32: GL tag expansion.
  have row_32 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_33)
  -- chapter_39_line_31: GL tag disintegration.
  have row_31 : (gl_fXYZ add N N N) := by
    exact row_32.1.1.1.1.1.1.1.1.1.2
  -- chapter_39_line_30: GL tag expansion.
  have row_30 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_31)
  -- chapter_39_line_29: GL tag disintegration.
  have row_29 : (gl_implication12 add N) := by
    exact row_30.1.1.1.2
  -- chapter_39_line_28: GL tag expansion.
  have row_28 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_29)
  -- chapter_39_line_27: GL tag disintegration.
  have row_27 : (gl_implication11 add N) := by
    exact row_30.1.1.1.1
  -- chapter_39_line_26: GL tag expansion.
  have row_26 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_27)
  -- chapter_39_line_25: GL tag disintegration.
  have row_25 : (gl_implication13 add N) := by
    exact row_30.1.1.2
  -- chapter_39_line_24: GL tag expansion.
  have row_24 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_25)
  -- chapter_39_line_23: GL tag disintegration.
  have row_23 : (gl_implication18 N N N add) := by
    exact row_30.1.2
  -- chapter_39_line_22: GL tag expansion.
  have row_22 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication18] using (row_23)
  -- chapter_39_line_21: GL tag implication grouping.
  have row_21 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_26
    have auxiliary_source_2 := row_28
    have auxiliary_source_3 := row_24
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_39_line_20: GL tag task formulation.
  have row_20 : (add v4 v5 v6) := by
    exact premise_2
  -- chapter_39_line_19: GL tag implication.
  have row_19 : (gl_and21 v4 N v5 N v6 N) := by
    apply row_21
    exact row_20
  -- chapter_39_line_18: GL tag expansion.
  have row_18 : (((N v4) ∧ (N v5)) ∧ (N v6)) := by
    simpa only [gl_and21] using (row_19)
  -- chapter_39_line_17: GL tag disintegration.
  have row_17 : (N v4) := by
    exact row_18.1.1
  -- chapter_39_line_16: GL tag task formulation.
  have row_16 : (add v2 v1 v5) := by
    exact premise_3
  -- chapter_39_line_15: GL tag implication.
  have row_15 : (N v2) := by
    apply row_26
    exact row_16
  -- chapter_39_line_14: GL tag implication.
  have row_14 : (gl_existence1 N v4 v2 add) := by
    apply row_22
    exact row_17
    exact row_15
  -- chapter_39_line_13: GL tag implication.
  have row_13 : (N v1) := by
    apply row_28
    exact row_16
  -- chapter_39_line_12: GL tag implication.
  have row_12 : (gl_existence1 N v4 v1 add) := by
    apply row_22
    exact row_17
    exact row_13
  -- chapter_39_line_11: GL tag expansion.
  have row_11 : (¬ (∀ (v7 : α), ((N v7) → (¬ (add v4 v1 v7))))) := by
    simpa only [gl_existence1] using (row_12)
  have exists_row_11 : ∃ (v7 : α), ((N v7) ∧ (add v4 v1 v7)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_11
      intro v7 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v7, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v7, witness_row_11⟩ := exists_row_11
  -- chapter_39_line_10: GL tag disintegration.
  have row_10 : (add v4 v1 v7) := by
    exact witness_row_11.2
  -- chapter_39_line_9: GL tag expansion.
  have row_9 : (¬ (∀ (v11 : α), ((N v11) → (¬ (add v4 v2 v11))))) := by
    simpa only [gl_existence1] using (row_14)
  have exists_row_9 : ∃ (v11 : α), ((N v11) ∧ (add v4 v2 v11)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_9
      intro v11 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v11, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v11, witness_row_9⟩ := exists_row_9
  -- chapter_39_line_8: GL tag disintegration.
  have row_8 : (add v4 v2 v11) := by
    exact witness_row_9.2
  -- chapter_39_line_7: GL tag theorem.
  have row_7 := peano_source_019 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_39_line_6: GL tag implication.
  have row_6 : (add v11 v1 v6) := by
    apply row_7
    exact row_8
    exact row_20
    exact row_16
  -- chapter_39_line_5: GL tag theorem.
  have row_5 := peano_source_021 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_39_line_4: GL tag implication.
  have row_4 : (add v7 v2 v6) := by
    apply row_5
    exact row_6
    exact row_8
    exact row_10
  -- chapter_39_line_3: GL tag task formulation.
  have row_3 : (add v1 v2 v3) := by
    exact premise_1
  -- chapter_39_line_2: GL tag theorem.
  have row_2 := peano_source_022 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_39_line_1: GL tag implication.
  have row_1 : (add v4 v3 v6) := by
    apply row_2
    exact row_4
    exact row_10
    exact row_3
  exact row_1

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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α), ((add v1 v4 v3) → (v2 = v4))))) := by
  intro v1
  intro v2
  intro v3
  intro premise_1
  intro v4
  intro premise_2
  -- chapter_71_line_8: GL tag task formulation.
  have row_8 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_71_line_7: GL tag theorem.
  have row_7 := peano_source_036 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_71_line_6: GL tag task formulation.
  have row_6 : (add v1 v2 v3) := by
    exact premise_1
  -- chapter_71_line_5: GL tag implication.
  have row_5 : (add v2 v1 v3) := by
    apply row_7
    exact row_6
  -- chapter_71_line_4: GL tag task formulation.
  have row_4 : (add v1 v4 v3) := by
    exact premise_2
  -- chapter_71_line_3: GL tag implication.
  have row_3 : (add v4 v1 v3) := by
    apply row_7
    exact row_4
  -- chapter_71_line_2: GL tag theorem.
  have row_2 := peano_source_031 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_71_line_1: GL tag implication.
  have row_1 : (v2 = v4) := by
    apply row_2
    exact row_5
    exact row_3
  exact row_1

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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((gl_preorder N add v1 v2) → ((gl_preorder N add v2 v1) → (v1 = v2)))) := by
  intro v1
  intro v2
  intro premise_1
  intro premise_2
  -- chapter_73_line_40: GL tag task formulation.
  have row_40 : (gl_preorder N add v2 v1) := by
    exact premise_2
  -- chapter_73_line_39: GL tag expansion.
  have row_39 : (¬ (∀ (v6 : α), ((N v6) → (¬ (add v2 v6 v1))))) := by
    simpa only [gl_preorder] using (row_40)
  have exists_row_39 : ∃ (v6 : α), ((N v6) ∧ (add v2 v6 v1)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_39
      intro v6 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v6, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v6, witness_row_39⟩ := exists_row_39
  -- chapter_73_line_38: GL tag disintegration.
  have row_38 : (N v6) := by
    exact witness_row_39.1
  -- chapter_73_line_37: GL tag disintegration.
  have row_37 : (add v2 v6 v1) := by
    exact witness_row_39.2
  -- chapter_73_line_36: GL tag task formulation.
  have row_36 : (gl_preorder N add v1 v2) := by
    exact premise_1
  -- chapter_73_line_35: GL tag expansion.
  have row_35 : (¬ (∀ (v5 : α), ((N v5) → (¬ (add v1 v5 v2))))) := by
    simpa only [gl_preorder] using (row_36)
  have exists_row_35 : ∃ (v5 : α), ((N v5) ∧ (add v1 v5 v2)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_35
      intro v5 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v5, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v5, witness_row_35⟩ := exists_row_35
  -- chapter_73_line_34: GL tag disintegration.
  have row_34 : (add v1 v5 v2) := by
    exact witness_row_35.2
  -- chapter_73_line_33: GL tag disintegration.
  have row_33 : (N v5) := by
    exact witness_row_35.1
  -- chapter_73_line_32: GL tag task formulation.
  have row_32 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_73_line_31: GL tag expansion.
  have row_31 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_32)
  -- chapter_73_line_30: GL tag disintegration.
  have row_30 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_31.1
  -- chapter_73_line_29: GL tag expansion.
  have row_29 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_30)
  -- chapter_73_line_28: GL tag disintegration.
  have row_28 : (gl_fXYZ add N N N) := by
    exact row_29.1.1.1.1.1.1.1.1.1.2
  -- chapter_73_line_27: GL tag expansion.
  have row_27 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_28)
  -- chapter_73_line_26: GL tag disintegration.
  have row_26 : (gl_implication13 add N) := by
    exact row_27.1.1.2
  -- chapter_73_line_25: GL tag expansion.
  have row_25 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_26)
  -- chapter_73_line_24: GL tag disintegration.
  have row_24 : (gl_implication18 N N N add) := by
    exact row_27.1.2
  -- chapter_73_line_23: GL tag expansion.
  have row_23 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication18] using (row_24)
  -- chapter_73_line_22: GL tag disintegration.
  have row_22 : (gl_implication21 N zero add) := by
    exact row_29.1.1.1.1.1.1.1.1.2
  -- chapter_73_line_21: GL tag expansion.
  have row_21 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((add w1 zero w2) → (w1 = w2))))) := by
    simpa only [gl_implication21] using (row_22)
  -- chapter_73_line_20: GL tag implication.
  have row_20 : (N v1) := by
    apply row_25
    exact row_37
  -- chapter_73_line_19: GL tag implication.
  have row_19 : (gl_existence1 N v5 v6 add) := by
    apply row_23
    exact row_33
    exact row_38
  -- chapter_73_line_18: GL tag expansion.
  have row_18 : (¬ (∀ (v7 : α), ((N v7) → (¬ (add v5 v6 v7))))) := by
    simpa only [gl_existence1] using (row_19)
  have exists_row_18 : ∃ (v7 : α), ((N v7) ∧ (add v5 v6 v7)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_18
      intro v7 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v7, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v7, witness_row_18⟩ := exists_row_18
  -- chapter_73_line_17: GL tag disintegration.
  have row_17 : (add v5 v6 v7) := by
    exact witness_row_18.2
  -- chapter_73_line_16: GL tag variable copy.
  have row_16 : (v1 = v1) := by
    rfl
  -- chapter_73_line_15: GL tag equality1.
  have row_15 : (add v2 v6 v1) := by
    have equality_source := row_37
    have equality_step_1 := row_16
    cases equality_step_1
    exact equality_source
  -- chapter_73_line_14: GL tag theorem.
  have row_14 := peano_source_022 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_73_line_13: GL tag implication.
  have row_13 : (add v1 v7 v1) := by
    apply row_14
    exact row_15
    exact row_34
    exact row_17
  -- chapter_73_line_12: GL tag theorem.
  have row_12 := peano_source_036 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_73_line_11: GL tag implication.
  have row_11 : (add v7 v1 v1) := by
    apply row_12
    exact row_13
  -- chapter_73_line_10: GL tag implication.
  have row_10 : (add v6 v5 v7) := by
    apply row_12
    exact row_17
  -- chapter_73_line_9: GL tag theorem.
  have row_9 := peano_source_038 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_73_line_8: GL tag implication.
  have row_8 : (zero = v7) := by
    apply row_9
    exact row_11
    exact row_16
  -- chapter_73_line_7: GL tag symmetry of equality.
  have row_7 : (v7 = zero) := by
    exact Eq.symm row_8
  -- chapter_73_line_6: GL tag equality1.
  have row_6 : (add v6 v5 zero) := by
    have equality_source := row_10
    have equality_step_1 := row_7
    cases equality_step_1
    exact equality_source
  -- chapter_73_line_5: GL tag theorem.
  have row_5 := peano_source_029 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_73_line_4: GL tag implication.
  have row_4 : (zero = v5) := by
    apply row_5
    exact row_6
    exact row_38
  -- chapter_73_line_3: GL tag symmetry of equality.
  have row_3 : (v5 = zero) := by
    exact Eq.symm row_4
  -- chapter_73_line_2: GL tag equality1.
  have row_2 : (add v1 zero v2) := by
    have equality_source := row_34
    have equality_step_1 := row_3
    cases equality_step_1
    exact equality_source
  -- chapter_73_line_1: GL tag implication.
  have row_1 : (v1 = v2) := by
    apply row_21
    exact row_20
    exact row_2
  exact row_1

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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v2 : α)
    (v5 : α)
    (v6 : α)
    (assumption_2 : (add v2 v5 v6))
    : (N v5) := by
  -- chapter_80_line_10: GL tag task formulation.
  have row_10 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_80_line_9: GL tag expansion.
  have row_9 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_10)
  -- chapter_80_line_8: GL tag disintegration.
  have row_8 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_9.1
  -- chapter_80_line_7: GL tag expansion.
  have row_7 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_8)
  -- chapter_80_line_6: GL tag disintegration.
  have row_6 : (gl_fXYZ add N N N) := by
    exact row_7.1.1.1.1.1.1.1.1.1.2
  -- chapter_80_line_5: GL tag expansion.
  have row_5 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_6)
  -- chapter_80_line_4: GL tag disintegration.
  have row_4 : (gl_implication12 add N) := by
    exact row_5.1.1.1.2
  -- chapter_80_line_3: GL tag expansion.
  have row_3 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_4)
  -- chapter_80_line_2: GL tag task formulation.
  have row_2 : (add v2 v5 v6) := by
    exact assumption_2
  -- chapter_80_line_1: GL tag implication.
  have row_1 : (N v5) := by
    apply row_3
    exact row_2
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (v4 : α)
    (v5 : α)
    (v6 : α)
    (assumption_50 : (add v4 v5 v1))
    (assumption_49 : (v5 = zero))
    (assumption_31 : (add v1 v2 v3))
    (assumption_27 : (add v2 v5 v6))
    : (add v6 v4 v3) := by
  -- chapter_81_line_50: GL tag task formulation.
  have row_50 : (add v4 v5 v1) := by
    exact assumption_50
  -- chapter_81_line_49: GL tag recursion.
  have row_49 : (v5 = zero) := by
    exact assumption_49
  -- chapter_81_line_48: GL tag equality1.
  have row_48 : (add v4 zero v1) := by
    have equality_source := row_50
    have equality_step_1 := row_49
    cases equality_step_1
    exact equality_source
  -- chapter_81_line_47: GL tag task formulation.
  have row_47 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_81_line_46: GL tag expansion.
  have row_46 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_47)
  -- chapter_81_line_45: GL tag disintegration.
  have row_45 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_46.1
  -- chapter_81_line_44: GL tag expansion.
  have row_44 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_45)
  -- chapter_81_line_43: GL tag disintegration.
  have row_43 : (gl_fXYZ add N N N) := by
    exact row_44.1.1.1.1.1.1.1.1.1.2
  -- chapter_81_line_42: GL tag expansion.
  have row_42 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_43)
  -- chapter_81_line_41: GL tag disintegration.
  have row_41 : (gl_implication13 add N) := by
    exact row_42.1.1.2
  -- chapter_81_line_40: GL tag expansion.
  have row_40 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_41)
  -- chapter_81_line_39: GL tag disintegration.
  have row_39 : (gl_implication12 add N) := by
    exact row_42.1.1.1.2
  -- chapter_81_line_38: GL tag expansion.
  have row_38 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_39)
  -- chapter_81_line_37: GL tag disintegration.
  have row_37 : (gl_implication11 add N) := by
    exact row_42.1.1.1.1
  -- chapter_81_line_36: GL tag expansion.
  have row_36 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_37)
  -- chapter_81_line_35: GL tag disintegration.
  have row_35 : (gl_implication18 N N N add) := by
    exact row_42.1.2
  -- chapter_81_line_34: GL tag expansion.
  have row_34 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication18] using (row_35)
  -- chapter_81_line_33: GL tag disintegration.
  have row_33 : (N zero) := by
    exact row_44.1.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_81_line_32: GL tag implication grouping.
  have row_32 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_36
    have auxiliary_source_2 := row_38
    have auxiliary_source_3 := row_40
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_81_line_31: GL tag task formulation.
  have row_31 : (add v1 v2 v3) := by
    exact assumption_31
  -- chapter_81_line_30: GL tag implication.
  have row_30 : (gl_and21 v1 N v2 N v3 N) := by
    apply row_32
    exact row_31
  -- chapter_81_line_29: GL tag expansion.
  have row_29 : (((N v1) ∧ (N v2)) ∧ (N v3)) := by
    simpa only [gl_and21] using (row_30)
  -- chapter_81_line_28: GL tag disintegration.
  have row_28 : (N v2) := by
    exact row_29.1.2
  -- chapter_81_line_27: GL tag task formulation.
  have row_27 : (add v2 v5 v6) := by
    exact assumption_27
  -- chapter_81_line_26: GL tag equality1.
  have row_26 : (add v2 zero v6) := by
    have equality_source := row_27
    have equality_step_1 := row_49
    cases equality_step_1
    exact equality_source
  -- chapter_81_line_25: GL tag theorem.
  have row_25 := peano_source_011 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_81_line_24: GL tag implication.
  have row_24 : (v2 = v6) := by
    apply row_25
    exact row_26
    exact row_28
  -- chapter_81_line_23: GL tag implication.
  have row_23 : (gl_and21 v4 N v5 N v1 N) := by
    apply row_32
    exact row_50
  -- chapter_81_line_22: GL tag expansion.
  have row_22 : (((N v4) ∧ (N v5)) ∧ (N v1)) := by
    simpa only [gl_and21] using (row_23)
  -- chapter_81_line_21: GL tag disintegration.
  have row_21 : (N v4) := by
    exact row_22.1.1
  -- chapter_81_line_20: GL tag disintegration.
  have row_20 : (N v5) := by
    exact row_22.1.2
  -- chapter_81_line_19: GL tag implication.
  have row_19 : (v4 = v1) := by
    apply row_25
    exact row_48
    exact row_21
  -- chapter_81_line_18: GL tag implication.
  have row_18 : (gl_existence1 N v5 v2 add) := by
    apply row_34
    exact row_20
    exact row_28
  -- chapter_81_line_17: GL tag expansion.
  have row_17 : (¬ (∀ (v10 : α), ((N v10) → (¬ (add v5 v2 v10))))) := by
    simpa only [gl_existence1] using (row_18)
  have exists_row_17 : ∃ (v10 : α), ((N v10) ∧ (add v5 v2 v10)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_17
      intro v10 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v10, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v10, witness_row_17⟩ := exists_row_17
  -- chapter_81_line_16: GL tag disintegration.
  have row_16 : (add v5 v2 v10) := by
    exact witness_row_17.2
  -- chapter_81_line_15: GL tag theorem.
  have row_15 := peano_source_022 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_81_line_14: GL tag implication.
  have row_14 : (add v4 v10 v3) := by
    apply row_15
    exact row_31
    exact row_50
    exact row_16
  -- chapter_81_line_13: GL tag implication.
  have row_13 : (gl_and21 v4 N v10 N v3 N) := by
    apply row_32
    exact row_14
  -- chapter_81_line_12: GL tag expansion.
  have row_12 : (((N v4) ∧ (N v10)) ∧ (N v3)) := by
    simpa only [gl_and21] using (row_13)
  -- chapter_81_line_11: GL tag disintegration.
  have row_11 : (N v3) := by
    exact row_12.2
  -- chapter_81_line_10: GL tag implication.
  have row_10 : (gl_existence1 N v3 zero add) := by
    apply row_34
    exact row_11
    exact row_33
  -- chapter_81_line_9: GL tag expansion.
  have row_9 : (¬ (∀ (v7 : α), ((N v7) → (¬ (add v3 zero v7))))) := by
    simpa only [gl_existence1] using (row_10)
  have exists_row_9 : ∃ (v7 : α), ((N v7) ∧ (add v3 zero v7)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_9
      intro v7 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v7, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v7, witness_row_9⟩ := exists_row_9
  -- chapter_81_line_8: GL tag disintegration.
  have row_8 : (add v3 zero v7) := by
    exact witness_row_9.2
  -- chapter_81_line_7: GL tag implication.
  have row_7 : (v3 = v7) := by
    apply row_25
    exact row_8
    exact row_11
  -- chapter_81_line_6: GL tag equality1.
  have row_6 : (add v1 v2 v7) := by
    have equality_source := row_31
    have equality_step_1 := row_7
    cases equality_step_1
    exact equality_source
  -- chapter_81_line_5: GL tag theorem.
  have row_5 := peano_source_036 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_81_line_4: GL tag implication.
  have row_4 : (add v2 v1 v7) := by
    apply row_5
    exact row_6
  -- chapter_81_line_3: GL tag symmetry of equality.
  have row_3 : (v1 = v4) := by
    exact Eq.symm row_19
  -- chapter_81_line_2: GL tag symmetry of equality.
  have row_2 : (v7 = v3) := by
    exact Eq.symm row_7
  -- chapter_81_line_1: GL tag equality1.
  have row_1 : (add v6 v4 v3) := by
    have equality_source := row_4
    have equality_step_1 := row_3
    cases equality_step_1
    have equality_step_2 := row_24
    cases equality_step_2
    have equality_step_3 := row_2
    cases equality_step_3
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (v4 : α)
    (v5 : α)
    (v6 : α)
    (assumption_38 : (succ previous v5))
    (assumption_34 : (add v1 v2 v3))
    (assumption_25 : (add v4 v5 v1))
    (assumption_6 : (∀ (w1 : α) (w2 : α), ((add w1 v2 w2) → ((add v4 previous w1) → (∀ (w3 : α), ((add v2 previous w3) → (add w3 v4 w2)))))))
    (assumption_4 : (add v2 v5 v6))
    : (add v6 v4 v3) := by
  -- chapter_82_line_62: GL tag task formulation.
  have row_62 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_82_line_61: GL tag expansion.
  have row_61 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_62)
  -- chapter_82_line_60: GL tag disintegration.
  have row_60 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_61.1
  -- chapter_82_line_59: GL tag expansion.
  have row_59 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_60)
  -- chapter_82_line_58: GL tag disintegration.
  have row_58 : (gl_implication9 N zero succ) := by
    exact row_59.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_82_line_57: GL tag expansion.
  have row_57 : (∀ (w1 : α), ((N w1) → (¬ (succ w1 zero)))) := by
    simpa only [gl_implication9] using (row_58)
  -- chapter_82_line_56: GL tag disintegration.
  have row_56 : (gl_fXY succ N N) := by
    exact row_59.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_82_line_55: GL tag expansion.
  have row_55 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_56)
  -- chapter_82_line_54: GL tag disintegration.
  have row_54 : (gl_implication0 succ N) := by
    exact row_55.1.1.1
  -- chapter_82_line_53: GL tag expansion.
  have row_53 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_54)
  -- chapter_82_line_52: GL tag disintegration.
  have row_52 : (gl_implication6 N N succ) := by
    exact row_55.1.2
  -- chapter_82_line_51: GL tag expansion.
  have row_51 : (∀ (w1 : α), ((N w1) → (gl_existence0 N w1 succ))) := by
    simpa only [gl_implication6] using (row_52)
  -- chapter_82_line_50: GL tag disintegration.
  have row_50 : (gl_implication23 N succ add) := by
    exact row_59.1.1.1.1.1.1.2
  -- chapter_82_line_49: GL tag expansion.
  have row_49 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication23] using (row_50)
  -- chapter_82_line_48: GL tag disintegration.
  have row_48 : (gl_fXYZ add N N N) := by
    exact row_59.1.1.1.1.1.1.1.1.1.2
  -- chapter_82_line_47: GL tag expansion.
  have row_47 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_48)
  -- chapter_82_line_46: GL tag disintegration.
  have row_46 : (gl_implication13 add N) := by
    exact row_47.1.1.2
  -- chapter_82_line_45: GL tag expansion.
  have row_45 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_46)
  -- chapter_82_line_44: GL tag disintegration.
  have row_44 : (gl_implication12 add N) := by
    exact row_47.1.1.1.2
  -- chapter_82_line_43: GL tag expansion.
  have row_43 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_44)
  -- chapter_82_line_42: GL tag disintegration.
  have row_42 : (gl_implication11 add N) := by
    exact row_47.1.1.1.1
  -- chapter_82_line_41: GL tag expansion.
  have row_41 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_42)
  -- chapter_82_line_40: GL tag disintegration.
  have row_40 : (gl_implication18 N N N add) := by
    exact row_47.1.2
  -- chapter_82_line_39: GL tag expansion.
  have row_39 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication18] using (row_40)
  -- chapter_82_line_38: GL tag recursion.
  have row_38 : (succ previous v5) := by
    exact assumption_38
  -- chapter_82_line_37: GL tag implication.
  have row_37 : (N previous) := by
    apply row_53
    exact row_38
  -- chapter_82_line_36: GL tag implication grouping.
  have row_36 : (∀ (w1 : α), ((N w1) → (gl_and3 N w1 succ zero))) := by
    have auxiliary_source_1 := row_51
    have auxiliary_source_2 := row_57
    intro auxiliary_argument_1
    intro auxiliary_premise
    exact ⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_premise)⟩
  -- chapter_82_line_35: GL tag implication grouping.
  have row_35 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_41
    have auxiliary_source_2 := row_43
    have auxiliary_source_3 := row_45
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_82_line_34: GL tag task formulation.
  have row_34 : (add v1 v2 v3) := by
    exact assumption_34
  -- chapter_82_line_33: GL tag implication.
  have row_33 : (gl_and21 v1 N v2 N v3 N) := by
    apply row_35
    exact row_34
  -- chapter_82_line_32: GL tag expansion.
  have row_32 : (((N v1) ∧ (N v2)) ∧ (N v3)) := by
    simpa only [gl_and21] using (row_33)
  -- chapter_82_line_31: GL tag disintegration.
  have row_31 : (N v2) := by
    exact row_32.1.2
  -- chapter_82_line_30: GL tag implication.
  have row_30 : (gl_and3 N v2 succ zero) := by
    apply row_36
    exact row_31
  -- chapter_82_line_29: GL tag expansion.
  have row_29 : ((gl_existence0 N v2 succ) ∧ (¬ (succ v2 zero))) := by
    simpa only [gl_and3] using (row_30)
  -- chapter_82_line_28: GL tag disintegration.
  have row_28 : (gl_existence0 N v2 succ) := by
    exact row_29.1
  -- chapter_82_line_27: GL tag expansion.
  have row_27 : (¬ (∀ (v14 : α), ((N v14) → (¬ (succ v2 v14))))) := by
    simpa only [gl_existence0] using (row_28)
  have exists_row_27 : ∃ (v14 : α), ((N v14) ∧ (succ v2 v14)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_27
      intro v14 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v14, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v14, witness_row_27⟩ := exists_row_27
  -- chapter_82_line_26: GL tag disintegration.
  have row_26 : (succ v2 v14) := by
    exact witness_row_27.2
  -- chapter_82_line_25: GL tag task formulation.
  have row_25 : (add v4 v5 v1) := by
    exact assumption_25
  -- chapter_82_line_24: GL tag implication.
  have row_24 : (gl_and21 v4 N v5 N v1 N) := by
    apply row_35
    exact row_25
  -- chapter_82_line_23: GL tag expansion.
  have row_23 : (((N v4) ∧ (N v5)) ∧ (N v1)) := by
    simpa only [gl_and21] using (row_24)
  -- chapter_82_line_22: GL tag disintegration.
  have row_22 : (N v4) := by
    exact row_23.1.1
  -- chapter_82_line_21: GL tag implication.
  have row_21 : (gl_existence1 N v4 previous add) := by
    apply row_39
    exact row_22
    exact row_37
  -- chapter_82_line_20: GL tag expansion.
  have row_20 : (¬ (∀ (v15 : α), ((N v15) → (¬ (add v4 previous v15))))) := by
    simpa only [gl_existence1] using (row_21)
  have exists_row_20 : ∃ (v15 : α), ((N v15) ∧ (add v4 previous v15)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_20
      intro v15 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v15, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v15, witness_row_20⟩ := exists_row_20
  -- chapter_82_line_19: GL tag disintegration.
  have row_19 : (N v15) := by
    exact witness_row_20.1
  -- chapter_82_line_18: GL tag disintegration.
  have row_18 : (add v4 previous v15) := by
    exact witness_row_20.2
  -- chapter_82_line_17: GL tag implication.
  have row_17 : (gl_existence1 N v15 v2 add) := by
    apply row_39
    exact row_19
    exact row_31
  -- chapter_82_line_16: GL tag implication.
  have row_16 : (gl_existence1 N v2 previous add) := by
    apply row_39
    exact row_31
    exact row_37
  -- chapter_82_line_15: GL tag implication.
  have row_15 : (succ v15 v1) := by
    apply row_49
    exact row_37
    exact row_38
    exact row_18
    exact row_25
  -- chapter_82_line_14: GL tag theorem.
  have row_14 := peano_source_041 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_82_line_13: GL tag implication.
  have row_13 : (add v15 v14 v3) := by
    apply row_14
    exact row_34
    exact row_15
    exact row_26
  -- chapter_82_line_12: GL tag expansion.
  have row_12 : (¬ (∀ (v8 : α), ((N v8) → (¬ (add v15 v2 v8))))) := by
    simpa only [gl_existence1] using (row_17)
  have exists_row_12 : ∃ (v8 : α), ((N v8) ∧ (add v15 v2 v8)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_12
      intro v8 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v8, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v8, witness_row_12⟩ := exists_row_12
  -- chapter_82_line_11: GL tag disintegration.
  have row_11 : (add v15 v2 v8) := by
    exact witness_row_12.2
  -- chapter_82_line_10: GL tag theorem.
  have row_10 := peano_source_017 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_82_line_9: GL tag implication.
  have row_9 : (succ v8 v3) := by
    apply row_10
    exact row_13
    exact row_11
    exact row_26
  -- chapter_82_line_8: GL tag expansion.
  have row_8 : (¬ (∀ (v7 : α), ((N v7) → (¬ (add v2 previous v7))))) := by
    simpa only [gl_existence1] using (row_16)
  have exists_row_8 : ∃ (v7 : α), ((N v7) ∧ (add v2 previous v7)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_8
      intro v7 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v7, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v7, witness_row_8⟩ := exists_row_8
  -- chapter_82_line_7: GL tag disintegration.
  have row_7 : (add v2 previous v7) := by
    exact witness_row_8.2
  -- chapter_82_line_6: GL tag recursion.
  have row_6 : (∀ (w1 : α) (w2 : α), ((add w1 v2 w2) → ((add v4 previous w1) → (∀ (w3 : α), ((add v2 previous w3) → (add w3 v4 w2)))))) := by
    exact assumption_6
  -- chapter_82_line_5: GL tag implication.
  have row_5 : (add v7 v4 v8) := by
    apply row_6
    exact row_11
    exact row_18
    exact row_7
  -- chapter_82_line_4: GL tag task formulation.
  have row_4 : (add v2 v5 v6) := by
    exact assumption_4
  -- chapter_82_line_3: GL tag implication.
  have row_3 : (succ v7 v6) := by
    apply row_49
    exact row_37
    exact row_38
    exact row_7
    exact row_4
  -- chapter_82_line_2: GL tag theorem.
  have row_2 := peano_source_040 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_82_line_1: GL tag implication.
  have row_1 : (add v6 v4 v3) := by
    apply row_2
    exact row_5
    exact row_3
    exact row_9
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α) (v5 : α), ((add v4 v5 v1) → (∀ (v6 : α), ((add v2 v5 v6) → (add v6 v4 v3))))))) := by
  intro v1
  intro v2
  intro v3
  intro premise_1
  intro v4
  intro v5
  intro premise_2
  intro v6
  intro premise_3
  have inductionMember : N v5 := by
    have typingRule := peano_source_042_induction_typing N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact typingRule v2 v5 v6 premise_3
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α), ((add v4 zero v1) → (∀ (v6 : α), ((add v2 zero v6) → (add v6 v4 v3))))))) := by
    intro v1
    intro v2
    intro v3
    intro base_premise_1
    intro v4
    intro base_premise_2
    intro v6
    intro base_premise_3
    have zeroRule := peano_source_042_check_zero N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact zeroRule v1 v2 v3 v4 zero v6 base_premise_2 rfl base_premise_1 base_premise_3
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α), ((add v4 induction_n v1) → (∀ (v6 : α), ((add v2 induction_n v6) → (add v6 v4 v3))))))) → ∀ induction_m, succ induction_n induction_m → (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α), ((add v4 induction_m v1) → (∀ (v6 : α), ((add v2 induction_m v6) → (add v6 v4 v3))))))) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro v1
    intro v2
    intro v3
    intro step_premise_1
    intro v4
    intro step_premise_2
    intro v6
    intro step_premise_3
    have step_induction_assumption_1 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have step_induction_assumption_2 :
        (∀ (w1 : α) (w2 : α), ((add w1 v2 w2) → ((add v4 induction_n w1) → (∀ (w3 : α), ((add v2 induction_n w3) → (add w3 v4 w2)))))) := by
      intro w1
      intro w2
      intro step_induction_assumption_2_premise_1
      intro step_induction_assumption_2_premise_2
      intro w3
      intro step_induction_assumption_2_premise_3
      apply induction_hypothesis
      all_goals assumption
    have stepRule := peano_source_042_check_induction_condition N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact stepRule induction_n v1 v2 v3 v4 induction_m v6 step_induction_assumption_1 step_premise_1 step_premise_2 step_induction_assumption_2 step_premise_3
  have inductionProperty : (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α), ((add v4 v5 v1) → (∀ (v6 : α), ((add v2 v5 v6) → (add v6 v4 v3))))))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α), ((add v4 induction_value v1) → (∀ (v6 : α), ((add v2 induction_value v6) → (add v6 v4 v3))))))))
      v5
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v1 v2 v3 premise_1 v4 premise_2 v6 premise_3

private theorem peano_source_043_induction_typing
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v1 : α)
    (v4 : α)
    (v6 : α)
    (assumption_2 : (add v6 v4 v1))
    : (N v4) := by
  -- chapter_83_line_10: GL tag task formulation.
  have row_10 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_83_line_9: GL tag expansion.
  have row_9 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_10)
  -- chapter_83_line_8: GL tag disintegration.
  have row_8 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_9.1
  -- chapter_83_line_7: GL tag expansion.
  have row_7 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_8)
  -- chapter_83_line_6: GL tag disintegration.
  have row_6 : (gl_fXYZ add N N N) := by
    exact row_7.1.1.1.1.1.1.1.1.1.2
  -- chapter_83_line_5: GL tag expansion.
  have row_5 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_6)
  -- chapter_83_line_4: GL tag disintegration.
  have row_4 : (gl_implication12 add N) := by
    exact row_5.1.1.1.2
  -- chapter_83_line_3: GL tag expansion.
  have row_3 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_4)
  -- chapter_83_line_2: GL tag task formulation.
  have row_2 : (add v6 v4 v1) := by
    exact assumption_2
  -- chapter_83_line_1: GL tag implication.
  have row_1 : (N v4) := by
    apply row_3
    exact row_2
  exact row_1

private theorem peano_source_043_check_zero
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (v4 : α)
    (v5 : α)
    (v6 : α)
    (assumption_50 : (add v6 v4 v1))
    (assumption_49 : (v4 = zero))
    (assumption_31 : (add v1 v2 v3))
    (assumption_27 : (add v2 v4 v5))
    : (add v5 v6 v3) := by
  -- chapter_84_line_50: GL tag task formulation.
  have row_50 : (add v6 v4 v1) := by
    exact assumption_50
  -- chapter_84_line_49: GL tag recursion.
  have row_49 : (v4 = zero) := by
    exact assumption_49
  -- chapter_84_line_48: GL tag equality1.
  have row_48 : (add v6 zero v1) := by
    have equality_source := row_50
    have equality_step_1 := row_49
    cases equality_step_1
    exact equality_source
  -- chapter_84_line_47: GL tag task formulation.
  have row_47 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_84_line_46: GL tag expansion.
  have row_46 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_47)
  -- chapter_84_line_45: GL tag disintegration.
  have row_45 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_46.1
  -- chapter_84_line_44: GL tag expansion.
  have row_44 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_45)
  -- chapter_84_line_43: GL tag disintegration.
  have row_43 : (gl_fXYZ add N N N) := by
    exact row_44.1.1.1.1.1.1.1.1.1.2
  -- chapter_84_line_42: GL tag expansion.
  have row_42 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_43)
  -- chapter_84_line_41: GL tag disintegration.
  have row_41 : (gl_implication13 add N) := by
    exact row_42.1.1.2
  -- chapter_84_line_40: GL tag expansion.
  have row_40 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_41)
  -- chapter_84_line_39: GL tag disintegration.
  have row_39 : (gl_implication12 add N) := by
    exact row_42.1.1.1.2
  -- chapter_84_line_38: GL tag expansion.
  have row_38 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_39)
  -- chapter_84_line_37: GL tag disintegration.
  have row_37 : (gl_implication11 add N) := by
    exact row_42.1.1.1.1
  -- chapter_84_line_36: GL tag expansion.
  have row_36 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_37)
  -- chapter_84_line_35: GL tag disintegration.
  have row_35 : (gl_implication18 N N N add) := by
    exact row_42.1.2
  -- chapter_84_line_34: GL tag expansion.
  have row_34 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication18] using (row_35)
  -- chapter_84_line_33: GL tag disintegration.
  have row_33 : (N zero) := by
    exact row_44.1.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_84_line_32: GL tag implication grouping.
  have row_32 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_36
    have auxiliary_source_2 := row_38
    have auxiliary_source_3 := row_40
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_84_line_31: GL tag task formulation.
  have row_31 : (add v1 v2 v3) := by
    exact assumption_31
  -- chapter_84_line_30: GL tag implication.
  have row_30 : (gl_and21 v1 N v2 N v3 N) := by
    apply row_32
    exact row_31
  -- chapter_84_line_29: GL tag expansion.
  have row_29 : (((N v1) ∧ (N v2)) ∧ (N v3)) := by
    simpa only [gl_and21] using (row_30)
  -- chapter_84_line_28: GL tag disintegration.
  have row_28 : (N v2) := by
    exact row_29.1.2
  -- chapter_84_line_27: GL tag task formulation.
  have row_27 : (add v2 v4 v5) := by
    exact assumption_27
  -- chapter_84_line_26: GL tag equality1.
  have row_26 : (add v2 zero v5) := by
    have equality_source := row_27
    have equality_step_1 := row_49
    cases equality_step_1
    exact equality_source
  -- chapter_84_line_25: GL tag theorem.
  have row_25 := peano_source_011 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_84_line_24: GL tag implication.
  have row_24 : (v2 = v5) := by
    apply row_25
    exact row_26
    exact row_28
  -- chapter_84_line_23: GL tag implication.
  have row_23 : (gl_and21 v6 N v4 N v1 N) := by
    apply row_32
    exact row_50
  -- chapter_84_line_22: GL tag expansion.
  have row_22 : (((N v6) ∧ (N v4)) ∧ (N v1)) := by
    simpa only [gl_and21] using (row_23)
  -- chapter_84_line_21: GL tag disintegration.
  have row_21 : (N v6) := by
    exact row_22.1.1
  -- chapter_84_line_20: GL tag disintegration.
  have row_20 : (N v4) := by
    exact row_22.1.2
  -- chapter_84_line_19: GL tag implication.
  have row_19 : (v6 = v1) := by
    apply row_25
    exact row_48
    exact row_21
  -- chapter_84_line_18: GL tag implication.
  have row_18 : (gl_existence1 N v4 v2 add) := by
    apply row_34
    exact row_20
    exact row_28
  -- chapter_84_line_17: GL tag expansion.
  have row_17 : (¬ (∀ (v10 : α), ((N v10) → (¬ (add v4 v2 v10))))) := by
    simpa only [gl_existence1] using (row_18)
  have exists_row_17 : ∃ (v10 : α), ((N v10) ∧ (add v4 v2 v10)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_17
      intro v10 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v10, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v10, witness_row_17⟩ := exists_row_17
  -- chapter_84_line_16: GL tag disintegration.
  have row_16 : (add v4 v2 v10) := by
    exact witness_row_17.2
  -- chapter_84_line_15: GL tag theorem.
  have row_15 := peano_source_022 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_84_line_14: GL tag implication.
  have row_14 : (add v6 v10 v3) := by
    apply row_15
    exact row_31
    exact row_50
    exact row_16
  -- chapter_84_line_13: GL tag implication.
  have row_13 : (gl_and21 v6 N v10 N v3 N) := by
    apply row_32
    exact row_14
  -- chapter_84_line_12: GL tag expansion.
  have row_12 : (((N v6) ∧ (N v10)) ∧ (N v3)) := by
    simpa only [gl_and21] using (row_13)
  -- chapter_84_line_11: GL tag disintegration.
  have row_11 : (N v3) := by
    exact row_12.2
  -- chapter_84_line_10: GL tag implication.
  have row_10 : (gl_existence1 N v3 zero add) := by
    apply row_34
    exact row_11
    exact row_33
  -- chapter_84_line_9: GL tag expansion.
  have row_9 : (¬ (∀ (v7 : α), ((N v7) → (¬ (add v3 zero v7))))) := by
    simpa only [gl_existence1] using (row_10)
  have exists_row_9 : ∃ (v7 : α), ((N v7) ∧ (add v3 zero v7)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_9
      intro v7 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v7, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v7, witness_row_9⟩ := exists_row_9
  -- chapter_84_line_8: GL tag disintegration.
  have row_8 : (add v3 zero v7) := by
    exact witness_row_9.2
  -- chapter_84_line_7: GL tag implication.
  have row_7 : (v3 = v7) := by
    apply row_25
    exact row_8
    exact row_11
  -- chapter_84_line_6: GL tag equality1.
  have row_6 : (add v1 v2 v7) := by
    have equality_source := row_31
    have equality_step_1 := row_7
    cases equality_step_1
    exact equality_source
  -- chapter_84_line_5: GL tag theorem.
  have row_5 := peano_source_036 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_84_line_4: GL tag implication.
  have row_4 : (add v2 v1 v7) := by
    apply row_5
    exact row_6
  -- chapter_84_line_3: GL tag symmetry of equality.
  have row_3 : (v1 = v6) := by
    exact Eq.symm row_19
  -- chapter_84_line_2: GL tag symmetry of equality.
  have row_2 : (v7 = v3) := by
    exact Eq.symm row_7
  -- chapter_84_line_1: GL tag equality1.
  have row_1 : (add v5 v6 v3) := by
    have equality_source := row_4
    have equality_step_1 := row_3
    cases equality_step_1
    have equality_step_2 := row_24
    cases equality_step_2
    have equality_step_3 := row_2
    cases equality_step_3
    exact equality_source
  exact row_1

private theorem peano_source_043_check_induction_condition
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (v4 : α)
    (v5 : α)
    (v6 : α)
    (assumption_38 : (succ previous v4))
    (assumption_34 : (add v1 v2 v3))
    (assumption_25 : (add v6 v4 v1))
    (assumption_6 : (∀ (w1 : α) (w2 : α), ((add w1 v2 w2) → (∀ (w3 : α), ((add v2 previous w3) → ((add v6 previous w1) → (add w3 v6 w2)))))))
    (assumption_4 : (add v2 v4 v5))
    : (add v5 v6 v3) := by
  -- chapter_85_line_62: GL tag task formulation.
  have row_62 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_85_line_61: GL tag expansion.
  have row_61 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_62)
  -- chapter_85_line_60: GL tag disintegration.
  have row_60 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_61.1
  -- chapter_85_line_59: GL tag expansion.
  have row_59 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_60)
  -- chapter_85_line_58: GL tag disintegration.
  have row_58 : (gl_implication9 N zero succ) := by
    exact row_59.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_85_line_57: GL tag expansion.
  have row_57 : (∀ (w1 : α), ((N w1) → (¬ (succ w1 zero)))) := by
    simpa only [gl_implication9] using (row_58)
  -- chapter_85_line_56: GL tag disintegration.
  have row_56 : (gl_fXY succ N N) := by
    exact row_59.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_85_line_55: GL tag expansion.
  have row_55 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_56)
  -- chapter_85_line_54: GL tag disintegration.
  have row_54 : (gl_implication0 succ N) := by
    exact row_55.1.1.1
  -- chapter_85_line_53: GL tag expansion.
  have row_53 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_54)
  -- chapter_85_line_52: GL tag disintegration.
  have row_52 : (gl_implication6 N N succ) := by
    exact row_55.1.2
  -- chapter_85_line_51: GL tag expansion.
  have row_51 : (∀ (w1 : α), ((N w1) → (gl_existence0 N w1 succ))) := by
    simpa only [gl_implication6] using (row_52)
  -- chapter_85_line_50: GL tag disintegration.
  have row_50 : (gl_implication23 N succ add) := by
    exact row_59.1.1.1.1.1.1.2
  -- chapter_85_line_49: GL tag expansion.
  have row_49 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication23] using (row_50)
  -- chapter_85_line_48: GL tag disintegration.
  have row_48 : (gl_fXYZ add N N N) := by
    exact row_59.1.1.1.1.1.1.1.1.1.2
  -- chapter_85_line_47: GL tag expansion.
  have row_47 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_48)
  -- chapter_85_line_46: GL tag disintegration.
  have row_46 : (gl_implication13 add N) := by
    exact row_47.1.1.2
  -- chapter_85_line_45: GL tag expansion.
  have row_45 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_46)
  -- chapter_85_line_44: GL tag disintegration.
  have row_44 : (gl_implication12 add N) := by
    exact row_47.1.1.1.2
  -- chapter_85_line_43: GL tag expansion.
  have row_43 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_44)
  -- chapter_85_line_42: GL tag disintegration.
  have row_42 : (gl_implication11 add N) := by
    exact row_47.1.1.1.1
  -- chapter_85_line_41: GL tag expansion.
  have row_41 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_42)
  -- chapter_85_line_40: GL tag disintegration.
  have row_40 : (gl_implication18 N N N add) := by
    exact row_47.1.2
  -- chapter_85_line_39: GL tag expansion.
  have row_39 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication18] using (row_40)
  -- chapter_85_line_38: GL tag recursion.
  have row_38 : (succ previous v4) := by
    exact assumption_38
  -- chapter_85_line_37: GL tag implication.
  have row_37 : (N previous) := by
    apply row_53
    exact row_38
  -- chapter_85_line_36: GL tag implication grouping.
  have row_36 : (∀ (w1 : α), ((N w1) → (gl_and3 N w1 succ zero))) := by
    have auxiliary_source_1 := row_51
    have auxiliary_source_2 := row_57
    intro auxiliary_argument_1
    intro auxiliary_premise
    exact ⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_premise)⟩
  -- chapter_85_line_35: GL tag implication grouping.
  have row_35 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_41
    have auxiliary_source_2 := row_43
    have auxiliary_source_3 := row_45
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_85_line_34: GL tag task formulation.
  have row_34 : (add v1 v2 v3) := by
    exact assumption_34
  -- chapter_85_line_33: GL tag implication.
  have row_33 : (gl_and21 v1 N v2 N v3 N) := by
    apply row_35
    exact row_34
  -- chapter_85_line_32: GL tag expansion.
  have row_32 : (((N v1) ∧ (N v2)) ∧ (N v3)) := by
    simpa only [gl_and21] using (row_33)
  -- chapter_85_line_31: GL tag disintegration.
  have row_31 : (N v2) := by
    exact row_32.1.2
  -- chapter_85_line_30: GL tag implication.
  have row_30 : (gl_and3 N v2 succ zero) := by
    apply row_36
    exact row_31
  -- chapter_85_line_29: GL tag expansion.
  have row_29 : ((gl_existence0 N v2 succ) ∧ (¬ (succ v2 zero))) := by
    simpa only [gl_and3] using (row_30)
  -- chapter_85_line_28: GL tag disintegration.
  have row_28 : (gl_existence0 N v2 succ) := by
    exact row_29.1
  -- chapter_85_line_27: GL tag expansion.
  have row_27 : (¬ (∀ (v14 : α), ((N v14) → (¬ (succ v2 v14))))) := by
    simpa only [gl_existence0] using (row_28)
  have exists_row_27 : ∃ (v14 : α), ((N v14) ∧ (succ v2 v14)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_27
      intro v14 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v14, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v14, witness_row_27⟩ := exists_row_27
  -- chapter_85_line_26: GL tag disintegration.
  have row_26 : (succ v2 v14) := by
    exact witness_row_27.2
  -- chapter_85_line_25: GL tag task formulation.
  have row_25 : (add v6 v4 v1) := by
    exact assumption_25
  -- chapter_85_line_24: GL tag implication.
  have row_24 : (gl_and21 v6 N v4 N v1 N) := by
    apply row_35
    exact row_25
  -- chapter_85_line_23: GL tag expansion.
  have row_23 : (((N v6) ∧ (N v4)) ∧ (N v1)) := by
    simpa only [gl_and21] using (row_24)
  -- chapter_85_line_22: GL tag disintegration.
  have row_22 : (N v6) := by
    exact row_23.1.1
  -- chapter_85_line_21: GL tag implication.
  have row_21 : (gl_existence1 N v6 previous add) := by
    apply row_39
    exact row_22
    exact row_37
  -- chapter_85_line_20: GL tag expansion.
  have row_20 : (¬ (∀ (v15 : α), ((N v15) → (¬ (add v6 previous v15))))) := by
    simpa only [gl_existence1] using (row_21)
  have exists_row_20 : ∃ (v15 : α), ((N v15) ∧ (add v6 previous v15)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_20
      intro v15 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v15, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v15, witness_row_20⟩ := exists_row_20
  -- chapter_85_line_19: GL tag disintegration.
  have row_19 : (N v15) := by
    exact witness_row_20.1
  -- chapter_85_line_18: GL tag disintegration.
  have row_18 : (add v6 previous v15) := by
    exact witness_row_20.2
  -- chapter_85_line_17: GL tag implication.
  have row_17 : (gl_existence1 N v15 v2 add) := by
    apply row_39
    exact row_19
    exact row_31
  -- chapter_85_line_16: GL tag implication.
  have row_16 : (gl_existence1 N v2 previous add) := by
    apply row_39
    exact row_31
    exact row_37
  -- chapter_85_line_15: GL tag implication.
  have row_15 : (succ v15 v1) := by
    apply row_49
    exact row_37
    exact row_38
    exact row_18
    exact row_25
  -- chapter_85_line_14: GL tag theorem.
  have row_14 := peano_source_041 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_85_line_13: GL tag implication.
  have row_13 : (add v15 v14 v3) := by
    apply row_14
    exact row_34
    exact row_15
    exact row_26
  -- chapter_85_line_12: GL tag expansion.
  have row_12 : (¬ (∀ (v8 : α), ((N v8) → (¬ (add v15 v2 v8))))) := by
    simpa only [gl_existence1] using (row_17)
  have exists_row_12 : ∃ (v8 : α), ((N v8) ∧ (add v15 v2 v8)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_12
      intro v8 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v8, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v8, witness_row_12⟩ := exists_row_12
  -- chapter_85_line_11: GL tag disintegration.
  have row_11 : (add v15 v2 v8) := by
    exact witness_row_12.2
  -- chapter_85_line_10: GL tag theorem.
  have row_10 := peano_source_017 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_85_line_9: GL tag implication.
  have row_9 : (succ v8 v3) := by
    apply row_10
    exact row_13
    exact row_11
    exact row_26
  -- chapter_85_line_8: GL tag expansion.
  have row_8 : (¬ (∀ (v7 : α), ((N v7) → (¬ (add v2 previous v7))))) := by
    simpa only [gl_existence1] using (row_16)
  have exists_row_8 : ∃ (v7 : α), ((N v7) ∧ (add v2 previous v7)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_8
      intro v7 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v7, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v7, witness_row_8⟩ := exists_row_8
  -- chapter_85_line_7: GL tag disintegration.
  have row_7 : (add v2 previous v7) := by
    exact witness_row_8.2
  -- chapter_85_line_6: GL tag recursion.
  have row_6 : (∀ (w1 : α) (w2 : α), ((add w1 v2 w2) → (∀ (w3 : α), ((add v2 previous w3) → ((add v6 previous w1) → (add w3 v6 w2)))))) := by
    exact assumption_6
  -- chapter_85_line_5: GL tag implication.
  have row_5 : (add v7 v6 v8) := by
    apply row_6
    exact row_11
    exact row_7
    exact row_18
  -- chapter_85_line_4: GL tag task formulation.
  have row_4 : (add v2 v4 v5) := by
    exact assumption_4
  -- chapter_85_line_3: GL tag implication.
  have row_3 : (succ v7 v5) := by
    apply row_49
    exact row_37
    exact row_38
    exact row_7
    exact row_4
  -- chapter_85_line_2: GL tag theorem.
  have row_2 := peano_source_040 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_85_line_1: GL tag implication.
  have row_1 : (add v5 v6 v3) := by
    apply row_2
    exact row_5
    exact row_3
    exact row_9
  exact row_1

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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
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
    have typingRule := peano_source_043_induction_typing N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact typingRule v1 v4 v6 premise_3
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v5 : α), ((add v2 zero v5) → (∀ (v6 : α), ((add v6 zero v1) → (add v5 v6 v3))))))) := by
    intro v1
    intro v2
    intro v3
    intro base_premise_1
    intro v5
    intro base_premise_2
    intro v6
    intro base_premise_3
    have zeroRule := peano_source_043_check_zero N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact zeroRule v1 v2 v3 zero v5 v6 base_premise_3 rfl base_premise_1 base_premise_2
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
        (succ induction_n induction_m) := by
      exact induction_successor
    have step_induction_assumption_2 :
        (∀ (w1 : α) (w2 : α), ((add w1 v2 w2) → (∀ (w3 : α), ((add v2 induction_n w3) → ((add v6 induction_n w1) → (add w3 v6 w2)))))) := by
      intro w1
      intro w2
      intro step_induction_assumption_2_premise_1
      intro w3
      intro step_induction_assumption_2_premise_2
      intro step_induction_assumption_2_premise_3
      apply induction_hypothesis
      all_goals assumption
    have stepRule := peano_source_043_check_induction_condition N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact stepRule induction_n v1 v2 v3 induction_m v5 v6 step_induction_assumption_1 step_premise_1 step_premise_3 step_induction_assumption_2 step_premise_2
  have inductionProperty : (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v5 : α), ((add v2 v4 v5) → (∀ (v6 : α), ((add v6 v4 v1) → (add v5 v6 v3))))))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v5 : α), ((add v2 induction_value v5) → (∀ (v6 : α), ((add v6 induction_value v1) → (add v5 v6 v3))))))))
      v4
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v1 v2 v3 premise_1 v5 premise_2 v6 premise_3

private theorem peano_source_051_induction_typing
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (v5 : α)
    (assumption_2 : (mul v2 v1 v5))
    : (N v1) := by
  -- chapter_103_line_10: GL tag task formulation.
  have row_10 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_103_line_9: GL tag expansion.
  have row_9 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_10)
  -- chapter_103_line_8: GL tag disintegration.
  have row_8 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_9.1
  -- chapter_103_line_7: GL tag expansion.
  have row_7 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_8)
  -- chapter_103_line_6: GL tag disintegration.
  have row_6 : (gl_fXYZ mul N N N) := by
    exact row_7.1.1.1.1.2
  -- chapter_103_line_5: GL tag expansion.
  have row_5 : (((((gl_implication11 mul N) ∧ (gl_implication12 mul N)) ∧ (gl_implication13 mul N)) ∧ (gl_implication18 N N N mul)) ∧ (gl_implication19 N N mul)) := by
    simpa only [gl_fXYZ] using (row_6)
  -- chapter_103_line_4: GL tag disintegration.
  have row_4 : (gl_implication12 mul N) := by
    exact row_5.1.1.1.2
  -- chapter_103_line_3: GL tag expansion.
  have row_3 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_4)
  -- chapter_103_line_2: GL tag task formulation.
  have row_2 : (mul v2 v1 v5) := by
    exact assumption_2
  -- chapter_103_line_1: GL tag implication.
  have row_1 : (N v1) := by
    apply row_3
    exact row_2
  exact row_1

private theorem peano_source_051_check_zero
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (v4 : α)
    (v5 : α)
    (v6 : α)
    (assumption_29 : (v1 = zero))
    (assumption_28 : (mul v2 v1 v5))
    (assumption_11 : (mul v1 v2 v3))
    (assumption_2 : (mul v4 v5 v6))
    : (mul v4 v3 v6) := by
  -- chapter_104_line_29: GL tag recursion.
  have row_29 : (v1 = zero) := by
    exact assumption_29
  -- chapter_104_line_28: GL tag task formulation.
  have row_28 : (mul v2 v1 v5) := by
    exact assumption_28
  -- chapter_104_line_27: GL tag equality1.
  have row_27 : (mul v2 zero v5) := by
    have equality_source := row_28
    have equality_step_1 := row_29
    cases equality_step_1
    exact equality_source
  -- chapter_104_line_26: GL tag task formulation.
  have row_26 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_104_line_25: GL tag expansion.
  have row_25 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_26)
  -- chapter_104_line_24: GL tag disintegration.
  have row_24 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_25.1
  -- chapter_104_line_23: GL tag expansion.
  have row_23 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_24)
  -- chapter_104_line_22: GL tag disintegration.
  have row_22 : (gl_fXYZ mul N N N) := by
    exact row_23.1.1.1.1.2
  -- chapter_104_line_21: GL tag expansion.
  have row_21 : (((((gl_implication11 mul N) ∧ (gl_implication12 mul N)) ∧ (gl_implication13 mul N)) ∧ (gl_implication18 N N N mul)) ∧ (gl_implication19 N N mul)) := by
    simpa only [gl_fXYZ] using (row_22)
  -- chapter_104_line_20: GL tag disintegration.
  have row_20 : (gl_implication13 mul N) := by
    exact row_21.1.1.2
  -- chapter_104_line_19: GL tag expansion.
  have row_19 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_20)
  -- chapter_104_line_18: GL tag disintegration.
  have row_18 : (gl_implication12 mul N) := by
    exact row_21.1.1.1.2
  -- chapter_104_line_17: GL tag expansion.
  have row_17 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_18)
  -- chapter_104_line_16: GL tag disintegration.
  have row_16 : (gl_implication11 mul N) := by
    exact row_21.1.1.1.1
  -- chapter_104_line_15: GL tag expansion.
  have row_15 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_16)
  -- chapter_104_line_14: GL tag disintegration.
  have row_14 : (gl_implication25 N zero mul) := by
    exact row_23.1.1.1.2
  -- chapter_104_line_13: GL tag expansion.
  have row_13 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((mul w1 zero w2) → (w2 = zero))))) := by
    simpa only [gl_implication25] using (row_14)
  -- chapter_104_line_12: GL tag implication grouping.
  have row_12 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_15
    have auxiliary_source_2 := row_17
    have auxiliary_source_3 := row_19
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_104_line_11: GL tag task formulation.
  have row_11 : (mul v1 v2 v3) := by
    exact assumption_11
  -- chapter_104_line_10: GL tag implication.
  have row_10 : (gl_and21 v1 N v2 N v3 N) := by
    apply row_12
    exact row_11
  -- chapter_104_line_9: GL tag expansion.
  have row_9 : (((N v1) ∧ (N v2)) ∧ (N v3)) := by
    simpa only [gl_and21] using (row_10)
  -- chapter_104_line_8: GL tag disintegration.
  have row_8 : (N v2) := by
    exact row_9.1.2
  -- chapter_104_line_7: GL tag equality1.
  have row_7 : (mul zero v2 v3) := by
    have equality_source := row_11
    have equality_step_1 := row_29
    cases equality_step_1
    exact equality_source
  -- chapter_104_line_6: GL tag theorem.
  have row_6 := peano_source_048 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_104_line_5: GL tag implication.
  have row_5 : (zero = v3) := by
    apply row_6
    exact row_8
    exact row_7
  -- chapter_104_line_4: GL tag implication.
  have row_4 : (v5 = zero) := by
    apply row_13
    exact row_8
    exact row_27
  -- chapter_104_line_3: GL tag equality2.
  have row_3 : (v5 = v3) := by
    exact Eq.trans row_4 row_5
  -- chapter_104_line_2: GL tag task formulation.
  have row_2 : (mul v4 v5 v6) := by
    exact assumption_2
  -- chapter_104_line_1: GL tag equality1.
  have row_1 : (mul v4 v3 v6) := by
    have equality_source := row_2
    have equality_step_1 := row_3
    cases equality_step_1
    exact equality_source
  exact row_1

private theorem peano_source_051_check_induction_condition
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (v4 : α)
    (v5 : α)
    (v6 : α)
    (assumption_72 : (succ previous v1))
    (assumption_36 : (mul v1 v2 v3))
    (assumption_32 : (mul v4 v5 v6))
    (assumption_13 : (∀ (w1 : α), ((mul previous v2 w1) → (∀ (w2 : α) (w3 : α), ((mul v4 w2 w3) → ((mul v2 previous w2) → (mul v4 w1 w3)))))))
    (assumption_10 : (mul v2 v1 v5))
    : (mul v4 v3 v6) := by
  -- chapter_105_line_73: GL tag task formulation.
  have row_73 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_105_line_72: GL tag recursion.
  have row_72 : (succ previous v1) := by
    exact assumption_72
  -- chapter_105_line_71: GL tag theorem.
  have row_71 := peano_source_027 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_105_line_70: GL tag implication.
  have row_70 : (add previous one v1) := by
    apply row_71
    exact row_72
  -- chapter_105_line_69: GL tag expansion.
  have row_69 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_73)
  -- chapter_105_line_68: GL tag disintegration.
  have row_68 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_69.1
  -- chapter_105_line_67: GL tag expansion.
  have row_67 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_68)
  -- chapter_105_line_66: GL tag disintegration.
  have row_66 : (gl_fXY succ N N) := by
    exact row_67.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_105_line_65: GL tag expansion.
  have row_65 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_66)
  -- chapter_105_line_64: GL tag disintegration.
  have row_64 : (gl_implication0 succ N) := by
    exact row_65.1.1.1
  -- chapter_105_line_63: GL tag expansion.
  have row_63 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_64)
  -- chapter_105_line_62: GL tag disintegration.
  have row_62 : (gl_fXYZ add N N N) := by
    exact row_67.1.1.1.1.1.1.1.1.1.2
  -- chapter_105_line_61: GL tag expansion.
  have row_61 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_62)
  -- chapter_105_line_60: GL tag disintegration.
  have row_60 : (gl_implication18 N N N add) := by
    exact row_61.1.2
  -- chapter_105_line_59: GL tag expansion.
  have row_59 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication18] using (row_60)
  -- chapter_105_line_58: GL tag disintegration.
  have row_58 : (N zero) := by
    exact row_67.1.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_105_line_57: GL tag disintegration.
  have row_57 : (gl_implication23 N succ add) := by
    exact row_67.1.1.1.1.1.1.2
  -- chapter_105_line_56: GL tag expansion.
  have row_56 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication23] using (row_57)
  -- chapter_105_line_55: GL tag disintegration.
  have row_55 : (succ zero one) := by
    exact row_69.2
  -- chapter_105_line_54: GL tag disintegration.
  have row_54 : (gl_fXYZ mul N N N) := by
    exact row_67.1.1.1.1.2
  -- chapter_105_line_53: GL tag expansion.
  have row_53 : (((((gl_implication11 mul N) ∧ (gl_implication12 mul N)) ∧ (gl_implication13 mul N)) ∧ (gl_implication18 N N N mul)) ∧ (gl_implication19 N N mul)) := by
    simpa only [gl_fXYZ] using (row_54)
  -- chapter_105_line_52: GL tag disintegration.
  have row_52 : (gl_implication13 mul N) := by
    exact row_53.1.1.2
  -- chapter_105_line_51: GL tag expansion.
  have row_51 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_52)
  -- chapter_105_line_50: GL tag disintegration.
  have row_50 : (gl_implication12 mul N) := by
    exact row_53.1.1.1.2
  -- chapter_105_line_49: GL tag expansion.
  have row_49 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_50)
  -- chapter_105_line_48: GL tag disintegration.
  have row_48 : (gl_implication11 mul N) := by
    exact row_53.1.1.1.1
  -- chapter_105_line_47: GL tag expansion.
  have row_47 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_48)
  -- chapter_105_line_46: GL tag disintegration.
  have row_46 : (gl_implication18 N N N mul) := by
    exact row_53.1.2
  -- chapter_105_line_45: GL tag expansion.
  have row_45 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication18] using (row_46)
  -- chapter_105_line_44: GL tag disintegration.
  have row_44 : (gl_implication28 N succ mul add) := by
    exact row_67.2
  -- chapter_105_line_43: GL tag expansion.
  have row_43 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((mul w3 w1 w4) → (∀ (w5 : α), ((mul w3 w2 w5) → (add w4 w3 w5))))))))) := by
    simpa only [gl_implication28] using (row_44)
  -- chapter_105_line_42: GL tag implication.
  have row_42 : (N previous) := by
    apply row_63
    exact row_72
  -- chapter_105_line_41: GL tag implication.
  have row_41 : (gl_existence1 N previous zero add) := by
    apply row_59
    exact row_42
    exact row_58
  -- chapter_105_line_40: GL tag expansion.
  have row_40 : (¬ (∀ (v17 : α), ((N v17) → (¬ (add previous zero v17))))) := by
    simpa only [gl_existence1] using (row_41)
  have exists_row_40 : ∃ (v17 : α), ((N v17) ∧ (add previous zero v17)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_40
      intro v17 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v17, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v17, witness_row_40⟩ := exists_row_40
  -- chapter_105_line_39: GL tag disintegration.
  have row_39 : (add previous zero v17) := by
    exact witness_row_40.2
  -- chapter_105_line_38: GL tag implication.
  have row_38 : (succ v17 v1) := by
    apply row_56
    exact row_58
    exact row_55
    exact row_39
    exact row_70
  -- chapter_105_line_37: GL tag implication grouping.
  have row_37 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_47
    have auxiliary_source_2 := row_49
    have auxiliary_source_3 := row_51
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_105_line_36: GL tag task formulation.
  have row_36 : (mul v1 v2 v3) := by
    exact assumption_36
  -- chapter_105_line_35: GL tag implication.
  have row_35 : (gl_and21 v1 N v2 N v3 N) := by
    apply row_37
    exact row_36
  -- chapter_105_line_34: GL tag expansion.
  have row_34 : (((N v1) ∧ (N v2)) ∧ (N v3)) := by
    simpa only [gl_and21] using (row_35)
  -- chapter_105_line_33: GL tag disintegration.
  have row_33 : (N v2) := by
    exact row_34.1.2
  -- chapter_105_line_32: GL tag task formulation.
  have row_32 : (mul v4 v5 v6) := by
    exact assumption_32
  -- chapter_105_line_31: GL tag implication.
  have row_31 : (gl_and21 v4 N v5 N v6 N) := by
    apply row_37
    exact row_32
  -- chapter_105_line_30: GL tag expansion.
  have row_30 : (((N v4) ∧ (N v5)) ∧ (N v6)) := by
    simpa only [gl_and21] using (row_31)
  -- chapter_105_line_29: GL tag disintegration.
  have row_29 : (N v4) := by
    exact row_30.1.1
  -- chapter_105_line_28: GL tag implication.
  have row_28 : (gl_existence1 N previous v2 mul) := by
    apply row_45
    exact row_42
    exact row_33
  -- chapter_105_line_27: GL tag expansion.
  have row_27 : (¬ (∀ (v8 : α), ((N v8) → (¬ (mul previous v2 v8))))) := by
    simpa only [gl_existence1] using (row_28)
  have exists_row_27 : ∃ (v8 : α), ((N v8) ∧ (mul previous v2 v8)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_27
      intro v8 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v8, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v8, witness_row_27⟩ := exists_row_27
  -- chapter_105_line_26: GL tag disintegration.
  have row_26 : (mul previous v2 v8) := by
    exact witness_row_27.2
  -- chapter_105_line_25: GL tag theorem.
  have row_25 := peano_source_011 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_105_line_24: GL tag implication.
  have row_24 : (previous = v17) := by
    apply row_25
    exact row_39
    exact row_42
  -- chapter_105_line_23: GL tag equality1.
  have row_23 : (mul v17 v2 v8) := by
    have equality_source := row_26
    have equality_step_1 := row_24
    cases equality_step_1
    exact equality_source
  -- chapter_105_line_22: GL tag theorem.
  have row_22 := peano_source_028 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_105_line_21: GL tag implication.
  have row_21 : (add v2 v8 v3) := by
    apply row_22
    exact row_36
    exact row_23
    exact row_38
  -- chapter_105_line_20: GL tag implication.
  have row_20 : (gl_existence1 N v2 previous mul) := by
    apply row_45
    exact row_33
    exact row_42
  -- chapter_105_line_19: GL tag expansion.
  have row_19 : (¬ (∀ (v16 : α), ((N v16) → (¬ (mul v2 previous v16))))) := by
    simpa only [gl_existence1] using (row_20)
  have exists_row_19 : ∃ (v16 : α), ((N v16) ∧ (mul v2 previous v16)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_19
      intro v16 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v16, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v16, witness_row_19⟩ := exists_row_19
  -- chapter_105_line_18: GL tag disintegration.
  have row_18 : (N v16) := by
    exact witness_row_19.1
  -- chapter_105_line_17: GL tag disintegration.
  have row_17 : (mul v2 previous v16) := by
    exact witness_row_19.2
  -- chapter_105_line_16: GL tag implication.
  have row_16 : (gl_existence1 N v4 v16 mul) := by
    apply row_45
    exact row_29
    exact row_18
  -- chapter_105_line_15: GL tag expansion.
  have row_15 : (¬ (∀ (v9 : α), ((N v9) → (¬ (mul v4 v16 v9))))) := by
    simpa only [gl_existence1] using (row_16)
  have exists_row_15 : ∃ (v9 : α), ((N v9) ∧ (mul v4 v16 v9)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_15
      intro v9 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v9, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v9, witness_row_15⟩ := exists_row_15
  -- chapter_105_line_14: GL tag disintegration.
  have row_14 : (mul v4 v16 v9) := by
    exact witness_row_15.2
  -- chapter_105_line_13: GL tag recursion.
  have row_13 : (∀ (w1 : α), ((mul previous v2 w1) → (∀ (w2 : α) (w3 : α), ((mul v4 w2 w3) → ((mul v2 previous w2) → (mul v4 w1 w3)))))) := by
    exact assumption_13
  -- chapter_105_line_12: GL tag implication.
  have row_12 : (mul v4 v8 v9) := by
    apply row_13
    exact row_26
    exact row_14
    exact row_17
  -- chapter_105_line_11: GL tag implication.
  have row_11 : (gl_existence1 N v4 v2 mul) := by
    apply row_45
    exact row_29
    exact row_33
  -- chapter_105_line_10: GL tag task formulation.
  have row_10 : (mul v2 v1 v5) := by
    exact assumption_10
  -- chapter_105_line_9: GL tag implication.
  have row_9 : (add v16 v2 v5) := by
    apply row_43
    exact row_42
    exact row_72
    exact row_17
    exact row_10
  -- chapter_105_line_8: GL tag theorem.
  have row_8 := peano_source_036 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_105_line_7: GL tag implication.
  have row_7 : (add v2 v16 v5) := by
    apply row_8
    exact row_9
  -- chapter_105_line_6: GL tag expansion.
  have row_6 : (¬ (∀ (v7 : α), ((N v7) → (¬ (mul v4 v2 v7))))) := by
    simpa only [gl_existence1] using (row_11)
  have exists_row_6 : ∃ (v7 : α), ((N v7) ∧ (mul v4 v2 v7)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_6
      intro v7 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v7, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v7, witness_row_6⟩ := exists_row_6
  -- chapter_105_line_5: GL tag disintegration.
  have row_5 : (mul v4 v2 v7) := by
    exact witness_row_6.2
  -- chapter_105_line_4: GL tag theorem.
  have row_4 := peano_source_047 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_105_line_3: GL tag implication.
  have row_3 : (add v7 v9 v6) := by
    apply row_4
    exact row_5
    exact row_14
    exact row_32
    exact row_7
  -- chapter_105_line_2: GL tag theorem.
  have row_2 := peano_source_050 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_105_line_1: GL tag implication.
  have row_1 : (mul v4 v3 v6) := by
    apply row_2
    exact row_3
    exact row_21
    exact row_5
    exact row_12
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
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
  have inductionMember : N v1 := by
    have typingRule := peano_source_051_induction_typing N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact typingRule v1 v2 v5 premise_3
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v2 : α) (v3 : α), ((mul zero v2 v3) → (∀ (v4 : α) (v5 : α) (v6 : α), ((mul v4 v5 v6) → ((mul v2 zero v5) → (mul v4 v3 v6)))))) := by
    intro v2
    intro v3
    intro base_premise_1
    intro v4
    intro v5
    intro v6
    intro base_premise_2
    intro base_premise_3
    have zeroRule := peano_source_051_check_zero N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact zeroRule zero v2 v3 v4 v5 v6 rfl base_premise_3 base_premise_1 base_premise_2
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v2 : α) (v3 : α), ((mul induction_n v2 v3) → (∀ (v4 : α) (v5 : α) (v6 : α), ((mul v4 v5 v6) → ((mul v2 induction_n v5) → (mul v4 v3 v6)))))) → ∀ induction_m, succ induction_n induction_m → (∀ (v2 : α) (v3 : α), ((mul induction_m v2 v3) → (∀ (v4 : α) (v5 : α) (v6 : α), ((mul v4 v5 v6) → ((mul v2 induction_m v5) → (mul v4 v3 v6)))))) := by
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
        (succ induction_n induction_m) := by
      exact induction_successor
    have step_induction_assumption_2 :
        (∀ (w1 : α), ((mul induction_n v2 w1) → (∀ (w2 : α) (w3 : α), ((mul v4 w2 w3) → ((mul v2 induction_n w2) → (mul v4 w1 w3)))))) := by
      intro w1
      intro step_induction_assumption_2_premise_1
      intro w2
      intro w3
      intro step_induction_assumption_2_premise_2
      intro step_induction_assumption_2_premise_3
      apply induction_hypothesis
      all_goals assumption
    have stepRule := peano_source_051_check_induction_condition N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact stepRule induction_n induction_m v2 v3 v4 v5 v6 step_induction_assumption_1 step_premise_1 step_premise_2 step_induction_assumption_2 step_premise_3
  have inductionProperty : (∀ (v2 : α) (v3 : α), ((mul v1 v2 v3) → (∀ (v4 : α) (v5 : α) (v6 : α), ((mul v4 v5 v6) → ((mul v2 v1 v5) → (mul v4 v3 v6)))))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v2 : α) (v3 : α), ((mul induction_value v2 v3) → (∀ (v4 : α) (v5 : α) (v6 : α), ((mul v4 v5 v6) → ((mul v2 induction_value v5) → (mul v4 v3 v6)))))))
      v1
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v2 v3 premise_1 v4 v5 v6 premise_2 premise_3

private theorem peano_source_052_induction_typing
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v2 : α)
    (v4 : α)
    (assumption_2 : (succ v2 v4))
    : (N v2) := by
  -- chapter_106_line_10: GL tag task formulation.
  have row_10 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_106_line_9: GL tag expansion.
  have row_9 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_10)
  -- chapter_106_line_8: GL tag disintegration.
  have row_8 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_9.1
  -- chapter_106_line_7: GL tag expansion.
  have row_7 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_8)
  -- chapter_106_line_6: GL tag disintegration.
  have row_6 : (gl_fXY succ N N) := by
    exact row_7.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_106_line_5: GL tag expansion.
  have row_5 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_6)
  -- chapter_106_line_4: GL tag disintegration.
  have row_4 : (gl_implication0 succ N) := by
    exact row_5.1.1.1
  -- chapter_106_line_3: GL tag expansion.
  have row_3 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_4)
  -- chapter_106_line_2: GL tag task formulation.
  have row_2 : (succ v2 v4) := by
    exact assumption_2
  -- chapter_106_line_1: GL tag implication.
  have row_1 : (N v2) := by
    apply row_3
    exact row_2
  exact row_1

private theorem peano_source_052_check_zero
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (v4 : α)
    (v5 : α)
    (assumption_50 : (succ v2 v4))
    (assumption_20 : (mul v4 v1 v5))
    (assumption_13 : (v2 = zero))
    (assumption_12 : (mul v1 v2 v3))
    : (add v1 v3 v5) := by
  -- chapter_107_line_50: GL tag task formulation.
  have row_50 : (succ v2 v4) := by
    exact assumption_50
  -- chapter_107_line_49: GL tag task formulation.
  have row_49 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_107_line_48: GL tag theorem.
  have row_48 := peano_source_010 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_107_line_47: GL tag implication.
  have row_47 : (add one v2 v4) := by
    apply row_48
    exact row_50
  -- chapter_107_line_46: GL tag expansion.
  have row_46 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_49)
  -- chapter_107_line_45: GL tag disintegration.
  have row_45 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_46.1
  -- chapter_107_line_44: GL tag expansion.
  have row_44 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_45)
  -- chapter_107_line_43: GL tag disintegration.
  have row_43 : (gl_fXYZ add N N N) := by
    exact row_44.1.1.1.1.1.1.1.1.1.2
  -- chapter_107_line_42: GL tag expansion.
  have row_42 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_43)
  -- chapter_107_line_41: GL tag disintegration.
  have row_41 : (gl_implication13 add N) := by
    exact row_42.1.1.2
  -- chapter_107_line_40: GL tag expansion.
  have row_40 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_41)
  -- chapter_107_line_39: GL tag disintegration.
  have row_39 : (gl_implication12 add N) := by
    exact row_42.1.1.1.2
  -- chapter_107_line_38: GL tag expansion.
  have row_38 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_39)
  -- chapter_107_line_37: GL tag disintegration.
  have row_37 : (gl_implication11 add N) := by
    exact row_42.1.1.1.1
  -- chapter_107_line_36: GL tag expansion.
  have row_36 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_37)
  -- chapter_107_line_35: GL tag disintegration.
  have row_35 : (gl_fXYZ mul N N N) := by
    exact row_44.1.1.1.1.2
  -- chapter_107_line_34: GL tag expansion.
  have row_34 : (((((gl_implication11 mul N) ∧ (gl_implication12 mul N)) ∧ (gl_implication13 mul N)) ∧ (gl_implication18 N N N mul)) ∧ (gl_implication19 N N mul)) := by
    simpa only [gl_fXYZ] using (row_35)
  -- chapter_107_line_33: GL tag disintegration.
  have row_33 : (gl_implication13 mul N) := by
    exact row_34.1.1.2
  -- chapter_107_line_32: GL tag expansion.
  have row_32 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_33)
  -- chapter_107_line_31: GL tag disintegration.
  have row_31 : (gl_implication12 mul N) := by
    exact row_34.1.1.1.2
  -- chapter_107_line_30: GL tag expansion.
  have row_30 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_31)
  -- chapter_107_line_29: GL tag disintegration.
  have row_29 : (gl_implication11 mul N) := by
    exact row_34.1.1.1.1
  -- chapter_107_line_28: GL tag expansion.
  have row_28 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_29)
  -- chapter_107_line_27: GL tag disintegration.
  have row_27 : (gl_implication18 N N N mul) := by
    exact row_34.1.2
  -- chapter_107_line_26: GL tag expansion.
  have row_26 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication18] using (row_27)
  -- chapter_107_line_25: GL tag implication grouping.
  have row_25 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_36
    have auxiliary_source_2 := row_38
    have auxiliary_source_3 := row_40
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_107_line_24: GL tag implication.
  have row_24 : (gl_and21 one N v2 N v4 N) := by
    apply row_25
    exact row_47
  -- chapter_107_line_23: GL tag expansion.
  have row_23 : (((N one) ∧ (N v2)) ∧ (N v4)) := by
    simpa only [gl_and21] using (row_24)
  -- chapter_107_line_22: GL tag disintegration.
  have row_22 : (N v2) := by
    exact row_23.1.2
  -- chapter_107_line_21: GL tag implication grouping.
  have row_21 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_28
    have auxiliary_source_2 := row_30
    have auxiliary_source_3 := row_32
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_107_line_20: GL tag task formulation.
  have row_20 : (mul v4 v1 v5) := by
    exact assumption_20
  -- chapter_107_line_19: GL tag implication.
  have row_19 : (gl_and21 v4 N v1 N v5 N) := by
    apply row_21
    exact row_20
  -- chapter_107_line_18: GL tag expansion.
  have row_18 : (((N v4) ∧ (N v1)) ∧ (N v5)) := by
    simpa only [gl_and21] using (row_19)
  -- chapter_107_line_17: GL tag disintegration.
  have row_17 : (N v1) := by
    exact row_18.1.2
  -- chapter_107_line_16: GL tag implication.
  have row_16 : (gl_existence1 N v2 v1 mul) := by
    apply row_26
    exact row_22
    exact row_17
  -- chapter_107_line_15: GL tag expansion.
  have row_15 : (¬ (∀ (v6 : α), ((N v6) → (¬ (mul v2 v1 v6))))) := by
    simpa only [gl_existence1] using (row_16)
  have exists_row_15 : ∃ (v6 : α), ((N v6) ∧ (mul v2 v1 v6)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_15
      intro v6 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v6, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v6, witness_row_15⟩ := exists_row_15
  -- chapter_107_line_14: GL tag disintegration.
  have row_14 : (mul v2 v1 v6) := by
    exact witness_row_15.2
  -- chapter_107_line_13: GL tag recursion.
  have row_13 : (v2 = zero) := by
    exact assumption_13
  -- chapter_107_line_12: GL tag task formulation.
  have row_12 : (mul v1 v2 v3) := by
    exact assumption_12
  -- chapter_107_line_11: GL tag equality1.
  have row_11 : (mul v1 zero v3) := by
    have equality_source := row_12
    have equality_step_1 := row_13
    cases equality_step_1
    exact equality_source
  -- chapter_107_line_10: GL tag theorem.
  have row_10 := peano_source_025 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_107_line_9: GL tag implication.
  have row_9 : (zero = v3) := by
    apply row_10
    exact row_11
    exact row_17
  -- chapter_107_line_8: GL tag equality1.
  have row_8 : (mul zero v1 v6) := by
    have equality_source := row_14
    have equality_step_1 := row_13
    cases equality_step_1
    exact equality_source
  -- chapter_107_line_7: GL tag theorem.
  have row_7 := peano_source_048 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_107_line_6: GL tag implication.
  have row_6 : (zero = v6) := by
    apply row_7
    exact row_17
    exact row_8
  -- chapter_107_line_5: GL tag symmetry of equality.
  have row_5 : (v6 = zero) := by
    exact Eq.symm row_6
  -- chapter_107_line_4: GL tag equality2.
  have row_4 : (v6 = v3) := by
    exact Eq.trans row_5 row_9
  -- chapter_107_line_3: GL tag theorem.
  have row_3 := peano_source_028 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_107_line_2: GL tag implication.
  have row_2 : (add v1 v6 v5) := by
    apply row_3
    exact row_20
    exact row_14
    exact row_50
  -- chapter_107_line_1: GL tag equality1.
  have row_1 : (add v1 v3 v5) := by
    have equality_source := row_2
    have equality_step_1 := row_4
    cases equality_step_1
    exact equality_source
  exact row_1

private theorem peano_source_052_check_induction_condition
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (v4 : α)
    (v5 : α)
    (assumption_30 : (mul v4 v1 v5))
    (assumption_26 : (succ previous v2))
    (assumption_20 : (mul v1 v2 v3))
    (assumption_12 : (∀ (w1 : α), ((mul v1 previous w1) → (∀ (w2 : α) (w3 : α), ((mul w2 v1 w3) → ((succ previous w2) → (add v1 w1 w3)))))))
    (assumption_6 : (succ v2 v4))
    : (add v1 v3 v5) := by
  -- chapter_108_line_61: GL tag task formulation.
  have row_61 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_108_line_60: GL tag expansion.
  have row_60 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_61)
  -- chapter_108_line_59: GL tag disintegration.
  have row_59 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_60.1
  -- chapter_108_line_58: GL tag expansion.
  have row_58 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_59)
  -- chapter_108_line_57: GL tag disintegration.
  have row_57 : (gl_fXYZ add N N N) := by
    exact row_58.1.1.1.1.1.1.1.1.1.2
  -- chapter_108_line_56: GL tag expansion.
  have row_56 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_57)
  -- chapter_108_line_55: GL tag disintegration.
  have row_55 : (gl_implication19 N N add) := by
    exact row_56.2
  -- chapter_108_line_54: GL tag expansion.
  have row_54 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (∀ (w3 : α), ((add w1 w2 w3) → (∀ (w4 : α), ((add w1 w2 w4) → (w3 = w4))))))))) := by
    simpa only [gl_implication19] using (row_55)
  -- chapter_108_line_53: GL tag disintegration.
  have row_53 : (gl_fXYZ mul N N N) := by
    exact row_58.1.1.1.1.2
  -- chapter_108_line_52: GL tag expansion.
  have row_52 : (((((gl_implication11 mul N) ∧ (gl_implication12 mul N)) ∧ (gl_implication13 mul N)) ∧ (gl_implication18 N N N mul)) ∧ (gl_implication19 N N mul)) := by
    simpa only [gl_fXYZ] using (row_53)
  -- chapter_108_line_51: GL tag disintegration.
  have row_51 : (gl_implication13 mul N) := by
    exact row_52.1.1.2
  -- chapter_108_line_50: GL tag expansion.
  have row_50 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_51)
  -- chapter_108_line_49: GL tag disintegration.
  have row_49 : (gl_implication12 mul N) := by
    exact row_52.1.1.1.2
  -- chapter_108_line_48: GL tag expansion.
  have row_48 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_49)
  -- chapter_108_line_47: GL tag disintegration.
  have row_47 : (gl_implication11 mul N) := by
    exact row_52.1.1.1.1
  -- chapter_108_line_46: GL tag expansion.
  have row_46 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_47)
  -- chapter_108_line_45: GL tag disintegration.
  have row_45 : (gl_implication18 N N N mul) := by
    exact row_52.1.2
  -- chapter_108_line_44: GL tag expansion.
  have row_44 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication18] using (row_45)
  -- chapter_108_line_43: GL tag disintegration.
  have row_43 : (gl_fXY succ N N) := by
    exact row_58.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_108_line_42: GL tag expansion.
  have row_42 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_43)
  -- chapter_108_line_41: GL tag disintegration.
  have row_41 : (gl_implication0 succ N) := by
    exact row_42.1.1.1
  -- chapter_108_line_40: GL tag expansion.
  have row_40 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_41)
  -- chapter_108_line_39: GL tag disintegration.
  have row_39 : (gl_implication28 N succ mul add) := by
    exact row_58.2
  -- chapter_108_line_38: GL tag expansion.
  have row_38 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((mul w3 w1 w4) → (∀ (w5 : α), ((mul w3 w2 w5) → (add w4 w3 w5))))))))) := by
    simpa only [gl_implication28] using (row_39)
  -- chapter_108_line_37: GL tag disintegration.
  have row_37 : (gl_implication1 succ N) := by
    exact row_42.1.1.2
  -- chapter_108_line_36: GL tag expansion.
  have row_36 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using (row_37)
  -- chapter_108_line_35: GL tag disintegration.
  have row_35 : (gl_implication6 N N succ) := by
    exact row_42.1.2
  -- chapter_108_line_34: GL tag expansion.
  have row_34 : (∀ (w1 : α), ((N w1) → (gl_existence0 N w1 succ))) := by
    simpa only [gl_implication6] using (row_35)
  -- chapter_108_line_33: GL tag disintegration.
  have row_33 : (gl_implication7 N succ) := by
    exact row_42.2
  -- chapter_108_line_32: GL tag expansion.
  have row_32 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α), ((succ w1 w3) → (w2 = w3))))))) := by
    simpa only [gl_implication7] using (row_33)
  -- chapter_108_line_31: GL tag implication grouping.
  have row_31 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_46
    have auxiliary_source_2 := row_48
    have auxiliary_source_3 := row_50
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_108_line_30: GL tag task formulation.
  have row_30 : (mul v4 v1 v5) := by
    exact assumption_30
  -- chapter_108_line_29: GL tag implication.
  have row_29 : (gl_and21 v4 N v1 N v5 N) := by
    apply row_31
    exact row_30
  -- chapter_108_line_28: GL tag expansion.
  have row_28 : (((N v4) ∧ (N v1)) ∧ (N v5)) := by
    simpa only [gl_and21] using (row_29)
  -- chapter_108_line_27: GL tag disintegration.
  have row_27 : (N v1) := by
    exact row_28.1.2
  -- chapter_108_line_26: GL tag recursion.
  have row_26 : (succ previous v2) := by
    exact assumption_26
  -- chapter_108_line_25: GL tag implication.
  have row_25 : (N previous) := by
    apply row_40
    exact row_26
  -- chapter_108_line_24: GL tag implication.
  have row_24 : (gl_existence1 N v1 previous mul) := by
    apply row_44
    exact row_27
    exact row_25
  -- chapter_108_line_23: GL tag expansion.
  have row_23 : (¬ (∀ (v12 : α), ((N v12) → (¬ (mul v1 previous v12))))) := by
    simpa only [gl_existence1] using (row_24)
  have exists_row_23 : ∃ (v12 : α), ((N v12) ∧ (mul v1 previous v12)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_23
      intro v12 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v12, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v12, witness_row_23⟩ := exists_row_23
  -- chapter_108_line_22: GL tag disintegration.
  have row_22 : (mul v1 previous v12) := by
    exact witness_row_23.2
  -- chapter_108_line_21: GL tag disintegration.
  have row_21 : (N v12) := by
    exact witness_row_23.1
  -- chapter_108_line_20: GL tag task formulation.
  have row_20 : (mul v1 v2 v3) := by
    exact assumption_20
  -- chapter_108_line_19: GL tag implication.
  have row_19 : (add v12 v1 v3) := by
    apply row_38
    exact row_25
    exact row_26
    exact row_22
    exact row_20
  -- chapter_108_line_18: GL tag theorem.
  have row_18 := peano_source_036 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_108_line_17: GL tag implication.
  have row_17 : (add v1 v12 v3) := by
    apply row_18
    exact row_19
  -- chapter_108_line_16: GL tag implication.
  have row_16 : (N v2) := by
    apply row_36
    exact row_26
  -- chapter_108_line_15: GL tag implication.
  have row_15 : (gl_existence1 N v2 v1 mul) := by
    apply row_44
    exact row_16
    exact row_27
  -- chapter_108_line_14: GL tag expansion.
  have row_14 : (¬ (∀ (v6 : α), ((N v6) → (¬ (mul v2 v1 v6))))) := by
    simpa only [gl_existence1] using (row_15)
  have exists_row_14 : ∃ (v6 : α), ((N v6) ∧ (mul v2 v1 v6)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_14
      intro v6 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v6, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v6, witness_row_14⟩ := exists_row_14
  -- chapter_108_line_13: GL tag disintegration.
  have row_13 : (mul v2 v1 v6) := by
    exact witness_row_14.2
  -- chapter_108_line_12: GL tag recursion.
  have row_12 : (∀ (w1 : α), ((mul v1 previous w1) → (∀ (w2 : α) (w3 : α), ((mul w2 v1 w3) → ((succ previous w2) → (add v1 w1 w3)))))) := by
    exact assumption_12
  -- chapter_108_line_11: GL tag implication.
  have row_11 : (add v1 v12 v6) := by
    apply row_12
    exact row_22
    exact row_13
    exact row_26
  -- chapter_108_line_10: GL tag implication.
  have row_10 : (v6 = v3) := by
    apply row_54
    exact row_27
    exact row_21
    exact row_11
    exact row_17
  -- chapter_108_line_9: GL tag implication.
  have row_9 : (gl_existence0 N v2 succ) := by
    apply row_34
    exact row_16
  -- chapter_108_line_8: GL tag expansion.
  have row_8 : (¬ (∀ (v7 : α), ((N v7) → (¬ (succ v2 v7))))) := by
    simpa only [gl_existence0] using (row_9)
  have exists_row_8 : ∃ (v7 : α), ((N v7) ∧ (succ v2 v7)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_8
      intro v7 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v7, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v7, witness_row_8⟩ := exists_row_8
  -- chapter_108_line_7: GL tag disintegration.
  have row_7 : (succ v2 v7) := by
    exact witness_row_8.2
  -- chapter_108_line_6: GL tag task formulation.
  have row_6 : (succ v2 v4) := by
    exact assumption_6
  -- chapter_108_line_5: GL tag implication.
  have row_5 : (v4 = v7) := by
    apply row_32
    exact row_16
    exact row_6
    exact row_7
  -- chapter_108_line_4: GL tag equality1.
  have row_4 : (mul v7 v1 v5) := by
    have equality_source := row_30
    have equality_step_1 := row_5
    cases equality_step_1
    exact equality_source
  -- chapter_108_line_3: GL tag theorem.
  have row_3 := peano_source_028 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_108_line_2: GL tag implication.
  have row_2 : (add v1 v6 v5) := by
    apply row_3
    exact row_4
    exact row_13
    exact row_7
  -- chapter_108_line_1: GL tag equality1.
  have row_1 : (add v1 v3 v5) := by
    have equality_source := row_2
    have equality_step_1 := row_10
    cases equality_step_1
    exact equality_source
  exact row_1

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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α) (v3 : α), ((mul v1 v2 v3) → (∀ (v4 : α) (v5 : α), ((mul v4 v1 v5) → ((succ v2 v4) → (add v1 v3 v5)))))) := by
  intro v1
  intro v2
  intro v3
  intro premise_1
  intro v4
  intro v5
  intro premise_2
  intro premise_3
  have inductionMember : N v2 := by
    have typingRule := peano_source_052_induction_typing N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact typingRule v2 v4 premise_3
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v1 : α) (v3 : α), ((mul v1 zero v3) → (∀ (v4 : α) (v5 : α), ((mul v4 v1 v5) → ((succ zero v4) → (add v1 v3 v5)))))) := by
    intro v1
    intro v3
    intro base_premise_1
    intro v4
    intro v5
    intro base_premise_2
    intro base_premise_3
    have zeroRule := peano_source_052_check_zero N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact zeroRule v1 zero v3 v4 v5 base_premise_3 base_premise_2 rfl base_premise_1
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v1 : α) (v3 : α), ((mul v1 induction_n v3) → (∀ (v4 : α) (v5 : α), ((mul v4 v1 v5) → ((succ induction_n v4) → (add v1 v3 v5)))))) → ∀ induction_m, succ induction_n induction_m → (∀ (v1 : α) (v3 : α), ((mul v1 induction_m v3) → (∀ (v4 : α) (v5 : α), ((mul v4 v1 v5) → ((succ induction_m v4) → (add v1 v3 v5)))))) := by
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
        (succ induction_n induction_m) := by
      exact induction_successor
    have step_induction_assumption_2 :
        (∀ (w1 : α), ((mul v1 induction_n w1) → (∀ (w2 : α) (w3 : α), ((mul w2 v1 w3) → ((succ induction_n w2) → (add v1 w1 w3)))))) := by
      intro w1
      intro step_induction_assumption_2_premise_1
      intro w2
      intro w3
      intro step_induction_assumption_2_premise_2
      intro step_induction_assumption_2_premise_3
      apply induction_hypothesis
      all_goals assumption
    have stepRule := peano_source_052_check_induction_condition N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact stepRule induction_n v1 induction_m v3 v4 v5 step_premise_2 step_induction_assumption_1 step_premise_1 step_induction_assumption_2 step_premise_3
  have inductionProperty : (∀ (v1 : α) (v3 : α), ((mul v1 v2 v3) → (∀ (v4 : α) (v5 : α), ((mul v4 v1 v5) → ((succ v2 v4) → (add v1 v3 v5)))))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v1 : α) (v3 : α), ((mul v1 induction_value v3) → (∀ (v4 : α) (v5 : α), ((mul v4 v1 v5) → ((succ induction_value v4) → (add v1 v3 v5)))))))
      v2
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v1 v3 premise_1 v4 v5 premise_2 premise_3

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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α) (v5 : α), ((succ v4 v5) → ((mul v1 v4 v2) → (mul v1 v5 v3)))))) := by
  intro v1
  intro v2
  intro v3
  intro premise_1
  intro v4
  intro v5
  intro premise_2
  intro premise_3
  -- chapter_109_line_8: GL tag task formulation.
  have row_8 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_109_line_7: GL tag task formulation.
  have row_7 : (add v1 v2 v3) := by
    exact premise_1
  -- chapter_109_line_6: GL tag theorem.
  have row_6 := peano_source_036 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_109_line_5: GL tag implication.
  have row_5 : (add v2 v1 v3) := by
    apply row_6
    exact row_7
  -- chapter_109_line_4: GL tag task formulation.
  have row_4 : (mul v1 v4 v2) := by
    exact premise_3
  -- chapter_109_line_3: GL tag task formulation.
  have row_3 : (succ v4 v5) := by
    exact premise_2
  -- chapter_109_line_2: GL tag theorem.
  have row_2 := peano_source_024 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_109_line_1: GL tag implication.
  have row_1 : (mul v1 v5 v3) := by
    apply row_2
    exact row_5
    exact row_3
    exact row_4
  exact row_1

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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v3 : α)
    (v4 : α)
    (v7 : α)
    (assumption_2 : (mul v3 v4 v7))
    : (N v4) := by
  -- chapter_113_line_10: GL tag task formulation.
  have row_10 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_113_line_9: GL tag expansion.
  have row_9 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_10)
  -- chapter_113_line_8: GL tag disintegration.
  have row_8 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_9.1
  -- chapter_113_line_7: GL tag expansion.
  have row_7 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_8)
  -- chapter_113_line_6: GL tag disintegration.
  have row_6 : (gl_fXYZ mul N N N) := by
    exact row_7.1.1.1.1.2
  -- chapter_113_line_5: GL tag expansion.
  have row_5 : (((((gl_implication11 mul N) ∧ (gl_implication12 mul N)) ∧ (gl_implication13 mul N)) ∧ (gl_implication18 N N N mul)) ∧ (gl_implication19 N N mul)) := by
    simpa only [gl_fXYZ] using (row_6)
  -- chapter_113_line_4: GL tag disintegration.
  have row_4 : (gl_implication12 mul N) := by
    exact row_5.1.1.1.2
  -- chapter_113_line_3: GL tag expansion.
  have row_3 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_4)
  -- chapter_113_line_2: GL tag task formulation.
  have row_2 : (mul v3 v4 v7) := by
    exact assumption_2
  -- chapter_113_line_1: GL tag implication.
  have row_1 : (N v4) := by
    apply row_3
    exact row_2
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (v4 : α)
    (v5 : α)
    (v6 : α)
    (v7 : α)
    (assumption_66 : (v4 = zero))
    (assumption_34 : (add v1 v2 v3))
    (assumption_30 : (mul v2 v4 v6))
    (assumption_24 : (mul v1 v4 v5))
    (assumption_12 : (mul v3 v4 v7))
    : (add v5 v6 v7) := by
  -- chapter_114_line_66: GL tag recursion.
  have row_66 : (v4 = zero) := by
    exact assumption_66
  -- chapter_114_line_65: GL tag symmetry of equality.
  have row_65 : (zero = v4) := by
    exact Eq.symm row_66
  -- chapter_114_line_64: GL tag task formulation.
  have row_64 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_114_line_63: GL tag expansion.
  have row_63 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_64)
  -- chapter_114_line_62: GL tag disintegration.
  have row_62 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_63.1
  -- chapter_114_line_61: GL tag expansion.
  have row_61 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_62)
  -- chapter_114_line_60: GL tag disintegration.
  have row_60 : (gl_implication25 N zero mul) := by
    exact row_61.1.1.1.2
  -- chapter_114_line_59: GL tag disintegration.
  have row_59 : (gl_fXYZ add N N N) := by
    exact row_61.1.1.1.1.1.1.1.1.1.2
  -- chapter_114_line_58: GL tag expansion.
  have row_58 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_59)
  -- chapter_114_line_57: GL tag disintegration.
  have row_57 : (gl_implication19 N N add) := by
    exact row_58.2
  -- chapter_114_line_56: GL tag expansion.
  have row_56 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (∀ (w3 : α), ((add w1 w2 w3) → (∀ (w4 : α), ((add w1 w2 w4) → (w3 = w4))))))))) := by
    simpa only [gl_implication19] using (row_57)
  -- chapter_114_line_55: GL tag disintegration.
  have row_55 : (gl_implication13 add N) := by
    exact row_58.1.1.2
  -- chapter_114_line_54: GL tag expansion.
  have row_54 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_55)
  -- chapter_114_line_53: GL tag disintegration.
  have row_53 : (gl_implication12 add N) := by
    exact row_58.1.1.1.2
  -- chapter_114_line_52: GL tag expansion.
  have row_52 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_53)
  -- chapter_114_line_51: GL tag disintegration.
  have row_51 : (gl_implication11 add N) := by
    exact row_58.1.1.1.1
  -- chapter_114_line_50: GL tag expansion.
  have row_50 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_51)
  -- chapter_114_line_49: GL tag disintegration.
  have row_49 : (gl_fXYZ mul N N N) := by
    exact row_61.1.1.1.1.2
  -- chapter_114_line_48: GL tag expansion.
  have row_48 : (((((gl_implication11 mul N) ∧ (gl_implication12 mul N)) ∧ (gl_implication13 mul N)) ∧ (gl_implication18 N N N mul)) ∧ (gl_implication19 N N mul)) := by
    simpa only [gl_fXYZ] using (row_49)
  -- chapter_114_line_47: GL tag disintegration.
  have row_47 : (gl_implication13 mul N) := by
    exact row_48.1.1.2
  -- chapter_114_line_46: GL tag expansion.
  have row_46 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_47)
  -- chapter_114_line_45: GL tag disintegration.
  have row_45 : (gl_implication12 mul N) := by
    exact row_48.1.1.1.2
  -- chapter_114_line_44: GL tag expansion.
  have row_44 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_45)
  -- chapter_114_line_43: GL tag disintegration.
  have row_43 : (gl_implication11 mul N) := by
    exact row_48.1.1.1.1
  -- chapter_114_line_42: GL tag expansion.
  have row_42 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_43)
  -- chapter_114_line_41: GL tag disintegration.
  have row_41 : (gl_implication18 N N N add) := by
    exact row_58.1.2
  -- chapter_114_line_40: GL tag expansion.
  have row_40 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication18] using (row_41)
  -- chapter_114_line_39: GL tag disintegration.
  have row_39 : (gl_implication22 N zero add) := by
    exact row_61.1.1.1.1.1.1.1.2
  -- chapter_114_line_38: GL tag disintegration.
  have row_38 : (N zero) := by
    exact row_61.1.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_114_line_37: GL tag equality1.
  have row_37 : (gl_implication25 N v4 mul) := by
    have equality_source := row_60
    have equality_step_1 := row_65
    cases equality_step_1
    exact equality_source
  -- chapter_114_line_36: GL tag expansion.
  have row_36 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((mul w1 v4 w2) → (w2 = v4))))) := by
    simpa only [gl_implication25] using (row_37)
  -- chapter_114_line_35: GL tag implication grouping.
  have row_35 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_50
    have auxiliary_source_2 := row_52
    have auxiliary_source_3 := row_54
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_114_line_34: GL tag task formulation.
  have row_34 : (add v1 v2 v3) := by
    exact assumption_34
  -- chapter_114_line_33: GL tag implication.
  have row_33 : (gl_and21 v1 N v2 N v3 N) := by
    apply row_35
    exact row_34
  -- chapter_114_line_32: GL tag expansion.
  have row_32 : (((N v1) ∧ (N v2)) ∧ (N v3)) := by
    simpa only [gl_and21] using (row_33)
  -- chapter_114_line_31: GL tag disintegration.
  have row_31 : (N v2) := by
    exact row_32.1.2
  -- chapter_114_line_30: GL tag task formulation.
  have row_30 : (mul v2 v4 v6) := by
    exact assumption_30
  -- chapter_114_line_29: GL tag equality1.
  have row_29 : (mul v2 zero v6) := by
    have equality_source := row_30
    have equality_step_1 := row_66
    cases equality_step_1
    exact equality_source
  -- chapter_114_line_28: GL tag theorem.
  have row_28 := peano_source_025 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_114_line_27: GL tag implication.
  have row_27 : (zero = v6) := by
    apply row_28
    exact row_29
    exact row_31
  -- chapter_114_line_26: GL tag equality2.
  have row_26 : (v4 = v6) := by
    exact Eq.trans row_66 row_27
  -- chapter_114_line_25: GL tag implication grouping.
  have row_25 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_42
    have auxiliary_source_2 := row_44
    have auxiliary_source_3 := row_46
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_114_line_24: GL tag task formulation.
  have row_24 : (mul v1 v4 v5) := by
    exact assumption_24
  -- chapter_114_line_23: GL tag implication.
  have row_23 : (gl_and21 v1 N v4 N v5 N) := by
    apply row_25
    exact row_24
  -- chapter_114_line_22: GL tag expansion.
  have row_22 : (((N v1) ∧ (N v4)) ∧ (N v5)) := by
    simpa only [gl_and21] using (row_23)
  -- chapter_114_line_21: GL tag disintegration.
  have row_21 : (N v1) := by
    exact row_22.1.1
  -- chapter_114_line_20: GL tag disintegration.
  have row_20 : (N v4) := by
    exact row_22.1.2
  -- chapter_114_line_19: GL tag implication.
  have row_19 : (gl_existence1 N v2 v1 add) := by
    apply row_40
    exact row_31
    exact row_21
  -- chapter_114_line_18: GL tag expansion.
  have row_18 : (¬ (∀ (v11 : α), ((N v11) → (¬ (add v2 v1 v11))))) := by
    simpa only [gl_existence1] using (row_19)
  have exists_row_18 : ∃ (v11 : α), ((N v11) ∧ (add v2 v1 v11)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_18
      intro v11 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v11, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v11, witness_row_18⟩ := exists_row_18
  -- chapter_114_line_17: GL tag disintegration.
  have row_17 : (add v2 v1 v11) := by
    exact witness_row_18.2
  -- chapter_114_line_16: GL tag disintegration.
  have row_16 : (N v11) := by
    exact witness_row_18.1
  -- chapter_114_line_15: GL tag theorem.
  have row_15 := peano_source_036 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_114_line_14: GL tag implication.
  have row_14 : (add v1 v2 v11) := by
    apply row_15
    exact row_17
  -- chapter_114_line_13: GL tag implication.
  have row_13 : (v3 = v11) := by
    apply row_56
    exact row_21
    exact row_31
    exact row_34
    exact row_14
  -- chapter_114_line_12: GL tag task formulation.
  have row_12 : (mul v3 v4 v7) := by
    exact assumption_12
  -- chapter_114_line_11: GL tag equality1.
  have row_11 : (mul v11 v4 v7) := by
    have equality_source := row_12
    have equality_step_1 := row_13
    cases equality_step_1
    exact equality_source
  -- chapter_114_line_10: GL tag implication.
  have row_10 : (v7 = v4) := by
    apply row_36
    exact row_16
    exact row_11
  -- chapter_114_line_9: GL tag equality1.
  have row_9 : (mul v1 zero v5) := by
    have equality_source := row_24
    have equality_step_1 := row_66
    cases equality_step_1
    exact equality_source
  -- chapter_114_line_8: GL tag implication.
  have row_8 : (zero = v5) := by
    apply row_28
    exact row_9
    exact row_21
  -- chapter_114_line_7: GL tag equality1.
  have row_7 : (gl_implication22 N v4 add) := by
    have equality_source := row_39
    have equality_step_1 := row_65
    cases equality_step_1
    exact equality_source
  -- chapter_114_line_6: GL tag expansion.
  have row_6 : (∀ (w1 : α) (w2 : α), ((w1 = w2) → ((N w1) → ((N w2) → (add w1 v4 w2))))) := by
    simpa only [gl_implication22] using (row_7)
  -- chapter_114_line_5: GL tag implication.
  have row_5 : (add v4 v4 zero) := by
    apply row_6
    exact row_66
    exact row_20
    exact row_38
  -- chapter_114_line_4: GL tag equality1.
  have row_4 : (add v4 v4 v4) := by
    have equality_source := row_5
    have equality_step_1 := row_65
    cases equality_step_1
    exact equality_source
  -- chapter_114_line_3: GL tag equality2.
  have row_3 : (v4 = v5) := by
    exact Eq.trans row_66 row_8
  -- chapter_114_line_2: GL tag symmetry of equality.
  have row_2 : (v4 = v7) := by
    exact Eq.symm row_10
  -- chapter_114_line_1: GL tag equality1.
  have row_1 : (add v5 v6 v7) := by
    have equality_source := row_4
    have equality_step_1 := row_3
    cases equality_step_1
    have equality_step_2 := row_26
    cases equality_step_2
    have equality_step_3 := row_2
    cases equality_step_3
    exact equality_source
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
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
    (assumption_42 : (add v1 v2 v3))
    (assumption_36 : (mul v1 v4 v5))
    (assumption_32 : (succ previous v4))
    (assumption_18 : (mul v2 v4 v6))
    (assumption_14 : ((add v1 v2 v3) → (∀ (w1 : α), ((mul v1 previous w1) → (∀ (w2 : α), ((mul v2 previous w2) → (∀ (w3 : α), ((mul v3 previous w3) → (add w1 w2 w3)))))))))
    (assumption_9 : (mul v3 v4 v7))
    : (add v5 v6 v7) := by
  -- chapter_115_line_73: GL tag task formulation.
  have row_73 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_115_line_72: GL tag expansion.
  have row_72 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_73)
  -- chapter_115_line_71: GL tag disintegration.
  have row_71 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_72.1
  -- chapter_115_line_70: GL tag expansion.
  have row_70 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_71)
  -- chapter_115_line_69: GL tag disintegration.
  have row_69 : (gl_implication28 N succ mul add) := by
    exact row_70.2
  -- chapter_115_line_68: GL tag expansion.
  have row_68 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((mul w3 w1 w4) → (∀ (w5 : α), ((mul w3 w2 w5) → (add w4 w3 w5))))))))) := by
    simpa only [gl_implication28] using (row_69)
  -- chapter_115_line_67: GL tag disintegration.
  have row_67 : (gl_fXYZ add N N N) := by
    exact row_70.1.1.1.1.1.1.1.1.1.2
  -- chapter_115_line_66: GL tag expansion.
  have row_66 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_67)
  -- chapter_115_line_65: GL tag disintegration.
  have row_65 : (gl_implication18 N N N add) := by
    exact row_66.1.2
  -- chapter_115_line_64: GL tag expansion.
  have row_64 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication18] using (row_65)
  -- chapter_115_line_63: GL tag disintegration.
  have row_63 : (gl_implication13 add N) := by
    exact row_66.1.1.2
  -- chapter_115_line_62: GL tag expansion.
  have row_62 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_63)
  -- chapter_115_line_61: GL tag disintegration.
  have row_61 : (gl_implication12 add N) := by
    exact row_66.1.1.1.2
  -- chapter_115_line_60: GL tag expansion.
  have row_60 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_61)
  -- chapter_115_line_59: GL tag disintegration.
  have row_59 : (gl_implication11 add N) := by
    exact row_66.1.1.1.1
  -- chapter_115_line_58: GL tag expansion.
  have row_58 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_59)
  -- chapter_115_line_57: GL tag disintegration.
  have row_57 : (gl_fXY succ N N) := by
    exact row_70.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_115_line_56: GL tag expansion.
  have row_56 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_57)
  -- chapter_115_line_55: GL tag disintegration.
  have row_55 : (gl_implication0 succ N) := by
    exact row_56.1.1.1
  -- chapter_115_line_54: GL tag expansion.
  have row_54 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_55)
  -- chapter_115_line_53: GL tag disintegration.
  have row_53 : (gl_fXYZ mul N N N) := by
    exact row_70.1.1.1.1.2
  -- chapter_115_line_52: GL tag expansion.
  have row_52 : (((((gl_implication11 mul N) ∧ (gl_implication12 mul N)) ∧ (gl_implication13 mul N)) ∧ (gl_implication18 N N N mul)) ∧ (gl_implication19 N N mul)) := by
    simpa only [gl_fXYZ] using (row_53)
  -- chapter_115_line_51: GL tag disintegration.
  have row_51 : (gl_implication13 mul N) := by
    exact row_52.1.1.2
  -- chapter_115_line_50: GL tag expansion.
  have row_50 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_51)
  -- chapter_115_line_49: GL tag disintegration.
  have row_49 : (gl_implication12 mul N) := by
    exact row_52.1.1.1.2
  -- chapter_115_line_48: GL tag expansion.
  have row_48 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_49)
  -- chapter_115_line_47: GL tag disintegration.
  have row_47 : (gl_implication11 mul N) := by
    exact row_52.1.1.1.1
  -- chapter_115_line_46: GL tag expansion.
  have row_46 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_47)
  -- chapter_115_line_45: GL tag disintegration.
  have row_45 : (gl_implication18 N N N mul) := by
    exact row_52.1.2
  -- chapter_115_line_44: GL tag expansion.
  have row_44 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication18] using (row_45)
  -- chapter_115_line_43: GL tag implication grouping.
  have row_43 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_58
    have auxiliary_source_2 := row_60
    have auxiliary_source_3 := row_62
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_115_line_42: GL tag task formulation.
  have row_42 : (add v1 v2 v3) := by
    exact assumption_42
  -- chapter_115_line_41: GL tag implication.
  have row_41 : (gl_and21 v1 N v2 N v3 N) := by
    apply row_43
    exact row_42
  -- chapter_115_line_40: GL tag expansion.
  have row_40 : (((N v1) ∧ (N v2)) ∧ (N v3)) := by
    simpa only [gl_and21] using (row_41)
  -- chapter_115_line_39: GL tag disintegration.
  have row_39 : (N v2) := by
    exact row_40.1.2
  -- chapter_115_line_38: GL tag disintegration.
  have row_38 : (N v3) := by
    exact row_40.2
  -- chapter_115_line_37: GL tag implication grouping.
  have row_37 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_46
    have auxiliary_source_2 := row_48
    have auxiliary_source_3 := row_50
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_115_line_36: GL tag task formulation.
  have row_36 : (mul v1 v4 v5) := by
    exact assumption_36
  -- chapter_115_line_35: GL tag implication.
  have row_35 : (gl_and21 v1 N v4 N v5 N) := by
    apply row_37
    exact row_36
  -- chapter_115_line_34: GL tag expansion.
  have row_34 : (((N v1) ∧ (N v4)) ∧ (N v5)) := by
    simpa only [gl_and21] using (row_35)
  -- chapter_115_line_33: GL tag disintegration.
  have row_33 : (N v1) := by
    exact row_34.1.1
  -- chapter_115_line_32: GL tag recursion.
  have row_32 : (succ previous v4) := by
    exact assumption_32
  -- chapter_115_line_31: GL tag implication.
  have row_31 : (N previous) := by
    apply row_54
    exact row_32
  -- chapter_115_line_30: GL tag implication.
  have row_30 : (gl_existence1 N v1 previous mul) := by
    apply row_44
    exact row_33
    exact row_31
  -- chapter_115_line_29: GL tag implication.
  have row_29 : (gl_existence1 N v2 previous mul) := by
    apply row_44
    exact row_39
    exact row_31
  -- chapter_115_line_28: GL tag implication.
  have row_28 : (gl_existence1 N v3 previous mul) := by
    apply row_44
    exact row_38
    exact row_31
  -- chapter_115_line_27: GL tag expansion.
  have row_27 : (¬ (∀ (v11 : α), ((N v11) → (¬ (mul v3 previous v11))))) := by
    simpa only [gl_existence1] using (row_28)
  have exists_row_27 : ∃ (v11 : α), ((N v11) ∧ (mul v3 previous v11)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_27
      intro v11 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v11, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v11, witness_row_27⟩ := exists_row_27
  -- chapter_115_line_26: GL tag disintegration.
  have row_26 : (N v11) := by
    exact witness_row_27.1
  -- chapter_115_line_25: GL tag disintegration.
  have row_25 : (mul v3 previous v11) := by
    exact witness_row_27.2
  -- chapter_115_line_24: GL tag implication.
  have row_24 : (gl_existence1 N v11 v2 add) := by
    apply row_64
    exact row_26
    exact row_39
  -- chapter_115_line_23: GL tag expansion.
  have row_23 : (¬ (∀ (v9 : α), ((N v9) → (¬ (add v11 v2 v9))))) := by
    simpa only [gl_existence1] using (row_24)
  have exists_row_23 : ∃ (v9 : α), ((N v9) ∧ (add v11 v2 v9)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_23
      intro v9 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v9, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v9, witness_row_23⟩ := exists_row_23
  -- chapter_115_line_22: GL tag disintegration.
  have row_22 : (add v11 v2 v9) := by
    exact witness_row_23.2
  -- chapter_115_line_21: GL tag implication.
  have row_21 : (gl_existence1 N v11 v1 add) := by
    apply row_64
    exact row_26
    exact row_33
  -- chapter_115_line_20: GL tag expansion.
  have row_20 : (¬ (∀ (v10 : α), ((N v10) → (¬ (mul v2 previous v10))))) := by
    simpa only [gl_existence1] using (row_29)
  have exists_row_20 : ∃ (v10 : α), ((N v10) ∧ (mul v2 previous v10)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_20
      intro v10 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v10, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v10, witness_row_20⟩ := exists_row_20
  -- chapter_115_line_19: GL tag disintegration.
  have row_19 : (mul v2 previous v10) := by
    exact witness_row_20.2
  -- chapter_115_line_18: GL tag task formulation.
  have row_18 : (mul v2 v4 v6) := by
    exact assumption_18
  -- chapter_115_line_17: GL tag implication.
  have row_17 : (add v10 v2 v6) := by
    apply row_68
    exact row_31
    exact row_32
    exact row_19
    exact row_18
  -- chapter_115_line_16: GL tag expansion.
  have row_16 : (¬ (∀ (v8 : α), ((N v8) → (¬ (mul v1 previous v8))))) := by
    simpa only [gl_existence1] using (row_30)
  have exists_row_16 : ∃ (v8 : α), ((N v8) ∧ (mul v1 previous v8)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_16
      intro v8 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v8, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v8, witness_row_16⟩ := exists_row_16
  -- chapter_115_line_15: GL tag disintegration.
  have row_15 : (mul v1 previous v8) := by
    exact witness_row_16.2
  -- chapter_115_line_14: GL tag recursion.
  have row_14 : ((add v1 v2 v3) → (∀ (w1 : α), ((mul v1 previous w1) → (∀ (w2 : α), ((mul v2 previous w2) → (∀ (w3 : α), ((mul v3 previous w3) → (add w1 w2 w3)))))))) := by
    exact assumption_14
  -- chapter_115_line_13: GL tag implication.
  have row_13 : (add v8 v10 v11) := by
    apply row_14
    exact row_42
    exact row_15
    exact row_19
    exact row_25
  -- chapter_115_line_12: GL tag theorem.
  have row_12 := peano_source_022 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_115_line_11: GL tag implication.
  have row_11 : (add v8 v6 v9) := by
    apply row_12
    exact row_22
    exact row_13
    exact row_17
  -- chapter_115_line_10: GL tag implication.
  have row_10 : (add v8 v1 v5) := by
    apply row_68
    exact row_31
    exact row_32
    exact row_15
    exact row_36
  -- chapter_115_line_9: GL tag task formulation.
  have row_9 : (mul v3 v4 v7) := by
    exact assumption_9
  -- chapter_115_line_8: GL tag implication.
  have row_8 : (add v11 v3 v7) := by
    apply row_68
    exact row_31
    exact row_32
    exact row_25
    exact row_9
  -- chapter_115_line_7: GL tag expansion.
  have row_7 : (¬ (∀ (v20 : α), ((N v20) → (¬ (add v11 v1 v20))))) := by
    simpa only [gl_existence1] using (row_21)
  have exists_row_7 : ∃ (v20 : α), ((N v20) ∧ (add v11 v1 v20)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_7
      intro v20 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v20, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v20, witness_row_7⟩ := exists_row_7
  -- chapter_115_line_6: GL tag disintegration.
  have row_6 : (add v11 v1 v20) := by
    exact witness_row_7.2
  -- chapter_115_line_5: GL tag theorem.
  have row_5 := peano_source_019 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_115_line_4: GL tag implication.
  have row_4 : (add v20 v2 v7) := by
    apply row_5
    exact row_6
    exact row_8
    exact row_42
  -- chapter_115_line_3: GL tag theorem.
  have row_3 := peano_source_021 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_115_line_2: GL tag implication.
  have row_2 : (add v9 v1 v7) := by
    apply row_3
    exact row_4
    exact row_6
    exact row_22
  -- chapter_115_line_1: GL tag implication.
  have row_1 : (add v5 v6 v7) := by
    apply row_3
    exact row_2
    exact row_11
    exact row_10
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α) (v5 : α), ((mul v1 v4 v5) → (∀ (v6 : α), ((mul v2 v4 v6) → (∀ (v7 : α), ((mul v3 v4 v7) → (add v5 v6 v7))))))))) := by
  intro v1
  intro v2
  intro v3
  intro premise_1
  intro v4
  intro v5
  intro premise_2
  intro v6
  intro premise_3
  intro v7
  intro premise_4
  have inductionMember : N v4 := by
    have typingRule := peano_source_055_induction_typing N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact typingRule v3 v4 v7 premise_4
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v5 : α), ((mul v1 zero v5) → (∀ (v6 : α), ((mul v2 zero v6) → (∀ (v7 : α), ((mul v3 zero v7) → (add v5 v6 v7))))))))) := by
    intro v1
    intro v2
    intro v3
    intro base_premise_1
    intro v5
    intro base_premise_2
    intro v6
    intro base_premise_3
    intro v7
    intro base_premise_4
    have zeroRule := peano_source_055_check_zero N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact zeroRule v1 v2 v3 zero v5 v6 v7 rfl base_premise_1 base_premise_3 base_premise_2 base_premise_4
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v5 : α), ((mul v1 induction_n v5) → (∀ (v6 : α), ((mul v2 induction_n v6) → (∀ (v7 : α), ((mul v3 induction_n v7) → (add v5 v6 v7))))))))) → ∀ induction_m, succ induction_n induction_m → (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v5 : α), ((mul v1 induction_m v5) → (∀ (v6 : α), ((mul v2 induction_m v6) → (∀ (v7 : α), ((mul v3 induction_m v7) → (add v5 v6 v7))))))))) := by
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
    intro v7
    intro step_premise_4
    have step_induction_assumption_1 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have step_induction_assumption_2 :
        ((add v1 v2 v3) → (∀ (w1 : α), ((mul v1 induction_n w1) → (∀ (w2 : α), ((mul v2 induction_n w2) → (∀ (w3 : α), ((mul v3 induction_n w3) → (add w1 w2 w3)))))))) := by
      intro step_induction_assumption_2_premise_1
      intro w1
      intro step_induction_assumption_2_premise_2
      intro w2
      intro step_induction_assumption_2_premise_3
      intro w3
      intro step_induction_assumption_2_premise_4
      apply induction_hypothesis
      all_goals assumption
    have stepRule := peano_source_055_check_induction_condition N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact stepRule induction_n v1 v2 v3 induction_m v5 v6 v7 step_premise_1 step_premise_2 step_induction_assumption_1 step_premise_3 step_induction_assumption_2 step_premise_4
  have inductionProperty : (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v5 : α), ((mul v1 v4 v5) → (∀ (v6 : α), ((mul v2 v4 v6) → (∀ (v7 : α), ((mul v3 v4 v7) → (add v5 v6 v7))))))))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v5 : α), ((mul v1 induction_value v5) → (∀ (v6 : α), ((mul v2 induction_value v6) → (∀ (v7 : α), ((mul v3 induction_value v7) → (add v5 v6 v7))))))))))
      v4
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v1 v2 v3 premise_1 v5 premise_2 v6 premise_3 v7 premise_4

private theorem peano_source_054_induction_typing
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (assumption_2 : (mul v1 v2 v3))
    : (N v1) := by
  -- chapter_110_line_10: GL tag task formulation.
  have row_10 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_110_line_9: GL tag expansion.
  have row_9 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_10)
  -- chapter_110_line_8: GL tag disintegration.
  have row_8 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_9.1
  -- chapter_110_line_7: GL tag expansion.
  have row_7 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_8)
  -- chapter_110_line_6: GL tag disintegration.
  have row_6 : (gl_fXYZ mul N N N) := by
    exact row_7.1.1.1.1.2
  -- chapter_110_line_5: GL tag expansion.
  have row_5 : (((((gl_implication11 mul N) ∧ (gl_implication12 mul N)) ∧ (gl_implication13 mul N)) ∧ (gl_implication18 N N N mul)) ∧ (gl_implication19 N N mul)) := by
    simpa only [gl_fXYZ] using (row_6)
  -- chapter_110_line_4: GL tag disintegration.
  have row_4 : (gl_implication11 mul N) := by
    exact row_5.1.1.1.1
  -- chapter_110_line_3: GL tag expansion.
  have row_3 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_4)
  -- chapter_110_line_2: GL tag task formulation.
  have row_2 : (mul v1 v2 v3) := by
    exact assumption_2
  -- chapter_110_line_1: GL tag implication.
  have row_1 : (N v1) := by
    apply row_3
    exact row_2
  exact row_1

private theorem peano_source_054_check_zero
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (assumption_17 : (mul v1 v2 v3))
    (assumption_12 : (v1 = zero))
    : (mul v2 v1 v3) := by
  -- chapter_111_line_34: GL tag task formulation.
  have row_34 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_111_line_33: GL tag expansion.
  have row_33 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_34)
  -- chapter_111_line_32: GL tag disintegration.
  have row_32 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_33.1
  -- chapter_111_line_31: GL tag expansion.
  have row_31 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_32)
  -- chapter_111_line_30: GL tag disintegration.
  have row_30 : (gl_fXYZ mul N N N) := by
    exact row_31.1.1.1.1.2
  -- chapter_111_line_29: GL tag expansion.
  have row_29 : (((((gl_implication11 mul N) ∧ (gl_implication12 mul N)) ∧ (gl_implication13 mul N)) ∧ (gl_implication18 N N N mul)) ∧ (gl_implication19 N N mul)) := by
    simpa only [gl_fXYZ] using (row_30)
  -- chapter_111_line_28: GL tag disintegration.
  have row_28 : (gl_implication13 mul N) := by
    exact row_29.1.1.2
  -- chapter_111_line_27: GL tag expansion.
  have row_27 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_28)
  -- chapter_111_line_26: GL tag disintegration.
  have row_26 : (gl_implication12 mul N) := by
    exact row_29.1.1.1.2
  -- chapter_111_line_25: GL tag expansion.
  have row_25 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_26)
  -- chapter_111_line_24: GL tag disintegration.
  have row_24 : (gl_implication11 mul N) := by
    exact row_29.1.1.1.1
  -- chapter_111_line_23: GL tag expansion.
  have row_23 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_24)
  -- chapter_111_line_22: GL tag disintegration.
  have row_22 : (gl_implication18 N N N mul) := by
    exact row_29.1.2
  -- chapter_111_line_21: GL tag expansion.
  have row_21 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication18] using (row_22)
  -- chapter_111_line_20: GL tag disintegration.
  have row_20 : (gl_implication25 N zero mul) := by
    exact row_31.1.1.1.2
  -- chapter_111_line_19: GL tag expansion.
  have row_19 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((mul w1 zero w2) → (w2 = zero))))) := by
    simpa only [gl_implication25] using (row_20)
  -- chapter_111_line_18: GL tag implication grouping.
  have row_18 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_23
    have auxiliary_source_2 := row_25
    have auxiliary_source_3 := row_27
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_111_line_17: GL tag task formulation.
  have row_17 : (mul v1 v2 v3) := by
    exact assumption_17
  -- chapter_111_line_16: GL tag implication.
  have row_16 : (gl_and21 v1 N v2 N v3 N) := by
    apply row_18
    exact row_17
  -- chapter_111_line_15: GL tag expansion.
  have row_15 : (((N v1) ∧ (N v2)) ∧ (N v3)) := by
    simpa only [gl_and21] using (row_16)
  -- chapter_111_line_14: GL tag disintegration.
  have row_14 : (N v2) := by
    exact row_15.1.2
  -- chapter_111_line_13: GL tag disintegration.
  have row_13 : (N v1) := by
    exact row_15.1.1
  -- chapter_111_line_12: GL tag recursion.
  have row_12 : (v1 = zero) := by
    exact assumption_12
  -- chapter_111_line_11: GL tag equality1.
  have row_11 : (mul zero v2 v3) := by
    have equality_source := row_17
    have equality_step_1 := row_12
    cases equality_step_1
    exact equality_source
  -- chapter_111_line_10: GL tag theorem.
  have row_10 := peano_source_048 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_111_line_9: GL tag implication.
  have row_9 : (zero = v3) := by
    apply row_10
    exact row_14
    exact row_11
  -- chapter_111_line_8: GL tag implication.
  have row_8 : (gl_existence1 N v2 v1 mul) := by
    apply row_21
    exact row_14
    exact row_13
  -- chapter_111_line_7: GL tag expansion.
  have row_7 : (¬ (∀ (v4 : α), ((N v4) → (¬ (mul v2 v1 v4))))) := by
    simpa only [gl_existence1] using (row_8)
  have exists_row_7 : ∃ (v4 : α), ((N v4) ∧ (mul v2 v1 v4)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_7
      intro v4 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v4, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v4, witness_row_7⟩ := exists_row_7
  -- chapter_111_line_6: GL tag disintegration.
  have row_6 : (mul v2 v1 v4) := by
    exact witness_row_7.2
  -- chapter_111_line_5: GL tag equality1.
  have row_5 : (mul v2 zero v4) := by
    have equality_source := row_6
    have equality_step_1 := row_12
    cases equality_step_1
    exact equality_source
  -- chapter_111_line_4: GL tag implication.
  have row_4 : (v4 = zero) := by
    apply row_19
    exact row_14
    exact row_5
  -- chapter_111_line_3: GL tag equality2.
  have row_3 : (v4 = v3) := by
    exact Eq.trans row_4 row_9
  -- chapter_111_line_2: GL tag symmetry of equality.
  have row_2 : (zero = v1) := by
    exact Eq.symm row_12
  -- chapter_111_line_1: GL tag equality1.
  have row_1 : (mul v2 v1 v3) := by
    have equality_source := row_5
    have equality_step_1 := row_2
    cases equality_step_1
    have equality_step_2 := row_3
    cases equality_step_2
    exact equality_source
  exact row_1

private theorem peano_source_054_check_induction_condition
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (assumption_48 : (succ previous v1))
    (assumption_19 : (mul v1 v2 v3))
    : (mul v2 v1 v3) := by
  -- chapter_112_line_49: GL tag task formulation.
  have row_49 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_112_line_48: GL tag recursion.
  have row_48 : (succ previous v1) := by
    exact assumption_48
  -- chapter_112_line_47: GL tag theorem.
  have row_47 := peano_source_027 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_112_line_46: GL tag implication.
  have row_46 : (add previous one v1) := by
    apply row_47
    exact row_48
  -- chapter_112_line_45: GL tag expansion.
  have row_45 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_49)
  -- chapter_112_line_44: GL tag disintegration.
  have row_44 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_45.1
  -- chapter_112_line_43: GL tag expansion.
  have row_43 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_44)
  -- chapter_112_line_42: GL tag disintegration.
  have row_42 : (gl_fXYZ mul N N N) := by
    exact row_43.1.1.1.1.2
  -- chapter_112_line_41: GL tag expansion.
  have row_41 : (((((gl_implication11 mul N) ∧ (gl_implication12 mul N)) ∧ (gl_implication13 mul N)) ∧ (gl_implication18 N N N mul)) ∧ (gl_implication19 N N mul)) := by
    simpa only [gl_fXYZ] using (row_42)
  -- chapter_112_line_40: GL tag disintegration.
  have row_40 : (gl_implication13 mul N) := by
    exact row_41.1.1.2
  -- chapter_112_line_39: GL tag expansion.
  have row_39 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_40)
  -- chapter_112_line_38: GL tag disintegration.
  have row_38 : (gl_implication12 mul N) := by
    exact row_41.1.1.1.2
  -- chapter_112_line_37: GL tag expansion.
  have row_37 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_38)
  -- chapter_112_line_36: GL tag disintegration.
  have row_36 : (gl_implication11 mul N) := by
    exact row_41.1.1.1.1
  -- chapter_112_line_35: GL tag expansion.
  have row_35 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_36)
  -- chapter_112_line_34: GL tag disintegration.
  have row_34 : (gl_implication18 N N N mul) := by
    exact row_41.1.2
  -- chapter_112_line_33: GL tag expansion.
  have row_33 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication18] using (row_34)
  -- chapter_112_line_32: GL tag disintegration.
  have row_32 : (gl_fXY succ N N) := by
    exact row_43.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_112_line_31: GL tag expansion.
  have row_31 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_32)
  -- chapter_112_line_30: GL tag disintegration.
  have row_30 : (gl_implication0 succ N) := by
    exact row_31.1.1.1
  -- chapter_112_line_29: GL tag expansion.
  have row_29 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_30)
  -- chapter_112_line_28: GL tag disintegration.
  have row_28 : (gl_fXYZ add N N N) := by
    exact row_43.1.1.1.1.1.1.1.1.1.2
  -- chapter_112_line_27: GL tag expansion.
  have row_27 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_28)
  -- chapter_112_line_26: GL tag disintegration.
  have row_26 : (gl_implication18 N N N add) := by
    exact row_27.1.2
  -- chapter_112_line_25: GL tag expansion.
  have row_25 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication18] using (row_26)
  -- chapter_112_line_24: GL tag disintegration.
  have row_24 : (N zero) := by
    exact row_43.1.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_112_line_23: GL tag disintegration.
  have row_23 : (gl_implication23 N succ add) := by
    exact row_43.1.1.1.1.1.1.2
  -- chapter_112_line_22: GL tag expansion.
  have row_22 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication23] using (row_23)
  -- chapter_112_line_21: GL tag disintegration.
  have row_21 : (succ zero one) := by
    exact row_45.2
  -- chapter_112_line_20: GL tag implication grouping.
  have row_20 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_35
    have auxiliary_source_2 := row_37
    have auxiliary_source_3 := row_39
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_112_line_19: GL tag task formulation.
  have row_19 : (mul v1 v2 v3) := by
    exact assumption_19
  -- chapter_112_line_18: GL tag implication.
  have row_18 : (gl_and21 v1 N v2 N v3 N) := by
    apply row_20
    exact row_19
  -- chapter_112_line_17: GL tag expansion.
  have row_17 : (((N v1) ∧ (N v2)) ∧ (N v3)) := by
    simpa only [gl_and21] using (row_18)
  -- chapter_112_line_16: GL tag disintegration.
  have row_16 : (N v2) := by
    exact row_17.1.2
  -- chapter_112_line_15: GL tag implication.
  have row_15 : (N previous) := by
    apply row_29
    exact row_48
  -- chapter_112_line_14: GL tag implication.
  have row_14 : (gl_existence1 N v2 previous mul) := by
    apply row_33
    exact row_16
    exact row_15
  -- chapter_112_line_13: GL tag implication.
  have row_13 : (gl_existence1 N previous zero add) := by
    apply row_25
    exact row_15
    exact row_24
  -- chapter_112_line_12: GL tag expansion.
  have row_12 : (¬ (∀ (v4 : α), ((N v4) → (¬ (add previous zero v4))))) := by
    simpa only [gl_existence1] using (row_13)
  have exists_row_12 : ∃ (v4 : α), ((N v4) ∧ (add previous zero v4)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_12
      intro v4 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v4, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v4, witness_row_12⟩ := exists_row_12
  -- chapter_112_line_11: GL tag disintegration.
  have row_11 : (add previous zero v4) := by
    exact witness_row_12.2
  -- chapter_112_line_10: GL tag implication.
  have row_10 : (succ v4 v1) := by
    apply row_22
    exact row_24
    exact row_21
    exact row_11
    exact row_46
  -- chapter_112_line_9: GL tag theorem.
  have row_9 := peano_source_011 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_112_line_8: GL tag implication.
  have row_8 : (previous = v4) := by
    apply row_9
    exact row_11
    exact row_15
  -- chapter_112_line_7: GL tag expansion.
  have row_7 : (¬ (∀ (v5 : α), ((N v5) → (¬ (mul v2 previous v5))))) := by
    simpa only [gl_existence1] using (row_14)
  have exists_row_7 : ∃ (v5 : α), ((N v5) ∧ (mul v2 previous v5)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_7
      intro v5 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v5, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v5, witness_row_7⟩ := exists_row_7
  -- chapter_112_line_6: GL tag disintegration.
  have row_6 : (mul v2 previous v5) := by
    exact witness_row_7.2
  -- chapter_112_line_5: GL tag equality1.
  have row_5 : (mul v2 v4 v5) := by
    have equality_source := row_6
    have equality_step_1 := row_8
    cases equality_step_1
    exact equality_source
  -- chapter_112_line_4: GL tag theorem.
  have row_4 := peano_source_052 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_112_line_3: GL tag implication.
  have row_3 : (add v2 v5 v3) := by
    apply row_4
    exact row_5
    exact row_19
    exact row_10
  -- chapter_112_line_2: GL tag theorem.
  have row_2 := peano_source_053 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_112_line_1: GL tag implication.
  have row_1 : (mul v2 v1 v3) := by
    apply row_2
    exact row_3
    exact row_10
    exact row_5
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α) (v3 : α), ((mul v1 v2 v3) → (mul v2 v1 v3))) := by
  intro v1
  intro v2
  intro v3
  intro premise_1
  have inductionMember : N v1 := by
    have typingRule := peano_source_054_induction_typing N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact typingRule v1 v2 v3 premise_1
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v2 : α) (v3 : α), ((mul zero v2 v3) → (mul v2 zero v3))) := by
    intro v2
    intro v3
    intro base_premise_1
    have zeroRule := peano_source_054_check_zero N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact zeroRule zero v2 v3 base_premise_1 rfl
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v2 : α) (v3 : α), ((mul induction_n v2 v3) → (mul v2 induction_n v3))) → ∀ induction_m, succ induction_n induction_m → (∀ (v2 : α) (v3 : α), ((mul induction_m v2 v3) → (mul v2 induction_m v3))) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro v2
    intro v3
    intro step_premise_1
    have step_induction_assumption_1 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have stepRule := peano_source_054_check_induction_condition N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact stepRule induction_n induction_m v2 v3 step_induction_assumption_1 step_premise_1
  have inductionProperty : (∀ (v2 : α) (v3 : α), ((mul v1 v2 v3) → (mul v2 v1 v3))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v2 : α) (v3 : α), ((mul induction_value v2 v3) → (mul v2 induction_value v3))))
      v1
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v2 v3 premise_1

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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v1 : α)
    (v4 : α)
    (v5 : α)
    (assumption_4 : (mul v4 v5 v1))
    : (N v5) := by
  -- chapter_116_line_17: GL tag task formulation.
  have row_17 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_116_line_16: GL tag expansion.
  have row_16 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_17)
  -- chapter_116_line_15: GL tag disintegration.
  have row_15 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_16.1
  -- chapter_116_line_14: GL tag expansion.
  have row_14 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_15)
  -- chapter_116_line_13: GL tag disintegration.
  have row_13 : (gl_fXYZ mul N N N) := by
    exact row_14.1.1.1.1.2
  -- chapter_116_line_12: GL tag expansion.
  have row_12 : (((((gl_implication11 mul N) ∧ (gl_implication12 mul N)) ∧ (gl_implication13 mul N)) ∧ (gl_implication18 N N N mul)) ∧ (gl_implication19 N N mul)) := by
    simpa only [gl_fXYZ] using (row_13)
  -- chapter_116_line_11: GL tag disintegration.
  have row_11 : (gl_implication13 mul N) := by
    exact row_12.1.1.2
  -- chapter_116_line_10: GL tag expansion.
  have row_10 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_11)
  -- chapter_116_line_9: GL tag disintegration.
  have row_9 : (gl_implication12 mul N) := by
    exact row_12.1.1.1.2
  -- chapter_116_line_8: GL tag expansion.
  have row_8 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_9)
  -- chapter_116_line_7: GL tag disintegration.
  have row_7 : (gl_implication11 mul N) := by
    exact row_12.1.1.1.1
  -- chapter_116_line_6: GL tag expansion.
  have row_6 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_7)
  -- chapter_116_line_5: GL tag implication grouping.
  have row_5 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_6
    have auxiliary_source_2 := row_8
    have auxiliary_source_3 := row_10
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_116_line_4: GL tag task formulation.
  have row_4 : (mul v4 v5 v1) := by
    exact assumption_4
  -- chapter_116_line_3: GL tag implication.
  have row_3 : (gl_and21 v4 N v5 N v1 N) := by
    apply row_5
    exact row_4
  -- chapter_116_line_2: GL tag expansion.
  have row_2 : (((N v4) ∧ (N v5)) ∧ (N v1)) := by
    simpa only [gl_and21] using (row_3)
  -- chapter_116_line_1: GL tag disintegration.
  have row_1 : (N v5) := by
    exact row_2.1.2
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (v4 : α)
    (v5 : α)
    (v6 : α)
    (assumption_42 : (v5 = zero))
    (assumption_41 : (mul v4 v5 v1))
    (assumption_17 : (mul v1 v2 v3))
    (assumption_15 : (mul v4 v2 v6))
    : (mul v6 v5 v3) := by
  -- chapter_117_line_42: GL tag recursion.
  have row_42 : (v5 = zero) := by
    exact assumption_42
  -- chapter_117_line_41: GL tag task formulation.
  have row_41 : (mul v4 v5 v1) := by
    exact assumption_41
  -- chapter_117_line_40: GL tag equality1.
  have row_40 : (mul v4 zero v1) := by
    have equality_source := row_41
    have equality_step_1 := row_42
    cases equality_step_1
    exact equality_source
  -- chapter_117_line_39: GL tag task formulation.
  have row_39 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_117_line_38: GL tag expansion.
  have row_38 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_39)
  -- chapter_117_line_37: GL tag disintegration.
  have row_37 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_38.1
  -- chapter_117_line_36: GL tag expansion.
  have row_36 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_37)
  -- chapter_117_line_35: GL tag disintegration.
  have row_35 : (gl_fXYZ mul N N N) := by
    exact row_36.1.1.1.1.2
  -- chapter_117_line_34: GL tag expansion.
  have row_34 : (((((gl_implication11 mul N) ∧ (gl_implication12 mul N)) ∧ (gl_implication13 mul N)) ∧ (gl_implication18 N N N mul)) ∧ (gl_implication19 N N mul)) := by
    simpa only [gl_fXYZ] using (row_35)
  -- chapter_117_line_33: GL tag disintegration.
  have row_33 : (gl_implication13 mul N) := by
    exact row_34.1.1.2
  -- chapter_117_line_32: GL tag expansion.
  have row_32 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_33)
  -- chapter_117_line_31: GL tag disintegration.
  have row_31 : (gl_implication12 mul N) := by
    exact row_34.1.1.1.2
  -- chapter_117_line_30: GL tag expansion.
  have row_30 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_31)
  -- chapter_117_line_29: GL tag disintegration.
  have row_29 : (gl_implication11 mul N) := by
    exact row_34.1.1.1.1
  -- chapter_117_line_28: GL tag expansion.
  have row_28 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_29)
  -- chapter_117_line_27: GL tag disintegration.
  have row_27 : (gl_implication18 N N N mul) := by
    exact row_34.1.2
  -- chapter_117_line_26: GL tag expansion.
  have row_26 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication18] using (row_27)
  -- chapter_117_line_25: GL tag implication grouping.
  have row_25 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_28
    have auxiliary_source_2 := row_30
    have auxiliary_source_3 := row_32
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_117_line_24: GL tag implication.
  have row_24 : (gl_and21 v4 N v5 N v1 N) := by
    apply row_25
    exact row_41
  -- chapter_117_line_23: GL tag expansion.
  have row_23 : (((N v4) ∧ (N v5)) ∧ (N v1)) := by
    simpa only [gl_and21] using (row_24)
  -- chapter_117_line_22: GL tag disintegration.
  have row_22 : (N v4) := by
    exact row_23.1.1
  -- chapter_117_line_21: GL tag disintegration.
  have row_21 : (N v5) := by
    exact row_23.1.2
  -- chapter_117_line_20: GL tag theorem.
  have row_20 := peano_source_025 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_117_line_19: GL tag implication.
  have row_19 : (zero = v1) := by
    apply row_20
    exact row_40
    exact row_22
  -- chapter_117_line_18: GL tag symmetry of equality.
  have row_18 : (v1 = zero) := by
    exact Eq.symm row_19
  -- chapter_117_line_17: GL tag task formulation.
  have row_17 : (mul v1 v2 v3) := by
    exact assumption_17
  -- chapter_117_line_16: GL tag equality1.
  have row_16 : (mul zero v2 v3) := by
    have equality_source := row_17
    have equality_step_1 := row_18
    cases equality_step_1
    exact equality_source
  -- chapter_117_line_15: GL tag task formulation.
  have row_15 : (mul v4 v2 v6) := by
    exact assumption_15
  -- chapter_117_line_14: GL tag implication.
  have row_14 : (gl_and21 v4 N v2 N v6 N) := by
    apply row_25
    exact row_15
  -- chapter_117_line_13: GL tag expansion.
  have row_13 : (((N v4) ∧ (N v2)) ∧ (N v6)) := by
    simpa only [gl_and21] using (row_14)
  -- chapter_117_line_12: GL tag disintegration.
  have row_12 : (N v2) := by
    exact row_13.1.2
  -- chapter_117_line_11: GL tag disintegration.
  have row_11 : (N v6) := by
    exact row_13.2
  -- chapter_117_line_10: GL tag theorem.
  have row_10 := peano_source_048 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_117_line_9: GL tag implication.
  have row_9 : (zero = v3) := by
    apply row_10
    exact row_12
    exact row_16
  -- chapter_117_line_8: GL tag implication.
  have row_8 : (gl_existence1 N v6 v5 mul) := by
    apply row_26
    exact row_11
    exact row_21
  -- chapter_117_line_7: GL tag expansion.
  have row_7 : (¬ (∀ (v7 : α), ((N v7) → (¬ (mul v6 v5 v7))))) := by
    simpa only [gl_existence1] using (row_8)
  have exists_row_7 : ∃ (v7 : α), ((N v7) ∧ (mul v6 v5 v7)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_7
      intro v7 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v7, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v7, witness_row_7⟩ := exists_row_7
  -- chapter_117_line_6: GL tag disintegration.
  have row_6 : (mul v6 v5 v7) := by
    exact witness_row_7.2
  -- chapter_117_line_5: GL tag equality1.
  have row_5 : (mul v6 zero v7) := by
    have equality_source := row_6
    have equality_step_1 := row_42
    cases equality_step_1
    exact equality_source
  -- chapter_117_line_4: GL tag implication.
  have row_4 : (zero = v7) := by
    apply row_20
    exact row_5
    exact row_11
  -- chapter_117_line_3: GL tag symmetry of equality.
  have row_3 : (v7 = zero) := by
    exact Eq.symm row_4
  -- chapter_117_line_2: GL tag equality2.
  have row_2 : (v7 = v3) := by
    exact Eq.trans row_3 row_9
  -- chapter_117_line_1: GL tag equality1.
  have row_1 : (mul v6 v5 v3) := by
    have equality_source := row_6
    have equality_step_1 := row_2
    cases equality_step_1
    exact equality_source
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (v4 : α)
    (v5 : α)
    (v6 : α)
    (assumption_72 : (succ previous v5))
    (assumption_39 : (mul v4 v2 v6))
    (assumption_34 : (mul v4 v5 v1))
    (assumption_19 : (∀ (w1 : α) (w2 : α), ((mul w1 v2 w2) → ((mul v4 previous w1) → ((mul v4 v2 v6) → (mul v6 previous w2))))))
    (assumption_12 : (mul v1 v2 v3))
    : (mul v6 v5 v3) := by
  -- chapter_118_line_73: GL tag task formulation.
  have row_73 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_118_line_72: GL tag recursion.
  have row_72 : (succ previous v5) := by
    exact assumption_72
  -- chapter_118_line_71: GL tag theorem.
  have row_71 := peano_source_027 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_118_line_70: GL tag implication.
  have row_70 : (add previous one v5) := by
    apply row_71
    exact row_72
  -- chapter_118_line_69: GL tag expansion.
  have row_69 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_73)
  -- chapter_118_line_68: GL tag disintegration.
  have row_68 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_69.1
  -- chapter_118_line_67: GL tag expansion.
  have row_67 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_68)
  -- chapter_118_line_66: GL tag disintegration.
  have row_66 : (gl_fXYZ mul N N N) := by
    exact row_67.1.1.1.1.2
  -- chapter_118_line_65: GL tag expansion.
  have row_65 : (((((gl_implication11 mul N) ∧ (gl_implication12 mul N)) ∧ (gl_implication13 mul N)) ∧ (gl_implication18 N N N mul)) ∧ (gl_implication19 N N mul)) := by
    simpa only [gl_fXYZ] using (row_66)
  -- chapter_118_line_64: GL tag disintegration.
  have row_64 : (gl_implication19 N N mul) := by
    exact row_65.2
  -- chapter_118_line_63: GL tag expansion.
  have row_63 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (∀ (w3 : α), ((mul w1 w2 w3) → (∀ (w4 : α), ((mul w1 w2 w4) → (w3 = w4))))))))) := by
    simpa only [gl_implication19] using (row_64)
  -- chapter_118_line_62: GL tag disintegration.
  have row_62 : (gl_implication13 mul N) := by
    exact row_65.1.1.2
  -- chapter_118_line_61: GL tag expansion.
  have row_61 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_62)
  -- chapter_118_line_60: GL tag disintegration.
  have row_60 : (gl_implication12 mul N) := by
    exact row_65.1.1.1.2
  -- chapter_118_line_59: GL tag expansion.
  have row_59 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_60)
  -- chapter_118_line_58: GL tag disintegration.
  have row_58 : (gl_implication11 mul N) := by
    exact row_65.1.1.1.1
  -- chapter_118_line_57: GL tag expansion.
  have row_57 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_58)
  -- chapter_118_line_56: GL tag disintegration.
  have row_56 : (gl_fXY succ N N) := by
    exact row_67.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_118_line_55: GL tag expansion.
  have row_55 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_56)
  -- chapter_118_line_54: GL tag disintegration.
  have row_54 : (gl_implication0 succ N) := by
    exact row_55.1.1.1
  -- chapter_118_line_53: GL tag expansion.
  have row_53 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_54)
  -- chapter_118_line_52: GL tag disintegration.
  have row_52 : (gl_implication18 N N N mul) := by
    exact row_65.1.2
  -- chapter_118_line_51: GL tag expansion.
  have row_51 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication18] using (row_52)
  -- chapter_118_line_50: GL tag disintegration.
  have row_50 : (gl_implication28 N succ mul add) := by
    exact row_67.2
  -- chapter_118_line_49: GL tag expansion.
  have row_49 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((mul w3 w1 w4) → (∀ (w5 : α), ((mul w3 w2 w5) → (add w4 w3 w5))))))))) := by
    simpa only [gl_implication28] using (row_50)
  -- chapter_118_line_48: GL tag disintegration.
  have row_48 : (gl_fXYZ add N N N) := by
    exact row_67.1.1.1.1.1.1.1.1.1.2
  -- chapter_118_line_47: GL tag expansion.
  have row_47 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_48)
  -- chapter_118_line_46: GL tag disintegration.
  have row_46 : (gl_implication18 N N N add) := by
    exact row_47.1.2
  -- chapter_118_line_45: GL tag expansion.
  have row_45 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication18] using (row_46)
  -- chapter_118_line_44: GL tag disintegration.
  have row_44 : (N zero) := by
    exact row_67.1.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_118_line_43: GL tag disintegration.
  have row_43 : (gl_implication23 N succ add) := by
    exact row_67.1.1.1.1.1.1.2
  -- chapter_118_line_42: GL tag expansion.
  have row_42 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication23] using (row_43)
  -- chapter_118_line_41: GL tag disintegration.
  have row_41 : (succ zero one) := by
    exact row_69.2
  -- chapter_118_line_40: GL tag implication grouping.
  have row_40 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_57
    have auxiliary_source_2 := row_59
    have auxiliary_source_3 := row_61
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_118_line_39: GL tag task formulation.
  have row_39 : (mul v4 v2 v6) := by
    exact assumption_39
  -- chapter_118_line_38: GL tag implication.
  have row_38 : (gl_and21 v4 N v2 N v6 N) := by
    apply row_40
    exact row_39
  -- chapter_118_line_37: GL tag expansion.
  have row_37 : (((N v4) ∧ (N v2)) ∧ (N v6)) := by
    simpa only [gl_and21] using (row_38)
  -- chapter_118_line_36: GL tag disintegration.
  have row_36 : (N v6) := by
    exact row_37.2
  -- chapter_118_line_35: GL tag disintegration.
  have row_35 : (N v2) := by
    exact row_37.1.2
  -- chapter_118_line_34: GL tag task formulation.
  have row_34 : (mul v4 v5 v1) := by
    exact assumption_34
  -- chapter_118_line_33: GL tag implication.
  have row_33 : (gl_and21 v4 N v5 N v1 N) := by
    apply row_40
    exact row_34
  -- chapter_118_line_32: GL tag expansion.
  have row_32 : (((N v4) ∧ (N v5)) ∧ (N v1)) := by
    simpa only [gl_and21] using (row_33)
  -- chapter_118_line_31: GL tag disintegration.
  have row_31 : (N v4) := by
    exact row_32.1.1
  -- chapter_118_line_30: GL tag implication.
  have row_30 : (N previous) := by
    apply row_53
    exact row_72
  -- chapter_118_line_29: GL tag implication.
  have row_29 : (gl_existence1 N v4 previous mul) := by
    apply row_51
    exact row_31
    exact row_30
  -- chapter_118_line_28: GL tag implication.
  have row_28 : (gl_existence1 N v6 previous mul) := by
    apply row_51
    exact row_36
    exact row_30
  -- chapter_118_line_27: GL tag expansion.
  have row_27 : (¬ (∀ (v8 : α), ((N v8) → (¬ (mul v6 previous v8))))) := by
    simpa only [gl_existence1] using (row_28)
  have exists_row_27 : ∃ (v8 : α), ((N v8) ∧ (mul v6 previous v8)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_27
      intro v8 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v8, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v8, witness_row_27⟩ := exists_row_27
  -- chapter_118_line_26: GL tag disintegration.
  have row_26 : (mul v6 previous v8) := by
    exact witness_row_27.2
  -- chapter_118_line_25: GL tag expansion.
  have row_25 : (¬ (∀ (v14 : α), ((N v14) → (¬ (mul v4 previous v14))))) := by
    simpa only [gl_existence1] using (row_29)
  have exists_row_25 : ∃ (v14 : α), ((N v14) ∧ (mul v4 previous v14)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_25
      intro v14 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v14, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v14, witness_row_25⟩ := exists_row_25
  -- chapter_118_line_24: GL tag disintegration.
  have row_24 : (N v14) := by
    exact witness_row_25.1
  -- chapter_118_line_23: GL tag disintegration.
  have row_23 : (mul v4 previous v14) := by
    exact witness_row_25.2
  -- chapter_118_line_22: GL tag implication.
  have row_22 : (gl_existence1 N v14 v2 mul) := by
    apply row_51
    exact row_24
    exact row_35
  -- chapter_118_line_21: GL tag expansion.
  have row_21 : (¬ (∀ (v15 : α), ((N v15) → (¬ (mul v14 v2 v15))))) := by
    simpa only [gl_existence1] using (row_22)
  have exists_row_21 : ∃ (v15 : α), ((N v15) ∧ (mul v14 v2 v15)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_21
      intro v15 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v15, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v15, witness_row_21⟩ := exists_row_21
  -- chapter_118_line_20: GL tag disintegration.
  have row_20 : (mul v14 v2 v15) := by
    exact witness_row_21.2
  -- chapter_118_line_19: GL tag recursion.
  have row_19 : (∀ (w1 : α) (w2 : α), ((mul w1 v2 w2) → ((mul v4 previous w1) → ((mul v4 v2 v6) → (mul v6 previous w2))))) := by
    exact assumption_19
  -- chapter_118_line_18: GL tag implication.
  have row_18 : (mul v6 previous v15) := by
    apply row_19
    exact row_20
    exact row_23
    exact row_39
  -- chapter_118_line_17: GL tag implication.
  have row_17 : (v15 = v8) := by
    apply row_63
    exact row_36
    exact row_30
    exact row_18
    exact row_26
  -- chapter_118_line_16: GL tag equality1.
  have row_16 : (mul v14 v2 v8) := by
    have equality_source := row_20
    have equality_step_1 := row_17
    cases equality_step_1
    exact equality_source
  -- chapter_118_line_15: GL tag implication.
  have row_15 : (add v14 v4 v1) := by
    apply row_49
    exact row_30
    exact row_72
    exact row_23
    exact row_34
  -- chapter_118_line_14: GL tag theorem.
  have row_14 := peano_source_036 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_118_line_13: GL tag implication.
  have row_13 : (add v4 v14 v1) := by
    apply row_14
    exact row_15
  -- chapter_118_line_12: GL tag task formulation.
  have row_12 : (mul v1 v2 v3) := by
    exact assumption_12
  -- chapter_118_line_11: GL tag theorem.
  have row_11 := peano_source_055 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_118_line_10: GL tag implication.
  have row_10 : (add v6 v8 v3) := by
    apply row_11
    exact row_13
    exact row_39
    exact row_16
    exact row_12
  -- chapter_118_line_9: GL tag implication.
  have row_9 : (gl_existence1 N previous zero add) := by
    apply row_45
    exact row_30
    exact row_44
  -- chapter_118_line_8: GL tag expansion.
  have row_8 : (¬ (∀ (v7 : α), ((N v7) → (¬ (add previous zero v7))))) := by
    simpa only [gl_existence1] using (row_9)
  have exists_row_8 : ∃ (v7 : α), ((N v7) ∧ (add previous zero v7)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_8
      intro v7 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v7, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v7, witness_row_8⟩ := exists_row_8
  -- chapter_118_line_7: GL tag disintegration.
  have row_7 : (add previous zero v7) := by
    exact witness_row_8.2
  -- chapter_118_line_6: GL tag implication.
  have row_6 : (succ v7 v5) := by
    apply row_42
    exact row_44
    exact row_41
    exact row_7
    exact row_70
  -- chapter_118_line_5: GL tag theorem.
  have row_5 := peano_source_011 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_118_line_4: GL tag implication.
  have row_4 : (previous = v7) := by
    apply row_5
    exact row_7
    exact row_30
  -- chapter_118_line_3: GL tag equality1.
  have row_3 : (mul v6 v7 v8) := by
    have equality_source := row_26
    have equality_step_1 := row_4
    cases equality_step_1
    exact equality_source
  -- chapter_118_line_2: GL tag theorem.
  have row_2 := peano_source_053 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_118_line_1: GL tag implication.
  have row_1 : (mul v6 v5 v3) := by
    apply row_2
    exact row_10
    exact row_6
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α) (v3 : α), ((mul v1 v2 v3) → (∀ (v4 : α) (v5 : α), ((mul v4 v5 v1) → (∀ (v6 : α), ((mul v4 v2 v6) → (mul v6 v5 v3))))))) := by
  intro v1
  intro v2
  intro v3
  intro premise_1
  intro v4
  intro v5
  intro premise_2
  intro v6
  intro premise_3
  have inductionMember : N v5 := by
    have typingRule := peano_source_056_induction_typing N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact typingRule v1 v4 v5 premise_2
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v1 : α) (v2 : α) (v3 : α), ((mul v1 v2 v3) → (∀ (v4 : α), ((mul v4 zero v1) → (∀ (v6 : α), ((mul v4 v2 v6) → (mul v6 zero v3))))))) := by
    intro v1
    intro v2
    intro v3
    intro base_premise_1
    intro v4
    intro base_premise_2
    intro v6
    intro base_premise_3
    have zeroRule := peano_source_056_check_zero N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact zeroRule v1 v2 v3 v4 zero v6 rfl base_premise_2 base_premise_1 base_premise_3
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v1 : α) (v2 : α) (v3 : α), ((mul v1 v2 v3) → (∀ (v4 : α), ((mul v4 induction_n v1) → (∀ (v6 : α), ((mul v4 v2 v6) → (mul v6 induction_n v3))))))) → ∀ induction_m, succ induction_n induction_m → (∀ (v1 : α) (v2 : α) (v3 : α), ((mul v1 v2 v3) → (∀ (v4 : α), ((mul v4 induction_m v1) → (∀ (v6 : α), ((mul v4 v2 v6) → (mul v6 induction_m v3))))))) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro v1
    intro v2
    intro v3
    intro step_premise_1
    intro v4
    intro step_premise_2
    intro v6
    intro step_premise_3
    have step_induction_assumption_1 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have step_induction_assumption_2 :
        (∀ (w1 : α) (w2 : α), ((mul w1 v2 w2) → ((mul v4 induction_n w1) → ((mul v4 v2 v6) → (mul v6 induction_n w2))))) := by
      intro w1
      intro w2
      intro step_induction_assumption_2_premise_1
      intro step_induction_assumption_2_premise_2
      intro step_induction_assumption_2_premise_3
      apply induction_hypothesis
      all_goals assumption
    have stepRule := peano_source_056_check_induction_condition N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact stepRule induction_n v1 v2 v3 v4 induction_m v6 step_induction_assumption_1 step_premise_3 step_premise_2 step_induction_assumption_2 step_premise_1
  have inductionProperty : (∀ (v1 : α) (v2 : α) (v3 : α), ((mul v1 v2 v3) → (∀ (v4 : α), ((mul v4 v5 v1) → (∀ (v6 : α), ((mul v4 v2 v6) → (mul v6 v5 v3))))))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v1 : α) (v2 : α) (v3 : α), ((mul v1 v2 v3) → (∀ (v4 : α), ((mul v4 induction_value v1) → (∀ (v6 : α), ((mul v4 v2 v6) → (mul v6 induction_value v3))))))))
      v5
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v1 v2 v3 premise_1 v4 premise_2 v6 premise_3

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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (assumption_2 : (mul v1 v2 one))
    : (N v1) := by
  -- chapter_123_line_10: GL tag task formulation.
  have row_10 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_123_line_9: GL tag expansion.
  have row_9 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_10)
  -- chapter_123_line_8: GL tag disintegration.
  have row_8 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_9.1
  -- chapter_123_line_7: GL tag expansion.
  have row_7 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_8)
  -- chapter_123_line_6: GL tag disintegration.
  have row_6 : (gl_fXYZ mul N N N) := by
    exact row_7.1.1.1.1.2
  -- chapter_123_line_5: GL tag expansion.
  have row_5 : (((((gl_implication11 mul N) ∧ (gl_implication12 mul N)) ∧ (gl_implication13 mul N)) ∧ (gl_implication18 N N N mul)) ∧ (gl_implication19 N N mul)) := by
    simpa only [gl_fXYZ] using (row_6)
  -- chapter_123_line_4: GL tag disintegration.
  have row_4 : (gl_implication11 mul N) := by
    exact row_5.1.1.1.1
  -- chapter_123_line_3: GL tag expansion.
  have row_3 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_4)
  -- chapter_123_line_2: GL tag task formulation.
  have row_2 : (mul v1 v2 one) := by
    exact assumption_2
  -- chapter_123_line_1: GL tag implication.
  have row_1 : (N v1) := by
    apply row_3
    exact row_2
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v1 : α)
    (v2 : α)
    (assumption_22 : (mul v1 v2 one))
    (assumption_14 : (v1 = zero))
    : (mul v2 v1 one) := by
  -- chapter_124_line_49: GL tag task formulation.
  have row_49 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_124_line_48: GL tag expansion.
  have row_48 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_49)
  -- chapter_124_line_47: GL tag disintegration.
  have row_47 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_48.1
  -- chapter_124_line_46: GL tag expansion.
  have row_46 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_47)
  -- chapter_124_line_45: GL tag disintegration.
  have row_45 : (gl_fXYZ mul N N N) := by
    exact row_46.1.1.1.1.2
  -- chapter_124_line_44: GL tag expansion.
  have row_44 : (((((gl_implication11 mul N) ∧ (gl_implication12 mul N)) ∧ (gl_implication13 mul N)) ∧ (gl_implication18 N N N mul)) ∧ (gl_implication19 N N mul)) := by
    simpa only [gl_fXYZ] using (row_45)
  -- chapter_124_line_43: GL tag disintegration.
  have row_43 : (gl_implication13 mul N) := by
    exact row_44.1.1.2
  -- chapter_124_line_42: GL tag expansion.
  have row_42 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_43)
  -- chapter_124_line_41: GL tag disintegration.
  have row_41 : (gl_implication12 mul N) := by
    exact row_44.1.1.1.2
  -- chapter_124_line_40: GL tag expansion.
  have row_40 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_41)
  -- chapter_124_line_39: GL tag disintegration.
  have row_39 : (gl_implication11 mul N) := by
    exact row_44.1.1.1.1
  -- chapter_124_line_38: GL tag expansion.
  have row_38 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_39)
  -- chapter_124_line_37: GL tag disintegration.
  have row_37 : (gl_implication18 N N N mul) := by
    exact row_44.1.2
  -- chapter_124_line_36: GL tag expansion.
  have row_36 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication18] using (row_37)
  -- chapter_124_line_35: GL tag disintegration.
  have row_35 : (gl_implication25 N zero mul) := by
    exact row_46.1.1.1.2
  -- chapter_124_line_34: GL tag expansion.
  have row_34 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((mul w1 zero w2) → (w2 = zero))))) := by
    simpa only [gl_implication25] using (row_35)
  -- chapter_124_line_33: GL tag disintegration.
  have row_33 : (gl_implication19 N N mul) := by
    exact row_44.2
  -- chapter_124_line_32: GL tag expansion.
  have row_32 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (∀ (w3 : α), ((mul w1 w2 w3) → (∀ (w4 : α), ((mul w1 w2 w4) → (w3 = w4))))))))) := by
    simpa only [gl_implication19] using (row_33)
  -- chapter_124_line_31: GL tag disintegration.
  have row_31 : (succ zero one) := by
    exact row_48.2
  -- chapter_124_line_30: GL tag disintegration.
  have row_30 : (gl_implication9 N zero succ) := by
    exact row_46.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_124_line_29: GL tag expansion.
  have row_29 : (∀ (w1 : α), ((N w1) → (¬ (succ w1 zero)))) := by
    simpa only [gl_implication9] using (row_30)
  -- chapter_124_line_28: GL tag disintegration.
  have row_28 : (gl_fXY succ N N) := by
    exact row_46.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_124_line_27: GL tag expansion.
  have row_27 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_28)
  -- chapter_124_line_26: GL tag disintegration.
  have row_26 : (gl_implication6 N N succ) := by
    exact row_27.1.2
  -- chapter_124_line_25: GL tag expansion.
  have row_25 : (∀ (w1 : α), ((N w1) → (gl_existence0 N w1 succ))) := by
    simpa only [gl_implication6] using (row_26)
  -- chapter_124_line_24: GL tag disintegration.
  have row_24 : (N zero) := by
    exact row_46.1.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_124_line_23: GL tag implication grouping.
  have row_23 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_38
    have auxiliary_source_2 := row_40
    have auxiliary_source_3 := row_42
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_124_line_22: GL tag task formulation.
  have row_22 : (mul v1 v2 one) := by
    exact assumption_22
  -- chapter_124_line_21: GL tag implication.
  have row_21 : (gl_and21 v1 N v2 N one N) := by
    apply row_23
    exact row_22
  -- chapter_124_line_20: GL tag expansion.
  have row_20 : (((N v1) ∧ (N v2)) ∧ (N one)) := by
    simpa only [gl_and21] using (row_21)
  -- chapter_124_line_19: GL tag disintegration.
  have row_19 : (N v1) := by
    exact row_20.1.1
  -- chapter_124_line_18: GL tag disintegration.
  have row_18 : (N v2) := by
    exact row_20.1.2
  -- chapter_124_line_17: GL tag implication.
  have row_17 : (gl_existence1 N v2 v1 mul) := by
    apply row_36
    exact row_18
    exact row_19
  -- chapter_124_line_16: GL tag expansion.
  have row_16 : (¬ (∀ (v7 : α), ((N v7) → (¬ (mul v2 v1 v7))))) := by
    simpa only [gl_existence1] using (row_17)
  have exists_row_16 : ∃ (v7 : α), ((N v7) ∧ (mul v2 v1 v7)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_16
      intro v7 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v7, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v7, witness_row_16⟩ := exists_row_16
  -- chapter_124_line_15: GL tag disintegration.
  have row_15 : (mul v2 v1 v7) := by
    exact witness_row_16.2
  -- chapter_124_line_14: GL tag recursion.
  have row_14 : (v1 = zero) := by
    exact assumption_14
  -- chapter_124_line_13: GL tag equality1.
  have row_13 : (mul v2 zero v7) := by
    have equality_source := row_15
    have equality_step_1 := row_14
    cases equality_step_1
    exact equality_source
  -- chapter_124_line_12: GL tag implication.
  have row_12 : (v7 = zero) := by
    apply row_34
    exact row_18
    exact row_13
  -- chapter_124_line_11: GL tag theorem.
  have row_11 := peano_source_058 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_124_line_10: GL tag implication.
  have row_10 : (mul zero v2 v7) := by
    apply row_11
    exact row_13
  -- chapter_124_line_9: GL tag symmetry of equality.
  have row_9 : (zero = v1) := by
    exact Eq.symm row_14
  -- chapter_124_line_8: GL tag equality1.
  have row_8 : (mul v1 v2 zero) := by
    have equality_source := row_10
    have equality_step_1 := row_9
    cases equality_step_1
    have equality_step_2 := row_12
    cases equality_step_2
    exact equality_source
  -- chapter_124_line_7: GL tag implication.
  have row_7 : (one = zero) := by
    apply row_32
    exact row_19
    exact row_18
    exact row_22
    exact row_8
  -- chapter_124_line_6: GL tag equality1.
  have row_6 : (succ zero zero) := by
    have equality_source := row_31
    have equality_step_1 := row_7
    cases equality_step_1
    exact equality_source
  -- chapter_124_line_5: GL tag implication grouping.
  have row_5 : (∀ (w1 : α), ((N w1) → (gl_and3 N w1 succ zero))) := by
    have auxiliary_source_1 := row_25
    have auxiliary_source_2 := row_29
    intro auxiliary_argument_1
    intro auxiliary_premise
    exact ⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_premise)⟩
  -- chapter_124_line_4: GL tag implication.
  have row_4 : (gl_and3 N zero succ zero) := by
    apply row_5
    exact row_24
  -- chapter_124_line_3: GL tag expansion.
  have row_3 : ((gl_existence0 N zero succ) ∧ (¬ (succ zero zero))) := by
    simpa only [gl_and3] using (row_4)
  -- chapter_124_line_2: GL tag disintegration.
  have row_2 : (¬ (succ zero zero)) := by
    exact row_3.2
  -- chapter_124_line_1: GL tag vacuous truth.
  have row_1 : (mul v2 v1 one) := by
    exact False.elim (row_2 row_6)
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (previous : α)
    (v1 : α)
    (v2 : α)
    (assumption_49 : (succ previous v1))
    (assumption_20 : (mul v1 v2 one))
    : (mul v2 v1 one) := by
  -- chapter_125_line_50: GL tag task formulation.
  have row_50 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_125_line_49: GL tag recursion.
  have row_49 : (succ previous v1) := by
    exact assumption_49
  -- chapter_125_line_48: GL tag theorem.
  have row_48 := peano_source_027 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_125_line_47: GL tag implication.
  have row_47 : (add previous one v1) := by
    apply row_48
    exact row_49
  -- chapter_125_line_46: GL tag expansion.
  have row_46 : ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) := by
    simpa only [gl_AnchorPeano] using (row_50)
  -- chapter_125_line_45: GL tag disintegration.
  have row_45 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_46.1
  -- chapter_125_line_44: GL tag expansion.
  have row_44 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_45)
  -- chapter_125_line_43: GL tag disintegration.
  have row_43 : (gl_fXYZ mul N N N) := by
    exact row_44.1.1.1.1.2
  -- chapter_125_line_42: GL tag expansion.
  have row_42 : (((((gl_implication11 mul N) ∧ (gl_implication12 mul N)) ∧ (gl_implication13 mul N)) ∧ (gl_implication18 N N N mul)) ∧ (gl_implication19 N N mul)) := by
    simpa only [gl_fXYZ] using (row_43)
  -- chapter_125_line_41: GL tag disintegration.
  have row_41 : (gl_implication13 mul N) := by
    exact row_42.1.1.2
  -- chapter_125_line_40: GL tag expansion.
  have row_40 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_41)
  -- chapter_125_line_39: GL tag disintegration.
  have row_39 : (gl_implication12 mul N) := by
    exact row_42.1.1.1.2
  -- chapter_125_line_38: GL tag expansion.
  have row_38 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_39)
  -- chapter_125_line_37: GL tag disintegration.
  have row_37 : (gl_implication11 mul N) := by
    exact row_42.1.1.1.1
  -- chapter_125_line_36: GL tag expansion.
  have row_36 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_37)
  -- chapter_125_line_35: GL tag disintegration.
  have row_35 : (gl_implication18 N N N mul) := by
    exact row_42.1.2
  -- chapter_125_line_34: GL tag expansion.
  have row_34 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication18] using (row_35)
  -- chapter_125_line_33: GL tag disintegration.
  have row_33 : (gl_fXY succ N N) := by
    exact row_44.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_125_line_32: GL tag expansion.
  have row_32 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_33)
  -- chapter_125_line_31: GL tag disintegration.
  have row_31 : (gl_implication0 succ N) := by
    exact row_32.1.1.1
  -- chapter_125_line_30: GL tag expansion.
  have row_30 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_31)
  -- chapter_125_line_29: GL tag disintegration.
  have row_29 : (gl_fXYZ add N N N) := by
    exact row_44.1.1.1.1.1.1.1.1.1.2
  -- chapter_125_line_28: GL tag expansion.
  have row_28 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_29)
  -- chapter_125_line_27: GL tag disintegration.
  have row_27 : (gl_implication18 N N N add) := by
    exact row_28.1.2
  -- chapter_125_line_26: GL tag expansion.
  have row_26 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication18] using (row_27)
  -- chapter_125_line_25: GL tag disintegration.
  have row_25 : (N zero) := by
    exact row_44.1.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_125_line_24: GL tag disintegration.
  have row_24 : (gl_implication23 N succ add) := by
    exact row_44.1.1.1.1.1.1.2
  -- chapter_125_line_23: GL tag expansion.
  have row_23 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication23] using (row_24)
  -- chapter_125_line_22: GL tag disintegration.
  have row_22 : (succ zero one) := by
    exact row_46.2
  -- chapter_125_line_21: GL tag implication grouping.
  have row_21 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_36
    have auxiliary_source_2 := row_38
    have auxiliary_source_3 := row_40
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_125_line_20: GL tag task formulation.
  have row_20 : (mul v1 v2 one) := by
    exact assumption_20
  -- chapter_125_line_19: GL tag implication.
  have row_19 : (gl_and21 v1 N v2 N one N) := by
    apply row_21
    exact row_20
  -- chapter_125_line_18: GL tag expansion.
  have row_18 : (((N v1) ∧ (N v2)) ∧ (N one)) := by
    simpa only [gl_and21] using (row_19)
  -- chapter_125_line_17: GL tag disintegration.
  have row_17 : (N v2) := by
    exact row_18.1.2
  -- chapter_125_line_16: GL tag implication.
  have row_16 : (N previous) := by
    apply row_30
    exact row_49
  -- chapter_125_line_15: GL tag implication.
  have row_15 : (gl_existence1 N v2 previous mul) := by
    apply row_34
    exact row_17
    exact row_16
  -- chapter_125_line_14: GL tag implication.
  have row_14 : (gl_existence1 N previous zero add) := by
    apply row_26
    exact row_16
    exact row_25
  -- chapter_125_line_13: GL tag expansion.
  have row_13 : (¬ (∀ (v3 : α), ((N v3) → (¬ (add previous zero v3))))) := by
    simpa only [gl_existence1] using (row_14)
  have exists_row_13 : ∃ (v3 : α), ((N v3) ∧ (add previous zero v3)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_13
      intro v3 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v3, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v3, witness_row_13⟩ := exists_row_13
  -- chapter_125_line_12: GL tag disintegration.
  have row_12 : (add previous zero v3) := by
    exact witness_row_13.2
  -- chapter_125_line_11: GL tag implication.
  have row_11 : (succ v3 v1) := by
    apply row_23
    exact row_25
    exact row_22
    exact row_12
    exact row_47
  -- chapter_125_line_10: GL tag theorem.
  have row_10 := peano_source_011 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_125_line_9: GL tag implication.
  have row_9 : (previous = v3) := by
    apply row_10
    exact row_12
    exact row_16
  -- chapter_125_line_8: GL tag expansion.
  have row_8 : (¬ (∀ (v4 : α), ((N v4) → (¬ (mul v2 previous v4))))) := by
    simpa only [gl_existence1] using (row_15)
  have exists_row_8 : ∃ (v4 : α), ((N v4) ∧ (mul v2 previous v4)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_8
      intro v4 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v4, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v4, witness_row_8⟩ := exists_row_8
  -- chapter_125_line_7: GL tag disintegration.
  have row_7 : (mul v2 previous v4) := by
    exact witness_row_8.2
  -- chapter_125_line_6: GL tag equality1.
  have row_6 : (mul v2 v3 v4) := by
    have equality_source := row_7
    have equality_step_1 := row_9
    cases equality_step_1
    exact equality_source
  -- chapter_125_line_5: GL tag anchor handling.
  have row_5 : (gl_AnchorPeano N zero succ add mul one) := by
    exact row_50
  -- chapter_125_line_4: GL tag theorem.
  have row_4 := peano_source_052 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_125_line_3: GL tag implication.
  have row_3 : (add v2 v4 one) := by
    apply row_4
    exact row_6
    exact row_20
    exact row_11
  -- chapter_125_line_2: GL tag theorem.
  have row_2 := peano_source_053 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_125_line_1: GL tag implication.
  have row_1 : (mul v2 v1 one) := by
    apply row_2
    exact row_3
    exact row_11
    exact row_6
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((mul v1 v2 one) → (mul v2 v1 one))) := by
  intro v1
  intro v2
  intro premise_1
  have inductionMember : N v1 := by
    have typingRule := peano_source_059_induction_typing N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact typingRule v1 v2 premise_1
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorPeano, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v2 : α), ((mul zero v2 one) → (mul v2 zero one))) := by
    intro v2
    intro base_premise_1
    have zeroRule := peano_source_059_check_zero N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact zeroRule zero v2 base_premise_1 rfl
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v2 : α), ((mul induction_n v2 one) → (mul v2 induction_n one))) → ∀ induction_m, succ induction_n induction_m → (∀ (v2 : α), ((mul induction_m v2 one) → (mul v2 induction_m one))) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro v2
    intro step_premise_1
    have step_induction_assumption_1 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have stepRule := peano_source_059_check_induction_condition N zero succ add mul one anchor relationalInduction strongRelationalInduction
    exact stepRule induction_n induction_m v2 step_induction_assumption_1 step_premise_1
  have inductionProperty : (∀ (v2 : α), ((mul v1 v2 one) → (mul v2 v1 one))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v2 : α), ((mul induction_value v2 one) → (mul v2 induction_value one))))
      v1
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v2 premise_1

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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
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
  -- chapter_119_line_9: GL tag task formulation.
  have row_9 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_119_line_8: GL tag theorem.
  have row_8 := peano_source_054 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_119_line_7: GL tag task formulation.
  have row_7 : (mul v2 v4 v5) := by
    exact premise_2
  -- chapter_119_line_6: GL tag implication.
  have row_6 : (mul v4 v2 v5) := by
    apply row_8
    exact row_7
  -- chapter_119_line_5: GL tag task formulation.
  have row_5 : (mul v6 v4 v1) := by
    exact premise_3
  -- chapter_119_line_4: GL tag implication.
  have row_4 : (mul v4 v6 v1) := by
    apply row_8
    exact row_5
  -- chapter_119_line_3: GL tag task formulation.
  have row_3 : (mul v1 v2 v3) := by
    exact premise_1
  -- chapter_119_line_2: GL tag theorem.
  have row_2 := peano_source_056 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_119_line_1: GL tag implication.
  have row_1 : (mul v5 v6 v3) := by
    apply row_2
    exact row_3
    exact row_4
    exact row_6
  exact row_1

theorem peano_source_066
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α) (v3 : α), ((mul v1 v2 v3) → (∀ (v4 : α) (v5 : α), ((mul v4 v5 v1) → (∀ (v6 : α), ((mul v2 v5 v6) → (mul v6 v4 v3))))))) := by
  intro v1
  intro v2
  intro v3
  intro premise_1
  intro v4
  intro v5
  intro premise_2
  intro v6
  intro premise_3
  -- chapter_136_line_9: GL tag task formulation.
  have row_9 : (gl_AnchorPeano N zero succ add mul one) := by
    exact anchor
  -- chapter_136_line_8: GL tag theorem.
  have row_8 := peano_source_054 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_136_line_7: GL tag task formulation.
  have row_7 : (mul v2 v5 v6) := by
    exact premise_3
  -- chapter_136_line_6: GL tag implication.
  have row_6 : (mul v5 v2 v6) := by
    apply row_8
    exact row_7
  -- chapter_136_line_5: GL tag task formulation.
  have row_5 : (mul v4 v5 v1) := by
    exact premise_2
  -- chapter_136_line_4: GL tag implication.
  have row_4 : (mul v5 v4 v1) := by
    apply row_8
    exact row_5
  -- chapter_136_line_3: GL tag task formulation.
  have row_3 : (mul v1 v2 v3) := by
    exact premise_1
  -- chapter_136_line_2: GL tag theorem.
  have row_2 := peano_source_056 N zero succ add mul one anchor relationalInduction strongRelationalInduction
  -- chapter_136_line_1: GL tag implication.
  have row_1 : (mul v6 v4 v3) := by
    apply row_2
    exact row_3
    exact row_4
    exact row_6
  exact row_1

end GLExport
