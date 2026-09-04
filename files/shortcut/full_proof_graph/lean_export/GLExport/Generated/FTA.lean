/-
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-/

import GLExport.Generated.Definitions

set_option linter.unusedVariables false

namespace GLExport.FTA

universe u

private theorem anchorPeanoOfFTA
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    : GLExport.gl_AnchorPeano N zero succ add mul one := by
  simp only [gl_AnchorFTA, GLExport.gl_AnchorPeano] at anchor ⊢
  exact anchor.1.1

private theorem anchorGaussOfFTA
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    : GLExport.gl_AnchorGauss N zero succ add mul one two identity := by
  simp only [gl_AnchorFTA, GLExport.gl_AnchorGauss] at anchor ⊢
  exact anchor

theorem fta_source_001
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_006 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (x_4 x_8 x_7 x_9))))))
    (external_peano_externals_36_004 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (∀ (x_10 : α), ((x_4 x_7 x_10 x_9) → (x_8 = x_10))))))))
    : (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α), ((add v4 v2 v3) → (v1 = v4))))) := by
  intro v1
  intro v2
  intro v3
  intro premise_1
  intro v4
  intro premise_2
  -- chapter_1_line_15: GL tag task formulation.
  have row_15 : (add v1 v2 v3) := by
    exact premise_1
  -- chapter_1_line_12: GL tag variable copy.
  have row_12 : (v3 = v3) := by
    rfl
  -- chapter_1_line_14: GL tag equality1.
  have row_14 : (add v1 v2 v3) := by
    have equality_source := row_15
    have equality_step_1 := row_12
    cases equality_step_1
    exact equality_source
  -- chapter_1_line_11: GL tag task formulation.
  have row_11 : (add v4 v2 v3) := by
    exact premise_2
  -- chapter_1_line_10: GL tag equality1.
  have row_10 : (add v4 v2 v3) := by
    have equality_source := row_11
    have equality_step_1 := row_12
    cases equality_step_1
    exact equality_source
  -- chapter_1_line_7: GL tag task formulation.
  have row_7 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_1_line_6: GL tag expansion.
  have row_6 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_7
  -- chapter_1_line_8: GL tag disintegration.
  have row_8 : (succ zero one) := by
    exact row_6.1.1.2
  -- chapter_1_line_5: GL tag disintegration.
  have row_5 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_6.1.1.1
  -- chapter_1_line_4: GL tag expansion for integration.
  have row_4 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_1_line_3: GL tag reformulation for integration and.
  have row_3 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_1_line_2: GL tag implication.
  have row_2 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_3
    exact row_5
    exact row_8
  have rule_row_13 := external_peano_externals_36_006 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_1_line_13: GL tag implication.
  have row_13 : (add v2 v1 v3) := by
    apply rule_row_13
    exact row_14
  have rule_row_9 := external_peano_externals_36_006 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_1_line_9: GL tag implication.
  have row_9 : (add v2 v4 v3) := by
    apply rule_row_9
    exact row_10
  have rule_row_1 := external_peano_externals_36_004 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_1_line_1: GL tag implication.
  have row_1 : (v1 = v4) := by
    apply rule_row_1
    exact row_13
    exact row_9
  exact row_1

theorem fta_source_002
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    : (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (gl_preorder N add v1 v3))) := by
  intro v1
  intro v2
  intro v3
  intro premise_1
  -- chapter_2_line_13: GL tag task formulation.
  have row_13 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_2_line_12: GL tag expansion.
  have row_12 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_13
  -- chapter_2_line_11: GL tag disintegration.
  have row_11 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_12.1.1.1
  -- chapter_2_line_10: GL tag expansion.
  have row_10 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_11
  -- chapter_2_line_9: GL tag disintegration.
  have row_9 : (gl_fXYZ add N N N) := by
    exact row_10.1.1.1.1.1.1.1.1.2
  -- chapter_2_line_8: GL tag expansion.
  have row_8 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_9
  -- chapter_2_line_7: GL tag disintegration.
  have row_7 : (gl_implication9 add N) := by
    exact row_8.1.1.1.2
  -- chapter_2_line_6: GL tag expansion.
  have row_6 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication9] using row_7
  -- chapter_2_line_4: GL tag task formulation.
  have row_4 : (add v1 v2 v3) := by
    exact premise_1
  -- chapter_2_line_5: GL tag implication.
  have row_5 : (N v2) := by
    apply row_6
    exact row_4
  -- chapter_2_line_3: GL tag expansion for integration.
  have row_3 : ((gl_preorder N add v1 v3) ↔ (¬ (∀ (v5 : α), ((N v5) → (¬ (add v1 v5 v3)))))) := by
    exact Iff.rfl
  -- chapter_2_line_2: GL tag reformulation for integration >[bound].
  have row_2 : (∀ (v4 : α), ((N v4) → ((add v1 v4 v3) → (gl_preorder N add v1 v3)))) := by
    intro v4
    intro integration_premise_1
    intro integration_premise_2
    apply (row_3).2
    intro universal_counterexample
    exact universal_counterexample v4 integration_premise_1 integration_premise_2
  -- chapter_2_line_1: GL tag implication.
  have row_1 : (gl_preorder N add v1 v3) := by
    apply row_2
    exact row_5
    exact row_4
  exact row_1

theorem fta_source_003
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_008 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (∀ (w4 : α) (w5 : α), ((mul w4 w1 w5) → (∀ (w6 : α), ((mul w4 w2 w6) → (∀ (w7 : α), ((mul w4 w3 w7) → (add w5 w6 w7))))))))))))
    (external_peano_externals_36_004 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (∀ (x_10 : α), ((x_4 x_7 x_10 x_9) → (x_8 = x_10))))))))
    : (∀ (v1 : α) (v2 : α) (v3 : α), ((mul v1 v2 v3) → (∀ (v4 : α) (v5 : α), ((add v3 v4 v5) → (∀ (v6 : α), ((mul v1 v6 v5) → (∀ (v7 : α), ((add v2 v7 v6) → (mul v1 v7 v4))))))))) := by
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
  -- chapter_3_line_30: GL tag task formulation.
  have row_30 : (mul v1 v6 v5) := by
    exact premise_3
  -- chapter_3_line_28: GL tag task formulation.
  have row_28 : (add v3 v4 v5) := by
    exact premise_2
  -- chapter_3_line_26: GL tag expansion for integration.
  have row_26 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_3_line_25: GL tag reformulation for integration and.
  have row_25 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_3_line_22: GL tag task formulation.
  have row_22 : (mul v1 v2 v3) := by
    exact premise_1
  -- chapter_3_line_18: GL tag task formulation.
  have row_18 : (add v2 v7 v6) := by
    exact premise_4
  -- chapter_3_line_12: GL tag task formulation.
  have row_12 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_3_line_11: GL tag expansion.
  have row_11 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_12
  -- chapter_3_line_27: GL tag disintegration.
  have row_27 : (succ zero one) := by
    exact row_11.1.1.2
  -- chapter_3_line_10: GL tag disintegration.
  have row_10 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_11.1.1.1
  -- chapter_3_line_24: GL tag implication.
  have row_24 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_25
    exact row_10
    exact row_27
  -- chapter_3_line_9: GL tag expansion.
  have row_9 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_10
  -- chapter_3_line_17: GL tag disintegration.
  have row_17 : (gl_fXYZ add N N N) := by
    exact row_9.1.1.1.1.1.1.1.1.2
  -- chapter_3_line_16: GL tag expansion.
  have row_16 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_17
  -- chapter_3_line_15: GL tag disintegration.
  have row_15 : (gl_implication9 add N) := by
    exact row_16.1.1.1.2
  -- chapter_3_line_14: GL tag expansion.
  have row_14 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication9] using row_15
  -- chapter_3_line_13: GL tag implication.
  have row_13 : (N v7) := by
    apply row_14
    exact row_18
  -- chapter_3_line_8: GL tag disintegration.
  have row_8 : (gl_fXYZ mul N N N) := by
    exact row_9.1.1.1.2
  -- chapter_3_line_7: GL tag expansion.
  have row_7 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_8
  -- chapter_3_line_21: GL tag disintegration.
  have row_21 : (gl_implication8 mul N) := by
    exact row_7.1.1.1.1
  -- chapter_3_line_20: GL tag expansion.
  have row_20 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_21
  -- chapter_3_line_19: GL tag implication.
  have row_19 : (N v1) := by
    apply row_20
    exact row_22
  -- chapter_3_line_6: GL tag disintegration.
  have row_6 : (gl_implication13 N N N mul) := by
    exact row_7.1.2
  -- chapter_3_line_5: GL tag expansion.
  have row_5 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication13] using row_6
  -- chapter_3_line_4: GL tag implication.
  have row_4 : (gl_existence1 N v1 v7 mul) := by
    apply row_5
    exact row_19
    exact row_13
  -- chapter_3_line_3: GL tag expansion.
  have row_3 : (¬ (∀ (v8 : α), ((N v8) → (¬ (mul v1 v7 v8))))) := by
    simpa only [gl_existence1] using row_4
  have exists_row_3 : ∃ (v8 : α), ((N v8) ∧ (mul v1 v7 v8)) := existsAndOfNotForallImpNot row_3
  obtain ⟨v8, witness_row_3⟩ := exists_row_3
  -- chapter_3_line_2: GL tag disintegration.
  have row_2 : (mul v1 v7 v8) := by
    exact witness_row_3.2
  have rule_row_29 := external_peano_externals_36_008 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_3_line_29: GL tag implication.
  have row_29 : (add v3 v8 v5) := by
    apply rule_row_29
    exact row_18
    exact row_22
    exact row_2
    exact row_30
  have rule_row_23 := external_peano_externals_36_004 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_3_line_23: GL tag implication.
  have row_23 : (v8 = v4) := by
    apply rule_row_23
    exact row_29
    exact row_28
  -- chapter_3_line_1: GL tag equality1.
  have row_1 : (mul v1 v7 v4) := by
    have equality_source := row_2
    have equality_step_1 := row_23
    cases equality_step_1
    exact equality_source
  exact row_1

theorem fta_source_010
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_010 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_5 x_7 x_8 x_9) → (x_5 x_8 x_7 x_9))))))
    : (∀ (v1 : α) (v2 : α) (v3 : α), ((mul v1 v2 v3) → (gl_preorder N mul v2 v3))) := by
  intro v1
  intro v2
  intro v3
  intro premise_1
  -- chapter_12_line_12: GL tag task formulation.
  have row_12 : (mul v1 v2 v3) := by
    exact premise_1
  -- chapter_12_line_10: GL tag task formulation.
  have row_10 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_12_line_9: GL tag expansion.
  have row_9 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_10
  -- chapter_12_line_11: GL tag disintegration.
  have row_11 : (succ zero one) := by
    exact row_9.1.1.2
  -- chapter_12_line_8: GL tag disintegration.
  have row_8 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_9.1.1.1
  -- chapter_12_line_18: GL tag expansion.
  have row_18 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_8
  -- chapter_12_line_17: GL tag disintegration.
  have row_17 : (gl_fXYZ mul N N N) := by
    exact row_18.1.1.1.2
  -- chapter_12_line_16: GL tag expansion.
  have row_16 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_17
  -- chapter_12_line_15: GL tag disintegration.
  have row_15 : (gl_implication8 mul N) := by
    exact row_16.1.1.1.1
  -- chapter_12_line_14: GL tag expansion.
  have row_14 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_15
  -- chapter_12_line_13: GL tag implication.
  have row_13 : (N v1) := by
    apply row_14
    exact row_12
  -- chapter_12_line_7: GL tag expansion for integration.
  have row_7 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_12_line_6: GL tag reformulation for integration and.
  have row_6 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_12_line_5: GL tag implication.
  have row_5 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_6
    exact row_8
    exact row_11
  have rule_row_4 := external_peano_externals_36_010 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_12_line_4: GL tag implication.
  have row_4 : (mul v2 v1 v3) := by
    apply rule_row_4
    exact row_12
  -- chapter_12_line_3: GL tag expansion for integration.
  have row_3 : ((gl_preorder N mul v2 v3) ↔ (¬ (∀ (v5 : α), ((N v5) → (¬ (mul v2 v5 v3)))))) := by
    exact Iff.rfl
  -- chapter_12_line_2: GL tag reformulation for integration >[bound].
  have row_2 : (∀ (v4 : α), ((N v4) → ((mul v2 v4 v3) → (gl_preorder N mul v2 v3)))) := by
    intro v4
    intro integration_premise_1
    intro integration_premise_2
    apply (row_3).2
    intro universal_counterexample
    exact universal_counterexample v4 integration_premise_1 integration_premise_2
  -- chapter_12_line_1: GL tag implication.
  have row_1 : (gl_preorder N mul v2 v3) := by
    apply row_2
    exact row_13
    exact row_4
  exact row_1

theorem fta_source_011
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    : (∀ (v1 : α) (v2 : α), ((v1 = v2) → (¬ (gl_strictOrder N add v1 v2)))) := by
  intro v1
  intro v2
  intro premise_1
  -- chapter_13_line_5: GL tag task formulation.
  have row_5 : (v1 = v2) := by
    exact premise_1
  -- chapter_13_line_4: GL tag task formulation.
  have row_4 : ((gl_strictOrder N add v1 v2) → (gl_strictOrder N add v1 v2)) := by
    intro scope_premise_1
    exact scope_premise_1
  -- chapter_13_line_3: GL tag expansion.
  have row_3 : ((gl_strictOrder N add v1 v2) → ((gl_preorder N add v1 v2) ∧ (¬ (v1 = v2)))) := by
    simpa only [gl_strictOrder] using row_4
  -- chapter_13_line_2: GL tag disintegration.
  have row_2 : ((gl_strictOrder N add v1 v2) → (¬ (v1 = v2))) := by
    intro scope_premise_1
    have scoped_fact_1 := row_3 scope_premise_1
    exact scoped_fact_1.2
  -- chapter_13_line_1: GL tag contradiction.
  have row_1 : (¬ (gl_strictOrder N add v1 v2)) := by
    intro contradiction_assumption
    have scoped_contradiction := row_2 contradiction_assumption
    exact scoped_contradiction row_5
  exact row_1

theorem fta_source_013
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (v3 : α) (v4 : α), ((add v3 v4 v2) → (∀ (v5 : α), ((succ v5 v4) → (add v3 v5 v1))))))) := by
  intro v1
  intro v2
  intro premise_1
  intro v3
  intro v4
  intro premise_2
  intro v5
  intro premise_3
  -- chapter_15_line_26: GL tag task formulation.
  have row_26 : (succ v1 v2) := by
    exact premise_1
  -- chapter_15_line_22: GL tag task formulation.
  have row_22 : (succ v5 v4) := by
    exact premise_3
  -- chapter_15_line_16: GL tag task formulation.
  have row_16 : (add v3 v4 v2) := by
    exact premise_2
  -- chapter_15_line_12: GL tag task formulation.
  have row_12 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_15_line_11: GL tag expansion.
  have row_11 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_12
  -- chapter_15_line_10: GL tag disintegration.
  have row_10 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_11.1.1.1
  -- chapter_15_line_9: GL tag expansion.
  have row_9 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_10
  -- chapter_15_line_29: GL tag disintegration.
  have row_29 : (gl_implication17 N succ add) := by
    exact row_9.1.1.1.1.1.2
  -- chapter_15_line_28: GL tag expansion.
  have row_28 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication17] using row_29
  -- chapter_15_line_25: GL tag disintegration.
  have row_25 : (gl_implication7 N succ) := by
    exact row_9.1.1.1.1.1.1.1.1.1.2
  -- chapter_15_line_24: GL tag expansion.
  have row_24 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w2 w1) → (∀ (w3 : α), ((succ w3 w1) → (w2 = w3))))))) := by
    simpa only [gl_implication7] using row_25
  -- chapter_15_line_21: GL tag disintegration.
  have row_21 : (gl_fXY succ N N) := by
    exact row_9.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_15_line_20: GL tag expansion.
  have row_20 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_21
  -- chapter_15_line_19: GL tag disintegration.
  have row_19 : (gl_implication0 succ N) := by
    exact row_20.1.1.1
  -- chapter_15_line_18: GL tag expansion.
  have row_18 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_19
  -- chapter_15_line_17: GL tag implication.
  have row_17 : (N v5) := by
    apply row_18
    exact row_22
  -- chapter_15_line_8: GL tag disintegration.
  have row_8 : (gl_fXYZ add N N N) := by
    exact row_9.1.1.1.1.1.1.1.1.2
  -- chapter_15_line_7: GL tag expansion.
  have row_7 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_8
  -- chapter_15_line_32: GL tag disintegration.
  have row_32 : (gl_implication10 add N) := by
    exact row_7.1.1.2
  -- chapter_15_line_31: GL tag expansion.
  have row_31 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication10] using row_32
  -- chapter_15_line_30: GL tag implication.
  have row_30 : (N v2) := by
    apply row_31
    exact row_16
  -- chapter_15_line_15: GL tag disintegration.
  have row_15 : (gl_implication8 add N) := by
    exact row_7.1.1.1.1
  -- chapter_15_line_14: GL tag expansion.
  have row_14 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_15
  -- chapter_15_line_13: GL tag implication.
  have row_13 : (N v3) := by
    apply row_14
    exact row_16
  -- chapter_15_line_6: GL tag disintegration.
  have row_6 : (gl_implication13 N N N add) := by
    exact row_7.1.2
  -- chapter_15_line_5: GL tag expansion.
  have row_5 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication13] using row_6
  -- chapter_15_line_4: GL tag implication.
  have row_4 : (gl_existence1 N v3 v5 add) := by
    apply row_5
    exact row_13
    exact row_17
  -- chapter_15_line_3: GL tag expansion.
  have row_3 : (¬ (∀ (v6 : α), ((N v6) → (¬ (add v3 v5 v6))))) := by
    simpa only [gl_existence1] using row_4
  have exists_row_3 : ∃ (v6 : α), ((N v6) ∧ (add v3 v5 v6)) := existsAndOfNotForallImpNot row_3
  obtain ⟨v6, witness_row_3⟩ := exists_row_3
  -- chapter_15_line_2: GL tag disintegration.
  have row_2 : (add v3 v5 v6) := by
    exact witness_row_3.2
  -- chapter_15_line_27: GL tag implication.
  have row_27 : (succ v6 v2) := by
    apply row_28
    exact row_17
    exact row_22
    exact row_2
    exact row_16
  -- chapter_15_line_23: GL tag implication.
  have row_23 : (v6 = v1) := by
    apply row_24
    exact row_30
    exact row_27
    exact row_26
  -- chapter_15_line_1: GL tag equality1.
  have row_1 : (add v3 v5 v1) := by
    have equality_source := row_2
    have equality_step_1 := row_23
    cases equality_step_1
    exact equality_source
  exact row_1

theorem fta_source_014
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_gauss_externals_24_018 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α) (x_7 : α) (x_8 : GLBinaryRelation α), ((gl_AnchorGauss x_1 x_2 x_3 x_4 x_5 x_6 x_7 x_8) → (∀ (x_9 : α) (x_10 : α), ((x_3 x_9 x_10) → (gl_preorder x_1 x_4 x_9 x_10))))))
    (external_peano_externals_36_035 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (gl_existence11 N one succ))))
    (external_peano_externals_36_029 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((gl_preorder x_1 x_4 x_7 x_8) → ((gl_preorder x_1 x_4 x_8 x_7) → (x_7 = x_8)))))))
    (external_peano_externals_36_003 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (∀ (x_10 : α), ((x_3 x_8 x_10) → (∀ (x_11 : α), ((x_3 x_11 x_7) → (x_4 x_10 x_11 x_9))))))))))
    (external_peano_externals_36_005 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → ((x_8 = x_9) → (x_2 = x_7)))))))
    : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (v3 : α), ((gl_preorder N add v2 v3) → (¬ (v1 = v3)))))) := by
  intro v1
  intro v2
  intro premise_1
  intro v3
  intro premise_2
  intro reductio
  -- chapter_16_line_38: GL tag task formulation.
  have row_38 : (v1 = v3) := by
    exact reductio
  -- chapter_16_line_43: GL tag symmetry of equality.
  have row_43 : (v3 = v1) := by
    exact Eq.symm row_38
  -- chapter_16_line_37: GL tag task formulation.
  have row_37 : (succ v1 v2) := by
    exact premise_1
  -- chapter_16_line_36: GL tag equality1.
  have row_36 : (succ v3 v2) := by
    have equality_source := row_37
    have equality_step_1 := row_38
    cases equality_step_1
    exact equality_source
  -- chapter_16_line_33: GL tag expansion for integration.
  have row_33 : ((gl_AnchorGauss N zero succ add mul one two identity) ↔ ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity))) := by
    exact Iff.rfl
  -- chapter_16_line_32: GL tag reformulation for integration and.
  have row_32 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → ((succ one two) → ((gl_identity N identity) → (gl_AnchorGauss N zero succ add mul one two identity))))) := by
    intro integration_premise_1
    intro integration_premise_2
    intro integration_premise_3
    intro integration_premise_4
    simp only [gl_AnchorGauss]
    exact ⟨⟨⟨integration_premise_1, integration_premise_2⟩, integration_premise_3⟩, integration_premise_4⟩
  -- chapter_16_line_29: GL tag task formulation.
  have row_29 : (gl_preorder N add v2 v3) := by
    exact premise_2
  -- chapter_16_line_70: GL tag expansion.
  have row_70 : (¬ (∀ (v9 : α), ((N v9) → (¬ (add v2 v9 v3))))) := by
    simpa only [gl_preorder] using row_29
  have exists_row_70 : ∃ (v9 : α), ((N v9) ∧ (add v2 v9 v3)) := existsAndOfNotForallImpNot row_70
  obtain ⟨v9, witness_row_70⟩ := exists_row_70
  -- chapter_16_line_69: GL tag disintegration.
  have row_69 : (add v2 v9 v3) := by
    exact witness_row_70.2
  -- chapter_16_line_68: GL tag equality1.
  have row_68 : (add v2 v9 v1) := by
    have equality_source := row_69
    have equality_step_1 := row_43
    cases equality_step_1
    exact equality_source
  -- chapter_16_line_42: GL tag equality1.
  have row_42 : (gl_preorder N add v2 v1) := by
    have equality_source := row_29
    have equality_step_1 := row_43
    cases equality_step_1
    exact equality_source
  -- chapter_16_line_28: GL tag expansion for integration.
  have row_28 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_16_line_27: GL tag reformulation for integration and.
  have row_27 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_16_line_8: GL tag task formulation.
  have row_8 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_16_line_7: GL tag expansion.
  have row_7 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_8
  -- chapter_16_line_35: GL tag disintegration.
  have row_35 : (succ one two) := by
    exact row_7.1.2
  -- chapter_16_line_34: GL tag disintegration.
  have row_34 : (gl_identity N identity) := by
    exact row_7.2
  -- chapter_16_line_11: GL tag disintegration.
  have row_11 : (succ zero one) := by
    exact row_7.1.1.2
  -- chapter_16_line_6: GL tag disintegration.
  have row_6 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_7.1.1.1
  -- chapter_16_line_31: GL tag implication.
  have row_31 : (gl_AnchorGauss N zero succ add mul one two identity) := by
    apply row_32
    exact row_6
    exact row_11
    exact row_35
    exact row_34
  have rule_row_44 := external_gauss_externals_24_018 N zero succ add mul one two identity (anchorGaussOfFTA N zero succ add mul one two identity anchor)
  -- chapter_16_line_44: GL tag implication.
  have row_44 : (gl_preorder N add v1 v2) := by
    apply rule_row_44
    exact row_37
  have rule_row_30 := external_gauss_externals_24_018 N zero succ add mul one two identity (anchorGaussOfFTA N zero succ add mul one two identity anchor)
  -- chapter_16_line_30: GL tag implication.
  have row_30 : (gl_preorder N add v3 v2) := by
    apply rule_row_30
    exact row_36
  -- chapter_16_line_26: GL tag implication.
  have row_26 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_27
    exact row_6
    exact row_11
  have rule_row_52 := external_peano_externals_36_035 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_16_line_52: GL tag implication.
  have row_52 : (gl_existence11 N one succ) := by
    apply rule_row_52
  -- chapter_16_line_51: GL tag expansion.
  have row_51 : (¬ (∀ (v8 : α), ((N v8) → (¬ (succ v8 one))))) := by
    simpa only [gl_existence11] using row_52
  have exists_row_51 : ∃ (v8 : α), ((N v8) ∧ (succ v8 one)) := existsAndOfNotForallImpNot row_51
  obtain ⟨v8, witness_row_51⟩ := exists_row_51
  -- chapter_16_line_50: GL tag disintegration.
  have row_50 : (succ v8 one) := by
    exact witness_row_51.2
  have rule_row_45 := external_peano_externals_36_029 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_16_line_45: GL tag implication.
  have row_45 : (v3 = v2) := by
    apply rule_row_45
    exact row_30
    exact row_29
  have rule_row_41 := external_peano_externals_36_029 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_16_line_41: GL tag implication.
  have row_41 : (v2 = v1) := by
    apply rule_row_41
    exact row_42
    exact row_44
  -- chapter_16_line_40: GL tag equality1.
  have row_40 : (succ v2 v1) := by
    have equality_source := row_36
    have equality_step_1 := row_41
    cases equality_step_1
    have equality_step_2 := row_45
    cases equality_step_2
    exact equality_source
  have rule_row_25 := external_peano_externals_36_029 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_16_line_25: GL tag implication.
  have row_25 : (v2 = v3) := by
    apply rule_row_25
    exact row_29
    exact row_30
  -- chapter_16_line_5: GL tag expansion.
  have row_5 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_6
  -- chapter_16_line_67: GL tag disintegration.
  have row_67 : (gl_fXYZ add N N N) := by
    exact row_5.1.1.1.1.1.1.1.1.2
  -- chapter_16_line_66: GL tag expansion.
  have row_66 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_67
  -- chapter_16_line_65: GL tag disintegration.
  have row_65 : (gl_implication10 add N) := by
    exact row_66.1.1.2
  -- chapter_16_line_64: GL tag expansion.
  have row_64 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication10] using row_65
  -- chapter_16_line_63: GL tag implication.
  have row_63 : (N v1) := by
    apply row_64
    exact row_68
  -- chapter_16_line_61: GL tag disintegration.
  have row_61 : (gl_implication16 N zero add) := by
    exact row_5.1.1.1.1.1.1.2
  -- chapter_16_line_60: GL tag expansion.
  have row_60 : (∀ (w1 : α) (w2 : α), ((w1 = w2) → ((N w1) → ((N w2) → (add w1 zero w2))))) := by
    simpa only [gl_implication16] using row_61
  -- chapter_16_line_49: GL tag disintegration.
  have row_49 : (gl_implication7 N succ) := by
    exact row_5.1.1.1.1.1.1.1.1.1.2
  -- chapter_16_line_48: GL tag expansion.
  have row_48 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w2 w1) → (∀ (w3 : α), ((succ w3 w1) → (w2 = w3))))))) := by
    simpa only [gl_implication7] using row_49
  -- chapter_16_line_17: GL tag disintegration.
  have row_17 : (gl_fXY succ N N) := by
    exact row_5.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_16_line_16: GL tag expansion.
  have row_16 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_17
  -- chapter_16_line_55: GL tag disintegration.
  have row_55 : (gl_implication0 succ N) := by
    exact row_16.1.1.1
  -- chapter_16_line_54: GL tag expansion.
  have row_54 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_55
  -- chapter_16_line_62: GL tag implication.
  have row_62 : (N v3) := by
    apply row_54
    exact row_36
  -- chapter_16_line_59: GL tag implication.
  have row_59 : (add v3 zero v1) := by
    apply row_60
    exact row_43
    exact row_62
    exact row_63
  -- chapter_16_line_58: GL tag equality1.
  have row_58 : (add v3 zero v3) := by
    have equality_source := row_59
    have equality_step_1 := row_38
    cases equality_step_1
    exact equality_source
  -- chapter_16_line_53: GL tag implication.
  have row_53 : (N one) := by
    apply row_54
    exact row_35
  -- chapter_16_line_47: GL tag implication.
  have row_47 : (zero = v8) := by
    apply row_48
    exact row_53
    exact row_11
    exact row_50
  -- chapter_16_line_57: GL tag equality1.
  have row_57 : (add v3 v8 v3) := by
    have equality_source := row_58
    have equality_step_1 := row_47
    cases equality_step_1
    exact equality_source
  -- chapter_16_line_56: GL tag equality1.
  have row_56 : (add v1 v8 v3) := by
    have equality_source := row_57
    have equality_step_1 := row_43
    cases equality_step_1
    exact equality_source
  -- chapter_16_line_22: GL tag disintegration.
  have row_22 : (gl_implication4 N N succ) := by
    exact row_16.1.2
  -- chapter_16_line_21: GL tag expansion.
  have row_21 : (∀ (w1 : α), ((N w1) → (gl_existence0 N w1 succ))) := by
    simpa only [gl_implication4] using row_22
  -- chapter_16_line_15: GL tag disintegration.
  have row_15 : (gl_implication5 N succ) := by
    exact row_16.2
  -- chapter_16_line_14: GL tag expansion.
  have row_14 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α), ((succ w1 w3) → (w2 = w3))))))) := by
    simpa only [gl_implication5] using row_15
  -- chapter_16_line_9: GL tag disintegration.
  have row_9 : (N zero) := by
    exact row_5.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_16_line_20: GL tag implication.
  have row_20 : (gl_existence0 N zero succ) := by
    apply row_21
    exact row_9
  -- chapter_16_line_19: GL tag expansion.
  have row_19 : (¬ (∀ (v5 : α), ((N v5) → (¬ (succ zero v5))))) := by
    simpa only [gl_existence0] using row_20
  have exists_row_19 : ∃ (v5 : α), ((N v5) ∧ (succ zero v5)) := existsAndOfNotForallImpNot row_19
  obtain ⟨v5, witness_row_19⟩ := exists_row_19
  -- chapter_16_line_18: GL tag disintegration.
  have row_18 : (succ zero v5) := by
    exact witness_row_19.2
  -- chapter_16_line_46: GL tag equality1.
  have row_46 : (succ v8 v5) := by
    have equality_source := row_18
    have equality_step_1 := row_47
    cases equality_step_1
    exact equality_source
  have rule_row_39 := external_peano_externals_36_003 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_16_line_39: GL tag implication.
  have row_39 : (add v5 v2 v3) := by
    apply rule_row_39
    exact row_56
    exact row_46
    exact row_40
  have rule_row_24 := external_peano_externals_36_005 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_16_line_24: GL tag implication.
  have row_24 : (zero = v5) := by
    apply rule_row_24
    exact row_39
    exact row_25
  -- chapter_16_line_23: GL tag symmetry of equality.
  have row_23 : (v5 = zero) := by
    exact Eq.symm row_24
  -- chapter_16_line_13: GL tag implication.
  have row_13 : (one = v5) := by
    apply row_14
    exact row_9
    exact row_11
    exact row_18
  -- chapter_16_line_12: GL tag equality2.
  have row_12 : (one = zero) := by
    exact Eq.trans row_13 row_23
  -- chapter_16_line_10: GL tag equality1.
  have row_10 : (succ zero zero) := by
    have equality_source := row_11
    have equality_step_1 := row_12
    cases equality_step_1
    exact equality_source
  -- chapter_16_line_4: GL tag disintegration.
  have row_4 : (gl_implication6 N zero succ) := by
    exact row_5.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_16_line_3: GL tag expansion.
  have row_3 : (∀ (w1 : α), ((N w1) → (¬ (succ w1 zero)))) := by
    simpa only [gl_implication6] using row_4
  -- chapter_16_line_2: GL tag implication.
  have row_2 : (¬ (succ zero zero)) := by
    apply row_3
    exact row_9
  -- chapter_16_line_1: GL tag contradiction.
  have row_1 : (¬ (v1 = v3)) := by
    exact False.elim (row_2 row_10)
  exact row_1 reductio

theorem fta_source_019
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_015 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_3 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (add v1 one v2))) := by
  intro v1
  intro v2
  intro premise_1
  -- chapter_21_line_9: GL tag task formulation.
  have row_9 : (succ v1 v2) := by
    exact premise_1
  -- chapter_21_line_7: GL tag task formulation.
  have row_7 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_21_line_6: GL tag expansion.
  have row_6 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_7
  -- chapter_21_line_8: GL tag disintegration.
  have row_8 : (succ zero one) := by
    exact row_6.1.1.2
  -- chapter_21_line_5: GL tag disintegration.
  have row_5 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_6.1.1.1
  -- chapter_21_line_4: GL tag expansion for integration.
  have row_4 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_21_line_3: GL tag reformulation for integration and.
  have row_3 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_21_line_2: GL tag implication.
  have row_2 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_3
    exact row_5
    exact row_8
  have rule_row_1 := external_peano_externals_36_015 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_21_line_1: GL tag implication.
  have row_1 : (add v1 one v2) := by
    apply rule_row_1
    exact row_9
  exact row_1

theorem fta_source_020
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_024 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_8 x_2) → ((x_1 x_8) → (x_2 = x_7)))))))
    : (∀ (v1 : α) (v2 : α), ((add v1 v2 zero) → ((N v2) → (zero = v1)))) := by
  intro v1
  intro v2
  intro premise_1
  intro premise_2
  -- chapter_22_line_10: GL tag task formulation.
  have row_10 : (N v2) := by
    exact premise_2
  -- chapter_22_line_9: GL tag task formulation.
  have row_9 : (add v1 v2 zero) := by
    exact premise_1
  -- chapter_22_line_7: GL tag task formulation.
  have row_7 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_22_line_6: GL tag expansion.
  have row_6 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_7
  -- chapter_22_line_8: GL tag disintegration.
  have row_8 : (succ zero one) := by
    exact row_6.1.1.2
  -- chapter_22_line_5: GL tag disintegration.
  have row_5 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_6.1.1.1
  -- chapter_22_line_4: GL tag expansion for integration.
  have row_4 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_22_line_3: GL tag reformulation for integration and.
  have row_3 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_22_line_2: GL tag implication.
  have row_2 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_3
    exact row_5
    exact row_8
  have rule_row_1 := external_peano_externals_36_024 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_22_line_1: GL tag implication.
  have row_1 : (zero = v1) := by
    apply rule_row_1
    exact row_9
    exact row_10
  exact row_1

theorem fta_source_021
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_025 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_8 x_2) → (x_4 x_8 x_7 x_2))))))
    (external_peano_externals_36_024 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_8 x_2) → ((x_1 x_8) → (x_2 = x_7)))))))
    : (∀ (v1 : α) (v2 : α), ((add v1 v2 zero) → ((N v1) → (zero = v2)))) := by
  intro v1
  intro v2
  intro premise_1
  intro premise_2
  -- chapter_23_line_11: GL tag task formulation.
  have row_11 : (N v1) := by
    exact premise_2
  -- chapter_23_line_10: GL tag task formulation.
  have row_10 : (add v1 v2 zero) := by
    exact premise_1
  -- chapter_23_line_7: GL tag task formulation.
  have row_7 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_23_line_6: GL tag expansion.
  have row_6 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_7
  -- chapter_23_line_8: GL tag disintegration.
  have row_8 : (succ zero one) := by
    exact row_6.1.1.2
  -- chapter_23_line_5: GL tag disintegration.
  have row_5 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_6.1.1.1
  -- chapter_23_line_4: GL tag expansion for integration.
  have row_4 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_23_line_3: GL tag reformulation for integration and.
  have row_3 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_23_line_2: GL tag implication.
  have row_2 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_3
    exact row_5
    exact row_8
  have rule_row_9 := external_peano_externals_36_025 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_23_line_9: GL tag implication.
  have row_9 : (add v2 v1 zero) := by
    apply rule_row_9
    exact row_10
  have rule_row_1 := external_peano_externals_36_024 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_23_line_1: GL tag implication.
  have row_1 : (zero = v2) := by
    apply rule_row_1
    exact row_9
    exact row_11
  exact row_1

theorem fta_source_025
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    : (∀ (v1 : α) (v2 : α), ((mul v1 v2 one) → (¬ (zero = v2)))) := by
  intro v1
  intro v2
  intro premise_1
  intro reductio
  -- chapter_31_line_19: GL tag task formulation.
  have row_19 : (mul v1 v2 one) := by
    exact premise_1
  -- chapter_31_line_11: GL tag task formulation.
  have row_11 : (zero = v2) := by
    exact reductio
  -- chapter_31_line_26: GL tag symmetry of equality.
  have row_26 : (v2 = zero) := by
    exact Eq.symm row_11
  -- chapter_31_line_25: GL tag equality1.
  have row_25 : (mul v1 zero one) := by
    have equality_source := row_19
    have equality_step_1 := row_26
    cases equality_step_1
    exact equality_source
  -- chapter_31_line_9: GL tag task formulation.
  have row_9 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_31_line_8: GL tag expansion.
  have row_8 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_9
  -- chapter_31_line_14: GL tag disintegration.
  have row_14 : (succ zero one) := by
    exact row_8.1.1.2
  -- chapter_31_line_13: GL tag equality1.
  have row_13 : (succ v2 one) := by
    have equality_source := row_14
    have equality_step_1 := row_11
    cases equality_step_1
    exact equality_source
  -- chapter_31_line_7: GL tag disintegration.
  have row_7 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_8.1.1.1
  -- chapter_31_line_6: GL tag expansion.
  have row_6 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_7
  -- chapter_31_line_24: GL tag disintegration.
  have row_24 : (gl_fXYZ mul N N N) := by
    exact row_6.1.1.1.2
  -- chapter_31_line_23: GL tag expansion.
  have row_23 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_24
  -- chapter_31_line_22: GL tag disintegration.
  have row_22 : (gl_implication8 mul N) := by
    exact row_23.1.1.1.1
  -- chapter_31_line_21: GL tag expansion.
  have row_21 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_22
  -- chapter_31_line_20: GL tag implication.
  have row_20 : (N v1) := by
    apply row_21
    exact row_25
  -- chapter_31_line_18: GL tag disintegration.
  have row_18 : (gl_implication19 N zero mul) := by
    exact row_6.1.1.2
  -- chapter_31_line_17: GL tag equality1.
  have row_17 : (gl_implication19 N v2 mul) := by
    have equality_source := row_18
    have equality_step_1 := row_11
    cases equality_step_1
    exact equality_source
  -- chapter_31_line_16: GL tag expansion.
  have row_16 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((mul w1 v2 w2) → (w2 = v2))))) := by
    simpa only [gl_implication19] using row_17
  -- chapter_31_line_15: GL tag implication.
  have row_15 : (one = v2) := by
    apply row_16
    exact row_20
    exact row_19
  -- chapter_31_line_12: GL tag equality1.
  have row_12 : (succ v2 v2) := by
    have equality_source := row_13
    have equality_step_1 := row_15
    cases equality_step_1
    exact equality_source
  -- chapter_31_line_10: GL tag disintegration.
  have row_10 : (N zero) := by
    exact row_6.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_31_line_5: GL tag disintegration.
  have row_5 : (gl_implication6 N zero succ) := by
    exact row_6.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_31_line_4: GL tag expansion.
  have row_4 : (∀ (w1 : α), ((N w1) → (¬ (succ w1 zero)))) := by
    simpa only [gl_implication6] using row_5
  -- chapter_31_line_3: GL tag implication.
  have row_3 : (¬ (succ zero zero)) := by
    apply row_4
    exact row_10
  -- chapter_31_line_2: GL tag equality1.
  have row_2 : (¬ (succ v2 v2)) := by
    have equality_source := row_3
    have equality_step_1 := row_11
    cases equality_step_1
    exact equality_source
  -- chapter_31_line_1: GL tag contradiction.
  have row_1 : (¬ (zero = v2)) := by
    exact False.elim (row_2 row_12)
  exact row_1 reductio

theorem fta_source_030
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_010 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_5 x_7 x_8 x_9) → (x_5 x_8 x_7 x_9))))))
    (external_peano_externals_36_008 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (∀ (w4 : α) (w5 : α), ((mul w4 w1 w5) → (∀ (w6 : α), ((mul w4 w2 w6) → (∀ (w7 : α), ((mul w4 w3 w7) → (add w5 w6 w7))))))))))))
    : (∀ (v1 : α) (v2 : α), ((gl_preorder N add v1 v2) → (∀ (v3 : α) (v4 : α), ((mul v1 v3 v4) → (∀ (v5 : α), ((mul v2 v3 v5) → (gl_preorder N add v4 v5))))))) := by
  intro v1
  intro v2
  intro premise_1
  intro v3
  intro v4
  intro premise_2
  intro v5
  intro premise_3
  -- chapter_40_line_29: GL tag task formulation.
  have row_29 : (gl_preorder N add v1 v2) := by
    exact premise_1
  -- chapter_40_line_28: GL tag expansion.
  have row_28 : (¬ (∀ (v9 : α), ((N v9) → (¬ (add v1 v9 v2))))) := by
    simpa only [gl_preorder] using row_29
  have exists_row_28 : ∃ (v9 : α), ((N v9) ∧ (add v1 v9 v2)) := existsAndOfNotForallImpNot row_28
  obtain ⟨v9, witness_row_28⟩ := exists_row_28
  -- chapter_40_line_30: GL tag disintegration.
  have row_30 : (add v1 v9 v2) := by
    exact witness_row_28.2
  -- chapter_40_line_27: GL tag disintegration.
  have row_27 : (N v9) := by
    exact witness_row_28.1
  -- chapter_40_line_15: GL tag task formulation.
  have row_15 : (mul v1 v3 v4) := by
    exact premise_2
  -- chapter_40_line_13: GL tag task formulation.
  have row_13 : (mul v2 v3 v5) := by
    exact premise_3
  -- chapter_40_line_10: GL tag task formulation.
  have row_10 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_40_line_9: GL tag expansion.
  have row_9 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_10
  -- chapter_40_line_11: GL tag disintegration.
  have row_11 : (succ zero one) := by
    exact row_9.1.1.2
  -- chapter_40_line_8: GL tag disintegration.
  have row_8 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_9.1.1.1
  -- chapter_40_line_23: GL tag expansion.
  have row_23 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_8
  -- chapter_40_line_22: GL tag disintegration.
  have row_22 : (gl_fXYZ mul N N N) := by
    exact row_23.1.1.1.2
  -- chapter_40_line_21: GL tag expansion.
  have row_21 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_22
  -- chapter_40_line_26: GL tag disintegration.
  have row_26 : (gl_implication9 mul N) := by
    exact row_21.1.1.1.2
  -- chapter_40_line_25: GL tag expansion.
  have row_25 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication9] using row_26
  -- chapter_40_line_24: GL tag implication.
  have row_24 : (N v3) := by
    apply row_25
    exact row_13
  -- chapter_40_line_20: GL tag disintegration.
  have row_20 : (gl_implication13 N N N mul) := by
    exact row_21.1.2
  -- chapter_40_line_19: GL tag expansion.
  have row_19 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication13] using row_20
  -- chapter_40_line_18: GL tag implication.
  have row_18 : (gl_existence1 N v3 v9 mul) := by
    apply row_19
    exact row_24
    exact row_27
  -- chapter_40_line_17: GL tag expansion.
  have row_17 : (¬ (∀ (v7 : α), ((N v7) → (¬ (mul v3 v9 v7))))) := by
    simpa only [gl_existence1] using row_18
  have exists_row_17 : ∃ (v7 : α), ((N v7) ∧ (mul v3 v9 v7)) := existsAndOfNotForallImpNot row_17
  obtain ⟨v7, witness_row_17⟩ := exists_row_17
  -- chapter_40_line_31: GL tag disintegration.
  have row_31 : (N v7) := by
    exact witness_row_17.1
  -- chapter_40_line_16: GL tag disintegration.
  have row_16 : (mul v3 v9 v7) := by
    exact witness_row_17.2
  -- chapter_40_line_7: GL tag expansion for integration.
  have row_7 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_40_line_6: GL tag reformulation for integration and.
  have row_6 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_40_line_5: GL tag implication.
  have row_5 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_6
    exact row_8
    exact row_11
  have rule_row_14 := external_peano_externals_36_010 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_40_line_14: GL tag implication.
  have row_14 : (mul v3 v1 v4) := by
    apply rule_row_14
    exact row_15
  have rule_row_12 := external_peano_externals_36_010 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_40_line_12: GL tag implication.
  have row_12 : (mul v3 v2 v5) := by
    apply rule_row_12
    exact row_13
  have rule_row_4 := external_peano_externals_36_008 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_40_line_4: GL tag implication.
  have row_4 : (add v4 v7 v5) := by
    apply rule_row_4
    exact row_30
    exact row_14
    exact row_16
    exact row_12
  -- chapter_40_line_3: GL tag expansion for integration.
  have row_3 : ((gl_preorder N add v4 v5) ↔ (¬ (∀ (v8 : α), ((N v8) → (¬ (add v4 v8 v5)))))) := by
    exact Iff.rfl
  -- chapter_40_line_2: GL tag reformulation for integration >[bound].
  have row_2 : (∀ (v6 : α), ((N v6) → ((add v4 v6 v5) → (gl_preorder N add v4 v5)))) := by
    intro v6
    intro integration_premise_1
    intro integration_premise_2
    apply (row_3).2
    intro universal_counterexample
    exact universal_counterexample v6 integration_premise_1 integration_premise_2
  -- chapter_40_line_1: GL tag implication.
  have row_1 : (gl_preorder N add v4 v5) := by
    apply row_2
    exact row_31
    exact row_4
  exact row_1

theorem fta_source_031
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    : (∀ (v1 : α) (v2 : α), ((gl_preorder N add v1 v2) → ((¬ (v1 = v2)) → (gl_strictOrder N add v1 v2)))) := by
  intro v1
  intro v2
  intro premise_1
  intro premise_2
  -- chapter_41_line_5: GL tag task formulation.
  have row_5 : (gl_preorder N add v1 v2) := by
    exact premise_1
  -- chapter_41_line_4: GL tag task formulation.
  have row_4 : (¬ (v1 = v2)) := by
    exact premise_2
  -- chapter_41_line_3: GL tag expansion for integration.
  have row_3 : ((gl_strictOrder N add v1 v2) ↔ ((gl_preorder N add v1 v2) ∧ (¬ (v1 = v2)))) := by
    exact Iff.rfl
  -- chapter_41_line_2: GL tag reformulation for integration and.
  have row_2 : ((gl_preorder N add v1 v2) → ((¬ (v1 = v2)) → (gl_strictOrder N add v1 v2))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_strictOrder]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_41_line_1: GL tag implication.
  have row_1 : (gl_strictOrder N add v1 v2) := by
    apply row_2
    exact row_5
    exact row_4
  exact row_1

theorem fta_source_032
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    : (∀ (v1 : α) (v2 : α), ((gl_preorder N add v1 v2) → ((¬ (gl_strictOrder N add v1 v2)) → (v1 = v2)))) := by
  intro v1
  intro v2
  intro premise_1
  intro premise_2
  -- chapter_42_line_5: GL tag task formulation.
  have row_5 : (gl_preorder N add v1 v2) := by
    exact premise_1
  -- chapter_42_line_4: GL tag task formulation.
  have row_4 : (¬ (gl_strictOrder N add v1 v2)) := by
    exact premise_2
  -- chapter_42_line_3: GL tag expansion.
  have row_3 : (¬ ((gl_preorder N add v1 v2) ∧ (¬ (v1 = v2)))) := by
    simpa only [gl_strictOrder] using row_4
  -- chapter_42_line_2: GL tag disintegration.
  have row_2 : ((gl_preorder N add v1 v2) → (v1 = v2)) := by
    classical
    intro projection_premise
    apply Classical.byContradiction
    intro projection_counterexample
    exact row_3 ⟨projection_premise, projection_counterexample⟩
  -- chapter_42_line_1: GL tag implication.
  have row_1 : (v1 = v2) := by
    apply row_2
    exact row_5
  exact row_1

theorem fta_source_033
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_029 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((gl_preorder x_1 x_4 x_7 x_8) → ((gl_preorder x_1 x_4 x_8 x_7) → (x_7 = x_8)))))))
    : (∀ (v1 : α) (v2 : α), ((gl_preorder N add v1 v2) → ((gl_preorder N add v2 v1) → (v1 = v2)))) := by
  intro v1
  intro v2
  intro premise_1
  intro premise_2
  -- chapter_43_line_10: GL tag task formulation.
  have row_10 : (gl_preorder N add v1 v2) := by
    exact premise_1
  -- chapter_43_line_9: GL tag task formulation.
  have row_9 : (gl_preorder N add v2 v1) := by
    exact premise_2
  -- chapter_43_line_7: GL tag task formulation.
  have row_7 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_43_line_6: GL tag expansion.
  have row_6 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_7
  -- chapter_43_line_8: GL tag disintegration.
  have row_8 : (succ zero one) := by
    exact row_6.1.1.2
  -- chapter_43_line_5: GL tag disintegration.
  have row_5 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_6.1.1.1
  -- chapter_43_line_4: GL tag expansion for integration.
  have row_4 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_43_line_3: GL tag reformulation for integration and.
  have row_3 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_43_line_2: GL tag implication.
  have row_2 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_3
    exact row_5
    exact row_8
  have rule_row_1 := external_peano_externals_36_029 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_43_line_1: GL tag implication.
  have row_1 : (v1 = v2) := by
    apply rule_row_1
    exact row_10
    exact row_9
  exact row_1

theorem fta_source_037
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_010 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_5 x_7 x_8 x_9) → (x_5 x_8 x_7 x_9))))))
    (external_peano_externals_36_009 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (∀ (w4 : α) (w5 : α), ((mul w4 w2 w5) → (∀ (w6 : α), ((mul w3 w4 w6) → (mul w5 w1 w6))))))))))
    : (∀ (v1 : α) (v2 : α), ((gl_preorder N mul v1 v2) → (∀ (v3 : α) (v4 : α), ((mul v3 v1 v4) → (∀ (v5 : α), ((mul v3 v2 v5) → (gl_preorder N mul v4 v5))))))) := by
  intro v1
  intro v2
  intro premise_1
  intro v3
  intro v4
  intro premise_2
  intro v5
  intro premise_3
  -- chapter_47_line_18: GL tag task formulation.
  have row_18 : (gl_preorder N mul v1 v2) := by
    exact premise_1
  -- chapter_47_line_17: GL tag expansion.
  have row_17 : (¬ (∀ (v7 : α), ((N v7) → (¬ (mul v1 v7 v2))))) := by
    simpa only [gl_preorder] using row_18
  have exists_row_17 : ∃ (v7 : α), ((N v7) ∧ (mul v1 v7 v2)) := existsAndOfNotForallImpNot row_17
  obtain ⟨v7, witness_row_17⟩ := exists_row_17
  -- chapter_47_line_19: GL tag disintegration.
  have row_19 : (N v7) := by
    exact witness_row_17.1
  -- chapter_47_line_16: GL tag disintegration.
  have row_16 : (mul v1 v7 v2) := by
    exact witness_row_17.2
  -- chapter_47_line_14: GL tag task formulation.
  have row_14 : (mul v3 v1 v4) := by
    exact premise_2
  -- chapter_47_line_13: GL tag task formulation.
  have row_13 : (mul v3 v2 v5) := by
    exact premise_3
  -- chapter_47_line_10: GL tag task formulation.
  have row_10 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_47_line_9: GL tag expansion.
  have row_9 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_10
  -- chapter_47_line_11: GL tag disintegration.
  have row_11 : (succ zero one) := by
    exact row_9.1.1.2
  -- chapter_47_line_8: GL tag disintegration.
  have row_8 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_9.1.1.1
  -- chapter_47_line_7: GL tag expansion for integration.
  have row_7 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_47_line_6: GL tag reformulation for integration and.
  have row_6 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_47_line_5: GL tag implication.
  have row_5 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_6
    exact row_8
    exact row_11
  have rule_row_15 := external_peano_externals_36_010 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_47_line_15: GL tag implication.
  have row_15 : (mul v7 v1 v2) := by
    apply rule_row_15
    exact row_16
  have rule_row_12 := external_peano_externals_36_010 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_47_line_12: GL tag implication.
  have row_12 : (mul v2 v3 v5) := by
    apply rule_row_12
    exact row_13
  have rule_row_4 := external_peano_externals_36_009 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_47_line_4: GL tag implication.
  have row_4 : (mul v4 v7 v5) := by
    apply rule_row_4
    exact row_15
    exact row_14
    exact row_12
  -- chapter_47_line_3: GL tag expansion for integration.
  have row_3 : ((gl_preorder N mul v4 v5) ↔ (¬ (∀ (v8 : α), ((N v8) → (¬ (mul v4 v8 v5)))))) := by
    exact Iff.rfl
  -- chapter_47_line_2: GL tag reformulation for integration >[bound].
  have row_2 : (∀ (v6 : α), ((N v6) → ((mul v4 v6 v5) → (gl_preorder N mul v4 v5)))) := by
    intro v6
    intro integration_premise_1
    intro integration_premise_2
    apply (row_3).2
    intro universal_counterexample
    exact universal_counterexample v6 integration_premise_1 integration_premise_2
  -- chapter_47_line_1: GL tag implication.
  have row_1 : (gl_preorder N mul v4 v5) := by
    apply row_2
    exact row_19
    exact row_4
  exact row_1

theorem fta_source_040
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_000 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (∀ (x_10 : α) (x_11 : α) (x_12 : α), ((x_4 x_10 x_11 x_12) → (∀ (x_13 : α), ((x_5 x_13 x_10 x_7) → ((x_5 x_13 x_11 x_8) → (x_5 x_13 x_12 x_9)))))))))))
    : (∀ (v1 : α) (v2 : α), ((gl_preorder N mul v1 v2) → (∀ (v3 : α), ((gl_preorder N mul v1 v3) → (∀ (v4 : α), ((add v2 v3 v4) → (gl_preorder N mul v1 v4))))))) := by
  intro v1
  intro v2
  intro premise_1
  intro v3
  intro premise_2
  intro v4
  intro premise_3
  -- chapter_50_line_18: GL tag task formulation.
  have row_18 : (gl_preorder N mul v1 v3) := by
    exact premise_2
  -- chapter_50_line_17: GL tag expansion.
  have row_17 : (¬ (∀ (v9 : α), ((N v9) → (¬ (mul v1 v9 v3))))) := by
    simpa only [gl_preorder] using row_18
  have exists_row_17 : ∃ (v9 : α), ((N v9) ∧ (mul v1 v9 v3)) := existsAndOfNotForallImpNot row_17
  obtain ⟨v9, witness_row_17⟩ := exists_row_17
  -- chapter_50_line_28: GL tag disintegration.
  have row_28 : (N v9) := by
    exact witness_row_17.1
  -- chapter_50_line_16: GL tag disintegration.
  have row_16 : (mul v1 v9 v3) := by
    exact witness_row_17.2
  -- chapter_50_line_15: GL tag task formulation.
  have row_15 : (gl_preorder N mul v1 v2) := by
    exact premise_1
  -- chapter_50_line_14: GL tag expansion.
  have row_14 : (¬ (∀ (v8 : α), ((N v8) → (¬ (mul v1 v8 v2))))) := by
    simpa only [gl_preorder] using row_15
  have exists_row_14 : ∃ (v8 : α), ((N v8) ∧ (mul v1 v8 v2)) := existsAndOfNotForallImpNot row_14
  obtain ⟨v8, witness_row_14⟩ := exists_row_14
  -- chapter_50_line_27: GL tag disintegration.
  have row_27 : (N v8) := by
    exact witness_row_14.1
  -- chapter_50_line_13: GL tag disintegration.
  have row_13 : (mul v1 v8 v2) := by
    exact witness_row_14.2
  -- chapter_50_line_12: GL tag task formulation.
  have row_12 : (add v2 v3 v4) := by
    exact premise_3
  -- chapter_50_line_10: GL tag task formulation.
  have row_10 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_50_line_9: GL tag expansion.
  have row_9 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_10
  -- chapter_50_line_11: GL tag disintegration.
  have row_11 : (succ zero one) := by
    exact row_9.1.1.2
  -- chapter_50_line_8: GL tag disintegration.
  have row_8 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_9.1.1.1
  -- chapter_50_line_26: GL tag expansion.
  have row_26 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_8
  -- chapter_50_line_25: GL tag disintegration.
  have row_25 : (gl_fXYZ add N N N) := by
    exact row_26.1.1.1.1.1.1.1.1.2
  -- chapter_50_line_24: GL tag expansion.
  have row_24 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_25
  -- chapter_50_line_23: GL tag disintegration.
  have row_23 : (gl_implication13 N N N add) := by
    exact row_24.1.2
  -- chapter_50_line_22: GL tag expansion.
  have row_22 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication13] using row_23
  -- chapter_50_line_21: GL tag implication.
  have row_21 : (gl_existence1 N v8 v9 add) := by
    apply row_22
    exact row_27
    exact row_28
  -- chapter_50_line_20: GL tag expansion.
  have row_20 : (¬ (∀ (v6 : α), ((N v6) → (¬ (add v8 v9 v6))))) := by
    simpa only [gl_existence1] using row_21
  have exists_row_20 : ∃ (v6 : α), ((N v6) ∧ (add v8 v9 v6)) := existsAndOfNotForallImpNot row_20
  obtain ⟨v6, witness_row_20⟩ := exists_row_20
  -- chapter_50_line_29: GL tag disintegration.
  have row_29 : (N v6) := by
    exact witness_row_20.1
  -- chapter_50_line_19: GL tag disintegration.
  have row_19 : (add v8 v9 v6) := by
    exact witness_row_20.2
  -- chapter_50_line_7: GL tag expansion for integration.
  have row_7 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_50_line_6: GL tag reformulation for integration and.
  have row_6 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_50_line_5: GL tag implication.
  have row_5 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_6
    exact row_8
    exact row_11
  have rule_row_4 := external_peano_externals_36_000 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_50_line_4: GL tag implication.
  have row_4 : (mul v1 v6 v4) := by
    apply rule_row_4
    exact row_12
    exact row_19
    exact row_13
    exact row_16
  -- chapter_50_line_3: GL tag expansion for integration.
  have row_3 : ((gl_preorder N mul v1 v4) ↔ (¬ (∀ (v7 : α), ((N v7) → (¬ (mul v1 v7 v4)))))) := by
    exact Iff.rfl
  -- chapter_50_line_2: GL tag reformulation for integration >[bound].
  have row_2 : (∀ (v5 : α), ((N v5) → ((mul v1 v5 v4) → (gl_preorder N mul v1 v4)))) := by
    intro v5
    intro integration_premise_1
    intro integration_premise_2
    apply (row_3).2
    intro universal_counterexample
    exact universal_counterexample v5 integration_premise_1 integration_premise_2
  -- chapter_50_line_1: GL tag implication.
  have row_1 : (gl_preorder N mul v1 v4) := by
    apply row_2
    exact row_29
    exact row_4
  exact row_1

theorem fta_source_044
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_031 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α), ((x_1 x_7) → (∀ (x_8 : α), ((x_5 x_2 x_7 x_8) → (x_2 = x_8))))))))
    : (∀ (v1 : α) (v2 : α), ((gl_preorder N mul v1 v2) → ((gl_preorder N mul v2 v1) → ((zero = v2) → (v1 = v2))))) := by
  intro v1
  intro v2
  intro premise_1
  intro premise_2
  intro premise_3
  -- chapter_56_line_29: GL tag expansion for integration.
  have row_29 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_56_line_28: GL tag reformulation for integration and.
  have row_28 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_56_line_24: GL tag task formulation.
  have row_24 : (gl_preorder N mul v2 v1) := by
    exact premise_2
  -- chapter_56_line_23: GL tag expansion.
  have row_23 : (¬ (∀ (v6 : α), ((N v6) → (¬ (mul v2 v6 v1))))) := by
    simpa only [gl_preorder] using row_24
  have exists_row_23 : ∃ (v6 : α), ((N v6) ∧ (mul v2 v6 v1)) := existsAndOfNotForallImpNot row_23
  obtain ⟨v6, witness_row_23⟩ := exists_row_23
  -- chapter_56_line_32: GL tag disintegration.
  have row_32 : (N v6) := by
    exact witness_row_23.1
  -- chapter_56_line_22: GL tag disintegration.
  have row_22 : (mul v2 v6 v1) := by
    exact witness_row_23.2
  -- chapter_56_line_14: GL tag task formulation.
  have row_14 : (zero = v2) := by
    exact premise_3
  -- chapter_56_line_13: GL tag symmetry of equality.
  have row_13 : (v2 = zero) := by
    exact Eq.symm row_14
  -- chapter_56_line_31: GL tag equality1.
  have row_31 : (mul zero v6 v1) := by
    have equality_source := row_22
    have equality_step_1 := row_13
    cases equality_step_1
    exact equality_source
  -- chapter_56_line_7: GL tag task formulation.
  have row_7 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_56_line_6: GL tag expansion.
  have row_6 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_7
  -- chapter_56_line_30: GL tag disintegration.
  have row_30 : (succ zero one) := by
    exact row_6.1.1.2
  -- chapter_56_line_21: GL tag disintegration.
  have row_21 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_6.1.1.1
  -- chapter_56_line_27: GL tag implication.
  have row_27 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_28
    exact row_21
    exact row_30
  have rule_row_26 := external_peano_externals_36_031 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_56_line_26: GL tag implication.
  have row_26 : (zero = v1) := by
    apply rule_row_26
    exact row_32
    exact row_31
  -- chapter_56_line_25: GL tag equality2.
  have row_25 : (v2 = v1) := by
    exact Eq.trans row_13 row_26
  -- chapter_56_line_20: GL tag expansion.
  have row_20 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_21
  -- chapter_56_line_19: GL tag disintegration.
  have row_19 : (gl_fXYZ mul N N N) := by
    exact row_20.1.1.1.2
  -- chapter_56_line_18: GL tag expansion.
  have row_18 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_19
  -- chapter_56_line_17: GL tag disintegration.
  have row_17 : (gl_implication8 mul N) := by
    exact row_18.1.1.1.1
  -- chapter_56_line_16: GL tag expansion.
  have row_16 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_17
  -- chapter_56_line_15: GL tag implication.
  have row_15 : (N v2) := by
    apply row_16
    exact row_22
  -- chapter_56_line_5: GL tag disintegration.
  have row_5 : (gl_identity N identity) := by
    exact row_6.2
  -- chapter_56_line_4: GL tag expansion.
  have row_4 : (((gl_implication0 identity N) ∧ (gl_implication22 identity)) ∧ (gl_implication23 N identity)) := by
    simpa only [gl_identity] using row_5
  -- chapter_56_line_12: GL tag disintegration.
  have row_12 : (gl_implication23 N identity) := by
    exact row_4.2
  -- chapter_56_line_11: GL tag expansion.
  have row_11 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((w1 = w2) → (identity w1 w2))))) := by
    simpa only [gl_implication23] using row_12
  -- chapter_56_line_10: GL tag implication.
  have row_10 : (identity v2 zero) := by
    apply row_11
    exact row_15
    exact row_13
  -- chapter_56_line_9: GL tag equality1.
  have row_9 : (identity v2 v2) := by
    have equality_source := row_10
    have equality_step_1 := row_14
    cases equality_step_1
    exact equality_source
  -- chapter_56_line_8: GL tag equality1.
  have row_8 : (identity v1 v2) := by
    have equality_source := row_9
    have equality_step_1 := row_25
    cases equality_step_1
    exact equality_source
  -- chapter_56_line_3: GL tag disintegration.
  have row_3 : (gl_implication22 identity) := by
    exact row_4.1.2
  -- chapter_56_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α) (w2 : α), ((identity w1 w2) → (w1 = w2))) := by
    simpa only [gl_implication22] using row_3
  -- chapter_56_line_1: GL tag implication.
  have row_1 : (v1 = v2) := by
    apply row_2
    exact row_8
  exact row_1

theorem fta_source_046
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_029 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((gl_preorder x_1 x_4 x_7 x_8) → ((gl_preorder x_1 x_4 x_8 x_7) → (x_7 = x_8)))))))
    : (∀ (v1 : α) (v2 : α), ((gl_strictOrder N add v1 v2) → (¬ (gl_strictOrder N add v2 v1)))) := by
  intro v1
  intro v2
  intro premise_1
  -- chapter_58_line_15: GL tag task formulation.
  have row_15 : (gl_strictOrder N add v1 v2) := by
    exact premise_1
  -- chapter_58_line_14: GL tag expansion.
  have row_14 : ((gl_preorder N add v1 v2) ∧ (¬ (v1 = v2))) := by
    simpa only [gl_strictOrder] using row_15
  -- chapter_58_line_17: GL tag disintegration.
  have row_17 : (¬ (v1 = v2)) := by
    exact row_14.2
  -- chapter_58_line_16: GL tag symmetry of inequality.
  have row_16 : (¬ (v2 = v1)) := by
    exact fun equality => row_17 (Eq.symm equality)
  -- chapter_58_line_13: GL tag disintegration.
  have row_13 : (gl_preorder N add v1 v2) := by
    exact row_14.1
  -- chapter_58_line_12: GL tag task formulation.
  have row_12 : ((gl_strictOrder N add v2 v1) → (gl_strictOrder N add v2 v1)) := by
    intro scope_premise_1
    exact scope_premise_1
  -- chapter_58_line_11: GL tag expansion.
  have row_11 : ((gl_strictOrder N add v2 v1) → ((gl_preorder N add v2 v1) ∧ (¬ (v2 = v1)))) := by
    simpa only [gl_strictOrder] using row_12
  -- chapter_58_line_10: GL tag disintegration.
  have row_10 : ((gl_strictOrder N add v2 v1) → (gl_preorder N add v2 v1)) := by
    intro scope_premise_1
    have scoped_fact_1 := row_11 scope_premise_1
    exact scoped_fact_1.1
  -- chapter_58_line_8: GL tag task formulation.
  have row_8 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_58_line_7: GL tag expansion.
  have row_7 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_8
  -- chapter_58_line_9: GL tag disintegration.
  have row_9 : (succ zero one) := by
    exact row_7.1.1.2
  -- chapter_58_line_6: GL tag disintegration.
  have row_6 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_7.1.1.1
  -- chapter_58_line_5: GL tag expansion for integration.
  have row_5 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_58_line_4: GL tag reformulation for integration and.
  have row_4 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_58_line_3: GL tag implication.
  have row_3 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_4
    exact row_6
    exact row_9
  have rule_row_2 := external_peano_externals_36_029 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_58_line_2: GL tag implication.
  have row_2 : ((gl_strictOrder N add v2 v1) → (v2 = v1)) := by
    intro scope_premise_1
    have scoped_fact_2 := row_10 scope_premise_1
    apply rule_row_2
    exact scoped_fact_2
    exact row_13
  -- chapter_58_line_1: GL tag contradiction.
  have row_1 : (¬ (gl_strictOrder N add v2 v1)) := by
    intro contradiction_assumption
    have scoped_contradiction := row_2 contradiction_assumption
    exact row_16 scoped_contradiction
  exact row_1

theorem fta_source_047
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    : (∀ (v1 : α) (v2 : α), ((gl_strictOrder N add v1 v2) → (∀ (v3 : α), ((add v1 v3 v2) → (¬ (zero = v3)))))) := by
  intro v1
  intro v2
  intro premise_1
  intro v3
  intro premise_2
  intro reductio
  -- chapter_59_line_16: GL tag task formulation.
  have row_16 : (add v1 v3 v2) := by
    exact premise_2
  -- chapter_59_line_15: GL tag task formulation.
  have row_15 : (zero = v3) := by
    exact reductio
  -- chapter_59_line_23: GL tag symmetry of equality.
  have row_23 : (v3 = zero) := by
    exact Eq.symm row_15
  -- chapter_59_line_22: GL tag equality1.
  have row_22 : (add v1 zero v2) := by
    have equality_source := row_16
    have equality_step_1 := row_23
    cases equality_step_1
    exact equality_source
  -- chapter_59_line_14: GL tag task formulation.
  have row_14 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_59_line_13: GL tag expansion.
  have row_13 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_14
  -- chapter_59_line_12: GL tag disintegration.
  have row_12 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_13.1.1.1
  -- chapter_59_line_11: GL tag expansion.
  have row_11 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_12
  -- chapter_59_line_21: GL tag disintegration.
  have row_21 : (gl_fXYZ add N N N) := by
    exact row_11.1.1.1.1.1.1.1.1.2
  -- chapter_59_line_20: GL tag expansion.
  have row_20 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_21
  -- chapter_59_line_19: GL tag disintegration.
  have row_19 : (gl_implication8 add N) := by
    exact row_20.1.1.1.1
  -- chapter_59_line_18: GL tag expansion.
  have row_18 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_19
  -- chapter_59_line_17: GL tag implication.
  have row_17 : (N v1) := by
    apply row_18
    exact row_22
  -- chapter_59_line_10: GL tag disintegration.
  have row_10 : (gl_implication15 N zero add) := by
    exact row_11.1.1.1.1.1.1.1.2
  -- chapter_59_line_9: GL tag equality1.
  have row_9 : (gl_implication15 N v3 add) := by
    have equality_source := row_10
    have equality_step_1 := row_15
    cases equality_step_1
    exact equality_source
  -- chapter_59_line_8: GL tag expansion.
  have row_8 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((add w1 v3 w2) → (w1 = w2))))) := by
    simpa only [gl_implication15] using row_9
  -- chapter_59_line_7: GL tag implication.
  have row_7 : (v1 = v2) := by
    apply row_8
    exact row_17
    exact row_16
  -- chapter_59_line_6: GL tag symmetry of equality.
  have row_6 : (v2 = v1) := by
    exact Eq.symm row_7
  -- chapter_59_line_5: GL tag task formulation.
  have row_5 : (gl_strictOrder N add v1 v2) := by
    exact premise_1
  -- chapter_59_line_4: GL tag expansion.
  have row_4 : ((gl_preorder N add v1 v2) ∧ (¬ (v1 = v2))) := by
    simpa only [gl_strictOrder] using row_5
  -- chapter_59_line_3: GL tag disintegration.
  have row_3 : (¬ (v1 = v2)) := by
    exact row_4.2
  -- chapter_59_line_2: GL tag symmetry of inequality.
  have row_2 : (¬ (v2 = v1)) := by
    exact fun equality => row_3 (Eq.symm equality)
  -- chapter_59_line_1: GL tag contradiction.
  have row_1 : (¬ (zero = v3)) := by
    exact False.elim (row_2 row_6)
  exact row_1 reductio

theorem fta_source_049
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_033 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α), ((x_1 x_7) → (∀ (x_8 : α), ((x_5 x_6 x_8 x_7) → (x_7 = x_8))))))))
    : (∀ (v1 : α), ((N v1) → (∀ (v2 : α), ((v1 = v2) → (mul one v1 v2))))) := by
  intro v1
  intro premise_1
  intro v2
  intro premise_2
  -- chapter_61_line_26: GL tag expansion for integration.
  have row_26 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_61_line_25: GL tag reformulation for integration and.
  have row_25 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_61_line_15: GL tag task formulation.
  have row_15 : (v1 = v2) := by
    exact premise_2
  -- chapter_61_line_22: GL tag symmetry of equality.
  have row_22 : (v2 = v1) := by
    exact Eq.symm row_15
  -- chapter_61_line_14: GL tag task formulation.
  have row_14 : (N v1) := by
    exact premise_1
  -- chapter_61_line_13: GL tag equality1.
  have row_13 : (N v2) := by
    have equality_source := row_14
    have equality_step_1 := row_15
    cases equality_step_1
    exact equality_source
  -- chapter_61_line_12: GL tag task formulation.
  have row_12 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_61_line_11: GL tag expansion.
  have row_11 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_12
  -- chapter_61_line_27: GL tag disintegration.
  have row_27 : (succ zero one) := by
    exact row_11.1.1.2
  -- chapter_61_line_21: GL tag disintegration.
  have row_21 : (succ one two) := by
    exact row_11.1.2
  -- chapter_61_line_10: GL tag disintegration.
  have row_10 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_11.1.1.1
  -- chapter_61_line_24: GL tag implication.
  have row_24 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_25
    exact row_10
    exact row_27
  -- chapter_61_line_9: GL tag expansion.
  have row_9 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_10
  -- chapter_61_line_20: GL tag disintegration.
  have row_20 : (gl_fXY succ N N) := by
    exact row_9.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_61_line_19: GL tag expansion.
  have row_19 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_20
  -- chapter_61_line_18: GL tag disintegration.
  have row_18 : (gl_implication0 succ N) := by
    exact row_19.1.1.1
  -- chapter_61_line_17: GL tag expansion.
  have row_17 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_18
  -- chapter_61_line_16: GL tag implication.
  have row_16 : (N one) := by
    apply row_17
    exact row_21
  -- chapter_61_line_8: GL tag disintegration.
  have row_8 : (gl_fXYZ mul N N N) := by
    exact row_9.1.1.1.2
  -- chapter_61_line_7: GL tag expansion.
  have row_7 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_8
  -- chapter_61_line_6: GL tag disintegration.
  have row_6 : (gl_implication13 N N N mul) := by
    exact row_7.1.2
  -- chapter_61_line_5: GL tag expansion.
  have row_5 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication13] using row_6
  -- chapter_61_line_4: GL tag implication.
  have row_4 : (gl_existence1 N one v2 mul) := by
    apply row_5
    exact row_16
    exact row_13
  -- chapter_61_line_3: GL tag expansion.
  have row_3 : (¬ (∀ (v3 : α), ((N v3) → (¬ (mul one v2 v3))))) := by
    simpa only [gl_existence1] using row_4
  have exists_row_3 : ∃ (v3 : α), ((N v3) ∧ (mul one v2 v3)) := existsAndOfNotForallImpNot row_3
  obtain ⟨v3, witness_row_3⟩ := exists_row_3
  -- chapter_61_line_28: GL tag disintegration.
  have row_28 : (N v3) := by
    exact witness_row_3.1
  -- chapter_61_line_2: GL tag disintegration.
  have row_2 : (mul one v2 v3) := by
    exact witness_row_3.2
  have rule_row_23 := external_peano_externals_36_033 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_61_line_23: GL tag implication.
  have row_23 : (v3 = v2) := by
    apply rule_row_23
    exact row_28
    exact row_2
  -- chapter_61_line_1: GL tag equality1.
  have row_1 : (mul one v1 v2) := by
    have equality_source := row_2
    have equality_step_1 := row_22
    cases equality_step_1
    have equality_step_2 := row_23
    cases equality_step_2
    exact equality_source
  exact row_1

theorem fta_source_050
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    : (∀ (v1 : α), ((N v1) → (∀ (v2 : α), ((v1 = v2) → (gl_preorder N add v1 v2))))) := by
  intro v1
  intro premise_1
  intro v2
  intro premise_2
  -- chapter_62_line_13: GL tag task formulation.
  have row_13 : (N v1) := by
    exact premise_1
  -- chapter_62_line_11: GL tag task formulation.
  have row_11 : (v1 = v2) := by
    exact premise_2
  -- chapter_62_line_12: GL tag equality1.
  have row_12 : (N v2) := by
    have equality_source := row_13
    have equality_step_1 := row_11
    cases equality_step_1
    exact equality_source
  -- chapter_62_line_10: GL tag task formulation.
  have row_10 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_62_line_9: GL tag expansion.
  have row_9 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_10
  -- chapter_62_line_8: GL tag disintegration.
  have row_8 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_9.1.1.1
  -- chapter_62_line_7: GL tag expansion.
  have row_7 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_8
  -- chapter_62_line_14: GL tag disintegration.
  have row_14 : (N zero) := by
    exact row_7.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_62_line_6: GL tag disintegration.
  have row_6 : (gl_implication16 N zero add) := by
    exact row_7.1.1.1.1.1.1.2
  -- chapter_62_line_5: GL tag expansion.
  have row_5 : (∀ (w1 : α) (w2 : α), ((w1 = w2) → ((N w1) → ((N w2) → (add w1 zero w2))))) := by
    simpa only [gl_implication16] using row_6
  -- chapter_62_line_4: GL tag implication.
  have row_4 : (add v1 zero v2) := by
    apply row_5
    exact row_11
    exact row_13
    exact row_12
  -- chapter_62_line_3: GL tag expansion for integration.
  have row_3 : ((gl_preorder N add v1 v2) ↔ (¬ (∀ (v4 : α), ((N v4) → (¬ (add v1 v4 v2)))))) := by
    exact Iff.rfl
  -- chapter_62_line_2: GL tag reformulation for integration >[bound].
  have row_2 : (∀ (v3 : α), ((N v3) → ((add v1 v3 v2) → (gl_preorder N add v1 v2)))) := by
    intro v3
    intro integration_premise_1
    intro integration_premise_2
    apply (row_3).2
    intro universal_counterexample
    exact universal_counterexample v3 integration_premise_1 integration_premise_2
  -- chapter_62_line_1: GL tag implication.
  have row_1 : (gl_preorder N add v1 v2) := by
    apply row_2
    exact row_14
    exact row_4
  exact row_1

theorem fta_source_051
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_033 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α), ((x_1 x_7) → (∀ (x_8 : α), ((x_5 x_6 x_8 x_7) → (x_7 = x_8))))))))
    (external_peano_externals_36_018 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_5 x_6 x_7 x_8) → (x_5 x_7 x_6 x_8))))))
    : (∀ (v1 : α), ((N v1) → (∀ (v2 : α), ((v1 = v2) → (gl_preorder N mul v1 v2))))) := by
  intro v1
  intro premise_1
  intro v2
  intro premise_2
  -- chapter_63_line_24: GL tag task formulation.
  have row_24 : (v1 = v2) := by
    exact premise_2
  -- chapter_63_line_33: GL tag symmetry of equality.
  have row_33 : (v2 = v1) := by
    exact Eq.symm row_24
  -- chapter_63_line_23: GL tag task formulation.
  have row_23 : (N v1) := by
    exact premise_1
  -- chapter_63_line_22: GL tag equality1.
  have row_22 : (N v2) := by
    have equality_source := row_23
    have equality_step_1 := row_24
    cases equality_step_1
    exact equality_source
  -- chapter_63_line_12: GL tag task formulation.
  have row_12 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_63_line_11: GL tag expansion.
  have row_11 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_12
  -- chapter_63_line_30: GL tag disintegration.
  have row_30 : (succ one two) := by
    exact row_11.1.2
  -- chapter_63_line_13: GL tag disintegration.
  have row_13 : (succ zero one) := by
    exact row_11.1.1.2
  -- chapter_63_line_10: GL tag disintegration.
  have row_10 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_11.1.1.1
  -- chapter_63_line_21: GL tag expansion.
  have row_21 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_10
  -- chapter_63_line_29: GL tag disintegration.
  have row_29 : (gl_fXY succ N N) := by
    exact row_21.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_63_line_28: GL tag expansion.
  have row_28 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_29
  -- chapter_63_line_27: GL tag disintegration.
  have row_27 : (gl_implication0 succ N) := by
    exact row_28.1.1.1
  -- chapter_63_line_26: GL tag expansion.
  have row_26 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_27
  -- chapter_63_line_25: GL tag implication.
  have row_25 : (N one) := by
    apply row_26
    exact row_30
  -- chapter_63_line_20: GL tag disintegration.
  have row_20 : (gl_fXYZ mul N N N) := by
    exact row_21.1.1.1.2
  -- chapter_63_line_19: GL tag expansion.
  have row_19 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_20
  -- chapter_63_line_18: GL tag disintegration.
  have row_18 : (gl_implication13 N N N mul) := by
    exact row_19.1.2
  -- chapter_63_line_17: GL tag expansion.
  have row_17 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication13] using row_18
  -- chapter_63_line_16: GL tag implication.
  have row_16 : (gl_existence1 N one v2 mul) := by
    apply row_17
    exact row_25
    exact row_22
  -- chapter_63_line_15: GL tag expansion.
  have row_15 : (¬ (∀ (v5 : α), ((N v5) → (¬ (mul one v2 v5))))) := by
    simpa only [gl_existence1] using row_16
  have exists_row_15 : ∃ (v5 : α), ((N v5) ∧ (mul one v2 v5)) := existsAndOfNotForallImpNot row_15
  obtain ⟨v5, witness_row_15⟩ := exists_row_15
  -- chapter_63_line_32: GL tag disintegration.
  have row_32 : (N v5) := by
    exact witness_row_15.1
  -- chapter_63_line_14: GL tag disintegration.
  have row_14 : (mul one v2 v5) := by
    exact witness_row_15.2
  -- chapter_63_line_9: GL tag expansion for integration.
  have row_9 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_63_line_8: GL tag reformulation for integration and.
  have row_8 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_63_line_7: GL tag implication.
  have row_7 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_8
    exact row_10
    exact row_13
  have rule_row_31 := external_peano_externals_36_033 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_63_line_31: GL tag implication.
  have row_31 : (v5 = v2) := by
    apply rule_row_31
    exact row_32
    exact row_14
  have rule_row_6 := external_peano_externals_36_018 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_63_line_6: GL tag implication.
  have row_6 : (mul v2 one v5) := by
    apply rule_row_6
    exact row_14
  -- chapter_63_line_5: GL tag equality1.
  have row_5 : (mul v2 one v2) := by
    have equality_source := row_6
    have equality_step_1 := row_31
    cases equality_step_1
    exact equality_source
  -- chapter_63_line_4: GL tag equality1.
  have row_4 : (mul v1 one v2) := by
    have equality_source := row_5
    have equality_step_1 := row_33
    cases equality_step_1
    exact equality_source
  -- chapter_63_line_3: GL tag expansion for integration.
  have row_3 : ((gl_preorder N mul v1 v2) ↔ (¬ (∀ (v4 : α), ((N v4) → (¬ (mul v1 v4 v2)))))) := by
    exact Iff.rfl
  -- chapter_63_line_2: GL tag reformulation for integration >[bound].
  have row_2 : (∀ (v3 : α), ((N v3) → ((mul v1 v3 v2) → (gl_preorder N mul v1 v2)))) := by
    intro v3
    intro integration_premise_1
    intro integration_premise_2
    apply (row_3).2
    intro universal_counterexample
    exact universal_counterexample v3 integration_premise_1 integration_premise_2
  -- chapter_63_line_1: GL tag implication.
  have row_1 : (gl_preorder N mul v1 v2) := by
    apply row_2
    exact row_25
    exact row_4
  exact row_1

private theorem fta_source_059_induction_typing
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (v1 : α)
    (assumption_1 : (N v1))
    : (N v1) := by
  -- chapter_75_line_1: GL tag task formulation.
  have row_1 : (N v1) := by
    exact assumption_1
  exact row_1

private theorem fta_source_059_check_zero
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (v1 : α)
    (assumption_4 : (¬ (zero = v1)))
    (assumption_2 : (v1 = zero))
    : (gl_preorder N add one v1) := by
  -- chapter_76_line_4: GL tag task formulation.
  have row_4 : (¬ (zero = v1)) := by
    exact assumption_4
  -- chapter_76_line_3: GL tag symmetry of inequality.
  have row_3 : (¬ (v1 = zero)) := by
    exact fun equality => row_4 (Eq.symm equality)
  -- chapter_76_line_2: GL tag recursion.
  have row_2 : (v1 = zero) := by
    exact assumption_2
  -- chapter_76_line_1: GL tag vacuous truth.
  have row_1 : (gl_preorder N add one v1) := by
    exact False.elim (row_3 row_2)
  exact row_1

private theorem fta_source_059_check_induction_condition
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_015 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_3 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_022 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_6 x_8) → (x_4 x_6 x_7 x_8))))))
    (previous : α)
    (v1 : α)
    (assumption_13 : (succ previous v1))
    : (gl_preorder N add one v1) := by
  -- chapter_77_line_13: GL tag recursion.
  have row_13 : (succ previous v1) := by
    exact assumption_13
  -- chapter_77_line_10: GL tag task formulation.
  have row_10 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_77_line_9: GL tag expansion.
  have row_9 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_10
  -- chapter_77_line_11: GL tag disintegration.
  have row_11 : (succ zero one) := by
    exact row_9.1.1.2
  -- chapter_77_line_8: GL tag disintegration.
  have row_8 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_9.1.1.1
  -- chapter_77_line_19: GL tag expansion.
  have row_19 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_8
  -- chapter_77_line_18: GL tag disintegration.
  have row_18 : (gl_fXY succ N N) := by
    exact row_19.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_77_line_17: GL tag expansion.
  have row_17 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_18
  -- chapter_77_line_16: GL tag disintegration.
  have row_16 : (gl_implication0 succ N) := by
    exact row_17.1.1.1
  -- chapter_77_line_15: GL tag expansion.
  have row_15 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_16
  -- chapter_77_line_14: GL tag implication.
  have row_14 : (N previous) := by
    apply row_15
    exact row_13
  -- chapter_77_line_7: GL tag expansion for integration.
  have row_7 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_77_line_6: GL tag reformulation for integration and.
  have row_6 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_77_line_5: GL tag implication.
  have row_5 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_6
    exact row_8
    exact row_11
  have rule_row_12 := external_peano_externals_36_015 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_77_line_12: GL tag implication.
  have row_12 : (add previous one v1) := by
    apply rule_row_12
    exact row_13
  have rule_row_4 := external_peano_externals_36_022 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_77_line_4: GL tag implication.
  have row_4 : (add one previous v1) := by
    apply rule_row_4
    exact row_12
  -- chapter_77_line_3: GL tag expansion for integration.
  have row_3 : ((gl_preorder N add one v1) ↔ (¬ (∀ (v3 : α), ((N v3) → (¬ (add one v3 v1)))))) := by
    exact Iff.rfl
  -- chapter_77_line_2: GL tag reformulation for integration >[bound].
  have row_2 : (∀ (v2 : α), ((N v2) → ((add one v2 v1) → (gl_preorder N add one v1)))) := by
    intro v2
    intro integration_premise_1
    intro integration_premise_2
    apply (row_3).2
    intro universal_counterexample
    exact universal_counterexample v2 integration_premise_1 integration_premise_2
  -- chapter_77_line_1: GL tag implication.
  have row_1 : (gl_preorder N add one v1) := by
    apply row_2
    exact row_14
    exact row_4
  exact row_1

theorem fta_source_059
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_015 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_3 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_022 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_6 x_8) → (x_4 x_6 x_7 x_8))))))
    : (∀ (v1 : α), ((N v1) → ((¬ (zero = v1)) → (gl_preorder N add one v1)))) := by
  intro v1
  intro premise_1
  intro premise_2
  have inductionMember : N v1 := by
    have typingRule := fta_source_059_induction_typing N zero succ add mul one two identity anchor relationalInduction
    exact typingRule v1 premise_1
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorFTA, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : ((¬ (zero = zero)) → (gl_preorder N add one zero)) := by
    intro base_premise_1
    have zeroRule := fta_source_059_check_zero N zero succ add mul one two identity anchor relationalInduction
    exact zeroRule zero base_premise_1 rfl
  have inductionStep :
      ∀ induction_n, N induction_n → ((¬ (zero = induction_n)) → (gl_preorder N add one induction_n)) → ∀ induction_m, succ induction_n induction_m → ((¬ (zero = induction_m)) → (gl_preorder N add one induction_m)) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro step_premise_1
    have step_induction_assumption_1 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have stepRule := fta_source_059_check_induction_condition N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_015 external_peano_externals_36_022
    exact stepRule induction_n induction_m step_induction_assumption_1
  have inductionProperty : ((¬ (zero = v1)) → (gl_preorder N add one v1)) := by
    exact relationalInduction N zero succ add mul one two identity anchor
      (fun induction_value => ((¬ (zero = induction_value)) → (gl_preorder N add one induction_value)))
      v1
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty premise_2

private theorem fta_source_060_induction_typing
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (v1 : α)
    (assumption_1 : (N v1))
    : (N v1) := by
  -- chapter_78_line_1: GL tag task formulation.
  have row_1 : (N v1) := by
    exact assumption_1
  exact row_1

private theorem fta_source_060_check_zero
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (v1 : α)
    (assumption_2 : (v1 = zero))
    : (zero = v1) := by
  -- chapter_79_line_2: GL tag recursion.
  have row_2 : (v1 = zero) := by
    exact assumption_2
  -- chapter_79_line_1: GL tag symmetry of equality.
  have row_1 : (zero = v1) := by
    exact Eq.symm row_2
  exact row_1

private theorem fta_source_060_check_induction_condition
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_015 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_3 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_022 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_6 x_8) → (x_4 x_6 x_7 x_8))))))
    (previous : α)
    (v1 : α)
    (assumption_21 : ((N previous) → ((¬ (gl_preorder N add one previous)) → (zero = previous))))
    (assumption_18 : (succ previous v1))
    (assumption_5 : (¬ (gl_preorder N add one v1)))
    : (zero = v1) := by
  -- chapter_80_line_33: GL tag task formulation.
  have row_33 : ((gl_preorder N add one previous) → (gl_preorder N add one previous)) := by
    intro scope_premise_1
    exact scope_premise_1
  -- chapter_80_line_32: GL tag expansion.
  have row_32 : ((gl_preorder N add one previous) → (¬ (∀ (v4 : α), ((N v4) → (¬ (add one v4 previous)))))) := by
    simpa only [gl_preorder] using row_33
  -- chapter_80_line_38: GL tag disintegration.
  have row_38 : ((gl_preorder N add one previous) → (∃ (v4 : α), ((add one v4 previous) ∧ (N v4)))) := by
    intro scope_premise_1
    have scoped_fact_1 := row_32 scope_premise_1
    obtain ⟨v4, scoped_witness_bundle⟩ := existsAndOfNotForallImpNot scoped_fact_1
    exact ⟨v4, ⟨scoped_witness_bundle.2, scoped_witness_bundle.1⟩⟩
  -- chapter_80_line_31: GL tag disintegration.
  have row_31 : ((gl_preorder N add one previous) → (∃ (v4 : α), ((add one v4 previous) ∧ (N v4)))) := by
    intro scope_premise_1
    have scoped_fact_1 := row_32 scope_premise_1
    obtain ⟨v4, scoped_witness_bundle⟩ := existsAndOfNotForallImpNot scoped_fact_1
    exact ⟨v4, ⟨scoped_witness_bundle.2, scoped_witness_bundle.1⟩⟩
  -- chapter_80_line_21: GL tag recursion.
  have row_21 : ((N previous) → ((¬ (gl_preorder N add one previous)) → (zero = previous))) := by
    exact assumption_21
  -- chapter_80_line_18: GL tag recursion.
  have row_18 : (succ previous v1) := by
    exact assumption_18
  -- chapter_80_line_15: GL tag expansion for integration.
  have row_15 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_80_line_14: GL tag reformulation for integration and.
  have row_14 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_80_line_10: GL tag task formulation.
  have row_10 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_80_line_9: GL tag expansion.
  have row_9 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_10
  -- chapter_80_line_16: GL tag disintegration.
  have row_16 : (succ zero one) := by
    exact row_9.1.1.2
  -- chapter_80_line_8: GL tag disintegration.
  have row_8 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_9.1.1.1
  -- chapter_80_line_13: GL tag implication.
  have row_13 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_14
    exact row_8
    exact row_16
  have rule_row_17 := external_peano_externals_36_015 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_80_line_17: GL tag implication.
  have row_17 : (add previous one v1) := by
    apply rule_row_17
    exact row_18
  have rule_row_12 := external_peano_externals_36_022 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_80_line_12: GL tag implication.
  have row_12 : (add one previous v1) := by
    apply rule_row_12
    exact row_17
  -- chapter_80_line_7: GL tag expansion.
  have row_7 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_8
  -- chapter_80_line_36: GL tag disintegration.
  have row_36 : (gl_implication18 N succ add) := by
    exact row_7.1.1.1.1.2
  -- chapter_80_line_35: GL tag expansion.
  have row_35 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((succ w4 w5) → (add w3 w2 w5))))))))) := by
    simpa only [gl_implication18] using row_36
  -- chapter_80_line_30: GL tag disintegration.
  have row_30 : (gl_fXY succ N N) := by
    exact row_7.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_80_line_29: GL tag expansion.
  have row_29 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_30
  -- chapter_80_line_41: GL tag disintegration.
  have row_41 : (gl_implication0 succ N) := by
    exact row_29.1.1.1
  -- chapter_80_line_40: GL tag expansion.
  have row_40 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_41
  -- chapter_80_line_39: GL tag implication.
  have row_39 : (N previous) := by
    apply row_40
    exact row_18
  -- chapter_80_line_28: GL tag disintegration.
  have row_28 : (gl_implication4 N N succ) := by
    exact row_29.1.2
  -- chapter_80_line_27: GL tag expansion.
  have row_27 : (∀ (w1 : α), ((N w1) → (gl_existence0 N w1 succ))) := by
    simpa only [gl_implication4] using row_28
  -- chapter_80_line_26: GL tag implication.
  have row_26 : ((gl_preorder N add one previous) → (∀ (v4 : α), (((add one v4 previous) ∧ (N v4)) → (gl_existence0 N v4 succ)))) := by
    intro scope_premise_1
    intro v4
    intro witness_guard_1
    have scoped_fact_2 := row_31 scope_premise_1
    apply row_27
    exact witness_guard_1.2
  -- chapter_80_line_25: GL tag expansion.
  have row_25 : ((gl_preorder N add one previous) → (∀ (v4 : α), (((add one v4 previous) ∧ (N v4)) → (¬ (∀ (v3 : α), ((N v3) → (¬ (succ v4 v3)))))))) := by
    simpa only [gl_existence0] using row_26
  -- chapter_80_line_37: GL tag disintegration.
  have row_37 : ((gl_preorder N add one previous) → (∀ (v4 : α), (((add one v4 previous) ∧ (N v4)) → (∃ (v3 : α), ((succ v4 v3) ∧ (N v3)))))) := by
    intro scope_premise_1
    intro v4
    intro witness_guard_1
    have scoped_fact_1 := row_25 scope_premise_1 v4 witness_guard_1
    obtain ⟨v3, scoped_witness_bundle⟩ := existsAndOfNotForallImpNot scoped_fact_1
    exact ⟨v3, ⟨scoped_witness_bundle.2, scoped_witness_bundle.1⟩⟩
  -- chapter_80_line_34: GL tag implication.
  have row_34 : ((gl_preorder N add one previous) → (∀ (v4 : α), (((add one v4 previous) ∧ (N v4)) → (∀ (v3 : α), (((succ v4 v3) ∧ (N v3)) → (add one v3 v1)))))) := by
    intro scope_premise_1
    intro v4
    intro witness_guard_1
    intro v3
    intro witness_guard_2
    have scoped_fact_2 := row_31 scope_premise_1
    have scoped_fact_3 := row_37 scope_premise_1 v4 witness_guard_1
    have scoped_fact_4 := row_38 scope_premise_1
    apply row_35
    exact witness_guard_1.2
    exact witness_guard_2.1
    exact witness_guard_1.1
    exact row_18
  -- chapter_80_line_24: GL tag disintegration.
  have row_24 : ((gl_preorder N add one previous) → (∀ (v4 : α), (((add one v4 previous) ∧ (N v4)) → (∃ (v3 : α), ((succ v4 v3) ∧ (N v3)))))) := by
    intro scope_premise_1
    intro v4
    intro witness_guard_1
    have scoped_fact_1 := row_25 scope_premise_1 v4 witness_guard_1
    obtain ⟨v3, scoped_witness_bundle⟩ := existsAndOfNotForallImpNot scoped_fact_1
    exact ⟨v3, ⟨scoped_witness_bundle.2, scoped_witness_bundle.1⟩⟩
  -- chapter_80_line_6: GL tag disintegration.
  have row_6 : (N zero) := by
    exact row_7.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_80_line_5: GL tag task formulation.
  have row_5 : (¬ (gl_preorder N add one v1)) := by
    exact assumption_5
  -- chapter_80_line_4: GL tag expansion.
  have row_4 : (gl_implication24 N one v1 add) := by
    simp only [gl_implication24]
    intro existence_witness_1 compact_premise compact_negated
    have unfolded_row_5 := row_5
    simp only [gl_preorder] at unfolded_row_5
    apply unfolded_row_5
    intro positive_row_5
    exact positive_row_5 existence_witness_1 compact_premise compact_negated
  -- chapter_80_line_3: GL tag expansion.
  have row_3 : (∀ (w1 : α), ((N w1) → (¬ (add one w1 v1)))) := by
    simpa only [gl_implication24] using row_4
  -- chapter_80_line_23: GL tag implication.
  have row_23 : ((gl_preorder N add one previous) → (∀ (v4 : α), (((add one v4 previous) ∧ (N v4)) → (∀ (v3 : α), (((succ v4 v3) ∧ (N v3)) → (¬ (add one v3 v1))))))) := by
    intro scope_premise_1
    intro v4
    intro witness_guard_1
    intro v3
    intro witness_guard_2
    have scoped_fact_2 := row_24 scope_premise_1 v4 witness_guard_1
    apply row_3
    exact witness_guard_2.2
  -- chapter_80_line_22: GL tag contradiction.
  have row_22 : (¬ (gl_preorder N add one previous)) := by
    intro contradiction_assumption
    obtain ⟨v4, contradiction_witness_guard_1⟩ := row_38 contradiction_assumption
    obtain ⟨v3, contradiction_witness_guard_2⟩ := row_37 contradiction_assumption v4 contradiction_witness_guard_1
    have scoped_contradiction := row_23 contradiction_assumption v4 contradiction_witness_guard_1 v3 contradiction_witness_guard_2
    have scoped_contradiction_2 := row_34 contradiction_assumption v4 contradiction_witness_guard_1 v3 contradiction_witness_guard_2
    exact scoped_contradiction scoped_contradiction_2
  -- chapter_80_line_20: GL tag implication.
  have row_20 : (zero = previous) := by
    apply row_21
    exact row_39
    exact row_22
  -- chapter_80_line_19: GL tag symmetry of equality.
  have row_19 : (previous = zero) := by
    exact Eq.symm row_20
  -- chapter_80_line_11: GL tag equality1.
  have row_11 : (add one zero v1) := by
    have equality_source := row_12
    have equality_step_1 := row_19
    cases equality_step_1
    exact equality_source
  -- chapter_80_line_2: GL tag implication.
  have row_2 : (¬ (add one zero v1)) := by
    apply row_3
    exact row_6
  -- chapter_80_line_1: GL tag vacuous truth.
  have row_1 : (zero = v1) := by
    exact False.elim (row_2 row_11)
  exact row_1

theorem fta_source_060
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_015 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_3 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_022 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_6 x_8) → (x_4 x_6 x_7 x_8))))))
    : (∀ (v1 : α), ((N v1) → ((¬ (gl_preorder N add one v1)) → (zero = v1)))) := by
  intro v1
  intro premise_1
  intro premise_2
  have inductionMember : N v1 := by
    have typingRule := fta_source_060_induction_typing N zero succ add mul one two identity anchor relationalInduction
    exact typingRule v1 premise_1
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorFTA, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : ((¬ (gl_preorder N add one zero)) → (zero = zero)) := by
    intro base_premise_1
    have zeroRule := fta_source_060_check_zero N zero succ add mul one two identity anchor relationalInduction
    exact zeroRule zero rfl
  have inductionStep :
      ∀ induction_n, N induction_n → ((¬ (gl_preorder N add one induction_n)) → (zero = induction_n)) → ∀ induction_m, succ induction_n induction_m → ((¬ (gl_preorder N add one induction_m)) → (zero = induction_m)) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro step_premise_1
    have step_induction_assumption_1 :
        ((N induction_n) → ((¬ (gl_preorder N add one induction_n)) → (zero = induction_n))) := by
      intro step_induction_assumption_1_premise_1
      intro step_induction_assumption_1_premise_2
      apply induction_hypothesis
      all_goals assumption
    have step_induction_assumption_2 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have stepRule := fta_source_060_check_induction_condition N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_015 external_peano_externals_36_022
    exact stepRule induction_n induction_m step_induction_assumption_1 step_induction_assumption_2 step_premise_1
  have inductionProperty : ((¬ (gl_preorder N add one v1)) → (zero = v1)) := by
    exact relationalInduction N zero succ add mul one two identity anchor
      (fun induction_value => ((¬ (gl_preorder N add one induction_value)) → (zero = induction_value)))
      v1
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty premise_2

theorem fta_source_066
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_017 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_6 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_021 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_6 x_8) → (x_3 x_7 x_8))))))
    : (∀ (v1 : α), ((gl_preorder N add one v1) → (¬ (zero = v1)))) := by
  intro v1
  intro premise_1
  intro reductio
  -- chapter_88_line_21: GL tag task formulation.
  have row_21 : (zero = v1) := by
    exact reductio
  -- chapter_88_line_20: GL tag symmetry of equality.
  have row_20 : (v1 = zero) := by
    exact Eq.symm row_21
  -- chapter_88_line_16: GL tag expansion for integration.
  have row_16 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_88_line_15: GL tag reformulation for integration and.
  have row_15 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_88_line_11: GL tag task formulation.
  have row_11 : (gl_preorder N add one v1) := by
    exact premise_1
  -- chapter_88_line_10: GL tag expansion.
  have row_10 : (¬ (∀ (v2 : α), ((N v2) → (¬ (add one v2 v1))))) := by
    simpa only [gl_preorder] using row_11
  have exists_row_10 : ∃ (v2 : α), ((N v2) ∧ (add one v2 v1)) := existsAndOfNotForallImpNot row_10
  obtain ⟨v2, witness_row_10⟩ := exists_row_10
  -- chapter_88_line_19: GL tag disintegration.
  have row_19 : (add one v2 v1) := by
    exact witness_row_10.2
  -- chapter_88_line_9: GL tag disintegration.
  have row_9 : (N v2) := by
    exact witness_row_10.1
  -- chapter_88_line_8: GL tag task formulation.
  have row_8 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_88_line_7: GL tag expansion.
  have row_7 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_8
  -- chapter_88_line_17: GL tag disintegration.
  have row_17 : (succ zero one) := by
    exact row_7.1.1.2
  -- chapter_88_line_6: GL tag disintegration.
  have row_6 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_7.1.1.1
  -- chapter_88_line_14: GL tag implication.
  have row_14 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_15
    exact row_6
    exact row_17
  have rule_row_18 := external_peano_externals_36_017 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_88_line_18: GL tag implication.
  have row_18 : (add v2 one v1) := by
    apply rule_row_18
    exact row_19
  have rule_row_13 := external_peano_externals_36_021 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_88_line_13: GL tag implication.
  have row_13 : (succ v2 v1) := by
    apply rule_row_13
    exact row_18
  -- chapter_88_line_12: GL tag equality1.
  have row_12 : (succ v2 zero) := by
    have equality_source := row_13
    have equality_step_1 := row_20
    cases equality_step_1
    exact equality_source
  -- chapter_88_line_5: GL tag expansion.
  have row_5 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_6
  -- chapter_88_line_4: GL tag disintegration.
  have row_4 : (gl_implication6 N zero succ) := by
    exact row_5.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_88_line_3: GL tag expansion.
  have row_3 : (∀ (w1 : α), ((N w1) → (¬ (succ w1 zero)))) := by
    simpa only [gl_implication6] using row_4
  -- chapter_88_line_2: GL tag implication.
  have row_2 : (¬ (succ v2 zero)) := by
    apply row_3
    exact row_9
  -- chapter_88_line_1: GL tag contradiction.
  have row_1 : (¬ (zero = v1)) := by
    exact False.elim (row_2 row_12)
  exact row_1 reductio

theorem fta_source_072
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_024 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_8 x_2) → ((x_1 x_8) → (x_2 = x_7)))))))
    : (∀ (v1 : α), ((gl_preorder N add two v1) → (¬ (zero = v1)))) := by
  intro v1
  intro premise_1
  intro reductio
  -- chapter_100_line_26: GL tag task formulation.
  have row_26 : (zero = v1) := by
    exact reductio
  -- chapter_100_line_25: GL tag symmetry of equality.
  have row_25 : (v1 = zero) := by
    exact Eq.symm row_26
  -- chapter_100_line_24: GL tag task formulation.
  have row_24 : (gl_preorder N add two v1) := by
    exact premise_1
  -- chapter_100_line_23: GL tag expansion.
  have row_23 : (¬ (∀ (v4 : α), ((N v4) → (¬ (add two v4 v1))))) := by
    simpa only [gl_preorder] using row_24
  have exists_row_23 : ∃ (v4 : α), ((N v4) ∧ (add two v4 v1)) := existsAndOfNotForallImpNot row_23
  obtain ⟨v4, witness_row_23⟩ := exists_row_23
  -- chapter_100_line_27: GL tag disintegration.
  have row_27 : (N v4) := by
    exact witness_row_23.1
  -- chapter_100_line_22: GL tag disintegration.
  have row_22 : (add two v4 v1) := by
    exact witness_row_23.2
  -- chapter_100_line_21: GL tag equality1.
  have row_21 : (add two v4 zero) := by
    have equality_source := row_22
    have equality_step_1 := row_25
    cases equality_step_1
    exact equality_source
  -- chapter_100_line_19: GL tag expansion for integration.
  have row_19 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_100_line_18: GL tag reformulation for integration and.
  have row_18 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_100_line_4: GL tag task formulation.
  have row_4 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_100_line_3: GL tag expansion.
  have row_3 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_4
  -- chapter_100_line_20: GL tag disintegration.
  have row_20 : (succ zero one) := by
    exact row_3.1.1.2
  -- chapter_100_line_10: GL tag disintegration.
  have row_10 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_3.1.1.1
  -- chapter_100_line_17: GL tag implication.
  have row_17 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_18
    exact row_10
    exact row_20
  have rule_row_16 := external_peano_externals_36_024 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_100_line_16: GL tag implication.
  have row_16 : (zero = two) := by
    apply rule_row_16
    exact row_21
    exact row_27
  -- chapter_100_line_9: GL tag expansion.
  have row_9 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_10
  -- chapter_100_line_15: GL tag disintegration.
  have row_15 : (gl_fXY succ N N) := by
    exact row_9.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_100_line_14: GL tag expansion.
  have row_14 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_15
  -- chapter_100_line_13: GL tag disintegration.
  have row_13 : (gl_implication0 succ N) := by
    exact row_14.1.1.1
  -- chapter_100_line_12: GL tag expansion.
  have row_12 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_13
  -- chapter_100_line_8: GL tag disintegration.
  have row_8 : (gl_implication6 N zero succ) := by
    exact row_9.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_100_line_7: GL tag expansion.
  have row_7 : (∀ (w1 : α), ((N w1) → (¬ (succ w1 zero)))) := by
    simpa only [gl_implication6] using row_8
  -- chapter_100_line_2: GL tag disintegration.
  have row_2 : (succ one two) := by
    exact row_3.1.2
  -- chapter_100_line_11: GL tag implication.
  have row_11 : (N one) := by
    apply row_12
    exact row_2
  -- chapter_100_line_6: GL tag implication.
  have row_6 : (¬ (succ one zero)) := by
    apply row_7
    exact row_11
  -- chapter_100_line_5: GL tag equality1.
  have row_5 : (¬ (succ one two)) := by
    have equality_source := row_6
    have equality_step_1 := row_16
    cases equality_step_1
    exact equality_source
  -- chapter_100_line_1: GL tag contradiction.
  have row_1 : (¬ (zero = v1)) := by
    exact False.elim (row_5 row_2)
  exact row_1 reductio

theorem fta_source_073
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_027 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_8 x_6) → (x_4 x_8 x_7 x_6))))))
    (external_peano_externals_36_021 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_6 x_8) → (x_3 x_7 x_8))))))
    : (∀ (v1 : α), ((gl_preorder N add two v1) → (¬ (one = v1)))) := by
  intro v1
  intro premise_1
  intro reductio
  -- chapter_101_line_38: GL tag task formulation.
  have row_38 : (one = v1) := by
    exact reductio
  -- chapter_101_line_42: GL tag symmetry of equality.
  have row_42 : (v1 = one) := by
    exact Eq.symm row_38
  -- chapter_101_line_16: GL tag expansion for integration.
  have row_16 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_101_line_15: GL tag reformulation for integration and.
  have row_15 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_101_line_11: GL tag task formulation.
  have row_11 : (gl_preorder N add two v1) := by
    exact premise_1
  -- chapter_101_line_10: GL tag expansion.
  have row_10 : (¬ (∀ (v2 : α), ((N v2) → (¬ (add two v2 v1))))) := by
    simpa only [gl_preorder] using row_11
  have exists_row_10 : ∃ (v2 : α), ((N v2) ∧ (add two v2 v1)) := existsAndOfNotForallImpNot row_10
  obtain ⟨v2, witness_row_10⟩ := exists_row_10
  -- chapter_101_line_41: GL tag disintegration.
  have row_41 : (add two v2 v1) := by
    exact witness_row_10.2
  -- chapter_101_line_40: GL tag equality1.
  have row_40 : (add two v2 one) := by
    have equality_source := row_41
    have equality_step_1 := row_42
    cases equality_step_1
    exact equality_source
  -- chapter_101_line_9: GL tag disintegration.
  have row_9 : (N v2) := by
    exact witness_row_10.1
  -- chapter_101_line_8: GL tag task formulation.
  have row_8 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_101_line_7: GL tag expansion.
  have row_7 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_8
  -- chapter_101_line_30: GL tag disintegration.
  have row_30 : (succ one two) := by
    exact row_7.1.2
  -- chapter_101_line_37: GL tag equality1.
  have row_37 : (succ v1 two) := by
    have equality_source := row_30
    have equality_step_1 := row_38
    cases equality_step_1
    exact equality_source
  -- chapter_101_line_17: GL tag disintegration.
  have row_17 : (succ zero one) := by
    exact row_7.1.1.2
  -- chapter_101_line_6: GL tag disintegration.
  have row_6 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_7.1.1.1
  -- chapter_101_line_14: GL tag implication.
  have row_14 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_15
    exact row_6
    exact row_17
  have rule_row_39 := external_peano_externals_36_027 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_101_line_39: GL tag implication.
  have row_39 : (add v2 two one) := by
    apply rule_row_39
    exact row_40
  -- chapter_101_line_5: GL tag expansion.
  have row_5 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_6
  -- chapter_101_line_36: GL tag disintegration.
  have row_36 : (gl_implication17 N succ add) := by
    exact row_5.1.1.1.1.1.2
  -- chapter_101_line_35: GL tag expansion.
  have row_35 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication17] using row_36
  -- chapter_101_line_33: GL tag disintegration.
  have row_33 : (gl_implication7 N succ) := by
    exact row_5.1.1.1.1.1.1.1.1.1.2
  -- chapter_101_line_32: GL tag expansion.
  have row_32 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w2 w1) → (∀ (w3 : α), ((succ w3 w1) → (w2 = w3))))))) := by
    simpa only [gl_implication7] using row_33
  -- chapter_101_line_29: GL tag disintegration.
  have row_29 : (gl_fXY succ N N) := by
    exact row_5.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_101_line_28: GL tag expansion.
  have row_28 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_29
  -- chapter_101_line_27: GL tag disintegration.
  have row_27 : (gl_implication0 succ N) := by
    exact row_28.1.1.1
  -- chapter_101_line_26: GL tag expansion.
  have row_26 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_27
  -- chapter_101_line_25: GL tag implication.
  have row_25 : (N one) := by
    apply row_26
    exact row_30
  -- chapter_101_line_24: GL tag disintegration.
  have row_24 : (gl_fXYZ add N N N) := by
    exact row_5.1.1.1.1.1.1.1.1.2
  -- chapter_101_line_23: GL tag expansion.
  have row_23 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_24
  -- chapter_101_line_46: GL tag disintegration.
  have row_46 : (gl_implication10 add N) := by
    exact row_23.1.1.2
  -- chapter_101_line_45: GL tag expansion.
  have row_45 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication10] using row_46
  -- chapter_101_line_44: GL tag implication.
  have row_44 : (N v1) := by
    apply row_45
    exact row_41
  -- chapter_101_line_22: GL tag disintegration.
  have row_22 : (gl_implication13 N N N add) := by
    exact row_23.1.2
  -- chapter_101_line_21: GL tag expansion.
  have row_21 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication13] using row_22
  -- chapter_101_line_20: GL tag implication.
  have row_20 : (gl_existence1 N v2 one add) := by
    apply row_21
    exact row_9
    exact row_25
  -- chapter_101_line_19: GL tag expansion.
  have row_19 : (¬ (∀ (v4 : α), ((N v4) → (¬ (add v2 one v4))))) := by
    simpa only [gl_existence1] using row_20
  have exists_row_19 : ∃ (v4 : α), ((N v4) ∧ (add v2 one v4)) := existsAndOfNotForallImpNot row_19
  obtain ⟨v4, witness_row_19⟩ := exists_row_19
  -- chapter_101_line_18: GL tag disintegration.
  have row_18 : (add v2 one v4) := by
    exact witness_row_19.2
  -- chapter_101_line_43: GL tag equality1.
  have row_43 : (add v2 v1 v4) := by
    have equality_source := row_18
    have equality_step_1 := row_38
    cases equality_step_1
    exact equality_source
  -- chapter_101_line_34: GL tag implication.
  have row_34 : (succ v4 one) := by
    apply row_35
    exact row_44
    exact row_37
    exact row_43
    exact row_39
  -- chapter_101_line_31: GL tag implication.
  have row_31 : (v4 = zero) := by
    apply row_32
    exact row_25
    exact row_34
    exact row_17
  have rule_row_13 := external_peano_externals_36_021 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_101_line_13: GL tag implication.
  have row_13 : (succ v2 v4) := by
    apply rule_row_13
    exact row_18
  -- chapter_101_line_12: GL tag equality1.
  have row_12 : (succ v2 zero) := by
    have equality_source := row_13
    have equality_step_1 := row_31
    cases equality_step_1
    exact equality_source
  -- chapter_101_line_4: GL tag disintegration.
  have row_4 : (gl_implication6 N zero succ) := by
    exact row_5.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_101_line_3: GL tag expansion.
  have row_3 : (∀ (w1 : α), ((N w1) → (¬ (succ w1 zero)))) := by
    simpa only [gl_implication6] using row_4
  -- chapter_101_line_2: GL tag implication.
  have row_2 : (¬ (succ v2 zero)) := by
    apply row_3
    exact row_9
  -- chapter_101_line_1: GL tag contradiction.
  have row_1 : (¬ (one = v1)) := by
    exact False.elim (row_2 row_12)
  exact row_1 reductio

theorem fta_source_075
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_024 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_8 x_2) → ((x_1 x_8) → (x_2 = x_7)))))))
    : (∀ (v1 : α), ((gl_preorder N add v1 zero) → (zero = v1))) := by
  intro v1
  intro premise_1
  -- chapter_105_line_11: GL tag task formulation.
  have row_11 : (gl_preorder N add v1 zero) := by
    exact premise_1
  -- chapter_105_line_10: GL tag expansion.
  have row_10 : (¬ (∀ (v2 : α), ((N v2) → (¬ (add v1 v2 zero))))) := by
    simpa only [gl_preorder] using row_11
  have exists_row_10 : ∃ (v2 : α), ((N v2) ∧ (add v1 v2 zero)) := existsAndOfNotForallImpNot row_10
  obtain ⟨v2, witness_row_10⟩ := exists_row_10
  -- chapter_105_line_12: GL tag disintegration.
  have row_12 : (N v2) := by
    exact witness_row_10.1
  -- chapter_105_line_9: GL tag disintegration.
  have row_9 : (add v1 v2 zero) := by
    exact witness_row_10.2
  -- chapter_105_line_7: GL tag task formulation.
  have row_7 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_105_line_6: GL tag expansion.
  have row_6 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_7
  -- chapter_105_line_8: GL tag disintegration.
  have row_8 : (succ zero one) := by
    exact row_6.1.1.2
  -- chapter_105_line_5: GL tag disintegration.
  have row_5 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_6.1.1.1
  -- chapter_105_line_4: GL tag expansion for integration.
  have row_4 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_105_line_3: GL tag reformulation for integration and.
  have row_3 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_105_line_2: GL tag implication.
  have row_2 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_3
    exact row_5
    exact row_8
  have rule_row_1 := external_peano_externals_36_024 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_105_line_1: GL tag implication.
  have row_1 : (zero = v1) := by
    apply rule_row_1
    exact row_9
    exact row_12
  exact row_1

theorem fta_source_076
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_031 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α), ((x_1 x_7) → (∀ (x_8 : α), ((x_5 x_2 x_7 x_8) → (x_2 = x_8))))))))
    : (∀ (v1 : α), ((gl_preorder N mul zero v1) → (zero = v1))) := by
  intro v1
  intro premise_1
  -- chapter_106_line_11: GL tag task formulation.
  have row_11 : (gl_preorder N mul zero v1) := by
    exact premise_1
  -- chapter_106_line_10: GL tag expansion.
  have row_10 : (¬ (∀ (v2 : α), ((N v2) → (¬ (mul zero v2 v1))))) := by
    simpa only [gl_preorder] using row_11
  have exists_row_10 : ∃ (v2 : α), ((N v2) ∧ (mul zero v2 v1)) := existsAndOfNotForallImpNot row_10
  obtain ⟨v2, witness_row_10⟩ := exists_row_10
  -- chapter_106_line_12: GL tag disintegration.
  have row_12 : (N v2) := by
    exact witness_row_10.1
  -- chapter_106_line_9: GL tag disintegration.
  have row_9 : (mul zero v2 v1) := by
    exact witness_row_10.2
  -- chapter_106_line_7: GL tag task formulation.
  have row_7 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_106_line_6: GL tag expansion.
  have row_6 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_7
  -- chapter_106_line_8: GL tag disintegration.
  have row_8 : (succ zero one) := by
    exact row_6.1.1.2
  -- chapter_106_line_5: GL tag disintegration.
  have row_5 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_6.1.1.1
  -- chapter_106_line_4: GL tag expansion for integration.
  have row_4 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_106_line_3: GL tag reformulation for integration and.
  have row_3 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_106_line_2: GL tag implication.
  have row_2 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_3
    exact row_5
    exact row_8
  have rule_row_1 := external_peano_externals_36_031 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_106_line_1: GL tag implication.
  have row_1 : (zero = v1) := by
    apply rule_row_1
    exact row_12
    exact row_9
  exact row_1

theorem fta_source_077
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    : (gl_AnchorGauss N zero succ add mul one two identity) := by
  -- chapter_107_line_6: GL tag task formulation.
  have row_6 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_107_line_5: GL tag expansion.
  have row_5 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_6
  -- chapter_107_line_9: GL tag disintegration.
  have row_9 : (succ one two) := by
    exact row_5.1.2
  -- chapter_107_line_8: GL tag disintegration.
  have row_8 : (succ zero one) := by
    exact row_5.1.1.2
  -- chapter_107_line_7: GL tag disintegration.
  have row_7 : (gl_identity N identity) := by
    exact row_5.2
  -- chapter_107_line_4: GL tag disintegration.
  have row_4 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_5.1.1.1
  -- chapter_107_line_3: GL tag expansion for integration.
  have row_3 : ((gl_AnchorGauss N zero succ add mul one two identity) ↔ ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity))) := by
    exact Iff.rfl
  -- chapter_107_line_2: GL tag reformulation for integration and.
  have row_2 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → ((succ one two) → ((gl_identity N identity) → (gl_AnchorGauss N zero succ add mul one two identity))))) := by
    intro integration_premise_1
    intro integration_premise_2
    intro integration_premise_3
    intro integration_premise_4
    simp only [gl_AnchorGauss]
    exact ⟨⟨⟨integration_premise_1, integration_premise_2⟩, integration_premise_3⟩, integration_premise_4⟩
  -- chapter_107_line_1: GL tag implication.
  have row_1 : (gl_AnchorGauss N zero succ add mul one two identity) := by
    apply row_2
    exact row_4
    exact row_8
    exact row_9
    exact row_7
  exact row_1

theorem fta_source_078
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    : (gl_AnchorPeano N zero succ add mul one) := by
  -- chapter_108_line_6: GL tag task formulation.
  have row_6 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_108_line_5: GL tag expansion.
  have row_5 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_6
  -- chapter_108_line_7: GL tag disintegration.
  have row_7 : (succ zero one) := by
    exact row_5.1.1.2
  -- chapter_108_line_4: GL tag disintegration.
  have row_4 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_5.1.1.1
  -- chapter_108_line_3: GL tag expansion for integration.
  have row_3 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_108_line_2: GL tag reformulation for integration and.
  have row_2 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_108_line_1: GL tag implication.
  have row_1 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_2
    exact row_4
    exact row_7
  exact row_1

theorem fta_source_000
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_006 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (x_4 x_8 x_7 x_9))))))
    (external_peano_externals_36_002 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (∀ (w4 : α) (w5 : α), ((add w4 w2 w5) → (∀ (w6 : α), ((add w3 w4 w6) → (add w5 w1 w6))))))))))
    (external_peano_externals_36_004 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (∀ (x_10 : α), ((x_4 x_7 x_10 x_9) → (x_8 = x_10))))))))
    : (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α) (v5 : α), ((add v4 v2 v5) → ((gl_preorder N add v3 v5) → (gl_preorder N add v1 v4)))))) := by
  intro v1
  intro v2
  intro v3
  intro premise_1
  intro v4
  intro v5
  intro premise_2
  intro premise_3
  -- chapter_0_line_29: GL tag task formulation.
  have row_29 : (add v4 v2 v5) := by
    exact premise_2
  -- chapter_0_line_28: GL tag theorem.
  have row_28 := fta_source_001 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_006 external_peano_externals_36_004
  -- chapter_0_line_26: GL tag task formulation.
  have row_26 : (gl_preorder N add v3 v5) := by
    exact premise_3
  -- chapter_0_line_25: GL tag expansion.
  have row_25 : (¬ (∀ (v7 : α), ((N v7) → (¬ (add v3 v7 v5))))) := by
    simpa only [gl_preorder] using row_26
  have exists_row_25 : ∃ (v7 : α), ((N v7) ∧ (add v3 v7 v5)) := existsAndOfNotForallImpNot row_25
  obtain ⟨v7, witness_row_25⟩ := exists_row_25
  -- chapter_0_line_32: GL tag disintegration.
  have row_32 : (add v3 v7 v5) := by
    exact witness_row_25.2
  -- chapter_0_line_24: GL tag disintegration.
  have row_24 : (N v7) := by
    exact witness_row_25.1
  -- chapter_0_line_23: GL tag task formulation.
  have row_23 : (add v1 v2 v3) := by
    exact premise_1
  -- chapter_0_line_8: GL tag expansion for integration.
  have row_8 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_0_line_7: GL tag reformulation for integration and.
  have row_7 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_0_line_4: GL tag task formulation.
  have row_4 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_0_line_10: GL tag expansion.
  have row_10 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_4
  -- chapter_0_line_11: GL tag disintegration.
  have row_11 : (succ zero one) := by
    exact row_10.1.1.2
  -- chapter_0_line_9: GL tag disintegration.
  have row_9 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_10.1.1.1
  -- chapter_0_line_19: GL tag expansion.
  have row_19 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_9
  -- chapter_0_line_18: GL tag disintegration.
  have row_18 : (gl_fXYZ add N N N) := by
    exact row_19.1.1.1.1.1.1.1.1.2
  -- chapter_0_line_17: GL tag expansion.
  have row_17 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_18
  -- chapter_0_line_22: GL tag disintegration.
  have row_22 : (gl_implication8 add N) := by
    exact row_17.1.1.1.1
  -- chapter_0_line_21: GL tag expansion.
  have row_21 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_22
  -- chapter_0_line_20: GL tag implication.
  have row_20 : (N v1) := by
    apply row_21
    exact row_23
  -- chapter_0_line_16: GL tag disintegration.
  have row_16 : (gl_implication13 N N N add) := by
    exact row_17.1.2
  -- chapter_0_line_15: GL tag expansion.
  have row_15 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication13] using row_16
  -- chapter_0_line_14: GL tag implication.
  have row_14 : (gl_existence1 N v7 v1 add) := by
    apply row_15
    exact row_24
    exact row_20
  -- chapter_0_line_13: GL tag expansion.
  have row_13 : (¬ (∀ (v6 : α), ((N v6) → (¬ (add v7 v1 v6))))) := by
    simpa only [gl_existence1] using row_14
  have exists_row_13 : ∃ (v6 : α), ((N v6) ∧ (add v7 v1 v6)) := existsAndOfNotForallImpNot row_13
  obtain ⟨v6, witness_row_13⟩ := exists_row_13
  -- chapter_0_line_12: GL tag disintegration.
  have row_12 : (add v7 v1 v6) := by
    exact witness_row_13.2
  -- chapter_0_line_6: GL tag implication.
  have row_6 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_7
    exact row_9
    exact row_11
  have rule_row_31 := external_peano_externals_36_006 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_0_line_31: GL tag implication.
  have row_31 : (add v2 v1 v3) := by
    apply rule_row_31
    exact row_23
  have rule_row_30 := external_peano_externals_36_002 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_0_line_30: GL tag implication.
  have row_30 : (add v6 v2 v5) := by
    apply rule_row_30
    exact row_31
    exact row_12
    exact row_32
  -- chapter_0_line_27: GL tag implication.
  have row_27 : (v6 = v4) := by
    apply row_28
    exact row_30
    exact row_29
  have rule_row_5 := external_peano_externals_36_006 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_0_line_5: GL tag implication.
  have row_5 : (add v1 v7 v6) := by
    apply rule_row_5
    exact row_12
  -- chapter_0_line_3: GL tag theorem.
  have row_3 := fta_source_002 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_0_line_2: GL tag implication.
  have row_2 : (gl_preorder N add v1 v6) := by
    apply row_3
    exact row_5
  -- chapter_0_line_1: GL tag equality1.
  have row_1 : (gl_preorder N add v1 v4) := by
    have equality_source := row_2
    have equality_step_1 := row_27
    cases equality_step_1
    exact equality_source
  exact row_1

theorem fta_source_012
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_006 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (x_4 x_8 x_7 x_9))))))
    : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (v3 : α) (v4 : α), ((succ v3 v4) → ((gl_preorder N add v2 v4) → (gl_preorder N add v1 v3)))))) := by
  intro v1
  intro v2
  intro premise_1
  intro v3
  intro v4
  intro premise_2
  intro premise_3
  -- chapter_14_line_32: GL tag task formulation.
  have row_32 : (succ v3 v4) := by
    exact premise_2
  -- chapter_14_line_28: GL tag task formulation.
  have row_28 : (gl_preorder N add v2 v4) := by
    exact premise_3
  -- chapter_14_line_27: GL tag expansion.
  have row_27 : (¬ (∀ (v6 : α), ((N v6) → (¬ (add v2 v6 v4))))) := by
    simpa only [gl_preorder] using row_28
  have exists_row_27 : ∃ (v6 : α), ((N v6) ∧ (add v2 v6 v4)) := existsAndOfNotForallImpNot row_27
  obtain ⟨v6, witness_row_27⟩ := exists_row_27
  -- chapter_14_line_37: GL tag disintegration.
  have row_37 : (add v2 v6 v4) := by
    exact witness_row_27.2
  -- chapter_14_line_26: GL tag disintegration.
  have row_26 : (N v6) := by
    exact witness_row_27.1
  -- chapter_14_line_25: GL tag task formulation.
  have row_25 : (succ v1 v2) := by
    exact premise_1
  -- chapter_14_line_8: GL tag expansion for integration.
  have row_8 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_14_line_7: GL tag reformulation for integration and.
  have row_7 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_14_line_4: GL tag task formulation.
  have row_4 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_14_line_10: GL tag expansion.
  have row_10 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_4
  -- chapter_14_line_11: GL tag disintegration.
  have row_11 : (succ zero one) := by
    exact row_10.1.1.2
  -- chapter_14_line_9: GL tag disintegration.
  have row_9 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_10.1.1.1
  -- chapter_14_line_19: GL tag expansion.
  have row_19 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_9
  -- chapter_14_line_35: GL tag disintegration.
  have row_35 : (gl_implication17 N succ add) := by
    exact row_19.1.1.1.1.1.2
  -- chapter_14_line_34: GL tag expansion.
  have row_34 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication17] using row_35
  -- chapter_14_line_31: GL tag disintegration.
  have row_31 : (gl_implication7 N succ) := by
    exact row_19.1.1.1.1.1.1.1.1.1.2
  -- chapter_14_line_30: GL tag expansion.
  have row_30 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w2 w1) → (∀ (w3 : α), ((succ w3 w1) → (w2 = w3))))))) := by
    simpa only [gl_implication7] using row_31
  -- chapter_14_line_24: GL tag disintegration.
  have row_24 : (gl_fXY succ N N) := by
    exact row_19.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_14_line_23: GL tag expansion.
  have row_23 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_24
  -- chapter_14_line_22: GL tag disintegration.
  have row_22 : (gl_implication0 succ N) := by
    exact row_23.1.1.1
  -- chapter_14_line_21: GL tag expansion.
  have row_21 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_22
  -- chapter_14_line_20: GL tag implication.
  have row_20 : (N v1) := by
    apply row_21
    exact row_25
  -- chapter_14_line_18: GL tag disintegration.
  have row_18 : (gl_fXYZ add N N N) := by
    exact row_19.1.1.1.1.1.1.1.1.2
  -- chapter_14_line_17: GL tag expansion.
  have row_17 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_18
  -- chapter_14_line_40: GL tag disintegration.
  have row_40 : (gl_implication10 add N) := by
    exact row_17.1.1.2
  -- chapter_14_line_39: GL tag expansion.
  have row_39 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication10] using row_40
  -- chapter_14_line_38: GL tag implication.
  have row_38 : (N v4) := by
    apply row_39
    exact row_37
  -- chapter_14_line_16: GL tag disintegration.
  have row_16 : (gl_implication13 N N N add) := by
    exact row_17.1.2
  -- chapter_14_line_15: GL tag expansion.
  have row_15 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication13] using row_16
  -- chapter_14_line_14: GL tag implication.
  have row_14 : (gl_existence1 N v6 v1 add) := by
    apply row_15
    exact row_26
    exact row_20
  -- chapter_14_line_13: GL tag expansion.
  have row_13 : (¬ (∀ (v5 : α), ((N v5) → (¬ (add v6 v1 v5))))) := by
    simpa only [gl_existence1] using row_14
  have exists_row_13 : ∃ (v5 : α), ((N v5) ∧ (add v6 v1 v5)) := existsAndOfNotForallImpNot row_13
  obtain ⟨v5, witness_row_13⟩ := exists_row_13
  -- chapter_14_line_12: GL tag disintegration.
  have row_12 : (add v6 v1 v5) := by
    exact witness_row_13.2
  -- chapter_14_line_6: GL tag implication.
  have row_6 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_7
    exact row_9
    exact row_11
  have rule_row_36 := external_peano_externals_36_006 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_14_line_36: GL tag implication.
  have row_36 : (add v6 v2 v4) := by
    apply rule_row_36
    exact row_37
  -- chapter_14_line_33: GL tag implication.
  have row_33 : (succ v5 v4) := by
    apply row_34
    exact row_20
    exact row_25
    exact row_12
    exact row_36
  -- chapter_14_line_29: GL tag implication.
  have row_29 : (v5 = v3) := by
    apply row_30
    exact row_38
    exact row_33
    exact row_32
  have rule_row_5 := external_peano_externals_36_006 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_14_line_5: GL tag implication.
  have row_5 : (add v1 v6 v5) := by
    apply rule_row_5
    exact row_12
  -- chapter_14_line_3: GL tag theorem.
  have row_3 := fta_source_002 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_14_line_2: GL tag implication.
  have row_2 : (gl_preorder N add v1 v5) := by
    apply row_3
    exact row_5
  -- chapter_14_line_1: GL tag equality1.
  have row_1 : (gl_preorder N add v1 v3) := by
    have equality_source := row_2
    have equality_step_1 := row_29
    cases equality_step_1
    exact equality_source
  exact row_1

theorem fta_source_015
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_006 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (x_4 x_8 x_7 x_9))))))
    (external_gauss_externals_24_018 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α) (x_7 : α) (x_8 : GLBinaryRelation α), ((gl_AnchorGauss x_1 x_2 x_3 x_4 x_5 x_6 x_7 x_8) → (∀ (x_9 : α) (x_10 : α), ((x_3 x_9 x_10) → (gl_preorder x_1 x_4 x_9 x_10))))))
    (external_gauss_externals_24_021 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α) (x_7 : α) (x_8 : GLBinaryRelation α), ((gl_AnchorGauss x_1 x_2 x_3 x_4 x_5 x_6 x_7 x_8) → (∀ (x_9 : α) (x_10 : α), ((gl_preorder x_1 x_4 x_9 x_10) → (∀ (x_11 : α), ((gl_preorder x_1 x_4 x_10 x_11) → (gl_preorder x_1 x_4 x_9 x_11))))))))
    : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (v3 : α), ((gl_preorder N add v2 v3) → (gl_preorder N add v1 v3))))) := by
  intro v1
  intro v2
  intro premise_1
  intro v3
  intro premise_2
  -- chapter_17_line_33: GL tag task formulation.
  have row_33 : (gl_preorder N add v2 v3) := by
    exact premise_2
  -- chapter_17_line_32: GL tag expansion.
  have row_32 : (¬ (∀ (v5 : α), ((N v5) → (¬ (add v2 v5 v3))))) := by
    simpa only [gl_preorder] using row_33
  have exists_row_32 : ∃ (v5 : α), ((N v5) ∧ (add v2 v5 v3)) := existsAndOfNotForallImpNot row_32
  obtain ⟨v5, witness_row_32⟩ := exists_row_32
  -- chapter_17_line_39: GL tag disintegration.
  have row_39 : (add v2 v5 v3) := by
    exact witness_row_32.2
  -- chapter_17_line_31: GL tag disintegration.
  have row_31 : (N v5) := by
    exact witness_row_32.1
  -- chapter_17_line_30: GL tag task formulation.
  have row_30 : (succ v1 v2) := by
    exact premise_1
  -- chapter_17_line_16: GL tag expansion for integration.
  have row_16 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_17_line_15: GL tag reformulation for integration and.
  have row_15 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_17_line_12: GL tag theorem.
  have row_12 := fta_source_002 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_17_line_7: GL tag task formulation.
  have row_7 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_17_line_6: GL tag expansion.
  have row_6 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_7
  -- chapter_17_line_10: GL tag disintegration.
  have row_10 : (succ one two) := by
    exact row_6.1.2
  -- chapter_17_line_9: GL tag disintegration.
  have row_9 : (succ zero one) := by
    exact row_6.1.1.2
  -- chapter_17_line_8: GL tag disintegration.
  have row_8 : (gl_identity N identity) := by
    exact row_6.2
  -- chapter_17_line_5: GL tag disintegration.
  have row_5 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_6.1.1.1
  -- chapter_17_line_24: GL tag expansion.
  have row_24 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_5
  -- chapter_17_line_37: GL tag disintegration.
  have row_37 : (gl_implication17 N succ add) := by
    exact row_24.1.1.1.1.1.2
  -- chapter_17_line_36: GL tag expansion.
  have row_36 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication17] using row_37
  -- chapter_17_line_29: GL tag disintegration.
  have row_29 : (gl_fXY succ N N) := by
    exact row_24.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_17_line_28: GL tag expansion.
  have row_28 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_29
  -- chapter_17_line_27: GL tag disintegration.
  have row_27 : (gl_implication0 succ N) := by
    exact row_28.1.1.1
  -- chapter_17_line_26: GL tag expansion.
  have row_26 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_27
  -- chapter_17_line_25: GL tag implication.
  have row_25 : (N v1) := by
    apply row_26
    exact row_30
  -- chapter_17_line_23: GL tag disintegration.
  have row_23 : (gl_fXYZ add N N N) := by
    exact row_24.1.1.1.1.1.1.1.1.2
  -- chapter_17_line_22: GL tag expansion.
  have row_22 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_23
  -- chapter_17_line_21: GL tag disintegration.
  have row_21 : (gl_implication13 N N N add) := by
    exact row_22.1.2
  -- chapter_17_line_20: GL tag expansion.
  have row_20 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication13] using row_21
  -- chapter_17_line_19: GL tag implication.
  have row_19 : (gl_existence1 N v5 v1 add) := by
    apply row_20
    exact row_31
    exact row_25
  -- chapter_17_line_18: GL tag expansion.
  have row_18 : (¬ (∀ (v4 : α), ((N v4) → (¬ (add v5 v1 v4))))) := by
    simpa only [gl_existence1] using row_19
  have exists_row_18 : ∃ (v4 : α), ((N v4) ∧ (add v5 v1 v4)) := existsAndOfNotForallImpNot row_18
  obtain ⟨v4, witness_row_18⟩ := exists_row_18
  -- chapter_17_line_17: GL tag disintegration.
  have row_17 : (add v5 v1 v4) := by
    exact witness_row_18.2
  -- chapter_17_line_14: GL tag implication.
  have row_14 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_15
    exact row_5
    exact row_9
  have rule_row_38 := external_peano_externals_36_006 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_17_line_38: GL tag implication.
  have row_38 : (add v5 v2 v3) := by
    apply rule_row_38
    exact row_39
  -- chapter_17_line_35: GL tag implication.
  have row_35 : (succ v4 v3) := by
    apply row_36
    exact row_25
    exact row_30
    exact row_17
    exact row_38
  have rule_row_13 := external_peano_externals_36_006 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_17_line_13: GL tag implication.
  have row_13 : (add v1 v5 v4) := by
    apply rule_row_13
    exact row_17
  -- chapter_17_line_11: GL tag implication.
  have row_11 : (gl_preorder N add v1 v4) := by
    apply row_12
    exact row_13
  -- chapter_17_line_4: GL tag expansion for integration.
  have row_4 : ((gl_AnchorGauss N zero succ add mul one two identity) ↔ ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity))) := by
    exact Iff.rfl
  -- chapter_17_line_3: GL tag reformulation for integration and.
  have row_3 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → ((succ one two) → ((gl_identity N identity) → (gl_AnchorGauss N zero succ add mul one two identity))))) := by
    intro integration_premise_1
    intro integration_premise_2
    intro integration_premise_3
    intro integration_premise_4
    simp only [gl_AnchorGauss]
    exact ⟨⟨⟨integration_premise_1, integration_premise_2⟩, integration_premise_3⟩, integration_premise_4⟩
  -- chapter_17_line_2: GL tag implication.
  have row_2 : (gl_AnchorGauss N zero succ add mul one two identity) := by
    apply row_3
    exact row_5
    exact row_9
    exact row_10
    exact row_8
  have rule_row_34 := external_gauss_externals_24_018 N zero succ add mul one two identity (anchorGaussOfFTA N zero succ add mul one two identity anchor)
  -- chapter_17_line_34: GL tag implication.
  have row_34 : (gl_preorder N add v4 v3) := by
    apply rule_row_34
    exact row_35
  have rule_row_1 := external_gauss_externals_24_021 N zero succ add mul one two identity (anchorGaussOfFTA N zero succ add mul one two identity anchor)
  -- chapter_17_line_1: GL tag implication.
  have row_1 : (gl_preorder N add v1 v3) := by
    apply rule_row_1
    exact row_11
    exact row_34
  exact row_1

theorem fta_source_017
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_034 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (∀ (w1 : α), ((N w1) → (gl_or2 w1 zero N succ))))))
    : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (v3 : α), ((gl_strictOrder N add v3 v2) → (gl_preorder N add v3 v1))))) := by
  intro v1
  intro v2
  intro premise_1
  intro v3
  intro premise_2
  -- chapter_19_line_28: GL tag theorem.
  have row_28 := fta_source_047 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_19_line_25: GL tag task formulation.
  have row_25 : (gl_strictOrder N add v3 v2) := by
    exact premise_2
  -- chapter_19_line_24: GL tag expansion.
  have row_24 : ((gl_preorder N add v3 v2) ∧ (¬ (v3 = v2))) := by
    simpa only [gl_strictOrder] using row_25
  -- chapter_19_line_23: GL tag disintegration.
  have row_23 : (gl_preorder N add v3 v2) := by
    exact row_24.1
  -- chapter_19_line_22: GL tag expansion.
  have row_22 : (¬ (∀ (v7 : α), ((N v7) → (¬ (add v3 v7 v2))))) := by
    simpa only [gl_preorder] using row_23
  have exists_row_22 : ∃ (v7 : α), ((N v7) ∧ (add v3 v7 v2)) := existsAndOfNotForallImpNot row_22
  obtain ⟨v7, witness_row_22⟩ := exists_row_22
  -- chapter_19_line_29: GL tag disintegration.
  have row_29 : (add v3 v7 v2) := by
    exact witness_row_22.2
  -- chapter_19_line_21: GL tag disintegration.
  have row_21 : (N v7) := by
    exact witness_row_22.1
  -- chapter_19_line_17: GL tag expansion for integration.
  have row_17 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_19_line_16: GL tag reformulation for integration and.
  have row_16 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_19_line_7: GL tag task formulation.
  have row_7 : (succ v1 v2) := by
    exact premise_1
  -- chapter_19_line_6: GL tag task formulation.
  have row_6 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_19_line_27: GL tag implication.
  have row_27 : (¬ (zero = v7)) := by
    apply row_28
    exact row_25
    exact row_29
  -- chapter_19_line_26: GL tag symmetry of inequality.
  have row_26 : (¬ (v7 = zero)) := by
    exact fun equality => row_27 (Eq.symm equality)
  -- chapter_19_line_19: GL tag expansion.
  have row_19 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_6
  -- chapter_19_line_20: GL tag disintegration.
  have row_20 : (succ zero one) := by
    exact row_19.1.1.2
  -- chapter_19_line_18: GL tag disintegration.
  have row_18 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_19.1.1.1
  -- chapter_19_line_15: GL tag implication.
  have row_15 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_16
    exact row_18
    exact row_20
  have rule_row_14 := external_peano_externals_36_034 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_19_line_14: GL tag implication.
  have row_14 : (gl_or2 v7 zero N succ) := by
    apply rule_row_14
    exact row_21
  -- chapter_19_line_13: GL tag expansion.
  have row_13 : (¬ ((¬ (v7 = zero)) ∧ (¬ (gl_existence11 N v7 succ)))) := by
    simpa only [gl_or2, GLExport.orIffNotAndNot] using row_14
  -- chapter_19_line_12: GL tag disintegration.
  have row_12 : (gl_implication74 v7 zero N succ) := by
    simp only [gl_implication74]
    classical
    intro projection_premise
    apply Classical.byContradiction
    intro projection_counterexample
    exact row_13 ⟨projection_premise, projection_counterexample⟩
  -- chapter_19_line_11: GL tag expansion.
  have row_11 : ((¬ (v7 = zero)) → (gl_existence11 N v7 succ)) := by
    simpa only [gl_implication74] using row_12
  -- chapter_19_line_10: GL tag implication.
  have row_10 : (gl_existence11 N v7 succ) := by
    apply row_11
    exact row_26
  -- chapter_19_line_9: GL tag expansion.
  have row_9 : (¬ (∀ (v5 : α), ((N v5) → (¬ (succ v5 v7))))) := by
    simpa only [gl_existence11] using row_10
  have exists_row_9 : ∃ (v5 : α), ((N v5) ∧ (succ v5 v7)) := existsAndOfNotForallImpNot row_9
  obtain ⟨v5, witness_row_9⟩ := exists_row_9
  -- chapter_19_line_30: GL tag disintegration.
  have row_30 : (N v5) := by
    exact witness_row_9.1
  -- chapter_19_line_8: GL tag disintegration.
  have row_8 : (succ v5 v7) := by
    exact witness_row_9.2
  -- chapter_19_line_5: GL tag theorem.
  have row_5 := fta_source_013 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_19_line_4: GL tag implication.
  have row_4 : (add v3 v5 v1) := by
    apply row_5
    exact row_7
    exact row_29
    exact row_8
  -- chapter_19_line_3: GL tag expansion for integration.
  have row_3 : ((gl_preorder N add v3 v1) ↔ (¬ (∀ (v6 : α), ((N v6) → (¬ (add v3 v6 v1)))))) := by
    exact Iff.rfl
  -- chapter_19_line_2: GL tag reformulation for integration >[bound].
  have row_2 : (∀ (v4 : α), ((N v4) → ((add v3 v4 v1) → (gl_preorder N add v3 v1)))) := by
    intro v4
    intro integration_premise_1
    intro integration_premise_2
    apply (row_3).2
    intro universal_counterexample
    exact universal_counterexample v4 integration_premise_1 integration_premise_2
  -- chapter_19_line_1: GL tag implication.
  have row_1 : (gl_preorder N add v3 v1) := by
    apply row_2
    exact row_30
    exact row_4
  exact row_1

theorem fta_source_018
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_006 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (x_4 x_8 x_7 x_9))))))
    (external_peano_externals_36_034 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (∀ (w1 : α), ((N w1) → (gl_or2 w1 zero N succ))))))
    (external_peano_externals_36_003 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (∀ (x_10 : α), ((x_3 x_8 x_10) → (∀ (x_11 : α), ((x_3 x_11 x_7) → (x_4 x_10 x_11 x_9))))))))))
    (external_peano_externals_36_029 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((gl_preorder x_1 x_4 x_7 x_8) → ((gl_preorder x_1 x_4 x_8 x_7) → (x_7 = x_8)))))))
    : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (v3 : α), ((gl_strictOrder N add v1 v3) → (¬ (gl_strictOrder N add v3 v2)))))) := by
  intro v1
  intro v2
  intro premise_1
  intro v3
  intro premise_2
  -- chapter_20_line_38: GL tag task formulation.
  have row_38 : (succ v1 v2) := by
    exact premise_1
  -- chapter_20_line_35: GL tag theorem.
  have row_35 := fta_source_047 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_20_line_32: GL tag task formulation.
  have row_32 : (gl_strictOrder N add v1 v3) := by
    exact premise_2
  -- chapter_20_line_31: GL tag expansion.
  have row_31 : ((gl_preorder N add v1 v3) ∧ (¬ (v1 = v3))) := by
    simpa only [gl_strictOrder] using row_32
  -- chapter_20_line_30: GL tag disintegration.
  have row_30 : (gl_preorder N add v1 v3) := by
    exact row_31.1
  -- chapter_20_line_29: GL tag expansion.
  have row_29 : (¬ (∀ (v6 : α), ((N v6) → (¬ (add v1 v6 v3))))) := by
    simpa only [gl_preorder] using row_30
  have exists_row_29 : ∃ (v6 : α), ((N v6) ∧ (add v1 v6 v3)) := existsAndOfNotForallImpNot row_29
  obtain ⟨v6, witness_row_29⟩ := exists_row_29
  -- chapter_20_line_36: GL tag disintegration.
  have row_36 : (add v1 v6 v3) := by
    exact witness_row_29.2
  -- chapter_20_line_28: GL tag disintegration.
  have row_28 : (N v6) := by
    exact witness_row_29.1
  -- chapter_20_line_24: GL tag expansion for integration.
  have row_24 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_20_line_23: GL tag reformulation for integration and.
  have row_23 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_20_line_11: GL tag task formulation.
  have row_11 : ((gl_strictOrder N add v3 v2) → (gl_strictOrder N add v3 v2)) := by
    intro scope_premise_1
    exact scope_premise_1
  -- chapter_20_line_14: GL tag expansion.
  have row_14 : ((gl_strictOrder N add v3 v2) → ((gl_preorder N add v3 v2) ∧ (¬ (v3 = v2)))) := by
    simpa only [gl_strictOrder] using row_11
  -- chapter_20_line_13: GL tag disintegration.
  have row_13 : ((gl_strictOrder N add v3 v2) → (¬ (v3 = v2))) := by
    intro scope_premise_1
    have scoped_fact_1 := row_14 scope_premise_1
    exact scoped_fact_1.2
  -- chapter_20_line_12: GL tag symmetry of inequality.
  have row_12 : ((gl_strictOrder N add v3 v2) → (¬ (v2 = v3))) := by
    intro scope_premise_1
    have scoped_fact_1 := row_13 scope_premise_1
    exact fun equality => scoped_fact_1 (Eq.symm equality)
  -- chapter_20_line_10: GL tag task formulation.
  have row_10 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_20_line_34: GL tag implication.
  have row_34 : (¬ (zero = v6)) := by
    apply row_35
    exact row_32
    exact row_36
  -- chapter_20_line_33: GL tag symmetry of inequality.
  have row_33 : (¬ (v6 = zero)) := by
    exact fun equality => row_34 (Eq.symm equality)
  -- chapter_20_line_26: GL tag expansion.
  have row_26 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_10
  -- chapter_20_line_27: GL tag disintegration.
  have row_27 : (succ zero one) := by
    exact row_26.1.1.2
  -- chapter_20_line_25: GL tag disintegration.
  have row_25 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_26.1.1.1
  -- chapter_20_line_22: GL tag implication.
  have row_22 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_23
    exact row_25
    exact row_27
  have rule_row_40 := external_peano_externals_36_006 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_20_line_40: GL tag implication.
  have row_40 : (add v6 v1 v3) := by
    apply rule_row_40
    exact row_36
  have rule_row_21 := external_peano_externals_36_034 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_20_line_21: GL tag implication.
  have row_21 : (gl_or2 v6 zero N succ) := by
    apply rule_row_21
    exact row_28
  -- chapter_20_line_20: GL tag expansion.
  have row_20 : (¬ ((¬ (v6 = zero)) ∧ (¬ (gl_existence11 N v6 succ)))) := by
    simpa only [gl_or2, GLExport.orIffNotAndNot] using row_21
  -- chapter_20_line_19: GL tag disintegration.
  have row_19 : (gl_implication74 v6 zero N succ) := by
    simp only [gl_implication74]
    classical
    intro projection_premise
    apply Classical.byContradiction
    intro projection_counterexample
    exact row_20 ⟨projection_premise, projection_counterexample⟩
  -- chapter_20_line_18: GL tag expansion.
  have row_18 : ((¬ (v6 = zero)) → (gl_existence11 N v6 succ)) := by
    simpa only [gl_implication74] using row_19
  -- chapter_20_line_17: GL tag implication.
  have row_17 : (gl_existence11 N v6 succ) := by
    apply row_18
    exact row_33
  -- chapter_20_line_16: GL tag expansion.
  have row_16 : (¬ (∀ (v4 : α), ((N v4) → (¬ (succ v4 v6))))) := by
    simpa only [gl_existence11] using row_17
  have exists_row_16 : ∃ (v4 : α), ((N v4) ∧ (succ v4 v6)) := existsAndOfNotForallImpNot row_16
  obtain ⟨v4, witness_row_16⟩ := exists_row_16
  -- chapter_20_line_39: GL tag disintegration.
  have row_39 : (succ v4 v6) := by
    exact witness_row_16.2
  have rule_row_37 := external_peano_externals_36_003 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_20_line_37: GL tag implication.
  have row_37 : (add v2 v4 v3) := by
    apply rule_row_37
    exact row_40
    exact row_38
    exact row_39
  -- chapter_20_line_15: GL tag disintegration.
  have row_15 : (N v4) := by
    exact witness_row_16.1
  -- chapter_20_line_9: GL tag theorem.
  have row_9 := fta_source_046 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_029
  -- chapter_20_line_8: GL tag implication.
  have row_8 : ((gl_strictOrder N add v3 v2) → (¬ (gl_strictOrder N add v2 v3))) := by
    intro scope_premise_1
    have scoped_fact_2 := row_11 scope_premise_1
    apply row_9
    exact scoped_fact_2
  -- chapter_20_line_7: GL tag expansion.
  have row_7 : ((gl_strictOrder N add v3 v2) → (¬ ((gl_preorder N add v2 v3) ∧ (¬ (v2 = v3))))) := by
    simpa only [gl_strictOrder] using row_8
  -- chapter_20_line_6: GL tag disintegration.
  have row_6 : ((gl_strictOrder N add v3 v2) → ((¬ (v2 = v3)) → (¬ (gl_preorder N add v2 v3)))) := by
    intro scope_premise_1
    have scoped_fact_1 := row_7 scope_premise_1
    classical
    intro projection_premise
    intro projection_counterexample
    exact scoped_fact_1 ⟨projection_counterexample, projection_premise⟩
  -- chapter_20_line_5: GL tag implication.
  have row_5 : ((gl_strictOrder N add v3 v2) → (¬ (gl_preorder N add v2 v3))) := by
    intro scope_premise_1
    have scoped_fact_1 := row_6 scope_premise_1
    have scoped_fact_2 := row_12 scope_premise_1
    apply scoped_fact_1
    exact scoped_fact_2
  -- chapter_20_line_4: GL tag expansion.
  have row_4 : ((gl_strictOrder N add v3 v2) → (gl_implication24 N v2 v3 add)) := by
    intro scope_premise_1
    have scoped_fact_1 := row_5 scope_premise_1
    simp only [gl_implication24]
    intro existence_witness_1 compact_premise compact_negated
    have unfolded_row_5 := scoped_fact_1
    simp only [gl_preorder] at unfolded_row_5
    apply unfolded_row_5
    intro positive_row_5
    exact positive_row_5 existence_witness_1 compact_premise compact_negated
  -- chapter_20_line_3: GL tag expansion.
  have row_3 : ((gl_strictOrder N add v3 v2) → (∀ (w1 : α), ((N w1) → (¬ (add v2 w1 v3))))) := by
    simpa only [gl_implication24] using row_4
  -- chapter_20_line_2: GL tag implication.
  have row_2 : ((gl_strictOrder N add v3 v2) → (¬ (add v2 v4 v3))) := by
    intro scope_premise_1
    have scoped_fact_1 := row_3 scope_premise_1
    apply scoped_fact_1
    exact row_15
  -- chapter_20_line_1: GL tag contradiction.
  have row_1 : (¬ (gl_strictOrder N add v3 v2)) := by
    intro contradiction_assumption
    have scoped_contradiction := row_2 contradiction_assumption
    exact scoped_contradiction row_37
  exact row_1

private theorem fta_source_022_induction_typing
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (v1 : α)
    (v2 : α)
    (assumption_10 : (mul v1 v2 zero))
    : (N v2) := by
  -- chapter_24_line_10: GL tag task formulation.
  have row_10 : (mul v1 v2 zero) := by
    exact assumption_10
  -- chapter_24_line_9: GL tag task formulation.
  have row_9 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_24_line_8: GL tag expansion.
  have row_8 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_9
  -- chapter_24_line_7: GL tag disintegration.
  have row_7 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_8.1.1.1
  -- chapter_24_line_6: GL tag expansion.
  have row_6 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_7
  -- chapter_24_line_5: GL tag disintegration.
  have row_5 : (gl_fXYZ mul N N N) := by
    exact row_6.1.1.1.2
  -- chapter_24_line_4: GL tag expansion.
  have row_4 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_5
  -- chapter_24_line_3: GL tag disintegration.
  have row_3 : (gl_implication9 mul N) := by
    exact row_4.1.1.1.2
  -- chapter_24_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication9] using row_3
  -- chapter_24_line_1: GL tag implication.
  have row_1 : (N v2) := by
    apply row_2
    exact row_10
  exact row_1

private theorem fta_source_022_check_zero
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (v1 : α)
    (v2 : α)
    (assumption_4 : (¬ (zero = v2)))
    (assumption_2 : (v2 = zero))
    : (zero = v1) := by
  -- chapter_25_line_4: GL tag task formulation.
  have row_4 : (¬ (zero = v2)) := by
    exact assumption_4
  -- chapter_25_line_3: GL tag symmetry of inequality.
  have row_3 : (¬ (v2 = zero)) := by
    exact fun equality => row_4 (Eq.symm equality)
  -- chapter_25_line_2: GL tag recursion.
  have row_2 : (v2 = zero) := by
    exact assumption_2
  -- chapter_25_line_1: GL tag vacuous truth.
  have row_1 : (zero = v1) := by
    exact False.elim (row_3 row_2)
  exact row_1

private theorem fta_source_022_check_induction_condition
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_035 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (gl_existence11 N one succ))))
    (external_peano_externals_36_024 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_8 x_2) → ((x_1 x_8) → (x_2 = x_7)))))))
    (external_peano_externals_36_025 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_8 x_2) → (x_4 x_8 x_7 x_2))))))
    (previous : α)
    (v1 : α)
    (v2 : α)
    (assumption_20 : (N v1))
    (assumption_12 : (mul v1 v2 zero))
    (assumption_11 : (succ previous v2))
    : (zero = v1) := by
  -- chapter_26_line_39: GL tag expansion for integration.
  have row_39 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_26_line_38: GL tag reformulation for integration and.
  have row_38 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_26_line_29: GL tag theorem.
  have row_29 := fta_source_020 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_024
  -- chapter_26_line_20: GL tag task formulation.
  have row_20 : (N v1) := by
    exact assumption_20
  -- chapter_26_line_12: GL tag task formulation.
  have row_12 : (mul v1 v2 zero) := by
    exact assumption_12
  -- chapter_26_line_11: GL tag recursion.
  have row_11 : (succ previous v2) := by
    exact assumption_11
  -- chapter_26_line_3: GL tag task formulation.
  have row_3 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_26_line_10: GL tag expansion.
  have row_10 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_3
  -- chapter_26_line_41: GL tag disintegration.
  have row_41 : (succ one two) := by
    exact row_10.1.2
  -- chapter_26_line_33: GL tag disintegration.
  have row_33 : (succ zero one) := by
    exact row_10.1.1.2
  -- chapter_26_line_9: GL tag disintegration.
  have row_9 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_10.1.1.1
  -- chapter_26_line_37: GL tag implication.
  have row_37 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_38
    exact row_9
    exact row_33
  have rule_row_36 := external_peano_externals_36_035 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_26_line_36: GL tag implication.
  have row_36 : (gl_existence11 N one succ) := by
    apply rule_row_36
  -- chapter_26_line_35: GL tag expansion.
  have row_35 : (¬ (∀ (v3 : α), ((N v3) → (¬ (succ v3 one))))) := by
    simpa only [gl_existence11] using row_36
  have exists_row_35 : ∃ (v3 : α), ((N v3) ∧ (succ v3 one)) := existsAndOfNotForallImpNot row_35
  obtain ⟨v3, witness_row_35⟩ := exists_row_35
  -- chapter_26_line_42: GL tag disintegration.
  have row_42 : (N v3) := by
    exact witness_row_35.1
  -- chapter_26_line_34: GL tag disintegration.
  have row_34 : (succ v3 one) := by
    exact witness_row_35.2
  -- chapter_26_line_8: GL tag expansion.
  have row_8 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_9
  -- chapter_26_line_32: GL tag disintegration.
  have row_32 : (gl_implication7 N succ) := by
    exact row_8.1.1.1.1.1.1.1.1.1.2
  -- chapter_26_line_31: GL tag expansion.
  have row_31 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w2 w1) → (∀ (w3 : α), ((succ w3 w1) → (w2 = w3))))))) := by
    simpa only [gl_implication7] using row_32
  -- chapter_26_line_25: GL tag disintegration.
  have row_25 : (gl_fXY succ N N) := by
    exact row_8.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_26_line_24: GL tag expansion.
  have row_24 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_25
  -- chapter_26_line_23: GL tag disintegration.
  have row_23 : (gl_implication0 succ N) := by
    exact row_24.1.1.1
  -- chapter_26_line_22: GL tag expansion.
  have row_22 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_23
  -- chapter_26_line_40: GL tag implication.
  have row_40 : (N one) := by
    apply row_22
    exact row_41
  -- chapter_26_line_30: GL tag implication.
  have row_30 : (zero = v3) := by
    apply row_31
    exact row_40
    exact row_33
    exact row_34
  -- chapter_26_line_21: GL tag implication.
  have row_21 : (N previous) := by
    apply row_22
    exact row_11
  -- chapter_26_line_19: GL tag disintegration.
  have row_19 : (gl_fXYZ mul N N N) := by
    exact row_8.1.1.1.2
  -- chapter_26_line_18: GL tag expansion.
  have row_18 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_19
  -- chapter_26_line_17: GL tag disintegration.
  have row_17 : (gl_implication13 N N N mul) := by
    exact row_18.1.2
  -- chapter_26_line_16: GL tag expansion.
  have row_16 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication13] using row_17
  -- chapter_26_line_15: GL tag implication.
  have row_15 : (gl_existence1 N v1 previous mul) := by
    apply row_16
    exact row_20
    exact row_21
  -- chapter_26_line_14: GL tag expansion.
  have row_14 : (¬ (∀ (v4 : α), ((N v4) → (¬ (mul v1 previous v4))))) := by
    simpa only [gl_existence1] using row_15
  have exists_row_14 : ∃ (v4 : α), ((N v4) ∧ (mul v1 previous v4)) := existsAndOfNotForallImpNot row_14
  obtain ⟨v4, witness_row_14⟩ := exists_row_14
  -- chapter_26_line_13: GL tag disintegration.
  have row_13 : (mul v1 previous v4) := by
    exact witness_row_14.2
  -- chapter_26_line_7: GL tag disintegration.
  have row_7 : (gl_implication21 N succ mul add) := by
    exact row_8.2
  -- chapter_26_line_6: GL tag expansion.
  have row_6 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((mul w3 w1 w4) → (∀ (w5 : α), ((mul w3 w2 w5) → (add w4 w3 w5))))))))) := by
    simpa only [gl_implication21] using row_7
  -- chapter_26_line_5: GL tag implication.
  have row_5 : (add v4 v1 zero) := by
    apply row_6
    exact row_21
    exact row_11
    exact row_13
    exact row_12
  -- chapter_26_line_28: GL tag implication.
  have row_28 : (zero = v4) := by
    apply row_29
    exact row_5
    exact row_20
  -- chapter_26_line_27: GL tag symmetry of equality.
  have row_27 : (v4 = zero) := by
    exact Eq.symm row_28
  -- chapter_26_line_26: GL tag equality2.
  have row_26 : (v4 = v3) := by
    exact Eq.trans row_27 row_30
  -- chapter_26_line_4: GL tag equality1.
  have row_4 : (add v3 v1 zero) := by
    have equality_source := row_5
    have equality_step_1 := row_26
    cases equality_step_1
    exact equality_source
  -- chapter_26_line_2: GL tag theorem.
  have row_2 := fta_source_021 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_025 external_peano_externals_36_024
  -- chapter_26_line_1: GL tag implication.
  have row_1 : (zero = v1) := by
    apply row_2
    exact row_4
    exact row_42
  exact row_1

theorem fta_source_022
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_035 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (gl_existence11 N one succ))))
    (external_peano_externals_36_024 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_8 x_2) → ((x_1 x_8) → (x_2 = x_7)))))))
    (external_peano_externals_36_025 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_8 x_2) → (x_4 x_8 x_7 x_2))))))
    : (∀ (v1 : α) (v2 : α), ((mul v1 v2 zero) → ((N v1) → ((¬ (zero = v2)) → (zero = v1))))) := by
  intro v1
  intro v2
  intro premise_1
  intro premise_2
  intro premise_3
  have inductionMember : N v2 := by
    have typingRule := fta_source_022_induction_typing N zero succ add mul one two identity anchor relationalInduction
    exact typingRule v1 v2 premise_1
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorFTA, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v1 : α), ((mul v1 zero zero) → ((N v1) → ((¬ (zero = zero)) → (zero = v1))))) := by
    intro v1
    intro base_premise_1
    intro base_premise_2
    intro base_premise_3
    have zeroRule := fta_source_022_check_zero N zero succ add mul one two identity anchor relationalInduction
    exact zeroRule v1 zero base_premise_3 rfl
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v1 : α), ((mul v1 induction_n zero) → ((N v1) → ((¬ (zero = induction_n)) → (zero = v1))))) → ∀ induction_m, succ induction_n induction_m → (∀ (v1 : α), ((mul v1 induction_m zero) → ((N v1) → ((¬ (zero = induction_m)) → (zero = v1))))) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro v1
    intro step_premise_1
    intro step_premise_2
    intro step_premise_3
    have step_induction_assumption_1 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have stepRule := fta_source_022_check_induction_condition N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_035 external_peano_externals_36_024 external_peano_externals_36_025
    exact stepRule induction_n v1 induction_m step_premise_2 step_premise_1 step_induction_assumption_1
  have inductionProperty : (∀ (v1 : α), ((mul v1 v2 zero) → ((N v1) → ((¬ (zero = v2)) → (zero = v1))))) := by
    exact relationalInduction N zero succ add mul one two identity anchor
      (fun induction_value => (∀ (v1 : α), ((mul v1 induction_value zero) → ((N v1) → ((¬ (zero = induction_value)) → (zero = v1))))))
      v2
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v1 premise_1 premise_2 premise_3

private theorem fta_source_023_induction_typing
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (v1 : α)
    (v2 : α)
    (assumption_10 : (mul v1 v2 zero))
    : (N v2) := by
  -- chapter_27_line_10: GL tag task formulation.
  have row_10 : (mul v1 v2 zero) := by
    exact assumption_10
  -- chapter_27_line_9: GL tag task formulation.
  have row_9 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_27_line_8: GL tag expansion.
  have row_8 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_9
  -- chapter_27_line_7: GL tag disintegration.
  have row_7 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_8.1.1.1
  -- chapter_27_line_6: GL tag expansion.
  have row_6 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_7
  -- chapter_27_line_5: GL tag disintegration.
  have row_5 : (gl_fXYZ mul N N N) := by
    exact row_6.1.1.1.2
  -- chapter_27_line_4: GL tag expansion.
  have row_4 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_5
  -- chapter_27_line_3: GL tag disintegration.
  have row_3 : (gl_implication9 mul N) := by
    exact row_4.1.1.1.2
  -- chapter_27_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication9] using row_3
  -- chapter_27_line_1: GL tag implication.
  have row_1 : (N v2) := by
    apply row_2
    exact row_10
  exact row_1

private theorem fta_source_023_check_zero
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (v2 : α)
    (assumption_2 : (v2 = zero))
    : (zero = v2) := by
  -- chapter_28_line_2: GL tag recursion.
  have row_2 : (v2 = zero) := by
    exact assumption_2
  -- chapter_28_line_1: GL tag symmetry of equality.
  have row_1 : (zero = v2) := by
    exact Eq.symm row_2
  exact row_1

private theorem fta_source_023_check_induction_condition
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_035 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (gl_existence11 N one succ))))
    (external_peano_externals_36_024 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_8 x_2) → ((x_1 x_8) → (x_2 = x_7)))))))
    (external_peano_externals_36_025 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_8 x_2) → (x_4 x_8 x_7 x_2))))))
    (previous : α)
    (v1 : α)
    (v2 : α)
    (assumption_22 : (N v1))
    (assumption_14 : (mul v1 v2 zero))
    (assumption_13 : (succ previous v2))
    (assumption_2 : (¬ (zero = v1)))
    : (zero = v2) := by
  -- chapter_29_line_41: GL tag expansion for integration.
  have row_41 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_29_line_40: GL tag reformulation for integration and.
  have row_40 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_29_line_31: GL tag theorem.
  have row_31 := fta_source_020 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_024
  -- chapter_29_line_22: GL tag task formulation.
  have row_22 : (N v1) := by
    exact assumption_22
  -- chapter_29_line_14: GL tag task formulation.
  have row_14 : (mul v1 v2 zero) := by
    exact assumption_14
  -- chapter_29_line_13: GL tag recursion.
  have row_13 : (succ previous v2) := by
    exact assumption_13
  -- chapter_29_line_5: GL tag task formulation.
  have row_5 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_29_line_12: GL tag expansion.
  have row_12 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_5
  -- chapter_29_line_43: GL tag disintegration.
  have row_43 : (succ one two) := by
    exact row_12.1.2
  -- chapter_29_line_35: GL tag disintegration.
  have row_35 : (succ zero one) := by
    exact row_12.1.1.2
  -- chapter_29_line_11: GL tag disintegration.
  have row_11 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_12.1.1.1
  -- chapter_29_line_39: GL tag implication.
  have row_39 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_40
    exact row_11
    exact row_35
  have rule_row_38 := external_peano_externals_36_035 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_29_line_38: GL tag implication.
  have row_38 : (gl_existence11 N one succ) := by
    apply rule_row_38
  -- chapter_29_line_37: GL tag expansion.
  have row_37 : (¬ (∀ (v3 : α), ((N v3) → (¬ (succ v3 one))))) := by
    simpa only [gl_existence11] using row_38
  have exists_row_37 : ∃ (v3 : α), ((N v3) ∧ (succ v3 one)) := existsAndOfNotForallImpNot row_37
  obtain ⟨v3, witness_row_37⟩ := exists_row_37
  -- chapter_29_line_44: GL tag disintegration.
  have row_44 : (N v3) := by
    exact witness_row_37.1
  -- chapter_29_line_36: GL tag disintegration.
  have row_36 : (succ v3 one) := by
    exact witness_row_37.2
  -- chapter_29_line_10: GL tag expansion.
  have row_10 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_11
  -- chapter_29_line_34: GL tag disintegration.
  have row_34 : (gl_implication7 N succ) := by
    exact row_10.1.1.1.1.1.1.1.1.1.2
  -- chapter_29_line_33: GL tag expansion.
  have row_33 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w2 w1) → (∀ (w3 : α), ((succ w3 w1) → (w2 = w3))))))) := by
    simpa only [gl_implication7] using row_34
  -- chapter_29_line_27: GL tag disintegration.
  have row_27 : (gl_fXY succ N N) := by
    exact row_10.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_29_line_26: GL tag expansion.
  have row_26 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_27
  -- chapter_29_line_25: GL tag disintegration.
  have row_25 : (gl_implication0 succ N) := by
    exact row_26.1.1.1
  -- chapter_29_line_24: GL tag expansion.
  have row_24 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_25
  -- chapter_29_line_42: GL tag implication.
  have row_42 : (N one) := by
    apply row_24
    exact row_43
  -- chapter_29_line_32: GL tag implication.
  have row_32 : (zero = v3) := by
    apply row_33
    exact row_42
    exact row_35
    exact row_36
  -- chapter_29_line_23: GL tag implication.
  have row_23 : (N previous) := by
    apply row_24
    exact row_13
  -- chapter_29_line_21: GL tag disintegration.
  have row_21 : (gl_fXYZ mul N N N) := by
    exact row_10.1.1.1.2
  -- chapter_29_line_20: GL tag expansion.
  have row_20 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_21
  -- chapter_29_line_19: GL tag disintegration.
  have row_19 : (gl_implication13 N N N mul) := by
    exact row_20.1.2
  -- chapter_29_line_18: GL tag expansion.
  have row_18 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication13] using row_19
  -- chapter_29_line_17: GL tag implication.
  have row_17 : (gl_existence1 N v1 previous mul) := by
    apply row_18
    exact row_22
    exact row_23
  -- chapter_29_line_16: GL tag expansion.
  have row_16 : (¬ (∀ (v4 : α), ((N v4) → (¬ (mul v1 previous v4))))) := by
    simpa only [gl_existence1] using row_17
  have exists_row_16 : ∃ (v4 : α), ((N v4) ∧ (mul v1 previous v4)) := existsAndOfNotForallImpNot row_16
  obtain ⟨v4, witness_row_16⟩ := exists_row_16
  -- chapter_29_line_15: GL tag disintegration.
  have row_15 : (mul v1 previous v4) := by
    exact witness_row_16.2
  -- chapter_29_line_9: GL tag disintegration.
  have row_9 : (gl_implication21 N succ mul add) := by
    exact row_10.2
  -- chapter_29_line_8: GL tag expansion.
  have row_8 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((mul w3 w1 w4) → (∀ (w5 : α), ((mul w3 w2 w5) → (add w4 w3 w5))))))))) := by
    simpa only [gl_implication21] using row_9
  -- chapter_29_line_7: GL tag implication.
  have row_7 : (add v4 v1 zero) := by
    apply row_8
    exact row_23
    exact row_13
    exact row_15
    exact row_14
  -- chapter_29_line_30: GL tag implication.
  have row_30 : (zero = v4) := by
    apply row_31
    exact row_7
    exact row_22
  -- chapter_29_line_29: GL tag symmetry of equality.
  have row_29 : (v4 = zero) := by
    exact Eq.symm row_30
  -- chapter_29_line_28: GL tag equality2.
  have row_28 : (v4 = v3) := by
    exact Eq.trans row_29 row_32
  -- chapter_29_line_6: GL tag equality1.
  have row_6 : (add v3 v1 zero) := by
    have equality_source := row_7
    have equality_step_1 := row_28
    cases equality_step_1
    exact equality_source
  -- chapter_29_line_4: GL tag theorem.
  have row_4 := fta_source_021 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_025 external_peano_externals_36_024
  -- chapter_29_line_3: GL tag implication.
  have row_3 : (zero = v1) := by
    apply row_4
    exact row_6
    exact row_44
  -- chapter_29_line_2: GL tag task formulation.
  have row_2 : (¬ (zero = v1)) := by
    exact assumption_2
  -- chapter_29_line_1: GL tag vacuous truth.
  have row_1 : (zero = v2) := by
    exact False.elim (row_2 row_3)
  exact row_1

theorem fta_source_023
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_035 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (gl_existence11 N one succ))))
    (external_peano_externals_36_024 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_8 x_2) → ((x_1 x_8) → (x_2 = x_7)))))))
    (external_peano_externals_36_025 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_8 x_2) → (x_4 x_8 x_7 x_2))))))
    : (∀ (v1 : α) (v2 : α), ((mul v1 v2 zero) → ((N v1) → ((¬ (zero = v1)) → (zero = v2))))) := by
  intro v1
  intro v2
  intro premise_1
  intro premise_2
  intro premise_3
  have inductionMember : N v2 := by
    have typingRule := fta_source_023_induction_typing N zero succ add mul one two identity anchor relationalInduction
    exact typingRule v1 v2 premise_1
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorFTA, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v1 : α), ((mul v1 zero zero) → ((N v1) → ((¬ (zero = v1)) → (zero = zero))))) := by
    intro v1
    intro base_premise_1
    intro base_premise_2
    intro base_premise_3
    have zeroRule := fta_source_023_check_zero N zero succ add mul one two identity anchor relationalInduction
    exact zeroRule zero rfl
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v1 : α), ((mul v1 induction_n zero) → ((N v1) → ((¬ (zero = v1)) → (zero = induction_n))))) → ∀ induction_m, succ induction_n induction_m → (∀ (v1 : α), ((mul v1 induction_m zero) → ((N v1) → ((¬ (zero = v1)) → (zero = induction_m))))) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro v1
    intro step_premise_1
    intro step_premise_2
    intro step_premise_3
    have step_induction_assumption_1 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have stepRule := fta_source_023_check_induction_condition N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_035 external_peano_externals_36_024 external_peano_externals_36_025
    exact stepRule induction_n v1 induction_m step_premise_2 step_premise_1 step_induction_assumption_1 step_premise_3
  have inductionProperty : (∀ (v1 : α), ((mul v1 v2 zero) → ((N v1) → ((¬ (zero = v1)) → (zero = v2))))) := by
    exact relationalInduction N zero succ add mul one two identity anchor
      (fun induction_value => (∀ (v1 : α), ((mul v1 induction_value zero) → ((N v1) → ((¬ (zero = v1)) → (zero = induction_value))))))
      v2
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v1 premise_1 premise_2 premise_3

theorem fta_source_026
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_028 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_5 x_7 x_8 x_6) → (x_5 x_8 x_7 x_6))))))
    : (∀ (v1 : α) (v2 : α), ((mul v1 v2 one) → (¬ (zero = v1)))) := by
  intro v1
  intro v2
  intro premise_1
  -- chapter_32_line_11: GL tag task formulation.
  have row_11 : (mul v1 v2 one) := by
    exact premise_1
  -- chapter_32_line_7: GL tag expansion for integration.
  have row_7 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_32_line_6: GL tag reformulation for integration and.
  have row_6 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_32_line_3: GL tag task formulation.
  have row_3 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_32_line_9: GL tag expansion.
  have row_9 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_3
  -- chapter_32_line_10: GL tag disintegration.
  have row_10 : (succ zero one) := by
    exact row_9.1.1.2
  -- chapter_32_line_8: GL tag disintegration.
  have row_8 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_9.1.1.1
  -- chapter_32_line_5: GL tag implication.
  have row_5 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_6
    exact row_8
    exact row_10
  have rule_row_4 := external_peano_externals_36_028 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_32_line_4: GL tag implication.
  have row_4 : (mul v2 v1 one) := by
    apply rule_row_4
    exact row_11
  -- chapter_32_line_2: GL tag theorem.
  have row_2 := fta_source_025 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_32_line_1: GL tag implication.
  have row_1 : (¬ (zero = v1)) := by
    apply row_2
    exact row_4
  exact row_1

private theorem fta_source_027_induction_typing
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (v2 : α)
    (assumption_1 : (N v2))
    : (N v2) := by
  -- chapter_33_line_1: GL tag task formulation.
  have row_1 : (N v2) := by
    exact assumption_1
  exact row_1

private theorem fta_source_027_check_zero
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (v1 : α)
    (v2 : α)
    (assumption_7 : (mul v1 v2 one))
    (assumption_2 : (v2 = zero))
    : (one = v1) := by
  -- chapter_34_line_7: GL tag task formulation.
  have row_7 : (mul v1 v2 one) := by
    exact assumption_7
  -- chapter_34_line_6: GL tag task formulation.
  have row_6 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_34_line_5: GL tag theorem.
  have row_5 := fta_source_025 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_34_line_4: GL tag implication.
  have row_4 : (¬ (zero = v2)) := by
    apply row_5
    exact row_7
  -- chapter_34_line_3: GL tag symmetry of inequality.
  have row_3 : (¬ (v2 = zero)) := by
    exact fun equality => row_4 (Eq.symm equality)
  -- chapter_34_line_2: GL tag recursion.
  have row_2 : (v2 = zero) := by
    exact assumption_2
  -- chapter_34_line_1: GL tag vacuous truth.
  have row_1 : (one = v1) := by
    exact False.elim (row_3 row_2)
  exact row_1

private theorem fta_source_027_check_induction_condition
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_027 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_8 x_6) → (x_4 x_8 x_7 x_6))))))
    (external_peano_externals_36_028 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_5 x_7 x_8 x_6) → (x_5 x_8 x_7 x_6))))))
    (external_peano_externals_36_015 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_3 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_022 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_6 x_8) → (x_4 x_6 x_7 x_8))))))
    (external_peano_externals_36_029 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((gl_preorder x_1 x_4 x_7 x_8) → ((gl_preorder x_1 x_4 x_8 x_7) → (x_7 = x_8)))))))
    (previous : α)
    (v1 : α)
    (v2 : α)
    (assumption_29 : (succ previous v2))
    (assumption_16 : (mul v1 v2 one))
    : (one = v1) := by
  -- chapter_35_line_29: GL tag recursion.
  have row_29 : (succ previous v2) := by
    exact assumption_29
  -- chapter_35_line_24: GL tag theorem.
  have row_24 := fta_source_002 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_35_line_16: GL tag task formulation.
  have row_16 : (mul v1 v2 one) := by
    exact assumption_16
  -- chapter_35_line_12: GL tag expansion for integration.
  have row_12 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_35_line_11: GL tag reformulation for integration and.
  have row_11 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_35_line_8: GL tag theorem.
  have row_8 := fta_source_025 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_35_line_6: GL tag theorem.
  have row_6 := fta_source_059 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_015 external_peano_externals_36_022
  -- chapter_35_line_4: GL tag task formulation.
  have row_4 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_35_line_14: GL tag expansion.
  have row_14 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_4
  -- chapter_35_line_15: GL tag disintegration.
  have row_15 : (succ zero one) := by
    exact row_14.1.1.2
  -- chapter_35_line_13: GL tag disintegration.
  have row_13 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_14.1.1.1
  -- chapter_35_line_22: GL tag expansion.
  have row_22 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_13
  -- chapter_35_line_39: GL tag disintegration.
  have row_39 : (gl_fXY succ N N) := by
    exact row_22.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_35_line_38: GL tag expansion.
  have row_38 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_39
  -- chapter_35_line_37: GL tag disintegration.
  have row_37 : (gl_implication0 succ N) := by
    exact row_38.1.1.1
  -- chapter_35_line_36: GL tag expansion.
  have row_36 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_37
  -- chapter_35_line_35: GL tag implication.
  have row_35 : (N previous) := by
    apply row_36
    exact row_29
  -- chapter_35_line_28: GL tag disintegration.
  have row_28 : (gl_implication21 N succ mul add) := by
    exact row_22.2
  -- chapter_35_line_27: GL tag expansion.
  have row_27 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((mul w3 w1 w4) → (∀ (w5 : α), ((mul w3 w2 w5) → (add w4 w3 w5))))))))) := by
    simpa only [gl_implication21] using row_28
  -- chapter_35_line_21: GL tag disintegration.
  have row_21 : (gl_fXYZ mul N N N) := by
    exact row_22.1.1.1.2
  -- chapter_35_line_20: GL tag expansion.
  have row_20 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_21
  -- chapter_35_line_34: GL tag disintegration.
  have row_34 : (gl_implication13 N N N mul) := by
    exact row_20.1.2
  -- chapter_35_line_33: GL tag expansion.
  have row_33 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication13] using row_34
  -- chapter_35_line_19: GL tag disintegration.
  have row_19 : (gl_implication8 mul N) := by
    exact row_20.1.1.1.1
  -- chapter_35_line_18: GL tag expansion.
  have row_18 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_19
  -- chapter_35_line_17: GL tag implication.
  have row_17 : (N v1) := by
    apply row_18
    exact row_16
  -- chapter_35_line_32: GL tag implication.
  have row_32 : (gl_existence1 N v1 previous mul) := by
    apply row_33
    exact row_17
    exact row_35
  -- chapter_35_line_31: GL tag expansion.
  have row_31 : (¬ (∀ (v6 : α), ((N v6) → (¬ (mul v1 previous v6))))) := by
    simpa only [gl_existence1] using row_32
  have exists_row_31 : ∃ (v6 : α), ((N v6) ∧ (mul v1 previous v6)) := existsAndOfNotForallImpNot row_31
  obtain ⟨v6, witness_row_31⟩ := exists_row_31
  -- chapter_35_line_30: GL tag disintegration.
  have row_30 : (mul v1 previous v6) := by
    exact witness_row_31.2
  -- chapter_35_line_26: GL tag implication.
  have row_26 : (add v6 v1 one) := by
    apply row_27
    exact row_35
    exact row_29
    exact row_30
    exact row_16
  -- chapter_35_line_10: GL tag implication.
  have row_10 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_11
    exact row_13
    exact row_15
  have rule_row_25 := external_peano_externals_36_027 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_35_line_25: GL tag implication.
  have row_25 : (add v1 v6 one) := by
    apply rule_row_25
    exact row_26
  have rule_row_9 := external_peano_externals_36_028 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_35_line_9: GL tag implication.
  have row_9 : (mul v2 v1 one) := by
    apply rule_row_9
    exact row_16
  -- chapter_35_line_7: GL tag implication.
  have row_7 : (¬ (zero = v1)) := by
    apply row_8
    exact row_9
  -- chapter_35_line_5: GL tag implication.
  have row_5 : (gl_preorder N add one v1) := by
    apply row_6
    exact row_17
    exact row_7
  -- chapter_35_line_3: GL tag anchor handling.
  have row_3 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact row_4
  -- chapter_35_line_23: GL tag implication.
  have row_23 : (gl_preorder N add v1 one) := by
    apply row_24
    exact row_25
  -- chapter_35_line_2: GL tag theorem.
  have row_2 := fta_source_033 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_029
  -- chapter_35_line_1: GL tag implication.
  have row_1 : (one = v1) := by
    apply row_2
    exact row_5
    exact row_23
  exact row_1

theorem fta_source_027
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_027 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_8 x_6) → (x_4 x_8 x_7 x_6))))))
    (external_peano_externals_36_028 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_5 x_7 x_8 x_6) → (x_5 x_8 x_7 x_6))))))
    (external_peano_externals_36_015 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_3 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_022 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_6 x_8) → (x_4 x_6 x_7 x_8))))))
    (external_peano_externals_36_029 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((gl_preorder x_1 x_4 x_7 x_8) → ((gl_preorder x_1 x_4 x_8 x_7) → (x_7 = x_8)))))))
    : (∀ (v1 : α) (v2 : α), ((mul v1 v2 one) → ((N v2) → (one = v1)))) := by
  intro v1
  intro v2
  intro premise_1
  intro premise_2
  have inductionMember : N v2 := by
    have typingRule := fta_source_027_induction_typing N zero succ add mul one two identity anchor relationalInduction
    exact typingRule v2 premise_2
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorFTA, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v1 : α), ((mul v1 zero one) → ((N zero) → (one = v1)))) := by
    intro v1
    intro base_premise_1
    intro base_premise_2
    have zeroRule := fta_source_027_check_zero N zero succ add mul one two identity anchor relationalInduction
    exact zeroRule v1 zero base_premise_1 rfl
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v1 : α), ((mul v1 induction_n one) → ((N induction_n) → (one = v1)))) → ∀ induction_m, succ induction_n induction_m → (∀ (v1 : α), ((mul v1 induction_m one) → ((N induction_m) → (one = v1)))) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro v1
    intro step_premise_1
    intro step_premise_2
    have step_induction_assumption_1 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have stepRule := fta_source_027_check_induction_condition N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_027 external_peano_externals_36_028 external_peano_externals_36_015 external_peano_externals_36_022 external_peano_externals_36_029
    exact stepRule induction_n v1 induction_m step_induction_assumption_1 step_premise_1
  have inductionProperty : (∀ (v1 : α), ((mul v1 v2 one) → ((N v2) → (one = v1)))) := by
    exact relationalInduction N zero succ add mul one two identity anchor
      (fun induction_value => (∀ (v1 : α), ((mul v1 induction_value one) → ((N induction_value) → (one = v1)))))
      v2
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v1 premise_1 premise_2

private theorem fta_source_028_induction_typing
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (v1 : α)
    (assumption_1 : (N v1))
    : (N v1) := by
  -- chapter_36_line_1: GL tag task formulation.
  have row_1 : (N v1) := by
    exact assumption_1
  exact row_1

private theorem fta_source_028_check_zero
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_028 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_5 x_7 x_8 x_6) → (x_5 x_8 x_7 x_6))))))
    (v1 : α)
    (v2 : α)
    (assumption_14 : (mul v1 v2 one))
    (assumption_2 : (v1 = zero))
    : (one = v2) := by
  -- chapter_37_line_14: GL tag task formulation.
  have row_14 : (mul v1 v2 one) := by
    exact assumption_14
  -- chapter_37_line_10: GL tag expansion for integration.
  have row_10 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_37_line_9: GL tag reformulation for integration and.
  have row_9 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_37_line_6: GL tag task formulation.
  have row_6 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_37_line_12: GL tag expansion.
  have row_12 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_6
  -- chapter_37_line_13: GL tag disintegration.
  have row_13 : (succ zero one) := by
    exact row_12.1.1.2
  -- chapter_37_line_11: GL tag disintegration.
  have row_11 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_12.1.1.1
  -- chapter_37_line_8: GL tag implication.
  have row_8 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_9
    exact row_11
    exact row_13
  have rule_row_7 := external_peano_externals_36_028 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_37_line_7: GL tag implication.
  have row_7 : (mul v2 v1 one) := by
    apply rule_row_7
    exact row_14
  -- chapter_37_line_5: GL tag theorem.
  have row_5 := fta_source_025 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_37_line_4: GL tag implication.
  have row_4 : (¬ (zero = v1)) := by
    apply row_5
    exact row_7
  -- chapter_37_line_3: GL tag symmetry of inequality.
  have row_3 : (¬ (v1 = zero)) := by
    exact fun equality => row_4 (Eq.symm equality)
  -- chapter_37_line_2: GL tag recursion.
  have row_2 : (v1 = zero) := by
    exact assumption_2
  -- chapter_37_line_1: GL tag vacuous truth.
  have row_1 : (one = v2) := by
    exact False.elim (row_3 row_2)
  exact row_1

private theorem fta_source_028_check_induction_condition
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_028 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_5 x_7 x_8 x_6) → (x_5 x_8 x_7 x_6))))))
    (external_peano_externals_36_027 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_8 x_6) → (x_4 x_8 x_7 x_6))))))
    (external_peano_externals_36_015 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_3 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_022 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_6 x_8) → (x_4 x_6 x_7 x_8))))))
    (external_peano_externals_36_029 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((gl_preorder x_1 x_4 x_7 x_8) → ((gl_preorder x_1 x_4 x_8 x_7) → (x_7 = x_8)))))))
    (previous : α)
    (v1 : α)
    (v2 : α)
    (assumption_20 : (mul v1 v2 one))
    (assumption_18 : (succ previous v1))
    : (one = v2) := by
  -- chapter_38_line_39: GL tag theorem.
  have row_39 := fta_source_025 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_38_line_37: GL tag theorem.
  have row_37 := fta_source_059 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_015 external_peano_externals_36_022
  -- chapter_38_line_20: GL tag task formulation.
  have row_20 : (mul v1 v2 one) := by
    exact assumption_20
  -- chapter_38_line_18: GL tag recursion.
  have row_18 : (succ previous v1) := by
    exact assumption_18
  -- chapter_38_line_10: GL tag expansion for integration.
  have row_10 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_38_line_9: GL tag reformulation for integration and.
  have row_9 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_38_line_6: GL tag theorem.
  have row_6 := fta_source_002 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_38_line_4: GL tag task formulation.
  have row_4 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_38_line_38: GL tag implication.
  have row_38 : (¬ (zero = v2)) := by
    apply row_39
    exact row_20
  -- chapter_38_line_12: GL tag expansion.
  have row_12 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_4
  -- chapter_38_line_13: GL tag disintegration.
  have row_13 : (succ zero one) := by
    exact row_12.1.1.2
  -- chapter_38_line_11: GL tag disintegration.
  have row_11 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_12.1.1.1
  -- chapter_38_line_17: GL tag expansion.
  have row_17 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_11
  -- chapter_38_line_35: GL tag disintegration.
  have row_35 : (gl_fXY succ N N) := by
    exact row_17.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_38_line_34: GL tag expansion.
  have row_34 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_35
  -- chapter_38_line_33: GL tag disintegration.
  have row_33 : (gl_implication0 succ N) := by
    exact row_34.1.1.1
  -- chapter_38_line_32: GL tag expansion.
  have row_32 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_33
  -- chapter_38_line_31: GL tag implication.
  have row_31 : (N previous) := by
    apply row_32
    exact row_18
  -- chapter_38_line_27: GL tag disintegration.
  have row_27 : (gl_fXYZ mul N N N) := by
    exact row_17.1.1.1.2
  -- chapter_38_line_26: GL tag expansion.
  have row_26 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_27
  -- chapter_38_line_30: GL tag disintegration.
  have row_30 : (gl_implication9 mul N) := by
    exact row_26.1.1.1.2
  -- chapter_38_line_29: GL tag expansion.
  have row_29 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication9] using row_30
  -- chapter_38_line_28: GL tag implication.
  have row_28 : (N v2) := by
    apply row_29
    exact row_20
  -- chapter_38_line_36: GL tag implication.
  have row_36 : (gl_preorder N add one v2) := by
    apply row_37
    exact row_28
    exact row_38
  -- chapter_38_line_25: GL tag disintegration.
  have row_25 : (gl_implication13 N N N mul) := by
    exact row_26.1.2
  -- chapter_38_line_24: GL tag expansion.
  have row_24 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication13] using row_25
  -- chapter_38_line_23: GL tag implication.
  have row_23 : (gl_existence1 N v2 previous mul) := by
    apply row_24
    exact row_28
    exact row_31
  -- chapter_38_line_22: GL tag expansion.
  have row_22 : (¬ (∀ (v3 : α), ((N v3) → (¬ (mul v2 previous v3))))) := by
    simpa only [gl_existence1] using row_23
  have exists_row_22 : ∃ (v3 : α), ((N v3) ∧ (mul v2 previous v3)) := existsAndOfNotForallImpNot row_22
  obtain ⟨v3, witness_row_22⟩ := exists_row_22
  -- chapter_38_line_21: GL tag disintegration.
  have row_21 : (mul v2 previous v3) := by
    exact witness_row_22.2
  -- chapter_38_line_16: GL tag disintegration.
  have row_16 : (gl_implication21 N succ mul add) := by
    exact row_17.2
  -- chapter_38_line_15: GL tag expansion.
  have row_15 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((mul w3 w1 w4) → (∀ (w5 : α), ((mul w3 w2 w5) → (add w4 w3 w5))))))))) := by
    simpa only [gl_implication21] using row_16
  -- chapter_38_line_8: GL tag implication.
  have row_8 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_9
    exact row_11
    exact row_13
  have rule_row_19 := external_peano_externals_36_028 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_38_line_19: GL tag implication.
  have row_19 : (mul v2 v1 one) := by
    apply rule_row_19
    exact row_20
  -- chapter_38_line_14: GL tag implication.
  have row_14 : (add v3 v2 one) := by
    apply row_15
    exact row_31
    exact row_18
    exact row_21
    exact row_19
  have rule_row_7 := external_peano_externals_36_027 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_38_line_7: GL tag implication.
  have row_7 : (add v2 v3 one) := by
    apply rule_row_7
    exact row_14
  -- chapter_38_line_3: GL tag anchor handling.
  have row_3 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact row_4
  -- chapter_38_line_5: GL tag implication.
  have row_5 : (gl_preorder N add v2 one) := by
    apply row_6
    exact row_7
  -- chapter_38_line_2: GL tag theorem.
  have row_2 := fta_source_033 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_029
  -- chapter_38_line_1: GL tag implication.
  have row_1 : (one = v2) := by
    apply row_2
    exact row_36
    exact row_5
  exact row_1

theorem fta_source_028
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_028 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_5 x_7 x_8 x_6) → (x_5 x_8 x_7 x_6))))))
    (external_peano_externals_36_027 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_8 x_6) → (x_4 x_8 x_7 x_6))))))
    (external_peano_externals_36_015 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_3 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_022 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_6 x_8) → (x_4 x_6 x_7 x_8))))))
    (external_peano_externals_36_029 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((gl_preorder x_1 x_4 x_7 x_8) → ((gl_preorder x_1 x_4 x_8 x_7) → (x_7 = x_8)))))))
    : (∀ (v1 : α) (v2 : α), ((mul v1 v2 one) → ((N v1) → (one = v2)))) := by
  intro v1
  intro v2
  intro premise_1
  intro premise_2
  have inductionMember : N v1 := by
    have typingRule := fta_source_028_induction_typing N zero succ add mul one two identity anchor relationalInduction
    exact typingRule v1 premise_2
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorFTA, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v2 : α), ((mul zero v2 one) → ((N zero) → (one = v2)))) := by
    intro v2
    intro base_premise_1
    intro base_premise_2
    have zeroRule := fta_source_028_check_zero N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_028
    exact zeroRule zero v2 base_premise_1 rfl
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v2 : α), ((mul induction_n v2 one) → ((N induction_n) → (one = v2)))) → ∀ induction_m, succ induction_n induction_m → (∀ (v2 : α), ((mul induction_m v2 one) → ((N induction_m) → (one = v2)))) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro v2
    intro step_premise_1
    intro step_premise_2
    have step_induction_assumption_1 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have stepRule := fta_source_028_check_induction_condition N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_028 external_peano_externals_36_027 external_peano_externals_36_015 external_peano_externals_36_022 external_peano_externals_36_029
    exact stepRule induction_n induction_m v2 step_premise_1 step_induction_assumption_1
  have inductionProperty : (∀ (v2 : α), ((mul v1 v2 one) → ((N v1) → (one = v2)))) := by
    exact relationalInduction N zero succ add mul one two identity anchor
      (fun induction_value => (∀ (v2 : α), ((mul induction_value v2 one) → ((N induction_value) → (one = v2)))))
      v1
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v2 premise_1 premise_2

theorem fta_source_029
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_006 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (x_4 x_8 x_7 x_9))))))
    (external_peano_externals_36_002 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (∀ (w4 : α) (w5 : α), ((add w4 w2 w5) → (∀ (w6 : α), ((add w3 w4 w6) → (add w5 w1 w6))))))))))
    : (∀ (v1 : α) (v2 : α), ((gl_preorder N add v1 v2) → (∀ (v3 : α) (v4 : α), ((add v1 v3 v4) → (∀ (v5 : α), ((add v2 v3 v5) → (gl_preorder N add v4 v5))))))) := by
  intro v1
  intro v2
  intro premise_1
  intro v3
  intro v4
  intro premise_2
  intro v5
  intro premise_3
  -- chapter_39_line_17: GL tag task formulation.
  have row_17 : (gl_preorder N add v1 v2) := by
    exact premise_1
  -- chapter_39_line_16: GL tag expansion.
  have row_16 : (¬ (∀ (v6 : α), ((N v6) → (¬ (add v1 v6 v2))))) := by
    simpa only [gl_preorder] using row_17
  have exists_row_16 : ∃ (v6 : α), ((N v6) ∧ (add v1 v6 v2)) := existsAndOfNotForallImpNot row_16
  obtain ⟨v6, witness_row_16⟩ := exists_row_16
  -- chapter_39_line_15: GL tag disintegration.
  have row_15 : (add v1 v6 v2) := by
    exact witness_row_16.2
  -- chapter_39_line_13: GL tag task formulation.
  have row_13 : (add v1 v3 v4) := by
    exact premise_2
  -- chapter_39_line_11: GL tag task formulation.
  have row_11 : (add v2 v3 v5) := by
    exact premise_3
  -- chapter_39_line_7: GL tag expansion for integration.
  have row_7 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_39_line_6: GL tag reformulation for integration and.
  have row_6 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_39_line_3: GL tag task formulation.
  have row_3 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_39_line_9: GL tag expansion.
  have row_9 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_3
  -- chapter_39_line_10: GL tag disintegration.
  have row_10 : (succ zero one) := by
    exact row_9.1.1.2
  -- chapter_39_line_8: GL tag disintegration.
  have row_8 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_9.1.1.1
  -- chapter_39_line_5: GL tag implication.
  have row_5 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_6
    exact row_8
    exact row_10
  have rule_row_14 := external_peano_externals_36_006 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_39_line_14: GL tag implication.
  have row_14 : (add v6 v1 v2) := by
    apply rule_row_14
    exact row_15
  have rule_row_12 := external_peano_externals_36_006 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_39_line_12: GL tag implication.
  have row_12 : (add v3 v1 v4) := by
    apply rule_row_12
    exact row_13
  have rule_row_4 := external_peano_externals_36_002 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_39_line_4: GL tag implication.
  have row_4 : (add v4 v6 v5) := by
    apply rule_row_4
    exact row_14
    exact row_12
    exact row_11
  -- chapter_39_line_2: GL tag theorem.
  have row_2 := fta_source_002 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_39_line_1: GL tag implication.
  have row_1 : (gl_preorder N add v4 v5) := by
    apply row_2
    exact row_4
  exact row_1

theorem fta_source_034
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    : (∀ (v1 : α) (v2 : α), ((gl_preorder N add v1 v2) → (gl_or0 v1 v2 N add))) := by
  intro v1
  intro v2
  intro premise_1
  have or_parent_1 := fta_source_031 N zero succ add mul one two identity anchor relationalInduction
  have or_parent_2 := fta_source_032 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_44_line_1: GL tag or theorem.
  have row_1 : (∀ (v1 : α) (v2 : α), ((gl_preorder N add v1 v2) → (gl_or0 v1 v2 N add))) := by
    classical
    intro v1
    intro v2
    intro or_parent_premise_1
    simp only [gl_or0]
    by_cases or_case_1 : (v1 = v2)
    · exact Or.inl (or_case_1)
    ·
      exact Or.inr ((or_parent_1 v1 v2 or_parent_premise_1 or_case_1))
  solve_by_elim [row_1]

theorem fta_source_036
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_009 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (∀ (w4 : α) (w5 : α), ((mul w4 w2 w5) → (∀ (w6 : α), ((mul w3 w4 w6) → (mul w5 w1 w6))))))))))
    (external_peano_externals_36_010 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_5 x_7 x_8 x_9) → (x_5 x_8 x_7 x_9))))))
    : (∀ (v1 : α) (v2 : α), ((gl_preorder N mul v1 v2) → (∀ (v3 : α) (v4 : α), ((mul v2 v3 v4) → (gl_preorder N mul v1 v4))))) := by
  intro v1
  intro v2
  intro premise_1
  intro v3
  intro v4
  intro premise_2
  -- chapter_46_line_25: GL tag task formulation.
  have row_25 : (gl_preorder N mul v1 v2) := by
    exact premise_1
  -- chapter_46_line_24: GL tag expansion.
  have row_24 : (¬ (∀ (v6 : α), ((N v6) → (¬ (mul v1 v6 v2))))) := by
    simpa only [gl_preorder] using row_25
  have exists_row_24 : ∃ (v6 : α), ((N v6) ∧ (mul v1 v6 v2)) := existsAndOfNotForallImpNot row_24
  obtain ⟨v6, witness_row_24⟩ := exists_row_24
  -- chapter_46_line_26: GL tag disintegration.
  have row_26 : (mul v1 v6 v2) := by
    exact witness_row_24.2
  -- chapter_46_line_23: GL tag disintegration.
  have row_23 : (N v6) := by
    exact witness_row_24.1
  -- chapter_46_line_11: GL tag task formulation.
  have row_11 : (mul v2 v3 v4) := by
    exact premise_2
  -- chapter_46_line_7: GL tag expansion for integration.
  have row_7 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_46_line_6: GL tag reformulation for integration and.
  have row_6 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_46_line_3: GL tag task formulation.
  have row_3 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_46_line_9: GL tag expansion.
  have row_9 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_3
  -- chapter_46_line_10: GL tag disintegration.
  have row_10 : (succ zero one) := by
    exact row_9.1.1.2
  -- chapter_46_line_8: GL tag disintegration.
  have row_8 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_9.1.1.1
  -- chapter_46_line_19: GL tag expansion.
  have row_19 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_8
  -- chapter_46_line_18: GL tag disintegration.
  have row_18 : (gl_fXYZ mul N N N) := by
    exact row_19.1.1.1.2
  -- chapter_46_line_17: GL tag expansion.
  have row_17 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_18
  -- chapter_46_line_22: GL tag disintegration.
  have row_22 : (gl_implication9 mul N) := by
    exact row_17.1.1.1.2
  -- chapter_46_line_21: GL tag expansion.
  have row_21 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication9] using row_22
  -- chapter_46_line_20: GL tag implication.
  have row_20 : (N v3) := by
    apply row_21
    exact row_11
  -- chapter_46_line_16: GL tag disintegration.
  have row_16 : (gl_implication13 N N N mul) := by
    exact row_17.1.2
  -- chapter_46_line_15: GL tag expansion.
  have row_15 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication13] using row_16
  -- chapter_46_line_14: GL tag implication.
  have row_14 : (gl_existence1 N v3 v6 mul) := by
    apply row_15
    exact row_20
    exact row_23
  -- chapter_46_line_13: GL tag expansion.
  have row_13 : (¬ (∀ (v5 : α), ((N v5) → (¬ (mul v3 v6 v5))))) := by
    simpa only [gl_existence1] using row_14
  have exists_row_13 : ∃ (v5 : α), ((N v5) ∧ (mul v3 v6 v5)) := existsAndOfNotForallImpNot row_13
  obtain ⟨v5, witness_row_13⟩ := exists_row_13
  -- chapter_46_line_12: GL tag disintegration.
  have row_12 : (mul v3 v6 v5) := by
    exact witness_row_13.2
  -- chapter_46_line_5: GL tag implication.
  have row_5 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_6
    exact row_8
    exact row_10
  have rule_row_4 := external_peano_externals_36_009 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_46_line_4: GL tag implication.
  have row_4 : (mul v5 v1 v4) := by
    apply rule_row_4
    exact row_26
    exact row_12
    exact row_11
  -- chapter_46_line_2: GL tag theorem.
  have row_2 := fta_source_010 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_010
  -- chapter_46_line_1: GL tag implication.
  have row_1 : (gl_preorder N mul v1 v4) := by
    apply row_2
    exact row_4
  exact row_1

theorem fta_source_039
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_009 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (∀ (w4 : α) (w5 : α), ((mul w4 w2 w5) → (∀ (w6 : α), ((mul w3 w4 w6) → (mul w5 w1 w6))))))))))
    (external_peano_externals_36_010 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_5 x_7 x_8 x_9) → (x_5 x_8 x_7 x_9))))))
    : (∀ (v1 : α) (v2 : α), ((gl_preorder N mul v1 v2) → (∀ (v3 : α), ((gl_preorder N mul v2 v3) → (gl_preorder N mul v1 v3))))) := by
  intro v1
  intro v2
  intro premise_1
  intro v3
  intro premise_2
  -- chapter_49_line_16: GL tag task formulation.
  have row_16 : (gl_preorder N mul v1 v2) := by
    exact premise_1
  -- chapter_49_line_15: GL tag expansion.
  have row_15 : (¬ (∀ (v6 : α), ((N v6) → (¬ (mul v1 v6 v2))))) := by
    simpa only [gl_preorder] using row_16
  have exists_row_15 : ∃ (v6 : α), ((N v6) ∧ (mul v1 v6 v2)) := existsAndOfNotForallImpNot row_15
  obtain ⟨v6, witness_row_15⟩ := exists_row_15
  -- chapter_49_line_25: GL tag disintegration.
  have row_25 : (N v6) := by
    exact witness_row_15.1
  -- chapter_49_line_14: GL tag disintegration.
  have row_14 : (mul v1 v6 v2) := by
    exact witness_row_15.2
  -- chapter_49_line_13: GL tag task formulation.
  have row_13 : (gl_preorder N mul v2 v3) := by
    exact premise_2
  -- chapter_49_line_12: GL tag expansion.
  have row_12 : (¬ (∀ (v5 : α), ((N v5) → (¬ (mul v2 v5 v3))))) := by
    simpa only [gl_preorder] using row_13
  have exists_row_12 : ∃ (v5 : α), ((N v5) ∧ (mul v2 v5 v3)) := existsAndOfNotForallImpNot row_12
  obtain ⟨v5, witness_row_12⟩ := exists_row_12
  -- chapter_49_line_26: GL tag disintegration.
  have row_26 : (N v5) := by
    exact witness_row_12.1
  -- chapter_49_line_11: GL tag disintegration.
  have row_11 : (mul v2 v5 v3) := by
    exact witness_row_12.2
  -- chapter_49_line_7: GL tag expansion for integration.
  have row_7 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_49_line_6: GL tag reformulation for integration and.
  have row_6 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_49_line_3: GL tag task formulation.
  have row_3 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_49_line_9: GL tag expansion.
  have row_9 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_3
  -- chapter_49_line_10: GL tag disintegration.
  have row_10 : (succ zero one) := by
    exact row_9.1.1.2
  -- chapter_49_line_8: GL tag disintegration.
  have row_8 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_9.1.1.1
  -- chapter_49_line_24: GL tag expansion.
  have row_24 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_8
  -- chapter_49_line_23: GL tag disintegration.
  have row_23 : (gl_fXYZ mul N N N) := by
    exact row_24.1.1.1.2
  -- chapter_49_line_22: GL tag expansion.
  have row_22 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_23
  -- chapter_49_line_21: GL tag disintegration.
  have row_21 : (gl_implication13 N N N mul) := by
    exact row_22.1.2
  -- chapter_49_line_20: GL tag expansion.
  have row_20 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication13] using row_21
  -- chapter_49_line_19: GL tag implication.
  have row_19 : (gl_existence1 N v5 v6 mul) := by
    apply row_20
    exact row_26
    exact row_25
  -- chapter_49_line_18: GL tag expansion.
  have row_18 : (¬ (∀ (v4 : α), ((N v4) → (¬ (mul v5 v6 v4))))) := by
    simpa only [gl_existence1] using row_19
  have exists_row_18 : ∃ (v4 : α), ((N v4) ∧ (mul v5 v6 v4)) := existsAndOfNotForallImpNot row_18
  obtain ⟨v4, witness_row_18⟩ := exists_row_18
  -- chapter_49_line_17: GL tag disintegration.
  have row_17 : (mul v5 v6 v4) := by
    exact witness_row_18.2
  -- chapter_49_line_5: GL tag implication.
  have row_5 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_6
    exact row_8
    exact row_10
  have rule_row_4 := external_peano_externals_36_009 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_49_line_4: GL tag implication.
  have row_4 : (mul v4 v1 v3) := by
    apply rule_row_4
    exact row_14
    exact row_17
    exact row_11
  -- chapter_49_line_2: GL tag theorem.
  have row_2 := fta_source_010 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_010
  -- chapter_49_line_1: GL tag implication.
  have row_1 : (gl_preorder N mul v1 v3) := by
    apply row_2
    exact row_4
  exact row_1

private theorem fta_source_041_induction_typing
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (v1 : α)
    (v2 : α)
    (assumption_12 : (gl_preorder N mul v1 v2))
    : (N v1) := by
  -- chapter_51_line_12: GL tag task formulation.
  have row_12 : (gl_preorder N mul v1 v2) := by
    exact assumption_12
  -- chapter_51_line_11: GL tag expansion.
  have row_11 : (¬ (∀ (v6 : α), ((N v6) → (¬ (mul v1 v6 v2))))) := by
    simpa only [gl_preorder] using row_12
  have exists_row_11 : ∃ (v6 : α), ((N v6) ∧ (mul v1 v6 v2)) := existsAndOfNotForallImpNot row_11
  obtain ⟨v6, witness_row_11⟩ := exists_row_11
  -- chapter_51_line_10: GL tag disintegration.
  have row_10 : (mul v1 v6 v2) := by
    exact witness_row_11.2
  -- chapter_51_line_9: GL tag task formulation.
  have row_9 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_51_line_8: GL tag expansion.
  have row_8 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_9
  -- chapter_51_line_7: GL tag disintegration.
  have row_7 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_8.1.1.1
  -- chapter_51_line_6: GL tag expansion.
  have row_6 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_7
  -- chapter_51_line_5: GL tag disintegration.
  have row_5 : (gl_fXYZ mul N N N) := by
    exact row_6.1.1.1.2
  -- chapter_51_line_4: GL tag expansion.
  have row_4 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_5
  -- chapter_51_line_3: GL tag disintegration.
  have row_3 : (gl_implication8 mul N) := by
    exact row_4.1.1.1.1
  -- chapter_51_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_3
  -- chapter_51_line_1: GL tag implication.
  have row_1 : (N v1) := by
    apply row_2
    exact row_10
  exact row_1

private theorem fta_source_041_check_zero
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_010 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_5 x_7 x_8 x_9) → (x_5 x_8 x_7 x_9))))))
    (external_peano_externals_36_017 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_6 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_021 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_6 x_8) → (x_3 x_7 x_8))))))
    (v1 : α)
    (v2 : α)
    (assumption_28 : (v1 = zero))
    (assumption_27 : (gl_preorder N mul v1 v2))
    (assumption_11 : (gl_preorder N add one v2))
    : (gl_preorder N add one v1) := by
  -- chapter_52_line_28: GL tag recursion.
  have row_28 : (v1 = zero) := by
    exact assumption_28
  -- chapter_52_line_27: GL tag task formulation.
  have row_27 : (gl_preorder N mul v1 v2) := by
    exact assumption_27
  -- chapter_52_line_26: GL tag expansion.
  have row_26 : (¬ (∀ (v6 : α), ((N v6) → (¬ (mul v1 v6 v2))))) := by
    simpa only [gl_preorder] using row_27
  have exists_row_26 : ∃ (v6 : α), ((N v6) ∧ (mul v1 v6 v2)) := existsAndOfNotForallImpNot row_26
  obtain ⟨v6, witness_row_26⟩ := exists_row_26
  -- chapter_52_line_29: GL tag disintegration.
  have row_29 : (N v6) := by
    exact witness_row_26.1
  -- chapter_52_line_25: GL tag disintegration.
  have row_25 : (mul v1 v6 v2) := by
    exact witness_row_26.2
  -- chapter_52_line_16: GL tag expansion for integration.
  have row_16 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_52_line_15: GL tag reformulation for integration and.
  have row_15 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_52_line_11: GL tag task formulation.
  have row_11 : (gl_preorder N add one v2) := by
    exact assumption_11
  -- chapter_52_line_10: GL tag expansion.
  have row_10 : (¬ (∀ (v3 : α), ((N v3) → (¬ (add one v3 v2))))) := by
    simpa only [gl_preorder] using row_11
  have exists_row_10 : ∃ (v3 : α), ((N v3) ∧ (add one v3 v2)) := existsAndOfNotForallImpNot row_10
  obtain ⟨v3, witness_row_10⟩ := exists_row_10
  -- chapter_52_line_19: GL tag disintegration.
  have row_19 : (add one v3 v2) := by
    exact witness_row_10.2
  -- chapter_52_line_9: GL tag disintegration.
  have row_9 : (N v3) := by
    exact witness_row_10.1
  -- chapter_52_line_8: GL tag task formulation.
  have row_8 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_52_line_7: GL tag expansion.
  have row_7 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_8
  -- chapter_52_line_17: GL tag disintegration.
  have row_17 : (succ zero one) := by
    exact row_7.1.1.2
  -- chapter_52_line_6: GL tag disintegration.
  have row_6 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_7.1.1.1
  -- chapter_52_line_14: GL tag implication.
  have row_14 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_15
    exact row_6
    exact row_17
  have rule_row_24 := external_peano_externals_36_010 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_52_line_24: GL tag implication.
  have row_24 : (mul v6 v1 v2) := by
    apply rule_row_24
    exact row_25
  -- chapter_52_line_23: GL tag equality1.
  have row_23 : (mul v6 zero v2) := by
    have equality_source := row_24
    have equality_step_1 := row_28
    cases equality_step_1
    exact equality_source
  have rule_row_18 := external_peano_externals_36_017 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_52_line_18: GL tag implication.
  have row_18 : (add v3 one v2) := by
    apply rule_row_18
    exact row_19
  have rule_row_13 := external_peano_externals_36_021 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_52_line_13: GL tag implication.
  have row_13 : (succ v3 v2) := by
    apply rule_row_13
    exact row_18
  -- chapter_52_line_5: GL tag expansion.
  have row_5 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_6
  -- chapter_52_line_22: GL tag disintegration.
  have row_22 : (gl_implication19 N zero mul) := by
    exact row_5.1.1.2
  -- chapter_52_line_21: GL tag expansion.
  have row_21 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((mul w1 zero w2) → (w2 = zero))))) := by
    simpa only [gl_implication19] using row_22
  -- chapter_52_line_20: GL tag implication.
  have row_20 : (v2 = zero) := by
    apply row_21
    exact row_29
    exact row_23
  -- chapter_52_line_12: GL tag equality1.
  have row_12 : (succ v3 zero) := by
    have equality_source := row_13
    have equality_step_1 := row_20
    cases equality_step_1
    exact equality_source
  -- chapter_52_line_4: GL tag disintegration.
  have row_4 : (gl_implication6 N zero succ) := by
    exact row_5.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_52_line_3: GL tag expansion.
  have row_3 : (∀ (w1 : α), ((N w1) → (¬ (succ w1 zero)))) := by
    simpa only [gl_implication6] using row_4
  -- chapter_52_line_2: GL tag implication.
  have row_2 : (¬ (succ v3 zero)) := by
    apply row_3
    exact row_9
  -- chapter_52_line_1: GL tag vacuous truth.
  have row_1 : (gl_preorder N add one v1) := by
    exact False.elim (row_2 row_12)
  exact row_1

private theorem fta_source_041_check_induction_condition
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_015 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_3 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_022 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_6 x_8) → (x_4 x_6 x_7 x_8))))))
    (previous : α)
    (v1 : α)
    (assumption_13 : (succ previous v1))
    : (gl_preorder N add one v1) := by
  -- chapter_53_line_13: GL tag recursion.
  have row_13 : (succ previous v1) := by
    exact assumption_13
  -- chapter_53_line_8: GL tag expansion for integration.
  have row_8 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_53_line_7: GL tag reformulation for integration and.
  have row_7 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_53_line_4: GL tag task formulation.
  have row_4 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_53_line_10: GL tag expansion.
  have row_10 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_4
  -- chapter_53_line_11: GL tag disintegration.
  have row_11 : (succ zero one) := by
    exact row_10.1.1.2
  -- chapter_53_line_9: GL tag disintegration.
  have row_9 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_10.1.1.1
  -- chapter_53_line_6: GL tag implication.
  have row_6 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_7
    exact row_9
    exact row_11
  have rule_row_12 := external_peano_externals_36_015 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_53_line_12: GL tag implication.
  have row_12 : (add previous one v1) := by
    apply rule_row_12
    exact row_13
  have rule_row_5 := external_peano_externals_36_022 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_53_line_5: GL tag implication.
  have row_5 : (add one previous v1) := by
    apply rule_row_5
    exact row_12
  -- chapter_53_line_3: GL tag anchor handling.
  have row_3 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact row_4
  -- chapter_53_line_2: GL tag theorem.
  have row_2 := fta_source_002 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_53_line_1: GL tag implication.
  have row_1 : (gl_preorder N add one v1) := by
    apply row_2
    exact row_5
  exact row_1

theorem fta_source_041
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_010 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_5 x_7 x_8 x_9) → (x_5 x_8 x_7 x_9))))))
    (external_peano_externals_36_017 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_6 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_021 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_6 x_8) → (x_3 x_7 x_8))))))
    (external_peano_externals_36_015 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_3 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_022 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_6 x_8) → (x_4 x_6 x_7 x_8))))))
    : (∀ (v1 : α) (v2 : α), ((gl_preorder N mul v1 v2) → ((gl_preorder N add one v2) → (gl_preorder N add one v1)))) := by
  intro v1
  intro v2
  intro premise_1
  intro premise_2
  have inductionMember : N v1 := by
    have typingRule := fta_source_041_induction_typing N zero succ add mul one two identity anchor relationalInduction
    exact typingRule v1 v2 premise_1
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorFTA, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v2 : α), ((gl_preorder N mul zero v2) → ((gl_preorder N add one v2) → (gl_preorder N add one zero)))) := by
    intro v2
    intro base_premise_1
    intro base_premise_2
    have zeroRule := fta_source_041_check_zero N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_010 external_peano_externals_36_017 external_peano_externals_36_021
    exact zeroRule zero v2 rfl base_premise_1 base_premise_2
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v2 : α), ((gl_preorder N mul induction_n v2) → ((gl_preorder N add one v2) → (gl_preorder N add one induction_n)))) → ∀ induction_m, succ induction_n induction_m → (∀ (v2 : α), ((gl_preorder N mul induction_m v2) → ((gl_preorder N add one v2) → (gl_preorder N add one induction_m)))) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro v2
    intro step_premise_1
    intro step_premise_2
    have step_induction_assumption_1 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have stepRule := fta_source_041_check_induction_condition N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_015 external_peano_externals_36_022
    exact stepRule induction_n induction_m step_induction_assumption_1
  have inductionProperty : (∀ (v2 : α), ((gl_preorder N mul v1 v2) → ((gl_preorder N add one v2) → (gl_preorder N add one v1)))) := by
    exact relationalInduction N zero succ add mul one two identity anchor
      (fun induction_value => (∀ (v2 : α), ((gl_preorder N mul induction_value v2) → ((gl_preorder N add one v2) → (gl_preorder N add one induction_value)))))
      v1
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v2 premise_1 premise_2

private theorem fta_source_052_induction_typing
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (v2 : α)
    (assumption_1 : (N v2))
    : (N v2) := by
  -- chapter_64_line_1: GL tag task formulation.
  have row_1 : (N v2) := by
    exact assumption_1
  exact row_1

private theorem fta_source_052_check_zero
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_019 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_2 x_8) → (x_4 x_2 x_7 x_8))))))
    (v1 : α)
    (v2 : α)
    (assumption_22 : (v2 = zero))
    (assumption_19 : (N v1))
    : (gl_preorder N add v2 v1) := by
  -- chapter_65_line_22: GL tag recursion.
  have row_22 : (v2 = zero) := by
    exact assumption_22
  -- chapter_65_line_21: GL tag symmetry of equality.
  have row_21 : (zero = v2) := by
    exact Eq.symm row_22
  -- chapter_65_line_19: GL tag task formulation.
  have row_19 : (N v1) := by
    exact assumption_19
  -- chapter_65_line_18: GL tag variable copy.
  have row_18 : (v1 = v1) := by
    rfl
  -- chapter_65_line_20: GL tag equality1.
  have row_20 : (N v1) := by
    have equality_source := row_19
    have equality_step_1 := row_18
    cases equality_step_1
    exact equality_source
  -- chapter_65_line_17: GL tag symmetry of equality.
  have row_17 : (v1 = v1) := by
    exact Eq.symm row_18
  -- chapter_65_line_11: GL tag task formulation.
  have row_11 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_65_line_10: GL tag expansion.
  have row_10 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_11
  -- chapter_65_line_12: GL tag disintegration.
  have row_12 : (succ zero one) := by
    exact row_10.1.1.2
  -- chapter_65_line_9: GL tag disintegration.
  have row_9 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_10.1.1.1
  -- chapter_65_line_16: GL tag expansion.
  have row_16 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_9
  -- chapter_65_line_15: GL tag disintegration.
  have row_15 : (gl_implication16 N zero add) := by
    exact row_16.1.1.1.1.1.1.2
  -- chapter_65_line_14: GL tag expansion.
  have row_14 : (∀ (w1 : α) (w2 : α), ((w1 = w2) → ((N w1) → ((N w2) → (add w1 zero w2))))) := by
    simpa only [gl_implication16] using row_15
  -- chapter_65_line_13: GL tag implication.
  have row_13 : (add v1 zero v1) := by
    apply row_14
    exact row_17
    exact row_20
    exact row_19
  -- chapter_65_line_8: GL tag expansion for integration.
  have row_8 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_65_line_7: GL tag reformulation for integration and.
  have row_7 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_65_line_6: GL tag implication.
  have row_6 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_7
    exact row_9
    exact row_12
  have rule_row_5 := external_peano_externals_36_019 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_65_line_5: GL tag implication.
  have row_5 : (add zero v1 v1) := by
    apply rule_row_5
    exact row_13
  -- chapter_65_line_4: GL tag expansion for integration.
  have row_4 : ((gl_preorder N add zero v1) ↔ (¬ (∀ (v5 : α), ((N v5) → (¬ (add zero v5 v1)))))) := by
    exact Iff.rfl
  -- chapter_65_line_3: GL tag reformulation for integration >[bound].
  have row_3 : (∀ (v3 : α), ((N v3) → ((add zero v3 v1) → (gl_preorder N add zero v1)))) := by
    intro v3
    intro integration_premise_1
    intro integration_premise_2
    apply (row_4).2
    intro universal_counterexample
    exact universal_counterexample v3 integration_premise_1 integration_premise_2
  -- chapter_65_line_2: GL tag implication.
  have row_2 : (gl_preorder N add zero v1) := by
    apply row_3
    exact row_20
    exact row_5
  -- chapter_65_line_1: GL tag equality1.
  have row_1 : (gl_preorder N add v2 v1) := by
    have equality_source := row_2
    have equality_step_1 := row_21
    cases equality_step_1
    exact equality_source
  exact row_1

private theorem fta_source_052_check_induction_condition
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_006 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (x_4 x_8 x_7 x_9))))))
    (external_peano_externals_36_015 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_3 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_034 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (∀ (w1 : α), ((N w1) → (gl_or2 w1 zero N succ))))))
    (external_peano_externals_36_003 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (∀ (x_10 : α), ((x_3 x_8 x_10) → (∀ (x_11 : α), ((x_3 x_11 x_7) → (x_4 x_10 x_11 x_9))))))))))
    (external_peano_externals_36_004 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (∀ (x_10 : α), ((x_4 x_7 x_10 x_9) → (x_8 = x_10))))))))
    (previous : α)
    (v1 : α)
    (v2 : α)
    (assumption_36 : (N v1))
    (assumption_34 : (succ previous v2))
    (assumption_18 : (¬ (gl_preorder N add v1 v2)))
    (assumption_13 : ((N v1) → ((N previous) → ((¬ (gl_preorder N add v1 previous)) → (gl_preorder N add previous v1)))))
    : (gl_preorder N add v2 v1) := by
  -- chapter_66_line_77: GL tag variable copy.
  have row_77 : (v1 = v1) := by
    rfl
  -- chapter_66_line_79: GL tag symmetry of equality.
  have row_79 : (v1 = v1) := by
    exact Eq.symm row_77
  -- chapter_66_line_70: GL tag theorem.
  have row_70 := fta_source_001 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_006 external_peano_externals_36_004
  -- chapter_66_line_41: GL tag theorem.
  have row_41 := fta_source_002 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_66_line_36: GL tag task formulation.
  have row_36 : (N v1) := by
    exact assumption_36
  -- chapter_66_line_78: GL tag equality1.
  have row_78 : (N v1) := by
    have equality_source := row_36
    have equality_step_1 := row_77
    cases equality_step_1
    exact equality_source
  -- chapter_66_line_34: GL tag recursion.
  have row_34 : (succ previous v2) := by
    exact assumption_34
  -- chapter_66_line_29: GL tag task formulation.
  have row_29 : ((gl_preorder N add v1 previous) → (gl_preorder N add v1 previous)) := by
    intro scope_premise_1
    exact scope_premise_1
  -- chapter_66_line_28: GL tag expansion.
  have row_28 : ((gl_preorder N add v1 previous) → (¬ (∀ (v6 : α), ((N v6) → (¬ (add v1 v6 previous)))))) := by
    simpa only [gl_preorder] using row_29
  -- chapter_66_line_35: GL tag disintegration.
  have row_35 : ((gl_preorder N add v1 previous) → (∃ (v6 : α), ((add v1 v6 previous) ∧ (N v6)))) := by
    intro scope_premise_1
    have scoped_fact_1 := row_28 scope_premise_1
    obtain ⟨v6, scoped_witness_bundle⟩ := existsAndOfNotForallImpNot scoped_fact_1
    exact ⟨v6, ⟨scoped_witness_bundle.2, scoped_witness_bundle.1⟩⟩
  -- chapter_66_line_27: GL tag disintegration.
  have row_27 : ((gl_preorder N add v1 previous) → (∃ (v6 : α), ((add v1 v6 previous) ∧ (N v6)))) := by
    intro scope_premise_1
    have scoped_fact_1 := row_28 scope_premise_1
    obtain ⟨v6, scoped_witness_bundle⟩ := existsAndOfNotForallImpNot scoped_fact_1
    exact ⟨v6, ⟨scoped_witness_bundle.2, scoped_witness_bundle.1⟩⟩
  -- chapter_66_line_18: GL tag task formulation.
  have row_18 : (¬ (gl_preorder N add v1 v2)) := by
    exact assumption_18
  -- chapter_66_line_17: GL tag expansion.
  have row_17 : (gl_implication24 N v1 v2 add) := by
    simp only [gl_implication24]
    intro existence_witness_1 compact_premise compact_negated
    have unfolded_row_18 := row_18
    simp only [gl_preorder] at unfolded_row_18
    apply unfolded_row_18
    intro positive_row_18
    exact positive_row_18 existence_witness_1 compact_premise compact_negated
  -- chapter_66_line_16: GL tag expansion.
  have row_16 : (∀ (w1 : α), ((N w1) → (¬ (add v1 w1 v2)))) := by
    simpa only [gl_implication24] using row_17
  -- chapter_66_line_13: GL tag recursion.
  have row_13 : ((N v1) → ((N previous) → ((¬ (gl_preorder N add v1 previous)) → (gl_preorder N add previous v1)))) := by
    exact assumption_13
  -- chapter_66_line_8: GL tag task formulation.
  have row_8 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_66_line_7: GL tag expansion.
  have row_7 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_8
  -- chapter_66_line_9: GL tag disintegration.
  have row_9 : (succ zero one) := by
    exact row_7.1.1.2
  -- chapter_66_line_6: GL tag disintegration.
  have row_6 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_7.1.1.1
  -- chapter_66_line_26: GL tag expansion.
  have row_26 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_6
  -- chapter_66_line_76: GL tag disintegration.
  have row_76 : (gl_implication16 N zero add) := by
    exact row_26.1.1.1.1.1.1.2
  -- chapter_66_line_75: GL tag expansion.
  have row_75 : (∀ (w1 : α) (w2 : α), ((w1 = w2) → ((N w1) → ((N w2) → (add w1 zero w2))))) := by
    simpa only [gl_implication16] using row_76
  -- chapter_66_line_74: GL tag implication.
  have row_74 : (add v1 zero v1) := by
    apply row_75
    exact row_77
    exact row_36
    exact row_78
  -- chapter_66_line_73: GL tag equality1.
  have row_73 : (add v1 zero v1) := by
    have equality_source := row_74
    have equality_step_1 := row_79
    cases equality_step_1
    exact equality_source
  -- chapter_66_line_63: GL tag disintegration.
  have row_63 : (gl_implication15 N zero add) := by
    exact row_26.1.1.1.1.1.1.1.2
  -- chapter_66_line_62: GL tag expansion.
  have row_62 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((add w1 zero w2) → (w1 = w2))))) := by
    simpa only [gl_implication15] using row_63
  -- chapter_66_line_53: GL tag disintegration.
  have row_53 : (N zero) := by
    exact row_26.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_66_line_52: GL tag disintegration.
  have row_52 : (gl_fXYZ add N N N) := by
    exact row_26.1.1.1.1.1.1.1.1.2
  -- chapter_66_line_51: GL tag expansion.
  have row_51 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_52
  -- chapter_66_line_50: GL tag disintegration.
  have row_50 : (gl_implication13 N N N add) := by
    exact row_51.1.2
  -- chapter_66_line_49: GL tag expansion.
  have row_49 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication13] using row_50
  -- chapter_66_line_45: GL tag disintegration.
  have row_45 : (gl_implication17 N succ add) := by
    exact row_26.1.1.1.1.1.2
  -- chapter_66_line_44: GL tag expansion.
  have row_44 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication17] using row_45
  -- chapter_66_line_32: GL tag disintegration.
  have row_32 : (gl_implication18 N succ add) := by
    exact row_26.1.1.1.1.2
  -- chapter_66_line_31: GL tag expansion.
  have row_31 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((succ w4 w5) → (add w3 w2 w5))))))))) := by
    simpa only [gl_implication18] using row_32
  -- chapter_66_line_25: GL tag disintegration.
  have row_25 : (gl_fXY succ N N) := by
    exact row_26.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_66_line_24: GL tag expansion.
  have row_24 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_25
  -- chapter_66_line_39: GL tag disintegration.
  have row_39 : (gl_implication0 succ N) := by
    exact row_24.1.1.1
  -- chapter_66_line_38: GL tag expansion.
  have row_38 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_39
  -- chapter_66_line_37: GL tag implication.
  have row_37 : (N previous) := by
    apply row_38
    exact row_34
  -- chapter_66_line_48: GL tag implication.
  have row_48 : (gl_existence1 N previous zero add) := by
    apply row_49
    exact row_37
    exact row_53
  -- chapter_66_line_47: GL tag expansion.
  have row_47 : (¬ (∀ (v12 : α), ((N v12) → (¬ (add previous zero v12))))) := by
    simpa only [gl_existence1] using row_48
  have exists_row_47 : ∃ (v12 : α), ((N v12) ∧ (add previous zero v12)) := existsAndOfNotForallImpNot row_47
  obtain ⟨v12, witness_row_47⟩ := exists_row_47
  -- chapter_66_line_46: GL tag disintegration.
  have row_46 : (add previous zero v12) := by
    exact witness_row_47.2
  -- chapter_66_line_61: GL tag implication.
  have row_61 : (previous = v12) := by
    apply row_62
    exact row_37
    exact row_46
  -- chapter_66_line_23: GL tag disintegration.
  have row_23 : (gl_implication4 N N succ) := by
    exact row_24.1.2
  -- chapter_66_line_22: GL tag expansion.
  have row_22 : (∀ (w1 : α), ((N w1) → (gl_existence0 N w1 succ))) := by
    simpa only [gl_implication4] using row_23
  -- chapter_66_line_21: GL tag implication.
  have row_21 : ((gl_preorder N add v1 previous) → (∀ (v6 : α), (((add v1 v6 previous) ∧ (N v6)) → (gl_existence0 N v6 succ)))) := by
    intro scope_premise_1
    intro v6
    intro witness_guard_1
    have scoped_fact_2 := row_27 scope_premise_1
    apply row_22
    exact witness_guard_1.2
  -- chapter_66_line_20: GL tag expansion.
  have row_20 : ((gl_preorder N add v1 previous) → (∀ (v6 : α), (((add v1 v6 previous) ∧ (N v6)) → (¬ (∀ (v4 : α), ((N v4) → (¬ (succ v6 v4)))))))) := by
    simpa only [gl_existence0] using row_21
  -- chapter_66_line_33: GL tag disintegration.
  have row_33 : ((gl_preorder N add v1 previous) → (∀ (v6 : α), (((add v1 v6 previous) ∧ (N v6)) → (∃ (v4 : α), ((succ v6 v4) ∧ (N v4)))))) := by
    intro scope_premise_1
    intro v6
    intro witness_guard_1
    have scoped_fact_1 := row_20 scope_premise_1 v6 witness_guard_1
    obtain ⟨v4, scoped_witness_bundle⟩ := existsAndOfNotForallImpNot scoped_fact_1
    exact ⟨v4, ⟨scoped_witness_bundle.2, scoped_witness_bundle.1⟩⟩
  -- chapter_66_line_30: GL tag implication.
  have row_30 : ((gl_preorder N add v1 previous) → (∀ (v6 : α), (((add v1 v6 previous) ∧ (N v6)) → (∀ (v4 : α), (((succ v6 v4) ∧ (N v4)) → (add v1 v4 v2)))))) := by
    intro scope_premise_1
    intro v6
    intro witness_guard_1
    intro v4
    intro witness_guard_2
    have scoped_fact_2 := row_27 scope_premise_1
    have scoped_fact_3 := row_33 scope_premise_1 v6 witness_guard_1
    have scoped_fact_4 := row_35 scope_premise_1
    apply row_31
    exact witness_guard_1.2
    exact witness_guard_2.1
    exact witness_guard_1.1
    exact row_34
  -- chapter_66_line_19: GL tag disintegration.
  have row_19 : ((gl_preorder N add v1 previous) → (∀ (v6 : α), (((add v1 v6 previous) ∧ (N v6)) → (∃ (v4 : α), ((succ v6 v4) ∧ (N v4)))))) := by
    intro scope_premise_1
    intro v6
    intro witness_guard_1
    have scoped_fact_1 := row_20 scope_premise_1 v6 witness_guard_1
    obtain ⟨v4, scoped_witness_bundle⟩ := existsAndOfNotForallImpNot scoped_fact_1
    exact ⟨v4, ⟨scoped_witness_bundle.2, scoped_witness_bundle.1⟩⟩
  -- chapter_66_line_15: GL tag implication.
  have row_15 : ((gl_preorder N add v1 previous) → (∀ (v6 : α), (((add v1 v6 previous) ∧ (N v6)) → (∀ (v4 : α), (((succ v6 v4) ∧ (N v4)) → (¬ (add v1 v4 v2))))))) := by
    intro scope_premise_1
    intro v6
    intro witness_guard_1
    intro v4
    intro witness_guard_2
    have scoped_fact_2 := row_19 scope_premise_1 v6 witness_guard_1
    apply row_16
    exact witness_guard_2.2
  -- chapter_66_line_14: GL tag contradiction.
  have row_14 : (¬ (gl_preorder N add v1 previous)) := by
    intro contradiction_assumption
    obtain ⟨v6, contradiction_witness_guard_1⟩ := row_35 contradiction_assumption
    obtain ⟨v4, contradiction_witness_guard_2⟩ := row_33 contradiction_assumption v6 contradiction_witness_guard_1
    have scoped_contradiction := row_15 contradiction_assumption v6 contradiction_witness_guard_1 v4 contradiction_witness_guard_2
    have scoped_contradiction_2 := row_30 contradiction_assumption v6 contradiction_witness_guard_1 v4 contradiction_witness_guard_2
    exact scoped_contradiction scoped_contradiction_2
  -- chapter_66_line_68: GL tag expansion.
  have row_68 : (gl_implication24 N v1 previous add) := by
    simp only [gl_implication24]
    intro existence_witness_1 compact_premise compact_negated
    have unfolded_row_14 := row_14
    simp only [gl_preorder] at unfolded_row_14
    apply unfolded_row_14
    intro positive_row_14
    exact positive_row_14 existence_witness_1 compact_premise compact_negated
  -- chapter_66_line_67: GL tag expansion.
  have row_67 : (∀ (w1 : α), ((N w1) → (¬ (add v1 w1 previous)))) := by
    simpa only [gl_implication24] using row_68
  -- chapter_66_line_66: GL tag implication.
  have row_66 : (¬ (add v1 zero previous)) := by
    apply row_67
    exact row_53
  -- chapter_66_line_12: GL tag implication.
  have row_12 : (gl_preorder N add previous v1) := by
    apply row_13
    exact row_36
    exact row_37
    exact row_14
  -- chapter_66_line_11: GL tag expansion.
  have row_11 : (¬ (∀ (v3 : α), ((N v3) → (¬ (add previous v3 v1))))) := by
    simpa only [gl_preorder] using row_12
  have exists_row_11 : ∃ (v3 : α), ((N v3) ∧ (add previous v3 v1)) := existsAndOfNotForallImpNot row_11
  obtain ⟨v3, witness_row_11⟩ := exists_row_11
  -- chapter_66_line_60: GL tag disintegration.
  have row_60 : (add previous v3 v1) := by
    exact witness_row_11.2
  -- chapter_66_line_82: GL tag equality1.
  have row_82 : (add previous v3 v1) := by
    have equality_source := row_60
    have equality_step_1 := row_77
    cases equality_step_1
    exact equality_source
  -- chapter_66_line_10: GL tag disintegration.
  have row_10 : (N v3) := by
    exact witness_row_11.1
  -- chapter_66_line_5: GL tag expansion for integration.
  have row_5 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_66_line_4: GL tag reformulation for integration and.
  have row_4 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_66_line_3: GL tag implication.
  have row_3 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_4
    exact row_6
    exact row_9
  have rule_row_59 := external_peano_externals_36_006 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_66_line_59: GL tag implication.
  have row_59 : (add v3 previous v1) := by
    apply rule_row_59
    exact row_60
  -- chapter_66_line_58: GL tag equality1.
  have row_58 : (add v3 v12 v1) := by
    have equality_source := row_59
    have equality_step_1 := row_61
    cases equality_step_1
    exact equality_source
  have rule_row_54 := external_peano_externals_36_015 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_66_line_54: GL tag implication.
  have row_54 : (add previous one v2) := by
    apply rule_row_54
    exact row_34
  -- chapter_66_line_43: GL tag implication.
  have row_43 : (succ v12 v2) := by
    apply row_44
    exact row_53
    exact row_9
    exact row_46
    exact row_54
  have rule_row_2 := external_peano_externals_36_034 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_66_line_2: GL tag implication.
  have row_2 : (gl_or2 v3 zero N succ) := by
    apply rule_row_2
    exact row_10
  -- chapter_66_line_81: GL tag or disintegration.
  have row_81 : ((v3 = zero) → (v3 = zero)) := by
    intro scope_premise_1
    exact scope_premise_1
  -- chapter_66_line_80: GL tag symmetry of equality.
  have row_80 : ((v3 = zero) → (zero = v3)) := by
    intro scope_premise_1
    have scoped_fact_1 := row_81 scope_premise_1
    exact Eq.symm scoped_fact_1
  -- chapter_66_line_72: GL tag equality1.
  have row_72 : ((v3 = zero) → (add v1 v3 v1)) := by
    intro scope_premise_1
    have scoped_fact_2 := row_80 scope_premise_1
    have equality_source := row_73
    have equality_step_1 := scoped_fact_2
    cases equality_step_1
    exact equality_source
  -- chapter_66_line_71: GL tag equality1.
  have row_71 : ((v3 = zero) → (add v1 v3 v1)) := by
    intro scope_premise_1
    have scoped_fact_1 := row_72 scope_premise_1
    have equality_source := scoped_fact_1
    have equality_step_1 := row_77
    cases equality_step_1
    exact equality_source
  -- chapter_66_line_69: GL tag implication.
  have row_69 : ((v3 = zero) → (previous = v1)) := by
    intro scope_premise_1
    have scoped_fact_3 := row_71 scope_premise_1
    apply row_70
    exact row_82
    exact scoped_fact_3
  -- chapter_66_line_65: GL tag equality1.
  have row_65 : ((v3 = zero) → (¬ (add v1 zero v1))) := by
    intro scope_premise_1
    have scoped_fact_2 := row_69 scope_premise_1
    have equality_source := row_66
    have equality_step_1 := scoped_fact_2
    cases equality_step_1
    exact equality_source
  -- chapter_66_line_64: GL tag contradiction.
  have row_64 : (¬ (v3 = zero)) := by
    intro contradiction_assumption
    have scoped_contradiction := row_65 contradiction_assumption
    exact scoped_contradiction row_73
  -- chapter_66_line_57: GL tag or disintegration.
  have row_57 : ((gl_existence11 N v3 succ) → (gl_existence11 N v3 succ)) := by
    intro scope_premise_1
    exact scope_premise_1
  -- chapter_66_line_56: GL tag expansion.
  have row_56 : ((gl_existence11 N v3 succ) → (¬ (∀ (v11 : α), ((N v11) → (¬ (succ v11 v3)))))) := by
    simpa only [gl_existence11] using row_57
  -- chapter_66_line_55: GL tag disintegration.
  have row_55 : ((gl_existence11 N v3 succ) → (∃ (v11 : α), (succ v11 v3))) := by
    intro scope_premise_1
    have scoped_fact_1 := row_56 scope_premise_1
    obtain ⟨v11, scoped_witness_bundle⟩ := existsAndOfNotForallImpNot scoped_fact_1
    exact ⟨v11, scoped_witness_bundle.2⟩
  have rule_row_42 := external_peano_externals_36_003 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_66_line_42: GL tag implication.
  have row_42 : ((gl_existence11 N v3 succ) → (∀ (v11 : α), ((succ v11 v3) → (add v2 v11 v1)))) := by
    intro scope_premise_1
    intro v11
    intro witness_guard_1
    have scoped_fact_4 := row_55 scope_premise_1
    apply rule_row_42
    exact row_58
    exact row_43
    exact witness_guard_1
  -- chapter_66_line_40: GL tag implication.
  have row_40 : ((gl_existence11 N v3 succ) → (∀ (v11 : α), ((succ v11 v3) → (gl_preorder N add v2 v1)))) := by
    intro scope_premise_1
    intro v11
    intro witness_guard_1
    have scoped_fact_2 := row_42 scope_premise_1 v11 witness_guard_1
    apply row_41
    exact scoped_fact_2
  -- chapter_66_line_1: GL tag or convergence.
  have row_1 : (gl_preorder N add v2 v1) := by
    classical
    have or_cases := row_2
    simp only [gl_or2] at or_cases
    rcases or_cases with or_branch_1 | or_branch_2
    · have branch_compact := or_branch_1
      exact False.elim (row_64 branch_compact)
    · have branch_compact := or_branch_2
      obtain ⟨v11, or_witness_bundle_2⟩ := existsAndOfNotForallImpNot branch_compact
      exact row_40 branch_compact v11 or_witness_bundle_2.2
  exact row_1

theorem fta_source_052
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_019 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_2 x_8) → (x_4 x_2 x_7 x_8))))))
    (external_peano_externals_36_006 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (x_4 x_8 x_7 x_9))))))
    (external_peano_externals_36_015 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_3 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_034 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (∀ (w1 : α), ((N w1) → (gl_or2 w1 zero N succ))))))
    (external_peano_externals_36_003 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (∀ (x_10 : α), ((x_3 x_8 x_10) → (∀ (x_11 : α), ((x_3 x_11 x_7) → (x_4 x_10 x_11 x_9))))))))))
    (external_peano_externals_36_004 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (∀ (x_10 : α), ((x_4 x_7 x_10 x_9) → (x_8 = x_10))))))))
    : (∀ (v1 : α), ((N v1) → (∀ (v2 : α), ((N v2) → ((¬ (gl_preorder N add v1 v2)) → (gl_preorder N add v2 v1)))))) := by
  intro v1
  intro premise_1
  intro v2
  intro premise_2
  intro premise_3
  have inductionMember : N v2 := by
    have typingRule := fta_source_052_induction_typing N zero succ add mul one two identity anchor relationalInduction
    exact typingRule v2 premise_2
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorFTA, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v1 : α), ((N v1) → ((N zero) → ((¬ (gl_preorder N add v1 zero)) → (gl_preorder N add zero v1))))) := by
    intro v1
    intro base_premise_1
    intro base_premise_2
    intro base_premise_3
    have zeroRule := fta_source_052_check_zero N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_019
    exact zeroRule v1 zero rfl base_premise_1
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v1 : α), ((N v1) → ((N induction_n) → ((¬ (gl_preorder N add v1 induction_n)) → (gl_preorder N add induction_n v1))))) → ∀ induction_m, succ induction_n induction_m → (∀ (v1 : α), ((N v1) → ((N induction_m) → ((¬ (gl_preorder N add v1 induction_m)) → (gl_preorder N add induction_m v1))))) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro v1
    intro step_premise_1
    intro step_premise_2
    intro step_premise_3
    have step_induction_assumption_1 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have step_induction_assumption_2 :
        ((N v1) → ((N induction_n) → ((¬ (gl_preorder N add v1 induction_n)) → (gl_preorder N add induction_n v1)))) := by
      intro step_induction_assumption_2_premise_1
      intro step_induction_assumption_2_premise_2
      intro step_induction_assumption_2_premise_3
      apply induction_hypothesis
      all_goals assumption
    have stepRule := fta_source_052_check_induction_condition N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_006 external_peano_externals_36_015 external_peano_externals_36_034 external_peano_externals_36_003 external_peano_externals_36_004
    exact stepRule induction_n v1 induction_m step_premise_1 step_induction_assumption_1 step_premise_3 step_induction_assumption_2
  have inductionProperty : (∀ (v1 : α), ((N v1) → ((N v2) → ((¬ (gl_preorder N add v1 v2)) → (gl_preorder N add v2 v1))))) := by
    exact relationalInduction N zero succ add mul one two identity anchor
      (fun induction_value => (∀ (v1 : α), ((N v1) → ((N induction_value) → ((¬ (gl_preorder N add v1 induction_value)) → (gl_preorder N add induction_value v1))))))
      v2
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v1 premise_1 premise_2 premise_3

private theorem fta_source_057_induction_typing
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (v1 : α)
    (assumption_1 : (N v1))
    : (N v1) := by
  -- chapter_71_line_1: GL tag task formulation.
  have row_1 : (N v1) := by
    exact assumption_1
  exact row_1

private theorem fta_source_057_check_zero
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (v1 : α)
    (assumption_4 : (¬ (zero = v1)))
    (assumption_2 : (v1 = zero))
    : (gl_preorder N add two v1) := by
  -- chapter_72_line_4: GL tag task formulation.
  have row_4 : (¬ (zero = v1)) := by
    exact assumption_4
  -- chapter_72_line_3: GL tag symmetry of inequality.
  have row_3 : (¬ (v1 = zero)) := by
    exact fun equality => row_4 (Eq.symm equality)
  -- chapter_72_line_2: GL tag recursion.
  have row_2 : (v1 = zero) := by
    exact assumption_2
  -- chapter_72_line_1: GL tag vacuous truth.
  have row_1 : (gl_preorder N add two v1) := by
    exact False.elim (row_3 row_2)
  exact row_1

private theorem fta_source_057_check_induction_condition
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_034 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (∀ (w1 : α), ((N w1) → (gl_or2 w1 zero N succ))))))
    (external_peano_externals_36_015 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_3 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_022 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_6 x_8) → (x_4 x_6 x_7 x_8))))))
    (external_peano_externals_36_003 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (∀ (x_10 : α), ((x_3 x_8 x_10) → (∀ (x_11 : α), ((x_3 x_11 x_7) → (x_4 x_10 x_11 x_9))))))))))
    (external_gauss_externals_24_018 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α) (x_7 : α) (x_8 : GLBinaryRelation α), ((gl_AnchorGauss x_1 x_2 x_3 x_4 x_5 x_6 x_7 x_8) → (∀ (x_9 : α) (x_10 : α), ((x_3 x_9 x_10) → (gl_preorder x_1 x_4 x_9 x_10))))))
    (external_peano_externals_36_035 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (gl_existence11 N one succ))))
    (external_peano_externals_36_029 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((gl_preorder x_1 x_4 x_7 x_8) → ((gl_preorder x_1 x_4 x_8 x_7) → (x_7 = x_8)))))))
    (external_peano_externals_36_005 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → ((x_8 = x_9) → (x_2 = x_7)))))))
    (previous : α)
    (v1 : α)
    (assumption_65 : (¬ (zero = v1)))
    (assumption_63 : (N v1))
    (assumption_49 : (¬ (one = v1)))
    (assumption_36 : (succ previous v1))
    : (gl_preorder N add two v1) := by
  -- chapter_73_line_65: GL tag task formulation.
  have row_65 : (¬ (zero = v1)) := by
    exact assumption_65
  -- chapter_73_line_64: GL tag symmetry of inequality.
  have row_64 : (¬ (v1 = zero)) := by
    exact fun equality => row_65 (Eq.symm equality)
  -- chapter_73_line_63: GL tag task formulation.
  have row_63 : (N v1) := by
    exact assumption_63
  -- chapter_73_line_51: GL tag theorem.
  have row_51 := fta_source_002 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_73_line_49: GL tag task formulation.
  have row_49 : (¬ (one = v1)) := by
    exact assumption_49
  -- chapter_73_line_48: GL tag theorem.
  have row_48 := fta_source_031 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_73_line_44: GL tag theorem.
  have row_44 := fta_source_047 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_73_line_42: GL tag theorem.
  have row_42 := fta_source_059 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_015 external_peano_externals_36_022
  -- chapter_73_line_39: GL tag theorem.
  have row_39 := fta_source_014 N zero succ add mul one two identity anchor relationalInduction external_gauss_externals_24_018 external_peano_externals_36_035 external_peano_externals_36_029 external_peano_externals_36_003 external_peano_externals_36_005
  -- chapter_73_line_36: GL tag recursion.
  have row_36 : (succ previous v1) := by
    exact assumption_36
  -- chapter_73_line_10: GL tag task formulation.
  have row_10 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_73_line_52: GL tag anchor handling.
  have row_52 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact row_10
  -- chapter_73_line_40: GL tag anchor handling.
  have row_40 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact row_10
  -- chapter_73_line_9: GL tag expansion.
  have row_9 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_10
  -- chapter_73_line_13: GL tag disintegration.
  have row_13 : (succ one two) := by
    exact row_9.1.2
  -- chapter_73_line_11: GL tag disintegration.
  have row_11 : (succ zero one) := by
    exact row_9.1.1.2
  -- chapter_73_line_8: GL tag disintegration.
  have row_8 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_9.1.1.1
  -- chapter_73_line_19: GL tag expansion.
  have row_19 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_8
  -- chapter_73_line_81: GL tag disintegration.
  have row_81 : (gl_implication7 N succ) := by
    exact row_19.1.1.1.1.1.1.1.1.1.2
  -- chapter_73_line_80: GL tag expansion.
  have row_80 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w2 w1) → (∀ (w3 : α), ((succ w3 w1) → (w2 = w3))))))) := by
    simpa only [gl_implication7] using row_81
  -- chapter_73_line_78: GL tag disintegration.
  have row_78 : (gl_implication15 N zero add) := by
    exact row_19.1.1.1.1.1.1.1.2
  -- chapter_73_line_77: GL tag expansion.
  have row_77 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((add w1 zero w2) → (w1 = w2))))) := by
    simpa only [gl_implication15] using row_78
  -- chapter_73_line_73: GL tag disintegration.
  have row_73 : (gl_fXYZ add N N N) := by
    exact row_19.1.1.1.1.1.1.1.1.2
  -- chapter_73_line_72: GL tag expansion.
  have row_72 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_73
  -- chapter_73_line_71: GL tag disintegration.
  have row_71 : (gl_implication13 N N N add) := by
    exact row_72.1.2
  -- chapter_73_line_70: GL tag expansion.
  have row_70 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication13] using row_71
  -- chapter_73_line_55: GL tag disintegration.
  have row_55 : (gl_implication18 N succ add) := by
    exact row_19.1.1.1.1.2
  -- chapter_73_line_54: GL tag expansion.
  have row_54 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((succ w4 w5) → (add w3 w2 w5))))))))) := by
    simpa only [gl_implication18] using row_55
  -- chapter_73_line_25: GL tag disintegration.
  have row_25 : (N zero) := by
    exact row_19.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_73_line_18: GL tag disintegration.
  have row_18 : (gl_fXY succ N N) := by
    exact row_19.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_73_line_17: GL tag expansion.
  have row_17 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_18
  -- chapter_73_line_35: GL tag disintegration.
  have row_35 : (gl_implication0 succ N) := by
    exact row_17.1.1.1
  -- chapter_73_line_34: GL tag expansion.
  have row_34 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_35
  -- chapter_73_line_33: GL tag implication.
  have row_33 : (N previous) := by
    apply row_34
    exact row_36
  -- chapter_73_line_69: GL tag implication.
  have row_69 : (gl_existence1 N previous zero add) := by
    apply row_70
    exact row_33
    exact row_25
  -- chapter_73_line_68: GL tag expansion.
  have row_68 : (¬ (∀ (v12 : α), ((N v12) → (¬ (add previous zero v12))))) := by
    simpa only [gl_existence1] using row_69
  have exists_row_68 : ∃ (v12 : α), ((N v12) ∧ (add previous zero v12)) := existsAndOfNotForallImpNot row_68
  obtain ⟨v12, witness_row_68⟩ := exists_row_68
  -- chapter_73_line_67: GL tag disintegration.
  have row_67 : (add previous zero v12) := by
    exact witness_row_68.2
  -- chapter_73_line_76: GL tag implication.
  have row_76 : (previous = v12) := by
    apply row_77
    exact row_33
    exact row_67
  -- chapter_73_line_75: GL tag symmetry of equality.
  have row_75 : (v12 = previous) := by
    exact Eq.symm row_76
  -- chapter_73_line_24: GL tag disintegration.
  have row_24 : (gl_implication4 N N succ) := by
    exact row_17.1.2
  -- chapter_73_line_23: GL tag expansion.
  have row_23 : (∀ (w1 : α), ((N w1) → (gl_existence0 N w1 succ))) := by
    simpa only [gl_implication4] using row_24
  -- chapter_73_line_22: GL tag implication.
  have row_22 : (gl_existence0 N zero succ) := by
    apply row_23
    exact row_25
  -- chapter_73_line_21: GL tag expansion.
  have row_21 : (¬ (∀ (v5 : α), ((N v5) → (¬ (succ zero v5))))) := by
    simpa only [gl_existence0] using row_22
  have exists_row_21 : ∃ (v5 : α), ((N v5) ∧ (succ zero v5)) := existsAndOfNotForallImpNot row_21
  obtain ⟨v5, witness_row_21⟩ := exists_row_21
  -- chapter_73_line_20: GL tag disintegration.
  have row_20 : (succ zero v5) := by
    exact witness_row_21.2
  -- chapter_73_line_16: GL tag disintegration.
  have row_16 : (gl_implication5 N succ) := by
    exact row_17.2
  -- chapter_73_line_15: GL tag expansion.
  have row_15 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α), ((succ w1 w3) → (w2 = w3))))))) := by
    simpa only [gl_implication5] using row_16
  -- chapter_73_line_14: GL tag implication.
  have row_14 : (one = v5) := by
    apply row_15
    exact row_25
    exact row_11
    exact row_20
  -- chapter_73_line_12: GL tag equality1.
  have row_12 : (succ v5 two) := by
    have equality_source := row_13
    have equality_step_1 := row_14
    cases equality_step_1
    exact equality_source
  -- chapter_73_line_7: GL tag expansion for integration.
  have row_7 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_73_line_6: GL tag reformulation for integration and.
  have row_6 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_73_line_5: GL tag implication.
  have row_5 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_6
    exact row_8
    exact row_11
  have rule_row_62 := external_peano_externals_36_034 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_73_line_62: GL tag implication.
  have row_62 : (gl_or2 v1 zero N succ) := by
    apply rule_row_62
    exact row_63
  -- chapter_73_line_61: GL tag expansion.
  have row_61 : (¬ ((¬ (v1 = zero)) ∧ (¬ (gl_existence11 N v1 succ)))) := by
    simpa only [gl_or2, GLExport.orIffNotAndNot] using row_62
  -- chapter_73_line_60: GL tag disintegration.
  have row_60 : (gl_implication74 v1 zero N succ) := by
    simp only [gl_implication74]
    classical
    intro projection_premise
    apply Classical.byContradiction
    intro projection_counterexample
    exact row_61 ⟨projection_premise, projection_counterexample⟩
  -- chapter_73_line_59: GL tag expansion.
  have row_59 : ((¬ (v1 = zero)) → (gl_existence11 N v1 succ)) := by
    simpa only [gl_implication74] using row_60
  -- chapter_73_line_58: GL tag implication.
  have row_58 : (gl_existence11 N v1 succ) := by
    apply row_59
    exact row_64
  -- chapter_73_line_57: GL tag expansion.
  have row_57 : (¬ (∀ (v11 : α), ((N v11) → (¬ (succ v11 v1))))) := by
    simpa only [gl_existence11] using row_58
  have exists_row_57 : ∃ (v11 : α), ((N v11) ∧ (succ v11 v1)) := existsAndOfNotForallImpNot row_57
  obtain ⟨v11, witness_row_57⟩ := exists_row_57
  -- chapter_73_line_56: GL tag disintegration.
  have row_56 : (succ v11 v1) := by
    exact witness_row_57.2
  -- chapter_73_line_79: GL tag implication.
  have row_79 : (previous = v11) := by
    apply row_80
    exact row_63
    exact row_36
    exact row_56
  -- chapter_73_line_74: GL tag equality2.
  have row_74 : (v12 = v11) := by
    exact Eq.trans row_75 row_79
  -- chapter_73_line_66: GL tag equality1.
  have row_66 : (add previous zero v11) := by
    have equality_source := row_67
    have equality_step_1 := row_74
    cases equality_step_1
    exact equality_source
  -- chapter_73_line_53: GL tag implication.
  have row_53 : (add previous v5 v1) := by
    apply row_54
    exact row_25
    exact row_20
    exact row_66
    exact row_56
  have rule_row_46 := external_peano_externals_36_015 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_73_line_46: GL tag implication.
  have row_46 : (add previous one v1) := by
    apply rule_row_46
    exact row_36
  have rule_row_45 := external_peano_externals_36_022 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_73_line_45: GL tag implication.
  have row_45 : (add one previous v1) := by
    apply rule_row_45
    exact row_46
  -- chapter_73_line_50: GL tag implication.
  have row_50 : (gl_preorder N add one v1) := by
    apply row_51
    exact row_45
  -- chapter_73_line_47: GL tag implication.
  have row_47 : (gl_strictOrder N add one v1) := by
    apply row_48
    exact row_50
    exact row_49
  -- chapter_73_line_43: GL tag implication.
  have row_43 : (¬ (zero = previous)) := by
    apply row_44
    exact row_47
    exact row_45
  -- chapter_73_line_41: GL tag implication.
  have row_41 : (gl_preorder N add one previous) := by
    apply row_42
    exact row_33
    exact row_43
  -- chapter_73_line_38: GL tag implication.
  have row_38 : (¬ (zero = previous)) := by
    apply row_39
    exact row_11
    exact row_41
  -- chapter_73_line_37: GL tag symmetry of inequality.
  have row_37 : (¬ (previous = zero)) := by
    exact fun equality => row_38 (Eq.symm equality)
  have rule_row_32 := external_peano_externals_36_034 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_73_line_32: GL tag implication.
  have row_32 : (gl_or2 previous zero N succ) := by
    apply rule_row_32
    exact row_33
  -- chapter_73_line_31: GL tag expansion.
  have row_31 : (¬ ((¬ (previous = zero)) ∧ (¬ (gl_existence11 N previous succ)))) := by
    simpa only [gl_or2, GLExport.orIffNotAndNot] using row_32
  -- chapter_73_line_30: GL tag disintegration.
  have row_30 : (gl_implication74 previous zero N succ) := by
    simp only [gl_implication74]
    classical
    intro projection_premise
    apply Classical.byContradiction
    intro projection_counterexample
    exact row_31 ⟨projection_premise, projection_counterexample⟩
  -- chapter_73_line_29: GL tag expansion.
  have row_29 : ((¬ (previous = zero)) → (gl_existence11 N previous succ)) := by
    simpa only [gl_implication74] using row_30
  -- chapter_73_line_28: GL tag implication.
  have row_28 : (gl_existence11 N previous succ) := by
    apply row_29
    exact row_37
  -- chapter_73_line_27: GL tag expansion.
  have row_27 : (¬ (∀ (v3 : α), ((N v3) → (¬ (succ v3 previous))))) := by
    simpa only [gl_existence11] using row_28
  have exists_row_27 : ∃ (v3 : α), ((N v3) ∧ (succ v3 previous)) := existsAndOfNotForallImpNot row_27
  obtain ⟨v3, witness_row_27⟩ := exists_row_27
  -- chapter_73_line_82: GL tag disintegration.
  have row_82 : (N v3) := by
    exact witness_row_27.1
  -- chapter_73_line_26: GL tag disintegration.
  have row_26 : (succ v3 previous) := by
    exact witness_row_27.2
  have rule_row_4 := external_peano_externals_36_003 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_73_line_4: GL tag implication.
  have row_4 : (add two v3 v1) := by
    apply rule_row_4
    exact row_53
    exact row_12
    exact row_26
  -- chapter_73_line_3: GL tag expansion for integration.
  have row_3 : ((gl_preorder N add two v1) ↔ (¬ (∀ (v4 : α), ((N v4) → (¬ (add two v4 v1)))))) := by
    exact Iff.rfl
  -- chapter_73_line_2: GL tag reformulation for integration >[bound].
  have row_2 : (∀ (v2 : α), ((N v2) → ((add two v2 v1) → (gl_preorder N add two v1)))) := by
    intro v2
    intro integration_premise_1
    intro integration_premise_2
    apply (row_3).2
    intro universal_counterexample
    exact universal_counterexample v2 integration_premise_1 integration_premise_2
  -- chapter_73_line_1: GL tag implication.
  have row_1 : (gl_preorder N add two v1) := by
    apply row_2
    exact row_82
    exact row_4
  exact row_1

theorem fta_source_057
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_034 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (∀ (w1 : α), ((N w1) → (gl_or2 w1 zero N succ))))))
    (external_peano_externals_36_015 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_3 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_022 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_6 x_8) → (x_4 x_6 x_7 x_8))))))
    (external_peano_externals_36_003 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (∀ (x_10 : α), ((x_3 x_8 x_10) → (∀ (x_11 : α), ((x_3 x_11 x_7) → (x_4 x_10 x_11 x_9))))))))))
    (external_gauss_externals_24_018 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α) (x_7 : α) (x_8 : GLBinaryRelation α), ((gl_AnchorGauss x_1 x_2 x_3 x_4 x_5 x_6 x_7 x_8) → (∀ (x_9 : α) (x_10 : α), ((x_3 x_9 x_10) → (gl_preorder x_1 x_4 x_9 x_10))))))
    (external_peano_externals_36_035 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (gl_existence11 N one succ))))
    (external_peano_externals_36_029 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((gl_preorder x_1 x_4 x_7 x_8) → ((gl_preorder x_1 x_4 x_8 x_7) → (x_7 = x_8)))))))
    (external_peano_externals_36_005 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → ((x_8 = x_9) → (x_2 = x_7)))))))
    : (∀ (v1 : α), ((N v1) → ((¬ (zero = v1)) → ((¬ (one = v1)) → (gl_preorder N add two v1))))) := by
  intro v1
  intro premise_1
  intro premise_2
  intro premise_3
  have inductionMember : N v1 := by
    have typingRule := fta_source_057_induction_typing N zero succ add mul one two identity anchor relationalInduction
    exact typingRule v1 premise_1
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorFTA, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : ((¬ (zero = zero)) → ((¬ (one = zero)) → (gl_preorder N add two zero))) := by
    intro base_premise_1
    intro base_premise_2
    have zeroRule := fta_source_057_check_zero N zero succ add mul one two identity anchor relationalInduction
    exact zeroRule zero base_premise_1 rfl
  have inductionStep :
      ∀ induction_n, N induction_n → ((¬ (zero = induction_n)) → ((¬ (one = induction_n)) → (gl_preorder N add two induction_n))) → ∀ induction_m, succ induction_n induction_m → ((¬ (zero = induction_m)) → ((¬ (one = induction_m)) → (gl_preorder N add two induction_m))) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    have natural_numbers_step : gl_NaturalNumbers N zero succ add mul := anchor.1.1.1
    have successor_structure : gl_fXY succ N N := natural_numbers_step.1.1.1.1.1.1.1.1.1.1.1.2
    have successor_output_closed : gl_implication1 succ N := successor_structure.1.1.2
    have induction_m_member : N induction_m := successor_output_closed induction_n induction_m induction_successor
    intro step_premise_1
    intro step_premise_2
    have step_induction_assumption_1 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have stepRule := fta_source_057_check_induction_condition N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_034 external_peano_externals_36_015 external_peano_externals_36_022 external_peano_externals_36_003 external_gauss_externals_24_018 external_peano_externals_36_035 external_peano_externals_36_029 external_peano_externals_36_005
    exact stepRule induction_n induction_m step_premise_1 induction_m_member step_premise_2 step_induction_assumption_1
  have inductionProperty : ((¬ (zero = v1)) → ((¬ (one = v1)) → (gl_preorder N add two v1))) := by
    exact relationalInduction N zero succ add mul one two identity anchor
      (fun induction_value => ((¬ (zero = induction_value)) → ((¬ (one = induction_value)) → (gl_preorder N add two induction_value))))
      v1
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty premise_2 premise_3

theorem fta_source_061
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_015 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_3 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_022 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_6 x_8) → (x_4 x_6 x_7 x_8))))))
    : (∀ (v1 : α), ((N v1) → (gl_or3 N add one v1 zero))) := by
  intro v1
  intro premise_1
  have or_parent_1 := fta_source_060 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_015 external_peano_externals_36_022
  have or_parent_2 := fta_source_059 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_015 external_peano_externals_36_022
  -- chapter_81_line_1: GL tag or theorem.
  have row_1 : (∀ (v1 : α), ((N v1) → (gl_or3 N add one v1 zero))) := by
    classical
    intro v1
    intro or_parent_premise_1
    simp only [gl_or3]
    by_cases or_case_1 : (gl_preorder N add one v1)
    · exact Or.inl (or_case_1)
    ·
      exact Or.inr ((or_parent_1 v1 or_parent_premise_1 or_case_1))
  solve_by_elim [row_1]

private theorem fta_source_063_induction_typing
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (v1 : α)
    (assumption_1 : (N v1))
    : (N v1) := by
  -- chapter_83_line_1: GL tag task formulation.
  have row_1 : (N v1) := by
    exact assumption_1
  exact row_1

private theorem fta_source_063_check_zero
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (v1 : α)
    (assumption_10 : (v1 = zero))
    (assumption_8 : (N v1))
    : (gl_preorder N add zero v1) := by
  -- chapter_84_line_10: GL tag recursion.
  have row_10 : (v1 = zero) := by
    exact assumption_10
  -- chapter_84_line_11: GL tag symmetry of equality.
  have row_11 : (zero = v1) := by
    exact Eq.symm row_10
  -- chapter_84_line_8: GL tag task formulation.
  have row_8 : (N v1) := by
    exact assumption_8
  -- chapter_84_line_7: GL tag task formulation.
  have row_7 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_84_line_6: GL tag variable copy.
  have row_6 : (v1 = v1) := by
    rfl
  -- chapter_84_line_9: GL tag symmetry of equality.
  have row_9 : (v1 = v1) := by
    exact Eq.symm row_6
  -- chapter_84_line_5: GL tag theorem.
  have row_5 := fta_source_050 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_84_line_4: GL tag implication.
  have row_4 : (gl_preorder N add v1 v1) := by
    apply row_5
    exact row_8
    exact row_6
  -- chapter_84_line_3: GL tag equality1.
  have row_3 : (gl_preorder N add v1 v1) := by
    have equality_source := row_4
    have equality_step_1 := row_9
    cases equality_step_1
    exact equality_source
  -- chapter_84_line_2: GL tag equality1.
  have row_2 : (gl_preorder N add zero zero) := by
    have equality_source := row_3
    have equality_step_1 := row_10
    cases equality_step_1
    exact equality_source
  -- chapter_84_line_1: GL tag equality1.
  have row_1 : (gl_preorder N add zero v1) := by
    have equality_source := row_2
    have equality_step_1 := row_11
    cases equality_step_1
    exact equality_source
  exact row_1

private theorem fta_source_063_check_induction_condition
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_035 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (gl_existence11 N one succ))))
    (external_gauss_externals_24_018 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α) (x_7 : α) (x_8 : GLBinaryRelation α), ((gl_AnchorGauss x_1 x_2 x_3 x_4 x_5 x_6 x_7 x_8) → (∀ (x_9 : α) (x_10 : α), ((x_3 x_9 x_10) → (gl_preorder x_1 x_4 x_9 x_10))))))
    (external_gauss_externals_24_021 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α) (x_7 : α) (x_8 : GLBinaryRelation α), ((gl_AnchorGauss x_1 x_2 x_3 x_4 x_5 x_6 x_7 x_8) → (∀ (x_9 : α) (x_10 : α), ((gl_preorder x_1 x_4 x_9 x_10) → (∀ (x_11 : α), ((gl_preorder x_1 x_4 x_10 x_11) → (gl_preorder x_1 x_4 x_9 x_11))))))))
    (previous : α)
    (v1 : α)
    (assumption_21 : (succ previous v1))
    (assumption_14 : ((N previous) → (gl_preorder N add zero previous)))
    : (gl_preorder N add zero v1) := by
  -- chapter_85_line_30: GL tag expansion for integration.
  have row_30 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_85_line_29: GL tag reformulation for integration and.
  have row_29 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_85_line_21: GL tag recursion.
  have row_21 : (succ previous v1) := by
    exact assumption_21
  -- chapter_85_line_14: GL tag recursion.
  have row_14 : ((N previous) → (gl_preorder N add zero previous)) := by
    exact assumption_14
  -- chapter_85_line_8: GL tag task formulation.
  have row_8 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_85_line_7: GL tag expansion.
  have row_7 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_8
  -- chapter_85_line_11: GL tag disintegration.
  have row_11 : (succ one two) := by
    exact row_7.1.2
  -- chapter_85_line_10: GL tag disintegration.
  have row_10 : (succ zero one) := by
    exact row_7.1.1.2
  -- chapter_85_line_9: GL tag disintegration.
  have row_9 : (gl_identity N identity) := by
    exact row_7.2
  -- chapter_85_line_6: GL tag disintegration.
  have row_6 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_7.1.1.1
  -- chapter_85_line_28: GL tag implication.
  have row_28 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_29
    exact row_6
    exact row_10
  have rule_row_27 := external_peano_externals_36_035 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_85_line_27: GL tag implication.
  have row_27 : (gl_existence11 N one succ) := by
    apply rule_row_27
  -- chapter_85_line_26: GL tag expansion.
  have row_26 : (¬ (∀ (v2 : α), ((N v2) → (¬ (succ v2 one))))) := by
    simpa only [gl_existence11] using row_27
  have exists_row_26 : ∃ (v2 : α), ((N v2) ∧ (succ v2 one)) := existsAndOfNotForallImpNot row_26
  obtain ⟨v2, witness_row_26⟩ := exists_row_26
  -- chapter_85_line_25: GL tag disintegration.
  have row_25 : (succ v2 one) := by
    exact witness_row_26.2
  -- chapter_85_line_20: GL tag expansion.
  have row_20 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_6
  -- chapter_85_line_24: GL tag disintegration.
  have row_24 : (gl_implication7 N succ) := by
    exact row_20.1.1.1.1.1.1.1.1.1.2
  -- chapter_85_line_23: GL tag expansion.
  have row_23 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w2 w1) → (∀ (w3 : α), ((succ w3 w1) → (w2 = w3))))))) := by
    simpa only [gl_implication7] using row_24
  -- chapter_85_line_19: GL tag disintegration.
  have row_19 : (gl_fXY succ N N) := by
    exact row_20.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_85_line_18: GL tag expansion.
  have row_18 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_19
  -- chapter_85_line_17: GL tag disintegration.
  have row_17 : (gl_implication0 succ N) := by
    exact row_18.1.1.1
  -- chapter_85_line_16: GL tag expansion.
  have row_16 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_17
  -- chapter_85_line_31: GL tag implication.
  have row_31 : (N one) := by
    apply row_16
    exact row_11
  -- chapter_85_line_33: GL tag implication.
  have row_33 : (v2 = zero) := by
    apply row_23
    exact row_31
    exact row_25
    exact row_10
  -- chapter_85_line_22: GL tag implication.
  have row_22 : (zero = v2) := by
    apply row_23
    exact row_31
    exact row_10
    exact row_25
  -- chapter_85_line_15: GL tag implication.
  have row_15 : (N previous) := by
    apply row_16
    exact row_21
  -- chapter_85_line_13: GL tag implication.
  have row_13 : (gl_preorder N add zero previous) := by
    apply row_14
    exact row_15
  -- chapter_85_line_12: GL tag equality1.
  have row_12 : (gl_preorder N add v2 previous) := by
    have equality_source := row_13
    have equality_step_1 := row_22
    cases equality_step_1
    exact equality_source
  -- chapter_85_line_5: GL tag expansion for integration.
  have row_5 : ((gl_AnchorGauss N zero succ add mul one two identity) ↔ ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity))) := by
    exact Iff.rfl
  -- chapter_85_line_4: GL tag reformulation for integration and.
  have row_4 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → ((succ one two) → ((gl_identity N identity) → (gl_AnchorGauss N zero succ add mul one two identity))))) := by
    intro integration_premise_1
    intro integration_premise_2
    intro integration_premise_3
    intro integration_premise_4
    simp only [gl_AnchorGauss]
    exact ⟨⟨⟨integration_premise_1, integration_premise_2⟩, integration_premise_3⟩, integration_premise_4⟩
  -- chapter_85_line_3: GL tag implication.
  have row_3 : (gl_AnchorGauss N zero succ add mul one two identity) := by
    apply row_4
    exact row_6
    exact row_10
    exact row_11
    exact row_9
  have rule_row_32 := external_gauss_externals_24_018 N zero succ add mul one two identity (anchorGaussOfFTA N zero succ add mul one two identity anchor)
  -- chapter_85_line_32: GL tag implication.
  have row_32 : (gl_preorder N add previous v1) := by
    apply rule_row_32
    exact row_21
  have rule_row_2 := external_gauss_externals_24_021 N zero succ add mul one two identity (anchorGaussOfFTA N zero succ add mul one two identity anchor)
  -- chapter_85_line_2: GL tag implication.
  have row_2 : (gl_preorder N add v2 v1) := by
    apply rule_row_2
    exact row_12
    exact row_32
  -- chapter_85_line_1: GL tag equality1.
  have row_1 : (gl_preorder N add zero v1) := by
    have equality_source := row_2
    have equality_step_1 := row_33
    cases equality_step_1
    exact equality_source
  exact row_1

theorem fta_source_063
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_035 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (gl_existence11 N one succ))))
    (external_gauss_externals_24_018 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α) (x_7 : α) (x_8 : GLBinaryRelation α), ((gl_AnchorGauss x_1 x_2 x_3 x_4 x_5 x_6 x_7 x_8) → (∀ (x_9 : α) (x_10 : α), ((x_3 x_9 x_10) → (gl_preorder x_1 x_4 x_9 x_10))))))
    (external_gauss_externals_24_021 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α) (x_7 : α) (x_8 : GLBinaryRelation α), ((gl_AnchorGauss x_1 x_2 x_3 x_4 x_5 x_6 x_7 x_8) → (∀ (x_9 : α) (x_10 : α), ((gl_preorder x_1 x_4 x_9 x_10) → (∀ (x_11 : α), ((gl_preorder x_1 x_4 x_10 x_11) → (gl_preorder x_1 x_4 x_9 x_11))))))))
    : (∀ (v1 : α), ((N v1) → (gl_preorder N add zero v1))) := by
  intro v1
  intro premise_1
  have inductionMember : N v1 := by
    have typingRule := fta_source_063_induction_typing N zero succ add mul one two identity anchor relationalInduction
    exact typingRule v1 premise_1
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorFTA, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (gl_preorder N add zero zero) := by
    have zeroRule := fta_source_063_check_zero N zero succ add mul one two identity anchor relationalInduction
    exact zeroRule zero rfl inductionZeroMember
  have inductionStep :
      ∀ induction_n, N induction_n → (gl_preorder N add zero induction_n) → ∀ induction_m, succ induction_n induction_m → (gl_preorder N add zero induction_m) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    have step_induction_assumption_1 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have step_induction_assumption_2 :
        ((N induction_n) → (gl_preorder N add zero induction_n)) := by
      intro step_induction_assumption_2_premise_1
      apply induction_hypothesis
      all_goals assumption
    have stepRule := fta_source_063_check_induction_condition N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_035 external_gauss_externals_24_018 external_gauss_externals_24_021
    exact stepRule induction_n induction_m step_induction_assumption_1 step_induction_assumption_2
  have inductionProperty : (gl_preorder N add zero v1) := by
    exact relationalInduction N zero succ add mul one two identity anchor
      (fun induction_value => (gl_preorder N add zero induction_value))
      v1
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty

theorem fta_source_064
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_033 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α), ((x_1 x_7) → (∀ (x_8 : α), ((x_5 x_6 x_8 x_7) → (x_7 = x_8))))))))
    : (∀ (v1 : α), ((N v1) → (gl_preorder N mul one v1))) := by
  intro v1
  intro premise_1
  -- chapter_86_line_10: GL tag task formulation.
  have row_10 : (N v1) := by
    exact premise_1
  -- chapter_86_line_8: GL tag task formulation.
  have row_8 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_86_line_7: GL tag variable copy.
  have row_7 : (v1 = v1) := by
    rfl
  -- chapter_86_line_9: GL tag equality1.
  have row_9 : (N v1) := by
    have equality_source := row_10
    have equality_step_1 := row_7
    cases equality_step_1
    exact equality_source
  -- chapter_86_line_6: GL tag symmetry of equality.
  have row_6 : (v1 = v1) := by
    exact Eq.symm row_7
  -- chapter_86_line_5: GL tag theorem.
  have row_5 := fta_source_049 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_033
  -- chapter_86_line_4: GL tag implication.
  have row_4 : (mul one v1 v1) := by
    apply row_5
    exact row_9
    exact row_6
  -- chapter_86_line_3: GL tag expansion for integration.
  have row_3 : ((gl_preorder N mul one v1) ↔ (¬ (∀ (v4 : α), ((N v4) → (¬ (mul one v4 v1)))))) := by
    exact Iff.rfl
  -- chapter_86_line_2: GL tag reformulation for integration >[bound].
  have row_2 : (∀ (v2 : α), ((N v2) → ((mul one v2 v1) → (gl_preorder N mul one v1)))) := by
    intro v2
    intro integration_premise_1
    intro integration_premise_2
    apply (row_3).2
    intro universal_counterexample
    exact universal_counterexample v2 integration_premise_1 integration_premise_2
  -- chapter_86_line_1: GL tag implication.
  have row_1 : (gl_preorder N mul one v1) := by
    apply row_2
    exact row_9
    exact row_4
  exact row_1

private theorem fta_source_069_induction_typing
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (assumption_10 : (mul v2 v1 v3))
    : (N v1) := by
  -- chapter_93_line_10: GL tag task formulation.
  have row_10 : (mul v2 v1 v3) := by
    exact assumption_10
  -- chapter_93_line_9: GL tag task formulation.
  have row_9 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_93_line_8: GL tag expansion.
  have row_8 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_9
  -- chapter_93_line_7: GL tag disintegration.
  have row_7 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_8.1.1.1
  -- chapter_93_line_6: GL tag expansion.
  have row_6 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_7
  -- chapter_93_line_5: GL tag disintegration.
  have row_5 : (gl_fXYZ mul N N N) := by
    exact row_6.1.1.1.2
  -- chapter_93_line_4: GL tag expansion.
  have row_4 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_5
  -- chapter_93_line_3: GL tag disintegration.
  have row_3 : (gl_implication9 mul N) := by
    exact row_4.1.1.1.2
  -- chapter_93_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication9] using row_3
  -- chapter_93_line_1: GL tag implication.
  have row_1 : (N v1) := by
    apply row_2
    exact row_10
  exact row_1

private theorem fta_source_069_check_zero
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_017 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_6 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_021 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_6 x_8) → (x_3 x_7 x_8))))))
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (assumption_7 : (gl_preorder N add one v1))
    (assumption_2 : (v1 = zero))
    : (gl_preorder N add v2 v3) := by
  -- chapter_94_line_7: GL tag task formulation.
  have row_7 : (gl_preorder N add one v1) := by
    exact assumption_7
  -- chapter_94_line_6: GL tag task formulation.
  have row_6 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_94_line_5: GL tag theorem.
  have row_5 := fta_source_066 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_017 external_peano_externals_36_021
  -- chapter_94_line_4: GL tag implication.
  have row_4 : (¬ (zero = v1)) := by
    apply row_5
    exact row_7
  -- chapter_94_line_3: GL tag symmetry of inequality.
  have row_3 : (¬ (v1 = zero)) := by
    exact fun equality => row_4 (Eq.symm equality)
  -- chapter_94_line_2: GL tag recursion.
  have row_2 : (v1 = zero) := by
    exact assumption_2
  -- chapter_94_line_1: GL tag vacuous truth.
  have row_1 : (gl_preorder N add v2 v3) := by
    exact False.elim (row_3 row_2)
  exact row_1

private theorem fta_source_069_check_induction_condition
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_006 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (x_4 x_8 x_7 x_9))))))
    (previous : α)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (assumption_17 : (mul v2 v1 v3))
    (assumption_16 : (succ previous v1))
    : (gl_preorder N add v2 v3) := by
  -- chapter_95_line_17: GL tag task formulation.
  have row_17 : (mul v2 v1 v3) := by
    exact assumption_17
  -- chapter_95_line_16: GL tag recursion.
  have row_16 : (succ previous v1) := by
    exact assumption_16
  -- chapter_95_line_10: GL tag task formulation.
  have row_10 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_95_line_9: GL tag expansion.
  have row_9 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_10
  -- chapter_95_line_11: GL tag disintegration.
  have row_11 : (succ zero one) := by
    exact row_9.1.1.2
  -- chapter_95_line_8: GL tag disintegration.
  have row_8 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_9.1.1.1
  -- chapter_95_line_15: GL tag expansion.
  have row_15 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_8
  -- chapter_95_line_32: GL tag disintegration.
  have row_32 : (gl_fXY succ N N) := by
    exact row_15.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_95_line_31: GL tag expansion.
  have row_31 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_32
  -- chapter_95_line_30: GL tag disintegration.
  have row_30 : (gl_implication0 succ N) := by
    exact row_31.1.1.1
  -- chapter_95_line_29: GL tag expansion.
  have row_29 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_30
  -- chapter_95_line_28: GL tag implication.
  have row_28 : (N previous) := by
    apply row_29
    exact row_16
  -- chapter_95_line_24: GL tag disintegration.
  have row_24 : (gl_fXYZ mul N N N) := by
    exact row_15.1.1.1.2
  -- chapter_95_line_23: GL tag expansion.
  have row_23 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_24
  -- chapter_95_line_27: GL tag disintegration.
  have row_27 : (gl_implication8 mul N) := by
    exact row_23.1.1.1.1
  -- chapter_95_line_26: GL tag expansion.
  have row_26 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_27
  -- chapter_95_line_25: GL tag implication.
  have row_25 : (N v2) := by
    apply row_26
    exact row_17
  -- chapter_95_line_22: GL tag disintegration.
  have row_22 : (gl_implication13 N N N mul) := by
    exact row_23.1.2
  -- chapter_95_line_21: GL tag expansion.
  have row_21 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication13] using row_22
  -- chapter_95_line_20: GL tag implication.
  have row_20 : (gl_existence1 N v2 previous mul) := by
    apply row_21
    exact row_25
    exact row_28
  -- chapter_95_line_19: GL tag expansion.
  have row_19 : (¬ (∀ (v5 : α), ((N v5) → (¬ (mul v2 previous v5))))) := by
    simpa only [gl_existence1] using row_20
  have exists_row_19 : ∃ (v5 : α), ((N v5) ∧ (mul v2 previous v5)) := existsAndOfNotForallImpNot row_19
  obtain ⟨v5, witness_row_19⟩ := exists_row_19
  -- chapter_95_line_33: GL tag disintegration.
  have row_33 : (N v5) := by
    exact witness_row_19.1
  -- chapter_95_line_18: GL tag disintegration.
  have row_18 : (mul v2 previous v5) := by
    exact witness_row_19.2
  -- chapter_95_line_14: GL tag disintegration.
  have row_14 : (gl_implication21 N succ mul add) := by
    exact row_15.2
  -- chapter_95_line_13: GL tag expansion.
  have row_13 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((mul w3 w1 w4) → (∀ (w5 : α), ((mul w3 w2 w5) → (add w4 w3 w5))))))))) := by
    simpa only [gl_implication21] using row_14
  -- chapter_95_line_12: GL tag implication.
  have row_12 : (add v5 v2 v3) := by
    apply row_13
    exact row_28
    exact row_16
    exact row_18
    exact row_17
  -- chapter_95_line_7: GL tag expansion for integration.
  have row_7 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_95_line_6: GL tag reformulation for integration and.
  have row_6 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_95_line_5: GL tag implication.
  have row_5 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_6
    exact row_8
    exact row_11
  have rule_row_4 := external_peano_externals_36_006 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_95_line_4: GL tag implication.
  have row_4 : (add v2 v5 v3) := by
    apply rule_row_4
    exact row_12
  -- chapter_95_line_3: GL tag expansion for integration.
  have row_3 : ((gl_preorder N add v2 v3) ↔ (¬ (∀ (v6 : α), ((N v6) → (¬ (add v2 v6 v3)))))) := by
    exact Iff.rfl
  -- chapter_95_line_2: GL tag reformulation for integration >[bound].
  have row_2 : (∀ (v4 : α), ((N v4) → ((add v2 v4 v3) → (gl_preorder N add v2 v3)))) := by
    intro v4
    intro integration_premise_1
    intro integration_premise_2
    apply (row_3).2
    intro universal_counterexample
    exact universal_counterexample v4 integration_premise_1 integration_premise_2
  -- chapter_95_line_1: GL tag implication.
  have row_1 : (gl_preorder N add v2 v3) := by
    apply row_2
    exact row_33
    exact row_4
  exact row_1

theorem fta_source_069
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_017 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_6 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_021 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_6 x_8) → (x_3 x_7 x_8))))))
    (external_peano_externals_36_006 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (x_4 x_8 x_7 x_9))))))
    : (∀ (v1 : α), ((gl_preorder N add one v1) → (∀ (v2 : α) (v3 : α), ((mul v2 v1 v3) → (gl_preorder N add v2 v3))))) := by
  intro v1
  intro premise_1
  intro v2
  intro v3
  intro premise_2
  have inductionMember : N v1 := by
    have typingRule := fta_source_069_induction_typing N zero succ add mul one two identity anchor relationalInduction
    exact typingRule v1 v2 v3 premise_2
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorFTA, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : ((gl_preorder N add one zero) → (∀ (v2 : α) (v3 : α), ((mul v2 zero v3) → (gl_preorder N add v2 v3)))) := by
    intro base_premise_1
    intro v2
    intro v3
    intro base_premise_2
    have zeroRule := fta_source_069_check_zero N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_017 external_peano_externals_36_021
    exact zeroRule zero v2 v3 base_premise_1 rfl
  have inductionStep :
      ∀ induction_n, N induction_n → ((gl_preorder N add one induction_n) → (∀ (v2 : α) (v3 : α), ((mul v2 induction_n v3) → (gl_preorder N add v2 v3)))) → ∀ induction_m, succ induction_n induction_m → ((gl_preorder N add one induction_m) → (∀ (v2 : α) (v3 : α), ((mul v2 induction_m v3) → (gl_preorder N add v2 v3)))) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro step_premise_1
    intro v2
    intro v3
    intro step_premise_2
    have step_induction_assumption_1 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have stepRule := fta_source_069_check_induction_condition N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_006
    exact stepRule induction_n induction_m v2 v3 step_premise_2 step_induction_assumption_1
  have inductionProperty : ((gl_preorder N add one v1) → (∀ (v2 : α) (v3 : α), ((mul v2 v1 v3) → (gl_preorder N add v2 v3)))) := by
    exact relationalInduction N zero succ add mul one two identity anchor
      (fun induction_value => ((gl_preorder N add one induction_value) → (∀ (v2 : α) (v3 : α), ((mul v2 induction_value v3) → (gl_preorder N add v2 v3)))))
      v1
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty premise_1 v2 v3 premise_2

theorem fta_source_070
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_017 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_6 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_021 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_6 x_8) → (x_3 x_7 x_8))))))
    (external_peano_externals_36_006 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (x_4 x_8 x_7 x_9))))))
    (external_peano_externals_36_024 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_8 x_2) → ((x_1 x_8) → (x_2 = x_7)))))))
    (external_peano_externals_36_015 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_3 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_022 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_6 x_8) → (x_4 x_6 x_7 x_8))))))
    : (∀ (v1 : α), ((gl_preorder N add one v1) → (∀ (v2 : α), ((gl_preorder N add one v2) → (∀ (v3 : α), ((mul v1 v2 v3) → (gl_preorder N add one v3))))))) := by
  intro v1
  intro premise_1
  intro v2
  intro premise_2
  intro v3
  intro premise_3
  apply Classical.byContradiction
  intro reductio
  -- chapter_96_line_43: GL tag task formulation.
  have row_43 : (¬ (gl_preorder N add one v3)) := by
    exact reductio
  -- chapter_96_line_42: GL tag theorem.
  have row_42 := fta_source_060 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_015 external_peano_externals_36_022
  -- chapter_96_line_24: GL tag task formulation.
  have row_24 : (mul v1 v2 v3) := by
    exact premise_3
  -- chapter_96_line_23: GL tag task formulation.
  have row_23 : (gl_preorder N add one v2) := by
    exact premise_2
  -- chapter_96_line_22: GL tag expansion.
  have row_22 : (¬ (∀ (v10 : α), ((N v10) → (¬ (add one v10 v2))))) := by
    simpa only [gl_preorder] using row_23
  have exists_row_22 : ∃ (v10 : α), ((N v10) ∧ (add one v10 v2)) := existsAndOfNotForallImpNot row_22
  obtain ⟨v10, witness_row_22⟩ := exists_row_22
  -- chapter_96_line_39: GL tag disintegration.
  have row_39 : (N v10) := by
    exact witness_row_22.1
  -- chapter_96_line_21: GL tag disintegration.
  have row_21 : (add one v10 v2) := by
    exact witness_row_22.2
  -- chapter_96_line_11: GL tag task formulation.
  have row_11 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_96_line_10: GL tag expansion.
  have row_10 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_11
  -- chapter_96_line_12: GL tag disintegration.
  have row_12 : (succ zero one) := by
    exact row_10.1.1.2
  -- chapter_96_line_9: GL tag disintegration.
  have row_9 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_10.1.1.1
  -- chapter_96_line_18: GL tag expansion.
  have row_18 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_9
  -- chapter_96_line_36: GL tag disintegration.
  have row_36 : (gl_fXYZ add N N N) := by
    exact row_18.1.1.1.1.1.1.1.1.2
  -- chapter_96_line_35: GL tag expansion.
  have row_35 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_36
  -- chapter_96_line_34: GL tag disintegration.
  have row_34 : (gl_implication10 add N) := by
    exact row_35.1.1.2
  -- chapter_96_line_33: GL tag expansion.
  have row_33 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication10] using row_34
  -- chapter_96_line_31: GL tag disintegration.
  have row_31 : (gl_fXYZ mul N N N) := by
    exact row_18.1.1.1.2
  -- chapter_96_line_30: GL tag expansion.
  have row_30 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_31
  -- chapter_96_line_46: GL tag disintegration.
  have row_46 : (gl_implication10 mul N) := by
    exact row_30.1.1.2
  -- chapter_96_line_45: GL tag expansion.
  have row_45 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication10] using row_46
  -- chapter_96_line_44: GL tag implication.
  have row_44 : (N v3) := by
    apply row_45
    exact row_24
  -- chapter_96_line_41: GL tag implication.
  have row_41 : (zero = v3) := by
    apply row_42
    exact row_44
    exact row_43
  -- chapter_96_line_40: GL tag symmetry of equality.
  have row_40 : (v3 = zero) := by
    exact Eq.symm row_41
  -- chapter_96_line_29: GL tag disintegration.
  have row_29 : (gl_implication13 N N N mul) := by
    exact row_30.1.2
  -- chapter_96_line_28: GL tag expansion.
  have row_28 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication13] using row_29
  -- chapter_96_line_17: GL tag disintegration.
  have row_17 : (gl_implication21 N succ mul add) := by
    exact row_18.2
  -- chapter_96_line_16: GL tag expansion.
  have row_16 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((mul w3 w1 w4) → (∀ (w5 : α), ((mul w3 w2 w5) → (add w4 w3 w5))))))))) := by
    simpa only [gl_implication21] using row_17
  -- chapter_96_line_8: GL tag expansion for integration.
  have row_8 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_96_line_7: GL tag reformulation for integration and.
  have row_7 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_96_line_6: GL tag implication.
  have row_6 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_7
    exact row_9
    exact row_12
  have rule_row_20 := external_peano_externals_36_017 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_96_line_20: GL tag implication.
  have row_20 : (add v10 one v2) := by
    apply rule_row_20
    exact row_21
  have rule_row_19 := external_peano_externals_36_021 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_96_line_19: GL tag implication.
  have row_19 : (succ v10 v2) := by
    apply rule_row_19
    exact row_20
  -- chapter_96_line_2: GL tag task formulation.
  have row_2 : (gl_preorder N add one v1) := by
    exact premise_1
  -- chapter_96_line_38: GL tag expansion.
  have row_38 : (¬ (∀ (v11 : α), ((N v11) → (¬ (add one v11 v1))))) := by
    simpa only [gl_preorder] using row_2
  have exists_row_38 : ∃ (v11 : α), ((N v11) ∧ (add one v11 v1)) := existsAndOfNotForallImpNot row_38
  obtain ⟨v11, witness_row_38⟩ := exists_row_38
  -- chapter_96_line_37: GL tag disintegration.
  have row_37 : (add one v11 v1) := by
    exact witness_row_38.2
  -- chapter_96_line_32: GL tag implication.
  have row_32 : (N v1) := by
    apply row_33
    exact row_37
  -- chapter_96_line_27: GL tag implication.
  have row_27 : (gl_existence1 N v1 v10 mul) := by
    apply row_28
    exact row_32
    exact row_39
  -- chapter_96_line_26: GL tag expansion.
  have row_26 : (¬ (∀ (v4 : α), ((N v4) → (¬ (mul v1 v10 v4))))) := by
    simpa only [gl_existence1] using row_27
  have exists_row_26 : ∃ (v4 : α), ((N v4) ∧ (mul v1 v10 v4)) := existsAndOfNotForallImpNot row_26
  obtain ⟨v4, witness_row_26⟩ := exists_row_26
  -- chapter_96_line_47: GL tag disintegration.
  have row_47 : (N v4) := by
    exact witness_row_26.1
  -- chapter_96_line_25: GL tag disintegration.
  have row_25 : (mul v1 v10 v4) := by
    exact witness_row_26.2
  -- chapter_96_line_15: GL tag implication.
  have row_15 : (add v4 v1 v3) := by
    apply row_16
    exact row_39
    exact row_19
    exact row_25
    exact row_24
  have rule_row_14 := external_peano_externals_36_006 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_96_line_14: GL tag implication.
  have row_14 : (add v1 v4 v3) := by
    apply rule_row_14
    exact row_15
  -- chapter_96_line_13: GL tag equality1.
  have row_13 : (add v1 v4 zero) := by
    have equality_source := row_14
    have equality_step_1 := row_40
    cases equality_step_1
    exact equality_source
  have rule_row_5 := external_peano_externals_36_024 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_96_line_5: GL tag implication.
  have row_5 : (zero = v1) := by
    apply rule_row_5
    exact row_13
    exact row_47
  -- chapter_96_line_4: GL tag symmetry of equality.
  have row_4 : (v1 = zero) := by
    exact Eq.symm row_5
  -- chapter_96_line_3: GL tag equality2.
  have row_3 : (v1 = v3) := by
    exact Eq.trans row_4 row_41
  -- chapter_96_line_1: GL tag equality1.
  have row_1 : (gl_preorder N add one v3) := by
    have equality_source := row_2
    have equality_step_1 := row_3
    cases equality_step_1
    exact equality_source
  exact reductio row_1

private theorem fta_source_074_induction_typing
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (assumption_10 : (mul v1 v2 v3))
    : (N v2) := by
  -- chapter_102_line_10: GL tag task formulation.
  have row_10 : (mul v1 v2 v3) := by
    exact assumption_10
  -- chapter_102_line_9: GL tag task formulation.
  have row_9 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_102_line_8: GL tag expansion.
  have row_8 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_9
  -- chapter_102_line_7: GL tag disintegration.
  have row_7 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_8.1.1.1
  -- chapter_102_line_6: GL tag expansion.
  have row_6 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_7
  -- chapter_102_line_5: GL tag disintegration.
  have row_5 : (gl_fXYZ mul N N N) := by
    exact row_6.1.1.1.2
  -- chapter_102_line_4: GL tag expansion.
  have row_4 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_5
  -- chapter_102_line_3: GL tag disintegration.
  have row_3 : (gl_implication9 mul N) := by
    exact row_4.1.1.1.2
  -- chapter_102_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication9] using row_3
  -- chapter_102_line_1: GL tag implication.
  have row_1 : (N v2) := by
    apply row_2
    exact row_10
  exact row_1

private theorem fta_source_074_check_zero
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_017 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_6 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_021 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_6 x_8) → (x_3 x_7 x_8))))))
    (v2 : α)
    (v3 : α)
    (assumption_7 : (gl_preorder N add one v2))
    (assumption_2 : (v2 = zero))
    : (gl_preorder N add two v3) := by
  -- chapter_103_line_7: GL tag task formulation.
  have row_7 : (gl_preorder N add one v2) := by
    exact assumption_7
  -- chapter_103_line_6: GL tag task formulation.
  have row_6 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_103_line_5: GL tag theorem.
  have row_5 := fta_source_066 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_017 external_peano_externals_36_021
  -- chapter_103_line_4: GL tag implication.
  have row_4 : (¬ (zero = v2)) := by
    apply row_5
    exact row_7
  -- chapter_103_line_3: GL tag symmetry of inequality.
  have row_3 : (¬ (v2 = zero)) := by
    exact fun equality => row_4 (Eq.symm equality)
  -- chapter_103_line_2: GL tag recursion.
  have row_2 : (v2 = zero) := by
    exact assumption_2
  -- chapter_103_line_1: GL tag vacuous truth.
  have row_1 : (gl_preorder N add two v3) := by
    exact False.elim (row_3 row_2)
  exact row_1

private theorem fta_source_074_check_induction_condition
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_017 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_6 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_021 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_6 x_8) → (x_3 x_7 x_8))))))
    (external_peano_externals_36_006 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (x_4 x_8 x_7 x_9))))))
    (external_gauss_externals_24_021 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α) (x_7 : α) (x_8 : GLBinaryRelation α), ((gl_AnchorGauss x_1 x_2 x_3 x_4 x_5 x_6 x_7 x_8) → (∀ (x_9 : α) (x_10 : α), ((gl_preorder x_1 x_4 x_9 x_10) → (∀ (x_11 : α), ((gl_preorder x_1 x_4 x_10 x_11) → (gl_preorder x_1 x_4 x_9 x_11))))))))
    (external_peano_externals_36_004 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (∀ (x_10 : α), ((x_4 x_7 x_10 x_9) → (x_8 = x_10))))))))
    (external_peano_externals_36_015 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_3 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (previous : α)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (assumption_68 : (gl_preorder N add one v2))
    (assumption_38 : (gl_preorder N add two v1))
    (assumption_23 : (mul v1 v2 v3))
    (assumption_22 : (succ previous v2))
    : (gl_preorder N add two v3) := by
  -- chapter_104_line_90: GL tag theorem.
  have row_90 := fta_source_001 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_006 external_peano_externals_36_004
  -- chapter_104_line_68: GL tag task formulation.
  have row_68 : (gl_preorder N add one v2) := by
    exact assumption_68
  -- chapter_104_line_67: GL tag expansion.
  have row_67 : (¬ (∀ (v12 : α), ((N v12) → (¬ (add one v12 v2))))) := by
    simpa only [gl_preorder] using row_68
  have exists_row_67 : ∃ (v12 : α), ((N v12) ∧ (add one v12 v2)) := existsAndOfNotForallImpNot row_67
  obtain ⟨v12, witness_row_67⟩ := exists_row_67
  -- chapter_104_line_66: GL tag disintegration.
  have row_66 : (add one v12 v2) := by
    exact witness_row_67.2
  -- chapter_104_line_59: GL tag theorem.
  have row_59 := fta_source_019 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_015
  -- chapter_104_line_38: GL tag task formulation.
  have row_38 : (gl_preorder N add two v1) := by
    exact assumption_38
  -- chapter_104_line_37: GL tag expansion.
  have row_37 : (¬ (∀ (v11 : α), ((N v11) → (¬ (add two v11 v1))))) := by
    simpa only [gl_preorder] using row_38
  have exists_row_37 : ∃ (v11 : α), ((N v11) ∧ (add two v11 v1)) := existsAndOfNotForallImpNot row_37
  obtain ⟨v11, witness_row_37⟩ := exists_row_37
  -- chapter_104_line_36: GL tag disintegration.
  have row_36 : (add two v11 v1) := by
    exact witness_row_37.2
  -- chapter_104_line_23: GL tag task formulation.
  have row_23 : (mul v1 v2 v3) := by
    exact assumption_23
  -- chapter_104_line_22: GL tag recursion.
  have row_22 : (succ previous v2) := by
    exact assumption_22
  -- chapter_104_line_17: GL tag expansion for integration.
  have row_17 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_104_line_16: GL tag reformulation for integration and.
  have row_16 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_104_line_13: GL tag theorem.
  have row_13 := fta_source_002 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_104_line_8: GL tag task formulation.
  have row_8 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_104_line_60: GL tag anchor handling.
  have row_60 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact row_8
  -- chapter_104_line_58: GL tag implication.
  have row_58 : (add previous one v2) := by
    apply row_59
    exact row_22
  -- chapter_104_line_7: GL tag expansion.
  have row_7 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_8
  -- chapter_104_line_11: GL tag disintegration.
  have row_11 : (succ one two) := by
    exact row_7.1.2
  -- chapter_104_line_10: GL tag disintegration.
  have row_10 : (succ zero one) := by
    exact row_7.1.1.2
  -- chapter_104_line_9: GL tag disintegration.
  have row_9 : (gl_identity N identity) := by
    exact row_7.2
  -- chapter_104_line_75: GL tag expansion.
  have row_75 : (((gl_implication0 identity N) ∧ (gl_implication22 identity)) ∧ (gl_implication23 N identity)) := by
    simpa only [gl_identity] using row_9
  -- chapter_104_line_80: GL tag disintegration.
  have row_80 : (gl_implication23 N identity) := by
    exact row_75.2
  -- chapter_104_line_79: GL tag expansion.
  have row_79 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((w1 = w2) → (identity w1 w2))))) := by
    simpa only [gl_implication23] using row_80
  -- chapter_104_line_74: GL tag disintegration.
  have row_74 : (gl_implication22 identity) := by
    exact row_75.1.2
  -- chapter_104_line_73: GL tag expansion.
  have row_73 : (∀ (w1 : α) (w2 : α), ((identity w1 w2) → (w1 = w2))) := by
    simpa only [gl_implication22] using row_74
  -- chapter_104_line_6: GL tag disintegration.
  have row_6 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_7.1.1.1
  -- chapter_104_line_21: GL tag expansion.
  have row_21 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_6
  -- chapter_104_line_85: GL tag disintegration.
  have row_85 : (N zero) := by
    exact row_21.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_104_line_63: GL tag disintegration.
  have row_63 : (gl_implication7 N succ) := by
    exact row_21.1.1.1.1.1.1.1.1.1.2
  -- chapter_104_line_62: GL tag expansion.
  have row_62 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w2 w1) → (∀ (w3 : α), ((succ w3 w1) → (w2 = w3))))))) := by
    simpa only [gl_implication7] using row_63
  -- chapter_104_line_43: GL tag disintegration.
  have row_43 : (gl_fXY succ N N) := by
    exact row_21.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_104_line_42: GL tag expansion.
  have row_42 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_43
  -- chapter_104_line_71: GL tag disintegration.
  have row_71 : (gl_implication1 succ N) := by
    exact row_42.1.1.2
  -- chapter_104_line_70: GL tag expansion.
  have row_70 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using row_71
  -- chapter_104_line_69: GL tag implication.
  have row_69 : (N v2) := by
    apply row_70
    exact row_22
  -- chapter_104_line_53: GL tag disintegration.
  have row_53 : (gl_implication4 N N succ) := by
    exact row_42.1.2
  -- chapter_104_line_52: GL tag expansion.
  have row_52 : (∀ (w1 : α), ((N w1) → (gl_existence0 N w1 succ))) := by
    simpa only [gl_implication4] using row_53
  -- chapter_104_line_84: GL tag implication.
  have row_84 : (gl_existence0 N zero succ) := by
    apply row_52
    exact row_85
  -- chapter_104_line_83: GL tag expansion.
  have row_83 : (¬ (∀ (v13 : α), ((N v13) → (¬ (succ zero v13))))) := by
    simpa only [gl_existence0] using row_84
  have exists_row_83 : ∃ (v13 : α), ((N v13) ∧ (succ zero v13)) := existsAndOfNotForallImpNot row_83
  obtain ⟨v13, witness_row_83⟩ := exists_row_83
  -- chapter_104_line_82: GL tag disintegration.
  have row_82 : (succ zero v13) := by
    exact witness_row_83.2
  -- chapter_104_line_47: GL tag disintegration.
  have row_47 : (gl_implication5 N succ) := by
    exact row_42.2
  -- chapter_104_line_46: GL tag expansion.
  have row_46 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α), ((succ w1 w3) → (w2 = w3))))))) := by
    simpa only [gl_implication5] using row_47
  -- chapter_104_line_87: GL tag implication.
  have row_87 : (v13 = one) := by
    apply row_46
    exact row_85
    exact row_82
    exact row_10
  -- chapter_104_line_81: GL tag implication.
  have row_81 : (one = v13) := by
    apply row_46
    exact row_85
    exact row_10
    exact row_82
  -- chapter_104_line_91: GL tag equality1.
  have row_91 : (add v13 v12 v2) := by
    have equality_source := row_66
    have equality_step_1 := row_81
    cases equality_step_1
    exact equality_source
  -- chapter_104_line_41: GL tag disintegration.
  have row_41 : (gl_implication0 succ N) := by
    exact row_42.1.1.1
  -- chapter_104_line_40: GL tag expansion.
  have row_40 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_41
  -- chapter_104_line_86: GL tag implication.
  have row_86 : (N one) := by
    apply row_40
    exact row_11
  -- chapter_104_line_78: GL tag implication.
  have row_78 : (identity one v13) := by
    apply row_79
    exact row_86
    exact row_81
  -- chapter_104_line_77: GL tag equality1.
  have row_77 : (identity one one) := by
    have equality_source := row_78
    have equality_step_1 := row_87
    cases equality_step_1
    exact equality_source
  -- chapter_104_line_39: GL tag implication.
  have row_39 : (N previous) := by
    apply row_40
    exact row_22
  -- chapter_104_line_35: GL tag disintegration.
  have row_35 : (gl_fXYZ add N N N) := by
    exact row_21.1.1.1.1.1.1.1.1.2
  -- chapter_104_line_34: GL tag expansion.
  have row_34 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_35
  -- chapter_104_line_56: GL tag disintegration.
  have row_56 : (gl_implication9 add N) := by
    exact row_34.1.1.1.2
  -- chapter_104_line_55: GL tag expansion.
  have row_55 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication9] using row_56
  -- chapter_104_line_33: GL tag disintegration.
  have row_33 : (gl_implication10 add N) := by
    exact row_34.1.1.2
  -- chapter_104_line_32: GL tag expansion.
  have row_32 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication10] using row_33
  -- chapter_104_line_31: GL tag implication.
  have row_31 : (N v1) := by
    apply row_32
    exact row_36
  -- chapter_104_line_30: GL tag disintegration.
  have row_30 : (gl_fXYZ mul N N N) := by
    exact row_21.1.1.1.2
  -- chapter_104_line_29: GL tag expansion.
  have row_29 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_30
  -- chapter_104_line_28: GL tag disintegration.
  have row_28 : (gl_implication13 N N N mul) := by
    exact row_29.1.2
  -- chapter_104_line_27: GL tag expansion.
  have row_27 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication13] using row_28
  -- chapter_104_line_26: GL tag implication.
  have row_26 : (gl_existence1 N v1 previous mul) := by
    apply row_27
    exact row_31
    exact row_39
  -- chapter_104_line_25: GL tag expansion.
  have row_25 : (¬ (∀ (v5 : α), ((N v5) → (¬ (mul v1 previous v5))))) := by
    simpa only [gl_existence1] using row_26
  have exists_row_25 : ∃ (v5 : α), ((N v5) ∧ (mul v1 previous v5)) := existsAndOfNotForallImpNot row_25
  obtain ⟨v5, witness_row_25⟩ := exists_row_25
  -- chapter_104_line_24: GL tag disintegration.
  have row_24 : (mul v1 previous v5) := by
    exact witness_row_25.2
  -- chapter_104_line_20: GL tag disintegration.
  have row_20 : (gl_implication21 N succ mul add) := by
    exact row_21.2
  -- chapter_104_line_19: GL tag expansion.
  have row_19 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((mul w3 w1 w4) → (∀ (w5 : α), ((mul w3 w2 w5) → (add w4 w3 w5))))))))) := by
    simpa only [gl_implication21] using row_20
  -- chapter_104_line_18: GL tag implication.
  have row_18 : (add v5 v1 v3) := by
    apply row_19
    exact row_39
    exact row_22
    exact row_24
    exact row_23
  -- chapter_104_line_15: GL tag implication.
  have row_15 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_16
    exact row_6
    exact row_10
  have rule_row_65 := external_peano_externals_36_017 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_104_line_65: GL tag implication.
  have row_65 : (add v12 one v2) := by
    apply rule_row_65
    exact row_66
  have rule_row_64 := external_peano_externals_36_021 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_104_line_64: GL tag implication.
  have row_64 : (succ v12 v2) := by
    apply rule_row_64
    exact row_65
  -- chapter_104_line_61: GL tag implication.
  have row_61 : (previous = v12) := by
    apply row_62
    exact row_69
    exact row_22
    exact row_64
  -- chapter_104_line_57: GL tag equality1.
  have row_57 : (add v12 one v2) := by
    have equality_source := row_58
    have equality_step_1 := row_61
    cases equality_step_1
    exact equality_source
  have rule_row_92 := external_peano_externals_36_006 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_104_line_92: GL tag implication.
  have row_92 : (add one v12 v2) := by
    apply rule_row_92
    exact row_57
  -- chapter_104_line_89: GL tag implication.
  have row_89 : (v13 = one) := by
    apply row_90
    exact row_91
    exact row_92
  -- chapter_104_line_88: GL tag equality2.
  have row_88 : (one = one) := by
    exact Eq.trans row_81 row_89
  -- chapter_104_line_76: GL tag equality1.
  have row_76 : (identity one one) := by
    have equality_source := row_77
    have equality_step_1 := row_88
    cases equality_step_1
    exact equality_source
  -- chapter_104_line_72: GL tag implication.
  have row_72 : (one = one) := by
    apply row_73
    exact row_76
  -- chapter_104_line_54: GL tag implication.
  have row_54 : (N one) := by
    apply row_55
    exact row_57
  -- chapter_104_line_51: GL tag implication.
  have row_51 : (gl_existence0 N one succ) := by
    apply row_52
    exact row_54
  -- chapter_104_line_50: GL tag expansion.
  have row_50 : (¬ (∀ (v4 : α), ((N v4) → (¬ (succ one v4))))) := by
    simpa only [gl_existence0] using row_51
  have exists_row_50 : ∃ (v4 : α), ((N v4) ∧ (succ one v4)) := existsAndOfNotForallImpNot row_50
  obtain ⟨v4, witness_row_50⟩ := exists_row_50
  -- chapter_104_line_49: GL tag disintegration.
  have row_49 : (succ one v4) := by
    exact witness_row_50.2
  -- chapter_104_line_48: GL tag equality1.
  have row_48 : (succ one v4) := by
    have equality_source := row_49
    have equality_step_1 := row_72
    cases equality_step_1
    exact equality_source
  -- chapter_104_line_93: GL tag implication.
  have row_93 : (v4 = two) := by
    apply row_46
    exact row_86
    exact row_48
    exact row_11
  -- chapter_104_line_45: GL tag implication.
  have row_45 : (two = v4) := by
    apply row_46
    exact row_86
    exact row_11
    exact row_48
  -- chapter_104_line_44: GL tag equality1.
  have row_44 : (gl_preorder N add v4 v1) := by
    have equality_source := row_38
    have equality_step_1 := row_45
    cases equality_step_1
    exact equality_source
  have rule_row_14 := external_peano_externals_36_006 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_104_line_14: GL tag implication.
  have row_14 : (add v1 v5 v3) := by
    apply rule_row_14
    exact row_18
  -- chapter_104_line_12: GL tag implication.
  have row_12 : (gl_preorder N add v1 v3) := by
    apply row_13
    exact row_14
  -- chapter_104_line_5: GL tag expansion for integration.
  have row_5 : ((gl_AnchorGauss N zero succ add mul one two identity) ↔ ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity))) := by
    exact Iff.rfl
  -- chapter_104_line_4: GL tag reformulation for integration and.
  have row_4 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → ((succ one two) → ((gl_identity N identity) → (gl_AnchorGauss N zero succ add mul one two identity))))) := by
    intro integration_premise_1
    intro integration_premise_2
    intro integration_premise_3
    intro integration_premise_4
    simp only [gl_AnchorGauss]
    exact ⟨⟨⟨integration_premise_1, integration_premise_2⟩, integration_premise_3⟩, integration_premise_4⟩
  -- chapter_104_line_3: GL tag implication.
  have row_3 : (gl_AnchorGauss N zero succ add mul one two identity) := by
    apply row_4
    exact row_6
    exact row_10
    exact row_11
    exact row_9
  have rule_row_2 := external_gauss_externals_24_021 N zero succ add mul one two identity (anchorGaussOfFTA N zero succ add mul one two identity anchor)
  -- chapter_104_line_2: GL tag implication.
  have row_2 : (gl_preorder N add v4 v3) := by
    apply rule_row_2
    exact row_44
    exact row_12
  -- chapter_104_line_1: GL tag equality1.
  have row_1 : (gl_preorder N add two v3) := by
    have equality_source := row_2
    have equality_step_1 := row_93
    cases equality_step_1
    exact equality_source
  exact row_1

theorem fta_source_074
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_017 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_6 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_021 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_6 x_8) → (x_3 x_7 x_8))))))
    (external_peano_externals_36_006 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (x_4 x_8 x_7 x_9))))))
    (external_gauss_externals_24_021 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α) (x_7 : α) (x_8 : GLBinaryRelation α), ((gl_AnchorGauss x_1 x_2 x_3 x_4 x_5 x_6 x_7 x_8) → (∀ (x_9 : α) (x_10 : α), ((gl_preorder x_1 x_4 x_9 x_10) → (∀ (x_11 : α), ((gl_preorder x_1 x_4 x_10 x_11) → (gl_preorder x_1 x_4 x_9 x_11))))))))
    (external_peano_externals_36_004 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (∀ (x_10 : α), ((x_4 x_7 x_10 x_9) → (x_8 = x_10))))))))
    (external_peano_externals_36_015 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_3 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    : (∀ (v1 : α), ((gl_preorder N add two v1) → (∀ (v2 : α), ((gl_preorder N add one v2) → (∀ (v3 : α), ((mul v1 v2 v3) → (gl_preorder N add two v3))))))) := by
  intro v1
  intro premise_1
  intro v2
  intro premise_2
  intro v3
  intro premise_3
  have inductionMember : N v2 := by
    have typingRule := fta_source_074_induction_typing N zero succ add mul one two identity anchor relationalInduction
    exact typingRule v1 v2 v3 premise_3
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorFTA, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v1 : α), ((gl_preorder N add two v1) → ((gl_preorder N add one zero) → (∀ (v3 : α), ((mul v1 zero v3) → (gl_preorder N add two v3)))))) := by
    intro v1
    intro base_premise_1
    intro base_premise_2
    intro v3
    intro base_premise_3
    have zeroRule := fta_source_074_check_zero N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_017 external_peano_externals_36_021
    exact zeroRule zero v3 base_premise_2 rfl
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v1 : α), ((gl_preorder N add two v1) → ((gl_preorder N add one induction_n) → (∀ (v3 : α), ((mul v1 induction_n v3) → (gl_preorder N add two v3)))))) → ∀ induction_m, succ induction_n induction_m → (∀ (v1 : α), ((gl_preorder N add two v1) → ((gl_preorder N add one induction_m) → (∀ (v3 : α), ((mul v1 induction_m v3) → (gl_preorder N add two v3)))))) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro v1
    intro step_premise_1
    intro step_premise_2
    intro v3
    intro step_premise_3
    have step_induction_assumption_1 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have stepRule := fta_source_074_check_induction_condition N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_017 external_peano_externals_36_021 external_peano_externals_36_006 external_gauss_externals_24_021 external_peano_externals_36_004 external_peano_externals_36_015
    exact stepRule induction_n v1 induction_m v3 step_premise_2 step_premise_1 step_premise_3 step_induction_assumption_1
  have inductionProperty : (∀ (v1 : α), ((gl_preorder N add two v1) → ((gl_preorder N add one v2) → (∀ (v3 : α), ((mul v1 v2 v3) → (gl_preorder N add two v3)))))) := by
    exact relationalInduction N zero succ add mul one two identity anchor
      (fun induction_value => (∀ (v1 : α), ((gl_preorder N add two v1) → ((gl_preorder N add one induction_value) → (∀ (v3 : α), ((mul v1 induction_value v3) → (gl_preorder N add two v3)))))))
      v2
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v1 premise_1 premise_2 v3 premise_3

theorem fta_source_016
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_gauss_externals_24_018 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α) (x_7 : α) (x_8 : GLBinaryRelation α), ((gl_AnchorGauss x_1 x_2 x_3 x_4 x_5 x_6 x_7 x_8) → (∀ (x_9 : α) (x_10 : α), ((x_3 x_9 x_10) → (gl_preorder x_1 x_4 x_9 x_10))))))
    (external_peano_externals_36_006 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (x_4 x_8 x_7 x_9))))))
    (external_peano_externals_36_003 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (∀ (x_10 : α), ((x_3 x_8 x_10) → (∀ (x_11 : α), ((x_3 x_11 x_7) → (x_4 x_10 x_11 x_9))))))))))
    (external_peano_externals_36_002 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (∀ (w4 : α) (w5 : α), ((add w4 w2 w5) → (∀ (w6 : α), ((add w3 w4 w6) → (add w5 w1 w6))))))))))
    (external_peano_externals_36_029 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((gl_preorder x_1 x_4 x_7 x_8) → ((gl_preorder x_1 x_4 x_8 x_7) → (x_7 = x_8)))))))
    (external_peano_externals_36_035 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (gl_existence11 N one succ))))
    (external_peano_externals_36_005 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → ((x_8 = x_9) → (x_2 = x_7)))))))
    : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (v3 : α), ((gl_preorder N add v3 v1) → (gl_strictOrder N add v3 v2))))) := by
  intro v1
  intro v2
  intro premise_1
  intro v3
  intro premise_2
  apply Classical.byContradiction
  intro reductio
  -- chapter_18_line_63: GL tag theorem.
  have row_63 := fta_source_002 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_18_line_55: GL tag expansion for integration.
  have row_55 : ((gl_AnchorGauss N zero succ add mul one two identity) ↔ ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity))) := by
    exact Iff.rfl
  -- chapter_18_line_54: GL tag reformulation for integration and.
  have row_54 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → ((succ one two) → ((gl_identity N identity) → (gl_AnchorGauss N zero succ add mul one two identity))))) := by
    intro integration_premise_1
    intro integration_premise_2
    intro integration_premise_3
    intro integration_premise_4
    simp only [gl_AnchorGauss]
    exact ⟨⟨⟨integration_premise_1, integration_premise_2⟩, integration_premise_3⟩, integration_premise_4⟩
  -- chapter_18_line_41: GL tag expansion for integration.
  have row_41 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_18_line_40: GL tag reformulation for integration and.
  have row_40 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_18_line_37: GL tag theorem.
  have row_37 := fta_source_029 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_006 external_peano_externals_36_002
  -- chapter_18_line_34: GL tag theorem.
  have row_34 := fta_source_033 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_029
  -- chapter_18_line_31: GL tag theorem.
  have row_31 := fta_source_014 N zero succ add mul one two identity anchor relationalInduction external_gauss_externals_24_018 external_peano_externals_36_035 external_peano_externals_36_029 external_peano_externals_36_003 external_peano_externals_36_005
  -- chapter_18_line_25: GL tag task formulation.
  have row_25 : (gl_preorder N add v3 v1) := by
    exact premise_2
  -- chapter_18_line_24: GL tag expansion.
  have row_24 : (¬ (∀ (v12 : α), ((N v12) → (¬ (add v3 v12 v1))))) := by
    simpa only [gl_preorder] using row_25
  have exists_row_24 : ∃ (v12 : α), ((N v12) ∧ (add v3 v12 v1)) := existsAndOfNotForallImpNot row_24
  obtain ⟨v12, witness_row_24⟩ := exists_row_24
  -- chapter_18_line_26: GL tag disintegration.
  have row_26 : (add v3 v12 v1) := by
    exact witness_row_24.2
  -- chapter_18_line_23: GL tag disintegration.
  have row_23 : (N v12) := by
    exact witness_row_24.1
  -- chapter_18_line_15: GL tag task formulation.
  have row_15 : (succ v1 v2) := by
    exact premise_1
  -- chapter_18_line_8: GL tag expansion for integration.
  have row_8 : ((gl_preorder N add v3 v2) ↔ (¬ (∀ (v6 : α), ((N v6) → (¬ (add v3 v6 v2)))))) := by
    exact Iff.rfl
  -- chapter_18_line_7: GL tag reformulation for integration >[bound].
  have row_7 : (∀ (v4 : α), ((N v4) → ((add v3 v4 v2) → (gl_preorder N add v3 v2)))) := by
    intro v4
    intro integration_premise_1
    intro integration_premise_2
    apply (row_8).2
    intro universal_counterexample
    exact universal_counterexample v4 integration_premise_1 integration_premise_2
  -- chapter_18_line_5: GL tag task formulation.
  have row_5 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_18_line_14: GL tag expansion.
  have row_14 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_5
  -- chapter_18_line_57: GL tag disintegration.
  have row_57 : (succ one two) := by
    exact row_14.1.2
  -- chapter_18_line_56: GL tag disintegration.
  have row_56 : (gl_identity N identity) := by
    exact row_14.2
  -- chapter_18_line_42: GL tag disintegration.
  have row_42 : (succ zero one) := by
    exact row_14.1.1.2
  -- chapter_18_line_13: GL tag disintegration.
  have row_13 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_14.1.1.1
  -- chapter_18_line_53: GL tag implication.
  have row_53 : (gl_AnchorGauss N zero succ add mul one two identity) := by
    apply row_54
    exact row_13
    exact row_42
    exact row_57
    exact row_56
  -- chapter_18_line_39: GL tag implication.
  have row_39 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_40
    exact row_13
    exact row_42
  -- chapter_18_line_12: GL tag expansion.
  have row_12 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_13
  -- chapter_18_line_50: GL tag disintegration.
  have row_50 : (gl_fXYZ add N N N) := by
    exact row_12.1.1.1.1.1.1.1.1.2
  -- chapter_18_line_49: GL tag expansion.
  have row_49 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_50
  -- chapter_18_line_48: GL tag disintegration.
  have row_48 : (gl_implication8 add N) := by
    exact row_49.1.1.1.1
  -- chapter_18_line_47: GL tag expansion.
  have row_47 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_48
  -- chapter_18_line_46: GL tag implication.
  have row_46 : (N v3) := by
    apply row_47
    exact row_26
  -- chapter_18_line_22: GL tag disintegration.
  have row_22 : (gl_fXY succ N N) := by
    exact row_12.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_18_line_21: GL tag expansion.
  have row_21 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_22
  -- chapter_18_line_20: GL tag disintegration.
  have row_20 : (gl_implication4 N N succ) := by
    exact row_21.1.2
  -- chapter_18_line_19: GL tag expansion.
  have row_19 : (∀ (w1 : α), ((N w1) → (gl_existence0 N w1 succ))) := by
    simpa only [gl_implication4] using row_20
  -- chapter_18_line_45: GL tag implication.
  have row_45 : (gl_existence0 N v3 succ) := by
    apply row_19
    exact row_46
  -- chapter_18_line_44: GL tag expansion.
  have row_44 : (¬ (∀ (v13 : α), ((N v13) → (¬ (succ v3 v13))))) := by
    simpa only [gl_existence0] using row_45
  have exists_row_44 : ∃ (v13 : α), ((N v13) ∧ (succ v3 v13)) := existsAndOfNotForallImpNot row_44
  obtain ⟨v13, witness_row_44⟩ := exists_row_44
  -- chapter_18_line_43: GL tag disintegration.
  have row_43 : (succ v3 v13) := by
    exact witness_row_44.2
  have rule_row_52 := external_gauss_externals_24_018 N zero succ add mul one two identity (anchorGaussOfFTA N zero succ add mul one two identity anchor)
  -- chapter_18_line_52: GL tag implication.
  have row_52 : (gl_preorder N add v3 v13) := by
    apply rule_row_52
    exact row_43
  -- chapter_18_line_18: GL tag implication.
  have row_18 : (gl_existence0 N v12 succ) := by
    apply row_19
    exact row_23
  -- chapter_18_line_17: GL tag expansion.
  have row_17 : (¬ (∀ (v5 : α), ((N v5) → (¬ (succ v12 v5))))) := by
    simpa only [gl_existence0] using row_18
  have exists_row_17 : ∃ (v5 : α), ((N v5) ∧ (succ v12 v5)) := existsAndOfNotForallImpNot row_17
  obtain ⟨v5, witness_row_17⟩ := exists_row_17
  -- chapter_18_line_27: GL tag disintegration.
  have row_27 : (N v5) := by
    exact witness_row_17.1
  -- chapter_18_line_16: GL tag disintegration.
  have row_16 : (succ v12 v5) := by
    exact witness_row_17.2
  -- chapter_18_line_11: GL tag disintegration.
  have row_11 : (gl_implication18 N succ add) := by
    exact row_12.1.1.1.1.2
  -- chapter_18_line_10: GL tag expansion.
  have row_10 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((succ w4 w5) → (add w3 w2 w5))))))))) := by
    simpa only [gl_implication18] using row_11
  -- chapter_18_line_9: GL tag implication.
  have row_9 : (add v3 v5 v2) := by
    apply row_10
    exact row_23
    exact row_16
    exact row_26
    exact row_15
  have rule_row_51 := external_peano_externals_36_006 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_18_line_51: GL tag implication.
  have row_51 : (add v5 v3 v2) := by
    apply rule_row_51
    exact row_9
  have rule_row_38 := external_peano_externals_36_003 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_18_line_38: GL tag implication.
  have row_38 : (add v13 v12 v2) := by
    apply rule_row_38
    exact row_51
    exact row_43
    exact row_16
  -- chapter_18_line_62: GL tag implication.
  have row_62 : (gl_preorder N add v13 v2) := by
    apply row_63
    exact row_38
  -- chapter_18_line_36: GL tag implication.
  have row_36 : (gl_preorder N add v1 v2) := by
    apply row_37
    exact row_52
    exact row_26
    exact row_38
  -- chapter_18_line_6: GL tag implication.
  have row_6 : (gl_preorder N add v3 v2) := by
    apply row_7
    exact row_27
    exact row_9
  -- chapter_18_line_4: GL tag task formulation.
  have row_4 : (¬ (gl_strictOrder N add v3 v2)) := by
    exact reductio
  -- chapter_18_line_3: GL tag theorem.
  have row_3 := fta_source_032 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_18_line_2: GL tag implication.
  have row_2 : (v3 = v2) := by
    apply row_3
    exact row_6
    exact row_4
  -- chapter_18_line_65: GL tag symmetry of equality.
  have row_65 : (v2 = v3) := by
    exact Eq.symm row_2
  -- chapter_18_line_61: GL tag equality1.
  have row_61 : (gl_preorder N add v2 v13) := by
    have equality_source := row_52
    have equality_step_1 := row_2
    cases equality_step_1
    exact equality_source
  -- chapter_18_line_64: GL tag implication.
  have row_64 : (v13 = v2) := by
    apply row_34
    exact row_62
    exact row_61
  -- chapter_18_line_60: GL tag implication.
  have row_60 : (v2 = v13) := by
    apply row_34
    exact row_61
    exact row_62
  -- chapter_18_line_35: GL tag equality1.
  have row_35 : (gl_preorder N add v2 v1) := by
    have equality_source := row_25
    have equality_step_1 := row_2
    cases equality_step_1
    exact equality_source
  -- chapter_18_line_59: GL tag implication.
  have row_59 : (v1 = v2) := by
    apply row_34
    exact row_36
    exact row_35
  -- chapter_18_line_58: GL tag equality2.
  have row_58 : (v1 = v13) := by
    exact Eq.trans row_59 row_60
  -- chapter_18_line_33: GL tag implication.
  have row_33 : (v2 = v1) := by
    apply row_34
    exact row_35
    exact row_36
  -- chapter_18_line_32: GL tag equality1.
  have row_32 : (succ v13 v1) := by
    have equality_source := row_15
    have equality_step_1 := row_33
    cases equality_step_1
    have equality_step_2 := row_58
    cases equality_step_2
    exact equality_source
  -- chapter_18_line_30: GL tag implication.
  have row_30 : (¬ (v13 = v2)) := by
    apply row_31
    exact row_32
    exact row_36
  -- chapter_18_line_29: GL tag equality1.
  have row_29 : (¬ (v2 = v2)) := by
    have equality_source := row_30
    have equality_step_1 := row_64
    cases equality_step_1
    exact equality_source
  -- chapter_18_line_28: GL tag equality1.
  have row_28 : (¬ (v3 = v2)) := by
    have equality_source := row_29
    have equality_step_1 := row_65
    cases equality_step_1
    exact equality_source
  -- chapter_18_line_1: GL tag contradiction.
  have row_1 : (gl_strictOrder N add v3 v2) := by
    exact False.elim (row_28 row_2)
  exact reductio row_1

theorem fta_source_024
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_035 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (gl_existence11 N one succ))))
    (external_peano_externals_36_024 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_8 x_2) → ((x_1 x_8) → (x_2 = x_7)))))))
    (external_peano_externals_36_025 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_8 x_2) → (x_4 x_8 x_7 x_2))))))
    : (∀ (v1 : α) (v2 : α), ((mul v1 v2 zero) → ((N v1) → (gl_or4 zero v1 v2)))) := by
  intro v1
  intro v2
  intro premise_1
  intro premise_2
  have or_parent_1 := fta_source_023 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_035 external_peano_externals_36_024 external_peano_externals_36_025
  have or_parent_2 := fta_source_022 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_035 external_peano_externals_36_024 external_peano_externals_36_025
  -- chapter_30_line_1: GL tag or theorem.
  have row_1 : (∀ (v1 : α) (v2 : α), ((mul v1 v2 zero) → ((N v1) → (gl_or4 zero v1 v2)))) := by
    classical
    intro v1
    intro v2
    intro or_parent_premise_1
    intro or_parent_premise_2
    simp only [gl_or4]
    by_cases or_case_1 : (zero = v1)
    · exact Or.inl (or_case_1)
    ·
      exact Or.inr ((or_parent_1 v1 v2 or_parent_premise_1 or_parent_premise_2 or_case_1))
  solve_by_elim [row_1]

theorem fta_source_038
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_010 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_5 x_7 x_8 x_9) → (x_5 x_8 x_7 x_9))))))
    (external_peano_externals_36_009 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (∀ (w4 : α) (w5 : α), ((mul w4 w2 w5) → (∀ (w6 : α), ((mul w3 w4 w6) → (mul w5 w1 w6))))))))))
    : (∀ (v1 : α) (v2 : α), ((gl_preorder N mul v1 v2) → (∀ (v3 : α) (v4 : α), ((gl_preorder N mul v3 v4) → (∀ (v5 : α), ((mul v1 v3 v5) → (∀ (v6 : α), ((mul v2 v4 v6) → (gl_preorder N mul v5 v6))))))))) := by
  intro v1
  intro v2
  intro premise_1
  intro v3
  intro v4
  intro premise_2
  intro v5
  intro premise_3
  intro v6
  intro premise_4
  -- chapter_48_line_42: GL tag task formulation.
  have row_42 : (gl_preorder N mul v3 v4) := by
    exact premise_2
  -- chapter_48_line_41: GL tag expansion.
  have row_41 : (¬ (∀ (v12 : α), ((N v12) → (¬ (mul v3 v12 v4))))) := by
    simpa only [gl_preorder] using row_42
  have exists_row_41 : ∃ (v12 : α), ((N v12) ∧ (mul v3 v12 v4)) := existsAndOfNotForallImpNot row_41
  obtain ⟨v12, witness_row_41⟩ := exists_row_41
  -- chapter_48_line_40: GL tag disintegration.
  have row_40 : (mul v3 v12 v4) := by
    exact witness_row_41.2
  -- chapter_48_line_38: GL tag task formulation.
  have row_38 : (mul v2 v4 v6) := by
    exact premise_4
  -- chapter_48_line_27: GL tag task formulation.
  have row_27 : (gl_preorder N mul v1 v2) := by
    exact premise_1
  -- chapter_48_line_26: GL tag expansion.
  have row_26 : (¬ (∀ (v8 : α), ((N v8) → (¬ (mul v1 v8 v2))))) := by
    simpa only [gl_preorder] using row_27
  have exists_row_26 : ∃ (v8 : α), ((N v8) ∧ (mul v1 v8 v2)) := existsAndOfNotForallImpNot row_26
  obtain ⟨v8, witness_row_26⟩ := exists_row_26
  -- chapter_48_line_51: GL tag disintegration.
  have row_51 : (mul v1 v8 v2) := by
    exact witness_row_26.2
  -- chapter_48_line_25: GL tag disintegration.
  have row_25 : (N v8) := by
    exact witness_row_26.1
  -- chapter_48_line_24: GL tag task formulation.
  have row_24 : (mul v1 v3 v5) := by
    exact premise_3
  -- chapter_48_line_9: GL tag expansion for integration.
  have row_9 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_48_line_8: GL tag reformulation for integration and.
  have row_8 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_48_line_5: GL tag theorem.
  have row_5 := fta_source_010 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_010
  -- chapter_48_line_3: GL tag task formulation.
  have row_3 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_48_line_11: GL tag expansion.
  have row_11 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_3
  -- chapter_48_line_12: GL tag disintegration.
  have row_12 : (succ zero one) := by
    exact row_11.1.1.2
  -- chapter_48_line_10: GL tag disintegration.
  have row_10 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_11.1.1.1
  -- chapter_48_line_20: GL tag expansion.
  have row_20 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_10
  -- chapter_48_line_19: GL tag disintegration.
  have row_19 : (gl_fXYZ mul N N N) := by
    exact row_20.1.1.1.2
  -- chapter_48_line_18: GL tag expansion.
  have row_18 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_19
  -- chapter_48_line_47: GL tag disintegration.
  have row_47 : (gl_implication14 N N mul) := by
    exact row_18.2
  -- chapter_48_line_46: GL tag expansion.
  have row_46 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (∀ (w3 : α), ((mul w1 w2 w3) → (∀ (w4 : α), ((mul w1 w2 w4) → (w3 = w4))))))))) := by
    simpa only [gl_implication14] using row_47
  -- chapter_48_line_37: GL tag disintegration.
  have row_37 : (gl_implication8 mul N) := by
    exact row_18.1.1.1.1
  -- chapter_48_line_36: GL tag expansion.
  have row_36 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_37
  -- chapter_48_line_39: GL tag implication.
  have row_39 : (N v3) := by
    apply row_36
    exact row_40
  -- chapter_48_line_35: GL tag implication.
  have row_35 : (N v2) := by
    apply row_36
    exact row_38
  -- chapter_48_line_23: GL tag disintegration.
  have row_23 : (gl_implication10 mul N) := by
    exact row_18.1.1.2
  -- chapter_48_line_22: GL tag expansion.
  have row_22 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication10] using row_23
  -- chapter_48_line_21: GL tag implication.
  have row_21 : (N v5) := by
    apply row_22
    exact row_24
  -- chapter_48_line_17: GL tag disintegration.
  have row_17 : (gl_implication13 N N N mul) := by
    exact row_18.1.2
  -- chapter_48_line_16: GL tag expansion.
  have row_16 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication13] using row_17
  -- chapter_48_line_34: GL tag implication.
  have row_34 : (gl_existence1 N v2 v3 mul) := by
    apply row_16
    exact row_35
    exact row_39
  -- chapter_48_line_33: GL tag expansion.
  have row_33 : (¬ (∀ (v13 : α), ((N v13) → (¬ (mul v2 v3 v13))))) := by
    simpa only [gl_existence1] using row_34
  have exists_row_33 : ∃ (v13 : α), ((N v13) ∧ (mul v2 v3 v13)) := existsAndOfNotForallImpNot row_33
  obtain ⟨v13, witness_row_33⟩ := exists_row_33
  -- chapter_48_line_32: GL tag disintegration.
  have row_32 : (mul v2 v3 v13) := by
    exact witness_row_33.2
  -- chapter_48_line_15: GL tag implication.
  have row_15 : (gl_existence1 N v5 v8 mul) := by
    apply row_16
    exact row_21
    exact row_25
  -- chapter_48_line_14: GL tag expansion.
  have row_14 : (¬ (∀ (v7 : α), ((N v7) → (¬ (mul v5 v8 v7))))) := by
    simpa only [gl_existence1] using row_15
  have exists_row_14 : ∃ (v7 : α), ((N v7) ∧ (mul v5 v8 v7)) := existsAndOfNotForallImpNot row_14
  obtain ⟨v7, witness_row_14⟩ := exists_row_14
  -- chapter_48_line_13: GL tag disintegration.
  have row_13 : (mul v5 v8 v7) := by
    exact witness_row_14.2
  -- chapter_48_line_7: GL tag implication.
  have row_7 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_8
    exact row_10
    exact row_12
  have rule_row_50 := external_peano_externals_36_010 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_48_line_50: GL tag implication.
  have row_50 : (mul v8 v1 v2) := by
    apply rule_row_50
    exact row_51
  have rule_row_49 := external_peano_externals_36_010 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_48_line_49: GL tag implication.
  have row_49 : (mul v3 v1 v5) := by
    apply rule_row_49
    exact row_24
  have rule_row_48 := external_peano_externals_36_009 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_48_line_48: GL tag implication.
  have row_48 : (mul v2 v3 v7) := by
    apply rule_row_48
    exact row_49
    exact row_50
    exact row_13
  -- chapter_48_line_45: GL tag implication.
  have row_45 : (v13 = v7) := by
    apply row_46
    exact row_35
    exact row_39
    exact row_32
    exact row_48
  have rule_row_44 := external_peano_externals_36_010 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_48_line_44: GL tag implication.
  have row_44 : (mul v12 v3 v4) := by
    apply rule_row_44
    exact row_40
  have rule_row_43 := external_peano_externals_36_010 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_48_line_43: GL tag implication.
  have row_43 : (mul v4 v2 v6) := by
    apply rule_row_43
    exact row_38
  have rule_row_31 := external_peano_externals_36_009 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_48_line_31: GL tag implication.
  have row_31 : (mul v13 v12 v6) := by
    apply rule_row_31
    exact row_44
    exact row_32
    exact row_43
  have rule_row_30 := external_peano_externals_36_010 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_48_line_30: GL tag implication.
  have row_30 : (mul v12 v13 v6) := by
    apply rule_row_30
    exact row_31
  -- chapter_48_line_29: GL tag equality1.
  have row_29 : (mul v12 v7 v6) := by
    have equality_source := row_30
    have equality_step_1 := row_45
    cases equality_step_1
    exact equality_source
  -- chapter_48_line_28: GL tag implication.
  have row_28 : (gl_preorder N mul v7 v6) := by
    apply row_5
    exact row_29
  have rule_row_6 := external_peano_externals_36_010 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_48_line_6: GL tag implication.
  have row_6 : (mul v8 v5 v7) := by
    apply rule_row_6
    exact row_13
  -- chapter_48_line_4: GL tag implication.
  have row_4 : (gl_preorder N mul v5 v7) := by
    apply row_5
    exact row_6
  -- chapter_48_line_2: GL tag theorem.
  have row_2 := fta_source_039 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_009 external_peano_externals_36_010
  -- chapter_48_line_1: GL tag implication.
  have row_1 : (gl_preorder N mul v5 v6) := by
    apply row_2
    exact row_4
    exact row_28
  exact row_1

theorem fta_source_042
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_010 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_5 x_7 x_8 x_9) → (x_5 x_8 x_7 x_9))))))
    (external_peano_externals_36_017 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_6 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_021 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_6 x_8) → (x_3 x_7 x_8))))))
    (external_peano_externals_36_015 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_3 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_022 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_6 x_8) → (x_4 x_6 x_7 x_8))))))
    (external_peano_externals_36_006 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (x_4 x_8 x_7 x_9))))))
    : (∀ (v1 : α) (v2 : α), ((gl_preorder N mul v1 v2) → ((gl_preorder N add one v2) → (gl_preorder N add v1 v2)))) := by
  intro v1
  intro v2
  intro premise_1
  intro premise_2
  -- chapter_54_line_11: GL tag theorem.
  have row_11 := fta_source_010 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_010
  -- chapter_54_line_9: GL tag task formulation.
  have row_9 : (gl_preorder N add one v2) := by
    exact premise_2
  -- chapter_54_line_8: GL tag theorem.
  have row_8 := fta_source_041 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_010 external_peano_externals_36_017 external_peano_externals_36_021 external_peano_externals_36_015 external_peano_externals_36_022
  -- chapter_54_line_6: GL tag task formulation.
  have row_6 : (gl_preorder N mul v1 v2) := by
    exact premise_1
  -- chapter_54_line_5: GL tag expansion.
  have row_5 : (¬ (∀ (v3 : α), ((N v3) → (¬ (mul v1 v3 v2))))) := by
    simpa only [gl_preorder] using row_6
  have exists_row_5 : ∃ (v3 : α), ((N v3) ∧ (mul v1 v3 v2)) := existsAndOfNotForallImpNot row_5
  obtain ⟨v3, witness_row_5⟩ := exists_row_5
  -- chapter_54_line_4: GL tag disintegration.
  have row_4 : (mul v1 v3 v2) := by
    exact witness_row_5.2
  -- chapter_54_line_3: GL tag task formulation.
  have row_3 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_54_line_12: GL tag anchor handling.
  have row_12 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact row_3
  -- chapter_54_line_10: GL tag implication.
  have row_10 : (gl_preorder N mul v3 v2) := by
    apply row_11
    exact row_4
  -- chapter_54_line_7: GL tag implication.
  have row_7 : (gl_preorder N add one v3) := by
    apply row_8
    exact row_10
    exact row_9
  -- chapter_54_line_2: GL tag theorem.
  have row_2 := fta_source_069 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_017 external_peano_externals_36_021 external_peano_externals_36_006
  -- chapter_54_line_1: GL tag implication.
  have row_1 : (gl_preorder N add v1 v2) := by
    apply row_2
    exact row_7
    exact row_4
  exact row_1

theorem fta_source_045
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_009 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (∀ (w4 : α) (w5 : α), ((mul w4 w2 w5) → (∀ (w6 : α), ((mul w3 w4 w6) → (mul w5 w1 w6))))))))))
    (external_peano_externals_36_010 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_5 x_7 x_8 x_9) → (x_5 x_8 x_7 x_9))))))
    (external_peano_externals_36_007 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_5 x_7 x_8 x_9) → (∀ (x_10 : α) (x_11 : α) (x_12 : α), ((x_5 x_10 x_11 x_12) → ((x_5 x_8 x_7 x_11) → (x_5 x_10 x_9 x_12)))))))))
    (external_peano_externals_36_029 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((gl_preorder x_1 x_4 x_7 x_8) → ((gl_preorder x_1 x_4 x_8 x_7) → (x_7 = x_8)))))))
    (external_peano_externals_36_017 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_6 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_021 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_6 x_8) → (x_3 x_7 x_8))))))
    (external_peano_externals_36_015 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_3 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_022 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_6 x_8) → (x_4 x_6 x_7 x_8))))))
    (external_peano_externals_36_033 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α), ((x_1 x_7) → (∀ (x_8 : α), ((x_5 x_6 x_8 x_7) → (x_7 = x_8))))))))
    (external_peano_externals_36_008 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (∀ (w4 : α) (w5 : α), ((mul w4 w1 w5) → (∀ (w6 : α), ((mul w4 w2 w6) → (∀ (w7 : α), ((mul w4 w3 w7) → (add w5 w6 w7))))))))))))
    : (∀ (v1 : α) (v2 : α), ((gl_preorder N mul v1 v2) → ((gl_preorder N mul v2 v1) → ((gl_preorder N add one v2) → (v1 = v2))))) := by
  intro v1
  intro v2
  intro premise_1
  intro premise_2
  intro premise_3
  -- chapter_57_line_48: GL tag theorem.
  have row_48 := fta_source_010 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_010
  -- chapter_57_line_46: GL tag theorem.
  have row_46 := fta_source_036 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_009 external_peano_externals_36_010
  -- chapter_57_line_44: GL tag task formulation.
  have row_44 : (gl_preorder N add one v2) := by
    exact premise_3
  -- chapter_57_line_43: GL tag theorem.
  have row_43 := fta_source_041 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_010 external_peano_externals_36_017 external_peano_externals_36_021 external_peano_externals_36_015 external_peano_externals_36_022
  -- chapter_57_line_41: GL tag task formulation.
  have row_41 : (gl_preorder N mul v2 v1) := by
    exact premise_2
  -- chapter_57_line_40: GL tag expansion.
  have row_40 : (¬ (∀ (v4 : α), ((N v4) → (¬ (mul v2 v4 v1))))) := by
    simpa only [gl_preorder] using row_41
  have exists_row_40 : ∃ (v4 : α), ((N v4) ∧ (mul v2 v4 v1)) := existsAndOfNotForallImpNot row_40
  obtain ⟨v4, witness_row_40⟩ := exists_row_40
  -- chapter_57_line_53: GL tag disintegration.
  have row_53 : (N v4) := by
    exact witness_row_40.1
  -- chapter_57_line_39: GL tag disintegration.
  have row_39 : (mul v2 v4 v1) := by
    exact witness_row_40.2
  -- chapter_57_line_35: GL tag variable copy.
  have row_35 : (v1 = v1) := by
    rfl
  -- chapter_57_line_57: GL tag symmetry of equality.
  have row_57 : (v1 = v1) := by
    exact Eq.symm row_35
  -- chapter_57_line_22: GL tag task formulation.
  have row_22 : (gl_preorder N mul v1 v2) := by
    exact premise_1
  -- chapter_57_line_21: GL tag expansion.
  have row_21 : (¬ (∀ (v9 : α), ((N v9) → (¬ (mul v1 v9 v2))))) := by
    simpa only [gl_preorder] using row_22
  have exists_row_21 : ∃ (v9 : α), ((N v9) ∧ (mul v1 v9 v2)) := existsAndOfNotForallImpNot row_21
  obtain ⟨v9, witness_row_21⟩ := exists_row_21
  -- chapter_57_line_31: GL tag disintegration.
  have row_31 : (N v9) := by
    exact witness_row_21.1
  -- chapter_57_line_20: GL tag disintegration.
  have row_20 : (mul v1 v9 v2) := by
    exact witness_row_21.2
  -- chapter_57_line_37: GL tag equality1.
  have row_37 : (mul v1 v9 v2) := by
    have equality_source := row_20
    have equality_step_1 := row_35
    cases equality_step_1
    exact equality_source
  -- chapter_57_line_13: GL tag theorem.
  have row_13 := fta_source_049 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_033
  -- chapter_57_line_10: GL tag theorem.
  have row_10 := fta_source_030 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_010 external_peano_externals_36_008
  -- chapter_57_line_7: GL tag task formulation.
  have row_7 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_57_line_60: GL tag implication.
  have row_60 : (gl_preorder N mul v9 v2) := by
    apply row_48
    exact row_37
  -- chapter_57_line_59: GL tag implication.
  have row_59 : (gl_preorder N add one v9) := by
    apply row_43
    exact row_60
    exact row_44
  -- chapter_57_line_11: GL tag anchor handling.
  have row_11 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact row_7
  -- chapter_57_line_6: GL tag expansion.
  have row_6 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_7
  -- chapter_57_line_8: GL tag disintegration.
  have row_8 : (succ zero one) := by
    exact row_6.1.1.2
  -- chapter_57_line_5: GL tag disintegration.
  have row_5 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_6.1.1.1
  -- chapter_57_line_19: GL tag expansion.
  have row_19 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_5
  -- chapter_57_line_18: GL tag disintegration.
  have row_18 : (gl_fXYZ mul N N N) := by
    exact row_19.1.1.1.2
  -- chapter_57_line_17: GL tag expansion.
  have row_17 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_18
  -- chapter_57_line_30: GL tag disintegration.
  have row_30 : (gl_implication8 mul N) := by
    exact row_17.1.1.1.1
  -- chapter_57_line_29: GL tag expansion.
  have row_29 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_30
  -- chapter_57_line_58: GL tag implication.
  have row_58 : (N v1) := by
    apply row_29
    exact row_37
  -- chapter_57_line_56: GL tag implication.
  have row_56 : (mul one v1 v1) := by
    apply row_13
    exact row_58
    exact row_57
  -- chapter_57_line_28: GL tag implication.
  have row_28 : (N v1) := by
    apply row_29
    exact row_20
  -- chapter_57_line_27: GL tag disintegration.
  have row_27 : (gl_implication13 N N N mul) := by
    exact row_17.1.2
  -- chapter_57_line_26: GL tag expansion.
  have row_26 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication13] using row_27
  -- chapter_57_line_52: GL tag implication.
  have row_52 : (gl_existence1 N v4 v9 mul) := by
    apply row_26
    exact row_53
    exact row_31
  -- chapter_57_line_51: GL tag expansion.
  have row_51 : (¬ (∀ (v11 : α), ((N v11) → (¬ (mul v4 v9 v11))))) := by
    simpa only [gl_existence1] using row_52
  have exists_row_51 : ∃ (v11 : α), ((N v11) ∧ (mul v4 v9 v11)) := existsAndOfNotForallImpNot row_51
  obtain ⟨v11, witness_row_51⟩ := exists_row_51
  -- chapter_57_line_50: GL tag disintegration.
  have row_50 : (mul v4 v9 v11) := by
    exact witness_row_51.2
  -- chapter_57_line_25: GL tag implication.
  have row_25 : (gl_existence1 N v1 v9 mul) := by
    apply row_26
    exact row_28
    exact row_31
  -- chapter_57_line_24: GL tag expansion.
  have row_24 : (¬ (∀ (v3 : α), ((N v3) → (¬ (mul v1 v9 v3))))) := by
    simpa only [gl_existence1] using row_25
  have exists_row_24 : ∃ (v3 : α), ((N v3) ∧ (mul v1 v9 v3)) := existsAndOfNotForallImpNot row_24
  obtain ⟨v3, witness_row_24⟩ := exists_row_24
  -- chapter_57_line_32: GL tag disintegration.
  have row_32 : (N v3) := by
    exact witness_row_24.1
  -- chapter_57_line_23: GL tag disintegration.
  have row_23 : (mul v1 v9 v3) := by
    exact witness_row_24.2
  -- chapter_57_line_34: GL tag equality1.
  have row_34 : (mul v1 v9 v3) := by
    have equality_source := row_23
    have equality_step_1 := row_35
    cases equality_step_1
    exact equality_source
  -- chapter_57_line_16: GL tag disintegration.
  have row_16 : (gl_implication14 N N mul) := by
    exact row_17.2
  -- chapter_57_line_15: GL tag expansion.
  have row_15 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (∀ (w3 : α), ((mul w1 w2 w3) → (∀ (w4 : α), ((mul w1 w2 w4) → (w3 = w4))))))))) := by
    simpa only [gl_implication14] using row_16
  -- chapter_57_line_14: GL tag implication.
  have row_14 : (v3 = v2) := by
    apply row_15
    exact row_28
    exact row_31
    exact row_23
    exact row_20
  -- chapter_57_line_12: GL tag implication.
  have row_12 : (mul one v3 v2) := by
    apply row_13
    exact row_32
    exact row_14
  -- chapter_57_line_4: GL tag expansion for integration.
  have row_4 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_57_line_3: GL tag reformulation for integration and.
  have row_3 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_57_line_2: GL tag implication.
  have row_2 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_3
    exact row_5
    exact row_8
  have rule_row_54 := external_peano_externals_36_009 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_57_line_54: GL tag implication.
  have row_54 : (mul v11 v1 v1) := by
    apply rule_row_54
    exact row_37
    exact row_50
    exact row_39
  have rule_row_49 := external_peano_externals_36_009 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_57_line_49: GL tag implication.
  have row_49 : (mul v3 v4 v1) := by
    apply rule_row_49
    exact row_50
    exact row_34
    exact row_54
  -- chapter_57_line_47: GL tag implication.
  have row_47 : (gl_preorder N mul v4 v1) := by
    apply row_48
    exact row_49
  -- chapter_57_line_45: GL tag implication.
  have row_45 : (gl_preorder N mul v4 v2) := by
    apply row_46
    exact row_47
    exact row_20
  -- chapter_57_line_42: GL tag implication.
  have row_42 : (gl_preorder N add one v4) := by
    apply row_43
    exact row_45
    exact row_44
  have rule_row_38 := external_peano_externals_36_010 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_57_line_38: GL tag implication.
  have row_38 : (mul v4 v2 v1) := by
    apply rule_row_38
    exact row_39
  have rule_row_36 := external_peano_externals_36_010 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_57_line_36: GL tag implication.
  have row_36 : (mul v9 v1 v2) := by
    apply rule_row_36
    exact row_37
  -- chapter_57_line_55: GL tag implication.
  have row_55 : (gl_preorder N add v1 v2) := by
    apply row_10
    exact row_59
    exact row_56
    exact row_36
  have rule_row_33 := external_peano_externals_36_007 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_57_line_33: GL tag implication.
  have row_33 : (mul v4 v3 v1) := by
    apply rule_row_33
    exact row_34
    exact row_38
    exact row_36
  -- chapter_57_line_9: GL tag implication.
  have row_9 : (gl_preorder N add v2 v1) := by
    apply row_10
    exact row_42
    exact row_12
    exact row_33
  have rule_row_1 := external_peano_externals_36_029 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_57_line_1: GL tag implication.
  have row_1 : (v1 = v2) := by
    apply rule_row_1
    exact row_55
    exact row_9
  exact row_1

theorem fta_source_048
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_006 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (x_4 x_8 x_7 x_9))))))
    (external_peano_externals_36_010 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_5 x_7 x_8 x_9) → (x_5 x_8 x_7 x_9))))))
    (external_peano_externals_36_008 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (∀ (w4 : α) (w5 : α), ((mul w4 w1 w5) → (∀ (w6 : α), ((mul w4 w2 w6) → (∀ (w7 : α), ((mul w4 w3 w7) → (add w5 w6 w7))))))))))))
    (external_peano_externals_36_005 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → ((x_8 = x_9) → (x_2 = x_7)))))))
    (external_peano_externals_36_017 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_6 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_021 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_6 x_8) → (x_3 x_7 x_8))))))
    (external_peano_externals_36_035 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (gl_existence11 N one succ))))
    (external_peano_externals_36_024 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_8 x_2) → ((x_1 x_8) → (x_2 = x_7)))))))
    (external_peano_externals_36_025 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_8 x_2) → (x_4 x_8 x_7 x_2))))))
    : (∀ (v1 : α) (v2 : α), ((gl_strictOrder N add v1 v2) → (∀ (v3 : α), ((gl_preorder N add one v3) → (∀ (v4 : α), ((mul v1 v3 v4) → (∀ (v5 : α), ((mul v2 v3 v5) → (gl_strictOrder N add v4 v5))))))))) := by
  intro v1
  intro v2
  intro premise_1
  intro v3
  intro premise_2
  intro v4
  intro premise_3
  intro v5
  intro premise_4
  apply Classical.byContradiction
  intro reductio
  -- chapter_60_line_50: GL tag task formulation.
  have row_50 : (mul v1 v3 v4) := by
    exact premise_3
  -- chapter_60_line_48: GL tag task formulation.
  have row_48 : (mul v2 v3 v5) := by
    exact premise_4
  -- chapter_60_line_45: GL tag expansion for integration.
  have row_45 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_60_line_44: GL tag reformulation for integration and.
  have row_44 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_60_line_41: GL tag theorem.
  have row_41 := fta_source_002 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_60_line_39: GL tag task formulation.
  have row_39 : (¬ (gl_strictOrder N add v4 v5)) := by
    exact reductio
  -- chapter_60_line_38: GL tag theorem.
  have row_38 := fta_source_032 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_60_line_15: GL tag task formulation.
  have row_15 : (gl_preorder N add one v3) := by
    exact premise_2
  -- chapter_60_line_33: GL tag expansion.
  have row_33 : (¬ (∀ (v11 : α), ((N v11) → (¬ (add one v11 v3))))) := by
    simpa only [gl_preorder] using row_15
  have exists_row_33 : ∃ (v11 : α), ((N v11) ∧ (add one v11 v3)) := existsAndOfNotForallImpNot row_33
  obtain ⟨v11, witness_row_33⟩ := exists_row_33
  -- chapter_60_line_32: GL tag disintegration.
  have row_32 : (add one v11 v3) := by
    exact witness_row_33.2
  -- chapter_60_line_14: GL tag theorem.
  have row_14 := fta_source_066 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_017 external_peano_externals_36_021
  -- chapter_60_line_12: GL tag theorem.
  have row_12 := fta_source_023 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_035 external_peano_externals_36_024 external_peano_externals_36_025
  -- chapter_60_line_10: GL tag task formulation.
  have row_10 : (gl_strictOrder N add v1 v2) := by
    exact premise_1
  -- chapter_60_line_9: GL tag expansion.
  have row_9 : ((gl_preorder N add v1 v2) ∧ (¬ (v1 = v2))) := by
    simpa only [gl_strictOrder] using row_10
  -- chapter_60_line_8: GL tag disintegration.
  have row_8 : (gl_preorder N add v1 v2) := by
    exact row_9.1
  -- chapter_60_line_7: GL tag expansion.
  have row_7 : (¬ (∀ (v6 : α), ((N v6) → (¬ (add v1 v6 v2))))) := by
    simpa only [gl_preorder] using row_8
  have exists_row_7 : ∃ (v6 : α), ((N v6) ∧ (add v1 v6 v2)) := existsAndOfNotForallImpNot row_7
  obtain ⟨v6, witness_row_7⟩ := exists_row_7
  -- chapter_60_line_34: GL tag disintegration.
  have row_34 : (N v6) := by
    exact witness_row_7.1
  -- chapter_60_line_6: GL tag disintegration.
  have row_6 : (add v1 v6 v2) := by
    exact witness_row_7.2
  -- chapter_60_line_5: GL tag task formulation.
  have row_5 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_60_line_26: GL tag expansion.
  have row_26 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_5
  -- chapter_60_line_46: GL tag disintegration.
  have row_46 : (succ zero one) := by
    exact row_26.1.1.2
  -- chapter_60_line_25: GL tag disintegration.
  have row_25 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_26.1.1.1
  -- chapter_60_line_43: GL tag implication.
  have row_43 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_44
    exact row_25
    exact row_46
  have rule_row_52 := external_peano_externals_36_006 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_60_line_52: GL tag implication.
  have row_52 : (add v6 v1 v2) := by
    apply rule_row_52
    exact row_6
  have rule_row_49 := external_peano_externals_36_010 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_60_line_49: GL tag implication.
  have row_49 : (mul v3 v1 v4) := by
    apply rule_row_49
    exact row_50
  have rule_row_47 := external_peano_externals_36_010 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_60_line_47: GL tag implication.
  have row_47 : (mul v3 v2 v5) := by
    apply rule_row_47
    exact row_48
  -- chapter_60_line_24: GL tag expansion.
  have row_24 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_25
  -- chapter_60_line_31: GL tag disintegration.
  have row_31 : (gl_fXYZ add N N N) := by
    exact row_24.1.1.1.1.1.1.1.1.2
  -- chapter_60_line_30: GL tag expansion.
  have row_30 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_31
  -- chapter_60_line_29: GL tag disintegration.
  have row_29 : (gl_implication10 add N) := by
    exact row_30.1.1.2
  -- chapter_60_line_28: GL tag expansion.
  have row_28 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication10] using row_29
  -- chapter_60_line_27: GL tag implication.
  have row_27 : (N v3) := by
    apply row_28
    exact row_32
  -- chapter_60_line_23: GL tag disintegration.
  have row_23 : (gl_fXYZ mul N N N) := by
    exact row_24.1.1.1.2
  -- chapter_60_line_22: GL tag expansion.
  have row_22 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_23
  -- chapter_60_line_21: GL tag disintegration.
  have row_21 : (gl_implication13 N N N mul) := by
    exact row_22.1.2
  -- chapter_60_line_20: GL tag expansion.
  have row_20 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication13] using row_21
  -- chapter_60_line_19: GL tag implication.
  have row_19 : (gl_existence1 N v3 v6 mul) := by
    apply row_20
    exact row_27
    exact row_34
  -- chapter_60_line_18: GL tag expansion.
  have row_18 : (¬ (∀ (v7 : α), ((N v7) → (¬ (mul v3 v6 v7))))) := by
    simpa only [gl_existence1] using row_19
  have exists_row_18 : ∃ (v7 : α), ((N v7) ∧ (mul v3 v6 v7)) := existsAndOfNotForallImpNot row_18
  obtain ⟨v7, witness_row_18⟩ := exists_row_18
  -- chapter_60_line_17: GL tag disintegration.
  have row_17 : (mul v3 v6 v7) := by
    exact witness_row_18.2
  have rule_row_51 := external_peano_externals_36_008 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_60_line_51: GL tag implication.
  have row_51 : (add v7 v4 v5) := by
    apply rule_row_51
    exact row_52
    exact row_17
    exact row_49
    exact row_47
  have rule_row_42 := external_peano_externals_36_008 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_60_line_42: GL tag implication.
  have row_42 : (add v4 v7 v5) := by
    apply rule_row_42
    exact row_6
    exact row_49
    exact row_17
    exact row_47
  -- chapter_60_line_13: GL tag implication.
  have row_13 : (¬ (zero = v3)) := by
    apply row_14
    exact row_15
  -- chapter_60_line_4: GL tag anchor handling.
  have row_4 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact row_5
  -- chapter_60_line_40: GL tag implication.
  have row_40 : (gl_preorder N add v4 v5) := by
    apply row_41
    exact row_42
  -- chapter_60_line_37: GL tag implication.
  have row_37 : (v4 = v5) := by
    apply row_38
    exact row_40
    exact row_39
  have rule_row_36 := external_peano_externals_36_005 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_60_line_36: GL tag implication.
  have row_36 : (zero = v7) := by
    apply rule_row_36
    exact row_51
    exact row_37
  -- chapter_60_line_35: GL tag symmetry of equality.
  have row_35 : (v7 = zero) := by
    exact Eq.symm row_36
  -- chapter_60_line_16: GL tag equality1.
  have row_16 : (mul v3 v6 zero) := by
    have equality_source := row_17
    have equality_step_1 := row_35
    cases equality_step_1
    exact equality_source
  -- chapter_60_line_11: GL tag implication.
  have row_11 : (zero = v6) := by
    apply row_12
    exact row_16
    exact row_27
    exact row_13
  -- chapter_60_line_3: GL tag theorem.
  have row_3 := fta_source_047 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_60_line_2: GL tag implication.
  have row_2 : (¬ (zero = v6)) := by
    apply row_3
    exact row_10
    exact row_6
  -- chapter_60_line_1: GL tag contradiction.
  have row_1 : (gl_strictOrder N add v4 v5) := by
    exact False.elim (row_2 row_11)
  exact reductio row_1

theorem fta_source_053
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_019 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_2 x_8) → (x_4 x_2 x_7 x_8))))))
    (external_peano_externals_36_006 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (x_4 x_8 x_7 x_9))))))
    (external_peano_externals_36_015 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_3 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_034 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (∀ (w1 : α), ((N w1) → (gl_or2 w1 zero N succ))))))
    (external_peano_externals_36_003 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (∀ (x_10 : α), ((x_3 x_8 x_10) → (∀ (x_11 : α), ((x_3 x_11 x_7) → (x_4 x_10 x_11 x_9))))))))))
    (external_peano_externals_36_004 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (∀ (x_10 : α), ((x_4 x_7 x_10 x_9) → (x_8 = x_10))))))))
    : (∀ (v1 : α), ((N v1) → (∀ (v2 : α), ((N v2) → ((¬ (gl_strictOrder N add v1 v2)) → ((¬ (v1 = v2)) → (gl_strictOrder N add v2 v1))))))) := by
  intro v1
  intro premise_1
  intro v2
  intro premise_2
  intro premise_3
  intro premise_4
  apply Classical.byContradiction
  intro reductio
  -- chapter_67_line_24: GL tag task formulation.
  have row_24 : (N v2) := by
    exact premise_2
  -- chapter_67_line_22: GL tag theorem.
  have row_22 := fta_source_052 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_019 external_peano_externals_36_006 external_peano_externals_36_015 external_peano_externals_36_034 external_peano_externals_36_003 external_peano_externals_36_004
  -- chapter_67_line_20: GL tag task formulation.
  have row_20 : (¬ (gl_strictOrder N add v1 v2)) := by
    exact premise_3
  -- chapter_67_line_18: GL tag theorem.
  have row_18 := fta_source_032 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_67_line_14: GL tag task formulation.
  have row_14 : (N v1) := by
    exact premise_1
  -- chapter_67_line_13: GL tag task formulation.
  have row_13 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_67_line_12: GL tag variable copy.
  have row_12 : (v1 = v1) := by
    rfl
  -- chapter_67_line_25: GL tag equality1.
  have row_25 : (N v1) := by
    have equality_source := row_14
    have equality_step_1 := row_12
    cases equality_step_1
    exact equality_source
  -- chapter_67_line_19: GL tag equality1.
  have row_19 : (¬ (gl_strictOrder N add v1 v2)) := by
    have equality_source := row_20
    have equality_step_1 := row_12
    cases equality_step_1
    exact equality_source
  -- chapter_67_line_15: GL tag symmetry of equality.
  have row_15 : (v1 = v1) := by
    exact Eq.symm row_12
  -- chapter_67_line_11: GL tag theorem.
  have row_11 := fta_source_050 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_67_line_10: GL tag implication.
  have row_10 : (gl_preorder N add v1 v1) := by
    apply row_11
    exact row_14
    exact row_12
  -- chapter_67_line_9: GL tag equality1.
  have row_9 : (gl_preorder N add v1 v1) := by
    have equality_source := row_10
    have equality_step_1 := row_15
    cases equality_step_1
    exact equality_source
  -- chapter_67_line_7: GL tag task formulation.
  have row_7 : (¬ (v1 = v2)) := by
    exact premise_4
  -- chapter_67_line_6: GL tag symmetry of inequality.
  have row_6 : (¬ (v2 = v1)) := by
    exact fun equality => row_7 (Eq.symm equality)
  -- chapter_67_line_5: GL tag task formulation.
  have row_5 : (¬ (gl_strictOrder N add v2 v1)) := by
    exact reductio
  -- chapter_67_line_4: GL tag expansion.
  have row_4 : (¬ ((gl_preorder N add v2 v1) ∧ (¬ (v2 = v1)))) := by
    simpa only [gl_strictOrder] using row_5
  -- chapter_67_line_3: GL tag disintegration.
  have row_3 : ((¬ (v2 = v1)) → (¬ (gl_preorder N add v2 v1))) := by
    classical
    intro projection_premise
    intro projection_counterexample
    exact row_4 ⟨projection_counterexample, projection_premise⟩
  -- chapter_67_line_2: GL tag implication.
  have row_2 : (¬ (gl_preorder N add v2 v1)) := by
    apply row_3
    exact row_6
  -- chapter_67_line_23: GL tag equality1.
  have row_23 : (¬ (gl_preorder N add v2 v1)) := by
    have equality_source := row_2
    have equality_step_1 := row_12
    cases equality_step_1
    exact equality_source
  -- chapter_67_line_21: GL tag implication.
  have row_21 : (gl_preorder N add v1 v2) := by
    apply row_22
    exact row_24
    exact row_25
    exact row_23
  -- chapter_67_line_17: GL tag implication.
  have row_17 : (v1 = v2) := by
    apply row_18
    exact row_21
    exact row_19
  -- chapter_67_line_16: GL tag equality2.
  have row_16 : (v1 = v2) := by
    exact Eq.trans row_12 row_17
  -- chapter_67_line_8: GL tag equality1.
  have row_8 : (gl_preorder N add v2 v1) := by
    have equality_source := row_9
    have equality_step_1 := row_16
    cases equality_step_1
    exact equality_source
  -- chapter_67_line_1: GL tag contradiction.
  have row_1 : (gl_strictOrder N add v2 v1) := by
    exact False.elim (row_2 row_8)
  exact reductio row_1

theorem fta_source_056
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_019 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_2 x_8) → (x_4 x_2 x_7 x_8))))))
    (external_peano_externals_36_006 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (x_4 x_8 x_7 x_9))))))
    (external_peano_externals_36_015 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_3 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_034 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (∀ (w1 : α), ((N w1) → (gl_or2 w1 zero N succ))))))
    (external_peano_externals_36_003 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (∀ (x_10 : α), ((x_3 x_8 x_10) → (∀ (x_11 : α), ((x_3 x_11 x_7) → (x_4 x_10 x_11 x_9))))))))))
    (external_peano_externals_36_004 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (∀ (x_10 : α), ((x_4 x_7 x_10 x_9) → (x_8 = x_10))))))))
    : (∀ (v1 : α), ((N v1) → (∀ (v2 : α), ((N v2) → (gl_or5 N add v1 v2))))) := by
  intro v1
  intro premise_1
  intro v2
  intro premise_2
  have or_parent_1 := fta_source_052 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_019 external_peano_externals_36_006 external_peano_externals_36_015 external_peano_externals_36_034 external_peano_externals_36_003 external_peano_externals_36_004
  -- chapter_70_line_1: GL tag or theorem.
  have row_1 : (∀ (v1 : α), ((N v1) → (∀ (v2 : α), ((N v2) → (gl_or5 N add v1 v2))))) := by
    classical
    intro v1
    intro or_parent_premise_1
    intro v2
    intro or_parent_premise_2
    simp only [gl_or5]
    by_cases or_case_1 : (gl_preorder N add v1 v2)
    · exact Or.inl (or_case_1)
    ·
      exact Or.inr ((or_parent_1 v1 or_parent_premise_1 v2 or_parent_premise_2 or_case_1))
  solve_by_elim [row_1]

theorem fta_source_065
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_035 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (gl_existence11 N one succ))))
    (external_peano_externals_36_033 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α), ((x_1 x_7) → (∀ (x_8 : α), ((x_5 x_6 x_8 x_7) → (x_7 = x_8))))))))
    (external_peano_externals_36_018 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_5 x_6 x_7 x_8) → (x_5 x_7 x_6 x_8))))))
    (external_peano_externals_36_009 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (∀ (w4 : α) (w5 : α), ((mul w4 w2 w5) → (∀ (w6 : α), ((mul w3 w4 w6) → (mul w5 w1 w6))))))))))
    (external_peano_externals_36_010 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_5 x_7 x_8 x_9) → (x_5 x_8 x_7 x_9))))))
    : (∀ (v1 : α), ((N v1) → (gl_preorder N mul v1 zero))) := by
  intro v1
  intro premise_1
  -- chapter_87_line_37: GL tag theorem.
  have row_37 := fta_source_051 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_033 external_peano_externals_36_018
  -- chapter_87_line_35: GL tag variable copy.
  have row_35 : (v1 = v1) := by
    rfl
  -- chapter_87_line_28: GL tag expansion for integration.
  have row_28 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_87_line_27: GL tag reformulation for integration and.
  have row_27 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_87_line_18: GL tag task formulation.
  have row_18 : (N v1) := by
    exact premise_1
  -- chapter_87_line_4: GL tag task formulation.
  have row_4 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_87_line_36: GL tag implication.
  have row_36 : (gl_preorder N mul v1 v1) := by
    apply row_37
    exact row_18
    exact row_35
  -- chapter_87_line_16: GL tag expansion.
  have row_16 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_4
  -- chapter_87_line_34: GL tag disintegration.
  have row_34 : (succ one two) := by
    exact row_16.1.2
  -- chapter_87_line_22: GL tag disintegration.
  have row_22 : (succ zero one) := by
    exact row_16.1.1.2
  -- chapter_87_line_15: GL tag disintegration.
  have row_15 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_16.1.1.1
  -- chapter_87_line_26: GL tag implication.
  have row_26 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_27
    exact row_15
    exact row_22
  have rule_row_25 := external_peano_externals_36_035 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_87_line_25: GL tag implication.
  have row_25 : (gl_existence11 N one succ) := by
    apply rule_row_25
  -- chapter_87_line_24: GL tag expansion.
  have row_24 : (¬ (∀ (v4 : α), ((N v4) → (¬ (succ v4 one))))) := by
    simpa only [gl_existence11] using row_25
  have exists_row_24 : ∃ (v4 : α), ((N v4) ∧ (succ v4 one)) := existsAndOfNotForallImpNot row_24
  obtain ⟨v4, witness_row_24⟩ := exists_row_24
  -- chapter_87_line_23: GL tag disintegration.
  have row_23 : (succ v4 one) := by
    exact witness_row_24.2
  -- chapter_87_line_14: GL tag expansion.
  have row_14 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_15
  -- chapter_87_line_40: GL tag disintegration.
  have row_40 : (gl_implication19 N zero mul) := by
    exact row_14.1.1.2
  -- chapter_87_line_39: GL tag expansion.
  have row_39 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((mul w1 zero w2) → (w2 = zero))))) := by
    simpa only [gl_implication19] using row_40
  -- chapter_87_line_33: GL tag disintegration.
  have row_33 : (gl_fXY succ N N) := by
    exact row_14.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_87_line_32: GL tag expansion.
  have row_32 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_33
  -- chapter_87_line_31: GL tag disintegration.
  have row_31 : (gl_implication0 succ N) := by
    exact row_32.1.1.1
  -- chapter_87_line_30: GL tag expansion.
  have row_30 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_31
  -- chapter_87_line_29: GL tag implication.
  have row_29 : (N one) := by
    apply row_30
    exact row_34
  -- chapter_87_line_21: GL tag disintegration.
  have row_21 : (gl_implication7 N succ) := by
    exact row_14.1.1.1.1.1.1.1.1.1.2
  -- chapter_87_line_20: GL tag expansion.
  have row_20 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w2 w1) → (∀ (w3 : α), ((succ w3 w1) → (w2 = w3))))))) := by
    simpa only [gl_implication7] using row_21
  -- chapter_87_line_19: GL tag implication.
  have row_19 : (zero = v4) := by
    apply row_20
    exact row_29
    exact row_22
    exact row_23
  -- chapter_87_line_17: GL tag disintegration.
  have row_17 : (N zero) := by
    exact row_14.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_87_line_13: GL tag disintegration.
  have row_13 : (gl_fXYZ mul N N N) := by
    exact row_14.1.1.1.2
  -- chapter_87_line_12: GL tag expansion.
  have row_12 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_13
  -- chapter_87_line_11: GL tag disintegration.
  have row_11 : (gl_implication13 N N N mul) := by
    exact row_12.1.2
  -- chapter_87_line_10: GL tag expansion.
  have row_10 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication13] using row_11
  -- chapter_87_line_9: GL tag implication.
  have row_9 : (gl_existence1 N v1 zero mul) := by
    apply row_10
    exact row_18
    exact row_17
  -- chapter_87_line_8: GL tag expansion.
  have row_8 : (¬ (∀ (v2 : α), ((N v2) → (¬ (mul v1 zero v2))))) := by
    simpa only [gl_existence1] using row_9
  have exists_row_8 : ∃ (v2 : α), ((N v2) ∧ (mul v1 zero v2)) := existsAndOfNotForallImpNot row_8
  obtain ⟨v2, witness_row_8⟩ := exists_row_8
  -- chapter_87_line_7: GL tag disintegration.
  have row_7 : (mul v1 zero v2) := by
    exact witness_row_8.2
  -- chapter_87_line_38: GL tag implication.
  have row_38 : (v2 = zero) := by
    apply row_39
    exact row_18
    exact row_7
  -- chapter_87_line_6: GL tag equality1.
  have row_6 : (mul v1 v4 v2) := by
    have equality_source := row_7
    have equality_step_1 := row_19
    cases equality_step_1
    exact equality_source
  -- chapter_87_line_5: GL tag equality1.
  have row_5 : (mul v1 v4 v2) := by
    have equality_source := row_6
    have equality_step_1 := row_35
    cases equality_step_1
    exact equality_source
  -- chapter_87_line_3: GL tag theorem.
  have row_3 := fta_source_036 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_009 external_peano_externals_36_010
  -- chapter_87_line_2: GL tag implication.
  have row_2 : (gl_preorder N mul v1 v2) := by
    apply row_3
    exact row_36
    exact row_5
  -- chapter_87_line_1: GL tag equality1.
  have row_1 : (gl_preorder N mul v1 zero) := by
    have equality_source := row_2
    have equality_step_1 := row_38
    cases equality_step_1
    exact equality_source
  exact row_1

theorem fta_source_035
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_006 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (x_4 x_8 x_7 x_9))))))
    (external_peano_externals_36_010 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_5 x_7 x_8 x_9) → (x_5 x_8 x_7 x_9))))))
    (external_peano_externals_36_005 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → ((x_8 = x_9) → (x_2 = x_7)))))))
    (external_peano_externals_36_008 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (∀ (w4 : α) (w5 : α), ((mul w4 w1 w5) → (∀ (w6 : α), ((mul w4 w2 w6) → (∀ (w7 : α), ((mul w4 w3 w7) → (add w5 w6 w7))))))))))))
    (external_peano_externals_36_029 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((gl_preorder x_1 x_4 x_7 x_8) → ((gl_preorder x_1 x_4 x_8 x_7) → (x_7 = x_8)))))))
    (external_peano_externals_36_035 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (gl_existence11 N one succ))))
    (external_peano_externals_36_033 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α), ((x_1 x_7) → (∀ (x_8 : α), ((x_5 x_6 x_8 x_7) → (x_7 = x_8))))))))
    (external_peano_externals_36_018 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_5 x_6 x_7 x_8) → (x_5 x_7 x_6 x_8))))))
    (external_peano_externals_36_009 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (∀ (w4 : α) (w5 : α), ((mul w4 w2 w5) → (∀ (w6 : α), ((mul w3 w4 w6) → (mul w5 w1 w6))))))))))
    (external_peano_externals_36_004 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (∀ (x_10 : α), ((x_4 x_7 x_10 x_9) → (x_8 = x_10))))))))
    (external_peano_externals_36_019 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_2 x_8) → (x_4 x_2 x_7 x_8))))))
    (external_peano_externals_36_015 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_3 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_034 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (∀ (w1 : α), ((N w1) → (gl_or2 w1 zero N succ))))))
    (external_peano_externals_36_003 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (∀ (x_10 : α), ((x_3 x_8 x_10) → (∀ (x_11 : α), ((x_3 x_11 x_7) → (x_4 x_10 x_11 x_9))))))))))
    : (∀ (v1 : α) (v2 : α), ((gl_preorder N mul v1 v2) → (∀ (v3 : α) (v4 : α), ((add v2 v3 v4) → ((gl_preorder N mul v1 v4) → (gl_preorder N mul v1 v3)))))) := by
  intro v1
  intro v2
  intro premise_1
  intro v3
  intro v4
  intro premise_2
  intro premise_3
  -- chapter_45_line_45: GL tag expansion for integration.
  have row_45 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_45_line_44: GL tag reformulation for integration and.
  have row_44 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_45_line_41: GL tag theorem.
  have row_41 := fta_source_030 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_010 external_peano_externals_36_008
  -- chapter_45_line_39: GL tag theorem.
  have row_39 := fta_source_002 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_45_line_37: GL tag theorem.
  have row_37 := fta_source_033 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_029
  -- chapter_45_line_26: GL tag theorem.
  have row_26 := fta_source_065 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_035 external_peano_externals_36_033 external_peano_externals_36_018 external_peano_externals_36_009 external_peano_externals_36_010
  -- chapter_45_line_17: GL tag task formulation.
  have row_17 : (add v2 v3 v4) := by
    exact premise_2
  -- chapter_45_line_16: GL tag theorem.
  have row_16 := fta_source_003 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_008 external_peano_externals_36_004
  -- chapter_45_line_14: GL tag expansion for integration.
  have row_14 : ((gl_preorder N mul v1 v3) ↔ (¬ (∀ (v9 : α), ((N v9) → (¬ (mul v1 v9 v3)))))) := by
    exact Iff.rfl
  -- chapter_45_line_13: GL tag reformulation for integration >[bound].
  have row_13 : (∀ (v7 : α), ((N v7) → ((mul v1 v7 v3) → (gl_preorder N mul v1 v3)))) := by
    intro v7
    intro integration_premise_1
    intro integration_premise_2
    apply (row_14).2
    intro universal_counterexample
    exact universal_counterexample v7 integration_premise_1 integration_premise_2
  -- chapter_45_line_11: GL tag task formulation.
  have row_11 : (gl_preorder N mul v1 v4) := by
    exact premise_3
  -- chapter_45_line_10: GL tag expansion.
  have row_10 : (¬ (∀ (v6 : α), ((N v6) → (¬ (mul v1 v6 v4))))) := by
    simpa only [gl_preorder] using row_11
  have exists_row_10 : ∃ (v6 : α), ((N v6) ∧ (mul v1 v6 v4)) := existsAndOfNotForallImpNot row_10
  obtain ⟨v6, witness_row_10⟩ := exists_row_10
  -- chapter_45_line_19: GL tag disintegration.
  have row_19 : (mul v1 v6 v4) := by
    exact witness_row_10.2
  -- chapter_45_line_9: GL tag disintegration.
  have row_9 : (N v6) := by
    exact witness_row_10.1
  -- chapter_45_line_8: GL tag task formulation.
  have row_8 : (gl_preorder N mul v1 v2) := by
    exact premise_1
  -- chapter_45_line_7: GL tag expansion.
  have row_7 : (¬ (∀ (v5 : α), ((N v5) → (¬ (mul v1 v5 v2))))) := by
    simpa only [gl_preorder] using row_8
  have exists_row_7 : ∃ (v5 : α), ((N v5) ∧ (mul v1 v5 v2)) := existsAndOfNotForallImpNot row_7
  obtain ⟨v5, witness_row_7⟩ := exists_row_7
  -- chapter_45_line_18: GL tag disintegration.
  have row_18 : (mul v1 v5 v2) := by
    exact witness_row_7.2
  -- chapter_45_line_6: GL tag disintegration.
  have row_6 : (N v5) := by
    exact witness_row_7.1
  -- chapter_45_line_5: GL tag task formulation.
  have row_5 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_45_line_38: GL tag implication.
  have row_38 : (gl_preorder N add v2 v4) := by
    apply row_39
    exact row_17
  -- chapter_45_line_34: GL tag expansion.
  have row_34 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_5
  -- chapter_45_line_46: GL tag disintegration.
  have row_46 : (succ zero one) := by
    exact row_34.1.1.2
  -- chapter_45_line_33: GL tag disintegration.
  have row_33 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_34.1.1.1
  -- chapter_45_line_43: GL tag implication.
  have row_43 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_44
    exact row_33
    exact row_46
  have rule_row_49 := external_peano_externals_36_006 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_45_line_49: GL tag implication.
  have row_49 : (add v3 v2 v4) := by
    apply rule_row_49
    exact row_17
  have rule_row_47 := external_peano_externals_36_010 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_45_line_47: GL tag implication.
  have row_47 : (mul v6 v1 v4) := by
    apply rule_row_47
    exact row_19
  have rule_row_42 := external_peano_externals_36_010 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_45_line_42: GL tag implication.
  have row_42 : (mul v5 v1 v2) := by
    apply rule_row_42
    exact row_18
  -- chapter_45_line_32: GL tag expansion.
  have row_32 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_33
  -- chapter_45_line_31: GL tag disintegration.
  have row_31 : (gl_fXYZ mul N N N) := by
    exact row_32.1.1.1.2
  -- chapter_45_line_30: GL tag expansion.
  have row_30 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_31
  -- chapter_45_line_29: GL tag disintegration.
  have row_29 : (gl_implication8 mul N) := by
    exact row_30.1.1.1.1
  -- chapter_45_line_28: GL tag expansion.
  have row_28 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_29
  -- chapter_45_line_27: GL tag implication.
  have row_27 : (N v1) := by
    apply row_28
    exact row_19
  -- chapter_45_line_25: GL tag implication.
  have row_25 : (gl_preorder N mul v1 zero) := by
    apply row_26
    exact row_27
  -- chapter_45_line_4: GL tag compilation.
  have row_4 : (gl_implication211 (α := α)) := by
    simp only [gl_implication211]
    intro compiled_N compiled_zero compiled_succ compiled_add compiled_mul compiled_one compiled_two compiled_identity
    intro compiled_anchor
    exact fta_source_056 compiled_N compiled_zero compiled_succ compiled_add compiled_mul compiled_one compiled_two compiled_identity compiled_anchor relationalInduction external_peano_externals_36_019 external_peano_externals_36_006 external_peano_externals_36_015 external_peano_externals_36_034 external_peano_externals_36_003 external_peano_externals_36_004
  -- chapter_45_line_3: GL tag expansion.
  have row_3 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α) (two : α) (identity : GLBinaryRelation α), ((gl_AnchorFTA N zero succ add mul one two identity) → (∀ (v1 : α), ((N v1) → (∀ (v2 : α), ((N v2) → (gl_or5 N add v1 v2))))))) := by
    simpa only [gl_implication211] using row_4
  have rule_row_2 := fta_source_056 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_019 external_peano_externals_36_006 external_peano_externals_36_015 external_peano_externals_36_034 external_peano_externals_36_003 external_peano_externals_36_004
  -- chapter_45_line_2: GL tag implication.
  have row_2 : (gl_or5 N add v5 v6) := by
    apply rule_row_2
    exact row_6
    exact row_9
  -- chapter_45_line_48: GL tag or disintegration.
  have row_48 : ((gl_preorder N add v6 v5) → (gl_preorder N add v6 v5)) := by
    intro scope_premise_1
    exact scope_premise_1
  -- chapter_45_line_40: GL tag implication.
  have row_40 : ((gl_preorder N add v6 v5) → (gl_preorder N add v4 v2)) := by
    intro scope_premise_1
    have scoped_fact_2 := row_48 scope_premise_1
    apply row_41
    exact scoped_fact_2
    exact row_47
    exact row_42
  -- chapter_45_line_36: GL tag implication.
  have row_36 : ((gl_preorder N add v6 v5) → (v2 = v4)) := by
    intro scope_premise_1
    have scoped_fact_3 := row_40 scope_premise_1
    apply row_37
    exact row_38
    exact scoped_fact_3
  have rule_row_35 := external_peano_externals_36_005 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_45_line_35: GL tag implication.
  have row_35 : ((gl_preorder N add v6 v5) → (zero = v3)) := by
    intro scope_premise_1
    have scoped_fact_3 := row_36 scope_premise_1
    apply rule_row_35
    exact row_49
    exact scoped_fact_3
  -- chapter_45_line_24: GL tag equality1.
  have row_24 : ((gl_preorder N add v6 v5) → (gl_preorder N mul v1 v3)) := by
    intro scope_premise_1
    have scoped_fact_2 := row_35 scope_premise_1
    have equality_source := row_25
    have equality_step_1 := scoped_fact_2
    cases equality_step_1
    exact equality_source
  -- chapter_45_line_22: GL tag or disintegration.
  have row_22 : ((gl_preorder N add v5 v6) → (gl_preorder N add v5 v6)) := by
    intro scope_premise_1
    exact scope_premise_1
  -- chapter_45_line_21: GL tag expansion.
  have row_21 : ((gl_preorder N add v5 v6) → (¬ (∀ (v8 : α), ((N v8) → (¬ (add v5 v8 v6)))))) := by
    simpa only [gl_preorder] using row_22
  -- chapter_45_line_23: GL tag disintegration.
  have row_23 : ((gl_preorder N add v5 v6) → (∃ (v8 : α), ((N v8) ∧ (add v5 v8 v6)))) := by
    intro scope_premise_1
    have scoped_fact_1 := row_21 scope_premise_1
    obtain ⟨v8, scoped_witness_bundle⟩ := existsAndOfNotForallImpNot scoped_fact_1
    exact ⟨v8, ⟨scoped_witness_bundle.1, scoped_witness_bundle.2⟩⟩
  -- chapter_45_line_20: GL tag disintegration.
  have row_20 : ((gl_preorder N add v5 v6) → (∃ (v8 : α), ((N v8) ∧ (add v5 v8 v6)))) := by
    intro scope_premise_1
    have scoped_fact_1 := row_21 scope_premise_1
    obtain ⟨v8, scoped_witness_bundle⟩ := existsAndOfNotForallImpNot scoped_fact_1
    exact ⟨v8, ⟨scoped_witness_bundle.1, scoped_witness_bundle.2⟩⟩
  -- chapter_45_line_15: GL tag implication.
  have row_15 : ((gl_preorder N add v5 v6) → (∀ (v8 : α), (((N v8) ∧ (add v5 v8 v6)) → (mul v1 v8 v3)))) := by
    intro scope_premise_1
    intro v8
    intro witness_guard_1
    have scoped_fact_5 := row_20 scope_premise_1
    apply row_16
    exact row_18
    exact row_17
    exact row_19
    exact witness_guard_1.2
  -- chapter_45_line_12: GL tag implication.
  have row_12 : ((gl_preorder N add v5 v6) → (gl_preorder N mul v1 v3)) := by
    intro scope_premise_1
    obtain ⟨v8, witness_guard_1⟩ := row_23 scope_premise_1
    have scoped_fact_3 := row_15 scope_premise_1 v8 witness_guard_1
    apply row_13
    exact witness_guard_1.1
    exact scoped_fact_3
  -- chapter_45_line_1: GL tag or convergence.
  have row_1 : (gl_preorder N mul v1 v3) := by
    classical
    have or_cases := row_2
    simp only [gl_or5] at or_cases
    rcases or_cases with or_branch_1 | or_branch_2
    · have branch_compact := or_branch_1
      exact row_12 branch_compact
    · have branch_compact := or_branch_2
      exact row_24 branch_compact
  exact row_1

theorem fta_source_043
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_031 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α), ((x_1 x_7) → (∀ (x_8 : α), ((x_5 x_2 x_7 x_8) → (x_2 = x_8))))))))
    (external_peano_externals_36_009 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (∀ (w4 : α) (w5 : α), ((mul w4 w2 w5) → (∀ (w6 : α), ((mul w3 w4 w6) → (mul w5 w1 w6))))))))))
    (external_peano_externals_36_010 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_5 x_7 x_8 x_9) → (x_5 x_8 x_7 x_9))))))
    (external_peano_externals_36_007 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_5 x_7 x_8 x_9) → (∀ (x_10 : α) (x_11 : α) (x_12 : α), ((x_5 x_10 x_11 x_12) → ((x_5 x_8 x_7 x_11) → (x_5 x_10 x_9 x_12)))))))))
    (external_peano_externals_36_029 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((gl_preorder x_1 x_4 x_7 x_8) → ((gl_preorder x_1 x_4 x_8 x_7) → (x_7 = x_8)))))))
    (external_peano_externals_36_017 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_6 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_021 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_6 x_8) → (x_3 x_7 x_8))))))
    (external_peano_externals_36_015 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_3 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_022 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_6 x_8) → (x_4 x_6 x_7 x_8))))))
    (external_peano_externals_36_033 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α), ((x_1 x_7) → (∀ (x_8 : α), ((x_5 x_6 x_8 x_7) → (x_7 = x_8))))))))
    (external_peano_externals_36_008 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (∀ (w4 : α) (w5 : α), ((mul w4 w1 w5) → (∀ (w6 : α), ((mul w4 w2 w6) → (∀ (w7 : α), ((mul w4 w3 w7) → (add w5 w6 w7))))))))))))
    : (∀ (v1 : α) (v2 : α), ((gl_preorder N mul v1 v2) → ((gl_preorder N mul v2 v1) → (v1 = v2)))) := by
  intro v1
  intro v2
  intro premise_1
  intro premise_2
  have or_elim_parent_1 := fta_source_044 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_031
  have or_elim_parent_2 := fta_source_045 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_009 external_peano_externals_36_010 external_peano_externals_36_007 external_peano_externals_36_029 external_peano_externals_36_017 external_peano_externals_36_021 external_peano_externals_36_015 external_peano_externals_36_022 external_peano_externals_36_033 external_peano_externals_36_008
  have or_elim_parent_3 := fta_source_061 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_015 external_peano_externals_36_022
  -- chapter_55_line_1: GL tag or elimination.
  have row_1 : (∀ (v1 : α) (v2 : α), ((gl_preorder N mul v1 v2) → ((gl_preorder N mul v2 v1) → (v1 = v2)))) := by
    classical
    intro v1
    intro v2
    intro or_elimination_premise_1
    intro or_elimination_premise_2
    have preorder_witness := or_elimination_premise_1
    obtain ⟨preorder_middle, preorder_middle_in_N, preorder_relation⟩ := existsAndOfNotForallImpNot preorder_witness
    have natural_numbers : gl_NaturalNumbers N zero succ add mul := anchor.1.1.1
    have multiplication_structure : gl_fXYZ mul N N N := natural_numbers.1.1.1.2
    have multiplication_output_closed : gl_implication10 mul N := multiplication_structure.1.1.2
    have right_in_domain : N v2 := multiplication_output_closed v1 preorder_middle v2 preorder_relation
    have or_elim_cases := or_elim_parent_3 v2 right_in_domain
    simp only [gl_or3] at or_elim_cases
    rcases or_elim_cases with positive_case | zero_case
    · exact or_elim_parent_2 v1 v2 or_elimination_premise_1 or_elimination_premise_2 positive_case
    · exact or_elim_parent_1 v1 v2 or_elimination_premise_1 or_elimination_premise_2 zero_case
  solve_by_elim [row_1]

theorem fta_source_054
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_019 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_2 x_8) → (x_4 x_2 x_7 x_8))))))
    (external_peano_externals_36_006 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (x_4 x_8 x_7 x_9))))))
    (external_peano_externals_36_015 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_3 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_034 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (∀ (w1 : α), ((N w1) → (gl_or2 w1 zero N succ))))))
    (external_peano_externals_36_003 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (∀ (x_10 : α), ((x_3 x_8 x_10) → (∀ (x_11 : α), ((x_3 x_11 x_7) → (x_4 x_10 x_11 x_9))))))))))
    (external_peano_externals_36_004 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (∀ (x_10 : α), ((x_4 x_7 x_10 x_9) → (x_8 = x_10))))))))
    : (∀ (v1 : α), ((N v1) → (∀ (v2 : α), ((N v2) → ((¬ (gl_strictOrder N add v1 v2)) → ((¬ (gl_strictOrder N add v2 v1)) → (v1 = v2))))))) := by
  intro v1
  intro premise_1
  intro v2
  intro premise_2
  intro premise_3
  intro premise_4
  apply Classical.byContradiction
  intro reductio
  -- chapter_68_line_15: GL tag task formulation.
  have row_15 : (N v1) := by
    exact premise_1
  -- chapter_68_line_13: GL tag task formulation.
  have row_13 : (N v2) := by
    exact premise_2
  -- chapter_68_line_12: GL tag task formulation.
  have row_12 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_68_line_11: GL tag task formulation.
  have row_11 : (¬ (gl_strictOrder N add v2 v1)) := by
    exact premise_4
  -- chapter_68_line_9: GL tag variable copy.
  have row_9 : (v1 = v1) := by
    rfl
  -- chapter_68_line_16: GL tag symmetry of equality.
  have row_16 : (v1 = v1) := by
    exact Eq.symm row_9
  -- chapter_68_line_14: GL tag equality1.
  have row_14 : (N v1) := by
    have equality_source := row_15
    have equality_step_1 := row_9
    cases equality_step_1
    exact equality_source
  -- chapter_68_line_10: GL tag equality1.
  have row_10 : (¬ (gl_strictOrder N add v2 v1)) := by
    have equality_source := row_11
    have equality_step_1 := row_9
    cases equality_step_1
    exact equality_source
  -- chapter_68_line_8: GL tag task formulation.
  have row_8 : (¬ (v1 = v2)) := by
    exact reductio
  -- chapter_68_line_7: GL tag equality1.
  have row_7 : (¬ (v1 = v2)) := by
    have equality_source := row_8
    have equality_step_1 := row_9
    cases equality_step_1
    exact equality_source
  -- chapter_68_line_6: GL tag symmetry of inequality.
  have row_6 : (¬ (v2 = v1)) := by
    exact fun equality => row_7 (Eq.symm equality)
  -- chapter_68_line_5: GL tag theorem.
  have row_5 := fta_source_053 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_019 external_peano_externals_36_006 external_peano_externals_36_015 external_peano_externals_36_034 external_peano_externals_36_003 external_peano_externals_36_004
  -- chapter_68_line_4: GL tag implication.
  have row_4 : (gl_strictOrder N add v1 v2) := by
    apply row_5
    exact row_13
    exact row_14
    exact row_10
    exact row_6
  -- chapter_68_line_3: GL tag equality1.
  have row_3 : (gl_strictOrder N add v1 v2) := by
    have equality_source := row_4
    have equality_step_1 := row_16
    cases equality_step_1
    exact equality_source
  -- chapter_68_line_2: GL tag task formulation.
  have row_2 : (¬ (gl_strictOrder N add v1 v2)) := by
    exact premise_3
  -- chapter_68_line_1: GL tag contradiction.
  have row_1 : (v1 = v2) := by
    exact False.elim (row_2 row_3)
  exact reductio row_1

theorem fta_source_058
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_034 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (∀ (w1 : α), ((N w1) → (gl_or2 w1 zero N succ))))))
    (external_peano_externals_36_022 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_6 x_8) → (x_4 x_6 x_7 x_8))))))
    (external_peano_externals_36_035 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (gl_existence11 N one succ))))
    (external_peano_externals_36_015 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_3 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_gauss_externals_24_018 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α) (x_7 : α) (x_8 : GLBinaryRelation α), ((gl_AnchorGauss x_1 x_2 x_3 x_4 x_5 x_6 x_7 x_8) → (∀ (x_9 : α) (x_10 : α), ((x_3 x_9 x_10) → (gl_preorder x_1 x_4 x_9 x_10))))))
    (external_peano_externals_36_006 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (x_4 x_8 x_7 x_9))))))
    (external_peano_externals_36_003 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (∀ (x_10 : α), ((x_3 x_8 x_10) → (∀ (x_11 : α), ((x_3 x_11 x_7) → (x_4 x_10 x_11 x_9))))))))))
    (external_peano_externals_36_002 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (∀ (w4 : α) (w5 : α), ((add w4 w2 w5) → (∀ (w6 : α), ((add w3 w4 w6) → (add w5 w1 w6))))))))))
    (external_peano_externals_36_029 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((gl_preorder x_1 x_4 x_7 x_8) → ((gl_preorder x_1 x_4 x_8 x_7) → (x_7 = x_8)))))))
    (external_peano_externals_36_005 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → ((x_8 = x_9) → (x_2 = x_7)))))))
    : (∀ (v1 : α), ((N v1) → ((¬ (zero = v1)) → ((¬ (gl_preorder N add two v1)) → (one = v1))))) := by
  intro v1
  intro premise_1
  intro premise_2
  intro premise_3
  apply Classical.byContradiction
  intro reductio
  -- chapter_74_line_74: GL tag theorem.
  have row_74 := fta_source_002 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_74_line_72: GL tag task formulation.
  have row_72 : (¬ (one = v1)) := by
    exact reductio
  -- chapter_74_line_71: GL tag theorem.
  have row_71 := fta_source_031 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_74_line_69: GL tag theorem.
  have row_69 := fta_source_017 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_034
  -- chapter_74_line_67: GL tag variable copy.
  have row_67 : (v1 = v1) := by
    rfl
  -- chapter_74_line_76: GL tag symmetry of equality.
  have row_76 : (v1 = v1) := by
    exact Eq.symm row_67
  -- chapter_74_line_66: GL tag task formulation.
  have row_66 : (¬ (zero = v1)) := by
    exact premise_2
  -- chapter_74_line_65: GL tag symmetry of inequality.
  have row_65 : (¬ (v1 = zero)) := by
    exact fun equality => row_66 (Eq.symm equality)
  -- chapter_74_line_64: GL tag task formulation.
  have row_64 : (N v1) := by
    exact premise_1
  -- chapter_74_line_56: GL tag theorem.
  have row_56 := fta_source_019 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_015
  -- chapter_74_line_43: GL tag theorem.
  have row_43 := fta_source_050 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_74_line_35: GL tag theorem.
  have row_35 := fta_source_016 N zero succ add mul one two identity anchor relationalInduction external_gauss_externals_24_018 external_peano_externals_36_006 external_peano_externals_36_003 external_peano_externals_36_002 external_peano_externals_36_029 external_peano_externals_36_035 external_peano_externals_36_005
  -- chapter_74_line_13: GL tag expansion for integration.
  have row_13 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_74_line_12: GL tag reformulation for integration and.
  have row_12 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_74_line_7: GL tag task formulation.
  have row_7 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_74_line_15: GL tag expansion.
  have row_15 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_7
  -- chapter_74_line_30: GL tag disintegration.
  have row_30 : (succ one two) := by
    exact row_15.1.2
  -- chapter_74_line_16: GL tag disintegration.
  have row_16 : (succ zero one) := by
    exact row_15.1.1.2
  -- chapter_74_line_14: GL tag disintegration.
  have row_14 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_15.1.1.1
  -- chapter_74_line_29: GL tag expansion.
  have row_29 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_14
  -- chapter_74_line_50: GL tag disintegration.
  have row_50 : (N zero) := by
    exact row_29.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_74_line_46: GL tag disintegration.
  have row_46 : (gl_implication7 N succ) := by
    exact row_29.1.1.1.1.1.1.1.1.1.2
  -- chapter_74_line_45: GL tag expansion.
  have row_45 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w2 w1) → (∀ (w3 : α), ((succ w3 w1) → (w2 = w3))))))) := by
    simpa only [gl_implication7] using row_46
  -- chapter_74_line_28: GL tag disintegration.
  have row_28 : (gl_fXY succ N N) := by
    exact row_29.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_74_line_27: GL tag expansion.
  have row_27 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_28
  -- chapter_74_line_49: GL tag disintegration.
  have row_49 : (gl_implication0 succ N) := by
    exact row_27.1.1.1
  -- chapter_74_line_48: GL tag expansion.
  have row_48 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_49
  -- chapter_74_line_47: GL tag implication.
  have row_47 : (N one) := by
    apply row_48
    exact row_30
  -- chapter_74_line_26: GL tag disintegration.
  have row_26 : (gl_implication1 succ N) := by
    exact row_27.1.1.2
  -- chapter_74_line_25: GL tag expansion.
  have row_25 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using row_26
  -- chapter_74_line_24: GL tag implication.
  have row_24 : (N two) := by
    apply row_25
    exact row_30
  -- chapter_74_line_11: GL tag implication.
  have row_11 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_12
    exact row_14
    exact row_16
  have rule_row_63 := external_peano_externals_36_034 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_74_line_63: GL tag implication.
  have row_63 : (gl_or2 v1 zero N succ) := by
    apply rule_row_63
    exact row_64
  -- chapter_74_line_62: GL tag expansion.
  have row_62 : (¬ ((¬ (v1 = zero)) ∧ (¬ (gl_existence11 N v1 succ)))) := by
    simpa only [gl_or2, GLExport.orIffNotAndNot] using row_63
  -- chapter_74_line_61: GL tag disintegration.
  have row_61 : (gl_implication74 v1 zero N succ) := by
    simp only [gl_implication74]
    classical
    intro projection_premise
    apply Classical.byContradiction
    intro projection_counterexample
    exact row_62 ⟨projection_premise, projection_counterexample⟩
  -- chapter_74_line_60: GL tag expansion.
  have row_60 : ((¬ (v1 = zero)) → (gl_existence11 N v1 succ)) := by
    simpa only [gl_implication74] using row_61
  -- chapter_74_line_59: GL tag implication.
  have row_59 : (gl_existence11 N v1 succ) := by
    apply row_60
    exact row_65
  -- chapter_74_line_58: GL tag expansion.
  have row_58 : (¬ (∀ (v4 : α), ((N v4) → (¬ (succ v4 v1))))) := by
    simpa only [gl_existence11] using row_59
  have exists_row_58 : ∃ (v4 : α), ((N v4) ∧ (succ v4 v1)) := existsAndOfNotForallImpNot row_58
  obtain ⟨v4, witness_row_58⟩ := exists_row_58
  -- chapter_74_line_57: GL tag disintegration.
  have row_57 : (succ v4 v1) := by
    exact witness_row_58.2
  -- chapter_74_line_55: GL tag implication.
  have row_55 : (add v4 one v1) := by
    apply row_56
    exact row_57
  have rule_row_75 := external_peano_externals_36_022 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_74_line_75: GL tag implication.
  have row_75 : (add one v4 v1) := by
    apply rule_row_75
    exact row_55
  have rule_row_41 := external_peano_externals_36_035 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_74_line_41: GL tag implication.
  have row_41 : (gl_existence11 N one succ) := by
    apply rule_row_41
  -- chapter_74_line_40: GL tag expansion.
  have row_40 : (¬ (∀ (v7 : α), ((N v7) → (¬ (succ v7 one))))) := by
    simpa only [gl_existence11] using row_41
  have exists_row_40 : ∃ (v7 : α), ((N v7) ∧ (succ v7 one)) := existsAndOfNotForallImpNot row_40
  obtain ⟨v7, witness_row_40⟩ := exists_row_40
  -- chapter_74_line_39: GL tag disintegration.
  have row_39 : (succ v7 one) := by
    exact witness_row_40.2
  -- chapter_74_line_44: GL tag implication.
  have row_44 : (zero = v7) := by
    apply row_45
    exact row_47
    exact row_16
    exact row_39
  have rule_row_23 := external_peano_externals_36_034 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_74_line_23: GL tag implication.
  have row_23 : (gl_or2 two zero N succ) := by
    apply rule_row_23
    exact row_24
  -- chapter_74_line_22: GL tag expansion.
  have row_22 : (¬ ((¬ (two = zero)) ∧ (¬ (gl_existence11 N two succ)))) := by
    simpa only [gl_or2, GLExport.orIffNotAndNot] using row_23
  -- chapter_74_line_21: GL tag disintegration.
  have row_21 : (gl_implication74 two zero N succ) := by
    simp only [gl_implication74]
    classical
    intro projection_premise
    apply Classical.byContradiction
    intro projection_counterexample
    exact row_22 ⟨projection_premise, projection_counterexample⟩
  -- chapter_74_line_20: GL tag expansion.
  have row_20 : ((¬ (two = zero)) → (gl_existence11 N two succ)) := by
    simpa only [gl_implication74] using row_21
  -- chapter_74_line_6: GL tag anchor handling.
  have row_6 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact row_7
  -- chapter_74_line_73: GL tag implication.
  have row_73 : (gl_preorder N add one v1) := by
    apply row_74
    exact row_75
  -- chapter_74_line_70: GL tag implication.
  have row_70 : (gl_strictOrder N add one v1) := by
    apply row_71
    exact row_73
    exact row_72
  -- chapter_74_line_68: GL tag implication.
  have row_68 : (gl_preorder N add one v4) := by
    apply row_69
    exact row_57
    exact row_70
  -- chapter_74_line_42: GL tag implication.
  have row_42 : (gl_preorder N add zero v7) := by
    apply row_43
    exact row_50
    exact row_44
  -- chapter_74_line_38: GL tag implication.
  have row_38 : (gl_strictOrder N add zero one) := by
    apply row_35
    exact row_39
    exact row_42
  -- chapter_74_line_37: GL tag expansion.
  have row_37 : ((gl_preorder N add zero one) ∧ (¬ (zero = one))) := by
    simpa only [gl_strictOrder] using row_38
  -- chapter_74_line_36: GL tag disintegration.
  have row_36 : (gl_preorder N add zero one) := by
    exact row_37.1
  -- chapter_74_line_34: GL tag implication.
  have row_34 : (gl_strictOrder N add zero two) := by
    apply row_35
    exact row_30
    exact row_36
  -- chapter_74_line_33: GL tag expansion.
  have row_33 : ((gl_preorder N add zero two) ∧ (¬ (zero = two))) := by
    simpa only [gl_strictOrder] using row_34
  -- chapter_74_line_32: GL tag disintegration.
  have row_32 : (¬ (zero = two)) := by
    exact row_33.2
  -- chapter_74_line_31: GL tag symmetry of inequality.
  have row_31 : (¬ (two = zero)) := by
    exact fun equality => row_32 (Eq.symm equality)
  -- chapter_74_line_19: GL tag implication.
  have row_19 : (gl_existence11 N two succ) := by
    apply row_20
    exact row_31
  -- chapter_74_line_18: GL tag expansion.
  have row_18 : (¬ (∀ (v3 : α), ((N v3) → (¬ (succ v3 two))))) := by
    simpa only [gl_existence11] using row_19
  have exists_row_18 : ∃ (v3 : α), ((N v3) ∧ (succ v3 two)) := existsAndOfNotForallImpNot row_18
  obtain ⟨v3, witness_row_18⟩ := exists_row_18
  -- chapter_74_line_17: GL tag disintegration.
  have row_17 : (succ v3 two) := by
    exact witness_row_18.2
  -- chapter_74_line_52: GL tag implication.
  have row_52 : (one = v3) := by
    apply row_45
    exact row_24
    exact row_30
    exact row_17
  -- chapter_74_line_54: GL tag equality1.
  have row_54 : (add v4 v3 v1) := by
    have equality_source := row_55
    have equality_step_1 := row_52
    cases equality_step_1
    exact equality_source
  -- chapter_74_line_53: GL tag equality1.
  have row_53 : (add v4 v3 v1) := by
    have equality_source := row_54
    have equality_step_1 := row_67
    cases equality_step_1
    exact equality_source
  -- chapter_74_line_51: GL tag implication.
  have row_51 : (v3 = one) := by
    apply row_45
    exact row_24
    exact row_17
    exact row_30
  have rule_row_10 := external_peano_externals_36_015 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_74_line_10: GL tag implication.
  have row_10 : (add v3 one two) := by
    apply rule_row_10
    exact row_17
  -- chapter_74_line_9: GL tag equality1.
  have row_9 : (add one one two) := by
    have equality_source := row_10
    have equality_step_1 := row_51
    cases equality_step_1
    exact equality_source
  -- chapter_74_line_8: GL tag equality1.
  have row_8 : (add one v3 two) := by
    have equality_source := row_9
    have equality_step_1 := row_52
    cases equality_step_1
    exact equality_source
  -- chapter_74_line_5: GL tag theorem.
  have row_5 := fta_source_029 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_006 external_peano_externals_36_002
  -- chapter_74_line_4: GL tag implication.
  have row_4 : (gl_preorder N add two v1) := by
    apply row_5
    exact row_68
    exact row_8
    exact row_53
  -- chapter_74_line_3: GL tag equality1.
  have row_3 : (gl_preorder N add two v1) := by
    have equality_source := row_4
    have equality_step_1 := row_76
    cases equality_step_1
    exact equality_source
  -- chapter_74_line_2: GL tag task formulation.
  have row_2 : (¬ (gl_preorder N add two v1)) := by
    exact premise_3
  -- chapter_74_line_1: GL tag contradiction.
  have row_1 : (one = v1) := by
    exact False.elim (row_2 row_3)
  exact reductio row_1

private theorem fta_source_067_induction_typing
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (assumption_10 : (mul v2 v1 v3))
    : (N v2) := by
  -- chapter_89_line_10: GL tag task formulation.
  have row_10 : (mul v2 v1 v3) := by
    exact assumption_10
  -- chapter_89_line_9: GL tag task formulation.
  have row_9 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_89_line_8: GL tag expansion.
  have row_8 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_9
  -- chapter_89_line_7: GL tag disintegration.
  have row_7 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_8.1.1.1
  -- chapter_89_line_6: GL tag expansion.
  have row_6 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_7
  -- chapter_89_line_5: GL tag disintegration.
  have row_5 : (gl_fXYZ mul N N N) := by
    exact row_6.1.1.1.2
  -- chapter_89_line_4: GL tag expansion.
  have row_4 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_5
  -- chapter_89_line_3: GL tag disintegration.
  have row_3 : (gl_implication8 mul N) := by
    exact row_4.1.1.1.1
  -- chapter_89_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_3
  -- chapter_89_line_1: GL tag implication.
  have row_1 : (N v2) := by
    apply row_2
    exact row_10
  exact row_1

private theorem fta_source_067_check_zero
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_035 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (gl_existence11 N one succ))))
    (external_gauss_externals_24_018 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α) (x_7 : α) (x_8 : GLBinaryRelation α), ((gl_AnchorGauss x_1 x_2 x_3 x_4 x_5 x_6 x_7 x_8) → (∀ (x_9 : α) (x_10 : α), ((x_3 x_9 x_10) → (gl_preorder x_1 x_4 x_9 x_10))))))
    (external_gauss_externals_24_021 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α) (x_7 : α) (x_8 : GLBinaryRelation α), ((gl_AnchorGauss x_1 x_2 x_3 x_4 x_5 x_6 x_7 x_8) → (∀ (x_9 : α) (x_10 : α), ((gl_preorder x_1 x_4 x_9 x_10) → (∀ (x_11 : α), ((gl_preorder x_1 x_4 x_10 x_11) → (gl_preorder x_1 x_4 x_9 x_11))))))))
    (v1 : α)
    (v2 : α)
    (v4 : α)
    (v5 : α)
    (assumption_15 : (v2 = zero))
    (assumption_13 : (mul v4 v1 v5))
    : (gl_preorder N add v2 v4) := by
  -- chapter_90_line_15: GL tag recursion.
  have row_15 : (v2 = zero) := by
    exact assumption_15
  -- chapter_90_line_14: GL tag symmetry of equality.
  have row_14 : (zero = v2) := by
    exact Eq.symm row_15
  -- chapter_90_line_13: GL tag task formulation.
  have row_13 : (mul v4 v1 v5) := by
    exact assumption_13
  -- chapter_90_line_4: GL tag task formulation.
  have row_4 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_90_line_12: GL tag expansion.
  have row_12 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_4
  -- chapter_90_line_11: GL tag disintegration.
  have row_11 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_12.1.1.1
  -- chapter_90_line_10: GL tag expansion.
  have row_10 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_11
  -- chapter_90_line_9: GL tag disintegration.
  have row_9 : (gl_fXYZ mul N N N) := by
    exact row_10.1.1.1.2
  -- chapter_90_line_8: GL tag expansion.
  have row_8 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_9
  -- chapter_90_line_7: GL tag disintegration.
  have row_7 : (gl_implication8 mul N) := by
    exact row_8.1.1.1.1
  -- chapter_90_line_6: GL tag expansion.
  have row_6 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_7
  -- chapter_90_line_5: GL tag implication.
  have row_5 : (N v4) := by
    apply row_6
    exact row_13
  -- chapter_90_line_3: GL tag theorem.
  have row_3 := fta_source_063 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_035 external_gauss_externals_24_018 external_gauss_externals_24_021
  -- chapter_90_line_2: GL tag implication.
  have row_2 : (gl_preorder N add zero v4) := by
    apply row_3
    exact row_5
  -- chapter_90_line_1: GL tag equality1.
  have row_1 : (gl_preorder N add v2 v4) := by
    have equality_source := row_2
    have equality_step_1 := row_14
    cases equality_step_1
    exact equality_source
  exact row_1

private theorem fta_source_067_check_induction_condition
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_010 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_5 x_7 x_8 x_9) → (x_5 x_8 x_7 x_9))))))
    (external_peano_externals_36_035 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (gl_existence11 N one succ))))
    (external_peano_externals_36_017 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_6 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_021 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_6 x_8) → (x_3 x_7 x_8))))))
    (external_peano_externals_36_015 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_3 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_002 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (∀ (w4 : α) (w5 : α), ((add w4 w2 w5) → (∀ (w6 : α), ((add w3 w4 w6) → (add w5 w1 w6))))))))))
    (external_gauss_externals_24_021 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α) (x_7 : α) (x_8 : GLBinaryRelation α), ((gl_AnchorGauss x_1 x_2 x_3 x_4 x_5 x_6 x_7 x_8) → (∀ (x_9 : α) (x_10 : α), ((gl_preorder x_1 x_4 x_9 x_10) → (∀ (x_11 : α), ((gl_preorder x_1 x_4 x_10 x_11) → (gl_preorder x_1 x_4 x_9 x_11))))))))
    (external_peano_externals_36_006 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (x_4 x_8 x_7 x_9))))))
    (external_peano_externals_36_034 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (∀ (w1 : α), ((N w1) → (gl_or2 w1 zero N succ))))))
    (external_peano_externals_36_003 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (∀ (x_10 : α), ((x_3 x_8 x_10) → (∀ (x_11 : α), ((x_3 x_11 x_7) → (x_4 x_10 x_11 x_9))))))))))
    (external_peano_externals_36_008 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (∀ (w4 : α) (w5 : α), ((mul w4 w1 w5) → (∀ (w6 : α), ((mul w4 w2 w6) → (∀ (w7 : α), ((mul w4 w3 w7) → (add w5 w6 w7))))))))))))
    (external_peano_externals_36_004 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (∀ (x_10 : α), ((x_4 x_7 x_10 x_9) → (x_8 = x_10))))))))
    (external_gauss_externals_24_018 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α) (x_7 : α) (x_8 : GLBinaryRelation α), ((gl_AnchorGauss x_1 x_2 x_3 x_4 x_5 x_6 x_7 x_8) → (∀ (x_9 : α) (x_10 : α), ((x_3 x_9 x_10) → (gl_preorder x_1 x_4 x_9 x_10))))))
    (external_peano_externals_36_029 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((gl_preorder x_1 x_4 x_7 x_8) → ((gl_preorder x_1 x_4 x_8 x_7) → (x_7 = x_8)))))))
    (external_peano_externals_36_005 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → ((x_8 = x_9) → (x_2 = x_7)))))))
    (external_peano_externals_36_022 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_6 x_8) → (x_4 x_6 x_7 x_8))))))
    (previous : α)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (v4 : α)
    (v5 : α)
    (assumption_51 : (mul v2 v1 v3))
    (assumption_44 : (gl_preorder N add v3 v5))
    (assumption_37 : (succ previous v2))
    (assumption_31 : (gl_preorder N add one v1))
    (assumption_14 : (mul v4 v1 v5))
    (assumption_13 : ((gl_preorder N add one v1) → (∀ (w1 : α), ((mul previous v1 w1) → ((mul v4 v1 v5) → ((gl_preorder N add w1 v5) → (gl_preorder N add previous v4)))))))
    : (gl_preorder N add v2 v4) := by
  -- chapter_91_line_121: GL tag theorem.
  have row_121 := fta_source_003 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_008 external_peano_externals_36_004
  -- chapter_91_line_119: GL tag theorem.
  have row_119 := fta_source_010 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_010
  -- chapter_91_line_107: GL tag theorem.
  have row_107 := fta_source_063 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_035 external_gauss_externals_24_018 external_gauss_externals_24_021
  -- chapter_91_line_95: GL tag theorem.
  have row_95 := fta_source_001 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_006 external_peano_externals_36_004
  -- chapter_91_line_84: GL tag theorem.
  have row_84 := fta_source_016 N zero succ add mul one two identity anchor relationalInduction external_gauss_externals_24_018 external_peano_externals_36_006 external_peano_externals_36_003 external_peano_externals_36_002 external_peano_externals_36_029 external_peano_externals_36_035 external_peano_externals_36_005
  -- chapter_91_line_79: GL tag theorem.
  have row_79 := fta_source_059 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_015 external_peano_externals_36_022
  -- chapter_91_line_77: GL tag theorem.
  have row_77 := fta_source_041 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_010 external_peano_externals_36_017 external_peano_externals_36_021 external_peano_externals_36_015 external_peano_externals_36_022
  -- chapter_91_line_75: GL tag theorem.
  have row_75 := fta_source_066 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_017 external_peano_externals_36_021
  -- chapter_91_line_51: GL tag task formulation.
  have row_51 : (mul v2 v1 v3) := by
    exact assumption_51
  -- chapter_91_line_46: GL tag theorem.
  have row_46 := fta_source_002 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_91_line_44: GL tag task formulation.
  have row_44 : (gl_preorder N add v3 v5) := by
    exact assumption_44
  -- chapter_91_line_93: GL tag expansion.
  have row_93 : (¬ (∀ (v20 : α), ((N v20) → (¬ (add v3 v20 v5))))) := by
    simpa only [gl_preorder] using row_44
  have exists_row_93 : ∃ (v20 : α), ((N v20) ∧ (add v3 v20 v5)) := existsAndOfNotForallImpNot row_93
  obtain ⟨v20, witness_row_93⟩ := exists_row_93
  -- chapter_91_line_100: GL tag disintegration.
  have row_100 : (add v3 v20 v5) := by
    exact witness_row_93.2
  -- chapter_91_line_92: GL tag disintegration.
  have row_92 : (N v20) := by
    exact witness_row_93.1
  -- chapter_91_line_41: GL tag expansion for integration.
  have row_41 : ((gl_AnchorGauss N zero succ add mul one two identity) ↔ ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity))) := by
    exact Iff.rfl
  -- chapter_91_line_40: GL tag reformulation for integration and.
  have row_40 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → ((succ one two) → ((gl_identity N identity) → (gl_AnchorGauss N zero succ add mul one two identity))))) := by
    intro integration_premise_1
    intro integration_premise_2
    intro integration_premise_3
    intro integration_premise_4
    simp only [gl_AnchorGauss]
    exact ⟨⟨⟨integration_premise_1, integration_premise_2⟩, integration_premise_3⟩, integration_premise_4⟩
  -- chapter_91_line_37: GL tag recursion.
  have row_37 : (succ previous v2) := by
    exact assumption_37
  -- chapter_91_line_31: GL tag task formulation.
  have row_31 : (gl_preorder N add one v1) := by
    exact assumption_31
  -- chapter_91_line_30: GL tag expansion.
  have row_30 : (¬ (∀ (v12 : α), ((N v12) → (¬ (add one v12 v1))))) := by
    simpa only [gl_preorder] using row_31
  have exists_row_30 : ∃ (v12 : α), ((N v12) ∧ (add one v12 v1)) := existsAndOfNotForallImpNot row_30
  obtain ⟨v12, witness_row_30⟩ := exists_row_30
  -- chapter_91_line_104: GL tag disintegration.
  have row_104 : (N v12) := by
    exact witness_row_30.1
  -- chapter_91_line_29: GL tag disintegration.
  have row_29 : (add one v12 v1) := by
    exact witness_row_30.2
  -- chapter_91_line_14: GL tag task formulation.
  have row_14 : (mul v4 v1 v5) := by
    exact assumption_14
  -- chapter_91_line_13: GL tag recursion.
  have row_13 : ((gl_preorder N add one v1) → (∀ (w1 : α), ((mul previous v1 w1) → ((mul v4 v1 v5) → ((gl_preorder N add w1 v5) → (gl_preorder N add previous v4)))))) := by
    exact assumption_13
  -- chapter_91_line_8: GL tag task formulation.
  have row_8 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_91_line_7: GL tag expansion.
  have row_7 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_8
  -- chapter_91_line_43: GL tag disintegration.
  have row_43 : (succ one two) := by
    exact row_7.1.2
  -- chapter_91_line_42: GL tag disintegration.
  have row_42 : (gl_identity N identity) := by
    exact row_7.2
  -- chapter_91_line_9: GL tag disintegration.
  have row_9 : (succ zero one) := by
    exact row_7.1.1.2
  -- chapter_91_line_6: GL tag disintegration.
  have row_6 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_7.1.1.1
  -- chapter_91_line_39: GL tag implication.
  have row_39 : (gl_AnchorGauss N zero succ add mul one two identity) := by
    apply row_40
    exact row_6
    exact row_9
    exact row_43
    exact row_42
  -- chapter_91_line_23: GL tag expansion.
  have row_23 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_6
  -- chapter_91_line_111: GL tag disintegration.
  have row_111 : (gl_implication7 N succ) := by
    exact row_23.1.1.1.1.1.1.1.1.1.2
  -- chapter_91_line_110: GL tag expansion.
  have row_110 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w2 w1) → (∀ (w3 : α), ((succ w3 w1) → (w2 = w3))))))) := by
    simpa only [gl_implication7] using row_111
  -- chapter_91_line_72: GL tag disintegration.
  have row_72 : (gl_implication15 N zero add) := by
    exact row_23.1.1.1.1.1.1.1.2
  -- chapter_91_line_71: GL tag expansion.
  have row_71 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((add w1 zero w2) → (w1 = w2))))) := by
    simpa only [gl_implication15] using row_72
  -- chapter_91_line_62: GL tag disintegration.
  have row_62 : (N zero) := by
    exact row_23.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_91_line_56: GL tag disintegration.
  have row_56 : (gl_implication17 N succ add) := by
    exact row_23.1.1.1.1.1.2
  -- chapter_91_line_55: GL tag expansion.
  have row_55 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication17] using row_56
  -- chapter_91_line_49: GL tag disintegration.
  have row_49 : (gl_implication21 N succ mul add) := by
    exact row_23.2
  -- chapter_91_line_48: GL tag expansion.
  have row_48 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((mul w3 w1 w4) → (∀ (w5 : α), ((mul w3 w2 w5) → (add w4 w3 w5))))))))) := by
    simpa only [gl_implication21] using row_49
  -- chapter_91_line_36: GL tag disintegration.
  have row_36 : (gl_fXY succ N N) := by
    exact row_23.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_91_line_35: GL tag expansion.
  have row_35 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_36
  -- chapter_91_line_34: GL tag disintegration.
  have row_34 : (gl_implication0 succ N) := by
    exact row_35.1.1.1
  -- chapter_91_line_33: GL tag expansion.
  have row_33 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_34
  -- chapter_91_line_115: GL tag implication.
  have row_115 : (N one) := by
    apply row_33
    exact row_43
  -- chapter_91_line_32: GL tag implication.
  have row_32 : (N previous) := by
    apply row_33
    exact row_37
  -- chapter_91_line_28: GL tag disintegration.
  have row_28 : (gl_fXYZ add N N N) := by
    exact row_23.1.1.1.1.1.1.1.1.2
  -- chapter_91_line_27: GL tag expansion.
  have row_27 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_28
  -- chapter_91_line_61: GL tag disintegration.
  have row_61 : (gl_implication13 N N N add) := by
    exact row_27.1.2
  -- chapter_91_line_60: GL tag expansion.
  have row_60 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication13] using row_61
  -- chapter_91_line_103: GL tag implication.
  have row_103 : (gl_existence1 N v20 v12 add) := by
    apply row_60
    exact row_92
    exact row_104
  -- chapter_91_line_102: GL tag expansion.
  have row_102 : (¬ (∀ (v19 : α), ((N v19) → (¬ (add v20 v12 v19))))) := by
    simpa only [gl_existence1] using row_103
  have exists_row_102 : ∃ (v19 : α), ((N v19) ∧ (add v20 v12 v19)) := existsAndOfNotForallImpNot row_102
  obtain ⟨v19, witness_row_102⟩ := exists_row_102
  -- chapter_91_line_108: GL tag disintegration.
  have row_108 : (N v19) := by
    exact witness_row_102.1
  -- chapter_91_line_106: GL tag implication.
  have row_106 : (gl_preorder N add zero v19) := by
    apply row_107
    exact row_108
  -- chapter_91_line_101: GL tag disintegration.
  have row_101 : (add v20 v12 v19) := by
    exact witness_row_102.2
  -- chapter_91_line_59: GL tag implication.
  have row_59 : (gl_existence1 N previous zero add) := by
    apply row_60
    exact row_32
    exact row_62
  -- chapter_91_line_58: GL tag expansion.
  have row_58 : (¬ (∀ (v16 : α), ((N v16) → (¬ (add previous zero v16))))) := by
    simpa only [gl_existence1] using row_59
  have exists_row_58 : ∃ (v16 : α), ((N v16) ∧ (add previous zero v16)) := existsAndOfNotForallImpNot row_58
  obtain ⟨v16, witness_row_58⟩ := exists_row_58
  -- chapter_91_line_57: GL tag disintegration.
  have row_57 : (add previous zero v16) := by
    exact witness_row_58.2
  -- chapter_91_line_70: GL tag implication.
  have row_70 : (previous = v16) := by
    apply row_71
    exact row_32
    exact row_57
  -- chapter_91_line_26: GL tag disintegration.
  have row_26 : (gl_implication10 add N) := by
    exact row_27.1.1.2
  -- chapter_91_line_25: GL tag expansion.
  have row_25 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication10] using row_26
  -- chapter_91_line_24: GL tag implication.
  have row_24 : (N v1) := by
    apply row_25
    exact row_29
  -- chapter_91_line_91: GL tag implication.
  have row_91 : (gl_existence1 N v20 v1 add) := by
    apply row_60
    exact row_92
    exact row_24
  -- chapter_91_line_90: GL tag expansion.
  have row_90 : (¬ (∀ (v21 : α), ((N v21) → (¬ (add v20 v1 v21))))) := by
    simpa only [gl_existence1] using row_91
  have exists_row_90 : ∃ (v21 : α), ((N v21) ∧ (add v20 v1 v21)) := existsAndOfNotForallImpNot row_90
  obtain ⟨v21, witness_row_90⟩ := exists_row_90
  -- chapter_91_line_89: GL tag disintegration.
  have row_89 : (add v20 v1 v21) := by
    exact witness_row_90.2
  -- chapter_91_line_22: GL tag disintegration.
  have row_22 : (gl_fXYZ mul N N N) := by
    exact row_23.1.1.1.2
  -- chapter_91_line_21: GL tag expansion.
  have row_21 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_22
  -- chapter_91_line_20: GL tag disintegration.
  have row_20 : (gl_implication13 N N N mul) := by
    exact row_21.1.2
  -- chapter_91_line_19: GL tag expansion.
  have row_19 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication13] using row_20
  -- chapter_91_line_18: GL tag implication.
  have row_18 : (gl_existence1 N v1 previous mul) := by
    apply row_19
    exact row_24
    exact row_32
  -- chapter_91_line_17: GL tag expansion.
  have row_17 : (¬ (∀ (v8 : α), ((N v8) → (¬ (mul v1 previous v8))))) := by
    simpa only [gl_existence1] using row_18
  have exists_row_17 : ∃ (v8 : α), ((N v8) ∧ (mul v1 previous v8)) := existsAndOfNotForallImpNot row_17
  obtain ⟨v8, witness_row_17⟩ := exists_row_17
  -- chapter_91_line_16: GL tag disintegration.
  have row_16 : (mul v1 previous v8) := by
    exact witness_row_17.2
  -- chapter_91_line_123: GL tag equality1.
  have row_123 : (mul v1 v16 v8) := by
    have equality_source := row_16
    have equality_step_1 := row_70
    cases equality_step_1
    exact equality_source
  -- chapter_91_line_5: GL tag expansion for integration.
  have row_5 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_91_line_4: GL tag reformulation for integration and.
  have row_4 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_91_line_3: GL tag implication.
  have row_3 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_4
    exact row_6
    exact row_9
  have rule_row_122 := external_peano_externals_36_010 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_91_line_122: GL tag implication.
  have row_122 : (mul v1 v4 v5) := by
    apply rule_row_122
    exact row_14
  have rule_row_114 := external_peano_externals_36_035 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_91_line_114: GL tag implication.
  have row_114 : (gl_existence11 N one succ) := by
    apply rule_row_114
  -- chapter_91_line_113: GL tag expansion.
  have row_113 : (¬ (∀ (v18 : α), ((N v18) → (¬ (succ v18 one))))) := by
    simpa only [gl_existence11] using row_114
  have exists_row_113 : ∃ (v18 : α), ((N v18) ∧ (succ v18 one)) := existsAndOfNotForallImpNot row_113
  obtain ⟨v18, witness_row_113⟩ := exists_row_113
  -- chapter_91_line_112: GL tag disintegration.
  have row_112 : (succ v18 one) := by
    exact witness_row_113.2
  -- chapter_91_line_116: GL tag implication.
  have row_116 : (v18 = zero) := by
    apply row_110
    exact row_115
    exact row_112
    exact row_9
  -- chapter_91_line_109: GL tag implication.
  have row_109 : (zero = v18) := by
    apply row_110
    exact row_115
    exact row_9
    exact row_112
  -- chapter_91_line_105: GL tag equality1.
  have row_105 : (gl_preorder N add v18 v19) := by
    have equality_source := row_106
    have equality_step_1 := row_109
    cases equality_step_1
    exact equality_source
  have rule_row_87 := external_peano_externals_36_017 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_91_line_87: GL tag implication.
  have row_87 : (add v12 one v1) := by
    apply rule_row_87
    exact row_29
  have rule_row_86 := external_peano_externals_36_021 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_91_line_86: GL tag implication.
  have row_86 : (succ v12 v1) := by
    apply rule_row_86
    exact row_87
  have rule_row_63 := external_peano_externals_36_015 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_91_line_63: GL tag implication.
  have row_63 : (add previous one v2) := by
    apply rule_row_63
    exact row_37
  -- chapter_91_line_54: GL tag implication.
  have row_54 : (succ v16 v2) := by
    apply row_55
    exact row_62
    exact row_9
    exact row_57
    exact row_63
  have rule_row_50 := external_peano_externals_36_010 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_91_line_50: GL tag implication.
  have row_50 : (mul v1 v2 v3) := by
    apply rule_row_50
    exact row_51
  -- chapter_91_line_47: GL tag implication.
  have row_47 : (add v8 v1 v3) := by
    apply row_48
    exact row_32
    exact row_37
    exact row_16
    exact row_50
  have rule_row_99 := external_peano_externals_36_002 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_91_line_99: GL tag implication.
  have row_99 : (add v21 v8 v5) := by
    apply rule_row_99
    exact row_47
    exact row_89
    exact row_100
  -- chapter_91_line_45: GL tag implication.
  have row_45 : (gl_preorder N add v8 v3) := by
    apply row_46
    exact row_47
  have rule_row_38 := external_gauss_externals_24_021 N zero succ add mul one two identity (anchorGaussOfFTA N zero succ add mul one two identity anchor)
  -- chapter_91_line_38: GL tag implication.
  have row_38 : (gl_preorder N add v8 v5) := by
    apply rule_row_38
    exact row_45
    exact row_44
  -- chapter_91_line_98: GL tag expansion.
  have row_98 : (¬ (∀ (v17 : α), ((N v17) → (¬ (add v8 v17 v5))))) := by
    simpa only [gl_preorder] using row_38
  have exists_row_98 : ∃ (v17 : α), ((N v17) ∧ (add v8 v17 v5)) := existsAndOfNotForallImpNot row_98
  obtain ⟨v17, witness_row_98⟩ := exists_row_98
  -- chapter_91_line_117: GL tag disintegration.
  have row_117 : (N v17) := by
    exact witness_row_98.1
  -- chapter_91_line_97: GL tag disintegration.
  have row_97 : (add v8 v17 v5) := by
    exact witness_row_98.2
  have rule_row_96 := external_peano_externals_36_006 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_91_line_96: GL tag implication.
  have row_96 : (add v17 v8 v5) := by
    apply rule_row_96
    exact row_97
  -- chapter_91_line_94: GL tag implication.
  have row_94 : (v21 = v17) := by
    apply row_95
    exact row_99
    exact row_96
  -- chapter_91_line_88: GL tag equality1.
  have row_88 : (add v20 v1 v17) := by
    have equality_source := row_89
    have equality_step_1 := row_94
    cases equality_step_1
    exact equality_source
  -- chapter_91_line_85: GL tag implication.
  have row_85 : (succ v19 v17) := by
    apply row_55
    exact row_104
    exact row_86
    exact row_101
    exact row_88
  -- chapter_91_line_83: GL tag implication.
  have row_83 : (gl_strictOrder N add v18 v17) := by
    apply row_84
    exact row_85
    exact row_105
  -- chapter_91_line_82: GL tag equality1.
  have row_82 : (gl_strictOrder N add zero v17) := by
    have equality_source := row_83
    have equality_step_1 := row_116
    cases equality_step_1
    exact equality_source
  -- chapter_91_line_81: GL tag expansion.
  have row_81 : ((gl_preorder N add zero v17) ∧ (¬ (zero = v17))) := by
    simpa only [gl_strictOrder] using row_82
  -- chapter_91_line_80: GL tag disintegration.
  have row_80 : (¬ (zero = v17)) := by
    exact row_81.2
  -- chapter_91_line_78: GL tag implication.
  have row_78 : (gl_preorder N add one v17) := by
    apply row_79
    exact row_117
    exact row_80
  have rule_row_15 := external_peano_externals_36_010 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_91_line_15: GL tag implication.
  have row_15 : (mul previous v1 v8) := by
    apply rule_row_15
    exact row_16
  -- chapter_91_line_12: GL tag implication.
  have row_12 : (gl_preorder N add previous v4) := by
    apply row_13
    exact row_31
    exact row_15
    exact row_14
    exact row_38
  -- chapter_91_line_11: GL tag expansion.
  have row_11 : (¬ (∀ (v6 : α), ((N v6) → (¬ (add previous v6 v4))))) := by
    simpa only [gl_preorder] using row_12
  have exists_row_11 : ∃ (v6 : α), ((N v6) ∧ (add previous v6 v4)) := existsAndOfNotForallImpNot row_11
  obtain ⟨v6, witness_row_11⟩ := exists_row_11
  -- chapter_91_line_69: GL tag disintegration.
  have row_69 : (add previous v6 v4) := by
    exact witness_row_11.2
  -- chapter_91_line_68: GL tag equality1.
  have row_68 : (add v16 v6 v4) := by
    have equality_source := row_69
    have equality_step_1 := row_70
    cases equality_step_1
    exact equality_source
  -- chapter_91_line_120: GL tag implication.
  have row_120 : (mul v1 v6 v17) := by
    apply row_121
    exact row_123
    exact row_97
    exact row_122
    exact row_68
  -- chapter_91_line_118: GL tag implication.
  have row_118 : (gl_preorder N mul v6 v17) := by
    apply row_119
    exact row_120
  -- chapter_91_line_76: GL tag implication.
  have row_76 : (gl_preorder N add one v6) := by
    apply row_77
    exact row_118
    exact row_78
  -- chapter_91_line_74: GL tag implication.
  have row_74 : (¬ (zero = v6)) := by
    apply row_75
    exact row_76
  -- chapter_91_line_73: GL tag symmetry of inequality.
  have row_73 : (¬ (v6 = zero)) := by
    exact fun equality => row_74 (Eq.symm equality)
  have rule_row_67 := external_peano_externals_36_006 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_91_line_67: GL tag implication.
  have row_67 : (add v6 v16 v4) := by
    apply rule_row_67
    exact row_68
  -- chapter_91_line_10: GL tag disintegration.
  have row_10 : (N v6) := by
    exact witness_row_11.1
  have rule_row_2 := external_peano_externals_36_034 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_91_line_2: GL tag implication.
  have row_2 : (gl_or2 v6 zero N succ) := by
    apply rule_row_2
    exact row_10
  -- chapter_91_line_66: GL tag or disintegration.
  have row_66 : ((gl_existence11 N v6 succ) → (gl_existence11 N v6 succ)) := by
    intro scope_premise_1
    exact scope_premise_1
  -- chapter_91_line_65: GL tag expansion.
  have row_65 : ((gl_existence11 N v6 succ) → (¬ (∀ (v15 : α), ((N v15) → (¬ (succ v15 v6)))))) := by
    simpa only [gl_existence11] using row_66
  -- chapter_91_line_64: GL tag disintegration.
  have row_64 : ((gl_existence11 N v6 succ) → (∃ (v15 : α), (succ v15 v6))) := by
    intro scope_premise_1
    have scoped_fact_1 := row_65 scope_premise_1
    obtain ⟨v15, scoped_witness_bundle⟩ := existsAndOfNotForallImpNot scoped_fact_1
    exact ⟨v15, scoped_witness_bundle.2⟩
  have rule_row_53 := external_peano_externals_36_003 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_91_line_53: GL tag implication.
  have row_53 : ((gl_existence11 N v6 succ) → (∀ (v15 : α), ((succ v15 v6) → (add v2 v15 v4)))) := by
    intro scope_premise_1
    intro v15
    intro witness_guard_1
    have scoped_fact_4 := row_64 scope_premise_1
    apply rule_row_53
    exact row_67
    exact row_54
    exact witness_guard_1
  -- chapter_91_line_52: GL tag implication.
  have row_52 : ((gl_existence11 N v6 succ) → (∀ (v15 : α), ((succ v15 v6) → (gl_preorder N add v2 v4)))) := by
    intro scope_premise_1
    intro v15
    intro witness_guard_1
    have scoped_fact_2 := row_53 scope_premise_1 v15 witness_guard_1
    apply row_46
    exact scoped_fact_2
  -- chapter_91_line_1: GL tag or convergence.
  have row_1 : (gl_preorder N add v2 v4) := by
    classical
    have or_cases := row_2
    simp only [gl_or2] at or_cases
    rcases or_cases with or_branch_1 | or_branch_2
    · have branch_compact := or_branch_1
      exact False.elim (row_73 branch_compact)
    · have branch_compact := or_branch_2
      obtain ⟨v15, or_witness_bundle_2⟩ := existsAndOfNotForallImpNot branch_compact
      exact row_52 branch_compact v15 or_witness_bundle_2.2
  exact row_1

theorem fta_source_067
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_035 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (gl_existence11 N one succ))))
    (external_gauss_externals_24_018 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α) (x_7 : α) (x_8 : GLBinaryRelation α), ((gl_AnchorGauss x_1 x_2 x_3 x_4 x_5 x_6 x_7 x_8) → (∀ (x_9 : α) (x_10 : α), ((x_3 x_9 x_10) → (gl_preorder x_1 x_4 x_9 x_10))))))
    (external_gauss_externals_24_021 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α) (x_7 : α) (x_8 : GLBinaryRelation α), ((gl_AnchorGauss x_1 x_2 x_3 x_4 x_5 x_6 x_7 x_8) → (∀ (x_9 : α) (x_10 : α), ((gl_preorder x_1 x_4 x_9 x_10) → (∀ (x_11 : α), ((gl_preorder x_1 x_4 x_10 x_11) → (gl_preorder x_1 x_4 x_9 x_11))))))))
    (external_peano_externals_36_010 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_5 x_7 x_8 x_9) → (x_5 x_8 x_7 x_9))))))
    (external_peano_externals_36_017 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_6 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_021 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_6 x_8) → (x_3 x_7 x_8))))))
    (external_peano_externals_36_015 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_3 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_002 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (∀ (w4 : α) (w5 : α), ((add w4 w2 w5) → (∀ (w6 : α), ((add w3 w4 w6) → (add w5 w1 w6))))))))))
    (external_peano_externals_36_006 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (x_4 x_8 x_7 x_9))))))
    (external_peano_externals_36_034 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (∀ (w1 : α), ((N w1) → (gl_or2 w1 zero N succ))))))
    (external_peano_externals_36_003 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (∀ (x_10 : α), ((x_3 x_8 x_10) → (∀ (x_11 : α), ((x_3 x_11 x_7) → (x_4 x_10 x_11 x_9))))))))))
    (external_peano_externals_36_008 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (∀ (w4 : α) (w5 : α), ((mul w4 w1 w5) → (∀ (w6 : α), ((mul w4 w2 w6) → (∀ (w7 : α), ((mul w4 w3 w7) → (add w5 w6 w7))))))))))))
    (external_peano_externals_36_004 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (∀ (x_10 : α), ((x_4 x_7 x_10 x_9) → (x_8 = x_10))))))))
    (external_peano_externals_36_029 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((gl_preorder x_1 x_4 x_7 x_8) → ((gl_preorder x_1 x_4 x_8 x_7) → (x_7 = x_8)))))))
    (external_peano_externals_36_005 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → ((x_8 = x_9) → (x_2 = x_7)))))))
    (external_peano_externals_36_022 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_6 x_8) → (x_4 x_6 x_7 x_8))))))
    : (∀ (v1 : α), ((gl_preorder N add one v1) → (∀ (v2 : α) (v3 : α), ((mul v2 v1 v3) → (∀ (v4 : α) (v5 : α), ((mul v4 v1 v5) → ((gl_preorder N add v3 v5) → (gl_preorder N add v2 v4)))))))) := by
  intro v1
  intro premise_1
  intro v2
  intro v3
  intro premise_2
  intro v4
  intro v5
  intro premise_3
  intro premise_4
  have inductionMember : N v2 := by
    have typingRule := fta_source_067_induction_typing N zero succ add mul one two identity anchor relationalInduction
    exact typingRule v1 v2 v3 premise_2
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorFTA, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v1 : α), ((gl_preorder N add one v1) → (∀ (v3 : α), ((mul zero v1 v3) → (∀ (v4 : α) (v5 : α), ((mul v4 v1 v5) → ((gl_preorder N add v3 v5) → (gl_preorder N add zero v4)))))))) := by
    intro v1
    intro base_premise_1
    intro v3
    intro base_premise_2
    intro v4
    intro v5
    intro base_premise_3
    intro base_premise_4
    have zeroRule := fta_source_067_check_zero N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_035 external_gauss_externals_24_018 external_gauss_externals_24_021
    exact zeroRule v1 zero v4 v5 rfl base_premise_3
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v1 : α), ((gl_preorder N add one v1) → (∀ (v3 : α), ((mul induction_n v1 v3) → (∀ (v4 : α) (v5 : α), ((mul v4 v1 v5) → ((gl_preorder N add v3 v5) → (gl_preorder N add induction_n v4)))))))) → ∀ induction_m, succ induction_n induction_m → (∀ (v1 : α), ((gl_preorder N add one v1) → (∀ (v3 : α), ((mul induction_m v1 v3) → (∀ (v4 : α) (v5 : α), ((mul v4 v1 v5) → ((gl_preorder N add v3 v5) → (gl_preorder N add induction_m v4)))))))) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro v1
    intro step_premise_1
    intro v3
    intro step_premise_2
    intro v4
    intro v5
    intro step_premise_3
    intro step_premise_4
    have step_induction_assumption_1 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have step_induction_assumption_2 :
        ((gl_preorder N add one v1) → (∀ (w1 : α), ((mul induction_n v1 w1) → ((mul v4 v1 v5) → ((gl_preorder N add w1 v5) → (gl_preorder N add induction_n v4)))))) := by
      intro step_induction_assumption_2_premise_1
      intro w1
      intro step_induction_assumption_2_premise_2
      intro step_induction_assumption_2_premise_3
      intro step_induction_assumption_2_premise_4
      apply induction_hypothesis
      all_goals assumption
    have stepRule := fta_source_067_check_induction_condition N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_010 external_peano_externals_36_035 external_peano_externals_36_017 external_peano_externals_36_021 external_peano_externals_36_015 external_peano_externals_36_002 external_gauss_externals_24_021 external_peano_externals_36_006 external_peano_externals_36_034 external_peano_externals_36_003 external_peano_externals_36_008 external_peano_externals_36_004 external_gauss_externals_24_018 external_peano_externals_36_029 external_peano_externals_36_005 external_peano_externals_36_022
    exact stepRule induction_n v1 induction_m v3 v4 v5 step_premise_2 step_premise_4 step_induction_assumption_1 step_premise_1 step_premise_3 step_induction_assumption_2
  have inductionProperty : (∀ (v1 : α), ((gl_preorder N add one v1) → (∀ (v3 : α), ((mul v2 v1 v3) → (∀ (v4 : α) (v5 : α), ((mul v4 v1 v5) → ((gl_preorder N add v3 v5) → (gl_preorder N add v2 v4)))))))) := by
    exact relationalInduction N zero succ add mul one two identity anchor
      (fun induction_value => (∀ (v1 : α), ((gl_preorder N add one v1) → (∀ (v3 : α), ((mul induction_value v1 v3) → (∀ (v4 : α) (v5 : α), ((mul v4 v1 v5) → ((gl_preorder N add v3 v5) → (gl_preorder N add induction_value v4)))))))))
      v2
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v1 premise_1 v3 premise_2 v4 v5 premise_3 premise_4

private theorem fta_source_071_induction_typing
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (assumption_10 : (mul v1 v2 v3))
    : (N v1) := by
  -- chapter_97_line_10: GL tag task formulation.
  have row_10 : (mul v1 v2 v3) := by
    exact assumption_10
  -- chapter_97_line_9: GL tag task formulation.
  have row_9 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_97_line_8: GL tag expansion.
  have row_8 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_9
  -- chapter_97_line_7: GL tag disintegration.
  have row_7 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_8.1.1.1
  -- chapter_97_line_6: GL tag expansion.
  have row_6 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_7
  -- chapter_97_line_5: GL tag disintegration.
  have row_5 : (gl_fXYZ mul N N N) := by
    exact row_6.1.1.1.2
  -- chapter_97_line_4: GL tag expansion.
  have row_4 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_5
  -- chapter_97_line_3: GL tag disintegration.
  have row_3 : (gl_implication8 mul N) := by
    exact row_4.1.1.1.1
  -- chapter_97_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_3
  -- chapter_97_line_1: GL tag implication.
  have row_1 : (N v1) := by
    apply row_2
    exact row_10
  exact row_1

private theorem fta_source_071_check_zero
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_017 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_6 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_021 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_6 x_8) → (x_3 x_7 x_8))))))
    (v1 : α)
    (v3 : α)
    (assumption_7 : (gl_preorder N add one v1))
    (assumption_2 : (v1 = zero))
    : (gl_strictOrder N add v1 v3) := by
  -- chapter_98_line_7: GL tag task formulation.
  have row_7 : (gl_preorder N add one v1) := by
    exact assumption_7
  -- chapter_98_line_6: GL tag task formulation.
  have row_6 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_98_line_5: GL tag theorem.
  have row_5 := fta_source_066 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_017 external_peano_externals_36_021
  -- chapter_98_line_4: GL tag implication.
  have row_4 : (¬ (zero = v1)) := by
    apply row_5
    exact row_7
  -- chapter_98_line_3: GL tag symmetry of inequality.
  have row_3 : (¬ (v1 = zero)) := by
    exact fun equality => row_4 (Eq.symm equality)
  -- chapter_98_line_2: GL tag recursion.
  have row_2 : (v1 = zero) := by
    exact assumption_2
  -- chapter_98_line_1: GL tag vacuous truth.
  have row_1 : (gl_strictOrder N add v1 v3) := by
    exact False.elim (row_3 row_2)
  exact row_1

private theorem fta_source_071_check_induction_condition
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_006 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (x_4 x_8 x_7 x_9))))))
    (external_peano_externals_36_015 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_3 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_010 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_5 x_7 x_8 x_9) → (x_5 x_8 x_7 x_9))))))
    (external_peano_externals_36_017 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_6 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_021 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_6 x_8) → (x_3 x_7 x_8))))))
    (external_peano_externals_36_022 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_6 x_8) → (x_4 x_6 x_7 x_8))))))
    (external_gauss_externals_24_018 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α) (x_7 : α) (x_8 : GLBinaryRelation α), ((gl_AnchorGauss x_1 x_2 x_3 x_4 x_5 x_6 x_7 x_8) → (∀ (x_9 : α) (x_10 : α), ((x_3 x_9 x_10) → (gl_preorder x_1 x_4 x_9 x_10))))))
    (external_peano_externals_36_035 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (gl_existence11 N one succ))))
    (external_peano_externals_36_029 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((gl_preorder x_1 x_4 x_7 x_8) → ((gl_preorder x_1 x_4 x_8 x_7) → (x_7 = x_8)))))))
    (external_peano_externals_36_003 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (∀ (x_10 : α), ((x_3 x_8 x_10) → (∀ (x_11 : α), ((x_3 x_11 x_7) → (x_4 x_10 x_11 x_9))))))))))
    (external_peano_externals_36_005 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → ((x_8 = x_9) → (x_2 = x_7)))))))
    (external_peano_externals_36_033 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α), ((x_1 x_7) → (∀ (x_8 : α), ((x_5 x_6 x_8 x_7) → (x_7 = x_8))))))))
    (external_peano_externals_36_008 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (∀ (w4 : α) (w5 : α), ((mul w4 w1 w5) → (∀ (w6 : α), ((mul w4 w2 w6) → (∀ (w7 : α), ((mul w4 w3 w7) → (add w5 w6 w7))))))))))))
    (external_peano_externals_36_024 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_8 x_2) → ((x_1 x_8) → (x_2 = x_7)))))))
    (external_peano_externals_36_025 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_8 x_2) → (x_4 x_8 x_7 x_2))))))
    (previous : α)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (assumption_57 : (mul v1 v2 v3))
    (assumption_51 : (gl_preorder N add two v2))
    (assumption_33 : (gl_preorder N add one v1))
    (assumption_21 : (succ previous v1))
    : (gl_strictOrder N add v1 v3) := by
  -- chapter_99_line_101: GL tag theorem.
  have row_101 := fta_source_019 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_015
  -- chapter_99_line_97: GL tag theorem.
  have row_97 := fta_source_002 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_99_line_82: GL tag theorem.
  have row_82 := fta_source_014 N zero succ add mul one two identity anchor relationalInduction external_gauss_externals_24_018 external_peano_externals_36_035 external_peano_externals_36_029 external_peano_externals_36_003 external_peano_externals_36_005
  -- chapter_99_line_77: GL tag theorem.
  have row_77 := fta_source_031 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_99_line_63: GL tag theorem.
  have row_63 := fta_source_049 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_033
  -- chapter_99_line_57: GL tag task formulation.
  have row_57 : (mul v1 v2 v3) := by
    exact assumption_57
  -- chapter_99_line_51: GL tag task formulation.
  have row_51 : (gl_preorder N add two v2) := by
    exact assumption_51
  -- chapter_99_line_50: GL tag expansion.
  have row_50 : (¬ (∀ (v13 : α), ((N v13) → (¬ (add two v13 v2))))) := by
    simpa only [gl_preorder] using row_51
  have exists_row_50 : ∃ (v13 : α), ((N v13) ∧ (add two v13 v2)) := existsAndOfNotForallImpNot row_50
  obtain ⟨v13, witness_row_50⟩ := exists_row_50
  -- chapter_99_line_94: GL tag disintegration.
  have row_94 : (N v13) := by
    exact witness_row_50.1
  -- chapter_99_line_49: GL tag disintegration.
  have row_49 : (add two v13 v2) := by
    exact witness_row_50.2
  -- chapter_99_line_33: GL tag task formulation.
  have row_33 : (gl_preorder N add one v1) := by
    exact assumption_33
  -- chapter_99_line_32: GL tag expansion.
  have row_32 : (¬ (∀ (v11 : α), ((N v11) → (¬ (add one v11 v1))))) := by
    simpa only [gl_preorder] using row_33
  have exists_row_32 : ∃ (v11 : α), ((N v11) ∧ (add one v11 v1)) := existsAndOfNotForallImpNot row_32
  obtain ⟨v11, witness_row_32⟩ := exists_row_32
  -- chapter_99_line_52: GL tag disintegration.
  have row_52 : (N v11) := by
    exact witness_row_32.1
  -- chapter_99_line_31: GL tag disintegration.
  have row_31 : (add one v11 v1) := by
    exact witness_row_32.2
  -- chapter_99_line_28: GL tag expansion for integration.
  have row_28 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_99_line_27: GL tag reformulation for integration and.
  have row_27 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_99_line_21: GL tag recursion.
  have row_21 : (succ previous v1) := by
    exact assumption_21
  -- chapter_99_line_3: GL tag task formulation.
  have row_3 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_99_line_83: GL tag anchor handling.
  have row_83 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact row_3
  -- chapter_99_line_9: GL tag expansion.
  have row_9 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_3
  -- chapter_99_line_84: GL tag disintegration.
  have row_84 : (succ one two) := by
    exact row_9.1.2
  -- chapter_99_line_81: GL tag implication.
  have row_81 : (¬ (one = v2)) := by
    apply row_82
    exact row_84
    exact row_51
  -- chapter_99_line_29: GL tag disintegration.
  have row_29 : (succ zero one) := by
    exact row_9.1.1.2
  -- chapter_99_line_8: GL tag disintegration.
  have row_8 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_9.1.1.1
  -- chapter_99_line_26: GL tag implication.
  have row_26 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_27
    exact row_8
    exact row_29
  have rule_row_109 := external_peano_externals_36_006 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_99_line_109: GL tag implication.
  have row_109 : (add v13 two v2) := by
    apply rule_row_109
    exact row_49
  have rule_row_65 := external_peano_externals_36_015 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_99_line_65: GL tag implication.
  have row_65 : (add previous one v1) := by
    apply rule_row_65
    exact row_21
  have rule_row_56 := external_peano_externals_36_010 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_99_line_56: GL tag implication.
  have row_56 : (mul v2 v1 v3) := by
    apply rule_row_56
    exact row_57
  have rule_row_30 := external_peano_externals_36_017 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_99_line_30: GL tag implication.
  have row_30 : (add v11 one v1) := by
    apply rule_row_30
    exact row_31
  have rule_row_25 := external_peano_externals_36_021 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_99_line_25: GL tag implication.
  have row_25 : (succ v11 v1) := by
    apply rule_row_25
    exact row_30
  -- chapter_99_line_7: GL tag expansion.
  have row_7 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_8
  -- chapter_99_line_104: GL tag disintegration.
  have row_104 : (gl_implication17 N succ add) := by
    exact row_7.1.1.1.1.1.2
  -- chapter_99_line_103: GL tag expansion.
  have row_103 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication17] using row_104
  -- chapter_99_line_73: GL tag disintegration.
  have row_73 : (N zero) := by
    exact row_7.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_99_line_55: GL tag disintegration.
  have row_55 : (gl_implication21 N succ mul add) := by
    exact row_7.2
  -- chapter_99_line_54: GL tag expansion.
  have row_54 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((mul w3 w1 w4) → (∀ (w5 : α), ((mul w3 w2 w5) → (add w4 w3 w5))))))))) := by
    simpa only [gl_implication21] using row_55
  -- chapter_99_line_48: GL tag disintegration.
  have row_48 : (gl_fXYZ add N N N) := by
    exact row_7.1.1.1.1.1.1.1.1.2
  -- chapter_99_line_47: GL tag expansion.
  have row_47 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_48
  -- chapter_99_line_92: GL tag disintegration.
  have row_92 : (gl_implication13 N N N add) := by
    exact row_47.1.2
  -- chapter_99_line_91: GL tag expansion.
  have row_91 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication13] using row_92
  -- chapter_99_line_87: GL tag disintegration.
  have row_87 : (gl_implication14 N N add) := by
    exact row_47.2
  -- chapter_99_line_86: GL tag expansion.
  have row_86 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (∀ (w3 : α), ((add w1 w2 w3) → (∀ (w4 : α), ((add w1 w2 w4) → (w3 = w4))))))))) := by
    simpa only [gl_implication14] using row_87
  -- chapter_99_line_46: GL tag disintegration.
  have row_46 : (gl_implication10 add N) := by
    exact row_47.1.1.2
  -- chapter_99_line_45: GL tag expansion.
  have row_45 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication10] using row_46
  -- chapter_99_line_44: GL tag implication.
  have row_44 : (N v2) := by
    apply row_45
    exact row_49
  -- chapter_99_line_43: GL tag disintegration.
  have row_43 : (gl_fXYZ mul N N N) := by
    exact row_7.1.1.1.2
  -- chapter_99_line_42: GL tag expansion.
  have row_42 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_43
  -- chapter_99_line_41: GL tag disintegration.
  have row_41 : (gl_implication13 N N N mul) := by
    exact row_42.1.2
  -- chapter_99_line_40: GL tag expansion.
  have row_40 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication13] using row_41
  -- chapter_99_line_39: GL tag implication.
  have row_39 : (gl_existence1 N v2 v11 mul) := by
    apply row_40
    exact row_44
    exact row_52
  -- chapter_99_line_38: GL tag expansion.
  have row_38 : (¬ (∀ (v12 : α), ((N v12) → (¬ (mul v2 v11 v12))))) := by
    simpa only [gl_existence1] using row_39
  have exists_row_38 : ∃ (v12 : α), ((N v12) ∧ (mul v2 v11 v12)) := existsAndOfNotForallImpNot row_38
  obtain ⟨v12, witness_row_38⟩ := exists_row_38
  -- chapter_99_line_37: GL tag disintegration.
  have row_37 : (mul v2 v11 v12) := by
    exact witness_row_38.2
  -- chapter_99_line_53: GL tag implication.
  have row_53 : (add v12 v2 v3) := by
    apply row_54
    exact row_52
    exact row_25
    exact row_37
    exact row_56
  -- chapter_99_line_24: GL tag disintegration.
  have row_24 : (gl_implication7 N succ) := by
    exact row_7.1.1.1.1.1.1.1.1.1.2
  -- chapter_99_line_23: GL tag expansion.
  have row_23 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w2 w1) → (∀ (w3 : α), ((succ w3 w1) → (w2 = w3))))))) := by
    simpa only [gl_implication7] using row_24
  -- chapter_99_line_17: GL tag disintegration.
  have row_17 : (gl_fXY succ N N) := by
    exact row_7.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_99_line_16: GL tag expansion.
  have row_16 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_17
  -- chapter_99_line_68: GL tag disintegration.
  have row_68 : (gl_implication5 N succ) := by
    exact row_16.2
  -- chapter_99_line_67: GL tag expansion.
  have row_67 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α), ((succ w1 w3) → (w2 = w3))))))) := by
    simpa only [gl_implication5] using row_68
  -- chapter_99_line_36: GL tag disintegration.
  have row_36 : (gl_implication1 succ N) := by
    exact row_16.1.1.2
  -- chapter_99_line_35: GL tag expansion.
  have row_35 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using row_36
  -- chapter_99_line_93: GL tag implication.
  have row_93 : (N two) := by
    apply row_35
    exact row_84
  -- chapter_99_line_90: GL tag implication.
  have row_90 : (gl_existence1 N two v13 add) := by
    apply row_91
    exact row_93
    exact row_94
  -- chapter_99_line_89: GL tag expansion.
  have row_89 : (¬ (∀ (v14 : α), ((N v14) → (¬ (add two v13 v14))))) := by
    simpa only [gl_existence1] using row_90
  have exists_row_89 : ∃ (v14 : α), ((N v14) ∧ (add two v13 v14)) := existsAndOfNotForallImpNot row_89
  obtain ⟨v14, witness_row_89⟩ := exists_row_89
  -- chapter_99_line_88: GL tag disintegration.
  have row_88 : (add two v13 v14) := by
    exact witness_row_89.2
  -- chapter_99_line_95: GL tag implication.
  have row_95 : (v14 = v2) := by
    apply row_86
    exact row_93
    exact row_94
    exact row_88
    exact row_49
  -- chapter_99_line_85: GL tag implication.
  have row_85 : (v2 = v14) := by
    apply row_86
    exact row_93
    exact row_94
    exact row_49
    exact row_88
  -- chapter_99_line_80: GL tag equality1.
  have row_80 : (¬ (one = v14)) := by
    have equality_source := row_81
    have equality_step_1 := row_85
    cases equality_step_1
    exact equality_source
  -- chapter_99_line_34: GL tag implication.
  have row_34 : (N v1) := by
    apply row_35
    exact row_21
  -- chapter_99_line_64: GL tag implication.
  have row_64 : (v11 = previous) := by
    apply row_23
    exact row_34
    exact row_25
    exact row_21
  -- chapter_99_line_62: GL tag implication.
  have row_62 : (mul one v11 previous) := by
    apply row_63
    exact row_52
    exact row_64
  -- chapter_99_line_22: GL tag implication.
  have row_22 : (previous = v11) := by
    apply row_23
    exact row_34
    exact row_21
    exact row_25
  -- chapter_99_line_20: GL tag disintegration.
  have row_20 : (gl_implication0 succ N) := by
    exact row_16.1.1.1
  -- chapter_99_line_19: GL tag expansion.
  have row_19 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_20
  -- chapter_99_line_108: GL tag implication.
  have row_108 : (N one) := by
    apply row_19
    exact row_84
  -- chapter_99_line_107: GL tag implication.
  have row_107 : (gl_existence1 N v13 one add) := by
    apply row_91
    exact row_94
    exact row_108
  -- chapter_99_line_106: GL tag expansion.
  have row_106 : (¬ (∀ (v15 : α), ((N v15) → (¬ (add v13 one v15))))) := by
    simpa only [gl_existence1] using row_107
  have exists_row_106 : ∃ (v15 : α), ((N v15) ∧ (add v13 one v15)) := existsAndOfNotForallImpNot row_106
  obtain ⟨v15, witness_row_106⟩ := exists_row_106
  -- chapter_99_line_105: GL tag disintegration.
  have row_105 : (add v13 one v15) := by
    exact witness_row_106.2
  -- chapter_99_line_102: GL tag implication.
  have row_102 : (succ v15 v2) := by
    apply row_103
    exact row_108
    exact row_84
    exact row_105
    exact row_109
  -- chapter_99_line_100: GL tag implication.
  have row_100 : (add v15 one v2) := by
    apply row_101
    exact row_102
  have rule_row_99 := external_peano_externals_36_022 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_99_line_99: GL tag implication.
  have row_99 : (add one v15 v2) := by
    apply rule_row_99
    exact row_100
  -- chapter_99_line_18: GL tag implication.
  have row_18 : (N previous) := by
    apply row_19
    exact row_21
  -- chapter_99_line_15: GL tag disintegration.
  have row_15 : (gl_implication4 N N succ) := by
    exact row_16.1.2
  -- chapter_99_line_14: GL tag expansion.
  have row_14 : (∀ (w1 : α), ((N w1) → (gl_existence0 N w1 succ))) := by
    simpa only [gl_implication4] using row_15
  -- chapter_99_line_72: GL tag implication.
  have row_72 : (gl_existence0 N zero succ) := by
    apply row_14
    exact row_73
  -- chapter_99_line_71: GL tag expansion.
  have row_71 : (¬ (∀ (v5 : α), ((N v5) → (¬ (succ zero v5))))) := by
    simpa only [gl_existence0] using row_72
  have exists_row_71 : ∃ (v5 : α), ((N v5) ∧ (succ zero v5)) := existsAndOfNotForallImpNot row_71
  obtain ⟨v5, witness_row_71⟩ := exists_row_71
  -- chapter_99_line_70: GL tag disintegration.
  have row_70 : (succ zero v5) := by
    exact witness_row_71.2
  -- chapter_99_line_69: GL tag implication.
  have row_69 : (one = v5) := by
    apply row_67
    exact row_73
    exact row_29
    exact row_70
  -- chapter_99_line_98: GL tag equality1.
  have row_98 : (add v5 v15 v2) := by
    have equality_source := row_99
    have equality_step_1 := row_69
    cases equality_step_1
    exact equality_source
  -- chapter_99_line_96: GL tag implication.
  have row_96 : (gl_preorder N add v5 v2) := by
    apply row_97
    exact row_98
  -- chapter_99_line_79: GL tag equality1.
  have row_79 : (¬ (v5 = v14)) := by
    have equality_source := row_80
    have equality_step_1 := row_69
    cases equality_step_1
    exact equality_source
  -- chapter_99_line_78: GL tag equality1.
  have row_78 : (¬ (v5 = v2)) := by
    have equality_source := row_79
    have equality_step_1 := row_95
    cases equality_step_1
    exact equality_source
  -- chapter_99_line_76: GL tag implication.
  have row_76 : (gl_strictOrder N add v5 v2) := by
    apply row_77
    exact row_96
    exact row_78
  -- chapter_99_line_13: GL tag implication.
  have row_13 : (gl_existence0 N previous succ) := by
    apply row_14
    exact row_18
  -- chapter_99_line_12: GL tag expansion.
  have row_12 : (¬ (∀ (v4 : α), ((N v4) → (¬ (succ previous v4))))) := by
    simpa only [gl_existence0] using row_13
  have exists_row_12 : ∃ (v4 : α), ((N v4) ∧ (succ previous v4)) := existsAndOfNotForallImpNot row_12
  obtain ⟨v4, witness_row_12⟩ := exists_row_12
  -- chapter_99_line_11: GL tag disintegration.
  have row_11 : (succ previous v4) := by
    exact witness_row_12.2
  -- chapter_99_line_10: GL tag equality1.
  have row_10 : (succ v11 v4) := by
    have equality_source := row_11
    have equality_step_1 := row_22
    cases equality_step_1
    exact equality_source
  -- chapter_99_line_74: GL tag implication.
  have row_74 : (v1 = v4) := by
    apply row_67
    exact row_52
    exact row_25
    exact row_10
  -- chapter_99_line_75: GL tag equality1.
  have row_75 : (gl_preorder N add one v4) := by
    have equality_source := row_33
    have equality_step_1 := row_74
    cases equality_step_1
    exact equality_source
  -- chapter_99_line_66: GL tag implication.
  have row_66 : (v4 = v1) := by
    apply row_67
    exact row_52
    exact row_10
    exact row_25
  -- chapter_99_line_6: GL tag disintegration.
  have row_6 : (gl_implication20 N succ mul add) := by
    exact row_7.1.2
  -- chapter_99_line_5: GL tag expansion.
  have row_5 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((mul w3 w1 w4) → (∀ (w5 : α), ((add w4 w3 w5) → (mul w3 w2 w5))))))))) := by
    simpa only [gl_implication20] using row_6
  -- chapter_99_line_61: GL tag implication.
  have row_61 : (mul one v4 v1) := by
    apply row_5
    exact row_52
    exact row_10
    exact row_62
    exact row_65
  -- chapter_99_line_60: GL tag equality1.
  have row_60 : (mul one v1 v1) := by
    have equality_source := row_61
    have equality_step_1 := row_66
    cases equality_step_1
    exact equality_source
  -- chapter_99_line_59: GL tag equality1.
  have row_59 : (mul v5 v1 v1) := by
    have equality_source := row_60
    have equality_step_1 := row_69
    cases equality_step_1
    exact equality_source
  -- chapter_99_line_58: GL tag equality1.
  have row_58 : (mul v5 v4 v1) := by
    have equality_source := row_59
    have equality_step_1 := row_74
    cases equality_step_1
    exact equality_source
  -- chapter_99_line_4: GL tag implication.
  have row_4 : (mul v2 v4 v3) := by
    apply row_5
    exact row_52
    exact row_10
    exact row_37
    exact row_53
  -- chapter_99_line_2: GL tag theorem.
  have row_2 := fta_source_048 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_006 external_peano_externals_36_010 external_peano_externals_36_008 external_peano_externals_36_005 external_peano_externals_36_017 external_peano_externals_36_021 external_peano_externals_36_035 external_peano_externals_36_024 external_peano_externals_36_025
  -- chapter_99_line_1: GL tag implication.
  have row_1 : (gl_strictOrder N add v1 v3) := by
    apply row_2
    exact row_76
    exact row_75
    exact row_58
    exact row_4
  exact row_1

theorem fta_source_071
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_017 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_6 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_021 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_6 x_8) → (x_3 x_7 x_8))))))
    (external_peano_externals_36_006 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (x_4 x_8 x_7 x_9))))))
    (external_peano_externals_36_015 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_3 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_010 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_5 x_7 x_8 x_9) → (x_5 x_8 x_7 x_9))))))
    (external_peano_externals_36_022 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_6 x_8) → (x_4 x_6 x_7 x_8))))))
    (external_gauss_externals_24_018 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α) (x_7 : α) (x_8 : GLBinaryRelation α), ((gl_AnchorGauss x_1 x_2 x_3 x_4 x_5 x_6 x_7 x_8) → (∀ (x_9 : α) (x_10 : α), ((x_3 x_9 x_10) → (gl_preorder x_1 x_4 x_9 x_10))))))
    (external_peano_externals_36_035 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (gl_existence11 N one succ))))
    (external_peano_externals_36_029 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((gl_preorder x_1 x_4 x_7 x_8) → ((gl_preorder x_1 x_4 x_8 x_7) → (x_7 = x_8)))))))
    (external_peano_externals_36_003 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (∀ (x_10 : α), ((x_3 x_8 x_10) → (∀ (x_11 : α), ((x_3 x_11 x_7) → (x_4 x_10 x_11 x_9))))))))))
    (external_peano_externals_36_005 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → ((x_8 = x_9) → (x_2 = x_7)))))))
    (external_peano_externals_36_033 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α), ((x_1 x_7) → (∀ (x_8 : α), ((x_5 x_6 x_8 x_7) → (x_7 = x_8))))))))
    (external_peano_externals_36_008 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (∀ (w4 : α) (w5 : α), ((mul w4 w1 w5) → (∀ (w6 : α), ((mul w4 w2 w6) → (∀ (w7 : α), ((mul w4 w3 w7) → (add w5 w6 w7))))))))))))
    (external_peano_externals_36_024 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_8 x_2) → ((x_1 x_8) → (x_2 = x_7)))))))
    (external_peano_externals_36_025 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_8 x_2) → (x_4 x_8 x_7 x_2))))))
    : (∀ (v1 : α), ((gl_preorder N add one v1) → (∀ (v2 : α), ((gl_preorder N add two v2) → (∀ (v3 : α), ((mul v1 v2 v3) → (gl_strictOrder N add v1 v3))))))) := by
  intro v1
  intro premise_1
  intro v2
  intro premise_2
  intro v3
  intro premise_3
  have inductionMember : N v1 := by
    have typingRule := fta_source_071_induction_typing N zero succ add mul one two identity anchor relationalInduction
    exact typingRule v1 v2 v3 premise_3
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorFTA, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : ((gl_preorder N add one zero) → (∀ (v2 : α), ((gl_preorder N add two v2) → (∀ (v3 : α), ((mul zero v2 v3) → (gl_strictOrder N add zero v3)))))) := by
    intro base_premise_1
    intro v2
    intro base_premise_2
    intro v3
    intro base_premise_3
    have zeroRule := fta_source_071_check_zero N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_017 external_peano_externals_36_021
    exact zeroRule zero v3 base_premise_1 rfl
  have inductionStep :
      ∀ induction_n, N induction_n → ((gl_preorder N add one induction_n) → (∀ (v2 : α), ((gl_preorder N add two v2) → (∀ (v3 : α), ((mul induction_n v2 v3) → (gl_strictOrder N add induction_n v3)))))) → ∀ induction_m, succ induction_n induction_m → ((gl_preorder N add one induction_m) → (∀ (v2 : α), ((gl_preorder N add two v2) → (∀ (v3 : α), ((mul induction_m v2 v3) → (gl_strictOrder N add induction_m v3)))))) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro step_premise_1
    intro v2
    intro step_premise_2
    intro v3
    intro step_premise_3
    have step_induction_assumption_1 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have stepRule := fta_source_071_check_induction_condition N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_006 external_peano_externals_36_015 external_peano_externals_36_010 external_peano_externals_36_017 external_peano_externals_36_021 external_peano_externals_36_022 external_gauss_externals_24_018 external_peano_externals_36_035 external_peano_externals_36_029 external_peano_externals_36_003 external_peano_externals_36_005 external_peano_externals_36_033 external_peano_externals_36_008 external_peano_externals_36_024 external_peano_externals_36_025
    exact stepRule induction_n induction_m v2 v3 step_premise_3 step_premise_2 step_premise_1 step_induction_assumption_1
  have inductionProperty : ((gl_preorder N add one v1) → (∀ (v2 : α), ((gl_preorder N add two v2) → (∀ (v3 : α), ((mul v1 v2 v3) → (gl_strictOrder N add v1 v3)))))) := by
    exact relationalInduction N zero succ add mul one two identity anchor
      (fun induction_value => ((gl_preorder N add one induction_value) → (∀ (v2 : α), ((gl_preorder N add two v2) → (∀ (v3 : α), ((mul induction_value v2 v3) → (gl_strictOrder N add induction_value v3)))))))
      v1
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty premise_1 v2 premise_2 v3 premise_3

private theorem fta_source_004_induction_typing
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (v2 : α)
    (v4 : α)
    (v5 : α)
    (assumption_10 : (mul v4 v2 v5))
    : (N v2) := by
  -- chapter_4_line_10: GL tag task formulation.
  have row_10 : (mul v4 v2 v5) := by
    exact assumption_10
  -- chapter_4_line_9: GL tag task formulation.
  have row_9 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_4_line_8: GL tag expansion.
  have row_8 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_9
  -- chapter_4_line_7: GL tag disintegration.
  have row_7 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_8.1.1.1
  -- chapter_4_line_6: GL tag expansion.
  have row_6 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_7
  -- chapter_4_line_5: GL tag disintegration.
  have row_5 : (gl_fXYZ mul N N N) := by
    exact row_6.1.1.1.2
  -- chapter_4_line_4: GL tag expansion.
  have row_4 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_5
  -- chapter_4_line_3: GL tag disintegration.
  have row_3 : (gl_implication9 mul N) := by
    exact row_4.1.1.1.2
  -- chapter_4_line_2: GL tag expansion.
  have row_2 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication9] using row_3
  -- chapter_4_line_1: GL tag implication.
  have row_1 : (N v2) := by
    apply row_2
    exact row_10
  exact row_1

private theorem fta_source_004_check_zero
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_035 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (gl_existence11 N one succ))))
    (external_peano_externals_36_010 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_5 x_7 x_8 x_9) → (x_5 x_8 x_7 x_9))))))
    (external_peano_externals_36_009 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (∀ (w4 : α) (w5 : α), ((mul w4 w2 w5) → (∀ (w6 : α), ((mul w3 w4 w6) → (mul w5 w1 w6))))))))))
    (external_peano_externals_36_031 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α), ((x_1 x_7) → (∀ (x_8 : α), ((x_5 x_2 x_7 x_8) → (x_2 = x_8))))))))
    (external_peano_externals_36_033 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α), ((x_1 x_7) → (∀ (x_8 : α), ((x_5 x_6 x_8 x_7) → (x_7 = x_8))))))))
    (external_peano_externals_36_018 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_5 x_6 x_7 x_8) → (x_5 x_7 x_6 x_8))))))
    (external_peano_externals_36_007 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_5 x_7 x_8 x_9) → (∀ (x_10 : α) (x_11 : α) (x_12 : α), ((x_5 x_10 x_11 x_12) → ((x_5 x_8 x_7 x_11) → (x_5 x_10 x_9 x_12)))))))))
    (external_peano_externals_36_029 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((gl_preorder x_1 x_4 x_7 x_8) → ((gl_preorder x_1 x_4 x_8 x_7) → (x_7 = x_8)))))))
    (external_peano_externals_36_017 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_6 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_021 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_6 x_8) → (x_3 x_7 x_8))))))
    (external_peano_externals_36_015 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_3 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_022 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_6 x_8) → (x_4 x_6 x_7 x_8))))))
    (external_peano_externals_36_008 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (∀ (w4 : α) (w5 : α), ((mul w4 w1 w5) → (∀ (w6 : α), ((mul w4 w2 w6) → (∀ (w7 : α), ((mul w4 w3 w7) → (add w5 w6 w7))))))))))))
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (v4 : α)
    (v5 : α)
    (assumption_52 : (mul v4 v2 v5))
    (assumption_30 : (mul v1 v2 v3))
    (assumption_15 : (gl_strictOrder N add v3 v5))
    (assumption_2 : (v2 = zero))
    : (gl_strictOrder N add v1 v4) := by
  -- chapter_5_line_77: GL tag theorem.
  have row_77 := fta_source_036 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_009 external_peano_externals_36_010
  -- chapter_5_line_60: GL tag theorem.
  have row_60 := fta_source_076 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_031
  -- chapter_5_line_52: GL tag task formulation.
  have row_52 : (mul v4 v2 v5) := by
    exact assumption_52
  -- chapter_5_line_51: GL tag theorem.
  have row_51 := fta_source_010 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_010
  -- chapter_5_line_39: GL tag theorem.
  have row_39 := fta_source_065 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_035 external_peano_externals_36_033 external_peano_externals_36_018 external_peano_externals_36_009 external_peano_externals_36_010
  -- chapter_5_line_30: GL tag task formulation.
  have row_30 : (mul v1 v2 v3) := by
    exact assumption_30
  -- chapter_5_line_26: GL tag expansion for integration.
  have row_26 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_5_line_25: GL tag reformulation for integration and.
  have row_25 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_5_line_22: GL tag expansion for integration.
  have row_22 : ((gl_preorder N mul v2 v3) ↔ (¬ (∀ (v9 : α), ((N v9) → (¬ (mul v2 v9 v3)))))) := by
    exact Iff.rfl
  -- chapter_5_line_21: GL tag reformulation for integration >[bound].
  have row_21 : (∀ (v8 : α), ((N v8) → ((mul v2 v8 v3) → (gl_preorder N mul v2 v3)))) := by
    intro v8
    intro integration_premise_1
    intro integration_premise_2
    apply (row_22).2
    intro universal_counterexample
    exact universal_counterexample v8 integration_premise_1 integration_premise_2
  -- chapter_5_line_19: GL tag task formulation.
  have row_19 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_5_line_50: GL tag implication.
  have row_50 : (gl_preorder N mul v2 v5) := by
    apply row_51
    exact row_52
  -- chapter_5_line_28: GL tag expansion.
  have row_28 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_19
  -- chapter_5_line_73: GL tag disintegration.
  have row_73 : (succ one two) := by
    exact row_28.1.2
  -- chapter_5_line_29: GL tag disintegration.
  have row_29 : (succ zero one) := by
    exact row_28.1.1.2
  -- chapter_5_line_27: GL tag disintegration.
  have row_27 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_28.1.1.1
  -- chapter_5_line_36: GL tag expansion.
  have row_36 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_27
  -- chapter_5_line_72: GL tag disintegration.
  have row_72 : (gl_fXY succ N N) := by
    exact row_36.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_5_line_71: GL tag expansion.
  have row_71 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_72
  -- chapter_5_line_70: GL tag disintegration.
  have row_70 : (gl_implication0 succ N) := by
    exact row_71.1.1.1
  -- chapter_5_line_69: GL tag expansion.
  have row_69 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_70
  -- chapter_5_line_68: GL tag implication.
  have row_68 : (N one) := by
    apply row_69
    exact row_73
  -- chapter_5_line_64: GL tag disintegration.
  have row_64 : (gl_implication7 N succ) := by
    exact row_36.1.1.1.1.1.1.1.1.1.2
  -- chapter_5_line_63: GL tag expansion.
  have row_63 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w2 w1) → (∀ (w3 : α), ((succ w3 w1) → (w2 = w3))))))) := by
    simpa only [gl_implication7] using row_64
  -- chapter_5_line_44: GL tag disintegration.
  have row_44 : (gl_fXYZ add N N N) := by
    exact row_36.1.1.1.1.1.1.1.1.2
  -- chapter_5_line_43: GL tag expansion.
  have row_43 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_44
  -- chapter_5_line_57: GL tag disintegration.
  have row_57 : (gl_implication10 add N) := by
    exact row_43.1.1.2
  -- chapter_5_line_56: GL tag expansion.
  have row_56 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication10] using row_57
  -- chapter_5_line_42: GL tag disintegration.
  have row_42 : (gl_implication8 add N) := by
    exact row_43.1.1.1.1
  -- chapter_5_line_41: GL tag expansion.
  have row_41 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_42
  -- chapter_5_line_35: GL tag disintegration.
  have row_35 : (gl_fXYZ mul N N N) := by
    exact row_36.1.1.1.2
  -- chapter_5_line_34: GL tag expansion.
  have row_34 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_35
  -- chapter_5_line_83: GL tag disintegration.
  have row_83 : (gl_implication13 N N N mul) := by
    exact row_34.1.2
  -- chapter_5_line_82: GL tag expansion.
  have row_82 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication13] using row_83
  -- chapter_5_line_33: GL tag disintegration.
  have row_33 : (gl_implication8 mul N) := by
    exact row_34.1.1.1.1
  -- chapter_5_line_32: GL tag expansion.
  have row_32 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_33
  -- chapter_5_line_31: GL tag implication.
  have row_31 : (N v1) := by
    apply row_32
    exact row_30
  -- chapter_5_line_24: GL tag implication.
  have row_24 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_25
    exact row_27
    exact row_29
  have rule_row_67 := external_peano_externals_36_035 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_5_line_67: GL tag implication.
  have row_67 : (gl_existence11 N one succ) := by
    apply rule_row_67
  -- chapter_5_line_66: GL tag expansion.
  have row_66 : (¬ (∀ (v7 : α), ((N v7) → (¬ (succ v7 one))))) := by
    simpa only [gl_existence11] using row_67
  have exists_row_66 : ∃ (v7 : α), ((N v7) ∧ (succ v7 one)) := existsAndOfNotForallImpNot row_66
  obtain ⟨v7, witness_row_66⟩ := exists_row_66
  -- chapter_5_line_65: GL tag disintegration.
  have row_65 : (succ v7 one) := by
    exact witness_row_66.2
  -- chapter_5_line_62: GL tag implication.
  have row_62 : (zero = v7) := by
    apply row_63
    exact row_68
    exact row_29
    exact row_65
  have rule_row_23 := external_peano_externals_36_010 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_5_line_23: GL tag implication.
  have row_23 : (mul v2 v1 v3) := by
    apply rule_row_23
    exact row_30
  -- chapter_5_line_20: GL tag implication.
  have row_20 : (gl_preorder N mul v2 v3) := by
    apply row_21
    exact row_31
    exact row_23
  -- chapter_5_line_18: GL tag compilation.
  have row_18 : (gl_implication199 (α := α)) := by
    simp only [gl_implication199]
    intro compiled_N compiled_zero compiled_succ compiled_add compiled_mul compiled_one compiled_two compiled_identity
    intro compiled_anchor
    exact fta_source_043 compiled_N compiled_zero compiled_succ compiled_add compiled_mul compiled_one compiled_two compiled_identity compiled_anchor relationalInduction external_peano_externals_36_031 external_peano_externals_36_009 external_peano_externals_36_010 external_peano_externals_36_007 external_peano_externals_36_029 external_peano_externals_36_017 external_peano_externals_36_021 external_peano_externals_36_015 external_peano_externals_36_022 external_peano_externals_36_033 external_peano_externals_36_008
  -- chapter_5_line_17: GL tag expansion.
  have row_17 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α) (two : α) (identity : GLBinaryRelation α), ((gl_AnchorFTA N zero succ add mul one two identity) → (∀ (v1 : α) (v2 : α), ((gl_preorder N mul v1 v2) → ((gl_preorder N mul v2 v1) → (v1 = v2)))))) := by
    simpa only [gl_implication199] using row_18
  -- chapter_5_line_15: GL tag task formulation.
  have row_15 : (gl_strictOrder N add v3 v5) := by
    exact assumption_15
  -- chapter_5_line_14: GL tag expansion.
  have row_14 : ((gl_preorder N add v3 v5) ∧ (¬ (v3 = v5))) := by
    simpa only [gl_strictOrder] using row_15
  -- chapter_5_line_47: GL tag disintegration.
  have row_47 : (gl_preorder N add v3 v5) := by
    exact row_14.1
  -- chapter_5_line_46: GL tag expansion.
  have row_46 : (¬ (∀ (v13 : α), ((N v13) → (¬ (add v3 v13 v5))))) := by
    simpa only [gl_preorder] using row_47
  have exists_row_46 : ∃ (v13 : α), ((N v13) ∧ (add v3 v13 v5)) := existsAndOfNotForallImpNot row_46
  obtain ⟨v13, witness_row_46⟩ := exists_row_46
  -- chapter_5_line_45: GL tag disintegration.
  have row_45 : (add v3 v13 v5) := by
    exact witness_row_46.2
  -- chapter_5_line_55: GL tag implication.
  have row_55 : (N v5) := by
    apply row_56
    exact row_45
  -- chapter_5_line_81: GL tag implication.
  have row_81 : (gl_existence1 N v5 v1 mul) := by
    apply row_82
    exact row_55
    exact row_31
  -- chapter_5_line_80: GL tag expansion.
  have row_80 : (¬ (∀ (v6 : α), ((N v6) → (¬ (mul v5 v1 v6))))) := by
    simpa only [gl_existence1] using row_81
  have exists_row_80 : ∃ (v6 : α), ((N v6) ∧ (mul v5 v1 v6)) := existsAndOfNotForallImpNot row_80
  obtain ⟨v6, witness_row_80⟩ := exists_row_80
  -- chapter_5_line_88: GL tag disintegration.
  have row_88 : (N v6) := by
    exact witness_row_80.1
  -- chapter_5_line_87: GL tag implication.
  have row_87 : (gl_preorder N mul v6 zero) := by
    apply row_39
    exact row_88
  -- chapter_5_line_86: GL tag equality1.
  have row_86 : (gl_preorder N mul v6 v7) := by
    have equality_source := row_87
    have equality_step_1 := row_62
    cases equality_step_1
    exact equality_source
  -- chapter_5_line_79: GL tag disintegration.
  have row_79 : (mul v5 v1 v6) := by
    exact witness_row_80.2
  have rule_row_78 := external_peano_externals_36_009 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_5_line_78: GL tag implication.
  have row_78 : (mul v3 v4 v6) := by
    apply rule_row_78
    exact row_52
    exact row_30
    exact row_79
  -- chapter_5_line_76: GL tag implication.
  have row_76 : (gl_preorder N mul v2 v6) := by
    apply row_77
    exact row_20
    exact row_78
  -- chapter_5_line_54: GL tag implication.
  have row_54 : (gl_preorder N mul v5 zero) := by
    apply row_39
    exact row_55
  -- chapter_5_line_40: GL tag implication.
  have row_40 : (N v3) := by
    apply row_41
    exact row_45
  -- chapter_5_line_38: GL tag implication.
  have row_38 : (gl_preorder N mul v3 zero) := by
    apply row_39
    exact row_40
  -- chapter_5_line_13: GL tag disintegration.
  have row_13 : (¬ (v3 = v5)) := by
    exact row_14.2
  -- chapter_5_line_2: GL tag recursion.
  have row_2 : (v2 = zero) := by
    exact assumption_2
  -- chapter_5_line_95: GL tag equality1.
  have row_95 : (gl_preorder N mul zero v3) := by
    have equality_source := row_20
    have equality_step_1 := row_2
    cases equality_step_1
    exact equality_source
  -- chapter_5_line_94: GL tag implication.
  have row_94 : (zero = v3) := by
    apply row_60
    exact row_95
  -- chapter_5_line_93: GL tag symmetry of equality.
  have row_93 : (v3 = zero) := by
    exact Eq.symm row_94
  -- chapter_5_line_61: GL tag equality1.
  have row_61 : (gl_preorder N mul zero v5) := by
    have equality_source := row_50
    have equality_step_1 := row_2
    cases equality_step_1
    exact equality_source
  -- chapter_5_line_59: GL tag implication.
  have row_59 : (zero = v5) := by
    apply row_60
    exact row_61
  -- chapter_5_line_58: GL tag symmetry of equality.
  have row_58 : (v5 = zero) := by
    exact Eq.symm row_59
  -- chapter_5_line_48: GL tag symmetry of equality.
  have row_48 : (zero = v2) := by
    exact Eq.symm row_2
  -- chapter_5_line_91: GL tag equality1.
  have row_91 : (gl_preorder N mul v6 v2) := by
    have equality_source := row_87
    have equality_step_1 := row_48
    cases equality_step_1
    exact equality_source
  have rule_row_90 := fta_source_043 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_031 external_peano_externals_36_009 external_peano_externals_36_010 external_peano_externals_36_007 external_peano_externals_36_029 external_peano_externals_36_017 external_peano_externals_36_021 external_peano_externals_36_015 external_peano_externals_36_022 external_peano_externals_36_033 external_peano_externals_36_008
  -- chapter_5_line_90: GL tag implication.
  have row_90 : (v6 = v2) := by
    apply rule_row_90
    exact row_91
    exact row_76
  -- chapter_5_line_85: GL tag equality1.
  have row_85 : (succ v2 one) := by
    have equality_source := row_29
    have equality_step_1 := row_48
    cases equality_step_1
    exact equality_source
  -- chapter_5_line_84: GL tag implication.
  have row_84 : (v2 = v7) := by
    apply row_63
    exact row_68
    exact row_85
    exact row_65
  -- chapter_5_line_75: GL tag equality1.
  have row_75 : (gl_preorder N mul v7 v6) := by
    have equality_source := row_76
    have equality_step_1 := row_84
    cases equality_step_1
    exact equality_source
  have rule_row_74 := fta_source_043 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_031 external_peano_externals_36_009 external_peano_externals_36_010 external_peano_externals_36_007 external_peano_externals_36_029 external_peano_externals_36_017 external_peano_externals_36_021 external_peano_externals_36_015 external_peano_externals_36_022 external_peano_externals_36_033 external_peano_externals_36_008
  -- chapter_5_line_74: GL tag implication.
  have row_74 : (v7 = v6) := by
    apply rule_row_74
    exact row_75
    exact row_86
  -- chapter_5_line_53: GL tag equality1.
  have row_53 : (gl_preorder N mul v5 v2) := by
    have equality_source := row_54
    have equality_step_1 := row_48
    cases equality_step_1
    exact equality_source
  have rule_row_89 := fta_source_043 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_031 external_peano_externals_36_009 external_peano_externals_36_010 external_peano_externals_36_007 external_peano_externals_36_029 external_peano_externals_36_017 external_peano_externals_36_021 external_peano_externals_36_015 external_peano_externals_36_022 external_peano_externals_36_033 external_peano_externals_36_008
  -- chapter_5_line_89: GL tag implication.
  have row_89 : (v5 = v2) := by
    apply rule_row_89
    exact row_53
    exact row_50
  have rule_row_49 := fta_source_043 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_031 external_peano_externals_36_009 external_peano_externals_36_010 external_peano_externals_36_007 external_peano_externals_36_029 external_peano_externals_36_017 external_peano_externals_36_021 external_peano_externals_36_015 external_peano_externals_36_022 external_peano_externals_36_033 external_peano_externals_36_008
  -- chapter_5_line_49: GL tag implication.
  have row_49 : (v2 = v5) := by
    apply rule_row_49
    exact row_50
    exact row_53
  -- chapter_5_line_37: GL tag equality1.
  have row_37 : (gl_preorder N mul v3 v2) := by
    have equality_source := row_38
    have equality_step_1 := row_48
    cases equality_step_1
    exact equality_source
  have rule_row_92 := fta_source_043 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_031 external_peano_externals_36_009 external_peano_externals_36_010 external_peano_externals_36_007 external_peano_externals_36_029 external_peano_externals_36_017 external_peano_externals_36_021 external_peano_externals_36_015 external_peano_externals_36_022 external_peano_externals_36_033 external_peano_externals_36_008
  -- chapter_5_line_92: GL tag implication.
  have row_92 : (v2 = v3) := by
    apply rule_row_92
    exact row_20
    exact row_37
  have rule_row_16 := fta_source_043 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_031 external_peano_externals_36_009 external_peano_externals_36_010 external_peano_externals_36_007 external_peano_externals_36_029 external_peano_externals_36_017 external_peano_externals_36_021 external_peano_externals_36_015 external_peano_externals_36_022 external_peano_externals_36_033 external_peano_externals_36_008
  -- chapter_5_line_16: GL tag implication.
  have row_16 : (v3 = v2) := by
    apply rule_row_16
    exact row_37
    exact row_20
  -- chapter_5_line_12: GL tag equality1.
  have row_12 : (¬ (v2 = v5)) := by
    have equality_source := row_13
    have equality_step_1 := row_16
    cases equality_step_1
    exact equality_source
  -- chapter_5_line_11: GL tag equality1.
  have row_11 : (¬ (v5 = v5)) := by
    have equality_source := row_12
    have equality_step_1 := row_49
    cases equality_step_1
    exact equality_source
  -- chapter_5_line_10: GL tag equality1.
  have row_10 : (¬ (zero = v5)) := by
    have equality_source := row_11
    have equality_step_1 := row_58
    cases equality_step_1
    exact equality_source
  -- chapter_5_line_9: GL tag equality1.
  have row_9 : (¬ (v7 = v5)) := by
    have equality_source := row_10
    have equality_step_1 := row_62
    cases equality_step_1
    exact equality_source
  -- chapter_5_line_8: GL tag equality1.
  have row_8 : (¬ (v6 = v5)) := by
    have equality_source := row_9
    have equality_step_1 := row_74
    cases equality_step_1
    exact equality_source
  -- chapter_5_line_7: GL tag equality1.
  have row_7 : (¬ (v6 = v2)) := by
    have equality_source := row_8
    have equality_step_1 := row_89
    cases equality_step_1
    exact equality_source
  -- chapter_5_line_6: GL tag equality1.
  have row_6 : (¬ (v2 = v2)) := by
    have equality_source := row_7
    have equality_step_1 := row_90
    cases equality_step_1
    exact equality_source
  -- chapter_5_line_5: GL tag equality1.
  have row_5 : (¬ (v3 = v2)) := by
    have equality_source := row_6
    have equality_step_1 := row_92
    cases equality_step_1
    exact equality_source
  -- chapter_5_line_4: GL tag equality1.
  have row_4 : (¬ (zero = v2)) := by
    have equality_source := row_5
    have equality_step_1 := row_93
    cases equality_step_1
    exact equality_source
  -- chapter_5_line_3: GL tag symmetry of inequality.
  have row_3 : (¬ (v2 = zero)) := by
    exact fun equality => row_4 (Eq.symm equality)
  -- chapter_5_line_1: GL tag vacuous truth.
  have row_1 : (gl_strictOrder N add v1 v4) := by
    exact False.elim (row_3 row_2)
  exact row_1

private theorem fta_source_004_check_induction_condition
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_010 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_5 x_7 x_8 x_9) → (x_5 x_8 x_7 x_9))))))
    (external_peano_externals_36_035 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (gl_existence11 N one succ))))
    (external_peano_externals_36_015 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_3 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_006 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (x_4 x_8 x_7 x_9))))))
    (external_peano_externals_36_029 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((gl_preorder x_1 x_4 x_7 x_8) → ((gl_preorder x_1 x_4 x_8 x_7) → (x_7 = x_8)))))))
    (external_gauss_externals_24_018 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α) (x_7 : α) (x_8 : GLBinaryRelation α), ((gl_AnchorGauss x_1 x_2 x_3 x_4 x_5 x_6 x_7 x_8) → (∀ (x_9 : α) (x_10 : α), ((x_3 x_9 x_10) → (gl_preorder x_1 x_4 x_9 x_10))))))
    (external_peano_externals_36_003 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (∀ (x_10 : α), ((x_3 x_8 x_10) → (∀ (x_11 : α), ((x_3 x_11 x_7) → (x_4 x_10 x_11 x_9))))))))))
    (external_peano_externals_36_002 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (∀ (w4 : α) (w5 : α), ((add w4 w2 w5) → (∀ (w6 : α), ((add w3 w4 w6) → (add w5 w1 w6))))))))))
    (external_peano_externals_36_005 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → ((x_8 = x_9) → (x_2 = x_7)))))))
    (external_peano_externals_36_022 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_6 x_8) → (x_4 x_6 x_7 x_8))))))
    (external_gauss_externals_24_021 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α) (x_7 : α) (x_8 : GLBinaryRelation α), ((gl_AnchorGauss x_1 x_2 x_3 x_4 x_5 x_6 x_7 x_8) → (∀ (x_9 : α) (x_10 : α), ((gl_preorder x_1 x_4 x_9 x_10) → (∀ (x_11 : α), ((gl_preorder x_1 x_4 x_10 x_11) → (gl_preorder x_1 x_4 x_9 x_11))))))))
    (external_peano_externals_36_017 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_6 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_021 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_6 x_8) → (x_3 x_7 x_8))))))
    (external_peano_externals_36_034 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (∀ (w1 : α), ((N w1) → (gl_or2 w1 zero N succ))))))
    (external_peano_externals_36_008 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (∀ (w4 : α) (w5 : α), ((mul w4 w1 w5) → (∀ (w6 : α), ((mul w4 w2 w6) → (∀ (w7 : α), ((mul w4 w3 w7) → (add w5 w6 w7))))))))))))
    (external_peano_externals_36_004 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (∀ (x_10 : α), ((x_4 x_7 x_10 x_9) → (x_8 = x_10))))))))
    (previous : α)
    (v1 : α)
    (v2 : α)
    (v3 : α)
    (v4 : α)
    (v5 : α)
    (assumption_43 : (succ previous v2))
    (assumption_15 : (gl_strictOrder N add v3 v5))
    (assumption_12 : (mul v1 v2 v3))
    (assumption_11 : (mul v4 v2 v5))
    : (gl_strictOrder N add v1 v4) := by
  -- chapter_6_line_73: GL tag theorem.
  have row_73 := fta_source_046 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_029
  -- chapter_6_line_49: GL tag theorem.
  have row_49 := fta_source_002 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_6_line_47: GL tag expansion for integration.
  have row_47 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_6_line_46: GL tag reformulation for integration and.
  have row_46 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_6_line_43: GL tag recursion.
  have row_43 : (succ previous v2) := by
    exact assumption_43
  -- chapter_6_line_22: GL tag theorem.
  have row_22 := fta_source_016 N zero succ add mul one two identity anchor relationalInduction external_gauss_externals_24_018 external_peano_externals_36_006 external_peano_externals_36_003 external_peano_externals_36_002 external_peano_externals_36_029 external_peano_externals_36_035 external_peano_externals_36_005
  -- chapter_6_line_17: GL tag theorem.
  have row_17 := fta_source_059 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_015 external_peano_externals_36_022
  -- chapter_6_line_15: GL tag task formulation.
  have row_15 : (gl_strictOrder N add v3 v5) := by
    exact assumption_15
  -- chapter_6_line_14: GL tag expansion.
  have row_14 : ((gl_preorder N add v3 v5) ∧ (¬ (v3 = v5))) := by
    simpa only [gl_strictOrder] using row_15
  -- chapter_6_line_75: GL tag disintegration.
  have row_75 : (¬ (v3 = v5)) := by
    exact row_14.2
  -- chapter_6_line_74: GL tag symmetry of inequality.
  have row_74 : (¬ (v5 = v3)) := by
    exact fun equality => row_75 (Eq.symm equality)
  -- chapter_6_line_13: GL tag disintegration.
  have row_13 : (gl_preorder N add v3 v5) := by
    exact row_14.1
  -- chapter_6_line_113: GL tag expansion.
  have row_113 : (¬ (∀ (v15 : α), ((N v15) → (¬ (add v3 v15 v5))))) := by
    simpa only [gl_preorder] using row_13
  have exists_row_113 : ∃ (v15 : α), ((N v15) ∧ (add v3 v15 v5)) := existsAndOfNotForallImpNot row_113
  obtain ⟨v15, witness_row_113⟩ := exists_row_113
  -- chapter_6_line_112: GL tag disintegration.
  have row_112 : (add v3 v15 v5) := by
    exact witness_row_113.2
  -- chapter_6_line_12: GL tag task formulation.
  have row_12 : (mul v1 v2 v3) := by
    exact assumption_12
  -- chapter_6_line_11: GL tag task formulation.
  have row_11 : (mul v4 v2 v5) := by
    exact assumption_11
  -- chapter_6_line_10: GL tag theorem.
  have row_10 := fta_source_067 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_035 external_gauss_externals_24_018 external_gauss_externals_24_021 external_peano_externals_36_010 external_peano_externals_36_017 external_peano_externals_36_021 external_peano_externals_36_015 external_peano_externals_36_002 external_peano_externals_36_006 external_peano_externals_36_034 external_peano_externals_36_003 external_peano_externals_36_008 external_peano_externals_36_004 external_peano_externals_36_029 external_peano_externals_36_005 external_peano_externals_36_022
  -- chapter_6_line_8: GL tag task formulation.
  have row_8 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_6_line_72: GL tag implication.
  have row_72 : (¬ (gl_strictOrder N add v5 v3)) := by
    apply row_73
    exact row_15
  -- chapter_6_line_71: GL tag expansion.
  have row_71 : (¬ ((gl_preorder N add v5 v3) ∧ (¬ (v5 = v3)))) := by
    simpa only [gl_strictOrder] using row_72
  -- chapter_6_line_70: GL tag disintegration.
  have row_70 : ((¬ (v5 = v3)) → (¬ (gl_preorder N add v5 v3))) := by
    classical
    intro projection_premise
    intro projection_counterexample
    exact row_71 ⟨projection_counterexample, projection_premise⟩
  -- chapter_6_line_69: GL tag implication.
  have row_69 : (¬ (gl_preorder N add v5 v3)) := by
    apply row_70
    exact row_74
  -- chapter_6_line_68: GL tag expansion.
  have row_68 : (gl_implication24 N v5 v3 add) := by
    simp only [gl_implication24]
    intro existence_witness_1 compact_premise compact_negated
    have unfolded_row_69 := row_69
    simp only [gl_preorder] at unfolded_row_69
    apply unfolded_row_69
    intro positive_row_69
    exact positive_row_69 existence_witness_1 compact_premise compact_negated
  -- chapter_6_line_67: GL tag expansion.
  have row_67 : (∀ (w1 : α), ((N w1) → (¬ (add v5 w1 v3)))) := by
    simpa only [gl_implication24] using row_68
  -- chapter_6_line_28: GL tag expansion.
  have row_28 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_8
  -- chapter_6_line_59: GL tag disintegration.
  have row_59 : (succ one two) := by
    exact row_28.1.2
  -- chapter_6_line_29: GL tag disintegration.
  have row_29 : (succ zero one) := by
    exact row_28.1.1.2
  -- chapter_6_line_27: GL tag disintegration.
  have row_27 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_28.1.1.1
  -- chapter_6_line_45: GL tag implication.
  have row_45 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_46
    exact row_27
    exact row_29
  have rule_row_84 := external_peano_externals_36_010 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_6_line_84: GL tag implication.
  have row_84 : (mul v2 v4 v5) := by
    apply rule_row_84
    exact row_11
  have rule_row_82 := external_peano_externals_36_010 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_6_line_82: GL tag implication.
  have row_82 : (mul v2 v1 v3) := by
    apply rule_row_82
    exact row_12
  have rule_row_57 := external_peano_externals_36_035 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_6_line_57: GL tag implication.
  have row_57 : (gl_existence11 N one succ) := by
    apply rule_row_57
  -- chapter_6_line_56: GL tag expansion.
  have row_56 : (¬ (∀ (v6 : α), ((N v6) → (¬ (succ v6 one))))) := by
    simpa only [gl_existence11] using row_57
  have exists_row_56 : ∃ (v6 : α), ((N v6) ∧ (succ v6 one)) := existsAndOfNotForallImpNot row_56
  obtain ⟨v6, witness_row_56⟩ := exists_row_56
  -- chapter_6_line_55: GL tag disintegration.
  have row_55 : (succ v6 one) := by
    exact witness_row_56.2
  have rule_row_44 := external_peano_externals_36_015 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_6_line_44: GL tag implication.
  have row_44 : (add previous one v2) := by
    apply rule_row_44
    exact row_43
  -- chapter_6_line_26: GL tag expansion.
  have row_26 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_27
  -- chapter_6_line_98: GL tag disintegration.
  have row_98 : (gl_implication21 N succ mul add) := by
    exact row_26.2
  -- chapter_6_line_97: GL tag expansion.
  have row_97 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((mul w3 w1 w4) → (∀ (w5 : α), ((mul w3 w2 w5) → (add w4 w3 w5))))))))) := by
    simpa only [gl_implication21] using row_98
  -- chapter_6_line_91: GL tag disintegration.
  have row_91 : (gl_implication16 N zero add) := by
    exact row_26.1.1.1.1.1.1.2
  -- chapter_6_line_90: GL tag expansion.
  have row_90 : (∀ (w1 : α) (w2 : α), ((w1 = w2) → ((N w1) → ((N w2) → (add w1 zero w2))))) := by
    simpa only [gl_implication16] using row_91
  -- chapter_6_line_80: GL tag disintegration.
  have row_80 : (gl_fXYZ mul N N N) := by
    exact row_26.1.1.1.2
  -- chapter_6_line_79: GL tag expansion.
  have row_79 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_80
  -- chapter_6_line_103: GL tag disintegration.
  have row_103 : (gl_implication13 N N N mul) := by
    exact row_79.1.2
  -- chapter_6_line_102: GL tag expansion.
  have row_102 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication13] using row_103
  -- chapter_6_line_87: GL tag disintegration.
  have row_87 : (gl_implication8 mul N) := by
    exact row_79.1.1.1.1
  -- chapter_6_line_86: GL tag expansion.
  have row_86 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_87
  -- chapter_6_line_104: GL tag implication.
  have row_104 : (N v1) := by
    apply row_86
    exact row_12
  -- chapter_6_line_85: GL tag implication.
  have row_85 : (N v4) := by
    apply row_86
    exact row_11
  -- chapter_6_line_78: GL tag disintegration.
  have row_78 : (gl_implication14 N N mul) := by
    exact row_79.2
  -- chapter_6_line_77: GL tag expansion.
  have row_77 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (∀ (w3 : α), ((mul w1 w2 w3) → (∀ (w4 : α), ((mul w1 w2 w4) → (w3 = w4))))))))) := by
    simpa only [gl_implication14] using row_78
  -- chapter_6_line_54: GL tag disintegration.
  have row_54 : (gl_implication7 N succ) := by
    exact row_26.1.1.1.1.1.1.1.1.1.2
  -- chapter_6_line_53: GL tag expansion.
  have row_53 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w2 w1) → (∀ (w3 : α), ((succ w3 w1) → (w2 = w3))))))) := by
    simpa only [gl_implication7] using row_54
  -- chapter_6_line_42: GL tag disintegration.
  have row_42 : (gl_fXY succ N N) := by
    exact row_26.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_6_line_41: GL tag expansion.
  have row_41 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_42
  -- chapter_6_line_63: GL tag disintegration.
  have row_63 : (gl_implication1 succ N) := by
    exact row_41.1.1.2
  -- chapter_6_line_62: GL tag expansion.
  have row_62 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using row_63
  -- chapter_6_line_61: GL tag implication.
  have row_61 : (N v2) := by
    apply row_62
    exact row_43
  -- chapter_6_line_40: GL tag disintegration.
  have row_40 : (gl_implication0 succ N) := by
    exact row_41.1.1.1
  -- chapter_6_line_39: GL tag expansion.
  have row_39 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_40
  -- chapter_6_line_58: GL tag implication.
  have row_58 : (N one) := by
    apply row_39
    exact row_59
  -- chapter_6_line_60: GL tag implication.
  have row_60 : (v6 = zero) := by
    apply row_53
    exact row_58
    exact row_55
    exact row_29
  -- chapter_6_line_52: GL tag implication.
  have row_52 : (zero = v6) := by
    apply row_53
    exact row_58
    exact row_29
    exact row_55
  -- chapter_6_line_38: GL tag implication.
  have row_38 : (N previous) := by
    apply row_39
    exact row_43
  -- chapter_6_line_101: GL tag implication.
  have row_101 : (gl_existence1 N v1 previous mul) := by
    apply row_102
    exact row_104
    exact row_38
  -- chapter_6_line_100: GL tag expansion.
  have row_100 : (¬ (∀ (v14 : α), ((N v14) → (¬ (mul v1 previous v14))))) := by
    simpa only [gl_existence1] using row_101
  have exists_row_100 : ∃ (v14 : α), ((N v14) ∧ (mul v1 previous v14)) := existsAndOfNotForallImpNot row_100
  obtain ⟨v14, witness_row_100⟩ := exists_row_100
  -- chapter_6_line_108: GL tag disintegration.
  have row_108 : (N v14) := by
    exact witness_row_100.1
  -- chapter_6_line_99: GL tag disintegration.
  have row_99 : (mul v1 previous v14) := by
    exact witness_row_100.2
  -- chapter_6_line_96: GL tag implication.
  have row_96 : (add v14 v1 v3) := by
    apply row_97
    exact row_38
    exact row_43
    exact row_99
    exact row_12
  have rule_row_95 := external_peano_externals_36_006 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_6_line_95: GL tag implication.
  have row_95 : (add v1 v14 v3) := by
    apply rule_row_95
    exact row_96
  -- chapter_6_line_37: GL tag disintegration.
  have row_37 : (N zero) := by
    exact row_26.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_6_line_66: GL tag implication.
  have row_66 : (¬ (add v5 zero v3)) := by
    apply row_67
    exact row_37
  -- chapter_6_line_36: GL tag disintegration.
  have row_36 : (gl_fXYZ add N N N) := by
    exact row_26.1.1.1.1.1.1.1.1.2
  -- chapter_6_line_35: GL tag expansion.
  have row_35 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_36
  -- chapter_6_line_111: GL tag disintegration.
  have row_111 : (gl_implication8 add N) := by
    exact row_35.1.1.1.1
  -- chapter_6_line_110: GL tag expansion.
  have row_110 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_111
  -- chapter_6_line_109: GL tag implication.
  have row_109 : (N v3) := by
    apply row_110
    exact row_112
  -- chapter_6_line_94: GL tag disintegration.
  have row_94 : (gl_implication14 N N add) := by
    exact row_35.2
  -- chapter_6_line_93: GL tag expansion.
  have row_93 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (∀ (w3 : α), ((add w1 w2 w3) → (∀ (w4 : α), ((add w1 w2 w4) → (w3 = w4))))))))) := by
    simpa only [gl_implication14] using row_94
  -- chapter_6_line_34: GL tag disintegration.
  have row_34 : (gl_implication13 N N N add) := by
    exact row_35.1.2
  -- chapter_6_line_33: GL tag expansion.
  have row_33 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication13] using row_34
  -- chapter_6_line_107: GL tag implication.
  have row_107 : (gl_existence1 N v1 v14 add) := by
    apply row_33
    exact row_104
    exact row_108
  -- chapter_6_line_106: GL tag expansion.
  have row_106 : (¬ (∀ (v13 : α), ((N v13) → (¬ (add v1 v14 v13))))) := by
    simpa only [gl_existence1] using row_107
  have exists_row_106 : ∃ (v13 : α), ((N v13) ∧ (add v1 v14 v13)) := existsAndOfNotForallImpNot row_106
  obtain ⟨v13, witness_row_106⟩ := exists_row_106
  -- chapter_6_line_114: GL tag disintegration.
  have row_114 : (N v13) := by
    exact witness_row_106.1
  -- chapter_6_line_105: GL tag disintegration.
  have row_105 : (add v1 v14 v13) := by
    exact witness_row_106.2
  -- chapter_6_line_115: GL tag implication.
  have row_115 : (v13 = v3) := by
    apply row_93
    exact row_104
    exact row_108
    exact row_105
    exact row_95
  -- chapter_6_line_92: GL tag implication.
  have row_92 : (v3 = v13) := by
    apply row_93
    exact row_104
    exact row_108
    exact row_95
    exact row_105
  -- chapter_6_line_89: GL tag implication.
  have row_89 : (add v3 zero v13) := by
    apply row_90
    exact row_92
    exact row_109
    exact row_114
  -- chapter_6_line_88: GL tag equality1.
  have row_88 : (add v3 zero v3) := by
    have equality_source := row_89
    have equality_step_1 := row_115
    cases equality_step_1
    exact equality_source
  -- chapter_6_line_32: GL tag implication.
  have row_32 : (gl_existence1 N previous zero add) := by
    apply row_33
    exact row_38
    exact row_37
  -- chapter_6_line_31: GL tag expansion.
  have row_31 : (¬ (∀ (v7 : α), ((N v7) → (¬ (add previous zero v7))))) := by
    simpa only [gl_existence1] using row_32
  have exists_row_31 : ∃ (v7 : α), ((N v7) ∧ (add previous zero v7)) := existsAndOfNotForallImpNot row_31
  obtain ⟨v7, witness_row_31⟩ := exists_row_31
  -- chapter_6_line_30: GL tag disintegration.
  have row_30 : (add previous zero v7) := by
    exact witness_row_31.2
  -- chapter_6_line_51: GL tag equality1.
  have row_51 : (add previous v6 v7) := by
    have equality_source := row_30
    have equality_step_1 := row_52
    cases equality_step_1
    exact equality_source
  have rule_row_50 := external_peano_externals_36_006 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_6_line_50: GL tag implication.
  have row_50 : (add v6 previous v7) := by
    apply rule_row_50
    exact row_51
  -- chapter_6_line_48: GL tag implication.
  have row_48 : (gl_preorder N add v6 v7) := by
    apply row_49
    exact row_50
  -- chapter_6_line_25: GL tag disintegration.
  have row_25 : (gl_implication17 N succ add) := by
    exact row_26.1.1.1.1.1.2
  -- chapter_6_line_24: GL tag expansion.
  have row_24 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication17] using row_25
  -- chapter_6_line_23: GL tag implication.
  have row_23 : (succ v7 v2) := by
    apply row_24
    exact row_37
    exact row_29
    exact row_30
    exact row_44
  -- chapter_6_line_21: GL tag implication.
  have row_21 : (gl_strictOrder N add v6 v2) := by
    apply row_22
    exact row_23
    exact row_48
  -- chapter_6_line_20: GL tag equality1.
  have row_20 : (gl_strictOrder N add zero v2) := by
    have equality_source := row_21
    have equality_step_1 := row_60
    cases equality_step_1
    exact equality_source
  -- chapter_6_line_19: GL tag expansion.
  have row_19 : ((gl_preorder N add zero v2) ∧ (¬ (zero = v2))) := by
    simpa only [gl_strictOrder] using row_20
  -- chapter_6_line_18: GL tag disintegration.
  have row_18 : (¬ (zero = v2)) := by
    exact row_19.2
  -- chapter_6_line_16: GL tag implication.
  have row_16 : (gl_preorder N add one v2) := by
    apply row_17
    exact row_61
    exact row_18
  -- chapter_6_line_9: GL tag implication.
  have row_9 : (gl_preorder N add v1 v4) := by
    apply row_10
    exact row_16
    exact row_12
    exact row_11
    exact row_13
  -- chapter_6_line_7: GL tag compilation.
  have row_7 : (gl_implication56 (α := α)) := by
    simp only [gl_implication56]
    intro compiled_N compiled_zero compiled_succ compiled_add compiled_mul compiled_one compiled_two compiled_identity
    intro compiled_anchor
    exact fta_source_034 compiled_N compiled_zero compiled_succ compiled_add compiled_mul compiled_one compiled_two compiled_identity compiled_anchor relationalInduction
  -- chapter_6_line_6: GL tag expansion.
  have row_6 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α) (two : α) (identity : GLBinaryRelation α), ((gl_AnchorFTA N zero succ add mul one two identity) → (∀ (v1 : α) (v2 : α), ((gl_preorder N add v1 v2) → (gl_or0 v1 v2 N add))))) := by
    simpa only [gl_implication56] using row_7
  have rule_row_5 := fta_source_034 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_6_line_5: GL tag implication.
  have row_5 : (gl_or0 v1 v4 N add) := by
    apply rule_row_5
    exact row_9
  -- chapter_6_line_83: GL tag or disintegration.
  have row_83 : ((v1 = v4) → (v1 = v4)) := by
    intro scope_premise_1
    exact scope_premise_1
  -- chapter_6_line_81: GL tag equality1.
  have row_81 : ((v1 = v4) → (mul v2 v4 v3)) := by
    intro scope_premise_1
    have scoped_fact_2 := row_83 scope_premise_1
    have equality_source := row_82
    have equality_step_1 := scoped_fact_2
    cases equality_step_1
    exact equality_source
  -- chapter_6_line_76: GL tag implication.
  have row_76 : ((v1 = v4) → (v5 = v3)) := by
    intro scope_premise_1
    have scoped_fact_5 := row_81 scope_premise_1
    apply row_77
    exact row_61
    exact row_85
    exact row_84
    exact scoped_fact_5
  -- chapter_6_line_65: GL tag equality1.
  have row_65 : ((v1 = v4) → (¬ (add v3 zero v3))) := by
    intro scope_premise_1
    have scoped_fact_2 := row_76 scope_premise_1
    have equality_source := row_66
    have equality_step_1 := scoped_fact_2
    cases equality_step_1
    exact equality_source
  -- chapter_6_line_64: GL tag contradiction.
  have row_64 : (¬ (v1 = v4)) := by
    intro contradiction_assumption
    have scoped_contradiction := row_65 contradiction_assumption
    exact scoped_contradiction row_88
  -- chapter_6_line_4: GL tag expansion.
  have row_4 : (¬ ((¬ (v1 = v4)) ∧ (¬ (gl_strictOrder N add v1 v4)))) := by
    simpa only [gl_or0, GLExport.orIffNotAndNot] using row_5
  -- chapter_6_line_3: GL tag disintegration.
  have row_3 : (gl_implication50 v1 v4 N add) := by
    simp only [gl_implication50]
    classical
    intro projection_premise
    apply Classical.byContradiction
    intro projection_counterexample
    exact row_4 ⟨projection_premise, projection_counterexample⟩
  -- chapter_6_line_2: GL tag expansion.
  have row_2 : ((¬ (v1 = v4)) → (gl_strictOrder N add v1 v4)) := by
    simpa only [gl_implication50] using row_3
  -- chapter_6_line_1: GL tag implication.
  have row_1 : (gl_strictOrder N add v1 v4) := by
    apply row_2
    exact row_64
  exact row_1

theorem fta_source_004
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_035 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (gl_existence11 N one succ))))
    (external_peano_externals_36_010 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_5 x_7 x_8 x_9) → (x_5 x_8 x_7 x_9))))))
    (external_peano_externals_36_009 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (∀ (w4 : α) (w5 : α), ((mul w4 w2 w5) → (∀ (w6 : α), ((mul w3 w4 w6) → (mul w5 w1 w6))))))))))
    (external_peano_externals_36_031 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α), ((x_1 x_7) → (∀ (x_8 : α), ((x_5 x_2 x_7 x_8) → (x_2 = x_8))))))))
    (external_peano_externals_36_033 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α), ((x_1 x_7) → (∀ (x_8 : α), ((x_5 x_6 x_8 x_7) → (x_7 = x_8))))))))
    (external_peano_externals_36_018 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_5 x_6 x_7 x_8) → (x_5 x_7 x_6 x_8))))))
    (external_peano_externals_36_007 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_5 x_7 x_8 x_9) → (∀ (x_10 : α) (x_11 : α) (x_12 : α), ((x_5 x_10 x_11 x_12) → ((x_5 x_8 x_7 x_11) → (x_5 x_10 x_9 x_12)))))))))
    (external_peano_externals_36_029 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((gl_preorder x_1 x_4 x_7 x_8) → ((gl_preorder x_1 x_4 x_8 x_7) → (x_7 = x_8)))))))
    (external_peano_externals_36_017 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_6 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_021 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_6 x_8) → (x_3 x_7 x_8))))))
    (external_peano_externals_36_015 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_3 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_022 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_6 x_8) → (x_4 x_6 x_7 x_8))))))
    (external_peano_externals_36_008 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (∀ (w4 : α) (w5 : α), ((mul w4 w1 w5) → (∀ (w6 : α), ((mul w4 w2 w6) → (∀ (w7 : α), ((mul w4 w3 w7) → (add w5 w6 w7))))))))))))
    (external_peano_externals_36_006 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (x_4 x_8 x_7 x_9))))))
    (external_gauss_externals_24_018 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α) (x_7 : α) (x_8 : GLBinaryRelation α), ((gl_AnchorGauss x_1 x_2 x_3 x_4 x_5 x_6 x_7 x_8) → (∀ (x_9 : α) (x_10 : α), ((x_3 x_9 x_10) → (gl_preorder x_1 x_4 x_9 x_10))))))
    (external_peano_externals_36_003 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (∀ (x_10 : α), ((x_3 x_8 x_10) → (∀ (x_11 : α), ((x_3 x_11 x_7) → (x_4 x_10 x_11 x_9))))))))))
    (external_peano_externals_36_002 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (∀ (w4 : α) (w5 : α), ((add w4 w2 w5) → (∀ (w6 : α), ((add w3 w4 w6) → (add w5 w1 w6))))))))))
    (external_peano_externals_36_005 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → ((x_8 = x_9) → (x_2 = x_7)))))))
    (external_gauss_externals_24_021 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α) (x_7 : α) (x_8 : GLBinaryRelation α), ((gl_AnchorGauss x_1 x_2 x_3 x_4 x_5 x_6 x_7 x_8) → (∀ (x_9 : α) (x_10 : α), ((gl_preorder x_1 x_4 x_9 x_10) → (∀ (x_11 : α), ((gl_preorder x_1 x_4 x_10 x_11) → (gl_preorder x_1 x_4 x_9 x_11))))))))
    (external_peano_externals_36_034 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (∀ (w1 : α), ((N w1) → (gl_or2 w1 zero N succ))))))
    (external_peano_externals_36_004 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (∀ (x_10 : α), ((x_4 x_7 x_10 x_9) → (x_8 = x_10))))))))
    : (∀ (v1 : α) (v2 : α) (v3 : α), ((mul v1 v2 v3) → (∀ (v4 : α) (v5 : α), ((mul v4 v2 v5) → ((gl_strictOrder N add v3 v5) → (gl_strictOrder N add v1 v4)))))) := by
  intro v1
  intro v2
  intro v3
  intro premise_1
  intro v4
  intro v5
  intro premise_2
  intro premise_3
  have inductionMember : N v2 := by
    have typingRule := fta_source_004_induction_typing N zero succ add mul one two identity anchor relationalInduction
    exact typingRule v2 v4 v5 premise_2
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorFTA, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v1 : α) (v3 : α), ((mul v1 zero v3) → (∀ (v4 : α) (v5 : α), ((mul v4 zero v5) → ((gl_strictOrder N add v3 v5) → (gl_strictOrder N add v1 v4)))))) := by
    intro v1
    intro v3
    intro base_premise_1
    intro v4
    intro v5
    intro base_premise_2
    intro base_premise_3
    have zeroRule := fta_source_004_check_zero N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_035 external_peano_externals_36_010 external_peano_externals_36_009 external_peano_externals_36_031 external_peano_externals_36_033 external_peano_externals_36_018 external_peano_externals_36_007 external_peano_externals_36_029 external_peano_externals_36_017 external_peano_externals_36_021 external_peano_externals_36_015 external_peano_externals_36_022 external_peano_externals_36_008
    exact zeroRule v1 zero v3 v4 v5 base_premise_2 base_premise_1 base_premise_3 rfl
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v1 : α) (v3 : α), ((mul v1 induction_n v3) → (∀ (v4 : α) (v5 : α), ((mul v4 induction_n v5) → ((gl_strictOrder N add v3 v5) → (gl_strictOrder N add v1 v4)))))) → ∀ induction_m, succ induction_n induction_m → (∀ (v1 : α) (v3 : α), ((mul v1 induction_m v3) → (∀ (v4 : α) (v5 : α), ((mul v4 induction_m v5) → ((gl_strictOrder N add v3 v5) → (gl_strictOrder N add v1 v4)))))) := by
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
    have stepRule := fta_source_004_check_induction_condition N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_010 external_peano_externals_36_035 external_peano_externals_36_015 external_peano_externals_36_006 external_peano_externals_36_029 external_gauss_externals_24_018 external_peano_externals_36_003 external_peano_externals_36_002 external_peano_externals_36_005 external_peano_externals_36_022 external_gauss_externals_24_021 external_peano_externals_36_017 external_peano_externals_36_021 external_peano_externals_36_034 external_peano_externals_36_008 external_peano_externals_36_004
    exact stepRule induction_n v1 induction_m v3 v4 v5 step_induction_assumption_1 step_premise_3 step_premise_1 step_premise_2
  have inductionProperty : (∀ (v1 : α) (v3 : α), ((mul v1 v2 v3) → (∀ (v4 : α) (v5 : α), ((mul v4 v2 v5) → ((gl_strictOrder N add v3 v5) → (gl_strictOrder N add v1 v4)))))) := by
    exact relationalInduction N zero succ add mul one two identity anchor
      (fun induction_value => (∀ (v1 : α) (v3 : α), ((mul v1 induction_value v3) → (∀ (v4 : α) (v5 : α), ((mul v4 induction_value v5) → ((gl_strictOrder N add v3 v5) → (gl_strictOrder N add v1 v4)))))))
      v2
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v1 v3 premise_1 v4 v5 premise_2 premise_3

theorem fta_source_006
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_010 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_5 x_7 x_8 x_9) → (x_5 x_8 x_7 x_9))))))
    (external_peano_externals_36_033 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α), ((x_1 x_7) → (∀ (x_8 : α), ((x_5 x_6 x_8 x_7) → (x_7 = x_8))))))))
    (external_peano_externals_36_024 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_8 x_2) → ((x_1 x_8) → (x_2 = x_7)))))))
    (external_peano_externals_36_015 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_3 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_022 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_6 x_8) → (x_4 x_6 x_7 x_8))))))
    (external_peano_externals_36_017 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_6 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_021 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_6 x_8) → (x_3 x_7 x_8))))))
    (external_peano_externals_36_034 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (∀ (w1 : α), ((N w1) → (gl_or2 w1 zero N succ))))))
    (external_peano_externals_36_035 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (gl_existence11 N one succ))))
    (external_gauss_externals_24_018 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α) (x_7 : α) (x_8 : GLBinaryRelation α), ((gl_AnchorGauss x_1 x_2 x_3 x_4 x_5 x_6 x_7 x_8) → (∀ (x_9 : α) (x_10 : α), ((x_3 x_9 x_10) → (gl_preorder x_1 x_4 x_9 x_10))))))
    (external_peano_externals_36_006 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (x_4 x_8 x_7 x_9))))))
    (external_peano_externals_36_003 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (∀ (x_10 : α), ((x_3 x_8 x_10) → (∀ (x_11 : α), ((x_3 x_11 x_7) → (x_4 x_10 x_11 x_9))))))))))
    (external_peano_externals_36_002 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (∀ (w4 : α) (w5 : α), ((add w4 w2 w5) → (∀ (w6 : α), ((add w3 w4 w6) → (add w5 w1 w6))))))))))
    (external_peano_externals_36_029 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((gl_preorder x_1 x_4 x_7 x_8) → ((gl_preorder x_1 x_4 x_8 x_7) → (x_7 = x_8)))))))
    (external_peano_externals_36_005 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → ((x_8 = x_9) → (x_2 = x_7)))))))
    : (∀ (v1 : α) (v2 : α) (v3 : α), ((mul v1 v2 v3) → ((gl_preorder N add two v3) → ((N v1) → ((¬ (gl_preorder N add two v2)) → (gl_preorder N add two v1)))))) := by
  intro v1
  intro v2
  intro v3
  intro premise_1
  intro premise_2
  intro premise_3
  intro premise_4
  -- chapter_8_line_37: GL tag task formulation.
  have row_37 : (¬ (gl_preorder N add two v2)) := by
    exact premise_4
  -- chapter_8_line_36: GL tag task formulation.
  have row_36 : (N v1) := by
    exact premise_3
  -- chapter_8_line_35: GL tag expansion for integration.
  have row_35 : ((gl_preorder N mul v2 v3) ↔ (¬ (∀ (v9 : α), ((N v9) → (¬ (mul v2 v9 v3)))))) := by
    exact Iff.rfl
  -- chapter_8_line_34: GL tag reformulation for integration >[bound].
  have row_34 : (∀ (v8 : α), ((N v8) → ((mul v2 v8 v3) → (gl_preorder N mul v2 v3)))) := by
    intro v8
    intro integration_premise_1
    intro integration_premise_2
    apply (row_35).2
    intro universal_counterexample
    exact universal_counterexample v8 integration_premise_1 integration_premise_2
  -- chapter_8_line_24: GL tag theorem.
  have row_24 := fta_source_072 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_024
  -- chapter_8_line_22: GL tag theorem.
  have row_22 := fta_source_059 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_015 external_peano_externals_36_022
  -- chapter_8_line_20: GL tag theorem.
  have row_20 := fta_source_041 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_010 external_peano_externals_36_017 external_peano_externals_36_021 external_peano_externals_36_015 external_peano_externals_36_022
  -- chapter_8_line_18: GL tag theorem.
  have row_18 := fta_source_066 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_017 external_peano_externals_36_021
  -- chapter_8_line_16: GL tag theorem.
  have row_16 := fta_source_058 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_034 external_peano_externals_36_022 external_peano_externals_36_035 external_peano_externals_36_015 external_gauss_externals_24_018 external_peano_externals_36_006 external_peano_externals_36_003 external_peano_externals_36_002 external_peano_externals_36_029 external_peano_externals_36_005
  -- chapter_8_line_13: GL tag task formulation.
  have row_13 : (mul v1 v2 v3) := by
    exact premise_1
  -- chapter_8_line_9: GL tag task formulation.
  have row_9 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_8_line_8: GL tag expansion.
  have row_8 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_9
  -- chapter_8_line_10: GL tag disintegration.
  have row_10 : (succ zero one) := by
    exact row_8.1.1.2
  -- chapter_8_line_7: GL tag disintegration.
  have row_7 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_8.1.1.1
  -- chapter_8_line_30: GL tag expansion.
  have row_30 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_7
  -- chapter_8_line_42: GL tag disintegration.
  have row_42 : (gl_fXYZ mul N N N) := by
    exact row_30.1.1.1.2
  -- chapter_8_line_41: GL tag expansion.
  have row_41 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_42
  -- chapter_8_line_40: GL tag disintegration.
  have row_40 : (gl_implication9 mul N) := by
    exact row_41.1.1.1.2
  -- chapter_8_line_39: GL tag expansion.
  have row_39 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication9] using row_40
  -- chapter_8_line_38: GL tag implication.
  have row_38 : (N v2) := by
    apply row_39
    exact row_13
  -- chapter_8_line_29: GL tag disintegration.
  have row_29 : (gl_fXYZ add N N N) := by
    exact row_30.1.1.1.1.1.1.1.1.2
  -- chapter_8_line_28: GL tag expansion.
  have row_28 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_29
  -- chapter_8_line_27: GL tag disintegration.
  have row_27 : (gl_implication10 add N) := by
    exact row_28.1.1.2
  -- chapter_8_line_26: GL tag expansion.
  have row_26 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication10] using row_27
  -- chapter_8_line_6: GL tag expansion for integration.
  have row_6 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_8_line_5: GL tag reformulation for integration and.
  have row_5 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_8_line_4: GL tag implication.
  have row_4 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_5
    exact row_7
    exact row_10
  have rule_row_12 := external_peano_externals_36_010 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_8_line_12: GL tag implication.
  have row_12 : (mul v2 v1 v3) := by
    apply rule_row_12
    exact row_13
  -- chapter_8_line_33: GL tag implication.
  have row_33 : (gl_preorder N mul v2 v3) := by
    apply row_34
    exact row_36
    exact row_12
  -- chapter_8_line_2: GL tag task formulation.
  have row_2 : (gl_preorder N add two v3) := by
    exact premise_2
  -- chapter_8_line_32: GL tag expansion.
  have row_32 : (¬ (∀ (v7 : α), ((N v7) → (¬ (add two v7 v3))))) := by
    simpa only [gl_preorder] using row_2
  have exists_row_32 : ∃ (v7 : α), ((N v7) ∧ (add two v7 v3)) := existsAndOfNotForallImpNot row_32
  obtain ⟨v7, witness_row_32⟩ := exists_row_32
  -- chapter_8_line_31: GL tag disintegration.
  have row_31 : (add two v7 v3) := by
    exact witness_row_32.2
  -- chapter_8_line_25: GL tag implication.
  have row_25 : (N v3) := by
    apply row_26
    exact row_31
  -- chapter_8_line_23: GL tag implication.
  have row_23 : (¬ (zero = v3)) := by
    apply row_24
    exact row_2
  -- chapter_8_line_21: GL tag implication.
  have row_21 : (gl_preorder N add one v3) := by
    apply row_22
    exact row_25
    exact row_23
  -- chapter_8_line_19: GL tag implication.
  have row_19 : (gl_preorder N add one v2) := by
    apply row_20
    exact row_33
    exact row_21
  -- chapter_8_line_17: GL tag implication.
  have row_17 : (¬ (zero = v2)) := by
    apply row_18
    exact row_19
  -- chapter_8_line_15: GL tag implication.
  have row_15 : (one = v2) := by
    apply row_16
    exact row_38
    exact row_17
    exact row_37
  -- chapter_8_line_14: GL tag symmetry of equality.
  have row_14 : (v2 = one) := by
    exact Eq.symm row_15
  -- chapter_8_line_11: GL tag equality1.
  have row_11 : (mul one v1 v3) := by
    have equality_source := row_12
    have equality_step_1 := row_14
    cases equality_step_1
    exact equality_source
  have rule_row_3 := external_peano_externals_36_033 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_8_line_3: GL tag implication.
  have row_3 : (v3 = v1) := by
    apply rule_row_3
    exact row_25
    exact row_11
  -- chapter_8_line_1: GL tag equality1.
  have row_1 : (gl_preorder N add two v1) := by
    have equality_source := row_2
    have equality_step_1 := row_3
    cases equality_step_1
    exact equality_source
  exact row_1

theorem fta_source_007
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_010 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_5 x_7 x_8 x_9) → (x_5 x_8 x_7 x_9))))))
    (external_peano_externals_36_033 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α), ((x_1 x_7) → (∀ (x_8 : α), ((x_5 x_6 x_8 x_7) → (x_7 = x_8))))))))
    (external_peano_externals_36_024 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_8 x_2) → ((x_1 x_8) → (x_2 = x_7)))))))
    (external_peano_externals_36_015 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_3 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_022 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_6 x_8) → (x_4 x_6 x_7 x_8))))))
    (external_peano_externals_36_017 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_6 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_021 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_6 x_8) → (x_3 x_7 x_8))))))
    (external_peano_externals_36_034 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (∀ (w1 : α), ((N w1) → (gl_or2 w1 zero N succ))))))
    (external_peano_externals_36_035 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (gl_existence11 N one succ))))
    (external_gauss_externals_24_018 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α) (x_7 : α) (x_8 : GLBinaryRelation α), ((gl_AnchorGauss x_1 x_2 x_3 x_4 x_5 x_6 x_7 x_8) → (∀ (x_9 : α) (x_10 : α), ((x_3 x_9 x_10) → (gl_preorder x_1 x_4 x_9 x_10))))))
    (external_peano_externals_36_006 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (x_4 x_8 x_7 x_9))))))
    (external_peano_externals_36_003 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (∀ (x_10 : α), ((x_3 x_8 x_10) → (∀ (x_11 : α), ((x_3 x_11 x_7) → (x_4 x_10 x_11 x_9))))))))))
    (external_peano_externals_36_002 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (∀ (w4 : α) (w5 : α), ((add w4 w2 w5) → (∀ (w6 : α), ((add w3 w4 w6) → (add w5 w1 w6))))))))))
    (external_peano_externals_36_029 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((gl_preorder x_1 x_4 x_7 x_8) → ((gl_preorder x_1 x_4 x_8 x_7) → (x_7 = x_8)))))))
    (external_peano_externals_36_005 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → ((x_8 = x_9) → (x_2 = x_7)))))))
    : (∀ (v1 : α) (v2 : α) (v3 : α), ((mul v1 v2 v3) → ((gl_preorder N add two v3) → ((N v1) → ((¬ (gl_preorder N add two v1)) → (gl_preorder N add two v2)))))) := by
  intro v1
  intro v2
  intro v3
  intro premise_1
  intro premise_2
  intro premise_3
  intro premise_4
  -- chapter_9_line_37: GL tag task formulation.
  have row_37 : (N v1) := by
    exact premise_3
  -- chapter_9_line_36: GL tag task formulation.
  have row_36 : (¬ (gl_preorder N add two v1)) := by
    exact premise_4
  -- chapter_9_line_33: GL tag theorem.
  have row_33 := fta_source_010 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_010
  -- chapter_9_line_23: GL tag theorem.
  have row_23 := fta_source_072 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_024
  -- chapter_9_line_21: GL tag theorem.
  have row_21 := fta_source_059 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_015 external_peano_externals_36_022
  -- chapter_9_line_19: GL tag theorem.
  have row_19 := fta_source_041 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_010 external_peano_externals_36_017 external_peano_externals_36_021 external_peano_externals_36_015 external_peano_externals_36_022
  -- chapter_9_line_17: GL tag theorem.
  have row_17 := fta_source_066 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_017 external_peano_externals_36_021
  -- chapter_9_line_15: GL tag theorem.
  have row_15 := fta_source_058 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_034 external_peano_externals_36_022 external_peano_externals_36_035 external_peano_externals_36_015 external_gauss_externals_24_018 external_peano_externals_36_006 external_peano_externals_36_003 external_peano_externals_36_002 external_peano_externals_36_029 external_peano_externals_36_005
  -- chapter_9_line_12: GL tag task formulation.
  have row_12 : (mul v1 v2 v3) := by
    exact premise_1
  -- chapter_9_line_9: GL tag task formulation.
  have row_9 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_9_line_34: GL tag anchor handling.
  have row_34 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact row_9
  -- chapter_9_line_8: GL tag expansion.
  have row_8 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_9
  -- chapter_9_line_10: GL tag disintegration.
  have row_10 : (succ zero one) := by
    exact row_8.1.1.2
  -- chapter_9_line_7: GL tag disintegration.
  have row_7 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_8.1.1.1
  -- chapter_9_line_29: GL tag expansion.
  have row_29 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_7
  -- chapter_9_line_28: GL tag disintegration.
  have row_28 : (gl_fXYZ add N N N) := by
    exact row_29.1.1.1.1.1.1.1.1.2
  -- chapter_9_line_27: GL tag expansion.
  have row_27 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_28
  -- chapter_9_line_26: GL tag disintegration.
  have row_26 : (gl_implication10 add N) := by
    exact row_27.1.1.2
  -- chapter_9_line_25: GL tag expansion.
  have row_25 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication10] using row_26
  -- chapter_9_line_6: GL tag expansion for integration.
  have row_6 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_9_line_5: GL tag reformulation for integration and.
  have row_5 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_9_line_4: GL tag implication.
  have row_4 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_5
    exact row_7
    exact row_10
  have rule_row_35 := external_peano_externals_36_010 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_9_line_35: GL tag implication.
  have row_35 : (mul v2 v1 v3) := by
    apply rule_row_35
    exact row_12
  -- chapter_9_line_32: GL tag implication.
  have row_32 : (gl_preorder N mul v1 v3) := by
    apply row_33
    exact row_35
  -- chapter_9_line_2: GL tag task formulation.
  have row_2 : (gl_preorder N add two v3) := by
    exact premise_2
  -- chapter_9_line_31: GL tag expansion.
  have row_31 : (¬ (∀ (v7 : α), ((N v7) → (¬ (add two v7 v3))))) := by
    simpa only [gl_preorder] using row_2
  have exists_row_31 : ∃ (v7 : α), ((N v7) ∧ (add two v7 v3)) := existsAndOfNotForallImpNot row_31
  obtain ⟨v7, witness_row_31⟩ := exists_row_31
  -- chapter_9_line_30: GL tag disintegration.
  have row_30 : (add two v7 v3) := by
    exact witness_row_31.2
  -- chapter_9_line_24: GL tag implication.
  have row_24 : (N v3) := by
    apply row_25
    exact row_30
  -- chapter_9_line_22: GL tag implication.
  have row_22 : (¬ (zero = v3)) := by
    apply row_23
    exact row_2
  -- chapter_9_line_20: GL tag implication.
  have row_20 : (gl_preorder N add one v3) := by
    apply row_21
    exact row_24
    exact row_22
  -- chapter_9_line_18: GL tag implication.
  have row_18 : (gl_preorder N add one v1) := by
    apply row_19
    exact row_32
    exact row_20
  -- chapter_9_line_16: GL tag implication.
  have row_16 : (¬ (zero = v1)) := by
    apply row_17
    exact row_18
  -- chapter_9_line_14: GL tag implication.
  have row_14 : (one = v1) := by
    apply row_15
    exact row_37
    exact row_16
    exact row_36
  -- chapter_9_line_13: GL tag symmetry of equality.
  have row_13 : (v1 = one) := by
    exact Eq.symm row_14
  -- chapter_9_line_11: GL tag equality1.
  have row_11 : (mul one v2 v3) := by
    have equality_source := row_12
    have equality_step_1 := row_13
    cases equality_step_1
    exact equality_source
  have rule_row_3 := external_peano_externals_36_033 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_9_line_3: GL tag implication.
  have row_3 : (v3 = v2) := by
    apply rule_row_3
    exact row_24
    exact row_11
  -- chapter_9_line_1: GL tag equality1.
  have row_1 : (gl_preorder N add two v2) := by
    have equality_source := row_2
    have equality_step_1 := row_3
    cases equality_step_1
    exact equality_source
  exact row_1

theorem fta_source_009
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_010 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_5 x_7 x_8 x_9) → (x_5 x_8 x_7 x_9))))))
    (external_peano_externals_36_017 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_6 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_021 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_6 x_8) → (x_3 x_7 x_8))))))
    (external_peano_externals_36_015 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_3 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_022 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_6 x_8) → (x_4 x_6 x_7 x_8))))))
    (external_peano_externals_36_006 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (x_4 x_8 x_7 x_9))))))
    (external_gauss_externals_24_018 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α) (x_7 : α) (x_8 : GLBinaryRelation α), ((gl_AnchorGauss x_1 x_2 x_3 x_4 x_5 x_6 x_7 x_8) → (∀ (x_9 : α) (x_10 : α), ((x_3 x_9 x_10) → (gl_preorder x_1 x_4 x_9 x_10))))))
    (external_peano_externals_36_035 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (gl_existence11 N one succ))))
    (external_peano_externals_36_029 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((gl_preorder x_1 x_4 x_7 x_8) → ((gl_preorder x_1 x_4 x_8 x_7) → (x_7 = x_8)))))))
    (external_peano_externals_36_003 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (∀ (x_10 : α), ((x_3 x_8 x_10) → (∀ (x_11 : α), ((x_3 x_11 x_7) → (x_4 x_10 x_11 x_9))))))))))
    (external_peano_externals_36_005 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → ((x_8 = x_9) → (x_2 = x_7)))))))
    (external_peano_externals_36_033 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α), ((x_1 x_7) → (∀ (x_8 : α), ((x_5 x_6 x_8 x_7) → (x_7 = x_8))))))))
    (external_peano_externals_36_008 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (∀ (w4 : α) (w5 : α), ((mul w4 w1 w5) → (∀ (w6 : α), ((mul w4 w2 w6) → (∀ (w7 : α), ((mul w4 w3 w7) → (add w5 w6 w7))))))))))))
    (external_peano_externals_36_024 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_8 x_2) → ((x_1 x_8) → (x_2 = x_7)))))))
    (external_peano_externals_36_025 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_8 x_2) → (x_4 x_8 x_7 x_2))))))
    : (∀ (v1 : α) (v2 : α) (v3 : α), ((mul v1 v2 v3) → ((gl_preorder N add two v1) → ((gl_preorder N add one v3) → (gl_strictOrder N add v2 v3))))) := by
  intro v1
  intro v2
  intro v3
  intro premise_1
  intro premise_2
  intro premise_3
  -- chapter_11_line_26: GL tag task formulation.
  have row_26 : (gl_preorder N add two v1) := by
    exact premise_2
  -- chapter_11_line_25: GL tag expansion.
  have row_25 : (¬ (∀ (v9 : α), ((N v9) → (¬ (add two v9 v1))))) := by
    simpa only [gl_preorder] using row_26
  have exists_row_25 : ∃ (v9 : α), ((N v9) ∧ (add two v9 v1)) := existsAndOfNotForallImpNot row_25
  obtain ⟨v9, witness_row_25⟩ := exists_row_25
  -- chapter_11_line_24: GL tag disintegration.
  have row_24 : (add two v9 v1) := by
    exact witness_row_25.2
  -- chapter_11_line_17: GL tag expansion for integration.
  have row_17 : ((gl_preorder N mul v2 v3) ↔ (¬ (∀ (v5 : α), ((N v5) → (¬ (mul v2 v5 v3)))))) := by
    exact Iff.rfl
  -- chapter_11_line_16: GL tag reformulation for integration >[bound].
  have row_16 : (∀ (v4 : α), ((N v4) → ((mul v2 v4 v3) → (gl_preorder N mul v2 v3)))) := by
    intro v4
    intro integration_premise_1
    intro integration_premise_2
    apply (row_17).2
    intro universal_counterexample
    exact universal_counterexample v4 integration_premise_1 integration_premise_2
  -- chapter_11_line_14: GL tag task formulation.
  have row_14 : (gl_preorder N add one v3) := by
    exact premise_3
  -- chapter_11_line_13: GL tag theorem.
  have row_13 := fta_source_041 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_010 external_peano_externals_36_017 external_peano_externals_36_021 external_peano_externals_36_015 external_peano_externals_36_022
  -- chapter_11_line_11: GL tag task formulation.
  have row_11 : (mul v1 v2 v3) := by
    exact premise_1
  -- chapter_11_line_7: GL tag expansion for integration.
  have row_7 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_11_line_6: GL tag reformulation for integration and.
  have row_6 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_11_line_3: GL tag task formulation.
  have row_3 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_11_line_9: GL tag expansion.
  have row_9 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_3
  -- chapter_11_line_10: GL tag disintegration.
  have row_10 : (succ zero one) := by
    exact row_9.1.1.2
  -- chapter_11_line_8: GL tag disintegration.
  have row_8 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_9.1.1.1
  -- chapter_11_line_23: GL tag expansion.
  have row_23 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_8
  -- chapter_11_line_22: GL tag disintegration.
  have row_22 : (gl_fXYZ add N N N) := by
    exact row_23.1.1.1.1.1.1.1.1.2
  -- chapter_11_line_21: GL tag expansion.
  have row_21 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_22
  -- chapter_11_line_20: GL tag disintegration.
  have row_20 : (gl_implication10 add N) := by
    exact row_21.1.1.2
  -- chapter_11_line_19: GL tag expansion.
  have row_19 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication10] using row_20
  -- chapter_11_line_18: GL tag implication.
  have row_18 : (N v1) := by
    apply row_19
    exact row_24
  -- chapter_11_line_5: GL tag implication.
  have row_5 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_6
    exact row_8
    exact row_10
  have rule_row_4 := external_peano_externals_36_010 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_11_line_4: GL tag implication.
  have row_4 : (mul v2 v1 v3) := by
    apply rule_row_4
    exact row_11
  -- chapter_11_line_15: GL tag implication.
  have row_15 : (gl_preorder N mul v2 v3) := by
    apply row_16
    exact row_18
    exact row_4
  -- chapter_11_line_12: GL tag implication.
  have row_12 : (gl_preorder N add one v2) := by
    apply row_13
    exact row_15
    exact row_14
  -- chapter_11_line_2: GL tag theorem.
  have row_2 := fta_source_071 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_017 external_peano_externals_36_021 external_peano_externals_36_006 external_peano_externals_36_015 external_peano_externals_36_010 external_peano_externals_36_022 external_gauss_externals_24_018 external_peano_externals_36_035 external_peano_externals_36_029 external_peano_externals_36_003 external_peano_externals_36_005 external_peano_externals_36_033 external_peano_externals_36_008 external_peano_externals_36_024 external_peano_externals_36_025
  -- chapter_11_line_1: GL tag implication.
  have row_1 : (gl_strictOrder N add v2 v3) := by
    apply row_2
    exact row_12
    exact row_26
    exact row_4
  exact row_1

theorem fta_source_055
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_019 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_2 x_8) → (x_4 x_2 x_7 x_8))))))
    (external_peano_externals_36_006 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (x_4 x_8 x_7 x_9))))))
    (external_peano_externals_36_015 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_3 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_034 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (∀ (w1 : α), ((N w1) → (gl_or2 w1 zero N succ))))))
    (external_peano_externals_36_003 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (∀ (x_10 : α), ((x_3 x_8 x_10) → (∀ (x_11 : α), ((x_3 x_11 x_7) → (x_4 x_10 x_11 x_9))))))))))
    (external_peano_externals_36_004 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (∀ (x_10 : α), ((x_4 x_7 x_10 x_9) → (x_8 = x_10))))))))
    : (∀ (v1 : α), ((N v1) → (∀ (v2 : α), ((N v2) → (gl_or10 N add v1 v2))))) := by
  intro v1
  intro premise_1
  intro v2
  intro premise_2
  have or_parent_1 := fta_source_053 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_019 external_peano_externals_36_006 external_peano_externals_36_015 external_peano_externals_36_034 external_peano_externals_36_003 external_peano_externals_36_004
  have or_parent_2 := fta_source_054 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_019 external_peano_externals_36_006 external_peano_externals_36_015 external_peano_externals_36_034 external_peano_externals_36_003 external_peano_externals_36_004
  -- chapter_69_line_1: GL tag or theorem.
  have row_1 : (∀ (v1 : α), ((N v1) → (∀ (v2 : α), ((N v2) → (gl_or10 N add v1 v2))))) := by
    classical
    intro v1
    intro or_parent_premise_1
    intro v2
    intro or_parent_premise_2
    simp only [gl_or10]
    by_cases or_case_1 : (gl_strictOrder N add v1 v2)
    · exact Or.inl (Or.inl (or_case_1))
    ·
      by_cases or_case_2 : (v1 = v2)
      · exact Or.inl (Or.inr (or_case_2))
      ·
        exact Or.inr ((or_parent_1 v1 or_parent_premise_1 v2 or_parent_premise_2 or_case_1 or_case_2))
  solve_by_elim [row_1]

theorem fta_source_062
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_034 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (∀ (w1 : α), ((N w1) → (gl_or2 w1 zero N succ))))))
    (external_peano_externals_36_015 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_3 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_022 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_6 x_8) → (x_4 x_6 x_7 x_8))))))
    (external_peano_externals_36_003 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (∀ (x_10 : α), ((x_3 x_8 x_10) → (∀ (x_11 : α), ((x_3 x_11 x_7) → (x_4 x_10 x_11 x_9))))))))))
    (external_gauss_externals_24_018 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α) (x_7 : α) (x_8 : GLBinaryRelation α), ((gl_AnchorGauss x_1 x_2 x_3 x_4 x_5 x_6 x_7 x_8) → (∀ (x_9 : α) (x_10 : α), ((x_3 x_9 x_10) → (gl_preorder x_1 x_4 x_9 x_10))))))
    (external_peano_externals_36_035 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (gl_existence11 N one succ))))
    (external_peano_externals_36_029 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((gl_preorder x_1 x_4 x_7 x_8) → ((gl_preorder x_1 x_4 x_8 x_7) → (x_7 = x_8)))))))
    (external_peano_externals_36_005 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → ((x_8 = x_9) → (x_2 = x_7)))))))
    (external_peano_externals_36_006 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (x_4 x_8 x_7 x_9))))))
    (external_peano_externals_36_002 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (∀ (w4 : α) (w5 : α), ((add w4 w2 w5) → (∀ (w6 : α), ((add w3 w4 w6) → (add w5 w1 w6))))))))))
    : (∀ (v1 : α), ((N v1) → (gl_or6 zero v1 one N add two))) := by
  intro v1
  intro premise_1
  have or_parent_1 := fta_source_057 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_034 external_peano_externals_36_015 external_peano_externals_36_022 external_peano_externals_36_003 external_gauss_externals_24_018 external_peano_externals_36_035 external_peano_externals_36_029 external_peano_externals_36_005
  have or_parent_2 := fta_source_058 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_034 external_peano_externals_36_022 external_peano_externals_36_035 external_peano_externals_36_015 external_gauss_externals_24_018 external_peano_externals_36_006 external_peano_externals_36_003 external_peano_externals_36_002 external_peano_externals_36_029 external_peano_externals_36_005
  -- chapter_82_line_1: GL tag or theorem.
  have row_1 : (∀ (v1 : α), ((N v1) → (gl_or6 zero v1 one N add two))) := by
    classical
    intro v1
    intro or_parent_premise_1
    simp only [gl_or6]
    by_cases or_case_1 : (zero = v1)
    · exact Or.inl (Or.inl (or_case_1))
    ·
      by_cases or_case_2 : (one = v1)
      · exact Or.inl (Or.inr (or_case_2))
      ·
        exact Or.inr ((or_parent_1 v1 or_parent_premise_1 or_case_1 or_case_2))
  solve_by_elim [row_1]

theorem fta_source_068
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_029 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((gl_preorder x_1 x_4 x_7 x_8) → ((gl_preorder x_1 x_4 x_8 x_7) → (x_7 = x_8)))))))
    (external_peano_externals_36_035 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (gl_existence11 N one succ))))
    (external_gauss_externals_24_018 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α) (x_7 : α) (x_8 : GLBinaryRelation α), ((gl_AnchorGauss x_1 x_2 x_3 x_4 x_5 x_6 x_7 x_8) → (∀ (x_9 : α) (x_10 : α), ((x_3 x_9 x_10) → (gl_preorder x_1 x_4 x_9 x_10))))))
    (external_gauss_externals_24_021 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α) (x_7 : α) (x_8 : GLBinaryRelation α), ((gl_AnchorGauss x_1 x_2 x_3 x_4 x_5 x_6 x_7 x_8) → (∀ (x_9 : α) (x_10 : α), ((gl_preorder x_1 x_4 x_9 x_10) → (∀ (x_11 : α), ((gl_preorder x_1 x_4 x_10 x_11) → (gl_preorder x_1 x_4 x_9 x_11))))))))
    (external_peano_externals_36_010 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_5 x_7 x_8 x_9) → (x_5 x_8 x_7 x_9))))))
    (external_peano_externals_36_017 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_6 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_021 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_6 x_8) → (x_3 x_7 x_8))))))
    (external_peano_externals_36_015 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_3 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_002 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (∀ (w4 : α) (w5 : α), ((add w4 w2 w5) → (∀ (w6 : α), ((add w3 w4 w6) → (add w5 w1 w6))))))))))
    (external_peano_externals_36_006 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (x_4 x_8 x_7 x_9))))))
    (external_peano_externals_36_034 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (∀ (w1 : α), ((N w1) → (gl_or2 w1 zero N succ))))))
    (external_peano_externals_36_003 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (∀ (x_10 : α), ((x_3 x_8 x_10) → (∀ (x_11 : α), ((x_3 x_11 x_7) → (x_4 x_10 x_11 x_9))))))))))
    (external_peano_externals_36_008 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (∀ (w4 : α) (w5 : α), ((mul w4 w1 w5) → (∀ (w6 : α), ((mul w4 w2 w6) → (∀ (w7 : α), ((mul w4 w3 w7) → (add w5 w6 w7))))))))))))
    (external_peano_externals_36_004 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (∀ (x_10 : α), ((x_4 x_7 x_10 x_9) → (x_8 = x_10))))))))
    (external_peano_externals_36_005 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → ((x_8 = x_9) → (x_2 = x_7)))))))
    (external_peano_externals_36_022 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_6 x_8) → (x_4 x_6 x_7 x_8))))))
    : (∀ (v1 : α), ((gl_preorder N add one v1) → (∀ (v2 : α) (v3 : α), ((mul v2 v1 v3) → (∀ (v4 : α), ((mul v4 v1 v3) → (v2 = v4))))))) := by
  intro v1
  intro premise_1
  intro v2
  intro v3
  intro premise_2
  intro v4
  intro premise_3
  -- chapter_92_line_23: GL tag task formulation.
  have row_23 : (gl_preorder N add one v1) := by
    exact premise_1
  -- chapter_92_line_16: GL tag theorem.
  have row_16 := fta_source_050 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_92_line_14: GL tag variable copy.
  have row_14 : (v3 = v3) := by
    rfl
  -- chapter_92_line_26: GL tag symmetry of equality.
  have row_26 : (v3 = v3) := by
    exact Eq.symm row_14
  -- chapter_92_line_13: GL tag task formulation.
  have row_13 : (mul v4 v1 v3) := by
    exact premise_3
  -- chapter_92_line_12: GL tag equality1.
  have row_12 : (mul v4 v1 v3) := by
    have equality_source := row_13
    have equality_step_1 := row_14
    cases equality_step_1
    exact equality_source
  -- chapter_92_line_11: GL tag task formulation.
  have row_11 : (mul v2 v1 v3) := by
    exact premise_2
  -- chapter_92_line_28: GL tag equality1.
  have row_28 : (mul v2 v1 v3) := by
    have equality_source := row_11
    have equality_step_1 := row_14
    cases equality_step_1
    exact equality_source
  -- chapter_92_line_10: GL tag theorem.
  have row_10 := fta_source_067 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_035 external_gauss_externals_24_018 external_gauss_externals_24_021 external_peano_externals_36_010 external_peano_externals_36_017 external_peano_externals_36_021 external_peano_externals_36_015 external_peano_externals_36_002 external_peano_externals_36_006 external_peano_externals_36_034 external_peano_externals_36_003 external_peano_externals_36_008 external_peano_externals_36_004 external_peano_externals_36_029 external_peano_externals_36_005 external_peano_externals_36_022
  -- chapter_92_line_7: GL tag task formulation.
  have row_7 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_92_line_6: GL tag expansion.
  have row_6 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_7
  -- chapter_92_line_8: GL tag disintegration.
  have row_8 : (succ zero one) := by
    exact row_6.1.1.2
  -- chapter_92_line_5: GL tag disintegration.
  have row_5 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_6.1.1.1
  -- chapter_92_line_22: GL tag expansion.
  have row_22 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_5
  -- chapter_92_line_21: GL tag disintegration.
  have row_21 : (gl_fXYZ mul N N N) := by
    exact row_22.1.1.1.2
  -- chapter_92_line_20: GL tag expansion.
  have row_20 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_21
  -- chapter_92_line_19: GL tag disintegration.
  have row_19 : (gl_implication10 mul N) := by
    exact row_20.1.1.2
  -- chapter_92_line_18: GL tag expansion.
  have row_18 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication10] using row_19
  -- chapter_92_line_27: GL tag implication.
  have row_27 : (N v3) := by
    apply row_18
    exact row_28
  -- chapter_92_line_25: GL tag implication.
  have row_25 : (gl_preorder N add v3 v3) := by
    apply row_16
    exact row_27
    exact row_26
  -- chapter_92_line_24: GL tag implication.
  have row_24 : (gl_preorder N add v4 v2) := by
    apply row_10
    exact row_23
    exact row_12
    exact row_11
    exact row_25
  -- chapter_92_line_17: GL tag implication.
  have row_17 : (N v3) := by
    apply row_18
    exact row_13
  -- chapter_92_line_15: GL tag implication.
  have row_15 : (gl_preorder N add v3 v3) := by
    apply row_16
    exact row_17
    exact row_14
  -- chapter_92_line_9: GL tag implication.
  have row_9 : (gl_preorder N add v2 v4) := by
    apply row_10
    exact row_23
    exact row_11
    exact row_12
    exact row_15
  -- chapter_92_line_4: GL tag expansion for integration.
  have row_4 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_92_line_3: GL tag reformulation for integration and.
  have row_3 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_92_line_2: GL tag implication.
  have row_2 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_3
    exact row_5
    exact row_8
  have rule_row_1 := external_peano_externals_36_029 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_92_line_1: GL tag implication.
  have row_1 : (v2 = v4) := by
    apply rule_row_1
    exact row_9
    exact row_24
  exact row_1

theorem fta_source_005
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_010 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_5 x_7 x_8 x_9) → (x_5 x_8 x_7 x_9))))))
    (external_peano_externals_36_009 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (∀ (w4 : α) (w5 : α), ((mul w4 w2 w5) → (∀ (w6 : α), ((mul w3 w4 w6) → (mul w5 w1 w6))))))))))
    (external_peano_externals_36_029 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((gl_preorder x_1 x_4 x_7 x_8) → ((gl_preorder x_1 x_4 x_8 x_7) → (x_7 = x_8)))))))
    (external_peano_externals_36_035 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (gl_existence11 N one succ))))
    (external_gauss_externals_24_018 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α) (x_7 : α) (x_8 : GLBinaryRelation α), ((gl_AnchorGauss x_1 x_2 x_3 x_4 x_5 x_6 x_7 x_8) → (∀ (x_9 : α) (x_10 : α), ((x_3 x_9 x_10) → (gl_preorder x_1 x_4 x_9 x_10))))))
    (external_gauss_externals_24_021 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α) (x_7 : α) (x_8 : GLBinaryRelation α), ((gl_AnchorGauss x_1 x_2 x_3 x_4 x_5 x_6 x_7 x_8) → (∀ (x_9 : α) (x_10 : α), ((gl_preorder x_1 x_4 x_9 x_10) → (∀ (x_11 : α), ((gl_preorder x_1 x_4 x_10 x_11) → (gl_preorder x_1 x_4 x_9 x_11))))))))
    (external_peano_externals_36_017 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_6 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_021 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_6 x_8) → (x_3 x_7 x_8))))))
    (external_peano_externals_36_015 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_3 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_002 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (∀ (w4 : α) (w5 : α), ((add w4 w2 w5) → (∀ (w6 : α), ((add w3 w4 w6) → (add w5 w1 w6))))))))))
    (external_peano_externals_36_006 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (x_4 x_8 x_7 x_9))))))
    (external_peano_externals_36_034 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (∀ (w1 : α), ((N w1) → (gl_or2 w1 zero N succ))))))
    (external_peano_externals_36_003 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (∀ (x_10 : α), ((x_3 x_8 x_10) → (∀ (x_11 : α), ((x_3 x_11 x_7) → (x_4 x_10 x_11 x_9))))))))))
    (external_peano_externals_36_008 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (∀ (w4 : α) (w5 : α), ((mul w4 w1 w5) → (∀ (w6 : α), ((mul w4 w2 w6) → (∀ (w7 : α), ((mul w4 w3 w7) → (add w5 w6 w7))))))))))))
    (external_peano_externals_36_004 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (∀ (x_10 : α), ((x_4 x_7 x_10 x_9) → (x_8 = x_10))))))))
    (external_peano_externals_36_005 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → ((x_8 = x_9) → (x_2 = x_7)))))))
    (external_peano_externals_36_022 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_6 x_8) → (x_4 x_6 x_7 x_8))))))
    : (∀ (v1 : α) (v2 : α) (v3 : α), ((mul v1 v2 v3) → (∀ (v4 : α) (v5 : α), ((mul v1 v4 v5) → ((gl_preorder N mul v3 v5) → ((gl_preorder N add one v1) → (gl_preorder N mul v2 v4))))))) := by
  intro v1
  intro v2
  intro v3
  intro premise_1
  intro v4
  intro v5
  intro premise_2
  intro premise_3
  intro premise_4
  -- chapter_7_line_32: GL tag task formulation.
  have row_32 : (gl_preorder N add one v1) := by
    exact premise_4
  -- chapter_7_line_29: GL tag task formulation.
  have row_29 : (mul v1 v4 v5) := by
    exact premise_2
  -- chapter_7_line_27: GL tag expansion for integration.
  have row_27 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_7_line_26: GL tag reformulation for integration and.
  have row_26 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_7_line_23: GL tag theorem.
  have row_23 := fta_source_068 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_029 external_peano_externals_36_035 external_gauss_externals_24_018 external_gauss_externals_24_021 external_peano_externals_36_010 external_peano_externals_36_017 external_peano_externals_36_021 external_peano_externals_36_015 external_peano_externals_36_002 external_peano_externals_36_006 external_peano_externals_36_034 external_peano_externals_36_003 external_peano_externals_36_008 external_peano_externals_36_004 external_peano_externals_36_005 external_peano_externals_36_022
  -- chapter_7_line_21: GL tag task formulation.
  have row_21 : (gl_preorder N mul v3 v5) := by
    exact premise_3
  -- chapter_7_line_20: GL tag expansion.
  have row_20 : (¬ (∀ (v7 : α), ((N v7) → (¬ (mul v3 v7 v5))))) := by
    simpa only [gl_preorder] using row_21
  have exists_row_20 : ∃ (v7 : α), ((N v7) ∧ (mul v3 v7 v5)) := existsAndOfNotForallImpNot row_20
  obtain ⟨v7, witness_row_20⟩ := exists_row_20
  -- chapter_7_line_31: GL tag disintegration.
  have row_31 : (mul v3 v7 v5) := by
    exact witness_row_20.2
  -- chapter_7_line_19: GL tag disintegration.
  have row_19 : (N v7) := by
    exact witness_row_20.1
  -- chapter_7_line_18: GL tag task formulation.
  have row_18 : (mul v1 v2 v3) := by
    exact premise_1
  -- chapter_7_line_4: GL tag task formulation.
  have row_4 : (gl_AnchorFTA N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_7_line_14: GL tag expansion.
  have row_14 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorFTA] using row_4
  -- chapter_7_line_28: GL tag disintegration.
  have row_28 : (succ zero one) := by
    exact row_14.1.1.2
  -- chapter_7_line_13: GL tag disintegration.
  have row_13 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_14.1.1.1
  -- chapter_7_line_25: GL tag implication.
  have row_25 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_26
    exact row_13
    exact row_28
  have rule_row_24 := external_peano_externals_36_010 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_7_line_24: GL tag implication.
  have row_24 : (mul v4 v1 v5) := by
    apply rule_row_24
    exact row_29
  -- chapter_7_line_12: GL tag expansion.
  have row_12 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_13
  -- chapter_7_line_11: GL tag disintegration.
  have row_11 : (gl_fXYZ mul N N N) := by
    exact row_12.1.1.1.2
  -- chapter_7_line_10: GL tag expansion.
  have row_10 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_11
  -- chapter_7_line_17: GL tag disintegration.
  have row_17 : (gl_implication9 mul N) := by
    exact row_10.1.1.1.2
  -- chapter_7_line_16: GL tag expansion.
  have row_16 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication9] using row_17
  -- chapter_7_line_15: GL tag implication.
  have row_15 : (N v2) := by
    apply row_16
    exact row_18
  -- chapter_7_line_9: GL tag disintegration.
  have row_9 : (gl_implication13 N N N mul) := by
    exact row_10.1.2
  -- chapter_7_line_8: GL tag expansion.
  have row_8 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication13] using row_9
  -- chapter_7_line_7: GL tag implication.
  have row_7 : (gl_existence1 N v7 v2 mul) := by
    apply row_8
    exact row_19
    exact row_15
  -- chapter_7_line_6: GL tag expansion.
  have row_6 : (¬ (∀ (v6 : α), ((N v6) → (¬ (mul v7 v2 v6))))) := by
    simpa only [gl_existence1] using row_7
  have exists_row_6 : ∃ (v6 : α), ((N v6) ∧ (mul v7 v2 v6)) := existsAndOfNotForallImpNot row_6
  obtain ⟨v6, witness_row_6⟩ := exists_row_6
  -- chapter_7_line_5: GL tag disintegration.
  have row_5 : (mul v7 v2 v6) := by
    exact witness_row_6.2
  have rule_row_30 := external_peano_externals_36_009 N zero succ add mul one (anchorPeanoOfFTA N zero succ add mul one two identity anchor)
  -- chapter_7_line_30: GL tag implication.
  have row_30 : (mul v6 v1 v5) := by
    apply rule_row_30
    exact row_18
    exact row_5
    exact row_31
  -- chapter_7_line_22: GL tag implication.
  have row_22 : (v6 = v4) := by
    apply row_23
    exact row_32
    exact row_30
    exact row_24
  -- chapter_7_line_3: GL tag theorem.
  have row_3 := fta_source_010 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_010
  -- chapter_7_line_2: GL tag implication.
  have row_2 : (gl_preorder N mul v2 v6) := by
    apply row_3
    exact row_5
  -- chapter_7_line_1: GL tag equality1.
  have row_1 : (gl_preorder N mul v2 v4) := by
    have equality_source := row_2
    have equality_step_1 := row_22
    cases equality_step_1
    exact equality_source
  exact row_1

theorem fta_source_008
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorFTA N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (induction_N : GLSet α)
        (induction_zero : α)
        (induction_succ : GLBinaryRelation α)
        (induction_add induction_mul : GLTernaryRelation α)
        (induction_one induction_two : α)
        (induction_identity : GLBinaryRelation α),
        gl_AnchorFTA induction_N induction_zero induction_succ
          induction_add induction_mul induction_one induction_two
          induction_identity →
        ∀ (P : α → Prop) (k : α),
          P induction_zero →
          (∀ n, induction_N n → P n →
            ∀ m, induction_succ n m → P m) →
          induction_N k →
          P k)
    (external_peano_externals_36_010 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_5 x_7 x_8 x_9) → (x_5 x_8 x_7 x_9))))))
    (external_peano_externals_36_033 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α), ((x_1 x_7) → (∀ (x_8 : α), ((x_5 x_6 x_8 x_7) → (x_7 = x_8))))))))
    (external_peano_externals_36_024 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_8 x_2) → ((x_1 x_8) → (x_2 = x_7)))))))
    (external_peano_externals_36_015 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_3 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_022 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_6 x_8) → (x_4 x_6 x_7 x_8))))))
    (external_peano_externals_36_017 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_6 x_7 x_8) → (x_4 x_7 x_6 x_8))))))
    (external_peano_externals_36_021 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((x_4 x_7 x_6 x_8) → (x_3 x_7 x_8))))))
    (external_peano_externals_36_034 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (∀ (w1 : α), ((N w1) → (gl_or2 w1 zero N succ))))))
    (external_peano_externals_36_035 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (gl_existence11 N one succ))))
    (external_gauss_externals_24_018 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α) (x_7 : α) (x_8 : GLBinaryRelation α), ((gl_AnchorGauss x_1 x_2 x_3 x_4 x_5 x_6 x_7 x_8) → (∀ (x_9 : α) (x_10 : α), ((x_3 x_9 x_10) → (gl_preorder x_1 x_4 x_9 x_10))))))
    (external_peano_externals_36_006 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (x_4 x_8 x_7 x_9))))))
    (external_peano_externals_36_003 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (∀ (x_10 : α), ((x_3 x_8 x_10) → (∀ (x_11 : α), ((x_3 x_11 x_7) → (x_4 x_10 x_11 x_9))))))))))
    (external_peano_externals_36_002 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (∀ (w4 : α) (w5 : α), ((add w4 w2 w5) → (∀ (w6 : α), ((add w3 w4 w6) → (add w5 w1 w6))))))))))
    (external_peano_externals_36_029 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α), ((gl_preorder x_1 x_4 x_7 x_8) → ((gl_preorder x_1 x_4 x_8 x_7) → (x_7 = x_8)))))))
    (external_peano_externals_36_005 : (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → ((x_8 = x_9) → (x_2 = x_7)))))))
    : (∀ (v1 : α) (v2 : α) (v3 : α), ((mul v1 v2 v3) → ((gl_preorder N add two v3) → ((N v1) → (gl_or9 N add two v2 v1))))) := by
  intro v1
  intro v2
  intro v3
  intro premise_1
  intro premise_2
  intro premise_3
  have or_parent_1 := fta_source_006 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_010 external_peano_externals_36_033 external_peano_externals_36_024 external_peano_externals_36_015 external_peano_externals_36_022 external_peano_externals_36_017 external_peano_externals_36_021 external_peano_externals_36_034 external_peano_externals_36_035 external_gauss_externals_24_018 external_peano_externals_36_006 external_peano_externals_36_003 external_peano_externals_36_002 external_peano_externals_36_029 external_peano_externals_36_005
  have or_parent_2 := fta_source_007 N zero succ add mul one two identity anchor relationalInduction external_peano_externals_36_010 external_peano_externals_36_033 external_peano_externals_36_024 external_peano_externals_36_015 external_peano_externals_36_022 external_peano_externals_36_017 external_peano_externals_36_021 external_peano_externals_36_034 external_peano_externals_36_035 external_gauss_externals_24_018 external_peano_externals_36_006 external_peano_externals_36_003 external_peano_externals_36_002 external_peano_externals_36_029 external_peano_externals_36_005
  -- chapter_10_line_1: GL tag or theorem.
  have row_1 : (∀ (v1 : α) (v2 : α) (v3 : α), ((mul v1 v2 v3) → ((gl_preorder N add two v3) → ((N v1) → (gl_or9 N add two v2 v1))))) := by
    classical
    intro v1
    intro v2
    intro v3
    intro or_parent_premise_1
    intro or_parent_premise_2
    intro or_parent_premise_3
    simp only [gl_or9]
    by_cases or_case_1 : (gl_preorder N add two v2)
    · exact Or.inl (or_case_1)
    ·
      exact Or.inr ((or_parent_1 v1 v2 v3 or_parent_premise_1 or_parent_premise_2 or_parent_premise_3 or_case_1))
  solve_by_elim [row_1]

end GLExport.FTA
