/-
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-/

import GLExport.Generated.Peano

set_option linter.unusedVariables false

namespace GLExport

universe u

private theorem anchorPeanoOfGauss
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorGauss N zero succ add mul one two identity)
    : gl_AnchorPeano N zero succ add mul one := by
  simp only [gl_AnchorGauss, gl_AnchorPeano] at anchor ⊢
  exact anchor.1.1

theorem gauss_source_064
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorGauss N zero succ add mul one two identity)
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
  -- chapter_140_line_9: GL tag task formulation.
  have row_9 : (add v1 v2 v3) := by
    exact premise_1
  -- chapter_140_line_7: GL tag task formulation.
  have row_7 : (gl_AnchorGauss N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_140_line_6: GL tag expansion.
  have row_6 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorGauss] using row_7
  -- chapter_140_line_8: GL tag disintegration.
  have row_8 : (succ zero one) := by
    exact row_6.1.1.2
  -- chapter_140_line_5: GL tag disintegration.
  have row_5 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_6.1.1.1
  -- chapter_140_line_4: GL tag expansion for integration.
  have row_4 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_140_line_3: GL tag reformulation for integration and.
  have row_3 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_140_line_2: GL tag implication.
  have row_2 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_3
    exact row_5
    exact row_8
  have rule_row_1 := peano_source_016 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_140_line_1: GL tag implication.
  have row_1 : (add v2 v1 v3) := by
    apply rule_row_1
    exact row_9
  exact row_1

theorem gauss_source_067
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorGauss N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (V1 : GLBinaryRelation α) (V2 : GLSet α), ((gl_fXY V1 V2 N) → (∀ (v1 : α), ((gl_interval N add zero v1 V2) → (gl_sequence N add zero v1 V1))))) := by
  intro V1
  intro V2
  intro premise_1
  intro v1
  intro premise_2
  -- chapter_147_line_5: GL tag task formulation.
  have row_5 : (gl_interval N add zero v1 V2) := by
    exact premise_2
  -- chapter_147_line_4: GL tag task formulation.
  have row_4 : (gl_fXY V1 V2 N) := by
    exact premise_1
  -- chapter_147_line_3: GL tag expansion for integration.
  have row_3 : ((gl_sequence N add zero v1 V1) ↔ (¬ (∀ (V4 : GLSet α), ((gl_interval N add zero v1 V4) → (¬ (gl_fXY V1 V4 N)))))) := by
    exact Iff.rfl
  -- chapter_147_line_2: GL tag reformulation for integration >[bound].
  have row_2 : (∀ (V3 : GLSet α), ((gl_interval N add zero v1 V3) → ((gl_fXY V1 V3 N) → (gl_sequence N add zero v1 V1)))) := by
    intro V3
    intro integration_premise_1
    intro integration_premise_2
    apply (row_3).2
    intro universal_counterexample
    exact universal_counterexample V3 integration_premise_1 integration_premise_2
  -- chapter_147_line_1: GL tag implication.
  have row_1 : (gl_sequence N add zero v1 V1) := by
    apply row_2
    exact row_5
    exact row_4
  exact row_1

theorem gauss_source_085
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorGauss N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (v3 : α), ((gl_preorder N add v3 v1) → (gl_preorder N add v3 v2))))) := by
  intro v1
  intro v2
  intro premise_1
  intro v3
  intro premise_2
  -- chapter_165_line_21: GL tag task formulation.
  have row_21 : (gl_preorder N add v3 v1) := by
    exact premise_2
  -- chapter_165_line_20: GL tag expansion.
  have row_20 : (¬ (∀ (v12 : α), ((N v12) → (¬ (add v3 v12 v1))))) := by
    simpa only [gl_preorder] using row_21
  have exists_row_20 : ∃ (v12 : α), ((N v12) ∧ (add v3 v12 v1)) := existsAndOfNotForallImpNot row_20
  obtain ⟨v12, witness_row_20⟩ := exists_row_20
  -- chapter_165_line_22: GL tag disintegration.
  have row_22 : (add v3 v12 v1) := by
    exact witness_row_20.2
  -- chapter_165_line_19: GL tag disintegration.
  have row_19 : (N v12) := by
    exact witness_row_20.1
  -- chapter_165_line_11: GL tag task formulation.
  have row_11 : (succ v1 v2) := by
    exact premise_1
  -- chapter_165_line_10: GL tag task formulation.
  have row_10 : (gl_AnchorGauss N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_165_line_9: GL tag expansion.
  have row_9 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorGauss] using row_10
  -- chapter_165_line_8: GL tag disintegration.
  have row_8 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_9.1.1.1
  -- chapter_165_line_7: GL tag expansion.
  have row_7 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_8
  -- chapter_165_line_18: GL tag disintegration.
  have row_18 : (gl_fXY succ N N) := by
    exact row_7.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_165_line_17: GL tag expansion.
  have row_17 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_18
  -- chapter_165_line_16: GL tag disintegration.
  have row_16 : (gl_implication4 N N succ) := by
    exact row_17.1.2
  -- chapter_165_line_15: GL tag expansion.
  have row_15 : (∀ (w1 : α), ((N w1) → (gl_existence0 N w1 succ))) := by
    simpa only [gl_implication4] using row_16
  -- chapter_165_line_14: GL tag implication.
  have row_14 : (gl_existence0 N v12 succ) := by
    apply row_15
    exact row_19
  -- chapter_165_line_13: GL tag expansion.
  have row_13 : (¬ (∀ (v5 : α), ((N v5) → (¬ (succ v12 v5))))) := by
    simpa only [gl_existence0] using row_14
  have exists_row_13 : ∃ (v5 : α), ((N v5) ∧ (succ v12 v5)) := existsAndOfNotForallImpNot row_13
  obtain ⟨v5, witness_row_13⟩ := exists_row_13
  -- chapter_165_line_23: GL tag disintegration.
  have row_23 : (N v5) := by
    exact witness_row_13.1
  -- chapter_165_line_12: GL tag disintegration.
  have row_12 : (succ v12 v5) := by
    exact witness_row_13.2
  -- chapter_165_line_6: GL tag disintegration.
  have row_6 : (gl_implication18 N succ add) := by
    exact row_7.1.1.1.1.2
  -- chapter_165_line_5: GL tag expansion.
  have row_5 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((succ w4 w5) → (add w3 w2 w5))))))))) := by
    simpa only [gl_implication18] using row_6
  -- chapter_165_line_4: GL tag implication.
  have row_4 : (add v3 v5 v2) := by
    apply row_5
    exact row_19
    exact row_12
    exact row_22
    exact row_11
  -- chapter_165_line_3: GL tag expansion for integration.
  have row_3 : ((gl_preorder N add v3 v2) ↔ (¬ (∀ (v6 : α), ((N v6) → (¬ (add v3 v6 v2)))))) := by
    exact Iff.rfl
  -- chapter_165_line_2: GL tag reformulation for integration >[bound].
  have row_2 : (∀ (v4 : α), ((N v4) → ((add v3 v4 v2) → (gl_preorder N add v3 v2)))) := by
    intro v4
    intro integration_premise_1
    intro integration_premise_2
    apply (row_3).2
    intro universal_counterexample
    exact universal_counterexample v4 integration_premise_1 integration_premise_2
  -- chapter_165_line_1: GL tag implication.
  have row_1 : (gl_preorder N add v3 v2) := by
    apply row_2
    exact row_23
    exact row_4
  exact row_1

theorem gauss_source_087
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorGauss N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (gl_preorder N add v1 v2))) := by
  intro v1
  intro v2
  intro premise_1
  -- chapter_167_line_13: GL tag task formulation.
  have row_13 : (succ v1 v2) := by
    exact premise_1
  -- chapter_167_line_10: GL tag task formulation.
  have row_10 : (gl_AnchorGauss N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_167_line_9: GL tag expansion.
  have row_9 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorGauss] using row_10
  -- chapter_167_line_20: GL tag disintegration.
  have row_20 : (succ one two) := by
    exact row_9.1.2
  -- chapter_167_line_11: GL tag disintegration.
  have row_11 : (succ zero one) := by
    exact row_9.1.1.2
  -- chapter_167_line_8: GL tag disintegration.
  have row_8 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_9.1.1.1
  -- chapter_167_line_19: GL tag expansion.
  have row_19 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_8
  -- chapter_167_line_18: GL tag disintegration.
  have row_18 : (gl_fXY succ N N) := by
    exact row_19.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_167_line_17: GL tag expansion.
  have row_17 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_18
  -- chapter_167_line_16: GL tag disintegration.
  have row_16 : (gl_implication0 succ N) := by
    exact row_17.1.1.1
  -- chapter_167_line_15: GL tag expansion.
  have row_15 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_16
  -- chapter_167_line_14: GL tag implication.
  have row_14 : (N one) := by
    apply row_15
    exact row_20
  -- chapter_167_line_7: GL tag expansion for integration.
  have row_7 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_167_line_6: GL tag reformulation for integration and.
  have row_6 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_167_line_5: GL tag implication.
  have row_5 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_6
    exact row_8
    exact row_11
  have rule_row_12 := peano_source_033 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_167_line_12: GL tag implication.
  have row_12 : (add one v1 v2) := by
    apply rule_row_12
    exact row_13
  have rule_row_4 := peano_source_037 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_167_line_4: GL tag implication.
  have row_4 : (add v1 one v2) := by
    apply rule_row_4
    exact row_12
  -- chapter_167_line_3: GL tag expansion for integration.
  have row_3 : ((gl_preorder N add v1 v2) ↔ (¬ (∀ (v4 : α), ((N v4) → (¬ (add v1 v4 v2)))))) := by
    exact Iff.rfl
  -- chapter_167_line_2: GL tag reformulation for integration >[bound].
  have row_2 : (∀ (v3 : α), ((N v3) → ((add v1 v3 v2) → (gl_preorder N add v1 v2)))) := by
    intro v3
    intro integration_premise_1
    intro integration_premise_2
    apply (row_3).2
    intro universal_counterexample
    exact universal_counterexample v3 integration_premise_1 integration_premise_2
  -- chapter_167_line_1: GL tag implication.
  have row_1 : (gl_preorder N add v1 v2) := by
    apply row_2
    exact row_14
    exact row_4
  exact row_1

theorem gauss_source_088
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorGauss N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (V1 : GLSet α), ((gl_interval N add zero v1 V1) → (∀ (V2 : GLBinaryRelation α), ((gl_sequence N add zero v1 V2) → (gl_fXY V2 V1 N))))) := by
  intro v1
  intro V1
  intro premise_1
  intro V2
  intro premise_2
  -- chapter_168_line_48: GL tag expansion for integration.
  have row_48 : ((gl_implication5 V1 V2) ↔ (∀ (v10 : α), ((V1 v10) → (∀ (v7 : α), ((V2 v10 v7) → (∀ (v8 : α), ((V2 v10 v8) → (v7 = v8)))))))) := by
    exact Iff.rfl
  -- chapter_168_line_49: GL tag premise element.
  have row_49 : (∀ (v10 : α), ((V1 v10) → (∀ (v7 : α), ((V2 v10 v7) → (∀ (v8 : α), ((V2 v10 v8) → (V2 v10 v8))))))) := by
    intro v10
    intro scope_premise_1
    intro v7
    intro scope_premise_2
    intro v8
    intro scope_premise_3
    exact scope_premise_3
  -- chapter_168_line_47: GL tag premise element.
  have row_47 : (∀ (v10 : α), ((V1 v10) → (∀ (v7 : α), ((V2 v10 v7) → (∀ (v8 : α), ((V2 v10 v8) → (V2 v10 v7))))))) := by
    intro v10
    intro scope_premise_1
    intro v7
    intro scope_premise_2
    intro v8
    intro scope_premise_3
    exact scope_premise_2
  -- chapter_168_line_39: GL tag expansion for integration.
  have row_39 : ((gl_implication4 V1 N V2) ↔ (∀ (v6 : α), ((V1 v6) → (gl_existence0 N v6 V2)))) := by
    exact Iff.rfl
  -- chapter_168_line_38: GL tag premise element.
  have row_38 : (∀ (v6 : α), ((V1 v6) → (V1 v6))) := by
    intro v6
    intro scope_premise_1
    exact scope_premise_1
  -- chapter_168_line_23: GL tag expansion for integration.
  have row_23 : ((gl_implication0 V2 V1) ↔ (∀ (v2 : α) (v5 : α), ((V2 v2 v5) → (V1 v2)))) := by
    exact Iff.rfl
  -- chapter_168_line_22: GL tag premise element.
  have row_22 : (∀ (v2 : α) (v5 : α), ((V2 v2 v5) → (V2 v2 v5))) := by
    intro v2
    intro v5
    intro scope_premise_1
    exact scope_premise_1
  -- chapter_168_line_16: GL tag task formulation.
  have row_16 : (gl_sequence N add zero v1 V2) := by
    exact premise_2
  -- chapter_168_line_15: GL tag expansion.
  have row_15 : (¬ (∀ (V3 : GLSet α), ((gl_interval N add zero v1 V3) → (¬ (gl_fXY V2 V3 N))))) := by
    simpa only [gl_sequence] using row_16
  have exists_row_15 : ∃ (V3 : GLSet α), ((gl_interval N add zero v1 V3) ∧ (gl_fXY V2 V3 N)) := existsAndOfNotForallImpNot row_15
  obtain ⟨V3, witness_row_15⟩ := exists_row_15
  -- chapter_168_line_21: GL tag disintegration.
  have row_21 : (gl_fXY V2 V3 N) := by
    exact witness_row_15.2
  -- chapter_168_line_20: GL tag expansion.
  have row_20 : ((((gl_implication0 V2 V3) ∧ (gl_implication1 V2 N)) ∧ (gl_implication4 V3 N V2)) ∧ (gl_implication5 V3 V2)) := by
    simpa only [gl_fXY] using row_21
  -- chapter_168_line_46: GL tag disintegration.
  have row_46 : (gl_implication5 V3 V2) := by
    exact row_20.2
  -- chapter_168_line_45: GL tag expansion.
  have row_45 : (∀ (w1 : α), ((V3 w1) → (∀ (w2 : α), ((V2 w1 w2) → (∀ (w3 : α), ((V2 w1 w3) → (w2 = w3))))))) := by
    simpa only [gl_implication5] using row_46
  -- chapter_168_line_31: GL tag disintegration.
  have row_31 : (gl_implication4 V3 N V2) := by
    exact row_20.1.2
  -- chapter_168_line_30: GL tag expansion.
  have row_30 : (∀ (w1 : α), ((V3 w1) → (gl_existence0 N w1 V2))) := by
    simpa only [gl_implication4] using row_31
  -- chapter_168_line_27: GL tag disintegration.
  have row_27 : (gl_implication1 V2 N) := by
    exact row_20.1.1.2
  -- chapter_168_line_19: GL tag disintegration.
  have row_19 : (gl_implication0 V2 V3) := by
    exact row_20.1.1.1
  -- chapter_168_line_18: GL tag expansion.
  have row_18 : (∀ (w1 : α) (w2 : α), ((V2 w1 w2) → (V3 w1))) := by
    simpa only [gl_implication0] using row_19
  -- chapter_168_line_50: GL tag implication.
  have row_50 : (∀ (v10 : α), ((V1 v10) → (∀ (v7 : α), ((V2 v10 v7) → (∀ (v8 : α), ((V2 v10 v8) → (V3 v10))))))) := by
    intro v10
    intro scope_premise_1
    intro v7
    intro scope_premise_2
    intro v8
    intro scope_premise_3
    have scoped_fact_2 := row_47 v10 scope_premise_1 v7 scope_premise_2 v8 scope_premise_3
    apply row_18
    exact scoped_fact_2
  -- chapter_168_line_44: GL tag implication.
  have row_44 : (∀ (v10 : α), ((V1 v10) → (∀ (v7 : α), ((V2 v10 v7) → (∀ (v8 : α), ((V2 v10 v8) → (v7 = v8))))))) := by
    intro v10
    intro scope_premise_1
    intro v7
    intro scope_premise_2
    intro v8
    intro scope_premise_3
    have scoped_fact_2 := row_50 v10 scope_premise_1 v7 scope_premise_2 v8 scope_premise_3
    have scoped_fact_3 := row_47 v10 scope_premise_1 v7 scope_premise_2 v8 scope_premise_3
    have scoped_fact_4 := row_49 v10 scope_premise_1 v7 scope_premise_2 v8 scope_premise_3
    apply row_45
    exact scoped_fact_2
    exact scoped_fact_3
    exact scoped_fact_4
  -- chapter_168_line_43: GL tag validity name.
  have row_43 : (gl_implication5 V1 V2) := by
    simpa only [gl_implication5] using row_44
  -- chapter_168_line_17: GL tag implication.
  have row_17 : (∀ (v2 : α) (v5 : α), ((V2 v2 v5) → (V3 v2))) := by
    intro v2
    intro v5
    intro scope_premise_1
    have scoped_fact_2 := row_22 v2 v5 scope_premise_1
    apply row_18
    exact scoped_fact_2
  -- chapter_168_line_14: GL tag disintegration.
  have row_14 : (gl_interval N add zero v1 V3) := by
    exact witness_row_15.1
  -- chapter_168_line_13: GL tag expansion.
  have row_13 : (((((gl_implication26 V3 N add zero) ∧ (gl_implication27 V3 N add v1)) ∧ (gl_implication28 N add zero v1 V3)) ∧ (N zero)) ∧ (N v1)) := by
    simpa only [gl_interval] using row_14
  -- chapter_168_line_34: GL tag disintegration.
  have row_34 : (gl_implication28 N add zero v1 V3) := by
    exact row_13.1.1.2
  -- chapter_168_line_33: GL tag expansion.
  have row_33 : (∀ (w1 : α), ((gl_preorder N add zero w1) → ((gl_preorder N add w1 v1) → (V3 w1)))) := by
    simpa only [gl_implication28] using row_34
  -- chapter_168_line_26: GL tag disintegration.
  have row_26 : (gl_implication27 V3 N add v1) := by
    exact row_13.1.1.1.2
  -- chapter_168_line_25: GL tag expansion.
  have row_25 : (∀ (w1 : α), ((V3 w1) → (gl_preorder N add w1 v1))) := by
    simpa only [gl_implication27] using row_26
  -- chapter_168_line_24: GL tag implication.
  have row_24 : (∀ (v2 : α) (v5 : α), ((V2 v2 v5) → (gl_preorder N add v2 v1))) := by
    intro v2
    intro v5
    intro scope_premise_1
    have scoped_fact_2 := row_17 v2 v5 scope_premise_1
    apply row_25
    exact scoped_fact_2
  -- chapter_168_line_12: GL tag disintegration.
  have row_12 : (gl_implication26 V3 N add zero) := by
    exact row_13.1.1.1.1
  -- chapter_168_line_11: GL tag expansion.
  have row_11 : (∀ (w1 : α), ((V3 w1) → (gl_preorder N add zero w1))) := by
    simpa only [gl_implication26] using row_12
  -- chapter_168_line_10: GL tag implication.
  have row_10 : (∀ (v2 : α) (v5 : α), ((V2 v2 v5) → (gl_preorder N add zero v2))) := by
    intro v2
    intro v5
    intro scope_premise_1
    have scoped_fact_2 := row_17 v2 v5 scope_premise_1
    apply row_11
    exact scoped_fact_2
  -- chapter_168_line_9: GL tag task formulation.
  have row_9 : (gl_interval N add zero v1 V1) := by
    exact premise_1
  -- chapter_168_line_8: GL tag expansion.
  have row_8 : (((((gl_implication26 V1 N add zero) ∧ (gl_implication27 V1 N add v1)) ∧ (gl_implication28 N add zero v1 V1)) ∧ (N zero)) ∧ (N v1)) := by
    simpa only [gl_interval] using row_9
  -- chapter_168_line_42: GL tag disintegration.
  have row_42 : (gl_implication27 V1 N add v1) := by
    exact row_8.1.1.1.2
  -- chapter_168_line_41: GL tag expansion.
  have row_41 : (∀ (w1 : α), ((V1 w1) → (gl_preorder N add w1 v1))) := by
    simpa only [gl_implication27] using row_42
  -- chapter_168_line_40: GL tag implication.
  have row_40 : (∀ (v6 : α), ((V1 v6) → (gl_preorder N add v6 v1))) := by
    intro v6
    intro scope_premise_1
    have scoped_fact_2 := row_38 v6 scope_premise_1
    apply row_41
    exact scoped_fact_2
  -- chapter_168_line_37: GL tag disintegration.
  have row_37 : (gl_implication26 V1 N add zero) := by
    exact row_8.1.1.1.1
  -- chapter_168_line_36: GL tag expansion.
  have row_36 : (∀ (w1 : α), ((V1 w1) → (gl_preorder N add zero w1))) := by
    simpa only [gl_implication26] using row_37
  -- chapter_168_line_35: GL tag implication.
  have row_35 : (∀ (v6 : α), ((V1 v6) → (gl_preorder N add zero v6))) := by
    intro v6
    intro scope_premise_1
    have scoped_fact_2 := row_38 v6 scope_premise_1
    apply row_36
    exact scoped_fact_2
  -- chapter_168_line_32: GL tag implication.
  have row_32 : (∀ (v6 : α), ((V1 v6) → (V3 v6))) := by
    intro v6
    intro scope_premise_1
    have scoped_fact_2 := row_35 v6 scope_premise_1
    have scoped_fact_3 := row_40 v6 scope_premise_1
    apply row_33
    exact scoped_fact_2
    exact scoped_fact_3
  -- chapter_168_line_29: GL tag implication.
  have row_29 : (∀ (v6 : α), ((V1 v6) → (gl_existence0 N v6 V2))) := by
    intro v6
    intro scope_premise_1
    have scoped_fact_2 := row_32 v6 scope_premise_1
    apply row_30
    exact scoped_fact_2
  -- chapter_168_line_28: GL tag validity name.
  have row_28 : (gl_implication4 V1 N V2) := by
    simpa only [gl_implication4] using row_29
  -- chapter_168_line_7: GL tag disintegration.
  have row_7 : (gl_implication28 N add zero v1 V1) := by
    exact row_8.1.1.2
  -- chapter_168_line_6: GL tag expansion.
  have row_6 : (∀ (w1 : α), ((gl_preorder N add zero w1) → ((gl_preorder N add w1 v1) → (V1 w1)))) := by
    simpa only [gl_implication28] using row_7
  -- chapter_168_line_5: GL tag implication.
  have row_5 : (∀ (v2 : α) (v5 : α), ((V2 v2 v5) → (V1 v2))) := by
    intro v2
    intro v5
    intro scope_premise_1
    have scoped_fact_2 := row_10 v2 v5 scope_premise_1
    have scoped_fact_3 := row_24 v2 v5 scope_premise_1
    apply row_6
    exact scoped_fact_2
    exact scoped_fact_3
  -- chapter_168_line_4: GL tag validity name.
  have row_4 : (gl_implication0 V2 V1) := by
    simpa only [gl_implication0] using row_5
  -- chapter_168_line_3: GL tag expansion for integration.
  have row_3 : ((gl_fXY V2 V1 N) ↔ ((((gl_implication0 V2 V1) ∧ (gl_implication1 V2 N)) ∧ (gl_implication4 V1 N V2)) ∧ (gl_implication5 V1 V2))) := by
    exact Iff.rfl
  -- chapter_168_line_2: GL tag reformulation for integration and.
  have row_2 : ((gl_implication0 V2 V1) → ((gl_implication1 V2 N) → ((gl_implication4 V1 N V2) → ((gl_implication5 V1 V2) → (gl_fXY V2 V1 N))))) := by
    intro integration_premise_1
    intro integration_premise_2
    intro integration_premise_3
    intro integration_premise_4
    simp only [gl_fXY]
    exact ⟨⟨⟨integration_premise_1, integration_premise_2⟩, integration_premise_3⟩, integration_premise_4⟩
  -- chapter_168_line_1: GL tag implication.
  have row_1 : (gl_fXY V2 V1 N) := by
    apply row_2
    exact row_4
    exact row_27
    exact row_28
    exact row_43
  exact row_1

theorem gauss_source_089
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorGauss N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (V1 : GLSet α), ((gl_interval N add zero v1 V1) → ((gl_interval N add zero one V1) → (V1 v1)))) := by
  intro v1
  intro V1
  intro premise_1
  intro premise_2
  -- chapter_169_line_35: GL tag task formulation.
  have row_35 : (gl_interval N add zero one V1) := by
    exact premise_2
  -- chapter_169_line_34: GL tag expansion.
  have row_34 : (((((gl_implication26 V1 N add zero) ∧ (gl_implication27 V1 N add one)) ∧ (gl_implication28 N add zero one V1)) ∧ (N zero)) ∧ (N one)) := by
    simpa only [gl_interval] using row_35
  -- chapter_169_line_33: GL tag disintegration.
  have row_33 : (N zero) := by
    exact row_34.1.2
  -- chapter_169_line_30: GL tag expansion for integration.
  have row_30 : ((gl_preorder N add v1 v1) ↔ (¬ (∀ (v8 : α), ((N v8) → (¬ (add v1 v8 v1)))))) := by
    exact Iff.rfl
  -- chapter_169_line_29: GL tag reformulation for integration >[bound].
  have row_29 : (∀ (v7 : α), ((N v7) → ((add v1 v7 v1) → (gl_preorder N add v1 v1)))) := by
    intro v7
    intro integration_premise_1
    intro integration_premise_2
    apply (row_30).2
    intro universal_counterexample
    exact universal_counterexample v7 integration_premise_1 integration_premise_2
  -- chapter_169_line_24: GL tag variable copy.
  have row_24 : (v1 = v1) := by
    rfl
  -- chapter_169_line_23: GL tag symmetry of equality.
  have row_23 : (v1 = v1) := by
    exact Eq.symm row_24
  -- chapter_169_line_17: GL tag task formulation.
  have row_17 : (gl_AnchorGauss N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_169_line_16: GL tag expansion.
  have row_16 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorGauss] using row_17
  -- chapter_169_line_18: GL tag disintegration.
  have row_18 : (succ zero one) := by
    exact row_16.1.1.2
  -- chapter_169_line_15: GL tag disintegration.
  have row_15 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_16.1.1.1
  -- chapter_169_line_22: GL tag expansion.
  have row_22 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_15
  -- chapter_169_line_21: GL tag disintegration.
  have row_21 : (gl_implication16 N zero add) := by
    exact row_22.1.1.1.1.1.1.2
  -- chapter_169_line_20: GL tag expansion.
  have row_20 : (∀ (w1 : α) (w2 : α), ((w1 = w2) → ((N w1) → ((N w2) → (add w1 zero w2))))) := by
    simpa only [gl_implication16] using row_21
  -- chapter_169_line_14: GL tag expansion for integration.
  have row_14 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_169_line_13: GL tag reformulation for integration and.
  have row_13 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_169_line_12: GL tag implication.
  have row_12 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_13
    exact row_15
    exact row_18
  -- chapter_169_line_10: GL tag expansion for integration.
  have row_10 : ((gl_preorder N add zero v1) ↔ (¬ (∀ (v5 : α), ((N v5) → (¬ (add zero v5 v1)))))) := by
    exact Iff.rfl
  -- chapter_169_line_9: GL tag reformulation for integration >[bound].
  have row_9 : (∀ (v4 : α), ((N v4) → ((add zero v4 v1) → (gl_preorder N add zero v1)))) := by
    intro v4
    intro integration_premise_1
    intro integration_premise_2
    apply (row_10).2
    intro universal_counterexample
    exact universal_counterexample v4 integration_premise_1 integration_premise_2
  -- chapter_169_line_6: GL tag task formulation.
  have row_6 : (gl_interval N add zero v1 V1) := by
    exact premise_1
  -- chapter_169_line_5: GL tag expansion.
  have row_5 : (((((gl_implication26 V1 N add zero) ∧ (gl_implication27 V1 N add v1)) ∧ (gl_implication28 N add zero v1 V1)) ∧ (N zero)) ∧ (N v1)) := by
    simpa only [gl_interval] using row_6
  -- chapter_169_line_25: GL tag disintegration.
  have row_25 : (N v1) := by
    exact row_5.2
  -- chapter_169_line_26: GL tag equality1.
  have row_26 : (N v1) := by
    have equality_source := row_25
    have equality_step_1 := row_24
    cases equality_step_1
    exact equality_source
  -- chapter_169_line_32: GL tag implication.
  have row_32 : (add v1 zero v1) := by
    apply row_20
    exact row_24
    exact row_25
    exact row_26
  -- chapter_169_line_31: GL tag equality1.
  have row_31 : (add v1 zero v1) := by
    have equality_source := row_32
    have equality_step_1 := row_23
    cases equality_step_1
    exact equality_source
  -- chapter_169_line_28: GL tag implication.
  have row_28 : (gl_preorder N add v1 v1) := by
    apply row_29
    exact row_33
    exact row_31
  -- chapter_169_line_27: GL tag equality1.
  have row_27 : (gl_preorder N add v1 v1) := by
    have equality_source := row_28
    have equality_step_1 := row_24
    cases equality_step_1
    exact equality_source
  -- chapter_169_line_19: GL tag implication.
  have row_19 : (add v1 zero v1) := by
    apply row_20
    exact row_23
    exact row_26
    exact row_25
  have rule_row_11 := peano_source_040 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_169_line_11: GL tag implication.
  have row_11 : (add zero v1 v1) := by
    apply rule_row_11
    exact row_19
  -- chapter_169_line_8: GL tag implication.
  have row_8 : (gl_preorder N add zero v1) := by
    apply row_9
    exact row_26
    exact row_11
  -- chapter_169_line_7: GL tag equality1.
  have row_7 : (gl_preorder N add zero v1) := by
    have equality_source := row_8
    have equality_step_1 := row_24
    cases equality_step_1
    exact equality_source
  -- chapter_169_line_4: GL tag disintegration.
  have row_4 : (gl_implication28 N add zero v1 V1) := by
    exact row_5.1.1.2
  -- chapter_169_line_3: GL tag expansion.
  have row_3 : (∀ (w1 : α), ((gl_preorder N add zero w1) → ((gl_preorder N add w1 v1) → (V1 w1)))) := by
    simpa only [gl_implication28] using row_4
  -- chapter_169_line_2: GL tag implication.
  have row_2 : (V1 v1) := by
    apply row_3
    exact row_7
    exact row_27
  -- chapter_169_line_1: GL tag equality1.
  have row_1 : (V1 v1) := by
    have equality_source := row_2
    have equality_step_1 := row_23
    cases equality_step_1
    exact equality_source
  exact row_1

theorem gauss_source_090
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorGauss N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (V1 : GLSet α), ((gl_interval N add zero v1 V1) → (V1 v1))) := by
  intro v1
  intro V1
  intro premise_1
  -- chapter_170_line_30: GL tag expansion for integration.
  have row_30 : ((gl_preorder N add v1 v1) ↔ (¬ (∀ (v8 : α), ((N v8) → (¬ (add v1 v8 v1)))))) := by
    exact Iff.rfl
  -- chapter_170_line_29: GL tag reformulation for integration >[bound].
  have row_29 : (∀ (v7 : α), ((N v7) → ((add v1 v7 v1) → (gl_preorder N add v1 v1)))) := by
    intro v7
    intro integration_premise_1
    intro integration_premise_2
    apply (row_30).2
    intro universal_counterexample
    exact universal_counterexample v7 integration_premise_1 integration_premise_2
  -- chapter_170_line_24: GL tag variable copy.
  have row_24 : (v1 = v1) := by
    rfl
  -- chapter_170_line_23: GL tag symmetry of equality.
  have row_23 : (v1 = v1) := by
    exact Eq.symm row_24
  -- chapter_170_line_17: GL tag task formulation.
  have row_17 : (gl_AnchorGauss N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_170_line_16: GL tag expansion.
  have row_16 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorGauss] using row_17
  -- chapter_170_line_18: GL tag disintegration.
  have row_18 : (succ zero one) := by
    exact row_16.1.1.2
  -- chapter_170_line_15: GL tag disintegration.
  have row_15 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_16.1.1.1
  -- chapter_170_line_22: GL tag expansion.
  have row_22 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_15
  -- chapter_170_line_21: GL tag disintegration.
  have row_21 : (gl_implication16 N zero add) := by
    exact row_22.1.1.1.1.1.1.2
  -- chapter_170_line_20: GL tag expansion.
  have row_20 : (∀ (w1 : α) (w2 : α), ((w1 = w2) → ((N w1) → ((N w2) → (add w1 zero w2))))) := by
    simpa only [gl_implication16] using row_21
  -- chapter_170_line_14: GL tag expansion for integration.
  have row_14 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_170_line_13: GL tag reformulation for integration and.
  have row_13 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_170_line_12: GL tag implication.
  have row_12 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_13
    exact row_15
    exact row_18
  -- chapter_170_line_10: GL tag expansion for integration.
  have row_10 : ((gl_preorder N add zero v1) ↔ (¬ (∀ (v5 : α), ((N v5) → (¬ (add zero v5 v1)))))) := by
    exact Iff.rfl
  -- chapter_170_line_9: GL tag reformulation for integration >[bound].
  have row_9 : (∀ (v4 : α), ((N v4) → ((add zero v4 v1) → (gl_preorder N add zero v1)))) := by
    intro v4
    intro integration_premise_1
    intro integration_premise_2
    apply (row_10).2
    intro universal_counterexample
    exact universal_counterexample v4 integration_premise_1 integration_premise_2
  -- chapter_170_line_6: GL tag task formulation.
  have row_6 : (gl_interval N add zero v1 V1) := by
    exact premise_1
  -- chapter_170_line_5: GL tag expansion.
  have row_5 : (((((gl_implication26 V1 N add zero) ∧ (gl_implication27 V1 N add v1)) ∧ (gl_implication28 N add zero v1 V1)) ∧ (N zero)) ∧ (N v1)) := by
    simpa only [gl_interval] using row_6
  -- chapter_170_line_33: GL tag disintegration.
  have row_33 : (N zero) := by
    exact row_5.1.2
  -- chapter_170_line_25: GL tag disintegration.
  have row_25 : (N v1) := by
    exact row_5.2
  -- chapter_170_line_26: GL tag equality1.
  have row_26 : (N v1) := by
    have equality_source := row_25
    have equality_step_1 := row_24
    cases equality_step_1
    exact equality_source
  -- chapter_170_line_32: GL tag implication.
  have row_32 : (add v1 zero v1) := by
    apply row_20
    exact row_24
    exact row_25
    exact row_26
  -- chapter_170_line_31: GL tag equality1.
  have row_31 : (add v1 zero v1) := by
    have equality_source := row_32
    have equality_step_1 := row_23
    cases equality_step_1
    exact equality_source
  -- chapter_170_line_28: GL tag implication.
  have row_28 : (gl_preorder N add v1 v1) := by
    apply row_29
    exact row_33
    exact row_31
  -- chapter_170_line_27: GL tag equality1.
  have row_27 : (gl_preorder N add v1 v1) := by
    have equality_source := row_28
    have equality_step_1 := row_24
    cases equality_step_1
    exact equality_source
  -- chapter_170_line_19: GL tag implication.
  have row_19 : (add v1 zero v1) := by
    apply row_20
    exact row_23
    exact row_26
    exact row_25
  have rule_row_11 := peano_source_040 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_170_line_11: GL tag implication.
  have row_11 : (add zero v1 v1) := by
    apply rule_row_11
    exact row_19
  -- chapter_170_line_8: GL tag implication.
  have row_8 : (gl_preorder N add zero v1) := by
    apply row_9
    exact row_26
    exact row_11
  -- chapter_170_line_7: GL tag equality1.
  have row_7 : (gl_preorder N add zero v1) := by
    have equality_source := row_8
    have equality_step_1 := row_24
    cases equality_step_1
    exact equality_source
  -- chapter_170_line_4: GL tag disintegration.
  have row_4 : (gl_implication28 N add zero v1 V1) := by
    exact row_5.1.1.2
  -- chapter_170_line_3: GL tag expansion.
  have row_3 : (∀ (w1 : α), ((gl_preorder N add zero w1) → ((gl_preorder N add w1 v1) → (V1 w1)))) := by
    simpa only [gl_implication28] using row_4
  -- chapter_170_line_2: GL tag implication.
  have row_2 : (V1 v1) := by
    apply row_3
    exact row_7
    exact row_27
  -- chapter_170_line_1: GL tag equality1.
  have row_1 : (V1 v1) := by
    have equality_source := row_2
    have equality_step_1 := row_23
    cases equality_step_1
    exact equality_source
  exact row_1

theorem gauss_source_093
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorGauss N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (gl_AnchorPeano N zero succ add mul one) := by
  -- chapter_173_line_6: GL tag task formulation.
  have row_6 : (gl_AnchorGauss N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_173_line_5: GL tag expansion.
  have row_5 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorGauss] using row_6
  -- chapter_173_line_7: GL tag disintegration.
  have row_7 : (succ zero one) := by
    exact row_5.1.1.2
  -- chapter_173_line_4: GL tag disintegration.
  have row_4 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_5.1.1.1
  -- chapter_173_line_3: GL tag expansion for integration.
  have row_3 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_173_line_2: GL tag reformulation for integration and.
  have row_2 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_173_line_1: GL tag implication.
  have row_1 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_2
    exact row_4
    exact row_7
  exact row_1

theorem gauss_source_070
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorGauss N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (V1 : GLSet α) (V2 : GLSet α), ((gl_limitSet N add V1 v2 V2) → ((gl_interval N add zero v1 V1) → (¬ (gl_interval N add zero v2 V2))))))) := by
  intro v1
  intro v2
  intro premise_1
  intro V1
  intro V2
  intro premise_2
  intro premise_3
  intro reductio
  -- chapter_150_line_49: GL tag expansion for integration.
  have row_49 : ((gl_preorder N add v1 v2) ↔ (¬ (∀ (v11 : α), ((N v11) → (¬ (add v1 v11 v2)))))) := by
    exact Iff.rfl
  -- chapter_150_line_48: GL tag reformulation for integration >[bound].
  have row_48 : (∀ (v10 : α), ((N v10) → ((add v1 v10 v2) → (gl_preorder N add v1 v2)))) := by
    intro v10
    intro integration_premise_1
    intro integration_premise_2
    apply (row_49).2
    intro universal_counterexample
    exact universal_counterexample v10 integration_premise_1 integration_premise_2
  -- chapter_150_line_46: GL tag task formulation.
  have row_46 : (gl_interval N add zero v2 V2) := by
    exact reductio
  -- chapter_150_line_63: GL tag expansion.
  have row_63 : (((((gl_implication26 V2 N add zero) ∧ (gl_implication27 V2 N add v2)) ∧ (gl_implication28 N add zero v2 V2)) ∧ (N zero)) ∧ (N v2)) := by
    simpa only [gl_interval] using row_46
  -- chapter_150_line_62: GL tag disintegration.
  have row_62 : (N zero) := by
    exact row_63.1.2
  -- chapter_150_line_45: GL tag theorem.
  have row_45 := gauss_source_090 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_150_line_43: GL tag task formulation.
  have row_43 : (gl_limitSet N add V1 v2 V2) := by
    exact premise_2
  -- chapter_150_line_42: GL tag expansion.
  have row_42 : (((gl_implication41 V2 V1) ∧ (gl_implication27 V2 N add v2)) ∧ (gl_implication42 V1 N add v2 V2)) := by
    simpa only [gl_limitSet] using row_43
  -- chapter_150_line_41: GL tag disintegration.
  have row_41 : (gl_implication41 V2 V1) := by
    exact row_42.1.1
  -- chapter_150_line_40: GL tag expansion.
  have row_40 : (∀ (w1 : α), ((V2 w1) → (V1 w1))) := by
    simpa only [gl_implication41] using row_41
  -- chapter_150_line_38: GL tag task formulation.
  have row_38 : (gl_interval N add zero v1 V1) := by
    exact premise_3
  -- chapter_150_line_37: GL tag expansion.
  have row_37 : (((((gl_implication26 V1 N add zero) ∧ (gl_implication27 V1 N add v1)) ∧ (gl_implication28 N add zero v1 V1)) ∧ (N zero)) ∧ (N v1)) := by
    simpa only [gl_interval] using row_38
  -- chapter_150_line_64: GL tag disintegration.
  have row_64 : (N v1) := by
    exact row_37.2
  -- chapter_150_line_36: GL tag disintegration.
  have row_36 : (gl_implication27 V1 N add v1) := by
    exact row_37.1.1.1.2
  -- chapter_150_line_35: GL tag expansion.
  have row_35 : (∀ (w1 : α), ((V1 w1) → (gl_preorder N add w1 v1))) := by
    simpa only [gl_implication27] using row_36
  -- chapter_150_line_32: GL tag task formulation.
  have row_32 : (succ v1 v2) := by
    exact premise_1
  -- chapter_150_line_26: GL tag expansion for integration.
  have row_26 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_150_line_25: GL tag reformulation for integration and.
  have row_25 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_150_line_4: GL tag task formulation.
  have row_4 : (gl_AnchorGauss N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_150_line_44: GL tag implication.
  have row_44 : (V2 v2) := by
    apply row_45
    exact row_46
  -- chapter_150_line_39: GL tag implication.
  have row_39 : (V1 v2) := by
    apply row_40
    exact row_44
  -- chapter_150_line_34: GL tag implication.
  have row_34 : (gl_preorder N add v2 v1) := by
    apply row_35
    exact row_39
  -- chapter_150_line_3: GL tag expansion.
  have row_3 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorGauss] using row_4
  -- chapter_150_line_20: GL tag disintegration.
  have row_20 : (succ zero one) := by
    exact row_3.1.1.2
  -- chapter_150_line_10: GL tag disintegration.
  have row_10 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_3.1.1.1
  -- chapter_150_line_24: GL tag implication.
  have row_24 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_25
    exact row_10
    exact row_20
  have rule_row_51 := peano_source_033 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_150_line_51: GL tag implication.
  have row_51 : (add one v1 v2) := by
    apply rule_row_51
    exact row_32
  have rule_row_50 := peano_source_037 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_150_line_50: GL tag implication.
  have row_50 : (add v1 one v2) := by
    apply rule_row_50
    exact row_51
  have rule_row_23 := peano_source_063 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_150_line_23: GL tag implication.
  have row_23 : (gl_existence3 N one succ) := by
    apply rule_row_23
  -- chapter_150_line_22: GL tag expansion.
  have row_22 : (¬ (∀ (v5 : α), ((N v5) → (¬ (succ v5 one))))) := by
    simpa only [gl_existence3] using row_23
  have exists_row_22 : ∃ (v5 : α), ((N v5) ∧ (succ v5 one)) := existsAndOfNotForallImpNot row_22
  obtain ⟨v5, witness_row_22⟩ := exists_row_22
  -- chapter_150_line_21: GL tag disintegration.
  have row_21 : (succ v5 one) := by
    exact witness_row_22.2
  -- chapter_150_line_9: GL tag expansion.
  have row_9 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_10
  -- chapter_150_line_61: GL tag disintegration.
  have row_61 : (gl_fXYZ add N N N) := by
    exact row_9.1.1.1.1.1.1.1.1.2
  -- chapter_150_line_60: GL tag expansion.
  have row_60 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_61
  -- chapter_150_line_59: GL tag disintegration.
  have row_59 : (gl_implication13 N N N add) := by
    exact row_60.1.2
  -- chapter_150_line_58: GL tag expansion.
  have row_58 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication13] using row_59
  -- chapter_150_line_57: GL tag implication.
  have row_57 : (gl_existence1 N v1 zero add) := by
    apply row_58
    exact row_64
    exact row_62
  -- chapter_150_line_56: GL tag expansion.
  have row_56 : (¬ (∀ (v9 : α), ((N v9) → (¬ (add v1 zero v9))))) := by
    simpa only [gl_existence1] using row_57
  have exists_row_56 : ∃ (v9 : α), ((N v9) ∧ (add v1 zero v9)) := existsAndOfNotForallImpNot row_56
  obtain ⟨v9, witness_row_56⟩ := exists_row_56
  -- chapter_150_line_55: GL tag disintegration.
  have row_55 : (add v1 zero v9) := by
    exact witness_row_56.2
  -- chapter_150_line_54: GL tag disintegration.
  have row_54 : (gl_implication15 N zero add) := by
    exact row_9.1.1.1.1.1.1.1.2
  -- chapter_150_line_53: GL tag expansion.
  have row_53 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((add w1 zero w2) → (w1 = w2))))) := by
    simpa only [gl_implication15] using row_54
  -- chapter_150_line_52: GL tag implication.
  have row_52 : (v1 = v9) := by
    apply row_53
    exact row_64
    exact row_55
  -- chapter_150_line_72: GL tag symmetry of equality.
  have row_72 : (v9 = v1) := by
    exact Eq.symm row_52
  -- chapter_150_line_67: GL tag equality1.
  have row_67 : (gl_preorder N add v2 v9) := by
    have equality_source := row_34
    have equality_step_1 := row_52
    cases equality_step_1
    exact equality_source
  -- chapter_150_line_30: GL tag disintegration.
  have row_30 : (gl_implication18 N succ add) := by
    exact row_9.1.1.1.1.2
  -- chapter_150_line_29: GL tag expansion.
  have row_29 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((succ w4 w5) → (add w3 w2 w5))))))))) := by
    simpa only [gl_implication18] using row_30
  -- chapter_150_line_19: GL tag disintegration.
  have row_19 : (gl_implication7 N succ) := by
    exact row_9.1.1.1.1.1.1.1.1.1.2
  -- chapter_150_line_18: GL tag expansion.
  have row_18 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w2 w1) → (∀ (w3 : α), ((succ w3 w1) → (w2 = w3))))))) := by
    simpa only [gl_implication7] using row_19
  -- chapter_150_line_15: GL tag disintegration.
  have row_15 : (gl_fXY succ N N) := by
    exact row_9.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_150_line_14: GL tag expansion.
  have row_14 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_15
  -- chapter_150_line_13: GL tag disintegration.
  have row_13 : (gl_implication0 succ N) := by
    exact row_14.1.1.1
  -- chapter_150_line_12: GL tag expansion.
  have row_12 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_13
  -- chapter_150_line_8: GL tag disintegration.
  have row_8 : (gl_implication6 N zero succ) := by
    exact row_9.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_150_line_7: GL tag expansion.
  have row_7 : (∀ (w1 : α), ((N w1) → (¬ (succ w1 zero)))) := by
    simpa only [gl_implication6] using row_8
  -- chapter_150_line_2: GL tag disintegration.
  have row_2 : (succ one two) := by
    exact row_3.1.2
  -- chapter_150_line_11: GL tag implication.
  have row_11 : (N one) := by
    apply row_12
    exact row_2
  -- chapter_150_line_47: GL tag implication.
  have row_47 : (gl_preorder N add v1 v2) := by
    apply row_48
    exact row_11
    exact row_50
  have rule_row_69 := peano_source_052 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_150_line_69: GL tag implication.
  have row_69 : (v1 = v2) := by
    apply rule_row_69
    exact row_47
    exact row_34
  -- chapter_150_line_71: GL tag equality1.
  have row_71 : (add v2 zero v1) := by
    have equality_source := row_55
    have equality_step_1 := row_69
    cases equality_step_1
    have equality_step_2 := row_72
    cases equality_step_2
    exact equality_source
  -- chapter_150_line_68: GL tag equality1.
  have row_68 : (gl_preorder N add v9 v2) := by
    have equality_source := row_47
    have equality_step_1 := row_52
    cases equality_step_1
    exact equality_source
  have rule_row_66 := peano_source_052 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_150_line_66: GL tag implication.
  have row_66 : (v2 = v9) := by
    apply rule_row_66
    exact row_67
    exact row_68
  -- chapter_150_line_65: GL tag equality1.
  have row_65 : (add v2 one v9) := by
    have equality_source := row_50
    have equality_step_1 := row_66
    cases equality_step_1
    have equality_step_2 := row_69
    cases equality_step_2
    exact equality_source
  have rule_row_33 := peano_source_052 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_150_line_33: GL tag implication.
  have row_33 : (v2 = v1) := by
    apply rule_row_33
    exact row_34
    exact row_47
  -- chapter_150_line_31: GL tag equality1.
  have row_31 : (succ v9 v1) := by
    have equality_source := row_32
    have equality_step_1 := row_33
    cases equality_step_1
    have equality_step_2 := row_52
    cases equality_step_2
    exact equality_source
  -- chapter_150_line_28: GL tag implication.
  have row_28 : (add v2 two v1) := by
    apply row_29
    exact row_11
    exact row_2
    exact row_65
    exact row_31
  -- chapter_150_line_17: GL tag implication.
  have row_17 : (zero = v5) := by
    apply row_18
    exact row_11
    exact row_20
    exact row_21
  -- chapter_150_line_70: GL tag equality1.
  have row_70 : (add v2 v5 v1) := by
    have equality_source := row_71
    have equality_step_1 := row_17
    cases equality_step_1
    exact equality_source
  have rule_row_27 := peano_source_013 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_150_line_27: GL tag implication.
  have row_27 : (v5 = two) := by
    apply rule_row_27
    exact row_70
    exact row_28
  -- chapter_150_line_16: GL tag equality2.
  have row_16 : (zero = two) := by
    exact Eq.trans row_17 row_27
  -- chapter_150_line_6: GL tag implication.
  have row_6 : (¬ (succ one zero)) := by
    apply row_7
    exact row_11
  -- chapter_150_line_5: GL tag equality1.
  have row_5 : (¬ (succ one two)) := by
    have equality_source := row_6
    have equality_step_1 := row_16
    cases equality_step_1
    exact equality_source
  -- chapter_150_line_1: GL tag contradiction.
  have row_1 : (¬ (gl_interval N add zero v2 V2)) := by
    exact False.elim (row_5 row_2)
  exact row_1 reductio

theorem gauss_source_071
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorGauss N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (V1 : GLSet α) (V2 : GLSet α), ((gl_limitSet N add V1 zero V2) → ((gl_interval N add zero v1 V1) → (¬ (gl_interval N add zero v2 V2))))))) := by
  intro v1
  intro v2
  intro premise_1
  intro V1
  intro V2
  intro premise_2
  intro premise_3
  intro reductio
  -- chapter_151_line_40: GL tag task formulation.
  have row_40 : (gl_interval N add zero v2 V2) := by
    exact reductio
  -- chapter_151_line_45: GL tag expansion.
  have row_45 : (((((gl_implication26 V2 N add zero) ∧ (gl_implication27 V2 N add v2)) ∧ (gl_implication28 N add zero v2 V2)) ∧ (N zero)) ∧ (N v2)) := by
    simpa only [gl_interval] using row_40
  -- chapter_151_line_44: GL tag disintegration.
  have row_44 : (gl_implication26 V2 N add zero) := by
    exact row_45.1.1.1.1
  -- chapter_151_line_43: GL tag expansion.
  have row_43 : (∀ (w1 : α), ((V2 w1) → (gl_preorder N add zero w1))) := by
    simpa only [gl_implication26] using row_44
  -- chapter_151_line_39: GL tag theorem.
  have row_39 := gauss_source_090 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_151_line_37: GL tag task formulation.
  have row_37 : (gl_limitSet N add V1 zero V2) := by
    exact premise_2
  -- chapter_151_line_36: GL tag expansion.
  have row_36 : (((gl_implication41 V2 V1) ∧ (gl_implication27 V2 N add zero)) ∧ (gl_implication42 V1 N add zero V2)) := by
    simpa only [gl_limitSet] using row_37
  -- chapter_151_line_35: GL tag disintegration.
  have row_35 : (gl_implication27 V2 N add zero) := by
    exact row_36.1.2
  -- chapter_151_line_34: GL tag expansion.
  have row_34 : (∀ (w1 : α), ((V2 w1) → (gl_preorder N add w1 zero))) := by
    simpa only [gl_implication27] using row_35
  -- chapter_151_line_24: GL tag expansion for integration.
  have row_24 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_151_line_23: GL tag reformulation for integration and.
  have row_23 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_151_line_13: GL tag task formulation.
  have row_13 : (gl_interval N add zero v1 V1) := by
    exact premise_3
  -- chapter_151_line_12: GL tag expansion.
  have row_12 : (((((gl_implication26 V1 N add zero) ∧ (gl_implication27 V1 N add v1)) ∧ (gl_implication28 N add zero v1 V1)) ∧ (N zero)) ∧ (N v1)) := by
    simpa only [gl_interval] using row_13
  -- chapter_151_line_11: GL tag disintegration.
  have row_11 : (N v1) := by
    exact row_12.2
  -- chapter_151_line_10: GL tag task formulation.
  have row_10 : (gl_AnchorGauss N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_151_line_38: GL tag implication.
  have row_38 : (V2 v2) := by
    apply row_39
    exact row_40
  -- chapter_151_line_42: GL tag implication.
  have row_42 : (gl_preorder N add zero v2) := by
    apply row_43
    exact row_38
  -- chapter_151_line_33: GL tag implication.
  have row_33 : (gl_preorder N add v2 zero) := by
    apply row_34
    exact row_38
  -- chapter_151_line_9: GL tag expansion.
  have row_9 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorGauss] using row_10
  -- chapter_151_line_30: GL tag disintegration.
  have row_30 : (succ one two) := by
    exact row_9.1.2
  -- chapter_151_line_18: GL tag disintegration.
  have row_18 : (succ zero one) := by
    exact row_9.1.1.2
  -- chapter_151_line_8: GL tag disintegration.
  have row_8 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_9.1.1.1
  -- chapter_151_line_22: GL tag implication.
  have row_22 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_23
    exact row_8
    exact row_18
  have rule_row_21 := peano_source_063 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_151_line_21: GL tag implication.
  have row_21 : (gl_existence3 N one succ) := by
    apply rule_row_21
  -- chapter_151_line_20: GL tag expansion.
  have row_20 : (¬ (∀ (v4 : α), ((N v4) → (¬ (succ v4 one))))) := by
    simpa only [gl_existence3] using row_21
  have exists_row_20 : ∃ (v4 : α), ((N v4) ∧ (succ v4 one)) := existsAndOfNotForallImpNot row_20
  obtain ⟨v4, witness_row_20⟩ := exists_row_20
  -- chapter_151_line_19: GL tag disintegration.
  have row_19 : (succ v4 one) := by
    exact witness_row_20.2
  -- chapter_151_line_7: GL tag expansion.
  have row_7 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_8
  -- chapter_151_line_29: GL tag disintegration.
  have row_29 : (gl_fXY succ N N) := by
    exact row_7.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_151_line_28: GL tag expansion.
  have row_28 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_29
  -- chapter_151_line_27: GL tag disintegration.
  have row_27 : (gl_implication0 succ N) := by
    exact row_28.1.1.1
  -- chapter_151_line_26: GL tag expansion.
  have row_26 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_27
  -- chapter_151_line_25: GL tag implication.
  have row_25 : (N one) := by
    apply row_26
    exact row_30
  -- chapter_151_line_17: GL tag disintegration.
  have row_17 : (gl_implication7 N succ) := by
    exact row_7.1.1.1.1.1.1.1.1.1.2
  -- chapter_151_line_16: GL tag expansion.
  have row_16 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w2 w1) → (∀ (w3 : α), ((succ w3 w1) → (w2 = w3))))))) := by
    simpa only [gl_implication7] using row_17
  -- chapter_151_line_15: GL tag implication.
  have row_15 : (zero = v4) := by
    apply row_16
    exact row_25
    exact row_18
    exact row_19
  -- chapter_151_line_41: GL tag equality1.
  have row_41 : (gl_preorder N add v4 v2) := by
    have equality_source := row_42
    have equality_step_1 := row_15
    cases equality_step_1
    exact equality_source
  -- chapter_151_line_32: GL tag equality1.
  have row_32 : (gl_preorder N add v2 v4) := by
    have equality_source := row_33
    have equality_step_1 := row_15
    cases equality_step_1
    exact equality_source
  have rule_row_31 := peano_source_052 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_151_line_31: GL tag implication.
  have row_31 : (v4 = v2) := by
    apply rule_row_31
    exact row_41
    exact row_32
  -- chapter_151_line_14: GL tag equality2.
  have row_14 : (zero = v2) := by
    exact Eq.trans row_15 row_31
  -- chapter_151_line_6: GL tag disintegration.
  have row_6 : (gl_implication6 N zero succ) := by
    exact row_7.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_151_line_5: GL tag expansion.
  have row_5 : (∀ (w1 : α), ((N w1) → (¬ (succ w1 zero)))) := by
    simpa only [gl_implication6] using row_6
  -- chapter_151_line_4: GL tag implication.
  have row_4 : (¬ (succ v1 zero)) := by
    apply row_5
    exact row_11
  -- chapter_151_line_3: GL tag equality1.
  have row_3 : (¬ (succ v1 v2)) := by
    have equality_source := row_4
    have equality_step_1 := row_14
    cases equality_step_1
    exact equality_source
  -- chapter_151_line_2: GL tag task formulation.
  have row_2 : (succ v1 v2) := by
    exact premise_1
  -- chapter_151_line_1: GL tag contradiction.
  have row_1 : (¬ (gl_interval N add zero v2 V2)) := by
    exact False.elim (row_3 row_2)
  exact row_1 reductio

theorem gauss_source_072
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorGauss N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (V1 : GLSet α) (V2 : GLSet α), ((gl_limitSet N add V1 one V2) → ((gl_interval N add zero v1 V1) → (¬ (gl_interval N add zero v2 V2))))))) := by
  intro v1
  intro v2
  intro premise_1
  intro V1
  intro V2
  intro premise_2
  intro premise_3
  intro reductio
  -- chapter_152_line_49: GL tag expansion for integration.
  have row_49 : ((gl_preorder N add v1 v2) ↔ (¬ (∀ (v11 : α), ((N v11) → (¬ (add v1 v11 v2)))))) := by
    exact Iff.rfl
  -- chapter_152_line_48: GL tag reformulation for integration >[bound].
  have row_48 : (∀ (v10 : α), ((N v10) → ((add v1 v10 v2) → (gl_preorder N add v1 v2)))) := by
    intro v10
    intro integration_premise_1
    intro integration_premise_2
    apply (row_49).2
    intro universal_counterexample
    exact universal_counterexample v10 integration_premise_1 integration_premise_2
  -- chapter_152_line_46: GL tag task formulation.
  have row_46 : (gl_interval N add zero v2 V2) := by
    exact reductio
  -- chapter_152_line_63: GL tag expansion.
  have row_63 : (((((gl_implication26 V2 N add zero) ∧ (gl_implication27 V2 N add v2)) ∧ (gl_implication28 N add zero v2 V2)) ∧ (N zero)) ∧ (N v2)) := by
    simpa only [gl_interval] using row_46
  -- chapter_152_line_62: GL tag disintegration.
  have row_62 : (N zero) := by
    exact row_63.1.2
  -- chapter_152_line_45: GL tag theorem.
  have row_45 := gauss_source_090 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_152_line_43: GL tag task formulation.
  have row_43 : (gl_limitSet N add V1 one V2) := by
    exact premise_2
  -- chapter_152_line_42: GL tag expansion.
  have row_42 : (((gl_implication41 V2 V1) ∧ (gl_implication27 V2 N add one)) ∧ (gl_implication42 V1 N add one V2)) := by
    simpa only [gl_limitSet] using row_43
  -- chapter_152_line_41: GL tag disintegration.
  have row_41 : (gl_implication41 V2 V1) := by
    exact row_42.1.1
  -- chapter_152_line_40: GL tag expansion.
  have row_40 : (∀ (w1 : α), ((V2 w1) → (V1 w1))) := by
    simpa only [gl_implication41] using row_41
  -- chapter_152_line_38: GL tag task formulation.
  have row_38 : (gl_interval N add zero v1 V1) := by
    exact premise_3
  -- chapter_152_line_37: GL tag expansion.
  have row_37 : (((((gl_implication26 V1 N add zero) ∧ (gl_implication27 V1 N add v1)) ∧ (gl_implication28 N add zero v1 V1)) ∧ (N zero)) ∧ (N v1)) := by
    simpa only [gl_interval] using row_38
  -- chapter_152_line_64: GL tag disintegration.
  have row_64 : (N v1) := by
    exact row_37.2
  -- chapter_152_line_36: GL tag disintegration.
  have row_36 : (gl_implication27 V1 N add v1) := by
    exact row_37.1.1.1.2
  -- chapter_152_line_35: GL tag expansion.
  have row_35 : (∀ (w1 : α), ((V1 w1) → (gl_preorder N add w1 v1))) := by
    simpa only [gl_implication27] using row_36
  -- chapter_152_line_32: GL tag task formulation.
  have row_32 : (succ v1 v2) := by
    exact premise_1
  -- chapter_152_line_26: GL tag expansion for integration.
  have row_26 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_152_line_25: GL tag reformulation for integration and.
  have row_25 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_152_line_4: GL tag task formulation.
  have row_4 : (gl_AnchorGauss N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_152_line_44: GL tag implication.
  have row_44 : (V2 v2) := by
    apply row_45
    exact row_46
  -- chapter_152_line_39: GL tag implication.
  have row_39 : (V1 v2) := by
    apply row_40
    exact row_44
  -- chapter_152_line_34: GL tag implication.
  have row_34 : (gl_preorder N add v2 v1) := by
    apply row_35
    exact row_39
  -- chapter_152_line_3: GL tag expansion.
  have row_3 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorGauss] using row_4
  -- chapter_152_line_20: GL tag disintegration.
  have row_20 : (succ zero one) := by
    exact row_3.1.1.2
  -- chapter_152_line_10: GL tag disintegration.
  have row_10 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_3.1.1.1
  -- chapter_152_line_24: GL tag implication.
  have row_24 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_25
    exact row_10
    exact row_20
  have rule_row_51 := peano_source_033 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_152_line_51: GL tag implication.
  have row_51 : (add one v1 v2) := by
    apply rule_row_51
    exact row_32
  have rule_row_50 := peano_source_037 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_152_line_50: GL tag implication.
  have row_50 : (add v1 one v2) := by
    apply rule_row_50
    exact row_51
  have rule_row_23 := peano_source_063 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_152_line_23: GL tag implication.
  have row_23 : (gl_existence3 N one succ) := by
    apply rule_row_23
  -- chapter_152_line_22: GL tag expansion.
  have row_22 : (¬ (∀ (v5 : α), ((N v5) → (¬ (succ v5 one))))) := by
    simpa only [gl_existence3] using row_23
  have exists_row_22 : ∃ (v5 : α), ((N v5) ∧ (succ v5 one)) := existsAndOfNotForallImpNot row_22
  obtain ⟨v5, witness_row_22⟩ := exists_row_22
  -- chapter_152_line_21: GL tag disintegration.
  have row_21 : (succ v5 one) := by
    exact witness_row_22.2
  -- chapter_152_line_9: GL tag expansion.
  have row_9 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_10
  -- chapter_152_line_61: GL tag disintegration.
  have row_61 : (gl_fXYZ add N N N) := by
    exact row_9.1.1.1.1.1.1.1.1.2
  -- chapter_152_line_60: GL tag expansion.
  have row_60 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_61
  -- chapter_152_line_59: GL tag disintegration.
  have row_59 : (gl_implication13 N N N add) := by
    exact row_60.1.2
  -- chapter_152_line_58: GL tag expansion.
  have row_58 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication13] using row_59
  -- chapter_152_line_57: GL tag implication.
  have row_57 : (gl_existence1 N v1 zero add) := by
    apply row_58
    exact row_64
    exact row_62
  -- chapter_152_line_56: GL tag expansion.
  have row_56 : (¬ (∀ (v9 : α), ((N v9) → (¬ (add v1 zero v9))))) := by
    simpa only [gl_existence1] using row_57
  have exists_row_56 : ∃ (v9 : α), ((N v9) ∧ (add v1 zero v9)) := existsAndOfNotForallImpNot row_56
  obtain ⟨v9, witness_row_56⟩ := exists_row_56
  -- chapter_152_line_55: GL tag disintegration.
  have row_55 : (add v1 zero v9) := by
    exact witness_row_56.2
  -- chapter_152_line_54: GL tag disintegration.
  have row_54 : (gl_implication15 N zero add) := by
    exact row_9.1.1.1.1.1.1.1.2
  -- chapter_152_line_53: GL tag expansion.
  have row_53 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((add w1 zero w2) → (w1 = w2))))) := by
    simpa only [gl_implication15] using row_54
  -- chapter_152_line_52: GL tag implication.
  have row_52 : (v1 = v9) := by
    apply row_53
    exact row_64
    exact row_55
  -- chapter_152_line_72: GL tag symmetry of equality.
  have row_72 : (v9 = v1) := by
    exact Eq.symm row_52
  -- chapter_152_line_67: GL tag equality1.
  have row_67 : (gl_preorder N add v2 v9) := by
    have equality_source := row_34
    have equality_step_1 := row_52
    cases equality_step_1
    exact equality_source
  -- chapter_152_line_30: GL tag disintegration.
  have row_30 : (gl_implication18 N succ add) := by
    exact row_9.1.1.1.1.2
  -- chapter_152_line_29: GL tag expansion.
  have row_29 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((succ w4 w5) → (add w3 w2 w5))))))))) := by
    simpa only [gl_implication18] using row_30
  -- chapter_152_line_19: GL tag disintegration.
  have row_19 : (gl_implication7 N succ) := by
    exact row_9.1.1.1.1.1.1.1.1.1.2
  -- chapter_152_line_18: GL tag expansion.
  have row_18 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w2 w1) → (∀ (w3 : α), ((succ w3 w1) → (w2 = w3))))))) := by
    simpa only [gl_implication7] using row_19
  -- chapter_152_line_15: GL tag disintegration.
  have row_15 : (gl_fXY succ N N) := by
    exact row_9.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_152_line_14: GL tag expansion.
  have row_14 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_15
  -- chapter_152_line_13: GL tag disintegration.
  have row_13 : (gl_implication0 succ N) := by
    exact row_14.1.1.1
  -- chapter_152_line_12: GL tag expansion.
  have row_12 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_13
  -- chapter_152_line_8: GL tag disintegration.
  have row_8 : (gl_implication6 N zero succ) := by
    exact row_9.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_152_line_7: GL tag expansion.
  have row_7 : (∀ (w1 : α), ((N w1) → (¬ (succ w1 zero)))) := by
    simpa only [gl_implication6] using row_8
  -- chapter_152_line_2: GL tag disintegration.
  have row_2 : (succ one two) := by
    exact row_3.1.2
  -- chapter_152_line_11: GL tag implication.
  have row_11 : (N one) := by
    apply row_12
    exact row_2
  -- chapter_152_line_47: GL tag implication.
  have row_47 : (gl_preorder N add v1 v2) := by
    apply row_48
    exact row_11
    exact row_50
  have rule_row_69 := peano_source_052 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_152_line_69: GL tag implication.
  have row_69 : (v1 = v2) := by
    apply rule_row_69
    exact row_47
    exact row_34
  -- chapter_152_line_71: GL tag equality1.
  have row_71 : (add v2 zero v1) := by
    have equality_source := row_55
    have equality_step_1 := row_69
    cases equality_step_1
    have equality_step_2 := row_72
    cases equality_step_2
    exact equality_source
  -- chapter_152_line_68: GL tag equality1.
  have row_68 : (gl_preorder N add v9 v2) := by
    have equality_source := row_47
    have equality_step_1 := row_52
    cases equality_step_1
    exact equality_source
  have rule_row_66 := peano_source_052 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_152_line_66: GL tag implication.
  have row_66 : (v2 = v9) := by
    apply rule_row_66
    exact row_67
    exact row_68
  -- chapter_152_line_65: GL tag equality1.
  have row_65 : (add v2 one v9) := by
    have equality_source := row_50
    have equality_step_1 := row_66
    cases equality_step_1
    have equality_step_2 := row_69
    cases equality_step_2
    exact equality_source
  have rule_row_33 := peano_source_052 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_152_line_33: GL tag implication.
  have row_33 : (v2 = v1) := by
    apply rule_row_33
    exact row_34
    exact row_47
  -- chapter_152_line_31: GL tag equality1.
  have row_31 : (succ v9 v1) := by
    have equality_source := row_32
    have equality_step_1 := row_33
    cases equality_step_1
    have equality_step_2 := row_52
    cases equality_step_2
    exact equality_source
  -- chapter_152_line_28: GL tag implication.
  have row_28 : (add v2 two v1) := by
    apply row_29
    exact row_11
    exact row_2
    exact row_65
    exact row_31
  -- chapter_152_line_17: GL tag implication.
  have row_17 : (zero = v5) := by
    apply row_18
    exact row_11
    exact row_20
    exact row_21
  -- chapter_152_line_70: GL tag equality1.
  have row_70 : (add v2 v5 v1) := by
    have equality_source := row_71
    have equality_step_1 := row_17
    cases equality_step_1
    exact equality_source
  have rule_row_27 := peano_source_013 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_152_line_27: GL tag implication.
  have row_27 : (v5 = two) := by
    apply rule_row_27
    exact row_70
    exact row_28
  -- chapter_152_line_16: GL tag equality2.
  have row_16 : (zero = two) := by
    exact Eq.trans row_17 row_27
  -- chapter_152_line_6: GL tag implication.
  have row_6 : (¬ (succ one zero)) := by
    apply row_7
    exact row_11
  -- chapter_152_line_5: GL tag equality1.
  have row_5 : (¬ (succ one two)) := by
    have equality_source := row_6
    have equality_step_1 := row_16
    cases equality_step_1
    exact equality_source
  -- chapter_152_line_1: GL tag contradiction.
  have row_1 : (¬ (gl_interval N add zero v2 V2)) := by
    exact False.elim (row_5 row_2)
  exact row_1 reductio

theorem gauss_source_073
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorGauss N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (V1 : GLSet α) (V2 : GLSet α), ((gl_limitSet N add V1 two V2) → ((gl_interval N add zero v1 V1) → (¬ (gl_interval N add zero v2 V2))))))) := by
  intro v1
  intro v2
  intro premise_1
  intro V1
  intro V2
  intro premise_2
  intro premise_3
  intro reductio
  -- chapter_153_line_49: GL tag expansion for integration.
  have row_49 : ((gl_preorder N add v1 v2) ↔ (¬ (∀ (v11 : α), ((N v11) → (¬ (add v1 v11 v2)))))) := by
    exact Iff.rfl
  -- chapter_153_line_48: GL tag reformulation for integration >[bound].
  have row_48 : (∀ (v10 : α), ((N v10) → ((add v1 v10 v2) → (gl_preorder N add v1 v2)))) := by
    intro v10
    intro integration_premise_1
    intro integration_premise_2
    apply (row_49).2
    intro universal_counterexample
    exact universal_counterexample v10 integration_premise_1 integration_premise_2
  -- chapter_153_line_46: GL tag task formulation.
  have row_46 : (gl_interval N add zero v2 V2) := by
    exact reductio
  -- chapter_153_line_63: GL tag expansion.
  have row_63 : (((((gl_implication26 V2 N add zero) ∧ (gl_implication27 V2 N add v2)) ∧ (gl_implication28 N add zero v2 V2)) ∧ (N zero)) ∧ (N v2)) := by
    simpa only [gl_interval] using row_46
  -- chapter_153_line_62: GL tag disintegration.
  have row_62 : (N zero) := by
    exact row_63.1.2
  -- chapter_153_line_45: GL tag theorem.
  have row_45 := gauss_source_090 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_153_line_43: GL tag task formulation.
  have row_43 : (gl_limitSet N add V1 two V2) := by
    exact premise_2
  -- chapter_153_line_42: GL tag expansion.
  have row_42 : (((gl_implication41 V2 V1) ∧ (gl_implication27 V2 N add two)) ∧ (gl_implication42 V1 N add two V2)) := by
    simpa only [gl_limitSet] using row_43
  -- chapter_153_line_41: GL tag disintegration.
  have row_41 : (gl_implication41 V2 V1) := by
    exact row_42.1.1
  -- chapter_153_line_40: GL tag expansion.
  have row_40 : (∀ (w1 : α), ((V2 w1) → (V1 w1))) := by
    simpa only [gl_implication41] using row_41
  -- chapter_153_line_38: GL tag task formulation.
  have row_38 : (gl_interval N add zero v1 V1) := by
    exact premise_3
  -- chapter_153_line_37: GL tag expansion.
  have row_37 : (((((gl_implication26 V1 N add zero) ∧ (gl_implication27 V1 N add v1)) ∧ (gl_implication28 N add zero v1 V1)) ∧ (N zero)) ∧ (N v1)) := by
    simpa only [gl_interval] using row_38
  -- chapter_153_line_64: GL tag disintegration.
  have row_64 : (N v1) := by
    exact row_37.2
  -- chapter_153_line_36: GL tag disintegration.
  have row_36 : (gl_implication27 V1 N add v1) := by
    exact row_37.1.1.1.2
  -- chapter_153_line_35: GL tag expansion.
  have row_35 : (∀ (w1 : α), ((V1 w1) → (gl_preorder N add w1 v1))) := by
    simpa only [gl_implication27] using row_36
  -- chapter_153_line_32: GL tag task formulation.
  have row_32 : (succ v1 v2) := by
    exact premise_1
  -- chapter_153_line_26: GL tag expansion for integration.
  have row_26 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_153_line_25: GL tag reformulation for integration and.
  have row_25 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_153_line_4: GL tag task formulation.
  have row_4 : (gl_AnchorGauss N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_153_line_44: GL tag implication.
  have row_44 : (V2 v2) := by
    apply row_45
    exact row_46
  -- chapter_153_line_39: GL tag implication.
  have row_39 : (V1 v2) := by
    apply row_40
    exact row_44
  -- chapter_153_line_34: GL tag implication.
  have row_34 : (gl_preorder N add v2 v1) := by
    apply row_35
    exact row_39
  -- chapter_153_line_3: GL tag expansion.
  have row_3 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorGauss] using row_4
  -- chapter_153_line_20: GL tag disintegration.
  have row_20 : (succ zero one) := by
    exact row_3.1.1.2
  -- chapter_153_line_10: GL tag disintegration.
  have row_10 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_3.1.1.1
  -- chapter_153_line_24: GL tag implication.
  have row_24 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_25
    exact row_10
    exact row_20
  have rule_row_51 := peano_source_033 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_153_line_51: GL tag implication.
  have row_51 : (add one v1 v2) := by
    apply rule_row_51
    exact row_32
  have rule_row_50 := peano_source_037 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_153_line_50: GL tag implication.
  have row_50 : (add v1 one v2) := by
    apply rule_row_50
    exact row_51
  have rule_row_23 := peano_source_063 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_153_line_23: GL tag implication.
  have row_23 : (gl_existence3 N one succ) := by
    apply rule_row_23
  -- chapter_153_line_22: GL tag expansion.
  have row_22 : (¬ (∀ (v5 : α), ((N v5) → (¬ (succ v5 one))))) := by
    simpa only [gl_existence3] using row_23
  have exists_row_22 : ∃ (v5 : α), ((N v5) ∧ (succ v5 one)) := existsAndOfNotForallImpNot row_22
  obtain ⟨v5, witness_row_22⟩ := exists_row_22
  -- chapter_153_line_21: GL tag disintegration.
  have row_21 : (succ v5 one) := by
    exact witness_row_22.2
  -- chapter_153_line_9: GL tag expansion.
  have row_9 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_10
  -- chapter_153_line_61: GL tag disintegration.
  have row_61 : (gl_fXYZ add N N N) := by
    exact row_9.1.1.1.1.1.1.1.1.2
  -- chapter_153_line_60: GL tag expansion.
  have row_60 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_61
  -- chapter_153_line_59: GL tag disintegration.
  have row_59 : (gl_implication13 N N N add) := by
    exact row_60.1.2
  -- chapter_153_line_58: GL tag expansion.
  have row_58 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication13] using row_59
  -- chapter_153_line_57: GL tag implication.
  have row_57 : (gl_existence1 N v1 zero add) := by
    apply row_58
    exact row_64
    exact row_62
  -- chapter_153_line_56: GL tag expansion.
  have row_56 : (¬ (∀ (v9 : α), ((N v9) → (¬ (add v1 zero v9))))) := by
    simpa only [gl_existence1] using row_57
  have exists_row_56 : ∃ (v9 : α), ((N v9) ∧ (add v1 zero v9)) := existsAndOfNotForallImpNot row_56
  obtain ⟨v9, witness_row_56⟩ := exists_row_56
  -- chapter_153_line_55: GL tag disintegration.
  have row_55 : (add v1 zero v9) := by
    exact witness_row_56.2
  -- chapter_153_line_54: GL tag disintegration.
  have row_54 : (gl_implication15 N zero add) := by
    exact row_9.1.1.1.1.1.1.1.2
  -- chapter_153_line_53: GL tag expansion.
  have row_53 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((add w1 zero w2) → (w1 = w2))))) := by
    simpa only [gl_implication15] using row_54
  -- chapter_153_line_52: GL tag implication.
  have row_52 : (v1 = v9) := by
    apply row_53
    exact row_64
    exact row_55
  -- chapter_153_line_72: GL tag symmetry of equality.
  have row_72 : (v9 = v1) := by
    exact Eq.symm row_52
  -- chapter_153_line_67: GL tag equality1.
  have row_67 : (gl_preorder N add v2 v9) := by
    have equality_source := row_34
    have equality_step_1 := row_52
    cases equality_step_1
    exact equality_source
  -- chapter_153_line_30: GL tag disintegration.
  have row_30 : (gl_implication18 N succ add) := by
    exact row_9.1.1.1.1.2
  -- chapter_153_line_29: GL tag expansion.
  have row_29 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((succ w4 w5) → (add w3 w2 w5))))))))) := by
    simpa only [gl_implication18] using row_30
  -- chapter_153_line_19: GL tag disintegration.
  have row_19 : (gl_implication7 N succ) := by
    exact row_9.1.1.1.1.1.1.1.1.1.2
  -- chapter_153_line_18: GL tag expansion.
  have row_18 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w2 w1) → (∀ (w3 : α), ((succ w3 w1) → (w2 = w3))))))) := by
    simpa only [gl_implication7] using row_19
  -- chapter_153_line_15: GL tag disintegration.
  have row_15 : (gl_fXY succ N N) := by
    exact row_9.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_153_line_14: GL tag expansion.
  have row_14 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_15
  -- chapter_153_line_13: GL tag disintegration.
  have row_13 : (gl_implication0 succ N) := by
    exact row_14.1.1.1
  -- chapter_153_line_12: GL tag expansion.
  have row_12 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_13
  -- chapter_153_line_8: GL tag disintegration.
  have row_8 : (gl_implication6 N zero succ) := by
    exact row_9.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_153_line_7: GL tag expansion.
  have row_7 : (∀ (w1 : α), ((N w1) → (¬ (succ w1 zero)))) := by
    simpa only [gl_implication6] using row_8
  -- chapter_153_line_2: GL tag disintegration.
  have row_2 : (succ one two) := by
    exact row_3.1.2
  -- chapter_153_line_11: GL tag implication.
  have row_11 : (N one) := by
    apply row_12
    exact row_2
  -- chapter_153_line_47: GL tag implication.
  have row_47 : (gl_preorder N add v1 v2) := by
    apply row_48
    exact row_11
    exact row_50
  have rule_row_69 := peano_source_052 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_153_line_69: GL tag implication.
  have row_69 : (v1 = v2) := by
    apply rule_row_69
    exact row_47
    exact row_34
  -- chapter_153_line_71: GL tag equality1.
  have row_71 : (add v2 zero v1) := by
    have equality_source := row_55
    have equality_step_1 := row_69
    cases equality_step_1
    have equality_step_2 := row_72
    cases equality_step_2
    exact equality_source
  -- chapter_153_line_68: GL tag equality1.
  have row_68 : (gl_preorder N add v9 v2) := by
    have equality_source := row_47
    have equality_step_1 := row_52
    cases equality_step_1
    exact equality_source
  have rule_row_66 := peano_source_052 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_153_line_66: GL tag implication.
  have row_66 : (v2 = v9) := by
    apply rule_row_66
    exact row_67
    exact row_68
  -- chapter_153_line_65: GL tag equality1.
  have row_65 : (add v2 one v9) := by
    have equality_source := row_50
    have equality_step_1 := row_66
    cases equality_step_1
    have equality_step_2 := row_69
    cases equality_step_2
    exact equality_source
  have rule_row_33 := peano_source_052 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_153_line_33: GL tag implication.
  have row_33 : (v2 = v1) := by
    apply rule_row_33
    exact row_34
    exact row_47
  -- chapter_153_line_31: GL tag equality1.
  have row_31 : (succ v9 v1) := by
    have equality_source := row_32
    have equality_step_1 := row_33
    cases equality_step_1
    have equality_step_2 := row_52
    cases equality_step_2
    exact equality_source
  -- chapter_153_line_28: GL tag implication.
  have row_28 : (add v2 two v1) := by
    apply row_29
    exact row_11
    exact row_2
    exact row_65
    exact row_31
  -- chapter_153_line_17: GL tag implication.
  have row_17 : (zero = v5) := by
    apply row_18
    exact row_11
    exact row_20
    exact row_21
  -- chapter_153_line_70: GL tag equality1.
  have row_70 : (add v2 v5 v1) := by
    have equality_source := row_71
    have equality_step_1 := row_17
    cases equality_step_1
    exact equality_source
  have rule_row_27 := peano_source_013 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_153_line_27: GL tag implication.
  have row_27 : (v5 = two) := by
    apply rule_row_27
    exact row_70
    exact row_28
  -- chapter_153_line_16: GL tag equality2.
  have row_16 : (zero = two) := by
    exact Eq.trans row_17 row_27
  -- chapter_153_line_6: GL tag implication.
  have row_6 : (¬ (succ one zero)) := by
    apply row_7
    exact row_11
  -- chapter_153_line_5: GL tag equality1.
  have row_5 : (¬ (succ one two)) := by
    have equality_source := row_6
    have equality_step_1 := row_16
    cases equality_step_1
    exact equality_source
  -- chapter_153_line_1: GL tag contradiction.
  have row_1 : (¬ (gl_interval N add zero v2 V2)) := by
    exact False.elim (row_5 row_2)
  exact row_1 reductio

theorem gauss_source_074
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorGauss N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (V1 : GLSet α) (V2 : GLSet α), ((gl_limitSet N add V1 v1 V2) → ((gl_interval N add zero v2 V1) → (gl_interval N add zero v1 V2)))))) := by
  intro v1
  intro v2
  intro premise_1
  intro V1
  intro V2
  intro premise_2
  intro premise_3
  -- chapter_154_line_30: GL tag task formulation.
  have row_30 : (succ v1 v2) := by
    exact premise_1
  -- chapter_154_line_29: GL tag task formulation.
  have row_29 : (gl_AnchorGauss N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_154_line_40: GL tag expansion.
  have row_40 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorGauss] using row_29
  -- chapter_154_line_39: GL tag disintegration.
  have row_39 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_40.1.1.1
  -- chapter_154_line_38: GL tag expansion.
  have row_38 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_39
  -- chapter_154_line_37: GL tag disintegration.
  have row_37 : (gl_fXY succ N N) := by
    exact row_38.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_154_line_36: GL tag expansion.
  have row_36 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_37
  -- chapter_154_line_35: GL tag disintegration.
  have row_35 : (gl_implication0 succ N) := by
    exact row_36.1.1.1
  -- chapter_154_line_34: GL tag expansion.
  have row_34 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_35
  -- chapter_154_line_33: GL tag implication.
  have row_33 : (N v1) := by
    apply row_34
    exact row_30
  -- chapter_154_line_28: GL tag theorem.
  have row_28 := gauss_source_085 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_154_line_26: GL tag expansion for integration.
  have row_26 : ((gl_implication28 N add zero v1 V2) ↔ (∀ (v5 : α), ((gl_preorder N add zero v5) → ((gl_preorder N add v5 v1) → (V2 v5))))) := by
    exact Iff.rfl
  -- chapter_154_line_31: GL tag premise element.
  have row_31 : (∀ (v5 : α), ((gl_preorder N add zero v5) → ((gl_preorder N add v5 v1) → (gl_preorder N add v5 v1)))) := by
    intro v5
    intro scope_premise_1
    intro scope_premise_2
    exact scope_premise_2
  -- chapter_154_line_27: GL tag implication.
  have row_27 : (∀ (v5 : α), ((gl_preorder N add zero v5) → ((gl_preorder N add v5 v1) → (gl_preorder N add v5 v2)))) := by
    intro v5
    intro scope_premise_1
    intro scope_premise_2
    have scoped_fact_3 := row_31 v5 scope_premise_1 scope_premise_2
    apply row_28
    exact row_30
    exact scoped_fact_3
  -- chapter_154_line_25: GL tag premise element.
  have row_25 : (∀ (v5 : α), ((gl_preorder N add zero v5) → ((gl_preorder N add v5 v1) → (gl_preorder N add zero v5)))) := by
    intro v5
    intro scope_premise_1
    intro scope_premise_2
    exact scope_premise_1
  -- chapter_154_line_16: GL tag expansion for integration.
  have row_16 : ((gl_implication26 V2 N add zero) ↔ (∀ (v3 : α), ((V2 v3) → (gl_preorder N add zero v3)))) := by
    exact Iff.rfl
  -- chapter_154_line_15: GL tag premise element.
  have row_15 : (∀ (v3 : α), ((V2 v3) → (V2 v3))) := by
    intro v3
    intro scope_premise_1
    exact scope_premise_1
  -- chapter_154_line_14: GL tag task formulation.
  have row_14 : (gl_limitSet N add V1 v1 V2) := by
    exact premise_2
  -- chapter_154_line_13: GL tag expansion.
  have row_13 : (((gl_implication41 V2 V1) ∧ (gl_implication27 V2 N add v1)) ∧ (gl_implication42 V1 N add v1 V2)) := by
    simpa only [gl_limitSet] using row_14
  -- chapter_154_line_21: GL tag disintegration.
  have row_21 : (gl_implication42 V1 N add v1 V2) := by
    exact row_13.2
  -- chapter_154_line_20: GL tag expansion.
  have row_20 : (∀ (w1 : α), ((V1 w1) → ((gl_preorder N add w1 v1) → (V2 w1)))) := by
    simpa only [gl_implication42] using row_21
  -- chapter_154_line_17: GL tag disintegration.
  have row_17 : (gl_implication27 V2 N add v1) := by
    exact row_13.1.2
  -- chapter_154_line_12: GL tag disintegration.
  have row_12 : (gl_implication41 V2 V1) := by
    exact row_13.1.1
  -- chapter_154_line_11: GL tag expansion.
  have row_11 : (∀ (w1 : α), ((V2 w1) → (V1 w1))) := by
    simpa only [gl_implication41] using row_12
  -- chapter_154_line_10: GL tag implication.
  have row_10 : (∀ (v3 : α), ((V2 v3) → (V1 v3))) := by
    intro v3
    intro scope_premise_1
    have scoped_fact_2 := row_15 v3 scope_premise_1
    apply row_11
    exact scoped_fact_2
  -- chapter_154_line_9: GL tag task formulation.
  have row_9 : (gl_interval N add zero v2 V1) := by
    exact premise_3
  -- chapter_154_line_8: GL tag expansion.
  have row_8 : (((((gl_implication26 V1 N add zero) ∧ (gl_implication27 V1 N add v2)) ∧ (gl_implication28 N add zero v2 V1)) ∧ (N zero)) ∧ (N v2)) := by
    simpa only [gl_interval] using row_9
  -- chapter_154_line_32: GL tag disintegration.
  have row_32 : (N zero) := by
    exact row_8.1.2
  -- chapter_154_line_24: GL tag disintegration.
  have row_24 : (gl_implication28 N add zero v2 V1) := by
    exact row_8.1.1.2
  -- chapter_154_line_23: GL tag expansion.
  have row_23 : (∀ (w1 : α), ((gl_preorder N add zero w1) → ((gl_preorder N add w1 v2) → (V1 w1)))) := by
    simpa only [gl_implication28] using row_24
  -- chapter_154_line_22: GL tag implication.
  have row_22 : (∀ (v5 : α), ((gl_preorder N add zero v5) → ((gl_preorder N add v5 v1) → (V1 v5)))) := by
    intro v5
    intro scope_premise_1
    intro scope_premise_2
    have scoped_fact_2 := row_25 v5 scope_premise_1 scope_premise_2
    have scoped_fact_3 := row_27 v5 scope_premise_1 scope_premise_2
    apply row_23
    exact scoped_fact_2
    exact scoped_fact_3
  -- chapter_154_line_19: GL tag implication.
  have row_19 : (∀ (v5 : α), ((gl_preorder N add zero v5) → ((gl_preorder N add v5 v1) → (V2 v5)))) := by
    intro v5
    intro scope_premise_1
    intro scope_premise_2
    have scoped_fact_2 := row_22 v5 scope_premise_1 scope_premise_2
    have scoped_fact_3 := row_31 v5 scope_premise_1 scope_premise_2
    apply row_20
    exact scoped_fact_2
    exact scoped_fact_3
  -- chapter_154_line_18: GL tag validity name.
  have row_18 : (gl_implication28 N add zero v1 V2) := by
    simpa only [gl_implication28] using row_19
  -- chapter_154_line_7: GL tag disintegration.
  have row_7 : (gl_implication26 V1 N add zero) := by
    exact row_8.1.1.1.1
  -- chapter_154_line_6: GL tag expansion.
  have row_6 : (∀ (w1 : α), ((V1 w1) → (gl_preorder N add zero w1))) := by
    simpa only [gl_implication26] using row_7
  -- chapter_154_line_5: GL tag implication.
  have row_5 : (∀ (v3 : α), ((V2 v3) → (gl_preorder N add zero v3))) := by
    intro v3
    intro scope_premise_1
    have scoped_fact_2 := row_10 v3 scope_premise_1
    apply row_6
    exact scoped_fact_2
  -- chapter_154_line_4: GL tag validity name.
  have row_4 : (gl_implication26 V2 N add zero) := by
    simpa only [gl_implication26] using row_5
  -- chapter_154_line_3: GL tag expansion for integration.
  have row_3 : ((gl_interval N add zero v1 V2) ↔ (((((gl_implication26 V2 N add zero) ∧ (gl_implication27 V2 N add v1)) ∧ (gl_implication28 N add zero v1 V2)) ∧ (N zero)) ∧ (N v1))) := by
    exact Iff.rfl
  -- chapter_154_line_2: GL tag reformulation for integration and.
  have row_2 : ((gl_implication26 V2 N add zero) → ((gl_implication27 V2 N add v1) → ((gl_implication28 N add zero v1 V2) → ((N zero) → ((N v1) → (gl_interval N add zero v1 V2)))))) := by
    intro integration_premise_1
    intro integration_premise_2
    intro integration_premise_3
    intro integration_premise_4
    intro integration_premise_5
    simp only [gl_interval]
    exact ⟨⟨⟨⟨integration_premise_1, integration_premise_2⟩, integration_premise_3⟩, integration_premise_4⟩, integration_premise_5⟩
  -- chapter_154_line_1: GL tag implication.
  have row_1 : (gl_interval N add zero v1 V2) := by
    apply row_2
    exact row_4
    exact row_17
    exact row_18
    exact row_32
    exact row_33
  exact row_1

theorem gauss_source_075
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorGauss N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (V1 : GLSet α) (V2 : GLSet α), ((gl_limitSet N add V1 v1 V2) → ((gl_interval N add zero one V1) → (¬ (gl_interval N add zero v2 V2))))))) := by
  intro v1
  intro v2
  intro premise_1
  intro V1
  intro V2
  intro premise_2
  intro premise_3
  intro reductio
  -- chapter_155_line_42: GL tag expansion for integration.
  have row_42 : ((gl_preorder N add v1 v2) ↔ (¬ (∀ (v11 : α), ((N v11) → (¬ (add v1 v11 v2)))))) := by
    exact Iff.rfl
  -- chapter_155_line_41: GL tag reformulation for integration >[bound].
  have row_41 : (∀ (v10 : α), ((N v10) → ((add v1 v10 v2) → (gl_preorder N add v1 v2)))) := by
    intro v10
    intro integration_premise_1
    intro integration_premise_2
    apply (row_42).2
    intro universal_counterexample
    exact universal_counterexample v10 integration_premise_1 integration_premise_2
  -- chapter_155_line_39: GL tag task formulation.
  have row_39 : (gl_interval N add zero v2 V2) := by
    exact reductio
  -- chapter_155_line_56: GL tag expansion.
  have row_56 : (((((gl_implication26 V2 N add zero) ∧ (gl_implication27 V2 N add v2)) ∧ (gl_implication28 N add zero v2 V2)) ∧ (N zero)) ∧ (N v2)) := by
    simpa only [gl_interval] using row_39
  -- chapter_155_line_55: GL tag disintegration.
  have row_55 : (N zero) := by
    exact row_56.1.2
  -- chapter_155_line_38: GL tag theorem.
  have row_38 := gauss_source_090 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_155_line_36: GL tag task formulation.
  have row_36 : (gl_limitSet N add V1 v1 V2) := by
    exact premise_2
  -- chapter_155_line_35: GL tag expansion.
  have row_35 : (((gl_implication41 V2 V1) ∧ (gl_implication27 V2 N add v1)) ∧ (gl_implication42 V1 N add v1 V2)) := by
    simpa only [gl_limitSet] using row_36
  -- chapter_155_line_34: GL tag disintegration.
  have row_34 : (gl_implication27 V2 N add v1) := by
    exact row_35.1.2
  -- chapter_155_line_33: GL tag expansion.
  have row_33 : (∀ (w1 : α), ((V2 w1) → (gl_preorder N add w1 v1))) := by
    simpa only [gl_implication27] using row_34
  -- chapter_155_line_30: GL tag task formulation.
  have row_30 : (succ v1 v2) := by
    exact premise_1
  -- chapter_155_line_24: GL tag expansion for integration.
  have row_24 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_155_line_23: GL tag reformulation for integration and.
  have row_23 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_155_line_13: GL tag task formulation.
  have row_13 : (gl_interval N add zero one V1) := by
    exact premise_3
  -- chapter_155_line_12: GL tag expansion.
  have row_12 : (((((gl_implication26 V1 N add zero) ∧ (gl_implication27 V1 N add one)) ∧ (gl_implication28 N add zero one V1)) ∧ (N zero)) ∧ (N one)) := by
    simpa only [gl_interval] using row_13
  -- chapter_155_line_11: GL tag disintegration.
  have row_11 : (N one) := by
    exact row_12.2
  -- chapter_155_line_4: GL tag task formulation.
  have row_4 : (gl_AnchorGauss N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_155_line_37: GL tag implication.
  have row_37 : (V2 v2) := by
    apply row_38
    exact row_39
  -- chapter_155_line_32: GL tag implication.
  have row_32 : (gl_preorder N add v2 v1) := by
    apply row_33
    exact row_37
  -- chapter_155_line_3: GL tag expansion.
  have row_3 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorGauss] using row_4
  -- chapter_155_line_18: GL tag disintegration.
  have row_18 : (succ zero one) := by
    exact row_3.1.1.2
  -- chapter_155_line_10: GL tag disintegration.
  have row_10 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_3.1.1.1
  -- chapter_155_line_22: GL tag implication.
  have row_22 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_23
    exact row_10
    exact row_18
  have rule_row_44 := peano_source_033 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_155_line_44: GL tag implication.
  have row_44 : (add one v1 v2) := by
    apply rule_row_44
    exact row_30
  have rule_row_43 := peano_source_037 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_155_line_43: GL tag implication.
  have row_43 : (add v1 one v2) := by
    apply rule_row_43
    exact row_44
  -- chapter_155_line_40: GL tag implication.
  have row_40 : (gl_preorder N add v1 v2) := by
    apply row_41
    exact row_11
    exact row_43
  have rule_row_66 := peano_source_052 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_155_line_66: GL tag implication.
  have row_66 : (v1 = v2) := by
    apply rule_row_66
    exact row_40
    exact row_32
  have rule_row_31 := peano_source_052 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_155_line_31: GL tag implication.
  have row_31 : (v2 = v1) := by
    apply rule_row_31
    exact row_32
    exact row_40
  have rule_row_21 := peano_source_063 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_155_line_21: GL tag implication.
  have row_21 : (gl_existence3 N one succ) := by
    apply rule_row_21
  -- chapter_155_line_20: GL tag expansion.
  have row_20 : (¬ (∀ (v4 : α), ((N v4) → (¬ (succ v4 one))))) := by
    simpa only [gl_existence3] using row_21
  have exists_row_20 : ∃ (v4 : α), ((N v4) ∧ (succ v4 one)) := existsAndOfNotForallImpNot row_20
  obtain ⟨v4, witness_row_20⟩ := exists_row_20
  -- chapter_155_line_19: GL tag disintegration.
  have row_19 : (succ v4 one) := by
    exact witness_row_20.2
  -- chapter_155_line_9: GL tag expansion.
  have row_9 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_10
  -- chapter_155_line_61: GL tag disintegration.
  have row_61 : (gl_fXY succ N N) := by
    exact row_9.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_155_line_60: GL tag expansion.
  have row_60 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_61
  -- chapter_155_line_59: GL tag disintegration.
  have row_59 : (gl_implication0 succ N) := by
    exact row_60.1.1.1
  -- chapter_155_line_58: GL tag expansion.
  have row_58 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_59
  -- chapter_155_line_57: GL tag implication.
  have row_57 : (N v1) := by
    apply row_58
    exact row_30
  -- chapter_155_line_54: GL tag disintegration.
  have row_54 : (gl_fXYZ add N N N) := by
    exact row_9.1.1.1.1.1.1.1.1.2
  -- chapter_155_line_53: GL tag expansion.
  have row_53 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_54
  -- chapter_155_line_52: GL tag disintegration.
  have row_52 : (gl_implication13 N N N add) := by
    exact row_53.1.2
  -- chapter_155_line_51: GL tag expansion.
  have row_51 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication13] using row_52
  -- chapter_155_line_50: GL tag implication.
  have row_50 : (gl_existence1 N v1 zero add) := by
    apply row_51
    exact row_57
    exact row_55
  -- chapter_155_line_49: GL tag expansion.
  have row_49 : (¬ (∀ (v9 : α), ((N v9) → (¬ (add v1 zero v9))))) := by
    simpa only [gl_existence1] using row_50
  have exists_row_49 : ∃ (v9 : α), ((N v9) ∧ (add v1 zero v9)) := existsAndOfNotForallImpNot row_49
  obtain ⟨v9, witness_row_49⟩ := exists_row_49
  -- chapter_155_line_48: GL tag disintegration.
  have row_48 : (add v1 zero v9) := by
    exact witness_row_49.2
  -- chapter_155_line_47: GL tag disintegration.
  have row_47 : (gl_implication15 N zero add) := by
    exact row_9.1.1.1.1.1.1.1.2
  -- chapter_155_line_46: GL tag expansion.
  have row_46 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((add w1 zero w2) → (w1 = w2))))) := by
    simpa only [gl_implication15] using row_47
  -- chapter_155_line_45: GL tag implication.
  have row_45 : (v1 = v9) := by
    apply row_46
    exact row_57
    exact row_48
  -- chapter_155_line_69: GL tag symmetry of equality.
  have row_69 : (v9 = v1) := by
    exact Eq.symm row_45
  -- chapter_155_line_68: GL tag equality1.
  have row_68 : (add v2 zero v1) := by
    have equality_source := row_48
    have equality_step_1 := row_66
    cases equality_step_1
    have equality_step_2 := row_69
    cases equality_step_2
    exact equality_source
  -- chapter_155_line_65: GL tag equality1.
  have row_65 : (gl_preorder N add v9 v2) := by
    have equality_source := row_40
    have equality_step_1 := row_45
    cases equality_step_1
    exact equality_source
  -- chapter_155_line_64: GL tag equality1.
  have row_64 : (gl_preorder N add v2 v9) := by
    have equality_source := row_32
    have equality_step_1 := row_45
    cases equality_step_1
    exact equality_source
  have rule_row_63 := peano_source_052 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_155_line_63: GL tag implication.
  have row_63 : (v2 = v9) := by
    apply rule_row_63
    exact row_64
    exact row_65
  -- chapter_155_line_62: GL tag equality1.
  have row_62 : (add v2 one v9) := by
    have equality_source := row_43
    have equality_step_1 := row_63
    cases equality_step_1
    have equality_step_2 := row_66
    cases equality_step_2
    exact equality_source
  -- chapter_155_line_29: GL tag equality1.
  have row_29 : (succ v9 v1) := by
    have equality_source := row_30
    have equality_step_1 := row_31
    cases equality_step_1
    have equality_step_2 := row_45
    cases equality_step_2
    exact equality_source
  -- chapter_155_line_28: GL tag disintegration.
  have row_28 : (gl_implication18 N succ add) := by
    exact row_9.1.1.1.1.2
  -- chapter_155_line_27: GL tag expansion.
  have row_27 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((succ w4 w5) → (add w3 w2 w5))))))))) := by
    simpa only [gl_implication18] using row_28
  -- chapter_155_line_17: GL tag disintegration.
  have row_17 : (gl_implication7 N succ) := by
    exact row_9.1.1.1.1.1.1.1.1.1.2
  -- chapter_155_line_16: GL tag expansion.
  have row_16 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w2 w1) → (∀ (w3 : α), ((succ w3 w1) → (w2 = w3))))))) := by
    simpa only [gl_implication7] using row_17
  -- chapter_155_line_15: GL tag implication.
  have row_15 : (zero = v4) := by
    apply row_16
    exact row_11
    exact row_18
    exact row_19
  -- chapter_155_line_67: GL tag equality1.
  have row_67 : (add v2 v4 v1) := by
    have equality_source := row_68
    have equality_step_1 := row_15
    cases equality_step_1
    exact equality_source
  -- chapter_155_line_8: GL tag disintegration.
  have row_8 : (gl_implication6 N zero succ) := by
    exact row_9.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_155_line_7: GL tag expansion.
  have row_7 : (∀ (w1 : α), ((N w1) → (¬ (succ w1 zero)))) := by
    simpa only [gl_implication6] using row_8
  -- chapter_155_line_6: GL tag implication.
  have row_6 : (¬ (succ one zero)) := by
    apply row_7
    exact row_11
  -- chapter_155_line_2: GL tag disintegration.
  have row_2 : (succ one two) := by
    exact row_3.1.2
  -- chapter_155_line_26: GL tag implication.
  have row_26 : (add v2 two v1) := by
    apply row_27
    exact row_11
    exact row_2
    exact row_62
    exact row_29
  have rule_row_25 := peano_source_013 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_155_line_25: GL tag implication.
  have row_25 : (v4 = two) := by
    apply rule_row_25
    exact row_67
    exact row_26
  -- chapter_155_line_14: GL tag equality2.
  have row_14 : (zero = two) := by
    exact Eq.trans row_15 row_25
  -- chapter_155_line_5: GL tag equality1.
  have row_5 : (¬ (succ one two)) := by
    have equality_source := row_6
    have equality_step_1 := row_14
    cases equality_step_1
    exact equality_source
  -- chapter_155_line_1: GL tag contradiction.
  have row_1 : (¬ (gl_interval N add zero v2 V2)) := by
    exact False.elim (row_5 row_2)
  exact row_1 reductio

theorem gauss_source_076
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorGauss N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (V1 : GLSet α) (V2 : GLSet α), ((gl_limitSet N add V1 v1 V2) → ((gl_interval N add zero v1 V1) → (¬ (gl_interval N add zero v2 V2))))))) := by
  intro v1
  intro v2
  intro premise_1
  intro V1
  intro V2
  intro premise_2
  intro premise_3
  intro reductio
  -- chapter_156_line_61: GL tag task formulation.
  have row_61 : (gl_interval N add zero v1 V1) := by
    exact premise_3
  -- chapter_156_line_60: GL tag expansion.
  have row_60 : (((((gl_implication26 V1 N add zero) ∧ (gl_implication27 V1 N add v1)) ∧ (gl_implication28 N add zero v1 V1)) ∧ (N zero)) ∧ (N v1)) := by
    simpa only [gl_interval] using row_61
  -- chapter_156_line_59: GL tag disintegration.
  have row_59 : (N v1) := by
    exact row_60.2
  -- chapter_156_line_44: GL tag expansion for integration.
  have row_44 : ((gl_preorder N add v1 v2) ↔ (¬ (∀ (v11 : α), ((N v11) → (¬ (add v1 v11 v2)))))) := by
    exact Iff.rfl
  -- chapter_156_line_43: GL tag reformulation for integration >[bound].
  have row_43 : (∀ (v10 : α), ((N v10) → ((add v1 v10 v2) → (gl_preorder N add v1 v2)))) := by
    intro v10
    intro integration_premise_1
    intro integration_premise_2
    apply (row_44).2
    intro universal_counterexample
    exact universal_counterexample v10 integration_premise_1 integration_premise_2
  -- chapter_156_line_41: GL tag task formulation.
  have row_41 : (gl_interval N add zero v2 V2) := by
    exact reductio
  -- chapter_156_line_58: GL tag expansion.
  have row_58 : (((((gl_implication26 V2 N add zero) ∧ (gl_implication27 V2 N add v2)) ∧ (gl_implication28 N add zero v2 V2)) ∧ (N zero)) ∧ (N v2)) := by
    simpa only [gl_interval] using row_41
  -- chapter_156_line_57: GL tag disintegration.
  have row_57 : (N zero) := by
    exact row_58.1.2
  -- chapter_156_line_40: GL tag theorem.
  have row_40 := gauss_source_090 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_156_line_38: GL tag task formulation.
  have row_38 : (gl_limitSet N add V1 v1 V2) := by
    exact premise_2
  -- chapter_156_line_37: GL tag expansion.
  have row_37 : (((gl_implication41 V2 V1) ∧ (gl_implication27 V2 N add v1)) ∧ (gl_implication42 V1 N add v1 V2)) := by
    simpa only [gl_limitSet] using row_38
  -- chapter_156_line_36: GL tag disintegration.
  have row_36 : (gl_implication27 V2 N add v1) := by
    exact row_37.1.2
  -- chapter_156_line_35: GL tag expansion.
  have row_35 : (∀ (w1 : α), ((V2 w1) → (gl_preorder N add w1 v1))) := by
    simpa only [gl_implication27] using row_36
  -- chapter_156_line_32: GL tag task formulation.
  have row_32 : (succ v1 v2) := by
    exact premise_1
  -- chapter_156_line_26: GL tag expansion for integration.
  have row_26 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_156_line_25: GL tag reformulation for integration and.
  have row_25 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_156_line_4: GL tag task formulation.
  have row_4 : (gl_AnchorGauss N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_156_line_39: GL tag implication.
  have row_39 : (V2 v2) := by
    apply row_40
    exact row_41
  -- chapter_156_line_34: GL tag implication.
  have row_34 : (gl_preorder N add v2 v1) := by
    apply row_35
    exact row_39
  -- chapter_156_line_3: GL tag expansion.
  have row_3 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorGauss] using row_4
  -- chapter_156_line_20: GL tag disintegration.
  have row_20 : (succ zero one) := by
    exact row_3.1.1.2
  -- chapter_156_line_10: GL tag disintegration.
  have row_10 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_3.1.1.1
  -- chapter_156_line_24: GL tag implication.
  have row_24 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_25
    exact row_10
    exact row_20
  have rule_row_46 := peano_source_033 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_156_line_46: GL tag implication.
  have row_46 : (add one v1 v2) := by
    apply rule_row_46
    exact row_32
  have rule_row_45 := peano_source_037 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_156_line_45: GL tag implication.
  have row_45 : (add v1 one v2) := by
    apply rule_row_45
    exact row_46
  have rule_row_23 := peano_source_063 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_156_line_23: GL tag implication.
  have row_23 : (gl_existence3 N one succ) := by
    apply rule_row_23
  -- chapter_156_line_22: GL tag expansion.
  have row_22 : (¬ (∀ (v5 : α), ((N v5) → (¬ (succ v5 one))))) := by
    simpa only [gl_existence3] using row_23
  have exists_row_22 : ∃ (v5 : α), ((N v5) ∧ (succ v5 one)) := existsAndOfNotForallImpNot row_22
  obtain ⟨v5, witness_row_22⟩ := exists_row_22
  -- chapter_156_line_21: GL tag disintegration.
  have row_21 : (succ v5 one) := by
    exact witness_row_22.2
  -- chapter_156_line_9: GL tag expansion.
  have row_9 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_10
  -- chapter_156_line_56: GL tag disintegration.
  have row_56 : (gl_fXYZ add N N N) := by
    exact row_9.1.1.1.1.1.1.1.1.2
  -- chapter_156_line_55: GL tag expansion.
  have row_55 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_56
  -- chapter_156_line_54: GL tag disintegration.
  have row_54 : (gl_implication13 N N N add) := by
    exact row_55.1.2
  -- chapter_156_line_53: GL tag expansion.
  have row_53 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication13] using row_54
  -- chapter_156_line_52: GL tag implication.
  have row_52 : (gl_existence1 N v1 zero add) := by
    apply row_53
    exact row_59
    exact row_57
  -- chapter_156_line_51: GL tag expansion.
  have row_51 : (¬ (∀ (v9 : α), ((N v9) → (¬ (add v1 zero v9))))) := by
    simpa only [gl_existence1] using row_52
  have exists_row_51 : ∃ (v9 : α), ((N v9) ∧ (add v1 zero v9)) := existsAndOfNotForallImpNot row_51
  obtain ⟨v9, witness_row_51⟩ := exists_row_51
  -- chapter_156_line_50: GL tag disintegration.
  have row_50 : (add v1 zero v9) := by
    exact witness_row_51.2
  -- chapter_156_line_49: GL tag disintegration.
  have row_49 : (gl_implication15 N zero add) := by
    exact row_9.1.1.1.1.1.1.1.2
  -- chapter_156_line_48: GL tag expansion.
  have row_48 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((add w1 zero w2) → (w1 = w2))))) := by
    simpa only [gl_implication15] using row_49
  -- chapter_156_line_47: GL tag implication.
  have row_47 : (v1 = v9) := by
    apply row_48
    exact row_59
    exact row_50
  -- chapter_156_line_69: GL tag symmetry of equality.
  have row_69 : (v9 = v1) := by
    exact Eq.symm row_47
  -- chapter_156_line_64: GL tag equality1.
  have row_64 : (gl_preorder N add v2 v9) := by
    have equality_source := row_34
    have equality_step_1 := row_47
    cases equality_step_1
    exact equality_source
  -- chapter_156_line_30: GL tag disintegration.
  have row_30 : (gl_implication18 N succ add) := by
    exact row_9.1.1.1.1.2
  -- chapter_156_line_29: GL tag expansion.
  have row_29 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((succ w4 w5) → (add w3 w2 w5))))))))) := by
    simpa only [gl_implication18] using row_30
  -- chapter_156_line_19: GL tag disintegration.
  have row_19 : (gl_implication7 N succ) := by
    exact row_9.1.1.1.1.1.1.1.1.1.2
  -- chapter_156_line_18: GL tag expansion.
  have row_18 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w2 w1) → (∀ (w3 : α), ((succ w3 w1) → (w2 = w3))))))) := by
    simpa only [gl_implication7] using row_19
  -- chapter_156_line_15: GL tag disintegration.
  have row_15 : (gl_fXY succ N N) := by
    exact row_9.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_156_line_14: GL tag expansion.
  have row_14 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_15
  -- chapter_156_line_13: GL tag disintegration.
  have row_13 : (gl_implication0 succ N) := by
    exact row_14.1.1.1
  -- chapter_156_line_12: GL tag expansion.
  have row_12 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_13
  -- chapter_156_line_8: GL tag disintegration.
  have row_8 : (gl_implication6 N zero succ) := by
    exact row_9.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_156_line_7: GL tag expansion.
  have row_7 : (∀ (w1 : α), ((N w1) → (¬ (succ w1 zero)))) := by
    simpa only [gl_implication6] using row_8
  -- chapter_156_line_2: GL tag disintegration.
  have row_2 : (succ one two) := by
    exact row_3.1.2
  -- chapter_156_line_11: GL tag implication.
  have row_11 : (N one) := by
    apply row_12
    exact row_2
  -- chapter_156_line_42: GL tag implication.
  have row_42 : (gl_preorder N add v1 v2) := by
    apply row_43
    exact row_11
    exact row_45
  have rule_row_66 := peano_source_052 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_156_line_66: GL tag implication.
  have row_66 : (v1 = v2) := by
    apply rule_row_66
    exact row_42
    exact row_34
  -- chapter_156_line_68: GL tag equality1.
  have row_68 : (add v2 zero v1) := by
    have equality_source := row_50
    have equality_step_1 := row_66
    cases equality_step_1
    have equality_step_2 := row_69
    cases equality_step_2
    exact equality_source
  -- chapter_156_line_65: GL tag equality1.
  have row_65 : (gl_preorder N add v9 v2) := by
    have equality_source := row_42
    have equality_step_1 := row_47
    cases equality_step_1
    exact equality_source
  have rule_row_63 := peano_source_052 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_156_line_63: GL tag implication.
  have row_63 : (v2 = v9) := by
    apply rule_row_63
    exact row_64
    exact row_65
  -- chapter_156_line_62: GL tag equality1.
  have row_62 : (add v2 one v9) := by
    have equality_source := row_45
    have equality_step_1 := row_63
    cases equality_step_1
    have equality_step_2 := row_66
    cases equality_step_2
    exact equality_source
  have rule_row_33 := peano_source_052 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_156_line_33: GL tag implication.
  have row_33 : (v2 = v1) := by
    apply rule_row_33
    exact row_34
    exact row_42
  -- chapter_156_line_31: GL tag equality1.
  have row_31 : (succ v9 v1) := by
    have equality_source := row_32
    have equality_step_1 := row_33
    cases equality_step_1
    have equality_step_2 := row_47
    cases equality_step_2
    exact equality_source
  -- chapter_156_line_28: GL tag implication.
  have row_28 : (add v2 two v1) := by
    apply row_29
    exact row_11
    exact row_2
    exact row_62
    exact row_31
  -- chapter_156_line_17: GL tag implication.
  have row_17 : (zero = v5) := by
    apply row_18
    exact row_11
    exact row_20
    exact row_21
  -- chapter_156_line_67: GL tag equality1.
  have row_67 : (add v2 v5 v1) := by
    have equality_source := row_68
    have equality_step_1 := row_17
    cases equality_step_1
    exact equality_source
  have rule_row_27 := peano_source_013 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_156_line_27: GL tag implication.
  have row_27 : (v5 = two) := by
    apply rule_row_27
    exact row_67
    exact row_28
  -- chapter_156_line_16: GL tag equality2.
  have row_16 : (zero = two) := by
    exact Eq.trans row_17 row_27
  -- chapter_156_line_6: GL tag implication.
  have row_6 : (¬ (succ one zero)) := by
    apply row_7
    exact row_11
  -- chapter_156_line_5: GL tag equality1.
  have row_5 : (¬ (succ one two)) := by
    have equality_source := row_6
    have equality_step_1 := row_16
    cases equality_step_1
    exact equality_source
  -- chapter_156_line_1: GL tag contradiction.
  have row_1 : (¬ (gl_interval N add zero v2 V2)) := by
    exact False.elim (row_5 row_2)
  exact row_1 reductio

theorem gauss_source_079
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorGauss N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (V1 : GLSet α), ((gl_interval N add zero v1 V1) → (¬ (gl_interval N add zero v2 V1)))))) := by
  intro v1
  intro v2
  intro premise_1
  intro V1
  intro premise_2
  intro reductio
  -- chapter_159_line_44: GL tag expansion for integration.
  have row_44 : ((gl_preorder N add v1 v2) ↔ (¬ (∀ (v11 : α), ((N v11) → (¬ (add v1 v11 v2)))))) := by
    exact Iff.rfl
  -- chapter_159_line_43: GL tag reformulation for integration >[bound].
  have row_43 : (∀ (v10 : α), ((N v10) → ((add v1 v10 v2) → (gl_preorder N add v1 v2)))) := by
    intro v10
    intro integration_premise_1
    intro integration_premise_2
    apply (row_44).2
    intro universal_counterexample
    exact universal_counterexample v10 integration_premise_1 integration_premise_2
  -- chapter_159_line_41: GL tag task formulation.
  have row_41 : (gl_interval N add zero v2 V1) := by
    exact reductio
  -- chapter_159_line_58: GL tag expansion.
  have row_58 : (((((gl_implication26 V1 N add zero) ∧ (gl_implication27 V1 N add v2)) ∧ (gl_implication28 N add zero v2 V1)) ∧ (N zero)) ∧ (N v2)) := by
    simpa only [gl_interval] using row_41
  -- chapter_159_line_57: GL tag disintegration.
  have row_57 : (N zero) := by
    exact row_58.1.2
  -- chapter_159_line_40: GL tag theorem.
  have row_40 := gauss_source_090 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_159_line_38: GL tag task formulation.
  have row_38 : (gl_interval N add zero v1 V1) := by
    exact premise_2
  -- chapter_159_line_37: GL tag expansion.
  have row_37 : (((((gl_implication26 V1 N add zero) ∧ (gl_implication27 V1 N add v1)) ∧ (gl_implication28 N add zero v1 V1)) ∧ (N zero)) ∧ (N v1)) := by
    simpa only [gl_interval] using row_38
  -- chapter_159_line_59: GL tag disintegration.
  have row_59 : (N v1) := by
    exact row_37.2
  -- chapter_159_line_36: GL tag disintegration.
  have row_36 : (gl_implication27 V1 N add v1) := by
    exact row_37.1.1.1.2
  -- chapter_159_line_35: GL tag expansion.
  have row_35 : (∀ (w1 : α), ((V1 w1) → (gl_preorder N add w1 v1))) := by
    simpa only [gl_implication27] using row_36
  -- chapter_159_line_32: GL tag task formulation.
  have row_32 : (succ v1 v2) := by
    exact premise_1
  -- chapter_159_line_26: GL tag expansion for integration.
  have row_26 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_159_line_25: GL tag reformulation for integration and.
  have row_25 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_159_line_4: GL tag task formulation.
  have row_4 : (gl_AnchorGauss N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_159_line_39: GL tag implication.
  have row_39 : (V1 v2) := by
    apply row_40
    exact row_41
  -- chapter_159_line_34: GL tag implication.
  have row_34 : (gl_preorder N add v2 v1) := by
    apply row_35
    exact row_39
  -- chapter_159_line_3: GL tag expansion.
  have row_3 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorGauss] using row_4
  -- chapter_159_line_20: GL tag disintegration.
  have row_20 : (succ zero one) := by
    exact row_3.1.1.2
  -- chapter_159_line_10: GL tag disintegration.
  have row_10 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_3.1.1.1
  -- chapter_159_line_24: GL tag implication.
  have row_24 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_25
    exact row_10
    exact row_20
  have rule_row_46 := peano_source_033 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_159_line_46: GL tag implication.
  have row_46 : (add one v1 v2) := by
    apply rule_row_46
    exact row_32
  have rule_row_45 := peano_source_037 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_159_line_45: GL tag implication.
  have row_45 : (add v1 one v2) := by
    apply rule_row_45
    exact row_46
  have rule_row_23 := peano_source_063 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_159_line_23: GL tag implication.
  have row_23 : (gl_existence3 N one succ) := by
    apply rule_row_23
  -- chapter_159_line_22: GL tag expansion.
  have row_22 : (¬ (∀ (v5 : α), ((N v5) → (¬ (succ v5 one))))) := by
    simpa only [gl_existence3] using row_23
  have exists_row_22 : ∃ (v5 : α), ((N v5) ∧ (succ v5 one)) := existsAndOfNotForallImpNot row_22
  obtain ⟨v5, witness_row_22⟩ := exists_row_22
  -- chapter_159_line_21: GL tag disintegration.
  have row_21 : (succ v5 one) := by
    exact witness_row_22.2
  -- chapter_159_line_9: GL tag expansion.
  have row_9 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_10
  -- chapter_159_line_56: GL tag disintegration.
  have row_56 : (gl_fXYZ add N N N) := by
    exact row_9.1.1.1.1.1.1.1.1.2
  -- chapter_159_line_55: GL tag expansion.
  have row_55 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_56
  -- chapter_159_line_54: GL tag disintegration.
  have row_54 : (gl_implication13 N N N add) := by
    exact row_55.1.2
  -- chapter_159_line_53: GL tag expansion.
  have row_53 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication13] using row_54
  -- chapter_159_line_52: GL tag implication.
  have row_52 : (gl_existence1 N v1 zero add) := by
    apply row_53
    exact row_59
    exact row_57
  -- chapter_159_line_51: GL tag expansion.
  have row_51 : (¬ (∀ (v9 : α), ((N v9) → (¬ (add v1 zero v9))))) := by
    simpa only [gl_existence1] using row_52
  have exists_row_51 : ∃ (v9 : α), ((N v9) ∧ (add v1 zero v9)) := existsAndOfNotForallImpNot row_51
  obtain ⟨v9, witness_row_51⟩ := exists_row_51
  -- chapter_159_line_50: GL tag disintegration.
  have row_50 : (add v1 zero v9) := by
    exact witness_row_51.2
  -- chapter_159_line_49: GL tag disintegration.
  have row_49 : (gl_implication15 N zero add) := by
    exact row_9.1.1.1.1.1.1.1.2
  -- chapter_159_line_48: GL tag expansion.
  have row_48 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((add w1 zero w2) → (w1 = w2))))) := by
    simpa only [gl_implication15] using row_49
  -- chapter_159_line_47: GL tag implication.
  have row_47 : (v1 = v9) := by
    apply row_48
    exact row_59
    exact row_50
  -- chapter_159_line_67: GL tag symmetry of equality.
  have row_67 : (v9 = v1) := by
    exact Eq.symm row_47
  -- chapter_159_line_62: GL tag equality1.
  have row_62 : (gl_preorder N add v2 v9) := by
    have equality_source := row_34
    have equality_step_1 := row_47
    cases equality_step_1
    exact equality_source
  -- chapter_159_line_30: GL tag disintegration.
  have row_30 : (gl_implication18 N succ add) := by
    exact row_9.1.1.1.1.2
  -- chapter_159_line_29: GL tag expansion.
  have row_29 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((succ w4 w5) → (add w3 w2 w5))))))))) := by
    simpa only [gl_implication18] using row_30
  -- chapter_159_line_19: GL tag disintegration.
  have row_19 : (gl_implication7 N succ) := by
    exact row_9.1.1.1.1.1.1.1.1.1.2
  -- chapter_159_line_18: GL tag expansion.
  have row_18 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w2 w1) → (∀ (w3 : α), ((succ w3 w1) → (w2 = w3))))))) := by
    simpa only [gl_implication7] using row_19
  -- chapter_159_line_15: GL tag disintegration.
  have row_15 : (gl_fXY succ N N) := by
    exact row_9.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_159_line_14: GL tag expansion.
  have row_14 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_15
  -- chapter_159_line_13: GL tag disintegration.
  have row_13 : (gl_implication0 succ N) := by
    exact row_14.1.1.1
  -- chapter_159_line_12: GL tag expansion.
  have row_12 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_13
  -- chapter_159_line_8: GL tag disintegration.
  have row_8 : (gl_implication6 N zero succ) := by
    exact row_9.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_159_line_7: GL tag expansion.
  have row_7 : (∀ (w1 : α), ((N w1) → (¬ (succ w1 zero)))) := by
    simpa only [gl_implication6] using row_8
  -- chapter_159_line_2: GL tag disintegration.
  have row_2 : (succ one two) := by
    exact row_3.1.2
  -- chapter_159_line_11: GL tag implication.
  have row_11 : (N one) := by
    apply row_12
    exact row_2
  -- chapter_159_line_42: GL tag implication.
  have row_42 : (gl_preorder N add v1 v2) := by
    apply row_43
    exact row_11
    exact row_45
  have rule_row_64 := peano_source_052 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_159_line_64: GL tag implication.
  have row_64 : (v1 = v2) := by
    apply rule_row_64
    exact row_42
    exact row_34
  -- chapter_159_line_66: GL tag equality1.
  have row_66 : (add v2 zero v1) := by
    have equality_source := row_50
    have equality_step_1 := row_64
    cases equality_step_1
    have equality_step_2 := row_67
    cases equality_step_2
    exact equality_source
  -- chapter_159_line_63: GL tag equality1.
  have row_63 : (gl_preorder N add v9 v2) := by
    have equality_source := row_42
    have equality_step_1 := row_47
    cases equality_step_1
    exact equality_source
  have rule_row_61 := peano_source_052 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_159_line_61: GL tag implication.
  have row_61 : (v2 = v9) := by
    apply rule_row_61
    exact row_62
    exact row_63
  -- chapter_159_line_60: GL tag equality1.
  have row_60 : (add v2 one v9) := by
    have equality_source := row_45
    have equality_step_1 := row_61
    cases equality_step_1
    have equality_step_2 := row_64
    cases equality_step_2
    exact equality_source
  have rule_row_33 := peano_source_052 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_159_line_33: GL tag implication.
  have row_33 : (v2 = v1) := by
    apply rule_row_33
    exact row_34
    exact row_42
  -- chapter_159_line_31: GL tag equality1.
  have row_31 : (succ v9 v1) := by
    have equality_source := row_32
    have equality_step_1 := row_33
    cases equality_step_1
    have equality_step_2 := row_47
    cases equality_step_2
    exact equality_source
  -- chapter_159_line_28: GL tag implication.
  have row_28 : (add v2 two v1) := by
    apply row_29
    exact row_11
    exact row_2
    exact row_60
    exact row_31
  -- chapter_159_line_17: GL tag implication.
  have row_17 : (zero = v5) := by
    apply row_18
    exact row_11
    exact row_20
    exact row_21
  -- chapter_159_line_65: GL tag equality1.
  have row_65 : (add v2 v5 v1) := by
    have equality_source := row_66
    have equality_step_1 := row_17
    cases equality_step_1
    exact equality_source
  have rule_row_27 := peano_source_013 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_159_line_27: GL tag implication.
  have row_27 : (v5 = two) := by
    apply rule_row_27
    exact row_65
    exact row_28
  -- chapter_159_line_16: GL tag equality2.
  have row_16 : (zero = two) := by
    exact Eq.trans row_17 row_27
  -- chapter_159_line_6: GL tag implication.
  have row_6 : (¬ (succ one zero)) := by
    apply row_7
    exact row_11
  -- chapter_159_line_5: GL tag equality1.
  have row_5 : (¬ (succ one two)) := by
    have equality_source := row_6
    have equality_step_1 := row_16
    cases equality_step_1
    exact equality_source
  -- chapter_159_line_1: GL tag contradiction.
  have row_1 : (¬ (gl_interval N add zero v2 V1)) := by
    exact False.elim (row_5 row_2)
  exact row_1 reductio

theorem gauss_source_091
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorGauss N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((gl_preorder N add v1 v2) → (∀ (v3 : α), ((gl_preorder N add v2 v3) → (gl_preorder N add v1 v3))))) := by
  intro v1
  intro v2
  intro premise_1
  intro v3
  intro premise_2
  -- chapter_171_line_19: GL tag task formulation.
  have row_19 : (gl_preorder N add v1 v2) := by
    exact premise_1
  -- chapter_171_line_18: GL tag expansion.
  have row_18 : (¬ (∀ (v8 : α), ((N v8) → (¬ (add v1 v8 v2))))) := by
    simpa only [gl_preorder] using row_19
  have exists_row_18 : ∃ (v8 : α), ((N v8) ∧ (add v1 v8 v2)) := existsAndOfNotForallImpNot row_18
  obtain ⟨v8, witness_row_18⟩ := exists_row_18
  -- chapter_171_line_28: GL tag disintegration.
  have row_28 : (N v8) := by
    exact witness_row_18.1
  -- chapter_171_line_17: GL tag disintegration.
  have row_17 : (add v1 v8 v2) := by
    exact witness_row_18.2
  -- chapter_171_line_16: GL tag task formulation.
  have row_16 : (gl_preorder N add v2 v3) := by
    exact premise_2
  -- chapter_171_line_15: GL tag expansion.
  have row_15 : (¬ (∀ (v7 : α), ((N v7) → (¬ (add v2 v7 v3))))) := by
    simpa only [gl_preorder] using row_16
  have exists_row_15 : ∃ (v7 : α), ((N v7) ∧ (add v2 v7 v3)) := existsAndOfNotForallImpNot row_15
  obtain ⟨v7, witness_row_15⟩ := exists_row_15
  -- chapter_171_line_29: GL tag disintegration.
  have row_29 : (N v7) := by
    exact witness_row_15.1
  -- chapter_171_line_14: GL tag disintegration.
  have row_14 : (add v2 v7 v3) := by
    exact witness_row_15.2
  -- chapter_171_line_10: GL tag expansion for integration.
  have row_10 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_171_line_9: GL tag reformulation for integration and.
  have row_9 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_171_line_6: GL tag task formulation.
  have row_6 : (gl_AnchorGauss N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_171_line_12: GL tag expansion.
  have row_12 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorGauss] using row_6
  -- chapter_171_line_13: GL tag disintegration.
  have row_13 : (succ zero one) := by
    exact row_12.1.1.2
  -- chapter_171_line_11: GL tag disintegration.
  have row_11 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_12.1.1.1
  -- chapter_171_line_27: GL tag expansion.
  have row_27 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_11
  -- chapter_171_line_26: GL tag disintegration.
  have row_26 : (gl_fXYZ add N N N) := by
    exact row_27.1.1.1.1.1.1.1.1.2
  -- chapter_171_line_25: GL tag expansion.
  have row_25 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_26
  -- chapter_171_line_24: GL tag disintegration.
  have row_24 : (gl_implication13 N N N add) := by
    exact row_25.1.2
  -- chapter_171_line_23: GL tag expansion.
  have row_23 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication13] using row_24
  -- chapter_171_line_22: GL tag implication.
  have row_22 : (gl_existence1 N v7 v8 add) := by
    apply row_23
    exact row_29
    exact row_28
  -- chapter_171_line_21: GL tag expansion.
  have row_21 : (¬ (∀ (v5 : α), ((N v5) → (¬ (add v7 v8 v5))))) := by
    simpa only [gl_existence1] using row_22
  have exists_row_21 : ∃ (v5 : α), ((N v5) ∧ (add v7 v8 v5)) := existsAndOfNotForallImpNot row_21
  obtain ⟨v5, witness_row_21⟩ := exists_row_21
  -- chapter_171_line_30: GL tag disintegration.
  have row_30 : (N v5) := by
    exact witness_row_21.1
  -- chapter_171_line_20: GL tag disintegration.
  have row_20 : (add v7 v8 v5) := by
    exact witness_row_21.2
  -- chapter_171_line_8: GL tag implication.
  have row_8 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_9
    exact row_11
    exact row_13
  have rule_row_7 := peano_source_007 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_171_line_7: GL tag implication.
  have row_7 : (add v5 v1 v3) := by
    apply rule_row_7
    exact row_14
    exact row_20
    exact row_17
  -- chapter_171_line_5: GL tag theorem.
  have row_5 := gauss_source_064 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_171_line_4: GL tag implication.
  have row_4 : (add v1 v5 v3) := by
    apply row_5
    exact row_7
  -- chapter_171_line_3: GL tag expansion for integration.
  have row_3 : ((gl_preorder N add v1 v3) ↔ (¬ (∀ (v6 : α), ((N v6) → (¬ (add v1 v6 v3)))))) := by
    exact Iff.rfl
  -- chapter_171_line_2: GL tag reformulation for integration >[bound].
  have row_2 : (∀ (v4 : α), ((N v4) → ((add v1 v4 v3) → (gl_preorder N add v1 v3)))) := by
    intro v4
    intro integration_premise_1
    intro integration_premise_2
    apply (row_3).2
    intro universal_counterexample
    exact universal_counterexample v4 integration_premise_1 integration_premise_2
  -- chapter_171_line_1: GL tag implication.
  have row_1 : (gl_preorder N add v1 v3) := by
    apply row_2
    exact row_30
    exact row_4
  exact row_1

theorem gauss_source_092
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorGauss N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (V1 : GLSet α), ((gl_interval N add zero one V1) → (V1 one))) := by
  intro V1
  intro premise_1
  -- chapter_172_line_7: GL tag variable copy.
  have row_7 : (one = one) := by
    rfl
  -- chapter_172_line_8: GL tag symmetry of equality.
  have row_8 : (one = one) := by
    exact Eq.symm row_7
  -- chapter_172_line_5: GL tag task formulation.
  have row_5 : (gl_interval N add zero one V1) := by
    exact premise_1
  -- chapter_172_line_6: GL tag equality1.
  have row_6 : (gl_interval N add zero one V1) := by
    have equality_source := row_5
    have equality_step_1 := row_7
    cases equality_step_1
    exact equality_source
  -- chapter_172_line_4: GL tag task formulation.
  have row_4 : (gl_AnchorGauss N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_172_line_3: GL tag theorem.
  have row_3 := gauss_source_089 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_172_line_2: GL tag implication.
  have row_2 : (V1 one) := by
    apply row_3
    exact row_6
    exact row_5
  -- chapter_172_line_1: GL tag equality1.
  have row_1 : (V1 one) := by
    have equality_source := row_2
    have equality_step_1 := row_8
    cases equality_step_1
    exact equality_source
  exact row_1

theorem gauss_source_077
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorGauss N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (V1 : GLSet α), ((gl_interval N add zero v2 V1) → (gl_existence9 N add V1 v1 zero))))) := by
  have reformulationSource := gauss_source_074 N zero succ add mul one two identity anchor relationalInduction
  have defined_output_totality : (∀ (V1 : GLSet α) (v1 : α), (gl_limitSet N add V1 v1 (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 v1)))) := by
    simp only [gl_limitSet, gl_implication41, gl_implication27, gl_implication42]
    intro V1 v1
    constructor
    · constructor
      · intro value witness
        exact witness.1
      · intro value witness
        exact witness.2
    · intro value member ordering
      exact ⟨member, ordering⟩
  -- chapter_157_line_1: GL tag reformulated from.
  have row_1 : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (V1 : GLSet α), ((gl_interval N add zero v2 V1) → (gl_existence9 N add V1 v1 zero))))) := by
    classical
    intro v1
    intro v2
    intro reformulation_premise_1
    intro V1
    intro reformulation_premise_2
    simp only [gl_existence9]
    intro no_defined_output
    have defined_output : (gl_limitSet N add V1 v1 (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 v1))) := defined_output_totality V1 v1
    have output_result : (gl_interval N add zero v1 (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 v1))) := by
      exact reformulationSource v1 v2 reformulation_premise_1 V1 (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 v1)) defined_output reformulation_premise_2
    exact no_defined_output (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 v1)) defined_output output_result
  exact row_1

theorem gauss_source_078
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorGauss N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (V1 : GLSet α), ((gl_interval N add zero one V1) → (gl_existence10 N add V1 v1 zero v2))))) := by
  have reformulationSource := gauss_source_075 N zero succ add mul one two identity anchor relationalInduction
  have defined_output_totality : (∀ (V1 : GLSet α) (v1 : α), (gl_limitSet N add V1 v1 (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 v1)))) := by
    simp only [gl_limitSet, gl_implication41, gl_implication27, gl_implication42]
    intro V1 v1
    constructor
    · constructor
      · intro value witness
        exact witness.1
      · intro value witness
        exact witness.2
    · intro value member ordering
      exact ⟨member, ordering⟩
  -- chapter_158_line_1: GL tag reformulated from.
  have row_1 : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (V1 : GLSet α), ((gl_interval N add zero one V1) → (gl_existence10 N add V1 v1 zero v2))))) := by
    classical
    intro v1
    intro v2
    intro reformulation_premise_1
    intro V1
    intro reformulation_premise_2
    simp only [gl_existence10]
    intro no_defined_output
    have defined_output : (gl_limitSet N add V1 v1 (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 v1))) := defined_output_totality V1 v1
    have output_result : (¬ (gl_interval N add zero v2 (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 v1)))) := by
      exact reformulationSource v1 v2 reformulation_premise_1 V1 (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 v1)) defined_output reformulation_premise_2
    exact no_defined_output (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 v1)) defined_output output_result
  exact row_1

theorem gauss_source_080
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorGauss N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (V1 : GLSet α), ((gl_interval N add zero v1 V1) → (gl_existence11 N add V1 v1 zero v2))))) := by
  have reformulationSource := gauss_source_076 N zero succ add mul one two identity anchor relationalInduction
  have defined_output_totality : (∀ (V1 : GLSet α) (v1 : α), (gl_limitSet N add V1 v1 (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 v1)))) := by
    simp only [gl_limitSet, gl_implication41, gl_implication27, gl_implication42]
    intro V1 v1
    constructor
    · constructor
      · intro value witness
        exact witness.1
      · intro value witness
        exact witness.2
    · intro value member ordering
      exact ⟨member, ordering⟩
  -- chapter_160_line_1: GL tag reformulated from.
  have row_1 : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (V1 : GLSet α), ((gl_interval N add zero v1 V1) → (gl_existence11 N add V1 v1 zero v2))))) := by
    classical
    intro v1
    intro v2
    intro reformulation_premise_1
    intro V1
    intro reformulation_premise_2
    simp only [gl_existence11]
    intro no_defined_output
    have defined_output : (gl_limitSet N add V1 v1 (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 v1))) := defined_output_totality V1 v1
    have output_result : (¬ (gl_interval N add zero v2 (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 v1)))) := by
      exact reformulationSource v1 v2 reformulation_premise_1 V1 (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 v1)) defined_output reformulation_premise_2
    exact no_defined_output (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 v1)) defined_output output_result
  exact row_1

theorem gauss_source_081
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorGauss N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (V1 : GLSet α), ((gl_interval N add zero v1 V1) → (gl_existence12 N add V1 v2 zero))))) := by
  have reformulationSource := gauss_source_070 N zero succ add mul one two identity anchor relationalInduction
  have defined_output_totality : (∀ (V1 : GLSet α) (v2 : α), (gl_limitSet N add V1 v2 (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 v2)))) := by
    simp only [gl_limitSet, gl_implication41, gl_implication27, gl_implication42]
    intro V1 v2
    constructor
    · constructor
      · intro value witness
        exact witness.1
      · intro value witness
        exact witness.2
    · intro value member ordering
      exact ⟨member, ordering⟩
  -- chapter_161_line_1: GL tag reformulated from.
  have row_1 : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (V1 : GLSet α), ((gl_interval N add zero v1 V1) → (gl_existence12 N add V1 v2 zero))))) := by
    classical
    intro v1
    intro v2
    intro reformulation_premise_1
    intro V1
    intro reformulation_premise_2
    simp only [gl_existence12]
    intro no_defined_output
    have defined_output : (gl_limitSet N add V1 v2 (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 v2))) := defined_output_totality V1 v2
    have output_result : (¬ (gl_interval N add zero v2 (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 v2)))) := by
      exact reformulationSource v1 v2 reformulation_premise_1 V1 (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 v2)) defined_output reformulation_premise_2
    exact no_defined_output (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 v2)) defined_output output_result
  exact row_1

theorem gauss_source_082
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorGauss N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (V1 : GLSet α), ((gl_interval N add zero v1 V1) → (gl_existence13 N add V1 one zero v2))))) := by
  have reformulationSource := gauss_source_072 N zero succ add mul one two identity anchor relationalInduction
  have defined_output_totality : (∀ (V1 : GLSet α), (gl_limitSet N add V1 one (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 one)))) := by
    simp only [gl_limitSet, gl_implication41, gl_implication27, gl_implication42]
    intro V1
    constructor
    · constructor
      · intro value witness
        exact witness.1
      · intro value witness
        exact witness.2
    · intro value member ordering
      exact ⟨member, ordering⟩
  -- chapter_162_line_1: GL tag reformulated from.
  have row_1 : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (V1 : GLSet α), ((gl_interval N add zero v1 V1) → (gl_existence13 N add V1 one zero v2))))) := by
    classical
    intro v1
    intro v2
    intro reformulation_premise_1
    intro V1
    intro reformulation_premise_2
    simp only [gl_existence13]
    intro no_defined_output
    have defined_output : (gl_limitSet N add V1 one (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 one))) := defined_output_totality V1
    have output_result : (¬ (gl_interval N add zero v2 (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 one)))) := by
      exact reformulationSource v1 v2 reformulation_premise_1 V1 (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 one)) defined_output reformulation_premise_2
    exact no_defined_output (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 one)) defined_output output_result
  exact row_1

theorem gauss_source_083
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorGauss N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (V1 : GLSet α), ((gl_interval N add zero v1 V1) → (gl_existence14 N add V1 two zero v2))))) := by
  have reformulationSource := gauss_source_073 N zero succ add mul one two identity anchor relationalInduction
  have defined_output_totality : (∀ (V1 : GLSet α), (gl_limitSet N add V1 two (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 two)))) := by
    simp only [gl_limitSet, gl_implication41, gl_implication27, gl_implication42]
    intro V1
    constructor
    · constructor
      · intro value witness
        exact witness.1
      · intro value witness
        exact witness.2
    · intro value member ordering
      exact ⟨member, ordering⟩
  -- chapter_163_line_1: GL tag reformulated from.
  have row_1 : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (V1 : GLSet α), ((gl_interval N add zero v1 V1) → (gl_existence14 N add V1 two zero v2))))) := by
    classical
    intro v1
    intro v2
    intro reformulation_premise_1
    intro V1
    intro reformulation_premise_2
    simp only [gl_existence14]
    intro no_defined_output
    have defined_output : (gl_limitSet N add V1 two (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 two))) := defined_output_totality V1
    have output_result : (¬ (gl_interval N add zero v2 (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 two)))) := by
      exact reformulationSource v1 v2 reformulation_premise_1 V1 (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 two)) defined_output reformulation_premise_2
    exact no_defined_output (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 two)) defined_output output_result
  exact row_1

theorem gauss_source_084
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorGauss N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (V1 : GLSet α), ((gl_interval N add zero v1 V1) → (gl_existence8 N add V1 zero v2))))) := by
  have reformulationSource := gauss_source_071 N zero succ add mul one two identity anchor relationalInduction
  have defined_output_totality : (∀ (V1 : GLSet α), (gl_limitSet N add V1 zero (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 zero)))) := by
    simp only [gl_limitSet, gl_implication41, gl_implication27, gl_implication42]
    intro V1
    constructor
    · constructor
      · intro value witness
        exact witness.1
      · intro value witness
        exact witness.2
    · intro value member ordering
      exact ⟨member, ordering⟩
  -- chapter_164_line_1: GL tag reformulated from.
  have row_1 : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (V1 : GLSet α), ((gl_interval N add zero v1 V1) → (gl_existence8 N add V1 zero v2))))) := by
    classical
    intro v1
    intro v2
    intro reformulation_premise_1
    intro V1
    intro reformulation_premise_2
    simp only [gl_existence8]
    intro no_defined_output
    have defined_output : (gl_limitSet N add V1 zero (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 zero))) := defined_output_totality V1
    have output_result : (¬ (gl_interval N add zero v2 (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 zero)))) := by
      exact reformulationSource v1 v2 reformulation_premise_1 V1 (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 zero)) defined_output reformulation_premise_2
    exact no_defined_output (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 zero)) defined_output output_result
  exact row_1

theorem gauss_source_069
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorGauss N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (V1 : GLBinaryRelation α) (V2 : GLBinaryRelation α), ((gl_limitSequence N add v1 V1 V2) → ((gl_sequence N add zero v2 V1) → (gl_sequence N add zero v1 V2)))))) := by
  intro v1
  intro v2
  intro premise_1
  intro V1
  intro V2
  intro premise_2
  intro premise_3
  -- chapter_149_line_70: GL tag expansion for integration.
  have row_70 : (∀ (V4 : GLSet α), ((gl_implication5 V4 V2) ↔ (∀ (v16 : α), ((V4 v16) → (∀ (v13 : α), ((V2 v16 v13) → (∀ (v14 : α), ((V2 v16 v14) → (v13 = v14))))))))) := by
    intro V4
    exact Iff.rfl
  -- chapter_149_line_74: GL tag premise element.
  have row_74 : (∀ (V4 : GLSet α) (v16 : α), ((V4 v16) → (∀ (v13 : α), ((V2 v16 v13) → (∀ (v14 : α), ((V2 v16 v14) → (V4 v16))))))) := by
    intro V4
    intro v16
    intro scope_premise_1
    intro v13
    intro scope_premise_2
    intro v14
    intro scope_premise_3
    exact scope_premise_1
  -- chapter_149_line_72: GL tag premise element.
  have row_72 : (∀ (V4 : GLSet α) (v16 : α), ((V4 v16) → (∀ (v13 : α), ((V2 v16 v13) → (∀ (v14 : α), ((V2 v16 v14) → (V2 v16 v14))))))) := by
    intro V4
    intro v16
    intro scope_premise_1
    intro v13
    intro scope_premise_2
    intro v14
    intro scope_premise_3
    exact scope_premise_3
  -- chapter_149_line_69: GL tag premise element.
  have row_69 : (∀ (V4 : GLSet α) (v16 : α), ((V4 v16) → (∀ (v13 : α), ((V2 v16 v13) → (∀ (v14 : α), ((V2 v16 v14) → (V2 v16 v13))))))) := by
    intro V4
    intro v16
    intro scope_premise_1
    intro v13
    intro scope_premise_2
    intro v14
    intro scope_premise_3
    exact scope_premise_2
  -- chapter_149_line_59: GL tag expansion for integration.
  have row_59 : (∀ (V4 : GLSet α), ((gl_implication4 V4 N V2) ↔ (∀ (v9 : α), ((V4 v9) → (gl_existence0 N v9 V2))))) := by
    intro V4
    exact Iff.rfl
  -- chapter_149_line_58: GL tag premise element.
  have row_58 : (∀ (V4 : GLSet α) (v9 : α), ((V4 v9) → (V4 v9))) := by
    intro V4
    intro v9
    intro scope_premise_1
    exact scope_premise_1
  -- chapter_149_line_46: GL tag expansion for integration.
  have row_46 : (∀ (v9 : α), ((gl_existence0 N v9 V2) ↔ (¬ (∀ (v12 : α), ((N v12) → (¬ (V2 v9 v12))))))) := by
    intro v9
    exact Iff.rfl
  -- chapter_149_line_45: GL tag reformulation for integration >[bound].
  have row_45 : (∀ (V4 : GLSet α) (v9 : α), ((V4 v9) → (∀ (v10 : α), ((N v10) → ((V2 v9 v10) → (gl_existence0 N v9 V2)))))) := by
    intro V4
    intro v9
    intro scope_premise_1
    intro v10
    intro integration_premise_1
    intro integration_premise_2
    apply (row_46 v9).2
    intro universal_counterexample
    exact universal_counterexample v10 integration_premise_1 integration_premise_2
  -- chapter_149_line_42: GL tag expansion for integration.
  have row_42 : ((gl_implication1 V2 N) ↔ (∀ (v8 : α) (v7 : α), ((V2 v8 v7) → (N v7)))) := by
    exact Iff.rfl
  -- chapter_149_line_41: GL tag premise element.
  have row_41 : (∀ (v8 : α) (v7 : α), ((V2 v8 v7) → (V2 v8 v7))) := by
    intro v8
    intro v7
    intro scope_premise_1
    exact scope_premise_1
  -- chapter_149_line_32: GL tag expansion for integration.
  have row_32 : (∀ (V4 : GLSet α), ((gl_implication0 V2 V4) ↔ (∀ (v3 : α) (v6 : α), ((V2 v3 v6) → (V4 v3))))) := by
    intro V4
    exact Iff.rfl
  -- chapter_149_line_31: GL tag premise element.
  have row_31 : (∀ (V4 : GLSet α) (v3 : α) (v6 : α), ((V2 v3 v6) → (V2 v3 v6))) := by
    intro V4
    intro v3
    intro v6
    intro scope_premise_1
    exact scope_premise_1
  -- chapter_149_line_30: GL tag task formulation.
  have row_30 : (gl_limitSequence N add v1 V1 V2) := by
    exact premise_2
  -- chapter_149_line_29: GL tag expansion.
  have row_29 : (((gl_implication38 V2 N add v1) ∧ (gl_implication39 V2 V1)) ∧ (gl_implication40 N add v1 V1 V2)) := by
    simpa only [gl_limitSequence] using row_30
  -- chapter_149_line_49: GL tag disintegration.
  have row_49 : (gl_implication40 N add v1 V1 V2) := by
    exact row_29.2
  -- chapter_149_line_48: GL tag expansion.
  have row_48 : (∀ (w1 : α), ((gl_preorder N add w1 v1) → (∀ (w2 : α), ((V1 w1 w2) → (V2 w1 w2))))) := by
    simpa only [gl_implication40] using row_49
  -- chapter_149_line_35: GL tag disintegration.
  have row_35 : (gl_implication38 V2 N add v1) := by
    exact row_29.1.1
  -- chapter_149_line_34: GL tag expansion.
  have row_34 : (∀ (w1 : α) (w2 : α), ((V2 w1 w2) → (gl_preorder N add w1 v1))) := by
    simpa only [gl_implication38] using row_35
  -- chapter_149_line_33: GL tag implication.
  have row_33 : (∀ (V4 : GLSet α) (v3 : α) (v6 : α), ((V2 v3 v6) → (gl_preorder N add v3 v1))) := by
    intro V4
    intro v3
    intro v6
    intro scope_premise_1
    have scoped_fact_2 := row_31 V4 v3 v6 scope_premise_1
    apply row_34
    exact scoped_fact_2
  -- chapter_149_line_28: GL tag disintegration.
  have row_28 : (gl_implication39 V2 V1) := by
    exact row_29.1.2
  -- chapter_149_line_27: GL tag expansion.
  have row_27 : (∀ (w1 : α) (w2 : α), ((V2 w1 w2) → (V1 w1 w2))) := by
    simpa only [gl_implication39] using row_28
  -- chapter_149_line_71: GL tag implication.
  have row_71 : (∀ (V4 : GLSet α) (v16 : α), ((V4 v16) → (∀ (v13 : α), ((V2 v16 v13) → (∀ (v14 : α), ((V2 v16 v14) → (V1 v16 v14))))))) := by
    intro V4
    intro v16
    intro scope_premise_1
    intro v13
    intro scope_premise_2
    intro v14
    intro scope_premise_3
    have scoped_fact_2 := row_72 V4 v16 scope_premise_1 v13 scope_premise_2 v14 scope_premise_3
    apply row_27
    exact scoped_fact_2
  -- chapter_149_line_68: GL tag implication.
  have row_68 : (∀ (V4 : GLSet α) (v16 : α), ((V4 v16) → (∀ (v13 : α), ((V2 v16 v13) → (∀ (v14 : α), ((V2 v16 v14) → (V1 v16 v13))))))) := by
    intro V4
    intro v16
    intro scope_premise_1
    intro v13
    intro scope_premise_2
    intro v14
    intro scope_premise_3
    have scoped_fact_2 := row_69 V4 v16 scope_premise_1 v13 scope_premise_2 v14 scope_premise_3
    apply row_27
    exact scoped_fact_2
  -- chapter_149_line_40: GL tag implication.
  have row_40 : (∀ (v8 : α) (v7 : α), ((V2 v8 v7) → (V1 v8 v7))) := by
    intro v8
    intro v7
    intro scope_premise_1
    have scoped_fact_2 := row_41 v8 v7 scope_premise_1
    apply row_27
    exact scoped_fact_2
  -- chapter_149_line_26: GL tag implication.
  have row_26 : (∀ (V4 : GLSet α) (v3 : α) (v6 : α), ((V2 v3 v6) → (V1 v3 v6))) := by
    intro V4
    intro v3
    intro v6
    intro scope_premise_1
    have scoped_fact_2 := row_31 V4 v3 v6 scope_premise_1
    apply row_27
    exact scoped_fact_2
  -- chapter_149_line_20: GL tag task formulation.
  have row_20 : (gl_sequence N add zero v2 V1) := by
    exact premise_3
  -- chapter_149_line_19: GL tag expansion.
  have row_19 : (¬ (∀ (V6 : GLSet α), ((gl_interval N add zero v2 V6) → (¬ (gl_fXY V1 V6 N))))) := by
    simpa only [gl_sequence] using row_20
  have exists_row_19 : ∃ (V6 : GLSet α), ((gl_interval N add zero v2 V6) ∧ (gl_fXY V1 V6 N)) := existsAndOfNotForallImpNot row_19
  obtain ⟨V6, witness_row_19⟩ := exists_row_19
  -- chapter_149_line_25: GL tag disintegration.
  have row_25 : (gl_fXY V1 V6 N) := by
    exact witness_row_19.2
  -- chapter_149_line_24: GL tag expansion.
  have row_24 : ((((gl_implication0 V1 V6) ∧ (gl_implication1 V1 N)) ∧ (gl_implication4 V6 N V1)) ∧ (gl_implication5 V6 V1)) := by
    simpa only [gl_fXY] using row_25
  -- chapter_149_line_67: GL tag disintegration.
  have row_67 : (gl_implication5 V6 V1) := by
    exact row_24.2
  -- chapter_149_line_66: GL tag expansion.
  have row_66 : (∀ (w1 : α), ((V6 w1) → (∀ (w2 : α), ((V1 w1 w2) → (∀ (w3 : α), ((V1 w1 w3) → (w2 = w3))))))) := by
    simpa only [gl_implication5] using row_67
  -- chapter_149_line_54: GL tag disintegration.
  have row_54 : (gl_implication4 V6 N V1) := by
    exact row_24.1.2
  -- chapter_149_line_53: GL tag expansion.
  have row_53 : (∀ (w1 : α), ((V6 w1) → (gl_existence0 N w1 V1))) := by
    simpa only [gl_implication4] using row_54
  -- chapter_149_line_39: GL tag disintegration.
  have row_39 : (gl_implication1 V1 N) := by
    exact row_24.1.1.2
  -- chapter_149_line_38: GL tag expansion.
  have row_38 : (∀ (w1 : α) (w2 : α), ((V1 w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using row_39
  -- chapter_149_line_37: GL tag implication.
  have row_37 : (∀ (v8 : α) (v7 : α), ((V2 v8 v7) → (N v7))) := by
    intro v8
    intro v7
    intro scope_premise_1
    have scoped_fact_2 := row_40 v8 v7 scope_premise_1
    apply row_38
    exact scoped_fact_2
  -- chapter_149_line_36: GL tag validity name.
  have row_36 : (gl_implication1 V2 N) := by
    simpa only [gl_implication1] using row_37
  -- chapter_149_line_23: GL tag disintegration.
  have row_23 : (gl_implication0 V1 V6) := by
    exact row_24.1.1.1
  -- chapter_149_line_22: GL tag expansion.
  have row_22 : (∀ (w1 : α) (w2 : α), ((V1 w1 w2) → (V6 w1))) := by
    simpa only [gl_implication0] using row_23
  -- chapter_149_line_21: GL tag implication.
  have row_21 : (∀ (V4 : GLSet α) (v3 : α) (v6 : α), ((V2 v3 v6) → (V6 v3))) := by
    intro V4
    intro v3
    intro v6
    intro scope_premise_1
    have scoped_fact_2 := row_26 V4 v3 v6 scope_premise_1
    apply row_22
    exact scoped_fact_2
  -- chapter_149_line_18: GL tag disintegration.
  have row_18 : (gl_interval N add zero v2 V6) := by
    exact witness_row_19.1
  -- chapter_149_line_17: GL tag task formulation.
  have row_17 : (succ v1 v2) := by
    exact premise_1
  -- chapter_149_line_16: GL tag task formulation.
  have row_16 : (gl_AnchorGauss N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_149_line_15: GL tag theorem.
  have row_15 := gauss_source_077 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_149_line_14: GL tag implication.
  have row_14 : (gl_existence9 N add V6 v1 zero) := by
    apply row_15
    exact row_17
    exact row_18
  -- chapter_149_line_13: GL tag expansion.
  have row_13 : (¬ (∀ (V4 : GLSet α), ((gl_limitSet N add V6 v1 V4) → (¬ (gl_interval N add zero v1 V4))))) := by
    simpa only [gl_existence9] using row_14
  have exists_row_13 : ∃ (V4 : GLSet α), ((gl_limitSet N add V6 v1 V4) ∧ (gl_interval N add zero v1 V4)) := existsAndOfNotForallImpNot row_13
  obtain ⟨V4, witness_row_13⟩ := exists_row_13
  -- chapter_149_line_75: GL tag disintegration.
  have row_75 : (gl_interval N add zero v1 V4) := by
    exact witness_row_13.2
  -- chapter_149_line_12: GL tag disintegration.
  have row_12 : (gl_limitSet N add V6 v1 V4) := by
    exact witness_row_13.1
  -- chapter_149_line_11: GL tag expansion.
  have row_11 : (((gl_implication41 V4 V6) ∧ (gl_implication27 V4 N add v1)) ∧ (gl_implication42 V6 N add v1 V4)) := by
    simpa only [gl_limitSet] using row_12
  -- chapter_149_line_62: GL tag disintegration.
  have row_62 : (gl_implication27 V4 N add v1) := by
    exact row_11.1.2
  -- chapter_149_line_61: GL tag expansion.
  have row_61 : (∀ (w1 : α), ((V4 w1) → (gl_preorder N add w1 v1))) := by
    simpa only [gl_implication27] using row_62
  -- chapter_149_line_60: GL tag implication.
  have row_60 : (∀ (v9 : α), ((V4 v9) → (gl_preorder N add v9 v1))) := by
    intro v9
    intro scope_premise_1
    have scoped_fact_2 := row_58 V4 v9 scope_premise_1
    apply row_61
    exact scoped_fact_2
  -- chapter_149_line_57: GL tag disintegration.
  have row_57 : (gl_implication41 V4 V6) := by
    exact row_11.1.1
  -- chapter_149_line_56: GL tag expansion.
  have row_56 : (∀ (w1 : α), ((V4 w1) → (V6 w1))) := by
    simpa only [gl_implication41] using row_57
  -- chapter_149_line_73: GL tag implication.
  have row_73 : (∀ (v16 : α), ((V4 v16) → (∀ (v13 : α), ((V2 v16 v13) → (∀ (v14 : α), ((V2 v16 v14) → (V6 v16))))))) := by
    intro v16
    intro scope_premise_1
    intro v13
    intro scope_premise_2
    intro v14
    intro scope_premise_3
    have scoped_fact_2 := row_74 V4 v16 scope_premise_1 v13 scope_premise_2 v14 scope_premise_3
    apply row_56
    exact scoped_fact_2
  -- chapter_149_line_65: GL tag implication.
  have row_65 : (∀ (v16 : α), ((V4 v16) → (∀ (v13 : α), ((V2 v16 v13) → (∀ (v14 : α), ((V2 v16 v14) → (v13 = v14))))))) := by
    intro v16
    intro scope_premise_1
    intro v13
    intro scope_premise_2
    intro v14
    intro scope_premise_3
    have scoped_fact_2 := row_73 v16 scope_premise_1 v13 scope_premise_2 v14 scope_premise_3
    have scoped_fact_3 := row_68 V4 v16 scope_premise_1 v13 scope_premise_2 v14 scope_premise_3
    have scoped_fact_4 := row_71 V4 v16 scope_premise_1 v13 scope_premise_2 v14 scope_premise_3
    apply row_66
    exact scoped_fact_2
    exact scoped_fact_3
    exact scoped_fact_4
  -- chapter_149_line_64: GL tag validity name.
  have row_64 : (gl_implication5 V4 V2) := by
    simpa only [gl_implication5] using row_65
  -- chapter_149_line_55: GL tag implication.
  have row_55 : (∀ (v9 : α), ((V4 v9) → (V6 v9))) := by
    intro v9
    intro scope_premise_1
    have scoped_fact_2 := row_58 V4 v9 scope_premise_1
    apply row_56
    exact scoped_fact_2
  -- chapter_149_line_52: GL tag implication.
  have row_52 : (∀ (v9 : α), ((V4 v9) → (gl_existence0 N v9 V1))) := by
    intro v9
    intro scope_premise_1
    have scoped_fact_2 := row_55 v9 scope_premise_1
    apply row_53
    exact scoped_fact_2
  -- chapter_149_line_51: GL tag expansion.
  have row_51 : (∀ (v9 : α), ((V4 v9) → (¬ (∀ (v11 : α), ((N v11) → (¬ (V1 v9 v11))))))) := by
    simpa only [gl_existence0] using row_52
  -- chapter_149_line_63: GL tag disintegration.
  have row_63 : (∀ (v9 : α), ((V4 v9) → (∃ (v11 : α), ((N v11) ∧ (V1 v9 v11))))) := by
    intro v9
    intro scope_premise_1
    have scoped_fact_1 := row_51 v9 scope_premise_1
    obtain ⟨v11, scoped_witness_bundle⟩ := existsAndOfNotForallImpNot scoped_fact_1
    exact ⟨v11, ⟨scoped_witness_bundle.1, scoped_witness_bundle.2⟩⟩
  -- chapter_149_line_50: GL tag disintegration.
  have row_50 : (∀ (v9 : α), ((V4 v9) → (∃ (v11 : α), ((N v11) ∧ (V1 v9 v11))))) := by
    intro v9
    intro scope_premise_1
    have scoped_fact_1 := row_51 v9 scope_premise_1
    obtain ⟨v11, scoped_witness_bundle⟩ := existsAndOfNotForallImpNot scoped_fact_1
    exact ⟨v11, ⟨scoped_witness_bundle.1, scoped_witness_bundle.2⟩⟩
  -- chapter_149_line_47: GL tag implication.
  have row_47 : (∀ (v9 : α), ((V4 v9) → (∀ (v11 : α), (((N v11) ∧ (V1 v9 v11)) → (V2 v9 v11))))) := by
    intro v9
    intro scope_premise_1
    intro v11
    intro witness_guard_1
    have scoped_fact_3 := row_50 v9 scope_premise_1
    have scoped_fact_2 := row_60 v9 scope_premise_1
    apply row_48
    exact scoped_fact_2
    exact witness_guard_1.2
  -- chapter_149_line_44: GL tag implication.
  have row_44 : (∀ (v9 : α), ((V4 v9) → (gl_existence0 N v9 V2))) := by
    intro v9
    intro scope_premise_1
    obtain ⟨v11, witness_guard_1⟩ := row_63 v9 scope_premise_1
    have scoped_fact_1 := row_45 V4 v9 scope_premise_1
    have scoped_fact_3 := row_47 v9 scope_premise_1 v11 witness_guard_1
    apply scoped_fact_1
    exact witness_guard_1.1
    exact scoped_fact_3
  -- chapter_149_line_43: GL tag validity name.
  have row_43 : (gl_implication4 V4 N V2) := by
    simpa only [gl_implication4] using row_44
  -- chapter_149_line_10: GL tag disintegration.
  have row_10 : (gl_implication42 V6 N add v1 V4) := by
    exact row_11.2
  -- chapter_149_line_9: GL tag expansion.
  have row_9 : (∀ (w1 : α), ((V6 w1) → ((gl_preorder N add w1 v1) → (V4 w1)))) := by
    simpa only [gl_implication42] using row_10
  -- chapter_149_line_8: GL tag implication.
  have row_8 : (∀ (v3 : α) (v6 : α), ((V2 v3 v6) → (V4 v3))) := by
    intro v3
    intro v6
    intro scope_premise_1
    have scoped_fact_2 := row_21 V4 v3 v6 scope_premise_1
    have scoped_fact_3 := row_33 V4 v3 v6 scope_premise_1
    apply row_9
    exact scoped_fact_2
    exact scoped_fact_3
  -- chapter_149_line_7: GL tag validity name.
  have row_7 : (gl_implication0 V2 V4) := by
    simpa only [gl_implication0] using row_8
  -- chapter_149_line_6: GL tag expansion for integration.
  have row_6 : ((gl_fXY V2 V4 N) ↔ ((((gl_implication0 V2 V4) ∧ (gl_implication1 V2 N)) ∧ (gl_implication4 V4 N V2)) ∧ (gl_implication5 V4 V2))) := by
    exact Iff.rfl
  -- chapter_149_line_5: GL tag reformulation for integration and.
  have row_5 : ((gl_implication0 V2 V4) → ((gl_implication1 V2 N) → ((gl_implication4 V4 N V2) → ((gl_implication5 V4 V2) → (gl_fXY V2 V4 N))))) := by
    intro integration_premise_1
    intro integration_premise_2
    intro integration_premise_3
    intro integration_premise_4
    simp only [gl_fXY]
    exact ⟨⟨⟨integration_premise_1, integration_premise_2⟩, integration_premise_3⟩, integration_premise_4⟩
  -- chapter_149_line_4: GL tag implication.
  have row_4 : (gl_fXY V2 V4 N) := by
    apply row_5
    exact row_7
    exact row_36
    exact row_43
    exact row_64
  -- chapter_149_line_3: GL tag expansion for integration.
  have row_3 : ((gl_sequence N add zero v1 V2) ↔ (¬ (∀ (V5 : GLSet α), ((gl_interval N add zero v1 V5) → (¬ (gl_fXY V2 V5 N)))))) := by
    exact Iff.rfl
  -- chapter_149_line_2: GL tag reformulation for integration >[bound].
  have row_2 : (∀ (V3 : GLSet α), ((gl_interval N add zero v1 V3) → ((gl_fXY V2 V3 N) → (gl_sequence N add zero v1 V2)))) := by
    intro V3
    intro integration_premise_1
    intro integration_premise_2
    apply (row_3).2
    intro universal_counterexample
    exact universal_counterexample V3 integration_premise_1 integration_premise_2
  -- chapter_149_line_1: GL tag implication.
  have row_1 : (gl_sequence N add zero v1 V2) := by
    apply row_2
    exact row_75
    exact row_4
  exact row_1

theorem gauss_source_086
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorGauss N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (V1 : GLBinaryRelation α), ((gl_sequence N add zero v2 V1) → (gl_existence15 N add v1 V1 zero))))) := by
  have reformulationSource := gauss_source_069 N zero succ add mul one two identity anchor relationalInduction
  have defined_output_totality : (∀ (v1 : α) (V1 : GLBinaryRelation α), (gl_limitSequence N add v1 V1 (fun (x_1 : α) (x_2 : α) => (gl_preorder N add x_1 v1) ∧ (V1 x_1 x_2)))) := by
    simp only [gl_limitSequence, gl_implication38, gl_implication39, gl_implication40]
    intro v1 V1
    constructor
    · constructor
      · intro first second witness
        exact witness.1
      · intro first second witness
        exact witness.2
    · intro first ordering second member
      exact ⟨ordering, member⟩
  -- chapter_166_line_1: GL tag reformulated from.
  have row_1 : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (V1 : GLBinaryRelation α), ((gl_sequence N add zero v2 V1) → (gl_existence15 N add v1 V1 zero))))) := by
    classical
    intro v1
    intro v2
    intro reformulation_premise_1
    intro V1
    intro reformulation_premise_2
    simp only [gl_existence15]
    intro no_defined_output
    have defined_output : (gl_limitSequence N add v1 V1 (fun (x_1 : α) (x_2 : α) => (gl_preorder N add x_1 v1) ∧ (V1 x_1 x_2))) := defined_output_totality v1 V1
    have output_result : (gl_sequence N add zero v1 (fun (x_1 : α) (x_2 : α) => (gl_preorder N add x_1 v1) ∧ (V1 x_1 x_2))) := by
      exact reformulationSource v1 v2 reformulation_premise_1 V1 (fun (x_1 : α) (x_2 : α) => (gl_preorder N add x_1 v1) ∧ (V1 x_1 x_2)) defined_output reformulation_premise_2
    exact no_defined_output (fun (x_1 : α) (x_2 : α) => (gl_preorder N add x_1 v1) ∧ (V1 x_1 x_2)) defined_output output_result
  exact row_1

private theorem gauss_source_065_induction_typing
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorGauss N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v2 : α)
    (v4 : α)
    (assumption_5 : (gl_fold N succ add identity zero v2 v4))
    : (N v2) := by
  -- chapter_141_line_5: GL tag task formulation.
  have row_5 : (gl_fold N succ add identity zero v2 v4) := by
    exact assumption_5
  -- chapter_141_line_4: GL tag expansion.
  have row_4 : (¬ (∀ (V1 : GLSet α), ((gl_interval N add zero v2 V1) → (¬ (gl_existence2 V1 N zero identity v2 v4 succ add))))) := by
    simpa only [gl_fold] using row_5
  have exists_row_4 : ∃ (V1 : GLSet α), ((gl_interval N add zero v2 V1) ∧ (gl_existence2 V1 N zero identity v2 v4 succ add)) := existsAndOfNotForallImpNot row_4
  obtain ⟨V1, witness_row_4⟩ := exists_row_4
  -- chapter_141_line_3: GL tag disintegration.
  have row_3 : (gl_interval N add zero v2 V1) := by
    exact witness_row_4.1
  -- chapter_141_line_2: GL tag expansion.
  have row_2 : (((((gl_implication26 V1 N add zero) ∧ (gl_implication27 V1 N add v2)) ∧ (gl_implication28 N add zero v2 V1)) ∧ (N zero)) ∧ (N v2)) := by
    simpa only [gl_interval] using row_3
  -- chapter_141_line_1: GL tag disintegration.
  have row_1 : (N v2) := by
    exact row_2.2
  exact row_1

private theorem gauss_source_065_check_zero
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorGauss N zero succ add mul one two identity)
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
    (assumption_112 : (succ v2 v1))
    (assumption_107 : (mul v1 v2 v3))
    (assumption_53 : (v2 = zero))
    (assumption_21 : (gl_fold N succ add identity zero v2 v4))
    : (mul two v4 v3) := by
  -- chapter_142_line_112: GL tag task formulation.
  have row_112 : (succ v2 v1) := by
    exact assumption_112
  -- chapter_142_line_107: GL tag task formulation.
  have row_107 : (mul v1 v2 v3) := by
    exact assumption_107
  -- chapter_142_line_62: GL tag theorem.
  have row_62 := gauss_source_064 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_142_line_57: GL tag expansion for integration.
  have row_57 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_142_line_56: GL tag reformulation for integration and.
  have row_56 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_142_line_53: GL tag recursion.
  have row_53 : (v2 = zero) := by
    exact assumption_53
  -- chapter_142_line_52: GL tag symmetry of equality.
  have row_52 : (zero = v2) := by
    exact Eq.symm row_53
  -- chapter_142_line_21: GL tag task formulation.
  have row_21 : (gl_fold N succ add identity zero v2 v4) := by
    exact assumption_21
  -- chapter_142_line_20: GL tag expansion.
  have row_20 : (¬ (∀ (V2 : GLSet α), ((gl_interval N add zero v2 V2) → (¬ (gl_existence2 V2 N zero identity v2 v4 succ add))))) := by
    simpa only [gl_fold] using row_21
  have exists_row_20 : ∃ (V2 : GLSet α), ((gl_interval N add zero v2 V2) ∧ (gl_existence2 V2 N zero identity v2 v4 succ add)) := existsAndOfNotForallImpNot row_20
  obtain ⟨V2, witness_row_20⟩ := exists_row_20
  -- chapter_142_line_46: GL tag disintegration.
  have row_46 : (gl_interval N add zero v2 V2) := by
    exact witness_row_20.1
  -- chapter_142_line_45: GL tag expansion.
  have row_45 : (((((gl_implication26 V2 N add zero) ∧ (gl_implication27 V2 N add v2)) ∧ (gl_implication28 N add zero v2 V2)) ∧ (N zero)) ∧ (N v2)) := by
    simpa only [gl_interval] using row_46
  -- chapter_142_line_44: GL tag disintegration.
  have row_44 : (N v2) := by
    exact row_45.2
  -- chapter_142_line_19: GL tag disintegration.
  have row_19 : (gl_existence2 V2 N zero identity v2 v4 succ add) := by
    exact witness_row_20.2
  -- chapter_142_line_18: GL tag expansion.
  have row_18 : (¬ (∀ (V1 : GLBinaryRelation α), ((gl_fXY V1 V2 N) → (¬ (gl_and0 zero identity V1 v2 v4 V2 succ add))))) := by
    simpa only [gl_existence2] using row_19
  have exists_row_18 : ∃ (V1 : GLBinaryRelation α), ((gl_fXY V1 V2 N) ∧ (gl_and0 zero identity V1 v2 v4 V2 succ add)) := existsAndOfNotForallImpNot row_18
  obtain ⟨V1, witness_row_18⟩ := exists_row_18
  -- chapter_142_line_24: GL tag disintegration.
  have row_24 : (gl_and0 zero identity V1 v2 v4 V2 succ add) := by
    exact witness_row_18.2
  -- chapter_142_line_23: GL tag expansion.
  have row_23 : (((gl_implication32 zero identity V1) ∧ (V1 v2 v4)) ∧ (gl_implication33 V2 succ V1 identity add)) := by
    simpa only [gl_and0] using row_24
  -- chapter_142_line_84: GL tag disintegration.
  have row_84 : (gl_implication32 zero identity V1) := by
    exact row_23.1.1
  -- chapter_142_line_83: GL tag expansion.
  have row_83 : (∀ (w1 : α), ((identity zero w1) → (V1 zero w1))) := by
    simpa only [gl_implication32] using row_84
  -- chapter_142_line_22: GL tag disintegration.
  have row_22 : (V1 v2 v4) := by
    exact row_23.1.2
  -- chapter_142_line_97: GL tag equality1.
  have row_97 : (V1 zero v4) := by
    have equality_source := row_22
    have equality_step_1 := row_53
    cases equality_step_1
    exact equality_source
  -- chapter_142_line_17: GL tag disintegration.
  have row_17 : (gl_fXY V1 V2 N) := by
    exact witness_row_18.1
  -- chapter_142_line_16: GL tag expansion.
  have row_16 : ((((gl_implication0 V1 V2) ∧ (gl_implication1 V1 N)) ∧ (gl_implication4 V2 N V1)) ∧ (gl_implication5 V2 V1)) := by
    simpa only [gl_fXY] using row_17
  -- chapter_142_line_100: GL tag disintegration.
  have row_100 : (gl_implication0 V1 V2) := by
    exact row_16.1.1.1
  -- chapter_142_line_99: GL tag expansion.
  have row_99 : (∀ (w1 : α) (w2 : α), ((V1 w1 w2) → (V2 w1))) := by
    simpa only [gl_implication0] using row_100
  -- chapter_142_line_78: GL tag disintegration.
  have row_78 : (gl_implication5 V2 V1) := by
    exact row_16.2
  -- chapter_142_line_77: GL tag expansion.
  have row_77 : (∀ (w1 : α), ((V2 w1) → (∀ (w2 : α), ((V1 w1 w2) → (∀ (w3 : α), ((V1 w1 w3) → (w2 = w3))))))) := by
    simpa only [gl_implication5] using row_78
  -- chapter_142_line_15: GL tag disintegration.
  have row_15 : (gl_implication1 V1 N) := by
    exact row_16.1.1.2
  -- chapter_142_line_14: GL tag expansion.
  have row_14 : (∀ (w1 : α) (w2 : α), ((V1 w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using row_15
  -- chapter_142_line_13: GL tag implication.
  have row_13 : (N v4) := by
    apply row_14
    exact row_22
  -- chapter_142_line_12: GL tag task formulation.
  have row_12 : (gl_AnchorGauss N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_142_line_11: GL tag expansion.
  have row_11 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorGauss] using row_12
  -- chapter_142_line_89: GL tag disintegration.
  have row_89 : (gl_identity N identity) := by
    exact row_11.2
  -- chapter_142_line_88: GL tag expansion.
  have row_88 : (((gl_implication0 identity N) ∧ (gl_implication22 identity)) ∧ (gl_implication23 N identity)) := by
    simpa only [gl_identity] using row_89
  -- chapter_142_line_87: GL tag disintegration.
  have row_87 : (gl_implication23 N identity) := by
    exact row_88.2
  -- chapter_142_line_86: GL tag expansion.
  have row_86 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((w1 = w2) → (identity w1 w2))))) := by
    simpa only [gl_implication23] using row_87
  -- chapter_142_line_58: GL tag disintegration.
  have row_58 : (succ zero one) := by
    exact row_11.1.1.2
  -- chapter_142_line_111: GL tag equality1.
  have row_111 : (succ v2 one) := by
    have equality_source := row_58
    have equality_step_1 := row_52
    cases equality_step_1
    exact equality_source
  -- chapter_142_line_30: GL tag disintegration.
  have row_30 : (succ one two) := by
    exact row_11.1.2
  -- chapter_142_line_10: GL tag disintegration.
  have row_10 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_11.1.1.1
  -- chapter_142_line_55: GL tag implication.
  have row_55 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_56
    exact row_10
    exact row_58
  have rule_row_106 := peano_source_022 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_142_line_106: GL tag implication.
  have row_106 : (mul v2 v1 v3) := by
    apply rule_row_106
    exact row_107
  have rule_row_95 := peano_source_063 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_142_line_95: GL tag implication.
  have row_95 : (gl_existence3 N one succ) := by
    apply rule_row_95
  -- chapter_142_line_94: GL tag expansion.
  have row_94 : (¬ (∀ (v13 : α), ((N v13) → (¬ (succ v13 one))))) := by
    simpa only [gl_existence3] using row_95
  have exists_row_94 : ∃ (v13 : α), ((N v13) ∧ (succ v13 one)) := existsAndOfNotForallImpNot row_94
  obtain ⟨v13, witness_row_94⟩ := exists_row_94
  -- chapter_142_line_93: GL tag disintegration.
  have row_93 : (succ v13 one) := by
    exact witness_row_94.2
  -- chapter_142_line_9: GL tag expansion.
  have row_9 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_10
  -- chapter_142_line_92: GL tag disintegration.
  have row_92 : (gl_implication7 N succ) := by
    exact row_9.1.1.1.1.1.1.1.1.1.2
  -- chapter_142_line_91: GL tag expansion.
  have row_91 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w2 w1) → (∀ (w3 : α), ((succ w3 w1) → (w2 = w3))))))) := by
    simpa only [gl_implication7] using row_92
  -- chapter_142_line_65: GL tag disintegration.
  have row_65 : (gl_implication21 N succ mul add) := by
    exact row_9.2
  -- chapter_142_line_64: GL tag expansion.
  have row_64 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((mul w3 w1 w4) → (∀ (w5 : α), ((mul w3 w2 w5) → (add w4 w3 w5))))))))) := by
    simpa only [gl_implication21] using row_65
  -- chapter_142_line_51: GL tag disintegration.
  have row_51 : (gl_implication15 N zero add) := by
    exact row_9.1.1.1.1.1.1.1.2
  -- chapter_142_line_50: GL tag expansion.
  have row_50 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((add w1 zero w2) → (w1 = w2))))) := by
    simpa only [gl_implication15] using row_51
  -- chapter_142_line_47: GL tag disintegration.
  have row_47 : (N zero) := by
    exact row_9.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_142_line_35: GL tag disintegration.
  have row_35 : (gl_fXYZ add N N N) := by
    exact row_9.1.1.1.1.1.1.1.1.2
  -- chapter_142_line_34: GL tag expansion.
  have row_34 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_35
  -- chapter_142_line_43: GL tag disintegration.
  have row_43 : (gl_implication13 N N N add) := by
    exact row_34.1.2
  -- chapter_142_line_42: GL tag expansion.
  have row_42 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication13] using row_43
  -- chapter_142_line_41: GL tag implication.
  have row_41 : (gl_existence1 N v2 zero add) := by
    apply row_42
    exact row_44
    exact row_47
  -- chapter_142_line_40: GL tag expansion.
  have row_40 : (¬ (∀ (v10 : α), ((N v10) → (¬ (add v2 zero v10))))) := by
    simpa only [gl_existence1] using row_41
  have exists_row_40 : ∃ (v10 : α), ((N v10) ∧ (add v2 zero v10)) := existsAndOfNotForallImpNot row_40
  obtain ⟨v10, witness_row_40⟩ := exists_row_40
  -- chapter_142_line_39: GL tag disintegration.
  have row_39 : (add v2 zero v10) := by
    exact witness_row_40.2
  -- chapter_142_line_49: GL tag implication.
  have row_49 : (v2 = v10) := by
    apply row_50
    exact row_44
    exact row_39
  -- chapter_142_line_48: GL tag symmetry of equality.
  have row_48 : (v10 = v2) := by
    exact Eq.symm row_49
  -- chapter_142_line_38: GL tag equality1.
  have row_38 : (add v2 zero v2) := by
    have equality_source := row_39
    have equality_step_1 := row_48
    cases equality_step_1
    exact equality_source
  -- chapter_142_line_37: GL tag equality1.
  have row_37 : (add v2 v2 v2) := by
    have equality_source := row_38
    have equality_step_1 := row_52
    cases equality_step_1
    exact equality_source
  -- chapter_142_line_33: GL tag disintegration.
  have row_33 : (gl_implication14 N N add) := by
    exact row_34.2
  -- chapter_142_line_32: GL tag expansion.
  have row_32 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (∀ (w3 : α), ((add w1 w2 w3) → (∀ (w4 : α), ((add w1 w2 w4) → (w3 = w4))))))))) := by
    simpa only [gl_implication14] using row_33
  -- chapter_142_line_29: GL tag disintegration.
  have row_29 : (gl_fXY succ N N) := by
    exact row_9.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_142_line_28: GL tag expansion.
  have row_28 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_29
  -- chapter_142_line_110: GL tag disintegration.
  have row_110 : (gl_implication5 N succ) := by
    exact row_28.2
  -- chapter_142_line_109: GL tag expansion.
  have row_109 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α), ((succ w1 w3) → (w2 = w3))))))) := by
    simpa only [gl_implication5] using row_110
  -- chapter_142_line_118: GL tag implication.
  have row_118 : (one = v1) := by
    apply row_109
    exact row_44
    exact row_111
    exact row_112
  -- chapter_142_line_108: GL tag implication.
  have row_108 : (v1 = one) := by
    apply row_109
    exact row_44
    exact row_112
    exact row_111
  -- chapter_142_line_105: GL tag equality1.
  have row_105 : (mul v2 one v3) := by
    have equality_source := row_106
    have equality_step_1 := row_108
    cases equality_step_1
    exact equality_source
  -- chapter_142_line_71: GL tag disintegration.
  have row_71 : (gl_implication0 succ N) := by
    exact row_28.1.1.1
  -- chapter_142_line_70: GL tag expansion.
  have row_70 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_71
  -- chapter_142_line_69: GL tag implication.
  have row_69 : (N one) := by
    apply row_70
    exact row_30
  -- chapter_142_line_96: GL tag implication.
  have row_96 : (v13 = zero) := by
    apply row_91
    exact row_69
    exact row_93
    exact row_58
  -- chapter_142_line_90: GL tag implication.
  have row_90 : (zero = v13) := by
    apply row_91
    exact row_69
    exact row_58
    exact row_93
  -- chapter_142_line_85: GL tag implication.
  have row_85 : (identity zero v13) := by
    apply row_86
    exact row_47
    exact row_90
  -- chapter_142_line_82: GL tag implication.
  have row_82 : (V1 zero v13) := by
    apply row_83
    exact row_85
  -- chapter_142_line_98: GL tag implication.
  have row_98 : (V2 zero) := by
    apply row_99
    exact row_82
  -- chapter_142_line_81: GL tag equality1.
  have row_81 : (V1 zero zero) := by
    have equality_source := row_82
    have equality_step_1 := row_96
    cases equality_step_1
    exact equality_source
  -- chapter_142_line_80: GL tag equality1.
  have row_80 : (V1 v2 v2) := by
    have equality_source := row_81
    have equality_step_1 := row_52
    cases equality_step_1
    exact equality_source
  -- chapter_142_line_79: GL tag equality1.
  have row_79 : (V1 zero v2) := by
    have equality_source := row_80
    have equality_step_1 := row_53
    cases equality_step_1
    exact equality_source
  -- chapter_142_line_122: GL tag implication.
  have row_122 : (v2 = v4) := by
    apply row_77
    exact row_98
    exact row_79
    exact row_97
  -- chapter_142_line_76: GL tag implication.
  have row_76 : (v4 = v2) := by
    apply row_77
    exact row_98
    exact row_97
    exact row_79
  -- chapter_142_line_27: GL tag disintegration.
  have row_27 : (gl_implication1 succ N) := by
    exact row_28.1.1.2
  -- chapter_142_line_26: GL tag expansion.
  have row_26 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using row_27
  -- chapter_142_line_25: GL tag implication.
  have row_25 : (N two) := by
    apply row_26
    exact row_30
  -- chapter_142_line_8: GL tag disintegration.
  have row_8 : (gl_fXYZ mul N N N) := by
    exact row_9.1.1.1.2
  -- chapter_142_line_7: GL tag expansion.
  have row_7 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_8
  -- chapter_142_line_121: GL tag disintegration.
  have row_121 : (gl_implication8 mul N) := by
    exact row_7.1.1.1.1
  -- chapter_142_line_120: GL tag expansion.
  have row_120 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication8] using row_121
  -- chapter_142_line_119: GL tag implication.
  have row_119 : (N v1) := by
    apply row_120
    exact row_107
  -- chapter_142_line_104: GL tag disintegration.
  have row_104 : (gl_implication14 N N mul) := by
    exact row_7.2
  -- chapter_142_line_103: GL tag expansion.
  have row_103 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (∀ (w3 : α), ((mul w1 w2 w3) → (∀ (w4 : α), ((mul w1 w2 w4) → (w3 = w4))))))))) := by
    simpa only [gl_implication14] using row_104
  -- chapter_142_line_6: GL tag disintegration.
  have row_6 : (gl_implication13 N N N mul) := by
    exact row_7.1.2
  -- chapter_142_line_5: GL tag expansion.
  have row_5 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication13] using row_6
  -- chapter_142_line_68: GL tag implication.
  have row_68 : (gl_existence1 N v4 one mul) := by
    apply row_5
    exact row_13
    exact row_69
  -- chapter_142_line_67: GL tag expansion.
  have row_67 : (¬ (∀ (v11 : α), ((N v11) → (¬ (mul v4 one v11))))) := by
    simpa only [gl_existence1] using row_68
  have exists_row_67 : ∃ (v11 : α), ((N v11) ∧ (mul v4 one v11)) := existsAndOfNotForallImpNot row_67
  obtain ⟨v11, witness_row_67⟩ := exists_row_67
  -- chapter_142_line_75: GL tag disintegration.
  have row_75 : (N v11) := by
    exact witness_row_67.1
  -- chapter_142_line_66: GL tag disintegration.
  have row_66 : (mul v4 one v11) := by
    exact witness_row_67.2
  have rule_row_74 := peano_source_044 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_142_line_74: GL tag implication.
  have row_74 : (mul one v4 v11) := by
    apply rule_row_74
    exact row_66
  have rule_row_73 := peano_source_058 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_142_line_73: GL tag implication.
  have row_73 : (v11 = v4) := by
    apply rule_row_73
    exact row_75
    exact row_74
  -- chapter_142_line_117: GL tag equality1.
  have row_117 : (mul one v4 v4) := by
    have equality_source := row_74
    have equality_step_1 := row_73
    cases equality_step_1
    exact equality_source
  -- chapter_142_line_116: GL tag equality1.
  have row_116 : (mul v1 v4 v4) := by
    have equality_source := row_117
    have equality_step_1 := row_118
    cases equality_step_1
    exact equality_source
  -- chapter_142_line_115: GL tag equality1.
  have row_115 : (mul v1 v2 v4) := by
    have equality_source := row_116
    have equality_step_1 := row_76
    cases equality_step_1
    exact equality_source
  -- chapter_142_line_114: GL tag implication.
  have row_114 : (v3 = v4) := by
    apply row_103
    exact row_119
    exact row_44
    exact row_107
    exact row_115
  -- chapter_142_line_113: GL tag equality1.
  have row_113 : (mul v2 one v4) := by
    have equality_source := row_105
    have equality_step_1 := row_114
    cases equality_step_1
    exact equality_source
  -- chapter_142_line_102: GL tag implication.
  have row_102 : (v4 = v3) := by
    apply row_103
    exact row_44
    exact row_69
    exact row_113
    exact row_105
  -- chapter_142_line_4: GL tag implication.
  have row_4 : (gl_existence1 N two v4 mul) := by
    apply row_5
    exact row_25
    exact row_13
  -- chapter_142_line_3: GL tag expansion.
  have row_3 : (¬ (∀ (v5 : α), ((N v5) → (¬ (mul two v4 v5))))) := by
    simpa only [gl_existence1] using row_4
  have exists_row_3 : ∃ (v5 : α), ((N v5) ∧ (mul two v4 v5)) := existsAndOfNotForallImpNot row_3
  obtain ⟨v5, witness_row_3⟩ := exists_row_3
  -- chapter_142_line_2: GL tag disintegration.
  have row_2 : (mul two v4 v5) := by
    exact witness_row_3.2
  have rule_row_72 := peano_source_022 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_142_line_72: GL tag implication.
  have row_72 : (mul v4 two v5) := by
    apply rule_row_72
    exact row_2
  -- chapter_142_line_63: GL tag implication.
  have row_63 : (add v11 v4 v5) := by
    apply row_64
    exact row_69
    exact row_30
    exact row_66
    exact row_72
  -- chapter_142_line_61: GL tag implication.
  have row_61 : (add v4 v11 v5) := by
    apply row_62
    exact row_63
  -- chapter_142_line_60: GL tag equality1.
  have row_60 : (add v4 v4 v5) := by
    have equality_source := row_61
    have equality_step_1 := row_73
    cases equality_step_1
    exact equality_source
  -- chapter_142_line_123: GL tag equality1.
  have row_123 : (add v2 v4 v5) := by
    have equality_source := row_60
    have equality_step_1 := row_76
    cases equality_step_1
    exact equality_source
  -- chapter_142_line_101: GL tag equality1.
  have row_101 : (add v4 v3 v5) := by
    have equality_source := row_60
    have equality_step_1 := row_102
    cases equality_step_1
    exact equality_source
  -- chapter_142_line_59: GL tag equality1.
  have row_59 : (add v4 v2 v5) := by
    have equality_source := row_60
    have equality_step_1 := row_76
    cases equality_step_1
    exact equality_source
  have rule_row_54 := peano_source_013 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_142_line_54: GL tag implication.
  have row_54 : (v2 = v3) := by
    apply rule_row_54
    exact row_59
    exact row_101
  -- chapter_142_line_36: GL tag equality1.
  have row_36 : (add v2 v4 v3) := by
    have equality_source := row_37
    have equality_step_1 := row_54
    cases equality_step_1
    have equality_step_2 := row_122
    cases equality_step_2
    exact equality_source
  -- chapter_142_line_31: GL tag implication.
  have row_31 : (v5 = v3) := by
    apply row_32
    exact row_44
    exact row_13
    exact row_123
    exact row_36
  -- chapter_142_line_1: GL tag equality1.
  have row_1 : (mul two v4 v3) := by
    have equality_source := row_2
    have equality_step_1 := row_31
    cases equality_step_1
    exact equality_source
  exact row_1

private theorem gauss_source_065_check_induction_condition
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorGauss N zero succ add mul one two identity)
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
    (assumption_218 : (succ v2 v1))
    (assumption_212 : (mul v1 v2 v3))
    (assumption_38 : (gl_fold N succ add identity zero v2 v4))
    (assumption_31 : (succ previous v2))
    (assumption_10 : (∀ (w1 : α) (w2 : α), ((mul w1 previous w2) → (∀ (w3 : α), ((gl_fold N succ add identity zero previous w3) → ((succ previous w1) → (mul two w3 w2)))))))
    : (mul two v4 v3) := by
  -- chapter_143_line_218: GL tag task formulation.
  have row_218 : (succ v2 v1) := by
    exact assumption_218
  -- chapter_143_line_212: GL tag task formulation.
  have row_212 : (mul v1 v2 v3) := by
    exact assumption_212
  -- chapter_143_line_196: GL tag theorem.
  have row_196 := gauss_source_064 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_143_line_111: GL tag theorem.
  have row_111 := gauss_source_088 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_143_line_86: GL tag expansion for integration.
  have row_86 : (∀ (V1 : GLSet α) (V2 : GLBinaryRelation α), ((gl_implication33 V1 succ V2 identity add) ↔ (∀ (v21 : α), ((V1 v21) → (∀ (v17 : α), ((succ v21 v17) → ((V1 v17) → (∀ (v22 : α), ((V2 v21 v22) → (∀ (v23 : α), ((identity v17 v23) → (∀ (v18 : α), ((add v22 v23 v18) → (V2 v17 v18)))))))))))))) := by
    intro V1 V2
    exact Iff.rfl
  -- chapter_143_line_100: GL tag premise element.
  have row_100 : (∀ (V1 : GLSet α) (V2 : GLBinaryRelation α) (v21 : α), ((V1 v21) → (∀ (v17 : α), ((succ v21 v17) → ((V1 v17) → (∀ (v22 : α), ((V2 v21 v22) → (∀ (v23 : α), ((identity v17 v23) → (∀ (v18 : α), ((add v22 v23 v18) → (V1 v17)))))))))))) := by
    intro V1
    intro V2
    intro v21
    intro scope_premise_1
    intro v17
    intro scope_premise_2
    intro scope_premise_3
    intro v22
    intro scope_premise_4
    intro v23
    intro scope_premise_5
    intro v18
    intro scope_premise_6
    exact scope_premise_3
  -- chapter_143_line_98: GL tag premise element.
  have row_98 : (∀ (V1 : GLSet α) (V2 : GLBinaryRelation α) (v21 : α), ((V1 v21) → (∀ (v17 : α), ((succ v21 v17) → ((V1 v17) → (∀ (v22 : α), ((V2 v21 v22) → (∀ (v23 : α), ((identity v17 v23) → (∀ (v18 : α), ((add v22 v23 v18) → (V1 v21)))))))))))) := by
    intro V1
    intro V2
    intro v21
    intro scope_premise_1
    intro v17
    intro scope_premise_2
    intro scope_premise_3
    intro v22
    intro scope_premise_4
    intro v23
    intro scope_premise_5
    intro v18
    intro scope_premise_6
    exact scope_premise_1
  -- chapter_143_line_92: GL tag premise element.
  have row_92 : (∀ (V1 : GLSet α) (V2 : GLBinaryRelation α) (v21 : α), ((V1 v21) → (∀ (v17 : α), ((succ v21 v17) → ((V1 v17) → (∀ (v22 : α), ((V2 v21 v22) → (∀ (v23 : α), ((identity v17 v23) → (∀ (v18 : α), ((add v22 v23 v18) → (add v22 v23 v18)))))))))))) := by
    intro V1
    intro V2
    intro v21
    intro scope_premise_1
    intro v17
    intro scope_premise_2
    intro scope_premise_3
    intro v22
    intro scope_premise_4
    intro v23
    intro scope_premise_5
    intro v18
    intro scope_premise_6
    exact scope_premise_6
  -- chapter_143_line_91: GL tag premise element.
  have row_91 : (∀ (V1 : GLSet α) (V2 : GLBinaryRelation α) (v21 : α), ((V1 v21) → (∀ (v17 : α), ((succ v21 v17) → ((V1 v17) → (∀ (v22 : α), ((V2 v21 v22) → (∀ (v23 : α), ((identity v17 v23) → (∀ (v18 : α), ((add v22 v23 v18) → (identity v17 v23)))))))))))) := by
    intro V1
    intro V2
    intro v21
    intro scope_premise_1
    intro v17
    intro scope_premise_2
    intro scope_premise_3
    intro v22
    intro scope_premise_4
    intro v23
    intro scope_premise_5
    intro v18
    intro scope_premise_6
    exact scope_premise_5
  -- chapter_143_line_90: GL tag premise element.
  have row_90 : (∀ (V1 : GLSet α) (V2 : GLBinaryRelation α) (v21 : α), ((V1 v21) → (∀ (v17 : α), ((succ v21 v17) → ((V1 v17) → (∀ (v22 : α), ((V2 v21 v22) → (∀ (v23 : α), ((identity v17 v23) → (∀ (v18 : α), ((add v22 v23 v18) → (V2 v21 v22)))))))))))) := by
    intro V1
    intro V2
    intro v21
    intro scope_premise_1
    intro v17
    intro scope_premise_2
    intro scope_premise_3
    intro v22
    intro scope_premise_4
    intro v23
    intro scope_premise_5
    intro v18
    intro scope_premise_6
    exact scope_premise_4
  -- chapter_143_line_85: GL tag premise element.
  have row_85 : (∀ (V1 : GLSet α) (V2 : GLBinaryRelation α) (v21 : α), ((V1 v21) → (∀ (v17 : α), ((succ v21 v17) → ((V1 v17) → (∀ (v22 : α), ((V2 v21 v22) → (∀ (v23 : α), ((identity v17 v23) → (∀ (v18 : α), ((add v22 v23 v18) → (succ v21 v17)))))))))))) := by
    intro V1
    intro V2
    intro v21
    intro scope_premise_1
    intro v17
    intro scope_premise_2
    intro scope_premise_3
    intro v22
    intro scope_premise_4
    intro v23
    intro scope_premise_5
    intro v18
    intro scope_premise_6
    exact scope_premise_2
  -- chapter_143_line_79: GL tag expansion for integration.
  have row_79 : (∀ (V2 : GLBinaryRelation α), ((gl_implication32 zero identity V2) ↔ (∀ (v12 : α), ((identity zero v12) → (V2 zero v12))))) := by
    intro V2
    exact Iff.rfl
  -- chapter_143_line_78: GL tag premise element.
  have row_78 : (∀ (V2 : GLBinaryRelation α) (v12 : α), ((identity zero v12) → (identity zero v12))) := by
    intro V2
    intro v12
    intro scope_premise_1
    exact scope_premise_1
  -- chapter_143_line_73: GL tag theorem.
  have row_73 := gauss_source_090 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_143_line_71: GL tag theorem.
  have row_71 := gauss_source_077 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_143_line_38: GL tag task formulation.
  have row_38 : (gl_fold N succ add identity zero v2 v4) := by
    exact assumption_38
  -- chapter_143_line_37: GL tag expansion.
  have row_37 : (¬ (∀ (V4 : GLSet α), ((gl_interval N add zero v2 V4) → (¬ (gl_existence2 V4 N zero identity v2 v4 succ add))))) := by
    simpa only [gl_fold] using row_38
  have exists_row_37 : ∃ (V4 : GLSet α), ((gl_interval N add zero v2 V4) ∧ (gl_existence2 V4 N zero identity v2 v4 succ add)) := existsAndOfNotForallImpNot row_37
  obtain ⟨V4, witness_row_37⟩ := exists_row_37
  -- chapter_143_line_39: GL tag disintegration.
  have row_39 : (gl_interval N add zero v2 V4) := by
    exact witness_row_37.1
  -- chapter_143_line_132: GL tag expansion.
  have row_132 : (((((gl_implication26 V4 N add zero) ∧ (gl_implication27 V4 N add v2)) ∧ (gl_implication28 N add zero v2 V4)) ∧ (N zero)) ∧ (N v2)) := by
    simpa only [gl_interval] using row_39
  -- chapter_143_line_131: GL tag disintegration.
  have row_131 : (N v2) := by
    exact row_132.2
  -- chapter_143_line_36: GL tag disintegration.
  have row_36 : (gl_existence2 V4 N zero identity v2 v4 succ add) := by
    exact witness_row_37.2
  -- chapter_143_line_35: GL tag expansion.
  have row_35 : (¬ (∀ (V3 : GLBinaryRelation α), ((gl_fXY V3 V4 N) → (¬ (gl_and0 zero identity V3 v2 v4 V4 succ add))))) := by
    simpa only [gl_existence2] using row_36
  have exists_row_35 : ∃ (V3 : GLBinaryRelation α), ((gl_fXY V3 V4 N) ∧ (gl_and0 zero identity V3 v2 v4 V4 succ add)) := existsAndOfNotForallImpNot row_35
  obtain ⟨V3, witness_row_35⟩ := exists_row_35
  -- chapter_143_line_44: GL tag disintegration.
  have row_44 : (gl_and0 zero identity V3 v2 v4 V4 succ add) := by
    exact witness_row_35.2
  -- chapter_143_line_43: GL tag expansion.
  have row_43 : (((gl_implication32 zero identity V3) ∧ (V3 v2 v4)) ∧ (gl_implication33 V4 succ V3 identity add)) := by
    simpa only [gl_and0] using row_44
  -- chapter_143_line_179: GL tag disintegration.
  have row_179 : (V3 v2 v4) := by
    exact row_43.1.2
  -- chapter_143_line_84: GL tag disintegration.
  have row_84 : (gl_implication33 V4 succ V3 identity add) := by
    exact row_43.2
  -- chapter_143_line_83: GL tag expansion.
  have row_83 : (∀ (w1 : α), ((V4 w1) → (∀ (w2 : α), ((succ w1 w2) → ((V4 w2) → (∀ (w3 : α), ((V3 w1 w3) → (∀ (w4 : α), ((identity w2 w4) → (∀ (w5 : α), ((add w3 w4 w5) → (V3 w2 w5)))))))))))) := by
    simpa only [gl_implication33] using row_84
  -- chapter_143_line_42: GL tag disintegration.
  have row_42 : (gl_implication32 zero identity V3) := by
    exact row_43.1.1
  -- chapter_143_line_41: GL tag expansion.
  have row_41 : (∀ (w1 : α), ((identity zero w1) → (V3 zero w1))) := by
    simpa only [gl_implication32] using row_42
  -- chapter_143_line_34: GL tag disintegration.
  have row_34 : (gl_fXY V3 V4 N) := by
    exact witness_row_35.1
  -- chapter_143_line_178: GL tag expansion.
  have row_178 : ((((gl_implication0 V3 V4) ∧ (gl_implication1 V3 N)) ∧ (gl_implication4 V4 N V3)) ∧ (gl_implication5 V4 V3)) := by
    simpa only [gl_fXY] using row_34
  -- chapter_143_line_207: GL tag disintegration.
  have row_207 : (gl_implication0 V3 V4) := by
    exact row_178.1.1.1
  -- chapter_143_line_206: GL tag expansion.
  have row_206 : (∀ (w1 : α) (w2 : α), ((V3 w1 w2) → (V4 w1))) := by
    simpa only [gl_implication0] using row_207
  -- chapter_143_line_205: GL tag implication.
  have row_205 : (V4 v2) := by
    apply row_206
    exact row_179
  -- chapter_143_line_177: GL tag disintegration.
  have row_177 : (gl_implication5 V4 V3) := by
    exact row_178.2
  -- chapter_143_line_176: GL tag expansion.
  have row_176 : (∀ (w1 : α), ((V4 w1) → (∀ (w2 : α), ((V3 w1 w2) → (∀ (w3 : α), ((V3 w1 w3) → (w2 = w3))))))) := by
    simpa only [gl_implication5] using row_177
  -- chapter_143_line_33: GL tag theorem.
  have row_33 := gauss_source_067 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_143_line_31: GL tag recursion.
  have row_31 : (succ previous v2) := by
    exact assumption_31
  -- chapter_143_line_30: GL tag theorem.
  have row_30 := gauss_source_086 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_143_line_19: GL tag expansion for integration.
  have row_19 : (∀ (V1 : GLSet α) (V2 : GLBinaryRelation α) (v5 : α), ((gl_and0 zero identity V2 previous v5 V1 succ add) ↔ (((gl_implication32 zero identity V2) ∧ (V2 previous v5)) ∧ (gl_implication33 V1 succ V2 identity add)))) := by
    intro V1 V2 v5
    exact Iff.rfl
  -- chapter_143_line_18: GL tag reformulation for integration and.
  have row_18 : (∀ (V1 : GLSet α) (V2 : GLBinaryRelation α) (v5 : α), ((gl_implication32 zero identity V2) → ((V2 previous v5) → ((gl_implication33 V1 succ V2 identity add) → (gl_and0 zero identity V2 previous v5 V1 succ add))))) := by
    intro V1
    intro V2
    intro v5
    intro integration_premise_1
    intro integration_premise_2
    intro integration_premise_3
    simp only [gl_and0]
    exact ⟨⟨integration_premise_1, integration_premise_2⟩, integration_premise_3⟩
  -- chapter_143_line_16: GL tag expansion for integration.
  have row_16 : (∀ (V1 : GLSet α) (v5 : α), ((gl_existence2 V1 N zero identity previous v5 succ add) ↔ (¬ (∀ (V2 : GLBinaryRelation α), ((gl_fXY V2 V1 N) → (¬ (gl_and0 zero identity V2 previous v5 V1 succ add))))))) := by
    intro V1 v5
    exact Iff.rfl
  -- chapter_143_line_15: GL tag reformulation for integration >[].
  have row_15 : (∀ (V1 : GLSet α) (V2 : GLBinaryRelation α) (v5 : α), ((gl_fXY V2 V1 N) → ((gl_and0 zero identity V2 previous v5 V1 succ add) → (gl_existence2 V1 N zero identity previous v5 succ add)))) := by
    intro V1
    intro V2
    intro v5
    intro integration_premise_1
    intro integration_premise_2
    apply (row_16 V1 v5).2
    intro universal_counterexample
    exact universal_counterexample V2 integration_premise_1 integration_premise_2
  -- chapter_143_line_13: GL tag expansion for integration.
  have row_13 : (∀ (v5 : α), ((gl_fold N succ add identity zero previous v5) ↔ (¬ (∀ (V1 : GLSet α), ((gl_interval N add zero previous V1) → (¬ (gl_existence2 V1 N zero identity previous v5 succ add))))))) := by
    intro v5
    exact Iff.rfl
  -- chapter_143_line_12: GL tag reformulation for integration >[].
  have row_12 : (∀ (V1 : GLSet α) (v5 : α), ((gl_interval N add zero previous V1) → ((gl_existence2 V1 N zero identity previous v5 succ add) → (gl_fold N succ add identity zero previous v5)))) := by
    intro V1
    intro v5
    intro integration_premise_1
    intro integration_premise_2
    apply (row_13 v5).2
    intro universal_counterexample
    exact universal_counterexample V1 integration_premise_1 integration_premise_2
  -- chapter_143_line_10: GL tag recursion.
  have row_10 : (∀ (w1 : α) (w2 : α), ((mul w1 previous w2) → (∀ (w3 : α), ((gl_fold N succ add identity zero previous w3) → ((succ previous w1) → (mul two w3 w2)))))) := by
    exact assumption_10
  -- chapter_143_line_7: GL tag task formulation.
  have row_7 : (gl_AnchorGauss N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_143_line_197: GL tag anchor handling.
  have row_197 : (gl_AnchorGauss N zero succ add mul one two identity) := by
    exact row_7
  -- chapter_143_line_70: GL tag implication.
  have row_70 : (gl_existence9 N add V4 previous zero) := by
    apply row_71
    exact row_31
    exact row_39
  -- chapter_143_line_69: GL tag expansion.
  have row_69 : (¬ (∀ (V1 : GLSet α), ((gl_limitSet N add V4 previous V1) → (¬ (gl_interval N add zero previous V1))))) := by
    simpa only [gl_existence9] using row_70
  have exists_row_69 : ∃ (V1 : GLSet α), ((gl_limitSet N add V4 previous V1) ∧ (gl_interval N add zero previous V1)) := existsAndOfNotForallImpNot row_69
  obtain ⟨V1, witness_row_69⟩ := exists_row_69
  -- chapter_143_line_97: GL tag disintegration.
  have row_97 : (gl_limitSet N add V4 previous V1) := by
    exact witness_row_69.1
  -- chapter_143_line_96: GL tag expansion.
  have row_96 : (((gl_implication41 V1 V4) ∧ (gl_implication27 V1 N add previous)) ∧ (gl_implication42 V4 N add previous V1)) := by
    simpa only [gl_limitSet] using row_97
  -- chapter_143_line_103: GL tag disintegration.
  have row_103 : (gl_implication27 V1 N add previous) := by
    exact row_96.1.2
  -- chapter_143_line_102: GL tag expansion.
  have row_102 : (∀ (w1 : α), ((V1 w1) → (gl_preorder N add w1 previous))) := by
    simpa only [gl_implication27] using row_103
  -- chapter_143_line_101: GL tag implication.
  have row_101 : (∀ (V2 : GLBinaryRelation α) (v21 : α), ((V1 v21) → (∀ (v17 : α), ((succ v21 v17) → ((V1 v17) → (∀ (v22 : α), ((V2 v21 v22) → (∀ (v23 : α), ((identity v17 v23) → (∀ (v18 : α), ((add v22 v23 v18) → (gl_preorder N add v17 previous)))))))))))) := by
    intro V2
    intro v21
    intro scope_premise_1
    intro v17
    intro scope_premise_2
    intro scope_premise_3
    intro v22
    intro scope_premise_4
    intro v23
    intro scope_premise_5
    intro v18
    intro scope_premise_6
    have scoped_fact_2 := row_100 V1 V2 v21 scope_premise_1 v17 scope_premise_2 scope_premise_3 v22 scope_premise_4 v23 scope_premise_5 v18 scope_premise_6
    apply row_102
    exact scoped_fact_2
  -- chapter_143_line_95: GL tag disintegration.
  have row_95 : (gl_implication41 V1 V4) := by
    exact row_96.1.1
  -- chapter_143_line_94: GL tag expansion.
  have row_94 : (∀ (w1 : α), ((V1 w1) → (V4 w1))) := by
    simpa only [gl_implication41] using row_95
  -- chapter_143_line_99: GL tag implication.
  have row_99 : (∀ (V2 : GLBinaryRelation α) (v21 : α), ((V1 v21) → (∀ (v17 : α), ((succ v21 v17) → ((V1 v17) → (∀ (v22 : α), ((V2 v21 v22) → (∀ (v23 : α), ((identity v17 v23) → (∀ (v18 : α), ((add v22 v23 v18) → (V4 v17)))))))))))) := by
    intro V2
    intro v21
    intro scope_premise_1
    intro v17
    intro scope_premise_2
    intro scope_premise_3
    intro v22
    intro scope_premise_4
    intro v23
    intro scope_premise_5
    intro v18
    intro scope_premise_6
    have scoped_fact_2 := row_100 V1 V2 v21 scope_premise_1 v17 scope_premise_2 scope_premise_3 v22 scope_premise_4 v23 scope_premise_5 v18 scope_premise_6
    apply row_94
    exact scoped_fact_2
  -- chapter_143_line_93: GL tag implication.
  have row_93 : (∀ (V2 : GLBinaryRelation α) (v21 : α), ((V1 v21) → (∀ (v17 : α), ((succ v21 v17) → ((V1 v17) → (∀ (v22 : α), ((V2 v21 v22) → (∀ (v23 : α), ((identity v17 v23) → (∀ (v18 : α), ((add v22 v23 v18) → (V4 v21)))))))))))) := by
    intro V2
    intro v21
    intro scope_premise_1
    intro v17
    intro scope_premise_2
    intro scope_premise_3
    intro v22
    intro scope_premise_4
    intro v23
    intro scope_premise_5
    intro v18
    intro scope_premise_6
    have scoped_fact_2 := row_98 V1 V2 v21 scope_premise_1 v17 scope_premise_2 scope_premise_3 v22 scope_premise_4 v23 scope_premise_5 v18 scope_premise_6
    apply row_94
    exact scoped_fact_2
  -- chapter_143_line_68: GL tag disintegration.
  have row_68 : (gl_interval N add zero previous V1) := by
    exact witness_row_69.2
  -- chapter_143_line_72: GL tag implication.
  have row_72 : (V1 previous) := by
    apply row_73
    exact row_68
  -- chapter_143_line_67: GL tag expansion.
  have row_67 : (((((gl_implication26 V1 N add zero) ∧ (gl_implication27 V1 N add previous)) ∧ (gl_implication28 N add zero previous V1)) ∧ (N zero)) ∧ (N previous)) := by
    simpa only [gl_interval] using row_68
  -- chapter_143_line_66: GL tag disintegration.
  have row_66 : (gl_implication26 V1 N add zero) := by
    exact row_67.1.1.1.1
  -- chapter_143_line_65: GL tag expansion.
  have row_65 : (∀ (w1 : α), ((V1 w1) → (gl_preorder N add zero w1))) := by
    simpa only [gl_implication26] using row_66
  -- chapter_143_line_64: GL tag implication.
  have row_64 : (gl_preorder N add zero previous) := by
    apply row_65
    exact row_72
  -- chapter_143_line_32: GL tag implication.
  have row_32 : (gl_sequence N add zero v2 V3) := by
    apply row_33
    exact row_34
    exact row_39
  -- chapter_143_line_29: GL tag implication.
  have row_29 : (gl_existence15 N add previous V3 zero) := by
    apply row_30
    exact row_31
    exact row_32
  -- chapter_143_line_28: GL tag expansion.
  have row_28 : (¬ (∀ (V2 : GLBinaryRelation α), ((gl_limitSequence N add previous V3 V2) → (¬ (gl_sequence N add zero previous V2))))) := by
    simpa only [gl_existence15] using row_29
  have exists_row_28 : ∃ (V2 : GLBinaryRelation α), ((gl_limitSequence N add previous V3 V2) ∧ (gl_sequence N add zero previous V2)) := existsAndOfNotForallImpNot row_28
  obtain ⟨V2, witness_row_28⟩ := exists_row_28
  -- chapter_143_line_123: GL tag disintegration.
  have row_123 : (gl_sequence N add zero previous V2) := by
    exact witness_row_28.2
  -- chapter_143_line_27: GL tag disintegration.
  have row_27 : (gl_limitSequence N add previous V3 V2) := by
    exact witness_row_28.1
  -- chapter_143_line_26: GL tag expansion.
  have row_26 : (((gl_implication38 V2 N add previous) ∧ (gl_implication39 V2 V3)) ∧ (gl_implication40 N add previous V3 V2)) := by
    simpa only [gl_limitSequence] using row_27
  -- chapter_143_line_89: GL tag disintegration.
  have row_89 : (gl_implication39 V2 V3) := by
    exact row_26.1.2
  -- chapter_143_line_88: GL tag expansion.
  have row_88 : (∀ (w1 : α) (w2 : α), ((V2 w1 w2) → (V3 w1 w2))) := by
    simpa only [gl_implication39] using row_89
  -- chapter_143_line_87: GL tag implication.
  have row_87 : (∀ (v21 : α), ((V1 v21) → (∀ (v17 : α), ((succ v21 v17) → ((V1 v17) → (∀ (v22 : α), ((V2 v21 v22) → (∀ (v23 : α), ((identity v17 v23) → (∀ (v18 : α), ((add v22 v23 v18) → (V3 v21 v22)))))))))))) := by
    intro v21
    intro scope_premise_1
    intro v17
    intro scope_premise_2
    intro scope_premise_3
    intro v22
    intro scope_premise_4
    intro v23
    intro scope_premise_5
    intro v18
    intro scope_premise_6
    have scoped_fact_2 := row_90 V1 V2 v21 scope_premise_1 v17 scope_premise_2 scope_premise_3 v22 scope_premise_4 v23 scope_premise_5 v18 scope_premise_6
    apply row_88
    exact scoped_fact_2
  -- chapter_143_line_82: GL tag implication.
  have row_82 : (∀ (v21 : α), ((V1 v21) → (∀ (v17 : α), ((succ v21 v17) → ((V1 v17) → (∀ (v22 : α), ((V2 v21 v22) → (∀ (v23 : α), ((identity v17 v23) → (∀ (v18 : α), ((add v22 v23 v18) → (V3 v17 v18)))))))))))) := by
    intro v21
    intro scope_premise_1
    intro v17
    intro scope_premise_2
    intro scope_premise_3
    intro v22
    intro scope_premise_4
    intro v23
    intro scope_premise_5
    intro v18
    intro scope_premise_6
    have scoped_fact_2 := row_93 V2 v21 scope_premise_1 v17 scope_premise_2 scope_premise_3 v22 scope_premise_4 v23 scope_premise_5 v18 scope_premise_6
    have scoped_fact_3 := row_85 V1 V2 v21 scope_premise_1 v17 scope_premise_2 scope_premise_3 v22 scope_premise_4 v23 scope_premise_5 v18 scope_premise_6
    have scoped_fact_4 := row_99 V2 v21 scope_premise_1 v17 scope_premise_2 scope_premise_3 v22 scope_premise_4 v23 scope_premise_5 v18 scope_premise_6
    have scoped_fact_5 := row_87 v21 scope_premise_1 v17 scope_premise_2 scope_premise_3 v22 scope_premise_4 v23 scope_premise_5 v18 scope_premise_6
    have scoped_fact_6 := row_91 V1 V2 v21 scope_premise_1 v17 scope_premise_2 scope_premise_3 v22 scope_premise_4 v23 scope_premise_5 v18 scope_premise_6
    have scoped_fact_7 := row_92 V1 V2 v21 scope_premise_1 v17 scope_premise_2 scope_premise_3 v22 scope_premise_4 v23 scope_premise_5 v18 scope_premise_6
    apply row_83
    exact scoped_fact_2
    exact scoped_fact_3
    exact scoped_fact_4
    exact scoped_fact_5
    exact scoped_fact_6
    exact scoped_fact_7
  -- chapter_143_line_25: GL tag disintegration.
  have row_25 : (gl_implication40 N add previous V3 V2) := by
    exact row_26.2
  -- chapter_143_line_24: GL tag expansion.
  have row_24 : (∀ (w1 : α), ((gl_preorder N add w1 previous) → (∀ (w2 : α), ((V3 w1 w2) → (V2 w1 w2))))) := by
    simpa only [gl_implication40] using row_25
  -- chapter_143_line_81: GL tag implication.
  have row_81 : (∀ (v21 : α), ((V1 v21) → (∀ (v17 : α), ((succ v21 v17) → ((V1 v17) → (∀ (v22 : α), ((V2 v21 v22) → (∀ (v23 : α), ((identity v17 v23) → (∀ (v18 : α), ((add v22 v23 v18) → (V2 v17 v18)))))))))))) := by
    intro v21
    intro scope_premise_1
    intro v17
    intro scope_premise_2
    intro scope_premise_3
    intro v22
    intro scope_premise_4
    intro v23
    intro scope_premise_5
    intro v18
    intro scope_premise_6
    have scoped_fact_2 := row_101 V2 v21 scope_premise_1 v17 scope_premise_2 scope_premise_3 v22 scope_premise_4 v23 scope_premise_5 v18 scope_premise_6
    have scoped_fact_3 := row_82 v21 scope_premise_1 v17 scope_premise_2 scope_premise_3 v22 scope_premise_4 v23 scope_premise_5 v18 scope_premise_6
    apply row_24
    exact scoped_fact_2
    exact scoped_fact_3
  -- chapter_143_line_80: GL tag validity name.
  have row_80 : (gl_implication33 V1 succ V2 identity add) := by
    simpa only [gl_implication33] using row_81
  -- chapter_143_line_6: GL tag expansion.
  have row_6 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorGauss] using row_7
  -- chapter_143_line_62: GL tag disintegration.
  have row_62 : (succ one two) := by
    exact row_6.1.2
  -- chapter_143_line_49: GL tag disintegration.
  have row_49 : (gl_identity N identity) := by
    exact row_6.2
  -- chapter_143_line_48: GL tag expansion.
  have row_48 : (((gl_implication0 identity N) ∧ (gl_implication22 identity)) ∧ (gl_implication23 N identity)) := by
    simpa only [gl_identity] using row_49
  -- chapter_143_line_77: GL tag disintegration.
  have row_77 : (gl_implication22 identity) := by
    exact row_48.1.2
  -- chapter_143_line_76: GL tag expansion.
  have row_76 : (∀ (w1 : α) (w2 : α), ((identity w1 w2) → (w1 = w2))) := by
    simpa only [gl_implication22] using row_77
  -- chapter_143_line_75: GL tag implication.
  have row_75 : (∀ (v12 : α), ((identity zero v12) → (zero = v12))) := by
    intro v12
    intro scope_premise_1
    have scoped_fact_2 := row_78 V2 v12 scope_premise_1
    apply row_76
    exact scoped_fact_2
  -- chapter_143_line_47: GL tag disintegration.
  have row_47 : (gl_implication23 N identity) := by
    exact row_48.2
  -- chapter_143_line_46: GL tag expansion.
  have row_46 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((w1 = w2) → (identity w1 w2))))) := by
    simpa only [gl_implication23] using row_47
  -- chapter_143_line_8: GL tag disintegration.
  have row_8 : (succ zero one) := by
    exact row_6.1.1.2
  -- chapter_143_line_5: GL tag disintegration.
  have row_5 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_6.1.1.1
  -- chapter_143_line_53: GL tag expansion.
  have row_53 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_5
  -- chapter_143_line_217: GL tag disintegration.
  have row_217 : (gl_implication18 N succ add) := by
    exact row_53.1.1.1.1.2
  -- chapter_143_line_216: GL tag expansion.
  have row_216 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((succ w4 w5) → (add w3 w2 w5))))))))) := by
    simpa only [gl_implication18] using row_217
  -- chapter_143_line_189: GL tag disintegration.
  have row_189 : (gl_implication15 N zero add) := by
    exact row_53.1.1.1.1.1.1.1.2
  -- chapter_143_line_188: GL tag expansion.
  have row_188 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((add w1 zero w2) → (w1 = w2))))) := by
    simpa only [gl_implication15] using row_189
  -- chapter_143_line_184: GL tag disintegration.
  have row_184 : (gl_implication17 N succ add) := by
    exact row_53.1.1.1.1.1.2
  -- chapter_143_line_183: GL tag expansion.
  have row_183 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication17] using row_184
  -- chapter_143_line_148: GL tag disintegration.
  have row_148 : (gl_implication16 N zero add) := by
    exact row_53.1.1.1.1.1.1.2
  -- chapter_143_line_147: GL tag expansion.
  have row_147 : (∀ (w1 : α) (w2 : α), ((w1 = w2) → ((N w1) → ((N w2) → (add w1 zero w2))))) := by
    simpa only [gl_implication16] using row_148
  -- chapter_143_line_130: GL tag disintegration.
  have row_130 : (gl_fXYZ mul N N N) := by
    exact row_53.1.1.1.2
  -- chapter_143_line_129: GL tag expansion.
  have row_129 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_130
  -- chapter_143_line_128: GL tag disintegration.
  have row_128 : (gl_implication13 N N N mul) := by
    exact row_129.1.2
  -- chapter_143_line_127: GL tag expansion.
  have row_127 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication13] using row_128
  -- chapter_143_line_120: GL tag disintegration.
  have row_120 : (gl_fXYZ add N N N) := by
    exact row_53.1.1.1.1.1.1.1.1.2
  -- chapter_143_line_119: GL tag expansion.
  have row_119 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_120
  -- chapter_143_line_143: GL tag disintegration.
  have row_143 : (gl_implication14 N N add) := by
    exact row_119.2
  -- chapter_143_line_142: GL tag expansion.
  have row_142 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (∀ (w3 : α), ((add w1 w2 w3) → (∀ (w4 : α), ((add w1 w2 w4) → (w3 = w4))))))))) := by
    simpa only [gl_implication14] using row_143
  -- chapter_143_line_118: GL tag disintegration.
  have row_118 : (gl_implication13 N N N add) := by
    exact row_119.1.2
  -- chapter_143_line_117: GL tag expansion.
  have row_117 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication13] using row_118
  -- chapter_143_line_63: GL tag disintegration.
  have row_63 : (N zero) := by
    exact row_53.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_143_line_167: GL tag implication.
  have row_167 : (gl_existence1 N v2 zero add) := by
    apply row_117
    exact row_131
    exact row_63
  -- chapter_143_line_166: GL tag expansion.
  have row_166 : (¬ (∀ (v7 : α), ((N v7) → (¬ (add v2 zero v7))))) := by
    simpa only [gl_existence1] using row_167
  have exists_row_166 : ∃ (v7 : α), ((N v7) ∧ (add v2 zero v7)) := existsAndOfNotForallImpNot row_166
  obtain ⟨v7, witness_row_166⟩ := exists_row_166
  -- chapter_143_line_165: GL tag disintegration.
  have row_165 : (add v2 zero v7) := by
    exact witness_row_166.2
  -- chapter_143_line_61: GL tag disintegration.
  have row_61 : (gl_fXY succ N N) := by
    exact row_53.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_143_line_60: GL tag expansion.
  have row_60 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_61
  -- chapter_143_line_156: GL tag disintegration.
  have row_156 : (gl_implication4 N N succ) := by
    exact row_60.1.2
  -- chapter_143_line_155: GL tag expansion.
  have row_155 : (∀ (w1 : α), ((N w1) → (gl_existence0 N w1 succ))) := by
    simpa only [gl_implication4] using row_156
  -- chapter_143_line_151: GL tag disintegration.
  have row_151 : (gl_implication5 N succ) := by
    exact row_60.2
  -- chapter_143_line_150: GL tag expansion.
  have row_150 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α), ((succ w1 w3) → (w2 = w3))))))) := by
    simpa only [gl_implication5] using row_151
  -- chapter_143_line_140: GL tag disintegration.
  have row_140 : (gl_implication1 succ N) := by
    exact row_60.1.1.2
  -- chapter_143_line_139: GL tag expansion.
  have row_139 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using row_140
  -- chapter_143_line_138: GL tag implication.
  have row_138 : (N two) := by
    apply row_139
    exact row_62
  -- chapter_143_line_137: GL tag implication.
  have row_137 : (gl_existence1 N v2 two mul) := by
    apply row_127
    exact row_131
    exact row_138
  -- chapter_143_line_136: GL tag expansion.
  have row_136 : (¬ (∀ (v8 : α), ((N v8) → (¬ (mul v2 two v8))))) := by
    simpa only [gl_existence1] using row_137
  have exists_row_136 : ∃ (v8 : α), ((N v8) ∧ (mul v2 two v8)) := existsAndOfNotForallImpNot row_136
  obtain ⟨v8, witness_row_136⟩ := exists_row_136
  -- chapter_143_line_135: GL tag disintegration.
  have row_135 : (mul v2 two v8) := by
    exact witness_row_136.2
  -- chapter_143_line_59: GL tag disintegration.
  have row_59 : (gl_implication0 succ N) := by
    exact row_60.1.1.1
  -- chapter_143_line_58: GL tag expansion.
  have row_58 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_59
  -- chapter_143_line_121: GL tag implication.
  have row_121 : (N previous) := by
    apply row_58
    exact row_31
  -- chapter_143_line_192: GL tag implication.
  have row_192 : (gl_existence1 N previous zero add) := by
    apply row_117
    exact row_121
    exact row_63
  -- chapter_143_line_191: GL tag expansion.
  have row_191 : (¬ (∀ (v28 : α), ((N v28) → (¬ (add previous zero v28))))) := by
    simpa only [gl_existence1] using row_192
  have exists_row_191 : ∃ (v28 : α), ((N v28) ∧ (add previous zero v28)) := existsAndOfNotForallImpNot row_191
  obtain ⟨v28, witness_row_191⟩ := exists_row_191
  -- chapter_143_line_201: GL tag disintegration.
  have row_201 : (N v28) := by
    exact witness_row_191.1
  -- chapter_143_line_190: GL tag disintegration.
  have row_190 : (add previous zero v28) := by
    exact witness_row_191.2
  -- chapter_143_line_187: GL tag implication.
  have row_187 : (previous = v28) := by
    apply row_188
    exact row_121
    exact row_190
  -- chapter_143_line_154: GL tag implication.
  have row_154 : (gl_existence0 N previous succ) := by
    apply row_155
    exact row_121
  -- chapter_143_line_153: GL tag expansion.
  have row_153 : (¬ (∀ (v26 : α), ((N v26) → (¬ (succ previous v26))))) := by
    simpa only [gl_existence0] using row_154
  have exists_row_153 : ∃ (v26 : α), ((N v26) ∧ (succ previous v26)) := existsAndOfNotForallImpNot row_153
  obtain ⟨v26, witness_row_153⟩ := exists_row_153
  -- chapter_143_line_157: GL tag disintegration.
  have row_157 : (N v26) := by
    exact witness_row_153.1
  -- chapter_143_line_152: GL tag disintegration.
  have row_152 : (succ previous v26) := by
    exact witness_row_153.2
  -- chapter_143_line_158: GL tag implication.
  have row_158 : (v26 = v2) := by
    apply row_150
    exact row_121
    exact row_152
    exact row_31
  -- chapter_143_line_149: GL tag implication.
  have row_149 : (v2 = v26) := by
    apply row_150
    exact row_121
    exact row_31
    exact row_152
  -- chapter_143_line_146: GL tag implication.
  have row_146 : (add v2 zero v26) := by
    apply row_147
    exact row_149
    exact row_131
    exact row_157
  -- chapter_143_line_145: GL tag equality1.
  have row_145 : (add v2 zero v2) := by
    have equality_source := row_146
    have equality_step_1 := row_158
    cases equality_step_1
    exact equality_source
  -- chapter_143_line_126: GL tag implication.
  have row_126 : (gl_existence1 N v2 previous mul) := by
    apply row_127
    exact row_131
    exact row_121
  -- chapter_143_line_125: GL tag expansion.
  have row_125 : (¬ (∀ (v6 : α), ((N v6) → (¬ (mul v2 previous v6))))) := by
    simpa only [gl_existence1] using row_126
  have exists_row_125 : ∃ (v6 : α), ((N v6) ∧ (mul v2 previous v6)) := existsAndOfNotForallImpNot row_125
  obtain ⟨v6, witness_row_125⟩ := exists_row_125
  -- chapter_143_line_124: GL tag disintegration.
  have row_124 : (mul v2 previous v6) := by
    exact witness_row_125.2
  -- chapter_143_line_213: GL tag equality1.
  have row_213 : (mul v2 v28 v6) := by
    have equality_source := row_124
    have equality_step_1 := row_187
    cases equality_step_1
    exact equality_source
  -- chapter_143_line_116: GL tag implication.
  have row_116 : (gl_existence1 N zero previous add) := by
    apply row_117
    exact row_63
    exact row_121
  -- chapter_143_line_115: GL tag expansion.
  have row_115 : (¬ (∀ (v24 : α), ((N v24) → (¬ (add zero previous v24))))) := by
    simpa only [gl_existence1] using row_116
  have exists_row_115 : ∃ (v24 : α), ((N v24) ∧ (add zero previous v24)) := existsAndOfNotForallImpNot row_115
  obtain ⟨v24, witness_row_115⟩ := exists_row_115
  -- chapter_143_line_114: GL tag disintegration.
  have row_114 : (add zero previous v24) := by
    exact witness_row_115.2
  -- chapter_143_line_200: GL tag equality1.
  have row_200 : (add zero v28 v24) := by
    have equality_source := row_114
    have equality_step_1 := row_187
    cases equality_step_1
    exact equality_source
  -- chapter_143_line_57: GL tag implication.
  have row_57 : (N one) := by
    apply row_58
    exact row_62
  -- chapter_143_line_163: GL tag implication.
  have row_163 : (gl_existence1 N one previous add) := by
    apply row_117
    exact row_57
    exact row_121
  -- chapter_143_line_162: GL tag expansion.
  have row_162 : (¬ (∀ (v25 : α), ((N v25) → (¬ (add one previous v25))))) := by
    simpa only [gl_existence1] using row_163
  have exists_row_162 : ∃ (v25 : α), ((N v25) ∧ (add one previous v25)) := existsAndOfNotForallImpNot row_162
  obtain ⟨v25, witness_row_162⟩ := exists_row_162
  -- chapter_143_line_168: GL tag disintegration.
  have row_168 : (N v25) := by
    exact witness_row_162.1
  -- chapter_143_line_161: GL tag disintegration.
  have row_161 : (add one previous v25) := by
    exact witness_row_162.2
  -- chapter_143_line_52: GL tag disintegration.
  have row_52 : (gl_implication7 N succ) := by
    exact row_53.1.1.1.1.1.1.1.1.1.2
  -- chapter_143_line_51: GL tag expansion.
  have row_51 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w2 w1) → (∀ (w3 : α), ((succ w3 w1) → (w2 = w3))))))) := by
    simpa only [gl_implication7] using row_52
  -- chapter_143_line_4: GL tag expansion for integration.
  have row_4 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_143_line_3: GL tag reformulation for integration and.
  have row_3 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_143_line_2: GL tag implication.
  have row_2 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_3
    exact row_5
    exact row_8
  have rule_row_211 := peano_source_022 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_143_line_211: GL tag implication.
  have row_211 : (mul v2 v1 v3) := by
    apply rule_row_211
    exact row_212
  have rule_row_160 := peano_source_033 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_143_line_160: GL tag implication.
  have row_160 : (add one previous v2) := by
    apply rule_row_160
    exact row_31
  have rule_row_219 := peano_source_037 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_143_line_219: GL tag implication.
  have row_219 : (add previous one v2) := by
    apply rule_row_219
    exact row_160
  -- chapter_143_line_215: GL tag implication.
  have row_215 : (add previous two v1) := by
    apply row_216
    exact row_57
    exact row_62
    exact row_219
    exact row_218
  -- chapter_143_line_214: GL tag equality1.
  have row_214 : (add v28 two v1) := by
    have equality_source := row_215
    have equality_step_1 := row_187
    cases equality_step_1
    exact equality_source
  have rule_row_210 := peano_source_018 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_143_line_210: GL tag implication.
  have row_210 : (add v6 v8 v3) := by
    apply rule_row_210
    exact row_213
    exact row_135
    exact row_211
    exact row_214
  -- chapter_143_line_199: GL tag implication.
  have row_199 : (v25 = v2) := by
    apply row_142
    exact row_57
    exact row_121
    exact row_161
    exact row_160
  -- chapter_143_line_159: GL tag implication.
  have row_159 : (v2 = v25) := by
    apply row_142
    exact row_57
    exact row_121
    exact row_160
    exact row_161
  -- chapter_143_line_198: GL tag equality1.
  have row_198 : (add v2 zero v25) := by
    have equality_source := row_145
    have equality_step_1 := row_159
    cases equality_step_1
    exact equality_source
  -- chapter_143_line_195: GL tag implication.
  have row_195 : (add zero v2 v25) := by
    apply row_196
    exact row_198
  -- chapter_143_line_194: GL tag equality1.
  have row_194 : (add zero v2 v2) := by
    have equality_source := row_195
    have equality_step_1 := row_199
    cases equality_step_1
    exact equality_source
  -- chapter_143_line_164: GL tag equality1.
  have row_164 : (add v25 zero v7) := by
    have equality_source := row_165
    have equality_step_1 := row_159
    cases equality_step_1
    exact equality_source
  -- chapter_143_line_144: GL tag equality1.
  have row_144 : (add v25 zero v2) := by
    have equality_source := row_145
    have equality_step_1 := row_159
    cases equality_step_1
    exact equality_source
  -- chapter_143_line_141: GL tag implication.
  have row_141 : (v2 = v7) := by
    apply row_142
    exact row_168
    exact row_63
    exact row_144
    exact row_164
  -- chapter_143_line_193: GL tag equality1.
  have row_193 : (add zero v7 v2) := by
    have equality_source := row_194
    have equality_step_1 := row_141
    cases equality_step_1
    exact equality_source
  -- chapter_143_line_186: GL tag equality1.
  have row_186 : (succ previous v7) := by
    have equality_source := row_31
    have equality_step_1 := row_141
    cases equality_step_1
    exact equality_source
  -- chapter_143_line_185: GL tag equality1.
  have row_185 : (succ v28 v7) := by
    have equality_source := row_186
    have equality_step_1 := row_187
    cases equality_step_1
    exact equality_source
  -- chapter_143_line_182: GL tag implication.
  have row_182 : (succ v24 v2) := by
    apply row_183
    exact row_201
    exact row_185
    exact row_200
    exact row_193
  -- chapter_143_line_181: GL tag implication.
  have row_181 : (identity v2 v7) := by
    apply row_46
    exact row_131
    exact row_141
  -- chapter_143_line_134: GL tag equality1.
  have row_134 : (mul v7 two v8) := by
    have equality_source := row_135
    have equality_step_1 := row_141
    cases equality_step_1
    exact equality_source
  have rule_row_133 := peano_source_022 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_143_line_133: GL tag implication.
  have row_133 : (mul two v7 v8) := by
    apply rule_row_133
    exact row_134
  have rule_row_113 := peano_source_055 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_143_line_113: GL tag implication.
  have row_113 : (previous = v24) := by
    apply rule_row_113
    exact row_121
    exact row_114
  -- chapter_143_line_122: GL tag equality1.
  have row_122 : (gl_sequence N add zero v24 V2) := by
    have equality_source := row_123
    have equality_step_1 := row_113
    cases equality_step_1
    exact equality_source
  -- chapter_143_line_112: GL tag equality1.
  have row_112 : (gl_interval N add zero v24 V1) := by
    have equality_source := row_68
    have equality_step_1 := row_113
    cases equality_step_1
    exact equality_source
  -- chapter_143_line_209: GL tag implication.
  have row_209 : (V1 v24) := by
    apply row_73
    exact row_112
  -- chapter_143_line_208: GL tag implication.
  have row_208 : (V4 v24) := by
    apply row_94
    exact row_209
  -- chapter_143_line_110: GL tag implication.
  have row_110 : (gl_fXY V2 V1 N) := by
    apply row_111
    exact row_112
    exact row_122
  -- chapter_143_line_109: GL tag expansion.
  have row_109 : ((((gl_implication0 V2 V1) ∧ (gl_implication1 V2 N)) ∧ (gl_implication4 V1 N V2)) ∧ (gl_implication5 V1 V2)) := by
    simpa only [gl_fXY] using row_110
  -- chapter_143_line_108: GL tag disintegration.
  have row_108 : (gl_implication4 V1 N V2) := by
    exact row_109.1.2
  -- chapter_143_line_107: GL tag expansion.
  have row_107 : (∀ (w1 : α), ((V1 w1) → (gl_existence0 N w1 V2))) := by
    simpa only [gl_implication4] using row_108
  -- chapter_143_line_106: GL tag implication.
  have row_106 : (gl_existence0 N previous V2) := by
    apply row_107
    exact row_72
  -- chapter_143_line_105: GL tag expansion.
  have row_105 : (¬ (∀ (v5 : α), ((N v5) → (¬ (V2 previous v5))))) := by
    simpa only [gl_existence0] using row_106
  have exists_row_105 : ∃ (v5 : α), ((N v5) ∧ (V2 previous v5)) := existsAndOfNotForallImpNot row_105
  obtain ⟨v5, witness_row_105⟩ := exists_row_105
  -- chapter_143_line_174: GL tag disintegration.
  have row_174 : (N v5) := by
    exact witness_row_105.1
  -- chapter_143_line_173: GL tag implication.
  have row_173 : (gl_existence1 N v5 v2 add) := by
    apply row_117
    exact row_174
    exact row_131
  -- chapter_143_line_172: GL tag expansion.
  have row_172 : (¬ (∀ (v27 : α), ((N v27) → (¬ (add v5 v2 v27))))) := by
    simpa only [gl_existence1] using row_173
  have exists_row_172 : ∃ (v27 : α), ((N v27) ∧ (add v5 v2 v27)) := existsAndOfNotForallImpNot row_172
  obtain ⟨v27, witness_row_172⟩ := exists_row_172
  -- chapter_143_line_171: GL tag disintegration.
  have row_171 : (add v5 v2 v27) := by
    exact witness_row_172.2
  -- chapter_143_line_204: GL tag equality1.
  have row_204 : (add v5 v7 v27) := by
    have equality_source := row_171
    have equality_step_1 := row_141
    cases equality_step_1
    exact equality_source
  -- chapter_143_line_104: GL tag disintegration.
  have row_104 : (V2 previous v5) := by
    exact witness_row_105.2
  -- chapter_143_line_203: GL tag equality1.
  have row_203 : (V2 v24 v5) := by
    have equality_source := row_104
    have equality_step_1 := row_113
    cases equality_step_1
    exact equality_source
  -- chapter_143_line_202: GL tag implication.
  have row_202 : (V3 v24 v5) := by
    apply row_88
    exact row_203
  -- chapter_143_line_180: GL tag implication.
  have row_180 : (V3 v2 v27) := by
    apply row_83
    exact row_208
    exact row_182
    exact row_205
    exact row_202
    exact row_181
    exact row_204
  -- chapter_143_line_175: GL tag implication.
  have row_175 : (v27 = v4) := by
    apply row_176
    exact row_205
    exact row_180
    exact row_179
  -- chapter_143_line_170: GL tag equality1.
  have row_170 : (add v5 v2 v4) := by
    have equality_source := row_171
    have equality_step_1 := row_175
    cases equality_step_1
    exact equality_source
  -- chapter_143_line_169: GL tag equality1.
  have row_169 : (add v5 v7 v4) := by
    have equality_source := row_170
    have equality_step_1 := row_141
    cases equality_step_1
    exact equality_source
  have rule_row_56 := peano_source_063 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_143_line_56: GL tag implication.
  have row_56 : (gl_existence3 N one succ) := by
    apply rule_row_56
  -- chapter_143_line_55: GL tag expansion.
  have row_55 : (¬ (∀ (v13 : α), ((N v13) → (¬ (succ v13 one))))) := by
    simpa only [gl_existence3] using row_56
  have exists_row_55 : ∃ (v13 : α), ((N v13) ∧ (succ v13 one)) := existsAndOfNotForallImpNot row_55
  obtain ⟨v13, witness_row_55⟩ := exists_row_55
  -- chapter_143_line_54: GL tag disintegration.
  have row_54 : (succ v13 one) := by
    exact witness_row_55.2
  -- chapter_143_line_74: GL tag implication.
  have row_74 : (v13 = zero) := by
    apply row_51
    exact row_57
    exact row_54
    exact row_8
  -- chapter_143_line_50: GL tag implication.
  have row_50 : (zero = v13) := by
    apply row_51
    exact row_57
    exact row_8
    exact row_54
  -- chapter_143_line_45: GL tag implication.
  have row_45 : (identity zero v13) := by
    apply row_46
    exact row_63
    exact row_50
  -- chapter_143_line_40: GL tag implication.
  have row_40 : (V3 zero v13) := by
    apply row_41
    exact row_45
  -- chapter_143_line_23: GL tag implication.
  have row_23 : (V2 zero v13) := by
    apply row_24
    exact row_64
    exact row_40
  -- chapter_143_line_22: GL tag equality1.
  have row_22 : (V2 zero zero) := by
    have equality_source := row_23
    have equality_step_1 := row_74
    cases equality_step_1
    exact equality_source
  -- chapter_143_line_21: GL tag equality1.
  have row_21 : (∀ (v12 : α), ((identity zero v12) → (V2 zero v12))) := by
    intro v12
    intro scope_premise_1
    have scoped_fact_2 := row_75 v12 scope_premise_1
    have equality_source := row_22
    have equality_step_1 := scoped_fact_2
    cases equality_step_1
    exact equality_source
  -- chapter_143_line_20: GL tag validity name.
  have row_20 : (gl_implication32 zero identity V2) := by
    simpa only [gl_implication32] using row_21
  -- chapter_143_line_17: GL tag implication.
  have row_17 : (gl_and0 zero identity V2 previous v5 V1 succ add) := by
    apply row_18
    exact row_20
    exact row_104
    exact row_80
  -- chapter_143_line_14: GL tag implication.
  have row_14 : (gl_existence2 V1 N zero identity previous v5 succ add) := by
    apply row_15
    exact row_110
    exact row_17
  -- chapter_143_line_11: GL tag implication.
  have row_11 : (gl_fold N succ add identity zero previous v5) := by
    apply row_12
    exact row_68
    exact row_14
  -- chapter_143_line_9: GL tag implication.
  have row_9 : (mul two v5 v6) := by
    apply row_10
    exact row_124
    exact row_11
    exact row_31
  have rule_row_1 := peano_source_000 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_143_line_1: GL tag implication.
  have row_1 : (mul two v4 v3) := by
    apply rule_row_1
    exact row_210
    exact row_169
    exact row_9
    exact row_133
  exact row_1

theorem gauss_source_065
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorGauss N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α) (v3 : α), ((mul v1 v2 v3) → (∀ (v4 : α), ((gl_fold N succ add identity zero v2 v4) → ((succ v2 v1) → (mul two v4 v3)))))) := by
  intro v1
  intro v2
  intro v3
  intro premise_1
  intro v4
  intro premise_2
  intro premise_3
  have inductionMember : N v2 := by
    have typingRule := gauss_source_065_induction_typing N zero succ add mul one two identity anchor relationalInduction
    exact typingRule v2 v4 premise_2
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorGauss, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v1 : α) (v3 : α), ((mul v1 zero v3) → (∀ (v4 : α), ((gl_fold N succ add identity zero zero v4) → ((succ zero v1) → (mul two v4 v3)))))) := by
    intro v1
    intro v3
    intro base_premise_1
    intro v4
    intro base_premise_2
    intro base_premise_3
    have zeroRule := gauss_source_065_check_zero N zero succ add mul one two identity anchor relationalInduction
    exact zeroRule v1 zero v3 v4 base_premise_3 base_premise_1 rfl base_premise_2
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v1 : α) (v3 : α), ((mul v1 induction_n v3) → (∀ (v4 : α), ((gl_fold N succ add identity zero induction_n v4) → ((succ induction_n v1) → (mul two v4 v3)))))) → ∀ induction_m, succ induction_n induction_m → (∀ (v1 : α) (v3 : α), ((mul v1 induction_m v3) → (∀ (v4 : α), ((gl_fold N succ add identity zero induction_m v4) → ((succ induction_m v1) → (mul two v4 v3)))))) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
    intro v1
    intro v3
    intro step_premise_1
    intro v4
    intro step_premise_2
    intro step_premise_3
    have step_induction_assumption_1 :
        (succ induction_n induction_m) := by
      exact induction_successor
    have step_induction_assumption_2 :
        (∀ (w1 : α) (w2 : α), ((mul w1 induction_n w2) → (∀ (w3 : α), ((gl_fold N succ add identity zero induction_n w3) → ((succ induction_n w1) → (mul two w3 w2)))))) := by
      intro w1
      intro w2
      intro step_induction_assumption_2_premise_1
      intro w3
      intro step_induction_assumption_2_premise_2
      intro step_induction_assumption_2_premise_3
      apply induction_hypothesis
      all_goals assumption
    have stepRule := gauss_source_065_check_induction_condition N zero succ add mul one two identity anchor relationalInduction
    exact stepRule induction_n v1 induction_m v3 v4 step_premise_3 step_premise_1 step_premise_2 step_induction_assumption_1 step_induction_assumption_2
  have inductionProperty : (∀ (v1 : α) (v3 : α), ((mul v1 v2 v3) → (∀ (v4 : α), ((gl_fold N succ add identity zero v2 v4) → ((succ v2 v1) → (mul two v4 v3)))))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v1 : α) (v3 : α), ((mul v1 induction_value v3) → (∀ (v4 : α), ((gl_fold N succ add identity zero induction_value v4) → ((succ induction_value v1) → (mul two v4 v3)))))))
      v2
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v1 v3 premise_1 v4 premise_2 premise_3

private theorem gauss_source_066_induction_typing
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorGauss N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    (v1 : α)
    (v4 : α)
    (assumption_5 : (gl_fold N succ add identity zero v1 v4))
    : (N v1) := by
  -- chapter_144_line_5: GL tag task formulation.
  have row_5 : (gl_fold N succ add identity zero v1 v4) := by
    exact assumption_5
  -- chapter_144_line_4: GL tag expansion.
  have row_4 : (¬ (∀ (V1 : GLSet α), ((gl_interval N add zero v1 V1) → (¬ (gl_existence2 V1 N zero identity v1 v4 succ add))))) := by
    simpa only [gl_fold] using row_5
  have exists_row_4 : ∃ (V1 : GLSet α), ((gl_interval N add zero v1 V1) ∧ (gl_existence2 V1 N zero identity v1 v4 succ add)) := existsAndOfNotForallImpNot row_4
  obtain ⟨V1, witness_row_4⟩ := exists_row_4
  -- chapter_144_line_3: GL tag disintegration.
  have row_3 : (gl_interval N add zero v1 V1) := by
    exact witness_row_4.1
  -- chapter_144_line_2: GL tag expansion.
  have row_2 : (((((gl_implication26 V1 N add zero) ∧ (gl_implication27 V1 N add v1)) ∧ (gl_implication28 N add zero v1 V1)) ∧ (N zero)) ∧ (N v1)) := by
    simpa only [gl_interval] using row_3
  -- chapter_144_line_1: GL tag disintegration.
  have row_1 : (N v1) := by
    exact row_2.2
  exact row_1

private theorem gauss_source_066_check_zero
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorGauss N zero succ add mul one two identity)
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
    (assumption_42 : (v1 = zero))
    (assumption_41 : (mul v1 v2 v3))
    (assumption_21 : (gl_fold N succ add identity zero v1 v4))
    : (mul two v4 v3) := by
  -- chapter_145_line_42: GL tag recursion.
  have row_42 : (v1 = zero) := by
    exact assumption_42
  -- chapter_145_line_85: GL tag symmetry of equality.
  have row_85 : (zero = v1) := by
    exact Eq.symm row_42
  -- chapter_145_line_41: GL tag task formulation.
  have row_41 : (mul v1 v2 v3) := by
    exact assumption_41
  -- chapter_145_line_40: GL tag equality1.
  have row_40 : (mul zero v2 v3) := by
    have equality_source := row_41
    have equality_step_1 := row_42
    cases equality_step_1
    exact equality_source
  -- chapter_145_line_38: GL tag expansion for integration.
  have row_38 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_145_line_37: GL tag reformulation for integration and.
  have row_37 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_145_line_21: GL tag task formulation.
  have row_21 : (gl_fold N succ add identity zero v1 v4) := by
    exact assumption_21
  -- chapter_145_line_20: GL tag expansion.
  have row_20 : (¬ (∀ (V2 : GLSet α), ((gl_interval N add zero v1 V2) → (¬ (gl_existence2 V2 N zero identity v1 v4 succ add))))) := by
    simpa only [gl_fold] using row_21
  have exists_row_20 : ∃ (V2 : GLSet α), ((gl_interval N add zero v1 V2) ∧ (gl_existence2 V2 N zero identity v1 v4 succ add)) := existsAndOfNotForallImpNot row_20
  obtain ⟨V2, witness_row_20⟩ := exists_row_20
  -- chapter_145_line_19: GL tag disintegration.
  have row_19 : (gl_existence2 V2 N zero identity v1 v4 succ add) := by
    exact witness_row_20.2
  -- chapter_145_line_18: GL tag expansion.
  have row_18 : (¬ (∀ (V1 : GLBinaryRelation α), ((gl_fXY V1 V2 N) → (¬ (gl_and0 zero identity V1 v1 v4 V2 succ add))))) := by
    simpa only [gl_existence2] using row_19
  have exists_row_18 : ∃ (V1 : GLBinaryRelation α), ((gl_fXY V1 V2 N) ∧ (gl_and0 zero identity V1 v1 v4 V2 succ add)) := existsAndOfNotForallImpNot row_18
  obtain ⟨V1, witness_row_18⟩ := exists_row_18
  -- chapter_145_line_24: GL tag disintegration.
  have row_24 : (gl_and0 zero identity V1 v1 v4 V2 succ add) := by
    exact witness_row_18.2
  -- chapter_145_line_23: GL tag expansion.
  have row_23 : (((gl_implication32 zero identity V1) ∧ (V1 v1 v4)) ∧ (gl_implication33 V2 succ V1 identity add)) := by
    simpa only [gl_and0] using row_24
  -- chapter_145_line_71: GL tag disintegration.
  have row_71 : (gl_implication32 zero identity V1) := by
    exact row_23.1.1
  -- chapter_145_line_70: GL tag expansion.
  have row_70 : (∀ (w1 : α), ((identity zero w1) → (V1 zero w1))) := by
    simpa only [gl_implication32] using row_71
  -- chapter_145_line_22: GL tag disintegration.
  have row_22 : (V1 v1 v4) := by
    exact row_23.1.2
  -- chapter_145_line_66: GL tag equality1.
  have row_66 : (V1 zero v4) := by
    have equality_source := row_22
    have equality_step_1 := row_42
    cases equality_step_1
    exact equality_source
  -- chapter_145_line_17: GL tag disintegration.
  have row_17 : (gl_fXY V1 V2 N) := by
    exact witness_row_18.1
  -- chapter_145_line_16: GL tag expansion.
  have row_16 : ((((gl_implication0 V1 V2) ∧ (gl_implication1 V1 N)) ∧ (gl_implication4 V2 N V1)) ∧ (gl_implication5 V2 V1)) := by
    simpa only [gl_fXY] using row_17
  -- chapter_145_line_88: GL tag disintegration.
  have row_88 : (gl_implication0 V1 V2) := by
    exact row_16.1.1.1
  -- chapter_145_line_87: GL tag expansion.
  have row_87 : (∀ (w1 : α) (w2 : α), ((V1 w1 w2) → (V2 w1))) := by
    simpa only [gl_implication0] using row_88
  -- chapter_145_line_93: GL tag implication.
  have row_93 : (V2 v1) := by
    apply row_87
    exact row_22
  -- chapter_145_line_65: GL tag disintegration.
  have row_65 : (gl_implication5 V2 V1) := by
    exact row_16.2
  -- chapter_145_line_64: GL tag expansion.
  have row_64 : (∀ (w1 : α), ((V2 w1) → (∀ (w2 : α), ((V1 w1 w2) → (∀ (w3 : α), ((V1 w1 w3) → (w2 = w3))))))) := by
    simpa only [gl_implication5] using row_65
  -- chapter_145_line_15: GL tag disintegration.
  have row_15 : (gl_implication1 V1 N) := by
    exact row_16.1.1.2
  -- chapter_145_line_14: GL tag expansion.
  have row_14 : (∀ (w1 : α) (w2 : α), ((V1 w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using row_15
  -- chapter_145_line_13: GL tag implication.
  have row_13 : (N v4) := by
    apply row_14
    exact row_22
  -- chapter_145_line_12: GL tag task formulation.
  have row_12 : (gl_AnchorGauss N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_145_line_11: GL tag expansion.
  have row_11 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorGauss] using row_12
  -- chapter_145_line_76: GL tag disintegration.
  have row_76 : (gl_identity N identity) := by
    exact row_11.2
  -- chapter_145_line_75: GL tag expansion.
  have row_75 : (((gl_implication0 identity N) ∧ (gl_implication22 identity)) ∧ (gl_implication23 N identity)) := by
    simpa only [gl_identity] using row_76
  -- chapter_145_line_74: GL tag disintegration.
  have row_74 : (gl_implication23 N identity) := by
    exact row_75.2
  -- chapter_145_line_73: GL tag expansion.
  have row_73 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((w1 = w2) → (identity w1 w2))))) := by
    simpa only [gl_implication23] using row_74
  -- chapter_145_line_39: GL tag disintegration.
  have row_39 : (succ zero one) := by
    exact row_11.1.1.2
  -- chapter_145_line_30: GL tag disintegration.
  have row_30 : (succ one two) := by
    exact row_11.1.2
  -- chapter_145_line_10: GL tag disintegration.
  have row_10 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_11.1.1.1
  -- chapter_145_line_36: GL tag implication.
  have row_36 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_37
    exact row_10
    exact row_39
  have rule_row_82 := peano_source_063 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_145_line_82: GL tag implication.
  have row_82 : (gl_existence3 N one succ) := by
    apply rule_row_82
  -- chapter_145_line_81: GL tag expansion.
  have row_81 : (¬ (∀ (v9 : α), ((N v9) → (¬ (succ v9 one))))) := by
    simpa only [gl_existence3] using row_82
  have exists_row_81 : ∃ (v9 : α), ((N v9) ∧ (succ v9 one)) := existsAndOfNotForallImpNot row_81
  obtain ⟨v9, witness_row_81⟩ := exists_row_81
  -- chapter_145_line_80: GL tag disintegration.
  have row_80 : (succ v9 one) := by
    exact witness_row_81.2
  -- chapter_145_line_9: GL tag expansion.
  have row_9 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_10
  -- chapter_145_line_83: GL tag disintegration.
  have row_83 : (N zero) := by
    exact row_9.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_145_line_79: GL tag disintegration.
  have row_79 : (gl_implication7 N succ) := by
    exact row_9.1.1.1.1.1.1.1.1.1.2
  -- chapter_145_line_78: GL tag expansion.
  have row_78 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w2 w1) → (∀ (w3 : α), ((succ w3 w1) → (w2 = w3))))))) := by
    simpa only [gl_implication7] using row_79
  -- chapter_145_line_52: GL tag disintegration.
  have row_52 : (gl_implication16 N zero add) := by
    exact row_9.1.1.1.1.1.1.2
  -- chapter_145_line_51: GL tag expansion.
  have row_51 : (∀ (w1 : α) (w2 : α), ((w1 = w2) → ((N w1) → ((N w2) → (add w1 zero w2))))) := by
    simpa only [gl_implication16] using row_52
  -- chapter_145_line_34: GL tag disintegration.
  have row_34 : (gl_implication19 N zero mul) := by
    exact row_9.1.1.2
  -- chapter_145_line_29: GL tag disintegration.
  have row_29 : (gl_fXY succ N N) := by
    exact row_9.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_145_line_28: GL tag expansion.
  have row_28 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_29
  -- chapter_145_line_60: GL tag disintegration.
  have row_60 : (gl_implication0 succ N) := by
    exact row_28.1.1.1
  -- chapter_145_line_59: GL tag expansion.
  have row_59 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_60
  -- chapter_145_line_58: GL tag implication.
  have row_58 : (N one) := by
    apply row_59
    exact row_30
  -- chapter_145_line_84: GL tag implication.
  have row_84 : (v9 = zero) := by
    apply row_78
    exact row_58
    exact row_80
    exact row_39
  -- chapter_145_line_77: GL tag implication.
  have row_77 : (zero = v9) := by
    apply row_78
    exact row_58
    exact row_39
    exact row_80
  -- chapter_145_line_72: GL tag implication.
  have row_72 : (identity zero v9) := by
    apply row_73
    exact row_83
    exact row_77
  -- chapter_145_line_69: GL tag implication.
  have row_69 : (V1 zero v9) := by
    apply row_70
    exact row_72
  -- chapter_145_line_86: GL tag implication.
  have row_86 : (V2 zero) := by
    apply row_87
    exact row_69
  -- chapter_145_line_89: GL tag implication.
  have row_89 : (v4 = v9) := by
    apply row_64
    exact row_86
    exact row_66
    exact row_69
  -- chapter_145_line_68: GL tag equality1.
  have row_68 : (V1 zero zero) := by
    have equality_source := row_69
    have equality_step_1 := row_84
    cases equality_step_1
    exact equality_source
  -- chapter_145_line_92: GL tag equality1.
  have row_92 : (V1 v1 zero) := by
    have equality_source := row_68
    have equality_step_1 := row_85
    cases equality_step_1
    exact equality_source
  -- chapter_145_line_91: GL tag implication.
  have row_91 : (zero = v4) := by
    apply row_64
    exact row_93
    exact row_92
    exact row_22
  -- chapter_145_line_67: GL tag equality1.
  have row_67 : (V1 zero v1) := by
    have equality_source := row_68
    have equality_step_1 := row_85
    cases equality_step_1
    exact equality_source
  -- chapter_145_line_63: GL tag implication.
  have row_63 : (v4 = v1) := by
    apply row_64
    exact row_86
    exact row_66
    exact row_67
  -- chapter_145_line_27: GL tag disintegration.
  have row_27 : (gl_implication1 succ N) := by
    exact row_28.1.1.2
  -- chapter_145_line_26: GL tag expansion.
  have row_26 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using row_27
  -- chapter_145_line_25: GL tag implication.
  have row_25 : (N two) := by
    apply row_26
    exact row_30
  -- chapter_145_line_8: GL tag disintegration.
  have row_8 : (gl_fXYZ mul N N N) := by
    exact row_9.1.1.1.2
  -- chapter_145_line_7: GL tag expansion.
  have row_7 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_8
  -- chapter_145_line_45: GL tag disintegration.
  have row_45 : (gl_implication9 mul N) := by
    exact row_7.1.1.1.2
  -- chapter_145_line_44: GL tag expansion.
  have row_44 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication9] using row_45
  -- chapter_145_line_43: GL tag implication.
  have row_43 : (N v2) := by
    apply row_44
    exact row_41
  have rule_row_35 := peano_source_056 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_145_line_35: GL tag implication.
  have row_35 : (zero = v3) := by
    apply rule_row_35
    exact row_43
    exact row_40
  -- chapter_145_line_33: GL tag equality1.
  have row_33 : (gl_implication19 N v3 mul) := by
    have equality_source := row_34
    have equality_step_1 := row_35
    cases equality_step_1
    exact equality_source
  -- chapter_145_line_32: GL tag expansion.
  have row_32 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((mul w1 v3 w2) → (w2 = v3))))) := by
    simpa only [gl_implication19] using row_33
  -- chapter_145_line_6: GL tag disintegration.
  have row_6 : (gl_implication13 N N N mul) := by
    exact row_7.1.2
  -- chapter_145_line_5: GL tag expansion.
  have row_5 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication13] using row_6
  -- chapter_145_line_57: GL tag implication.
  have row_57 : (gl_existence1 N v4 one mul) := by
    apply row_5
    exact row_13
    exact row_58
  -- chapter_145_line_56: GL tag expansion.
  have row_56 : (¬ (∀ (v10 : α), ((N v10) → (¬ (mul v4 one v10))))) := by
    simpa only [gl_existence1] using row_57
  have exists_row_56 : ∃ (v10 : α), ((N v10) ∧ (mul v4 one v10)) := existsAndOfNotForallImpNot row_56
  obtain ⟨v10, witness_row_56⟩ := exists_row_56
  -- chapter_145_line_61: GL tag disintegration.
  have row_61 : (N v10) := by
    exact witness_row_56.1
  -- chapter_145_line_55: GL tag disintegration.
  have row_55 : (mul v4 one v10) := by
    exact witness_row_56.2
  have rule_row_54 := peano_source_044 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_145_line_54: GL tag implication.
  have row_54 : (mul one v4 v10) := by
    apply rule_row_54
    exact row_55
  have rule_row_62 := peano_source_058 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_145_line_62: GL tag implication.
  have row_62 : (v10 = v4) := by
    apply rule_row_62
    exact row_61
    exact row_54
  have rule_row_53 := peano_source_057 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_145_line_53: GL tag implication.
  have row_53 : (v4 = v10) := by
    apply rule_row_53
    exact row_13
    exact row_54
  -- chapter_145_line_50: GL tag implication.
  have row_50 : (add v4 zero v10) := by
    apply row_51
    exact row_53
    exact row_13
    exact row_61
  -- chapter_145_line_49: GL tag equality1.
  have row_49 : (add v4 zero v4) := by
    have equality_source := row_50
    have equality_step_1 := row_62
    cases equality_step_1
    exact equality_source
  -- chapter_145_line_90: GL tag equality1.
  have row_90 : (add v1 v4 v9) := by
    have equality_source := row_49
    have equality_step_1 := row_63
    cases equality_step_1
    have equality_step_2 := row_89
    cases equality_step_2
    have equality_step_3 := row_91
    cases equality_step_3
    exact equality_source
  -- chapter_145_line_48: GL tag equality1.
  have row_48 : (add v1 v3 v9) := by
    have equality_source := row_49
    have equality_step_1 := row_63
    cases equality_step_1
    have equality_step_2 := row_89
    cases equality_step_2
    have equality_step_3 := row_35
    cases equality_step_3
    exact equality_source
  have rule_row_47 := peano_source_013 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_145_line_47: GL tag implication.
  have row_47 : (v4 = v3) := by
    apply rule_row_47
    exact row_90
    exact row_48
  -- chapter_145_line_4: GL tag implication.
  have row_4 : (gl_existence1 N two v4 mul) := by
    apply row_5
    exact row_25
    exact row_13
  -- chapter_145_line_3: GL tag expansion.
  have row_3 : (¬ (∀ (v5 : α), ((N v5) → (¬ (mul two v4 v5))))) := by
    simpa only [gl_existence1] using row_4
  have exists_row_3 : ∃ (v5 : α), ((N v5) ∧ (mul two v4 v5)) := existsAndOfNotForallImpNot row_3
  obtain ⟨v5, witness_row_3⟩ := exists_row_3
  -- chapter_145_line_2: GL tag disintegration.
  have row_2 : (mul two v4 v5) := by
    exact witness_row_3.2
  -- chapter_145_line_46: GL tag equality1.
  have row_46 : (mul two v3 v5) := by
    have equality_source := row_2
    have equality_step_1 := row_47
    cases equality_step_1
    exact equality_source
  -- chapter_145_line_31: GL tag implication.
  have row_31 : (v5 = v3) := by
    apply row_32
    exact row_25
    exact row_46
  -- chapter_145_line_1: GL tag equality1.
  have row_1 : (mul two v4 v3) := by
    have equality_source := row_2
    have equality_step_1 := row_31
    cases equality_step_1
    exact equality_source
  exact row_1

private theorem gauss_source_066_check_induction_condition
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorGauss N zero succ add mul one two identity)
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
    (assumption_219 : (mul v1 v2 v3))
    (assumption_217 : (succ v1 v2))
    (assumption_38 : (gl_fold N succ add identity zero v1 v4))
    (assumption_31 : (succ previous v1))
    (assumption_10 : (∀ (w1 : α) (w2 : α), ((mul previous w1 w2) → (∀ (w3 : α), ((gl_fold N succ add identity zero previous w3) → ((succ previous w1) → (mul two w3 w2)))))))
    : (mul two v4 v3) := by
  -- chapter_146_line_219: GL tag task formulation.
  have row_219 : (mul v1 v2 v3) := by
    exact assumption_219
  -- chapter_146_line_217: GL tag task formulation.
  have row_217 : (succ v1 v2) := by
    exact assumption_217
  -- chapter_146_line_172: GL tag theorem.
  have row_172 := gauss_source_064 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_146_line_111: GL tag theorem.
  have row_111 := gauss_source_088 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_146_line_86: GL tag expansion for integration.
  have row_86 : (∀ (V1 : GLSet α) (V2 : GLBinaryRelation α), ((gl_implication33 V1 succ V2 identity add) ↔ (∀ (v21 : α), ((V1 v21) → (∀ (v17 : α), ((succ v21 v17) → ((V1 v17) → (∀ (v22 : α), ((V2 v21 v22) → (∀ (v23 : α), ((identity v17 v23) → (∀ (v18 : α), ((add v22 v23 v18) → (V2 v17 v18)))))))))))))) := by
    intro V1 V2
    exact Iff.rfl
  -- chapter_146_line_100: GL tag premise element.
  have row_100 : (∀ (V1 : GLSet α) (V2 : GLBinaryRelation α) (v21 : α), ((V1 v21) → (∀ (v17 : α), ((succ v21 v17) → ((V1 v17) → (∀ (v22 : α), ((V2 v21 v22) → (∀ (v23 : α), ((identity v17 v23) → (∀ (v18 : α), ((add v22 v23 v18) → (V1 v17)))))))))))) := by
    intro V1
    intro V2
    intro v21
    intro scope_premise_1
    intro v17
    intro scope_premise_2
    intro scope_premise_3
    intro v22
    intro scope_premise_4
    intro v23
    intro scope_premise_5
    intro v18
    intro scope_premise_6
    exact scope_premise_3
  -- chapter_146_line_98: GL tag premise element.
  have row_98 : (∀ (V1 : GLSet α) (V2 : GLBinaryRelation α) (v21 : α), ((V1 v21) → (∀ (v17 : α), ((succ v21 v17) → ((V1 v17) → (∀ (v22 : α), ((V2 v21 v22) → (∀ (v23 : α), ((identity v17 v23) → (∀ (v18 : α), ((add v22 v23 v18) → (V1 v21)))))))))))) := by
    intro V1
    intro V2
    intro v21
    intro scope_premise_1
    intro v17
    intro scope_premise_2
    intro scope_premise_3
    intro v22
    intro scope_premise_4
    intro v23
    intro scope_premise_5
    intro v18
    intro scope_premise_6
    exact scope_premise_1
  -- chapter_146_line_92: GL tag premise element.
  have row_92 : (∀ (V1 : GLSet α) (V2 : GLBinaryRelation α) (v21 : α), ((V1 v21) → (∀ (v17 : α), ((succ v21 v17) → ((V1 v17) → (∀ (v22 : α), ((V2 v21 v22) → (∀ (v23 : α), ((identity v17 v23) → (∀ (v18 : α), ((add v22 v23 v18) → (add v22 v23 v18)))))))))))) := by
    intro V1
    intro V2
    intro v21
    intro scope_premise_1
    intro v17
    intro scope_premise_2
    intro scope_premise_3
    intro v22
    intro scope_premise_4
    intro v23
    intro scope_premise_5
    intro v18
    intro scope_premise_6
    exact scope_premise_6
  -- chapter_146_line_91: GL tag premise element.
  have row_91 : (∀ (V1 : GLSet α) (V2 : GLBinaryRelation α) (v21 : α), ((V1 v21) → (∀ (v17 : α), ((succ v21 v17) → ((V1 v17) → (∀ (v22 : α), ((V2 v21 v22) → (∀ (v23 : α), ((identity v17 v23) → (∀ (v18 : α), ((add v22 v23 v18) → (identity v17 v23)))))))))))) := by
    intro V1
    intro V2
    intro v21
    intro scope_premise_1
    intro v17
    intro scope_premise_2
    intro scope_premise_3
    intro v22
    intro scope_premise_4
    intro v23
    intro scope_premise_5
    intro v18
    intro scope_premise_6
    exact scope_premise_5
  -- chapter_146_line_90: GL tag premise element.
  have row_90 : (∀ (V1 : GLSet α) (V2 : GLBinaryRelation α) (v21 : α), ((V1 v21) → (∀ (v17 : α), ((succ v21 v17) → ((V1 v17) → (∀ (v22 : α), ((V2 v21 v22) → (∀ (v23 : α), ((identity v17 v23) → (∀ (v18 : α), ((add v22 v23 v18) → (V2 v21 v22)))))))))))) := by
    intro V1
    intro V2
    intro v21
    intro scope_premise_1
    intro v17
    intro scope_premise_2
    intro scope_premise_3
    intro v22
    intro scope_premise_4
    intro v23
    intro scope_premise_5
    intro v18
    intro scope_premise_6
    exact scope_premise_4
  -- chapter_146_line_85: GL tag premise element.
  have row_85 : (∀ (V1 : GLSet α) (V2 : GLBinaryRelation α) (v21 : α), ((V1 v21) → (∀ (v17 : α), ((succ v21 v17) → ((V1 v17) → (∀ (v22 : α), ((V2 v21 v22) → (∀ (v23 : α), ((identity v17 v23) → (∀ (v18 : α), ((add v22 v23 v18) → (succ v21 v17)))))))))))) := by
    intro V1
    intro V2
    intro v21
    intro scope_premise_1
    intro v17
    intro scope_premise_2
    intro scope_premise_3
    intro v22
    intro scope_premise_4
    intro v23
    intro scope_premise_5
    intro v18
    intro scope_premise_6
    exact scope_premise_2
  -- chapter_146_line_79: GL tag expansion for integration.
  have row_79 : (∀ (V2 : GLBinaryRelation α), ((gl_implication32 zero identity V2) ↔ (∀ (v12 : α), ((identity zero v12) → (V2 zero v12))))) := by
    intro V2
    exact Iff.rfl
  -- chapter_146_line_78: GL tag premise element.
  have row_78 : (∀ (V2 : GLBinaryRelation α) (v12 : α), ((identity zero v12) → (identity zero v12))) := by
    intro V2
    intro v12
    intro scope_premise_1
    exact scope_premise_1
  -- chapter_146_line_73: GL tag theorem.
  have row_73 := gauss_source_090 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_146_line_71: GL tag theorem.
  have row_71 := gauss_source_077 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_146_line_38: GL tag task formulation.
  have row_38 : (gl_fold N succ add identity zero v1 v4) := by
    exact assumption_38
  -- chapter_146_line_37: GL tag expansion.
  have row_37 : (¬ (∀ (V4 : GLSet α), ((gl_interval N add zero v1 V4) → (¬ (gl_existence2 V4 N zero identity v1 v4 succ add))))) := by
    simpa only [gl_fold] using row_38
  have exists_row_37 : ∃ (V4 : GLSet α), ((gl_interval N add zero v1 V4) ∧ (gl_existence2 V4 N zero identity v1 v4 succ add)) := existsAndOfNotForallImpNot row_37
  obtain ⟨V4, witness_row_37⟩ := exists_row_37
  -- chapter_146_line_39: GL tag disintegration.
  have row_39 : (gl_interval N add zero v1 V4) := by
    exact witness_row_37.1
  -- chapter_146_line_132: GL tag expansion.
  have row_132 : (((((gl_implication26 V4 N add zero) ∧ (gl_implication27 V4 N add v1)) ∧ (gl_implication28 N add zero v1 V4)) ∧ (N zero)) ∧ (N v1)) := by
    simpa only [gl_interval] using row_39
  -- chapter_146_line_131: GL tag disintegration.
  have row_131 : (N v1) := by
    exact row_132.2
  -- chapter_146_line_36: GL tag disintegration.
  have row_36 : (gl_existence2 V4 N zero identity v1 v4 succ add) := by
    exact witness_row_37.2
  -- chapter_146_line_35: GL tag expansion.
  have row_35 : (¬ (∀ (V3 : GLBinaryRelation α), ((gl_fXY V3 V4 N) → (¬ (gl_and0 zero identity V3 v1 v4 V4 succ add))))) := by
    simpa only [gl_existence2] using row_36
  have exists_row_35 : ∃ (V3 : GLBinaryRelation α), ((gl_fXY V3 V4 N) ∧ (gl_and0 zero identity V3 v1 v4 V4 succ add)) := existsAndOfNotForallImpNot row_35
  obtain ⟨V3, witness_row_35⟩ := exists_row_35
  -- chapter_146_line_44: GL tag disintegration.
  have row_44 : (gl_and0 zero identity V3 v1 v4 V4 succ add) := by
    exact witness_row_35.2
  -- chapter_146_line_43: GL tag expansion.
  have row_43 : (((gl_implication32 zero identity V3) ∧ (V3 v1 v4)) ∧ (gl_implication33 V4 succ V3 identity add)) := by
    simpa only [gl_and0] using row_44
  -- chapter_146_line_181: GL tag disintegration.
  have row_181 : (V3 v1 v4) := by
    exact row_43.1.2
  -- chapter_146_line_84: GL tag disintegration.
  have row_84 : (gl_implication33 V4 succ V3 identity add) := by
    exact row_43.2
  -- chapter_146_line_83: GL tag expansion.
  have row_83 : (∀ (w1 : α), ((V4 w1) → (∀ (w2 : α), ((succ w1 w2) → ((V4 w2) → (∀ (w3 : α), ((V3 w1 w3) → (∀ (w4 : α), ((identity w2 w4) → (∀ (w5 : α), ((add w3 w4 w5) → (V3 w2 w5)))))))))))) := by
    simpa only [gl_implication33] using row_84
  -- chapter_146_line_42: GL tag disintegration.
  have row_42 : (gl_implication32 zero identity V3) := by
    exact row_43.1.1
  -- chapter_146_line_41: GL tag expansion.
  have row_41 : (∀ (w1 : α), ((identity zero w1) → (V3 zero w1))) := by
    simpa only [gl_implication32] using row_42
  -- chapter_146_line_34: GL tag disintegration.
  have row_34 : (gl_fXY V3 V4 N) := by
    exact witness_row_35.1
  -- chapter_146_line_180: GL tag expansion.
  have row_180 : ((((gl_implication0 V3 V4) ∧ (gl_implication1 V3 N)) ∧ (gl_implication4 V4 N V3)) ∧ (gl_implication5 V4 V3)) := by
    simpa only [gl_fXY] using row_34
  -- chapter_146_line_208: GL tag disintegration.
  have row_208 : (gl_implication0 V3 V4) := by
    exact row_180.1.1.1
  -- chapter_146_line_207: GL tag expansion.
  have row_207 : (∀ (w1 : α) (w2 : α), ((V3 w1 w2) → (V4 w1))) := by
    simpa only [gl_implication0] using row_208
  -- chapter_146_line_206: GL tag implication.
  have row_206 : (V4 v1) := by
    apply row_207
    exact row_181
  -- chapter_146_line_179: GL tag disintegration.
  have row_179 : (gl_implication5 V4 V3) := by
    exact row_180.2
  -- chapter_146_line_178: GL tag expansion.
  have row_178 : (∀ (w1 : α), ((V4 w1) → (∀ (w2 : α), ((V3 w1 w2) → (∀ (w3 : α), ((V3 w1 w3) → (w2 = w3))))))) := by
    simpa only [gl_implication5] using row_179
  -- chapter_146_line_33: GL tag theorem.
  have row_33 := gauss_source_067 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_146_line_31: GL tag recursion.
  have row_31 : (succ previous v1) := by
    exact assumption_31
  -- chapter_146_line_30: GL tag theorem.
  have row_30 := gauss_source_086 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_146_line_19: GL tag expansion for integration.
  have row_19 : (∀ (V1 : GLSet α) (V2 : GLBinaryRelation α) (v5 : α), ((gl_and0 zero identity V2 previous v5 V1 succ add) ↔ (((gl_implication32 zero identity V2) ∧ (V2 previous v5)) ∧ (gl_implication33 V1 succ V2 identity add)))) := by
    intro V1 V2 v5
    exact Iff.rfl
  -- chapter_146_line_18: GL tag reformulation for integration and.
  have row_18 : (∀ (V1 : GLSet α) (V2 : GLBinaryRelation α) (v5 : α), ((gl_implication32 zero identity V2) → ((V2 previous v5) → ((gl_implication33 V1 succ V2 identity add) → (gl_and0 zero identity V2 previous v5 V1 succ add))))) := by
    intro V1
    intro V2
    intro v5
    intro integration_premise_1
    intro integration_premise_2
    intro integration_premise_3
    simp only [gl_and0]
    exact ⟨⟨integration_premise_1, integration_premise_2⟩, integration_premise_3⟩
  -- chapter_146_line_16: GL tag expansion for integration.
  have row_16 : (∀ (V1 : GLSet α) (v5 : α), ((gl_existence2 V1 N zero identity previous v5 succ add) ↔ (¬ (∀ (V2 : GLBinaryRelation α), ((gl_fXY V2 V1 N) → (¬ (gl_and0 zero identity V2 previous v5 V1 succ add))))))) := by
    intro V1 v5
    exact Iff.rfl
  -- chapter_146_line_15: GL tag reformulation for integration >[].
  have row_15 : (∀ (V1 : GLSet α) (V2 : GLBinaryRelation α) (v5 : α), ((gl_fXY V2 V1 N) → ((gl_and0 zero identity V2 previous v5 V1 succ add) → (gl_existence2 V1 N zero identity previous v5 succ add)))) := by
    intro V1
    intro V2
    intro v5
    intro integration_premise_1
    intro integration_premise_2
    apply (row_16 V1 v5).2
    intro universal_counterexample
    exact universal_counterexample V2 integration_premise_1 integration_premise_2
  -- chapter_146_line_13: GL tag expansion for integration.
  have row_13 : (∀ (v5 : α), ((gl_fold N succ add identity zero previous v5) ↔ (¬ (∀ (V1 : GLSet α), ((gl_interval N add zero previous V1) → (¬ (gl_existence2 V1 N zero identity previous v5 succ add))))))) := by
    intro v5
    exact Iff.rfl
  -- chapter_146_line_12: GL tag reformulation for integration >[].
  have row_12 : (∀ (V1 : GLSet α) (v5 : α), ((gl_interval N add zero previous V1) → ((gl_existence2 V1 N zero identity previous v5 succ add) → (gl_fold N succ add identity zero previous v5)))) := by
    intro V1
    intro v5
    intro integration_premise_1
    intro integration_premise_2
    apply (row_13 v5).2
    intro universal_counterexample
    exact universal_counterexample V1 integration_premise_1 integration_premise_2
  -- chapter_146_line_10: GL tag recursion.
  have row_10 : (∀ (w1 : α) (w2 : α), ((mul previous w1 w2) → (∀ (w3 : α), ((gl_fold N succ add identity zero previous w3) → ((succ previous w1) → (mul two w3 w2)))))) := by
    exact assumption_10
  -- chapter_146_line_7: GL tag task formulation.
  have row_7 : (gl_AnchorGauss N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_146_line_198: GL tag anchor handling.
  have row_198 : (gl_AnchorGauss N zero succ add mul one two identity) := by
    exact row_7
  -- chapter_146_line_70: GL tag implication.
  have row_70 : (gl_existence9 N add V4 previous zero) := by
    apply row_71
    exact row_31
    exact row_39
  -- chapter_146_line_69: GL tag expansion.
  have row_69 : (¬ (∀ (V1 : GLSet α), ((gl_limitSet N add V4 previous V1) → (¬ (gl_interval N add zero previous V1))))) := by
    simpa only [gl_existence9] using row_70
  have exists_row_69 : ∃ (V1 : GLSet α), ((gl_limitSet N add V4 previous V1) ∧ (gl_interval N add zero previous V1)) := existsAndOfNotForallImpNot row_69
  obtain ⟨V1, witness_row_69⟩ := exists_row_69
  -- chapter_146_line_97: GL tag disintegration.
  have row_97 : (gl_limitSet N add V4 previous V1) := by
    exact witness_row_69.1
  -- chapter_146_line_96: GL tag expansion.
  have row_96 : (((gl_implication41 V1 V4) ∧ (gl_implication27 V1 N add previous)) ∧ (gl_implication42 V4 N add previous V1)) := by
    simpa only [gl_limitSet] using row_97
  -- chapter_146_line_103: GL tag disintegration.
  have row_103 : (gl_implication27 V1 N add previous) := by
    exact row_96.1.2
  -- chapter_146_line_102: GL tag expansion.
  have row_102 : (∀ (w1 : α), ((V1 w1) → (gl_preorder N add w1 previous))) := by
    simpa only [gl_implication27] using row_103
  -- chapter_146_line_101: GL tag implication.
  have row_101 : (∀ (V2 : GLBinaryRelation α) (v21 : α), ((V1 v21) → (∀ (v17 : α), ((succ v21 v17) → ((V1 v17) → (∀ (v22 : α), ((V2 v21 v22) → (∀ (v23 : α), ((identity v17 v23) → (∀ (v18 : α), ((add v22 v23 v18) → (gl_preorder N add v17 previous)))))))))))) := by
    intro V2
    intro v21
    intro scope_premise_1
    intro v17
    intro scope_premise_2
    intro scope_premise_3
    intro v22
    intro scope_premise_4
    intro v23
    intro scope_premise_5
    intro v18
    intro scope_premise_6
    have scoped_fact_2 := row_100 V1 V2 v21 scope_premise_1 v17 scope_premise_2 scope_premise_3 v22 scope_premise_4 v23 scope_premise_5 v18 scope_premise_6
    apply row_102
    exact scoped_fact_2
  -- chapter_146_line_95: GL tag disintegration.
  have row_95 : (gl_implication41 V1 V4) := by
    exact row_96.1.1
  -- chapter_146_line_94: GL tag expansion.
  have row_94 : (∀ (w1 : α), ((V1 w1) → (V4 w1))) := by
    simpa only [gl_implication41] using row_95
  -- chapter_146_line_99: GL tag implication.
  have row_99 : (∀ (V2 : GLBinaryRelation α) (v21 : α), ((V1 v21) → (∀ (v17 : α), ((succ v21 v17) → ((V1 v17) → (∀ (v22 : α), ((V2 v21 v22) → (∀ (v23 : α), ((identity v17 v23) → (∀ (v18 : α), ((add v22 v23 v18) → (V4 v17)))))))))))) := by
    intro V2
    intro v21
    intro scope_premise_1
    intro v17
    intro scope_premise_2
    intro scope_premise_3
    intro v22
    intro scope_premise_4
    intro v23
    intro scope_premise_5
    intro v18
    intro scope_premise_6
    have scoped_fact_2 := row_100 V1 V2 v21 scope_premise_1 v17 scope_premise_2 scope_premise_3 v22 scope_premise_4 v23 scope_premise_5 v18 scope_premise_6
    apply row_94
    exact scoped_fact_2
  -- chapter_146_line_93: GL tag implication.
  have row_93 : (∀ (V2 : GLBinaryRelation α) (v21 : α), ((V1 v21) → (∀ (v17 : α), ((succ v21 v17) → ((V1 v17) → (∀ (v22 : α), ((V2 v21 v22) → (∀ (v23 : α), ((identity v17 v23) → (∀ (v18 : α), ((add v22 v23 v18) → (V4 v21)))))))))))) := by
    intro V2
    intro v21
    intro scope_premise_1
    intro v17
    intro scope_premise_2
    intro scope_premise_3
    intro v22
    intro scope_premise_4
    intro v23
    intro scope_premise_5
    intro v18
    intro scope_premise_6
    have scoped_fact_2 := row_98 V1 V2 v21 scope_premise_1 v17 scope_premise_2 scope_premise_3 v22 scope_premise_4 v23 scope_premise_5 v18 scope_premise_6
    apply row_94
    exact scoped_fact_2
  -- chapter_146_line_68: GL tag disintegration.
  have row_68 : (gl_interval N add zero previous V1) := by
    exact witness_row_69.2
  -- chapter_146_line_72: GL tag implication.
  have row_72 : (V1 previous) := by
    apply row_73
    exact row_68
  -- chapter_146_line_67: GL tag expansion.
  have row_67 : (((((gl_implication26 V1 N add zero) ∧ (gl_implication27 V1 N add previous)) ∧ (gl_implication28 N add zero previous V1)) ∧ (N zero)) ∧ (N previous)) := by
    simpa only [gl_interval] using row_68
  -- chapter_146_line_66: GL tag disintegration.
  have row_66 : (gl_implication26 V1 N add zero) := by
    exact row_67.1.1.1.1
  -- chapter_146_line_65: GL tag expansion.
  have row_65 : (∀ (w1 : α), ((V1 w1) → (gl_preorder N add zero w1))) := by
    simpa only [gl_implication26] using row_66
  -- chapter_146_line_64: GL tag implication.
  have row_64 : (gl_preorder N add zero previous) := by
    apply row_65
    exact row_72
  -- chapter_146_line_32: GL tag implication.
  have row_32 : (gl_sequence N add zero v1 V3) := by
    apply row_33
    exact row_34
    exact row_39
  -- chapter_146_line_29: GL tag implication.
  have row_29 : (gl_existence15 N add previous V3 zero) := by
    apply row_30
    exact row_31
    exact row_32
  -- chapter_146_line_28: GL tag expansion.
  have row_28 : (¬ (∀ (V2 : GLBinaryRelation α), ((gl_limitSequence N add previous V3 V2) → (¬ (gl_sequence N add zero previous V2))))) := by
    simpa only [gl_existence15] using row_29
  have exists_row_28 : ∃ (V2 : GLBinaryRelation α), ((gl_limitSequence N add previous V3 V2) ∧ (gl_sequence N add zero previous V2)) := existsAndOfNotForallImpNot row_28
  obtain ⟨V2, witness_row_28⟩ := exists_row_28
  -- chapter_146_line_123: GL tag disintegration.
  have row_123 : (gl_sequence N add zero previous V2) := by
    exact witness_row_28.2
  -- chapter_146_line_27: GL tag disintegration.
  have row_27 : (gl_limitSequence N add previous V3 V2) := by
    exact witness_row_28.1
  -- chapter_146_line_26: GL tag expansion.
  have row_26 : (((gl_implication38 V2 N add previous) ∧ (gl_implication39 V2 V3)) ∧ (gl_implication40 N add previous V3 V2)) := by
    simpa only [gl_limitSequence] using row_27
  -- chapter_146_line_89: GL tag disintegration.
  have row_89 : (gl_implication39 V2 V3) := by
    exact row_26.1.2
  -- chapter_146_line_88: GL tag expansion.
  have row_88 : (∀ (w1 : α) (w2 : α), ((V2 w1 w2) → (V3 w1 w2))) := by
    simpa only [gl_implication39] using row_89
  -- chapter_146_line_87: GL tag implication.
  have row_87 : (∀ (v21 : α), ((V1 v21) → (∀ (v17 : α), ((succ v21 v17) → ((V1 v17) → (∀ (v22 : α), ((V2 v21 v22) → (∀ (v23 : α), ((identity v17 v23) → (∀ (v18 : α), ((add v22 v23 v18) → (V3 v21 v22)))))))))))) := by
    intro v21
    intro scope_premise_1
    intro v17
    intro scope_premise_2
    intro scope_premise_3
    intro v22
    intro scope_premise_4
    intro v23
    intro scope_premise_5
    intro v18
    intro scope_premise_6
    have scoped_fact_2 := row_90 V1 V2 v21 scope_premise_1 v17 scope_premise_2 scope_premise_3 v22 scope_premise_4 v23 scope_premise_5 v18 scope_premise_6
    apply row_88
    exact scoped_fact_2
  -- chapter_146_line_82: GL tag implication.
  have row_82 : (∀ (v21 : α), ((V1 v21) → (∀ (v17 : α), ((succ v21 v17) → ((V1 v17) → (∀ (v22 : α), ((V2 v21 v22) → (∀ (v23 : α), ((identity v17 v23) → (∀ (v18 : α), ((add v22 v23 v18) → (V3 v17 v18)))))))))))) := by
    intro v21
    intro scope_premise_1
    intro v17
    intro scope_premise_2
    intro scope_premise_3
    intro v22
    intro scope_premise_4
    intro v23
    intro scope_premise_5
    intro v18
    intro scope_premise_6
    have scoped_fact_2 := row_93 V2 v21 scope_premise_1 v17 scope_premise_2 scope_premise_3 v22 scope_premise_4 v23 scope_premise_5 v18 scope_premise_6
    have scoped_fact_3 := row_85 V1 V2 v21 scope_premise_1 v17 scope_premise_2 scope_premise_3 v22 scope_premise_4 v23 scope_premise_5 v18 scope_premise_6
    have scoped_fact_4 := row_99 V2 v21 scope_premise_1 v17 scope_premise_2 scope_premise_3 v22 scope_premise_4 v23 scope_premise_5 v18 scope_premise_6
    have scoped_fact_5 := row_87 v21 scope_premise_1 v17 scope_premise_2 scope_premise_3 v22 scope_premise_4 v23 scope_premise_5 v18 scope_premise_6
    have scoped_fact_6 := row_91 V1 V2 v21 scope_premise_1 v17 scope_premise_2 scope_premise_3 v22 scope_premise_4 v23 scope_premise_5 v18 scope_premise_6
    have scoped_fact_7 := row_92 V1 V2 v21 scope_premise_1 v17 scope_premise_2 scope_premise_3 v22 scope_premise_4 v23 scope_premise_5 v18 scope_premise_6
    apply row_83
    exact scoped_fact_2
    exact scoped_fact_3
    exact scoped_fact_4
    exact scoped_fact_5
    exact scoped_fact_6
    exact scoped_fact_7
  -- chapter_146_line_25: GL tag disintegration.
  have row_25 : (gl_implication40 N add previous V3 V2) := by
    exact row_26.2
  -- chapter_146_line_24: GL tag expansion.
  have row_24 : (∀ (w1 : α), ((gl_preorder N add w1 previous) → (∀ (w2 : α), ((V3 w1 w2) → (V2 w1 w2))))) := by
    simpa only [gl_implication40] using row_25
  -- chapter_146_line_81: GL tag implication.
  have row_81 : (∀ (v21 : α), ((V1 v21) → (∀ (v17 : α), ((succ v21 v17) → ((V1 v17) → (∀ (v22 : α), ((V2 v21 v22) → (∀ (v23 : α), ((identity v17 v23) → (∀ (v18 : α), ((add v22 v23 v18) → (V2 v17 v18)))))))))))) := by
    intro v21
    intro scope_premise_1
    intro v17
    intro scope_premise_2
    intro scope_premise_3
    intro v22
    intro scope_premise_4
    intro v23
    intro scope_premise_5
    intro v18
    intro scope_premise_6
    have scoped_fact_2 := row_101 V2 v21 scope_premise_1 v17 scope_premise_2 scope_premise_3 v22 scope_premise_4 v23 scope_premise_5 v18 scope_premise_6
    have scoped_fact_3 := row_82 v21 scope_premise_1 v17 scope_premise_2 scope_premise_3 v22 scope_premise_4 v23 scope_premise_5 v18 scope_premise_6
    apply row_24
    exact scoped_fact_2
    exact scoped_fact_3
  -- chapter_146_line_80: GL tag validity name.
  have row_80 : (gl_implication33 V1 succ V2 identity add) := by
    simpa only [gl_implication33] using row_81
  -- chapter_146_line_6: GL tag expansion.
  have row_6 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorGauss] using row_7
  -- chapter_146_line_62: GL tag disintegration.
  have row_62 : (succ one two) := by
    exact row_6.1.2
  -- chapter_146_line_49: GL tag disintegration.
  have row_49 : (gl_identity N identity) := by
    exact row_6.2
  -- chapter_146_line_48: GL tag expansion.
  have row_48 : (((gl_implication0 identity N) ∧ (gl_implication22 identity)) ∧ (gl_implication23 N identity)) := by
    simpa only [gl_identity] using row_49
  -- chapter_146_line_77: GL tag disintegration.
  have row_77 : (gl_implication22 identity) := by
    exact row_48.1.2
  -- chapter_146_line_76: GL tag expansion.
  have row_76 : (∀ (w1 : α) (w2 : α), ((identity w1 w2) → (w1 = w2))) := by
    simpa only [gl_implication22] using row_77
  -- chapter_146_line_75: GL tag implication.
  have row_75 : (∀ (v12 : α), ((identity zero v12) → (zero = v12))) := by
    intro v12
    intro scope_premise_1
    have scoped_fact_2 := row_78 V2 v12 scope_premise_1
    apply row_76
    exact scoped_fact_2
  -- chapter_146_line_47: GL tag disintegration.
  have row_47 : (gl_implication23 N identity) := by
    exact row_48.2
  -- chapter_146_line_46: GL tag expansion.
  have row_46 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((w1 = w2) → (identity w1 w2))))) := by
    simpa only [gl_implication23] using row_47
  -- chapter_146_line_8: GL tag disintegration.
  have row_8 : (succ zero one) := by
    exact row_6.1.1.2
  -- chapter_146_line_5: GL tag disintegration.
  have row_5 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_6.1.1.1
  -- chapter_146_line_53: GL tag expansion.
  have row_53 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_5
  -- chapter_146_line_216: GL tag disintegration.
  have row_216 : (gl_implication18 N succ add) := by
    exact row_53.1.1.1.1.2
  -- chapter_146_line_215: GL tag expansion.
  have row_215 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((succ w4 w5) → (add w3 w2 w5))))))))) := by
    simpa only [gl_implication18] using row_216
  -- chapter_146_line_191: GL tag disintegration.
  have row_191 : (gl_implication15 N zero add) := by
    exact row_53.1.1.1.1.1.1.1.2
  -- chapter_146_line_190: GL tag expansion.
  have row_190 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((add w1 zero w2) → (w1 = w2))))) := by
    simpa only [gl_implication15] using row_191
  -- chapter_146_line_186: GL tag disintegration.
  have row_186 : (gl_implication17 N succ add) := by
    exact row_53.1.1.1.1.1.2
  -- chapter_146_line_185: GL tag expansion.
  have row_185 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication17] using row_186
  -- chapter_146_line_148: GL tag disintegration.
  have row_148 : (gl_implication16 N zero add) := by
    exact row_53.1.1.1.1.1.1.2
  -- chapter_146_line_147: GL tag expansion.
  have row_147 : (∀ (w1 : α) (w2 : α), ((w1 = w2) → ((N w1) → ((N w2) → (add w1 zero w2))))) := by
    simpa only [gl_implication16] using row_148
  -- chapter_146_line_130: GL tag disintegration.
  have row_130 : (gl_fXYZ mul N N N) := by
    exact row_53.1.1.1.2
  -- chapter_146_line_129: GL tag expansion.
  have row_129 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_130
  -- chapter_146_line_128: GL tag disintegration.
  have row_128 : (gl_implication13 N N N mul) := by
    exact row_129.1.2
  -- chapter_146_line_127: GL tag expansion.
  have row_127 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication13] using row_128
  -- chapter_146_line_120: GL tag disintegration.
  have row_120 : (gl_fXYZ add N N N) := by
    exact row_53.1.1.1.1.1.1.1.1.2
  -- chapter_146_line_119: GL tag expansion.
  have row_119 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_120
  -- chapter_146_line_143: GL tag disintegration.
  have row_143 : (gl_implication14 N N add) := by
    exact row_119.2
  -- chapter_146_line_142: GL tag expansion.
  have row_142 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (∀ (w3 : α), ((add w1 w2 w3) → (∀ (w4 : α), ((add w1 w2 w4) → (w3 = w4))))))))) := by
    simpa only [gl_implication14] using row_143
  -- chapter_146_line_118: GL tag disintegration.
  have row_118 : (gl_implication13 N N N add) := by
    exact row_119.1.2
  -- chapter_146_line_117: GL tag expansion.
  have row_117 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication13] using row_118
  -- chapter_146_line_63: GL tag disintegration.
  have row_63 : (N zero) := by
    exact row_53.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_146_line_167: GL tag implication.
  have row_167 : (gl_existence1 N v1 zero add) := by
    apply row_117
    exact row_131
    exact row_63
  -- chapter_146_line_166: GL tag expansion.
  have row_166 : (¬ (∀ (v7 : α), ((N v7) → (¬ (add v1 zero v7))))) := by
    simpa only [gl_existence1] using row_167
  have exists_row_166 : ∃ (v7 : α), ((N v7) ∧ (add v1 zero v7)) := existsAndOfNotForallImpNot row_166
  obtain ⟨v7, witness_row_166⟩ := exists_row_166
  -- chapter_146_line_165: GL tag disintegration.
  have row_165 : (add v1 zero v7) := by
    exact witness_row_166.2
  -- chapter_146_line_61: GL tag disintegration.
  have row_61 : (gl_fXY succ N N) := by
    exact row_53.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_146_line_60: GL tag expansion.
  have row_60 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_61
  -- chapter_146_line_156: GL tag disintegration.
  have row_156 : (gl_implication4 N N succ) := by
    exact row_60.1.2
  -- chapter_146_line_155: GL tag expansion.
  have row_155 : (∀ (w1 : α), ((N w1) → (gl_existence0 N w1 succ))) := by
    simpa only [gl_implication4] using row_156
  -- chapter_146_line_151: GL tag disintegration.
  have row_151 : (gl_implication5 N succ) := by
    exact row_60.2
  -- chapter_146_line_150: GL tag expansion.
  have row_150 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α), ((succ w1 w3) → (w2 = w3))))))) := by
    simpa only [gl_implication5] using row_151
  -- chapter_146_line_140: GL tag disintegration.
  have row_140 : (gl_implication1 succ N) := by
    exact row_60.1.1.2
  -- chapter_146_line_139: GL tag expansion.
  have row_139 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using row_140
  -- chapter_146_line_138: GL tag implication.
  have row_138 : (N two) := by
    apply row_139
    exact row_62
  -- chapter_146_line_137: GL tag implication.
  have row_137 : (gl_existence1 N v1 two mul) := by
    apply row_127
    exact row_131
    exact row_138
  -- chapter_146_line_136: GL tag expansion.
  have row_136 : (¬ (∀ (v8 : α), ((N v8) → (¬ (mul v1 two v8))))) := by
    simpa only [gl_existence1] using row_137
  have exists_row_136 : ∃ (v8 : α), ((N v8) ∧ (mul v1 two v8)) := existsAndOfNotForallImpNot row_136
  obtain ⟨v8, witness_row_136⟩ := exists_row_136
  -- chapter_146_line_135: GL tag disintegration.
  have row_135 : (mul v1 two v8) := by
    exact witness_row_136.2
  -- chapter_146_line_59: GL tag disintegration.
  have row_59 : (gl_implication0 succ N) := by
    exact row_60.1.1.1
  -- chapter_146_line_58: GL tag expansion.
  have row_58 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_59
  -- chapter_146_line_121: GL tag implication.
  have row_121 : (N previous) := by
    apply row_58
    exact row_31
  -- chapter_146_line_194: GL tag implication.
  have row_194 : (gl_existence1 N previous zero add) := by
    apply row_117
    exact row_121
    exact row_63
  -- chapter_146_line_193: GL tag expansion.
  have row_193 : (¬ (∀ (v28 : α), ((N v28) → (¬ (add previous zero v28))))) := by
    simpa only [gl_existence1] using row_194
  have exists_row_193 : ∃ (v28 : α), ((N v28) ∧ (add previous zero v28)) := existsAndOfNotForallImpNot row_193
  obtain ⟨v28, witness_row_193⟩ := exists_row_193
  -- chapter_146_line_202: GL tag disintegration.
  have row_202 : (N v28) := by
    exact witness_row_193.1
  -- chapter_146_line_192: GL tag disintegration.
  have row_192 : (add previous zero v28) := by
    exact witness_row_193.2
  -- chapter_146_line_189: GL tag implication.
  have row_189 : (previous = v28) := by
    apply row_190
    exact row_121
    exact row_192
  -- chapter_146_line_154: GL tag implication.
  have row_154 : (gl_existence0 N previous succ) := by
    apply row_155
    exact row_121
  -- chapter_146_line_153: GL tag expansion.
  have row_153 : (¬ (∀ (v26 : α), ((N v26) → (¬ (succ previous v26))))) := by
    simpa only [gl_existence0] using row_154
  have exists_row_153 : ∃ (v26 : α), ((N v26) ∧ (succ previous v26)) := existsAndOfNotForallImpNot row_153
  obtain ⟨v26, witness_row_153⟩ := exists_row_153
  -- chapter_146_line_157: GL tag disintegration.
  have row_157 : (N v26) := by
    exact witness_row_153.1
  -- chapter_146_line_152: GL tag disintegration.
  have row_152 : (succ previous v26) := by
    exact witness_row_153.2
  -- chapter_146_line_158: GL tag implication.
  have row_158 : (v26 = v1) := by
    apply row_150
    exact row_121
    exact row_152
    exact row_31
  -- chapter_146_line_149: GL tag implication.
  have row_149 : (v1 = v26) := by
    apply row_150
    exact row_121
    exact row_31
    exact row_152
  -- chapter_146_line_146: GL tag implication.
  have row_146 : (add v1 zero v26) := by
    apply row_147
    exact row_149
    exact row_131
    exact row_157
  -- chapter_146_line_145: GL tag equality1.
  have row_145 : (add v1 zero v1) := by
    have equality_source := row_146
    have equality_step_1 := row_158
    cases equality_step_1
    exact equality_source
  -- chapter_146_line_126: GL tag implication.
  have row_126 : (gl_existence1 N previous v1 mul) := by
    apply row_127
    exact row_121
    exact row_131
  -- chapter_146_line_125: GL tag expansion.
  have row_125 : (¬ (∀ (v6 : α), ((N v6) → (¬ (mul previous v1 v6))))) := by
    simpa only [gl_existence1] using row_126
  have exists_row_125 : ∃ (v6 : α), ((N v6) ∧ (mul previous v1 v6)) := existsAndOfNotForallImpNot row_125
  obtain ⟨v6, witness_row_125⟩ := exists_row_125
  -- chapter_146_line_124: GL tag disintegration.
  have row_124 : (mul previous v1 v6) := by
    exact witness_row_125.2
  -- chapter_146_line_116: GL tag implication.
  have row_116 : (gl_existence1 N zero previous add) := by
    apply row_117
    exact row_63
    exact row_121
  -- chapter_146_line_115: GL tag expansion.
  have row_115 : (¬ (∀ (v24 : α), ((N v24) → (¬ (add zero previous v24))))) := by
    simpa only [gl_existence1] using row_116
  have exists_row_115 : ∃ (v24 : α), ((N v24) ∧ (add zero previous v24)) := existsAndOfNotForallImpNot row_115
  obtain ⟨v24, witness_row_115⟩ := exists_row_115
  -- chapter_146_line_114: GL tag disintegration.
  have row_114 : (add zero previous v24) := by
    exact witness_row_115.2
  -- chapter_146_line_201: GL tag equality1.
  have row_201 : (add zero v28 v24) := by
    have equality_source := row_114
    have equality_step_1 := row_189
    cases equality_step_1
    exact equality_source
  -- chapter_146_line_57: GL tag implication.
  have row_57 : (N one) := by
    apply row_58
    exact row_62
  -- chapter_146_line_163: GL tag implication.
  have row_163 : (gl_existence1 N one previous add) := by
    apply row_117
    exact row_57
    exact row_121
  -- chapter_146_line_162: GL tag expansion.
  have row_162 : (¬ (∀ (v25 : α), ((N v25) → (¬ (add one previous v25))))) := by
    simpa only [gl_existence1] using row_163
  have exists_row_162 : ∃ (v25 : α), ((N v25) ∧ (add one previous v25)) := existsAndOfNotForallImpNot row_162
  obtain ⟨v25, witness_row_162⟩ := exists_row_162
  -- chapter_146_line_168: GL tag disintegration.
  have row_168 : (N v25) := by
    exact witness_row_162.1
  -- chapter_146_line_161: GL tag disintegration.
  have row_161 : (add one previous v25) := by
    exact witness_row_162.2
  -- chapter_146_line_52: GL tag disintegration.
  have row_52 : (gl_implication7 N succ) := by
    exact row_53.1.1.1.1.1.1.1.1.1.2
  -- chapter_146_line_51: GL tag expansion.
  have row_51 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w2 w1) → (∀ (w3 : α), ((succ w3 w1) → (w2 = w3))))))) := by
    simpa only [gl_implication7] using row_52
  -- chapter_146_line_4: GL tag expansion for integration.
  have row_4 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_146_line_3: GL tag reformulation for integration and.
  have row_3 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_146_line_2: GL tag implication.
  have row_2 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_3
    exact row_5
    exact row_8
  have rule_row_221 := peano_source_022 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_146_line_221: GL tag implication.
  have row_221 : (mul v1 previous v6) := by
    apply rule_row_221
    exact row_124
  -- chapter_146_line_220: GL tag equality1.
  have row_220 : (mul v1 v28 v6) := by
    have equality_source := row_221
    have equality_step_1 := row_189
    cases equality_step_1
    exact equality_source
  have rule_row_160 := peano_source_033 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_146_line_160: GL tag implication.
  have row_160 : (add one previous v1) := by
    apply rule_row_160
    exact row_31
  have rule_row_218 := peano_source_037 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_146_line_218: GL tag implication.
  have row_218 : (add previous one v1) := by
    apply rule_row_218
    exact row_160
  -- chapter_146_line_214: GL tag implication.
  have row_214 : (add previous two v2) := by
    apply row_215
    exact row_57
    exact row_62
    exact row_218
    exact row_217
  have rule_row_213 := peano_source_016 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_146_line_213: GL tag implication.
  have row_213 : (add two previous v2) := by
    apply rule_row_213
    exact row_214
  -- chapter_146_line_212: GL tag equality1.
  have row_212 : (add two v28 v2) := by
    have equality_source := row_213
    have equality_step_1 := row_189
    cases equality_step_1
    exact equality_source
  have rule_row_211 := peano_source_018 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_146_line_211: GL tag implication.
  have row_211 : (add v8 v6 v3) := by
    apply rule_row_211
    exact row_135
    exact row_220
    exact row_219
    exact row_212
  -- chapter_146_line_200: GL tag implication.
  have row_200 : (v25 = v1) := by
    apply row_142
    exact row_57
    exact row_121
    exact row_161
    exact row_160
  -- chapter_146_line_159: GL tag implication.
  have row_159 : (v1 = v25) := by
    apply row_142
    exact row_57
    exact row_121
    exact row_160
    exact row_161
  -- chapter_146_line_199: GL tag equality1.
  have row_199 : (add v1 zero v25) := by
    have equality_source := row_145
    have equality_step_1 := row_159
    cases equality_step_1
    exact equality_source
  -- chapter_146_line_197: GL tag implication.
  have row_197 : (add zero v1 v25) := by
    apply row_172
    exact row_199
  -- chapter_146_line_196: GL tag equality1.
  have row_196 : (add zero v1 v1) := by
    have equality_source := row_197
    have equality_step_1 := row_200
    cases equality_step_1
    exact equality_source
  -- chapter_146_line_164: GL tag equality1.
  have row_164 : (add v25 zero v7) := by
    have equality_source := row_165
    have equality_step_1 := row_159
    cases equality_step_1
    exact equality_source
  -- chapter_146_line_144: GL tag equality1.
  have row_144 : (add v25 zero v1) := by
    have equality_source := row_145
    have equality_step_1 := row_159
    cases equality_step_1
    exact equality_source
  -- chapter_146_line_141: GL tag implication.
  have row_141 : (v1 = v7) := by
    apply row_142
    exact row_168
    exact row_63
    exact row_144
    exact row_164
  -- chapter_146_line_195: GL tag equality1.
  have row_195 : (add zero v7 v1) := by
    have equality_source := row_196
    have equality_step_1 := row_141
    cases equality_step_1
    exact equality_source
  -- chapter_146_line_188: GL tag equality1.
  have row_188 : (succ previous v7) := by
    have equality_source := row_31
    have equality_step_1 := row_141
    cases equality_step_1
    exact equality_source
  -- chapter_146_line_187: GL tag equality1.
  have row_187 : (succ v28 v7) := by
    have equality_source := row_188
    have equality_step_1 := row_189
    cases equality_step_1
    exact equality_source
  -- chapter_146_line_184: GL tag implication.
  have row_184 : (succ v24 v1) := by
    apply row_185
    exact row_202
    exact row_187
    exact row_201
    exact row_195
  -- chapter_146_line_183: GL tag implication.
  have row_183 : (identity v1 v7) := by
    apply row_46
    exact row_131
    exact row_141
  -- chapter_146_line_134: GL tag equality1.
  have row_134 : (mul v7 two v8) := by
    have equality_source := row_135
    have equality_step_1 := row_141
    cases equality_step_1
    exact equality_source
  have rule_row_133 := peano_source_022 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_146_line_133: GL tag implication.
  have row_133 : (mul two v7 v8) := by
    apply rule_row_133
    exact row_134
  have rule_row_113 := peano_source_055 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_146_line_113: GL tag implication.
  have row_113 : (previous = v24) := by
    apply rule_row_113
    exact row_121
    exact row_114
  -- chapter_146_line_122: GL tag equality1.
  have row_122 : (gl_sequence N add zero v24 V2) := by
    have equality_source := row_123
    have equality_step_1 := row_113
    cases equality_step_1
    exact equality_source
  -- chapter_146_line_112: GL tag equality1.
  have row_112 : (gl_interval N add zero v24 V1) := by
    have equality_source := row_68
    have equality_step_1 := row_113
    cases equality_step_1
    exact equality_source
  -- chapter_146_line_210: GL tag implication.
  have row_210 : (V1 v24) := by
    apply row_73
    exact row_112
  -- chapter_146_line_209: GL tag implication.
  have row_209 : (V4 v24) := by
    apply row_94
    exact row_210
  -- chapter_146_line_110: GL tag implication.
  have row_110 : (gl_fXY V2 V1 N) := by
    apply row_111
    exact row_112
    exact row_122
  -- chapter_146_line_109: GL tag expansion.
  have row_109 : ((((gl_implication0 V2 V1) ∧ (gl_implication1 V2 N)) ∧ (gl_implication4 V1 N V2)) ∧ (gl_implication5 V1 V2)) := by
    simpa only [gl_fXY] using row_110
  -- chapter_146_line_108: GL tag disintegration.
  have row_108 : (gl_implication4 V1 N V2) := by
    exact row_109.1.2
  -- chapter_146_line_107: GL tag expansion.
  have row_107 : (∀ (w1 : α), ((V1 w1) → (gl_existence0 N w1 V2))) := by
    simpa only [gl_implication4] using row_108
  -- chapter_146_line_106: GL tag implication.
  have row_106 : (gl_existence0 N previous V2) := by
    apply row_107
    exact row_72
  -- chapter_146_line_105: GL tag expansion.
  have row_105 : (¬ (∀ (v5 : α), ((N v5) → (¬ (V2 previous v5))))) := by
    simpa only [gl_existence0] using row_106
  have exists_row_105 : ∃ (v5 : α), ((N v5) ∧ (V2 previous v5)) := existsAndOfNotForallImpNot row_105
  obtain ⟨v5, witness_row_105⟩ := exists_row_105
  -- chapter_146_line_176: GL tag disintegration.
  have row_176 : (N v5) := by
    exact witness_row_105.1
  -- chapter_146_line_175: GL tag implication.
  have row_175 : (gl_existence1 N v5 v1 add) := by
    apply row_117
    exact row_176
    exact row_131
  -- chapter_146_line_174: GL tag expansion.
  have row_174 : (¬ (∀ (v27 : α), ((N v27) → (¬ (add v5 v1 v27))))) := by
    simpa only [gl_existence1] using row_175
  have exists_row_174 : ∃ (v27 : α), ((N v27) ∧ (add v5 v1 v27)) := existsAndOfNotForallImpNot row_174
  obtain ⟨v27, witness_row_174⟩ := exists_row_174
  -- chapter_146_line_173: GL tag disintegration.
  have row_173 : (add v5 v1 v27) := by
    exact witness_row_174.2
  -- chapter_146_line_205: GL tag equality1.
  have row_205 : (add v5 v7 v27) := by
    have equality_source := row_173
    have equality_step_1 := row_141
    cases equality_step_1
    exact equality_source
  -- chapter_146_line_171: GL tag implication.
  have row_171 : (add v1 v5 v27) := by
    apply row_172
    exact row_173
  -- chapter_146_line_104: GL tag disintegration.
  have row_104 : (V2 previous v5) := by
    exact witness_row_105.2
  -- chapter_146_line_204: GL tag equality1.
  have row_204 : (V2 v24 v5) := by
    have equality_source := row_104
    have equality_step_1 := row_113
    cases equality_step_1
    exact equality_source
  -- chapter_146_line_203: GL tag implication.
  have row_203 : (V3 v24 v5) := by
    apply row_88
    exact row_204
  -- chapter_146_line_182: GL tag implication.
  have row_182 : (V3 v1 v27) := by
    apply row_83
    exact row_209
    exact row_184
    exact row_206
    exact row_203
    exact row_183
    exact row_205
  -- chapter_146_line_177: GL tag implication.
  have row_177 : (v27 = v4) := by
    apply row_178
    exact row_206
    exact row_182
    exact row_181
  -- chapter_146_line_170: GL tag equality1.
  have row_170 : (add v1 v5 v4) := by
    have equality_source := row_171
    have equality_step_1 := row_177
    cases equality_step_1
    exact equality_source
  -- chapter_146_line_169: GL tag equality1.
  have row_169 : (add v7 v5 v4) := by
    have equality_source := row_170
    have equality_step_1 := row_141
    cases equality_step_1
    exact equality_source
  have rule_row_56 := peano_source_063 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_146_line_56: GL tag implication.
  have row_56 : (gl_existence3 N one succ) := by
    apply rule_row_56
  -- chapter_146_line_55: GL tag expansion.
  have row_55 : (¬ (∀ (v13 : α), ((N v13) → (¬ (succ v13 one))))) := by
    simpa only [gl_existence3] using row_56
  have exists_row_55 : ∃ (v13 : α), ((N v13) ∧ (succ v13 one)) := existsAndOfNotForallImpNot row_55
  obtain ⟨v13, witness_row_55⟩ := exists_row_55
  -- chapter_146_line_54: GL tag disintegration.
  have row_54 : (succ v13 one) := by
    exact witness_row_55.2
  -- chapter_146_line_74: GL tag implication.
  have row_74 : (v13 = zero) := by
    apply row_51
    exact row_57
    exact row_54
    exact row_8
  -- chapter_146_line_50: GL tag implication.
  have row_50 : (zero = v13) := by
    apply row_51
    exact row_57
    exact row_8
    exact row_54
  -- chapter_146_line_45: GL tag implication.
  have row_45 : (identity zero v13) := by
    apply row_46
    exact row_63
    exact row_50
  -- chapter_146_line_40: GL tag implication.
  have row_40 : (V3 zero v13) := by
    apply row_41
    exact row_45
  -- chapter_146_line_23: GL tag implication.
  have row_23 : (V2 zero v13) := by
    apply row_24
    exact row_64
    exact row_40
  -- chapter_146_line_22: GL tag equality1.
  have row_22 : (V2 zero zero) := by
    have equality_source := row_23
    have equality_step_1 := row_74
    cases equality_step_1
    exact equality_source
  -- chapter_146_line_21: GL tag equality1.
  have row_21 : (∀ (v12 : α), ((identity zero v12) → (V2 zero v12))) := by
    intro v12
    intro scope_premise_1
    have scoped_fact_2 := row_75 v12 scope_premise_1
    have equality_source := row_22
    have equality_step_1 := scoped_fact_2
    cases equality_step_1
    exact equality_source
  -- chapter_146_line_20: GL tag validity name.
  have row_20 : (gl_implication32 zero identity V2) := by
    simpa only [gl_implication32] using row_21
  -- chapter_146_line_17: GL tag implication.
  have row_17 : (gl_and0 zero identity V2 previous v5 V1 succ add) := by
    apply row_18
    exact row_20
    exact row_104
    exact row_80
  -- chapter_146_line_14: GL tag implication.
  have row_14 : (gl_existence2 V1 N zero identity previous v5 succ add) := by
    apply row_15
    exact row_110
    exact row_17
  -- chapter_146_line_11: GL tag implication.
  have row_11 : (gl_fold N succ add identity zero previous v5) := by
    apply row_12
    exact row_68
    exact row_14
  -- chapter_146_line_9: GL tag implication.
  have row_9 : (mul two v5 v6) := by
    apply row_10
    exact row_124
    exact row_11
    exact row_31
  have rule_row_1 := peano_source_000 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_146_line_1: GL tag implication.
  have row_1 : (mul two v4 v3) := by
    apply rule_row_1
    exact row_211
    exact row_169
    exact row_133
    exact row_9
  exact row_1

theorem gauss_source_066
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorGauss N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α) (v3 : α), ((mul v1 v2 v3) → (∀ (v4 : α), ((gl_fold N succ add identity zero v1 v4) → ((succ v1 v2) → (mul two v4 v3)))))) := by
  intro v1
  intro v2
  intro v3
  intro premise_1
  intro v4
  intro premise_2
  intro premise_3
  have inductionMember : N v1 := by
    have typingRule := gauss_source_066_induction_typing N zero succ add mul one two identity anchor relationalInduction
    exact typingRule v1 v4 premise_2
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorGauss, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v2 : α) (v3 : α), ((mul zero v2 v3) → (∀ (v4 : α), ((gl_fold N succ add identity zero zero v4) → ((succ zero v2) → (mul two v4 v3)))))) := by
    intro v2
    intro v3
    intro base_premise_1
    intro v4
    intro base_premise_2
    intro base_premise_3
    have zeroRule := gauss_source_066_check_zero N zero succ add mul one two identity anchor relationalInduction
    exact zeroRule zero v2 v3 v4 rfl base_premise_1 base_premise_2
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v2 : α) (v3 : α), ((mul induction_n v2 v3) → (∀ (v4 : α), ((gl_fold N succ add identity zero induction_n v4) → ((succ induction_n v2) → (mul two v4 v3)))))) → ∀ induction_m, succ induction_n induction_m → (∀ (v2 : α) (v3 : α), ((mul induction_m v2 v3) → (∀ (v4 : α), ((gl_fold N succ add identity zero induction_m v4) → ((succ induction_m v2) → (mul two v4 v3)))))) := by
    intro induction_n induction_n_member induction_hypothesis
    intro induction_m induction_successor
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
        (∀ (w1 : α) (w2 : α), ((mul induction_n w1 w2) → (∀ (w3 : α), ((gl_fold N succ add identity zero induction_n w3) → ((succ induction_n w1) → (mul two w3 w2)))))) := by
      intro w1
      intro w2
      intro step_induction_assumption_2_premise_1
      intro w3
      intro step_induction_assumption_2_premise_2
      intro step_induction_assumption_2_premise_3
      apply induction_hypothesis
      all_goals assumption
    have stepRule := gauss_source_066_check_induction_condition N zero succ add mul one two identity anchor relationalInduction
    exact stepRule induction_n induction_m v2 v3 v4 step_premise_1 step_premise_3 step_premise_2 step_induction_assumption_1 step_induction_assumption_2
  have inductionProperty : (∀ (v2 : α) (v3 : α), ((mul v1 v2 v3) → (∀ (v4 : α), ((gl_fold N succ add identity zero v1 v4) → ((succ v1 v2) → (mul two v4 v3)))))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v2 : α) (v3 : α), ((mul induction_value v2 v3) → (∀ (v4 : α), ((gl_fold N succ add identity zero induction_value v4) → ((succ induction_value v2) → (mul two v4 v3)))))))
      v1
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v2 v3 premise_1 v4 premise_2 premise_3

theorem gauss_source_068
    {α : Type u}
    (N : GLSet α)
    (zero : α)
    (succ : GLBinaryRelation α)
    (add mul : GLTernaryRelation α)
    (one : α)
    (two : α)
    (identity : GLBinaryRelation α)
    (anchor : gl_AnchorGauss N zero succ add mul one two identity)
    (relationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → P n → ∀ m, succ n m → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((gl_fold N succ add identity zero v1 v2) → (∀ (v3 : α), ((succ v1 v3) → (∀ (v4 : α), ((mul two v2 v4) → (mul v1 v3 v4))))))) := by
  intro v1
  intro v2
  intro premise_1
  intro v3
  intro premise_2
  intro v4
  intro premise_3
  -- chapter_148_line_50: GL tag expansion for integration.
  have row_50 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_148_line_49: GL tag reformulation for integration and.
  have row_49 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_148_line_29: GL tag theorem.
  have row_29 := gauss_source_065 N zero succ add mul one two identity anchor relationalInduction
  -- chapter_148_line_27: GL tag task formulation.
  have row_27 : (mul two v2 v4) := by
    exact premise_3
  -- chapter_148_line_23: GL tag task formulation.
  have row_23 : (gl_fold N succ add identity zero v1 v2) := by
    exact premise_1
  -- chapter_148_line_22: GL tag expansion.
  have row_22 : (¬ (∀ (V1 : GLSet α), ((gl_interval N add zero v1 V1) → (¬ (gl_existence2 V1 N zero identity v1 v2 succ add))))) := by
    simpa only [gl_fold] using row_23
  have exists_row_22 : ∃ (V1 : GLSet α), ((gl_interval N add zero v1 V1) ∧ (gl_existence2 V1 N zero identity v1 v2 succ add)) := existsAndOfNotForallImpNot row_22
  obtain ⟨V1, witness_row_22⟩ := exists_row_22
  -- chapter_148_line_71: GL tag disintegration.
  have row_71 : (gl_existence2 V1 N zero identity v1 v2 succ add) := by
    exact witness_row_22.2
  -- chapter_148_line_70: GL tag expansion.
  have row_70 : (¬ (∀ (V2 : GLBinaryRelation α), ((gl_fXY V2 V1 N) → (¬ (gl_and0 zero identity V2 v1 v2 V1 succ add))))) := by
    simpa only [gl_existence2] using row_71
  have exists_row_70 : ∃ (V2 : GLBinaryRelation α), ((gl_fXY V2 V1 N) ∧ (gl_and0 zero identity V2 v1 v2 V1 succ add)) := existsAndOfNotForallImpNot row_70
  obtain ⟨V2, witness_row_70⟩ := exists_row_70
  -- chapter_148_line_74: GL tag disintegration.
  have row_74 : (gl_and0 zero identity V2 v1 v2 V1 succ add) := by
    exact witness_row_70.2
  -- chapter_148_line_73: GL tag expansion.
  have row_73 : (((gl_implication32 zero identity V2) ∧ (V2 v1 v2)) ∧ (gl_implication33 V1 succ V2 identity add)) := by
    simpa only [gl_and0] using row_74
  -- chapter_148_line_72: GL tag disintegration.
  have row_72 : (V2 v1 v2) := by
    exact row_73.1.2
  -- chapter_148_line_69: GL tag disintegration.
  have row_69 : (gl_fXY V2 V1 N) := by
    exact witness_row_70.1
  -- chapter_148_line_68: GL tag expansion.
  have row_68 : ((((gl_implication0 V2 V1) ∧ (gl_implication1 V2 N)) ∧ (gl_implication4 V1 N V2)) ∧ (gl_implication5 V1 V2)) := by
    simpa only [gl_fXY] using row_69
  -- chapter_148_line_67: GL tag disintegration.
  have row_67 : (gl_implication1 V2 N) := by
    exact row_68.1.1.2
  -- chapter_148_line_66: GL tag expansion.
  have row_66 : (∀ (w1 : α) (w2 : α), ((V2 w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using row_67
  -- chapter_148_line_65: GL tag implication.
  have row_65 : (N v2) := by
    apply row_66
    exact row_72
  -- chapter_148_line_21: GL tag disintegration.
  have row_21 : (gl_interval N add zero v1 V1) := by
    exact witness_row_22.1
  -- chapter_148_line_20: GL tag expansion.
  have row_20 : (((((gl_implication26 V1 N add zero) ∧ (gl_implication27 V1 N add v1)) ∧ (gl_implication28 N add zero v1 V1)) ∧ (N zero)) ∧ (N v1)) := by
    simpa only [gl_interval] using row_21
  -- chapter_148_line_19: GL tag disintegration.
  have row_19 : (N v1) := by
    exact row_20.2
  -- chapter_148_line_18: GL tag task formulation.
  have row_18 : (succ v1 v3) := by
    exact premise_2
  -- chapter_148_line_12: GL tag task formulation.
  have row_12 : (gl_AnchorGauss N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_148_line_11: GL tag expansion.
  have row_11 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorGauss] using row_12
  -- chapter_148_line_61: GL tag disintegration.
  have row_61 : (succ one two) := by
    exact row_11.1.2
  -- chapter_148_line_51: GL tag disintegration.
  have row_51 : (succ zero one) := by
    exact row_11.1.1.2
  -- chapter_148_line_10: GL tag disintegration.
  have row_10 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_11.1.1.1
  -- chapter_148_line_48: GL tag implication.
  have row_48 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_49
    exact row_10
    exact row_51
  have rule_row_53 := peano_source_033 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_148_line_53: GL tag implication.
  have row_53 : (add one v1 v3) := by
    apply rule_row_53
    exact row_18
  have rule_row_52 := peano_source_037 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_148_line_52: GL tag implication.
  have row_52 : (add v1 one v3) := by
    apply rule_row_52
    exact row_53
  have rule_row_47 := peano_source_063 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_148_line_47: GL tag implication.
  have row_47 : (gl_existence3 N one succ) := by
    apply rule_row_47
  -- chapter_148_line_46: GL tag expansion.
  have row_46 : (¬ (∀ (v12 : α), ((N v12) → (¬ (succ v12 one))))) := by
    simpa only [gl_existence3] using row_47
  have exists_row_46 : ∃ (v12 : α), ((N v12) ∧ (succ v12 one)) := existsAndOfNotForallImpNot row_46
  obtain ⟨v12, witness_row_46⟩ := exists_row_46
  -- chapter_148_line_62: GL tag disintegration.
  have row_62 : (N v12) := by
    exact witness_row_46.1
  -- chapter_148_line_45: GL tag disintegration.
  have row_45 : (succ v12 one) := by
    exact witness_row_46.2
  -- chapter_148_line_9: GL tag expansion.
  have row_9 : (((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication6 N zero succ)) ∧ (gl_implication7 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication15 N zero add)) ∧ (gl_implication16 N zero add)) ∧ (gl_implication17 N succ add)) ∧ (gl_implication18 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication19 N zero mul)) ∧ (gl_implication20 N succ mul add)) ∧ (gl_implication21 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using row_10
  -- chapter_148_line_57: GL tag disintegration.
  have row_57 : (gl_implication7 N succ) := by
    exact row_9.1.1.1.1.1.1.1.1.1.2
  -- chapter_148_line_56: GL tag expansion.
  have row_56 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w2 w1) → (∀ (w3 : α), ((succ w3 w1) → (w2 = w3))))))) := by
    simpa only [gl_implication7] using row_57
  -- chapter_148_line_44: GL tag disintegration.
  have row_44 : (gl_implication17 N succ add) := by
    exact row_9.1.1.1.1.1.2
  -- chapter_148_line_43: GL tag expansion.
  have row_43 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication17] using row_44
  -- chapter_148_line_41: GL tag disintegration.
  have row_41 : (N zero) := by
    exact row_9.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_148_line_40: GL tag disintegration.
  have row_40 : (gl_fXYZ add N N N) := by
    exact row_9.1.1.1.1.1.1.1.1.2
  -- chapter_148_line_39: GL tag expansion.
  have row_39 : (((((gl_implication8 add N) ∧ (gl_implication9 add N)) ∧ (gl_implication10 add N)) ∧ (gl_implication13 N N N add)) ∧ (gl_implication14 N N add)) := by
    simpa only [gl_fXYZ] using row_40
  -- chapter_148_line_38: GL tag disintegration.
  have row_38 : (gl_implication13 N N N add) := by
    exact row_39.1.2
  -- chapter_148_line_37: GL tag expansion.
  have row_37 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication13] using row_38
  -- chapter_148_line_36: GL tag implication.
  have row_36 : (gl_existence1 N v1 zero add) := by
    apply row_37
    exact row_19
    exact row_41
  -- chapter_148_line_35: GL tag expansion.
  have row_35 : (¬ (∀ (v10 : α), ((N v10) → (¬ (add v1 zero v10))))) := by
    simpa only [gl_existence1] using row_36
  have exists_row_35 : ∃ (v10 : α), ((N v10) ∧ (add v1 zero v10)) := existsAndOfNotForallImpNot row_35
  obtain ⟨v10, witness_row_35⟩ := exists_row_35
  -- chapter_148_line_34: GL tag disintegration.
  have row_34 : (add v1 zero v10) := by
    exact witness_row_35.2
  -- chapter_148_line_33: GL tag disintegration.
  have row_33 : (gl_implication15 N zero add) := by
    exact row_9.1.1.1.1.1.1.1.2
  -- chapter_148_line_32: GL tag expansion.
  have row_32 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((add w1 zero w2) → (w1 = w2))))) := by
    simpa only [gl_implication15] using row_33
  -- chapter_148_line_31: GL tag implication.
  have row_31 : (v1 = v10) := by
    apply row_32
    exact row_19
    exact row_34
  -- chapter_148_line_30: GL tag equality1.
  have row_30 : (gl_fold N succ add identity zero v10 v2) := by
    have equality_source := row_23
    have equality_step_1 := row_31
    cases equality_step_1
    exact equality_source
  -- chapter_148_line_17: GL tag disintegration.
  have row_17 : (gl_fXY succ N N) := by
    exact row_9.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_148_line_16: GL tag expansion.
  have row_16 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication4 N N succ)) ∧ (gl_implication5 N succ)) := by
    simpa only [gl_fXY] using row_17
  -- chapter_148_line_60: GL tag disintegration.
  have row_60 : (gl_implication0 succ N) := by
    exact row_16.1.1.1
  -- chapter_148_line_59: GL tag expansion.
  have row_59 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using row_60
  -- chapter_148_line_58: GL tag implication.
  have row_58 : (N one) := by
    apply row_59
    exact row_61
  -- chapter_148_line_55: GL tag implication.
  have row_55 : (zero = v12) := by
    apply row_56
    exact row_58
    exact row_51
    exact row_45
  -- chapter_148_line_54: GL tag equality1.
  have row_54 : (add v1 v12 v10) := by
    have equality_source := row_34
    have equality_step_1 := row_55
    cases equality_step_1
    exact equality_source
  -- chapter_148_line_42: GL tag implication.
  have row_42 : (succ v10 v3) := by
    apply row_43
    exact row_62
    exact row_45
    exact row_54
    exact row_52
  -- chapter_148_line_15: GL tag disintegration.
  have row_15 : (gl_implication1 succ N) := by
    exact row_16.1.1.2
  -- chapter_148_line_14: GL tag expansion.
  have row_14 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using row_15
  -- chapter_148_line_75: GL tag implication.
  have row_75 : (N two) := by
    apply row_14
    exact row_61
  -- chapter_148_line_13: GL tag implication.
  have row_13 : (N v3) := by
    apply row_14
    exact row_18
  -- chapter_148_line_8: GL tag disintegration.
  have row_8 : (gl_fXYZ mul N N N) := by
    exact row_9.1.1.1.2
  -- chapter_148_line_7: GL tag expansion.
  have row_7 : (((((gl_implication8 mul N) ∧ (gl_implication9 mul N)) ∧ (gl_implication10 mul N)) ∧ (gl_implication13 N N N mul)) ∧ (gl_implication14 N N mul)) := by
    simpa only [gl_fXYZ] using row_8
  -- chapter_148_line_26: GL tag disintegration.
  have row_26 : (gl_implication14 N N mul) := by
    exact row_7.2
  -- chapter_148_line_25: GL tag expansion.
  have row_25 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (∀ (w3 : α), ((mul w1 w2 w3) → (∀ (w4 : α), ((mul w1 w2 w4) → (w3 = w4))))))))) := by
    simpa only [gl_implication14] using row_26
  -- chapter_148_line_6: GL tag disintegration.
  have row_6 : (gl_implication13 N N N mul) := by
    exact row_7.1.2
  -- chapter_148_line_5: GL tag expansion.
  have row_5 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication13] using row_6
  -- chapter_148_line_4: GL tag implication.
  have row_4 : (gl_existence1 N v1 v3 mul) := by
    apply row_5
    exact row_19
    exact row_13
  -- chapter_148_line_3: GL tag expansion.
  have row_3 : (¬ (∀ (v5 : α), ((N v5) → (¬ (mul v1 v3 v5))))) := by
    simpa only [gl_existence1] using row_4
  have exists_row_3 : ∃ (v5 : α), ((N v5) ∧ (mul v1 v3 v5)) := existsAndOfNotForallImpNot row_3
  obtain ⟨v5, witness_row_3⟩ := exists_row_3
  -- chapter_148_line_2: GL tag disintegration.
  have row_2 : (mul v1 v3 v5) := by
    exact witness_row_3.2
  have rule_row_64 := peano_source_022 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction
  -- chapter_148_line_64: GL tag implication.
  have row_64 : (mul v3 v1 v5) := by
    apply rule_row_64
    exact row_2
  -- chapter_148_line_63: GL tag equality1.
  have row_63 : (mul v3 v10 v5) := by
    have equality_source := row_64
    have equality_step_1 := row_31
    cases equality_step_1
    exact equality_source
  -- chapter_148_line_28: GL tag implication.
  have row_28 : (mul two v2 v5) := by
    apply row_29
    exact row_63
    exact row_30
    exact row_42
  -- chapter_148_line_24: GL tag implication.
  have row_24 : (v5 = v4) := by
    apply row_25
    exact row_75
    exact row_65
    exact row_28
    exact row_27
  -- chapter_148_line_1: GL tag equality1.
  have row_1 : (mul v1 v3 v4) := by
    have equality_source := row_2
    have equality_step_1 := row_24
    cases equality_step_1
    exact equality_source
  exact row_1

end GLExport
