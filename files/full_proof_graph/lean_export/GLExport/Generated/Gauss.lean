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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (gl_preorder N add v1 v2))) := by
  intro v1
  intro v2
  intro premise_1
  -- chapter_137_line_26: GL tag task formulation.
  have row_26 : (gl_AnchorGauss N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_137_line_25: GL tag task formulation.
  have row_25 : (succ v1 v2) := by
    exact premise_1
  -- chapter_137_line_24: GL tag expansion.
  have row_24 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorGauss] using (row_26)
  -- chapter_137_line_23: GL tag disintegration.
  have row_23 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_24.1.1.1
  -- chapter_137_line_22: GL tag expansion.
  have row_22 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_23)
  -- chapter_137_line_21: GL tag disintegration.
  have row_21 : (gl_fXY succ N N) := by
    exact row_22.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_137_line_20: GL tag expansion.
  have row_20 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_21)
  -- chapter_137_line_19: GL tag disintegration.
  have row_19 : (gl_implication1 succ N) := by
    exact row_20.1.1.2
  -- chapter_137_line_18: GL tag expansion.
  have row_18 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using (row_19)
  -- chapter_137_line_17: GL tag disintegration.
  have row_17 : (gl_implication0 succ N) := by
    exact row_20.1.1.1
  -- chapter_137_line_16: GL tag expansion.
  have row_16 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_17)
  -- chapter_137_line_15: GL tag disintegration.
  have row_15 : (succ zero one) := by
    exact row_24.1.1.2
  -- chapter_137_line_14: GL tag implication grouping.
  have row_14 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (gl_and20 w1 N w2 N))) := by
    have auxiliary_source_1 := row_16
    have auxiliary_source_2 := row_18
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_premise
    exact ⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_premise)⟩
  -- chapter_137_line_13: GL tag implication.
  have row_13 : (gl_and20 zero N one N) := by
    apply row_14
    exact row_15
  -- chapter_137_line_12: GL tag expansion.
  have row_12 : ((N zero) ∧ (N one)) := by
    simpa only [gl_and20] using (row_13)
  -- chapter_137_line_11: GL tag disintegration.
  have row_11 : (N one) := by
    exact row_12.2
  -- chapter_137_line_10: GL tag expansion for integration.
  have row_10 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_137_line_9_rule: GL tag reformulation for integration and.
  have row_9_rule : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_137_line_9: GL tag reformulation for integration and.
  have row_9 : (gl_implication76 N zero succ add mul one) := by
    simpa only [gl_implication76] using row_9_rule
  -- chapter_137_line_8: GL tag expansion.
  have row_8 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    simpa only [gl_implication76] using (row_9)
  -- chapter_137_line_7: GL tag implication.
  have row_7 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_8
    exact row_23
    exact row_15
  have rule_row_6 := peano_source_010 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_137_line_6: GL tag implication.
  have row_6 : (add one v1 v2) := by
    apply rule_row_6
    exact row_25
  have rule_row_5 := peano_source_012 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_137_line_5: GL tag implication.
  have row_5 : (add v1 one v2) := by
    apply rule_row_5
    exact row_6
  -- chapter_137_line_4: GL tag expansion for integration.
  have row_4 : ((gl_preorder N add v1 v2) ↔ (¬ (∀ (v4 : α), ((N v4) → (¬ (add v1 v4 v2)))))) := by
    exact Iff.rfl
  -- chapter_137_line_3_rule: GL tag reformulation for integration >[bound].
  have row_3_rule : (∀ (cr3_1 : α), ((N cr3_1) → ((add v1 cr3_1 v2) → (gl_preorder N add v1 v2)))) := by
    intro cr3_1
    intro integration_premise_1
    intro integration_premise_2
    apply (row_4).2
    intro universal_counterexample
    exact universal_counterexample cr3_1 integration_premise_1 integration_premise_2
  -- chapter_137_line_3: GL tag reformulation for integration >[bound].
  have row_3 : (gl_implication87 N v1 v2 add) := by
    simpa only [gl_implication87] using row_3_rule
  -- chapter_137_line_2: GL tag expansion.
  have row_2 : (∀ (v3 : α), ((N v3) → ((add v1 v3 v2) → (gl_preorder N add v1 v2)))) := by
    simpa only [gl_implication87] using (row_3)
  -- chapter_137_line_1: GL tag implication.
  have row_1 : (gl_preorder N add v1 v2) := by
    apply row_2
    exact row_11
    exact row_5
  exact row_1

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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (gl_AnchorPeano N zero succ add mul one) := by
  -- chapter_138_line_8: GL tag task formulation.
  have row_8 : (gl_AnchorGauss N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_138_line_7: GL tag expansion for integration.
  have row_7 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_138_line_6_rule: GL tag reformulation for integration and.
  have row_6_rule : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_138_line_6: GL tag reformulation for integration and.
  have row_6 : (gl_implication76 N zero succ add mul one) := by
    simpa only [gl_implication76] using row_6_rule
  -- chapter_138_line_5: GL tag expansion.
  have row_5 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    simpa only [gl_implication76] using (row_6)
  -- chapter_138_line_4: GL tag expansion.
  have row_4 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorGauss] using (row_8)
  -- chapter_138_line_3: GL tag disintegration.
  have row_3 : (succ zero one) := by
    exact row_4.1.1.2
  -- chapter_138_line_2: GL tag disintegration.
  have row_2 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_4.1.1.1
  -- chapter_138_line_1: GL tag implication.
  have row_1 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_5
    exact row_2
    exact row_3
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (V1 : GLSet α), ((gl_interval N add zero v1 V1) → (V1 v1))) := by
  intro v1
  intro V1
  intro premise_1
  -- chapter_139_line_36: GL tag task formulation.
  have row_36 : (gl_AnchorGauss N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_139_line_35: GL tag expansion.
  have row_35 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorGauss] using (row_36)
  -- chapter_139_line_34: GL tag disintegration.
  have row_34 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_35.1.1.1
  -- chapter_139_line_33: GL tag expansion.
  have row_33 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_34)
  -- chapter_139_line_32: GL tag disintegration.
  have row_32 : (gl_implication22 N zero add) := by
    exact row_33.1.1.1.1.1.1.1.2
  -- chapter_139_line_31: GL tag expansion.
  have row_31 : (∀ (w1 : α) (w2 : α), ((w1 = w2) → ((N w1) → ((N w2) → (add w1 zero w2))))) := by
    simpa only [gl_implication22] using (row_32)
  -- chapter_139_line_30: GL tag disintegration.
  have row_30 : (succ zero one) := by
    exact row_35.1.1.2
  -- chapter_139_line_29: GL tag variable copy.
  have row_29 : (v1 = v1) := by
    rfl
  -- chapter_139_line_28: GL tag symmetry of equality.
  have row_28 : (v1 = v1) := by
    exact Eq.symm row_29
  -- chapter_139_line_27: GL tag task formulation.
  have row_27 : (gl_interval N add zero v1 V1) := by
    exact premise_1
  -- chapter_139_line_26: GL tag expansion.
  have row_26 : (((((gl_implication89 V1 N add zero) ∧ (gl_implication90 V1 N add v1)) ∧ (gl_implication91 N add zero v1 V1)) ∧ (N zero)) ∧ (N v1)) := by
    simpa only [gl_interval] using (row_27)
  -- chapter_139_line_25: GL tag disintegration.
  have row_25 : (N v1) := by
    exact row_26.2
  -- chapter_139_line_24: GL tag disintegration.
  have row_24 : (N zero) := by
    exact row_26.1.2
  -- chapter_139_line_23: GL tag disintegration.
  have row_23 : (gl_implication91 N add zero v1 V1) := by
    exact row_26.1.1.2
  -- chapter_139_line_22: GL tag expansion.
  have row_22 : (∀ (w1 : α), ((gl_preorder N add zero w1) → ((gl_preorder N add w1 v1) → (V1 w1)))) := by
    simpa only [gl_implication91] using (row_23)
  -- chapter_139_line_21: GL tag equality1.
  have row_21 : (N v1) := by
    have equality_source := row_25
    have equality_step_1 := row_29
    cases equality_step_1
    exact equality_source
  -- chapter_139_line_20: GL tag expansion for integration.
  have row_20 : ((gl_preorder N add v1 v1) ↔ (¬ (∀ (v8 : α), ((N v8) → (¬ (add v1 v8 v1)))))) := by
    exact Iff.rfl
  -- chapter_139_line_19_rule: GL tag reformulation for integration >[bound].
  have row_19_rule : (∀ (cr19_1 : α), ((N cr19_1) → ((add v1 cr19_1 v1) → (gl_preorder N add v1 v1)))) := by
    intro cr19_1
    intro integration_premise_1
    intro integration_premise_2
    apply (row_20).2
    intro universal_counterexample
    exact universal_counterexample cr19_1 integration_premise_1 integration_premise_2
  -- chapter_139_line_19: GL tag reformulation for integration >[bound].
  have row_19 : (gl_implication87 N v1 v1 add) := by
    simpa only [gl_implication87] using row_19_rule
  -- chapter_139_line_18: GL tag expansion.
  have row_18 : (∀ (v7 : α), ((N v7) → ((add v1 v7 v1) → (gl_preorder N add v1 v1)))) := by
    simpa only [gl_implication87] using (row_19)
  -- chapter_139_line_17: GL tag implication.
  have row_17 : (add v1 zero v1) := by
    apply row_31
    exact row_29
    exact row_25
    exact row_21
  -- chapter_139_line_16: GL tag equality1.
  have row_16 : (add v1 zero v1) := by
    have equality_source := row_17
    have equality_step_1 := row_28
    cases equality_step_1
    exact equality_source
  -- chapter_139_line_15: GL tag implication.
  have row_15 : (gl_preorder N add v1 v1) := by
    apply row_18
    exact row_24
    exact row_16
  -- chapter_139_line_14: GL tag expansion for integration.
  have row_14 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_139_line_13_rule: GL tag reformulation for integration and.
  have row_13_rule : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_139_line_13: GL tag reformulation for integration and.
  have row_13 : (gl_implication76 N zero succ add mul one) := by
    simpa only [gl_implication76] using row_13_rule
  -- chapter_139_line_12: GL tag expansion.
  have row_12 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    simpa only [gl_implication76] using (row_13)
  -- chapter_139_line_11: GL tag implication.
  have row_11 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_12
    exact row_34
    exact row_30
  -- chapter_139_line_10: GL tag expansion for integration.
  have row_10 : ((gl_preorder N add zero v1) ↔ (¬ (∀ (v5 : α), ((N v5) → (¬ (add zero v5 v1)))))) := by
    exact Iff.rfl
  -- chapter_139_line_9_rule: GL tag reformulation for integration >[bound].
  have row_9_rule : (∀ (cr9_1 : α), ((N cr9_1) → ((add zero cr9_1 v1) → (gl_preorder N add zero v1)))) := by
    intro cr9_1
    intro integration_premise_1
    intro integration_premise_2
    apply (row_10).2
    intro universal_counterexample
    exact universal_counterexample cr9_1 integration_premise_1 integration_premise_2
  -- chapter_139_line_9: GL tag reformulation for integration >[bound].
  have row_9 : (gl_implication87 N zero v1 add) := by
    simpa only [gl_implication87] using row_9_rule
  -- chapter_139_line_8: GL tag expansion.
  have row_8 : (∀ (v4 : α), ((N v4) → ((add zero v4 v1) → (gl_preorder N add zero v1)))) := by
    simpa only [gl_implication87] using (row_9)
  -- chapter_139_line_7: GL tag implication.
  have row_7 : (add v1 zero v1) := by
    apply row_31
    exact row_28
    exact row_21
    exact row_25
  have rule_row_6 := peano_source_035 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_139_line_6: GL tag implication.
  have row_6 : (add zero v1 v1) := by
    apply rule_row_6
    exact row_7
  -- chapter_139_line_5: GL tag implication.
  have row_5 : (gl_preorder N add zero v1) := by
    apply row_8
    exact row_21
    exact row_6
  -- chapter_139_line_4: GL tag equality1.
  have row_4 : (gl_preorder N add v1 v1) := by
    have equality_source := row_15
    have equality_step_1 := row_29
    cases equality_step_1
    exact equality_source
  -- chapter_139_line_3: GL tag equality1.
  have row_3 : (gl_preorder N add zero v1) := by
    have equality_source := row_5
    have equality_step_1 := row_29
    cases equality_step_1
    exact equality_source
  -- chapter_139_line_2: GL tag implication.
  have row_2 : (V1 v1) := by
    apply row_22
    exact row_3
    exact row_4
  -- chapter_139_line_1: GL tag equality1.
  have row_1 : (V1 v1) := by
    have equality_source := row_2
    have equality_step_1 := row_28
    cases equality_step_1
    exact equality_source
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
  -- chapter_140_line_10: GL tag task formulation.
  have row_10 : (gl_AnchorGauss N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_140_line_9: GL tag expansion.
  have row_9 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorGauss] using (row_10)
  -- chapter_140_line_8: GL tag disintegration.
  have row_8 : (succ zero one) := by
    exact row_9.1.1.2
  -- chapter_140_line_7: GL tag disintegration.
  have row_7 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_9.1.1.1
  -- chapter_140_line_6: GL tag expansion for integration.
  have row_6 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_140_line_5_rule: GL tag reformulation for integration and.
  have row_5_rule : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_140_line_5: GL tag reformulation for integration and.
  have row_5 : (gl_implication76 N zero succ add mul one) := by
    simpa only [gl_implication76] using row_5_rule
  -- chapter_140_line_4: GL tag expansion.
  have row_4 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    simpa only [gl_implication76] using (row_5)
  -- chapter_140_line_3: GL tag implication.
  have row_3 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_4
    exact row_7
    exact row_8
  -- chapter_140_line_2: GL tag task formulation.
  have row_2 : (add v1 v2 v3) := by
    exact premise_1
  have rule_row_1 := peano_source_036 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_140_line_1: GL tag implication.
  have row_1 : (add v2 v1 v3) := by
    apply rule_row_1
    exact row_2
  exact row_1

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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (v3 : α), ((gl_preorder N add v3 v1) → (gl_preorder N add v3 v2))))) := by
  intro v1
  intro v2
  intro premise_1
  intro v3
  intro premise_2
  -- chapter_142_line_24: GL tag task formulation.
  have row_24 : (gl_preorder N add v3 v1) := by
    exact premise_2
  -- chapter_142_line_23: GL tag task formulation.
  have row_23 : (gl_AnchorGauss N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_142_line_22: GL tag task formulation.
  have row_22 : (succ v1 v2) := by
    exact premise_1
  -- chapter_142_line_21: GL tag expansion.
  have row_21 : (¬ (∀ (v12 : α), ((N v12) → (¬ (add v3 v12 v1))))) := by
    simpa only [gl_preorder] using (row_24)
  have exists_row_21 : ∃ (v12 : α), ((N v12) ∧ (add v3 v12 v1)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_21
      intro v12 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v12, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v12, witness_row_21⟩ := exists_row_21
  -- chapter_142_line_20: GL tag disintegration.
  have row_20 : (N v12) := by
    exact witness_row_21.1
  -- chapter_142_line_19: GL tag disintegration.
  have row_19 : (add v3 v12 v1) := by
    exact witness_row_21.2
  -- chapter_142_line_18: GL tag expansion.
  have row_18 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorGauss] using (row_23)
  -- chapter_142_line_17: GL tag disintegration.
  have row_17 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_18.1.1.1
  -- chapter_142_line_16: GL tag expansion.
  have row_16 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_17)
  -- chapter_142_line_15: GL tag disintegration.
  have row_15 : (gl_fXY succ N N) := by
    exact row_16.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_142_line_14: GL tag expansion.
  have row_14 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_15)
  -- chapter_142_line_13: GL tag disintegration.
  have row_13 : (gl_implication6 N N succ) := by
    exact row_14.1.2
  -- chapter_142_line_12: GL tag expansion.
  have row_12 : (∀ (w1 : α), ((N w1) → (gl_existence0 N w1 succ))) := by
    simpa only [gl_implication6] using (row_13)
  -- chapter_142_line_11: GL tag disintegration.
  have row_11 : (gl_implication24 N succ add) := by
    exact row_16.1.1.1.1.1.2
  -- chapter_142_line_10: GL tag expansion.
  have row_10 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((succ w4 w5) → (add w3 w2 w5))))))))) := by
    simpa only [gl_implication24] using (row_11)
  -- chapter_142_line_9: GL tag implication.
  have row_9 : (gl_existence0 N v12 succ) := by
    apply row_12
    exact row_20
  -- chapter_142_line_8: GL tag expansion.
  have row_8 : (¬ (∀ (v5 : α), ((N v5) → (¬ (succ v12 v5))))) := by
    simpa only [gl_existence0] using (row_9)
  have exists_row_8 : ∃ (v5 : α), ((N v5) ∧ (succ v12 v5)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_8
      intro v5 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v5, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v5, witness_row_8⟩ := exists_row_8
  -- chapter_142_line_7: GL tag disintegration.
  have row_7 : (succ v12 v5) := by
    exact witness_row_8.2
  -- chapter_142_line_6: GL tag disintegration.
  have row_6 : (N v5) := by
    exact witness_row_8.1
  -- chapter_142_line_5: GL tag implication.
  have row_5 : (add v3 v5 v2) := by
    apply row_10
    exact row_20
    exact row_7
    exact row_19
    exact row_22
  -- chapter_142_line_4: GL tag expansion for integration.
  have row_4 : ((gl_preorder N add v3 v2) ↔ (¬ (∀ (v6 : α), ((N v6) → (¬ (add v3 v6 v2)))))) := by
    exact Iff.rfl
  -- chapter_142_line_3_rule: GL tag reformulation for integration >[bound].
  have row_3_rule : (∀ (cr3_1 : α), ((N cr3_1) → ((add v3 cr3_1 v2) → (gl_preorder N add v3 v2)))) := by
    intro cr3_1
    intro integration_premise_1
    intro integration_premise_2
    apply (row_4).2
    intro universal_counterexample
    exact universal_counterexample cr3_1 integration_premise_1 integration_premise_2
  -- chapter_142_line_3: GL tag reformulation for integration >[bound].
  have row_3 : (gl_implication87 N v3 v2 add) := by
    simpa only [gl_implication87] using row_3_rule
  -- chapter_142_line_2: GL tag expansion.
  have row_2 : (∀ (v4 : α), ((N v4) → ((add v3 v4 v2) → (gl_preorder N add v3 v2)))) := by
    simpa only [gl_implication87] using (row_3)
  -- chapter_142_line_1: GL tag implication.
  have row_1 : (gl_preorder N add v3 v2) := by
    apply row_2
    exact row_6
    exact row_5
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (V1 : GLBinaryRelation α) (V2 : GLSet α), ((gl_fXY V1 V2 N) → (∀ (v1 : α), ((gl_interval N add zero v1 V2) → (gl_sequence N add zero v1 V1))))) := by
  intro V1
  intro V2
  intro premise_1
  intro v1
  intro premise_2
  -- chapter_147_line_6: GL tag task formulation.
  have row_6 : (gl_interval N add zero v1 V2) := by
    exact premise_2
  -- chapter_147_line_5: GL tag task formulation.
  have row_5 : (gl_fXY V1 V2 N) := by
    exact premise_1
  -- chapter_147_line_4: GL tag expansion for integration.
  have row_4 : ((gl_sequence N add zero v1 V1) ↔ (¬ (∀ (V4 : GLSet α), ((gl_interval N add zero v1 V4) → (¬ (gl_fXY V1 V4 N)))))) := by
    exact Iff.rfl
  -- chapter_147_line_3_rule: GL tag reformulation for integration >[bound].
  have row_3_rule : (∀ (cr3_1 : GLSet α), ((gl_interval N add zero v1 cr3_1) → ((gl_fXY V1 cr3_1 N) → (gl_sequence N add zero v1 V1)))) := by
    intro cr3_1
    intro integration_premise_1
    intro integration_premise_2
    apply (row_4).2
    intro universal_counterexample
    exact universal_counterexample cr3_1 integration_premise_1 integration_premise_2
  -- chapter_147_line_3: GL tag reformulation for integration >[bound].
  have row_3 : (gl_implication95 N add zero v1 V1) := by
    simpa only [gl_implication95] using row_3_rule
  -- chapter_147_line_2: GL tag expansion.
  have row_2 : (∀ (V3 : GLSet α), ((gl_interval N add zero v1 V3) → ((gl_fXY V1 V3 N) → (gl_sequence N add zero v1 V1)))) := by
    simpa only [gl_implication95] using (row_3)
  -- chapter_147_line_1: GL tag implication.
  have row_1 : (gl_sequence N add zero v1 V1) := by
    apply row_2
    exact row_6
    exact row_5
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (V1 : GLSet α), ((gl_interval N add zero v1 V1) → (∀ (V2 : GLBinaryRelation α), ((gl_sequence N add zero v1 V2) → (gl_fXY V2 V1 N))))) := by
  intro v1
  intro V1
  intro premise_1
  intro V2
  intro premise_2
  -- chapter_148_line_51: GL tag task formulation.
  have row_51 : (gl_sequence N add zero v1 V2) := by
    exact premise_2
  -- chapter_148_line_50: GL tag task formulation.
  have row_50 : (gl_interval N add zero v1 V1) := by
    exact premise_1
  -- chapter_148_line_49: GL tag expansion for integration.
  have row_49 : ((gl_implication7 V1 V2) ↔ (∀ (v10 : α), ((V1 v10) → (∀ (v7 : α), ((V2 v10 v7) → (∀ (v8 : α), ((V2 v10 v8) → (v7 = v8)))))))) := by
    exact Iff.rfl
  -- chapter_148_line_48: GL tag expansion for integration.
  have row_48 : ((gl_implication6 V1 N V2) ↔ (∀ (v6 : α), ((V1 v6) → (gl_existence0 N v6 V2)))) := by
    exact Iff.rfl
  -- chapter_148_line_47: GL tag expansion for integration.
  have row_47 : ((gl_implication0 V2 V1) ↔ (∀ (v2 : α) (v5 : α), ((V2 v2 v5) → (V1 v2)))) := by
    exact Iff.rfl
  -- chapter_148_line_46: GL tag expansion.
  have row_46 : (¬ (∀ (V3 : GLSet α), ((gl_interval N add zero v1 V3) → (¬ (gl_fXY V2 V3 N))))) := by
    simpa only [gl_sequence] using (row_51)
  have exists_row_46 : ∃ (V3 : GLSet α), ((gl_interval N add zero v1 V3) ∧ (gl_fXY V2 V3 N)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_46
      intro V3 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨V3, counterexample_premise_0, counterexample_head⟩
  obtain ⟨V3, witness_row_46⟩ := exists_row_46
  -- chapter_148_line_45: GL tag disintegration.
  have row_45 : (gl_fXY V2 V3 N) := by
    exact witness_row_46.2
  -- chapter_148_line_44: GL tag expansion.
  have row_44 : ((((gl_implication0 V2 V3) ∧ (gl_implication1 V2 N)) ∧ (gl_implication6 V3 N V2)) ∧ (gl_implication7 V3 V2)) := by
    simpa only [gl_fXY] using (row_45)
  -- chapter_148_line_43: GL tag disintegration.
  have row_43 : (gl_implication7 V3 V2) := by
    exact row_44.2
  -- chapter_148_line_42: GL tag expansion.
  have row_42 : (∀ (w1 : α), ((V3 w1) → (∀ (w2 : α), ((V2 w1 w2) → (∀ (w3 : α), ((V2 w1 w3) → (w2 = w3))))))) := by
    simpa only [gl_implication7] using (row_43)
  -- chapter_148_line_41: GL tag disintegration.
  have row_41 : (gl_implication0 V2 V3) := by
    exact row_44.1.1.1
  -- chapter_148_line_40: GL tag expansion.
  have row_40 : (∀ (w1 : α) (w2 : α), ((V2 w1 w2) → (V3 w1))) := by
    simpa only [gl_implication0] using (row_41)
  -- chapter_148_line_39: GL tag disintegration.
  have row_39 : (gl_implication6 V3 N V2) := by
    exact row_44.1.2
  -- chapter_148_line_38: GL tag expansion.
  have row_38 : (∀ (w1 : α), ((V3 w1) → (gl_existence0 N w1 V2))) := by
    simpa only [gl_implication6] using (row_39)
  -- chapter_148_line_37: GL tag disintegration.
  have row_37 : (gl_interval N add zero v1 V3) := by
    exact witness_row_46.1
  -- chapter_148_line_36: GL tag expansion.
  have row_36 : (((((gl_implication89 V3 N add zero) ∧ (gl_implication90 V3 N add v1)) ∧ (gl_implication91 N add zero v1 V3)) ∧ (N zero)) ∧ (N v1)) := by
    simpa only [gl_interval] using (row_37)
  -- chapter_148_line_35: GL tag disintegration.
  have row_35 : (gl_implication91 N add zero v1 V3) := by
    exact row_36.1.1.2
  -- chapter_148_line_34: GL tag expansion.
  have row_34 : (∀ (w1 : α), ((gl_preorder N add zero w1) → ((gl_preorder N add w1 v1) → (V3 w1)))) := by
    simpa only [gl_implication91] using (row_35)
  -- chapter_148_line_33: GL tag disintegration.
  have row_33 : (gl_implication90 V3 N add v1) := by
    exact row_36.1.1.1.2
  -- chapter_148_line_32: GL tag expansion.
  have row_32 : (∀ (w1 : α), ((V3 w1) → (gl_preorder N add w1 v1))) := by
    simpa only [gl_implication90] using (row_33)
  -- chapter_148_line_31: GL tag disintegration.
  have row_31 : (gl_implication89 V3 N add zero) := by
    exact row_36.1.1.1.1
  -- chapter_148_line_30: GL tag expansion.
  have row_30 : (∀ (w1 : α), ((V3 w1) → (gl_preorder N add zero w1))) := by
    simpa only [gl_implication89] using (row_31)
  -- chapter_148_line_29: GL tag disintegration.
  have row_29 : (gl_implication1 V2 N) := by
    exact row_44.1.1.2
  -- chapter_148_line_28: GL tag premise element.
  have row_28 : (∀ (v10 : α), ((V1 v10) → (∀ (v7 : α), ((V2 v10 v7) → (∀ (v8 : α), ((V2 v10 v8) → (V2 v10 v7))))))) := by
    intro v10
    intro scope_premise_1
    intro v7
    intro scope_premise_2
    intro v8
    intro scope_premise_3
    exact scope_premise_2
  -- chapter_148_line_27: GL tag implication.
  have row_27 : (∀ (v10 : α), ((V1 v10) → (∀ (v7 : α), ((V2 v10 v7) → (∀ (v8 : α), ((V2 v10 v8) → (V3 v10))))))) := by
    intro v10
    intro scope_premise_1
    intro v7
    intro scope_premise_2
    intro v8
    intro scope_premise_3
    have scoped_fact_2 := row_28 v10 scope_premise_1 v7 scope_premise_2 v8 scope_premise_3
    apply row_40
    exact scoped_fact_2
  -- chapter_148_line_26: GL tag premise element.
  have row_26 : (∀ (v10 : α), ((V1 v10) → (∀ (v7 : α), ((V2 v10 v7) → (∀ (v8 : α), ((V2 v10 v8) → (V2 v10 v8))))))) := by
    intro v10
    intro scope_premise_1
    intro v7
    intro scope_premise_2
    intro v8
    intro scope_premise_3
    exact scope_premise_3
  -- chapter_148_line_25: GL tag implication.
  have row_25 : (∀ (v10 : α), ((V1 v10) → (∀ (v7 : α), ((V2 v10 v7) → (∀ (v8 : α), ((V2 v10 v8) → (v7 = v8))))))) := by
    intro v10
    intro scope_premise_1
    intro v7
    intro scope_premise_2
    intro v8
    intro scope_premise_3
    have scoped_fact_2 := row_27 v10 scope_premise_1 v7 scope_premise_2 v8 scope_premise_3
    have scoped_fact_3 := row_28 v10 scope_premise_1 v7 scope_premise_2 v8 scope_premise_3
    have scoped_fact_4 := row_26 v10 scope_premise_1 v7 scope_premise_2 v8 scope_premise_3
    apply row_42
    exact scoped_fact_2
    exact scoped_fact_3
    exact scoped_fact_4
  -- chapter_148_line_24: GL tag implication composition.
  have row_24 : (∀ (w1 : α), ((gl_preorder N add zero w1) → ((gl_preorder N add w1 v1) → (gl_existence0 N w1 V2)))) := by
    have composition_source_1 := row_34
    have composition_source_2 := row_38
    intro composition_argument_1
    intro composition_premise_0
    intro composition_premise_1
    exact composition_source_2 composition_argument_1 (composition_source_1 composition_argument_1 composition_premise_0 composition_premise_1)
  -- chapter_148_line_23: GL tag expansion.
  have row_23 : (((((gl_implication89 V1 N add zero) ∧ (gl_implication90 V1 N add v1)) ∧ (gl_implication91 N add zero v1 V1)) ∧ (N zero)) ∧ (N v1)) := by
    simpa only [gl_interval] using (row_50)
  -- chapter_148_line_22: GL tag disintegration.
  have row_22 : (gl_implication90 V1 N add v1) := by
    exact row_23.1.1.1.2
  -- chapter_148_line_21: GL tag expansion.
  have row_21 : (∀ (w1 : α), ((V1 w1) → (gl_preorder N add w1 v1))) := by
    simpa only [gl_implication90] using (row_22)
  -- chapter_148_line_20: GL tag disintegration.
  have row_20 : (gl_implication89 V1 N add zero) := by
    exact row_23.1.1.1.1
  -- chapter_148_line_19: GL tag expansion.
  have row_19 : (∀ (w1 : α), ((V1 w1) → (gl_preorder N add zero w1))) := by
    simpa only [gl_implication89] using (row_20)
  -- chapter_148_line_18: GL tag disintegration.
  have row_18 : (gl_implication91 N add zero v1 V1) := by
    exact row_23.1.1.2
  -- chapter_148_line_17: GL tag expansion.
  have row_17 : (∀ (w1 : α), ((gl_preorder N add zero w1) → ((gl_preorder N add w1 v1) → (V1 w1)))) := by
    simpa only [gl_implication91] using (row_18)
  -- chapter_148_line_16: GL tag premise element.
  have row_16 : (∀ (v6 : α), ((V1 v6) → (V1 v6))) := by
    intro v6
    intro scope_premise_1
    exact scope_premise_1
  -- chapter_148_line_15: GL tag implication.
  have row_15 : (∀ (v6 : α), ((V1 v6) → (gl_preorder N add v6 v1))) := by
    intro v6
    intro scope_premise_1
    have scoped_fact_2 := row_16 v6 scope_premise_1
    apply row_21
    exact scoped_fact_2
  -- chapter_148_line_14: GL tag implication.
  have row_14 : (∀ (v6 : α), ((V1 v6) → (gl_preorder N add zero v6))) := by
    intro v6
    intro scope_premise_1
    have scoped_fact_2 := row_16 v6 scope_premise_1
    apply row_19
    exact scoped_fact_2
  -- chapter_148_line_13: GL tag implication.
  have row_13 : (∀ (v6 : α), ((V1 v6) → (gl_existence0 N v6 V2))) := by
    intro v6
    intro scope_premise_1
    have scoped_fact_2 := row_14 v6 scope_premise_1
    have scoped_fact_3 := row_15 v6 scope_premise_1
    apply row_24
    exact scoped_fact_2
    exact scoped_fact_3
  -- chapter_148_line_12: GL tag premise element.
  have row_12 : (∀ (v2 : α) (v5 : α), ((V2 v2 v5) → (V2 v2 v5))) := by
    intro v2
    intro v5
    intro scope_premise_1
    exact scope_premise_1
  -- chapter_148_line_11: GL tag implication.
  have row_11 : (∀ (v2 : α) (v5 : α), ((V2 v2 v5) → (V3 v2))) := by
    intro v2
    intro v5
    intro scope_premise_1
    have scoped_fact_2 := row_12 v2 v5 scope_premise_1
    apply row_40
    exact scoped_fact_2
  -- chapter_148_line_10: GL tag implication.
  have row_10 : (∀ (v2 : α) (v5 : α), ((V2 v2 v5) → (gl_preorder N add v2 v1))) := by
    intro v2
    intro v5
    intro scope_premise_1
    have scoped_fact_2 := row_11 v2 v5 scope_premise_1
    apply row_32
    exact scoped_fact_2
  -- chapter_148_line_9: GL tag implication.
  have row_9 : (∀ (v2 : α) (v5 : α), ((V2 v2 v5) → (gl_preorder N add zero v2))) := by
    intro v2
    intro v5
    intro scope_premise_1
    have scoped_fact_2 := row_11 v2 v5 scope_premise_1
    apply row_30
    exact scoped_fact_2
  -- chapter_148_line_8: GL tag implication.
  have row_8 : (∀ (v2 : α) (v5 : α), ((V2 v2 v5) → (V1 v2))) := by
    intro v2
    intro v5
    intro scope_premise_1
    have scoped_fact_2 := row_9 v2 v5 scope_premise_1
    have scoped_fact_3 := row_10 v2 v5 scope_premise_1
    apply row_17
    exact scoped_fact_2
    exact scoped_fact_3
  -- chapter_148_line_7: GL tag expansion for integration.
  have row_7 : ((gl_fXY V2 V1 N) ↔ ((((gl_implication0 V2 V1) ∧ (gl_implication1 V2 N)) ∧ (gl_implication6 V1 N V2)) ∧ (gl_implication7 V1 V2))) := by
    exact Iff.rfl
  -- chapter_148_line_6_rule: GL tag reformulation for integration and.
  have row_6_rule : ((gl_implication0 V2 V1) → ((gl_implication1 V2 N) → ((gl_implication6 V1 N V2) → ((gl_implication7 V1 V2) → (gl_fXY V2 V1 N))))) := by
    intro integration_premise_1
    intro integration_premise_2
    intro integration_premise_3
    intro integration_premise_4
    simp only [gl_fXY]
    exact ⟨⟨⟨integration_premise_1, integration_premise_2⟩, integration_premise_3⟩, integration_premise_4⟩
  -- chapter_148_line_6: GL tag reformulation for integration and.
  have row_6 : (gl_implication8 V2 V1 N) := by
    simpa only [gl_implication8] using row_6_rule
  -- chapter_148_line_5: GL tag expansion.
  have row_5 : ((gl_implication0 V2 V1) → ((gl_implication1 V2 N) → ((gl_implication6 V1 N V2) → ((gl_implication7 V1 V2) → (gl_fXY V2 V1 N))))) := by
    simpa only [gl_implication8] using (row_6)
  -- chapter_148_line_4: GL tag validity name.
  have row_4 : (gl_implication6 V1 N V2) := by
    simpa only [gl_implication6] using (row_13)
  -- chapter_148_line_3: GL tag validity name.
  have row_3 : (gl_implication7 V1 V2) := by
    simpa only [gl_implication7] using (row_25)
  -- chapter_148_line_2: GL tag validity name.
  have row_2 : (gl_implication0 V2 V1) := by
    simpa only [gl_implication0] using (row_8)
  -- chapter_148_line_1: GL tag implication.
  have row_1 : (gl_fXY V2 V1 N) := by
    apply row_5
    exact row_2
    exact row_29
    exact row_4
    exact row_3
  exact row_1

theorem gauss_source_097
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (V1 : GLSet α), ((gl_interval N add zero v1 V1) → ((gl_interval N add zero one V1) → (V1 v1)))) := by
  intro v1
  intro V1
  intro premise_1
  intro premise_2
  -- chapter_171_line_38: GL tag task formulation.
  have row_38 : (gl_interval N add zero one V1) := by
    exact premise_2
  -- chapter_171_line_37: GL tag task formulation.
  have row_37 : (gl_AnchorGauss N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_171_line_36: GL tag expansion.
  have row_36 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorGauss] using (row_37)
  -- chapter_171_line_35: GL tag disintegration.
  have row_35 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_36.1.1.1
  -- chapter_171_line_34: GL tag expansion.
  have row_34 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_35)
  -- chapter_171_line_33: GL tag disintegration.
  have row_33 : (gl_implication22 N zero add) := by
    exact row_34.1.1.1.1.1.1.1.2
  -- chapter_171_line_32: GL tag expansion.
  have row_32 : (∀ (w1 : α) (w2 : α), ((w1 = w2) → ((N w1) → ((N w2) → (add w1 zero w2))))) := by
    simpa only [gl_implication22] using (row_33)
  -- chapter_171_line_31: GL tag disintegration.
  have row_31 : (succ zero one) := by
    exact row_36.1.1.2
  -- chapter_171_line_30: GL tag variable copy.
  have row_30 : (v1 = v1) := by
    rfl
  -- chapter_171_line_29: GL tag symmetry of equality.
  have row_29 : (v1 = v1) := by
    exact Eq.symm row_30
  -- chapter_171_line_28: GL tag task formulation.
  have row_28 : (gl_interval N add zero v1 V1) := by
    exact premise_1
  -- chapter_171_line_27: GL tag expansion.
  have row_27 : (((((gl_implication89 V1 N add zero) ∧ (gl_implication90 V1 N add v1)) ∧ (gl_implication91 N add zero v1 V1)) ∧ (N zero)) ∧ (N v1)) := by
    simpa only [gl_interval] using (row_28)
  -- chapter_171_line_26: GL tag disintegration.
  have row_26 : (N v1) := by
    exact row_27.2
  -- chapter_171_line_25: GL tag disintegration.
  have row_25 : (gl_implication91 N add zero v1 V1) := by
    exact row_27.1.1.2
  -- chapter_171_line_24: GL tag expansion.
  have row_24 : (∀ (w1 : α), ((gl_preorder N add zero w1) → ((gl_preorder N add w1 v1) → (V1 w1)))) := by
    simpa only [gl_implication91] using (row_25)
  -- chapter_171_line_23: GL tag equality1.
  have row_23 : (N v1) := by
    have equality_source := row_26
    have equality_step_1 := row_30
    cases equality_step_1
    exact equality_source
  -- chapter_171_line_22: GL tag expansion.
  have row_22 : (((((gl_implication89 V1 N add zero) ∧ (gl_implication90 V1 N add one)) ∧ (gl_implication91 N add zero one V1)) ∧ (N zero)) ∧ (N one)) := by
    simpa only [gl_interval] using (row_38)
  -- chapter_171_line_21: GL tag disintegration.
  have row_21 : (N zero) := by
    exact row_22.1.2
  -- chapter_171_line_20: GL tag expansion for integration.
  have row_20 : ((gl_preorder N add v1 v1) ↔ (¬ (∀ (v8 : α), ((N v8) → (¬ (add v1 v8 v1)))))) := by
    exact Iff.rfl
  -- chapter_171_line_19_rule: GL tag reformulation for integration >[bound].
  have row_19_rule : (∀ (cr19_1 : α), ((N cr19_1) → ((add v1 cr19_1 v1) → (gl_preorder N add v1 v1)))) := by
    intro cr19_1
    intro integration_premise_1
    intro integration_premise_2
    apply (row_20).2
    intro universal_counterexample
    exact universal_counterexample cr19_1 integration_premise_1 integration_premise_2
  -- chapter_171_line_19: GL tag reformulation for integration >[bound].
  have row_19 : (gl_implication87 N v1 v1 add) := by
    simpa only [gl_implication87] using row_19_rule
  -- chapter_171_line_18: GL tag expansion.
  have row_18 : (∀ (v7 : α), ((N v7) → ((add v1 v7 v1) → (gl_preorder N add v1 v1)))) := by
    simpa only [gl_implication87] using (row_19)
  -- chapter_171_line_17: GL tag implication.
  have row_17 : (add v1 zero v1) := by
    apply row_32
    exact row_30
    exact row_26
    exact row_23
  -- chapter_171_line_16: GL tag equality1.
  have row_16 : (add v1 zero v1) := by
    have equality_source := row_17
    have equality_step_1 := row_29
    cases equality_step_1
    exact equality_source
  -- chapter_171_line_15: GL tag implication.
  have row_15 : (gl_preorder N add v1 v1) := by
    apply row_18
    exact row_21
    exact row_16
  -- chapter_171_line_14: GL tag expansion for integration.
  have row_14 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_171_line_13_rule: GL tag reformulation for integration and.
  have row_13_rule : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_171_line_13: GL tag reformulation for integration and.
  have row_13 : (gl_implication76 N zero succ add mul one) := by
    simpa only [gl_implication76] using row_13_rule
  -- chapter_171_line_12: GL tag expansion.
  have row_12 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    simpa only [gl_implication76] using (row_13)
  -- chapter_171_line_11: GL tag implication.
  have row_11 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_12
    exact row_35
    exact row_31
  -- chapter_171_line_10: GL tag expansion for integration.
  have row_10 : ((gl_preorder N add zero v1) ↔ (¬ (∀ (v5 : α), ((N v5) → (¬ (add zero v5 v1)))))) := by
    exact Iff.rfl
  -- chapter_171_line_9_rule: GL tag reformulation for integration >[bound].
  have row_9_rule : (∀ (cr9_1 : α), ((N cr9_1) → ((add zero cr9_1 v1) → (gl_preorder N add zero v1)))) := by
    intro cr9_1
    intro integration_premise_1
    intro integration_premise_2
    apply (row_10).2
    intro universal_counterexample
    exact universal_counterexample cr9_1 integration_premise_1 integration_premise_2
  -- chapter_171_line_9: GL tag reformulation for integration >[bound].
  have row_9 : (gl_implication87 N zero v1 add) := by
    simpa only [gl_implication87] using row_9_rule
  -- chapter_171_line_8: GL tag expansion.
  have row_8 : (∀ (v4 : α), ((N v4) → ((add zero v4 v1) → (gl_preorder N add zero v1)))) := by
    simpa only [gl_implication87] using (row_9)
  -- chapter_171_line_7: GL tag implication.
  have row_7 : (add v1 zero v1) := by
    apply row_32
    exact row_29
    exact row_23
    exact row_26
  have rule_row_6 := peano_source_035 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_171_line_6: GL tag implication.
  have row_6 : (add zero v1 v1) := by
    apply rule_row_6
    exact row_7
  -- chapter_171_line_5: GL tag implication.
  have row_5 : (gl_preorder N add zero v1) := by
    apply row_8
    exact row_23
    exact row_6
  -- chapter_171_line_4: GL tag equality1.
  have row_4 : (gl_preorder N add v1 v1) := by
    have equality_source := row_15
    have equality_step_1 := row_30
    cases equality_step_1
    exact equality_source
  -- chapter_171_line_3: GL tag equality1.
  have row_3 : (gl_preorder N add zero v1) := by
    have equality_source := row_5
    have equality_step_1 := row_30
    cases equality_step_1
    exact equality_source
  -- chapter_171_line_2: GL tag implication.
  have row_2 : (V1 v1) := by
    apply row_24
    exact row_3
    exact row_4
  -- chapter_171_line_1: GL tag equality1.
  have row_1 : (V1 v1) := by
    have equality_source := row_2
    have equality_step_1 := row_29
    cases equality_step_1
    exact equality_source
  exact row_1

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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((gl_preorder N add v1 v2) → (∀ (v3 : α), ((gl_preorder N add v2 v3) → (gl_preorder N add v1 v3))))) := by
  intro v1
  intro v2
  intro premise_1
  intro v3
  intro premise_2
  apply Classical.byContradiction
  intro reductio
  -- chapter_141_line_35: GL tag compilation.
  have row_35 : (gl_implication1965 (α := α)) := by
    simp only [gl_implication1965]
    intro compiled_N compiled_zero compiled_succ compiled_add compiled_mul compiled_one
    intro compiled_anchor compiled_induction compiled_strong_induction
    exact peano_source_042 compiled_N compiled_zero compiled_succ compiled_add compiled_mul compiled_one compiled_anchor compiled_induction compiled_strong_induction
  -- chapter_141_line_34: GL tag expansion.
  have row_34 : (∀ (N : GLSet α) (zero : α) (succ : GLBinaryRelation α) (add : GLTernaryRelation α) (mul : GLTernaryRelation α) (one : α), ((gl_AnchorPeano N zero succ add mul one) → (RelationalInduction N zero succ → StrongRelationalInduction N zero succ add → (∀ (v1 : α) (v2 : α) (v3 : α), ((add v1 v2 v3) → (∀ (v4 : α) (v5 : α), ((add v4 v5 v1) → (∀ (v6 : α), ((add v2 v5 v6) → (add v6 v4 v3)))))))))) := by
    simpa only [gl_implication1965] using (row_35)
  -- chapter_141_line_33: GL tag task formulation.
  have row_33 : (¬ (gl_preorder N add v1 v3)) := by
    exact reductio
  -- chapter_141_line_32: GL tag expansion.
  have row_32 : (gl_implication86 v1 v3 add N) := by
    simp only [gl_implication86]
    intro existence_witness_1 compact_premise compact_negated
    have unfolded_row_33 := row_33
    simp only [gl_preorder] at unfolded_row_33
    apply unfolded_row_33
    intro positive_row_33
    exact positive_row_33 existence_witness_1 compact_negated compact_premise
  -- chapter_141_line_31: GL tag expansion.
  have row_31 : (∀ (w1 : α), ((add v1 w1 v3) → (¬ (N w1)))) := by
    simpa only [gl_implication86] using (row_32)
  -- chapter_141_line_30: GL tag task formulation.
  have row_30 : (gl_preorder N add v2 v3) := by
    exact premise_2
  -- chapter_141_line_29: GL tag expansion.
  have row_29 : (¬ (∀ (v5 : α), ((N v5) → (¬ (add v2 v5 v3))))) := by
    simpa only [gl_preorder] using (row_30)
  have exists_row_29 : ∃ (v5 : α), ((N v5) ∧ (add v2 v5 v3)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_29
      intro v5 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v5, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v5, witness_row_29⟩ := exists_row_29
  -- chapter_141_line_28: GL tag disintegration.
  have row_28 : (N v5) := by
    exact witness_row_29.1
  -- chapter_141_line_27: GL tag disintegration.
  have row_27 : (add v2 v5 v3) := by
    exact witness_row_29.2
  -- chapter_141_line_26: GL tag task formulation.
  have row_26 : (gl_preorder N add v1 v2) := by
    exact premise_1
  -- chapter_141_line_25: GL tag expansion.
  have row_25 : (¬ (∀ (v6 : α), ((N v6) → (¬ (add v1 v6 v2))))) := by
    simpa only [gl_preorder] using (row_26)
  have exists_row_25 : ∃ (v6 : α), ((N v6) ∧ (add v1 v6 v2)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_25
      intro v6 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v6, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v6, witness_row_25⟩ := exists_row_25
  -- chapter_141_line_24: GL tag disintegration.
  have row_24 : (N v6) := by
    exact witness_row_25.1
  -- chapter_141_line_23: GL tag disintegration.
  have row_23 : (add v1 v6 v2) := by
    exact witness_row_25.2
  -- chapter_141_line_22: GL tag task formulation.
  have row_22 : (gl_AnchorGauss N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_141_line_21: GL tag expansion.
  have row_21 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorGauss] using (row_22)
  -- chapter_141_line_20: GL tag disintegration.
  have row_20 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_21.1.1.1
  -- chapter_141_line_19: GL tag disintegration.
  have row_19 : (succ zero one) := by
    exact row_21.1.1.2
  -- chapter_141_line_18: GL tag expansion.
  have row_18 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_20)
  -- chapter_141_line_17: GL tag disintegration.
  have row_17 : (gl_fXYZ add N N N) := by
    exact row_18.1.1.1.1.1.1.1.1.1.2
  -- chapter_141_line_16: GL tag expansion.
  have row_16 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_17)
  -- chapter_141_line_15: GL tag disintegration.
  have row_15 : (gl_implication18 N N N add) := by
    exact row_16.1.2
  -- chapter_141_line_14: GL tag expansion.
  have row_14 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication18] using (row_15)
  -- chapter_141_line_13: GL tag expansion for integration.
  have row_13 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_141_line_12_rule: GL tag reformulation for integration and.
  have row_12_rule : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_141_line_12: GL tag reformulation for integration and.
  have row_12 : (gl_implication76 N zero succ add mul one) := by
    simpa only [gl_implication76] using row_12_rule
  -- chapter_141_line_11: GL tag expansion.
  have row_11 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    simpa only [gl_implication76] using (row_12)
  -- chapter_141_line_10: GL tag implication.
  have row_10 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_11
    exact row_20
    exact row_19
  -- chapter_141_line_9: GL tag implication.
  have row_9 : (gl_existence1 N v5 v6 add) := by
    apply row_14
    exact row_28
    exact row_24
  -- chapter_141_line_8: GL tag expansion.
  have row_8 : (¬ (∀ (v4 : α), ((N v4) → (¬ (add v5 v6 v4))))) := by
    simpa only [gl_existence1] using (row_9)
  have exists_row_8 : ∃ (v4 : α), ((N v4) ∧ (add v5 v6 v4)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_8
      intro v4 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v4, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v4, witness_row_8⟩ := exists_row_8
  -- chapter_141_line_7: GL tag disintegration.
  have row_7 : (add v5 v6 v4) := by
    exact witness_row_8.2
  -- chapter_141_line_6: GL tag disintegration.
  have row_6 : (N v4) := by
    exact witness_row_8.1
  have rule_row_5 := peano_source_042 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_141_line_5: GL tag implication.
  have row_5 : (add v4 v1 v3) := by
    apply rule_row_5
    exact row_27
    exact row_23
    exact row_7
  -- chapter_141_line_4: GL tag theorem.
  have row_4 := gauss_source_070 N zero succ add mul one two identity anchor relationalInduction strongRelationalInduction
  -- chapter_141_line_3: GL tag implication.
  have row_3 : (add v1 v4 v3) := by
    apply row_4
    exact row_5
  -- chapter_141_line_2: GL tag implication.
  have row_2 : (¬ (N v4)) := by
    apply row_31
    exact row_3
  -- chapter_141_line_1: GL tag contradiction.
  have row_1 : (gl_preorder N add v1 v3) := by
    exact False.elim (row_2 row_6)
  exact reductio row_1

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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
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
  -- chapter_143_line_50: GL tag theorem.
  have row_50 := gauss_source_072 N zero succ add mul one two identity anchor relationalInduction strongRelationalInduction
  -- chapter_143_line_49: GL tag task formulation.
  have row_49 : (gl_AnchorGauss N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_143_line_48: GL tag task formulation.
  have row_48 : (succ v1 v2) := by
    exact premise_1
  -- chapter_143_line_47: GL tag task formulation.
  have row_47 : (gl_interval N add zero v2 V1) := by
    exact premise_3
  -- chapter_143_line_46: GL tag task formulation.
  have row_46 : (gl_limitSet N add V1 v1 V2) := by
    exact premise_2
  -- chapter_143_line_45: GL tag expansion for integration.
  have row_45 : ((gl_implication91 N add zero v1 V2) ↔ (∀ (v5 : α), ((gl_preorder N add zero v5) → ((gl_preorder N add v5 v1) → (V2 v5))))) := by
    exact Iff.rfl
  -- chapter_143_line_44: GL tag expansion for integration.
  have row_44 : ((gl_implication89 V2 N add zero) ↔ (∀ (v3 : α), ((V2 v3) → (gl_preorder N add zero v3)))) := by
    exact Iff.rfl
  -- chapter_143_line_43: GL tag expansion.
  have row_43 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorGauss] using (row_49)
  -- chapter_143_line_42: GL tag disintegration.
  have row_42 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_43.1.1.1
  -- chapter_143_line_41: GL tag expansion.
  have row_41 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_42)
  -- chapter_143_line_40: GL tag disintegration.
  have row_40 : (gl_fXY succ N N) := by
    exact row_41.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_143_line_39: GL tag expansion.
  have row_39 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_40)
  -- chapter_143_line_38: GL tag disintegration.
  have row_38 : (gl_implication1 succ N) := by
    exact row_39.1.1.2
  -- chapter_143_line_37: GL tag expansion.
  have row_37 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using (row_38)
  -- chapter_143_line_36: GL tag disintegration.
  have row_36 : (gl_implication0 succ N) := by
    exact row_39.1.1.1
  -- chapter_143_line_35: GL tag expansion.
  have row_35 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_36)
  -- chapter_143_line_34: GL tag implication grouping.
  have row_34 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (gl_and20 w1 N w2 N))) := by
    have auxiliary_source_1 := row_35
    have auxiliary_source_2 := row_37
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_premise
    exact ⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_premise)⟩
  -- chapter_143_line_33: GL tag implication.
  have row_33 : (gl_and20 v1 N v2 N) := by
    apply row_34
    exact row_48
  -- chapter_143_line_32: GL tag expansion.
  have row_32 : ((N v1) ∧ (N v2)) := by
    simpa only [gl_and20] using (row_33)
  -- chapter_143_line_31: GL tag disintegration.
  have row_31 : (N v1) := by
    exact row_32.1
  -- chapter_143_line_30: GL tag expansion.
  have row_30 : (((((gl_implication89 V1 N add zero) ∧ (gl_implication90 V1 N add v2)) ∧ (gl_implication91 N add zero v2 V1)) ∧ (N zero)) ∧ (N v2)) := by
    simpa only [gl_interval] using (row_47)
  -- chapter_143_line_29: GL tag disintegration.
  have row_29 : (gl_implication91 N add zero v2 V1) := by
    exact row_30.1.1.2
  -- chapter_143_line_28: GL tag expansion.
  have row_28 : (∀ (w1 : α), ((gl_preorder N add zero w1) → ((gl_preorder N add w1 v2) → (V1 w1)))) := by
    simpa only [gl_implication91] using (row_29)
  -- chapter_143_line_27: GL tag disintegration.
  have row_27 : (gl_implication89 V1 N add zero) := by
    exact row_30.1.1.1.1
  -- chapter_143_line_26: GL tag expansion.
  have row_26 : (∀ (w1 : α), ((V1 w1) → (gl_preorder N add zero w1))) := by
    simpa only [gl_implication89] using (row_27)
  -- chapter_143_line_25: GL tag disintegration.
  have row_25 : (N zero) := by
    exact row_30.1.2
  -- chapter_143_line_24: GL tag expansion.
  have row_24 : (((gl_subset V2 V1) ∧ (gl_implication90 V2 N add v1)) ∧ (gl_implication141 V1 N add v1 V2)) := by
    simpa only [gl_limitSet] using (row_46)
  -- chapter_143_line_23: GL tag disintegration.
  have row_23 : (gl_implication90 V2 N add v1) := by
    exact row_24.1.2
  -- chapter_143_line_22: GL tag expansion.
  have row_22 : (∀ (w1 : α), ((V2 w1) → (gl_preorder N add w1 v1))) := by
    simpa only [gl_implication90] using (row_23)
  -- chapter_143_line_21: GL tag disintegration.
  have row_21 : (gl_subset V2 V1) := by
    exact row_24.1.1
  -- chapter_143_line_20: GL tag expansion.
  have row_20 : (∀ (w1 : α), ((V2 w1) → (V1 w1))) := by
    simpa only [gl_subset] using (row_21)
  -- chapter_143_line_19: GL tag disintegration.
  have row_19 : (gl_implication141 V1 N add v1 V2) := by
    exact row_24.2
  -- chapter_143_line_18: GL tag expansion.
  have row_18 : (∀ (w1 : α), ((V1 w1) → ((gl_preorder N add w1 v1) → (V2 w1)))) := by
    simpa only [gl_implication141] using (row_19)
  -- chapter_143_line_17: GL tag premise element.
  have row_17 : (∀ (v5 : α), ((gl_preorder N add zero v5) → ((gl_preorder N add v5 v1) → (gl_preorder N add v5 v1)))) := by
    intro v5
    intro scope_premise_1
    intro scope_premise_2
    exact scope_premise_2
  -- chapter_143_line_16: GL tag implication.
  have row_16 : (∀ (v5 : α), ((gl_preorder N add zero v5) → ((gl_preorder N add v5 v1) → (gl_preorder N add v5 v2)))) := by
    intro v5
    intro scope_premise_1
    intro scope_premise_2
    have scoped_fact_3 := row_17 v5 scope_premise_1 scope_premise_2
    apply row_50
    exact row_48
    exact scoped_fact_3
  -- chapter_143_line_15: GL tag premise element.
  have row_15 : (∀ (v5 : α), ((gl_preorder N add zero v5) → ((gl_preorder N add v5 v1) → (gl_preorder N add zero v5)))) := by
    intro v5
    intro scope_premise_1
    intro scope_premise_2
    exact scope_premise_1
  -- chapter_143_line_14: GL tag implication.
  have row_14 : (∀ (v5 : α), ((gl_preorder N add zero v5) → ((gl_preorder N add v5 v1) → (V1 v5)))) := by
    intro v5
    intro scope_premise_1
    intro scope_premise_2
    have scoped_fact_2 := row_15 v5 scope_premise_1 scope_premise_2
    have scoped_fact_3 := row_16 v5 scope_premise_1 scope_premise_2
    apply row_28
    exact scoped_fact_2
    exact scoped_fact_3
  -- chapter_143_line_13: GL tag implication.
  have row_13 : (∀ (v5 : α), ((gl_preorder N add zero v5) → ((gl_preorder N add v5 v1) → (V2 v5)))) := by
    intro v5
    intro scope_premise_1
    intro scope_premise_2
    have scoped_fact_2 := row_14 v5 scope_premise_1 scope_premise_2
    have scoped_fact_3 := row_17 v5 scope_premise_1 scope_premise_2
    apply row_18
    exact scoped_fact_2
    exact scoped_fact_3
  -- chapter_143_line_12: GL tag implication composition.
  have row_12 : (∀ (w1 : α), ((V2 w1) → (gl_preorder N add zero w1))) := by
    have composition_source_1 := row_20
    have composition_source_2 := row_26
    intro composition_argument_1
    intro composition_premise_0
    exact composition_source_2 composition_argument_1 (composition_source_1 composition_argument_1 composition_premise_0)
  -- chapter_143_line_11: GL tag implication composition.
  have row_11 : (∀ (w1 : α), ((V1 w1) → ((gl_preorder N add w1 v1) → (gl_preorder N add zero w1)))) := by
    have composition_source_1 := row_18
    have composition_source_2 := row_12
    intro composition_argument_1
    intro composition_premise_0
    intro composition_premise_1
    exact composition_source_2 composition_argument_1 (composition_source_1 composition_argument_1 composition_premise_0 composition_premise_1)
  -- chapter_143_line_10: GL tag premise element.
  have row_10 : (∀ (v3 : α), ((V2 v3) → (V2 v3))) := by
    intro v3
    intro scope_premise_1
    exact scope_premise_1
  -- chapter_143_line_9: GL tag implication.
  have row_9 : (∀ (v3 : α), ((V2 v3) → (gl_preorder N add v3 v1))) := by
    intro v3
    intro scope_premise_1
    have scoped_fact_2 := row_10 v3 scope_premise_1
    apply row_22
    exact scoped_fact_2
  -- chapter_143_line_8: GL tag implication.
  have row_8 : (∀ (v3 : α), ((V2 v3) → (V1 v3))) := by
    intro v3
    intro scope_premise_1
    have scoped_fact_2 := row_10 v3 scope_premise_1
    apply row_20
    exact scoped_fact_2
  -- chapter_143_line_7: GL tag implication.
  have row_7 : (∀ (v3 : α), ((V2 v3) → (gl_preorder N add zero v3))) := by
    intro v3
    intro scope_premise_1
    have scoped_fact_2 := row_8 v3 scope_premise_1
    have scoped_fact_3 := row_9 v3 scope_premise_1
    apply row_11
    exact scoped_fact_2
    exact scoped_fact_3
  -- chapter_143_line_6: GL tag expansion for integration.
  have row_6 : ((gl_interval N add zero v1 V2) ↔ (((((gl_implication89 V2 N add zero) ∧ (gl_implication90 V2 N add v1)) ∧ (gl_implication91 N add zero v1 V2)) ∧ (N zero)) ∧ (N v1))) := by
    exact Iff.rfl
  -- chapter_143_line_5_rule: GL tag reformulation for integration and.
  have row_5_rule : ((gl_implication89 V2 N add zero) → ((gl_implication90 V2 N add v1) → ((gl_implication91 N add zero v1 V2) → ((N zero) → ((N v1) → (gl_interval N add zero v1 V2)))))) := by
    intro integration_premise_1
    intro integration_premise_2
    intro integration_premise_3
    intro integration_premise_4
    intro integration_premise_5
    simp only [gl_interval]
    exact ⟨⟨⟨⟨integration_premise_1, integration_premise_2⟩, integration_premise_3⟩, integration_premise_4⟩, integration_premise_5⟩
  -- chapter_143_line_5: GL tag reformulation for integration and.
  have row_5 : (gl_implication92 V2 N add zero v1) := by
    simpa only [gl_implication92] using row_5_rule
  -- chapter_143_line_4: GL tag expansion.
  have row_4 : ((gl_implication89 V2 N add zero) → ((gl_implication90 V2 N add v1) → ((gl_implication91 N add zero v1 V2) → ((N zero) → ((N v1) → (gl_interval N add zero v1 V2)))))) := by
    simpa only [gl_implication92] using (row_5)
  -- chapter_143_line_3: GL tag validity name.
  have row_3 : (gl_implication91 N add zero v1 V2) := by
    simpa only [gl_implication91] using (row_13)
  -- chapter_143_line_2: GL tag validity name.
  have row_2 : (gl_implication89 V2 N add zero) := by
    simpa only [gl_implication89] using (row_7)
  -- chapter_143_line_1: GL tag implication.
  have row_1 : (gl_interval N add zero v1 V2) := by
    apply row_4
    exact row_2
    exact row_23
    exact row_3
    exact row_25
    exact row_31
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
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
  -- chapter_155_line_73: GL tag task formulation.
  have row_73 : (gl_interval N add zero v2 V2) := by
    exact reductio
  -- chapter_155_line_72: GL tag expansion.
  have row_72 : (((((gl_implication89 V2 N add zero) ∧ (gl_implication90 V2 N add v2)) ∧ (gl_implication91 N add zero v2 V2)) ∧ (N zero)) ∧ (N v2)) := by
    simpa only [gl_interval] using (row_73)
  -- chapter_155_line_71: GL tag disintegration.
  have row_71 : (N zero) := by
    exact row_72.1.2
  -- chapter_155_line_70: GL tag task formulation.
  have row_70 : (gl_limitSet N add V1 v1 V2) := by
    exact premise_2
  -- chapter_155_line_69: GL tag expansion.
  have row_69 : (((gl_subset V2 V1) ∧ (gl_implication90 V2 N add v1)) ∧ (gl_implication141 V1 N add v1 V2)) := by
    simpa only [gl_limitSet] using (row_70)
  -- chapter_155_line_68: GL tag disintegration.
  have row_68 : (gl_implication90 V2 N add v1) := by
    exact row_69.1.2
  -- chapter_155_line_67: GL tag expansion.
  have row_67 : (∀ (w1 : α), ((V2 w1) → (gl_preorder N add w1 v1))) := by
    simpa only [gl_implication90] using (row_68)
  -- chapter_155_line_66: GL tag task formulation.
  have row_66 : (gl_interval N add zero one V1) := by
    exact premise_3
  -- chapter_155_line_65: GL tag expansion.
  have row_65 : (((((gl_implication89 V1 N add zero) ∧ (gl_implication90 V1 N add one)) ∧ (gl_implication91 N add zero one V1)) ∧ (N zero)) ∧ (N one)) := by
    simpa only [gl_interval] using (row_66)
  -- chapter_155_line_64: GL tag disintegration.
  have row_64 : (N one) := by
    exact row_65.2
  -- chapter_155_line_63: GL tag task formulation.
  have row_63 : (gl_AnchorGauss N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_155_line_62: GL tag theorem.
  have row_62 := gauss_source_069 N zero succ add mul one two identity anchor relationalInduction strongRelationalInduction
  -- chapter_155_line_61: GL tag implication.
  have row_61 : (V2 v2) := by
    apply row_62
    exact row_73
  -- chapter_155_line_60: GL tag implication.
  have row_60 : (gl_preorder N add v2 v1) := by
    apply row_67
    exact row_61
  -- chapter_155_line_59: GL tag expansion.
  have row_59 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorGauss] using (row_63)
  -- chapter_155_line_58: GL tag disintegration.
  have row_58 : (succ zero one) := by
    exact row_59.1.1.2
  -- chapter_155_line_57: GL tag disintegration.
  have row_57 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_59.1.1.1
  -- chapter_155_line_56: GL tag expansion.
  have row_56 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_57)
  -- chapter_155_line_55: GL tag disintegration.
  have row_55 : (gl_fXY succ N N) := by
    exact row_56.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_155_line_54: GL tag expansion.
  have row_54 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_55)
  -- chapter_155_line_53: GL tag disintegration.
  have row_53 : (gl_implication1 succ N) := by
    exact row_54.1.1.2
  -- chapter_155_line_52: GL tag expansion.
  have row_52 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using (row_53)
  -- chapter_155_line_51: GL tag disintegration.
  have row_51 : (gl_implication0 succ N) := by
    exact row_54.1.1.1
  -- chapter_155_line_50: GL tag expansion.
  have row_50 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_51)
  -- chapter_155_line_49: GL tag disintegration.
  have row_49 : (gl_fXYZ add N N N) := by
    exact row_56.1.1.1.1.1.1.1.1.1.2
  -- chapter_155_line_48: GL tag expansion.
  have row_48 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_49)
  -- chapter_155_line_47: GL tag disintegration.
  have row_47 : (gl_implication18 N N N add) := by
    exact row_48.1.2
  -- chapter_155_line_46: GL tag expansion.
  have row_46 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication18] using (row_47)
  -- chapter_155_line_45: GL tag disintegration.
  have row_45 : (gl_implication21 N zero add) := by
    exact row_56.1.1.1.1.1.1.1.1.2
  -- chapter_155_line_44: GL tag expansion.
  have row_44 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((add w1 zero w2) → (w1 = w2))))) := by
    simpa only [gl_implication21] using (row_45)
  -- chapter_155_line_43: GL tag disintegration.
  have row_43 : (gl_implication10 N succ) := by
    exact row_56.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_155_line_42: GL tag expansion.
  have row_42 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w2 w1) → (∀ (w3 : α), ((succ w3 w1) → (w2 = w3))))))) := by
    simpa only [gl_implication10] using (row_43)
  -- chapter_155_line_41: GL tag disintegration.
  have row_41 : (gl_implication24 N succ add) := by
    exact row_56.1.1.1.1.1.2
  -- chapter_155_line_40: GL tag expansion.
  have row_40 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((succ w4 w5) → (add w3 w2 w5))))))))) := by
    simpa only [gl_implication24] using (row_41)
  -- chapter_155_line_39: GL tag disintegration.
  have row_39 : (succ one two) := by
    exact row_59.1.2
  -- chapter_155_line_38: GL tag disintegration.
  have row_38 : (gl_implication9 N zero succ) := by
    exact row_56.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_155_line_37: GL tag expansion.
  have row_37 : (∀ (w1 : α), ((N w1) → (¬ (succ w1 zero)))) := by
    simpa only [gl_implication9] using (row_38)
  -- chapter_155_line_36: GL tag expansion for integration.
  have row_36 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_155_line_35_rule: GL tag reformulation for integration and.
  have row_35_rule : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_155_line_35: GL tag reformulation for integration and.
  have row_35 : (gl_implication76 N zero succ add mul one) := by
    simpa only [gl_implication76] using row_35_rule
  -- chapter_155_line_34: GL tag expansion.
  have row_34 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    simpa only [gl_implication76] using (row_35)
  -- chapter_155_line_33: GL tag implication.
  have row_33 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_34
    exact row_57
    exact row_58
  -- chapter_155_line_32: GL tag task formulation.
  have row_32 : (succ v1 v2) := by
    exact premise_1
  have rule_row_31 := peano_source_010 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_155_line_31: GL tag implication.
  have row_31 : (add one v1 v2) := by
    apply rule_row_31
    exact row_32
  have rule_row_30 := peano_source_012 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_155_line_30: GL tag implication.
  have row_30 : (add v1 one v2) := by
    apply rule_row_30
    exact row_31
  -- chapter_155_line_29: GL tag expansion for integration.
  have row_29 : ((gl_preorder N add v1 v2) ↔ (¬ (∀ (v11 : α), ((N v11) → (¬ (add v1 v11 v2)))))) := by
    exact Iff.rfl
  -- chapter_155_line_28_rule: GL tag reformulation for integration >[bound].
  have row_28_rule : (∀ (cr28_1 : α), ((N cr28_1) → ((add v1 cr28_1 v2) → (gl_preorder N add v1 v2)))) := by
    intro cr28_1
    intro integration_premise_1
    intro integration_premise_2
    apply (row_29).2
    intro universal_counterexample
    exact universal_counterexample cr28_1 integration_premise_1 integration_premise_2
  -- chapter_155_line_28: GL tag reformulation for integration >[bound].
  have row_28 : (gl_implication87 N v1 v2 add) := by
    simpa only [gl_implication87] using row_28_rule
  -- chapter_155_line_27: GL tag expansion.
  have row_27 : (∀ (v10 : α), ((N v10) → ((add v1 v10 v2) → (gl_preorder N add v1 v2)))) := by
    simpa only [gl_implication87] using (row_28)
  -- chapter_155_line_26: GL tag implication.
  have row_26 : (gl_preorder N add v1 v2) := by
    apply row_27
    exact row_64
    exact row_30
  -- chapter_155_line_25: GL tag implication grouping.
  have row_25 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (gl_and20 w1 N w2 N))) := by
    have auxiliary_source_1 := row_50
    have auxiliary_source_2 := row_52
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_premise
    exact ⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_premise)⟩
  -- chapter_155_line_24: GL tag implication.
  have row_24 : (gl_and20 v1 N v2 N) := by
    apply row_25
    exact row_32
  -- chapter_155_line_23: GL tag expansion.
  have row_23 : ((N v1) ∧ (N v2)) := by
    simpa only [gl_and20] using (row_24)
  -- chapter_155_line_22: GL tag disintegration.
  have row_22 : (N v1) := by
    exact row_23.1
  -- chapter_155_line_21: GL tag implication.
  have row_21 : (gl_existence1 N v1 zero add) := by
    apply row_46
    exact row_22
    exact row_71
  -- chapter_155_line_20: GL tag expansion.
  have row_20 : (¬ (∀ (v9 : α), ((N v9) → (¬ (add v1 zero v9))))) := by
    simpa only [gl_existence1] using (row_21)
  have exists_row_20 : ∃ (v9 : α), ((N v9) ∧ (add v1 zero v9)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_20
      intro v9 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v9, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v9, witness_row_20⟩ := exists_row_20
  -- chapter_155_line_19: GL tag disintegration.
  have row_19 : (add v1 zero v9) := by
    exact witness_row_20.2
  -- chapter_155_line_18: GL tag implication.
  have row_18 : (v1 = v9) := by
    apply row_44
    exact row_22
    exact row_19
  have rule_row_17 := peano_source_039 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_155_line_17: GL tag implication.
  have row_17 : (v2 = v1) := by
    apply rule_row_17
    exact row_60
    exact row_26
  -- chapter_155_line_16: GL tag equality1.
  have row_16 : (succ v9 v1) := by
    have equality_source := row_32
    have equality_step_1 := row_17
    cases equality_step_1
    have equality_step_2 := row_18
    cases equality_step_2
    exact equality_source
  -- chapter_155_line_15: GL tag equality1.
  have row_15 : (v2 = v9) := by
    have equality_source := row_17
    have equality_step_1 := row_18
    cases equality_step_1
    exact equality_source
  have rule_row_14 := peano_source_039 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_155_line_14: GL tag implication.
  have row_14 : (v1 = v2) := by
    apply rule_row_14
    exact row_26
    exact row_60
  -- chapter_155_line_13: GL tag equality1.
  have row_13 : (add v2 one v9) := by
    have equality_source := row_30
    have equality_step_1 := row_15
    cases equality_step_1
    have equality_step_2 := row_14
    cases equality_step_2
    exact equality_source
  have rule_row_12 := peano_source_065 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_155_line_12: GL tag implication.
  have row_12 : (gl_inSequence N one succ) := by
    apply rule_row_12
  -- chapter_155_line_11: GL tag expansion.
  have row_11 : (¬ (∀ (v4 : α), ((N v4) → (¬ (succ v4 one))))) := by
    simpa only [gl_inSequence] using (row_12)
  have exists_row_11 : ∃ (v4 : α), ((N v4) ∧ (succ v4 one)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_11
      intro v4 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v4, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v4, witness_row_11⟩ := exists_row_11
  -- chapter_155_line_10: GL tag disintegration.
  have row_10 : (succ v4 one) := by
    exact witness_row_11.2
  -- chapter_155_line_9: GL tag implication.
  have row_9 : (zero = v4) := by
    apply row_42
    exact row_64
    exact row_58
    exact row_10
  -- chapter_155_line_8: GL tag symmetry of equality.
  have row_8 : (v9 = v1) := by
    exact Eq.symm row_18
  -- chapter_155_line_7: GL tag equality1.
  have row_7 : (add v2 v4 v1) := by
    have equality_source := row_19
    have equality_step_1 := row_14
    cases equality_step_1
    have equality_step_2 := row_9
    cases equality_step_2
    have equality_step_3 := row_8
    cases equality_step_3
    exact equality_source
  -- chapter_155_line_6: GL tag implication.
  have row_6 : (add v2 two v1) := by
    apply row_40
    exact row_64
    exact row_39
    exact row_13
    exact row_16
  have rule_row_5 := peano_source_037 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_155_line_5: GL tag implication.
  have row_5 : (v4 = two) := by
    apply rule_row_5
    exact row_7
    exact row_6
  -- chapter_155_line_4: GL tag equality2.
  have row_4 : (zero = two) := by
    exact Eq.trans row_9 row_5
  -- chapter_155_line_3: GL tag implication.
  have row_3 : (¬ (succ one zero)) := by
    apply row_37
    exact row_64
  -- chapter_155_line_2: GL tag equality1.
  have row_2 : (¬ (succ one two)) := by
    have equality_source := row_3
    have equality_step_1 := row_4
    cases equality_step_1
    exact equality_source
  -- chapter_155_line_1: GL tag contradiction.
  have row_1 : (¬ (gl_interval N add zero v2 V2)) := by
    exact False.elim (row_2 row_39)
  exact row_1 reductio

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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (V1 : GLSet α), ((gl_interval N add zero v1 V1) → (¬ (gl_interval N add zero v2 V1)))))) := by
  intro v1
  intro v2
  intro premise_1
  intro V1
  intro premise_2
  intro reductio
  -- chapter_157_line_71: GL tag task formulation.
  have row_71 : (gl_interval N add zero v2 V1) := by
    exact reductio
  -- chapter_157_line_70: GL tag expansion.
  have row_70 : (((((gl_implication89 V1 N add zero) ∧ (gl_implication90 V1 N add v2)) ∧ (gl_implication91 N add zero v2 V1)) ∧ (N zero)) ∧ (N v2)) := by
    simpa only [gl_interval] using (row_71)
  -- chapter_157_line_69: GL tag disintegration.
  have row_69 : (N zero) := by
    exact row_70.1.2
  -- chapter_157_line_68: GL tag task formulation.
  have row_68 : (gl_interval N add zero v1 V1) := by
    exact premise_2
  -- chapter_157_line_67: GL tag expansion.
  have row_67 : (((((gl_implication89 V1 N add zero) ∧ (gl_implication90 V1 N add v1)) ∧ (gl_implication91 N add zero v1 V1)) ∧ (N zero)) ∧ (N v1)) := by
    simpa only [gl_interval] using (row_68)
  -- chapter_157_line_66: GL tag disintegration.
  have row_66 : (gl_implication90 V1 N add v1) := by
    exact row_67.1.1.1.2
  -- chapter_157_line_65: GL tag expansion.
  have row_65 : (∀ (w1 : α), ((V1 w1) → (gl_preorder N add w1 v1))) := by
    simpa only [gl_implication90] using (row_66)
  -- chapter_157_line_64: GL tag disintegration.
  have row_64 : (N v1) := by
    exact row_67.2
  -- chapter_157_line_63: GL tag task formulation.
  have row_63 : (gl_AnchorGauss N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_157_line_62: GL tag theorem.
  have row_62 := gauss_source_069 N zero succ add mul one two identity anchor relationalInduction strongRelationalInduction
  -- chapter_157_line_61: GL tag implication.
  have row_61 : (V1 v2) := by
    apply row_62
    exact row_71
  -- chapter_157_line_60: GL tag implication.
  have row_60 : (gl_preorder N add v2 v1) := by
    apply row_65
    exact row_61
  -- chapter_157_line_59: GL tag expansion.
  have row_59 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorGauss] using (row_63)
  -- chapter_157_line_58: GL tag disintegration.
  have row_58 : (succ zero one) := by
    exact row_59.1.1.2
  -- chapter_157_line_57: GL tag disintegration.
  have row_57 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_59.1.1.1
  -- chapter_157_line_56: GL tag expansion.
  have row_56 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_57)
  -- chapter_157_line_55: GL tag disintegration.
  have row_55 : (gl_fXYZ add N N N) := by
    exact row_56.1.1.1.1.1.1.1.1.1.2
  -- chapter_157_line_54: GL tag expansion.
  have row_54 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_55)
  -- chapter_157_line_53: GL tag disintegration.
  have row_53 : (gl_implication18 N N N add) := by
    exact row_54.1.2
  -- chapter_157_line_52: GL tag expansion.
  have row_52 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication18] using (row_53)
  -- chapter_157_line_51: GL tag disintegration.
  have row_51 : (gl_implication21 N zero add) := by
    exact row_56.1.1.1.1.1.1.1.1.2
  -- chapter_157_line_50: GL tag expansion.
  have row_50 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((add w1 zero w2) → (w1 = w2))))) := by
    simpa only [gl_implication21] using (row_51)
  -- chapter_157_line_49: GL tag disintegration.
  have row_49 : (gl_implication24 N succ add) := by
    exact row_56.1.1.1.1.1.2
  -- chapter_157_line_48: GL tag expansion.
  have row_48 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((succ w4 w5) → (add w3 w2 w5))))))))) := by
    simpa only [gl_implication24] using (row_49)
  -- chapter_157_line_47: GL tag disintegration.
  have row_47 : (gl_fXY succ N N) := by
    exact row_56.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_157_line_46: GL tag expansion.
  have row_46 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_47)
  -- chapter_157_line_45: GL tag disintegration.
  have row_45 : (gl_implication1 succ N) := by
    exact row_46.1.1.2
  -- chapter_157_line_44: GL tag expansion.
  have row_44 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using (row_45)
  -- chapter_157_line_43: GL tag disintegration.
  have row_43 : (gl_implication0 succ N) := by
    exact row_46.1.1.1
  -- chapter_157_line_42: GL tag expansion.
  have row_42 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_43)
  -- chapter_157_line_41: GL tag disintegration.
  have row_41 : (succ one two) := by
    exact row_59.1.2
  -- chapter_157_line_40: GL tag disintegration.
  have row_40 : (gl_implication10 N succ) := by
    exact row_56.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_157_line_39: GL tag expansion.
  have row_39 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w2 w1) → (∀ (w3 : α), ((succ w3 w1) → (w2 = w3))))))) := by
    simpa only [gl_implication10] using (row_40)
  -- chapter_157_line_38: GL tag disintegration.
  have row_38 : (gl_implication9 N zero succ) := by
    exact row_56.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_157_line_37: GL tag expansion.
  have row_37 : (∀ (w1 : α), ((N w1) → (¬ (succ w1 zero)))) := by
    simpa only [gl_implication9] using (row_38)
  -- chapter_157_line_36: GL tag expansion for integration.
  have row_36 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_157_line_35_rule: GL tag reformulation for integration and.
  have row_35_rule : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_157_line_35: GL tag reformulation for integration and.
  have row_35 : (gl_implication76 N zero succ add mul one) := by
    simpa only [gl_implication76] using row_35_rule
  -- chapter_157_line_34: GL tag expansion.
  have row_34 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    simpa only [gl_implication76] using (row_35)
  -- chapter_157_line_33: GL tag implication.
  have row_33 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_34
    exact row_57
    exact row_58
  -- chapter_157_line_32: GL tag task formulation.
  have row_32 : (succ v1 v2) := by
    exact premise_1
  have rule_row_31 := peano_source_010 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_157_line_31: GL tag implication.
  have row_31 : (add one v1 v2) := by
    apply rule_row_31
    exact row_32
  have rule_row_30 := peano_source_012 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_157_line_30: GL tag implication.
  have row_30 : (add v1 one v2) := by
    apply rule_row_30
    exact row_31
  -- chapter_157_line_29: GL tag implication.
  have row_29 : (gl_existence1 N v1 zero add) := by
    apply row_52
    exact row_64
    exact row_69
  -- chapter_157_line_28: GL tag expansion.
  have row_28 : (¬ (∀ (v9 : α), ((N v9) → (¬ (add v1 zero v9))))) := by
    simpa only [gl_existence1] using (row_29)
  have exists_row_28 : ∃ (v9 : α), ((N v9) ∧ (add v1 zero v9)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_28
      intro v9 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v9, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v9, witness_row_28⟩ := exists_row_28
  -- chapter_157_line_27: GL tag disintegration.
  have row_27 : (add v1 zero v9) := by
    exact witness_row_28.2
  -- chapter_157_line_26: GL tag implication.
  have row_26 : (v1 = v9) := by
    apply row_50
    exact row_64
    exact row_27
  -- chapter_157_line_25: GL tag implication grouping.
  have row_25 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (gl_and20 w1 N w2 N))) := by
    have auxiliary_source_1 := row_42
    have auxiliary_source_2 := row_44
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_premise
    exact ⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_premise)⟩
  -- chapter_157_line_24: GL tag implication.
  have row_24 : (gl_and20 zero N one N) := by
    apply row_25
    exact row_58
  -- chapter_157_line_23: GL tag expansion.
  have row_23 : ((N zero) ∧ (N one)) := by
    simpa only [gl_and20] using (row_24)
  -- chapter_157_line_22: GL tag disintegration.
  have row_22 : (N one) := by
    exact row_23.2
  -- chapter_157_line_21: GL tag expansion for integration.
  have row_21 : ((gl_preorder N add v1 v2) ↔ (¬ (∀ (v11 : α), ((N v11) → (¬ (add v1 v11 v2)))))) := by
    exact Iff.rfl
  -- chapter_157_line_20_rule: GL tag reformulation for integration >[bound].
  have row_20_rule : (∀ (cr20_1 : α), ((N cr20_1) → ((add v1 cr20_1 v2) → (gl_preorder N add v1 v2)))) := by
    intro cr20_1
    intro integration_premise_1
    intro integration_premise_2
    apply (row_21).2
    intro universal_counterexample
    exact universal_counterexample cr20_1 integration_premise_1 integration_premise_2
  -- chapter_157_line_20: GL tag reformulation for integration >[bound].
  have row_20 : (gl_implication87 N v1 v2 add) := by
    simpa only [gl_implication87] using row_20_rule
  -- chapter_157_line_19: GL tag expansion.
  have row_19 : (∀ (v10 : α), ((N v10) → ((add v1 v10 v2) → (gl_preorder N add v1 v2)))) := by
    simpa only [gl_implication87] using (row_20)
  -- chapter_157_line_18: GL tag implication.
  have row_18 : (gl_preorder N add v1 v2) := by
    apply row_19
    exact row_22
    exact row_30
  have rule_row_17 := peano_source_039 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_157_line_17: GL tag implication.
  have row_17 : (v2 = v1) := by
    apply rule_row_17
    exact row_60
    exact row_18
  -- chapter_157_line_16: GL tag equality1.
  have row_16 : (succ v9 v1) := by
    have equality_source := row_32
    have equality_step_1 := row_17
    cases equality_step_1
    have equality_step_2 := row_26
    cases equality_step_2
    exact equality_source
  -- chapter_157_line_15: GL tag equality1.
  have row_15 : (v2 = v9) := by
    have equality_source := row_17
    have equality_step_1 := row_26
    cases equality_step_1
    exact equality_source
  have rule_row_14 := peano_source_039 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_157_line_14: GL tag implication.
  have row_14 : (v1 = v2) := by
    apply rule_row_14
    exact row_18
    exact row_60
  -- chapter_157_line_13: GL tag equality1.
  have row_13 : (add v2 one v9) := by
    have equality_source := row_30
    have equality_step_1 := row_15
    cases equality_step_1
    have equality_step_2 := row_14
    cases equality_step_2
    exact equality_source
  -- chapter_157_line_12: GL tag implication.
  have row_12 : (add v2 two v1) := by
    apply row_48
    exact row_22
    exact row_41
    exact row_13
    exact row_16
  have rule_row_11 := peano_source_065 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_157_line_11: GL tag implication.
  have row_11 : (gl_inSequence N one succ) := by
    apply rule_row_11
  -- chapter_157_line_10: GL tag expansion.
  have row_10 : (¬ (∀ (v5 : α), ((N v5) → (¬ (succ v5 one))))) := by
    simpa only [gl_inSequence] using (row_11)
  have exists_row_10 : ∃ (v5 : α), ((N v5) ∧ (succ v5 one)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_10
      intro v5 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v5, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v5, witness_row_10⟩ := exists_row_10
  -- chapter_157_line_9: GL tag disintegration.
  have row_9 : (succ v5 one) := by
    exact witness_row_10.2
  -- chapter_157_line_8: GL tag implication.
  have row_8 : (zero = v5) := by
    apply row_39
    exact row_22
    exact row_58
    exact row_9
  -- chapter_157_line_7: GL tag symmetry of equality.
  have row_7 : (v9 = v1) := by
    exact Eq.symm row_26
  -- chapter_157_line_6: GL tag equality1.
  have row_6 : (add v2 v5 v1) := by
    have equality_source := row_27
    have equality_step_1 := row_14
    cases equality_step_1
    have equality_step_2 := row_8
    cases equality_step_2
    have equality_step_3 := row_7
    cases equality_step_3
    exact equality_source
  have rule_row_5 := peano_source_037 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_157_line_5: GL tag implication.
  have row_5 : (v5 = two) := by
    apply rule_row_5
    exact row_6
    exact row_12
  -- chapter_157_line_4: GL tag equality2.
  have row_4 : (zero = two) := by
    exact Eq.trans row_8 row_5
  -- chapter_157_line_3: GL tag implication.
  have row_3 : (¬ (succ one zero)) := by
    apply row_37
    exact row_22
  -- chapter_157_line_2: GL tag equality1.
  have row_2 : (¬ (succ one two)) := by
    have equality_source := row_3
    have equality_step_1 := row_4
    cases equality_step_1
    exact equality_source
  -- chapter_157_line_1: GL tag contradiction.
  have row_1 : (¬ (gl_interval N add zero v2 V1)) := by
    exact False.elim (row_2 row_41)
  exact row_1 reductio

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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
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
  -- chapter_161_line_76: GL tag task formulation.
  have row_76 : (gl_interval N add zero v2 V2) := by
    exact reductio
  -- chapter_161_line_75: GL tag expansion.
  have row_75 : (((((gl_implication89 V2 N add zero) ∧ (gl_implication90 V2 N add v2)) ∧ (gl_implication91 N add zero v2 V2)) ∧ (N zero)) ∧ (N v2)) := by
    simpa only [gl_interval] using (row_76)
  -- chapter_161_line_74: GL tag disintegration.
  have row_74 : (N zero) := by
    exact row_75.1.2
  -- chapter_161_line_73: GL tag task formulation.
  have row_73 : (gl_limitSet N add V1 one V2) := by
    exact premise_2
  -- chapter_161_line_72: GL tag expansion.
  have row_72 : (((gl_subset V2 V1) ∧ (gl_implication90 V2 N add one)) ∧ (gl_implication141 V1 N add one V2)) := by
    simpa only [gl_limitSet] using (row_73)
  -- chapter_161_line_71: GL tag disintegration.
  have row_71 : (gl_subset V2 V1) := by
    exact row_72.1.1
  -- chapter_161_line_70: GL tag expansion.
  have row_70 : (∀ (w1 : α), ((V2 w1) → (V1 w1))) := by
    simpa only [gl_subset] using (row_71)
  -- chapter_161_line_69: GL tag task formulation.
  have row_69 : (gl_interval N add zero v1 V1) := by
    exact premise_3
  -- chapter_161_line_68: GL tag expansion.
  have row_68 : (((((gl_implication89 V1 N add zero) ∧ (gl_implication90 V1 N add v1)) ∧ (gl_implication91 N add zero v1 V1)) ∧ (N zero)) ∧ (N v1)) := by
    simpa only [gl_interval] using (row_69)
  -- chapter_161_line_67: GL tag disintegration.
  have row_67 : (gl_implication90 V1 N add v1) := by
    exact row_68.1.1.1.2
  -- chapter_161_line_66: GL tag expansion.
  have row_66 : (∀ (w1 : α), ((V1 w1) → (gl_preorder N add w1 v1))) := by
    simpa only [gl_implication90] using (row_67)
  -- chapter_161_line_65: GL tag disintegration.
  have row_65 : (N v1) := by
    exact row_68.2
  -- chapter_161_line_64: GL tag task formulation.
  have row_64 : (gl_AnchorGauss N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_161_line_63: GL tag theorem.
  have row_63 := gauss_source_069 N zero succ add mul one two identity anchor relationalInduction strongRelationalInduction
  -- chapter_161_line_62: GL tag implication.
  have row_62 : (V2 v2) := by
    apply row_63
    exact row_76
  -- chapter_161_line_61: GL tag implication.
  have row_61 : (V1 v2) := by
    apply row_70
    exact row_62
  -- chapter_161_line_60: GL tag implication.
  have row_60 : (gl_preorder N add v2 v1) := by
    apply row_66
    exact row_61
  -- chapter_161_line_59: GL tag expansion.
  have row_59 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorGauss] using (row_64)
  -- chapter_161_line_58: GL tag disintegration.
  have row_58 : (succ zero one) := by
    exact row_59.1.1.2
  -- chapter_161_line_57: GL tag disintegration.
  have row_57 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_59.1.1.1
  -- chapter_161_line_56: GL tag expansion.
  have row_56 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_57)
  -- chapter_161_line_55: GL tag disintegration.
  have row_55 : (gl_fXYZ add N N N) := by
    exact row_56.1.1.1.1.1.1.1.1.1.2
  -- chapter_161_line_54: GL tag expansion.
  have row_54 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_55)
  -- chapter_161_line_53: GL tag disintegration.
  have row_53 : (gl_implication18 N N N add) := by
    exact row_54.1.2
  -- chapter_161_line_52: GL tag expansion.
  have row_52 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication18] using (row_53)
  -- chapter_161_line_51: GL tag disintegration.
  have row_51 : (gl_implication21 N zero add) := by
    exact row_56.1.1.1.1.1.1.1.1.2
  -- chapter_161_line_50: GL tag expansion.
  have row_50 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((add w1 zero w2) → (w1 = w2))))) := by
    simpa only [gl_implication21] using (row_51)
  -- chapter_161_line_49: GL tag disintegration.
  have row_49 : (gl_implication24 N succ add) := by
    exact row_56.1.1.1.1.1.2
  -- chapter_161_line_48: GL tag expansion.
  have row_48 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((succ w4 w5) → (add w3 w2 w5))))))))) := by
    simpa only [gl_implication24] using (row_49)
  -- chapter_161_line_47: GL tag disintegration.
  have row_47 : (gl_fXY succ N N) := by
    exact row_56.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_161_line_46: GL tag expansion.
  have row_46 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_47)
  -- chapter_161_line_45: GL tag disintegration.
  have row_45 : (gl_implication1 succ N) := by
    exact row_46.1.1.2
  -- chapter_161_line_44: GL tag expansion.
  have row_44 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using (row_45)
  -- chapter_161_line_43: GL tag disintegration.
  have row_43 : (gl_implication0 succ N) := by
    exact row_46.1.1.1
  -- chapter_161_line_42: GL tag expansion.
  have row_42 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_43)
  -- chapter_161_line_41: GL tag disintegration.
  have row_41 : (succ one two) := by
    exact row_59.1.2
  -- chapter_161_line_40: GL tag disintegration.
  have row_40 : (gl_implication10 N succ) := by
    exact row_56.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_161_line_39: GL tag expansion.
  have row_39 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w2 w1) → (∀ (w3 : α), ((succ w3 w1) → (w2 = w3))))))) := by
    simpa only [gl_implication10] using (row_40)
  -- chapter_161_line_38: GL tag disintegration.
  have row_38 : (gl_implication9 N zero succ) := by
    exact row_56.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_161_line_37: GL tag expansion.
  have row_37 : (∀ (w1 : α), ((N w1) → (¬ (succ w1 zero)))) := by
    simpa only [gl_implication9] using (row_38)
  -- chapter_161_line_36: GL tag expansion for integration.
  have row_36 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_161_line_35_rule: GL tag reformulation for integration and.
  have row_35_rule : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_161_line_35: GL tag reformulation for integration and.
  have row_35 : (gl_implication76 N zero succ add mul one) := by
    simpa only [gl_implication76] using row_35_rule
  -- chapter_161_line_34: GL tag expansion.
  have row_34 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    simpa only [gl_implication76] using (row_35)
  -- chapter_161_line_33: GL tag implication.
  have row_33 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_34
    exact row_57
    exact row_58
  -- chapter_161_line_32: GL tag task formulation.
  have row_32 : (succ v1 v2) := by
    exact premise_1
  have rule_row_31 := peano_source_010 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_161_line_31: GL tag implication.
  have row_31 : (add one v1 v2) := by
    apply rule_row_31
    exact row_32
  have rule_row_30 := peano_source_012 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_161_line_30: GL tag implication.
  have row_30 : (add v1 one v2) := by
    apply rule_row_30
    exact row_31
  -- chapter_161_line_29: GL tag implication.
  have row_29 : (gl_existence1 N v1 zero add) := by
    apply row_52
    exact row_65
    exact row_74
  -- chapter_161_line_28: GL tag expansion.
  have row_28 : (¬ (∀ (v9 : α), ((N v9) → (¬ (add v1 zero v9))))) := by
    simpa only [gl_existence1] using (row_29)
  have exists_row_28 : ∃ (v9 : α), ((N v9) ∧ (add v1 zero v9)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_28
      intro v9 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v9, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v9, witness_row_28⟩ := exists_row_28
  -- chapter_161_line_27: GL tag disintegration.
  have row_27 : (add v1 zero v9) := by
    exact witness_row_28.2
  -- chapter_161_line_26: GL tag implication.
  have row_26 : (v1 = v9) := by
    apply row_50
    exact row_65
    exact row_27
  -- chapter_161_line_25: GL tag implication grouping.
  have row_25 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (gl_and20 w1 N w2 N))) := by
    have auxiliary_source_1 := row_42
    have auxiliary_source_2 := row_44
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_premise
    exact ⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_premise)⟩
  -- chapter_161_line_24: GL tag implication.
  have row_24 : (gl_and20 zero N one N) := by
    apply row_25
    exact row_58
  -- chapter_161_line_23: GL tag expansion.
  have row_23 : ((N zero) ∧ (N one)) := by
    simpa only [gl_and20] using (row_24)
  -- chapter_161_line_22: GL tag disintegration.
  have row_22 : (N one) := by
    exact row_23.2
  -- chapter_161_line_21: GL tag expansion for integration.
  have row_21 : ((gl_preorder N add v1 v2) ↔ (¬ (∀ (v11 : α), ((N v11) → (¬ (add v1 v11 v2)))))) := by
    exact Iff.rfl
  -- chapter_161_line_20_rule: GL tag reformulation for integration >[bound].
  have row_20_rule : (∀ (cr20_1 : α), ((N cr20_1) → ((add v1 cr20_1 v2) → (gl_preorder N add v1 v2)))) := by
    intro cr20_1
    intro integration_premise_1
    intro integration_premise_2
    apply (row_21).2
    intro universal_counterexample
    exact universal_counterexample cr20_1 integration_premise_1 integration_premise_2
  -- chapter_161_line_20: GL tag reformulation for integration >[bound].
  have row_20 : (gl_implication87 N v1 v2 add) := by
    simpa only [gl_implication87] using row_20_rule
  -- chapter_161_line_19: GL tag expansion.
  have row_19 : (∀ (v10 : α), ((N v10) → ((add v1 v10 v2) → (gl_preorder N add v1 v2)))) := by
    simpa only [gl_implication87] using (row_20)
  -- chapter_161_line_18: GL tag implication.
  have row_18 : (gl_preorder N add v1 v2) := by
    apply row_19
    exact row_22
    exact row_30
  have rule_row_17 := peano_source_039 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_161_line_17: GL tag implication.
  have row_17 : (v2 = v1) := by
    apply rule_row_17
    exact row_60
    exact row_18
  -- chapter_161_line_16: GL tag equality1.
  have row_16 : (succ v9 v1) := by
    have equality_source := row_32
    have equality_step_1 := row_17
    cases equality_step_1
    have equality_step_2 := row_26
    cases equality_step_2
    exact equality_source
  -- chapter_161_line_15: GL tag equality1.
  have row_15 : (v2 = v9) := by
    have equality_source := row_17
    have equality_step_1 := row_26
    cases equality_step_1
    exact equality_source
  have rule_row_14 := peano_source_039 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_161_line_14: GL tag implication.
  have row_14 : (v1 = v2) := by
    apply rule_row_14
    exact row_18
    exact row_60
  -- chapter_161_line_13: GL tag equality1.
  have row_13 : (add v2 one v9) := by
    have equality_source := row_30
    have equality_step_1 := row_15
    cases equality_step_1
    have equality_step_2 := row_14
    cases equality_step_2
    exact equality_source
  -- chapter_161_line_12: GL tag implication.
  have row_12 : (add v2 two v1) := by
    apply row_48
    exact row_22
    exact row_41
    exact row_13
    exact row_16
  have rule_row_11 := peano_source_065 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_161_line_11: GL tag implication.
  have row_11 : (gl_inSequence N one succ) := by
    apply rule_row_11
  -- chapter_161_line_10: GL tag expansion.
  have row_10 : (¬ (∀ (v5 : α), ((N v5) → (¬ (succ v5 one))))) := by
    simpa only [gl_inSequence] using (row_11)
  have exists_row_10 : ∃ (v5 : α), ((N v5) ∧ (succ v5 one)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_10
      intro v5 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v5, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v5, witness_row_10⟩ := exists_row_10
  -- chapter_161_line_9: GL tag disintegration.
  have row_9 : (succ v5 one) := by
    exact witness_row_10.2
  -- chapter_161_line_8: GL tag implication.
  have row_8 : (zero = v5) := by
    apply row_39
    exact row_22
    exact row_58
    exact row_9
  -- chapter_161_line_7: GL tag symmetry of equality.
  have row_7 : (v9 = v1) := by
    exact Eq.symm row_26
  -- chapter_161_line_6: GL tag equality1.
  have row_6 : (add v2 v5 v1) := by
    have equality_source := row_27
    have equality_step_1 := row_14
    cases equality_step_1
    have equality_step_2 := row_8
    cases equality_step_2
    have equality_step_3 := row_7
    cases equality_step_3
    exact equality_source
  have rule_row_5 := peano_source_037 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_161_line_5: GL tag implication.
  have row_5 : (v5 = two) := by
    apply rule_row_5
    exact row_6
    exact row_12
  -- chapter_161_line_4: GL tag equality2.
  have row_4 : (zero = two) := by
    exact Eq.trans row_8 row_5
  -- chapter_161_line_3: GL tag implication.
  have row_3 : (¬ (succ one zero)) := by
    apply row_37
    exact row_22
  -- chapter_161_line_2: GL tag equality1.
  have row_2 : (¬ (succ one two)) := by
    have equality_source := row_3
    have equality_step_1 := row_4
    cases equality_step_1
    exact equality_source
  -- chapter_161_line_1: GL tag contradiction.
  have row_1 : (¬ (gl_interval N add zero v2 V2)) := by
    exact False.elim (row_2 row_41)
  exact row_1 reductio

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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
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
  -- chapter_163_line_76: GL tag task formulation.
  have row_76 : (gl_interval N add zero v2 V2) := by
    exact reductio
  -- chapter_163_line_75: GL tag expansion.
  have row_75 : (((((gl_implication89 V2 N add zero) ∧ (gl_implication90 V2 N add v2)) ∧ (gl_implication91 N add zero v2 V2)) ∧ (N zero)) ∧ (N v2)) := by
    simpa only [gl_interval] using (row_76)
  -- chapter_163_line_74: GL tag disintegration.
  have row_74 : (N zero) := by
    exact row_75.1.2
  -- chapter_163_line_73: GL tag task formulation.
  have row_73 : (gl_limitSet N add V1 two V2) := by
    exact premise_2
  -- chapter_163_line_72: GL tag expansion.
  have row_72 : (((gl_subset V2 V1) ∧ (gl_implication90 V2 N add two)) ∧ (gl_implication141 V1 N add two V2)) := by
    simpa only [gl_limitSet] using (row_73)
  -- chapter_163_line_71: GL tag disintegration.
  have row_71 : (gl_subset V2 V1) := by
    exact row_72.1.1
  -- chapter_163_line_70: GL tag expansion.
  have row_70 : (∀ (w1 : α), ((V2 w1) → (V1 w1))) := by
    simpa only [gl_subset] using (row_71)
  -- chapter_163_line_69: GL tag task formulation.
  have row_69 : (gl_interval N add zero v1 V1) := by
    exact premise_3
  -- chapter_163_line_68: GL tag expansion.
  have row_68 : (((((gl_implication89 V1 N add zero) ∧ (gl_implication90 V1 N add v1)) ∧ (gl_implication91 N add zero v1 V1)) ∧ (N zero)) ∧ (N v1)) := by
    simpa only [gl_interval] using (row_69)
  -- chapter_163_line_67: GL tag disintegration.
  have row_67 : (gl_implication90 V1 N add v1) := by
    exact row_68.1.1.1.2
  -- chapter_163_line_66: GL tag expansion.
  have row_66 : (∀ (w1 : α), ((V1 w1) → (gl_preorder N add w1 v1))) := by
    simpa only [gl_implication90] using (row_67)
  -- chapter_163_line_65: GL tag disintegration.
  have row_65 : (N v1) := by
    exact row_68.2
  -- chapter_163_line_64: GL tag task formulation.
  have row_64 : (gl_AnchorGauss N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_163_line_63: GL tag theorem.
  have row_63 := gauss_source_069 N zero succ add mul one two identity anchor relationalInduction strongRelationalInduction
  -- chapter_163_line_62: GL tag implication.
  have row_62 : (V2 v2) := by
    apply row_63
    exact row_76
  -- chapter_163_line_61: GL tag implication.
  have row_61 : (V1 v2) := by
    apply row_70
    exact row_62
  -- chapter_163_line_60: GL tag implication.
  have row_60 : (gl_preorder N add v2 v1) := by
    apply row_66
    exact row_61
  -- chapter_163_line_59: GL tag expansion.
  have row_59 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorGauss] using (row_64)
  -- chapter_163_line_58: GL tag disintegration.
  have row_58 : (succ zero one) := by
    exact row_59.1.1.2
  -- chapter_163_line_57: GL tag disintegration.
  have row_57 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_59.1.1.1
  -- chapter_163_line_56: GL tag expansion.
  have row_56 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_57)
  -- chapter_163_line_55: GL tag disintegration.
  have row_55 : (gl_fXYZ add N N N) := by
    exact row_56.1.1.1.1.1.1.1.1.1.2
  -- chapter_163_line_54: GL tag expansion.
  have row_54 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_55)
  -- chapter_163_line_53: GL tag disintegration.
  have row_53 : (gl_implication18 N N N add) := by
    exact row_54.1.2
  -- chapter_163_line_52: GL tag expansion.
  have row_52 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication18] using (row_53)
  -- chapter_163_line_51: GL tag disintegration.
  have row_51 : (gl_implication21 N zero add) := by
    exact row_56.1.1.1.1.1.1.1.1.2
  -- chapter_163_line_50: GL tag expansion.
  have row_50 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((add w1 zero w2) → (w1 = w2))))) := by
    simpa only [gl_implication21] using (row_51)
  -- chapter_163_line_49: GL tag disintegration.
  have row_49 : (gl_implication24 N succ add) := by
    exact row_56.1.1.1.1.1.2
  -- chapter_163_line_48: GL tag expansion.
  have row_48 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((succ w4 w5) → (add w3 w2 w5))))))))) := by
    simpa only [gl_implication24] using (row_49)
  -- chapter_163_line_47: GL tag disintegration.
  have row_47 : (gl_fXY succ N N) := by
    exact row_56.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_163_line_46: GL tag expansion.
  have row_46 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_47)
  -- chapter_163_line_45: GL tag disintegration.
  have row_45 : (gl_implication1 succ N) := by
    exact row_46.1.1.2
  -- chapter_163_line_44: GL tag expansion.
  have row_44 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using (row_45)
  -- chapter_163_line_43: GL tag disintegration.
  have row_43 : (gl_implication0 succ N) := by
    exact row_46.1.1.1
  -- chapter_163_line_42: GL tag expansion.
  have row_42 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_43)
  -- chapter_163_line_41: GL tag disintegration.
  have row_41 : (succ one two) := by
    exact row_59.1.2
  -- chapter_163_line_40: GL tag disintegration.
  have row_40 : (gl_implication10 N succ) := by
    exact row_56.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_163_line_39: GL tag expansion.
  have row_39 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w2 w1) → (∀ (w3 : α), ((succ w3 w1) → (w2 = w3))))))) := by
    simpa only [gl_implication10] using (row_40)
  -- chapter_163_line_38: GL tag disintegration.
  have row_38 : (gl_implication9 N zero succ) := by
    exact row_56.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_163_line_37: GL tag expansion.
  have row_37 : (∀ (w1 : α), ((N w1) → (¬ (succ w1 zero)))) := by
    simpa only [gl_implication9] using (row_38)
  -- chapter_163_line_36: GL tag expansion for integration.
  have row_36 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_163_line_35_rule: GL tag reformulation for integration and.
  have row_35_rule : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_163_line_35: GL tag reformulation for integration and.
  have row_35 : (gl_implication76 N zero succ add mul one) := by
    simpa only [gl_implication76] using row_35_rule
  -- chapter_163_line_34: GL tag expansion.
  have row_34 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    simpa only [gl_implication76] using (row_35)
  -- chapter_163_line_33: GL tag implication.
  have row_33 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_34
    exact row_57
    exact row_58
  -- chapter_163_line_32: GL tag task formulation.
  have row_32 : (succ v1 v2) := by
    exact premise_1
  have rule_row_31 := peano_source_010 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_163_line_31: GL tag implication.
  have row_31 : (add one v1 v2) := by
    apply rule_row_31
    exact row_32
  have rule_row_30 := peano_source_012 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_163_line_30: GL tag implication.
  have row_30 : (add v1 one v2) := by
    apply rule_row_30
    exact row_31
  -- chapter_163_line_29: GL tag implication.
  have row_29 : (gl_existence1 N v1 zero add) := by
    apply row_52
    exact row_65
    exact row_74
  -- chapter_163_line_28: GL tag expansion.
  have row_28 : (¬ (∀ (v9 : α), ((N v9) → (¬ (add v1 zero v9))))) := by
    simpa only [gl_existence1] using (row_29)
  have exists_row_28 : ∃ (v9 : α), ((N v9) ∧ (add v1 zero v9)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_28
      intro v9 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v9, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v9, witness_row_28⟩ := exists_row_28
  -- chapter_163_line_27: GL tag disintegration.
  have row_27 : (add v1 zero v9) := by
    exact witness_row_28.2
  -- chapter_163_line_26: GL tag implication.
  have row_26 : (v1 = v9) := by
    apply row_50
    exact row_65
    exact row_27
  -- chapter_163_line_25: GL tag implication grouping.
  have row_25 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (gl_and20 w1 N w2 N))) := by
    have auxiliary_source_1 := row_42
    have auxiliary_source_2 := row_44
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_premise
    exact ⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_premise)⟩
  -- chapter_163_line_24: GL tag implication.
  have row_24 : (gl_and20 zero N one N) := by
    apply row_25
    exact row_58
  -- chapter_163_line_23: GL tag expansion.
  have row_23 : ((N zero) ∧ (N one)) := by
    simpa only [gl_and20] using (row_24)
  -- chapter_163_line_22: GL tag disintegration.
  have row_22 : (N one) := by
    exact row_23.2
  -- chapter_163_line_21: GL tag expansion for integration.
  have row_21 : ((gl_preorder N add v1 v2) ↔ (¬ (∀ (v11 : α), ((N v11) → (¬ (add v1 v11 v2)))))) := by
    exact Iff.rfl
  -- chapter_163_line_20_rule: GL tag reformulation for integration >[bound].
  have row_20_rule : (∀ (cr20_1 : α), ((N cr20_1) → ((add v1 cr20_1 v2) → (gl_preorder N add v1 v2)))) := by
    intro cr20_1
    intro integration_premise_1
    intro integration_premise_2
    apply (row_21).2
    intro universal_counterexample
    exact universal_counterexample cr20_1 integration_premise_1 integration_premise_2
  -- chapter_163_line_20: GL tag reformulation for integration >[bound].
  have row_20 : (gl_implication87 N v1 v2 add) := by
    simpa only [gl_implication87] using row_20_rule
  -- chapter_163_line_19: GL tag expansion.
  have row_19 : (∀ (v10 : α), ((N v10) → ((add v1 v10 v2) → (gl_preorder N add v1 v2)))) := by
    simpa only [gl_implication87] using (row_20)
  -- chapter_163_line_18: GL tag implication.
  have row_18 : (gl_preorder N add v1 v2) := by
    apply row_19
    exact row_22
    exact row_30
  have rule_row_17 := peano_source_039 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_163_line_17: GL tag implication.
  have row_17 : (v2 = v1) := by
    apply rule_row_17
    exact row_60
    exact row_18
  -- chapter_163_line_16: GL tag equality1.
  have row_16 : (succ v9 v1) := by
    have equality_source := row_32
    have equality_step_1 := row_17
    cases equality_step_1
    have equality_step_2 := row_26
    cases equality_step_2
    exact equality_source
  -- chapter_163_line_15: GL tag equality1.
  have row_15 : (v2 = v9) := by
    have equality_source := row_17
    have equality_step_1 := row_26
    cases equality_step_1
    exact equality_source
  have rule_row_14 := peano_source_039 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_163_line_14: GL tag implication.
  have row_14 : (v1 = v2) := by
    apply rule_row_14
    exact row_18
    exact row_60
  -- chapter_163_line_13: GL tag equality1.
  have row_13 : (add v2 one v9) := by
    have equality_source := row_30
    have equality_step_1 := row_15
    cases equality_step_1
    have equality_step_2 := row_14
    cases equality_step_2
    exact equality_source
  -- chapter_163_line_12: GL tag implication.
  have row_12 : (add v2 two v1) := by
    apply row_48
    exact row_22
    exact row_41
    exact row_13
    exact row_16
  have rule_row_11 := peano_source_065 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_163_line_11: GL tag implication.
  have row_11 : (gl_inSequence N one succ) := by
    apply rule_row_11
  -- chapter_163_line_10: GL tag expansion.
  have row_10 : (¬ (∀ (v5 : α), ((N v5) → (¬ (succ v5 one))))) := by
    simpa only [gl_inSequence] using (row_11)
  have exists_row_10 : ∃ (v5 : α), ((N v5) ∧ (succ v5 one)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_10
      intro v5 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v5, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v5, witness_row_10⟩ := exists_row_10
  -- chapter_163_line_9: GL tag disintegration.
  have row_9 : (succ v5 one) := by
    exact witness_row_10.2
  -- chapter_163_line_8: GL tag implication.
  have row_8 : (zero = v5) := by
    apply row_39
    exact row_22
    exact row_58
    exact row_9
  -- chapter_163_line_7: GL tag symmetry of equality.
  have row_7 : (v9 = v1) := by
    exact Eq.symm row_26
  -- chapter_163_line_6: GL tag equality1.
  have row_6 : (add v2 v5 v1) := by
    have equality_source := row_27
    have equality_step_1 := row_14
    cases equality_step_1
    have equality_step_2 := row_8
    cases equality_step_2
    have equality_step_3 := row_7
    cases equality_step_3
    exact equality_source
  have rule_row_5 := peano_source_037 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_163_line_5: GL tag implication.
  have row_5 : (v5 = two) := by
    apply rule_row_5
    exact row_6
    exact row_12
  -- chapter_163_line_4: GL tag equality2.
  have row_4 : (zero = two) := by
    exact Eq.trans row_8 row_5
  -- chapter_163_line_3: GL tag implication.
  have row_3 : (¬ (succ one zero)) := by
    apply row_37
    exact row_22
  -- chapter_163_line_2: GL tag equality1.
  have row_2 : (¬ (succ one two)) := by
    have equality_source := row_3
    have equality_step_1 := row_4
    cases equality_step_1
    exact equality_source
  -- chapter_163_line_1: GL tag contradiction.
  have row_1 : (¬ (gl_interval N add zero v2 V2)) := by
    exact False.elim (row_2 row_41)
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
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
  -- chapter_165_line_73: GL tag task formulation.
  have row_73 : (gl_interval N add zero v1 V1) := by
    exact premise_3
  -- chapter_165_line_72: GL tag expansion.
  have row_72 : (((((gl_implication89 V1 N add zero) ∧ (gl_implication90 V1 N add v1)) ∧ (gl_implication91 N add zero v1 V1)) ∧ (N zero)) ∧ (N v1)) := by
    simpa only [gl_interval] using (row_73)
  -- chapter_165_line_71: GL tag disintegration.
  have row_71 : (N v1) := by
    exact row_72.2
  -- chapter_165_line_70: GL tag task formulation.
  have row_70 : (gl_interval N add zero v2 V2) := by
    exact reductio
  -- chapter_165_line_69: GL tag expansion.
  have row_69 : (((((gl_implication89 V2 N add zero) ∧ (gl_implication90 V2 N add v2)) ∧ (gl_implication91 N add zero v2 V2)) ∧ (N zero)) ∧ (N v2)) := by
    simpa only [gl_interval] using (row_70)
  -- chapter_165_line_68: GL tag disintegration.
  have row_68 : (N zero) := by
    exact row_69.1.2
  -- chapter_165_line_67: GL tag task formulation.
  have row_67 : (gl_limitSet N add V1 v1 V2) := by
    exact premise_2
  -- chapter_165_line_66: GL tag expansion.
  have row_66 : (((gl_subset V2 V1) ∧ (gl_implication90 V2 N add v1)) ∧ (gl_implication141 V1 N add v1 V2)) := by
    simpa only [gl_limitSet] using (row_67)
  -- chapter_165_line_65: GL tag disintegration.
  have row_65 : (gl_implication90 V2 N add v1) := by
    exact row_66.1.2
  -- chapter_165_line_64: GL tag expansion.
  have row_64 : (∀ (w1 : α), ((V2 w1) → (gl_preorder N add w1 v1))) := by
    simpa only [gl_implication90] using (row_65)
  -- chapter_165_line_63: GL tag task formulation.
  have row_63 : (gl_AnchorGauss N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_165_line_62: GL tag theorem.
  have row_62 := gauss_source_069 N zero succ add mul one two identity anchor relationalInduction strongRelationalInduction
  -- chapter_165_line_61: GL tag implication.
  have row_61 : (V2 v2) := by
    apply row_62
    exact row_70
  -- chapter_165_line_60: GL tag implication.
  have row_60 : (gl_preorder N add v2 v1) := by
    apply row_64
    exact row_61
  -- chapter_165_line_59: GL tag expansion.
  have row_59 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorGauss] using (row_63)
  -- chapter_165_line_58: GL tag disintegration.
  have row_58 : (succ zero one) := by
    exact row_59.1.1.2
  -- chapter_165_line_57: GL tag disintegration.
  have row_57 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_59.1.1.1
  -- chapter_165_line_56: GL tag expansion.
  have row_56 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_57)
  -- chapter_165_line_55: GL tag disintegration.
  have row_55 : (gl_fXYZ add N N N) := by
    exact row_56.1.1.1.1.1.1.1.1.1.2
  -- chapter_165_line_54: GL tag expansion.
  have row_54 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_55)
  -- chapter_165_line_53: GL tag disintegration.
  have row_53 : (gl_implication18 N N N add) := by
    exact row_54.1.2
  -- chapter_165_line_52: GL tag expansion.
  have row_52 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication18] using (row_53)
  -- chapter_165_line_51: GL tag disintegration.
  have row_51 : (gl_implication21 N zero add) := by
    exact row_56.1.1.1.1.1.1.1.1.2
  -- chapter_165_line_50: GL tag expansion.
  have row_50 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((add w1 zero w2) → (w1 = w2))))) := by
    simpa only [gl_implication21] using (row_51)
  -- chapter_165_line_49: GL tag disintegration.
  have row_49 : (gl_implication24 N succ add) := by
    exact row_56.1.1.1.1.1.2
  -- chapter_165_line_48: GL tag expansion.
  have row_48 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((succ w4 w5) → (add w3 w2 w5))))))))) := by
    simpa only [gl_implication24] using (row_49)
  -- chapter_165_line_47: GL tag disintegration.
  have row_47 : (gl_fXY succ N N) := by
    exact row_56.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_165_line_46: GL tag expansion.
  have row_46 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_47)
  -- chapter_165_line_45: GL tag disintegration.
  have row_45 : (gl_implication1 succ N) := by
    exact row_46.1.1.2
  -- chapter_165_line_44: GL tag expansion.
  have row_44 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using (row_45)
  -- chapter_165_line_43: GL tag disintegration.
  have row_43 : (gl_implication0 succ N) := by
    exact row_46.1.1.1
  -- chapter_165_line_42: GL tag expansion.
  have row_42 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_43)
  -- chapter_165_line_41: GL tag disintegration.
  have row_41 : (succ one two) := by
    exact row_59.1.2
  -- chapter_165_line_40: GL tag disintegration.
  have row_40 : (gl_implication10 N succ) := by
    exact row_56.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_165_line_39: GL tag expansion.
  have row_39 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w2 w1) → (∀ (w3 : α), ((succ w3 w1) → (w2 = w3))))))) := by
    simpa only [gl_implication10] using (row_40)
  -- chapter_165_line_38: GL tag disintegration.
  have row_38 : (gl_implication9 N zero succ) := by
    exact row_56.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_165_line_37: GL tag expansion.
  have row_37 : (∀ (w1 : α), ((N w1) → (¬ (succ w1 zero)))) := by
    simpa only [gl_implication9] using (row_38)
  -- chapter_165_line_36: GL tag expansion for integration.
  have row_36 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_165_line_35_rule: GL tag reformulation for integration and.
  have row_35_rule : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_165_line_35: GL tag reformulation for integration and.
  have row_35 : (gl_implication76 N zero succ add mul one) := by
    simpa only [gl_implication76] using row_35_rule
  -- chapter_165_line_34: GL tag expansion.
  have row_34 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    simpa only [gl_implication76] using (row_35)
  -- chapter_165_line_33: GL tag implication.
  have row_33 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_34
    exact row_57
    exact row_58
  -- chapter_165_line_32: GL tag task formulation.
  have row_32 : (succ v1 v2) := by
    exact premise_1
  have rule_row_31 := peano_source_010 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_165_line_31: GL tag implication.
  have row_31 : (add one v1 v2) := by
    apply rule_row_31
    exact row_32
  have rule_row_30 := peano_source_012 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_165_line_30: GL tag implication.
  have row_30 : (add v1 one v2) := by
    apply rule_row_30
    exact row_31
  -- chapter_165_line_29: GL tag implication.
  have row_29 : (gl_existence1 N v1 zero add) := by
    apply row_52
    exact row_71
    exact row_68
  -- chapter_165_line_28: GL tag expansion.
  have row_28 : (¬ (∀ (v9 : α), ((N v9) → (¬ (add v1 zero v9))))) := by
    simpa only [gl_existence1] using (row_29)
  have exists_row_28 : ∃ (v9 : α), ((N v9) ∧ (add v1 zero v9)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_28
      intro v9 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v9, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v9, witness_row_28⟩ := exists_row_28
  -- chapter_165_line_27: GL tag disintegration.
  have row_27 : (add v1 zero v9) := by
    exact witness_row_28.2
  -- chapter_165_line_26: GL tag implication.
  have row_26 : (v1 = v9) := by
    apply row_50
    exact row_71
    exact row_27
  -- chapter_165_line_25: GL tag implication grouping.
  have row_25 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (gl_and20 w1 N w2 N))) := by
    have auxiliary_source_1 := row_42
    have auxiliary_source_2 := row_44
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_premise
    exact ⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_premise)⟩
  -- chapter_165_line_24: GL tag implication.
  have row_24 : (gl_and20 zero N one N) := by
    apply row_25
    exact row_58
  -- chapter_165_line_23: GL tag expansion.
  have row_23 : ((N zero) ∧ (N one)) := by
    simpa only [gl_and20] using (row_24)
  -- chapter_165_line_22: GL tag disintegration.
  have row_22 : (N one) := by
    exact row_23.2
  -- chapter_165_line_21: GL tag expansion for integration.
  have row_21 : ((gl_preorder N add v1 v2) ↔ (¬ (∀ (v11 : α), ((N v11) → (¬ (add v1 v11 v2)))))) := by
    exact Iff.rfl
  -- chapter_165_line_20_rule: GL tag reformulation for integration >[bound].
  have row_20_rule : (∀ (cr20_1 : α), ((N cr20_1) → ((add v1 cr20_1 v2) → (gl_preorder N add v1 v2)))) := by
    intro cr20_1
    intro integration_premise_1
    intro integration_premise_2
    apply (row_21).2
    intro universal_counterexample
    exact universal_counterexample cr20_1 integration_premise_1 integration_premise_2
  -- chapter_165_line_20: GL tag reformulation for integration >[bound].
  have row_20 : (gl_implication87 N v1 v2 add) := by
    simpa only [gl_implication87] using row_20_rule
  -- chapter_165_line_19: GL tag expansion.
  have row_19 : (∀ (v10 : α), ((N v10) → ((add v1 v10 v2) → (gl_preorder N add v1 v2)))) := by
    simpa only [gl_implication87] using (row_20)
  -- chapter_165_line_18: GL tag implication.
  have row_18 : (gl_preorder N add v1 v2) := by
    apply row_19
    exact row_22
    exact row_30
  have rule_row_17 := peano_source_039 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_165_line_17: GL tag implication.
  have row_17 : (v2 = v1) := by
    apply rule_row_17
    exact row_60
    exact row_18
  -- chapter_165_line_16: GL tag equality1.
  have row_16 : (succ v9 v1) := by
    have equality_source := row_32
    have equality_step_1 := row_17
    cases equality_step_1
    have equality_step_2 := row_26
    cases equality_step_2
    exact equality_source
  -- chapter_165_line_15: GL tag equality1.
  have row_15 : (v2 = v9) := by
    have equality_source := row_17
    have equality_step_1 := row_26
    cases equality_step_1
    exact equality_source
  have rule_row_14 := peano_source_039 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_165_line_14: GL tag implication.
  have row_14 : (v1 = v2) := by
    apply rule_row_14
    exact row_18
    exact row_60
  -- chapter_165_line_13: GL tag equality1.
  have row_13 : (add v2 one v9) := by
    have equality_source := row_30
    have equality_step_1 := row_15
    cases equality_step_1
    have equality_step_2 := row_14
    cases equality_step_2
    exact equality_source
  -- chapter_165_line_12: GL tag implication.
  have row_12 : (add v2 two v1) := by
    apply row_48
    exact row_22
    exact row_41
    exact row_13
    exact row_16
  have rule_row_11 := peano_source_065 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_165_line_11: GL tag implication.
  have row_11 : (gl_inSequence N one succ) := by
    apply rule_row_11
  -- chapter_165_line_10: GL tag expansion.
  have row_10 : (¬ (∀ (v5 : α), ((N v5) → (¬ (succ v5 one))))) := by
    simpa only [gl_inSequence] using (row_11)
  have exists_row_10 : ∃ (v5 : α), ((N v5) ∧ (succ v5 one)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_10
      intro v5 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v5, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v5, witness_row_10⟩ := exists_row_10
  -- chapter_165_line_9: GL tag disintegration.
  have row_9 : (succ v5 one) := by
    exact witness_row_10.2
  -- chapter_165_line_8: GL tag implication.
  have row_8 : (zero = v5) := by
    apply row_39
    exact row_22
    exact row_58
    exact row_9
  -- chapter_165_line_7: GL tag symmetry of equality.
  have row_7 : (v9 = v1) := by
    exact Eq.symm row_26
  -- chapter_165_line_6: GL tag equality1.
  have row_6 : (add v2 v5 v1) := by
    have equality_source := row_27
    have equality_step_1 := row_14
    cases equality_step_1
    have equality_step_2 := row_8
    cases equality_step_2
    have equality_step_3 := row_7
    cases equality_step_3
    exact equality_source
  have rule_row_5 := peano_source_037 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_165_line_5: GL tag implication.
  have row_5 : (v5 = two) := by
    apply rule_row_5
    exact row_6
    exact row_12
  -- chapter_165_line_4: GL tag equality2.
  have row_4 : (zero = two) := by
    exact Eq.trans row_8 row_5
  -- chapter_165_line_3: GL tag implication.
  have row_3 : (¬ (succ one zero)) := by
    apply row_37
    exact row_22
  -- chapter_165_line_2: GL tag equality1.
  have row_2 : (¬ (succ one two)) := by
    have equality_source := row_3
    have equality_step_1 := row_4
    cases equality_step_1
    exact equality_source
  -- chapter_165_line_1: GL tag contradiction.
  have row_1 : (¬ (gl_interval N add zero v2 V2)) := by
    exact False.elim (row_2 row_41)
  exact row_1 reductio

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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
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
  -- chapter_167_line_76: GL tag task formulation.
  have row_76 : (gl_interval N add zero v2 V2) := by
    exact reductio
  -- chapter_167_line_75: GL tag expansion.
  have row_75 : (((((gl_implication89 V2 N add zero) ∧ (gl_implication90 V2 N add v2)) ∧ (gl_implication91 N add zero v2 V2)) ∧ (N zero)) ∧ (N v2)) := by
    simpa only [gl_interval] using (row_76)
  -- chapter_167_line_74: GL tag disintegration.
  have row_74 : (N zero) := by
    exact row_75.1.2
  -- chapter_167_line_73: GL tag task formulation.
  have row_73 : (gl_limitSet N add V1 v2 V2) := by
    exact premise_2
  -- chapter_167_line_72: GL tag expansion.
  have row_72 : (((gl_subset V2 V1) ∧ (gl_implication90 V2 N add v2)) ∧ (gl_implication141 V1 N add v2 V2)) := by
    simpa only [gl_limitSet] using (row_73)
  -- chapter_167_line_71: GL tag disintegration.
  have row_71 : (gl_subset V2 V1) := by
    exact row_72.1.1
  -- chapter_167_line_70: GL tag expansion.
  have row_70 : (∀ (w1 : α), ((V2 w1) → (V1 w1))) := by
    simpa only [gl_subset] using (row_71)
  -- chapter_167_line_69: GL tag task formulation.
  have row_69 : (gl_interval N add zero v1 V1) := by
    exact premise_3
  -- chapter_167_line_68: GL tag expansion.
  have row_68 : (((((gl_implication89 V1 N add zero) ∧ (gl_implication90 V1 N add v1)) ∧ (gl_implication91 N add zero v1 V1)) ∧ (N zero)) ∧ (N v1)) := by
    simpa only [gl_interval] using (row_69)
  -- chapter_167_line_67: GL tag disintegration.
  have row_67 : (gl_implication90 V1 N add v1) := by
    exact row_68.1.1.1.2
  -- chapter_167_line_66: GL tag expansion.
  have row_66 : (∀ (w1 : α), ((V1 w1) → (gl_preorder N add w1 v1))) := by
    simpa only [gl_implication90] using (row_67)
  -- chapter_167_line_65: GL tag disintegration.
  have row_65 : (N v1) := by
    exact row_68.2
  -- chapter_167_line_64: GL tag task formulation.
  have row_64 : (gl_AnchorGauss N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_167_line_63: GL tag theorem.
  have row_63 := gauss_source_069 N zero succ add mul one two identity anchor relationalInduction strongRelationalInduction
  -- chapter_167_line_62: GL tag implication.
  have row_62 : (V2 v2) := by
    apply row_63
    exact row_76
  -- chapter_167_line_61: GL tag implication.
  have row_61 : (V1 v2) := by
    apply row_70
    exact row_62
  -- chapter_167_line_60: GL tag implication.
  have row_60 : (gl_preorder N add v2 v1) := by
    apply row_66
    exact row_61
  -- chapter_167_line_59: GL tag expansion.
  have row_59 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorGauss] using (row_64)
  -- chapter_167_line_58: GL tag disintegration.
  have row_58 : (succ zero one) := by
    exact row_59.1.1.2
  -- chapter_167_line_57: GL tag disintegration.
  have row_57 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_59.1.1.1
  -- chapter_167_line_56: GL tag expansion.
  have row_56 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_57)
  -- chapter_167_line_55: GL tag disintegration.
  have row_55 : (gl_fXYZ add N N N) := by
    exact row_56.1.1.1.1.1.1.1.1.1.2
  -- chapter_167_line_54: GL tag expansion.
  have row_54 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_55)
  -- chapter_167_line_53: GL tag disintegration.
  have row_53 : (gl_implication18 N N N add) := by
    exact row_54.1.2
  -- chapter_167_line_52: GL tag expansion.
  have row_52 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication18] using (row_53)
  -- chapter_167_line_51: GL tag disintegration.
  have row_51 : (gl_implication21 N zero add) := by
    exact row_56.1.1.1.1.1.1.1.1.2
  -- chapter_167_line_50: GL tag expansion.
  have row_50 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((add w1 zero w2) → (w1 = w2))))) := by
    simpa only [gl_implication21] using (row_51)
  -- chapter_167_line_49: GL tag disintegration.
  have row_49 : (gl_implication24 N succ add) := by
    exact row_56.1.1.1.1.1.2
  -- chapter_167_line_48: GL tag expansion.
  have row_48 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((succ w4 w5) → (add w3 w2 w5))))))))) := by
    simpa only [gl_implication24] using (row_49)
  -- chapter_167_line_47: GL tag disintegration.
  have row_47 : (gl_fXY succ N N) := by
    exact row_56.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_167_line_46: GL tag expansion.
  have row_46 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_47)
  -- chapter_167_line_45: GL tag disintegration.
  have row_45 : (gl_implication1 succ N) := by
    exact row_46.1.1.2
  -- chapter_167_line_44: GL tag expansion.
  have row_44 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using (row_45)
  -- chapter_167_line_43: GL tag disintegration.
  have row_43 : (gl_implication0 succ N) := by
    exact row_46.1.1.1
  -- chapter_167_line_42: GL tag expansion.
  have row_42 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_43)
  -- chapter_167_line_41: GL tag disintegration.
  have row_41 : (succ one two) := by
    exact row_59.1.2
  -- chapter_167_line_40: GL tag disintegration.
  have row_40 : (gl_implication10 N succ) := by
    exact row_56.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_167_line_39: GL tag expansion.
  have row_39 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w2 w1) → (∀ (w3 : α), ((succ w3 w1) → (w2 = w3))))))) := by
    simpa only [gl_implication10] using (row_40)
  -- chapter_167_line_38: GL tag disintegration.
  have row_38 : (gl_implication9 N zero succ) := by
    exact row_56.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_167_line_37: GL tag expansion.
  have row_37 : (∀ (w1 : α), ((N w1) → (¬ (succ w1 zero)))) := by
    simpa only [gl_implication9] using (row_38)
  -- chapter_167_line_36: GL tag expansion for integration.
  have row_36 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_167_line_35_rule: GL tag reformulation for integration and.
  have row_35_rule : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_167_line_35: GL tag reformulation for integration and.
  have row_35 : (gl_implication76 N zero succ add mul one) := by
    simpa only [gl_implication76] using row_35_rule
  -- chapter_167_line_34: GL tag expansion.
  have row_34 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    simpa only [gl_implication76] using (row_35)
  -- chapter_167_line_33: GL tag implication.
  have row_33 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_34
    exact row_57
    exact row_58
  -- chapter_167_line_32: GL tag task formulation.
  have row_32 : (succ v1 v2) := by
    exact premise_1
  have rule_row_31 := peano_source_010 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_167_line_31: GL tag implication.
  have row_31 : (add one v1 v2) := by
    apply rule_row_31
    exact row_32
  have rule_row_30 := peano_source_012 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_167_line_30: GL tag implication.
  have row_30 : (add v1 one v2) := by
    apply rule_row_30
    exact row_31
  -- chapter_167_line_29: GL tag implication.
  have row_29 : (gl_existence1 N v1 zero add) := by
    apply row_52
    exact row_65
    exact row_74
  -- chapter_167_line_28: GL tag expansion.
  have row_28 : (¬ (∀ (v9 : α), ((N v9) → (¬ (add v1 zero v9))))) := by
    simpa only [gl_existence1] using (row_29)
  have exists_row_28 : ∃ (v9 : α), ((N v9) ∧ (add v1 zero v9)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_28
      intro v9 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v9, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v9, witness_row_28⟩ := exists_row_28
  -- chapter_167_line_27: GL tag disintegration.
  have row_27 : (add v1 zero v9) := by
    exact witness_row_28.2
  -- chapter_167_line_26: GL tag implication.
  have row_26 : (v1 = v9) := by
    apply row_50
    exact row_65
    exact row_27
  -- chapter_167_line_25: GL tag implication grouping.
  have row_25 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (gl_and20 w1 N w2 N))) := by
    have auxiliary_source_1 := row_42
    have auxiliary_source_2 := row_44
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_premise
    exact ⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_premise)⟩
  -- chapter_167_line_24: GL tag implication.
  have row_24 : (gl_and20 zero N one N) := by
    apply row_25
    exact row_58
  -- chapter_167_line_23: GL tag expansion.
  have row_23 : ((N zero) ∧ (N one)) := by
    simpa only [gl_and20] using (row_24)
  -- chapter_167_line_22: GL tag disintegration.
  have row_22 : (N one) := by
    exact row_23.2
  -- chapter_167_line_21: GL tag expansion for integration.
  have row_21 : ((gl_preorder N add v1 v2) ↔ (¬ (∀ (v11 : α), ((N v11) → (¬ (add v1 v11 v2)))))) := by
    exact Iff.rfl
  -- chapter_167_line_20_rule: GL tag reformulation for integration >[bound].
  have row_20_rule : (∀ (cr20_1 : α), ((N cr20_1) → ((add v1 cr20_1 v2) → (gl_preorder N add v1 v2)))) := by
    intro cr20_1
    intro integration_premise_1
    intro integration_premise_2
    apply (row_21).2
    intro universal_counterexample
    exact universal_counterexample cr20_1 integration_premise_1 integration_premise_2
  -- chapter_167_line_20: GL tag reformulation for integration >[bound].
  have row_20 : (gl_implication87 N v1 v2 add) := by
    simpa only [gl_implication87] using row_20_rule
  -- chapter_167_line_19: GL tag expansion.
  have row_19 : (∀ (v10 : α), ((N v10) → ((add v1 v10 v2) → (gl_preorder N add v1 v2)))) := by
    simpa only [gl_implication87] using (row_20)
  -- chapter_167_line_18: GL tag implication.
  have row_18 : (gl_preorder N add v1 v2) := by
    apply row_19
    exact row_22
    exact row_30
  have rule_row_17 := peano_source_039 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_167_line_17: GL tag implication.
  have row_17 : (v2 = v1) := by
    apply rule_row_17
    exact row_60
    exact row_18
  -- chapter_167_line_16: GL tag equality1.
  have row_16 : (succ v9 v1) := by
    have equality_source := row_32
    have equality_step_1 := row_17
    cases equality_step_1
    have equality_step_2 := row_26
    cases equality_step_2
    exact equality_source
  -- chapter_167_line_15: GL tag equality1.
  have row_15 : (v2 = v9) := by
    have equality_source := row_17
    have equality_step_1 := row_26
    cases equality_step_1
    exact equality_source
  have rule_row_14 := peano_source_039 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_167_line_14: GL tag implication.
  have row_14 : (v1 = v2) := by
    apply rule_row_14
    exact row_18
    exact row_60
  -- chapter_167_line_13: GL tag equality1.
  have row_13 : (add v2 one v9) := by
    have equality_source := row_30
    have equality_step_1 := row_15
    cases equality_step_1
    have equality_step_2 := row_14
    cases equality_step_2
    exact equality_source
  -- chapter_167_line_12: GL tag implication.
  have row_12 : (add v2 two v1) := by
    apply row_48
    exact row_22
    exact row_41
    exact row_13
    exact row_16
  have rule_row_11 := peano_source_065 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_167_line_11: GL tag implication.
  have row_11 : (gl_inSequence N one succ) := by
    apply rule_row_11
  -- chapter_167_line_10: GL tag expansion.
  have row_10 : (¬ (∀ (v5 : α), ((N v5) → (¬ (succ v5 one))))) := by
    simpa only [gl_inSequence] using (row_11)
  have exists_row_10 : ∃ (v5 : α), ((N v5) ∧ (succ v5 one)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_10
      intro v5 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v5, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v5, witness_row_10⟩ := exists_row_10
  -- chapter_167_line_9: GL tag disintegration.
  have row_9 : (succ v5 one) := by
    exact witness_row_10.2
  -- chapter_167_line_8: GL tag implication.
  have row_8 : (zero = v5) := by
    apply row_39
    exact row_22
    exact row_58
    exact row_9
  -- chapter_167_line_7: GL tag symmetry of equality.
  have row_7 : (v9 = v1) := by
    exact Eq.symm row_26
  -- chapter_167_line_6: GL tag equality1.
  have row_6 : (add v2 v5 v1) := by
    have equality_source := row_27
    have equality_step_1 := row_14
    cases equality_step_1
    have equality_step_2 := row_8
    cases equality_step_2
    have equality_step_3 := row_7
    cases equality_step_3
    exact equality_source
  have rule_row_5 := peano_source_037 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_167_line_5: GL tag implication.
  have row_5 : (v5 = two) := by
    apply rule_row_5
    exact row_6
    exact row_12
  -- chapter_167_line_4: GL tag equality2.
  have row_4 : (zero = two) := by
    exact Eq.trans row_8 row_5
  -- chapter_167_line_3: GL tag implication.
  have row_3 : (¬ (succ one zero)) := by
    apply row_37
    exact row_22
  -- chapter_167_line_2: GL tag equality1.
  have row_2 : (¬ (succ one two)) := by
    have equality_source := row_3
    have equality_step_1 := row_4
    cases equality_step_1
    exact equality_source
  -- chapter_167_line_1: GL tag contradiction.
  have row_1 : (¬ (gl_interval N add zero v2 V2)) := by
    exact False.elim (row_2 row_41)
  exact row_1 reductio

theorem gauss_source_095
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
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
  -- chapter_169_line_50: GL tag task formulation.
  have row_50 : (gl_interval N add zero v1 V1) := by
    exact premise_3
  -- chapter_169_line_49: GL tag expansion.
  have row_49 : (((((gl_implication89 V1 N add zero) ∧ (gl_implication90 V1 N add v1)) ∧ (gl_implication91 N add zero v1 V1)) ∧ (N zero)) ∧ (N v1)) := by
    simpa only [gl_interval] using (row_50)
  -- chapter_169_line_48: GL tag disintegration.
  have row_48 : (N v1) := by
    exact row_49.2
  -- chapter_169_line_47: GL tag task formulation.
  have row_47 : (gl_interval N add zero v2 V2) := by
    exact reductio
  -- chapter_169_line_46: GL tag task formulation.
  have row_46 : (gl_AnchorGauss N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_169_line_45: GL tag theorem.
  have row_45 := gauss_source_069 N zero succ add mul one two identity anchor relationalInduction strongRelationalInduction
  -- chapter_169_line_44: GL tag implication.
  have row_44 : (V2 v2) := by
    apply row_45
    exact row_47
  -- chapter_169_line_43: GL tag expansion.
  have row_43 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorGauss] using (row_46)
  -- chapter_169_line_42: GL tag disintegration.
  have row_42 : (succ zero one) := by
    exact row_43.1.1.2
  -- chapter_169_line_41: GL tag disintegration.
  have row_41 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_43.1.1.1
  -- chapter_169_line_40: GL tag expansion.
  have row_40 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_41)
  -- chapter_169_line_39: GL tag disintegration.
  have row_39 : (gl_fXY succ N N) := by
    exact row_40.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_169_line_38: GL tag expansion.
  have row_38 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_39)
  -- chapter_169_line_37: GL tag disintegration.
  have row_37 : (gl_implication1 succ N) := by
    exact row_38.1.1.2
  -- chapter_169_line_36: GL tag expansion.
  have row_36 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using (row_37)
  -- chapter_169_line_35: GL tag disintegration.
  have row_35 : (gl_implication0 succ N) := by
    exact row_38.1.1.1
  -- chapter_169_line_34: GL tag expansion.
  have row_34 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_35)
  -- chapter_169_line_33: GL tag disintegration.
  have row_33 : (gl_implication10 N succ) := by
    exact row_40.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_169_line_32: GL tag expansion.
  have row_32 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w2 w1) → (∀ (w3 : α), ((succ w3 w1) → (w2 = w3))))))) := by
    simpa only [gl_implication10] using (row_33)
  -- chapter_169_line_31: GL tag disintegration.
  have row_31 : (gl_implication9 N zero succ) := by
    exact row_40.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_169_line_30: GL tag expansion.
  have row_30 : (∀ (w1 : α), ((N w1) → (¬ (succ w1 zero)))) := by
    simpa only [gl_implication9] using (row_31)
  -- chapter_169_line_29: GL tag expansion for integration.
  have row_29 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_169_line_28_rule: GL tag reformulation for integration and.
  have row_28_rule : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_169_line_28: GL tag reformulation for integration and.
  have row_28 : (gl_implication76 N zero succ add mul one) := by
    simpa only [gl_implication76] using row_28_rule
  -- chapter_169_line_27: GL tag expansion.
  have row_27 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    simpa only [gl_implication76] using (row_28)
  -- chapter_169_line_26: GL tag implication.
  have row_26 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_27
    exact row_41
    exact row_42
  have rule_row_25 := peano_source_065 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_169_line_25: GL tag implication.
  have row_25 : (gl_inSequence N one succ) := by
    apply rule_row_25
  -- chapter_169_line_24: GL tag expansion.
  have row_24 : (¬ (∀ (v4 : α), ((N v4) → (¬ (succ v4 one))))) := by
    simpa only [gl_inSequence] using (row_25)
  have exists_row_24 : ∃ (v4 : α), ((N v4) ∧ (succ v4 one)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_24
      intro v4 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v4, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v4, witness_row_24⟩ := exists_row_24
  -- chapter_169_line_23: GL tag disintegration.
  have row_23 : (succ v4 one) := by
    exact witness_row_24.2
  -- chapter_169_line_22: GL tag implication grouping.
  have row_22 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (gl_and20 w1 N w2 N))) := by
    have auxiliary_source_1 := row_34
    have auxiliary_source_2 := row_36
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_premise
    exact ⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_premise)⟩
  -- chapter_169_line_21: GL tag implication.
  have row_21 : (gl_and20 zero N one N) := by
    apply row_22
    exact row_42
  -- chapter_169_line_20: GL tag expansion.
  have row_20 : ((N zero) ∧ (N one)) := by
    simpa only [gl_and20] using (row_21)
  -- chapter_169_line_19: GL tag disintegration.
  have row_19 : (N one) := by
    exact row_20.2
  -- chapter_169_line_18: GL tag implication.
  have row_18 : (zero = v4) := by
    apply row_32
    exact row_19
    exact row_42
    exact row_23
  -- chapter_169_line_17: GL tag expansion.
  have row_17 : (((((gl_implication89 V2 N add zero) ∧ (gl_implication90 V2 N add v2)) ∧ (gl_implication91 N add zero v2 V2)) ∧ (N zero)) ∧ (N v2)) := by
    simpa only [gl_interval] using (row_47)
  -- chapter_169_line_16: GL tag disintegration.
  have row_16 : (gl_implication89 V2 N add zero) := by
    exact row_17.1.1.1.1
  -- chapter_169_line_15: GL tag expansion.
  have row_15 : (∀ (w1 : α), ((V2 w1) → (gl_preorder N add zero w1))) := by
    simpa only [gl_implication89] using (row_16)
  -- chapter_169_line_14: GL tag implication.
  have row_14 : (gl_preorder N add zero v2) := by
    apply row_15
    exact row_44
  -- chapter_169_line_13: GL tag equality1.
  have row_13 : (gl_preorder N add v4 v2) := by
    have equality_source := row_14
    have equality_step_1 := row_18
    cases equality_step_1
    exact equality_source
  -- chapter_169_line_12: GL tag task formulation.
  have row_12 : (gl_limitSet N add V1 zero V2) := by
    exact premise_2
  -- chapter_169_line_11: GL tag expansion.
  have row_11 : (((gl_subset V2 V1) ∧ (gl_implication90 V2 N add zero)) ∧ (gl_implication141 V1 N add zero V2)) := by
    simpa only [gl_limitSet] using (row_12)
  -- chapter_169_line_10: GL tag disintegration.
  have row_10 : (gl_implication90 V2 N add zero) := by
    exact row_11.1.2
  -- chapter_169_line_9: GL tag expansion.
  have row_9 : (∀ (w1 : α), ((V2 w1) → (gl_preorder N add w1 zero))) := by
    simpa only [gl_implication90] using (row_10)
  -- chapter_169_line_8: GL tag implication.
  have row_8 : (gl_preorder N add v2 zero) := by
    apply row_9
    exact row_44
  -- chapter_169_line_7: GL tag equality1.
  have row_7 : (gl_preorder N add v2 v4) := by
    have equality_source := row_8
    have equality_step_1 := row_18
    cases equality_step_1
    exact equality_source
  have rule_row_6 := peano_source_039 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_169_line_6: GL tag implication.
  have row_6 : (v4 = v2) := by
    apply rule_row_6
    exact row_13
    exact row_7
  -- chapter_169_line_5: GL tag equality2.
  have row_5 : (zero = v2) := by
    exact Eq.trans row_18 row_6
  -- chapter_169_line_4: GL tag implication.
  have row_4 : (¬ (succ v1 zero)) := by
    apply row_30
    exact row_48
  -- chapter_169_line_3: GL tag equality1.
  have row_3 : (¬ (succ v1 v2)) := by
    have equality_source := row_4
    have equality_step_1 := row_5
    cases equality_step_1
    exact equality_source
  -- chapter_169_line_2: GL tag task formulation.
  have row_2 : (succ v1 v2) := by
    exact premise_1
  -- chapter_169_line_1: GL tag contradiction.
  have row_1 : (¬ (gl_interval N add zero v2 V2)) := by
    exact False.elim (row_3 row_2)
  exact row_1 reductio

theorem gauss_source_098
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (V1 : GLSet α), ((gl_interval N add zero one V1) → (V1 one))) := by
  intro V1
  intro premise_1
  -- chapter_172_line_8: GL tag variable copy.
  have row_8 : (one = one) := by
    rfl
  -- chapter_172_line_7: GL tag task formulation.
  have row_7 : (gl_interval N add zero one V1) := by
    exact premise_1
  -- chapter_172_line_6: GL tag equality1.
  have row_6 : (gl_interval N add zero one V1) := by
    have equality_source := row_7
    have equality_step_1 := row_8
    cases equality_step_1
    exact equality_source
  -- chapter_172_line_5: GL tag task formulation.
  have row_5 : (gl_AnchorGauss N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_172_line_4: GL tag theorem.
  have row_4 := gauss_source_097 N zero succ add mul one two identity anchor relationalInduction strongRelationalInduction
  -- chapter_172_line_3: GL tag implication.
  have row_3 : (V1 one) := by
    apply row_4
    exact row_6
    exact row_7
  -- chapter_172_line_2: GL tag symmetry of equality.
  have row_2 : (one = one) := by
    exact Eq.symm row_8
  -- chapter_172_line_1: GL tag equality1.
  have row_1 : (V1 one) := by
    have equality_source := row_3
    have equality_step_1 := row_2
    cases equality_step_1
    exact equality_source
  exact row_1

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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (V1 : GLSet α), ((gl_interval N add zero v2 V1) → (gl_existence8 N add V1 v1 zero))))) := by
  have reformulationSource := gauss_source_073 N zero succ add mul one two identity anchor relationalInduction strongRelationalInduction
  have defined_output_totality : (∀ (V1 : GLSet α) (v1 : α), (gl_limitSet N add V1 v1 (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 v1)))) := by
    simp only [gl_limitSet, gl_subset, gl_implication90, gl_implication141]
    intro V1 v1
    exact ⟨⟨(fun x_1 definition_premise_0 => definition_premise_0.1), (fun x_1 definition_premise_0 => definition_premise_0.2)⟩, (fun x_1 definition_premise_0 definition_premise_1 => ⟨definition_premise_0, definition_premise_1⟩)⟩
  -- chapter_144_line_1: GL tag reformulated from.
  have row_1 : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (V1 : GLSet α), ((gl_interval N add zero v2 V1) → (gl_existence8 N add V1 v1 zero))))) := by
    classical
    intro v1
    intro v2
    intro reformulation_premise_1
    intro V1
    intro reformulation_premise_2
    simp only [gl_existence8]
    intro no_defined_output
    have defined_output : (gl_limitSet N add V1 v1 (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 v1))) := defined_output_totality V1 v1
    have output_result : (gl_interval N add zero v1 (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 v1))) := by
      exact reformulationSource _ _ reformulation_premise_1 _ (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 v1)) defined_output reformulation_premise_2
    exact no_defined_output (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 v1)) defined_output output_result
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (V1 : GLSet α), ((gl_interval N add zero one V1) → (gl_existence10 N add V1 v1 zero v2))))) := by
  have reformulationSource := gauss_source_081 N zero succ add mul one two identity anchor relationalInduction strongRelationalInduction
  have defined_output_totality : (∀ (V1 : GLSet α) (v1 : α), (gl_limitSet N add V1 v1 (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 v1)))) := by
    simp only [gl_limitSet, gl_subset, gl_implication90, gl_implication141]
    intro V1 v1
    exact ⟨⟨(fun x_1 definition_premise_0 => definition_premise_0.1), (fun x_1 definition_premise_0 => definition_premise_0.2)⟩, (fun x_1 definition_premise_0 definition_premise_1 => ⟨definition_premise_0, definition_premise_1⟩)⟩
  -- chapter_156_line_1: GL tag reformulated from.
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
      exact reformulationSource _ _ reformulation_premise_1 _ (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 v1)) defined_output reformulation_premise_2
    exact no_defined_output (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 v1)) defined_output output_result
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (V1 : GLSet α) (V2 : GLSet α), ((gl_limitSet N add V1 v2 V2) → ((gl_interval N add zero v1 V2) → (¬ (gl_interval N add zero v2 V1))))))) := by
  intro v1
  intro v2
  intro premise_1
  intro V1
  intro V2
  intro premise_2
  intro premise_3
  intro reductio
  -- chapter_158_line_53: GL tag task formulation.
  have row_53 : (gl_limitSet N add V1 v2 V2) := by
    exact premise_2
  -- chapter_158_line_52: GL tag expansion.
  have row_52 : (((gl_subset V2 V1) ∧ (gl_implication90 V2 N add v2)) ∧ (gl_implication141 V1 N add v2 V2)) := by
    simpa only [gl_limitSet] using (row_53)
  -- chapter_158_line_51: GL tag disintegration.
  have row_51 : (gl_implication141 V1 N add v2 V2) := by
    exact row_52.2
  -- chapter_158_line_50: GL tag expansion.
  have row_50 : (∀ (w1 : α), ((V1 w1) → ((gl_preorder N add w1 v2) → (V2 w1)))) := by
    simpa only [gl_implication141] using (row_51)
  -- chapter_158_line_49: GL tag disintegration.
  have row_49 : (gl_implication90 V2 N add v2) := by
    exact row_52.1.2
  -- chapter_158_line_48: GL tag task formulation.
  have row_48 : (gl_interval N add zero v1 V2) := by
    exact premise_3
  -- chapter_158_line_47: GL tag expansion.
  have row_47 : (((((gl_implication89 V2 N add zero) ∧ (gl_implication90 V2 N add v1)) ∧ (gl_implication91 N add zero v1 V2)) ∧ (N zero)) ∧ (N v1)) := by
    simpa only [gl_interval] using (row_48)
  -- chapter_158_line_46: GL tag disintegration.
  have row_46 : (N v1) := by
    exact row_47.2
  -- chapter_158_line_45: GL tag disintegration.
  have row_45 : (gl_implication89 V2 N add zero) := by
    exact row_47.1.1.1.1
  -- chapter_158_line_44: GL tag task formulation.
  have row_44 : (gl_AnchorGauss N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_158_line_43: GL tag expansion.
  have row_43 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorGauss] using (row_44)
  -- chapter_158_line_42: GL tag disintegration.
  have row_42 : (succ zero one) := by
    exact row_43.1.1.2
  -- chapter_158_line_41: GL tag disintegration.
  have row_41 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_43.1.1.1
  -- chapter_158_line_40: GL tag expansion.
  have row_40 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_41)
  -- chapter_158_line_39: GL tag disintegration.
  have row_39 : (gl_fXYZ add N N N) := by
    exact row_40.1.1.1.1.1.1.1.1.1.2
  -- chapter_158_line_38: GL tag expansion.
  have row_38 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_39)
  -- chapter_158_line_37: GL tag disintegration.
  have row_37 : (gl_implication18 N N N add) := by
    exact row_38.1.2
  -- chapter_158_line_36: GL tag expansion.
  have row_36 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication18] using (row_37)
  -- chapter_158_line_35: GL tag disintegration.
  have row_35 : (gl_implication21 N zero add) := by
    exact row_40.1.1.1.1.1.1.1.1.2
  -- chapter_158_line_34: GL tag expansion.
  have row_34 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((add w1 zero w2) → (w1 = w2))))) := by
    simpa only [gl_implication21] using (row_35)
  -- chapter_158_line_33: GL tag disintegration.
  have row_33 : (gl_implication23 N succ add) := by
    exact row_40.1.1.1.1.1.1.2
  -- chapter_158_line_32: GL tag expansion.
  have row_32 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication23] using (row_33)
  -- chapter_158_line_31: GL tag expansion for integration.
  have row_31 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_158_line_30_rule: GL tag reformulation for integration and.
  have row_30_rule : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_158_line_30: GL tag reformulation for integration and.
  have row_30 : (gl_implication76 N zero succ add mul one) := by
    simpa only [gl_implication76] using row_30_rule
  -- chapter_158_line_29: GL tag expansion.
  have row_29 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    simpa only [gl_implication76] using (row_30)
  -- chapter_158_line_28: GL tag implication.
  have row_28 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_29
    exact row_41
    exact row_42
  -- chapter_158_line_27: GL tag task formulation.
  have row_27 : (succ v1 v2) := by
    exact premise_1
  have rule_row_26 := peano_source_010 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_158_line_26: GL tag implication.
  have row_26 : (add one v1 v2) := by
    apply rule_row_26
    exact row_27
  have rule_row_25 := peano_source_012 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_158_line_25: GL tag implication.
  have row_25 : (add v1 one v2) := by
    apply rule_row_25
    exact row_26
  -- chapter_158_line_24: GL tag task formulation.
  have row_24 : (gl_interval N add zero v2 V1) := by
    exact reductio
  -- chapter_158_line_23: GL tag expansion.
  have row_23 : (((((gl_implication89 V1 N add zero) ∧ (gl_implication90 V1 N add v2)) ∧ (gl_implication91 N add zero v2 V1)) ∧ (N zero)) ∧ (N v2)) := by
    simpa only [gl_interval] using (row_24)
  -- chapter_158_line_22: GL tag disintegration.
  have row_22 : (N zero) := by
    exact row_23.1.2
  -- chapter_158_line_21: GL tag disintegration.
  have row_21 : (N v2) := by
    exact row_23.2
  -- chapter_158_line_20: GL tag disintegration.
  have row_20 : (gl_implication91 N add zero v2 V1) := by
    exact row_23.1.1.2
  -- chapter_158_line_19: GL tag expansion.
  have row_19 : (∀ (w1 : α), ((gl_preorder N add zero w1) → ((gl_preorder N add w1 v2) → (V1 w1)))) := by
    simpa only [gl_implication91] using (row_20)
  -- chapter_158_line_18: GL tag implication.
  have row_18 : (gl_existence1 N v1 zero add) := by
    apply row_36
    exact row_46
    exact row_22
  -- chapter_158_line_17: GL tag expansion.
  have row_17 : (¬ (∀ (v4 : α), ((N v4) → (¬ (add v1 zero v4))))) := by
    simpa only [gl_existence1] using (row_18)
  have exists_row_17 : ∃ (v4 : α), ((N v4) ∧ (add v1 zero v4)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_17
      intro v4 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v4, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v4, witness_row_17⟩ := exists_row_17
  -- chapter_158_line_16: GL tag disintegration.
  have row_16 : (add v1 zero v4) := by
    exact witness_row_17.2
  -- chapter_158_line_15: GL tag implication.
  have row_15 : (v1 = v4) := by
    apply row_34
    exact row_46
    exact row_16
  -- chapter_158_line_14: GL tag implication.
  have row_14 : (succ v4 v2) := by
    apply row_32
    exact row_22
    exact row_42
    exact row_16
    exact row_25
  -- chapter_158_line_13: GL tag equality1.
  have row_13 : (gl_interval N add zero v4 V2) := by
    have equality_source := row_48
    have equality_step_1 := row_15
    cases equality_step_1
    exact equality_source
  -- chapter_158_line_12: GL tag theorem.
  have row_12 := gauss_source_083 N zero succ add mul one two identity anchor relationalInduction strongRelationalInduction
  -- chapter_158_line_11: GL tag implication.
  have row_11 : (¬ (gl_interval N add zero v2 V2)) := by
    apply row_12
    exact row_14
    exact row_13
  -- chapter_158_line_10: GL tag expansion.
  have row_10 : (¬ (((((gl_implication89 V2 N add zero) ∧ (gl_implication90 V2 N add v2)) ∧ (gl_implication91 N add zero v2 V2)) ∧ (N zero)) ∧ (N v2))) := by
    simpa only [gl_interval] using (row_11)
  -- chapter_158_line_9: GL tag disintegration.
  have row_9 : ((gl_implication89 V2 N add zero) → ((gl_implication90 V2 N add v2) → ((N zero) → ((N v2) → (¬ (gl_implication91 N add zero v2 V2)))))) := by
    classical
    intro projection_premise_1 projection_premise_2 projection_premise_3 projection_premise_4
    intro projection_counterexample
    exact (row_10) ⟨⟨⟨⟨projection_premise_1, projection_premise_2⟩, projection_counterexample⟩, projection_premise_3⟩, projection_premise_4⟩
  -- chapter_158_line_8: GL tag implication.
  have row_8 : (¬ (gl_implication91 N add zero v2 V2)) := by
    apply row_9
    exact row_45
    exact row_49
    exact row_22
    exact row_21
  -- chapter_158_line_7: GL tag expansion.
  have row_7 : (¬ (∀ (v3 : α), ((gl_preorder N add zero v3) → ((gl_preorder N add v3 v2) → (V2 v3))))) := by
    simpa only [gl_implication91] using (row_8)
  have exists_row_7 : ∃ (v3 : α), ((gl_preorder N add zero v3) ∧ ((gl_preorder N add v3 v2) ∧ (¬ (V2 v3)))) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_7
      intro v3 counterexample_premise_0 counterexample_premise_1
      apply Classical.byContradiction
      intro counterexample_head
      exact no_counterexample ⟨v3, counterexample_premise_0, counterexample_premise_1, counterexample_head⟩
  obtain ⟨v3, witness_row_7⟩ := exists_row_7
  -- chapter_158_line_6: GL tag disintegration.
  have row_6 : (gl_preorder N add v3 v2) := by
    exact witness_row_7.2.1
  -- chapter_158_line_5: GL tag disintegration.
  have row_5 : (gl_preorder N add zero v3) := by
    exact witness_row_7.1
  -- chapter_158_line_4: GL tag disintegration.
  have row_4 : (¬ (V2 v3)) := by
    exact witness_row_7.2.2
  -- chapter_158_line_3: GL tag implication.
  have row_3 : (V1 v3) := by
    apply row_19
    exact row_5
    exact row_6
  -- chapter_158_line_2: GL tag implication.
  have row_2 : (V2 v3) := by
    apply row_50
    exact row_3
    exact row_6
  -- chapter_158_line_1: GL tag contradiction.
  have row_1 : (¬ (gl_interval N add zero v2 V1)) := by
    exact False.elim (row_4 row_2)
  exact row_1 reductio

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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (V1 : GLSet α) (V2 : GLSet α), ((gl_limitSet N add V1 v2 V2) → ((gl_interval N add zero v2 V1) → (¬ (gl_interval N add zero v1 V2))))))) := by
  intro v1
  intro v2
  intro premise_1
  intro V1
  intro V2
  intro premise_2
  intro premise_3
  intro reductio
  -- chapter_159_line_28: GL tag task formulation.
  have row_28 : (gl_interval N add zero v2 V1) := by
    exact premise_3
  -- chapter_159_line_27: GL tag expansion.
  have row_27 : (((((gl_implication89 V1 N add zero) ∧ (gl_implication90 V1 N add v2)) ∧ (gl_implication91 N add zero v2 V1)) ∧ (N zero)) ∧ (N v2)) := by
    simpa only [gl_interval] using (row_28)
  -- chapter_159_line_26: GL tag disintegration.
  have row_26 : (gl_implication91 N add zero v2 V1) := by
    exact row_27.1.1.2
  -- chapter_159_line_25: GL tag expansion.
  have row_25 : (∀ (w1 : α), ((gl_preorder N add zero w1) → ((gl_preorder N add w1 v2) → (V1 w1)))) := by
    simpa only [gl_implication91] using (row_26)
  -- chapter_159_line_24: GL tag disintegration.
  have row_24 : (N v2) := by
    exact row_27.2
  -- chapter_159_line_23: GL tag task formulation.
  have row_23 : (gl_limitSet N add V1 v2 V2) := by
    exact premise_2
  -- chapter_159_line_22: GL tag expansion.
  have row_22 : (((gl_subset V2 V1) ∧ (gl_implication90 V2 N add v2)) ∧ (gl_implication141 V1 N add v2 V2)) := by
    simpa only [gl_limitSet] using (row_23)
  -- chapter_159_line_21: GL tag disintegration.
  have row_21 : (gl_implication141 V1 N add v2 V2) := by
    exact row_22.2
  -- chapter_159_line_20: GL tag expansion.
  have row_20 : (∀ (w1 : α), ((V1 w1) → ((gl_preorder N add w1 v2) → (V2 w1)))) := by
    simpa only [gl_implication141] using (row_21)
  -- chapter_159_line_19: GL tag disintegration.
  have row_19 : (gl_implication90 V2 N add v2) := by
    exact row_22.1.2
  -- chapter_159_line_18: GL tag task formulation.
  have row_18 : (gl_interval N add zero v1 V2) := by
    exact reductio
  -- chapter_159_line_17: GL tag task formulation.
  have row_17 : (succ v1 v2) := by
    exact premise_1
  -- chapter_159_line_16: GL tag task formulation.
  have row_16 : (gl_AnchorGauss N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_159_line_15: GL tag theorem.
  have row_15 := gauss_source_083 N zero succ add mul one two identity anchor relationalInduction strongRelationalInduction
  -- chapter_159_line_14: GL tag implication.
  have row_14 : (¬ (gl_interval N add zero v2 V2)) := by
    apply row_15
    exact row_17
    exact row_18
  -- chapter_159_line_13: GL tag expansion.
  have row_13 : (¬ (((((gl_implication89 V2 N add zero) ∧ (gl_implication90 V2 N add v2)) ∧ (gl_implication91 N add zero v2 V2)) ∧ (N zero)) ∧ (N v2))) := by
    simpa only [gl_interval] using (row_14)
  -- chapter_159_line_12: GL tag disintegration.
  have row_12 : ((gl_implication89 V2 N add zero) → ((gl_implication90 V2 N add v2) → ((N zero) → ((N v2) → (¬ (gl_implication91 N add zero v2 V2)))))) := by
    classical
    intro projection_premise_1 projection_premise_2 projection_premise_3 projection_premise_4
    intro projection_counterexample
    exact (row_13) ⟨⟨⟨⟨projection_premise_1, projection_premise_2⟩, projection_counterexample⟩, projection_premise_3⟩, projection_premise_4⟩
  -- chapter_159_line_11: GL tag expansion.
  have row_11 : (((((gl_implication89 V2 N add zero) ∧ (gl_implication90 V2 N add v1)) ∧ (gl_implication91 N add zero v1 V2)) ∧ (N zero)) ∧ (N v1)) := by
    simpa only [gl_interval] using (row_18)
  -- chapter_159_line_10: GL tag disintegration.
  have row_10 : (gl_implication89 V2 N add zero) := by
    exact row_11.1.1.1.1
  -- chapter_159_line_9: GL tag disintegration.
  have row_9 : (N zero) := by
    exact row_11.1.2
  -- chapter_159_line_8: GL tag implication.
  have row_8 : (¬ (gl_implication91 N add zero v2 V2)) := by
    apply row_12
    exact row_10
    exact row_19
    exact row_9
    exact row_24
  -- chapter_159_line_7: GL tag expansion.
  have row_7 : (¬ (∀ (v3 : α), ((gl_preorder N add zero v3) → ((gl_preorder N add v3 v2) → (V2 v3))))) := by
    simpa only [gl_implication91] using (row_8)
  have exists_row_7 : ∃ (v3 : α), ((gl_preorder N add zero v3) ∧ ((gl_preorder N add v3 v2) ∧ (¬ (V2 v3)))) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_7
      intro v3 counterexample_premise_0 counterexample_premise_1
      apply Classical.byContradiction
      intro counterexample_head
      exact no_counterexample ⟨v3, counterexample_premise_0, counterexample_premise_1, counterexample_head⟩
  obtain ⟨v3, witness_row_7⟩ := exists_row_7
  -- chapter_159_line_6: GL tag disintegration.
  have row_6 : (gl_preorder N add v3 v2) := by
    exact witness_row_7.2.1
  -- chapter_159_line_5: GL tag disintegration.
  have row_5 : (gl_preorder N add zero v3) := by
    exact witness_row_7.1
  -- chapter_159_line_4: GL tag disintegration.
  have row_4 : (¬ (V2 v3)) := by
    exact witness_row_7.2.2
  -- chapter_159_line_3: GL tag implication.
  have row_3 : (V1 v3) := by
    apply row_25
    exact row_5
    exact row_6
  -- chapter_159_line_2: GL tag implication.
  have row_2 : (V2 v3) := by
    apply row_20
    exact row_3
    exact row_6
  -- chapter_159_line_1: GL tag contradiction.
  have row_1 : (¬ (gl_interval N add zero v1 V2)) := by
    exact False.elim (row_4 row_2)
  exact row_1 reductio

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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (V1 : GLSet α), ((gl_interval N add zero v1 V1) → (gl_existence10 N add V1 one zero v2))))) := by
  have reformulationSource := gauss_source_087 N zero succ add mul one two identity anchor relationalInduction strongRelationalInduction
  have defined_output_totality : (∀ (V1 : GLSet α), (gl_limitSet N add V1 one (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 one)))) := by
    simp only [gl_limitSet, gl_subset, gl_implication90, gl_implication141]
    intro V1
    exact ⟨⟨(fun x_1 definition_premise_0 => definition_premise_0.1), (fun x_1 definition_premise_0 => definition_premise_0.2)⟩, (fun x_1 definition_premise_0 definition_premise_1 => ⟨definition_premise_0, definition_premise_1⟩)⟩
  -- chapter_162_line_1: GL tag reformulated from.
  have row_1 : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (V1 : GLSet α), ((gl_interval N add zero v1 V1) → (gl_existence10 N add V1 one zero v2))))) := by
    classical
    intro v1
    intro v2
    intro reformulation_premise_1
    intro V1
    intro reformulation_premise_2
    simp only [gl_existence10]
    intro no_defined_output
    have defined_output : (gl_limitSet N add V1 one (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 one))) := defined_output_totality V1
    have output_result : (¬ (gl_interval N add zero v2 (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 one)))) := by
      exact reformulationSource _ _ reformulation_premise_1 _ (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 one)) defined_output reformulation_premise_2
    exact no_defined_output (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 one)) defined_output output_result
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (V1 : GLSet α), ((gl_interval N add zero v1 V1) → (gl_existence10 N add V1 two zero v2))))) := by
  have reformulationSource := gauss_source_089 N zero succ add mul one two identity anchor relationalInduction strongRelationalInduction
  have defined_output_totality : (∀ (V1 : GLSet α), (gl_limitSet N add V1 two (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 two)))) := by
    simp only [gl_limitSet, gl_subset, gl_implication90, gl_implication141]
    intro V1
    exact ⟨⟨(fun x_1 definition_premise_0 => definition_premise_0.1), (fun x_1 definition_premise_0 => definition_premise_0.2)⟩, (fun x_1 definition_premise_0 definition_premise_1 => ⟨definition_premise_0, definition_premise_1⟩)⟩
  -- chapter_164_line_1: GL tag reformulated from.
  have row_1 : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (V1 : GLSet α), ((gl_interval N add zero v1 V1) → (gl_existence10 N add V1 two zero v2))))) := by
    classical
    intro v1
    intro v2
    intro reformulation_premise_1
    intro V1
    intro reformulation_premise_2
    simp only [gl_existence10]
    intro no_defined_output
    have defined_output : (gl_limitSet N add V1 two (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 two))) := defined_output_totality V1
    have output_result : (¬ (gl_interval N add zero v2 (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 two)))) := by
      exact reformulationSource _ _ reformulation_premise_1 _ (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 two)) defined_output reformulation_premise_2
    exact no_defined_output (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 two)) defined_output output_result
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (V1 : GLSet α), ((gl_interval N add zero v1 V1) → (gl_existence10 N add V1 v1 zero v2))))) := by
  have reformulationSource := gauss_source_091 N zero succ add mul one two identity anchor relationalInduction strongRelationalInduction
  have defined_output_totality : (∀ (V1 : GLSet α) (v1 : α), (gl_limitSet N add V1 v1 (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 v1)))) := by
    simp only [gl_limitSet, gl_subset, gl_implication90, gl_implication141]
    intro V1 v1
    exact ⟨⟨(fun x_1 definition_premise_0 => definition_premise_0.1), (fun x_1 definition_premise_0 => definition_premise_0.2)⟩, (fun x_1 definition_premise_0 definition_premise_1 => ⟨definition_premise_0, definition_premise_1⟩)⟩
  -- chapter_166_line_1: GL tag reformulated from.
  have row_1 : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (V1 : GLSet α), ((gl_interval N add zero v1 V1) → (gl_existence10 N add V1 v1 zero v2))))) := by
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
      exact reformulationSource _ _ reformulation_premise_1 _ (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 v1)) defined_output reformulation_premise_2
    exact no_defined_output (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 v1)) defined_output output_result
  exact row_1

theorem gauss_source_094
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (V1 : GLSet α), ((gl_interval N add zero v1 V1) → (gl_existence11 N add V1 v2 zero))))) := by
  have reformulationSource := gauss_source_093 N zero succ add mul one two identity anchor relationalInduction strongRelationalInduction
  have defined_output_totality : (∀ (V1 : GLSet α) (v2 : α), (gl_limitSet N add V1 v2 (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 v2)))) := by
    simp only [gl_limitSet, gl_subset, gl_implication90, gl_implication141]
    intro V1 v2
    exact ⟨⟨(fun x_1 definition_premise_0 => definition_premise_0.1), (fun x_1 definition_premise_0 => definition_premise_0.2)⟩, (fun x_1 definition_premise_0 definition_premise_1 => ⟨definition_premise_0, definition_premise_1⟩)⟩
  -- chapter_168_line_1: GL tag reformulated from.
  have row_1 : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (V1 : GLSet α), ((gl_interval N add zero v1 V1) → (gl_existence11 N add V1 v2 zero))))) := by
    classical
    intro v1
    intro v2
    intro reformulation_premise_1
    intro V1
    intro reformulation_premise_2
    simp only [gl_existence11]
    intro no_defined_output
    have defined_output : (gl_limitSet N add V1 v2 (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 v2))) := defined_output_totality V1 v2
    have output_result : (¬ (gl_interval N add zero v2 (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 v2)))) := by
      exact reformulationSource _ _ reformulation_premise_1 _ (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 v2)) defined_output reformulation_premise_2
    exact no_defined_output (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 v2)) defined_output output_result
  exact row_1

theorem gauss_source_096
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (V1 : GLSet α), ((gl_interval N add zero v1 V1) → (gl_existence9 N add V1 zero v2))))) := by
  have reformulationSource := gauss_source_095 N zero succ add mul one two identity anchor relationalInduction strongRelationalInduction
  have defined_output_totality : (∀ (V1 : GLSet α), (gl_limitSet N add V1 zero (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 zero)))) := by
    simp only [gl_limitSet, gl_subset, gl_implication90, gl_implication141]
    intro V1
    exact ⟨⟨(fun x_1 definition_premise_0 => definition_premise_0.1), (fun x_1 definition_premise_0 => definition_premise_0.2)⟩, (fun x_1 definition_premise_0 definition_premise_1 => ⟨definition_premise_0, definition_premise_1⟩)⟩
  -- chapter_170_line_1: GL tag reformulated from.
  have row_1 : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (V1 : GLSet α), ((gl_interval N add zero v1 V1) → (gl_existence9 N add V1 zero v2))))) := by
    classical
    intro v1
    intro v2
    intro reformulation_premise_1
    intro V1
    intro reformulation_premise_2
    simp only [gl_existence9]
    intro no_defined_output
    have defined_output : (gl_limitSet N add V1 zero (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 zero))) := defined_output_totality V1
    have output_result : (¬ (gl_interval N add zero v2 (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 zero)))) := by
      exact reformulationSource _ _ reformulation_premise_1 _ (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 zero)) defined_output reformulation_premise_2
    exact no_defined_output (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 zero)) defined_output output_result
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
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
  -- chapter_145_line_73: GL tag task formulation.
  have row_73 : (gl_limitSequence N add v1 V1 V2) := by
    exact premise_2
  -- chapter_145_line_72: GL tag task formulation.
  have row_72 : (gl_sequence N add zero v2 V1) := by
    exact premise_3
  -- chapter_145_line_71: GL tag task formulation.
  have row_71 : (succ v1 v2) := by
    exact premise_1
  -- chapter_145_line_70: GL tag task formulation.
  have row_70 : (gl_AnchorGauss N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_145_line_69: GL tag theorem.
  have row_69 := gauss_source_074 N zero succ add mul one two identity anchor relationalInduction strongRelationalInduction
  -- chapter_145_line_68: GL tag expansion for integration.
  have row_68 : (∀ (V4 : GLSet α), ((gl_implication7 V4 V2) ↔ (∀ (v16 : α), ((V4 v16) → (∀ (v13 : α), ((V2 v16 v13) → (∀ (v14 : α), ((V2 v16 v14) → (v13 = v14))))))))) := by
    intro V4
    exact Iff.rfl
  -- chapter_145_line_67: GL tag expansion for integration.
  have row_67 : (∀ (V4 : GLSet α), ((gl_implication6 V4 N V2) ↔ (∀ (v9 : α), ((V4 v9) → (gl_existence0 N v9 V2))))) := by
    intro V4
    exact Iff.rfl
  -- chapter_145_line_66: GL tag expansion for integration.
  have row_66 : ((gl_implication1 V2 N) ↔ (∀ (v8 : α) (v7 : α), ((V2 v8 v7) → (N v7)))) := by
    exact Iff.rfl
  -- chapter_145_line_65: GL tag expansion for integration.
  have row_65 : (∀ (V4 : GLSet α), ((gl_implication0 V2 V4) ↔ (∀ (v3 : α) (v6 : α), ((V2 v3 v6) → (V4 v3))))) := by
    intro V4
    exact Iff.rfl
  -- chapter_145_line_64: GL tag expansion.
  have row_64 : (((gl_implication136 V2 N add v1) ∧ (gl_implication137 V2 V1)) ∧ (gl_implication138 N add v1 V1 V2)) := by
    simpa only [gl_limitSequence] using (row_73)
  -- chapter_145_line_63: GL tag disintegration.
  have row_63 : (gl_implication138 N add v1 V1 V2) := by
    exact row_64.2
  -- chapter_145_line_62: GL tag expansion.
  have row_62 : (∀ (w1 : α), ((gl_preorder N add w1 v1) → (∀ (w2 : α), ((V1 w1 w2) → (V2 w1 w2))))) := by
    simpa only [gl_implication138] using (row_63)
  -- chapter_145_line_61: GL tag disintegration.
  have row_61 : (gl_implication136 V2 N add v1) := by
    exact row_64.1.1
  -- chapter_145_line_60: GL tag expansion.
  have row_60 : (∀ (w1 : α) (w2 : α), ((V2 w1 w2) → (gl_preorder N add w1 v1))) := by
    simpa only [gl_implication136] using (row_61)
  -- chapter_145_line_59: GL tag disintegration.
  have row_59 : (gl_implication137 V2 V1) := by
    exact row_64.1.2
  -- chapter_145_line_58: GL tag expansion.
  have row_58 : (∀ (w1 : α) (w2 : α), ((V2 w1 w2) → (V1 w1 w2))) := by
    simpa only [gl_implication137] using (row_59)
  -- chapter_145_line_57: GL tag expansion.
  have row_57 : (¬ (∀ (V5 : GLSet α), ((gl_interval N add zero v2 V5) → (¬ (gl_fXY V1 V5 N))))) := by
    simpa only [gl_sequence] using (row_72)
  have exists_row_57 : ∃ (V5 : GLSet α), ((gl_interval N add zero v2 V5) ∧ (gl_fXY V1 V5 N)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_57
      intro V5 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨V5, counterexample_premise_0, counterexample_head⟩
  obtain ⟨V5, witness_row_57⟩ := exists_row_57
  -- chapter_145_line_56: GL tag disintegration.
  have row_56 : (gl_fXY V1 V5 N) := by
    exact witness_row_57.2
  -- chapter_145_line_55: GL tag expansion.
  have row_55 : ((((gl_implication0 V1 V5) ∧ (gl_implication1 V1 N)) ∧ (gl_implication6 V5 N V1)) ∧ (gl_implication7 V5 V1)) := by
    simpa only [gl_fXY] using (row_56)
  -- chapter_145_line_54: GL tag disintegration.
  have row_54 : (gl_implication7 V5 V1) := by
    exact row_55.2
  -- chapter_145_line_53: GL tag expansion.
  have row_53 : (∀ (w1 : α), ((V5 w1) → (∀ (w2 : α), ((V1 w1 w2) → (∀ (w3 : α), ((V1 w1 w3) → (w2 = w3))))))) := by
    simpa only [gl_implication7] using (row_54)
  -- chapter_145_line_52: GL tag disintegration.
  have row_52 : (gl_implication6 V5 N V1) := by
    exact row_55.1.2
  -- chapter_145_line_51: GL tag expansion.
  have row_51 : (∀ (w1 : α), ((V5 w1) → (gl_existence0 N w1 V1))) := by
    simpa only [gl_implication6] using (row_52)
  -- chapter_145_line_50: GL tag disintegration.
  have row_50 : (gl_implication1 V1 N) := by
    exact row_55.1.1.2
  -- chapter_145_line_49: GL tag expansion.
  have row_49 : (∀ (w1 : α) (w2 : α), ((V1 w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using (row_50)
  -- chapter_145_line_48: GL tag disintegration.
  have row_48 : (gl_implication0 V1 V5) := by
    exact row_55.1.1.1
  -- chapter_145_line_47: GL tag expansion.
  have row_47 : (∀ (w1 : α) (w2 : α), ((V1 w1 w2) → (V5 w1))) := by
    simpa only [gl_implication0] using (row_48)
  -- chapter_145_line_46: GL tag disintegration.
  have row_46 : (gl_interval N add zero v2 V5) := by
    exact witness_row_57.1
  -- chapter_145_line_45: GL tag implication composition.
  have row_45 : (∀ (w1 : α) (w2 : α), ((V2 w1 w2) → (N w2))) := by
    have composition_source_1 := row_58
    have composition_source_2 := row_49
    intro composition_argument_1
    intro composition_argument_2
    intro composition_premise_0
    exact composition_source_2 composition_argument_1 composition_argument_2 (composition_source_1 composition_argument_1 composition_argument_2 composition_premise_0)
  -- chapter_145_line_44: GL tag premise element.
  have row_44 : (∀ (v8 : α) (v7 : α), ((V2 v8 v7) → (V2 v8 v7))) := by
    intro v8
    intro v7
    intro scope_premise_1
    exact scope_premise_1
  -- chapter_145_line_43: GL tag implication.
  have row_43 : (∀ (v8 : α) (v7 : α), ((V2 v8 v7) → (N v7))) := by
    intro v8
    intro v7
    intro scope_premise_1
    have scoped_fact_2 := row_44 v8 v7 scope_premise_1
    apply row_45
    exact scoped_fact_2
  -- chapter_145_line_42: GL tag implication.
  have row_42 : (gl_existence8 N add V5 v1 zero) := by
    apply row_69
    exact row_71
    exact row_46
  -- chapter_145_line_41: GL tag expansion.
  have row_41 : (¬ (∀ (V4 : GLSet α), ((gl_limitSet N add V5 v1 V4) → (¬ (gl_interval N add zero v1 V4))))) := by
    simpa only [gl_existence8] using (row_42)
  have exists_row_41 : ∃ (V4 : GLSet α), ((gl_limitSet N add V5 v1 V4) ∧ (gl_interval N add zero v1 V4)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_41
      intro V4 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨V4, counterexample_premise_0, counterexample_head⟩
  obtain ⟨V4, witness_row_41⟩ := exists_row_41
  -- chapter_145_line_40: GL tag disintegration.
  have row_40 : (gl_limitSet N add V5 v1 V4) := by
    exact witness_row_41.1
  -- chapter_145_line_39: GL tag expansion.
  have row_39 : (((gl_subset V4 V5) ∧ (gl_implication90 V4 N add v1)) ∧ (gl_implication141 V5 N add v1 V4)) := by
    simpa only [gl_limitSet] using (row_40)
  -- chapter_145_line_38: GL tag disintegration.
  have row_38 : (gl_implication90 V4 N add v1) := by
    exact row_39.1.2
  -- chapter_145_line_37: GL tag expansion.
  have row_37 : (∀ (w1 : α), ((V4 w1) → (gl_preorder N add w1 v1))) := by
    simpa only [gl_implication90] using (row_38)
  -- chapter_145_line_36: GL tag disintegration.
  have row_36 : (gl_subset V4 V5) := by
    exact row_39.1.1
  -- chapter_145_line_35: GL tag expansion.
  have row_35 : (∀ (w1 : α), ((V4 w1) → (V5 w1))) := by
    simpa only [gl_subset] using (row_36)
  -- chapter_145_line_34: GL tag disintegration.
  have row_34 : (gl_implication141 V5 N add v1 V4) := by
    exact row_39.2
  -- chapter_145_line_33: GL tag expansion.
  have row_33 : (∀ (w1 : α), ((V5 w1) → ((gl_preorder N add w1 v1) → (V4 w1)))) := by
    simpa only [gl_implication141] using (row_34)
  -- chapter_145_line_32: GL tag disintegration.
  have row_32 : (gl_interval N add zero v1 V4) := by
    exact witness_row_41.2
  -- chapter_145_line_31: GL tag premise element.
  have row_31 : (∀ (v16 : α), ((V4 v16) → (∀ (v13 : α), ((V2 v16 v13) → (∀ (v14 : α), ((V2 v16 v14) → (V4 v16))))))) := by
    intro v16
    intro scope_premise_1
    intro v13
    intro scope_premise_2
    intro v14
    intro scope_premise_3
    exact scope_premise_1
  -- chapter_145_line_30: GL tag implication.
  have row_30 : (∀ (v16 : α), ((V4 v16) → (∀ (v13 : α), ((V2 v16 v13) → (∀ (v14 : α), ((V2 v16 v14) → (V5 v16))))))) := by
    intro v16
    intro scope_premise_1
    intro v13
    intro scope_premise_2
    intro v14
    intro scope_premise_3
    have scoped_fact_2 := row_31 v16 scope_premise_1 v13 scope_premise_2 v14 scope_premise_3
    apply row_35
    exact scoped_fact_2
  -- chapter_145_line_29: GL tag premise element.
  have row_29 : (∀ (v16 : α), ((V4 v16) → (∀ (v13 : α), ((V2 v16 v13) → (∀ (v14 : α), ((V2 v16 v14) → (V2 v16 v14))))))) := by
    intro v16
    intro scope_premise_1
    intro v13
    intro scope_premise_2
    intro v14
    intro scope_premise_3
    exact scope_premise_3
  -- chapter_145_line_28: GL tag implication.
  have row_28 : (∀ (v16 : α), ((V4 v16) → (∀ (v13 : α), ((V2 v16 v13) → (∀ (v14 : α), ((V2 v16 v14) → (V1 v16 v14))))))) := by
    intro v16
    intro scope_premise_1
    intro v13
    intro scope_premise_2
    intro v14
    intro scope_premise_3
    have scoped_fact_2 := row_29 v16 scope_premise_1 v13 scope_premise_2 v14 scope_premise_3
    apply row_58
    exact scoped_fact_2
  -- chapter_145_line_27: GL tag premise element.
  have row_27 : (∀ (v16 : α), ((V4 v16) → (∀ (v13 : α), ((V2 v16 v13) → (∀ (v14 : α), ((V2 v16 v14) → (V2 v16 v13))))))) := by
    intro v16
    intro scope_premise_1
    intro v13
    intro scope_premise_2
    intro v14
    intro scope_premise_3
    exact scope_premise_2
  -- chapter_145_line_26: GL tag implication.
  have row_26 : (∀ (v16 : α), ((V4 v16) → (∀ (v13 : α), ((V2 v16 v13) → (∀ (v14 : α), ((V2 v16 v14) → (V1 v16 v13))))))) := by
    intro v16
    intro scope_premise_1
    intro v13
    intro scope_premise_2
    intro v14
    intro scope_premise_3
    have scoped_fact_2 := row_27 v16 scope_premise_1 v13 scope_premise_2 v14 scope_premise_3
    apply row_58
    exact scoped_fact_2
  -- chapter_145_line_25: GL tag implication.
  have row_25 : (∀ (v16 : α), ((V4 v16) → (∀ (v13 : α), ((V2 v16 v13) → (∀ (v14 : α), ((V2 v16 v14) → (v13 = v14))))))) := by
    intro v16
    intro scope_premise_1
    intro v13
    intro scope_premise_2
    intro v14
    intro scope_premise_3
    have scoped_fact_2 := row_30 v16 scope_premise_1 v13 scope_premise_2 v14 scope_premise_3
    have scoped_fact_3 := row_26 v16 scope_premise_1 v13 scope_premise_2 v14 scope_premise_3
    have scoped_fact_4 := row_28 v16 scope_premise_1 v13 scope_premise_2 v14 scope_premise_3
    apply row_53
    exact scoped_fact_2
    exact scoped_fact_3
    exact scoped_fact_4
  -- chapter_145_line_24: GL tag premise element.
  have row_24 : (∀ (v9 : α), ((V4 v9) → (V4 v9))) := by
    intro v9
    intro scope_premise_1
    exact scope_premise_1
  -- chapter_145_line_23: GL tag implication.
  have row_23 : (∀ (v9 : α), ((V4 v9) → (V5 v9))) := by
    intro v9
    intro scope_premise_1
    have scoped_fact_2 := row_24 v9 scope_premise_1
    apply row_35
    exact scoped_fact_2
  -- chapter_145_line_22: GL tag implication.
  have row_22 : (∀ (v9 : α), ((V4 v9) → (gl_existence0 N v9 V1))) := by
    intro v9
    intro scope_premise_1
    have scoped_fact_2 := row_23 v9 scope_premise_1
    apply row_51
    exact scoped_fact_2
  -- chapter_145_line_21: GL tag expansion.
  have row_21 : (∀ (v9 : α), ((V4 v9) → (¬ (∀ (v11 : α), ((N v11) → (¬ (V1 v9 v11))))))) := by
    simpa only [gl_existence0] using (row_22)
  -- chapter_145_line_20: GL tag disintegration.
  have row_20 : (∀ (v9 : α), ((V4 v9) → (∃ (v11 : α), ((N v11) ∧ (V1 v9 v11))))) := by
    intro v9
    intro scope_premise_1
    have scoped_fact_1 := row_21 v9 scope_premise_1
    have counterexample_exists : ∃ (v11 : α), ((N v11) ∧ (V1 v9 v11)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply scoped_fact_1
      intro v11 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v11, counterexample_premise_0, counterexample_head⟩
    obtain ⟨v11, scoped_witness_bundle⟩ := counterexample_exists
    exact ⟨v11, ⟨scoped_witness_bundle.1, scoped_witness_bundle.2⟩⟩
  -- chapter_145_line_19: GL tag disintegration.
  have row_19 : (∀ (v9 : α), ((V4 v9) → (∃ (v11 : α), ((N v11) ∧ (V1 v9 v11))))) := by
    intro v9
    intro scope_premise_1
    have scoped_fact_1 := row_21 v9 scope_premise_1
    have counterexample_exists : ∃ (v11 : α), ((N v11) ∧ (V1 v9 v11)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply scoped_fact_1
      intro v11 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v11, counterexample_premise_0, counterexample_head⟩
    obtain ⟨v11, scoped_witness_bundle⟩ := counterexample_exists
    exact ⟨v11, ⟨scoped_witness_bundle.1, scoped_witness_bundle.2⟩⟩
  -- chapter_145_line_18: GL tag implication.
  have row_18 : (∀ (v9 : α), ((V4 v9) → (gl_preorder N add v9 v1))) := by
    intro v9
    intro scope_premise_1
    have scoped_fact_2 := row_24 v9 scope_premise_1
    apply row_37
    exact scoped_fact_2
  -- chapter_145_line_17: GL tag implication.
  have row_17 : (∀ (v9 : α), ((V4 v9) → (∀ (v11 : α), (((N v11) ∧ (V1 v9 v11)) → (V2 v9 v11))))) := by
    intro v9
    intro scope_premise_1
    intro v11
    intro witness_guard_1
    have scoped_fact_3 := row_20 v9 scope_premise_1
    have scoped_fact_2 := row_18 v9 scope_premise_1
    apply row_62
    exact scoped_fact_2
    exact witness_guard_1.2
  -- chapter_145_line_16: GL tag expansion for integration.
  have row_16 : (∀ (v9 : α), ((gl_existence0 N v9 V2) ↔ (¬ (∀ (v12 : α), ((N v12) → (¬ (V2 v9 v12))))))) := by
    intro v9
    exact Iff.rfl
  -- chapter_145_line_15_rule: GL tag reformulation for integration >[bound].
  have row_15_rule : (∀ (v9 : α), ((V4 v9) → (∀ (cr15_1 : α), ((N cr15_1) → ((V2 v9 cr15_1) → (gl_existence0 N v9 V2)))))) := by
    intro v9
    intro scope_premise_1
    intro cr15_1
    intro integration_premise_1
    intro integration_premise_2
    apply (row_16 v9).2
    intro universal_counterexample
    exact universal_counterexample cr15_1 integration_premise_1 integration_premise_2
  -- chapter_145_line_15: GL tag reformulation for integration >[bound].
  have row_15 : (∀ (v9 : α), ((V4 v9) → (gl_implication4 N v9 V2))) := by
    simpa only [gl_implication4] using row_15_rule
  -- chapter_145_line_14: GL tag expansion.
  have row_14 : (∀ (v9 : α), ((V4 v9) → (∀ (v10 : α), ((N v10) → ((V2 v9 v10) → (gl_existence0 N v9 V2)))))) := by
    simpa only [gl_implication4] using (row_15)
  -- chapter_145_line_13: GL tag implication.
  have row_13 : (∀ (v9 : α), ((V4 v9) → (gl_existence0 N v9 V2))) := by
    intro v9
    intro scope_premise_1
    obtain ⟨v11, witness_guard_1⟩ := row_19 v9 scope_premise_1
    have scoped_fact_1 := row_14 v9 scope_premise_1
    have scoped_fact_3 := row_17 v9 scope_premise_1 v11 witness_guard_1
    apply scoped_fact_1
    exact witness_guard_1.1
    exact scoped_fact_3
  -- chapter_145_line_12: GL tag premise element.
  have row_12 : (∀ (v3 : α) (v6 : α), ((V2 v3 v6) → (V2 v3 v6))) := by
    intro v3
    intro v6
    intro scope_premise_1
    exact scope_premise_1
  -- chapter_145_line_11: GL tag implication.
  have row_11 : (∀ (v3 : α) (v6 : α), ((V2 v3 v6) → (V1 v3 v6))) := by
    intro v3
    intro v6
    intro scope_premise_1
    have scoped_fact_2 := row_12 v3 v6 scope_premise_1
    apply row_58
    exact scoped_fact_2
  -- chapter_145_line_10: GL tag implication.
  have row_10 : (∀ (v3 : α) (v6 : α), ((V2 v3 v6) → (V5 v3))) := by
    intro v3
    intro v6
    intro scope_premise_1
    have scoped_fact_2 := row_11 v3 v6 scope_premise_1
    apply row_47
    exact scoped_fact_2
  -- chapter_145_line_9: GL tag implication.
  have row_9 : (∀ (v3 : α) (v6 : α), ((V2 v3 v6) → (gl_preorder N add v3 v1))) := by
    intro v3
    intro v6
    intro scope_premise_1
    have scoped_fact_2 := row_12 v3 v6 scope_premise_1
    apply row_60
    exact scoped_fact_2
  -- chapter_145_line_8: GL tag implication.
  have row_8 : (∀ (v3 : α) (v6 : α), ((V2 v3 v6) → (V4 v3))) := by
    intro v3
    intro v6
    intro scope_premise_1
    have scoped_fact_2 := row_10 v3 v6 scope_premise_1
    have scoped_fact_3 := row_9 v3 v6 scope_premise_1
    apply row_33
    exact scoped_fact_2
    exact scoped_fact_3
  -- chapter_145_line_7_rule: GL tag reformulation for integration flattened.
  have row_7_rule : (∀ (cr7_1 : GLSet α), ((gl_interval N add zero v1 cr7_1) → ((gl_implication0 V2 cr7_1) → ((gl_implication1 V2 N) → ((gl_implication6 cr7_1 N V2) → ((gl_implication7 cr7_1 V2) → (gl_sequence N add zero v1 V2))))))) := by
    intro cr7_1
    intro integration_premise_1
    intro integration_premise_2
    intro integration_premise_3
    intro integration_premise_4
    intro integration_premise_5
    exact (fun flat_counterexample_1 => flat_counterexample_1 cr7_1 (integration_premise_1) (⟨⟨⟨integration_premise_2, integration_premise_3⟩, integration_premise_4⟩, integration_premise_5⟩))
  -- chapter_145_line_7: GL tag reformulation for integration flattened.
  have row_7 : (gl_implication97 N add zero v1 V2) := by
    simpa only [gl_implication97] using row_7_rule
  -- chapter_145_line_6: GL tag expansion.
  have row_6 : (∀ (V3 : GLSet α), ((gl_interval N add zero v1 V3) → ((gl_implication0 V2 V3) → ((gl_implication1 V2 N) → ((gl_implication6 V3 N V2) → ((gl_implication7 V3 V2) → (gl_sequence N add zero v1 V2))))))) := by
    simpa only [gl_implication97] using (row_7)
  -- chapter_145_line_5: GL tag validity name.
  have row_5 : (gl_implication7 V4 V2) := by
    simpa only [gl_implication7] using (row_25)
  -- chapter_145_line_4: GL tag validity name.
  have row_4 : (gl_implication6 V4 N V2) := by
    simpa only [gl_implication6] using (row_13)
  -- chapter_145_line_3: GL tag validity name.
  have row_3 : (gl_implication1 V2 N) := by
    simpa only [gl_implication1] using (row_43)
  -- chapter_145_line_2: GL tag validity name.
  have row_2 : (gl_implication0 V2 V4) := by
    simpa only [gl_implication0] using (row_8)
  -- chapter_145_line_1: GL tag implication.
  have row_1 : (gl_sequence N add zero v1 V2) := by
    apply row_6
    exact row_32
    exact row_2
    exact row_3
    exact row_4
    exact row_5
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (V1 : GLSet α), ((gl_interval N add zero v2 V1) → (gl_existence10 N add V1 v2 zero v1))))) := by
  have reformulationSource := gauss_source_085 N zero succ add mul one two identity anchor relationalInduction strongRelationalInduction
  have defined_output_totality : (∀ (V1 : GLSet α) (v2 : α), (gl_limitSet N add V1 v2 (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 v2)))) := by
    simp only [gl_limitSet, gl_subset, gl_implication90, gl_implication141]
    intro V1 v2
    exact ⟨⟨(fun x_1 definition_premise_0 => definition_premise_0.1), (fun x_1 definition_premise_0 => definition_premise_0.2)⟩, (fun x_1 definition_premise_0 definition_premise_1 => ⟨definition_premise_0, definition_premise_1⟩)⟩
  -- chapter_160_line_1: GL tag reformulated from.
  have row_1 : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (V1 : GLSet α), ((gl_interval N add zero v2 V1) → (gl_existence10 N add V1 v2 zero v1))))) := by
    classical
    intro v1
    intro v2
    intro reformulation_premise_1
    intro V1
    intro reformulation_premise_2
    simp only [gl_existence10]
    intro no_defined_output
    have defined_output : (gl_limitSet N add V1 v2 (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 v2))) := defined_output_totality V1 v2
    have output_result : (¬ (gl_interval N add zero v1 (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 v2)))) := by
      exact reformulationSource _ _ reformulation_premise_1 _ (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 v2)) defined_output reformulation_premise_2
    exact no_defined_output (fun (x_1 : α) => (V1 x_1) ∧ (gl_preorder N add x_1 v2)) defined_output output_result
  exact row_1

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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (V1 : GLBinaryRelation α), ((gl_sequence N add zero v2 V1) → (gl_existence12 N add v1 V1 zero))))) := by
  have reformulationSource := gauss_source_075 N zero succ add mul one two identity anchor relationalInduction strongRelationalInduction
  have defined_output_totality : (∀ (v1 : α) (V1 : GLBinaryRelation α), (gl_limitSequence N add v1 V1 (fun (x_1 : α) (x_2 : α) => (gl_preorder N add x_1 v1) ∧ (V1 x_1 x_2)))) := by
    simp only [gl_limitSequence, gl_implication136, gl_implication137, gl_implication138]
    intro v1 V1
    exact ⟨⟨(fun x_1 x_2 definition_premise_0 => definition_premise_0.1), (fun x_1 x_2 definition_premise_0 => definition_premise_0.2)⟩, (fun x_1 definition_premise_0 x_2 definition_premise_1 => ⟨definition_premise_0, definition_premise_1⟩)⟩
  -- chapter_146_line_1: GL tag reformulated from.
  have row_1 : (∀ (v1 : α) (v2 : α), ((succ v1 v2) → (∀ (V1 : GLBinaryRelation α), ((gl_sequence N add zero v2 V1) → (gl_existence12 N add v1 V1 zero))))) := by
    classical
    intro v1
    intro v2
    intro reformulation_premise_1
    intro V1
    intro reformulation_premise_2
    simp only [gl_existence12]
    intro no_defined_output
    have defined_output : (gl_limitSequence N add v1 V1 (fun (x_1 : α) (x_2 : α) => (gl_preorder N add x_1 v1) ∧ (V1 x_1 x_2))) := defined_output_totality v1 V1
    have output_result : (gl_sequence N add zero v1 (fun (x_1 : α) (x_2 : α) => (gl_preorder N add x_1 v1) ∧ (V1 x_1 x_2))) := by
      exact reformulationSource _ _ reformulation_premise_1 _ (fun (x_1 : α) (x_2 : α) => (gl_preorder N add x_1 v1) ∧ (V1 x_1 x_2)) defined_output reformulation_premise_2
    exact no_defined_output (fun (x_1 : α) (x_2 : α) => (gl_preorder N add x_1 v1) ∧ (V1 x_1 x_2)) defined_output output_result
  exact row_1

private theorem gauss_source_079_induction_typing
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v1 : α)
    (v4 : α)
    (assumption_5 : (gl_fold N succ add add identity zero v1 v4))
    : (N v1) := by
  -- chapter_149_line_5: GL tag task formulation.
  have row_5 : (gl_fold N succ add add identity zero v1 v4) := by
    exact assumption_5
  -- chapter_149_line_4: GL tag expansion.
  have row_4 : (¬ (∀ (V1 : GLSet α), ((gl_interval N add zero v1 V1) → (¬ (gl_existence2 V1 N zero identity v1 v4 succ add))))) := by
    simpa only [gl_fold] using (row_5)
  have exists_row_4 : ∃ (V1 : GLSet α), ((gl_interval N add zero v1 V1) ∧ (gl_existence2 V1 N zero identity v1 v4 succ add)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_4
      intro V1 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨V1, counterexample_premise_0, counterexample_head⟩
  obtain ⟨V1, witness_row_4⟩ := exists_row_4
  -- chapter_149_line_3: GL tag disintegration.
  have row_3 : (gl_interval N add zero v1 V1) := by
    exact witness_row_4.1
  -- chapter_149_line_2: GL tag expansion.
  have row_2 : (((((gl_implication89 V1 N add zero) ∧ (gl_implication90 V1 N add v1)) ∧ (gl_implication91 N add zero v1 V1)) ∧ (N zero)) ∧ (N v1)) := by
    simpa only [gl_interval] using (row_3)
  -- chapter_149_line_1: GL tag disintegration.
  have row_1 : (N v1) := by
    exact row_2.2
  exact row_1

private theorem gauss_source_079_check_zero
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
    (assumption_95 : (gl_fold N succ add add identity zero v1 v4))
    (assumption_74 : (v1 = zero))
    (assumption_73 : (mul v1 v2 v3))
    (assumption_30 : (succ v1 v2))
    : (mul two v4 v3) := by
  -- chapter_150_line_95: GL tag task formulation.
  have row_95 : (gl_fold N succ add add identity zero v1 v4) := by
    exact assumption_95
  -- chapter_150_line_94: GL tag expansion.
  have row_94 : (¬ (∀ (V1 : GLSet α), ((gl_interval N add zero v1 V1) → (¬ (gl_existence2 V1 N zero identity v1 v4 succ add))))) := by
    simpa only [gl_fold] using (row_95)
  have exists_row_94 : ∃ (V1 : GLSet α), ((gl_interval N add zero v1 V1) ∧ (gl_existence2 V1 N zero identity v1 v4 succ add)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_94
      intro V1 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨V1, counterexample_premise_0, counterexample_head⟩
  obtain ⟨V1, witness_row_94⟩ := exists_row_94
  -- chapter_150_line_93: GL tag disintegration.
  have row_93 : (gl_existence2 V1 N zero identity v1 v4 succ add) := by
    exact witness_row_94.2
  -- chapter_150_line_92: GL tag expansion.
  have row_92 : (¬ (∀ (V2 : GLBinaryRelation α), ((gl_fXY V2 V1 N) → (¬ (gl_and0 zero identity V2 v1 v4 V1 succ add))))) := by
    simpa only [gl_existence2] using (row_93)
  have exists_row_92 : ∃ (V2 : GLBinaryRelation α), ((gl_fXY V2 V1 N) ∧ (gl_and0 zero identity V2 v1 v4 V1 succ add)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_92
      intro V2 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨V2, counterexample_premise_0, counterexample_head⟩
  obtain ⟨V2, witness_row_92⟩ := exists_row_92
  -- chapter_150_line_91: GL tag disintegration.
  have row_91 : (gl_and0 zero identity V2 v1 v4 V1 succ add) := by
    exact witness_row_92.2
  -- chapter_150_line_90: GL tag expansion.
  have row_90 : (((gl_implication121 zero identity V2) ∧ (V2 v1 v4)) ∧ (gl_implication122 V1 succ V2 identity add)) := by
    simpa only [gl_and0] using (row_91)
  -- chapter_150_line_89: GL tag disintegration.
  have row_89 : (gl_implication121 zero identity V2) := by
    exact row_90.1.1
  -- chapter_150_line_88: GL tag expansion.
  have row_88 : (∀ (w1 : α), ((identity zero w1) → (V2 zero w1))) := by
    simpa only [gl_implication121] using (row_89)
  -- chapter_150_line_87: GL tag disintegration.
  have row_87 : (gl_fXY V2 V1 N) := by
    exact witness_row_92.1
  -- chapter_150_line_86: GL tag expansion.
  have row_86 : ((((gl_implication0 V2 V1) ∧ (gl_implication1 V2 N)) ∧ (gl_implication6 V1 N V2)) ∧ (gl_implication7 V1 V2)) := by
    simpa only [gl_fXY] using (row_87)
  -- chapter_150_line_85: GL tag disintegration.
  have row_85 : (gl_implication1 V2 N) := by
    exact row_86.1.1.2
  -- chapter_150_line_84: GL tag expansion.
  have row_84 : (∀ (w1 : α) (w2 : α), ((V2 w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using (row_85)
  -- chapter_150_line_83: GL tag disintegration.
  have row_83 : (gl_implication0 V2 V1) := by
    exact row_86.1.1.1
  -- chapter_150_line_82: GL tag expansion.
  have row_82 : (∀ (w1 : α) (w2 : α), ((V2 w1 w2) → (V1 w1))) := by
    simpa only [gl_implication0] using (row_83)
  -- chapter_150_line_81: GL tag disintegration.
  have row_81 : (V2 v1 v4) := by
    exact row_90.1.2
  -- chapter_150_line_80: GL tag disintegration.
  have row_80 : (gl_implication7 V1 V2) := by
    exact row_86.2
  -- chapter_150_line_79: GL tag expansion.
  have row_79 : (∀ (w1 : α), ((V1 w1) → (∀ (w2 : α), ((V2 w1 w2) → (∀ (w3 : α), ((V2 w1 w3) → (w2 = w3))))))) := by
    simpa only [gl_implication7] using (row_80)
  -- chapter_150_line_78: GL tag implication grouping.
  have row_78 : (∀ (w1 : α) (w2 : α), ((V2 w1 w2) → (gl_and20 w1 V1 w2 N))) := by
    have auxiliary_source_1 := row_82
    have auxiliary_source_2 := row_84
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_premise
    exact ⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_premise)⟩
  -- chapter_150_line_77: GL tag implication.
  have row_77 : (gl_and20 v1 V1 v4 N) := by
    apply row_78
    exact row_81
  -- chapter_150_line_76: GL tag expansion.
  have row_76 : ((V1 v1) ∧ (N v4)) := by
    simpa only [gl_and20] using (row_77)
  -- chapter_150_line_75: GL tag disintegration.
  have row_75 : (V1 v1) := by
    exact row_76.1
  -- chapter_150_line_74: GL tag recursion.
  have row_74 : (v1 = zero) := by
    exact assumption_74
  -- chapter_150_line_73: GL tag task formulation.
  have row_73 : (mul v1 v2 v3) := by
    exact assumption_73
  -- chapter_150_line_72: GL tag equality1.
  have row_72 : (mul zero v2 v3) := by
    have equality_source := row_73
    have equality_step_1 := row_74
    cases equality_step_1
    exact equality_source
  -- chapter_150_line_71: GL tag task formulation.
  have row_71 : (gl_AnchorGauss N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_150_line_70: GL tag expansion.
  have row_70 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorGauss] using (row_71)
  -- chapter_150_line_69: GL tag disintegration.
  have row_69 : (gl_identity N identity) := by
    exact row_70.2
  -- chapter_150_line_68: GL tag expansion.
  have row_68 : (((gl_implication0 identity N) ∧ (gl_implication37 identity)) ∧ (gl_implication38 N identity)) := by
    simpa only [gl_identity] using (row_69)
  -- chapter_150_line_67: GL tag disintegration.
  have row_67 : (gl_implication38 N identity) := by
    exact row_68.2
  -- chapter_150_line_66: GL tag expansion.
  have row_66 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((w1 = w2) → (identity w1 w2))))) := by
    simpa only [gl_implication38] using (row_67)
  -- chapter_150_line_65: GL tag disintegration.
  have row_65 : (succ zero one) := by
    exact row_70.1.1.2
  -- chapter_150_line_64: GL tag disintegration.
  have row_64 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_70.1.1.1
  -- chapter_150_line_63: GL tag expansion.
  have row_63 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_64)
  -- chapter_150_line_62: GL tag disintegration.
  have row_62 : (gl_fXYZ mul N N N) := by
    exact row_63.1.1.1.1.2
  -- chapter_150_line_61: GL tag expansion.
  have row_61 : (((((gl_implication11 mul N) ∧ (gl_implication12 mul N)) ∧ (gl_implication13 mul N)) ∧ (gl_implication18 N N N mul)) ∧ (gl_implication19 N N mul)) := by
    simpa only [gl_fXYZ] using (row_62)
  -- chapter_150_line_60: GL tag disintegration.
  have row_60 : (gl_implication13 mul N) := by
    exact row_61.1.1.2
  -- chapter_150_line_59: GL tag expansion.
  have row_59 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_60)
  -- chapter_150_line_58: GL tag disintegration.
  have row_58 : (gl_implication12 mul N) := by
    exact row_61.1.1.1.2
  -- chapter_150_line_57: GL tag expansion.
  have row_57 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_58)
  -- chapter_150_line_56: GL tag disintegration.
  have row_56 : (gl_implication11 mul N) := by
    exact row_61.1.1.1.1
  -- chapter_150_line_55: GL tag expansion.
  have row_55 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_56)
  -- chapter_150_line_54: GL tag disintegration.
  have row_54 : (gl_fXY succ N N) := by
    exact row_63.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_150_line_53: GL tag expansion.
  have row_53 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_54)
  -- chapter_150_line_52: GL tag disintegration.
  have row_52 : (gl_implication1 succ N) := by
    exact row_53.1.1.2
  -- chapter_150_line_51: GL tag expansion.
  have row_51 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using (row_52)
  -- chapter_150_line_50: GL tag disintegration.
  have row_50 : (gl_implication0 succ N) := by
    exact row_53.1.1.1
  -- chapter_150_line_49: GL tag expansion.
  have row_49 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_50)
  -- chapter_150_line_48: GL tag disintegration.
  have row_48 : (gl_fXYZ add N N N) := by
    exact row_63.1.1.1.1.1.1.1.1.1.2
  -- chapter_150_line_47: GL tag expansion.
  have row_47 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_48)
  -- chapter_150_line_46: GL tag disintegration.
  have row_46 : (gl_implication13 add N) := by
    exact row_47.1.1.2
  -- chapter_150_line_45: GL tag expansion.
  have row_45 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_46)
  -- chapter_150_line_44: GL tag disintegration.
  have row_44 : (gl_implication12 add N) := by
    exact row_47.1.1.1.2
  -- chapter_150_line_43: GL tag expansion.
  have row_43 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_44)
  -- chapter_150_line_42: GL tag disintegration.
  have row_42 : (gl_implication11 add N) := by
    exact row_47.1.1.1.1
  -- chapter_150_line_41: GL tag expansion.
  have row_41 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_42)
  -- chapter_150_line_40: GL tag disintegration.
  have row_40 : (gl_implication10 N succ) := by
    exact row_63.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_150_line_39: GL tag expansion.
  have row_39 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w2 w1) → (∀ (w3 : α), ((succ w3 w1) → (w2 = w3))))))) := by
    simpa only [gl_implication10] using (row_40)
  -- chapter_150_line_38: GL tag disintegration.
  have row_38 : (N zero) := by
    exact row_63.1.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_150_line_37: GL tag disintegration.
  have row_37 : (succ one two) := by
    exact row_70.1.2
  -- chapter_150_line_36: GL tag disintegration.
  have row_36 : (gl_implication26 zero N mul) := by
    exact row_63.1.1.2
  -- chapter_150_line_35: GL tag expansion.
  have row_35 : (∀ (w1 : α), ((w1 = zero) → (∀ (w2 : α), ((N w2) → ((N w1) → (mul w2 zero w1)))))) := by
    simpa only [gl_implication26] using (row_36)
  -- chapter_150_line_34: GL tag expansion for integration.
  have row_34 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_150_line_33_rule: GL tag reformulation for integration and.
  have row_33_rule : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_150_line_33: GL tag reformulation for integration and.
  have row_33 : (gl_implication76 N zero succ add mul one) := by
    simpa only [gl_implication76] using row_33_rule
  -- chapter_150_line_32: GL tag expansion.
  have row_32 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    simpa only [gl_implication76] using (row_33)
  -- chapter_150_line_31: GL tag implication.
  have row_31 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_32
    exact row_64
    exact row_65
  -- chapter_150_line_30: GL tag task formulation.
  have row_30 : (succ v1 v2) := by
    exact assumption_30
  have rule_row_29 := peano_source_010 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_150_line_29: GL tag implication.
  have row_29 : (add one v1 v2) := by
    apply rule_row_29
    exact row_30
  have rule_row_28 := peano_source_065 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_150_line_28: GL tag implication.
  have row_28 : (gl_inSequence N one succ) := by
    apply rule_row_28
  -- chapter_150_line_27: GL tag expansion.
  have row_27 : (¬ (∀ (v5 : α), ((N v5) → (¬ (succ v5 one))))) := by
    simpa only [gl_inSequence] using (row_28)
  have exists_row_27 : ∃ (v5 : α), ((N v5) ∧ (succ v5 one)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_27
      intro v5 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v5, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v5, witness_row_27⟩ := exists_row_27
  -- chapter_150_line_26: GL tag disintegration.
  have row_26 : (succ v5 one) := by
    exact witness_row_27.2
  -- chapter_150_line_25: GL tag disintegration.
  have row_25 : (N v5) := by
    exact witness_row_27.1
  have rule_row_24 := peano_source_054 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_150_line_24: GL tag implication.
  have row_24 : (mul v2 v1 v3) := by
    apply rule_row_24
    exact row_73
  -- chapter_150_line_23: GL tag implication grouping.
  have row_23 : (∀ (w1 : α) (w2 : α) (w3 : α), ((mul w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_55
    have auxiliary_source_2 := row_57
    have auxiliary_source_3 := row_59
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_150_line_22: GL tag implication.
  have row_22 : (gl_and21 v2 N v1 N v3 N) := by
    apply row_23
    exact row_24
  -- chapter_150_line_21: GL tag expansion.
  have row_21 : (((N v2) ∧ (N v1)) ∧ (N v3)) := by
    simpa only [gl_and21] using (row_22)
  -- chapter_150_line_20: GL tag disintegration.
  have row_20 : (N v2) := by
    exact row_21.1.1
  have rule_row_19 := peano_source_048 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_150_line_19: GL tag implication.
  have row_19 : (zero = v3) := by
    apply rule_row_19
    exact row_20
    exact row_72
  -- chapter_150_line_18: GL tag implication grouping.
  have row_18 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (gl_and20 w1 N w2 N))) := by
    have auxiliary_source_1 := row_49
    have auxiliary_source_2 := row_51
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_premise
    exact ⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_premise)⟩
  -- chapter_150_line_17: GL tag implication grouping.
  have row_17 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_41
    have auxiliary_source_2 := row_43
    have auxiliary_source_3 := row_45
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_150_line_16: GL tag implication.
  have row_16 : (gl_and21 one N v1 N v2 N) := by
    apply row_17
    exact row_29
  -- chapter_150_line_15: GL tag expansion.
  have row_15 : (((N one) ∧ (N v1)) ∧ (N v2)) := by
    simpa only [gl_and21] using (row_16)
  -- chapter_150_line_14: GL tag disintegration.
  have row_14 : (N one) := by
    exact row_15.1.1
  -- chapter_150_line_13: GL tag implication.
  have row_13 : (zero = v5) := by
    apply row_39
    exact row_14
    exact row_65
    exact row_26
  -- chapter_150_line_12: GL tag implication.
  have row_12 : (identity zero v5) := by
    apply row_66
    exact row_38
    exact row_13
  -- chapter_150_line_11: GL tag implication.
  have row_11 : (V2 zero v5) := by
    apply row_88
    exact row_12
  -- chapter_150_line_10: GL tag implication.
  have row_10 : (v5 = zero) := by
    apply row_39
    exact row_14
    exact row_26
    exact row_65
  -- chapter_150_line_9: GL tag symmetry of equality.
  have row_9 : (zero = v1) := by
    exact Eq.symm row_74
  -- chapter_150_line_8: GL tag equality1.
  have row_8 : (V2 v1 zero) := by
    have equality_source := row_11
    have equality_step_1 := row_9
    cases equality_step_1
    have equality_step_2 := row_10
    cases equality_step_2
    exact equality_source
  -- chapter_150_line_7: GL tag implication.
  have row_7 : (zero = v4) := by
    apply row_79
    exact row_75
    exact row_8
    exact row_81
  -- chapter_150_line_6: GL tag implication.
  have row_6 : (gl_and20 one N two N) := by
    apply row_18
    exact row_37
  -- chapter_150_line_5: GL tag expansion.
  have row_5 : ((N one) ∧ (N two)) := by
    simpa only [gl_and20] using (row_6)
  -- chapter_150_line_4: GL tag disintegration.
  have row_4 : (N two) := by
    exact row_5.2
  -- chapter_150_line_3: GL tag implication.
  have row_3 : (mul two zero v5) := by
    apply row_35
    exact row_10
    exact row_4
    exact row_25
  -- chapter_150_line_2: GL tag equality1.
  have row_2 : (mul two zero zero) := by
    have equality_source := row_3
    have equality_step_1 := row_10
    cases equality_step_1
    exact equality_source
  -- chapter_150_line_1: GL tag equality1.
  have row_1 : (mul two v4 v3) := by
    have equality_source := row_2
    have equality_step_1 := row_19
    cases equality_step_1
    have equality_step_2 := row_7
    cases equality_step_2
    exact equality_source
  exact row_1

private theorem gauss_source_079_check_induction_condition
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
    (assumption_189 : (gl_fold N succ add add identity zero v1 v4))
    (assumption_168 : (succ previous v1))
    (assumption_71 : (∀ (w1 : α) (w2 : α), ((mul previous w1 w2) → (∀ (w3 : α), ((gl_fold N succ add add identity zero previous w3) → ((succ previous w1) → (mul two w3 w2)))))))
    (assumption_13 : (succ v1 v2))
    (assumption_9 : (mul v1 v2 v3))
    : (mul two v4 v3) := by
  -- chapter_151_line_189: GL tag task formulation.
  have row_189 : (gl_fold N succ add add identity zero v1 v4) := by
    exact assumption_189
  -- chapter_151_line_188: GL tag expansion.
  have row_188 : (¬ (∀ (V1 : GLSet α), ((gl_interval N add zero v1 V1) → (¬ (gl_existence2 V1 N zero identity v1 v4 succ add))))) := by
    simpa only [gl_fold] using (row_189)
  have exists_row_188 : ∃ (V1 : GLSet α), ((gl_interval N add zero v1 V1) ∧ (gl_existence2 V1 N zero identity v1 v4 succ add)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_188
      intro V1 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨V1, counterexample_premise_0, counterexample_head⟩
  obtain ⟨V1, witness_row_188⟩ := exists_row_188
  -- chapter_151_line_187: GL tag disintegration.
  have row_187 : (gl_existence2 V1 N zero identity v1 v4 succ add) := by
    exact witness_row_188.2
  -- chapter_151_line_186: GL tag expansion.
  have row_186 : (¬ (∀ (V4 : GLBinaryRelation α), ((gl_fXY V4 V1 N) → (¬ (gl_and0 zero identity V4 v1 v4 V1 succ add))))) := by
    simpa only [gl_existence2] using (row_187)
  have exists_row_186 : ∃ (V4 : GLBinaryRelation α), ((gl_fXY V4 V1 N) ∧ (gl_and0 zero identity V4 v1 v4 V1 succ add)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_186
      intro V4 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨V4, counterexample_premise_0, counterexample_head⟩
  obtain ⟨V4, witness_row_186⟩ := exists_row_186
  -- chapter_151_line_185: GL tag disintegration.
  have row_185 : (gl_and0 zero identity V4 v1 v4 V1 succ add) := by
    exact witness_row_186.2
  -- chapter_151_line_184: GL tag expansion.
  have row_184 : (((gl_implication121 zero identity V4) ∧ (V4 v1 v4)) ∧ (gl_implication122 V1 succ V4 identity add)) := by
    simpa only [gl_and0] using (row_185)
  -- chapter_151_line_183: GL tag disintegration.
  have row_183 : (gl_implication122 V1 succ V4 identity add) := by
    exact row_184.2
  -- chapter_151_line_182: GL tag expansion.
  have row_182 : (∀ (w1 : α), ((V1 w1) → (∀ (w2 : α), ((succ w1 w2) → ((V1 w2) → (∀ (w3 : α), ((V4 w1 w3) → (∀ (w4 : α), ((identity w2 w4) → (∀ (w5 : α), ((add w3 w4 w5) → (V4 w2 w5)))))))))))) := by
    simpa only [gl_implication122] using (row_183)
  -- chapter_151_line_181: GL tag disintegration.
  have row_181 : (gl_implication121 zero identity V4) := by
    exact row_184.1.1
  -- chapter_151_line_180: GL tag expansion.
  have row_180 : (∀ (w1 : α), ((identity zero w1) → (V4 zero w1))) := by
    simpa only [gl_implication121] using (row_181)
  -- chapter_151_line_179: GL tag disintegration.
  have row_179 : (V4 v1 v4) := by
    exact row_184.1.2
  -- chapter_151_line_178: GL tag disintegration.
  have row_178 : (gl_fXY V4 V1 N) := by
    exact witness_row_186.1
  -- chapter_151_line_177: GL tag expansion.
  have row_177 : ((((gl_implication0 V4 V1) ∧ (gl_implication1 V4 N)) ∧ (gl_implication6 V1 N V4)) ∧ (gl_implication7 V1 V4)) := by
    simpa only [gl_fXY] using (row_178)
  -- chapter_151_line_176: GL tag disintegration.
  have row_176 : (gl_implication0 V4 V1) := by
    exact row_177.1.1.1
  -- chapter_151_line_175: GL tag expansion.
  have row_175 : (∀ (w1 : α) (w2 : α), ((V4 w1 w2) → (V1 w1))) := by
    simpa only [gl_implication0] using (row_176)
  -- chapter_151_line_174: GL tag disintegration.
  have row_174 : (gl_implication7 V1 V4) := by
    exact row_177.2
  -- chapter_151_line_173: GL tag expansion.
  have row_173 : (∀ (w1 : α), ((V1 w1) → (∀ (w2 : α), ((V4 w1 w2) → (∀ (w3 : α), ((V4 w1 w3) → (w2 = w3))))))) := by
    simpa only [gl_implication7] using (row_174)
  -- chapter_151_line_172: GL tag disintegration.
  have row_172 : (gl_interval N add zero v1 V1) := by
    exact witness_row_188.1
  -- chapter_151_line_171: GL tag expansion.
  have row_171 : (((((gl_implication89 V1 N add zero) ∧ (gl_implication90 V1 N add v1)) ∧ (gl_implication91 N add zero v1 V1)) ∧ (N zero)) ∧ (N v1)) := by
    simpa only [gl_interval] using (row_172)
  -- chapter_151_line_170: GL tag disintegration.
  have row_170 : (N v1) := by
    exact row_171.2
  -- chapter_151_line_169: GL tag implication.
  have row_169 : (V1 v1) := by
    apply row_175
    exact row_179
  -- chapter_151_line_168: GL tag recursion.
  have row_168 : (succ previous v1) := by
    exact assumption_168
  -- chapter_151_line_167: GL tag task formulation.
  have row_167 : (gl_AnchorGauss N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_151_line_166: GL tag theorem.
  have row_166 := gauss_source_074 N zero succ add mul one two identity anchor relationalInduction strongRelationalInduction
  -- chapter_151_line_165: GL tag implication.
  have row_165 : (gl_existence8 N add V1 previous zero) := by
    apply row_166
    exact row_168
    exact row_172
  -- chapter_151_line_164: GL tag expansion.
  have row_164 : (¬ (∀ (V2 : GLSet α), ((gl_limitSet N add V1 previous V2) → (¬ (gl_interval N add zero previous V2))))) := by
    simpa only [gl_existence8] using (row_165)
  have exists_row_164 : ∃ (V2 : GLSet α), ((gl_limitSet N add V1 previous V2) ∧ (gl_interval N add zero previous V2)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_164
      intro V2 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨V2, counterexample_premise_0, counterexample_head⟩
  obtain ⟨V2, witness_row_164⟩ := exists_row_164
  -- chapter_151_line_163: GL tag disintegration.
  have row_163 : (gl_limitSet N add V1 previous V2) := by
    exact witness_row_164.1
  -- chapter_151_line_162: GL tag expansion.
  have row_162 : (((gl_subset V2 V1) ∧ (gl_implication90 V2 N add previous)) ∧ (gl_implication141 V1 N add previous V2)) := by
    simpa only [gl_limitSet] using (row_163)
  -- chapter_151_line_161: GL tag disintegration.
  have row_161 : (gl_implication90 V2 N add previous) := by
    exact row_162.1.2
  -- chapter_151_line_160: GL tag expansion.
  have row_160 : (∀ (w1 : α), ((V2 w1) → (gl_preorder N add w1 previous))) := by
    simpa only [gl_implication90] using (row_161)
  -- chapter_151_line_159: GL tag disintegration.
  have row_159 : (gl_subset V2 V1) := by
    exact row_162.1.1
  -- chapter_151_line_158: GL tag expansion.
  have row_158 : (∀ (w1 : α), ((V2 w1) → (V1 w1))) := by
    simpa only [gl_subset] using (row_159)
  -- chapter_151_line_157: GL tag disintegration.
  have row_157 : (gl_interval N add zero previous V2) := by
    exact witness_row_164.2
  -- chapter_151_line_156: GL tag expansion.
  have row_156 : (((((gl_implication89 V2 N add zero) ∧ (gl_implication90 V2 N add previous)) ∧ (gl_implication91 N add zero previous V2)) ∧ (N zero)) ∧ (N previous)) := by
    simpa only [gl_interval] using (row_157)
  -- chapter_151_line_155: GL tag disintegration.
  have row_155 : (gl_implication89 V2 N add zero) := by
    exact row_156.1.1.1.1
  -- chapter_151_line_154: GL tag expansion.
  have row_154 : (∀ (w1 : α), ((V2 w1) → (gl_preorder N add zero w1))) := by
    simpa only [gl_implication89] using (row_155)
  -- chapter_151_line_153: GL tag theorem.
  have row_153 := gauss_source_069 N zero succ add mul one two identity anchor relationalInduction strongRelationalInduction
  -- chapter_151_line_152: GL tag implication.
  have row_152 : (V2 previous) := by
    apply row_153
    exact row_157
  -- chapter_151_line_151: GL tag theorem.
  have row_151 := gauss_source_077 N zero succ add mul one two identity anchor relationalInduction strongRelationalInduction
  -- chapter_151_line_150: GL tag implication.
  have row_150 : (gl_sequence N add zero v1 V4) := by
    apply row_151
    exact row_178
    exact row_172
  -- chapter_151_line_149: GL tag theorem.
  have row_149 := gauss_source_076 N zero succ add mul one two identity anchor relationalInduction strongRelationalInduction
  -- chapter_151_line_148: GL tag implication.
  have row_148 : (gl_existence12 N add previous V4 zero) := by
    apply row_149
    exact row_168
    exact row_150
  -- chapter_151_line_147: GL tag expansion.
  have row_147 : (¬ (∀ (V3 : GLBinaryRelation α), ((gl_limitSequence N add previous V4 V3) → (¬ (gl_sequence N add zero previous V3))))) := by
    simpa only [gl_existence12] using (row_148)
  have exists_row_147 : ∃ (V3 : GLBinaryRelation α), ((gl_limitSequence N add previous V4 V3) ∧ (gl_sequence N add zero previous V3)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_147
      intro V3 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨V3, counterexample_premise_0, counterexample_head⟩
  obtain ⟨V3, witness_row_147⟩ := exists_row_147
  -- chapter_151_line_146: GL tag disintegration.
  have row_146 : (gl_limitSequence N add previous V4 V3) := by
    exact witness_row_147.1
  -- chapter_151_line_145: GL tag expansion.
  have row_145 : (((gl_implication136 V3 N add previous) ∧ (gl_implication137 V3 V4)) ∧ (gl_implication138 N add previous V4 V3)) := by
    simpa only [gl_limitSequence] using (row_146)
  -- chapter_151_line_144: GL tag disintegration.
  have row_144 : (gl_implication137 V3 V4) := by
    exact row_145.1.2
  -- chapter_151_line_143: GL tag expansion.
  have row_143 : (∀ (w1 : α) (w2 : α), ((V3 w1 w2) → (V4 w1 w2))) := by
    simpa only [gl_implication137] using (row_144)
  -- chapter_151_line_142: GL tag disintegration.
  have row_142 : (gl_sequence N add zero previous V3) := by
    exact witness_row_147.2
  -- chapter_151_line_141: GL tag disintegration.
  have row_141 : (gl_implication138 N add previous V4 V3) := by
    exact row_145.2
  -- chapter_151_line_140: GL tag expansion.
  have row_140 : (∀ (w1 : α), ((gl_preorder N add w1 previous) → (∀ (w2 : α), ((V4 w1 w2) → (V3 w1 w2))))) := by
    simpa only [gl_implication138] using (row_141)
  -- chapter_151_line_139: GL tag expansion.
  have row_139 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorGauss] using (row_167)
  -- chapter_151_line_138: GL tag disintegration.
  have row_138 : (gl_identity N identity) := by
    exact row_139.2
  -- chapter_151_line_137: GL tag expansion.
  have row_137 : (((gl_implication0 identity N) ∧ (gl_implication37 identity)) ∧ (gl_implication38 N identity)) := by
    simpa only [gl_identity] using (row_138)
  -- chapter_151_line_136: GL tag disintegration.
  have row_136 : (gl_implication37 identity) := by
    exact row_137.1.2
  -- chapter_151_line_135: GL tag expansion.
  have row_135 : (∀ (w1 : α) (w2 : α), ((identity w1 w2) → (w1 = w2))) := by
    simpa only [gl_implication37] using (row_136)
  -- chapter_151_line_134: GL tag disintegration.
  have row_134 : (gl_implication38 N identity) := by
    exact row_137.2
  -- chapter_151_line_133: GL tag expansion.
  have row_133 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((w1 = w2) → (identity w1 w2))))) := by
    simpa only [gl_implication38] using (row_134)
  -- chapter_151_line_132: GL tag disintegration.
  have row_132 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_139.1.1.1
  -- chapter_151_line_131: GL tag expansion.
  have row_131 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_132)
  -- chapter_151_line_130: GL tag disintegration.
  have row_130 : (gl_fXYZ add N N N) := by
    exact row_131.1.1.1.1.1.1.1.1.1.2
  -- chapter_151_line_129: GL tag expansion.
  have row_129 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_130)
  -- chapter_151_line_128: GL tag disintegration.
  have row_128 : (gl_implication18 N N N add) := by
    exact row_129.1.2
  -- chapter_151_line_127: GL tag expansion.
  have row_127 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication18] using (row_128)
  -- chapter_151_line_126: GL tag disintegration.
  have row_126 : (gl_implication21 N zero add) := by
    exact row_131.1.1.1.1.1.1.1.1.2
  -- chapter_151_line_125: GL tag expansion.
  have row_125 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((add w1 zero w2) → (w1 = w2))))) := by
    simpa only [gl_implication21] using (row_126)
  -- chapter_151_line_124: GL tag disintegration.
  have row_124 : (N zero) := by
    exact row_131.1.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_151_line_123: GL tag disintegration.
  have row_123 : (gl_implication10 N succ) := by
    exact row_131.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_151_line_122: GL tag expansion.
  have row_122 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w2 w1) → (∀ (w3 : α), ((succ w3 w1) → (w2 = w3))))))) := by
    simpa only [gl_implication10] using (row_123)
  -- chapter_151_line_121: GL tag disintegration.
  have row_121 : (gl_fXY succ N N) := by
    exact row_131.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_151_line_120: GL tag expansion.
  have row_120 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_121)
  -- chapter_151_line_119: GL tag disintegration.
  have row_119 : (gl_implication0 succ N) := by
    exact row_120.1.1.1
  -- chapter_151_line_118: GL tag expansion.
  have row_118 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_119)
  -- chapter_151_line_117: GL tag disintegration.
  have row_117 : (gl_implication1 succ N) := by
    exact row_120.1.1.2
  -- chapter_151_line_116: GL tag expansion.
  have row_116 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using (row_117)
  -- chapter_151_line_115: GL tag disintegration.
  have row_115 : (gl_fXYZ mul N N N) := by
    exact row_131.1.1.1.1.2
  -- chapter_151_line_114: GL tag expansion.
  have row_114 : (((((gl_implication11 mul N) ∧ (gl_implication12 mul N)) ∧ (gl_implication13 mul N)) ∧ (gl_implication18 N N N mul)) ∧ (gl_implication19 N N mul)) := by
    simpa only [gl_fXYZ] using (row_115)
  -- chapter_151_line_113: GL tag disintegration.
  have row_113 : (gl_implication18 N N N mul) := by
    exact row_114.1.2
  -- chapter_151_line_112: GL tag expansion.
  have row_112 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication18] using (row_113)
  -- chapter_151_line_111: GL tag disintegration.
  have row_111 : (succ one two) := by
    exact row_139.1.2
  -- chapter_151_line_110: GL tag disintegration.
  have row_110 : (succ zero one) := by
    exact row_139.1.1.2
  -- chapter_151_line_109: GL tag disintegration.
  have row_109 : (gl_implication24 N succ add) := by
    exact row_131.1.1.1.1.1.2
  -- chapter_151_line_108: GL tag expansion.
  have row_108 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((succ w4 w5) → (add w3 w2 w5))))))))) := by
    simpa only [gl_implication24] using (row_109)
  -- chapter_151_line_107: GL tag disintegration.
  have row_107 : (gl_implication23 N succ add) := by
    exact row_131.1.1.1.1.1.1.2
  -- chapter_151_line_106: GL tag expansion.
  have row_106 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication23] using (row_107)
  -- chapter_151_line_105: GL tag implication.
  have row_105 : (gl_existence1 N v1 zero add) := by
    apply row_127
    exact row_170
    exact row_124
  -- chapter_151_line_104: GL tag expansion.
  have row_104 : (¬ (∀ (v25 : α), ((N v25) → (¬ (add v1 zero v25))))) := by
    simpa only [gl_existence1] using (row_105)
  have exists_row_104 : ∃ (v25 : α), ((N v25) ∧ (add v1 zero v25)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_104
      intro v25 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v25, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v25, witness_row_104⟩ := exists_row_104
  -- chapter_151_line_103: GL tag disintegration.
  have row_103 : (add v1 zero v25) := by
    exact witness_row_104.2
  -- chapter_151_line_102: GL tag implication.
  have row_102 : (v1 = v25) := by
    apply row_125
    exact row_170
    exact row_103
  -- chapter_151_line_101: GL tag implication.
  have row_101 : (N previous) := by
    apply row_118
    exact row_168
  -- chapter_151_line_100: GL tag implication.
  have row_100 : (gl_existence1 N previous zero add) := by
    apply row_127
    exact row_101
    exact row_124
  -- chapter_151_line_99: GL tag expansion.
  have row_99 : (¬ (∀ (v23 : α), ((N v23) → (¬ (add previous zero v23))))) := by
    simpa only [gl_existence1] using (row_100)
  have exists_row_99 : ∃ (v23 : α), ((N v23) ∧ (add previous zero v23)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_99
      intro v23 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v23, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v23, witness_row_99⟩ := exists_row_99
  -- chapter_151_line_98: GL tag disintegration.
  have row_98 : (add previous zero v23) := by
    exact witness_row_99.2
  -- chapter_151_line_97: GL tag implication.
  have row_97 : (previous = v23) := by
    apply row_125
    exact row_101
    exact row_98
  -- chapter_151_line_96: GL tag equality1.
  have row_96 : (gl_interval N add zero v23 V2) := by
    have equality_source := row_157
    have equality_step_1 := row_97
    cases equality_step_1
    exact equality_source
  -- chapter_151_line_95: GL tag implication.
  have row_95 : (V2 v23) := by
    apply row_153
    exact row_96
  -- chapter_151_line_94: GL tag implication.
  have row_94 : (V1 v23) := by
    apply row_158
    exact row_95
  -- chapter_151_line_93: GL tag equality1.
  have row_93 : (gl_sequence N add zero v23 V3) := by
    have equality_source := row_142
    have equality_step_1 := row_97
    cases equality_step_1
    exact equality_source
  -- chapter_151_line_92: GL tag theorem.
  have row_92 := gauss_source_078 N zero succ add mul one two identity anchor relationalInduction strongRelationalInduction
  -- chapter_151_line_91: GL tag implication.
  have row_91 : (gl_fXY V3 V2 N) := by
    apply row_92
    exact row_96
    exact row_93
  -- chapter_151_line_90: GL tag implication.
  have row_90 : (identity v1 v25) := by
    apply row_133
    exact row_170
    exact row_102
  -- chapter_151_line_89: GL tag expansion.
  have row_89 : ((((gl_implication0 V3 V2) ∧ (gl_implication1 V3 N)) ∧ (gl_implication6 V2 N V3)) ∧ (gl_implication7 V2 V3)) := by
    simpa only [gl_fXY] using (row_91)
  -- chapter_151_line_88: GL tag disintegration.
  have row_88 : (gl_implication6 V2 N V3) := by
    exact row_89.1.2
  -- chapter_151_line_87: GL tag expansion.
  have row_87 : (∀ (w1 : α), ((V2 w1) → (gl_existence0 N w1 V3))) := by
    simpa only [gl_implication6] using (row_88)
  -- chapter_151_line_86: GL tag implication.
  have row_86 : (gl_existence0 N previous V3) := by
    apply row_87
    exact row_152
  -- chapter_151_line_85: GL tag expansion.
  have row_85 : (¬ (∀ (v6 : α), ((N v6) → (¬ (V3 previous v6))))) := by
    simpa only [gl_existence0] using (row_86)
  have exists_row_85 : ∃ (v6 : α), ((N v6) ∧ (V3 previous v6)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_85
      intro v6 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v6, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v6, witness_row_85⟩ := exists_row_85
  -- chapter_151_line_84: GL tag disintegration.
  have row_84 : (N v6) := by
    exact witness_row_85.1
  -- chapter_151_line_83: GL tag disintegration.
  have row_83 : (V3 previous v6) := by
    exact witness_row_85.2
  -- chapter_151_line_82: GL tag implication.
  have row_82 : (gl_existence1 N v6 v1 add) := by
    apply row_127
    exact row_84
    exact row_170
  -- chapter_151_line_81: GL tag expansion.
  have row_81 : (¬ (∀ (v24 : α), ((N v24) → (¬ (add v6 v1 v24))))) := by
    simpa only [gl_existence1] using (row_82)
  have exists_row_81 : ∃ (v24 : α), ((N v24) ∧ (add v6 v1 v24)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_81
      intro v24 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v24, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v24, witness_row_81⟩ := exists_row_81
  -- chapter_151_line_80: GL tag disintegration.
  have row_80 : (add v6 v1 v24) := by
    exact witness_row_81.2
  -- chapter_151_line_79: GL tag theorem.
  have row_79 := gauss_source_070 N zero succ add mul one two identity anchor relationalInduction strongRelationalInduction
  -- chapter_151_line_78: GL tag implication.
  have row_78 : (add v1 v6 v24) := by
    apply row_79
    exact row_80
  -- chapter_151_line_77: GL tag equality1.
  have row_77 : (V3 v23 v6) := by
    have equality_source := row_83
    have equality_step_1 := row_97
    cases equality_step_1
    exact equality_source
  -- chapter_151_line_76: GL tag implication.
  have row_76 : (V4 v23 v6) := by
    apply row_143
    exact row_77
  -- chapter_151_line_75: GL tag implication grouping.
  have row_75 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (gl_and20 w1 N w2 N))) := by
    have auxiliary_source_1 := row_118
    have auxiliary_source_2 := row_116
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_premise
    exact ⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_premise)⟩
  -- chapter_151_line_74: GL tag implication.
  have row_74 : (gl_existence1 N previous v1 mul) := by
    apply row_112
    exact row_101
    exact row_170
  -- chapter_151_line_73: GL tag expansion.
  have row_73 : (¬ (∀ (v7 : α), ((N v7) → (¬ (mul previous v1 v7))))) := by
    simpa only [gl_existence1] using (row_74)
  have exists_row_73 : ∃ (v7 : α), ((N v7) ∧ (mul previous v1 v7)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_73
      intro v7 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v7, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v7, witness_row_73⟩ := exists_row_73
  -- chapter_151_line_72: GL tag disintegration.
  have row_72 : (mul previous v1 v7) := by
    exact witness_row_73.2
  -- chapter_151_line_71: GL tag recursion.
  have row_71 : (∀ (w1 : α) (w2 : α), ((mul previous w1 w2) → (∀ (w3 : α), ((gl_fold N succ add add identity zero previous w3) → ((succ previous w1) → (mul two w3 w2)))))) := by
    exact assumption_71
  -- chapter_151_line_70: GL tag implication.
  have row_70 : (gl_and20 one N two N) := by
    apply row_75
    exact row_111
  -- chapter_151_line_69: GL tag expansion.
  have row_69 : ((N one) ∧ (N two)) := by
    simpa only [gl_and20] using (row_70)
  -- chapter_151_line_68: GL tag disintegration.
  have row_68 : (N two) := by
    exact row_69.2
  -- chapter_151_line_67: GL tag implication.
  have row_67 : (gl_existence1 N v1 two mul) := by
    apply row_112
    exact row_170
    exact row_68
  -- chapter_151_line_66: GL tag expansion.
  have row_66 : (¬ (∀ (v5 : α), ((N v5) → (¬ (mul v1 two v5))))) := by
    simpa only [gl_existence1] using (row_67)
  have exists_row_66 : ∃ (v5 : α), ((N v5) ∧ (mul v1 two v5)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_66
      intro v5 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v5, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v5, witness_row_66⟩ := exists_row_66
  -- chapter_151_line_65: GL tag disintegration.
  have row_65 : (mul v1 two v5) := by
    exact witness_row_66.2
  -- chapter_151_line_64: GL tag implication.
  have row_64 : (gl_and20 zero N one N) := by
    apply row_75
    exact row_110
  -- chapter_151_line_63: GL tag expansion.
  have row_63 : ((N zero) ∧ (N one)) := by
    simpa only [gl_and20] using (row_64)
  -- chapter_151_line_62: GL tag disintegration.
  have row_62 : (N one) := by
    exact row_63.2
  -- chapter_151_line_61: GL tag expansion for integration.
  have row_61 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_151_line_60_rule: GL tag reformulation for integration and.
  have row_60_rule : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_151_line_60: GL tag reformulation for integration and.
  have row_60 : (gl_implication76 N zero succ add mul one) := by
    simpa only [gl_implication76] using row_60_rule
  -- chapter_151_line_59: GL tag expansion.
  have row_59 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    simpa only [gl_implication76] using (row_60)
  -- chapter_151_line_58: GL tag implication.
  have row_58 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_59
    exact row_132
    exact row_110
  -- chapter_151_line_57: GL tag expansion for integration.
  have row_57 : ((gl_implication122 V2 succ V3 identity add) ↔ (∀ (v20 : α), ((V2 v20) → (∀ (v16 : α), ((succ v20 v16) → ((V2 v16) → (∀ (v21 : α), ((V3 v20 v21) → (∀ (v22 : α), ((identity v16 v22) → (∀ (v17 : α), ((add v21 v22 v17) → (V3 v16 v17))))))))))))) := by
    exact Iff.rfl
  -- chapter_151_line_56: GL tag expansion for integration.
  have row_56 : ((gl_implication121 zero identity V3) ↔ (∀ (v13 : α), ((identity zero v13) → (V3 zero v13)))) := by
    exact Iff.rfl
  -- chapter_151_line_55: GL tag implication.
  have row_55 : (gl_preorder N add zero previous) := by
    apply row_154
    exact row_152
  -- chapter_151_line_54: GL tag premise element.
  have row_54 : (∀ (v20 : α), ((V2 v20) → (∀ (v16 : α), ((succ v20 v16) → ((V2 v16) → (∀ (v21 : α), ((V3 v20 v21) → (∀ (v22 : α), ((identity v16 v22) → (∀ (v17 : α), ((add v21 v22 v17) → (V2 v16)))))))))))) := by
    intro v20
    intro scope_premise_1
    intro v16
    intro scope_premise_2
    intro scope_premise_3
    intro v21
    intro scope_premise_4
    intro v22
    intro scope_premise_5
    intro v17
    intro scope_premise_6
    exact scope_premise_3
  -- chapter_151_line_53: GL tag implication.
  have row_53 : (∀ (v20 : α), ((V2 v20) → (∀ (v16 : α), ((succ v20 v16) → ((V2 v16) → (∀ (v21 : α), ((V3 v20 v21) → (∀ (v22 : α), ((identity v16 v22) → (∀ (v17 : α), ((add v21 v22 v17) → (V1 v16)))))))))))) := by
    intro v20
    intro scope_premise_1
    intro v16
    intro scope_premise_2
    intro scope_premise_3
    intro v21
    intro scope_premise_4
    intro v22
    intro scope_premise_5
    intro v17
    intro scope_premise_6
    have scoped_fact_2 := row_54 v20 scope_premise_1 v16 scope_premise_2 scope_premise_3 v21 scope_premise_4 v22 scope_premise_5 v17 scope_premise_6
    apply row_158
    exact scoped_fact_2
  -- chapter_151_line_52: GL tag premise element.
  have row_52 : (∀ (v20 : α), ((V2 v20) → (∀ (v16 : α), ((succ v20 v16) → ((V2 v16) → (∀ (v21 : α), ((V3 v20 v21) → (∀ (v22 : α), ((identity v16 v22) → (∀ (v17 : α), ((add v21 v22 v17) → (V2 v20)))))))))))) := by
    intro v20
    intro scope_premise_1
    intro v16
    intro scope_premise_2
    intro scope_premise_3
    intro v21
    intro scope_premise_4
    intro v22
    intro scope_premise_5
    intro v17
    intro scope_premise_6
    exact scope_premise_1
  -- chapter_151_line_51: GL tag implication.
  have row_51 : (∀ (v20 : α), ((V2 v20) → (∀ (v16 : α), ((succ v20 v16) → ((V2 v16) → (∀ (v21 : α), ((V3 v20 v21) → (∀ (v22 : α), ((identity v16 v22) → (∀ (v17 : α), ((add v21 v22 v17) → (V1 v20)))))))))))) := by
    intro v20
    intro scope_premise_1
    intro v16
    intro scope_premise_2
    intro scope_premise_3
    intro v21
    intro scope_premise_4
    intro v22
    intro scope_premise_5
    intro v17
    intro scope_premise_6
    have scoped_fact_2 := row_52 v20 scope_premise_1 v16 scope_premise_2 scope_premise_3 v21 scope_premise_4 v22 scope_premise_5 v17 scope_premise_6
    apply row_158
    exact scoped_fact_2
  -- chapter_151_line_50: GL tag premise element.
  have row_50 : (∀ (v20 : α), ((V2 v20) → (∀ (v16 : α), ((succ v20 v16) → ((V2 v16) → (∀ (v21 : α), ((V3 v20 v21) → (∀ (v22 : α), ((identity v16 v22) → (∀ (v17 : α), ((add v21 v22 v17) → (V3 v20 v21)))))))))))) := by
    intro v20
    intro scope_premise_1
    intro v16
    intro scope_premise_2
    intro scope_premise_3
    intro v21
    intro scope_premise_4
    intro v22
    intro scope_premise_5
    intro v17
    intro scope_premise_6
    exact scope_premise_4
  -- chapter_151_line_49: GL tag implication.
  have row_49 : (∀ (v20 : α), ((V2 v20) → (∀ (v16 : α), ((succ v20 v16) → ((V2 v16) → (∀ (v21 : α), ((V3 v20 v21) → (∀ (v22 : α), ((identity v16 v22) → (∀ (v17 : α), ((add v21 v22 v17) → (V4 v20 v21)))))))))))) := by
    intro v20
    intro scope_premise_1
    intro v16
    intro scope_premise_2
    intro scope_premise_3
    intro v21
    intro scope_premise_4
    intro v22
    intro scope_premise_5
    intro v17
    intro scope_premise_6
    have scoped_fact_2 := row_50 v20 scope_premise_1 v16 scope_premise_2 scope_premise_3 v21 scope_premise_4 v22 scope_premise_5 v17 scope_premise_6
    apply row_143
    exact scoped_fact_2
  -- chapter_151_line_48: GL tag premise element.
  have row_48 : (∀ (v20 : α), ((V2 v20) → (∀ (v16 : α), ((succ v20 v16) → ((V2 v16) → (∀ (v21 : α), ((V3 v20 v21) → (∀ (v22 : α), ((identity v16 v22) → (∀ (v17 : α), ((add v21 v22 v17) → (identity v16 v22)))))))))))) := by
    intro v20
    intro scope_premise_1
    intro v16
    intro scope_premise_2
    intro scope_premise_3
    intro v21
    intro scope_premise_4
    intro v22
    intro scope_premise_5
    intro v17
    intro scope_premise_6
    exact scope_premise_5
  -- chapter_151_line_47: GL tag premise element.
  have row_47 : (∀ (v20 : α), ((V2 v20) → (∀ (v16 : α), ((succ v20 v16) → ((V2 v16) → (∀ (v21 : α), ((V3 v20 v21) → (∀ (v22 : α), ((identity v16 v22) → (∀ (v17 : α), ((add v21 v22 v17) → (add v21 v22 v17)))))))))))) := by
    intro v20
    intro scope_premise_1
    intro v16
    intro scope_premise_2
    intro scope_premise_3
    intro v21
    intro scope_premise_4
    intro v22
    intro scope_premise_5
    intro v17
    intro scope_premise_6
    exact scope_premise_6
  -- chapter_151_line_46: GL tag premise element.
  have row_46 : (∀ (v20 : α), ((V2 v20) → (∀ (v16 : α), ((succ v20 v16) → ((V2 v16) → (∀ (v21 : α), ((V3 v20 v21) → (∀ (v22 : α), ((identity v16 v22) → (∀ (v17 : α), ((add v21 v22 v17) → (succ v20 v16)))))))))))) := by
    intro v20
    intro scope_premise_1
    intro v16
    intro scope_premise_2
    intro scope_premise_3
    intro v21
    intro scope_premise_4
    intro v22
    intro scope_premise_5
    intro v17
    intro scope_premise_6
    exact scope_premise_2
  -- chapter_151_line_45: GL tag implication.
  have row_45 : (∀ (v20 : α), ((V2 v20) → (∀ (v16 : α), ((succ v20 v16) → ((V2 v16) → (∀ (v21 : α), ((V3 v20 v21) → (∀ (v22 : α), ((identity v16 v22) → (∀ (v17 : α), ((add v21 v22 v17) → (V4 v16 v17)))))))))))) := by
    intro v20
    intro scope_premise_1
    intro v16
    intro scope_premise_2
    intro scope_premise_3
    intro v21
    intro scope_premise_4
    intro v22
    intro scope_premise_5
    intro v17
    intro scope_premise_6
    have scoped_fact_2 := row_51 v20 scope_premise_1 v16 scope_premise_2 scope_premise_3 v21 scope_premise_4 v22 scope_premise_5 v17 scope_premise_6
    have scoped_fact_3 := row_46 v20 scope_premise_1 v16 scope_premise_2 scope_premise_3 v21 scope_premise_4 v22 scope_premise_5 v17 scope_premise_6
    have scoped_fact_4 := row_53 v20 scope_premise_1 v16 scope_premise_2 scope_premise_3 v21 scope_premise_4 v22 scope_premise_5 v17 scope_premise_6
    have scoped_fact_5 := row_49 v20 scope_premise_1 v16 scope_premise_2 scope_premise_3 v21 scope_premise_4 v22 scope_premise_5 v17 scope_premise_6
    have scoped_fact_6 := row_48 v20 scope_premise_1 v16 scope_premise_2 scope_premise_3 v21 scope_premise_4 v22 scope_premise_5 v17 scope_premise_6
    have scoped_fact_7 := row_47 v20 scope_premise_1 v16 scope_premise_2 scope_premise_3 v21 scope_premise_4 v22 scope_premise_5 v17 scope_premise_6
    apply row_182
    exact scoped_fact_2
    exact scoped_fact_3
    exact scoped_fact_4
    exact scoped_fact_5
    exact scoped_fact_6
    exact scoped_fact_7
  -- chapter_151_line_44: GL tag implication.
  have row_44 : (∀ (v20 : α), ((V2 v20) → (∀ (v16 : α), ((succ v20 v16) → ((V2 v16) → (∀ (v21 : α), ((V3 v20 v21) → (∀ (v22 : α), ((identity v16 v22) → (∀ (v17 : α), ((add v21 v22 v17) → (gl_preorder N add v16 previous)))))))))))) := by
    intro v20
    intro scope_premise_1
    intro v16
    intro scope_premise_2
    intro scope_premise_3
    intro v21
    intro scope_premise_4
    intro v22
    intro scope_premise_5
    intro v17
    intro scope_premise_6
    have scoped_fact_2 := row_54 v20 scope_premise_1 v16 scope_premise_2 scope_premise_3 v21 scope_premise_4 v22 scope_premise_5 v17 scope_premise_6
    apply row_160
    exact scoped_fact_2
  -- chapter_151_line_43: GL tag implication.
  have row_43 : (∀ (v20 : α), ((V2 v20) → (∀ (v16 : α), ((succ v20 v16) → ((V2 v16) → (∀ (v21 : α), ((V3 v20 v21) → (∀ (v22 : α), ((identity v16 v22) → (∀ (v17 : α), ((add v21 v22 v17) → (V3 v16 v17)))))))))))) := by
    intro v20
    intro scope_premise_1
    intro v16
    intro scope_premise_2
    intro scope_premise_3
    intro v21
    intro scope_premise_4
    intro v22
    intro scope_premise_5
    intro v17
    intro scope_premise_6
    have scoped_fact_2 := row_44 v20 scope_premise_1 v16 scope_premise_2 scope_premise_3 v21 scope_premise_4 v22 scope_premise_5 v17 scope_premise_6
    have scoped_fact_3 := row_45 v20 scope_premise_1 v16 scope_premise_2 scope_premise_3 v21 scope_premise_4 v22 scope_premise_5 v17 scope_premise_6
    apply row_140
    exact scoped_fact_2
    exact scoped_fact_3
  -- chapter_151_line_42: GL tag expansion for integration.
  have row_42 : ((gl_and0 zero identity V3 previous v6 V2 succ add) ↔ (((gl_implication121 zero identity V3) ∧ (V3 previous v6)) ∧ (gl_implication122 V2 succ V3 identity add))) := by
    exact Iff.rfl
  -- chapter_151_line_41_rule: GL tag reformulation for integration and.
  have row_41_rule : ((gl_implication121 zero identity V3) → ((V3 previous v6) → ((gl_implication122 V2 succ V3 identity add) → (gl_and0 zero identity V3 previous v6 V2 succ add)))) := by
    intro integration_premise_1
    intro integration_premise_2
    intro integration_premise_3
    simp only [gl_and0]
    exact ⟨⟨integration_premise_1, integration_premise_2⟩, integration_premise_3⟩
  -- chapter_151_line_41: GL tag reformulation for integration and.
  have row_41 : (gl_implication123 zero identity V3 previous v6 V2 succ add) := by
    simpa only [gl_implication123] using row_41_rule
  -- chapter_151_line_40: GL tag expansion.
  have row_40 : ((gl_implication121 zero identity V3) → ((V3 previous v6) → ((gl_implication122 V2 succ V3 identity add) → (gl_and0 zero identity V3 previous v6 V2 succ add)))) := by
    simpa only [gl_implication123] using (row_41)
  -- chapter_151_line_39: GL tag validity name.
  have row_39 : (gl_implication122 V2 succ V3 identity add) := by
    simpa only [gl_implication122] using (row_43)
  have rule_row_38 := peano_source_065 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_151_line_38: GL tag implication.
  have row_38 : (gl_inSequence N one succ) := by
    apply rule_row_38
  -- chapter_151_line_37: GL tag expansion.
  have row_37 : (¬ (∀ (v14 : α), ((N v14) → (¬ (succ v14 one))))) := by
    simpa only [gl_inSequence] using (row_38)
  have exists_row_37 : ∃ (v14 : α), ((N v14) ∧ (succ v14 one)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_37
      intro v14 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v14, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v14, witness_row_37⟩ := exists_row_37
  -- chapter_151_line_36: GL tag disintegration.
  have row_36 : (succ v14 one) := by
    exact witness_row_37.2
  -- chapter_151_line_35: GL tag implication.
  have row_35 : (v14 = zero) := by
    apply row_122
    exact row_62
    exact row_36
    exact row_110
  -- chapter_151_line_34: GL tag implication.
  have row_34 : (zero = v14) := by
    apply row_122
    exact row_62
    exact row_110
    exact row_36
  -- chapter_151_line_33: GL tag implication.
  have row_33 : (identity zero v14) := by
    apply row_133
    exact row_124
    exact row_34
  -- chapter_151_line_32: GL tag implication.
  have row_32 : (V4 zero v14) := by
    apply row_180
    exact row_33
  -- chapter_151_line_31: GL tag implication.
  have row_31 : (V3 zero v14) := by
    apply row_140
    exact row_55
    exact row_32
  -- chapter_151_line_30: GL tag equality1.
  have row_30 : (V3 zero zero) := by
    have equality_source := row_31
    have equality_step_1 := row_35
    cases equality_step_1
    exact equality_source
  -- chapter_151_line_29: GL tag premise element.
  have row_29 : (∀ (v13 : α), ((identity zero v13) → (identity zero v13))) := by
    intro v13
    intro scope_premise_1
    exact scope_premise_1
  -- chapter_151_line_28: GL tag implication.
  have row_28 : (∀ (v13 : α), ((identity zero v13) → (zero = v13))) := by
    intro v13
    intro scope_premise_1
    have scoped_fact_2 := row_29 v13 scope_premise_1
    apply row_135
    exact scoped_fact_2
  -- chapter_151_line_27: GL tag equality1.
  have row_27 : (∀ (v13 : α), ((identity zero v13) → (V3 zero v13))) := by
    intro v13
    intro scope_premise_1
    have scoped_fact_2 := row_28 v13 scope_premise_1
    have equality_source := row_30
    have equality_step_1 := scoped_fact_2
    cases equality_step_1
    exact equality_source
  -- chapter_151_line_26: GL tag validity name.
  have row_26 : (gl_implication121 zero identity V3) := by
    simpa only [gl_implication121] using (row_27)
  -- chapter_151_line_25: GL tag implication.
  have row_25 : (gl_and0 zero identity V3 previous v6 V2 succ add) := by
    apply row_40
    exact row_26
    exact row_83
    exact row_39
  -- chapter_151_line_24: GL tag expansion for integration.
  have row_24 : ((gl_existence2 V2 N zero identity previous v6 succ add) ↔ (¬ (∀ (V3 : GLBinaryRelation α), ((gl_fXY V3 V2 N) → (¬ (gl_and0 zero identity V3 previous v6 V2 succ add)))))) := by
    exact Iff.rfl
  -- chapter_151_line_23_rule: GL tag reformulation for integration >[].
  have row_23_rule : ((gl_fXY V3 V2 N) → ((gl_and0 zero identity V3 previous v6 V2 succ add) → (gl_existence2 V2 N zero identity previous v6 succ add))) := by
    intro integration_premise_1
    intro integration_premise_2
    apply (row_24).2
    intro universal_counterexample
    exact universal_counterexample V3 integration_premise_1 integration_premise_2
  -- chapter_151_line_23: GL tag reformulation for integration >[].
  have row_23 : (gl_implication125 V3 V2 N zero identity previous v6 succ add) := by
    simpa only [gl_implication125] using row_23_rule
  -- chapter_151_line_22: GL tag expansion.
  have row_22 : ((gl_fXY V3 V2 N) → ((gl_and0 zero identity V3 previous v6 V2 succ add) → (gl_existence2 V2 N zero identity previous v6 succ add))) := by
    simpa only [gl_implication125] using (row_23)
  -- chapter_151_line_21: GL tag implication.
  have row_21 : (gl_existence2 V2 N zero identity previous v6 succ add) := by
    apply row_22
    exact row_91
    exact row_25
  -- chapter_151_line_20: GL tag expansion for integration.
  have row_20 : ((gl_fold N succ add add identity zero previous v6) ↔ (¬ (∀ (V2 : GLSet α), ((gl_interval N add zero previous V2) → (¬ (gl_existence2 V2 N zero identity previous v6 succ add)))))) := by
    exact Iff.rfl
  -- chapter_151_line_19_rule: GL tag reformulation for integration >[].
  have row_19_rule : ((gl_interval N add zero previous V2) → ((gl_existence2 V2 N zero identity previous v6 succ add) → (gl_fold N succ add add identity zero previous v6))) := by
    intro integration_premise_1
    intro integration_premise_2
    apply (row_20).2
    intro universal_counterexample
    exact universal_counterexample V2 integration_premise_1 integration_premise_2
  -- chapter_151_line_19: GL tag reformulation for integration >[].
  have row_19 : (gl_implication129 N add zero previous V2 identity v6 succ add) := by
    simpa only [gl_implication129] using row_19_rule
  -- chapter_151_line_18: GL tag expansion.
  have row_18 : ((gl_interval N add zero previous V2) → ((gl_existence2 V2 N zero identity previous v6 succ add) → (gl_fold N succ add add identity zero previous v6))) := by
    simpa only [gl_implication129] using (row_19)
  -- chapter_151_line_17: GL tag implication.
  have row_17 : (gl_fold N succ add add identity zero previous v6) := by
    apply row_18
    exact row_157
    exact row_21
  -- chapter_151_line_16: GL tag implication.
  have row_16 : (mul two v6 v7) := by
    apply row_71
    exact row_72
    exact row_17
    exact row_168
  have rule_row_15 := peano_source_010 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_151_line_15: GL tag implication.
  have row_15 : (add one previous v1) := by
    apply rule_row_15
    exact row_168
  have rule_row_14 := peano_source_012 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_151_line_14: GL tag implication.
  have row_14 : (add previous one v1) := by
    apply rule_row_14
    exact row_15
  -- chapter_151_line_13: GL tag task formulation.
  have row_13 : (succ v1 v2) := by
    exact assumption_13
  -- chapter_151_line_12: GL tag implication.
  have row_12 : (add previous two v2) := by
    apply row_108
    exact row_62
    exact row_111
    exact row_14
    exact row_13
  have rule_row_11 := peano_source_036 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_151_line_11: GL tag implication.
  have row_11 : (add two previous v2) := by
    apply rule_row_11
    exact row_12
  have rule_row_10 := peano_source_054 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_151_line_10: GL tag implication.
  have row_10 : (mul v1 previous v7) := by
    apply rule_row_10
    exact row_72
  -- chapter_151_line_9: GL tag task formulation.
  have row_9 : (mul v1 v2 v3) := by
    exact assumption_9
  have rule_row_8 := peano_source_047 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_151_line_8: GL tag implication.
  have row_8 : (add v5 v7 v3) := by
    apply rule_row_8
    exact row_65
    exact row_10
    exact row_9
    exact row_11
  -- chapter_151_line_7: GL tag implication.
  have row_7 : (succ v23 v1) := by
    apply row_106
    exact row_124
    exact row_110
    exact row_98
    exact row_14
  -- chapter_151_line_6: GL tag equality1.
  have row_6 : (add v6 v25 v24) := by
    have equality_source := row_80
    have equality_step_1 := row_102
    cases equality_step_1
    exact equality_source
  -- chapter_151_line_5: GL tag implication.
  have row_5 : (V4 v1 v24) := by
    apply row_182
    exact row_94
    exact row_7
    exact row_169
    exact row_76
    exact row_90
    exact row_6
  -- chapter_151_line_4: GL tag implication.
  have row_4 : (v24 = v4) := by
    apply row_173
    exact row_169
    exact row_5
    exact row_179
  -- chapter_151_line_3: GL tag equality1.
  have row_3 : (add v1 v6 v4) := by
    have equality_source := row_78
    have equality_step_1 := row_4
    cases equality_step_1
    exact equality_source
  have rule_row_2 := peano_source_054 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_151_line_2: GL tag implication.
  have row_2 : (mul two v1 v5) := by
    apply rule_row_2
    exact row_65
  have rule_row_1 := peano_source_050 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_151_line_1: GL tag implication.
  have row_1 : (mul two v4 v3) := by
    apply rule_row_1
    exact row_8
    exact row_3
    exact row_2
    exact row_16
  exact row_1

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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α) (v3 : α), ((mul v1 v2 v3) → (∀ (v4 : α), ((gl_fold N succ add add identity zero v1 v4) → ((succ v1 v2) → (mul two v4 v3)))))) := by
  intro v1
  intro v2
  intro v3
  intro premise_1
  intro v4
  intro premise_2
  intro premise_3
  have inductionMember : N v1 := by
    have typingRule := gauss_source_079_induction_typing N zero succ add mul one two identity anchor relationalInduction strongRelationalInduction
    exact typingRule v1 v4 premise_2
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorGauss, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v2 : α) (v3 : α), ((mul zero v2 v3) → (∀ (v4 : α), ((gl_fold N succ add add identity zero zero v4) → ((succ zero v2) → (mul two v4 v3)))))) := by
    intro v2
    intro v3
    intro base_premise_1
    intro v4
    intro base_premise_2
    intro base_premise_3
    have zeroRule := gauss_source_079_check_zero N zero succ add mul one two identity anchor relationalInduction strongRelationalInduction
    exact zeroRule zero v2 v3 v4 base_premise_2 rfl base_premise_1 base_premise_3
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v2 : α) (v3 : α), ((mul induction_n v2 v3) → (∀ (v4 : α), ((gl_fold N succ add add identity zero induction_n v4) → ((succ induction_n v2) → (mul two v4 v3)))))) → ∀ induction_m, succ induction_n induction_m → (∀ (v2 : α) (v3 : α), ((mul induction_m v2 v3) → (∀ (v4 : α), ((gl_fold N succ add add identity zero induction_m v4) → ((succ induction_m v2) → (mul two v4 v3)))))) := by
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
        (∀ (w1 : α) (w2 : α), ((mul induction_n w1 w2) → (∀ (w3 : α), ((gl_fold N succ add add identity zero induction_n w3) → ((succ induction_n w1) → (mul two w3 w2)))))) := by
      intro w1
      intro w2
      intro step_induction_assumption_2_premise_1
      intro w3
      intro step_induction_assumption_2_premise_2
      intro step_induction_assumption_2_premise_3
      apply induction_hypothesis
      all_goals assumption
    have stepRule := gauss_source_079_check_induction_condition N zero succ add mul one two identity anchor relationalInduction strongRelationalInduction
    exact stepRule induction_n induction_m v2 v3 v4 step_premise_2 step_induction_assumption_1 step_induction_assumption_2 step_premise_3 step_premise_1
  have inductionProperty : (∀ (v2 : α) (v3 : α), ((mul v1 v2 v3) → (∀ (v4 : α), ((gl_fold N succ add add identity zero v1 v4) → ((succ v1 v2) → (mul two v4 v3)))))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v2 : α) (v3 : α), ((mul induction_value v2 v3) → (∀ (v4 : α), ((gl_fold N succ add add identity zero induction_value v4) → ((succ induction_value v2) → (mul two v4 v3)))))))
      v1
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v2 v3 premise_1 v4 premise_2 premise_3

private theorem gauss_source_080_induction_typing
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    (v2 : α)
    (v4 : α)
    (assumption_5 : (gl_fold N succ add add identity zero v2 v4))
    : (N v2) := by
  -- chapter_152_line_5: GL tag task formulation.
  have row_5 : (gl_fold N succ add add identity zero v2 v4) := by
    exact assumption_5
  -- chapter_152_line_4: GL tag expansion.
  have row_4 : (¬ (∀ (V1 : GLSet α), ((gl_interval N add zero v2 V1) → (¬ (gl_existence2 V1 N zero identity v2 v4 succ add))))) := by
    simpa only [gl_fold] using (row_5)
  have exists_row_4 : ∃ (V1 : GLSet α), ((gl_interval N add zero v2 V1) ∧ (gl_existence2 V1 N zero identity v2 v4 succ add)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_4
      intro V1 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨V1, counterexample_premise_0, counterexample_head⟩
  obtain ⟨V1, witness_row_4⟩ := exists_row_4
  -- chapter_152_line_3: GL tag disintegration.
  have row_3 : (gl_interval N add zero v2 V1) := by
    exact witness_row_4.1
  -- chapter_152_line_2: GL tag expansion.
  have row_2 : (((((gl_implication89 V1 N add zero) ∧ (gl_implication90 V1 N add v2)) ∧ (gl_implication91 N add zero v2 V1)) ∧ (N zero)) ∧ (N v2)) := by
    simpa only [gl_interval] using (row_3)
  -- chapter_152_line_1: GL tag disintegration.
  have row_1 : (N v2) := by
    exact row_2.2
  exact row_1

private theorem gauss_source_080_check_zero
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
    (assumption_90 : (gl_fold N succ add add identity zero v2 v4))
    (assumption_72 : (v2 = zero))
    (assumption_37 : (succ v2 v1))
    (assumption_34 : (mul v1 v2 v3))
    : (mul two v4 v3) := by
  -- chapter_153_line_90: GL tag task formulation.
  have row_90 : (gl_fold N succ add add identity zero v2 v4) := by
    exact assumption_90
  -- chapter_153_line_89: GL tag expansion.
  have row_89 : (¬ (∀ (V1 : GLSet α), ((gl_interval N add zero v2 V1) → (¬ (gl_existence2 V1 N zero identity v2 v4 succ add))))) := by
    simpa only [gl_fold] using (row_90)
  have exists_row_89 : ∃ (V1 : GLSet α), ((gl_interval N add zero v2 V1) ∧ (gl_existence2 V1 N zero identity v2 v4 succ add)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_89
      intro V1 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨V1, counterexample_premise_0, counterexample_head⟩
  obtain ⟨V1, witness_row_89⟩ := exists_row_89
  -- chapter_153_line_88: GL tag disintegration.
  have row_88 : (gl_existence2 V1 N zero identity v2 v4 succ add) := by
    exact witness_row_89.2
  -- chapter_153_line_87: GL tag expansion.
  have row_87 : (¬ (∀ (V2 : GLBinaryRelation α), ((gl_fXY V2 V1 N) → (¬ (gl_and0 zero identity V2 v2 v4 V1 succ add))))) := by
    simpa only [gl_existence2] using (row_88)
  have exists_row_87 : ∃ (V2 : GLBinaryRelation α), ((gl_fXY V2 V1 N) ∧ (gl_and0 zero identity V2 v2 v4 V1 succ add)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_87
      intro V2 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨V2, counterexample_premise_0, counterexample_head⟩
  obtain ⟨V2, witness_row_87⟩ := exists_row_87
  -- chapter_153_line_86: GL tag disintegration.
  have row_86 : (gl_and0 zero identity V2 v2 v4 V1 succ add) := by
    exact witness_row_87.2
  -- chapter_153_line_85: GL tag expansion.
  have row_85 : (((gl_implication121 zero identity V2) ∧ (V2 v2 v4)) ∧ (gl_implication122 V1 succ V2 identity add)) := by
    simpa only [gl_and0] using (row_86)
  -- chapter_153_line_84: GL tag disintegration.
  have row_84 : (gl_implication121 zero identity V2) := by
    exact row_85.1.1
  -- chapter_153_line_83: GL tag expansion.
  have row_83 : (∀ (w1 : α), ((identity zero w1) → (V2 zero w1))) := by
    simpa only [gl_implication121] using (row_84)
  -- chapter_153_line_82: GL tag disintegration.
  have row_82 : (gl_fXY V2 V1 N) := by
    exact witness_row_87.1
  -- chapter_153_line_81: GL tag expansion.
  have row_81 : ((((gl_implication0 V2 V1) ∧ (gl_implication1 V2 N)) ∧ (gl_implication6 V1 N V2)) ∧ (gl_implication7 V1 V2)) := by
    simpa only [gl_fXY] using (row_82)
  -- chapter_153_line_80: GL tag disintegration.
  have row_80 : (gl_implication1 V2 N) := by
    exact row_81.1.1.2
  -- chapter_153_line_79: GL tag expansion.
  have row_79 : (∀ (w1 : α) (w2 : α), ((V2 w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using (row_80)
  -- chapter_153_line_78: GL tag disintegration.
  have row_78 : (gl_implication0 V2 V1) := by
    exact row_81.1.1.1
  -- chapter_153_line_77: GL tag expansion.
  have row_77 : (∀ (w1 : α) (w2 : α), ((V2 w1 w2) → (V1 w1))) := by
    simpa only [gl_implication0] using (row_78)
  -- chapter_153_line_76: GL tag disintegration.
  have row_76 : (gl_implication7 V1 V2) := by
    exact row_81.2
  -- chapter_153_line_75: GL tag expansion.
  have row_75 : (∀ (w1 : α), ((V1 w1) → (∀ (w2 : α), ((V2 w1 w2) → (∀ (w3 : α), ((V2 w1 w3) → (w2 = w3))))))) := by
    simpa only [gl_implication7] using (row_76)
  -- chapter_153_line_74: GL tag disintegration.
  have row_74 : (V2 v2 v4) := by
    exact row_85.1.2
  -- chapter_153_line_73: GL tag implication grouping.
  have row_73 : (∀ (w1 : α) (w2 : α), ((V2 w1 w2) → (gl_and20 w1 V1 w2 N))) := by
    have auxiliary_source_1 := row_77
    have auxiliary_source_2 := row_79
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_premise
    exact ⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_premise)⟩
  -- chapter_153_line_72: GL tag recursion.
  have row_72 : (v2 = zero) := by
    exact assumption_72
  -- chapter_153_line_71: GL tag equality1.
  have row_71 : (V2 zero v4) := by
    have equality_source := row_74
    have equality_step_1 := row_72
    cases equality_step_1
    exact equality_source
  -- chapter_153_line_70: GL tag task formulation.
  have row_70 : (gl_AnchorGauss N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_153_line_69: GL tag expansion.
  have row_69 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorGauss] using (row_70)
  -- chapter_153_line_68: GL tag disintegration.
  have row_68 : (gl_identity N identity) := by
    exact row_69.2
  -- chapter_153_line_67: GL tag expansion.
  have row_67 : (((gl_implication0 identity N) ∧ (gl_implication37 identity)) ∧ (gl_implication38 N identity)) := by
    simpa only [gl_identity] using (row_68)
  -- chapter_153_line_66: GL tag disintegration.
  have row_66 : (gl_implication38 N identity) := by
    exact row_67.2
  -- chapter_153_line_65: GL tag expansion.
  have row_65 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((w1 = w2) → (identity w1 w2))))) := by
    simpa only [gl_implication38] using (row_66)
  -- chapter_153_line_64: GL tag disintegration.
  have row_64 : (succ zero one) := by
    exact row_69.1.1.2
  -- chapter_153_line_63: GL tag disintegration.
  have row_63 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_69.1.1.1
  -- chapter_153_line_62: GL tag expansion.
  have row_62 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_63)
  -- chapter_153_line_61: GL tag disintegration.
  have row_61 : (gl_fXY succ N N) := by
    exact row_62.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_153_line_60: GL tag expansion.
  have row_60 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_61)
  -- chapter_153_line_59: GL tag disintegration.
  have row_59 : (gl_implication1 succ N) := by
    exact row_60.1.1.2
  -- chapter_153_line_58: GL tag expansion.
  have row_58 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using (row_59)
  -- chapter_153_line_57: GL tag disintegration.
  have row_57 : (gl_implication0 succ N) := by
    exact row_60.1.1.1
  -- chapter_153_line_56: GL tag expansion.
  have row_56 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_57)
  -- chapter_153_line_55: GL tag disintegration.
  have row_55 : (succ one two) := by
    exact row_69.1.2
  -- chapter_153_line_54: GL tag disintegration.
  have row_54 : (gl_fXYZ add N N N) := by
    exact row_62.1.1.1.1.1.1.1.1.1.2
  -- chapter_153_line_53: GL tag expansion.
  have row_53 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_54)
  -- chapter_153_line_52: GL tag disintegration.
  have row_52 : (gl_implication13 add N) := by
    exact row_53.1.1.2
  -- chapter_153_line_51: GL tag expansion.
  have row_51 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w3))) := by
    simpa only [gl_implication13] using (row_52)
  -- chapter_153_line_50: GL tag disintegration.
  have row_50 : (gl_implication12 add N) := by
    exact row_53.1.1.1.2
  -- chapter_153_line_49: GL tag expansion.
  have row_49 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w2))) := by
    simpa only [gl_implication12] using (row_50)
  -- chapter_153_line_48: GL tag disintegration.
  have row_48 : (gl_implication11 add N) := by
    exact row_53.1.1.1.1
  -- chapter_153_line_47: GL tag expansion.
  have row_47 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (N w1))) := by
    simpa only [gl_implication11] using (row_48)
  -- chapter_153_line_46: GL tag disintegration.
  have row_46 : (gl_implication10 N succ) := by
    exact row_62.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_153_line_45: GL tag expansion.
  have row_45 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w2 w1) → (∀ (w3 : α), ((succ w3 w1) → (w2 = w3))))))) := by
    simpa only [gl_implication10] using (row_46)
  -- chapter_153_line_44: GL tag disintegration.
  have row_44 : (N zero) := by
    exact row_62.1.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_153_line_43: GL tag disintegration.
  have row_43 : (gl_implication26 zero N mul) := by
    exact row_62.1.1.2
  -- chapter_153_line_42: GL tag expansion.
  have row_42 : (∀ (w1 : α), ((w1 = zero) → (∀ (w2 : α), ((N w2) → ((N w1) → (mul w2 zero w1)))))) := by
    simpa only [gl_implication26] using (row_43)
  -- chapter_153_line_41: GL tag expansion for integration.
  have row_41 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_153_line_40_rule: GL tag reformulation for integration and.
  have row_40_rule : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_153_line_40: GL tag reformulation for integration and.
  have row_40 : (gl_implication76 N zero succ add mul one) := by
    simpa only [gl_implication76] using row_40_rule
  -- chapter_153_line_39: GL tag expansion.
  have row_39 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    simpa only [gl_implication76] using (row_40)
  -- chapter_153_line_38: GL tag implication.
  have row_38 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_39
    exact row_63
    exact row_64
  -- chapter_153_line_37: GL tag task formulation.
  have row_37 : (succ v2 v1) := by
    exact assumption_37
  have rule_row_36 := peano_source_010 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_153_line_36: GL tag implication.
  have row_36 : (add one v2 v1) := by
    apply rule_row_36
    exact row_37
  have rule_row_35 := peano_source_012 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_153_line_35: GL tag implication.
  have row_35 : (add v2 one v1) := by
    apply rule_row_35
    exact row_36
  -- chapter_153_line_34: GL tag task formulation.
  have row_34 : (mul v1 v2 v3) := by
    exact assumption_34
  have rule_row_33 := peano_source_054 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_153_line_33: GL tag implication.
  have row_33 : (mul v2 v1 v3) := by
    apply rule_row_33
    exact row_34
  -- chapter_153_line_32: GL tag equality1.
  have row_32 : (mul zero v1 v3) := by
    have equality_source := row_33
    have equality_step_1 := row_72
    cases equality_step_1
    exact equality_source
  have rule_row_31 := peano_source_065 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_153_line_31: GL tag implication.
  have row_31 : (gl_inSequence N one succ) := by
    apply rule_row_31
  -- chapter_153_line_30: GL tag expansion.
  have row_30 : (¬ (∀ (v5 : α), ((N v5) → (¬ (succ v5 one))))) := by
    simpa only [gl_inSequence] using (row_31)
  have exists_row_30 : ∃ (v5 : α), ((N v5) ∧ (succ v5 one)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_30
      intro v5 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v5, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v5, witness_row_30⟩ := exists_row_30
  -- chapter_153_line_29: GL tag disintegration.
  have row_29 : (succ v5 one) := by
    exact witness_row_30.2
  -- chapter_153_line_28: GL tag disintegration.
  have row_28 : (N v5) := by
    exact witness_row_30.1
  -- chapter_153_line_27: GL tag implication grouping.
  have row_27 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (gl_and20 w1 N w2 N))) := by
    have auxiliary_source_1 := row_56
    have auxiliary_source_2 := row_58
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_premise
    exact ⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_premise)⟩
  -- chapter_153_line_26: GL tag implication.
  have row_26 : (gl_and20 one N two N) := by
    apply row_27
    exact row_55
  -- chapter_153_line_25: GL tag implication grouping.
  have row_25 : (∀ (w1 : α) (w2 : α) (w3 : α), ((add w1 w2 w3) → (gl_and21 w1 N w2 N w3 N))) := by
    have auxiliary_source_1 := row_47
    have auxiliary_source_2 := row_49
    have auxiliary_source_3 := row_51
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_argument_3
    intro auxiliary_premise
    exact ⟨⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩, (auxiliary_source_3 auxiliary_argument_1 auxiliary_argument_2 auxiliary_argument_3 auxiliary_premise)⟩
  -- chapter_153_line_24: GL tag implication.
  have row_24 : (gl_and21 v2 N one N v1 N) := by
    apply row_25
    exact row_35
  -- chapter_153_line_23: GL tag expansion.
  have row_23 : (((N v2) ∧ (N one)) ∧ (N v1)) := by
    simpa only [gl_and21] using (row_24)
  -- chapter_153_line_22: GL tag disintegration.
  have row_22 : (N v1) := by
    exact row_23.2
  -- chapter_153_line_21: GL tag disintegration.
  have row_21 : (N one) := by
    exact row_23.1.2
  have rule_row_20 := peano_source_048 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_153_line_20: GL tag implication.
  have row_20 : (zero = v3) := by
    apply rule_row_20
    exact row_22
    exact row_32
  -- chapter_153_line_19: GL tag implication.
  have row_19 : (zero = v5) := by
    apply row_45
    exact row_21
    exact row_64
    exact row_29
  -- chapter_153_line_18: GL tag implication.
  have row_18 : (identity zero v5) := by
    apply row_65
    exact row_44
    exact row_19
  -- chapter_153_line_17: GL tag implication.
  have row_17 : (v5 = zero) := by
    apply row_45
    exact row_21
    exact row_29
    exact row_64
  -- chapter_153_line_16: GL tag implication.
  have row_16 : (V2 zero v5) := by
    apply row_83
    exact row_18
  -- chapter_153_line_15: GL tag implication.
  have row_15 : (gl_and20 zero V1 v5 N) := by
    apply row_73
    exact row_16
  -- chapter_153_line_14: GL tag equality1.
  have row_14 : (gl_and20 zero V1 zero N) := by
    have equality_source := row_15
    have equality_step_1 := row_17
    cases equality_step_1
    exact equality_source
  -- chapter_153_line_13: GL tag expansion.
  have row_13 : ((V1 zero) ∧ (N zero)) := by
    simpa only [gl_and20] using (row_14)
  -- chapter_153_line_12: GL tag disintegration.
  have row_12 : (V1 zero) := by
    exact row_13.1
  -- chapter_153_line_11: GL tag equality1.
  have row_11 : (V2 zero zero) := by
    have equality_source := row_16
    have equality_step_1 := row_17
    cases equality_step_1
    exact equality_source
  -- chapter_153_line_10: GL tag symmetry of equality.
  have row_10 : (zero = v2) := by
    exact Eq.symm row_72
  -- chapter_153_line_9: GL tag equality1.
  have row_9 : (V2 zero v2) := by
    have equality_source := row_11
    have equality_step_1 := row_10
    cases equality_step_1
    exact equality_source
  -- chapter_153_line_8: GL tag implication.
  have row_8 : (v2 = v4) := by
    apply row_75
    exact row_12
    exact row_9
    exact row_71
  -- chapter_153_line_7: GL tag expansion.
  have row_7 : ((N one) ∧ (N two)) := by
    simpa only [gl_and20] using (row_26)
  -- chapter_153_line_6: GL tag disintegration.
  have row_6 : (N two) := by
    exact row_7.2
  -- chapter_153_line_5: GL tag implication.
  have row_5 : (mul two zero v5) := by
    apply row_42
    exact row_17
    exact row_6
    exact row_28
  -- chapter_153_line_4: GL tag equality1.
  have row_4 : (mul two zero zero) := by
    have equality_source := row_5
    have equality_step_1 := row_17
    cases equality_step_1
    exact equality_source
  -- chapter_153_line_3: GL tag equality1.
  have row_3 : (mul two v2 v2) := by
    have equality_source := row_4
    have equality_step_1 := row_10
    cases equality_step_1
    exact equality_source
  -- chapter_153_line_2: GL tag equality2.
  have row_2 : (v2 = v3) := by
    exact Eq.trans row_72 row_20
  -- chapter_153_line_1: GL tag equality1.
  have row_1 : (mul two v4 v3) := by
    have equality_source := row_3
    have equality_step_1 := row_2
    cases equality_step_1
    have equality_step_2 := row_8
    cases equality_step_2
    exact equality_source
  exact row_1

private theorem gauss_source_080_check_induction_condition
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
    (assumption_209 : (gl_fold N succ add add identity zero v2 v4))
    (assumption_185 : (succ previous v2))
    (assumption_100 : (mul v1 v2 v3))
    (assumption_96 : (succ v2 v1))
    (assumption_14 : (∀ (w1 : α) (w2 : α), ((mul w1 previous w2) → (∀ (w3 : α), ((gl_fold N succ add add identity zero previous w3) → ((succ previous w1) → (mul two w3 w2)))))))
    : (mul two v4 v3) := by
  -- chapter_154_line_209: GL tag task formulation.
  have row_209 : (gl_fold N succ add add identity zero v2 v4) := by
    exact assumption_209
  -- chapter_154_line_208: GL tag expansion.
  have row_208 : (¬ (∀ (V1 : GLSet α), ((gl_interval N add zero v2 V1) → (¬ (gl_existence2 V1 N zero identity v2 v4 succ add))))) := by
    simpa only [gl_fold] using (row_209)
  have exists_row_208 : ∃ (V1 : GLSet α), ((gl_interval N add zero v2 V1) ∧ (gl_existence2 V1 N zero identity v2 v4 succ add)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_208
      intro V1 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨V1, counterexample_premise_0, counterexample_head⟩
  obtain ⟨V1, witness_row_208⟩ := exists_row_208
  -- chapter_154_line_207: GL tag disintegration.
  have row_207 : (gl_existence2 V1 N zero identity v2 v4 succ add) := by
    exact witness_row_208.2
  -- chapter_154_line_206: GL tag expansion.
  have row_206 : (¬ (∀ (V4 : GLBinaryRelation α), ((gl_fXY V4 V1 N) → (¬ (gl_and0 zero identity V4 v2 v4 V1 succ add))))) := by
    simpa only [gl_existence2] using (row_207)
  have exists_row_206 : ∃ (V4 : GLBinaryRelation α), ((gl_fXY V4 V1 N) ∧ (gl_and0 zero identity V4 v2 v4 V1 succ add)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_206
      intro V4 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨V4, counterexample_premise_0, counterexample_head⟩
  obtain ⟨V4, witness_row_206⟩ := exists_row_206
  -- chapter_154_line_205: GL tag disintegration.
  have row_205 : (gl_and0 zero identity V4 v2 v4 V1 succ add) := by
    exact witness_row_206.2
  -- chapter_154_line_204: GL tag expansion.
  have row_204 : (((gl_implication121 zero identity V4) ∧ (V4 v2 v4)) ∧ (gl_implication122 V1 succ V4 identity add)) := by
    simpa only [gl_and0] using (row_205)
  -- chapter_154_line_203: GL tag disintegration.
  have row_203 : (gl_implication121 zero identity V4) := by
    exact row_204.1.1
  -- chapter_154_line_202: GL tag expansion.
  have row_202 : (∀ (w1 : α), ((identity zero w1) → (V4 zero w1))) := by
    simpa only [gl_implication121] using (row_203)
  -- chapter_154_line_201: GL tag disintegration.
  have row_201 : (gl_implication122 V1 succ V4 identity add) := by
    exact row_204.2
  -- chapter_154_line_200: GL tag expansion.
  have row_200 : (∀ (w1 : α), ((V1 w1) → (∀ (w2 : α), ((succ w1 w2) → ((V1 w2) → (∀ (w3 : α), ((V4 w1 w3) → (∀ (w4 : α), ((identity w2 w4) → (∀ (w5 : α), ((add w3 w4 w5) → (V4 w2 w5)))))))))))) := by
    simpa only [gl_implication122] using (row_201)
  -- chapter_154_line_199: GL tag disintegration.
  have row_199 : (V4 v2 v4) := by
    exact row_204.1.2
  -- chapter_154_line_198: GL tag disintegration.
  have row_198 : (gl_fXY V4 V1 N) := by
    exact witness_row_206.1
  -- chapter_154_line_197: GL tag expansion.
  have row_197 : ((((gl_implication0 V4 V1) ∧ (gl_implication1 V4 N)) ∧ (gl_implication6 V1 N V4)) ∧ (gl_implication7 V1 V4)) := by
    simpa only [gl_fXY] using (row_198)
  -- chapter_154_line_196: GL tag disintegration.
  have row_196 : (gl_implication0 V4 V1) := by
    exact row_197.1.1.1
  -- chapter_154_line_195: GL tag expansion.
  have row_195 : (∀ (w1 : α) (w2 : α), ((V4 w1 w2) → (V1 w1))) := by
    simpa only [gl_implication0] using (row_196)
  -- chapter_154_line_194: GL tag disintegration.
  have row_194 : (gl_implication7 V1 V4) := by
    exact row_197.2
  -- chapter_154_line_193: GL tag expansion.
  have row_193 : (∀ (w1 : α), ((V1 w1) → (∀ (w2 : α), ((V4 w1 w2) → (∀ (w3 : α), ((V4 w1 w3) → (w2 = w3))))))) := by
    simpa only [gl_implication7] using (row_194)
  -- chapter_154_line_192: GL tag disintegration.
  have row_192 : (gl_interval N add zero v2 V1) := by
    exact witness_row_208.1
  -- chapter_154_line_191: GL tag expansion.
  have row_191 : (((((gl_implication89 V1 N add zero) ∧ (gl_implication90 V1 N add v2)) ∧ (gl_implication91 N add zero v2 V1)) ∧ (N zero)) ∧ (N v2)) := by
    simpa only [gl_interval] using (row_192)
  -- chapter_154_line_190: GL tag disintegration.
  have row_190 : (N v2) := by
    exact row_191.2
  -- chapter_154_line_189: GL tag implication.
  have row_189 : (V1 v2) := by
    apply row_195
    exact row_199
  -- chapter_154_line_188: GL tag task formulation.
  have row_188 : (gl_AnchorGauss N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_154_line_187: GL tag theorem.
  have row_187 := gauss_source_077 N zero succ add mul one two identity anchor relationalInduction strongRelationalInduction
  -- chapter_154_line_186: GL tag implication.
  have row_186 : (gl_sequence N add zero v2 V4) := by
    apply row_187
    exact row_198
    exact row_192
  -- chapter_154_line_185: GL tag recursion.
  have row_185 : (succ previous v2) := by
    exact assumption_185
  -- chapter_154_line_184: GL tag theorem.
  have row_184 := gauss_source_076 N zero succ add mul one two identity anchor relationalInduction strongRelationalInduction
  -- chapter_154_line_183: GL tag implication.
  have row_183 : (gl_existence12 N add previous V4 zero) := by
    apply row_184
    exact row_185
    exact row_186
  -- chapter_154_line_182: GL tag expansion.
  have row_182 : (¬ (∀ (V2 : GLBinaryRelation α), ((gl_limitSequence N add previous V4 V2) → (¬ (gl_sequence N add zero previous V2))))) := by
    simpa only [gl_existence12] using (row_183)
  have exists_row_182 : ∃ (V2 : GLBinaryRelation α), ((gl_limitSequence N add previous V4 V2) ∧ (gl_sequence N add zero previous V2)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_182
      intro V2 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨V2, counterexample_premise_0, counterexample_head⟩
  obtain ⟨V2, witness_row_182⟩ := exists_row_182
  -- chapter_154_line_181: GL tag disintegration.
  have row_181 : (gl_limitSequence N add previous V4 V2) := by
    exact witness_row_182.1
  -- chapter_154_line_180: GL tag expansion.
  have row_180 : (((gl_implication136 V2 N add previous) ∧ (gl_implication137 V2 V4)) ∧ (gl_implication138 N add previous V4 V2)) := by
    simpa only [gl_limitSequence] using (row_181)
  -- chapter_154_line_179: GL tag disintegration.
  have row_179 : (gl_implication138 N add previous V4 V2) := by
    exact row_180.2
  -- chapter_154_line_178: GL tag expansion.
  have row_178 : (∀ (w1 : α), ((gl_preorder N add w1 previous) → (∀ (w2 : α), ((V4 w1 w2) → (V2 w1 w2))))) := by
    simpa only [gl_implication138] using (row_179)
  -- chapter_154_line_177: GL tag disintegration.
  have row_177 : (gl_implication137 V2 V4) := by
    exact row_180.1.2
  -- chapter_154_line_176: GL tag expansion.
  have row_176 : (∀ (w1 : α) (w2 : α), ((V2 w1 w2) → (V4 w1 w2))) := by
    simpa only [gl_implication137] using (row_177)
  -- chapter_154_line_175: GL tag disintegration.
  have row_175 : (gl_sequence N add zero previous V2) := by
    exact witness_row_182.2
  -- chapter_154_line_174: GL tag theorem.
  have row_174 := gauss_source_074 N zero succ add mul one two identity anchor relationalInduction strongRelationalInduction
  -- chapter_154_line_173: GL tag implication.
  have row_173 : (gl_existence8 N add V1 previous zero) := by
    apply row_174
    exact row_185
    exact row_192
  -- chapter_154_line_172: GL tag expansion.
  have row_172 : (¬ (∀ (V3 : GLSet α), ((gl_limitSet N add V1 previous V3) → (¬ (gl_interval N add zero previous V3))))) := by
    simpa only [gl_existence8] using (row_173)
  have exists_row_172 : ∃ (V3 : GLSet α), ((gl_limitSet N add V1 previous V3) ∧ (gl_interval N add zero previous V3)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_172
      intro V3 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨V3, counterexample_premise_0, counterexample_head⟩
  obtain ⟨V3, witness_row_172⟩ := exists_row_172
  -- chapter_154_line_171: GL tag disintegration.
  have row_171 : (gl_limitSet N add V1 previous V3) := by
    exact witness_row_172.1
  -- chapter_154_line_170: GL tag expansion.
  have row_170 : (((gl_subset V3 V1) ∧ (gl_implication90 V3 N add previous)) ∧ (gl_implication141 V1 N add previous V3)) := by
    simpa only [gl_limitSet] using (row_171)
  -- chapter_154_line_169: GL tag disintegration.
  have row_169 : (gl_implication90 V3 N add previous) := by
    exact row_170.1.2
  -- chapter_154_line_168: GL tag expansion.
  have row_168 : (∀ (w1 : α), ((V3 w1) → (gl_preorder N add w1 previous))) := by
    simpa only [gl_implication90] using (row_169)
  -- chapter_154_line_167: GL tag disintegration.
  have row_167 : (gl_subset V3 V1) := by
    exact row_170.1.1
  -- chapter_154_line_166: GL tag expansion.
  have row_166 : (∀ (w1 : α), ((V3 w1) → (V1 w1))) := by
    simpa only [gl_subset] using (row_167)
  -- chapter_154_line_165: GL tag disintegration.
  have row_165 : (gl_interval N add zero previous V3) := by
    exact witness_row_172.2
  -- chapter_154_line_164: GL tag expansion.
  have row_164 : (((((gl_implication89 V3 N add zero) ∧ (gl_implication90 V3 N add previous)) ∧ (gl_implication91 N add zero previous V3)) ∧ (N zero)) ∧ (N previous)) := by
    simpa only [gl_interval] using (row_165)
  -- chapter_154_line_163: GL tag disintegration.
  have row_163 : (gl_implication89 V3 N add zero) := by
    exact row_164.1.1.1.1
  -- chapter_154_line_162: GL tag expansion.
  have row_162 : (∀ (w1 : α), ((V3 w1) → (gl_preorder N add zero w1))) := by
    simpa only [gl_implication89] using (row_163)
  -- chapter_154_line_161: GL tag theorem.
  have row_161 := gauss_source_069 N zero succ add mul one two identity anchor relationalInduction strongRelationalInduction
  -- chapter_154_line_160: GL tag implication.
  have row_160 : (V3 previous) := by
    apply row_161
    exact row_165
  -- chapter_154_line_159: GL tag expansion.
  have row_159 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorGauss] using (row_188)
  -- chapter_154_line_158: GL tag disintegration.
  have row_158 : (gl_identity N identity) := by
    exact row_159.2
  -- chapter_154_line_157: GL tag expansion.
  have row_157 : (((gl_implication0 identity N) ∧ (gl_implication37 identity)) ∧ (gl_implication38 N identity)) := by
    simpa only [gl_identity] using (row_158)
  -- chapter_154_line_156: GL tag disintegration.
  have row_156 : (gl_implication37 identity) := by
    exact row_157.1.2
  -- chapter_154_line_155: GL tag expansion.
  have row_155 : (∀ (w1 : α) (w2 : α), ((identity w1 w2) → (w1 = w2))) := by
    simpa only [gl_implication37] using (row_156)
  -- chapter_154_line_154: GL tag disintegration.
  have row_154 : (gl_implication38 N identity) := by
    exact row_157.2
  -- chapter_154_line_153: GL tag expansion.
  have row_153 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((w1 = w2) → (identity w1 w2))))) := by
    simpa only [gl_implication38] using (row_154)
  -- chapter_154_line_152: GL tag disintegration.
  have row_152 : (succ one two) := by
    exact row_159.1.2
  -- chapter_154_line_151: GL tag disintegration.
  have row_151 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_159.1.1.1
  -- chapter_154_line_150: GL tag expansion.
  have row_150 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_151)
  -- chapter_154_line_149: GL tag disintegration.
  have row_149 : (gl_implication24 N succ add) := by
    exact row_150.1.1.1.1.1.2
  -- chapter_154_line_148: GL tag expansion.
  have row_148 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((succ w4 w5) → (add w3 w2 w5))))))))) := by
    simpa only [gl_implication24] using (row_149)
  -- chapter_154_line_147: GL tag disintegration.
  have row_147 : (gl_implication10 N succ) := by
    exact row_150.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_154_line_146: GL tag expansion.
  have row_146 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w2 w1) → (∀ (w3 : α), ((succ w3 w1) → (w2 = w3))))))) := by
    simpa only [gl_implication10] using (row_147)
  -- chapter_154_line_145: GL tag disintegration.
  have row_145 : (gl_fXYZ mul N N N) := by
    exact row_150.1.1.1.1.2
  -- chapter_154_line_144: GL tag expansion.
  have row_144 : (((((gl_implication11 mul N) ∧ (gl_implication12 mul N)) ∧ (gl_implication13 mul N)) ∧ (gl_implication18 N N N mul)) ∧ (gl_implication19 N N mul)) := by
    simpa only [gl_fXYZ] using (row_145)
  -- chapter_154_line_143: GL tag disintegration.
  have row_143 : (gl_implication18 N N N mul) := by
    exact row_144.1.2
  -- chapter_154_line_142: GL tag expansion.
  have row_142 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication18] using (row_143)
  -- chapter_154_line_141: GL tag disintegration.
  have row_141 : (gl_implication23 N succ add) := by
    exact row_150.1.1.1.1.1.1.2
  -- chapter_154_line_140: GL tag expansion.
  have row_140 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication23] using (row_141)
  -- chapter_154_line_139: GL tag disintegration.
  have row_139 : (gl_implication21 N zero add) := by
    exact row_150.1.1.1.1.1.1.1.1.2
  -- chapter_154_line_138: GL tag expansion.
  have row_138 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((add w1 zero w2) → (w1 = w2))))) := by
    simpa only [gl_implication21] using (row_139)
  -- chapter_154_line_137: GL tag disintegration.
  have row_137 : (gl_fXY succ N N) := by
    exact row_150.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_154_line_136: GL tag expansion.
  have row_136 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_137)
  -- chapter_154_line_135: GL tag disintegration.
  have row_135 : (gl_implication0 succ N) := by
    exact row_136.1.1.1
  -- chapter_154_line_134: GL tag expansion.
  have row_134 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_135)
  -- chapter_154_line_133: GL tag disintegration.
  have row_133 : (gl_implication1 succ N) := by
    exact row_136.1.1.2
  -- chapter_154_line_132: GL tag expansion.
  have row_132 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using (row_133)
  -- chapter_154_line_131: GL tag disintegration.
  have row_131 : (gl_fXYZ add N N N) := by
    exact row_150.1.1.1.1.1.1.1.1.1.2
  -- chapter_154_line_130: GL tag expansion.
  have row_130 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_131)
  -- chapter_154_line_129: GL tag disintegration.
  have row_129 : (gl_implication18 N N N add) := by
    exact row_130.1.2
  -- chapter_154_line_128: GL tag expansion.
  have row_128 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication18] using (row_129)
  -- chapter_154_line_127: GL tag disintegration.
  have row_127 : (N zero) := by
    exact row_150.1.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_154_line_126: GL tag disintegration.
  have row_126 : (succ zero one) := by
    exact row_159.1.1.2
  -- chapter_154_line_125: GL tag disintegration.
  have row_125 : (gl_implication19 N N add) := by
    exact row_130.2
  -- chapter_154_line_124: GL tag expansion.
  have row_124 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (∀ (w3 : α), ((add w1 w2 w3) → (∀ (w4 : α), ((add w1 w2 w4) → (w3 = w4))))))))) := by
    simpa only [gl_implication19] using (row_125)
  -- chapter_154_line_123: GL tag implication grouping.
  have row_123 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (gl_and20 w1 N w2 N))) := by
    have auxiliary_source_1 := row_134
    have auxiliary_source_2 := row_132
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_premise
    exact ⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_premise)⟩
  -- chapter_154_line_122: GL tag implication.
  have row_122 : (gl_and20 one N two N) := by
    apply row_123
    exact row_152
  -- chapter_154_line_121: GL tag expansion.
  have row_121 : ((N one) ∧ (N two)) := by
    simpa only [gl_and20] using (row_122)
  -- chapter_154_line_120: GL tag disintegration.
  have row_120 : (N two) := by
    exact row_121.2
  -- chapter_154_line_119: GL tag implication.
  have row_119 : (gl_existence1 N v2 two mul) := by
    apply row_142
    exact row_190
    exact row_120
  -- chapter_154_line_118: GL tag expansion.
  have row_118 : (¬ (∀ (v6 : α), ((N v6) → (¬ (mul v2 two v6))))) := by
    simpa only [gl_existence1] using (row_119)
  have exists_row_118 : ∃ (v6 : α), ((N v6) ∧ (mul v2 two v6)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_118
      intro v6 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v6, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v6, witness_row_118⟩ := exists_row_118
  -- chapter_154_line_117: GL tag disintegration.
  have row_117 : (mul v2 two v6) := by
    exact witness_row_118.2
  -- chapter_154_line_116: GL tag implication.
  have row_116 : (N previous) := by
    apply row_134
    exact row_185
  -- chapter_154_line_115: GL tag implication.
  have row_115 : (gl_existence1 N v2 previous mul) := by
    apply row_142
    exact row_190
    exact row_116
  -- chapter_154_line_114: GL tag expansion.
  have row_114 : (¬ (∀ (v7 : α), ((N v7) → (¬ (mul v2 previous v7))))) := by
    simpa only [gl_existence1] using (row_115)
  have exists_row_114 : ∃ (v7 : α), ((N v7) ∧ (mul v2 previous v7)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_114
      intro v7 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v7, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v7, witness_row_114⟩ := exists_row_114
  -- chapter_154_line_113: GL tag disintegration.
  have row_113 : (mul v2 previous v7) := by
    exact witness_row_114.2
  -- chapter_154_line_112: GL tag implication.
  have row_112 : (gl_existence1 N previous zero add) := by
    apply row_128
    exact row_116
    exact row_127
  -- chapter_154_line_111: GL tag expansion.
  have row_111 : (¬ (∀ (v16 : α), ((N v16) → (¬ (add previous zero v16))))) := by
    simpa only [gl_existence1] using (row_112)
  have exists_row_111 : ∃ (v16 : α), ((N v16) ∧ (add previous zero v16)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_111
      intro v16 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v16, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v16, witness_row_111⟩ := exists_row_111
  -- chapter_154_line_110: GL tag disintegration.
  have row_110 : (add previous zero v16) := by
    exact witness_row_111.2
  -- chapter_154_line_109: GL tag disintegration.
  have row_109 : (N v16) := by
    exact witness_row_111.1
  -- chapter_154_line_108: GL tag implication.
  have row_108 : (previous = v16) := by
    apply row_138
    exact row_116
    exact row_110
  -- chapter_154_line_107: GL tag expansion for integration.
  have row_107 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_154_line_106_rule: GL tag reformulation for integration and.
  have row_106_rule : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_154_line_106: GL tag reformulation for integration and.
  have row_106 : (gl_implication76 N zero succ add mul one) := by
    simpa only [gl_implication76] using row_106_rule
  -- chapter_154_line_105: GL tag expansion.
  have row_105 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    simpa only [gl_implication76] using (row_106)
  -- chapter_154_line_104: GL tag implication.
  have row_104 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_105
    exact row_151
    exact row_126
  -- chapter_154_line_103: GL tag implication.
  have row_103 : (gl_and20 zero N one N) := by
    apply row_123
    exact row_126
  -- chapter_154_line_102: GL tag expansion.
  have row_102 : ((N zero) ∧ (N one)) := by
    simpa only [gl_and20] using (row_103)
  -- chapter_154_line_101: GL tag disintegration.
  have row_101 : (N one) := by
    exact row_102.2
  -- chapter_154_line_100: GL tag task formulation.
  have row_100 : (mul v1 v2 v3) := by
    exact assumption_100
  have rule_row_99 := peano_source_054 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_154_line_99: GL tag implication.
  have row_99 : (mul v2 v1 v3) := by
    apply rule_row_99
    exact row_100
  have rule_row_98 := peano_source_010 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_154_line_98: GL tag implication.
  have row_98 : (add one previous v2) := by
    apply rule_row_98
    exact row_185
  have rule_row_97 := peano_source_012 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_154_line_97: GL tag implication.
  have row_97 : (add previous one v2) := by
    apply rule_row_97
    exact row_98
  -- chapter_154_line_96: GL tag task formulation.
  have row_96 : (succ v2 v1) := by
    exact assumption_96
  -- chapter_154_line_95: GL tag implication.
  have row_95 : (add previous two v1) := by
    apply row_148
    exact row_101
    exact row_152
    exact row_97
    exact row_96
  have rule_row_94 := peano_source_036 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_154_line_94: GL tag implication.
  have row_94 : (add two previous v1) := by
    apply rule_row_94
    exact row_95
  have rule_row_93 := peano_source_047 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_154_line_93: GL tag implication.
  have row_93 : (add v6 v7 v3) := by
    apply rule_row_93
    exact row_117
    exact row_113
    exact row_99
    exact row_94
  -- chapter_154_line_92: GL tag implication.
  have row_92 : (gl_existence1 N one previous add) := by
    apply row_128
    exact row_101
    exact row_116
  -- chapter_154_line_91: GL tag expansion.
  have row_91 : (¬ (∀ (v17 : α), ((N v17) → (¬ (add one previous v17))))) := by
    simpa only [gl_existence1] using (row_92)
  have exists_row_91 : ∃ (v17 : α), ((N v17) ∧ (add one previous v17)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_91
      intro v17 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v17, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v17, witness_row_91⟩ := exists_row_91
  -- chapter_154_line_90: GL tag disintegration.
  have row_90 : (add one previous v17) := by
    exact witness_row_91.2
  -- chapter_154_line_89: GL tag implication.
  have row_89 : (v17 = v2) := by
    apply row_124
    exact row_101
    exact row_116
    exact row_90
    exact row_98
  -- chapter_154_line_88: GL tag implication.
  have row_88 : (v2 = v17) := by
    apply row_124
    exact row_101
    exact row_116
    exact row_98
    exact row_90
  -- chapter_154_line_87: GL tag implication.
  have row_87 : (gl_existence1 N v2 zero add) := by
    apply row_128
    exact row_190
    exact row_127
  -- chapter_154_line_86: GL tag expansion.
  have row_86 : (¬ (∀ (v15 : α), ((N v15) → (¬ (add v2 zero v15))))) := by
    simpa only [gl_existence1] using (row_87)
  have exists_row_86 : ∃ (v15 : α), ((N v15) ∧ (add v2 zero v15)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_86
      intro v15 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v15, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v15, witness_row_86⟩ := exists_row_86
  -- chapter_154_line_85: GL tag disintegration.
  have row_85 : (add v2 zero v15) := by
    exact witness_row_86.2
  -- chapter_154_line_84: GL tag equality1.
  have row_84 : (add v17 zero v15) := by
    have equality_source := row_85
    have equality_step_1 := row_88
    cases equality_step_1
    exact equality_source
  -- chapter_154_line_83: GL tag anchor handling.
  have row_83 : (gl_AnchorGauss N zero succ add mul one two identity) := by
    exact row_188
  -- chapter_154_line_82: GL tag theorem.
  have row_82 := gauss_source_070 N zero succ add mul one two identity anchor relationalInduction strongRelationalInduction
  -- chapter_154_line_81: GL tag implication.
  have row_81 : (add zero v17 v15) := by
    apply row_82
    exact row_84
  -- chapter_154_line_80: GL tag implication.
  have row_80 : (v2 = v15) := by
    apply row_138
    exact row_190
    exact row_85
  -- chapter_154_line_79: GL tag symmetry of equality.
  have row_79 : (v15 = v2) := by
    exact Eq.symm row_80
  -- chapter_154_line_78: GL tag equality1.
  have row_78 : (add zero v2 v2) := by
    have equality_source := row_81
    have equality_step_1 := row_79
    cases equality_step_1
    have equality_step_2 := row_89
    cases equality_step_2
    exact equality_source
  -- chapter_154_line_77: GL tag equality1.
  have row_77 : (add zero v15 v2) := by
    have equality_source := row_78
    have equality_step_1 := row_80
    cases equality_step_1
    exact equality_source
  -- chapter_154_line_76: GL tag equality1.
  have row_76 : (succ v16 v15) := by
    have equality_source := row_185
    have equality_step_1 := row_108
    cases equality_step_1
    have equality_step_2 := row_80
    cases equality_step_2
    exact equality_source
  -- chapter_154_line_75: GL tag implication.
  have row_75 : (gl_existence1 N zero previous add) := by
    apply row_128
    exact row_127
    exact row_116
  -- chapter_154_line_74: GL tag expansion.
  have row_74 : (¬ (∀ (v11 : α), ((N v11) → (¬ (add zero previous v11))))) := by
    simpa only [gl_existence1] using (row_75)
  have exists_row_74 : ∃ (v11 : α), ((N v11) ∧ (add zero previous v11)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_74
      intro v11 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v11, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v11, witness_row_74⟩ := exists_row_74
  -- chapter_154_line_73: GL tag disintegration.
  have row_73 : (add zero previous v11) := by
    exact witness_row_74.2
  -- chapter_154_line_72: GL tag equality1.
  have row_72 : (add zero v16 v11) := by
    have equality_source := row_73
    have equality_step_1 := row_108
    cases equality_step_1
    exact equality_source
  -- chapter_154_line_71: GL tag implication.
  have row_71 : (succ v11 v2) := by
    apply row_140
    exact row_109
    exact row_76
    exact row_72
    exact row_77
  -- chapter_154_line_70: GL tag implication.
  have row_70 : (identity v2 v15) := by
    apply row_153
    exact row_190
    exact row_80
  have rule_row_69 := peano_source_032 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_154_line_69: GL tag implication.
  have row_69 : (previous = v11) := by
    apply rule_row_69
    exact row_116
    exact row_73
  -- chapter_154_line_68: GL tag equality1.
  have row_68 : (gl_interval N add zero v11 V3) := by
    have equality_source := row_165
    have equality_step_1 := row_69
    cases equality_step_1
    exact equality_source
  -- chapter_154_line_67: GL tag implication.
  have row_67 : (V3 v11) := by
    apply row_161
    exact row_68
  -- chapter_154_line_66: GL tag implication.
  have row_66 : (V1 v11) := by
    apply row_166
    exact row_67
  -- chapter_154_line_65: GL tag equality1.
  have row_65 : (gl_sequence N add zero v11 V2) := by
    have equality_source := row_175
    have equality_step_1 := row_69
    cases equality_step_1
    exact equality_source
  -- chapter_154_line_64: GL tag theorem.
  have row_64 := gauss_source_078 N zero succ add mul one two identity anchor relationalInduction strongRelationalInduction
  -- chapter_154_line_63: GL tag implication.
  have row_63 : (gl_fXY V2 V3 N) := by
    apply row_64
    exact row_68
    exact row_65
  -- chapter_154_line_62: GL tag expansion.
  have row_62 : ((((gl_implication0 V2 V3) ∧ (gl_implication1 V2 N)) ∧ (gl_implication6 V3 N V2)) ∧ (gl_implication7 V3 V2)) := by
    simpa only [gl_fXY] using (row_63)
  -- chapter_154_line_61: GL tag disintegration.
  have row_61 : (gl_implication6 V3 N V2) := by
    exact row_62.1.2
  -- chapter_154_line_60: GL tag expansion.
  have row_60 : (∀ (w1 : α), ((V3 w1) → (gl_existence0 N w1 V2))) := by
    simpa only [gl_implication6] using (row_61)
  -- chapter_154_line_59: GL tag implication.
  have row_59 : (gl_existence0 N previous V2) := by
    apply row_60
    exact row_160
  -- chapter_154_line_58: GL tag expansion.
  have row_58 : (¬ (∀ (v5 : α), ((N v5) → (¬ (V2 previous v5))))) := by
    simpa only [gl_existence0] using (row_59)
  have exists_row_58 : ∃ (v5 : α), ((N v5) ∧ (V2 previous v5)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_58
      intro v5 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v5, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v5, witness_row_58⟩ := exists_row_58
  -- chapter_154_line_57: GL tag disintegration.
  have row_57 : (V2 previous v5) := by
    exact witness_row_58.2
  -- chapter_154_line_56: GL tag disintegration.
  have row_56 : (N v5) := by
    exact witness_row_58.1
  -- chapter_154_line_55: GL tag expansion for integration.
  have row_55 : ((gl_implication122 V3 succ V2 identity add) ↔ (∀ (v25 : α), ((V3 v25) → (∀ (v23 : α), ((succ v25 v23) → ((V3 v23) → (∀ (v26 : α), ((V2 v25 v26) → (∀ (v27 : α), ((identity v23 v27) → (∀ (v24 : α), ((add v26 v27 v24) → (V2 v23 v24))))))))))))) := by
    exact Iff.rfl
  -- chapter_154_line_54: GL tag expansion for integration.
  have row_54 : ((gl_implication121 zero identity V2) ↔ (∀ (v21 : α), ((identity zero v21) → (V2 zero v21)))) := by
    exact Iff.rfl
  -- chapter_154_line_53: GL tag premise element.
  have row_53 : (∀ (v25 : α), ((V3 v25) → (∀ (v23 : α), ((succ v25 v23) → ((V3 v23) → (∀ (v26 : α), ((V2 v25 v26) → (∀ (v27 : α), ((identity v23 v27) → (∀ (v24 : α), ((add v26 v27 v24) → (V3 v23)))))))))))) := by
    intro v25
    intro scope_premise_1
    intro v23
    intro scope_premise_2
    intro scope_premise_3
    intro v26
    intro scope_premise_4
    intro v27
    intro scope_premise_5
    intro v24
    intro scope_premise_6
    exact scope_premise_3
  -- chapter_154_line_52: GL tag implication.
  have row_52 : (∀ (v25 : α), ((V3 v25) → (∀ (v23 : α), ((succ v25 v23) → ((V3 v23) → (∀ (v26 : α), ((V2 v25 v26) → (∀ (v27 : α), ((identity v23 v27) → (∀ (v24 : α), ((add v26 v27 v24) → (V1 v23)))))))))))) := by
    intro v25
    intro scope_premise_1
    intro v23
    intro scope_premise_2
    intro scope_premise_3
    intro v26
    intro scope_premise_4
    intro v27
    intro scope_premise_5
    intro v24
    intro scope_premise_6
    have scoped_fact_2 := row_53 v25 scope_premise_1 v23 scope_premise_2 scope_premise_3 v26 scope_premise_4 v27 scope_premise_5 v24 scope_premise_6
    apply row_166
    exact scoped_fact_2
  -- chapter_154_line_51: GL tag premise element.
  have row_51 : (∀ (v25 : α), ((V3 v25) → (∀ (v23 : α), ((succ v25 v23) → ((V3 v23) → (∀ (v26 : α), ((V2 v25 v26) → (∀ (v27 : α), ((identity v23 v27) → (∀ (v24 : α), ((add v26 v27 v24) → (V3 v25)))))))))))) := by
    intro v25
    intro scope_premise_1
    intro v23
    intro scope_premise_2
    intro scope_premise_3
    intro v26
    intro scope_premise_4
    intro v27
    intro scope_premise_5
    intro v24
    intro scope_premise_6
    exact scope_premise_1
  -- chapter_154_line_50: GL tag implication.
  have row_50 : (∀ (v25 : α), ((V3 v25) → (∀ (v23 : α), ((succ v25 v23) → ((V3 v23) → (∀ (v26 : α), ((V2 v25 v26) → (∀ (v27 : α), ((identity v23 v27) → (∀ (v24 : α), ((add v26 v27 v24) → (V1 v25)))))))))))) := by
    intro v25
    intro scope_premise_1
    intro v23
    intro scope_premise_2
    intro scope_premise_3
    intro v26
    intro scope_premise_4
    intro v27
    intro scope_premise_5
    intro v24
    intro scope_premise_6
    have scoped_fact_2 := row_51 v25 scope_premise_1 v23 scope_premise_2 scope_premise_3 v26 scope_premise_4 v27 scope_premise_5 v24 scope_premise_6
    apply row_166
    exact scoped_fact_2
  -- chapter_154_line_49: GL tag premise element.
  have row_49 : (∀ (v25 : α), ((V3 v25) → (∀ (v23 : α), ((succ v25 v23) → ((V3 v23) → (∀ (v26 : α), ((V2 v25 v26) → (∀ (v27 : α), ((identity v23 v27) → (∀ (v24 : α), ((add v26 v27 v24) → (V2 v25 v26)))))))))))) := by
    intro v25
    intro scope_premise_1
    intro v23
    intro scope_premise_2
    intro scope_premise_3
    intro v26
    intro scope_premise_4
    intro v27
    intro scope_premise_5
    intro v24
    intro scope_premise_6
    exact scope_premise_4
  -- chapter_154_line_48: GL tag implication.
  have row_48 : (∀ (v25 : α), ((V3 v25) → (∀ (v23 : α), ((succ v25 v23) → ((V3 v23) → (∀ (v26 : α), ((V2 v25 v26) → (∀ (v27 : α), ((identity v23 v27) → (∀ (v24 : α), ((add v26 v27 v24) → (V4 v25 v26)))))))))))) := by
    intro v25
    intro scope_premise_1
    intro v23
    intro scope_premise_2
    intro scope_premise_3
    intro v26
    intro scope_premise_4
    intro v27
    intro scope_premise_5
    intro v24
    intro scope_premise_6
    have scoped_fact_2 := row_49 v25 scope_premise_1 v23 scope_premise_2 scope_premise_3 v26 scope_premise_4 v27 scope_premise_5 v24 scope_premise_6
    apply row_176
    exact scoped_fact_2
  -- chapter_154_line_47: GL tag premise element.
  have row_47 : (∀ (v25 : α), ((V3 v25) → (∀ (v23 : α), ((succ v25 v23) → ((V3 v23) → (∀ (v26 : α), ((V2 v25 v26) → (∀ (v27 : α), ((identity v23 v27) → (∀ (v24 : α), ((add v26 v27 v24) → (identity v23 v27)))))))))))) := by
    intro v25
    intro scope_premise_1
    intro v23
    intro scope_premise_2
    intro scope_premise_3
    intro v26
    intro scope_premise_4
    intro v27
    intro scope_premise_5
    intro v24
    intro scope_premise_6
    exact scope_premise_5
  -- chapter_154_line_46: GL tag premise element.
  have row_46 : (∀ (v25 : α), ((V3 v25) → (∀ (v23 : α), ((succ v25 v23) → ((V3 v23) → (∀ (v26 : α), ((V2 v25 v26) → (∀ (v27 : α), ((identity v23 v27) → (∀ (v24 : α), ((add v26 v27 v24) → (add v26 v27 v24)))))))))))) := by
    intro v25
    intro scope_premise_1
    intro v23
    intro scope_premise_2
    intro scope_premise_3
    intro v26
    intro scope_premise_4
    intro v27
    intro scope_premise_5
    intro v24
    intro scope_premise_6
    exact scope_premise_6
  -- chapter_154_line_45: GL tag premise element.
  have row_45 : (∀ (v25 : α), ((V3 v25) → (∀ (v23 : α), ((succ v25 v23) → ((V3 v23) → (∀ (v26 : α), ((V2 v25 v26) → (∀ (v27 : α), ((identity v23 v27) → (∀ (v24 : α), ((add v26 v27 v24) → (succ v25 v23)))))))))))) := by
    intro v25
    intro scope_premise_1
    intro v23
    intro scope_premise_2
    intro scope_premise_3
    intro v26
    intro scope_premise_4
    intro v27
    intro scope_premise_5
    intro v24
    intro scope_premise_6
    exact scope_premise_2
  -- chapter_154_line_44: GL tag implication.
  have row_44 : (∀ (v25 : α), ((V3 v25) → (∀ (v23 : α), ((succ v25 v23) → ((V3 v23) → (∀ (v26 : α), ((V2 v25 v26) → (∀ (v27 : α), ((identity v23 v27) → (∀ (v24 : α), ((add v26 v27 v24) → (V4 v23 v24)))))))))))) := by
    intro v25
    intro scope_premise_1
    intro v23
    intro scope_premise_2
    intro scope_premise_3
    intro v26
    intro scope_premise_4
    intro v27
    intro scope_premise_5
    intro v24
    intro scope_premise_6
    have scoped_fact_2 := row_50 v25 scope_premise_1 v23 scope_premise_2 scope_premise_3 v26 scope_premise_4 v27 scope_premise_5 v24 scope_premise_6
    have scoped_fact_3 := row_45 v25 scope_premise_1 v23 scope_premise_2 scope_premise_3 v26 scope_premise_4 v27 scope_premise_5 v24 scope_premise_6
    have scoped_fact_4 := row_52 v25 scope_premise_1 v23 scope_premise_2 scope_premise_3 v26 scope_premise_4 v27 scope_premise_5 v24 scope_premise_6
    have scoped_fact_5 := row_48 v25 scope_premise_1 v23 scope_premise_2 scope_premise_3 v26 scope_premise_4 v27 scope_premise_5 v24 scope_premise_6
    have scoped_fact_6 := row_47 v25 scope_premise_1 v23 scope_premise_2 scope_premise_3 v26 scope_premise_4 v27 scope_premise_5 v24 scope_premise_6
    have scoped_fact_7 := row_46 v25 scope_premise_1 v23 scope_premise_2 scope_premise_3 v26 scope_premise_4 v27 scope_premise_5 v24 scope_premise_6
    apply row_200
    exact scoped_fact_2
    exact scoped_fact_3
    exact scoped_fact_4
    exact scoped_fact_5
    exact scoped_fact_6
    exact scoped_fact_7
  -- chapter_154_line_43: GL tag implication.
  have row_43 : (∀ (v25 : α), ((V3 v25) → (∀ (v23 : α), ((succ v25 v23) → ((V3 v23) → (∀ (v26 : α), ((V2 v25 v26) → (∀ (v27 : α), ((identity v23 v27) → (∀ (v24 : α), ((add v26 v27 v24) → (gl_preorder N add v23 previous)))))))))))) := by
    intro v25
    intro scope_premise_1
    intro v23
    intro scope_premise_2
    intro scope_premise_3
    intro v26
    intro scope_premise_4
    intro v27
    intro scope_premise_5
    intro v24
    intro scope_premise_6
    have scoped_fact_2 := row_53 v25 scope_premise_1 v23 scope_premise_2 scope_premise_3 v26 scope_premise_4 v27 scope_premise_5 v24 scope_premise_6
    apply row_168
    exact scoped_fact_2
  -- chapter_154_line_42: GL tag implication.
  have row_42 : (∀ (v25 : α), ((V3 v25) → (∀ (v23 : α), ((succ v25 v23) → ((V3 v23) → (∀ (v26 : α), ((V2 v25 v26) → (∀ (v27 : α), ((identity v23 v27) → (∀ (v24 : α), ((add v26 v27 v24) → (V2 v23 v24)))))))))))) := by
    intro v25
    intro scope_premise_1
    intro v23
    intro scope_premise_2
    intro scope_premise_3
    intro v26
    intro scope_premise_4
    intro v27
    intro scope_premise_5
    intro v24
    intro scope_premise_6
    have scoped_fact_2 := row_43 v25 scope_premise_1 v23 scope_premise_2 scope_premise_3 v26 scope_premise_4 v27 scope_premise_5 v24 scope_premise_6
    have scoped_fact_3 := row_44 v25 scope_premise_1 v23 scope_premise_2 scope_premise_3 v26 scope_premise_4 v27 scope_premise_5 v24 scope_premise_6
    apply row_178
    exact scoped_fact_2
    exact scoped_fact_3
  -- chapter_154_line_41: GL tag implication.
  have row_41 : (gl_preorder N add zero previous) := by
    apply row_162
    exact row_160
  have rule_row_40 := peano_source_065 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_154_line_40: GL tag implication.
  have row_40 : (gl_inSequence N one succ) := by
    apply rule_row_40
  -- chapter_154_line_39: GL tag expansion.
  have row_39 : (¬ (∀ (v22 : α), ((N v22) → (¬ (succ v22 one))))) := by
    simpa only [gl_inSequence] using (row_40)
  have exists_row_39 : ∃ (v22 : α), ((N v22) ∧ (succ v22 one)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_39
      intro v22 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v22, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v22, witness_row_39⟩ := exists_row_39
  -- chapter_154_line_38: GL tag disintegration.
  have row_38 : (succ v22 one) := by
    exact witness_row_39.2
  -- chapter_154_line_37: GL tag implication.
  have row_37 : (v22 = zero) := by
    apply row_146
    exact row_101
    exact row_38
    exact row_126
  -- chapter_154_line_36: GL tag implication.
  have row_36 : (zero = v22) := by
    apply row_146
    exact row_101
    exact row_126
    exact row_38
  -- chapter_154_line_35: GL tag implication.
  have row_35 : (identity zero v22) := by
    apply row_153
    exact row_127
    exact row_36
  -- chapter_154_line_34: GL tag implication.
  have row_34 : (V4 zero v22) := by
    apply row_202
    exact row_35
  -- chapter_154_line_33: GL tag implication.
  have row_33 : (V2 zero v22) := by
    apply row_178
    exact row_41
    exact row_34
  -- chapter_154_line_32: GL tag equality1.
  have row_32 : (V2 zero zero) := by
    have equality_source := row_33
    have equality_step_1 := row_37
    cases equality_step_1
    exact equality_source
  -- chapter_154_line_31: GL tag premise element.
  have row_31 : (∀ (v21 : α), ((identity zero v21) → (identity zero v21))) := by
    intro v21
    intro scope_premise_1
    exact scope_premise_1
  -- chapter_154_line_30: GL tag implication.
  have row_30 : (∀ (v21 : α), ((identity zero v21) → (zero = v21))) := by
    intro v21
    intro scope_premise_1
    have scoped_fact_2 := row_31 v21 scope_premise_1
    apply row_155
    exact scoped_fact_2
  -- chapter_154_line_29: GL tag equality1.
  have row_29 : (∀ (v21 : α), ((identity zero v21) → (V2 zero v21))) := by
    intro v21
    intro scope_premise_1
    have scoped_fact_2 := row_30 v21 scope_premise_1
    have equality_source := row_32
    have equality_step_1 := scoped_fact_2
    cases equality_step_1
    exact equality_source
  -- chapter_154_line_28: GL tag expansion for integration.
  have row_28 : ((gl_and0 zero identity V2 previous v5 V3 succ add) ↔ (((gl_implication121 zero identity V2) ∧ (V2 previous v5)) ∧ (gl_implication122 V3 succ V2 identity add))) := by
    exact Iff.rfl
  -- chapter_154_line_27_rule: GL tag reformulation for integration and.
  have row_27_rule : ((gl_implication121 zero identity V2) → ((V2 previous v5) → ((gl_implication122 V3 succ V2 identity add) → (gl_and0 zero identity V2 previous v5 V3 succ add)))) := by
    intro integration_premise_1
    intro integration_premise_2
    intro integration_premise_3
    simp only [gl_and0]
    exact ⟨⟨integration_premise_1, integration_premise_2⟩, integration_premise_3⟩
  -- chapter_154_line_27: GL tag reformulation for integration and.
  have row_27 : (gl_implication123 zero identity V2 previous v5 V3 succ add) := by
    simpa only [gl_implication123] using row_27_rule
  -- chapter_154_line_26: GL tag expansion.
  have row_26 : ((gl_implication121 zero identity V2) → ((V2 previous v5) → ((gl_implication122 V3 succ V2 identity add) → (gl_and0 zero identity V2 previous v5 V3 succ add)))) := by
    simpa only [gl_implication123] using (row_27)
  -- chapter_154_line_25: GL tag validity name.
  have row_25 : (gl_implication121 zero identity V2) := by
    simpa only [gl_implication121] using (row_29)
  -- chapter_154_line_24: GL tag validity name.
  have row_24 : (gl_implication122 V3 succ V2 identity add) := by
    simpa only [gl_implication122] using (row_42)
  -- chapter_154_line_23: GL tag implication.
  have row_23 : (gl_and0 zero identity V2 previous v5 V3 succ add) := by
    apply row_26
    exact row_25
    exact row_57
    exact row_24
  -- chapter_154_line_22: GL tag expansion for integration.
  have row_22 : ((gl_existence2 V3 N zero identity previous v5 succ add) ↔ (¬ (∀ (V2 : GLBinaryRelation α), ((gl_fXY V2 V3 N) → (¬ (gl_and0 zero identity V2 previous v5 V3 succ add)))))) := by
    exact Iff.rfl
  -- chapter_154_line_21_rule: GL tag reformulation for integration >[].
  have row_21_rule : ((gl_fXY V2 V3 N) → ((gl_and0 zero identity V2 previous v5 V3 succ add) → (gl_existence2 V3 N zero identity previous v5 succ add))) := by
    intro integration_premise_1
    intro integration_premise_2
    apply (row_22).2
    intro universal_counterexample
    exact universal_counterexample V2 integration_premise_1 integration_premise_2
  -- chapter_154_line_21: GL tag reformulation for integration >[].
  have row_21 : (gl_implication125 V2 V3 N zero identity previous v5 succ add) := by
    simpa only [gl_implication125] using row_21_rule
  -- chapter_154_line_20: GL tag expansion.
  have row_20 : ((gl_fXY V2 V3 N) → ((gl_and0 zero identity V2 previous v5 V3 succ add) → (gl_existence2 V3 N zero identity previous v5 succ add))) := by
    simpa only [gl_implication125] using (row_21)
  -- chapter_154_line_19: GL tag implication.
  have row_19 : (gl_existence2 V3 N zero identity previous v5 succ add) := by
    apply row_20
    exact row_63
    exact row_23
  -- chapter_154_line_18: GL tag expansion for integration.
  have row_18 : ((gl_fold N succ add add identity zero previous v5) ↔ (¬ (∀ (V3 : GLSet α), ((gl_interval N add zero previous V3) → (¬ (gl_existence2 V3 N zero identity previous v5 succ add)))))) := by
    exact Iff.rfl
  -- chapter_154_line_17_rule: GL tag reformulation for integration >[].
  have row_17_rule : ((gl_interval N add zero previous V3) → ((gl_existence2 V3 N zero identity previous v5 succ add) → (gl_fold N succ add add identity zero previous v5))) := by
    intro integration_premise_1
    intro integration_premise_2
    apply (row_18).2
    intro universal_counterexample
    exact universal_counterexample V3 integration_premise_1 integration_premise_2
  -- chapter_154_line_17: GL tag reformulation for integration >[].
  have row_17 : (gl_implication129 N add zero previous V3 identity v5 succ add) := by
    simpa only [gl_implication129] using row_17_rule
  -- chapter_154_line_16: GL tag expansion.
  have row_16 : ((gl_interval N add zero previous V3) → ((gl_existence2 V3 N zero identity previous v5 succ add) → (gl_fold N succ add add identity zero previous v5))) := by
    simpa only [gl_implication129] using (row_17)
  -- chapter_154_line_15: GL tag implication.
  have row_15 : (gl_fold N succ add add identity zero previous v5) := by
    apply row_16
    exact row_165
    exact row_19
  -- chapter_154_line_14: GL tag recursion.
  have row_14 : (∀ (w1 : α) (w2 : α), ((mul w1 previous w2) → (∀ (w3 : α), ((gl_fold N succ add add identity zero previous w3) → ((succ previous w1) → (mul two w3 w2)))))) := by
    exact assumption_14
  -- chapter_154_line_13: GL tag implication.
  have row_13 : (mul two v5 v7) := by
    apply row_14
    exact row_113
    exact row_15
    exact row_185
  -- chapter_154_line_12: GL tag equality1.
  have row_12 : (V2 v11 v5) := by
    have equality_source := row_57
    have equality_step_1 := row_69
    cases equality_step_1
    exact equality_source
  -- chapter_154_line_11: GL tag implication.
  have row_11 : (V4 v11 v5) := by
    apply row_176
    exact row_12
  -- chapter_154_line_10: GL tag implication.
  have row_10 : (gl_existence1 N v5 v2 add) := by
    apply row_128
    exact row_56
    exact row_190
  -- chapter_154_line_9: GL tag expansion.
  have row_9 : (¬ (∀ (v8 : α), ((N v8) → (¬ (add v5 v2 v8))))) := by
    simpa only [gl_existence1] using (row_10)
  have exists_row_9 : ∃ (v8 : α), ((N v8) ∧ (add v5 v2 v8)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_9
      intro v8 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v8, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v8, witness_row_9⟩ := exists_row_9
  -- chapter_154_line_8: GL tag disintegration.
  have row_8 : (add v5 v2 v8) := by
    exact witness_row_9.2
  -- chapter_154_line_7: GL tag equality1.
  have row_7 : (add v5 v15 v8) := by
    have equality_source := row_8
    have equality_step_1 := row_80
    cases equality_step_1
    exact equality_source
  -- chapter_154_line_6: GL tag implication.
  have row_6 : (V4 v2 v8) := by
    apply row_200
    exact row_66
    exact row_71
    exact row_189
    exact row_11
    exact row_70
    exact row_7
  -- chapter_154_line_5: GL tag implication.
  have row_5 : (add v2 v5 v8) := by
    apply row_82
    exact row_8
  -- chapter_154_line_4: GL tag implication.
  have row_4 : (v8 = v4) := by
    apply row_193
    exact row_189
    exact row_6
    exact row_199
  -- chapter_154_line_3: GL tag equality1.
  have row_3 : (add v2 v5 v4) := by
    have equality_source := row_5
    have equality_step_1 := row_4
    cases equality_step_1
    exact equality_source
  have rule_row_2 := peano_source_054 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_154_line_2: GL tag implication.
  have row_2 : (mul two v2 v6) := by
    apply rule_row_2
    exact row_117
  have rule_row_1 := peano_source_050 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_154_line_1: GL tag implication.
  have row_1 : (mul two v4 v3) := by
    apply rule_row_1
    exact row_93
    exact row_3
    exact row_2
    exact row_13
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α) (v3 : α), ((mul v1 v2 v3) → (∀ (v4 : α), ((gl_fold N succ add add identity zero v2 v4) → ((succ v2 v1) → (mul two v4 v3)))))) := by
  intro v1
  intro v2
  intro v3
  intro premise_1
  intro v4
  intro premise_2
  intro premise_3
  have inductionMember : N v2 := by
    have typingRule := gauss_source_080_induction_typing N zero succ add mul one two identity anchor relationalInduction strongRelationalInduction
    exact typingRule v2 v4 premise_2
  have inductionZeroMember : N zero := by
    simp only [gl_AnchorGauss, gl_NaturalNumbers] at anchor
    exact anchor.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1
  have inductionBase : (∀ (v1 : α) (v3 : α), ((mul v1 zero v3) → (∀ (v4 : α), ((gl_fold N succ add add identity zero zero v4) → ((succ zero v1) → (mul two v4 v3)))))) := by
    intro v1
    intro v3
    intro base_premise_1
    intro v4
    intro base_premise_2
    intro base_premise_3
    have zeroRule := gauss_source_080_check_zero N zero succ add mul one two identity anchor relationalInduction strongRelationalInduction
    exact zeroRule v1 zero v3 v4 base_premise_2 rfl base_premise_3 base_premise_1
  have inductionStep :
      ∀ induction_n, N induction_n → (∀ (v1 : α) (v3 : α), ((mul v1 induction_n v3) → (∀ (v4 : α), ((gl_fold N succ add add identity zero induction_n v4) → ((succ induction_n v1) → (mul two v4 v3)))))) → ∀ induction_m, succ induction_n induction_m → (∀ (v1 : α) (v3 : α), ((mul v1 induction_m v3) → (∀ (v4 : α), ((gl_fold N succ add add identity zero induction_m v4) → ((succ induction_m v1) → (mul two v4 v3)))))) := by
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
        (∀ (w1 : α) (w2 : α), ((mul w1 induction_n w2) → (∀ (w3 : α), ((gl_fold N succ add add identity zero induction_n w3) → ((succ induction_n w1) → (mul two w3 w2)))))) := by
      intro w1
      intro w2
      intro step_induction_assumption_2_premise_1
      intro w3
      intro step_induction_assumption_2_premise_2
      intro step_induction_assumption_2_premise_3
      apply induction_hypothesis
      all_goals assumption
    have stepRule := gauss_source_080_check_induction_condition N zero succ add mul one two identity anchor relationalInduction strongRelationalInduction
    exact stepRule induction_n v1 induction_m v3 v4 step_premise_2 step_induction_assumption_1 step_premise_1 step_premise_3 step_induction_assumption_2
  have inductionProperty : (∀ (v1 : α) (v3 : α), ((mul v1 v2 v3) → (∀ (v4 : α), ((gl_fold N succ add add identity zero v2 v4) → ((succ v2 v1) → (mul two v4 v3)))))) := by
    exact relationalInduction
      (fun induction_value => (∀ (v1 : α) (v3 : α), ((mul v1 induction_value v3) → (∀ (v4 : α), ((gl_fold N succ add add identity zero induction_value v4) → ((succ induction_value v1) → (mul two v4 v3)))))))
      v2
      inductionBase
      inductionStep
      inductionMember
  exact inductionProperty v1 v3 premise_1 v4 premise_2 premise_3

theorem gauss_source_099
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
    (strongRelationalInduction :
      ∀ (P : α → Prop) (k : α),
        P zero →
        (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
        N k →
        P k)
    : (∀ (v1 : α) (v2 : α), ((gl_fold N succ add add identity zero v1 v2) → (∀ (v3 : α), ((succ v1 v3) → (∀ (v4 : α), ((mul two v2 v4) → (mul v1 v3 v4))))))) := by
  intro v1
  intro v2
  intro premise_1
  intro v3
  intro premise_2
  intro v4
  intro premise_3
  -- chapter_173_line_83: GL tag task formulation.
  have row_83 : (gl_fold N succ add add identity zero v1 v2) := by
    exact premise_1
  -- chapter_173_line_82: GL tag expansion.
  have row_82 : (¬ (∀ (V1 : GLSet α), ((gl_interval N add zero v1 V1) → (¬ (gl_existence2 V1 N zero identity v1 v2 succ add))))) := by
    simpa only [gl_fold] using (row_83)
  have exists_row_82 : ∃ (V1 : GLSet α), ((gl_interval N add zero v1 V1) ∧ (gl_existence2 V1 N zero identity v1 v2 succ add)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_82
      intro V1 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨V1, counterexample_premise_0, counterexample_head⟩
  obtain ⟨V1, witness_row_82⟩ := exists_row_82
  -- chapter_173_line_81: GL tag disintegration.
  have row_81 : (gl_existence2 V1 N zero identity v1 v2 succ add) := by
    exact witness_row_82.2
  -- chapter_173_line_80: GL tag expansion.
  have row_80 : (¬ (∀ (V2 : GLBinaryRelation α), ((gl_fXY V2 V1 N) → (¬ (gl_and0 zero identity V2 v1 v2 V1 succ add))))) := by
    simpa only [gl_existence2] using (row_81)
  have exists_row_80 : ∃ (V2 : GLBinaryRelation α), ((gl_fXY V2 V1 N) ∧ (gl_and0 zero identity V2 v1 v2 V1 succ add)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_80
      intro V2 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨V2, counterexample_premise_0, counterexample_head⟩
  obtain ⟨V2, witness_row_80⟩ := exists_row_80
  -- chapter_173_line_79: GL tag disintegration.
  have row_79 : (gl_and0 zero identity V2 v1 v2 V1 succ add) := by
    exact witness_row_80.2
  -- chapter_173_line_78: GL tag expansion.
  have row_78 : (((gl_implication121 zero identity V2) ∧ (V2 v1 v2)) ∧ (gl_implication122 V1 succ V2 identity add)) := by
    simpa only [gl_and0] using (row_79)
  -- chapter_173_line_77: GL tag disintegration.
  have row_77 : (V2 v1 v2) := by
    exact row_78.1.2
  -- chapter_173_line_76: GL tag disintegration.
  have row_76 : (gl_fXY V2 V1 N) := by
    exact witness_row_80.1
  -- chapter_173_line_75: GL tag expansion.
  have row_75 : ((((gl_implication0 V2 V1) ∧ (gl_implication1 V2 N)) ∧ (gl_implication6 V1 N V2)) ∧ (gl_implication7 V1 V2)) := by
    simpa only [gl_fXY] using (row_76)
  -- chapter_173_line_74: GL tag disintegration.
  have row_74 : (gl_implication1 V2 N) := by
    exact row_75.1.1.2
  -- chapter_173_line_73: GL tag expansion.
  have row_73 : (∀ (w1 : α) (w2 : α), ((V2 w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using (row_74)
  -- chapter_173_line_72: GL tag disintegration.
  have row_72 : (gl_interval N add zero v1 V1) := by
    exact witness_row_82.1
  -- chapter_173_line_71: GL tag expansion.
  have row_71 : (((((gl_implication89 V1 N add zero) ∧ (gl_implication90 V1 N add v1)) ∧ (gl_implication91 N add zero v1 V1)) ∧ (N zero)) ∧ (N v1)) := by
    simpa only [gl_interval] using (row_72)
  -- chapter_173_line_70: GL tag disintegration.
  have row_70 : (N v1) := by
    exact row_71.2
  -- chapter_173_line_69: GL tag implication.
  have row_69 : (N v2) := by
    apply row_73
    exact row_77
  -- chapter_173_line_68: GL tag task formulation.
  have row_68 : (gl_AnchorGauss N zero succ add mul one two identity) := by
    exact anchor
  -- chapter_173_line_67: GL tag expansion.
  have row_67 : ((((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one)) ∧ (succ one two)) ∧ (gl_identity N identity)) := by
    simpa only [gl_AnchorGauss] using (row_68)
  -- chapter_173_line_66: GL tag disintegration.
  have row_66 : (gl_NaturalNumbers N zero succ add mul) := by
    exact row_67.1.1.1
  -- chapter_173_line_65: GL tag expansion.
  have row_65 : ((((((((((((((N zero) ∧ (gl_fXY succ N N)) ∧ (gl_implication9 N zero succ)) ∧ (gl_implication10 N succ)) ∧ (gl_fXYZ add N N N)) ∧ (gl_implication21 N zero add)) ∧ (gl_implication22 N zero add)) ∧ (gl_implication23 N succ add)) ∧ (gl_implication24 N succ add)) ∧ (gl_fXYZ mul N N N)) ∧ (gl_implication25 N zero mul)) ∧ (gl_implication26 zero N mul)) ∧ (gl_implication27 N succ mul add)) ∧ (gl_implication28 N succ mul add)) := by
    simpa only [gl_NaturalNumbers] using (row_66)
  -- chapter_173_line_64: GL tag disintegration.
  have row_64 : (gl_implication23 N succ add) := by
    exact row_65.1.1.1.1.1.1.2
  -- chapter_173_line_63: GL tag expansion.
  have row_63 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w1 w2) → (∀ (w3 : α) (w4 : α), ((add w3 w1 w4) → (∀ (w5 : α), ((add w3 w2 w5) → (succ w4 w5))))))))) := by
    simpa only [gl_implication23] using (row_64)
  -- chapter_173_line_62: GL tag disintegration.
  have row_62 : (gl_fXYZ add N N N) := by
    exact row_65.1.1.1.1.1.1.1.1.1.2
  -- chapter_173_line_61: GL tag expansion.
  have row_61 : (((((gl_implication11 add N) ∧ (gl_implication12 add N)) ∧ (gl_implication13 add N)) ∧ (gl_implication18 N N N add)) ∧ (gl_implication19 N N add)) := by
    simpa only [gl_fXYZ] using (row_62)
  -- chapter_173_line_60: GL tag disintegration.
  have row_60 : (gl_implication18 N N N add) := by
    exact row_61.1.2
  -- chapter_173_line_59: GL tag expansion.
  have row_59 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 add))))) := by
    simpa only [gl_implication18] using (row_60)
  -- chapter_173_line_58: GL tag disintegration.
  have row_58 : (N zero) := by
    exact row_65.1.1.1.1.1.1.1.1.1.1.1.1.1
  -- chapter_173_line_57: GL tag disintegration.
  have row_57 : (gl_implication21 N zero add) := by
    exact row_65.1.1.1.1.1.1.1.1.2
  -- chapter_173_line_56: GL tag expansion.
  have row_56 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((add w1 zero w2) → (w1 = w2))))) := by
    simpa only [gl_implication21] using (row_57)
  -- chapter_173_line_55: GL tag disintegration.
  have row_55 : (gl_implication10 N succ) := by
    exact row_65.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_173_line_54: GL tag expansion.
  have row_54 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((succ w2 w1) → (∀ (w3 : α), ((succ w3 w1) → (w2 = w3))))))) := by
    simpa only [gl_implication10] using (row_55)
  -- chapter_173_line_53: GL tag disintegration.
  have row_53 : (gl_fXY succ N N) := by
    exact row_65.1.1.1.1.1.1.1.1.1.1.1.1.2
  -- chapter_173_line_52: GL tag expansion.
  have row_52 : ((((gl_implication0 succ N) ∧ (gl_implication1 succ N)) ∧ (gl_implication6 N N succ)) ∧ (gl_implication7 N succ)) := by
    simpa only [gl_fXY] using (row_53)
  -- chapter_173_line_51: GL tag disintegration.
  have row_51 : (gl_implication1 succ N) := by
    exact row_52.1.1.2
  -- chapter_173_line_50: GL tag expansion.
  have row_50 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w2))) := by
    simpa only [gl_implication1] using (row_51)
  -- chapter_173_line_49: GL tag disintegration.
  have row_49 : (gl_implication0 succ N) := by
    exact row_52.1.1.1
  -- chapter_173_line_48: GL tag expansion.
  have row_48 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (N w1))) := by
    simpa only [gl_implication0] using (row_49)
  -- chapter_173_line_47: GL tag disintegration.
  have row_47 : (succ one two) := by
    exact row_67.1.2
  -- chapter_173_line_46: GL tag disintegration.
  have row_46 : (gl_fXYZ mul N N N) := by
    exact row_65.1.1.1.1.2
  -- chapter_173_line_45: GL tag expansion.
  have row_45 : (((((gl_implication11 mul N) ∧ (gl_implication12 mul N)) ∧ (gl_implication13 mul N)) ∧ (gl_implication18 N N N mul)) ∧ (gl_implication19 N N mul)) := by
    simpa only [gl_fXYZ] using (row_46)
  -- chapter_173_line_44: GL tag disintegration.
  have row_44 : (gl_implication18 N N N mul) := by
    exact row_45.1.2
  -- chapter_173_line_43: GL tag expansion.
  have row_43 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (gl_existence1 N w1 w2 mul))))) := by
    simpa only [gl_implication18] using (row_44)
  -- chapter_173_line_42: GL tag disintegration.
  have row_42 : (succ zero one) := by
    exact row_67.1.1.2
  -- chapter_173_line_41: GL tag disintegration.
  have row_41 : (gl_implication19 N N mul) := by
    exact row_45.2
  -- chapter_173_line_40: GL tag expansion.
  have row_40 : (∀ (w1 : α), ((N w1) → (∀ (w2 : α), ((N w2) → (∀ (w3 : α), ((mul w1 w2 w3) → (∀ (w4 : α), ((mul w1 w2 w4) → (w3 = w4))))))))) := by
    simpa only [gl_implication19] using (row_41)
  -- chapter_173_line_39: GL tag implication.
  have row_39 : (gl_existence1 N v1 zero add) := by
    apply row_59
    exact row_70
    exact row_58
  -- chapter_173_line_38: GL tag expansion.
  have row_38 : (¬ (∀ (v10 : α), ((N v10) → (¬ (add v1 zero v10))))) := by
    simpa only [gl_existence1] using (row_39)
  have exists_row_38 : ∃ (v10 : α), ((N v10) ∧ (add v1 zero v10)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_38
      intro v10 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v10, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v10, witness_row_38⟩ := exists_row_38
  -- chapter_173_line_37: GL tag disintegration.
  have row_37 : (add v1 zero v10) := by
    exact witness_row_38.2
  -- chapter_173_line_36: GL tag implication.
  have row_36 : (v1 = v10) := by
    apply row_56
    exact row_70
    exact row_37
  -- chapter_173_line_35: GL tag implication grouping.
  have row_35 : (∀ (w1 : α) (w2 : α), ((succ w1 w2) → (gl_and20 w1 N w2 N))) := by
    have auxiliary_source_1 := row_48
    have auxiliary_source_2 := row_50
    intro auxiliary_argument_1
    intro auxiliary_argument_2
    intro auxiliary_premise
    exact ⟨(auxiliary_source_1 auxiliary_argument_1 auxiliary_argument_2 auxiliary_premise), (auxiliary_source_2 auxiliary_argument_1 auxiliary_argument_2 auxiliary_premise)⟩
  -- chapter_173_line_34: GL tag implication.
  have row_34 : (gl_and20 one N two N) := by
    apply row_35
    exact row_47
  -- chapter_173_line_33: GL tag task formulation.
  have row_33 : (succ v1 v3) := by
    exact premise_2
  -- chapter_173_line_32: GL tag implication.
  have row_32 : (gl_and20 v1 N v3 N) := by
    apply row_35
    exact row_33
  -- chapter_173_line_31: GL tag expansion.
  have row_31 : ((N v1) ∧ (N v3)) := by
    simpa only [gl_and20] using (row_32)
  -- chapter_173_line_30: GL tag disintegration.
  have row_30 : (N v3) := by
    exact row_31.2
  -- chapter_173_line_29: GL tag implication.
  have row_29 : (gl_existence1 N v3 v1 mul) := by
    apply row_43
    exact row_30
    exact row_70
  -- chapter_173_line_28: GL tag expansion for integration.
  have row_28 : ((gl_AnchorPeano N zero succ add mul one) ↔ ((gl_NaturalNumbers N zero succ add mul) ∧ (succ zero one))) := by
    exact Iff.rfl
  -- chapter_173_line_27_rule: GL tag reformulation for integration and.
  have row_27_rule : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    intro integration_premise_1
    intro integration_premise_2
    simp only [gl_AnchorPeano]
    exact ⟨integration_premise_1, integration_premise_2⟩
  -- chapter_173_line_27: GL tag reformulation for integration and.
  have row_27 : (gl_implication76 N zero succ add mul one) := by
    simpa only [gl_implication76] using row_27_rule
  -- chapter_173_line_26: GL tag expansion.
  have row_26 : ((gl_NaturalNumbers N zero succ add mul) → ((succ zero one) → (gl_AnchorPeano N zero succ add mul one))) := by
    simpa only [gl_implication76] using (row_27)
  -- chapter_173_line_25: GL tag implication.
  have row_25 : (gl_AnchorPeano N zero succ add mul one) := by
    apply row_26
    exact row_66
    exact row_42
  have rule_row_24 := peano_source_010 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_173_line_24: GL tag implication.
  have row_24 : (add one v1 v3) := by
    apply rule_row_24
    exact row_33
  have rule_row_23 := peano_source_012 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_173_line_23: GL tag implication.
  have row_23 : (add v1 one v3) := by
    apply rule_row_23
    exact row_24
  -- chapter_173_line_22: GL tag implication.
  have row_22 : (gl_and20 zero N one N) := by
    apply row_35
    exact row_42
  -- chapter_173_line_21: GL tag expansion.
  have row_21 : ((N zero) ∧ (N one)) := by
    simpa only [gl_and20] using (row_22)
  -- chapter_173_line_20: GL tag disintegration.
  have row_20 : (N one) := by
    exact row_21.2
  have rule_row_19 := peano_source_065 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_173_line_19: GL tag implication.
  have row_19 : (gl_inSequence N one succ) := by
    apply rule_row_19
  -- chapter_173_line_18: GL tag expansion.
  have row_18 : (¬ (∀ (v12 : α), ((N v12) → (¬ (succ v12 one))))) := by
    simpa only [gl_inSequence] using (row_19)
  have exists_row_18 : ∃ (v12 : α), ((N v12) ∧ (succ v12 one)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_18
      intro v12 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v12, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v12, witness_row_18⟩ := exists_row_18
  -- chapter_173_line_17: GL tag disintegration.
  have row_17 : (succ v12 one) := by
    exact witness_row_18.2
  -- chapter_173_line_16: GL tag disintegration.
  have row_16 : (N v12) := by
    exact witness_row_18.1
  -- chapter_173_line_15: GL tag implication.
  have row_15 : (zero = v12) := by
    apply row_54
    exact row_20
    exact row_42
    exact row_17
  -- chapter_173_line_14: GL tag equality1.
  have row_14 : (add v1 v12 v10) := by
    have equality_source := row_37
    have equality_step_1 := row_15
    cases equality_step_1
    exact equality_source
  -- chapter_173_line_13: GL tag implication.
  have row_13 : (succ v10 v3) := by
    apply row_63
    exact row_16
    exact row_17
    exact row_14
    exact row_23
  -- chapter_173_line_12: GL tag expansion.
  have row_12 : (¬ (∀ (v5 : α), ((N v5) → (¬ (mul v3 v1 v5))))) := by
    simpa only [gl_existence1] using (row_29)
  have exists_row_12 : ∃ (v5 : α), ((N v5) ∧ (mul v3 v1 v5)) := by
      classical
      apply Classical.byContradiction
      intro no_counterexample
      apply row_12
      intro v5 counterexample_premise_0
      intro counterexample_head
      exact no_counterexample ⟨v5, counterexample_premise_0, counterexample_head⟩
  obtain ⟨v5, witness_row_12⟩ := exists_row_12
  -- chapter_173_line_11: GL tag disintegration.
  have row_11 : (mul v3 v1 v5) := by
    exact witness_row_12.2
  -- chapter_173_line_10: GL tag equality1.
  have row_10 : (mul v3 v10 v5) := by
    have equality_source := row_11
    have equality_step_1 := row_36
    cases equality_step_1
    exact equality_source
  -- chapter_173_line_9: GL tag equality1.
  have row_9 : (gl_fold N succ add add identity zero v10 v2) := by
    have equality_source := row_83
    have equality_step_1 := row_36
    cases equality_step_1
    exact equality_source
  -- chapter_173_line_8: GL tag theorem.
  have row_8 := gauss_source_080 N zero succ add mul one two identity anchor relationalInduction strongRelationalInduction
  -- chapter_173_line_7: GL tag implication.
  have row_7 : (mul two v2 v5) := by
    apply row_8
    exact row_10
    exact row_9
    exact row_13
  -- chapter_173_line_6: GL tag expansion.
  have row_6 : ((N one) ∧ (N two)) := by
    simpa only [gl_and20] using (row_34)
  -- chapter_173_line_5: GL tag disintegration.
  have row_5 : (N two) := by
    exact row_6.2
  -- chapter_173_line_4: GL tag task formulation.
  have row_4 : (mul two v2 v4) := by
    exact premise_3
  -- chapter_173_line_3: GL tag implication.
  have row_3 : (v5 = v4) := by
    apply row_40
    exact row_5
    exact row_69
    exact row_7
    exact row_4
  have rule_row_2 := peano_source_054 N zero succ add mul one (anchorPeanoOfGauss N zero succ add mul one two identity anchor) relationalInduction strongRelationalInduction
  -- chapter_173_line_2: GL tag implication.
  have row_2 : (mul v1 v3 v5) := by
    apply rule_row_2
    exact row_11
  -- chapter_173_line_1: GL tag equality1.
  have row_1 : (mul v1 v3 v4) := by
    have equality_source := row_2
    have equality_step_1 := row_3
    cases equality_step_1
    exact equality_source
  exact row_1

end GLExport
