/-
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-/

import GLExport.ProofSupport

namespace GLExport

universe u

def gl_existence0 {α : Type u} (u_1 : GLSet α) (u_2 : α) (u_3 : GLBinaryRelation α) : Prop :=
  (¬ (∀ (x_1 : α), ((u_1 x_1) → (¬ (u_3 u_2 x_1)))))

def gl_existence1 {α : Type u} (u_1 : GLSet α) (u_2 : α) (u_3 : α) (u_4 : GLTernaryRelation α) : Prop :=
  (¬ (∀ (x_1 : α), ((u_1 x_1) → (¬ (u_4 u_2 u_3 x_1)))))

def gl_existence11 {α : Type u} (u_1 : GLSet α) (u_2 : α) (u_3 : GLBinaryRelation α) : Prop :=
  (¬ (∀ (x_1 : α), ((u_1 x_1) → (¬ (u_3 x_1 u_2)))))

def gl_implication0 {α : Type u} (u_1 : GLBinaryRelation α) (u_2 : GLSet α) : Prop :=
  (∀ (x_1 : α) (x_2 : α), ((u_1 x_1 x_2) → (u_2 x_1)))

def gl_implication1 {α : Type u} (u_1 : GLBinaryRelation α) (u_2 : GLSet α) : Prop :=
  (∀ (x_1 : α) (x_2 : α), ((u_1 x_1 x_2) → (u_2 x_2)))

def gl_implication10 {α : Type u} (u_1 : GLTernaryRelation α) (u_2 : GLSet α) : Prop :=
  (∀ (x_1 : α) (x_2 : α) (x_3 : α), ((u_1 x_1 x_2 x_3) → (u_2 x_3)))

def gl_implication14 {α : Type u} (u_1 : GLSet α) (u_2 : GLSet α) (u_3 : GLTernaryRelation α) : Prop :=
  (∀ (x_1 : α), ((u_1 x_1) → (∀ (x_2 : α), ((u_2 x_2) → (∀ (x_3 : α), ((u_3 x_1 x_2 x_3) → (∀ (x_4 : α), ((u_3 x_1 x_2 x_4) → (x_3 = x_4)))))))))

def gl_implication15 {α : Type u} (u_1 : GLSet α) (u_2 : α) (u_3 : GLTernaryRelation α) : Prop :=
  (∀ (x_1 : α), ((u_1 x_1) → (∀ (x_2 : α), ((u_3 x_1 u_2 x_2) → (x_1 = x_2)))))

def gl_implication16 {α : Type u} (u_1 : GLSet α) (u_2 : α) (u_3 : GLTernaryRelation α) : Prop :=
  (∀ (x_1 : α) (x_2 : α), ((x_1 = x_2) → ((u_1 x_1) → ((u_1 x_2) → (u_3 x_1 u_2 x_2)))))

def gl_implication17 {α : Type u} (u_1 : GLSet α) (u_2 : GLBinaryRelation α) (u_3 : GLTernaryRelation α) : Prop :=
  (∀ (x_1 : α), ((u_1 x_1) → (∀ (x_2 : α), ((u_2 x_1 x_2) → (∀ (x_3 : α) (x_4 : α), ((u_3 x_3 x_1 x_4) → (∀ (x_5 : α), ((u_3 x_3 x_2 x_5) → (u_2 x_4 x_5)))))))))

def gl_implication18 {α : Type u} (u_1 : GLSet α) (u_2 : GLBinaryRelation α) (u_3 : GLTernaryRelation α) : Prop :=
  (∀ (x_1 : α), ((u_1 x_1) → (∀ (x_2 : α), ((u_2 x_1 x_2) → (∀ (x_3 : α) (x_4 : α), ((u_3 x_3 x_1 x_4) → (∀ (x_5 : α), ((u_2 x_4 x_5) → (u_3 x_3 x_2 x_5)))))))))

def gl_implication19 {α : Type u} (u_1 : GLSet α) (u_2 : α) (u_3 : GLTernaryRelation α) : Prop :=
  (∀ (x_1 : α), ((u_1 x_1) → (∀ (x_2 : α), ((u_3 x_1 u_2 x_2) → (x_2 = u_2)))))

def gl_implication20 {α : Type u} (u_1 : GLSet α) (u_2 : GLBinaryRelation α) (u_3 : GLTernaryRelation α) (u_4 : GLTernaryRelation α) : Prop :=
  (∀ (x_1 : α), ((u_1 x_1) → (∀ (x_2 : α), ((u_2 x_1 x_2) → (∀ (x_3 : α) (x_4 : α), ((u_3 x_3 x_1 x_4) → (∀ (x_5 : α), ((u_4 x_4 x_3 x_5) → (u_3 x_3 x_2 x_5)))))))))

def gl_implication21 {α : Type u} (u_1 : GLSet α) (u_2 : GLBinaryRelation α) (u_3 : GLTernaryRelation α) (u_4 : GLTernaryRelation α) : Prop :=
  (∀ (x_1 : α), ((u_1 x_1) → (∀ (x_2 : α), ((u_2 x_1 x_2) → (∀ (x_3 : α) (x_4 : α), ((u_3 x_3 x_1 x_4) → (∀ (x_5 : α), ((u_3 x_3 x_2 x_5) → (u_4 x_4 x_3 x_5)))))))))

def gl_implication22 {α : Type u} (u_1 : GLBinaryRelation α) : Prop :=
  (∀ (x_1 : α) (x_2 : α), ((u_1 x_1 x_2) → (x_1 = x_2)))

def gl_implication23 {α : Type u} (u_1 : GLSet α) (u_2 : GLBinaryRelation α) : Prop :=
  (∀ (x_1 : α), ((u_1 x_1) → (∀ (x_2 : α), ((x_1 = x_2) → (u_2 x_1 x_2)))))

def gl_implication24 {α : Type u} (u_1 : GLSet α) (u_2 : α) (u_3 : α) (u_4 : GLTernaryRelation α) : Prop :=
  (∀ (x_1 : α), ((u_1 x_1) → (¬ (u_4 u_2 x_1 u_3))))

def gl_implication5 {α : Type u} (u_1 : GLSet α) (u_2 : GLBinaryRelation α) : Prop :=
  (∀ (x_1 : α), ((u_1 x_1) → (∀ (x_2 : α), ((u_2 x_1 x_2) → (∀ (x_3 : α), ((u_2 x_1 x_3) → (x_2 = x_3)))))))

def gl_implication6 {α : Type u} (u_1 : GLSet α) (u_2 : α) (u_3 : GLBinaryRelation α) : Prop :=
  (∀ (x_1 : α), ((u_1 x_1) → (¬ (u_3 x_1 u_2))))

def gl_implication7 {α : Type u} (u_1 : GLSet α) (u_2 : GLBinaryRelation α) : Prop :=
  (∀ (x_1 : α), ((u_1 x_1) → (∀ (x_2 : α), ((u_2 x_2 x_1) → (∀ (x_3 : α), ((u_2 x_3 x_1) → (x_2 = x_3)))))))

def gl_implication8 {α : Type u} (u_1 : GLTernaryRelation α) (u_2 : GLSet α) : Prop :=
  (∀ (x_1 : α) (x_2 : α) (x_3 : α), ((u_1 x_1 x_2 x_3) → (u_2 x_1)))

def gl_implication9 {α : Type u} (u_1 : GLTernaryRelation α) (u_2 : GLSet α) : Prop :=
  (∀ (x_1 : α) (x_2 : α) (x_3 : α), ((u_1 x_1 x_2 x_3) → (u_2 x_2)))

def gl_or4 {α : Type u} (u_1 : α) (u_2 : α) (u_3 : α) : Prop :=
  ((u_1 = u_2) ∨ (u_1 = u_3))

def gl_preorder {α : Type u} (u_1 : GLSet α) (u_2 : GLTernaryRelation α) (u_3 : α) (u_4 : α) : Prop :=
  (¬ (∀ (x_1 : α), ((u_1 x_1) → (¬ (u_2 u_3 x_1 u_4)))))

def gl_identity {α : Type u} (u_1 : GLSet α) (u_2 : GLBinaryRelation α) : Prop :=
  (((gl_implication0 u_2 u_1) ∧ (gl_implication22 u_2)) ∧ (gl_implication23 u_1 u_2))

def gl_implication13 {α : Type u} (u_1 : GLSet α) (u_2 : GLSet α) (u_3 : GLSet α) (u_4 : GLTernaryRelation α) : Prop :=
  (∀ (x_1 : α), ((u_1 x_1) → (∀ (x_2 : α), ((u_2 x_2) → (gl_existence1 u_3 x_1 x_2 u_4)))))

def gl_implication4 {α : Type u} (u_1 : GLSet α) (u_2 : GLSet α) (u_3 : GLBinaryRelation α) : Prop :=
  (∀ (x_1 : α), ((u_1 x_1) → (gl_existence0 u_2 x_1 u_3)))

def gl_implication74 {α : Type u} (u_1 : α) (u_2 : α) (u_3 : GLSet α) (u_4 : GLBinaryRelation α) : Prop :=
  ((¬ (u_1 = u_2)) → (gl_existence11 u_3 u_1 u_4))

def gl_or2 {α : Type u} (u_1 : α) (u_2 : α) (u_3 : GLSet α) (u_4 : GLBinaryRelation α) : Prop :=
  ((u_1 = u_2) ∨ (gl_existence11 u_3 u_1 u_4))

def gl_or3 {α : Type u} (u_1 : GLSet α) (u_2 : GLTernaryRelation α) (u_3 : α) (u_4 : α) (u_5 : α) : Prop :=
  ((gl_preorder u_1 u_2 u_3 u_4) ∨ (u_5 = u_4))

def gl_or5 {α : Type u} (u_1 : GLSet α) (u_2 : GLTernaryRelation α) (u_3 : α) (u_4 : α) : Prop :=
  ((gl_preorder u_1 u_2 u_3 u_4) ∨ (gl_preorder u_1 u_2 u_4 u_3))

def gl_or6 {α : Type u} (u_1 : α) (u_2 : α) (u_3 : α) (u_4 : GLSet α) (u_5 : GLTernaryRelation α) (u_6 : α) : Prop :=
  (((u_1 = u_2) ∨ (u_3 = u_2)) ∨ (gl_preorder u_4 u_5 u_6 u_2))

def gl_or9 {α : Type u} (u_1 : GLSet α) (u_2 : GLTernaryRelation α) (u_3 : α) (u_4 : α) (u_5 : α) : Prop :=
  ((gl_preorder u_1 u_2 u_3 u_4) ∨ (gl_preorder u_1 u_2 u_3 u_5))

def gl_strictOrder {α : Type u} (u_1 : GLSet α) (u_2 : GLTernaryRelation α) (u_3 : α) (u_4 : α) : Prop :=
  ((gl_preorder u_1 u_2 u_3 u_4) ∧ (¬ (u_3 = u_4)))

def gl_fXY {α : Type u} (u_1 : GLBinaryRelation α) (u_2 : GLSet α) (u_3 : GLSet α) : Prop :=
  ((((gl_implication0 u_1 u_2) ∧ (gl_implication1 u_1 u_3)) ∧ (gl_implication4 u_2 u_3 u_1)) ∧ (gl_implication5 u_2 u_1))

def gl_fXYZ {α : Type u} (u_1 : GLTernaryRelation α) (u_2 : GLSet α) (u_3 : GLSet α) (u_4 : GLSet α) : Prop :=
  (((((gl_implication8 u_1 u_2) ∧ (gl_implication9 u_1 u_3)) ∧ (gl_implication10 u_1 u_4)) ∧ (gl_implication13 u_2 u_3 u_4 u_1)) ∧ (gl_implication14 u_2 u_3 u_1))

def gl_implication50 {α : Type u} (u_1 : α) (u_2 : α) (u_3 : GLSet α) (u_4 : GLTernaryRelation α) : Prop :=
  ((¬ (u_1 = u_2)) → (gl_strictOrder u_3 u_4 u_1 u_2))

def gl_or0 {α : Type u} (u_1 : α) (u_2 : α) (u_3 : GLSet α) (u_4 : GLTernaryRelation α) : Prop :=
  ((u_1 = u_2) ∨ (gl_strictOrder u_3 u_4 u_1 u_2))

def gl_or10 {α : Type u} (u_1 : GLSet α) (u_2 : GLTernaryRelation α) (u_3 : α) (u_4 : α) : Prop :=
  (((gl_strictOrder u_1 u_2 u_3 u_4) ∨ (u_3 = u_4)) ∨ (gl_strictOrder u_1 u_2 u_4 u_3))

def gl_NaturalNumbers {α : Type u} (u_1 : GLSet α) (u_2 : α) (u_3 : GLBinaryRelation α) (u_4 : GLTernaryRelation α) (u_5 : GLTernaryRelation α) : Prop :=
  (((((((((((((u_1 u_2) ∧ (gl_fXY u_3 u_1 u_1)) ∧ (gl_implication6 u_1 u_2 u_3)) ∧ (gl_implication7 u_1 u_3)) ∧ (gl_fXYZ u_4 u_1 u_1 u_1)) ∧ (gl_implication15 u_1 u_2 u_4)) ∧ (gl_implication16 u_1 u_2 u_4)) ∧ (gl_implication17 u_1 u_3 u_4)) ∧ (gl_implication18 u_1 u_3 u_4)) ∧ (gl_fXYZ u_5 u_1 u_1 u_1)) ∧ (gl_implication19 u_1 u_2 u_5)) ∧ (gl_implication20 u_1 u_3 u_5 u_4)) ∧ (gl_implication21 u_1 u_3 u_5 u_4))

def gl_AnchorFTA {α : Type u} (u_1 : GLSet α) (u_2 : α) (u_3 : GLBinaryRelation α) (u_4 : GLTernaryRelation α) (u_5 : GLTernaryRelation α) (u_6 : α) (u_7 : α) (u_8 : GLBinaryRelation α) : Prop :=
  ((((gl_NaturalNumbers u_1 u_2 u_3 u_4 u_5) ∧ (u_3 u_2 u_6)) ∧ (u_3 u_6 u_7)) ∧ (gl_identity u_1 u_8))

def gl_AnchorGauss {α : Type u} (u_1 : GLSet α) (u_2 : α) (u_3 : GLBinaryRelation α) (u_4 : GLTernaryRelation α) (u_5 : GLTernaryRelation α) (u_6 : α) (u_7 : α) (u_8 : GLBinaryRelation α) : Prop :=
  ((((gl_NaturalNumbers u_1 u_2 u_3 u_4 u_5) ∧ (u_3 u_2 u_6)) ∧ (u_3 u_6 u_7)) ∧ (gl_identity u_1 u_8))

def gl_AnchorPeano {α : Type u} (u_1 : GLSet α) (u_2 : α) (u_3 : GLBinaryRelation α) (u_4 : GLTernaryRelation α) (u_5 : GLTernaryRelation α) (u_6 : α) : Prop :=
  ((gl_NaturalNumbers u_1 u_2 u_3 u_4 u_5) ∧ (u_3 u_2 u_6))

def gl_implication199 {α : Type u}  : Prop :=
  (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α) (x_7 : α) (x_8 : GLBinaryRelation α), ((gl_AnchorFTA x_1 x_2 x_3 x_4 x_5 x_6 x_7 x_8) → (∀ (x_9 : α) (x_10 : α), ((gl_preorder x_1 x_5 x_9 x_10) → ((gl_preorder x_1 x_5 x_10 x_9) → (x_9 = x_10))))))

def gl_implication211 {α : Type u}  : Prop :=
  (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α) (x_7 : α) (x_8 : GLBinaryRelation α), ((gl_AnchorFTA x_1 x_2 x_3 x_4 x_5 x_6 x_7 x_8) → (∀ (x_9 : α), ((x_1 x_9) → (∀ (x_10 : α), ((x_1 x_10) → (gl_or5 x_1 x_4 x_9 x_10)))))))

def gl_implication56 {α : Type u}  : Prop :=
  (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α) (x_7 : α) (x_8 : GLBinaryRelation α), ((gl_AnchorFTA x_1 x_2 x_3 x_4 x_5 x_6 x_7 x_8) → (∀ (x_9 : α) (x_10 : α), ((gl_preorder x_1 x_4 x_9 x_10) → (gl_or0 x_9 x_10 x_1 x_4)))))

end GLExport
