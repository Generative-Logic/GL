/-
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-/

import GLExport.ProofSupport

namespace GLExport

universe u

def gl_and20 {α : Type u} (u_1 : α) (u_2 : GLSet α) (u_3 : α) (u_4 : GLSet α) : Prop :=
  ((u_2 u_1) ∧ (u_4 u_3))

def gl_and21 {α : Type u} (u_1 : α) (u_2 : GLSet α) (u_3 : α) (u_4 : GLSet α) (u_5 : α) (u_6 : GLSet α) : Prop :=
  (((u_2 u_1) ∧ (u_4 u_3)) ∧ (u_6 u_5))

def gl_and23 {α : Type u} (u_1 : α) (u_2 : α) (u_3 : α) (u_4 : α) : Prop :=
  ((u_1 = u_2) ∧ (u_3 = u_4))

def gl_and24 {α : Type u} (u_1 : α) (u_2 : α) (u_3 : α) (u_4 : GLTernaryRelation α) (u_5 : α) (u_6 : α) (u_7 : α) (u_8 : α) (u_9 : GLTernaryRelation α) : Prop :=
  (((u_4 u_1 u_2 u_3) ∧ (u_4 u_5 u_6 u_7)) ∧ (u_9 u_7 u_8 u_3))

def gl_and5 {α : Type u} (u_1 : α) (u_2 : GLSet α) (u_3 : α) : Prop :=
  ((u_2 u_1) ∧ (u_1 = u_3))

def gl_coprime {α : Type u} (u_1 : GLSet α) (u_2 : α) (u_3 : GLBinaryRelation α) (u_4 : GLTernaryRelation α) (u_5 : α) (u_6 : α) : Prop :=
  (∀ (x_1 : α), ((u_1 x_1) → (∀ (x_2 : α), ((u_4 x_1 x_2 u_5) → (∀ (x_3 : α), ((u_4 x_1 x_3 u_6) → (u_3 u_2 x_1)))))))

def gl_existence0 {α : Type u} (u_1 : GLSet α) (u_2 : α) (u_3 : GLBinaryRelation α) : Prop :=
  (¬ (∀ (x_1 : α), ((u_1 x_1) → (¬ (u_3 u_2 x_1)))))

def gl_existence1 {α : Type u} (u_1 : GLSet α) (u_2 : α) (u_3 : α) (u_4 : GLTernaryRelation α) : Prop :=
  (¬ (∀ (x_1 : α), ((u_1 x_1) → (¬ (u_4 u_2 u_3 x_1)))))

def gl_implication0 {α : Type u} (u_1 : GLBinaryRelation α) (u_2 : GLSet α) : Prop :=
  (∀ (x_1 : α) (x_2 : α), ((u_1 x_1 x_2) → (u_2 x_1)))

def gl_implication1 {α : Type u} (u_1 : GLBinaryRelation α) (u_2 : GLSet α) : Prop :=
  (∀ (x_1 : α) (x_2 : α), ((u_1 x_1 x_2) → (u_2 x_2)))

def gl_implication10 {α : Type u} (u_1 : GLSet α) (u_2 : GLBinaryRelation α) : Prop :=
  (∀ (x_1 : α), ((u_1 x_1) → (∀ (x_2 : α), ((u_2 x_2 x_1) → (∀ (x_3 : α), ((u_2 x_3 x_1) → (x_2 = x_3)))))))

def gl_implication11 {α : Type u} (u_1 : GLTernaryRelation α) (u_2 : GLSet α) : Prop :=
  (∀ (x_1 : α) (x_2 : α) (x_3 : α), ((u_1 x_1 x_2 x_3) → (u_2 x_1)))

def gl_implication12 {α : Type u} (u_1 : GLTernaryRelation α) (u_2 : GLSet α) : Prop :=
  (∀ (x_1 : α) (x_2 : α) (x_3 : α), ((u_1 x_1 x_2 x_3) → (u_2 x_2)))

def gl_implication121 {α : Type u} (u_1 : α) (u_2 : GLBinaryRelation α) (u_3 : GLBinaryRelation α) : Prop :=
  (∀ (x_1 : α), ((u_2 u_1 x_1) → (u_3 u_1 x_1)))

def gl_implication122 {α : Type u} (u_1 : GLSet α) (u_2 : GLBinaryRelation α) (u_3 : GLBinaryRelation α) (u_4 : GLBinaryRelation α) (u_5 : GLTernaryRelation α) : Prop :=
  (∀ (x_1 : α), ((u_1 x_1) → (∀ (x_2 : α), ((u_2 x_1 x_2) → ((u_1 x_2) → (∀ (x_3 : α), ((u_3 x_1 x_3) → (∀ (x_4 : α), ((u_4 x_2 x_4) → (∀ (x_5 : α), ((u_5 x_3 x_4 x_5) → (u_3 x_2 x_5))))))))))))

def gl_implication13 {α : Type u} (u_1 : GLTernaryRelation α) (u_2 : GLSet α) : Prop :=
  (∀ (x_1 : α) (x_2 : α) (x_3 : α), ((u_1 x_1 x_2 x_3) → (u_2 x_3)))

def gl_implication137 {α : Type u} (u_1 : GLBinaryRelation α) (u_2 : GLBinaryRelation α) : Prop :=
  (∀ (x_1 : α) (x_2 : α), ((u_1 x_1 x_2) → (u_2 x_1 x_2)))

def gl_implication165 {α : Type u} (u_1 : α) (u_2 : GLBinaryRelation α) (u_3 : GLSet α) : Prop :=
  (∀ (x_1 : α), ((u_2 x_1 u_1) → (¬ (u_3 x_1))))

def gl_implication1756 {α : Type u} (u_1 : α) (u_2 : α) (u_3 : α) (u_4 : α) : Prop :=
  ((u_1 = u_2) → (u_3 = u_4))

def gl_implication1757 {α : Type u} (u_1 : α) (u_2 : α) (u_3 : α) (u_4 : α) : Prop :=
  ((¬ (u_1 = u_2)) → (¬ (u_3 = u_4)))

def gl_implication19 {α : Type u} (u_1 : GLSet α) (u_2 : GLSet α) (u_3 : GLTernaryRelation α) : Prop :=
  (∀ (x_1 : α), ((u_1 x_1) → (∀ (x_2 : α), ((u_2 x_2) → (∀ (x_3 : α), ((u_3 x_1 x_2 x_3) → (∀ (x_4 : α), ((u_3 x_1 x_2 x_4) → (x_3 = x_4)))))))))

def gl_implication21 {α : Type u} (u_1 : GLSet α) (u_2 : α) (u_3 : GLTernaryRelation α) : Prop :=
  (∀ (x_1 : α), ((u_1 x_1) → (∀ (x_2 : α), ((u_3 x_1 u_2 x_2) → (x_1 = x_2)))))

def gl_implication22 {α : Type u} (u_1 : GLSet α) (u_2 : α) (u_3 : GLTernaryRelation α) : Prop :=
  (∀ (x_1 : α) (x_2 : α), ((x_1 = x_2) → ((u_1 x_1) → ((u_1 x_2) → (u_3 x_1 u_2 x_2)))))

def gl_implication23 {α : Type u} (u_1 : GLSet α) (u_2 : GLBinaryRelation α) (u_3 : GLTernaryRelation α) : Prop :=
  (∀ (x_1 : α), ((u_1 x_1) → (∀ (x_2 : α), ((u_2 x_1 x_2) → (∀ (x_3 : α) (x_4 : α), ((u_3 x_3 x_1 x_4) → (∀ (x_5 : α), ((u_3 x_3 x_2 x_5) → (u_2 x_4 x_5)))))))))

def gl_implication24 {α : Type u} (u_1 : GLSet α) (u_2 : GLBinaryRelation α) (u_3 : GLTernaryRelation α) : Prop :=
  (∀ (x_1 : α), ((u_1 x_1) → (∀ (x_2 : α), ((u_2 x_1 x_2) → (∀ (x_3 : α) (x_4 : α), ((u_3 x_3 x_1 x_4) → (∀ (x_5 : α), ((u_2 x_4 x_5) → (u_3 x_3 x_2 x_5)))))))))

def gl_implication2437 {α : Type u} (u_1 : GLBinaryRelation α) (u_2 : α) : Prop :=
  (∀ (x_1 : α) (x_2 : α), ((u_1 x_1 x_2) → (x_1 = u_2)))

def gl_implication2438 {α : Type u} (u_1 : GLBinaryRelation α) (u_2 : α) : Prop :=
  (∀ (x_1 : α) (x_2 : α), ((u_1 x_1 x_2) → (x_2 = u_2)))

def gl_implication2439 {α : Type u} (u_1 : α) (u_2 : α) (u_3 : GLBinaryRelation α) : Prop :=
  (∀ (x_1 : α), ((x_1 = u_1) → (∀ (x_2 : α), ((x_2 = u_2) → (u_3 x_1 x_2)))))

def gl_implication2441 {α : Type u} (u_1 : GLSet α) (u_2 : α) : Prop :=
  (∀ (x_1 : α), ((u_1 x_1) → (x_1 = u_2)))

def gl_implication2442 {α : Type u} (u_1 : α) (u_2 : GLSet α) : Prop :=
  (∀ (x_1 : α), ((x_1 = u_1) → (u_2 x_1)))

def gl_implication2444 {α : Type u} (u_1 : α) (u_2 : α) (u_3 : GLSet α) : Prop :=
  ((¬ (u_1 = u_2)) → (u_3 u_1))

def gl_implication2483 {α : Type u} (u_1 : α) (u_2 : α) (u_3 : α) : Prop :=
  ((¬ (u_1 = u_2)) → (u_1 = u_3))

def gl_implication25 {α : Type u} (u_1 : GLSet α) (u_2 : α) (u_3 : GLTernaryRelation α) : Prop :=
  (∀ (x_1 : α), ((u_1 x_1) → (∀ (x_2 : α), ((u_3 x_1 u_2 x_2) → (x_2 = u_2)))))

def gl_implication26 {α : Type u} (u_1 : α) (u_2 : GLSet α) (u_3 : GLTernaryRelation α) : Prop :=
  (∀ (x_1 : α), ((x_1 = u_1) → (∀ (x_2 : α), ((u_2 x_2) → ((u_2 x_1) → (u_3 x_2 u_1 x_1))))))

def gl_implication27 {α : Type u} (u_1 : GLSet α) (u_2 : GLBinaryRelation α) (u_3 : GLTernaryRelation α) (u_4 : GLTernaryRelation α) : Prop :=
  (∀ (x_1 : α), ((u_1 x_1) → (∀ (x_2 : α), ((u_2 x_1 x_2) → (∀ (x_3 : α) (x_4 : α), ((u_3 x_3 x_1 x_4) → (∀ (x_5 : α), ((u_4 x_4 x_3 x_5) → (u_3 x_3 x_2 x_5)))))))))

def gl_implication28 {α : Type u} (u_1 : GLSet α) (u_2 : GLBinaryRelation α) (u_3 : GLTernaryRelation α) (u_4 : GLTernaryRelation α) : Prop :=
  (∀ (x_1 : α), ((u_1 x_1) → (∀ (x_2 : α), ((u_2 x_1 x_2) → (∀ (x_3 : α) (x_4 : α), ((u_3 x_3 x_1 x_4) → (∀ (x_5 : α), ((u_3 x_3 x_2 x_5) → (u_4 x_4 x_3 x_5)))))))))

def gl_implication37 {α : Type u} (u_1 : GLBinaryRelation α) : Prop :=
  (∀ (x_1 : α) (x_2 : α), ((u_1 x_1 x_2) → (x_1 = x_2)))

def gl_implication38 {α : Type u} (u_1 : GLSet α) (u_2 : GLBinaryRelation α) : Prop :=
  (∀ (x_1 : α), ((u_1 x_1) → (∀ (x_2 : α), ((x_1 = x_2) → (u_2 x_1 x_2)))))

def gl_implication7 {α : Type u} (u_1 : GLSet α) (u_2 : GLBinaryRelation α) : Prop :=
  (∀ (x_1 : α), ((u_1 x_1) → (∀ (x_2 : α), ((u_2 x_1 x_2) → (∀ (x_3 : α), ((u_2 x_1 x_3) → (x_2 = x_3)))))))

def gl_implication86 {α : Type u} (u_1 : α) (u_2 : α) (u_3 : GLTernaryRelation α) (u_4 : GLSet α) : Prop :=
  (∀ (x_1 : α), ((u_3 u_1 x_1 u_2) → (¬ (u_4 x_1))))

def gl_implication9 {α : Type u} (u_1 : GLSet α) (u_2 : α) (u_3 : GLBinaryRelation α) : Prop :=
  (∀ (x_1 : α), ((u_1 x_1) → (¬ (u_3 x_1 u_2))))

def gl_inSequence {α : Type u} (u_1 : GLSet α) (u_2 : α) (u_3 : GLBinaryRelation α) : Prop :=
  (¬ (∀ (x_1 : α), ((u_1 x_1) → (¬ (u_3 x_1 u_2)))))

def gl_or0 {α : Type u} (u_1 : α) (u_2 : α) (u_3 : α) (u_4 : α) : Prop :=
  ((u_1 = u_2) ∨ (¬ (u_3 = u_4)))

def gl_or4 {α : Type u} (u_1 : α) (u_2 : GLSet α) (u_3 : α) : Prop :=
  ((u_2 u_1) ∨ (u_1 = u_3))

def gl_or7 {α : Type u} (u_1 : α) (u_2 : α) (u_3 : α) : Prop :=
  ((u_1 = u_2) ∨ (u_1 = u_3))

def gl_preorder {α : Type u} (u_1 : GLSet α) (u_2 : GLTernaryRelation α) (u_3 : α) (u_4 : α) : Prop :=
  (¬ (∀ (x_1 : α), ((u_1 x_1) → (¬ (u_2 u_3 x_1 u_4)))))

def gl_subset {α : Type u} (u_1 : GLSet α) (u_2 : GLSet α) : Prop :=
  (∀ (x_1 : α), ((u_1 x_1) → (u_2 x_1)))

def gl_EnumerationRelationSet1 {α : Type u} (u_1 : α) (u_2 : α) (u_3 : GLBinaryRelation α) : Prop :=
  (((gl_implication2437 u_3 u_1) ∧ (gl_implication2438 u_3 u_2)) ∧ (gl_implication2439 u_1 u_2 u_3))

def gl_EnumerationSet1 {α : Type u} (u_1 : α) (u_2 : GLSet α) : Prop :=
  ((gl_implication2441 u_2 u_1) ∧ (gl_implication2442 u_1 u_2))

def gl_and0 {α : Type u} (u_1 : α) (u_2 : GLBinaryRelation α) (u_3 : GLBinaryRelation α) (u_4 : α) (u_5 : α) (u_6 : GLSet α) (u_7 : GLBinaryRelation α) (u_8 : GLTernaryRelation α) : Prop :=
  (((gl_implication121 u_1 u_2 u_3) ∧ (u_3 u_4 u_5)) ∧ (gl_implication122 u_6 u_7 u_3 u_2 u_8))

def gl_and12 {α : Type u} (u_1 : GLSet α) (u_2 : GLTernaryRelation α) (u_3 : α) (u_4 : α) (u_5 : α) : Prop :=
  ((gl_preorder u_1 u_2 u_3 u_4) ∧ (gl_preorder u_1 u_2 u_4 u_5))

def gl_and22 {α : Type u} (u_1 : α) (u_2 : GLSet α) (u_3 : GLSet α) (u_4 : GLTernaryRelation α) (u_5 : α) : Prop :=
  ((u_2 u_1) ∧ (gl_preorder u_3 u_4 u_1 u_5))

def gl_and27 {α : Type u} (u_1 : GLSet α) (u_2 : GLTernaryRelation α) (u_3 : α) (u_4 : α) (u_5 : α) (u_6 : GLTernaryRelation α) : Prop :=
  (((gl_preorder u_1 u_2 u_3 u_4) ∧ (gl_preorder u_1 u_2 u_4 u_5)) ∧ (gl_preorder u_1 u_6 u_4 u_5))

def gl_and3 {α : Type u} (u_1 : GLSet α) (u_2 : α) (u_3 : GLBinaryRelation α) (u_4 : α) : Prop :=
  ((gl_existence0 u_1 u_2 u_3) ∧ (¬ (u_3 u_2 u_4)))

def gl_existence13 {α : Type u} (u_1 : GLSet α) (u_2 : α) (u_3 : α) (u_4 : α) (u_5 : GLTernaryRelation α) (u_6 : α) (u_7 : α) (u_8 : α) (u_9 : GLTernaryRelation α) : Prop :=
  (¬ (∀ (x_1 : α), ((u_1 x_1) → (¬ (gl_and24 u_2 u_3 u_4 u_5 u_6 u_7 x_1 u_8 u_9)))))

def gl_identity {α : Type u} (u_1 : GLSet α) (u_2 : GLBinaryRelation α) : Prop :=
  (((gl_implication0 u_2 u_1) ∧ (gl_implication37 u_2)) ∧ (gl_implication38 u_1 u_2))

def gl_implication136 {α : Type u} (u_1 : GLBinaryRelation α) (u_2 : GLSet α) (u_3 : GLTernaryRelation α) (u_4 : α) : Prop :=
  (∀ (x_1 : α) (x_2 : α), ((u_1 x_1 x_2) → (gl_preorder u_2 u_3 x_1 u_4)))

def gl_implication138 {α : Type u} (u_1 : GLSet α) (u_2 : GLTernaryRelation α) (u_3 : α) (u_4 : GLBinaryRelation α) (u_5 : GLBinaryRelation α) : Prop :=
  (∀ (x_1 : α), ((gl_preorder u_1 u_2 x_1 u_3) → (∀ (x_2 : α), ((u_4 x_1 x_2) → (u_5 x_1 x_2)))))

def gl_implication141 {α : Type u} (u_1 : GLSet α) (u_2 : GLSet α) (u_3 : GLTernaryRelation α) (u_4 : α) (u_5 : GLSet α) : Prop :=
  (∀ (x_1 : α), ((u_1 x_1) → ((gl_preorder u_2 u_3 x_1 u_4) → (u_5 x_1))))

def gl_implication18 {α : Type u} (u_1 : GLSet α) (u_2 : GLSet α) (u_3 : GLSet α) (u_4 : GLTernaryRelation α) : Prop :=
  (∀ (x_1 : α), ((u_1 x_1) → (∀ (x_2 : α), ((u_2 x_2) → (gl_existence1 u_3 x_1 x_2 u_4)))))

def gl_implication1873 {α : Type u} (u_1 : GLSet α) (u_2 : α) (u_3 : GLBinaryRelation α) : Prop :=
  (∀ (x_1 : α), ((u_1 x_1) → ((u_3 x_1 u_2) → (gl_inSequence u_1 u_2 u_3))))

def gl_implication1885 {α : Type u} (u_1 : α) (u_2 : α) (u_3 : GLSet α) (u_4 : GLBinaryRelation α) : Prop :=
  ((¬ (u_1 = u_2)) → (gl_inSequence u_3 u_1 u_4))

def gl_implication1886 {α : Type u} (u_1 : GLSet α) (u_2 : α) (u_3 : GLBinaryRelation α) (u_4 : α) : Prop :=
  ((¬ (gl_inSequence u_1 u_2 u_3)) → (u_2 = u_4))

def gl_implication2446 {α : Type u} (u_1 : GLSet α) (u_2 : GLSet α) (u_3 : α) : Prop :=
  (∀ (x_1 : α), ((u_1 x_1) → (gl_or4 x_1 u_2 u_3)))

def gl_implication2447 {α : Type u} (u_1 : GLSet α) (u_2 : α) (u_3 : GLSet α) : Prop :=
  (∀ (x_1 : α), ((gl_or4 x_1 u_1 u_2) → (u_3 x_1)))

def gl_implication2450 {α : Type u} (u_1 : α) (u_2 : α) (u_3 : α) (u_4 : α) (u_5 : GLBinaryRelation α) : Prop :=
  ((¬ (gl_and23 u_1 u_2 u_3 u_4)) → (u_5 u_1 u_3))

def gl_implication2451 {α : Type u} (u_1 : α) (u_2 : α) (u_3 : GLBinaryRelation α) (u_4 : α) (u_5 : α) : Prop :=
  ((¬ (u_3 u_1 u_2)) → (gl_and23 u_1 u_4 u_2 u_5))

def gl_implication2475 {α : Type u} (u_1 : GLSet α) (u_2 : GLTernaryRelation α) (u_3 : α) (u_4 : α) (u_5 : GLTernaryRelation α) (u_6 : GLSet α) : Prop :=
  (∀ (x_1 : α), ((gl_preorder u_1 u_2 u_3 x_1) → ((gl_preorder u_1 u_2 x_1 u_4) → ((gl_preorder u_1 u_5 x_1 u_4) → (u_6 x_1)))))

def gl_implication2484 {α : Type u} (u_1 : GLSet α) (u_2 : GLTernaryRelation α) (u_3 : α) (u_4 : α) : Prop :=
  (∀ (x_1 : α), ((u_1 x_1) → ((gl_preorder u_1 u_2 x_1 u_3) → (gl_or7 x_1 u_4 u_3))))

def gl_implication2667 {α : Type u} (u_1 : GLSet α) (u_2 : α) (u_3 : GLBinaryRelation α) : Prop :=
  (∀ (x_1 : α), ((u_1 x_1) → ((u_3 x_1 u_2) → (gl_inSequence u_1 u_2 u_3))))

def gl_implication2932 {α : Type u} (u_1 : GLBinaryRelation α) (u_2 : GLBinaryRelation α) : Prop :=
  ((gl_implication137 u_1 u_2) → ((gl_implication137 u_2 u_1) → (u_1 = u_2)))

def gl_implication2933 {α : Type u} (u_1 : GLSet α) (u_2 : GLSet α) : Prop :=
  ((gl_subset u_1 u_2) → ((gl_subset u_2 u_1) → (u_1 = u_2)))

def gl_implication3028 {α : Type u} (u_1 : α) (u_2 : α) (u_3 : GLSet α) (u_4 : GLTernaryRelation α) (u_5 : α) : Prop :=
  ((¬ (u_1 = u_2)) → (gl_preorder u_3 u_4 u_5 u_2))

def gl_implication4 {α : Type u} (u_1 : GLSet α) (u_2 : α) (u_3 : GLBinaryRelation α) : Prop :=
  (∀ (x_1 : α), ((u_1 x_1) → ((u_3 u_2 x_1) → (gl_existence0 u_1 u_2 u_3))))

def gl_implication6 {α : Type u} (u_1 : GLSet α) (u_2 : GLSet α) (u_3 : GLBinaryRelation α) : Prop :=
  (∀ (x_1 : α), ((u_1 x_1) → (gl_existence0 u_2 x_1 u_3)))

def gl_implication87 {α : Type u} (u_1 : GLSet α) (u_2 : α) (u_3 : α) (u_4 : GLTernaryRelation α) : Prop :=
  (∀ (x_1 : α), ((u_1 x_1) → ((u_4 u_2 x_1 u_3) → (gl_preorder u_1 u_4 u_2 u_3))))

def gl_implication89 {α : Type u} (u_1 : GLSet α) (u_2 : GLSet α) (u_3 : GLTernaryRelation α) (u_4 : α) : Prop :=
  (∀ (x_1 : α), ((u_1 x_1) → (gl_preorder u_2 u_3 u_4 x_1)))

def gl_implication90 {α : Type u} (u_1 : GLSet α) (u_2 : GLSet α) (u_3 : GLTernaryRelation α) (u_4 : α) : Prop :=
  (∀ (x_1 : α), ((u_1 x_1) → (gl_preorder u_2 u_3 x_1 u_4)))

def gl_implication91 {α : Type u} (u_1 : GLSet α) (u_2 : GLTernaryRelation α) (u_3 : α) (u_4 : α) (u_5 : GLSet α) : Prop :=
  (∀ (x_1 : α), ((gl_preorder u_1 u_2 u_3 x_1) → ((gl_preorder u_1 u_2 x_1 u_4) → (u_5 x_1))))

def gl_or1 {α : Type u} (u_1 : GLSet α) (u_2 : α) (u_3 : GLBinaryRelation α) (u_4 : α) : Prop :=
  ((gl_inSequence u_1 u_2 u_3) ∨ (u_2 = u_4))

def gl_or10 {α : Type u} (u_1 : α) (u_2 : α) (u_3 : α) (u_4 : GLSet α) (u_5 : GLTernaryRelation α) (u_6 : α) : Prop :=
  (((u_1 = u_2) ∨ (u_3 = u_2)) ∨ (gl_preorder u_4 u_5 u_6 u_2))

def gl_or11 {α : Type u} (u_1 : GLSet α) (u_2 : GLTernaryRelation α) (u_3 : α) (u_4 : α) : Prop :=
  ((gl_preorder u_1 u_2 u_3 u_4) ∨ (gl_preorder u_1 u_2 u_4 u_3))

def gl_or16 {α : Type u} (u_1 : GLSet α) (u_2 : GLTernaryRelation α) (u_3 : α) (u_4 : α) (u_5 : α) : Prop :=
  ((gl_preorder u_1 u_2 u_3 u_4) ∨ (gl_preorder u_1 u_2 u_3 u_5))

def gl_or5 {α : Type u} (u_1 : α) (u_2 : α) (u_3 : GLBinaryRelation α) (u_4 : α) (u_5 : α) : Prop :=
  ((u_3 u_1 u_2) ∨ (gl_and23 u_1 u_4 u_2 u_5))

def gl_or9 {α : Type u} (u_1 : α) (u_2 : α) (u_3 : GLSet α) (u_4 : GLTernaryRelation α) (u_5 : α) : Prop :=
  ((u_1 = u_2) ∨ (gl_preorder u_3 u_4 u_5 u_2))

def gl_ordered {α : Type u} (u_1 : GLSet α) (u_2 : GLTernaryRelation α) (u_3 : GLBinaryRelation α) : Prop :=
  (∀ (x_1 : α) (x_2 : α), ((u_3 x_1 x_2) → (∀ (x_3 : α) (x_4 : α), ((u_3 x_3 x_4) → ((gl_preorder u_1 u_2 x_1 x_3) → (gl_preorder u_1 u_2 x_4 x_2))))))

def gl_strictOrder {α : Type u} (u_1 : GLSet α) (u_2 : GLTernaryRelation α) (u_3 : α) (u_4 : α) : Prop :=
  ((gl_preorder u_1 u_2 u_3 u_4) ∧ (¬ (u_3 = u_4)))

def gl_addToSet {α : Type u} (u_1 : GLSet α) (u_2 : α) (u_3 : GLSet α) : Prop :=
  ((gl_implication2446 u_3 u_1 u_2) ∧ (gl_implication2447 u_1 u_2 u_3))

def gl_and26 {α : Type u} (u_1 : α) (u_2 : α) (u_3 : α) (u_4 : GLTernaryRelation α) (u_5 : α) (u_6 : α) (u_7 : GLTernaryRelation α) (u_8 : GLSet α) : Prop :=
  (((u_4 u_1 u_2 u_3) ∧ (u_7 u_3 u_5 u_6)) ∧ (gl_strictOrder u_8 u_7 u_5 u_2))

def gl_bezEq {α : Type u} (u_1 : GLSet α) (u_2 : GLTernaryRelation α) (u_3 : GLTernaryRelation α) (u_4 : α) (u_5 : α) (u_6 : α) (u_7 : α) (u_8 : α) : Prop :=
  (¬ (∀ (x_1 : α), ((u_1 x_1) → (¬ (gl_existence13 u_1 u_5 u_6 x_1 u_3 u_7 u_8 u_4 u_2)))))

def gl_divisorSet {α : Type u} (u_1 : GLSet α) (u_2 : GLTernaryRelation α) (u_3 : GLTernaryRelation α) (u_4 : α) (u_5 : α) (u_6 : GLSet α) : Prop :=
  ((((gl_implication89 u_6 u_1 u_2 u_4) ∧ (gl_implication90 u_6 u_1 u_2 u_5)) ∧ (gl_implication90 u_6 u_1 u_3 u_5)) ∧ (gl_implication2475 u_1 u_2 u_4 u_5 u_3 u_6))

def gl_existence24 {α : Type u} (u_1 : α) (u_2 : α) (u_3 : GLSet α) (u_4 : α) : Prop :=
  (¬ (∀ (x_1 : GLBinaryRelation α), ((gl_EnumerationRelationSet1 u_1 u_2 x_1) → (¬ (gl_inSequence u_3 u_4 x_1)))))

def gl_existence25 {α : Type u} (u_1 : α) (u_2 : α) (u_3 : GLBinaryRelation α) : Prop :=
  (¬ (∀ (x_1 : GLBinaryRelation α), ((gl_EnumerationRelationSet1 u_1 u_2 x_1) → (¬ (x_1 = u_3)))))

def gl_existence26 {α : Type u} (u_1 : α) (u_2 : α) (u_3 : GLBinaryRelation α) : Prop :=
  (¬ (∀ (x_1 : GLBinaryRelation α), ((gl_EnumerationRelationSet1 u_1 u_2 x_1) → (¬ (u_3 = x_1)))))

def gl_existence27 {α : Type u} (u_1 : α) (u_2 : α) (u_3 : GLSet α) (u_4 : GLTernaryRelation α) : Prop :=
  (¬ (∀ (x_1 : GLBinaryRelation α), ((gl_EnumerationRelationSet1 u_1 u_2 x_1) → (¬ (gl_ordered u_3 u_4 x_1)))))

def gl_fXY {α : Type u} (u_1 : GLBinaryRelation α) (u_2 : GLSet α) (u_3 : GLSet α) : Prop :=
  ((((gl_implication0 u_1 u_2) ∧ (gl_implication1 u_1 u_3)) ∧ (gl_implication6 u_2 u_3 u_1)) ∧ (gl_implication7 u_2 u_1))

def gl_fXYZ {α : Type u} (u_1 : GLTernaryRelation α) (u_2 : GLSet α) (u_3 : GLSet α) (u_4 : GLSet α) : Prop :=
  (((((gl_implication11 u_1 u_2) ∧ (gl_implication12 u_1 u_3)) ∧ (gl_implication13 u_1 u_4)) ∧ (gl_implication18 u_2 u_3 u_4 u_1)) ∧ (gl_implication19 u_2 u_3 u_1))

def gl_implication123 {α : Type u} (u_1 : α) (u_2 : GLBinaryRelation α) (u_3 : GLBinaryRelation α) (u_4 : α) (u_5 : α) (u_6 : GLSet α) (u_7 : GLBinaryRelation α) (u_8 : GLTernaryRelation α) : Prop :=
  ((gl_implication121 u_1 u_2 u_3) → ((u_3 u_4 u_5) → ((gl_implication122 u_6 u_7 u_3 u_2 u_8) → (gl_and0 u_1 u_2 u_3 u_4 u_5 u_6 u_7 u_8))))

def gl_implication2452 {α : Type u} (u_1 : GLBinaryRelation α) (u_2 : GLBinaryRelation α) (u_3 : α) (u_4 : α) : Prop :=
  (∀ (x_1 : α) (x_2 : α), ((u_1 x_1 x_2) → (gl_or5 x_1 x_2 u_2 u_3 u_4)))

def gl_implication2453 {α : Type u} (u_1 : GLBinaryRelation α) (u_2 : α) (u_3 : α) (u_4 : GLBinaryRelation α) : Prop :=
  (∀ (x_1 : α) (x_2 : α), ((gl_or5 x_1 x_2 u_1 u_2 u_3) → (u_4 x_1 x_2)))

def gl_implication261 {α : Type u} (u_1 : GLSet α) (u_2 : GLSet α) (u_3 : GLSet α) (u_4 : GLTernaryRelation α) (u_5 : α) : Prop :=
  (∀ (x_1 : α), ((u_1 x_1) → (gl_and22 x_1 u_2 u_3 u_4 u_5)))

def gl_implication2669 {α : Type u} (u_1 : GLSet α) (u_2 : GLTernaryRelation α) (u_3 : α) (u_4 : α) : Prop :=
  ((gl_preorder u_1 u_2 u_3 u_4) → ((¬ (u_3 = u_4)) → (gl_strictOrder u_1 u_2 u_3 u_4)))

def gl_implication2934 {α : Type u} (u_1 : GLSet α) (u_2 : GLTernaryRelation α) (u_3 : α) (u_4 : α) : Prop :=
  ((¬ (gl_strictOrder u_1 u_2 u_3 u_4)) → (u_3 = u_4))

def gl_implication2935 {α : Type u} (u_1 : α) (u_2 : α) (u_3 : GLSet α) (u_4 : GLTernaryRelation α) : Prop :=
  ((¬ (u_1 = u_2)) → (gl_strictOrder u_3 u_4 u_1 u_2))

def gl_implication3110 {α : Type u} (u_1 : GLSet α) (u_2 : GLTernaryRelation α) (u_3 : α) (u_4 : α) : Prop :=
  ((¬ (gl_strictOrder u_1 u_2 u_3 u_4)) → (u_4 = u_3))

def gl_implication3111 {α : Type u} (u_1 : α) (u_2 : α) (u_3 : GLSet α) (u_4 : GLTernaryRelation α) : Prop :=
  ((¬ (u_1 = u_2)) → (gl_strictOrder u_3 u_4 u_2 u_1))

def gl_interval {α : Type u} (u_1 : GLSet α) (u_2 : GLTernaryRelation α) (u_3 : α) (u_4 : α) (u_5 : GLSet α) : Prop :=
  (((((gl_implication89 u_5 u_1 u_2 u_3) ∧ (gl_implication90 u_5 u_1 u_2 u_4)) ∧ (gl_implication91 u_1 u_2 u_3 u_4 u_5)) ∧ (u_1 u_3)) ∧ (u_1 u_4))

def gl_limitSequence {α : Type u} (u_1 : GLSet α) (u_2 : GLTernaryRelation α) (u_3 : α) (u_4 : GLBinaryRelation α) (u_5 : GLBinaryRelation α) : Prop :=
  (((gl_implication136 u_5 u_1 u_2 u_3) ∧ (gl_implication137 u_5 u_4)) ∧ (gl_implication138 u_1 u_2 u_3 u_4 u_5))

def gl_limitSet {α : Type u} (u_1 : GLSet α) (u_2 : GLTernaryRelation α) (u_3 : GLSet α) (u_4 : α) (u_5 : GLSet α) : Prop :=
  (((gl_subset u_5 u_3) ∧ (gl_implication90 u_5 u_1 u_2 u_4)) ∧ (gl_implication141 u_3 u_1 u_2 u_4 u_5))

def gl_minimum {α : Type u} (u_1 : GLSet α) (u_2 : GLTernaryRelation α) (u_3 : GLSet α) (u_4 : α) : Prop :=
  ((u_3 u_4) ∧ (gl_implication89 u_3 u_1 u_2 u_4))

def gl_or12 {α : Type u} (u_1 : GLSet α) (u_2 : GLTernaryRelation α) (u_3 : α) (u_4 : α) : Prop :=
  (((gl_strictOrder u_1 u_2 u_3 u_4) ∨ (u_3 = u_4)) ∨ (gl_strictOrder u_1 u_2 u_4 u_3))

def gl_or13 {α : Type u} (u_1 : α) (u_2 : α) (u_3 : GLSet α) (u_4 : GLTernaryRelation α) : Prop :=
  ((u_1 = u_2) ∨ (gl_strictOrder u_3 u_4 u_2 u_1))

def gl_or8 {α : Type u} (u_1 : α) (u_2 : α) (u_3 : GLSet α) (u_4 : GLTernaryRelation α) : Prop :=
  ((u_1 = u_2) ∨ (gl_strictOrder u_3 u_4 u_1 u_2))

def gl_prime {α : Type u} (u_1 : GLSet α) (u_2 : GLTernaryRelation α) (u_3 : GLTernaryRelation α) (u_4 : α) (u_5 : α) (u_6 : α) : Prop :=
  ((gl_preorder u_1 u_2 u_5 u_6) ∧ (gl_implication2484 u_1 u_3 u_6 u_4))

def gl_NaturalNumbers {α : Type u} (u_1 : GLSet α) (u_2 : α) (u_3 : GLBinaryRelation α) (u_4 : GLTernaryRelation α) (u_5 : GLTernaryRelation α) : Prop :=
  ((((((((((((((u_1 u_2) ∧ (gl_fXY u_3 u_1 u_1)) ∧ (gl_implication9 u_1 u_2 u_3)) ∧ (gl_implication10 u_1 u_3)) ∧ (gl_fXYZ u_4 u_1 u_1 u_1)) ∧ (gl_implication21 u_1 u_2 u_4)) ∧ (gl_implication22 u_1 u_2 u_4)) ∧ (gl_implication23 u_1 u_3 u_4)) ∧ (gl_implication24 u_1 u_3 u_4)) ∧ (gl_fXYZ u_5 u_1 u_1 u_1)) ∧ (gl_implication25 u_1 u_2 u_5)) ∧ (gl_implication26 u_2 u_1 u_5)) ∧ (gl_implication27 u_1 u_3 u_5 u_4)) ∧ (gl_implication28 u_1 u_3 u_5 u_4))

def gl_appendToSequence {α : Type u} (u_1 : α) (u_2 : α) (u_3 : GLBinaryRelation α) (u_4 : GLBinaryRelation α) : Prop :=
  ((gl_implication2452 u_4 u_3 u_1 u_2) ∧ (gl_implication2453 u_3 u_1 u_2 u_4))

def gl_existence10 {α : Type u} (u_1 : GLSet α) (u_2 : GLTernaryRelation α) (u_3 : GLSet α) (u_4 : α) (u_5 : α) (u_6 : α) : Prop :=
  (¬ (∀ (x_1 : GLSet α), ((gl_limitSet u_1 u_2 u_3 u_4 x_1) → (¬ (¬ (gl_interval u_1 u_2 u_5 u_6 x_1))))))

def gl_existence11 {α : Type u} (u_1 : GLSet α) (u_2 : GLTernaryRelation α) (u_3 : GLSet α) (u_4 : α) (u_5 : α) : Prop :=
  (¬ (∀ (x_1 : GLSet α), ((gl_limitSet u_1 u_2 u_3 u_4 x_1) → (¬ (¬ (gl_interval u_1 u_2 u_5 u_4 x_1))))))

def gl_existence16 {α : Type u} (u_1 : GLSet α) (u_2 : α) (u_3 : α) (u_4 : GLTernaryRelation α) (u_5 : α) (u_6 : α) (u_7 : GLTernaryRelation α) : Prop :=
  (¬ (∀ (x_1 : α), ((u_1 x_1) → (¬ (gl_and26 u_2 u_3 x_1 u_4 u_5 u_6 u_7 u_1)))))

def gl_existence17 {α : Type u} (u_1 : α) (u_2 : α) (u_3 : GLTernaryRelation α) (u_4 : GLSet α) (u_5 : GLTernaryRelation α) (u_6 : α) (u_7 : α) (u_8 : α) (u_9 : α) : Prop :=
  (¬ (∀ (x_1 : α), ((u_3 u_1 u_2 x_1) → (¬ (gl_bezEq u_4 u_3 u_5 u_6 u_7 u_8 u_9 x_1)))))

def gl_existence18 {α : Type u} (u_1 : α) (u_2 : α) (u_3 : GLTernaryRelation α) (u_4 : GLSet α) (u_5 : GLTernaryRelation α) (u_6 : α) (u_7 : α) (u_8 : α) (u_9 : α) : Prop :=
  (¬ (∀ (x_1 : α), ((u_3 u_1 u_2 x_1) → (¬ (gl_bezEq u_4 u_3 u_5 u_6 u_7 x_1 u_8 u_9)))))

def gl_existence2 {α : Type u} (u_1 : GLSet α) (u_2 : GLSet α) (u_3 : α) (u_4 : GLBinaryRelation α) (u_5 : α) (u_6 : α) (u_7 : GLBinaryRelation α) (u_8 : GLTernaryRelation α) : Prop :=
  (¬ (∀ (x_1 : GLBinaryRelation α), ((gl_fXY x_1 u_1 u_2) → (¬ (gl_and0 u_3 u_4 x_1 u_5 u_6 u_1 u_7 u_8)))))

def gl_existence28 {α : Type u} (u_1 : α) (u_2 : GLSet α) (u_3 : GLTernaryRelation α) : Prop :=
  (¬ (∀ (x_1 : GLSet α), ((gl_EnumerationSet1 u_1 x_1) → (¬ (gl_interval u_2 u_3 u_1 u_1 x_1)))))

def gl_existence29 {α : Type u} (u_1 : GLSet α) (u_2 : GLTernaryRelation α) (u_3 : α) (u_4 : GLBinaryRelation α) (u_5 : GLBinaryRelation α) : Prop :=
  (¬ (∀ (x_1 : GLBinaryRelation α), ((gl_limitSequence u_1 u_2 u_3 u_4 x_1) → (¬ (x_1 = u_5)))))

def gl_existence30 {α : Type u} (u_1 : GLSet α) (u_2 : α) (u_3 : GLSet α) (u_4 : GLTernaryRelation α) (u_5 : α) : Prop :=
  (¬ (∀ (x_1 : GLSet α), ((gl_addToSet u_1 u_2 x_1) → (¬ (gl_interval u_3 u_4 u_5 u_2 x_1)))))

def gl_existence8 {α : Type u} (u_1 : GLSet α) (u_2 : GLTernaryRelation α) (u_3 : GLSet α) (u_4 : α) (u_5 : α) : Prop :=
  (¬ (∀ (x_1 : GLSet α), ((gl_limitSet u_1 u_2 u_3 u_4 x_1) → (¬ (gl_interval u_1 u_2 u_5 u_4 x_1)))))

def gl_existence9 {α : Type u} (u_1 : GLSet α) (u_2 : GLTernaryRelation α) (u_3 : GLSet α) (u_4 : α) (u_5 : α) : Prop :=
  (¬ (∀ (x_1 : GLSet α), ((gl_limitSet u_1 u_2 u_3 u_4 x_1) → (¬ (¬ (gl_interval u_1 u_2 u_4 u_5 x_1))))))

def gl_hasMinimum {α : Type u} (u_1 : GLSet α) (u_2 : GLTernaryRelation α) (u_3 : GLSet α) : Prop :=
  (¬ (∀ (x_1 : α), ((u_1 x_1) → (¬ (gl_minimum u_1 u_2 u_3 x_1)))))

def gl_implication2474 {α : Type u} (u_1 : GLSet α) (u_2 : α) (u_3 : GLTernaryRelation α) (u_4 : α) (u_5 : α) (u_6 : GLTernaryRelation α) : Prop :=
  (∀ (x_1 : α), ((u_1 x_1) → (∀ (x_2 : α), ((u_1 x_2) → (∀ (x_3 : α), ((u_3 u_2 x_3 x_1) → (∀ (x_4 : α), ((u_3 u_4 x_4 x_2) → ((u_6 x_2 u_5 x_1) → (gl_bezEq u_1 u_6 u_3 u_5 u_2 x_3 u_4 x_4))))))))))

def gl_implication2477 {α : Type u} (u_1 : α) (u_2 : GLSet α) (u_3 : GLSet α) (u_4 : GLTernaryRelation α) : Prop :=
  ((u_2 u_1) → ((gl_implication89 u_2 u_3 u_4 u_1) → (gl_minimum u_3 u_4 u_2 u_1)))

def gl_implication2479 {α : Type u} (u_1 : GLSet α) (u_2 : GLTernaryRelation α) (u_3 : GLSet α) : Prop :=
  (∀ (x_1 : α), ((gl_minimum u_1 u_2 u_3 x_1) → (¬ (u_1 x_1))))

def gl_implication2485 {α : Type u} (u_1 : GLSet α) (u_2 : GLTernaryRelation α) (u_3 : α) (u_4 : α) (u_5 : GLTernaryRelation α) (u_6 : α) : Prop :=
  ((gl_preorder u_1 u_2 u_3 u_4) → ((gl_implication2484 u_1 u_5 u_4 u_6) → (gl_prime u_1 u_2 u_5 u_6 u_3 u_4)))

def gl_implication3113 {α : Type u} (u_1 : GLSet α) (u_2 : GLTernaryRelation α) (u_3 : α) (u_4 : α) : Prop :=
  ((¬ (gl_strictOrder u_1 u_2 u_3 u_4)) → (gl_or13 u_3 u_4 u_1 u_2))

def gl_implication8 {α : Type u} (u_1 : GLBinaryRelation α) (u_2 : GLSet α) (u_3 : GLSet α) : Prop :=
  ((gl_implication0 u_1 u_2) → ((gl_implication1 u_1 u_3) → ((gl_implication6 u_2 u_3 u_1) → ((gl_implication7 u_2 u_1) → (gl_fXY u_1 u_2 u_3)))))

def gl_implication92 {α : Type u} (u_1 : GLSet α) (u_2 : GLSet α) (u_3 : GLTernaryRelation α) (u_4 : α) (u_5 : α) : Prop :=
  ((gl_implication89 u_1 u_2 u_3 u_4) → ((gl_implication90 u_1 u_2 u_3 u_5) → ((gl_implication91 u_2 u_3 u_4 u_5 u_1) → ((u_2 u_4) → ((u_2 u_5) → (gl_interval u_2 u_3 u_4 u_5 u_1))))))

def gl_or6 {α : Type u} (u_1 : GLSet α) (u_2 : GLTernaryRelation α) (u_3 : GLTernaryRelation α) (u_4 : α) (u_5 : α) (u_6 : α) (u_7 : α) (u_8 : α) : Prop :=
  ((gl_bezEq u_1 u_2 u_3 u_4 u_5 u_6 u_7 u_8) ∨ (gl_bezEq u_1 u_2 u_3 u_4 u_7 u_8 u_5 u_6))

def gl_primeseq {α : Type u} (u_1 : GLSet α) (u_2 : GLTernaryRelation α) (u_3 : GLTernaryRelation α) (u_4 : α) (u_5 : α) (u_6 : GLBinaryRelation α) : Prop :=
  (∀ (x_1 : α) (x_2 : α), ((u_6 x_1 x_2) → (gl_prime u_1 u_2 u_3 u_4 u_5 x_2)))

def gl_sequence {α : Type u} (u_1 : GLSet α) (u_2 : GLTernaryRelation α) (u_3 : α) (u_4 : α) (u_5 : GLBinaryRelation α) : Prop :=
  (¬ (∀ (x_1 : GLSet α), ((gl_interval u_1 u_2 u_3 u_4 x_1) → (¬ (gl_fXY u_5 x_1 u_1)))))

def gl_AnchorFTA {α : Type u} (u_1 : GLSet α) (u_2 : α) (u_3 : GLBinaryRelation α) (u_4 : GLTernaryRelation α) (u_5 : GLTernaryRelation α) (u_6 : α) (u_7 : α) (u_8 : GLBinaryRelation α) : Prop :=
  ((((gl_NaturalNumbers u_1 u_2 u_3 u_4 u_5) ∧ (u_3 u_2 u_6)) ∧ (u_3 u_6 u_7)) ∧ (gl_identity u_1 u_8))

def gl_AnchorGauss {α : Type u} (u_1 : GLSet α) (u_2 : α) (u_3 : GLBinaryRelation α) (u_4 : GLTernaryRelation α) (u_5 : GLTernaryRelation α) (u_6 : α) (u_7 : α) (u_8 : GLBinaryRelation α) : Prop :=
  ((((gl_NaturalNumbers u_1 u_2 u_3 u_4 u_5) ∧ (u_3 u_2 u_6)) ∧ (u_3 u_6 u_7)) ∧ (gl_identity u_1 u_8))

def gl_AnchorPeano {α : Type u} (u_1 : GLSet α) (u_2 : α) (u_3 : GLBinaryRelation α) (u_4 : GLTernaryRelation α) (u_5 : GLTernaryRelation α) (u_6 : α) : Prop :=
  ((gl_NaturalNumbers u_1 u_2 u_3 u_4 u_5) ∧ (u_3 u_2 u_6))

def gl_existence12 {α : Type u} (u_1 : GLSet α) (u_2 : GLTernaryRelation α) (u_3 : α) (u_4 : GLBinaryRelation α) (u_5 : α) : Prop :=
  (¬ (∀ (x_1 : GLBinaryRelation α), ((gl_limitSequence u_1 u_2 u_3 u_4 x_1) → (¬ (gl_sequence u_1 u_2 u_5 u_3 x_1)))))

def gl_existence14 {α : Type u} (u_1 : GLSet α) (u_2 : GLTernaryRelation α) (u_3 : GLTernaryRelation α) (u_4 : α) (u_5 : α) (u_6 : α) (u_7 : α) : Prop :=
  (¬ (∀ (x_1 : α), ((u_1 x_1) → (¬ (gl_or6 u_1 u_2 u_3 u_4 u_5 u_6 u_7 x_1)))))

def gl_existence19 {α : Type u} (u_1 : GLSet α) (u_2 : GLTernaryRelation α) (u_3 : α) (u_4 : α) (u_5 : α) (u_6 : α) (u_7 : GLBinaryRelation α) : Prop :=
  (¬ (∀ (x_1 : GLBinaryRelation α), ((gl_sequence u_1 u_2 u_3 u_4 x_1) → (¬ (gl_appendToSequence u_5 u_6 x_1 u_7)))))

def gl_existence21 {α : Type u} (u_1 : α) (u_2 : α) (u_3 : GLSet α) (u_4 : GLTernaryRelation α) (u_5 : GLTernaryRelation α) (u_6 : α) (u_7 : α) : Prop :=
  (¬ (∀ (x_1 : GLBinaryRelation α), ((gl_EnumerationRelationSet1 u_1 u_2 x_1) → (¬ (gl_primeseq u_3 u_4 u_5 u_6 u_7 x_1)))))

def gl_existence22 {α : Type u} (u_1 : α) (u_2 : α) (u_3 : GLBinaryRelation α) (u_4 : GLBinaryRelation α) : Prop :=
  (¬ (∀ (x_1 : GLBinaryRelation α), ((gl_appendToSequence u_1 u_2 u_3 x_1) → (¬ (x_1 = u_4)))))

def gl_existence23 {α : Type u} (u_1 : α) (u_2 : α) (u_3 : GLBinaryRelation α) (u_4 : GLBinaryRelation α) : Prop :=
  (¬ (∀ (x_1 : GLBinaryRelation α), ((gl_appendToSequence u_1 u_2 u_3 x_1) → (¬ (u_4 = x_1)))))

def gl_existence31 {α : Type u} (u_1 : α) (u_2 : α) (u_3 : GLSet α) (u_4 : GLTernaryRelation α) : Prop :=
  (¬ (∀ (x_1 : GLBinaryRelation α), ((gl_EnumerationRelationSet1 u_1 u_2 x_1) → (¬ (gl_sequence u_3 u_4 u_1 u_1 x_1)))))

def gl_existence32 {α : Type u} (u_1 : α) (u_2 : α) (u_3 : GLBinaryRelation α) (u_4 : GLSet α) (u_5 : GLTernaryRelation α) : Prop :=
  (¬ (∀ (x_1 : GLBinaryRelation α), ((gl_appendToSequence u_1 u_2 u_3 x_1) → (¬ (gl_ordered u_4 u_5 x_1)))))

def gl_existence33 {α : Type u} (u_1 : GLSet α) (u_2 : GLTernaryRelation α) (u_3 : α) (u_4 : GLBinaryRelation α) (u_5 : α) (u_6 : α) : Prop :=
  (¬ (∀ (x_1 : GLBinaryRelation α), ((gl_limitSequence u_1 u_2 u_3 u_4 x_1) → (¬ (gl_appendToSequence u_5 u_6 x_1 u_4)))))

def gl_existence35 {α : Type u} (u_1 : α) (u_2 : α) (u_3 : GLBinaryRelation α) (u_4 : GLSet α) (u_5 : GLTernaryRelation α) (u_6 : α) : Prop :=
  (¬ (∀ (x_1 : GLBinaryRelation α), ((gl_appendToSequence u_1 u_2 u_3 x_1) → (¬ (gl_sequence u_4 u_5 u_6 u_1 x_1)))))

def gl_existence36 {α : Type u} (u_1 : GLSet α) (u_2 : GLTernaryRelation α) (u_3 : GLTernaryRelation α) (u_4 : α) (u_5 : α) : Prop :=
  (¬ (∀ (x_1 : GLSet α), ((gl_divisorSet u_1 u_2 u_3 u_4 u_5 x_1) → (¬ (gl_hasMinimum u_1 u_2 x_1)))))

def gl_fold {α : Type u} (u_1 : GLSet α) (u_2 : GLBinaryRelation α) (u_3 : GLTernaryRelation α) (u_4 : GLTernaryRelation α) (u_5 : GLBinaryRelation α) (u_6 : α) (u_7 : α) (u_8 : α) : Prop :=
  (¬ (∀ (x_1 : GLSet α), ((gl_interval u_1 u_3 u_6 u_7 x_1) → (¬ (gl_existence2 x_1 u_1 u_6 u_5 u_7 u_8 u_2 u_4)))))

def gl_implication125 {α : Type u} (u_1 : GLBinaryRelation α) (u_2 : GLSet α) (u_3 : GLSet α) (u_4 : α) (u_5 : GLBinaryRelation α) (u_6 : α) (u_7 : α) (u_8 : GLBinaryRelation α) (u_9 : GLTernaryRelation α) : Prop :=
  ((gl_fXY u_1 u_2 u_3) → ((gl_and0 u_4 u_5 u_1 u_6 u_7 u_2 u_8 u_9) → (gl_existence2 u_2 u_3 u_4 u_5 u_6 u_7 u_8 u_9)))

def gl_implication2454 {α : Type u} (u_1 : GLBinaryRelation α) (u_2 : GLBinaryRelation α) (u_3 : α) (u_4 : α) : Prop :=
  ((gl_implication2452 u_1 u_2 u_3 u_4) → ((gl_implication2453 u_2 u_3 u_4 u_1) → (gl_appendToSequence u_3 u_4 u_2 u_1)))

def gl_implication2472 {α : Type u} (u_1 : GLSet α) (u_2 : GLTernaryRelation α) (u_3 : GLTernaryRelation α) (u_4 : α) (u_5 : α) (u_6 : α) : Prop :=
  (∀ (x_1 : α) (x_2 : α), ((gl_bezEq u_1 u_2 u_3 u_4 u_5 x_1 u_6 x_2) → (gl_or6 u_1 u_2 u_3 u_4 u_5 x_1 u_6 x_2)))

def gl_implication2473 {α : Type u} (u_1 : GLSet α) (u_2 : GLTernaryRelation α) (u_3 : GLTernaryRelation α) (u_4 : α) (u_5 : α) (u_6 : α) : Prop :=
  (∀ (x_1 : α) (x_2 : α), ((gl_bezEq u_1 u_2 u_3 u_4 u_5 x_1 u_6 x_2) → (gl_or6 u_1 u_2 u_3 u_4 u_6 x_2 u_5 x_1)))

def gl_implication2906 {α : Type u} (u_1 : α) (u_2 : α) (u_3 : GLTernaryRelation α) (u_4 : GLSet α) (u_5 : α) (u_6 : α) (u_7 : GLTernaryRelation α) (u_8 : α) (u_9 : α) : Prop :=
  (∀ (x_1 : α), ((u_3 u_1 u_2 x_1) → (∀ (x_2 : α), ((u_4 x_2) → (∀ (x_3 : α), ((u_4 x_3) → ((u_7 u_5 u_6 x_2) → ((u_7 u_8 x_1 x_3) → ((u_3 x_3 u_9 x_2) → (gl_existence17 u_1 u_2 u_3 u_4 u_7 u_9 u_5 u_6 u_8))))))))))

def gl_implication2911 {α : Type u} (u_1 : α) (u_2 : α) (u_3 : GLTernaryRelation α) (u_4 : GLSet α) (u_5 : α) (u_6 : GLTernaryRelation α) (u_7 : α) (u_8 : α) (u_9 : α) : Prop :=
  (∀ (x_1 : α), ((u_3 u_1 u_2 x_1) → (∀ (x_2 : α), ((u_4 x_2) → (∀ (x_3 : α), ((u_4 x_3) → ((u_6 u_5 x_1 x_2) → ((u_6 u_7 u_8 x_3) → ((u_3 x_3 u_9 x_2) → (gl_existence18 u_1 u_2 u_3 u_4 u_6 u_9 u_5 u_7 u_8))))))))))

def gl_implication3192 {α : Type u} (u_1 : GLSet α) (u_2 : GLTernaryRelation α) (u_3 : α) (u_4 : α) (u_5 : GLSet α) : Prop :=
  ((gl_preorder u_1 u_2 u_3 u_4) → (gl_hasMinimum u_1 u_2 u_5))

def gl_implication95 {α : Type u} (u_1 : GLSet α) (u_2 : GLTernaryRelation α) (u_3 : α) (u_4 : α) (u_5 : GLBinaryRelation α) : Prop :=
  (∀ (x_1 : GLSet α), ((gl_interval u_1 u_2 u_3 u_4 x_1) → ((gl_fXY u_5 x_1 u_1) → (gl_sequence u_1 u_2 u_3 u_4 u_5))))

def gl_implication97 {α : Type u} (u_1 : GLSet α) (u_2 : GLTernaryRelation α) (u_3 : α) (u_4 : α) (u_5 : GLBinaryRelation α) : Prop :=
  (∀ (x_1 : GLSet α), ((gl_interval u_1 u_2 u_3 u_4 x_1) → ((gl_implication0 u_5 x_1) → ((gl_implication1 u_5 u_1) → ((gl_implication6 x_1 u_1 u_5) → ((gl_implication7 x_1 u_5) → (gl_sequence u_1 u_2 u_3 u_4 u_5)))))))

def gl_or15 {α : Type u} (u_1 : GLSet α) (u_2 : GLTernaryRelation α) (u_3 : GLSet α) (u_4 : α) (u_5 : α) : Prop :=
  ((gl_hasMinimum u_1 u_2 u_3) ∨ (¬ (gl_preorder u_1 u_2 u_4 u_5)))

def gl_remainder {α : Type u} (u_1 : GLSet α) (u_2 : GLTernaryRelation α) (u_3 : GLTernaryRelation α) (u_4 : α) (u_5 : α) (u_6 : α) : Prop :=
  (¬ (∀ (x_1 : α), ((u_1 x_1) → (¬ (gl_existence16 u_1 x_1 u_5 u_3 u_6 u_4 u_2)))))

def gl_bez1 {α : Type u} (u_1 : GLSet α) (u_2 : GLTernaryRelation α) (u_3 : GLTernaryRelation α) (u_4 : α) (u_5 : α) (u_6 : α) : Prop :=
  (¬ (∀ (x_1 : α), ((u_1 x_1) → (¬ (gl_existence14 u_1 u_2 u_3 u_4 u_5 x_1 u_6)))))

def gl_existence20 {α : Type u} (u_1 : GLSet α) (u_2 : GLTernaryRelation α) (u_3 : GLTernaryRelation α) (u_4 : α) (u_5 : α) : Prop :=
  (¬ (∀ (x_1 : α), ((u_1 x_1) → (¬ (gl_remainder u_1 u_2 u_3 u_4 u_5 x_1)))))

def gl_existence34 {α : Type u} (u_1 : α) (u_2 : α) (u_3 : GLSet α) (u_4 : GLBinaryRelation α) (u_5 : GLTernaryRelation α) (u_6 : GLTernaryRelation α) : Prop :=
  (¬ (∀ (x_1 : GLBinaryRelation α), ((gl_EnumerationRelationSet1 u_1 u_2 x_1) → (¬ (gl_fold u_3 u_4 u_5 u_6 x_1 u_1 u_1 u_2)))))

def gl_existence37 {α : Type u} (u_1 : α) (u_2 : α) (u_3 : GLBinaryRelation α) (u_4 : GLSet α) (u_5 : GLBinaryRelation α) (u_6 : GLTernaryRelation α) (u_7 : GLTernaryRelation α) (u_8 : α) (u_9 : α) : Prop :=
  (¬ (∀ (x_1 : GLBinaryRelation α), ((gl_appendToSequence u_1 u_2 u_3 x_1) → (¬ (gl_fold u_4 u_5 u_6 u_7 x_1 u_8 u_1 u_9)))))

def gl_implication129 {α : Type u} (u_1 : GLSet α) (u_2 : GLTernaryRelation α) (u_3 : α) (u_4 : α) (u_5 : GLSet α) (u_6 : GLBinaryRelation α) (u_7 : α) (u_8 : GLBinaryRelation α) (u_9 : GLTernaryRelation α) : Prop :=
  ((gl_interval u_1 u_2 u_3 u_4 u_5) → ((gl_existence2 u_5 u_1 u_3 u_6 u_4 u_7 u_8 u_9) → (gl_fold u_1 u_8 u_2 u_9 u_6 u_3 u_4 u_7)))

def gl_implication130 {α : Type u} (u_1 : GLSet α) (u_2 : GLTernaryRelation α) (u_3 : α) (u_4 : α) (u_5 : GLBinaryRelation α) (u_6 : α) (u_7 : GLBinaryRelation α) (u_8 : GLTernaryRelation α) : Prop :=
  (∀ (x_1 : GLSet α), ((gl_interval u_1 u_2 u_3 u_4 x_1) → (∀ (x_2 : GLBinaryRelation α), ((gl_fXY x_2 x_1 u_1) → ((gl_implication121 u_3 u_5 x_2) → ((x_2 u_4 u_6) → ((gl_implication122 x_1 u_7 x_2 u_5 u_8) → (gl_fold u_1 u_7 u_2 u_8 u_5 u_3 u_4 u_6))))))))

def gl_implication1954 {α : Type u}  : Prop :=
  (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (RelationalInduction x_1 x_2 x_3 → StrongRelationalInduction x_1 x_2 x_3 x_4 → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (∀ (x_10 : α), ((x_3 x_10 x_7) → (∀ (x_11 : α), ((x_3 x_8 x_11) → (x_4 x_11 x_10 x_9))))))))))

def gl_implication1965 {α : Type u}  : Prop :=
  (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (RelationalInduction x_1 x_2 x_3 → StrongRelationalInduction x_1 x_2 x_3 x_4 → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_4 x_7 x_8 x_9) → (∀ (x_10 : α) (x_11 : α), ((x_4 x_10 x_11 x_7) → (∀ (x_12 : α), ((x_4 x_8 x_11 x_12) → (x_4 x_12 x_10 x_9))))))))))

def gl_implication1998 {α : Type u}  : Prop :=
  (∀ (x_1 : GLSet α) (x_2 : α) (x_3 : GLBinaryRelation α) (x_4 : GLTernaryRelation α) (x_5 : GLTernaryRelation α) (x_6 : α), ((gl_AnchorPeano x_1 x_2 x_3 x_4 x_5 x_6) → (RelationalInduction x_1 x_2 x_3 → StrongRelationalInduction x_1 x_2 x_3 x_4 → (∀ (x_7 : α) (x_8 : α) (x_9 : α), ((x_5 x_7 x_8 x_9) → (∀ (x_10 : α) (x_11 : α), ((x_5 x_10 x_11 x_7) → (∀ (x_12 : α), ((x_5 x_8 x_11 x_12) → (x_5 x_12 x_10 x_9))))))))))

def gl_implication2916 {α : Type u} (u_1 : GLSet α) (u_2 : GLTernaryRelation α) (u_3 : α) (u_4 : α) (u_5 : GLBinaryRelation α) (u_6 : α) (u_7 : α) : Prop :=
  (∀ (x_1 : GLBinaryRelation α), ((gl_sequence u_1 u_2 u_3 u_4 x_1) → ((gl_implication2452 u_5 x_1 u_6 u_7) → ((gl_implication2453 x_1 u_6 u_7 u_5) → (gl_existence19 u_1 u_2 u_3 u_4 u_6 u_7 u_5)))))

def gl_implication40 {α : Type u} (u_1 : GLSet α) (u_2 : α) (u_3 : GLBinaryRelation α) (u_4 : GLTernaryRelation α) (u_5 : GLTernaryRelation α) (u_6 : α) (u_7 : α) (u_8 : GLBinaryRelation α) : Prop :=
  ((gl_NaturalNumbers u_1 u_2 u_3 u_4 u_5) → ((u_3 u_2 u_6) → ((u_3 u_6 u_7) → ((gl_identity u_1 u_8) → (gl_AnchorGauss u_1 u_2 u_3 u_4 u_5 u_6 u_7 u_8)))))

def gl_implication76 {α : Type u} (u_1 : GLSet α) (u_2 : α) (u_3 : GLBinaryRelation α) (u_4 : GLTernaryRelation α) (u_5 : GLTernaryRelation α) (u_6 : α) : Prop :=
  ((gl_NaturalNumbers u_1 u_2 u_3 u_4 u_5) → ((u_3 u_2 u_6) → (gl_AnchorPeano u_1 u_2 u_3 u_4 u_5 u_6)))

def gl_primeFactorSeq {α : Type u} (u_1 : GLSet α) (u_2 : GLBinaryRelation α) (u_3 : GLTernaryRelation α) (u_4 : GLTernaryRelation α) (u_5 : α) (u_6 : α) (u_7 : α) (u_8 : α) (u_9 : GLBinaryRelation α) (u_10 : α) : Prop :=
  ((((gl_sequence u_1 u_3 u_5 u_8 u_9) ∧ (gl_primeseq u_1 u_3 u_4 u_6 u_7 u_9)) ∧ (gl_ordered u_1 u_3 u_9)) ∧ (gl_fold u_1 u_2 u_3 u_4 u_9 u_5 u_8 u_10))

def gl_existence15 {α : Type u} (u_1 : GLSet α) (u_2 : GLTernaryRelation α) (u_3 : α) (u_4 : α) (u_5 : GLBinaryRelation α) (u_6 : GLTernaryRelation α) (u_7 : α) (u_8 : α) (u_9 : α) : Prop :=
  (¬ (∀ (x_1 : GLBinaryRelation α), ((gl_sequence u_1 u_2 u_3 u_4 x_1) → (¬ (gl_primeFactorSeq u_1 u_5 u_2 u_6 u_3 u_7 u_8 u_4 x_1 u_9)))))

def gl_implication2471 {α : Type u} (u_1 : GLSet α) (u_2 : GLTernaryRelation α) (u_3 : GLTernaryRelation α) (u_4 : α) (u_5 : α) (u_6 : α) : Prop :=
  (∀ (x_1 : α), ((u_1 x_1) → (∀ (x_2 : α), ((u_1 x_2) → ((gl_or6 u_1 u_2 u_3 u_4 u_5 x_1 u_6 x_2) → (gl_bez1 u_1 u_2 u_3 u_4 u_5 u_6))))))

def gl_implication2487 {α : Type u} (u_1 : GLSet α) (u_2 : GLTernaryRelation α) (u_3 : α) (u_4 : α) (u_5 : GLBinaryRelation α) (u_6 : GLTernaryRelation α) (u_7 : α) (u_8 : α) (u_9 : GLBinaryRelation α) (u_10 : α) : Prop :=
  ((gl_sequence u_1 u_2 u_3 u_4 u_5) → ((gl_primeseq u_1 u_2 u_6 u_7 u_8 u_5) → ((gl_ordered u_1 u_2 u_5) → ((gl_fold u_1 u_9 u_2 u_6 u_5 u_3 u_4 u_10) → (gl_primeFactorSeq u_1 u_9 u_2 u_6 u_3 u_7 u_8 u_4 u_5 u_10)))))

def gl_implication2929 {α : Type u} (u_1 : GLSet α) (u_2 : α) (u_3 : GLTernaryRelation α) (u_4 : α) (u_5 : GLTernaryRelation α) : Prop :=
  (∀ (x_1 : α), ((u_1 x_1) → (∀ (x_2 : α), ((u_1 x_2) → (∀ (x_3 : α), ((u_1 x_3) → ((u_3 x_2 u_2 x_3) → ((u_5 x_3 x_1 u_4) → ((gl_strictOrder u_1 u_5 x_1 u_2) → (gl_existence20 u_1 u_5 u_3 u_4 u_2))))))))))

def gl_hasPrimeFactorization {α : Type u} (u_1 : GLSet α) (u_2 : GLBinaryRelation α) (u_3 : GLTernaryRelation α) (u_4 : GLTernaryRelation α) (u_5 : α) (u_6 : α) (u_7 : α) (u_8 : α) : Prop :=
  (¬ (∀ (x_1 : α), ((u_1 x_1) → (¬ (gl_existence15 u_1 u_3 u_5 x_1 u_2 u_4 u_6 u_7 u_8)))))

def gl_implication2593 {α : Type u} (u_1 : GLSet α) (u_2 : GLTernaryRelation α) (u_3 : α) (u_4 : GLTernaryRelation α) (u_5 : α) (u_6 : α) (u_7 : GLBinaryRelation α) (u_8 : α) : Prop :=
  (∀ (x_1 : α), ((u_1 x_1) → (∀ (x_2 : GLBinaryRelation α), ((gl_sequence u_1 u_2 u_3 x_1 x_2) → ((gl_primeseq u_1 u_2 u_4 u_5 u_6 x_2) → ((gl_ordered u_1 u_2 x_2) → ((gl_fold u_1 u_7 u_2 u_4 x_2 u_3 x_1 u_8) → (gl_hasPrimeFactorization u_1 u_7 u_2 u_4 u_3 u_5 u_6 u_8))))))))

end GLExport
