/-
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-/

import GLExport.Generated.Definitions

set_option linter.unusedSimpArgs false

namespace GLExport.NatModel

def domain : GLSet Nat := fun _ => True
def successor : GLBinaryRelation Nat := fun x y => y = Nat.succ x
def addition : GLTernaryRelation Nat := fun x y z => z = x + y
def multiplication : GLTernaryRelation Nat := fun x y z => z = x * y
def identity : GLBinaryRelation Nat := fun x y => y = x

theorem induction : RelationalInduction domain 0 successor := by
  intro P k base step _
  exact Nat.rec base (fun n hypothesis => step n True.intro hypothesis (Nat.succ n) rfl) k

theorem strongInduction : StrongRelationalInduction domain 0 successor addition := by
  intro P k base step _
  refine Nat.strongRecOn (motive := P) k ?_
  intro k hypothesis
  cases k with
  | zero => exact base
  | succ n =>
    refine step n True.intro (Nat.succ n) rfl ?_
    intro j x sum different
    apply hypothesis
    unfold addition at sum
    omega

theorem anchorPeano : gl_AnchorPeano domain 0 successor addition multiplication 1 := by
  classical
  simp [gl_AnchorFTA, gl_AnchorGauss, gl_AnchorPeano, gl_NaturalNumbers, gl_existence0, gl_existence1, gl_fXY, gl_fXYZ, gl_identity, gl_implication0, gl_implication1, gl_implication10, gl_implication11, gl_implication12, gl_implication13, gl_implication18, gl_implication19, gl_implication21, gl_implication22, gl_implication23, gl_implication24, gl_implication25, gl_implication26, gl_implication27, gl_implication28, gl_implication37, gl_implication38, gl_implication6, gl_implication7, gl_implication9, domain, successor, addition, multiplication, identity,
    Nat.add_succ, Nat.mul_succ, -Nat.succ_eq_add_one]
  intro z x left y right
  exact Nat.succ.inj (left.symm.trans right)

theorem anchorGauss : gl_AnchorGauss domain 0 successor addition multiplication 1 2 identity := by
  classical
  simp [gl_AnchorFTA, gl_AnchorGauss, gl_AnchorPeano, gl_NaturalNumbers, gl_existence0, gl_existence1, gl_fXY, gl_fXYZ, gl_identity, gl_implication0, gl_implication1, gl_implication10, gl_implication11, gl_implication12, gl_implication13, gl_implication18, gl_implication19, gl_implication21, gl_implication22, gl_implication23, gl_implication24, gl_implication25, gl_implication26, gl_implication27, gl_implication28, gl_implication37, gl_implication38, gl_implication6, gl_implication7, gl_implication9, domain, successor, addition, multiplication, identity,
    Nat.add_succ, Nat.mul_succ, -Nat.succ_eq_add_one]
  intro z x left y right
  exact Nat.succ.inj (left.symm.trans right)

theorem anchorFTA : gl_AnchorFTA domain 0 successor addition multiplication 1 2 identity := by
  classical
  simp [gl_AnchorFTA, gl_AnchorGauss, gl_AnchorPeano, gl_NaturalNumbers, gl_existence0, gl_existence1, gl_fXY, gl_fXYZ, gl_identity, gl_implication0, gl_implication1, gl_implication10, gl_implication11, gl_implication12, gl_implication13, gl_implication18, gl_implication19, gl_implication21, gl_implication22, gl_implication23, gl_implication24, gl_implication25, gl_implication26, gl_implication27, gl_implication28, gl_implication37, gl_implication38, gl_implication6, gl_implication7, gl_implication9, domain, successor, addition, multiplication, identity,
    Nat.add_succ, Nat.mul_succ, -Nat.succ_eq_add_one]
  intro z x left y right
  exact Nat.succ.inj (left.symm.trans right)

end GLExport.NatModel
