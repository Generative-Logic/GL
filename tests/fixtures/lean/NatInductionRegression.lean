/-
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-/

import GLExport.NatModel

open GLExport

example : ¬ RelationalInduction NatModel.domain 0
    (fun n m => m = n + 2) := by
  intro doubleInduction
  have excludesOne := doubleInduction (fun n => n ≠ 1) 1 (by decide)
    (by intro n _ _ m successor; omega) True.intro
  exact excludesOne rfl

#print axioms GLExport.NatModel.induction
#print axioms GLExport.NatModel.anchorPeano
#print axioms GLExport.NatModel.anchorGauss
#print axioms GLExport.NatModel.anchorFTA
