/-
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-/

import GLExport.Generated.FTA
import GLExport.NatModel

namespace GLExport.NatResults

theorem peano_source_000 : (∀ (v1 : Nat) (v2 : Nat), ((¬ (v1 = v2)) → ((NatModel.addition v2 v1 0) → (¬ (0 = v1))))) := by
  exact GLExport.peano_source_000 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_001 : (∀ (v1 : Nat), ((1 = v1) → (gl_inSequence NatModel.domain v1 NatModel.successor))) := by
  exact GLExport.peano_source_001 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_002 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.successor v1 v2) → (∀ (v3 : Nat) (v4 : Nat), ((NatModel.successor v3 v4) → ((v1 = v3) → (v2 = v4)))))) := by
  exact GLExport.peano_source_002 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_003 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.successor v1 v2) → (∀ (v3 : Nat) (v4 : Nat), ((NatModel.successor v3 v4) → ((¬ (v2 = v4)) → (¬ (v1 = v3))))))) := by
  exact GLExport.peano_source_003 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_006 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.successor v1 v2) → (∀ (v3 : Nat), ((NatModel.successor v3 v2) → (v1 = v3))))) := by
  exact GLExport.peano_source_006 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_008 : (∀ (v1 : Nat) (v2 : Nat) (v3 : Nat), ((NatModel.multiplication v1 v2 v3) → (∀ (v4 : Nat) (v5 : Nat), ((NatModel.multiplication v1 v4 v5) → ((NatModel.successor v4 v2) → (NatModel.addition v5 v1 v3)))))) := by
  exact GLExport.peano_source_008 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_009 : (∀ (v1 : Nat) (v2 : Nat), ((v1 = v2) → ((NatModel.multiplication v1 0 v2) → (0 = v1)))) := by
  exact GLExport.peano_source_009 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_010 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.successor v1 v2) → (NatModel.addition 1 v1 v2))) := by
  exact GLExport.peano_source_010 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_011 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.addition v1 0 v2) → ((NatModel.domain v1) → (v1 = v2)))) := by
  exact GLExport.peano_source_011 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_017 : (∀ (v1 : Nat) (v2 : Nat) (v3 : Nat), ((NatModel.addition v1 v2 v3) → (∀ (v4 : Nat) (v5 : Nat), ((NatModel.addition v1 v4 v5) → ((NatModel.successor v4 v2) → (NatModel.successor v5 v3)))))) := by
  exact GLExport.peano_source_017 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_018 : (∀ (v1 : Nat) (v2 : Nat) (v3 : Nat), ((NatModel.addition v1 v2 v3) → (∀ (v4 : Nat), ((NatModel.successor v2 v4) → (∀ (v5 : Nat), ((NatModel.successor v3 v5) → (NatModel.addition v1 v4 v5))))))) := by
  exact GLExport.peano_source_018 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_024 : (∀ (v1 : Nat) (v2 : Nat) (v3 : Nat), ((NatModel.addition v1 v2 v3) → (∀ (v4 : Nat) (v5 : Nat), ((NatModel.successor v4 v5) → ((NatModel.multiplication v2 v4 v1) → (NatModel.multiplication v2 v5 v3)))))) := by
  exact GLExport.peano_source_024 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_025 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.multiplication v1 0 v2) → ((NatModel.domain v1) → (0 = v2)))) := by
  exact GLExport.peano_source_025 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_029 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.addition v1 v2 0) → ((NatModel.domain v1) → (0 = v2)))) := by
  exact GLExport.peano_source_029 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_044 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.successor v1 v2) → ((NatModel.successor 0 v2) → (0 = v1)))) := by
  exact GLExport.peano_source_044 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_045 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.addition v1 v2 0) → ((NatModel.domain v1) → (v1 = v2)))) := by
  exact GLExport.peano_source_045 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_062 : (∀ (v1 : Nat), ((NatModel.domain v1) → ((¬ (gl_inSequence NatModel.domain v1 NatModel.successor)) → (v1 = 0)))) := by
  exact GLExport.peano_source_062 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_065 : (gl_inSequence NatModel.domain 1 NatModel.successor) := by
  exact GLExport.peano_source_065 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_004 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.successor v1 v2) → (∀ (v3 : Nat) (v4 : Nat), ((NatModel.successor v3 v4) → (gl_or0 v2 v4 v1 v3))))) := by
  exact GLExport.peano_source_004 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_005 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.successor v1 v2) → (∀ (v3 : Nat) (v4 : Nat), ((NatModel.successor v3 v4) → (gl_or0 v2 v4 v1 v3))))) := by
  exact GLExport.peano_source_005 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_007 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.successor v1 v2) → (∀ (v3 : Nat) (v4 : Nat), ((NatModel.successor v3 v4) → (∀ (v5 : Nat), ((NatModel.addition v5 v2 v4) → (NatModel.addition v5 v1 v3))))))) := by
  exact GLExport.peano_source_007 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_012 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.addition 1 v1 v2) → (NatModel.addition v1 1 v2))) := by
  exact GLExport.peano_source_012 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_013 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.addition v1 v2 0) → ((NatModel.domain v2) → (0 = v1)))) := by
  exact GLExport.peano_source_013 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_014 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.addition v1 v2 0) → (NatModel.addition v2 v1 0))) := by
  exact GLExport.peano_source_014 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_015 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.addition v1 1 v2) → (NatModel.successor v1 v2))) := by
  exact GLExport.peano_source_015 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_019 : (∀ (v1 : Nat) (v2 : Nat) (v3 : Nat), ((NatModel.addition v1 v2 v3) → (∀ (v4 : Nat) (v5 : Nat), ((NatModel.addition v1 v4 v5) → (∀ (v6 : Nat), ((NatModel.addition v2 v6 v4) → (NatModel.addition v3 v6 v5))))))) := by
  exact GLExport.peano_source_019 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_022 : (∀ (v1 : Nat) (v2 : Nat) (v3 : Nat), ((NatModel.addition v1 v2 v3) → (∀ (v4 : Nat) (v5 : Nat), ((NatModel.addition v4 v5 v1) → (∀ (v6 : Nat), ((NatModel.addition v5 v2 v6) → (NatModel.addition v4 v6 v3))))))) := by
  exact GLExport.peano_source_022 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_027 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.successor v1 v2) → (NatModel.addition v1 1 v2))) := by
  exact GLExport.peano_source_027 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_030 : (∀ (v1 : Nat) (v2 : Nat), ((v1 = v2) → ((NatModel.multiplication v2 v1 0) → (0 = v2)))) := by
  exact GLExport.peano_source_030 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_031 : (∀ (v1 : Nat) (v2 : Nat) (v3 : Nat), ((NatModel.addition v1 v2 v3) → (∀ (v4 : Nat), ((NatModel.addition v4 v2 v3) → (v1 = v4))))) := by
  exact GLExport.peano_source_031 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_032 : (∀ (v1 : Nat), ((NatModel.domain v1) → (∀ (v2 : Nat), ((NatModel.addition 0 v1 v2) → (v1 = v2))))) := by
  exact GLExport.peano_source_032 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_041 : (∀ (v1 : Nat) (v2 : Nat) (v3 : Nat), ((NatModel.addition v1 v2 v3) → (∀ (v4 : Nat), ((NatModel.successor v4 v1) → (∀ (v5 : Nat), ((NatModel.successor v2 v5) → (NatModel.addition v4 v5 v3))))))) := by
  exact GLExport.peano_source_041 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_048 : (∀ (v1 : Nat), ((NatModel.domain v1) → (∀ (v2 : Nat), ((NatModel.multiplication 0 v1 v2) → (0 = v2))))) := by
  exact GLExport.peano_source_048 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_058 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.multiplication v1 0 v2) → (NatModel.multiplication 0 v1 v2))) := by
  exact GLExport.peano_source_058 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_063 : (∀ (v1 : Nat), ((NatModel.domain v1) → (gl_or1 NatModel.domain v1 NatModel.successor 0))) := by
  exact GLExport.peano_source_063 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_064 : (∀ (v1 : Nat), ((NatModel.domain v1) → (gl_or1 NatModel.domain v1 NatModel.successor 0))) := by
  exact GLExport.peano_source_064 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_016 : (∀ (v1 : Nat), ((NatModel.domain v1) → (∀ (v2 : Nat), ((NatModel.multiplication 1 v2 v1) → (v1 = v2))))) := by
  exact GLExport.peano_source_016 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_020 : (∀ (v1 : Nat) (v2 : Nat) (v3 : Nat), ((NatModel.addition v1 v2 v3) → (∀ (v4 : Nat), ((NatModel.successor v1 v4) → (∀ (v5 : Nat), ((NatModel.successor v5 v2) → (NatModel.addition v4 v5 v3))))))) := by
  exact GLExport.peano_source_020 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_026 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.multiplication v1 1 v2) → (NatModel.multiplication 1 v1 v2))) := by
  exact GLExport.peano_source_026 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_035 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.addition v1 0 v2) → (NatModel.addition 0 v1 v2))) := by
  exact GLExport.peano_source_035 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_040 : (∀ (v1 : Nat) (v2 : Nat) (v3 : Nat), ((NatModel.addition v1 v2 v3) → (∀ (v4 : Nat), ((NatModel.successor v1 v4) → (∀ (v5 : Nat), ((NatModel.successor v3 v5) → (NatModel.addition v4 v2 v5))))))) := by
  exact GLExport.peano_source_040 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_046 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.addition v1 v2 1) → (NatModel.addition v2 v1 1))) := by
  exact GLExport.peano_source_046 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_047 : (∀ (v1 : Nat) (v2 : Nat) (v3 : Nat), ((NatModel.multiplication v1 v2 v3) → (∀ (v4 : Nat) (v5 : Nat), ((NatModel.multiplication v1 v4 v5) → (∀ (v6 : Nat) (v7 : Nat), ((NatModel.multiplication v1 v6 v7) → ((NatModel.addition v2 v4 v6) → (NatModel.addition v3 v5 v7)))))))) := by
  exact GLExport.peano_source_047 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_049 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.multiplication v1 v2 0) → (NatModel.multiplication v2 v1 0))) := by
  exact GLExport.peano_source_049 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_060 : (∀ (v1 : Nat), ((NatModel.domain v1) → (∀ (v2 : Nat), ((NatModel.multiplication 1 v1 v2) → (v1 = v2))))) := by
  exact GLExport.peano_source_060 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_021 : (∀ (v1 : Nat) (v2 : Nat) (v3 : Nat), ((NatModel.addition v1 v2 v3) → (∀ (v4 : Nat) (v5 : Nat), ((NatModel.addition v4 v5 v1) → (∀ (v6 : Nat), ((NatModel.addition v4 v2 v6) → (NatModel.addition v6 v5 v3))))))) := by
  exact GLExport.peano_source_021 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_028 : (∀ (v1 : Nat) (v2 : Nat) (v3 : Nat), ((NatModel.multiplication v1 v2 v3) → (∀ (v4 : Nat) (v5 : Nat), ((NatModel.multiplication v4 v2 v5) → ((NatModel.successor v4 v1) → (NatModel.addition v2 v5 v3)))))) := by
  exact GLExport.peano_source_028 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_033 : (∀ (v1 : Nat) (v2 : Nat) (v3 : Nat), ((NatModel.addition v1 v2 v3) → (∀ (v4 : Nat), ((NatModel.successor v2 v4) → (∀ (v5 : Nat), ((NatModel.successor v5 v1) → (NatModel.addition v4 v5 v3))))))) := by
  exact GLExport.peano_source_033 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_034 : (∀ (v1 : Nat) (v2 : Nat) (v3 : Nat), ((NatModel.addition v1 v2 v3) → (∀ (v4 : Nat), ((NatModel.successor v4 v1) → (∀ (v5 : Nat), ((NatModel.successor v2 v5) → (NatModel.addition v5 v4 v3))))))) := by
  exact GLExport.peano_source_034 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_036 : (∀ (v1 : Nat) (v2 : Nat) (v3 : Nat), ((NatModel.addition v1 v2 v3) → (NatModel.addition v2 v1 v3))) := by
  exact GLExport.peano_source_036 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_038 : (∀ (v1 : Nat) (v2 : Nat) (v3 : Nat), ((NatModel.addition v1 v2 v3) → ((v2 = v3) → (0 = v1)))) := by
  exact GLExport.peano_source_038 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_050 : (∀ (v1 : Nat) (v2 : Nat) (v3 : Nat), ((NatModel.addition v1 v2 v3) → (∀ (v4 : Nat) (v5 : Nat) (v6 : Nat), ((NatModel.addition v4 v5 v6) → (∀ (v7 : Nat), ((NatModel.multiplication v7 v4 v1) → ((NatModel.multiplication v7 v5 v2) → (NatModel.multiplication v7 v6 v3)))))))) := by
  exact GLExport.peano_source_050 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_061 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.multiplication 1 v1 v2) → (NatModel.multiplication v1 1 v2))) := by
  exact GLExport.peano_source_061 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_023 : (∀ (v1 : Nat) (v2 : Nat) (v3 : Nat), ((NatModel.addition v1 v2 v3) → (∀ (v4 : Nat) (v5 : Nat) (v6 : Nat), ((NatModel.addition v4 v5 v6) → ((NatModel.addition v2 v1 v5) → (NatModel.addition v4 v3 v6)))))) := by
  exact GLExport.peano_source_023 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_037 : (∀ (v1 : Nat) (v2 : Nat) (v3 : Nat), ((NatModel.addition v1 v2 v3) → (∀ (v4 : Nat), ((NatModel.addition v1 v4 v3) → (v2 = v4))))) := by
  exact GLExport.peano_source_037 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_039 : (∀ (v1 : Nat) (v2 : Nat), ((gl_preorder NatModel.domain NatModel.addition v1 v2) → ((gl_preorder NatModel.domain NatModel.addition v2 v1) → (v1 = v2)))) := by
  exact GLExport.peano_source_039 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_042 : (∀ (v1 : Nat) (v2 : Nat) (v3 : Nat), ((NatModel.addition v1 v2 v3) → (∀ (v4 : Nat) (v5 : Nat), ((NatModel.addition v4 v5 v1) → (∀ (v6 : Nat), ((NatModel.addition v2 v5 v6) → (NatModel.addition v6 v4 v3))))))) := by
  exact GLExport.peano_source_042 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_043 : (∀ (v1 : Nat) (v2 : Nat) (v3 : Nat), ((NatModel.addition v1 v2 v3) → (∀ (v4 : Nat) (v5 : Nat), ((NatModel.addition v2 v4 v5) → (∀ (v6 : Nat), ((NatModel.addition v6 v4 v1) → (NatModel.addition v5 v6 v3))))))) := by
  exact GLExport.peano_source_043 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_051 : (∀ (v1 : Nat) (v2 : Nat) (v3 : Nat), ((NatModel.multiplication v1 v2 v3) → (∀ (v4 : Nat) (v5 : Nat) (v6 : Nat), ((NatModel.multiplication v4 v5 v6) → ((NatModel.multiplication v2 v1 v5) → (NatModel.multiplication v4 v3 v6)))))) := by
  exact GLExport.peano_source_051 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_052 : (∀ (v1 : Nat) (v2 : Nat) (v3 : Nat), ((NatModel.multiplication v1 v2 v3) → (∀ (v4 : Nat) (v5 : Nat), ((NatModel.multiplication v4 v1 v5) → ((NatModel.successor v2 v4) → (NatModel.addition v1 v3 v5)))))) := by
  exact GLExport.peano_source_052 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_053 : (∀ (v1 : Nat) (v2 : Nat) (v3 : Nat), ((NatModel.addition v1 v2 v3) → (∀ (v4 : Nat) (v5 : Nat), ((NatModel.successor v4 v5) → ((NatModel.multiplication v1 v4 v2) → (NatModel.multiplication v1 v5 v3)))))) := by
  exact GLExport.peano_source_053 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_055 : (∀ (v1 : Nat) (v2 : Nat) (v3 : Nat), ((NatModel.addition v1 v2 v3) → (∀ (v4 : Nat) (v5 : Nat), ((NatModel.multiplication v1 v4 v5) → (∀ (v6 : Nat), ((NatModel.multiplication v2 v4 v6) → (∀ (v7 : Nat), ((NatModel.multiplication v3 v4 v7) → (NatModel.addition v5 v6 v7))))))))) := by
  exact GLExport.peano_source_055 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_054 : (∀ (v1 : Nat) (v2 : Nat) (v3 : Nat), ((NatModel.multiplication v1 v2 v3) → (NatModel.multiplication v2 v1 v3))) := by
  exact GLExport.peano_source_054 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_056 : (∀ (v1 : Nat) (v2 : Nat) (v3 : Nat), ((NatModel.multiplication v1 v2 v3) → (∀ (v4 : Nat) (v5 : Nat), ((NatModel.multiplication v4 v5 v1) → (∀ (v6 : Nat), ((NatModel.multiplication v4 v2 v6) → (NatModel.multiplication v6 v5 v3))))))) := by
  exact GLExport.peano_source_056 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_059 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.multiplication v1 v2 1) → (NatModel.multiplication v2 v1 1))) := by
  exact GLExport.peano_source_059 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_057 : (∀ (v1 : Nat) (v2 : Nat) (v3 : Nat), ((NatModel.multiplication v1 v2 v3) → (∀ (v4 : Nat) (v5 : Nat), ((NatModel.multiplication v2 v4 v5) → (∀ (v6 : Nat), ((NatModel.multiplication v6 v4 v1) → (NatModel.multiplication v5 v6 v3))))))) := by
  exact GLExport.peano_source_057 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem peano_source_066 : (∀ (v1 : Nat) (v2 : Nat) (v3 : Nat), ((NatModel.multiplication v1 v2 v3) → (∀ (v4 : Nat) (v5 : Nat), ((NatModel.multiplication v4 v5 v1) → (∀ (v6 : Nat), ((NatModel.multiplication v2 v5 v6) → (NatModel.multiplication v6 v4 v3))))))) := by
  exact GLExport.peano_source_066 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 NatModel.anchorPeano NatModel.induction NatModel.strongInduction

theorem gauss_source_067 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.successor v1 v2) → (gl_preorder NatModel.domain NatModel.addition v1 v2))) := by
  exact GLExport.gauss_source_067 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorGauss NatModel.induction NatModel.strongInduction

theorem gauss_source_068 : (gl_AnchorPeano NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1) := by
  exact GLExport.gauss_source_068 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorGauss NatModel.induction NatModel.strongInduction

theorem gauss_source_069 : (∀ (v1 : Nat) (V1 : GLSet Nat), ((gl_interval NatModel.domain NatModel.addition 0 v1 V1) → (V1 v1))) := by
  exact GLExport.gauss_source_069 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorGauss NatModel.induction NatModel.strongInduction

theorem gauss_source_070 : (∀ (v1 : Nat) (v2 : Nat) (v3 : Nat), ((NatModel.addition v1 v2 v3) → (NatModel.addition v2 v1 v3))) := by
  exact GLExport.gauss_source_070 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorGauss NatModel.induction NatModel.strongInduction

theorem gauss_source_072 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.successor v1 v2) → (∀ (v3 : Nat), ((gl_preorder NatModel.domain NatModel.addition v3 v1) → (gl_preorder NatModel.domain NatModel.addition v3 v2))))) := by
  exact GLExport.gauss_source_072 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorGauss NatModel.induction NatModel.strongInduction

theorem gauss_source_077 : (∀ (V1 : GLBinaryRelation Nat) (V2 : GLSet Nat), ((gl_fXY V1 V2 NatModel.domain) → (∀ (v1 : Nat), ((gl_interval NatModel.domain NatModel.addition 0 v1 V2) → (gl_sequence NatModel.domain NatModel.addition 0 v1 V1))))) := by
  exact GLExport.gauss_source_077 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorGauss NatModel.induction NatModel.strongInduction

theorem gauss_source_078 : (∀ (v1 : Nat) (V1 : GLSet Nat), ((gl_interval NatModel.domain NatModel.addition 0 v1 V1) → (∀ (V2 : GLBinaryRelation Nat), ((gl_sequence NatModel.domain NatModel.addition 0 v1 V2) → (gl_fXY V2 V1 NatModel.domain))))) := by
  exact GLExport.gauss_source_078 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorGauss NatModel.induction NatModel.strongInduction

theorem gauss_source_097 : (∀ (v1 : Nat) (V1 : GLSet Nat), ((gl_interval NatModel.domain NatModel.addition 0 v1 V1) → ((gl_interval NatModel.domain NatModel.addition 0 1 V1) → (V1 v1)))) := by
  exact GLExport.gauss_source_097 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorGauss NatModel.induction NatModel.strongInduction

theorem gauss_source_071 : (∀ (v1 : Nat) (v2 : Nat), ((gl_preorder NatModel.domain NatModel.addition v1 v2) → (∀ (v3 : Nat), ((gl_preorder NatModel.domain NatModel.addition v2 v3) → (gl_preorder NatModel.domain NatModel.addition v1 v3))))) := by
  exact GLExport.gauss_source_071 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorGauss NatModel.induction NatModel.strongInduction

theorem gauss_source_073 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.successor v1 v2) → (∀ (V1 : GLSet Nat) (V2 : GLSet Nat), ((gl_limitSet NatModel.domain NatModel.addition V1 v1 V2) → ((gl_interval NatModel.domain NatModel.addition 0 v2 V1) → (gl_interval NatModel.domain NatModel.addition 0 v1 V2)))))) := by
  exact GLExport.gauss_source_073 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorGauss NatModel.induction NatModel.strongInduction

theorem gauss_source_081 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.successor v1 v2) → (∀ (V1 : GLSet Nat) (V2 : GLSet Nat), ((gl_limitSet NatModel.domain NatModel.addition V1 v1 V2) → ((gl_interval NatModel.domain NatModel.addition 0 1 V1) → (¬ (gl_interval NatModel.domain NatModel.addition 0 v2 V2))))))) := by
  exact GLExport.gauss_source_081 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorGauss NatModel.induction NatModel.strongInduction

theorem gauss_source_083 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.successor v1 v2) → (∀ (V1 : GLSet Nat), ((gl_interval NatModel.domain NatModel.addition 0 v1 V1) → (¬ (gl_interval NatModel.domain NatModel.addition 0 v2 V1)))))) := by
  exact GLExport.gauss_source_083 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorGauss NatModel.induction NatModel.strongInduction

theorem gauss_source_087 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.successor v1 v2) → (∀ (V1 : GLSet Nat) (V2 : GLSet Nat), ((gl_limitSet NatModel.domain NatModel.addition V1 1 V2) → ((gl_interval NatModel.domain NatModel.addition 0 v1 V1) → (¬ (gl_interval NatModel.domain NatModel.addition 0 v2 V2))))))) := by
  exact GLExport.gauss_source_087 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorGauss NatModel.induction NatModel.strongInduction

theorem gauss_source_089 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.successor v1 v2) → (∀ (V1 : GLSet Nat) (V2 : GLSet Nat), ((gl_limitSet NatModel.domain NatModel.addition V1 2 V2) → ((gl_interval NatModel.domain NatModel.addition 0 v1 V1) → (¬ (gl_interval NatModel.domain NatModel.addition 0 v2 V2))))))) := by
  exact GLExport.gauss_source_089 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorGauss NatModel.induction NatModel.strongInduction

theorem gauss_source_091 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.successor v1 v2) → (∀ (V1 : GLSet Nat) (V2 : GLSet Nat), ((gl_limitSet NatModel.domain NatModel.addition V1 v1 V2) → ((gl_interval NatModel.domain NatModel.addition 0 v1 V1) → (¬ (gl_interval NatModel.domain NatModel.addition 0 v2 V2))))))) := by
  exact GLExport.gauss_source_091 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorGauss NatModel.induction NatModel.strongInduction

theorem gauss_source_093 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.successor v1 v2) → (∀ (V1 : GLSet Nat) (V2 : GLSet Nat), ((gl_limitSet NatModel.domain NatModel.addition V1 v2 V2) → ((gl_interval NatModel.domain NatModel.addition 0 v1 V1) → (¬ (gl_interval NatModel.domain NatModel.addition 0 v2 V2))))))) := by
  exact GLExport.gauss_source_093 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorGauss NatModel.induction NatModel.strongInduction

theorem gauss_source_095 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.successor v1 v2) → (∀ (V1 : GLSet Nat) (V2 : GLSet Nat), ((gl_limitSet NatModel.domain NatModel.addition V1 0 V2) → ((gl_interval NatModel.domain NatModel.addition 0 v1 V1) → (¬ (gl_interval NatModel.domain NatModel.addition 0 v2 V2))))))) := by
  exact GLExport.gauss_source_095 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorGauss NatModel.induction NatModel.strongInduction

theorem gauss_source_098 : (∀ (V1 : GLSet Nat), ((gl_interval NatModel.domain NatModel.addition 0 1 V1) → (V1 1))) := by
  exact GLExport.gauss_source_098 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorGauss NatModel.induction NatModel.strongInduction

theorem gauss_source_074 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.successor v1 v2) → (∀ (V1 : GLSet Nat), ((gl_interval NatModel.domain NatModel.addition 0 v2 V1) → (gl_existence8 NatModel.domain NatModel.addition V1 v1 0))))) := by
  exact GLExport.gauss_source_074 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorGauss NatModel.induction NatModel.strongInduction

theorem gauss_source_082 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.successor v1 v2) → (∀ (V1 : GLSet Nat), ((gl_interval NatModel.domain NatModel.addition 0 1 V1) → (gl_existence10 NatModel.domain NatModel.addition V1 v1 0 v2))))) := by
  exact GLExport.gauss_source_082 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorGauss NatModel.induction NatModel.strongInduction

theorem gauss_source_084 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.successor v1 v2) → (∀ (V1 : GLSet Nat) (V2 : GLSet Nat), ((gl_limitSet NatModel.domain NatModel.addition V1 v2 V2) → ((gl_interval NatModel.domain NatModel.addition 0 v1 V2) → (¬ (gl_interval NatModel.domain NatModel.addition 0 v2 V1))))))) := by
  exact GLExport.gauss_source_084 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorGauss NatModel.induction NatModel.strongInduction

theorem gauss_source_085 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.successor v1 v2) → (∀ (V1 : GLSet Nat) (V2 : GLSet Nat), ((gl_limitSet NatModel.domain NatModel.addition V1 v2 V2) → ((gl_interval NatModel.domain NatModel.addition 0 v2 V1) → (¬ (gl_interval NatModel.domain NatModel.addition 0 v1 V2))))))) := by
  exact GLExport.gauss_source_085 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorGauss NatModel.induction NatModel.strongInduction

theorem gauss_source_088 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.successor v1 v2) → (∀ (V1 : GLSet Nat), ((gl_interval NatModel.domain NatModel.addition 0 v1 V1) → (gl_existence10 NatModel.domain NatModel.addition V1 1 0 v2))))) := by
  exact GLExport.gauss_source_088 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorGauss NatModel.induction NatModel.strongInduction

theorem gauss_source_090 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.successor v1 v2) → (∀ (V1 : GLSet Nat), ((gl_interval NatModel.domain NatModel.addition 0 v1 V1) → (gl_existence10 NatModel.domain NatModel.addition V1 2 0 v2))))) := by
  exact GLExport.gauss_source_090 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorGauss NatModel.induction NatModel.strongInduction

theorem gauss_source_092 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.successor v1 v2) → (∀ (V1 : GLSet Nat), ((gl_interval NatModel.domain NatModel.addition 0 v1 V1) → (gl_existence10 NatModel.domain NatModel.addition V1 v1 0 v2))))) := by
  exact GLExport.gauss_source_092 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorGauss NatModel.induction NatModel.strongInduction

theorem gauss_source_094 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.successor v1 v2) → (∀ (V1 : GLSet Nat), ((gl_interval NatModel.domain NatModel.addition 0 v1 V1) → (gl_existence11 NatModel.domain NatModel.addition V1 v2 0))))) := by
  exact GLExport.gauss_source_094 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorGauss NatModel.induction NatModel.strongInduction

theorem gauss_source_096 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.successor v1 v2) → (∀ (V1 : GLSet Nat), ((gl_interval NatModel.domain NatModel.addition 0 v1 V1) → (gl_existence9 NatModel.domain NatModel.addition V1 0 v2))))) := by
  exact GLExport.gauss_source_096 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorGauss NatModel.induction NatModel.strongInduction

theorem gauss_source_075 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.successor v1 v2) → (∀ (V1 : GLBinaryRelation Nat) (V2 : GLBinaryRelation Nat), ((gl_limitSequence NatModel.domain NatModel.addition v1 V1 V2) → ((gl_sequence NatModel.domain NatModel.addition 0 v2 V1) → (gl_sequence NatModel.domain NatModel.addition 0 v1 V2)))))) := by
  exact GLExport.gauss_source_075 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorGauss NatModel.induction NatModel.strongInduction

theorem gauss_source_086 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.successor v1 v2) → (∀ (V1 : GLSet Nat), ((gl_interval NatModel.domain NatModel.addition 0 v2 V1) → (gl_existence10 NatModel.domain NatModel.addition V1 v2 0 v1))))) := by
  exact GLExport.gauss_source_086 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorGauss NatModel.induction NatModel.strongInduction

theorem gauss_source_076 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.successor v1 v2) → (∀ (V1 : GLBinaryRelation Nat), ((gl_sequence NatModel.domain NatModel.addition 0 v2 V1) → (gl_existence12 NatModel.domain NatModel.addition v1 V1 0))))) := by
  exact GLExport.gauss_source_076 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorGauss NatModel.induction NatModel.strongInduction

theorem gauss_source_079 : (∀ (v1 : Nat) (v2 : Nat) (v3 : Nat), ((NatModel.multiplication v1 v2 v3) → (∀ (v4 : Nat), ((gl_fold NatModel.domain NatModel.successor NatModel.addition NatModel.addition NatModel.identity 0 v1 v4) → ((NatModel.successor v1 v2) → (NatModel.multiplication 2 v4 v3)))))) := by
  exact GLExport.gauss_source_079 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorGauss NatModel.induction NatModel.strongInduction

theorem gauss_source_080 : (∀ (v1 : Nat) (v2 : Nat) (v3 : Nat), ((NatModel.multiplication v1 v2 v3) → (∀ (v4 : Nat), ((gl_fold NatModel.domain NatModel.successor NatModel.addition NatModel.addition NatModel.identity 0 v2 v4) → ((NatModel.successor v2 v1) → (NatModel.multiplication 2 v4 v3)))))) := by
  exact GLExport.gauss_source_080 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorGauss NatModel.induction NatModel.strongInduction

theorem gauss_source_099 : (∀ (v1 : Nat) (v2 : Nat), ((gl_fold NatModel.domain NatModel.successor NatModel.addition NatModel.addition NatModel.identity 0 v1 v2) → (∀ (v3 : Nat), ((NatModel.successor v1 v3) → (∀ (v4 : Nat), ((NatModel.multiplication 2 v2 v4) → (NatModel.multiplication v1 v3 v4))))))) := by
  exact GLExport.gauss_source_099 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorGauss NatModel.induction NatModel.strongInduction

theorem fta_source_100 : (∀ (v1 : Nat) (v2 : Nat) (V1 : GLBinaryRelation Nat) (V2 : GLBinaryRelation Nat), ((gl_appendToSequence v1 v2 V1 V2) → ((gl_ordered NatModel.domain NatModel.addition V2) → (gl_ordered NatModel.domain NatModel.addition V1)))) := by
  exact GLExport.FTA.fta_source_100 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_101 : (∀ (v1 : Nat) (v2 : Nat) (V1 : GLBinaryRelation Nat) (V2 : GLBinaryRelation Nat), ((gl_appendToSequence v1 v2 V1 V2) → ((gl_primeseq NatModel.domain NatModel.addition NatModel.multiplication 1 2 V2) → (gl_primeseq NatModel.domain NatModel.addition NatModel.multiplication 1 2 V1)))) := by
  exact GLExport.FTA.fta_source_101 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_102 : (gl_AnchorGauss NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity) := by
  exact GLExport.FTA.fta_source_102 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_103 : (gl_AnchorPeano NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1) := by
  exact GLExport.FTA.fta_source_103 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_104 : (∀ (v1 : Nat) (v2 : Nat) (V1 : GLBinaryRelation Nat) (V2 : GLBinaryRelation Nat), ((gl_appendToSequence v1 v2 V1 V2) → (∀ (V3 : GLBinaryRelation Nat), ((gl_appendToSequence v1 v2 V1 V3) → (V2 = V3))))) := by
  exact GLExport.FTA.fta_source_104 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_107 : (∀ (v1 : Nat) (V1 : GLBinaryRelation Nat), ((gl_EnumerationRelationSet1 0 v1 V1) → (∀ (v2 : Nat), ((v1 = v2) → (gl_inSequence NatModel.domain v2 V1))))) := by
  exact GLExport.FTA.fta_source_107 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_109 : (∀ (v1 : Nat) (V1 : GLBinaryRelation Nat), ((gl_EnumerationRelationSet1 0 v1 V1) → (∀ (V2 : GLBinaryRelation Nat), ((gl_EnumerationRelationSet1 0 v1 V2) → (V1 = V2))))) := by
  exact GLExport.FTA.fta_source_109 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_112 : (∀ (v1 : Nat) (V1 : GLSet Nat), ((gl_interval NatModel.domain NatModel.addition 0 v1 V1) → (∀ (v2 : Nat), ((gl_interval NatModel.domain NatModel.addition 0 v2 V1) → (v1 = v2))))) := by
  exact GLExport.FTA.fta_source_112 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_113 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.addition v1 1 v2) → (∀ (v3 : Nat) (v4 : Nat), ((NatModel.multiplication v3 v1 v4) → (∀ (v5 : Nat), ((NatModel.multiplication v3 v2 v5) → (NatModel.addition v4 v3 v5))))))) := by
  exact GLExport.FTA.fta_source_113 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_114 : (∀ (v1 : Nat) (v2 : Nat) (v3 : Nat), ((NatModel.addition v1 v2 v3) → (gl_preorder NatModel.domain NatModel.addition v1 v3))) := by
  exact GLExport.FTA.fta_source_114 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_116 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.successor v1 v2) → (∀ (V1 : GLBinaryRelation Nat), ((gl_sequence NatModel.domain NatModel.addition 0 v1 V1) → (∀ (v3 : Nat) (V2 : GLBinaryRelation Nat), ((gl_appendToSequence v2 v3 V1 V2) → (∀ (V3 : GLBinaryRelation Nat), ((gl_limitSequence NatModel.domain NatModel.addition v1 V2 V3) → (V3 = V1))))))))) := by
  exact GLExport.FTA.fta_source_116 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_118 : (∀ (v1 : Nat) (V1 : GLBinaryRelation Nat), ((gl_sequence NatModel.domain NatModel.addition 0 v1 V1) → (∀ (v2 : Nat) (v3 : Nat), ((V1 v2 v3) → (∀ (v4 : Nat) (v5 : Nat), ((V1 v4 v5) → ((v2 = v4) → (v3 = v5)))))))) := by
  exact GLExport.FTA.fta_source_118 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_119 : (∀ (v1 : Nat) (v2 : Nat) (v3 : Nat), ((NatModel.multiplication v1 v2 v3) → (∀ (v4 : Nat) (v5 : Nat), ((NatModel.addition v3 v4 v5) → (∀ (v6 : Nat) (v7 : Nat), ((gl_bezEq NatModel.domain NatModel.addition NatModel.multiplication 1 v4 v6 v1 v7) → (∀ (v8 : Nat), ((NatModel.multiplication v2 v6 v8) → (gl_existence17 v8 v7 NatModel.addition NatModel.domain NatModel.multiplication 1 v5 v6 v1))))))))) := by
  exact GLExport.FTA.fta_source_119 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_120 : (∀ (v1 : Nat) (v2 : Nat) (v3 : Nat), ((NatModel.multiplication v1 v2 v3) → (∀ (v4 : Nat) (v5 : Nat), ((NatModel.addition v3 v4 v5) → (∀ (v6 : Nat) (v7 : Nat), ((gl_bezEq NatModel.domain NatModel.addition NatModel.multiplication 1 v1 v6 v4 v7) → (∀ (v8 : Nat), ((NatModel.multiplication v2 v7 v8) → (gl_existence18 v6 v8 NatModel.addition NatModel.domain NatModel.multiplication 1 v1 v5 v7))))))))) := by
  exact GLExport.FTA.fta_source_120 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_122 : (∀ (v1 : Nat) (V1 : GLBinaryRelation Nat), ((gl_sequence NatModel.domain NatModel.addition 0 v1 V1) → (∀ (v2 : Nat) (v3 : Nat), ((V1 v2 v3) → (∀ (v4 : Nat) (v5 : Nat), ((V1 v4 v5) → ((¬ (v3 = v5)) → (¬ (v2 = v4))))))))) := by
  exact GLExport.FTA.fta_source_122 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_124 : (∀ (V1 : GLBinaryRelation Nat) (V2 : GLSet Nat), ((gl_fXY V1 V2 NatModel.domain) → (∀ (V3 : GLSet Nat), ((gl_fXY V1 V3 NatModel.domain) → (V2 = V3))))) := by
  exact GLExport.FTA.fta_source_124 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_126 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.successor v1 v2) → (∀ (v3 : Nat) (v4 : Nat), ((NatModel.addition v3 v4 v2) → (∀ (v5 : Nat), ((NatModel.successor v5 v4) → (NatModel.addition v3 v5 v1))))))) := by
  exact GLExport.FTA.fta_source_126 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_127 : (∀ (v1 : Nat) (v2 : Nat), ((gl_strictOrder NatModel.domain NatModel.addition v1 v2) → (∀ (v3 : Nat), ((NatModel.addition v1 v3 v2) → (¬ (0 = v3)))))) := by
  exact GLExport.FTA.fta_source_127 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_129 : (∀ (v1 : Nat) (v2 : Nat), ((gl_preorder NatModel.domain NatModel.addition v1 v2) → ((¬ (v1 = v2)) → (gl_strictOrder NatModel.domain NatModel.addition v1 v2)))) := by
  exact GLExport.FTA.fta_source_129 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_133 : (∀ (v1 : Nat) (v2 : Nat), ((gl_strictOrder NatModel.domain NatModel.addition v1 v2) → (¬ (gl_strictOrder NatModel.domain NatModel.addition v2 v1)))) := by
  exact GLExport.FTA.fta_source_133 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_135 : (∀ (v1 : Nat), ((NatModel.domain v1) → (∀ (v2 : Nat), ((v1 = v2) → (NatModel.multiplication 1 v1 v2))))) := by
  exact GLExport.FTA.fta_source_135 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_137 : (∀ (v1 : Nat), ((NatModel.domain v1) → (∀ (v2 : Nat), ((v1 = v2) → (gl_preorder NatModel.domain NatModel.addition v1 v2))))) := by
  exact GLExport.FTA.fta_source_137 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_138 : (∀ (v1 : Nat) (v2 : Nat), ((gl_preorder NatModel.domain NatModel.addition v1 v2) → ((¬ (gl_strictOrder NatModel.domain NatModel.addition v1 v2)) → (v1 = v2)))) := by
  exact GLExport.FTA.fta_source_138 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_143 : (∀ (v1 : Nat), ((NatModel.domain v1) → ((¬ (0 = v1)) → (gl_preorder NatModel.domain NatModel.addition 1 v1)))) := by
  exact GLExport.FTA.fta_source_143 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_144 : (∀ (v1 : Nat), ((NatModel.domain v1) → ((¬ (gl_preorder NatModel.domain NatModel.addition 1 v1)) → (0 = v1)))) := by
  exact GLExport.FTA.fta_source_144 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_146 : (∀ (v1 : Nat), ((NatModel.domain v1) → (gl_preorder NatModel.domain NatModel.addition 0 v1))) := by
  exact GLExport.FTA.fta_source_146 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_147 : (∀ (v1 : Nat) (v2 : Nat), ((gl_preorder NatModel.domain NatModel.addition v1 v2) → (∀ (v3 : Nat) (v4 : Nat), ((NatModel.addition v1 v3 v4) → (∀ (v5 : Nat), ((NatModel.addition v2 v3 v5) → (gl_preorder NatModel.domain NatModel.addition v4 v5))))))) := by
  exact GLExport.FTA.fta_source_147 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_148 : (∀ (v1 : Nat) (v2 : Nat), ((gl_preorder NatModel.domain NatModel.addition v1 v2) → (∀ (v3 : Nat), ((¬ (gl_preorder NatModel.domain NatModel.addition v3 v2)) → (¬ (v1 = v3)))))) := by
  exact GLExport.FTA.fta_source_148 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_154 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.successor v1 v2) → (∀ (v3 : Nat), ((gl_preorder NatModel.domain NatModel.addition v2 v3) → (¬ (v1 = v3)))))) := by
  exact GLExport.FTA.fta_source_154 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_157 : (∀ (v1 : Nat) (v2 : Nat) (v3 : Nat), ((NatModel.multiplication v1 v2 v3) → (gl_preorder NatModel.domain NatModel.multiplication v2 v3))) := by
  exact GLExport.FTA.fta_source_157 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_158 : (∀ (v1 : Nat) (V1 : GLSet Nat), ((gl_divisorSet NatModel.domain NatModel.addition NatModel.multiplication 2 v1 V1) → (gl_subset V1 NatModel.domain))) := by
  exact GLExport.FTA.fta_source_158 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_160 : (∀ (v1 : Nat) (v2 : Nat), ((gl_preorder NatModel.domain NatModel.multiplication v1 v2) → ((gl_preorder NatModel.domain NatModel.addition 1 v2) → (gl_preorder NatModel.domain NatModel.addition 1 v1)))) := by
  exact GLExport.FTA.fta_source_160 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_161 : (∀ (v1 : Nat), ((gl_preorder NatModel.domain NatModel.addition 1 v1) → (¬ (0 = v1)))) := by
  exact GLExport.FTA.fta_source_161 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_164 : (∀ (v1 : Nat) (V1 : GLSet Nat), ((gl_interval NatModel.domain NatModel.addition 0 v1 V1) → (∀ (V2 : GLSet Nat), ((gl_interval NatModel.domain NatModel.addition 0 v1 V2) → (V1 = V2))))) := by
  exact GLExport.FTA.fta_source_164 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_166 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.successor v1 v2) → (∀ (v3 : Nat), ((gl_preorder NatModel.domain NatModel.addition v2 v3) → (gl_preorder NatModel.domain NatModel.addition v1 v3))))) := by
  exact GLExport.FTA.fta_source_166 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_169 : (∀ (v1 : Nat) (v2 : Nat), ((v1 = v2) → (¬ (gl_strictOrder NatModel.domain NatModel.addition v1 v2)))) := by
  exact GLExport.FTA.fta_source_169 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_170 : (∀ (v1 : Nat) (v2 : Nat) (V1 : GLBinaryRelation Nat) (V2 : GLBinaryRelation Nat), ((gl_appendToSequence v1 v2 V1 V2) → (V2 v1 v2))) := by
  exact GLExport.FTA.fta_source_170 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_177 : (∀ (v1 : Nat) (v2 : Nat), ((gl_coprime NatModel.domain 0 NatModel.successor NatModel.multiplication v1 v2) → (∀ (v3 : Nat) (v4 : Nat), ((NatModel.multiplication v3 v2 v4) → (∀ (v5 : Nat), ((NatModel.addition v4 v5 v1) → (gl_coprime NatModel.domain 0 NatModel.successor NatModel.multiplication v2 v5))))))) := by
  exact GLExport.FTA.fta_source_177 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_179 : (∀ (v1 : Nat) (v2 : Nat) (v3 : Nat), ((NatModel.multiplication v1 v2 v3) → (∀ (v4 : Nat) (v5 : Nat), ((NatModel.addition v3 v4 v5) → (∀ (v6 : Nat), ((NatModel.multiplication v1 v6 v5) → (∀ (v7 : Nat), ((NatModel.addition v2 v7 v6) → (NatModel.multiplication v1 v7 v4))))))))) := by
  exact GLExport.FTA.fta_source_179 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_184 : (∀ (v1 : Nat) (v2 : Nat), ((gl_preorder NatModel.domain NatModel.multiplication v1 v2) → (∀ (v3 : Nat) (v4 : Nat), ((NatModel.multiplication v2 v3 v4) → (gl_preorder NatModel.domain NatModel.multiplication v1 v4))))) := by
  exact GLExport.FTA.fta_source_184 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_185 : (∀ (v1 : Nat), ((NatModel.domain v1) → (∀ (v2 : Nat), ((v1 = v2) → (gl_preorder NatModel.domain NatModel.multiplication v1 v2))))) := by
  exact GLExport.FTA.fta_source_185 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_193 : (∀ (v1 : Nat) (V1 : GLBinaryRelation Nat), ((gl_sequence NatModel.domain NatModel.addition 0 v1 V1) → (∀ (v2 : Nat), ((V1 v1 v2) → (gl_inSequence NatModel.domain v2 V1))))) := by
  exact GLExport.FTA.fta_source_193 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_197 : (∀ (v1 : Nat), ((gl_preorder NatModel.domain NatModel.addition 2 v1) → (¬ (1 = v1)))) := by
  exact GLExport.FTA.fta_source_197 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_201 : (∀ (v1 : Nat), ((NatModel.domain v1) → ((¬ (0 = v1)) → ((¬ (1 = v1)) → (gl_preorder NatModel.domain NatModel.addition 2 v1))))) := by
  exact GLExport.FTA.fta_source_201 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_202 : (∀ (v1 : Nat) (v2 : Nat), ((gl_preorder NatModel.domain NatModel.multiplication v1 v2) → (∀ (v3 : Nat), ((gl_preorder NatModel.domain NatModel.multiplication v2 v3) → (gl_preorder NatModel.domain NatModel.multiplication v1 v3))))) := by
  exact GLExport.FTA.fta_source_202 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_204 : (∀ (v1 : Nat), ((gl_prime NatModel.domain NatModel.addition NatModel.multiplication 1 2 v1) → (∀ (v2 : Nat), ((¬ (gl_preorder NatModel.domain NatModel.multiplication v1 v2)) → (∀ (v3 : Nat), ((gl_preorder NatModel.domain NatModel.multiplication v3 v1) → ((gl_preorder NatModel.domain NatModel.multiplication v3 v2) → (v3 = 1)))))))) := by
  exact GLExport.FTA.fta_source_204 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_210 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.successor v1 v2) → (∀ (V1 : GLBinaryRelation Nat), ((gl_sequence NatModel.domain NatModel.addition 0 v1 V1) → (∀ (v3 : Nat) (V2 : GLBinaryRelation Nat), ((gl_appendToSequence v2 v3 V1 V2) → (∀ (v4 : Nat) (v5 : Nat), ((V1 v4 v5) → (gl_inSequence NatModel.domain v5 V2))))))))) := by
  exact GLExport.FTA.fta_source_210 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_214 : (∀ (v1 : Nat) (V1 : GLBinaryRelation Nat), ((gl_sequence NatModel.domain NatModel.addition 0 v1 V1) → (∀ (v2 : Nat), ((V1 v1 v2) → ((gl_ordered NatModel.domain NatModel.addition V1) → (∀ (v3 : Nat), ((gl_inSequence NatModel.domain v3 V1) → (gl_preorder NatModel.domain NatModel.addition v2 v3)))))))) := by
  exact GLExport.FTA.fta_source_214 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_219 : (∀ (V1 : GLSet Nat), ((gl_EnumerationSet1 0 V1) → (gl_interval NatModel.domain NatModel.addition 0 0 V1))) := by
  exact GLExport.FTA.fta_source_219 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_221 : (∀ (v1 : Nat) (V1 : GLBinaryRelation Nat), ((gl_EnumerationRelationSet1 0 v1 V1) → (∀ (v2 : Nat) (v3 : Nat), ((V1 v2 v3) → (v3 = v1))))) := by
  exact GLExport.FTA.fta_source_221 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_225 : (∀ (v1 : Nat), ((gl_prime NatModel.domain NatModel.addition NatModel.multiplication 1 2 v1) → (∀ (V1 : GLBinaryRelation Nat), ((gl_EnumerationRelationSet1 0 v1 V1) → (gl_primeseq NatModel.domain NatModel.addition NatModel.multiplication 1 2 V1))))) := by
  exact GLExport.FTA.fta_source_225 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_227 : (∀ (V1 : GLBinaryRelation Nat), ((gl_primeseq NatModel.domain NatModel.addition NatModel.multiplication 1 2 V1) → (∀ (v1 : Nat), ((gl_prime NatModel.domain NatModel.addition NatModel.multiplication 1 2 v1) → (∀ (v2 : Nat) (V2 : GLBinaryRelation Nat), ((gl_appendToSequence v2 v1 V1 V2) → (gl_primeseq NatModel.domain NatModel.addition NatModel.multiplication 1 2 V2))))))) := by
  exact GLExport.FTA.fta_source_227 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_105 : (∀ (v1 : Nat) (v2 : Nat) (V1 : GLBinaryRelation Nat) (V2 : GLBinaryRelation Nat), ((gl_appendToSequence v1 v2 V1 V2) → (gl_existence23 v1 v2 V1 V2))) := by
  exact GLExport.FTA.fta_source_105 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_106 : (∀ (v1 : Nat) (v2 : Nat) (V1 : GLBinaryRelation Nat) (V2 : GLBinaryRelation Nat), ((gl_appendToSequence v1 v2 V1 V2) → (gl_existence22 v1 v2 V1 V2))) := by
  exact GLExport.FTA.fta_source_106 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_108 : (∀ (v1 : Nat) (v2 : Nat), ((v1 = v2) → (gl_existence24 0 v1 NatModel.domain v2))) := by
  exact GLExport.FTA.fta_source_108 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_110 : (∀ (v1 : Nat) (V1 : GLBinaryRelation Nat), ((gl_EnumerationRelationSet1 0 v1 V1) → (gl_existence26 0 v1 V1))) := by
  exact GLExport.FTA.fta_source_110 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_111 : (∀ (v1 : Nat) (V1 : GLBinaryRelation Nat), ((gl_EnumerationRelationSet1 0 v1 V1) → (gl_existence25 0 v1 V1))) := by
  exact GLExport.FTA.fta_source_111 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_115 : (∀ (v1 : Nat), ((gl_preorder NatModel.domain NatModel.addition 1 v1) → (∀ (v2 : Nat), ((gl_preorder NatModel.domain NatModel.addition 1 v2) → (∀ (v3 : Nat), ((NatModel.multiplication v1 v2 v3) → (gl_preorder NatModel.domain NatModel.addition 1 v3))))))) := by
  exact GLExport.FTA.fta_source_115 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_117 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.successor v1 v2) → (∀ (V1 : GLBinaryRelation Nat), ((gl_sequence NatModel.domain NatModel.addition 0 v1 V1) → (∀ (v3 : Nat) (V2 : GLBinaryRelation Nat), ((gl_appendToSequence v2 v3 V1 V2) → (gl_existence29 NatModel.domain NatModel.addition v1 V2 V1))))))) := by
  exact GLExport.FTA.fta_source_117 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_121 : (∀ (v1 : Nat) (v2 : Nat), ((gl_preorder NatModel.domain NatModel.multiplication v1 v2) → (∀ (v3 : Nat) (v4 : Nat), ((NatModel.addition v2 v3 v4) → ((gl_bez1 NatModel.domain NatModel.addition NatModel.multiplication 1 v1 v3) → (gl_bez1 NatModel.domain NatModel.addition NatModel.multiplication 1 v4 v1)))))) := by
  exact GLExport.FTA.fta_source_121 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_123 : (∀ (v1 : Nat) (V1 : GLBinaryRelation Nat), ((gl_sequence NatModel.domain NatModel.addition 0 v1 V1) → (∀ (v2 : Nat) (v3 : Nat), ((V1 v2 v3) → (∀ (v4 : Nat) (v5 : Nat), ((V1 v4 v5) → (gl_or0 v3 v5 v2 v4))))))) := by
  exact GLExport.FTA.fta_source_123 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_125 : (∀ (v1 : Nat) (V1 : GLBinaryRelation Nat), ((gl_sequence NatModel.domain NatModel.addition 0 v1 V1) → (∀ (v2 : Nat), ((gl_sequence NatModel.domain NatModel.addition 0 v2 V1) → (v1 = v2))))) := by
  exact GLExport.FTA.fta_source_125 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_128 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.successor v1 v2) → (∀ (v3 : Nat), ((gl_strictOrder NatModel.domain NatModel.addition v3 v2) → (gl_preorder NatModel.domain NatModel.addition v3 v1))))) := by
  exact GLExport.FTA.fta_source_128 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_134 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.successor v1 v2) → (∀ (v3 : Nat), ((gl_strictOrder NatModel.domain NatModel.addition v1 v3) → (¬ (gl_strictOrder NatModel.domain NatModel.addition v3 v2)))))) := by
  exact GLExport.FTA.fta_source_134 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_136 : (∀ (v1 : Nat), ((NatModel.domain v1) → (gl_preorder NatModel.domain NatModel.multiplication 1 v1))) := by
  exact GLExport.FTA.fta_source_136 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_139 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.successor v1 v2) → (∀ (v3 : Nat), ((gl_preorder NatModel.domain NatModel.addition v3 v1) → (gl_strictOrder NatModel.domain NatModel.addition v3 v2))))) := by
  exact GLExport.FTA.fta_source_139 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_140 : (∀ (v1 : Nat) (V1 : GLBinaryRelation Nat), ((gl_sequence NatModel.domain NatModel.addition 0 v1 V1) → (∀ (v2 : Nat), ((V1 v1 v2) → ((gl_ordered NatModel.domain NatModel.addition V1) → (∀ (v3 : Nat), ((gl_preorder NatModel.domain NatModel.addition v3 v2) → (∀ (v4 : Nat) (v5 : Nat) (v6 : Nat), ((gl_or5 v4 v5 V1 v6 v3) → (gl_preorder NatModel.domain NatModel.addition v3 v5)))))))))) := by
  exact GLExport.FTA.fta_source_140 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_145 : (∀ (v1 : Nat), ((NatModel.domain v1) → (gl_or9 0 v1 NatModel.domain NatModel.addition 1))) := by
  exact GLExport.FTA.fta_source_145 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_153 : (∀ (v1 : Nat) (v2 : Nat), ((gl_preorder NatModel.domain NatModel.addition v1 v2) → (gl_or8 v1 v2 NatModel.domain NatModel.addition))) := by
  exact GLExport.FTA.fta_source_153 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_165 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.successor v1 v2) → (∀ (V1 : GLBinaryRelation Nat), ((gl_sequence NatModel.domain NatModel.addition 0 v1 V1) → (∀ (v3 : Nat) (V2 : GLBinaryRelation Nat), ((gl_appendToSequence v2 v3 V1 V2) → (∀ (v4 : Nat), ((gl_fold NatModel.domain NatModel.successor NatModel.addition NatModel.multiplication V1 0 v1 v4) → (gl_fold NatModel.domain NatModel.successor NatModel.addition NatModel.multiplication V2 0 v1 v4))))))))) := by
  exact GLExport.FTA.fta_source_165 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_180 : (∀ (v1 : Nat), ((gl_preorder NatModel.domain NatModel.addition 1 v1) → (∀ (v2 : Nat) (v3 : Nat), ((NatModel.multiplication v2 v1 v3) → (∀ (v4 : Nat) (v5 : Nat), ((NatModel.multiplication v4 v1 v5) → ((gl_preorder NatModel.domain NatModel.addition v3 v5) → (gl_preorder NatModel.domain NatModel.addition v2 v4)))))))) := by
  exact GLExport.FTA.fta_source_180 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_183 : (∀ (v1 : Nat) (V1 : GLBinaryRelation Nat) (V2 : GLBinaryRelation Nat), ((gl_limitSequence NatModel.domain NatModel.addition v1 V1 V2) → (∀ (v2 : Nat), ((gl_fold NatModel.domain NatModel.successor NatModel.addition NatModel.multiplication V2 0 v1 v2) → (gl_fold NatModel.domain NatModel.successor NatModel.addition NatModel.multiplication V1 0 v1 v2))))) := by
  exact GLExport.FTA.fta_source_183 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_190 : (∀ (v1 : Nat), ((gl_preorder NatModel.domain NatModel.addition 1 v1) → (∀ (v2 : Nat) (v3 : Nat), ((NatModel.multiplication v2 v1 v3) → (gl_preorder NatModel.domain NatModel.addition v2 v3))))) := by
  exact GLExport.FTA.fta_source_190 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_195 : (∀ (v1 : Nat), ((gl_preorder NatModel.domain NatModel.addition 2 v1) → (∀ (v2 : Nat), ((gl_preorder NatModel.domain NatModel.addition 1 v2) → (∀ (v3 : Nat), ((NatModel.multiplication v1 v2 v3) → (gl_preorder NatModel.domain NatModel.addition 2 v3))))))) := by
  exact GLExport.FTA.fta_source_195 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_199 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.successor v1 v2) → (∀ (v3 : Nat) (v4 : Nat), ((NatModel.successor v3 v4) → ((gl_preorder NatModel.domain NatModel.addition v2 v4) → (gl_preorder NatModel.domain NatModel.addition v1 v3)))))) := by
  exact GLExport.FTA.fta_source_199 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_205 : (∀ (v1 : Nat), ((gl_prime NatModel.domain NatModel.addition NatModel.multiplication 1 2 v1) → (∀ (v2 : Nat), ((¬ (gl_preorder NatModel.domain NatModel.multiplication v1 v2)) → (gl_coprime NatModel.domain 0 NatModel.successor NatModel.multiplication v1 v2))))) := by
  exact GLExport.FTA.fta_source_205 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_209 : (∀ (v1 : Nat), ((gl_prime NatModel.domain NatModel.addition NatModel.multiplication 1 2 v1) → (∀ (v2 : Nat), ((gl_preorder NatModel.domain NatModel.multiplication v2 v1) → ((gl_preorder NatModel.domain NatModel.addition 2 v2) → (v2 = v1)))))) := by
  exact GLExport.FTA.fta_source_209 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_217 : (∀ (v1 : Nat), ((NatModel.domain v1) → (∀ (V1 : GLBinaryRelation Nat), ((gl_EnumerationRelationSet1 0 v1 V1) → (gl_ordered NatModel.domain NatModel.addition V1))))) := by
  exact GLExport.FTA.fta_source_217 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_220 : (gl_existence28 0 NatModel.domain NatModel.addition) := by
  exact GLExport.FTA.fta_source_220 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_226 : (∀ (v1 : Nat), ((NatModel.domain v1) → (gl_or10 0 v1 1 NatModel.domain NatModel.addition 2))) := by
  exact GLExport.FTA.fta_source_226 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_229 : (∀ (v1 : Nat), ((gl_prime NatModel.domain NatModel.addition NatModel.multiplication 1 2 v1) → (gl_existence21 0 v1 NatModel.domain NatModel.addition NatModel.multiplication 1 2))) := by
  exact GLExport.FTA.fta_source_229 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_130 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.successor v1 v2) → (∀ (V1 : GLBinaryRelation Nat), ((gl_sequence NatModel.domain NatModel.addition 0 v2 V1) → (∀ (v3 : Nat), ((V1 v2 v3) → (gl_existence19 NatModel.domain NatModel.addition 0 v1 v2 v3 V1))))))) := by
  exact GLExport.FTA.fta_source_130 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_131 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.successor v1 v2) → (∀ (V1 : GLBinaryRelation Nat), ((gl_sequence NatModel.domain NatModel.addition 0 v2 V1) → (∀ (v3 : Nat), ((V1 v2 v3) → (∀ (V2 : GLBinaryRelation Nat), ((gl_limitSequence NatModel.domain NatModel.addition v1 V1 V2) → (gl_appendToSequence v2 v3 V2 V1))))))))) := by
  exact GLExport.FTA.fta_source_131 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_141 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.successor v1 v2) → (∀ (V1 : GLBinaryRelation Nat), ((gl_sequence NatModel.domain NatModel.addition 0 v1 V1) → (∀ (v3 : Nat), ((V1 v1 v3) → ((gl_ordered NatModel.domain NatModel.addition V1) → (∀ (v4 : Nat), ((gl_preorder NatModel.domain NatModel.addition v4 v3) → (∀ (V2 : GLBinaryRelation Nat), ((gl_appendToSequence v2 v4 V1 V2) → (gl_ordered NatModel.domain NatModel.addition V2)))))))))))) := by
  exact GLExport.FTA.fta_source_141 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_149 : (∀ (v1 : Nat), ((NatModel.domain v1) → (∀ (v2 : Nat), ((NatModel.domain v2) → ((¬ (gl_preorder NatModel.domain NatModel.addition v1 v2)) → (gl_preorder NatModel.domain NatModel.addition v2 v1)))))) := by
  exact GLExport.FTA.fta_source_149 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_163 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.successor v1 v2) → (∀ (V1 : GLSet Nat), ((gl_interval NatModel.domain NatModel.addition 0 v1 V1) → (∀ (V2 : GLSet Nat), ((gl_addToSet V1 v2 V2) → (gl_interval NatModel.domain NatModel.addition 0 v2 V2))))))) := by
  exact GLExport.FTA.fta_source_163 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_167 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.successor v1 v2) → (∀ (V1 : GLBinaryRelation Nat), ((gl_sequence NatModel.domain NatModel.addition 0 v2 V1) → (∀ (v3 : Nat), ((V1 v2 v3) → (∀ (v4 : Nat), ((gl_fold NatModel.domain NatModel.successor NatModel.addition NatModel.multiplication V1 0 v1 v4) → (∀ (v5 : Nat), ((gl_fold NatModel.domain NatModel.successor NatModel.addition NatModel.multiplication V1 0 v2 v5) → (NatModel.multiplication v4 v3 v5))))))))))) := by
  exact GLExport.FTA.fta_source_167 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_176 : (∀ (v1 : Nat), ((NatModel.domain v1) → (∀ (v2 : Nat), ((gl_preorder NatModel.domain NatModel.addition 1 v2) → (gl_existence20 NatModel.domain NatModel.addition NatModel.multiplication v1 v2))))) := by
  exact GLExport.FTA.fta_source_176 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_191 : (∀ (v1 : Nat) (v2 : Nat), ((gl_strictOrder NatModel.domain NatModel.addition v1 v2) → (∀ (v3 : Nat), ((gl_preorder NatModel.domain NatModel.addition 1 v3) → (∀ (v4 : Nat), ((NatModel.multiplication v1 v3 v4) → (∀ (v5 : Nat), ((NatModel.multiplication v2 v3 v5) → (gl_strictOrder NatModel.domain NatModel.addition v4 v5))))))))) := by
  exact GLExport.FTA.fta_source_191 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_192 : (∀ (v1 : Nat) (v2 : Nat), ((gl_preorder NatModel.domain NatModel.multiplication v1 v2) → ((gl_preorder NatModel.domain NatModel.addition 1 v2) → (gl_preorder NatModel.domain NatModel.addition v1 v2)))) := by
  exact GLExport.FTA.fta_source_192 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_198 : (∀ (v1 : Nat), ((gl_preorder NatModel.domain NatModel.addition 1 v1) → (∀ (v2 : Nat), ((gl_preorder NatModel.domain NatModel.addition 2 v2) → (∀ (v3 : Nat), ((NatModel.multiplication v1 v2 v3) → (gl_strictOrder NatModel.domain NatModel.addition v1 v3))))))) := by
  exact GLExport.FTA.fta_source_198 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_203 : (∀ (v1 : Nat), ((gl_preorder NatModel.domain NatModel.addition 2 v1) → (∀ (V1 : GLSet Nat), ((gl_divisorSet NatModel.domain NatModel.addition NatModel.multiplication 2 v1 V1) → (∀ (v2 : Nat), ((gl_minimum NatModel.domain NatModel.addition V1 v2) → (gl_prime NatModel.domain NatModel.addition NatModel.multiplication 1 2 v2))))))) := by
  exact GLExport.FTA.fta_source_203 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_212 : (∀ (v1 : Nat), ((gl_preorder NatModel.domain NatModel.addition 2 v1) → (¬ (0 = v1)))) := by
  exact GLExport.FTA.fta_source_212 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_213 : (∀ (v1 : Nat), ((gl_preorder NatModel.domain NatModel.addition 1 v1) → (∀ (v2 : Nat) (v3 : Nat), ((NatModel.multiplication v2 v1 v3) → (∀ (v4 : Nat), ((NatModel.multiplication v4 v1 v3) → (v2 = v4))))))) := by
  exact GLExport.FTA.fta_source_213 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_218 : (∀ (v1 : Nat), ((NatModel.domain v1) → (gl_existence27 0 v1 NatModel.domain NatModel.addition))) := by
  exact GLExport.FTA.fta_source_218 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_222 : (∀ (v1 : Nat), ((NatModel.domain v1) → (∀ (V1 : GLBinaryRelation Nat), ((gl_EnumerationRelationSet1 0 v1 V1) → (gl_sequence NatModel.domain NatModel.addition 0 0 V1))))) := by
  exact GLExport.FTA.fta_source_222 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_132 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.successor v1 v2) → (∀ (V1 : GLBinaryRelation Nat), ((gl_sequence NatModel.domain NatModel.addition 0 v2 V1) → (∀ (v3 : Nat), ((V1 v2 v3) → (gl_existence33 NatModel.domain NatModel.addition v1 V1 v2 v3))))))) := by
  exact GLExport.FTA.fta_source_132 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_142 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.successor v1 v2) → (∀ (V1 : GLBinaryRelation Nat), ((gl_sequence NatModel.domain NatModel.addition 0 v1 V1) → (∀ (v3 : Nat), ((V1 v1 v3) → ((gl_ordered NatModel.domain NatModel.addition V1) → (∀ (v4 : Nat), ((gl_preorder NatModel.domain NatModel.addition v4 v3) → (gl_existence32 v2 v4 V1 NatModel.domain NatModel.addition)))))))))) := by
  exact GLExport.FTA.fta_source_142 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_150 : (∀ (v1 : Nat), ((NatModel.domain v1) → (∀ (v2 : Nat), ((NatModel.domain v2) → (gl_or11 NatModel.domain NatModel.addition v1 v2))))) := by
  exact GLExport.FTA.fta_source_150 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_151 : (∀ (v1 : Nat), ((NatModel.domain v1) → (∀ (v2 : Nat), ((NatModel.domain v2) → ((¬ (gl_strictOrder NatModel.domain NatModel.addition v1 v2)) → ((¬ (v1 = v2)) → (gl_strictOrder NatModel.domain NatModel.addition v2 v1))))))) := by
  exact GLExport.FTA.fta_source_151 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_168 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.successor v1 v2) → (∀ (V1 : GLSet Nat), ((gl_interval NatModel.domain NatModel.addition 0 v1 V1) → (gl_existence30 V1 v2 NatModel.domain NatModel.addition 0))))) := by
  exact GLExport.FTA.fta_source_168 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_178 : (∀ (v1 : Nat), ((NatModel.domain v1) → (∀ (v2 : Nat), ((NatModel.domain v2) → ((gl_coprime NatModel.domain 0 NatModel.successor NatModel.multiplication v1 v2) → (gl_bez1 NatModel.domain NatModel.addition NatModel.multiplication 1 v1 v2)))))) := by
  exact GLExport.FTA.fta_source_178 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_186 : (∀ (v1 : Nat) (V1 : GLBinaryRelation Nat), ((gl_sequence NatModel.domain NatModel.addition 0 v1 V1) → (∀ (v2 : Nat), ((gl_fold NatModel.domain NatModel.successor NatModel.addition NatModel.multiplication V1 0 v1 v2) → (∀ (v3 : Nat), ((gl_inSequence NatModel.domain v3 V1) → (gl_preorder NatModel.domain NatModel.multiplication v3 v2))))))) := by
  exact GLExport.FTA.fta_source_186 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_196 : (∀ (v1 : Nat) (V1 : GLBinaryRelation Nat), ((gl_sequence NatModel.domain NatModel.addition 0 v1 V1) → (∀ (v2 : Nat), ((gl_fold NatModel.domain NatModel.successor NatModel.addition NatModel.multiplication V1 0 v1 v2) → ((gl_primeseq NatModel.domain NatModel.addition NatModel.multiplication 1 2 V1) → (gl_preorder NatModel.domain NatModel.addition 2 v2)))))) := by
  exact GLExport.FTA.fta_source_196 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_200 : (∀ (v1 : Nat), ((gl_strictOrder NatModel.domain NatModel.addition 0 v1) → (∀ (V1 : GLBinaryRelation Nat), ((gl_sequence NatModel.domain NatModel.addition 0 v1 V1) → (∀ (v2 : Nat), ((V1 v1 v2) → (∀ (v3 : Nat), ((gl_fold NatModel.domain NatModel.successor NatModel.addition NatModel.multiplication V1 0 v1 v3) → ((gl_primeseq NatModel.domain NatModel.addition NatModel.multiplication 1 2 V1) → (gl_strictOrder NatModel.domain NatModel.addition v2 v3)))))))))) := by
  exact GLExport.FTA.fta_source_200 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_215 : (∀ (v1 : Nat) (v2 : Nat) (v3 : Nat), ((NatModel.multiplication v1 v2 v3) → ((gl_preorder NatModel.domain NatModel.addition 2 v1) → ((gl_preorder NatModel.domain NatModel.addition 1 v3) → (gl_strictOrder NatModel.domain NatModel.addition v2 v3))))) := by
  exact GLExport.FTA.fta_source_215 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_223 : (∀ (v1 : Nat), ((NatModel.domain v1) → (gl_existence31 0 v1 NatModel.domain NatModel.addition))) := by
  exact GLExport.FTA.fta_source_223 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_152 : (∀ (v1 : Nat), ((NatModel.domain v1) → (∀ (v2 : Nat), ((NatModel.domain v2) → (gl_or12 NatModel.domain NatModel.addition v1 v2))))) := by
  exact GLExport.FTA.fta_source_152 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_171 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.successor v1 v2) → (∀ (V1 : GLBinaryRelation Nat), ((gl_sequence NatModel.domain NatModel.addition 0 v1 V1) → (∀ (v3 : Nat), ((NatModel.domain v3) → (∀ (V2 : GLBinaryRelation Nat), ((gl_appendToSequence v2 v3 V1 V2) → (∀ (v4 : Nat), ((gl_fold NatModel.domain NatModel.successor NatModel.addition NatModel.multiplication V1 0 v1 v4) → (∀ (v5 : Nat), ((gl_fold NatModel.domain NatModel.successor NatModel.addition NatModel.multiplication V2 0 v2 v5) → (NatModel.multiplication v4 v3 v5))))))))))))) := by
  exact GLExport.FTA.fta_source_171 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_172 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.successor v1 v2) → (∀ (V1 : GLBinaryRelation Nat), ((gl_sequence NatModel.domain NatModel.addition 0 v1 V1) → (∀ (v3 : Nat), ((NatModel.domain v3) → (∀ (V2 : GLBinaryRelation Nat), ((gl_appendToSequence v2 v3 V1 V2) → (gl_sequence NatModel.domain NatModel.addition 0 v2 V2))))))))) := by
  exact GLExport.FTA.fta_source_172 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_194 : (∀ (v1 : Nat) (V1 : GLBinaryRelation Nat) (v2 : Nat), ((gl_primeFactorSeq NatModel.domain NatModel.successor NatModel.addition NatModel.multiplication 0 1 2 v1 V1 v2) → (∀ (v3 : Nat) (v4 : Nat), ((NatModel.multiplication v3 v2 v4) → (∀ (V2 : GLSet Nat), ((gl_divisorSet NatModel.domain NatModel.addition NatModel.multiplication 2 v4 V2) → ((gl_minimum NatModel.domain NatModel.addition V2 v3) → (∀ (v5 : Nat), ((V1 v1 v5) → (gl_preorder NatModel.domain NatModel.addition v3 v5)))))))))) := by
  exact GLExport.FTA.fta_source_194 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_224 : (∀ (v1 : Nat), ((NatModel.domain v1) → (∀ (V1 : GLBinaryRelation Nat), ((gl_EnumerationRelationSet1 0 v1 V1) → (gl_fold NatModel.domain NatModel.successor NatModel.addition NatModel.multiplication V1 0 0 v1))))) := by
  exact GLExport.FTA.fta_source_224 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_155 : (∀ (v1 : Nat), ((NatModel.domain v1) → (∀ (V1 : GLSet Nat), ((gl_subset V1 NatModel.domain) → ((¬ (gl_hasMinimum NatModel.domain NatModel.addition V1)) → (∀ (v2 : Nat), ((V1 v2) → (¬ (gl_preorder NatModel.domain NatModel.addition v2 v1))))))))) := by
  exact GLExport.FTA.fta_source_155 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_162 : (∀ (v1 : Nat) (v2 : Nat) (v3 : Nat), ((NatModel.multiplication v1 v2 v3) → (∀ (v4 : Nat) (v5 : Nat), ((NatModel.multiplication v4 v2 v5) → ((gl_strictOrder NatModel.domain NatModel.addition v3 v5) → (gl_strictOrder NatModel.domain NatModel.addition v1 v4)))))) := by
  exact GLExport.FTA.fta_source_162 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_173 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.successor v1 v2) → (∀ (V1 : GLBinaryRelation Nat), ((gl_sequence NatModel.domain NatModel.addition 0 v1 V1) → (∀ (v3 : Nat), ((NatModel.domain v3) → (gl_existence35 v2 v3 V1 NatModel.domain NatModel.addition 0))))))) := by
  exact GLExport.FTA.fta_source_173 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_230 : (∀ (v1 : Nat), ((NatModel.domain v1) → (gl_existence34 0 v1 NatModel.domain NatModel.successor NatModel.addition NatModel.multiplication))) := by
  exact GLExport.FTA.fta_source_230 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_156 : (∀ (v1 : Nat), ((NatModel.domain v1) → (∀ (V1 : GLSet Nat), ((gl_subset V1 NatModel.domain) → (∀ (v2 : Nat), ((V1 v2) → (gl_or15 NatModel.domain NatModel.addition V1 v2 v1))))))) := by
  exact GLExport.FTA.fta_source_156 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_174 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.successor v1 v2) → (∀ (V1 : GLBinaryRelation Nat), ((gl_sequence NatModel.domain NatModel.addition 0 v2 V1) → (∀ (v3 : Nat), ((V1 v2 v3) → (∀ (v4 : Nat), ((gl_fold NatModel.domain NatModel.successor NatModel.addition NatModel.multiplication V1 0 v1 v4) → (∀ (v5 : Nat), ((NatModel.multiplication v4 v3 v5) → (gl_fold NatModel.domain NatModel.successor NatModel.addition NatModel.multiplication V1 0 v2 v5))))))))))) := by
  exact GLExport.FTA.fta_source_174 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_175 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.successor v1 v2) → (∀ (V1 : GLBinaryRelation Nat), ((gl_sequence NatModel.domain NatModel.addition 0 v1 V1) → (∀ (v3 : Nat) (V2 : GLBinaryRelation Nat), ((gl_appendToSequence v2 v3 V1 V2) → (∀ (v4 : Nat), ((gl_fold NatModel.domain NatModel.successor NatModel.addition NatModel.multiplication V1 0 v1 v4) → (∀ (v5 : Nat), ((NatModel.multiplication v4 v3 v5) → (gl_fold NatModel.domain NatModel.successor NatModel.addition NatModel.multiplication V2 0 v2 v5))))))))))) := by
  exact GLExport.FTA.fta_source_175 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_181 : (∀ (v1 : Nat) (v2 : Nat), ((gl_preorder NatModel.domain NatModel.multiplication v1 v2) → (∀ (v3 : Nat) (v4 : Nat), ((NatModel.addition v2 v3 v4) → ((gl_preorder NatModel.domain NatModel.multiplication v1 v4) → (gl_preorder NatModel.domain NatModel.multiplication v1 v3)))))) := by
  exact GLExport.FTA.fta_source_181 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_159 : (∀ (v1 : Nat), ((gl_preorder NatModel.domain NatModel.addition 2 v1) → (∀ (V1 : GLSet Nat), ((gl_divisorSet NatModel.domain NatModel.addition NatModel.multiplication 2 v1 V1) → (gl_hasMinimum NatModel.domain NatModel.addition V1))))) := by
  exact GLExport.FTA.fta_source_159 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_182 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.successor v1 v2) → (∀ (V1 : GLBinaryRelation Nat), ((gl_sequence NatModel.domain NatModel.addition 0 v1 V1) → (∀ (v3 : Nat), ((gl_fold NatModel.domain NatModel.successor NatModel.addition NatModel.multiplication V1 0 v1 v3) → (∀ (v4 : Nat) (v5 : Nat), ((NatModel.multiplication v3 v4 v5) → (gl_existence37 v2 v4 V1 NatModel.domain NatModel.successor NatModel.addition NatModel.multiplication 0 v5))))))))) := by
  exact GLExport.FTA.fta_source_182 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_187 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.addition v1 1 v2) → (∀ (v3 : Nat) (v4 : Nat), ((NatModel.multiplication v3 v1 v4) → (∀ (v5 : Nat), ((gl_preorder NatModel.domain NatModel.multiplication v5 v2) → ((gl_preorder NatModel.domain NatModel.multiplication v5 v4) → (gl_preorder NatModel.domain NatModel.multiplication v5 v3)))))))) := by
  exact GLExport.FTA.fta_source_187 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_188 : (∀ (v1 : Nat) (v2 : Nat), ((NatModel.addition v1 1 v2) → (∀ (v3 : Nat) (v4 : Nat), ((NatModel.multiplication v3 v2 v4) → (∀ (v5 : Nat), ((gl_preorder NatModel.domain NatModel.multiplication v5 v1) → ((gl_preorder NatModel.domain NatModel.multiplication v5 v4) → (gl_preorder NatModel.domain NatModel.multiplication v5 v3)))))))) := by
  exact GLExport.FTA.fta_source_188 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_189 : (∀ (v1 : Nat) (v2 : Nat), ((gl_bez1 NatModel.domain NatModel.addition NatModel.multiplication 1 v1 v2) → (∀ (v3 : Nat) (v4 : Nat), ((NatModel.multiplication v2 v3 v4) → ((gl_preorder NatModel.domain NatModel.multiplication v1 v4) → (gl_preorder NatModel.domain NatModel.multiplication v1 v3)))))) := by
  exact GLExport.FTA.fta_source_189 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_216 : (∀ (v1 : Nat), ((gl_preorder NatModel.domain NatModel.addition 2 v1) → (gl_existence36 NatModel.domain NatModel.addition NatModel.multiplication 2 v1))) := by
  exact GLExport.FTA.fta_source_216 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_206 : (∀ (v1 : Nat), ((gl_prime NatModel.domain NatModel.addition NatModel.multiplication 1 2 v1) → (∀ (v2 : Nat) (v3 : Nat) (v4 : Nat), ((NatModel.multiplication v2 v3 v4) → ((gl_preorder NatModel.domain NatModel.multiplication v1 v4) → ((¬ (gl_preorder NatModel.domain NatModel.multiplication v1 v2)) → (gl_preorder NatModel.domain NatModel.multiplication v1 v3))))))) := by
  exact GLExport.FTA.fta_source_206 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_207 : (∀ (v1 : Nat), ((gl_prime NatModel.domain NatModel.addition NatModel.multiplication 1 2 v1) → (∀ (v2 : Nat) (v3 : Nat) (v4 : Nat), ((NatModel.multiplication v2 v3 v4) → ((gl_preorder NatModel.domain NatModel.multiplication v1 v4) → ((¬ (gl_preorder NatModel.domain NatModel.multiplication v1 v3)) → (gl_preorder NatModel.domain NatModel.multiplication v1 v2))))))) := by
  exact GLExport.FTA.fta_source_207 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_228 : (∀ (v1 : Nat), ((gl_preorder NatModel.domain NatModel.addition 2 v1) → (gl_hasPrimeFactorization NatModel.domain NatModel.successor NatModel.addition NatModel.multiplication 0 1 2 v1))) := by
  exact GLExport.FTA.fta_source_228 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_208 : (∀ (v1 : Nat), ((gl_prime NatModel.domain NatModel.addition NatModel.multiplication 1 2 v1) → (∀ (v2 : Nat) (v3 : Nat) (v4 : Nat), ((NatModel.multiplication v2 v3 v4) → ((gl_preorder NatModel.domain NatModel.multiplication v1 v4) → (gl_or16 NatModel.domain NatModel.multiplication v1 v2 v3)))))) := by
  exact GLExport.FTA.fta_source_208 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_211 : (∀ (v1 : Nat) (V1 : GLBinaryRelation Nat), ((gl_sequence NatModel.domain NatModel.addition 0 v1 V1) → (∀ (v2 : Nat), ((gl_fold NatModel.domain NatModel.successor NatModel.addition NatModel.multiplication V1 0 v1 v2) → ((gl_primeseq NatModel.domain NatModel.addition NatModel.multiplication 1 2 V1) → (∀ (v3 : Nat), ((gl_prime NatModel.domain NatModel.addition NatModel.multiplication 1 2 v3) → ((gl_preorder NatModel.domain NatModel.multiplication v3 v2) → (gl_inSequence NatModel.domain v3 V1))))))))) := by
  exact GLExport.FTA.fta_source_211 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

theorem fta_source_231 : (∀ (v1 : Nat) (V1 : GLBinaryRelation Nat) (v2 : Nat), ((gl_primeFactorSeq NatModel.domain NatModel.successor NatModel.addition NatModel.multiplication 0 1 2 v1 V1 v2) → (∀ (v3 : Nat) (V2 : GLBinaryRelation Nat), ((gl_primeFactorSeq NatModel.domain NatModel.successor NatModel.addition NatModel.multiplication 0 1 2 v3 V2 v2) → (V1 = V2))))) := by
  exact GLExport.FTA.fta_source_231 NatModel.domain 0 NatModel.successor NatModel.addition NatModel.multiplication 1 2 NatModel.identity NatModel.anchorFTA NatModel.induction NatModel.strongInduction

end GLExport.NatResults
