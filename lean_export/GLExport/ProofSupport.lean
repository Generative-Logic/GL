/-
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-/

import Std.Tactic

namespace GLExport

universe u

/-- An ordinary unary predicate represents a GL set. -/
abbrev GLSet (α : Type u) := α → Prop

/-- An ordinary binary predicate represents a GL binary relation. -/
abbrev GLBinaryRelation (α : Type u) := α → α → Prop

/-- An ordinary ternary predicate represents a GL ternary relation. -/
abbrev GLTernaryRelation (α : Type u) := α → α → α → Prop

/-- Classical elimination for GL's negated-universal existence encoding. -/
theorem existsAndOfNotForallImpNot
    {α : Type u}
    {P Q : α → Prop}
    (encoded : ¬ ∀ value, P value → ¬ Q value) :
    ∃ value, P value ∧ Q value := by
  classical
  apply Classical.byContradiction
  intro no_witness
  apply encoded
  intro value p_value q_value
  apply no_witness
  exact ⟨value, p_value, q_value⟩

/-- Classical equivalence used when an external compact OR fact is replayed
against GL's historical negated-conjunction expansion. -/
theorem orIffNotAndNot (left right : Prop) :
    (left ∨ right) ↔ ¬ (¬ left ∧ ¬ right) := by
  constructor
  · intro disjunction both_absent
    cases disjunction with
    | inl left_value => exact both_absent.1 left_value
    | inr right_value => exact both_absent.2 right_value
  · intro not_both_absent
    apply Classical.byContradiction
    intro no_disjunction
    apply not_both_absent
    constructor
    · intro left_value
      exact no_disjunction (Or.inl left_value)
    · intro right_value
      exact no_disjunction (Or.inr right_value)

end GLExport
