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

/-- Induction for one fixed domain, zero, and successor relation. -/
def RelationalInduction {α : Type u} (N : GLSet α) (zero : α)
    (succ : GLBinaryRelation α) : Prop :=
  ∀ (P : α → Prop) (k : α), P zero →
    (∀ n, N n → P n → ∀ m, succ n m → P m) → N k → P k

/-- Strong induction for one fixed domain, zero, successor and addition
relation: the successor step may use the property at every value `j` below the
current value `m`, written `j + x = m` and `j ≠ m`. -/
def StrongRelationalInduction {α : Type u} (N : GLSet α) (zero : α)
    (succ : GLBinaryRelation α) (add : GLTernaryRelation α) : Prop :=
  ∀ (P : α → Prop) (k : α), P zero →
    (∀ n, N n → ∀ m, succ n m → (∀ j x, add j x m → ¬ j = m → P j) → P m) →
    N k → P k

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

/-- Classical De Morgan law for a negated conjunction. GL splits a negated
conjunction premise of an integrated implication into the cases of its
complemented conjuncts; the case elimination rewrites the unfolded premise
with this law. -/
theorem notAndIffNotOrNot (left right : Prop) :
    ¬ (left ∧ right) ↔ (¬ left ∨ ¬ right) := by
  constructor
  · intro not_both
    apply Classical.byContradiction
    intro no_disjunction
    apply not_both
    constructor
    · apply Classical.byContradiction
      intro no_left
      exact no_disjunction (Or.inl no_left)
    · apply Classical.byContradiction
      intro no_right
      exact no_disjunction (Or.inr no_right)
  · intro disjunction both
    cases disjunction with
    | inl no_left => exact no_left both.1
    | inr no_right => exact no_right both.2

/-- Classical double negation: a negated conjunct complements to its
positive core. -/
theorem notNotIff (value : Prop) : ¬ ¬ value ↔ value := by
  constructor
  · intro not_not_value
    exact Classical.byContradiction not_not_value
  · intro holds absent
    exact absent holds

/-- Extensionality of GL sets: two sets are equal exactly when each is a
subset of the other. GL integrates a set-typed equality through its two
subsets. -/
theorem setEqIffSubsets {α : Type u} (A B : GLSet α) :
    A = B ↔ ((∀ x, A x → B x) ∧ (∀ x, B x → A x)) := by
  constructor
  · intro equal
    subst equal
    exact ⟨fun _ member => member, fun _ member => member⟩
  · intro subsets
    funext x
    exact propext ⟨subsets.1 x, subsets.2 x⟩

/-- Extensionality of GL binary relations: equal exactly when each contains
the other. -/
theorem binaryRelationEqIffSubsets {α : Type u} (A B : GLBinaryRelation α) :
    A = B ↔ ((∀ x y, A x y → B x y) ∧ (∀ x y, B x y → A x y)) := by
  constructor
  · intro equal
    subst equal
    exact ⟨fun _ _ member => member, fun _ _ member => member⟩
  · intro subsets
    funext x y
    exact propext ⟨subsets.1 x y, subsets.2 x y⟩

/-- Extensionality of GL ternary relations: equal exactly when each contains
the other. -/
theorem ternaryRelationEqIffSubsets {α : Type u} (A B : GLTernaryRelation α) :
    A = B ↔ ((∀ x y z, A x y z → B x y z) ∧ (∀ x y z, B x y z → A x y z)) := by
  constructor
  · intro equal
    subst equal
    exact ⟨fun _ _ _ member => member, fun _ _ _ member => member⟩
  · intro subsets
    funext x y z
    exact propext ⟨subsets.1 x y z, subsets.2 x y z⟩

end GLExport
