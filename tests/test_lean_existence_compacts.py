# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt)
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Tests for the Lean writer's replay of an `expansion` row that turns a
negated existence compact into one of its implication compacts
(``proof_export.lean._existence_implication_direction``)."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proof_export import lean  # noqa: E402
from proof_export.mpl import parse_mpl  # noqa: E402


DEFINITIONS = {
    # ¬ ∀ x, (in[x,u_1]) → ¬ (in2[x,u_2,u_3])   — "some element of u_1 is a u_3-successor of u_2"
    "existence3": {"arity": 3, "category": "existence", "elements": ["(in[1,u_1])", "(in2[1,u_2,u_3])"]},
    # ∀ x, (in2[x,u_1,u_2]) → ¬ (in[x,u_3])    — right → ¬ left
    "implication1247": {"arity": 3, "category": "implication", "elements": ["(in2[1,u_1,u_2])", "!(in[1,u_3])"]},
    # ∀ x, (in[x,u_1]) → ¬ (in2[x,u_2,u_3])    — left → ¬ right
    "implication9": {"arity": 3, "category": "implication", "elements": ["(in[2,u_1])", "!(in2[2,u_2,u_3])"]},
    # ∀ x, (in[x,u_1]) → (in2[x,u_2,u_3])       — not a negated conclusion
    "implication0": {"arity": 3, "category": "implication", "elements": ["(in[1,u_1])", "(in2[1,u_2,u_3])"]},
    # ∀ x, (in[x,u_1]) → ¬ (in[x,u_2])          — unrelated to the existence
    "implication5": {"arity": 2, "category": "implication", "elements": ["(in[1,u_1])", "!(in[1,u_2])"]},
}
NEGATED_EXISTENCE = parse_mpl("!(existence3[N,v1,s])")


class ExistenceImplicationDirectionTests(unittest.TestCase):
    def test_swapped_orientation_is_recognized(self) -> None:
        self.assertEqual(
            lean._existence_implication_direction(
                NEGATED_EXISTENCE, parse_mpl("(implication1247[v1,s,N])"), DEFINITIONS,
            ),
            (["1"], True),
        )

    def test_same_orientation_is_recognized_with_its_own_placeholder(self) -> None:
        self.assertEqual(
            lean._existence_implication_direction(
                NEGATED_EXISTENCE, parse_mpl("(implication9[N,v1,s])"), DEFINITIONS,
            ),
            (["1"], False),
        )

    def test_other_shapes_take_the_ordinary_unfold(self) -> None:
        self.assertIsNone(
            lean._existence_implication_direction(
                parse_mpl("(existence3[N,v1,s])"), parse_mpl("(implication1247[v1,s,N])"), DEFINITIONS,
            )
        )
        self.assertIsNone(
            lean._existence_implication_direction(
                NEGATED_EXISTENCE, parse_mpl("(implication0[N,v1,s])"), DEFINITIONS,
            )
        )
        self.assertIsNone(
            lean._existence_implication_direction(NEGATED_EXISTENCE, parse_mpl("(in[v1,N])"), DEFINITIONS)
        )
        self.assertIsNone(
            lean._existence_implication_direction(
                NEGATED_EXISTENCE, parse_mpl("(implication1247[v1,s,N])"), None,
            )
        )

    def test_a_compact_of_another_existence_asserts(self) -> None:
        with self.assertRaises(AssertionError):
            lean._existence_implication_direction(
                NEGATED_EXISTENCE, parse_mpl("(implication5[N,v1])"), DEFINITIONS,
            )


if __name__ == "__main__":
    unittest.main()
