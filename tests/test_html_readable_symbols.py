# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt)
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Tests for the order-relation symbols of the HTML export's readable
captions (``visu_helpers.ReadableForm``), and for the internal witness-name
detector of ``generate_full_proof_graph``."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import generate_full_proof_graph as gen  # noqa: E402
import visu_helpers  # noqa: E402


def caption(expression: str, as_html: bool = False) -> str:
    """Theorem caption of a one-premise chain or an atom."""
    return visu_helpers.ReadableForm({}, {}).theorem(expression, as_html=as_html)


class OrderSymbolTests(unittest.TestCase):
    """Every order shape of the shortcut corpus, positive and negated."""

    def test_natural_order_and_its_negation(self) -> None:
        self.assertEqual(caption("(preorder[N,+,i1,v3])"), "1 ≤ v₃")
        self.assertEqual(caption("!(preorder[N,+,v1,v2])"), "v₁ ≰ v₂")

    def test_divisibility_and_its_negation(self) -> None:
        self.assertEqual(caption("(preorder[N,*,v1,v2])"), "v₁ ∣ v₂")
        self.assertEqual(caption("!(preorder[N,*,i2,v2])"), "2 ∤ v₂")

    def test_strict_order_and_its_negation(self) -> None:
        self.assertEqual(caption("(strictOrder[N,+,v1,v2])"), "v₁ < v₂")
        self.assertEqual(caption("!(strictOrder[N,+,v1,v2])"), "v₁ ≮ v₂")

    def test_proper_divisor_has_no_single_glyph(self) -> None:
        self.assertEqual(caption("(strictOrder[N,*,v1,v2])"), "v₁ ∣ v₂ ∧ v₁ ≠ v₂")
        self.assertEqual(caption("!(strictOrder[N,*,v1,v2])"), "¬(v₁ ∣ v₂ ∧ v₁ ≠ v₂)")

    def test_several_atoms_in_one_caption(self) -> None:
        self.assertEqual(
            caption("(>[](preorder[N,*,v1,v2])(>[]!(strictOrder[N,+,v2,v3])(preorder[N,+,v1,v3])))"),
            "from v₁ ∣ v₂, v₂ ≮ v₃ follows v₁ ≤ v₃")

    def test_unknown_operation_slot_asserts(self) -> None:
        with self.assertRaises(AssertionError):
            caption("(preorder[N,s,v1,v2])")

    def test_html_escapes_the_strict_order_sign(self) -> None:
        html_text = caption("(strictOrder[N,+,v1,v2])", as_html=True)
        self.assertNotIn("<sub>1</sub> < ", html_text)
        self.assertIn("v<sub>1</sub> &lt; v<sub>2</sub>", html_text)

    def test_html_keeps_divisibility_and_carrier_free(self) -> None:
        html_text = caption("(preorder[N,*,v1,v2])", as_html=True)
        self.assertIn("v<sub>1</sub> ∣ v<sub>2</sub>", html_text)
        self.assertNotIn("ℕ", html_text)


class InternalWitnessNameTests(unittest.TestCase):
    def test_current_and_historical_witness_spellings(self) -> None:
        for name in ("force_lev_2_9", "int_0_lev_2_9", "int_10_lev_2_9", "it_3_lev_2_9", "int_lev_2_9"):
            for prefix in ("", "u_", "u_u_"):
                with self.subTest(name=prefix + name):
                    self.assertTrue(gen._looks_like_internal_var_token(prefix + name))
        for name in ("force_lev_2", "int_x_lev_2_9", "int_1_lev_2_9_extra", "ordinary"):
            self.assertFalse(gen._looks_like_internal_var_token(name))


if __name__ == "__main__":
    unittest.main()
