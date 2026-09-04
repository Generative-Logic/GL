# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt)
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Tests for the order-relation symbols of the HTML export's readable
captions (``generate_full_proof_graph._apply_order_symbols`` and its use in
``_htmlify_readable``)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import generate_full_proof_graph as gen  # noqa: E402


class OrderSymbolTests(unittest.TestCase):
    """Every order shape of the shortcut corpus, positive and negated."""

    def test_natural_order_and_its_negation(self) -> None:
        self.assertEqual(gen._apply_order_symbols("(preorder[N,+,1,v3])"), "1 ≤ v3")
        self.assertEqual(gen._apply_order_symbols("!(preorder[N,+,v1,v2])"), "v1 ≰ v2")

    def test_divisibility_and_its_negation(self) -> None:
        self.assertEqual(gen._apply_order_symbols("(preorder[N,*,v1,v2])"), "v1 ∣ v2")
        self.assertEqual(gen._apply_order_symbols("!(preorder[N,*,2,v2])"), "2 ∤ v2")

    def test_strict_order_and_its_negation(self) -> None:
        self.assertEqual(gen._apply_order_symbols("(strictOrder[N,+,v1,v2])"), "v1 < v2")
        self.assertEqual(gen._apply_order_symbols("!(strictOrder[N,+,v1,v2])"), "v1 ≮ v2")

    def test_proper_divisor_has_no_single_glyph(self) -> None:
        self.assertEqual(
            gen._apply_order_symbols("(strictOrder[N,*,v1,v2])"),
            "v1 ∣ v2, v1 ≠ v2")
        self.assertEqual(
            gen._apply_order_symbols("!(strictOrder[N,*,v1,v2])"),
            "¬(v1 ∣ v2, v1 ≠ v2)")

    def test_several_atoms_in_one_caption(self) -> None:
        caption = "from (preorder[N,*,v1,v2]), !(strictOrder[N,+,v2,v3]) follows (preorder[N,+,v1,v3])"
        self.assertEqual(
            gen._apply_order_symbols(caption),
            "from v1 ∣ v2, v2 ≮ v3 follows v1 ≤ v3")

    def test_unknown_operation_slot_asserts(self) -> None:
        with self.assertRaises(AssertionError):
            gen._apply_order_symbols("(preorder[N,s,v1,v2])")

    def test_htmlify_escapes_the_strict_order_sign(self) -> None:
        html_text = gen._htmlify_readable("(strictOrder[N,+,v1,v2])")
        self.assertIn("&lt;", html_text)
        self.assertNotIn("<sub>1</sub> < ", html_text)
        self.assertIn("v<sub>1</sub> &lt; v<sub>2</sub>", html_text)

    def test_htmlify_keeps_divisibility_and_carrier_free(self) -> None:
        html_text = gen._htmlify_readable("(preorder[N,*,v1,v2])")
        self.assertIn("v<sub>1</sub> ∣ v<sub>2</sub>", html_text)
        self.assertNotIn("ℕ", html_text)


if __name__ == "__main__":
    unittest.main()
