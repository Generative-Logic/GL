# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt)
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Tests for the grey readable captions of the HTML proof export
(``visu_helpers.ReadableForm`` and its helpers)."""

import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import visu_helpers as vh  # noqa: E402
from proof_export.mpl import Atom, Conjunction, Implication, Negation, parse_mpl  # noqa: E402

# Compiled definitions shaped exactly like the GL binary entries they mirror.
DEFINITIONS = {
    "existence3": {"category": "existence", "signature": "(existence3[u_1,u_2,u_3])",
                   "elements": ["(in[1,u_1])", "(in2[1,u_2,u_3])"]},
    "existence22": {"category": "existence", "signature": "(existence22[u_1,u_2,u_3,u_4,u_5])",
                    "elements": ["(limitSequence[u_1,u_2,u_3,u_4,1])", "(=[u_5,1])"]},
    "or1": {"category": "or", "signature": "(or1[u_1,u_2,u_3,u_4])",
            "elements": ["(existence3[u_1,u_2,u_3])", "(=[u_2,u_4])"]},
    "or8": {"category": "or", "signature": "(or8[u_1,u_2,u_3,u_4,u_5])",
            "elements": ["(in2[u_1,u_2,u_3])", "(and27[u_2,u_4,u_1,u_5])"]},
    "or15": {"category": "or", "signature": "(or15[u_1,u_2,u_3,u_4])",
             "elements": ["(strictOrder[u_1,u_2,u_3,u_4])", "(=[u_3,u_4])",
                          "(strictOrder[u_1,u_2,u_4,u_3])"]},
    "and27": {"category": "and", "signature": "(and27[u_1,u_2,u_3,u_4])",
              "elements": ["(=[u_1,u_2])", "(=[u_3,u_4])"]},
    "and22": {"category": "and", "signature": "(and22[u_1,u_2,u_3,u_4,u_5,u_6])",
              "elements": ["(in[u_1,u_2])", "(in[u_3,u_4])", "(in[u_5,u_6])"]},
    "implication33": {"category": "implication",
                      "signature": "(implication33[u_1,u_2,u_3,u_4,u_5])",
                      "elements": ["(in[1,u_1])", "(in2[1,2,u_2])", "(in[2,u_1])",
                                   "(in2[1,3,u_3])", "(in2[2,4,u_4])", "(in3[3,4,5,u_5])",
                                   "(in2[2,5,u_3])"]},
    "implication26": {"category": "implication", "signature": "(implication26[u_1,u_2,u_3,u_4])",
                      "elements": ["(in[1,u_1])", "(preorder[u_2,u_3,u_4,1])"]},
    "implication35": {"category": "implication", "signature": "(implication35[u_1,u_2])",
                      "elements": ["(in2[1,2,u_1])", "(in2[1,2,u_2])"]},
    "implication2525": {"category": "implication", "signature": "(implication2525[])",
                        "elements": ["(AnchorFTA[1,2,3,4,5,6,7,8])", "(in[9,1])", "(in[10,1])",
                                     "(or15[1,4,9,10])"]},
    "residual": {"category": "and", "signature": "(residual[u_1,u_2,u_3,u_4])",
                 "elements": ["(in3[u_3,u_2,u_1,u_4])"]},
    "infiniteSequence": {"category": "and", "signature": "(infiniteSequence[u_1,u_2])",
                         "elements": ["(fXY[u_1,u_2,u_2])"]},
    "nonInterval": {"category": "and", "signature": "(nonInterval[u_1,u_2,u_3,u_4,u_5,u_6,u_7,u_8])",
                    "elements": ["(limitSet[u_1,u_3,u_4,u_5,u_6])", "(in2[u_5,u_7,u_2])",
                                 "(in2[u_7,u_8,u_2])", "(in[u_8,u_4])", "!(in[u_7,u_4])"]},
}
ANCHORS = {"AnchorFTA": ("N", "i0", "s", "+", "*", "i1", "i2", "id"),
           "AnchorPeano": ("N", "i0", "s", "+", "*", "i1")}

PEANO = "(>[N,i0,s,+,*,i1](AnchorPeano[N,i0,s,+,*,i1])"
FTA = "(>[N,i0,s,+,*,i1,i2,id](AnchorFTA[N,i0,s,+,*,i1,i2,id])"


def renderer() -> vh.ReadableForm:
    return vh.ReadableForm(DEFINITIONS, ANCHORS)


class TokenTests(unittest.TestCase):
    def test_numerals_variables_carrier_multiplication_and_copies(self):
        self.assertEqual(vh.display_token("i2", vh._TEXT), "2")
        self.assertEqual(vh.display_token("v12", vh._TEXT), "v₁₂")
        self.assertEqual(vh.display_token("V3", vh._HTML), "V<sub>3</sub>")
        self.assertEqual(vh.display_token("N", vh._HTML), '<span class="bb-N">ℕ</span>')
        self.assertEqual(vh.display_token("*", vh._TEXT), "·")
        self.assertEqual(vh.display_token("v1_copy", vh._TEXT), "ṽ₁")
        self.assertEqual(vh.display_token("i0_copy", vh._TEXT), "0̃")
        self.assertEqual(vh.display_token("i12_copy", vh._TEXT), "1̃2̃")
        self.assertEqual(vh.display_token("rec", vh._TEXT), "rec")

    def test_spelling_of_both_forms(self):
        self.assertEqual(vh._HTML.text("a<b"), "a&lt;b")
        self.assertEqual(vh._TEXT.text("a<b"), "a<b")
        self.assertEqual(vh._TEXT.keyword("from", True), "from ")
        self.assertEqual(vh._TEXT.keyword("follows", False), " follows ")
        self.assertIn("rf-lead", vh._HTML.keyword("from", True))
        self.assertNotIn("rf-lead", vh._HTML.label("APPLIED:", False))
        self.assertEqual(vh._TEXT.big_operator("∑", "0", "n", "i"), "∑(i=0..n) i")
        self.assertIn("rf-big-sym", vh._HTML.big_operator("∏", "0", "n", "f(i)"))
        self.assertEqual(vh._TEXT.big_parentheses("x"), "(x)")
        product = vh._HTML.big_operator("∏", "0", "n", "f(i)")
        bracketed = vh._HTML.big_parentheses(product)
        self.assertTrue(bracketed.startswith('<span class="rf-unit"><span class="rf-bigparen">(</span>'
                                             '<span class="rf-big">'))
        self.assertTrue(bracketed.endswith(' f(i)<span class="rf-bigparen">)</span>'))
        self.assertEqual(bracketed.count('class="rf-big"'), 1)
        with self.assertRaises(AssertionError):
            vh._HTML.big_parentheses("x")
        self.assertEqual(vh._HTML.restriction("f", "≤ n"),
                         '<span class="rf-unit">f|<sub>≤ n</sub></span>')
        self.assertEqual(vh._TEXT.restriction("f", "≤ n"), "f|≤ n")
        self.assertEqual(vh._TEXT.indexed("w", "4", 0), "w₄")
        self.assertEqual(vh._TEXT.indexed("w", "4", 1), "w̃₄")
        self.assertEqual(vh._HTML.nowrap("a = b"), '<span class="rf-nb">a = b</span>')
        self.assertEqual(vh._TEXT.nowrap("a = b"), "a = b")
        self.assertEqual(vh._HTML.line_break(), "<br>")
        self.assertEqual(vh._TEXT.line_break(), "   ")

    def test_negation_keeps_its_sign_with_the_operand(self):
        implication = vh._implication_columns(["A", "B"], "C")
        negated = vh._HTML.negation(implication)
        self.assertTrue(negated.startswith('<span class="rf-negation">¬<span class="rf-parentheses">'))
        self.assertEqual(vh._leading_span_end(negated), len(negated))
        enclosed = vh._HTML.parentheses(implication)
        self.assertEqual(negated, '<span class="rf-negation">¬' + enclosed + "</span>")
        self.assertEqual(vh._HTML.negation(enclosed), negated)
        self.assertEqual(vh._HTML.negation("a = b"), "¬(a = b)")
        self.assertEqual(vh._TEXT.negation("a ⇒ b"), "¬(a ⇒ b)")

    def test_leading_span_end(self):
        fragment = '<span class="a"><span>x</span><svg></svg></span> rest'
        self.assertEqual(fragment[:vh._leading_span_end(fragment)],
                         '<span class="a"><span>x</span><svg></svg></span>')
        with self.assertRaises(AssertionError):
            vh._leading_span_end("x <span>y</span>")
        with self.assertRaisesRegex(AssertionError, "Unclosed"):
            vh._leading_span_end('<span><span>x</span>')

    def test_copy_mark(self):
        self.assertEqual(vh._copy_marked("1", 0), "1")
        self.assertEqual(vh._copy_marked("1", 1), "1̃")
        self.assertEqual(vh._copy_marked("12", 1), "1̃2̃")
        self.assertEqual(vh._copy_marked("v", 2), "ṽ̃")
        self.assertEqual(vh._TEXT.naturals(), "ℕ")
        self.assertEqual(vh._TEXT.label("RULE:", True), "RULE: ")


class TermTests(unittest.TestCase):
    def test_precedence_parentheses_and_large_operators(self):
        caption = vh._Caption({}, vh._TEXT)
        a, b, c = vh._Leaf("v1"), vh._Leaf("v2"), vh._Leaf("v3")
        self.assertEqual(caption.term(vh._Operation("+", vh._Operation("*", a, b), c)), "v₁ · v₂ + v₃")
        self.assertEqual(caption.term(vh._Operation("+", vh._Operation("+", a, b), c)), "(v₁ + v₂) + v₃")
        self.assertEqual(caption.term(vh._Operation("*", a, vh._Operation("+", b, c))), "v₁ · (v₂ + v₃)")
        self.assertEqual(caption.term(vh._Apply(vh._Leaf("s"), vh._Operation("+", a, b))), "s(v₁ + v₂)")
        big = vh._Big("*", "V1", vh._Leaf("i0"), vh._Apply(vh._Leaf("s"), a))
        self.assertEqual(caption.term(vh._Operation("*", big, b)), "(∏(i=0..s(v₁)) V₁(i)) · v₂")
        self.assertEqual(caption.term(vh._Operation("F", a, b)), "F(v₁, v₂)")
        self.assertEqual(caption.term(vh._Operation("div", a, b)), "v₁ div v₂")
        self.assertEqual(caption.operand(a, 2, last=True), "v₁")
        self.assertEqual(caption.operand(big, 2, last=True), "∏(i=0..s(v₁)) V₁(i)")
        self.assertEqual(caption.operand(big, 2, last=False), "(∏(i=0..s(v₁)) V₁(i))")
        self.assertEqual(caption.term(vh._Operation("*", b, big)), "v₂ · ∏(i=0..s(v₁)) V₁(i)")
        self.assertEqual(vh._term_precedence(big), 0)
        self.assertEqual(vh._term_precedence(vh._Operation("+", a, b)), 1)
        self.assertEqual(vh._term_precedence(a), 3)

    def test_tokens_and_substitution(self):
        term = vh._Operation("+", vh._Leaf("v1"), vh._Apply(vh._Leaf("s"), vh._Leaf("v2")))
        self.assertEqual(vh._term_tokens(term), {"+", "v1", "s", "v2"})
        replaced = vh._substitute_term(term, {"v2": vh._Leaf("v9")})
        self.assertEqual(replaced, vh._Operation("+", vh._Leaf("v1"), vh._Apply(vh._Leaf("s"), vh._Leaf("v9"))))
        fold = vh._Fold((vh._Leaf("N"), vh._Leaf("v1")))
        self.assertEqual(vh._term_tokens(fold), {"N", "v1"})
        self.assertEqual(vh._substitute_term(fold, {"v1": vh._Leaf("v2")}).slots[1], vh._Leaf("v2"))

    def test_fold_values(self):
        self.assertEqual(vh._fold_value(("N", "s", "+", "+", "id", "i0", "v1", "v2")),
                         vh._Big("+", "id", vh._Leaf("i0"), vh._Leaf("v1")))
        self.assertIsInstance(vh._fold_value(("N", "s", "*", "*", "f", "n", "m", "p")), vh._Fold)
        with self.assertRaises(AssertionError):
            vh._fold_value(("N", "s", "+", "id", "n", "m", "p"))

    def test_output_definitions(self):
        self.assertEqual(vh._output_definition(parse_mpl("(in2[v1,v2,s])")),
                         ("v2", vh._Apply(vh._Leaf("s"), vh._Leaf("v1"))))
        self.assertEqual(vh._output_definition(parse_mpl("(in3[v1,v2,v3,*])"))[0], "v3")
        self.assertEqual(vh._output_definition(parse_mpl("(remainder[N,+,*,v1,v2,v3])")),
                         ("v3", vh._Operation("mod", vh._Leaf("v1"), vh._Leaf("v2"))))
        self.assertEqual(vh._output_definition(parse_mpl("(fold[N,s,+,*,V1,i0,v1,v2])"))[0], "v2")
        self.assertIsNone(vh._output_definition(parse_mpl("(in[v1,N])")))


class ChainTests(unittest.TestCase):
    def test_flatten_drops_the_anchor_and_merges_repeated_binders(self):
        chain = vh._flatten_chain(parse_mpl(PEANO + "(>[w1](in[w1,N])(>[w1](in2[w1,w1,s])(=[w1,i0]))))"))
        self.assertEqual(chain.bound, ("w1",))
        self.assertEqual(len(chain.premises), 2)
        self.assertIn("i0", chain.anchor_slots)
        self.assertTrue(vh._is_anchor(parse_mpl("(AnchorPeano[N,i0,s,+,*,i1])")))
        self.assertFalse(vh._is_anchor(parse_mpl("(in[v1,N])")))

    def test_conjunction_atom_and_binder_walks(self):
        node = parse_mpl("(&(&(in[v1,N])(in[v2,N]))!(>[v3](in[v3,N])(=[v3,v1])))")
        self.assertEqual(len(vh._conjunction_parts(node)), 3)
        self.assertEqual([atom.head for atom in vh._atoms_of(node)], ["in", "in", "in", "="])
        self.assertEqual(list(vh._binders_of(node)), [("v3",)])
        self.assertEqual(vh._element_slots(parse_mpl("(in3[a,b,c,+])")), (0, 1, 2))
        self.assertEqual(vh._element_slots(parse_mpl("(fXY[f,X,Y])")), ())

    def test_definitions_drop_duplicates_unused_typing_only_and_collapsing_ones(self):
        def kept(expression):
            return set(vh._definitions(vh._flatten_chain(parse_mpl(expression)), DEFINITIONS, ANCHORS, {}))
        self.assertEqual(kept("(>[v1,v2,v3](in3[v1,v2,v3,+])(in3[v2,v1,v3,+]))"), {"v3"})
        self.assertEqual(kept("(>[w1](in[w1,N])(>[w2](in2[w1,w2,s])(>[w3](in2[w1,w3,s])(=[w2,w3]))))"), set())
        self.assertEqual(kept("(>[w1,w2](in2[w1,w2,s])(in[w1,N]))"), set())
        self.assertEqual(kept("(>[w1,w2,w3](in3[w1,w2,w3,+])(in[w3,N]))"), set())
        self.assertEqual(kept("(>[w1,w2](in2[w1,w2,V6])(or8[w1,w2,V6,v3,v2]))"), set())
        self.assertEqual(kept("(>[v1,v2](in3[v1,v2,v1,+])(=[v1,v2]))"), set())

    def test_cycle_resolution_and_consumed_premises(self):
        kept = {"v3": (0, vh._Operation("+", vh._Leaf("v1"), vh._Leaf("v2")))}
        self.assertTrue(vh._reaches({"v3"}, "v3", kept))
        self.assertTrue(vh._reaches({"v4"}, "v4", {}))
        self.assertFalse(vh._reaches({"v1"}, "v3", kept))
        resolved = vh._resolve({"v2": (0, vh._Apply(vh._Leaf("s"), vh._Leaf("v1"))),
                                "v3": (1, vh._Operation("+", vh._Leaf("v2"), vh._Leaf("v2")))}, {})
        self.assertEqual(resolved["v3"], vh._Operation("+", resolved["v2"], resolved["v2"]))
        chain = vh._flatten_chain(parse_mpl("(>[v1,v2](in2[v1,v2,s])(>[v3](in2[v1,v2,s])(in[v2,N])))"))
        self.assertEqual(vh._defining_premises(chain, {"v2": vh._Leaf("x")}), {0})


class TheoremCaptionTests(unittest.TestCase):
    def setUp(self):
        self.readable = renderer()

    def caption(self, expression):
        return self.readable.theorem(expression, as_html=False)

    def test_closed_forms(self):
        self.assertEqual(self.caption(PEANO + "(>[v1,v2,v3](in3[v1,v2,v3,+])(in3[v2,v1,v3,+])))"),
                         "v₁ + v₂ = v₂ + v₁")
        self.assertEqual(self.caption("(>[v1](in[v1,N])(>[v2](in3[v1,i0,v2,+])(=[v1,v2])))"),
                         "from v₁ ∈ ℕ follows v₁ + 0 = v₁")

    def test_f3_product_step_keeps_its_sequence_condition(self):
        self.assertEqual(
            self.caption(FTA + "(>[v1,v2](in2[v1,v2,s])(>[V1](sequence[N,+,i0,v2,V1])(>[v3](in2[v2,v3,V1])"
                         "(>[v4](fold[N,s,+,*,V1,i0,v1,v4])(>[v5](in3[v4,v3,v5,*])"
                         "(fold[N,s,+,*,V1,i0,v2,v5])))))))"),
            "from V₁: [0, s(v₁)] → ℕ follows (∏(i=0..v₁) V₁(i)) · V₁(s(v₁)) = ∏(i=0..s(v₁)) V₁(i)")

    def test_second_definition_of_a_variable_becomes_an_equation(self):
        self.assertEqual(
            self.caption("(>[v1,v2,v3](in3[v1,v2,v3,*])(>[v4,v5](in3[v3,v4,v5,+])(>[v6](in3[v1,v6,v5,*])"
                         "(>[v7](in3[v2,v7,v6,+])(in3[v1,v7,v4,*])))))"),
            "from v₁ · (v₂ + v₇) = v₁ · v₂ + v₄ follows v₁ · v₇ = v₄")

    def test_explicit_premises_where_substitution_would_mislead(self):
        self.assertEqual(self.caption("(>[w1](in[w1,N])(>[w2](in2[w1,w2,s])(>[w3](in2[w1,w3,s])(=[w2,w3]))))"),
                         "from w₁ ∈ ℕ, s(w₁) = w₂, s(w₁) = w₃ follows w₂ = w₃")
        self.assertEqual(self.caption("(>[w1,w2](in2[w1,w2,s])(in[w1,N]))"), "from s(w₁) = w₂ follows w₁ ∈ ℕ")
        self.assertEqual(self.caption("(>[w1,w2,w3](in3[w1,w2,w3,+])(and22[w1,N,w2,N,w3,N]))"),
                         "from w₁ + w₂ = w₃ follows w₁ ∈ ℕ ∧ w₂ ∈ ℕ ∧ w₃ ∈ ℕ")
        self.assertEqual(self.caption("(>[w1,w2](in2[w1,w2,V6])(or8[w1,w2,V6,v3,v2]))"),
                         "from V₆(w₁) = w₂ follows V₆(w₁) = w₂ ∨ (w₂ = v₃ ∧ w₁ = v₂)")

    def test_compiled_names_expand(self):
        self.assertEqual(self.caption("(>[v1](in[v1,N])(or1[N,v1,s,i0]))"),
                         "from v₁ ∈ ℕ follows (∃ y₁ ∈ ℕ: s(y₁) = v₁) ∨ v₁ = 0")
        self.assertEqual(self.caption("(>[v1](in[v1,N])(>[]!(existence3[N,v1,s])(=[v1,i0])))"),
                         "from v₁ ∈ ℕ, (∄ y₁ ∈ ℕ: s(y₁) = v₁) follows v₁ = 0")
        self.assertEqual(self.caption("(>[v1,V1,V2](limitSequence[N,+,v1,V1,V2])(existence22[N,+,v1,V1,V2]))"),
                         "from V₂ = V₁|≤ v₁ follows ∃ Y₁: Y₁ = V₁|≤ v₁ ∧ V₂ = Y₁")
        self.assertEqual(self.caption("(>[](implication35[V2,V3])(>[](implication35[V3,V2])(=[V2,V3])))"),
                         "from V₂ ⊆ V₃, V₃ ⊆ V₂ follows V₂ = V₃")
        self.assertEqual(self.caption("(existence3[N,i1,s])"), "∃ y₁ ∈ ℕ: s(y₁) = 1")

    def test_substitution_keeps_premises_with_content(self):
        cases = {
            # A relation's argument without another premise: the subset relation.
            "(>[V2,V3](implication35[V2,V3])(=[V2,V3]))": "from V₂ ⊆ V₃ follows V₂ = V₃",
            # Identity: the premise carries the domain the head types.
            "(>[w1,w2](in2[w1,w2,id])(&(in[w1,N])(=[w1,w2])))":
                "from id(w₁) = w₂ follows w₁ ∈ ℕ ∧ w₁ = w₂",
            # A second application of s to w1: no tautology.
            "(>[w1](in[w1,N])(>[w2](in2[w1,w2,s])(>[](in2[w1,w1,s])(=[w2,w1]))))":
                "from w₁ ∈ ℕ, s(w₁) = w₂, s(w₁) = w₁ follows w₂ = w₁",
            # A defined variable typed in ℕ stays a variable.
            "(>[v1](in[v1,N])(>[v2](in3[i1,v2,v1,*])(=[v1,v2])))":
                "from v₁ ∈ ℕ, 1 · v₂ = v₁ follows v₁ = v₂",
            # The remainder presupposes a nonzero divisor.
            "(>[v3](remainder[N,+,*,v1,v2,v3])(>[](preorder[N,*,v2,v1])(=[v3,i0])))":
                "from v₁ mod v₂ = v₃, v₂ ∣ v₁ follows v₃ = 0",
            # A product over an unrestricted function keeps its existence premise.
            "(>[p](fold[N,s,+,*,f,n,m,p])(>[r](in3[p,k,r,+])(=[r,q])))":
                "from ∏(i=n..m) f(i) = p follows p + k = q",
            # A sequence statement restricts the function: closed form.
            "(>[f](sequence[N,+,n,m,f])(>[p](fold[N,s,+,*,f,n,m,p])(>[r](in3[p,k,r,+])(=[r,q]))))":
                "from f: [n, m] → ℕ follows (∏(i=n..m) f(i)) + k = q",
            # A sum as the last operand needs no parentheses.
            "(>[v2](fold[N,s,+,+,id,i0,v1,v2])(>[v3](in3[i2,v2,v3,*])(=[v3,v4])))":
                "2 · ∑(i=0..v₁) i = v₄",
        }
        for expression, expected in cases.items():
            with self.subTest(expression=expression):
                self.assertEqual(self.caption(expression), expected)

    def test_user_definitions_unknown_predicates_anchors_and_html(self):
        self.assertEqual(self.caption("(>[v1,v2,v3](residual[v1,v2,v3,+])(in[v3,N]))"),
                         "from v₃ + v₂ = v₁ follows v₃ ∈ ℕ")
        with self.assertRaises(AssertionError):
            self.caption("(>[v1](myPredicate[v1,N])(in[v1,N]))")
        self.assertEqual(self.caption("(AnchorPeano[N,i0,s,+,*,i1])"), "Peano anchor (ℕ, 0, s, +, ·, 1)")
        html = self.readable.theorem("(>[v1](in[v1,N])(strictOrder[N,+,i0,v1]))")
        self.assertIn('<span class="rf-kw rf-lead">from</span>', html)
        self.assertIn("0 &lt; v<sub>1</sub>", html)

    def test_missing_compiled_definition_asserts(self):
        with self.assertRaises(AssertionError):
            vh.ReadableForm({}, ANCHORS).theorem("(>[v1](in[v1,N])(or1[N,v1,s,i0]))")


class RowCaptionTests(unittest.TestCase):
    def setUp(self):
        self.readable = renderer()

    def test_rule_and_its_application_in_rule_order(self):
        caption = self.readable.rule_application(
            "(in3[v1,v2,v8,+])",
            PEANO + "(>[w1,w2,w3](in3[w1,w2,w3,+])(>[w4,w5](in3[w4,w5,w1,+])(>[w6](in3[w5,w2,w6,+])"
            "(in3[w4,w6,w3,+])))))",
            ["(AnchorPeano[N,i0,s,+,*,i1])", "(in3[v1,v13,v14,+])", "(in3[v13,v7,v2,+])",
             "(in3[v14,v7,v8,+])"], as_html=False)
        self.assertEqual(caption, "RULE: (w₄ + w₅) + w₂ = w₄ + (w₅ + w₂)   "
                                  "APPLIED WITH: w₂ = v₇, w₄ = v₁, w₅ = v₁₃")
        html_caption = self.readable.rule_application(
            "(in3[v1,v2,v8,+])",
            PEANO + "(>[w1,w2,w3](in3[w1,w2,w3,+])(>[w4,w5](in3[w4,w5,w1,+])(>[w6](in3[w5,w2,w6,+])"
            "(in3[w4,w6,w3,+])))))",
            ["(AnchorPeano[N,i0,s,+,*,i1])", "(in3[v1,v13,v14,+])", "(in3[v13,v7,v2,+])",
             "(in3[v14,v7,v8,+])"])
        self.assertIn("<br>", html_caption)
        self.assertIn('<span class="rf-unit">w<sub>2</sub> = v<sub>7</sub></span>', html_caption)

    def test_rule_without_bound_variables_is_shown_once(self):
        self.assertEqual(self.readable.rule_application("(=[v2,i1])", "(>[](=[i0,i0])(=[v2,i1]))",
                                                        ["(=[i0,i0])"], as_html=False),
                         "RULE: from 0 = 0 follows v₂ = 1")

    def test_premises_that_do_not_instantiate_the_rule_assert(self):
        with self.assertRaises(AssertionError):
            self.readable.rule_application("(in[v1,N])", "(>[w1,w2](in2[w1,w2,s])(in[w1,N]))",
                                           ["(in2[v2,v3,s])"])

    def test_rule_matching(self):
        rule = parse_mpl("(>[w1,w2](in2[w1,w2,s])(>[w3](in2[w2,w3,s])(in3[w1,i2,w3,+])))")
        premises = [parse_mpl("(in2[v5,v6,s])"), parse_mpl("(in2[v4,v5,s])")]
        self.assertEqual(vh._match_rule(rule, premises, parse_mpl("(in3[v4,i2,v6,+])")), [1, 0])
        order, binding = vh._rule_binding(rule, premises, parse_mpl("(in3[v4,i2,v6,+])"))
        self.assertEqual(order, [1, 0])
        self.assertEqual(binding, {"w1": "v4", "w2": "v5", "w3": "v6"})
        self.assertIsNone(vh._rule_binding(rule, premises, parse_mpl("(in3[v4,i1,v6,+])")))
        self.assertIsNone(vh._match_rule(rule, premises, parse_mpl("(in3[v4,i1,v6,+])")))
        self.assertIsNone(vh._match_rule(rule, premises[:1], parse_mpl("(in3[v4,i2,v6,+])")))
        sigma = {}
        self.assertTrue(vh._unify(parse_mpl("!(>[w1](in[w1,N])(&(=[w1,v1])(=[v1,w1])))"),
                                  parse_mpl("!(>[v9](in[v9,N])(&(=[v9,v1])(=[v1,v9])))"), {"w1"}, sigma))
        self.assertEqual(sigma, {"w1": "v9"})
        self.assertFalse(vh._unify_token("v1", "v2", {"w1"}, {}))
        self.assertTrue(vh._unify_token("w1", "v2", {"w1"}, {"w1": "v2"}))

    def test_readings(self):
        self.assertEqual(self.readable.reading("(in[v2,V5])"), "")
        self.assertEqual(self.readable.reading("!(in2[v1,v2,s])"), "")
        self.assertEqual(self.readable.reading("(AnchorFTA[N,i0,s,+,*,i1,i2,id])"), "")
        self.assertEqual(self.readable.reading("(implication33[V7,s,V6,V1,*])", as_html=False),
                         "READS: ∀ y₁ ∈ V₇: s(y₁) ∈ V₇ ⇒ V₆(y₁) · V₁(s(y₁)) = V₆(s(y₁))")
        self.assertEqual(self.readable.reading("(implication2525[])", as_html=False),
                         "READS: ∀ y₁ ∈ ℕ, y₂ ∈ ℕ: y₁ < y₂ ∨ y₁ = y₂ ∨ y₂ < y₁")
        self.assertEqual(self.readable.reading("!(>[V5](interval[N,+,i0,v2,V5])!(fXY[V1,V5,N]))", as_html=False),
                         "READS: ∃ V₅: V₅ = [0, v₂] ∧ (V₁: V₅ → ℕ)")
        self.assertEqual(self.readable.reading("!(&!(=[i0,v8])!(preorder[N,+,i1,v8]))", as_html=False),
                         "READS: 0 = v₈ ∨ 1 ≤ v₈")

    def test_definition_templates(self):
        cases = {
            "(appendToSequence[v2,v3,V6,V9])": "V₉ = V₆ ∪ {v₂ ↦ v₃}",
            "(addToSet[V1,v2,V2])": "V₂ = V₁ ∪ {v₂}",
            "(limitSet[N,+,V1,v2,V2])": "V₂ = V₁ ∩ [0, v₂]",
            "(quotient[N,+,*,v1,v2,v3])": "v₁ div v₂ = v₃",
            "!(prime[N,+,*,i1,i2,v1])": "v₁ is not prime",
            "(EnumerationSet2[i0,i1,V1])": "V₁ = {0, 1}",
            "(EnumerationRelationSet1[i0,v1,V1])": "V₁ = {0 ↦ v₁}",
            "(identity[N,id])": "id is the identity on ℕ",
            "!(identity[N,id])": "id is not the identity on ℕ",
            "(implication35[V2,V3])": "V₂ ⊆ V₃",
            "!(implication35[V2,V3])": "V₂ ⊈ V₃",
            "(or1[N,v1,s,i0])": "(∃ y₁ ∈ ℕ: s(y₁) = v₁) ∨ v₁ = 0",
            "!(&!(or15[N,+,v1,v2])!(=[v1,i0]))": "v₁ < v₂ ∨ v₁ = v₂ ∨ v₂ < v₁ ∨ v₁ = 0",
            "(fXYZ[+,N,N,N])": "+: ℕ × ℕ → ℕ",
            "(split[N,+,i0,v1,V1,V2])": "V₂ = V₁|[0, v₁)",
            "(nontrivialFactor[N,+,*,i2,v1,v2])": "v₂ is a nontrivial factor of v₁",
            "(NaturalNumbers[N,i0,s,+,*])": "Peano axioms (ℕ, 0, s, +, ·)",
            "(fold[N,s,+,+,id,i0,v1,v2])": "∑(i=0..v₁) i = v₂",
            "(fold[N,s,*,*,V1,i0,v1,v2])": "fold(ℕ, s, ·, ·, V₁, 0, v₁) = v₂",
            "(&(in[v1,N])(fXY[V1,V2,N]))": "v₁ ∈ ℕ ∧ (V₁: V₂ → ℕ)",
            "(&(strictOrder[N,*,v1,v2])(in[v1,N]))": "v₁ ∣ v₂ ∧ v₁ ≠ v₂ ∧ v₁ ∈ ℕ",
            "(residual[v1,v2,v3,+])": "v₃ + v₂ = v₁",
            "(infiniteSequence[V1,N])": "V₁: ℕ → ℕ",
            "(nonInterval[N,s,+,V1,v1,V2,v2,v3])":
                "V₂ = V₁ ∩ [0, v₁] ∧ s(v₁) = v₂ ∉ V₁ ∧ s(v₂) = v₃ ∈ V₁",
            "!(nonInterval[N,s,+,V1,v1,V2,v2,v3])":
                "¬(V₂ = V₁ ∩ [0, v₁] ∧ s(v₁) = v₂ ∉ V₁ ∧ s(v₂) = v₃ ∈ V₁)",
        }
        for expression, expected in cases.items():
            with self.subTest(expression=expression):
                self.assertEqual(self.readable.reading(expression, as_html=False), "READS: " + expected)

    def test_mapping_pairs_and_intervals_are_unbreakable_units(self):
        units = {
            "(appendToSequence[v2,v3,V6,V9])": ['v<sub>2</sub> ↦ v<sub>3</sub>'],
            "(EnumerationRelationSet2[i0,v1,i1,v2,V1])": ['0 ↦ v<sub>1</sub>', '1 ↦ v<sub>2</sub>'],
            "(interval[N,+,i0,v2,V5])": ['[0, v<sub>2</sub>]'],
            "(split[N,+,i0,v1,V1,V2])": ['V<sub>1</sub>|<sub>[0, v<sub>1</sub>)</sub>'],
            "(limitSet[N,+,V1,v2,V2])": ['[0, v<sub>2</sub>]'],
            "(EnumerationSet2[i0,i1,V1])": ['{0, 1}'],
        }
        for expression, expected in units.items():
            with self.subTest(expression=expression):
                html = self.readable.reading(expression)
                for unit in expected:
                    self.assertIn(f'<span class="rf-unit">{unit}</span>', html)
        self.assertEqual(vh._HTML.unit("a ↦ b"), '<span class="rf-unit">a ↦ b</span>')
        self.assertEqual(vh._TEXT.unit("a ↦ b"), "a ↦ b")

    def test_template_predicates_with_another_argument_count_assert(self):
        for expression in ("(interval[N,+,v1,V1])", "(in2[v1,v2])", "!(prime[N,+,*,i1,v1])",
                           "(sequence[N,+,i0,v1])"):
            with self.subTest(expression=expression):
                with self.assertRaises(AssertionError):
                    self.readable.reading(expression)

    def test_reformulation_shows_both_forms_literally(self):
        self.assertEqual(self.readable.reformulation("(in3[i1,i1,i1,*])", "(>[w1](in3[i1,i1,w1,*])(=[w1,i1]))",
                                                     "BACK-REFORMULATED FROM:", as_html=False),
                         "1 · 1 = 1   BACK-REFORMULATED FROM: from 1 · 1 = w₁ follows w₁ = 1")


class CompiledDefinitionTests(unittest.TestCase):
    def test_instantiation_names_and_anchor_slots(self):
        caption = vh._Caption(DEFINITIONS, vh._TEXT)
        category, parts, anchor, names = vh._instantiate(
            DEFINITIONS["existence3"], parse_mpl("(existence3[N,v1,s])"),
            lambda _digit, element: caption.fresh(element), ANCHORS)
        self.assertEqual(category, "existence")
        self.assertEqual(anchor, [])
        self.assertEqual(caption.token(names[0]), "y₁")
        self.assertEqual(parts[1], Atom("in2", (names[0], "v1", "s")))
        _category, parts, anchor, names = vh._instantiate(
            DEFINITIONS["implication2525"], parse_mpl("(implication2525[])"),
            lambda digit, _element: "x" + digit, ANCHORS)
        self.assertEqual(anchor, list(ANCHORS["AnchorFTA"]))
        self.assertEqual(names, ["x9", "x10"])
        with self.assertRaises(AssertionError):
            vh._instantiate(DEFINITIONS["implication2525"], parse_mpl("(implication2525[])"),
                            lambda digit, _element: digit, {})
        with self.assertRaises(AssertionError):
            vh._instantiate(DEFINITIONS["existence3"], parse_mpl("(existence3[N,v1])"),
                            lambda digit, _element: digit, ANCHORS)

    def test_expanded_atoms_follow_compiled_names_and_signs(self):
        found = list(vh._expanded_atoms(parse_mpl("!(or8[v1,v2,V6,v3,v4])"), DEFINITIONS, ANCHORS))
        self.assertEqual([(atom.head, positive) for atom, positive in found],
                         [("in2", False), ("=", False), ("=", False)])
        implied = list(vh._expanded_atoms(parse_mpl("(>[](in[v1,N])(=[v1,v2]))"), DEFINITIONS, ANCHORS))
        self.assertEqual([positive for _atom, positive in implied], [False, True])

    def test_fresh_names_are_numbered_in_order_of_display(self):
        caption = vh._Caption(DEFINITIONS, vh._TEXT)
        first, second, third = caption.fresh(True), caption.fresh(False), caption.fresh(True)
        self.assertEqual(caption.token(third), "y₁")
        self.assertEqual(caption.token(first), "y₂")
        self.assertEqual(caption.token(second), "Y₁")
        self.assertEqual(caption.token(third), "y₁")

    def test_load_compiled_definitions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = {"category": "or", "signature": "(or1[u_1,u_2])", "elements": ["(in[u_1,u_2])"],
                     "implications": []}
            atomic = {"category": "atomic", "signature": "(in[u_1,u_2])", "elements": [""]}
            (root / "GL_binary_shared.json").write_text(
                json.dumps({"or1": entry, "residual": dict(entry, category="and"), "in": atomic}),
                encoding="utf-8")
            batch = dict(entry, elements=["(=[u_1,u_2])"])
            (root / "GL_binary_B.json").write_text(json.dumps({"or1": batch, "or2": batch}),
                                                   encoding="utf-8")
            loaded = vh.load_compiled_definitions(root)
            self.assertEqual(set(loaded), {"or1", "residual"})
            self.assertEqual(loaded["or1"]["elements"], ["(in[u_1,u_2])"])
            self.assertEqual(vh.load_compiled_definitions(root / "missing"), {})

    def test_read_shared_definition_registry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stored = {"in": {"category": "atomic", "signature": "(in[u_1,u_2])", "elements": [""]},
                      "or1": {"category": "or", "signature": "(or1[u_1])", "elements": ["(in[u_1,u_1])"]}}
            (root / "GL_binary_Peano.json").write_text(json.dumps({"or9": stored["or1"]}), encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "missing shared definition registry"):
                vh.read_shared_definition_registry(root)
            (root / vh.SHARED_DEFINITION_REGISTRY).write_text(json.dumps(stored), encoding="utf-8")
            self.assertEqual(vh.read_shared_definition_registry(root), stored)
            self.assertEqual(vh.read_shared_definition_registry(root / "missing"), {})


class HelperTests(unittest.TestCase):
    def chain(self, expression):
        return vh._flatten_chain(parse_mpl(expression))

    def test_substitutable(self):
        chain = self.chain("(>[v1,v2](in2[v1,v2,s])(>[V1](sequence[N,+,i0,v2,V1])(>[v3](in2[v2,v3,V1])"
                           "(>[v4](fold[N,s,+,*,V1,i0,v1,v4])(>[v5](remainder[N,+,*,v4,v3,v5])(=[v5,i0]))))))")
        self.assertTrue(vh._substitutable(chain, 0))      # s(v1): total
        self.assertTrue(vh._substitutable(chain, 2))      # V1(v2): v2 restricted
        self.assertTrue(vh._substitutable(chain, 3))      # product over V1: sequence premise
        self.assertFalse(vh._substitutable(chain, 4))     # remainder: never
        loose = self.chain("(>[w1,w2](in2[w1,w2,V2])(in2[w1,w2,V3]))")
        self.assertFalse(vh._substitutable(loose, 0))
        typed = self.chain("(>[v1](in[v1,N])(>[v2](in3[i1,v2,v1,*])(=[v1,v2])))")
        self.assertFalse(vh._substitutable(typed, 1))
        twice = self.chain("(>[w1](in[w1,N])(>[w2](in2[w1,w2,s])(>[](in2[w1,w1,s])(=[w2,w1]))))")
        self.assertFalse(vh._substitutable(twice, 1))
        with self.assertRaises(AssertionError):
            vh._substitutable(typed, 0)

    def test_restricted_and_same_application(self):
        chain = self.chain("(>[v1,v2](in[v1,V7])(>[](in2[v1,v2,V6])(=[v2,v2])))")
        self.assertTrue(vh._restricted("v1", chain, 1))
        self.assertFalse(vh._restricted("v2", chain, 1))
        self.assertTrue(vh._same_application(Atom("in2", ("a", "b", "F")), Atom("in2", ("a", "c", "F"))))
        self.assertFalse(vh._same_application(Atom("in2", ("a", "b", "F")), Atom("in2", ("b", "c", "F"))))
        self.assertTrue(vh._same_application(Atom("in3", ("a", "b", "c", "+")), Atom("in3", ("a", "b", "d", "+"))))
        self.assertFalse(vh._same_application(Atom("in3", ("a", "b", "c", "+")), Atom("in3", ("b", "a", "c", "+"))))
        self.assertFalse(vh._same_application(Atom("=", ("a", "b")), Atom("=", ("a", "b"))))

    def test_inclusion(self):
        self.assertEqual(vh._inclusion(self.chain("(>[w1,w2](in2[w1,w2,V2])(in2[w1,w2,V3]))")), ("V2", "V3"))
        self.assertIsNone(vh._inclusion(self.chain("(>[w1,w2](in2[w1,w2,V2])(in2[w2,w1,V3]))")))
        self.assertIsNone(vh._inclusion(self.chain("(>[w1](in2[w1,v2,V2])(in2[w1,v2,V3]))")))
        self.assertIsNone(vh._inclusion(self.chain("(>[w1,w2](in2[w1,w2,V2])(in2[w1,w2,V2]))")))
        self.assertIsNone(vh._inclusion(self.chain("(>[w1,w2](in[w1,N])(>[](in2[w1,w2,V2])(in2[w1,w2,V3])))")))

    def test_has_template_and_element_slots(self):
        for head in ("=", "in2", "fold", "interval", "prime", "NaturalNumbers", "AnchorFTA",
                     "EnumerationSet3", "EnumerationRelationSet1"):
            self.assertTrue(vh._has_template(head), head)
        for head in ("or1", "implication33", "existence3", "and27"):
            self.assertFalse(vh._has_template(head), head)
        for head in vh._NAMED_TEMPLATES:
            self.assertTrue(vh._has_template(head), head)
        self.assertEqual(vh._element_slots(Atom("preorder", ("N", "+", "1", "2"))), (2, 3))
        self.assertEqual(vh._element_slots(Atom("fold", tuple("NsxyFabc"))), (5, 6, 7))
        self.assertEqual(vh._element_slots(Atom("fXY", ("1", "X", "Y"))), ())

    def test_formula_reads_plain_relations_too(self):
        readable = renderer()
        self.assertEqual(readable.formula("(in3[i2,v4,v3,*])", as_html=False), "2 · v₄ = v₃")
        self.assertEqual(readable.formula("(or1[N,v1,s,i0])", as_html=False),
                         "(∃ y₁ ∈ ℕ: s(y₁) = v₁) ∨ v₁ = 0")

    def test_check_arity(self):
        vh._check_arity(parse_mpl("(in3[v1,v2,v3,+])"))
        vh._check_arity(parse_mpl("(EnumerationSet3[i0,i1,i2,V1])"))
        vh._check_arity(parse_mpl("(residual[v1,v2,v3,+])"))
        with self.assertRaises(AssertionError):
            vh._check_arity(parse_mpl("(in3[v1,v2,v3])"))
        with self.assertRaises(AssertionError):
            vh._check_arity(parse_mpl("(limitSet[N,+,V1,v2])"))

    def test_order_relations(self):
        def term(token):
            return vh.display_token(token, vh._TEXT)
        self.assertEqual(vh._order_relation("preorder", ("N", "+", "v1", "v2"), False, term, vh._TEXT)[0], "v₁ ≤ v₂")
        self.assertEqual(vh._order_relation("preorder", ("N", "*", "v1", "v2"), True, term, vh._TEXT)[0], "v₁ ∤ v₂")
        self.assertEqual(vh._order_relation("strictOrder", ("N", "+", "v1", "v2"), True, term, vh._TEXT)[0],
                         "v₁ ≮ v₂")
        self.assertEqual(vh._order_relation("strictOrder", ("N", "*", "v1", "v2"), True, term, vh._TEXT)[0],
                         "¬(v₁ ∣ v₂ ∧ v₁ ≠ v₂)")
        self.assertIn("&lt;", vh._order_relation("strictOrder", ("N", "+", "v1", "v2"), False, term, vh._HTML)[0])
        with self.assertRaises(AssertionError):
            vh._order_relation("preorder", ("N", "s", "v1", "v2"), False, term, vh._TEXT)

    def test_function_types(self):
        def term(token):
            return vh.display_token(token, vh._TEXT)
        self.assertEqual(vh._function_type("fXY", ("f", "X", "N"), term, vh._TEXT), "f: X → ℕ")
        self.assertEqual(vh._function_type("sequence", ("N", "+", "i0", "v1", "V1"), term, vh._TEXT),
                         "V₁: [0, v₁] → ℕ")
        self.assertEqual(vh._function_type("sequence", ("N", "+", "i0", "v1", "V1"), term, vh._HTML),
                         'V₁: <span class="rf-unit">[0, v₁]</span> → ℕ')
        with self.assertRaises(AssertionError):
            vh._function_type("fXY", ("f", "X"), term, vh._TEXT)

    def test_formula_strengths(self):
        readable = renderer()
        caption = vh._Caption(DEFINITIONS, vh._TEXT)
        self.assertEqual(readable._formula(parse_mpl("(in[v1,N])"), {}, caption), ("v₁ ∈ ℕ", vh._RELATION))
        self.assertEqual(readable._formula_in(parse_mpl("(fXY[V1,V2,N])"), {}, caption, vh._AND), "(V₁: V₂ → ℕ)")
        self.assertEqual(readable._formula(parse_mpl("!!(in[v1,N])"), {}, caption)[0], "v₁ ∈ ℕ")
        self.assertEqual(readable._formula(parse_mpl("!(&(in[v1,N])(in[v2,N]))"), {}, caption)[0],
                         "¬(v₁ ∈ ℕ ∧ v₂ ∈ ℕ)")
        existence = vh._Existence(("v9",), (parse_mpl("(in[v9,N])"), parse_mpl("(=[v9,v1])")))
        self.assertEqual(readable._existence(existence, True, {}, caption), ("∄ v₉ ∈ ℕ: v₉ = v₁", vh._QUANTIFIED))
        self.assertEqual(readable._atom(parse_mpl("(in2[v1,v2,s])"), True, {}, caption)[0], "s(v₁) ≠ v₂")
        self.assertEqual(readable._compiled(parse_mpl("(and27[v1,v2,v3,v4])"), True, {}, caption)[0],
                         "¬(v₁ = v₂ ∧ v₃ = v₄)")
        chain = vh._flatten_chain(parse_mpl("(>[v1](in2[v1,v2,s])(=[v2,v3]))"))
        self.assertEqual(readable._head(chain.head, {"v2": vh._Leaf("v7")}, {"v2": vh._Leaf("v7")}, caption),
                         "v₇ = v₃")
        self.assertEqual(readable._head_with_strength(parse_mpl("(=[v3,v2])"), {"v2": vh._Leaf("v7")},
                                                      {"v2": vh._Leaf("v7")}, caption), ("v₇ = v₃", vh._RELATION))
        self.assertEqual(readable._chain(chain, caption, substitute=False, nested=True)[0], "∀ v₁: s(v₁) = v₂ ⇒ v₂ = v₃")


class GeneratorCaptionTests(unittest.TestCase):
    """The caption hooks of `generate_full_proof_graph` around the renderer."""

    def setUp(self):
        import expression_utils
        import generate_full_proof_graph as gen
        from configuration_reader import configuration_reader
        self.gen = gen
        root = Path(__file__).resolve().parents[1]
        expression_utils.set_configuration(configuration_reader(root / "files/config/ConfigVisu.json"))
        self.saved = gen._readable_form_for_render
        gen._readable_form_for_render = renderer()

    def tearDown(self):
        self.gen._readable_form_for_render = self.saved

    def test_anchor_signatures_come_from_the_configuration(self):
        signatures = self.gen._anchor_signatures()
        self.assertEqual(signatures["AnchorFTA"], ANCHORS["AnchorFTA"])
        self.assertEqual(signatures["AnchorPeano"], ANCHORS["AnchorPeano"])
        self.assertTrue(all(name.startswith("Anchor") for name in signatures))

    def test_theorem_caption_in_both_spellings(self):
        theorem = PEANO + "(>[v1,v2,v3](in3[v1,v2,v3,+])(in3[v2,v1,v3,+])))"
        self.assertEqual(self.gen._readable_theorem(theorem, as_html=False), "v₁ + v₂ = v₂ + v₁")
        self.assertIn("v<sub>1</sub>", self.gen._readable_theorem(theorem))

    def test_row_captions_per_tag(self):
        rule = PEANO + "(>[w1,w2,w3](in3[w1,w2,w3,+])(in3[w2,w1,w3,+])))"
        implication = ["(in3[v2,v1,v3,+])", "main", "implication", rule, "main",
                       "(AnchorPeano[N,i0,s,+,*,i1])", "main", "(in3[v1,v2,v3,+])", "main"]
        self.assertIn("readable-grey implication", self.gen._row_caption_html(implication))
        self.assertIn("RULE:", self.gen._row_caption_html(implication))
        reformulated = ["(existence3[N,i1,s])", "main", "reformulated from", "(in2[i0,i1,s])", "main"]
        self.assertIn("REFORMULATED FROM:", self.gen._row_caption_html(reformulated))
        back = ["(in3[i1,i1,i1,*])", "main", "incubator back reformulation",
                "(>[w1](in3[i1,i1,w1,*])(=[w1,i1]))", "main"]
        self.assertIn("BACK-REFORMULATED FROM:", self.gen._row_caption_html(back))
        expansion = ["(implication33[V7,s,V6,V1,*])", "main", "disintegration", "(&(in[v1,N])(in[v2,N]))", "main"]
        self.assertIn("readable-grey reads", self.gen._row_caption_html(expansion))
        derived_reading = ["(existence3[N,v1,s])", "main", "implication",
                           "(>[w1](in2[i0,w1,s])(existence3[N,w1,s]))", "main", "(in2[i0,v1,s])", "main"]
        self.assertIn("∃", self.gen._row_caption_html(derived_reading))
        unfolded = ["(&(in[1,N])(in2[1,v1,s]))", "main", "expansion", "(existence3[N,v1,s])", "main"]
        self.assertIn("READS:", self.gen._row_caption_html(unfolded))
        repeated = ["(>[y1,y2](in2[y1,y2,V2])(in2[y1,y2,V3]))", "main", "expansion",
                    "(implication35[V2,V3])", "main"]
        self.assertEqual(self.gen._row_caption_html(repeated), "")
        anchor = ["(&(NaturalNumbers[N,i0,s,+,*])(in[i1,N]))", "main", "expansion",
                  "(AnchorPeano[N,i0,s,+,*,i1])", "main"]
        self.assertEqual(self.gen._row_caption_html(anchor), "")
        goal = ["(existence3[N,v1,s])_integration_goal", "main", "expansion for integration",
                "(or1[N,v1,s,i0])_integration_goal", "main"]
        self.assertIn("∃", self.gen._row_caption_html(goal))
        self.assertEqual(self.gen._row_caption_html(["(in[v1,N])", "main", "task formulation"]), "")
        self.assertEqual(self.gen._row_caption_html(["(in[v1,N])", "main", "validity name", "x", "main"]), "")

    def test_subproof_title_readings(self):
        proves = self.gen._subproof_title_reading(
            {"kind": "subproof", "implication_expr": "(>[v1](in[v1,N])(in2[v1,v2,s]))"})
        self.assertIn("PROVES:", proves)
        self.assertIn("follows", proves)
        case = self.gen._subproof_title_reading(
            {"kind": "_ordis_ branch", "disjunct": "(=[v1,i0])", "or_expr": "(or1[N,v1,s,i0])"})
        self.assertIn("CASE:", case)
        self.assertIn("OF:", case)
        self.assertIn("∃", case)
        asserts = self.gen._subproof_title_reading(
            {"kind": "_orint_ subproof", "disjunct": "(=[v1,i0])", "or_expr": "(or1[N,v1,s,i0])"})
        self.assertIn("ASSERTS:", asserts)
        self.assertEqual(self.gen._subproof_title_reading({"kind": "subproof"}), "")
        self.assertEqual(self.gen._subproof_title_reading({"kind": "other"}), "")

    def test_notation_key_and_zoom_in_description(self):
        self.assertIn('id="readable-notation"', self.gen._READABLE_NOTATION_HTML)
        self.assertIn("APPLIED WITH", self.gen._READABLE_NOTATION_HTML)
        self.assertIn("zoom in", self.gen.TAG_DESCRIPTIONS)


# Numbered definitions that name a named definition inside: their unfolding
# must stop at the name.
NAMED_INSIDE = {
    "existence40": {"category": "existence", "signature": "(existence40[u_1])",
                    "elements": ["(in[1,u_1])", "(hasPrimeFactorization[u_1,s,+,*,i0,i1,i2,1])"]},
    "and41": {"category": "and", "signature": "(and41[u_1,u_2,u_3])",
              "elements": ["(ordered[u_1,+,u_2])", "(inSequence[u_1,u_3,u_2])"]},
    "myNamed": {"category": "and", "signature": "(myNamed[u_1,u_2])",
                "elements": ["(in[u_1,u_2])"]},
}


class NamedTemplateTests(unittest.TestCase):
    """Every named definition reads through its own template, never unfolded."""

    def setUp(self):
        self.readable = vh.ReadableForm(dict(DEFINITIONS, **NAMED_INSIDE), ANCHORS)

    def read(self, expression, as_html=False):
        return self.readable.formula(expression, as_html=as_html)

    def test_number_theory_and_set_templates(self):
        cases = {
            "(subset[V1,N])": "V₁ ⊆ ℕ",
            "!(subset[V1,V2])": "V₁ ⊈ V₂",
            "(empty[V1])": "V₁ = ∅",
            "!(empty[V1])": "V₁ ≠ ∅",
            "(minimum[N,+,V1,v1])": "v₁ = min V₁",
            "!(minimum[N,+,V1,v1])": "v₁ ≠ min V₁",
            "(hasMinimum[N,+,V1])": "V₁ has a minimum",
            "!(hasMinimum[N,+,V1])": "V₁ has no minimum",
            "(divisorSet[N,+,*,i2,v1,V1])": "V₁ = {d ∈ ℕ : 2 ≤ d ≤ v₁, d ∣ v₁}",
            "!(divisorSet[N,+,*,i2,v1,V1])": "V₁ ≠ {d ∈ ℕ : 2 ≤ d ≤ v₁, d ∣ v₁}",
            "(coprime[N,i0,s,*,v1,v2])": "gcd(v₁, v₂) = 1",
            "!(coprime[N,i0,s,*,v1,v2])": "gcd(v₁, v₂) ≠ 1",
            "(bezEq[N,+,*,i1,v1,v2,v3,v4])": "v₁ · v₂ = v₃ · v₄ + 1",
            "!(bezEq[N,+,*,i1,v1,v2,v3,v4])": "v₁ · v₂ ≠ v₃ · v₄ + 1",
            "(bez1[N,+,*,i1,v1,v2])": "∃ m, n ∈ ℕ: v₁ · m = v₂ · n ± 1",
            "!(bez1[N,+,*,i1,v1,v2])": "∄ m, n ∈ ℕ: v₁ · m = v₂ · n ± 1",
        }
        for expression, expected in cases.items():
            with self.subTest(expression=expression):
                self.assertEqual(self.read(expression), expected)

    def test_sequence_and_factorization_templates(self):
        cases = {
            "(inSequence[N,v1,V1])": "v₁ is a value of V₁",
            "!(inSequence[N,v1,V1])": "v₁ is not a value of V₁",
            "(primeseq[N,+,*,i1,i2,V1])": "all values of V₁ are prime",
            "!(primeseq[N,+,*,i1,i2,V1])": "not all values of V₁ are prime",
            "(ordered[N,+,V1])": "V₁ is non-increasing",
            "!(ordered[N,+,V1])": "V₁ is not non-increasing",
            "(primeFactorSeq[N,s,+,*,i0,i1,i2,v1,V1,v2])":
                "v₂ = ∏(i=0..v₁) V₁(i) with primes V₁(0) ≥ V₁(1) ≥ … ≥ V₁(v₁)",
            "!(primeFactorSeq[N,s,+,*,i0,i1,i2,v1,V1,v2])":
                "¬(v₂ = ∏(i=0..v₁) V₁(i) with primes V₁(0) ≥ V₁(1) ≥ … ≥ V₁(v₁))",
            "(hasPrimeFactorization[N,s,+,*,i0,i1,i2,v1])":
                "∃ primes p₀ ≥ p₁ ≥ … ≥ pₖ: v₁ = ∏(i=0..k) pᵢ",
            "!(hasPrimeFactorization[N,s,+,*,i0,i1,i2,v1])":
                "∄ primes p₀ ≥ p₁ ≥ … ≥ pₖ: v₁ = ∏(i=0..k) pᵢ",
        }
        for expression, expected in cases.items():
            with self.subTest(expression=expression):
                self.assertEqual(self.read(expression), expected)

    def test_incubator_templates(self):
        cases = {
            "(constSeq[N,+,V1,i0,v1,V2,v2])": "V₂ = [0, v₁] ∧ V₁: V₂ → ℕ ∧ V₁ ≡ v₂",
            "(nonSequence[N,s,+,V1,v1,V2,v2,v3,v4])":
                "V₂ = V₁|≤ v₁ ∧ s(v₁) = v₄ ∧ V₁(v₄) = v₂, V₁(v₄) = v₃",
            "!(nonSequence[N,s,+,V1,v1,V2,v2,v3,v4])":
                "¬(V₂ = V₁|≤ v₁ ∧ s(v₁) = v₄ ∧ V₁(v₄) = v₂, V₁(v₄) = v₃)",
        }
        for expression, expected in cases.items():
            with self.subTest(expression=expression):
                self.assertEqual(self.read(expression), expected)

    def test_substituted_arguments_keep_their_grouping(self):
        theorem = FTA + "(>[v1,v2,v3](in3[v1,v2,v3,+])(bez1[N,+,*,i1,v3,v1])))"
        self.assertEqual(self.readable.theorem(theorem, as_html=False),
                         "∃ m, n ∈ ℕ: (v₁ + v₂) · m = v₁ · n ± 1")

    def test_numbered_unfolding_stops_at_named_definitions(self):
        self.assertEqual(self.read("(existence40[N])"),
                         "∃ y₁ ∈ ℕ: (∃ primes p₀ ≥ p₁ ≥ … ≥ pₖ: y₁ = ∏(i=0..k) pᵢ)")
        self.assertEqual(self.read("(and41[N,V1,v1])"), "V₁ is non-increasing ∧ v₁ is a value of V₁")

    def test_named_definition_without_template_asserts(self):
        with self.assertRaises(AssertionError):
            self.read("(myNamed[v1,N])")
        with self.assertRaises(AssertionError):
            self.read("(bez1[N,+,*,i1,v1])")

    def test_html_keeps_relations_and_units_whole(self):
        html = self.read("(divisorSet[N,+,*,i2,v1,V1])", as_html=True)
        self.assertTrue(html.startswith('<span class="rf-nb">'))
        self.assertIn('<span class="rf-unit">{d ∈ <span class="bb-N">ℕ</span> : 2 ≤ d ≤ v<sub>1</sub>, '
                      'd ∣ v<sub>1</sub>}</span>', html)
        factorization = self.read("(hasPrimeFactorization[N,s,+,*,i0,i1,i2,v1])", as_html=True)
        self.assertIn("p<sub>0</sub>", factorization)
        self.assertIn("p<sub>k</sub>", factorization)
        self.assertIn('<span class="rf-big"><span class="rf-big-hi">k</span>'
                      '<span class="rf-big-sym">∏</span><span class="rf-big-lo">i = 0</span></span> '
                      'p<sub>i</sub>', factorization)
        self.assertEqual(factorization.count('<span class="rf-nb">'), 2)
        self.assertIn('<span class="rf-unit">non-increasing</span>', self.read("(ordered[N,+,V1])", as_html=True))
        product = self.read("(primeFactorSeq[N,s,+,*,i0,i1,i2,v1,V1,v2])", as_html=True)
        self.assertEqual(product.count('<span class="rf-nb">'), 2)
        self.assertIn('<span class="rf-big-sym">∏</span><span class="rf-big-lo">i = 0</span></span> '
                      'V<sub>1</sub>(i)', product)

    def test_template_bound_names_never_meet_page_names(self):
        # A name a template binds comes from the template letters, which no
        # proof page uses for a variable (v, w, V, W, y, Y with digits, a
        # fold's index i); every page-style name in a reading is an argument.
        # A template's product binds its own index i, as every fold does.
        samples = {
            "residual": "(residual[v1,v2,v3,+])", "infiniteSequence": "(infiniteSequence[V1,N])",
            "constSeq": "(constSeq[N,+,V1,i0,v1,V2,v2])", "nonInterval": "(nonInterval[N,s,+,V1,v1,V2,v2,v3])",
            "nonSequence": "(nonSequence[N,s,+,V1,v1,V2,v2,v3,v4])", "bezEq": "(bezEq[N,+,*,i1,v1,v2,v3,v4])",
            "bez1": "(bez1[N,+,*,i1,v1,v2])", "coprime": "(coprime[N,i0,s,*,v1,v2])",
            "inSequence": "(inSequence[N,v1,V1])", "primeseq": "(primeseq[N,+,*,i1,i2,V1])",
            "ordered": "(ordered[N,+,V1])", "subset": "(subset[V1,V2])", "empty": "(empty[V1])",
            "minimum": "(minimum[N,+,V1,v1])", "hasMinimum": "(hasMinimum[N,+,V1])",
            "divisorSet": "(divisorSet[N,+,*,i2,v1,V1])",
            "primeFactorSeq": "(primeFactorSeq[N,s,+,*,i0,i1,i2,v1,V1,v2])",
            "hasPrimeFactorization": "(hasPrimeFactorization[N,s,+,*,i0,i1,i2,v1])",
            "limitSet": "(limitSet[N,+,V1,v2,V2])",
        }
        self.assertEqual(set(samples) - {"limitSet"}, set(vh._NAMED_TEMPLATES))
        page_name = re.compile(r"[vwyVWY][₀₁₂₃₄₅₆₇₈₉]+")
        for head, expression in samples.items():
            arguments = {vh.display_token(token, vh._TEXT) for token in parse_mpl(expression).arguments}
            for sign in ("", "!"):
                with self.subTest(expression=sign + expression):
                    text = self.read(sign + expression)
                    self.assertTrue(set(page_name.findall(text)) <= arguments, text)
                    bare = re.sub(r"[∑∏]\(i=|\(i\)", "", text)
                    self.assertIsNone(re.search(r"(?<!\w)[vwyVWYi](?!\w)", bare), text)
                    bound = [group for group in re.findall(r"[∃∄] (?:primes )?([^:]+):", text)]
                    bound += re.findall(r"\{(\w) ∈", text)
                    for group in bound:
                        letters = set(re.findall(r"[A-Za-z]", group.split("∈")[0]))
                        self.assertTrue(letters <= vh._TEMPLATE_LETTERS, text)

    def test_subscript_spelling(self):
        self.assertEqual(vh._TEXT.subscript("p", "0"), "p₀")
        self.assertEqual(vh._TEXT.subscript("p", "k"), "pₖ")
        self.assertEqual(vh._HTML.subscript("p", "k"), "p<sub>k</sub>")
        with self.assertRaises(AssertionError):
            vh._TEXT.subscript("p", "m")
        with self.assertRaises(AssertionError):
            vh._TEXT.subscript("y", "1")


if __name__ == "__main__":
    unittest.main()
