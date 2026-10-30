# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Tests for compact-first integration rows in the certificate and the Lean
writer (D-503): the rule a compact stands for
(``proof_export.integration_compact``), the derived rule step plus the
compact's fold (``proof_export.certificate._chapter_certificate``) and the
fold's Lean replay (action ``integration_compact``)."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proof_export import lean  # noqa: E402
from proof_export.certificate import _chapter_certificate  # noqa: E402
from proof_export.flat_integration import build_flat_integration_plan  # noqa: E402
from proof_export.graph import Chapter, Dependency, ProofRow, parse_scoped_expression  # noqa: E402
from proof_export.integration_compact import compact_rule_rows, expand_integration_compact  # noqa: E402
from proof_export.mpl import parse_mpl  # noqa: E402

DEFINITIONS = {
    "andT": {"category": "and", "signature": "(andT[u_1,u_2])",
             "elements": ["(p[u_1,u_2])", "(q[u_2,u_1])"]},
    "implicationT1": {"category": "implication", "signature": "(implicationT1[u_1,u_2])",
                      "elements": ["(p[u_1,u_2])", "(q[u_2,u_1])", "(andT[u_1,u_2])"]},
    "exT": {"category": "existence", "signature": "(exT[u_1,u_2,u_3])",
            "elements": ["(in[1,u_1])", "(existence901[u_1,u_2,u_3,1])"]},
    "existence901": {"category": "existence", "signature": "(existence901[u_1,u_2,u_3,u_4])",
                     "elements": ["(in[1,u_1])", "(in3[u_2,u_3,u_4,1])"]},
    "implicationT2": {"category": "implication", "signature": "(implicationT2[u_1,u_2])",
                      "elements": ["(in[1,u_1])", "(in[2,u_1])", "(in3[u_2,3,1,2])",
                                   "(exT[u_1,u_2,3])"]},
    "or901": {"category": "or", "signature": "(or901[u_1,u_2])",
              "elements": ["(p[u_1,u_2])", "(q[u_2,u_1])"]},
    "implicationT3": {"category": "implication", "signature": "(implicationT3[u_1,u_2])",
                      "elements": ["(or901[u_1,u_2])", "(andT[u_1,u_2])"]},
    "implicationT4": {"category": "implication", "signature": "(implicationT4[u_1,u_2])",
                      "elements": ["(p[u_1,u_2])", "(or901[u_1,u_2])"]},
    "implicationT5": {"category": "implication", "signature": "(implicationT5[u_1,u_2])",
                      "elements": ["(p[u_1,u_1])", "(or901[u_1,u_2])"]},
    "implicationT6": {"category": "implication", "signature": "(implicationT6[u_1])",
                      "elements": ["(in[1,u_1])", "(in[u_1,1])"]},
    "implicationT7": {"category": "implication", "signature": "(implicationT7[u_1])",
                      "elements": ["(in[u_1,u_1])", "(in[1,u_1])"]},
}
SIGNATURES = {"p": ("element", "element"), "q": ("element", "element"),
              "andT": ("element", "element"), "implicationT1": ("element", "element")}


def _row(line: int, conclusion: str, tag: str, *dependencies: str) -> ProofRow:
    """One main-scope row citing main-scope dependencies."""
    return ProofRow(line, parse_scoped_expression(conclusion), "main", tag,
                    tuple(Dependency(parse_scoped_expression(text), "main") for text in dependencies))


def _chapter(*rows: ProofRow) -> Chapter:
    return Chapter(Path("1_single_chapter_proof.txt"), 1, "single_chapter_proof", rows)


class CompactExpansionTests(unittest.TestCase):
    def test_placeholders_stay_apart_from_actual_arguments(self) -> None:
        """A placeholder ``1`` never merges with an actual argument ``1``."""
        rule = expand_integration_compact(parse_mpl("(implicationT6[1])"), DEFINITIONS, "cr4_")
        self.assertEqual(rule, parse_mpl("(>[cr4_1](in[cr4_1,1])(in[1,cr4_1]))"))

    def test_each_local_is_bound_at_its_first_premise(self) -> None:
        rule = expand_integration_compact(parse_mpl("(implicationT2[N,b])"), DEFINITIONS, "cr1_")
        self.assertEqual(rule, parse_mpl(
            "(>[cr1_1](in[cr1_1,N])(>[cr1_2](in[cr1_2,N])"
            "(>[cr1_3](in3[b,cr1_3,cr1_1,cr1_2])(exT[N,b,cr1_3]))))"))

    def test_a_local_in_the_head_is_rejected(self) -> None:
        with self.assertRaises(AssertionError):
            expand_integration_compact(parse_mpl("(implicationT7[a])"), DEFINITIONS, "cr1_")

    def test_a_non_implication_is_rejected(self) -> None:
        with self.assertRaises(AssertionError):
            expand_integration_compact(parse_mpl("(andT[a,b])"), DEFINITIONS, "cr1_")


class CompactRuleRowTests(unittest.TestCase):
    def test_integration_and_introduction_rows_map_to_their_rules(self) -> None:
        chapter = _chapter(
            _row(1, "(implicationT3[a,b])", "reformulation for integration flattened",
                 "(andT[a,b])_integration_goal"),
            _row(2, "(implicationT4[a,b])", "expansion", "(implicationT3[a,b])"),
            _row(3, "(>[](or901[a,b])(andT[a,b]))", "expansion", "(implicationT3[a,b])"))
        rules = compact_rule_rows(chapter, DEFINITIONS)
        self.assertEqual(sorted(rules), [1, 2])
        self.assertEqual(rules[1]["rule"], parse_mpl("(>[](or901[a,b])(andT[a,b]))"))
        self.assertEqual(rules[2]["rule"], parse_mpl("(>[](p[a,b])(or901[a,b]))"))
        self.assertIn("introduction", rules[2])

    def test_an_introduction_of_no_disjunct_is_rejected(self) -> None:
        chapter = _chapter(_row(2, "(implicationT5[a,b])", "expansion", "(implicationT3[a,b])"))
        with self.assertRaises(AssertionError):
            compact_rule_rows(chapter, DEFINITIONS)


class FlatTemplateBinderTests(unittest.TestCase):
    RULE = "(>[w1](in[w1,N])(>[w2](in[w2,N])(>[x](in3[b,x,w1,w2])(exT[N,b,x]))))"

    def test_a_head_binder_stands_for_the_template_argument(self) -> None:
        plan = build_flat_integration_plan(parse_mpl(self.RULE), parse_mpl("(exT[N,b,X])"), DEFINITIONS)
        self.assertEqual(plan["premise_count"], 3)
        self.assertEqual(plan["construction"]["head"], "exT")

    def test_a_head_binder_onto_a_used_name_is_rejected(self) -> None:
        with self.assertRaises(AssertionError):
            build_flat_integration_plan(parse_mpl(self.RULE), parse_mpl("(exT[N,b,b])"), DEFINITIONS)

    def test_another_free_head_argument_is_rejected(self) -> None:
        with self.assertRaises(AssertionError):
            build_flat_integration_plan(parse_mpl(self.RULE), parse_mpl("(exT[N,c,X])"), DEFINITIONS)


class CompactRowCertificateTests(unittest.TestCase):
    ROWS = (
        _row(1, "(&(p[a,b])(q[b,a]))_integration_goal", "expansion for integration",
             "(andT[a,b])_integration_goal"),
        _row(2, "(implicationT1[a,b])", "reformulation for integration and",
             "(&(p[a,b])(q[b,a]))_integration_goal"),
        _row(3, "(>[](p[a,b])(>[](q[b,a])(andT[a,b])))", "expansion", "(implicationT1[a,b])"))

    def certificate(self) -> dict[str, object]:
        """The chapter certificate; the certificate hashes its chapter file."""
        with tempfile.TemporaryDirectory(prefix="compact_rows_") as temporary:
            path = Path(temporary) / "1_single_chapter_proof.txt"
            path.write_text("compact-first fixture", encoding="utf-8")
            chapter = Chapter(path, 1, "single_chapter_proof", self.ROWS)
            return _chapter_certificate(chapter, SIGNATURES, {}, {}, certificate_schema=3,
                                        binary_entries=DEFINITIONS)

    def test_a_compact_row_is_its_derived_rule_and_the_fold(self) -> None:
        steps = {step["id"]: step for step in self.certificate()["steps"]}
        self.assertEqual(sorted(steps), ["chapter_1_line_1", "chapter_1_line_2",
                                         "chapter_1_line_2_rule", "chapter_1_line_3"])
        rule = steps["chapter_1_line_2_rule"]
        self.assertTrue(rule["derived_rule"])
        self.assertEqual(rule["action"], "integration_conjoin")
        self.assertEqual(rule["reformulation_definition"], "andT")
        self.assertEqual(parse_mpl(rule["conclusion"]["mpl"]),
                         parse_mpl("(>[](p[a,b])(>[](q[b,a])(andT[a,b])))"))
        fold = steps["chapter_1_line_2"]
        self.assertEqual(fold["action"], "integration_compact")
        self.assertEqual(fold["dependencies"][0]["step"], "chapter_1_line_2_rule")
        self.assertNotIn("reformulation_definition", fold)
        self.assertEqual(steps["chapter_1_line_3"]["dependencies"][0]["step"], "chapter_1_line_2")

    def test_the_lean_fold_unfolds_the_compact_onto_the_derived_rule(self) -> None:
        chapter = self.certificate()
        lines, dispositions = lean._render_chapter_steps(
            chapter, "compact_rows", {}, {}, ambient_variables={"a", "b"},
            binary_definitions=DEFINITIONS)
        text = "\n".join(lines)
        self.assertIn(f"simpa only [{lean.lean_constant('implicationT1')}] using row_2_rule", text)
        self.assertEqual(len(dispositions), 3)
        self.assertEqual(lean._chapter_manifest(chapter, dispositions)["source_row_count"], 3)


if __name__ == "__main__":
    unittest.main()
