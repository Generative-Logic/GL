# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
# Dual-licensed under AGPLv3 and a commercial license.
# See https://generative-logic.com/license. Contributions require CLA.

"""Structural and Lean-kernel tests for explicit flat integration replay."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from proof_export.certificate import _schema3_action_metadata
from proof_export.flat_integration import build_flat_integration_plan, render_flat_integration_term
from proof_export.graph import Chapter, Dependency, ProofRow, parse_scoped_expression
from proof_export.mpl import parse_mpl
from tests.lean_toolchain import LEAN_PROJECT, lean_environment, lean_executable


ROOT = Path(__file__).resolve().parents[1]
DEFINITIONS = {
    "root": {"category": "existence", "signature": "(root[u_1])",
             "elements": ["(in[1,u_1])", "(both[1,u_1])"]},
    "both": {"category": "and", "signature": "(both[u_1,u_2])",
             "elements": ["(p[u_1,u_2])", "(q[u_1])"]},
}
RULE = "(>[w](in[w,N])(>[](p[w,N])(>[](q[w])(root[N]))))"


class FlatIntegrationExportTests(unittest.TestCase):
    def test_e6_missing_second_factor_bounds_is_rejected_by_both_checkers(self):
        from tests.test_harness import make_state_minimal, make_proof_line
        from verifier import check_reformulation_for_integration_flattened

        fixture = json.loads((ROOT / 'tests/fixtures/e6_invalid_flat_rule.json').read_text())
        with self.assertRaisesRegex(AssertionError, 'do not match'):
            build_flat_integration_plan(parse_mpl(fixture['conclusion']),
                                        parse_mpl(fixture['compact']), fixture['definitions'])
        state = make_state_minimal()
        state.current_gl_binary = fixture['definitions']
        row = make_proof_line(fixture['conclusion'], 'main',
                             'reformulation for integration flattened',
                             fixture['compact'] + '_integration_goal', 'main')
        self.assertFalse(check_reformulation_for_integration_flattened(row, [row], state))

    def plan(self, rule=RULE):
        return build_flat_integration_plan(parse_mpl(rule), parse_mpl("(root[N])"), DEFINITIONS)

    def test_plan_uses_exact_premise_positions_and_witness(self):
        plan = self.plan()
        self.assertEqual(plan["premise_count"], 3)
        root = plan["construction"]
        self.assertEqual(root["witness_groups"], [["w"]])
        self.assertEqual(root["children"][0], {"kind": "premise", "index": 0})
        self.assertEqual(root["children"][1]["children"],
                         [{"kind": "premise", "index": 1}, {"kind": "premise", "index": 2}])

    def test_rejects_missing_or_wrong_premises_and_unused_binders(self):
        invalid = [
            "(>[w](in[w,N])(>[](p[w,N])(root[N])))",
            RULE.replace("(q[w])", "!(q[w])"),
            RULE.replace("(>[w]", "(>[w,unused]"),
            "(>[w](in[w,N])(>[](both[w,N])(root[N])))",
        ]
        for rule in invalid:
            with self.subTest(rule=rule), self.assertRaises(AssertionError):
                self.plan(rule)

    def test_certificate_metadata_recognizes_the_existing_tag(self):
        row = ProofRow(1, parse_scoped_expression(RULE), "main",
                       "reformulation for integration flattened",
                       (Dependency(parse_scoped_expression("(root[N])_integration_goal"), "main"),))
        chapter = Chapter(Path("1_single_chapter_proof.txt"), 1, "single_chapter_proof", (row,))
        metadata, contexts = _schema3_action_metadata(chapter, DEFINITIONS, {}, {})
        self.assertFalse(contexts)
        self.assertEqual(metadata[1]["flat_integration_plan"], self.plan())

    def test_d1_intact_and_extended_variants(self):
        from tests.test_verifier_flattened import GOAL, RULE_LT, RULE_LE, RULE_FLAT

        definitions = json.loads((ROOT / "tests/fixtures/GL_binaries/GL_binary_FTA.json").read_text())
        for rule in (RULE_LT, RULE_LE, RULE_FLAT):
            with self.subTest(rule=rule):
                plan = build_flat_integration_plan(parse_mpl(rule), parse_mpl(GOAL), definitions)
                term = render_flat_integration_term(
                    plan, [f"h{i}" for i in range(plan["premise_count"])], lambda name: name)
                self.assertIn("fun flat_counterexample_", term)
                self.assertNotIn("sorry", term)

    def test_lean_kernel_checks_recorded_premise_order(self):
        lean = lean_executable()
        reordered = "(>[w](in[w,N])(>[](q[w])(>[](p[w,N])(root[N]))))"
        terms = [render_flat_integration_term(self.plan(rule), ["h1", "h2", "h3"], lambda name: name)
                 for rule in (RULE, reordered)]
        source = """def FlatPair {U : Type} (N : U → Prop) (P : U → (U → Prop) → Prop)
    (Q : U → Prop) (w : U) : Prop := P w N ∧ Q w
def FlatGoal {U : Type} (N : U → Prop) (P : U → (U → Prop) → Prop)
    (Q : U → Prop) : Prop := ¬ (∀ w, N w → ¬ FlatPair N P Q w)
"""
        for index, term in enumerate(terms):
            hypotheses = "(h2 : P w N) (h3 : Q w)" if index == 0 else "(h2 : Q w) (h3 : P w N)"
            source += (f"example {{U : Type}} (N : U → Prop) (P : U → (U → Prop) → Prop) "
                       f"(Q : U → Prop) (w : U) (h1 : N w) {hypotheses} : FlatGoal N P Q := by\n"
                       f"  exact {term}\n")
        with tempfile.TemporaryDirectory(prefix="flat_lean_") as temporary:
            proof = Path(temporary) / "Flat.lean"
            proof.write_text(source, encoding="utf-8")
            environment = lean_environment()
            result = subprocess.run([str(lean), str(proof)], cwd=LEAN_PROJECT,
                                    env=environment, text=True, encoding="utf-8",
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout)


if __name__ == "__main__":
    unittest.main()
