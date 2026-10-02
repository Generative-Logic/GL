# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt)
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Induction helpers may close their recorded terminal goal by contradiction."""

import unittest

from proof_export.lean import _render_induction_theorem


def induction_fixture(negative=False, reductio_roles=("check_zero", "check_induction_condition"),
                      unrelated_role=None):
    goal = "!(=[v1,i0])" if negative else "(=[v1,v1])"
    opposite = "(=[v1,i0])" if negative else "!(=[v1,v1])"
    variable_types = {"N": "set", "v1": "element", "i0": "element"}
    chapters = []
    for number, role in enumerate(("induction_typing", "check_zero", "check_induction_condition"), 1):
        premise = "(in[v1,N])" if role == "induction_typing" else goal
        first = {"id": f"chapter_{number}_line_1", "source_line": 1,
                 "source_tag": "task formulation", "action": "assume", "namespace": "main",
                 "conclusion": {"mpl": premise}, "dependencies": []}
        steps = [first]
        if role in reductio_roles:
            second = {"id": f"chapter_{number}_line_2", "source_line": 2,
                      "source_tag": "task formulation", "action": "assume", "namespace": "main",
                      "conclusion": {"mpl": opposite}, "dependencies": []}
            steps.append(second)
            if role == unrelated_role:
                steps.append({"id": f"chapter_{number}_line_3", "source_line": 3,
                              "source_tag": "task formulation", "action": "assume", "namespace": "main",
                              "conclusion": {"mpl": "!(in[v1,N])"}, "dependencies": []})
            final_line = len(steps) + 1
            steps.append({"id": f"chapter_{number}_line_{final_line}", "source_line": final_line,
                          "source_tag": "contradiction", "action": "ex_falso", "namespace": "main",
                          "conclusion": {"mpl": goal}, "dependencies": [
                              {"step": row["id"], "namespace": "main", "expression": row["conclusion"]}
                              for row in (first, second)]})
        chapters.append({"number": number, "role": role, "source_file": f"{number}_{role}.txt",
                         "variable_types": variable_types, "steps": steps})
    return {"id": "induction_reductio_fixture", "source_index": 0, "method": "induction",
            "induction_variable": "v1", "variable_types": variable_types, "chapters": chapters,
            "theorem": {"mpl": "(>[N,i0,s,+,*,i1](AnchorPeano[N,i0,s,+,*,i1])"
                        "(>[v1](in[v1,N])(>[]" + goal + goal + ")))"}}


class InductionReductioTests(unittest.TestCase):
    def render(self, **kwargs):
        lines, _ = _render_induction_theorem(induction_fixture(**kwargs))
        return "\n".join(lines)

    def test_positive_head_discharges_each_recorded_negated_goal(self):
        source = self.render()
        self.assertEqual(source.count("apply Classical.byContradiction"), 2)
        self.assertIn("intro base_reductio", source)
        self.assertIn("intro step_reductio", source)
        self.assertIn("exact base_reductio (zeroRule zero base_premise_1 base_reductio)", source)
        self.assertIn("exact step_reductio (stepRule induction_m step_premise_1 step_reductio)", source)

    def test_negative_head_introduces_the_positive_counterpart(self):
        source = self.render(negative=True)
        self.assertNotIn("apply Classical.byContradiction", source)
        self.assertIn("exact (zeroRule zero base_premise_1 base_reductio) base_reductio", source)
        self.assertIn("exact (stepRule induction_m step_premise_1 step_reductio) step_reductio", source)

    def test_only_the_helper_that_records_reductio_introduces_it(self):
        for role, other in (("check_zero", "step"), ("check_induction_condition", "base")):
            with self.subTest(role=role):
                source = self.render(reductio_roles=(role,))
                self.assertEqual(source.count("apply Classical.byContradiction"), 1)
                self.assertNotIn(f"intro {other}_reductio", source)

    def test_ordinary_helpers_keep_direct_applications(self):
        source = self.render(reductio_roles=())
        self.assertNotIn("intro base_reductio", source)
        self.assertNotIn("intro step_reductio", source)
        self.assertIn("exact zeroRule zero base_premise_1", source)
        self.assertIn("exact stepRule induction_m step_premise_1", source)

    def test_unrelated_main_assumption_is_rejected(self):
        for role in ("check_zero", "check_induction_condition"):
            with self.subTest(role=role), self.assertRaisesRegex(AssertionError, "cannot resolve .*helper assumption"):
                self.render(reductio_roles=(role,), unrelated_role=role)

    def test_existing_property_premise_is_not_introduced_again(self):
        fixture = induction_fixture()
        fixture["theorem"]["mpl"] = (
            "(>[N,i0,s,+,*,i1](AnchorPeano[N,i0,s,+,*,i1])"
            "(>[v1](in[v1,N])(>[](=[v1,v1])(>[]!(=[v1,v1])(=[v1,v1])))))"
        )
        lines, _ = _render_induction_theorem(fixture)
        source = "\n".join(lines)
        self.assertNotIn("intro base_reductio", source)
        self.assertNotIn("intro step_reductio", source)
        self.assertIn("exact zeroRule zero base_premise_1 base_premise_2", source)
        self.assertIn("exact stepRule induction_m step_premise_1 step_premise_2", source)


if __name__ == "__main__":
    unittest.main()
