# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt)
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""An implication subproof opened inside an OR case.

An OR at ``main`` opens the case C; inside C an implication subproof S is
opened and closed into C. The certificate builds C's context before S's,
although subproof contexts are scheduled before OR contexts, and S's context
carries C's layer first (a twin weak step block opens such a subproof in its
successor case).
"""

from pathlib import Path
import unittest

from proof_export.certificate import _schema3_action_metadata
from proof_export.graph import Chapter, Dependency, ProofRow, parse_scoped_expression
from proof_export.lean import _render_chapter_steps, _scope_layers
from proof_export.mpl import parse_mpl, to_mpl

CASE = "main_boundary_ordis_(or98[x])_((p[x]))"
SUBPROOF = CASE + "_boundary_(implication99[x])"
DEFINITIONS = {
    "or98": {"arity": 1, "category": "or", "elements": ["(p[u_1])", "(q[u_1])"]},
    "implication99": {"arity": 1, "category": "implication",
                      "elements": ["(>[w](=[w,u_1])(=[w,u_1]))"]},
}


def proof_row(line, conclusion, namespace, tag, *dependencies):
    return ProofRow(line, parse_scoped_expression(conclusion), namespace, tag,
                    tuple(Dependency(parse_scoped_expression(text), scope)
                          for text, scope in dependencies))


class SubproofInsideOrCaseTests(unittest.TestCase):
    def rows(self, with_case=True):
        rows = [
            proof_row(1, "(or98[x])", "main", "task formulation"),
            proof_row(2, "(p[x])", CASE, "or disintegration", ("(or98[x])", "main")),
            proof_row(3, "(implication99[x])", CASE, "validity name", ("(=[w,x])", SUBPROOF)),
            proof_row(4, "(=[w,x])", SUBPROOF, "premise element",
                      ("(>[w](=[w,x])(=[w,x]))_integration_goal", CASE)),
            proof_row(5, "(>[w](=[w,x])(=[w,x]))_integration_goal", CASE,
                      "expansion for integration", ("(implication99[x])_integration_goal", CASE)),
        ]
        if not with_case:
            rows = [row for row in rows if row.tag != "or disintegration"]
        return tuple(rows)

    def metadata(self, **changes):
        chapter = Chapter(Path("fixture.txt"), 0, "check_induction_condition", self.rows(**changes))
        return _schema3_action_metadata(chapter, DEFINITIONS, {}, {})

    def test_the_case_context_is_built_before_the_subproof_inside_it(self):
        metadata, contexts = self.metadata()
        self.assertEqual([context["namespace"] for context in contexts], [CASE, SUBPROOF])
        by_namespace = {context["namespace"]: context for context in contexts}
        self.assertEqual(by_namespace[CASE]["scope_kind"], "or_branch")
        subproof = by_namespace[SUBPROOF]
        self.assertEqual(subproof["parent_namespace"], CASE)
        self.assertEqual(to_mpl(parse_mpl(subproof["template"]["mpl"])),
                         "(>[](p[x])(>[w](=[w,x])(=[w,x])))")
        self.assertEqual([to_mpl(premise) for _, premise in _scope_layers(subproof)],
                         ["(p[x])", "(=[w,x])"])
        self.assertEqual(metadata[3], {"scope_context": subproof})
        self.assertEqual(metadata[4], {"scope_namespace": SUBPROOF})
        self.assertEqual(metadata[2], {"scope_namespace": CASE})

    def test_a_subproof_whose_opening_scope_has_no_context_is_rejected(self):
        with self.assertRaisesRegex(AssertionError, "parent scope is not established"):
            self.metadata(with_case=False)


class CompactionInsideOrCaseRenderingTests(unittest.TestCase):
    """A local definition compaction inside an OR case folds a row of that
    case, instantiated at the case's prefix."""

    def chapter(self):
        case_context = {"namespace": CASE, "parent_namespace": "main", "scope_kind": "or_branch",
                        "definition_head": "or98",
                        "template": {"mpl": "(>[](p[x])(p[x]))"},
                        "compact": {"mpl": "(or98[x])"}}
        steps = [
            {"id": "chapter_1_line_1", "source_line": 1, "namespace": CASE, "action": "assume",
             "source_tag": "or disintegration", "conclusion": {"mpl": "(p[x])"},
             "dependencies": [], "scope_namespace": CASE},
            {"id": "chapter_1_line_2", "source_line": 2, "namespace": CASE,
             "action": "definition_compact", "source_tag": "compilation",
             "conclusion": {"mpl": "(implication97[x])"},
             "dependencies": [{"step": "chapter_1_line_1", "namespace": CASE,
                               "source_kind": "local_step", "expression": {"mpl": "(p[x])"}}]},
        ]
        return {"steps": steps, "variable_types": {"x": "element"}, "scope_contexts": [case_context]}

    def render(self, chapter):
        lines, manifest = _render_chapter_steps(chapter, "compaction_inside_or_case", {}, {},
                                                ambient_variables={"x"})
        self.assertEqual(len(manifest), 2)
        return "\n".join(lines)

    def test_the_compaction_unfolds_the_cited_row_inside_the_case(self):
        proof = self.render(self.chapter())
        self.assertIn("have scoped_fact_1 := row_1 scope_premise_1", proof)
        self.assertIn("using (scoped_fact_1)", proof)

    def test_a_compaction_of_a_row_from_another_scope_is_rejected(self):
        chapter = self.chapter()
        chapter["steps"][1]["dependencies"][0]["namespace"] = "main"
        with self.assertRaises(AssertionError):
            self.render(chapter)


if __name__ == "__main__":
    unittest.main()
