# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt)
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Subproofs and contradiction scopes opened inside a contradiction scope.

The contradiction-scope demand opens an implication subproof and the
reductio beside it inside a contradiction scope S. The subproof closes into
S and the reductio's contradiction lands at S. Their certificate contexts
carry S's layers first, and the Lean replay enters S before them.
"""

from pathlib import Path
import subprocess
import tempfile
import unittest

from proof_export.certificate import _schema3_action_metadata
from proof_export.graph import Chapter, Dependency, ProofRow, parse_scoped_expression
from proof_export.lean import _render_chapter_steps, _scope_layers
from proof_export.mpl import parse_mpl, to_mpl
from tests.lean_toolchain import LEAN_PROJECT, lean_environment, lean_executable

SCOPE = "main_boundary_contradiction_!(p[x])"
REDUCTIO = SCOPE + "_boundary_contradiction_!(implication99[x])"
SUBPROOF = SCOPE + "_boundary_(implication99[x])"
DEFINITIONS = {"implication99": {"arity": 1, "category": "implication",
                                 "elements": ["(>[w](=[w,u_1])(=[w,u_1]))"]}}


def proof_row(line, conclusion, namespace, tag, *dependencies):
    return ProofRow(line, parse_scoped_expression(conclusion), namespace, tag,
                    tuple(Dependency(parse_scoped_expression(text), scope)
                          for text, scope in dependencies))


class NestedContextMetadataTests(unittest.TestCase):
    def rows(self, validity_scope=SCOPE, reductio_closure=SCOPE, template_scope=SCOPE):
        return (
            proof_row(1, "!(p[x])", "main", "contradiction",
                      ("!(implication99[x])", SCOPE), ("(implication99[x])", SCOPE)),
            proof_row(2, "!(implication99[x])", reductio_closure, "contradiction",
                      ("!(r[x])", REDUCTIO), ("(r[x])", "main")),
            proof_row(3, "!(r[x])", REDUCTIO, "implication"),
            proof_row(4, "(implication99[x])", REDUCTIO, "task formulation"),
            proof_row(5, "(implication99[x])", validity_scope, "validity name",
                      ("(=[w,x])", SUBPROOF)),
            proof_row(6, "(=[w,x])", SUBPROOF, "premise element",
                      ("(>[w](=[w,x])(=[w,x]))_integration_goal", template_scope)),
            proof_row(7, "(>[w](=[w,x])(=[w,x]))_integration_goal", template_scope,
                      "expansion for integration",
                      ("(implication99[x])_integration_goal", template_scope)),
            proof_row(8, "(p[x])", SCOPE, "task formulation"),
            proof_row(9, "(r[x])", "main", "task formulation"),
        )

    def metadata(self, **changes):
        chapter = Chapter(Path("fixture.txt"), 0, "check_zero", self.rows(**changes))
        return _schema3_action_metadata(chapter, DEFINITIONS, {}, {})

    def test_contexts_inside_a_contradiction_scope_carry_its_layers(self):
        metadata, contexts = self.metadata()
        by_namespace = {context["namespace"]: context for context in contexts}
        self.assertEqual([context["namespace"] for context in contexts][0], SCOPE)
        self.assertNotIn("parent_namespace", by_namespace[SCOPE])
        reductio = by_namespace[REDUCTIO]
        self.assertEqual(reductio["parent_namespace"], SCOPE)
        self.assertEqual(reductio["scope_kind"], "contradiction_assumption")
        self.assertEqual(to_mpl(parse_mpl(reductio["template"]["mpl"])),
                         "(>[](p[x])(>[](implication99[x])(implication99[x])))")
        subproof = by_namespace[SUBPROOF]
        self.assertEqual(subproof["parent_namespace"], SCOPE)
        self.assertEqual(subproof["template"]["proof_scope"], "integration_goal")
        self.assertEqual([to_mpl(premise) for _, premise in _scope_layers(subproof)],
                         ["(p[x])", "(=[w,x])"])
        self.assertEqual(metadata[5], {"scope_context": subproof})
        self.assertEqual(metadata[4], {"scope_namespace": REDUCTIO})
        self.assertEqual(metadata[6], {"scope_namespace": SUBPROOF})

    def test_a_closure_above_the_opening_scope_is_rejected(self):
        with self.assertRaises(AssertionError):
            self.metadata(validity_scope="main")

    def test_a_reductio_closed_outside_its_parent_is_rejected(self):
        with self.assertRaises(AssertionError):
            self.metadata(reductio_closure="main")

    def test_a_template_away_from_the_opening_scope_is_rejected(self):
        with self.assertRaisesRegex(AssertionError, "not cited at the scope that opened it"):
            self.metadata(template_scope="main")


class NestedContextRenderingTests(unittest.TestCase):
    def chapter(self):
        scope_context = {"namespace": SCOPE, "scope_kind": "contradiction_assumption",
                         "template": {"mpl": "(>[](=[x,y])(=[x,y]))"},
                         "compact": {"mpl": "(=[x,y])"}}
        subproof_context = {"namespace": SUBPROOF, "parent_namespace": SCOPE,
                            "definition_head": "implication99",
                            "template": {"mpl": "(>[](=[x,y])(>[w](=[w,x])(=[w,x])))"},
                            "compact": {"mpl": "(implication99[x])"}}
        reductio_context = {"namespace": REDUCTIO, "parent_namespace": SCOPE,
                            "scope_kind": "contradiction_assumption",
                            "template": {"mpl": "(>[](=[x,y])(>[](=[y,z])(=[y,z])))"},
                            "compact": {"mpl": "(=[y,z])"}}
        steps = []

        def row(expression, action, namespace, dependencies=(), **metadata):
            index = len(steps) + 1
            step = {"id": f"chapter_1_line_{index}", "source_line": index,
                    "namespace": namespace, "action": action, "source_tag": action,
                    "conclusion": {"mpl": expression}, "dependencies": list(dependencies),
                    **metadata}
            steps.append(step)
            return {"step": step["id"], "namespace": namespace,
                    "source_kind": "local_step", "expression": step["conclusion"]}

        premise = row("(=[w,x])", "integration_premise", SUBPROOF,
                      [{"namespace": SCOPE, "source_kind": "integration_goal_template",
                        "expression": {"mpl": "(>[w](=[w,x])(=[w,x]))"}}],
                      scope_namespace=SUBPROOF)
        row("(implication99[x])", "integration_discharge", SCOPE, [premise],
            scope_context=subproof_context)
        row("(=[y,z])", "assume", REDUCTIO, scope_namespace=REDUCTIO)
        return {"steps": steps,
                "variable_types": {name: "element" for name in ("x", "y", "z", "w")},
                "scope_contexts": [scope_context, subproof_context, reductio_context]}

    def render(self):
        lines, manifest = _render_chapter_steps(self.chapter(), "scopes_inside_contradiction",
                                                {}, {}, ambient_variables={"x", "y", "z"})
        self.assertEqual(len(manifest), 3)
        return "\n".join(lines)

    def test_replay_enters_the_opening_scope_first(self):
        proof = self.render()
        self.assertIn("exact scope_premise_2", proof)
        self.assertIn("using (row_1 scope_premise_1)", proof)

    def test_a_template_cited_away_from_the_parent_is_rejected(self):
        chapter = self.chapter()
        chapter["steps"][0]["dependencies"][0]["namespace"] = REDUCTIO
        with self.assertRaises(AssertionError):
            _render_chapter_steps(chapter, "scopes_inside_contradiction", {}, {},
                                  ambient_variables={"x", "y", "z"})

    def test_a_discharge_outside_the_opening_scope_is_rejected(self):
        chapter = self.chapter()
        chapter["steps"][1]["namespace"] = "main"
        with self.assertRaises(AssertionError):
            _render_chapter_steps(chapter, "scopes_inside_contradiction", {}, {},
                                  ambient_variables={"x", "y", "z"})

    def test_kernel_checks_the_nested_premise_discharge_and_assumption(self):
        source = ("def gl_implication99 {α : Type} (x : α) : Prop := ∀ (w : α), w = x → w = x\n"
                  "example {α : Type} (x y z : α) : True := by\n"
                  + self.render() + "\n  trivial\n")
        with tempfile.TemporaryDirectory(prefix="scopes_inside_contradiction_") as directory:
            path = Path(directory) / "ScopesInsideContradiction.lean"
            path.write_text(source, encoding="utf-8")
            result = subprocess.run([lean_executable(), str(path)],
                cwd=LEAN_PROJECT, env=lean_environment(),
                text=True, encoding="utf-8", stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + "\n" + source)


if __name__ == "__main__":
    unittest.main()
