# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt)
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Kernel-check contradiction elimination inside ordinary implication scopes."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from proof_export.lean import _render_chapter_steps
from tests.lean_toolchain import LEAN_PROJECT, lean_environment, lean_executable


class ScopedVacuityLeanTests(unittest.TestCase):
    def render(self, same_scope):
        scope = "main_boundary_(implication99[x,y,z])"
        template = "(>[](=[x,y])" + (
            "(>[]!(=[x,y])(=[x,z])))" if same_scope else "(=[x,z]))")
        steps = []

        def premise(expression, local):
            index = len(steps) + 1
            step = {"id": f"chapter_1_line_{index}", "source_line": index,
                    "namespace": scope if local else "main",
                    "action": "integration_premise" if local else "assume",
                    "source_tag": "premise element" if local else "task formulation",
                    "conclusion": {"mpl": expression}, "dependencies": []}
            if local:
                step["scope_namespace"] = scope
                step["dependencies"] = [{"namespace": "main",
                    "source_kind": "integration_goal_template", "expression": {"mpl": template}}]
            steps.append(step)
            return {"step": step["id"], "namespace": step["namespace"],
                    "expression": step["conclusion"]}

        positive = premise("(=[x,y])", True)
        negative = premise("!(=[x,y])", same_scope)
        steps.append({"id": "chapter_1_line_3", "source_line": 3,
            "namespace": scope, "action": "ex_falso", "source_tag": "vacuous truth",
            "conclusion": {"mpl": "(=[x,z])"}, "dependencies": [positive, negative]})
        chapter = {"steps": steps,
            "variable_types": {name: "element" for name in ("x", "y", "z")},
            "scope_contexts": [{"namespace": scope, "parent_namespace": "main",
                "template": {"mpl": template}, "compact": {"mpl": "(implication99[x,y,z])"}}]}
        lines, manifest = _render_chapter_steps(chapter, "scoped_vacuity",
            {} if same_scope else {"!(=[x,y])": "opposite"}, {},
            ambient_variables={"x", "y", "z"})
        self.assertEqual(len(manifest), 3)
        return "\n".join(lines)

    def test_scope_entry_specializes_each_dependency_at_its_own_scope(self):
        for same_scope in (False, True):
            with self.subTest(same_scope=same_scope):
                proof = self.render(same_scope)
                expected = "scoped_fact_2" if same_scope else "row_2"
                self.assertIn(f"exact False.elim ({expected} scoped_fact_1)", proof)

    def test_kernel_checks_same_scope_and_ancestor_contradictions(self):
        root = Path(__file__).resolve().parent.parent
        source = ""
        for same_scope in (False, True):
            context = "" if same_scope else " (opposite : ¬ (x = y))"
            result = "x = y → " + ("¬ (x = y) → " if same_scope else "") + "x = z"
            source += (f"example {{α : Type}} (x y z : α){context} : {result} := by\n"
                + self.render(same_scope) + "\n  exact row_3\n\n")
        with tempfile.TemporaryDirectory(prefix="scoped_vacuity_") as directory:
            path = Path(directory) / "ScopedVacuity.lean"
            path.write_text(source, encoding="utf-8")
            result = subprocess.run([lean_executable(), str(path)],
                cwd=LEAN_PROJECT, env=lean_environment(),
                text=True, encoding="utf-8", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout)

    def test_kernel_checks_later_use_of_a_quantified_scope_closure(self):
        scope = "main_boundary_(implication99[x])"
        template = "(>[](=[x,x])(=[x,x]))"
        context = {"namespace": scope, "parent_namespace": "main",
            "definition_head": "implication99", "compact": {"mpl": "(implication99[x])"},
            "template": {"mpl": template}}
        rule = "(>[x](implication99[x])(=[x,x]))"
        steps = []

        def row(expression, action, namespace="main", dependencies=(), **metadata):
            index = len(steps) + 1
            step = {"id": f"chapter_1_line_{index}", "source_line": index,
                "namespace": namespace, "action": action, "source_tag": action,
                "conclusion": {"mpl": expression}, "dependencies": list(dependencies), **metadata}
            steps.append(step)
            return {"step": step["id"], "namespace": namespace,
                "source_kind": "local_step", "expression": step["conclusion"]}

        premise = row("(=[x,x])", "integration_premise", scope,
            [{"namespace": "main", "source_kind": "integration_goal_template",
              "expression": {"mpl": template}}], scope_namespace=scope)
        compact = row("(implication99[x])", "integration_discharge",
            dependencies=[premise], scope_context=context)
        implication = row(rule, "assume")
        row("(=[x,x])", "implication_apply", dependencies=[implication, compact])
        chapter = {"steps": steps, "variable_types": {"x": "element"},
            "scope_contexts": [context]}
        lines, _ = _render_chapter_steps(chapter, "quantified_closure",
            {rule: "source_assumption"}, {}, ambient_variables=set())
        proof = "\n".join(lines)
        self.assertIn("exact row_2 x", proof)
        source = ("def gl_implication99 {α : Type} (x : α) : Prop := x = x → x = x\n"
            "example {α : Type} (source_assumption : ∀ (x : α), gl_implication99 x → x = x) "
            ": ∀ (x : α), x = x := by\n" + proof + "\n  exact row_4\n")
        root = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory(prefix="quantified_closure_") as directory:
            path = Path(directory) / "QuantifiedClosure.lean"
            path.write_text(source, encoding="utf-8")
            result = subprocess.run([lean_executable(), str(path)],
                cwd=LEAN_PROJECT, env=lean_environment(),
                text=True, encoding="utf-8", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout)


if __name__ == "__main__":
    unittest.main()
