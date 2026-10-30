# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt)
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Kernel-check conjunction construction inside recorded implication scopes."""

from pathlib import Path
import subprocess
import tempfile
import unittest

from proof_export.lean import _render_chapter_steps
from tests.lean_toolchain import LEAN_PROJECT, lean_environment, lean_executable


class IntegrationConjoinScopeTests(unittest.TestCase):
    def render(self, scoped):
        namespace = "main_boundary_(implication99[R])" if scoped else "main"
        rule = "(>[](=[x,a])(>[](=[y,b])(pair[x,a,y,b])))"
        chapter = {
            "steps": [{
                "id": "chapter_1_line_1", "source_line": 1,
                "namespace": namespace, "action": "integration_conjoin",
                "source_tag": "reformulation for integration and",
                "conclusion": {"mpl": rule},
                "reformulation_definition": "pair",
                "dependencies": [{"namespace": namespace,
                    "source_kind": "integration_goal_template",
                    "expression": {"mpl": "(&(=[x,a])(=[y,b]))"}}],
            }],
            "variable_types": {**{v: "element" for v in ("x", "y", "a", "b")},
                               "R": "binary_relation"},
            "scope_contexts": ([{"namespace": namespace, "parent_namespace": "main",
                "compact": {"mpl": "(implication99[R])"},
                "template": {"mpl": "(>[x,y](in2[x,y,R])(=[x,y]))"}}] if scoped else []),
        }
        ambient = {"a", "b", "R"} if scoped else {"x", "y", "a", "b", "R"}
        lines, manifest = _render_chapter_steps(chapter, "conjoin", {}, {},
            ambient_variables=ambient,
            binary_definitions={"pair": {"category": "and",
                "signature": "(pair[u_1,u_2,u_3,u_4])",
                "elements": ["(=[u_1,u_2])", "(=[u_3,u_4])"]}})
        self.assertEqual(len(manifest), 1)
        self.assertEqual(manifest[0]["status"], "emitted")
        return "\n".join(lines)

    def test_scope_is_entered_before_local_conjunction_premises(self):
        proof = self.render(True)
        for token in ("intro x", "intro y", "intro scope_premise_1"):
            self.assertLess(proof.index(token), proof.index("intro integration_premise_1"))
        self.assertIn("exact ⟨integration_premise_1, integration_premise_2⟩", proof)
        self.assertNotIn("scope_premise_1, integration_premise", proof)

    def test_main_scope_has_no_extra_scope_introduction(self):
        proof = self.render(False)
        self.assertNotIn("scope_premise", proof)
        self.assertIn("exact ⟨integration_premise_1, integration_premise_2⟩", proof)

    def test_kernel_checks_main_and_quantified_membership_scopes(self):
        source = """abbrev GLBinaryRelation (α : Type) := α → α → Prop
 def gl_pair {α : Type} (x a y b : α) : Prop := x = a ∧ y = b
"""
        for scoped in (False, True):
            variables = "" if scoped else " (x y : α)"
            result = "∀ x y, R x y → " if scoped else ""
            result += "x = a → y = b → gl_pair x a y b"
            source += (f"example {{α : Type}} (a b : α) (R : GLBinaryRelation α){variables} : {result} := by\n"
                       + self.render(scoped) + "\n  exact row_1\n\n")
        with tempfile.TemporaryDirectory(prefix="integration_conjoin_") as directory:
            path = Path(directory) / "IntegrationConjoin.lean"
            path.write_text(source, encoding="utf-8")
            result = subprocess.run([lean_executable(), str(path)],
                cwd=LEAN_PROJECT, env=lean_environment(), text=True, encoding="utf-8",
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout)


if __name__ == "__main__":
    unittest.main()
