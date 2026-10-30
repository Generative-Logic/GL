# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""A definition expansion of a scope witness projection unfolds the witness guard."""
from pathlib import Path
import subprocess
import tempfile
import unittest

from proof_export.lean import _render_chapter_steps
from tests.lean_toolchain import LEAN_PROJECT, lean_environment, lean_executable


EXISTENCE = "!(>[w](in[w,N])!(twice[w,N]))"
COMPACT = "(implication99[a,b,N])"
TEMPLATE = "(>[](=[a,b])(in[a,N]))"
CHILD = "main_boundary_(implication99[a,b,N])"
DEFINITIONS = {
    "twice": {"arity": 2, "category": "and", "elements": ["(in[u_1,u_2])", "(in[u_1,u_2])"]},
    "implication99": {"arity": 3, "category": "implication",
                      "elements": ["(=[u_1,u_2])", "(in[u_1,u_3])"]},
}


def chapter(project_membership_first: bool) -> dict[str, object]:
    """
    @brief Build a scoped chapter whose witness projection is expanded.
    @details
    The existence `some w in N with twice(w, N)` is inherited into an
    implication scope, its witness is projected there (optionally after a
    membership projection of the same witness, so the witness bundle becomes a
    conjunction), the projected `twice(w, N)` is expanded by definition, and a
    conjunct of the expansion is projected.
    @param project_membership_first Whether `in(w, N)` is projected before
    `twice(w, N)`.
    @return Neutral certificate chapter for `_render_chapter_steps`.
    """

    steps: list[dict[str, object]] = []

    def row(expression, action, namespace, dependencies):
        step = {"id": f"chapter_1_line_{len(steps) + 1}", "source_line": len(steps) + 1,
                "source_tag": action, "action": action, "namespace": namespace,
                "conclusion": {"mpl": expression}, "dependencies": dependencies}
        steps.append(step)
        return {"step": step["id"], "namespace": namespace,
                "expression": step["conclusion"], "source_kind": "local_step"}

    source = row(EXISTENCE, "assume", "main", [])
    local_source = row(EXISTENCE, "scope_inherit", CHILD, [source])
    if project_membership_first:
        row("(in[w,N])", "compound_project", CHILD, [local_source])
    projection = row("(twice[w,N])", "compound_project", CHILD, [local_source])
    expansion = row("(&(in[w,N])(in[w,N]))", "definition_expand", CHILD, [projection])
    row("(in[w,N])", "compound_project", CHILD, [expansion])
    context = {"namespace": CHILD, "parent_namespace": "main", "definition_head": "implication99",
               "compact": {"mpl": COMPACT}, "template": {"mpl": TEMPLATE}}
    return {"steps": steps, "scope_contexts": [context],
            "variable_types": {"N": "set", "a": "element", "b": "element", "w": "element"}}


class ScopedWitnessExpansionTests(unittest.TestCase):
    def test_kernel_checks_expansion_of_projected_witness_under_its_guard(self):
        source = ("def gl_twice {α : Type} (w : α) (N : α → Prop) : Prop := N w ∧ N w\n")
        for project_membership_first, guard in ((False, "witness_guard_1"),
                                                (True, "witness_guard_1.2")):
            lines, manifest = _render_chapter_steps(
                chapter(project_membership_first), "scoped_witness_expansion",
                {EXISTENCE: "given_witness"}, {},
                ambient_variables={"a", "b"}, binary_definitions=DEFINITIONS)
            rendered = "\n".join(lines)
            self.assertEqual(len(manifest), 6 if project_membership_first else 5)
            self.assertIn(f"    simpa only [gl_twice] using ({guard})", rendered)
            self.assertNotIn("    simpa only [gl_twice] using (row_", rendered)
            source += ("example {α : Type} (N : α → Prop) (a b : α)\n"
                       " (given_witness : ¬ (∀ w, N w → ¬ gl_twice w N)) : True := by\n"
                       + rendered + "\n  trivial\n")
        with tempfile.TemporaryDirectory(prefix="scoped_witness_expansion_") as directory:
            path = Path(directory) / "ScopedWitnessExpansion.lean"
            path.write_text(source, encoding="utf-8")
            result = subprocess.run(
                [lean_executable(), str(path)], cwd=LEAN_PROJECT, env=lean_environment(),
                text=True, encoding="utf-8", stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + "\n" + source)

    def test_expansion_of_an_unguarded_witness_is_rejected(self):
        data = chapter(False)
        expansion = data["steps"][3]
        assert expansion["action"] == "definition_expand"
        expansion["conclusion"] = {"mpl": "(&(in[a,N])(in[a,N]))"}
        with self.assertRaises(AssertionError):
            _render_chapter_steps(data, "bad", {EXISTENCE: "given_witness"}, {},
                                  ambient_variables={"a", "b"}, binary_definitions=DEFINITIONS)


if __name__ == "__main__":
    unittest.main()
