# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt)
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""An equality1 row at main whose statement has a free variable.

A main row is universally closed over its free variables. The processor
route's FTA proof rewrote `(implication7[V5,V1])`, which a validity-name row
states for every V5, into `(implication7[V5,V4])` by the equality
`(=[V1,V4])`, at main. The equality1 rendering applied the source to V5
without introducing V5, and Lean rejected the module ("Unknown identifier
`V5`"). This chapter mirrors that in miniature: `(in[V,A])` for every V, the
equality `(=[A,B])`, and the equality1 row `(in[V,B])`.
"""

from pathlib import Path
import subprocess
import tempfile
import unittest

from proof_export.lean import _render_chapter_steps
from tests.lean_toolchain import LEAN_PROJECT, lean_environment, lean_executable


def equality1_chapter():
    """
    @brief Build a main chapter whose equality1 row is quantified over V.
    @return Neutral certificate chapter for `_render_chapter_steps`.
    """
    steps = []

    def row(expression, action, dependencies):
        step = {"id": f"chapter_1_line_{len(steps) + 1}", "source_line": len(steps) + 1,
                "source_tag": action, "action": action, "namespace": "main",
                "conclusion": {"mpl": expression, "proof_scope": "ordinary"},
                "dependencies": dependencies}
        steps.append(step)
        return {"step": step["id"], "namespace": "main", "expression": step["conclusion"],
                "source_kind": "local_step"}

    member = row("(in[V,A])", "assume", [])
    equality = row("(=[A,B])", "assume", [])
    row("(in[V,B])", "equality_substitute", [member, equality])
    return {"steps": steps, "scope_contexts": [],
            "variable_types": {"A": "set", "B": "set", "V": "element"}}


class MainEquality1PrefixTests(unittest.TestCase):
    def render(self):
        """
        @brief Render the fixture chapter as Lean facts.
        @return The rendered lines, joined, and the manifest dispositions.
        """
        lines, manifest = _render_chapter_steps(
            equality1_chapter(), "main_equality1_prefix",
            {"(in[V,A])": "given_member", "(=[A,B])": "given_equal"}, {},
            ambient_variables={"A", "B"})
        return "\n".join(lines), manifest

    def test_the_row_introduces_its_variable_before_citing_the_source(self):
        rendered, manifest = self.render()
        self.assertEqual(len(manifest), 3)
        self.assertIn("  have row_3 : (∀ (V : α), (B V)) := by\n"
                      "    intro V\n"
                      "    have equality_source := row_1 V\n"
                      "    have equality_step_1 := row_2\n"
                      "    cases equality_step_1\n"
                      "    exact equality_source", rendered)

    def test_kernel_accepts_the_chapter(self):
        rendered, _ = self.render()
        support = (Path(__file__).resolve().parents[1] / "lean_export/GLExport/ProofSupport.lean"
                   ).read_text(encoding="utf-8")
        source = (support + "\nopen GLExport\n"
                  "example {α : Type} (A B : GLSet α) (given_member : ∀ (V : α), A V)\n"
                  "    (given_equal : A = B) : True := by\n"
                  + rendered + "\n  trivial\n")
        with tempfile.TemporaryDirectory(prefix="main_equality1_") as directory:
            path = Path(directory) / "MainEquality1.lean"
            path.write_text(source, encoding="utf-8")
            result = subprocess.run([lean_executable(), str(path)], cwd=LEAN_PROJECT, env=lean_environment(),
                                    text=True, encoding="utf-8", stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + "\n" + rendered)


if __name__ == "__main__":
    unittest.main()
