# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Integration reuses its recorded witness even after a prior row bound it."""
import copy
from pathlib import Path
import subprocess
import tempfile
import unittest
from proof_export.lean import _render_chapter_steps
from tests.lean_toolchain import LEAN_PROJECT, lean_environment, lean_executable


class AvailableIntegrationWitnessTests(unittest.TestCase):
    def test_kernel_checks_fresh_and_already_available_witness(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "lean_export/GLExport/ProofSupport.lean"
        ).read_text(encoding="utf-8")
        source += "\nopen GLExport\ndef gl_some {α : Type} (N : GLSet α) : Prop := ¬ ∀ x, N x → ¬ N x\n"
        expansion = "(>[](in[x,N])(in[x,N]))"
        rule = "(>[](in[x,N])(>[](in[x,N])(some[N])))"
        chapter = {
            "variable_types": {"x": "element", "N": "set"},
            "steps": [
                {
                    "id": "chapter_1_line_2",
                    "source_line": 2,
                    "source_tag": "expansion for integration",
                    "action": "definition_expand",
                    "namespace": "main",
                    "conclusion": {"mpl": expansion, "proof_scope": "integration_goal"},
                    "dependencies": [
                        {
                            "source_kind": "integration_goal_template",
                            "namespace": "main",
                            "expression": {"mpl": "(some[N])"},
                        }
                    ],
                },
                {
                    "id": "chapter_1_line_1",
                    "source_line": 1,
                    "source_tag": "reformulation for integration >[]",
                    "action": "classical_reformulate_empty",
                    "namespace": "main",
                    "conclusion": {"mpl": rule},
                    "dependencies": [
                        {
                            "source_kind": "local_step",
                            "namespace": "main",
                            "step": "chapter_1_line_2",
                            "expression": {"mpl": expansion},
                        }
                    ],
                    "reformulation_definition": "some",
                },
            ],
        }
        definitions = {
            "some": {
                "category": "existence",
                "signature": "(some[u_1])",
                "elements": ["(in[1,u_1])", "(in[1,u_1])"],
            }
        }
        for available in (False, True):
            lines, manifest = _render_chapter_steps(
                chapter,
                "some",
                {},
                {},
                ambient_variables={"N", "x"} if available else {"N"},
                binary_definitions=definitions,
            )
            proof = "\n".join(lines)
            self.assertIn(
                "universal_counterexample x integration_premise_1 integration_premise_2",
                proof,
            )
            self.assertEqual(len(manifest), 2)
            binders = "(x : α)" if available else ""
            goal = (
                "N x → N x → gl_some N" if available else "∀ x, N x → N x → gl_some N"
            )
            source += f"example {{α : Type}} (N : GLSet α) {binders} : {goal} := by\n{proof}\n  exact row_1\n"
        broken = copy.deepcopy(chapter)
        broken["steps"][-1]["dependencies"][0]["step"] = "chapter_1_line_99"
        with self.assertRaises(AssertionError):
            _render_chapter_steps(
                broken,
                "bad",
                {},
                {},
                ambient_variables={"N", "x"},
                binary_definitions=definitions,
            )
        with tempfile.TemporaryDirectory(prefix="integration_available_") as directory:
            path = Path(directory) / "Available.lean"
            path.write_text(source, encoding="utf-8")
            result = subprocess.run(
                [lean_executable(), str(path)],
                cwd=LEAN_PROJECT,
                env=lean_environment(),
                text=True,
                encoding="utf-8",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stdout)


if __name__ == "__main__":
    unittest.main()
