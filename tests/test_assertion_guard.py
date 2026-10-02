# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt)
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Every GL entry point refuses to run with Python's assert statements removed."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from assertion_guard import require_assertions  # noqa: E402

ENTRY_POINTS = ("main.py", "verifier.py", "incubator_to_simple_facts.py",
                "proof_export/export_lean.py")
PROBE = "import assertion_guard; assertion_guard.require_assertions('probe')"


def _python(arguments: list[str], **environment: str) -> subprocess.CompletedProcess:
    """Run the current Python on `arguments` from the repository root."""
    env = {key: value for key, value in os.environ.items() if key != "PYTHONOPTIMIZE"}
    env.update(environment)
    return subprocess.run([sys.executable, *arguments], cwd=REPO_ROOT, env=env,
                          capture_output=True, text=True)


class RequireAssertionsTests(unittest.TestCase):
    def test_enabled_assertions_return(self) -> None:
        self.assertIsNone(require_assertions("probe"))

    def test_optimized_python_exits_with_status_2(self) -> None:
        for flag in ("-O", "-OO"):
            completed = _python([flag, "-c", PROBE])
            self.assertEqual(completed.returncode, 2, flag)
            self.assertIn("probe: GL's checks are Python assert statements", completed.stderr)

    def test_pythonoptimize_exits_with_status_2(self) -> None:
        completed = _python(["-c", PROBE], PYTHONOPTIMIZE="1")
        self.assertEqual(completed.returncode, 2)
        self.assertIn("probe:", completed.stderr)

    def test_entry_points_refuse_optimized_python(self) -> None:
        """Run as a user would; a missing check would print the version or help and exit 0."""
        for arguments in (["main.py", "--version"], ["verifier.py", "--help"],
                          ["-m", "proof_export.export_lean", "--help"]):
            completed = _python(["-O", *arguments])
            self.assertEqual(completed.returncode, 2, arguments)
            self.assertIn("python -O", completed.stderr)

    def test_every_entry_point_checks_before_any_work(self) -> None:
        """After its imports, each `__main__` block first calls `require_assertions`."""
        for relative in ENTRY_POINTS:
            tree = ast.parse((REPO_ROOT / relative).read_text(encoding="utf-8"))
            blocks = [node for node in tree.body if isinstance(node, ast.If)
                      and ast.unparse(node.test) == "__name__ == '__main__'"]
            self.assertEqual(len(blocks), 1, relative)
            first = next(statement for statement in blocks[0].body
                         if not isinstance(statement, (ast.Import, ast.ImportFrom)))
            self.assertIsInstance(first, ast.Expr, relative)
            self.assertIsInstance(first.value, ast.Call, relative)
            self.assertEqual(ast.unparse(first.value.func), "require_assertions", relative)


if __name__ == "__main__":
    unittest.main()
