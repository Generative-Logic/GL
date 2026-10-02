# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt)
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Direct tests of main.py's native unit-test gate route selection."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import main


class UnitTestGateTests(unittest.TestCase):
    """The gate passes the run's Phase 2 route to the native harness."""

    def _gate_command(self, route: str, returncode: int = 0) -> list[str]:
        calls: list[list[str]] = []

        def fake_run(command, check):
            calls.append(list(command))
            return SimpleNamespace(returncode=returncode)

        with mock.patch.object(main.os.path, "exists", return_value=True), \
                mock.patch.object(main.subprocess, "run", side_effect=fake_run):
            main._run_unit_test_gate(route)
        self.assertEqual(len(calls), 1)
        return calls[0]

    def test_processor_run_selects_the_processor_gate(self) -> None:
        command = self._gate_command("cpu")
        self.assertEqual(command[1:], ["--unit-tests", "--phase2-backend", "cpu"])

    def test_gpu_run_selects_the_cuda_gate(self) -> None:
        command = self._gate_command("cuda")
        self.assertEqual(command[1:], ["--unit-tests", "--phase2-backend", "cuda"])

    def test_failing_gate_exits_with_the_harness_code(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            self._gate_command("cpu", returncode=3)
        self.assertEqual(raised.exception.code, 3)

    def test_unknown_route_is_refused(self) -> None:
        with self.assertRaises(AssertionError):
            main._run_unit_test_gate(None)


if __name__ == "__main__":
    unittest.main()
