# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# ------------------------------------------------------------------------------
#
# This software is also available under a commercial license. For details,
# see: https://generative-logic.com/license
#
# Contributions to this project must be made under the terms of the
# Contributor License Agreement (CLA). See the project's CONTRIBUTING.md file.

"""A run whose verifier reports failures ends in a failed process status."""

from __future__ import annotations

import io
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import main  # noqa: E402
import run_modes  # noqa: E402


class VerifierExitStatusTests(unittest.TestCase):
    def test_failures_exit_with_status_one(self):
        with redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit) as ctx:
            main._exit_on_verifier_failures(3)
        self.assertEqual(ctx.exception.code, 1)
        self.assertIn("3 failure(s)", err.getvalue())

    def test_an_airtight_run_returns(self):
        self.assertIsNone(main._exit_on_verifier_failures(0))

    def test_a_negative_count_asserts(self):
        with self.assertRaises(AssertionError):
            main._exit_on_verifier_failures(-1)

    def test_full_run_returns_main_and_incubator_failures(self):
        """The finalization's combined failure count (main plus incubator)
        is the full run's result; expensive stages are not run."""
        with mock.patch.object(run_modes, "RUN_INCUBATOR", False), \
                mock.patch.object(run_modes, "RUN_MAIN_PATH", False), \
                mock.patch.object(run_modes, "CLEAN_RUN", False), \
                mock.patch.object(run_modes, "convert_incubator_theorems"), \
                mock.patch.object(run_modes, "_finish_full_graphs", return_value=(10, 2)) as finish, \
                redirect_stdout(io.StringIO()):
            self.assertEqual(run_modes.full_run(phase2_backend="cpu"), 2)
        finish.assert_called_once()
        with mock.patch.object(run_modes, "RUN_INCUBATOR", False), \
                mock.patch.object(run_modes, "RUN_MAIN_PATH", False), \
                mock.patch.object(run_modes, "CLEAN_RUN", False), \
                mock.patch.object(run_modes, "convert_incubator_theorems"), \
                mock.patch.object(run_modes, "_finish_full_graphs", return_value=(10, 0)), \
                redirect_stdout(io.StringIO()) as out:
            self.assertEqual(run_modes.full_run(phase2_backend="cpu"), 0)
        self.assertIn("airtight", out.getvalue())


if __name__ == "__main__":
    unittest.main()
