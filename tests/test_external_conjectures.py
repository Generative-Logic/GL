# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt)
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""External conjectures are goals; they never enter the proved theorem pool."""

import tempfile
import unittest
import contextlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import run_modes
from configuration_reader import configuration_reader


class ExternalConjectureTests(unittest.TestCase):
    def test_full_run_schedules_fta_after_gauss(self):
        calls = []
        with patch.object(run_modes, "CLEAN_RUN", False), \
                patch.object(run_modes, "RUN_INCUBATOR", False), \
                patch.object(run_modes, "RUN_MAIN_PATH", True), \
                patch.object(run_modes, "convert_incubator_theorems"), \
                patch.object(run_modes, "_run_batch", side_effect=lambda tag, **kw: calls.append((tag, kw))), \
                patch.object(run_modes, "_finish_full_graphs", return_value=(0, 0)), \
                contextlib.redirect_stdout(io.StringIO()):
            run_modes.full_run(phase2_backend="cuda")
        self.assertEqual([tag for tag, _ in calls], ["Peano", "Gauss", "FTA"])
        self.assertEqual(calls[-1][1]["prev_tags"], ["Peano", "Gauss"])
        self.assertFalse(calls[-1][1]["add_cross_anchor"])
        self.assertTrue(all(options["phase2_backend"] == "cuda" for _, options in calls))

    def test_finalization_worker_uses_utf8_even_with_a_legacy_parent_encoding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".debug").mkdir()
            def spawn(command, **options):
                encoding = options["env"]["PYTHONIOENCODING"]
                options["stdout"].buffer.write("worker — complete\n".encode(encoding))
                Path(command[-2]).write_text(json.dumps({"success": 2, "failure": 0}), encoding="utf-8")
                return SimpleNamespace(wait=lambda: 0)
            output = io.StringIO()
            with patch.object(run_modes, "PROJECT_ROOT", root), \
                    patch.dict(os.environ, {"PYTHONIOENCODING": "cp1252"}), \
                    patch.object(run_modes.subprocess, "Popen", side_effect=spawn), \
                    contextlib.redirect_stdout(output):
                self.assertEqual(run_modes._finish_full_graphs(None, False, True), (2, 0))
                self.assertEqual(os.environ["PYTHONIOENCODING"], "cp1252")
            self.assertIn("worker — complete", output.getvalue())

    def test_supplied_only_preserves_inputs_and_previous_proofs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "supplied.txt"
            source.write_text("(=[x,x])\n", encoding="utf-8")
            target = root / "pool"
            target.mkdir()
            for name in ("theorems.txt", "externally_provided_theorems.txt"):
                (target / name).write_text("previous proof\n", encoding="utf-8")
            (target / "or_pairs.txt").write_text("old batch metadata", encoding="utf-8")
            config = SimpleNamespace(generate_conjectures=False, external_conjectures_file=str(source))
            with patch.object(run_modes, "PROJECT_ROOT", root), patch.object(run_modes.subprocess, "run") as native:
                run_modes._prepare_batch_conjectures("FTA", config, target)
            native.assert_not_called()
            self.assertEqual((target / "conjectures.txt").read_text(), source.read_text())
            self.assertEqual((target / "or_pairs.txt").read_text(), "")
            for name in ("theorems.txt", "externally_provided_theorems.txt"):
                self.assertEqual((target / name).read_text(), "previous proof\n")

    def test_disabled_generation_requires_nonempty_external_input(self):
        with tempfile.TemporaryDirectory() as directory:
            config = SimpleNamespace(generate_conjectures=False, external_conjectures_file="")
            with self.assertRaisesRegex(AssertionError, "external conjectures are empty"):
                run_modes._prepare_batch_conjectures("FTA", config, Path(directory))

    def test_configured_external_file_must_exist(self):
        with tempfile.TemporaryDirectory() as directory:
            config = SimpleNamespace(generate_conjectures=True,
                                     external_conjectures_file=str(Path(directory) / "missing.txt"))
            with self.assertRaisesRegex(AssertionError, "file is missing"):
                run_modes._prepare_batch_conjectures("FTA", config, Path(directory))

    def test_existing_batches_generate_and_fta_uses_supplied_goals(self):
        root = Path(__file__).resolve().parents[1]
        for tag in ("Gauss", "IncubatorPeano1", "IncubatorPeano2",
                    "IncubatorGauss1", "IncubatorGauss2", "IncubatorGauss3"):
            config = configuration_reader(root / "files/config" / f"Config{tag}.json")
            self.assertTrue(config.generate_conjectures)
            self.assertEqual(config.external_conjectures_file, "")
        # Peano generates its conjectures and also reads the supplied lemmas.
        config = configuration_reader(root / "files/config/ConfigPeano.json")
        self.assertTrue(config.generate_conjectures)
        self.assertEqual(config.external_conjectures_file, "files/supplied_conjectures/peano.txt")
        self.assertTrue((root / config.external_conjectures_file).read_text().strip())
        config = configuration_reader(root / "files/config/ConfigFTA.json")
        self.assertFalse(config.generate_conjectures)
        self.assertTrue((root / config.external_conjectures_file).read_text().strip())
        self.assertEqual(config.theorems_folder, "files/theorems")


if __name__ == "__main__":
    unittest.main()
