# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt)
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Full FTA boundary replay preserves inputs, roles and prior proofs."""

import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import fta_shortcut


class FtaShortcutTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.files = {
            "files/config/ConfigFTA.json": json.dumps({
                "external_conjectures_file": "files/shortcut/theorems/conjectures.txt"}),
            "files/definitions/Peano.mpl": "definition",
            "files/shortcut/theorems/conjectures.txt": "(goal)\n",
            "files/shortcut/theorems/externally_provided_theorems.txt": "legacy snapshot\n",
            "files/GL_binaries/GL_binary_FTA.json": '{"or17": {"category": "or"}}',
            "files/GL_binaries/GL_binary_shared.json": '{"or17": {"category": "or"}}',
            "files/simple_facts/facts.txt": "prior facts\n",
            "files/theorems/conjectures.txt": "(goal)\n",
            "files/theorems/theorems.txt": "(proved)\n",
            "files/theorems/compiled_theorems.txt": "(compiled)\t(proved)\n",
            "files/theorems/compressed_external_theorems.txt": "(external)\n",
            "files/theorems/externally_provided_theorems.txt": "(external)\n",
            "files/raw_proof_graph/Peano/chapter.txt": "prior proof\n",
        }
        for relative, content in self.files.items():
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content.encode())

    def test_sources_include_preceding_definitions_and_goals(self):
        fingerprints = fta_shortcut.source_fingerprints(self.root)
        self.assertEqual(set(fingerprints), {
            "files/config/ConfigFTA.json", "files/definitions/Peano.mpl",
            "files/shortcut/theorems/conjectures.txt"})
        (self.root / "files/definitions/Peano.mpl").write_text("changed")
        self.assertNotEqual(fingerprints, fta_shortcut.source_fingerprints(self.root))

    def test_compilation_state_is_captured_and_restored_with_the_registry(self):
        state = {"schema": 1, "next_counters": {"implication": 31, "existence": 20, "and": 4, "or": 18},
                 "replacements": {"existence16": {"target": "preorder", "arguments": [1, 4, 2, 3]}}}
        paths = [self.root / f"files/GL_binaries/GL_compilation_state_{tag}.json" for tag in ("shared", "FTA")]
        for path in paths:
            path.write_text(json.dumps(state), encoding="utf-8")
        fta_shortcut.capture(self.root)
        for path in paths:
            path.write_text("{}", encoding="utf-8")
        fta_shortcut.restore(self.root)
        for path in paths:
            self.assertEqual(json.loads(path.read_text()), state)

    def test_capture_replays_exact_bytes_and_separate_theorem_roles(self):
        fta_shortcut.capture(self.root)
        (self.root / "files/GL_binaries/stale.json").write_text("stale")
        (self.root / "files/GL_binaries/GL_binary_FTA.json").write_text("post FTA")
        fta_shortcut.restore(self.root)
        self.assertFalse((self.root / "files/GL_binaries/stale.json").exists())
        for relative, original in self.files.items():
            destination = relative
            if relative.startswith("files/theorems/"):
                if relative.endswith("/externally_provided_theorems.txt"):
                    continue
                destination = relative.replace("files/theorems/", "files/shortcut/theorems/")
                if relative.endswith("/conjectures.txt"):
                    destination = destination.replace("/conjectures.txt", "/replay_conjectures.txt")
            elif relative.startswith("files/raw_proof_graph/"):
                destination = relative.replace("files/raw_proof_graph/", "files/shortcut/raw_proof_graph/")
            self.assertEqual((self.root / destination).read_bytes(), original.encode(), destination)
        self.assertEqual((self.root / "files/theorems/theorems.txt").read_text(), "(proved)\n")
        # Each replay starts from the saved prior pool, never its own previous result.
        (self.root / "files/shortcut/theorems/theorems.txt").write_text("post replay")
        fta_shortcut.restore(self.root)
        self.assertEqual((self.root / "files/shortcut/theorems/theorems.txt").read_text(), "(proved)\n")

    def test_missing_capture_fails_without_cleanup(self):
        with self.assertRaisesRegex(AssertionError, "missing"):
            fta_shortcut.restore(self.root)
        self.assertTrue((self.root / "files/GL_binaries/GL_binary_FTA.json").is_file())

    def test_changed_inputs_fail_before_cleanup(self):
        fta_shortcut.capture(self.root)
        (self.root / "files/definitions/Peano.mpl").write_text("changed")
        with self.assertRaisesRegex(AssertionError, "inputs changed"):
            fta_shortcut.restore(self.root)
        self.assertTrue((self.root / "files/GL_binaries/GL_binary_FTA.json").is_file())

    def test_corrupted_capture_fails_before_cleanup(self):
        fta_shortcut.capture(self.root)
        archive_path = self.root / "files/shortcut/full_input.zip"
        with ZipFile(archive_path) as archive:
            entries = {name: archive.read(name) for name in archive.namelist()}
        entries["files/theorems/theorems.txt"] = b"wrong theorem"
        with ZipFile(archive_path, "w") as archive:
            for name, data in entries.items():
                archive.writestr(name, data)
        with self.assertRaisesRegex(AssertionError, "theorems.txt"):
            fta_shortcut.restore(self.root)
        self.assertTrue((self.root / "files/GL_binaries/GL_binary_FTA.json").is_file())

    def test_empty_goals_are_rejected(self):
        (self.root / "files/shortcut/theorems/conjectures.txt").write_text("")
        with self.assertRaises(AssertionError):
            fta_shortcut.source_fingerprints(self.root)

    def test_full_batch_captures_after_seed_before_native(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import run_modes
        calls = []
        config = SimpleNamespace(theorems_folder="files/theorems",
                                 raw_proof_graph_folder="files/raw_proof_graph")
        with patch.object(run_modes, "PROJECT_ROOT", self.root), \
                patch.object(run_modes, "configuration_reader", return_value=config), \
                patch.object(run_modes, "_prepare_batch_conjectures", side_effect=lambda *_: calls.append("goals")), \
                patch.object(run_modes, "_seed_per_batch_binary", side_effect=lambda *_: calls.append("seed")), \
                patch.object(fta_shortcut, "capture", side_effect=lambda *_: calls.append("capture")), \
                patch.object(run_modes, "run_gl_quick", side_effect=lambda *_, **__: calls.append("native")), \
                patch.object(run_modes.compilation_state, "migrate_batch", side_effect=lambda *_, **__: calls.append("migration") or {"retired": 0, "rewritten_files": 0}), \
                patch.object(run_modes, "_merge_into_shared", side_effect=lambda *_: calls.append("merge")), \
                patch.object(run_modes, "_filter_vacuous_tainted", side_effect=lambda *_: calls.append("vacuity")):
            run_modes._run_batch("FTA", [], add_cross_anchor=False, phase2_backend="cuda")
        self.assertEqual(calls, ["goals", "seed", "capture", "native", "migration", "merge", "vacuity"])

    def test_capture_requires_prior_registry(self):
        (self.root / "files/GL_binaries/GL_binary_FTA.json").write_text("{}")
        with self.assertRaises(AssertionError):
            fta_shortcut.capture(self.root)


if __name__ == "__main__":
    unittest.main()
