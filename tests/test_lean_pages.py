# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt)
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Tests for the Lean twin pages of the HTML proof graph (``lean_pages``)."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lean_pages  # noqa: E402


MODULE = """/- license -/

namespace GLExport

universe u

private theorem peano_source_001_check_zero
    {α : Type u}
    : True := by
  -- chapter_1_line_1: GL tag task formulation.
  exact trivial

theorem peano_source_001
    {α : Type u}
    : True := by
  exact trivial  -- done

theorem peano_source_010
    {α : Type u}
    : True := by
  exact trivial

end GLExport
"""


class SliceTests(unittest.TestCase):
    def test_public_theorem_and_its_private_helpers_are_kept_together(self) -> None:
        source = lean_pages.slice_theorem_source(MODULE, "peano_source_001")
        self.assertIn("private theorem peano_source_001_check_zero", source)
        self.assertIn("theorem peano_source_001\n", source)
        self.assertNotIn("peano_source_010", source)
        self.assertTrue(source.endswith("exact trivial  -- done\n"))

    def test_last_theorem_ends_at_the_module_end(self) -> None:
        source = lean_pages.slice_theorem_source(MODULE, "peano_source_010")
        self.assertEqual(source, "theorem peano_source_010\n    {α : Type u}\n    : True := by\n  exact trivial\n")

    def test_unknown_theorem_asserts(self) -> None:
        with self.assertRaises(AssertionError):
            lean_pages.slice_theorem_source(MODULE, "peano_source_999")


class HighlightTests(unittest.TestCase):
    def test_comments_and_keywords_are_marked_and_html_is_escaped(self) -> None:
        rendered = lean_pages.highlight_lean("  exact f <x>  -- tag <b>\n")
        self.assertIn('<span class="lk">exact</span>', rendered)
        self.assertIn('<span class="lc">-- tag &lt;b&gt;</span>', rendered)
        self.assertIn("&lt;x&gt;", rendered)
        self.assertNotIn("<b>", rendered)

    def test_keywords_inside_comments_are_not_marked(self) -> None:
        rendered = lean_pages.highlight_lean("-- exact here\n")
        self.assertEqual(rendered, '<span class="lc">-- exact here</span>')


def _write_export(root: Path, graph_hash: str, kernel_exit: int = 0) -> None:
    export = root / "full" / lean_pages.EXPORT_FOLDER
    (export / "certificates" / "peano").mkdir(parents=True)
    (export / "GLExport" / "Generated").mkdir(parents=True)
    (export / lean_pages.KERNEL_LOG).write_text(f"$ lake build\nexit {kernel_exit}\n", encoding="utf-8")
    (export / lean_pages.TOOLCHAIN_FILE).write_text("leanprover/lean4:v4.30.0-rc2\n", encoding="utf-8")
    (export / "GLExport" / "Generated" / "Peano.lean").write_text(MODULE, encoding="utf-8")
    (export / "certificates" / "peano" / "certificate.json").write_text(
        json.dumps({"source": {"theorem_list_sha256": graph_hash}}), encoding="utf-8",
    )
    (export / "lean_manifest.json").write_text(json.dumps({"theorems": [
        {"id": "peano_source_001", "method": "induction", "source_index": 1, "chapters": [
            {"role": "check_zero", "source_chapter": "1_check_zero.txt", "dispositions": [
                {"status": "emitted"}, {"status": "emitted"}]}]},
        {"id": "peano_source_010", "method": "direct", "source_index": 10, "chapters": [
            {"role": "direct_proof", "source_chapter": "12_direct_proof.txt", "dispositions": [{"status": "emitted"}]}]},
    ]}), encoding="utf-8")


class LoadTests(unittest.TestCase):
    def test_export_of_this_graph_is_read_and_a_foreign_corpus_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proc = root / "proc"
            proc.mkdir()
            (proc / "global_theorem_list.txt").write_bytes(b"row\n")
            graph_hash = hashlib.sha256(b"row\n").hexdigest().upper()
            _write_export(root, graph_hash)
            view = lean_pages.load_lean_export(root / "full", proc)
            self.assertIsNotNone(view)
            self.assertEqual(sorted(view.theorems), [1, 10])
            self.assertEqual(view.row_facts, 3)
            self.assertEqual(view.theorems[1].chapters, (("check_zero", "1_check_zero.txt", 2),))
            self.assertIn("private theorem peano_source_001_check_zero", view.theorems[1].source)
            # A folder whose corpora were built from another graph holds nothing for this one.
            (proc / "global_theorem_list.txt").write_bytes(b"other\n")
            with self.assertRaises(AssertionError):
                lean_pages.load_lean_export(root / "full", proc)

    def test_absent_export_means_no_pages_and_a_failed_log_asserts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proc = root / "proc"
            proc.mkdir()
            (proc / "global_theorem_list.txt").write_bytes(b"row\n")
            (root / "full").mkdir()
            self.assertIsNone(lean_pages.load_lean_export(root / "full", proc))
            _write_export(root, hashlib.sha256(b"row\n").hexdigest().upper(), kernel_exit=1)
            with self.assertRaises(AssertionError):
                lean_pages.load_lean_export(root / "full", proc)


class RenderTests(unittest.TestCase):
    def test_links_follow_presence(self) -> None:
        view = lean_pages.LeanTheoremView(1, "peano_source_001", "direct", "m.lean", "man.json", "c.json", "theorem x", ())
        export = lean_pages.LeanExportView("lean_export", "tc", 1, {1: view})
        self.assertEqual(lean_pages.nav_link(export, 1, 7), "<a href='chapter7_lean.html'>Lean 4 proof</a>")
        self.assertEqual(lean_pages.nav_link(export, 2, 8), "")
        self.assertEqual(lean_pages.nav_link(None, 1, 7), "")
        self.assertIn("chapter7_lean.html", lean_pages.toc_link(export, 1, 7))
        self.assertEqual(lean_pages.render_index_summary(None), "")
        self.assertIn("1 theorems and 1 named row facts", lean_pages.render_index_summary(export))

    def test_page_carries_the_source_and_the_links(self) -> None:
        view = lean_pages.LeanTheoremView(
            1, "peano_source_001", "direct", "GLExport/Generated/Peano.lean", "lean_manifest.json",
            "certificates/peano/certificate.json", "theorem peano_source_001 : True := by\n  exact trivial\n",
            (("direct_proof", "1_direct_proof.txt", 2),),
        )
        export = lean_pages.LeanExportView("lean_export", "leanprover/lean4:v4.30.0-rc2", 2, {1: view})
        page = lean_pages.render_lean_page(
            view, export, 7, "chapter7.html", "(&gt;[N](in[N]))", "readable", "<style></style>", "<!-- l -->", "  <!-- m -->", "  <div>f</div>",
        )
        for needle in (
            "chapter7.html", "lean_export/GLExport/Generated/Peano.lean", "lean_export/certificates/peano/certificate.json",
            "lean_export/lean_manifest.json", "lean_export/kernel_check.log", "peano_source_001",
            '<span class="lk">theorem</span>', "v4.30.0-rc2", "2 row facts",
        ):
            self.assertIn(needle, page)


if __name__ == "__main__":
    unittest.main()
