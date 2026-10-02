"""Focused tests for single-chapter proof headings in the final HTML."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from generate_full_proof_graph import single_chapter_proof_heading


class SingleChapterProofHeadingTests(unittest.TestCase):
    def _heading(self, rows: str) -> str:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "0_single_chapter_proof.txt"
            path.write_text(rows, encoding="utf-8")
            return single_chapter_proof_heading(path)

    def test_goal_contradiction_is_contradiction_proof(self):
        self.assertEqual(
            self._heading("!(p)\tmain\tcontradiction\t!(q)\tmain\t(q)\tmain\n"),
            "Contradiction Proof",
        )

    def test_ordinary_goal_is_direct_proof(self):
        self.assertEqual(
            self._heading("(p)\tmain\timplication\t(rule)\tmain\n"),
            "Direct Proof",
        )

    def test_later_contradiction_does_not_relabel_chapter(self):
        self.assertEqual(
            self._heading(
                "(p)\tmain\timplication\t(rule)\tmain\n"
                "!(q)\tmain\tcontradiction\t!(r)\tmain\t(r)\tmain\n"
            ),
            "Direct Proof",
        )

    def test_empty_chapter_asserts(self):
        with self.assertRaises(AssertionError):
            self._heading("")


if __name__ == "__main__":
    unittest.main()
