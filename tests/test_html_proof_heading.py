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

    def test_debug_chapter_selects_layout_from_its_supplied_theorem(self):
        from unittest.mock import patch
        from generate_full_proof_graph import debugging
        for anchor, hierarchical in (('AnchorPeano', True), ('AnchorIncubator', False)):
            theorem = f'(>[N]({anchor}[N])(in[x,N]))'
            rows = [['(in[x,N])', 'main', 'task formulation']]
            with patch('generate_full_proof_graph.read_stack', return_value=rows) as read:
                with patch('generate_full_proof_graph.render_stack_with_subproofs', return_value='rendered') as render:
                    self.assertEqual(debugging(theorem, 'debug.txt', 'debug_'), 'rendered')
                    read.assert_called_once_with('debug.txt', 'debugging')
                    render.assert_called_once_with(rows, 'debug_', hierarchical=hierarchical)

    def test_empty_chapter_asserts(self):
        with self.assertRaises(AssertionError):
            self._heading("")


if __name__ == "__main__":
    unittest.main()
