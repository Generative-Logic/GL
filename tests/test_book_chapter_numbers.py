"""Book chapter numbers: one consecutive number per theorem, whatever its method (D-591)."""
import unittest
from unittest import mock

import generate_full_proof_graph as gen


class BookChapterNumbersTest(unittest.TestCase):
    def test_an_induction_theorem_takes_one_number(self):
        theorems = [("(A)", "single chapter proof", ""), ("(B)", "induction", "v1"),
                    ("(C)", "strong induction", "v1"), ("(D)", "reformulated statement", "")]
        self.assertEqual(gen.book_chapter_numbers(theorems), [0, 1, 2, 3])

    def test_an_empty_book_has_no_chapters(self):
        self.assertEqual(gen.book_chapter_numbers([]), [])

    def test_a_record_tuple_instead_of_a_list_is_refused(self):
        with self.assertRaises(AssertionError):
            gen.book_chapter_numbers(("(A)", "induction", "v1"))

    def test_used_by_names_book_chapters(self):
        theorems = [("(A)", "induction", "v1"), ("(B)", "single chapter proof", "")]
        with mock.patch.object(gen, "read_stack", return_value=[["(A)", "x", "theorem"]]), \
                mock.patch.object(gen.os.path, "exists", return_value=True):
            used_by = gen._build_used_by(theorems, {"(B)": ["b.txt"]}, gen.book_chapter_numbers(theorems))
        # Chapter 1 cites chapter 0; file numbering would have made the citing chapter 3.
        self.assertEqual(used_by, {0: {1}})

    def test_used_by_requires_one_number_per_record(self):
        with self.assertRaises(AssertionError):
            gen._build_used_by([("(A)", "induction", "v1")], {}, [0, 1])


if __name__ == "__main__":
    unittest.main()
