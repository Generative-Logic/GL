# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Book permutation and exact Lean-selection contracts."""
import unittest
from unittest.mock import patch
from proof_export import live
from proof_export.certificate import selected_source_indices
from proof_export.graph import TheoremRecord
from proof_export.mpl import parse_mpl
from proof_processing import arrange_anchor_stretches, order_main_book
from process_proof_graphs import _build_fname_list
P = "(>[N](AnchorPeano[N])(in[x,N]))"
Q = "(>[N](AnchorPeano[N])(in[y,N]))"
G = "(>[N](AnchorGauss[N])(in[z,N]))"


def peano(name):
    """Distinct Peano theorem text for a book-order fixture."""
    return f"(>[N](AnchorPeano[N])(in[{name},N]))"


def cites(theorem, *sources):
    """Chapter rows: the theorem's goal row, then one theorem row per cited source."""
    return [[theorem, "main", "premise"]] + [[source, "main", "theorem"] for source in sources]

class BookTests(unittest.TestCase):
    def test_dependencies_and_bodies_follow_permutation(self):
        theorems = [[Q,"single chapter proof","-1"],[P,"single chapter proof","-1"]]
        groups = _build_fname_list(theorems)
        stacks = {groups[0][0]:[[Q,"main","implication",P,"main"],[P,"main","theorem"]],
                  groups[1][0]:[[P,"main","premise"]]}
        ordered, result = order_main_book(theorems, stacks, groups)
        self.assertEqual(ordered, list(reversed(theorems)))
        self.assertEqual(result[groups[0][0]], stacks[groups[1][0]])
        self.assertEqual(result[groups[1][0]], stacks[groups[0][0]])

    def test_equivalent_copies_keep_canonical_identity(self):
        theorems = [[Q,"single chapter proof","-1"],[P,"single chapter proof","-1"],
                    [P,"single chapter proof","-1"]]
        groups = _build_fname_list(theorems)
        stacks = {groups[0][0]:[[Q,"main","premise"]],
                  groups[1][0]:[[P,"main","implication",Q,"main"],[Q,"main","theorem"]],
                  groups[2][0]:[[P,"main","premise"]]}
        ordered, result = order_main_book(theorems,stacks,groups)
        self.assertEqual(ordered,theorems); self.assertEqual(result,stacks)

    def test_induction_bundle_preserves_all_parts(self):
        theorems = [[Q,"single chapter proof","-1"],[P,"induction","n"]]
        groups = _build_fname_list(theorems)
        stacks = {groups[0][0]:[[P,"main","theorem"]]}
        stacks.update({name:[[P,"main","premise"]] for name in groups[1]})
        ordered,result = order_main_book(theorems,stacks,groups)
        self.assertEqual(ordered[0],theorems[1]);self.assertEqual(list(result),sum(_build_fname_list(ordered),[]))

    def test_cycles_and_mixed_books_assert(self):
        theorems = [[Q,"single chapter proof","-1"],[P,"single chapter proof","-1"]]
        groups = _build_fname_list(theorems)
        stacks = {groups[0][0]:[[P,"main","theorem"]],groups[1][0]:[[Q,"main","theorem"]]}
        with self.assertRaisesRegex(AssertionError,"cyclic"):order_main_book(theorems,stacks,groups)
        theorems[0][0] = Q.replace("AnchorPeano","AnchorIncubator")
        with self.assertRaisesRegex(AssertionError,"mixed"):order_main_book(theorems,stacks,groups)

    def test_unsupported_book_is_unchanged(self):
        theorems = [[Q.replace("AnchorPeano","AnchorIncubator"),"single chapter proof","-1"]]
        stacks = {"0_single_chapter_proof.txt":[[P,"main","premise"]]}
        self.assertEqual(order_main_book(theorems,stacks,_build_fname_list(theorems)),(theorems,stacks))

    def test_deepest_theorem_ends_its_stretch(self):
        # The reported FTA tail: restatements nothing cites must not follow the main result.
        l1, x1, l2, x2, f = (peano(name) for name in "abcde")
        theorems = [[text, "single chapter proof", "-1"] for text in (l1, x1, l2, x2, f)]
        groups = _build_fname_list(theorems)
        rows = (cites(l1), cites(x1, l1), cites(l2), cites(x2, l2), cites(f, l1, l2))
        stacks = {group[0]: chapter for group, chapter in zip(groups, rows)}
        ordered, result = order_main_book(theorems, stacks, groups)
        texts = [theorem[0] for theorem in ordered]
        self.assertEqual(texts[-1], f)
        for consumer, sources in ((x1, [l1]), (x2, [l2]), (f, [l1, l2])):
            self.assertTrue(all(texts.index(source) < texts.index(consumer) for source in sources))
        self.assertEqual(sorted(map(tuple, ordered)), sorted(map(tuple, theorems)))
        self.assertEqual(sorted(map(str, result.values())), sorted(map(str, stacks.values())))

    def test_chapters_never_leave_their_anchor_stretch(self):
        # A whole-book compaction would pull the Gauss chapter next to the Peano chapter it cites.
        p1, p2, p3 = (peano(name) for name in "abc")
        theorems = [[text, "single chapter proof", "-1"] for text in (p1, p2, p3, G)]
        groups = _build_fname_list(theorems)
        rows = (cites(p1), cites(p2), cites(p3), cites(G, p1))
        stacks = {group[0]: chapter for group, chapter in zip(groups, rows)}
        ordered, _ = order_main_book(theorems, stacks, groups)
        self.assertEqual([theorem[0] for theorem in ordered][-1], G)


class AnchorStretchTests(unittest.TestCase):
    def test_each_stretch_is_compacted_and_ends_with_its_deepest_theorem(self):
        result = arrange_anchor_stretches(
            [0, 1, 2, 3, 4], ["A", "A", "A", "B", "B"], ["t0", "t1", "t2", "t3", "t4"],
            {(0, 2), (1, 2), (0, 3)}, set())
        self.assertEqual(sorted(result[:2]), [0, 1])
        # Vertex 3 rests on vertex 0 across the stretch boundary, vertex 4 on nothing.
        self.assertEqual(result[2:], [2, 4, 3])

    def test_depth_ties_go_to_the_larger_text_and_precedence_holds(self):
        self.assertEqual(arrange_anchor_stretches([0, 1], ["A", "A"], ["b", "a"], set(), set()), [1, 0])
        result = arrange_anchor_stretches([0, 1, 2], ["A"] * 3, ["x", "y", "z"], {(0, 2)}, {(0, 1)})
        self.assertEqual(result[0], 0)
        self.assertEqual(result[-1], 2)
        bounded = arrange_anchor_stretches([0, 1, 2], ["A"] * 3, ["x", "y", "z"], {(0, 2)}, {(0, 1)},
                                           search_window=1)
        self.assertEqual(bounded, result)

    def test_unsorted_input_bad_window_and_text_ties_assert(self):
        with self.assertRaisesRegex(AssertionError, "not citation-sorted"):
            arrange_anchor_stretches([1, 0], ["A", "A"], ["x", "y"], {(0, 1)}, set())
        with self.assertRaisesRegex(AssertionError, "not citation-sorted"):
            arrange_anchor_stretches([1, 0], ["A", "A"], ["x", "y"], set(), {(0, 1)})
        with self.assertRaises(AssertionError):
            arrange_anchor_stretches([0, 1], ["A", "A"], ["x", "y"], set(), set(), search_window=0)
        with self.assertRaisesRegex(AssertionError, "share one text"):
            arrange_anchor_stretches([0, 1], ["A", "A"], ["x", "x"], set(), set())

class ExplicitSelectionTests(unittest.TestCase):
    def setUp(self):
        self.records = tuple(TheoremRecord(i,parse_mpl(text),"single chapter proof","-1",())
                             for i,text in enumerate((P,G,Q)))
        self.selection = {"source_indices":[0,2],"complete_anchor":"AnchorPeano","excluded_sources":[]}

    def test_interleaved_complete_anchor_and_legacy_range(self):
        self.assertEqual(selected_source_indices(self.selection,self.records),[0,2])
        self.assertEqual(selected_source_indices({"source_range":{"first":0,"last":2},
                          "excluded_sources":[{"source_index":1}]},self.records),[0,2])
        self.assertEqual(live.theorem_anchor_indices(self.records),{"AnchorPeano":[0,2],"AnchorGauss":[1]})
        self.assertEqual(live.canonical_sources(self.records,[0,2]),[0,2])
        with patch("proof_export.live.find_theorem",side_effect=lambda records,text,method,index:records[index]):
            parsed = live.parsed_sources(self.records,[0,2])
        self.assertEqual(parsed,[self.records[0],self.records[2]])
        self.assertEqual(live.imports_outside_sources(self.records,parsed,[0,2]),[])

    def test_invalid_membership_asserts(self):
        for indices in ([0],[0,1,2],[0,2,2],[2,0],[0,3],[-1,0],[],[False,2]):
            with self.subTest(indices=indices),self.assertRaises(AssertionError):
                selected_source_indices(dict(self.selection,source_indices=indices),self.records)
        for update in ({"source_range":{"first":0,"last":2}},
                       {"excluded_sources":[{"source_index":2}]},{"complete_anchor":"AnchorUnknown"}):
            with self.subTest(update=update),self.assertRaises(AssertionError):
                selected_source_indices(dict(self.selection,**update),self.records)

    def test_helper_preconditions(self):
        for helper in (live.parsed_sources,live.canonical_sources):
            for indices in ([],[0,0],[2,0],[3]):
                with self.subTest(helper=helper.__name__,indices=indices),self.assertRaises(AssertionError):
                    helper(self.records,indices)
        with self.assertRaises(AssertionError):live.theorem_anchor_indices(tuple(reversed(self.records)))
        with self.assertRaises(AssertionError):live.imports_outside_sources(self.records,[],[0])

if __name__ == "__main__":unittest.main()
