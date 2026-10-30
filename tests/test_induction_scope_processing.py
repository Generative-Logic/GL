# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.
"""Unit tests for deleting the prover's induction scopes from processed chapters."""

import unittest

from process_proof_graphs import (_chapter_scope_cells, _has_citation_cycle, _reachable_rows,
                                  strip_induction_scope_segments, strip_induction_scopes)

LEAF = "main_boundary_is_(in3[2,3,marker,4])_it_0_lev_2_7"
NESTED = LEAF + "_boundary_is_(in3[it_0_lev_2_7,5,marker,4])_it_0_lev_2_8"
BRANCH = "main_boundary_ordis_(or9[a,6,9])_((=[a,9]))"


class ChapterScopeCells(unittest.TestCase):
    def test_own_and_dependency_validities(self):
        self.assertEqual(_chapter_scope_cells(["(G)", "main", "implication", "(F)", LEAF, "(E)", "main"]),
                         [1, 4, 6])
        self.assertEqual(_chapter_scope_cells(["(S)", "main", "task formulation"]), [1])
        self.assertEqual(_chapter_scope_cells([]), [])


class ReachableRows(unittest.TestCase):
    def test_first_row_reaches_only_its_citations(self):
        rows = [
            ["(G)", "main", "implication", "(F)", "main"],
            [],
            ["(F)", "main", "task formulation"],
            ["(K)", "main", "task formulation"],
        ]
        keys = [(row[0], row[1]) if len(row) >= 2 else None for row in rows]
        self.assertEqual(_reachable_rows(rows, keys), {0, 2})

    def test_removed_rows_are_neither_start_nor_target(self):
        rows = [
            ["(G)", "main", "implication", "(F)", "main"],
            ["(F)", "main", "task formulation"],
        ]
        self.assertEqual(_reachable_rows(rows, [None, ("(F)", "main")]), {1})
        self.assertEqual(_reachable_rows(rows, [("(G)", "main"), None]), {0})
        self.assertEqual(_reachable_rows(rows, [None, None]), set())


class HasCitationCycle(unittest.TestCase):
    def test_shared_citations_are_no_cycle(self):
        self.assertFalse(_has_citation_cycle([
            ["(G)", "main", "implication", "(F)", "main", "(E)", "main"],
            ["(F)", "main", "implication", "(E)", "main"],
            ["(E)", "main", "task formulation"],
            [],
        ]))

    def test_mutual_and_self_citations_are_cycles(self):
        self.assertTrue(_has_citation_cycle([
            ["(G)", "main", "implication", "(F)", "main"],
            ["(F)", "main", "implication", "(G)", "main"],
        ]))
        self.assertTrue(_has_citation_cycle([["(G)", "main", "induction scope", "(G)", "main"]]))


class StripInductionScopeSegments(unittest.TestCase):
    def test_only_induction_segments_are_deleted(self):
        self.assertEqual(strip_induction_scope_segments("main"), "main")
        self.assertEqual(strip_induction_scope_segments(LEAF), "main")
        self.assertEqual(strip_induction_scope_segments(NESTED), "main")
        self.assertEqual(strip_induction_scope_segments(LEAF + "_boundary_" + BRANCH[len("main_boundary_"):]),
                         BRANCH)
        self.assertEqual(strip_induction_scope_segments(BRANCH), BRANCH)
        self.assertEqual(strip_induction_scope_segments(BRANCH + "_boundary_is_(p[marker])_x"), BRANCH)


class StripInductionScopes(unittest.TestCase):
    def test_goal_row_takes_the_leaf_derivation(self):
        chapter = [
            ["(G)", "main", "induction scope", "(G)", NESTED],
            ["(G)", NESTED, "implication", "(F)", NESTED, "(E)", "main"],
            ["(F)", NESTED, "disintegration", "(S)", LEAF],
            ["(S)", LEAF, "zoom in", "(S)", "main"],
            ["(S)", "main", "task formulation"],
            ["(E)", "main", "task formulation"],
        ]
        self.assertEqual(strip_induction_scopes(chapter), [
            ["(G)", "main", "implication", "(F)", "main", "(E)", "main"],
            ["(F)", "main", "disintegration", "(S)", "main"],
            ["(S)", "main", "task formulation"],
            ["(E)", "main", "task formulation"],
        ])

    def test_sibling_duplicates_merge_and_their_orphans_drop(self):
        first = "main_boundary_is_(p[marker])_x"
        second = "main_boundary_is_(q[marker])_y"
        chapter = [
            ["(G)", "main", "implication", "(F)", first, "(F)", second],
            ["(F)", first, "implication", "(K)", first],
            ["(K)", first, "task formulation"],
            ["(F)", second, "implication", "(L)", second],
            ["(L)", second, "task formulation"],
        ]
        self.assertEqual(strip_induction_scopes(chapter), [
            ["(G)", "main", "implication", "(F)", "main", "(F)", "main"],
            ["(F)", "main", "implication", "(K)", "main"],
            ["(K)", "main", "task formulation"],
        ])

    def test_nested_branches_move_up_with_their_scope(self):
        inner = LEAF + "_boundary_" + BRANCH[len("main_boundary_"):]
        chapter = [
            ["(G)", "main", "induction scope", "(G)", LEAF],
            ["(G)", LEAF, "or convergence", "(H)", inner],
            ["(H)", inner, "or disintegration", "(O)", LEAF],
            ["(O)", LEAF, "task formulation"],
        ]
        self.assertEqual(strip_induction_scopes(chapter), [
            ["(G)", "main", "or convergence", "(H)", BRANCH],
            ["(H)", BRANCH, "or disintegration", "(O)", "main"],
            ["(O)", "main", "task formulation"],
        ])

    def test_chapter_without_induction_scopes_is_unchanged(self):
        chapter = [["(G)", "main", "implication", "(F)", BRANCH],
                   ["(F)", BRANCH, "task formulation"], []]
        self.assertIs(strip_induction_scopes(chapter), chapter)

    def test_identity_without_its_source_asserts(self):
        with self.assertRaises(AssertionError):
            strip_induction_scopes([["(G)", "main", "induction scope", "(G)", LEAF]])


# Int scopes of premise-bound integrations (D-578).
INT_P = "main_boundary_int_scope_(primeFactorSeq[1,3,4,5,2,6,7,9,10,11])_int_0_lev_3_0"
INT_Q = "main_boundary_int_scope_(primeFactorSeq[1,3,4,5,2,6,7,9,12,11])_int_0_lev_3_2"


class StripIntScopeSegments(unittest.TestCase):
    def test_int_scope_segments_are_deleted(self):
        self.assertEqual(strip_induction_scope_segments(INT_P), "main")
        self.assertEqual(strip_induction_scope_segments(INT_P + "_boundary_" + BRANCH[len("main_boundary_"):]),
                         BRANCH)
        self.assertEqual(strip_induction_scope_segments(LEAF + "_boundary_int_scope_(p[1,9])_x"), "main")

    def test_witness_names_and_other_payloads_stay(self):
        trial = "main_boundary_integration_int_0_lev_6_119"
        self.assertEqual(strip_induction_scope_segments(trial), trial)
        subproof = "main_boundary_(implication3[int_0_lev_3_0,9])"
        self.assertEqual(strip_induction_scope_segments(subproof), subproof)


class StripIntScopes(unittest.TestCase):
    def test_promoted_rows_take_the_leaf_derivations(self):
        chapter = [
            ["(G)", "main", "implication", "(A)", "main", "(B)", "main"],
            ["(A)", "main", "int scope", "(A)", INT_P],
            ["(A)", INT_P, "implication", "(R)", INT_P],
            ["(R)", INT_P, "disintegration", "(S)", INT_P],
            ["(S)", INT_P, "zoom in", "(S)", "main"],
            ["(S)", "main", "task formulation"],
            ["(B)", "main", "int scope", "(B)", INT_Q],
            ["(B)", INT_Q, "implication", "(T)", "main"],
            ["(T)", "main", "task formulation"],
        ]
        self.assertEqual(strip_induction_scopes(chapter), [
            ["(G)", "main", "implication", "(A)", "main", "(B)", "main"],
            ["(A)", "main", "implication", "(R)", "main"],
            ["(R)", "main", "disintegration", "(S)", "main"],
            ["(S)", "main", "task formulation"],
            ["(B)", "main", "implication", "(T)", "main"],
            ["(T)", "main", "task formulation"],
        ])

    def test_sibling_int_scopes_merge_their_duplicates(self):
        chapter = [
            ["(G)", "main", "implication", "(F)", INT_P, "(F)", INT_Q],
            ["(F)", INT_P, "implication", "(K)", INT_P],
            ["(K)", INT_P, "task formulation"],
            ["(F)", INT_Q, "implication", "(L)", INT_Q],
            ["(L)", INT_Q, "task formulation"],
        ]
        self.assertEqual(strip_induction_scopes(chapter), [
            ["(G)", "main", "implication", "(F)", "main", "(F)", "main"],
            ["(F)", "main", "implication", "(K)", "main"],
            ["(K)", "main", "task formulation"],
        ])

    def test_an_induction_scope_inside_an_int_scope_strips_with_it(self):
        nested = INT_P + "_boundary_is_(in3[2,3,marker,4])_it_0_lev_2_7"
        chapter = [
            ["(G)", "main", "int scope", "(G)", INT_P],
            ["(G)", INT_P, "induction scope", "(G)", nested],
            ["(G)", nested, "implication", "(F)", "main"],
            ["(F)", "main", "task formulation"],
        ]
        self.assertEqual(strip_induction_scopes(chapter), [
            ["(G)", "main", "implication", "(F)", "main"],
            ["(F)", "main", "task formulation"],
        ])

    def test_lone_int_scope_row_asserts(self):
        with self.assertRaises(AssertionError):
            strip_induction_scopes([["(G)", "main", "int scope", "(G)", INT_P]])


if __name__ == "__main__":
    unittest.main()
