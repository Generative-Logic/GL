"""Proof-row ordering tests, including rejected unsupported dependencies."""
from collections import Counter
from functools import cache
import unittest

from proof_processing import citation_key, ordered_scope_indices, order_scope_rows, order_main_scope_rows


class ScopeOrderTests(unittest.TestCase):
    def order(self, rows, theorems=()):
        key = cache(citation_key)
        return order_scope_rows(rows, {key(t) for t in theorems}, key)

    def test_later_stored_fact_must_appear_first_for_reader(self):
        rows = [["(Q[x])","main","implication","(P[x])","main"],
                ["(P[x])","main","implication","(R[x])","main"],
                ["(A[x])","main","task formulation"],
                ["(R[x])","main","task formulation"]]
        result = self.order(rows)
        self.assertEqual(result[0], rows[0])
        reader = [row[0] for row in reversed(result)]
        self.assertLess(reader.index("(R[x])"), reader.index("(P[x])"))
        self.assertLess(reader.index("(P[x])"), reader.index("(Q[x])"))
        self.assertEqual(Counter(map(tuple, result)), Counter(map(tuple, rows)))

    def test_rows_move_only_inside_exact_scope_positions(self):
        child = "main_boundary_subproof_(P[x])"
        rows = [["(Q[x])","main","implication","(R[x])","main"],
                ["(B[x])",child,"implication","(A[x])",child],
                ["(R[x])","main","task formulation"],
                ["(A[x])",child,"task formulation"]]
        result = self.order(rows)
        self.assertEqual([r[1] for r in rows], [r[1] for r in result])
        self.assertEqual(Counter(map(tuple, rows)), Counter(map(tuple, result)))

    def test_cross_scope_interface_is_not_a_local_forward_reference(self):
        rows = [["(Q[x])","main","validity name","(P[x])","child"],
                ["(P[x])","child","task formulation"]]
        self.assertEqual(self.order(rows), rows)

    def test_theorem_alias_keeps_free_names_and_template_kind(self):
        a = "(>[v1](P[v1,k])(Q[v1,k]))"
        b = "(>[w1](P[w1,k])(Q[w1,k]))"
        self.assertEqual(citation_key(a), citation_key(b))
        self.assertNotEqual(citation_key(a), citation_key(a.replace(",k",",j")))
        self.assertNotEqual(citation_key("(P[x])"), citation_key("(P[x])_integration_goal"))
        rows = [["(Q[x,k])","main","implication",b,"main"],
                [a,"main","theorem"]]
        self.assertEqual(self.order(rows, [a]), rows)

    def test_integration_template_is_not_required_as_a_fact(self):
        rows = [["(Q[x])","main","premise element","(P[x])_integration_goal","main"]]
        self.assertEqual(self.order(rows), rows)

    def test_compilation_can_reference_registered_theorem(self):
        theorem = "(>[x](P[x])(Q[x]))"
        rows = [["(implication0[x])","main","compilation",theorem,"main"]]
        self.assertEqual(self.order(rows,[theorem]),rows)

    def test_optional_second_or_license_is_preserved(self):
        a,b = "(>[x](P[x])(Q[x]))","(>[x](R[x])(S[x]))"
        rows = [["(or0[x])","main","or theorem",a,"main",b,"main"]]
        self.assertEqual(self.order(rows,[a]),rows)

    def test_missing_ordinary_source_asserts(self):
        with self.assertRaisesRegex(AssertionError,"unresolved citation"):
            self.order([["(Q[x])","main","implication","(P[x])","main"]])

    def test_cycle_and_goal_used_before_final_row_assert(self):
        rows = [["(Q[x])","main","implication","(P[x])","main"],
                ["(P[x])","main","implication","(Q[x])","main"]]
        with self.assertRaisesRegex(AssertionError,"cannot be ordered"):
            self.order(rows)

    def test_direct_scope_indices_and_empty_rows(self):
        rows = [["(Q[x])","main","implication","(P[x])","main"],[],
                ["(P[x])","main","task formulation"]]
        key = cache(citation_key)
        self.assertEqual(ordered_scope_indices(rows,[0,2],set(),key),[2,0])
        self.assertEqual(self.order(rows),rows)

    def test_selected_scopes_preserve_rows_and_registry(self):
        main = "(>[N,x](AnchorPeano[N])(P[x]))"
        incubator = "(>[N,x](AnchorIncubator3[N])(P[x]))"
        rows = [["(P[x])","main","task formulation"]]
        theorems = [[main,"single chapter proof","-1"],[incubator,"single chapter proof","-1"]]
        stacks = {"a.txt":rows,"b.txt":rows}
        result = order_main_scope_rows(theorems,stacks,[["a.txt"],["b.txt"]],set())
        self.assertEqual(result,stacks)
        self.assertEqual(result["b.txt"],rows)
        with self.assertRaisesRegex(AssertionError,"missing processed"):
            order_main_scope_rows(theorems,{},[["a.txt"],["b.txt"]],set())

    def test_empty_or_malformed_chapter_asserts(self):
        with self.assertRaises(AssertionError):
            self.order([])
        with self.assertRaises(AssertionError):
            self.order([["(P[x])","main","implication","(Q[x])"]])


if __name__ == "__main__":
    unittest.main()

