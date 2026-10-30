# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Frontier scheduler legality, exact objective gates and scope preservation."""
import itertools,unittest
from functools import cache
from proof_layout import frontier_order,improve_frontier
from proof_processing import order_scope_rows,order_main_scope_rows,citation_key
from tests.test_proof_layout import metric

class FrontierTests(unittest.TestCase):
    def test_closing_a_chain_reduces_the_frontier(self):
        order=[0,1,2,3];edges={(0,3),(1,2)}
        self.assertEqual(frontier_order(order,edges),[0,3,1,2])
        self.assertEqual(improve_frontier(order,edges),[0,3,1,2])
        self.assertEqual(improve_frontier(order,edges,{(0,1)},3),order)
        self.assertEqual(improve_frontier(order,set()),order)

    def test_all_four_vertex_graphs_have_no_objective_regression(self):
        pairs=list(itertools.combinations(range(4),2))
        for mask in range(1<<len(pairs)):
            edges={edge for bit,edge in enumerate(pairs) if mask&(1<<bit)}
            for order in itertools.permutations(range(4)):
                positions={vertex:i for i,vertex in enumerate(order)}
                if not all(positions[a]<positions[b] for a,b in edges):continue
                for terminal in (None,order[-1]):
                    candidate=improve_frontier(order,edges,terminal=terminal)
                    before=metric.arrangement(range(4),edges,order)
                    after=metric.arrangement(range(4),edges,candidate)
                    self.assertEqual(after['forward_references'],0)
                    self.assertLessEqual(after['cost'],before['cost']+1e-14)
                    if terminal is not None:self.assertEqual(candidate[-1],terminal)

    def test_bad_constraints_and_empty_input_assert(self):
        for edges in ({(2,0)},{(0,0)},{(0,3)}):
            with self.assertRaises(AssertionError):frontier_order([0,1,2],edges)
        with self.assertRaises(AssertionError):frontier_order([],set())
        with self.assertRaises(AssertionError):frontier_order([0,1],set(),terminal=0)
        self.assertEqual(frontier_order([None,1],set()),[None,1])

    def test_scope_wrappers_keep_goal_and_namespace_slots(self):
        rows=[["(D[x])","main","implication","(A[x])","main"],
              ["(C[x])","main","implication","(B[x])","main"],
              ["(B[x])","main","task formulation"],["(A[x])","main","task formulation"]]
        result=order_scope_rows(rows,set(),cache(citation_key),group_frontier=True)
        self.assertEqual(result[0],rows[0]);self.assertCountEqual(result,rows)
        theorem="(>[N](AnchorPeano[N])(in[x,N]))"
        self.assertEqual(order_main_scope_rows([[theorem]],{'a':rows},[['a']],set(),group_frontier=True)['a'],result)
        other=theorem.replace('AnchorPeano','AnchorIncubator')
        self.assertEqual(order_main_scope_rows([[other]],{'a':rows},[['a']],set(),group_frontier=True)['a'],rows)

class IncrementalFrontierOracleTests(unittest.TestCase):
    def test_same_choices_as_full_remaining_graph_rescan(self):
        import random
        rng = random.Random(291078)
        for size in (1, 5, 12, 30):
            for _ in range(20):
                order = list(range(size))
                rng.shuffle(order)
                rank = {v: i for i, v in enumerate(order)}
                possible = [(a, b) for i, a in enumerate(order) for b in order[i + 1:]]
                edges = {e for e in possible if rng.random() < .2}
                precedence = {e for e in possible if rng.random() < .1}
                terminal = order[-1]
                outgoing = {v: {b for a, b in edges if a == v} for v in order}
                required = {v: {a for a, b in edges | precedence if b == v} for v in order}
                remaining, live, expected = set(order), set(), []
                while remaining:
                    ready = [v for v in remaining if not required[v] & remaining
                             and (v != terminal or len(remaining) == 1)]
                    options = []
                    for v in ready:
                        rest = remaining - {v}
                        closed = {a for a in live if not outgoing[a] & rest}
                        cost = len(live) - len(closed) + bool(outgoing[v] & rest)
                        options.append(((cost, -len(closed), rank[v]), v, closed))
                    _, chosen, closed = min(options)
                    remaining.remove(chosen)
                    live -= closed
                    if outgoing[chosen] & remaining:
                        live.add(chosen)
                    expected.append(chosen)
                self.assertEqual(frontier_order(order, edges, precedence, terminal), expected)


if __name__=='__main__':unittest.main()
