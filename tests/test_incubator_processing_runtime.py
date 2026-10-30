"""Bounded Incubator processing retains proof content and strict ordering."""
from collections import Counter
from pathlib import Path
import random
import unittest
from unittest.mock import patch

from process_proof_graphs import _build_fname_list
from proof_convergence import process_incubator_book, process_to_fixed_point
from proof_hierarchy import (hierarchical_scope_items, order_main_hierarchies,
                             resolved_citations, resolved_dependencies, theorem_alpha_keys)
from proof_export.graph import parse_scoped_expression
from proof_export.mpl import alpha_key
from proof_reader import row_node, schedule_nodes
from tests.test_proof_hierarchy import fixture, THEOREM


class IncubatorProcessingTests(unittest.TestCase):
    def test_bounded_path_preserves_nested_proof_and_rejects_missing_bundles(self):
        theorem = THEOREM.replace('AnchorPeano', 'AnchorIncubator3')
        registry = [[theorem, 'single chapter proof', '-1']]
        groups = _build_fname_list(registry)
        name = groups[0][0]
        rows = fixture()
        originals = {name: rows}
        saved_rows = tuple(map(tuple, rows))
        result_registry, result = process_incubator_book(
            registry, originals, _build_fname_list, set())
        result = order_main_hierarchies(result_registry, result, groups, set())
        self.assertEqual(result_registry, registry)
        self.assertEqual(Counter(map(tuple, result[name])), Counter(map(tuple, rows)))
        self.assertEqual(result[name][0], rows[0])
        hierarchical_scope_items(result[name])
        self.assertTrue(all(a > b for a, b in resolved_dependencies(result[name], set())))
        self.assertEqual(tuple(map(tuple, originals[name])), saved_rows)
        with self.assertRaisesRegex(AssertionError, 'bundle coverage'):
            process_incubator_book(registry, {}, _build_fname_list, set())

    def test_dispatch_never_calls_main_shortening_or_repeated_scope_search(self):
        theorem = THEOREM.replace('AnchorPeano', 'AnchorIncubator8')
        registry = [[theorem, 'single chapter proof', '-1']]
        name = _build_fname_list(registry)[0][0]
        from proof_convergence import order_main_scope_rows
        with patch('proof_convergence.compact_main_equalities', side_effect=AssertionError('main-only')):
            with patch('proof_convergence.order_main_scope_rows', wraps=order_main_scope_rows) as scopes:
                result = process_to_fixed_point(registry, {name: fixture()}, _build_fname_list,
                                                set(), Path('no-incubator-binary-needed'))
        self.assertEqual(scopes.call_count, 1)
        self.assertEqual(result[0], registry)
        with self.assertRaisesRegex(AssertionError, 'cannot be mixed'):
            process_to_fixed_point(registry + [[THEOREM, 'single chapter proof', '-1']], {},
                                   _build_fname_list, set(), Path('missing'))

    def test_chapter_citation_precedes_consumer_and_keeps_exact_bundles(self):
        producer = '(>[N,x](AnchorIncubator3[N])(R[x]))'
        consumer = '(>[N,x](AnchorIncubator3[N])(Q[x]))'
        registry = [[consumer, 'single chapter proof', '-1'],
                    [producer, 'single chapter proof', '-1']]
        groups = _build_fname_list(registry)
        use = [['(Q[x])', 'main', 'implication', producer, 'main', '(R[x])', 'main'],
               [producer, 'main', 'theorem'], ['(R[x])', 'main', 'task formulation']]
        source = [['(R[x])', 'main', 'task formulation']]
        result_registry, result = process_incubator_book(
            registry, {groups[0][0]: use, groups[1][0]: source}, _build_fname_list, set())
        self.assertEqual(result_registry, list(reversed(registry)))
        self.assertEqual(result[groups[0][0]], source)
        self.assertEqual(Counter(map(tuple, result[groups[1][0]])), Counter(map(tuple, use)))

    def test_theorem_alpha_keys_identify_renamed_theorems_only(self):
        source = '(>[x](P[x])(Q[x]))'
        keys = theorem_alpha_keys([source, '(>[y](P[y])(Q[y]))'])
        self.assertIsInstance(keys, frozenset)
        self.assertEqual(keys, frozenset({alpha_key(parse_scoped_expression(source).expression)}))
        self.assertEqual(len(theorem_alpha_keys([source, '(>[x](P[x])(R[x]))'])), 2)
        self.assertEqual(theorem_alpha_keys([]), frozenset())
        with self.assertRaises(AssertionError):
            theorem_alpha_keys(['(P[x])trailing'])

    def test_shared_registry_keys_preserve_aliases_and_missing_source_failure(self):
        source = '(>[x](P[x])(Q[x]))'
        alias = '(>[y](P[y])(Q[y]))'
        known = [source]
        keys = theorem_alpha_keys(known)
        self.assertEqual(keys, frozenset(alpha_key(parse_scoped_expression(t).expression) for t in known))
        rows = [['(Q[z])', 'main', 'implication', alias, 'main'],
                [source, 'main', 'theorem']]
        self.assertEqual(resolved_citations(rows, known), resolved_citations(rows, known, keys))
        self.assertEqual(resolved_dependencies(rows, known), resolved_dependencies(rows, known, keys))
        missing = [['(Q[z])', 'main', 'implication', '(Missing[z])', 'main']]
        for shared in (None, keys):
            with self.assertRaisesRegex(AssertionError, 'unresolved citation'):
                resolved_citations(missing, known, shared)

    def test_priority_scheduler_matches_independent_full_ready_scan(self):
        rng = random.Random(1010)
        for size in (1, 3, 12, 35):
            for trial in range(20):
                order = list(range(size))
                rng.shuffle(order)
                edges = {(a, b) for i, a in enumerate(order) for b in order[i + 1:]
                         if rng.random() < 0.15}
                terminal = order[-1]
                pending = set(order)
                expected = []
                while pending:
                    ready = [v for v in pending if not any(a in pending and b == v for a, b in edges)
                             and (v != terminal or len(pending) == 1)]
                    chosen = max(ready)
                    expected.append(chosen)
                    pending.remove(chosen)
                actual = schedule_nodes([row_node(v) for v in order], edges, terminal, False)
                self.assertEqual([node['index'] for node in actual], expected)
        with self.assertRaisesRegex(AssertionError, 'cyclic reader block contraction'):
            schedule_nodes([row_node(0), row_node(1)], {(0, 1), (1, 0)}, group_or=False)


if __name__ == '__main__':
    unittest.main()
