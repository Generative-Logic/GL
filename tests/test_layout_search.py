"""Independent full-score oracle for incremental graph arrangement search."""
import itertools,random,unittest
from functools import cache
from proof_processing import order_scope_rows,order_main_scope_rows,citation_key
from proof_layout import layout_score
from proof_layout_search import LayoutState,search_layout

class LayoutSearchTests(unittest.TestCase):
    def test_every_adjacent_delta_on_every_four_vertex_dag(self):
        vertices=list(range(4));possible=list(itertools.combinations(vertices,2))
        for mask in range(1<<len(possible)):
            edges={edge for i,edge in enumerate(possible) if mask&(1<<i)}
            for order in itertools.permutations(vertices):
                positions={v:i for i,v in enumerate(order)}
                if not all(positions[a]<positions[b] for a,b in edges):continue
                state=LayoutState(order,edges)
                for i in range(3):
                    delta=state.adjacent(i)
                    if (order[i],order[i+1]) in edges:self.assertIsNone(delta);continue
                    candidate=list(order);candidate[i:i+2]=candidate[i:i+2][::-1]
                    expected=layout_score(candidate,edges)-layout_score(order,edges)
                    self.assertEqual(delta,expected,(order,edges,i))
                    state.adjacent(i,True);self.assertEqual(state.score,layout_score(candidate,edges))
                    state.adjacent(i,True);self.assertEqual(state.order,list(order));self.assertEqual(state.score,layout_score(order,edges))

    def test_scope_order_cache_does_not_alias_rows_or_ignore_options(self):
        from unittest.mock import patch
        import proof_processing as module
        theorem = '(>[N](AnchorPeano[N])(in[x,N]))'
        rows = [['(B[x])', 'main', 'implication', '(A[x])', 'main'],
                ['(A[x])', 'main', 'task formulation']]
        with patch.object(module, 'order_scope_rows', wraps=module.order_scope_rows) as actual:
            with module.main_scope_order_cache():
                first = module.order_main_scope_rows([[theorem]], {'a': rows}, [['a']], ())
                first['a'][0][0] = '(changed[x])'
                second = module.order_main_scope_rows([[theorem]], {'b': rows}, [['b']], ())
                self.assertEqual(second['b'], rows)
                self.assertEqual(actual.call_count, 1)
                module.order_main_scope_rows([[theorem]], {'a': rows}, [['a']], (), group_private=True)
                module.order_main_scope_rows([[theorem]], {'a': rows}, [['a']], ('(known[x])',))
                self.assertEqual(actual.call_count, 3)
                invalid = [rows[0]]
                with self.assertRaises(AssertionError):
                    module.order_main_scope_rows([[theorem]], {'a': invalid}, [['a']], ())
            with module.main_scope_order_cache():
                module.order_main_scope_rows([[theorem]], {'a': rows}, [['a']], ())
                self.assertEqual(actual.call_count, 5)

    def test_insertion_scores_equal_independent_candidate_enumeration(self):
        rng = random.Random(39027)
        for size in (5, 9, 16):
            for _ in range(10):
                order = list(range(size))
                rng.shuffle(order)
                edges = {(a, b) for i, a in enumerate(order) for b in order[i + 1:]
                         if rng.random() < 0.22}
                precedence = {(order[0], order[-1])}
                for vertex in order:
                    for direction in (-1, 1):
                        window = rng.choice((1, 4, size))
                        state = LayoutState(order, edges, precedence, order[-1])
                        best, score = order.copy(), layout_score(order, edges)
                        start = order.index(vertex)
                        for offset in range(1, window + 1):
                            target = start + direction * offset
                            if not 0 <= target < size:
                                break
                            candidate = order.copy()
                            candidate.remove(vertex)
                            candidate.insert(target, vertex)
                            positions = {v: i for i, v in enumerate(candidate)}
                            if candidate[-1] != order[-1] or not all(
                                    positions[a] < positions[b] for a, b in edges | precedence):
                                break
                            value = layout_score(candidate, edges)
                            if value < score:
                                best, score = candidate, value
                        state.improve_insertion(vertex, direction, window)
                        self.assertEqual(state.order, best, (order, edges, vertex, direction, window))
                        self.assertEqual(state.score, score)
                        self.assertEqual(state.peak, max(state.frontier))
        state = LayoutState([0, 1], set())
        for vertex, direction, window in ((2, 1, 1), (0, 0, 1), (0, 1, 0)):
            with self.assertRaises(AssertionError):
                state.improve_insertion(vertex, direction, window)

    def test_crossing_balance_matches_pairwise_crossings(self):
        edges = {(0, 4), (1, 5), (2, 4), (3, 5)}
        state = LayoutState(list(range(7)), edges)
        def crossings(order):
            positions = {v: i for i, v in enumerate(order)}
            intervals = [(positions[a], positions[b]) for a, b in edges]
            return sum(a < c < b < d or c < a < d < b
                       for (a, b), (c, d) in itertools.combinations(intervals, 2))
        for i in range(6):
            x, y = state.order[i:i + 2]
            if (x, y) in edges:
                continue
            swapped = state.order.copy()
            swapped[i:i + 2] = [y, x]
            self.assertEqual(state.crossing_balance(x, y), crossings(swapped) - crossings(state.order))
        with self.assertRaises(AssertionError):
            state.crossing_balance(0, 4)
        with self.assertRaises(AssertionError):
            state.crossing_balance(0, 0)

    def test_full_score_matches_pairwise_reference(self):
        import math
        rng = random.Random(4850)
        for size in range(1, 19):
            for _ in range(15):
                edges = {(a, b) for a, b in itertools.combinations(range(size), 2)
                         if rng.random() < 0.4}
                outgoing = {a: [b for x, b in edges if x == a] for a in range(size)}
                private = [(a, b) for a, b in edges if len(outgoing[a]) == 1]
                distance = sum(b - a for a, b in edges)
                gaps = sum(b - a - 1 for a, b in private)
                frontier = max(sum(a <= cut < max(children) for a, children in outgoing.items()
                                   if children) for cut in range(size))
                crossings = sum(a < c < b < d or c < a < d < b
                                for (a, b), (c, d) in itertools.combinations(edges, 2))
                count = len(edges)
                denominators = (max(1, count * (size - 1)),
                                max(1, len(private) * max(0, size - 2)),
                                max(1, size - 1), max(1, count * (count - 1) // 2))
                common = math.lcm(*denominators)
                expected = sum(value * (common // denominator)
                               for value, denominator in zip((distance, gaps, frontier, crossings), denominators))
                self.assertEqual(layout_score(list(range(size)), edges), expected)

    def test_cache_complete_key_and_immutable_results(self):
        from unittest.mock import patch
        import proof_layout_search as module
        order = list(range(10))
        edges = {(0, 4), (1, 4), (4, 9)}
        expected = search_layout(order, edges)
        with patch.object(module, '_search_layout', wraps=module._search_layout) as actual:
            with module.layout_search_cache():
                result = search_layout(order, edges)
                result.clear()
                self.assertEqual(search_layout(order, edges), expected)
                renamed = [str(v) for v in order]
                self.assertEqual(search_layout(renamed, {(str(a), str(b)) for a, b in edges}),
                                 [str(v) for v in expected])
                self.assertEqual(actual.call_count, 1)
                search_layout(order, edges, window=7)
                search_layout(order, edges, terminal=9)
                search_layout(order, edges, precedence={(2, 3)})
                search_layout(order, edges | {(3, 8)})
                self.assertEqual(actual.call_count, 5)
                with self.assertRaises(AssertionError):
                    search_layout(order, edges, window=0)
                with self.assertRaises(AssertionError):
                    search_layout([0, 0], set())
                with self.assertRaises(AssertionError):
                    search_layout(order, edges | {(9, 0)})
            with module.layout_search_cache():
                search_layout(order, edges)
                self.assertEqual(actual.call_count, 6)
                with self.assertRaisesRegex(RuntimeError, 'deliberate'):
                    with module.layout_search_cache():
                        search_layout(order, edges)
                        raise RuntimeError('deliberate')
                search_layout(order, edges)
                self.assertEqual(actual.call_count, 7)
            search_layout(order, edges)
            self.assertEqual(actual.call_count, 8)

    def test_random_walk_keeps_incremental_state_exact(self):
        rng = random.Random(29107)
        for size in (9, 20, 47):
            edges = {(a, b) for a, b in itertools.combinations(range(size), 2)
                     if rng.random() < 0.13}
            state = LayoutState(list(range(size)), edges)
            for _ in range(600):
                index = rng.randrange(size - 1)
                before = state.score
                delta = state.adjacent(index)
                if delta is None:
                    continue
                # An intervening evaluation must not reuse a different candidate.
                state.adjacent((index + 1) % (size - 1))
                self.assertEqual(state.adjacent(index, True), delta)
                self.assertEqual(state.score, before + delta)
                self.assertEqual(state.score, layout_score(state.order, edges))
                self.assertEqual(state.peak, max(state.frontier))
                self.assertEqual(state.frontier_counts,
                                 [state.frontier.count(n) for n in range(size + 1)])
                for vertex in state.order:
                    self.assertEqual(state.incoming_bits[vertex],
                                     sum(1 << state.positions[v] for v in state.incoming[vertex]))
                    self.assertEqual(state.outgoing_bits[vertex],
                                     sum(1 << state.positions[v] for v in state.outgoing[vertex]))

    def test_exact_search_equals_independent_permutation_minimum(self):
        edges={(0,3),(1,3),(2,4),(3,5),(4,5)};precedence={(0,2)};order=list(range(6))
        candidates=[]
        for prefix in itertools.permutations(order[:-1]):
            candidate=list(prefix)+[5];p={v:i for i,v in enumerate(candidate)}
            if all(p[a]<p[b] for a,b in edges|precedence):candidates.append(candidate)
        result=search_layout(order,edges,precedence,5)
        self.assertEqual(layout_score(result,edges),min(layout_score(c,edges) for c in candidates))
        self.assertEqual(result[-1],5)

    def test_large_search_is_monotone_legal_and_adjacent_fixed(self):
        rng=random.Random(719)
        for _ in range(12):
            order=list(range(13));edges={(a,b) for a,b in itertools.combinations(order,2) if rng.random()<.2}
            result=search_layout(order,edges,terminal=12,window=4)
            self.assertLessEqual(layout_score(result,edges),layout_score(order,edges))
            state=LayoutState(result,edges,terminal=12)
            self.assertEqual(result[-1],12)
            for i in range(12):
                delta=state.adjacent(i);self.assertTrue(delta is None or delta>=0)

    def test_hard_constraints_ties_and_generic_none_vertex(self):
        state=LayoutState([0,1,2],set(),{(0,1)},2)
        self.assertIsNone(state.adjacent(0));self.assertIsNone(state.adjacent(1))
        self.assertEqual(search_layout([0,None,2],set()),[0,None,2])
        state=LayoutState([0,None,2],set());self.assertEqual(state.adjacent(0),0)
        with self.assertRaises(AssertionError):LayoutState([0,0],set())
        with self.assertRaises(AssertionError):LayoutState([0,1],{(1,0)})
        with self.assertRaises(AssertionError):LayoutState([0,1],set(),terminal=0)
        with self.assertRaises(AssertionError):state.adjacent(2)
        with self.assertRaises(AssertionError):search_layout([0,1],set(),window=0)

    def test_scope_wrappers_keep_goal_and_namespace_slots(self):
        rows=[["(D[x])","main","implication","(A[x])","main"],
              ["(C[x])","main","implication","(B[x])","main"],
              ["(B[x])","main","task formulation"],["(A[x])","main","task formulation"]]
        result=order_scope_rows(rows,set(),cache(citation_key),group_search=True)
        self.assertEqual(result[0],rows[0]);self.assertCountEqual(result,rows)
        theorem="(>[N](AnchorPeano[N])(in[x,N]))"
        self.assertEqual(order_main_scope_rows([[theorem]],{'a':rows},[['a']],set(),group_search=True)['a'],result)
        other=theorem.replace('AnchorPeano','AnchorIncubator')
        self.assertEqual(order_main_scope_rows([[other]],{'a':rows},[['a']],set(),group_search=True)['a'],rows)

    def test_full_range_polish_preserves_the_short_search_gain(self):
        order=list(range(12));edges={(0,11),(1,6),(2,6),(3,7),(4,7),(5,8),(6,9),(7,10),(8,11)}
        short=search_layout(order,edges,terminal=11)
        polished=search_layout(short,edges,terminal=11,window=11)
        self.assertLessEqual(layout_score(polished,edges),layout_score(short,edges))
        state=LayoutState(polished,edges,terminal=11)
        for i in range(11):
            delta=state.adjacent(i);self.assertTrue(delta is None or delta>=0)

if __name__=='__main__':unittest.main()
