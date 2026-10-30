# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.


"""Independent scheduling and full-state oracles for exact layout acceleration."""
import itertools
import random
import unittest
from proof_layout import private_helper_order, frontier_order
from proof_layout_search import LayoutState


class ExactLayoutRuntimeTests(unittest.TestCase):
    def test_schedulers_match_full_scan_with_generic_vertices_and_terminals(self):
        randomizer = random.Random(83951)
        for size in (1, 4, 9, 31):
            for trial in range(16):
                order = [None] + [("vertex", index) for index in range(1, size)]
                randomizer.shuffle(order)
                rank = {vertex: index for index, vertex in enumerate(order)}
                possible = list(itertools.combinations(order, 2))
                edges = {edge for edge in possible if randomizer.random() < .2}
                precedence = {edge for edge in possible if randomizer.random() < .08}
                terminal = order[-1] if trial % 2 else None
                outgoing = {vertex: {b for a, b in edges if a == vertex} for vertex in order}
                remaining, reverse = set(order), []
                while remaining:
                    ready = [vertex for vertex in remaining if not any(
                        a == vertex and b in remaining for a, b in edges | precedence)]
                    if not reverse and terminal is not None:
                        chosen = terminal
                    else:
                        helpers = [vertex for vertex in ready
                                   if reverse and outgoing[vertex] == {reverse[-1]}]
                        chosen = max(helpers or ready, key=rank.__getitem__)
                    reverse.append(chosen)
                    remaining.remove(chosen)
                self.assertEqual(private_helper_order(order, edges, precedence, terminal),
                                 list(reversed(reverse)))
                remaining, live, expected = set(order), set(), []
                while remaining:
                    ready = [vertex for vertex in remaining if not any(
                        b == vertex and a in remaining for a, b in edges | precedence)
                        and (terminal is None or vertex != terminal or len(remaining) == 1)]
                    candidates = []
                    for vertex in ready:
                        rest = remaining - {vertex}
                        closed = {parent for parent in live if not outgoing[parent] & rest}
                        cost = int(bool(outgoing[vertex] & rest)) - len(closed)
                        candidates.append(((cost, -len(closed), rank[vertex]), vertex, closed))
                    _, chosen, closed = min(candidates, key=lambda item: item[0])
                    remaining.remove(chosen)
                    live -= closed
                    if outgoing[chosen] & remaining:
                        live.add(chosen)
                    expected.append(chosen)
                self.assertEqual(frontier_order(order, edges, precedence, terminal), expected)

    def test_insertion_sequences_retain_the_complete_exact_state(self):
        randomizer = random.Random(40383)
        for size in (9, 23, 53):
            for trial in range(5):
                order = list(range(size))
                randomizer.shuffle(order)
                edges = {edge for edge in itertools.combinations(order, 2)
                         if randomizer.random() < .15}
                precedence = {(order[0], order[-1])}
                state = LayoutState(order, edges, precedence, order[-1])
                for step in range(45):
                    # Leave a cached neighboring proposal before the insertion.
                    state.adjacent(randomizer.randrange(size - 1))
                    state.improve_insertion(randomizer.choice(order),
                                            randomizer.choice((-1, 1)),
                                            randomizer.choice((1, 6, size)))
                    fresh = LayoutState(state.order, edges, precedence, order[-1])
                    for attribute in ("positions", "score", "peak", "frontier", "frontier_counts",
                                      "incoming_bits", "outgoing_bits"):
                        self.assertEqual(getattr(state, attribute), getattr(fresh, attribute),
                                         (size, trial, step, attribute))


if __name__ == "__main__":
    unittest.main()
