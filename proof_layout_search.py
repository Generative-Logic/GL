# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Deterministic local search under the frozen proof arrangement objective."""
import math
from bisect import bisect_right
from contextlib import contextmanager
from contextvars import ContextVar
from itertools import permutations
from proof_layout import layout_score


_SEARCH_CACHE = ContextVar("proof_layout_search_cache", default=None)


@contextmanager
def layout_search_cache():
    """
    @brief Bound pure layout-result reuse to one complete processing call.

    @details Every invocation owns a fresh dictionary. Nested calls restore
    the outer dictionary on exit, including exceptions; no proof or registry
    survives the processing call through this cache.

    @return Context manager providing an isolated immutable-result cache.
    @invariant A cache hit requires the complete graph and all search options.
    """
    token = _SEARCH_CACHE.set({})
    try:
        yield
    finally:
        _SEARCH_CACHE.reset(token)


class LayoutState:
    """Exact adjacent-move state; all vertices remain in the same proof graph."""

    def __init__(self, order, edges, precedence=(), terminal=None):
        """
        @brief Construct the measured state of one legal graph layout.

        @details Store exact integer weights, per-cut producer counts and
        neighbor-position bitsets. The bitsets represent the current order;
        they do not change which proof edges or precedences are present.

        @param order Unique vertices in dependency order.
        @param edges Actual proof edges.
        @param precedence Additional unscored ordering constraints.
        @param terminal Optional final goal vertex.
        @return Initialized exact search state.
        """
        self.order = list(order)
        self.edges = set(edges)
        self.required = self.edges | set(precedence)
        self.positions = {v: i for i, v in enumerate(order)}
        assert order and len(self.positions) == len(order)
        assert all(
            a in self.positions
            and b in self.positions
            and self.positions[a] < self.positions[b]
            for a, b in self.required
        )
        assert terminal is None or order[-1] == terminal
        self.terminal = terminal
        self.outgoing = {v: set() for v in order}
        self.incoming = {v: set() for v in order}
        for a, b in self.edges:
            self.outgoing[a].add(b)
            self.incoming[b].add(a)
        self.private = {(a, b) for a, b in self.edges if len(self.outgoing[a]) == 1}
        n, m = len(order), len(self.edges)
        denominators = (
            max(1, m * (n - 1)),
            max(1, len(self.private) * max(0, n - 2)),
            max(1, n - 1),
            max(1, m * (m - 1) // 2),
        )
        common = math.lcm(*denominators)
        self.weights = tuple(common // d for d in denominators)
        self.incoming_bits = {
            v: sum(1 << self.positions[parent] for parent in self.incoming[v])
            for v in order
        }
        self.outgoing_bits = {
            v: sum(1 << self.positions[child] for child in self.outgoing[v])
            for v in order
        }
        self.linear_balance = {
            v: (
                (len(self.incoming[v]) - len(self.outgoing[v])) * self.weights[0]
                + (
                    sum(len(self.outgoing[parent]) == 1 for parent in self.incoming[v])
                    - (len(self.outgoing[v]) == 1)
                )
                * self.weights[1]
            )
            for v in order
        }
        self.frontier = [0] * n
        for v, consumers in self.outgoing.items():
            if consumers:
                for i in range(
                    self.positions[v], self.outgoing_bits[v].bit_length() - 1
                ):
                    self.frontier[i] += 1
        self.frontier_counts = [0] * (n + 1)
        for count in self.frontier:
            self.frontier_counts[count] += 1
        self.peak = max(self.frontier)
        self.score = layout_score(order, self.edges)
        self.candidate = None

    def crossing_balance(self, x, y):
        """
        @brief Count the crossing change when independent x passes y to the right.

        @details Only the relative orders of their neighbors matter. The same
        balance applies throughout an insertion trajectory because all vertices
        other than the moving one keep their relative order, and no crossed
        vertex is its neighbor. Shared endpoints contribute zero.

        @param x Left vertex of the exchange, independent of y.
        @param y Right vertex of the exchange, independent of x.
        @return Exact change in crossing count for that adjacent exchange.
        """
        assert x != y and y not in self.outgoing[x] and x not in self.outgoing[y]
        crossing = len(self.incoming[x]) * len(self.outgoing[y]) - len(
            self.outgoing[x]
        ) * len(self.incoming[y])
        for neighbors, masks in (
            (self.incoming, self.incoming_bits),
            (self.outgoing, self.outgoing_bits),
        ):
            left, right = neighbors[x], neighbors[y]
            mask, sign = masks[y], 1
            if len(left) > len(right):
                left, right = right, left
                mask, sign = masks[x], -1
            for vertex in left:
                bit = 1 << self.positions[vertex]
                crossing += sign * (
                    2 * (mask & (bit - 1)).bit_count() + bool(mask & bit) - len(right)
                )
        return crossing

    def improve_insertion(self, vertex, direction, window):
        """
        @brief Score a complete legal insertion trajectory without trial moves.

        @details Moving right replaces a prefix by its original successor
        prefix minus the moving vertex. Moving left adds that vertex to each
        earlier prefix. These sets determine the exact replacement frontier
        cuts. Neighbor order gives each crossing change without changing any
        graph index. Only the first strictly best prefix is finally applied,
        preserving the candidate order and tie policy of adjacent trial moves.
        Sorted last-consumer positions give exact closed-producer counts.
        Each saved delta and frontier belongs to the preceding trajectory
        prefix, so applying that prefix reuses its score without recomputation.

        @param vertex Existing vertex to move.
        @param direction Negative one for left, positive one for right.
        @param window Maximum legal displacement to consider.
        @return None; applies only a strictly lower-cost insertion if present.
        @invariant Every candidate has the same exact score as full recomputation.
        """
        assert vertex in self.positions and direction in (-1, 1)
        assert isinstance(window, int) and window >= 1
        start = self.positions[vertex]
        counts = self.frontier_counts.copy()
        moves = []
        peak = self.peak
        score = best_score = self.score
        best_count = 0
        outgoing = bool(self.outgoing[vertex])
        other_last_consumers = sorted(
            (self.outgoing_bits[parent] & ~(1 << start)).bit_length()
            for parent in self.incoming[vertex]
        )
        for offset in range(window):
            index = start - offset - 1 if direction < 0 else start + offset
            if not 0 <= index < len(self.order) - 1:
                break
            if direction < 0:
                x, y = self.order[index], vertex
            else:
                x, y = vertex, self.order[index + 1]
            if (x, y) in self.required or (
                self.terminal is not None and y == self.terminal
            ):
                break
            if direction > 0:
                frontier = (
                    self.frontier[index + 1]
                    - outgoing
                    + bisect_right(other_last_consumers, index + 2)
                )
            else:
                frontier = (
                    (self.frontier[index - 1] if index else 0)
                    + outgoing
                    - bisect_right(other_last_consumers, index)
                )
            assert frontier >= 0
            counts[self.frontier[index]] -= 1
            counts[frontier] += 1
            new_peak = max(peak, frontier)
            while counts[new_peak] == 0:
                new_peak -= 1
            delta = (
                self.linear_balance[x]
                - self.linear_balance[y]
                + self.crossing_balance(x, y) * self.weights[3]
                + (new_peak - peak) * self.weights[2]
            )
            score += delta
            moves.append((index, delta, frontier, new_peak))
            peak = new_peak
            if score < best_score:
                best_score, best_count = score, offset + 1
        for move in moves[:best_count]:
            self.candidate = move
            delta = self.adjacent(move[0], True)
            assert delta is not None
        assert self.score == best_score, (
            vertex,
            direction,
            window,
            start,
            self.order,
            self.edges,
            self.score,
            best_score,
        )

    def adjacent(self, index, apply=False):
        """
        @brief Evaluate or apply one legal adjacent exchange exactly.

        @details An exchange changes degree-weighted spans, crossings between
        the two vertices' incident edges, and one frontier cut. Position
        bitsets count neighbor pairs without a quadratic pair walk. Only
        neighbors of the moved vertices need bit updates. A cached candidate
        belongs to this exact state and is invalidated by every applied move.
        None denotes a forbidden move, not a failed computation.

        @param index Position of the left vertex, before the last position.
        @param apply Whether to commit this legal move to this search state.
        @return Exact score difference, or None for an inadmissible move.
        @invariant Every applied move preserves all hard precedences.
        """
        assert 0 <= index < len(self.order) - 1
        x, y = self.order[index : index + 2]
        if (x, y) in self.required or (
            self.terminal is not None and y == self.terminal
        ):
            return None
        if self.candidate is not None and self.candidate[0] == index:
            _, delta, new_frontier, peak = self.candidate
        else:
            crossing = self.crossing_balance(x, y)
            closed = sum(
                x not in self.outgoing[v]
                and self.outgoing_bits[v].bit_length() == index + 2
                for v in self.incoming[y]
            )
            new_frontier = (
                (self.frontier[index - 1] if index else 0)
                - closed
                + bool(self.outgoing[y])
            )
            assert new_frontier >= 0
            peak = self.peak
            if self.frontier[index] == peak and self.frontier_counts[peak] == 1:
                peak -= 1
                while peak >= 0 and self.frontier_counts[peak] == 0:
                    peak -= 1
            peak = max(peak, new_frontier)
            delta = (
                self.linear_balance[x]
                - self.linear_balance[y]
                + (peak - self.peak) * self.weights[2]
                + crossing * self.weights[3]
            )
            self.candidate = (index, delta, new_frontier, peak)
        if apply:
            exchanged = (1 << index) | (1 << (index + 1))
            for vertex in self.outgoing[x] ^ self.outgoing[y]:
                self.incoming_bits[vertex] ^= exchanged
            for vertex in self.incoming[x] ^ self.incoming[y]:
                self.outgoing_bits[vertex] ^= exchanged
            self.order[index : index + 2] = [y, x]
            self.positions[x], self.positions[y] = index + 1, index
            self.frontier_counts[self.frontier[index]] -= 1
            self.frontier_counts[new_frontier] += 1
            self.frontier[index] = new_frontier
            self.peak = peak
            self.score += delta
            self.candidate = None
        return delta


def search_layout(order, edges, precedence=(), terminal=None, window=6):
    """
    @brief Reuse an exact search only for the same complete arrangement problem.

    @details Vertex labels are replaced by their input-order ranks. Search
    decisions depend on that order, edges, hard precedences, terminal and
    displacement range, all of which form the immutable cache key. Results
    are copied back to the caller's labels; caller mutation cannot enter the
    cache. Without an active processing context, run the search directly.

    @param order Existing legal layout with unique hashable vertices.
    @param edges Actual proof edges.
    @param precedence Required unscored format precedence.
    @param terminal Optional final goal vertex.
    @param window Maximum displacement considered for an insertion.
    @return Independent list containing the exact deterministic search result.
    @invariant Reuse never changes search depth, order, objective or convergence.
    """
    results = _SEARCH_CACHE.get()
    if results is None:
        return _search_layout(order, edges, precedence, terminal, window)
    order = tuple(order)
    edges, precedence = frozenset(edges), frozenset(precedence)
    assert isinstance(window, int) and window >= 1
    rank = {vertex: index for index, vertex in enumerate(order)}
    assert order and len(rank) == len(order)
    assert all(
        a in rank and b in rank and rank[a] < rank[b] for a, b in edges | precedence
    )
    assert terminal is None or order[-1] == terminal
    key = (
        len(order),
        frozenset((rank[a], rank[b]) for a, b in edges),
        frozenset((rank[a], rank[b]) for a, b in precedence),
        None if terminal is None else rank[terminal],
        window,
    )
    if key not in results:
        result = _search_layout(order, edges, precedence, terminal, window)
        results[key] = tuple(rank[vertex] for vertex in result)
    return [order[index] for index in results[key]]


def _search_layout(order, edges, precedence=(), terminal=None, window=6):
    """
    @brief Minimize the fixed layout cost with exact tiny and bounded large search.

    @details At most eight vertices are exhaustively searched. Larger graphs
    use adjacent descent to a fixed point and two deterministic insertion
    sweeps of up to window adjacent legal moves. Temporary losses are allowed
    only in a candidate trajectory; commit the best strict improvement.
    Final adjacent descent returns a local fixed point, not a global optimum.

    @param order Existing legal layout.
    @param edges Actual proof edges.
    @param precedence Required unscored format precedence.
    @param terminal Optional final goal vertex.
    @param window Maximum displacement considered for an insertion.
    @return Legal layout whose frozen cost does not increase.
    @invariant No proof vertex, edge or scope is changed.
    """
    assert isinstance(window, int) and window >= 1
    edges, precedence = set(edges), set(precedence)
    state = LayoutState(order, edges, precedence, terminal)
    if not edges:
        return list(order)
    if len(order) <= 8:
        best = list(order)
        score = state.score
        movable = list(order[:-1] if terminal is not None else order)
        for prefix in permutations(movable):
            candidate = list(prefix) + ([terminal] if terminal is not None else [])
            positions = {v: i for i, v in enumerate(candidate)}
            if not all(positions[a] < positions[b] for a, b in edges | precedence):
                continue
            value = layout_score(candidate, edges)
            if value < score:
                best, score = candidate, value
        return best
    for sweep in range(3):
        while True:
            changed = False
            for index in range(len(order) - 1):
                delta = state.adjacent(index)
                if delta is not None and delta < 0:
                    state.adjacent(index, True)
                    changed = True
            if not changed:
                break
        if sweep == 2:
            break
        for direction in (-1, 1):
            for vertex in list(state.order):
                state.improve_insertion(vertex, direction, window)
    assert state.score == layout_score(state.order, edges)
    assert state.score <= layout_score(order, edges)
    return state.order
