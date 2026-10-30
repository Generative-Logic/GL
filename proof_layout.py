# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Exact fixed-objective layout choices for independently scoped proof graphs."""
import math
import heapq


def layout_score(order, edges):
    """
    @brief Evaluate the frozen four-component arrangement cost with exact integers.

    @details Legal orders only: distance, private-helper gaps, maximum live
    producer frontier and crossing arcs have the original equal weights and
    normalization. A graph-constant common denominator avoids floating ties.

    @param order Complete unique vertices in legal producer-before-consumer order.
    @param edges Distinct proof edges; extra format constraints are not scored.
    @return Integer proportional to the fixed normalized cost for this graph.
    @invariant Cost comparisons never change weights to favor a candidate.
    """
    positions = {vertex: index for index, vertex in enumerate(order)}
    assert len(positions) == len(order) and order
    edges = set(edges)
    assert all(
        a in positions and b in positions and positions[a] < positions[b]
        for a, b in edges
    )
    outgoing = {vertex: [] for vertex in order}
    intervals = []
    for a, b in edges:
        outgoing[a].append(b)
        intervals.append((positions[a], positions[b]))
    private = [(a, b) for a, b in edges if len(outgoing[a]) == 1]
    distance = sum(b - a for a, b in intervals)
    gaps = sum(positions[b] - positions[a] - 1 for a, b in private)
    changes = [0] * len(order)
    for vertex, consumers in outgoing.items():
        if consumers:
            changes[positions[vertex]] += 1
            changes[max(positions[child] for child in consumers)] -= 1
    frontier = peak = 0
    for change in changes:
        frontier += change
        peak = max(peak, frontier)
    # Scan producers together so edges with a shared endpoint never cross.
    # The Fenwick tree counts prior endpoints strictly inside each interval.
    endpoint_counts = [0] * (len(order) + 1)
    crossings = 0
    for source in order:
        a = positions[source]
        for consumer in outgoing[source]:
            b = positions[consumer]
            cursor = b
            while cursor:
                crossings += endpoint_counts[cursor]
                cursor -= cursor & -cursor
            cursor = a + 1
            while cursor:
                crossings -= endpoint_counts[cursor]
                cursor -= cursor & -cursor
        for consumer in outgoing[source]:
            cursor = positions[consumer] + 1
            while cursor < len(endpoint_counts):
                endpoint_counts[cursor] += 1
                cursor += cursor & -cursor
    n, m = len(order), len(edges)
    denominators = (
        max(1, m * (n - 1)),
        max(1, len(private) * max(0, n - 2)),
        max(1, n - 1),
        max(1, m * (m - 1) // 2),
    )
    common = 4 * math.lcm(*denominators)
    return sum(
        value * (common // (4 * denominator))
        for value, denominator in zip((distance, gaps, peak, crossings), denominators)
    )


def private_helper_order(order, edges, precedence=(), terminal=None):
    """
    @brief Schedule private helper chains immediately before their consumers.

    @details Build a legal order backwards. Prefer a ready producer used only
    by the most recently placed consumer, otherwise keep the original reverse
    priority. Extra canonical-proof constraints and a terminal goal are hard
    gates, independent of the scored proof edges. Prerequisite counts and
    a rank heap avoid rescanning the remaining vertices; private producers
    are indexed by their sole consumer and retain the same rank tie order.

    @param order Initial legal order of every graph vertex.
    @param edges Actual producer-to-consumer proof edges.
    @param precedence Additional required pairs excluded from geometric cost.
    @param terminal Optional vertex required to stay last for the goal convention.
    @return Complete candidate order preserving every required precedence.
    @invariant No vertex, scope or proof edge is added or removed.
    """
    rank = {vertex: index for index, vertex in enumerate(order)}
    assert order and len(rank) == len(order)
    edges, precedence = set(edges), set(precedence)
    assert all(
        a in rank and b in rank and rank[a] < rank[b] for a, b in edges | precedence
    )
    assert terminal is None or order[-1] == terminal
    outgoing = {vertex: set() for vertex in order}
    required = {vertex: set() for vertex in order}
    for a, b in edges:
        outgoing[a].add(b)
    for a, b in edges | precedence:
        required[a].add(b)
    predecessors = {vertex: set() for vertex in order}
    private_parents = {vertex: [] for vertex in order}
    for vertex in order:
        for consumer in required[vertex]:
            predecessors[consumer].add(vertex)
        if len(outgoing[vertex]) == 1:
            private_parents[next(iter(outgoing[vertex]))].append(vertex)
    required_left = {vertex: len(required[vertex]) for vertex in order}
    ready = [(-rank[vertex], vertex) for vertex in order if not required_left[vertex]]
    heapq.heapify(ready)
    remaining = set(order)
    reverse = []
    while remaining:
        while ready and ready[0][1] not in remaining:
            heapq.heappop(ready)
        assert ready, "cyclic layout constraints"
        if not reverse and terminal is not None:
            assert required_left[terminal] == 0
            chosen = terminal
        else:
            helpers = [
                vertex for vertex in (private_parents[reverse[-1]] if reverse else ())
                if vertex in remaining and required_left[vertex] == 0
            ]
            chosen = (max(helpers, key=rank.__getitem__)
                      if helpers else heapq.heappop(ready)[1])
        reverse.append(chosen)
        remaining.remove(chosen)
        for parent in predecessors[chosen]:
            required_left[parent] -= 1
            if required_left[parent] == 0:
                heapq.heappush(ready, (-rank[parent], parent))
    return list(reversed(reverse))


def improve_private_helpers(order, edges, precedence=(), terminal=None):
    """
    @brief Accept private-helper grouping only for a strict fixed-cost improvement.

    @details Both outcomes are valid layouts: keep the supplied order when the
    candidate ties or loses, otherwise use the measured better arrangement.
    Proof and format constraints are asserted before either choice.

    @param order Existing complete legal order.
    @param edges Actual proof edges with the frozen geometric interpretation.
    @param precedence Additional required canonical-proof ordering pairs.
    @param terminal Optional terminal goal vertex.
    @return A legal order with cost no greater than the supplied order.
    @invariant Equal-cost candidates never create gratuitous artifact changes.
    """
    edges = set(edges)
    candidate = private_helper_order(order, edges, precedence, terminal)
    return (
        candidate
        if layout_score(candidate, edges) < layout_score(order, edges)
        else list(order)
    )


def frontier_order(order, edges, precedence=(), terminal=None):
    """
    @brief Schedule ready vertices to reduce the live producer frontier.

    @details A producer is live exactly while at least one consumer remains.
    Choose the ready vertex with the smallest resulting frontier, preferring
    more closed producers on a tie, then the supplied rank. Canonical proof
    precedence and a terminal goal remain hard unscored constraints.
    A heap stores those exact preference keys. Only the final remaining
    consumer gains a closed producer when a producer reaches one consumer;
    updating that key avoids rescanning every ready candidate.

    @param order Complete existing legal order.
    @param edges Actual proof dependency edges.
    @param precedence Additional required canonical-proof ordering pairs.
    @param terminal Optional goal required last.
    @return Deterministic legal frontier-oriented candidate order.
    @invariant Only vertex positions change; all graph content remains exact.
    """
    rank = {vertex: index for index, vertex in enumerate(order)}
    assert order and len(rank) == len(order)
    edges, precedence = set(edges), set(precedence)
    assert all(
        a in rank and b in rank and rank[a] < rank[b] for a, b in edges | precedence
    )
    assert terminal is None or order[-1] == terminal
    outgoing = {vertex: set() for vertex in order}
    incoming = {vertex: set() for vertex in order}
    successors = {vertex: set() for vertex in order}
    required = {vertex: 0 for vertex in order}
    for a, b in edges:
        outgoing[a].add(b)
        incoming[b].add(a)
    for a, b in edges | precedence:
        successors[a].add(b)
        required[b] += 1
    consumers_left = {vertex: set(outgoing[vertex]) for vertex in order}
    closed = {
        vertex: sum(len(outgoing[parent]) == 1 for parent in incoming[vertex])
        for vertex in order
    }
    ready = {vertex for vertex in order if required[vertex] == 0}
    options = [
        (int(bool(outgoing[vertex])) - closed[vertex], -closed[vertex], rank[vertex], vertex)
        for vertex in ready if terminal is None or vertex != terminal
    ]
    heapq.heapify(options)
    result = []
    while len(result) < len(order):
        if terminal is not None and len(result) + 1 == len(order):
            assert terminal in ready, "cyclic or impossible terminal layout"
            chosen = terminal
        else:
            while options and (options[0][3] not in ready
                               or options[0][1] != -closed[options[0][3]]):
                heapq.heappop(options)
            assert options, "cyclic or impossible terminal layout"
            chosen = heapq.heappop(options)[3]
        ready.remove(chosen)
        result.append(chosen)
        for parent in incoming[chosen]:
            consumers_left[parent].remove(chosen)
            if len(consumers_left[parent]) == 1:
                consumer = next(iter(consumers_left[parent]))
                closed[consumer] += 1
                if consumer in ready and (terminal is None or consumer != terminal):
                    heapq.heappush(options, (
                        int(bool(outgoing[consumer])) - closed[consumer],
                        -closed[consumer], rank[consumer], consumer,
                    ))
        for consumer in successors[chosen]:
            required[consumer] -= 1
            if required[consumer] == 0:
                ready.add(consumer)
                if terminal is None or consumer != terminal:
                    heapq.heappush(options, (
                        int(bool(outgoing[consumer])) - closed[consumer],
                        -closed[consumer], rank[consumer], consumer,
                    ))
    return result


def improve_frontier(order, edges, precedence=(), terminal=None):
    """
    @brief Accept frontier scheduling only for a strict fixed-objective gain.

    @details A non-improving candidate intentionally keeps the valid existing
    order. No thresholds, weights or proof edges change between the choices.

    @param order Existing complete legal order.
    @param edges Actual proof edges.
    @param precedence Additional required ordering pairs.
    @param terminal Optional terminal goal vertex.
    @return Valid order with no fixed-cost regression.
    """
    edges = set(edges)
    candidate = frontier_order(order, edges, precedence, terminal)
    return (
        candidate
        if layout_score(candidate, edges) < layout_score(order, edges)
        else list(order)
    )
