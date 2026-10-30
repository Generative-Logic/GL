# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Dependency-first placement of intact, recursively nested proof scopes."""
from proof_export.corpora import ORDERED_BOOK_ANCHORS
from proof_export.citation_policy import resolve_citation_source
from collections import Counter
from functools import cache
from proof_export.graph import INTEGRATION_GOAL_SCOPE, parse_scoped_expression
from proof_export.mpl import Atom, Implication, alpha_key, parse_mpl, to_mpl


def is_reader_theorem(theorem):
    """
    @brief Select main and incubator anchors for the shared reader hierarchy.

    @details Other theorem anchors retain their existing layout.
    This predicate is shared by processing and presentation.

    @param theorem Complete theorem MPL from the processed registry.
    @return Whether its outer premise is a supported proof-book anchor.
    """
    expression = parse_mpl(theorem)
    return (
        isinstance(expression, Implication)
        and isinstance(expression.premise, Atom)
        and expression.premise.head in ORDERED_BOOK_ANCHORS
    )


def theorem_alpha_keys(theorems):
    """
    @brief Compute the alpha-equivalence keys of a theorem registry once.

    @details Citation resolution matches a cited expression against registered
    theorems up to renaming of bound variables. A caller that resolves many
    chapters against one registry computes these keys once and passes them as
    `known_theorem_keys`; `resolved_citations` computes the same keys itself
    when it receives none.

    @param theorems Iterable of registered theorem MPL strings, optionally scoped.
    @return Immutable set holding the alpha key of every theorem.
    @see resolved_citations
    """
    return frozenset(alpha_key(parse_scoped_expression(t).expression) for t in theorems)


def resolved_dependencies(rows, known_theorems, known_theorem_keys=None):
    """
    @brief Resolve every local citation, including cross-scope proof edges.

    @details Replay the certificate's reverse-priority readiness rule: exact
    scoped facts precede theorem aliases, compilation may cite an imported
    theorem, and absent integration templates are not derived facts. The
    optional unproved companion of an OR license has no local proof edge.
    Every ordinary unresolved citation or dependency cycle asserts.

    @param rows Complete nonempty processed chapter rows.
    @param known_theorems Registered theorem MPL strings, including externals.
    @param known_theorem_keys Optional immutable alpha keys for that exact registry.
    @return Set of producer-to-consumer row-index pairs.
    @invariant Expressions, proof-scope markers and namespaces stay distinct.
    @see proof_export.certificate._chapter_certificate
    """
    sources = resolved_citations(rows, known_theorems, known_theorem_keys)
    return {(source, consumer) for consumer, citations in sources.items()
            for kind, source in citations if kind == "local"}


def resolved_citations(rows, known_theorems, known_theorem_keys=None):
    """
    @brief Resolve each citation with the shared scoped proof policy.

    @details The dependency graph and HTML navigation consume the same ordered
    local, theorem, template or optional dispositions. No display-only resolver
    may select an equal expression from another namespace.

    @param rows Complete reverse-stored formal chapter.
    @param known_theorems Registered theorem statements including externals.
    @param known_theorem_keys Optional immutable alpha keys for that exact registry.
    @return Row-index to ordered citation disposition/source pairs.
    @invariant Every required citation resolves before its consumer is emitted.
    """
    assert rows and all(len(r) >= 3 and (len(r) - 3) % 2 == 0 for r in rows)
    parsed = cache(parse_scoped_expression)
    key = lambda text, ns: (
        to_mpl(parsed(text).expression),
        ns,
        parsed(text).proof_scope,
    )
    alias = lambda text, ns: (
        alpha_key(parsed(text).expression),
        ns,
        parsed(text).proof_scope,
    )
    known = (theorem_alpha_keys(known_theorems)
             if known_theorem_keys is None else known_theorem_keys)
    assert isinstance(known, frozenset)
    produced = {key(r[0], r[1]) for r in rows}
    available, theorem_available, resolved = {}, {}, {}
    pending = list(range(len(rows) - 1, -1, -1))
    while pending:
        chosen = None
        chosen_sources = []
        for index in pending:
            row = rows[index]
            sources = []
            ready = True
            for j in range(3, len(row), 2):
                text, ns = row[j : j + 2]
                exact = key(text, ns)
                imported = alpha_key(parsed(text).expression) in known
                kind, source = resolve_citation_source(
                    row[2],
                    exact,
                    alias(text, ns),
                    available,
                    theorem_available,
                    produced,
                    imported,
                    exact[2] == INTEGRATION_GOAL_SCOPE,
                    optional_companion=row[2] == "or theorem",
                )
                sources.append((kind, source))
                if kind == "pending":
                    ready = False
                    break
            if ready:
                chosen, chosen_sources = index, sources
                break
        assert (
            chosen is not None
        ), f"unresolved citation or cyclic proof rows: {pending}"
        pending.remove(chosen)
        resolved[chosen] = chosen_sources
        row = rows[chosen]
        available[key(row[0], row[1])] = chosen
        if row[2] == "theorem":
            token = alias(row[0], row[1])
            assert (
                token not in theorem_available
            ), "duplicate alpha-equivalent theorem import"
            theorem_available[token] = chosen
    return resolved


def nested_order(rows, edges):
    """
    @brief Place each intact child subproof between its inputs and its users.

    @details Project each edge to immediate items at its lowest common scope.
    Topologically place ordinary rows and child blocks, preferring the existing
    reverse-stored order whenever legal. Expand recursively and store backwards.
    A child-block contraction cycle asserts instead of splitting a scope.

    @param rows Complete chapter rows with the main goal first.
    @param edges Actual local dependency pairs, including cross-scope edges.
    @return Verbatim rows in reverse expanded hierarchical order.
    @invariant Every row occurs once, every scope is contiguous, goal stays first.
    """
    assert rows and rows[0][1] == "main"
    paths, parents = {}, {"main": None}
    for row in rows:
        parts = row[1].split("_boundary_")
        assert parts[0] == "main", f"unknown scope root: {row[1]}"
        path = ["_boundary_".join(parts[:i]) for i in range(1, len(parts) + 1)]
        paths[row[1]] = path
        for a, b in zip(path, path[1:]):
            parents[b] = a
    graphs = {ns: {} for ns in parents}
    priority = {}
    for ns, parent in parents.items():
        if parent is not None:
            item = ("scope", ns)
            graphs[parent][item] = set()
            priority[item] = min(i for i, r in enumerate(rows) if ns in paths[r[1]])
    for i, row in enumerate(rows):
        item = ("row", i)
        graphs[row[1]][item] = set()
        priority[item] = i
    for a, b in edges:
        assert 0 <= a < len(rows) and 0 <= b < len(rows) and a != b
        pa, pb = paths[rows[a][1]], paths[rows[b][1]]
        common = 0
        while common < min(len(pa), len(pb)) and pa[common] == pb[common]:
            common += 1
        ns = pa[common - 1]
        x = ("row", a) if common == len(pa) else ("scope", pa[common])
        y = ("row", b) if common == len(pb) else ("scope", pb[common])
        assert x != y
        graphs[ns][y].add(x)
    graphs["main"][("row", 0)].update(
        item for item in graphs["main"] if item != ("row", 0)
    )
    orders = {}
    for ns, graph in graphs.items():
        outgoing = {item: set() for item in graph}
        for item, required in graph.items():
            for source in required:
                outgoing[source].add(item)
        pending = set(graph)
        reverse = []
        while pending:
            ready = [
                item for item in pending if not outgoing[item].intersection(pending)
            ]
            assert (
                ready
            ), f"cyclic nested subproof placement at {ns}: {sorted(pending, key=repr)}"
            item = min(ready, key=lambda x: (priority[x], repr(x)))
            reverse.append(item)
            pending.remove(item)
        orders[ns] = list(reversed(reverse))

    def expand(ns):
        """
        @brief Expand one scheduled scope into its original row indices.

        @details Child items recursively contribute their complete intervals;
        ordinary items contribute exactly one row of the owning scope.

        @param ns Exact namespace whose ordered items are expanded.
        @return Original row indices in dependency-first reader order.
        @invariant Every scheduled item contributes once to the expansion.
        """
        return [
            i
            for kind, value in orders[ns]
            for i in ([value] if kind == "row" else expand(value))
        ]

    order = expand("main")
    assert sorted(order) == list(range(len(rows))) and order[-1] == 0
    positions = {v: i for i, v in enumerate(order)}
    assert all(positions[a] < positions[b] for a, b in edges)
    result = [list(rows[i]) for i in reversed(order)]
    assert (
        Counter(map(tuple, result)) == Counter(map(tuple, rows))
        and result[0] == rows[0]
    )
    hierarchical_scope_items(result)
    return result


def hierarchical_scope_items(rows):
    """
    @brief Decode reverse storage into rows and nested cards at each scope.

    @details Each row item carries its reader index within its own scope;
    each scope item names one immediate child. Re-entering a closed scope
    asserts, so the HTML cannot hide a noncontiguous proof behind one card.

    @param rows Nonempty, reverse-stored hierarchical processed proof.
    @return Namespace-to-item lists in expanded reader order.
    @invariant Card expansion reproduces exactly the stored row sequence.
    """
    assert rows and rows[0][1] == "main"
    items, counts, closed, active = {"main": []}, {"main": 0}, set(), ["main"]
    for row in reversed(rows):
        parts = row[1].split("_boundary_")
        assert parts[0] == "main"
        path = ["_boundary_".join(parts[:i]) for i in range(1, len(parts) + 1)]
        common = 0
        while common < min(len(path), len(active)) and path[common] == active[common]:
            common += 1
        closed.update(active[common:])
        for ns in path[common:]:
            assert ns not in closed, f"noncontiguous subproof: {ns}"
            parent = "_boundary_".join(ns.split("_boundary_")[:-1])
            assert ns not in items
            items[parent].append(("scope", ns))
            items[ns], counts[ns] = [], 0
        active = path
        ns = row[1]
        items[ns].append(("row", counts[ns]))
        counts[ns] += 1
    assert sum(counts.values()) == len(rows)
    return items


def order_main_hierarchies(theorems, stacks, filename_groups, external_theorems):
    """
    @brief Finish main or incubator processing with an idempotent nested permutation.

    @details Mathematical compaction has already reached its fixed point.
    This final pass retains all inference rows and induction-section roles;
    all row contents and the book's theorem order remain unchanged.

    @param theorems Complete processed theorem registry.
    @param stacks Mapping of physical proof filenames to row lists.
    @param filename_groups Complete ordered theorem-to-file bundles.
    @param external_theorems Registered external theorem spellings.
    @return Mapping with supported proof rows hierarchically permuted.
    @invariant No proof step is added, removed or rewritten.
    """
    assert len(theorems) == len(filename_groups)
    known = [r[0] for r in theorems] + list(external_theorems)
    known_keys = theorem_alpha_keys(known)
    result = dict(stacks)
    for theorem, names in zip(theorems, filename_groups):
        if is_reader_theorem(theorem[0]):
            for name in names:
                assert name in stacks, f"missing processed chapter {name}"
                rows = stacks[name]
                from proof_reader import reader_plan, expanded_rows
                edges = resolved_dependencies(rows, known, known_keys)
                plan = reader_plan(rows, edges)
                result[name] = [list(rows[i]) for i in reversed(expanded_rows(plan))]
                hierarchical_scope_items(result[name])
    return result
