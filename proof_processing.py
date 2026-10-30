# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Proof-preserving compaction and ordering of the main proof book."""

from proof_export.corpora import MAIN_ANCHORS, ORDERED_BOOK_ANCHORS
from proof_export.citation_policy import resolve_citation_source
from collections import Counter, defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from functools import cache
from pathlib import Path
import json
import heapq

from proof_layout_search import search_layout
from proof_layout import improve_private_helpers, improve_frontier

from proof_export.certificate import theorem_citations_with_tags
from proof_export.definition_projection import find_definition_projection
from proof_export.graph import Chapter, Dependency, ProofRow, TheoremRecord

from proof_export.graph import INTEGRATION_GOAL_SCOPE, parse_scoped_expression
from proof_export.mpl import Atom, Implication, Negation, alpha_key, parse_mpl


_SCOPE_ORDER_CACHE = ContextVar("proof_main_scope_order_cache", default=None)


@contextmanager
def main_scope_order_cache():
    """
    @brief Isolate unchanged-chapter ordering reuse to one processing call.

    @details Complete immutable rows, registered theorem keys and all layout
    options identify a result. Context exit releases every cached chapter,
    and nested invocations restore their caller's separate cache.

    @return Context manager owning a fresh scope-order result dictionary.
    """
    token = _SCOPE_ORDER_CACHE.set({})
    try:
        yield
    finally:
        _SCOPE_ORDER_CACHE.reset(token)


def citation_key(text):
    """
    @brief Identify a citation without confusing its binders or proof scope.

    @details Bound names are alpha-normalized; free arguments stay exact.
    Integration-goal templates retain a distinct key from ordinary facts.

    @param text One complete processed expression, optionally scope-marked.
    @return Pair of alpha-normalized expression and proof-scope kind.
    """
    scoped = parse_scoped_expression(text)
    return alpha_key(scoped.expression), scoped.proof_scope


def ordered_scope_indices(
    rows,
    positions,
    known_theorems,
    key_for,
    group_private=False,
    group_frontier=False,
    group_search=False,
):
    """
    @brief Order one scope's rows with every local producer before its consumer.

    @details Exact local citations take precedence over theorem aliases.
    Cross-scope references retain their recorded interfaces. A registered
    theorem, an integration template, and the optional second OR license are
    the only nonlocal same-scope citations. The first physical chapter row is
    reserved for the end of the main scope's reader order.

    @param rows Complete chapter rows, unchanged by this function.
    @param positions Physical positions belonging to exactly one namespace.
    @param known_theorems Alpha/proof-scope keys of registered theorem imports.
    @param key_for Call-local cached citation-key function.
    @param group_private Whether to accept strictly better private-helper grouping.
    @param group_search Whether to run exact tiny or bounded insertion search.
    @param group_frontier Whether to accept strictly better frontier scheduling.
    @return Physical row indices in dependency-first reader order.
    @invariant No row is dropped and no namespace or dependency is rewritten.
    @see proof_export.certificate._chapter_certificate
    """
    assert positions and all(0 <= i < len(rows) for i in positions)
    namespace = rows[positions[0]][1]
    assert all(rows[i][1] == namespace for i in positions)
    produced = {rows[i][0] for i in positions}
    theorem_aliases = {
        key_for(rows[i][0]) for i in positions if rows[i][2] == "theorem"
    }
    exact_sources = {expression: ("exact", expression) for expression in produced}
    alias_sources = {key: ("theorem", key) for key in theorem_aliases}
    requirements = {}
    for index in positions:
        row = rows[index]
        assert len(row) >= 3 and (len(row) - 3) % 2 == 0
        required = set()
        for field in range(3, len(row), 2):
            expression, source_namespace = row[field : field + 2]
            if source_namespace != namespace:
                continue
            key = key_for(expression)
            kind, source = resolve_citation_source(
                row[2],
                expression,
                key,
                exact_sources,
                alias_sources,
                produced,
                key in known_theorems,
                key[1] == INTEGRATION_GOAL_SCOPE,
                optional_companion=row[2] == "or theorem" and field == 5,
            )
            assert (
                kind != "pending"
            ), f"unresolved citation at physical row {index+1}: {expression} ({namespace})"
            if kind == "local":
                required.add(source)
        requirements[index] = required

    pending = list(reversed(positions))
    available = set()
    sources = {}
    edges = set()
    ordered = []
    while pending:
        ready = next(
            (
                index
                for index in pending
                if requirements[index] <= available
                and (index != 0 or len(pending) == 1)
            ),
            None,
        )
        assert ready is not None, (
            f"scope cannot be ordered with the chapter goal last: {namespace}; "
            f"remaining physical rows {[i+1 for i in pending]}"
        )
        pending.remove(ready)
        ordered.append(ready)
        edges.update((sources[token], ready) for token in requirements[ready])
        row = rows[ready]
        available.add(("exact", row[0]))
        sources.setdefault(("exact", row[0]), ready)
        if row[2] == "theorem":
            available.add(("theorem", key_for(row[0])))
            sources.setdefault(("theorem", key_for(row[0])), ready)
    terminal = 0 if 0 in positions else None
    if group_private:
        ordered = improve_private_helpers(ordered, edges, terminal=terminal)
    if group_frontier:
        ordered = improve_frontier(ordered, edges, terminal=terminal)
    if group_search:
        ordered = search_layout(ordered, edges, terminal=terminal)
        if len(ordered) > 8:
            ordered = search_layout(
                ordered, edges, terminal=terminal, window=len(ordered) - 1
            )
    return ordered


def order_scope_rows(
    rows,
    known_theorems,
    key_for,
    group_private=False,
    group_frontier=False,
    group_search=False,
):
    """
    @brief Permute rows inside their original namespace positions.

    @details The renderer reverses each scope's stored rows, so the result
    stores the dependency order backwards. Namespace positions, child-section
    discovery order, empty rows and the first goal row stay unchanged.

    @param rows Nonempty chapter with its main-scope goal first.
    @param known_theorems Alpha/proof-scope keys of known theorem imports.
    @param key_for Call-local cached citation-key function.
    @param group_private Whether to accept strictly better private-helper grouping.
    @param group_search Whether to run exact tiny or bounded insertion search.
    @param group_frontier Whether to accept strictly better frontier scheduling.
    @return Independent row lists with the same exact row multiset.
    @invariant The first row and every row's namespace position are preserved.
    @see ordered_scope_indices
    """
    assert rows and len(rows[0]) >= 3 and rows[0][1] == "main"
    positions_by_scope = defaultdict(list)
    for index, row in enumerate(rows):
        if row:
            assert len(row) >= 3 and (len(row) - 3) % 2 == 0
            positions_by_scope[row[1]].append(index)
    result = [list(row) for row in rows]
    for positions in positions_by_scope.values():
        ordered = ordered_scope_indices(
            rows,
            positions,
            known_theorems,
            key_for,
            group_private,
            group_frontier,
            group_search,
        )
        for target, source in zip(positions, reversed(ordered)):
            result[target] = list(rows[source])
    assert result[0] == rows[0]
    assert [r[1] if r else None for r in result] == [r[1] if r else None for r in rows]
    assert sorted(map(tuple, result)) == sorted(map(tuple, rows))
    return result


def order_main_scope_rows(
    theorems,
    stacks,
    filename_groups,
    external_theorems,
    group_private=False,
    group_frontier=False,
    group_search=False,
):
    """
    @brief Order every scope of the main or incubator proof book.

    @details The theorem registry and filename groups stay in their original
    order. Main and incubator use the same ordering policy. Every selected
    row remains verbatim; cached alpha keys live only for this processing call.

    @param theorems Processed theorem-list rows.
    @param stacks Filename-to-processed-chapter row mapping.
    @param filename_groups Complete ordered filenames for each theorem.
    @param external_theorems Registered external theorem spellings.
    @param group_private Whether to optimize each already separated scope layout.
    @param group_search Whether to run exact tiny or bounded insertion search.
    @param group_frontier Whether to optimize each scope with frontier scheduling.
    @return Chapter mapping with supported scopes ordered for the reader.
    @invariant No theorem, proof row, scope or dependency is added or removed.
    @see process_proof_graphs.create_processed_proof_graph
    """
    assert len(theorems) == len(filename_groups)
    key_for = cache(citation_key)
    known_theorems = {key_for(row[0]) for row in theorems}
    known_theorems.update(key_for(text) for text in external_theorems)
    known_theorems = frozenset(known_theorems)
    cached = _SCOPE_ORDER_CACHE.get()
    result = dict(stacks)
    for theorem, filenames in zip(theorems, filename_groups):
        expression = parse_mpl(theorem[0])
        assert isinstance(expression, Implication) and isinstance(
            expression.premise, Atom
        )
        if expression.premise.head not in ORDERED_BOOK_ANCHORS:
            continue
        for filename in filenames:
            assert filename in stacks, f"missing processed chapter {filename}"
            rows = stacks[filename]
            if cached is None:
                result[filename] = order_scope_rows(
                    rows,
                    known_theorems,
                    key_for,
                    group_private,
                    group_frontier,
                    group_search,
                )
            else:
                key = (
                    tuple(map(tuple, rows)),
                    known_theorems,
                    group_private,
                    group_frontier,
                    group_search,
                )
                if key not in cached:
                    ordered = order_scope_rows(
                        rows,
                        known_theorems,
                        key_for,
                        group_private,
                        group_frontier,
                        group_search,
                    )
                    cached[key] = tuple(map(tuple, ordered))
                result[filename] = [list(row) for row in cached[key]]
    return result


def arrange_anchor_stretches(order, anchors, texts, proof_edges, precedence, search_window=None):
    """
    @brief Compact each anchor stretch of a citation-sorted book and end it with its deepest theorem.

    @details A stretch is a maximal run of consecutive chapters with the same
    anchor in ``order``. The layout passes (private helpers, frontier, then the
    insertion search) rearrange each stretch using only the citations and
    canonical-copy precedences between its own chapters. No chapter leaves its
    stretch, so every citation keeps pointing backwards and the stretches keep
    their order. Then the stretch's deepest theorem that no other chapter of
    the stretch cites or must precede moves to the stretch's end. Depth is the
    number of distinct theorems a theorem rests on through citations, across
    the whole book; ties go to the larger theorem text, so the choice does not
    depend on the order and repeated processing rounds settle
    (D-592).

    @param order Vertex ids in citation-sorted order: every proof edge and
    precedence pair points forward.
    @param anchors Anchor name of each vertex, indexed by vertex id.
    @param texts Theorem text of each vertex, indexed by vertex id; breaks
    depth ties.
    @param proof_edges Citation pairs (source, consumer), scored by the layout.
    @param precedence Canonical-copy pairs (original, copy), kept but not scored.
    @param search_window Optional bounded insertion distance inside a stretch;
    None runs the complete two-stage search.
    @return The rearranged vertex order.
    @invariant Every proof edge and precedence pair still points forward; each
    stretch keeps its position and membership.
    @see order_main_book
    """
    position = {vertex: index for index, vertex in enumerate(order)}
    assert len(position) == len(order)
    assert all(
        position[a] < position[b] for a, b in set(proof_edges) | set(precedence)
    ), "the book order is not citation-sorted"
    assert search_window is None or (isinstance(search_window, int) and search_window >= 1)
    sources = defaultdict(list)
    for a, b in proof_edges:
        sources[b].append(a)
    ancestors = {}
    for vertex in order:
        mask = 0
        for source in sources[vertex]:
            mask |= ancestors[source] | (1 << position[source])
        ancestors[vertex] = mask
    stretches = []
    for vertex in order:
        if stretches and anchors[stretches[-1][-1]] == anchors[vertex]:
            stretches[-1].append(vertex)
        else:
            stretches.append([vertex])
    result = []
    for stretch in stretches:
        members = set(stretch)
        inner_edges = {(a, b) for a, b in proof_edges if a in members and b in members}
        inner_precedence = {(a, b) for a, b in precedence if a in members and b in members}
        arranged = improve_private_helpers(stretch, inner_edges, inner_precedence)
        arranged = improve_frontier(arranged, inner_edges, inner_precedence)
        if search_window is None:
            arranged = search_layout(arranged, inner_edges, inner_precedence)
            if len(arranged) > 8:
                arranged = search_layout(
                    arranged, inner_edges, inner_precedence, window=len(arranged) - 1
                )
        else:
            arranged = search_layout(
                arranged, inner_edges, inner_precedence, window=search_window
            )
        assert sorted(arranged) == sorted(stretch)
        cited = {a for a, _ in inner_edges | inner_precedence}
        depth = lambda vertex: (ancestors[vertex].bit_count(), texts[vertex])
        candidates = [vertex for vertex in arranged if vertex not in cited]
        deepest = max(candidates, key=depth)
        assert [depth(vertex) for vertex in candidates].count(depth(deepest)) == 1, \
            "two equally deep uncited theorems share one text"
        result += [vertex for vertex in arranged if vertex != deepest] + [deepest]
    return result


def order_main_book(theorems, stacks, filename_groups, search_window=None):
    """
    @brief Order main or incubator chapters: citation sort, then compaction inside anchor stretches.

    @details Step 1: the certificate citation contract supplies the dependency
    graph, and a stable topological order takes the lowest incoming ready
    index, so a chapter moves later only when it cites a theorem listed after
    it; the registry arrives in the prover's order, batch by batch. Equivalent
    copies retain their original canonical source by an additional precedence
    constraint. Steps 2 and 3 (`arrange_anchor_stretches`): the layout passes
    compact each anchor stretch, a maximal run of consecutive chapters with one
    anchor, and each stretch ends with its deepest theorem that nothing in the
    stretch cites (D-592). Each induction bundle stays
    together; all rows and namespace positions remain verbatim. Unsupported
    books retain their order.

    @param theorems Complete processed theorem registry in source order.
    @param stacks Filename-to-row mapping for the complete book.
    @param filename_groups Chapter filename bundles parallel to the registry.
    @param search_window Optional bounded insertion distance inside a stretch;
    None runs the complete two-stage search used by the main book.
    @return Pair of permuted registry and consistently renumbered chapter mapping.
    @invariant Every original theorem and chapter body occurs exactly once.
    @invariant Every cited theorem precedes its consumer, and no chapter leaves
    its anchor stretch.
    @see proof_export.certificate.theorem_citations_with_tags
    """
    assert len(theorems) == len(filename_groups)
    expressions = [parse_mpl(row[0]) for row in theorems]
    assert all(
        isinstance(e, Implication) and isinstance(e.premise, Atom) for e in expressions
    )
    main = [e.premise.head in MAIN_ANCHORS for e in expressions]
    supported = [e.premise.head in ORDERED_BOOK_ANCHORS for e in expressions]
    if not any(supported):
        return theorems, stacks
    assert all(supported), "supported and unsupported books cannot be mixed"
    assert all(main) or not any(main), "main and incubator books cannot be mixed"
    canonical = {}
    for index, expression in enumerate(expressions):
        canonical.setdefault(alpha_key(expression), index)
    dependencies = {index: set() for index in range(len(theorems))}
    proof_edges = set()
    precedence = set()
    for index, (theorem, filenames) in enumerate(zip(theorems, filename_groups)):
        chapters = []
        for filename in filenames:
            rows = []
            for line, fields in enumerate(stacks[filename], 1):
                assert len(fields) >= 3 and (len(fields) - 3) % 2 == 0
                rows.append(
                    ProofRow(
                        line,
                        parse_scoped_expression(fields[0]),
                        fields[1],
                        fields[2],
                        tuple(
                            Dependency(
                                parse_scoped_expression(fields[j]), fields[j + 1]
                            )
                            for j in range(3, len(fields), 2)
                        ),
                    )
                )
            number, role = filename[:-4].split("_", 1)
            chapters.append(Chapter(Path(filename), int(number), role, tuple(rows)))
        record = TheoremRecord(
            index, expressions[index], theorem[1], theorem[2], tuple(chapters)
        )
        for tag, cited in theorem_citations_with_tags(record):
            key = alpha_key(cited)
            if key in canonical:
                dependencies[index].add(canonical[key])
                proof_edges.add((canonical[key], index))
        preferred = canonical[alpha_key(expressions[index])]
        if preferred != index:
            dependencies[index].add(preferred)
            precedence.add((preferred, index))
    following = {index: [] for index in dependencies}
    counts = {index: len(sources) for index, sources in dependencies.items()}
    for index, sources in dependencies.items():
        for source in sources:
            following[source].append(index)
    ready = [index for index, count in counts.items() if count == 0]
    heapq.heapify(ready)
    ordered = []
    while ready:
        chosen = heapq.heappop(ready)
        ordered.append(chosen)
        for consumer in following[chosen]:
            counts[consumer] -= 1
            if counts[consumer] == 0:
                heapq.heappush(ready, consumer)
    assert len(ordered) == len(dependencies), "book citations or canonical-proof constraints are cyclic"
    ordered = arrange_anchor_stretches(
        ordered,
        [expression.premise.head for expression in expressions],
        [theorem[0] for theorem in theorems],
        proof_edges,
        precedence,
        search_window,
    )
    result = {}
    number = 0
    for index in ordered:
        for filename in filename_groups[index]:
            result[f"{number}_{filename.split('_', 1)[1]}"] = stacks[filename]
            number += 1
    assert len(result) == len(stacks)
    return [theorems[index] for index in ordered], result


def fuse_equality_chains(rows):
    """
    @brief Fuse consecutive substitutions justified by their explicit equalities.

    @details Only the existing equality1 rule is used. Every differing argument
    needs its directly oriented equality from the two source rows. The removed
    intermediate has one consumer, shares the result namespace, and cannot be
    a namespace's first physical row. Each move removes one row, so the search
    ends at a fixed point without changing any theorem or scope boundary.

    @param rows Complete processed chapter rows with the goal first.
    @return Compacted independent rows preserving all remaining conclusions.
    @invariant No assumption, definition, witness or scope-boundary row is removed.
    @see verifier.check_equality1
    """
    result = [list(row) for row in rows]
    while True:
        by_key = {(row[0], row[1]): index for index, row in enumerate(result)}
        assert len(by_key) == len(result), "ambiguous exact scoped row identity"
        consumers = Counter(
            (row[j], row[j + 1]) for row in result for j in range(3, len(row), 2)
        )
        first = {}
        for index, row in enumerate(result):
            first.setdefault(row[1], index)
        changed = False
        for index, row in enumerate(result):
            if row[2] != "equality1":
                continue
            assert len(row) >= 7 and (len(row) - 3) % 2 == 0
            parent_key = (row[3], row[4])
            if parent_key not in by_key:
                continue
            parent_index = by_key[parent_key]
            parent = result[parent_index]
            if (
                parent[2] != "equality1"
                or parent[1] != row[1]
                or consumers[parent_key] != 1
                or parent_index == first[parent[1]]
            ):
                continue
            assert len(parent) >= 7 and (len(parent) - 3) % 2 == 0
            source, target = parse_mpl(parent[3]), parse_mpl(row[0])
            assert isinstance(source, Negation) == isinstance(target, Negation)
            if isinstance(source, Negation):
                source, target = source.inner, target.inner
            assert isinstance(source, Atom) and isinstance(target, Atom)
            assert source.head == target.head and len(source.arguments) == len(
                target.arguments
            )
            required = list(
                dict.fromkeys(
                    (a, b) for a, b in zip(source.arguments, target.arguments) if a != b
                )
            )
            if not required:
                continue
            equalities = {}
            for cited_row in (parent, row):
                for field in range(5, len(cited_row), 2):
                    equality = parse_mpl(cited_row[field])
                    assert (
                        isinstance(equality, Atom)
                        and equality.head == "="
                        and len(equality.arguments) == 2
                    )
                    namespace = cited_row[field + 1]
                    assert namespace == row[1] or row[1].startswith(
                        namespace + "_boundary_"
                    )
                    equalities.setdefault(
                        equality.arguments, cited_row[field : field + 2]
                    )
            if not all(pair in equalities for pair in required):
                continue
            assert parent[4] == row[1] or row[1].startswith(parent[4] + "_boundary_")
            replacement = row[:3] + parent[3:5]
            for pair in required:
                replacement += equalities[pair]
            result[index] = replacement
            result = [
                entry
                for position, entry in enumerate(result)
                if position != parent_index
            ]
            changed = True
            break
        if not changed:
            return result


def compact_main_equalities(theorems, stacks, filename_groups):
    """
    @brief Apply equality-chain fusion only to chapters in the main proof book.

    @details Every induction part and every subproof remains in its original
    chapter. Incubator rows are returned unchanged. Fusion starts from the accepted
    scope order and preserves it because every new citation was already an
    ancestor of the consumer. The actual compacted graph feeds all outputs.

    @param theorems Processed theorem registry.
    @param stacks Complete filename-to-row mapping.
    @param filename_groups Chapter filename bundles parallel to the registry.
    @return Chapter mapping with only validated local equality-chain moves.
    @invariant The theorem registry and chapter bundle structure are unchanged.
    @see fuse_equality_chains
    """
    assert len(theorems) == len(filename_groups)
    result = dict(stacks)
    for theorem, names in zip(theorems, filename_groups):
        expression = parse_mpl(theorem[0])
        assert isinstance(expression, Implication) and isinstance(
            expression.premise, Atom
        )
        if expression.premise.head in MAIN_ANCHORS:
            for name in names:
                result[name] = fuse_equality_chains(stacks[name])
    return result


def compact_definition_projections(rows, definitions):
    """
    @brief Replace positive conjunction expansion chains by checked direct projections.

    @details Input scopes are already in reversed dependency order. Only existing
    expansion/disintegration pairs and their new positive projection chains are
    shortened, in exactly the same namespace. Remove an unused transparent
    intermediate only outside each namespace's first row. Ordinary assumptions,
    witnesses, other rules and all cross-scope interfaces remain untouched.

    @param rows Complete dependency-ordered chapter rows, goal first physically.
    @param definitions Exact compiled registry for this chapter's anchor.
    @return Shorter or equal-sized chapter using explicit definition projections.
    @invariant No new free argument, scope, theorem or witness is introduced.
    @see proof_export.definition_projection.build_definition_projection_plan
    """
    result = [list(row) for row in rows]
    transparent_expansions = set()
    plan_for = cache(
        lambda target, source: find_definition_projection(
            parse_mpl(target), parse_mpl(source), definitions
        )
    )
    while True:
        by_key = {(row[0], row[1]): index for index, row in enumerate(result)}
        assert len(by_key) == len(result), "ambiguous exact scoped row identity"
        changed = False
        for index, row in enumerate(result):
            if row[2] not in {"disintegration", "definition projection"}:
                continue
            assert len(row) == 5
            if row[1] != row[4] or (row[3], row[4]) not in by_key:
                continue
            parent = result[by_key[(row[3], row[4])]]
            expected = (
                "expansion" if row[2] == "disintegration" else "definition projection"
            )
            if parent[2] != expected or len(parent) != 5 or parent[1] != parent[4]:
                continue
            assert (
                by_key[(row[3], row[4])] > index
            ), "input must preserve reader dependency order"
            source = parse_scoped_expression(parent[3])
            target = parse_scoped_expression(row[0])
            if source.proof_scope != "ordinary" or target.proof_scope != "ordinary":
                continue
            plan = plan_for(row[0], parent[3])
            if plan is None:
                continue
            replacement = [
                row[0],
                row[1],
                "definition projection",
                parent[3],
                parent[4],
            ]
            assert (
                replacement != row
            ), "positive projection cannot cycle to its own source"
            if parent[2] == "expansion":
                transparent_expansions.add((parent[0], parent[1]))
            result[index] = replacement
            changed = True
        consumed = {
            (row[j], row[j + 1]) for row in result for j in range(3, len(row), 2)
        }
        first = {}
        for index, row in enumerate(result):
            first.setdefault(row[1], index)
        retained = []
        for index, row in enumerate(result):
            key = (row[0], row[1])
            removable = (
                row[2] == "definition projection" or key in transparent_expansions
            )
            if removable and key not in consumed and index != first[row[1]]:
                changed = True
            else:
                retained.append(row)
        result = retained
        if not changed:
            return result


def compact_main_definitions(theorems, stacks, filename_groups, binary_dir):
    """
    @brief Compact main-book definition projections against exact anchor registries.

    @details Each selected anchor loads its own existing GL binary. Incubator
    follows its original path with no registry read or row rewrite. Missing
    matching main registries assert rather than selecting another batch.

    @param theorems Complete processed theorem registry.
    @param stacks Filename-to-chapter row mapping.
    @param filename_groups Complete chapter bundles parallel to the registry.
    @param binary_dir Directory holding the current run's GL binary registries.
    @return Chapter mapping with same-scope transparent projection chains compacted.
    @invariant Native prover output and theorem membership remain unchanged.
    """
    assert len(theorems) == len(filename_groups)
    result = dict(stacks)
    registries = {}
    for theorem, names in zip(theorems, filename_groups):
        expression = parse_mpl(theorem[0])
        assert isinstance(expression, Implication) and isinstance(
            expression.premise, Atom
        )
        anchor = expression.premise.head
        if anchor not in MAIN_ANCHORS:
            continue
        if anchor not in registries:
            path = Path(binary_dir) / f"GL_binary_{anchor.removeprefix('Anchor')}.json"
            assert path.is_file(), f"missing exact main registry: {path}"
            registries[anchor] = json.loads(path.read_text(encoding="utf-8"))
        for name in names:
            result[name] = compact_definition_projections(
                stacks[name], registries[anchor]
            )
    return result
