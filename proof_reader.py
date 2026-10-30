# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Reader-only operation blocks over the complete, checked proof graph."""
import heapq
from collections import defaultdict

DISINTEGRATION_TAGS = frozenset({"expansion", "disintegration"})
INTEGRATION_PREFIX = "reformulation for integration"
BOUNDARY_TAGS = frozenset(
    {
        "task formulation",
        "recursion",
        "premise element",
        "or disintegration",
        "or branch assumption",
        "zoom in",
        "theorem",
        "externally provided theorem",
        "anchor handling",
    }
)


def row_node(index):
    """
    @brief Construct one formal row leaf.

    @details Row indices address the unchanged input graph, including imports.

    @param index Index of the formal row.
    @return Leaf record with exactly one member.
    """
    return {"kind": "row", "members": frozenset({index}), "index": index}


def block_node(kind, children, **metadata):
    """
    @brief Construct a block without replacing any formal proof row.

    @details Children must be disjoint and collectively own the block's rows.
    Metadata supplies the scope, original source or goal shown in its header.

    @param kind Semantic block kind.
    @param children Nonempty sequence of row or block records.
    @param metadata Header information independent of proof inference.
    @return Block containing every child row exactly once.
    @invariant Reader headers never enter the formal row collection.
    """
    assert children, "empty reader block"
    members = frozenset(i for child in children for i in child["members"])
    assert len(members) == sum(len(child["members"]) for child in children)
    return {"kind": kind, "children": list(children), "members": members, **metadata}


def expanded_rows(node):
    """
    @brief Expand a reader tree to the original formal row indices.

    @details Expansion retains imports and all nested proof scopes.

    @param node Root, block or row record.
    @return Indices in reader order.
    """
    if node["kind"] == "row":
        return [node["index"]]
    return [i for child in node["children"] for i in expanded_rows(child)]


def schedule_nodes(nodes, edges, terminal=None, group_or=True):
    """
    @brief Schedule whole reader units with all dependencies before consumers.

    @details Project formal edges onto units. OR sibling cards form one
    consecutive scheduling cohort, without adding a counted wrapper header.
    Stable input order breaks ties; OR cards additionally use namespace order.

    @param nodes Disjoint row or block units.
    @param edges Formal producer-to-consumer row-index pairs.
    @param terminal Optional formal goal that must be last in the expansion.
    @param group_or Whether sibling OR cards form a consecutive cohort.
    @return Dependency-first units with no introduced proof rows.
    @invariant A cyclic block contraction asserts rather than hiding a forward use.
    """
    if not nodes:
        return []
    or_nodes = [n for n in nodes if n["kind"] in ("orint", "ordis")]
    if group_or and len(or_nodes) > 1:
        other = [n for n in nodes if n["kind"] not in ("orint", "ordis")]
        cohort = block_node("cohort", schedule_nodes(or_nodes, edges, group_or=False))
        ordered = schedule_nodes(other + [cohort], edges, terminal, False)
        return [
            part
            for node in ordered
            for part in (node["children"] if node["kind"] == "cohort" else [node])
        ]
    owner = {row: j for j, node in enumerate(nodes) for row in node["members"]}
    assert len(owner) == sum(len(n["members"]) for n in nodes)
    required = {j: set() for j in range(len(nodes))}
    following = {j: set() for j in required}
    for a, b in edges:
        if a in owner and b in owner and owner[a] != owner[b]:
            required[owner[b]].add(owner[a])
    if terminal in owner:
        required[owner[terminal]].update(j for j in required if j != owner[terminal])
    for j, sources in required.items():
        for source in sources:
            following[source].add(j)
    pending = set(required)
    counts = {j: len(sources) for j, sources in required.items()}
    priority = {j: (-max(node["members"]), node.get("namespace", ""), j)
                for j, node in enumerate(nodes)}
    ready = [priority[j] for j, count in counts.items() if count == 0]
    heapq.heapify(ready)
    result = []
    while pending:
        assert (
            ready
        ), f"cyclic reader block contraction: {[nodes[j]['kind'] for j in pending]}"
        chosen = heapq.heappop(ready)[2]
        result.append(nodes[chosen])
        pending.remove(chosen)
        for consumer in following[chosen]:
            counts[consumer] -= 1
            if counts[consumer] == 0:
                heapq.heappush(ready, priority[consumer])
    return result


def reader_plan(rows, edges):
    """
    @brief Build nested scope, integration and disintegration reader blocks.

    @details Disintegration groups the transparent expansion chain of a single
    source. Integration groups a recorded construction application with its
    private support, including complete child scopes. Shared premises stay
    outside. Every outgoing product and each scope goal is recorded for headers.

    @param rows Complete reverse-stored processed chapter, main goal first.
    @param edges Resolved local proof dependencies from the shared citation policy.
    @return Root tree with exactly the original rows, in dependency-first order.
    @invariant No scope is split, no row is dropped, and no header is an inference.
    """
    assert rows and rows[0][1] == "main"
    assert all(0 <= a < len(rows) and 0 <= b < len(rows) and a != b for a, b in edges)
    by_key = {(r[0], r[1]): i for i, r in enumerate(rows)}
    assert len(by_key) == len(rows), "ambiguous scoped proof identity"
    incoming, outgoing = defaultdict(set), defaultdict(set)
    for a, b in edges:
        incoming[b].add(a)
        outgoing[a].add(b)
    scopes = {"main": []}
    parents = {}
    for i, row in enumerate(rows):
        path = row[1].split("_boundary_")
        assert path[0] == "main"
        for length in range(2, len(path) + 1):
            child = "_boundary_".join(path[:length])
            parents[child] = "_boundary_".join(path[: length - 1])
            scopes.setdefault(child, [])
        scopes[row[1]].append(i)
    built = {}
    for namespace in sorted(scopes, key=lambda s: (-s.count("_boundary_"), s)):
        nodes = [row_node(i) for i in scopes[namespace]]
        nodes += [
            built[child] for child, parent in parents.items() if parent == namespace
        ]
        chains = defaultdict(set)
        for i in scopes[namespace]:
            row = rows[i]
            if row[2] not in DISINTEGRATION_TAGS or len(row) != 5:
                continue
            source = (row[3], row[4])
            seen = {i}
            while source in by_key:
                parent = by_key[source]
                assert parent not in seen, "cyclic disintegration chain"
                seen.add(parent)
                previous = rows[parent]
                if (
                    previous[1] != namespace
                    or previous[2] not in DISINTEGRATION_TAGS
                    or len(previous) != 5
                ):
                    break
                source = (previous[3], previous[4])
            chains[source].add(i)
        for source, members in sorted(chains.items()):
            if not any(rows[i][2] == "disintegration" for i in members):
                continue
            children = [n for n in nodes if n["members"] <= members]
            assert frozenset(i for n in children for i in n["members"]) == members
            exports = sorted(
                (
                    i
                    for i in members
                    if not outgoing[i] <= members or i == min(scopes[namespace])
                ),
                reverse=True,
            )
            block = block_node(
                "disintegration",
                schedule_nodes(children, edges, 0 if 0 in members else None),
                source=source,
                outputs=exports,
                namespace=namespace,
            )
            nodes = [n for n in nodes if n["members"].isdisjoint(members)] + [block]
        for goal in sorted(scopes[namespace], reverse=True):
            row = rows[goal]
            if row[2] != "implication" or len(row) < 5:
                continue
            rule = by_key.get((row[3], row[4]))
            if rule is None:
                continue
            templates, todo = set(), [rule]
            while todo:
                index = todo.pop()
                if index in templates:
                    continue
                tag = rows[index][2]
                if (
                    tag.startswith(INTEGRATION_PREFIX)
                    or tag in ("expansion for integration", "expansion")
                    or tag in ("implication grouping", "compilation")
                ):
                    templates.add(index)
                    todo.extend(incoming[index])
            if not any(rows[i][2].startswith(INTEGRATION_PREFIX) for i in templates):
                continue
            owner = {i: j for j, n in enumerate(nodes) for i in n["members"]}
            selected = {owner[goal]}
            changed = True
            while changed:
                changed = False
                members = frozenset(i for j in selected for i in nodes[j]["members"])
                for j, node in enumerate(nodes):
                    if j in selected:
                        continue
                    if not any(
                        outgoing[i].intersection(members) for i in node["members"]
                    ):
                        continue
                    if any(
                        outgoing[i] - members - node["members"] for i in node["members"]
                    ):
                        continue
                    if (
                        node["kind"] == "row"
                        and rows[node["index"]][2] in BOUNDARY_TAGS
                    ):
                        continue
                    selected.add(j)
                    changed = True
            children = [n for j, n in enumerate(nodes) if j in selected]
            block = block_node(
                "integration",
                schedule_nodes(children, edges, goal),
                goal=goal,
                outputs=[goal],
                namespace=namespace,
            )
            nodes = [n for j, n in enumerate(nodes) if j not in selected] + [block]
        terminal = 0 if namespace == "main" else None
        ordered = schedule_nodes(nodes, edges, terminal)
        payload = namespace.rsplit("_boundary_", 1)[-1]
        kind = (
            "root"
            if namespace == "main"
            else (
                "orint"
                if payload.startswith("orint_")
                else "ordis" if payload.startswith("ordis_") else "subproof"
            )
        )
        own_goal = min(scopes[namespace]) if scopes[namespace] else None
        built[namespace] = block_node(kind, ordered, namespace=namespace, goal=own_goal)
    root = built["main"]
    order = expanded_rows(root)
    assert sorted(order) == list(range(len(rows))), "reader row coverage changed"
    positions = {i: j for j, i in enumerate(order)}
    assert all(
        positions[a] < positions[b] for a, b in edges
    ), "reader forward dependency"
    assert order[-1] == 0, "reader main goal must close the proof"
    return root


def reader_metrics(root, rows):
    """
    @brief Count visible main rows and all formal steps independently.

    @details A top-level closed semantic block counts once. Imports stay part
    of the formal count but are not displayed. Nested blocks are counted by
    kind as diagnostics, never as additional top-level rows.

    @param root Complete reader plan.
    @param rows Original formal proof rows.
    @return Counts suitable for summing across all parts of a reader chapter.
    """
    counts = defaultdict(int)
    pending = [root]
    while pending:
        node = pending.pop()
        if node["kind"] != "row":
            counts[node["kind"]] += 1
            pending.extend(node["children"])
    main_rows = sum(
        n["kind"] != "row" or rows[n["index"]][2] != "theorem" for n in root["children"]
    )
    return {"main_rows": main_rows, "proof_steps": len(rows), "blocks": dict(counts)}
