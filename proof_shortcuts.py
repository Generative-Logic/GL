# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Proof-preserving equality support shortcuts after certified graph compaction."""
from proof_export.corpora import MAIN_ANCHORS
from collections import Counter
from pathlib import Path
import json
from proof_export.mpl import Atom, Implication, Negation, parse_mpl
from proof_export.definition_projection import find_definition_projection


def shorten_equality_paths(rows, definitions):
    """
    @brief Bypass equality intermediates with already available direct evidence.

    @details Input scopes are in reversed dependency order. An equality1 result
    may use an earlier direct oriented equality outside the original two-row
    support pool. A definition projection may bypass a rewrite of arguments
    irrelevant to its exact property. Remove only a single-consumer equality1
    intermediate outside the first row of every namespace. All involved old
    and new references remain in that namespace, preserving cross-scope edges.

    @param rows Complete already ordered chapter rows.
    @param definitions Exact compiled registry for this chapter's anchor.
    @return Compacted rows using only accepted existing inference rules.
    @invariant All new supports were already available before the consumer.
    """
    assert rows and all(len(row) >= 3 and (len(row) - 3) % 2 == 0 for row in rows)
    result = [list(row) for row in rows]
    while True:
        by_key = {(row[0], row[1]): index for index, row in enumerate(result)}
        assert len(by_key) == len(result), "ambiguous exact scoped row identity"
        uses = Counter(
            (row[j], row[j + 1]) for row in result for j in range(3, len(row), 2)
        )
        first = {}
        for index, row in enumerate(result):
            first.setdefault(row[1], index)
        changed = False
        for index, row in enumerate(result):
            if row[2] not in ("equality1", "definition projection"):
                continue
            assert len(row) >= 7 if row[2] == "equality1" else len(row) == 5
            namespace = row[1]
            key = (row[3], row[4])
            if key not in by_key or row[4] != namespace:
                continue
            parent_index = by_key[key]
            parent = result[parent_index]
            if (
                parent[2] != "equality1"
                or uses[key] != 1
                or parent_index == first[namespace]
            ):
                continue
            assert len(parent) >= 7
            if any(
                r[j + 1] != namespace
                for r in (row, parent)
                for j in range(3, len(r), 2)
            ):
                continue
            assert parent_index > index, "input must preserve reader dependency order"
            replacement = None
            if row[2] == "definition projection":
                assert len(row) == 5
                if (
                    find_definition_projection(
                        parse_mpl(row[0]), parse_mpl(parent[3]), definitions
                    )
                    is not None
                ):
                    replacement = row[:3] + parent[3:5]
            else:
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
                        (a, b)
                        for a, b in zip(source.arguments, target.arguments)
                        if a != b
                    )
                )
                supports = [(f"(=[{a},{b}])", namespace) for a, b in required]
                if supports and all(
                    support in by_key and by_key[support] > index and support != key
                    for support in supports
                ):
                    replacement = (
                        row[:3]
                        + parent[3:5]
                        + [value for support in supports for value in support]
                    )
            if replacement is None:
                continue
            assert key not in set(zip(replacement[3::2], replacement[4::2]))
            result[index] = replacement
            result = [
                r for position, r in enumerate(result) if position != parent_index
            ]
            changed = True
            break
        if not changed:
            return result


def shortcut_main_equalities(theorems, stacks, filename_groups, binary_dir):
    """
    @brief Shorten equality support paths only in the main book.

    @details Matching anchor registries are mandatory. Incubator is unchanged;
    all induction parts and namespaces retain their existing ownership.

    @param theorems Complete processed theorem registry.
    @param stacks Filename-to-row mapping.
    @param filename_groups Chapter bundles parallel to the registry.
    @param binary_dir Current run's compiled registry directory.
    @return Mapping with checked equality-support shortcuts applied.
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
            result[name] = shorten_equality_paths(stacks[name], registries[anchor])
    return result
