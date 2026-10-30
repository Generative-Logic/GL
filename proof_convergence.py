# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Converge main proof rewrites and bound Incubator arrangement work."""
from collections import Counter
from proof_export.corpora import INCUBATOR_ANCHORS
from proof_export.mpl import Atom, Implication, parse_mpl
from proof_layout_search import layout_search_cache
from proof_processing import (
    main_scope_order_cache,
    order_main_scope_rows,
    order_main_book,
    compact_main_equalities,
    compact_main_definitions,
)
from proof_shortcuts import shortcut_main_equalities
from proof_goal_slice import apply_main_rows as slice_main_rows
from proof_rewrite_placement import apply_main_rows as place_main_rewrites


def process_incubator_book(theorems, stacks, filename_groups_for, external_theorems):
    """
    @brief Arrange an Incubator book with bounded search and unchanged proof rows.

    @details Apply dependency, private-helper and frontier scheduling together
    and arrangement search once per scope, then order the chapters once
    (`order_main_book`), its search inside each anchor stretch bounded to
    insertion distance 512.
    The final hierarchical pass still contracts complete scopes and operations
    and asserts every proof-before-use obligation. Incubator has no semantic
    shortening passes, so its layout does not require a rewrite fixed point.

    @param theorems Complete Incubator theorem registry.
    @param stacks Complete physical proof chapters.
    @param filename_groups_for Callable computing complete filename bundles.
    @param external_theorems Registered external theorem spellings.
    @return Permuted registry and chapters with identical scoped proof content.
    @invariant Every theorem, goal, namespace, row and citation is retained.
    @see proof_hierarchy.order_main_hierarchies
    """
    groups = filename_groups_for(theorems)
    names = [name for group in groups for name in group]
    assert len(groups) == len(theorems) and len(names) == len(set(names))
    assert set(names) == set(stacks), "incomplete proof bundle coverage"
    result = order_main_scope_rows(
        theorems, stacks, groups, external_theorems,
        group_private=True, group_frontier=True, group_search=True,
    )
    for name, old in stacks.items():
        new = result[name]
        assert old and new and new[0][:2] == old[0][:2], "main goal changed"
        assert Counter(map(tuple, old)) == Counter(map(tuple, new)), "proof rows changed"
        assert [r[1] for r in old] == [r[1] for r in new], "namespace positions changed"
    return order_main_book(theorems, result, groups, search_window=512)


@layout_search_cache()
@main_scope_order_cache()
def process_to_fixed_point(
    theorems, stacks, filename_groups_for, external_theorems, binary_dir
):
    """
    @brief Converge main proof transformations or arrange Incubator once.

    @details Each main-book round repeats the same scope ordering, equality fusion,
    private grouping, equality support, main slicing,
    rewrite placement, frontier search and book ordering. Later layout may
    expose an earlier reusable fact, and a rewrite may leave unused supports.
    Stop only when every theorem, row, namespace and position is unchanged.
    A repeated nonterminal state asserts; there is no iteration-limit fallback.
    Incubator dispatches to one bounded permutation pass; semantic row-reduction
    and exact rewrite convergence remain main-book-only. Other books retain
    their existing behavior.

    @param theorems Complete processed theorem registry.
    @param stacks Mapping of all physical proof filenames to their rows.
    @param filename_groups_for Callable computing complete bundles for a registry.
    @param external_theorems Exact registered external theorem spellings.
    @param binary_dir Exact compiled definition registry directory.
    @return Main registry and chapters at a fixed point, or the bounded
    Incubator permutation retaining its exact proof content.
    @invariant No physical chapter grows within a round, no goal or namespace
    discovery is lost, and each book permutation keeps its complete bundles.
    @see process_proof_graphs.create_processed_proof_graph
    """
    anchors = set()
    for theorem in theorems:
        expression = parse_mpl(theorem[0])
        assert isinstance(expression, Implication) and isinstance(expression.premise, Atom)
        anchors.add(expression.premise.head)
    if anchors & INCUBATOR_ANCHORS:
        assert anchors <= INCUBATOR_ANCHORS, "main and incubator books cannot be mixed"
        return process_incubator_book(theorems, stacks, filename_groups_for, external_theorems)
    state_key = lambda ts, ss: (
        tuple(map(tuple, ts)),
        tuple((name, tuple(map(tuple, ss[name]))) for name in sorted(ss)),
    )
    current = state_key(theorems, stacks)
    seen = {current}
    while True:
        groups = filename_groups_for(theorems)
        names = [name for group in groups for name in group]
        assert len(groups) == len(theorems) and len(names) == len(set(names))
        assert set(names) == set(stacks), "incomplete proof bundle coverage"
        before = stacks
        result = order_main_scope_rows(theorems, stacks, groups, external_theorems)
        result = compact_main_equalities(theorems, result, groups)
        result = order_main_scope_rows(
            theorems, result, groups, external_theorems, group_private=True
        )
        result = shortcut_main_equalities(theorems, result, groups, binary_dir)
        result = slice_main_rows(theorems, result, groups)
        result = place_main_rewrites(theorems, result, groups, external_theorems)
        result = order_main_scope_rows(
            theorems, result, groups, external_theorems, group_frontier=True
        )
        result = order_main_scope_rows(
            theorems, result, groups, external_theorems, group_search=True
        )
        assert (
            result.keys() == before.keys()
        ), "proof filename coverage changed before book permutation"
        for name, old in before.items():
            new = result[name]
            assert (
                old and new and len(new) <= len(old)
            ), "proof row growth or missing chapter"
            assert new[0][:2] == old[0][:2], "main goal changed"
            assert list(dict.fromkeys(r[1] for r in old)) == list(
                dict.fromkeys(r[1] for r in new)
            ), "namespace discovery changed"
        theorems, stacks = order_main_book(theorems, result, groups)
        updated = state_key(theorems, stacks)
        if updated == current:
            return theorems, stacks
        assert updated not in seen, "proof processing cycle"
        seen.add(updated)
        current = updated
