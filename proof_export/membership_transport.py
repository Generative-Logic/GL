# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Checked positive conjunction transport for constructive output witnesses."""


def membership_transport_plan(source: tuple, target: tuple) -> dict | None:
    """
    @brief Recognize exact membership trees modulo positive conjunction swaps.
    @details
    Identical trees need no transport. Positive binary conjunctions may exchange
    their two children, and ordered OR cases may transport their own bodies.
    Every leaf, sign, argument, repetition and binder remains exact. Negation
    and implication are opaque; no reassociation or OR permutation is allowed.
    The returned tree records every projection and injection for Lean replay.
    @param source Expanded structural key of the available membership body.
    @param target Expanded structural key required by the chosen witness body.
    @return A deterministic transport plan, or None for a nonmatching pair.
    @invariant Each accepted child is used exactly once at its original polarity.
    @see certificate._membership_body_key
    """

    if source == target:
        return {"kind": "identity"}
    if source[0] not in {"and", "or"} or source[0] != target[0]:
        return None
    assert len(source) == len(target) == 3
    orders = ((0, 1), (1, 0)) if source[0] == "and" else ((0, 1),)
    for order in orders:
        children = [membership_transport_plan(source[index + 1], target[position + 1])
                    for position, index in enumerate(order)]
        if all(child is not None for child in children):
            return {"kind": source[0], "sources": list(order), "children": children}
    return None
