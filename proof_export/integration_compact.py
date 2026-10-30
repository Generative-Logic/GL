# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Compact-first integration rows (D-503).

An integration rule's history row states its compact implication: the compact
takes the tag and the citation the rule took, and the rule follows it as an
``expansion`` row citing the compact. An OR introduction that serves such a
rule states its own compact and cites the rule's compact. The certificate
derives the rule a compact stands for here and proves the compact by folding
that rule into it.
"""

from __future__ import annotations

from .graph import Chapter, ProofRow
from .mpl import Atom, Expression, Implication, iter_atoms, map_arguments, parse_mpl
from .or_introduction import match_or_introduction

INTEGRATION_RULE_TAGS = frozenset({
    "reformulation for integration >[bound]",
    "reformulation for integration >[]",
    "reformulation for integration and",
    "reformulation for integration flattened",
})


def is_implication_compact(expression: Expression, definitions: dict[str, dict[str, object]]) -> bool:
    """
    @brief Whether an expression is a compact implication of the registry.
    @details A compact is a positive atom whose head names an
    ``implication``-category entry.
    @param expression Any expression.
    @param definitions The certificate's compiled definition entries.
    @return True for a compact implication.
    """

    return (
        isinstance(expression, Atom)
        and definitions.get(expression.head, {}).get("category") == "implication"
    )


def expand_integration_compact(
    compact: Atom,
    definitions: dict[str, dict[str, object]],
    prefix: str,
) -> Expression:
    """
    @brief The rule a compact implication stands for.
    @details
    The entry's signature tokens take the compact's arguments positionally;
    every other argument of its elements is a local of the rule and takes the
    fresh name ``<prefix><k>`` in first-occurrence order, bound at the first
    premise that uses it, as the prover's full-bind reconstruction binds it.
    The last element is the head and introduces no local.
    @param compact A compact implication (asserted).
    @param definitions The certificate's compiled definition entries.
    @param prefix Name prefix for the rule's locals, unique in the chapter.
    @return The rule as a nested implication chain.
    """

    assert is_implication_compact(compact, definitions), f"{compact.head!r} is not a compact implication"
    entry = definitions[compact.head]
    signature = parse_mpl(str(entry["signature"]))
    assert isinstance(signature, Atom) and len(signature.arguments) == len(compact.arguments)
    parameters = dict(zip(signature.arguments, compact.arguments))
    elements = [parse_mpl(str(element)) for element in entry["elements"]]
    assert len(elements) >= 2, f"compact implication {compact.head} has no premise"
    locals_: dict[str, str] = {}
    introduced: list[list[str]] = []
    for element in elements:
        here: list[str] = []
        for atom in iter_atoms(element):
            for argument in atom.arguments:
                if argument in parameters or argument in locals_:
                    continue
                name = f"{prefix}{len(locals_) + 1}"
                assert name not in compact.arguments
                locals_[argument] = name
                here.append(name)
        introduced.append(here)
    assert not introduced[-1], f"compact implication {compact.head} binds a local in its head"
    instantiated = [
        map_arguments(element, lambda name: locals_.get(name, parameters.get(name, name)))
        for element in elements
    ]
    body = instantiated[-1]
    for index in reversed(range(len(instantiated) - 1)):
        body = Implication(tuple(introduced[index]), instantiated[index], body)
    return body


def is_introduction_compact_row(row: ProofRow, definitions: dict[str, dict[str, object]]) -> bool:
    """
    @brief Whether a row states an OR introduction's compact.
    @details The one ``expansion`` row whose conclusion and single citation
    are both compact implications: the introduction's compact citing the
    compact of the rule it serves.
    @param row Processed proof row.
    @param definitions The certificate's compiled definition entries.
    @return True for an introduction compact row.
    """

    return (
        row.tag == "expansion"
        and len(row.dependencies) == 1
        and is_implication_compact(row.conclusion.expression, definitions)
        and is_implication_compact(row.dependencies[0].expression, definitions)
    )


def compact_rule_rows(
    chapter: Chapter,
    definitions: dict[str, dict[str, object]],
) -> dict[int, dict[str, object]]:
    """
    @brief The rules behind a chapter's compact-first integration rows.
    @details
    An integration-tag row stating a compact implication maps to the rule the
    compact stands for; an introduction compact row maps to its introduction
    rule and the construction matched against the cited rule compact (asserted
    to match). Each row's locals are named ``cr<line>_<k>``, so the names are
    unique in the chapter.
    @param chapter Processed proof chapter.
    @param definitions The certificate's compiled definition entries.
    @return Source line -> ``{"rule": Expression}`` or ``{"rule": Expression,
        "introduction": dict}``.
    """

    rules: dict[int, dict[str, object]] = {}
    for row in chapter.rows:
        compact = row.conclusion.expression
        if row.tag in INTEGRATION_RULE_TAGS and is_implication_compact(compact, definitions):
            assert isinstance(compact, Atom)
            rules[row.source_line] = {
                "rule": expand_integration_compact(compact, definitions, f"cr{row.source_line}_"),
            }
        elif is_introduction_compact_row(row, definitions):
            assert isinstance(compact, Atom)
            rule = expand_integration_compact(compact, definitions, f"cr{row.source_line}_")
            introduction = match_or_introduction(rule, row.dependencies[0].expression, definitions)
            assert introduction is not None, (
                f"introduction compact {compact.head} in {chapter.path.name}:{row.source_line} "
                "is no OR introduction of its cited rule compact"
            )
            assert row.dependencies[0].namespace == row.namespace
            rules[row.source_line] = {"rule": rule, "introduction": introduction}
    return rules
