# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Pure structural implication matching shared by processing and captions."""
from .mpl import Atom, Conjunction, Implication, Negation


def _binders_of(expression: object):
    """
    @brief Iterate the binder lists of every implication in an expression.

    @details Visit nested premises, conclusions, conjunctions and negations.

    @param expression Parsed MPL expression.
    @return Iterator over binder tuples.
    """
    if isinstance(expression, Implication):
        yield expression.bound_variables
        yield from _binders_of(expression.premise)
        yield from _binders_of(expression.conclusion)
    elif isinstance(expression, Conjunction):
        for part in expression.parts:
            yield from _binders_of(part)
    elif isinstance(expression, Negation):
        yield from _binders_of(expression.inner)


def _unify_token(
    rule: str, actual: str, bound: set[str], sigma: dict[str, str]
) -> bool:
    """
    @brief Match one token: bound variables map consistently, others are equal.

    @details An existing assignment is checked; a new bound assignment is recorded.

    @param rule Rule token.
    @param actual Actual token.
    @param bound Variables bound in the rule.
    @param sigma Current variable mapping (extended on success).
    @return True when the tokens match.
    """
    if rule in bound:
        if rule in sigma:
            return sigma[rule] == actual
        sigma[rule] = actual
        return True
    return rule == actual


def _unify(
    rule: object, actual: object, bound: set[str], sigma: dict[str, str]
) -> bool:
    """
    @brief Structural match of a rule node against an actual node.

    @details Bound rule variables map consistently through `sigma` (updated
    in place); other tokens must be equal; heads, arities and connectives
    must agree.

    @param rule Rule node.
    @param actual Actual node.
    @param bound Variables bound in the rule.
    @param sigma Current variable mapping (extended on success).
    @return True when the nodes match under the extended mapping.
    """
    if isinstance(rule, Atom):
        return (
            isinstance(actual, Atom)
            and rule.head == actual.head
            and len(rule.arguments) == len(actual.arguments)
            and all(
                _unify_token(r, a, bound, sigma)
                for r, a in zip(rule.arguments, actual.arguments)
            )
        )
    if isinstance(rule, Negation):
        return isinstance(actual, Negation) and _unify(
            rule.inner, actual.inner, bound, sigma
        )
    if isinstance(rule, Conjunction):
        return (
            isinstance(actual, Conjunction)
            and len(rule.parts) == len(actual.parts)
            and all(
                _unify(r, a, bound, sigma) for r, a in zip(rule.parts, actual.parts)
            )
        )
    assert isinstance(rule, Implication), f"unknown MPL node {rule!r}"
    return (
        isinstance(actual, Implication)
        and len(rule.bound_variables) == len(actual.bound_variables)
        and all(
            _unify_token(r, a, bound, sigma)
            for r, a in zip(rule.bound_variables, actual.bound_variables)
        )
        and _unify(rule.premise, actual.premise, bound, sigma)
        and _unify(rule.conclusion, actual.conclusion, bound, sigma)
    )


def _rule_binding(
    rule: object, premises: list[object], derived: object
) -> tuple[list[int], dict[str, str]] | None:
    """
    @brief Align a row with the rule it instantiates and return the binding.

    @details The search of `_match_rule`; on success also the mapping of every
    rule-bound name to the token it takes in the row.

    @param rule The cited implication (parsed).
    @param premises The row's premises (parsed), in row order.
    @param derived The row's derived expression (parsed).
    @return `(order, binding)`, or None when no consistent alignment exists.
    """
    found: list[dict[str, str]] = []
    order = _match_rule(rule, premises, derived, found)
    return None if order is None else (order, found[0])


def _match_rule(
    rule: object,
    premises: list[object],
    derived: object,
    binding: list[dict[str, str]] | None = None,
) -> list[int] | None:
    """
    @brief Align a row's premises with the premises of the rule they instantiate.

    @details Mirrors the verifier's implication check: the rule's premises
    (anchor included) and head match the row's premises (in some order) and
    derived expression structurally, every variable bound anywhere in the
    rule mapping consistently to one token and every other token matching
    literally. Backtracking search, first match wins.

    @param rule The cited implication (parsed).
    @param premises The row's premises (parsed), in row order.
    @param derived The row's derived expression (parsed).
    @param binding When given, receives the final name mapping on success.
    @return For each rule premise, the index of its row premise; None when no
    consistent alignment exists.
    """
    bound: set[str] = set()
    for found in _binders_of(rule):
        bound.update(found)
    rule_premises: list[object] = []
    node = rule
    while isinstance(node, Implication):
        rule_premises.append(node.premise)
        node = node.conclusion
    if len(rule_premises) != len(premises):
        return None
    start: dict[str, str] = {}
    if not _unify(node, derived, bound, start):
        return None
    order: list[int] = []
    used: set[int] = set()

    def search(position: int, sigma: dict[str, str]) -> bool:
        """
        @brief Assign row premises to the rule premises from `position` on.

        @details Backtrack through unused premises in their recorded order.

        @param position Index of the next rule premise.
        @param sigma Variable mapping so far.
        @return True when the remaining premises align.
        """
        if position == len(rule_premises):
            if binding is not None:
                binding.append(sigma)
            return True
        for index, actual in enumerate(premises):
            if index in used:
                continue
            attempt = dict(sigma)
            if _unify(rule_premises[position], actual, bound, attempt):
                used.add(index)
                order.append(index)
                if search(position + 1, attempt):
                    return True
                used.discard(index)
                order.pop()
        return False

    return order if search(0, start) else None
