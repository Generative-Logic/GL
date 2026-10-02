# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Explicit substitutions for a producer chain composed with a unary consumer."""

from __future__ import annotations

from .mpl import Atom, Expression, Implication, Negation, iter_atoms, map_arguments, to_mpl


def validate_composition_scopes(conclusion: str, first: str, second: str) -> None:
    """@brief Check the exact validity join of two source implications.
    @details Sources must be comparable; the conclusion uses their deeper scope.
    @param conclusion Composed rule's namespace.
    @param first Producer namespace.
    @param second Consumer namespace.
    @return None.
    """
    assert first == second or first.startswith(second + '_boundary_') or second.startswith(first + '_boundary_'), (
        'composition sources have incomparable scopes')
    assert conclusion == max((first, second), key=len), 'composition has the wrong joined scope'


def build_implication_composition_plan(
    conclusion: Expression, first: Expression, second: Expression,
) -> dict[str, object]:
    """@brief Check a composed implication against its two exact source rules.

    @details Source binders have distinct namespaces and must match bijectively
    through the intermediate ports. No binder becomes a fixed parameter or
    collapses with another binder; the conclusion retains the same binder set.
    Conclusion arguments and free source parameters are rigid. Endpoint ports
    constrain the common renaming; contradictory constraints assert.
    No premise is proved backwards and no uncited rule is consulted.
    @param conclusion Claimed implication retaining every ordered producer premise.
    @param first Producer implication.
    @param second Consumer implication.
    @return Explicit source instantiations and their common intermediate formula.
    @invariant Both complete source binders map bijectively onto the conclusion binder.
    """
    assert isinstance(conclusion, Implication)
    assert isinstance(first, Implication) and isinstance(second, Implication)
    rules = (first, second, conclusion)
    premise_runs, heads, binder_groups, bound_runs = [], [], [], []
    for rule in rules:
        cursor, premises, groups, bound = rule, [], [], []
        while isinstance(cursor, Implication):
            group = list(cursor.bound_variables)
            assert len(group) == len(set(group)) and not set(group).intersection(bound), 'composition binder capture'
            groups.append(group)
            bound.extend(group)
            premises.append(cursor.premise)
            cursor = cursor.conclusion
        introduced = set()
        for group, premise in zip(groups, premises, strict=True):
            introduced.update(group)
            used = {argument for item in iter_atoms(premise) for argument in item.arguments}
            assert used.intersection(bound) <= introduced, 'composition variable used outside binder scope'
        premise_runs.append(premises)
        heads.append(cursor)
        binder_groups.append(groups)
        bound_runs.append(bound)
    assert len(premise_runs[1]) == 1, 'composition consumer must be unary'
    assert len(premise_runs[0]) == len(premise_runs[2]), 'composition changes producer premise count'
    source_fixed = [{argument for item in iter_atoms(rule) for argument in item.arguments} - set(bound_runs[i])
                    for i, rule in enumerate(rules[:2])]
    assert not set(bound_runs[2]).intersection(source_fixed[0] | source_fixed[1]), 'composition captures fixed parameters'
    parents: dict[tuple[int, str], tuple[int, str]] = {}

    def atom(expression: Expression) -> tuple[bool, Atom]:
        """@brief Read a signed flat atom.
        @details Each chain member is a signed flat atom; nested implications form the chain.
        @param expression The signed atom.
        @return Sign and atom payload.
        """
        negative = isinstance(expression, Negation)
        payload = expression.inner if negative else expression
        assert isinstance(payload, Atom), 'composition requires flat signed chain members'
        return negative, payload

    def term(side: int, name: str) -> tuple[int, str]:
        """@brief Separate bound source variables from rigid target arguments.
        @details Free parameters retain exact spelling across both source rules.
        @param side Rule position: producer, consumer or conclusion.
        @param name Argument token.
        @return Unification identity.
        """
        key = (side, name) if side < 2 and name in bound_runs[side] else (-1, name)
        parents.setdefault(key, key)
        return key

    def root(key: tuple[int, str]) -> tuple[int, str]:
        """@brief Find the representative of a unification class.
        @details The finite parent forest has no cycles.
        @param key Existing argument identity.
        @return Its current representative.
        """
        while parents[key] != key:
            key = parents[key]
        return key

    def unify(left: Expression, left_side: int, right: Expression, right_side: int) -> None:
        """@brief Unify corresponding atom ports while retaining rigid parameters.
        @details Two distinct rigid arguments or signed operators cannot unify.
        @param left Left expression.
        @param left_side Left binder namespace.
        @param right Right expression.
        @param right_side Right binder namespace.
        @return None.
        """
        left_sign, left_atom = atom(left)
        right_sign, right_atom = atom(right)
        assert (left_sign, left_atom.head, len(left_atom.arguments)) == (
            right_sign, right_atom.head, len(right_atom.arguments)), 'composition atom mismatch'
        for a, b in zip(left_atom.arguments, right_atom.arguments):
            a, b = root(term(left_side, a)), root(term(right_side, b))
            if a == b:
                continue
            assert not (a[0] == b[0] == -1), 'composition changes a rigid argument'
            if a[0] != -1 and (b[0] == -1 or b < a):
                a, b = b, a
            parents[b] = a

    _, producer_head = atom(heads[0])
    _, consumer_premise = atom(premise_runs[1][0])
    first_bound, second_bound = set(bound_runs[0]), set(bound_runs[1])
    assert len(first_bound) == len(second_bound) == len(bound_runs[2]), (
        'composition requires the same complete bound-variable set')
    forward, backward = {}, {}
    assert len(producer_head.arguments) == len(consumer_premise.arguments), 'composition atom mismatch'
    for left, right in zip(producer_head.arguments, consumer_premise.arguments):
        assert (left in first_bound) == (right in second_bound), (
            'composition cannot match a bound variable to a fixed parameter')
        if left not in first_bound:
            assert left == right, 'composition changes a fixed parameter'
            continue
        assert forward.setdefault(left, right) == right and backward.setdefault(right, left) == left, (
            'composition cannot collapse distinct bound variables')
    assert set(forward) == first_bound and set(backward) == second_bound, (
        'composition intermediate does not match the complete binder sets')

    for original, result in zip(premise_runs[0], premise_runs[2], strict=True):
        unify(original, 0, result, 2)
    unify(heads[1], 1, heads[2], 2)
    unify(heads[0], 0, premise_runs[1][0], 1)
    substitutions = []
    for side, rule in enumerate((first, second)):
        mapping = {}
        for variable in bound_runs[side]:
            value = root(term(side, variable))
            assert value[0] == -1, 'composition binder has no endpoint-supported argument'
            mapping[variable] = value[1]
        assert set(mapping.values()) == set(bound_runs[2]), (
            'composition does not preserve the conclusion binder set')
        substitutions.append(mapping)
    intermediate = map_arguments(heads[0], lambda name: substitutions[0].get(name, name))
    assert intermediate == map_arguments(premise_runs[1][0], lambda name: substitutions[1].get(name, name))
    return {'source_arguments': [[substitutions[i][v] for v in bound_runs[i]]
                                 for i in range(2)],
            'source_binder_groups': [[[substitutions[i][v] for v in group] for group in binder_groups[i]]
                                     for i in range(2)],
            'result_binder_groups': binder_groups[2],
            'intermediate': to_mpl(intermediate)}
