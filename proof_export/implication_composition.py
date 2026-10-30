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

    The conclusion may list the producer premises in another order: the
    compiler identifies rules up to the order of their premises, so a
    composed rule can be stored with a permutation of the producer chain.
    Each producer premise is paired with one conclusion premise of the same
    sign, operator and arity; the pairings are tried in lexicographic order
    (the identity first) and the first one under which all ports unify is
    returned as `premise_order`. Every conclusion premise is still checked
    against its own binder scope, so a permutation never uses a variable
    before its binder.
    @param conclusion Claimed implication retaining every producer premise.
    @param first Producer implication.
    @param second Consumer implication.
    @return Explicit source instantiations, their common intermediate formula
            and `premise_order`: for each producer premise, the index of the
            conclusion premise it is paired with.
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

    def signature(expression: Expression) -> tuple[bool, str, int]:
        """@brief Sign, operator and arity of a chain member.
        @details Only premises with equal signatures can be paired.
        @param expression The signed atom.
        @return Its signature.
        """
        negative, payload = atom(expression)
        return negative, payload.head, len(payload.arguments)

    producer_signatures = [signature(premise) for premise in premise_runs[0]]
    conclusion_signatures = [signature(premise) for premise in premise_runs[2]]

    def solve(order: tuple[int, ...]) -> tuple[list[dict[str, str]] | None, str]:
        """@brief Unify both sources with the conclusion under one premise pairing.
        @details Builds a fresh unification forest: producer premise `i` against
        conclusion premise `order[i]`, the consumer head against the conclusion
        head, the producer head against the consumer premise. Bound source
        variables unify freely; two distinct rigid arguments do not.
        @param order For each producer premise, the paired conclusion premise.
        @return The two source substitutions and an empty message, or no
                substitutions and the reason the pairing fails.
        """
        parents: dict[tuple[int, str], tuple[int, str]] = {}

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

        def unify(left: Expression, left_side: int, right: Expression, right_side: int) -> str:
            """@brief Unify corresponding atom ports while retaining rigid parameters.
            @details Two distinct rigid arguments or signed operators cannot unify.
            @param left Left expression.
            @param left_side Left binder namespace.
            @param right Right expression.
            @param right_side Right binder namespace.
            @return An empty string, or the reason the atoms do not unify.
            """
            left_sign, left_atom = atom(left)
            right_sign, right_atom = atom(right)
            if (left_sign, left_atom.head, len(left_atom.arguments)) != (
                    right_sign, right_atom.head, len(right_atom.arguments)):
                return 'composition atom mismatch'
            for a, b in zip(left_atom.arguments, right_atom.arguments):
                a, b = root(term(left_side, a)), root(term(right_side, b))
                if a == b:
                    continue
                if a[0] == b[0] == -1:
                    return 'composition changes a rigid argument'
                if a[0] != -1 and (b[0] == -1 or b < a):
                    a, b = b, a
                parents[b] = a
            return ''

        steps = [(original, 0, premise_runs[2][order[index]], 2)
                 for index, original in enumerate(premise_runs[0])]
        steps += [(heads[1], 1, heads[2], 2), (heads[0], 0, premise_runs[1][0], 1)]
        for step in steps:
            failure = unify(*step)
            if failure:
                return None, failure
        substitutions = []
        for side in range(2):
            mapping = {}
            for variable in bound_runs[side]:
                value = root(term(side, variable))
                if value[0] != -1:
                    return None, 'composition binder has no endpoint-supported argument'
                mapping[variable] = value[1]
            if set(mapping.values()) != set(bound_runs[2]):
                return None, 'composition does not preserve the conclusion binder set'
            substitutions.append(mapping)
        if map_arguments(heads[0], lambda name: substitutions[0].get(name, name)) != map_arguments(
                premise_runs[1][0], lambda name: substitutions[1].get(name, name)):
            return None, 'composition intermediate mismatch'
        return substitutions, ''

    def pairings(index: int, used: list[bool], chosen: list[int]):
        """@brief Enumerate signature-compatible premise pairings.
        @details Depth first over the producer premises, each taking an unused
        conclusion premise of the same signature in increasing index order, so
        the pairings come in lexicographic order and the identity first.
        @param index Next producer premise.
        @param used Conclusion premises already paired.
        @param chosen Pairing built so far.
        @return Generator of complete pairings.
        """
        if index == len(producer_signatures):
            yield tuple(chosen)
            return
        for candidate, taken in enumerate(used):
            if taken or conclusion_signatures[candidate] != producer_signatures[index]:
                continue
            used[candidate] = True
            chosen.append(candidate)
            yield from pairings(index + 1, used, chosen)
            chosen.pop()
            used[candidate] = False

    first_failure = ''
    for order in pairings(0, [False] * len(conclusion_signatures), []):
        substitutions, failure = solve(order)
        if substitutions is None:
            first_failure = first_failure or failure
            continue
        intermediate = map_arguments(heads[0], lambda name: substitutions[0].get(name, name))
        return {'source_arguments': [[substitutions[i][v] for v in bound_runs[i]]
                                     for i in range(2)],
                'source_binder_groups': [[[substitutions[i][v] for v in group] for group in binder_groups[i]]
                                         for i in range(2)],
                'result_binder_groups': binder_groups[2],
                'intermediate': to_mpl(intermediate),
                'premise_order': list(order)}
    raise AssertionError(first_failure or 'composition atom mismatch')
