# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschraenkt).
# Dual-licensed under AGPLv3 and a commercial license.
# See https://generative-logic.com/license. Contributions require CLA.

"""Checked source substitutions for grouping and contraposition."""

from .mpl import Atom, Conjunction, Expression, Implication, Negation, iter_atoms, map_arguments, parse_mpl


def complement(expression: Expression) -> Expression:
    """@brief Complement one formula, cancelling double negation.
    @details Polarity changes only at the outer connective.
    @param expression Formula to complement.
    @return Complemented formula.
    """
    return expression.inner if isinstance(expression, Negation) else Negation(expression)


def validate_auxiliary_scopes(target: str, sources: list[str]) -> None:
    """@brief Check the exact common validity of cited auxiliary sources.
    @details The deepest source must be the target and every other source an ancestor.
    @param target Claimed result scope.
    @param sources Nonempty source scopes.
    @return None.
    """
    assert sources and target == max(sources, key=len), 'incorrect auxiliary result scope'
    assert all(target == scope or target.startswith(scope + '_boundary_') for scope in sources), (
        'auxiliary sources have incomparable scopes')


def _alignment(source: Implication, target: Implication,
               left: Expression, right: Expression) -> dict[str, str]:
    """@brief Align complete binder sets through corresponding signed atoms.
    @details Fixed parameters and repeated ports are rigid; the binder map is bijective.
    @param source Original implication.
    @param target Claimed derived implication.
    @param left Original atom fixing source binders.
    @param right Corresponding target atom.
    @return Complete source-to-target binder substitution.
    """
    assert isinstance(source, Implication) and isinstance(target, Implication)
    source_bound, target_bound = set(source.bound_variables), set(target.bound_variables)
    source_fixed = {argument for atom in iter_atoms(source) for argument in atom.arguments} - source_bound
    assert not source_fixed.intersection(target_bound), 'target binder captures a fixed source parameter'
    assert len(source_bound) == len(source.bound_variables)
    assert len(target_bound) == len(target.bound_variables) == len(source_bound)
    assert isinstance(left, Negation) == isinstance(right, Negation)
    if isinstance(left, Negation):
        left, right = left.inner, right.inner
    assert isinstance(left, Atom) and isinstance(right, Atom), 'expected a single flat premise'
    assert left.head == right.head and len(left.arguments) == len(right.arguments)
    mapping, inverse = {}, {}
    for a, b in zip(left.arguments, right.arguments, strict=True):
        assert (a in source_bound) == (b in target_bound), 'bound/fixed mismatch'
        if a in source_bound:
            assert mapping.setdefault(a, b) == b and inverse.setdefault(b, a) == a, 'binder collapse'
        else:
            assert a == b, 'fixed parameter changed'
    assert set(mapping) == source_bound and set(inverse) == target_bound, 'incomplete binder coverage'
    return mapping


def build_contraposition_plan(target: Expression, source: Expression) -> dict:
    """@brief Check exact contraposition of a unary implication.
    @details Both ends cover every binder and the two polarities reverse exactly.
    @param target Claimed contrapositive.
    @param source Sole cited original rule.
    @return Explicit source arguments for proof replay.
    """
    assert isinstance(target, Implication) and isinstance(source, Implication)
    mapping = _alignment(source, target, source.premise, complement(target.conclusion))
    head_variables = {arg for atom in iter_atoms(source.conclusion) for arg in atom.arguments}
    assert set(source.bound_variables) <= head_variables, 'head omits a bound variable'
    assert not isinstance(source.conclusion, Implication), 'multiple-premise source'
    rewritten_head = map_arguments(source.conclusion, lambda value: mapping.get(value, value))
    assert complement(rewritten_head) == target.premise, 'incorrect contrapositive premise'
    return {'source_arguments': [[mapping[value] for value in source.bound_variables]]}


def build_grouping_plan(target: Expression, sources: list[Expression],
                        definitions: dict) -> dict:
    """@brief Check a grouped implication against all and only its cited sources.
    @details Source binders align through the complete common premise. The exact
    conjunction tree is instantiated from checked registry definitions and each
    contributing source head is used exactly once.
    @param target Claimed grouped implication.
    @param sources Ordered contributing unary implications.
    @param definitions Compiled definitions used to unfold conjunction heads.
    @return Source substitutions and an explicit conjunction-construction tree.
    """
    assert isinstance(target, Implication) and len(sources) >= 2
    heads, arguments = [], []
    for source in sources:
        assert isinstance(source, Implication) and not isinstance(source.conclusion, Implication)
        mapping = _alignment(source, target, source.premise, target.premise)
        heads.append(map_arguments(source.conclusion, lambda value: mapping.get(value, value)))
        arguments.append([mapping[value] for value in source.bound_variables])
    used = set()

    def tree(expression: Expression, active: tuple[str, ...] = ()) -> int | list:
        """@brief Resolve a conjunction tree to the exact supporting source heads.
        @details Only conjunction definitions unfold; recursive definitions assert.
        @param expression Current target head or conjunction node.
        @param active Current compact-definition ancestry.
        @return Source index or binary construction tree.
        """
        for index, head in enumerate(heads):
            if index not in used and expression == head:
                used.add(index)
                return index
        if isinstance(expression, Atom):
            assert expression.head not in active, 'cyclic conjunction definition'
            assert expression.head in definitions, 'unjustified grouped head'
            entry = definitions[expression.head]
            assert entry['category'] == 'and'
            signature = entry['signature']
            if isinstance(signature, str):
                formal = parse_mpl(signature)
                assert isinstance(formal, Atom)
                parameters = formal.arguments
            else:
                assert isinstance(signature, list) and len(signature) == entry['arity']
                parameters = tuple(f'u_{index + 1}' for index in range(entry['arity']))
            assert len(parameters) == len(expression.arguments)
            mapping = dict(zip(parameters, expression.arguments, strict=True))
            parts = [map_arguments(parse_mpl(value), lambda name: mapping[name]) for value in entry['elements']]
            assert len(parts) >= 2
            active = (*active, expression.head)
            expression = Conjunction(tuple(parts))
        assert isinstance(expression, Conjunction) and len(expression.parts) >= 2, 'unjustified grouped head'
        # GL expands a multi-element compact into a left-associated conjunction.
        result = tree(expression.parts[0], active)
        for part in expression.parts[1:]:
            result = [result, tree(part, active)]
        return result

    construction = tree(target.conclusion)
    assert used == set(range(len(sources))), 'unused or missing grouping source'
    covered = {arg for head in heads for atom in iter_atoms(head) for arg in atom.arguments}
    assert set(target.bound_variables) <= covered, 'group omits a bound variable'
    return {'source_arguments': arguments, 'construction': construction}
