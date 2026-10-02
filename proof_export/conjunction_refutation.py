# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschraenkt).
# Dual-licensed under AGPLv3 and a commercial license.
# See https://generative-logic.com/license. Contributions require CLA.

"""Exact structural replay for scoped, concrete conjunction refutations."""

from .contraposition import complement
from .mpl import Atom, Conjunction, Expression, Implication, Negation, iter_atoms, map_arguments, parse_mpl


def is_conjunction_refutation_expansion(target: Expression, sources: list[Expression], definitions: dict) -> bool:
    """@brief Identify the structural-refutation expansion shape.
    @details Recognition selects a checker; it does not establish validity.
    @param target Claimed fixed-argument implication.
    @param sources Ordered cited OR and optional expanded contrapositive.
    @param definitions Registry definitions.
    @return Whether the row belongs to the structural-refutation checker.
    """
    conclusion = target
    while isinstance(conclusion, Implication):
        conclusion = conclusion.conclusion
    return (isinstance(target, Implication) and isinstance(conclusion, Negation)
            and isinstance(conclusion.inner, Atom)
            and definitions.get(conclusion.inner.head, {}).get('category') == 'and'
            and bool(sources) and (isinstance(sources[0], Implication)
                or (isinstance(sources[0], Atom)
                    and definitions.get(sources[0].head, {}).get('category') == 'or')))



def build_guarded_support_plan(target: Expression, source: Expression, definitions: dict) -> dict:
    """@brief Validate a guarded helper against its owning implication.
    @details Exactly one immediate part is complemented; all other signed parts
    remain in definition order. Only source binders occurring in the conjunction
    enter the helper, bijectively; fixed parameters and repeated ports stay exact.
    Nested conjunction parts are opaque and cannot authorize recursive support.
    @param target Claimed universally quantified supporting implication.
    @param source Existing implication containing the negated conjunction premise.
    @param definitions Matching compiled registry.
    @return Immediate projection path, selected guard and explicit binder metadata.
    @invariant No source premise or helper guard is asserted by this validation.
    """
    assert isinstance(target, Implication) and isinstance(source, Implication)
    target_bound, binder_counts, premises = [], [], []
    head = target
    while isinstance(head, Implication):
        assert not set(target_bound).intersection(head.bound_variables), 'duplicate helper binder'
        target_bound.extend(head.bound_variables)
        binder_counts.append(len(head.bound_variables))
        premises.append(head.premise)
        head = head.conclusion
    introduced = 0
    for premise, count in zip(premises, binder_counts, strict=True):
        introduced += count
        used = {a for atom in iter_atoms(premise) for a in atom.arguments}
        assert not used.intersection(target_bound[introduced:]), 'helper uses a later binder'
    assert len(set(target_bound)) == len(target_bound)
    assert isinstance(head, Negation) and isinstance(head.inner, Atom)
    conjunction = head.inner
    entry = definitions[conjunction.head]
    assert entry['category'] == 'and' and entry['elements']
    signature = entry['signature']
    formal = (parse_mpl(signature).arguments if isinstance(signature, str)
              else tuple(f'u_{i + 1}' for i in range(entry['arity'])))
    assert len(formal) == len(conjunction.arguments)
    mapping = dict(zip(formal, conjunction.arguments, strict=True))
    parts = [map_arguments(parse_mpl(value), lambda argument: mapping.get(argument, argument))
             for value in entry['elements']]
    assert len(premises) == len(parts), 'missing or additional supporting guard'
    selected = [i for i, part in enumerate(parts) if premises[i] != part]
    assert len(selected) == 1 and premises[selected[0]] == complement(parts[selected[0]]), (
        'support must complement exactly one immediate part')
    source_bound = set()
    source_premises = []
    tail = source
    while isinstance(tail, Implication):
        assert not source_bound.intersection(tail.bound_variables), 'duplicate source binder'
        source_bound.update(tail.bound_variables)
        source_premises.append(tail.premise)
        tail = tail.conclusion
    source_fixed = {a for atom in iter_atoms(source) for a in atom.arguments} - source_bound
    assert not source_fixed.intersection(target_bound), 'helper captures a source parameter'
    matched = False
    for premise in source_premises:
        if not isinstance(premise, Negation) or not isinstance(premise.inner, Atom):
            continue
        original = premise.inner
        if original.head != conjunction.head or len(original.arguments) != len(conjunction.arguments):
            continue
        binding, inverse = {}, {}
        compatible = True
        for before, after in zip(original.arguments, conjunction.arguments, strict=True):
            if before in source_bound:
                if (after not in target_bound or binding.setdefault(before, after) != after
                        or inverse.setdefault(after, before) != before):
                    compatible = False
                    break
            elif before != after or after in target_bound:
                compatible = False
                break
        if compatible and set(inverse) == set(target_bound):
            matched = True
            break
    assert matched, 'source does not supply the exact negated conjunction premise'
    covered = {a for p in premises for atom in iter_atoms(p) for a in atom.arguments}
    assert set(target_bound) <= covered, 'unbound supporting conclusion argument'
    index = selected[0]
    count = len(parts)
    path = ([1] * (count - 1) if index == 0 else [1] * (count - index - 1) + [2])
    return {'path': path, 'unfolding': [conjunction.head],
            'negative_member': isinstance(parts[index], Negation), 'guarded': True,
            'bound_variables': list(target_bound), 'binder_counts': binder_counts,
            'premise_count': count, 'selected': index}


def build_conjunction_refutation_plan(target: Expression, sources: list[Expression], definitions: dict) -> dict:
    """@brief Validate concrete refutation and retain its exact projection path.
    @details The cited OR exposes a positive conjunction directly or fixes every
    binder of the cited unary contrapositive through an alternative complement.
    Only positive conjunction definitions unfold; signs and fixed arguments are
    exact. The result is a structural implication, never a positive conjunction.
    @param target Claimed implication with no local binders.
    @param sources Exact OR and optional expanded contrapositive citations.
    @param definitions Raw or typed registry definitions.
    @return Explicit conjunction unfoldings and binary member projection path.
    @invariant Incomplete bindings, extra sources and mismatching members assert.
    """
    assert is_conjunction_refutation_expansion(target, sources, definitions)
    if isinstance(sources[0], Implication):
        assert len(sources) == 1, 'supporting expansion has exactly one owning implication'
        return build_guarded_support_plan(target, sources[0], definitions)
    assert isinstance(target.conclusion, Negation)
    assert not target.bound_variables and len(sources) in (1, 2)
    conjunction = target.conclusion.inner

    def children(atom: Atom, category: str) -> list[Expression]:
        """@brief Substitute one exact compiled connective instance.
        @details Both registry signature representations preserve argument order.
        @param atom Concrete compact.
        @param category Required connective category.
        @return Ordered substituted children.
        """
        entry = definitions[atom.head]
        assert entry['category'] == category
        signature = entry['signature']
        if isinstance(signature, str):
            formal = parse_mpl(signature)
            assert isinstance(formal, Atom)
            parameters = formal.arguments
        else:
            assert isinstance(signature, list) and len(signature) == entry['arity']
            parameters = tuple(f'u_{index + 1}' for index in range(entry['arity']))
        assert len(parameters) == len(atom.arguments)
        substitution = dict(zip(parameters, atom.arguments, strict=True))
        return [map_arguments(parse_mpl(element), lambda value: substitution.get(value, value))
                for element in entry['elements']]

    alternatives = []
    pending = [(sources[0], ())]
    while pending:
        current, ancestry = pending.pop()
        if isinstance(current, Atom) and definitions.get(current.head, {}).get('category') == 'or':
            assert current.head not in ancestry, 'cyclic OR definition'
            pending.extend((child, (*ancestry, current.head))
                           for child in reversed(children(current, 'or')))
        else:
            alternatives.append(current)
    if len(sources) == 1:
        assert conjunction in alternatives, 'conjunction is not an encountered OR alternative'
    else:
        rule = sources[1]
        assert isinstance(rule, Implication) and not isinstance(rule.conclusion, Implication)
        assert isinstance(rule.premise, Negation) and isinstance(rule.premise.inner, Atom)
        assert definitions[rule.premise.inner.head]['category'] == 'and'
        bound = set(rule.bound_variables)
        assert len(bound) == len(rule.bound_variables)
        matched = False
        for alternative in alternatives:
            expected = complement(alternative)
            head = rule.conclusion
            if isinstance(head, Negation) != isinstance(expected, Negation):
                continue
            if isinstance(head, Negation):
                head, expected = head.inner, expected.inner
            if not isinstance(head, Atom) or not isinstance(expected, Atom):
                continue
            if head.head != expected.head or len(head.arguments) != len(expected.arguments):
                continue
            substitution = {}
            compatible = True
            for formal, actual in zip(head.arguments, expected.arguments, strict=True):
                if formal in bound:
                    if substitution.setdefault(formal, actual) != actual:
                        compatible = False
                elif formal != actual:
                    compatible = False
            if not compatible or set(substitution) != bound:
                continue
            instance = map_arguments(rule.premise.inner, lambda value: substitution.get(value, value))
            if instance == conjunction:
                matched = True
                break
        assert matched, 'contrapositive does not determine this complete conjunction'

    member = complement(target.premise)
    pending = [(conjunction, [], (), [])]
    while pending:
        current, path, ancestry, unfolding = pending.pop()
        if unfolding and current == member:
            return {'path': path, 'unfolding': unfolding,
                    'negative_member': isinstance(member, Negation)}
        if isinstance(current, Atom) and definitions.get(current.head, {}).get('category') == 'and':
            assert current.head not in ancestry, 'cyclic conjunction definition'
            parts = children(current, 'and')
            assert parts
            tree = parts[0]
            for part in parts[1:]:
                tree = Conjunction((tree, part))
            pending.append((tree, path, (*ancestry, current.head), [*unfolding, current.head]))
        elif isinstance(current, Conjunction):
            assert len(current.parts) == 2
            pending.append((current.parts[1], [*path, 2], ancestry, unfolding))
            pending.append((current.parts[0], [*path, 1], ancestry, unfolding))
    raise AssertionError('refuted member is not in the exact conjunction tree')
