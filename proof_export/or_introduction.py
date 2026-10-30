# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Exact structural OR introduction from an ordered conjunction alternative."""

from .mpl import Atom, Expression, Implication, map_arguments, parse_mpl


def _source_premises(source: Expression, definitions: dict, children) -> tuple[list[Expression], set[str]] | None:
    """@brief The premises a cited rule offers and its rigid parameters.
    @details A compact implication offers its compiled premises; its arguments
    are rigid. A full implication chain (the flat rule of a root integration
    goal, D-492) offers the premise of every
    level; its free names are rigid and its bound names are local.
    @param source Cited rule.
    @param definitions Raw or typed compiled registry.
    @param children Instantiates one compiled definition.
    @return Ordered premises and rigid names, or None for another source shape.
    """
    if isinstance(source, Atom):
        if definitions.get(source.head, {}).get('category') != 'implication':
            return None
        return children(source)[:-1], set(source.arguments)
    if not isinstance(source, Implication):
        return None
    from .mpl import iter_atoms
    premises, bound, cursor = [], set(), source
    while isinstance(cursor, Implication):
        bound.update(cursor.bound_variables)
        premises.append(cursor.premise)
        cursor = cursor.conclusion
    names = {argument for atom in iter_atoms(source) for argument in atom.arguments}
    return premises, names - bound


def match_or_introduction(target: Expression, source: Expression, definitions: dict) -> dict | None:
    """@brief Match a structural introduction against its cited definition.
    @details A matching OR premise supplies one exact alternative. Positive
    conjunction definitions flatten in order; signed leaves remain intact.
    Source parameters are rigid and local names map bijectively to the complete
    target binder set, with every use inside its binder's scope.
    @param target Claimed ordered implication chain.
    @param source Cited compact implication, or the full implication chain
    whose premise is the OR (a flattened root integration goal's rule,
    D-492).
    @param definitions Raw or typed compiled registry.
    @return Exact construction tree and OR path, or None for an incompatible row.
    @invariant Matching creates no assumptions and never omits a conjunct.
    """

    def children(atom: Atom) -> list[Expression]:
        """@brief Instantiate one compiled definition without changing local names.
        @details Registry parameters are positional in both supported schemas.
        @param atom Exact compact instance.
        @return Ordered substituted elements.
        """
        entry = definitions[atom.head]
        signature = entry['signature']
        formal = (parse_mpl(signature).arguments if isinstance(signature, str)
                  else tuple(f'u_{i + 1}' for i in range(entry['arity'])))
        assert len(formal) == len(atom.arguments)
        substitution = dict(zip(formal, atom.arguments, strict=True))
        return [map_arguments(parse_mpl(element), lambda name: substitution.get(name, name))
                for element in entry['elements']]

    offered = _source_premises(source, definitions, children)
    if offered is None:
        return None
    source_premises, rigid = offered
    premises, binders, groups = [], [], []
    cursor = target
    available = set(rigid)
    while isinstance(cursor, Implication):
        group = list(cursor.bound_variables)
        if len(set(group)) != len(group) or set(group) & available:
            return None
        binders.extend(group)
        groups.append(group)
        available.update(group)
        from .mpl import iter_atoms
        if any(argument not in available for atom in iter_atoms(cursor.premise) for argument in atom.arguments):
            return None
        premises.append(cursor.premise)
        cursor = cursor.conclusion
    if not premises or not isinstance(cursor, Atom) or definitions.get(cursor.head, {}).get('category') != 'or':
        return None
    if any(argument not in available for argument in cursor.arguments):
        return None

    for source_premise in source_premises:
        if not isinstance(source_premise, Atom) or source_premise.head != cursor.head:
            continue
        if len(source_premise.arguments) != len(cursor.arguments):
            continue
        mapping, reverse, compatible = {}, {}, True
        for formal, actual in zip(source_premise.arguments, cursor.arguments, strict=True):
            if formal in rigid:
                compatible &= formal == actual
            else:
                compatible &= (actual in binders and mapping.setdefault(formal, actual) == actual
                               and reverse.setdefault(actual, formal) == formal)
        if not compatible or set(reverse) != set(binders):
            continue
        root = map_arguments(source_premise, lambda name: mapping.get(name, name))
        pending = [(root, [], ())]
        while pending:
            node, path, ancestry = pending.pop()
            if isinstance(node, Atom) and definitions.get(node.head, {}).get('category') == 'or':
                assert node.head not in ancestry, 'cyclic OR definition'
                parts = children(node)
                assert len(parts) >= 2
                for index in reversed(range(len(parts))):
                    branch = ([1] * (len(parts) - 1) if index == 0
                              else [1] * (len(parts) - index - 1) + [2])
                    pending.append((parts[index], path + branch, (*ancestry, node.head)))
                continue
            leaves = []

            def conjunction(expression: Expression, parents: tuple) -> int | list:
                """@brief Build the selected positive conjunction's exact tree.
                @details Every signed leaf occupies one ordered premise slot.
                @param expression Current conjunct or signed leaf.
                @param parents Compiled conjunction ancestors for cycle checking.
                @return Premise ordinal or a binary conjunction construction.
                """
                if isinstance(expression, Atom) and definitions.get(expression.head, {}).get('category') == 'and':
                    assert expression.head not in parents, 'cyclic conjunction definition'
                    parts = children(expression)
                    assert parts
                    tree = conjunction(parts[0], (*parents, expression.head))
                    for part in parts[1:]:
                        tree = [tree, conjunction(part, (*parents, expression.head))]
                    return tree
                leaves.append(expression)
                return len(leaves) - 1

            tree = conjunction(node, ())
            if leaves == premises:
                return {'binder_groups': groups, 'tree': tree, 'path': path}
    return None
