# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Checked branch-local equality support for a fixed OR conclusion."""

from .mpl import Atom, Expression, Negation, map_arguments, parse_mpl, to_mpl


def convergence_cases(compact: Atom, parent: str, definitions: dict) -> list[tuple[Expression, str]]:
    """@brief Instantiate the exact ordered cases of a compiled disjunction.
    @details Positive nested OR definitions flatten recursively. Signed leaves
    retain their polarity and the outer compact supplies every case scope.
    @param compact The cited outer disjunction.
    @param parent Its exact parent scope.
    @param definitions Matching raw or typed compiled definitions.
    @return Ordered leaf expressions and their exact case scopes.
    @invariant No intermediate case or fresh binder is introduced.
    """
    result = []

    def visit(expression: Expression, ancestry: tuple[str, ...]) -> None:
        """@brief Append the non-OR leaves of one instantiated expression.
        @details Recursion follows only positive category-or definitions.
        @param expression Current expression.
        @param ancestry Definition heads on the current recursion path.
        @return Nothing; appends exact leaves to the enclosing result.
        """
        if not isinstance(expression, Atom) or definitions.get(expression.head, {}).get('category') != 'or':
            result.append((expression, parent + '_boundary_ordis_' + to_mpl(compact) + '_(' + to_mpl(expression) + ')'))
            return
        assert expression.head not in ancestry, 'cyclic OR definition'
        definition = definitions[expression.head]
        signature = definition.get('signature')
        formals = (parse_mpl(signature).arguments if isinstance(signature, str)
                   else tuple(f'u_{i+1}' for i in range(len(expression.arguments))))
        assert len(formals) == len(expression.arguments)
        substitution = dict(zip(formals, expression.arguments, strict=True))
        assert len(definition['elements']) >= 2
        for element in definition['elements']:
            visit(map_arguments(parse_mpl(element), lambda name: substitution.get(name, name)),
                  (*ancestry, expression.head))

    assert definitions.get(compact.head, {}).get('category') == 'or'
    visit(compact, ())
    assert len(result) >= 2
    return result


def match_equality_convergence(target: Expression, parent: str,
                               cases: list[tuple[Expression, str]],
                               dependencies: list[tuple[Expression, str]]) -> list[dict] | None:
    """@brief Check every case against one fixed conclusion using cited equalities.
    @details Each case uses a source at that case or its ancestors, and only
    positive equalities visible there. Flat signed predicate arguments must
    agree positionally modulo those equality edges. The returned plan cites
    exact dependency indexes; all supplied dependencies must participate.
    @param target The fixed parent conclusion.
    @param parent The exact cohort parent.
    @param cases Ordered disjunct expressions with their case scopes.
    @param dependencies Cited actual facts and equalities, excluding the OR.
    @return Per-case source and equality indexes, or None for an incompatible row.
    @invariant Sibling equalities, binder substitution and uncited facts cannot contribute.
    """
    def visible(source: str, scope: str) -> bool:
        """@brief Test exact scope ancestry.
        @details Boundary-delimited ancestry excludes siblings and descendants.
        @param source Cited source scope.
        @param scope The case using that source.
        @return Whether the source is inherited by this case.
        """
        return source == scope or scope.startswith(source + '_boundary_')

    def signed_atom(expression: Expression) -> tuple[bool, Atom] | None:
        """@brief Classify a flat signed predicate.
        @details Compound binder-bearing expressions are outside this contract.
        @param expression Claimed source or conclusion.
        @return Sign and atom, or None for a compound expression.
        """
        negative = isinstance(expression, Negation)
        atom = expression.inner if negative else expression
        return (negative, atom) if isinstance(atom, Atom) else None

    goal = signed_atom(target)
    if goal is None or len(cases) < 2 or len(set(dependencies)) != len(dependencies):
        return None
    plans, used, changed = [], set(), False
    for disjunct, branch in cases:
        if not branch.startswith(parent + '_boundary_ordis_'):
            return None
        # Retired cases keep the existing explicit refutation contract.
        negated = disjunct.inner if isinstance(disjunct, Negation) else Negation(disjunct)
        retired = next((i for i, (expression, scope) in enumerate(dependencies)
                        if expression == negated and (visible(scope, parent) or scope == branch)), None)
        if retired is not None:
            plans.append({'source': retired, 'equalities': [], 'retired': True})
            used.add(retired)
            continue
        equalities = [(i, expression) for i, (expression, scope) in enumerate(dependencies)
                      if visible(scope, branch) and isinstance(expression, Atom)
                      and expression.head == '=' and len(expression.arguments) == 2]
        matched = None
        for i, (expression, scope) in enumerate(dependencies):
            source = signed_atom(expression)
            if not visible(scope, branch) or source is None:
                continue
            if source[0] != goal[0] or source[1].head != goal[1].head or len(source[1].arguments) != len(goal[1].arguments):
                continue
            edges = set()
            compatible = True
            for left, right in zip(source[1].arguments, goal[1].arguments, strict=True):
                reached = {left: []}
                pending = [left]
                for value in pending:
                    for index, equality in equalities:
                        a, b = equality.arguments
                        if value != a and value != b:
                            continue
                        peer = b if value == a else a
                        if peer not in reached:
                            reached[peer] = [*reached[value], index]
                            pending.append(peer)
                if right not in reached:
                    compatible = False
                    break
                edges.update(reached[right])
            if compatible:
                matched = {'source': i, 'equalities': sorted(edges), 'retired': False}
                changed |= expression != target
                used.add(i)
                used.update(edges)
                break
        if matched is None:
            return None
        plans.append(matched)
    if not changed or used != set(range(len(dependencies))):
        return None
    return plans
