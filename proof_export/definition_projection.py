# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""A narrow, capture-safe positive definition-and-conjunction projection rule."""
from collections import deque
from .mpl import Atom, Conjunction, Implication, Negation, iter_atoms, parse_mpl


def projection_alpha_equal(left, right):
    """
    @brief Compare formulas by lexical binders while keeping free tokens exact.

    @details Bound references carry tagged lexical identities, never printable
    variable names. A free token cannot collide with a normalized binder name.

    @param left First expression tree.
    @param right Second expression tree.
    @return Whether the trees differ only in bound-variable spelling.
    """

    def visit(a, b, left_bound, right_bound, depth):
        """
        @brief Compare one pair of nodes under their lexical binder environments.
        @details Each implication extends copied environments for its own scope.
        @param a Left node.
        @param b Right node.
        @param left_bound Left bound-token identities.
        @param right_bound Right bound-token identities.
        @param depth Lexical nesting depth.
        @return Exact structural and binding agreement.
        """
        if type(a) is not type(b):
            return False
        if isinstance(a, Atom):
            return (
                a.head == b.head
                and len(a.arguments) == len(b.arguments)
                and all(
                    left_bound.get(x, ("free", x)) == right_bound.get(y, ("free", y))
                    for x, y in zip(a.arguments, b.arguments)
                )
            )
        if isinstance(a, Conjunction):
            return len(a.parts) == len(b.parts) and all(
                visit(x, y, left_bound, right_bound, depth)
                for x, y in zip(a.parts, b.parts)
            )
        if isinstance(a, Negation):
            return visit(a.inner, b.inner, left_bound, right_bound, depth)
        assert isinstance(a, Implication)
        if len(a.bound_variables) != len(b.bound_variables):
            return False
        assert len(set(a.bound_variables)) == len(a.bound_variables)
        assert len(set(b.bound_variables)) == len(b.bound_variables)
        next_left, next_right = dict(left_bound), dict(right_bound)
        for index, (x, y) in enumerate(zip(a.bound_variables, b.bound_variables)):
            next_left[x] = next_right[y] = ("bound", depth, index)
        return visit(a.premise, b.premise, next_left, next_right, depth + 1) and visit(
            a.conclusion, b.conclusion, next_left, next_right, depth + 1
        )

    return visit(left, right, {}, {}, 0)


def instantiate_projection_part(expression, parameters):
    """
    @brief Instantiate a definition constituent without capturing a free argument.

    @details Freshen every local binder before replacing free formal parameters.
    Binder shadowing remains lexical. Free tokens outside the formal signature
    are invalid for a positive conjunction definition and assert.

    @param expression Constituent in formal definition vocabulary.
    @param parameters Formal-to-actual argument mapping.
    @return Capture-free instantiated expression.
    """
    occupied = {
        argument for atom in iter_atoms(expression) for argument in atom.arguments
    }
    occupied.update(parameters)
    occupied.update(parameters.values())
    next_index = 0

    def visit(node, bound):
        """
        @brief Instantiate one lexical expression node.
        @details A copied binder map separates local names from formal ports.
        @param node Current expression tree.
        @param bound Original-to-fresh bound-name mapping.
        @return Rebuilt instantiated node.
        """
        nonlocal next_index
        if isinstance(node, Atom):
            assert all(
                x in bound or x in parameters for x in node.arguments
            ), "unbound definition port"
            return Atom(
                node.head,
                tuple(
                    bound[x] if x in bound else parameters[x] for x in node.arguments
                ),
            )
        if isinstance(node, Conjunction):
            return Conjunction(tuple(visit(part, bound) for part in node.parts))
        if isinstance(node, Negation):
            return Negation(visit(node.inner, bound))
        assert isinstance(node, Implication)
        assert len(set(node.bound_variables)) == len(node.bound_variables)
        nested = dict(bound)
        names = []
        for old in node.bound_variables:
            while True:
                next_index += 1
                name = f"projection_bound_{next_index}"
                if name not in occupied:
                    break
            occupied.add(name)
            nested[old] = name
            names.append(name)
        return Implication(
            tuple(names), visit(node.premise, nested), visit(node.conclusion, nested)
        )

    return visit(expression, {})


def projection_children(source, definitions):
    """
    @brief Enumerate the only legal one-hop positive projection transitions.

    @details Binary conjunctions expose their two parts. A named and-definition
    exposes its instantiated constituents with the writer's left-associated
    product paths. Every other expression is an opaque leaf with zero children;
    implication, negation, disjunction and existence elimination are excluded.

    @param source Current expression on the projection path.
    @param definitions Exact raw or neutral compiled-definition registry.
    @return List of child expressions paired with explicit projection metadata.
    """
    if isinstance(source, Conjunction):
        assert len(source.parts) == 2, "projection requires a binary conjunction tree"
        return [
            (
                part,
                {
                    "kind": "conjunction",
                    "element": index,
                    "count": 2,
                    "directions": [index + 1],
                },
            )
            for index, part in enumerate(source.parts)
        ]
    if not isinstance(source, Atom) or source.head not in definitions:
        return []
    definition = definitions[source.head]
    if definition["category"] != "and":
        return []
    signature = definition["signature"]
    if isinstance(signature, str):
        formal = parse_mpl(signature)
        assert isinstance(formal, Atom) and formal.head == source.head
        names = formal.arguments
    else:
        assert isinstance(signature, list)
        names = tuple(f"u_{index+1}" for index in range(int(definition["arity"])))
    assert (
        len(names)
        == len(set(names))
        == len(source.arguments)
        == int(definition["arity"])
    )
    parts = definition["elements"]
    assert isinstance(parts, list) and parts
    parameters = dict(zip(names, source.arguments))
    count = len(parts)
    return [
        (
            instantiate_projection_part(parse_mpl(text), parameters),
            {
                "kind": "definition",
                "head": source.head,
                "element": index,
                "count": count,
                "directions": [1] * (count - 1 - index) + ([2] if index > 0 else []),
            },
        )
        for index, text in enumerate(parts)
    ]


def find_definition_projection(target, source, definitions):
    """
    @brief Find a shortest legal positive projection path, if the rule applies.

    @details Breadth-first search counts explicit definition/conjunction hops.
    Opaque leaves and absent targets are ordinary non-applicable candidates.
    Recursive positive definitions assert instead of yielding an incomplete path.
    The rule requires at least one hop; it never serves as an identity alias.

    @param target Claimed conclusion with exact free arguments.
    @param source Explicit existing source fact in the same namespace.
    @param definitions Matching compiled-definition registry.
    @return Checked path/count dictionary, or None when outside this rule.
    """
    pending = deque([(source, [], frozenset())])
    while pending:
        current, path, unfolded = pending.popleft()
        if path and projection_alpha_equal(current, target):
            return {
                "path": path,
                "definition_unfolds": sum(hop["kind"] == "definition" for hop in path),
                "conjunction_projections": sum(len(hop["directions"]) for hop in path),
            }
        children = projection_children(current, definitions)
        if children and isinstance(current, Atom):
            assert current.head not in unfolded, "recursive conjunction definition"
            unfolded = unfolded | {current.head}
        for child, hop in children:
            pending.append((child, path + [hop], unfolded))
    return None


def build_definition_projection_plan(target, source, definitions):
    """
    @brief Require a fully justified direct definition-projection proof plan.

    @details Certificate construction and the verifier reject unsupported
    conclusions; no unproved target or fallback proof action is returned.

    @param target Claimed conclusion.
    @param source Explicit source fact.
    @param definitions Matching compiled-definition registry.
    @return Complete positive projection path and expanded-work counts.
    """
    plan = find_definition_projection(target, source, definitions)
    assert plan is not None, "conclusion is not a positive definition projection"
    return plan
