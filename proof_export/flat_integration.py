# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
# Dual-licensed under AGPLv3 and a commercial license.
# See https://generative-logic.com/license. Contributions require CLA.

"""Explicit definition-construction certificates for flat integration rows."""

from __future__ import annotations

import itertools
import re
from collections.abc import Callable, Iterator

from .mpl import Atom, Expression, Implication, iter_atoms, map_arguments, parse_mpl, to_mpl


def build_flat_integration_plan(
    conclusion: Expression,
    compact: Atom,
    definitions: dict[str, dict[str, object]],
) -> dict[str, object]:
    """@brief Certify an explicit construction from a flat row's own premises.

    @details Enumerates only definition unfoldings allowed by the flat tag.
    Each distinct leaf must match exactly one recorded premise under an
    injective renaming of definition witnesses onto the row's bound
    variables; a leaf that repeats an earlier leaf of the same tree (an
    existence's binder premise that is also the first conjunct of the bundle
    it binds) reuses that leaf's premise, as the prover names it once. The
    result records premise positions and witness names, not a proof-search
    command or an additional theorem dependency. A disjunct's flat rule
    (D-508) binds the disjunct's open arguments,
    which stand at argument positions of its head: the head equals the
    template up to an injective renaming of those binders onto the template's
    arguments, none of which the rule otherwise uses, and the construction
    builds the head itself, its head binders being parameters, not witnesses.
    @param conclusion The complete implication row.
    @param compact Its cited integration-goal compact.
    @param definitions The certificate's compiled definition entries.
    @return A JSON-compatible conjunction/existence construction plan.
    """
    premises: list[Expression] = []
    binders: set[str] = set()
    cursor = conclusion
    while isinstance(cursor, Implication):
        binders.update(cursor.bound_variables)
        premises.append(cursor.premise)
        cursor = cursor.conclusion
    assert isinstance(cursor, Atom) and premises, "flat integration head or premises mismatch"
    assert cursor.head == compact.head and len(cursor.arguments) == len(compact.arguments)
    used = {argument for atom in iter_atoms(conclusion) for argument in atom.arguments}
    renaming: dict[str, str] = {}
    for head_argument, template_argument in zip(cursor.arguments, compact.arguments):
        if head_argument in binders:
            assert renaming.setdefault(head_argument, template_argument) == template_argument
            assert head_argument == template_argument or template_argument not in used, (
                "a head binder renamed onto a name the flat rule uses"
            )
        else:
            assert head_argument == template_argument, "flat integration head mismatch"
    assert len(set(renaming.values())) == len(renaming)
    binders -= set(renaming)
    compact = cursor
    assert compact.head in definitions
    assert definitions[compact.head]["category"] in {"and", "existence"}
    occupied = {argument for atom in iter_atoms(conclusion) for argument in atom.arguments}
    fresh_names: set[str] = set()
    serial = 0

    def fresh() -> str:
        """@brief Mint a definition-local witness token disjoint from the row.
        @details Tokens identify structural pattern variables only.
        @return One new token.
        """
        nonlocal serial
        serial += 1
        token = f"__flat_witness_{serial}"
        assert token not in occupied
        fresh_names.add(token)
        return token

    def variants(expression: Expression, active: tuple[str, ...], root: bool = False) -> Iterator[dict]:
        """@brief Enumerate the intact/extended definition trees at one node.
        @details Ad-hoc containers always unfold; configured containers may
        remain leaves. Implications, disjunctions and negations remain leaves.
        @param expression The instantiated node.
        @param active Definition heads on the current expansion path.
        @param root Whether this is the mandatory root expansion.
        @return An iterator of explicit construction trees.
        """
        entry = definitions.get(expression.head) if isinstance(expression, Atom) else None
        container = entry is not None and entry["category"] in {"and", "existence"}
        ad_hoc = isinstance(expression, Atom) and re.fullmatch(r"(?:and|existence)\d+", expression.head)
        if not root and (not container or not ad_hoc):
            yield {"kind": "premise", "expression": expression}
        if not container:
            assert not root
            return
        assert isinstance(expression, Atom) and isinstance(entry, dict)
        assert expression.head not in active, "cyclic flat integration definition"
        signature = parse_mpl(str(entry["signature"]))
        assert isinstance(signature, Atom) and len(signature.arguments) == len(expression.arguments)
        parameters = dict(zip(signature.arguments, expression.arguments))
        elements = [parse_mpl(str(element)) for element in entry["elements"]]
        assert len(elements) >= 2
        locals_by_name: dict[str, str] = {}
        groups: list[list[str]] = []
        for element in elements:
            introduced: list[str] = []
            for atom in iter_atoms(element):
                for argument in atom.arguments:
                    if not argument.startswith("u_") and argument not in locals_by_name:
                        locals_by_name[argument] = fresh()
                        introduced.append(locals_by_name[argument])
            groups.append(introduced)
        if entry["category"] == "and":
            assert not locals_by_name, "conjunction definition introduces an unbound witness"
        else:
            assert not groups[-1], "existence conclusion introduces an unbound witness"
        instantiated = [
            map_arguments(element, lambda name: locals_by_name.get(name, parameters.get(name, name)))
            for element in elements
        ]
        choices = [tuple(variants(element, (*active, expression.head))) for element in instantiated]
        for children in itertools.product(*choices):
            yield {"kind": entry["category"], "head": expression.head,
                   "children": list(children), "witness_groups": groups[:-1]}

    def leaves(tree: dict) -> list[Expression]:
        """@brief List definition leaves in construction order.
        @details Each position will be matched to a recorded premise.
        @param tree One unfolding tree.
        @return Its ordered leaf expressions.
        """
        if tree["kind"] == "premise":
            return [tree["expression"]]
        return [leaf for child in tree["children"] for leaf in leaves(child)]

    def assignments(expected: list[Expression]) -> Iterator[tuple[dict[str, str], list[int]]]:
        """@brief Match a complete leaf multiset to the exact row premises.
        @details Witness substitutions are consistent and injective; every
        recorded premise and binder must participate.
        @param expected Definition leaves in construction order.
        @return Valid witness maps and premise-position permutations.
        """
        if len(expected) != len(premises):
            return
        tokenize = lambda value: re.findall(r"[^\[\],()!]+|[\[\],()!]", to_mpl(value))
        expected_tokens = [tokenize(value) for value in expected]
        actual_tokens = [tokenize(value) for value in premises]

        def walk(index: int, mapping: dict[str, str], positions: list[int]):
            """@brief Match the next definition leaf without reusing a premise.
            @details Candidate order is the row's original premise order.
            @param index Next leaf position.
            @param mapping Witness substitutions already established.
            @param positions Premise positions already consumed.
            @return Complete structural assignments.
            """
            if index == len(expected_tokens):
                if set(mapping.values()) == binders:
                    yield mapping, positions
                return
            for position, actual in enumerate(actual_tokens):
                if position in positions or len(actual) != len(expected_tokens[index]):
                    continue
                candidate = dict(mapping)
                for left, right in zip(expected_tokens[index], actual):
                    if left in fresh_names:
                        if left in candidate:
                            if candidate[left] != right:
                                break
                        elif right not in binders or right in candidate.values():
                            break
                        else:
                            candidate[left] = right
                    elif left != right:
                        break
                else:
                    yield from walk(index + 1, candidate, [*positions, position])

        yield from walk(0, {}, [])

    for tree in variants(compact, (), root=True):
        # The one-level rule belongs to its existing integration tag.
        if all(child["kind"] == "premise" for child in tree["children"]):
            continue
        # A repeated leaf names the same premise as its first occurrence.
        distinct: list[Expression] = []
        slot_of_leaf: list[int] = []
        for leaf in leaves(tree):
            text = to_mpl(leaf)
            slot = next((k for k, kept in enumerate(distinct) if to_mpl(kept) == text), len(distinct))
            if slot == len(distinct):
                distinct.append(leaf)
            slot_of_leaf.append(slot)
        for mapping, positions in assignments(distinct):
            next_leaf = iter([positions[slot] for slot in slot_of_leaf])

            def seal(node: dict) -> dict:
                """@brief Replace pattern leaves and witnesses by certificate identifiers.
                @details The resulting plan contains no unresolved placeholders.
                @param node Matched construction node.
                @return A JSON-compatible proof-construction node.
                """
                if node["kind"] == "premise":
                    return {"kind": "premise", "index": next(next_leaf)}
                return {"kind": node["kind"], "head": node["head"],
                        "children": [seal(child) for child in node["children"]],
                        "witness_groups": [[mapping[name] for name in group]
                                           for group in node["witness_groups"]]}

            return {"premise_count": len(premises), "construction": seal(tree)}
    raise AssertionError("flat integration premises do not match a definition unfolding")


def render_flat_integration_term(
    plan: dict[str, object], premise_names: list[str], variable_name: Callable[[str], str]
) -> str:
    """@brief Render a certified flat construction as an explicit Lean term.
    @details Uses only the row's introduced premises, conjunction constructors,
    and explicit witnesses applied to negated-universal existence definitions.
    @param plan The structurally checked construction plan.
    @param premise_names Lean names of the row's premises in source order.
    @param variable_name Renderer for the row's bound witness names.
    @return A Lean proof term containing no proof-search tactic.
    """
    assert plan["premise_count"] == len(premise_names)
    serial = 0

    def render(node: dict) -> str:
        """@brief Render one constructor or premise reference.
        @details Fresh counterexample binders are local to their proof term.
        @param node A sealed construction node.
        @return Its explicit Lean proof term.
        """
        nonlocal serial
        if node["kind"] == "premise":
            index = int(node["index"])
            assert 0 <= index < len(premise_names)
            return premise_names[index]
        children = [render(child) for child in node["children"]]
        assert len(children) >= 2
        if node["kind"] == "and":
            term = children[0]
            for child in children[1:]:
                term = f"⟨{term}, {child}⟩"
            return term
        assert node["kind"] == "existence"
        groups = node["witness_groups"]
        assert len(groups) == len(children) - 1
        serial += 1
        counterexample = f"flat_counterexample_{serial}"
        arguments = [counterexample]
        for group, child in zip(groups, children[:-1]):
            arguments.extend(variable_name(name) for name in group)
            arguments.append(f"({child})")
        arguments.append(f"({children[-1]})")
        return f"(fun {counterexample} => {' '.join(arguments)})"

    return render(plan["construction"])
