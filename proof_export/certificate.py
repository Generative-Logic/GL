# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt)
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Construction and deterministic serialization of neutral GL proof certificates."""

from __future__ import annotations

import json
import re
from itertools import permutations
from pathlib import Path

from .citation_policy import resolve_citation_source
from .graph import (
    ORDINARY_SCOPE,
    INTEGRATION_GOAL_SCOPE,
    Chapter,
    ProofRow,
    ScopedExpression,
    TheoremRecord,
    find_theorem,
    load_theorem_records,
    sha256_file,
)
from .flat_integration import build_flat_integration_plan
from .implication_composition import build_implication_composition_plan, validate_composition_scopes
from .integration_compact import compact_rule_rows, is_introduction_compact_row
from .mpl import (
    Atom,
    Conjunction,
    Expression,
    Implication,
    Negation,
    alpha_key,
    expression_json,
    iter_atoms,
    map_arguments,
    parse_mpl,
    to_mpl,
)
from .types import EQUALITY, infer_expression_variables, load_type_environment


# Element positions before the set position of the membership operators that
# subset implications use (I-267).
SUBSET_MEMBERSHIP_WIDTH = {"in": 1, "in2": 2, "in3": 3}

# The alias a twin weak step block gives its twin partner's predecessor with
# the variable copy `(=[q,rec2])`; like `<source>_copy` it is a second name of
# one source (D-581).
TWIN_PREDECESSOR_ALIAS = "rec2"


def is_subset_implication(entry: dict[str, object]) -> bool:
    """
    @brief Decide whether one binary entry is a subset implication.
    @details
    A subset implication is `E[u_1,u_2] := (op[b..,u_1]) -> (op[b..,u_2])` with
    `op` one of `in`, `in2`, `in3`: the same distinct bound variables fill the
    element positions of premise and head, and the set position holds `u_1` in
    the premise and `u_2` in the head. Two of them, one per direction,
    integrate a set- or relation-typed `=` goal (I-267).
    @param entry One GL binary definition.
    @return True for a subset implication.
    """

    if entry.get("category") != "implication":
        return False
    signature = parse_mpl(str(entry.get("signature", "")))
    if not isinstance(signature, Atom) or signature.arguments != ("u_1", "u_2"):
        return False
    elements = entry.get("elements")
    if not isinstance(elements, list) or len(elements) != 2:
        return False
    premise, head = (parse_mpl(str(element)) for element in elements)
    if not isinstance(premise, Atom) or not isinstance(head, Atom):
        return False
    width = SUBSET_MEMBERSHIP_WIDTH.get(premise.head)
    if width is None or head.head != premise.head:
        return False
    if len(premise.arguments) != width + 1 or len(head.arguments) != width + 1:
        return False
    bound = premise.arguments[:width]
    return (
        bound == head.arguments[:width]
        and len(set(bound)) == width
        and not any(argument.startswith("u_") for argument in bound)
        and premise.arguments[width] == "u_1"
        and head.arguments[width] == "u_2"
    )


ACTION_BY_TAG = {
    "zoom in": "scope_inherit",
    "anchor handling": "anchor_specialize",
    "compilation": "definition_compact",
    "contradiction": "ex_falso",
    "disintegration": "compound_project",
    "definition projection": "definition_project",
    "equality1": "equality_substitute",
    "equality2": "equality_transitive",
    "expansion": "definition_expand",
    "expansion for integration": "definition_expand",
    "implication": "implication_apply",
    "implication composition": "implication_compose",
    "implication grouping": "implication_group",
    "contraposition": "implication_contrapose",
    "or theorem": "or_construct",
    "or elimination": "or_eliminate_theorem",
    "or convergence": "or_eliminate",
    "or disintegration": "or_branch",
    "or branch assumption": "or_integration_assume",
    "or branch proven": "or_introduce",
    "recursion": "induction_assumption",
    "reformulation for integration >[bound]": "classical_reformulate_bound",
    "reformulation for integration >[]": "classical_reformulate_empty",
    "reformulation for integration and": "integration_conjoin",
    "reformulation for integration flattened": "definition_construct",
    "premise element": "integration_premise",
    "reformulated from": "theorem_reformulate",
    "symmetry of equality": "equality_symmetric",
    "symmetry of inequality": "inequality_symmetric",
    "task formulation": "assume",
    "theorem": "theorem_reference",
    "vacuous truth": "ex_falso",
    "variable copy": "variable_alias",
    "validity name": "integration_discharge",
}


def _portable_source_path(path: Path) -> str:
    """
    @brief Form a portable source label without checkout-specific prefixes.
    @details Known repository roots are retained from their stable directory
    marker; an external input falls back to its filename.
    @param path Source artifact path.
    @return Repository-relative or filename-only source label.
    """

    parts = path.resolve().parts
    for marker in ("proof_export", "files", "tests"):
        if marker in parts:
            return Path(*parts[parts.index(marker):]).as_posix()
    return path.name


def _expression_key(
    scoped_expression: ScopedExpression,
    namespace: str,
) -> tuple[str, str, str]:
    """
    @brief Form the exact local citation key for a scoped expression.
    @details Logical MPL, namespace, and proof scope remain distinct key parts.
    @param scoped_expression Parsed expression with its proof-graph spelling.
    @param namespace Proof-graph namespace label.
    @return Exact expression, namespace, and proof-scope tuple.
    """

    return (
        to_mpl(scoped_expression.expression),
        namespace,
        scoped_expression.proof_scope,
    )


def _alpha_expression_key(
    scoped_expression: ScopedExpression,
    namespace: str,
) -> tuple[str, str, str]:
    """
    @brief Form a citation key insensitive to bound-variable spelling.
    @details Namespace and proof scope remain exact while logical binders are
    alpha-normalized.
    @param scoped_expression Parsed expression with its proof-graph spelling.
    @param namespace Proof-graph namespace label.
    @return Alpha-normalized expression, namespace, and proof-scope tuple.
    """

    return (
        alpha_key(scoped_expression.expression),
        namespace,
        scoped_expression.proof_scope,
    )


def _scoped_expression_json(scoped_expression: ScopedExpression) -> dict[str, object]:
    """
    @brief Encode logical MPL separately from its proof-graph scope spelling.
    @details The structured expression is augmented with the exact source text
    and its normalized proof-scope classification.
    @param scoped_expression Parsed scoped proof-graph expression.
    @return JSON-ready structured expression record.
    """

    encoded = expression_json(scoped_expression.expression)
    encoded["source_mpl"] = scoped_expression.source_mpl
    encoded["proof_scope"] = scoped_expression.proof_scope
    return encoded


def _or_construction_disjuncts(
    theorem_expression: Expression, row: ProofRow
) -> tuple[Atom, list[Expression], list[list[object]]]:
    """
    @brief Read a constructed OR theorem's disjuncts from its two cited parents.
    @details
    The first parent states ``not D1 implies ... implies Dn`` over the shared
    theorem premises, so the constructed head denotes the recorded
    disjunction in that construction order. A folded premise ``not Di`` may
    stand before shared premises that bind variables it does not use; the
    constructed theorem states every shared premise first. Each parent
    premise is therefore either the theorem's next premise layer (binders
    alpha-renamed, premise asserted identical) or a binder-free folded
    premise. The second parent must contain the same disjunct multiset in a
    different checked order.

    Disjuncts carry their true polarity and a NEGATED disjunct keeps its
    negation. The parent premise for a disjunct ``D`` is ``not D`` with double
    negation cancelled: a positive ``D`` appears as the negation premise
    ``not D``, while a negated ``D = not X`` appears as the PLAIN premise
    ``X``. Both parent walks therefore invert each premise with the same
    cancellation instead of demanding a literal negation node — a blind
    negation prefix would flip a negated disjunct's sign.
    @param theorem_expression The constructed OR theorem.
    @param row Its single ``or theorem`` row, citing the source parent first
        and the companion second.
    @return The theorem's OR atom; its disjuncts in construction order, in
        the theorem's variable names; and the first parent's premise order
        over the theorem body (after the anchor): ``["premise", k]`` for the
        body's k-th premise layer, ``["case", k]`` for the negation of the
        k-th construction disjunct.
    @invariant Every OR row has at least two distinct disjuncts and two
        equivalent parents.
    """

    assert row.tag == "or theorem" and len(row.dependencies) == 2

    def align_parent(parent):
        """
        @brief Align one OR parent theorem with the constructed theorem's premises.
        @details A parent premise that matches the theorem's next premise
        layer is shared: its binders are alpha-renamed and the premise is
        asserted identical. Any other parent premise is a binder-free folded
        premise of a disjunct, kept in chain order. Every theorem layer must
        be matched.
        @param parent Parsed parent theorem expression.
        @return Constructed OR atom, the folded premises, the parent's final
            head, the binder renaming, and the parent's premise order.
        """

        target_cursor = theorem_expression
        parent_cursor = parent
        renaming: dict[str, str] = {}
        folded: list[Expression] = []
        order: list[list[object]] = []
        layer = 0
        while isinstance(parent_cursor, Implication):
            shared = False
            if isinstance(target_cursor, Implication) and len(
                target_cursor.bound_variables
            ) == len(parent_cursor.bound_variables):
                candidate = dict(renaming)
                for parent_name, target_name in zip(
                    parent_cursor.bound_variables,
                    target_cursor.bound_variables,
                    strict=True,
                ):
                    existing = candidate.get(parent_name)
                    assert existing is None or existing == target_name
                    candidate[parent_name] = target_name
                mapped_premise = map_arguments(
                    parent_cursor.premise,
                    lambda name: candidate.get(name, name),
                )
                shared = mapped_premise == target_cursor.premise
            if shared:
                renaming = candidate
                order.append(["premise", layer])
                layer += 1
                target_cursor = target_cursor.conclusion
            else:
                assert not parent_cursor.bound_variables
                folded.append(parent_cursor.premise)
                order.append(["case", len(folded)])
            parent_cursor = parent_cursor.conclusion
        assert isinstance(target_cursor, Atom) and target_cursor.head.startswith("or")
        return target_cursor, folded, parent_cursor, renaming, order

    def negated_premise_to_disjunct(premise: Expression) -> Expression:
        """
        @brief Recover a disjunct from its ``not D`` parent premise.
        @details Double negation is cancelled at emission, so a negation
        premise denotes a positive disjunct and a plain premise denotes a
        negated one. Both are defined, intended forms.
        @param premise Parent premise expression at one disjunct position.
        @return The disjunct in its true recorded polarity.
        """

        if isinstance(premise, Negation):
            return premise.inner
        return Negation(premise)

    or_atom, source_folded, source_head, source_renaming, source_order = align_parent(
        row.dependencies[0].expression
    )
    disjuncts = [negated_premise_to_disjunct(premise) for premise in source_folded]
    disjuncts.append(source_head)
    assert len(disjuncts) >= 2, "constructed OR needs at least two disjuncts"
    mapped_disjuncts = [
        map_arguments(item, lambda name: source_renaming.get(name, name))
        for item in disjuncts
    ]

    companion_atom, companion_folded, companion_head, companion_renaming, _ = align_parent(
        row.dependencies[1].expression
    )
    assert companion_atom == or_atom
    companion_disjuncts = [
        negated_premise_to_disjunct(premise) for premise in companion_folded
    ]
    companion_disjuncts.append(companion_head)
    mapped_companion = [
        map_arguments(item, lambda name: companion_renaming.get(name, name))
        for item in companion_disjuncts
    ]
    assert sorted(to_mpl(item) for item in mapped_companion) == sorted(
        to_mpl(item) for item in mapped_disjuncts
    )
    assert len({to_mpl(item) for item in mapped_disjuncts}) == len(
        mapped_disjuncts
    )
    # The anchor is the first shared layer; the replay works on the body.
    assert source_order[0] == ["premise", 0]
    body_order = [
        [kind, index - 1] if kind == "premise" else [kind, index]
        for kind, index in source_order[1:]
    ]
    return or_atom, mapped_disjuncts, body_order


def _or_disjunct_positions(
    definition: dict[str, object], or_atom: Atom, disjuncts: list[Expression]
) -> list[int]:
    """
    @brief Place each construction disjunct of an OR theorem in its operator.
    @details
    One OR operator serves every disjunct order of its body (GL reuses the
    operator of a canonical body), so an OR theorem may list its disjuncts in
    another order than the operator stores them. The stored elements,
    instantiated with the theorem's head arguments, must be exactly the
    construction disjuncts in some order; the result names, per construction
    disjunct, its stored position.
    @param definition The operator: ``elements`` over ``u_<k>`` parameters in
        stored order.
    @param or_atom The theorem's OR atom; its arguments fill ``u_1``, ``u_2``, ...
    @param disjuncts The construction disjuncts in the theorem's variable names.
    @return Zero-based stored position of each construction disjunct.
    @invariant The positions are a permutation of the stored positions.
    """

    parameters = {
        f"u_{index + 1}": argument
        for index, argument in enumerate(or_atom.arguments)
    }
    instances = [
        to_mpl(map_arguments(parse_mpl(str(element)), lambda name: parameters.get(name, name)))
        for element in definition["elements"]
    ]
    keys = [to_mpl(item) for item in disjuncts]
    assert len(set(keys)) == len(keys) == len(instances), (
        f"{or_atom.head}: the theorem's disjuncts do not match the operator's elements"
    )
    assert sorted(keys) == sorted(instances), (
        f"{or_atom.head}: the theorem's disjuncts do not match the operator's elements"
    )
    return [instances.index(key) for key in keys]


def _constructed_or_definition(
    theorem: TheoremRecord, binary_entries: dict[str, dict[str, object]]
) -> dict[str, object]:
    """
    @brief Reconstruct one post-prove OR operator from its OR theorem.
    @details
    An operator GL's binary holds keeps GL's stored element order and
    signature: GL keys every OR by its canonical body, so the theorem may
    state the disjunction in another order, and the theorem's disjuncts must
    be the stored elements instantiated with its head arguments (asserted).
    An operator the binary lacks (a shortcut-local head, a test fixture) is
    defined by the theorem itself: its elements in construction order, its
    head arguments in their first appearance in that order (asserted). The
    result is a neutral certificate fact checked against the processed row,
    never a backend guess.
    @param theorem Parsed theorem record whose method is ``or theorem``.
    @param binary_entries GL binary definitions of the source batch.
    @return The operator name, the theorem's head arguments, and the elements.
    @invariant Every OR row has at least two distinct disjuncts and two equivalent parents.
    """

    assert theorem.method == "or theorem"
    assert len(theorem.chapters) == 1
    chapter = theorem.chapters[0]
    assert len(chapter.rows) == 1
    or_atom, mapped_disjuncts, _ = _or_construction_disjuncts(
        theorem.expression, chapter.rows[0]
    )
    stored = binary_entries.get(or_atom.head)
    if stored is not None:
        assert stored["category"] == "or"
        _or_disjunct_positions(stored, or_atom, mapped_disjuncts)
        return {
            "head": or_atom.head,
            "arity": len(or_atom.arguments),
            "category": "or",
            "elements": list(stored["elements"]),
            "arguments": list(or_atom.arguments),
            "signature": stored["signature"],
            "definition_source": "or_theorem_action",
        }

    ordered_arguments: list[str] = []
    for disjunct in mapped_disjuncts:
        for atom in iter_atoms(disjunct):
            for argument in atom.arguments:
                if argument not in ordered_arguments:
                    ordered_arguments.append(argument)
    assert tuple(ordered_arguments) == or_atom.arguments
    argument_to_parameter = {
        argument: f"u_{index + 1}"
        for index, argument in enumerate(ordered_arguments)
    }
    elements = [
        to_mpl(
            map_arguments(
                disjunct,
                lambda name: argument_to_parameter[name],
            )
        )
        for disjunct in mapped_disjuncts
    ]
    return {
        "head": or_atom.head,
        "arity": len(ordered_arguments),
        "category": "or",
        "elements": elements,
        "arguments": ordered_arguments,
        "signature": (
            f"({or_atom.head}["
            + ",".join(f"u_{index + 1}" for index in range(len(ordered_arguments)))
            + "])"
        ),
        "definition_source": "or_theorem_action",
    }


def _definition_from_expansion(row: ProofRow) -> dict[str, object]:
    """
    @brief Reconstruct a missing compiled definition from one checked source row.
    @details
    Expansion rows expose conjunction, disjunction, implication, or classical-
    existence bodies directly. Compilation rows expose the reverse definitional
    direction from an expanded implication to its compact atom. A reformulated-
    statement row exposes an existence body by
    matching the target theorem's additional premises against the cited source
    theorem's premises and retaining the unmatched witness-bearing elements.
    The source head is never a candidate: reformulation keeps it inside the
    existence together with the moved definition, so a retained premise can
    only be a source premise. In both
    cases the compact atom fixes parameter order and local names are assigned
    deterministically by first appearance.
    @param row Processed expansion, compilation, or ``reformulated from`` source row.
    @return Canonical compiled-definition entry suitable for type inference and rendering.
    @invariant Every non-parameter argument is bound by the checked source shape.
    """

    assert len(row.dependencies) == 1
    elements: list[Expression] = []
    bound_variables: set[str] = set()
    target_bound: set[str] = set()
    source_to_target: dict[str, str] = {}

    if row.tag == "reformulated from":
        category = "existence"
        target_cursor = row.conclusion.expression
        source_cursor = row.dependencies[0].expression
        while isinstance(target_cursor, Implication) and isinstance(
            source_cursor,
            Implication,
        ):
            if len(target_cursor.bound_variables) != len(
                source_cursor.bound_variables
            ):
                break
            trial = dict(source_to_target)
            for source_name, target_name in zip(
                source_cursor.bound_variables,
                target_cursor.bound_variables,
                strict=True,
            ):
                existing = trial.get(source_name)
                assert existing is None or existing == target_name
                trial[source_name] = target_name
            mapped_source_premise = map_arguments(
                source_cursor.premise,
                lambda name: trial.get(name, name),
            )
            if mapped_source_premise != target_cursor.premise:
                break
            source_to_target = trial
            target_cursor = target_cursor.conclusion
            source_cursor = source_cursor.conclusion

        target_premises: list[Expression] = []
        while isinstance(target_cursor, Implication):
            target_bound.update(target_cursor.bound_variables)
            target_premises.append(target_cursor.premise)
            target_cursor = target_cursor.conclusion
        assert isinstance(target_cursor, Atom)
        source = target_cursor
        assert len(source.arguments) == len(set(source.arguments))

        source_bound: set[str] = set()
        while isinstance(source_cursor, Implication):
            source_bound.update(source_cursor.bound_variables)
            elements.append(source_cursor.premise)
            source_cursor = source_cursor.conclusion
        elements.append(source_cursor)
        # The head stays in the existence body; only premises can be retained.
        head_index = len(elements) - 1

        unmatched = list(range(len(elements)))
        for target_premise in target_premises:
            solutions: list[tuple[int, dict[str, str]]] = []
            candidates = [index for index in unmatched if index != head_index]
            for element_index in candidates:
                trial = dict(source_to_target)
                pending_pairs: list[tuple[Expression, Expression]] = [
                    (elements[element_index], target_premise)
                ]
                valid = True
                while pending_pairs and valid:
                    source_part, target_part = pending_pairs.pop()
                    if type(source_part) is not type(target_part):
                        valid = False
                    elif isinstance(source_part, Atom):
                        assert isinstance(target_part, Atom)
                        if (
                            source_part.head != target_part.head
                            or len(source_part.arguments)
                            != len(target_part.arguments)
                        ):
                            valid = False
                            continue
                        for source_name, target_name in zip(
                            source_part.arguments,
                            target_part.arguments,
                            strict=True,
                        ):
                            if source_name in trial:
                                if trial[source_name] != target_name:
                                    valid = False
                                    break
                            elif source_name in source_bound:
                                trial[source_name] = target_name
                            elif source_name != target_name:
                                valid = False
                                break
                    elif isinstance(source_part, Conjunction):
                        assert isinstance(target_part, Conjunction)
                        if len(source_part.parts) != len(target_part.parts):
                            valid = False
                        else:
                            pending_pairs.extend(
                                zip(source_part.parts, target_part.parts, strict=True)
                            )
                    elif isinstance(source_part, Negation):
                        assert isinstance(target_part, Negation)
                        pending_pairs.append((source_part.inner, target_part.inner))
                    else:
                        assert isinstance(source_part, Implication)
                        assert isinstance(target_part, Implication)
                        if len(source_part.bound_variables) != len(
                            target_part.bound_variables
                        ):
                            valid = False
                            continue
                        for source_name, target_name in zip(
                            source_part.bound_variables,
                            target_part.bound_variables,
                            strict=True,
                        ):
                            trial[source_name] = target_name
                        pending_pairs.extend(
                            (
                                (source_part.premise, target_part.premise),
                                (source_part.conclusion, target_part.conclusion),
                            )
                        )
                if valid:
                    solutions.append((element_index, trial))
            assert len(solutions) == 1
            matched_index, source_to_target = solutions[0]
            unmatched.remove(matched_index)
        elements = [elements[index] for index in unmatched]
        bound_variables = source_bound
    else:
        assert row.tag in {
            "compilation",
            "expansion",
            "expansion for integration",
        }
        source = (
            row.conclusion.expression
            if row.tag == "compilation"
            else row.dependencies[0].expression
        )
        assert isinstance(source, Atom)
        assert len(source.arguments) == len(set(source.arguments))
        cursor = (
            row.dependencies[0].expression
            if row.tag == "compilation"
            else row.conclusion.expression
        )
        if isinstance(cursor, Conjunction):
            category = "and"
            pending_conjuncts: list[Expression] = [cursor]
            while pending_conjuncts:
                expression = pending_conjuncts.pop()
                if isinstance(expression, Conjunction):
                    pending_conjuncts.extend(reversed(expression.parts))
                else:
                    elements.append(expression)
        else:
            if isinstance(cursor, Negation):
                if isinstance(cursor.inner, Conjunction) and all(
                    isinstance(part, Negation) for part in cursor.inner.parts
                ):
                    category = "or"
                    elements.extend(
                        part.inner
                        for part in cursor.inner.parts
                        if isinstance(part, Negation)
                    )
                    cursor = None
                else:
                    category = "existence"
                    cursor = cursor.inner
            else:
                category = "implication"
            if cursor is not None:
                while isinstance(cursor, Implication):
                    bound_variables.update(cursor.bound_variables)
                    elements.append(cursor.premise)
                    cursor = cursor.conclusion
                if category == "existence":
                    assert isinstance(cursor, Negation)
                    cursor = cursor.inner
                elements.append(cursor)
    assert len(elements) >= 2

    replacements = {
        argument: f"u_{index + 1}"
        for index, argument in enumerate(source.arguments)
    }
    used_element_names = {
        argument
        for element in elements
        for atom in iter_atoms(element)
        for argument in atom.arguments
    }
    for source_name, target_name in source_to_target.items():
        if source_name not in used_element_names:
            continue
        if target_name in replacements:
            replacements[source_name] = replacements[target_name]
        else:
            assert target_name in target_bound, (
                f"used reformulation target {target_name!r} from {source_name!r} "
                "is neither a compact parameter nor a checked target binder"
            )
            replacements[source_name] = target_name
    next_local = 0

    def canonical_name(name: str) -> str:
        """
        @brief Map one expansion variable into canonical definition spelling.
        @details Operator arguments use `u_` parameters and existential locals
        receive monotonically increasing numeric names.
        @param name Source expansion variable.
        @return Canonical definition variable token.
        """

        nonlocal next_local
        if name in replacements:
            return replacements[name]
        assert name in bound_variables
        next_local += 1
        replacements[name] = str(next_local)
        return replacements[name]

    canonical_elements = [
        to_mpl(map_arguments(element, canonical_name)) for element in elements
    ]
    assert set(source.arguments).issubset(replacements)
    assert bound_variables.issubset(replacements)
    return {
        "head": source.head,
        "arity": len(source.arguments),
        "category": category,
        "elements": canonical_elements,
        "signature": (
            f"({source.head}["
            + ",".join(f"u_{index + 1}" for index in range(len(source.arguments)))
            + "])"
        ),
        "definition_source": (
            "reformulation_action"
            if row.tag == "reformulated from"
            else "compilation_action"
            if row.tag == "compilation"
            else "expansion_action"
        ),
    }


_COMPACT_CATEGORIES = {"and", "or", "implication", "existence"}

# The compiler's spontaneous compact families. Base form (the registry-
# independent spelling of a theorem, `expandToBaseForm` on the prover side)
# unfolds exactly these; every config-defined operator — anchors,
# `NaturalNumbers`, `preorder`, `fold`, … — stays an atom in base form.
_SPONTANEOUS_HEAD = re.compile(r"^(existence|or|implication)\d+$")


def _shape_key(expression: Expression) -> str:
    """
    @brief Name-free structural key of an expression, for canonical ordering.
    @details Every argument token is replaced by its first-occurrence ordinal
    inside the expression, so two parts of one connective sort identically on
    both sides of a comparison whatever their variables are called.
    @param expression Any expression.
    @return MPL text with ordinal arguments.
    """

    ordinals: dict[str, str] = {}

    def ordinal(name: str) -> str:
        return ordinals.setdefault(name, f"#{len(ordinals)}")

    return to_mpl(map_arguments(expression, ordinal))


def base_form(
    expression: Expression,
    definitions: dict[str, dict[str, object]],
    counter: list[int],
) -> Expression:
    """
    @brief Canonical registry-independent form of an expression.
    @details
    Every spontaneous compact (`existence<N>`, `or<N>`, `implication<N>`) is
    unfolded through its compiled definition: `and` becomes a conjunction and
    `or` the negated conjunction of the negated parts, both with parts in
    `_shape_key` order (the connectives commute); an implication compact
    becomes the bound implication chain with binders at their first-use
    element; an existence compact becomes the negated universal
    `¬ ∀ x, e_1 → … → ¬ e_k`. A double negation cancels, so the compiler's
    `¬ existence` and the base spelling's plain universal agree. Definition
    locals get fresh binder names from `counter`; parameters `u_k` take the
    atom's arguments. Config-defined operators stay atoms — the base form of
    the prover keeps them too. The result is for comparison only.
    @param expression Expression in one vocabulary.
    @param definitions That vocabulary's completed binary entries.
    @param counter One-element list, the fresh-binder counter.
    @return The canonical form.
    """

    if isinstance(expression, Negation):
        inner = base_form(expression.inner, definitions, counter)
        if isinstance(inner, Negation):
            return inner.inner
        return Negation(inner)
    if isinstance(expression, Conjunction):
        parts = [base_form(part, definitions, counter) for part in expression.parts]
        return Conjunction(tuple(sorted(parts, key=_shape_key)))
    if isinstance(expression, Implication):
        return Implication(
            expression.bound_variables,
            base_form(expression.premise, definitions, counter),
            base_form(expression.conclusion, definitions, counter),
        )
    assert isinstance(expression, Atom)
    if not _SPONTANEOUS_HEAD.fullmatch(expression.head):
        return expression
    entry = definitions.get(expression.head)
    assert entry is not None and str(entry.get("category")) in _COMPACT_CATEGORIES, (
        f"spontaneous compact {expression.head!r} has no compiled definition to unfold"
    )
    category = str(entry["category"])
    parameters = {f"u_{index + 1}": argument for index, argument in enumerate(expression.arguments)}
    locals_: dict[str, str] = {}
    elements = [parse_mpl(str(element)) for element in entry["elements"]]
    introduced: list[list[str]] = []
    for element in elements:
        here: list[str] = []
        for atom in iter_atoms(element):
            for argument in atom.arguments:
                if not argument.startswith("u_") and argument not in locals_:
                    counter[0] += 1
                    locals_[argument] = f"b{counter[0]}"
                    here.append(locals_[argument])
        introduced.append(here)
    instantiated = [
        base_form(
            map_arguments(element, lambda name: locals_.get(name, parameters.get(name, name))),
            definitions,
            counter,
        )
        for element in elements
    ]
    if category == "and":
        return Conjunction(tuple(sorted(instantiated, key=_shape_key)))
    if category == "or":
        negated = [base_form(Negation(part), definitions, counter) for part in instantiated]
        return Negation(Conjunction(tuple(sorted(negated, key=_shape_key))))
    assert len(instantiated) >= 2
    if category == "implication":
        # A local of a compiled rule is introduced at a premise; a local
        # appearing first in the conclusion would be unbound in the rule.
        assert not introduced[-1], f"implication compact {expression.head} binds a local in its conclusion"
        body = instantiated[-1]
        for index in reversed(range(len(instantiated) - 1)):
            body = Implication(tuple(introduced[index]), instantiated[index], body)
        return body
    assert category == "existence"
    body: Expression = base_form(Negation(instantiated[-1]), definitions, counter)
    for index in reversed(range(len(instantiated) - 1)):
        body = Implication(tuple(introduced[index]), instantiated[index], body)
    return Negation(body)


def base_form_equivalent(
    source: Expression,
    target: Expression,
    source_definitions: dict[str, dict[str, object]],
    target_definitions: dict[str, dict[str, object]],
) -> bool:
    """
    @brief Decide whether two theorem spellings are the same theorem in base form.
    @details
    A fresh compilation may name, order and orient its compact definitions
    differently (the shortcut's `or2[w1,i0,N,s]` is the externals' base
    spelling of "zero or a predecessor"), so both sides are brought to
    `base_form` and compared alpha-equivalently as universally quantified
    Horn propositions with every binder merged into one group and under one
    premise permutation. A negative answer is the defined result a candidate
    search relies on.
    @param source Theorem of the dependency (a certificate theorem or an
        externals row, which is already in base form).
    @param target Theorem spelling cited by the citing proof graph.
    @param source_definitions Completed definitions of the dependency side.
    @param target_definitions Completed definitions of the citing corpus.
    @return True when the base forms are alpha-equivalent under some premise order.
    """

    def flattened(expression: Expression, order: tuple[int, ...] | None) -> Expression:
        """
        @brief One canonical Horn spine: every binder in one top group.
        @details The spine's binder groups are merged into one group ordered
        by first use in the (possibly permuted) premises and the conclusion,
        so a spelling that binds a variable after its first use and one that
        binds it before compare equal; inner implications carry empty groups.
        @param expression Universally quantified Horn theorem.
        @param order Premise permutation, or None for the given order.
        @return The flattened spine.
        """

        bound: set[str] = set()
        premises: list[Expression] = []
        cursor = expression
        while isinstance(cursor, Implication):
            bound.update(cursor.bound_variables)
            premises.append(cursor.premise)
            cursor = cursor.conclusion
        if order is not None:
            premises = [premises[position] for position in order]
        first_use: list[str] = []
        for part in [*premises, cursor]:
            for atom in iter_atoms(part):
                for argument in atom.arguments:
                    if argument in bound and argument not in first_use:
                        first_use.append(argument)
        assert set(first_use) == bound, "a bound variable of the theorem is never used"
        rebuilt = cursor
        for position in reversed(range(len(premises))):
            rebuilt = Implication(
                tuple(first_use) if position == 0 else (),
                premises[position],
                rebuilt,
            )
        return rebuilt

    target_key = alpha_key(base_form(flattened(target, None), target_definitions, [0]))
    premise_count = len(_horn_premises(source))
    if premise_count > 8:
        return False
    for order in permutations(range(premise_count)):
        rebuilt = flattened(source, order)
        if alpha_key(base_form(rebuilt, source_definitions, [0])) == target_key:
            return True
    return False


def _horn_premises(expression: Expression) -> list[Expression]:
    """
    @brief The premises of a universally quantified Horn theorem, in order.
    @param expression The theorem.
    @return Its premises from the outermost implication inwards.
    """

    premises: list[Expression] = []
    cursor = expression
    while isinstance(cursor, Implication):
        premises.append(cursor.premise)
        cursor = cursor.conclusion
    return premises


def theorem_adaptation_solutions(
    source: Expression,
    target: Expression,
    source_definitions: dict[str, dict[str, object]],
    target_definitions: dict[str, dict[str, object]],
    head_aliases: dict[str, str],
) -> list[dict[str, str]]:
    """
    @brief Match one external theorem under binder reordering and definition aliases.
    @details
    The shortcut consumes registry-independent Peano and Gauss facts, but a
    fresh compilation may assign different names to identical compiled
    definitions and may reorder universal premise groups. Every declared head
    alias is checked definition-by-definition. The source and adapted target
    are then matched as universally quantified Horn propositions under one
    bijective variable renaming and one premise permutation.
    @param source The theorem exported by the dependency certificate.
    @param target The exact theorem spelling cited by the shortcut proof graph.
    @param source_definitions Checked dependency-certificate definitions.
    @param target_definitions Checked shortcut-certificate definitions.
    @param head_aliases Shortcut compiled heads mapped to dependency heads.
    @return Every source-to-target bound-variable renaming that matches, one per
        premise permutation; empty when the pair is structurally incompatible
        (an alias whose definitions differ, or a binder or premise count that
        differs) - the negative answer a candidate search relies on.
    @invariant Every returned renaming is a total bijection and preserves every premise.
    """

    def rename_heads(expression: Expression) -> Expression:
        """
        @brief Apply the checked target-to-source compiled-head aliases recursively.
        @details Arguments, binders, connective shape, and unaliased heads remain exact.
        @param expression Target-side expression to rename.
        @return Expression in the dependency certificate's compiled vocabulary.
        """

        if isinstance(expression, Atom):
            return Atom(
                head_aliases.get(expression.head, expression.head),
                expression.arguments,
            )
        if isinstance(expression, Conjunction):
            return Conjunction(tuple(rename_heads(part) for part in expression.parts))
        if isinstance(expression, Negation):
            return Negation(rename_heads(expression.inner))
        assert isinstance(expression, Implication)
        return Implication(
            expression.bound_variables,
            rename_heads(expression.premise),
            rename_heads(expression.conclusion),
        )

    for target_head, source_head in sorted(head_aliases.items()):
        if target_head not in target_definitions or source_head not in source_definitions:
            return []
        target_entry = target_definitions[target_head]
        source_entry = source_definitions[source_head]
        if int(target_entry["arity"]) != int(source_entry["arity"]):
            return []
        if str(target_entry["category"]) != str(source_entry["category"]):
            return []
        renamed_elements = [
            to_mpl(rename_heads(parse_mpl(str(element))))
            for element in target_entry["elements"]
        ]
        if renamed_elements != [str(element) for element in source_entry["elements"]]:
            return []

    adapted_target = rename_heads(target)

    def flatten_horn(
        expression: Expression,
    ) -> tuple[set[str], list[Expression], Expression]:
        """
        @brief Flatten the universal implication spine of one exported theorem.
        @details Binder groups become one set and premises retain source order.
        @param expression Complete exported theorem expression.
        @return Bound-variable set, ordered premises, and terminal conclusion.
        """

        bound: set[str] = set()
        premises: list[Expression] = []
        cursor = expression
        while isinstance(cursor, Implication):
            assert not bound.intersection(cursor.bound_variables)
            bound.update(cursor.bound_variables)
            premises.append(cursor.premise)
            cursor = cursor.conclusion
        return bound, premises, cursor

    source_bound, source_premises, source_conclusion = flatten_horn(source)
    target_bound, target_premises, target_conclusion = flatten_horn(adapted_target)
    if len(source_bound) != len(target_bound):
        return []
    if len(source_premises) != len(target_premises):
        return []

    def align_expression(
        source_expression: Expression,
        target_expression: Expression,
        renaming: dict[str, str],
    ) -> bool:
        """
        @brief Extend one bound-variable renaming across a pair of expressions.
        @details Heads and connective shape must match exactly; free tokens never rename.
        @param source_expression Dependency-side expression.
        @param target_expression Adapted shortcut-side expression.
        @param renaming Mutable candidate source-to-target bound-variable map.
        @return True when the pair is structurally compatible.
        """

        pending = [(source_expression, target_expression)]
        while pending:
            source_part, target_part = pending.pop()
            if type(source_part) is not type(target_part):
                return False
            if isinstance(source_part, Atom):
                assert isinstance(target_part, Atom)
                if (
                    source_part.head != target_part.head
                    or len(source_part.arguments) != len(target_part.arguments)
                ):
                    return False
                for source_name, target_name in zip(
                    source_part.arguments,
                    target_part.arguments,
                    strict=True,
                ):
                    if source_name in source_bound:
                        if target_name not in target_bound:
                            return False
                        existing = renaming.get(source_name)
                        if existing is not None and existing != target_name:
                            return False
                        if existing is None and target_name in renaming.values():
                            return False
                        renaming[source_name] = target_name
                    elif source_name != target_name:
                        return False
            elif isinstance(source_part, Conjunction):
                assert isinstance(target_part, Conjunction)
                if len(source_part.parts) != len(target_part.parts):
                    return False
                pending.extend(zip(source_part.parts, target_part.parts, strict=True))
            elif isinstance(source_part, Negation):
                assert isinstance(target_part, Negation)
                pending.append((source_part.inner, target_part.inner))
            else:
                assert isinstance(source_part, Implication)
                assert isinstance(target_part, Implication)
                if len(source_part.bound_variables) != len(
                    target_part.bound_variables
                ):
                    return False
                pending.extend(
                    (
                        (source_part.premise, target_part.premise),
                        (source_part.conclusion, target_part.conclusion),
                    )
                )
        return True

    solutions: list[dict[str, str]] = []
    for target_order in permutations(target_premises):
        candidate: dict[str, str] = {}
        if not align_expression(source_conclusion, target_conclusion, candidate):
            continue
        if not all(
            align_expression(source_premise, target_premise, candidate)
            for source_premise, target_premise in zip(
                source_premises,
                target_order,
                strict=True,
            )
        ):
            continue
        if set(candidate) != source_bound or set(candidate.values()) != target_bound:
            continue
        solutions.append(candidate)
    return solutions


def _validate_theorem_adaptation(
    source: Expression,
    target: Expression,
    source_definitions: dict[str, dict[str, object]],
    target_definitions: dict[str, dict[str, object]],
    head_aliases: dict[str, str],
    base_form_adaptation: bool = False,
) -> dict[str, str]:
    """
    @brief Validate one declared external theorem adaptation.
    @details
    The declared path of a selection's `theorem_adaptations`: the pair must
    match under the checked aliases, one bijective variable renaming and one
    premise permutation, or the certificate is rejected. The canonical
    renaming is the lexicographically least solution so the certificate bytes
    stay deterministic. A `base_form` adaptation instead requires
    `base_form_equivalent` (registry-independent canonical forms) and records
    no renaming.
    @param base_form_adaptation Whether the declaration is a base-form one.
    @param source The theorem exported by the dependency certificate.
    @param target The exact theorem spelling cited by the shortcut proof graph.
    @param source_definitions Checked dependency-certificate definitions.
    @param target_definitions Checked shortcut-certificate definitions.
    @param head_aliases Shortcut compiled heads mapped to dependency heads.
    @return Source-bound variables mapped to target-bound variables.
    @invariant The returned renaming is a total bijection and preserves every premise.
    """

    if base_form_adaptation:
        assert not head_aliases, "a base-form adaptation declares no head aliases"
        assert base_form_equivalent(source, target, source_definitions, target_definitions), (
            "external theorem adaptation is not base-form equivalent"
        )
        return {}
    solutions = theorem_adaptation_solutions(
        source,
        target,
        source_definitions,
        target_definitions,
        head_aliases,
    )
    assert solutions, "external theorem adaptation has no structural equivalence"
    return min(solutions, key=lambda item: tuple(sorted(item.items())))


def _schema3_action_metadata(
    chapter: Chapter,
    binary_entries: dict[str, dict[str, object]],
    theorem_references: dict[str, list[dict[str, object]]],
    signatures: dict[str, tuple[str, ...]] | dict[str, list[str]],
) -> tuple[dict[int, dict[str, object]], list[dict[str, object]]]:
    """
    @brief Validate scoped schema-3 actions and record their neutral metadata.
    @details
    Schema 3 makes integration and OR-branch scopes plus theorem reformulations explicit. A
    boundary namespace is accepted only when one ``premise element`` template,
    one matching integration expansion, and one ``validity name`` discharge
    agree on the same compiled implication. Integration reformulations must
    end in a compiled definition of the required category.
    An OR branch must be one checked disjunct of its compact definition, and an
    OR convergence must cite that same compact definition at its established
    parent scope, with branch results belonging to that exact parent. OR
    integration proves a selected disjunct under the complements of its peers;
    its parent discharge introduces the compact OR without assuming that goal.
    A subproof or a contradiction scope may be opened inside another such
    scope or inside an OR case: its ``validity name`` discharge or closing
    ``contradiction`` row sits at the scope that opened it, and its context
    carries that parent's layers before its own, so parents are established
    first. One worklist builds every context, subproofs and contradiction
    scopes in depth order and then the OR contexts, and defers a context
    whose parent is not built yet to a later round.
    A ``reformulated
    from`` row must cite one alpha-resolved theorem, end in a
    compiled existence definition, and carry constructive totality evidence
    for its well-defined set or relation output. An integration row that
    states its compact is checked on the rule the compact stands for
    (D-503).
    @param chapter Parsed processed-proof chapter.
    @param binary_entries Exact compiled definitions available to the certificate.
    @param theorem_references Alpha-keyed selected and imported theorem references.
    @param signatures Exact argument types for every compiled definition.
    @return Per-source-line action metadata and the chapter's explicit scope contexts.
    @invariant Every non-main namespace has exactly one checked integration or OR context.
    """

    metadata: dict[int, dict[str, object]] = {}
    premise_templates: dict[str, ScopedExpression] = {}
    premise_template_namespaces: dict[str, str] = {}
    validity_rows: dict[str, ProofRow] = {}
    compact_rules = compact_rule_rows(chapter, binary_entries)

    for row in chapter.rows:
        if row.tag == "premise element":
            assert row.namespace != "main" and len(row.dependencies) == 1
            dependency = row.dependencies[0]
            assert dependency.scoped_expression.proof_scope == INTEGRATION_GOAL_SCOPE
            existing = premise_templates.get(row.namespace)
            assert existing is None or existing == dependency.scoped_expression
            premise_templates[row.namespace] = dependency.scoped_expression
            existing_namespace = premise_template_namespaces.get(row.namespace)
            assert existing_namespace is None or existing_namespace == dependency.namespace
            premise_template_namespaces[row.namespace] = dependency.namespace
        elif row.tag == "validity name":
            assert len(row.dependencies) == 1
            boundary = row.dependencies[0].namespace
            assert boundary != "main" and boundary not in validity_rows
            # A closed subproof's compact lands at the scope that opened it,
            # its direct parent (I-332).
            assert boundary.startswith(row.namespace + "_boundary_")
            assert "_boundary_" not in boundary[len(row.namespace) + len("_boundary_"):]
            validity_rows[boundary] = row

    assert set(premise_templates) == set(validity_rows)
    scope_contexts: list[dict[str, object]] = []
    context_by_namespace: dict[str, dict[str, object]] = {}

    def under_parent(expression: Expression, parent: str) -> Expression:
        """
        @brief Prefix a scope template with the layers of the scope that opened it.
        @details A scope opened at ``main`` keeps its template. A scope opened
        inside an established context states that context's binders and
        premises first, ancestor layers before child layers, as every scoped
        proposition is rendered.
        @param expression The scope's own template.
        @param parent The namespace of the scope that opened it.
        @return The template under the parent's complete layer chain.
        """

        if parent == "main":
            return expression
        assert parent in context_by_namespace, "scope parent is not established"
        cursor = parse_mpl(str(context_by_namespace[parent]["template"]["mpl"]))
        layers = []
        while isinstance(cursor, Implication):
            layers.append((cursor.bound_variables, cursor.premise))
            cursor = cursor.conclusion
        assert layers
        for bound_variables, premise in reversed(layers):
            expression = Implication(bound_variables, premise, expression)
        return expression

    contradiction_seeds: dict[str, ProofRow] = {}
    known_scope_namespaces = set(premise_templates) | {row.namespace for row in chapter.rows if row.tag == "or disintegration"}
    for row in chapter.rows:
        if (
            row.tag != "task formulation"
            or row.namespace == "main"
            or row.namespace in known_scope_namespaces
        ):
            continue
        assert not row.dependencies
        assert row.namespace not in contradiction_seeds
        contradiction_seeds[row.namespace] = row

    # Parents before children: the contradiction-scope demand opens a
    # subproof and its reductio inside a contradiction scope.
    build_order = sorted(
        [(namespace.count("_boundary_"), 0, position, namespace)
         for position, namespace in enumerate(sorted(premise_templates))]
        + [(namespace.count("_boundary_"), 1, position, namespace)
           for position, namespace in enumerate(contradiction_seeds)]
    )
    contradiction_assumptions: dict[str, ProofRow] = {}
    or_branch_rows: dict[str, ProofRow] = {}
    or_integration_rows: dict[str, ProofRow] = {}

    def build_scope_context(kind: int, namespace: str) -> None:
        """
        @brief Establish the context of one subproof or contradiction scope.
        @details The worklist below calls it once the context of the scope
        that opened it is established.
        @param kind 1 for a contradiction scope, 0 for an implication subproof.
        @param namespace The scope's namespace.
        @return Nothing; the context and its rows' metadata are recorded.
        """
        if kind == 1:
            row = contradiction_seeds[namespace]
            cut = namespace.rfind("_boundary_")
            assert cut > 0
            parent = namespace[:cut]
            assert namespace[cut + len("_boundary_"):].startswith("contradiction_")
            contradiction_rows = [
                candidate
                for candidate in chapter.rows
                if (
                    candidate.tag == "contradiction"
                    and candidate.namespace == parent
                    and any(
                        dependency.namespace == namespace
                        for dependency in candidate.dependencies
                    )
                )
            ]
            assert len(contradiction_rows) == 1
            assert contradiction_rows[0].conclusion.expression == Negation(
                row.conclusion.expression
            )
            template_expression = under_parent(Implication(
                (),
                row.conclusion.expression,
                row.conclusion.expression,
            ), parent)
            template = ScopedExpression(
                template_expression,
                to_mpl(template_expression),
                row.conclusion.proof_scope,
            )
            scope_context = {
                "namespace": namespace,
                "compact": _scoped_expression_json(row.conclusion),
                "template": _scoped_expression_json(template),
                "scope_kind": "contradiction_assumption",
            }
            if parent != "main":
                scope_context["parent_namespace"] = parent
            scope_contexts.append(scope_context)
            context_by_namespace[namespace] = scope_context
            metadata[row.source_line] = {"scope_namespace": namespace}
            contradiction_assumptions[namespace] = row
            return

        template = premise_templates[namespace]
        validity = validity_rows[namespace]
        parent = validity.namespace
        assert premise_template_namespaces[namespace] == parent, (
            "subproof template is not cited at the scope that opened it")
        compact = validity.conclusion.expression
        assert isinstance(compact, Atom)
        assert compact.head in binary_entries
        assert binary_entries[compact.head].get("category") == "implication"

        cursor = template.expression
        premises: list[Expression] = []
        while isinstance(cursor, Implication):
            premises.append(cursor.premise)
            cursor = cursor.conclusion
        assert validity.dependencies[0].expression == cursor

        matching_expansions = [
            row
            for row in chapter.rows
            if (
                row.tag == "expansion for integration"
                and len(row.dependencies) == 1
                and row.dependencies[0].scoped_expression.proof_scope
                == INTEGRATION_GOAL_SCOPE
                and row.conclusion.proof_scope == INTEGRATION_GOAL_SCOPE
                and row.dependencies[0].expression == compact
                and row.conclusion.expression == template.expression
            )
        ]
        assert len(matching_expansions) == 1
        if parent != "main":
            context_expression = under_parent(template.expression, parent)
            template = ScopedExpression(
                context_expression,
                to_mpl(context_expression),
                template.proof_scope,
            )
        scope_context = {
            "namespace": namespace,
            "compact": _scoped_expression_json(validity.conclusion),
            "template": _scoped_expression_json(template),
            "definition_head": compact.head,
        }
        if parent != "main":
            scope_context["parent_namespace"] = parent
        scope_contexts.append(scope_context)
        context_by_namespace[namespace] = scope_context
        metadata[validity.source_line] = {"scope_context": scope_context}

        for row in chapter.rows:
            if row.tag != "premise element" or row.namespace != namespace:
                continue
            assert row.conclusion.expression in premises
            metadata[row.source_line] = {"scope_namespace": namespace}

    def build_or_context(row: ProofRow) -> None:
        """
        @brief Establish the context of one OR case or OR integration scope.
        @details The worklist below calls it once the context of the scope
        that opened it is established.
        @param row The scope's ``or disintegration`` or ``or branch proven`` row.
        @return Nothing; the context and its rows' metadata are recorded.
        """
        if row.tag == "or branch proven":
            assert len(row.dependencies) == 1
            assert row.namespace == "main" or row.namespace in context_by_namespace
            compact = row.conclusion.expression
            assert isinstance(compact, Atom) and compact.head in binary_entries
            definition = binary_entries[compact.head]
            assert definition["category"] == "or"
            parameter_map = {f"u_{index + 1}": argument for index, argument in enumerate(compact.arguments)}
            disjuncts = [map_arguments(parse_mpl(str(element)), lambda name: parameter_map.get(name, name))
                        for element in definition["elements"]]
            selected = row.dependencies[0].expression
            assert len(disjuncts) >= 2 and disjuncts.count(selected) == 1
            selected_case = disjuncts.index(selected)
            namespace = row.dependencies[0].namespace
            assert namespace == row.namespace + "_boundary_orint_" + to_mpl(compact) + "_(" + to_mpl(selected) + ")"
            assert namespace not in context_by_namespace
            peer_assumptions = [element.inner if isinstance(element, Negation) else Negation(element)
                                for index, element in enumerate(disjuncts) if index != selected_case]
            template_expression = selected
            for premise in reversed(peer_assumptions):
                template_expression = Implication((), premise, template_expression)
            if row.namespace != "main":
                parent_template = parse_mpl(context_by_namespace[row.namespace]["template"]["mpl"])
                parent_layers = []
                while isinstance(parent_template, Implication):
                    parent_layers.append((parent_template.bound_variables, parent_template.premise))
                    parent_template = parent_template.conclusion
                assert parent_layers
                for bound_variables, premise in reversed(parent_layers):
                    template_expression = Implication(bound_variables, premise, template_expression)
            scope_context = {
                "namespace": namespace,
                "compact": _scoped_expression_json(row.conclusion),
                "template": expression_json(template_expression),
                "definition_head": compact.head,
                "scope_kind": "or_integration",
                "parent_namespace": row.namespace,
                "selected_case": selected_case,
            }
            scope_contexts.append(scope_context)
            context_by_namespace[namespace] = scope_context
            or_integration_rows[namespace] = row
            metadata[row.source_line] = {
                "or_definition_head": compact.head, "branch_namespace": namespace,
                "selected_case": selected_case,
            }
            for assumption in chapter.rows:
                if assumption.tag != "or branch assumption" or assumption.namespace != namespace:
                    continue
                assert len(assumption.dependencies) == 1
                dependency = assumption.dependencies[0]
                assert dependency.namespace == row.namespace and dependency.expression == compact
                assert dependency.scoped_expression.proof_scope == INTEGRATION_GOAL_SCOPE
                assert assumption.conclusion.expression in peer_assumptions
                metadata[assumption.source_line] = {"scope_namespace": namespace}
            return
        assert row.namespace != "main" and len(row.dependencies) == 1
        assert row.namespace not in premise_templates
        assert row.namespace not in or_branch_rows
        dependency = row.dependencies[0]
        assert row.namespace.startswith(dependency.namespace + "_boundary_")
        assert dependency.namespace == "main" or dependency.namespace in context_by_namespace, (
            "OR parent scope is not established")
        compact = dependency.expression
        # A compiled disjunction, or a negated conjunction integration
        # premise whose cases are its complemented conjuncts
        # (D-460).
        from .or_convergence import case_split_cases
        source_atom, disjuncts = case_split_cases(compact, binary_entries)
        assert disjuncts.count(row.conclusion.expression) == 1
        template_expression = Implication(
            (),
            row.conclusion.expression,
            row.conclusion.expression,
        )
        if dependency.namespace != "main":
            parent_template = parse_mpl(context_by_namespace[dependency.namespace]["template"]["mpl"])
            parent_layers = []
            while isinstance(parent_template, Implication):
                parent_layers.append((parent_template.bound_variables, parent_template.premise))
                parent_template = parent_template.conclusion
            assert parent_layers
            for bound_variables, premise in reversed(parent_layers):
                template_expression = Implication(bound_variables, premise, template_expression)
        template = ScopedExpression(
            template_expression,
            to_mpl(template_expression),
            row.conclusion.proof_scope,
        )
        scope_context = {
            "namespace": row.namespace,
            "compact": _scoped_expression_json(dependency.scoped_expression),
            "template": _scoped_expression_json(template),
            "definition_head": source_atom.head,
            "scope_kind": "or_branch",
            "parent_namespace": dependency.namespace,
        }
        scope_contexts.append(scope_context)
        context_by_namespace[row.namespace] = scope_context
        metadata[row.source_line] = {"scope_namespace": row.namespace}
        or_branch_rows[row.namespace] = row

    def opening_scope(item: tuple) -> str:
        """
        @brief The namespace of the scope that opens a scheduled context.
        @param item ``("scope", kind, namespace)`` or ``("or", row)``.
        @return That namespace, ``main`` for a scope opened at the root.
        """
        if item[0] == "scope":
            _, kind, namespace = item
            if kind == 1:
                cut = namespace.rfind("_boundary_")
                assert cut > 0
                return namespace[:cut]
            return validity_rows[namespace].namespace
        row = item[1]
        if row.tag == "or branch proven":
            return row.namespace
        assert row.dependencies
        return row.dependencies[0].namespace

    # Every context is built after the context of the scope that opened it:
    # subproofs and contradiction scopes in depth order, then OR cases and OR
    # integrations by name length. A context whose parent is not built yet
    # waits for a later round, as a subproof opened inside an OR case does;
    # a chapter without such nesting is built in this order in one round.
    pending = (
        [("scope", kind, namespace) for _, kind, _, namespace in build_order]
        + [("or", row) for row in sorted(
            (item for item in chapter.rows if item.tag in {"or disintegration", "or branch proven"}),
            key=lambda item: (
                len(item.namespace if item.tag == "or disintegration" else item.dependencies[0].namespace),
                item.source_line,
            ),
        )]
    )
    while pending:
        waiting = []
        for item in pending:
            parent = opening_scope(item)
            if parent != "main" and parent not in context_by_namespace:
                waiting.append(item)
            elif item[0] == "scope":
                build_scope_context(item[1], item[2])
            else:
                build_or_context(item[1])
        assert len(waiting) < len(pending), "parent scope is not established"
        pending = waiting

    for row in chapter.rows:
        if row.tag != "or convergence":
            continue
        assert len(row.dependencies) >= 2
        assert row.namespace == "main" or row.namespace in context_by_namespace, (
            "OR convergence parent scope is not established"
        )
        compact_dependency = row.dependencies[0]
        compact = compact_dependency.expression
        assert compact_dependency.namespace == row.namespace, (
            "OR convergence must cite its OR at the exact parent scope"
        )
        from .or_convergence import case_split_cases
        source_atom, _ = case_split_cases(compact, binary_entries)
        branch_dependencies = [
            dependency
            for dependency in row.dependencies[1:]
            if dependency.namespace != "main" and not (
                dependency.namespace == row.namespace
                or row.namespace.startswith(dependency.namespace + "_boundary_")
            )
        ]
        assert branch_dependencies
        assert all(
            dependency.namespace in or_branch_rows
            and or_branch_rows[dependency.namespace].dependencies[0].expression
            == compact
            and or_branch_rows[dependency.namespace].dependencies[0].namespace == row.namespace
            for dependency in branch_dependencies
        )
        metadata[row.source_line] = {"or_definition_head": source_atom.head}
        from .or_convergence import convergence_cases, match_equality_convergence
        cases = convergence_cases(compact, row.namespace, binary_entries)
        plan = match_equality_convergence(row.conclusion.expression, row.namespace, cases,
            [(dependency.expression, dependency.namespace) for dependency in row.dependencies[1:]])
        if plan is not None:
            metadata[row.source_line]["or_equality_plan"] = plan
        else:
            assert len(row.dependencies) == len(cases) + 1, "unchecked OR equality support"

    boundary_namespaces = {
        namespace
        for row in chapter.rows
        for namespace in (
            row.namespace,
            *(dependency.namespace for dependency in row.dependencies),
        )
        if namespace != "main"
    }
    assert boundary_namespaces == (
        set(premise_templates)
        | set(or_branch_rows)
        | set(or_integration_rows)
        | set(contradiction_assumptions)
    )

    for row in chapter.rows:
        if row.tag == "reformulated from":
            assert len(row.dependencies) == 1
            matches = theorem_references.get(
                alpha_key(row.dependencies[0].expression),
                [],
            )
            assert len(matches) == 1
            cursor = row.conclusion.expression
            while isinstance(cursor, Implication):
                cursor = cursor.conclusion
            assert isinstance(cursor, Atom) and cursor.head in binary_entries
            assert binary_entries[cursor.head].get("category") == "existence"
            existence_entry = binary_entries[cursor.head]
            existence_elements = existence_entry["elements"]
            assert isinstance(existence_elements, list) and len(existence_elements) == 2
            defining_template = parse_mpl(str(existence_elements[0]))
            assert isinstance(defining_template, Atom)
            defining_expression = map_arguments(
                defining_template,
                lambda name: (
                    cursor.arguments[int(name[2:]) - 1]
                    if name.startswith("u_")
                    else name
                ),
            )
            assert isinstance(defining_expression, Atom)
            totality = _defined_output_totality(
                defining_expression.head,
                binary_entries,
                signatures,
            )
            output_argument = int(totality["output_argument"])
            assert defining_expression.arguments[output_argument - 1] in set(
                totality["bound_variables"]
            )
            parameter_map = {
                f"u_{index + 1}": argument
                for index, argument in enumerate(defining_expression.arguments)
            }
            instantiated_elements = [
                map_arguments(
                    parse_mpl(str(element)),
                    lambda name: parameter_map.get(name, name),
                )
                for element in totality["witness_elements"]
            ]
            metadata[row.source_line] = {
                "reformulation_definition": cursor.head,
                "defined_output_totality": {
                    **totality,
                    "defining_expression": expression_json(defining_expression),
                    "witness_elements": [
                        expression_json(element) for element in instantiated_elements
                    ],
                },
            }
        elif row.tag == "reformulation for integration flattened":
            assert len(row.dependencies) == 1
            dependency = row.dependencies[0]
            assert dependency.namespace == row.namespace
            assert dependency.scoped_expression.proof_scope == INTEGRATION_GOAL_SCOPE
            compact = dependency.expression
            assert isinstance(compact, Atom)
            rule = (compact_rules[row.source_line]["rule"]
                    if row.source_line in compact_rules else row.conclusion.expression)
            metadata[row.source_line] = {
                "reformulation_definition": compact.head,
                "flat_integration_plan": build_flat_integration_plan(rule, compact, binary_entries),
            }
        elif row.tag in {
            "reformulation for integration >[bound]",
            "reformulation for integration >[]",
            "reformulation for integration and",
        }:
            assert len(row.dependencies) == 1
            cursor = (compact_rules[row.source_line]["rule"]
                      if row.source_line in compact_rules else row.conclusion.expression)
            while isinstance(cursor, Implication):
                cursor = cursor.conclusion
            if (
                isinstance(cursor, Atom)
                and cursor.head == EQUALITY
                and row.tag == "reformulation for integration and"
            ):
                # A set- or relation-typed `=` integrates through the
                # conjunction of its two subset implications.
                subsets = row.dependencies[0].expression
                assert isinstance(subsets, Conjunction) and len(subsets.parts) == 2
                left, right = cursor.arguments
                forward, backward = subsets.parts
                assert isinstance(forward, Atom) and isinstance(backward, Atom)
                assert forward.head == backward.head
                assert forward.arguments == (left, right)
                assert backward.arguments == (right, left)
                assert is_subset_implication(binary_entries[forward.head])
                metadata[row.source_line] = {
                    "reformulation_definition": EQUALITY,
                }
                continue
            assert isinstance(cursor, Atom) and cursor.head in binary_entries
            expected_category = (
                "existence"
                if row.tag
                in {
                    "reformulation for integration >[bound]",
                    "reformulation for integration >[]",
                }
                else "and"
            )
            assert binary_entries[cursor.head].get("category") == expected_category
            if expected_category == "and":
                assert isinstance(row.dependencies[0].expression, Conjunction)
            metadata[row.source_line] = {
                "reformulation_definition": cursor.head,
            }

    expected_lines = {
        row.source_line
        for row in chapter.rows
        if row.tag in {
            "premise element",
            "or convergence",
            "or disintegration",
            "or branch assumption",
            "or branch proven",
            "reformulated from",
            "reformulation for integration >[bound]",
            "reformulation for integration >[]",
            "reformulation for integration and",
            "reformulation for integration flattened",
            "validity name",
        }
    }
    expected_lines.update(
        row.source_line for row in contradiction_assumptions.values()
    )
    assert set(metadata) == expected_lines
    return metadata, scope_contexts


def _membership_body_key(
    expression: Expression,
    definitions: dict[str, dict[str, object]],
    unfolding: tuple[str, ...] = (),
) -> tuple:
    """
    @brief Compare membership bodies through exact connective definitions.
    @details Conjunction and OR compacts are unfolded with their actual ports.
    Order, polarity and all opaque predicates remain exact; no commutation or
    logical search is performed. Recursive connective definitions assert.
    @param expression Membership body or one of its constituent expressions.
    @param definitions Compiled definitions that give connective bodies.
    @param unfolding Heads on the current recursive expansion path.
    @return Immutable structural key preserving the connective tree.
    @invariant Equal keys justify the same Lean proposition by definition unfolding.
    """

    if isinstance(expression, Negation):
        return ("not", _membership_body_key(expression.inner, definitions, unfolding))
    if isinstance(expression, Conjunction):
        parts = [_membership_body_key(part, definitions, unfolding) for part in expression.parts]
        key = parts[-1]
        for part in reversed(parts[:-1]):
            key = ("and", part, key)
        return key
    if isinstance(expression, Implication):
        return ("implies", expression.bound_variables,
                _membership_body_key(expression.premise, definitions, unfolding),
                _membership_body_key(expression.conclusion, definitions, unfolding))
    assert isinstance(expression, Atom)
    if expression.head not in definitions or definitions[expression.head]["category"] not in {"and", "or"}:
        return ("atom", expression.head, expression.arguments)
    assert expression.head not in unfolding, "recursive membership connective"
    definition = definitions[expression.head]
    assert len(expression.arguments) == int(definition["arity"])
    parameters = {f"u_{index + 1}": argument for index, argument in enumerate(expression.arguments)}
    parts = [parse_mpl(str(element)) for element in definition["elements"]]
    assert len(parts) >= 2
    assert all(argument in parameters for part in parts for atom in iter_atoms(part) for argument in atom.arguments)
    keys = [_membership_body_key(map_arguments(part, parameters.__getitem__), definitions, (*unfolding, expression.head))
            for part in parts]
    key = keys[0]
    for part in keys[1:]:
        key = (definition["category"], key, part)
    return key


def _defined_output_totality(
    definition_head: str,
    binary_entries: dict[str, dict[str, object]],
    signatures: dict[str, tuple[str, ...]] | dict[str, list[str]],
) -> dict[str, object]:
    """
    @brief Certify that one GL definition has a constructive output witness.
    @details
    A well-defined set or relation is represented by forward membership rules
    from the output into a predicate body and one reverse rule from that checked
    body back into output membership. The function expands the compiled
    implication layer and compares connective aliases structurally. Positive
    conjunction swaps carry an explicit transport plan; OR order, grouping,
    leaves and polarity remain exact. It proves the definition graph acyclic, rejects
    every other occurrence of the output parameter, and records the predicate
    whose Lean comprehension or lambda abstraction is the witness.
    @param definition_head Compiled GL definition whose output must exist.
    @param binary_entries Exact compiled definitions available to the certificate.
    @param signatures Exact argument types for every compiled definition.
    @return Backend-neutral totality evidence for one set or binary relation output.
    @invariant No recursive definition or one-directional membership contract is accepted.
    """

    from .membership_transport import membership_transport_plan

    assert definition_head in binary_entries
    definition = binary_entries[definition_head]
    assert definition.get("category") == "and"
    signature = tuple(str(item) for item in signatures[definition_head])
    assert len(signature) == int(definition["arity"])

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(head: str) -> None:
        """
        @brief Assert acyclicity of the compiled definition dependency closure.
        @details Atomic or already checked heads terminate the depth-first walk.
        @param head Current compiled definition head.
        @return None; visiting and visited sets record the proof.
        """

        if head not in binary_entries or head in visited:
            return
        assert head not in visiting, f"recursive compiled definition {head!r}"
        visiting.add(head)
        entry = binary_entries[head]
        if entry.get("category") != "atomic":
            for source_element in entry.get("elements", []):
                assert str(source_element)
                for atom in iter_atoms(parse_mpl(str(source_element))):
                    if atom.head in binary_entries:
                        visit(atom.head)
        visiting.remove(head)
        visited.add(head)

    visit(definition_head)

    clauses: list[tuple[list[Expression], Expression]] = []
    unfold_definitions = [definition_head]
    for source_element in definition["elements"]:
        compact = parse_mpl(str(source_element))
        assert isinstance(compact, Atom) and compact.head in binary_entries
        implication = binary_entries[compact.head]
        assert implication.get("category") == "implication"
        replacements = {
            f"u_{index + 1}": argument
            for index, argument in enumerate(compact.arguments)
        }
        parts = [
            map_arguments(
                parse_mpl(str(element)),
                lambda name: replacements.get(name, name),
            )
            for element in implication["elements"]
        ]
        assert len(parts) >= 2
        clauses.append((parts[:-1], parts[-1]))
        unfold_definitions.append(compact.head)

    candidates: list[dict[str, object]] = []
    for output_index, output_type in enumerate(signature):
        if output_type not in {"set", "binary_relation"}:
            continue
        output_parameter = f"u_{output_index + 1}"
        membership_head = "in" if output_type == "set" else "in2"
        membership_arity = 2 if output_type == "set" else 3

        def is_output_membership(expression: Expression) -> bool:
            """
            @brief Recognize membership in the candidate output parameter.
            @details The atomic head and arity follow the candidate output type.
            @param expression Clause fragment to inspect.
            @return True exactly for membership in the candidate output.
            """

            return (
                isinstance(expression, Atom)
                and expression.head == membership_head
                and len(expression.arguments) == membership_arity
                and expression.arguments[-1] == output_parameter
            )

        occurrences = [
            atom
            for premises, conclusion in clauses
            for expression in [*premises, conclusion]
            for atom in iter_atoms(expression)
            if output_parameter in atom.arguments
        ]
        if not occurrences or not all(is_output_membership(atom) for atom in occurrences):
            continue
        membership = occurrences[0]
        if any(atom != membership for atom in occurrences):
            continue
        forward = [
            conclusion
            for premises, conclusion in clauses
            if len(premises) == 1 and premises[0] == membership
        ]
        backward = [
            premises
            for premises, conclusion in clauses
            if conclusion == membership
        ]
        if len(backward) != 1 or not forward or len(backward[0]) != len(forward):
            continue
        transports = [membership_transport_plan(
            _membership_body_key(source, binary_entries),
            _membership_body_key(target, binary_entries),
        ) for source, target in zip(backward[0], forward)]
        if any(plan is None for plan in transports):
            continue
        if len(forward) + 1 != len(clauses):
            continue
        bound_variables = list(membership.arguments[:-1])
        assert len(bound_variables) == (1 if output_type == "set" else 2)
        assert all(name.isdigit() for name in bound_variables)
        candidates.append(
            {
                "definition_head": definition_head,
                "output_argument": output_index + 1,
                "output_type": output_type,
                "bound_variables": bound_variables,
                "witness_elements": [to_mpl(element) for element in forward],
                "unfold_definitions": unfold_definitions,
                "definition_dependency_acyclic": True,
            }
        )
        if any(plan["kind"] != "identity" for plan in transports):
            candidates[-1]["reverse_membership_transport"] = transports

    assert len(candidates) == 1, (
        f"definition {definition_head!r} does not have one constructive defined output"
    )
    return candidates[0]


def theorem_citations(theorem: TheoremRecord) -> list[Expression]:
    """
    @brief List every expression a theorem's chapters cite from outside themselves.
    @details The expressions of `theorem_citations_with_tags`, without the
    tag of the citing row.
    @param theorem Theorem record whose chapters are already parsed.
    @return Cited expressions in chapter and row order, duplicates kept.
    """

    return [expression for _, expression in theorem_citations_with_tags(theorem)]


def theorem_citations_with_tags(theorem: TheoremRecord) -> list[tuple[str, Expression]]:
    """
    @brief List every citation of a theorem's chapters with the citing row's tag.
    @details
    The schema-3 citation contract: a `theorem` row cites its conclusion, a
    `compilation` row cites its single dependency, and every other row cites
    each dependency that no row of the same chapter produces (alpha-normalized
    conclusion, namespace and proof scope). Citations that resolve to a theorem
    of the same list are dependency edges; the rest must be supplied by a
    dependency certificate — except an `or theorem` row's citation of its
    head-switched companion, which the chapter builder drops when unresolvable
    (D-217: the companion is an alpha-permutation of the proved parent and may
    never have been proved separately). Order follows the chapters and rows.
    @param theorem Theorem record whose chapters are already parsed.
    @return (row tag, cited expression) pairs in chapter and row order.
    """

    cited_expressions: list[tuple[str, Expression]] = []
    for chapter in theorem.chapters:
        produced = {
            _alpha_expression_key(row.conclusion, row.namespace)
            for row in chapter.rows
        }
        for row in chapter.rows:
            if row.tag == "theorem":
                cited_expressions.append((row.tag, row.conclusion.expression))
            elif row.tag == "compilation":
                assert len(row.dependencies) == 1
                cited_expressions.append((row.tag, row.dependencies[0].expression))
            for dependency in row.dependencies:
                # A scoped dependency (an integration-goal template) is a
                # scope of this very chapter, resolved by the chapter
                # certificate from its rows; it never names a theorem.
                if (
                    row.tag != "compilation"
                    and dependency.scoped_expression.proof_scope == ORDINARY_SCOPE
                    and
                    _alpha_expression_key(
                        dependency.scoped_expression,
                        dependency.namespace,
                    )
                    not in produced
                ):
                    cited_expressions.append((row.tag, dependency.expression))
    return cited_expressions


def _citation_source_indices(
    selection: dict[str, object],
    theorems: list[TheoremRecord],
) -> set[int]:
    """
    @brief Validate the selected proofs authorized to resolve theorem citations.
    @details An explicit citation list names one selected source per alpha-equivalent
    proposition while every selected proof remains in the exported corpus. Without
    this field all selected sources remain authorized and ambiguity still asserts.
    @param selection Parsed certificate selection.
    @param theorems Complete selected theorem records, including equivalent copies.
    @return Selected source indices authorized as citation references.
    @invariant Explicit choices cover every selected proposition exactly once and
    cannot authorize an unselected source.
    """

    by_index = {theorem.source_index: alpha_key(theorem.expression) for theorem in theorems}
    explicit = selection.get("citation_source_indices")
    if explicit is None:
        return set(by_index)
    assert isinstance(explicit, list)
    assert all(type(index) is int for index in explicit)
    chosen = set(explicit)
    assert len(chosen) == len(explicit), "duplicate citation source index"
    assert chosen <= set(by_index), "citation source is not selected"
    chosen_keys = {by_index[index] for index in chosen}
    assert len(chosen_keys) == len(chosen), "citation sources are alpha-ambiguous"
    assert chosen_keys == set(by_index.values()), "citation sources omit a selected proposition"
    return chosen


def selected_source_indices(selection, records):
    """
    @brief Validate exact source membership for a range or explicit selection.

    @details The two representations are mutually exclusive. An optional
    complete-anchor contract requires every theorem of that anchor, with no
    exclusions. Existing range fixtures retain their precise old bounds.

    @param selection Schema-two or schema-three selection document.
    @param records Complete source theorem registry.
    @return Sorted nonempty selected source indices after declared exclusions.
    @invariant No duplicate, missing complete-anchor, or out-of-range source passes.
    """
    explicit = selection.get("source_indices")
    source_range = selection.get("source_range")
    assert (explicit is None) != (source_range is None), "select indices or range, exclusively"
    if explicit is not None:
        assert isinstance(explicit, list) and explicit
        assert all(type(index) is int for index in explicit)
        assert explicit == sorted(set(explicit)), "indices must be sorted and unique"
        assert 0 <= explicit[0] <= explicit[-1] < len(records), "source index out of range"
        indices = explicit
    else:
        assert isinstance(source_range, dict)
        first, last = int(source_range["first"]), int(source_range["last"])
        assert 0 <= first <= last < len(records)
        indices = list(range(first, last + 1))
    exclusions = selection.get("excluded_sources")
    assert isinstance(exclusions, list)
    excluded = [int(item["source_index"]) for item in exclusions]
    assert len(excluded) == len(set(excluded))
    assert set(excluded) <= set(indices)
    selected = [index for index in indices if index not in set(excluded)]
    assert selected, "empty source selection"
    if "complete_anchor" in selection:
        anchor = selection["complete_anchor"]
        assert anchor in {"AnchorPeano", "AnchorGauss", "AnchorFTA"}
        assert not excluded, "complete anchor cannot exclude sources"
        expected = [index for index, record in enumerate(records)
                    if isinstance(record.expression, Implication)
                    and isinstance(record.expression.premise, Atom)
                    and record.expression.premise.head == anchor]
        assert selected == expected, "complete-anchor membership mismatch"
    return selected


def _ordered_selected_theorems(
    selection: dict[str, object],
    records: tuple[TheoremRecord, ...],
) -> tuple[list[tuple[dict[str, object], TheoremRecord]], dict[int, set[int]]]:
    """
    @brief Resolve the selected theorem corpus and topologically order its dependencies.
    @details
    Schema 1 keeps the explicit pilot list. Schemas 2 and 3 select an exact
    index list or inclusive range minus explicit exclusions and derive stable theorem
    identifiers. Schema 3 also admits exact dependencies supplied by a hashed
    certificate. A cycle is rejected before any certificate action is emitted.
    @param selection Parsed selection document.
    @param records Complete processed theorem registry.
    @return Topologically ordered selection-record pairs and source-index edges.
    @invariant Every theorem citation resolves alpha-equivalently to exactly one selection.
    """

    schema_version = int(selection.get("schema_version", 0))
    if schema_version == 1:
        raw_selections = selection.get("theorems")
        assert isinstance(raw_selections, list) and raw_selections
    else:
        assert schema_version in {2, 3}
        indices = selected_source_indices(selection, records)
        identifier_prefix = (
            "peano_source"
            if schema_version == 2
            else str(selection["theorem_id_prefix"])
        )
        name_prefix = (
            "Peano source"
            if schema_version == 2
            else str(selection["theorem_name_prefix"])
        )
        raw_selections = [
            {
                "id": f"{identifier_prefix}_{index:03d}",
                "name": f"{name_prefix} {index}",
                "method": records[index].method,
                "source_index": index,
                "theorem": to_mpl(records[index].expression),
            }
            for index in indices
        ]

    resolved: list[tuple[dict[str, object], TheoremRecord]] = []
    selected_ids: set[str] = set()
    for raw_selected in raw_selections:
        assert isinstance(raw_selected, dict)
        selected = dict(raw_selected)
        selected_id = str(selected["id"])
        assert selected_id not in selected_ids, f"duplicate selection id {selected_id!r}"
        selected_ids.add(selected_id)
        source_index = (
            int(selected["source_index"])
            if "source_index" in selected
            else None
        )
        resolved.append(
            (
                selected,
                find_theorem(
                    records,
                    str(selected["theorem"]),
                    str(selected["method"]),
                    source_index,
                ),
            )
        )

    citation_indices = _citation_source_indices(selection, [theorem for _, theorem in resolved])
    selected_by_alpha: dict[str, list[TheoremRecord]] = {}
    for _, theorem in resolved:
        if theorem.source_index in citation_indices:
            selected_by_alpha.setdefault(alpha_key(theorem.expression), []).append(theorem)
    dependencies: dict[int, set[int]] = {
        theorem.source_index: set() for _, theorem in resolved
    }
    if schema_version in {1, 2}:
        for _, theorem in resolved:
            for chapter in theorem.chapters:
                for row in chapter.rows:
                    cited_expressions = []
                    if row.tag == "theorem":
                        cited_expressions.append(row.conclusion.expression)
                    elif row.tag in {"or theorem", "compilation"}:
                        cited_expressions.extend(
                            dependency.expression for dependency in row.dependencies
                        )
                    for position, cited in enumerate(cited_expressions):
                        matches = selected_by_alpha.get(alpha_key(cited), [])
                        if row.tag == "compilation" and not matches:
                            # A locally constructed compilation has no selected
                            # theorem reference; chapter resolution checks its producer.
                            continue
                        if row.tag == "or theorem" and position == 1 and not matches:
                            # Single-direction or licensing (D-217): the head-switch
                            # companion need not be a registered theorem (one proved
                            # after its in-run or theorem is withheld), and the
                            # chapter builder drops its citation; so does the order.
                            continue
                        assert len(matches) == 1, (
                            f"source {theorem.source_index} citation resolves to "
                            f"{len(matches)} selected theorems"
                        )
                        dependencies[theorem.source_index].add(matches[0].source_index)
    else:
        all_by_alpha: dict[str, list[TheoremRecord]] = {}
        for theorem in records:
            all_by_alpha.setdefault(alpha_key(theorem.expression), []).append(theorem)
        selected_indices = {theorem.source_index for _, theorem in resolved}
        dependency_specs = selection.get("certificate_dependencies")
        assert isinstance(dependency_specs, list)
        imported_indices = {
            int(source_index)
            for dependency in dependency_specs
            if dependency.get("theorem_list_scope", "same") == "same"
            for source_index in dependency["required_source_indices"]
        }
        assert not selected_indices.intersection(imported_indices)
        for _, theorem in resolved:
            for cited in theorem_citations(theorem):
                matches = all_by_alpha.get(alpha_key(cited), [])
                if not matches:
                    continue
                matches = [match for match in matches
                           if match.source_index in citation_indices | imported_indices]
                assert matches, (
                    f"source {theorem.source_index} cites an unselected theorem")
                assert len(matches) == 1, (
                    f"source {theorem.source_index} citation is alpha-ambiguous"
                )
                cited_index = matches[0].source_index
                assert cited_index in selected_indices | imported_indices, (
                    f"source {theorem.source_index} cites unselected theorem "
                    f"source {cited_index}"
                )
                dependencies[theorem.source_index].add(cited_index)
        observed_same_list_imports = {
            cited
            for cited_set in dependencies.values()
            for cited in cited_set
            if cited not in selected_indices
        }
        assert observed_same_list_imports == imported_indices

    rank = {
        theorem.source_index: index
        for index, (_, theorem) in enumerate(resolved)
    }
    selected_indices = set(dependencies)
    remaining = {
        source_index: set(cited).intersection(selected_indices)
        for source_index, cited in dependencies.items()
    }
    ordered_indices: list[int] = []
    while remaining:
        ready = sorted(
            (source_index for source_index, cited in remaining.items() if not cited),
            key=rank.__getitem__,
        )
        assert ready, (
            "selected theorem-dependency graph is cyclic: "
            f"{dict(sorted(remaining.items()))!r}"
        )
        ordered_indices.extend(ready)
        for source_index in ready:
            remaining.pop(source_index)
        for cited in remaining.values():
            cited.difference_update(ready)
    by_index = {theorem.source_index: pair for pair in resolved for theorem in [pair[1]]}
    return [by_index[index] for index in ordered_indices], dependencies


def _chapter_certificate(
    chapter: Chapter,
    signatures: dict[str, tuple[str, ...]],
    selected_theorems: dict[str, list[dict[str, object]]],
    constructed_or_definitions: dict[str, dict[str, object]],
    certificate_theorems: dict[str, list[dict[str, object]]] | None = None,
    certificate_schema: int = 2,
    binary_entries: dict[str, dict[str, object]] | None = None,
) -> dict:
    """
    @brief Convert one proof chapter into typed topological certificate steps.
    @details Dependencies are resolved against exact local rows, alpha-equivalent
    theorem rows, integration templates, and explicitly selected prior theorems.
    @param chapter Parsed processed-proof chapter.
    @param signatures Typed atom signatures in the selected binary closure.
    @param selected_theorems Alpha-keyed selected theorem reference records.
    @param constructed_or_definitions Checked OR definitions reconstructed from rows.
    @param certificate_theorems Alpha-keyed references imported from a dependency certificate.
    @param certificate_schema Neutral certificate schema version for this chapter.
    @param binary_entries Exact compiled definitions used by schema-3 validation.
    @return JSON-ready chapter certificate.
    """

    compact_rules = compact_rule_rows(chapter, binary_entries or {})
    expressions = [record["rule"] for record in compact_rules.values()]
    imported_theorem_keys = set((certificate_theorems or {}))
    selected_theorem_keys = set(selected_theorems)
    for row in chapter.rows:
        conclusion_key = alpha_key(row.conclusion.expression)
        if not (
            certificate_schema == 3
            and conclusion_key in selected_theorem_keys | imported_theorem_keys
        ):
            expressions.append(row.conclusion.expression)
        for dependency in row.dependencies:
            dependency_key = alpha_key(dependency.expression)
            if (
                certificate_schema == 3
                and dependency_key in selected_theorem_keys | imported_theorem_keys
            ):
                continue
            expressions.append(dependency.expression)
    try:
        variable_types = infer_expression_variables(expressions, signatures)
    except AssertionError as error:
        raise AssertionError(
            f"type inference failed in {chapter.path.name}: {error}"
        ) from error

    produced_keys = {
        _expression_key(row.conclusion, row.namespace)
        for row in chapter.rows
    }
    produced_alpha_keys = {
        _alpha_expression_key(row.conclusion, row.namespace)
        for row in chapter.rows
    }
    available: dict[tuple[str, str, str], str] = {}
    theorem_available: dict[tuple[str, str, str], str] = {}

    def citation_resolution(dependency, tag):
        """
        @brief Apply the shared citation policy to current certificate sources.

        @details Only emitted rows are available locally. Registered theorem
        ambiguity is checked by the certificate's unique-reference selector.

        @param dependency Parsed expression with exact namespace and proof scope.
        @param tag Tag of the row consuming this citation.
        @return Shared disposition and local step identifier when applicable.
        """
        return resolve_citation_source(
            tag, _expression_key(dependency.scoped_expression, dependency.namespace),
            _alpha_expression_key(dependency.scoped_expression, dependency.namespace),
            available, theorem_available, produced_keys,
            selected_theorem_reference(dependency.expression) is not None,
            dependency.scoped_expression.proof_scope == INTEGRATION_GOAL_SCOPE,
            optional_companion=tag == "or theorem")

    imported_theorems = certificate_theorems or {}

    def selected_theorem_reference(
        expression,
    ) -> tuple[str, dict[str, object]] | None:
        """
        @brief Resolve one expression to a unique selected prior theorem.
        @details Alpha-equivalent ambiguity is rejected before a reference is
        recorded.
        @param expression Parsed cited theorem expression.
        @return Source-kind plus theorem reference record, or None when absent.
        """

        matches = selected_theorems.get(alpha_key(expression), [])
        imported_matches = imported_theorems.get(alpha_key(expression), [])
        assert not (matches and imported_matches), (
            "theorem citation occurs in both selected and imported certificates"
        )
        if matches:
            assert len(matches) == 1, (
                "alpha-equivalent theorem citation is ambiguous across selected sources"
            )
            return "selected_theorem", matches[0]
        if imported_matches:
            assert len(imported_matches) == 1, (
                "alpha-equivalent theorem citation is ambiguous across imported certificates"
            )
            return "certificate_theorem", imported_matches[0]
        return None

    schema3_metadata: dict[int, dict[str, object]] = {}
    scope_contexts: list[dict[str, object]] = []
    if certificate_schema == 3:
        assert binary_entries is not None
        combined_references: dict[str, list[dict[str, object]]] = {}
        for key in set(selected_theorems) | set(imported_theorems):
            combined_references[key] = [
                *selected_theorems.get(key, []),
                *imported_theorems.get(key, []),
            ]
        schema3_metadata, scope_contexts = _schema3_action_metadata(
            chapter,
            binary_entries,
            combined_references,
            signatures,
        )

    steps: list[dict[str, object]] = []
    pending = list(chapter.rows)
    while pending:
        ready_index = next(
            (index for index in range(len(pending) - 1, -1, -1)
             if all(citation_resolution(dependency, pending[index].tag)[0] != "pending"
                    for dependency in pending[index].dependencies)), None)
        if ready_index is None:
            unresolved = [
                (row.source_line,
                 [_expression_key(dependency.scoped_expression, dependency.namespace)
                  for dependency in row.dependencies
                  if citation_resolution(dependency, row.tag)[0] == "pending"])
                for row in pending]
            raise AssertionError(
                f"proof rows are not topologically resolvable in {chapter.path.name}: "
                f"{unresolved!r}"
            )
        row = pending.pop(ready_index)
        assert row.tag in ACTION_BY_TAG, (
            f"unsupported proof tag {row.tag!r} in {chapter.path.name}:{row.source_line}"
        )
        step_id = f"chapter_{chapter.number}_line_{row.source_line}"
        references: list[dict[str, object]] = []
        for dependency in row.dependencies:
            key = _expression_key(
                dependency.scoped_expression,
                dependency.namespace,
            )
            reference = {
                "expression": _scoped_expression_json(
                    dependency.scoped_expression
                ),
                "namespace": dependency.namespace,
            }
            matched_reference = selected_theorem_reference(dependency.expression)
            kind, resolved_step = citation_resolution(dependency, row.tag)
            if kind == "optional":
                continue
            if kind == "theorem":
                assert matched_reference is not None
                source_kind, theorem_reference = matched_reference
                reference["source_kind"] = source_kind
                reference["theorem_reference"] = theorem_reference
            elif kind == "local":
                reference["source_kind"] = "local_step"
                reference["step"] = resolved_step
            else:
                assert (kind == "template"
                        and dependency.scoped_expression.proof_scope == INTEGRATION_GOAL_SCOPE
                        and key not in produced_keys), (
                    f"unresolved dependency in {chapter.path.name}:{row.source_line}: "
                    f"{key[0]} at {key[1]} ({key[2]})")
                reference["source_kind"] = "integration_goal_template"
            references.append(reference)
        compact_rule = compact_rules.get(row.source_line)
        if compact_rule is not None:
            # A compact-first row (D-503): a
            # derived step proves the rule the compact stands for with the
            # row's own action and citations, and the row's step folds that
            # rule into the compact.
            rule = ScopedExpression(compact_rule["rule"], to_mpl(compact_rule["rule"]), ORDINARY_SCOPE)
            rule_step = {
                "id": f"{step_id}_rule",
                "source_line": row.source_line,
                "source_tag": row.tag,
                "action": ACTION_BY_TAG[row.tag],
                "conclusion": _scoped_expression_json(rule),
                "namespace": row.namespace,
                "dependencies": references,
                "derived_rule": True,
            }
            if "introduction" in compact_rule:
                rule_step["action"] = "structural_or_introduce"
                rule_step["introduction"] = compact_rule["introduction"]
            elif row.source_line in schema3_metadata:
                rule_step.update(schema3_metadata[row.source_line])
            steps.append(rule_step)
            steps.append({
                "id": step_id,
                "source_line": row.source_line,
                "source_tag": row.tag,
                "action": "integration_compact",
                "conclusion": _scoped_expression_json(row.conclusion),
                "namespace": row.namespace,
                "dependencies": [{
                    "expression": _scoped_expression_json(rule),
                    "namespace": row.namespace,
                    "source_kind": "local_step",
                    "step": rule_step["id"],
                }],
            })
            available[_expression_key(row.conclusion, row.namespace)] = step_id
            continue
        step = {
            "id": step_id,
            "source_line": row.source_line,
            "source_tag": row.tag,
            "action": ACTION_BY_TAG[row.tag],
            "conclusion": _scoped_expression_json(row.conclusion),
            "namespace": row.namespace,
            "dependencies": references,
        }
        from .conjunction_refutation import (
            build_conjunction_refutation_plan, is_conjunction_refutation_expansion)
        refutation_sources = [item.expression for item in row.dependencies]
        from .or_introduction import match_or_introduction
        introduction = (match_or_introduction(row.conclusion.expression, refutation_sources[0], binary_entries or {})
                        if row.tag == "expansion" and len(refutation_sources) == 1 else None)
        if introduction is not None:
            assert row.namespace == row.dependencies[0].namespace
            step["action"] = "structural_or_introduce"
            step["introduction"] = introduction
        elif row.tag == "expansion" and is_conjunction_refutation_expansion(
                row.conclusion.expression, refutation_sources, binary_entries or {}):
            from .contraposition import validate_auxiliary_scopes
            validate_auxiliary_scopes(row.namespace, [item.namespace for item in row.dependencies])
            step["action"] = "conjunction_refute"
            step["refutation"] = build_conjunction_refutation_plan(
                row.conclusion.expression, refutation_sources, binary_entries or {})
        elif row.tag == "definition projection":
            from .definition_projection import build_definition_projection_plan
            assert len(row.dependencies) == len(references) == 1
            source = row.dependencies[0]
            assert source.namespace == row.namespace
            assert source.scoped_expression.proof_scope == row.conclusion.proof_scope == ORDINARY_SCOPE
            assert references[0]["source_kind"] == "local_step"
            step["projection"] = build_definition_projection_plan(
                row.conclusion.expression, source.expression, binary_entries or {})
        elif row.tag == "zoom in":
            assert len(row.dependencies) == 1 and len(references) == 1
            source = row.dependencies[0]
            assert row.conclusion == source.scoped_expression
            assert row.namespace.startswith(source.namespace + "_boundary_")
            assert _expression_key(source.scoped_expression, source.namespace) in produced_keys
            assert references[0]["source_kind"] == "local_step"
        elif row.tag == "implication composition":
            assert len(row.dependencies) == 2 and len(references) == 2
            first, second = row.dependencies
            validate_composition_scopes(row.namespace, first.namespace, second.namespace)
            step["composition"] = build_implication_composition_plan(
                row.conclusion.expression, first.scoped_expression.expression,
                second.scoped_expression.expression)
        elif row.tag in ("implication grouping", "contraposition"):
            from .contraposition import (
                build_contraposition_plan, build_grouping_plan, validate_auxiliary_scopes)
            assert len(references) == len(row.dependencies)
            validate_auxiliary_scopes(row.namespace, [item.namespace for item in row.dependencies])
            sources = [item.scoped_expression.expression for item in row.dependencies]
            if row.tag == "contraposition":
                assert len(sources) == 1
                step["auxiliary"] = build_contraposition_plan(row.conclusion.expression, sources[0])
            else:
                step["auxiliary"] = build_grouping_plan(row.conclusion.expression, sources,
                                                      binary_entries or {})
        elif row.tag == "theorem":
            theorem_key = alpha_key(row.conclusion.expression)
            matches = [
                *selected_theorems.get(theorem_key, []),
                *imported_theorems.get(theorem_key, []),
            ]
            assert len(matches) == 1, (
                f"theorem row in {chapter.path.name}:{row.source_line} does not "
                "reference exactly one selected or imported theorem"
            )
            step["theorem_reference"] = matches[0]
        elif row.tag == "or theorem":
            # The shared definition fixes the operator; the head arguments are
            # this row's own, since OR theorems sharing a head may bind
            # different names, and so is the place of each construction
            # disjunct among the operator's stored elements.
            matching_definitions = [
                {**definition, "arguments": list(atom.arguments)}
                for atom in iter_atoms(row.conclusion.expression)
                if (definition := constructed_or_definitions.get(atom.head))
                is not None
            ]
            assert len(matching_definitions) == 1
            or_atom, disjuncts, parent_order = _or_construction_disjuncts(
                row.conclusion.expression, row
            )
            definition = matching_definitions[0]
            assert definition["head"] == or_atom.head
            assert definition["arguments"] == list(or_atom.arguments)
            definition["disjunct_positions"] = _or_disjunct_positions(
                definition, or_atom, disjuncts
            )
            step["constructed_or_definition"] = definition
            # Recorded only when a folded premise stands before a shared one;
            # otherwise every shared layer precedes every case.
            premise_count = sum(kind == "premise" for kind, _ in parent_order)
            default_order = (
                [["premise", index] for index in range(premise_count)]
                + [["case", index] for index in range(1, len(disjuncts))]
            )
            if parent_order != default_order:
                step["or_parent_order"] = parent_order
            assert references, (
                f"or-theorem row in {chapter.path.name}:{row.source_line} "
                "resolved no parent citation at all"
            )
            assert all(
                dependency["source_kind"] == "selected_theorem"
                for dependency in references
            )
        elif row.tag == "or elimination":
            # Pre-split merge: three cited theorems (variant A, variant B,
            # the licensing or theorem). The or head for the Lean-side
            # simp comes from the licensing citation's expression — its
            # or<N> atom must be a constructed-or definition of this
            # certificate (the license is itself an exported or theorem).
            assert len(row.dependencies) == 3 and len(references) == 3
            assert all(
                dependency["source_kind"] == "selected_theorem"
                for dependency in references
            )
            license_heads = [
                definition["head"]
                for atom in iter_atoms(
                    row.dependencies[2].scoped_expression.expression)
                if (definition := constructed_or_definitions.get(atom.head))
                is not None
            ]
            assert len(license_heads) == 1, (
                f"or-elimination row in {chapter.path.name}:{row.source_line} "
                "does not cite exactly one constructed-or license"
            )
            step["or_definition_head"] = license_heads[0]
        elif row.tag == "variable copy":
            assert not references
            conclusion = row.conclusion.expression
            assert isinstance(conclusion, Atom)
            assert conclusion.head == "=" and len(conclusion.arguments) == 2
            source, alias = conclusion.arguments
            assert alias == source + "_copy" or (
                alias == TWIN_PREDECESSOR_ALIAS and source != alias)
            step["variable_alias"] = {"source": source, "alias": alias}
        if row.source_line in schema3_metadata:
            step.update(schema3_metadata[row.source_line])
        steps.append(step)
        available[_expression_key(row.conclusion, row.namespace)] = step_id
        if row.tag == "theorem":
            alpha_row_key = _alpha_expression_key(row.conclusion, row.namespace)
            assert alpha_row_key not in theorem_available, (
                f"duplicate alpha-equivalent theorem row in {chapter.path.name}"
            )
            theorem_available[alpha_row_key] = step_id

    result = {
        "number": chapter.number,
        "role": chapter.role,
        "source_file": chapter.path.name,
        "source_sha256": sha256_file(chapter.path),
        "variable_types": variable_types,
        "steps": steps,
    }
    if certificate_schema == 3:
        result["scope_contexts"] = scope_contexts
    return result


def binary_entries_with_overrides(
    records: tuple[TheoremRecord, ...],
    resolved_selections: list[tuple[object, TheoremRecord]],
    binary_path: Path,
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    """
    @brief Load the GL binary and complete it from the selected chapters' own rows.
    @details
    Constructed OR theorems define their compact head: in GL's stored order
    when the binary file holds it (one operator per canonical body, so its
    theorems may state the disjunction in other orders), else from the theorem
    itself; expansion, compilation and reformulation rows expose definitions
    the binary file may lack (a shortcut-local compact head, a stale fixture
    entry). Every
    reconstructed definition must agree with any earlier reconstruction of the
    same head, and it overrides the file entry. A reformulation row is the
    exception: an existence the binary file holds keeps GL's body, because GL
    keys every definition by its canonical form (one body per name) while the
    rows citing that name may state its output equality either way round; the
    Lean rendering supplies the symmetry step. Several OR theorems may share
    one compact head with their own bound-variable names, so their
    reconstructions agree on the operator (arity, elements, signature) but not
    on ``arguments``, the head arguments of the theorem that produced them; the
    first reconstruction is kept, and each ``or theorem`` step reads its own
    head arguments from its row. The result is the definition vocabulary every
    later validation step reads.
    @param records Complete theorem registry of the proof graph (for the OR heads).
    @param resolved_selections Selected (selection, parsed theorem) pairs; only
        the theorem record of each pair is read.
    @param binary_path GL binary definition file of the source batch.
    @return Constructed OR definitions, every override (OR plus row-derived),
        and the completed binary entries.
    """

    binary_entries = json.loads(binary_path.read_text(encoding="utf-8"))
    assert isinstance(binary_entries, dict)
    constructed_or_definitions: dict[str, dict[str, object]] = {}
    for theorem_record in records:
        if theorem_record.method != "or theorem":
            continue
        theorem = find_theorem(
            records,
            to_mpl(theorem_record.expression),
            theorem_record.method,
            theorem_record.source_index,
        )
        definition = _constructed_or_definition(theorem, binary_entries)
        head = str(definition["head"])
        existing = constructed_or_definitions.get(head)
        if existing is None:
            constructed_or_definitions[head] = definition
            continue
        assert (
            {key: value for key, value in existing.items() if key != "arguments"}
            == {key: value for key, value in definition.items() if key != "arguments"}
        ), f"OR theorems define {head} differently"

    compiled_overrides = dict(constructed_or_definitions)
    for _, theorem in resolved_selections:
        for chapter in theorem.chapters:
            for row in chapter.rows:
                from .conjunction_refutation import is_conjunction_refutation_expansion
                if row.tag == "expansion" and is_conjunction_refutation_expansion(
                        row.conclusion.expression, [item.expression for item in row.dependencies], binary_entries):
                    continue
                if len(row.dependencies) != 1:
                    continue
                if is_introduction_compact_row(row, binary_entries):
                    continue
                from .or_introduction import match_or_introduction
                if row.tag == "expansion" and match_or_introduction(
                        row.conclusion.expression, row.dependencies[0].expression, binary_entries) is not None:
                    continue
                if row.tag == "reformulated from":
                    target_cursor = row.conclusion.expression
                    while isinstance(target_cursor, Implication):
                        target_cursor = target_cursor.conclusion
                    assert isinstance(target_cursor, Atom)
                    source = target_cursor
                elif row.tag in {"expansion", "expansion for integration"}:
                    source = row.dependencies[0].expression
                elif row.tag == "compilation":
                    source = row.conclusion.expression
                else:
                    continue
                if (
                    not isinstance(source, Atom)
                    # A typed `=` expands to its two subsets by
                    # extensionality: `=` is atomic, never a definition.
                    or source.head == EQUALITY
                    or source.head in compiled_overrides
                    or (
                        source.head in binary_entries
                        and len(source.arguments) != len(set(source.arguments))
                    )
                    # A reformulation existence keeps GL's own body: one
                    # canonical body per name, which several rows may cite
                    # with its equality stated either way round.
                    or (
                        row.tag == "reformulated from"
                        and source.head in binary_entries
                    )
                ):
                    continue
                definition = _definition_from_expansion(row)
                existing = compiled_overrides.get(source.head)
                assert existing is None or existing == definition
                compiled_overrides[source.head] = definition
    binary_entries.update(compiled_overrides)
    return constructed_or_definitions, compiled_overrides, binary_entries


def build_certificate(
    selection_path: Path,
    proof_graph: Path,
    config_path: Path,
    binary_path: Path,
) -> dict:
    """
    @brief Build a typed certificate for every selected theorem.
    @details The complete theorem graph, compiled definition closure, chapter
    rows, action metadata, hashes, coverage, and exclusions are asserted and
    serialized without invoking GL.
    @param selection_path Selection schema naming the desired theorem corpus.
    @param proof_graph Existing processed proof-graph directory.
    @param config_path Typed GL configuration used by the source corpus.
    @param binary_path Tracked GL binary definition fixture.
    @return Complete backend-neutral certificate.
    """

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    assert isinstance(selection, dict)
    selection_schema = int(selection.get("schema_version", 0))
    assert selection_schema in {1, 2, 3}
    records = load_theorem_records(proof_graph)
    resolved_selections, theorem_edges = _ordered_selected_theorems(
        selection,
        records,
    )
    theorem_list_hash = sha256_file(proof_graph / "global_theorem_list.txt")
    if selection_schema in {2, 3}:
        assert theorem_list_hash == str(selection["theorem_list_sha256"])

    dependency_certificates: list[dict[str, object]] = []
    dependency_certificate_by_id: dict[str, dict[str, object]] = {}
    imported_references: dict[str, list[dict[str, object]]] = {}
    imported_binary_definitions: dict[str, dict[str, object]] = {}
    imported_types: dict[str, list[str]] = {}
    if selection_schema == 3:
        dependency_specs = selection.get("certificate_dependencies")
        assert isinstance(dependency_specs, list)
        for dependency_spec in dependency_specs:
            assert isinstance(dependency_spec, dict)
            dependency_path = (
                selection_path.parent / str(dependency_spec["certificate"])
            ).resolve()
            dependency_hash = sha256_file(dependency_path)
            assert dependency_hash == str(dependency_spec["certificate_sha256"])
            dependency_certificate = json.loads(
                dependency_path.read_text(encoding="utf-8")
            )
            assert isinstance(dependency_certificate, dict)
            dependency_id = str(dependency_spec["id"])
            assert dependency_certificate["corpus"]["id"] == dependency_id
            theorem_list_scope = str(
                dependency_spec.get("theorem_list_scope", "same")
            )
            assert theorem_list_scope in {"same", "external"}
            if theorem_list_scope == "same":
                assert (
                    dependency_certificate["source"]["theorem_list_sha256"]
                    == theorem_list_hash
                )
            assert dependency_id not in dependency_certificate_by_id
            dependency_certificate_by_id[dependency_id] = dependency_certificate
            by_source = {
                int(theorem["source_index"]): theorem
                for theorem in dependency_certificate["theorems"]
            }
            for head, entry in dependency_certificate["binary_definitions"].items():
                assert isinstance(entry, dict)
                existing_definition = imported_binary_definitions.get(str(head))
                assert existing_definition is None or existing_definition == entry
                imported_binary_definitions[str(head)] = entry
            for head, signature in dependency_certificate["types"].items():
                assert isinstance(signature, list)
                existing_signature = imported_types.get(str(head))
                assert existing_signature is None or existing_signature == signature
                imported_types[str(head)] = signature
            required_indices = [
                int(source_index)
                for source_index in dependency_spec["required_source_indices"]
            ]
            assert len(required_indices) == len(set(required_indices))
            adaptations = dependency_spec.get("theorem_adaptations", [])
            assert isinstance(adaptations, list)
            adaptation_by_source = {
                int(adaptation["source_index"]): adaptation
                for adaptation in adaptations
            }
            assert len(adaptation_by_source) == len(adaptations)
            assert set(adaptation_by_source).issubset(required_indices)
            required_references: list[dict[str, object]] = []
            for source_index in required_indices:
                assert source_index in by_source
                theorem = by_source[source_index]
                adaptation = adaptation_by_source.get(source_index)
                target_theorem = (
                    parse_mpl(str(adaptation["target_theorem"]))
                    if adaptation is not None
                    else parse_mpl(str(theorem["theorem"]["mpl"]))
                )
                reference = {
                    "certificate_id": dependency_id,
                    "theorem_id": str(theorem["id"]),
                    "source_index": source_index,
                    "theorem": expression_json(target_theorem),
                }
                if theorem_list_scope == "external":
                    reference["lean_theorem"] = (
                        f"GLExport.{str(theorem['id'])}"
                    )
                if adaptation is not None:
                    head_aliases = adaptation.get("head_aliases", {})
                    assert isinstance(head_aliases, dict)
                    reference["adaptation"] = {
                        "source_theorem": theorem["theorem"],
                        "head_aliases": {
                            str(target): str(source)
                            for target, source in sorted(head_aliases.items())
                        },
                    }
                    if bool(adaptation.get("base_form", False)):
                        reference["adaptation"]["base_form"] = True
                required_references.append(reference)
                imported_references.setdefault(
                    alpha_key(target_theorem),
                    [],
                ).append(reference)
            dependency_certificates.append(
                {
                    "id": dependency_id,
                    "certificate": _portable_source_path(dependency_path),
                    "certificate_sha256": dependency_hash,
                    "theorem_list_scope": theorem_list_scope,
                    "theorem_list_sha256": dependency_certificate["source"][
                        "theorem_list_sha256"
                    ],
                    "required_theorems": required_references,
                    "provided_definition_heads": sorted(
                        dependency_certificate["binary_definitions"]
                    ),
                    "dependency_edges": dependency_certificate["corpus"][
                        "dependency_edges"
                    ],
                }
            )

    constructed_or_definitions, compiled_overrides, binary_entries = (
        binary_entries_with_overrides(records, resolved_selections, binary_path)
    )
    selected_expressions = []
    for _, theorem in resolved_selections:
        selected_expressions.append(theorem.expression)
        for chapter in theorem.chapters:
            for row in chapter.rows:
                selected_expressions.append(row.conclusion.expression)
                selected_expressions.extend(
                    dependency.expression
                    for dependency in row.dependencies
                )
    compiled_roots = {
        atom.head
        for expression in selected_expressions
        for atom in iter_atoms(expression)
        if (
            atom.head in binary_entries
            and binary_entries[atom.head].get("category") != "atomic"
        )
    }
    environment = load_type_environment(
        config_path,
        binary_path,
        extra_roots=tuple(sorted(compiled_roots - {"AnchorPeano"})),
        compiled_overrides=compiled_overrides,
        extra_signature_names=tuple(
            sorted(
                {
                    atom.head
                    for expression in selected_expressions
                    for atom in iter_atoms(expression)
                }
            )
        ),
    )

    for references in imported_references.values():
        for reference in references:
            target_expression = parse_mpl(str(reference["theorem"]["mpl"]))
            reference["target_variable_types"] = infer_expression_variables(
                [target_expression],
                environment.signatures,
            )
            adaptation = reference.get("adaptation")
            if adaptation is None:
                continue
            assert isinstance(adaptation, dict)
            dependency_id = str(reference["certificate_id"])
            dependency_certificate = dependency_certificate_by_id[dependency_id]
            source_expression = parse_mpl(
                str(adaptation["source_theorem"]["mpl"])
            )
            head_aliases = adaptation["head_aliases"]
            assert isinstance(head_aliases, dict)
            adaptation["variable_renaming"] = _validate_theorem_adaptation(
                source_expression,
                target_expression,
                dependency_certificate["binary_definitions"],
                binary_entries,
                {
                    str(target): str(source)
                    for target, source in head_aliases.items()
                },
                bool(adaptation.get("base_form", False)),
            )
            adaptation["target_variable_types"] = reference["target_variable_types"]

    citation_indices = _citation_source_indices(
        selection, [theorem for _, theorem in resolved_selections]
    )
    selected_references: dict[str, list[dict[str, object]]] = {}
    for selected, theorem in resolved_selections:
        if theorem.source_index not in citation_indices:
            continue
        selected_references.setdefault(alpha_key(theorem.expression), []).append(
            {
                "theorem_id": str(selected["id"]),
                "source_index": theorem.source_index,
                "theorem": expression_json(theorem.expression),
            }
        )

    theorem_certificates: list[dict[str, object]] = []
    for selected, theorem in resolved_selections:
        selected_id = str(selected["id"])
        theorem_types = infer_expression_variables([theorem.expression], environment.signatures)
        chapters = [
            _chapter_certificate(
                chapter,
                environment.signatures,
                selected_references,
                constructed_or_definitions,
                certificate_theorems=imported_references,
                certificate_schema=(3 if selection_schema == 3 else 2),
                binary_entries=binary_entries,
            )
            for chapter in theorem.chapters
        ]
        source_tags = {
            row.tag
            for chapter in theorem.chapters
            for row in chapter.rows
        }
        theorem_dependencies_by_id: dict[tuple[str, str], dict[str, object]] = {}
        for chapter in chapters:
            for step in chapter["steps"]:
                if "theorem_reference" in step:
                    reference = step["theorem_reference"]
                    theorem_dependencies_by_id[
                        (
                            str(reference.get("certificate_id", "")),
                            str(reference["theorem_id"]),
                        )
                    ] = reference
                for dependency in step["dependencies"]:
                    if "theorem_reference" in dependency:
                        reference = dependency["theorem_reference"]
                        theorem_dependencies_by_id[
                            (
                                str(reference.get("certificate_id", "")),
                                str(reference["theorem_id"]),
                            )
                        ] = reference
        theorem_dependencies = [
            theorem_dependencies_by_id[reference_key]
            for reference_key in sorted(theorem_dependencies_by_id)
        ]
        external_dependencies = sorted(
            {
                to_mpl(row.conclusion.expression)
                for chapter in theorem.chapters
                for row in chapter.rows
                if row.tag == "externally provided theorem"
            }
        )
        if bool(selected.get("self_contained")):
            assert not theorem_dependencies, (
                f"selection {selected['id']!r} cites prior GL theorems"
            )
            assert not external_dependencies, (
                f"selection {selected['id']!r} cites external GL theorems"
            )
        theorem_certificates.append(
            {
                "id": selected_id,
                "name": str(selected["name"]),
                "source_index": theorem.source_index,
                "method": theorem.method,
                "induction_variable": (
                    theorem.reference
                    if theorem.method in ("induction", "strong induction")
                    else None
                ),
                "theorem": expression_json(theorem.expression),
                "variable_types": theorem_types,
                "source_tags": sorted(source_tags),
                "theorem_dependencies": theorem_dependencies,
                "external_dependencies": external_dependencies,
                "chapters": chapters,
            }
        )

    binary_definitions = {
        name: {
            "arity": int(environment.binary_entries[name]["arity"]),
            "category": str(environment.binary_entries[name]["category"]),
            "elements": list(environment.binary_entries[name]["elements"]),
            "signature": list(environment.signatures[name]),
        }
        for name in environment.binary_closure
    }
    isolate_definitions = bool(selection.get("isolate_definitions", False))
    if not isolate_definitions:
        for name, entry in imported_binary_definitions.items():
            existing = binary_definitions.get(name)
            assert existing is None or existing == entry
            binary_definitions[name] = entry
    certificate_types = {
        name: list(signature)
        for name, signature in sorted(environment.signatures.items())
    }
    if not isolate_definitions:
        for name, signature in imported_types.items():
            existing = certificate_types.get(name)
            assert existing is None or existing == signature
            certificate_types[name] = signature
    certificate = {
        "schema": "gl-neutral-proof-certificate",
        "schema_version": 3 if selection_schema == 3 else 2,
        "source": {
            "selection": _portable_source_path(selection_path),
            "selection_sha256": sha256_file(selection_path),
            "theorem_list": _portable_source_path(
                proof_graph / "global_theorem_list.txt"
            ),
            "theorem_list_sha256": theorem_list_hash,
            "config": _portable_source_path(config_path),
            "config_sha256": sha256_file(config_path),
            "gl_binary": _portable_source_path(binary_path),
            "gl_binary_sha256": sha256_file(binary_path),
        },
        "types": dict(sorted(certificate_types.items())),
        "binary_definitions": binary_definitions,
        "theorems": theorem_certificates,
    }
    if selection_schema == 3:
        certificate["certificate_dependencies"] = dependency_certificates
    if isolate_definitions:
        assert selection_schema == 3
        definition_base_id = str(selection["definition_base_certificate_id"])
        assert definition_base_id in dependency_certificate_by_id
        base_definitions = dependency_certificate_by_id[definition_base_id][
            "binary_definitions"
        ]
        shared_heads = sorted(
            name
            for name, entry in binary_definitions.items()
            if base_definitions.get(name) == entry
        )
        local_heads = sorted(set(binary_definitions) - set(shared_heads))
        assert local_heads
        certificate["definition_isolation"] = {
            "namespace": str(selection["definition_namespace"]),
            "base_certificate_id": definition_base_id,
            "shared_heads": shared_heads,
            "local_heads": local_heads,
        }
    if selection_schema == 2:
        method_counts: dict[str, int] = {}
        for theorem in theorem_certificates:
            method = str(theorem["method"])
            method_counts[method] = method_counts.get(method, 0) + 1
        coverage = {
            "theorems": len(theorem_certificates),
            "chapters": sum(
                len(theorem["chapters"]) for theorem in theorem_certificates
            ),
            "rows": sum(
                sum(not step.get("derived_rule", False) for step in chapter["steps"])
                for theorem in theorem_certificates
                for chapter in theorem["chapters"]
            ),
            "actions": sum(
                len(chapter["steps"])
                for theorem in theorem_certificates
                for chapter in theorem["chapters"]
            ),
            "methods": dict(sorted(method_counts.items())),
        }
        if "expected_coverage" in selection:
            assert coverage == selection["expected_coverage"]
        # An empty exclusion list is the defined state of an acyclic corpus;
        # exclusions exist only for recorded defects (the former Peano
        # circular-theorem pair).
        exclusions = selection["excluded_sources"]
        assert isinstance(exclusions, list)
        excluded_indices = {int(item["source_index"]) for item in exclusions}
        assert not excluded_indices.intersection(theorem_edges)
        certificate["corpus"] = {
            "id": str(selection["id"]),
            "coverage": coverage,
            "dependency_edges": [
                {
                    "source_index": source_index,
                    "cited_source_indices": sorted(cited),
                }
                for source_index, cited in sorted(theorem_edges.items())
            ],
        }
        certificate["excluded_sources"] = exclusions
    elif selection_schema == 3:
        method_counts: dict[str, int] = {}
        for theorem in theorem_certificates:
            method = str(theorem["method"])
            method_counts[method] = method_counts.get(method, 0) + 1
        coverage = {
            "theorems": len(theorem_certificates),
            "chapters": sum(
                len(theorem["chapters"]) for theorem in theorem_certificates
            ),
            "rows": sum(
                sum(not step.get("derived_rule", False) for step in chapter["steps"])
                for theorem in theorem_certificates
                for chapter in theorem["chapters"]
            ),
            "actions": sum(
                len(chapter["steps"])
                for theorem in theorem_certificates
                for chapter in theorem["chapters"]
            ),
            "methods": dict(sorted(method_counts.items())),
        }
        if "expected_coverage" in selection:
            assert coverage == selection["expected_coverage"]

        external_dependency_ids = {
            str(dependency["id"])
            for dependency in dependency_certificates
            if dependency["theorem_list_scope"] == "external"
        }
        if external_dependency_ids:
            expected_external_references = {
                (str(dependency["id"]), int(reference["source_index"]))
                for dependency in dependency_certificates
                if dependency["theorem_list_scope"] == "external"
                for reference in dependency["required_theorems"]
            }
            observed_external_references = {
                (
                    str(reference.get("certificate_id", "")),
                    int(reference["source_index"]),
                )
                for theorem in theorem_certificates
                for reference in theorem["theorem_dependencies"]
                if str(reference.get("certificate_id", ""))
                in external_dependency_ids
            }
            assert observed_external_references == expected_external_references
            for dependency_id in external_dependency_ids:
                dependency_certificate = dependency_certificate_by_id[dependency_id]
                dependency_corpus = dependency_certificate["corpus"]
                if int(dependency_certificate["schema_version"]) == 3:
                    assert dependency_corpus["combined_dependency_closure_acyclic"]
                else:
                    remaining_dependency = {
                        int(edge["source_index"]): {
                            int(cited) for cited in edge["cited_source_indices"]
                        }
                        for edge in dependency_corpus["dependency_edges"]
                    }
                    while remaining_dependency:
                        ready = sorted(
                            source_index
                            for source_index, cited in remaining_dependency.items()
                            if not cited
                        )
                        assert ready
                        for source_index in ready:
                            remaining_dependency.pop(source_index)
                        for cited in remaining_dependency.values():
                            cited.difference_update(ready)
        else:
            combined_edges: dict[int, set[int]] = {
                source_index: set(cited)
                for source_index, cited in theorem_edges.items()
            }
            for dependency in dependency_certificates:
                for edge in dependency["dependency_edges"]:
                    source_index = int(edge["source_index"])
                    assert source_index not in combined_edges
                    combined_edges[source_index] = {
                        int(cited) for cited in edge["cited_source_indices"]
                    }
            combined_nodes = set(combined_edges)
            assert all(
                cited in combined_nodes
                for cited_set in combined_edges.values()
                for cited in cited_set
            )
            remaining = {
                source_index: set(cited)
                for source_index, cited in combined_edges.items()
            }
            while remaining:
                ready = sorted(
                    source_index
                    for source_index, cited in remaining.items()
                    if not cited
                )
                assert ready, (
                    "selected theorem-dependency closure is cyclic: "
                    f"{dict(sorted(remaining.items()))!r}"
                )
                for source_index in ready:
                    remaining.pop(source_index)
                for cited in remaining.values():
                    cited.difference_update(ready)

        target_source_indices = [
            int(source_index)
            for source_index in selection.get("target_source_indices", [])
        ]
        assert len(target_source_indices) == len(set(target_source_indices))
        selected_source_indices = {
            int(theorem["source_index"]) for theorem in theorem_certificates
        }
        assert set(target_source_indices).issubset(selected_source_indices)
        support_source_indices = sorted(
            selected_source_indices - set(target_source_indices)
        )

        certificate["corpus"] = {
            "id": str(selection["id"]),
            "coverage": coverage,
            "dependency_edges": [
                {
                    "source_index": source_index,
                    "cited_source_indices": sorted(cited),
                }
                for source_index, cited in sorted(theorem_edges.items())
            ],
            "combined_dependency_closure_acyclic": True,
            "target_source_indices": target_source_indices,
            "support_source_indices": support_source_indices,
        }
    return certificate


def write_certificate(certificate: dict, output_path: Path) -> None:
    """
    @brief Write a deterministic backend-neutral certificate.
    @details Keys are sorted; compact separators keep the file small and let
    the C encoder write it; UTF-8 line-feed output always ends with one newline.
    @param certificate Complete certificate record.
    @param output_path Destination JSON path.
    @return None.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(certificate, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
