# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt)
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Deterministic Lean 4 rendering of typed GL proof certificates."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .mpl import (
    Atom,
    Conjunction,
    Expression,
    Implication,
    Negation,
    alpha_key,
    iter_atoms,
    map_arguments,
    parse_mpl,
    to_mpl,
)


LEAN_LICENSE = """/-
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-/"""

LEAN_TYPE = {
    "element": "α",
    "set": "GLSet α",
    "binary_relation": "GLBinaryRelation α",
    "ternary_relation": "GLTernaryRelation α",
}

CONTEXT_NAMES = {
    "N": "N",
    "i0": "zero",
    "s": "succ",
    "+": "add",
    "*": "mul",
    "i1": "one",
    "i2": "two",
    "id": "identity",
    "rec": "previous",
}

CONTEXT_TOKENS = {"N", "i0", "s", "+", "*", "i1", "i2", "id"}

ANCHOR_MPL = {
    "AnchorPeano": "(AnchorPeano[N,i0,s,+,*,i1])",
    "AnchorGauss": "(AnchorGauss[N,i0,s,+,*,i1,i2,id])",
    "AnchorFTA": "(AnchorFTA[N,i0,s,+,*,i1,i2,id])",
}

LEAN_RESERVED = {
    "as",
    "by",
    "class",
    "def",
    "do",
    "else",
    "end",
    "example",
    "exists",
    "false",
    "for",
    "forall",
    "from",
    "fun",
    "have",
    "if",
    "import",
    "in",
    "inductive",
    "instance",
    "let",
    "match",
    "namespace",
    "open",
    "opaque",
    "section",
    "structure",
    "then",
    "theorem",
    "true",
    "universe",
    "variable",
    "where",
    "with",
}


def lean_name(token: str) -> str:
    """
    @brief Convert one MPL token into a stable Lean identifier.
    @details
    Operator glyphs and digit-leading names receive readable fixed spellings;
    every other unsupported character is replaced deterministically.
    @param token MPL argument or declaration token.
    @return A legal non-reserved Lean identifier.
    """

    if token in CONTEXT_NAMES:
        return CONTEXT_NAMES[token]
    sanitized = re.sub(r"[^A-Za-z0-9_']", "_", token)
    assert sanitized, f"cannot form Lean identifier from {token!r}"
    if sanitized[0].isdigit():
        sanitized = "x_" + sanitized
    if sanitized.lower() in LEAN_RESERVED:
        sanitized = "glv_" + sanitized
    assert sanitized[0].isalpha() or sanitized[0] == "_"
    return sanitized


def lean_constant(head: str) -> str:
    """
    @brief Name one compiled GL definition in Lean.
    @details
    A private prefix separates compiled definitions from Lean syntax and the
    ordinary relation parameters used by exported theorems.
    @param head Compiled MPL head.
    @return Stable Lean constant name.
    """

    return "gl_" + lean_name(head)


def _lean_type(kind: str) -> str:
    """
    @brief Translate one finite GL port type to its ordinary Lean type.
    @details
    Sets and operations remain predicates over one carrier; no family is
    indexed by a term and no GL value is represented by a Lean type.
    @param kind Neutral-certificate type label.
    @return Lean source for the corresponding ordinary type.
    """

    assert kind in LEAN_TYPE, f"unsupported certificate type {kind!r}"
    return LEAN_TYPE[kind]


def render_expression(
    expression: Expression,
    variable_types: dict[str, str],
    replacements: dict[str, str] | None = None,
) -> str:
    """
    @brief Render one typed MPL expression as a Lean proposition.
    @details
    Membership is predicate application, binary and ternary operations are
    relations, and every MPL binder receives its finite certificate type.
    @param expression Parsed MPL expression.
    @param variable_types Type of every binder that can occur in the expression.
    @param replacements Optional chapter-local token aliases.
    @return Parenthesized Lean proposition.
    """

    base_replacements = dict(replacements or {})

    def visit(node: Expression, environment: dict[str, str]) -> str:
        """
        @brief Render one expression node below the current binder environment.
        @details Binder aliases shadow chapter replacements while all recursive
        connective structure is preserved.
        @param node Current parsed MPL expression node.
        @param environment Lexically bound MPL tokens mapped to Lean names.
        @return Parenthesized Lean term for the node.
        """

        def argument(token: str) -> str:
            """
            @brief Resolve one atom argument to its Lean identifier.
            @details Lexical binders take precedence over chapter aliases and
            the deterministic generic spelling.
            @param token MPL atom argument token.
            @return Lean identifier used at this occurrence.
            """

            return environment.get(
                token,
                base_replacements.get(token, lean_name(token)),
            )

        if isinstance(node, Atom):
            arguments = [argument(item) for item in node.arguments]
            if node.head == "in":
                assert len(arguments) == 2
                return f"({arguments[1]} {arguments[0]})"
            if node.head == "=":
                assert len(arguments) == 2
                return f"({arguments[0]} = {arguments[1]})"
            if node.head == "in2":
                assert len(arguments) == 3
                return f"({arguments[2]} {arguments[0]} {arguments[1]})"
            if node.head == "in3":
                assert len(arguments) == 4
                return (
                    f"({arguments[3]} {arguments[0]} "
                    f"{arguments[1]} {arguments[2]})"
                )
            if not arguments:
                return f"({lean_constant(node.head)} (α := α))"
            return "(" + " ".join([lean_constant(node.head), *arguments]) + ")"
        if isinstance(node, Conjunction):
            return "(" + " ∧ ".join(visit(part, environment) for part in node.parts) + ")"
        if isinstance(node, Negation):
            return f"(¬ {visit(node.inner, environment)})"
        assert isinstance(node, Implication)
        nested = dict(environment)
        binders: list[str] = []
        for bound in node.bound_variables:
            rendered = base_replacements.get(bound, lean_name(bound))
            nested[bound] = rendered
            assert bound in variable_types, f"no Lean type for binder {bound!r}"
            binders.append(f"({rendered} : {_lean_type(variable_types[bound])})")
        implication = (
            f"({visit(node.premise, nested)} → "
            f"{visit(node.conclusion, nested)})"
        )
        if not binders:
            return implication
        return f"(∀ {' '.join(binders)}, {implication})"

    return visit(expression, {})


def _definition_order(definitions: dict[str, dict[str, object]]) -> list[str]:
    """
    @brief Topologically order compiled definitions by referenced heads.
    @details
    The order is a pure function of the recorded binary closure and asserts on
    a cycle instead of emitting forward declarations or a fallback encoding.
    @param definitions Compiled definitions stored in the certificate.
    @return Dependency-first definition names.
    """

    remaining = set(definitions)
    emitted: set[str] = set()
    ordered: list[str] = []
    while remaining:
        ready = sorted(
            name
            for name in remaining
            if {
                atom.head
                for element in definitions[name]["elements"]
                for atom in iter_atoms(parse_mpl(str(element)))
                if atom.head in definitions
            }.issubset(emitted)
        )
        assert ready, f"cyclic GL binary definition closure: {sorted(remaining)!r}"
        for name in ready:
            remaining.remove(name)
            emitted.add(name)
            ordered.append(name)
    return ordered


def _definition_variable_types(
    entry: dict[str, object],
    signatures: dict[str, list[str]],
) -> dict[str, str]:
    """
    @brief Infer local definition-variable types from finite relation ports.
    @details
    Every atom position contributes one concrete certificate type. Joining two
    different types is rejected, matching the neutral frontend's port check.
    @param entry One compiled GL-binary definition.
    @param signatures Type signatures for atomic and compiled heads.
    @return Type mapping for formal parameters and locally quantified tokens.
    """

    signature = entry["signature"]
    assert isinstance(signature, list)
    result = {
        f"u_{index + 1}": str(kind)
        for index, kind in enumerate(signature)
    }
    elements = entry["elements"]
    assert isinstance(elements, list)
    for element in elements:
        for atom in iter_atoms(parse_mpl(str(element))):
            assert atom.head in signatures, f"missing signature for {atom.head!r}"
            atom_signature = signatures[atom.head]
            assert len(atom.arguments) == len(atom_signature)
            for argument, kind in zip(atom.arguments, atom_signature, strict=True):
                previous = result.get(argument)
                assert previous is None or previous == kind, (
                    f"definition variable {argument!r} joins {previous!r} and {kind!r}"
                )
                result[argument] = str(kind)
    return result


def _definition_body(
    name: str,
    entry: dict[str, object],
    signatures: dict[str, list[str]],
) -> str:
    """
    @brief Render one compiled definition body with first-use quantifiers.
    @details
    Conjunction and disjunction retain GL's left association. Implication and
    negated-universal existence categories introduce local variables exactly at
    the first element in which they occur.
    @param name Compiled definition name, used in assertion messages.
    @param entry Compiled definition record.
    @param signatures Certificate type signatures.
    @return Lean proposition forming the definition body.
    """

    elements = [parse_mpl(str(element)) for element in entry["elements"]]
    assert elements, f"compiled GL definition {name!r} has no elements"
    variable_types = _definition_variable_types(entry, signatures)
    local_variables: list[str] = []
    introduced_variables: list[list[str]] = []
    for element in elements:
        introduced_here: list[str] = []
        for atom in iter_atoms(element):
            for argument in atom.arguments:
                if not argument.startswith("u_") and argument not in local_variables:
                    local_variables.append(argument)
                    introduced_here.append(argument)
        introduced_variables.append(introduced_here)
    rendered = [render_expression(element, variable_types) for element in elements]

    def quantify(variables: list[str], body: str) -> str:
        """
        @brief Add typed ordinary universal binders for one first-use group.
        @details An empty group leaves the body byte-identical; a non-empty
        group preserves the certificate's first-use order.
        @param variables MPL local variables introduced at this definition row.
        @param body Lean proposition placed under the binders.
        @return Universally quantified Lean proposition.
        """

        if not variables:
            return body
        binders = " ".join(
            f"({lean_name(variable)} : {_lean_type(variable_types[variable])})"
            for variable in variables
        )
        return f"(∀ {binders}, {body})"

    category = str(entry["category"])
    if category in {"and", "or"}:
        operator = " ∧ " if category == "and" else " ∨ "
        body = rendered[0]
        for item in rendered[1:]:
            body = f"({body}{operator}{item})"
        return body
    if category == "implication":
        assert len(rendered) >= 2
        body = quantify(introduced_variables[-1], rendered[-1])
        for index in reversed(range(len(rendered) - 1)):
            body = quantify(
                introduced_variables[index],
                f"({rendered[index]} → {body})",
            )
        return body
    if category == "existence":
        assert len(rendered) >= 2
        body = f"(¬ {rendered[-1]})"
        for index in reversed(range(len(rendered) - 1)):
            body = quantify(
                introduced_variables[index],
                f"({rendered[index]} → {body})",
            )
        return f"(¬ {body})"
    raise AssertionError(f"unsupported compiled GL category {category!r}")


def render_definitions(
    certificate: dict[str, object],
    definition_names: list[str] | None = None,
    namespace: str = "GLExport",
    import_module: str = "GLExport.ProofSupport",
) -> str:
    """
    @brief Render an exact certificate definition closure or isolated delta for Lean.
    @details
    All GL types are ordinary predicates over one abstract carrier. The output
    imports its declared semantic base and contains no axioms. An explicit name
    list isolates a corpus-local delta without copying or shadowing its base
    definitions in the same namespace.
    @param certificate Complete neutral proof certificate.
    @param definition_names Optional exact subset of compiled heads to emit.
    @param namespace Lean namespace receiving the emitted definitions.
    @param import_module Lean module providing proof support or shared definitions.
    @return Lean module source containing all compiled definitions.
    """

    definitions = certificate["binary_definitions"]
    signatures = certificate["types"]
    assert isinstance(definitions, dict) and isinstance(signatures, dict)
    selected_definitions = definitions
    if definition_names is not None:
        assert len(definition_names) == len(set(definition_names))
        assert all(name in definitions for name in definition_names)
        selected_definitions = {
            name: definitions[name]
            for name in definition_names
        }
    blocks = [
        LEAN_LICENSE,
        "",
        f"import {import_module}",
        "",
        f"namespace {namespace}",
        "",
        "universe u",
        "",
    ]
    for name in _definition_order(selected_definitions):
        entry = selected_definitions[name]
        assert isinstance(entry, dict)
        signature = entry["signature"]
        assert isinstance(signature, list)
        parameters = " ".join(
            f"(u_{index + 1} : {_lean_type(str(kind))})"
            for index, kind in enumerate(signature)
        )
        body = _definition_body(name, entry, signatures)
        blocks.extend(
            [
                f"def {lean_constant(name)} {{α : Type u}} {parameters} : Prop :=",
                f"  {body}",
                "",
            ]
        )
    blocks.extend([f"end {namespace}", ""])
    return "\n".join(blocks)


def write_definitions(
    certificate: dict[str, object],
    output_directory: Path,
) -> Path:
    """
    @brief Write the generated ordinary-type definition module.
    @details
    This narrow writer supports syntax-checking the semantic foundation before
    theorem replay is generated. It always writes UTF-8 with line-feed endings.
    @param certificate Complete neutral Peano certificate.
    @param output_directory Lean module directory receiving the generated file.
    @return Path to the written Definitions.lean module.
    """

    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / "Definitions.lean"
    output_path.write_text(
        render_definitions(certificate),
        encoding="utf-8",
        newline="\n",
    )
    return output_path


def _left_associated_projection(count: int, index: int) -> str:
    """
    @brief Lean projection path to one element of a left-associated conjunction.
    @details A compiled `and` definition renders as `((e_0 ∧ e_1) ∧ …) ∧ e_{k-1}`,
    so element `i` is reached by `k-1-i` left projections and, for every element
    but the first, one final right projection.
    @param count Number of elements of the conjunction.
    @param index Zero-based element position.
    @return The `.1`/`.2` chain (empty for a one-element conjunction).
    """

    assert 0 <= index < count
    return ".1" * (count - 1 - index) + (".2" if index > 0 else "")


def _conjunct_path(source: Expression, target: Expression) -> list[int]:
    """
    @brief Locate one target inside a binary MPL conjunction tree.
    @details
    Each returned direction is one Lean product projection: one selects the
    left conjunct and two selects the right conjunct.
    @param source Candidate conjunction or exact target.
    @param target Expression that must occur structurally within source.
    @return Projection directions from source to target.
    """

    if source == target:
        return []
    assert isinstance(source, Conjunction), (
        f"{to_mpl(target)!r} is not a conjunct of {to_mpl(source)!r}"
    )
    assert len(source.parts) == 2, "certificate contains non-binary conjunction projection"
    for index, part in enumerate(source.parts, start=1):
        try:
            return [index, *_conjunct_path(part, target)]
        except AssertionError:
            continue
    raise AssertionError(f"{to_mpl(target)!r} is not present in conjunction")


def _row_reference(step_id: str) -> str:
    """
    @brief Convert a certificate step identifier to its local Lean fact name.
    @details
    Chapter ownership already makes source line numbers unique inside one proof
    block, so the compact row prefix remains deterministic and readable.
    @param step_id Neutral certificate step identifier.
    @return Local Lean fact identifier.
    """

    match = re.fullmatch(r"chapter_\d+_line_(\d+)", step_id)
    assert match, f"unexpected certificate step id {step_id!r}"
    return "row_" + match.group(1)


def _is_anchor(expression: Expression) -> bool:
    """
    @brief Identify a compiled exported anchor proposition.
    @details
    Exported theorems receive exactly this proposition as an explicit ordinary
    premise alongside the relational induction premise.
    @param expression Parsed MPL expression to inspect.
    @return True exactly for a supported exported anchor atom.
    """

    return isinstance(expression, Atom) and expression.head in ANCHOR_MPL


def _theorem_body(expression: Expression) -> Expression:
    """
    @brief Remove the outer exported-anchor implication from a selected theorem.
    @details
    The generated Lean declaration fixes the carrier and relation parameters
    explicitly and takes either the Peano or Gauss anchor as a named premise,
    leaving this body as the theorem result.
    @param expression Complete selected theorem expression.
    @return Implication body following the Peano anchor.
    """

    assert isinstance(expression, Implication)
    assert _is_anchor(expression.premise)
    return expression.conclusion


def _flatten_implications(expression: Expression) -> tuple[list[Expression], Expression]:
    """
    @brief Split a theorem body into ordered premises and terminal conclusion.
    @details
    Binder groups remain attached to their implication nodes; the caller uses
    them to introduce typed variables before each named premise.
    @param expression The theorem body following its anchor.
    @return Ordered premises and final conclusion.
    """

    premises: list[Expression] = []
    conclusion = expression
    while isinstance(conclusion, Implication):
        premises.append(conclusion.premise)
        conclusion = conclusion.conclusion
    return premises, conclusion


def _free_tokens(expression: Expression) -> set[str]:
    """
    @brief Collect MPL argument tokens that are free in one expression.
    @details
    Bound variables are tracked lexically through nested implications. The
    result includes finite set and relation parameters as ordinary variables.
    @param expression Parsed MPL expression.
    @return Set of free argument tokens.
    """

    def visit(node: Expression, bound: set[str]) -> set[str]:
        """
        @brief Collect free tokens below one recursive expression node.
        @details The inherited bound set is copied at each implication binder.
        @param node Current expression node.
        @param bound Tokens bound by enclosing implications.
        @return Free tokens below node.
        """

        if isinstance(node, Atom):
            return {argument for argument in node.arguments if argument not in bound}
        if isinstance(node, Conjunction):
            return set().union(*(visit(part, bound) for part in node.parts))
        if isinstance(node, Negation):
            return visit(node.inner, bound)
        assert isinstance(node, Implication)
        nested = set(bound)
        nested.update(node.bound_variables)
        return visit(node.premise, nested) | visit(node.conclusion, nested)

    return visit(expression, set())


def _ordered_implication_dependencies(step: dict[str, object]) -> list[dict[str, object]]:
    """
    @brief Order certificate dependencies by the rule premises they satisfy.
    @details
    Provenance order is not logical implication order. A unique structural
    unification assigns every cited fact to one quantified premise.
    @param step Neutral implication-application action.
    @return Premise dependencies in rule order, excluding the rule itself.
    """

    dependencies = step["dependencies"]
    assert isinstance(dependencies, list) and len(dependencies) >= 2
    rule = parse_mpl(str(dependencies[0]["expression"]["mpl"]))
    premises: list[Expression] = []
    bound_variables: set[str] = set()
    conclusion = rule
    while isinstance(conclusion, Implication):
        bound_variables.update(conclusion.bound_variables)
        premises.append(conclusion.premise)
        conclusion = conclusion.conclusion

    def unify(
        pattern: Expression,
        actual: Expression,
        substitutions: dict[str, str],
    ) -> bool:
        """
        @brief Unify one rule fragment against one cited expression.
        @details Only rule-bound atom arguments may acquire substitutions.
        @param pattern Rule-side expression.
        @param actual Certificate-side cited expression.
        @param substitutions Mutable consistent substitution map.
        @return True exactly when the structural unification succeeds.
        """

        if type(pattern) is not type(actual):
            return False
        if isinstance(pattern, Atom):
            assert isinstance(actual, Atom)
            if pattern.head != actual.head or len(pattern.arguments) != len(actual.arguments):
                return False
            for expected, observed in zip(pattern.arguments, actual.arguments, strict=True):
                if expected in bound_variables:
                    if expected in substitutions and substitutions[expected] != observed:
                        return False
                    substitutions[expected] = observed
                elif expected != observed:
                    return False
            return True
        if isinstance(pattern, Conjunction):
            assert isinstance(actual, Conjunction)
            return len(pattern.parts) == len(actual.parts) and all(
                unify(expected, observed, substitutions)
                for expected, observed in zip(pattern.parts, actual.parts, strict=True)
            )
        if isinstance(pattern, Negation):
            assert isinstance(actual, Negation)
            return unify(pattern.inner, actual.inner, substitutions)
        assert isinstance(pattern, Implication) and isinstance(actual, Implication)
        return (
            pattern.bound_variables == actual.bound_variables
            and unify(pattern.premise, actual.premise, substitutions)
            and unify(pattern.conclusion, actual.conclusion, substitutions)
        )

    substitutions: dict[str, str] = {}
    assert unify(
        conclusion,
        parse_mpl(str(step["conclusion"]["mpl"])),
        substitutions,
    ), f"implication conclusion does not match certificate step {step['id']!r}"
    solutions: list[list[dict[str, object]]] = []

    def search(
        premise_index: int,
        remaining: list[dict[str, object]],
        current_substitutions: dict[str, str],
        ordered: list[dict[str, object]],
    ) -> None:
        """
        @brief Search the finite premise-to-dependency bijections.
        @details More than one complete ordering is retained only long enough
        to reject ambiguity.
        @param premise_index Next logical premise index.
        @param remaining Dependencies not yet assigned.
        @param current_substitutions Substitutions fixed by earlier matches.
        @param ordered Dependencies selected so far.
        @return None; complete solutions append to the enclosing list.
        """

        if premise_index == len(premises):
            if not remaining:
                solutions.append(list(ordered))
            return
        premise = premises[premise_index]
        for candidate_index, candidate in enumerate(remaining):
            trial = dict(current_substitutions)
            if not unify(
                premise,
                parse_mpl(str(candidate["expression"]["mpl"])),
                trial,
            ):
                continue
            search(
                premise_index + 1,
                remaining[:candidate_index] + remaining[candidate_index + 1:],
                trial,
                [*ordered, candidate],
            )
            if len(solutions) > 1:
                return

    search(0, list(dependencies[1:]), substitutions, [])
    assert len(solutions) == 1, (
        f"implication dependencies in {step['id']!r} have "
        f"{len(solutions)} logical orderings"
    )
    return solutions[0]


def _chapter_replacements(chapter: dict[str, object]) -> dict[str, str]:
    """
    @brief Derive the fixed context names and chapter-local copy aliases.
    @details
    Anchor specialization and variable-copy actions are identities. Rendering
    the copied token with the source identifier keeps that fact definitional.
    @param chapter Neutral certificate chapter.
    @return MPL token to Lean identifier replacements.
    """

    replacements = dict(CONTEXT_NAMES)
    steps = chapter["steps"]
    assert isinstance(steps, list)
    for step in steps:
        action = str(step["action"])
        main_unbound: tuple[str, ...] = ()
        if action == "variable_alias":
            alias = step["variable_alias"]
            assert isinstance(alias, dict)
            source = str(alias["source"])
            copied = str(alias["alias"])
            assert copied == source + "_copy"
            rendered_source = replacements.get(source, lean_name(source))
            assert copied not in replacements or replacements[copied] == rendered_source
            replacements[copied] = rendered_source
            continue
        if action != "anchor_specialize":
            continue
        dependencies = step["dependencies"]
        assert isinstance(dependencies, list) and len(dependencies) == 1
        source = parse_mpl(str(dependencies[0]["expression"]["mpl"]))
        target = parse_mpl(str(step["conclusion"]["mpl"]))
        assert isinstance(source, Atom) and isinstance(target, Atom)
        assert source.head == target.head and source.head in ANCHOR_MPL
        assert len(source.arguments) == len(target.arguments)
        changed = [
            (observed, expected)
            for expected, observed in zip(source.arguments, target.arguments, strict=True)
            if expected != observed
        ]
        assert changed
        for copied, original in changed:
            rendered_original = replacements.get(original, lean_name(original))
            assert copied not in replacements or replacements[copied] == rendered_original
            replacements[copied] = rendered_original
    return replacements


def _context_signature(anchor_head: str = "AnchorPeano") -> list[str]:
    """
    @brief Render the common abstract Peano, Gauss, or FTA context parameters.
    @details
    The context uses one carrier and ordinary predicate types. Relational
    induction is an explicit proposition-valued premise, never hidden in an axiom.
    FTA receives the same premise as an AnchorFTA-indexed schema so a certificate
    row that compacts an already proved theorem can reconstruct its globally
    quantified proposition rather than only its current-context specialization.
    @param anchor_head AnchorPeano, AnchorGauss, or AnchorFTA context selector.
    @return Lean declaration lines for the selected context.
    """

    assert anchor_head in ANCHOR_MPL
    lines = [
        "    {α : Type u}",
        "    (N : GLSet α)",
        "    (zero : α)",
        "    (succ : GLBinaryRelation α)",
        "    (add mul : GLTernaryRelation α)",
        "    (one : α)",
    ]
    if anchor_head in {"AnchorGauss", "AnchorFTA"}:
        lines.extend(
            [
                "    (two : α)",
                "    (identity : GLBinaryRelation α)",
            ]
        )
    anchor_arguments = (
        "N zero succ add mul one"
        if anchor_head == "AnchorPeano"
        else "N zero succ add mul one two identity"
    )
    lines.append(f"    (anchor : {lean_constant(anchor_head)} {anchor_arguments})")
    if anchor_head == "AnchorFTA":
        lines.extend(
            [
                "    (relationalInduction :",
                "      ∀ (induction_N : GLSet α)",
                "        (induction_zero : α)",
                "        (induction_succ : GLBinaryRelation α)",
                "        (induction_add induction_mul : GLTernaryRelation α)",
                "        (induction_one induction_two : α)",
                "        (induction_identity : GLBinaryRelation α),",
                "        gl_AnchorFTA induction_N induction_zero induction_succ",
                "          induction_add induction_mul induction_one induction_two",
                "          induction_identity →",
                "        ∀ (P : α → Prop) (k : α),",
                "          P induction_zero →",
                "          (∀ n, induction_N n → P n →",
                "            ∀ m, induction_succ n m → P m) →",
                "          induction_N k →",
                "          P k)",
            ]
        )
    else:
        lines.extend(
            [
                "    (relationalInduction :",
                "      ∀ (P : α → Prop) (k : α),",
                "        P zero →",
                "        (∀ n, N n → P n → ∀ m, succ n m → P m) →",
                "        N k →",
                "        P k)",
            ]
        )
    return lines


# The dependency-certificate ids that carry a Peano corpus. Callee context
# application and manifest coverage treat every member identically; the two
# ids differ only in which manifest generation they were selected from
# (peano_full_65 = pre-cycle-fix with two exclusions, peano_full_66 = the
# complete post-fix corpus, peano_full_64 = the complete corpus of the current
# prover, whose Peano batch proves two fewer induction theorems).
PEANO_CORPUS_IDS = frozenset({"peano_full_65", "peano_full_66", "peano_full_64"})

FTA_CORPUS_IDS = {"fta_shortlist_42", "fta_shortlist_79"}

# Theorem and row totals of the tracked release corpora, asserted whenever one
# of those ids is rendered; a live corpus (`peano_live_N`, `gauss_live_N`,
# `fta_live_N`, written by `proof_export.live` inside a pipeline run) records
# its coverage instead of pinning it.
TRACKED_CORPUS_COUNTS = {
    "gauss_main_29": (29, 1511),
    "fta_shortlist_42": (51, 1788),
    "fta_shortlist_79": (79, 2835),
}


def corpus_kind(corpus_id: str | None) -> str | None:
    """
    @brief Classify a corpus id as a Peano, Gauss or FTA corpus.
    @details
    Every corpus id starts with its kind — `peano_full_64`, `gauss_main_29`,
    `fta_shortlist_79`, or the live `peano_live_N` / `gauss_live_N` /
    `fta_live_N` — and the renderer's anchor context, imports and manifest
    shape follow the kind, never the individual id. A missing id (`None`) is
    the local corpus: a theorem calling a sibling of its own module.
    @param corpus_id Certificate or dependency-certificate identifier, or None.
    @return `"peano"`, `"gauss"`, `"fta"`, or None for a local reference.
    """

    if corpus_id is None:
        return None
    kind = corpus_id.split("_", 1)[0]
    assert kind in ("peano", "gauss", "fta"), f"unknown corpus id {corpus_id!r}"
    return kind


def _context_application(
    anchor_head: str = "AnchorPeano",
    certificate_id: str | None = None,
) -> str:
    """
    @brief Render arguments passed to an earlier generated theorem.
    @details
    The order exactly matches the selected context signature. Gauss and FTA
    theorems derive the smaller imported anchors through generated projection
    theorems whose bodies unfold the checked compiled anchor definitions.
    @param anchor_head Context of the theorem currently being rendered.
    @param certificate_id Optional dependency-certificate identifier of the callee.
    @return Space-separated Lean argument text.
    """

    assert anchor_head in ANCHOR_MPL
    if anchor_head == "AnchorPeano":
        assert corpus_kind(certificate_id) in (None, "peano")
        return "N zero succ add mul one anchor relationalInduction"
    if anchor_head == "AnchorGauss" and corpus_kind(certificate_id) == "peano":
        peano_anchor = (
            "(anchorPeanoOfGauss N zero succ add mul one two identity anchor)"
        )
        return (
            f"N zero succ add mul one {peano_anchor} relationalInduction"
        )
    if anchor_head == "AnchorGauss":
        assert certificate_id is None
        return (
            "N zero succ add mul one two identity anchor relationalInduction"
        )
    assert anchor_head == "AnchorFTA"
    if corpus_kind(certificate_id) == "peano":
        peano_anchor = (
            "(anchorPeanoOfFTA N zero succ add mul one two identity anchor)"
        )
        return f"N zero succ add mul one {peano_anchor} relationalInduction"
    if corpus_kind(certificate_id) == "gauss":
        gauss_anchor = (
            "(anchorGaussOfFTA N zero succ add mul one two identity anchor)"
        )
        return (
            "N zero succ add mul one two identity "
            f"{gauss_anchor} relationalInduction"
        )
    assert certificate_id is None
    return "N zero succ add mul one two identity anchor relationalInduction"


def _scoped_expression(
    conclusion: Expression,
    scope_context: dict[str, object],
    local_variables: tuple[str, ...] = (),
) -> Expression:
    """
    @brief Place one integration-row conclusion under its recorded scope.
    @details
    The schema-3 template supplies ordered binder and premise layers. Its
    terminal integration goal is replaced by the row conclusion, so Lean sees
    an ordinary universally quantified implication rather than a dependent
    type family or an implicit namespace.
    @param conclusion Parsed row conclusion inside the integration boundary.
    @param scope_context Validated schema-3 scope-context record.
    @param local_variables Free integration parameters closed by the scope.
    @return Rebuilt ordinary proposition with every scope layer applied once.
    @invariant The context template contains at least one implication layer.
    """

    template = parse_mpl(str(scope_context["template"]["mpl"]))
    layers: list[tuple[tuple[str, ...], Expression]] = []
    cursor = template
    while isinstance(cursor, Implication):
        layers.append((cursor.bound_variables, cursor.premise))
        cursor = cursor.conclusion
    assert layers
    first_bound, first_premise = layers[0]
    layers[0] = ((*local_variables, *first_bound), first_premise)
    result = conclusion
    for bound_variables, premise in reversed(layers):
        result = Implication(bound_variables, premise, result)
    return result


def _render_scoped_proposition(
    body: str,
    scope_context: dict[str, object],
    variable_types: dict[str, str],
    replacements: dict[str, str],
    local_variables: tuple[str, ...] = (),
) -> str:
    """
    @brief Wrap already rendered Lean text in one validated integration scope.
    @details
    This textual twin supports scoped existential witness bundles, which have
    no separate MPL syntax node. Binder types and premise order still come
    exclusively from the neutral scope template.
    @param body Already rendered Lean proposition placed at the scope terminal.
    @param scope_context Validated schema-3 scope-context record.
    @param variable_types Certificate types for every scope binder.
    @param replacements Chapter-local Lean identifier aliases.
    @param local_variables Free integration parameters closed by the scope.
    @return Universally quantified implication text for the scoped proposition.
    @invariant Every binder has one finite ordinary certificate type.
    """

    template = parse_mpl(str(scope_context["template"]["mpl"]))
    layers: list[tuple[tuple[str, ...], Expression]] = []
    cursor = template
    while isinstance(cursor, Implication):
        layers.append((cursor.bound_variables, cursor.premise))
        cursor = cursor.conclusion
    assert layers
    first_bound, first_premise = layers[0]
    layers[0] = ((*local_variables, *first_bound), first_premise)
    result = body
    for bound_variables, premise in reversed(layers):
        rendered_premise = render_expression(
            premise,
            variable_types,
            replacements,
        )
        result = f"({rendered_premise} → {result})"
        if bound_variables:
            binders = " ".join(
                f"({replacements.get(variable, lean_name(variable))} : "
                f"{_lean_type(variable_types[variable])})"
                for variable in bound_variables
            )
            result = f"(∀ {binders}, {result})"
    return result


def _existence_projection_sources(
    chapter: dict[str, object],
) -> dict[str, list[Expression]]:
    """
    @brief Group projections that share one classically extracted witness.
    @details
    GL may project several conjuncts from one negated-universal existence row.
    The generated proof extracts one witness and reuses it for every projection.
    @param chapter Neutral certificate chapter.
    @return Source step identifiers mapped to their projected expressions.
    """

    grouped: dict[str, list[Expression]] = {}
    steps = chapter["steps"]
    assert isinstance(steps, list)
    for step in steps:
        if step["action"] != "compound_project":
            continue
        dependencies = step["dependencies"]
        assert isinstance(dependencies, list) and len(dependencies) == 1
        source = parse_mpl(str(dependencies[0]["expression"]["mpl"]))
        if (
            isinstance(source, Negation)
            and isinstance(source.inner, Implication)
            and bool(source.inner.bound_variables)
        ):
            grouped.setdefault(str(dependencies[0]["step"]), []).append(
                parse_mpl(str(step["conclusion"]["mpl"]))
            )
    return grouped


def _manifest_disposition(
    step: dict[str, object],
    owner: str,
    fact: str,
    mechanism: str,
) -> dict[str, object]:
    """
    @brief Form the manifest record paired with one emitted GL proof row.
    @details
    The renderer invokes this exactly once after constructing the row's named
    Lean fact, so certificate coverage can be asserted before writing files.
    @param step Neutral certificate action.
    @param owner Generated theorem or helper owning the fact.
    @param fact Local Lean fact name.
    @param mechanism Concise checked proof mechanism.
    @return Deterministically ordered row disposition.
    """

    return {
        "certificate_step": step["id"],
        "source_line": step["source_line"],
        "source_tag": step["source_tag"],
        "lean_fact": f"{owner}/{fact}",
        "mechanism": mechanism,
        "status": "emitted",
    }


def _dependency_facts(dependencies: list[dict[str, object]]) -> list[str]:
    """
    @brief Render local fact names for step-backed dependencies.
    @details
    Selected-theorem edges are handled by their dedicated actions and therefore
    are rejected here rather than converted to an invented local name.
    @param dependencies Certificate dependency records with local step ids.
    @return Lean fact identifiers in certificate order.
    """

    facts: list[str] = []
    for dependency in dependencies:
        assert "step" in dependency, f"dependency has no local step: {dependency!r}"
        facts.append(_row_reference(str(dependency["step"])))
    return facts


def _scoped_dependency(
    dependency: dict[str, object],
    fact: str,
    scoped_prefix: dict[str, tuple[str, ...]],
    scoped_witness: dict[str, tuple[str, ...]],
    scoped_existential: set[str],
    scoped_existential_guard: dict[str, tuple[str, ...]],
) -> tuple[
    dict[str, object],
    str,
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    bool,
]:
    """
    @brief Pair one cited dependency with everything its instantiation needs.
    @details
    Instantiating a cited scoped fact requires that fact's own scope prefix, the
    scope witnesses it produces or consumes, any previously established witness
    guards an existential projection itself consumes, and whether it is the
    projecting row. A main-scope fact carries empty tuples and is never a
    projection, which is exactly what the absent-key defaults produce.
    @param dependency Neutral certificate dependency record.
    @param fact Lean fact name the dependency was rendered under.
    @param scoped_prefix Recorded scope prefix per rendered scoped fact.
    @param scoped_witness Recorded scope witnesses per rendered scoped fact.
    @param scoped_existential Facts rendered as a witness-stating existential.
    @param scoped_existential_guard Existing witnesses required by an
    existential fact before it can project its new witness.
    @return Instantiation record consumed by the scope-entry renderer.
    @see _scope_entry_and_specializations
    """

    return (
        dependency,
        fact,
        scoped_prefix.get(fact, ()),
        scoped_witness.get(fact, ()),
        scoped_existential_guard.get(fact, ()),
        fact in scoped_existential,
    )


def _scope_entry_and_specializations(
    namespace: str,
    scope_contexts: dict[str, dict[str, object]],
    local_variables: tuple[str, ...],
    row_witness: tuple[str, ...],
    dependencies: list[
        tuple[
            dict[str, object],
            str,
            tuple[str, ...],
            tuple[str, ...],
            tuple[str, ...],
            bool,
        ]
    ],
    replacements: dict[str, str],
) -> list[str]:
    """
    @brief Enter a row's integration scope and instantiate its cited facts there.
    @details
    A scoped row is rendered as an ordinary universally quantified implication
    built from the scope template, so its cited facts in the same scope carry the
    same quantifier prefix. Lean's first-order search cannot choose a value for a
    scope binder the cited fact's body never mentions, so a scoped row would fail
    from its complete citation while succeeding from the surrounding context. The
    remedy is structural rather than permissive: introduce the scope prefix once,
    then apply every same-scope cited fact to exactly that prefix, leaving the
    search a quantifier-free step over the row's own citation. Facts cited from
    the enclosing main scope need no instantiation and are passed through
    untouched. A main-scope row has no prefix to introduce and yields no lines.

    A scope-local existence witness is one variable shared by every GL row that
    speaks about it. The projecting row states it existentially and every other
    row is guarded by its bundle, so this routine obtains the witness once from
    the projection and applies each guarded row to exactly that witness. Letting
    each row quantify its own witness would lose the sharing and leave a step
    underivable from its complete citation.
    @param namespace Certificate namespace of the row being proved.
    @param scope_contexts Validated schema-3 scope-context records by namespace.
    @param local_variables Free integration parameters closed by the scope.
    @param row_witness Scope witnesses this row itself is quantified over.
    @param dependencies Cited dependency records, each paired with its Lean fact
    name, its own scope prefix, its produced or consumed witnesses, guards needed
    by an existential projection, and whether it is the projecting row.
    @param replacements Chapter-local MPL token to Lean identifier aliases.
    @return Lean tactic lines introducing the scope prefix and the instantiations.
    @invariant The introduced prefix is exactly the prefix the scope template
    renders, so an instantiation of a same-scope fact always typechecks.
    @see _scoped_expression
    """

    if namespace == "main":
        return []
    assert namespace in scope_contexts, f"unknown scope namespace {namespace!r}"
    template = parse_mpl(str(scope_contexts[namespace]["template"]["mpl"]))
    template_layers: list[tuple[str, ...]] = []
    cursor = template
    while isinstance(cursor, Implication):
        template_layers.append(cursor.bound_variables)
        cursor = cursor.conclusion
    assert template_layers, (
        f"scope template for {namespace!r} has no implication layer"
    )

    def prefix_arguments(prefix_variables: tuple[str, ...]) -> list[str]:
        """
        @brief List one scoped proposition's prefix arguments in binder order.
        @details
        The prefix interleaves each template layer's binders with that layer's
        premise, exactly as the scoped rendering builds it, so the returned list
        is the argument sequence that instantiates such a proposition.
        @param prefix_variables Free integration parameters closed by the scope.
        @return Binder and premise names in application order.
        """

        layers = list(template_layers)
        layers[0] = (*prefix_variables, *layers[0])
        arguments: list[str] = []
        for index, bound_variables in enumerate(layers, start=1):
            arguments.extend(
                replacements.get(variable, lean_name(variable))
                for variable in bound_variables
            )
            arguments.append(f"scope_premise_{index}")
        return arguments

    lines = [
        f"    intro {name}"
        for name in prefix_arguments(local_variables)
    ]
    guard_by_witness: dict[str, str] = {}
    for witness_variable in row_witness:
        name = replacements.get(witness_variable, lean_name(witness_variable))
        guard = f"witness_guard_{len(guard_by_witness) + 1}"
        lines.extend([f"    intro {name}", f"    intro {guard}"])
        guard_by_witness[witness_variable] = guard

    same_scope = [
        (position, *record)
        for position, record in enumerate(dependencies, start=1)
        if str(record[0]["namespace"]) == namespace
    ]
    # A projection that establishes a witness must be obtained before any row
    # guarded by that witness is applied, whatever the rule's premise order is.
    same_scope.sort(key=lambda record: not record[6])
    for (
        position,
        _,
        fact,
        dependency_prefix,
        witnesses,
        existential_guards,
        existential,
    ) in same_scope:
        application = " ".join(prefix_arguments(dependency_prefix))
        if existential:
            # A projection states its witness existentially: it takes the scope
            # prefix and nothing more. It supplies the witness when this row does
            # not already have one, and is otherwise an ordinary cited fact.
            for witness_variable in existential_guards:
                assert witness_variable in guard_by_witness, (
                    f"{fact} needs witness {witness_variable!r} that no cited "
                    "row establishes"
                )
                application += (
                    f" {replacements.get(witness_variable, lean_name(witness_variable))}"
                    f" {guard_by_witness[witness_variable]}"
                )
            if all(witness in guard_by_witness for witness in witnesses):
                lines.append(
                    f"    have scoped_fact_{position} := {fact} {application}"
                )
                continue
            assert len(witnesses) == 1, (
                f"{fact} projects {len(witnesses)} witnesses at once"
            )
            name = replacements.get(witnesses[0], lean_name(witnesses[0]))
            guard = f"witness_guard_{len(guard_by_witness) + 1}"
            lines.append(
                f"    obtain ⟨{name}, {guard}⟩ := {fact} {application}"
            )
            guard_by_witness[witnesses[0]] = guard
            continue
        assert not existential_guards
        for witness_variable in witnesses:
            assert witness_variable in guard_by_witness, (
                f"{fact} needs witness {witness_variable!r} that no cited row "
                f"establishes"
            )
            application += (
                f" {replacements.get(witness_variable, lean_name(witness_variable))}"
                f" {guard_by_witness[witness_variable]}"
            )
        lines.append(f"    have scoped_fact_{position} := {fact} {application}")
    return lines


def _existence_implication_direction(
    source: Expression,
    conclusion: Expression,
    binary_definitions: dict[str, object] | None,
) -> tuple[list[str], bool] | None:
    """
    @brief Recognize the expansion of a negated existence into one of its
    implication compacts and decide the premise order.
    @details
    A two-element existence compact `E[args]` unfolds to
    `¬ ∀ x, left → ¬ right`; its negation therefore yields the positive
    universal `∀ x, left → ¬ right`. The prover's existence-implication
    compacts (I-209) state that universal in one of its two orientations,
    `left → ¬ right` or `right → ¬ left`, and the proof graph records the
    step as an `expansion` row whose source is the negated existence and
    whose conclusion is the compact. The orientation is decided by
    instantiating both compiled definitions at the row's arguments and
    comparing them alpha-equivalently as bound implications; exactly one
    orientation must match.
    @param source Parsed source expression of the expansion row.
    @param conclusion Parsed conclusion expression of the row.
    @param binary_definitions Certificate definitions, or None when the row
        renderer runs without them.
    @return None when the row is not this shape (an ordinary unfold); else the
        existence's bound placeholders in first-use order and whether the
        compact states the swapped orientation `right → ¬ left`.
    """

    if (
        binary_definitions is None
        or not isinstance(source, Negation)
        or not isinstance(source.inner, Atom)
        or not isinstance(conclusion, Atom)
    ):
        return None
    existence_entry = binary_definitions.get(source.inner.head)
    compact_entry = binary_definitions.get(conclusion.head)
    if (
        not isinstance(existence_entry, dict)
        or not isinstance(compact_entry, dict)
        or existence_entry.get("category") != "existence"
        or compact_entry.get("category") != "implication"
    ):
        return None
    existence_elements = [parse_mpl(str(element)) for element in existence_entry["elements"]]
    compact_elements = [parse_mpl(str(element)) for element in compact_entry["elements"]]
    if (
        len(existence_elements) != 2
        or len(compact_elements) != 2
        or not isinstance(compact_elements[1], Negation)
    ):
        return None

    def placeholders_of(elements: list[Expression]) -> list[str]:
        """
        @brief Collect a definition's local placeholders in first-use order.
        @param elements Parsed definition elements.
        @return Every argument token that is not a `u_` parameter, once each.
        """

        found: list[str] = []
        for element in elements:
            for atom in iter_atoms(element):
                for argument in atom.arguments:
                    if not argument.startswith("u_") and argument not in found:
                        found.append(argument)
        return found

    def instantiate(expression: Expression, arguments: tuple[str, ...]) -> Expression:
        """
        @brief Substitute a definition's `u_` parameters by the row's arguments.
        @param expression Parsed definition element.
        @param arguments Arguments of the atom the definition is applied to.
        @return The element at the row's arguments; placeholders untouched.
        """

        parameter_map = {f"u_{index + 1}": argument for index, argument in enumerate(arguments)}
        return map_arguments(expression, lambda name: parameter_map.get(name, name))

    existence_bound = tuple(placeholders_of(existence_elements))
    compact_bound = tuple(placeholders_of(compact_elements))
    left = instantiate(existence_elements[0], source.inner.arguments)
    right = instantiate(existence_elements[1], source.inner.arguments)
    premise = instantiate(compact_elements[0], conclusion.arguments)
    negated = instantiate(compact_elements[1].inner, conclusion.arguments)
    compact_key = alpha_key(Implication(compact_bound, premise, Negation(negated)))
    same = compact_key == alpha_key(Implication(existence_bound, left, Negation(right)))
    swapped = compact_key == alpha_key(Implication(existence_bound, right, Negation(left)))
    assert same or swapped, (
        f"{conclusion.head} is not an implication compact of {source.inner.head}"
    )
    assert not (same and swapped), (
        f"{conclusion.head} matches both orientations of {source.inner.head}"
    )
    return list(existence_bound), swapped


def _render_chapter_steps(
    chapter: dict[str, object],
    owner: str,
    assumption_by_mpl: dict[str, str],
    replacements: dict[str, str],
    anchor_head: str = "AnchorPeano",
    ambient_variables: set[str] | None = None,
    binary_definitions: dict[str, object] | None = None,
    external_reference_facts: dict[tuple[str, int], str] | None = None,
    external_theorem_arguments: dict[str, list[str]] | None = None,
) -> tuple[list[str], list[dict[str, object]]]:
    """
    @brief Replay every neutral certificate action as one named Lean fact.
    @details
    Definitions, structural projections, and main-scope implication elimination
    use explicit Lean terms: an implication row applies the cited rule to the
    cited premises in the rule's own premise order, so no search stands between
    a GL rule firing and its Lean counterpart. The remaining rows — equality
    substitution is replayed by eliminating each cited equality in certificate
    order. Compound disintegration follows named conjunction paths, decodes
    negated-universal witnesses once, or constructs the licensed classical
    implication from a negated pair. Contradiction and integration rows likewise
    expose their cited pair, scope premise, conjunction, or equivalence. Every
    supported certificate row is therefore rendered without proof search.
    @param chapter Neutral certificate chapter in dependency order.
    @param owner Generated theorem or helper name.
    @param assumption_by_mpl Exact chapter assumptions mapped to Lean names.
    @param replacements Chapter-local token aliases.
    @param anchor_head AnchorPeano or AnchorGauss context selector.
    @param ambient_variables Tokens already bound in the surrounding Lean proof.
    @param binary_definitions Compiled definitions for schema-3 existence expansion.
    @param external_reference_facts Hash-pinned external certificate references
    mapped to explicit local Lean parameters.
    @param external_theorem_arguments External-fact arguments required by each
    previously rendered FTA theorem cited from this chapter.
    @return Lean source lines and one disposition per source row.
    """

    steps = chapter["steps"]
    variable_types = chapter["variable_types"]
    assert isinstance(steps, list) and isinstance(variable_types, dict)
    live_variables = set(CONTEXT_TOKENS)
    if ambient_variables is not None:
        live_variables.update(ambient_variables)
    unbound_tokens = lambda tokens: tuple(
        sorted(
            variable
            for variable in tokens
            if replacements.get(variable, lean_name(variable))
            not in {
                replacements.get(live, lean_name(live))
                for live in live_variables
            }
        )
    )
    steps_by_id = {str(step["id"]): step for step in steps}
    assert len(steps_by_id) == len(steps)
    witness_sources = _existence_projection_sources(chapter)
    emitted_witness_sources: set[str] = set()
    lines: list[str] = []
    dispositions: list[dict[str, object]] = []
    external_reference_facts = external_reference_facts or {}
    external_theorem_arguments = external_theorem_arguments or {}
    external_fact = lambda reference: external_reference_facts.get(
        (str(reference["certificate_id"]), int(reference["source_index"]))
    ) if reference.get("certificate_id") is not None else None
    reference_application = lambda reference: external_fact(reference) or " ".join(
        [
            str(reference.get("lean_theorem", reference["theorem_id"])),
            _context_application(anchor_head, reference.get("certificate_id")),
            *external_theorem_arguments.get(str(reference["theorem_id"]), []),
        ]
    )
    definition_constant = lambda head: lean_constant(head)
    expansion_rules = lambda head: ", ".join(
        [
            definition_constant(head),
            *(
                ["GLExport.orIffNotAndNot"]
                if anchor_head == "AnchorFTA" and head in {"or0", "or2"}
                else []
            ),
        ]
    )
    scope_contexts = {
        str(context["namespace"]): context
        for context in chapter.get("scope_contexts", [])
    }
    assert len(scope_contexts) == len(chapter.get("scope_contexts", []))
    scope_by_compact = {
        str(context["compact"]["mpl"]): context
        for context in scope_contexts.values()
        if context["template"].get("proof_scope") == "integration_goal"
    }
    assert len(scope_by_compact) == sum(
        context["template"].get("proof_scope") == "integration_goal"
        for context in scope_contexts.values()
    )

    scope_context_variables: dict[str, set[str]] = {}
    for namespace, context in scope_contexts.items():
        template = parse_mpl(str(context["template"]["mpl"]))
        compact = parse_mpl(str(context["compact"]["mpl"]))
        context_variables = _free_tokens(template) | _free_tokens(compact)
        assert all(variable in variable_types for variable in context_variables)
        scope_context_variables[namespace] = context_variables
    scope_witness_variables = {namespace: set() for namespace in scope_contexts}
    scope_facts: list[str] = [
        *sorted(set(assumption_by_mpl.values())),
        "relationalInduction",
    ]

    # A scoped row's quantifier prefix depends on which integration parameters
    # were still unbound when that row was rendered, and a main-scope witness
    # projection binds one more of them partway through the chapter. Two rows in
    # one namespace can therefore carry different prefixes, so each scoped row
    # records its own and a later instantiation replays that one.
    main_prefix: dict[str, tuple[str, ...]] = {}
    scoped_prefix: dict[str, tuple[str, ...]] = {}
    # A scope-local existence witness is one variable that GL's rows share. The
    # row that projects it states it existentially; every later row about that
    # same witness is universally quantified over it and guarded by the bundle,
    # so a consumer obtains the witness once and applies the guarded rows to it.
    # Quantifying each row's witness separately would silently drop the sharing.
    scope_witness_bundle: dict[tuple[str, str], Expression] = {}
    scoped_witness: dict[str, tuple[str, ...]] = {}
    scoped_existential: set[str] = set()
    scoped_existential_guard: dict[str, tuple[str, ...]] = {}
    scope_witness_projector: dict[tuple[str, str], str] = {}
    scope_row_guard = lambda candidate: (
        scoped_existential_guard.get(candidate, ())
        if candidate in scoped_existential
        else scoped_witness.get(candidate, ())
    )

    for step in steps:
        fact = _row_reference(str(step["id"]))
        conclusion_mpl = str(step["conclusion"]["mpl"])
        conclusion_expression = parse_mpl(conclusion_mpl)
        namespace = str(step["namespace"])
        dependencies = step["dependencies"]
        assert isinstance(dependencies, list)
        action = str(step["action"])
        projected_witness_bundle: Expression | None = None
        projected_witness_variables: tuple[str, ...] = ()
        if action == "compound_project" and dependencies:
            source_step = str(dependencies[0].get("step", ""))
            if source_step in witness_sources:
                source = parse_mpl(str(dependencies[0]["expression"]["mpl"]))
                assert isinstance(source, Negation)
                assert isinstance(source.inner, Implication)
                assert source.inner.bound_variables
                witness_variable = source.inner.bound_variables[0]
                projected_witness_variables = (witness_variable,)
                if namespace == "main":
                    live_variables.add(witness_variable)
                else:
                    assert namespace in scope_witness_variables
                    scope_witness_variables[namespace].add(witness_variable)
                    projections = witness_sources[source_step]
                    projected_witness_bundle = (
                        projections[0]
                        if len(projections) == 1
                        else Conjunction(tuple(projections))
                    )
                    recorded = scope_witness_bundle.setdefault(
                        (namespace, witness_variable),
                        projected_witness_bundle,
                    )
                    assert to_mpl(recorded) == to_mpl(projected_witness_bundle), (
                        f"witness {witness_variable!r} in {namespace!r} has two bundles"
                    )
        if action == "theorem_reference":
            proposition = ""
        else:
            if namespace != "main":
                assert namespace in scope_contexts
                scoped_prefix[fact] = unbound_tokens(
                    scope_context_variables[namespace]
                )
                witness_variables = tuple(
                    sorted(
                        _free_tokens(conclusion_expression)
                        & scope_witness_variables[namespace]
                    )
                )
                cited_facts = [
                    _row_reference(str(dependency["step"]))
                    for dependency in dependencies
                    if "step" in dependency
                ]
                establishable = {
                    witness
                    for cited in cited_facts
                    if cited in scoped_existential
                    for witness in scoped_witness.get(cited, ())
                }
                inherited = {
                    witness
                    for cited in cited_facts
                    if cited not in scoped_existential
                    for witness in scoped_witness.get(cited, ())
                }
                inherited.update(
                    witness
                    for cited in cited_facts
                    if cited in scoped_existential
                    for witness in scoped_existential_guard.get(cited, ())
                )
                # A row that states a scope witness is always quantified over it
                # so its consumers can apply it to the witness they obtained. A
                # row that merely consumes guarded facts needs the guard only for
                # witnesses no cited projection can supply.
                guard_set = set(witness_variables) | (inherited - establishable)
                pending = list(guard_set)
                while pending:
                    witness_variable = pending.pop()
                    bundle = scope_witness_bundle.get(
                        (namespace, witness_variable)
                    )
                    assert bundle is not None, (
                        f"row {step['id']!r} cites witness "
                        f"{witness_variable!r} before it is established"
                    )
                    prerequisites = (
                        _free_tokens(bundle)
                        & scope_witness_variables[namespace]
                    ) - guard_set - {witness_variable}
                    guard_set.update(prerequisites)
                    pending.extend(sorted(prerequisites))
                ordered_guards: list[str] = []
                remaining_guards = set(guard_set)
                while remaining_guards:
                    ready = sorted(
                        witness_variable
                        for witness_variable in remaining_guards
                        if not (
                            _free_tokens(
                                scope_witness_bundle[(namespace, witness_variable)]
                            )
                            & (remaining_guards - {witness_variable})
                        )
                    )
                    assert ready, (
                        f"row {step['id']!r} has cyclic witness guards "
                        f"{sorted(remaining_guards)}"
                    )
                    ordered_guards.extend(ready)
                    remaining_guards.difference_update(ready)
                guard_variables = tuple(ordered_guards)
                if projected_witness_bundle is None and guard_variables:
                    # This row speaks about a witness another row established, so
                    # it is quantified over that witness and guarded by its
                    # bundle. A consumer obtains the witness from the projecting
                    # row and applies this one to exactly that witness.
                    assert set(witness_variables) <= set(guard_variables), (
                        f"row {step['id']!r} states a witness it cannot bind"
                    )
                    scoped_witness[fact] = guard_variables
                    guarded = conclusion_expression
                    for witness_variable in reversed(guard_variables):
                        bundle = scope_witness_bundle.get(
                            (namespace, witness_variable)
                        )
                        assert bundle is not None, (
                            f"row {step['id']!r} cites witness "
                            f"{witness_variable!r} before it is established"
                        )
                        guarded = Implication(
                            (witness_variable,),
                            bundle,
                            guarded,
                        )
                    rendered_conclusion = _scoped_expression(
                        guarded,
                        scope_contexts[namespace],
                        unbound_tokens(scope_context_variables[namespace]),
                    )
                elif projected_witness_bundle is not None:
                    assert set(projected_witness_variables) <= set(
                        witness_variables
                    )
                    # The witnesses this projection needs before it can state
                    # its own: every guard of the row (its conclusion's other
                    # witnesses AND the guards of the cited source — a source
                    # already guarded by an earlier witness passes that guard
                    # on even when the projected conjunct never mentions it),
                    # in dependency order, minus the witness projected here.
                    inherited_witnesses = tuple(
                        witness_variable
                        for witness_variable in guard_variables
                        if witness_variable not in projected_witness_variables
                    )
                    assert projected_witness_bundle is not None, (
                        f"row {step['id']!r} states a scope witness without "
                        f"establishing it"
                    )
                    scoped_existential.add(fact)
                    scoped_witness[fact] = projected_witness_variables
                    scoped_existential_guard[fact] = inherited_witnesses
                    for projected_witness in projected_witness_variables:
                        scope_witness_projector.setdefault(
                            (namespace, projected_witness),
                            fact,
                        )
                    witness_expression = projected_witness_bundle
                    witness_body = render_expression(
                        witness_expression,
                        variable_types,
                        replacements,
                    )
                    witness_binders = " ".join(
                        f"({replacements.get(variable, lean_name(variable))} : "
                        f"{_lean_type(variable_types[variable])})"
                        for variable in projected_witness_variables
                    )
                    existential_body = f"(∃ {witness_binders}, {witness_body})"
                    for witness_variable in reversed(inherited_witnesses):
                        bundle = scope_witness_bundle.get(
                            (namespace, witness_variable)
                        )
                        assert bundle is not None, (
                            f"row {step['id']!r} needs inherited witness "
                            f"{witness_variable!r} before projecting another"
                        )
                        name = replacements.get(
                            witness_variable,
                            lean_name(witness_variable),
                        )
                        rendered_bundle = render_expression(
                            bundle,
                            variable_types,
                            replacements,
                        )
                        existential_body = (
                            f"(∀ ({name} : "
                            f"{_lean_type(variable_types[witness_variable])}), "
                            f"({rendered_bundle} → {existential_body}))"
                        )
                    proposition = _render_scoped_proposition(
                        existential_body,
                        scope_contexts[namespace],
                        variable_types,
                        replacements,
                        unbound_tokens(scope_context_variables[namespace]),
                    )
                    rendered_conclusion = None
                else:
                    rendered_conclusion = _scoped_expression(
                        conclusion_expression,
                        scope_contexts[namespace],
                        unbound_tokens(scope_context_variables[namespace]),
                    )
            else:
                rendered_conclusion = conclusion_expression
            if rendered_conclusion is not None:
                try:
                    proposition = render_expression(
                        rendered_conclusion,
                        variable_types,
                        replacements,
                    )
                    if namespace == "main":
                        unbound = unbound_tokens(_free_tokens(rendered_conclusion))
                        main_unbound = unbound
                        main_prefix[fact] = unbound
                        if unbound:
                            binders = " ".join(
                                f"({replacements.get(variable, lean_name(variable))} : "
                                f"{_lean_type(variable_types[variable])})"
                                for variable in unbound
                            )
                            proposition = f"(∀ {binders}, {proposition})"
                except AssertionError as error:
                    raise AssertionError(
                        f"Lean proposition rendering failed for {owner}/{step['id']}: {error}"
                    ) from error
        proof: list[str]
        mechanism: str
        scoped_action = namespace != "main" or any(
            str(dependency["namespace"]) != "main"
            for dependency in dependencies
        )

        if action in {"assume", "induction_assumption"}:
            if namespace == "main":
                assert conclusion_mpl in assumption_by_mpl, (
                    f"unmapped chapter assumption {conclusion_mpl!r}"
                )
                proof = [f"    exact {assumption_by_mpl[conclusion_mpl]}"]
                mechanism = "explicit theorem or chapter assumption"
            else:
                assert action == "assume" and step["scope_namespace"] == namespace
                proof = [
                    *_scope_entry_and_specializations(
                        namespace,
                        scope_contexts,
                        scoped_prefix.get(fact, ()),
                        (),
                        [],
                        replacements,
                    ),
                    "    exact scope_premise_1",
                ]
                mechanism = "explicit scoped assumption introduction"
        elif action == "definition_expand":
            assert len(dependencies) == 1
            source = parse_mpl(str(dependencies[0]["expression"]["mpl"]))
            definition_atom = (
                source
                if isinstance(source, Atom)
                else source.inner
                if isinstance(source, Negation) and isinstance(source.inner, Atom)
                else None
            )
            assert isinstance(definition_atom, Atom)
            if dependencies[0].get("source_kind") == "integration_goal_template":
                assert isinstance(source, Atom)
                compact_mpl = to_mpl(source)
                compact = render_expression(source, variable_types, replacements)
                expanded = render_expression(
                    conclusion_expression,
                    variable_types,
                    replacements,
                )
                definition_bound_variables: tuple[str, ...] = ()
                if binary_definitions is not None and source.head in binary_definitions:
                    definition_entry = binary_definitions[source.head]
                    assert isinstance(definition_entry, dict)
                    if (
                        definition_entry.get("category") == "existence"
                        and not isinstance(conclusion_expression, Negation)
                    ):
                        definition_elements = [
                            parse_mpl(str(element))
                            for element in definition_entry["elements"]
                        ]
                        parameter_map = {
                            f"u_{index + 1}": argument
                            for index, argument in enumerate(source.arguments)
                        }
                        local_placeholders: list[str] = []
                        introduced_placeholders: list[list[str]] = []
                        for element in definition_elements:
                            introduced_here: list[str] = []
                            for atom in iter_atoms(element):
                                for argument in atom.arguments:
                                    if (
                                        not argument.startswith("u_")
                                        and argument not in local_placeholders
                                    ):
                                        local_placeholders.append(argument)
                                        introduced_here.append(argument)
                            introduced_placeholders.append(introduced_here)
                        output_candidates = (
                            _free_tokens(conclusion_expression) - _free_tokens(source)
                        )
                        ordered_candidates: list[str] = []
                        for atom in iter_atoms(conclusion_expression):
                            for argument in atom.arguments:
                                if (
                                    argument in output_candidates
                                    and argument not in ordered_candidates
                                ):
                                    ordered_candidates.append(argument)
                        assert len(local_placeholders) == len(ordered_candidates)
                        output_map = dict(
                            zip(local_placeholders, ordered_candidates, strict=True)
                        )
                        definition_bound_variables = tuple(ordered_candidates)
                        instantiated_elements = [
                            map_arguments(
                                element,
                                lambda name: output_map.get(
                                    name,
                                    parameter_map.get(name, name),
                                ),
                            )
                            for element in definition_elements
                        ]
                        rendered_elements = [
                            render_expression(
                                element,
                                variable_types,
                                replacements,
                            )
                            for element in instantiated_elements
                        ]
                        definition_body = f"(¬ {rendered_elements[-1]})"
                        for index in reversed(range(len(rendered_elements) - 1)):
                            introduced = [
                                output_map[placeholder]
                                for placeholder in introduced_placeholders[index]
                            ]
                            definition_body = (
                                f"({rendered_elements[index]} → {definition_body})"
                            )
                            if introduced:
                                binders = " ".join(
                                    f"({replacements.get(variable, lean_name(variable))} : "
                                    f"{_lean_type(variable_types[variable])})"
                                    for variable in introduced
                                )
                                definition_body = f"(∀ {binders}, {definition_body})"
                        expanded = f"(¬ {definition_body})"
                equivalence = f"({compact} ↔ {expanded})"
                if compact_mpl in scope_by_compact:
                    integration_context = scope_by_compact[compact_mpl]
                    integration_namespace = str(integration_context["namespace"])
                    unbound = tuple(
                        variable
                        for variable in unbound_tokens(
                            scope_context_variables[integration_namespace]
                        )
                        if variable not in definition_bound_variables
                    )
                    main_prefix[fact] = unbound
                    if unbound:
                        binders = " ".join(
                            f"({replacements.get(variable, lean_name(variable))} : "
                            f"{_lean_type(variable_types[variable])})"
                            for variable in unbound
                        )
                        proposition = f"(∀ {binders}, {equivalence})"
                    else:
                        proposition = equivalence
                else:
                    unbound = unbound_tokens(
                        _free_tokens(source) | _free_tokens(conclusion_expression)
                    )
                    unbound = tuple(
                        variable
                        for variable in unbound
                        if variable not in definition_bound_variables
                    )
                    main_prefix[fact] = unbound
                    if unbound:
                        binders = " ".join(
                            f"({replacements.get(variable, lean_name(variable))} : "
                            f"{_lean_type(variable_types[variable])})"
                            for variable in unbound
                        )
                        proposition = f"(∀ {binders}, {equivalence})"
                    else:
                        proposition = equivalence
                if binary_definitions is not None:
                    proof = []
                    if unbound:
                        proof.append(
                            "    intro "
                            + " ".join(
                                replacements.get(variable, lean_name(variable))
                                for variable in unbound
                            )
                        )
                    proof.append("    exact Iff.rfl")
                else:
                    proof = [f"    simp only [{expansion_rules(source.head)}]"]
                mechanism = "integration-goal definitional equivalence"
            else:
                source_fact = _row_reference(str(dependencies[0]["step"]))
                existence_direction = _existence_implication_direction(
                    source,
                    conclusion_expression,
                    binary_definitions,
                )
                if existence_direction is None:
                    proof = [
                        f"    simpa only [{expansion_rules(definition_atom.head)}] using {source_fact}"
                    ]
                    mechanism = f"unfold {lean_constant(definition_atom.head)}"
                else:
                    placeholders, swapped = existence_direction
                    witnesses = [
                        f"existence_witness_{index + 1}"
                        for index in range(len(placeholders))
                    ]
                    premise_name = "compact_premise"
                    negated_name = "compact_negated"
                    unfolded_name = f"unfolded_{source_fact}"
                    positive_name = f"positive_{source_fact}"
                    hypothesis_order = (
                        [negated_name, premise_name]
                        if swapped
                        else [premise_name, negated_name]
                    )
                    # A scoped row first enters its scope and instantiates the
                    # cited negated existence there; the compact unfold and the
                    # witness introductions then act on the bare goal.
                    if scoped_action:
                        assert str(dependencies[0]["namespace"]) == namespace
                        entry_lines = _scope_entry_and_specializations(
                            namespace,
                            scope_contexts,
                            scoped_prefix.get(fact, ()),
                            scope_row_guard(fact),
                            [
                                _scoped_dependency(
                                    dependencies[0],
                                    source_fact,
                                    scoped_prefix,
                                    scoped_witness,
                                    scoped_existential,
                                    scoped_existential_guard,
                                )
                            ],
                            replacements,
                        )
                        source_term = "scoped_fact_1"
                    else:
                        entry_lines = []
                        source_term = source_fact
                    proof = [
                        *entry_lines,
                        f"    simp only [{expansion_rules(conclusion_expression.head)}]",
                        f"    intro {' '.join(witnesses + [premise_name, negated_name])}",
                        f"    have {unfolded_name} := {source_term}",
                        f"    simp only [{expansion_rules(definition_atom.head)}] at {unfolded_name}",
                        f"    apply {unfolded_name}",
                        f"    intro {positive_name}",
                        f"    exact {positive_name} {' '.join(witnesses + hypothesis_order)}",
                    ]
                    mechanism = (
                        f"negated {lean_constant(definition_atom.head)} to its "
                        f"implication compact {lean_constant(conclusion_expression.head)}"
                    )
        elif action == "compound_project":
            assert len(dependencies) == 1
            source = parse_mpl(str(dependencies[0]["expression"]["mpl"]))
            source_step = str(dependencies[0]["step"])
            source_fact = _row_reference(source_step)
            bundle_fact = f"witness_{source_fact}"
            proof = []
            if scoped_action:
                assert str(dependencies[0]["namespace"]) == namespace
                assert source_fact not in scoped_existential
                proof.extend(
                    _scope_entry_and_specializations(
                        namespace,
                        scope_contexts,
                        scoped_prefix.get(fact, ()),
                        scope_row_guard(fact),
                        [
                            _scoped_dependency(
                                dependencies[0],
                                source_fact,
                                scoped_prefix,
                                scoped_witness,
                                scoped_existential,
                                scoped_existential_guard,
                            )
                        ],
                        replacements,
                    )
                )
                source_term = "scoped_fact_1"
            else:
                assert str(dependencies[0]["namespace"]) == "main"
                source_prefix = main_prefix.get(source_fact, ())
                source_term = " ".join(
                    [
                        source_fact,
                        *(
                            replacements.get(variable, lean_name(variable))
                            for variable in source_prefix
                        ),
                    ]
                )
            if isinstance(source, Conjunction):
                path = _conjunct_path(source, conclusion_expression)
                projection = source_term + "".join(
                    ".1" if direction == 1 else ".2" for direction in path
                )
                proof.append(f"    exact {projection}")
                mechanism = (
                    "scoped named conjunction projection"
                    if scoped_action
                    else "named conjunction projection"
                )
            elif source_step in witness_sources:
                assert isinstance(source, Negation)
                inner = source.inner
                assert isinstance(inner, Implication)
                assert len(inner.bound_variables) == 1
                assert isinstance(inner.conclusion, Negation)
                witness_token = inner.bound_variables[0]
                witness_name = replacements.get(witness_token, lean_name(witness_token))
                source_witness_bundle = Conjunction(
                    (inner.premise, inner.conclusion.inner)
                )
                if scoped_action:
                    assert projected_witness_bundle is not None
                    projection_terms = []
                    for target in witness_sources[source_step]:
                        path = _conjunct_path(source_witness_bundle, target)
                        projection_terms.append(
                            "scoped_witness_bundle"
                            + "".join(
                                ".1" if direction == 1 else ".2"
                                for direction in path
                            )
                        )
                    assert len(projection_terms) in (1, 2)
                    projected_term = (
                        projection_terms[0]
                        if len(projection_terms) == 1
                        else f"⟨{projection_terms[0]}, {projection_terms[1]}⟩"
                    )
                    proof.extend(
                        [
                            (
                                f"    obtain ⟨{witness_name}, "
                                "scoped_witness_bundle⟩ := "
                                f"existsAndOfNotForallImpNot {source_term}"
                            ),
                            f"    exact ⟨{witness_name}, {projected_term}⟩",
                        ]
                    )
                    mechanism = "scoped explicit classical witness projection"
                else:
                    if source_step not in emitted_witness_sources:
                        bundle = render_expression(
                            source_witness_bundle,
                            variable_types,
                            replacements,
                        )
                        existential_fact = f"exists_{source_fact}"
                        lines.extend(
                            [
                                (
                                    f"  have {existential_fact} : ∃ "
                                    f"({witness_name} : "
                                    f"{_lean_type(variable_types[witness_token])}), "
                                    f"{bundle} := "
                                    f"existsAndOfNotForallImpNot {source_term}"
                                ),
                                (
                                    f"  obtain ⟨{witness_name}, {bundle_fact}⟩ "
                                    f":= {existential_fact}"
                                ),
                            ]
                        )
                        scope_facts.append(bundle_fact)
                        emitted_witness_sources.add(source_step)
                    path = _conjunct_path(
                        source_witness_bundle,
                        conclusion_expression,
                    )
                    projection = bundle_fact + "".join(
                        ".1" if direction == 1 else ".2"
                        for direction in path
                    )
                    proof.append(f"    exact {projection}")
                    mechanism = "shared explicit classical witness projection"
            else:
                assert isinstance(source, Negation)
                assert isinstance(source.inner, Conjunction)
                assert len(source.inner.parts) == 2
                # The projected implication is either spelled out or stated
                # as one of the disjunction's implication compacts (the
                # shortcut keeps a De Morgan disjunction `¬(¬A ∧ ¬B)` and
                # projects `implication<N>[…]` = `¬A → B` from it); the compact
                # is unfolded first and its instantiated definition is the
                # implication the classical projection then proves.
                unfold_lines: list[str] = []
                implication_expression = conclusion_expression
                if isinstance(conclusion_expression, Atom):
                    assert binary_definitions is not None
                    compact_entry = binary_definitions[conclusion_expression.head]
                    assert isinstance(compact_entry, dict)
                    assert str(compact_entry["category"]) == "implication"
                    compact_elements = [
                        parse_mpl(str(element)) for element in compact_entry["elements"]
                    ]
                    assert len(compact_elements) == 2
                    assert all(
                        argument.startswith("u_")
                        for element in compact_elements
                        for atom in iter_atoms(element)
                        for argument in atom.arguments
                    ), f"{conclusion_expression.head} binds a local; not a propositional projection"
                    compact_parameters = {
                        f"u_{index + 1}": argument
                        for index, argument in enumerate(conclusion_expression.arguments)
                    }
                    compact_premise, compact_conclusion = (
                        map_arguments(element, lambda name: compact_parameters.get(name, name))
                        for element in compact_elements
                    )
                    implication_expression = Implication((), compact_premise, compact_conclusion)
                    unfold_lines = [
                        f"    simp only [{expansion_rules(conclusion_expression.head)}]"
                    ]
                assert isinstance(implication_expression, Implication)
                assert not implication_expression.bound_variables
                left, right = source.inner.parts
                proof.extend([*unfold_lines, "    classical", "    intro projection_premise"])
                if (
                    isinstance(right, Negation)
                    and to_mpl(implication_expression.premise) == to_mpl(left)
                    and to_mpl(implication_expression.conclusion)
                    == to_mpl(right.inner)
                ):
                    proof.extend(
                        [
                            "    apply Classical.byContradiction",
                            "    intro projection_counterexample",
                            (
                                f"    exact {source_term} "
                                "⟨projection_premise, projection_counterexample⟩"
                            ),
                        ]
                    )
                else:
                    assert isinstance(implication_expression.conclusion, Negation)
                    assert to_mpl(implication_expression.premise) == to_mpl(right)
                    assert (
                        to_mpl(implication_expression.conclusion.inner)
                        == to_mpl(left)
                    )
                    proof.extend(
                        [
                            "    intro projection_counterexample",
                            (
                                f"    exact {source_term} "
                                "⟨projection_counterexample, projection_premise⟩"
                            ),
                        ]
                    )
                mechanism = "explicit classical negated-compound projection"
        elif action == "implication_apply":
            assert len(dependencies) >= 2
            rule_dependency = dependencies[0]
            ordered = _ordered_implication_dependencies(step)
            rule_is_theorem = rule_dependency["source_kind"] in {
                "selected_theorem",
                "certificate_theorem",
            }
            if rule_dependency["source_kind"] == "local_step":
                rule_fact = _row_reference(str(rule_dependency["step"]))
                rule_step = steps_by_id[str(rule_dependency["step"])]
                rule_is_theorem = rule_step["action"] == "theorem_reference"
            else:
                reference = rule_dependency["theorem_reference"]
                assert isinstance(reference, dict)
                rule_fact = f"rule_{fact}"
                interface_fact = external_fact(reference)
                theorem_application = reference_application(reference)
                adaptation = reference.get("adaptation")
                if adaptation is None:
                    lines.append(f"  have {rule_fact} := {theorem_application}")
                    scope_facts.append(rule_fact)
                else:
                    assert isinstance(adaptation, dict)
                    assert interface_fact is not None, (
                        "adapted external theorem must enter Lean through its "
                        "hash-validated target interface"
                    )
                    lines.append(f"  have {rule_fact} := {interface_fact}")
                    scope_facts.append(rule_fact)
            if rule_is_theorem:
                ordered = [
                    dependency
                    for dependency in ordered
                    if not _is_anchor(
                        parse_mpl(str(dependency["expression"]["mpl"]))
                    )
                ]
            facts = [rule_fact, *_dependency_facts(ordered)]
            assert facts
            if scoped_action:
                # Both scoped renderings — the plain conclusion and the witness
                # bundle — build the same quantifier prefix from the same scope
                # template, so entering the scope is uniform across them.
                scoped_dependencies = [
                    _scoped_dependency(
                        rule_dependency,
                        rule_fact,
                        scoped_prefix,
                        scoped_witness,
                        scoped_existential,
                        scoped_existential_guard,
                    ),
                    *(
                        _scoped_dependency(
                            dependency,
                            _row_reference(str(dependency["step"])),
                            scoped_prefix,
                            scoped_witness,
                            scoped_existential,
                            scoped_existential_guard,
                        )
                        for dependency in ordered
                    ),
                ]
                witness_guards = {
                    witness: f"witness_guard_{position}"
                    for position, witness in enumerate(
                        scope_row_guard(fact),
                        start=1,
                    )
                }
                projection_terms: dict[int, str] = {}
                for position, dependency_record in enumerate(
                    scoped_dependencies,
                    start=1,
                ):
                    (
                        dependency,
                        dependency_fact,
                        _,
                        witnesses,
                        _,
                        existential,
                    ) = (
                        dependency_record
                    )
                    if str(dependency["namespace"]) != namespace or not existential:
                        continue
                    missing_witnesses = [
                        witness
                        for witness in witnesses
                        if witness not in witness_guards
                    ]
                    if missing_witnesses:
                        assert len(witnesses) == 1, (
                            f"{dependency_fact} needs {len(missing_witnesses)} "
                            "unbound projected witnesses inside an implication"
                        )
                        witness_guards[witnesses[0]] = (
                            f"witness_guard_{len(witness_guards) + 1}"
                        )
                    dependency_expression = parse_mpl(
                        str(dependency["expression"]["mpl"])
                    )
                    matching_projections: list[str] = []
                    for witness in witnesses:
                        bundle = scope_witness_bundle.get((namespace, witness))
                        assert bundle is not None, (
                            f"{dependency_fact} has no recorded bundle for "
                            f"witness {witness!r}"
                        )
                        try:
                            path = _conjunct_path(bundle, dependency_expression)
                        except AssertionError:
                            continue
                        projection = "".join(
                            ".1" if direction == 1 else ".2"
                            for direction in path
                        )
                        matching_projections.append(
                            f"{witness_guards[witness]}{projection}"
                        )
                    assert matching_projections, (
                        f"{dependency_fact} does not project its cited "
                        f"expression {to_mpl(dependency_expression)!r}"
                    )
                    projection_terms[position] = matching_projections[0]
                application_facts = [
                    projection_terms.get(position, f"scoped_fact_{position}")
                    if str(dependency["namespace"]) == namespace
                    else dependency_fact
                    for position, (dependency, dependency_fact) in enumerate(
                        zip(
                            [rule_dependency, *ordered],
                            facts,
                            strict=True,
                        ),
                        start=1,
                    )
                ]
                proof = [
                    *_scope_entry_and_specializations(
                        namespace,
                        scope_contexts,
                        scoped_prefix.get(fact, ()),
                        scope_row_guard(fact),
                        scoped_dependencies,
                        replacements,
                    ),
                    f"    apply {application_facts[0]}",
                    *(
                        f"    exact {dependency}"
                        for dependency in application_facts[1:]
                    ),
                ]
            else:
                proof = [
                    f"    apply {rule_fact}",
                    *(f"    exact {dependency}" for dependency in facts[1:]),
                ]
            mechanism = (
                "scoped ordered implication elimination"
                if scoped_action
                else "ordered implication elimination"
            )
        elif action == "equality_transitive":
            assert len(dependencies) == 2
            facts = _dependency_facts(dependencies)
            if scoped_action:
                transitive_facts = [
                    f"scoped_fact_{position}"
                    if str(dependency["namespace"]) == namespace
                    else dependency_fact
                    for position, (dependency, dependency_fact) in enumerate(
                        zip(dependencies, facts, strict=True),
                        start=1,
                    )
                ]
                proof = [
                    *_scope_entry_and_specializations(
                        namespace,
                        scope_contexts,
                        scoped_prefix.get(fact, ()),
                        scope_row_guard(fact),
                        [
                            _scoped_dependency(
                                dependency,
                                dependency_fact,
                                scoped_prefix,
                                scoped_witness,
                                scoped_existential,
                                scoped_existential_guard,
                            )
                            for dependency, dependency_fact in zip(
                                dependencies,
                                facts,
                                strict=True,
                            )
                        ],
                        replacements,
                    ),
                    (
                        f"    exact Eq.trans {transitive_facts[0]} "
                        f"{transitive_facts[1]}"
                    ),
                ]
            else:
                proof = [f"    exact Eq.trans {facts[0]} {facts[1]}"]
            mechanism = "equality transitivity"
        elif action == "ex_falso" and step["source_tag"] == "contradiction":
            dependency_expressions = [
                parse_mpl(str(dependency["expression"]["mpl"]))
                for dependency in dependencies
            ]
            contradictory_pairs = [
                (negative_position, positive_position)
                for negative_position, negative in enumerate(dependency_expressions)
                if isinstance(negative, Negation)
                for positive_position, positive in enumerate(dependency_expressions)
                if to_mpl(negative.inner) == to_mpl(positive)
            ]
            assert len(contradictory_pairs) == 1
            negative_position, positive_position = contradictory_pairs[0]
            negative_dependency = dependencies[negative_position]
            positive_dependency = dependencies[positive_position]
            negative_fact = _row_reference(str(negative_dependency["step"]))
            positive_fact = _row_reference(str(positive_dependency["step"]))
            scoped_positions = [
                position
                for position in (negative_position, positive_position)
                if dependencies[position]["namespace"] != "main"
            ]
            if not scoped_positions:
                proof = [
                    f"    exact False.elim ({negative_fact} {positive_fact})"
                ]
            else:
                # One or both contradictory facts live in the contradiction
                # scope (a scoped negation against a main-scope fact, or two
                # scoped facts); the row concludes the negated assumption by
                # entering the scope once and instantiating every scoped fact
                # there, sharing the witnesses their guards need.
                scoped_namespaces = {
                    str(dependencies[position]["namespace"])
                    for position in scoped_positions
                }
                assert len(scoped_namespaces) == 1, "a contradiction spans two scopes"
                dependency_namespace = next(iter(scoped_namespaces))
                if len(scoped_positions) == 1:
                    main_position = (
                        positive_position
                        if scoped_positions[0] == negative_position
                        else negative_position
                    )
                    assert dependencies[main_position]["namespace"] == "main"
                assert isinstance(conclusion_expression, Negation)
                assert dependency_namespace in scope_contexts
                context_template = parse_mpl(
                    str(scope_contexts[dependency_namespace]["template"]["mpl"])
                )
                assert (
                    isinstance(context_template, Implication)
                    and not context_template.bound_variables
                    and to_mpl(context_template.premise)
                    == to_mpl(conclusion_expression.inner)
                    and to_mpl(context_template.conclusion)
                    == to_mpl(conclusion_expression.inner)
                )
                witness_lines: list[str] = []
                guard_by_witness: dict[str, str] = {}
                scoped_terms: dict[int, str] = {}
                for scoped_position in scoped_positions:
                    scoped_dependency = dependencies[scoped_position]
                    scoped_fact = _row_reference(str(scoped_dependency["step"]))
                    assert not scoped_prefix.get(scoped_fact, ())
                    witness_arguments: list[str] = []
                    for witness_variable in scoped_witness.get(scoped_fact, ()):
                        name = replacements.get(
                            witness_variable,
                            lean_name(witness_variable),
                        )
                        if witness_variable in guard_by_witness:
                            witness_arguments.extend([name, guard_by_witness[witness_variable]])
                            continue
                        projector_key = (dependency_namespace, witness_variable)
                        assert projector_key in scope_witness_projector, (
                            f"scoped contradiction dependency {scoped_fact} needs "
                            f"witness {witness_variable!r} without a projector"
                        )
                        projector = scope_witness_projector[projector_key]
                        assert projector in scoped_existential
                        assert not scoped_prefix.get(projector, ())
                        application = f"{projector} contradiction_assumption"
                        for inherited_witness in scoped_existential_guard.get(
                            projector,
                            (),
                        ):
                            assert inherited_witness in guard_by_witness
                            application += (
                                f" {replacements.get(inherited_witness, lean_name(inherited_witness))}"
                                f" {guard_by_witness[inherited_witness]}"
                            )
                        guard = f"contradiction_witness_guard_{len(guard_by_witness) + 1}"
                        witness_lines.append(
                            f"    obtain ⟨{name}, {guard}⟩ := {application}"
                        )
                        guard_by_witness[witness_variable] = guard
                        witness_arguments.extend([name, guard])
                    term = (
                        "scoped_contradiction"
                        if not scoped_terms
                        else f"scoped_contradiction_{len(scoped_terms) + 1}"
                    )
                    witness_lines.append(
                        f"    have {term} := "
                        + " ".join([scoped_fact, "contradiction_assumption", *witness_arguments])
                    )
                    scoped_terms[scoped_position] = term
                negative_term = scoped_terms.get(negative_position, negative_fact)
                positive_term = scoped_terms.get(positive_position, positive_fact)
                proof = [
                    "    intro contradiction_assumption",
                    *witness_lines,
                    f"    exact {negative_term} {positive_term}",
                ]
            mechanism = "explicit contradiction elimination"
        elif action == "ex_falso" and step["source_tag"] == "vacuous truth":
            dependency_expressions = [
                parse_mpl(str(dependency["expression"]["mpl"]))
                for dependency in dependencies
            ]
            dependency_facts = _dependency_facts(dependencies)
            contradictory_fact_pairs: list[tuple[str, str]] = []
            for negative_position, negative in enumerate(dependency_expressions):
                if not isinstance(negative, Negation):
                    continue
                for positive_position, positive in enumerate(dependency_expressions):
                    if to_mpl(negative.inner) != to_mpl(positive):
                        continue
                    pair = (
                        dependency_facts[negative_position],
                        dependency_facts[positive_position],
                    )
                    if pair not in contradictory_fact_pairs:
                        contradictory_fact_pairs.append(pair)
            assert len(contradictory_fact_pairs) == 1
            negative_fact, positive_fact = contradictory_fact_pairs[0]
            proof = [
                f"    exact False.elim ({negative_fact} {positive_fact})"
            ]
            mechanism = "explicit vacuous-truth contradiction elimination"
        elif action == "classical_reformulate_bound":
            assert len(dependencies) == 1
            source_fact = _row_reference(str(dependencies[0]["step"]))
            assert source_fact in main_prefix
            source_prefix = main_prefix[source_fact]
            outer_variables = main_unbound if namespace == "main" else ()
            output_variables = [
                variable
                for variable in outer_variables
                if variable not in source_prefix
            ]
            proof = []
            if scoped_action:
                proof.extend(
                    _scope_entry_and_specializations(
                        namespace,
                        scope_contexts,
                        scoped_prefix.get(fact, ()),
                        scope_row_guard(fact),
                        [],
                        replacements,
                    )
                )
            else:
                proof.extend(
                    f"    intro {replacements.get(variable, lean_name(variable))}"
                    for variable in outer_variables
                )
            cursor = conclusion_expression
            premise_labels: list[str] = []
            premise_index = 0
            while isinstance(cursor, Implication):
                for variable in cursor.bound_variables:
                    proof.append(
                        f"    intro {replacements.get(variable, lean_name(variable))}"
                    )
                    if variable not in source_prefix:
                        output_variables.append(variable)
                premise_index += 1
                label = f"integration_premise_{premise_index}"
                premise_labels.append(label)
                proof.append(f"    intro {label}")
                cursor = cursor.conclusion
            assert output_variables
            assert premise_labels
            source_arguments = " ".join(
                replacements.get(variable, lean_name(variable))
                for variable in source_prefix
            )
            specialized_source = (
                f"{source_fact} {source_arguments}"
                if source_arguments
                else source_fact
            )
            proof.extend(
                [
                    f"    apply ({specialized_source}).2",
                    "    intro universal_counterexample",
                    (
                        "    exact universal_counterexample "
                        + " ".join(
                            [
                                *(
                                    replacements.get(variable, lean_name(variable))
                                    for variable in output_variables
                                ),
                                *premise_labels,
                            ]
                        )
                    ),
                ]
            )
            mechanism = "explicit bounded integration reformulation"
        elif action == "equality_substitute":
            facts = _dependency_facts(dependencies)
            assert len(facts) >= 2
            source_expression = parse_mpl(
                str(dependencies[0]["expression"]["mpl"])
            )
            equality_expressions = [
                parse_mpl(str(dependency["expression"]["mpl"]))
                for dependency in dependencies[1:]
            ]
            assert all(
                isinstance(expression, Atom)
                and expression.head == "="
                and len(expression.arguments) == 2
                for expression in equality_expressions
            )
            normalized_source = map_arguments(
                source_expression,
                lambda name: replacements.get(name, name),
            )
            normalized_target = map_arguments(
                conclusion_expression,
                lambda name: replacements.get(name, name),
            )
            equivalence_classes: list[set[str]] = []
            for expression in equality_expressions:
                left, right = (
                    replacements.get(argument, argument)
                    for argument in expression.arguments
                )
                overlapping = [
                    position
                    for position, members in enumerate(equivalence_classes)
                    if left in members or right in members
                ]
                merged = {left, right}
                for position in reversed(overlapping):
                    merged.update(equivalence_classes.pop(position))
                equivalence_classes.append(merged)
            canonical = {
                member: min(members)
                for members in equivalence_classes
                for member in members
            }
            normalized_source = map_arguments(
                normalized_source,
                lambda name: canonical.get(name, name),
            )
            normalized_target = map_arguments(
                normalized_target,
                lambda name: canonical.get(name, name),
            )
            assert to_mpl(normalized_source) == to_mpl(normalized_target)
            proof = []
            if scoped_action:
                proof.extend(
                    _scope_entry_and_specializations(
                        namespace,
                        scope_contexts,
                        scoped_prefix.get(fact, ()),
                        scope_row_guard(fact),
                        [
                            _scoped_dependency(
                                dependency,
                                dependency_fact,
                                scoped_prefix,
                                scoped_witness,
                                scoped_existential,
                                scoped_existential_guard,
                            )
                            for dependency, dependency_fact in zip(
                                dependencies,
                                facts,
                                strict=True,
                            )
                        ],
                        replacements,
                    )
                )
            dependency_terms: list[str] = []
            for position, (dependency, dependency_fact) in enumerate(
                zip(dependencies, facts, strict=True),
                start=1,
            ):
                dependency_namespace = str(dependency["namespace"])
                if dependency_namespace == namespace and namespace != "main":
                    assert dependency_fact not in scoped_existential
                    dependency_terms.append(f"scoped_fact_{position}")
                    continue
                assert dependency_namespace == "main"
                prefix = main_prefix.get(dependency_fact, ())
                dependency_terms.append(
                    " ".join(
                        [
                            dependency_fact,
                            *(
                                replacements.get(variable, lean_name(variable))
                                for variable in prefix
                            ),
                        ]
                    )
                )
            proof.append(f"    have equality_source := {dependency_terms[0]}")
            for position, equality_term in enumerate(
                dependency_terms[1:],
                start=1,
            ):
                proof.extend(
                    [
                        f"    have equality_step_{position} := {equality_term}",
                        f"    cases equality_step_{position}",
                    ]
                )
            proof.append("    exact equality_source")
            mechanism = "ordered equality elimination"
        elif action == "ex_falso":
            raise AssertionError(
                "ex-falso rows must use the dedicated contradiction or "
                "vacuous-truth renderer"
            )
        elif action == "equality_symmetric":
            assert len(dependencies) == 1
            source_fact = _row_reference(str(dependencies[0]["step"]))
            if scoped_action:
                proof = [
                    *_scope_entry_and_specializations(
                        namespace,
                        scope_contexts,
                        scoped_prefix.get(fact, ()),
                        scope_row_guard(fact),
                        [
                            _scoped_dependency(
                                dependencies[0],
                                source_fact,
                                scoped_prefix,
                                scoped_witness,
                                scoped_existential,
                                scoped_existential_guard,
                            )
                        ],
                        replacements,
                    ),
                    "    exact Eq.symm scoped_fact_1",
                ]
            else:
                proof = [f"    exact Eq.symm {source_fact}"]
            mechanism = "equality symmetry"
        elif action == "inequality_symmetric":
            assert len(dependencies) == 1
            source_fact = _row_reference(str(dependencies[0]["step"]))
            if scoped_action:
                proof = [
                    *_scope_entry_and_specializations(
                        namespace,
                        scope_contexts,
                        scoped_prefix.get(fact, ()),
                        scope_row_guard(fact),
                        [
                            _scoped_dependency(
                                dependencies[0],
                                source_fact,
                                scoped_prefix,
                                scoped_witness,
                                scoped_existential,
                                scoped_existential_guard,
                            )
                        ],
                        replacements,
                    ),
                    "    exact fun equality => scoped_fact_1 (Eq.symm equality)",
                ]
            else:
                proof = [
                    f"    exact fun equality => {source_fact} (Eq.symm equality)"
                ]
            mechanism = "inequality symmetry"
        elif action == "anchor_specialize":
            assert len(dependencies) == 1
            source_fact = _row_reference(str(dependencies[0]["step"]))
            proof = [f"    exact {source_fact}"]
            mechanism = "certificate-recorded anchor copy"
        elif action == "theorem_reference":
            reference = step["theorem_reference"]
            assert isinstance(reference, dict)
            referenced_id = str(reference["theorem_id"])
            mechanism = f"explicit prior theorem reference: {referenced_id}"
            theorem_application = reference_application(reference)
            if binary_definitions is None:
                theorem_proposition = render_expression(
                    _theorem_body(conclusion_expression),
                    variable_types,
                    replacements,
                )
                lines.extend(
                    [
                        f"  -- {step['id']}: GL tag {step['source_tag']}.",
                        f"  have {fact} : {theorem_proposition} := by",
                        f"    exact {theorem_application}",
                    ]
                )
            else:
                lines.extend(
                    [
                        f"  -- {step['id']}: GL tag {step['source_tag']}.",
                        f"  have {fact} := {theorem_application}",
                    ]
                )
            scope_facts.append(fact)
            dispositions.append(_manifest_disposition(step, owner, fact, mechanism))
            continue
        elif action == "or_construct":
            # Two parents normally; one when the OR is licensed from a
            # single proved direction (D-217). Either shape has a first parent
            # that derives the final disjunct from negations of the preceding
            # disjuncts.
            assert len(dependencies) in (1, 2)
            definition = step["constructed_or_definition"]
            assert isinstance(definition, dict)
            head = str(definition["head"])
            proposition = render_expression(
                _theorem_body(conclusion_expression),
                variable_types,
                replacements,
            )
            parent_names: list[str] = []
            for index, dependency in enumerate(dependencies, start=1):
                reference = dependency["theorem_reference"]
                assert isinstance(reference, dict)
                parent_name = f"or_parent_{index}"
                parent_names.append(parent_name)
                parent_application = reference_application(reference)
                lines.append(
                    f"  have {parent_name} := {parent_application}"
                )
                scope_facts.append(parent_name)
            definition_arguments = {
                f"u_{position}": str(argument)
                for position, argument in enumerate(
                    definition["arguments"],
                    start=1,
                )
            }
            definition_elements = [
                map_arguments(
                    parse_mpl(str(element)),
                    lambda name: definition_arguments.get(name, name),
                )
                for element in definition["elements"]
            ]
            assert len(definition_elements) >= 2
            rendered_elements = [
                render_expression(element, variable_types, replacements)
                for element in definition_elements
            ]
            theorem_body = _theorem_body(conclusion_expression)
            theorem_arguments: list[str] = []
            proof: list[str] = ["    classical"]
            cursor = theorem_body
            premise_index = 0
            while isinstance(cursor, Implication):
                for variable in cursor.bound_variables:
                    rendered_variable = replacements.get(variable, lean_name(variable))
                    proof.append(f"    intro {rendered_variable}")
                    theorem_arguments.append(rendered_variable)
                premise_index += 1
                premise = f"or_parent_premise_{premise_index}"
                proof.append(f"    intro {premise}")
                theorem_arguments.append(premise)
                cursor = cursor.conclusion
            assert isinstance(cursor, Atom) and cursor.head == head
            parent_application = " ".join(
                [
                    parent_names[0],
                    *theorem_arguments,
                    *(
                        f"or_case_{index}"
                        for index in range(1, len(rendered_elements))
                    ),
                ]
            )
            branch_proofs = [
                *(f"or_case_{index}" for index in range(1, len(rendered_elements))),
                f"({parent_application})",
            ]
            constructed_branches: list[str] = []
            for branch_index, branch_proof in enumerate(branch_proofs):
                constructed = branch_proof
                for disjunction_size in range(2, len(branch_proofs) + 1):
                    if branch_index < disjunction_size - 1:
                        constructed = f"Or.inl ({constructed})"
                    elif branch_index == disjunction_size - 1:
                        constructed = f"Or.inr ({constructed})"
                constructed_branches.append(constructed)
            proof.append(f"    simp only [{definition_constant(head)}]")
            branch_indent = "    "
            for index in range(1, len(rendered_elements)):
                proof.extend(
                    [
                        f"{branch_indent}by_cases or_case_{index} : "
                        f"{rendered_elements[index - 1]}",
                        f"{branch_indent}· exact {constructed_branches[index - 1]}",
                        f"{branch_indent}·",
                    ]
                )
                branch_indent += "  "
            proof.append(f"{branch_indent}exact {constructed_branches[-1]}")
            mechanism = f"explicit constructed OR via {lean_constant(head)}"
        elif action == "or_eliminate_theorem":
            assert namespace == "main" and len(dependencies) == 3
            definition_head = str(step["or_definition_head"])
            assert anchor_head == "AnchorFTA" and definition_head == "or3"
            theorem_body = _theorem_body(conclusion_expression)
            assert isinstance(theorem_body, Implication)
            assert len(theorem_body.bound_variables) == 2
            assert isinstance(theorem_body.premise, Atom)
            assert theorem_body.premise.head == "preorder"
            assert isinstance(theorem_body.conclusion, Implication)
            assert not theorem_body.conclusion.bound_variables
            assert isinstance(theorem_body.conclusion.premise, Atom)
            assert theorem_body.conclusion.premise.head == "preorder"
            assert isinstance(theorem_body.conclusion.conclusion, Atom)
            assert theorem_body.conclusion.conclusion.head == "="
            left, right = theorem_body.bound_variables
            domain, relation, first_left, first_right = (
                theorem_body.premise.arguments
            )
            second_domain, second_relation, second_left, second_right = (
                theorem_body.conclusion.premise.arguments
            )
            equality_left, equality_right = (
                theorem_body.conclusion.conclusion.arguments
            )
            assert (first_left, first_right) == (left, right)
            assert (second_left, second_right) == (right, left)
            assert (domain, relation) == (second_domain, second_relation)
            assert (equality_left, equality_right) == (left, right)
            proposition = render_expression(
                theorem_body,
                variable_types,
                replacements,
            )
            elimination_parents: list[str] = []
            for index, dependency in enumerate(dependencies, start=1):
                reference = dependency["theorem_reference"]
                assert isinstance(reference, dict)
                parent_application = reference_application(reference)
                lines.append(
                    f"  have or_elim_parent_{index} := {parent_application}"
                )
                elimination_parents.append(f"or_elim_parent_{index}")
                scope_facts.append(f"or_elim_parent_{index}")
            left_name = replacements.get(left, lean_name(left))
            right_name = replacements.get(right, lean_name(right))
            domain_name = replacements.get(domain, lean_name(domain))
            relation_name = replacements.get(relation, lean_name(relation))
            # The anchor projections and the closure rule's compact name are
            # read off the compiled definitions of this corpus, never assumed:
            # a fresh compilation numbers the inner compacts differently and
            # may order the or's disjuncts either way.
            assert binary_definitions is not None
            anchor_elements = [
                parse_mpl(str(element))
                for element in binary_definitions[anchor_head]["elements"]
            ]
            natural_numbers_index = next(
                index
                for index, element in enumerate(anchor_elements)
                if isinstance(element, Atom) and element.head == "NaturalNumbers"
            )
            natural_numbers_elements = [
                parse_mpl(str(element))
                for element in binary_definitions["NaturalNumbers"]["elements"]
            ]
            multiplication_index = next(
                index
                for index, element in enumerate(natural_numbers_elements)
                if isinstance(element, Atom)
                and element.head == "fXYZ"
                and element.arguments[0] == "u_5"
            )
            fxyz_elements = [
                parse_mpl(str(element))
                for element in binary_definitions["fXYZ"]["elements"]
            ]
            output_closure_index, output_closure_head = next(
                (index, element.head)
                for index, element in enumerate(fxyz_elements)
                if isinstance(element, Atom)
                and [
                    str(part)
                    for part in binary_definitions[element.head]["elements"]
                ] == ["(in3[1,2,3,u_1])", "(in[3,u_2])"]
            )
            or_elements = [
                parse_mpl(str(element))
                for element in binary_definitions[definition_head]["elements"]
            ]
            assert len(or_elements) == 2
            equality_first = isinstance(or_elements[0], Atom) and or_elements[0].head == "="
            assert equality_first or (
                isinstance(or_elements[1], Atom) and or_elements[1].head == "="
            ), f"{definition_head} has no equality disjunct"
            case_pattern = (
                "zero_case | positive_case" if equality_first else "positive_case | zero_case"
            )
            # One bullet per disjunct, in the disjunction's own order.
            zero_bullet = (
                f"    · exact or_elim_parent_1 {left_name} {right_name} "
                "or_elimination_premise_1 or_elimination_premise_2 zero_case"
            )
            positive_bullet = (
                f"    · exact or_elim_parent_2 {left_name} {right_name} "
                "or_elimination_premise_1 or_elimination_premise_2 positive_case"
            )
            proof = [
                "    classical",
                *_proof_introductions(
                    theorem_body,
                    replacements,
                    "or_elimination_premise",
                ),
                "    have preorder_witness := or_elimination_premise_1",
                (
                    "    obtain ⟨preorder_middle, preorder_middle_in_N, "
                    "preorder_relation⟩ := "
                    "existsAndOfNotForallImpNot preorder_witness"
                ),
                (
                    f"    have natural_numbers : gl_NaturalNumbers "
                    f"{domain_name} zero succ add {relation_name} := "
                    f"anchor{_left_associated_projection(len(anchor_elements), natural_numbers_index)}"
                ),
                (
                    f"    have multiplication_structure : gl_fXYZ "
                    f"{relation_name} {domain_name} {domain_name} "
                    f"{domain_name} := natural_numbers"
                    f"{_left_associated_projection(len(natural_numbers_elements), multiplication_index)}"
                ),
                (
                    f"    have multiplication_output_closed : "
                    f"{lean_constant(output_closure_head)} {relation_name} {domain_name} := "
                    f"multiplication_structure"
                    f"{_left_associated_projection(len(fxyz_elements), output_closure_index)}"
                ),
                (
                    f"    have right_in_domain : {domain_name} {right_name} := "
                    f"multiplication_output_closed {left_name} "
                    f"preorder_middle {right_name} preorder_relation"
                ),
                (
                    f"    have or_elim_cases := or_elim_parent_3 "
                    f"{right_name} right_in_domain"
                ),
                f"    simp only [{definition_constant(definition_head)}] at or_elim_cases",
                f"    rcases or_elim_cases with {case_pattern}",
                *(
                    [zero_bullet, positive_bullet]
                    if equality_first
                    else [positive_bullet, zero_bullet]
                ),
            ]
            mechanism = (
                f"pre-split merge (or elimination) via {lean_constant(definition_head)}"
            )
        elif action == "or_branch":
            assert namespace != "main" and len(dependencies) == 1
            assert step["scope_namespace"] == namespace
            context_template = parse_mpl(
                str(scope_contexts[namespace]["template"]["mpl"])
            )
            assert isinstance(context_template, Implication)
            assert not context_template.bound_variables
            assert to_mpl(context_template.premise) == conclusion_mpl
            assert to_mpl(context_template.conclusion) == conclusion_mpl
            proof = [
                *_scope_entry_and_specializations(
                    namespace,
                    scope_contexts,
                    scoped_prefix.get(fact, ()),
                    scope_row_guard(fact),
                    [],
                    replacements,
                ),
                "    exact scope_premise_1",
            ]
            mechanism = "explicit OR branch identity"
        elif action == "or_eliminate":
            assert namespace == "main" and len(dependencies) >= 3
            definition_head = str(step["or_definition_head"])
            assert binary_definitions is not None
            or_expression = parse_mpl(str(dependencies[0]["expression"]["mpl"]))
            assert isinstance(or_expression, Atom)
            assert or_expression.head == definition_head
            definition = binary_definitions[definition_head]
            assert isinstance(definition, dict)
            definition_arguments = {
                f"u_{position}": argument
                for position, argument in enumerate(or_expression.arguments, start=1)
            }
            branch_expressions = [
                map_arguments(
                    parse_mpl(str(element)),
                    lambda name: definition_arguments.get(name, name),
                )
                for element in definition["elements"]
            ]
            assert len(branch_expressions) == 2
            branch_dependencies: list[dict[str, object]] = []
            for branch_expression in branch_expressions:
                matches: list[dict[str, object]] = []
                for dependency in dependencies[1:]:
                    dependency_expression = parse_mpl(
                        str(dependency["expression"]["mpl"])
                    )
                    dependency_namespace = str(dependency["namespace"])
                    if dependency_namespace == "main":
                        if (
                            isinstance(dependency_expression, Negation)
                            and to_mpl(dependency_expression.inner)
                            == to_mpl(branch_expression)
                        ):
                            matches.append(dependency)
                        continue
                    context_template = parse_mpl(
                        str(
                            scope_contexts[dependency_namespace]["template"]["mpl"]
                        )
                    )
                    assert isinstance(context_template, Implication)
                    if to_mpl(context_template.premise) == to_mpl(branch_expression):
                        matches.append(dependency)
                assert len(matches) == 1
                branch_dependencies.append(matches[0])

            or_fact = _row_reference(str(dependencies[0]["step"]))
            proof = [
                "    classical",
                f"    have or_cases := {or_fact}",
                f"    simp only [{definition_constant(definition_head)}] at or_cases",
                "    rcases or_cases with or_branch_1 | or_branch_2",
            ]
            for branch_position, (
                branch_expression,
                branch_dependency,
            ) in enumerate(
                zip(branch_expressions, branch_dependencies, strict=True),
                start=1,
            ):
                branch_fact = _row_reference(str(branch_dependency["step"]))
                dependency_expression = parse_mpl(
                    str(branch_dependency["expression"]["mpl"])
                )
                proof.append(f"    · have branch_compact := or_branch_{branch_position}")
                if str(branch_dependency["namespace"]) == "main":
                    assert isinstance(dependency_expression, Negation)
                    proof.append(
                        f"      exact False.elim ({branch_fact} branch_compact)"
                    )
                    continue
                assert not scoped_prefix.get(branch_fact, ())
                witnesses = scoped_witness.get(branch_fact, ())
                branch_application = f"{branch_fact} branch_compact"
                if witnesses:
                    assert len(witnesses) == 1
                    witness = witnesses[0]
                    assert isinstance(branch_expression, Atom)
                    existence_definition = binary_definitions[branch_expression.head]
                    assert isinstance(existence_definition, dict)
                    assert existence_definition["category"] == "existence"
                    existence_arguments = {
                        f"u_{position}": argument
                        for position, argument in enumerate(
                            branch_expression.arguments,
                            start=1,
                        )
                    }
                    local_placeholders = {
                        argument
                        for element in existence_definition["elements"]
                        for atom in iter_atoms(parse_mpl(str(element)))
                        for argument in atom.arguments
                        if not argument.startswith("u_")
                    }
                    assert len(local_placeholders) == 1
                    local_placeholder = next(iter(local_placeholders))
                    full_witness_elements = [
                        map_arguments(
                            parse_mpl(str(element)),
                            lambda name: (
                                witness
                                if name == local_placeholder
                                else existence_arguments.get(name, name)
                            ),
                        )
                        for element in existence_definition["elements"]
                    ]
                    full_witness_bundle = (
                        full_witness_elements[0]
                        if len(full_witness_elements) == 1
                        else Conjunction(tuple(full_witness_elements))
                    )
                    required_bundle = scope_witness_bundle.get(
                        (str(branch_dependency["namespace"]), witness)
                    )
                    assert required_bundle is not None
                    projection = "".join(
                        ".1" if direction == 1 else ".2"
                        for direction in _conjunct_path(
                            full_witness_bundle,
                            required_bundle,
                        )
                    )
                    witness_name = replacements.get(witness, lean_name(witness))
                    witness_bundle = f"or_witness_bundle_{branch_position}"
                    proof.extend(
                        [
                            (
                                f"      obtain ⟨{witness_name}, {witness_bundle}⟩ "
                                ":= existsAndOfNotForallImpNot branch_compact"
                            ),
                        ]
                    )
                    branch_application += (
                        f" {witness_name} {witness_bundle}{projection}"
                    )
                if to_mpl(dependency_expression) == conclusion_mpl:
                    proof.append(f"      exact {branch_application}")
                else:
                    assert (
                        isinstance(dependency_expression, Negation)
                        and to_mpl(dependency_expression.inner)
                        == to_mpl(branch_expression)
                    )
                    proof.append(
                        f"      exact False.elim "
                        f"(({branch_application}) branch_compact)"
                    )
            mechanism = f"explicit OR branch elimination via {lean_constant(definition_head)}"
        elif action == "definition_compact":
            assert namespace == "main" and len(dependencies) == 1
            assert isinstance(conclusion_expression, Atom)
            reference = dependencies[0]["theorem_reference"]
            assert isinstance(reference, dict)
            if (
                anchor_head == "AnchorFTA"
                and not conclusion_expression.arguments
                and reference.get("certificate_id") is None
            ):
                source_theorem = parse_mpl(str(reference["theorem"]["mpl"]))
                assert isinstance(source_theorem, Implication)
                assert source_theorem.bound_variables == (
                    "N", "i0", "s", "+", "*", "i1", "i2", "id"
                )
                assert isinstance(source_theorem.premise, Atom)
                assert source_theorem.premise.head == "AnchorFTA"
                compiled_context = [
                    "compiled_N",
                    "compiled_zero",
                    "compiled_succ",
                    "compiled_add",
                    "compiled_mul",
                    "compiled_one",
                    "compiled_two",
                    "compiled_identity",
                ]
                theorem_application = " ".join(
                    [
                        str(reference["theorem_id"]),
                        *compiled_context,
                        "compiled_anchor",
                        "relationalInduction",
                        *external_theorem_arguments.get(
                            str(reference["theorem_id"]), []
                        ),
                    ]
                )
                proof = [
                    f"    simp only [{expansion_rules(conclusion_expression.head)}]",
                    f"    intro {' '.join(compiled_context)}",
                    "    intro compiled_anchor",
                    f"    exact {theorem_application}",
                ]
                mechanism = (
                    "explicit global theorem compaction via "
                    f"{lean_constant(conclusion_expression.head)}"
                )
            else:
                source_fact = f"source_{fact}"
                theorem_application = reference_application(reference)
                lines.append(
                    f"  have {source_fact} := {theorem_application}"
                )
                scope_facts.append(source_fact)
                proof = [
                    f"    simpa only [{expansion_rules(conclusion_expression.head)}] using {source_fact}"
                ]
                mechanism = (
                    f"definition compaction via "
                    f"{lean_constant(conclusion_expression.head)}"
                )
        elif action == "variable_alias":
            assert not dependencies
            proof = ["    rfl"]
            mechanism = "certificate-checked local variable alias"
        elif action == "integration_premise":
            assert len(dependencies) == 1
            assert namespace != "main"
            assert step["scope_namespace"] == namespace
            assert dependencies[0]["namespace"] == "main"
            context_template = parse_mpl(
                str(scope_contexts[namespace]["template"]["mpl"])
            )
            dependency_expression = parse_mpl(
                str(dependencies[0]["expression"]["mpl"])
            )
            assert to_mpl(context_template) == to_mpl(dependency_expression)
            premise_layers: list[Expression] = []
            cursor = context_template
            while isinstance(cursor, Implication):
                premise_layers.append(cursor.premise)
                cursor = cursor.conclusion
            matching_premises = [
                position
                for position, premise in enumerate(premise_layers, start=1)
                if to_mpl(premise) == to_mpl(conclusion_expression)
            ]
            assert len(matching_premises) == 1
            proof = [
                *_scope_entry_and_specializations(
                    namespace,
                    scope_contexts,
                    scoped_prefix.get(fact, ()),
                    scope_row_guard(fact),
                    [],
                    replacements,
                ),
                f"    exact scope_premise_{matching_premises[0]}",
            ]
            mechanism = "explicit scoped premise introduction"
        elif action in {
            "classical_reformulate_empty",
            "integration_conjoin",
            "integration_discharge",
        }:
            assert len(dependencies) == 1
            if action == "integration_discharge":
                assert namespace == "main"
                context = step["scope_context"]
                assert isinstance(context, dict)
                assert dependencies[0]["namespace"] == context["namespace"]
            if action == "integration_conjoin":
                assert "reformulation_definition" in step
                proof = [
                    *(
                        f"    intro {replacements.get(variable, lean_name(variable))}"
                        for variable in main_unbound
                    ),
                ]
                cursor = conclusion_expression
                premise_labels: list[str] = []
                premise_index = 0
                while isinstance(cursor, Implication):
                    for variable in cursor.bound_variables:
                        proof.append(
                            f"    intro {replacements.get(variable, lean_name(variable))}"
                        )
                    premise_index += 1
                    label = f"integration_premise_{premise_index}"
                    premise_labels.append(label)
                    proof.append(f"    intro {label}")
                    cursor = cursor.conclusion
                assert len(premise_labels) >= 2
                proof.append(
                    f"    simp only [{lean_constant(str(step['reformulation_definition']))}]"
                )
                conjunction = premise_labels[0]
                for label in premise_labels[1:]:
                    conjunction = f"⟨{conjunction}, {label}⟩"
                proof.append(f"    exact {conjunction}")
            elif action == "integration_discharge":
                source_fact = _row_reference(str(dependencies[0]["step"]))
                proof = [
                    f"    simpa only [{definition_constant(str(step['scope_context']['definition_head']))}] "
                    f"using {source_fact}"
                ]
            elif action == "classical_reformulate_empty":
                source_fact = _row_reference(str(dependencies[0]["step"]))
                assert source_fact in main_prefix
                source_prefix = main_prefix[source_fact]
                output_variables = tuple(
                    variable
                    for variable in main_unbound
                    if variable not in source_prefix
                )
                assert output_variables
                proof = [
                    *(
                        f"    intro {replacements.get(variable, lean_name(variable))}"
                        for variable in main_unbound
                    ),
                ]
                cursor = conclusion_expression
                premise_labels: list[str] = []
                premise_index = 0
                while isinstance(cursor, Implication):
                    for variable in cursor.bound_variables:
                        proof.append(
                            f"    intro {replacements.get(variable, lean_name(variable))}"
                        )
                    premise_index += 1
                    label = f"integration_premise_{premise_index}"
                    premise_labels.append(label)
                    proof.append(f"    intro {label}")
                    cursor = cursor.conclusion
                assert premise_labels
                assert isinstance(cursor, Atom)
                assert cursor.head == step["reformulation_definition"]
                source_arguments = " ".join(
                    replacements.get(variable, lean_name(variable))
                    for variable in source_prefix
                )
                specialized_source = (
                    f"{source_fact} {source_arguments}"
                    if source_arguments
                    else source_fact
                )
                proof.extend(
                    [
                        f"    apply ({specialized_source}).2",
                        "    intro universal_counterexample",
                        (
                            "    exact universal_counterexample "
                            + " ".join(
                                [
                                    *(
                                        replacements.get(variable, lean_name(variable))
                                        for variable in output_variables
                                    ),
                                    *premise_labels,
                                ]
                            )
                        ),
                    ]
                )
            mechanism = {
                "classical_reformulate_empty": "classical empty-bound reformulation",
                "integration_conjoin": "integration conjunction reformulation",
                "integration_discharge": "explicit scope discharge",
            }[action]
        elif action == "theorem_reformulate":
            raise AssertionError(
                "theorem reformulation must use the dedicated Lean renderer"
            )
        else:
            raise AssertionError(f"unsupported Lean certificate action {action!r}")

        lines.extend(
            [
                f"  -- {step['id']}: GL tag {step['source_tag']}.",
                f"  have {fact} : {proposition} := by",
                *proof,
            ]
        )
        scope_facts.append(fact)
        dispositions.append(_manifest_disposition(step, owner, fact, mechanism))

    assert len(dispositions) == len(steps)
    return lines, dispositions


def _external_reference_interfaces(
    chapters: list[dict[str, object]],
) -> tuple[list[tuple[str, str]], dict[tuple[str, int], str]]:
    """
    @brief Collect the external theorem facts one FTA proof actually cites.
    @details
    Schema-3 external-certificate references already carry their hash-validated
    target theorem, including any structurally checked adaptation. This routine
    renders that complete target theorem as an explicit Lean parameter and gives
    every use of the same certificate/source pair one stable current-context
    specialization. Keeping the validated anchor implication makes the parameter
    reusable when a nullary compilation row reconstructs a globally quantified
    FTA theorem. Internal selected theorem references are omitted because their
    certificate id is absent.
    @param chapters Certificate chapters rendered inside one public theorem or
    private induction helper.
    @return Ordered ``(fact name, proposition)`` binders and a reference-key to
    fact-name map for the row renderer.
    @invariant Repeated uses of one certificate/source pair have one identical
    target proposition.
    @see _theorem_header
    """

    interfaces: list[tuple[str, str]] = []
    fact_by_reference: dict[tuple[str, int], str] = {}
    proposition_by_reference: dict[tuple[str, int], str] = {}
    for chapter in chapters:
        variable_types = chapter["variable_types"]
        steps = chapter["steps"]
        assert isinstance(variable_types, dict) and isinstance(steps, list)
        for step in steps:
            references = []
            direct_reference = step.get("theorem_reference")
            if direct_reference is not None:
                references.append(direct_reference)
            dependencies = step["dependencies"]
            assert isinstance(dependencies, list)
            references.extend(
                dependency["theorem_reference"]
                for dependency in dependencies
                if "theorem_reference" in dependency
            )
            for reference in references:
                assert isinstance(reference, dict)
                certificate_id = reference.get("certificate_id")
                if certificate_id is None:
                    continue
                key = (str(certificate_id), int(reference["source_index"]))
                fact = (
                    f"external_{lean_name(str(certificate_id))}_"
                    f"{int(reference['source_index']):03d}"
                )
                target_types = reference["target_variable_types"]
                assert isinstance(target_types, dict)
                target = parse_mpl(str(reference["theorem"]["mpl"]))
                proposition = render_expression(
                    target,
                    target_types,
                    CONTEXT_NAMES,
                )
                if corpus_kind(str(certificate_id)) == "peano":
                    fact_application = (
                        f"{fact} N zero succ add mul one "
                        "(anchorPeanoOfFTA N zero succ add mul one two identity anchor)"
                    )
                else:
                    assert corpus_kind(str(certificate_id)) == "gauss"
                    fact_application = (
                        f"{fact} N zero succ add mul one two identity "
                        "(anchorGaussOfFTA N zero succ add mul one two identity anchor)"
                    )
                if key in fact_by_reference:
                    assert fact_by_reference[key] == fact_application
                    assert proposition_by_reference[key] == proposition
                    continue
                fact_by_reference[key] = fact_application
                proposition_by_reference[key] = proposition
                interfaces.append((fact, proposition))
    return interfaces, fact_by_reference


def _theorem_header(
    theorem: dict[str, object],
    replacements: dict[str, str],
    anchor_head: str = "AnchorPeano",
    external_interfaces: list[tuple[str, str]] | None = None,
) -> tuple[list[str], list[str], dict[str, str], Expression, list[str]]:
    """
    @brief Render one public theorem declaration and its introduction sequence.
    @details
    The anchor and relational induction rule are explicit parameters. An FTA
    theorem additionally receives the exact hash-validated external facts its
    chapters cite. Nested MPL binders and premises are introduced in source order
    with stable names.
    @param theorem Neutral certificate theorem.
    @param replacements Chapter-local identifier aliases.
    @param anchor_head AnchorPeano or AnchorGauss context selector.
    @param external_interfaces Ordered external fact names and target propositions.
    @return Declaration lines, proof introductions, assumption map, terminal
    conclusion, and theorem-premise labels.
    """

    variable_types = theorem["variable_types"]
    assert isinstance(variable_types, dict)
    body = _theorem_body(parse_mpl(str(theorem["theorem"]["mpl"])))
    lines = [f"theorem {theorem['id']}", *_context_signature(anchor_head)]
    lines.extend(
        f"    ({fact} : {proposition})"
        for fact, proposition in (external_interfaces or [])
    )
    lines.extend(
        [
            f"    : {render_expression(body, variable_types, replacements)} := by",
        ]
    )
    introductions: list[str] = []
    assumption_by_mpl = {
        ANCHOR_MPL[anchor_head]: "anchor",
    }
    labels: list[str] = []
    current = body
    premise_index = 0
    while isinstance(current, Implication):
        for bound in current.bound_variables:
            introductions.append(
                f"  intro {replacements.get(bound, lean_name(bound))}"
            )
        premise_index += 1
        label = f"premise_{premise_index}"
        introductions.append(f"  intro {label}")
        premise_mpl = to_mpl(current.premise)
        assert premise_mpl not in assumption_by_mpl
        assumption_by_mpl[premise_mpl] = label
        labels.append(label)
        current = current.conclusion
    return lines, introductions, assumption_by_mpl, current, labels


def _proof_introductions(
    expression: Expression,
    replacements: dict[str, str],
    label_prefix: str,
) -> list[str]:
    """
    @brief Render introductions that reduce an implication property to its head.
    @details
    Explicitly introducing property binders and premises prevents first-order
    automation from exploring a large classical negation of the full property.
    @param expression Implication property to introduce.
    @param replacements MPL token to Lean identifier mapping.
    @param label_prefix Prefix for generated premise names.
    @return Lean tactic lines in binder and premise order.
    """

    lines: list[str] = []
    current = expression
    premise_index = 0
    while isinstance(current, Implication):
        for bound in current.bound_variables:
            lines.append(f"    intro {replacements.get(bound, lean_name(bound))}")
        premise_index += 1
        lines.append(f"    intro {label_prefix}_{premise_index}")
        current = current.conclusion
    return lines


def _chapter_manifest(
    chapter: dict[str, object],
    dispositions: list[dict[str, object]],
) -> dict[str, object]:
    """
    @brief Form one theorem-manifest chapter record.
    @details
    Source row count and disposition count remain separately recorded so the
    final writer can assert total row coverage.
    @param chapter Neutral certificate chapter.
    @param dispositions Emitted Lean disposition for every chapter row.
    @return Manifest chapter record.
    """

    steps = chapter["steps"]
    assert isinstance(steps, list)
    return {
        "role": chapter["role"],
        "source_chapter": chapter["source_file"],
        "source_row_count": len(steps),
        "dispositions": dispositions,
    }


def _render_direct_theorem(
    theorem: dict[str, object],
    anchor_head: str = "AnchorPeano",
    binary_definitions: dict[str, object] | None = None,
) -> tuple[list[str], dict[str, object]]:
    """
    @brief Render one direct selected theorem and all of its row facts.
    @details
    An extra chapter assumption is accepted only when it is exactly the reductio
    counterpart of the terminal theorem conclusion.
    @param theorem Neutral direct-theorem record.
    @param anchor_head AnchorPeano or AnchorGauss context selector.
    @param binary_definitions Compiled definitions used by schema-3 Gauss rows.
    @return Lean source lines and theorem-level manifest record.
    """

    # The proved-not-broadcast tier's chapter is an ordinary direct walk;
    # its Lean rendering is identical (the level verdict is a GL-side
    # registration policy, not a proof-content difference).
    assert theorem["method"] in ("direct", "proved not broadcast")
    chapters = theorem["chapters"]
    assert isinstance(chapters, list) and len(chapters) == 1
    chapter = chapters[0]
    assert isinstance(chapter, dict)
    replacements = _chapter_replacements(chapter)
    external_interfaces = (
        theorem.get("_external_interfaces", [])
        if anchor_head == "AnchorFTA"
        else []
    )
    external_facts = (
        chapter.get("_external_reference_facts", {})
        if anchor_head == "AnchorFTA"
        else {}
    )
    external_theorem_arguments = (
        chapter.get("_external_theorem_arguments", {})
        if anchor_head == "AnchorFTA"
        else {}
    )
    assert isinstance(external_interfaces, list)
    assert isinstance(external_facts, dict)
    assert isinstance(external_theorem_arguments, dict)
    lines, introductions, assumption_map, conclusion, _ = _theorem_header(
        theorem,
        replacements,
        anchor_head,
        external_interfaces,
    )
    lines.extend(introductions)
    reductio_expression = (
        conclusion.inner if isinstance(conclusion, Negation) else Negation(conclusion)
    )
    reductio_mpl = to_mpl(reductio_expression)
    reductio = False
    steps = chapter["steps"]
    assert isinstance(steps, list) and steps
    for step in steps:
        if step["action"] != "assume":
            continue
        if step["namespace"] != "main":
            continue
        assumption_mpl = str(step["conclusion"]["mpl"])
        if assumption_mpl in assumption_map:
            continue
        assert assumption_mpl == reductio_mpl, (
            f"direct proof has unstated assumption {assumption_mpl!r}"
        )
        assert not reductio
        assumption_map[assumption_mpl] = "reductio"
        reductio = True

    if reductio:
        if isinstance(conclusion, Negation):
            lines.append("  intro reductio")
        else:
            lines.extend(["  apply Classical.byContradiction", "  intro reductio"])
    theorem_body = _theorem_body(parse_mpl(str(theorem["theorem"]["mpl"])))
    ambient_variables: set[str] = set()
    cursor = theorem_body
    while isinstance(cursor, Implication):
        ambient_variables.update(cursor.bound_variables)
        cursor = cursor.conclusion
    step_lines, dispositions = _render_chapter_steps(
        chapter,
        str(theorem["id"]),
        assumption_map,
        replacements,
        anchor_head,
        ambient_variables,
        binary_definitions,
        external_facts,
        external_theorem_arguments,
    )
    lines.extend(step_lines)
    goal_fact = _row_reference(str(steps[-1]["id"]))
    assert str(steps[-1]["conclusion"]["mpl"]) == to_mpl(conclusion)
    if reductio:
        if isinstance(conclusion, Negation):
            lines.append(f"  exact {goal_fact} reductio")
        else:
            lines.append(f"  exact reductio {goal_fact}")
    else:
        lines.append(f"  exact {goal_fact}")
    lines.append("")
    return lines, {
        "id": theorem["id"],
        "source_index": theorem["source_index"],
        "method": theorem["method"],
        "chapters": [_chapter_manifest(chapter, dispositions)],
    }


def _render_or_theorem(
    theorem: dict[str, object],
    anchor_head: str = "AnchorPeano",
    binary_definitions: dict[str, object] | None = None,
) -> tuple[list[str], dict[str, object]]:
    """
    @brief Render one constructed OR theorem and its explicit parent edges.
    @details
    The row renderer proves the complete implication body from the two selected
    parent theorems and the checked compiled OR definition.
    @param theorem Neutral OR-theorem record.
    @param anchor_head Supported exported-anchor context selector.
    @param binary_definitions Compiled definitions used by schema-3 proof rows.
    @return Lean source lines and theorem-level manifest record.
    """

    assert theorem["method"] == "or theorem"
    chapters = theorem["chapters"]
    assert isinstance(chapters, list) and len(chapters) == 1
    chapter = chapters[0]
    assert isinstance(chapter, dict)
    chapter = dict(chapter)
    theorem_variable_types = theorem["variable_types"]
    chapter_variable_types = chapter["variable_types"]
    assert isinstance(theorem_variable_types, dict)
    assert isinstance(chapter_variable_types, dict)
    chapter["variable_types"] = {
        **theorem_variable_types,
        **chapter_variable_types,
    }
    steps = chapter["steps"]
    assert isinstance(steps, list) and len(steps) == 1
    assert steps[0]["action"] == "or_construct"
    replacements = _chapter_replacements(chapter)
    external_interfaces = (
        theorem.get("_external_interfaces", [])
        if anchor_head == "AnchorFTA"
        else []
    )
    external_facts = (
        chapter.get("_external_reference_facts", {})
        if anchor_head == "AnchorFTA"
        else {}
    )
    external_theorem_arguments = (
        chapter.get("_external_theorem_arguments", {})
        if anchor_head == "AnchorFTA"
        else {}
    )
    assert isinstance(external_interfaces, list)
    assert isinstance(external_facts, dict)
    assert isinstance(external_theorem_arguments, dict)
    lines, introductions, assumption_map, _, _ = _theorem_header(
        theorem,
        replacements,
        anchor_head,
        external_interfaces,
    )
    lines.extend(introductions)
    theorem_body = _theorem_body(parse_mpl(str(theorem["theorem"]["mpl"])))
    ambient_variables: set[str] = set()
    cursor = theorem_body
    while isinstance(cursor, Implication):
        ambient_variables.update(cursor.bound_variables)
        cursor = cursor.conclusion
    step_lines, dispositions = _render_chapter_steps(
        chapter,
        str(theorem["id"]),
        assumption_map,
        replacements,
        anchor_head,
        ambient_variables,
        binary_definitions,
        external_facts,
        external_theorem_arguments,
    )
    lines.extend(step_lines)
    goal_fact = _row_reference(str(steps[0]["id"]))
    assert goal_fact
    lines.extend([f"  solve_by_elim [{goal_fact}]", ""])
    return lines, {
        "id": theorem["id"],
        "source_index": theorem["source_index"],
        "method": theorem["method"],
        "chapters": [_chapter_manifest(chapter, dispositions)],
    }


def _render_or_elimination_theorem(
    theorem: dict[str, object],
    anchor_head: str = "AnchorPeano",
    binary_definitions: dict[str, object] | None = None,
) -> tuple[list[str], dict[str, object]]:
    """
    @brief Render one pre-split-merge (or elimination) theorem.
    @details
    The single-row chapter proves the merged implication body from the two
    guard-variant theorems and the licensing constructed-or theorem — the
    row renderer brings all three into scope and closes by unfolding the
    or definition. Mirror of ``_render_or_theorem`` with the
    ``or_eliminate_theorem`` action.
    @param theorem Neutral or-elimination record.
    @param anchor_head Supported exported-anchor context selector.
    @param binary_definitions Compiled definitions used by schema-3 proof rows.
    @return Lean source lines and theorem-level manifest record.
    """

    assert theorem["method"] == "or elimination"
    chapters = theorem["chapters"]
    assert isinstance(chapters, list) and len(chapters) == 1
    chapter = chapters[0]
    assert isinstance(chapter, dict)
    chapter = dict(chapter)
    theorem_variable_types = theorem["variable_types"]
    chapter_variable_types = chapter["variable_types"]
    assert isinstance(theorem_variable_types, dict)
    assert isinstance(chapter_variable_types, dict)
    chapter["variable_types"] = {
        **theorem_variable_types,
        **chapter_variable_types,
    }
    steps = chapter["steps"]
    assert isinstance(steps, list) and len(steps) == 1
    assert steps[0]["action"] == "or_eliminate_theorem"
    replacements = _chapter_replacements(chapter)
    external_interfaces = (
        theorem.get("_external_interfaces", [])
        if anchor_head == "AnchorFTA"
        else []
    )
    external_facts = (
        chapter.get("_external_reference_facts", {})
        if anchor_head == "AnchorFTA"
        else {}
    )
    external_theorem_arguments = (
        chapter.get("_external_theorem_arguments", {})
        if anchor_head == "AnchorFTA"
        else {}
    )
    assert isinstance(external_interfaces, list)
    assert isinstance(external_facts, dict)
    assert isinstance(external_theorem_arguments, dict)
    lines, introductions, assumption_map, _, _ = _theorem_header(
        theorem,
        replacements,
        anchor_head,
        external_interfaces,
    )
    lines.extend(introductions)
    theorem_body = _theorem_body(parse_mpl(str(theorem["theorem"]["mpl"])))
    ambient_variables: set[str] = set()
    cursor = theorem_body
    while isinstance(cursor, Implication):
        ambient_variables.update(cursor.bound_variables)
        cursor = cursor.conclusion
    step_lines, dispositions = _render_chapter_steps(
        chapter,
        str(theorem["id"]),
        assumption_map,
        replacements,
        anchor_head,
        ambient_variables,
        binary_definitions,
        external_facts,
        external_theorem_arguments,
    )
    lines.extend(step_lines)
    goal_fact = _row_reference(str(steps[0]["id"]))
    assert goal_fact
    lines.extend([f"  solve_by_elim [{goal_fact}]", ""])
    return lines, {
        "id": theorem["id"],
        "source_index": theorem["source_index"],
        "method": theorem["method"],
        "chapters": [_chapter_manifest(chapter, dispositions)],
    }


def _helper_free_tokens(
    chapter: dict[str, object],
    replacements: dict[str, str],
) -> list[str]:
    """
    @brief Determine explicit free variables required by an induction helper.
    @details
    Context tokens, copy aliases, and witnesses introduced from encoded
    existentials are excluded. Remaining tokens are ordered by Lean name; an
    original token sorts before a copy alias when both render to the same name.
    @param chapter Neutral induction chapter.
    @param replacements Chapter-local identifier aliases.
    @return Free MPL tokens requiring helper parameters.
    """

    steps = chapter["steps"]
    variable_types = chapter["variable_types"]
    assert isinstance(steps, list) and isinstance(variable_types, dict)
    tokens: set[str] = set()
    for step in steps:
        tokens.update(_free_tokens(parse_mpl(str(step["conclusion"]["mpl"]))))
        dependencies = step["dependencies"]
        assert isinstance(dependencies, list)
        for dependency in dependencies:
            tokens.update(_free_tokens(parse_mpl(str(dependency["expression"]["mpl"]))))
    witness_tokens: set[str] = set()
    for source_step in _existence_projection_sources(chapter):
        source_record = next(step for step in steps if str(step["id"]) == source_step)
        source = parse_mpl(str(source_record["conclusion"]["mpl"]))
        assert isinstance(source, Negation)
        inner = source.inner
        assert isinstance(inner, Implication) and inner.bound_variables
        witness_tokens.add(inner.bound_variables[0])
    scope_bound_tokens: set[str] = set()
    for context in chapter.get("scope_contexts", []):
        assert isinstance(context, dict)
        cursor = parse_mpl(str(context["template"]["mpl"]))
        while isinstance(cursor, Implication):
            scope_bound_tokens.update(cursor.bound_variables)
            cursor = cursor.conclusion
    tokens.difference_update(CONTEXT_TOKENS)
    tokens.difference_update(witness_tokens)
    tokens.difference_update(scope_bound_tokens)
    result: list[str] = []
    rendered: set[str] = {
        CONTEXT_NAMES[token]
        for token in CONTEXT_TOKENS
    }
    for token in sorted(
        tokens,
        key=lambda item: (
            replacements.get(item, lean_name(item)),
            item.endswith("_copy"),
            item,
        ),
    ):
        assert token in variable_types, f"helper token {token!r} has no type"
        identifier = replacements.get(token, lean_name(token))
        if identifier in rendered:
            continue
        rendered.add(identifier)
        result.append(token)
    return result


def _helper_assumptions(
    chapter: dict[str, object],
) -> list[tuple[dict[str, object], Expression]]:
    """
    @brief Collect the explicit non-anchor assumptions of an induction helper.
    @details
    The returned order is certificate step order and therefore also the exact
    argument order of the generated helper theorem.
    @param chapter Neutral induction chapter.
    @return Source step and parsed expression pairs for helper assumptions.
    """

    steps = chapter["steps"]
    assert isinstance(steps, list)
    assumptions: list[tuple[dict[str, object], Expression]] = []
    for step in steps:
        if step["action"] not in {"assume", "induction_assumption"}:
            continue
        if step["namespace"] != "main":
            continue
        expression = parse_mpl(str(step["conclusion"]["mpl"]))
        if not _is_anchor(expression):
            assumptions.append((step, expression))
    return assumptions


def _render_certificate_helper(
    chapter: dict[str, object],
    helper_name: str,
    anchor_head: str = "AnchorPeano",
    binary_definitions: dict[str, object] | None = None,
) -> tuple[list[str], dict[str, object], list[str]]:
    """
    @brief Render one induction subgraph as a private checked Lean theorem.
    @details
    Every non-anchor task or induction assumption is an explicit ordinary
    proposition parameter; every source row remains a named fact in the body.
    @param chapter Neutral induction chapter.
    @param helper_name Stable generated helper name.
    @param anchor_head AnchorPeano or AnchorGauss context selector.
    @param binary_definitions Compiled definitions used by schema-3 Gauss rows.
    @return Lean source lines, chapter-level manifest record, and ordered
    external-fact parameter names required by the helper.
    """

    replacements = _chapter_replacements(chapter)
    variable_types = chapter["variable_types"]
    steps = chapter["steps"]
    assert isinstance(variable_types, dict) and isinstance(steps, list) and steps
    assumption_map = {ANCHOR_MPL[anchor_head]: "anchor"}
    assumptions: list[tuple[str, Expression]] = []
    for step, expression in _helper_assumptions(chapter):
        mpl = to_mpl(expression)
        assert mpl not in assumption_map, f"duplicate chapter assumption {mpl!r}"
        label = f"assumption_{step['source_line']}"
        assumption_map[mpl] = label
        assumptions.append((label, expression))

    external_interfaces = (
        chapter.get("_external_interfaces", [])
        if anchor_head == "AnchorFTA"
        else []
    )
    external_facts = (
        chapter.get("_external_reference_facts", {})
        if anchor_head == "AnchorFTA"
        else {}
    )
    external_theorem_arguments = (
        chapter.get("_external_theorem_arguments", {})
        if anchor_head == "AnchorFTA"
        else {}
    )
    assert isinstance(external_interfaces, list)
    assert isinstance(external_facts, dict)
    assert isinstance(external_theorem_arguments, dict)
    lines = [f"private theorem {helper_name}", *_context_signature(anchor_head)]
    lines.extend(
        f"    ({fact} : {proposition})"
        for fact, proposition in external_interfaces
    )
    helper_free_tokens = _helper_free_tokens(chapter, replacements)
    for token in helper_free_tokens:
        lines.append(
            f"    ({replacements.get(token, lean_name(token))} : "
            f"{_lean_type(str(variable_types[token]))})"
        )
    for label, expression in assumptions:
        lines.append(
            f"    ({label} : "
            f"{render_expression(expression, variable_types, replacements)})"
        )
    conclusion = parse_mpl(str(steps[-1]["conclusion"]["mpl"]))
    lines.append(
        f"    : {render_expression(conclusion, variable_types, replacements)} := by"
    )
    step_lines, dispositions = _render_chapter_steps(
        chapter,
        helper_name,
        assumption_map,
        replacements,
        anchor_head,
        set(helper_free_tokens),
        binary_definitions,
        external_facts,
        external_theorem_arguments,
    )
    lines.extend(step_lines)
    lines.extend([f"  exact {_row_reference(str(steps[-1]['id']))}", ""])
    return (
        lines,
        _chapter_manifest(chapter, dispositions),
        [fact for fact, _ in external_interfaces],
    )


def _induction_property(theorem: dict[str, object]) -> Expression:
    """
    @brief Derive the ordinary relational-induction predicate from a theorem.
    @details
    A leading membership premise for the induction variable is removed because
    it is supplied separately to relationalInduction; other structure remains.
    @param theorem Neutral induction-theorem record.
    @return Property expression with the induction variable free.
    """

    induction_variable = str(theorem["induction_variable"])
    body = _theorem_body(parse_mpl(str(theorem["theorem"]["mpl"])))
    if isinstance(body, Implication):
        premise = body.premise
        if (
            isinstance(premise, Atom)
            and premise.head == "in"
            and premise.arguments == (induction_variable, "N")
        ):
            body = body.conclusion

    def remove_binder(expression: Expression) -> Expression:
        """
        @brief Remove the induction variable from nested binder lists.
        @details All logical connectives and other binder order are preserved.
        @param expression Current property expression node.
        @return Rebuilt expression without the induction-variable binder.
        """

        if isinstance(expression, Atom):
            return expression
        if isinstance(expression, Conjunction):
            return Conjunction(tuple(remove_binder(part) for part in expression.parts))
        if isinstance(expression, Negation):
            return Negation(remove_binder(expression.inner))
        assert isinstance(expression, Implication)
        return Implication(
            tuple(
                variable
                for variable in expression.bound_variables
                if variable != induction_variable
            ),
            remove_binder(expression.premise),
            remove_binder(expression.conclusion),
        )

    return remove_binder(body)


def _render_induction_theorem(
    theorem: dict[str, object],
    anchor_head: str = "AnchorPeano",
    binary_definitions: dict[str, object] | None = None,
) -> tuple[list[str], dict[str, object]]:
    """
    @brief Render three induction chapters and their public composed theorem.
    @details
    Private helpers replay every certificate row. The public proof composes
    them only through the explicit ordinary relationalInduction premise.
    @param theorem Neutral induction-theorem record.
    @param anchor_head AnchorPeano or AnchorGauss context selector.
    @param binary_definitions Compiled definitions used by schema-3 Gauss rows.
    @return Lean source lines and theorem-level manifest record.
    """

    assert theorem["method"] == "induction"
    chapters_list = theorem["chapters"]
    assert isinstance(chapters_list, list)
    chapters = {str(chapter["role"]): chapter for chapter in chapters_list}
    roles = ("induction_typing", "check_zero", "check_induction_condition")
    assert set(chapters) == set(roles)
    lines: list[str] = []
    chapter_manifests: list[dict[str, object]] = []
    helper_names: dict[str, str] = {}
    helper_external_arguments: dict[str, list[str]] = {}
    for role in roles:
        helper_name = f"{theorem['id']}_{role}"
        helper_names[role] = helper_name
        helper_lines, helper_manifest, external_arguments = _render_certificate_helper(
            chapters[role],
            helper_name,
            anchor_head,
            binary_definitions,
        )
        lines.extend(helper_lines)
        chapter_manifests.append(helper_manifest)
        helper_external_arguments[role] = external_arguments

    replacements = dict(CONTEXT_NAMES)
    external_interfaces = (
        theorem.get("_external_interfaces", [])
        if anchor_head == "AnchorFTA"
        else []
    )
    assert isinstance(external_interfaces, list)
    header, introductions, theorem_assumption_map, conclusion, labels = _theorem_header(
        theorem,
        replacements,
        anchor_head,
        external_interfaces,
    )
    lines.extend(header)
    lines.extend(introductions)
    property_expression = _induction_property(theorem)
    variable_types = theorem["variable_types"]
    assert isinstance(variable_types, dict)
    induction_variable = str(theorem["induction_variable"])
    target = replacements.get(induction_variable, lean_name(induction_variable))
    step_chapter = chapters["check_induction_condition"]
    step_replacements = _chapter_replacements(step_chapter)
    step_variable_types = step_chapter["variable_types"]
    assert isinstance(step_variable_types, dict)
    step_free_tokens = _helper_free_tokens(step_chapter, step_replacements)
    property_premises, _ = _flatten_implications(property_expression)
    step_premise_terms = {
        to_mpl(premise): f"step_premise_{index}"
        for index, premise in enumerate(property_premises, start=1)
    }

    def step_free_term(token: str) -> str:
        """
        @brief Map one helper free token to the current induction-step variable.
        @details The previous value and target value receive the two induction
        names; property binders retain their generated identifiers.
        @param token Helper MPL free token.
        @return Lean term supplied to the helper parameter.
        """

        if token == "rec":
            return "induction_n"
        if token == induction_variable:
            return "induction_m"
        return replacements.get(token, lean_name(token))

    def step_assumption_term(
        step: dict[str, object],
        expression: Expression,
    ) -> str:
        """
        @brief Resolve one helper assumption to an exact local proof term.
        @details
        Step-property premises win over identically spelled outer premises.
        Induction assumptions distinguish the predecessor property from the
        successor edge structurally.
        @param step Certificate assumption step.
        @param expression Parsed assumption expression.
        @return Lean local fact name supplied to the helper.
        """

        mpl = to_mpl(expression)
        if step["action"] == "induction_assumption":
            step_id = str(step["id"])
            assert step_id in specialized_induction_assumptions
            return specialized_induction_assumptions[step_id]
        if mpl in step_premise_terms:
            return step_premise_terms[mpl]
        if (
            isinstance(expression, Atom)
            and expression.head == "in"
            and expression.arguments == (induction_variable, "N")
        ):
            # The current induction value's membership: derived in the step
            # from the successor edge, never the theorem's own typing premise
            # (that one speaks about the theorem's variable, not induction_m).
            return "induction_m_member"
        if mpl in theorem_assumption_map:
            return theorem_assumption_map[mpl]
        if (
            isinstance(expression, Atom)
            and expression.head == "in"
            and expression.arguments == ("rec", "N")
        ):
            return "induction_n_member"
        raise AssertionError(
            f"cannot resolve induction-step helper assumption {mpl!r}"
        )

    specialization_replacements = dict(step_replacements)
    for token in step_free_tokens:
        specialization_replacements[token] = step_free_term(token)

    def property_at(name: str) -> str:
        """
        @brief Render the induction property at one Lean value name.
        @details Only the induction-variable replacement changes.
        @param name Lean identifier substituted for the induction variable.
        @return Lean proposition for the specialized property.
        """

        local = dict(replacements)
        local[induction_variable] = name
        return render_expression(property_expression, variable_types, local)

    specialized_induction_assumptions: dict[str, str] = {}
    specialized_induction_lines: list[str] = []
    for step, expression in _helper_assumptions(step_chapter):
        if step["action"] != "induction_assumption":
            continue
        label = f"step_induction_assumption_{len(specialized_induction_assumptions) + 1}"
        specialized_induction_assumptions[str(step["id"])] = label
        proposition = render_expression(
            expression,
            step_variable_types,
            specialization_replacements,
        )
        successor_proposition = (
            f"(succ induction_n induction_m)"
        )
        if proposition == successor_proposition:
            proof = ["      exact induction_successor"]
        else:
            proof = [
                *(
                    "  " + introduction
                    for introduction in _proof_introductions(
                        expression,
                        specialization_replacements,
                        f"{label}_premise",
                    )
                ),
                "      apply induction_hypothesis",
                "      all_goals assumption",
            ]
        specialized_induction_lines.extend(
            [
                f"    have {label} :",
                (
                    "        "
                    + proposition
                    + " := by"
                ),
                *proof,
            ]
        )

    step_application = [
        "stepRule",
        *(step_free_term(token) for token in step_free_tokens),
        *(
            step_assumption_term(step, expression)
            for step, expression in _helper_assumptions(step_chapter)
        ),
    ]

    helper_call_arguments = {
        role: " ".join(
            [
                _context_application(anchor_head),
                *helper_external_arguments[role],
            ]
        )
        for role in roles
    }

    typing_chapter = chapters["induction_typing"]
    typing_replacements = _chapter_replacements(typing_chapter)
    typing_variable_types = typing_chapter["variable_types"]
    assert isinstance(typing_variable_types, dict)
    typing_free_tokens = _helper_free_tokens(
        typing_chapter,
        typing_replacements,
    )
    theorem_assumption_by_proposition = {
        render_expression(
            parse_mpl(mpl),
            variable_types,
            replacements,
        ): label
        for mpl, label in theorem_assumption_map.items()
        if mpl != ANCHOR_MPL[anchor_head]
    }
    typing_assumption_terms: list[str] = []
    for step, expression in _helper_assumptions(typing_chapter):
        assert step["action"] == "assume"
        proposition = render_expression(
            expression,
            typing_variable_types,
            typing_replacements,
        )
        assert proposition in theorem_assumption_by_proposition, (
            f"cannot resolve induction-typing assumption {to_mpl(expression)!r} "
            f"for {theorem['id']}"
        )
        typing_assumption_terms.append(
            theorem_assumption_by_proposition[proposition]
        )
    typing_application = " ".join(
        [
            "typingRule",
            *(
                target
                if token == induction_variable
                else typing_replacements.get(token, lean_name(token))
                for token in typing_free_tokens
            ),
            *typing_assumption_terms,
        ]
    )

    base_replacements = dict(replacements)
    base_replacements[induction_variable] = "zero"
    base_introductions = _proof_introductions(
        property_expression,
        base_replacements,
        "base_premise",
    )
    base_premise_by_proposition = {
        render_expression(
            premise,
            variable_types,
            base_replacements,
        ): f"base_premise_{index}"
        for index, premise in enumerate(property_premises, start=1)
    }
    zero_chapter = chapters["check_zero"]
    zero_chapter_replacements = _chapter_replacements(zero_chapter)
    zero_free_tokens = _helper_free_tokens(
        zero_chapter,
        zero_chapter_replacements,
    )
    zero_replacements = dict(zero_chapter_replacements)
    zero_replacements[induction_variable] = "zero"
    zero_variable_types = zero_chapter["variable_types"]
    assert isinstance(zero_variable_types, dict)
    zero_assumption_terms: list[str] = []
    for step, expression in _helper_assumptions(zero_chapter):
        if step["action"] == "induction_assumption":
            assert (
                isinstance(expression, Atom)
                and expression.head == "="
                and expression.arguments == (induction_variable, "i0")
            ), (
                f"unsupported zero induction assumption in {theorem['id']}: "
                f"{to_mpl(expression)}"
            )
            zero_assumption_terms.append("rfl")
            continue
        proposition = render_expression(
            expression,
            zero_variable_types,
            zero_replacements,
        )
        if proposition in base_premise_by_proposition:
            zero_assumption_terms.append(
                base_premise_by_proposition[proposition]
            )
            continue
        if (
            isinstance(expression, Atom)
            and expression.head == "in"
            and expression.arguments == (induction_variable, "N")
        ):
            zero_assumption_terms.append("inductionZeroMember")
            continue
        raise AssertionError(
            f"cannot resolve zero helper assumption {to_mpl(expression)!r} "
            f"for {theorem['id']}"
        )
    zero_application = " ".join(
        [
            "zeroRule",
            *(
                "zero"
                if token == induction_variable
                else zero_replacements.get(token, lean_name(token))
                for token in zero_free_tokens
            ),
            *zero_assumption_terms,
        ]
    )

    property_application_terms: list[str] = []
    property_cursor = property_expression
    while isinstance(property_cursor, Implication):
        property_application_terms.extend(
            replacements.get(bound, lean_name(bound))
            for bound in property_cursor.bound_variables
        )
        premise_mpl = to_mpl(property_cursor.premise)
        assert premise_mpl in theorem_assumption_map
        property_application_terms.append(theorem_assumption_map[premise_mpl])
        property_cursor = property_cursor.conclusion
    zero_projection_depth = 12 + (
        1 if anchor_head == "AnchorPeano" else 3
    )
    zero_projection = "anchor" + ".1" * zero_projection_depth

    # When the step helper assumes the current value's membership
    # (`in[<induction variable>, N]`), the step derives it from the successor
    # edge through the anchor's successor-function closure, with every
    # projection path and compact name read off this corpus' definitions.
    successor_member_lines: list[str] = []
    if any(
        step["action"] != "induction_assumption"
        and isinstance(expression, Atom)
        and expression.head == "in"
        and expression.arguments == (induction_variable, "N")
        for step, expression in _helper_assumptions(step_chapter)
    ):
        assert binary_definitions is not None
        anchor_elements = [
            parse_mpl(str(element))
            for element in binary_definitions[anchor_head]["elements"]
        ]
        natural_numbers_index = next(
            index
            for index, element in enumerate(anchor_elements)
            if isinstance(element, Atom) and element.head == "NaturalNumbers"
        )
        natural_numbers_elements = [
            parse_mpl(str(element))
            for element in binary_definitions["NaturalNumbers"]["elements"]
        ]
        successor_index = next(
            index
            for index, element in enumerate(natural_numbers_elements)
            if isinstance(element, Atom)
            and element.head == "fXY"
            and element.arguments[0] == "u_3"
        )
        fxy_elements = [
            parse_mpl(str(element))
            for element in binary_definitions["fXY"]["elements"]
        ]
        closure_index, closure_head = next(
            (index, element.head)
            for index, element in enumerate(fxy_elements)
            if isinstance(element, Atom)
            and [str(part) for part in binary_definitions[element.head]["elements"]]
            == ["(in2[1,2,u_1])", "(in[2,u_2])"]
        )
        successor_member_lines = [
            (
                "    have natural_numbers_step : gl_NaturalNumbers N zero succ add mul := "
                f"anchor{_left_associated_projection(len(anchor_elements), natural_numbers_index)}"
            ),
            (
                "    have successor_structure : gl_fXY succ N N := natural_numbers_step"
                f"{_left_associated_projection(len(natural_numbers_elements), successor_index)}"
            ),
            (
                f"    have successor_output_closed : {lean_constant(closure_head)} succ N := "
                f"successor_structure{_left_associated_projection(len(fxy_elements), closure_index)}"
            ),
            (
                "    have induction_m_member : N induction_m := "
                "successor_output_closed induction_n induction_m induction_successor"
            ),
        ]

    lines.extend(
        [
            f"  have inductionMember : N {target} := by",
            (
                f"    have typingRule := {helper_names['induction_typing']} "
                f"{helper_call_arguments['induction_typing']}"
            ),
            f"    exact {typing_application}",
            "  have inductionZeroMember : N zero := by",
            f"    simp only [{lean_constant(anchor_head)}, gl_NaturalNumbers] at anchor",
            f"    exact {zero_projection}",
            f"  have inductionBase : {property_at('zero')} := by",
            *base_introductions,
            (
                f"    have zeroRule := {helper_names['check_zero']} "
                f"{helper_call_arguments['check_zero']}"
            ),
            f"    exact {zero_application}",
            "  have inductionStep :",
            (
                f"      ∀ induction_n, N induction_n → {property_at('induction_n')} → "
                f"∀ induction_m, succ induction_n induction_m → "
                f"{property_at('induction_m')} := by"
            ),
            "    intro induction_n induction_n_member induction_hypothesis",
            "    intro induction_m induction_successor",
            *successor_member_lines,
            *_proof_introductions(
                property_expression,
                {
                    **replacements,
                    induction_variable: "induction_m",
                },
                "step_premise",
            ),
            *specialized_induction_lines,
            (
                f"    have stepRule := {helper_names['check_induction_condition']} "
                f"{helper_call_arguments['check_induction_condition']}"
            ),
            f"    exact {' '.join(step_application)}",
            f"  have inductionProperty : {property_at(target)} := by",
            (
                "    exact relationalInduction "
                "N zero succ add mul one two identity anchor"
                if anchor_head == "AnchorFTA"
                else "    exact relationalInduction"
            ),
            f"      (fun induction_value => {property_at('induction_value')})",
            f"      {target}",
            "      inductionBase",
            "      inductionStep",
            "      inductionMember",
            (
                "  exact inductionProperty"
                + (
                    " " + " ".join(property_application_terms)
                    if property_application_terms
                    else ""
                )
            ),
            "",
        ]
    )
    assert render_expression(conclusion, variable_types, replacements)
    return lines, {
        "id": theorem["id"],
        "source_index": theorem["source_index"],
        "method": theorem["method"],
        "chapters": chapter_manifests,
    }


def _render_reformulated_theorem(
    theorem: dict[str, object],
    certificate: dict[str, object],
) -> tuple[list[str], dict[str, object]]:
    """
    @brief Render one Gauss theorem reconstructed from a cited theorem.
    @details
    The single row's checked totality metadata builds a set comprehension or
    binary-relation lambda for the GL-defined output. The generated proof
    establishes that output's definition, applies the cited source theorem,
    and folds both facts into the recorded existence compact. No existence
    axiom, dependent GL type family, or admitted declaration is introduced.
    @param theorem Neutral reformulated-statement certificate record.
    @param certificate Complete schema-3 Gauss certificate and type closure.
    @return Lean source lines and theorem-level manifest record.
    @invariant The theorem contains one checked theorem-reformulation action.
    """

    assert theorem["method"] == "reformulated statement"
    chapters = theorem["chapters"]
    assert isinstance(chapters, list) and len(chapters) == 1
    chapter = chapters[0]
    assert isinstance(chapter, dict)
    steps = chapter["steps"]
    assert isinstance(steps, list) and len(steps) == 1
    step = steps[0]
    assert step["action"] == "theorem_reformulate"
    dependencies = step["dependencies"]
    assert isinstance(dependencies, list) and len(dependencies) == 1
    source_reference = dependencies[0]["theorem_reference"]
    assert isinstance(source_reference, dict)
    replacements = _chapter_replacements(chapter)
    lines, _, _, _, _ = _theorem_header(
        theorem,
        replacements,
        "AnchorGauss",
    )
    lines.append(
        f"  have reformulationSource := {source_reference['theorem_id']} "
        f"{_context_application('AnchorGauss', source_reference.get('certificate_id'))}"
    )

    totality = step["defined_output_totality"]
    assert isinstance(totality, dict)
    assert totality["definition_dependency_acyclic"] is True
    defining_expression = parse_mpl(str(totality["defining_expression"]["mpl"]))
    assert isinstance(defining_expression, Atom)
    assert defining_expression.head == totality["definition_head"]
    output_argument = int(totality["output_argument"])
    assert 1 <= output_argument <= len(defining_expression.arguments)
    output_placeholder = defining_expression.arguments[output_argument - 1]
    bound_variables = totality["bound_variables"]
    assert isinstance(bound_variables, list)
    witness_elements = [
        parse_mpl(str(element["mpl"]))
        for element in totality["witness_elements"]
    ]
    assert witness_elements
    variable_types = dict(theorem["variable_types"])
    for bound_variable in bound_variables:
        variable_types[str(bound_variable)] = "element"
    witness_body = " ∧ ".join(
        render_expression(element, variable_types, replacements)
        for element in witness_elements
    )
    if totality["output_type"] == "set":
        assert len(bound_variables) == 1
        bound = lean_name(str(bound_variables[0]))
        witness_term = f"(fun ({bound} : α) => {witness_body})"
    else:
        assert totality["output_type"] == "binary_relation"
        assert len(bound_variables) == 2
        first = lean_name(str(bound_variables[0]))
        second = lean_name(str(bound_variables[1]))
        witness_term = (
            f"(fun ({first} : α) ({second} : α) => {witness_body})"
        )
    output_replacements = dict(replacements)
    output_replacements[output_placeholder] = witness_term
    totality_proposition = render_expression(
        defining_expression,
        variable_types,
        output_replacements,
    )
    general_parameters = [
        argument
        for argument in defining_expression.arguments
        if argument not in CONTEXT_TOKENS and argument != output_placeholder
    ]
    assert len(general_parameters) == len(set(general_parameters))
    if general_parameters:
        binders = " ".join(
            f"({replacements.get(argument, lean_name(argument))} : "
            f"{_lean_type(variable_types[argument])})"
            for argument in general_parameters
        )
        totality_proposition = f"(∀ {binders}, {totality_proposition})"
    unfold_definitions = totality["unfold_definitions"]
    assert isinstance(unfold_definitions, list) and unfold_definitions
    assert unfold_definitions[0] == defining_expression.head
    unfold_constants = ", ".join(
        lean_constant(str(name)) for name in unfold_definitions
    )
    totality_proof = [
        f"    simp only [{unfold_constants}]",
    ]
    if general_parameters:
        totality_proof.append(
            "    intro "
            + " ".join(
                replacements.get(argument, lean_name(argument))
                for argument in general_parameters
            )
        )
    if totality["output_type"] == "set":
        totality_proof.extend(
            [
                "    constructor",
                "    · constructor",
                "      · intro value witness",
                "        exact witness.1",
                "      · intro value witness",
                "        exact witness.2",
                "    · intro value member ordering",
                "      exact ⟨member, ordering⟩",
            ]
        )
    else:
        assert totality["output_type"] == "binary_relation"
        totality_proof.extend(
            [
                "    constructor",
                "    · constructor",
                "      · intro first second witness",
                "        exact witness.1",
                "      · intro first second witness",
                "        exact witness.2",
                "    · intro first ordering second member",
                "      exact ⟨ordering, member⟩",
            ]
        )
    lines.extend(
        [
            f"  have defined_output_totality : {totality_proposition} := by",
            *totality_proof,
        ]
    )

    conclusion_expression = parse_mpl(str(step["conclusion"]["mpl"]))
    body = _theorem_body(conclusion_expression)
    theorem_body = _theorem_body(parse_mpl(str(theorem["theorem"]["mpl"])))
    assert body == theorem_body
    existence_expression = body
    while isinstance(existence_expression, Implication):
        existence_expression = existence_expression.conclusion
    assert isinstance(existence_expression, Atom)
    fact = _row_reference(str(step["id"]))
    existence_head = str(step["reformulation_definition"])
    assert existence_expression.head == existence_head
    binary_definitions = certificate["binary_definitions"]
    assert isinstance(binary_definitions, dict)
    existence_definition = binary_definitions[existence_head]
    assert isinstance(existence_definition, dict)
    existence_elements = existence_definition["elements"]
    assert isinstance(existence_elements, list) and len(existence_elements) == 2
    existence_parameters = {
        f"u_{index + 1}": argument
        for index, argument in enumerate(existence_expression.arguments)
    }
    output_property_expression = map_arguments(
        parse_mpl(str(existence_elements[1])),
        lambda name: existence_parameters.get(name, name),
    )
    output_property = render_expression(
        output_property_expression,
        variable_types,
        output_replacements,
    )
    defined_output = render_expression(
        defining_expression,
        variable_types,
        output_replacements,
    )
    totality_arguments = " ".join(
        replacements.get(argument, lean_name(argument))
        for argument in general_parameters
    )
    proof_introductions = _proof_introductions(
        body,
        replacements,
        "reformulation_premise",
    )
    source_expression = parse_mpl(str(source_reference["theorem"]["mpl"]))
    source_body = _theorem_body(source_expression)
    assert isinstance(body, Implication)
    assert isinstance(body.conclusion, Implication)
    assert isinstance(source_body, Implication)
    assert isinstance(source_body.conclusion, Implication)
    assert isinstance(source_body.conclusion.conclusion, Implication)
    assert len(source_body.bound_variables) == len(body.bound_variables)
    assert len(source_body.conclusion.bound_variables) == (
        len(body.conclusion.bound_variables) + 1
    )
    assert not source_body.conclusion.conclusion.bound_variables
    source_application = " ".join(
        [
            "reformulationSource",
            *(
                replacements.get(variable, lean_name(variable))
                for variable in body.bound_variables
            ),
            "reformulation_premise_1",
            *(
                replacements.get(variable, lean_name(variable))
                for variable in body.conclusion.bound_variables
            ),
            witness_term,
            "defined_output",
            "reformulation_premise_2",
        ]
    )
    lines.extend(
        [
            f"  -- {step['id']}: GL tag {step['source_tag']}.",
            f"  have {fact} : {render_expression(body, variable_types, replacements)} := by",
            "    classical",
            *proof_introductions,
            f"    simp only [{lean_constant(existence_head)}]",
            "    intro no_defined_output",
            (
                f"    have defined_output : {defined_output} := "
                f"defined_output_totality {totality_arguments}"
            ).rstrip(),
            f"    have output_result : {output_property} := by",
            f"      exact {source_application}",
            (
                f"    exact no_defined_output {witness_term} "
                "defined_output output_result"
            ),
            f"  exact {fact}",
            "",
        ]
    )
    disposition = _manifest_disposition(
        step,
        str(theorem["id"]),
        fact,
        (
            "checked theorem reformulation with constructive defined-output "
            f"totality via {lean_constant(existence_head)}"
        ),
    )
    return lines, {
        "id": theorem["id"],
        "source_index": theorem["source_index"],
        "method": theorem["method"],
        "chapters": [_chapter_manifest(chapter, [disposition])],
    }


def render_gauss_theory(
    certificate: dict[str, object],
) -> tuple[str, list[dict[str, object]]]:
    """
    @brief Render a schema-3 Gauss or FTA corpus over ordinary predicates.
    @details
    Gauss imports Peano and derives its Peano anchor. FTA imports the existing
    Gauss module plus its isolated definition delta and derives both external
    anchors from AnchorFTA. Each corpus remains in certificate dependency order;
    every proof scope is rendered as an ordinary quantified implication.
    @param certificate Complete schema-3 Gauss or FTA certificate.
    @return Lean source and theorem-level manifest records.
    @invariant Every certificate row receives one named fact.
    """

    assert certificate["schema_version"] == 3
    corpus = certificate["corpus"]
    assert isinstance(corpus, dict)
    corpus_id = str(corpus["id"])
    theorems = certificate["theorems"]
    assert isinstance(theorems, list)
    if corpus_kind(corpus_id) == "gauss":
        if corpus_id in TRACKED_CORPUS_COUNTS:
            assert len(theorems) == TRACKED_CORPUS_COUNTS[corpus_id][0]
        anchor_head = "AnchorGauss"
        namespace = "GLExport"
        lines = [
            LEAN_LICENSE,
            "",
            "import GLExport.Generated.Peano",
            "",
            "set_option linter.unusedVariables false",
            "",
            "namespace GLExport",
            "",
            "universe u",
            "",
            "private theorem anchorPeanoOfGauss",
            "    {α : Type u}",
            "    (N : GLSet α)",
            "    (zero : α)",
            "    (succ : GLBinaryRelation α)",
            "    (add mul : GLTernaryRelation α)",
            "    (one two : α)",
            "    (identity : GLBinaryRelation α)",
            "    (anchor : gl_AnchorGauss N zero succ add mul one two identity)",
            "    : gl_AnchorPeano N zero succ add mul one := by",
            "  simp only [gl_AnchorGauss, gl_AnchorPeano] at anchor ⊢",
            "  exact anchor.1.1",
            "",
        ]
    else:
        assert corpus_kind(corpus_id) == "fta"
        if corpus_id in TRACKED_CORPUS_COUNTS:
            assert len(theorems) == TRACKED_CORPUS_COUNTS[corpus_id][0]
            assert corpus["coverage"]["rows"] == TRACKED_CORPUS_COUNTS[corpus_id][1]
        anchor_head = "AnchorFTA"
        namespace = "GLExport.FTA"
        # An isolated FTA corpus (the tracked release package) rides the
        # reviewed Gauss module plus its definition delta; a self-contained
        # corpus (the live shortcut export, independent of the main path)
        # carries every definition it uses in its own Definitions module.
        imports = (
            ["import GLExport.Generated.Gauss", "import GLExport.Generated.FTADefinitions"]
            if "definition_isolation" in certificate
            else ["import GLExport.Generated.Definitions"]
        )
        lines = [
            LEAN_LICENSE,
            "",
            *imports,
            "",
            "set_option linter.unusedVariables false",
            "",
            "namespace GLExport.FTA",
            "",
            "universe u",
            "",
            "private theorem anchorPeanoOfFTA",
            "    {α : Type u}",
            "    (N : GLSet α)",
            "    (zero : α)",
            "    (succ : GLBinaryRelation α)",
            "    (add mul : GLTernaryRelation α)",
            "    (one two : α)",
            "    (identity : GLBinaryRelation α)",
            "    (anchor : gl_AnchorFTA N zero succ add mul one two identity)",
            "    : GLExport.gl_AnchorPeano N zero succ add mul one := by",
            "  simp only [gl_AnchorFTA, GLExport.gl_AnchorPeano] at anchor ⊢",
            "  exact anchor.1.1",
            "",
            "private theorem anchorGaussOfFTA",
            "    {α : Type u}",
            "    (N : GLSet α)",
            "    (zero : α)",
            "    (succ : GLBinaryRelation α)",
            "    (add mul : GLTernaryRelation α)",
            "    (one two : α)",
            "    (identity : GLBinaryRelation α)",
            "    (anchor : gl_AnchorFTA N zero succ add mul one two identity)",
            "    : GLExport.gl_AnchorGauss N zero succ add mul one two identity := by",
            "  simp only [gl_AnchorFTA, GLExport.gl_AnchorGauss] at anchor ⊢",
            "  exact anchor",
            "",
        ]
    if anchor_head == "AnchorFTA":
        interfaces_by_theorem: dict[str, list[tuple[str, str]]] = {}
        for theorem in theorems:
            assert isinstance(theorem, dict)
            theorem_interfaces: list[tuple[str, str]] = []
            theorem_propositions: dict[str, str] = {}
            chapters = theorem["chapters"]
            assert isinstance(chapters, list)
            for chapter in chapters:
                assert isinstance(chapter, dict)
                direct_interfaces, direct_facts = _external_reference_interfaces(
                    [chapter]
                )
                chapter_interfaces = list(direct_interfaces)
                chapter_propositions = dict(direct_interfaces)
                theorem_arguments: dict[str, list[str]] = {}
                steps = chapter["steps"]
                assert isinstance(steps, list)
                for step in steps:
                    references: list[object] = []
                    direct_reference = step.get("theorem_reference")
                    if direct_reference is not None:
                        references.append(direct_reference)
                    dependencies = step["dependencies"]
                    assert isinstance(dependencies, list)
                    references.extend(
                        dependency["theorem_reference"]
                        for dependency in dependencies
                        if "theorem_reference" in dependency
                    )
                    for reference in references:
                        assert isinstance(reference, dict)
                        if reference.get("certificate_id") is not None:
                            continue
                        referenced_id = str(reference["theorem_id"])
                        assert referenced_id in interfaces_by_theorem, (
                            "FTA theorem references must remain topologically "
                            f"ordered: {theorem['id']} cites {referenced_id}"
                        )
                        required = interfaces_by_theorem[referenced_id]
                        required_names = [fact for fact, _ in required]
                        if referenced_id in theorem_arguments:
                            assert theorem_arguments[referenced_id] == required_names
                        else:
                            theorem_arguments[referenced_id] = required_names
                        for fact, proposition in required:
                            if fact in chapter_propositions:
                                assert chapter_propositions[fact] == proposition
                                continue
                            chapter_propositions[fact] = proposition
                            chapter_interfaces.append((fact, proposition))
                chapter["_external_interfaces"] = chapter_interfaces
                chapter["_external_reference_facts"] = direct_facts
                chapter["_external_theorem_arguments"] = theorem_arguments
                for fact, proposition in chapter_interfaces:
                    if fact in theorem_propositions:
                        assert theorem_propositions[fact] == proposition
                        continue
                    theorem_propositions[fact] = proposition
                    theorem_interfaces.append((fact, proposition))
            theorem["_external_interfaces"] = theorem_interfaces
            interfaces_by_theorem[str(theorem["id"])] = theorem_interfaces

    manifests: list[dict[str, object]] = []
    binary_definitions = certificate["binary_definitions"]
    assert isinstance(binary_definitions, dict)
    for theorem in theorems:
        assert isinstance(theorem, dict)
        method = str(theorem["method"])
        if method == "direct":
            theorem_lines, manifest = _render_direct_theorem(
                theorem,
                anchor_head,
                binary_definitions,
            )
        elif method == "induction":
            theorem_lines, manifest = _render_induction_theorem(
                theorem,
                anchor_head,
                binary_definitions,
            )
        elif method == "or theorem":
            assert corpus_kind(corpus_id) == "fta"
            theorem_lines, manifest = _render_or_theorem(
                theorem,
                anchor_head,
                binary_definitions,
            )
        elif method == "or elimination":
            assert corpus_kind(corpus_id) == "fta"
            theorem_lines, manifest = _render_or_elimination_theorem(
                theorem,
                anchor_head,
                binary_definitions,
            )
        elif method == "proved not broadcast":
            theorem_lines, manifest = _render_direct_theorem(
                theorem,
                anchor_head,
                binary_definitions,
            )
        else:
            assert corpus_kind(corpus_id) == "gauss" and method == "reformulated statement"
            theorem_lines, manifest = _render_reformulated_theorem(
                theorem,
                certificate,
            )
        lines.extend(theorem_lines)
        manifests.append(manifest)
    lines.extend([f"end {namespace}", ""])
    return "\n".join(lines), manifests


def render_peano_theory(
    certificate: dict[str, object],
) -> tuple[str, list[dict[str, object]]]:
    """
    @brief Render every selected Peano theorem in certificate order.
    @details
    The neutral frontend has already asserted the selected theorem graph is
    acyclic and topologically ordered. Direct, induction, and OR methods share
    one abstract ordinary-type context.
    @param certificate Complete schema-2 Peano certificate.
    @return Lean source and theorem-level manifest records.
    """

    theorems = certificate["theorems"]
    binary_definitions = certificate["binary_definitions"]
    assert isinstance(binary_definitions, dict)
    assert isinstance(theorems, list) and theorems
    lines = [
        LEAN_LICENSE,
        "",
        "import GLExport.Generated.Definitions",
        "",
        "set_option linter.unusedVariables false",
        "",
        "namespace GLExport",
        "",
        "universe u",
        "",
    ]
    manifests: list[dict[str, object]] = []
    for theorem in theorems:
        assert isinstance(theorem, dict)
        method = str(theorem["method"])
        if method == "direct":
            theorem_lines, manifest = _render_direct_theorem(theorem, "AnchorPeano", binary_definitions)
        elif method == "induction":
            theorem_lines, manifest = _render_induction_theorem(theorem, "AnchorPeano", binary_definitions)
        elif method == "or elimination":
            theorem_lines, manifest = _render_or_elimination_theorem(theorem)
        elif method == "proved not broadcast":
            theorem_lines, manifest = _render_direct_theorem(theorem, "AnchorPeano", binary_definitions)
        else:
            assert method == "or theorem"
            theorem_lines, manifest = _render_or_theorem(theorem, "AnchorPeano", binary_definitions)
        lines.extend(theorem_lines)
        manifests.append(manifest)
    lines.extend(["end GLExport", ""])
    return "\n".join(lines), manifests


def _assert_manifest_coverage(
    certificate: dict[str, object],
    rendered_theorems: list[dict[str, object]],
) -> list[dict[str, object]]:
    """
    @brief Assert one emitted Lean disposition for every certificate row.
    @details
    The check compares theorem identity, chapter count, source row counts, and
    disposition counts before any final manifest is written.
    @param certificate Complete neutral certificate.
    @param rendered_theorems Renderer-produced theorem manifest records.
    @return Manifest records ordered exactly like the certificate theorems.
    """

    certificate_theorems = certificate["theorems"]
    assert isinstance(certificate_theorems, list)
    rendered = {str(theorem["id"]): theorem for theorem in rendered_theorems}
    assert len(rendered) == len(rendered_theorems) == len(certificate_theorems)
    ordered: list[dict[str, object]] = []
    for theorem in certificate_theorems:
        theorem_id = str(theorem["id"])
        assert theorem_id in rendered
        emitted = rendered[theorem_id]
        source_chapters = theorem["chapters"]
        emitted_chapters = emitted["chapters"]
        assert isinstance(source_chapters, list) and isinstance(emitted_chapters, list)
        assert len(source_chapters) == len(emitted_chapters)
        for source_chapter, emitted_chapter in zip(
            source_chapters,
            emitted_chapters,
            strict=True,
        ):
            source_steps = source_chapter["steps"]
            dispositions = emitted_chapter["dispositions"]
            assert isinstance(source_steps, list) and isinstance(dispositions, list)
            assert emitted_chapter["source_row_count"] == len(source_steps)
            assert len(dispositions) == len(source_steps)
            assert all(disposition["status"] == "emitted" for disposition in dispositions)
        ordered.append(emitted)
    return ordered


def write_lean(
    certificate: dict[str, object],
    project_directory: Path,
) -> None:
    """
    @brief Write one complete strict Lean corpus and its row manifest.
    @details
    Definitions and theorem replay are generated from a schema-2 Peano or
    schema-3 Gauss/FTA certificate. The writer rejects proof admissions and
    asserts each corpus boundary. FTA writes only its isolated definition delta,
    theorem module, and manifest, preserving every reviewed Peano/Gauss artifact.
    @param certificate Complete neutral Peano, Gauss, or FTA certificate.
    @param project_directory Existing Lean project root.
    @return None.
    """

    schema_version = int(certificate["schema_version"])
    assert schema_version in {2, 3}
    corpus = certificate.get("corpus", {"id": "peano_pilot"})
    assert isinstance(corpus, dict)
    if schema_version == 2 and corpus["id"] == "peano_full_65":
        theorems = certificate["theorems"]
        exclusions = certificate["excluded_sources"]
        assert isinstance(theorems, list) and len(theorems) == 65
        assert isinstance(exclusions, list)
        assert [item["source_index"] for item in exclusions] == [24, 25]
    if schema_version == 2 and corpus["id"] in {"peano_full_66", "peano_full_64"}:
        theorems = certificate["theorems"]
        exclusions = certificate["excluded_sources"]
        expected = {"peano_full_66": 66, "peano_full_64": 64}[str(corpus["id"])]
        assert isinstance(theorems, list) and len(theorems) == expected
        assert isinstance(exclusions, list) and not exclusions
    if schema_version == 3:
        assert corpus_kind(str(corpus["id"])) in ("gauss", "fta")
        if str(corpus["id"]) in TRACKED_CORPUS_COUNTS:
            expected_coverage = TRACKED_CORPUS_COUNTS[str(corpus["id"])]
            assert corpus["coverage"]["theorems"] == expected_coverage[0]
            assert corpus["coverage"]["rows"] == expected_coverage[1]

    generated_directory = project_directory / "GLExport" / "Generated"
    generated_directory.mkdir(parents=True, exist_ok=True)
    if schema_version == 2:
        definitions = render_definitions(certificate)
        rendered_source, rendered_theorems = render_peano_theory(certificate)
        theorem_path = generated_directory / "Peano.lean"
        manifest_path = project_directory / "lean_manifest.json"
        root_imports = "\nimport GLExport.Generated.Peano"
        files = {
            generated_directory / "Definitions.lean": definitions,
            theorem_path: rendered_source,
            project_directory / "GLExport.lean": (
                LEAN_LICENSE
                + "\n\nimport GLExport.ProofSupport"
                + "\nimport GLExport.Generated.Definitions"
                + root_imports
                + "\n"
            ),
        }
    elif corpus_kind(str(corpus["id"])) == "gauss":
        definitions = render_definitions(certificate)
        rendered_source, rendered_theorems = render_gauss_theory(certificate)
        theorem_path = generated_directory / "Gauss.lean"
        manifest_path = project_directory / "lean_gauss_manifest.json"
        root_imports = (
            "\nimport GLExport.Generated.Peano"
            "\nimport GLExport.Generated.Gauss"
        )
        files = {
            generated_directory / "Definitions.lean": definitions,
            theorem_path: rendered_source,
            project_directory / "GLExport.lean": (
                LEAN_LICENSE
                + "\n\nimport GLExport.ProofSupport"
                + "\nimport GLExport.Generated.Definitions"
                + root_imports
                + "\n"
            ),
        }
    elif "definition_isolation" in certificate:
        assert corpus_kind(str(corpus["id"])) == "fta"
        isolation = certificate["definition_isolation"]
        assert isinstance(isolation, dict)
        assert isolation["namespace"] == "GLExport.FTA"
        definitions = render_definitions(
            certificate,
            list(isolation["local_heads"]),
            "GLExport.FTA",
            "GLExport.Generated.Definitions",
        )
        rendered_source, rendered_theorems = render_gauss_theory(certificate)
        theorem_path = generated_directory / "FTA.lean"
        manifest_path = project_directory / "lean_fta_manifest.json"
        files = {
            generated_directory / "FTADefinitions.lean": definitions,
            theorem_path: rendered_source,
        }
    else:
        # Self-contained FTA corpus: its own Definitions module, the FTA
        # module importing only that, and a root that makes FTA the default
        # `lake build` target — nothing from the Peano or Gauss modules.
        assert corpus_kind(str(corpus["id"])) == "fta"
        definitions = render_definitions(certificate)
        rendered_source, rendered_theorems = render_gauss_theory(certificate)
        theorem_path = generated_directory / "FTA.lean"
        manifest_path = project_directory / "lean_fta_manifest.json"
        files = {
            generated_directory / "Definitions.lean": definitions,
            theorem_path: rendered_source,
            project_directory / "GLExport.lean": (
                LEAN_LICENSE
                + "\n\nimport GLExport.ProofSupport"
                + "\nimport GLExport.Generated.Definitions"
                + "\nimport GLExport.Generated.FTA"
                + "\n"
            ),
        }
    forbidden = ("sorry", "admit", "axiom ", "opaque ", "unsafe ")
    lowered = (definitions + "\n" + rendered_source).lower()
    assert not any(token in lowered for token in forbidden), (
        "generated Lean source contains a forbidden trust shortcut"
    )
    ordered_manifest = _assert_manifest_coverage(certificate, rendered_theorems)
    manifest: dict[str, object] = {
        "schema_version": schema_version,
        "backend": "Lean 4 ordinary predicates",
        "theorems": ordered_manifest,
    }
    if schema_version == 2 and corpus_kind(str(corpus["id"])) == "peano":
        manifest["coverage"] = corpus["coverage"]
        manifest["excluded_sources"] = certificate["excluded_sources"]
    elif schema_version == 3:
        manifest["coverage"] = corpus["coverage"]
        manifest["certificate_dependencies"] = certificate[
            "certificate_dependencies"
        ]
        if corpus_kind(str(corpus["id"])) == "fta":
            manifest["target_source_indices"] = corpus["target_source_indices"]
            manifest["support_source_indices"] = corpus["support_source_indices"]
            if "definition_isolation" in certificate:
                manifest["definition_isolation"] = certificate["definition_isolation"]

    files[manifest_path] = json.dumps(
        manifest,
        indent=2,
        sort_keys=True,
    ) + "\n"
    for path, content in files.items():
        path.write_text(content, encoding="utf-8", newline="\n")
