# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt)
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Strict parser and canonical representation for Mathematical Programming Language."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator


@dataclass(frozen=True)
class Atom:
    """An MPL predicate, relation application, equality, or compiled name."""

    head: str
    arguments: tuple[str, ...]


@dataclass(frozen=True)
class Conjunction:
    """An MPL conjunction containing at least two expressions."""

    parts: tuple["Expression", ...]


@dataclass(frozen=True)
class Implication:
    """An MPL implication with its explicitly bound variables."""

    bound_variables: tuple[str, ...]
    premise: "Expression"
    conclusion: "Expression"


@dataclass(frozen=True)
class Negation:
    """An MPL prefix negation."""

    inner: "Expression"


Expression = Atom | Conjunction | Implication | Negation


class _Parser:
    """Stateful recursive-descent parser for one canonical MPL expression."""

    def __init__(self, text: str) -> None:
        """
        @brief Initialize a strict parser for one canonical MPL string.
        @details Empty input and any whitespace are rejected immediately.
        @param text Complete canonical MPL expression.
        @return None.
        """

        assert text, "MPL expression must not be empty"
        assert not any(character.isspace() for character in text), (
            "canonical MPL expressions contain no whitespace"
        )
        self.text = text

    def parse(self) -> Expression:
        """
        @brief Parse the complete input expression.
        @details The recursive result must consume every input byte.
        @return Immutable MPL expression tree.
        """

        expression, position = self._parse_at(0)
        assert position == len(self.text), (
            f"trailing MPL input at byte {position}: {self.text[position:position + 32]!r}"
        )
        return expression

    def _parse_at(self, position: int) -> tuple[Expression, int]:
        """
        @brief Parse one expression beginning at a byte position.
        @details Prefix negation and each parenthesized construct dispatch to
        their strict shape parser.
        @param position Zero-based input byte position.
        @return Parsed expression and first unconsumed byte position.
        """

        assert position < len(self.text), "unexpected end of MPL expression"
        if self.text[position] == "!":
            inner, end = self._parse_at(position + 1)
            return Negation(inner), end
        assert self.text[position] == "(", (
            f"expected '(' at byte {position} in {self.text!r}"
        )
        if self.text.startswith("(&", position):
            return self._parse_conjunction(position)
        if self.text.startswith("(>[", position):
            return self._parse_implication(position)
        return self._parse_atom(position)

    def _parse_conjunction(self, position: int) -> tuple[Conjunction, int]:
        """
        @brief Parse one canonical conjunction.
        @details The conjunction must contain at least two complete expressions.
        @param position Byte position of the opening parenthesis.
        @return Parsed conjunction and first unconsumed byte position.
        """

        cursor = position + 2
        parts: list[Expression] = []
        while cursor < len(self.text) and self.text[cursor] != ")":
            part, cursor = self._parse_at(cursor)
            parts.append(part)
        assert cursor < len(self.text), "unterminated MPL conjunction"
        assert len(parts) >= 2, "canonical MPL conjunction needs at least two parts"
        return Conjunction(tuple(parts)), cursor + 1

    def _parse_implication(self, position: int) -> tuple[Implication, int]:
        """
        @brief Parse one explicitly bound MPL implication.
        @details Exactly one premise and one conclusion must follow the binder.
        @param position Byte position of the opening parenthesis.
        @return Parsed implication and first unconsumed byte position.
        """

        close = self.text.find("]", position + 3)
        assert close >= 0, "unterminated MPL implication binder"
        binder = self.text[position + 3:close]
        bound_variables = tuple(binder.split(",")) if binder else ()
        assert all(bound_variables), "empty name in MPL implication binder"
        premise, cursor = self._parse_at(close + 1)
        conclusion, cursor = self._parse_at(cursor)
        assert cursor < len(self.text) and self.text[cursor] == ")", (
            "MPL implication must contain exactly one premise and one conclusion"
        )
        return Implication(bound_variables, premise, conclusion), cursor + 1

    def _parse_atom(self, position: int) -> tuple[Atom, int]:
        """
        @brief Parse one MPL atom and its flat argument list.
        @details Atom heads and every comma-separated argument must be non-empty.
        @param position Byte position of the opening parenthesis.
        @return Parsed atom and first unconsumed byte position.
        """

        open_arguments = self.text.find("[", position + 1)
        assert open_arguments >= 0, "MPL atom has no argument list"
        head = self.text[position + 1:open_arguments]
        assert head, "MPL atom has an empty head"
        close_arguments = self.text.find("]", open_arguments + 1)
        assert close_arguments >= 0, "unterminated MPL atom argument list"
        assert close_arguments + 1 < len(self.text), "unterminated MPL atom"
        assert self.text[close_arguments + 1] == ")", "nested MPL atom argument"
        raw_arguments = self.text[open_arguments + 1:close_arguments]
        arguments = tuple(raw_arguments.split(",")) if raw_arguments else ()
        assert all(arguments), "empty name in MPL atom argument list"
        return Atom(head, arguments), close_arguments + 2


def parse_mpl(text: str) -> Expression:
    """
    @brief Parse one canonical MPL expression.
    @details A new strict parser owns the complete input and rejects trailing data.
    @param text Canonical compact MPL spelling.
    @return Immutable MPL expression tree.
    """

    return _Parser(text).parse()


def to_mpl(expression: Expression) -> str:
    """
    @brief Serialize an MPL expression to canonical compact spelling.
    @details Every supported node has one deterministic representation.
    @param expression Immutable MPL expression tree.
    @return Canonical MPL string.
    """

    if isinstance(expression, Atom):
        return f"({expression.head}[{','.join(expression.arguments)}])"
    if isinstance(expression, Conjunction):
        return "(&" + "".join(to_mpl(part) for part in expression.parts) + ")"
    if isinstance(expression, Implication):
        binder = ",".join(expression.bound_variables)
        return (
            f"(>[{binder}]{to_mpl(expression.premise)}"
            f"{to_mpl(expression.conclusion)})"
        )
    assert isinstance(expression, Negation)
    return "!" + to_mpl(expression.inner)


def iter_atoms(expression: Expression) -> Iterator[Atom]:
    """
    @brief Iterate all atoms in deterministic syntax order.
    @details Connectives are traversed recursively from left to right.
    @param expression MPL expression tree.
    @return Iterator of atom nodes.
    """

    if isinstance(expression, Atom):
        yield expression
    elif isinstance(expression, Conjunction):
        for part in expression.parts:
            yield from iter_atoms(part)
    elif isinstance(expression, Implication):
        yield from iter_atoms(expression.premise)
        yield from iter_atoms(expression.conclusion)
    else:
        assert isinstance(expression, Negation)
        yield from iter_atoms(expression.inner)


def map_arguments(
    expression: Expression,
    transform: Callable[[str], str],
) -> Expression:
    """
    @brief Transform every binder and atom argument in one expression.
    @details Logical heads and connective structure remain unchanged.
    @param expression Source MPL expression.
    @param transform Deterministic token transformation.
    @return Rebuilt transformed expression.
    """

    if isinstance(expression, Atom):
        return Atom(expression.head, tuple(transform(arg) for arg in expression.arguments))
    if isinstance(expression, Conjunction):
        return Conjunction(tuple(map_arguments(part, transform) for part in expression.parts))
    if isinstance(expression, Implication):
        return Implication(
            tuple(transform(name) for name in expression.bound_variables),
            map_arguments(expression.premise, transform),
            map_arguments(expression.conclusion, transform),
        )
    assert isinstance(expression, Negation)
    return Negation(map_arguments(expression.inner, transform))


def alpha_key(expression: Expression) -> str:
    """
    @brief Form a stable key insensitive to bound-variable spelling.
    @details Binders are renamed in lexical encounter order while free tokens
    remain byte-identical.
    @param expression MPL expression tree.
    @return Canonical alpha-normalized MPL spelling.
    """

    next_index = 0

    def visit(node: Expression, environment: dict[str, str]) -> Expression:
        """
        @brief Alpha-normalize one node under a lexical renaming environment.
        @details Nested binders extend a copied environment with monotonic names.
        @param node Current MPL expression node.
        @param environment Enclosing source-to-canonical binder names.
        @return Rebuilt alpha-normalized node.
        """

        nonlocal next_index
        if isinstance(node, Atom):
            return Atom(
                node.head,
                tuple(environment.get(argument, argument) for argument in node.arguments),
            )
        if isinstance(node, Conjunction):
            return Conjunction(tuple(visit(part, environment) for part in node.parts))
        if isinstance(node, Negation):
            return Negation(visit(node.inner, environment))
        assert isinstance(node, Implication)
        nested = dict(environment)
        canonical: list[str] = []
        for bound in node.bound_variables:
            next_index += 1
            renamed = f"alpha_{next_index}"
            nested[bound] = renamed
            canonical.append(renamed)
        return Implication(
            tuple(canonical),
            visit(node.premise, nested),
            visit(node.conclusion, nested),
        )

    return to_mpl(visit(expression, {}))


def expression_json(expression: Expression) -> dict[str, object]:
    """
    @brief Encode one expression as backend-neutral structured JSON.
    @details Each node records its kind, children, and exact canonical MPL.
    @param expression MPL expression tree.
    @return JSON-ready expression record.
    """

    if isinstance(expression, Atom):
        return {
            "kind": "atom",
            "head": expression.head,
            "arguments": list(expression.arguments),
            "mpl": to_mpl(expression),
        }
    if isinstance(expression, Conjunction):
        return {
            "kind": "conjunction",
            "parts": [expression_json(part) for part in expression.parts],
            "mpl": to_mpl(expression),
        }
    if isinstance(expression, Implication):
        return {
            "kind": "implication",
            "bound_variables": list(expression.bound_variables),
            "premise": expression_json(expression.premise),
            "conclusion": expression_json(expression.conclusion),
            "mpl": to_mpl(expression),
        }
    assert isinstance(expression, Negation)
    return {
        "kind": "negation",
        "inner": expression_json(expression.inner),
        "mpl": to_mpl(expression),
    }
