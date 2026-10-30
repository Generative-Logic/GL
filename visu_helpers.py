# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# ------------------------------------------------------------------------------
#
# This software is also available under a commercial license. For details,
# see: https://generative-logic.com/license
#
# Contributions to this project must be made under the terms of the
# Contributor License Agreement (CLA). See the project's CONTRIBUTING.md file.

"""Readable form of MPL for the HTML proof export, plus anchor expansion.

The grey "Readable version" lines of the proof pages are built here: the
caption under every chapter title, the RULE / APPLIED line under every
implication row and the reformulation lines. Expressions are parsed with the
strict MPL parser of `proof_export.mpl`; numbered compiled names (`and<N>`,
`or<N>`, `existence<N>`, `implication<N>`) are unfolded from the GL binaries
until a named definition is met; every named definition renders through its
own prepared mathematical template and is never unfolded; and an implication
chain renders as "from <conditions> follows <head>" after the variables that
its premises define (`f(a) = b`, `a + b = c`, a fold, a quotient or a
remainder) have been substituted into the rest.

Every caption has two spellings: HTML for the pages and plain Unicode text for
the browser-tab title.
"""

from __future__ import annotations

import html as html_lib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import expression_utils
from proof_export.mpl import (Atom, Conjunction, Implication, Negation,
                              map_arguments, parse_mpl)


# ---------------------------------------------------------------------------
# Spelling: the two output forms
# ---------------------------------------------------------------------------

_SUBSCRIPT_DIGITS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
_SUBSCRIPT_FORMS = dict(zip("0123456789ik", "₀₁₂₃₄₅₆₇₈₉ᵢₖ"))
# Letters a reading template may bind. Proof pages name their variables
# v, w, V, W (proof graph), y, Y (numbered unfolding) with digit subscripts and
# a fold's index i; a template's own names come from this set only, so they
# never meet a name of the surrounding formula.
_TEMPLATE_LETTERS = frozenset("dkmnp")
_INDEXED_TOKEN = re.compile(r"([vVwWyY])(\d+)((?:_copy)*)")
_ANCHOR_NUMERAL = re.compile(r"i(\d+)((?:_copy)*)")
_COMPILED_NAME = re.compile(r"(and|or|existence|implication)\d+")
_ENUMERATION_SET = re.compile(r"EnumerationSet\d+")
_ENUMERATION_RELATION_SET = re.compile(r"EnumerationRelationSet\d+")
# Relations whose MPL already reads plainly; a row showing one needs no
# "READS:" caption.
_PLAIN_RELATIONS = frozenset({"in", "=", "in2", "in3", "preorder", "strictOrder"})


def _leading_span_end(html: str) -> int:
    """
    @brief Find where the span that opens a formula fragment closes.

    @details Formula HTML nests only `span` elements (a fence's vector curve
    is an `svg` inside a span), so counting opening and closing span tags
    from the start finds the end of the first element. The fragment must
    begin with a span that closes inside it.

    @param html Formula HTML that begins with a `span` element.
    @return Index just past the closing tag of the leading span.
    """
    assert html.startswith("<span"), "the fragment must begin with a span"
    depth = 0
    for tag in re.finditer(r"</?span\b[^>]*>", html):
        depth += -1 if tag.group().startswith("</") else 1
        if depth == 0:
            return tag.end()
    assert False, f"Unclosed leading span in formula HTML: {html!r}"


class _Spelling:
    """Leaf spellings of one output form: HTML for pages, Unicode text for titles."""

    def __init__(self, as_html: bool) -> None:
        """
        @brief Choose the output form.

        @details HTML spelling escapes text and marks indices, the natural
        numbers, keywords, labels and the large operators with the classes of
        the page style; text spelling uses Unicode subscript digits and inline
        bounds.

        @param as_html True for HTML, False for plain Unicode text.
        @return None.
        """
        self.as_html = as_html

    def text(self, value: str) -> str:
        """
        @brief Spell literal text.

        @details HTML escapes `<`, `>` and `&`; text keeps the value unchanged.

        @param value Literal text.
        @return The spelled text.
        """
        return html_lib.escape(value, quote=False) if self.as_html else value

    def indexed(self, letter: str, digits: str, copies: int) -> str:
        """
        @brief Spell an indexed variable such as v1 as v with subscript 1.

        @details HTML uses a `sub` element; text uses Unicode subscript digits.
        A copy carries the copy mark (a tilde) over its letter.

        @param letter Variable letter.
        @param digits Index digits.
        @param copies Number of `_copy` suffixes.
        @return The spelled variable.
        """
        letter = _copy_marked(letter, copies)
        if self.as_html:
            return f"{self.text(letter)}<sub>{digits}</sub>"
        return letter + digits.translate(_SUBSCRIPT_DIGITS)

    def subscript(self, letter: str, index: str) -> str:
        """
        @brief Spell a template-bound sequence member such as p with index k.

        @details A reading template that quantifies a sequence names its
        members `p₀`, `p₁`, `pₖ`. HTML uses a `sub` element; text uses the
        Unicode subscript forms, which exist for the digits and the index
        letter `k`.

        @param letter Member letter, one of `_TEMPLATE_LETTERS`.
        @param index Index: digits, the letter `k` or a product's index `i`.
        @return The spelled member.
        @invariant The letter is a template letter and every index character
        has a Unicode subscript form.
        """
        assert letter in _TEMPLATE_LETTERS, f"{letter!r} is not a template letter"
        if self.as_html:
            return f"{self.text(letter)}<sub>{self.text(index)}</sub>"
        assert all(character in _SUBSCRIPT_FORMS for character in index), (
            f"no subscript form for the index {index!r}")
        return letter + "".join(_SUBSCRIPT_FORMS[character] for character in index)

    def naturals(self) -> str:
        """
        @brief Spell the natural-numbers carrier as a blackboard-bold N.

        @details HTML wraps the glyph in the `bb-N` class that thickens it.

        @return The spelled carrier.
        """
        return '<span class="bb-N">ℕ</span>' if self.as_html else "ℕ"

    def keyword(self, word: str, leading: bool) -> str:
        """
        @brief Spell a sentence keyword (`from`, `follows`).

        @details HTML marks it with the `rf-kw` class, whose style supplies
        the spacing (`rf-lead` drops the left space of an opening keyword);
        text separates it by single spaces.

        @param word Keyword.
        @param leading True when the keyword opens the caption.
        @return The spelled keyword including its spacing.
        """
        if self.as_html:
            return f'<span class="rf-kw{" rf-lead" if leading else ""}">{word}</span>'
        return f"{word} " if leading else f" {word} "

    def label(self, word: str, leading: bool) -> str:
        """
        @brief Spell a caption label (`RULE:`, `APPLIED:`, ...).

        @details HTML marks it with the `rf-label` class (`rf-lead` drops the
        left space of an opening label); text separates it by spaces.

        @param word Label text including its colon.
        @param leading True when the label opens the caption.
        @return The spelled label.
        """
        if self.as_html:
            return f'<span class="rf-label{" rf-lead" if leading else ""}">{word}</span>'
        return f"{word} " if leading else f"   {word} "

    def big_operator(self, symbol: str, lower: str, upper: str, body: str) -> str:
        """
        @brief Spell a sum or product with its bounds.

        @details HTML stacks the upper bound, the symbol and `i = lower`
        vertically; text writes `∑(i=lower..upper) body`.

        @param symbol `∑` or `∏`.
        @param lower Spelled lower bound.
        @param upper Spelled upper bound.
        @param body Spelled summand or factor.
        @return The spelled large operator.
        """
        if self.as_html:
            return (f'<span class="rf-big"><span class="rf-big-hi">{upper}</span>'
                    f'<span class="rf-big-sym">{symbol}</span>'
                    f'<span class="rf-big-lo">i = {lower}</span></span> {body}')
        return f"{symbol}(i={lower}..{upper}) {body}"

    def nowrap(self, text: str) -> str:
        """
        @brief Keep a formula part on one line.

        @details HTML wraps it in the `rf-nb` class (`white-space: nowrap`),
        so a caption breaks only between premises, connectives and keywords,
        never inside a relation or between a large operator and its operand;
        text is unchanged.

        @param text Spelled formula part.
        @return The protected part.
        """
        return f'<span class="rf-nb">{text}</span>' if self.as_html else text

    def unit(self, text: str) -> str:
        """
        @brief Keep a short formula unit on one line at every width.

        @details A mapping pair `a ↦ b`, an interval `[a, b]`, a short set
        `{a, b}` or `{x ∈ X : x ≤ n}`, a restriction or a substitution `w = v`
        reads as one symbol. HTML wraps it in the `rf-unit` class
        (`white-space: nowrap`), which, unlike `rf-nb`, stays in force on
        phones, where formula pieces may otherwise wrap between words; text is
        unchanged.

        @param text Spelled unit.
        @return The protected unit.
        """
        return f'<span class="rf-unit">{text}</span>' if self.as_html else text

    def line_break(self) -> str:
        """
        @brief Start a new line inside a caption.

        @details HTML emits a `br` element; text separates by three spaces.

        @return The line break.
        """
        return "<br>" if self.as_html else "   "

    def big_parentheses(self, inner: str) -> str:
        """
        @brief Parenthesize a large operator used as an operand.

        @details HTML enlarges the parentheses to the height of the stacked
        bounds and keeps the opening parenthesis on one line with the stacked
        symbol (`rf-unit`): the symbol is a box of its own, before which a
        line may otherwise break. Text uses ordinary parentheses.

        @param inner Spelled large operator, beginning with its stacked symbol.
        @return The parenthesized operand.
        """
        if self.as_html:
            assert inner.startswith('<span class="rf-big">'), "a large operator begins with its symbol"
            end = _leading_span_end(inner)
            return (self.unit(f'<span class="rf-bigparen">(</span>{inner[:end]}')
                    + f'{inner[end:]}<span class="rf-bigparen">)</span>')
        return f"({inner})"

    def parentheses(self, inner: str) -> str:
        """
        @brief Enclose a formula with fences matching its rendered height.

        @details Multiline implication and conjunction HTML use stretched vector curves;
        ordinary inline formulas and plain text retain literal parentheses.
        Hidden text preserves the parentheses for assistive reading and copying.
        An already enclosed complete group keeps its existing pair; adjoining
        grouped expressions still receive their own enclosing pair when needed.

        @param inner Complete formatted formula, including nested implication grids.
        @return The same formula enclosed by a matched pair of parentheses.
        """
        if (self.as_html and inner.startswith('<span class="rf-parentheses">')
                and _leading_span_end(inner) == len(inner)):
            return inner
        if self.as_html and any(marker in inner for marker in
                                ("rf-implication", "rf-conjunction", "rf-premises", "rf-lines")):
            left = ('<span class="rf-fence"><span class="rf-fence-text">(</span>'
                    '<svg viewBox="0 0 12 100" preserveAspectRatio="none" aria-hidden="true">'
                    '<path d="M10 1 C2 18 2 82 10 99"/></svg></span>')
            right = ('<span class="rf-fence"><span class="rf-fence-text">)</span>'
                     '<svg viewBox="0 0 12 100" preserveAspectRatio="none" aria-hidden="true">'
                     '<path d="M2 1 C10 18 10 82 2 99"/></svg></span>')
            return (f'<span class="rf-parentheses">{left}'
                    f'<span class="rf-paren-content">{inner}</span>{right}</span>')
        return f"({inner})"

    def restriction(self, function: str, domain: str) -> str:
        """
        @brief Spell the restriction of a function to a domain.

        @details HTML writes the domain as a subscript after a bar and keeps
        function, bar and subscript on one line (`rf-unit`), since a line may
        otherwise break after the bar or inside the subscript; text writes
        `f|domain`.

        @param function Spelled function.
        @param domain Spelled domain description.
        @return The spelled restriction.
        """
        if self.as_html:
            return self.unit(f"{function}|<sub>{domain}</sub>")
        return f"{function}|{domain}"

    def negation(self, inner: str) -> str:
        """
        @brief Negate a formula, keeping the sign with its bracketed operand.

        @details The negation sign precedes the parenthesized formula. A
        stretched HTML bracket group is a box of its own, and a line may break
        before any box, which would leave the sign alone at a line end; HTML
        therefore sets sign and group side by side in one `rf-negation` row,
        in which the group narrows by the width of the sign. Literal
        parentheses and plain text keep the sign attached to `(` as ordinary
        text does.

        @param inner Complete formatted formula to negate.
        @return `¬` followed by the parenthesized formula.
        """
        group = self.parentheses(inner)
        if self.as_html and group.startswith('<span class="rf-parentheses">'):
            return f'<span class="rf-negation">¬{group}</span>'
        return "¬" + group


_HTML = _Spelling(True)
_TEXT = _Spelling(False)

# The copy mark: a combining tilde over every character of a copied symbol.
# A prime would read as the Peano successor (1′ = s(1)).
_COPY_MARK = "\u0303"


def _copy_marked(text: str, copies: int) -> str:
    """
    @brief Put the copy mark over every character of a symbol.

    @details Anchor handling makes fresh copies of symbols (`i1_copy`,
    `v3_copy`); each `_copy` suffix adds one combining tilde over every
    character, so `1̃` is a copy of 1 and never reads as its successor.

    @param text The symbol's spelling without the suffixes.
    @param copies Number of `_copy` suffixes.
    @return The marked symbol (unchanged for no copy).
    """
    if not copies:
        return text
    return "".join(character + _COPY_MARK * copies for character in text)


def display_token(token: str, spelling: _Spelling) -> str:
    """
    @brief Spell one MPL argument token for a readable caption.

    @details Anchor numerals `i<k>` become the digit string, indexed variables
    (`v`, `V`, `w`, `W`, `y`, `Y` with digits) get subscripts, the carrier `N`
    becomes ℕ, the multiplication slot `*` becomes `·`, and every `_copy`
    suffix adds the copy mark (`_copy_marked`). Any other token is spelled
    literally.

    @param token MPL argument token.
    @param spelling Output form.
    @return The spelled token.
    """
    match = _ANCHOR_NUMERAL.fullmatch(token)
    if match:
        return spelling.text(_copy_marked(match.group(1), len(match.group(2)) // 5))
    match = _INDEXED_TOKEN.fullmatch(token)
    if match:
        return spelling.indexed(match.group(1), match.group(2), len(match.group(3)) // 5)
    if token == "N":
        return spelling.naturals()
    if token == "*":
        return spelling.text("·")
    base = token
    copies = 0
    while base.endswith("_copy") and len(base) > 5:
        base = base[:-5]
        copies += 1
    return spelling.text(_copy_marked(base, copies))


# ---------------------------------------------------------------------------
# Terms: the values substitution builds
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Leaf:
    """A plain argument token."""

    token: str


@dataclass(frozen=True)
class _Apply:
    """A function application `f(a)` (from `in2[a,b,f]`)."""

    function: object
    argument: object


@dataclass(frozen=True)
class _Operation:
    """A binary operation `a op b` (from `in3`, `quotient`, `remainder`)."""

    operator: str
    left: object
    right: object


@dataclass(frozen=True)
class _Big:
    """A sum or product `∑/∏_{i=lower}^{upper} f(i)` (from a standard fold)."""

    operation: str
    function: str
    lower: object
    upper: object


@dataclass(frozen=True)
class _Fold:
    """A fold with a non-standard traversal, kept in explicit notation."""

    slots: tuple


_INFIX_PRECEDENCE = {"+": 1, "*": 2, "div": 2, "mod": 2}
_INFIX_SYMBOL = {"+": "+", "*": "·", "div": "div", "mod": "mod"}


def _term_precedence(term: object) -> int:
    """
    @brief Binding strength of a term for parenthesization.

    @details Sums and products bind loosest (they extend to the right),
    addition next, multiplication, quotient and remainder next, and leaves,
    applications and prefix operations tightest.

    @param term A term.
    @return 0 for a large operator, 1 or 2 for infix operations, 3 otherwise.
    """
    if isinstance(term, _Big):
        return 0
    if isinstance(term, _Operation) and term.operator in _INFIX_PRECEDENCE:
        return _INFIX_PRECEDENCE[term.operator]
    return 3


def _term_tokens(term: object) -> set[str]:
    """
    @brief Collect the argument tokens a term mentions.

    @details Used by the usage and cycle analysis of substitutions.

    @param term A term.
    @return Set of tokens.
    """
    if isinstance(term, _Leaf):
        return {term.token}
    if isinstance(term, _Apply):
        return _term_tokens(term.function) | _term_tokens(term.argument)
    if isinstance(term, _Operation):
        return {term.operator} | _term_tokens(term.left) | _term_tokens(term.right)
    if isinstance(term, _Big):
        return {term.function} | _term_tokens(term.lower) | _term_tokens(term.upper)
    assert isinstance(term, _Fold), f"unknown term {term!r}"
    tokens: set[str] = set()
    for slot in term.slots:
        tokens |= _term_tokens(slot)
    return tokens


def _substitute_term(term: object, values: dict[str, object]) -> object:
    """
    @brief Replace defined tokens of a term by their values.

    @details Values must already be fully resolved; the result is then fully
    resolved too.

    @param term A term.
    @param values Resolved values of the substituted variables.
    @return The substituted term.
    """
    if isinstance(term, _Leaf):
        return values.get(term.token, term)
    if isinstance(term, _Apply):
        return _Apply(_substitute_term(term.function, values),
                      _substitute_term(term.argument, values))
    if isinstance(term, _Operation):
        return _Operation(term.operator, _substitute_term(term.left, values),
                          _substitute_term(term.right, values))
    if isinstance(term, _Big):
        return _Big(term.operation, term.function, _substitute_term(term.lower, values),
                    _substitute_term(term.upper, values))
    assert isinstance(term, _Fold), f"unknown term {term!r}"
    return _Fold(tuple(_substitute_term(slot, values) for slot in term.slots))


def _fold_value(arguments: tuple[str, ...]) -> object:
    """
    @brief The value of an eight-argument fold `fold[N,s,+,op,f,n,m,p]`.

    @details An ordinary successor traversal over an additive interval with
    `op` addition or multiplication is a sum or product of `f(i)` for `i`
    from `n` to `m`, both bounds included; every other fold keeps its first
    seven slots in explicit notation.

    @param arguments The fold's eight arguments.
    @return A `_Big` or `_Fold` term.
    @invariant The fourth argument alone selects accumulation (I-257).
    """
    assert len(arguments) == 8, f"fold requires eight arguments: {arguments!r}"
    _domain, successor, addition, operation, function, lower, upper, _result = arguments
    if successor == "s" and addition == "+" and operation in ("+", "*"):
        return _Big(operation, function, _Leaf(lower), _Leaf(upper))
    return _Fold(tuple(_Leaf(token) for token in arguments[:7]))


# Argument counts of the predicates that have a reading template. A template
# predicate with another count is a malformed expression, never a reading.
# Every named definition has one: a name reads through its own template and
# is never unfolded; only numbered compiled names unfold (`_COMPILED_NAME`).
_TEMPLATE_ARITY = {
    "=": 2, "in": 2, "in2": 3, "in3": 4, "fold": 8, "quotient": 6, "remainder": 6,
    "preorder": 4, "strictOrder": 4, "interval": 5, "limitSequence": 5, "split": 6,
    "limitSet": 5, "appendToSequence": 4, "addToSet": 3, "identity": 2, "fXY": 3,
    "fXYZ": 4, "sequence": 5, "prime": 6, "nontrivialFactor": 6,
    "residual": 4, "infiniteSequence": 2, "constSeq": 7, "nonInterval": 8, "nonSequence": 9,
    "bezEq": 8, "bez1": 6, "coprime": 6, "inSequence": 3, "primeseq": 6, "ordered": 3,
    "subset": 2, "empty": 1, "minimum": 4, "hasMinimum": 3, "divisorSet": 6,
    "primeFactorSeq": 10, "hasPrimeFactorization": 8,
}

# Named definitions read by `ReadableForm._named`: number theory, sequences,
# sets and the incubator definitions.
_NAMED_TEMPLATES = frozenset({
    "residual", "infiniteSequence", "constSeq", "nonInterval", "nonSequence",
    "bezEq", "bez1", "coprime", "inSequence", "primeseq", "ordered", "subset", "empty",
    "minimum", "hasMinimum", "divisorSet", "primeFactorSeq", "hasPrimeFactorization",
})


def _has_template(head: str) -> bool:
    """
    @brief Whether a predicate reads through a template of `ReadableForm`.

    @details Relations, orders, folds and every named definition
    (`_TEMPLATE_ARITY`), enumerations, `NaturalNumbers` and the anchors. The
    only other predicates of a proof graph are the numbered compiled names
    (`and<N>`, `or<N>`, `existence<N>`, `implication<N>`), which expand from
    the GL binaries.

    @param head Predicate name.
    @return True for a template predicate.
    """
    return (head in _TEMPLATE_ARITY or head == "NaturalNumbers" or head.startswith("Anchor")
            or bool(_ENUMERATION_SET.fullmatch(head))
            or bool(_ENUMERATION_RELATION_SET.fullmatch(head)))


def _check_arity(atom: Atom) -> None:
    """
    @brief Assert that a template predicate carries its argument count.

    @details The reading templates address arguments by position, so a
    template predicate with another number of arguments cannot be read; it
    asserts instead of falling through to another reading. Enumerations,
    anchors and compiled names have no fixed count and are not constrained
    here.

    @param atom An atom.
    @invariant `len(atom.arguments) == _TEMPLATE_ARITY[atom.head]` for every
    template predicate.
    """
    expected = _TEMPLATE_ARITY.get(atom.head)
    assert expected is None or len(atom.arguments) == expected, (
        f"{atom.head} needs {expected} arguments: {atom.arguments!r}")


def _output_definition(atom: Atom) -> tuple[str, object] | None:
    """
    @brief The variable a positive atom defines, with its defining value.

    @details `in2[a,b,f]` defines `b = f(a)`, `in3[a,b,c,op]` defines
    `c = a op b`, a fold defines its result, `quotient[N,+,*,a,d,q]` defines
    `q = a div d` and `remainder[N,+,*,a,d,r]` defines `r = a mod d`. Every
    other atom defines nothing; that is an ordinary, defined outcome.

    @param atom An atom.
    @return `(variable, value term)` or None when the atom defines nothing.
    """
    _check_arity(atom)
    head, arguments = atom.head, atom.arguments
    if head == "in2":
        return arguments[1], _Apply(_Leaf(arguments[2]), _Leaf(arguments[0]))
    if head == "in3":
        return arguments[2], _Operation(arguments[3], _Leaf(arguments[0]), _Leaf(arguments[1]))
    if head == "fold":
        value = _fold_value(arguments)
        return arguments[7], value
    if head in ("quotient", "remainder"):
        operator = "div" if head == "quotient" else "mod"
        return arguments[5], _Operation(operator, _Leaf(arguments[3]), _Leaf(arguments[4]))
    return None


# ---------------------------------------------------------------------------
# Formulas and chains
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Disjunction:
    """A disjunction of signed parts (from a compiled `or<N>` or a negated conjunction)."""

    parts: tuple


@dataclass(frozen=True)
class _Existence:
    """`∃ variables: part1 ∧ part2 ∧ …` (from a compiled or a raw existence)."""

    variables: tuple[str, ...]
    parts: tuple


# Formula binding strengths: relations 4, conjunction 3, disjunction 2,
# function-type statements 1 (`f: X → Y`, parenthesized inside ∧ and ∨),
# quantified formulas and implications 0.
_RELATION, _AND, _OR, _TYPE, _QUANTIFIED = 4, 3, 2, 1, 0


@dataclass(frozen=True)
class _Chain:
    """A flattened implication: bound variables, premises and head."""

    bound: tuple[str, ...]
    premises: tuple
    head: object
    anchor_slots: frozenset


def _flatten_chain(expression: object) -> _Chain:
    """
    @brief Flatten nested implications into one premise chain.

    @details `(>[a](P)(>[b](Q)(H)))` becomes bound `(a, b)`, premises
    `(P, Q)` and head `H`. A name bound again (a multiplied rule can bind the
    same variable twice) is listed once: binders form one flat set, as in
    the verifier. A leading anchor premise (`Anchor…[…]`) is dropped and the
    variables of its binder become anchor slots, which are constants of the
    caption and never substituted.

    @param expression Parsed expression; a non-implication has no premises.
    @return The flattened chain.
    """
    bound: list[str] = []
    premises: list[object] = []
    anchor_slots: frozenset = frozenset()
    node = expression
    first = True
    while isinstance(node, Implication):
        if first and _is_anchor(node.premise):
            anchor_slots = frozenset(node.bound_variables)
        else:
            bound.extend(name for name in node.bound_variables if name not in bound)
            premises.append(node.premise)
        first = False
        node = node.conclusion
    return _Chain(tuple(bound), tuple(premises), node, anchor_slots)


def _is_anchor(expression: object) -> bool:
    """
    @brief Whether an expression is an anchor atom.

    @param expression Parsed expression.
    @return True for `(Anchor…[…])`.
    """
    return isinstance(expression, Atom) and expression.head.startswith("Anchor")


def _conjunction_parts(node: Conjunction) -> list[object]:
    """
    @brief The parts of a conjunction with nested conjunctions flattened.

    @param node Conjunction node.
    @return Ordered list of non-conjunction parts.
    """
    parts: list[object] = []
    for part in node.parts:
        if isinstance(part, Conjunction):
            parts.extend(_conjunction_parts(part))
        else:
            parts.append(part)
    return parts


def _atoms_of(expression: object):
    """
    @brief Iterate the atoms of an MPL expression in syntax order.

    @param expression Parsed MPL expression.
    @return Iterator over atoms.
    """
    if isinstance(expression, Atom):
        yield expression
    elif isinstance(expression, Conjunction):
        for part in expression.parts:
            yield from _atoms_of(part)
    elif isinstance(expression, Implication):
        yield from _atoms_of(expression.premise)
        yield from _atoms_of(expression.conclusion)
    else:
        assert isinstance(expression, Negation), f"unknown MPL node {expression!r}"
        yield from _atoms_of(expression.inner)


SHARED_DEFINITION_REGISTRY = "GL_binary_shared.json"


def read_shared_definition_registry(gl_binaries_dir) -> dict[str, dict]:
    """
    @brief Read the run's shared compiled-definition registry.

    @details After every batch the pipeline merges each non-atomic entry of the
    batch registry into `GL_binary_shared.json` (`run_modes._merge_into_shared`),
    and a name keeps one definition across batches, so the shared file alone
    holds every definition a proof page can name; the per-batch files add only
    atomic relations. A missing directory is a graph without compiled names and
    yields an empty registry; a directory without the shared file asserts.

    @param gl_binaries_dir Directory holding the GL binary JSON files.
    @return Mapping from definition name to its registry entry, as stored.
    @invariant Callers select the non-atomic entries themselves.
    """
    directory = Path(gl_binaries_dir)
    if not directory.exists():
        return {}
    path = directory / SHARED_DEFINITION_REGISTRY
    assert path.is_file(), f"missing shared definition registry {path}"
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_compiled_definitions(gl_binaries_dir) -> dict[str, dict]:
    """
    @brief Read the compiled-name and user definitions from the GL binaries.

    @details Every entry of the shared registry that is not atomic — the
    compiled names `and<N>`, `or<N>`, `existence<N>`, `implication<N>` and
    the user definitions (`interval`, `residual`, …) — is kept with its
    category, signature and elements, the same entries
    `generate_full_proof_graph.build_gl_binary_map` renders. A missing
    directory is a graph without compiled names and yields an empty registry.

    @param gl_binaries_dir Directory holding the GL binary JSON files.
    @return Mapping from definition name to its definition entry.
    @see read_shared_definition_registry
    """
    definitions: dict[str, dict] = {}
    for name, entry in read_shared_definition_registry(gl_binaries_dir).items():
        if entry["category"] != "atomic":
            definitions[name] = {
                "category": entry["category"],
                "signature": entry["signature"],
                "elements": list(entry["elements"]),
            }
    return definitions


_ELEMENT_SLOTS = {
    "in": (0,), "in2": (0, 1), "in3": (0, 1, 2), "=": (0, 1),
    "preorder": (2, 3), "strictOrder": (2, 3), "interval": (2, 3), "sequence": (2, 3),
    "limitSequence": (2,), "limitSet": (3,), "appendToSequence": (0, 1), "addToSet": (1,),
    "split": (2, 3), "quotient": (3, 4, 5), "remainder": (3, 4, 5), "prime": (5,),
    "nontrivialFactor": (4, 5), "fold": (5, 6, 7),
    "residual": (0, 1, 2), "constSeq": (3, 4, 6), "nonInterval": (4, 6, 7),
    "nonSequence": (4, 6, 7, 8), "bezEq": (4, 5, 6, 7), "bez1": (4, 5), "coprime": (4, 5),
    "inSequence": (1,), "minimum": (3,), "divisorSet": (4,), "primeFactorSeq": (7, 9),
    "hasPrimeFactorization": (7,),
}


def _element_slots(atom: Atom) -> tuple[int, ...]:
    """
    @brief Argument positions of an atom that hold elements (not sets).

    @details Decides whether a bound digit of a compiled definition names an
    element (`y`) or a set or function (`Y`): the element positions of the
    relations and of every user-definition template (order arguments,
    interval and sequence bounds, fold bounds and result, quotient and
    remainder operands, …).

    @param atom An atom of a compiled definition.
    @return The element positions.
    """
    return _ELEMENT_SLOTS.get(atom.head, ())


def _instantiate(entry: dict, atom: Atom, rename,
                 anchor_signatures: dict[str, tuple[str, ...]]) -> tuple[str, list[object], list[str], list[str]]:
    """
    @brief Instantiate a compiled definition's elements for one atom.

    @details The signature's `u_k` parameters take the atom's arguments. A
    compiled theorem (an implication whose first element is an anchor over
    bound digits, such as `implication2525[]`) gives those digits the
    anchor's canonical slot names (`N`, `i0`, `s`, `+`, `*`, …), so they read
    as the constants they are. Every other bound digit takes
    `rename(digit, is_element)`; it is an element when, for an existence,
    its first element types it as `in[d, S]`, and otherwise when it occurs
    in an element slot.

    @param entry Binary entry with category, signature and elements.
    @param atom The compiled-name atom.
    @param rename Callback naming a bound digit: `(digit, is_element) -> name`.
    @param anchor_signatures Anchor name → canonical slot names.
    @return `(category, instantiated elements, anchor slot names, other bound
    names in digit order)`.
    @invariant Every parameter of the elements belongs to the signature, and
    every anchor of a compiled theorem has its canonical slot names.
    """
    signature = parse_mpl(entry["signature"])
    assert isinstance(signature, Atom) and len(signature.arguments) == len(atom.arguments), (
        f"compiled name {atom.head} used with the wrong arity: {atom.arguments!r}")
    mapping = dict(zip(signature.arguments, atom.arguments))
    elements = [parse_mpl(element) for element in entry["elements"]]
    digits: list[str] = []
    element_digits: set[str] = set()
    for element in elements:
        for found in _atoms_of(element):
            for position, token in enumerate(found.arguments):
                if token.isdigit():
                    if token not in digits:
                        digits.append(token)
                    if position in _element_slots(found):
                        element_digits.add(token)
                else:
                    assert not token.startswith("u_") or token in mapping, (
                        f"compiled name {atom.head}: parameter {token} not in its signature")
    category = entry["category"]
    if category == "existence":
        first = elements[0]
        typed = isinstance(first, Atom) and first.head == "in" and first.arguments[0].isdigit()
        element_digits = {first.arguments[0]} if typed else set()
    names: dict[str, str] = {}
    anchor_names: list[str] = []
    if category == "implication" and _is_anchor(elements[0]):
        anchor = elements[0]
        assert anchor.head in anchor_signatures, f"anchor without canonical slot names: {anchor.head}"
        canonical = anchor_signatures[anchor.head]
        assert len(canonical) == len(anchor.arguments), f"anchor arity mismatch: {anchor!r}"
        for position, token in enumerate(anchor.arguments):
            if token.isdigit():
                names[token] = canonical[position]
                anchor_names.append(canonical[position])
    for digit in digits:
        if digit not in names:
            names[digit] = rename(digit, digit in element_digits)
    parts = [map_arguments(element, lambda token: mapping.get(token, names.get(token, token)))
             for element in elements]
    others = [names[digit] for digit in digits if names[digit] not in anchor_names]
    return category, parts, anchor_names, others


def _expanded_atoms(node: object, definitions: dict[str, dict],
                    anchor_signatures: dict[str, tuple[str, ...]], sign: bool = True):
    """
    @brief Iterate `(atom, positive)` pairs with compiled names expanded.

    @details Compiled names are replaced by their instantiated elements
    (bound digits keep their digit names, which never collide with proof
    variables); user predicates and relations stay atoms. Used to see where
    a variable really occurs inside compiled names. Compiled definitions are
    acyclic, so the recursion terminates.

    @param node MPL node or display node.
    @param definitions Compiled-name definitions.
    @param anchor_signatures Anchor name → canonical slot names.
    @param sign Whether the node occurs positively.
    @return Iterator over `(atom, positive)`.
    """
    if isinstance(node, Atom):
        if _COMPILED_NAME.fullmatch(node.head):
            entry = definitions.get(node.head)
            assert entry is not None, f"compiled name without a GL binary definition: {node.head}"
            _category, parts, _anchor, _names = _instantiate(
                entry, node, lambda digit, _element: digit, anchor_signatures)
            for part in parts:
                yield from _expanded_atoms(part, definitions, anchor_signatures, sign)
        else:
            yield node, sign
    elif isinstance(node, Negation):
        yield from _expanded_atoms(node.inner, definitions, anchor_signatures, not sign)
    elif isinstance(node, Conjunction):
        for part in node.parts:
            yield from _expanded_atoms(part, definitions, anchor_signatures, sign)
    elif isinstance(node, Implication):
        yield from _expanded_atoms(node.premise, definitions, anchor_signatures, not sign)
        yield from _expanded_atoms(node.conclusion, definitions, anchor_signatures, sign)
    else:
        assert False, f"unknown node {node!r}"


# The anchor operations: total on the naturals, so a value built from them
# needs no domain condition.
_TOTAL_OPERATIONS = frozenset({"s", "+", "*"})


def _restricted(token: str, chain: _Chain, index: int) -> bool:
    """
    @brief Whether a premise other than the one at `index` mentions `token`.

    @details Such a premise restricts the argument of a definition (a typing,
    an order, a sequence statement, …), so reading `F(x)` presupposes no more
    than the chain states.

    @param token Argument token.
    @param chain Flattened implication.
    @param index Premise excluded from the search.
    @return True when another premise mentions the token.
    """
    for other_index, premise in enumerate(chain.premises):
        if other_index == index:
            continue
        for atom in _atoms_of(premise):
            if token in atom.arguments:
                return True
    return False


def _same_application(first: Atom, second: Atom) -> bool:
    """
    @brief Whether two defining atoms apply one operation to the same arguments.

    @details `in2[a,b,F]` and `in2[a,c,F]` both state F(a); `in3[a,b,c,op]` and
    `in3[a,b,d,op]` both state `a op b`; folds, quotients and remainders with
    the same inputs likewise.

    @param first A defining atom.
    @param second Any atom.
    @return True when both atoms state a value of the same application.
    """
    if first.head != second.head or len(first.arguments) != len(second.arguments):
        return False
    inputs = {"in2": (0, 2), "in3": (0, 1, 3), "fold": tuple(range(7)),
              "quotient": tuple(range(5)), "remainder": tuple(range(5))}.get(first.head)
    if inputs is None:
        return False
    return all(first.arguments[k] == second.arguments[k] for k in inputs)


def _substitutable(chain: _Chain, index: int) -> bool:
    """
    @brief Whether the defining premise at `index` may be substituted away.

    @details Substituting `y := value` removes the premise, so the value must
    say everything the premise said:
    - `in2[x,y,F]` with F an anchor operation (`s`), `in3[a,b,c,op]` with
      `op` one of `+`, `·`: total operations, always defined;
    - `in2[x,y,F]` with any other F (a sequence, a relation, `id`): only when
      another premise restricts `x`, else "∀ x: F(x) = …" would claim F is
      defined everywhere (a subset relation `F ⊆ G` would read `F = G`);
    - a fold: when its function is `id` or another premise mentions it (its
      sequence statement), else the product's existence would vanish;
    - quotient and remainder never: `a div d` presupposes `d ≠ 0`.
    In addition the defined variable must not be typed `y ∈ ℕ` by a premise
    (the typing would turn into `s(x) ∈ ℕ`), and no other premise or the
    head may state a value of the same application (`s(w1) = w2` beside
    `s(w1) = w1` would make the rule read as a tautology).

    @param chain Flattened implication.
    @param index Index of a positive defining premise.
    @return True when the premise may be substituted.
    @invariant `_output_definition(chain.premises[index])` is not None.
    """
    premise = chain.premises[index]
    definition = _output_definition(premise)
    assert definition is not None, f"not a defining premise: {premise!r}"
    head, arguments = premise.head, premise.arguments
    if head in ("quotient", "remainder"):
        return False
    if head == "in3" and arguments[3] not in _TOTAL_OPERATIONS:
        return False
    if (head == "in2" and arguments[2] not in _TOTAL_OPERATIONS
            and not _restricted(arguments[0], chain, index)):
        return False
    if head == "fold" and arguments[4] != "id" and not _restricted(arguments[4], chain, index):
        return False
    variable = definition[0]
    others = [other for other_index, other in enumerate(chain.premises) if other_index != index]
    for other in others + [chain.head]:
        if not isinstance(other, Atom):
            continue
        if other.head == "in" and other.arguments == (variable, "N"):
            return False
        if _same_application(premise, other):
            return False
    return True


def _definitions(chain: _Chain, definitions: dict[str, dict],
                 anchor_signatures: dict[str, tuple[str, ...]],
                 outer: dict[str, object]) -> dict[str, object]:
    """
    @brief The substitutions of a chain: defined variable → resolved value.

    @details Candidates are the bound, non-anchor result variables of
    positive defining premises (`_output_definition`) that `_substitutable`
    admits, the first definition winning. A candidate is dropped when
    - it would close a cycle of definitions;
    - another candidate resolves to the same value (premises `f(a) = b` and
      `f(a) = c` stay explicit, else the chain collapses to `f(a) = f(a)`);
    - nothing else uses its variable (the premise then still carries a domain
      condition such as "s(w1) is defined");
    - its variable occurs elsewhere only as a typed element `x ∈ S` (the
      premise is then the subject of the chain, as in "from w1 + w2 = w3
      follows w3 ∈ ℕ");
    - its variable is the result of another defining atom whose value equals
      its own (`V(w1) = w2` substituted into `V(w1) = w2 ∨ …` would read
      `V(w1) = V(w1)`).
    Occurrences are read with compiled names expanded. The checks repeat
    until stable; every drop turns a premise back into a condition.

    @param chain Flattened implication.
    @param definitions Compiled-name definitions.
    @param anchor_signatures Anchor name → canonical slot names.
    @param outer Substitutions of an enclosing caption (already resolved).
    @return Resolved values of the kept substitutions (outer values applied).
    """
    substitutable = set(chain.bound) - set(chain.anchor_slots)
    candidates: dict[str, tuple[int, object]] = {}
    for index, premise in enumerate(chain.premises):
        if isinstance(premise, Atom):
            definition = _output_definition(premise)
            if (definition is not None and definition[0] in substitutable
                    and definition[0] not in candidates and _substitutable(chain, index)):
                candidates[definition[0]] = (index, definition[1])
    kept: dict[str, tuple[int, object]] = {}
    for variable, candidate in candidates.items():
        if not _reaches(_term_tokens(candidate[1]), variable, kept):
            kept[variable] = candidate
    while True:
        values = _resolve(kept, outer)
        dropped: set[str] = set()
        by_value: dict[object, list[str]] = {}
        for variable, resolved in values.items():
            by_value.setdefault(resolved, []).append(variable)
        for group in by_value.values():
            if len(group) > 1:
                dropped.update(group)
        defining = {index for index, _value in kept.values()}
        rest = [chain.head] + [premise for index, premise in enumerate(chain.premises)
                               if index not in defining]
        occurrences: dict[str, list[tuple[Atom, int]]] = {}
        for node in rest:
            for atom, _positive in _expanded_atoms(node, definitions, anchor_signatures):
                for position, token in enumerate(atom.arguments):
                    occurrences.setdefault(token, []).append((atom, position))
        used: set[str] = set(occurrences)
        frontier = [token for token in used if token in kept]
        while frontier:
            for dependency in _term_tokens(kept[frontier.pop()][1]):
                if dependency not in used:
                    used.add(dependency)
                    if dependency in kept:
                        frontier.append(dependency)
        for variable in kept:
            if variable not in used:
                dropped.add(variable)
                continue
            places = occurrences.get(variable, [])
            if places and all(atom.head == "in" and position == 0 for atom, position in places):
                dropped.add(variable)
                continue
            for atom, position in places:
                definition = _output_definition(atom)
                if (definition is not None and definition[0] == variable
                        and _substitute_term(_substitute_term(definition[1], values), outer)
                        == values[variable]):
                    dropped.add(variable)
                    break
        if not dropped:
            return values
        for variable in dropped:
            del kept[variable]


def _inclusion(chain: _Chain) -> tuple[str, str] | None:
    """
    @brief Recognize `∀ x, y: F(x) = y ⇒ G(x) = y`, the inclusion `F ⊆ G`.

    @details The chain binds exactly `x` and `y` (anchor slots aside), its one
    premise is `in2[x,y,F]` and its head `in2[x,y,G]` with F, G different and
    not bound by the chain: every pair of F is a pair of G.

    @param chain Flattened implication.
    @return `(F, G)`, or None for any other chain.
    """
    if len(chain.premises) != 1:
        return None
    premise, head = chain.premises[0], chain.head
    if not (isinstance(premise, Atom) and isinstance(head, Atom)
            and premise.head == "in2" and head.head == "in2"):
        return None
    x, y, first = premise.arguments
    second = head.arguments[2]
    bound = set(chain.bound) - set(chain.anchor_slots)
    if (head.arguments[:2] != (x, y) or x == y or bound != {x, y}
            or first == second or first in bound or second in bound):
        return None
    return first, second


def _reaches(tokens: set[str], target: str, kept: dict[str, tuple[int, object]]) -> bool:
    """
    @brief Whether `target` is reachable from `tokens` through kept definitions.

    @param tokens Starting tokens.
    @param target Variable searched for.
    @param kept Kept definitions (variable → (premise index, value)).
    @return True when adding the definition would close a cycle.
    """
    seen: set[str] = set()
    frontier = list(tokens)
    while frontier:
        token = frontier.pop()
        if token == target:
            return True
        if token in seen:
            continue
        seen.add(token)
        if token in kept:
            frontier.extend(_term_tokens(kept[token][1]))
    return False


def _resolve(kept: dict[str, tuple[int, object]], outer: dict[str, object]) -> dict[str, object]:
    """
    @brief Fully resolve kept definitions into closed values.

    @details Definitions are acyclic, so resolution terminates; outer
    substitutions apply inside the values.

    @param kept Kept definitions (variable → (premise index, value)).
    @param outer Substitutions of an enclosing caption.
    @return Variable → value with every kept and outer variable substituted.
    """
    resolved: dict[str, object] = {}

    def resolve(variable: str) -> object:
        if variable not in resolved:
            value = kept[variable][1]
            inner = {token: resolve(token) for token in _term_tokens(value) if token in kept}
            resolved[variable] = _substitute_term(_substitute_term(value, inner), outer)
        return resolved[variable]

    for variable in kept:
        resolve(variable)
    return resolved


def _defining_premises(chain: _Chain, own: dict[str, object]) -> set[int]:
    """
    @brief Indices of the premises consumed by the kept substitutions.

    @details The first premise defining a substituted variable is consumed;
    a later premise with the same result stays as a condition.

    @param chain Flattened implication.
    @param own Substitutions kept for this chain.
    @return Premise indices that do not appear as conditions.
    """
    consumed: set[int] = set()
    seen: set[str] = set()
    for index, premise in enumerate(chain.premises):
        if isinstance(premise, Atom):
            definition = _output_definition(premise)
            if definition is not None and definition[0] in own and definition[0] not in seen:
                seen.add(definition[0])
                consumed.add(index)
    return consumed


# ---------------------------------------------------------------------------
# Rule matching
# ---------------------------------------------------------------------------

from proof_export.rule_matching import _binders_of, _unify_token, _unify, _rule_binding, _match_rule


# ---------------------------------------------------------------------------
# The renderer
# ---------------------------------------------------------------------------

class _Caption:
    """Per-caption state: spelling, fresh bound names and their display order."""

    def __init__(self, definitions: dict[str, dict], spelling: _Spelling) -> None:
        """
        @brief Start one caption.

        @details Bound variables of expanded compiled names get fresh internal
        names; on display they are numbered `y<k>` (elements) and `Y<k>`
        (sets and functions) in order of first appearance, so the numbering
        is dense even when substitution removes some of them.

        @param definitions Compiled-name definitions.
        @param spelling Output form.
        @return None.
        """
        self.definitions = definitions
        self.spelling = spelling
        self.minted = 0
        self.fresh_letter: dict[str, str] = {}
        self.display_names: dict[str, str] = {}
        self.display_counts = {"y": 0, "Y": 0}

    def fresh(self, element: bool) -> str:
        """
        @brief Mint a fresh internal name for a bound variable.

        @details Internal names start with `§`, which no MPL token contains.

        @param element True for an element, False for a set or function.
        @return The internal name.
        """
        self.minted += 1
        name = f"§{self.minted}"
        self.fresh_letter[name] = "y" if element else "Y"
        return name

    def token(self, token: str) -> str:
        """
        @brief Spell a token, numbering fresh names on first display.

        @param token Argument token or fresh internal name.
        @return The spelled token.
        """
        letter = self.fresh_letter.get(token)
        if letter is None:
            return display_token(token, self.spelling)
        if token not in self.display_names:
            self.display_counts[letter] += 1
            self.display_names[token] = f"{letter}{self.display_counts[letter]}"
        return display_token(self.display_names[token], self.spelling)

    def term(self, term: object, last: bool = True) -> str:
        """
        @brief Spell a term.

        @details Infix operands are parenthesized when they bind no tighter
        than the operation (grouping of equal operations stays visible, as in
        associativity laws); a large operator used as an operand gets
        enlarged parentheses unless nothing follows it (`2 · ∑ i`); function
        arguments need no extra parentheses.

        @param term A term.
        @param last True when nothing follows the term in its formula side.
        @return The spelled term.
        """
        spelling = self.spelling
        if isinstance(term, _Leaf):
            return self.token(term.token)
        if isinstance(term, _Apply):
            return f"{self.term(term.function)}({self.term(term.argument)})"
        if isinstance(term, _Operation):
            if term.operator in _INFIX_PRECEDENCE:
                precedence = _INFIX_PRECEDENCE[term.operator]
                left = self.operand(term.left, precedence, last=False)
                right = self.operand(term.right, precedence, last=last)
                return f"{left} {spelling.text(_INFIX_SYMBOL[term.operator])} {right}"
            return f"{self.token(term.operator)}({self.term(term.left)}, {self.term(term.right)})"
        if isinstance(term, _Big):
            symbol = "∑" if term.operation == "+" else "∏"
            lower = self.term(term.lower)
            upper = self.term(term.upper)
            body = "i" if term.function == "id" else f"{self.token(term.function)}(i)"
            return spelling.big_operator(symbol, lower, upper, body)
        assert isinstance(term, _Fold), f"unknown term {term!r}"
        return "fold(" + ", ".join(self.term(slot) for slot in term.slots) + ")"

    def operand(self, term: object, parent_precedence: int, last: bool) -> str:
        """
        @brief Spell an operand of an infix operation with its parentheses.

        @details A large operator extends to the right, so it needs its
        enlarged parentheses only when something follows it.

        @param term Operand term.
        @param parent_precedence Precedence of the enclosing operation.
        @param last True when nothing follows the operand in its formula side.
        @return The spelled operand.
        """
        rendered = self.term(term, last)
        if isinstance(term, _Big):
            return rendered if last else self.spelling.big_parentheses(rendered)
        if _term_precedence(term) <= parent_precedence:
            return f"({rendered})"
        return rendered


def _implication_columns(premises: list[str], conclusion: str) -> str:
    """
    @brief Lay out an implication with a premise column and a right-hand head.

    @details Every supplied premise retains its complete formatted expression.
    A large arrow joins the indented premise column to the conclusion without
    introducing textual line-number labels or changing mathematical scope.

    @param premises Nonempty formatted HTML premises in original order.
    @param conclusion Formatted HTML conclusion.
    @return Accessible inline group containing the complete implication.
    @invariant Neither premises nor conclusion are omitted or abbreviated.
    """
    assert premises and conclusion, "an implication needs premises and a head"
    if len(premises) == 1 and any(marker in premises[0] for marker in
                                  ("rf-implication", "rf-conjunction", "rf-premises")):
        premises = [_HTML.parentheses(premises[0])]
    items = "".join(f"<span class='rf-premise'>{item}</span>" for item in premises)
    group = "<span class='rf-premises'>" + items + "</span>"
    if len(premises) > 1:
        group = _HTML.parentheses(group)
    return ("<span class='rf-implication'>" + group
            + "<span class='rf-arrow' aria-label='implies'>⇒</span>"
            + f"<span class='rf-conclusion'><span class='rf-conclusion-content'>{conclusion}</span></span></span>")


def _continued_lines(first: str, continuation: str) -> str:
    """
    @brief Lay out a two-part formula as a line and an indented continuation.

    @details A long template reading (a product followed by "with primes …")
    breaks only at its own seam: the continuation starts a new, indented line,
    so the second part never reads as a separate premise or conjunct.

    @param first Formatted HTML of the first line.
    @param continuation Formatted HTML of the continuation line.
    @return One inline group holding both lines.
    @invariant Both parts are nonempty and keep their complete text.
    """
    assert first and continuation, "both lines of a continued formula need text"
    return ("<span class='rf-lines'><span class='rf-line'>" + first + "</span>"
            "<span class='rf-line rf-continued'>" + continuation + "</span></span>")


class ReadableForm:
    """Renderer of the grey readable captions for one proof graph."""

    def __init__(self, compiled_definitions: dict[str, dict],
                 anchor_signatures: dict[str, tuple[str, ...]],
                 structured_implications: bool = False) -> None:
        """
        @brief Create a renderer over the compiled-name definitions of a graph.

        @details The definitions come from `load_compiled_definitions`, the
        anchor signatures from the visualizer configuration (`N`, `i0`, `s`,
        `+`, `*`, … per anchor). Captions are cached per renderer because
        chapter pages repeat rules.

        @param compiled_definitions Compiled name → category, signature, elements.
        @param anchor_signatures Anchor name → canonical slot names.
        @param structured_implications Use recursive HTML premise and conjunction columns.
        @return None.
        """
        self.definitions = compiled_definitions
        self.anchor_signatures = anchor_signatures
        self.structured_implications = structured_implications
        self.cache: dict[tuple, str] = {}

    # -- public captions ------------------------------------------------------

    def theorem(self, expression: str, as_html: bool = True) -> str:
        """
        @brief Caption of a theorem: "from <conditions> follows <head>".

        @details The anchor premise is dropped and variables defined by
        premises are substituted (see `_definitions`), so a closed formula
        such as `v1 + v2 = v2 + v1` remains, with the premises that are real
        conditions (typing, order, sets, …) listed after `from`.

        @param expression Theorem in MPL.
        @param as_html True for the page spelling, False for the tab title.
        @return The caption.
        """
        key = ("theorem", expression, as_html)
        if key not in self.cache:
            caption = _Caption(self.definitions, _HTML if as_html else _TEXT)
            self.cache[key] = self._chain(_flatten_chain(parse_mpl(expression)), caption,
                                          substitute=True, nested=False)[0]
        return self.cache[key]

    def rule_application(self, derived: str, rule: str, premises: list[str],
                         as_html: bool = True) -> str:
        """
        @brief Caption of an implication row: the rule and its application.

        @details The rule renders like a theorem (closed form after
        substitution). The application, on its own line, is the instantiation
        of the rule's variables that remain in that rendering (`w1 = v4,
        w2 = v2`, from the verifier's binding, `_rule_binding`), so the rule
        as shown turns into the row by renaming; variables substituted away
        and unchanged names are left out, and a rule with nothing to
        instantiate shows no application.

        @param derived The row's derived expression.
        @param rule The cited implication.
        @param premises The row's premise expressions in row order.
        @param as_html True for the page spelling, False for plain text.
        @return The caption.
        @invariant The premises instantiate the rule exactly (the verifier's
        implication check); otherwise this asserts.
        """
        key = ("rule", derived, rule, tuple(premises), as_html)
        if key in self.cache:
            return self.cache[key]
        spelling = _HTML if as_html else _TEXT
        rule_expression = parse_mpl(rule)
        derived_expression = parse_mpl(derived)
        premise_expressions = [parse_mpl(premise) for premise in premises]
        match = _rule_binding(rule_expression, premise_expressions, derived_expression)
        assert match is not None, (
            f"implication row does not instantiate its rule: derived {derived!r}, "
            f"rule {rule!r}, premises {premises!r}")
        binding = match[1]
        chain = _flatten_chain(rule_expression)
        caption_state = _Caption(self.definitions, spelling)
        rule_text = self._chain(chain, caption_state, substitute=True, nested=False)[0]
        substituted = _definitions(chain, self.definitions, self.anchor_signatures, {})
        shown = [] if _inclusion(chain) is not None else [
            variable for variable in chain.bound
            if variable not in chain.anchor_slots and variable not in substituted
            and binding.get(variable, variable) != variable]
        caption = f"{spelling.label('RULE:', True)}{rule_text}"
        if shown:
            pairs = ", ".join(spelling.unit(f"{caption_state.token(variable)} = "
                                            f"{caption_state.token(binding[variable])}")
                              for variable in shown)
            caption += f"{spelling.line_break()}{spelling.label('APPLIED WITH:', True)}{pairs}"
        self.cache[key] = caption
        return caption

    def reading(self, expression: str, as_html: bool = True) -> str:
        """
        @brief Caption "READS: …" of a row without a rule, or "".

        @details The row's own expression in mathematical notation, for rows
        whose MPL is not already a plain relation: compiled names,
        definitions, quantified formulas and conjunctions get a reading; a
        plain relation (`in`, `=`, `in2`, `in3`, an order, or the negation of
        one) and an anchor need none, which is their defined rendering.

        @param expression The row's expression in MPL.
        @param as_html True for the page spelling, False for plain text.
        @return The caption, or "" for a plain relation or an anchor.
        @invariant A plain relation without a caption still carries its
        argument count.
        """
        key = ("reading", expression, as_html)
        if key not in self.cache:
            node = parse_mpl(expression)
            plain = node.inner if isinstance(node, Negation) else node
            if isinstance(plain, Atom) and (plain.head in _PLAIN_RELATIONS or _is_anchor(plain)):
                _check_arity(plain)
                self.cache[key] = ""
            else:
                spelling = _HTML if as_html else _TEXT
                caption = _Caption(self.definitions, spelling)
                text = self._formula(node, {}, caption)[0]
                self.cache[key] = f"{spelling.label('READS:', True)}{text}"
        return self.cache[key]

    def formula(self, expression: str, as_html: bool = True) -> str:
        """
        @brief The mathematical reading of an expression, without a label.

        @details The `READS` text for any expression, plain relations
        included; used for subproof titles and goal lines.

        @param expression Expression in MPL.
        @param as_html True for the page spelling, False for plain text.
        @return The reading.
        """
        key = ("formula", expression, as_html)
        if key not in self.cache:
            caption = _Caption(self.definitions, _HTML if as_html else _TEXT)
            self.cache[key] = self._formula(parse_mpl(expression), {}, caption)[0]
        return self.cache[key]

    def reformulation(self, statement: str, source: str, label: str,
                      as_html: bool = True) -> str:
        """
        @brief Caption of a reformulation row: both forms, literally.

        @details Both expressions render without substitution, so the change
        of form stays visible (substitution could make the two forms read
        alike).

        @param statement The reformulated expression.
        @param source The expression it was reformulated from.
        @param label Label between them, e.g. `REFORMULATED FROM:`.
        @param as_html True for the page spelling, False for plain text.
        @return The caption.
        """
        key = ("reformulation", statement, source, label, as_html)
        if key not in self.cache:
            spelling = _HTML if as_html else _TEXT
            first = self._chain(_flatten_chain(parse_mpl(statement)),
                                _Caption(self.definitions, spelling),
                                substitute=False, nested=False)[0]
            second = self._chain(_flatten_chain(parse_mpl(source)),
                                 _Caption(self.definitions, spelling),
                                 substitute=False, nested=False)[0]
            self.cache[key] = f"{first}{spelling.label(label, False)}{second}"
        return self.cache[key]

    # -- chains ---------------------------------------------------------------

    def _chain(self, chain: _Chain, caption: _Caption, substitute: bool, nested: bool,
               outer: dict[str, object] | None = None) -> tuple[str, int]:
        """
        @brief Spell a flattened implication.

        @details With substitution, the variables `_definitions` keeps are
        replaced by their values everywhere and their defining premises
        disappear. When the head's own result is substituted the head reads
        "value = head value" (and an equation head puts its substituted side
        first), so the given form stands left of the derived one. A top-level
        chain reads "from C1, C2 follows H"; a nested chain reads
        "∀ vars: C1 ∧ C2 ⇒ H", with typing conditions folded into the
        quantifier when every quantified variable has one.

        @param chain Flattened implication.
        @param caption Per-caption state.
        @param substitute Whether defined variables are substituted.
        @param nested True inside a formula, False for a top-level caption.
        @param outer Substitutions of the enclosing caption, which stay in force
        inside a nested chain.
        @return `(text, strength)`.
        """
        spelling = caption.spelling
        outer = outer or {}
        inclusion = _inclusion(chain)
        if inclusion is not None:
            first, second = (caption.term(_substitute_term(_Leaf(name), outer)) for name in inclusion)
            return spelling.nowrap(f"{first} ⊆ {second}"), _RELATION
        own = _definitions(chain, self.definitions, self.anchor_signatures, outer) \
            if substitute else {}
        values = dict(outer)
        values.update(own)
        defining = _defining_premises(chain, own)
        conditions = [premise for index, premise in enumerate(chain.premises)
                      if index not in defining]
        if self.structured_implications and spelling.as_html:
            conditions = [part for condition in conditions
                          for part in (_conjunction_parts(condition)
                                       if isinstance(condition, Conjunction) else [condition])]
            conditions = self._display_parts(conditions, spelling)

        if not nested:
            parts = [self._formula_in(condition, values, caption, _TYPE)
                     for condition in conditions]
            listed = ", ".join(parts)
            head = self._head(chain.head, values, own, caption)
            if not conditions:
                return head, _QUANTIFIED
            if self.structured_implications and spelling.as_html:
                return _implication_columns(parts, head), _QUANTIFIED
            return (f"{spelling.keyword('from', True)}{listed}"
                    f"{spelling.keyword('follows', False)}{head}"), _QUANTIFIED
        quantified = [variable for variable in chain.bound
                      if variable not in own and variable not in chain.anchor_slots]
        typings: dict[str, Atom] = {}
        for condition in conditions:
            if (isinstance(condition, Atom) and condition.head == "in"
                    and condition.arguments[0] in quantified
                    and condition.arguments[0] not in typings):
                typings[condition.arguments[0]] = condition
        if quantified and all(variable in typings for variable in quantified):
            folded = set(id(typing) for typing in typings.values())
            conditions = [condition for condition in conditions if id(condition) not in folded]
            binder = ", ".join(
                f"{caption.token(variable)} ∈ "
                f"{caption.term(_substitute_term(_Leaf(typings[variable].arguments[1]), values))}"
                for variable in quantified)
        else:
            binder = ", ".join(caption.token(variable) for variable in quantified)
        parts = [self._formula_in(condition, values, caption, _AND)
                 for condition in conditions]
        listed = " ∧ ".join(parts)
        if conditions:
            head = self._head(chain.head, values, own, caption, context=_OR)
            body, strength = f"{listed} ⇒ {head}", _QUANTIFIED
            if self.structured_implications and spelling.as_html:
                body = _implication_columns(parts, head)
        else:
            head_text, head_strength = self._head_with_strength(chain.head, values, own, caption)
            body, strength = head_text, head_strength
        if quantified:
            return f"∀ {binder}: {body}", _QUANTIFIED
        return body, strength

    def _head(self, head: object, values: dict[str, object], own: dict[str, object],
              caption: _Caption, context: int = _QUANTIFIED) -> str:
        """
        @brief Spell the head of a chain, parenthesized for its context.

        @param head Head expression.
        @param values All substitutions in force.
        @param own Substitutions kept for this chain.
        @param caption Per-caption state.
        @param context Minimal binding strength the position requires.
        @return The spelled head.
        """
        text, strength = self._head_with_strength(head, values, own, caption)
        return caption.spelling.parentheses(text) if strength < context else text

    def _head_with_strength(self, head: object, values: dict[str, object],
                            own: dict[str, object], caption: _Caption) -> tuple[str, int]:
        """
        @brief Spell the head of a chain and report its binding strength.

        @details A positive defining atom whose result is substituted reads
        "value of the result = head value"; a positive equation with exactly
        one substituted side puts that side first; every other head is an
        ordinary formula.

        @param head Head expression.
        @param values All substitutions in force.
        @param own Substitutions kept for this chain.
        @param caption Per-caption state.
        @return `(text, strength)`.
        """
        if isinstance(head, Atom):
            definition = _output_definition(head)
            if definition is not None and definition[0] in own:
                given = caption.term(values[definition[0]])
                derived = caption.term(_substitute_term(definition[1], values))
                return f"{given} = {derived}", _RELATION
            if head.head == "=" and len(head.arguments) == 2:
                left, right = head.arguments
                if right in own and left not in own:
                    return (f"{caption.term(values[right])} = "
                            f"{caption.term(_substitute_term(_Leaf(left), values))}"), _RELATION
        return self._formula(head, values, caption)

    # -- formulas -------------------------------------------------------------

    def _formula_in(self, node: object, values: dict[str, object], caption: _Caption,
                    context: int) -> str:
        """
        @brief Spell a formula, parenthesized for its context.

        @details Parenthesized when it binds looser than the context requires
        (relations 4, conjunction 3, disjunction 2, function types 1,
        quantified formulas 0).

        @param node Formula node (MPL or display node).
        @param values Substitutions in force.
        @param caption Per-caption state.
        @param context Minimal binding strength the position requires.
        @return The spelled formula.
        """
        text, strength = self._formula(node, values, caption)
        return caption.spelling.parentheses(text) if strength < context else text

    def _display_parts(self, parts, spelling: _Spelling) -> list[object]:
        """
        @brief Retain the first occurrence of an exact conjunct in structured HTML.

        @details Conjunction is idempotent, including beneath negation. Equality
        is checked on complete parsed formulas before notation or bound-name
        rendering, never on displayed text. Different formal expressions remain
        distinct even when they have the same spelling. Plain text and legacy
        rendering retain every original occurrence; empty premise lists stay empty.

        @param parts Ordered parsed conjuncts or implication premises.
        @param spelling Output spelling of the current caption.
        @return Stable display list without altering the original parsed formulas.
        """
        if self.structured_implications and spelling.as_html:
            return list(dict.fromkeys(parts))
        return list(parts)

    def _conjunction(self, parts: list[str], spelling: _Spelling) -> str:
        """
        @brief Keep every conjunct on its own indented line in structured HTML.

        @details Each member owns a complete formula, including its recursively
        formatted implications and conjunctions. Explicit conjunction signs
        preserve the logical connective. Plain text and the legacy book keep
        their inline spelling; a single member needs no conjunction container.

        @param parts Nonempty ordered list of complete rendered conjuncts.
        @param spelling HTML or plain-text spelling for this caption.
        @return One formula retaining every member and conjunction connective.
        @invariant Empty conjunctions and empty members assert.
        """
        assert parts and all(parts), "A conjunction requires nonempty members"
        if not self.structured_implications or not spelling.as_html or len(parts) == 1:
            return " ∧ ".join(parts)
        members = "".join(
            '<span class="rf-conjunct"><span class="rf-conjunction-sign">'
            + ("∧" if index else "")
            + '</span><span class="rf-conjunct-body">' + part + "</span></span>"
            for index, part in enumerate(parts)
        )
        return spelling.parentheses('<span class="rf-conjunction">' + members + "</span>")

    def _formula(self, node: object, values: dict[str, object],
                 caption: _Caption) -> tuple[str, int]:
        """
        @brief Spell a formula and report its binding strength.

        @details Raw MPL: a negated implication `!(>[b](P)(Q))` is the
        existence "∃ b: P ∧ ¬Q"; a negated conjunction of negations is a
        disjunction; double negation cancels; any other negated compound is
        "¬(…)". Atoms go through `_atom`.

        @param node Formula node.
        @param values Substitutions in force.
        @param caption Per-caption state.
        @return `(text, strength)`.
        """
        if isinstance(node, Atom):
            return self._atom(node, False, values, caption)
        if isinstance(node, Conjunction):
            return self._conjunction(
                [self._formula_in(part, values, caption, _AND)
                 for part in self._display_parts(_conjunction_parts(node), caption.spelling)],
                caption.spelling), _AND
        if isinstance(node, _Disjunction):
            # ∨ is associative: a disjunction inside a disjunction needs no
            # parentheses; every other compound part keeps them.
            texts = []
            for part in node.parts:
                text, strength = self._formula(part, values, caption)
                texts.append(text if strength in (_RELATION, _OR) else caption.spelling.parentheses(text))
            return " ∨ ".join(texts), _OR
        if isinstance(node, _Existence):
            return self._existence(node, False, values, caption)
        if isinstance(node, Implication):
            return self._chain(_flatten_chain(node), caption, substitute=True, nested=True,
                               outer=values)
        assert isinstance(node, Negation), f"unknown formula node {node!r}"
        inner = node.inner
        if isinstance(inner, Negation):
            return self._formula(inner.inner, values, caption)
        if isinstance(inner, Atom):
            return self._atom(inner, True, values, caption)
        if isinstance(inner, Implication):
            chain = _flatten_chain(inner)
            inclusion = _inclusion(chain)
            if inclusion is not None:
                first, second = (caption.term(_substitute_term(_Leaf(name), values))
                                 for name in inclusion)
                return caption.spelling.nowrap(f"{first} ⊈ {second}"), _RELATION
            head = chain.head.inner if isinstance(chain.head, Negation) else Negation(chain.head)
            return self._existence(_Existence(chain.bound, chain.premises + (head,)),
                                   False, values, caption)
        if isinstance(inner, Conjunction):
            parts = _conjunction_parts(inner)
            if all(isinstance(part, Negation) for part in parts):
                return self._formula(_Disjunction(tuple(part.inner for part in parts)),
                                     values, caption)
        if isinstance(inner, _Existence):
            return self._existence(inner, True, values, caption)
        return caption.spelling.negation(self._formula(inner, values, caption)[0]), _RELATION

    def _existence(self, node: _Existence, negated: bool, values: dict[str, object],
                   caption: _Caption) -> tuple[str, int]:
        """
        @brief Spell `∃ variables: part1 ∧ part2 ∧ …`, or its negation with `∄`.

        @details A single variable typed by the first part `in[y, S]` reads
        "∃ y ∈ S: rest"; otherwise "∃ y: part1 ∧ part2". Without bound
        variables only the conjunction remains.

        @param node Existence node.
        @param negated Whether the existence is negated.
        @param values Substitutions in force.
        @param caption Per-caption state.
        @return `(text, strength)`.
        """
        if not node.variables:
            body = self._conjunction(
                [self._formula_in(part, values, caption, _AND)
                 for part in self._display_parts(node.parts, caption.spelling)],
                caption.spelling)
            return (caption.spelling.negation(body), _RELATION) if negated else (body, _AND)
        quantifier = "∄" if negated else "∃"
        parts = self._display_parts(node.parts, caption.spelling)
        first = parts[0]
        if (len(node.variables) == 1 and len(parts) > 1 and isinstance(first, Atom)
                and first.head == "in" and first.arguments[0] == node.variables[0]):
            binder = (f"{caption.token(node.variables[0])} ∈ "
                      f"{caption.term(_substitute_term(_Leaf(first.arguments[1]), values))}")
            parts = parts[1:]
        else:
            binder = ", ".join(caption.token(variable) for variable in node.variables)
        context = _OR if len(parts) == 1 else _AND
        body = self._conjunction(
            [self._formula_in(part, values, caption, context) for part in parts],
            caption.spelling)
        return f"{quantifier} {binder}: {body}", _QUANTIFIED

    def _atom(self, atom: Atom, negated: bool, values: dict[str, object],
              caption: _Caption) -> tuple[str, int]:
        """
        @brief Spell one atom through its mathematical template.

        @details Relations (`=`, `in`, `in2`, `in3`, folds, orders) read as
        equations and relations; every named definition reads through its
        own template (`M = [n, m]`, `f: X → Y`, `g = f ∪ {n ↦ a}`,
        `a div d = q`, `m = min M`, …) and is never unfolded. Only a
        numbered compiled name expands from the GL binaries; that expansion
        stops at every named definition it contains. Arguments carry the
        substitutions.

        @param atom The atom.
        @param negated Whether the atom is negated.
        @param values Substitutions in force.
        @param caption Per-caption state.
        @return `(text, strength)`.
        @invariant A template predicate carries its argument count; every
        other predicate is a numbered compiled name with a GL binary
        definition.
        """
        if not _has_template(atom.head):
            assert _COMPILED_NAME.fullmatch(atom.head), (
                f"named definition without a reading template: {atom.head}")
            return self._compiled(atom, negated, values, caption)
        if atom.head in _NAMED_TEMPLATES:
            _check_arity(atom)
            return self._named(atom, negated, values, caption)
        text, strength = self._template(atom, negated, values, caption)
        return caption.spelling.nowrap(text), strength

    def _template(self, atom: Atom, negated: bool, values: dict[str, object],
                  caption: _Caption) -> tuple[str, int]:
        """
        @brief Spell an atom that has a reading template.

        @details The body of `_atom` for relations and template predicates.

        @param atom The atom (`_has_template(atom.head)`).
        @param negated Whether the atom is negated.
        @param values Substitutions in force.
        @param caption Per-caption state.
        @return `(text, strength)`.
        """
        spelling = caption.spelling
        head, arguments = atom.head, atom.arguments

        def term(token: str) -> str:
            """
            @brief Spell an argument with the substitutions in force.

            @param token Argument token.
            @return The spelled argument.
            """
            return caption.term(_substitute_term(_Leaf(token), values))

        _check_arity(atom)
        equal = "≠" if negated else "="
        if head == "=":
            return f"{term(arguments[0])} {equal} {term(arguments[1])}", _RELATION
        if head == "in":
            return f"{term(arguments[0])} {'∉' if negated else '∈'} {term(arguments[1])}", _RELATION
        definition = _output_definition(atom)
        if definition is not None:
            value = caption.term(_substitute_term(definition[1], values))
            return f"{value} {equal} {term(definition[0])}", _RELATION
        if head in ("preorder", "strictOrder"):
            return _order_relation(head, arguments, negated, term, spelling,
                                   lambda parts: self._conjunction(parts, spelling))
        if head == "interval":
            return (f"{term(arguments[4])} {equal} "
                    f"{spelling.unit(f'[{term(arguments[2])}, {term(arguments[3])}]')}", _RELATION)
        if _ENUMERATION_SET.fullmatch(head):
            members = ", ".join(term(token) for token in arguments[:-1])
            return f"{term(arguments[-1])} {equal} {spelling.unit(f'{{{members}}}')}", _RELATION
        if _ENUMERATION_RELATION_SET.fullmatch(head):
            assert len(arguments) % 2 == 1, f"enumeration relation set needs pairs: {atom!r}"
            pairs = ", ".join(spelling.unit(f"{term(arguments[index])} ↦ {term(arguments[index + 1])}")
                              for index in range(0, len(arguments) - 1, 2))
            return f"{term(arguments[-1])} {equal} {{{pairs}}}", _RELATION
        if head == "limitSequence":
            restricted = spelling.restriction(term(arguments[3]),
                                              f"{spelling.text('≤')} {term(arguments[2])}")
            return f"{term(arguments[4])} {equal} {restricted}", _RELATION
        if head == "split":
            restricted = spelling.restriction(
                term(arguments[4]), f"[{term(arguments[2])}, {term(arguments[3])})")
            return f"{term(arguments[5])} {equal} {restricted}", _RELATION
        if head == "limitSet":
            # M ∩ [0, n] names no bound element, unlike {x ∈ M : x ≤ n}.
            return (f"{term(arguments[4])} {equal} {term(arguments[2])} ∩ "
                    f"{spelling.unit(f'[0, {term(arguments[3])}]')}", _RELATION)
        if head == "appendToSequence":
            return (f"{term(arguments[3])} {equal} {term(arguments[2])} ∪ "
                    f"{{{spelling.unit(f'{term(arguments[0])} ↦ {term(arguments[1])}')}}}", _RELATION)
        if head == "addToSet":
            return (f"{term(arguments[2])} {equal} {term(arguments[0])} ∪ {{{term(arguments[1])}}}",
                    _RELATION)
        if head == "identity":
            return (f"{term(arguments[1])} is {'not ' if negated else ''}the identity on "
                    f"{term(arguments[0])}"), _RELATION
        if head in ("fXY", "fXYZ", "sequence"):
            # A type statement `f: X → Y` is parenthesized inside ∧ / ∨, where
            # its colon and arrow would otherwise read across the connective.
            text = _function_type(head, arguments, term, spelling)
            return (caption.spelling.negation(text), _RELATION) if negated else (text, _TYPE)
        if head == "prime":
            return f"{term(arguments[5])} is {'not ' if negated else ''}prime", _RELATION
        if head == "nontrivialFactor":
            return (f"{term(arguments[5])} is {'not ' if negated else ''}a nontrivial factor of "
                    f"{term(arguments[4])}", _RELATION)
        if head == "NaturalNumbers":
            text = "Peano axioms (" + ", ".join(term(token) for token in arguments) + ")"
            return (caption.spelling.negation(text), _RELATION) if negated else (text, _RELATION)
        if head.startswith("Anchor"):
            text = (f"{spelling.text(head[len('Anchor'):])} anchor ("
                    + ", ".join(term(token) for token in arguments) + ")")
            return (caption.spelling.negation(text), _RELATION) if negated else (text, _RELATION)
        assert False, f"no template branch for {head}"

    def _named(self, atom: Atom, negated: bool, values: dict[str, object],
               caption: _Caption) -> tuple[str, int]:
        """
        @brief Spell a named definition through its own reading template.

        @details Each name has a prepared reading in standard mathematical
        language over its interface arguments, never its unfolded body:
        `X ⊆ Y`, `X = ∅`, `m = min M`, "M has a minimum",
        `D = {d ∈ ℕ : 2 ≤ d ≤ k, d ∣ k}`, `gcd(a, b) = 1`,
        `a · x = b · y + 1`, `∃ m, n ∈ ℕ: a · m = b · n ± 1`, "p is a value of
        P", "all values of P are prime", "P is non-increasing",
        `n = ∏ P(i) with primes P(0) ≥ P(1) ≥ … ≥ P(k)` and
        `∃ primes p₀ ≥ p₁ ≥ … ≥ pₖ: n = ∏ pᵢ`, both products from `i = 0` to
        `k` with the stacked bounds of a fold's product. The incubator
        definitions read as their few defining facts, one per conjunct. A
        name a reading binds itself comes from `_TEMPLATE_LETTERS`, letters
        no proof page uses for a variable, so it never meets a name of the
        surrounding formula. Fixed definition parameters (`0`, `1`, `2`, `+`,
        `·`) are spelled from the atom's own arguments; a compound argument
        in a sentence is parenthesized. Relation-sized pieces never break
        inside.

        @param atom Named-definition atom (`atom.head in _NAMED_TEMPLATES`).
        @param negated Whether the atom is negated.
        @param values Substitutions in force.
        @param caption Per-caption state.
        @return `(text, strength)`.
        @invariant The atom carries its template's argument count.
        """
        spelling = caption.spelling
        head, arguments = atom.head, atom.arguments
        equal = "≠" if negated else "="
        nowrap = spelling.nowrap

        def value(token: str) -> object:
            """
            @brief The term of an argument with the substitutions in force.

            @param token Argument token.
            @return The substituted term.
            """
            return _substitute_term(_Leaf(token), values)

        def term(token: str) -> str:
            """
            @brief Spell an argument with the substitutions in force.

            @param token Argument token.
            @return The spelled argument.
            """
            return caption.term(value(token))

        def word(token: str) -> str:
            """
            @brief Spell an argument that stands in a sentence.

            @details An operation or a large operator is parenthesized, so a
            compound argument stays one subject ("some multiples of (v₁ + v₂)
            and v₁ …"); leaves and applications need nothing.

            @param token Argument token.
            @return The spelled argument.
            """
            argument = value(token)
            if isinstance(argument, _Big):
                return spelling.big_parentheses(caption.term(argument))
            if isinstance(argument, (_Operation, _Fold)):
                return f"({caption.term(argument)})"
            return caption.term(argument)

        def bound(letter: str) -> _Leaf:
            """
            @brief A name the reading binds itself.

            @param letter One of `_TEMPLATE_LETTERS`.
            @return The name as a term.
            """
            assert letter in _TEMPLATE_LETTERS, f"{letter!r} is not a template letter"
            return _Leaf(letter)

        def negatable(text: str, strength: int) -> tuple[str, int]:
            """
            @brief Return a compound reading, negated as a whole when required.

            @param text The positive reading.
            @param strength Its binding strength.
            @return `(text, strength)` or the negation of the reading.
            """
            return (spelling.negation(text), _RELATION) if negated else (text, strength)

        if head == "subset":
            return nowrap(f"{term(arguments[0])} {'⊈' if negated else '⊆'} {term(arguments[1])}"), _RELATION
        if head == "empty":
            return nowrap(f"{term(arguments[0])} {equal} ∅"), _RELATION
        if head == "minimum":
            return nowrap(f"{term(arguments[3])} {equal} min {term(arguments[2])}"), _RELATION
        if head == "hasMinimum":
            return nowrap(f"{word(arguments[2])} has {'no' if negated else 'a'} minimum"), _RELATION
        if head == "divisorSet":
            carrier, two, number, divisors = arguments[0], arguments[3], arguments[4], arguments[5]
            d = caption.term(bound("d"))
            members = spelling.unit("{" + f"{d} ∈ {term(carrier)} : {term(two)} ≤ {d} ≤ {term(number)}, "
                                    f"{d} ∣ {term(number)}" + "}")
            return nowrap(f"{term(divisors)} {equal} {members}"), _RELATION
        if head == "coprime":
            # The definition's only common divisor is s(0), the number 1.
            return nowrap(f"gcd({term(arguments[4])}, {term(arguments[5])}) {equal} 1"), _RELATION
        if head == "bezEq":
            _carrier, addition, multiplication, one, a, x, b, y = arguments
            left = caption.term(_Operation(multiplication, value(a), value(x)))
            right = caption.term(_Operation(addition, _Operation(multiplication, value(b), value(y)),
                                            value(one)))
            return nowrap(f"{left} {equal} {right}"), _RELATION
        if head == "bez1":
            carrier, _addition, multiplication, one, a, b = arguments
            m, n = bound("m"), bound("n")
            left = caption.term(_Operation(multiplication, value(a), m))
            right = caption.term(_Operation(multiplication, value(b), n))
            binder = nowrap(f"{'∄' if negated else '∃'} {caption.term(m)}, {caption.term(n)} ∈ {term(carrier)}:")
            return f"{binder} {nowrap(f'{left} = {right} ± {term(one)}')}", _QUANTIFIED
        if head == "inSequence":
            return nowrap(f"{word(arguments[1])} is {'not ' if negated else ''}a value of "
                          f"{word(arguments[2])}"), _RELATION
        if head == "primeseq":
            return nowrap(f"{'not ' if negated else ''}all values of {word(arguments[5])} are prime"), _RELATION
        if head == "ordered":
            # The hyphenated word never breaks, not even on phones.
            return nowrap(f"{word(arguments[2])} is {'not ' if negated else ''}"
                          f"{spelling.unit('non-increasing')}"), _RELATION
        if head == "primeFactorSeq":
            zero, one, length, sequence, number = (arguments[4], arguments[5], arguments[7],
                                                   arguments[8], arguments[9])
            first, second, last = (caption.term(_Apply(value(sequence), value(index)))
                                   for index in (zero, one, length))
            factors = spelling.big_operator("∏", term(zero), term(length),
                                            f"{caption.term(value(sequence))}(i)")
            product = nowrap(f"{term(number)} = {factors}")
            primes = f"with primes {nowrap(f'{first} ≥ {second} ≥ … ≥ {last}')}"
            if self.structured_implications and spelling.as_html:
                return negatable(_continued_lines(product, primes), _AND)
            return negatable(f"{product} {primes}", _AND)
        if head == "hasPrimeFactorization":
            zero, one, number = arguments[4], arguments[5], arguments[7]
            first, second, last = (spelling.subscript("p", index) for index in (term(zero), term(one), "k"))
            binder = nowrap(f"{'∄' if negated else '∃'} primes {first} ≥ {second} ≥ … ≥ {last}:")
            factors = spelling.big_operator("∏", term(zero), caption.term(bound("k")),
                                            spelling.subscript("p", "i"))
            return f"{binder} {nowrap(f'{term(number)} = {factors}')}", _QUANTIFIED
        if head == "infiniteSequence":
            text = _function_type("fXY", (arguments[0], arguments[1], arguments[1]), term, spelling)
            return negatable(nowrap(text), _TYPE)
        if head == "residual":
            total, part, rest, addition = arguments
            return nowrap(f"{caption.term(_Operation(addition, value(rest), value(part)))} {equal} "
                          f"{term(total)}"), _RELATION
        if head == "constSeq":
            carrier, _addition, sequence, start, end, interval, constant = arguments
            parts = [nowrap(f"{term(interval)} = {spelling.unit(f'[{term(start)}, {term(end)}]')}"),
                     nowrap(f"{term(sequence)}: {term(interval)} → {term(carrier)}"),
                     nowrap(f"{term(sequence)} ≡ {term(constant)}")]
            return negatable(self._conjunction(parts, spelling), _AND)
        if head == "nonInterval":
            _carrier, successor, _addition, superset, bound, interval, next_, after = arguments
            parts = [nowrap(f"{term(interval)} = {term(superset)} ∩ {spelling.unit(f'[0, {term(bound)}]')}"),
                     nowrap(f"{caption.term(_Apply(value(successor), value(bound)))} = {term(next_)} "
                            f"∉ {term(superset)}"),
                     nowrap(f"{caption.term(_Apply(value(successor), value(next_)))} = {term(after)} "
                            f"∈ {term(superset)}")]
            return negatable(self._conjunction(parts, spelling), _AND)
        if head == "nonSequence":
            _carrier, successor, _addition, relation, bound, sequence, first, second, next_ = arguments
            restricted = spelling.restriction(term(relation), f"{spelling.text('≤')} {term(bound)}")
            image = caption.term(_Apply(value(relation), value(next_)))
            parts = [nowrap(f"{term(sequence)} = {restricted}"),
                     nowrap(f"{caption.term(_Apply(value(successor), value(bound)))} = {term(next_)}"),
                     nowrap(f"{image} = {term(first)}") + ", " + nowrap(f"{image} = {term(second)}")]
            return negatable(self._conjunction(parts, spelling), _AND)
        assert False, f"no named template for {head}"

    def _compiled(self, atom: Atom, negated: bool, values: dict[str, object],
                  caption: _Caption) -> tuple[str, int]:
        """
        @brief Unfold a numbered compiled name through its definition in the
        GL binaries.

        @details `and` is a conjunction, `or` a disjunction, `existence`
        "∃ y: first ∧ body" and `implication` a nested chain
        "∀ …: … ⇒ …"; bound digits get fresh caption names. The unfolding
        continues through nested numbered names and stops at every named
        definition, which reads through its own template (`_named`).

        @param atom Numbered compiled-name atom.
        @param negated Whether the atom is negated.
        @param values Substitutions in force.
        @param caption Per-caption state.
        @return `(text, strength)`.
        @invariant Every compiled name in a proof graph has its binary entry.
        """
        entry = self.definitions.get(atom.head)
        assert entry is not None, (
            f"predicate without a reading template or a GL binary definition: {atom.head}")
        category, parts, anchor_names, names = _instantiate(
            entry, atom, lambda _digit, element: caption.fresh(element), self.anchor_signatures)
        if category == "existence":
            assert len(parts) == 2, f"existence {atom.head} needs two elements"
            return self._existence(_Existence(tuple(names), tuple(parts)), negated, values, caption)
        if category == "implication":
            node = parts[-1]
            for premise in reversed(parts[1 if anchor_names else 0:-1]):
                node = Implication((), premise, node)
            if anchor_names:
                node = Implication(tuple(anchor_names), parts[0],
                                   Implication(tuple(names), node.premise, node.conclusion)
                                   if isinstance(node, Implication) and names else node)
            elif isinstance(node, Implication):
                node = Implication(tuple(names), node.premise, node.conclusion)
        elif len(parts) == 1:
            node = parts[0]
        elif category == "and":
            node = Conjunction(tuple(parts))
        else:
            assert category == "or", f"unknown compiled category {category!r}"
            node = _Disjunction(tuple(parts))
        if not negated:
            return self._formula(node, values, caption)
        if isinstance(node, Implication) and _inclusion(_flatten_chain(node)) is not None:
            return self._formula(Negation(node), values, caption)
        if isinstance(node, (Conjunction, _Disjunction, Implication)):
            return caption.spelling.negation(self._formula(node, values, caption)[0]), _RELATION
        return self._formula(Negation(node), values, caption)


def _order_relation(head: str, arguments: tuple[str, ...], negated: bool,
                    term, spelling: _Spelling, conjunction=None) -> tuple[str, int]:
    """
    @brief Spell `preorder[N,op,a,b]` or `strictOrder[N,op,a,b]`.

    @details `preorder` with `+` is `a ≤ b`, with `*` divisibility `a ∣ b`;
    `strictOrder` with `+` is `a < b` and with `*` the proper-divisor
    relation `a ∣ b ∧ a ≠ b`. A negation uses the crossed symbol. Any other
    operation slot is a contract violation: every anchor names these two.

    @param head `preorder` or `strictOrder`.
    @param arguments The four arguments.
    @param negated Whether the atom is negated.
    @param term Token speller with substitutions applied.
    @param spelling Output form (escapes the strict-order sign).
    @param conjunction Optional formatter for the proper-divisor conjunction.
    @return `(text, strength)`.
    """
    _carrier, operation, left, right = arguments
    assert operation in ("+", "*"), f"order atom with unknown operation slot {operation!r}"
    if (head, operation) == ("strictOrder", "*"):
        parts = [f"{term(left)} ∣ {term(right)}", f"{term(left)} ≠ {term(right)}"]
        body = conjunction(parts) if conjunction is not None else " ∧ ".join(parts)
        return (spelling.negation(body), _RELATION) if negated else (body, _AND)
    symbols = {("preorder", "+"): ("≤", "≰"), ("preorder", "*"): ("∣", "∤"),
               ("strictOrder", "+"): ("<", "≮")}
    positive, crossed = symbols[(head, operation)]
    symbol = spelling.text(crossed if negated else positive)
    return f"{term(left)} {symbol} {term(right)}", _RELATION


def _function_type(head: str, arguments: tuple[str, ...], term, spelling: _Spelling) -> str:
    """
    @brief Spell a function-type predicate.

    @details `fXY[f,X,Y]` is `f: X → Y`, `fXYZ[f,X,Y,Z]` is `f: X × Y → Z`,
    and `sequence[N,+,n,m,f]` is `f: [n, m] → N`.

    @param head `fXY`, `fXYZ` or `sequence`.
    @param arguments The predicate's arguments.
    @param term Token speller with substitutions applied.
    @param spelling Output form; a sequence's index interval is one unit.
    @return The spelled type statement.
    """
    if head == "fXY":
        assert len(arguments) == 3, f"fXY needs three arguments: {arguments!r}"
        return f"{term(arguments[0])}: {term(arguments[1])} → {term(arguments[2])}"
    if head == "fXYZ":
        assert len(arguments) == 4, f"fXYZ needs four arguments: {arguments!r}"
        return (f"{term(arguments[0])}: {term(arguments[1])} × {term(arguments[2])} → "
                f"{term(arguments[3])}")
    assert head == "sequence" and len(arguments) == 5, f"sequence needs five arguments: {arguments!r}"
    return (f"{term(arguments[4])}: {spelling.unit(f'[{term(arguments[2])}, {term(arguments[3])}]')} → "
            f"{term(arguments[0])}")


# ---------------------------------------------------------------------------
# Anchor expansion (used by the anchor symbol map)
# ---------------------------------------------------------------------------

def expand_expr(expr: str):
    expanded_expr = expr[:]
    magic_string = "@19023847@"

    for core_expr in expression_utils.get_configuration_data():
        index = expr.find(r"(" + core_expr + r"[")
        if index != -1:
            index += 1
            replacing_args = expression_utils.get_args(expr[index:])
            args_to_be_replaced = expression_utils.get_args(
                expression_utils.get_configuration_data()[core_expr].short_mpl_raw)

            replacement_map = {}
            for ind in range(len(args_to_be_replaced)):
                replacement_map[args_to_be_replaced[ind]] = replacing_args[ind] + magic_string

            expanded_expr = (
                expression_utils.replace_keys_in_string(
                    expression_utils.get_configuration_data()[core_expr].full_mpl[:], replacement_map))
            expanded_expr = expanded_expr.replace(magic_string, "")

    return expanded_expr
