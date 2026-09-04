# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt)
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Independent loader for processed GL theorem lists and proof chapters."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from .mpl import Expression, alpha_key, parse_mpl


CHAPTER_PATTERN = re.compile(r"^(\d+)_(.+)\.txt$")
INTEGRATION_GOAL_SUFFIX = "_integration_goal"
ORDINARY_SCOPE = "ordinary"
INTEGRATION_GOAL_SCOPE = "integration_goal"
METHOD_SUFFIXES = {
    "direct": ("direct_proof",),
    "induction": ("induction_typing", "check_zero", "check_induction_condition"),
    "or theorem": ("or_theorem",),
    "or elimination": ("or_elimination",),
    "proved not broadcast": ("proved_not_broadcast",),
    "reformulated statement": ("reformulated_statement",),
}


@dataclass(frozen=True)
class ScopedExpression:
    """One MPL expression plus its proof-only scope and exact source spelling."""

    expression: Expression
    source_mpl: str
    proof_scope: str


@dataclass(frozen=True)
class Dependency:
    """One expression and namespace cited by a processed proof row."""

    scoped_expression: ScopedExpression
    namespace: str

    @property
    def expression(self) -> Expression:
        """
        @brief Return the logical expression without its proof-scope marker.
        @details The exact source spelling remains available through the peer
        scoped-expression record.
        @return Parsed logical expression.
        """

        return self.scoped_expression.expression


@dataclass(frozen=True)
class ProofRow:
    """One parsed row with its stable one-based source line number."""

    source_line: int
    conclusion: ScopedExpression
    namespace: str
    tag: str
    dependencies: tuple[Dependency, ...]


@dataclass(frozen=True)
class Chapter:
    """One processed chapter and all rows in original file order."""

    path: Path
    number: int
    role: str
    rows: tuple[ProofRow, ...]


@dataclass(frozen=True)
class TheoremRecord:
    """One theorem-list row and the chapters assigned by sequence."""

    source_index: int
    expression: Expression
    method: str
    reference: str
    chapters: tuple[Chapter, ...]


def sha256_file(path: Path) -> str:
    """
    @brief Hash one source artifact for the certificate manifest.
    @details The digest uses the file's exact bytes and uppercase hexadecimal.
    @param path Artifact path.
    @return Uppercase SHA-256 digest.
    """

    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def parse_scoped_expression(text: str) -> ScopedExpression:
    """
    @brief Separate proof-only integration scope from canonical MPL.
    @details The suffix remains preserved in the source spelling but is removed
    before strict MPL parsing.
    @param text Processed proof-graph expression spelling.
    @return Parsed expression with normalized proof scope and exact source text.
    """

    if text.endswith(INTEGRATION_GOAL_SUFFIX):
        core = text[:-len(INTEGRATION_GOAL_SUFFIX)]
        assert core and not core.endswith(INTEGRATION_GOAL_SUFFIX), (
            f"malformed integration-goal expression {text!r}"
        )
        return ScopedExpression(parse_mpl(core), text, INTEGRATION_GOAL_SCOPE)
    return ScopedExpression(parse_mpl(text), text, ORDINARY_SCOPE)


def _load_chapter(path: Path, number: int, role: str) -> Chapter:
    """
    @brief Parse one processed proof chapter in source-row order.
    @details Tab-separated conclusion, namespace, tag, and dependency pairs are
    validated without importing GL parsing code.
    @param path Processed chapter file.
    @param number Contiguous chapter number.
    @param role Filename-derived chapter role.
    @return Parsed non-empty chapter.
    """

    rows: list[ProofRow] = []
    for source_line, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line:
            continue
        fields = raw_line.split("\t")
        assert len(fields) >= 3 and (len(fields) - 3) % 2 == 0, (
            f"malformed processed proof row {path.name}:{source_line}"
        )
        dependencies = tuple(
            Dependency(parse_scoped_expression(fields[index]), fields[index + 1])
            for index in range(3, len(fields), 2)
        )
        rows.append(
            ProofRow(
                source_line,
                parse_scoped_expression(fields[0]),
                fields[1],
                fields[2],
                dependencies,
            )
        )
    assert rows, f"processed chapter is empty: {path}"
    return Chapter(path, number, role, tuple(rows))


def load_theorem_records(proof_graph: Path) -> tuple[TheoremRecord, ...]:
    """
    @brief Load the theorem registry and assign its numbered chapters.
    @details Method-specific chapter sequences and contiguous numbering are
    asserted independently of the GL runtime.
    @param proof_graph Processed proof-graph directory.
    @return Theorem records with lazy chapter path assignments.
    """

    theorem_list = proof_graph / "global_theorem_list.txt"
    assert theorem_list.is_file(), f"missing theorem list: {theorem_list}"
    raw_records: list[tuple[Expression, str, str]] = []
    for source_index, raw_line in enumerate(
        theorem_list.read_text(encoding="utf-8").splitlines()
    ):
        fields = raw_line.split("\t")
        assert len(fields) == 3, f"malformed theorem row {source_index + 1}"
        assert fields[1] in METHOD_SUFFIXES, f"unknown theorem method {fields[1]!r}"
        raw_records.append((parse_mpl(fields[0]), fields[1], fields[2]))

    numbered: list[tuple[int, str, Path]] = []
    for path in proof_graph.glob("*.txt"):
        match = CHAPTER_PATTERN.fullmatch(path.name)
        if match:
            numbered.append((int(match.group(1)), match.group(2), path))
    numbered.sort(key=lambda item: item[0])
    assert numbered, f"no numbered proof chapters in {proof_graph}"
    assert [number for number, _, _ in numbered] == list(range(len(numbered))), (
        "processed proof chapter numbering is not contiguous from zero"
    )

    records: list[TheoremRecord] = []
    chapter_cursor = 0
    for source_index, (expression, method, reference) in enumerate(raw_records):
        chapters: list[Chapter] = []
        for expected_role in METHOD_SUFFIXES[method]:
            assert chapter_cursor < len(numbered), "theorem registry outlives chapter files"
            number, actual_role, path = numbered[chapter_cursor]
            assert actual_role == expected_role, (
                f"theorem {source_index} ({method}) expected {expected_role!r}, "
                f"found {path.name!r}"
            )
            chapters.append(Chapter(path, number, actual_role, ()))
            chapter_cursor += 1
        records.append(
            TheoremRecord(source_index, expression, method, reference, tuple(chapters))
        )
    assert chapter_cursor == len(numbered), "chapter files outlive theorem registry"
    return tuple(records)


def find_theorem(
    records: tuple[TheoremRecord, ...],
    theorem_mpl: str,
    method: str,
    source_index: int | None = None,
) -> TheoremRecord:
    """
    @brief Resolve and fully parse one theorem alpha-equivalently.
    @details Method and optional source index constrain the unique match before
    its assigned chapters are loaded.
    @param records Loaded theorem registry records.
    @param theorem_mpl Selected theorem MPL.
    @param method Required proof method.
    @param source_index Optional exact theorem-list index.
    @return Unique theorem record with parsed chapters.
    """

    wanted = alpha_key(parse_mpl(theorem_mpl))
    matches = [
        record
        for record in records
        if (
            record.method == method
            and alpha_key(record.expression) == wanted
            and (source_index is None or record.source_index == source_index)
        )
    ]
    assert len(matches) == 1, (
        f"expected one alpha-equivalent {method!r} theorem, found {len(matches)}"
    )
    match = matches[0]
    parsed_chapters = tuple(
        _load_chapter(chapter.path, chapter.number, chapter.role)
        for chapter in match.chapters
    )
    return TheoremRecord(
        match.source_index,
        match.expression,
        match.method,
        match.reference,
        parsed_chapters,
    )
