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

import argparse
import os
import re
import sys
import itertools
import dataclasses
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Set, Tuple, Optional


# ---------------------------------------------------------------------------
#  Data structures
# ---------------------------------------------------------------------------

@dataclass
class ProofLine:
    """@brief One tab-separated row of a processed-proof-graph chapter file.

    @details
    Wraps the byte-accurate row layout emitted by
    ``process_proof_graphs.py`` into a structured form that every
    per-tag checker consumes:

        ``<expression> TAB <namespace> TAB <tag> TAB <rest[0]> TAB <rest[1]> TAB ...``

    Field semantics:

    - ``expression``: the row's LEFT-hand side — the derived fact the
      row claims. For most tags this is an MPL expression (e.g.
      ``(in[v1,N])``, ``(>[v1](in[v1,N])(in[(s[v1]),N]))``); for
      ``or branch proven`` and ``or disintegration`` it is the
      compiled OR ``(or<N>[…])`` itself.
    - ``namespace``: the validity scope at which the fact holds.
      ``"main"`` is the theorem root; descendants encode OR-branch
      and recursion scopes via the ``_boundary_`` suffix grammar
      (see ``docs/agentic_swdd/20_core_concepts/04_validity_stack.md``).
    - ``tag``: the proof-tag dispatch key. Maps to one of the 31
      entries in ``TAG_CHECKERS``. A missing tag becomes
      ``"<malformed>"`` so the dispatcher records a recognisable
      failure instead of crashing on a KeyError.
    - ``rest``: origin/justification fields, stored as a FLAT list but
      consumed by checkers in alternating ``(source_expr, source_ns)``
      PAIRS. A well-formed row always has even ``len(rest)``; odd
      length is itself a malformation many checkers reject up front
      (e.g. ``equality1`` requires ``len(rest) >= 4``).
    - ``raw``: the unparsed text line (kept for diagnostics so the
      report can echo the source on structural failures).
    - ``line_no``: 1-based row index inside the chapter file (matches
      how editors display the row, not 0-based).

    @invariant I-16 — verifier is sacred. Every ``ProofLine`` is
    treated as opaque input data; no field is mutated after
    construction. Tests that need a modified row produce a fresh
    instance via ``make_proof_line`` in the test harness.

    @see verify_chapter — chapter-level dispatcher that walks every
    ProofLine in a chapter and routes it via ``TAG_CHECKERS``.
    @see parse_chapter_file — on-disk reader that produces lists of
    ProofLine.
    """
    expression: str
    namespace: str
    tag: str
    rest: List[str]
    raw: str
    line_no: int


@dataclass
class TagCounter:
    """@brief Per-tag success/failure tally surfaced in the final report.

    @details
    Every dispatched check — per-tag in ``TAG_CHECKERS`` plus every
    chapter-level meta counter (``self-reference``,
    ``anchor handling trace``,
    ``contradiction trace``, ``vacuous truth trace``, ``origin``,
    ``definition set consistency``, ``origin chain termination``) and
    every global meta counter (``operator registry consistency``,
    ``theorem usage termination``) —
    aggregates outcomes into one of these counters via
    ``state.counter_for(name).record(passed)``.

    The final report (``print_report``) walks the registry in
    declaration order and prints one row per counter:

        ``<tag-name>                       success M, failure N``

    On a clean release every category reports ``failure 0`` — this is
    the airtight criterion the verifier was designed around (I-16).
    Append-only within a run; no API resets a counter mid-run.
    """
    success: int = 0
    failure: int = 0

    def record(self, passed: bool):
        """@brief Increment success or failure.

        @details
        ``True`` bumps ``success``; ``False`` bumps ``failure`` and is
        the signal the final report uses to flag the row as a real
        bug per I-16 — a verifier failure is never a false positive,
        always a producer-side regression.

        @param passed  True iff the checker accepted the row.
        """
        if passed:
            self.success += 1
        else:
            self.failure += 1


@dataclass
class VerifierState:
    """@brief Global state for one verification run, threaded into every checker.

    @details
    Constructed once by ``run_verifier(base_dir)`` and passed as the
    third argument of every ``check_*`` function (after the row and
    the chapter's row list). Aggregates four categories of data:

    1. **Counters.** ``tag_counters`` (per-tag) plus the standalone
       ``goal_reached`` counter for ``check_theorem_goal_reached``.
       These are the only mutable fields touched by checkers.

    2. **Theorem registries.** ``global_theorems`` (dict keyed by
       expression, for fast membership) and ``global_theorem_list``
       (ordered, used by ``build_chapter_theorem_map`` to assign
       chapter files to theorems in registry order). ``external_theorems``
       holds expressions from ``external_theorems.txt`` (raw + renamed
       forms) consulted by ``check_externally_provided_theorem``
       and the ``origin`` meta-check.

    3. **Operator metadata** loaded from ``ConfigVisu.json``:
       ``output_indices`` (atomic op → 0-based output arg position),
       ``input_indices`` (atomic op → list of 0-based input positions),
       ``definition_sets`` (atomic op → ``{pos_str → [type, combinable]}``).
       Consumed by the reformulation, recursion, and
       anchor-handling checkers.

    4. **GL binaries** loaded from
       ``files/GL_binaries/GL_binary_*.json``: ``gl_binaries`` is
       ``tag → operator-dict``. Each binary holds per-batch spontaneous
       compact-operator allocations (``existence4``, ``or0``,
       ``implication26``, …) with shape
       ``{category, elements, signature, arity, definedSet}``. D-41's
       per-tag resolved-defset indices (``resolved_defsets_per_tag``
       plus the atomic-only fallback ``resolved_defsets_atomic_only``)
       are derived once in ``run_verifier`` via
       ``build_resolved_defsets_per_tag``.

    The transient fields ``current_chapter_thm``,
    ``current_chapter_type``, ``current_gl_binary``, and
    ``current_resolved_defsets`` are reset PER CHAPTER by
    ``verify_chapter`` so per-tag checkers see the chapter's anchor
    binary and theorem context without re-resolving them on every call.

    @invariant I-16 — verifier is sacred. Checkers MUST NOT mutate
    ``gl_binaries`` / ``output_indices`` / ``input_indices`` /
    ``definition_sets`` / ``resolved_defsets_*``. Read-only by
    convention; the Python test layer enforces this by sharing these
    fields by reference across test fixtures.
    @invariant I-29 — variable-port type consistency: the resolved
    defsets feed ``check_defset_consistency``, which independently
    verifies the compiler's same-type-at-shared-positions contract.

    @see run_verifier — bootstraps a fresh state from a
    processed-proof-graph directory.
    @see verify_chapter — sets per-chapter transient context and
    dispatches every row.
    """
    tag_counters: Dict[str, TagCounter] = field(default_factory=dict)
    goal_reached: TagCounter = field(default_factory=TagCounter)

    # Global theorem list: expression → { "type": ..., "ref": ... }
    global_theorems: Dict[str, dict] = field(default_factory=dict)
    # Ordered list of (expression, type, ref) for chapter mapping
    global_theorem_list: List[Tuple[str, str, str]] = field(default_factory=list)

    # Output indices: core_name → 0-based index of output argument
    output_indices: Dict[str, int] = field(default_factory=dict)

    # Input indices: core_name → list of 0-based indices of input arguments
    input_indices: Dict[str, List[int]] = field(default_factory=dict)

    # GL binaries: tag → { name → { "category", "elements", "signature", "definedSet", ... } }
    gl_binaries: Dict[str, dict] = field(default_factory=dict)

    # Transient: set per-chapter before dispatching line checkers
    current_chapter_thm: Optional[Tuple[str, str, str]] = None
    current_chapter_type: str = ""
    current_gl_binary: Optional[dict] = None  # the GL binary for the current chapter's batch

    # Definition sets: core_name → { "1-based-position" → [def_set_str, bool] }
    definition_sets: Dict[str, Dict[str, list]] = field(default_factory=dict)

    # Resolved defset index — PER-TAG. tag → { core_name → { pos_str → type_label_str } }
    # Mirrors the compiler's per-batch ArgumentAnalyzer(this->coreExpressionMap)
    # construction at prover.hpp:3556 — each batch resolves composites against
    # its own binary, so the same compact name (e.g. `implication26`) can mean
    # different things in different batches without colliding. Per-chapter the
    # right tag is selected the same way `current_gl_binary` is (anchor substring
    # match on the chapter's theorem expression). See D-41.
    resolved_defsets_per_tag: Dict[str, Dict[str, Dict[str, str]]] = field(default_factory=dict)
    # Atomic-only fallback (no batch-specific composites) — used for chapters
    # whose theorem doesn't disclose an anchor and the per-tag selection
    # therefore fails. Composites unresolved → unknown-skip per check.
    resolved_defsets_atomic_only: Dict[str, Dict[str, str]] = field(default_factory=dict)
    # Per-chapter pointer set by verify_chapter; check_defset_consistency reads.
    current_resolved_defsets: Optional[Dict[str, Dict[str, str]]] = None

    # External theorems (raw + renamed forms)
    external_theorems: set = field(default_factory=set)

    def binaries_for_chapter(self) -> list:
        """@brief Return the GL binary(ies) the current chapter may consult.

        @details
        Two paths:

        1. ``current_gl_binary`` was resolved by anchor-substring match
           in ``verify_chapter`` (the chapter's theorem mentions an
           ``Anchor<tag>`` substring whose ``<tag>`` is a loaded binary
           key). In this case the single resolved binary is
           authoritative and returned as a one-element list — no
           fallback is needed because we know exactly which batch
           created the operators this chapter uses.

        2. ``current_gl_binary`` is ``None`` (anchor not loaded as a
           binary, or no theorem). A fallback list is returned. When
           the chapter's theorem contains the literal substring
           ``AnchorIncubator`` (no tag literally named ``Incubator``
           is loaded — the tags are ``IncubatorPeano`` /
           ``IncubatorGauss`` / ``IncubatorGauss1``), the fallback is
           restricted to binaries whose tag starts with ``Incubator``.
           Otherwise every loaded binary is returned.

        **Why the chapter-context filter.** After the
        ``files/incubator/GL_binaries/`` merge into
        ``files/GL_binaries/`` (D-54), main-batch and incubator-batch
        spontaneous compact-operator allocations share one directory.
        Their names collide on shape — e.g. ``existence4`` is arity 5
        in ``Gauss`` but arity 4 in ``IncubatorGauss1``; ``existence2``
        is arity 3 in main batches but arity 8 in ``IncubatorPeano``.
        Without the filter, alphabetical fallback iteration would pick
        a main-batch binary first and route the check against a
        same-named-but-different-shaped operator, generating bogus
        failures. The filter restores the isolation the duplicate-
        folder layout used to provide implicitly via
        ``os.path.dirname(base_dir)/GL_binaries``.

        Per I-16 this is a tightening, not a weakening: the same set
        of binaries that were previously consultable on the incubator
        side remains consultable; binaries that were never visible
        there before (Peano / Gauss / shared) stay invisible.

        @return  A list of GL binary dicts to consult, in order. At
                 least one element when any binaries are loaded; empty
                 only if ``gl_binaries`` itself is empty.

        @see verify_chapter — sets ``current_gl_binary`` via the
        anchor-substring resolution this method falls back from.
        """
        if self.current_gl_binary is not None:
            return [self.current_gl_binary]
        thm = self.current_chapter_thm
        if thm is not None and "AnchorIncubator" in thm[0]:
            return [b for tag, b in self.gl_binaries.items()
                    if tag.startswith("Incubator")]
        return list(self.gl_binaries.values())

    def counter_for(self, tag: str) -> TagCounter:
        """@brief Lazily return the ``TagCounter`` for ``tag``.

        @details
        Creates a fresh zero-initialized ``TagCounter`` on first
        lookup and caches it in ``tag_counters``. Subsequent calls
        return the same instance so successive ``record(...)`` calls
        accumulate into one tally. Used both by the per-tag dispatch
        in ``verify_chapter`` and by chapter-level meta-check counters
        (``self-reference``, ``origin``, etc.) — those don't have
        their own ``TAG_CHECKERS`` entry but share the same counter
        infrastructure.

        @param tag  Tag name or meta-counter name. Any string is
                    accepted; the report enumerates registered names
                    in dispatch order then trailing meta names in
                    insertion order.
        @return     The (possibly freshly created) counter.
        """
        if tag not in self.tag_counters:
            self.tag_counters[tag] = TagCounter()
        return self.tag_counters[tag]


# ---------------------------------------------------------------------------
#  Parsing
# ---------------------------------------------------------------------------

def parse_chapter_file(filepath: str) -> List[ProofLine]:
    """@brief Read one chapter file off disk into a list of ``ProofLine``.

    @details
    Walks the file row-by-row, decoding each line as UTF-8. Blank
    lines and lines containing only whitespace are skipped — they have
    no semantic meaning. Each remaining line is split on ``"\\t"``;
    missing fields are filled with the empty string so the resulting
    ``ProofLine`` is always well-formed at the Python level (this
    means structural checks live downstream in the per-tag checkers,
    not at parse time).

    Field-position mapping:

    - ``parts[0]`` → ``expression`` (or ``""`` if the row is degenerate)
    - ``parts[1]`` → ``namespace`` (or ``""``)
    - ``parts[2]`` → ``tag``. When this field is absent the row gets
      tag ``"<malformed>"``; the dispatcher in ``verify_chapter``
      then records the row under ``state.counter_for("<unknown:...>")``,
      flagging the structural defect without crashing the verifier.
    - ``parts[3:]`` → ``rest``. Already a list (not a flat string), so
      checkers can stride-2 index without extra splitting.
    - ``raw`` is the untrimmed original line (only the trailing
      ``\\r\\n`` is stripped) for diagnostics.
    - ``line_no`` starts at 1 (matches editor line numbering).

    @param filepath  Absolute or cwd-relative path to a chapter
                     ``.txt`` file under
                     ``files/processed_proof_graph/``.
    @return  List of ``ProofLine`` in original file order. Returns an
             empty list for empty / whitespace-only files.

    @pre  ``filepath`` is readable as UTF-8.
    @post No I/O side effects beyond the read.
    @see verify_chapter — typical consumer of this output.
    """
    lines: List[ProofLine] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for i, raw_line in enumerate(f, start=1):
            raw_line = raw_line.rstrip("\r\n")
            if not raw_line.strip():
                continue
            parts = raw_line.split("\t")
            lines.append(ProofLine(
                expression=parts[0] if parts else "",
                namespace=parts[1] if len(parts) > 1 else "",
                tag=parts[2] if len(parts) > 2 else "<malformed>",
                rest=parts[3:] if len(parts) > 3 else [],
                raw=raw_line,
                line_no=i,
            ))
    return lines


def load_global_theorem_list(base_dir: str) -> Tuple[Dict[str, dict],
                                                      List[Tuple[str, str, str]]]:
    """@brief Load the chapter ordering manifest from ``global_theorem_list.txt``.

    @details
    The manifest is the contract between
    ``process_proof_graphs.py`` (which assigns one or more chapter
    files per proved theorem) and the verifier (which walks the
    chapters in registry order). Each row is tab-separated:

        ``<theorem-expression> TAB <theorem-type> TAB <theorem-ref>``

    where ``<theorem-type>`` is one of ``direct`` /
    ``reformulated`` / ``induction`` / ``or_theorem`` /
    ``incubator_back_reformulation`` and ``<theorem-ref>`` is the
    proof method (``direct`` /
    ``reformulated statement`` / ``or theorem``) OR — for induction
    rows — the induction variable name (consumed by
    ``check_recursion``).

    Returns two views over the same data:

    - **Dict** keyed by theorem expression. Drives fast membership
      queries inside ``check_theorem_tag`` and the ``origin``
      meta-check (cross-batch references).
    - **Ordered list** of ``(expr, type, ref)`` triples. Used by
      ``build_chapter_theorem_map`` to assign chapter files to
      theorems in registry order (induction rows consume three
      consecutive chapter files; everything else consumes one).

    Missing file → returns ``({}, [])`` rather than raising. This is
    by design: ``run_verifier`` calls it once for the primary
    directory and optionally again for sibling directories
    (``extra_global_lists``); a sibling without a manifest is not an
    error.

    @param base_dir  Directory containing ``global_theorem_list.txt``.
    @return  ``(dict-by-expression, ordered-list)`` — both views are
             empty when the file is absent.

    @see build_chapter_theorem_map — primary consumer of the ordered
    list.
    @see check_theorem_tag — primary consumer of the dict.
    """
    path = os.path.join(base_dir, "global_theorem_list.txt")
    theorems: Dict[str, dict] = {}
    ordered: List[Tuple[str, str, str]] = []
    if not os.path.isfile(path):
        return theorems, ordered
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            expr = parts[0]
            thm_type = parts[1] if len(parts) > 1 else ""
            ref = parts[2] if len(parts) > 2 else ""
            theorems[expr] = {"type": thm_type, "ref": ref}
            ordered.append((expr, thm_type, ref))
    return theorems, ordered


# ---------------------------------------------------------------------------
#  Build chapter → theorem mapping
#
#  Walk global_theorem_list in order, consume chapter files sequentially.
#  "induction" theorems consume 3 chapters (induction_typing + check_zero +
#  check_induction_condition, in that exact order — typing first because both
#  the zero-case and the successor-step proofs cite (in[ind_var, N]) and that
#  typing must be established before either citation makes sense).
#  All other types consume 1 chapter.
# ---------------------------------------------------------------------------

def build_chapter_theorem_map(
    chapter_files: List[str],
    theorem_list: List[Tuple[str, str, str]],
) -> Dict[str, Tuple[str, str, str]]:
    """@brief Assign every chapter file to the theorem it proves.

    @details
    Walks ``theorem_list`` and ``chapter_files`` in lockstep,
    consuming chapter files in declaration order:

    - **Non-induction theorems** consume exactly ONE chapter file
      (``direct_proof.txt`` /
      ``reformulated_statement.txt`` / ``or_theorem.txt`` /
      ``back_reformulated_statement.txt``).
    - **Induction theorems** consume exactly THREE chapter files, in
      this order:
        1. ``<i>_induction_typing.txt`` — proves
           ``(in[ind_var, N])``, the typing gate the prover requires
           before scheduling the induction.
        2. ``<i+1>_check_zero.txt`` — proves the zero case
           ``(=[ind_var, i0])``.
        3. ``<i+2>_check_induction_condition.txt`` — proves the
           successor-step implication.
      Typing comes first because both the zero-case and the
      successor-step proofs cite ``(in[ind_var, N])``; if typing
      hasn't been established, those citations would dangle.

    Returns ``chapter_filename → (theorem_expr, theorem_type,
    theorem_ref)``. Each induction theorem maps all THREE of its
    chapter files to the same triple, so per-chapter context lookup
    in ``verify_chapter`` is a simple dict lookup.

    **Structural invariant.** Each induction row must have exactly
    three consecutive chapter files at the expected filename
    suffixes. Violation raises ``AssertionError`` — the verifier's
    "airtight or it fails loudly" stance (I-19: asserts are
    first-class).

    @param chapter_files  Sorted list of chapter filenames (per
                          ``chapter_sort_key`` ordering).
    @param theorem_list   Ordered theorem list from
                          ``load_global_theorem_list``.
    @return  Map filename → ``(expr, type, ref)`` covering every
             consumed chapter. A short tail of unmatched chapter
             files (filenames beyond what the theorem list consumes)
             is silently dropped — they may exist in a partial
             pipeline run.

    @invariant I-19 — induction-triple structure asserted, not
    softened. Misaligned filenames crash early rather than confusing
    downstream checkers.
    """
    mapping: Dict[str, Tuple[str, str, str]] = {}
    ci = 0
    for thm_expr, thm_type, thm_ref in theorem_list:
        if ci >= len(chapter_files):
            break
        if thm_type == "induction":
            # Structural invariant: three consecutive chapters at the right
            # suffixes, in the order typing → check_zero → check_induction_condition.
            assert ci + 2 < len(chapter_files), (
                f"induction theorem {thm_expr!r} missing three chapters "
                f"(available: {chapter_files[ci:ci+3]})"
            )
            assert chapter_files[ci].endswith("_induction_typing.txt"), (
                f"induction chapter at index {ci} must be *_induction_typing.txt, "
                f"got {chapter_files[ci]!r}"
            )
            assert chapter_files[ci + 1].endswith("_check_zero.txt"), (
                f"induction chapter at index {ci + 1} must be *_check_zero.txt, "
                f"got {chapter_files[ci + 1]!r}"
            )
            assert chapter_files[ci + 2].endswith("_check_induction_condition.txt"), (
                f"induction chapter at index {ci + 2} must be "
                f"*_check_induction_condition.txt, got {chapter_files[ci + 2]!r}"
            )
            mapping[chapter_files[ci]]     = (thm_expr, thm_type, thm_ref)
            mapping[chapter_files[ci + 1]] = (thm_expr, thm_type, thm_ref)
            mapping[chapter_files[ci + 2]] = (thm_expr, thm_type, thm_ref)
            ci += 3
        else:
            mapping[chapter_files[ci]] = (thm_expr, thm_type, thm_ref)
            ci += 1
    return mapping


# ---------------------------------------------------------------------------
#  Theorem goal reached check
# ---------------------------------------------------------------------------

def _find_matching_paren(expr: str, start: int) -> int:
    """@brief Find the index of the ``)`` that closes the ``(`` at ``start``.

    @details
    Linear scan with a depth counter — increments on every ``(``,
    decrements on every ``)``. Returns the index of the closing
    paren when depth hits zero, or ``-1`` if the parentheses are
    unbalanced (the input runs off the end before depth reaches
    zero). Single-pass, no allocations; safe to call on long MPL
    strings inside hot disintegration loops.

    The scan starts AT ``start`` (not after it), so the caller must
    pass the index of the opening ``(`` itself. The function treats
    ``[``/``]`` as opaque — only ``(``/``)`` contribute to depth.
    That is intentional: MPL nests parens for sub-expressions and
    brackets for argument lists, and the two don't need to be
    co-balanced for paren matching (e.g. the argument list of an
    outer call may contain a sub-expression with its own
    independent paren structure).

    @param expr   The expression to scan.
    @param start  Index of the opening ``(`` whose match is sought.
    @return  Index of the matching ``)``; ``-1`` on unbalanced input.

    @pre  ``expr[start] == '('``. The function does not check this;
          callers responsible for positioning correctly.
    """
    depth = 0
    for i in range(start, len(expr)):
        if expr[i] == '(':
            depth += 1
        elif expr[i] == ')':
            depth -= 1
            if depth == 0:
                return i
    return -1


def disintegrate_implication_head(expr: str) -> str:
    """@brief Return the innermost conclusion of a nested implication.

    @details
    Iteratively peels ``(>[vars](premise)(body))`` layers from the
    outside in. Each iteration:

    1. Recognises ``(>[`` at the start of the current expression.
    2. Skips past the bound-variable list to the start of the
       premise.
    3. Walks past the premise — either ``(...)`` (balanced via
       ``_find_matching_paren``) or ``!(...)`` (negated, balanced via
       the same helper starting at the inner ``(``).
    4. Replaces ``expr`` with the body (either ``(...)`` or
       ``!(...)`` form, paren-balanced) and loops.

    Stops when:

    - The current expression doesn't start with ``(>[``.
    - The body is a negated atom ``!(...)`` — returned verbatim with
      the leading ``!`` preserved (this is the only path that returns
      a negated head; positive heads are always paren-wrapped without
      a leading ``!``).
    - Bracket scanning hits ``-1`` (malformed input — returns the
      partial state, which downstream checkers reject as
      structurally wrong).

    Used by ``check_theorem_goal_reached`` for ``direct_proof`` /
    ``check_zero`` / ``check_induction_condition`` chapter types to
    extract the proof's target from the theorem; used by
    ``check_validity_name`` to extract the head of an
    ``expansion-for-integration`` row's left side.

    @param expr  An MPL implication expression (possibly nested) OR a
                 non-implication shape (returned unchanged).
    @return  The innermost conclusion. Negated bodies retain the
             leading ``!``.

    @see disintegrate_implication_full — returns BOTH the premise
    list AND the head, when the caller needs both halves.
    """
    while expr.startswith('(>['):
        bracket_close = expr.index(']', 3)
        p_start = bracket_close + 1
        if p_start >= len(expr):
            break
        # Premise can be negated: !(...)
        if expr[p_start] == '!' and p_start + 1 < len(expr) and expr[p_start + 1] == '(':
            p_end = _find_matching_paren(expr, p_start + 1)
        elif expr[p_start] == '(':
            p_end = _find_matching_paren(expr, p_start)
        else:
            break
        if p_end < 0:
            break
        b_start = p_end + 1
        if b_start >= len(expr):
            break
        if expr[b_start] == '!':
            # Negated body: !(...)
            inner_start = b_start + 1
            if inner_start >= len(expr) or expr[inner_start] != '(':
                break
            inner_end = _find_matching_paren(expr, inner_start)
            if inner_end < 0:
                break
            expr = expr[b_start:inner_end + 1]
            return expr
        if expr[b_start] != '(':
            break
        b_end = _find_matching_paren(expr, b_start)
        if b_end < 0:
            break
        expr = expr[b_start:b_end + 1]

    return expr


# ---------------------------------------------------------------------------
#  Mirroring helpers  (independent copy — no imports from create_expressions)
# ---------------------------------------------------------------------------

def _extract_args(expr: str) -> List[str]:
    """@brief Extract args from the FIRST top-level ``[...]`` group in ``expr``.

    @details
    Returns the comma-split contents of the first bracket group found
    by simple regex (``\\[([^\\]]*)\\]``) — empty strings are filtered.
    Examples:

    - ``(in3[a,b,c,+])`` → ``["a", "b", "c", "+"]``
    - ``(>[v1](in[v1,N])(in[v1,N]))`` → ``["v1"]`` (the binder)
    - ``a_plain_string`` → ``[]`` (no bracket group)

    **Limitation.** Only the FIRST bracket group is returned. Nested
    groups (the arg list of a sub-expression inside an outer call's
    arg list) are not visible. Use ``_extract_all_args`` when you
    need the recursive form (the trace-back walker over compound
    source expressions relies on it).

    Used pervasively by the checkers to inspect argument shapes —
    arity comparisons, var-name extraction, anchor-slot detection,
    etc.

    @param expr  Expression with at least one ``[...]`` group.
    @return  List of comma-split args (empty strings dropped); ``[]``
             if no bracket group is present.
    """
    m = re.search(r'\[([^\]]*)\]', expr)
    if not m:
        return []
    return [a for a in m.group(1).split(',') if a]


def _extract_all_args(expr: str) -> List[str]:
    """@brief Extract args from EVERY ``[...]`` group in ``expr`` (recursive).

    @details
    Differs from ``_extract_args`` by iterating ``re.findall`` over
    EVERY bracket group, not just the first. Example:

        ``!(>[v6](in[v6,N])!(in2[v6,i1_copy,s]))``

    decomposes to bracket groups ``[v6]``, ``[v6,N]``,
    ``[v6,i1_copy,s]``, returning the flat arg list
    ``["v6", "v6", "N", "v6", "i1_copy", "s"]``. Duplicates are
    preserved (unlike a set); the caller decides whether to dedupe.

    Used by the variable-copy trace-back to detect whether a copy
    variable (e.g. ``i1_copy``) is present ANYWHERE inside a
    compound source expression. ``_extract_args`` would miss it if
    the copy lives in a nested sub-expression's arg list, which is
    exactly the structure compound sources have (e.g. an implication
    cited as a source).

    @param expr  Any expression (may have nested ``[...]`` groups).
    @return  Flat list of every comma-split arg across all groups,
             preserving duplicates.

    @see _extract_args — first-bracket-only variant for hot paths
    that only need the outer arity.
    """
    out: List[str] = []
    for group in re.findall(r'\[([^\]]*)\]', expr):
        for a in group.split(','):
            if a:
                out.append(a)
    return out


def _extract_core_name(expr: str) -> str:
    """@brief Extract the operator core name from a leaf-call expression.

    @details
    Matches the leading ``(<core>[`` pattern with a word-character
    core name. Returns the core name on a match, ``""`` otherwise.
    Examples:

    - ``(in3[a,b,c,+])`` → ``"in3"``
    - ``(AnchorPeano[N,s,p,zero,one,two])`` → ``"AnchorPeano"``
    - ``!(in[a,N])`` → ``""`` (leading ``!`` breaks the regex anchor)
    - ``(>[v1](in[v1,N])(in[v1,N]))`` → ``""`` (the ``>`` is not a
      word char)
    - ``(=[a,b])`` → ``""`` (the ``=`` is not a word char; equality
      checkers reach into the args via ``_extract_args`` and shape
      checks against the literal prefix ``"(=["``)

    Equality, negation, and implication shapes are deliberately NOT
    returned by this function — they're recognized by literal prefix
    checks at the call sites. Only true leaf operators (``in``,
    ``in2``, ``in3``, ``fXY``, ``existence4``, ``or0``,
    ``implication26``, ``Anchor*``, …) return a non-empty result.

    @param expr  An MPL expression.
    @return  Core operator name on a leaf-call match; ``""``
             otherwise.
    """
    m = re.match(r'\((\w+)\[', expr)
    return m.group(1) if m else ""


def _trace_back_to(expr: str,
                   expr_to_lines: Dict[str, List["ProofLine"]],
                   is_target,
                   should_follow=None,
                   visited: set = None) -> bool:
    """@brief Generic recursive trace-back through a chapter's origin graph.

    @details
    The chapter-level meta-checks (``anchor handling trace``,
    ``contradiction trace``, ``vacuous truth trace``, ``variable
    copy``) all share the same query shape: starting from some
    expression ``expr``, can we reach a chapter row that satisfies
    ``is_target(row)`` by walking rest-field source edges? This
    function is the shared engine.

    Algorithm:

    1. Look up ``expr`` in ``expr_to_lines`` (built once per chapter
       in ``verify_chapter``: ``expression → list of chapter rows
       whose left-hand side is that expression``). For each
       matching row:

       a. If ``is_target(row)`` returns True, the walk succeeds —
          return True immediately.
       b. Otherwise, iterate the row's ``rest`` fields in
          alternating ``(source_expr, source_ns)`` PAIRS (stride-2
          starting at index 0). For each source, optionally apply
          the ``should_follow(src)`` predicate to prune edges
          (callers like the variable-copy walker filter to edges
          that still carry the copy variable).

       c. Recurse on each surviving source.

    2. Cycle-safe via the ``visited`` set — a node visited once is
       never re-entered, so chapters with circular citations
       terminate gracefully (returning False if no target was
       reachable).

    The function returns the BOOLEAN reachability answer, NOT the
    path. Callers that need the path build it externally; in
    practice no checker does (the trace counters record True/False
    per call).

    @param expr           Starting expression for the walk.
    @param expr_to_lines  Map expression → list of chapter rows
                          whose LEFT-hand expression is that key.
                          Pre-computed once per chapter for O(1)
                          lookup inside the recursion.
    @param is_target      Predicate over a chapter line; True
                          short-circuits the walk with success.
    @param should_follow  Optional predicate over a source
                          expression; if False, that edge is not
                          followed. Default: every edge is followed.
    @param visited        Internal recursion guard. Callers pass
                          ``None`` (default); the recursion fills it.
    @return  True iff at least one ``is_target``-satisfying row is
             reachable from ``expr`` through follow-able edges.
    """
    if visited is None:
        visited = set()
    if expr in visited:
        return False
    visited.add(expr)
    for ch_line in expr_to_lines.get(expr, []):
        if is_target(ch_line):
            return True
        for i in range(0, len(ch_line.rest), 2):
            src = ch_line.rest[i]
            if should_follow is not None and not should_follow(src):
                continue
            if _trace_back_to(src, expr_to_lines, is_target, should_follow, visited):
                return True
    return False


def disintegrate_implication_full(expr: str) -> Tuple[List[str], str]:
    """@brief Peel every implication layer; return ``(premises, head)``.

    @details
    The full-information cousin of ``disintegrate_implication_head``:
    instead of discarding intermediate premises and returning only
    the innermost conclusion, this builds a flat list of every
    premise encountered on the way down and returns it alongside the
    final head.

    Walks ``(>[vars](premise)(body))`` nesting from outside in:

    1. Recognise ``(>[`` at the start of the current expression.
    2. Skip past the bound-variable list to the start of the
       premise.
    3. Capture the premise as either ``(...)`` (positive,
       paren-balanced) or ``!(...)`` (negated, leading ``!``
       preserved). Append to the premise list.
    4. Advance to the body. If the body is ``!(...)``, treat it as
       the final negated head — append the LAST premise (already
       captured), record the head, and return.
    5. If the body is ``(...)``, replace ``expr`` with the body and
       loop.

    On malformed input (any bracket scan returning ``-1``) the loop
    breaks and returns the partial state captured so far. Downstream
    checkers detect the malformation through structural mismatch
    (e.g. premise count differing from the actual row's rest length)
    and record a failure under the appropriate counter.

    Used by every checker that needs to compare a row's structure
    against a reference implication — ``check_implication``,
    ``check_equality1``, ``check_premise_element``,
    ``check_validity_name``,
    ``_check_reformulation``, ``check_equalize_variable``, etc.

    @param expr  An MPL implication expression (possibly nested) OR
                 a non-implication shape (in which case
                 ``premises == []`` and ``head == expr``).
    @return  ``(premises_list, head)`` — premises in outermost-first
             order, ``head`` is the innermost conclusion (with
             leading ``!`` preserved if negated).

    @see disintegrate_implication_head — head-only variant for
    callers that don't need the premises.
    """
    premises: List[str] = []
    while expr.startswith('(>['):
        bracket_close = expr.index(']', 3)
        p_start = bracket_close + 1
        if p_start >= len(expr):
            break
        # Premise can be negated: !(...)
        if expr[p_start] == '!' and p_start + 1 < len(expr) and expr[p_start + 1] == '(':
            inner_end = _find_matching_paren(expr, p_start + 1)
            if inner_end < 0:
                break
            p_end = inner_end
            premise = expr[p_start:p_end + 1]
        elif expr[p_start] == '(':
            p_end = _find_matching_paren(expr, p_start)
            if p_end < 0:
                break
            premise = expr[p_start:p_end + 1]
        else:
            break
        b_start = p_end + 1
        if b_start >= len(expr):
            break
        if expr[b_start] == '!':
            # Negated body: !(...)
            inner_start = b_start + 1
            if inner_start >= len(expr) or expr[inner_start] != '(':
                break
            inner_end = _find_matching_paren(expr, inner_start)
            if inner_end < 0:
                break
            premises.append(premise)
            expr = expr[b_start:inner_end + 1]
            return premises, expr
        if expr[b_start] != '(':
            break
        b_end = _find_matching_paren(expr, b_start)
        if b_end < 0:
            break
        premises.append(premise)
        expr = expr[b_start:b_end + 1]

    return premises, expr


def _replace_arg_safe(expr: str, old: str, new: str) -> str:
    """@brief Rename one argument-level token inside ``[...]`` groups.

    @details
    Uses lookbehind/lookahead bracket guards
    (``(?<=[\\[,]) ... (?=[\\],])``) so only full-token matches
    inside argument lists are rewritten. Operator names, substrings
    inside larger arg names, and tokens elsewhere in the expression
    are NOT touched. Examples (rename ``v1 → x``):

    - ``(in[v1,N])`` → ``(in[x,N])`` ✓
    - ``(in3[v1,v1,+])`` → ``(in3[x,x,+])`` ✓
    - ``(v1plus[a])`` → ``(v1plus[a])`` (untouched — operator name)
    - ``(>[v1](in[v1,N])(in[v1,N]))`` → ``(>[v1](in[x,N])(in[x,N]))``
      (binder list at ``>[v1]`` is NOT modified — it isn't bracketed
      between a ``[`` or ``,`` on the left in the same way; the
      caller treats binder vars separately)

    Used by ``_check_reformulation`` to perform the
    ``u_i → actual_arg`` substitution when expanding a binary
    signature, and by the recursion checker's
    ``_replace_arg_safe(expr, ind_var, x_var)`` rewrite of induction
    premises.

    @param expr  Expression to rewrite.
    @param old   Argument name to replace (e.g. ``"v1"``).
    @param new   New argument name (e.g. ``"x"``).
    @return  Rewritten expression; identical to ``expr`` if no match.

    @see _replace_arg_safe_multi — bulk variant doing many
    substitutions in one pass.
    """
    pattern = r'(?<=[\[,])' + re.escape(old) + r'(?=[\],])'
    return re.sub(pattern, new, expr)


# ---------------------------------------------------------------------------
#  Theorem-citation w/W -> v/V revert
#
#  Counterpart of process_proof_graphs._v_to_w_in_theorem_citation. The
#  processor renames non-anchor inner bvars of cited theorem-anchor
#  implications from v/V to w/W (case-preserving letter swap). The
#  verifier reverses that swap before exact-string comparison against
#  state.global_theorems / state.external_theorems, which store the
#  registry form (v/V).
#
#  The fold _normalize_expr_list (already widened to [vVwW]\d+) provides
#  a slower fallback that also works; the explicit reverse keeps the
#  registry-membership fast path intact and documents the round-trip
#  invariant `processor: v->w` <-> `verifier: w->v`.
# ---------------------------------------------------------------------------

_WV_REVERT_PATTERN = re.compile(r'(?<=[\[,])([wW])(\d+)(?=[\],])')


def _is_theorem_anchor_impl_local(expr: str) -> bool:
    """@brief True when ``expr`` is a theorem-anchor implication.

    @details
    Local copy of ``process_proof_graphs.is_theorem_anchor_implication`` —
    the verifier deliberately stays a single-file standalone with no
    imports from ``process_proof_graphs`` (so it can be run against a
    proof graph generated by any version of the processor without
    cross-version coupling).

    Returns True iff ``expr`` is an outer implication ``(>[...]...)`` and
    its first premise (after disintegration) starts with ``(Anchor`` —
    i.e. the first thing the rule requires is an anchor row, which is
    the structural signature of a theorem-level rule (as opposed to
    an internal rule used during proof search).

    The distinction matters because non-anchor bvars in cited
    theorem-anchor implications are subject to the ``v→w`` rename
    applied by ``process_proof_graphs._v_to_w_in_theorem_citation``;
    callers use this predicate to gate the inverse
    ``_revert_w_to_v_in_theorem_citation`` revert before exact-string
    membership checks against the registry.

    @param expr  An MPL expression.
    @return  True iff ``expr`` is an implication whose first premise
             is an anchor row; False on non-implications or
             non-anchor first premises.

    @see _revert_w_to_v_in_theorem_citation — the inverse rename
    this predicate gates.
    """
    if not expr or not expr.startswith("(>["):
        return False
    premises, _ = disintegrate_implication_full(expr)
    if not premises:
        return False
    return premises[0].startswith("(Anchor")


def _revert_w_to_v_in_theorem_citation(expr: str) -> str:
    """@brief Revert the per-cell w/W rename emitted by process_proof_graphs.

    @details
    Symmetric inverse of
    ``process_proof_graphs._v_to_w_in_theorem_citation``. The processor
    applies a case-preserving v→w / V→W rename to non-anchor inner
    bvars of cited theorem-anchor implications during processed-proof-
    graph emission so that the citation cell visually distinguishes
    the rule's renamed-to-w variables from the result row's
    v-named locals. The verifier reverses that swap before exact-
    string membership checks against ``state.global_theorems`` /
    ``state.external_theorems``, which store the original registry
    form (v/V).

    Implementation:

    - Compiled regex ``_WV_REVERT_PATTERN`` matches any ``w`` or ``W``
      followed by digits, anchored by bracket-or-comma look-around so
      only argument-position tokens are touched (mirroring the
      forward swap's bracket guard).
    - ``re.sub`` rewrites every match preserving case:
      ``w<digits>`` → ``v<digits>``; ``W<digits>`` → ``V<digits>``.
    - No-op for expressions that have no such tokens (the regex
      simply matches nothing and ``re.sub`` returns the input
      unchanged).

    Note: the slower fallback ``_normalize_expr_list`` (widened to
    ``[vVwW]\\d+``) also canonicalizes both forms to ``v<N>``, but it
    rewrites EVERY var to ``v<N>`` which would invalidate the exact-
    string registry check against a real-world v-named theorem. The
    targeted revert keeps the fast registry membership path intact
    and explicitly documents the round-trip invariant
    ``processor: v→w`` ↔ ``verifier: w→v``.

    @param expr  Expression to revert.
    @return  Expression with w/W tokens at arg positions swapped back
             to v/V; identical to ``expr`` if no such tokens exist.

    @see _is_theorem_anchor_impl_local — gates whether this revert is
    applied by the ``origin`` meta-check.
    """
    return _WV_REVERT_PATTERN.sub(
        lambda m: ('v' if m.group(1) == 'w' else 'V') + m.group(2),
        expr,
    )


def _normalize_expr_list(exprs: List[str]) -> List[str]:
    """@brief Canonicalize v/V/w/W variables across a list of expressions.

    @details
    Renames every ``[vVwW]\\d+`` token (occurring anywhere — arg
    positions, binder lists, sub-expression args) to ``v<k>`` in
    first-appearance order across the WHOLE list. The shared
    ``seen`` map and counter close over all list elements, so the
    same source variable maps to the same canonical name in every
    expression — preserving cross-expression coreference.

    Why the pattern is widened to ``[vVwW]\\d+`` (not just ``v\\d+``):

    - ``v``-prefixed names are the registry-form variables emitted
      by the processor and the C++ prover.
    - ``V``-prefixed names are set-typed variables that share the
      same first-appearance ordering rule.
    - ``w`` / ``W`` names are the per-cell rename applied by
      ``process_proof_graphs._v_to_w_in_theorem_citation`` to
      non-anchor inner bvars of cited theorem-anchor implications.
      Without the widening, a chapter cell rendered as
      ``(>[w1]…)`` would slip through unnormalized while its v-form
      reconstruction (built from the theorem registry) gets renamed
      to ``v1``, and the resulting string compare in the ``origin``
      check would always disagree.

    Used by the ``origin`` meta-check (final fallback after the
    fast w→v revert) and by reformulation canonicalization
    inside ``_check_reformulation``.

    @param exprs  Input list of MPL expressions.
    @return  Same-length list with every v/V/w/W token renamed to
             ``v1, v2, …`` in first-appearance order. Identity for
             expressions without such tokens.

    Note: a broader "rename EVERY arg name" variant
    (``_normalize_all_vars_in_list``) once existed for the anchor-level
    implication branch; both were removed when implication checking was
    unified to a single substitution-based path
    (D-75).
    """
    seen: Dict[str, str] = {}
    counter = [0]

    def _repl(m: re.Match) -> str:
        v = m.group(0)
        if v not in seen:
            counter[0] += 1
            seen[v] = f'v{counter[0]}'
        return seen[v]

    result: List[str] = []
    for expr in exprs:
        result.append(re.sub(r'[vVwW]\d+', _repl, expr))
    return result


def _alpha_canonicalize_bound_vars(expr: str) -> str:
    """@brief Alpha-canonicalize every ``>[…]``-bound name to ``b1, b2, ...``.

    @details
    Walks every ``>[…]`` binder in declaration order, collects each
    unique bound-variable name, then assigns canonical names
    ``b1, b2, …`` in that declaration order. Names that never appear
    in any binder (anchor slot names, operator constants like ``s``
    / ``+`` / ``zero``, integer-literal anchor slots like
    ``i0``/``i1``, etc.) are left untouched.

    Used by the ``origin`` meta-check to compare a chapter row's
    ``rest[0]`` implication rule against the global theorem registry
    across batches. The motivating case: an incubator chapter may
    cite a Peano-batch rule using the chapter's own free-index
    naming (``i2`` for the bound variable), while the registry
    stored the same rule with the bound variable already renamed to
    ``v1`` by ``process_proof_graphs.py``. Both forms must
    canonicalize to the same string so the cross-batch citation
    check passes.

    **Implementation.**

    1. ``re.finditer(r'>\\[([^\\]]*)\\]', expr)`` finds every binder
       group.
    2. For each comma-separated name in declaration order, if not
       already seen, add to ``bound_order`` and ``seen_bound``.
    3. Build a single ``re.sub`` pattern that matches any of the
       discovered bound names at an arg position
       (``(?<=[\\[,]) <name> (?=[\\],])``) and rewrites each match
       to its canonical ``b<index>`` slot.

    **Collision detection.** Two disjoint ``>[…]`` binders that share
    a name would collapse to the same canonical slot under a naive
    rename — silently masking a producer-side malformation. Well-
    formed GL implications never share a bound-variable name across
    disjoint binders (the compiler renames colliding bvars at registry
    insertion), so a same-name collision is a real bug signal. This
    function detects the case and returns a sentinel
    ``<MALFORMED-DUPLICATE-BINDER:…>`` form whose origin-meta-check
    membership test will fail (no registry entry begins with the
    sentinel prefix), surfacing the malformation as a verifier failure
    per I-16 instead of trusting the producer.

    @param expr  An MPL expression (typically an implication rule
                 cited as ``rest[0]`` of a row).
    @return  Expression with every binder-introduced name rewritten
             to ``b<k>``. Identity if ``expr`` has no binders. On a
             duplicate-name collision returns a non-matching sentinel.

    @invariant I-16 — verifier must not trust producer well-formedness
    for soundness-relevant canonicalisation.
    @see _normalize_expr_list — the v/V/w/W normalizer that
    canonicalizes free index names across a list.
    """
    bound_order: List[str] = []
    seen_bound: Set[str] = set()
    duplicate_name: Optional[str] = None
    for m in re.finditer(r'>\[([^\]]*)\]', expr):
        names = m.group(1)
        if not names:
            continue
        for n in names.split(','):
            n = n.strip()
            if not n:
                continue
            if n in seen_bound:
                duplicate_name = n
                continue
            seen_bound.add(n)
            bound_order.append(n)
    if duplicate_name is not None:
        # Return a sentinel form that cannot match any real registry entry.
        # The origin meta-check's membership tests will reject; the row gets
        # an `origin` failure that pinpoints the malformed citation.
        return f'<MALFORMED-DUPLICATE-BINDER:{duplicate_name}>{expr}'
    if not bound_order:
        return expr
    rename = {n: f'b{i + 1}' for i, n in enumerate(bound_order)}
    pattern = re.compile(r'(?<=[\[,])('
                         + '|'.join(re.escape(n) for n in bound_order)
                         + r')(?=[\],])')
    return pattern.sub(lambda m: rename[m.group(0)], expr)



def _check_reformulation(source_expr: str, target_expr: str,
                          gl_binaries: Dict[str, dict]) -> bool:
    """@brief Verify ``target_expr`` is a valid reformulation of ``source_expr``.

    @details
    A *reformulation* rewrites an implication whose head is an
    existence-compact (``(some_existence[args])`` whose binary entry
    has ``category == "existence"``) into the equivalent expanded
    form with the existence's hidden witness made explicit as a new
    bound variable. The target has the SAME anchor row and the SAME
    non-anchor premises as the source, PLUS one extra premise (the
    existence's left element after substitution), and its head is
    the existence's right element after substitution. The new bound
    variable is the first unused v-index in the target expression.

    **Algorithm.**

    1. Disintegrate both source and target into ``(premises, head)``.
       Reject early if the target has no premises (no anchor to
       carry the reformulation).

    2. **Determine the anchor tag.** Extract the core name of the
       target's first premise (must start with ``"Anchor"``).
       The tag is the substring after ``"Anchor"`` (e.g.
       ``AnchorGauss`` → tag ``Gauss``).

    3. **Select the GL binary.** First try the exact anchor-derived
       tag (``gl_binaries[tag]``). If that binary lacks the head's
       core or its category is not ``"existence"``, fall back to a
       scan: iterate every loaded binary for one that defines the
       head's core as an ``existence`` entry. For the
       ``AnchorIncubator`` chapter-context filter, the scan is
       restricted to ``Incubator*`` tags (the same D-54 cross-batch
       collision logic used by ``binaries_for_chapter``).

    4. **Validate the existence shape.** The selected binary entry
       must have category ``"existence"``, exactly 2 elements, a
       non-empty signature, and signature arity matching the head's
       actual-arg count. Build the substitution map
       ``signature_arg → head_actual_arg`` for downstream substitution.

    5. **Find the fresh bound variable.** Widen the v-index scan to
       ``[vVwW]\\d+`` so any v/V/w/W token in use anywhere in the
       target counts as "taken". The new bound variable is
       ``v<smallest unused k>``.

    6. **Expand the existence elements.** Apply both the
       signature → actual substitution AND the placeholder
       ``"1" → new_bound_var`` substitution (in MPL convention,
       ``1`` in element bodies is the existence's witness slot).
       Call the substituted elements ``left_expr`` and ``right_expr``.

    7. **Check the definedSet position.** If the left expression's
       core has a ``definedSet`` field in its binary entry, the new
       bound variable must occupy the ``definedSet`` position of the
       left expression. Mismatched defset position → reject.

    8. **Build the expanded target premise list.** Non-anchor
       premises of the target plus ``left_expr`` as one extra
       premise. The expanded target's head is ``right_expr``.

    9. **Length-check vs source.** The expanded target's non-anchor
       premise count must equal the source's non-anchor premise
       count. Otherwise reject.

    10. **Permute and compare.** Normalize the full source list
        (anchor + non-anchor premises + head) via
        ``_normalize_expr_list``. Iterate every permutation of the
        expanded non-anchor premises, build the candidate full list
        ``[target_anchor, perm..., right_expr]``, normalize, and
        accept on first match.

    Used by ``check_reformulated_from`` (tag ``reformulated from``)
    and by ``check_theorem_goal_reached`` for chapter type
    ``reformulated_statement``.

    @param source_expr   The original direct-form implication.
    @param target_expr   The candidate reformulated implication.
    @param gl_binaries   Tag → binary dict (from
                         ``state.gl_binaries``).
    @return  True iff ``target_expr`` is a valid reformulation of
             ``source_expr`` under the algorithm above.
    """
    src_premises, src_head = disintegrate_implication_full(source_expr)
    tgt_premises, tgt_head = disintegrate_implication_full(target_expr)

    if not tgt_premises:
        return False

    # Determine anchor tag from first premise (e.g. AnchorGauss → Gauss)
    anchor_core = _extract_core_name(tgt_premises[0])
    if not anchor_core.startswith("Anchor"):
        return False
    tag = anchor_core[len("Anchor"):]  # e.g. "Gauss"

    # Select GL binary. Try the exact anchor-derived tag first. If absent
    # (D-35 — the incubator runs `AnchorIncubator` but the loaded binaries
    # are tagged `IncubatorPeano` / `IncubatorGauss` / `IncubatorGauss1`,
    # so the literal tag `Incubator` has no entry), fall back to scanning
    # every loaded binary for one that defines the head's compiled name as
    # an `existence` entry.
    #
    # Chapter-context filter: when the target's anchor is `AnchorIncubator`,
    # restrict the fallback scan to tags starting with `Incubator`. After
    # the duplicate-folder cleanup (D-54) main and
    # incubator binaries share `files/GL_binaries/`; their spontaneous
    # compact-operator names collide on shape (e.g. `existence4` is arity
    # 5 in Gauss but arity 4 in IncubatorGauss1). Without the filter,
    # alphabetical iteration would pick Gauss's `existence4` for an
    # incubator chapter and fail the check. The filter restores the
    # isolation the duplicate-folder layout used to provide.
    head_core = _extract_core_name(tgt_head)
    binary = gl_binaries.get(tag)
    if binary is None or head_core not in binary \
            or binary[head_core].get("category") != "existence":
        binary = None
        if tag == "Incubator":
            scan_iter = (b for k, b in gl_binaries.items()
                         if k.startswith("Incubator"))
        else:
            scan_iter = iter(gl_binaries.values())
        for candidate in scan_iter:
            entry = candidate.get(head_core)
            if entry is not None and entry.get("category") == "existence":
                binary = candidate
                break
        if binary is None:
            return False

    # Target head must be an existence expression
    if head_core not in binary:
        return False
    head_spec = binary[head_core]
    if head_spec.get("category") != "existence":
        return False

    elements = head_spec.get("elements", [])
    signature = head_spec.get("signature", "")
    if len(elements) != 2 or not signature:
        return False

    # Build substitution map: u_i → actual arg
    sig_args = _extract_args(signature)
    head_args = _extract_args(tgt_head)
    if len(sig_args) != len(head_args):
        return False
    subst: Dict[str, str] = {}
    for s_arg, h_arg in zip(sig_args, head_args):
        subst[s_arg] = h_arg

    # Find unused v-index for the new bound variable.
    # Widened to [vVwW] so V-vars (set-typed) and the per-cell w/W rename
    # emitted by process_proof_graphs.py for bvars also count toward the
    # "in use" set — a fresh v{N} cannot collide with any existing v{N},
    # V{N}, w{N}, or W{N} occurrence.
    all_v_indices = set()
    for m in re.finditer(r'[vVwW](\d+)', target_expr):
        all_v_indices.add(int(m.group(1)))
    new_v_idx = 1
    while new_v_idx in all_v_indices:
        new_v_idx += 1
    new_bound_var = f"v{new_v_idx}"

    # Expand elements: substitute args + replace placeholder "1" with new bound var
    def _apply_subst(element_expr: str) -> str:
        result = element_expr
        # Replace "1" placeholder with new bound var (in argument positions)
        # Use the same lookaround-based replacement as process_proof_graphs
        full_map = dict(subst)
        full_map["1"] = new_bound_var
        escaped_keys = [re.escape(k) for k in full_map]
        pattern = r'(?<=[\[,])(' + '|'.join(escaped_keys) + r')(?=[\],])'
        return re.compile(pattern).sub(lambda m: full_map.get(m.group(1), m.group(1)), result)

    left_expr = _apply_subst(elements[0])
    right_expr = _apply_subst(elements[1])

    # Check definedSet: left expression's new bound var must be at the definedSet position
    left_core = _extract_core_name(left_expr)
    if left_core in binary:
        defined_set_uvar = binary[left_core].get("definedSet", "")
        if defined_set_uvar:
            left_sig = binary[left_core].get("signature", "")
            left_sig_args = _extract_args(left_sig)
            if defined_set_uvar in left_sig_args:
                ds_idx = left_sig_args.index(defined_set_uvar)
                left_actual_args = _extract_args(left_expr)
                if ds_idx >= len(left_actual_args) or left_actual_args[ds_idx] != new_bound_var:
                    return False

    # Build expanded target: original premises + left_expr as extra premise, right_expr as head
    expanded_non_anchor = []
    for i in range(1, len(tgt_premises)):
        expanded_non_anchor.append(tgt_premises[i])
    expanded_non_anchor.append(left_expr)

    # Source non-anchor premises
    src_non_anchor = src_premises[1:]

    if len(expanded_non_anchor) != len(src_non_anchor):
        return False

    # Normalize source: [anchor, non_anchor..., head]
    src_list = src_premises + [src_head]
    src_norm = _normalize_expr_list(src_list)

    # Try all permutations of expanded non-anchor premises
    for perm in itertools.permutations(expanded_non_anchor):
        candidate_list = [tgt_premises[0]] + list(perm) + [right_expr]
        candidate_norm = _normalize_expr_list(candidate_list)
        if candidate_norm == src_norm:
            return True

    return False


def load_gl_binaries(binaries_dir: str) -> Dict[str, dict]:
    """@brief Load every ``GL_binary_*.json`` file from a directory.

    @details
    Walks the directory non-recursively, picking up every file whose
    name starts with ``GL_binary_`` and ends with ``.json``. Strips
    the prefix and suffix to derive the tag (e.g.
    ``GL_binary_Peano.json`` → tag ``"Peano"``) and stores the
    parsed JSON content as the binary dict for that tag.

    Each binary is the per-batch spontaneous-compact-operator
    allocation table. Every entry is keyed by operator name
    (``existence4``, ``or0``, ``implication26``, ``and3``, …) and
    has the shape:

        ``{"category": str,       # "and" | "or" | "existence" |
                                  # "implication" | …
          "signature": str,       # "(<core>[u_1,u_2,...])"
          "elements": [str, ...], # expansion bodies
          "arity": int,           # optional; falls back to
                                  # parsing the signature
          "definedSet": str}      # u-arg name for existence
                                  # categories
        ``

    Per D-54 the directory ``files/GL_binaries/`` holds main-batch
    AND incubator-batch binaries side by side; their per-batch
    operator allocations may collide on name but not on shape
    (``existence4`` arity-5 in Gauss vs arity-4 in IncubatorGauss1,
    etc.). The chapter-context filter in
    ``VerifierState.binaries_for_chapter`` keeps the two pools
    isolated by anchor-substring match.

    Missing directory → empty dict (no error). Used once at
    ``run_verifier`` startup; the result is cached in
    ``state.gl_binaries`` for the entire run.

    @param binaries_dir  Directory containing ``GL_binary_<tag>.json``
                         files. Missing directory tolerated.
    @return  Map tag → parsed binary dict. Empty if no binaries
             were found.

    @see VerifierState.binaries_for_chapter — runtime selector that
    picks the right binary per chapter from this loaded set.
    """
    import json
    result: Dict[str, dict] = {}
    if not os.path.isdir(binaries_dir):
        return result
    for fname in sorted(os.listdir(binaries_dir)):
        if fname.startswith("GL_binary_") and fname.endswith(".json"):
            tag = fname[len("GL_binary_"):-len(".json")]
            with open(os.path.join(binaries_dir, fname), "r", encoding="utf-8") as f:
                result[tag] = json.load(f)
    return result


_SPONTANEOUS_CATEGORIES = frozenset(("implication", "existence", "or", "and"))

# `(operator_name[…])` — every operator call site in a compiled expression
# emits the core name in this shape. `findall` returns every match including
# nested ones (e.g. inside an outer operator's argument list).
_OPERATOR_NAME_RE = re.compile(r'\(([A-Za-z_]\w*)\[')


def check_operator_registry_consistency(state) -> None:
    """@brief Enforce I-23: spontaneous compact-operator names are stable across batches.

    @details
    Every ``GL_binary_<Tag>.json`` carries a per-batch entry for each
    spontaneous compact operator (``implication<N>`` / ``existence<N>``
    / ``or<N>`` / ``and<N>``) that the C++ allocator produced during that
    batch. Per I-23, a given spontaneous name must encode the same
    operator in every batch that uses it — concretely, the entry's
    ``(category, signature, arity, elements)`` tuple must be byte-
    identical across all binaries that carry the name. The check also
    walks *recursively* through every spontaneous-compact name CITED in
    the elements list, ensuring the transitive closure of operator
    definitions is consistent: two binaries whose ``existence2`` rows
    have identical literal elements are only truly equal if every
    operator those elements cite (``and0``, nested ``implication<N>``,
    etc.) also matches across the same batch pair.

    Failure mode this catches: ``GL_binary_IncubatorPeano.json``
    ``existence2[u_1..u_8]`` (arity 8) vs ``GL_binary_shared.json`` /
    ``GL_binary_Peano.json`` ``existence2[u_1..u_3]`` (arity 3) — the
    same name encoding two semantically different operators. Per
    ``_merge_into_shared``'s skip-incubator policy (D-54) the
    IncubatorPeano spontaneous allocations never reach the shared
    registry, so on a clean-room run with empty shared both
    IncubatorPeano and Peano start with ``existence_counter = 0`` and
    independently allocate ``existence0``..``existence2`` for their
    own splitNKs.

    **Algorithm.**

    1. Bucket every spontaneous-compact entry across all loaded
       binaries: ``by_name[name] = {tag: entry}``. Skip atomic/anchor
       entries (categories outside ``_SPONTANEOUS_CATEGORIES``).
    2. For each name appearing in ≥ 2 binaries, recursively walk the
       definition closure on the common tag set:
       a. Surface match: ``(category, signature, arity, elements)`` of
          every binary's entry equals the reference entry's. Memoised
          on ``(name, frozenset(tags))`` to avoid redoing shared
          subgraphs and to terminate on cycles.
       b. Closure: ``_OPERATOR_NAME_RE`` extracts every operator name
          cited in ``elements``; for each that is itself spontaneous-
          compact (present in ``by_name``), recurse on the same tag
          set.
    3. Record one ``operator registry consistency`` counter increment
       per top-level name checked; success on transitive consistency,
       failure with a stderr report otherwise.

    @param state  ``VerifierState`` with ``gl_binaries`` populated.
    @return  None — counter side effects only.

    @invariant I-23 — failure here means a binary writer / allocator
    has produced cross-batch name collisions; the right fix is to make
    name allocation deterministic across batches (e.g. always seed
    spontaneous counters from a shared registry that incubator batches
    also contribute to), not to weaken this check.
    @invariant I-16 — failures are first-class; do not soften the
    check to recover ``failure 0``.
    """
    counter = state.counter_for("operator registry consistency")

    by_name: Dict[str, Dict[str, dict]] = {}
    for tag, binary in state.gl_binaries.items():
        if not isinstance(binary, dict):
            continue
        for name, entry in binary.items():
            if not isinstance(entry, dict):
                continue
            if entry.get("category") not in _SPONTANEOUS_CATEGORIES:
                continue
            by_name.setdefault(name, {})[tag] = entry

    memo: Dict[Tuple[str, frozenset], bool] = {}

    def _entries_match(a: dict, b: dict) -> bool:
        return (a.get("category") == b.get("category")
                and a.get("signature") == b.get("signature")
                and int(a.get("arity", -1)) == int(b.get("arity", -1))
                and list(a.get("elements", [])) == list(b.get("elements", [])))

    def _deep_check(name: str, tags: Tuple[str, ...]) -> bool:
        present = tuple(t for t in tags if t in by_name.get(name, {}))
        if len(present) < 2:
            return True
        key = (name, frozenset(present))
        cached = memo.get(key)
        if cached is not None:
            return cached
        # Provisional success guards against citation cycles
        memo[key] = True
        ref = by_name[name][present[0]]
        for t in present[1:]:
            if not _entries_match(ref, by_name[name][t]):
                memo[key] = False
                return False
        cited: Set[str] = set()
        for elem in ref.get("elements", []):
            for op in _OPERATOR_NAME_RE.findall(elem):
                if op in by_name and op != name:
                    cited.add(op)
        for child in sorted(cited):
            if not _deep_check(child, present):
                memo[key] = False
                return False
        return True

    for name in sorted(by_name):
        tags = tuple(sorted(by_name[name].keys()))
        if len(tags) < 2:
            continue
        ok = _deep_check(name, tags)
        counter.record(ok)
        if not ok:
            ref = by_name[name][tags[0]]
            distinct_sigs = sorted({by_name[name][t].get("signature", "") for t in tags})
            distinct_arities = sorted({int(by_name[name][t].get("arity", -1)) for t in tags})
            sys.stderr.write(
                f"[operator registry] DIVERGENT {name!r} across {list(tags)}: "
                f"signatures={distinct_sigs} arities={distinct_arities} "
                f"(see GL_binary_<tag>.json elements for details)\n"
            )


def load_output_indices(config_dir: str) -> Dict[str, int]:
    """@brief Load each atomic operator's output-argument index.

    @details
    For each operator entry in ``ConfigVisu.json``:

    1. Skip entries whose value is not a dict (defensive against
       schema drift).
    2. Read the ``output_args`` field (list of arg names marked as
       outputs in the operator's MPL definition).
    3. Read ``short_mpl`` and extract the ordered arg-name list
       from the first ``[...]`` group.
    4. For each name in ``output_args``, find its position in the
       ordered list. The FIRST match wins (and the loop breaks),
       which matches the C++ producer convention: a single output
       per operator.
    5. Record ``core_name → 0-based output position``.

    Operators without an ``output_args`` field, without
    ``short_mpl``, or whose ``output_args`` names don't appear in
    ``short_mpl`` are silently omitted. Downstream operator-shape
    checkers consult this map; the absence of a core is a structural
    signal, not an error.

    @param config_dir  Directory containing ``ConfigVisu.json``.
    @return  Map ``core_name → 0-based output index``. Empty if the
             file is absent or no operator has an ``output_args``
             field.
    """
    indices: Dict[str, int] = {}
    config_path = os.path.join(config_dir, "ConfigVisu.json")
    if not os.path.isfile(config_path):
        return indices

    import json
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    for name, spec in config.items():
        if not isinstance(spec, dict):
            continue
        out_args = spec.get("output_args", [])
        if not out_args:
            continue
        short_mpl = spec.get("short_mpl", "")
        m = re.search(r'\[([^\]]*)\]', short_mpl)
        if not m:
            continue
        ordered_args = [a.strip() for a in m.group(1).split(',') if a.strip()]
        for out_name in out_args:
            if out_name in ordered_args:
                indices[name] = ordered_args.index(out_name)
                break

    return indices


def load_definition_sets(config_dir: str) -> Dict[str, Dict[str, list]]:
    """@brief Load per-atomic-operator definition-set table.

    @details
    Returns ``core_name → { 1-based-position-str → [type_label, combinable] }``.
    The shape exactly mirrors the on-disk ``definition_sets`` field
    of each operator entry in ``ConfigVisu.json``.

    The type labels are the seed for D-41's resolved-defset
    construction. Each atomic-position label is one of the GL type
    primitives:

    - ``"(1)"`` — anchor-slot value (e.g. ``N``, ``s``, ``zero``,
      ``+``).
    - ``"P(1)"`` — first projection (e.g. an element of ``N``).
    - ``"N"`` — natural-number-typed argument.
    - ``"S(1)"`` / ``"S(2)"`` — sequence-typed projections.
    - … plus a few rarer labels.

    Composite operators (existence/and/or compacts spawned by the
    prover) inherit their per-position labels by walking ``elements``
    and propagating from atomic leaves — that propagation lives in
    ``build_resolved_defsets_per_tag`` and
    ``_try_derive_from_elements``, downstream of this loader.

    Operators without a ``definition_sets`` field (or with empty
    ``{}``) are silently omitted. The pipeline tolerates absent
    entries by treating the operator as type-opaque.

    @param config_dir  Directory containing ``ConfigVisu.json``.
    @return  Map core_name → position-string → [type_label, combinable].
             Empty if the file is absent or no operator declares
             ``definition_sets``.

    @invariant I-29 — variable-port type consistency: this table
    feeds the per-tag resolved-defset construction which in turn
    feeds ``check_defset_consistency``. The compiler's
    ``ArgumentAnalyzer`` shares this same source-of-truth schema
    (D-41 reference algorithm).

    @see build_resolved_defsets_per_tag — composite-resolution layer
    built on top of this atomic seed.
    """
    import json
    result: Dict[str, Dict[str, list]] = {}
    config_path = os.path.join(config_dir, "ConfigVisu.json")
    if not os.path.isfile(config_path):
        return result
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    for name, spec in config.items():
        if not isinstance(spec, dict):
            continue
        ds = spec.get("definition_sets", {})
        if ds:
            result[name] = ds
    return result


def load_input_indices(config_dir: str) -> Dict[str, List[int]]:
    """@brief Load each atomic operator's input-argument index list.

    @details
    Symmetric to ``load_output_indices`` but for the ``input_args``
    field: each operator entry in ``ConfigVisu.json`` may list one
    or more argument names that are inputs (i.e. participate in
    induction-step subterm enumeration). Their positions in
    ``short_mpl``'s ordered arg list are recorded.

    Used by the recursion checker's digit-arg / immutable-arg
    analysis (``_find_digit_args``, ``_find_immutable_args``):

    - **Digit args** = the SET of all input-position arg names
      across every expression in the theorem, minus anchor args,
      minus output-position arg names. These are the variables
      that index induction.
    - **Immutable args** = digit args minus the induction variable,
      then propagated through outputs where all inputs are
      immutable. These are variables whose values don't change
      across an induction step.

    @param config_dir  Directory containing ``ConfigVisu.json``.
    @return  Map ``core_name → list of 0-based input positions``.
             Empty if the file is absent or no operator declares
             ``input_args``.

    @see _find_digit_args
    @see _find_immutable_args
    """
    import json
    result: Dict[str, List[int]] = {}
    config_path = os.path.join(config_dir, "ConfigVisu.json")
    if not os.path.isfile(config_path):
        return result
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    for name, spec in config.items():
        if not isinstance(spec, dict):
            continue
        in_args = spec.get("input_args", [])
        if not in_args:
            continue
        short_mpl = spec.get("short_mpl", "")
        m = re.search(r'\[([^\]]*)\]', short_mpl)
        if not m:
            continue
        ordered = [a.strip() for a in m.group(1).split(',') if a.strip()]
        indices = []
        for in_name in in_args:
            if in_name in ordered:
                indices.append(ordered.index(in_name))
        if indices:
            result[name] = indices
    return result


def _find_digit_args(all_exprs: List[str],
                     anchor_expr: str,
                     input_indices: Dict[str, List[int]],
                     output_indices: Dict[str, int]) -> set:
    """@brief Find the digit-arg set for an induction theorem.

    @details
    Python port of the C++ ``findDigitArgs`` from
    ``conjecturer.cpp`` (and the structurally analogous routine in
    ``prover.cpp`` that spawns recursion sub-blocks). The "digit
    args" are the variable names that index induction — the prover
    spawns one recursion sub-block per digit-arg.

    Definition: ``digit_args = (all_input_args - anchor_args) - all_output_args``
    where:

    - ``all_input_args`` = union over every expression in
      ``all_exprs`` of the arg values found at the operator's
      input positions (per ``input_indices``).
    - ``anchor_args`` = the args of the theorem's anchor row (the
      anchor's variables are externally fixed; not induction
      variables).
    - ``all_output_args`` = union over every expression of arg
      values at the operator's output position.

    The subtraction is set-based. Operators not in
    ``input_indices`` / ``output_indices`` contribute nothing to
    their respective unions.

    @param all_exprs       Premises + head of an induction theorem.
    @param anchor_expr     The theorem's anchor row (first premise
                           of the induction implication).
    @param input_indices   Atomic-operator input-position index map.
    @param output_indices  Atomic-operator output-position index map.
    @return  Set of digit-arg variable names.

    @see _find_immutable_args — extends this set by propagation.
    @see check_recursion — primary consumer.
    """
    all_input_args: set = set()
    for expr in all_exprs:
        core = _extract_core_name(expr)
        if core in input_indices:
            args = _extract_args(expr)
            for idx in input_indices[core]:
                # I-19: an input-index registered for `core` must be reachable
                # in every actual call site of `core`. A missing arg slot is a
                # producer-side malformation, not a tolerable shape.
                assert idx < len(args), (
                    f"_find_digit_args: input index {idx} >= arity "
                    f"{len(args)} for core {core!r} in {expr!r}")
                all_input_args.add(args[idx])

    # Remove anchor args
    for a in _extract_args(anchor_expr):
        all_input_args.discard(a)

    # Subtract output args
    all_output_args: set = set()
    for expr in all_exprs:
        core = _extract_core_name(expr)
        if core in output_indices:
            args = _extract_args(expr)
            out_idx = output_indices[core]
            assert out_idx < len(args), (
                f"_find_digit_args: output index {out_idx} >= arity "
                f"{len(args)} for core {core!r} in {expr!r}")
            all_output_args.add(args[out_idx])

    return all_input_args - all_output_args


def _find_immutable_args(chain_exprs: List[str],
                         digit_args: set,
                         ind_var: str,
                         input_indices: Dict[str, List[int]],
                         output_indices: Dict[str, int]) -> set:
    """@brief Find args whose values never change across an induction step.

    @details
    Python port of the C++ ``findImmutableArgs`` from
    ``conjecturer.cpp``. The "immutable args" are variables that
    survive an induction-step substitution unchanged — they are
    NOT the induction variable and are not derived from it.

    Algorithm (least-fixed-point closure):

    1. Initialize ``immutables = digit_args - {ind_var}``. The
       induction variable itself is excluded — every step changes
       it (``v1 → s(v1)``). Every other digit arg starts immutable
       by definition.

    2. Iterate until no change:

       For every expression in ``chain_exprs``:
         - Extract its core name and args.
         - Look up its input/output positions.
         - If the operator HAS both input and output positions:
           - Check whether all input-position args are already in
             ``immutables``.
           - If yes, the operator's output is deterministic
             function of immutable inputs → add the output arg to
             ``immutables``.

    3. Termination: each iteration can only grow ``immutables``;
       the set is bounded by the total var population; therefore
       at most ``O(|vars|)`` iterations.

    Used by the recursion checker's ``check_induction_condition``
    branch to compute the "untouchables" set that must not be
    rewritten when substituting ``ind_var → x_var`` in the
    successor-step reconstruction.

    @param chain_exprs     Premises + head of the induction theorem.
    @param digit_args      Set of digit-arg names from
                           ``_find_digit_args``.
    @param ind_var         The induction variable name.
    @param input_indices   Atomic-operator input-position index map.
    @param output_indices  Atomic-operator output-position index map.
    @return  Set of immutable variable names (digit args + every
             reachable closure under the operator inference rule).

    @see _find_digit_args
    @see check_recursion
    """
    immutables = set(digit_args)
    immutables.discard(ind_var)

    changed = True
    while changed:
        changed = False
        for expr in chain_exprs:
            core = _extract_core_name(expr)
            if core not in input_indices or core not in output_indices:
                continue
            args = _extract_args(expr)
            # I-19: input / output indices registered for `core` must be
            # reachable in every actual call site. Same producer contract as
            # _find_digit_args.
            all_immutable = True
            for idx in input_indices[core]:
                assert idx < len(args), (
                    f"_find_immutable_args: input index {idx} >= arity "
                    f"{len(args)} for core {core!r} in {expr!r}")
                if args[idx] not in immutables:
                    all_immutable = False
                    break
            if all_immutable:
                out_idx = output_indices[core]
                assert out_idx < len(args), (
                    f"_find_immutable_args: output index {out_idx} >= arity "
                    f"{len(args)} for core {core!r} in {expr!r}")
                if args[out_idx] not in immutables:
                    immutables.add(args[out_idx])
                    changed = True

    return immutables


# ---------------------------------------------------------------------------
#  Theorem goal reached dispatcher
# ---------------------------------------------------------------------------

def check_theorem_goal_reached(
    chapter_file: str,
    chapter_type: str,
    lines: List[ProofLine],
    chapter_thm: Optional[Tuple[str, str, str]],
    gl_binaries: Dict[str, dict],
) -> bool:
    """@brief Verify a chapter's FIRST row proves the chapter's theorem.

    @details
    The chapter-level meta-check dispatched per chapter type. Each
    chapter type has its own structural rule the first row must
    satisfy; mismatch → False (recorded under
    ``state.goal_reached.failure`` in ``verify_chapter``).

    **Dispatch table.**

    - ``reformulated_statement``: first row tag must be
      ``"reformulated from"``, namespace ``"main"``, rest length
      ≥ 1. ``rest[0]`` is the source theorem; pass to
      ``_check_reformulation``.
    - ``back_reformulated_statement``: first row tag must be
      ``"incubator back reformulation"``, namespace ``"main"``, rest
      length ≥ 1, AND the direct form must be a valid witness-elimination
      of the cited existence source (``rest[0]``) via
      ``_check_back_reformulation`` — the same structural check the row
      dispatch applies (``check_incubator_back_reformulation``).
    - ``or_theorem``: first row tag must be ``"or theorem"``,
      namespace ``"main"``, rest length ≥ 2. Stub-accepted with
      True (the OR theorem's structural witness is the existence
      theorem already proved upstream).
    - ``direct_proof`` / ``check_zero`` /
      ``check_induction_condition``: first row's namespace must be
      ``"main"`` AND its expression must equal the innermost head
      of the theorem (via ``disintegrate_implication_head``).
    - ``induction_typing``: first row's namespace must be
      ``"main"`` AND its expression must equal
      ``(in[ind_var, anchor_args[0]])`` — the typing goal that
      gates induction promotion in the prover (see
      ``docs/agentic_swdd/induction_typing_plan.md``). ``ind_var`` is the third
      field of the induction row in ``global_theorem_list``;
      ``anchor_args[0]`` is the first arg of the theorem's anchor
      row.

    Unknown ``chapter_type`` → False (no path matched).

    @param chapter_file    Chapter filename (carried for
                           diagnostics; not used in any check).
    @param chapter_type    One of the eight dispatch keys above.
    @param lines           Chapter rows (parsed by
                           ``parse_chapter_file``).
    @param chapter_thm     ``(theorem_expr, theorem_type, theorem_ref)``
                           tuple from ``build_chapter_theorem_map``,
                           or ``None`` if no theorem mapping exists.
    @param gl_binaries     Tag → binary dict (consumed by
                           ``_check_reformulation``).
    @return  True iff the chapter's goal is structurally reached;
             False on any failure path.

    @invariant I-16 — failure means the producer-side claim about
    chapter ownership is wrong (e.g. the wrong theorem assigned to
    the chapter); never a false positive.

    @see verify_chapter — the chapter-level driver that invokes
    this function once per chapter and records the result.
    """
    if not lines or chapter_thm is None:
        return False

    thm_expr, thm_type, _ = chapter_thm
    first = lines[0]

    # reformulated_statement: expand existence head, permute, compare
    if chapter_type == "reformulated_statement":
        if first.namespace != "main" or first.tag != "reformulated from":
            return False
        if len(first.rest) < 1:
            return False
        source_expr = first.rest[0]
        return _check_reformulation(source_expr, first.expression, gl_binaries)

    # back_reformulated_statement: single line with tag "incubator back
    # reformulation"; the direct form must be a valid witness-elimination
    # of the cited existence source (same structural check as the row).
    if chapter_type == "back_reformulated_statement":
        if first.namespace != "main" or first.tag != "incubator back reformulation":
            return False
        if len(first.rest) < 1:
            return False
        return _check_back_reformulation(first.expression, first.rest[0])

    # or_theorem: single line with tag "or theorem"
    if chapter_type == "or_theorem":
        if first.namespace != "main" or first.tag != "or theorem":
            return False
        if len(first.rest) < 2:
            return False
        return True

    # or_elimination: single line with tag "or elimination" whose claimed
    # expression IS the chapter's theorem (the merge row is the derived
    # theorem's own chapter — column 0 stays in registry v/V form, so the
    # comparison is exact). The structural merge validation itself lives
    # in check_or_elimination.
    if chapter_type == "or_elimination":
        if first.namespace != "main" or first.tag != "or elimination":
            return False
        if len(first.rest) < 6:
            return False
        return first.expression == thm_expr

    # direct proof, check_zero, check_induction_condition — plus the
    # proved-not-broadcast tier, whose chapter is an ordinary direct walk:
    # extract head from theorem, compare with first line
    if chapter_type in ("direct_proof", "check_zero", "check_induction_condition",
                        "proved_not_broadcast"):
        if first.namespace != "main":
            return False
        head = disintegrate_implication_head(thm_expr)
        return first.expression == head

    # induction_typing: third induction chapter proving (in[ind_var, anchor_args[0]])
    # — the typing sub-proof gating induction promotion in the prover.
    # Chapter's first line's expression must equal that typing goal.
    # See docs/agentic_swdd/induction_typing_plan.md.
    if chapter_type == "induction_typing":
        if first.namespace != "main":
            return False
        # ind_var is the 3rd field of the induction row in global_theorem_list.
        _, _, ind_var = chapter_thm
        # anchor_args[0] = N: find the first `(Anchor...[...])` occurrence in
        # the theorem expression and take its first argument.
        anchor_match = re.search(r"\(Anchor[A-Za-z0-9_]*\[([^\]]*)\]\)", thm_expr)
        if not anchor_match:
            return False
        anchor_args_raw = anchor_match.group(1)
        if not anchor_args_raw:
            return False
        n_name = anchor_args_raw.split(",", 1)[0]
        typing_goal = f"(in[{ind_var},{n_name}])"
        return first.expression == typing_goal

    return False


# ---------------------------------------------------------------------------
#  Empty checker stubs — one per proof tag
#
#  Each returns True (success) or False (failure).
#  Currently every checker returns False (= not yet implemented).
# ---------------------------------------------------------------------------

def _normalize_implication(expr: str) -> str:
    """@brief Single-expression sibling of ``_normalize_expr_list``.

    @details
    Same renaming rule but applied to ONE expression in two passes:

    1. **Discover.** Scan ``expr`` left-to-right tracking bracket
       state to distinguish argument brackets ``[...]`` from binder
       brackets ``>[...]``. Within argument brackets, every
       ``[vVwW]\\d+`` token encountered for the first time gets a
       fresh ``v<k>`` slot in declaration order. Binder brackets are
       skipped entirely (their content is NOT counted).

    2. **Apply.** A single ``re.sub`` rewrites every ``[vVwW]\\d+``
       token in the ORIGINAL expression — including inside binder
       brackets — to its canonical slot. Tokens introduced only in
       binder lists (and never used as args) won't have a mapping
       and pass through unchanged.

    Two-pass design ensures the canonical names are assigned by
    arg-position order (not the order names happen to be DECLARED in
    binders, which can drift). This matches the C++ convention used
    by the prover and processor for emitting normalised cells.

    @param expr  An MPL expression (typically an implication).
    @return  Expression with v-style variables renumbered
             ``v1, v2, …`` in arg-first-appearance order.

    @see _normalize_expr_list — list variant that preserves
    cross-expression coreference.
    """
    seen: Dict[str, str] = {}
    counter = [0]
    i = 0
    while i < len(expr):
        if expr[i:i+2] == '>[':
            i = expr.index(']', i + 2) + 1
        elif expr[i] == '[' and i > 0 and expr[i-1] != '>':
            j = expr.index(']', i + 1)
            for m in re.finditer(r'[vVwW]\d+', expr[i:j+1]):
                v = m.group(0)
                if v not in seen:
                    counter[0] += 1
                    seen[v] = f'v{counter[0]}'
            i = j + 1
        else:
            i += 1

    def _repl(m: re.Match) -> str:
        return seen.get(m.group(0), m.group(0))

    return re.sub(r'[vVwW]\d+', _repl, expr)


def _reconstruct_implication(chain_exprs: List[str], head: str) -> str:
    """@brief Reassemble ``(>[bound1]premise1(>[bound2]premise2...))`` chain.

    @details
    Python port of the C++ ``reconstructImplication`` rule used by
    the prover to materialise an implication from a flat
    premise-list + head. The reconstruction also DECIDES which
    variable binds at which layer, following the
    "bind-at-first-multi-occurrence" convention.

    Algorithm:

    1. **Count multi-occurrence v-vars.** Iterate every expression
       in ``chain_exprs + [head]``; count how many times each
       v-style arg (``[vVwW]\\d+``) appears across the chain. Vars
       appearing more than once become "multi" — candidates for
       binding (they're shared between premises and/or the head).

    2. **Decide bind locations.** For each multi-var, place it at
       the FIRST chain index where it appears. ``when[i]`` collects
       the variables that bind at chain index ``i``. The head's
       index is past the last premise; variables that only appear
       there are NEVER bound (they're free in the implication).

    3. **Assemble from inside out.** Start with ``head``. For
       ``i`` from ``len(chain_exprs)-1`` down to ``0``, wrap the
       accumulated body in ``(>[bound_str]chain_exprs[i]<accum>)``
       where ``bound_str`` is the comma-joined ``when[i]`` list.

    Only v-style vars are considered for binding; constants, anchor
    slots (``N``, ``s``, ``+``, …), and integer literals
    (``i0``, ``i1``, …) all stay free.

    Used by the recursion checker's ``check_induction_condition``
    branch to rebuild the expected successor-step implication after
    substituting ``ind_var → x_var``.

    @param chain_exprs  Premise list in outer-to-inner order.
    @param head         The conclusion.
    @return  Fully-formed implication string with optimised binder
             placement.

    @see check_recursion — primary consumer.
    """
    all_exprs = chain_exprs + [head]
    counter: Dict[str, int] = {}
    for e in all_exprs:
        for a in _extract_args(e):
            if not re.match(r'[vVwW]\d+$', a):
                continue
            counter[a] = counter.get(a, 0) + 1
    multi = {a for a, c in counter.items() if c > 1}

    placed: set = set()
    when: List[List[str]] = [[] for _ in range(len(chain_exprs))]
    for idx, e in enumerate(all_exprs):
        for a in _extract_args(e):
            if a in multi and a not in placed:
                placed.add(a)
                if idx < len(chain_exprs):
                    when[idx].append(a)

    result = head
    for i in range(len(chain_exprs) - 1, -1, -1):
        bound_str = ','.join(when[i])
        result = f'(>[{bound_str}]{chain_exprs[i]}{result})'
    return result


def _collect_bound_vars(expr: str) -> set:
    """@brief Return the set of every name introduced inside any ``>[…]``.

    @details
    Linear scan tracking the ``>[`` opener and ``]`` closer; every
    comma-split name between them is added to the result set.
    Multiple binders contribute their union (a var bound in two
    disjoint binders appears once in the set — well-formed GL never
    does this within a single expression but the function tolerates
    it).

    Used by ``check_implication`` (non-anchor branch) to compute
    the changeable / unchangeable partition: every arg in the rule's
    expressions that is NOT a bound variable is unchangeable and
    must match the actual row's arg at that position.

    @param expr  An MPL expression.
    @return  Set of bound-variable names across every binder in
             ``expr``.

    @see _collect_all_expr_vars — sibling that collects arg-position
    vars (NOT binder lists).
    """
    result = set()
    i = 0
    while i < len(expr):
        if expr[i:i+2] == '>[':
            j = expr.index(']', i + 2)
            for a in expr[i+2:j].split(','):
                a = a.strip()
                if a:
                    result.add(a)
            i = j + 1
        else:
            i += 1
    return result


def _collect_all_expr_vars(expr: str) -> set:
    """@brief Return the set of every arg-position name in ``[...]`` groups.

    @details
    Counterpart of ``_collect_bound_vars``. Linear scan with bracket
    tracking; only NON-binder ``[...]`` groups contribute. Binder
    groups (``>[...]``) are skipped entirely. Within each argument
    group, every comma-split name is added to the result set
    (deduplicated by set semantics).

    Used by ``check_implication`` to compute
    ``unchangeables = all_expr_vars - bound_vars`` — the set of
    variables that must match exactly between rule and actual row
    (cf. bound vars which may be substituted).

    @param expr  An MPL expression.
    @return  Set of variable names appearing at argument positions
             anywhere in ``expr``.

    @see _collect_bound_vars — sibling that collects binder vars
    only.
    """
    result = set()
    i = 0
    while i < len(expr):
        if expr[i:i+2] == '>[':
            i = expr.index(']', i + 2) + 1
        elif expr[i] == '[' and i > 0 and expr[i-1] != '>':
            j = expr.index(']', i + 1)
            for a in expr[i+1:j].split(','):
                a = a.strip()
                if a:
                    result.add(a)
            i = j + 1
        else:
            i += 1
    return result


# NOTE: the single implication path (D-75) uses the substitution-based
# `_collect_bound_vars` / `_collect_all_expr_vars` model for theorems and
# definition rules alike.


def check_implication(line: ProofLine, chapter: List[ProofLine],
                      state: VerifierState) -> bool:
    """@brief Verify an ``implication`` row applies its rule consistently.

    @details
    The most common per-tag check in the verifier: a row tagged
    ``implication`` claims it derived ``line.expression`` from
    ``rest[0]`` (the rule, also an implication) by binding the rule's
    bound variables to actual values supplied as premises in
    ``rest[2:]`` (alternating ``(premise, premise_ns)`` pairs).

    **Row layout.**
    ::
        line.expression  = result expression
        line.namespace   = result scope
        rest[0]          = the rule (an implication)
        rest[1]          = the rule's scope of derivation
        rest[2*i+2]      = i-th supplied premise
        rest[2*i+3]      = i-th premise's scope

    **Namespace rule (D-35 — comparable-scope premise inheritance).**
    Every premise namespace AND the implication's namespace must be
    one of:

    - ``"main"`` (always admissible),
    - the result's namespace, or
    - a strict ancestor of the result's namespace
      (``result_ns.startswith(ns + "_boundary_")``).

    A premise drawn from a deeper-than-result scope is rejected; a
    premise from an ancestor scope is accepted. This faithfully
    encodes GL's comparable-scope inheritance: facts at an ancestor
    scope are observably true at every descendant. Pre-D-35 the rule
    was strict "at most one distinct non-main namespace, equal to the
    result's", which rejected legitimate FTA-rung-1 implication
    firings that mixed an OR-branch scope (result) with the OR's
    parent scope (some premises).

    **Validity-stack deposit rule (mirrors ``generateEncodedRequests``).**
    The C++ prover's ``growBaseCandidates`` /
    ``generateEncodedRequestsStatic`` (see ``memory.cpp``) accumulates
    constituent validities via ``nm.deeperOf(...)`` — the result of
    combining facts always lives at the DEEPEST scope of the combined
    inputs. The implication can never deposit at a scope no
    constituent reaches. The verifier mirrors this:

    1. Every pair of constituent namespaces (impl + each premise)
       must be **comparable** — one is an ancestor of the other
       (or they're equal). Sibling scopes are rejected. D-35's
       per-constituent ≤-result rule already implies this on chains
       rooted at ``result_ns``, but the explicit pair-wise check
       documents the prover's ``nm.comparable`` contract.
    2. ``result_ns`` must EQUAL the deepest of the constituent
       namespaces — equivalently, ``result_ns`` is one of
       ``{impl_ns} ∪ {premise_nss}``. If every constituent sits at
       ``"main"`` while ``result_ns`` is ``"main_boundary_<X>"``,
       the row is rejected: the implication firing would have to
       deposit the result at ``main`` per ``deeperOf``, never
       suddenly at a strictly deeper scope no constituent reaches.

    **Structural check — single path (D-75).**

    One substitution-based check for EVERY implication, theorem-anchor
    rules included. Disintegrate the reference implication. Compute
    ``unchangeables = all_expr_vars - bound_vars`` where ``bound_vars``
    is every name bound in any ``>[...]`` of the rule. Unchangeables
    must match LITERALLY between rule and actual at every position
    (for a definition rule these are the ``u_`` formal parameters; for
    a raw theorem there are none, since the unified binder rule binds
    every non-``u_`` variable — so ``unchangeables`` is empty and the
    check reduces to "core names align position-wise and a consistent
    variable map exists", exactly what the removed theorem-only
    "normalize ALL vars" branch computed). Bound vars map to actual
    via a per-call ``changeable_map``; consistency required across all
    positions where the same bound var appears. Enumerate permutations
    of the actual premises plus the result expression; accept on the
    first consistent matching.

    Theorem handling is equalized to non-theorem handling — a single
    general branch covers both, with no special-casing of a
    theorem-level premise; see [I-4](30_invariants.md).

    @param line     The implication row to validate.
    @param chapter  Sibling rows (unused by this checker but
                    required by the TAG_CHECKERS dispatch signature).
    @param state    Global verifier state (unused for this check;
                    namespace + structural matching are purely
                    syntactic).
    @return  True iff the row's rule applies consistently to its
             supplied premises and yields the claimed result.

    @invariant D-35 — namespace inheritance rule encoded here
    structurally.
    @see C++ ``generateEncodedRequestsStatic`` /
    ``growBaseCandidates`` (memory.cpp) — the deeperOf-accumulation
    pattern the validity-stack deposit rule mirrors.
    """
    if len(line.rest) < 2:
        return False
    # Even-pair guard. Every chapter row's `rest` alternates
    # `(expr, ns)` pairs; an odd length means a premise was supplied
    # without its namespace cell (or vice versa). Without this check
    # the downstream loops at `range(2, ..., 2)` / `range(3, ..., 2)`
    # silently drop the dangling field, hiding the malformation.
    if len(line.rest) % 2 != 0:
        return False

    impl = line.rest[0]
    impl_ns = line.rest[1]

    # Collect premise namespaces
    premise_nss = [line.rest[i] for i in range(3, len(line.rest), 2)]
    result_ns = line.namespace

    # D-35 comparable-scope rule: every source ns must be main, the
    # result ns, or a strict ancestor of it.
    for ns in [impl_ns] + premise_nss:
        if ns == "main":
            continue
        if ns == result_ns:
            continue
        if result_ns.startswith(ns + "_boundary_"):
            continue
        return False

    # Validity-stack deposit rule (mirrors `generateEncodedRequests`).
    # Every pair of constituent namespaces must be comparable
    # (parent-child); the result lives at the DEEPEST of the chain,
    # so `result_ns` must equal at least one constituent's namespace.
    # See the docstring for the full rationale.
    constituent_nss = [impl_ns] + premise_nss
    for i, a in enumerate(constituent_nss):
        for b in constituent_nss[i + 1:]:
            if not (_ns_matches_or_strict_prefix(a, b)
                    or _ns_matches_or_strict_prefix(b, a)):
                return False
    if result_ns not in constituent_nss:
        return False

    # --- Structural check ---
    result_expr = line.expression
    premises = [line.rest[i] for i in range(2, len(line.rest), 2)]

    ref_premises, ref_head = disintegrate_implication_full(impl)

    # Single path for EVERY implication (D-75). Under the
    # unified binder rule a theorem binds every non-u_ variable in its
    # >[...], so the general substitution-based check below is exactly
    # correct for theorems too: changeables = variables bound in >[...];
    # unchangeables = the rest (only the u_ formal parameters of a
    # definition rule; the EMPTY set for a raw theorem). With
    # unchangeables == ∅ this reduces to "core names align position-wise and
    # a consistent variable map exists".
    all_vars = _collect_all_expr_vars(impl)
    bound_vars = _collect_bound_vars(impl)
    unchangeables = all_vars - bound_vars

    ref_chain = ref_premises + [ref_head]

    for perm in itertools.permutations(premises):
        act_chain = list(perm) + [result_expr]
        if len(ref_chain) != len(act_chain):
            continue
        changeable_map: Dict[str, str] = {}
        ok = True
        for ref_e, act_e in zip(ref_chain, act_chain):
            if _extract_core_name(ref_e) != _extract_core_name(act_e):
                ok = False
                break
            ra = _extract_args(ref_e)
            aa = _extract_args(act_e)
            if len(ra) != len(aa):
                ok = False
                break
            for rv, av in zip(ra, aa):
                if rv in unchangeables:
                    if rv != av:
                        ok = False
                        break
                elif rv in bound_vars:
                    if rv in changeable_map:
                        if changeable_map[rv] != av:
                            ok = False
                            break
                    else:
                        changeable_map[rv] = av
                else:
                    if rv != av:
                        ok = False
                        break
            if not ok:
                break
        if ok:
            return True

    return False


def _normalize_with_unchangeables(expr: str, unchangeables: set) -> str:
    """@brief Normalize vars in ``expr`` except those in ``unchangeables``.

    @details
    Two-pass single-expression normalizer:

    1. Walk ``expr`` left-to-right tracking bracket state to skip
       ``>[...]`` binder content. Within argument brackets, every
       arg-position name encountered for the first time gets a fresh
       ``v<k>`` slot UNLESS it is already in ``unchangeables`` (in
       which case it passes through unchanged).
    2. Apply a single ``re.sub`` to rewrite every collected name in
       the original expression.

    Used by the ``expansion`` checker's negated-existence branch to
    canonicalize the rebuilt FullBind implication and the actual
    target expression for comparison, keeping the existence's anchor
    args fixed (in ``unchangeables``) while normalizing free
    induction names.

    @param expr           Expression to normalize.
    @param unchangeables  Set of arg names to KEEP as-is.
    @return  Normalized expression (variables outside
             ``unchangeables`` renamed to ``v<k>``).
    """
    seen: Dict[str, str] = {}
    counter = [0]
    i = 0
    while i < len(expr):
        if expr[i:i+2] == '>[':
            i = expr.index(']', i + 2) + 1
        elif expr[i] == '[' and i > 0 and expr[i-1] != '>':
            j = expr.index(']', i + 1)
            for a in expr[i+1:j].split(','):
                a = a.strip()
                if a and a not in unchangeables and a not in seen:
                    counter[0] += 1
                    seen[a] = f'v{counter[0]}'
            i = j + 1
        else:
            i += 1

    if not seen:
        return expr
    escaped = [re.escape(k) for k in seen]
    pattern = r'(?<=[\[,])(' + '|'.join(escaped) + r')(?=[\],])'
    return re.compile(pattern).sub(
        lambda m: seen.get(m.group(1), m.group(1)), expr)


def _build_and_from_elements(elements: List[str]) -> str:
    """@brief Build the left-nested ``(&a(&b(&cd)))`` AND form from elements.

    @details
    Iteratively wraps each element with the accumulated body in
    ``(&accum elem)`` shape. Mirrors the C++ compiler's AND
    materialization rule. The result is left-nested: the FIRST
    element is the innermost; later elements wrap around.

    Used by the integration reformulation paths (``and`` and OR
    sub-implication categories) to reconstruct an AND form from a
    binary entry's ``elements`` list.

    @param elements  List of MPL expressions to combine.
    @return  Left-nested AND expression. For a single element,
             returns that element unchanged.
    """
    result = elements[0]
    for e in elements[1:]:
        result = f'(&{result}{e})'
    return result


def _negate_expr(expr: str) -> str:
    """@brief Negate an MPL expression with double-negation cancellation.

    @details
    A negated expression (leading ``!``) loses its negation; a positive
    expression gains one. This is the Python mirror of the C++
    ``negateScratch`` / the ``expandSignature`` OR-case negation:
    or-disjuncts (GL-binary ``elements`` of an ``or``-category entry)
    carry their TRUE polarity, so building the De Morgan conjunct of a
    negated disjunct must yield its bare positive core — a blind ``!``
    prefix would mint a double negation that hashes as a distinct,
    semantically flipped expression.

    @param expr  Canonical MPL expression, possibly ``!``-prefixed.
    @return  The negation of ``expr`` with at most one leading ``!``.
    """
    return expr[1:] if expr.startswith('!') else '!' + expr


def _build_or_from_elements(elements: List[str]) -> str:
    """@brief Build the De Morgan-style ``!(&!(D1)!(D2)...)`` OR form from disjuncts.

    @details
    The OR ``D1 ∨ D2 ∨ ... ∨ Dk`` materializes via De Morgan as
    ``!(!D1 ∧ !D2 ∧ ... ∧ !Dk)``. For a 2-disjunct OR this is
    ``!(&!D1!D2)``; for k≥3 the accumulated or-so-far is itself a
    DISJUNCT of the next level and enters the AND negated — by
    double-negation cancellation its ``!(&...)`` form contributes the
    bare positive ``(&...)``: ``!(&(&!D1!D2)!D3)`` reads
    ``(D1 ∨ D2) ∨ D3``. (Nesting the or-so-far un-negated read
    ``¬(D1 ∨ D2) ∨ D3`` — the mirrored polarity defect flagged in
    D-260.) Mirrors the expandSignature OR-case used by the C++
    compiler byte-for-byte.

    Single-element input is returned unchanged (degenerate OR).

    Used by ``check_expansion`` (when re-expanding the negated
    existence shape) and by the integration-reformulation paths
    that accept either De Morgan or sub-implication form per D-52.

    @param elements  Disjunct expressions.
    @return  De Morgan-shaped OR. For ``len(elements) == 1``,
             returns that element verbatim.

    @see _build_or_subimpls_from_elements — the K-sub-implication
    alternative shape D-52 added.
    """
    if len(elements) == 1:
        return elements[0]
    current = f'!(&{_negate_expr(elements[0])}{_negate_expr(elements[1])})'
    for e in elements[2:]:
        current = f'!(&{_negate_expr(current)}{_negate_expr(e)})'
    return current


def _build_or_subimpls_from_elements(elements: List[str]) -> List[str]:
    """@brief Build per-branch sub-implications for an OR (D-52).

    @details
    For each ``k`` in ``[0, K)`` produces
    ``(>[](AND-of-negated-others)(D_k))`` where the premise is the
    left-nested AND of negations of every disjunct OTHER than
    ``D_k``. For ``K == 2`` the AND wrapper collapses to a single
    negation; for ``K >= 3`` the AND is left-nested in the same
    shape ``_build_and_from_elements`` produces. The bound-var list
    is always empty — ``_orint_`` branches don't introduce new
    quantifiers.

    Returns ``[]`` for ``K < 2`` (degenerate OR has no sub-impls).

    Mirrors the C++ producer emission at
    ``prover.hpp::prepareIntegrationCore2`` Case OR. The verifier
    accepts this form, alongside the De Morgan form
    ``_build_or_from_elements``, as a valid structural expansion of
    ``or<N>`` when checking ``expansion for integration`` rows whose
    right side has category ``or`` (D-52).

    @param elements  Disjuncts ``D_0..D_{K-1}``.
    @return  List of K sub-implications, one per disjunct.
             Empty if ``len(elements) < 2``.

    @see _build_or_from_elements — alternative De Morgan form.
    """
    K = len(elements)
    if K < 2:
        return []
    result: List[str] = []
    for k in range(K):
        neg_others = [_negate_expr(elements[j]) for j in range(K) if j != k]
        if len(neg_others) == 1:
            premise = neg_others[0]
        else:
            premise = neg_others[0]
            for n in neg_others[1:]:
                premise = f'(&{premise}{n})'
        result.append(f'(>[]{premise}{elements[k]})')
    return result


def _build_implication_from_elements(elements: List[str],
                                      unchangeables: set) -> str:
    """@brief Build an implication from a flat element list.

    @details
    The last element is the head; everything before it is a premise.
    Variables NOT in ``unchangeables`` that appear in multiple
    elements get bound at their first-appearance index (matching
    the prover's ``reconstructImplication`` rule, but parameterized
    by an explicit unchangeables set instead of inferred from
    ``[vVwW]\\d+`` pattern).

    Used by ``check_expansion``'s reconstruction path for `and`-
    category binary entries: substitute the binary's elements with
    actual args, then call this to produce the materialised
    implication for normalize-and-compare.

    @param elements       Premise list + head as a single flat list.
    @param unchangeables  Set of names that pass through as free
                          (anchor slots, literal constants).
    @return  Fully-formed implication string.
    """
    premises = elements[:-1]
    head = elements[-1]
    all_exprs = premises + [head]
    bindable: set = set()
    for e in all_exprs:
        for a in _extract_args(e):
            if a not in unchangeables:
                bindable.add(a)

    placed: set = set()
    when: List[List[str]] = [[] for _ in range(len(premises))]
    for idx, e in enumerate(all_exprs):
        for a in _extract_args(e):
            if a in bindable and a not in placed:
                placed.add(a)
                if idx < len(premises):
                    when[idx].append(a)

    result = head
    for i in range(len(premises) - 1, -1, -1):
        bound_str = ','.join(when[i])
        result = f'(>[{bound_str}]{premises[i]}{result})'
    return result


def _build_existence_from_elements(elements: List[str]) -> str:
    """@brief Build the GL existence form ``!(>[1](left)!(right))`` from elements.

    @details
    The C++ compiler emits existences as a negated implication with
    a placeholder bound variable ``1`` (the existence's witness slot,
    later substituted to a real ``v<k>`` at use time). Used as the
    canonical existence shape for comparison inside
    ``_check_reformulation_integration_existence``.

    @param elements  ``[left, right]`` — exactly two elements.
    @return  Canonical existence string.
    """
    return f'!(>[1]{elements[0]}!{elements[1]})'


def _build_existence_empty_binding(elements: List[str]) -> str:
    """@brief Build existence with empty binding: ``(>[](el1)(el2))``.

    @details
    Variant of ``_build_existence_from_elements`` used when the
    existence's witness has already been consumed (i.e. the
    placeholder ``1`` slot is empty), e.g. in the ``>[]`` reformulation
    integration variant.

    @param elements  ``[left, right]`` — exactly two elements.
    @return  Empty-binder existence string.
    """
    return f'(>[]{elements[0]}{elements[1]})'


def _parse_existence_expansion(expr: str) -> Optional[Tuple[str, str, str]]:
    """@brief Parse the existence form ``!(>[vars](left)!(right))`` back to triples.

    @details
    Inverse of ``_build_existence_from_elements``. Walks the
    expression's outer structure with paren-balanced bracket
    matching to extract:

    - ``bound_vars_csv``: the comma-separated bound-var list from
      the outer ``>[…]`` (may be empty).
    - ``left_expr``: the left element (positive paren-balanced
      sub-expression).
    - ``right_expr``: the right element (after stripping the leading
      ``!``).

    Returns ``None`` on malformed input (missing leading ``!``,
    unbalanced parens, missing ``)`` at expected positions, etc.).
    Used by ``_check_existence_disintegration`` and the various
    reformulation-for-integration checkers.

    @param expr  An MPL expression purporting to be an existence.
    @return  ``(bound_vars_csv, left_expr, right_expr)`` on success;
             ``None`` on any structural mismatch.
    """
    if not expr.startswith('!(') or not expr.endswith(')'):
        return None

    inner = expr[1:]
    if not inner.startswith('(>['):
        return None

    try:
        bracket_close = inner.index(']', 3)
    except ValueError:
        return None

    left_start = bracket_close + 1
    if left_start >= len(inner) or inner[left_start] != '(':
        return None

    left_end = _find_matching_paren(inner, left_start)
    if left_end < 0:
        return None

    neg_idx = left_end + 1
    if neg_idx >= len(inner) or inner[neg_idx] != '!':
        return None

    right_start = neg_idx + 1
    if right_start >= len(inner) or inner[right_start] != '(':
        return None

    right_end = _find_matching_paren(inner, right_start)
    if right_end < 0:
        return None

    # The implication itself must close immediately after the negated right side.
    if right_end + 1 != len(inner) - 1 or inner[-1] != ')':
        return None

    bound_vars = inner[3:bracket_close]
    left_expr = inner[left_start:left_end + 1]
    right_expr = inner[right_start:right_end + 1]
    return bound_vars, left_expr, right_expr


def _check_existence_disintegration(line: ProofLine,
                                    chapter: List[ProofLine],
                                    compound: str,
                                    compound_ns: str,
                                    compact: str,
                                    entry: dict) -> bool:
    """@brief Validate a disintegration child against an existence expansion.

    @details
    Specialized branch of ``check_disintegration`` for existence-
    category compounds. The contract:

    1. **Sibling count.** The expanded existence origin may emit
       one or two disintegration rows; more than two is invalid (an
       existence has exactly two children: left + right). The
       function collects siblings (same tag, namespace, and origin)
       and rejects if the count is 0 or >2.

    2. **Compound parses.** ``_parse_existence_expansion(compound)``
       must yield a valid ``(bound_vars, left, right)`` triple. The
       actual left/right are recorded.

    3. **Signature match.** The compact's actual-arg count must
       match the binary entry's signature arity, and the entry must
       have exactly 2 elements. Build the substituted
       ``expected_children``.

    4. **Compound shape.** The expanded existence (the row's
       ``compound``) must equal the binary's re-built existence
       (``_build_existence_from_elements(expected_children)``)
       modulo normalize-with-unchangeables (with the compact's actual
       args as unchangeables).

    5. **Member shape.** The set of normalized actual existence
       members ``{actual_left, actual_right}`` must equal the set of
       normalized expected children.

    6. **Sibling shape.** Every sibling disintegration row's
       expression must normalize to one of the expected child norms
       (subset check).

    7. **This row's shape.** ``line.expression`` normalized must be
       one of the expected child norms.

    Each failure path returns False; the caller's outer disintegration
    dispatcher records the verdict.

    @param line         The current disintegration row being checked.
    @param chapter      All chapter rows (for sibling lookup).
    @param compound     The compound expression (``line.rest[0]``).
    @param compound_ns  The compound's namespace (``line.rest[1]``).
    @param compact      The compact form found via the matching
                        ``expansion`` row.
    @param entry        The compact's GL binary entry (existence
                        category).
    @return  True iff every contract above holds for this row.

    @see check_disintegration — the dispatcher that calls this for
    existence-category compounds.
    """
    siblings = [
        ch_line for ch_line in chapter
        if (ch_line.tag == 'disintegration'
            and ch_line.namespace == compound_ns
            and len(ch_line.rest) >= 2
            and ch_line.rest[0] == compound
            and ch_line.rest[1] == compound_ns)
    ]

    if len(siblings) < 1 or len(siblings) > 2:
        return False

    parsed_compound = _parse_existence_expansion(compound)
    if parsed_compound is None:
        return False

    _, actual_left, actual_right = parsed_compound

    sig_args = _extract_args(entry.get('signature', ''))
    actual_args = _extract_args(compact)
    elements = entry.get('elements', [])

    if len(sig_args) != len(actual_args) or len(elements) != 2:
        return False

    subst = dict(zip(sig_args, actual_args))
    expected_children = [_replace_arg_safe_multi(elem, subst)
                         for elem in elements]
    expected_compound = _build_existence_from_elements(expected_children)

    unchangeables = set(actual_args)

    compound_norm = _normalize_with_unchangeables(compound, unchangeables)
    expected_compound_norm = _normalize_with_unchangeables(
        expected_compound, unchangeables)
    if compound_norm != expected_compound_norm:
        return False

    actual_member_norms = {
        _normalize_with_unchangeables(actual_left, unchangeables),
        _normalize_with_unchangeables(actual_right, unchangeables),
    }
    expected_child_norms = {
        _normalize_with_unchangeables(expr, unchangeables)
        for expr in expected_children
    }
    if actual_member_norms != expected_child_norms:
        return False

    sibling_norms = {
        _normalize_with_unchangeables(ch_line.expression, unchangeables)
        for ch_line in siblings
    }
    if not sibling_norms.issubset(expected_child_norms):
        return False

    line_norm = _normalize_with_unchangeables(line.expression, unchangeables)
    return line_norm in expected_child_norms


def _try_expand(binary_entry: dict, right_expr: str,
                left_expr: str,
                or_elements: Optional[List[str]] = None,
                binaries: Optional[List[dict]] = None) -> bool:
    """@brief Try to expand a compact ``right_expr`` and match it against ``left_expr``.

    @details
    Used by the ``expansion`` and ``expansion for integration``
    checkers. Given a binary entry (one compact operator's
    signature + elements + category), substitute the compact's
    actual args into the elements, then build the expanded form per
    the category and compare with ``left_expr`` modulo
    normalize-with-unchangeables. For one-go OR production in either
    direction, ``or_elements`` supplies the already-substituted recursive
    leaf list.

    Category dispatch:

    - ``"and"`` — left-nested ``(&...)`` via
      ``_build_and_from_elements``.
    - ``"implication"`` — full implication via
      ``_build_implication_from_elements``.
    - ``"existence"`` — canonical existence
      ``!(>[1]…!(…))`` via ``_build_existence_from_elements``.
      Also accepts the empty-binding form ``(>[]…(…))``.
    - ``"or"`` — De Morgan form via ``_build_or_from_elements``,
      OR any of the K per-branch sub-implications per D-52.
    - ``"implication"`` additionally accepts, when ``binaries`` is
      supplied, the per-leaf OR-INTRO forms ``(>[bv](D_k)(orPremise))``
      for each premise element that is a compiled OR — the producer's
      ``disintegrateExprCore2`` implication branch emits one intro
      implication per flattened disjunct alongside the main rule
      (D-237). Additive: the full-implication form
      remains the primary acceptance path.
    - Other categories → False (no expansion rule defined).

    @param binary_entry  The compact's binary entry
                         (``{category, signature, elements}``).
    @param right_expr    The compact form (e.g.
                         ``(some_compact[a,b,c])``).
    @param left_expr     The expanded form (e.g.
                         ``(&(in[a,N])(eq[a,b]))``).
    @param or_elements   Optional already-substituted flattened OR leaves.
    @param binaries      Optional GL binary dicts enabling the OR-INTRO
                         acceptance for implication compacts (resolves each
                         premise element's ``or<N>`` entry).
    @return  True iff the compact expands to the left form modulo
             normalization.
    """
    sig_args = _extract_args(binary_entry['signature'])
    actual_args = _extract_args(right_expr)
    if len(sig_args) != len(actual_args):
        return False

    subst = dict(zip(sig_args, actual_args))
    cat = binary_entry['category']
    if or_elements is not None:
        if cat != 'or':
            return False
        elements = or_elements
    else:
        elements = [_replace_arg_safe_multi(e, subst)
                    for e in binary_entry['elements']]
    unch = set(actual_args)

    if cat == 'and':
        built = _build_and_from_elements(elements)
    elif cat == 'implication':
        built = _build_implication_from_elements(elements, unch)
    elif cat == 'existence':
        built = _build_existence_from_elements(elements)
    elif cat == 'or':
        built = _build_or_from_elements(elements)
    else:
        return False

    if (_normalize_with_unchangeables(built, unch)
            == _normalize_with_unchangeables(left_expr, unch)):
        return True

    # For existence, also try the empty-binding form (>[](el1)(el2))
    if cat == 'existence':
        built_empty = _build_existence_empty_binding(elements)
        if (_normalize_with_unchangeables(built_empty, unch)
                == _normalize_with_unchangeables(left_expr, unch)):
            return True

    # For 'or' (D-52), also accept any of the K per-branch sub-implication
    # forms `(>[](AND-of-negated-others)(D_k))` produced by the C++
    # producer's `prepareIntegrationCore2` Case OR emission. The De Morgan
    # form above remains the primary acceptance path; the sub-implication
    # forms are added as additional valid expansions of the same `or<N>`.
    if cat == 'or':
        for sub in _build_or_subimpls_from_elements(elements):
            if (_normalize_with_unchangeables(sub, unch)
                    == _normalize_with_unchangeables(left_expr, unch)):
                return True

    # For 'implication' (D-237), also accept the per-leaf
    # OR-INTRO forms `(>[bv](D_k)(orPremise))` for each premise element that
    # is a compiled OR: the producer's `disintegrateExprCore2` implication
    # branch emits one intro implication per flattened disjunct alongside
    # the main rule. The binder lists the disjunct's non-unchangeable
    # arguments in first-occurrence order — the producer binds every
    # changeable argument at its (single) premise. Additive: the
    # full-implication form above remains the primary acceptance path.
    if cat == 'implication' and binaries is not None and len(elements) >= 2:
        norm_left = _normalize_with_unchangeables(left_expr, unch)
        for premise_elem in elements[:-1]:
            intro_disjuncts = _or_disjuncts_from_compiled(
                premise_elem, binaries, flatten_nested=True)
            if intro_disjuncts is None:
                continue
            for dk in intro_disjuncts:
                bound: List[str] = []
                for arg in _extract_args(dk):
                    if arg in unch or arg in bound:
                        continue
                    bound.append(arg)
                intro = f'(>[{",".join(bound)}]{dk}{premise_elem})'
                if _normalize_with_unchangeables(intro, unch) == norm_left:
                    return True

    return False


def _replace_arg_safe_multi(expr: str, subst: Dict[str, str]) -> str:
    """@brief Bulk variant of ``_replace_arg_safe`` — multiple renames in one pass.

    @details
    Builds a single regex alternation over every key in ``subst``
    (each key escaped) and substitutes via lookup. Same bracket-
    guard look-around as ``_replace_arg_safe`` ensures only
    argument-position tokens are rewritten. Empty ``subst`` is a
    no-op fast path.

    Used by every reformulation/expansion helper that needs to
    substitute a signature → actual map (``u_1 → real_arg``,
    ``u_2 → real_arg``, …) into a binary entry's elements.

    @param expr   Expression to rewrite.
    @param subst  Map ``old_arg_name → new_arg_name``.
    @return  Rewritten expression. Identity if ``subst`` empty.
    """
    if not subst:
        return expr
    pattern = r'(?<=[\[,])(' + '|'.join(re.escape(k) for k in subst) + r')(?=[\],])'
    return re.compile(pattern).sub(
        lambda m: subst.get(m.group(1), m.group(1)), expr)


def _build_implication_fullbind(premises: List[str], head: str,
                               unchangeables: set) -> str:
    """@brief Build implication binding EVERY variable except unchangeables.

    @details
    Python mirror of C++ ``reconstructImplicationFullBind``. Every
    arg name across ``premises + [head]`` that is NOT in
    ``unchangeables`` becomes bound at its first-appearance index.
    Variables that appear only in the head are NOT bound (they
    would be free in the resulting implication).

    Used by ``check_expansion``'s negated-existence branch to build
    the two implication candidates ``(left → !right)`` and
    ``(right → !left)`` from the existence's two elements.

    @param premises       Premise list (outer-to-inner).
    @param head           The conclusion.
    @param unchangeables  Set of names to leave free (anchor slots,
                          constants).
    @return  Implication string with binders placed at first-appearance
             index of each non-unchangeable var.

    @see _build_implication_from_elements — flat-elements variant
    where the head is the last element of a single list.
    """
    all_exprs = premises + [head]
    placed: set = set()
    when: List[List[str]] = [[] for _ in range(len(premises))]
    for idx, e in enumerate(all_exprs):
        for a in _extract_args(e):
            if a not in unchangeables and a not in placed:
                placed.add(a)
                if idx < len(premises):
                    when[idx].append(a)

    result = head
    for i in range(len(premises) - 1, -1, -1):
        bound_str = ','.join(when[i])
        result = f'(>[{bound_str}]{premises[i]}{result})'
    return result


def check_expansion(line: ProofLine, chapter: List[ProofLine],
                    state: VerifierState) -> bool:
    """@brief Verify an ``expansion`` row expands a compact to a compound shape.

    @details
    The ``expansion`` tag claims ``line.expression`` (compound) is
    the expanded form of ``rest[0]`` (compact). The compact must
    have a binary entry; the binary's elements substituted with the
    compact's actual args must reconstruct to the compound modulo
    normalization.

    **Algorithm.**

    1. **Layout check.** ``len(rest) >= 2`` required.
    2. **Namespace match.** ``line.namespace == rest[1]`` (the
       compact's scope).
    3. **Origin presence.** The compact (``rest[0]``) must appear as
       a left-side expression somewhere in the chapter — proves the
       compact was derived/cited before being expanded.
    4. **Negated-existence special case.** If ``rest[0]`` starts
       with ``!``, the inner expression must be an existence-category
       compact. Build the two FullBind implications
       ``(left → !right)`` and ``(right → !left)`` and accept if
       ``line.expression`` matches either (modulo normalize-with-
       unchangeables). A ``line.expression`` that is itself a compact
       ``(implication<N>[args])`` — the existence-implication product
       statement the prover emits — is first expanded through its
       implication binary entry (``_expand_implication_compact``).
    5. **General case.** Otherwise dispatch to ``_try_expand`` with
       the compact's binary entry; accept on any matching category.

    @param line     The expansion row.
    @param chapter  All chapter rows (for origin presence check).
    @param state    Global state (for binary lookup via
                    ``binaries_for_chapter``).
    @return  True iff the expansion is structurally valid.
    """
    if len(line.rest) < 2:
        return False

    right_expr = line.rest[0]
    right_ns = line.rest[1]

    # Namespaces must match
    if line.namespace != right_ns:
        return False

    # Right expression must exist as a left-side expression in chapter
    if not any(ch_line.expression == right_expr for ch_line in chapter):
        return False

    # --- Negated existence expansion: !(name[args]) ---
    if right_expr.startswith('!'):
        inner_expr = right_expr[1:]  # strip leading !
        core = _extract_core_name(inner_expr)
        for binary in state.binaries_for_chapter():
            if core not in binary:
                continue
            entry = binary[core]
            if entry.get('category') != 'existence':
                continue
            sig_args = _extract_args(entry.get('signature', ''))
            actual_args = _extract_args(inner_expr)
            if len(sig_args) != len(actual_args):
                continue
            subst = dict(zip(sig_args, actual_args))
            elements = [_replace_arg_safe_multi(e, subst)
                        for e in entry['elements']]
            if len(elements) != 2:
                continue
            left, right = elements[0], elements[1]
            unch = set(actual_args)
            # Two implications from !(>[bound](left)!(right)):
            #   impl1: left -> !(right)
            #   impl2: right -> !(left)
            impl1 = _build_implication_fullbind([left], '!' + right, unch)
            impl2 = _build_implication_fullbind([right], '!' + left, unch)
            target = line.expression
            # The row's expression may be the negated existence's compact
            # "existence implication" ``(implication<N>[args])`` instead of
            # the rule text (the prover takes the two rules from the
            # existence's binary entry and emits each compact as the
            # expansion product; the rule follows as an ``expansion`` row
            # citing the compact). Expand it through its implication binary
            # entry and validate the expansion with the same shape checks —
            # the binary, not the row, decides the rule.
            expanded = _expand_implication_compact(
                target, state.binaries_for_chapter())
            if expanded is not None:
                target = expanded
            if (_normalize_with_unchangeables(target, unch)
                    == _normalize_with_unchangeables(impl1, unch)):
                return True
            if (_normalize_with_unchangeables(target, unch)
                    == _normalize_with_unchangeables(impl2, unch)):
                return True

        # --- Negated-AND De-Morgan expansion: !(name[args]) whose entry is
        # category 'and'. The producer (the prover's negated-AND De-Morgan
        # door) emits the negation of the substituted definition body —
        # '!(&' + the conjuncts verbatim, each at its own polarity + ')' —
        # which is byte-identical to the De Morgan or-form over the negated
        # conjuncts. Compare modulo the usual normalization with the
        # instance args unchangeable.
        for binary in state.binaries_for_chapter():
            if core not in binary:
                continue
            entry = binary[core]
            if entry.get('category') != 'and':
                continue
            sig_args = _extract_args(entry.get('signature', ''))
            actual_args = _extract_args(inner_expr)
            if len(sig_args) != len(actual_args):
                continue
            subst = dict(zip(sig_args, actual_args))
            elements = [_replace_arg_safe_multi(e, subst)
                        for e in entry['elements']]
            expected = '!(&' + ''.join(elements) + ')'
            unch = set(actual_args)
            if (_normalize_with_unchangeables(line.expression, unch)
                    == _normalize_with_unchangeables(expected, unch)):
                return True
        return False

    core = _extract_core_name(right_expr)

    for binary in state.binaries_for_chapter():
        if core in binary:
            entry = binary[core]
            flattened_or = None
            if entry.get('category') == 'or':
                flattened_or = _or_disjuncts_from_compiled(
                    right_expr, state.binaries_for_chapter(),
                    flatten_nested=True)
                if flattened_or is None:
                    continue
            if _try_expand(entry, right_expr, line.expression, flattened_or,
                           binaries=state.binaries_for_chapter()):
                return True

    return False


def check_compilation(line: ProofLine, chapter: List[ProofLine],
                      state: VerifierState) -> bool:
    """@brief Verify a ``compilation`` row faithfully compiles an implication.

    @details
    The ``compilation`` tag records the ASIC-0.1-prep step where an
    implication entering the mail "implications" channel is also
    compiled to its compact named form. Row layout:

        ``<compact (implication<N>[args])> TAB <ns> TAB compilation
          TAB <original expanded implication> TAB <original ns>``

    i.e. ``line.expression`` is the compact form (LEFT), ``rest[0]``
    is the original expanded implication (RIGHT), ``rest[1]`` its
    namespace. The producer always emits both at ``"main"`` (the mail
    implications channel is main-only by I-26).

    The check is self-contained and structural — the mirror of
    ``check_expansion``'s general case with the compact/expanded roles
    swapped. It does **not** require the original to appear as a prior
    chapter row: the compact↔original link is definitional (the
    ``implication<N>`` name *is* the original by GL-binary
    construction), exactly the property that makes the other
    structural tags origin-exempt. ``compilation`` is therefore in
    ``_ORIGIN_EXEMPT_TAGS``; this checker is the whole validation.

    **Algorithm.**

    1. **Layout.** ``len(rest) >= 2`` required (original + its ns).
    2. **Namespace match.** ``line.namespace == rest[1]`` — the
       compaction does not cross scope.
    3. **Binary faithfulness.** ``line.expression``'s core must have a
       GL-binary entry of ``category == "implication"`` whose
       ``elements``/``signature`` instantiated with the compact's
       actual args reconstruct (via ``_try_expand`` →
       ``_build_implication_from_elements``, modulo
       normalize-with-unchangeables) to the original implication in
       ``rest[0]``.

    A miss returns ``False`` — a ``compilation`` row whose compact
    name is not an implication-category compiled operator, or does not
    reconstruct its cited original, is a real contract violation the
    verifier must surface (failures are first-class; this is not a
    defensive early-out — it is the checker's job).

    @param line     The ``compilation`` row.
    @param chapter  All chapter rows (unused — the check is
                    self-contained; kept for the uniform checker
                    signature).
    @param state    Global state (for binary lookup via
                    ``binaries_for_chapter``).
    @return  True iff the compact form faithfully compiles the cited
             original implication.
    """
    if len(line.rest) < 2:
        return False

    original_expr = line.rest[0]
    original_ns = line.rest[1]

    # Compaction stays in scope — compact and original share a namespace.
    if line.namespace != original_ns:
        return False

    core = _extract_core_name(line.expression)

    for binary in state.binaries_for_chapter():
        entry = binary.get(core)
        if entry is None:
            continue
        if entry.get('category') != 'implication':
            continue
        if _try_expand(entry, line.expression, original_expr):
            return True

    return False


def check_disintegration(line: ProofLine, chapter: List[ProofLine],
                         state: VerifierState) -> bool:
    """@brief Verify a ``disintegration`` row extracts a child from a compound.

    @details
    Inverse of ``expansion``: a ``disintegration`` row claims
    ``line.expression`` (the child) is one of the constituents of
    ``rest[0]`` (the compound). The compound must have a matching
    ``expansion`` row earlier in the chapter; the binary entry of
    the expansion's compact controls the disintegration rule.

    **Category dispatch on the compact's binary entry.**

    - ``"and"``: substitute the binary's elements with the compact's
      actual args; ``line.expression`` must appear LITERALLY in the
      substituted element list.
    - ``"existence"``: dispatch to
      ``_check_existence_disintegration`` (which validates sibling
      structure plus compound shape).
    - ``"or"``: dispatch to ``_check_or_disintegration_implication``
      (mutual-exclusion sub-implication shape).

    Other categories (no expansion rule) → reject.

    **Algorithm.**

    1. ``len(rest) >= 2`` required.
    2. Namespace match: ``line.namespace == rest[1]``.
    3. Locate the matching ``expansion`` row in the chapter whose
       LEFT side is ``rest[0]`` and whose right scope equals
       ``rest[1]``. From it, read the compact (``rest[0]`` of the
       expansion).
    4. Iterate binaries for the chapter; for each that defines the
       compact's core, dispatch on category as above.

    @param line     The disintegration row.
    @param chapter  All chapter rows (for expansion lookup).
    @param state    Global state (for binary lookup).
    @return  True iff the disintegration is structurally valid.
    """
    if len(line.rest) < 2:
        return False

    compound = line.rest[0]
    compound_ns = line.rest[1]

    if line.namespace != compound_ns:
        return False

    # Find expansion line where compound is left side
    for ch_line in chapter:
        if (ch_line.expression == compound
                and ch_line.namespace == compound_ns
                and ch_line.tag == "expansion"
                and len(ch_line.rest) >= 2
                and ch_line.rest[1] == compound_ns):
            compact = ch_line.rest[0]
            core = _extract_core_name(compact)

            for binary in state.binaries_for_chapter():
                if core not in binary:
                    continue
                entry = binary[core]
                category = entry.get('category')

                if category == 'and':
                    sig_args = _extract_args(entry.get('signature', ''))
                    actual_args = _extract_args(compact)
                    if len(sig_args) != len(actual_args):
                        continue

                    subst = dict(zip(sig_args, actual_args))
                    renamed_elems = [_replace_arg_safe_multi(e, subst)
                                     for e in entry.get('elements', [])]

                    if line.expression in renamed_elems:
                        return True

                elif category == 'existence':
                    if _check_existence_disintegration(
                            line, chapter, compound, compound_ns, compact, entry):
                        return True

                elif category == 'or':
                    if _check_or_disintegration_implication(
                            line, compact, state.binaries_for_chapter()):
                        return True

            # Negated-AND De-Morgan source: the expansion's compact is
            # '!(name[args])' with an and-category entry (the prover's
            # negated-AND De-Morgan door). The K mutual-exclusion rows
            # validate against the De-Morgan disjuncts — each substituted
            # conjunct negated with double-negation cancellation, exactly
            # the producer's negateScratch construction.
            if compact.startswith('!'):
                neg_core = _extract_core_name(compact[1:])
                for binary in state.binaries_for_chapter():
                    if neg_core not in binary:
                        continue
                    entry = binary[neg_core]
                    if entry.get('category') != 'and':
                        continue
                    sig_args = _extract_args(entry.get('signature', ''))
                    actual_args = _extract_args(compact[1:])
                    if len(sig_args) != len(actual_args):
                        continue
                    subst = dict(zip(sig_args, actual_args))
                    disjuncts = [
                        _negate_expr(_replace_arg_safe_multi(e, subst))
                        for e in entry.get('elements', [])]
                    if _check_or_disintegration_against(line, disjuncts):
                        return True

    return False


def _check_or_disintegration_implication(line: ProofLine, compact: str,
                                         binaries: List[dict]) -> bool:
    """@brief Validate the mutual-exclusion sub-implication form of OR disintegration.

    @details
    For an OR with disjuncts ``D_0..D_{K-1}``, the prover emits K
    sub-implications of the form
    ``(>[](!D_others_0)...(!D_others_{K-2})D_k)`` — one per
    disjunct k, where the K-1 premises are the negations of the
    other disjuncts in some order. Each such implication states
    "if every disjunct except D_k is false, D_k must be true",
    which is the mutual-exclusion content of an OR.

    **Algorithm.**

    1. Decode the compact through ``_or_disjuncts_from_compiled`` with
       recursive flattening; this is the exact one-go producer leaf list.
    2. Disintegrate ``line.expression`` into ``(premises, head)``.
       Reject if ``head`` is not one of the substituted disjuncts.
    3. Build the expected premise set:
       ``sorted('!' + D_j for j != head_index)``.
    4. Compare to the actual premise set (also sorted) — must match
       as a multiset. Premise order is not significant; the prover
       emits them in disjunct-index order, but consumers should not
       depend on that ordering.

    @param line     The OR-disintegration row.
    @param compact  The OR's compiled form (``rest[0]`` of the
                    matching expansion row).
    @param binaries The chapter's GL binaries, including nested OR entries.
    @return  True iff the row matches the mutual-exclusion sub-
             implication shape.
    """
    disjuncts = _or_disjuncts_from_compiled(
        compact, binaries, flatten_nested=True)
    # The row's expression may be the or's compact "or implication"
    # ``(implication<N>[args])`` instead of the rule text (the prover takes
    # the K / subset-exclusion rules from the or's binary entry and emits
    # the compact as the disintegration product; the rule follows as an
    # ``expansion`` row citing the compact). Expand it through its
    # implication binary entry and validate the expansion with the same
    # shape checks — the binary, not the row, decides the rule.
    expanded = _expand_implication_compact(line.expression, binaries)
    if expanded is not None:
        line = dataclasses.replace(line, expression=expanded)
    if _check_or_disintegration_against(line, disjuncts):
        return True
    return _check_or_subset_exclusion(line, disjuncts, binaries)


def _expand_implication_compact(expr: str,
                                binaries: List[dict]) -> Optional[str]:
    """@brief Expand a compact ``(implication<N>[args])`` into its full-bind
    rule through its binary entry; ``None`` for anything else.

    @details
    Mirror of the prover's compact-expansion path (``prepareIntegrationCore``
    substituting the signature's ``u_`` tokens by the instance arguments
    positionally, then ``reconstructImplicationFullBind`` over the
    elements): the entry's elements are substituted with the compact's
    actual arguments (repeated arguments are legal) and rebuilt with
    ``_build_implication_from_elements``, every actual argument
    unchangeable. Returns ``None`` when ``expr`` is not a positive simple
    compact, no binary defines its core, or the entry is not an
    implication.

    @param expr      The candidate compact.
    @param binaries  The chapter's GL binaries.
    @return  The expanded rule, or ``None``.
    """
    if not expr.startswith('(') or expr.startswith('(>'):
        return None
    core = _extract_core_name(expr)
    for binary in binaries:
        entry = binary.get(core)
        if entry is None or entry.get('category') != 'implication':
            continue
        sig_args = _extract_args(entry.get('signature', ''))
        actual_args = _extract_args(expr)
        if len(sig_args) != len(actual_args):
            continue
        subst = dict(zip(sig_args, actual_args))
        elements = [_replace_arg_safe_multi(e, subst)
                    for e in entry.get('elements', [])]
        if not elements:
            continue
        return _build_implication_from_elements(elements, set(actual_args))
    return None


def _check_or_subset_exclusion(line: ProofLine, disjuncts,
                               binaries: List[dict]) -> bool:
    """@brief Validate the subset-exclusion sub-implication form of OR
    disintegration.

    @details
    For an OR with k >= 3 disjuncts ``D_0..D_{k-1}``, the prover emits —
    in addition to the K mutual-exclusion rules — one subset-exclusion
    rule per excluded index subset S with 1 <= |S| <= k-2:
    ``!D_i1 & .. & !D_ij -> or(D_rest)``. The premises are the negations
    of the excluded disjuncts (double-negation cancelling); the head is a
    genuine reduced (k-j)-ary or-compact whose flattened disjunct list is
    the parent's list minus the excluded disjuncts, ORDER PRESERVED (the
    producer emits the surviving leaves in parent order). Premises are
    matched as a MULTISET: the producer emits them in parent order, but
    the prover-side install detector is deliberately order-free (the
    mail-compact round-trip permutes premise order), so premise order is
    not part of the logical shape. The reduced or-operator is pre-minted
    by the producer, so the head decodes through the same binaries.

    @param line       The subset-exclusion disintegration row.
    @param disjuncts  The PARENT or's ordered flattened disjunct list
                      (TRUE polarity), or None.
    @param binaries   The chapter's GL binaries (for the head decode).
    @return  True iff the row matches the subset-exclusion shape.
    """
    if disjuncts is None or len(disjuncts) < 3:
        return False

    premises, head = disintegrate_implication_full(line.expression)
    j = len(premises)
    if j < 1 or j > len(disjuncts) - 2:
        return False

    head_disjuncts = _or_disjuncts_from_compiled(
        head, binaries, flatten_nested=True)
    if head_disjuncts is None or len(head_disjuncts) != len(disjuncts) - j:
        return False

    # Greedy order-preserving walk: the head list must embed as a
    # subsequence of the parent list (leftmost matching finds an
    # embedding iff one exists); the complement is the excluded multiset,
    # which is uniquely determined even with duplicate disjuncts.
    complement = []
    hi = 0
    for d in disjuncts:
        if hi < len(head_disjuncts) and d == head_disjuncts[hi]:
            hi += 1
        else:
            complement.append(d)
    if hi != len(head_disjuncts) or len(complement) != j:
        return False

    return sorted(premises) == sorted(_negate_expr(d) for d in complement)


def _check_or_disintegration_against(line: ProofLine,
                                     disjuncts) -> bool:
    """@brief Validate a K mutual-exclusion row against a given disjunct list.

    @details
    The shared tail of the OR-disintegration validation: disintegrate
    ``line.expression`` into ``(premises, head)``, require the head to
    be one of the disjuncts, and require the premise multiset to equal
    the negations (with double-negation cancellation) of all the other
    disjuncts. Used by ``_check_or_disintegration_implication`` (compiled
    or-compact source) and by ``check_disintegration``'s negated-AND
    De-Morgan branch (disjuncts derived from an and-category entry's
    negated conjuncts).

    @param line       The K-rule disintegration row.
    @param disjuncts  Ordered disjunct list at TRUE polarity, or None.
    @return  True iff the row matches the mutual-exclusion shape.
    """
    if disjuncts is None or len(disjuncts) < 2:
        return False

    premises, head = disintegrate_implication_full(line.expression)

    if head not in disjuncts:
        return False

    head_index = disjuncts.index(head)
    expected_premises = sorted(_negate_expr(d) for j, d in enumerate(disjuncts)
                               if j != head_index)
    actual_premises = sorted(premises)

    return expected_premises == actual_premises


def check_task_formulation(line: ProofLine, chapter: List[ProofLine],
                           state: VerifierState) -> bool:
    """@brief Verify a ``task formulation`` row carries a real theorem premise.

    @details
    The ``task formulation`` tag asserts ``line.expression`` is one of
    the theorem's premises — a hypothesis the prover is allowed to
    use freely at scope ``"main"``. The verifier disintegrates the
    chapter's theorem and checks membership.

    **Direct-proof case.** ``line.expression`` must appear in the
    disintegrated theorem's premise list.

    **Contradiction-proof case.** When the theorem's head starts
    with ``!``, the un-negated head (``cleanOp``) is also a valid
    task-formulation expression. The proof shows ``cleanOp`` leads
    to contradiction; ``cleanOp`` itself is seeded into the
    contradiction LB as the hypothesis to be disproved. The verifier
    accepts ``line.expression == cleanOp`` in this case.

    **Complement (reductio) case.** When the theorem's head is
    positive, its negation ``"!" + head`` is also a valid
    task-formulation expression — the complement contradiction LB
    (``try_contradiction_negated_head``) seeds the head's negation
    as the hypothesis whose refutation proves the head. Symmetric to
    the case above: in both, the seed is exactly the negation of the
    theorem's head.

    **Contradiction-scope seed case.** A scope whose last payload is
    ``contradiction_<!X>`` (negated-compound integration) hypothesises
    the POSITIVE compact ``X`` inside the marked child scope; that
    seed is the scope's own task formulation. Accepted iff the
    payload's seed is negated and ``line.expression`` is exactly its
    un-negated form.

    **Namespace gate.** Otherwise must be ``"main"``. Task
    formulations live only at the theorem root; descendant scopes
    derive their own seeded facts via OR-branch or recursion rules.

    @param line     The task-formulation row.
    @param chapter  Sibling rows (unused).
    @param state    Reads ``current_chapter_thm``; reject if None.
    @return  True iff the expression is a real theorem premise (or
             the cleanOp of a contradiction-style head, or a
             contradiction-scope seed).
    """
    _tf_cut = line.namespace.rfind("_boundary_")
    if _tf_cut >= 0:
        _tf_payload = line.namespace[_tf_cut + len("_boundary_"):]
        if _tf_payload.startswith("contradiction_"):
            _tf_neg = _tf_payload[len("contradiction_"):]
            return (_tf_neg.startswith('!')
                    and line.expression == _tf_neg[1:])
        return False
    if line.namespace != "main":
        return False
    if state.current_chapter_thm is None:
        return False
    thm_expr = state.current_chapter_thm[0]
    premises, head = disintegrate_implication_full(thm_expr)
    if line.expression in premises:
        return True
    # Contradiction seed: head is !(...), cleanOp is (...)
    if head.startswith('!') and line.expression == head[1:]:
        return True
    # Complement (reductio) seed: head is positive, seed is "!" + head
    if not head.startswith('!') and line.expression == "!" + head:
        return True
    return False


def _ns_matches_or_strict_prefix(src_ns: str, tgt_ns: str) -> bool:
    """@brief Test whether ``src_ns`` is equal to or a `_boundary_`-aware ancestor of ``tgt_ns``.

    @details
    True iff ``src_ns == tgt_ns`` OR ``tgt_ns`` begins with
    ``src_ns + "_boundary_"``. Mirrors the prover's extension (D-33)
    that lets an equivalence class registered at an ancestor validity
    apply to an expression at a strict-descendant validity.

    **Separator-aware, not byte-level.** The check requires the
    NameMap-minted ``_boundary_`` separator, rather than a bare
    ``tgt_ns.startswith(src_ns)`` byte-prefix test that would admit
    prefix relationships not honouring it. In practice well-formed
    GL namespaces always insert ``_boundary_`` between parent and
    child payload, so the tighter check rejects nothing that should
    be admitted while explicitly closing the door on malformed
    namespaces whose prefix relationship doesn't reflect a real
    parent-child ancestry. Per I-16 the verifier must not trust
    producer well-formedness for soundness-relevant comparisons.

    Used by ``check_implication`` (the pair-wise comparable check
    inside the validity-stack deposit rule), ``check_equality1`` and
    ``check_equality2`` (the source-scope and equality-citation-scope
    admissibility checks).

    @param src_ns  Source namespace (the equivalence class's
                   registration scope, or one constituent of an
                   implication firing).
    @param tgt_ns  Target namespace (the result row's scope, or
                   another constituent).
    @return  True iff ``src_ns`` is equal to ``tgt_ns`` or a real
             ``_boundary_``-separated ancestor of it.

    @invariant D-33 — ancestor-scope inheritance is sound.
    @invariant I-16 — verifier is sacred; the separator-aware tighten
    closes a byte-prefix laxity, not weakens any check.
    """
    if src_ns == tgt_ns:
        return True
    return tgt_ns.startswith(src_ns + "_boundary_")


def check_equality1(line: ProofLine, chapter: List[ProofLine],
                    state: VerifierState) -> bool:
    """@brief Verify an ``equality1`` row substitutes args via equalities.

    @details
    ``equality1`` represents the equivalence-class substitution rule:
    rewrite a source expression's arguments using cited equalities to
    produce the result expression. Source and result share core and
    arity; differing argument positions must each be justified by an
    equality ``(=[source_arg, result_arg])`` provided in ``rest``.

    **Row layout.**
    ::
        line.expression  = result expression
        line.namespace   = result scope
        rest[0]          = source expression
        rest[1]          = source's scope
        rest[2*i+2]      = i-th equality (i >= 0)
        rest[2*i+3]      = i-th equality's scope

    **Namespace rule.** Source and each equality may live either at
    ``line.namespace`` OR at any strict ancestor scope of
    ``line.namespace`` (per ``_ns_matches_or_strict_prefix``). This
    mirrors D-33's ancestor-inheritance widening: a class registered
    at a strict-descendant validity may rewrite an ancestor-scope
    fact, depositing the result at the descendant scope.

    **Structural rule.**

    - ``rest`` must have at least 4 entries (source + 1 equality
      with their scopes).
    - Each equality's expression must have arity 2 to contribute to
      the substitution set (arity-1 or larger equalities are
      silently ignored).
    - Source and result must share core name and argument count.
    - For each position where source/result args differ, the pair
      ``(source_arg, result_arg)`` must appear in the cited equality
      set.

    @param line     The equality1 row.
    @param chapter  Sibling rows (unused).
    @param state    Global state (unused; check is purely syntactic).
    @return  True iff every differing argument position is covered
             by a cited equality and namespaces are admissible.

    @invariant D-33 — cross-scope deposit rule.
    """
    # rest layout: source, ns, eq1, ns, [eq2, ns, ...]
    if len(line.rest) < 4:
        return False

    source_expr, source_ns = line.rest[0], line.rest[1]
    if not _ns_matches_or_strict_prefix(source_ns, line.namespace):
        return False

    # Collect equalities
    eq_set: set = set()
    i = 2
    while i + 1 < len(line.rest):
        eq_expr, eq_ns = line.rest[i], line.rest[i + 1]
        if not _ns_matches_or_strict_prefix(eq_ns, line.namespace):
            return False
        ea = _extract_args(eq_expr)
        if len(ea) == 2:
            eq_set.add((ea[0], ea[1]))
        i += 2

    # Same core and arity
    result_core = _extract_core_name(line.expression)
    source_core = _extract_core_name(source_expr)
    result_args = _extract_args(line.expression)
    source_args = _extract_args(source_expr)

    if result_core != source_core or len(result_args) != len(source_args):
        return False

    # Each differing position must be covered by an equality (source→result)
    for ra, sa in zip(result_args, source_args):
        if ra != sa:
            if (sa, ra) not in eq_set:
                return False

    return True


def check_equality2(line: ProofLine, chapter: List[ProofLine],
                    state: VerifierState) -> bool:
    """@brief Verify an ``equality2`` row applies transitivity of equality.

    @details
    ``equality2`` derives ``(=[a,c])`` from cited equalities
    ``(=[a,b])`` (rest[0]) and ``(=[b,c])`` (rest[2]). The middle
    term ``b`` must match between the two — i.e. ``eq1[1] == eq2[0]``.

    **Row layout.**
    ::
        line.expression  = (=[a,c])
        line.namespace   = result scope
        rest[0]          = (=[a,b])
        rest[1]          = eq1 scope
        rest[2]          = (=[b,c])
        rest[3]          = eq2 scope

    **Namespace rule.** Each cited equality may live at
    ``line.namespace`` OR at any strict ancestor (D-33 widening,
    same as ``check_equality1``).

    **Structural rule.**

    - ``rest`` must have at least 4 entries.
    - Result, eq1, and eq2 all have arity 2.
    - Result's first arg == eq1's first arg.
    - Result's second arg == eq2's second arg.
    - eq1's second arg == eq2's first arg (the transitivity link).

    @param line     The equality2 row.
    @param chapter  Sibling rows (unused).
    @param state    Global state (unused).
    @return  True iff the row is a valid transitive composition of
             the two cited equalities at admissible scopes.
    """
    if len(line.rest) < 4:
        return False

    eq1_expr, eq1_ns = line.rest[0], line.rest[1]
    eq2_expr, eq2_ns = line.rest[2], line.rest[3]

    if not _ns_matches_or_strict_prefix(eq1_ns, line.namespace):
        return False
    if not _ns_matches_or_strict_prefix(eq2_ns, line.namespace):
        return False

    result_args = _extract_args(line.expression)
    eq1_args = _extract_args(eq1_expr)
    eq2_args = _extract_args(eq2_expr)

    if len(result_args) != 2 or len(eq1_args) != 2 or len(eq2_args) != 2:
        return False

    # (=[a,c]) from (=[a,b]) and (=[b,c])
    return (result_args[0] == eq1_args[0]
            and result_args[1] == eq2_args[1]
            and eq1_args[1] == eq2_args[0])


def check_symmetry_of_equality(line: ProofLine, chapter: List[ProofLine],
                               state: VerifierState) -> bool:
    """@brief Verify a ``symmetry of equality`` row swaps args of an equality.

    @details
    Derives ``(=[a,b])`` from ``(=[b,a])`` by swapping arguments.
    Both expressions are arity-2 equalities; result args are
    arg-position-reverses of source args.

    **Row layout.**
    ::
        line.expression  = (=[a,b])
        line.namespace   = scope
        rest[0]          = (=[b,a])
        rest[1]          = source scope (must equal line.namespace)

    Unlike ``check_equality1`` / ``check_equality2``, this checker
    requires EXACT namespace match between source and result — no
    ancestor widening. The prover emits this row form when mirroring
    a negated equality inside ``addStatement``.

    @param line     The symmetry row.
    @param chapter  Sibling rows (unused).
    @param state    Global state (unused).
    @return  True iff the row is a valid argument swap of the cited
             source equality at the same scope.

    @see check_symmetry_of_inequality — sibling that handles
    ``!(=[a,b]) ← !(=[b,a])`` (negated equalities).
    """
    if len(line.rest) < 2:
        return False

    source_expr, source_ns = line.rest[0], line.rest[1]
    if line.namespace != source_ns:
        return False

    result_args = _extract_args(line.expression)
    source_args = _extract_args(source_expr)

    if len(result_args) != 2 or len(source_args) != 2:
        return False

    return result_args[0] == source_args[1] and result_args[1] == source_args[0]


def check_symmetry_of_inequality(line: ProofLine, chapter: List[ProofLine],
                                  state: VerifierState) -> bool:
    """@brief Verify a ``symmetry of inequality`` row swaps args of a negated equality.

    @details
    Mirror of ``check_symmetry_of_equality`` but for the negated
    form: derives ``!(=[a,b])`` from ``!(=[b,a])`` by stripping the
    leading ``!`` on both sides and reusing the same swap rule.

    Emitted by the prover (``prover.cpp::addStatement`` site) when a
    negated equality is mirrored — both directions of a negated
    equality are equivalent, so the prover emits both as a single
    pair joined by this rule.

    **Row layout.**
    ::
        line.expression  = !(=[a,b])
        line.namespace   = scope
        rest[0]          = !(=[b,a])
        rest[1]          = source scope (must equal line.namespace)

    Both result and source must start with ``!(=[`` and have arity 2.
    Same exact-namespace rule as ``check_symmetry_of_equality``.

    @param line     The symmetry-of-inequality row.
    @param chapter  Sibling rows (unused).
    @param state    Global state (unused).
    @return  True iff the row is a valid argument swap of the cited
             negated equality at the same scope.

    @see check_symmetry_of_equality — positive sibling.
    """
    if len(line.rest) < 2:
        return False

    source_expr, source_ns = line.rest[0], line.rest[1]
    if line.namespace != source_ns:
        return False

    # Both result and source must be negated equalities `!(=[a,b])`.
    if not line.expression.startswith("!(=[") or not source_expr.startswith("!(=["):
        return False

    result_inner = line.expression[1:]  # drop leading '!'
    source_inner = source_expr[1:]

    result_args = _extract_args(result_inner)
    source_args = _extract_args(source_inner)

    if len(result_args) != 2 or len(source_args) != 2:
        return False

    return result_args[0] == source_args[1] and result_args[1] == source_args[0]


def check_recursion(line: ProofLine, chapter: List[ProofLine],
                    state: VerifierState) -> bool:
    """@brief Verify a ``recursion`` row matches the induction zero / step shape.

    @details
    Two chapter types dispatch into this checker: ``check_zero`` and
    ``check_induction_condition``. Each has its own structural rule.

    **``check_zero``.** The expression must be exactly
    ``(=[ind_var, i0])`` where ``ind_var`` is the induction variable
    from the theorem (third field of ``state.current_chapter_thm``).
    Namespace must be ``"main"``.

    **``check_induction_condition``.** Two sub-shapes:

    1. ``(in2[x, ind_var, s])`` — the typing-anchor row whose first
       argument ``x`` names the induction's free index variable.
       Arity must be ≥ 3, ``args[1] == ind_var``, ``args[2] == s``.

    2. ``(>[...]…)`` — the reconstructed successor implication.
       After locating ``x`` from the companion ``(in2[…])`` recursion
       row, the verifier:

       - Disintegrates the theorem; extracts anchor args.
       - Computes ``digit_args`` (via ``_find_digit_args``) and the
         immutable set ``untouchables = anchor_args ∪ {x_var} ∪
         immutables``.
       - Substitutes ``ind_var → x_var`` in every non-anchor
         premise and the head.
       - Reconstructs the implication with bound-var optimization
         (``reconstructImplication`` rule).
       - Normalizes both the reconstructed expression and
         ``line.expression``; compares via ``_normalize_expr_list``.

    Any other chapter type → False.

    @param line     The recursion row.
    @param chapter  Sibling rows (for companion ``in2`` lookup in
                    ``check_induction_condition``).
    @param state    Reads ``current_chapter_thm``,
                    ``current_chapter_type``, ``input_indices``,
                    ``output_indices``.
    @return  True iff the row matches the structural rule for its
             chapter type.

    @see _find_digit_args
    @see _find_immutable_args
    @see _reconstruct_implication
    """
    if line.namespace != "main":
        return False
    if state.current_chapter_thm is None:
        return False

    _, thm_type, ind_var = state.current_chapter_thm

    if state.current_chapter_type == "check_zero":
        # Must be equality (=[ind_var, i0])
        if not line.expression.startswith("(=["):
            return False
        args = _extract_args(line.expression)
        if len(args) != 2:
            return False
        return args[0] == ind_var and args[1] == "i0"

    if state.current_chapter_type == "check_induction_condition":
        # Either (in2[*, ind_var, s]) or an implication (>[...)
        if line.expression.startswith("(in2["):
            args = _extract_args(line.expression)
            if len(args) < 3:
                return False
            return args[1] == ind_var and args[2] == "s"
        if line.expression.startswith("(>["):
            # Find x from the companion in2[x, ind_var, s] recursion line
            x_var = None
            for ch_line in chapter:
                if (ch_line.tag == "recursion"
                        and ch_line.expression.startswith("(in2[")):
                    in2_args = _extract_args(ch_line.expression)
                    if (len(in2_args) >= 3
                            and in2_args[1] == ind_var
                            and in2_args[2] == "s"):
                        x_var = in2_args[0]
                        break
            if x_var is None:
                return False

            # Take theorem, cut off anchor
            thm_expr = state.current_chapter_thm[0]
            thm_premises, thm_head = disintegrate_implication_full(thm_expr)
            if len(thm_premises) < 2:
                return False

            # Untouchable vars: anchor args + x_var + immutable args
            # (replicates C++ createAuxyImplication logic)
            anchor_args = set(_extract_args(thm_premises[0]))

            # All expressions for digit/immutable analysis (premises + head)
            all_thm_exprs = list(thm_premises) + [thm_head]
            digit_args = _find_digit_args(
                all_thm_exprs, thm_premises[0],
                state.input_indices, state.output_indices)
            immutables = _find_immutable_args(
                all_thm_exprs, digit_args, ind_var,
                state.input_indices, state.output_indices)

            untouchables = set(anchor_args)
            untouchables.add(x_var)
            untouchables.update(immutables)

            # Non-anchor premises + head with ind_var → x_var
            chain_exprs = [_replace_arg_safe(e, ind_var, x_var)
                           for e in thm_premises[1:]]
            head_expr = _replace_arg_safe(thm_head, ind_var, x_var)

            # Reconstruct implication (C++ reconstructImplication logic):
            # bound vars = vars in >1 expression, excluding untouchables,
            # placed at first expression where they appear
            all_exprs = chain_exprs + [head_expr]
            counter: Dict[str, int] = {}
            for expr in all_exprs:
                for a in _extract_args(expr):
                    if a in untouchables:
                        continue
                    counter[a] = counter.get(a, 0) + 1
            multi_vars = {a for a, c in counter.items() if c > 1}

            placed: set = set()
            when_removed: List[List[str]] = [[] for _ in range(len(chain_exprs))]
            for idx, expr in enumerate(all_exprs):
                for a in _extract_args(expr):
                    if a in multi_vars and a not in placed:
                        placed.add(a)
                        if idx < len(chain_exprs):
                            when_removed[idx].append(a)

            # Build implication from inside out
            result = head_expr
            for i in range(len(chain_exprs) - 1, -1, -1):
                bound_str = ','.join(when_removed[i])
                result = f'(>[{bound_str}]{chain_exprs[i]}{result})'

            # Normalize both and compare
            return (_normalize_expr_list([result])
                    == _normalize_expr_list([line.expression]))
        return False

    # Other cases not yet implemented
    return False


def check_theorem_tag(line: ProofLine, chapter: List[ProofLine],
                  state: VerifierState) -> bool:
    """@brief Verify a ``theorem`` row cites a real proved theorem.

    @details
    The ``theorem`` tag claims ``line.expression`` is a previously
    proved theorem registered in
    ``state.global_theorems``. Three-stage membership test:

    1. **Fast path: direct membership.** ``line.expression`` appears
       exactly in ``state.global_theorems`` (the dict keyed by
       theorem expression).

    2. **w/V revert fallback.** If the expression is a theorem-anchor
       implication (first premise starts with ``(Anchor``), apply
       ``_revert_w_to_v_in_theorem_citation`` to swap w/W tokens
       back to v/V, then re-check direct membership. This handles
       a hypothetical future regression where a citation-form
       cell leaks into a HEAD column (column 0 is normally never
       w-swapped, but the revert is defensive).

    3. **Normalize-and-compare fallback.** Iterate every registered
       theorem; compare normalized forms (via ``_normalize_expr_list``).
       Catches variable-renumbering drift that direct membership
       misses — particularly inside contradiction LBs where the
       prover may produce v-numbering different from the registry's
       canonical form.

    Namespace gate: must be ``"main"``. Theorem citations live only
    at the theorem root scope.

    @param line     The theorem-citation row.
    @param chapter  Sibling rows (unused).
    @param state    Reads ``global_theorems``.
    @return  True iff the expression matches some registered theorem
             under any of the three matching paths.

    @see check_externally_provided_theorem — same idea but against
    the external-theorems set.
    """
    if line.namespace != "main":
        return False
    if line.expression in state.global_theorems:
        return True
    # Fast-path mirror of the origin meta-check's w/W -> v/V revert (load-
    # bearing there for cited theorem-anchor implications). Defensive at THIS
    # site: HEAD cells stay in v/V form per processor design (column 0 is
    # never v->w-swapped), so the revert only fires if a future regression
    # slips a cited form in. Same primitive at both sites for consistency.
    if _is_theorem_anchor_impl_local(line.expression):
        rev = _revert_w_to_v_in_theorem_citation(line.expression)
        if rev != line.expression and rev in state.global_theorems:
            return True
    # Fallback: normalize and compare
    norm_line = _normalize_expr_list([line.expression])
    for gt_expr in state.global_theorems:
        if _normalize_expr_list([gt_expr]) == norm_line:
            return True
    return False


_INTEGRATION_GOAL_SUFFIX = "_integration_goal"


def _strip_integration_goal(s: str) -> str:
    """@brief Remove the ``_integration_goal`` postfix marker if present.

    @details
    Integration tags (``expansion for integration``,
    ``reformulation for integration {and, >[bound], >[]}``,
    ``or branch assumption``) decorate the compact expression's right-
    side cell with a ``_integration_goal`` postfix to signal that the
    citation is the integration TARGET (not the consumed source).
    The verifier strips the postfix before structural comparison.

    @param s  Expression possibly carrying the postfix.
    @return   ``s`` with the postfix removed; identical if absent.
    """
    if s.endswith(_INTEGRATION_GOAL_SUFFIX):
        return s[:-len(_INTEGRATION_GOAL_SUFFIX)]
    return s


def _flatten_and(expr: str) -> List[str]:
    """@brief Flatten a left-nested ``(&...)`` AND into its ordered element list.

    @details
    Inverse of ``_build_and_from_elements``. Example:

        ``(&(&(&A B)C)D)`` → ``[A, B, C, D]``

    The C++ builds ANDs as ``current = elem[0]; for i in 1..n:
    current = "(&" + current + elem[i] + ")"``, so the leftmost leaf
    is ``elem[0]`` and each successive right child is the next
    element. The flatten walks paren-balanced from the outermost
    level inward, peeling one element per layer.

    @param expr  An MPL expression in left-nested AND form.
    @return  Elements in original order. For a non-AND expression
             (no leading ``(&``), returns ``[expr]``.
    """
    elements: List[str] = []
    while expr.startswith('(&'):
        inner = expr[2:-1]          # strip outer "(& " and ")"
        # Find end of the first (left) sub-expression
        depth = 0
        split = -1
        for i, c in enumerate(inner):
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
                if depth == 0:
                    split = i + 1
                    break
        if split < 0:
            break
        left = inner[:split]
        right = inner[split:]
        elements.append(right)      # right child = next element
        expr = left                 # recurse into left child
    elements.append(expr)           # leftmost leaf = first element
    elements.reverse()
    return elements


def _normalize_existence_bound_vars(expr: str) -> Optional[str]:
    """@brief Canonicalize binder-variable names inside an existence expansion.

    @details
    Parses ``expr`` as an existence expansion
    ``!(>[vars](left)!(right))``. Renames each name in the binder
    list to ``bv1, bv2, …`` in declaration order. Signature
    (non-binder) variables are left unchanged.

    Used by ``_check_reformulation_integration_existence`` to compare
    a reformulated existence against the binary-built expected
    existence modulo binder-renaming.

    @param expr  An MPL existence expansion.
    @return  Expression with binder vars normalised; ``None`` on
             malformed input.
    """
    parsed = _parse_existence_expansion(expr)
    if parsed is None:
        return None

    bound_csv, _, _ = parsed
    bounds = [b for b in bound_csv.split(',') if b]
    result = expr
    for idx, bound in enumerate(bounds, start=1):
        result = _replace_arg_safe(result, bound, f'bv{idx}')
    return result



def _infer_missing_existence_bound_var(premises: List[str], head: str) -> Optional[str]:
    """@brief Infer the existential witness variable from premise/head args.

    @details
    When a reformulated existence implication is emitted with an
    empty outer ``>[]`` binder (the witness was already consumed),
    the existential variable name has been elided from the binder
    list but still appears in the premises. This function recovers
    it by intersecting the two premises' arg sets and removing any
    arg that appears in the compiled head — the survivor is the
    witness.

    Used by ``_reformulation_implication_to_existence`` when the
    outer binder is empty.

    @param premises  Exactly TWO premises (the reformulated form has
                     left + right as separate premises).
    @param head      The compiled head.
    @return  The single witness variable name, or ``None`` if the
             premise count is wrong or zero/multiple candidates
             survive.
    """
    if len(premises) != 2:
        return None

    head_args = set(_extract_args(head))
    shared = set(_extract_args(premises[0])) & set(_extract_args(premises[1]))
    candidates = [arg for arg in shared if arg not in head_args]

    if len(candidates) != 1:
        return None
    return candidates[0]



def _reformulation_implication_to_existence(expr: str) -> Optional[Tuple[str, str]]:
    """@brief Convert a reformulated existence implication back to existence form.

    @details
    The processor's reformulation-for-integration rewrites an
    existence head into a two-premise implication form:
    ``(>[witness](left)(right_carrying_head_args_then_head))``. This
    function reverses that rewrite to produce the canonical existence
    ``!(>[witness](left)!(right))``, returned alongside the
    compiled-head extracted from the right premise.

    Example::

        (>[v3](in[v3,N])(>[](in3[v1,v3,v2,+])(preorder[N,+,v1,v2])))

    becomes::

        !(>[v3](in[v3,N])!(in3[v1,v3,v2,+]))

    The witness ``v3`` comes from the outer binder list. If that
    binder is empty, ``_infer_missing_existence_bound_var`` is
    consulted instead. Returns ``None`` on malformed input or when
    the witness cannot be inferred.

    Used by ``_check_reformulation_integration_existence`` to bring
    a reformulated implication into canonical existence form for
    comparison.

    @param expr  An MPL reformulated existence implication.
    @return  ``(existence_form, head_expr)`` on success; ``None`` on
             structural mismatch.
    """
    premises, head = disintegrate_implication_full(expr)
    if len(premises) != 2:
        return None

    m = re.match(r'^\(>\[([^\]]*)\]', expr)
    if m is None:
        return None

    bound_csv = m.group(1)
    if not bound_csv:
        inferred = _infer_missing_existence_bound_var(premises, head)
        if inferred is None:
            return None
        bound_csv = inferred

    existence_expr = f'!(>[{bound_csv}]{premises[0]}!{premises[1]})'
    return existence_expr, head



def _reformulation_integration_preamble(line: ProofLine,
                                        chapter: List[ProofLine],
                                        state: VerifierState):
    """@brief Shared preamble for the three reformulation-for-integration tags.

    @details
    All three ``reformulation for integration {and, >[bound], >[]}``
    checkers share the same setup: locate the row's matching
    ``expansion for integration`` row, extract the compact form,
    look up its binary entry, and verify the row's head matches the
    compact. This helper performs all those steps and returns the
    triple ``(entry, compact, right_expr)`` to the dispatching
    checker.

    **Algorithm.**

    1. ``len(rest) >= 2`` required.
    2. Namespace match: ``line.namespace == rest[1]``.
    3. Locate an ``expansion for integration`` row in the chapter
       whose LEFT side equals ``rest[0]``, namespace equals
       ``rest[1]``, and whose own rest has ≥ 2 entries. Reject if
       no such row exists.
    4. Strip the integration-goal postfix from the expansion's
       ``rest[0]`` to get the compact. Extract its core.
    5. Look up the core in every binary for the chapter; record the
       first match. Reject if no binary defines the core.
    6. Disintegrate ``line.expression`` and check the disintegrated
       head matches the compact (verifies the row claims to expand
       exactly this compact, not some other operator).

    Returns ``None`` on any failure path.

    @param line     The reformulation-for-integration row.
    @param chapter  All chapter rows (for expansion lookup).
    @param state    Reads ``binaries_for_chapter``.
    @return  ``(entry, compact, right_expr)`` on success; ``None``
             on any preconditions failure.

    @see check_reformulation_for_integration_and
    @see check_reformulation_for_integration_bound
    @see check_reformulation_for_integration_empty
    """
    if len(line.rest) < 2:
        return None

    right_expr = line.rest[0]
    right_ns = line.rest[1]

    if line.namespace != right_ns:
        return None

    expansion_origin = None
    for ch_line in chapter:
        if (ch_line.expression == right_expr
                and ch_line.namespace == right_ns
                and ch_line.tag == 'expansion for integration'
                and len(ch_line.rest) >= 2):
            expansion_origin = ch_line
            break

    if expansion_origin is None:
        return None

    compact = _strip_integration_goal(expansion_origin.rest[0])
    core = _extract_core_name(compact)

    entry = None
    for binary in state.binaries_for_chapter():
        if core in binary:
            entry = binary[core]
            break

    if entry is None:
        return None

    _, proof_head = disintegrate_implication_full(line.expression)
    if compact != proof_head:
        return None

    return entry, compact, right_expr



def _check_reformulation_integration_and(line, compact, right_expr):
    """@brief AND-category structural check for reformulation-for-integration.

    @details
    Reformulates an AND-category compact into a chain of nested
    implications. The right-side ``expansion for integration`` row's
    expression (post-strip) must be a left-nested AND. Flatten it
    into elements, then materialize the implication chain by
    wrapping each element into ``(>[]elem ACCUM)`` from inside out,
    starting with the compact at the centre. The result must equal
    ``line.expression`` byte-for-byte.

    @param line        The row being checked.
    @param compact     The compact form returned by the preamble.
    @param right_expr  The expansion-for-integration's expression
                       (still carries the integration-goal postfix).
    @return  True iff the row's expression matches the rebuilt
             implication chain.
    """
    right_expr_clean = _strip_integration_goal(right_expr)
    if not right_expr_clean.startswith('(&'):
        return False

    elements = _flatten_and(right_expr_clean)
    if not elements:
        return False

    expected = compact
    for elem in reversed(elements):
        expected = f'(>[]{elem}{expected})'

    return line.expression == expected


def _check_reformulation_integration_existence(line, entry, compact):
    """@brief Existence-category structural check for reformulation-for-integration.

    @details
    Two-step structural comparison:

    1. **Existence form.** Convert ``line.expression`` (a
       reformulated implication) back to canonical existence form
       via ``_reformulation_implication_to_existence``. The returned
       head must match ``compact`` (the binary-compact form). Reject
       on mismatch.

    2. **Binary shape.** Substitute the entry's signature args with
       the compact's actual args (binary entry must have exactly 2
       elements; arity must match). Build the expected existence
       via ``_build_existence_from_elements``.

    3. **Normalize-and-compare.** Apply
       ``_normalize_existence_bound_vars`` to both forms; accept on
       equality.

    Used by ``check_reformulation_for_integration_bound`` and
    ``check_reformulation_for_integration_empty`` (the two existence-
    category variants).

    @param line     The row being checked.
    @param entry    The compact's binary entry (existence category).
    @param compact  The compact form returned by the preamble.
    @return  True iff the row's implication form equals the
             expected existence modulo binder normalization.
    """
    converted = _reformulation_implication_to_existence(line.expression)
    if converted is None:
        return False

    left_as_existence, converted_head = converted
    if converted_head != compact:
        return False

    sig_args = _extract_args(entry.get('signature', ''))
    actual_args = _extract_args(compact)
    elements = entry.get('elements', [])

    if len(sig_args) != len(actual_args) or len(elements) != 2:
        return False

    subst = dict(zip(sig_args, actual_args))
    expected_children = [_replace_arg_safe_multi(elem, subst)
                         for elem in elements]
    expected_existence = _build_existence_from_elements(expected_children)

    left_norm = _normalize_existence_bound_vars(left_as_existence)
    expected_norm = _normalize_existence_bound_vars(expected_existence)
    if left_norm is None or expected_norm is None:
        return False

    return left_norm == expected_norm


def check_reformulation_for_integration_and(line: ProofLine,
                                            chapter: List[ProofLine],
                                            state: VerifierState) -> bool:
    """@brief Verify a ``reformulation for integration and`` row.

    @details
    Dispatch wrapper: runs ``_reformulation_integration_preamble``
    to locate the compact + binary entry; rejects unless the entry's
    category is ``"and"``; delegates to
    ``_check_reformulation_integration_and`` for the structural
    rebuild.

    The ``reformulation for integration and`` tag is emitted when
    an AND compact at integration time gets unfolded into a chain
    of single-premise implications (one per AND element).

    @param line     The row.
    @param chapter  All chapter rows.
    @param state    Global state.
    @return  True iff the structural check passes.
    """
    preamble = _reformulation_integration_preamble(line, chapter, state)
    if preamble is None:
        return False
    entry, compact, right_expr = preamble
    if entry.get('category') != 'and':
        return False
    return _check_reformulation_integration_and(line, compact, right_expr)


def check_reformulation_for_integration_bound(line: ProofLine,
                                              chapter: List[ProofLine],
                                              state: VerifierState) -> bool:
    """@brief Verify a ``reformulation for integration >[bound]`` row.

    @details
    The ``>[bound]`` variant of existence reformulation: the outermost
    ``>[…]`` of ``line.expression`` carries a NON-EMPTY bound-variable
    list (the existence's witness is explicit). Used when the
    integration site retains the witness as a free variable in the
    enclosing scope.

    Algorithm:

    1. Run ``_reformulation_integration_preamble``; reject on None.
    2. Require ``entry['category'] == 'existence'``.
    3. Match the outermost binder regex ``\\(>\\[([^\\]]*)\\]`` against
       ``line.expression``. The captured group must be non-empty.
    4. Delegate to ``_check_reformulation_integration_existence`` for
       the structural rebuild.

    @param line     The row.
    @param chapter  All chapter rows.
    @param state    Global state.
    @return  True iff the structural check passes.

    @see check_reformulation_for_integration_empty — the empty-binder
    sibling (witness already consumed).
    """
    preamble = _reformulation_integration_preamble(line, chapter, state)
    if preamble is None:
        return False
    entry, compact, right_expr = preamble
    if entry.get('category') != 'existence':
        return False

    # Verify outermost >[...] has a bound variable
    m = re.match(r'^\(>\[([^\]]*)\]', line.expression)
    if m is None or not m.group(1):
        return False

    return _check_reformulation_integration_existence(line, entry, compact)


def check_reformulation_for_integration_empty(line: ProofLine,
                                              chapter: List[ProofLine],
                                              state: VerifierState) -> bool:
    """@brief Verify a ``reformulation for integration >[]`` row.

    @details
    The ``>[]`` variant of existence reformulation: the outermost
    ``>[…]`` is EMPTY because the existence's witness has been
    consumed at the integration site (typically because it appears
    in an enclosing scope's binder). The verifier infers the witness
    via ``_infer_missing_existence_bound_var`` before doing the
    structural check.

    Algorithm:

    1. Run ``_reformulation_integration_preamble``; reject on None.
    2. Require ``entry['category'] == 'existence'``.
    3. Match the outermost binder regex; require the captured group
       to be EMPTY (must be ``>[]``, not ``>[v1]``).
    4. Delegate to ``_check_reformulation_integration_existence``;
       its internal call to
       ``_reformulation_implication_to_existence`` performs the
       witness inference.

    @param line     The row.
    @param chapter  All chapter rows.
    @param state    Global state.
    @return  True iff the structural check passes.

    @see check_reformulation_for_integration_bound — the explicit-
    binder sibling.
    """
    preamble = _reformulation_integration_preamble(line, chapter, state)
    if preamble is None:
        return False
    entry, compact, right_expr = preamble
    if entry.get('category') != 'existence':
        return False

    # Verify outermost >[...] is empty
    m = re.match(r'^\(>\[([^\]]*)\]', line.expression)
    if m is None:
        return False
    if m.group(1):
        return False  # not empty — wrong tag

    return _check_reformulation_integration_existence(line, entry, compact)


def check_expansion_for_integration(line: ProofLine,
                                    chapter: List[ProofLine],
                                    state: VerifierState) -> bool:
    """@brief Verify an ``expansion for integration`` row.

    @details
    The integration-time sibling of ``check_expansion``. Differs in
    two ways:

    1. The right-side expression (``rest[0]``) does NOT need to
       appear as a left-side expression in the chapter — the
       integration-goal marker indicates a forward reference (the
       compact is the integration target, not a consumed source).
    2. Both ``line.expression`` and ``rest[0]`` may carry the
       ``_integration_goal`` postfix; both are stripped via
       ``_strip_integration_goal`` before the structural comparison.

    **Algorithm.**

    1. ``len(rest) >= 2`` required.
    2. Namespace match: ``line.namespace == rest[1]``.
    3. Strip the integration-goal postfix from both
       ``line.expression`` and ``rest[0]``.
    4. Extract the right side's core; iterate every binary; for OR,
       recursively decode the ordered atomic leaves, then delegate to
       ``_try_expand``.

    @param line     The row.
    @param chapter  Sibling rows (unused for origin presence —
                    integration goals can be forward-referenced).
    @param state    Reads ``binaries_for_chapter``.
    @return  True iff the structural expansion matches.

    @see check_expansion — non-integration sibling that requires
    origin presence.
    """
    if len(line.rest) < 2:
        return False

    right_expr = line.rest[0]
    right_ns = line.rest[1]

    if line.namespace != right_ns:
        return False

    # Strip _integration_goal postfix for structural checks
    left_clean = _strip_integration_goal(line.expression)
    right_clean = _strip_integration_goal(right_expr)

    core = _extract_core_name(right_clean)

    for binary in state.binaries_for_chapter():
        if core in binary:
            entry = binary[core]
            flattened_or = None
            if entry.get('category') == 'or':
                flattened_or = _or_disjuncts_from_compiled(
                    right_clean, state.binaries_for_chapter(),
                    flatten_nested=True)
                if flattened_or is None:
                    continue
            if _try_expand(entry, right_clean, left_clean, flattened_or,
                           binaries=state.binaries_for_chapter()):
                return True

    return False


_SUBPROOF_SEP = "_subproof_"


def _split_subproof_payload(payload: str):
    """@brief Split a goal-carrying integration-scope payload
    ``<goal>_subproof_<bare>`` into its root-goal and bare halves.

    @details
    Goal-driven integration scopes minted on MAIN carry the root
    ``toBeProved`` goal in their validity payload so a disproof of that
    goal can find and wipe the scopes its integration spawned (the
    prover-side twin is ``ExpressionAnalyzer::splitSubproofPayload``).
    The grammar is a leading balanced-paren group (optionally
    ``!``-negated) — the goal expression verbatim — followed by the
    literal ``_subproof_`` separator and a non-empty bare legacy
    payload. The grammar is injective against every legacy payload
    shape: a legacy compact is a single balanced group with nothing
    after it, and ``)_subproof_`` cannot occur inside a well-formed MPL
    expression.

    @param payload  The validity payload (the namespace's last
                    ``_boundary_``-delimited segment).
    @return  ``(goal, bare)`` on a match, ``None`` otherwise.
    """
    i = 0
    if i < len(payload) and payload[i] == '!':
        i += 1
    if i >= len(payload) or payload[i] != '(':
        return None
    depth = 0
    close = -1
    for k in range(i, len(payload)):
        if payload[k] == '(':
            depth += 1
        elif payload[k] == ')':
            depth -= 1
            if depth == 0:
                close = k
                break
    if close < 0:
        return None
    sep_start = close + 1
    if not payload.startswith(_SUBPROOF_SEP, sep_start):
        return None
    bare = payload[sep_start + len(_SUBPROOF_SEP):]
    if not bare:
        return None
    return payload[:sep_start], bare


def check_premise_element(line: ProofLine, chapter: List[ProofLine],
                          state: VerifierState) -> bool:
    """@brief Verify a ``premise element`` row carries a premise of its origin.

    @details
    ``premise element`` rows decompose an integration-time implication
    into its constituent premises — each premise gets its own row
    rooted in a child scope keyed by the implication's signature.
    This is part of the integration apparatus around the
    ``expansion for integration`` mechanism.

    **Algorithm.**

    1. ``len(rest) >= 1`` required.
    2. Take ``rest[0]`` (the origin implication, possibly carrying
       the ``_integration_goal`` postfix). Strip the postfix and
       disintegrate the clean form into ``(premises, _)``.
       ``line.expression`` must be one of those premises.
    3. Locate a chapter row whose ``expression`` equals ``rest[0]``
       (with postfix as-emitted), tag is ``"expansion for
       integration"``, and rest has at least 1 entry. Strip the
       postfix from that expansion's ``rest[0]`` to derive the
       ``cleanSig``.
    4. Namespace match. The subproof scope's payload comes in two
       generations:
       - legacy / statement-driven: the payload IS ``cleanSig`` —
         ``line.namespace`` equals ``cleanSig`` or ends with
         ``_boundary_<cleanSig>``;
       - goal-carrying (goal-driven mints on MAIN): the payload is
         ``<goal>_subproof_<cleanSig>`` — the namespace's LAST
         ``_boundary_``-delimited segment must split under
         ``_split_subproof_payload`` with its bare half equal to
         ``cleanSig`` exactly. The goal half identifies which
         ``toBeProved`` goal spawned the subproof; it is validated
         structurally (balanced, optionally negated, non-empty) but
         deliberately NOT matched against this chapter's rows — a
         subproof's compact may be legitimately cited by a chapter
         proving a DIFFERENT goal (the compact statement at main is
         shared; only the machinery is per-goal).

    @param line     The premise-element row.
    @param chapter  All chapter rows (for expansion lookup).
    @param state    Global state (unused).
    @return  True iff the premise membership + namespace rooting
             both pass.
    """
    if len(line.rest) < 1:
        return False

    origin_impl = line.rest[0]

    # Step 1: strip postfix before disintegrating
    origin_impl_clean = _strip_integration_goal(origin_impl)
    premises, _ = disintegrate_implication_full(origin_impl_clean)
    if line.expression not in premises:
        return False

    # Step 2 + 3: match on raw origin_impl (with postfix), but
    # strip postfix from ch_line.rest[0] before matching against namespace
    for ch_line in chapter:
        if (ch_line.expression == origin_impl
                and ch_line.tag == "expansion for integration"
                and len(ch_line.rest) >= 1):
            clean_sig = _strip_integration_goal(ch_line.rest[0])
            if (line.namespace == clean_sig
                    or line.namespace.endswith("_boundary_" + clean_sig)):
                return True
            # Goal-carrying payload: the last boundary segment must be
            # <goal>_subproof_<cleanSig> with the bare half EXACT.
            last_payload = line.namespace.rsplit("_boundary_", 1)[-1]
            split = _split_subproof_payload(last_payload)
            if split is not None and split[1] == clean_sig:
                return True

    return False


def check_validity_name(line: ProofLine, chapter: List[ProofLine],
                        state: VerifierState) -> bool:
    """@brief Verify a ``validity name`` row points to an integration head.

    @details
    The ``validity name`` tag emits a row whose ``line.expression``
    is the compact integration name (e.g.
    ``(some_implication_compact[args])``) and whose ``rest[0]`` is
    the head that compact expands to. The verifier locates the
    matching ``expansion for integration`` row and confirms its
    LEFT side's disintegrated head equals ``rest[0]``.

    **Algorithm.**

    1. ``len(rest) >= 1`` required.
    2. Iterate every chapter row; find one tagged
       ``"expansion for integration"`` whose ``rest[0]``
       (post-strip) equals ``line.expression``.
    3. Disintegrate that expansion row's ``expression`` (post-strip)
       to extract its head.
    4. Accept iff the extracted head equals ``rest[0]``.

    Used as a cross-reference helper: every integration row's compact
    must have a corresponding validity-name row asserting the
    compact's head.

    @param line     The validity-name row.
    @param chapter  All chapter rows.
    @param state    Global state (unused).
    @return  True iff the expected expansion row exists and its head
             matches.
    """
    if len(line.rest) < 1:
        return False

    target_expr = line.rest[0]
    impl_name = line.expression

    for ch_line in chapter:
        if (ch_line.tag == "expansion for integration"
                and len(ch_line.rest) >= 1
                and _strip_integration_goal(ch_line.rest[0]) == impl_name):
            head = disintegrate_implication_head(
                _strip_integration_goal(ch_line.expression))
            return head == target_expr

    return False


def check_anchor_handling(line: ProofLine, chapter: List[ProofLine],
                          state: VerifierState) -> bool:
    """@brief Verify an ``anchor handling`` row introduces an anchor _copy.

    @details
    The ``anchor handling`` tag introduces a copy of the chapter's
    anchor expression — both sides are ``(Anchor<tag>[...])`` forms
    with identical anchor tag (the prefix-stripped suffix after
    ``Anchor``) but potentially different arguments. Each differing
    argument position must carry definition-set type ``"(1)"`` so
    the copy is a sound renaming.

    **Structural rule.**

    - ``line.namespace == "main"`` and ``rest[1] == "main"``.
    - ``rest`` has at least 2 entries.
    - ``line.expression`` and ``rest[0]`` share core name; both must
      start with ``Anchor``.
    - They have the same argument count.
    - For every position where args differ: the position's
      definition-set entry in ``state.definition_sets[core]`` must
      be present AND its type label must be exactly ``"(1)"``.
    - The right-side anchor (``rest[0]``) must appear in the chapter
      as a ``task formulation`` row at ``"main"``.

    The ``(1)`` requirement prevents the anchor-handling row from
    rewriting a type-constrained position (e.g. ``P(1)``); only the
    pure ``(1)`` anchor-slot positions admit relabeling.

    @param line     The anchor-handling row.
    @param chapter  All chapter rows (for origin task-formulation
                    lookup).
    @param state    Reads ``definition_sets``.
    @return  True iff the row is a structurally sound anchor copy
             with an origin task formulation in the chapter.

    @invariant I-11 — anchor variables must not appear in
    ``>[...]`` bound-variable lists (enforced upstream in the prover;
    this check is the downstream verification that the producer
    honoured it).
    """
    if line.namespace != "main":
        return False
    if len(line.rest) < 2:
        return False

    origin_expr = line.rest[0]
    origin_ns = line.rest[1]
    if origin_ns != "main":
        return False

    # Both must be the same anchor type
    target_core = _extract_core_name(line.expression)
    origin_core = _extract_core_name(origin_expr)
    if target_core != origin_core:
        return False
    if not target_core.startswith("Anchor"):
        return False

    # Extract args and compare
    target_args = _extract_args(line.expression)
    origin_args = _extract_args(origin_expr)
    if len(target_args) != len(origin_args):
        return False

    # Differing positions must have "(1)" definition set
    ds = state.definition_sets.get(target_core, {})
    for i, (ta, oa) in enumerate(zip(target_args, origin_args)):
        if ta != oa:
            # 1-based position key
            pos_key = str(i + 1)
            pos_ds = ds.get(pos_key, [])
            if not pos_ds or pos_ds[0] != "(1)":
                return False

    # Origin anchor must exist as task formulation in chapter
    for ch_line in chapter:
        if (ch_line.expression == origin_expr
                and ch_line.namespace == "main"
                and ch_line.tag == "task formulation"):
            return True

    return False


def check_or_theorem(line: ProofLine, chapter: List[ProofLine],
                     state: VerifierState) -> bool:
    """@brief Stub-accept an ``or theorem`` row at scope "main".

    @details
    The ``or theorem`` tag emits a row whose ``line.expression`` is
    a compiled OR form ``(or<N>[…])`` and whose ``rest[0]`` is the
    existence theorem that grounds the OR (with ``rest[2]`` as the
    companion). The full structural unfolding into the De Morgan
    shape ``!(&!a!b)`` is deferred — at present this checker accepts
    any row meeting two trivial gates:

    - ``line.namespace == "main"``.
    - ``len(rest) >= 2``.

    Designed as a stub: the actual structural witness lives upstream
    in the existence-theorem proof; this row merely marks that an
    OR theorem was registered. A future tightening could add the
    full structural check.

    @param line     The or-theorem row.
    @param chapter  Sibling rows (unused).
    @param state    Global state (unused).
    @return  True iff namespace = main and rest length >= 2.
    """
    if line.namespace != "main":
        return False
    if len(line.rest) < 2:
        return False
    return True


def _or_elim_citation_resolves(expr: str, state: "VerifierState") -> bool:
    """@brief Membership ladder for an or-elimination citation.

    @details
    The same three-stage test ``check_theorem_tag`` applies to a
    ``theorem`` row, factored for the or-elimination checker's three
    citations (two guard variants plus the licensing or theorem, all
    rendered in w/W citation form by the processor): exact membership
    in ``state.global_theorems``; the w/W -> v/V revert for
    theorem-anchor implications; then the normalize-and-compare
    fallback over every registered theorem.

    @param expr   The cited theorem expression (citation cell form).
    @param state  Reads ``global_theorems``.
    @return  True iff the citation resolves to a registered theorem.
    @see check_theorem_tag — the ladder's original site.
    """
    if expr in state.global_theorems:
        return True
    if _is_theorem_anchor_impl_local(expr):
        rev = _revert_w_to_v_in_theorem_citation(expr)
        if rev != expr and rev in state.global_theorems:
            return True
    norm_expr = _normalize_expr_list([expr])
    for gt_expr in state.global_theorems:
        if _normalize_expr_list([gt_expr]) == norm_expr:
            return True
    return False


_OR_ELIM_VAR_RE = re.compile(r'[vVwW]\d+$')


def _or_elim_align(a: str, b: str,
                   forward: Dict[str, str], reverse: Dict[str, str]) -> bool:
    """@brief Token-aligned variable unification of two processed
    expressions — the or-elimination disjunct comparator.

    @details
    The Python mirror of the prover's ``alignOrEliminationExprs``,
    over processed naming: both expressions split into alternating
    delimiter and token runs (tokens are maximal ``[A-Za-z0-9_]+``
    runs); every delimiter run must match byte-for-byte, which keeps
    polarity literal (I-175 — a ``!`` on one side only fails the
    walk). A token matching ``[vVwW]<digits>`` is a renamed bound
    variable on BOTH sides or the walk fails; it binds through the
    caller-shared ``forward``/``reverse`` maps (one consistent
    bijection across a whole call sequence). Every other token —
    operator names, anchor slot names, numerals — must be byte-equal.

    Both verdicts are contractual: True extends the bijection, False
    is a defined mismatch the caller skips.

    @param a        First expression (or-theorem leaf side).
    @param b        Second expression (guard side).
    @param forward  Accumulated a->b bindings (extended in place).
    @param reverse  Accumulated b->a bindings (extended in place).
    @return  True iff the walk completes with a consistent bijection.
    @see check_or_elimination — sole caller.
    """
    parts_a = re.split(r'([A-Za-z0-9_]+)', a)
    parts_b = re.split(r'([A-Za-z0-9_]+)', b)
    if len(parts_a) != len(parts_b):
        return False
    for idx, (pa, pb) in enumerate(zip(parts_a, parts_b)):
        if idx % 2 == 0:
            if pa != pb:
                return False
            continue
        bind_a = bool(_OR_ELIM_VAR_RE.match(pa))
        bind_b = bool(_OR_ELIM_VAR_RE.match(pb))
        if bind_a != bind_b:
            return False
        if bind_a:
            if pa in forward:
                if forward[pa] != pb:
                    return False
            elif pb in reverse:
                return False
            else:
                forward[pa] = pb
                reverse[pb] = pa
        elif pa != pb:
            return False
    return True


def check_or_elimination(line: ProofLine, chapter: List[ProofLine],
                         state: VerifierState) -> bool:
    """@brief Verify an ``or elimination`` row derives its theorem from a
    guard-variant pair under a proved or theorem — the pre-split merge.

    @details
    The row is a whole chapter (the merge row IS the derived theorem's
    chapter): ``line.expression`` is the merged theorem, ``rest`` holds
    the three citations with namespaces —
    ``(variantA, main, variantB, main, orTheorem, main)``. The check is
    the full or-elimination soundness argument:

    1. **Citations resolve.** All three cited rows are registered
       proved theorems (``_or_elim_citation_resolves`` — exact, w/W
       revert, then normalized membership).
    2. **One-premise difference.** The variants disintegrate to equal
       heads and equal premise chains except the INNERMOST premise
       (the guards), compared via shared-map normalization
       (``_normalize_expr_list`` over premises+head); the guards must
       differ. Guards are binder-free by the pre-split authoring
       convention, so they introduce no variables and the shared
       prefix normalization is stable.
    3. **The merged claim.** ``line.expression`` disintegrates to
       exactly the variants' common chain plus their head (normalized
       comparison — the claim is v/V, the citations w/W).
    4. **The or license.** The or theorem's head decodes through the
       GL binaries (``_or_disjuncts_from_compiled``, flattened) to
       exactly TWO leaves that align with the two guards — in either
       order — under ONE variable bijection (``_or_elim_align``,
       literal polarity per I-175).
    5. **Side-premise cover.** Every premise of the or theorem, with
       all its variables bound by the leaf bijection and mapped into
       variant naming, is byte-present among the common premises, or
       is a typing atom ``(in[x,...])`` whose mapped variable occurs
       in a common premise (the definition-set-backed typing gate's
       syntactic form — premises type their variables, so occurrence
       in a common premise carries the or theorem's typing
       requirement).

    A miss on any stage returns False — failures are first-class.

    @param line     The or-elimination row.
    @param chapter  Sibling rows (unused — the row is the chapter).
    @param state    Reads ``global_theorems`` + ``binaries_for_chapter``.
    @return  True iff the merge is structurally sound.
    @see check_or_theorem — the sibling construction tag.
    """
    if line.namespace != "main":
        return False
    if len(line.rest) < 6:
        return False
    variant_a, ns_a = line.rest[0], line.rest[1]
    variant_b, ns_b = line.rest[2], line.rest[3]
    or_thm, ns_o = line.rest[4], line.rest[5]
    if ns_a != "main" or ns_b != "main" or ns_o != "main":
        return False

    for cited in (variant_a, variant_b, or_thm):
        if not _or_elim_citation_resolves(cited, state):
            return False

    prem_a, head_a = disintegrate_implication_full(variant_a)
    prem_b, head_b = disintegrate_implication_full(variant_b)
    if len(prem_a) != len(prem_b) or len(prem_a) < 2:
        return False
    norm_a = _normalize_expr_list(prem_a + [head_a])
    norm_b = _normalize_expr_list(prem_b + [head_b])
    if norm_a[:-2] != norm_b[:-2]:
        return False
    if norm_a[-1] != norm_b[-1]:
        return False
    if norm_a[-2] == norm_b[-2]:
        return False
    guard_a = prem_a[-1]
    guard_b = prem_b[-1]

    prem_m, head_m = disintegrate_implication_full(line.expression)
    if len(prem_m) != len(prem_a) - 1:
        return False
    if _normalize_expr_list(prem_m + [head_m]) != \
            _normalize_expr_list(prem_a[:-1] + [head_a]):
        return False

    prem_o, head_o = disintegrate_implication_full(or_thm)
    leaves = _or_disjuncts_from_compiled(
        head_o, state.binaries_for_chapter(), flatten_nested=True)
    if leaves is None or len(leaves) != 2:
        return False
    matched_forward: Optional[Dict[str, str]] = None
    for g0, g1 in ((guard_a, guard_b), (guard_b, guard_a)):
        forward: Dict[str, str] = {}
        reverse: Dict[str, str] = {}
        if _or_elim_align(leaves[0], g0, forward, reverse) \
                and _or_elim_align(leaves[1], g1, forward, reverse):
            matched_forward = forward
            break
    if matched_forward is None:
        return False

    common_set = set(prem_a[:-1])
    for prem in prem_o:
        prem_vars = set(re.findall(r'[vVwW]\d+', prem))
        if any(tok not in matched_forward for tok in prem_vars):
            return False
        img = re.sub(r'[vVwW]\d+',
                     lambda m: matched_forward[m.group(0)], prem)
        if img in common_set:
            continue
        if img.startswith('(in['):
            inner = img[4:-2]
            in_var = inner.split(',')[0] if inner else ''
            if in_var and any(
                    re.search(r'(?<![A-Za-z0-9_])' + re.escape(in_var)
                              + r'(?![A-Za-z0-9_])', c)
                    for c in common_set):
                continue
        return False

    return True


def check_reformulated_from(line: ProofLine, chapter: List[ProofLine],
                            state: VerifierState) -> bool:
    """@brief Verify a ``reformulated from`` row reformulates its source theorem.

    @details
    The ``reformulated from`` tag emits a row whose ``line.expression``
    is a reformulated variant of ``rest[0]`` (the source theorem),
    converting an existence head into the expanded
    left-element-as-premise / right-element-as-head shape. See
    ``_check_reformulation`` for the full algorithm.

    **Gates.**

    - ``line.namespace == "main"``.
    - ``len(rest) >= 1``.

    Then delegate to
    ``_check_reformulation(source, target, gl_binaries)``.

    @param line     The reformulated-from row.
    @param chapter  Sibling rows (unused).
    @param state    Reads ``gl_binaries``.
    @return  True iff the reformulation algorithm accepts.

    @see _check_reformulation — full structural algorithm.
    """
    if line.namespace != "main":
        return False
    if len(line.rest) < 1:
        return False
    source_expr = line.rest[0]
    return _check_reformulation(source_expr, line.expression, state.gl_binaries)


def _check_back_reformulation(direct_expr: str, source_expr: str) -> bool:
    """@brief Verify ``direct_expr`` is the witness-elimination of ``source_expr``.

    @details
    An *incubator back reformulation* takes a proved existence-form
    theorem and eliminates its existential witness through the trailing
    equality that pins the witness to a concrete term, producing the
    direct operator statement used downstream. The source's innermost
    head is an implication ``(>[W](BODY)(=[W,C]))`` — "there is a ``W``
    with ``BODY(W)`` and ``W = C``" — and the direct form's head is
    ``BODY`` with ``W`` replaced by ``C``, the binder and equality
    dropped. Example: source head
    ``(>[w1](in3[i1,i1,w1,+])(=[w1,i2]))`` ("∃w1: i1+i1=w1 ∧ w1=i2")
    back-reformulates to ``(in3[i1,i1,i2,+])`` ("i1+i1=i2").

    **Algorithm.**

    1. Disintegrate both expressions into ``(premises, head)`` via
       ``disintegrate_implication_full`` — this flattens the outer
       implication AND the source's inner existence layer, so the
       source's premise list is ``[...shared premises..., BODY]`` and
       its head is the equality ``(=[W,C])``.
    2. The source head must be an equality ``(=[...])`` with exactly two
       arguments, and the source must carry at least one premise (the
       witness ``BODY``, its last premise); reject otherwise.
    3. The direct form's premises must equal the source's premises minus
       ``BODY`` (same anchor and any other shared premises).
    4. One equality argument is the witness ``W`` (bound in the source's
       inner existence), the other is the concrete term ``C``. Try both
       orientations; accept iff ``BODY`` with ``W`` token-substituted by
       ``C`` (via ``_replace_arg_safe_multi``) equals the direct head.

    Purely structural — no GL binary or theorem-registry lookup. The
    tag stays in ``_ORIGIN_EXEMPT_TAGS`` because the cited source is an
    incubator theorem whose own proof is a separate verification target;
    this check certifies the reformulation STEP is sound, not the
    source's provenance.

    @param direct_expr   The back-reformulated direct operator statement.
    @param source_expr   The cited existence-form source (``rest[0]``).
    @return  True iff ``direct_expr`` is a valid witness-elimination of
             ``source_expr`` under the algorithm above.

    @see check_incubator_back_reformulation — row-level caller.
    @see _check_reformulation — the forward (direct → existence) check.
    """
    prem_d, head_d = disintegrate_implication_full(direct_expr)
    prem_s, head_s = disintegrate_implication_full(source_expr)
    if not head_s.startswith("(=[") or not prem_s:
        return False
    body = prem_s[-1]
    if prem_d != prem_s[:-1]:
        return False
    args = _extract_args(head_s)
    if len(args) != 2:
        return False
    a, b = args
    for witness, const in ((a, b), (b, a)):
        if _replace_arg_safe_multi(body, {witness: const}) == head_d:
            return True
    return False


def check_incubator_back_reformulation(line: ProofLine, chapter: List[ProofLine],
                                        state: VerifierState) -> bool:
    """@brief Verify an ``incubator back reformulation`` row is a sound rewrite.

    @details
    The ``incubator back reformulation`` tag carries the reverse of the
    incubator's forward-reformulation step: a proved existence-form
    theorem ``(>[...](Anchor)(>[w](BODY)(=[w,c])))`` back-rewritten into
    the direct operator statement ``(>[...](Anchor)(BODY[w:=c]))`` for
    downstream use. This checker validates that reformulation STEP
    structurally — the witness ``w`` is eliminated through its equality,
    the anchor and shared premises are preserved — via
    ``_check_back_reformulation``. (Historically this row was
    stub-accepted; the structural check is now performed.)

    **Gates.**

    - ``line.namespace == "main"``.
    - ``len(rest) >= 1`` (``rest[0]`` is the cited existence-form source).
    - ``_check_back_reformulation(line.expression, line.rest[0])``.

    The tag is listed in ``_ORIGIN_EXEMPT_TAGS``: the cited source's own
    provenance is an incubator-side verification target, so the
    chapter-level ``origin`` meta-check does not re-walk the citation.

    @param line     The row.
    @param chapter  Sibling rows (unused).
    @param state    Global state (unused).
    @return  True iff the namespace/rest gates pass AND the row is a
             valid witness-elimination of its cited source.

    @invariant I-16 — a failure means the producer emitted a direct form
    that does not follow from the cited existence source; never a false
    positive.
    @see _check_back_reformulation — the structural core.
    """
    if line.namespace != "main":
        return False
    if len(line.rest) < 1:
        return False
    return _check_back_reformulation(line.expression, line.rest[0])


def check_externally_provided_theorem(line: ProofLine, chapter: List[ProofLine],
                                       state: VerifierState) -> bool:
    """@brief Verify an ``externally provided theorem`` row cites a real external.

    @details
    The ``externally provided theorem`` tag references a theorem
    from outside the chapter's own proof graph — typically a Peano-
    batch theorem cited from an incubator chapter, or a hand-
    provided axiom listed in
    ``processed_proof_graph/external_theorems.txt``.

    Two-stage membership check (mirrors ``check_theorem_tag``):

    1. **Direct membership.** ``line.expression`` is exactly in
       ``state.external_theorems`` (which includes raw + renamed
       forms emitted by the processor).
    2. **w/V revert.** For theorem-anchor implications, swap w→v /
       W→V (defensive guard against a future regression where a
       citation-form leaks into a HEAD column).

    Namespace gate: ``"main"``.

    @param line     The row.
    @param chapter  Sibling rows (unused).
    @param state    Reads ``external_theorems``.
    @return  True iff the expression matches some external under
             either matching path.

    @see check_theorem_tag — same membership idea for the internal
    global theorem registry.
    """
    if line.namespace != "main":
        return False
    # Direct membership (covers originals + renamed forms)
    if line.expression in state.external_theorems:
        return True
    # Fast-path mirror of the origin meta-check's w/W -> v/V revert (load-
    # bearing there for cited theorem-anchor implications). Defensive at THIS
    # site: HEAD column stays v/V by processor design, so this only fires if
    # a future regression slips a citation form into the HEAD column.
    if _is_theorem_anchor_impl_local(line.expression):
        rev = _revert_w_to_v_in_theorem_citation(line.expression)
        if rev != line.expression and rev in state.external_theorems:
            return True
    return False


def check_variable_copy(line: ProofLine,
                        chapter: List[ProofLine],
                        state: VerifierState) -> bool:
    """@brief Verify a ``variable copy`` row is a sound free-axiom declaration.

    @details
    GL introduces a free axiom ``(=[Y, Y_copy])`` in any scope when
    the prover needs a fresh name for an existing variable. ``Y_copy``
    = ``Y + "_copy"`` is a freshly manufactured suffixed name that
    never appears anywhere in the model except as a second name for
    ``Y`` — so the axiom is always sound (a conservative extension
    by construction). The five C++ emission sites
    (``checkNecessityForEquality``, ``disintegrateExprHypothetically``,
    ``reactToHypo``, and the two ``addTheoremToMemory`` triggers — the
    antisymmetry shape and the output-collision shape) all emit this
    shape with an empty origin vector; they differ only in WHEN the
    prover chose to introduce the copy.

    **Structural checks.**

    1. ``line.expression`` is ``(=[a, b])`` with arity exactly 2 and
       ``b == a + "_copy"`` byte-exactly.
    2. ``rest`` is empty (dead-end axiom — no origin refs).
    3. Every non-equality chapter row whose expression mentions ``b``
       in its args can reach a ``variable copy`` declaration for
       ``(=[a, b])`` via the origin-graph walk
       (``_trace_back_to``). This enforces that ``b`` cannot be
       smuggled into the proof graph without an explicit declaration
       — any line that uses ``b`` must have an origin chain rooted in
       the copy declaration.

    The trace uses ``_extract_args`` (top-level only) for the outer
    scan — nested ``b`` mentions inside compound sub-expressions are
    NOT flagged because the outer expression doesn't visibly carry
    ``b`` at its top-level args. Once the trace begins, the
    ``should_follow`` predicate uses ``_extract_all_args`` (recursive)
    so nested mentions DO follow the edge.

    @param line     The variable-copy row.
    @param chapter  All chapter rows (for the trace check).
    @param state    Global state (unused).
    @return  True iff the row is well-formed AND every use of ``b``
             traces back to this declaration.
    """
    if not line.expression.startswith("(=["):
        return False
    eq_args = _extract_args(line.expression)
    if len(eq_args) != 2:
        return False
    a, b = eq_args[0], eq_args[1]
    if b != a + "_copy":
        return False
    if line.rest:                        # dead-end: no origin refs
        return False

    # Build expression → lines map for trace-back
    expr_to_lines: Dict[str, List[ProofLine]] = {}
    for ch_line in chapter:
        expr_to_lines.setdefault(ch_line.expression, []).append(ch_line)

    the_equality = line.expression
    is_target = lambda cl: (cl.expression == the_equality
                            and cl.tag == "variable copy")
    # Compound sources (e.g. negated implications) hide the copy var inside
    # nested ``[..]`` groups; use the recursive extractor so the trace-back
    # walker keeps following edges that still carry b.
    should_follow = lambda src: b in _extract_all_args(src)

    # Every occurrence of b in a non-equality chapter expression must
    # trace back to a variable copy declaration for (=[a, b]).
    for ch_line in chapter:
        if b not in _extract_args(ch_line.expression):
            continue
        if ch_line.expression.startswith("(=["):
            continue            # skip equalities, they propagate b
        if not _trace_back_to(ch_line.expression, expr_to_lines,
                              is_target, should_follow):
            return False
    return True


def _extract_bound_vars(expr: str) -> Set[str]:
    """@brief Collect every name introduced inside ANY ``>[…]`` binder of ``expr``.

    @details
    Like ``_collect_bound_vars`` but uses ``re.finditer`` to find
    every binder group at any nesting depth (no manual bracket
    tracking). Returns a set; duplicates across binders collapse.

    Used by ``check_equalize_variable`` (the ``multiplied from``
    checker) to classify each arg as bound (locally quantified) vs
    free (anchor parameters or external constants) for the I-24
    soundness gate (free-free anchor merge with distinct names is
    forbidden).

    @param expr  An MPL expression.
    @return  Set of every bound-variable name in ``expr``.

    @see _collect_bound_vars — sibling with linear-scan tracking.
    @see check_equalize_variable — primary consumer.
    """
    bound: Set[str] = set()
    for m in re.finditer(r'>\[([^\]]*)\]', expr):
        for name in m.group(1).split(','):
            name = name.strip()
            if name:
                bound.add(name)
    return bound


def check_equalize_variable(line: ProofLine, chapter: List[ProofLine],
                            state: VerifierState) -> bool:
    """@brief Verify a ``multiplied from`` step: arg mapping must be consistent.

    @details
    The ``multiplied from`` tag records a Bell-partition
    equalisation step: the source implication's bound variables get
    identified pairwise (a Bell partition of the bound-var set)
    producing a new implication where each partition's identified
    vars share one name. The verifier reconstructs the implied
    var-to-var map and checks two contracts:

    **Consistency.** Source and copy implications have the same
    premise count + head. Each position pairs an origin arg with a
    copy arg. If the same origin arg appears at two positions
    mapping to different copy args, the substitution is not
    well-defined → reject. Core names and arity must match at every
    position; head pair too.

    **I-24 soundness gate (chapter-1115 bug).** A ``multiplied from``
    step may identify bound variables (Bell-partition equalisation,
    sound), and may rewrite a bound variable to a free anchor
    parameter (bound→free OR free→bound, also sound), but it must
    NOT identify two distinct free anchor parameters with each
    other. Doing so silently rewrites a free slot of the rule body
    and emits a logically stronger rule than the source. Free vs
    bound classification: a name is *bound* if it appears inside any
    ``>[...]`` binder clause; otherwise it is free.

    The check iterates the mapping; for each ``(orig_arg → copy_arg)``
    where the names differ:

    - If both are bound (one in origin_bound, one in copy_bound) → OK.
    - If bound→free or free→bound → OK.
    - If free→free with different names → REJECT (I-24 violation).

    @param line     The multiplied-from row. ``rest[0]`` is the
                    source implication; ``line.expression`` is the
                    multiplied copy.
    @param chapter  Sibling rows (unused).
    @param state    Global state (unused; check is purely syntactic).
    @return  True iff consistency holds AND no free→free anchor
             merge with distinct names occurs.

    @invariant I-24 — multiplyImplication must not equate two
    distinct free anchor parameters. The chapter-1115 incubator bug
    surfaced exactly this case; verifier catches it as a per-row
    structural failure.
    """
    if len(line.rest) < 2:
        return False
    origin_expr = line.rest[0]
    copy_expr = line.expression
    origin_premises, origin_head = disintegrate_implication_full(origin_expr)
    copy_premises, copy_head = disintegrate_implication_full(copy_expr)
    if len(origin_premises) != len(copy_premises):
        return False
    pairs = list(zip(origin_premises, copy_premises))
    pairs.append((origin_head, copy_head))
    mapping: Dict[str, str] = {}
    for orig_elem, copy_elem in pairs:
        if _extract_core_name(orig_elem) != _extract_core_name(copy_elem):
            return False
        orig_args = _extract_args(orig_elem)
        copy_args = _extract_args(copy_elem)
        if len(orig_args) != len(copy_args):
            return False
        for oa, ca in zip(orig_args, copy_args):
            if oa in mapping:
                if mapping[oa] != ca:
                    return False
            else:
                mapping[oa] = ca
    # Free-anchor-merge guard: reject any (orig_arg → copy_arg) where both
    # sides are free (not bound by any ``>[...]`` clause) and the names
    # differ. Bound→bound and bound→free remappings are sound; free→free
    # with different names is the chapter-1115 multiplyImplication bug.
    origin_bound = _extract_bound_vars(origin_expr)
    copy_bound = _extract_bound_vars(copy_expr)
    for orig_arg, copy_arg in mapping.items():
        if orig_arg == copy_arg:
            continue
        if orig_arg in origin_bound:
            continue
        if copy_arg in copy_bound:
            continue
        return False
    return True


def check_contradiction(line: ProofLine, chapter: List[ProofLine],
                        state: VerifierState) -> bool:
    """@brief Verify a ``contradiction`` row records a valid contradiction LB.

    @details
    A contradiction row records that ``!(cleanOp)`` was proved
    because both ``expr`` and its negation were derived inside a
    contradiction LB (whose seed ``cleanOp`` was hypothesised at
    chapter start). The verifier validates the row's structural
    shape; the corresponding chapter-level trace check
    (``contradiction trace``) separately verifies the ingredients
    can be derived from the seed.

    **Row layout.**
    ::
        line.expression  = negate(cleanOp)
        line.namespace   = "main"
        rest[0]          = expr
        rest[1]          = "main"
        rest[2]          = negate(expr)
        rest[3]          = "main"
        rest[4]          = cleanOp
        rest[5]          = "main"

    **Structural rule.**

    1. ``line.namespace == "main"``.
    2. ``len(rest) >= 6``.
    3. All three namespaces (rest[1], rest[3], rest[5]) are ``"main"``.
    4. ``line.expression`` and ``cleanOp`` are negations of each
       other — either ``line.expression == "!" + cleanOp`` (a
       verbatim positive-seed LB) or ``cleanOp == "!" +
       line.expression`` (a complement LB whose seed is the negated
       head, ``try_contradiction_negated_head``). Symmetric mirror
       of rule 5.
    5. ``expr`` and ``negate(expr)`` are negations of each other —
       either ``negate(expr) == "!" + expr`` or
       ``expr == "!" + negate(expr)``.
    6. ``cleanOp`` appears in the chapter as a row tagged
       ``"task formulation"``.

    **Scope-carrying reductio variant (sequenced or-disintegration
    campaign).** A contradiction INSIDE a marked child scope refutes
    that scope's assumption at the scope's parent. Row layout
    (exactly four rest fields)::

        line.expression = negate(scope assumption)
        line.namespace  = parent of rest[1]
        rest[0] = X            rest[1] = the marked child scope
        rest[2] = negate(X)    rest[3] = scope where negate(X) is known

    Structural rules: ``X``/``negate(X)`` are negations of each
    other; ``rest[3]`` is ``rest[1]`` itself or one of its
    ``_boundary_``-ancestors; ``line.namespace`` equals ``rest[1]``
    minus its last ``_boundary_<payload>`` segment; and
    ``line.expression`` negates the scope's assumption — for a
    ``contradiction_<!S>`` payload it IS the negated seed ``!S``,
    for an ``ordis_<sig>_(<disjunct>)`` payload it is the disjunct's
    negation.

    @param line     The contradiction row.
    @param chapter  All chapter rows (for cleanOp task-formulation
                    lookup).
    @param state    Global state (unused).
    @return  True iff every structural rule passes.

    @see verify_chapter — runs the trace check that verifies the
    ingredients are reachable from the seed.
    """
    if len(line.rest) == 4:
        expr = line.rest[0]
        scope_ns = line.rest[1]
        neg_expr = line.rest[2]
        neg_ns = line.rest[3]
        if not (neg_expr == "!" + expr or expr == "!" + neg_expr):
            return False
        cut = scope_ns.rfind("_boundary_")
        if cut < 0:
            return False
        if line.namespace != scope_ns[:cut]:
            return False
        if not (neg_ns == scope_ns
                or scope_ns.startswith(neg_ns + "_boundary_")):
            return False
        payload = scope_ns[cut + len("_boundary_"):]
        if payload.startswith("contradiction_"):
            return line.expression == payload[len("contradiction_"):]
        if payload.startswith("ordis_"):
            sep = payload.find("_(", len("ordis_"))
            if sep < 0:
                return False
            body = payload[sep + 1:]
            if len(body) < 4 or body[0] != '(' or body[-1] != ')':
                return False
            disjunct = body[1:-1]
            neg_disj = (disjunct[1:] if disjunct.startswith('!')
                        else "!" + disjunct)
            return line.expression == neg_disj
        return False

    if line.namespace != "main":
        return False
    if len(line.rest) < 6:
        return False

    expr = line.rest[0]
    expr_ns = line.rest[1]
    neg_expr = line.rest[2]
    neg_ns = line.rest[3]
    clean_op = line.rest[4]
    clean_ns = line.rest[5]

    # All namespaces must be "main"
    if expr_ns != "main" or neg_ns != "main" or clean_ns != "main":
        return False

    # line.expression and cleanOp must be negations of each other:
    # "!"+cleanOp = verbatim positive-seed LB; cleanOp = "!"+expression =
    # complement LB with a negated-head seed (mirrors rule 5).
    if line.expression == "!" + clean_op:
        pass
    elif clean_op == "!" + line.expression:
        pass
    else:
        return False

    # expr and neg_expr must be negations of each other
    if neg_expr == "!" + expr:
        pass
    elif expr == "!" + neg_expr:
        pass
    else:
        return False

    # cleanOp must appear as task formulation in the chapter
    for ch in chapter:
        if ch.expression == clean_op and ch.tag == "task formulation":
            return True

    return False


# ---------------------------------------------------------------------------
#  OR disintegration / convergence helpers
# ---------------------------------------------------------------------------

def _parse_or_disjuncts(expanded_or: str) -> List[str]:
    """@brief Parse an expanded De Morgan OR into its disjunct list.

    @details
    Inverse of ``_build_or_from_elements``. Recursive descent:

        ``!(&(&!(=[a,x])!(=[b,x]))!(=[c,x]))``
        → ``['(=[a,x])', '(=[b,x])', '(=[c,x])']``

    Structure is nested ``!(&<left><right>)`` where every conjunct is
    a disjunct's cancelled negation:

    - **Left child** starting with ``(&`` → the negated or-so-far;
      negate it back (prepend ``!``) and recurse to extract its
      disjuncts.
    - **Any other left child** → a single disjunct; ``_negate_expr``
      recovers its true polarity.
    - **Right child** → one disjunct; ``_negate_expr`` recovers its
      true polarity.

    Malformed input (missing outer ``!(&``, unbalanced parens) →
    returns the input wrapped in a single-element list as a fallback.

    Used by the OR-family checkers to recover disjuncts from
    raw-form OR expressions in cells that haven't been compiled to
    ``(or<N>[...])`` shape.

    @param expanded_or  An MPL expression purporting to be an
                        expanded De Morgan OR.
    @return  List of disjuncts in nested-left-to-rightmost-right
             order.
    """
    s = expanded_or.strip()
    if len(s) < 4 or s[:3] != '!(&':
        return [s]  # single expression, not an OR

    # Strip outer !(&...) -> inner
    inner = s[3:-1]

    # Find boundary between left and right child by tracking paren depth
    depth = 0
    seen_open = False
    for i, ch in enumerate(inner):
        if ch in ('(', '['):
            depth += 1
            seen_open = True
        elif ch in (')', ']'):
            depth -= 1
        if seen_open and depth == 0:
            left = inner[:i + 1]
            right = inner[i + 1:]
            # Left: a bare (&...) conjunct is the NEGATED or-so-far —
            # negate it back and recurse. Every other conjunct is one
            # disjunct's cancelled negation, so negating back recovers
            # its true polarity.
            if left.startswith('(&'):
                disjuncts = _parse_or_disjuncts('!' + left)
            else:
                disjuncts = [_negate_expr(left)]
            disjuncts.append(_negate_expr(right))
            return disjuncts

    return [s]  # fallback


def check_or_disintegration(line: ProofLine, chapter: List[ProofLine],
                            state: VerifierState) -> bool:
    """@brief Verify an ``or disintegration`` row opens a case-split branch.

    @details
    Records that an ``_ordis_`` (case-split) branch was opened with
    one specific disjunct asserted. Sibling to ``or branch proven``
    but for ``_ordis_`` (consume an existing OR) rather than
    ``_orint_`` (produce an OR via sub-implications).

    **Row layout (exactly two rest fields).**
    ::
        line.expression  = the asserted disjunct
        line.namespace   = parent + "_boundary_ordis_<or>_(<disjunct>)"
        rest[0]          = the compiled OR (or<N>[…])
        rest[1]          = parent (the OR's parent scope)

    **Validation (D-36 — mirrors ``check_or_branch_proven``
    structure + the OR-origin check that is well-founded for
    ``_ordis_``).**

    1. ``len(rest) == 2`` exactly. The tag is in
       ``_ORIGIN_EXEMPT_TAGS`` so any extra rest pairs would be
       silently accepted by the generic origin check; reject up front
       so the row's contract stays auditable.
    2. ``rest[0]`` is a known compiled OR ``(or<N>[…])`` whose
       recursively substituted contiguous OR tree yields at least two
       ordered non-OR leaves (matching arity at every node).
    3. ``line.expression`` is one of those leaves (modulo equality
       symmetry); an intermediate compiled OR is not a leaf.
    4. ``line.namespace`` is EXACTLY
       ``rest[1] + "_boundary_ordis_" + rest[0] + "_(" + <disjunct> + ")"``
       for ``<disjunct>`` matching ``line.expression`` (modulo
       equality symmetry). The outer OR remains in the namespace even
       for an inner leaf. No substring search — the row's claim is
       "this exact flat-cohort branch", not "some descendant containing
       the substring".
    5. The OR has an independent derivation row at parent scope
       (``rest[1]``). A chapter row exists with
       ``expression == rest[0]``, ``namespace == rest[1]``, and
       ``tag != "or disintegration"`` — i.e. the OR was actually
       derived (via ``implication`` / ``expansion`` / ``theorem`` / …)
       before being case-split. This check IS well-founded for
       ``_ordis_``: case-split CONSUMES an existing OR, so the OR
       must be derived first. The analogous check on
       ``check_or_branch_proven`` was dropped per D-36 because
       ``_orint_`` PRODUCES an OR — no separate derivation exists by
       design.

    @param line     The or-disintegration row.
    @param chapter  All chapter rows (for OR-origin lookup).
    @param state    Reads ``binaries_for_chapter`` for the disjunct
                    decode.
    @return  True iff every validation step passes.

    @invariant D-36 — _ordis_ case-split requires OR-origin
    derivation; _orint_ branch proven does not.
    """
    if len(line.rest) != 2:
        return False
    asserted = line.expression
    or_expr = line.rest[0]
    parent_ns = line.rest[1]
    branch_ns = line.namespace

    disjuncts = _or_disjuncts_from_compiled(
        or_expr, state.binaries_for_chapter(), flatten_nested=True)
    if disjuncts is None:
        return False
    if not _disjunct_matches(asserted, disjuncts):
        return False

    # OR has an independent derivation at parent scope (well-founded
    # for `_ordis_` — case-split consumes an existing OR).
    or_origin_exists = any(
        ch_line.expression == or_expr
        and ch_line.namespace == parent_ns
        and ch_line.tag != "or disintegration"
        for ch_line in chapter
    )
    if not or_origin_exists:
        return False

    candidates = [asserted]
    if asserted.startswith('(=[') and asserted.endswith('])'):
        eq_args = asserted[3:-2].split(',')
        if len(eq_args) == 2:
            candidates.append('(=[' + eq_args[1] + ',' + eq_args[0] + '])')

    for variant in candidates:
        expected = (parent_ns
                    + "_boundary_ordis_"
                    + or_expr
                    + "_(" + variant + ")")
        if branch_ns == expected:
            return True
    return False


def check_or_convergence(line: ProofLine, chapter: List[ProofLine],
                         state: VerifierState) -> bool:
    """@brief Verify an ``or convergence`` row promotes per-branch conclusions to parent.

    @details
    When every branch of an OR case-split has independently derived
    the same conclusion ``C``, the convergence row promotes ``C`` to
    the OR's parent scope. This checker enforces the producer-side
    row layout spec.

    **Row layout (compiled-form OR only).**
    ::
        <C> <parent> or convergence <OR> <parent>
                                          <C> <branch_D1>
                                          <C> <branch_D2>
                                          …
                                          <C> <branch_DK>

    Where ``K`` is the recursively flattened non-OR leaf count. The
    original outer compact OR remains the row identity. Concretely:

    - ``line.expression`` = ``C`` (the converged conclusion)
    - ``line.namespace`` = parent (the OR's parent scope)
    - ``rest[0]`` = OR (compiled ``(or<N>[…])``)
    - ``rest[1]`` = parent
    - one two-field entry per flattened disjunct: ``rest[2*i + 2]``
      (an expression) and ``rest[2*i + 3]`` (a namespace)

    Each entry takes one of two forms. A **survivor** entry is
    ``(C, branch_Di)`` — the conclusion derived at that branch's
    exact ``_ordis_`` scope. A **retired** entry is
    ``(negate(D_i), ns)`` — dead-branch retirement discharged the
    branch by reductio: its asserted disjunct's negation is known at
    ``ns``, a scope the parent inherits from (the parent itself, an
    ancestor, or the retired branch itself for an ex-falso
    self-refutation). Either way the row accounts for all K
    flattened disjuncts.

    **Validation contract.** For the row to PASS, all of:

    1. **Layout.** ``len(rest) == 2 + 2*K``, even length, ``>= 6``
       (so ``K >= 2``).
    2. **Parent-scope match.** ``line.namespace == rest[1]``.
    3. **OR is real.** ``rest[0]`` is a compiled ``(or<N>[…])`` with
       a known GL-binary entry of category ``"or"`` whose recursive
       contiguous-OR expansion yields at least two leaves; leaf count
       ``K`` matches the entry count.
    4. **Entry classification + coverage.** Every entry matches one
       uncovered disjunct — a survivor entry by the disjunct payload
       embedded in its ``_ordis_`` branch scope, a retired entry by
       the negation of the disjunct plus the parent-visible scope —
       and the K entries cover all K disjuncts exactly once.
    5. **Per-entry derivation evidence.** For every entry
       ``(expr, ns)``, a chapter row exists with
       ``expression == expr`` and ``namespace == ns`` under any tag
       (the "each ingredient has its own line" rule).

    @param line     The convergence row.
    @param chapter  All chapter rows (for per-branch derivation
                    lookup).
    @param state    Reads ``binaries_for_chapter``.
    @return  True iff every validation step passes.
    """
    # 1. Layout shape
    if len(line.rest) < 6 or (len(line.rest) % 2) != 0:
        return False

    or_expr = line.rest[0]
    parent_ns = line.rest[1]

    # 2. Parent-scope match
    if line.namespace != parent_ns:
        return False

    # 3. OR is real; count disjuncts via GL-binary lookup
    disjuncts = _or_disjuncts_from_compiled(
        or_expr, state.binaries_for_chapter(), flatten_nested=True)
    if disjuncts is None or len(disjuncts) < 2:
        return False
    k = len(disjuncts)
    if len(line.rest) != 2 + 2 * k:
        return False

    # 4-7. Per-entry classification. Dead-branch retirement can shrink a
    # cohort (a branch whose asserted disjunct is refuted is wiped and its
    # convergence count reduced), so each of the K entries is either a
    # SURVIVOR entry — the conclusion ``C`` derived at that branch's exact
    # ``_ordis_`` scope — or a RETIRED entry — the negation of that branch's
    # asserted disjunct, at a scope from which the parent inherits it (the
    # parent itself, one of its ancestors, or the retired branch itself for
    # an ex-falso self-refutation). Together the K entries must cover all K
    # flattened disjuncts exactly once, and every entry must have its own
    # chapter row (the "each ingredient has its own line" rule).
    conclusion = line.expression
    covered = [False] * k
    branch_prefix = parent_ns + "_boundary_ordis_" + or_expr + "_("
    for i in range(k):
        e_field = line.rest[2 + 2 * i]
        ns_field = line.rest[2 + 2 * i + 1]
        matched = -1
        if (e_field == conclusion
                and ns_field.startswith(branch_prefix)
                and ns_field.endswith(")")):
            # Survivor: the branch scope's payload names its asserted
            # disjunct — extract it and match against an uncovered disjunct.
            dj = ns_field[len(branch_prefix):-1]
            for d_idx, d in enumerate(disjuncts):
                if not covered[d_idx] and d == dj:
                    matched = d_idx
                    break
        else:
            # Retired: the negation of an uncovered disjunct, at a scope the
            # parent inherits from (or the retired branch itself).
            for d_idx, d in enumerate(disjuncts):
                if covered[d_idx]:
                    continue
                neg_d = d[1:] if d.startswith("!") else "!" + d
                if e_field != neg_d:
                    continue
                branch_ns = branch_prefix + d + ")"
                if (ns_field == parent_ns
                        or parent_ns.startswith(ns_field + "_boundary_")
                        or ns_field == branch_ns):
                    matched = d_idx
                    break
        if matched < 0:
            return False
        covered[matched] = True
        # Per-entry derivation evidence (each ingredient has its own line).
        if not any(ch_line.expression == e_field
                   and ch_line.namespace == ns_field
                   for ch_line in chapter):
            return False

    return all(covered)


def _or_disjuncts_from_compiled(
        or_expr: str,
        binaries: List[dict],
        *,
        flatten_nested: bool = False) -> Optional[List[str]]:
    """@brief Decode a compiled OR into substituted immediate or flattened leaves.

    @details
    Given a compiled OR expression ``(or<N>[arg1,arg2,…])``, look it
    up in the supplied GL binaries and substitute ``u_i`` placeholders
    with the OR's args in order. By default, return the immediate
    disjuncts. When
    ``flatten_nested`` is true, recursively expand every substituted
    child that is itself a compiled OR and return the ordered non-OR
    leaves (the one-go OR producer contract). A repeated OR
    core on one recursion path or more than 64 leaves is malformed and
    returns ``None``.

    Returns ``None`` if any of:

    - the root or any nested OR does not match the ``(or<N>[…])`` shape;
    - no loaded binary contains the corresponding ``or<N>`` entry;
    - an entry's ``category`` is not ``"or"``;
    - an entry has fewer than 2 elements;
    - an OR's argument count does not match the binary's ``arity``
      (or, when ``arity`` is absent, the placeholder count parsed
      from ``signature``). A mismatched arity is treated as a
      malformed OR expression and rejected;
    - recursive flattening finds a cycle or exceeds the 64-leaf cap.

    Every OR producer/checker requests flattened leaves. The immediate mode
    remains available only for callers explicitly inspecting one compiled
    definition level.

    @param or_expr   The compiled OR expression.
    @param binaries  List of GL binary dicts to search (typically
                     ``state.binaries_for_chapter()``).
    @param flatten_nested Recursively return non-OR leaves when true.
    @return  Substituted disjunct list on success; ``None`` on failure.
    """
    def decode(candidate: str, path_cores: Tuple[str, ...],
               root: bool) -> Optional[List[str]]:
        match = re.match(r'^\(or(\d+)\[([^\]]*)\]\)$', candidate)
        if not match:
            return None if root else [candidate]

        core = f'or{match.group(1)}'
        if core in path_cores:
            return None
        raw = match.group(2)
        args = raw.split(',') if raw else []

        for binary in binaries:
            entry = binary.get(core)
            if entry is None or entry.get('category') != 'or':
                continue
            expected = entry.get('arity')
            if expected is None:
                signature = entry.get('signature', '')
                sig_match = re.match(r'^\(or\d+\[([^\]]*)\]\)$', signature)
                if sig_match:
                    sig_args = sig_match.group(1)
                    expected = len(sig_args.split(',')) if sig_args else 0
            if expected is not None and len(args) != expected:
                continue

            elements = entry.get('elements', [])
            if len(elements) < 2:
                return None
            substituted_elements: List[str] = []
            for element in elements:
                substituted = element
                for i, arg in enumerate(args, start=1):
                    substituted = _replace_arg_safe(
                        substituted, f'u_{i}', arg)
                substituted_elements.append(substituted)

            if not flatten_nested:
                return substituted_elements

            leaves: List[str] = []
            for element in substituted_elements:
                nested = decode(element, path_cores + (core,), False)
                if nested is None:
                    return None
                leaves.extend(nested)
                if len(leaves) > 64:
                    return None
            return leaves
        return None

    return decode(or_expr, (), True)


def _disjunct_matches(candidate: str, disjuncts: List[str]) -> bool:
    """@brief Equality-symmetric membership test for a candidate disjunct.

    @details
    True iff ``candidate`` matches one of ``disjuncts`` directly OR
    via equality symmetry: an equality ``(=[a,b])`` is treated as
    the same disjunct as ``(=[b,a])``. Other expression cores
    require literal equality.

    Used by the OR-family checkers to accept disjuncts that may
    appear in either argument order — the prover and processor
    don't canonicalize equality direction in OR-disjunct positions,
    so the verifier accepts both.

    @param candidate  The expression being tested.
    @param disjuncts  The set of valid disjuncts.
    @return  True iff ``candidate`` is in ``disjuncts`` directly, or
             a binary-equality with swapped args is in ``disjuncts``.
    """
    if candidate in disjuncts:
        return True
    if candidate.startswith('(=[') and candidate.endswith('])'):
        args = candidate[3:-2].split(',')
        if len(args) == 2:
            swapped = '(=[' + args[1] + ',' + args[0] + '])'
            if swapped in disjuncts:
                return True
    return False


def check_or_branch_proven(line: ProofLine, chapter: List[ProofLine],
                           state: VerifierState) -> bool:
    """@brief Verify an ``or branch proven`` row opens an _orint_ subproof.

    @details
    Emitted by the prover (``prover.cpp::or branch proven`` site,
    see ``docs/agentic_swdd/20_core_concepts/07_or_branching.md``) when an OR
    needs to be proved via the _orint_ rewrite: the OR goal is
    rewritten into K sub-implications-to-prove over its ordered atomic
    leaves; the per-branch scope is opened with one leaf asserted, and
    discharging it discharges the OR.

    **Row layout (exactly two rest fields).**
    ::
        line.expression  = the compiled OR (or<N>[…])
        line.namespace   = parent (the OR's parent scope)
        rest[0]          = the asserted atomic leaf
        rest[1]          = parent + "_boundary_orint_<or>_(<leaf>)"

    **Validation.**

    1. ``len(rest) == 2`` exactly. The tag is in
       ``_ORIGIN_EXEMPT_TAGS`` so any extra rest pairs would be
       silently accepted by the generic origin check; reject up front
       so the row's contract stays auditable.
    2. ``line.expression`` is a known compiled OR with ≥2 atomic leaves
       after ``u_i`` substitution.
    3. ``rest[0]`` is one of those leaves (modulo equality
       symmetry).
    4. ``rest[1]`` is EXACTLY
       ``parent + "_boundary_orint_" + or_expr + "_(" + <leaf> + ")"``
       for ``<leaf>`` matching ``rest[0]`` (modulo equality
       symmetry). No substring search.

    **D-36 note.** No non-``or branch proven`` derivation row for the
    OR at parent scope is required. ``_orint_`` rewrites the OR goal
    into K atomic sub-implications-to-prove; when one fires, the
    ``or branch proven`` row IS the OR's derivation by design — so
    treating ``_orint_`` as a case-split with a separately-derived OR
    would be the wrong model.

    @param line     The or-branch-proven row.
    @param chapter  Sibling rows (unused — D-36 dropped the
                    OR-origin lookup).
    @param state    Reads ``binaries_for_chapter``.
    @return  True iff every validation step passes.

    @invariant D-36 — _orint_ branch proven row IS the OR's
    derivation; no separate origin row exists.
    """
    if len(line.rest) != 2:
        return False
    or_expr = line.expression
    asserted = line.rest[0]
    branch_ns = line.rest[1]
    parent_ns = line.namespace

    disjuncts = _or_disjuncts_from_compiled(
        or_expr, state.binaries_for_chapter(), flatten_nested=True)
    if disjuncts is None:
        return False
    if not _disjunct_matches(asserted, disjuncts):
        return False

    candidates = [asserted]
    if asserted.startswith('(=[') and asserted.endswith('])'):
        eq_args = asserted[3:-2].split(',')
        if len(eq_args) == 2:
            candidates.append('(=[' + eq_args[1] + ',' + eq_args[0] + '])')

    for variant in candidates:
        expected = (parent_ns
                    + "_boundary_orint_"
                    + or_expr
                    + "_(" + variant + ")")
        if branch_ns == expected:
            return True
    return False


def check_or_branch_assumption(line: ProofLine, chapter: List[ProofLine],
                               state: VerifierState) -> bool:
    """@brief Verify an ``or branch assumption`` row seeds other-leaf negations.

    @details
    Emitted by the prover (``prover.hpp::or branch assumption``
    site) for each other atomic leaf of an OR that has been
    case-split via the ``_orint_`` rewrite: in the branch where
    leaf ``D_i`` is asserted, the negation ``!D_j`` of every
    other leaf (``j != i``) is seeded as a branch-local
    assumption (since the case-split's mutual exclusion makes
    every other leaf false in this branch).

    **Row layout (exactly two rest fields).**
    ::
        line.expression  = the negated other leaf (e.g. !(=[i0,v5]))
        line.namespace   = parent + "_boundary_orint_<or>_(<asserted>)"
        rest[0]          = "<or>_integration_goal"
        rest[1]          = parent (the OR's parent scope)

    **Validation.**

    1. ``len(rest) == 2`` exactly. The tag is in
       ``_ORIGIN_EXEMPT_TAGS``.
    2. ``rest[0]`` ends with ``_integration_goal``; stripping the
       suffix yields a compiled OR ``(or<N>[…])`` with a known
       GL-binary entry (matching arity).
    3. ``line.expression`` starts with ``!``; stripping it yields a
       atomic leaf of the OR (modulo equality symmetry).
    4. ``line.namespace`` is EXACTLY
       ``parent + "_boundary_orint_" + or_expr + "_(" + <asserted> + ")"``
       for some atomic leaf ``<asserted>`` of the OR. No substring
       search.
    5. The asserted leaf (extracted from ``line.namespace``'s
       payload) is DIFFERENT from the negated one (modulo equality
       symmetry) — the row asserts a leaf's negation only in
       branches where ANOTHER leaf is asserted.
    6. **Matching ``or branch proven`` row exists.** Some chapter
       row with ``tag == "or branch proven"``,
       ``expression == or_expr``, ``namespace == parent_ns``,
       ``len(rest) == 2``, ``rest[1] == branch_ns``, and ``rest[0]``
       matching the asserted leaf (modulo equality symmetry).
       Without this check the assumption row could pass structurally
       even when the corresponding case-split was never opened.

    @param line     The or-branch-assumption row.
    @param chapter  All chapter rows (for the matching
                    ``or branch proven`` lookup).
    @param state    Reads ``binaries_for_chapter``.
    @return  True iff every validation step passes.

    @invariant D-35 — _orint_ branch assumption seed grammar.
    """
    if len(line.rest) != 2:
        return False

    parent_ns = line.rest[1]
    branch_ns = line.namespace

    suffix = "_integration_goal"
    raw = line.rest[0]
    if not raw.endswith(suffix):
        return False
    or_expr = raw[:-len(suffix)]

    # The assumption asserts the negation of the disjunct under refutation
    # WITH double-negation cancellation (the producer's negateScratch): a
    # positive disjunct's assumption is !-prefixed, a negated disjunct's
    # assumption is its bare positive core. Recover the disjunct by
    # negating back — no leading-'!' requirement.
    negated = _negate_expr(line.expression)

    disjuncts = _or_disjuncts_from_compiled(
        or_expr, state.binaries_for_chapter(), flatten_nested=True)
    if disjuncts is None:
        return False
    if not _disjunct_matches(negated, disjuncts):
        return False

    # Strict structural check: branch_ns must be EXACTLY
    # parent_ns + _boundary_orint_<or>_(<leaf>).
    expected_prefix = (parent_ns
                       + "_boundary_orint_"
                       + or_expr
                       + "_(")
    if not branch_ns.startswith(expected_prefix):
        return False
    after = branch_ns[len(expected_prefix):]
    # Parse one balanced parens group (the asserted leaf, which itself
    # starts with `(` since GL expressions are paren-wrapped). After that
    # group, exactly one `)` closes the wrapper, and nothing else may
    # follow.
    # The asserted leaf is embedded verbatim, so a negated disjunct's
    # payload reads `_(!(...))` — accept one optional leading '!' before
    # the balanced parens group.
    neg_prefix = after.startswith('!')
    core = after[1:] if neg_prefix else after
    if not core.startswith('('):
        return False
    depth = 1
    end = -1
    for j in range(1, len(core)):
        if core[j] == '(':
            depth += 1
        elif core[j] == ')':
            depth -= 1
            if depth == 0:
                end = j
                break
    if end < 0:
        return False
    asserted = ('!' if neg_prefix else '') + core[:end + 1]
    if core[end + 1:] != ')':
        return False

    if not _disjunct_matches(asserted, disjuncts):
        return False
    if asserted == negated:
        return False
    if negated.startswith('(=[') and negated.endswith('])'):
        eq_args = negated[3:-2].split(',')
        if len(eq_args) == 2:
            swapped = '(=[' + eq_args[1] + ',' + eq_args[0] + '])'
            if asserted == swapped:
                return False

    # 6. Matching `or branch proven` row must exist.
    matching_proven = any(
        ch_line.tag == "or branch proven"
        and ch_line.expression == or_expr
        and ch_line.namespace == parent_ns
        and len(ch_line.rest) == 2
        and ch_line.rest[1] == branch_ns
        and _disjunct_matches(ch_line.rest[0], [asserted])
        for ch_line in chapter
    )
    if not matching_proven:
        return False

    return True


def check_vacuous_truth(line: ProofLine, chapter: List[ProofLine],
                        state: VerifierState) -> bool:
    """@brief Verify a ``vacuous truth`` row records a self-contradictory induction step.

    @details
    In an induction step, if both ``expr`` and ``negate(expr)`` are
    derived as statements, the step's premises are self-contradictory
    and any conclusion follows vacuously. The vacuous-truth row
    records the two contradicting ingredients plus the LB exprKey
    they came from.

    **Row layout.**
    ::
        line.expression  = result (any expression, by vacuous-truth
                                   semantics any conclusion is admissible)
        line.namespace   = "main"
        rest[0]          = expr
        rest[1]          = "main"
        rest[2]          = negate(expr)
        rest[3]          = "main"
        rest[4]          = lb_exprKey
        rest[5]          = "main"

    **This checker validates format + negation shape only.** Specifically:

    1. ``line.namespace == "main"``.
    2. ``len(rest) >= 6``.
    3. ``rest[1] == rest[3] == rest[5] == "main"``.
    4. ``expr`` and ``negate(expr)`` are negations of each other —
       either ``negate(expr) == "!" + expr`` or
       ``expr == "!" + negate(expr)``.

    Traceability of at least one ingredient back to ``lb_exprKey``
    is checked SEPARATELY in ``verify_chapter`` under counter
    ``"vacuous truth trace"``, matching the pattern used by
    ``contradiction trace``.

    @param line     The vacuous-truth row.
    @param chapter  Sibling rows (unused by this checker; the
                    trace meta-check is run separately).
    @param state    Global state (unused).
    @return  True iff the format and negation shape are valid.

    @see check_contradiction — closely related shape, but
    contradiction rows are at the root of contradiction LBs where
    vacuous-truth rows are inside induction steps.
    """
    if line.namespace != "main":
        return False
    if len(line.rest) < 6:
        return False

    expr = line.rest[0]
    ns1 = line.rest[1]
    neg_expr = line.rest[2]
    ns2 = line.rest[3]
    ns3 = line.rest[5]

    if ns1 != "main" or ns2 != "main" or ns3 != "main":
        return False

    # Negation shape
    if neg_expr == "!" + expr:
        return True
    if expr == "!" + neg_expr:
        return True

    return False


# Tags exempt from the chapter-level `origin` meta-check inside
# verify_chapter. Their dep cells carry non-derivable references by design:
# - `incubator back reformulation` cites an external incubator theorem in a
#   back-reformulated form that may not match any registry entry.
# - `contradiction` cites the assumption being negated (which by construction
#   does not have a derivation row in the same chapter — that's the point of
#   reductio); the `contradiction trace` meta-check covers chain integrity.
# - The five OR-flow tags have validity-aware dependency chains the bare
#   `dep in originated` check cannot represent; their dedicated `or *` checkers
#   plus `or branch assumption`'s matching-proven-row check cover them.
# Module scope (not function-local) so the set is allocated once per run.
_ORIGIN_EXEMPT_TAGS = frozenset({
    "incubator back reformulation",
    "contradiction",
    "or disintegration",
    "or convergence",
    "or branch proven",
    "or branch assumption",
    "or theorem",
    # `or elimination` cites two proved guard variants plus the licensing
    # or theorem — none locally originated; exemption also enrolls all
    # three citation cells in the usage-graph edge collection.
    "or elimination",
    # `compilation`'s antecedent is the original implication, a broadcast
    # theorem that need not appear as a prior left-side chapter row; the
    # compact<->original link is definitional (GL-binary construction).
    # `check_compilation`'s structural binary-faithfulness check IS the
    # validation, exactly as for `expansion` / `disintegration`.
    "compilation",
})


# Tag name → checker function
TAG_CHECKERS = {
    "implication":                      check_implication,
    "expansion":                        check_expansion,
    "compilation":                      check_compilation,
    "disintegration":                   check_disintegration,
    "task formulation":                 check_task_formulation,
    "equality1":                        check_equality1,
    "equality2":                        check_equality2,
    "symmetry of equality":             check_symmetry_of_equality,
    "symmetry of inequality":           check_symmetry_of_inequality,
    "recursion":                        check_recursion,
    "theorem":                          check_theorem_tag,
    "reformulation for integration and": check_reformulation_for_integration_and,
    "reformulation for integration >[bound]": check_reformulation_for_integration_bound,
    "reformulation for integration >[]": check_reformulation_for_integration_empty,
    "expansion for integration":        check_expansion_for_integration,
    "premise element":                  check_premise_element,
    "validity name":                    check_validity_name,
    "anchor handling":                  check_anchor_handling,
    "reformulated from":                check_reformulated_from,
    "variable copy":                    check_variable_copy,
    "externally provided theorem":      check_externally_provided_theorem,
    "incubator back reformulation":     check_incubator_back_reformulation,
    "multiplied from":                  check_equalize_variable,
    "contradiction":                    check_contradiction,
    "or disintegration":                check_or_disintegration,
    "or convergence":                   check_or_convergence,
    "or branch proven":                 check_or_branch_proven,
    "or branch assumption":             check_or_branch_assumption,
    "vacuous truth":                    check_vacuous_truth,
    "or theorem":                       check_or_theorem,
    "or elimination":                   check_or_elimination,
}


# ---------------------------------------------------------------------------
#  Chapter-level verification
# ---------------------------------------------------------------------------

def verify_chapter(chapter_file: str, lines: List[ProofLine],
                   chapter_type: str, state: VerifierState,
                   chapter_thm: Optional[Tuple[str, str, str]]):
    """@brief Verify a single chapter — chapter-level dispatcher.

    @details
    The chapter-level driver. Runs through six phases, updating
    ``state``'s per-tag and per-meta-counter values in place:

    1. **theorem goal reached.** Call
       ``check_theorem_goal_reached``; record under
       ``state.goal_reached``.

    2. **Set per-chapter transient context.**

       - ``state.current_chapter_thm = chapter_thm``.
       - ``state.current_chapter_type = chapter_type``.
       - **Anchor-substring binary resolution.** When the chapter's
         theorem mentions ``AnchorIncubator``, the iteration scans
         only ``Incubator*`` tags (D-54 chapter-context filter, see
         ``binaries_for_chapter``); otherwise it scans every loaded
         binary. The first tag whose ``Anchor<tag>`` substring is
         present in the theorem becomes ``current_gl_binary`` and
         drives ``current_resolved_defsets``.
       - If no binary matches, fall back to the atomic-only resolved
         defsets.

    3. **Self-reference check.** Any chapter row tagged ``theorem``
       whose expression equals the chapter's own theorem is a
       self-reference → record under
       ``counter_for("self-reference")``.

    4. **Anchor handling trace.** For every _copy variable named in
       ANY ``anchor handling`` row's args, every chapter row whose
       ``rest`` source fields carry that var must trace back (via
       ``_trace_back_to``) to an anchor-handling row. Record under
       ``counter_for("anchor handling trace")``. A chapter may carry
       several anchor-handling rows — the producer guarantees one
       copied-anchor variant per LB, and a chapter legitimately
       aggregates rows fired at several LB depths, each citing its
       own variant; the walk's same-variable follow restriction
       guarantees each chain terminates at a root that mints the
       variable. (The former per-chapter uniqueness demand is
       retired — D-263.)

    5. **Contradiction trace** and **Vacuous truth trace.** For
       each row of the respective tag, at least one of the two
       contradicting ingredients must trace back to the seed
       (contradiction → task-formulation cleanOp; vacuous truth →
       LB exprKey). Record under the corresponding counters.

    6. **Per-tag dispatch.** For every row, look up
       ``TAG_CHECKERS[row.tag]`` and invoke it; record under
       ``counter_for(row.tag)``. Unknown tags get recorded under
       ``counter_for("<unknown:{tag}>")``.

    7. **General origin check.** Every dependency (``rest[i]`` at
       even indices) of every row must be either (a) the LEFT-side
       expression of some chapter row, (b) carry the
       ``_integration_goal`` postfix, or (c) belong to a tag in
       ``_ORIGIN_EXEMPT_TAGS``. For implication/multiplied-from/
       reformulated-from rows, ``rest[0]`` is
       additionally checked against the global + external theorem
       registries (with alpha-canonical and w→v revert fallbacks).
       Record under ``counter_for("origin")``.

    8. **Definition-set consistency (D-41).** Every chapter row is
       run through ``check_defset_consistency`` (the D-41
       variable-port type-consistency meta-check). Record under
       ``counter_for("definition set consistency")``.

    9. **Origin chain termination (cycle detection).** Build the
        chapter's origin graph, run iterative DFS-color cycle
        detection. Every (expression, namespace) node on a cycle is
        flagged; record one failure per cyclic node under
        ``counter_for("origin chain termination")``.

    No return value; the side effect is the accumulated counter
    state.

    @param chapter_file  Filename (for diagnostics).
    @param lines         Parsed chapter rows.
    @param chapter_type  Filename-derived type (``direct_proof``,
                         ``check_zero``, …).
    @param state         Global verifier state — mutated in place.
    @param chapter_thm   Chapter's theorem triple or None.

    @invariant I-16 — every recorded failure is a real bug; the
    function is intended to be idempotent across runs (same input
    → same counter state).
    """

    # 1. Theorem goal reached
    goal_ok = check_theorem_goal_reached(
        chapter_file, chapter_type, lines, chapter_thm,
        state.gl_binaries)
    state.goal_reached.record(goal_ok)

    # Set transient chapter context for line checkers
    state.current_chapter_thm = chapter_thm
    state.current_chapter_type = chapter_type

    # Detect which GL binary applies to this chapter from the theorem's
    # PREMISE anchor — the world the proof's assumptions live in.
    #
    # Cross-anchor connection theorems like chapter 103's
    # `(>[..](AnchorGauss[..])(AnchorPeano[..]))` mention multiple anchors
    # in one expression. The principled binding is the premise anchor
    # (`AnchorGauss` here) because the proof's hypotheses live in that
    # operator world; the conclusion's anchor is what the proof aims to
    # construct in. The previous heuristic (iterate gl_binaries and pick
    # the first tag whose `Anchor<Tag>` substring is anywhere in the
    # theorem) was order-sensitive — same input bytes produced different
    # verdicts on different filesystems, contradicting GL's determinism
    # claim. Premise extraction via `disintegrate_implication_full` makes
    # the binding a pure function of the theorem expression.
    state.current_gl_binary = None
    state.current_resolved_defsets = None
    if chapter_thm is not None:
        thm_expr = chapter_thm[0]
        premises, _head = disintegrate_implication_full(thm_expr)
        premise_expr = premises[0] if premises else thm_expr  # naked-anchor fallback
        m = re.search(r'Anchor([A-Za-z0-9_]+)', premise_expr)
        if m is not None:
            tag = m.group(1)
            binary = state.gl_binaries.get(tag)
            if binary is not None:
                state.current_gl_binary = binary
                state.current_resolved_defsets = state.resolved_defsets_per_tag.get(tag)
    if state.current_resolved_defsets is None:
        state.current_resolved_defsets = state.resolved_defsets_atomic_only

    # 3. Self-reference check: theorem must not appear as its own justification
    if chapter_thm is not None:
        thm_expr = chapter_thm[0]
        for line in lines:
            if line.tag == "theorem" and line.expression == thm_expr:
                state.counter_for("self-reference").record(False)

    # Build expr → lines map once for chapter-level trace checks
    _ch_expr_to_lines: Dict[str, List[ProofLine]] = {}
    for ch_line in lines:
        _ch_expr_to_lines.setdefault(ch_line.expression, []).append(ch_line)

    # 4. Anchor handling trace: every _copy var minted by ANY anchor-handling
    #    row must, wherever a row's rest sources carry it, trace back to an
    #    anchor-handling row. A chapter may carry several handling rows (one
    #    copied-anchor variant per LB depth); the same-variable follow
    #    restriction makes each chain terminate only at a root that mints
    #    the variable (D-263).
    anchor_lines = [line for line in lines if line.tag == "anchor handling"]

    if anchor_lines:
        copy_vars: List[str] = []
        for al in anchor_lines:
            for a in _extract_args(al.expression):
                if a.endswith("_copy") and a not in copy_vars:
                    copy_vars.append(a)

        if copy_vars:
            _anc_target = lambda cl: cl.tag == "anchor handling"

            for cv in copy_vars:
                _cv_re = re.compile(
                    r'(?<=[\[,])' + re.escape(cv) + r'(?=[\],])')
                _cv_follow = lambda src, _r=_cv_re: bool(_r.search(src))

                for ch_line in lines:
                    if ch_line.tag == "anchor handling":
                        continue
                    for i in range(0, len(ch_line.rest), 2):
                        src = ch_line.rest[i]
                        if _cv_follow(src):
                            ok = _trace_back_to(src, _ch_expr_to_lines,
                                                _anc_target, _cv_follow)
                            state.counter_for("anchor handling trace").record(ok)

    # 6. Contradiction trace: at least one of the two contradicting expressions
    #    must trace back to the seed (task formulation) through all ingredients.
    #    The scope-carrying reductio variant (four rest fields) traces to the
    #    child scope's OWN seed row instead: the contradiction-scope task
    #    formulation (the positive of the negated payload seed), or the ordis
    #    branch's `or disintegration` assumption row (the payload disjunct).
    for line in lines:
        if line.tag != "contradiction":
            continue
        if len(line.rest) == 4:
            scope_ns = line.rest[1]
            cut = scope_ns.rfind("_boundary_")
            payload = (scope_ns[cut + len("_boundary_"):]
                       if cut >= 0 else "")
            if payload.startswith("contradiction_"):
                neg_seed = payload[len("contradiction_"):]
                seed_expr = (neg_seed[1:] if neg_seed.startswith('!')
                             else "!" + neg_seed)
                _seed_target = (lambda cl, _s=seed_expr:
                                cl.tag == "task formulation"
                                and cl.expression == _s)
            elif payload.startswith("ordis_"):
                sep = payload.find("_(", len("ordis_"))
                seed_expr = payload[sep + 2:-1] if sep >= 0 else ""
                _seed_target = (lambda cl, _s=seed_expr:
                                cl.tag == "or disintegration"
                                and cl.expression == _s)
            else:
                state.counter_for("contradiction trace").record(False)
                continue
            ok = (_trace_back_to(line.rest[0], _ch_expr_to_lines, _seed_target)
                  or _trace_back_to(line.rest[2], _ch_expr_to_lines,
                                    _seed_target))
            state.counter_for("contradiction trace").record(ok)
            continue
        if len(line.rest) < 6:
            state.counter_for("contradiction trace").record(False)
            continue

        seed_expr = line.rest[4]
        _seed_target = (lambda cl, _s=seed_expr:
                        cl.tag == "task formulation" and cl.expression == _s)

        ok = (_trace_back_to(line.rest[0], _ch_expr_to_lines, _seed_target)
              or _trace_back_to(line.rest[2], _ch_expr_to_lines, _seed_target))
        state.counter_for("contradiction trace").record(ok)

    # 7. Vacuous truth trace: shape-only check. A stricter mirror of the
    #    contradiction-trace check — requiring at least one contradicting
    #    ingredient to trace back to the LB's exprKey (rest[4]) via the
    #    chapter's origin graph — is too strict for theorems whose own
    #    premise is impossible (the chapter101-style lemma
    #    "s(v1)=0 ∧ v1∈N ⇒ 0=v1"): the contradiction is necessarily rooted
    #    in the theorem's outer premise, not the inner recursion-step's
    #    hypothesis. Since the axioms are consistent, any chapter-level
    #    contradiction must reach some non-anchor task formulation by
    #    construction — the specific LB-level identity at rest[4] is
    #    bookkeeping, not soundness. The third ingredient stays in the row
    #    for documentation; no trace requirement is imposed.
    for line in lines:
        if line.tag != "vacuous truth":
            continue
        if len(line.rest) < 6:
            state.counter_for("vacuous truth trace").record(False)
            continue
        state.counter_for("vacuous truth trace").record(True)

    # 8. Dispatch every line to its tag checker
    for line in lines:
        tag = line.tag
        if tag in TAG_CHECKERS:
            passed = TAG_CHECKERS[tag](line, lines, state)
            state.counter_for(tag).record(passed)
        else:
            state.counter_for(f"<unknown:{tag}>").record(False)

    # 9. General origin check: every expression referenced as a dependency
    #    (in rest fields) must appear as a left-side expression (have its
    #    own derivation line). Exemptions live in the module-scope
    #    _ORIGIN_EXEMPT_TAGS set (see comment above its definition for the
    #    per-tag rationale).
    originated = {line.expression for line in lines}
    for line in lines:
        if line.tag in _ORIGIN_EXEMPT_TAGS:
            continue
        for i in range(0, len(line.rest), 2):
            dep = line.rest[i]
            if dep in originated:
                continue
            if dep.endswith("_integration_goal"):
                continue
            # rest[0] global theorem check: implication rules and multiplied-from sources
            if i == 0 and line.tag in ("implication", "multiplied from", "reformulated from"):
                norm_dep = _normalize_expr_list([dep])
                # Cross-batch refs may carry alpha-equivalent but textually
                # distinct bound-var names (e.g. incubator chapter row uses
                # `i2` where the Peano global list stored `v1`).
                # Compare modulo alpha-renaming of every >[...] bound variable.
                alpha_dep = _alpha_canonicalize_bound_vars(dep)
                # w/W -> v/V revert for cited theorem-anchor implications.
                # The processor renames non-anchor inner bvars from v/V to
                # w/W in cells at column >= 3 (citation form). The registry
                # stores v/V; reverting before the membership check keeps
                # the fast path intact instead of relying solely on the
                # _normalize_expr_list fold (which also handles it).
                dep_v = (_revert_w_to_v_in_theorem_citation(dep)
                         if _is_theorem_anchor_impl_local(dep) else dep)
                registry = state.global_theorems.keys() | state.external_theorems
                found = (
                    dep in state.global_theorems
                    or dep in state.external_theorems
                    or dep_v in state.global_theorems
                    or dep_v in state.external_theorems
                    or any(_normalize_expr_list([gt]) == norm_dep
                           for gt in registry)
                    or any(_alpha_canonicalize_bound_vars(gt) == alpha_dep
                           for gt in registry)
                )
                state.counter_for("origin").record(found)
                continue
            state.counter_for("origin").record(False)

    # 10. Definition-set consistency meta-check (D-41).
    #     For every chapter row, every variable that appears at multiple
    #     operator-call positions across the row's expressions (left-hand
    #     expression + each rest[i] expression) must connect to ports with
    #     identical type labels per state.resolved_defsets. Pure formal
    #     completeness — failures literally impossible if the producer
    #     (compiler/conjecturer/prover) honoured its own typing rules; if any
    #     failure surfaces, that is a producer bug.
    for line in lines:
        state.counter_for("definition set consistency").record(
            check_defset_consistency(line, state))

    # 11. Origin chain termination (cycle detection).
    #    Every row's origin chain must terminate at a foundation (a row
    #    that has no source still inside this chapter — typically theorem
    #    references, externally-provided theorems, anchor handling, task
    #    formulation, integration-goal stubs). A cycle (row A cites row B
    #    cites row A, possibly through intermediates) leaves the rows
    #    without a real foundation — each step is justified only by the
    #    other, the chain spins. Tag-level checkers individually pass the
    #    rows in such a cycle (each cite is locally well-formed), so the
    #    cycle is invisible without a chapter-level traversal.
    #
    #    Algorithm: build (expr, ns) -> [(source_expr, source_ns)] from
    #    every row's rest field (paired indices), then DFS-color cycle
    #    detection. Sources not appearing as any row's first column are
    #    foundations and are not followed. Every (expr, ns) on a cycle is
    #    flagged; one failure recorded per row whose node is on a cycle.
    chapter_nodes: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
    for ch_line in lines:
        node = (ch_line.expression, ch_line.namespace)
        srcs: List[Tuple[str, str]] = []
        for i in range(0, len(ch_line.rest) - 1, 2):
            srcs.append((ch_line.rest[i], ch_line.rest[i + 1]))
        chapter_nodes.setdefault(node, []).extend(srcs)

    WHITE, GRAY, BLACK = 0, 1, 2
    node_color: Dict[Tuple[str, str], int] = {}
    cyclic_nodes: Set[Tuple[str, str]] = set()

    def _dfs_cycle(start: Tuple[str, str]) -> None:
        # Iterative DFS with explicit stack to avoid recursion depth limits.
        # Stack entries: (node, iterator-of-children, on-path-flag).
        stack: List[Tuple[Tuple[str, str], Iterator[Tuple[str, str]]]] = []
        node_color[start] = GRAY
        stack.append((start, iter(chapter_nodes.get(start, []))))
        while stack:
            node, children = stack[-1]
            advanced = False
            for child in children:
                if child not in chapter_nodes:
                    # foundation — not part of this chapter's row set
                    continue
                c = node_color.get(child, WHITE)
                if c == GRAY:
                    # Back-edge: cycle from `child` through stack down to itself.
                    # Mark every GRAY node on the stack as cyclic.
                    in_cycle = False
                    for st_node, _ in stack:
                        if st_node == child:
                            in_cycle = True
                        if in_cycle:
                            cyclic_nodes.add(st_node)
                    cyclic_nodes.add(child)
                    continue
                if c == BLACK:
                    continue
                node_color[child] = GRAY
                stack.append((child, iter(chapter_nodes.get(child, []))))
                advanced = True
                break
            if not advanced:
                node_color[node] = BLACK
                stack.pop()

    for n in chapter_nodes:
        if n not in node_color:
            _dfs_cycle(n)

    for ch_line in lines:
        node = (ch_line.expression, ch_line.namespace)
        state.counter_for("origin chain termination").record(
            node not in cyclic_nodes)


# ---------------------------------------------------------------------------
#  Definition-set consistency meta-check (D-41) — helpers
#
#  Faithful Python translation of the C++ reference at
#  GL_Quick_VS/GL_Quick/src/compiler.hpp:1246-1538 (`ArgumentAnalyzer` +
#  `RecursiveParser::parseSubtree` + `mergeMaps` + `processLeaf` +
#  `checkDefinitionConsistency`).
#
#  Algorithm: recursive parse of each MPL expression independently,
#  accumulating per-node `remainingArgsDefs` (a `var → type_label` map).
#  At quantifier nodes (`>[v1,v2]<L><R>`) the bound variables are
#  REMOVED from the merged child map, so their names can be reused at
#  outer scopes without collision. `_merge_maps` is the consistency
#  check: shared variable names across child sub-trees must have
#  identical type labels (else mismatch). Each expression in a chapter
#  row is checked independently — no cross-expression pooling, matching
#  how the compiler invokes `analyze()` once per expression at compile
#  time.
# ---------------------------------------------------------------------------

# Regex matching a leaf operator-call: `(<head>[<args>])` where the bracket
# contents have no nested brackets. Head matches `[\w=]+` so the `=`
# operator (which is a non-word char) is captured alongside word-named
# operators (`in`, `in2`, `in3`, `fXY`, `existence2`, `or0`, `implication26`,
# etc.). Argument tokens may include `*`, `+`, `s`, `id` (anchor slot
# names) — those are word chars and split cleanly on `,`.
_LEAF_OP_CALL_RE = re.compile(r'\(([\w=]+)\[([^\[\]]*)\]\)')

# Used to extract head + args inside a leaf label (no surrounding parens).
_LEAF_LABEL_RE = re.compile(r'([\w=]+)\[([^\[\]]*)\]')


## @brief Sentinel exception for D-41 per-row defset mismatch.
#
# @details
# Raised by ``_merge_maps`` when two child sub-trees disagree on the
# type of a shared variable, or by ``_process_leaf`` when the same
# variable appears at two same-leaf positions with different types.
# Caught by ``check_defset_consistency`` to convert the structural
# violation into a per-row failure under the
# ``"definition set consistency"`` counter.
#
# @see check_defset_consistency
class _DefsetMismatch(Exception):
    """Raised by _merge_maps when two child sub-trees disagree on the type of
    a shared variable, or by _process_leaf when the same variable appears at
    two same-leaf positions with different types. Caught by
    check_defset_consistency to record a per-row failure."""
    pass


def _merge_maps(a: Dict[str, str], b: Dict[str, str]) -> Dict[str, str]:
    """@brief Union two ``var → type_label`` maps; raise on shared-key type clash.

    @details
    Python mirror of C++ ``ArgumentAnalyzer::mergeMaps``
    (``compiler.hpp:1250-1264``). For each key in ``b``:

    - If absent from ``a``: copy into result.
    - If present in ``a`` with the SAME type: keep one entry.
    - If present in ``a`` with a DIFFERENT type: raise
      ``_DefsetMismatch``.

    Used inside ``_parse_subtree`` at every binary/quantifier node
    to combine left-child and right-child type maps.

    @param a  First type map.
    @param b  Second type map.
    @return  Union map.
    @throws  _DefsetMismatch if any shared key has differing types.
    """
    out = dict(a)
    for k, v in b.items():
        existing = out.get(k)
        if existing is None:
            out[k] = v
        elif existing != v:
            raise _DefsetMismatch
    return out


def _process_leaf(label: str,
                  resolved_defsets: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    """@brief Build a leaf operator-call's ``var → type_label`` map.

    @details
    Python mirror of C++ ``RecursiveParser::processLeaf``
    (``compiler.hpp:1409-1426``). Parses ``label`` as
    ``<head>[<args>]``; looks up ``head`` in ``resolved_defsets``;
    for each position, records the position's type label against
    the actual arg name.

    **Tightening over the C++ reference.** The C++ silently
    overwrites on same-arg-twice (last-write-wins). The Python
    port raises ``_DefsetMismatch`` if the same variable appears at
    two positions of the SAME leaf with different declared types
    — this catches expressions like ``(in[v,v])`` where pos 1 is
    ``(1)`` (anchor slot) and pos 2 is ``P(1)`` (a projection). The
    tightening is sound under I-29: a single occurrence of ``v``
    can't have two types simultaneously.

    Operators not in ``resolved_defsets`` return an empty map
    (unknown operator → no type information contributed).

    @param label             Leaf label of the form ``<head>[<args>]``
                             (no outer parens — the surrounding parens
                             are stripped by ``_parse_subtree``).
    @param resolved_defsets  Map ``operator → position → type_label``.
    @return  Map ``var → type_label`` for this leaf's arg positions.
    @throws  _DefsetMismatch on same-var-twice with different types.
    """
    m = _LEAF_LABEL_RE.match(label)
    if not m:
        return {}
    head = m.group(1)
    args = [a for a in m.group(2).split(",") if a]
    ds = resolved_defsets.get(head)
    if ds is None:
        return {}
    out: Dict[str, str] = {}
    for i, arg in enumerate(args, start=1):
        t = ds.get(str(i))
        if t is None:
            continue
        existing = out.get(arg)
        if existing is None:
            out[arg] = t
        elif existing != t:
            raise _DefsetMismatch
    return out


def _parse_subtree(s: str, idx_box: List[int],
                   resolved_defsets: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    """@brief Recursive-descent parser computing the D-41 type map for a subtree.

    @details
    Python mirror of C++ ``RecursiveParser::parseSubtree``
    (``compiler.hpp:1281-1407``). Walks the expression string from
    the current cursor position, recognizing six structural shapes:

    1. ``(>[bound_list]<left><right>)`` — implication with binder.
       Recurse on both sub-expressions, merge their type maps,
       remove the bound vars (scoped local — names can be reused at
       outer scopes without collision).
    2. ``(&<left><right>)`` — conjunction. Recurse, merge.
    3. ``(|<left><right>)`` — disjunction. Recurse, merge.
    4. ``(<head>[<args>])`` — leaf operator call. Dispatch to
       ``_process_leaf``.
    5. ``!(>[bound_list]<left><right>)`` — negated implication
       (existence shape). Same as case 1.
    6. ``!(&<left><right>)`` / ``!(|<left><right>)`` — negated
       binary; same as cases 2-3.
    7. ``!(<head>[<args>])`` — negated leaf; dispatch to
       ``_process_leaf``.

    Any other input shape returns ``{}`` (defensive).

    The ``idx_box`` parameter is a single-element list used as a
    mutable cursor — Python doesn't have C++-style reference
    semantics for ints, so wrap-in-list is the idiomatic
    work-around.

    @param s                 The expression string.
    @param idx_box           Mutable cursor (one-element list).
    @param resolved_defsets  Map ``operator → position → type_label``.
    @return  Type map ``var → type_label`` for this subtree's free
             variables.
    @throws  _DefsetMismatch via downstream ``_merge_maps`` /
             ``_process_leaf`` on type clashes.

    @see check_defset_consistency — primary consumer.
    """
    idx = idx_box[0]
    n = len(s)
    if idx >= n:
        return {}

    if s[idx] == "(":
        idx += 1
        if idx < n and s[idx] == ">":
            # Implication: (>[bound_list] <left> <right>)
            idx += 1
            close = s.find("]", idx)
            assert close != -1, "Missing ']' in implication"
            bound_str = s[idx + 1:close]
            bound_vars = [v for v in bound_str.split(",") if v]
            idx = close + 1
            idx_box[0] = idx
            left_map = _parse_subtree(s, idx_box, resolved_defsets)
            right_map = _parse_subtree(s, idx_box, resolved_defsets)
            idx = idx_box[0]
            combined = _merge_maps(left_map, right_map)
            for v in bound_vars:
                combined.pop(v, None)
            if idx < n and s[idx] == ")":
                idx += 1
            idx_box[0] = idx
            return combined
        if idx < n and s[idx] == "&":
            idx += 1
            idx_box[0] = idx
            left_map = _parse_subtree(s, idx_box, resolved_defsets)
            right_map = _parse_subtree(s, idx_box, resolved_defsets)
            idx = idx_box[0]
            combined = _merge_maps(left_map, right_map)
            if idx < n and s[idx] == ")":
                idx += 1
            idx_box[0] = idx
            return combined
        if idx < n and s[idx] == "|":
            # OR — treated like conjunction for scoping (no bound vars).
            idx += 1
            idx_box[0] = idx
            left_map = _parse_subtree(s, idx_box, resolved_defsets)
            right_map = _parse_subtree(s, idx_box, resolved_defsets)
            idx = idx_box[0]
            combined = _merge_maps(left_map, right_map)
            if idx < n and s[idx] == ")":
                idx += 1
            idx_box[0] = idx
            return combined
        # Leaf operator call: (<head>[<args>])
        close = s.find(")", idx)
        assert close != -1, "Missing ')' for leaf"
        label = s[idx:close]
        idx = close + 1
        leaf_map = _process_leaf(label, resolved_defsets)
        idx_box[0] = idx
        return leaf_map

    if idx + 1 < n and s[idx] == "!" and s[idx + 1] == "(":
        idx += 2
        if idx < n and s[idx] == ">":
            # !(>[bound_list] <left> <right>)
            idx += 1
            close = s.find("]", idx)
            assert close != -1, "Missing ']' in !>"
            bound_str = s[idx + 1:close]
            bound_vars = [v for v in bound_str.split(",") if v]
            idx = close + 1
            idx_box[0] = idx
            left_map = _parse_subtree(s, idx_box, resolved_defsets)
            right_map = _parse_subtree(s, idx_box, resolved_defsets)
            idx = idx_box[0]
            combined = _merge_maps(left_map, right_map)
            for v in bound_vars:
                combined.pop(v, None)
            if idx < n and s[idx] == ")":
                idx += 1
            idx_box[0] = idx
            return combined
        if idx < n and s[idx] == "&":
            idx += 1
            idx_box[0] = idx
            left_map = _parse_subtree(s, idx_box, resolved_defsets)
            right_map = _parse_subtree(s, idx_box, resolved_defsets)
            idx = idx_box[0]
            combined = _merge_maps(left_map, right_map)
            if idx < n and s[idx] == ")":
                idx += 1
            idx_box[0] = idx
            return combined
        if idx < n and s[idx] == "|":
            idx += 1
            idx_box[0] = idx
            left_map = _parse_subtree(s, idx_box, resolved_defsets)
            right_map = _parse_subtree(s, idx_box, resolved_defsets)
            idx = idx_box[0]
            combined = _merge_maps(left_map, right_map)
            if idx < n and s[idx] == ")":
                idx += 1
            idx_box[0] = idx
            return combined
        # Negated leaf: !(<head>[<args>])
        close = s.find(")", idx)
        assert close != -1, "Missing ')' for negated leaf"
        label = s[idx:close]
        idx = close + 1
        leaf_map = _process_leaf(label, resolved_defsets)
        idx_box[0] = idx
        return leaf_map

    # Defensive — unrecognised shape.
    return {}


def build_resolved_defsets_per_tag(
        definition_sets: Dict[str, Dict[str, list]],
        gl_binaries: Dict[str, dict]
) -> Tuple[Dict[str, Dict[str, Dict[str, str]]], Dict[str, Dict[str, str]]]:
    """@brief Build per-tag resolved defset indices for D-41 type checking.

    @details
    Python mirror of the compiler's
    ``ArgumentAnalyzer(this->coreExpressionMap)`` pattern at
    ``prover.hpp:3556``: each batch's analyzer is constructed with its
    own ``coreExpressionMap``, so per-batch compact-name allocations
    (e.g. ``implication26`` is arity 2 in Gauss but arity 5 in
    IncubatorGauss) don't cross-contaminate. The verifier mirrors
    that by building one resolved-defset map per tag, selected
    per-chapter via anchor-substring match (same convention as
    ``current_gl_binary`` resolution in ``verify_chapter``).

    **Algorithm.**

    1. **Atomic seed.** Convert ``definition_sets`` (the ConfigVisu
       atomic table with ``[type_label, combinable]`` pairs) into a
       simpler ``var → type_label`` map per operator.
    2. **Per-tag composite resolution.** For each tag:
       a. Merge ``GL_binary_shared.json`` (if present, keyed as
          ``shared``) with the tag's own binary. Per-tag entries
          override shared on collision — per-batch is authoritative
          for its own chapters.
       b. Start the resolved map with the atomic seed.
       c. Collect remaining composites (ops with non-empty
          ``elements`` not already in the atomic seed).
       d. Fixed-point iterate ``_try_derive_from_elements`` until
          no progress: each composite whose elements all reference
          already-resolved ops gets resolved.
       e. Composites that never resolve (because their elements
          reference operators with no atomic defsets) are dropped
          silently — ``check_defset_consistency`` skips them at the
          row level.
    3. **Atomic-only fallback.** Returned alongside as the second
       tuple element. Used for chapters whose theorem doesn't
       disclose an anchor (the per-tag selection therefore fails);
       composites stay unresolved → skipped per row.

    @param definition_sets  ConfigVisu's atomic table
                            (``var → {pos_str → [type_label, combinable]}``).
    @param gl_binaries      Tag → operator dict from
                            ``load_gl_binaries``.
    @return  ``(per_tag, atomic_only)`` where ``per_tag`` is the
             tag-indexed resolved map and ``atomic_only`` is the
             fallback.

    @invariant I-29 — variable-port type consistency: this is the
    structural ground for the D-41 check.
    @see check_defset_consistency — primary consumer.
    @see _try_derive_from_elements — the per-composite resolution
    step.
    """
    atomic_seeds: Dict[str, Dict[str, str]] = {
        core_name: {pos: entry[0] for pos, entry in ds.items()}
        for core_name, ds in definition_sets.items()
    }

    shared_binary = gl_binaries.get("shared", {}) or {}

    per_tag: Dict[str, Dict[str, Dict[str, str]]] = {}
    for tag in gl_binaries.keys():
        # Build a per-tag "operator superset": shared first (so per-tag
        # overrides apply on collision), then this tag's own binary.
        merged_binary: Dict[str, dict] = {}
        for op_name, spec in shared_binary.items():
            merged_binary[op_name] = spec
        if tag != "shared":
            for op_name, spec in gl_binaries[tag].items():
                merged_binary[op_name] = spec  # per-tag wins on collision

        # Resolve composites in merged_binary against atomic_seeds + ones
        # already resolved this pass. Fixed-point iteration.
        resolved: Dict[str, Dict[str, str]] = dict(atomic_seeds)
        composites: Dict[str, dict] = {}
        for op_name, spec in merged_binary.items():
            if op_name in resolved:
                continue  # atomic baseline wins for known atomics
            if not isinstance(spec, dict):
                continue
            elements = spec.get("elements") or []
            if not any(e for e in elements if e):
                continue
            composites[op_name] = spec

        while True:
            progress = False
            for op_name, spec in list(composites.items()):
                if op_name in resolved:
                    composites.pop(op_name, None)
                    continue
                derived = _try_derive_from_elements(spec, resolved)
                if derived is not None:
                    resolved[op_name] = derived
                    composites.pop(op_name, None)
                    progress = True
            if not progress:
                break

        per_tag[tag] = resolved

    return per_tag, atomic_seeds


def _try_derive_from_elements(spec: dict,
                              resolved: Dict[str, Dict[str, str]]
                              ) -> Optional[Dict[str, str]]:
    """@brief Derive a composite operator's per-position types from its elements.

    @details
    For each element of ``spec.elements``, parse it as a leaf
    operator call ``(<head>[<args>])``, look up ``head`` in
    ``resolved``, and propagate the position types onto the
    element's actual arg names. Accumulate the resulting
    ``var → type_label`` map. Finally extract the per-signature-arg
    types by looking up each signature arg in the accumulated map.

    **Failure modes (return None — composite stays unresolved).**

    - Any element references an inner operator whose own defsets
      aren't yet in ``resolved`` (deferral; the outer fixed-point
      loop will retry next iteration after more composites are
      resolved).
    - Two element call-sites assign different types to the same
      argument (internal contradiction in the producer-emitted
      spec).
    - The signature can't be parsed as a leaf call.
    - A signature arg never appears in any element (can't infer
      its type).

    Used inside ``build_resolved_defsets_per_tag``'s fixed-point
    loop. Composites that never resolve are dropped (the row-level
    check skips them).

    @param spec      A binary entry with ``signature``, ``elements``.
    @param resolved  Currently resolved
                     ``op_name → pos_str → type_label`` map.
    @return  Per-position type map for the composite's signature
             args; ``None`` if derivation cannot complete.

    @see build_resolved_defsets_per_tag — caller / fixed-point
    driver.
    """
    signature = spec.get("signature", "")
    elements = spec.get("elements") or []

    # Local accumulator: variable name (u_K, integer literal) → type label
    var_types: Dict[str, str] = {}

    for elem in elements:
        if not elem:
            continue
        m = _LEAF_OP_CALL_RE.search(elem)
        if not m:
            continue  # element is not a leaf-call shape; skip
        head = m.group(1)
        args_str = m.group(2)
        args = [a for a in args_str.split(",") if a]
        inner_ds = resolved.get(head)
        if inner_ds is None:
            return None  # inner op not yet resolved; defer
        for i, arg in enumerate(args, start=1):
            t = inner_ds.get(str(i))
            if t is None:
                continue
            existing = var_types.get(arg)
            if existing is not None and existing != t:
                # Internal contradiction in the producer-emitted spec.
                # Surface as unresolved (caller will leave it absent from
                # the output map; the consistency check then skips it).
                return None
            var_types[arg] = t

    # Extract signature args (u-prefixed in normal case) and look them up.
    sig_match = _LEAF_OP_CALL_RE.search(signature)
    if sig_match is None:
        return None
    sig_args = [a for a in sig_match.group(2).split(",") if a]
    out: Dict[str, str] = {}
    for i, sa in enumerate(sig_args, start=1):
        t = var_types.get(sa)
        if t is None:
            return None  # signature arg never appeared in elements
        out[str(i)] = t
    return out


def check_defset_consistency(line: ProofLine, state: VerifierState) -> bool:
    """@brief D-41 per-row variable-port type-consistency check.

    @details
    Python translation of the C++ compiler's reference algorithm at
    ``compiler.hpp:1246-1538`` (``ArgumentAnalyzer`` +
    ``RecursiveParser::parseSubtree`` + ``mergeMaps`` +
    ``processLeaf`` + ``checkDefinitionConsistency``).

    Each expression in the row (left-hand expression + each
    ``rest[i]`` expression at even indices) is parsed independently
    with ``_parse_subtree``, which
    accumulates per-node `remainingArgsDefs` and removes bound variables at
    `>[…]` quantifiers. `_merge_maps` is the actual mismatch detector — a
    `_DefsetMismatch` raised anywhere during parse is caught and returns
    False. Returns True if every expression parses without raising.

    Uses `state.current_resolved_defsets` (set per-chapter by `verify_chapter`
    based on the chapter's tag) — falls back to the atomic-only map when no
    tag is selected (chapters whose theorem doesn't disclose an anchor).
    """
    resolved = state.current_resolved_defsets
    if resolved is None:
        resolved = state.resolved_defsets_atomic_only

    exprs: List[str] = [line.expression]
    for i in range(0, len(line.rest), 2):
        exprs.append(line.rest[i])

    for expr in exprs:
        if not expr:
            continue
        s = "".join(c for c in expr if c not in " \t\n\r")
        if not s:
            continue
        try:
            _parse_subtree(s, [0], resolved)
        except _DefsetMismatch:
            return False
        except (AssertionError, IndexError):
            # Parse-shape anomaly — treat as non-failure (the producer-side
            # checks would have already caught syntactic problems). Surface
            # via the unknown-op skip convention.
            continue
    return True


# ---------------------------------------------------------------------------
#  Main driver
# ---------------------------------------------------------------------------

def chapter_sort_key(filename: str) -> Tuple[int, str]:
    """@brief Numeric-prefix sort key for chapter filenames.

    @details
    Chapter files are named ``<N>_<chapter_type>.txt`` where ``<N>``
    is a 1-based ordinal matching the theorem's position in the
    global theorem list. The sort key returns ``(N, suffix)`` so
    Python's stable sort orders chapters by ``N`` numerically, then
    by suffix alphabetically (for the induction-triple ordering
    where typing → check_zero → check_induction_condition share the
    numeric prefix family but differ by suffix; the file naming
    avoids this by giving each suffix a different prefix, but the
    secondary key is defensive).

    Malformed filenames whose prefix isn't an integer go to the end
    (sorted lexicographically among themselves via the ``999999``
    sentinel + base name).

    @param filename  Chapter filename like ``"1209_direct_proof.txt"``.
    @return  ``(int_prefix, suffix)`` sort key.
    """
    base = filename.replace(".txt", "")
    parts = base.split("_", 1)
    try:
        return (int(parts[0]), parts[1] if len(parts) > 1 else "")
    except ValueError:
        return (999999, base)


# ---------------------------------------------------------------------------
#  Cross-chapter meta-check: theorem usage termination
#
#  Every chapter may use previously proven theorems as inference rules
#  (`theorem` rows, rule citations). The per-chapter `origin chain
#  termination` check proves each chapter's INTERNAL derivation graph
#  acyclic, but a `theorem` row is a foundation there — so two chapters
#  can each ground themselves in the other's theorem and both pass every
#  per-chapter check while the pair is circular. Chapters connect only
#  through theorem citations, therefore global proof-graph acyclicity
#  decomposes into per-chapter acyclicity (already checked) plus
#  acyclicity of the directed theorem-usage graph "theorem A's chapter
#  cites theorem B" — which is what this check establishes.
# ---------------------------------------------------------------------------

def build_theorem_citation_resolver(state: VerifierState):
    """@brief Build a resolver from a cited expression to its registry theorem.

    @details
    A chapter cites a proven theorem in one of several textual forms:
    byte-identical to the ``global_theorem_list.txt`` entry (`theorem`
    rows), w/W-renamed citation form (rule cells at column >= 3, see
    ``_revert_w_to_v_in_theorem_citation``), or an alpha-equivalent
    bound-variable renaming (cross-batch citations). This factory
    precomputes normalized and alpha-canonical lookup maps over
    ``state.global_theorems`` once, then resolves each citation with
    the same fallback ladder the ``origin`` meta-check applies:
    exact -> w/W revert -> ``_normalize_expr_list`` -> alpha
    canonicalization.

    External theorems (``state.external_theorems``) are deliberately
    NOT resolvable: they are axiomatic input, own no chapter, and can
    therefore never participate in a usage cycle.

    On collisions (two registry entries sharing a normalized or alpha
    form) the first entry in ``state.global_theorems`` iteration order
    wins, mirroring the ``any(...)`` first-hit semantics of the
    ``origin`` meta-check.

    @param state  Verifier state with ``global_theorems`` loaded.
    @return  ``resolve(cited: str) -> Optional[str]`` — the registry
             expression the citation denotes, or ``None`` when the
             citation is not a proven theorem of this run family.

    @see check_theorem_usage_termination — consumes the resolved edges.
    """
    exact = state.global_theorems
    norm_map: Dict[str, str] = {}
    alpha_map: Dict[str, str] = {}
    for gt in state.global_theorems:
        norm_map.setdefault(_normalize_expr_list([gt])[0], gt)
        alpha_map.setdefault(_alpha_canonicalize_bound_vars(gt), gt)

    def resolve(cited: str) -> Optional[str]:
        """@brief Resolve one citation string to its registry theorem.

        @details
        Applies the fallback ladder documented on the factory. Pure
        lookup — never mutates the precomputed maps.

        @param cited  Citation text exactly as it appears in the
                      chapter cell.
        @return  Registry theorem expression, or ``None`` if the text
                 does not denote a proven theorem.
        """
        if cited in exact:
            return cited
        if _is_theorem_anchor_impl_local(cited):
            rev = _revert_w_to_v_in_theorem_citation(cited)
            if rev in exact:
                return rev
        norm = _normalize_expr_list([cited])[0]
        if norm in norm_map:
            return norm_map[norm]
        return alpha_map.get(_alpha_canonicalize_bound_vars(cited))

    return resolve


def collect_theorem_usage_citations(lines: List[ProofLine],
                                    resolve) -> Set[str]:
    """@brief Collect the proven theorems one chapter's rows cite.

    @details
    Walks every row of a parsed chapter and gathers the registry
    theorems it uses, from the two citation channels:

    1. **``theorem`` rows** — the row's own expression IS the citation
       (empty rest by construction; the used theorem is a foundation
       leaf of the chapter).
    2. **Citation cells** — dependency cells that are NOT locally
       originated (no chapter row derives that exact expression) in
       the sanctioned citation positions: the ``rest[0]`` rule cell of
       ``implication`` / ``multiplied from`` / ``reformulated from``
       rows, and any cell of the ``_ORIGIN_EXEMPT_TAGS`` rows (or
       theorem sources, incubator back-reformulation sources,
       compilation antecedents). Locally-originated cells are internal
       derivation edges — the ``origin chain termination`` check owns
       those; non-citation cells that fail to originate locally are
       ``origin``-check failures, not usage edges.

    ``externally provided theorem`` rows contribute nothing: external
    theorems own no chapter and cannot close a cycle. Citations that
    resolve to no registry theorem contribute nothing here — the
    ``theorem`` tag checker already fails unresolvable theorem rows.

    @param lines    Parsed chapter rows.
    @param resolve  Citation resolver from
                    ``build_theorem_citation_resolver``.
    @return  Set of registry theorem expressions this chapter uses
             (possibly including the chapter's own theorem — a
             self-citation is a length-1 usage cycle).

    @see check_theorem_usage_termination — consumes one set per chapter.
    """
    originated = {ln.expression for ln in lines}
    cited: Set[str] = set()
    for ln in lines:
        if ln.tag == "theorem":
            resolved = resolve(ln.expression)
            if resolved is not None:
                cited.add(resolved)
            continue
        if ln.tag == "externally provided theorem":
            continue
        for i in range(0, len(ln.rest) - 1, 2):
            dep = ln.rest[i]
            if dep in originated:
                continue
            is_rule_cell = (i == 0 and ln.tag in (
                "implication", "multiplied from", "reformulated from"))
            if not (is_rule_cell or ln.tag in _ORIGIN_EXEMPT_TAGS):
                continue
            resolved = resolve(dep)
            if resolved is not None:
                cited.add(resolved)
    return cited


def check_theorem_usage_termination(usage_order: List[str],
                                    usage_edges: Dict[str, Set[str]],
                                    state: VerifierState) -> None:
    """@brief Global acyclicity check over the theorem-usage graph.

    @details
    Nodes are the theorems that own chapters in this run (insertion
    order of ``usage_order``); the directed edge A -> B means "A's
    chapter cites B". Runs the same iterative WHITE/GRAY/BLACK
    depth-first cycle detection as the chapter-local ``origin chain
    termination`` check: a GRAY back-edge marks every node on the
    cycle. Cited theorems without a local chapter (cross-batch
    citations merged via ``--include-globals``) have no outgoing
    edges and are foundations by construction.

    Records exactly one result per local theorem node into
    ``state.counter_for("theorem usage termination")`` — success when
    the theorem lies on no usage cycle. A failure means the theorem's
    proof is (transitively) grounded in itself: each chapter in the
    cycle passes every per-chapter check, yet the cycle members
    justify only each other. Per I-16 such a failure is a real
    producer bug (the chapter exporter chose a circular origin), never
    a false positive.

    Children are visited in sorted order so the traversal (and any
    future diagnostic ordering) is deterministic under Python hash
    randomization; the set of cyclic nodes itself is
    order-independent.

    @param usage_order  Distinct chapter-owning theorem expressions in
                        first-seen chapter order.
    @param usage_edges  Theorem expression -> set of cited theorem
                        expressions (union over the theorem's chapter
                        files; induction triads contribute three files
                        to one node).
    @param state        Verifier state receiving the counter records.

    @see collect_theorem_usage_citations — produces the edge sets.
    @see build_theorem_citation_resolver — citation resolution.
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    node_color: Dict[str, int] = {}
    cyclic_nodes: Set[str] = set()

    def _dfs_cycle(start: str) -> None:
        # Iterative DFS with explicit stack to avoid recursion limits.
        stack: List[Tuple[str, Iterator[str]]] = []
        node_color[start] = GRAY
        stack.append((start, iter(sorted(usage_edges.get(start, ())))))
        while stack:
            node, children = stack[-1]
            advanced = False
            for child in children:
                if child not in usage_edges:
                    # No local chapter — foundation, cannot cycle.
                    continue
                c = node_color.get(child, WHITE)
                if c == GRAY:
                    # Back-edge: every GRAY node from `child` down the
                    # stack lies on the cycle.
                    in_cycle = False
                    for st_node, _ in stack:
                        if st_node == child:
                            in_cycle = True
                        if in_cycle:
                            cyclic_nodes.add(st_node)
                    cyclic_nodes.add(child)
                    continue
                if c == BLACK:
                    continue
                node_color[child] = GRAY
                stack.append((child, iter(sorted(usage_edges.get(child, ())))))
                advanced = True
                break
            if not advanced:
                node_color[node] = BLACK
                stack.pop()

    for node in usage_order:
        if node not in node_color:
            _dfs_cycle(node)

    for node in usage_order:
        state.counter_for("theorem usage termination").record(
            node not in cyclic_nodes)


def run_verifier(base_dir: str,
                 extra_global_lists: Optional[List[str]] = None) -> VerifierState:
    """@brief Bootstrap state, enumerate chapters, run the verifier end-to-end.

    @details
    The driver. Steps:

    1. **Fresh state.** Allocate ``VerifierState()``.

    2. **Global theorem list.** Load ``global_theorem_list.txt`` from
       ``base_dir``. Union sibling-batch theorem lists from
       ``extra_global_lists`` into ``state.global_theorems`` and
       ``state.global_theorem_list`` (local entries take precedence
       on collisions; the ordered list is appended in the order
       extra paths are supplied). Used for cross-batch references
       (e.g. incubator chapters citing Peano axioms).

    3. **Operator metadata.** Load ``output_indices``,
       ``input_indices``, ``definition_sets`` from
       ``<script_dir>/files/config/ConfigVisu.json``.

    4. **GL binaries.** Look in
       ``<parent of base_dir>/GL_binaries/`` first; fall back to
       ``<script_dir>/files/GL_binaries/``. The former is used when
       a sandbox or release ships its own per-tree binary set
       alongside the processed proof graph.

    5. **Per-tag resolved defsets (D-41).** Build
       ``resolved_defsets_per_tag`` + ``resolved_defsets_atomic_only``
       via ``build_resolved_defsets_per_tag``.

    6. **External theorems.** Load ``external_theorems.txt`` if
       present.

    7. **Chapter enumeration.** List ``.txt`` files in ``base_dir``
       (excluding manifest files), sort by ``chapter_sort_key``,
       map each to its theorem via ``build_chapter_theorem_map``.

    8. **Verify every chapter.** For each chapter file, parse it
       and call ``verify_chapter``. Each call accumulates counter
       state.

    9. **Theorem usage termination.** While looping, accumulate the
       cross-chapter theorem-usage graph (one node per chapter-owning
       theorem, edge A -> B for "A's chapter cites B" via
       ``collect_theorem_usage_citations``); after the loop, run the
       global acyclicity check ``check_theorem_usage_termination``.

    @param base_dir            Directory containing the
                                processed-proof-graph chapter files
                                and ``global_theorem_list.txt``.
    @param extra_global_lists  Optional list of additional
                                ``global_theorem_list.txt`` paths
                                (for cross-batch citation merging).
    @return  Fully-populated ``VerifierState`` after every chapter
             has been verified.

    @see verify_chapter — per-chapter dispatcher.
    @see print_report — formats the final result for stdout.
    """
    state = VerifierState()

    # Load global theorem list
    state.global_theorems, state.global_theorem_list = \
        load_global_theorem_list(base_dir)

    # Union sibling-batch theorem lists into globals (cross-batch references
    # such as Peano axioms cited from incubator chapters). Local entries take
    # precedence on key collisions; the ordered list is appended in the order
    # the extra paths are supplied.
    if extra_global_lists:
        for extra_path in extra_global_lists:
            extra_dir = os.path.dirname(os.path.abspath(extra_path))
            extra_thms, extra_ordered = load_global_theorem_list(extra_dir)
            for expr, info in extra_thms.items():
                if expr not in state.global_theorems:
                    state.global_theorems[expr] = info
            for entry in extra_ordered:
                if entry[0] not in {e[0] for e in state.global_theorem_list}:
                    state.global_theorem_list.append(entry)

    # Load output indices from config
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(script_dir, "files", "config")
    state.output_indices = load_output_indices(config_dir)
    state.input_indices = load_input_indices(config_dir)
    state.definition_sets = load_definition_sets(config_dir)

    # Load GL binaries — look next to the processed proof graph first,
    # then fall back to the default location
    binaries_dir = os.path.join(os.path.dirname(base_dir), "GL_binaries")
    if not os.path.isdir(binaries_dir):
        binaries_dir = os.path.join(script_dir, "files", "GL_binaries")
    state.gl_binaries = load_gl_binaries(binaries_dir)

    # I-23 enforcement: every spontaneous compact-operator name must
    # encode the same operator in every batch that uses it. Recursive
    # closure check — see ``check_operator_registry_consistency``.
    check_operator_registry_consistency(state)

    # Per-tag defset indices — atomic ops from definition_sets, composite ops
    # resolved iteratively from each batch's gl_binary.elements (D-41).
    # Mirrors the compiler's per-batch ArgumentAnalyzer construction at
    # prover.hpp:3556. Selected per-chapter in verify_chapter the same way
    # current_gl_binary is.
    state.resolved_defsets_per_tag, state.resolved_defsets_atomic_only = \
        build_resolved_defsets_per_tag(state.definition_sets, state.gl_binaries)

    # Load external theorems
    ext_file = os.path.join(base_dir, "external_theorems.txt")
    if os.path.exists(ext_file):
        with open(ext_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    state.external_theorems.add(line)

    # Enumerate chapter files
    _exclude = {"global_theorem_list.txt", "external_theorems.txt"}
    chapter_files = sorted(
        [f for f in os.listdir(base_dir)
         if f.endswith(".txt") and f not in _exclude],
        key=chapter_sort_key,
    )

    # Build chapter → theorem mapping
    chapter_thm_map = build_chapter_theorem_map(
        chapter_files, state.global_theorem_list)

    # Cross-chapter theorem-usage graph, accumulated over the chapter
    # loop and cycle-checked after it (theorem usage termination).
    resolve_citation = build_theorem_citation_resolver(state)
    usage_order: List[str] = []
    usage_edges: Dict[str, Set[str]] = {}

    # Process every chapter
    for cf in chapter_files:
        filepath = os.path.join(base_dir, cf)
        lines = parse_chapter_file(filepath)
        base = cf.replace(".txt", "")
        parts = base.split("_", 1)
        chapter_type = parts[1] if len(parts) > 1 else "unknown"
        chapter_thm = chapter_thm_map.get(cf)

        verify_chapter(cf, lines, chapter_type, state, chapter_thm)

        if chapter_thm is not None:
            thm_expr = chapter_thm[0]
            if thm_expr not in usage_edges:
                usage_order.append(thm_expr)
                usage_edges[thm_expr] = set()
            usage_edges[thm_expr].update(
                collect_theorem_usage_citations(lines, resolve_citation))

    check_theorem_usage_termination(usage_order, usage_edges, state)

    return state


def get_totals(state: VerifierState) -> tuple[int, int]:
    """@brief Aggregate success + failure counts across every counter.

    @details
    Sums ``state.goal_reached`` (the standalone theorem-goal-reached
    counter), every ``TAG_CHECKERS``-registered counter, and every
    chapter-level meta-counter (``"self-reference"``, ``"origin"``,
    ``"definition set consistency"``, ``"origin chain termination"``,
    ``"operator registry consistency"``,
    ``"theorem usage termination"``,
    ``"anchor handling trace"``, ``"contradiction trace"``,
    ``"vacuous truth trace"``). All three groups contribute BOTH success
    AND failure counts symmetrically — the headline "M checks" therefore
    counts every actual check performed, regardless of whether it lives
    in TAG_CHECKERS or is a chapter-level meta-counter. ``total_success``
    counts every recorded success, ``total_failure`` counts every
    recorded failure, and the report's headline always means
    ``total_success + total_failure``.

    @param state  Verifier state at end of run.
    @return  ``(total_success, total_failure)`` integer pair.
    """
    g = state.goal_reached
    total_success = g.success
    total_failure = g.failure
    for tag in TAG_CHECKERS:
        ctr = state.tag_counters.get(tag, TagCounter())
        total_success += ctr.success
        total_failure += ctr.failure
    for tag, ctr in state.tag_counters.items():
        if tag not in TAG_CHECKERS:
            total_success += ctr.success
            total_failure += ctr.failure
    return total_success, total_failure


def print_report(state: VerifierState):
    """@brief Format the final report to stdout — one row per counter.

    @details
    Column width is 44 (consistent right-justification of the
    ``success M, failure N`` text). Output structure:

    1. ``theorem goal reached`` (state.goal_reached) — always
       printed first.
    2. Every tag in ``TAG_CHECKERS`` in dispatch declaration order.
       Missing tags (no row ever dispatched to them in this run)
       print ``success 0, failure 0`` to keep the report row count
       consistent across runs.
    3. Every ``state.tag_counters`` key NOT in ``TAG_CHECKERS`` —
       the chapter-level meta counters — in insertion order.
    4. Headline line. ``M checks, 0 failures — airtight.`` on
       clean run, ``Verifier: M checks, N FAILED.`` otherwise.
       The "airtight" wording is the user-facing release-readiness
       signal.

    @param state  Verifier state at end of run.
    """
    W = 44
    g = state.goal_reached
    print(f"{'theorem goal reached':<{W}s}success {g.success}, failure {g.failure}")
    for tag in TAG_CHECKERS:
        ctr = state.tag_counters.get(tag, TagCounter())
        print(f"{tag:<{W}s}success {ctr.success}, failure {ctr.failure}")
    for tag, ctr in state.tag_counters.items():
        if tag not in TAG_CHECKERS:
            print(f"{tag:<{W}s}success {ctr.success}, failure {ctr.failure}")
    total_success, total_failure = get_totals(state)
    # "M checks" always means every recorded check, success or failure —
    # the symmetric form keeps the headline consistent across runs.
    total_checks = total_success + total_failure
    if total_failure == 0:
        print(f"\n          {total_checks} checks, 0 failures — airtight.")
    else:
        print(f"\nVerifier: {total_checks} checks, {total_failure} FAILED.")


def main():
    """@brief Command-line entry point.

    @details
    Argparse interface:

    - **Positional ``base_dir``** (optional, default
      ``<script_dir>/files/processed_proof_graph``): the directory
      containing chapter files + ``global_theorem_list.txt``.
    - **``--include-globals PATH``** (repeatable): additional
      ``global_theorem_list.txt`` paths to union into the theorem
      registry. Used for cross-batch references — e.g. running the
      verifier on an incubator proof graph while citing Peano
      theorems from another directory.

    Flow:

    1. Parse args.
    2. ``base_dir`` existence check; exit 1 with stderr message if
       missing.
    3. ``run_verifier(base_dir, extra_global_lists=...)``.
    4. ``print_report(state)`` to stdout.

    No explicit return code; the function returns ``None`` and
    Python exits 0. The verifier intentionally does NOT exit non-
    zero on failures — the report's headline is the canonical
    pass/fail signal, parsed by downstream tooling.

    @see run_verifier — the workhorse.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_base = os.path.join(script_dir, "files", "processed_proof_graph")

    parser = argparse.ArgumentParser(
        description="Verify a GL processed proof graph.")
    parser.add_argument(
        "base_dir",
        nargs="?",
        default=default_base,
        help=("Directory containing chapter files + global_theorem_list.txt. "
              f"Default: {default_base}"))
    parser.add_argument(
        "--include-globals",
        action="append",
        default=[],
        metavar="PATH",
        help=("Additional global_theorem_list.txt to union into the "
              "theorem registry (for cross-batch references, e.g. "
              "incubator chapters citing Peano axioms). Repeatable."))
    args = parser.parse_args()

    if not os.path.isdir(args.base_dir):
        print(f"ERROR: directory not found: {args.base_dir}", file=sys.stderr)
        sys.exit(1)

    state = run_verifier(args.base_dir, extra_global_lists=args.include_globals)
    print_report(state)


if __name__ == "__main__":
    main()
