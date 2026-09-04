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

"""Cross-chapter theorem-usage acyclicity tests (theorem usage termination).

Direct tests for the three global-check functions in ``verifier.py``:
``build_theorem_citation_resolver`` (citation -> registry theorem),
``collect_theorem_usage_citations`` (one chapter's used-theorem set) and
``check_theorem_usage_termination`` (WHITE/GRAY/BLACK cycle detection over
the theorem-usage graph, one counter record per chapter-owning theorem).
These run outside ``verify_chapter``, driven by ``run_verifier``, so the
chapter-level assertion helpers do not reach them — the tests call the
functions directly and inspect the ``theorem usage termination`` counter.

See ``tests/test_harness.py`` for fixtures and runner.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_harness import (  # noqa: E402
    register, make_state_minimal, make_proof_line, run_all_tests,
)
from verifier import (  # noqa: E402
    build_theorem_citation_resolver,
    collect_theorem_usage_citations,
    check_theorem_usage_termination,
)


_COUNTER = "theorem usage termination"

# Real registry/citation pair from the live Peano artifacts (chapter 58):
# the v-form registry entry and its w-renamed rule-cell citation form.
_MIRROR_V = ("(>[N,i0,s,+,*,i1](AnchorPeano[N,i0,s,+,*,i1])"
             "(>[v1,v2]!(=[v1,v2])(>[](in3[v2,v1,i0,+])!(=[i0,v1]))))")
_MIRROR_W = ("(>[N,i0,s,+,*,i1](AnchorPeano[N,i0,s,+,*,i1])"
             "(>[w1,w2]!(=[w1,w2])(>[](in3[w2,w1,i0,+])!(=[i0,w1]))))")


def _state_with_theorems(*exprs):
    """Minimal state whose global registry holds ``exprs`` (direct type)."""
    state = make_state_minimal()
    for expr in exprs:
        state.global_theorems[expr] = {"type": "direct", "ref": "-1"}
        state.global_theorem_list.append((expr, "direct", "-1"))
    return state


def _counts(state):
    """(success, failure) of the usage-termination counter (0, 0 if absent)."""
    ctr = state.tag_counters.get(_COUNTER)
    return (ctr.success, ctr.failure) if ctr is not None else (0, 0)


# ===========================================================================
#  build_theorem_citation_resolver
# ===========================================================================

@register
def test_resolver_exact_hit_and_miss():
    """Exact registry member resolves to itself; unknown text resolves to
    None; an external theorem is deliberately NOT resolvable."""
    state = _state_with_theorems("(T1)")
    state.external_theorems.add("(EXT)")
    resolve = build_theorem_citation_resolver(state)
    assert resolve("(T1)") == "(T1)", "exact registry hit must resolve"
    assert resolve("(NOPE)") is None, "unknown citation must not resolve"
    assert resolve("(EXT)") is None, (
        "external theorems own no chapter and must not resolve")


@register
def test_resolver_w_form_citation():
    """The w/W-renamed rule-cell citation form resolves to the v-form
    registry entry (chapter-58 live pair)."""
    state = _state_with_theorems(_MIRROR_V)
    resolve = build_theorem_citation_resolver(state)
    assert resolve(_MIRROR_W) == _MIRROR_V, (
        "w-form citation must resolve to the v-form registry entry")


# ===========================================================================
#  collect_theorem_usage_citations
# ===========================================================================

@register
def test_collect_theorem_row_and_rule_cell_dedup():
    """A `theorem` row and the matching w-form `implication` rule cell
    produce ONE cited registry entry (chapter-58 shape)."""
    state = _state_with_theorems(_MIRROR_V)
    resolve = build_theorem_citation_resolver(state)
    lines = [
        make_proof_line(
            "!(=[i0,v2])", "main", "implication",
            _MIRROR_W, "main",
            "!(=[v2,v1])", "main",
        ),
        make_proof_line(_MIRROR_V, "main", "theorem"),
        make_proof_line("!(=[v2,v1])", "main", "task formulation"),
    ]
    assert collect_theorem_usage_citations(lines, resolve) == {_MIRROR_V}


@register
def test_collect_skips_local_external_and_noncitation_cells():
    """No edge from: a locally-originated rule cell equal to a registry
    entry; an `externally provided theorem` row; a non-rule premise cell
    (i > 0, non-exempt tag) textually equal to a registry entry."""
    state = _state_with_theorems("(T2)", "(T3)")
    resolve = build_theorem_citation_resolver(state)
    lines = [
        make_proof_line(
            "(A)", "main", "implication",
            "(T2)", "main",          # rule cell, but locally originated below
            "(T3)", "main",          # premise cell at i=2: not a citation slot
        ),
        make_proof_line("(T2)", "main", "task formulation"),
        make_proof_line("(EXT)", "main", "externally provided theorem"),
    ]
    assert collect_theorem_usage_citations(lines, resolve) == set()


@register
def test_collect_origin_exempt_cells_cite():
    """`or theorem` rows (origin-exempt) contribute every non-local cell
    that resolves — the synthetic-chapter citation channel."""
    state = _state_with_theorems("(T2)", "(T3)")
    resolve = build_theorem_citation_resolver(state)
    lines = [
        make_proof_line(
            "(or2[(T2),(T3)])", "main", "or theorem",
            "(T2)", "main",
            "(T3)", "main",
        ),
    ]
    assert collect_theorem_usage_citations(lines, resolve) == {"(T2)", "(T3)"}


# ===========================================================================
#  check_theorem_usage_termination
# ===========================================================================

@register
def test_usage_acyclic_chain_passes():
    """A -> B -> C, C leaf: three successes, zero failures."""
    state = make_state_minimal()
    check_theorem_usage_termination(
        ["(A)", "(B)", "(C)"],
        {"(A)": {"(B)"}, "(B)": {"(C)"}, "(C)": set()},
        state)
    assert _counts(state) == (3, 0)


@register
def test_usage_cycle_two_node():
    """A <-> B: both fail; an acyclic bystander citing into the cycle
    still passes (cycle membership, not reachability, is flagged)."""
    state = make_state_minimal()
    check_theorem_usage_termination(
        ["(A)", "(B)", "(D)"],
        {"(A)": {"(B)"}, "(B)": {"(A)"}, "(D)": {"(A)"}},
        state)
    assert _counts(state) == (1, 2)


@register
def test_usage_cycle_three_node():
    """A -> B -> C -> A: all three fail."""
    state = make_state_minimal()
    check_theorem_usage_termination(
        ["(A)", "(B)", "(C)"],
        {"(A)": {"(B)"}, "(B)": {"(C)"}, "(C)": {"(A)"}},
        state)
    assert _counts(state) == (0, 3)


@register
def test_usage_self_citation_fails():
    """A -> A: the length-1 usage cycle fails its single node."""
    state = make_state_minimal()
    check_theorem_usage_termination(
        ["(A)"], {"(A)": {"(A)"}}, state)
    assert _counts(state) == (0, 1)


@register
def test_usage_foreign_citation_is_foundation():
    """An edge to a theorem without a local chapter (cross-batch citation)
    is a foundation and cannot fail the citing node."""
    state = make_state_minimal()
    check_theorem_usage_termination(
        ["(A)"], {"(A)": {"(X-no-chapter)"}}, state)
    assert _counts(state) == (1, 0)


if __name__ == "__main__":
    sys.exit(run_all_tests())
