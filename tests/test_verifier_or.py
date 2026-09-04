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

"""Failure tests for OR-family checkers (or theorem, or disintegration, or
convergence, or branch proven, or branch assumption).

Each test injects exactly ONE subtle malformation. See
``tests/test_harness.py`` for fixtures and runner.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_harness import (  # noqa: E402
    register, make_state_with_binaries, make_proof_line,
    assert_failure, assert_pass, run_all_tests,
)
from verifier import (  # noqa: E402
    check_or_theorem, check_or_disintegration, check_or_convergence,
    check_or_branch_proven, check_or_branch_assumption,
    _negate_expr, _build_or_from_elements, _build_or_subimpls_from_elements,
)


def _state_with_or3() -> object:
    """Helper: install a 3-disjunct compiled-OR binary entry and return state.
    Used by or-disintegration / or-convergence / or-branch tests.
    """
    state = make_state_with_binaries(("Peano",))
    state.gl_binaries = dict(state.gl_binaries)
    state.gl_binaries["Peano"] = dict(state.gl_binaries["Peano"])
    state.gl_binaries["Peano"]["or3"] = {
        "category": "or",
        "signature": "(or3[u_1,u_2,u_3])",
        "arity": 3,
        "elements": ["(=[u_1,X])", "(=[u_2,X])", "(=[u_3,X])"],
    }
    state.current_gl_binary = state.gl_binaries["Peano"]
    return state


def _state_with_nested_or() -> object:
    """Install an outer OR whose first child is another compiled OR."""
    state = make_state_with_binaries(("Peano",))
    state.gl_binaries = dict(state.gl_binaries)
    state.gl_binaries["Peano"] = dict(state.gl_binaries["Peano"])
    state.gl_binaries["Peano"]["or90"] = {
        "category": "or",
        "signature": "(or90[u_1,u_2])",
        "arity": 2,
        "elements": ["(=[u_1,X])", "(=[u_2,X])"],
    }
    state.gl_binaries["Peano"]["or91"] = {
        "category": "or",
        "signature": "(or91[u_1,u_2,u_3])",
        "arity": 3,
        "elements": ["(or90[u_1,u_2])", "(=[u_3,X])"],
    }
    state.current_gl_binary = state.gl_binaries["Peano"]
    return state


# ===========================================================================
#  tag: or theorem
# ===========================================================================

@register
def test_or_theorem_namespace_not_main():
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(or2[a,b,c,d])",
        "main_boundary_orint_X_((=[a,b]))",
        "or theorem",
        "(some_existence)", "main",
    )
    assert_failure(check_or_theorem, line, [line], state)


@register
def test_or_theorem_rest_empty():
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line("(or2[a,b,c,d])", "main", "or theorem")
    assert_failure(check_or_theorem, line, [line], state)


@register
def test_or_theorem_rest_length_one():
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(or2[a,b,c,d])", "main", "or theorem",
        "(some_existence)",
    )
    assert_failure(check_or_theorem, line, [line], state)


@register
def test_or_theorem_descendant_namespace():
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(or2[a,b,c,d])",
        "main_boundary_recursion_((=[v1,i0]))",
        "or theorem",
        "(some_existence)", "main",
    )
    assert_failure(check_or_theorem, line, [line], state)


@register
def test_or_theorem_empty_namespace():
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(or2[a,b,c,d])", "", "or theorem",
        "(some_existence)", "main",
    )
    assert_failure(check_or_theorem, line, [line], state)


@register
def test_or_theorem_random_namespace():
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(or2[a,b,c,d])", "subsidiary", "or theorem",
        "(some_existence)", "main",
    )
    assert_failure(check_or_theorem, line, [line], state)


# ===========================================================================
#  tag: or disintegration
# ===========================================================================

@register
def test_or_disintegration_rest_empty():
    state = _state_with_or3()
    line = make_proof_line("(=[a,X])", "main_boundary_ordis_(or3[a,b,c])_((=[a,X]))",
                           "or disintegration")
    assert_failure(check_or_disintegration, line, [line], state)


@register
def test_or_disintegration_rest_length_one():
    state = _state_with_or3()
    line = make_proof_line(
        "(=[a,X])", "main_boundary_ordis_(or3[a,b,c])_((=[a,X]))",
        "or disintegration",
        "(or3[a,b,c])",
    )
    assert_failure(check_or_disintegration, line, [line], state)


@register
def test_or_disintegration_rest_length_three():
    state = _state_with_or3()
    line = make_proof_line(
        "(=[a,X])", "main_boundary_ordis_(or3[a,b,c])_((=[a,X]))",
        "or disintegration",
        "(or3[a,b,c])", "main",
        "extra",
    )
    assert_failure(check_or_disintegration, line, [line], state)


@register
def test_or_disintegration_or_expr_not_compiled():
    state = _state_with_or3()
    line = make_proof_line(
        "(=[a,X])", "main_boundary_ordis_NOT_OR_((=[a,X]))",
        "or disintegration",
        "(notor[a,b,c])", "main",   # not (or<N>[...])
    )
    assert_failure(check_or_disintegration, line, [line], state)


@register
def test_or_disintegration_or_unknown_in_binary():
    """(or99[…]) parses as compiled OR shape but no binary defines or99."""
    state = _state_with_or3()
    line = make_proof_line(
        "(=[a,X])", "main_boundary_ordis_(or99[a,b])_((=[a,X]))",
        "or disintegration",
        "(or99[a,b])", "main",
    )
    assert_failure(check_or_disintegration, line, [line], state)


@register
def test_or_disintegration_or_arity_mismatch():
    """or3 expects arity 3; supplied OR has arity 2."""
    state = _state_with_or3()
    line = make_proof_line(
        "(=[a,X])", "main_boundary_ordis_(or3[a,b])_((=[a,X]))",
        "or disintegration",
        "(or3[a,b])", "main",   # arity 2, but or3.arity=3
    )
    assert_failure(check_or_disintegration, line, [line], state)


@register
def test_or_disintegration_asserted_not_a_disjunct():
    """asserted disjunct doesn't match any of or3's substituted disjuncts."""
    state = _state_with_or3()
    line = make_proof_line(
        "(=[unrelated,Y])",
        "main_boundary_ordis_(or3[a,b,c])_((=[unrelated,Y]))",
        "or disintegration",
        "(or3[a,b,c])", "main",
    )
    # or3's disjuncts are (=[a,X]), (=[b,X]), (=[c,X]); asserted is none.
    # But we need an OR origin row to pass step 5 first; missing it triggers
    # earlier check anyway. Add the origin row to isolate the asserted-mismatch.
    origin = make_proof_line("(or3[a,b,c])", "main", "task formulation")
    assert_failure(check_or_disintegration, line, [line, origin], state)


@register
def test_or_disintegration_branch_ns_format_wrong():
    """branch_ns missing the required _boundary_ordis_ structure."""
    state = _state_with_or3()
    line = make_proof_line(
        "(=[a,X])",
        "main_boundary_orint_(or3[a,b,c])_((=[a,X]))",   # _orint_ not _ordis_
        "or disintegration",
        "(or3[a,b,c])", "main",
    )
    origin = make_proof_line("(or3[a,b,c])", "main", "task formulation")
    assert_failure(check_or_disintegration, line, [line, origin], state)


@register
def test_or_disintegration_branch_ns_wrong_disjunct_payload():
    """branch_ns names a DIFFERENT disjunct than line.expression."""
    state = _state_with_or3()
    line = make_proof_line(
        "(=[a,X])",
        "main_boundary_ordis_(or3[a,b,c])_((=[b,X]))",   # mismatched disjunct
        "or disintegration",
        "(or3[a,b,c])", "main",
    )
    origin = make_proof_line("(or3[a,b,c])", "main", "task formulation")
    assert_failure(check_or_disintegration, line, [line, origin], state)


@register
def test_or_disintegration_no_or_origin_in_chapter():
    """No chapter row at parent scope derives the OR; case-split has nothing
    to consume."""
    state = _state_with_or3()
    line = make_proof_line(
        "(=[a,X])",
        "main_boundary_ordis_(or3[a,b,c])_((=[a,X]))",
        "or disintegration",
        "(or3[a,b,c])", "main",
    )
    # Chapter contains only the disintegration row itself.
    assert_failure(check_or_disintegration, line, [line], state)


@register
def test_nested_or_disintegration_rejects_intermediate_or_branch():
    """The old two-hashburst intermediate ``or90`` branch is not a leaf."""
    state = _state_with_nested_or()
    line = make_proof_line(
        "(or90[a,b])",
        "main_boundary_ordis_(or91[a,b,c])_((or90[a,b]))",
        "or disintegration",
        "(or91[a,b,c])", "main",
    )
    origin = make_proof_line("(or91[a,b,c])", "main", "task formulation")
    assert_failure(check_or_disintegration, line, [line, origin], state)


# ===========================================================================
#  tag: or convergence
# ===========================================================================

@register
def test_or_convergence_rest_empty():
    state = _state_with_or3()
    line = make_proof_line("(conclusion)", "main", "or convergence")
    assert_failure(check_or_convergence, line, [line], state)


@register
def test_or_convergence_rest_too_short():
    """rest length 4 < 6."""
    state = _state_with_or3()
    line = make_proof_line(
        "(conclusion)", "main", "or convergence",
        "(or3[a,b,c])", "main",
    )
    assert_failure(check_or_convergence, line, [line], state)


@register
def test_or_convergence_rest_odd_length():
    """rest length 7 -> reject (must be even)."""
    state = _state_with_or3()
    line = make_proof_line(
        "(conclusion)", "main", "or convergence",
        "(or3[a,b,c])", "main",
        "(conclusion)", "main_boundary_ordis_(or3[a,b,c])_((=[a,X]))",
        "(conclusion)",
    )
    assert_failure(check_or_convergence, line, [line], state)


@register
def test_or_convergence_parent_ns_mismatch():
    """line.namespace doesn't equal rest[1]."""
    state = _state_with_or3()
    line = make_proof_line(
        "(conclusion)",
        "main",   # but rest[1] below says "other_parent"
        "or convergence",
        "(or3[a,b,c])", "other_parent",
        "(conclusion)", "other_parent_boundary_ordis_(or3[a,b,c])_((=[a,X]))",
        "(conclusion)", "other_parent_boundary_ordis_(or3[a,b,c])_((=[b,X]))",
        "(conclusion)", "other_parent_boundary_ordis_(or3[a,b,c])_((=[c,X]))",
    )
    assert_failure(check_or_convergence, line, [line], state)


@register
def test_or_convergence_or_unknown_in_binary():
    state = _state_with_or3()
    line = make_proof_line(
        "(conclusion)", "main", "or convergence",
        "(or99[a,b,c])", "main",   # not in binary
        "(conclusion)", "main_boundary_ordis_x",
        "(conclusion)", "main_boundary_ordis_y",
    )
    assert_failure(check_or_convergence, line, [line], state)


@register
def test_or_convergence_disjuncts_count_under_two():
    """Install or_single with only 1 disjunct -> rejected by `len(disjuncts) < 2`."""
    state = make_state_with_binaries(("Peano",))
    state.gl_binaries = dict(state.gl_binaries)
    state.gl_binaries["Peano"] = dict(state.gl_binaries["Peano"])
    state.gl_binaries["Peano"]["or1"] = {
        "category": "or",
        "signature": "(or1[u_1])",
        "arity": 1,
        "elements": ["(=[u_1,X])"],   # only 1 element
    }
    state.current_gl_binary = state.gl_binaries["Peano"]
    line = make_proof_line(
        "(conclusion)", "main", "or convergence",
        "(or1[a])", "main",
        "(conclusion)", "main_boundary_ordis_(or1[a])_((=[a,X]))",
    )
    assert_failure(check_or_convergence, line, [line], state)


@register
def test_or_convergence_pair_count_doesnt_match_k():
    """k=3 disjuncts; supplied 2 (C, branch) pairs -> len(rest) != 2+2k."""
    state = _state_with_or3()
    line = make_proof_line(
        "(conclusion)", "main", "or convergence",
        "(or3[a,b,c])", "main",
        "(conclusion)", "main_boundary_ordis_(or3[a,b,c])_((=[a,X]))",
        "(conclusion)", "main_boundary_ordis_(or3[a,b,c])_((=[b,X]))",
        # missing the third branch
    )
    assert_failure(check_or_convergence, line, [line], state)


@register
def test_or_convergence_c_field_repetition_breaks():
    """One c_field doesn't equal line.expression."""
    state = _state_with_or3()
    line = make_proof_line(
        "(conclusion)", "main", "or convergence",
        "(or3[a,b,c])", "main",
        "(conclusion)", "main_boundary_ordis_(or3[a,b,c])_((=[a,X]))",
        "(WRONG_C)", "main_boundary_ordis_(or3[a,b,c])_((=[b,X]))",
        "(conclusion)", "main_boundary_ordis_(or3[a,b,c])_((=[c,X]))",
    )
    assert_failure(check_or_convergence, line, [line], state)


@register
def test_or_convergence_branch_ns_not_descendant():
    """A branch_ns doesn't start with parent + '_boundary_'."""
    state = _state_with_or3()
    line = make_proof_line(
        "(conclusion)", "main", "or convergence",
        "(or3[a,b,c])", "main",
        "(conclusion)", "main_boundary_ordis_(or3[a,b,c])_((=[a,X]))",
        "(conclusion)", "main_boundary_ordis_(or3[a,b,c])_((=[b,X]))",
        "(conclusion)", "completely_unrelated_namespace",
    )
    assert_failure(check_or_convergence, line, [line], state)


@register
def test_or_convergence_branch_ns_duplicate():
    """Two of the three branch_ns values are identical -> distinctness fails."""
    state = _state_with_or3()
    line = make_proof_line(
        "(conclusion)", "main", "or convergence",
        "(or3[a,b,c])", "main",
        "(conclusion)", "main_boundary_ordis_(or3[a,b,c])_((=[a,X]))",
        "(conclusion)", "main_boundary_ordis_(or3[a,b,c])_((=[a,X]))",
        "(conclusion)", "main_boundary_ordis_(or3[a,b,c])_((=[c,X]))",
    )
    assert_failure(check_or_convergence, line, [line], state)


@register
def test_or_convergence_branch_derivation_missing():
    """Layout OK, branches distinct, but no chapter row derives the
    conclusion at one of the branch namespaces."""
    state = _state_with_or3()
    convergence = make_proof_line(
        "(conclusion)", "main", "or convergence",
        "(or3[a,b,c])", "main",
        "(conclusion)", "main_boundary_ordis_(or3[a,b,c])_((=[a,X]))",
        "(conclusion)", "main_boundary_ordis_(or3[a,b,c])_((=[b,X]))",
        "(conclusion)", "main_boundary_ordis_(or3[a,b,c])_((=[c,X]))",
    )
    # Provide derivations for only 2 of the 3 branches.
    deriv_a = make_proof_line(
        "(conclusion)",
        "main_boundary_ordis_(or3[a,b,c])_((=[a,X]))",
        "task formulation",
    )
    deriv_b = make_proof_line(
        "(conclusion)",
        "main_boundary_ordis_(or3[a,b,c])_((=[b,X]))",
        "task formulation",
    )
    chapter = [convergence, deriv_a, deriv_b]
    assert_failure(check_or_convergence, convergence, chapter, state)


@register
def test_or_convergence_or_arity_mismatch():
    """OR has arity 3; chapter row's compiled OR has arity 2 -> binary lookup
    finds no entry."""
    state = _state_with_or3()
    line = make_proof_line(
        "(conclusion)", "main", "or convergence",
        "(or3[a,b])", "main",   # arity 2, mismatched with or3.arity=3
        "(conclusion)", "main_boundary_ordis_(or3[a,b])_((=[a,X]))",
        "(conclusion)", "main_boundary_ordis_(or3[a,b])_((=[b,X]))",
    )
    assert_failure(check_or_convergence, line, [line], state)


@register
def test_or_convergence_reduced_cohort_mixed_row_passes():
    """Dead-branch retirement: one survivor entry plus two retired entries
    (the dead branches' asserted disjuncts refuted at main) cover all three
    disjuncts, each with its own chapter row -> PASS."""
    state = _state_with_or3()
    convergence = make_proof_line(
        "(conclusion)", "main", "or convergence",
        "(or3[a,b,c])", "main",
        "(conclusion)", "main_boundary_ordis_(or3[a,b,c])_((=[c,X]))",
        "!(=[a,X])", "main",
        "!(=[b,X])", "main",
    )
    deriv_c = make_proof_line(
        "(conclusion)", "main_boundary_ordis_(or3[a,b,c])_((=[c,X]))",
        "task formulation")
    ref_a = make_proof_line("!(=[a,X])", "main", "task formulation")
    ref_b = make_proof_line("!(=[b,X])", "main", "task formulation")
    chapter = [convergence, deriv_c, ref_a, ref_b]
    assert_pass(check_or_convergence, convergence, chapter, state)


@register
def test_or_convergence_retired_self_refutation_in_branch_passes():
    """A retired entry may cite the refutation inside the retired branch
    itself (ex-falso self-refutation: the branch assuming D derived !D)."""
    state = _state_with_or3()
    branch_a = "main_boundary_ordis_(or3[a,b,c])_((=[a,X]))"
    convergence = make_proof_line(
        "(conclusion)", "main", "or convergence",
        "(or3[a,b,c])", "main",
        "!(=[a,X])", branch_a,
        "(conclusion)", "main_boundary_ordis_(or3[a,b,c])_((=[b,X]))",
        "(conclusion)", "main_boundary_ordis_(or3[a,b,c])_((=[c,X]))",
    )
    ref_a = make_proof_line("!(=[a,X])", branch_a, "task formulation")
    deriv_b = make_proof_line(
        "(conclusion)", "main_boundary_ordis_(or3[a,b,c])_((=[b,X]))",
        "task formulation")
    deriv_c = make_proof_line(
        "(conclusion)", "main_boundary_ordis_(or3[a,b,c])_((=[c,X]))",
        "task formulation")
    chapter = [convergence, ref_a, deriv_b, deriv_c]
    assert_pass(check_or_convergence, convergence, chapter, state)


@register
def test_or_convergence_retired_negation_not_a_disjunct():
    """A retired entry whose expression negates nothing in the OR -> reject."""
    state = _state_with_or3()
    convergence = make_proof_line(
        "(conclusion)", "main", "or convergence",
        "(or3[a,b,c])", "main",
        "(conclusion)", "main_boundary_ordis_(or3[a,b,c])_((=[c,X]))",
        "!(=[z,X])", "main",
        "!(=[b,X])", "main",
    )
    assert_failure(check_or_convergence, convergence, [convergence], state)


@register
def test_or_convergence_retired_scope_not_parent_visible():
    """A retired entry at a scope the parent does not inherit from -> reject."""
    state = _state_with_or3()
    convergence = make_proof_line(
        "(conclusion)", "main", "or convergence",
        "(or3[a,b,c])", "main",
        "(conclusion)", "main_boundary_ordis_(or3[a,b,c])_((=[c,X]))",
        "!(=[a,X])", "main_boundary_(unrelated)",
        "!(=[b,X])", "main",
    )
    assert_failure(check_or_convergence, convergence, [convergence], state)


@register
def test_or_convergence_double_cover_same_disjunct():
    """Two entries covering the same disjunct leave another uncovered ->
    reject (exactly-once coverage)."""
    state = _state_with_or3()
    convergence = make_proof_line(
        "(conclusion)", "main", "or convergence",
        "(or3[a,b,c])", "main",
        "(conclusion)", "main_boundary_ordis_(or3[a,b,c])_((=[c,X]))",
        "!(=[c,X])", "main",
        "!(=[b,X])", "main",
    )
    assert_failure(check_or_convergence, convergence, [convergence], state)


@register
def test_or_convergence_retired_evidence_missing():
    """Mixed row structurally OK, but one retired refutation has no chapter
    row of its own -> reject."""
    state = _state_with_or3()
    convergence = make_proof_line(
        "(conclusion)", "main", "or convergence",
        "(or3[a,b,c])", "main",
        "(conclusion)", "main_boundary_ordis_(or3[a,b,c])_((=[c,X]))",
        "!(=[a,X])", "main",
        "!(=[b,X])", "main",
    )
    deriv_c = make_proof_line(
        "(conclusion)", "main_boundary_ordis_(or3[a,b,c])_((=[c,X]))",
        "task formulation")
    ref_a = make_proof_line("!(=[a,X])", "main", "task formulation")
    chapter = [convergence, deriv_c, ref_a]   # ref_b missing
    assert_failure(check_or_convergence, convergence, chapter, state)


# ===========================================================================
#  tag: or branch proven
# ===========================================================================

@register
def test_nested_or_branch_proven_accepts_atomic_leaf():
    state = _state_with_nested_or()
    line = make_proof_line(
        "(or91[a,b,c])", "main", "or branch proven",
        "(=[a,X])",
        "main_boundary_orint_(or91[a,b,c])_((=[a,X]))",
    )
    assert_pass(check_or_branch_proven, line, [line], state)


@register
def test_nested_or_branch_proven_rejects_intermediate_or():
    state = _state_with_nested_or()
    line = make_proof_line(
        "(or91[a,b,c])", "main", "or branch proven",
        "(or90[a,b])",
        "main_boundary_orint_(or91[a,b,c])_((or90[a,b]))",
    )
    assert_failure(check_or_branch_proven, line, [line], state)


@register
def test_or_branch_proven_rest_empty():
    state = _state_with_or3()
    line = make_proof_line(
        "(or3[a,b,c])", "main", "or branch proven")
    assert_failure(check_or_branch_proven, line, [line], state)


@register
def test_or_branch_proven_rest_length_one():
    state = _state_with_or3()
    line = make_proof_line(
        "(or3[a,b,c])", "main", "or branch proven",
        "(=[a,X])",
    )
    assert_failure(check_or_branch_proven, line, [line], state)


@register
def test_or_branch_proven_rest_length_three():
    state = _state_with_or3()
    line = make_proof_line(
        "(or3[a,b,c])", "main", "or branch proven",
        "(=[a,X])", "main_boundary_orint_(or3[a,b,c])_((=[a,X]))",
        "extra",
    )
    assert_failure(check_or_branch_proven, line, [line], state)


@register
def test_or_branch_proven_or_expr_not_compiled():
    state = _state_with_or3()
    line = make_proof_line(
        "(notor[a,b,c])", "main", "or branch proven",
        "(=[a,X])", "main_boundary_orint_(notor[a,b,c])_((=[a,X]))",
    )
    assert_failure(check_or_branch_proven, line, [line], state)


@register
def test_or_branch_proven_or_unknown():
    state = _state_with_or3()
    line = make_proof_line(
        "(or99[a,b,c])", "main", "or branch proven",
        "(=[a,X])", "main_boundary_orint_(or99[a,b,c])_((=[a,X]))",
    )
    assert_failure(check_or_branch_proven, line, [line], state)


@register
def test_or_branch_proven_asserted_not_a_disjunct():
    state = _state_with_or3()
    line = make_proof_line(
        "(or3[a,b,c])", "main", "or branch proven",
        "(=[unrelated,Y])",
        "main_boundary_orint_(or3[a,b,c])_((=[unrelated,Y]))",
    )
    assert_failure(check_or_branch_proven, line, [line], state)


@register
def test_or_branch_proven_branch_ns_wrong_separator():
    state = _state_with_or3()
    line = make_proof_line(
        "(or3[a,b,c])", "main", "or branch proven",
        "(=[a,X])",
        "main_boundary_ordis_(or3[a,b,c])_((=[a,X]))",   # _ordis_ not _orint_
    )
    assert_failure(check_or_branch_proven, line, [line], state)


@register
def test_or_branch_proven_branch_ns_trailing_junk():
    state = _state_with_or3()
    line = make_proof_line(
        "(or3[a,b,c])", "main", "or branch proven",
        "(=[a,X])",
        "main_boundary_orint_(or3[a,b,c])_((=[a,X])) trailing",
    )
    assert_failure(check_or_branch_proven, line, [line], state)


@register
def test_or_branch_proven_branch_ns_missing_close():
    state = _state_with_or3()
    line = make_proof_line(
        "(or3[a,b,c])", "main", "or branch proven",
        "(=[a,X])",
        "main_boundary_orint_(or3[a,b,c])_((=[a,X])",   # missing final ')'
    )
    assert_failure(check_or_branch_proven, line, [line], state)


@register
def test_or_branch_proven_branch_ns_wrong_disjunct():
    state = _state_with_or3()
    line = make_proof_line(
        "(or3[a,b,c])", "main", "or branch proven",
        "(=[a,X])",
        "main_boundary_orint_(or3[a,b,c])_((=[b,X]))",   # ns names different disjunct
    )
    assert_failure(check_or_branch_proven, line, [line], state)


# ===========================================================================
#  tag: or branch assumption
# ===========================================================================

@register
def test_nested_or_branch_assumption_accepts_other_atomic_leaf():
    state = _state_with_nested_or()
    branch = "main_boundary_orint_(or91[a,b,c])_((=[a,X]))"
    proven = make_proof_line(
        "(or91[a,b,c])", "main", "or branch proven",
        "(=[a,X])", branch,
    )
    assumption = make_proof_line(
        "!(=[c,X])", branch, "or branch assumption",
        "(or91[a,b,c])_integration_goal", "main",
    )
    assert_pass(
        check_or_branch_assumption, assumption, [proven, assumption], state)


@register
def test_nested_or_branch_assumption_rejects_intermediate_or():
    state = _state_with_nested_or()
    branch = "main_boundary_orint_(or91[a,b,c])_((=[c,X]))"
    proven = make_proof_line(
        "(or91[a,b,c])", "main", "or branch proven",
        "(=[c,X])", branch,
    )
    assumption = make_proof_line(
        "!(or90[a,b])", branch, "or branch assumption",
        "(or91[a,b,c])_integration_goal", "main",
    )
    assert_failure(
        check_or_branch_assumption, assumption, [proven, assumption], state)


@register
def test_or_branch_assumption_rest_empty():
    state = _state_with_or3()
    line = make_proof_line(
        "!(=[b,X])",
        "main_boundary_orint_(or3[a,b,c])_((=[a,X]))",
        "or branch assumption",
    )
    assert_failure(check_or_branch_assumption, line, [line], state)


@register
def test_or_branch_assumption_rest_length_one():
    state = _state_with_or3()
    line = make_proof_line(
        "!(=[b,X])",
        "main_boundary_orint_(or3[a,b,c])_((=[a,X]))",
        "or branch assumption",
        "(or3[a,b,c])_integration_goal",
    )
    assert_failure(check_or_branch_assumption, line, [line], state)


@register
def test_or_branch_assumption_rest_length_three():
    state = _state_with_or3()
    line = make_proof_line(
        "!(=[b,X])",
        "main_boundary_orint_(or3[a,b,c])_((=[a,X]))",
        "or branch assumption",
        "(or3[a,b,c])_integration_goal", "main", "extra",
    )
    assert_failure(check_or_branch_assumption, line, [line], state)


@register
def test_or_branch_assumption_or_field_no_suffix():
    state = _state_with_or3()
    line = make_proof_line(
        "!(=[b,X])",
        "main_boundary_orint_(or3[a,b,c])_((=[a,X]))",
        "or branch assumption",
        "(or3[a,b,c])",   # missing _integration_goal suffix
        "main",
    )
    assert_failure(check_or_branch_assumption, line, [line], state)


@register
def test_or_branch_assumption_or_stripped_unknown():
    state = _state_with_or3()
    line = make_proof_line(
        "!(=[b,X])",
        "main_boundary_orint_(or99[a,b,c])_((=[a,X]))",
        "or branch assumption",
        "(or99[a,b,c])_integration_goal",
        "main",
    )
    assert_failure(check_or_branch_assumption, line, [line], state)


@register
def test_or_branch_assumption_expression_not_negated():
    state = _state_with_or3()
    line = make_proof_line(
        "(=[b,X])",   # NOT negated
        "main_boundary_orint_(or3[a,b,c])_((=[a,X]))",
        "or branch assumption",
        "(or3[a,b,c])_integration_goal", "main",
    )
    assert_failure(check_or_branch_assumption, line, [line], state)


@register
def test_or_branch_assumption_negated_not_disjunct():
    state = _state_with_or3()
    line = make_proof_line(
        "!(=[unrelated,Y])",
        "main_boundary_orint_(or3[a,b,c])_((=[a,X]))",
        "or branch assumption",
        "(or3[a,b,c])_integration_goal", "main",
    )
    assert_failure(check_or_branch_assumption, line, [line], state)


@register
def test_or_branch_assumption_branch_ns_wrong_prefix():
    state = _state_with_or3()
    line = make_proof_line(
        "!(=[b,X])",
        "main_boundary_ordis_(or3[a,b,c])_((=[a,X]))",   # _ordis_ not _orint_
        "or branch assumption",
        "(or3[a,b,c])_integration_goal", "main",
    )
    assert_failure(check_or_branch_assumption, line, [line], state)


@register
def test_or_branch_assumption_branch_ns_trailing_junk():
    state = _state_with_or3()
    line = make_proof_line(
        "!(=[b,X])",
        "main_boundary_orint_(or3[a,b,c])_((=[a,X])) trailing",
        "or branch assumption",
        "(or3[a,b,c])_integration_goal", "main",
    )
    assert_failure(check_or_branch_assumption, line, [line], state)


@register
def test_or_branch_assumption_asserted_equals_negated():
    """branch_ns says asserted = (=[a,X]) but negated = (=[a,X]) (same)."""
    state = _state_with_or3()
    line = make_proof_line(
        "!(=[a,X])",
        "main_boundary_orint_(or3[a,b,c])_((=[a,X]))",
        "or branch assumption",
        "(or3[a,b,c])_integration_goal", "main",
    )
    assert_failure(check_or_branch_assumption, line, [line], state)


@register
def test_or_branch_assumption_no_matching_proven_row():
    """Structurally valid but no matching `or branch proven` row in chapter."""
    state = _state_with_or3()
    assumption = make_proof_line(
        "!(=[b,X])",
        "main_boundary_orint_(or3[a,b,c])_((=[a,X]))",
        "or branch assumption",
        "(or3[a,b,c])_integration_goal", "main",
    )
    assert_failure(check_or_branch_assumption, assumption, [assumption], state)


@register
def test_or_branch_assumption_proven_row_wrong_disjunct():
    """An `or branch proven` row exists at the right parent_ns but for a
    different disjunct than the branch_ns asserts."""
    state = _state_with_or3()
    assumption = make_proof_line(
        "!(=[b,X])",
        "main_boundary_orint_(or3[a,b,c])_((=[a,X]))",
        "or branch assumption",
        "(or3[a,b,c])_integration_goal", "main",
    )
    # `or branch proven` is for disjunct (=[c,X]), not (=[a,X])
    proven = make_proof_line(
        "(or3[a,b,c])", "main", "or branch proven",
        "(=[c,X])", "main_boundary_orint_(or3[a,b,c])_((=[c,X]))",
    )
    assert_failure(check_or_branch_assumption, assumption, [assumption, proven], state)


@register
def test_negate_expr_cancels_double_negation():
    """_negate_expr adds one '!' to a positive expression and strips the
    leading '!' from a negated one (Python mirror of the C++
    negateScratch / expandSignature OR-case negation)."""
    assert _negate_expr("(=[a,b])") == "!(=[a,b])"
    assert _negate_expr("!(=[a,b])") == "(=[a,b])"


@register
def test_build_or_from_elements_negated_disjunct_cancels():
    """A negated disjunct's De Morgan conjunct is its bare positive core.
    A blind '!' prefix would flip the OR's meaning — (a=b) OR (c=d)
    instead of (a=b) OR !(c=d) — the exact defect that minted a false
    or theorem and contradicted the incubator anchor."""
    assert (_build_or_from_elements(["(=[a,b])", "!(=[c,d])"])
            == "!(&!(=[a,b])(=[c,d]))")
    # All-positive elements keep the historical bytes.
    assert (_build_or_from_elements(["(P[a])", "(Q[b])"])
            == "!(&!(P[a])!(Q[b]))")


@register
def test_build_or_subimpls_negated_disjunct_cancels():
    """The K-sub-implication premises negate co-disjuncts with
    cancellation: the negated disjunct !(=[c,d]) contributes the premise
    (=[c,d]); each branch head stays the disjunct verbatim."""
    subs = _build_or_subimpls_from_elements(["(=[a,b])", "!(=[c,d])"])
    assert subs == ["(>[](=[c,d])(=[a,b]))", "(>[]!(=[a,b])!(=[c,d]))"]


@register
def test_build_or_from_elements_three_disjuncts_nests_negated():
    """k>=3: the or-so-far is a DISJUNCT of the next level and enters the
    AND negated — its !(&...) form contributes the bare positive (&...)
    by double-negation cancellation, so the nest reads (D1 v D2) v D3.
    The former builder inserted the or-so-far un-negated, which read
    NOT(D1 v D2) v D3 — the D-260 mirrored polarity defect, live once
    3-element ors are consumed in-run."""
    built = _build_or_from_elements(["(P[a])", "(Q[b])", "(R[c])"])
    assert built == "!(&(&!(P[a])!(Q[b]))!(R[c]))"
    # Four disjuncts nest the same way one level deeper.
    built4 = _build_or_from_elements(["(P[a])", "(Q[b])", "(R[c])", "(S[d])"])
    assert built4 == "!(&(&(&!(P[a])!(Q[b]))!(R[c]))!(S[d]))"
    # A negated THIRD disjunct still cancels to its bare positive core.
    built_neg = _build_or_from_elements(["(P[a])", "(Q[b])", "!(R[c])"])
    assert built_neg == "!(&(&!(P[a])!(Q[b]))(R[c]))"


@register
def test_parse_or_disjuncts_roundtrips_builder():
    """_parse_or_disjuncts is the exact inverse of the fixed builder for
    2, 3, and 4 disjuncts, including a negated disjunct's polarity."""
    from verifier import _parse_or_disjuncts
    for elements in (
        ["(P[a])", "(Q[b])"],
        ["(P[a])", "!(Q[b])"],
        ["(P[a])", "(Q[b])", "(R[c])"],
        ["(P[a])", "(Q[b])", "!(R[c])"],
        ["(P[a])", "(Q[b])", "(R[c])", "(S[d])"],
    ):
        assert _parse_or_disjuncts(_build_or_from_elements(elements)) == elements


if __name__ == "__main__":
    sys.exit(run_all_tests())
