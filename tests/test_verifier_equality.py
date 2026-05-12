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

"""Failure tests for equality1 / equality2 / symmetry-of-(in)equality checkers.

Each test injects exactly ONE subtle malformation and asserts the verifier
rejects the row. See ``tests/test_harness.py`` for the registry, fixtures,
and runner.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_harness import (  # noqa: E402
    register, make_state_minimal, make_proof_line, assert_failure,
    run_all_tests,
)
from verifier import (  # noqa: E402
    check_equality1, check_equality2,
    check_symmetry_of_equality, check_symmetry_of_inequality,
)


# ===========================================================================
#  tag: equality1
# ===========================================================================

@register
def test_equality1_rest_empty():
    """rest empty -> ``len(rest) < 4`` short-circuits."""
    state = make_state_minimal()
    line = make_proof_line("(in[a,N])", "main", "equality1")
    assert_failure(check_equality1, line, [line], state)


@register
def test_equality1_rest_length_two():
    """Only source + source_ns supplied; missing equality justification."""
    state = make_state_minimal()
    line = make_proof_line(
        "(in[a,N])", "main", "equality1",
        "(in[b,N])", "main",
    )
    assert_failure(check_equality1, line, [line], state)


@register
def test_equality1_rest_length_three():
    """Odd length (3): source + source_ns + dangling eq with no eq_ns."""
    state = make_state_minimal()
    line = make_proof_line(
        "(in[a,N])", "main", "equality1",
        "(in[b,N])", "main",
        "(=[b,a])",
    )
    assert_failure(check_equality1, line, [line], state)


@register
def test_equality1_source_ns_sibling_scope():
    """source_ns is a sibling scope, not a strict prefix of result_ns."""
    state = make_state_minimal()
    line = make_proof_line(
        "(in[a,N])", "main_boundary_orint_X_((=[a,b]))",
        "equality1",
        "(in[b,N])", "main_boundary_orint_Y_((=[c,d]))",
        "(=[b,a])", "main",
    )
    assert_failure(check_equality1, line, [line], state)


@register
def test_equality1_eq_ns_sibling_scope():
    """eq_ns is a sibling scope -> ``_ns_matches_or_strict_prefix`` rejects."""
    state = make_state_minimal()
    line = make_proof_line(
        "(in[a,N])", "main_boundary_orint_X_((=[a,b]))",
        "equality1",
        "(in[b,N])", "main_boundary_orint_X_((=[a,b]))",
        "(=[b,a])", "main_boundary_orint_Y_((=[c,d]))",
    )
    assert_failure(check_equality1, line, [line], state)


@register
def test_equality1_core_mismatch():
    """Source core 'in' vs result core '=' -> reject."""
    state = make_state_minimal()
    line = make_proof_line(
        "(=[a,N])", "main", "equality1",
        "(in[b,N])", "main",
        "(=[b,a])", "main",
    )
    assert_failure(check_equality1, line, [line], state)


@register
def test_equality1_arity_mismatch():
    """Source arity 3, result arity 2 -> reject."""
    state = make_state_minimal()
    line = make_proof_line(
        "(in[a,N])", "main", "equality1",
        "(in[b,N,extra])", "main",
        "(=[b,a])", "main",
    )
    assert_failure(check_equality1, line, [line], state)


@register
def test_equality1_differing_arg_uncovered():
    """First arg differs (b -> a) but supplied equality covers (c, a),
    not (b, a) -> uncovered position."""
    state = make_state_minimal()
    line = make_proof_line(
        "(in[a,N])", "main", "equality1",
        "(in[b,N])", "main",
        "(=[c,a])", "main",   # wrong source pair
    )
    assert_failure(check_equality1, line, [line], state)


@register
def test_equality1_equality_direction_reversed():
    """Need (source=b, result=a) -> eq (b, a); supplied (a, b) backwards."""
    state = make_state_minimal()
    line = make_proof_line(
        "(in[a,N])", "main", "equality1",
        "(in[b,N])", "main",
        "(=[a,b])", "main",   # reversed; eq_set has (a, b), need (b, a)
    )
    assert_failure(check_equality1, line, [line], state)


@register
def test_equality1_equality_arity_one():
    """Equality has arity 1 -> ``len(ea) == 2`` guard drops it from eq_set."""
    state = make_state_minimal()
    line = make_proof_line(
        "(in[a,N])", "main", "equality1",
        "(in[b,N])", "main",
        "(=[b])", "main",   # arity 1 — eq_set won't include any pair
    )
    assert_failure(check_equality1, line, [line], state)


@register
def test_equality1_two_differing_args_only_one_covered():
    """Two positions differ; only one equality supplied -> second uncovered."""
    state = make_state_minimal()
    line = make_proof_line(
        "(in[a,N])", "main", "equality1",
        "(in[b,M])", "main",
        "(=[b,a])", "main",   # covers position 1, leaves position 2 (M vs N) uncovered
    )
    assert_failure(check_equality1, line, [line], state)


@register
def test_equality1_partial_equality_set_after_swap():
    """3 differing positions, only 2 equalities supplied -> third uncovered."""
    state = make_state_minimal()
    line = make_proof_line(
        "(in[a,b,c])", "main", "equality1",
        "(in[x,y,z])", "main",
        "(=[x,a])", "main",
        "(=[y,b])", "main",
        # position 3 (z -> c) uncovered
    )
    assert_failure(check_equality1, line, [line], state)


# ===========================================================================
#  tag: equality2
# ===========================================================================

@register
def test_equality2_rest_empty():
    """rest empty -> ``len(rest) < 4`` short-circuits."""
    state = make_state_minimal()
    line = make_proof_line("(=[a,c])", "main", "equality2")
    assert_failure(check_equality2, line, [line], state)


@register
def test_equality2_rest_length_two():
    """Only one of the two source equalities supplied."""
    state = make_state_minimal()
    line = make_proof_line(
        "(=[a,c])", "main", "equality2",
        "(=[a,b])", "main",
    )
    assert_failure(check_equality2, line, [line], state)


@register
def test_equality2_eq1_ns_sibling():
    """eq1_ns is sibling scope of result_ns -> reject."""
    state = make_state_minimal()
    line = make_proof_line(
        "(=[a,c])", "main_boundary_orint_X_((=[i,j]))",
        "equality2",
        "(=[a,b])", "main_boundary_orint_Y_((=[k,l]))",
        "(=[b,c])", "main",
    )
    assert_failure(check_equality2, line, [line], state)


@register
def test_equality2_eq2_ns_sibling():
    """eq2_ns is sibling scope -> reject."""
    state = make_state_minimal()
    line = make_proof_line(
        "(=[a,c])", "main_boundary_orint_X_((=[i,j]))",
        "equality2",
        "(=[a,b])", "main",
        "(=[b,c])", "main_boundary_orint_Y_((=[k,l]))",
    )
    assert_failure(check_equality2, line, [line], state)


@register
def test_equality2_result_arity_three():
    """Result equality is arity 3 (malformed) -> reject."""
    state = make_state_minimal()
    line = make_proof_line(
        "(=[a,c,extra])", "main", "equality2",
        "(=[a,b])", "main",
        "(=[b,c])", "main",
    )
    assert_failure(check_equality2, line, [line], state)


@register
def test_equality2_eq1_arity_three():
    """eq1 is arity 3 (malformed) -> reject."""
    state = make_state_minimal()
    line = make_proof_line(
        "(=[a,c])", "main", "equality2",
        "(=[a,b,extra])", "main",
        "(=[b,c])", "main",
    )
    assert_failure(check_equality2, line, [line], state)


@register
def test_equality2_eq1_b_neq_eq2_a():
    """Middle term doesn't match: eq1.[1] is 'b', eq2.[0] is 'x'."""
    state = make_state_minimal()
    line = make_proof_line(
        "(=[a,c])", "main", "equality2",
        "(=[a,b])", "main",
        "(=[x,c])", "main",   # x != b — transitivity broken
    )
    assert_failure(check_equality2, line, [line], state)


@register
def test_equality2_result_endpoints_wrong():
    """result.[1] expected to equal eq2.[1] but mismatches."""
    state = make_state_minimal()
    line = make_proof_line(
        "(=[a,z])", "main", "equality2",
        "(=[a,b])", "main",
        "(=[b,c])", "main",   # eq2.[1] is c, not z
    )
    assert_failure(check_equality2, line, [line], state)


# ===========================================================================
#  tag: symmetry of equality
# ===========================================================================

@register
def test_symmetry_of_equality_rest_empty():
    """rest empty -> ``len(rest) < 2`` short-circuits."""
    state = make_state_minimal()
    line = make_proof_line("(=[a,b])", "main", "symmetry of equality")
    assert_failure(check_symmetry_of_equality, line, [line], state)


@register
def test_symmetry_of_equality_ns_different():
    """source_ns != line.namespace -> reject (no ancestry tolerance here)."""
    state = make_state_minimal()
    line = make_proof_line(
        "(=[a,b])", "main",
        "symmetry of equality",
        "(=[b,a])", "main_boundary_orint_X_((=[c,d]))",
    )
    assert_failure(check_symmetry_of_equality, line, [line], state)


@register
def test_symmetry_of_equality_result_arity_three():
    """Result has arity 3 -> reject (only arity 2 supported)."""
    state = make_state_minimal()
    line = make_proof_line(
        "(=[a,b,extra])", "main",
        "symmetry of equality",
        "(=[b,a])", "main",
    )
    assert_failure(check_symmetry_of_equality, line, [line], state)


@register
def test_symmetry_of_equality_source_arity_three():
    """Source has arity 3 -> reject."""
    state = make_state_minimal()
    line = make_proof_line(
        "(=[a,b])", "main",
        "symmetry of equality",
        "(=[b,a,extra])", "main",
    )
    assert_failure(check_symmetry_of_equality, line, [line], state)


@register
def test_symmetry_of_equality_args_not_swapped():
    """Args identical between source and result -> not a swap."""
    state = make_state_minimal()
    line = make_proof_line(
        "(=[a,b])", "main",
        "symmetry of equality",
        "(=[a,b])", "main",   # NOT swapped
    )
    assert_failure(check_symmetry_of_equality, line, [line], state)


@register
def test_symmetry_of_equality_only_one_arg_swapped():
    """First arg matches source's second, but second arg doesn't match
    source's first."""
    state = make_state_minimal()
    line = make_proof_line(
        "(=[b,c])", "main",
        "symmetry of equality",
        "(=[b,a])", "main",   # result.[0]==source.[1]? c vs b -> no
    )
    assert_failure(check_symmetry_of_equality, line, [line], state)


# ===========================================================================
#  tag: symmetry of inequality
# ===========================================================================

@register
def test_symmetry_of_inequality_rest_empty():
    """rest empty -> reject."""
    state = make_state_minimal()
    line = make_proof_line("!(=[a,b])", "main", "symmetry of inequality")
    assert_failure(check_symmetry_of_inequality, line, [line], state)


@register
def test_symmetry_of_inequality_ns_different():
    """Result and source namespaces differ -> reject."""
    state = make_state_minimal()
    line = make_proof_line(
        "!(=[a,b])", "main",
        "symmetry of inequality",
        "!(=[b,a])", "main_boundary_orint_X_((=[c,d]))",
    )
    assert_failure(check_symmetry_of_inequality, line, [line], state)


@register
def test_symmetry_of_inequality_result_not_negated_equality():
    """Result does NOT start with '!(=['."""
    state = make_state_minimal()
    line = make_proof_line(
        "(=[a,b])", "main",   # missing leading '!'
        "symmetry of inequality",
        "!(=[b,a])", "main",
    )
    assert_failure(check_symmetry_of_inequality, line, [line], state)


@register
def test_symmetry_of_inequality_source_not_negated_equality():
    """Source does NOT start with '!(=['."""
    state = make_state_minimal()
    line = make_proof_line(
        "!(=[a,b])", "main",
        "symmetry of inequality",
        "(=[b,a])", "main",   # missing leading '!'
    )
    assert_failure(check_symmetry_of_inequality, line, [line], state)


@register
def test_symmetry_of_inequality_arity_three():
    """Result negated-equality has arity 3 -> reject."""
    state = make_state_minimal()
    line = make_proof_line(
        "!(=[a,b,extra])", "main",
        "symmetry of inequality",
        "!(=[b,a])", "main",
    )
    assert_failure(check_symmetry_of_inequality, line, [line], state)


@register
def test_symmetry_of_inequality_args_not_swapped():
    """Args identical between source and result -> not a swap."""
    state = make_state_minimal()
    line = make_proof_line(
        "!(=[a,b])", "main",
        "symmetry of inequality",
        "!(=[a,b])", "main",   # NOT swapped
    )
    assert_failure(check_symmetry_of_inequality, line, [line], state)


if __name__ == "__main__":
    sys.exit(run_all_tests())
