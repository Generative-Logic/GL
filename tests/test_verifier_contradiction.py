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

"""Failure tests for contradiction / vacuous truth checkers.

Each test injects exactly ONE subtle malformation. See
``tests/test_harness.py`` for fixtures and runner.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_harness import (  # noqa: E402
    register, make_state_with_binaries, make_proof_line,
    assert_failure, run_all_tests,
)
from verifier import (  # noqa: E402
    check_contradiction, check_vacuous_truth,
)


# ===========================================================================
#  tag: contradiction
# ===========================================================================

@register
def test_contradiction_namespace_not_main():
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "!(in[a,N])",
        "main_boundary_orint_X_((=[a,b]))",
        "contradiction",
        "(in[a,N])", "main",
        "!(in[a,N])", "main",
        "(in[a,N])", "main",
    )
    assert_failure(check_contradiction, line, [line], state)


@register
def test_contradiction_rest_empty():
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line("!(in[a,N])", "main", "contradiction")
    assert_failure(check_contradiction, line, [line], state)


@register
def test_contradiction_rest_too_short():
    """rest has 5 entries; need >=6."""
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "!(in[a,N])", "main", "contradiction",
        "(in[a,N])", "main",
        "!(in[a,N])", "main",
        "(in[a,N])",
    )
    assert_failure(check_contradiction, line, [line], state)


@register
def test_contradiction_expr_ns_not_main():
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "!(in[a,N])", "main", "contradiction",
        "(in[a,N])", "main_boundary_orint_X_((=[a,b]))",
        "!(in[a,N])", "main",
        "(in[a,N])", "main",
    )
    assert_failure(check_contradiction, line, [line], state)


@register
def test_contradiction_neg_ns_not_main():
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "!(in[a,N])", "main", "contradiction",
        "(in[a,N])", "main",
        "!(in[a,N])", "main_boundary_orint_X_((=[a,b]))",
        "(in[a,N])", "main",
    )
    assert_failure(check_contradiction, line, [line], state)


@register
def test_contradiction_clean_ns_not_main():
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "!(in[a,N])", "main", "contradiction",
        "(in[a,N])", "main",
        "!(in[a,N])", "main",
        "(in[a,N])", "main_boundary_orint_X_((=[a,b]))",
    )
    assert_failure(check_contradiction, line, [line], state)


@register
def test_contradiction_line_expr_not_negated_clean_op():
    """line.expression != '!' + clean_op."""
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "!(in[a,M])",   # not "!" + "(in[a,N])"
        "main", "contradiction",
        "(in[a,N])", "main",
        "!(in[a,N])", "main",
        "(in[a,N])", "main",
    )
    assert_failure(check_contradiction, line, [line], state)


@register
def test_contradiction_positive_expr_clean_op_not_its_negation():
    """Complement polarity requires clean_op == '!' + line.expression;
    a positive row with an unrelated clean_op is rejected."""
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(in[a,N])", "main", "contradiction",
        "(in[a,N])", "main",
        "!(in[a,N])", "main",
        "!(in[b,N])", "main",
    )
    assert_failure(check_contradiction, line, [line], state)


@register
def test_contradiction_expr_and_neg_unrelated():
    """expr and neg_expr are NOT negations of each other."""
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "!(in[a,N])", "main", "contradiction",
        "(in[a,N])", "main",
        "!(in[b,N])", "main",    # different expression
        "(in[a,N])", "main",
    )
    assert_failure(check_contradiction, line, [line], state)


@register
def test_contradiction_clean_op_missing_from_chapter():
    """clean_op doesn't appear in chapter as task formulation."""
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "!(in[a,N])", "main", "contradiction",
        "(in[a,N])", "main",
        "!(in[a,N])", "main",
        "(in[a,N])", "main",
    )
    # Chapter has ONLY the contradiction row; no task formulation row.
    assert_failure(check_contradiction, line, [line], state)


@register
def test_contradiction_clean_op_present_but_wrong_tag():
    """clean_op exists in chapter but tagged as something other than
    'task formulation' -> reject."""
    state = make_state_with_binaries(("Peano",))
    contr_line = make_proof_line(
        "!(in[a,N])", "main", "contradiction",
        "(in[a,N])", "main",
        "!(in[a,N])", "main",
        "(in[a,N])", "main",
    )
    impostor = make_proof_line(
        "(in[a,N])", "main", "implication",   # wrong tag
        "(>[v1](in[v1,N])(in[v1,N]))", "main",
        "(in[a,N])", "main",
    )
    chapter = [contr_line, impostor]
    assert_failure(check_contradiction, contr_line, chapter, state)


# ===========================================================================
#  tag: vacuous truth
# ===========================================================================

@register
def test_vacuous_truth_namespace_not_main():
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(eq[v1,v1])",
        "main_boundary_orint_X_((=[a,b]))",
        "vacuous truth",
        "(in[v1,N])", "main",
        "!(in[v1,N])", "main",
        "(some_lb_key)", "main",
    )
    assert_failure(check_vacuous_truth, line, [line], state)


@register
def test_vacuous_truth_rest_empty():
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line("(eq[v1,v1])", "main", "vacuous truth")
    assert_failure(check_vacuous_truth, line, [line], state)


@register
def test_vacuous_truth_rest_too_short():
    """5 entries instead of >=6."""
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(eq[v1,v1])", "main", "vacuous truth",
        "(in[v1,N])", "main",
        "!(in[v1,N])", "main",
        "(some_lb_key)",
    )
    assert_failure(check_vacuous_truth, line, [line], state)


@register
def test_vacuous_truth_ns1_not_main():
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(eq[v1,v1])", "main", "vacuous truth",
        "(in[v1,N])", "main_boundary_orint_X_((=[a,b]))",
        "!(in[v1,N])", "main",
        "(some_lb_key)", "main",
    )
    assert_failure(check_vacuous_truth, line, [line], state)


@register
def test_vacuous_truth_ns2_not_main():
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(eq[v1,v1])", "main", "vacuous truth",
        "(in[v1,N])", "main",
        "!(in[v1,N])", "main_boundary_orint_X_((=[a,b]))",
        "(some_lb_key)", "main",
    )
    assert_failure(check_vacuous_truth, line, [line], state)


@register
def test_vacuous_truth_ns3_not_main():
    """rest[5] (lb_key's ns) is not main."""
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(eq[v1,v1])", "main", "vacuous truth",
        "(in[v1,N])", "main",
        "!(in[v1,N])", "main",
        "(some_lb_key)", "main_boundary_orint_X_((=[a,b]))",
    )
    assert_failure(check_vacuous_truth, line, [line], state)


@register
def test_vacuous_truth_pair_not_negations():
    """expr and neg_expr are unrelated."""
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(eq[v1,v1])", "main", "vacuous truth",
        "(in[v1,N])", "main",
        "!(in[v2,N])", "main",   # different vars; not a negation pair
        "(some_lb_key)", "main",
    )
    assert_failure(check_vacuous_truth, line, [line], state)


@register
def test_vacuous_truth_negation_has_extra_char():
    """neg_expr almost equals '!' + expr but has trailing junk."""
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(eq[v1,v1])", "main", "vacuous truth",
        "(in[v1,N])", "main",
        "!(in[v1,N])-trailing", "main",
        "(some_lb_key)", "main",
    )
    assert_failure(check_vacuous_truth, line, [line], state)


@register
def test_vacuous_truth_both_unnegated():
    """Neither expr nor neg_expr starts with '!'."""
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(eq[v1,v1])", "main", "vacuous truth",
        "(in[v1,N])", "main",
        "(in[v1,N])", "main",    # same expression, not a negation
        "(some_lb_key)", "main",
    )
    assert_failure(check_vacuous_truth, line, [line], state)


@register
def test_vacuous_truth_both_negated():
    """Both expr and neg_expr start with '!'."""
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(eq[v1,v1])", "main", "vacuous truth",
        "!(in[v1,N])", "main",
        "!(in[v1,N])", "main",   # same negated, NOT a negation pair
        "(some_lb_key)", "main",
    )
    assert_failure(check_vacuous_truth, line, [line], state)


if __name__ == "__main__":
    sys.exit(run_all_tests())
