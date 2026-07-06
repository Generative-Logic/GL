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

"""Failure tests for integration-family checkers (expansion for integration,
premise element, validity name, reformulation for integration {and, >[bound],
>[]}).

Each test injects exactly ONE subtle malformation. See
``tests/test_harness.py`` for fixtures and runner.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_harness import (  # noqa: E402
    register, make_state_with_binaries, make_proof_line,
    assert_failure, run_all_tests, set_chapter_context,
)
from verifier import (  # noqa: E402
    check_expansion_for_integration, check_premise_element,
    check_validity_name,
    check_reformulation_for_integration_and,
    check_reformulation_for_integration_bound,
    check_reformulation_for_integration_empty,
)


# ===========================================================================
#  tag: expansion for integration
# ===========================================================================

@register
def test_expansion_for_integration_rest_empty():
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(&[a,b])", "main", "expansion for integration")
    assert_failure(check_expansion_for_integration, line, [line], state)


@register
def test_expansion_for_integration_rest_length_one():
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(&[a,b])", "main", "expansion for integration",
        "(some_compact[a,b])",
    )
    assert_failure(check_expansion_for_integration, line, [line], state)


@register
def test_expansion_for_integration_ns_mismatch():
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(&[a,b])", "main", "expansion for integration",
        "(some_compact[a,b])",
        "main_boundary_orint_X_((=[c,d]))",
    )
    assert_failure(check_expansion_for_integration, line, [line], state)


@register
def test_expansion_for_integration_core_not_in_any_binary():
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(&[a,b])", "main", "expansion for integration",
        "(nonexistent_core[a,b])", "main",
    )
    assert_failure(check_expansion_for_integration, line, [line], state)


@register
def test_expansion_for_integration_expand_fails():
    state = make_state_with_binaries(("Peano",))
    state.gl_binaries = dict(state.gl_binaries)
    state.gl_binaries["Peano"] = dict(state.gl_binaries["Peano"])
    state.gl_binaries["Peano"]["partial_and"] = {
        "category": "and",
        "signature": "(partial_and[u_1,u_2])",
        "elements": ["(in[u_1,N])", "(eq[u_1,u_2])"],
    }
    state.current_gl_binary = state.gl_binaries["Peano"]
    line = make_proof_line(
        "(unrelated[x,y,z])", "main", "expansion for integration",
        "(partial_and[a,b])", "main",
    )
    assert_failure(check_expansion_for_integration, line, [line], state)


@register
def test_expansion_for_integration_integration_goal_postfix_misuse():
    """Right side has _integration_goal postfix but the inner core still
    doesn't match anything."""
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(&[a,b])", "main", "expansion for integration",
        "(mystery_core[a,b])_integration_goal", "main",
    )
    assert_failure(check_expansion_for_integration, line, [line], state)


@register
def test_expansion_for_integration_arity_mismatch():
    state = make_state_with_binaries(("Peano",))
    state.gl_binaries = dict(state.gl_binaries)
    state.gl_binaries["Peano"] = dict(state.gl_binaries["Peano"])
    state.gl_binaries["Peano"]["arity2"] = {
        "category": "and",
        "signature": "(arity2[u_1,u_2])",
        "elements": ["(in[u_1,N])", "(in[u_2,N])"],
    }
    state.current_gl_binary = state.gl_binaries["Peano"]
    line = make_proof_line(
        "(&[a,b])", "main", "expansion for integration",
        "(arity2[a,b,c])", "main",   # arity 3 vs signature arity 2
    )
    assert_failure(check_expansion_for_integration, line, [line], state)


@register
def test_expansion_for_integration_left_unrelated():
    state = make_state_with_binaries(("Peano",))
    state.gl_binaries = dict(state.gl_binaries)
    state.gl_binaries["Peano"] = dict(state.gl_binaries["Peano"])
    state.gl_binaries["Peano"]["test_and"] = {
        "category": "and",
        "signature": "(test_and[u_1])",
        "elements": ["(in[u_1,N])"],
    }
    state.current_gl_binary = state.gl_binaries["Peano"]
    line = make_proof_line(
        "(completely[unrelated,target])", "main", "expansion for integration",
        "(test_and[a])", "main",
    )
    assert_failure(check_expansion_for_integration, line, [line], state)


# ===========================================================================
#  tag: premise element
# ===========================================================================

@register
def test_premise_element_rest_empty():
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(in[a,N])", "main_boundary_(impl_clean_sig)", "premise element")
    assert_failure(check_premise_element, line, [line], state)


@register
def test_premise_element_expr_not_in_origin_premises():
    state = make_state_with_binaries(("Peano",))
    expansion_line = make_proof_line(
        "(>[v1](in[v1,N])(in[v1,N]))", "main",
        "expansion for integration",
        "(some_compact)", "main",
    )
    premise_line = make_proof_line(
        "(in[unrelated,N])",   # not a premise of the origin
        "main_boundary_(some_compact)",
        "premise element",
        "(>[v1](in[v1,N])(in[v1,N]))", "main",
    )
    assert_failure(check_premise_element, premise_line,
                   [expansion_line, premise_line], state)


@register
def test_premise_element_no_expansion_for_integration_row():
    """Origin impl is a valid implication but no expansion-for-integration row
    exists in chapter to vouch for the cleanSig namespace."""
    state = make_state_with_binaries(("Peano",))
    premise_line = make_proof_line(
        "(in[v1,N])",
        "main_boundary_(some_compact)",
        "premise element",
        "(>[v1](in[v1,N])(eq[v1,v1]))", "main",
    )
    assert_failure(check_premise_element, premise_line,
                   [premise_line], state)


@register
def test_premise_element_origin_wrong_tag():
    """Chapter row has the origin impl but with wrong tag (not
    expansion-for-integration)."""
    state = make_state_with_binaries(("Peano",))
    impostor = make_proof_line(
        "(>[v1](in[v1,N])(eq[v1,v1]))", "main", "implication",  # wrong tag
        "(other_impl)", "main",
        "(in[v1,N])", "main",
    )
    premise_line = make_proof_line(
        "(in[v1,N])",
        "main_boundary_(some_compact)",
        "premise element",
        "(>[v1](in[v1,N])(eq[v1,v1]))", "main",
    )
    assert_failure(check_premise_element, premise_line,
                   [impostor, premise_line], state)


@register
def test_premise_element_origin_rest_empty():
    """Expansion-for-integration row found, but rest is empty so cleanSig
    can't be extracted."""
    state = make_state_with_binaries(("Peano",))
    expansion = make_proof_line(
        "(>[v1](in[v1,N])(eq[v1,v1]))", "main",
        "expansion for integration",
    )
    premise_line = make_proof_line(
        "(in[v1,N])",
        "main_boundary_(some_compact)",
        "premise element",
        "(>[v1](in[v1,N])(eq[v1,v1]))", "main",
    )
    assert_failure(check_premise_element, premise_line,
                   [expansion, premise_line], state)


@register
def test_premise_element_namespace_not_rooted_in_clean_sig():
    """premise element's namespace doesn't equal cleanSig and doesn't end
    with `_boundary_<cleanSig>`."""
    state = make_state_with_binaries(("Peano",))
    expansion = make_proof_line(
        "(>[v1](in[v1,N])(eq[v1,v1]))", "main",
        "expansion for integration",
        "(some_compact)", "main",
    )
    premise_line = make_proof_line(
        "(in[v1,N])",
        "main_boundary_(completely_different_compact)",   # not (some_compact)
        "premise element",
        "(>[v1](in[v1,N])(eq[v1,v1]))", "main",
    )
    assert_failure(check_premise_element, premise_line,
                   [expansion, premise_line], state)


@register
def test_premise_element_origin_form_mismatch_with_postfix():
    """The premise element row carries the origin impl WITHOUT
    `_integration_goal` postfix, but the chapter's expansion-for-integration
    row carries it WITH the postfix; expression equality required."""
    state = make_state_with_binaries(("Peano",))
    expansion = make_proof_line(
        "(>[v1](in[v1,N])(eq[v1,v1]))_integration_goal", "main",
        "expansion for integration",
        "(some_compact)", "main",
    )
    premise_line = make_proof_line(
        "(in[v1,N])",
        "main_boundary_(some_compact)",
        "premise element",
        "(>[v1](in[v1,N])(eq[v1,v1]))", "main",   # NO postfix
    )
    # The chapter walk's `ch_line.expression == origin_impl` test fails.
    assert_failure(check_premise_element, premise_line,
                   [expansion, premise_line], state)


@register
def test_premise_element_namespace_unrelated_string():
    """ns is a completely-random string with no relation to cleanSig."""
    state = make_state_with_binaries(("Peano",))
    expansion = make_proof_line(
        "(>[v1](in[v1,N])(eq[v1,v1]))", "main",
        "expansion for integration",
        "(some_compact)", "main",
    )
    premise_line = make_proof_line(
        "(in[v1,N])",
        "completely_unrelated_string",
        "premise element",
        "(>[v1](in[v1,N])(eq[v1,v1]))", "main",
    )
    assert_failure(check_premise_element, premise_line,
                   [expansion, premise_line], state)


# ===========================================================================
#  tag: validity name
# ===========================================================================

@register
def test_validity_name_rest_empty():
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line("(some_impl_name)", "main", "validity name")
    assert_failure(check_validity_name, line, [line], state)


@register
def test_validity_name_no_matching_expansion_row():
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(some_impl_name)", "main", "validity name",
        "(some_target_expr)", "main",
    )
    # No `expansion for integration` row in chapter referencing
    # `(some_impl_name)`.
    assert_failure(check_validity_name, line, [line], state)


@register
def test_validity_name_matching_row_wrong_tag():
    state = make_state_with_binaries(("Peano",))
    impostor = make_proof_line(
        "(>[v1](in[v1,N])(eq[v1,v1]))", "main", "implication",   # wrong tag
        "(some_impl_name)", "main",
    )
    line = make_proof_line(
        "(some_impl_name)", "main", "validity name",
        "(eq[v1,v1])", "main",
    )
    assert_failure(check_validity_name, line, [impostor, line], state)


@register
def test_validity_name_matching_row_rest_empty():
    state = make_state_with_binaries(("Peano",))
    expansion = make_proof_line(
        "(>[v1](in[v1,N])(eq[v1,v1]))", "main",
        "expansion for integration",   # rest empty
    )
    line = make_proof_line(
        "(some_impl_name)", "main", "validity name",
        "(eq[v1,v1])", "main",
    )
    assert_failure(check_validity_name, line, [expansion, line], state)


@register
def test_validity_name_head_mismatch_with_target():
    state = make_state_with_binaries(("Peano",))
    expansion = make_proof_line(
        "(>[v1](in[v1,N])(eq[v1,v1]))", "main",
        "expansion for integration",
        "(some_impl_name)", "main",
    )
    line = make_proof_line(
        "(some_impl_name)", "main", "validity name",
        "(in[v1,N])", "main",   # head of expansion is (eq[v1,v1]), not (in[v1,N])
    )
    assert_failure(check_validity_name, line, [expansion, line], state)


@register
def test_validity_name_impl_name_mismatch_after_strip():
    """Chapter row has impl_name '(other_name)'; line.expression is
    '(some_impl_name)' — they differ after stripping integration_goal."""
    state = make_state_with_binaries(("Peano",))
    expansion = make_proof_line(
        "(>[v1](in[v1,N])(eq[v1,v1]))", "main",
        "expansion for integration",
        "(other_name)", "main",
    )
    line = make_proof_line(
        "(some_impl_name)", "main", "validity name",
        "(eq[v1,v1])", "main",
    )
    assert_failure(check_validity_name, line, [expansion, line], state)


# ===========================================================================
#  tag: reformulation for integration and
# ===========================================================================

@register
def test_reformulation_for_integration_and_rest_empty():
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(>[](in[a,N])(eq[a,a]))", "main",
        "reformulation for integration and",
    )
    assert_failure(check_reformulation_for_integration_and, line, [line], state)


@register
def test_reformulation_for_integration_and_no_expansion_row():
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(>[](in[a,N])(eq[a,a]))", "main",
        "reformulation for integration and",
        "(&(in[a,N])(eq[a,a]))", "main",
    )
    assert_failure(check_reformulation_for_integration_and, line, [line], state)


@register
def test_reformulation_for_integration_and_ns_mismatch():
    state = make_state_with_binaries(("Peano",))
    expansion = make_proof_line(
        "(&(in[a,N])(eq[a,a]))", "main", "expansion for integration",
        "(some_compact)", "main",
    )
    line = make_proof_line(
        "(>[](in[a,N])(eq[a,a]))",
        "main_boundary_orint_X_((=[c,d]))",
        "reformulation for integration and",
        "(&(in[a,N])(eq[a,a]))", "main_boundary_orint_X_((=[c,d]))",
    )
    assert_failure(check_reformulation_for_integration_and, line,
                   [expansion, line], state)


@register
def test_reformulation_for_integration_and_entry_wrong_category():
    """Expansion row points to a compact whose binary entry is 'existence',
    not 'and'."""
    state = make_state_with_binaries(("Peano",))
    state.gl_binaries = dict(state.gl_binaries)
    state.gl_binaries["Peano"] = dict(state.gl_binaries["Peano"])
    state.gl_binaries["Peano"]["existence_test"] = {
        "category": "existence",
        "signature": "(existence_test[u_1])",
        "elements": ["(in[u_1,N])", "(eq[u_1,u_1])"],
    }
    state.current_gl_binary = state.gl_binaries["Peano"]
    expansion = make_proof_line(
        "(>[v1](in[v1,N])(eq[v1,v1]))", "main", "expansion for integration",
        "(existence_test[v1])", "main",
    )
    line = make_proof_line(
        "(>[v1](in[v1,N])(existence_test[v1]))", "main",
        "reformulation for integration and",
        "(>[v1](in[v1,N])(eq[v1,v1]))", "main",
    )
    assert_failure(check_reformulation_for_integration_and, line,
                   [expansion, line], state)


@register
def test_reformulation_for_integration_and_right_not_and_shape():
    """Right_expr_clean doesn't start with '(&' so _check_reformulation_integration_and
    rejects."""
    state = make_state_with_binaries(("Peano",))
    state.gl_binaries = dict(state.gl_binaries)
    state.gl_binaries["Peano"] = dict(state.gl_binaries["Peano"])
    state.gl_binaries["Peano"]["and_compact"] = {
        "category": "and",
        "signature": "(and_compact[u_1])",
        "elements": ["(in[u_1,N])"],
    }
    state.current_gl_binary = state.gl_binaries["Peano"]
    expansion = make_proof_line(
        "(in[a,N])", "main", "expansion for integration",   # NOT (&...)
        "(and_compact[a])", "main",
    )
    line = make_proof_line(
        "(>[](in[a,N])(and_compact[a]))", "main",
        "reformulation for integration and",
        "(in[a,N])", "main",
    )
    assert_failure(check_reformulation_for_integration_and, line,
                   [expansion, line], state)


@register
def test_reformulation_for_integration_and_reconstruction_mismatch():
    """All preconditions hold but reconstructed implication != line.expression."""
    state = make_state_with_binaries(("Peano",))
    state.gl_binaries = dict(state.gl_binaries)
    state.gl_binaries["Peano"] = dict(state.gl_binaries["Peano"])
    state.gl_binaries["Peano"]["and_compact"] = {
        "category": "and",
        "signature": "(and_compact[u_1])",
        "elements": ["(in[u_1,N])"],
    }
    state.current_gl_binary = state.gl_binaries["Peano"]
    expansion = make_proof_line(
        "(&(in[a,N])(eq[a,a]))", "main", "expansion for integration",
        "(and_compact[a])", "main",
    )
    line = make_proof_line(
        "(unrelated[a])",   # not the rebuilt implication
        "main",
        "reformulation for integration and",
        "(&(in[a,N])(eq[a,a]))", "main",
    )
    assert_failure(check_reformulation_for_integration_and, line,
                   [expansion, line], state)


# ===========================================================================
#  tag: reformulation for integration >[bound]
# ===========================================================================

@register
def test_reformulation_for_integration_bound_rest_empty():
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(>[v1](in[v1,N])(eq[v1,v1]))", "main",
        "reformulation for integration >[bound]",
    )
    assert_failure(check_reformulation_for_integration_bound, line,
                   [line], state)


@register
def test_reformulation_for_integration_bound_no_expansion_row():
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(>[v1](in[v1,N])(eq[v1,v1]))", "main",
        "reformulation for integration >[bound]",
        "(some_compact)", "main",
    )
    assert_failure(check_reformulation_for_integration_bound, line,
                   [line], state)


@register
def test_reformulation_for_integration_bound_entry_wrong_category():
    state = make_state_with_binaries(("Peano",))
    state.gl_binaries = dict(state.gl_binaries)
    state.gl_binaries["Peano"] = dict(state.gl_binaries["Peano"])
    state.gl_binaries["Peano"]["and_only"] = {
        "category": "and",
        "signature": "(and_only[u_1])",
        "elements": ["(in[u_1,N])"],
    }
    state.current_gl_binary = state.gl_binaries["Peano"]
    expansion = make_proof_line(
        "(&(in[a,N]))", "main", "expansion for integration",
        "(and_only[a])", "main",
    )
    line = make_proof_line(
        "(>[v1](in[v1,N])(and_only[v1]))", "main",
        "reformulation for integration >[bound]",
        "(&(in[a,N]))", "main",
    )
    assert_failure(check_reformulation_for_integration_bound, line,
                   [expansion, line], state)


@register
def test_reformulation_for_integration_bound_outer_empty():
    """Outermost `>[...]` is empty; this tag REQUIRES non-empty bound vars."""
    state = make_state_with_binaries(("Peano",))
    state.gl_binaries = dict(state.gl_binaries)
    state.gl_binaries["Peano"] = dict(state.gl_binaries["Peano"])
    state.gl_binaries["Peano"]["bound_exist"] = {
        "category": "existence",
        "signature": "(bound_exist[u_1])",
        "elements": ["(in[u_1,N])", "(eq[u_1,u_1])"],
    }
    state.current_gl_binary = state.gl_binaries["Peano"]
    expansion = make_proof_line(
        "(some_expanded_form)", "main", "expansion for integration",
        "(bound_exist[a])", "main",
    )
    line = make_proof_line(
        "(>[](in[a,N])(eq[a,a]))",   # empty >[]
        "main",
        "reformulation for integration >[bound]",
        "(some_expanded_form)", "main",
    )
    assert_failure(check_reformulation_for_integration_bound, line,
                   [expansion, line], state)


@register
def test_reformulation_for_integration_bound_existence_mismatch():
    """Outermost >[v1] non-empty, entry category 'existence', but the
    existence rebuild doesn't match line.expression."""
    state = make_state_with_binaries(("Peano",))
    state.gl_binaries = dict(state.gl_binaries)
    state.gl_binaries["Peano"] = dict(state.gl_binaries["Peano"])
    state.gl_binaries["Peano"]["bound_exist2"] = {
        "category": "existence",
        "signature": "(bound_exist2[u_1])",
        "elements": ["(in[u_1,N])", "(eq[u_1,u_1])"],
    }
    state.current_gl_binary = state.gl_binaries["Peano"]
    expansion = make_proof_line(
        "(some_form)", "main", "expansion for integration",
        "(bound_exist2[a])", "main",
    )
    line = make_proof_line(
        "(>[v1](completely[unrelated,v1])(eq[v1,v1]))", "main",
        "reformulation for integration >[bound]",
        "(some_form)", "main",
    )
    assert_failure(check_reformulation_for_integration_bound, line,
                   [expansion, line], state)


@register
def test_reformulation_for_integration_bound_compact_head_mismatch():
    """Preamble fails because compact != proof_head."""
    state = make_state_with_binaries(("Peano",))
    state.gl_binaries = dict(state.gl_binaries)
    state.gl_binaries["Peano"] = dict(state.gl_binaries["Peano"])
    state.gl_binaries["Peano"]["compact_check"] = {
        "category": "existence",
        "signature": "(compact_check[u_1])",
        "elements": ["(in[u_1,N])", "(eq[u_1,u_1])"],
    }
    state.current_gl_binary = state.gl_binaries["Peano"]
    expansion = make_proof_line(
        "(some_form)", "main", "expansion for integration",
        "(compact_check[a])", "main",
    )
    line = make_proof_line(
        "(>[v1](in[v1,N])(other_head[v1]))",   # head is (other_head[v1])
        "main",
        "reformulation for integration >[bound]",
        "(some_form)", "main",
    )
    # proof_head from disintegrate is (other_head[v1]); compact is
    # (compact_check[a]) -> mismatch -> preamble None -> reject.
    assert_failure(check_reformulation_for_integration_bound, line,
                   [expansion, line], state)


# ===========================================================================
#  tag: reformulation for integration >[]
# ===========================================================================

@register
def test_reformulation_for_integration_empty_rest_empty():
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(>[](in[a,N])(eq[a,a]))", "main",
        "reformulation for integration >[]",
    )
    assert_failure(check_reformulation_for_integration_empty, line,
                   [line], state)


@register
def test_reformulation_for_integration_empty_no_expansion_row():
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(>[](in[a,N])(eq[a,a]))", "main",
        "reformulation for integration >[]",
        "(some_compact)", "main",
    )
    assert_failure(check_reformulation_for_integration_empty, line,
                   [line], state)


@register
def test_reformulation_for_integration_empty_entry_wrong_category():
    state = make_state_with_binaries(("Peano",))
    state.gl_binaries = dict(state.gl_binaries)
    state.gl_binaries["Peano"] = dict(state.gl_binaries["Peano"])
    state.gl_binaries["Peano"]["and_only2"] = {
        "category": "and",
        "signature": "(and_only2[u_1])",
        "elements": ["(in[u_1,N])"],
    }
    state.current_gl_binary = state.gl_binaries["Peano"]
    expansion = make_proof_line(
        "(some_form)", "main", "expansion for integration",
        "(and_only2[a])", "main",
    )
    line = make_proof_line(
        "(>[](in[a,N])(and_only2[a]))", "main",
        "reformulation for integration >[]",
        "(some_form)", "main",
    )
    assert_failure(check_reformulation_for_integration_empty, line,
                   [expansion, line], state)


@register
def test_reformulation_for_integration_empty_outer_non_empty():
    """Outermost >[v1] is NOT empty; this tag REQUIRES empty >[]."""
    state = make_state_with_binaries(("Peano",))
    state.gl_binaries = dict(state.gl_binaries)
    state.gl_binaries["Peano"] = dict(state.gl_binaries["Peano"])
    state.gl_binaries["Peano"]["empty_exist"] = {
        "category": "existence",
        "signature": "(empty_exist[u_1])",
        "elements": ["(in[u_1,N])", "(eq[u_1,u_1])"],
    }
    state.current_gl_binary = state.gl_binaries["Peano"]
    expansion = make_proof_line(
        "(some_form)", "main", "expansion for integration",
        "(empty_exist[a])", "main",
    )
    line = make_proof_line(
        "(>[v1](in[v1,N])(eq[v1,v1]))",   # non-empty >[v1]
        "main",
        "reformulation for integration >[]",
        "(some_form)", "main",
    )
    assert_failure(check_reformulation_for_integration_empty, line,
                   [expansion, line], state)


@register
def test_reformulation_for_integration_empty_existence_mismatch():
    """Outer >[] is empty, entry existence, but rebuild fails."""
    state = make_state_with_binaries(("Peano",))
    state.gl_binaries = dict(state.gl_binaries)
    state.gl_binaries["Peano"] = dict(state.gl_binaries["Peano"])
    state.gl_binaries["Peano"]["empty_exist2"] = {
        "category": "existence",
        "signature": "(empty_exist2[u_1])",
        "elements": ["(in[u_1,N])", "(eq[u_1,u_1])"],
    }
    state.current_gl_binary = state.gl_binaries["Peano"]
    expansion = make_proof_line(
        "(some_form)", "main", "expansion for integration",
        "(empty_exist2[a])", "main",
    )
    line = make_proof_line(
        "(>[](completely[bogus,a])(other[a]))", "main",
        "reformulation for integration >[]",
        "(some_form)", "main",
    )
    # The compact (empty_exist2[a]) doesn't match the proof_head
    # ((other[a])) -> preamble None -> reject.
    assert_failure(check_reformulation_for_integration_empty, line,
                   [expansion, line], state)


@register
def test_reformulation_for_integration_empty_arity_mismatch():
    """Existence signature arity 1; head supplied arity 2."""
    state = make_state_with_binaries(("Peano",))
    state.gl_binaries = dict(state.gl_binaries)
    state.gl_binaries["Peano"] = dict(state.gl_binaries["Peano"])
    state.gl_binaries["Peano"]["arity1_exist"] = {
        "category": "existence",
        "signature": "(arity1_exist[u_1])",
        "elements": ["(in[u_1,N])", "(eq[u_1,u_1])"],
    }
    state.current_gl_binary = state.gl_binaries["Peano"]
    expansion = make_proof_line(
        "(some_form)", "main", "expansion for integration",
        "(arity1_exist[a,b])", "main",   # arity 2 vs signature arity 1
    )
    line = make_proof_line(
        "(>[](in[a,N])(arity1_exist[a,b]))", "main",
        "reformulation for integration >[]",
        "(some_form)", "main",
    )
    assert_failure(check_reformulation_for_integration_empty, line,
                   [expansion, line], state)


# ===========================================================================
#  premise-anchor chapter→binary binding (regression test)
# ===========================================================================

@register
def test_cross_anchor_binding_picks_premise_anchor():
    """Cross-anchor connection theorems mention multiple Anchor<Tag>
    substrings. The principled binding is the PREMISE anchor (the world
    the proof's assumptions live in), not whichever tag the iteration
    happens to hit first. This test exercises a Gauss-premise / Peano-
    conclusion theorem (the shape of chapter 103) and confirms binding
    is independent of gl_binaries dict insertion order — the bug a
    Linux container surfaced on 2026-05-12 where filesystem listdir
    order swapped Gauss and Peano position."""
    thm = (
        "(>[N,i0,s,+,*,i1](AnchorGauss[N,i0,s,+,*,i1,i2,id])"
        "(AnchorPeano[N,i0,s,+,*,i1]))",
        "direct",
        "-1",
    )

    # Natural order (sorted Gauss, IncubatorGauss, ..., Peano, shared)
    state1 = make_state_with_binaries(("Gauss",))
    set_chapter_context(state1, thm=thm)
    assert state1.current_gl_binary is state1.gl_binaries["Gauss"], (
        "premise-anchor binding picked the wrong binary on natural dict order")

    # Reversed dict order — what a non-alphabetical filesystem produced
    state2 = make_state_with_binaries(("Gauss",))
    state2.gl_binaries = dict(reversed(list(state2.gl_binaries.items())))
    set_chapter_context(state2, thm=thm)
    assert state2.current_gl_binary is state2.gl_binaries["Gauss"], (
        "premise-anchor binding must be order-independent — reversed dict "
        "still picks Gauss (the premise anchor)")


if __name__ == "__main__":
    sys.exit(run_all_tests())
