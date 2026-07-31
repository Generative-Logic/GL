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

"""Failure tests for disintegration / expansion / recursion / anchor handling.

Each test injects exactly ONE subtle malformation and asserts the verifier
rejects the row. See ``tests/test_harness.py`` for the registry, fixtures,
and runner.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_harness import (  # noqa: E402
    register, make_state_minimal, make_state_with_binaries,
    make_proof_line, set_chapter_context, assert_failure, run_all_tests,
)
from verifier import (  # noqa: E402
    check_disintegration, check_expansion, check_recursion,
    check_anchor_handling, check_compilation,
)


# ===========================================================================
#  tag: disintegration
# ===========================================================================

@register
def test_disintegration_rest_empty():
    """rest empty -> ``len(rest) < 2`` short-circuits."""
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line("(in[a,N])", "main", "disintegration")
    assert_failure(check_disintegration, line, [line], state)


@register
def test_disintegration_rest_one_field():
    """Only compound but no compound_ns -> reject."""
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(in[a,N])", "main", "disintegration",
        "(&[a,b])",
    )
    assert_failure(check_disintegration, line, [line], state)


@register
def test_disintegration_ns_mismatch():
    """line.namespace != compound_ns -> reject."""
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(in[a,N])", "main", "disintegration",
        "(&[a,b])", "main_boundary_orint_X_((=[c,d]))",
    )
    assert_failure(check_disintegration, line, [line], state)


@register
def test_disintegration_no_matching_expansion_line():
    """No chapter row has the compound as a left-side ``expansion`` line."""
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(in[a,N])", "main", "disintegration",
        "(unknown_compound[a,b])", "main",
    )
    # Only the disintegration row itself is in the chapter; no expansion row.
    assert_failure(check_disintegration, line, [line], state)


@register
def test_disintegration_compound_ns_in_expansion_line_mismatches():
    """An ``expansion`` row exists for the compound but at a different
    namespace than the disintegration row claims."""
    state = make_state_with_binaries(("Peano",))
    disint_line = make_proof_line(
        "(in[a,N])", "main", "disintegration",
        "(some_compound[a,b])", "main",
    )
    expansion_line = make_proof_line(
        "(some_compound[a,b])", "main_boundary_orint_X_((=[c,d]))",
        "expansion",
        "(some_compact[a,b])", "main",
    )
    # disint_line's compound_ns says main, but the matching expansion is at
    # a child scope -> the loop's `ch_line.namespace == compound_ns` guard
    # fails, no match path executes, returns False.
    assert_failure(check_disintegration, disint_line,
                   [disint_line, expansion_line], state)


@register
def test_disintegration_core_not_in_binary():
    """compact's core is not present in any GL binary."""
    state = make_state_with_binaries(("Peano",))
    disint_line = make_proof_line(
        "(in[a,N])", "main", "disintegration",
        "(some_compound[a,b])", "main",
    )
    expansion_line = make_proof_line(
        "(some_compound[a,b])", "main", "expansion",
        "(nonexistent_compact_core[a,b])", "main",
    )
    assert_failure(check_disintegration, disint_line,
                   [disint_line, expansion_line], state)


@register
def test_disintegration_and_target_not_in_elements():
    """Category 'and', but the disintegrated target is not among the
    instantiated elements."""
    state = make_state_with_binaries(("Peano",))
    state.gl_binaries = dict(state.gl_binaries)
    state.gl_binaries["Peano"] = dict(state.gl_binaries["Peano"])
    state.gl_binaries["Peano"]["andtest1"] = {
        "category": "and",
        "signature": "(andtest1[u_1,u_2])",
        "elements": ["(in[u_1,N])", "(eq[u_1,u_2])"],
    }
    state.current_gl_binary = state.gl_binaries["Peano"]
    disint_line = make_proof_line(
        "(neither[a,b])", "main", "disintegration",  # not in elements
        "(some_compound[a,b])", "main",
    )
    expansion_line = make_proof_line(
        "(some_compound[a,b])", "main", "expansion",
        "(andtest1[a,b])", "main",
    )
    assert_failure(check_disintegration, disint_line,
                   [disint_line, expansion_line], state)


@register
def test_disintegration_and_signature_arity_mismatch():
    """Category 'and', but compact's actual-args count differs from
    signature args count."""
    state = make_state_with_binaries(("Peano",))
    state.gl_binaries = dict(state.gl_binaries)
    state.gl_binaries["Peano"] = dict(state.gl_binaries["Peano"])
    state.gl_binaries["Peano"]["andtest2"] = {
        "category": "and",
        "signature": "(andtest2[u_1,u_2])",     # arity 2
        "elements": ["(in[u_1,N])", "(eq[u_1,u_2])"],
    }
    state.current_gl_binary = state.gl_binaries["Peano"]
    disint_line = make_proof_line(
        "(in[a,N])", "main", "disintegration",
        "(some_compound[a,b])", "main",
    )
    expansion_line = make_proof_line(
        "(some_compound[a,b])", "main", "expansion",
        "(andtest2[a,b,c])", "main",   # arity 3 — mismatch
    )
    assert_failure(check_disintegration, disint_line,
                   [disint_line, expansion_line], state)


@register
def test_disintegration_or_premise_set_mismatch():
    """Category 'or', actual premises don't match expected
    mutual-exclusion shape: head is a disjunct, but premises set
    is missing one of the `!D_other` entries."""
    state = make_state_with_binaries(("Peano",))
    state.gl_binaries = dict(state.gl_binaries)
    state.gl_binaries["Peano"] = dict(state.gl_binaries["Peano"])
    state.gl_binaries["Peano"]["ortest1"] = {
        "category": "or",
        "signature": "(ortest1[u_1,u_2,u_3])",
        "elements": ["(eq[u_1])", "(eq[u_2])", "(eq[u_3])"],
    }
    state.current_gl_binary = state.gl_binaries["Peano"]
    # Result is the mutual-exclusion implication for disjunct (eq[a]).
    # Expected premises = sorted([!(eq[b]), !(eq[c])]); we deliver only one
    # of them.
    disint_line = make_proof_line(
        "(>[]!(eq[b])(eq[a]))", "main", "disintegration",
        "(some_compound[a,b,c])", "main",
    )
    expansion_line = make_proof_line(
        "(some_compound[a,b,c])", "main", "expansion",
        "(ortest1[a,b,c])", "main",
    )
    assert_failure(check_disintegration, disint_line,
                   [disint_line, expansion_line], state)


@register
def test_disintegration_category_unknown():
    """Category is not 'and', 'existence', or 'or' -> no branch returns True."""
    state = make_state_with_binaries(("Peano",))
    state.gl_binaries = dict(state.gl_binaries)
    state.gl_binaries["Peano"] = dict(state.gl_binaries["Peano"])
    state.gl_binaries["Peano"]["mystery_op"] = {
        "category": "fantasy",
        "signature": "(mystery_op[u_1])",
        "elements": ["(in[u_1,N])"],
    }
    state.current_gl_binary = state.gl_binaries["Peano"]
    disint_line = make_proof_line(
        "(in[a,N])", "main", "disintegration",
        "(some_compound[a])", "main",
    )
    expansion_line = make_proof_line(
        "(some_compound[a])", "main", "expansion",
        "(mystery_op[a])", "main",
    )
    assert_failure(check_disintegration, disint_line,
                   [disint_line, expansion_line], state)


# ===========================================================================
#  tag: expansion
# ===========================================================================

@register
def test_expansion_rest_empty():
    """rest empty -> ``len(rest) < 2`` short-circuits."""
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line("(&[a,b])", "main", "expansion")
    assert_failure(check_expansion, line, [line], state)


@register
def test_expansion_ns_mismatch():
    """line.namespace != right_ns -> reject."""
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(&[a,b])", "main", "expansion",
        "(some_compact[a,b])", "main_boundary_orint_X_((=[c,d]))",
    )
    assert_failure(check_expansion, line, [line], state)


@register
def test_expansion_right_not_in_chapter():
    """right_expr (rest[0]) must appear as a left-side expression somewhere
    in the chapter; otherwise reject."""
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(&[a,b])", "main", "expansion",
        "(some_compact[a,b])", "main",
    )
    # Chapter contains ONLY the expansion row; the compact never appears
    # as a left-side expression -> reject.
    assert_failure(check_expansion, line, [line], state)


@register
def test_expansion_core_not_in_binary_with_right_side_present():
    """right_expr appears as a left-side row, but its core is not in any
    GL binary -> no `_try_expand` path succeeds."""
    state = make_state_with_binaries(("Peano",))
    left_line = make_proof_line(
        "(&[a,b])", "main", "expansion",
        "(unknown_core[a,b])", "main",
    )
    support_line = make_proof_line(
        "(unknown_core[a,b])", "main", "task formulation",
    )
    chapter = [left_line, support_line]
    assert_failure(check_expansion, left_line, chapter, state)


@register
def test_expansion_or_intro_wrong_disjunct_rejected():
    """D-237 negative: an intro-shaped row whose premise
    is NOT one of the or premise's flattened disjuncts (here the equality
    uses the implication's set argument M instead of a disjunct value)
    must still reject — the intro acceptance admits exactly the compiled
    leaves, nothing wider."""
    state = make_state_with_binaries(("Peano",))
    state.gl_binaries = dict(state.gl_binaries)
    state.gl_binaries["Peano"] = dict(state.gl_binaries["Peano"])
    state.gl_binaries["Peano"]["or93"] = {
        "category": "or",
        "signature": "(or93[u_1,u_2,u_3])",
        "arity": 3,
        "elements": ["(=[u_1,u_2])", "(=[u_3,u_2])"],
    }
    state.gl_binaries["Peano"]["impltest_or2"] = {
        "category": "implication",
        "signature": "(impltest_or2[u_1,u_2,u_3])",
        "elements": ["(or93[u_1,1,u_2])", "(in[1,u_3])"],
    }
    state.current_gl_binary = state.gl_binaries["Peano"]
    compact_origin = make_proof_line(
        "(impltest_or2[a,b,M])", "main", "task formulation",
    )
    bad_intro = make_proof_line(
        "(>[w1](=[M,w1])(or93[a,w1,b]))", "main", "expansion",
        "(impltest_or2[a,b,M])", "main",
    )
    chapter = [compact_origin, bad_intro]
    assert_failure(check_expansion, bad_intro, chapter, state)


@register
def test_expansion_negated_existence_wrong_category():
    """right_expr is `!(name[args])` but the inner core's binary entry is
    NOT category 'existence' -> reject in the negated branch."""
    state = make_state_with_binaries(("Peano",))
    state.gl_binaries = dict(state.gl_binaries)
    state.gl_binaries["Peano"] = dict(state.gl_binaries["Peano"])
    state.gl_binaries["Peano"]["andthing"] = {
        "category": "and",
        "signature": "(andthing[u_1])",
        "elements": ["(in[u_1,N])"],
    }
    state.current_gl_binary = state.gl_binaries["Peano"]
    line = make_proof_line(
        "(>[](in[a,N])!(in[a,M]))", "main", "expansion",
        "!(andthing[a])", "main",
    )
    support = make_proof_line("!(andthing[a])", "main", "task formulation")
    assert_failure(check_expansion, line, [line, support], state)


@register
def test_expansion_negated_existence_element_count_not_two():
    """category 'existence' but its elements list has !=2 elements; the
    negated branch requires exactly 2."""
    state = make_state_with_binaries(("Peano",))
    state.gl_binaries = dict(state.gl_binaries)
    state.gl_binaries["Peano"] = dict(state.gl_binaries["Peano"])
    state.gl_binaries["Peano"]["existence_three"] = {
        "category": "existence",
        "signature": "(existence_three[u_1])",
        "elements": ["(a)", "(b)", "(c)"],   # 3 elements, not 2
    }
    state.current_gl_binary = state.gl_binaries["Peano"]
    line = make_proof_line(
        "(>[](a)!(b))", "main", "expansion",
        "!(existence_three[a])", "main",
    )
    support = make_proof_line("!(existence_three[a])",
                              "main", "task formulation")
    assert_failure(check_expansion, line, [line, support], state)


@register
def test_expansion_negated_existence_target_doesnt_match():
    """category 'existence', 2 elements, but the target expression doesn't
    normalize to either implication form -> reject."""
    state = make_state_with_binaries(("Peano",))
    state.gl_binaries = dict(state.gl_binaries)
    state.gl_binaries["Peano"] = dict(state.gl_binaries["Peano"])
    state.gl_binaries["Peano"]["existence_two"] = {
        "category": "existence",
        "signature": "(existence_two[u_1])",
        "elements": ["(left[u_1])", "(right[u_1])"],
    }
    state.current_gl_binary = state.gl_binaries["Peano"]
    line = make_proof_line(
        "(eq[a,b])", "main", "expansion",       # arbitrary expr, not impl
        "!(existence_two[a])", "main",
    )
    support = make_proof_line("!(existence_two[a])",
                              "main", "task formulation")
    assert_failure(check_expansion, line, [line, support], state)


@register
def test_expansion_signature_arity_mismatch():
    """Signature has arity 2; actual right_expr's compact has arity 3."""
    state = make_state_with_binaries(("Peano",))
    state.gl_binaries = dict(state.gl_binaries)
    state.gl_binaries["Peano"] = dict(state.gl_binaries["Peano"])
    state.gl_binaries["Peano"]["sigtest"] = {
        "category": "and",
        "signature": "(sigtest[u_1,u_2])",   # arity 2
        "elements": ["(in[u_1,N])", "(eq[u_1,u_2])"],
    }
    state.current_gl_binary = state.gl_binaries["Peano"]
    line = make_proof_line(
        "(&[a,b])", "main", "expansion",
        "(sigtest[a,b,c])", "main",          # arity 3 — mismatch
    )
    support = make_proof_line("(sigtest[a,b,c])", "main", "task formulation")
    assert_failure(check_expansion, line, [line, support], state)


@register
def test_expansion_target_reconstruction_doesnt_match():
    """Compact known, args fit, but the reconstructed compound doesn't
    normalize to the actual target."""
    state = make_state_with_binaries(("Peano",))
    state.gl_binaries = dict(state.gl_binaries)
    state.gl_binaries["Peano"] = dict(state.gl_binaries["Peano"])
    state.gl_binaries["Peano"]["andshape"] = {
        "category": "and",
        "signature": "(andshape[u_1,u_2])",
        "elements": ["(in[u_1,N])", "(eq[u_1,u_2])"],
    }
    state.current_gl_binary = state.gl_binaries["Peano"]
    # Reconstruction would yield (&(in[a,N])(eq[a,b])) modulo paren
    # convention; our target is the totally unrelated expression below.
    line = make_proof_line(
        "(or[x,y,z])", "main", "expansion",
        "(andshape[a,b])", "main",
    )
    support = make_proof_line("(andshape[a,b])", "main", "task formulation")
    assert_failure(check_expansion, line, [line, support], state)


@register
def test_expansion_negated_inner_missing_paren():
    """right_expr is `!something` but the inner isn't paren-wrapped, so
    _extract_core_name returns empty and binary lookup fails."""
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(&[a,b])", "main", "expansion",
        "!barebones", "main",   # not a real GL expression
    )
    support = make_proof_line("!barebones", "main", "task formulation")
    assert_failure(check_expansion, line, [line, support], state)


# ===========================================================================
#  tag: compilation
# ===========================================================================

@register
def test_compilation_rest_empty():
    """rest empty -> ``len(rest) < 2`` short-circuits to reject."""
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line("(implication0[a,b])", "main", "compilation")
    assert_failure(check_compilation, line, [line], state)


@register
def test_compilation_rest_one_field():
    """Only the original but no original_ns -> ``len(rest) < 2`` reject."""
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(implication0[a,b])", "main", "compilation",
        "(>[1](in[1,a])(in[1,b]))",
    )
    assert_failure(check_compilation, line, [line], state)


@register
def test_compilation_ns_mismatch():
    """line.namespace != original_ns -> compaction must stay in scope."""
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(implication0[a,b])", "main", "compilation",
        "(>[1](in[1,a])(in[1,b]))", "main_boundary_orint_X_((=[c,d]))",
    )
    assert_failure(check_compilation, line, [line], state)


@register
def test_compilation_core_not_in_binary():
    """compact's core is in no GL binary -> reject."""
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(nonexistent_impl_core[a,b])", "main", "compilation",
        "(>[1](in[1,a])(in[1,b]))", "main",
    )
    assert_failure(check_compilation, line, [line], state)


@register
def test_compilation_category_not_implication():
    """Binary entry exists but its category is not 'implication' -> reject
    (a `compilation` row's compact MUST be an implication-compiled name)."""
    state = make_state_with_binaries(("Peano",))
    state.gl_binaries = dict(state.gl_binaries)
    state.gl_binaries["Peano"] = dict(state.gl_binaries["Peano"])
    state.gl_binaries["Peano"]["andshaped1"] = {
        "category": "and",
        "signature": "(andshaped1[u_1,u_2])",
        "elements": ["(in[1,u_1])", "(in[1,u_2])"],
    }
    state.current_gl_binary = state.gl_binaries["Peano"]
    line = make_proof_line(
        "(andshaped1[a,b])", "main", "compilation",
        "(>[1](in[1,a])(in[1,b]))", "main",
    )
    assert_failure(check_compilation, line, [line], state)


@register
def test_compilation_does_not_reconstruct_original():
    """Implication-category entry, but rest[0] is not what the compact's
    elements reconstruct to -> reject."""
    state = make_state_with_binaries(("Peano",))
    state.gl_binaries = dict(state.gl_binaries)
    state.gl_binaries["Peano"] = dict(state.gl_binaries["Peano"])
    state.gl_binaries["Peano"]["impltest9"] = {
        "category": "implication",
        "signature": "(impltest9[u_1,u_2])",
        "elements": ["(in[1,u_1])", "(in[1,u_2])"],
    }
    state.current_gl_binary = state.gl_binaries["Peano"]
    line = make_proof_line(
        "(impltest9[a,b])", "main", "compilation",
        "(>[1](in[1,a])(eq[1,b]))", "main",   # head differs (eq vs in)
    )
    assert_failure(check_compilation, line, [line], state)


# ===========================================================================
#  tag: recursion
# ===========================================================================

@register
def test_recursion_namespace_not_main():
    """recursion requires namespace == 'main'."""
    state = make_state_with_binaries(("Peano",))
    set_chapter_context(state,
                        thm=("(>[v1](AnchorPeano[N,s,p,zero,one,two])"
                             "(in[v1,N]))", "induction", "v1"),
                        chapter_type="check_zero")
    line = make_proof_line(
        "(=[v1,i0])",
        "main_boundary_orint_X_((=[a,b]))",
        "recursion",
    )
    assert_failure(check_recursion, line, [line], state)


@register
def test_recursion_no_chapter_thm():
    """state.current_chapter_thm None -> reject."""
    state = make_state_with_binaries(("Peano",))
    state.current_chapter_thm = None
    state.current_chapter_type = "check_zero"
    line = make_proof_line("(=[v1,i0])", "main", "recursion")
    assert_failure(check_recursion, line, [line], state)


@register
def test_recursion_chapter_type_unknown():
    """chapter_type is neither 'check_zero' nor 'check_induction_condition'."""
    state = make_state_with_binaries(("Peano",))
    set_chapter_context(state,
                        thm=("(>[v1](AnchorPeano[N,s,p,zero,one,two])"
                             "(in[v1,N]))", "direct", "v1"),
                        chapter_type="direct")   # neither check_zero nor _induction_condition
    line = make_proof_line("(=[v1,i0])", "main", "recursion")
    assert_failure(check_recursion, line, [line], state)


@register
def test_recursion_check_zero_expr_wrong_prefix():
    """check_zero: expression doesn't start with '(=['."""
    state = make_state_with_binaries(("Peano",))
    set_chapter_context(state,
                        thm=("(>[v1](AnchorPeano[N,s,p,zero,one,two])"
                             "(in[v1,N]))", "induction", "v1"),
                        chapter_type="check_zero")
    line = make_proof_line("(in[v1,N])", "main", "recursion")  # not an =
    assert_failure(check_recursion, line, [line], state)


@register
def test_recursion_check_zero_arity_three():
    """check_zero: equality has arity 3 instead of 2."""
    state = make_state_with_binaries(("Peano",))
    set_chapter_context(state,
                        thm=("(>[v1](AnchorPeano[N,s,p,zero,one,two])"
                             "(in[v1,N]))", "induction", "v1"),
                        chapter_type="check_zero")
    line = make_proof_line("(=[v1,i0,extra])", "main", "recursion")
    assert_failure(check_recursion, line, [line], state)


@register
def test_recursion_check_zero_ind_var_mismatch():
    """check_zero: first arg of equality isn't the chapter's induction
    variable."""
    state = make_state_with_binaries(("Peano",))
    set_chapter_context(state,
                        thm=("(>[v1](AnchorPeano[N,s,p,zero,one,two])"
                             "(in[v1,N]))", "induction", "v1"),
                        chapter_type="check_zero")
    line = make_proof_line("(=[w1,i0])", "main", "recursion")
    assert_failure(check_recursion, line, [line], state)


@register
def test_recursion_check_zero_second_arg_not_i0():
    """check_zero: second arg isn't 'i0'."""
    state = make_state_with_binaries(("Peano",))
    set_chapter_context(state,
                        thm=("(>[v1](AnchorPeano[N,s,p,zero,one,two])"
                             "(in[v1,N]))", "induction", "v1"),
                        chapter_type="check_zero")
    line = make_proof_line("(=[v1,i1])", "main", "recursion")
    assert_failure(check_recursion, line, [line], state)


@register
def test_recursion_check_induction_condition_in2_arity_low():
    """check_induction_condition with (in2[...]): need ≥3 args, only 2 supplied."""
    state = make_state_with_binaries(("Peano",))
    set_chapter_context(state,
                        thm=("(>[v1](AnchorPeano[N,s,p,zero,one,two])"
                             "(in[v1,N]))", "induction", "v1"),
                        chapter_type="check_induction_condition")
    line = make_proof_line("(in2[x,v1])", "main", "recursion")
    assert_failure(check_recursion, line, [line], state)


# ===========================================================================
#  tag: anchor handling
# ===========================================================================

@register
def test_anchor_handling_namespace_not_main():
    """anchor handling requires namespace == 'main'."""
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(AnchorPeano[N,s,p,zero,one,two])",
        "main_boundary_orint_X_((=[a,b]))",
        "anchor handling",
        "(AnchorPeano[N,s,p,zero,one,two])", "main",
    )
    assert_failure(check_anchor_handling, line, [line], state)


@register
def test_anchor_handling_rest_empty():
    """rest empty -> ``len(rest) < 2`` short-circuits."""
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(AnchorPeano[N,s,p,zero,one,two])",
        "main", "anchor handling",
    )
    assert_failure(check_anchor_handling, line, [line], state)


@register
def test_anchor_handling_origin_ns_not_main():
    """origin_ns (rest[1]) must equal 'main'."""
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(AnchorPeano[N,s,p,zero,one,two])", "main",
        "anchor handling",
        "(AnchorPeano[N,s,p,zero,one,two])",
        "main_boundary_orint_X_((=[a,b]))",
    )
    assert_failure(check_anchor_handling, line, [line], state)


@register
def test_anchor_handling_target_core_mismatch():
    """target_core differs from origin_core."""
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(AnchorPeano[N,s,p,zero,one,two])", "main",
        "anchor handling",
        "(AnchorGauss[N,s,p,zero,one,two])", "main",
    )
    assert_failure(check_anchor_handling, line, [line], state)


@register
def test_anchor_handling_core_not_starting_with_anchor():
    """target_core does not start with 'Anchor' -> reject."""
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(NotAnchor[N,s,p,zero,one,two])", "main",
        "anchor handling",
        "(NotAnchor[N,s,p,zero,one,two])", "main",
    )
    assert_failure(check_anchor_handling, line, [line], state)


@register
def test_anchor_handling_args_arity_mismatch():
    """target_args length != origin_args length."""
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(AnchorPeano[N,s,p,zero,one,two])", "main",
        "anchor handling",
        "(AnchorPeano[N,s,p,zero,one])", "main",   # arity 5 vs 6
    )
    assert_failure(check_anchor_handling, line, [line], state)


@register
def test_anchor_handling_differing_position_not_in_defs():
    """An arg differs at position 1; state.definition_sets has no entry
    for the AnchorMystery core, so the (1) check finds no `(1)` and rejects."""
    state = make_state_with_binaries(("Peano",))
    # state.definition_sets has no AnchorMystery entry.
    line = make_proof_line(
        "(AnchorMystery[X,s,p,zero,one,two])", "main",
        "anchor handling",
        "(AnchorMystery[N,s,p,zero,one,two])", "main",   # position 1 differs
    )
    # No task formulation for origin needed because the (1) check rejects first.
    assert_failure(check_anchor_handling, line, [line], state)


@register
def test_anchor_handling_no_task_formulation_for_origin():
    """All structural checks pass (args identical), but the origin anchor
    is not present in chapter as a task formulation row."""
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(AnchorMystery[N,s,p,zero,one,two])", "main",
        "anchor handling",
        "(AnchorMystery[N,s,p,zero,one,two])", "main",
    )
    # Chapter contains ONLY the anchor handling row; no task formulation.
    assert_failure(check_anchor_handling, line, [line], state)


if __name__ == "__main__":
    sys.exit(run_all_tests())
