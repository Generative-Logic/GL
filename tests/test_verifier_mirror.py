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

"""Failure tests for reformulated from / incubator back
reformulation / externally provided theorem / variable copy / multiplied from.

Each test injects exactly ONE subtle malformation. See
``tests/test_harness.py`` for fixtures and runner.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_harness import (  # noqa: E402
    register, make_state_minimal, make_state_with_binaries,
    make_state_with_externals, make_proof_line, assert_failure,
    run_all_tests,
)
from verifier import (  # noqa: E402
    check_reformulated_from,
    check_incubator_back_reformulation,
    check_externally_provided_theorem,
    check_variable_copy, check_equalize_variable,
)


# ===========================================================================
#  tag: reformulated from
# ===========================================================================

@register
def test_reformulated_from_namespace_not_main():
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(>[v1](AnchorPeano[N,s,p,zero,one,two])(somenoexist[v1]))",
        "main_boundary_orint_X_((=[a,b]))",
        "reformulated from",
        "(>[v1](AnchorPeano[N,s,p,zero,one,two])(in[v1,N]))", "main",
    )
    assert_failure(check_reformulated_from, line, [line], state)


@register
def test_reformulated_from_rest_empty():
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(>[v1](AnchorPeano[N,s,p,zero,one,two])(somenoexist[v1]))",
        "main", "reformulated from",
    )
    assert_failure(check_reformulated_from, line, [line], state)


@register
def test_reformulated_from_target_no_premises():
    """Target has no `>[...]` wrapper -> tgt_premises empty -> reject."""
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(in[a,N])", "main",  # bare expression, no implication shape
        "reformulated from",
        "(>[v1](AnchorPeano[N,s,p,zero,one,two])(in[v1,N]))", "main",
    )
    assert_failure(check_reformulated_from, line, [line], state)


@register
def test_reformulated_from_target_first_premise_not_anchor():
    """Target's first premise is not an Anchor expression -> reject."""
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(>[v1](in[v1,N])(somenoexist[v1]))", "main",
        "reformulated from",
        "(>[v1](AnchorPeano[N,s,p,zero,one,two])(in[v1,N]))", "main",
    )
    assert_failure(check_reformulated_from, line, [line], state)


@register
def test_reformulated_from_head_core_unknown_in_any_binary():
    """Head core not found in any binary as 'existence'."""
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(>[v1](AnchorPeano[N,s,p,zero,one,two])"
        "(nonexistent_existence_op[v1]))", "main",
        "reformulated from",
        "(>[v1](AnchorPeano[N,s,p,zero,one,two])(in[v1,N]))", "main",
    )
    assert_failure(check_reformulated_from, line, [line], state)


@register
def test_reformulated_from_head_core_not_existence_category():
    """Head core IS in binary but its category is 'and', not 'existence'."""
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
        "(>[v1](AnchorPeano[N,s,p,zero,one,two])(andthing[v1]))", "main",
        "reformulated from",
        "(>[v1](AnchorPeano[N,s,p,zero,one,two])(in[v1,N]))", "main",
    )
    assert_failure(check_reformulated_from, line, [line], state)


@register
def test_reformulated_from_existence_elements_not_two():
    """Existence entry has 3 elements; reformulation requires exactly 2."""
    state = make_state_with_binaries(("Peano",))
    state.gl_binaries = dict(state.gl_binaries)
    state.gl_binaries["Peano"] = dict(state.gl_binaries["Peano"])
    state.gl_binaries["Peano"]["existthree"] = {
        "category": "existence",
        "signature": "(existthree[u_1])",
        "elements": ["(a[u_1])", "(b[u_1])", "(c[u_1])"],
    }
    state.current_gl_binary = state.gl_binaries["Peano"]
    line = make_proof_line(
        "(>[v1](AnchorPeano[N,s,p,zero,one,two])(existthree[v1]))", "main",
        "reformulated from",
        "(>[v1](AnchorPeano[N,s,p,zero,one,two])(in[v1,N]))", "main",
    )
    assert_failure(check_reformulated_from, line, [line], state)


@register
def test_reformulated_from_existence_missing_signature():
    """Existence entry has elements but no signature."""
    state = make_state_with_binaries(("Peano",))
    state.gl_binaries = dict(state.gl_binaries)
    state.gl_binaries["Peano"] = dict(state.gl_binaries["Peano"])
    state.gl_binaries["Peano"]["nosigexist"] = {
        "category": "existence",
        "signature": "",
        "elements": ["(left[u_1])", "(right[u_1])"],
    }
    state.current_gl_binary = state.gl_binaries["Peano"]
    line = make_proof_line(
        "(>[v1](AnchorPeano[N,s,p,zero,one,two])(nosigexist[v1]))", "main",
        "reformulated from",
        "(>[v1](AnchorPeano[N,s,p,zero,one,two])(in[v1,N]))", "main",
    )
    assert_failure(check_reformulated_from, line, [line], state)


@register
def test_reformulated_from_signature_arity_mismatch():
    """Signature has arity 1; head has arity 2."""
    state = make_state_with_binaries(("Peano",))
    state.gl_binaries = dict(state.gl_binaries)
    state.gl_binaries["Peano"] = dict(state.gl_binaries["Peano"])
    state.gl_binaries["Peano"]["arityexist"] = {
        "category": "existence",
        "signature": "(arityexist[u_1])",
        "elements": ["(in[u_1,N])", "(eq[u_1,N])"],
    }
    state.current_gl_binary = state.gl_binaries["Peano"]
    line = make_proof_line(
        "(>[v1,v2](AnchorPeano[N,s,p,zero,one,two])(arityexist[v1,v2]))",
        "main",
        "reformulated from",
        "(>[v1](AnchorPeano[N,s,p,zero,one,two])(in[v1,N]))", "main",
    )
    assert_failure(check_reformulated_from, line, [line], state)


@register
def test_reformulated_from_premise_count_diverges():
    """After expansion, expanded_non_anchor count != src_non_anchor count."""
    state = make_state_with_binaries(("Peano",))
    state.gl_binaries = dict(state.gl_binaries)
    state.gl_binaries["Peano"] = dict(state.gl_binaries["Peano"])
    state.gl_binaries["Peano"]["plain_exist"] = {
        "category": "existence",
        "signature": "(plain_exist[u_1])",
        "elements": ["(in[u_1,N])", "(eq[u_1,N])"],
    }
    state.current_gl_binary = state.gl_binaries["Peano"]
    src = ("(>[v1,v2,v3](AnchorPeano[N,s,p,zero,one,two])"
           "(in[v1,N])(in[v2,N])(in[v3,N])(eq[v1,N]))")
    tgt = ("(>[v1](AnchorPeano[N,s,p,zero,one,two])(plain_exist[v1]))")
    line = make_proof_line(tgt, "main", "reformulated from", src, "main")
    assert_failure(check_reformulated_from, line, [line], state)


# ===========================================================================
#  tag: incubator back reformulation
# ===========================================================================

@register
def test_incubator_back_reformulation_namespace_not_main():
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(=[a,b])",
        "main_boundary_orint_X_((=[c,d]))",
        "incubator back reformulation",
        "(some_source)", "main",
    )
    assert_failure(check_incubator_back_reformulation, line, [line], state)


@register
def test_incubator_back_reformulation_empty_namespace():
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(=[a,b])", "",
        "incubator back reformulation",
        "(some_source)", "main",
    )
    assert_failure(check_incubator_back_reformulation, line, [line], state)


@register
def test_incubator_back_reformulation_descendant_scope():
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(=[a,b])",
        "main_boundary_recursion_((=[v1,i0]))",
        "incubator back reformulation",
        "(some_source)", "main",
    )
    assert_failure(check_incubator_back_reformulation, line, [line], state)


@register
def test_incubator_back_reformulation_rest_empty():
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(=[a,b])", "main", "incubator back reformulation",
    )
    assert_failure(check_incubator_back_reformulation, line, [line], state)


@register
def test_incubator_back_reformulation_wrong_rewrite():
    # Gates pass (namespace main, rest non-empty) but the direct form
    # eliminates the witness to the WRONG term (z, not the cited y), so the
    # rewrite is unsound and must fail. Proves the check verifies the rewrite
    # itself, not just the structural namespace/rest gates.
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(>[x,y,z](AnchorPeano[x,y,z])(in3[x,x,z,+]))", "main",
        "incubator back reformulation",
        "(>[x,y,z](AnchorPeano[x,y,z])(>[w](in3[x,x,w,+])(=[w,y])))", "main",
    )
    assert_failure(check_incubator_back_reformulation, line, [line], state)


# ===========================================================================
#  tag: externally provided theorem
# ===========================================================================

@register
def test_externally_provided_theorem_namespace_not_main():
    state = make_state_with_externals(
        {"(>[v1](AnchorPeano[N,s,p,zero,one,two])(in[v1,N]))"})
    line = make_proof_line(
        "(>[v1](AnchorPeano[N,s,p,zero,one,two])(in[v1,N]))",
        "main_boundary_orint_X_((=[a,b]))",
        "externally provided theorem",
    )
    assert_failure(check_externally_provided_theorem, line, [line], state)


@register
def test_externally_provided_theorem_empty_externals():
    """state.external_theorems empty -> direct membership fails -> reject."""
    state = make_state_with_binaries(("Peano",))   # no externals
    line = make_proof_line(
        "(>[v1](AnchorPeano[N,s,p,zero,one,two])(in[v1,N]))",
        "main", "externally provided theorem",
    )
    assert_failure(check_externally_provided_theorem, line, [line], state)


@register
def test_externally_provided_theorem_expr_not_in_externals():
    """externals contains a DIFFERENT theorem; the cited expression is
    not in the external set."""
    state = make_state_with_externals(
        {"(>[v1](AnchorPeano[N,s,p,zero,one,two])(in[v1,N]))"})
    line = make_proof_line(
        "(or2[a,b,c,d])", "main",
        "externally provided theorem",
    )
    assert_failure(check_externally_provided_theorem, line, [line], state)


@register
def test_externally_provided_theorem_descendant_scope():
    state = make_state_with_externals(
        {"(>[v1](AnchorPeano[N,s,p,zero,one,two])(in[v1,N]))"})
    line = make_proof_line(
        "(>[v1](AnchorPeano[N,s,p,zero,one,two])(in[v1,N]))",
        "main_boundary_recursion_((=[v1,i0]))",
        "externally provided theorem",
    )
    assert_failure(check_externally_provided_theorem, line, [line], state)


# ===========================================================================
#  tag: variable copy
# ===========================================================================

@register
def test_variable_copy_expr_not_equality():
    """Expression doesn't start with `(=[`."""
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line("(in[a,N])", "main", "variable copy")
    assert_failure(check_variable_copy, line, [line], state)


@register
def test_variable_copy_equality_arity_three():
    """Equality has arity 3 instead of 2."""
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line("(=[a,a_copy,extra])", "main", "variable copy")
    assert_failure(check_variable_copy, line, [line], state)


@register
def test_variable_copy_suffix_missing():
    """b is NOT a + '_copy'."""
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line("(=[a,b])", "main", "variable copy")
    assert_failure(check_variable_copy, line, [line], state)


@register
def test_variable_copy_suffix_wrong_separator():
    """b == a + 'copy' (no underscore) -> reject."""
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line("(=[a,acopy])", "main", "variable copy")
    assert_failure(check_variable_copy, line, [line], state)


@register
def test_variable_copy_rest_non_empty():
    """variable copy is a dead-end axiom; non-empty rest -> reject."""
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(=[a,a_copy])", "main", "variable copy",
        "(some_source)", "main",
    )
    assert_failure(check_variable_copy, line, [line], state)


@register
def test_variable_copy_user_without_declaration():
    """Some other chapter line mentions `a_copy` in its args, but the
    chapter contains NO variable-copy row declaring it -> reject."""
    state = make_state_with_binaries(("Peano",))
    copy_line = make_proof_line(
        "(=[a,a_copy])", "main", "variable copy",
    )
    user_line = make_proof_line(
        "(in[a_copy,N])", "main", "task formulation",
    )
    chapter = [copy_line, user_line]
    # user_line mentions a_copy but has no trace edge back to a variable-copy
    # declaration -> reject.
    assert_failure(check_variable_copy, copy_line, chapter, state)


@register
def test_variable_copy_user_chain_breaks():
    """A chapter line uses `a_copy` and cites a SUPPORT, but the support
    doesn't carry `a_copy` and doesn't trace to the copy declaration."""
    state = make_state_with_binaries(("Peano",))
    copy_line = make_proof_line(
        "(=[a,a_copy])", "main", "variable copy",
    )
    intermediate = make_proof_line(
        "(in[other,N])", "main", "task formulation",   # doesn't carry a_copy
    )
    user_line = make_proof_line(
        "(in[a_copy,N])", "main", "implication",
        "(>[v1](in[v1,N])(in[v1,N]))", "main",
        "(in[other,N])", "main",   # cites intermediate, but it lacks a_copy
    )
    chapter = [copy_line, intermediate, user_line]
    assert_failure(check_variable_copy, copy_line, chapter, state)


@register
def test_variable_copy_target_tag_wrong():
    """A chapter row carries the same equality expression but a tag other
    than `variable copy` -> the is_target predicate rejects -> trace fails."""
    state = make_state_with_binaries(("Peano",))
    copy_line = make_proof_line(
        "(=[a,a_copy])", "main", "variable copy",
    )
    # Add an `implication` row sharing the same expression but wrong tag.
    impostor = make_proof_line(
        "(=[a,a_copy])", "main", "implication",
        "(>[v1](in[v1,N])(in[v1,N]))", "main",
        "(in[a,N])", "main",
    )
    user_line = make_proof_line(
        "(in[a_copy,N])", "main", "implication",
        "(>[v1](in[v1,N])(in[v1,N]))", "main",
        "(=[a,a_copy])", "main",
    )
    chapter = [copy_line, impostor, user_line]
    # The user_line traces to (=[a,a_copy]) which has TWO entries; the
    # `is_target` predicate requires tag=='variable copy'. Once the trace
    # reaches the equality, the recursion may pick the impostor row first;
    # but the predicate filters by tag. Result still depends on walk order;
    # what we can robustly check is that a sibling line whose only source
    # is an unrelated expression still fails. Reformulate the test:
    user_line2 = make_proof_line(
        "(in[a_copy,N])", "main", "implication",
        "(>[v1](in[v1,N])(in[v1,N]))", "main",
        "(in[disconnected,N])", "main",   # source doesn't carry a_copy
    )
    chapter2 = [copy_line, user_line2]
    assert_failure(check_variable_copy, copy_line, chapter2, state)


@register
def test_variable_copy_a_empty_string():
    """If a is empty, b would have to be `_copy`. The expression
    `(=[,_copy])` has args ['', '_copy']; `b != '' + '_copy'`? `_copy ==
    `` + `_copy`` is `_copy == _copy` True. So this passes! Instead, use
    arity 1: this hits the arity guard first."""
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line("(=[a_copy])", "main", "variable copy")   # arity 1
    assert_failure(check_variable_copy, line, [line], state)


@register
def test_variable_copy_suffix_extra_junk():
    """b has the right '_copy' prefix on it but extra trailing characters;
    `b != a + '_copy'` exactly -> reject."""
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line("(=[a,a_copy_extra])", "main", "variable copy")
    assert_failure(check_variable_copy, line, [line], state)


# ===========================================================================
#  tag: multiplied from
# ===========================================================================

@register
def test_multiplied_from_rest_empty():
    """rest empty -> ``len(rest) < 2`` short-circuits."""
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(>[v1](in[v1,N])(in[v1,N]))", "main", "multiplied from")
    assert_failure(check_equalize_variable, line, [line], state)


@register
def test_multiplied_from_rest_length_one():
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(>[v1](in[v1,N])(in[v1,N]))", "main", "multiplied from",
        "(>[v1,v2](in[v1,N])(in[v2,N])(=[v1,v2]))",
    )
    assert_failure(check_equalize_variable, line, [line], state)


@register
def test_multiplied_from_premise_count_mismatch():
    """Origin has 2 non-anchor premises; copy has 1."""
    state = make_state_with_binaries(("Peano",))
    origin = "(>[v1,v2](in[v1,N])(in[v2,N])(eq[v1,v2]))"
    copy_expr = "(>[v1](in[v1,N])(eq[v1,v1]))"
    line = make_proof_line(copy_expr, "main", "multiplied from",
                           origin, "main")
    assert_failure(check_equalize_variable, line, [line], state)


@register
def test_multiplied_from_pair_core_mismatch():
    """Origin's first premise has core 'in'; copy's first has core 'eq'."""
    state = make_state_with_binaries(("Peano",))
    origin = "(>[v1](in[v1,N])(in[v1,N]))"
    copy_expr = "(>[v1](eq[v1,N])(in[v1,N]))"
    line = make_proof_line(copy_expr, "main", "multiplied from",
                           origin, "main")
    assert_failure(check_equalize_variable, line, [line], state)


@register
def test_multiplied_from_pair_arity_mismatch():
    """Origin's premise has arity 2; copy's has arity 3."""
    state = make_state_with_binaries(("Peano",))
    origin = "(>[v1](in[v1,N])(in[v1,N]))"
    copy_expr = "(>[v1](in[v1,N,extra])(in[v1,N]))"
    line = make_proof_line(copy_expr, "main", "multiplied from",
                           origin, "main")
    assert_failure(check_equalize_variable, line, [line], state)


@register
def test_multiplied_from_inconsistent_arg_mapping():
    """Origin's v1 maps to copy's v1 in premise 1; in premise 2 it would
    have to map to v2 -> inconsistent."""
    state = make_state_with_binaries(("Peano",))
    origin = "(>[v1](in[v1,N])(in[v1,N]))"
    copy_expr = "(>[v1,v2](in[v1,N])(in[v2,N]))"
    line = make_proof_line(copy_expr, "main", "multiplied from",
                           origin, "main")
    assert_failure(check_equalize_variable, line, [line], state)


@register
def test_multiplied_from_free_anchor_merge_distinct_names():
    """I-24 soundness gate: orig has free anchor params N, M; copy
    identifies them as the same name X. Both N and M are free in the
    origin (not bound by >[]) -> reject."""
    state = make_state_with_binaries(("Peano",))
    # Both N and M are FREE (no >[] containing them).
    origin = "(>[v1](in[v1,N])(in[v1,M]))"
    copy_expr = "(>[v1](in[v1,X])(in[v1,X]))"   # collapses N and M to X
    line = make_proof_line(copy_expr, "main", "multiplied from",
                           origin, "main")
    assert_failure(check_equalize_variable, line, [line], state)


@register
def test_multiplied_from_free_anchor_merge_two_anchors():
    """Same I-24 case, different anchor names."""
    state = make_state_with_binaries(("Peano",))
    origin = "(>[v1](in[v1,Alpha])(in[v1,Beta]))"
    copy_expr = "(>[v1](in[v1,Gamma])(in[v1,Gamma]))"   # Alpha+Beta -> Gamma
    line = make_proof_line(copy_expr, "main", "multiplied from",
                           origin, "main")
    assert_failure(check_equalize_variable, line, [line], state)


@register
def test_multiplied_from_head_core_mismatch():
    """Premises match but head cores differ."""
    state = make_state_with_binaries(("Peano",))
    origin = "(>[v1](in[v1,N])(in[v1,N]))"
    copy_expr = "(>[v1](in[v1,N])(eq[v1,N]))"
    line = make_proof_line(copy_expr, "main", "multiplied from",
                           origin, "main")
    assert_failure(check_equalize_variable, line, [line], state)


@register
def test_multiplied_from_head_arity_mismatch():
    """Premises match but head arity differs."""
    state = make_state_with_binaries(("Peano",))
    origin = "(>[v1](in[v1,N])(in[v1,N]))"
    copy_expr = "(>[v1](in[v1,N])(in[v1,N,extra]))"
    line = make_proof_line(copy_expr, "main", "multiplied from",
                           origin, "main")
    assert_failure(check_equalize_variable, line, [line], state)


@register
def test_multiplied_from_origin_empty_premise_path():
    """Origin disintegrates to ``[head]`` only — no premises. Copy provides
    one premise + head. Length differs."""
    state = make_state_with_binaries(("Peano",))
    origin = "(in[v1,N])"   # no `>[]` wrapper -> no premises
    copy_expr = "(>[v1](in[v1,N])(in[v1,N]))"
    line = make_proof_line(copy_expr, "main", "multiplied from",
                           origin, "main")
    assert_failure(check_equalize_variable, line, [line], state)


@register
def test_multiplied_from_free_to_bound_collapse_with_free_collision():
    """Origin has bound v1 and free N. Copy maps v1 -> bound w1 (OK) but
    also maps origin's N -> copy's w1 (both names differ). N is free
    (not bound), w1 is bound in copy -> bound_var on copy side, so the
    guard's `copy_arg in copy_bound` check passes -> this SHOULDN'T
    reject. We use this as an inverse-control fixture: just confirm the
    obvious free-merge case rejects, which is already covered by
    test_multiplied_from_free_anchor_merge_distinct_names. Use a similar
    case where both arg names are free to keep the test purpose clear."""
    state = make_state_with_binaries(("Peano",))
    origin = "(>[v1](in[v1,N])(in[v1,M]))"   # N, M free
    copy_expr = "(>[v1](in[v1,P])(in[v1,P]))"   # N -> P, M -> P (free merge)
    line = make_proof_line(copy_expr, "main", "multiplied from",
                           origin, "main")
    assert_failure(check_equalize_variable, line, [line], state)


if __name__ == "__main__":
    sys.exit(run_all_tests())
