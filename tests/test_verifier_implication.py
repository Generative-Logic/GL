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

"""Failure tests for implication / theorem / task formulation checkers.

Each test injects exactly ONE subtle malformation and asserts the verifier
rejects the row (the checker returns False). See ``tests/test_harness.py``
for the registry, fixtures, and runner.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_harness import (  # noqa: E402
    register, make_state_minimal, make_state_with_binaries,
    make_state_with_globals, make_proof_line, set_chapter_context,
    assert_failure, run_all_tests,
)
from verifier import (  # noqa: E402
    check_implication, check_theorem_tag, check_task_formulation,
)


# ===========================================================================
#  tag: implication
# ===========================================================================

@register
def test_implication_rest_empty():
    """rest empty -> ``len(rest) < 2`` short-circuit."""
    state = make_state_minimal()
    line = make_proof_line("(in[a,N])", "main", "implication")
    assert_failure(check_implication, line, [line], state)


@register
def test_implication_rest_one_field():
    """rest has impl but no impl_ns -> ``len(rest) < 2`` short-circuit."""
    state = make_state_minimal()
    line = make_proof_line(
        "(in[a,N])", "main", "implication",
        "(>[v1](in[v1,N])(in[v1,N]))",
    )
    assert_failure(check_implication, line, [line], state)


@register
def test_implication_rest_odd_length_dangling_premise():
    """rest has impl + impl_ns + a premise expression but no premise_ns
    (odd length 3). The even-pair guard rejects up front; without it,
    the dangling premise would be silently dropped by the stride-2
    indexing in `range(2, len(rest), 2)` / `range(3, ..., 2)`."""
    state = make_state_minimal()
    line = make_proof_line(
        "(in[a,N])", "main", "implication",
        "(>[v1](in[v1,N])(in[v1,N]))", "main",
        "(in[a,N])",   # premise with no namespace cell
    )
    assert_failure(check_implication, line, [line], state)


@register
def test_implication_rest_odd_length_dangling_premise_ns():
    """Odd length 5: two premises supplied, but second premise has its
    expression without a namespace cell. Even-pair guard rejects."""
    state = make_state_minimal()
    line = make_proof_line(
        "(in[a,N])", "main", "implication",
        "(>[v1,v2](in[v1,N])(in[v2,N])(in[v1,N]))", "main",
        "(in[a,N])", "main",
        "(in[b,N])",   # second premise with no namespace
    )
    assert_failure(check_implication, line, [line], state)


@register
def test_implication_impl_ns_sibling_scope():
    """impl_ns is a sibling OR-branch (neither main, nor result, nor strict
    ancestor of result) -> D-35 comparable-scope rule rejects."""
    state = make_state_minimal()
    line = make_proof_line(
        "(in[a,N])", "main_boundary_orint_X_((=[a,b]))",
        "implication",
        "(>[v1](in[v1,N])(in[v1,N]))",
        "main_boundary_orint_Y_((=[c,d]))",  # sibling, not ancestor of result
        "(in[a,N])", "main_boundary_orint_X_((=[a,b]))",
    )
    assert_failure(check_implication, line, [line], state)


@register
def test_implication_premise_ns_descendant_of_result():
    """A premise_ns is a strict DESCENDANT of result_ns -> D-35 rejects
    (only main / result / strict ancestor are admissible)."""
    state = make_state_minimal()
    line = make_proof_line(
        "(in[a,N])", "main",
        "implication",
        "(>[v1](in[v1,N])(in[v1,N]))", "main",
        "(in[a,N])", "main_boundary_orint_X_((=[a,b]))",
    )
    assert_failure(check_implication, line, [line], state)


@register
def test_implication_premise_core_mismatch():
    """Premise core differs from reference rule's expected core; no
    permutation can match."""
    state = make_state_minimal()
    line = make_proof_line(
        "(in[(s[a]),N])", "main",
        "implication",
        "(>[v1](in[v1,N])(in[(s[v1]),N]))", "main",
        "(=[a,N])", "main",  # '=' instead of 'in'
    )
    assert_failure(check_implication, line, [line], state)


@register
def test_implication_premise_count_short():
    """Reference rule has two premises; only one supplied -> chain length
    mismatch."""
    state = make_state_minimal()
    line = make_proof_line(
        "(in[(s[a]),N])", "main",
        "implication",
        "(>[v1,v2](in[v1,N])(in[v2,N])(in[(s[v1]),N]))", "main",
        "(in[a,N])", "main",
    )
    assert_failure(check_implication, line, [line], state)


@register
def test_implication_premise_count_extra():
    """Reference rule has one premise; two supplied -> chain length
    mismatch."""
    state = make_state_minimal()
    line = make_proof_line(
        "(in[(s[a]),N])", "main",
        "implication",
        "(>[v1](in[v1,N])(in[(s[v1]),N]))", "main",
        "(in[a,N])", "main",
        "(in[b,N])", "main",
    )
    assert_failure(check_implication, line, [line], state)


@register
def test_implication_changeable_inconsistent_mapping():
    """Same changeable maps to two different actual values across premise
    slots; well-definedness violated."""
    state = make_state_minimal()
    line = make_proof_line(
        "(in[(s[a]),N])", "main",
        "implication",
        "(>[v1](in[v1,N])(in[v1,N])(in[(s[v1]),N]))", "main",
        "(in[a,N])", "main",
        "(in[b,N])", "main",   # v1 -> a here, v1 -> b there
    )
    assert_failure(check_implication, line, [line], state)


@register
def test_implication_unchangeable_mismatch():
    """Free (unchangeable) variable in rule body doesn't match the actual."""
    state = make_state_minimal()
    line = make_proof_line(
        "(in[(s[a]),N])", "main",
        "implication",
        "(>[v1](in[v1,M])(in[(s[v1]),M]))", "main",  # rule references M
        "(in[a,M])", "main",
    )
    assert_failure(check_implication, line, [line], state)


@register
def test_implication_premise_arity_mismatch():
    """Reference premise is arity 2; actual premise is arity 3."""
    state = make_state_minimal()
    line = make_proof_line(
        "(in[(s[a]),N])", "main",
        "implication",
        "(>[v1](in[v1,N])(in[(s[v1]),N]))", "main",
        "(in[a,N,extra])", "main",
    )
    assert_failure(check_implication, line, [line], state)


@register
def test_implication_head_arity_mismatch():
    """Result expression is arity 3; reference head is arity 2."""
    state = make_state_minimal()
    line = make_proof_line(
        "(in[(s[a]),N,extra])", "main",
        "implication",
        "(>[v1](in[v1,N])(in[(s[v1]),N]))", "main",
        "(in[a,N])", "main",
    )
    assert_failure(check_implication, line, [line], state)


@register
def test_implication_anchor_head_core_mismatch():
    """Anchor-level rule's head has core 'in'; actual result has core '='
    -> after normalization the head-position expressions still differ
    on core, no permutation closes the gap."""
    state = make_state_minimal()
    line = make_proof_line(
        "(=[a,b])", "main",
        "implication",
        "(>[v1](AnchorPeano[N,s,p,zero,one,two])(in[v1,N]))", "main",
        "(AnchorPeano[N,s,p,zero,one,two])", "main",
    )
    assert_failure(check_implication, line, [line], state)


@register
def test_implication_head_core_mismatch():
    """Reference head core '=' vs actual result head core 'in'."""
    state = make_state_minimal()
    line = make_proof_line(
        "(in[a,N])", "main",
        "implication",
        "(>[v1](in[v1,N])(=[v1,a]))", "main",
        "(in[a,N])", "main",
    )
    assert_failure(check_implication, line, [line], state)


@register
def test_implication_impl_ns_strict_ancestor_to_sibling():
    """impl_ns marks an ancestor scope but premise_ns is a different
    branch of that ancestor; D-35 rejects on the premise_ns."""
    state = make_state_minimal()
    line = make_proof_line(
        "(in[a,N])", "main_boundary_orint_X_((=[a,b]))",
        "implication",
        "(>[v1](in[v1,N])(in[v1,N]))", "main",
        "(in[a,N])", "main_boundary_orint_X_((=[c,d]))",
        # premise_ns is a sibling of result_ns, not main/result/ancestor
    )
    assert_failure(check_implication, line, [line], state)


@register
def test_implication_head_expression_completely_different():
    """Reference rule says `(in[(s[v1]),N])` but result is `(=[a,b])`."""
    state = make_state_minimal()
    line = make_proof_line(
        "(=[a,b])", "main",
        "implication",
        "(>[v1](in[v1,N])(in[(s[v1]),N]))", "main",
        "(in[a,N])", "main",
    )
    assert_failure(check_implication, line, [line], state)


@register
def test_implication_result_deeper_than_every_constituent():
    """Validity-stack deposit rule (mirrors `generateEncodedRequests`):
    every constituent (impl + premises) is at ``main`` but the result
    is at ``main_boundary_(...)``. D-35 alone admits this (constituents
    ARE ancestors of result_ns), but the deeperOf-accumulation in the
    prover would deposit the result at ``main``, never at a strictly
    deeper scope no constituent reaches. Reject."""
    state = make_state_minimal()
    line = make_proof_line(
        "(=[it_0_lev_0_32,2])",
        "main_boundary_(implication23[2,8,int_lev_4_2365])",
        "implication",
        "(>[1](in[1,u_1])(>[2](in2[2,1,u_3])(>[3](in2[3,1,u_3])(=[2,3]))))",
        "main",
        "(in2[2,6,3])", "main",
        "(in2[it_0_lev_0_32,6,3])", "main",
        "(in[6,1])", "main",
    )
    assert_failure(check_implication, line, [line], state)


@register
def test_implication_result_deeper_than_every_constituent_simple():
    """Minimal variant: rule + single premise at main, result at
    main_boundary_<X>. The result has no constituent depositing it
    there -> reject under the validity-stack deposit rule."""
    state = make_state_minimal()
    line = make_proof_line(
        "(in[a,N])",
        "main_boundary_orint_(or2[a,b,c,d])_((=[a,b]))",
        "implication",
        "(>[v1](in[v1,N])(in[v1,N]))", "main",
        "(in[a,N])", "main",
    )
    assert_failure(check_implication, line, [line], state)


@register
def test_implication_premise_ns_sibling_via_pair_comparable():
    """Two premise namespaces are siblings (neither is ancestor of the
    other), even though both are ancestors of result_ns. Strictly
    speaking D-35 alone (every constituent vs result) would admit this
    in isolation, but the pair-wise comparable check from
    `generateEncodedRequests` rejects it. Construct an artificial case:
    impl_ns is sibling of premise_ns, while both share a common
    descendant as result_ns.

    Note: in well-formed GL chapters this case is also caught by D-35
    (since at most one constituent can be a strict ancestor of result_ns
    while the other is a sibling — D-35 already rejects the sibling).
    But the pair-wise check documents the prover's `nm.comparable`
    contract explicitly."""
    state = make_state_minimal()
    line = make_proof_line(
        "(in[a,N])",
        "main_boundary_orint_X_((=[a,b]))",
        "implication",
        "(>[v1](in[v1,N])(in[v1,N]))",
        "main_boundary_orint_X_((=[a,b]))",   # this constituent == result_ns (OK)
        "(in[a,N])",
        "main_boundary_orint_Y_((=[c,d]))",   # sibling of impl_ns; D-35 rejects on this row
    )
    assert_failure(check_implication, line, [line], state)


# ===========================================================================
#  tag: theorem
# ===========================================================================

@register
def test_theorem_namespace_not_main():
    """theorem tag requires namespace == 'main'."""
    state = make_state_with_globals(
        [("(>[v1](in[v1,N])(in[v1,N]))", "direct", "ref")])
    line = make_proof_line(
        "(>[v1](in[v1,N])(in[v1,N]))",
        "main_boundary_orint_X_((=[a,b]))",
        "theorem",
    )
    assert_failure(check_theorem_tag, line, [line], state)


@register
def test_theorem_not_in_global_registry():
    """Expression not present in state.global_theorems and no alpha-canonical
    or w/V revert match -> reject."""
    state = make_state_with_globals(
        [("(>[v1](in[v1,N])(in[v1,N]))", "direct", "ref")])
    line = make_proof_line(
        "(>[v1](=[v1,v1])(=[v1,v1]))",  # different theorem
        "main",
        "theorem",
    )
    assert_failure(check_theorem_tag, line, [line], state)


@register
def test_theorem_empty_globals():
    """No theorems loaded -> any expression rejects."""
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(>[v1](in[v1,N])(in[v1,N]))", "main", "theorem",
    )
    assert_failure(check_theorem_tag, line, [line], state)


@register
def test_theorem_malformed_expression():
    """Malformed (no opening paren) -> not matched, not normalized to any
    registered theorem."""
    state = make_state_with_globals(
        [("(>[v1](in[v1,N])(in[v1,N]))", "direct", "ref")])
    line = make_proof_line("in[v1,N]", "main", "theorem")
    assert_failure(check_theorem_tag, line, [line], state)


@register
def test_theorem_alpha_canonical_arity_differs():
    """Even after alpha-canonicalization, a different arity won't match."""
    state = make_state_with_globals(
        [("(>[v1](in[v1,N])(in[(s[v1]),N]))", "direct", "ref")])
    line = make_proof_line(
        "(>[w1,w2](in[w1,w2,N])(in[(s[w1]),N]))", "main", "theorem")
    assert_failure(check_theorem_tag, line, [line], state)


@register
def test_theorem_namespace_descendant_scope():
    """theorem citation from a descendant scope is rejected (only 'main')."""
    state = make_state_with_globals(
        [("(>[v1](in[v1,N])(in[v1,N]))", "direct", "ref")])
    line = make_proof_line(
        "(>[v1](in[v1,N])(in[v1,N]))",
        "main_boundary_X_assumption",
        "theorem",
    )
    assert_failure(check_theorem_tag, line, [line], state)


@register
def test_theorem_rest_with_garbage_ignored_but_main_check_fails():
    """Even with garbage rest fields, the ns check is the dominant gate."""
    state = make_state_with_globals(
        [("(>[v1](in[v1,N])(in[v1,N]))", "direct", "ref")])
    line = make_proof_line(
        "(>[v1](in[v1,N])(in[v1,N]))",
        "not_main",
        "theorem",
        "ignored", "main",
    )
    assert_failure(check_theorem_tag, line, [line], state)


@register
def test_theorem_disjoint_globals_set():
    """state.global_theorems contains other theorems, but not the cited one."""
    state = make_state_with_globals([
        ("(>[v1](=[v1,v1])(=[v1,v1]))", "direct", "ref-1"),
        ("(>[v1,v2](in[v1,v2])(in[v1,v2]))", "direct", "ref-2"),
    ])
    line = make_proof_line(
        "(>[w1](mul[w1,zero])(=[w1,zero]))", "main", "theorem")
    assert_failure(check_theorem_tag, line, [line], state)


# ===========================================================================
#  tag: task formulation
# ===========================================================================

@register
def test_task_formulation_namespace_not_main():
    """task formulation requires namespace == 'main'."""
    state = make_state_with_binaries(("Peano",))
    set_chapter_context(state,
                        thm=("(>[v1](in[v1,N])(in[v1,N]))", "direct", "ref"))
    line = make_proof_line(
        "(in[v1,N])",
        "main_boundary_orint_X_((=[a,b]))",
        "task formulation",
    )
    assert_failure(check_task_formulation, line, [line], state)


@register
def test_task_formulation_no_chapter_thm():
    """state.current_chapter_thm is None -> reject (no premise context)."""
    state = make_state_with_binaries(("Peano",))
    # explicitly leave current_chapter_thm = None
    line = make_proof_line("(in[v1,N])", "main", "task formulation")
    assert_failure(check_task_formulation, line, [line], state)


@register
def test_task_formulation_expr_not_a_premise():
    """Expression doesn't appear in theorem's disintegrated premise list."""
    state = make_state_with_binaries(("Peano",))
    set_chapter_context(state,
                        thm=("(>[v1](in[v1,N])(in[(s[v1]),N]))",
                             "direct", "ref"))
    line = make_proof_line(
        "(=[zero,zero])",   # not a premise; also not the head
        "main", "task formulation",
    )
    assert_failure(check_task_formulation, line, [line], state)


@register
def test_task_formulation_expr_is_head_not_premise():
    """Direct-proof head; head is NOT in the premise set -> reject (head is
    only a valid task-formulation expr for contradiction-style proofs where
    head starts with '!')."""
    state = make_state_with_binaries(("Peano",))
    set_chapter_context(state,
                        thm=("(>[v1](in[v1,N])(in[(s[v1]),N]))",
                             "direct", "ref"))
    line = make_proof_line(
        "(in[(s[v1]),N])",   # the head, not a premise
        "main", "task formulation",
    )
    assert_failure(check_task_formulation, line, [line], state)


@register
def test_task_formulation_contradiction_wrong_clean_op():
    """Contradiction-style head is `!(X)`; a task-formulation row may carry
    `(X)` (the cleanOp). Here we present a DIFFERENT expression -> reject."""
    state = make_state_with_binaries(("Peano",))
    set_chapter_context(state,
                        thm=("(>[v1](in[v1,N])!(=[v1,zero]))",
                             "direct", "ref"))
    line = make_proof_line(
        "(=[v1,one])",   # not the head's cleanOp; not a premise either
        "main", "task formulation",
    )
    assert_failure(check_task_formulation, line, [line], state)


@register
def test_task_formulation_premise_namespace_descendant():
    """Premise textually matches but namespace is not 'main'."""
    state = make_state_with_binaries(("Peano",))
    set_chapter_context(state,
                        thm=("(>[v1](in[v1,N])(in[(s[v1]),N]))",
                             "direct", "ref"))
    line = make_proof_line(
        "(in[v1,N])",
        "main_boundary_orint_X_((=[a,b]))",
        "task formulation",
    )
    assert_failure(check_task_formulation, line, [line], state)


@register
def test_task_formulation_completely_unrelated_expression():
    """Random expression not even structurally close to any premise."""
    state = make_state_with_binaries(("Peano",))
    set_chapter_context(state,
                        thm=("(>[v1](in[v1,N])(in[v1,N]))", "direct", "ref"))
    line = make_proof_line(
        "(or2[a,b,c,d])",
        "main", "task formulation",
    )
    assert_failure(check_task_formulation, line, [line], state)


@register
def test_task_formulation_negated_premise_mismatched():
    """Premise list contains `(in[v1,N])`; task formulation row carries
    `!(in[v1,N])` (the negation). That isn't a premise — and with the
    head being a different expression, it isn't the reductio seed
    (the head's exact negation) either."""
    state = make_state_with_binaries(("Peano",))
    set_chapter_context(state,
                        thm=("(>[v1](in[v1,N])(in[(s[v1]),N]))",
                             "direct", "ref"))
    line = make_proof_line(
        "!(in[v1,N])",
        "main", "task formulation",
    )
    assert_failure(check_task_formulation, line, [line], state)


if __name__ == "__main__":
    # Standalone-runnable: only this file's @register side-effects have fired,
    # so run_all_tests() runs ONLY this group's tests.
    sys.exit(run_all_tests())
