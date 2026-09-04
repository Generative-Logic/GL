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

"""Positive sanity tests — well-formed inputs that MUST pass each checker.

Rationale: if the fixture builder is broken (e.g. ``make_state_with_binaries``
loads no binaries, ``make_proof_line`` mangles the rest list), every failure
test in the other ``test_verifier_<group>.py`` files would trivially "pass"
without the verifier doing anything. These positive tests prove the rig is
intact — they exercise checkers on inputs designed to be accepted.

See ``tests/test_harness.py`` for fixtures and runner.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_harness import (  # noqa: E402
    register, make_state_with_binaries, make_state_with_globals,
    make_state_with_externals, make_proof_line, set_chapter_context,
    assert_pass, run_all_tests,
)
from verifier import (  # noqa: E402
    check_implication, check_expansion, check_disintegration,
    check_compilation,
    check_task_formulation, check_equality1, check_equality2,
    check_symmetry_of_equality, check_symmetry_of_inequality,
    check_recursion, check_theorem_tag,
    check_anchor_handling,
    check_variable_copy, check_externally_provided_theorem,
    check_incubator_back_reformulation, check_equalize_variable,
    check_contradiction, check_or_disintegration, check_or_convergence,
    check_or_branch_proven, check_or_branch_assumption,
    check_vacuous_truth, check_or_theorem,
    check_expansion_for_integration,
    check_reformulation_for_integration_and,
    check_premise_element, check_validity_name,
)


# ===========================================================================
#  Trivial-stub positives
# ===========================================================================

@register
def test_pos_or_theorem_minimal():
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(or2[a,b,c,d])", "main", "or theorem",
        "(some_existence)", "main",
    )
    assert_pass(check_or_theorem, line, [line], state)


@register
def test_pos_incubator_back_reformulation_minimal():
    state = make_state_with_binaries(("Peano",))
    # Direct form (in3[x,x,y,+]) is the witness-elimination of the existence
    # source (>[w](in3[x,x,w,+])(=[w,y])): substitute the witness w := y.
    line = make_proof_line(
        "(>[x,y](AnchorPeano[x,y])(in3[x,x,y,+]))", "main",
        "incubator back reformulation",
        "(>[x,y](AnchorPeano[x,y])(>[w](in3[x,x,w,+])(=[w,y])))", "main",
    )
    assert_pass(check_incubator_back_reformulation, line, [line], state)


# ===========================================================================
#  Equality family
# ===========================================================================

@register
def test_pos_equality1_identity_no_diff():
    """source == result, so no differing arg position; loop never fires;
    returns True."""
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(in[a,N])", "main", "equality1",
        "(in[a,N])", "main",
        "(=[a,a])", "main",
    )
    assert_pass(check_equality1, line, [line], state)


@register
def test_pos_equality1_single_swap():
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(in[b,N])", "main", "equality1",
        "(in[a,N])", "main",
        "(=[a,b])", "main",
    )
    assert_pass(check_equality1, line, [line], state)


@register
def test_pos_equality2_transitive():
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(=[a,c])", "main", "equality2",
        "(=[a,b])", "main",
        "(=[b,c])", "main",
    )
    assert_pass(check_equality2, line, [line], state)


@register
def test_pos_symmetry_of_equality():
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(=[a,b])", "main", "symmetry of equality",
        "(=[b,a])", "main",
    )
    assert_pass(check_symmetry_of_equality, line, [line], state)


@register
def test_pos_symmetry_of_inequality():
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "!(=[a,b])", "main", "symmetry of inequality",
        "!(=[b,a])", "main",
    )
    assert_pass(check_symmetry_of_inequality, line, [line], state)


# ===========================================================================
#  Task formulation + theorem + externally provided
# ===========================================================================

@register
def test_pos_task_formulation_premise_of_theorem():
    state = make_state_with_binaries(("Peano",))
    set_chapter_context(state,
                        thm=("(>[v1](in[v1,N])(in[(s[v1]),N]))",
                             "direct", "ref"))
    line = make_proof_line("(in[v1,N])", "main", "task formulation")
    assert_pass(check_task_formulation, line, [line], state)


@register
def test_pos_theorem_in_global_registry():
    state = make_state_with_globals(
        [("(>[v1](in[v1,N])(in[v1,N]))", "direct", "ref")])
    line = make_proof_line(
        "(>[v1](in[v1,N])(in[v1,N]))", "main", "theorem")
    assert_pass(check_theorem_tag, line, [line], state)


@register
def test_pos_externally_provided_theorem_direct_membership():
    state = make_state_with_externals(
        {"(>[v1](AnchorPeano[N,s,p,zero,one,two])(in[v1,N]))"})
    line = make_proof_line(
        "(>[v1](AnchorPeano[N,s,p,zero,one,two])(in[v1,N]))",
        "main", "externally provided theorem",
    )
    assert_pass(check_externally_provided_theorem, line, [line], state)


# ===========================================================================
#  Recursion (check_zero)
# ===========================================================================

@register
def test_pos_recursion_check_zero():
    state = make_state_with_binaries(("Peano",))
    set_chapter_context(state,
                        thm=("(>[v1](AnchorPeano[N,s,p,zero,one,two])"
                             "(in[v1,N]))", "induction", "v1"),
                        chapter_type="check_zero")
    line = make_proof_line("(=[v1,i0])", "main", "recursion")
    assert_pass(check_recursion, line, [line], state)


# ===========================================================================
#  Variable copy / multiplied from
# ===========================================================================

@register
def test_pos_variable_copy_isolated():
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line("(=[a,a_copy])", "main", "variable copy")
    # No other chapter rows mention a_copy -> no trace needed.
    assert_pass(check_variable_copy, line, [line], state)


@register
def test_pos_multiplied_from_identity_mapping():
    state = make_state_with_binaries(("Peano",))
    impl = "(>[v1](in[v1,N])(in[v1,N]))"
    line = make_proof_line(
        impl, "main", "multiplied from",
        impl, "main",
    )
    assert_pass(check_equalize_variable, line, [line], state)


# ===========================================================================
#  Contradiction + vacuous truth (shape only)
# ===========================================================================

@register
def test_pos_contradiction_shape():
    state = make_state_with_binaries(("Peano",))
    contr = make_proof_line(
        "!(in[a,N])", "main", "contradiction",
        "(in[a,N])", "main",
        "!(in[a,N])", "main",
        "(in[a,N])", "main",
    )
    seed = make_proof_line("(in[a,N])", "main", "task formulation")
    chapter = [contr, seed]
    assert_pass(check_contradiction, contr, chapter, state)


@register
def test_pos_contradiction_complement_polarity():
    """Complement (reductio) LB: positive row expression, negated seed —
    clean_op == '!' + line.expression (try_contradiction_negated_head)."""
    state = make_state_with_binaries(("Peano",))
    contr = make_proof_line(
        "(in[a,N])", "main", "contradiction",
        "(in[a,N])", "main",
        "!(in[a,N])", "main",
        "!(in[a,N])", "main",
    )
    seed = make_proof_line("!(in[a,N])", "main", "task formulation")
    chapter = [contr, seed]
    assert_pass(check_contradiction, contr, chapter, state)


@register
def test_pos_task_formulation_reductio_seed():
    """Positive-headed theorem proved by reductio: the head's negation is
    a valid task-formulation row (the complement LB's assumed hypothesis)."""
    state = make_state_with_binaries(("Peano",))
    set_chapter_context(state,
                        thm=("(>[v1](in[v1,N])(in[(s[v1]),N]))",
                             "direct", "ref"))
    line = make_proof_line("!(in[(s[v1]),N])", "main", "task formulation")
    assert_pass(check_task_formulation, line, [line], state)


@register
def test_pos_vacuous_truth_shape():
    state = make_state_with_binaries(("Peano",))
    line = make_proof_line(
        "(eq[v1,v1])", "main", "vacuous truth",
        "(in[v1,N])", "main",
        "!(in[v1,N])", "main",
        "(some_lb_key)", "main",
    )
    assert_pass(check_vacuous_truth, line, [line], state)


# ===========================================================================
#  Anchor handling (no differing args; origin present)
# ===========================================================================

@register
def test_pos_anchor_handling_identical_args():
    state = make_state_with_binaries(("Peano",))
    target = make_proof_line(
        "(AnchorPeano[N,s,p,zero,one,two])", "main", "anchor handling",
        "(AnchorPeano[N,s,p,zero,one,two])", "main",
    )
    origin = make_proof_line(
        "(AnchorPeano[N,s,p,zero,one,two])", "main", "task formulation",
    )
    chapter = [target, origin]
    assert_pass(check_anchor_handling, target, chapter, state)


# ===========================================================================
#  Implication (anchor-level identity)
# ===========================================================================

@register
def test_pos_implication_anchor_identity():
    """Anchor rule with self-identity head: premise + head normalize equally
    to the supplied actual chain."""
    state = make_state_with_binaries(("Peano",))
    impl = "(>[v1](AnchorPeano[N,s,p,zero,one,two])(in[v1,N]))"
    line = make_proof_line(
        "(in[a,N])", "main", "implication",
        impl, "main",
        "(AnchorPeano[N,s,p,zero,one,two])", "main",
    )
    assert_pass(check_implication, line, [line], state)


@register
def test_pos_implication_non_anchor_simple_subst():
    """Non-anchor rule `(>[v1](in[v1,N])(in[v1,N]))`; actual premise (in[a,N])
    yields head (in[a,N]) — consistent substitution v1->a."""
    state = make_state_with_binaries(("Peano",))
    impl = "(>[v1](in[v1,N])(in[v1,N]))"
    line = make_proof_line(
        "(in[a,N])", "main", "implication",
        impl, "main",
        "(in[a,N])", "main",
    )
    assert_pass(check_implication, line, [line], state)


# ===========================================================================
#  OR family (well-formed)
# ===========================================================================

def _state_with_or3() -> object:
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
    """Install ``or91`` whose first child is the compiled ``or90``."""
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


@register
def test_pos_or_disintegration_well_formed():
    state = _state_with_or3()
    line = make_proof_line(
        "(=[a,X])",
        "main_boundary_ordis_(or3[a,b,c])_((=[a,X]))",
        "or disintegration",
        "(or3[a,b,c])", "main",
    )
    origin = make_proof_line("(or3[a,b,c])", "main", "task formulation")
    chapter = [line, origin]
    assert_pass(check_or_disintegration, line, chapter, state)


@register
def test_pos_nested_or_disintegration_is_one_flat_cohort():
    state = _state_with_nested_or()
    line = make_proof_line(
        "(=[b,X])",
        "main_boundary_ordis_(or91[a,b,c])_((=[b,X]))",
        "or disintegration",
        "(or91[a,b,c])", "main",
    )
    origin = make_proof_line("(or91[a,b,c])", "main", "task formulation")
    assert_pass(check_or_disintegration, line, [line, origin], state)


@register
def test_pos_nested_or_convergence_requires_all_flat_leaves():
    state = _state_with_nested_or()
    branches = [
        "main_boundary_ordis_(or91[a,b,c])_((=[a,X]))",
        "main_boundary_ordis_(or91[a,b,c])_((=[b,X]))",
        "main_boundary_ordis_(or91[a,b,c])_((=[c,X]))",
    ]
    convergence = make_proof_line(
        "(conclusion)", "main", "or convergence",
        "(or91[a,b,c])", "main",
        "(conclusion)", branches[0],
        "(conclusion)", branches[1],
        "(conclusion)", branches[2],
    )
    derivations = [
        make_proof_line("(conclusion)", branch, "task formulation")
        for branch in branches
    ]
    assert_pass(check_or_convergence, convergence,
                [convergence] + derivations, state)


@register
def test_pos_nested_or_expansion_and_mutual_implication_use_flat_leaves():
    state = _state_with_nested_or()
    # Flat-leaf De Morgan form with the or-so-far nested NEGATED (its
    # !(&...) cancels to the bare (&...) conjunct) — the D-260 polarity
    # fix; the former fixture bytes !(&!(&...)...) encoded the defect.
    expanded = "!(&(&!(=[a,X])!(=[b,X]))!(=[c,X]))"
    origin = make_proof_line("(or91[a,b,c])", "main", "task formulation")
    expansion = make_proof_line(
        expanded, "main", "expansion", "(or91[a,b,c])", "main")
    implication = make_proof_line(
        "(>[X]!(=[b,X])(>[]!(=[c,X])(=[a,X])))", "main", "disintegration",
        expanded, "main")
    chapter = [origin, expansion, implication]
    assert_pass(check_expansion, expansion, chapter, state)
    assert_pass(check_disintegration, implication, chapter, state)


@register
def test_pos_or_branch_proven_well_formed():
    state = _state_with_or3()
    line = make_proof_line(
        "(or3[a,b,c])", "main", "or branch proven",
        "(=[a,X])",
        "main_boundary_orint_(or3[a,b,c])_((=[a,X]))",
    )
    assert_pass(check_or_branch_proven, line, [line], state)


@register
def test_pos_or_branch_assumption_well_formed():
    state = _state_with_or3()
    proven = make_proof_line(
        "(or3[a,b,c])", "main", "or branch proven",
        "(=[a,X])",
        "main_boundary_orint_(or3[a,b,c])_((=[a,X]))",
    )
    assumption = make_proof_line(
        "!(=[b,X])",
        "main_boundary_orint_(or3[a,b,c])_((=[a,X]))",
        "or branch assumption",
        "(or3[a,b,c])_integration_goal", "main",
    )
    chapter = [proven, assumption]
    assert_pass(check_or_branch_assumption, assumption, chapter, state)


# ===========================================================================
#  Expansion + Disintegration (require a custom binary entry)
# ===========================================================================

def _state_with_and_test() -> object:
    state = make_state_with_binaries(("Peano",))
    state.gl_binaries = dict(state.gl_binaries)
    state.gl_binaries["Peano"] = dict(state.gl_binaries["Peano"])
    state.gl_binaries["Peano"]["andsimple"] = {
        "category": "and",
        "signature": "(andsimple[u_1])",
        "elements": ["(in[u_1,N])"],
    }
    state.current_gl_binary = state.gl_binaries["Peano"]
    return state


@register
def test_pos_disintegration_and_well_formed():
    state = _state_with_and_test()
    disint = make_proof_line(
        "(in[a,N])", "main", "disintegration",
        "(some_compound)", "main",
    )
    expansion = make_proof_line(
        "(some_compound)", "main", "expansion",
        "(andsimple[a])", "main",
    )
    chapter = [disint, expansion]
    assert_pass(check_disintegration, disint, chapter, state)


@register
def test_pos_expansion_and_well_formed():
    """A single-element AND compact expands to the bare element. Need origin
    presence for the compact + normalize-with-unchangeables match between the
    expanded form and the binary's reconstructed form."""
    state = _state_with_and_test()
    # Single-element AND collapses: _build_and_from_elements(["(in[a,N])"])
    # returns "(in[a,N])" unchanged. So compound = "(in[a,N])".
    compact_origin = make_proof_line(
        "(andsimple[a])", "main", "task formulation",
    )
    expansion = make_proof_line(
        "(in[a,N])", "main", "expansion",
        "(andsimple[a])", "main",
    )
    chapter = [compact_origin, expansion]
    assert_pass(check_expansion, expansion, chapter, state)


@register
def test_pos_expansion_or_intro_implication_premise():
    """D-237: an `expansion` row whose left side is a
    per-leaf OR-INTRO rule `(>[bv](D_k)(orPremise))` of an implication
    compact with an or-shaped premise element is accepted.

    Synthetic impltest_or := premise (or92[u_1,1,u_2]), head (in[1,u_3]);
    or92 := (=[u_1,u_2]) v (=[u_3,u_2]). Compact (impltest_or[a,b,M]) ->
    substituted premise (or92[a,1,b]) -> flattened disjuncts (=[a,1]) /
    (=[b,1]) -> intro candidates (>[1](=[a,1])(or92[a,1,b])) etc. The row
    carries the chapter-renamed bound variable w1; the
    normalize-with-unchangeables comparison equates it with the compiled
    bound variable. The full-implication primary path stays intact (the
    main rule row also passes, asserted alongside)."""
    state = make_state_with_binaries(("Peano",))
    state.gl_binaries = dict(state.gl_binaries)
    state.gl_binaries["Peano"] = dict(state.gl_binaries["Peano"])
    state.gl_binaries["Peano"]["or92"] = {
        "category": "or",
        "signature": "(or92[u_1,u_2,u_3])",
        "arity": 3,
        "elements": ["(=[u_1,u_2])", "(=[u_3,u_2])"],
    }
    state.gl_binaries["Peano"]["impltest_or"] = {
        "category": "implication",
        "signature": "(impltest_or[u_1,u_2,u_3])",
        "elements": ["(or92[u_1,1,u_2])", "(in[1,u_3])"],
    }
    state.current_gl_binary = state.gl_binaries["Peano"]
    compact_origin = make_proof_line(
        "(impltest_or[a,b,M])", "main", "task formulation",
    )
    intro_first = make_proof_line(
        "(>[w1](=[a,w1])(or92[a,w1,b]))", "main", "expansion",
        "(impltest_or[a,b,M])", "main",
    )
    intro_second = make_proof_line(
        "(>[w1](=[b,w1])(or92[a,w1,b]))", "main", "expansion",
        "(impltest_or[a,b,M])", "main",
    )
    main_rule = make_proof_line(
        "(>[w1](or92[a,w1,b])(in[w1,M]))", "main", "expansion",
        "(impltest_or[a,b,M])", "main",
    )
    chapter = [compact_origin, intro_first, intro_second, main_rule]
    assert_pass(check_expansion, intro_first, chapter, state)
    assert_pass(check_expansion, intro_second, chapter, state)
    assert_pass(check_expansion, main_rule, chapter, state)


@register
def test_pos_compilation_well_formed():
    """A `compilation` row whose compact (implication<N>[...]) faithfully
    reconstructs the cited original implication via the GL binary.

    Synthetic entry impltest1 := premise (in[1,u_1]), head (in[1,u_2]).
    Compact (impltest1[a,b]) -> subst {u_1:a,u_2:b} ->
    _build_implication_from_elements(["(in[1,a])","(in[1,b])"], {a,b})
    == "(>[1](in[1,a])(in[1,b]))" (bound var 1 placed at its first
    premise). rest[0] is set to exactly that, so the
    normalize-with-unchangeables comparison inside _try_expand is an
    identity match. Self-contained (compilation is origin-exempt) — the
    chapter is just the row itself."""
    state = make_state_with_binaries(("Peano",))
    state.gl_binaries = dict(state.gl_binaries)
    state.gl_binaries["Peano"] = dict(state.gl_binaries["Peano"])
    state.gl_binaries["Peano"]["impltest1"] = {
        "category": "implication",
        "signature": "(impltest1[u_1,u_2])",
        "elements": ["(in[1,u_1])", "(in[1,u_2])"],
    }
    state.current_gl_binary = state.gl_binaries["Peano"]
    line = make_proof_line(
        "(impltest1[a,b])", "main", "compilation",
        "(>[1](in[1,a])(in[1,b]))", "main",
    )
    assert_pass(check_compilation, line, [line], state)


# ===========================================================================
#  Integration family (expansion + reformulation for integration)
# ===========================================================================

def _state_with_andint_two_elem() -> object:
    """Two-element AND compact for the integration reformulation test."""
    state = make_state_with_binaries(("Peano",))
    state.gl_binaries = dict(state.gl_binaries)
    state.gl_binaries["Peano"] = dict(state.gl_binaries["Peano"])
    state.gl_binaries["Peano"]["andint1"] = {
        "category": "and",
        "signature": "(andint1[u_1,u_2])",
        "elements": ["(in[u_1,N])", "(=[u_1,u_2])"],
    }
    state.current_gl_binary = state.gl_binaries["Peano"]
    return state


@register
def test_pos_expansion_for_integration_well_formed():
    """Integration-time expansion: no origin presence required; structural
    match alone suffices."""
    state = _state_with_and_test()
    row = make_proof_line(
        "(in[a,N])", "main", "expansion for integration",
        "(andsimple[a])", "main",
    )
    assert_pass(check_expansion_for_integration, row, [row], state)


@register
def test_pos_reformulation_for_integration_and_well_formed():
    """The reformulated AND chain. For elements [E1, E2] the chain is
    (>[]E1(>[]E2 compact)) — flatten produces [E1, E2] in original order,
    and the chain wraps each elem around the compact from inside out."""
    state = _state_with_andint_two_elem()
    expansion = make_proof_line(
        "(&(in[a,N])(=[a,b]))", "main", "expansion for integration",
        "(andint1[a,b])", "main",
    )
    # _flatten_and("(&(in[a,N])(=[a,b]))") = ["(in[a,N])", "(=[a,b])"].
    # Chain: expected = compact; for elem in reversed(elements):
    #   expected = "(>[](=[a,b])(andint1[a,b]))"     # first iter
    #   expected = "(>[](in[a,N])(>[](=[a,b])(andint1[a,b])))"  # second
    row = make_proof_line(
        "(>[](in[a,N])(>[](=[a,b])(andint1[a,b])))",
        "main", "reformulation for integration and",
        "(&(in[a,N])(=[a,b]))", "main",
    )
    chapter = [row, expansion]
    assert_pass(check_reformulation_for_integration_and, row, chapter, state)


@register
def test_pos_premise_element_well_formed():
    """premise element row whose expression is one of the disintegrated
    premises of the cited origin implication, and whose namespace ends with
    the cleanSig boundary marker."""
    state = make_state_with_binaries(("Peano",))
    expansion = make_proof_line(
        "(>[](in[a,N])(=[a,b]))", "main", "expansion for integration",
        "(impl_compact)", "main",
    )
    prem = make_proof_line(
        "(in[a,N])", "(impl_compact)", "premise element",
        "(>[](in[a,N])(=[a,b]))", "main",
    )
    chapter = [expansion, prem]
    assert_pass(check_premise_element, prem, chapter, state)


@register
def test_pos_validity_name_well_formed():
    """validity name row asserts that the cited compact's head equals
    rest[0]."""
    state = make_state_with_binaries(("Peano",))
    expansion = make_proof_line(
        "(>[](in[a,N])(=[a,b]))", "main", "expansion for integration",
        "(impl_compact)", "main",
    )
    val = make_proof_line(
        "(impl_compact)", "main", "validity name",
        "(=[a,b])", "main",
    )
    chapter = [expansion, val]
    assert_pass(check_validity_name, val, chapter, state)


@register
def test_pos_recursion_check_induction_condition_typing_anchor():
    """The simple branch of check_induction_condition: a (in2[*,ind_var,s])
    typing-anchor row. The heavier reconstruction branch needs a carefully
    chosen theorem whose digit-arg / immutable-arg analysis lines up with the
    fixture's input_indices / output_indices; not covered here."""
    state = make_state_with_binaries(("Peano",))
    set_chapter_context(
        state,
        thm=("(>[v1](AnchorPeano[N,s,p,zero,one,two])(in[v1,N]))",
             "induction", "v1"),
        chapter_type="check_induction_condition",
    )
    line = make_proof_line("(in2[x_var,v1,s])", "main", "recursion")
    assert_pass(check_recursion, line, [line], state)


# ===========================================================================
#  Chapter-level: theorem goal reached (well-formed direct proof)
# ===========================================================================

# Tested implicitly by other positives that pass through verify_chapter; one
# explicit positive added below in the chapter-meta section.


# ===========================================================================
#  Chapter-level meta-check passes (rig sanity for verify_chapter)
# ===========================================================================

@register
def test_pos_meta_well_formed_chapter_passes_all():
    """Build a tiny well-formed direct-proof chapter and assert
    verify_chapter records ZERO failures on every meta-check we care about
    (theorem goal reached, origin, definition set consistency, origin chain
    termination)."""
    from tests.test_harness import assert_chapter_meta_pass
    chapter_thm = (
        "(>[v1](AnchorPeano[N,s,p,zero,one,two])(in[v1,N]))",
        "direct", "ref",
    )
    chapter = [
        make_proof_line("(in[v1,N])", "main", "task formulation"),
    ]
    for meta in ("theorem goal reached", "origin",
                 "definition set consistency", "origin chain termination"):
        assert_chapter_meta_pass(
            chapter, meta,
            chapter_thm=chapter_thm, chapter_type="direct_proof",
        )


@register
def test_pos_meta_no_self_reference_recorded():
    """Without any 'theorem' rows in chapter, self-reference counter never
    records anything (success=0, failure=0). assert_chapter_meta_pass
    accepts that (zero failures)."""
    from tests.test_harness import assert_chapter_meta_pass
    chapter_thm = (
        "(>[v1](AnchorPeano[N,s,p,zero,one,two])(in[v1,N]))",
        "direct", "ref",
    )
    chapter = [
        make_proof_line("(in[v1,N])", "main", "task formulation"),
    ]
    assert_chapter_meta_pass(
        chapter, "self-reference",
        chapter_thm=chapter_thm, chapter_type="direct_proof",
    )


@register
def test_pos_meta_well_formed_chapter_no_cycle():
    """Two-row chain B <- A (B cites A; no cycles)."""
    from tests.test_harness import assert_chapter_meta_pass
    chapter_thm = (
        "(>[v1](AnchorPeano[N,s,p,zero,one,two])(in[v1,N]))",
        "direct", "ref",
    )
    a = make_proof_line(
        "(in[v1,N])", "main", "task formulation",
    )
    b = make_proof_line(
        "(eq[v1,v1])", "main", "task formulation",
        "(in[v1,N])", "main",
    )
    chapter = [a, b]
    assert_chapter_meta_pass(
        chapter, "origin chain termination",
        chapter_thm=chapter_thm, chapter_type="direct_proof",
    )


if __name__ == "__main__":
    sys.exit(run_all_tests())
