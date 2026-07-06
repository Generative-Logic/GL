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

"""Chapter-level meta-check failure tests.

Each test builds a chapter (list of ProofLine) that violates exactly ONE
chapter-level invariant enforced inside ``verify_chapter`` and asserts the
corresponding counter records the failure. Covered meta-checks: theorem
goal reached, self-reference, anchor handling uniqueness, anchor handling
trace, contradiction trace, vacuous truth trace, origin, definition set
consistency (D-41), origin chain termination (cycle detection).

See ``tests/test_harness.py`` for fixtures and runner.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_harness import (  # noqa: E402
    register, make_state_with_binaries, make_proof_line,
    assert_chapter_meta_fail, assert_chapter_meta_pass, run_all_tests,
)


_GOAL_THM = ("(>[v1](AnchorPeano[N,s,p,zero,one,two])(in[v1,N]))",
             "direct", "ref")


def _goal_passing_first_line():
    """A first chapter row that satisfies check_theorem_goal_reached for
    the ``direct_proof`` chapter type with ``_GOAL_THM`` (head is
    ``(in[v1,N])``)."""
    return make_proof_line("(in[v1,N])", "main", "task formulation")


# ===========================================================================
#  meta: theorem goal reached
# ===========================================================================

@register
def test_theorem_goal_reached_no_chapter_thm():
    """chapter_thm None -> goal_reached records failure."""
    chapter = [make_proof_line("(in[v1,N])", "main", "task formulation")]
    assert_chapter_meta_fail(
        chapter, "theorem goal reached",
        chapter_thm=None, chapter_type="direct_proof",
    )


@register
def test_theorem_goal_reached_first_line_wrong_expression():
    """direct_proof: first line's expression differs from theorem head."""
    chapter = [make_proof_line("(eq[v1,v2])", "main", "task formulation")]
    assert_chapter_meta_fail(
        chapter, "theorem goal reached",
        chapter_thm=_GOAL_THM, chapter_type="direct_proof",
    )


@register
def test_theorem_goal_reached_first_line_wrong_namespace():
    """direct_proof: first line's namespace is not main."""
    chapter = [make_proof_line(
        "(in[v1,N])",
        "main_boundary_orint_X_((=[a,b]))",
        "task formulation")]
    assert_chapter_meta_fail(
        chapter, "theorem goal reached",
        chapter_thm=_GOAL_THM, chapter_type="direct_proof",
    )


@register
def test_theorem_goal_reached_unknown_chapter_type():
    """An unrecognised chapter_type returns False from
    check_theorem_goal_reached."""
    chapter = [_goal_passing_first_line()]
    assert_chapter_meta_fail(
        chapter, "theorem goal reached",
        chapter_thm=_GOAL_THM, chapter_type="fantasy_unknown",
    )


# ===========================================================================
#  meta: self-reference
# ===========================================================================

@register
def test_self_reference_theorem_cites_itself():
    """A 'theorem' row in the chapter has expression == chapter's own
    theorem expression -> self-reference failure recorded."""
    chapter = [
        _goal_passing_first_line(),
        make_proof_line(
            "(>[v1](AnchorPeano[N,s,p,zero,one,two])(in[v1,N]))",
            "main", "theorem",
        ),
    ]
    assert_chapter_meta_fail(
        chapter, "self-reference",
        chapter_thm=_GOAL_THM, chapter_type="direct_proof",
    )


@register
def test_self_reference_via_two_rows():
    """Two 'theorem' rows both cite the chapter's own theorem -> 2 failures
    recorded on the self-reference counter."""
    chapter = [
        _goal_passing_first_line(),
        make_proof_line(
            "(>[v1](AnchorPeano[N,s,p,zero,one,two])(in[v1,N]))",
            "main", "theorem",
        ),
        make_proof_line(
            "(>[v1](AnchorPeano[N,s,p,zero,one,two])(in[v1,N]))",
            "main", "theorem",
        ),
    ]
    assert_chapter_meta_fail(
        chapter, "self-reference",
        chapter_thm=_GOAL_THM, chapter_type="direct_proof",
        expected_failures=2,
    )


@register
def test_self_reference_chapter_type_irrelevant():
    """Self-reference fires irrespective of chapter_type."""
    chapter = [
        make_proof_line("(in[v1,N])", "main", "reformulated from",
                        "(>[v1](AnchorPeano[N,s,p,zero,one,two])"
                        "(in[v1,N]))", "main"),
        make_proof_line(
            "(>[v1](AnchorPeano[N,s,p,zero,one,two])(in[v1,N]))",
            "main", "theorem",
        ),
    ]
    assert_chapter_meta_fail(
        chapter, "self-reference",
        chapter_thm=_GOAL_THM, chapter_type="reformulated_statement",
    )


@register
def test_self_reference_ignores_tag_mismatch():
    """A row whose expression equals the theorem but whose tag is NOT
    'theorem' must NOT trip self-reference (so check passes there). We
    add a real self-ref row alongside to ensure the counter still records
    the genuine violation."""
    chapter = [
        _goal_passing_first_line(),
        # Tag is 'task formulation', not 'theorem' -> no self-ref hit.
        make_proof_line(
            "(>[v1](AnchorPeano[N,s,p,zero,one,two])(in[v1,N]))",
            "main", "task formulation",
        ),
        # Tag IS 'theorem' on the same expression -> self-ref hit.
        make_proof_line(
            "(>[v1](AnchorPeano[N,s,p,zero,one,two])(in[v1,N]))",
            "main", "theorem",
        ),
    ]
    assert_chapter_meta_fail(
        chapter, "self-reference",
        chapter_thm=_GOAL_THM, chapter_type="direct_proof",
        expected_failures=1,
    )


# ===========================================================================
#  meta: anchor handling uniqueness
# ===========================================================================

@register
def test_anchor_handling_uniqueness_two_rows():
    """Two 'anchor handling' rows in one chapter -> uniqueness failure."""
    chapter = [
        _goal_passing_first_line(),
        make_proof_line(
            "(AnchorPeano[N,s,p,zero,one,two])", "main", "anchor handling",
            "(AnchorPeano[N,s,p,zero,one,two])", "main",
        ),
        make_proof_line(
            "(AnchorPeano[N,s,p,zero,one,two])", "main", "anchor handling",
            "(AnchorPeano[N,s,p,zero,one,two])", "main",
        ),
    ]
    assert_chapter_meta_fail(
        chapter, "anchor handling uniqueness",
        chapter_thm=_GOAL_THM, chapter_type="direct_proof",
    )


@register
def test_anchor_handling_uniqueness_three_rows():
    chapter = [
        _goal_passing_first_line(),
        make_proof_line("(AnchorPeano[N,s,p,zero,one,two])",
                        "main", "anchor handling",
                        "(AnchorPeano[N,s,p,zero,one,two])", "main"),
        make_proof_line("(AnchorPeano[N,s,p,zero,one,two])",
                        "main", "anchor handling",
                        "(AnchorPeano[N,s,p,zero,one,two])", "main"),
        make_proof_line("(AnchorPeano[N,s,p,zero,one,two])",
                        "main", "anchor handling",
                        "(AnchorPeano[N,s,p,zero,one,two])", "main"),
    ]
    assert_chapter_meta_fail(
        chapter, "anchor handling uniqueness",
        chapter_thm=_GOAL_THM, chapter_type="direct_proof",
    )


@register
def test_anchor_handling_uniqueness_four_rows():
    chapter = [_goal_passing_first_line()]
    for _ in range(4):
        chapter.append(make_proof_line(
            "(AnchorPeano[N,s,p,zero,one,two])", "main", "anchor handling",
            "(AnchorPeano[N,s,p,zero,one,two])", "main",
        ))
    assert_chapter_meta_fail(
        chapter, "anchor handling uniqueness",
        chapter_thm=_GOAL_THM, chapter_type="direct_proof",
    )


# ===========================================================================
#  meta: anchor handling trace
# ===========================================================================

@register
def test_anchor_handling_trace_copy_var_untraceable():
    """anchor handling row has _copy var X_copy; another row's rest cites
    an expression containing X_copy without an edge back to anchor."""
    anchor_handling = make_proof_line(
        "(AnchorPeano[X_copy,s,p,zero,one,two])", "main", "anchor handling",
        "(AnchorPeano[X,s,p,zero,one,two])", "main",
    )
    user = make_proof_line(
        "(target_result)", "main", "implication",
        "(>[v1](in[v1,N])(in[v1,N]))", "main",
        "(in[X_copy,N])", "main",   # X_copy in REST source
    )
    chapter = [_goal_passing_first_line(), anchor_handling, user]
    assert_chapter_meta_fail(
        chapter, "anchor handling trace",
        chapter_thm=_GOAL_THM, chapter_type="direct_proof",
    )


@register
def test_anchor_handling_trace_compound_user_uncovered():
    """User row cites a compound source that contains X_copy; trace_back
    finds no anchor handling along the source chain."""
    anchor_handling = make_proof_line(
        "(AnchorPeano[X_copy,s,p,zero,one,two])", "main", "anchor handling",
        "(AnchorPeano[X,s,p,zero,one,two])", "main",
    )
    middle = make_proof_line(
        "(unrelated[a,b])", "main", "task formulation",
    )
    user = make_proof_line(
        "(another_target)", "main", "implication",
        "(>[v1](in[v1,N])(in[v1,N]))", "main",
        "(=[X_copy,unrelated_b])", "main",   # X_copy in REST source
    )
    chapter = [_goal_passing_first_line(), anchor_handling, middle, user]
    assert_chapter_meta_fail(
        chapter, "anchor handling trace",
        chapter_thm=_GOAL_THM, chapter_type="direct_proof",
    )


@register
def test_anchor_handling_trace_multi_user_all_uncovered():
    """Two user rows reference X_copy in their sources; both have no edge
    back to anchor handling -> 2 failures recorded."""
    anchor_handling = make_proof_line(
        "(AnchorPeano[X_copy,s,p,zero,one,two])", "main", "anchor handling",
        "(AnchorPeano[X,s,p,zero,one,two])", "main",
    )
    user1 = make_proof_line(
        "(in[a,N])", "main", "implication",
        "(>[v1](in[v1,N])(in[v1,N]))", "main",
        "(in[X_copy,N])", "main",
    )
    user2 = make_proof_line(
        "(in[b,N])", "main", "implication",
        "(>[v1](in[v1,N])(in[v1,N]))", "main",
        "(in[X_copy,N])", "main",
    )
    chapter = [_goal_passing_first_line(), anchor_handling, user1, user2]
    assert_chapter_meta_fail(
        chapter, "anchor handling trace",
        chapter_thm=_GOAL_THM, chapter_type="direct_proof",
        expected_failures=2,
    )


@register
def test_anchor_handling_trace_dangling_source():
    """User row's source ns equals expression and trace target is missing
    completely; trace_back returns False -> failure recorded."""
    anchor_handling = make_proof_line(
        "(AnchorPeano[Y_copy,s,p,zero,one,two])", "main", "anchor handling",
        "(AnchorPeano[Y,s,p,zero,one,two])", "main",
    )
    user = make_proof_line(
        "(in[Y_copy,M])", "main", "task formulation",
        "(in[Y_copy,N])", "main",
    )
    chapter = [_goal_passing_first_line(), anchor_handling, user]
    assert_chapter_meta_fail(
        chapter, "anchor handling trace",
        chapter_thm=_GOAL_THM, chapter_type="direct_proof",
    )


@register
def test_anchor_handling_trace_indirect_chain_drops_copy():
    """User cites a chain that drops Z_copy mid-walk -> trace fails."""
    anchor_handling = make_proof_line(
        "(AnchorPeano[Z_copy,s,p,zero,one,two])", "main", "anchor handling",
        "(AnchorPeano[Z,s,p,zero,one,two])", "main",
    )
    user = make_proof_line(
        "(target_2)", "main", "implication",
        "(>[v1](in[v1,N])(in[v1,N]))", "main",
        "(in[Z_copy,N])", "main",   # carries Z_copy in source
    )
    chapter = [_goal_passing_first_line(), anchor_handling, user]
    assert_chapter_meta_fail(
        chapter, "anchor handling trace",
        chapter_thm=_GOAL_THM, chapter_type="direct_proof",
    )


@register
def test_anchor_handling_trace_user_in_isolated_row():
    """A single user row carries the copy var in its first source and has
    no further support; trace walk dead-ends."""
    anchor_handling = make_proof_line(
        "(AnchorPeano[W_copy,s,p,zero,one,two])", "main", "anchor handling",
        "(AnchorPeano[W,s,p,zero,one,two])", "main",
    )
    user = make_proof_line(
        "(in[a,N])", "main", "implication",
        "(>[v1](in[v1,N])(in[v1,N]))", "main",
        "(in[W_copy,N])", "main",
    )
    chapter = [_goal_passing_first_line(), anchor_handling, user]
    assert_chapter_meta_fail(
        chapter, "anchor handling trace",
        chapter_thm=_GOAL_THM, chapter_type="direct_proof",
    )


# ===========================================================================
#  meta: contradiction trace
# ===========================================================================

def _build_unreachable_contradiction(seed_expr: str = "(cleanop_seed)"):
    """Standard contradiction row + 'task formulation' seed where neither
    contradicting expression can be traced back to seed."""
    contr = make_proof_line(
        "!" + seed_expr, "main", "contradiction",
        "(in[a,N])", "main",
        "!(in[a,N])", "main",
        seed_expr, "main",
    )
    seed = make_proof_line(seed_expr, "main", "task formulation")
    return [contr, seed]


@register
def test_contradiction_trace_no_paths_to_seed():
    chapter = [_goal_passing_first_line()] + _build_unreachable_contradiction()
    assert_chapter_meta_fail(
        chapter, "contradiction trace",
        chapter_thm=_GOAL_THM, chapter_type="direct_proof",
    )


@register
def test_contradiction_trace_rest_too_short():
    """Contradiction row's rest has fewer than 6 entries -> trace records
    failure on the short row."""
    contr = make_proof_line(
        "!(cleanop)", "main", "contradiction",
        "(in[a,N])", "main",
        "!(in[a,N])", "main",
        # missing cleanop pair
    )
    chapter = [_goal_passing_first_line(), contr]
    assert_chapter_meta_fail(
        chapter, "contradiction trace",
        chapter_thm=_GOAL_THM, chapter_type="direct_proof",
    )


@register
def test_contradiction_trace_two_contradictions_both_dangling():
    chapter = [_goal_passing_first_line()]
    chapter.extend(_build_unreachable_contradiction("(seed_A)"))
    chapter.extend(_build_unreachable_contradiction("(seed_B)"))
    assert_chapter_meta_fail(
        chapter, "contradiction trace",
        chapter_thm=_GOAL_THM, chapter_type="direct_proof",
        expected_failures=2,
    )


@register
def test_contradiction_trace_seed_present_but_no_chain():
    """The seed (task formulation) is present; the contradicting ingredients
    are present too — but neither traces to the seed via origin edges."""
    contr = make_proof_line(
        "!(seed_with_chain)", "main", "contradiction",
        "(in[a,N])", "main",
        "!(in[a,N])", "main",
        "(seed_with_chain)", "main",
    )
    in_line = make_proof_line("(in[a,N])", "main", "task formulation")
    neg_in_line = make_proof_line("!(in[a,N])", "main", "task formulation")
    seed = make_proof_line("(seed_with_chain)", "main", "task formulation")
    chapter = [_goal_passing_first_line(), contr,
               in_line, neg_in_line, seed]
    assert_chapter_meta_fail(
        chapter, "contradiction trace",
        chapter_thm=_GOAL_THM, chapter_type="direct_proof",
    )


@register
def test_contradiction_trace_different_seed_expression():
    """Contradiction cites cleanop = (seed_A); but the chapter's only
    task formulation has expression (seed_B). Trace target predicate
    never matches."""
    contr = make_proof_line(
        "!(seed_A)", "main", "contradiction",
        "(in[a,N])", "main",
        "!(in[a,N])", "main",
        "(seed_A)", "main",
    )
    seed = make_proof_line("(seed_B)", "main", "task formulation")
    chapter = [_goal_passing_first_line(), contr, seed]
    assert_chapter_meta_fail(
        chapter, "contradiction trace",
        chapter_thm=_GOAL_THM, chapter_type="direct_proof",
    )


@register
def test_contradiction_trace_seed_in_wrong_namespace():
    """Seed exists in chapter as task formulation but in a non-main ns;
    the target predicate requires (cl.tag=='task formulation' and
    cl.expression == seed), no ns filter — so this would PASS the trace.
    Use a different malformation: seed has wrong TAG."""
    contr = make_proof_line(
        "!(my_seed)", "main", "contradiction",
        "(in[a,N])", "main",
        "!(in[a,N])", "main",
        "(my_seed)", "main",
    )
    seed_impostor = make_proof_line(
        "(my_seed)", "main", "implication",  # wrong tag
        "(>[v1](in[v1,N])(in[v1,N]))", "main",
        "(in[a,N])", "main",
    )
    chapter = [_goal_passing_first_line(), contr, seed_impostor]
    assert_chapter_meta_fail(
        chapter, "contradiction trace",
        chapter_thm=_GOAL_THM, chapter_type="direct_proof",
    )


# ===========================================================================
#  meta: vacuous truth trace
# ===========================================================================
#
# The check used to require that at least one of the contradicting
# ingredients trace back through the chapter's origin graph to the LB
# expression at rest[4]. That requirement was dropped on
# 2026-05-26 evening (D-99) because
# it rejected legitimate vacuous discharges where the theorem's own
# premise is impossible — there the contradiction is rooted in the
# theorem's outer premise + axioms, not in the inner recursion step's
# hypothesis. The check is now shape-only: at least six rest fields.
# Tests that previously expected failure under the trace requirement
# now expect pass; only the malformed-row case remains a failure.


def _build_unreachable_vacuous(lb_key: str = "(lb_key)"):
    vac = make_proof_line(
        "(eq[v1,v1])", "main", "vacuous truth",
        "(in[v1,N])", "main",
        "!(in[v1,N])", "main",
        lb_key, "main",
    )
    return [vac]


@register
def test_vacuous_truth_trace_well_formed_row_passes():
    """Well-formed vacuous-truth row passes the shape-only check even
    when no chapter row traces back to rest[4]. Pre-relaxation this was
    a failure; post-relaxation it is accepted."""
    chapter = [_goal_passing_first_line()] + _build_unreachable_vacuous()
    assert_chapter_meta_pass(
        chapter, "vacuous truth trace",
        chapter_thm=_GOAL_THM, chapter_type="direct_proof",
    )


@register
def test_vacuous_truth_trace_rest_too_short():
    """A vacuous-truth row with fewer than six rest fields is malformed
    and still records a failure under the shape-only check."""
    vac = make_proof_line(
        "(eq[v1,v1])", "main", "vacuous truth",
        "(in[v1,N])", "main",
        "!(in[v1,N])", "main",
        # missing lb_key pair
    )
    chapter = [_goal_passing_first_line(), vac]
    assert_chapter_meta_fail(
        chapter, "vacuous truth trace",
        chapter_thm=_GOAL_THM, chapter_type="direct_proof",
    )


@register
def test_vacuous_truth_trace_two_well_formed_rows_both_pass():
    """Two well-formed vacuous-truth rows in the same chapter, each
    naming a distinct lb_key. Both pass the shape-only check."""
    chapter = [_goal_passing_first_line()]
    chapter.extend(_build_unreachable_vacuous("(lb_key_A)"))
    chapter.extend(_build_unreachable_vacuous("(lb_key_B)"))
    assert_chapter_meta_pass(
        chapter, "vacuous truth trace",
        chapter_thm=_GOAL_THM, chapter_type="direct_proof",
    )


@register
def test_vacuous_truth_trace_lb_key_present_but_unreachable_passes():
    """The lb_key expression exists in chapter but no origin path from
    the contradicting ingredients reaches it. Pre-relaxation this was a
    failure; post-relaxation it passes (the soundness argument relies
    on axiom consistency, not on the rest[4]-trace requirement)."""
    vac = make_proof_line(
        "(eq[v1,v1])", "main", "vacuous truth",
        "(in[v1,N])", "main",
        "!(in[v1,N])", "main",
        "(lb_key)", "main",
    )
    lb_present = make_proof_line(
        "(lb_key)", "main", "task formulation",
    )
    in_line = make_proof_line("(in[v1,N])", "main", "task formulation")
    neg_in_line = make_proof_line("!(in[v1,N])", "main", "task formulation")
    chapter = [_goal_passing_first_line(), vac,
               in_line, neg_in_line, lb_present]
    assert_chapter_meta_pass(
        chapter, "vacuous truth trace",
        chapter_thm=_GOAL_THM, chapter_type="direct_proof",
    )


@register
def test_vacuous_truth_trace_lb_key_completely_absent_passes():
    """rest[4] names an expression that does not appear anywhere else
    in the chapter. Post-relaxation passes (the third dep is now
    informational, not enforced)."""
    vac = make_proof_line(
        "(eq[v1,v1])", "main", "vacuous truth",
        "(in[v1,N])", "main",
        "!(in[v1,N])", "main",
        "(lb_key_never_present)", "main",
    )
    chapter = [_goal_passing_first_line(), vac]
    assert_chapter_meta_pass(
        chapter, "vacuous truth trace",
        chapter_thm=_GOAL_THM, chapter_type="direct_proof",
    )


@register
def test_vacuous_truth_trace_with_dangling_sources_passes():
    """The contradicting ingredients have origin rows that terminate at
    dangling sources (never reach the lb_key). Post-relaxation passes."""
    vac = make_proof_line(
        "(eq[v1,v1])", "main", "vacuous truth",
        "(in[v1,N])", "main",
        "!(in[v1,N])", "main",
        "(lb_key)", "main",
    )
    in_line = make_proof_line(
        "(in[v1,N])", "main", "implication",
        "(>[v1](in[v1,N])(in[v1,N]))", "main",
        "(dangling_A)", "main",
    )
    neg_line = make_proof_line(
        "!(in[v1,N])", "main", "implication",
        "(>[v1](in[v1,N])!(in[v1,N]))", "main",
        "(dangling_B)", "main",
    )
    chapter = [_goal_passing_first_line(), vac, in_line, neg_line]
    assert_chapter_meta_pass(
        chapter, "vacuous truth trace",
        chapter_thm=_GOAL_THM, chapter_type="direct_proof",
    )


# ===========================================================================
#  meta: origin
# ===========================================================================

@register
def test_origin_unreferenced_dependency():
    """An implication row cites a dep that doesn't exist as a chapter row's
    left-side expression."""
    impl_line = make_proof_line(
        "(in[a,N])", "main", "implication",
        "(>[v1](in[v1,N])(in[v1,N]))", "main",
        "(in[ghost,N])", "main",   # ghost dep
    )
    chapter = [_goal_passing_first_line(), impl_line]
    assert_chapter_meta_fail(
        chapter, "origin",
        chapter_thm=_GOAL_THM, chapter_type="direct_proof",
    )


@register
def test_origin_multiple_unreferenced_deps():
    """Two ghost deps in one row -> 2 origin failures."""
    impl_line = make_proof_line(
        "(in[a,N])", "main", "implication",
        "(>[v1,v2](in[v1,N])(in[v2,N])(in[v1,N]))", "main",
        "(ghost_1)", "main",
        "(ghost_2)", "main",
    )
    chapter = [_goal_passing_first_line(), impl_line]
    assert_chapter_meta_fail(
        chapter, "origin",
        chapter_thm=_GOAL_THM, chapter_type="direct_proof",
        expected_failures=2,
    )


@register
def test_origin_implication_with_bad_rest0():
    """rest[0] of an implication isn't in global_theorems either."""
    impl_line = make_proof_line(
        "(in[a,N])", "main", "implication",
        "(>[v1](unknown_op[v1])(in[v1,N]))", "main",   # not in any registry
        "(unknown_op[a])", "main",
    )
    chapter = [_goal_passing_first_line(), impl_line]
    assert_chapter_meta_fail(
        chapter, "origin",
        chapter_thm=_GOAL_THM, chapter_type="direct_proof",
    )


@register
def test_origin_multiplied_from_bad_rest0():
    """multiplied from's rest[0] isn't in global_theorems."""
    line = make_proof_line(
        "(>[v1](in[v1,N])(in[v1,N]))", "main", "multiplied from",
        "(>[v1,v2](in[v1,unknown_M])(in[v2,unknown_M])(eq[v1,v2]))", "main",
    )
    chapter = [_goal_passing_first_line(), line]
    assert_chapter_meta_fail(
        chapter, "origin",
        chapter_thm=_GOAL_THM, chapter_type="direct_proof",
    )


@register
def test_origin_reformulated_from_bad_rest0():
    line = make_proof_line(
        "(>[v1](AnchorPeano[N,s,p,zero,one,two])(existence_X[v1]))",
        "main", "reformulated from",
        "(>[v1](AnchorPeano[N,s,p,zero,one,two])(unknown_op[v1]))", "main",
    )
    chapter = [_goal_passing_first_line(), line]
    assert_chapter_meta_fail(
        chapter, "origin",
        chapter_thm=_GOAL_THM, chapter_type="direct_proof",
    )


@register
def test_origin_simple_unreferenced_inside_disintegration():
    """A 'disintegration' row cites a compound that isn't in chapter at all."""
    line = make_proof_line(
        "(in[a,N])", "main", "disintegration",
        "(missing_compound[a])", "main",
    )
    chapter = [_goal_passing_first_line(), line]
    assert_chapter_meta_fail(
        chapter, "origin",
        chapter_thm=_GOAL_THM, chapter_type="direct_proof",
    )


@register
def test_origin_unreferenced_in_expansion():
    """expansion row cites a compact that isn't a left-side row anywhere."""
    line = make_proof_line(
        "(&[a,b])", "main", "expansion",
        "(missing_compact[a,b])", "main",
    )
    chapter = [_goal_passing_first_line(), line]
    assert_chapter_meta_fail(
        chapter, "origin",
        chapter_thm=_GOAL_THM, chapter_type="direct_proof",
    )


@register
def test_origin_chained_unreferenced():
    """Two rows cite each other's source, but one cites an external ghost."""
    a = make_proof_line(
        "(=[x,y])", "main", "implication",
        "(>[](=[x,y])(=[y,x]))", "main",
        "(=[y,x])", "main",
    )
    b = make_proof_line(
        "(=[y,x])", "main", "implication",
        "(>[](=[x,y])(=[y,x]))", "main",
        "(ghost_dep)", "main",   # ghost
    )
    chapter = [_goal_passing_first_line(), a, b]
    assert_chapter_meta_fail(
        chapter, "origin",
        chapter_thm=_GOAL_THM, chapter_type="direct_proof",
    )


@register
def test_origin_unreferenced_in_disintegration_chain():
    """A chain of disintegration + expansion where the inner expansion's
    rest cites a ghost compact."""
    disint = make_proof_line(
        "(in[a,N])", "main", "disintegration",
        "(some_compound[a])", "main",
    )
    expansion = make_proof_line(
        "(some_compound[a])", "main", "expansion",
        "(ghost_compact[a])", "main",
    )
    chapter = [_goal_passing_first_line(), disint, expansion]
    assert_chapter_meta_fail(
        chapter, "origin",
        chapter_thm=_GOAL_THM, chapter_type="direct_proof",
    )


# ===========================================================================
#  meta: definition set consistency (D-41)
# ===========================================================================

def _state_with_typed_atomics(**type_pairs):
    """Helper: build a state with custom atomic ops whose definition_sets
    are pre-registered. Each kwarg is op_name=type_label, e.g.
    ``_state_with_typed_atomics(t1='(TypeA)', t2='(TypeB)')``."""
    from verifier import build_resolved_defsets_per_tag
    state = make_state_with_binaries(("Peano",))
    state.gl_binaries = dict(state.gl_binaries)
    state.gl_binaries["Peano"] = dict(state.gl_binaries["Peano"])
    state.definition_sets = dict(state.definition_sets)
    for op_name, type_label in type_pairs.items():
        state.definition_sets[op_name] = {"1": [type_label, True]}
    state.resolved_defsets_per_tag, state.resolved_defsets_atomic_only = \
        build_resolved_defsets_per_tag(
            state.definition_sets, state.gl_binaries)
    state.current_resolved_defsets = state.resolved_defsets_atomic_only
    return state


@register
def test_defset_consistency_conjunction_var_two_types():
    """Single conjunction expression `(&(t1[x])(t2[x]))`: x at t1 (TypeA)
    AND at t2 (TypeB). _merge_maps fires _DefsetMismatch."""
    state = _state_with_typed_atomics(t1="(TypeA)", t2="(TypeB)")
    line = make_proof_line(
        "(&(t1[x])(t2[x]))", "main", "task formulation",
    )
    chapter = [_goal_passing_first_line(), line]
    assert_chapter_meta_fail(
        chapter, "definition set consistency",
        chapter_thm=_GOAL_THM, chapter_type="direct_proof",
        state=state,
    )


@register
def test_defset_consistency_implication_var_two_types():
    """Implication `(>[v1](t1[x])(t2[x]))`: x is FREE (not bound by v1).
    Merge of left/right subtrees fires _DefsetMismatch."""
    state = _state_with_typed_atomics(t1="(TypeA)", t2="(TypeB)")
    line = make_proof_line(
        "(>[v1](t1[x])(t2[x]))", "main", "task formulation",
    )
    chapter = [_goal_passing_first_line(), line]
    assert_chapter_meta_fail(
        chapter, "definition set consistency",
        chapter_thm=_GOAL_THM, chapter_type="direct_proof",
        state=state,
    )


@register
def test_defset_consistency_negated_conjunction_var_two_types():
    """`!(&(t1[x])(t2[x]))`: negated conjunction; subtree merge still fires."""
    state = _state_with_typed_atomics(t1="(TypeA)", t2="(TypeB)")
    line = make_proof_line(
        "!(&(t1[x])(t2[x]))", "main", "task formulation",
    )
    chapter = [_goal_passing_first_line(), line]
    assert_chapter_meta_fail(
        chapter, "definition set consistency",
        chapter_thm=_GOAL_THM, chapter_type="direct_proof",
        state=state,
    )


@register
def test_defset_consistency_two_rows_each_with_internal_conflict():
    """Two rows; each one's LEFT expression has an internal type collision
    -> 2 failures recorded."""
    state = _state_with_typed_atomics(t1="(TypeA)", t2="(TypeB)")
    line_a = make_proof_line(
        "(&(t1[a])(t2[a]))", "main", "task formulation",
    )
    line_b = make_proof_line(
        "(&(t1[b])(t2[b]))", "main", "task formulation",
    )
    chapter = [_goal_passing_first_line(), line_a, line_b]
    assert_chapter_meta_fail(
        chapter, "definition set consistency",
        chapter_thm=_GOAL_THM, chapter_type="direct_proof",
        expected_failures=2,
        state=state,
    )


@register
def test_defset_consistency_three_way_conjunction_collide():
    """Three-way conjunction `(&(&(t1[x])(t2[x]))(t3[x]))`: each merge fires."""
    state = _state_with_typed_atomics(
        t1="(TypeA)", t2="(TypeB)", t3="(TypeC)")
    line = make_proof_line(
        "(&(&(t1[x])(t2[x]))(t3[x]))", "main", "task formulation",
    )
    chapter = [_goal_passing_first_line(), line]
    assert_chapter_meta_fail(
        chapter, "definition set consistency",
        chapter_thm=_GOAL_THM, chapter_type="direct_proof",
        state=state,
    )


@register
def test_defset_consistency_rest_expression_internal_conflict():
    """The conflict lives in a rest[i] expression. _parse_subtree runs
    over rest[i] expressions too."""
    state = _state_with_typed_atomics(t1="(TypeA)", t2="(TypeB)")
    line = make_proof_line(
        "(some_target)", "main", "task formulation",
        "(&(t1[y])(t2[y]))", "main",   # conflict in rest[0]
    )
    chapter = [_goal_passing_first_line(), line]
    assert_chapter_meta_fail(
        chapter, "definition set consistency",
        chapter_thm=_GOAL_THM, chapter_type="direct_proof",
        state=state,
    )


@register
def test_defset_consistency_three_rows_independent_violations():
    """Three rows, each with its own internal conflict."""
    state = _state_with_typed_atomics(t1="(TypeA)", t2="(TypeB)")
    chapter = [_goal_passing_first_line()]
    for var in ("alpha", "beta", "gamma"):
        chapter.append(make_proof_line(
            f"(&(t1[{var}])(t2[{var}]))", "main", "task formulation",
        ))
    assert_chapter_meta_fail(
        chapter, "definition set consistency",
        chapter_thm=_GOAL_THM, chapter_type="direct_proof",
        expected_failures=3,
        state=state,
    )


@register
def test_defset_consistency_same_leaf_var_repeated_with_clash():
    """`(t12[w,w])` where t12 expects pos1 type A and pos2 type B -> single
    leaf, _process_leaf raises _DefsetMismatch on the same-var-twice case."""
    from verifier import build_resolved_defsets_per_tag
    state = make_state_with_binaries(("Peano",))
    state.definition_sets = dict(state.definition_sets)
    state.definition_sets["t12"] = {
        "1": ["(TypeA)", True],
        "2": ["(TypeB)", True],
    }
    state.resolved_defsets_per_tag, state.resolved_defsets_atomic_only = \
        build_resolved_defsets_per_tag(
            state.definition_sets, state.gl_binaries)
    state.current_resolved_defsets = state.resolved_defsets_atomic_only
    line = make_proof_line(
        "(t12[w,w])", "main", "task formulation",
    )
    chapter = [_goal_passing_first_line(), line]
    assert_chapter_meta_fail(
        chapter, "definition set consistency",
        chapter_thm=_GOAL_THM, chapter_type="direct_proof",
        state=state,
    )


# ===========================================================================
#  meta: origin chain termination (cycle detection)
# ===========================================================================

@register
def test_cycle_two_node():
    """A -> B -> A: classic 2-cycle. Each row tagged as 'task formulation'
    so origin exemption doesn't kick in."""
    a = make_proof_line(
        "(A)", "main", "implication",
        "(>[](B)(A))", "main",
        "(B)", "main",
    )
    b = make_proof_line(
        "(B)", "main", "implication",
        "(>[](A)(B))", "main",
        "(A)", "main",
    )
    # Provide stubs for the implications cited from rest[0]; otherwise their
    # absence triggers an 'origin' counter failure on the rest[0] check,
    # but the cycle detection still records its own failure.
    a_rule = make_proof_line(
        "(>[](B)(A))", "main", "task formulation",
    )
    b_rule = make_proof_line(
        "(>[](A)(B))", "main", "task formulation",
    )
    chapter = [_goal_passing_first_line(), a, b, a_rule, b_rule]
    assert_chapter_meta_fail(
        chapter, "origin chain termination",
        chapter_thm=_GOAL_THM, chapter_type="direct_proof",
        expected_failures=2,
    )


@register
def test_cycle_three_node():
    """A -> B -> C -> A."""
    a = make_proof_line(
        "(A)", "main", "implication",
        "(>[](C)(A))", "main",
        "(C)", "main",
    )
    b = make_proof_line(
        "(B)", "main", "implication",
        "(>[](A)(B))", "main",
        "(A)", "main",
    )
    c = make_proof_line(
        "(C)", "main", "implication",
        "(>[](B)(C))", "main",
        "(B)", "main",
    )
    rules = [
        make_proof_line("(>[](C)(A))", "main", "task formulation"),
        make_proof_line("(>[](A)(B))", "main", "task formulation"),
        make_proof_line("(>[](B)(C))", "main", "task formulation"),
    ]
    chapter = [_goal_passing_first_line(), a, b, c] + rules
    assert_chapter_meta_fail(
        chapter, "origin chain termination",
        chapter_thm=_GOAL_THM, chapter_type="direct_proof",
        expected_failures=3,
    )


@register
def test_cycle_four_node():
    a = make_proof_line(
        "(A)", "main", "implication",
        "(>[](D)(A))", "main", "(D)", "main",
    )
    b = make_proof_line(
        "(B)", "main", "implication",
        "(>[](A)(B))", "main", "(A)", "main",
    )
    c = make_proof_line(
        "(C)", "main", "implication",
        "(>[](B)(C))", "main", "(B)", "main",
    )
    d = make_proof_line(
        "(D)", "main", "implication",
        "(>[](C)(D))", "main", "(C)", "main",
    )
    rules = [
        make_proof_line("(>[](D)(A))", "main", "task formulation"),
        make_proof_line("(>[](A)(B))", "main", "task formulation"),
        make_proof_line("(>[](B)(C))", "main", "task formulation"),
        make_proof_line("(>[](C)(D))", "main", "task formulation"),
    ]
    chapter = [_goal_passing_first_line(), a, b, c, d] + rules
    assert_chapter_meta_fail(
        chapter, "origin chain termination",
        chapter_thm=_GOAL_THM, chapter_type="direct_proof",
        expected_failures=4,
    )


@register
def test_cycle_self_loop():
    """A cites itself."""
    a = make_proof_line(
        "(A)", "main", "implication",
        "(>[](A)(A))", "main",
        "(A)", "main",
    )
    rule = make_proof_line(
        "(>[](A)(A))", "main", "task formulation",
    )
    chapter = [_goal_passing_first_line(), a, rule]
    assert_chapter_meta_fail(
        chapter, "origin chain termination",
        chapter_thm=_GOAL_THM, chapter_type="direct_proof",
    )


@register
def test_cycle_with_foundation_outside():
    """Cycle A <-> B, plus a foundation row C used by both — cycle still
    flagged on A and B but C remains green."""
    a = make_proof_line(
        "(A)", "main", "implication",
        "(>[](B)(A))", "main", "(B)", "main",
        "(C)", "main",   # foundation; not on cycle
    )
    b = make_proof_line(
        "(B)", "main", "implication",
        "(>[](A)(B))", "main", "(A)", "main",
        "(C)", "main",
    )
    c = make_proof_line(
        "(C)", "main", "task formulation",
    )
    rules = [
        make_proof_line("(>[](B)(A))", "main", "task formulation"),
        make_proof_line("(>[](A)(B))", "main", "task formulation"),
    ]
    chapter = [_goal_passing_first_line(), a, b, c] + rules
    assert_chapter_meta_fail(
        chapter, "origin chain termination",
        chapter_thm=_GOAL_THM, chapter_type="direct_proof",
        expected_failures=2,
    )


@register
def test_cycle_dual_pair():
    """Two separate 2-cycles: A<->B and C<->D."""
    a = make_proof_line(
        "(A)", "main", "implication",
        "(>[](B)(A))", "main", "(B)", "main",
    )
    b = make_proof_line(
        "(B)", "main", "implication",
        "(>[](A)(B))", "main", "(A)", "main",
    )
    c = make_proof_line(
        "(C)", "main", "implication",
        "(>[](D)(C))", "main", "(D)", "main",
    )
    d = make_proof_line(
        "(D)", "main", "implication",
        "(>[](C)(D))", "main", "(C)", "main",
    )
    rules = [
        make_proof_line("(>[](B)(A))", "main", "task formulation"),
        make_proof_line("(>[](A)(B))", "main", "task formulation"),
        make_proof_line("(>[](D)(C))", "main", "task formulation"),
        make_proof_line("(>[](C)(D))", "main", "task formulation"),
    ]
    chapter = [_goal_passing_first_line(), a, b, c, d] + rules
    assert_chapter_meta_fail(
        chapter, "origin chain termination",
        chapter_thm=_GOAL_THM, chapter_type="direct_proof",
        expected_failures=4,
    )


# ===========================================================================
#  meta: operator registry consistency (I-23, cross-batch name uniqueness)
# ===========================================================================

def _make_registry_state(binaries):
    """Synthetic VerifierState with hand-crafted gl_binaries for the
    duplicity check. Bypasses load_gl_binaries entirely so tests can
    construct arbitrary collisions."""
    from verifier import VerifierState
    state = VerifierState()
    state.gl_binaries = binaries
    return state


def _entry(category, signature, arity, elements):
    return {"category": category, "signature": signature,
            "arity": arity, "elements": list(elements),
            "definedSet": ""}


def _silent_check_registry_consistency(state):
    """Run the duplicity check with stderr captured. Used by negative-
    path tests that deliberately feed divergent synthetic binaries —
    the check's per-name stderr report is verified by inspecting
    ``state.tag_counters``, and bleeding the report into the surrounding
    log (e.g. main.py's verifier unit-test gate) creates false-alarm
    output that looks like a production I-23 violation."""
    import contextlib, io
    from verifier import check_operator_registry_consistency
    with contextlib.redirect_stderr(io.StringIO()):
        check_operator_registry_consistency(state)


@register
def test_registry_consistency_clean_run():
    """Two binaries carrying byte-identical existence2 → success, no failure."""
    shared_e2 = _entry("existence", "(existence2[u_1,u_2,u_3])", 3,
                       ["(in[1,u_1])", "(in2[1,u_2,u_3])"])
    state = _make_registry_state({
        "shared": {"existence2": shared_e2},
        "Peano":  {"existence2": dict(shared_e2)},
    })
    _silent_check_registry_consistency(state)
    c = state.tag_counters["operator registry consistency"]
    assert c.success == 1 and c.failure == 0, (
        f"consistent existence2 must record success only; got "
        f"success={c.success} failure={c.failure}")


@register
def test_registry_consistency_surface_divergence_existence2():
    """The actual production divergence: IncubatorPeano existence2[8-arg]
    vs shared/Peano existence2[3-arg]. Recorded as 1 failure."""
    shared_e2 = _entry("existence", "(existence2[u_1,u_2,u_3])", 3,
                       ["(in[1,u_1])", "(in2[1,u_2,u_3])"])
    incub_e2 = _entry("existence",
                      "(existence2[u_1,u_2,u_3,u_4,u_5,u_6,u_7,u_8])", 8,
                      ["(fXY[1,u_1,u_2])",
                       "(and0[u_3,u_4,1,u_5,u_6,u_1,u_7,u_8])"])
    state = _make_registry_state({
        "shared":         {"existence2": shared_e2},
        "Peano":          {"existence2": dict(shared_e2)},
        "IncubatorPeano": {"existence2": incub_e2},
    })
    _silent_check_registry_consistency(state)
    c = state.tag_counters["operator registry consistency"]
    assert c.failure == 1 and c.success == 0, (
        f"divergent existence2 must record failure; got "
        f"success={c.success} failure={c.failure}")


@register
def test_registry_consistency_recursive_divergence_via_cited_op():
    """Surface-identical existence2 across two binaries, but the cited
    operator and0 diverges between them → recursive check catches it."""
    e2 = _entry("existence", "(existence2[u_1,u_2,u_3])", 3,
                ["(in[1,u_1])", "(and0[u_2,u_3,1])"])
    and0_a = _entry("and", "(and0[u_1,u_2,u_3])", 3,
                    ["(in[1,u_1])", "(in[1,u_2])", "(in[1,u_3])"])
    and0_b = _entry("and", "(and0[u_1,u_2,u_3])", 3,
                    ["(in2[1,u_1,u_2])", "(in[1,u_3])"])  # different elements
    state = _make_registry_state({
        "Peano":          {"existence2": dict(e2), "and0": and0_a},
        "IncubatorPeano": {"existence2": dict(e2), "and0": and0_b},
    })
    _silent_check_registry_consistency(state)
    c = state.tag_counters["operator registry consistency"]
    # existence2 is identical at the surface but its cited operator
    # diverges; the deep check rolls that up as a failure on existence2.
    # and0 is also checked top-level → another failure.
    assert c.failure == 2 and c.success == 0, (
        f"recursive divergence (existence2 cites diverging and0) must "
        f"record 2 failures (existence2 + and0); got "
        f"success={c.success} failure={c.failure}")


@register
def test_registry_consistency_single_binary_skipped():
    """A name present in only ONE binary contributes neither success nor
    failure — the check only fires on cross-batch presence."""
    e2 = _entry("existence", "(existence2[u_1,u_2,u_3])", 3,
                ["(in[1,u_1])", "(in2[1,u_2,u_3])"])
    state = _make_registry_state({
        "IncubatorPeano": {"existence2": e2},
    })
    _silent_check_registry_consistency(state)
    c = state.tag_counters.get("operator registry consistency")
    success = c.success if c is not None else 0
    failure = c.failure if c is not None else 0
    assert success == 0 and failure == 0, (
        f"single-binary name must not record any consistency event; got "
        f"success={success} failure={failure}")


@register
def test_registry_consistency_atomic_entries_ignored():
    """Atomic entries (category outside the spontaneous set, e.g. anchor
    operators) are skipped — they're batch-local by design."""
    anchor_a = {"category": "anchor",
                "signature": "(AnchorPeano[N,i0,s,+,*,i1])",
                "arity": 6,
                "elements": ["(>[N,i0,s,+,*,i1](NaturalNumbers[...])(...))"],
                "definedSet": ""}
    anchor_b = {"category": "anchor",
                "signature": "(AnchorPeano[N,i0,s,+,*,i1])",
                "arity": 6,
                "elements": ["completely different body"],  # would fail surface
                "definedSet": ""}
    state = _make_registry_state({
        "Peano":          {"AnchorPeano": anchor_a},
        "IncubatorPeano": {"AnchorPeano": anchor_b},
    })
    _silent_check_registry_consistency(state)
    c = state.tag_counters.get("operator registry consistency")
    success = c.success if c is not None else 0
    failure = c.failure if c is not None else 0
    assert success == 0 and failure == 0, (
        f"anchor entries (category != spontaneous) must not be checked; "
        f"got success={success} failure={failure}")


if __name__ == "__main__":
    sys.exit(run_all_tests())
