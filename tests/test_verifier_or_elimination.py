# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
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

"""Tests for ``check_or_elimination`` — the pre-split merge checker.

One positive case (the C13-shaped merge under an anchor-conditioned or
theorem), then one subtle malformation per gate. See
``tests/test_harness.py`` for fixtures and runner.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_harness import (  # noqa: E402
    register, make_state_with_globals, make_proof_line,
    assert_failure, assert_pass, run_all_tests,
)
from verifier import check_or_elimination  # noqa: E402


# Processed-form fixture (Peano-anchored). The merged claim (column 0)
# stays in registry v/V form; the three citations carry the processor's
# w/W citation form. The or theorem's head decodes through a synthetic
# `or900` binary entry to the two guards (true polarity).
MERGED = (
    "(>[N,i0,s,+,*,i1](AnchorPeano[N,i0,s,+,*,i1])"
    "(>[v1,v2](preorder[N,*,v1,v2])(>[](preorder[N,*,v2,v1])(=[v1,v2]))))"
)
VARIANT_A = (
    "(>[N,i0,s,+,*,i1](AnchorPeano[N,i0,s,+,*,i1])"
    "(>[w1,w2](preorder[N,*,w1,w2])(>[](preorder[N,*,w2,w1])"
    "(>[](preorder[N,+,i1,w2])(=[w1,w2])))))"
)
VARIANT_B = (
    "(>[N,i0,s,+,*,i1](AnchorPeano[N,i0,s,+,*,i1])"
    "(>[w1,w2](preorder[N,*,w1,w2])(>[](preorder[N,*,w2,w1])"
    "(>[](=[i0,w2])(=[w1,w2])))))"
)
OR_THM = (
    "(>[N,i0,s,+,*,i1](AnchorPeano[N,i0,s,+,*,i1])"
    "(>[w1](in[w1,N])(or900[i0,w1,N,+,i1])))"
)

OR900_ENTRY = {
    "arity": 5,
    "category": "or",
    "definedSet": "",
    "elements": ["(=[u_1,u_2])", "(preorder[u_3,u_4,u_5,u_2])"],
    "signature": "(or900[u_1,u_2,u_3,u_4,u_5])",
}


def _state(globals_rows=None, or_entry=OR900_ENTRY):
    rows = globals_rows if globals_rows is not None else [
        (VARIANT_A, "direct", "-1"),
        (VARIANT_B, "direct", "-1"),
        (OR_THM, "or theorem", "-1"),
    ]
    state = make_state_with_globals(rows)
    binary = dict(state.gl_binaries["Peano"])
    binary["or900"] = dict(or_entry)
    state.gl_binaries["Peano"] = binary
    state.current_gl_binary = binary
    return state


def _line(merged=MERGED, variant_a=VARIANT_A, variant_b=VARIANT_B,
          or_thm=OR_THM):
    return make_proof_line(merged, "main", "or elimination",
                           variant_a, "main", variant_b, "main",
                           or_thm, "main")


@register
def test_positive_merge_accepted():
    assert_pass(check_or_elimination, _line(), [], _state())


@register
def test_positive_reversed_guard_order():
    assert_pass(check_or_elimination,
                _line(variant_a=VARIANT_B, variant_b=VARIANT_A), [], _state())


@register
def test_unresolved_citation_rejected():
    state = _state(globals_rows=[
        (VARIANT_A, "direct", "-1"),
        (OR_THM, "or theorem", "-1"),
    ])
    assert_failure(check_or_elimination, _line(), [], state)


@register
def test_polarity_flip_rejected():
    variant_b_neg = VARIANT_B.replace("(>[](=[i0,w2])", "(>[]!(=[i0,w2])")
    state = _state(globals_rows=[
        (VARIANT_A, "direct", "-1"),
        (variant_b_neg, "direct", "-1"),
        (OR_THM, "or theorem", "-1"),
    ])
    assert_failure(check_or_elimination,
                   _line(variant_b=variant_b_neg), [], state)


@register
def test_two_premise_difference_rejected():
    variant_b_two = VARIANT_B.replace("(preorder[N,*,w2,w1])",
                                      "(preorder[N,+,w2,w1])")
    state = _state(globals_rows=[
        (VARIANT_A, "direct", "-1"),
        (variant_b_two, "direct", "-1"),
        (OR_THM, "or theorem", "-1"),
    ])
    assert_failure(check_or_elimination,
                   _line(variant_b=variant_b_two), [], state)


@register
def test_disjunct_mismatch_rejected():
    wrong_entry = dict(OR900_ENTRY)
    wrong_entry["elements"] = ["(=[u_1,u_2])",
                               "(strictOrder[u_3,u_4,u_5,u_2])"]
    assert_failure(check_or_elimination, _line(), [],
                   _state(or_entry=wrong_entry))


@register
def test_unbound_side_premise_rejected():
    or_thm_extra = OR_THM.replace(
        "(>[w1](in[w1,N])",
        "(>[w1](in[w1,N])(>[w3](in[w3,N])")
    or_thm_extra = or_thm_extra[:-1] + ")"
    state = _state(globals_rows=[
        (VARIANT_A, "direct", "-1"),
        (VARIANT_B, "direct", "-1"),
        (or_thm_extra, "or theorem", "-1"),
    ])
    assert_failure(check_or_elimination,
                   _line(or_thm=or_thm_extra), [], state)


@register
def test_merged_claim_shape_rejected():
    merged_short = (
        "(>[N,i0,s,+,*,i1](AnchorPeano[N,i0,s,+,*,i1])"
        "(>[v1,v2](preorder[N,*,v1,v2])(=[v1,v2])))"
    )
    assert_failure(check_or_elimination, _line(merged=merged_short),
                   [], _state())


@register
def test_row_shape_gates():
    state = _state()
    short = make_proof_line(MERGED, "main", "or elimination",
                            VARIANT_A, "main", VARIANT_B, "main")
    assert_failure(check_or_elimination, short, [], state)
    off_ns = make_proof_line(MERGED, "other", "or elimination",
                             VARIANT_A, "main", VARIANT_B, "main",
                             OR_THM, "main")
    assert_failure(check_or_elimination, off_ns, [], state)


if __name__ == "__main__":
    sys.exit(run_all_tests())
