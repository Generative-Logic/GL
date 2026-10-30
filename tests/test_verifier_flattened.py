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

"""Tests for the ``reformulation for integration flattened`` checker.

The positive rows are the flat integration-rule variants the prover's
``emitFlatIntegrationVariants`` installs for the D1 goal
``(existence4[1,4,5,9,10])`` (FTA binary: existence4 -> remainder ->
existence3 -> and1 -> strictOrder -> preorder). Each failure test injects
exactly ONE malformation. See ``tests/test_harness.py`` for the registry,
fixtures, and runner.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_harness import (  # noqa: E402
    register, make_state_minimal, make_state_with_binaries, make_proof_line,
    assert_failure, assert_pass, run_all_tests,
)
from verifier import (  # noqa: E402
    check_reformulation_for_integration_flattened, _flat_unfold_variants,
)

TAG = "reformulation for integration flattened"
GOAL = "(existence4[1,4,5,9,10])"
GOAL_REF = GOAL + "_integration_goal"

# strictOrder intact — the D1 closure rule (maintainer's mental model).
RULE_LT = ("(>[r](in[r,1])(>[q](in[q,1])(>[t](in[t,1])(>[](in3[q,10,t,5])"
           "(>[](in3[t,r,9,4])(>[](strictOrder[1,4,r,10])"
           "(existence4[1,4,5,9,10])))))))")
# preorder intact.
RULE_LE = ("(>[r](in[r,1])(>[q](in[q,1])(>[t](in[t,1])(>[](in3[q,10,t,5])"
           "(>[](in3[t,r,9,4])(>[](preorder[1,4,r,10])(>[]!(=[r,10])"
           "(existence4[1,4,5,9,10]))))))))")
# fully flat.
RULE_FLAT = ("(>[r](in[r,1])(>[q](in[q,1])(>[t](in[t,1])(>[](in3[q,10,t,5])"
             "(>[](in3[t,r,9,4])(>[p](in[p,1])(>[](in3[r,p,10,4])(>[]!(=[r,10])"
             "(existence4[1,4,5,9,10])))))))))")
# Case B's own one-level rule — never emitted under this tag.
RULE_ONE_LEVEL = ("(>[r](in[r,1])(>[](remainder[1,4,5,9,10,r])"
                  "(existence4[1,4,5,9,10])))")


def _row(expr, ref=GOAL_REF, ns="main", ref_ns="main"):
    return make_proof_line(expr, ns, TAG, ref, ref_ns)


# ===========================================================================
#  positive rows
# ===========================================================================

@register
def test_flattened_accepts_strictorder_intact_variant():
    """The D1 closure rule verifies against the FTA binary."""
    state = make_state_with_binaries(("FTA",))
    line = _row(RULE_LT)
    assert_pass(check_reformulation_for_integration_flattened, line, [line], state)


@register
def test_flattened_accepts_preorder_intact_variant():
    """strictOrder extended one level, preorder intact, negated leaf verbatim."""
    state = make_state_with_binaries(("FTA",))
    line = _row(RULE_LE)
    assert_pass(check_reformulation_for_integration_flattened, line, [line], state)


@register
def test_flattened_accepts_fully_flat_variant():
    """Every config compound extended; four witnesses bound."""
    state = make_state_with_binaries(("FTA",))
    line = _row(RULE_FLAT)
    assert_pass(check_reformulation_for_integration_flattened, line, [line], state)


@register
def test_flattened_accepts_premise_permutation_and_renaming():
    """Premise order and witness names are free (multiset modulo renaming)."""
    state = make_state_with_binaries(("FTA",))
    permuted = ("(>[v3](in[v3,1])(>[v1](in[v1,1])(>[v2](in[v2,1])"
                "(>[](in3[v2,v1,9,4])(>[](strictOrder[1,4,v1,10])"
                "(>[](in3[v3,10,v2,5])(existence4[1,4,5,9,10])))))))")
    line = _row(permuted)
    assert_pass(check_reformulation_for_integration_flattened, line, [line], state)


@register
def test_flattened_unfold_enumerates_four_variants_for_d1():
    """The enumeration is the prover's: one-level, strictOrder intact,
    preorder intact, fully flat — in that order."""
    state = make_state_with_binaries(("FTA",))
    binaries = state.binaries_for_chapter()
    entry = None
    for b in binaries:
        if "existence4" in b:
            entry = b["existence4"]
            break
    assert entry is not None
    variants = _flat_unfold_variants(GOAL, entry, binaries)
    assert [len(v) for v in variants] == [2, 6, 7, 8], [len(v) for v in variants]
    assert any(e.startswith("(strictOrder[") for e in variants[1])
    assert any(e.startswith("(preorder[") for e in variants[2])
    assert not any(e.startswith("(preorder[") or e.startswith("(strictOrder[")
                   or e.startswith("(remainder[") for e in variants[3])


# ===========================================================================
#  failure rows — one malformation each
# ===========================================================================

@register
def test_flattened_rejects_rest_too_short():
    state = make_state_minimal()
    line = make_proof_line(RULE_LT, "main", TAG, GOAL_REF)
    assert_failure(check_reformulation_for_integration_flattened, line, [line], state)


@register
def test_flattened_rejects_namespace_mismatch():
    state = make_state_with_binaries(("FTA",))
    line = _row(RULE_LT, ref_ns="main_boundary_x")
    assert_failure(check_reformulation_for_integration_flattened, line, [line], state)


@register
def test_flattened_rejects_missing_goal_postfix():
    state = make_state_with_binaries(("FTA",))
    line = _row(RULE_LT, ref=GOAL)
    assert_failure(check_reformulation_for_integration_flattened, line, [line], state)


@register
def test_flattened_rejects_wrong_head():
    """Head is a different instantiation of the compact."""
    state = make_state_with_binaries(("FTA",))
    line = _row(RULE_LT.replace("(existence4[1,4,5,9,10])))))))",
                                "(existence4[1,4,5,9,11])))))))"))
    assert_failure(check_reformulation_for_integration_flattened, line, [line], state)


@register
def test_flattened_rejects_missing_premise():
    """strictOrder conjunct dropped — a weaker, unsound rule."""
    state = make_state_with_binaries(("FTA",))
    weaker = ("(>[r](in[r,1])(>[q](in[q,1])(>[t](in[t,1])(>[](in3[q,10,t,5])"
              "(>[](in3[t,r,9,4])(existence4[1,4,5,9,10]))))))")
    line = _row(weaker)
    assert_failure(check_reformulation_for_integration_flattened, line, [line], state)


@register
def test_flattened_rejects_extra_premise():
    """An extra atomic premise is not part of any unfolding."""
    state = make_state_with_binaries(("FTA",))
    extra = RULE_LT.replace("(>[](strictOrder[1,4,r,10])",
                            "(>[](in3[r,r,9,4])(>[](strictOrder[1,4,r,10])") + ")"
    line = _row(extra)
    assert_failure(check_reformulation_for_integration_flattened, line, [line], state)


@register
def test_flattened_rejects_intact_compound_with_its_children():
    """strictOrder kept AND its children listed — no single choice yields both."""
    state = make_state_with_binaries(("FTA",))
    both = RULE_LT.replace("(>[](strictOrder[1,4,r,10])",
                           "(>[](preorder[1,4,r,10])(>[]!(=[r,10])"
                           "(>[](strictOrder[1,4,r,10])") + "))"
    line = _row(both)
    assert_failure(check_reformulation_for_integration_flattened, line, [line], state)


@register
def test_flattened_rejects_one_level_rule():
    """Case B's own rule is never a flattened variant."""
    state = make_state_with_binaries(("FTA",))
    line = _row(RULE_ONE_LEVEL)
    assert_failure(check_reformulation_for_integration_flattened, line, [line], state)


@register
def test_flattened_rejects_witness_collapse():
    """Two witnesses merged into one name — the renaming must be injective."""
    state = make_state_with_binaries(("FTA",))
    merged = ("(>[r](in[r,1])(>[q](in[q,1])(>[](in3[q,10,q,5])"
              "(>[](in3[q,r,9,4])(>[](strictOrder[1,4,r,10])"
              "(existence4[1,4,5,9,10]))))))")
    line = _row(merged)
    assert_failure(check_reformulation_for_integration_flattened, line, [line], state)


@register
def test_flattened_rejects_free_variable_where_witness_expected():
    """A witness slot holding a free (unbound) name is not a witness."""
    state = make_state_with_binaries(("FTA",))
    free = ("(>[r](in[r,1])(>[q](in[q,1])(>[](in[9,1])(>[](in3[q,10,9,5])"
            "(>[](in3[9,r,9,4])(>[](strictOrder[1,4,r,10])"
            "(existence4[1,4,5,9,10])))))))")
    line = _row(free)
    assert_failure(check_reformulation_for_integration_flattened, line, [line], state)


@register
def test_flattened_rejects_unknown_compact():
    state = make_state_with_binaries(("FTA",))
    line = _row(RULE_LT, ref="(nosuchop[1,2])_integration_goal")
    assert_failure(check_reformulation_for_integration_flattened, line, [line], state)


# The hasPrimeFactorization shape: the inner existence binds P by
# seq(k,P), and seq(k,P) is also the first conjunct of the bundle it binds.
SHARED = {
    "hasPF": {"category": "existence", "signature": "(hasPF[u_1])",
              "elements": ["(in[1,u_1])", "(existence36[u_1,1])"]},
    "existence36": {"category": "existence", "signature": "(existence36[u_1,u_2])",
                    "elements": ["(seq[u_2,1])", "(bundle[u_2,1])"]},
    "bundle": {"category": "and", "signature": "(bundle[u_1,u_2])",
               "elements": ["(seq[u_1,u_2])", "(ord[u_2])"]},
}
SHARED_RULE = "(>[k](in[k,N])(>[P](seq[k,P])(>[](ord[P])(hasPF[N]))))"


def _shared_state():
    state = make_state_minimal()
    state.current_gl_binary = SHARED
    return state


@register
def test_flattened_unfold_names_a_shared_premise_once():
    """Extending the bundle adds its seq conjunct no second time: bundle
    intact, bundle extended (the ad-hoc inner existence always dissolves)."""
    variants = _flat_unfold_variants("(hasPF[N])", SHARED["hasPF"], [SHARED])
    assert [len(v) for v in variants] == [3, 3], variants
    assert all(len(set(v)) == len(v) for v in variants), variants
    assert any(e.startswith("(bundle[") for e in variants[0])
    assert any(e.startswith("(ord[") for e in variants[1])


@register
def test_flattened_accepts_shared_premise_once():
    line = _row(SHARED_RULE, ref="(hasPF[N])_integration_goal")
    assert_pass(check_reformulation_for_integration_flattened, line, [line], _shared_state())


@register
def test_flattened_rejects_shared_premise_twice():
    """The rule the prover no longer emits: seq(k,P) named twice."""
    twice = "(>[k](in[k,N])(>[P](seq[k,P])(>[](seq[k,P])(>[](ord[P])(hasPF[N])))))"
    line = _row(twice, ref="(hasPF[N])_integration_goal")
    assert_failure(check_reformulation_for_integration_flattened, line, [line], _shared_state())


if __name__ == "__main__":
    sys.exit(run_all_tests())
