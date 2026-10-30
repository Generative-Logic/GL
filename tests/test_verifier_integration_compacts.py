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

"""Tests for compact-first integration rows (D-503).

An integration rule's row states its compact: the compact takes the tag and
citation the rule took, and the checks run on the compact's expansion through
its binary entry. The binaries are small synthetic registries; each failure
test injects exactly ONE malformation. See ``tests/test_harness.py`` for the
registry, fixtures, and runner.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_harness import (  # noqa: E402
    register, make_state_minimal, make_proof_line, assert_failure, assert_pass,
    run_all_tests,
)
from verifier import (  # noqa: E402
    _expand_implication_compact, check_expansion,
    check_reformulation_for_integration_and,
    check_reformulation_for_integration_flattened,
)

BINARY = {
    # An and entity and its one-level integration compact.
    "andT": {"category": "and", "signature": "(andT[u_1,u_2])",
             "elements": ["(in[u_1,u_2])", "(in[u_2,u_1])"]},
    "implicationT1": {"category": "implication", "signature": "(implicationT1[u_1,u_2])",
                      "elements": ["(in[u_1,u_2])", "(in[u_2,u_1])", "(andT[u_1,u_2])"]},
    # A disjunct-shaped existence: exT(N,b,x) = exists w1: in[w1,N] and
    # existence901(N,b,x,w1); existence901 is ad hoc (always dissolved).
    "exT": {"category": "existence", "signature": "(exT[u_1,u_2,u_3])",
            "elements": ["(in[1,u_1])", "(existence901[u_1,u_2,u_3,1])"]},
    "existence901": {"category": "existence", "signature": "(existence901[u_1,u_2,u_3,u_4])",
                     "elements": ["(in[1,u_1])", "(in3[u_2,u_3,u_4,1])"]},
    # Its flat rule over the owner tokens N, b; the witnesses and the open x
    # (local 3) are bound.
    "implicationT2": {"category": "implication", "signature": "(implicationT2[u_1,u_2])",
                      "elements": ["(in[1,u_1])", "(in[2,u_1])", "(in3[u_2,3,1,2])",
                                   "(exT[u_1,u_2,3])"]},
    # An OR premise, the rule it is a premise of, and an introduction.
    "or901": {"category": "or", "signature": "(or901[u_1,u_2])",
              "elements": ["(in[u_1,u_2])", "(in[u_2,u_1])"]},
    "implicationT3": {"category": "implication", "signature": "(implicationT3[u_1,u_2])",
                      "elements": ["(or901[u_1,u_2])", "(andT[u_1,u_2])"]},
    "implicationT4": {"category": "implication", "signature": "(implicationT4[u_1,u_2])",
                      "elements": ["(in[u_1,u_2])", "(or901[u_1,u_2])"]},
    "implicationT5": {"category": "implication", "signature": "(implicationT5[u_1,u_2])",
                      "elements": ["(in[u_1,u_1])", "(or901[u_1,u_2])"]},
    # An entry whose bound placeholder reads like an actual argument.
    "implicationT6": {"category": "implication", "signature": "(implicationT6[u_1])",
                      "elements": ["(in[1,u_1])", "(in[u_1,1])"]},
}


def _state():
    state = make_state_minimal()
    state.current_gl_binary = BINARY
    return state


EXPANSION_ROW = make_proof_line("(&(in[a,b])(in[b,a]))_integration_goal", "main",
                                "expansion for integration", "(andT[a,b])_integration_goal", "main")


@register
def test_expand_compact_keeps_placeholders_apart_from_actual_arguments():
    """A placeholder ``1`` never merges with an actual argument ``1``."""
    rule = _expand_implication_compact("(implicationT6[1])", [BINARY])
    assert rule == "(>[__b1](in[__b1,1])(in[1,__b1]))", rule


@register
def test_and_compact_row_checks_its_expansion():
    """The one-level compact of an and cites the expansion-for-integration row."""
    row = make_proof_line("(implicationT1[a,b])", "main", "reformulation for integration and",
                          "(&(in[a,b])(in[b,a]))_integration_goal", "main")
    assert_pass(check_reformulation_for_integration_and, row, [EXPANSION_ROW, row], _state())


@register
def test_and_compact_row_with_swapped_arguments_rejected():
    """The compact's expansion must be the and's reformulation exactly."""
    row = make_proof_line("(implicationT1[b,a])", "main", "reformulation for integration and",
                          "(&(in[a,b])(in[b,a]))_integration_goal", "main")
    assert_failure(check_reformulation_for_integration_and, row, [EXPANSION_ROW, row], _state())


@register
def test_disjunct_flat_compact_binds_the_template_open_argument():
    """A disjunct rule binds the template's open argument at its head."""
    row = make_proof_line("(implicationT2[N,b])", "main", "reformulation for integration flattened",
                          "(exT[N,b,X])_integration_goal", "main")
    assert_pass(check_reformulation_for_integration_flattened, row, [row], _state())


@register
def test_disjunct_flat_compact_template_on_a_free_name_rejected():
    """A head binder may not be renamed onto a name the rule already uses."""
    row = make_proof_line("(implicationT2[N,b])", "main", "reformulation for integration flattened",
                          "(exT[N,b,b])_integration_goal", "main")
    assert_failure(check_reformulation_for_integration_flattened, row, [row], _state())


@register
def test_disjunct_flat_compact_other_owner_rejected():
    """Every free head argument must equal the template's."""
    row = make_proof_line("(implicationT2[N,c])", "main", "reformulation for integration flattened",
                          "(exT[N,b,X])_integration_goal", "main")
    assert_failure(check_reformulation_for_integration_flattened, row, [row], _state())


RULE_ROW = make_proof_line("(implicationT3[a,b])", "main", "reformulation for integration flattened",
                           "(andT[a,b])_integration_goal", "main")


@register
def test_introduction_compact_cites_its_rule_compact():
    """An introduction compact citing the compact of the rule it serves."""
    row = make_proof_line("(implicationT4[a,b])", "main", "expansion", "(implicationT3[a,b])", "main")
    assert_pass(check_expansion, row, [RULE_ROW, row], _state())


@register
def test_introduction_compact_of_another_disjunct_rejected():
    """The introduction's premise must be a disjunct of the cited OR premise."""
    row = make_proof_line("(implicationT5[a,b])", "main", "expansion", "(implicationT3[a,b])", "main")
    assert_failure(check_expansion, row, [RULE_ROW, row], _state())


if __name__ == "__main__":
    sys.exit(run_all_tests())
