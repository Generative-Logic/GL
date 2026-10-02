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

"""Tests for chapter coverage: every manifest theorem must have its chapters.

A missing chapter, a stray or misnumbered chapter, an unknown proof method
and a missing primary manifest are failures; the command line exits 1 on
any failure. See ``tests/test_harness.py`` for fixtures and runner.
"""

import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_harness import (  # noqa: E402
    register, make_state_minimal, run_all_tests,
)
from verifier import (  # noqa: E402
    build_chapter_theorem_map, check_chapter_coverage, expected_chapter_files,
    run_verifier, get_totals,
)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MANIFEST = [
    ("(T0)", "single chapter proof", "-1"),
    ("(T1)", "induction", "v1"),
    ("(T2)", "or theorem", "-1"),
]
_FILES = ["0_single_chapter_proof.txt", "1_induction_typing.txt", "2_check_zero.txt",
          "3_check_induction_condition.txt", "4_or_theorem.txt"]


def _coverage(state):
    """(success, failure) of the chapter-coverage counter (0, 0 if absent)."""
    ctr = state.tag_counters.get("chapter coverage")
    return (ctr.success, ctr.failure) if ctr is not None else (0, 0)


@register
def test_expected_chapter_files_follow_the_producer_naming():
    names = [name for name, _, _ in expected_chapter_files(_MANIFEST)]
    assert names == _FILES, names
    unknown = expected_chapter_files([("(T9)", "mystery method", "-1")])
    assert unknown == [("0_unknown_mystery_method.txt", ("(T9)", "mystery method", "-1"), False)]


@register
def test_complete_chapters_cover_the_manifest():
    state = make_state_minimal()
    check_chapter_coverage(_FILES, _MANIFEST, state)
    assert _coverage(state) == (5, 0), _coverage(state)
    mapping = build_chapter_theorem_map(_FILES, _MANIFEST)
    assert mapping["2_check_zero.txt"] == _MANIFEST[1]
    assert mapping["4_or_theorem.txt"] == _MANIFEST[2]


@register
def test_missing_chapters_fail():
    """A missing tail, a missing middle chapter and no chapter at all."""
    for present, failures in ((_FILES[:-1], 1), (_FILES[:2] + _FILES[3:], 1), ([], 5)):
        state = make_state_minimal()
        check_chapter_coverage(present, _MANIFEST, state)
        assert _coverage(state)[1] == failures, (present, _coverage(state))
    # The mapping never shifts later chapters onto the wrong theorem.
    mapping = build_chapter_theorem_map(_FILES[:2] + _FILES[3:], _MANIFEST)
    assert mapping["4_or_theorem.txt"] == _MANIFEST[2]
    assert "2_check_zero.txt" not in mapping


@register
def test_stray_misnumbered_and_unknown_chapters_fail():
    state = make_state_minimal()
    check_chapter_coverage(_FILES + ["5_single_chapter_proof.txt"], _MANIFEST, state)
    assert _coverage(state) == (5, 1), _coverage(state)
    state = make_state_minimal()
    swapped = ["0_or_theorem.txt"] + _FILES[1:4] + ["4_single_chapter_proof.txt"]
    check_chapter_coverage(swapped, _MANIFEST, state)
    assert _coverage(state) == (3, 4), _coverage(state)
    state = make_state_minimal()
    check_chapter_coverage(["0_unknown_mystery.txt"], [("(T9)", "mystery", "-1")], state)
    assert _coverage(state)[1] >= 1, _coverage(state)


def _graph(folder, manifest_rows):
    os.makedirs(folder)
    if manifest_rows is not None:
        with open(os.path.join(folder, "global_theorem_list.txt"), "w", encoding="utf-8") as f:
            for row in manifest_rows:
                f.write("\t".join(row) + "\n")


@register
def test_a_manifest_without_chapters_fails_verification():
    """End to end: the verifier reports failures and the command line exits 1
    when every chapter is missing, or when the manifest itself is missing."""
    with tempfile.TemporaryDirectory() as tmp:
        no_chapters = os.path.join(tmp, "no_chapters")
        _graph(no_chapters, [("(>[v1](AnchorPeano[N,s,p,zero,one,two])(in[v1,N]))",
                              "single chapter proof", "-1")])
        state = run_verifier(no_chapters)
        assert get_totals(state)[1] == 1, get_totals(state)
        no_manifest = os.path.join(tmp, "no_manifest")
        _graph(no_manifest, None)
        assert get_totals(run_verifier(no_manifest))[1] == 1
        completed = subprocess.run([sys.executable, os.path.join(_REPO, "verifier.py"), no_chapters],
                                   capture_output=True, text=True)
        assert completed.returncode == 1, (completed.returncode, completed.stdout[-300:])


if __name__ == "__main__":
    sys.exit(run_all_tests())
