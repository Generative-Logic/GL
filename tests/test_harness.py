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

"""Unit-test harness for ``verifier.py``.

Layout
------
- ``TESTS`` (module-level list) — global registry of (qualified_name, callable).
- ``@register`` decorator — appends a test function under its ``module.name``.
- Fixtures — ``make_state_with_binaries``, ``make_state_with_globals``,
  ``make_state_with_externals``, ``make_state_minimal``, ``make_proof_line``,
  ``set_chapter_context``.
- Assertion helpers — ``assert_failure``, ``assert_pass``,
  ``assert_chapter_meta_fail``, ``assert_chapter_meta_pass``.
- Runner — ``run_all_tests()`` walks TESTS, returns 0 on all-pass, 1 on any
  failure.
- ``main()`` — imports every sibling ``test_verifier_*`` module (triggering
  ``@register`` side-effects), then runs the full suite.

Black-box: this module imports ``verifier`` and calls its checker functions
as-is. Zero modifications to ``verifier.py`` — invariant I-16 (verifier
sacred) is respected. Tests inject malformed ``ProofLine`` inputs and assert
the checker reports a failure.

Entry points
------------
- ``python tests/test_harness.py``                         (full suite)
- ``python tests/test_verifier_<group>.py``               (one group)
- ``main.py`` gate ``_run_verifier_unit_test_gate()``     (pipeline start)

Performance
-----------
Read-only fixture data (``gl_binaries``, ``output_indices``, ``input_indices``,
``definition_sets``, ``resolved_defsets_*``) is loaded once into a prototype
and shared by reference across tests. Tests must not mutate these fields;
only the per-test mutable fields (``tag_counters``, ``goal_reached``,
``global_theorems``, ``global_theorem_list``, ``external_theorems``,
``current_*``) are owned by each fresh state.
"""

import importlib
import os
import sys
from typing import Callable, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
#  sys.path shim
# ---------------------------------------------------------------------------
# Make the repository root importable so ``import verifier`` works regardless
# of how this file is invoked: ``python tests/test_harness.py`` (script mode),
# ``python -m tests.test_harness`` (module mode), or via subprocess from
# ``main.py``'s ``_run_verifier_unit_test_gate``.

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TESTS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# When invoked as ``python tests/test_harness.py`` this file loads as
# ``__main__``. Sibling test files do ``from tests.test_harness import
# register`` which triggers a SECOND load of this file under the canonical
# name ``tests.test_harness``; without the alias below, @register would
# populate that second copy's TESTS list rather than ours, and the runner
# would see "no tests registered". The alias makes both module names point
# at the same module object so the registry is shared.
sys.modules.setdefault("tests.test_harness", sys.modules[__name__])


# ---------------------------------------------------------------------------
#  verifier imports
# ---------------------------------------------------------------------------

import verifier  # noqa: E402  (sys.path shim must precede this import)
from verifier import (  # noqa: E402
    ProofLine, VerifierState, TagCounter, TAG_CHECKERS,
    verify_chapter,
    check_implication, check_expansion, check_disintegration,
    check_task_formulation, check_equality1, check_equality2,
    check_symmetry_of_equality, check_symmetry_of_inequality,
    check_recursion, check_theorem_tag,
    check_reformulation_for_integration_and,
    check_reformulation_for_integration_bound,
    check_reformulation_for_integration_empty,
    check_expansion_for_integration, check_premise_element,
    check_validity_name, check_anchor_handling,
    check_mirrored_from, check_reformulated_from,
    check_variable_copy, check_externally_provided_theorem,
    check_incubator_back_reformulation, check_equalize_variable,
    check_contradiction, check_or_disintegration,
    check_or_convergence, check_or_branch_proven,
    check_or_branch_assumption, check_vacuous_truth, check_or_theorem,
    check_defset_consistency,
    load_gl_binaries, load_output_indices, load_input_indices,
    load_definition_sets, build_resolved_defsets_per_tag,
)


# ---------------------------------------------------------------------------
#  Registry
# ---------------------------------------------------------------------------

# Each entry: (qualified_test_name, callable). The qualified name is
# ``module.function`` so a failure line in the report points to the exact
# file and function.
TESTS: List[Tuple[str, Callable[[], None]]] = []


def register(fn: Callable[[], None]) -> Callable[[], None]:
    """Decorator: append ``fn`` to TESTS under ``module.fn_name``."""
    TESTS.append((f"{fn.__module__}.{fn.__name__}", fn))
    return fn


# ---------------------------------------------------------------------------
#  Prototype state — read-only fields shared across tests
# ---------------------------------------------------------------------------

_STATE_PROTOTYPE: Optional[VerifierState] = None


def _build_state_prototype() -> VerifierState:
    """Load every read-only fixture field once. Subsequent fixture calls
    share these by reference (test convention: do not mutate)."""
    proto = VerifierState()
    config_dir = os.path.join(_REPO_ROOT, "files", "config")
    binaries_dir = os.path.join(_REPO_ROOT, "files", "GL_binaries")
    proto.gl_binaries = load_gl_binaries(binaries_dir)
    proto.output_indices = load_output_indices(config_dir)
    proto.input_indices = load_input_indices(config_dir)
    proto.definition_sets = load_definition_sets(config_dir)
    proto.resolved_defsets_per_tag, proto.resolved_defsets_atomic_only = \
        build_resolved_defsets_per_tag(proto.definition_sets, proto.gl_binaries)
    return proto


def _get_prototype() -> VerifierState:
    global _STATE_PROTOTYPE
    if _STATE_PROTOTYPE is None:
        _STATE_PROTOTYPE = _build_state_prototype()
    return _STATE_PROTOTYPE


# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------

def make_state_minimal() -> VerifierState:
    """Empty VerifierState — no binaries, no globals, no defsets. Used when
    the checker's decision does not depend on the global registry (e.g. when
    only the ``rest`` length or namespace string is being probed)."""
    return VerifierState()


def make_state_with_binaries(tags: Sequence[str] = ("Peano",)) -> VerifierState:
    """Fresh state with every loader bootstrapped. Read-only fields share by
    reference with the prototype; mutable fields (counters, global registries,
    current_*) are fresh per call. The first tag in ``tags`` (if present in
    ``gl_binaries``) sets ``current_gl_binary`` + ``current_resolved_defsets``;
    that matches verify_chapter's anchor-substring resolution for tests that
    bypass the dispatcher."""
    proto = _get_prototype()
    state = VerifierState()
    state.gl_binaries = proto.gl_binaries
    state.output_indices = proto.output_indices
    state.input_indices = proto.input_indices
    state.definition_sets = proto.definition_sets
    state.resolved_defsets_per_tag = proto.resolved_defsets_per_tag
    state.resolved_defsets_atomic_only = proto.resolved_defsets_atomic_only
    if tags and tags[0] in state.gl_binaries:
        state.current_gl_binary = state.gl_binaries[tags[0]]
        state.current_resolved_defsets = state.resolved_defsets_per_tag.get(tags[0])
    if state.current_resolved_defsets is None:
        state.current_resolved_defsets = state.resolved_defsets_atomic_only
    return state


def make_state_with_globals(thms: Sequence[Tuple[str, str, str]],
                            tags: Sequence[str] = ("Peano",)) -> VerifierState:
    """Fresh state with binaries plus a custom global_theorems registry."""
    state = make_state_with_binaries(tags)
    for entry in thms:
        expr, type_, ref = entry
        state.global_theorems[expr] = {"type": type_, "ref": ref}
        state.global_theorem_list.append(entry)
    return state


def make_state_with_externals(thms: Sequence[str],
                              tags: Sequence[str] = ("Peano",)) -> VerifierState:
    """Fresh state with binaries plus a custom external_theorems set."""
    state = make_state_with_binaries(tags)
    for thm in thms:
        state.external_theorems.add(thm)
    return state


def make_proof_line(expr: str, ns: str, tag: str,
                    *rest: str, line_no: int = 1) -> ProofLine:
    """Terse ProofLine constructor.

    Mirrors the field order used by ``parse_chapter_file``: expression,
    namespace, tag, then the alternating (source_expr, source_ns) pairs
    in ``rest``.
    """
    return ProofLine(
        expression=expr,
        namespace=ns,
        tag=tag,
        rest=list(rest),
        raw="\t".join([expr, ns, tag] + list(rest)),
        line_no=line_no,
    )


def set_chapter_context(state: VerifierState,
                        thm: Optional[Tuple[str, str, str]] = None,
                        chapter_type: str = "direct") -> None:
    """Mirror ``verify_chapter``'s transient context setup. Per-tag tests
    that bypass ``verify_chapter`` use this so checkers that read
    ``state.current_chapter_thm`` / ``state.current_gl_binary`` /
    ``state.current_resolved_defsets`` see the right values.
    """
    state.current_chapter_thm = thm
    state.current_chapter_type = chapter_type
    state.current_gl_binary = None
    state.current_resolved_defsets = None
    if thm is not None:
        thm_expr = thm[0]
        if "AnchorIncubator" in thm_expr:
            tag_iter = ((t, b) for t, b in state.gl_binaries.items()
                        if t.startswith("Incubator"))
        else:
            tag_iter = state.gl_binaries.items()
        for tag, binary in tag_iter:
            if f'Anchor{tag}' in thm_expr:
                state.current_gl_binary = binary
                state.current_resolved_defsets = \
                    state.resolved_defsets_per_tag.get(tag)
                break
    if state.current_resolved_defsets is None:
        state.current_resolved_defsets = state.resolved_defsets_atomic_only


# ---------------------------------------------------------------------------
#  Assertion helpers
# ---------------------------------------------------------------------------

def assert_failure(checker: Callable[..., bool], line: ProofLine,
                   chapter: List[ProofLine], state: VerifierState,
                   msg: str = "") -> None:
    """Assert ``checker(line, chapter, state) is False`` (verifier rejected)."""
    result = checker(line, chapter, state)
    assert result is False, (
        f"expected {checker.__name__} to REJECT the input (return False); "
        f"got {result!r}. {msg}".strip()
    )


def assert_pass(checker: Callable[..., bool], line: ProofLine,
                chapter: List[ProofLine], state: VerifierState,
                msg: str = "") -> None:
    """Assert ``checker(line, chapter, state) is True`` (verifier accepted).

    Used by ``tests/test_verifier_positive.py`` only — failure-mode tests
    should never use this helper.
    """
    result = checker(line, chapter, state)
    assert result is True, (
        f"expected {checker.__name__} to ACCEPT the input (return True); "
        f"got {result!r}. {msg}".strip()
    )


def assert_chapter_meta_fail(chapter_lines: List[ProofLine],
                             meta_name: str,
                             chapter_thm: Optional[Tuple[str, str, str]] = None,
                             chapter_type: str = "direct",
                             expected_failures: int = 1,
                             state: Optional[VerifierState] = None,
                             chapter_file: str = "test_chapter.txt"
                             ) -> VerifierState:
    """Run ``verify_chapter`` and assert at least ``expected_failures`` were
    recorded under counter ``meta_name``. Returns the state for follow-up
    introspection."""
    if state is None:
        state = make_state_with_binaries(("Peano",))
    verify_chapter(chapter_file, chapter_lines, chapter_type,
                   state, chapter_thm)
    if meta_name == "theorem goal reached":
        actual = state.goal_reached.failure
    else:
        ctr = state.tag_counters.get(meta_name)
        actual = ctr.failure if ctr is not None else 0
    assert actual >= expected_failures, (
        f"expected at least {expected_failures} failure(s) under "
        f"{meta_name!r}; got {actual}. counters: "
        f"{ {k: (c.success, c.failure) for k, c in state.tag_counters.items()} }"
    )
    return state


def assert_chapter_meta_pass(chapter_lines: List[ProofLine],
                             meta_name: str,
                             chapter_thm: Optional[Tuple[str, str, str]] = None,
                             chapter_type: str = "direct",
                             state: Optional[VerifierState] = None,
                             chapter_file: str = "test_chapter.txt"
                             ) -> VerifierState:
    """Run ``verify_chapter`` and assert ZERO failures under counter
    ``meta_name``. Used by positive-sanity tests on well-formed chapters."""
    if state is None:
        state = make_state_with_binaries(("Peano",))
    verify_chapter(chapter_file, chapter_lines, chapter_type,
                   state, chapter_thm)
    if meta_name == "theorem goal reached":
        actual = state.goal_reached.failure
    else:
        ctr = state.tag_counters.get(meta_name)
        actual = ctr.failure if ctr is not None else 0
    assert actual == 0, (
        f"expected zero failures under {meta_name!r}; got {actual}. counters: "
        f"{ {k: (c.success, c.failure) for k, c in state.tag_counters.items()} }"
    )
    return state


# ---------------------------------------------------------------------------
#  Runner
# ---------------------------------------------------------------------------

def run_all_tests() -> int:
    """Walk TESTS. Return 0 on all-pass, 1 on any failure (or test rig error).

    Failure lines go to stderr (``[verifier-tests] FAIL {name}: {msg}``); the
    final summary line lands on stdout when all pass so the C++-style headline
    is visible in the main.py gate's captured output.
    """
    if not TESTS:
        print("[verifier-tests] no tests registered (suite empty)",
              file=sys.stderr)
        return 1
    failures = 0
    for qualified, fn in TESTS:
        try:
            fn()
        except AssertionError as e:
            print(f"[verifier-tests] FAIL {qualified}: {e}", file=sys.stderr)
            failures += 1
        except Exception as e:  # noqa: BLE001 — any uncaught is a test-rig bug
            print(f"[verifier-tests] ERROR {qualified}: "
                  f"{type(e).__name__}: {e}", file=sys.stderr)
            failures += 1
    if failures:
        print(f"[verifier-tests] {failures}/{len(TESTS)} tests failed",
              file=sys.stderr)
        return 1
    print(f"{len(TESTS)}/{len(TESTS)} Verifier unit tests passed")
    return 0


def _import_all_test_modules() -> None:
    """Import every sibling ``tests/test_verifier_*.py`` so their
    ``@register`` decorators populate TESTS."""
    for fname in sorted(os.listdir(_TESTS_DIR)):
        if not fname.startswith("test_verifier_"):
            continue
        if not fname.endswith(".py"):
            continue
        importlib.import_module(f"tests.{fname[:-3]}")


def main() -> int:
    _import_all_test_modules()
    return run_all_tests()


if __name__ == "__main__":
    sys.exit(main())
