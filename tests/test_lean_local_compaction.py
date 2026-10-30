# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt)
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Tests for the Lean writer's replay of a `compilation` row that folds an
expansion proved earlier in the same chapter into its compact
(``proof_export.lean._render_chapter_steps``, action ``definition_compact``
with a ``local_step`` dependency)."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proof_export import lean  # noqa: E402

EXPANSION = '(>[w](in[w,X])(in[w,Y]))'
COMPACT = '(implication77[X,Y])'
DEFINITIONS = {
    'implication77': {'category': 'implication', 'arity': 2,
                      'signature': '(implication77[u_1,u_2])',
                      'elements': ['(in[1,u_1])', '(in[1,u_2])']},
}


def _chapter(dependency: dict[str, object]) -> dict[str, object]:
    """Chapter with the assumed expansion at line 1 and its compaction at line 2."""
    first = {'id': 'chapter_1_line_1', 'source_line': 1,
             'source_tag': 'task formulation', 'action': 'assume',
             'namespace': 'main', 'conclusion': {'mpl': EXPANSION}, 'dependencies': []}
    second = {'id': 'chapter_1_line_2', 'source_line': 2,
              'source_tag': 'compilation', 'action': 'definition_compact',
              'namespace': 'main', 'conclusion': {'mpl': COMPACT},
              'dependencies': [dependency]}
    return {'steps': [first, second],
            'variable_types': {'X': 'set', 'Y': 'set', 'w': 'element'}}


def _render(chapter: dict[str, object]) -> tuple[list[str], list[dict[str, object]]]:
    return lean._render_chapter_steps(
        chapter, 'local_compaction', {EXPANSION: 'source_assumption'}, {},
        ambient_variables={'X', 'Y'}, binary_definitions=DEFINITIONS)


class LocalCompactionTests(unittest.TestCase):
    def test_compact_folds_the_cited_local_row(self) -> None:
        lines, manifest = _render(_chapter({
            'step': 'chapter_1_line_1', 'namespace': 'main',
            'source_kind': 'local_step', 'expression': {'mpl': EXPANSION}}))
        text = '\n'.join(lines)
        self.assertIn(f'simpa only [{lean.lean_constant("implication77")}] using (row_1)', text)
        self.assertEqual(len(manifest), 2)

    def test_local_compaction_carrying_a_theorem_reference_asserts(self) -> None:
        with self.assertRaises(AssertionError):
            _render(_chapter({
                'step': 'chapter_1_line_1', 'namespace': 'main',
                'source_kind': 'local_step', 'expression': {'mpl': EXPANSION},
                'theorem_reference': {'theorem_id': 'gauss_source_1'}}))


if __name__ == '__main__':
    unittest.main()
