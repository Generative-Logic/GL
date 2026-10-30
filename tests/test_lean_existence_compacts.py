# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt)
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Tests for the Lean writer's replay of an `expansion` row that turns a
negated existence compact into one of its implication compacts
(``proof_export.lean._existence_implication_direction``)."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proof_export import lean  # noqa: E402
from proof_export.mpl import parse_mpl  # noqa: E402


class CounterexampleWitnessTests(unittest.TestCase):
    def test_negated_or_expansion_replays_complements(self) -> None:
        source = '!(or97[x,y,z])'
        target = '(&!(=[x,y])(=[x,z]))'
        definitions = {'or97': {'category': 'or', 'arity': 3,
            'signature': '(or97[u_1,u_2,u_3])',
            'elements': ['(=[u_1,u_2])', '!(=[u_1,u_3])']}}
        first = {'id': 'chapter_1_line_1', 'source_line': 1,
                 'source_tag': 'task formulation', 'action': 'assume',
                 'namespace': 'main', 'conclusion': {'mpl': source}, 'dependencies': []}
        second = {'id': 'chapter_1_line_2', 'source_line': 2,
                  'source_tag': 'expansion', 'action': 'definition_expand',
                  'namespace': 'main', 'conclusion': {'mpl': target},
                  'dependencies': [{'step': first['id'], 'namespace': 'main',
                                    'expression': {'mpl': source}}]}
        chapter = {'steps': [first, second],
                   'variable_types': {name: 'element' for name in ('x', 'y', 'z')}}
        lines, manifest = lean._render_chapter_steps(
            chapter, 'negative_or', {source: 'source_assumption'}, {},
            ambient_variables={'x', 'y', 'z'}, binary_definitions=definitions)
        self.assertIn('not_or, Classical.not_not', '\n'.join(lines))
        self.assertEqual(len(manifest), 2)

    def test_one_shared_witness_for_three_products(self) -> None:
        source = parse_mpl('!(>[w](in[w,N])(>[](in2[w,a,s])(=[w,a])))')
        witness, bundle, introductions, negative = lean._counterexample_witness_data(source)
        self.assertEqual(witness, 'w')
        self.assertEqual(introductions, ['w', 'counterexample_premise_0', 'counterexample_premise_1'])
        self.assertFalse(negative)
        self.assertEqual(lean._conjunct_path(bundle, parse_mpl('(in[w,N])')), [1])
        self.assertEqual(lean._conjunct_path(bundle, parse_mpl('(in2[w,a,s])')), [2, 1])
        self.assertEqual(lean._conjunct_path(bundle, parse_mpl('!(=[w,a])')), [2, 2])

    def test_later_binder_and_negative_head(self) -> None:
        source = parse_mpl('!(>[](in[a,N])(>[w](in[w,N])!(=[w,a])))')
        witness, bundle, introductions, negative = lean._counterexample_witness_data(source)
        self.assertEqual(witness, 'w')
        self.assertEqual(introductions, ['counterexample_premise_0', 'w', 'counterexample_premise_1'])
        self.assertTrue(negative)
        self.assertEqual(lean._conjunct_path(bundle, parse_mpl('(=[w,a])')), [2, 2])

    def test_multiple_witnesses_and_positive_source_assert(self) -> None:
        for text in ['!(>[w,z](in[w,N])(=[w,z]))', '(>[w](in[w,N])(=[w,a]))']:
            with self.assertRaises(AssertionError):
                lean._counterexample_witness_data(parse_mpl(text))

    def test_projection_group_follows_the_source_order(self) -> None:
        # A case witness's head projected before its typing: the group keeps
        # the source row's order, so the folded witness bundle is the
        # definition's conjunction (in, then the head).
        source = '!(>[w](in[w,N])!(existence13[N,a,w]))'
        first = {'id': 'chapter_1_line_1', 'source_line': 1,
                 'source_tag': 'task formulation', 'action': 'assume',
                 'namespace': 'main', 'conclusion': {'mpl': source}, 'dependencies': []}
        steps = [first]
        for index, fact in enumerate(['(existence13[N,a,w])', '(in[w,N])'], 2):
            steps.append({'id': f'chapter_1_line_{index}', 'source_line': index,
                          'source_tag': 'disintegration', 'action': 'compound_project',
                          'namespace': 'main', 'conclusion': {'mpl': fact},
                          'dependencies': [{'step': first['id'], 'namespace': 'main',
                                            'expression': {'mpl': source}}]})
        grouped = lean._existence_projection_sources({'steps': steps})
        self.assertEqual([lean.to_mpl(projection) for projection in grouped[first['id']]],
                         ['(in[w,N])', '(existence13[N,a,w])'])

    def test_projection_outside_the_source_bundle_asserts(self) -> None:
        source = '!(>[w](in[w,N])!(existence13[N,a,w]))'
        first = {'id': 'chapter_1_line_1', 'source_line': 1,
                 'source_tag': 'task formulation', 'action': 'assume',
                 'namespace': 'main', 'conclusion': {'mpl': source}, 'dependencies': []}
        stray = {'id': 'chapter_1_line_2', 'source_line': 2,
                 'source_tag': 'disintegration', 'action': 'compound_project',
                 'namespace': 'main', 'conclusion': {'mpl': '(in2[w,a,s])'},
                 'dependencies': [{'step': first['id'], 'namespace': 'main',
                                   'expression': {'mpl': source}}]}
        with self.assertRaises(AssertionError):
            lean._existence_projection_sources({'steps': [first, stray]})

    def test_renderer_extracts_once_and_projects_three_shared_facts(self) -> None:
        source = '!(>[w](in[w,N])(>[](in2[w,a,s])(=[w,a])))'
        first = {'id': 'chapter_1_line_1', 'source_line': 1,
                 'source_tag': 'task formulation', 'action': 'assume',
                 'namespace': 'main', 'conclusion': {'mpl': source}, 'dependencies': []}
        steps = [first]
        for index, fact in enumerate(['(in[w,N])', '(in2[w,a,s])', '!(=[w,a])'], 2):
            steps.append({'id': f'chapter_1_line_{index}', 'source_line': index,
                          'source_tag': 'disintegration', 'action': 'compound_project',
                          'namespace': 'main', 'conclusion': {'mpl': fact},
                          'dependencies': [{'step': first['id'], 'namespace': 'main',
                                            'expression': {'mpl': source}}]})
        chapter = {'steps': steps, 'variable_types': {
            'N': 'set', 'a': 'element', 'w': 'element', 's': 'binary_relation'}}
        lines, manifest = lean._render_chapter_steps(
            chapter, 'counterexample', {source: 'source_assumption'}, {},
            ambient_variables={'N', 'a', 's'}, binary_definitions={})
        text = '\n'.join(lines)
        self.assertEqual(text.count('obtain ⟨w,'), 1)
        self.assertIn('counterexample_premise_0, counterexample_premise_1, counterexample_head', text)
        self.assertEqual(len(manifest), 4)


DEFINITIONS = {
    # ¬ ∀ x, (in[x,u_1]) → ¬ (in2[x,u_2,u_3])   — "some element of u_1 is a u_3-successor of u_2"
    "existence3": {"arity": 3, "category": "existence", "elements": ["(in[1,u_1])", "(in2[1,u_2,u_3])"]},
    # ∀ x, (in2[x,u_1,u_2]) → ¬ (in[x,u_3])    — right → ¬ left
    "implication1247": {"arity": 3, "category": "implication", "elements": ["(in2[1,u_1,u_2])", "!(in[1,u_3])"]},
    # ∀ x, (in[x,u_1]) → ¬ (in2[x,u_2,u_3])    — left → ¬ right
    "implication9": {"arity": 3, "category": "implication", "elements": ["(in[2,u_1])", "!(in2[2,u_2,u_3])"]},
    # ∀ x, (in[x,u_1]) → (in2[x,u_2,u_3])       — not a negated conclusion
    "implication0": {"arity": 3, "category": "implication", "elements": ["(in[1,u_1])", "(in2[1,u_2,u_3])"]},
    # ∀ x, (in[x,u_1]) → ¬ (in[x,u_2])          — unrelated to the existence
    "implication5": {"arity": 2, "category": "implication", "elements": ["(in[1,u_1])", "!(in[1,u_2])"]},
}
NEGATED_EXISTENCE = parse_mpl("!(existence3[N,v1,s])")


class ExistenceImplicationDirectionTests(unittest.TestCase):
    def test_swapped_orientation_is_recognized(self) -> None:
        self.assertEqual(
            lean._existence_implication_direction(
                NEGATED_EXISTENCE, parse_mpl("(implication1247[v1,s,N])"), DEFINITIONS,
            ),
            (["1"], True),
        )

    def test_same_orientation_is_recognized_with_its_own_placeholder(self) -> None:
        self.assertEqual(
            lean._existence_implication_direction(
                NEGATED_EXISTENCE, parse_mpl("(implication9[N,v1,s])"), DEFINITIONS,
            ),
            (["1"], False),
        )

    def test_other_shapes_take_the_ordinary_unfold(self) -> None:
        self.assertIsNone(
            lean._existence_implication_direction(
                parse_mpl("(existence3[N,v1,s])"), parse_mpl("(implication1247[v1,s,N])"), DEFINITIONS,
            )
        )
        self.assertIsNone(
            lean._existence_implication_direction(
                NEGATED_EXISTENCE, parse_mpl("(implication0[N,v1,s])"), DEFINITIONS,
            )
        )
        self.assertIsNone(
            lean._existence_implication_direction(NEGATED_EXISTENCE, parse_mpl("(in[v1,N])"), DEFINITIONS)
        )
        self.assertIsNone(
            lean._existence_implication_direction(
                NEGATED_EXISTENCE, parse_mpl("(implication1247[v1,s,N])"), None,
            )
        )

    def test_a_compact_of_another_existence_asserts(self) -> None:
        with self.assertRaises(AssertionError):
            lean._existence_implication_direction(
                NEGATED_EXISTENCE, parse_mpl("(implication5[N,v1])"), DEFINITIONS,
            )


if __name__ == "__main__":
    unittest.main()
