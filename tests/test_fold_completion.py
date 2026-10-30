# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt)
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.


"""Fold argument contracts, configuration restrictions, and readable notation."""
import json
from pathlib import Path
import re
import unittest
from unittest.mock import patch

import analyze_incubator as incubator
from configuration_reader import configuration_reader
from proof_export.mpl import iter_atoms, parse_mpl
import visu_helpers as visu

ROOT = Path(__file__).resolve().parents[1]


class FoldContractTests(unittest.TestCase):
    def test_all_eight_configurations_agree_on_slots_and_types(self):
        # FTA and FTA2 are one AnchorFTA batch, so ConfigFTA2.json is gone.
        paths = [p for p in (ROOT / 'files/config').glob('Config*.json')
                 if 'fold' in json.loads(p.read_text(encoding='utf-8'))]
        self.assertEqual(len(paths), 8)
        for path in paths:
            with self.subTest(path=path.name):
                data = json.loads(path.read_text(encoding='utf-8'))['fold']
                config = configuration_reader(path)
                self.assertEqual(data['arity'], 8)
                self.assertEqual(data['short_mpl'], '(fold[N,s,+,op,f,n,m,p])')
                self.assertEqual(data['definition_sets']['3'], data['definition_sets']['4'])
                self.assertEqual(data['definition_sets']['4'], ['P(x(1)(x(1)(1)))', False])
                self.assertEqual(config.data['fold'].indices_output_args, [7])
                self.assertEqual(config.data['fold'].indices_input_args, [6])

    def test_definition_separates_interval_and_accumulation(self):
        definition = ''.join((ROOT / 'files/definitions/fold.mpl').read_text().split())
        atoms = list(iter_atoms(parse_mpl(definition)))
        self.assertEqual([a.arguments for a in atoms if a.head == 'interval'], [('N', '+', 'n', 'm', 'M')])
        self.assertEqual([a.arguments for a in atoms if a.head == 'in3'], [('r', 't', 'q', 'op')])
        self.assertIn(('i', 'j', 's'), [a.arguments for a in atoms if a.head == 'in2'])
        self.assertIn(('n', 'q', 'f'), [a.arguments for a in atoms if a.head == 'in2'])
        self.assertIn(('m', 'p', 'g'), [a.arguments for a in atoms if a.head == 'in2'])

    def test_gauss_filters_preserve_addition_only_and_argument_restrictions(self):
        for tag in ('Gauss', 'IncubatorGauss1', 'IncubatorGauss2'):
            data = json.loads((ROOT / f'files/config/Config{tag}.json').read_text())
            patterns = [re.compile(p) for p in data['patterns_to_exclude']]
            base = ['1', '3', '4', '4', '8', '2', '9', '10']
            def excluded(arguments):
                text = '(fold[' + ','.join(arguments) + '])'
                return any(p.search(text) for p in patterns)
            self.assertFalse(excluded(base), tag)
            for slot, value in ((1, '8'), (2, '5'), (3, '5'), (4, '3'), (5, '6')):
                changed = base.copy(); changed[slot] = value
                self.assertTrue(excluded(changed), (tag, slot))

    def test_main_gauss_budget_does_not_admit_four_old_binary_slots(self):
        data = json.loads((ROOT / 'files/config/ConfigGauss.json').read_text())
        self.assertEqual(data['parameters']['max_values_for_uncomb_def_sets']['P(x(1)(x(1)(1)))'], 4)
        guard = re.compile(data['patterns_to_exclude'][-1])
        atoms = ['(in3[9,10,11,4])', '(in3[12,13,14,5])',
                 '(interval[1,4,2,9,15])', '(interval[1,4,2,10,16])']
        self.assertFalse(guard.search(''.join(atoms[:3])))
        self.assertTrue(guard.search(''.join(atoms)))
        self.assertFalse(guard.search('(fold[1,3,4,4,8,2,9,10])' + ''.join(atoms[:2])))


    def test_batch_patterns_preserve_the_previous_set_parameter_scope(self):
        examples = {
            'Gauss': (['(interval[1,4,2,9,1])', '(fXY[9,1,1])',
                       '(limitSet[1,4,1,9,10])', '(limitSet[1,4,9,10,1])',
                       '(limitSet[1,4,9,10,9])'], ['(interval[1,4,2,9,10])',
                       '(fXY[9,10,1])', '(limitSet[1,4,9,10,11])']),
            'IncubatorGauss3': (['(interval[1,4,2,6,1])'], ['(interval[1,4,2,6,10])']),
            'IncubatorPeano1': (['(fXY[3,1,1])'], ['(fXY[3,9,1])']),
            'IncubatorPeano2': (['(fXY[8,1,1])'], ['(fXY[8,9,1])']),
        }
        for tag, (rejected, accepted) in examples.items():
            data = json.loads((ROOT / f'files/config/Config{tag}.json').read_text())
            patterns = [re.compile(p) for p in data['patterns_to_exclude']]
            for atom in rejected:
                self.assertTrue(any(p.search(atom) for p in patterns), (tag, atom))
            for atom in accepted:
                self.assertFalse(any(p.search(atom) for p in patterns), (tag, atom))


class FoldReadableTests(unittest.TestCase):
    def setUp(self):
        self.readable = visu.ReadableForm({}, {})

    def test_sum_and_product_with_identity_and_other_functions(self):
        for operation, glyph in (('+', '∑'), ('*', '∏')):
            for function in ('id', 'f'):
                with self.subTest(operation=operation, function=function):
                    atom = f'(fold[N,s,+,{operation},{function},2,4,p])'
                    body = 'i' if function == 'id' else 'f(i)'
                    self.assertEqual(self.readable.reading(atom, as_html=False),
                                     f'READS: {glyph}(i=2..4) {body} = p')
                    page = self.readable.reading(atom)
                    self.assertIn(glyph, page)
                    self.assertNotIn('∑' if glyph == '∏' else '∏', page)
                    self.assertIn('i = 2', page)
                    self.assertIn('>4<', page)

    def test_other_operations_and_traversals_remain_explicit(self):
        for atom in ('(fold[N,s,+,op,f,n,m,p])', '(fold[N,s,*,*,id,n,m,p])',
                     '(fold[N,next,+,*,id,n,m,p])'):
            result = self.readable.reading(atom, as_html=False)
            self.assertTrue(result.startswith('READS: fold('))
            self.assertNotIn('∑', result)
            self.assertNotIn('∏', result)

    def test_nested_product_keeps_function_bounds_and_parentheses(self):
        expression = '(>[f](sequence[N,+,n,m,f])(>[p](fold[N,s,+,*,f,n,m,p])(>[r](in3[p,k,r,+])(=[r,q]))))'
        caption = self.readable.theorem(expression, as_html=False)
        self.assertEqual(caption, 'from f: [n, m] → ℕ follows (∏(i=n..m) f(i)) + k = q')
        self.assertIn('rf-bigparen', self.readable.theorem(expression))
        self.assertEqual(self.readable.theorem('(>[p](fold[N,s,+,*,f,n,m,p])(>[r](in3[p,k,r,+])(=[r,q])))',
                                               as_html=False),
                         'from ∏(i=n..m) f(i) = p follows p + k = q')

    def test_negated_product_is_an_inequality(self):
        self.assertEqual(self.readable.reading('!(fold[N,s,+,*,id,n,m,p])', as_html=False),
                         'READS: ∏(i=n..m) i ≠ p')

    def test_seven_arguments_are_not_silently_reinterpreted(self):
        with self.assertRaises(AssertionError):
            self.readable.reading('(fold[N,s,+,id,n,m,p])')


class FoldIncubatorTests(unittest.TestCase):
    def test_nonzero_interval_distinguishes_sum_and_product(self):
        with patch.dict(incubator.VAL, {'20': 9, '21': 24}):
            self.assertTrue(incubator.is_true_fact('(fold[1,3,4,4,8,7,10,20])'))
            self.assertTrue(incubator.is_true_fact('(fold[1,3,4,5,8,7,10,21])'))
            self.assertFalse(incubator.is_true_fact('(fold[1,3,4,5,8,7,10,20])'))
            self.assertEqual(incubator.human_arith('(fold[1,3,4,5,8,7,10,21])'), 'prod(2..4)=24')

    def test_singleton_and_zero_factor(self):
        self.assertTrue(incubator.is_true_fact('(fold[1,3,4,5,8,7,7,7])'))
        self.assertTrue(incubator.is_true_fact('(fold[1,3,4,5,8,2,10,2])'))

    def test_unrecognized_traversal_function_or_empty_interval_is_unknown(self):
        for atom in ('(fold[1,3,5,5,8,7,10,2])', '(fold[1,8,4,5,8,7,10,2])',
                     '(fold[1,3,4,5,3,7,10,2])', '(fold[1,3,4,5,8,10,7,2])'):
            self.assertIsNone(incubator.is_true_fact(atom))


if __name__ == '__main__':
    unittest.main()
