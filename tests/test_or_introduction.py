# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Ordered OR-introduction proof contracts and explicit Lean replay."""
import copy
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from proof_export.mpl import parse_mpl
from proof_export.or_introduction import match_or_introduction
from tests.lean_toolchain import LEAN_PROJECT, lean_environment, lean_executable


def definitions():
    return {
        'pair': {'category': 'and', 'signature': '(pair[u_1,u_2,u_3,u_4])', 'arity': 4,
                 'elements': ['(=[u_1,u_2])', '(=[u_3,u_4])']},
        'choice': {'category': 'or', 'signature': '(choice[u_1,u_2,u_3,u_4,u_5])', 'arity': 5,
                   'elements': ['(in2[u_1,u_2,u_3])', '(pair[u_2,u_4,u_1,u_5])']},
        'rule': {'category': 'implication', 'signature': '(rule[u_1,u_2,u_3,u_4])', 'arity': 4,
                 'elements': ['(choice[x,y,u_1,u_2,u_3])', '(in2[x,y,u_4])']},
    }


class OrIntroductionTests(unittest.TestCase):
    source = '(rule[f,a,n,g])'
    target = '(>[p](=[p,a])(>[i](=[i,n])(choice[i,p,f,a,n])))'

    def match(self, target=None, registry=None):
        return match_or_introduction(parse_mpl(target or self.target), parse_mpl(self.source), registry or definitions())

    def test_ordered_premises_distributed_binders_and_fixed_arguments(self):
        self.assertEqual(self.match(), {'binder_groups': [['p'], ['i']], 'tree': [0, 1], 'path': [2]})
        self.assertIsNotNone(self.match('(>[p,i](=[p,a])(>[](=[i,n])(choice[i,p,f,a,n])))'))
        for invalid in [
            '(>[p,i](=[p,a])(choice[i,p,f,a,n]))',
            '(>[i](=[i,n])(>[p](=[p,a])(choice[i,p,f,a,n])))',
            '(>[p](=[p,n])(>[i](=[i,n])(choice[i,p,f,a,n])))',
            '(>[p](=[p,a])(>[i]!(=[i,n])(choice[i,p,f,a,n])))',
            '(>[p](=[p,a])(>[i](=[i,n])(choice[i,p,g,a,n])))',
            '(>[p](=[p,a])(>[p](=[p,n])(choice[p,p,f,a,n])))',
            '(>[p](=[p,a])(>[i](=[i,n])(choice[i,a,f,a,n])))',
            '(>[](=[p,a])(>[p,i](=[i,n])(choice[i,p,f,a,n])))',
        ]:
            with self.subTest(invalid=invalid):
                self.assertIsNone(self.match(invalid))

    def test_nested_signed_and_repeated_leaves_and_typed_registry(self):
        registry = definitions()
        registry['outer'] = {'category': 'and', 'signature': '(outer[u_1,u_2,u_3,u_4])', 'arity': 4,
                             'elements': ['(pair[u_1,u_2,u_3,u_4])', '!(pair[u_1,u_2,u_3,u_4])']}
        registry['choice']['elements'][1] = '(outer[u_2,u_4,u_1,u_5])'
        target = '(>[p](=[p,a])(>[i](=[i,n])(>[]!(pair[p,a,i,n])(choice[i,p,f,a,n]))))'
        self.assertEqual(self.match(target, registry)['tree'], [[0, 1], 2])
        typed = copy.deepcopy(registry)
        for entry in typed.values():
            entry['signature'] = ['element'] * entry['arity']
        self.assertEqual(self.match(target, typed), self.match(target, registry))
        registry['pair']['elements'][1] = '(=[u_1,u_2])'
        repeated = '(>[p,i](=[p,a])(>[](=[p,a])(>[]!(pair[p,a,i,n])(choice[i,p,f,a,n]))))'
        self.assertIsNotNone(self.match(repeated, registry))

    def test_verifier_scope_source_and_exact_conjuncts(self):
        from tests.test_harness import make_proof_line, make_state_minimal
        from verifier import check_expansion
        state = make_state_minimal()
        state.current_gl_binary = definitions()
        origin = make_proof_line(self.source, 'main', 'task formulation')
        row = make_proof_line(self.target, 'main', 'expansion', self.source, 'main')
        self.assertTrue(check_expansion(row, [origin, row], state))
        self.assertFalse(check_expansion(row, [row], state))
        row.namespace = 'main_boundary_case'
        self.assertFalse(check_expansion(row, [origin, row], state))

    def test_certificate_and_lean_kernel(self):
        from proof_export.certificate import _chapter_certificate
        from proof_export.graph import Chapter, ProofRow, Dependency, parse_scoped_expression
        from proof_export.lean import _render_chapter_steps
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix='or_introduction_') as directory:
            source = parse_scoped_expression(self.source)
            rows = (ProofRow(1, source, 'main', 'task formulation', ()),
                    ProofRow(2, parse_scoped_expression(self.target), 'main', 'expansion', (Dependency(source, 'main'),)))
            path = Path(directory)/'1_single_chapter_proof.txt'
            path.write_text(f'{self.source}\tmain\ttask formulation\n{self.target}\tmain\texpansion\t{self.source}\tmain\n', encoding='utf-8')
            signatures = {'=': ('element', 'element'), 'in2': ('element', 'element', 'binary_relation'),
                          'pair': ('element',)*4,
                          'choice': ('element', 'element', 'binary_relation', 'element', 'element'),
                          'rule': ('binary_relation', 'element', 'element', 'binary_relation')}
            certificate = _chapter_certificate(Chapter(path, 1, 'single_chapter_proof', rows),
                signatures, {}, {}, binary_entries=definitions())
            self.assertEqual(certificate['steps'][-1]['action'], 'structural_or_introduce')
            lines, _ = _render_chapter_steps(certificate, 'introduction', {self.source: 'source'},
                {name: name for name in ('f', 'a', 'n', 'g')}, ambient_variables={'f', 'a', 'n', 'g'},
                binary_definitions=definitions())
            lean = Path(directory)/'Introduction.lean'
            lean.write_text('def gl_pair {α : Type} (p a i n : α) : Prop := p=a ∧ i=n\n'
                'def gl_choice {α : Type} (i p : α) (f : α → α → Prop) (a n : α) : Prop := f i p ∨ gl_pair p a i n\n'
                'def gl_rule {α : Type} (f : α → α → Prop) (a n : α) (g : α → α → Prop) : Prop := ∀ x y, gl_choice x y f a n → g x y\n'
                'example {α : Type} (f g : α → α → Prop) (a n : α) (source : gl_rule f a n g) : '
                '(∀ p, p=a → ∀ i, i=n → gl_choice i p f a n) := by\n' + '\n'.join(lines) + '\n  exact row_2\n', encoding='utf-8')
            result = subprocess.run([lean_executable(), str(lean)], cwd=LEAN_PROJECT,
                env=lean_environment(), capture_output=True, text=True, encoding='utf-8', timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_kernel_replays_introduction_composition_and_shared_copied_witness(self):
        from proof_export.certificate import _chapter_certificate
        from proof_export.graph import Chapter, ProofRow, Dependency, parse_scoped_expression
        from proof_export.lean import _chapter_replacements, _render_chapter_steps
        consumer = '(>[x,y](choice[x,y,f,a,n])(in2[x,y,g]))'
        composed = '(>[p](=[p,a])(>[j](=[j,n])(in2[j,p,g])))'
        texts = [self.source, self.target, consumer, composed, '(=[a,a_copy])']
        expressions = [parse_scoped_expression(text) for text in texts]
        rows = (
            ProofRow(1, expressions[0], 'main', 'task formulation', ()),
            ProofRow(2, expressions[1], 'main', 'expansion', (Dependency(expressions[0], 'main'),)),
            ProofRow(3, expressions[2], 'main', 'expansion', (Dependency(expressions[0], 'main'),)),
            ProofRow(4, expressions[3], 'main', 'implication composition',
                     (Dependency(expressions[1], 'main'), Dependency(expressions[2], 'main'))),
            ProofRow(5, expressions[4], 'main', 'variable copy', ()),
        )
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix='endpoint_witness_') as directory:
            path = Path(directory)/'1_single_chapter_proof.txt'
            from proof_export.mpl import to_mpl
            path.write_text('\n'.join('\t'.join([to_mpl(row.conclusion.expression), row.namespace, row.tag,
                *(part for dependency in row.dependencies for part in (to_mpl(dependency.expression), dependency.namespace))])
                for row in rows), encoding='utf-8')
            signatures = {'=': ('element', 'element'), 'in2': ('element', 'element', 'binary_relation'),
                          'pair': ('element',)*4,
                          'choice': ('element', 'element', 'binary_relation', 'element', 'element'),
                          'rule': ('binary_relation', 'element', 'element', 'binary_relation')}
            certificate = _chapter_certificate(Chapter(path, 1, 'single_chapter_proof', rows),
                signatures, {}, {}, binary_entries=definitions())
            replacements = _chapter_replacements(certificate)
            replacements.update({name: name for name in ('f', 'a', 'n', 'g')})
            lines, _ = _render_chapter_steps(certificate, 'endpoint', {self.source: 'source'}, replacements,
                ambient_variables={'f', 'a', 'n', 'g'}, binary_definitions=definitions())
            lean = Path(directory)/'Endpoint.lean'
            lean.write_text('def gl_pair {α : Type} (p a i n : α) : Prop := p=a ∧ i=n\n'
                'def gl_choice {α : Type} (i p : α) (f : α → α → Prop) (a n : α) : Prop := f i p ∨ gl_pair p a i n\n'
                'def gl_rule {α : Type} (f : α → α → Prop) (a n : α) (g : α → α → Prop) : Prop := ∀ x y, gl_choice x y f a n → g x y\n'
                'example {α : Type} (f g : α → α → Prop) (N : α → Prop) (a n i : α) '
                '(source : gl_rule f a n g) (typed : N a) (endpoint : i=n) : ∃ p, N p ∧ g i p := by\n'
                + '\n'.join(lines) + '\n  exact ⟨a, typed, row_4 a row_5.symm i endpoint⟩\n', encoding='utf-8')
            result = subprocess.run([lean_executable(), str(lean)], cwd=LEAN_PROJECT,
                env=lean_environment(), capture_output=True, text=True, encoding='utf-8', timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
