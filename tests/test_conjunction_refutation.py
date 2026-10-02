# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschraenkt).
# Dual-licensed under AGPLv3 and a commercial license.
# See https://generative-logic.com/license. Contributions require CLA.

"""Exact structural refutation, provenance and Lean replay tests."""

import copy
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from proof_export.conjunction_refutation import (
    build_conjunction_refutation_plan, is_conjunction_refutation_expansion)
from proof_export.mpl import parse_mpl
from tests.lean_toolchain import LEAN_PROJECT, lean_environment, lean_executable


def definitions():
    return {
        'pair': {'category': 'and', 'signature': '(pair[u_1,u_2,u_3,u_4])', 'arity': 4,
                 'elements': ['(=[u_1,u_2])', '(=[u_3,u_4])']},
        'typing': {'category': 'and', 'signature': '(typing[u_1,u_2,u_3,u_4])', 'arity': 4,
                   'elements': ['(in[u_1,u_2])', '(in[u_3,u_4])']},
        'append': {'category': 'or', 'signature': '(append[u_1,u_2,u_3,u_4,u_5])', 'arity': 5,
                   'elements': ['(in2[u_1,u_2,u_3])', '(pair[u_1,u_4,u_2,u_5])']},
        'choice': {'category': 'or', 'signature': '(choice[u_1,u_2,u_3,u_4])', 'arity': 4,
                   'elements': ['(in[u_1,u_2])', '(typing[u_1,u_2,u_3,u_4])']},
    }


class ConjunctionRefutationTests(unittest.TestCase):
    def plan(self, target, *sources, registry=None):
        return build_conjunction_refutation_plan(parse_mpl(target),
            [parse_mpl(source) for source in sources], registry or definitions())

    def test_direct_new_pair_and_complete_fixed_arguments(self):
        result = self.plan('(>[]!(=[i,n])!(pair[i,n,p,a]))', '(append[i,p,f,n,a])')
        self.assertEqual(result['path'], [1])
        self.assertEqual(result['unfolding'], ['pair'])
        for target in ['(>[]!(=[i,n])!(pair[i,n,q,a]))',
                       '(>[p]!(=[i,n])!(pair[i,n,p,a]))',
                       '(>[]!(=[p,n])!(pair[i,n,p,a]))']:
            with self.assertRaises(AssertionError):
                self.plan(target, '(append[i,p,f,n,a])')

    def test_grouped_typing_and_head_match(self):
        reverse = '(>[x,y]!(typing[x,I,y,N])!(in2[x,y,f]))'
        result = self.plan('(>[]!(in[i,I])!(typing[i,I,p,N]))', '(append[i,p,f,n,a])', reverse)
        self.assertEqual(result['path'], [1])
        for invalid in ['(>[x,y,z]!(typing[x,I,y,N])!(in2[x,y,f]))',
                        '(>[x,y]!(typing[x,I,y,N])!(in2[x,y,g]))',
                        '(>[x,y]!(typing[x,I,y,N])(in2[x,y,f]))',
                        '(>[x]!(typing[x,I,x,N])!(in2[x,x,f]))']:
            with self.assertRaises(AssertionError):
                self.plan('(>[]!(in[i,I])!(typing[i,I,p,N]))', '(append[i,p,f,n,a])', invalid)
        with self.assertRaises(AssertionError):
            self.plan('(>[]!(in[i,I])!(typing[i,I,p,N]))', '(append[i,p,f,n,a])', reverse, reverse)

    def test_nested_signed_members_and_typed_registry(self):
        registry = definitions()
        registry['typing']['elements'][1] = '!(in[u_3,u_4])'
        registry['outer'] = {'category': 'and', 'signature': '(outer[u_1,u_2,u_3,u_4])', 'arity': 4,
                             'elements': ['(typing[u_1,u_2,u_3,u_4])', '(in[u_1,u_2])']}
        registry['choice']['elements'][1] = '(outer[u_1,u_2,u_3,u_4])'
        target = '(>[](in[p,N])!(outer[i,I,p,N]))'
        result = self.plan(target, '(choice[i,I,p,N])', registry=registry)
        self.assertEqual(result, {'path': [1, 2], 'unfolding': ['outer', 'typing'], 'negative_member': True})
        typed = copy.deepcopy(registry)
        for entry in typed.values():
            entry['signature'] = ['element'] * entry['arity']
        self.assertEqual(self.plan(target, '(choice[i,I,p,N])', registry=typed), result)
        self.assertFalse(is_conjunction_refutation_expansion(parse_mpl('(in[i,I])'), [], registry))

    def test_verifier_sources_and_scope_are_exact(self):
        from tests.test_harness import make_proof_line, make_state_minimal
        from verifier import check_expansion
        state = make_state_minimal()
        state.current_gl_binary = definitions()
        source = '(append[i,p,f,n,a])'
        target = '(>[]!(=[i,n])!(pair[i,n,p,a]))'
        origin = make_proof_line(source, 'main', 'task formulation')
        row = make_proof_line(target, 'main', 'expansion', source, 'main')
        self.assertTrue(check_expansion(row, [origin, row], state))
        self.assertFalse(check_expansion(row, [row], state))
        wrong_scope = make_proof_line(target, 'main_boundary_other', 'expansion', source, 'main_boundary_case')
        self.assertFalse(check_expansion(wrong_scope, [origin, wrong_scope], state))
        malformed = make_proof_line('(>[]!(=[q,n])!(pair[i,n,p,a]))', 'main', 'expansion', source, 'main')
        self.assertFalse(check_expansion(malformed, [origin, malformed], state))
        reverse = '(>[x,y]!(typing[x,I,y,N])!(in2[x,y,f]))'
        indirect = '(>[]!(in[i,I])!(typing[i,I,p,N]))'
        child = 'main_boundary_case'
        reverse_row = make_proof_line(reverse, child, 'task formulation')
        selected = make_proof_line(indirect, child, 'expansion', source, 'main', reverse, child)
        self.assertTrue(check_expansion(selected, [origin, reverse_row, selected], state))
        for result_scope, source_scope in [('main', child),
                                           (child, 'main_boundary_other')]:
            unsupported = make_proof_line(indirect, result_scope, 'expansion',
                                         source, 'main', reverse, source_scope)
            cited = make_proof_line(reverse, source_scope, 'task formulation')
            self.assertFalse(check_expansion(unsupported, [origin, cited, unsupported], state))
        self.assertFalse(check_expansion(selected, [origin, selected], state))

    def test_guarded_support_binders_signs_and_exact_premises(self):
        source = '(>[i,p]!(typing[i,I,p,N])!(in2[i,p,f]))'
        first = '(>[x,y]!(in[x,I])(>[](in[y,N])!(typing[x,I,y,N])))'
        second = '(>[x,y](in[x,I])(>[]!(in[y,N])!(typing[x,I,y,N])))'
        self.assertEqual(self.plan(first, source)['selected'], 0)
        self.assertEqual(self.plan(second, source)['selected'], 1)
        staged = '(>[x]!(in[x,I])(>[y](in[y,N])!(typing[x,I,y,N])))'
        self.assertEqual(self.plan(staged, source)['binder_counts'], [1, 1])
        with self.assertRaises(AssertionError):
            self.plan('(>[x]!(in[y,I])(>[y](in[x,N])!(typing[y,I,x,N])))', source)
        for invalid in [
            '(>[x,y]!(in[x,I])!(typing[x,I,y,N]))',
            '(>[x,y]!(in[x,I])(>[]!(in[y,N])!(typing[x,I,y,N])))',
            '(>[x,y]!(in[x,J])(>[](in[y,N])!(typing[x,I,y,N])))',
            '(>[x,y,z]!(in[x,I])(>[](in[y,N])!(typing[x,I,y,N])))',
            '(>[x]!(in[x,I])(>[](in[x,N])!(typing[x,I,x,N])))',
        ]:
            with self.subTest(invalid=invalid), self.assertRaises(AssertionError):
                self.plan(invalid, source)
        with self.assertRaises(AssertionError):
            self.plan(first, '(>[i,p](typing[i,I,p,N])!(in2[i,p,f]))')
        registry = definitions()
        registry['typing']['elements'][1] = '!(in[u_3,u_4])'
        signed = '(>[x,y](in[x,I])(>[](in[y,N])!(typing[x,I,y,N])))'
        self.assertTrue(self.plan(signed, source, registry=registry)['negative_member'])

    def test_guarded_support_does_not_unfold_nested_parts(self):
        registry = definitions()
        registry['outer'] = {'category': 'and', 'signature': '(outer[u_1,u_2,u_3,u_4])',
                             'arity': 4, 'elements': ['(typing[u_1,u_2,u_3,u_4])', '(in[u_1,u_2])']}
        source = '(>[i,p]!(outer[i,I,p,N])!(in2[i,p,f]))'
        direct = '(>[i,p]!(typing[i,I,p,N])(>[](in[i,I])!(outer[i,I,p,N])))'
        plan = self.plan(direct, source, registry=registry)
        self.assertEqual(plan['unfolding'], ['outer'])
        recursive = '(>[i,p]!(in[p,N])(>[](in[i,I])!(outer[i,I,p,N])))'
        with self.assertRaises(AssertionError):
            self.plan(recursive, source, registry=registry)

    def test_guarded_verifier_requires_the_exact_owner_and_scope(self):
        from verifier import check_expansion
        from tests.test_harness import make_proof_line, make_state_minimal
        registry = definitions()
        source = '(>[i,p]!(typing[i,I,p,N])!(in2[i,p,f]))'
        target = '(>[i,p]!(in[i,I])(>[](in[p,N])!(typing[i,I,p,N])))'
        state = make_state_minimal(); state.current_gl_binary = registry
        origin = make_proof_line(source, 'main', 'task formulation')
        row = make_proof_line(target, 'main', 'expansion', source, 'main')
        self.assertTrue(check_expansion(row, [origin, row], state))
        self.assertFalse(check_expansion(row, [row], state))
        wrong = make_proof_line(target, 'main_boundary_left', 'expansion', source, 'main_boundary_right')
        other = make_proof_line(source, 'main_boundary_right', 'task formulation')
        self.assertFalse(check_expansion(wrong, [other, wrong], state))

    def test_guarded_certificate_and_lean_quantified_replay(self):
        from proof_export.certificate import _chapter_certificate
        from proof_export.graph import Chapter, ProofRow, Dependency, parse_scoped_expression
        from proof_export.lean import _render_chapter_steps
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix='guarded_support_') as directory:
            examples = []
            for selected in (0, 1):
                registry = definitions()
                source = '(>[i,p]!(typing[i,I,p,N])!(in2[i,p,f]))'
                target = ('(>[i,p]!(in[i,I])(>[](in[p,N])!(typing[i,I,p,N])))' if selected == 0
                          else '(>[i](in[i,I])(>[p]!(in[p,N])!(typing[i,I,p,N])))')
                atom = parse_scoped_expression(source)
                rows = (ProofRow(1, atom, 'main', 'task formulation', ()),
                        ProofRow(2, parse_scoped_expression(target), 'main', 'expansion',
                                 (Dependency(atom, 'main'),)))
                file = Path(directory)/'1_single_chapter_proof.txt'
                file.write_text(f'{source}\tmain\ttask formulation\n{target}\tmain\texpansion\t{source}\tmain\n', encoding='utf-8')
                signatures = {'in': ('element', 'set'), 'in2': ('element', 'element', 'binary_relation'),
                              'typing': ('element', 'set', 'element', 'set')}
                certificate = _chapter_certificate(Chapter(file, 1, 'single_chapter_proof', rows),
                    signatures, {}, {}, binary_entries=registry)
                self.assertEqual(certificate['steps'][-1]['action'], 'conjunction_refute')
                lines, manifest = _render_chapter_steps(certificate, 'guarded', {source: 'source'},
                    {name: name for name in ('I','N','f')}, ambient_variables={'I','N','f'},
                    binary_definitions=registry)
                self.assertEqual(len(manifest), 2)
                statement = ('∀ i p, ¬ I i → N p → ¬ gl_typing i I p N' if selected == 0
                             else '∀ i, I i → ∀ p, ¬ N p → ¬ gl_typing i I p N')
                examples.append(f'namespace guarded_{selected}\n'
                    'def gl_typing {α : Type} (i : α) (I : α → Prop) (p : α) (N : α → Prop) : Prop := I i ∧ N p\n'
                    'example {α : Type} (I N : α → Prop) (f : α → α → Prop) '
                    '(source : ∀ i p, ¬ gl_typing i I p N → ¬ f i p) : '
                    f'({statement}) := by\n'
                    + '\n'.join(lines) + f'\n  exact row_2\nend guarded_{selected}\n')
            lean = Path(directory)/'Guarded.lean'; lean.write_text('\n'.join(examples),encoding='utf-8')
            result = subprocess.run([lean_executable(), str(lean)],
                cwd=LEAN_PROJECT, env=lean_environment(),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout)

    def test_composed_support_certificate_and_lean_replay(self):
        from proof_export.certificate import _chapter_certificate
        from proof_export.graph import Chapter, ProofRow, Dependency, parse_scoped_expression
        from proof_export.lean import _render_chapter_steps
        from proof_export.mpl import to_mpl
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix='composed_support_') as directory:
            examples = []
            for selected in (0, 1):
                registry = definitions()
                source = '(>[i,p]!(typing[i,I,p,N])!(in2[i,p,f]))'
                helper = ('(>[i]!(in[i,I])(>[p](in[p,N])!(typing[i,I,p,N])))' if selected == 0
                          else '(>[i](in[i,I])(>[p]!(in[p,N])!(typing[i,I,p,N])))')
                direct = helper.replace('!(typing[i,I,p,N])', '!(in2[i,p,f])')
                owner, support, result = [parse_scoped_expression(x) for x in (source, helper, direct)]
                rows = (ProofRow(1, owner, 'main', 'task formulation', ()),
                        ProofRow(2, support, 'main', 'expansion', (Dependency(owner, 'main'),)),
                        ProofRow(3, result, 'main', 'implication composition',
                                (Dependency(support, 'main'), Dependency(owner, 'main'))))
                path = Path(directory)/'1_single_chapter_proof.txt'
                path.write_text('\n'.join('\t'.join([to_mpl(row.conclusion.expression), row.namespace, row.tag,
                    *(value for dependency in row.dependencies for value in
                      (to_mpl(dependency.expression), dependency.namespace))]) for row in rows), encoding='utf-8')
                signatures = {'in': ('element', 'set'), 'in2': ('element', 'element', 'binary_relation'),
                              'typing': ('element', 'set', 'element', 'set')}
                certificate = _chapter_certificate(Chapter(path, 1, 'single_chapter_proof', rows),
                    signatures, {}, {}, binary_entries=registry)
                self.assertEqual([x['action'] for x in certificate['steps']][-2:],
                                 ['conjunction_refute', 'implication_compose'])
                lines, manifest = _render_chapter_steps(certificate, 'composed', {source: 'source'},
                    {name: name for name in ('I', 'N', 'f')}, ambient_variables={'I', 'N', 'f'},
                    binary_definitions=registry)
                self.assertEqual(len(manifest), 3)
                statement = ('∀ i, ¬ I i → ∀ p, N p → ¬ f i p' if selected == 0
                             else '∀ i, I i → ∀ p, ¬ N p → ¬ f i p')
                examples.append(f'namespace composed_{selected}\n'
                    'def gl_typing {α : Type} (i : α) (I : α → Prop) (p : α) (N : α → Prop) : Prop := I i ∧ N p\n'
                    'example {α : Type} (I N : α → Prop) (f : α → α → Prop) '
                    '(source : ∀ i p, ¬ gl_typing i I p N → ¬ f i p) : '
                    f'({statement}) := by\n' + '\n'.join(lines) + f'\n  exact row_3\nend composed_{selected}\n')
            lean = Path(directory)/'Composed.lean'; lean.write_text('\n'.join(examples), encoding='utf-8')
            result = subprocess.run([lean_executable(), str(lean)],
                cwd=LEAN_PROJECT, env=lean_environment(),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout)

    def test_certificate_and_lean_replay_positive_and_negative_members(self):
        from proof_export.certificate import _chapter_certificate
        from proof_export.graph import Chapter, ProofRow, Dependency, parse_scoped_expression
        from proof_export.lean import _render_chapter_steps
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix='conjunction_refutation_') as directory:
            examples = []
            for negative, single in ((False, False), (True, False), (False, True), (True, True)):
                registry = definitions()
                registry['typing']['elements'][1] = ('!' if negative else '') + '(in[u_3,u_4])'
                if single:
                    registry['typing']['elements'] = registry['typing']['elements'][1:]
                source = '(choice[i,I,p,N])'
                target = f'(>[]{"" if negative else "!"}(in[p,N])!(typing[i,I,p,N]))'
                scoped = parse_scoped_expression(source)
                rows = (ProofRow(1, scoped, 'main', 'task formulation', ()),
                        ProofRow(2, parse_scoped_expression(target), 'main', 'expansion',
                                 (Dependency(scoped, 'main'),)))
                file = Path(directory)/'1_single_chapter_proof.txt'
                file.write_text(f'{source}\tmain\ttask formulation\n{target}\tmain\texpansion\t{source}\tmain\n', encoding='utf-8')
                signatures = {'in': ('element', 'set'), 'typing': ('element', 'set', 'element', 'set'),
                              'choice': ('element', 'set', 'element', 'set')}
                certificate = _chapter_certificate(Chapter(file, 1, 'single_chapter_proof', rows),
                    signatures, {}, {}, binary_entries=registry)
                self.assertEqual(certificate['steps'][-1]['action'], 'conjunction_refute')
                lines, manifest = _render_chapter_steps(certificate, 'refutation', {source: 'source'},
                    {name: name for name in ('i', 'I', 'p', 'N')}, ambient_variables={'i', 'I', 'p', 'N'},
                    binary_definitions=registry)
                self.assertEqual(len(manifest), 2)
                suffix = ('negative' if negative else 'positive') + ('_alias' if single else '')
                examples.append(f'namespace {suffix}\n'
                    'def gl_typing {α : Type} (i : α) (I : α → Prop) (p : α) (N : α → Prop) : Prop := '
                    f'{"" if single else "I i ∧ "}{"¬ " if negative else ""}N p\n'
                    'def gl_choice {α : Type} (i : α) (I : α → Prop) (p : α) (N : α → Prop) : Prop := '
                    'I i ∨ gl_typing i I p N\n'
                    'example {α : Type} (i p : α) (I N : α → Prop) (source : gl_choice i I p N) : '
                    f'({"" if negative else "¬ "}N p → ¬ gl_typing i I p N) := by\n'
                    + '\n'.join(lines) + f'\n  exact row_2\nend {suffix}\n')
            lean = Path(directory)/'Refutation.lean'
            lean.write_text('\n'.join(examples), encoding='utf-8')
            result = subprocess.run([lean_executable(), str(lean)],
                cwd=LEAN_PROJECT, env=lean_environment(),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout)


if __name__ == '__main__':
    unittest.main()
