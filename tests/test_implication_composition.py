# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
# Dual-licensed under AGPLv3 and a commercial license.
# See https://generative-logic.com/license. Contributions require CLA.

import unittest
import os
import subprocess
import tempfile
from pathlib import Path

from proof_export.implication_composition import build_implication_composition_plan, validate_composition_scopes
from proof_export.mpl import parse_mpl
from tests.lean_toolchain import LEAN_PROJECT, lean_environment, lean_executable


class ImplicationCompositionTests(unittest.TestCase):
    def plan(self, result, first, second):
        return build_implication_composition_plan(*(parse_mpl(x) for x in (result, first, second)))

    def test_multiple_producer_premises_and_distributed_binders(self):
        first = '(>[p](=[p,a])(>[i](=[i,n])(in2[i,p,f])))'
        second = '(>[x,y](in2[x,y,f])(in2[x,y,g]))'
        result = '(>[q](=[q,a])(>[j](=[j,n])(in2[j,q,g])))'
        plan = self.plan(result, first, second)
        self.assertEqual(plan['source_binder_groups'], [[['q'], ['j']], [['j', 'q']]])
        self.assertEqual(plan['result_binder_groups'], [['q'], ['j']])
        for invalid in [
            '(>[q,j](=[q,a])(in2[j,q,g]))',
            '(>[j](=[j,n])(>[q](=[q,a])(in2[j,q,g])))',
            '(>[q](=[q,n])(>[j](=[j,n])(in2[j,q,g])))',
            '(>[q](=[q,a])(>[j]!(=[j,n])(in2[j,q,g])))',
            '(>[](=[q,a])(>[q,j](=[j,n])(in2[j,q,g])))',
            '(>[q](=[q,a])(>[q](=[q,n])(in2[q,q,g])))',
        ]:
            with self.subTest(invalid=invalid), self.assertRaises(AssertionError):
                self.plan(invalid, first, second)
        with self.assertRaises(AssertionError):
            self.plan(result, first, '(>[x,y](in2[x,y,f])(>[](=[x,n])(in2[x,y,g])))')

    def test_lean_kernel_checks_each_producer_premise_and_binder_group(self):
        from proof_export.certificate import _chapter_certificate
        from proof_export.graph import Chapter, ProofRow, Dependency, parse_scoped_expression
        from proof_export.lean import _render_chapter_steps
        first = '(>[p](=[p,a])(>[i](=[i,n])(in2[i,p,f])))'
        second = '(>[x,y](in2[x,y,f])(in2[x,y,g]))'
        conclusion = '(>[q](=[q,a])(>[j](=[j,n])(in2[j,q,g])))'
        expressions = [parse_scoped_expression(value) for value in (first, second, conclusion)]
        rows = (ProofRow(1, expressions[0], 'main', 'task formulation', ()),
                ProofRow(2, expressions[1], 'main', 'task formulation', ()),
                ProofRow(3, expressions[2], 'main', 'implication composition',
                         (Dependency(expressions[0], 'main'), Dependency(expressions[1], 'main'))))
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix='chain_composition_') as directory:
            path = Path(directory)/'1_single_chapter_proof.txt'
            path.write_text(f'{first}\tmain\ttask formulation\n{second}\tmain\ttask formulation\n'
                f'{conclusion}\tmain\timplication composition\t{first}\tmain\t{second}\tmain\n', encoding='utf-8')
            certificate = _chapter_certificate(Chapter(path, 1, 'single_chapter_proof', rows),
                {'=': ('element', 'element'), 'in2': ('element', 'element', 'binary_relation')}, {}, {})
            lines, _ = _render_chapter_steps(certificate, 'composition', {first: 'first', second: 'second'},
                {name: name for name in ('a', 'n', 'f', 'g')}, ambient_variables={'a', 'n', 'f', 'g'})
            lean = Path(directory)/'Composition.lean'
            lean.write_text('example {α : Type} (a n : α) (f g : α → α → Prop) '
                '(first : ∀ p, p=a → ∀ i, i=n → f i p) (second : ∀ x y, f x y → g x y) : '
                '(∀ q, q=a → ∀ j, j=n → g j q) := by\n' + '\n'.join(lines) + '\n  exact row_3\n', encoding='utf-8')
            result = subprocess.run([lean_executable(), str(lean)], cwd=LEAN_PROJECT,
                env=lean_environment(), capture_output=True, text=True, encoding='utf-8', timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_append_bridge_uses_both_source_substitutions(self):
        result = self.plan('(>[i,j](in2[i,j,f])(in2[i,j,g]))',
                           '(>[x,y](in2[x,y,f])(or8[x,y,f,a,k]))',
                           '(>[p,q](or8[p,q,f,a,k])(in2[p,q,g]))')
        self.assertEqual(result['source_arguments'], [['i', 'j'], ['i', 'j']])
        self.assertEqual(result['intermediate'], '(or8[i,j,f,a,k])')

    def test_repeated_arguments_and_negative_intermediate(self):
        result = self.plan('(>[a,b](P[a,b])(R[b,a]))', '(>[x,y](P[x,y])!(Q[x,y,x]))',
                           '(>[v,w]!(Q[v,w,v])(R[w,v]))')
        self.assertEqual(result['source_arguments'], [['a', 'b'], ['a', 'b']])

    def test_binders_are_separate_between_sources(self):
        result = self.plan('(>[a,b](P[a,b])(R[b,a]))',
                           '(>[x,y](P[x,y])(Q[x,y]))', '(>[x,y](Q[y,x])(R[x,y]))')
        self.assertEqual(result['source_arguments'], [['a', 'b'], ['b', 'a']])

    def test_composition_requires_the_same_complete_binder_set(self):
        cases = [
            ('(>[](=[u_2,u_10])(=[u_2,u_2]))',
             '(>[1](=[1,u_10])(in[1,u_int_lev_4_10]))',
             '(>[](in[u_2,u_int_lev_4_10])(=[u_2,u_2]))'),
            ('(>[z](P[z,z])(R[z]))', '(>[x,y](P[x,y])(Q[x,y]))', '(>[v](Q[v,v])(R[v]))'),
            ('(>[a,b](P[a,b])(R[a,b]))', '(>[x,y](P[x,y])(Q[x]))', '(>[v,w](Q[v])(R[v,w]))'),
            ('(>[a,b,c](P[a,b])(R[c,a]))', '(>[x,y](P[x,y])(Q[x]))', '(>[v,w](Q[w])(R[v,w]))'),
            ('(>[](P[u_a])(R[u_a]))', '(>[x](P[x])(Q[x]))', '(>[](Q[u_a])(R[u_a]))'),
            ('(>[a](P[a])(R[a]))', '(>[x,y](P[x])(Q[x,y]))', '(>[v,w](Q[v,w])(R[v]))'),
            ('(>[a,b](P[a])(R[b]))', '(>[x](P[x])(Q[x,u_a]))', '(>[y](Q[u_a,y])(R[y]))'),
        ]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(AssertionError):
                self.plan(*case)

    def test_altered_endpoints_fixed_parameters_and_polarity_fail(self):
        cases = [
            ('(>[z](P[z])(S[z]))', '(>[x](P[x])(Q[x]))', '(>[y](Q[y])(R[y]))'),
            ('(>[z](P[z])(R[z]))', '(>[x](P[x])(Q[x,u_a]))', '(>[y](Q[y,u_b])(R[y]))'),
            ('(>[z](P[z])(R[z]))', '(>[x](P[x])(Q[x]))', '(>[y]!(Q[y])(R[y]))'),
            ('(>[a,b](P[a])(R[b]))', '(>[x](P[x])(Q[x]))', '(>[y](Q[y])(R[y]))'),
            ('(>[z](P[z])(R[z]))', '(>[x](P[x])(>[](S[x])(Q[x])))', '(>[y](Q[y])(R[y]))'),
        ]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(AssertionError):
                self.plan(*case)

    def test_scope_join_accepts_ancestors_and_rejects_siblings_and_scope_lifting(self):
        child = 'main_boundary_left'
        validate_composition_scopes(child, 'main', child)
        validate_composition_scopes(child, child, child)
        for scopes in [('main', 'main', child), (child, child, 'main_boundary_right'),
                       (child + '_boundary_deeper', 'main', child)]:
            with self.assertRaises(AssertionError):
                validate_composition_scopes(*scopes)

    def test_verifier_dispatch_checks_both_dependencies_and_exact_scope(self):
        from tests.test_harness import make_state_minimal, make_proof_line
        from verifier import check_implication_composition, TAG_CHECKERS, _ORIGIN_EXEMPT_TAGS
        first = '(>[x](in[x,A])(in[x,B]))'
        second = '(>[y](in[y,B])(in[y,C]))'
        conclusion = '(>[z](in[z,A])(in[z,C]))'
        self.assertIs(TAG_CHECKERS['implication composition'], check_implication_composition)
        self.assertNotIn('implication composition', _ORIGIN_EXEMPT_TAGS)
        state = make_state_minimal()
        for target, left, right, expected in [
            ('main', 'main', 'main', True),
            ('main_boundary_case', 'main', 'main_boundary_case', True),
            ('main', 'main', 'main_boundary_case', False),
            ('main_boundary_case', 'main_boundary_other', 'main_boundary_case', False),
        ]:
            row = make_proof_line(conclusion, target, 'implication composition', first, left, second, right)
            self.assertEqual(check_implication_composition(row, [row], state), expected)
        row = make_proof_line(conclusion, 'main', 'implication composition', first, 'main')
        self.assertFalse(check_implication_composition(row, [row], state))
        row = make_proof_line(conclusion, 'main', 'implication composition', first, 'main', first, 'main')
        self.assertFalse(check_implication_composition(row, [row], state))

    def certificate(self):
        from proof_export.certificate import _chapter_certificate
        from proof_export.graph import Chapter, ProofRow, Dependency, parse_scoped_expression
        first = parse_scoped_expression('(>[x](in[x,A])(in[x,B]))')
        second = parse_scoped_expression('(>[y](in[y,B])(in[y,C]))')
        conclusion = parse_scoped_expression('(>[z](in[z,A])(in[z,C]))')
        rows = (
            ProofRow(1, first, 'main', 'task formulation', ()),
            ProofRow(2, second, 'main', 'task formulation', ()),
            ProofRow(3, conclusion, 'main', 'implication composition',
                     (Dependency(first, 'main'), Dependency(second, 'main'))),
        )
        from proof_export.mpl import to_mpl
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix='composition_certificate_') as directory:
            path = Path(directory)/'1_single_chapter_proof.txt'
            path.write_text('\n'.join('\t'.join([to_mpl(row.conclusion.expression), row.namespace, row.tag,
                *(value for dependency in row.dependencies for value in
                  (to_mpl(dependency.expression), dependency.namespace))]) for row in rows), encoding='utf-8')
            chapter = Chapter(path, 1, 'single_chapter_proof', rows)
            return _chapter_certificate(chapter, {'in': ('element', 'set')}, {}, {})

    def test_registered_sources_are_checked_and_counted_as_theorem_usage(self):
        from tests.test_harness import make_state_minimal, make_proof_line
        from verifier import verify_chapter, collect_theorem_usage_citations, check_theorem_usage_termination
        first = '(>[x](in[x,A])(in[x,B]))'
        second = '(>[y](in[y,B])(in[y,C]))'
        conclusion = '(>[z](in[z,A])(in[z,C]))'
        row = make_proof_line(conclusion, 'main', 'implication composition', first, 'main', second, 'main')
        for registered, failures in [({first, second}, 0), ({first}, 1), ({second}, 1), (set(), 2)]:
            state = make_state_minimal()
            state.global_theorems = {text: {'type': 'single_chapter_proof', 'ref': '1'} for text in registered}
            verify_chapter('1_single_chapter_proof.txt', [row], 'single_chapter_proof', state, None)
            self.assertEqual(state.counter_for('origin').failure, failures)
            cited = collect_theorem_usage_citations([row], lambda text: text if text in registered else None)
            self.assertEqual(cited, registered)
        state = make_state_minimal()
        check_theorem_usage_termination([conclusion, second],
            {conclusion: {second}, second: {conclusion}}, state)
        self.assertEqual(state.counter_for('theorem usage termination').failure, 2)
        scoped = make_proof_line(conclusion, 'main_boundary_case', 'implication composition',
            first, 'main', second, 'main_boundary_case')
        state = make_state_minimal()
        state.global_theorems = {text: {'type': 'single_chapter_proof', 'ref': '1'} for text in (first, second)}
        verify_chapter('1_single_chapter_proof.txt', [scoped], 'single_chapter_proof', state, None)
        self.assertEqual(state.counter_for('origin').failure, 1)

    def test_certificate_records_exact_source_instantiations(self):
        certificate = self.certificate()
        step = certificate['steps'][-1]
        self.assertEqual(step['action'], 'implication_compose')
        self.assertEqual(step['composition']['source_arguments'], [['z'], ['z']])
        self.assertEqual(len(step['dependencies']), 2)

    def test_lean_kernel_checks_explicit_composition(self):
        from proof_export.lean import _render_chapter_steps
        certificate = self.certificate()
        replacements = {name: name for name in ('A', 'B', 'C')}
        lines, manifest = _render_chapter_steps(certificate, 'composition', {
            '(>[x](in[x,A])(in[x,B]))': 'first',
            '(>[y](in[y,B])(in[y,C]))': 'second',
        }, replacements, ambient_variables={'A', 'B', 'C'})
        self.assertEqual(len(manifest), 3)
        proof = '\n'.join(lines)
        self.assertIn('composition_source_2', proof)
        self.assertNotIn('sorry', proof)
        source = ('example {α : Type} (A B C : α → Prop) '
                  '(first : ∀ x, A x → B x) (second : ∀ y, B y → C y) : '
                  '(∀ z, A z → C z) := by\n' + proof + '\n  exact row_3\n')
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix='unary_composition_') as directory:
            path = Path(directory)/'Composition.lean'
            path.write_text(source, encoding='utf-8')
            result = subprocess.run([lean_executable(), str(path)],
                cwd=LEAN_PROJECT, env=lean_environment(),
                text=True, encoding='utf-8', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout)

    def test_lean_kernel_checks_same_scope_and_ancestor_composition(self):
        from proof_export.lean import _render_chapter_steps
        first = '(>[x](in[x,A])(in[x,B]))'
        second = '(>[y](in[y,B])(in[y,C]))'
        conclusion = '(>[z](in[z,A])(in[z,C]))'
        scope = 'main_boundary_composition'
        examples = []
        for same_scope in (False, True):
            template = '(>[]' + second + '(in[t,C]))'
            if same_scope:
                template = '(>[]' + first + template + ')'
            steps = []
            for expression, local in ((first, same_scope), (second, True)):
                index = len(steps) + 1
                step = {'id': f'chapter_1_line_{index}', 'source_line': index,
                    'namespace': scope if local else 'main',
                    'action': 'integration_premise' if local else 'assume',
                    'source_tag': 'premise element' if local else 'task formulation',
                    'conclusion': {'mpl': expression}, 'dependencies': []}
                if local:
                    step['scope_namespace'] = scope
                    step['dependencies'] = [{'namespace': 'main', 'source_kind': 'integration_goal_template',
                                             'expression': {'mpl': template}}]
                steps.append(step)
            dependencies = [{'step': step['id'], 'namespace': step['namespace'],
                             'source_kind': 'local_step', 'expression': step['conclusion']} for step in steps]
            steps.append({'id': 'chapter_1_line_3', 'source_line': 3, 'namespace': scope,
                'action': 'implication_compose', 'source_tag': 'implication composition',
                'conclusion': {'mpl': conclusion}, 'dependencies': dependencies,
                'composition': self.plan(conclusion, first, second)})
            chapter = {'steps': steps, 'variable_types': {'A': 'set', 'B': 'set', 'C': 'set',
                'x': 'element', 'y': 'element', 'z': 'element', 't': 'element'},
                'scope_contexts': [{'namespace': scope, 'parent_namespace': 'main',
                    'template': {'mpl': template}, 'compact': {'mpl': '(implication99[A,B,C,t])'}}]}
            proof, manifest = _render_chapter_steps(chapter, 'scoped_composition',
                {} if same_scope else {first: 'first'}, {name: name for name in ('A', 'B', 'C', 't')},
                ambient_variables={'A', 'B', 'C', 't'})
            self.assertEqual(len(manifest), 3)
            parameter = '' if same_scope else ' (first : ∀ x, A x → B x)'
            first_premise = '(∀ x, A x → B x) → ' if same_scope else ''
            examples.append('example {α : Type} (A B C : α → Prop) (t : α)' + parameter + ' : '
                + first_premise + '(∀ y, B y → C y) → (∀ z, A z → C z) := by\n'
                + '\n'.join(proof) + '\n  exact row_3\n')
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix='scoped_composition_') as directory:
            path = Path(directory)/'ScopedComposition.lean'
            path.write_text('\n'.join(examples), encoding='utf-8')
            result = subprocess.run([lean_executable(), str(path)],
                cwd=LEAN_PROJECT, env=lean_environment(),
                text=True, encoding='utf-8', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout)


if __name__ == '__main__':
    unittest.main()
