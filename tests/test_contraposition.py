# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschraenkt).
# Dual-licensed under AGPLv3 and a commercial license.
# See https://generative-logic.com/license. Contributions require CLA.

import unittest
import os
import subprocess
import tempfile
from pathlib import Path

from proof_export.contraposition import (
    build_contraposition_plan, build_grouping_plan, validate_auxiliary_scopes)
from proof_export.mpl import parse_mpl
from tests.lean_toolchain import LEAN_PROJECT, lean_environment, lean_executable


class ContrapositionTests(unittest.TestCase):
    def reverse(self, target, source):
        return build_contraposition_plan(parse_mpl(target), parse_mpl(source))

    def test_direct_repeated_parameters_and_binder_permutation(self):
        result = self.reverse('(>[b,a]!(Q[b,a,b,k])!(P[a,b,k]))',
                              '(>[x,y](P[x,y,k])(Q[y,x,y,k]))')
        self.assertEqual(result['source_arguments'], [['a', 'b']])

    def test_all_polarity_combinations(self):
        for p in ('', '!'):
            for q in ('', '!'):
                with self.subTest(p=p, q=q):
                    self.reverse(f'(>[y]{"" if q else "!"}(Q[y]){"" if p else "!"}(P[y]))',
                                 f'(>[x]{p}(P[x]){q}(Q[x]))')

    def test_invalid_reversals(self):
        cases = [
            ('(>[x]!(Q[x])!(P[x]))', '(>[x,y](P[x,y])(Q[x]))'),
            ('(>[x,y]!(Q[x])!(P[x,y]))', '(>[x,y](P[x,y])(Q[x]))'),
            ('(>[x]!(Q[x,b])!(P[x,a]))', '(>[x](P[x,a])(Q[x,a]))'),
            ('(>[x]!(Q[x])(P[x]))', '(>[x](P[x])(Q[x]))'),
            ('(>[x]!(Q[x])!(P[x,x]))', '(>[x,y](P[x,y])(Q[y]))'),
            ('(>[a](=[a,a])!(P[a]))', '(>[x](P[x])!(=[x,a]))'),
        ]
        for target, source in cases:
            with self.subTest(target=target), self.assertRaises(AssertionError):
                self.reverse(target, source)

    def test_grouped_domain_and_three_head_superset(self):
        sources = [parse_mpl('(>[x,y](in2[x,y,f])(in[x,I]))'),
                   parse_mpl('(>[u,v](in2[u,v,f])(in[v,N]))'),
                   parse_mpl('(>[a,b](in2[a,b,f])(in2[a,b,g]))')]
        target = parse_mpl('(>[i,p](in2[i,p,f])(&(in[i,I])(&(in[p,N])(in2[i,p,g]))))')
        result = build_grouping_plan(target, sources, {})
        self.assertEqual(result['source_arguments'], [['i', 'p']] * 3)
        self.assertEqual(result['construction'], [0, [1, 2]])
        with self.assertRaises(AssertionError):
            build_grouping_plan(target, sources[:2], {})

    def test_compiled_conjunction_and_wrong_source(self):
        definition = {'and91': {'category': 'and', 'signature': '(and91[u_1,u_2,u_3,u_4])',
                               'elements': ['(in[u_1,u_2])', '(in[u_3,u_4])']}}
        target = parse_mpl('(>[i,p](in2[i,p,f])(and91[i,I,p,N]))')
        sources = [parse_mpl('(>[x,y](in2[x,y,f])(in[x,I]))'),
                   parse_mpl('(>[x,y](in2[x,y,f])(in[y,N]))')]
        self.assertEqual(build_grouping_plan(target, sources, definition)['construction'], [0, 1])
        typed = {'and91': dict(definition['and91'], arity=4,
                              signature=['element', 'set', 'element', 'set'])}
        self.assertEqual(build_grouping_plan(target, sources, typed),
                         build_grouping_plan(target, sources, definition))
        for bad in [sources[:1], sources + sources[:1],
                    [sources[0], parse_mpl('(>[x,y](in2[x,y,g])(in[y,N]))')]]:
            with self.assertRaises(AssertionError):
                build_grouping_plan(target, bad, definition)

    def test_scope_is_exact_join_without_siblings(self):
        validate_auxiliary_scopes('main_boundary_a', ['main', 'main_boundary_a'])
        for target, sources in [('main', ['main_boundary_a']),
                                ('main_boundary_a', ['main_boundary_b']),
                                ('main_boundary_a', ['main']), ('main', [])]:
            with self.assertRaises(AssertionError):
                validate_auxiliary_scopes(target, sources)

    def test_verifier_dispatch_and_source_scope(self):
        from tests.test_harness import make_proof_line, make_state_minimal
        from verifier import TAG_CHECKERS, _ORIGIN_EXEMPT_TAGS
        source = '(>[x](in[x,A])(in[x,B]))'
        target = '(>[y]!(in[y,B])!(in[y,A]))'
        self.assertNotIn('contraposition', _ORIGIN_EXEMPT_TAGS)
        self.assertNotIn('implication grouping', _ORIGIN_EXEMPT_TAGS)
        for scope, source_scope, valid in [('main', 'main', True),
            ('main', 'main_boundary_case', False),
            ('main_boundary_a', 'main_boundary_b', False)]:
            row = make_proof_line(target, scope, 'contraposition', source, source_scope)
            self.assertEqual(TAG_CHECKERS['contraposition'](row, [row], make_state_minimal()), valid)

    def test_certificate_and_lean_replay_grouping_then_contraposition(self):
        from proof_export.certificate import _chapter_certificate
        from proof_export.graph import Chapter, ProofRow, Dependency, parse_scoped_expression
        from proof_export.lean import _render_chapter_steps
        from proof_export.mpl import to_mpl
        texts = ['(>[x](in[x,A])(in[x,B]))', '(>[y](in[y,A])(in[y,C]))',
                 '(>[z](in[z,A])(&(in[z,B])(in[z,C])))',
                 '(>[t]!(&(in[t,B])(in[t,C]))!(in[t,A]))']
        expressions = [parse_scoped_expression(text) for text in texts]
        rows = (ProofRow(1, expressions[0], 'main', 'task formulation', ()),
                ProofRow(2, expressions[1], 'main', 'task formulation', ()),
                ProofRow(3, expressions[2], 'main', 'implication grouping',
                         tuple(Dependency(value, 'main') for value in expressions[:2])),
                ProofRow(4, expressions[3], 'main', 'contraposition',
                         (Dependency(expressions[2], 'main'),)))
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix='contraposition_') as directory:
            path = Path(directory)/'1_single_chapter_proof.txt'
            path.write_text('\n'.join('\t'.join([to_mpl(row.conclusion.expression), row.namespace,
                row.tag, *(value for dependency in row.dependencies for value in
                    (to_mpl(dependency.expression), dependency.namespace))]) for row in rows), encoding='utf-8')
            certificate = _chapter_certificate(Chapter(path, 1, 'single_chapter_proof', rows),
                                               {'in': ('element', 'set')}, {}, {})
            self.assertEqual(certificate['steps'][-1]['action'], 'implication_contrapose')
            lines, manifest = _render_chapter_steps(certificate, 'contraposition',
                {texts[0]: 'first', texts[1]: 'second'}, {value: value for value in ('A', 'B', 'C')},
                ambient_variables={'A', 'B', 'C'})
            self.assertEqual(len(manifest), 4)
            source = ('example {α : Type} (A B C : α → Prop) '
                '(first : ∀ x, A x → B x) (second : ∀ y, A y → C y) : '
                '(∀ t, ¬ (B t ∧ C t) → ¬ A t) := by\n  classical\n'
                + '\n'.join(lines) + '\n  exact row_4\n')
            lean = Path(directory)/'Contraposition.lean'
            lean.write_text(source, encoding='utf-8')
            result = subprocess.run([lean_executable(), str(lean)],
                cwd=LEAN_PROJECT, env=lean_environment(),
                text=True, encoding='utf-8', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout)

    def test_registered_sources_keep_origin_and_theorem_usage_checks(self):
        from tests.test_harness import make_proof_line, make_state_minimal
        from verifier import verify_chapter, collect_theorem_usage_citations
        source = '(>[x](in[x,A])(in[x,B]))'
        target = '(>[y]!(in[y,B])!(in[y,A]))'
        row = make_proof_line(target, 'main', 'contraposition', source, 'main')
        for present in (False, True):
            state = make_state_minimal()
            if present:
                state.global_theorems = {source: {'type': 'single_chapter_proof', 'ref': '1'}}
            verify_chapter('1_single_chapter_proof.txt', [row], 'single_chapter_proof', state, None)
            self.assertEqual(state.counter_for('origin').failure, 0 if present else 1)
            citations = collect_theorem_usage_citations([row], lambda text: text if present else None)
            self.assertEqual(citations, {source} if present else set())

    def test_lean_kernel_replays_all_four_polarities(self):
        from proof_export.certificate import _chapter_certificate
        from proof_export.graph import Chapter, ProofRow, Dependency, parse_scoped_expression
        from proof_export.lean import _render_chapter_steps
        root = Path(__file__).resolve().parents[1]
        examples = []
        with tempfile.TemporaryDirectory(prefix='contraposition_signs_') as directory:
            for p in ('', '!'):
                for q in ('', '!'):
                    source = f'(>[x]{p}(in[x,A]){q}(in[x,B]))'
                    target = f'(>[y]{"" if q else "!"}(in[y,B]){"" if p else "!"}(in[y,A]))'
                    original = parse_scoped_expression(source)
                    rows = (ProofRow(1, original, 'main', 'task formulation', ()),
                            ProofRow(2, parse_scoped_expression(target), 'main', 'contraposition',
                                     (Dependency(original, 'main'),)))
                    path = Path(directory)/'1_single_chapter_proof.txt'
                    path.write_text(f'{source}\tmain\ttask formulation\n{target}\tmain\tcontraposition\t{source}\tmain\n', encoding='utf-8')
                    certificate = _chapter_certificate(Chapter(path, 1, 'single_chapter_proof', rows),
                                                       {'in': ('element', 'set')}, {}, {})
                    lines, manifest = _render_chapter_steps(certificate, 'signs', {source: 'source'},
                        {'A': 'A', 'B': 'B'}, ambient_variables={'A', 'B'})
                    self.assertEqual(len(manifest), 2)
                    examples.append('example {α : Type} (A B : α → Prop) '
                        f'(source : ∀ x, {"¬ " if p else ""}A x → {"¬ " if q else ""}B x) : '
                        f'(∀ y, {"" if q else "¬ "}B y → {"" if p else "¬ "}A y) := by\n'
                        '  classical\n' + '\n'.join(lines) + '\n  exact row_2\n')
            lean = Path(directory)/'Polarities.lean'
            lean.write_text('\n'.join(examples), encoding='utf-8')
            result = subprocess.run([lean_executable(), str(lean)],
                cwd=LEAN_PROJECT, env=lean_environment(),
                text=True, encoding='utf-8', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout)


if __name__ == '__main__':
    unittest.main()
