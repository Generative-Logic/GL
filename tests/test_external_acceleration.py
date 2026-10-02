# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt)
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Output and lifetime regressions for external pipeline acceleration."""

import random
import sys
import tempfile
import unittest
import json
import contextlib
import io
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import generate_full_proof_graph as html
import process_proof_graphs as processed
from configuration_reader import configuration_reader
from proof_export import mpl
from dataclasses import FrozenInstanceError
import run_modes


class ReverseCitationTests(unittest.TestCase):
    def test_index_preserves_first_nonself_match_and_missing_sources(self):
        rng = random.Random(419)
        records = [(f'theorem{i}', 'direct', '-1') for i in range(36)]
        leading = [3*i for i in range(len(records))]
        paths = {name: [name] for name, _, _ in records}
        stacks = {name: [[f'theorem{rng.randrange(40)}', 'main', 'theorem']
                         for _ in range(14)] + [[], ['short'], [name, 'main', 'origin']]
                  for name, _, _ in records}
        # Different source names can have an equal display identity. First
        # source order and exclusion of the citing record must both survive.
        rename = lambda name: 'same' if name in ('theorem0', 'theorem1') else name
        stacks['theorem0'] += [['theorem0', 'main', 'theorem']]
        expected = {}
        for i, (name, _, _) in enumerate(records):
            for row in stacks[name]:
                if len(row) < 3 or row[2] != 'theorem':
                    continue
                for j, (source, _, _) in enumerate(records):
                    if i != j and rename(source) == rename(row[0]):
                        expected.setdefault(leading[j], set()).add(leading[i])
                        break
        with mock.patch.object(html.os.path, 'exists', return_value=True), \
             mock.patch.object(html, 'read_stack', side_effect=lambda path, _: stacks[path]), \
             mock.patch.object(html, 'rename_theorem', side_effect=rename) as renamed:
            self.assertEqual(html._build_used_by(records, paths, leading), expected)
            self.assertLess(renamed.call_count, 600)

    def test_empty_and_absent_chapters(self):
        self.assertEqual(html._build_used_by([], {}, []), {})
        with mock.patch.object(html.os.path, 'exists', return_value=False):
            self.assertEqual(html._build_used_by([('x', '', '')], {'x': ['absent']}, [0]), {})

    def test_misaligned_chapter_numbers_assert(self):
        with self.assertRaises(AssertionError):
            html._build_used_by([('x', '', '')], {}, [])


class RenderCacheTests(unittest.TestCase):
    def test_cache_scopes_restore_after_success_and_failure(self):
        def render(*_):
            self.assertIsNotNone(html._RENDER_CACHE.get())
            with mock.patch.object(html, '_resolve_theorem_target', return_value='one.html') as resolve:
                first = html.wrap_clickable('(p[x])')
                self.assertEqual(first, html.wrap_clickable('(p[x])'))
                self.assertEqual(resolve.call_count, 1)
            return first
        with mock.patch.object(html, '_generate_proof_graph_pages', side_effect=render):
            self.assertIn('one.html', html.generate_proof_graph_pages(None))
            self.assertIsNone(html._RENDER_CACHE.get())
        with mock.patch.object(html, '_generate_proof_graph_pages', side_effect=ValueError('expected')):
            with self.assertRaisesRegex(ValueError, 'expected'):
                html.generate_proof_graph_pages(None)
        self.assertIsNone(html._RENDER_CACHE.get())
        with mock.patch.object(html, '_resolve_theorem_target', return_value='two.html'):
            self.assertIn('two.html', html.wrap_clickable('(p[x])'))

    def test_chapter_cache_returns_independent_rows_and_expires(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'chapter.txt'
            path.write_text('(p[x])\tmain\torigin\n\n', encoding='utf-8')
            def render(*_):
                first = html.read_stack(path, None)
                first[0][0] = 'changed'
                first[1].append('changed')
                self.assertEqual(html.read_stack(path, None), [['(p[x])','main','origin'],[]])
            with mock.patch.object(html, '_generate_proof_graph_pages', side_effect=render):
                html.generate_proof_graph_pages(None)
            path.write_text('(q[x])\tmain\torigin\n', encoding='utf-8')
            self.assertEqual(html.read_stack(path, None)[0][0], '(q[x])')

    def test_actual_cached_and_uncached_render_match(self):
        config = configuration_reader(Path(__file__).resolve().parent.parent/'files/config/ConfigVisu.json')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'processed_proof_graph'
            source.mkdir()
            (source/'global_theorem_list.txt').write_text(
                '(in[v1,N])\tsingle chapter proof\t-1\n'
                '(in[v2,N])\tproved not broadcast\t-1\n',
                encoding='utf-8',
            )
            (source/'0_single_chapter_proof.txt').write_text('(in[v1,N])\tmain\torigin\n', encoding='utf-8')
            (source/'1_proved_not_broadcast.txt').write_text('(in[v2,N])\tmain\torigin\n', encoding='utf-8')
            with mock.patch.object(html, 'build_gl_binary_map', return_value={}):
                html._generate_proof_graph_pages(config, source, root/'oracle')
                html.generate_proof_graph_pages(config, source, root/'cached')
            expected = {p.name:p.read_bytes() for p in (root/'oracle').iterdir() if p.is_file()}
            self.assertEqual(expected, {p.name:p.read_bytes() for p in (root/'cached').iterdir() if p.is_file()})


class FinalizationTests(unittest.TestCase):
    def test_dependencies_counts_and_parent_failure_join(self):
        for parent_fails in (False, True):
            with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as patches:
                root = Path(directory)
                (root/'.debug').mkdir()
                calls = []
                def start(command, **kwargs):
                    calls.append('spawn')
                    Path(command[3]).write_text(json.dumps({'success':20,'failure':0}), encoding='utf-8')
                    return mock.Mock(wait=mock.Mock(side_effect=lambda: calls.append('join') or 0))
                def lean(*_):
                    calls.append('lean')
                    if parent_fails:
                        raise RuntimeError('parent failure')
                patches.enter_context(mock.patch.object(run_modes, 'PROJECT_ROOT', root))
                patches.enter_context(mock.patch.object(run_modes.process_proof_graphs, 'create_processed_proof_graph', side_effect=lambda *_:calls.append('process')))
                patches.enter_context(mock.patch.object(run_modes.subprocess, 'Popen', side_effect=start))
                patches.enter_context(mock.patch.object(run_modes.lean_live, 'export_main_graph', side_effect=lean))
                patches.enter_context(mock.patch.object(run_modes.generate_full_proof_graph, 'generate_proof_graph_pages', side_effect=lambda *_,**__:calls.append('html')))
                patches.enter_context(mock.patch.object(run_modes.verifier, 'run_verifier', side_effect=lambda *_:calls.append('verify')))
                patches.enter_context(mock.patch.object(run_modes.verifier, 'print_report'))
                patches.enter_context(mock.patch.object(run_modes.verifier, 'get_totals', return_value=(10,0)))
                patches.enter_context(contextlib.redirect_stdout(io.StringIO()))
                if parent_fails:
                    with self.assertRaisesRegex(RuntimeError, 'parent failure'):
                        run_modes._finish_full_graphs(None, True, True)
                    self.assertEqual(calls, ['process','spawn','lean','join'])
                else:
                    self.assertEqual(run_modes._finish_full_graphs(None, True, True), (30,0))
                    self.assertEqual(calls, ['process','spawn','lean','html','verify','join'])

    def test_failed_child_asserts_and_disabled_graphs_do_no_work(self):
        self.assertEqual(run_modes._finish_full_graphs(None, False, False), (0,0))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'.debug').mkdir()
            with mock.patch.object(run_modes,'PROJECT_ROOT',root), \
                 mock.patch.object(run_modes.subprocess,'Popen',return_value=mock.Mock(wait=mock.Mock(return_value=7))), \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(AssertionError, 'Incubator finalization failed'):
                    run_modes._finish_full_graphs(None, False, True)

    def test_worker_returns_real_verification_counts_and_keeps_sibling_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory)/'result.json'
            with mock.patch.object(run_modes,'configuration_reader',return_value=None), \
                 mock.patch.object(run_modes.generate_full_proof_graph,'generate_proof_graph_pages') as render, \
                 mock.patch.object(run_modes.verifier,'run_verifier',return_value='state'), \
                 mock.patch.object(run_modes.verifier,'print_report'), \
                 mock.patch.object(run_modes.verifier,'get_totals',return_value=(15,2)), \
                 contextlib.redirect_stdout(io.StringIO()):
                run_modes._finalize_incubator_worker(str(result), True)
            self.assertEqual(json.loads(result.read_text()), {'success':15,'failure':2})
            self.assertEqual(render.call_args.kwargs['sibling_graphs'],
                             [{'proc_dir':run_modes.PROJECT_ROOT/'files/processed_proof_graph',
                               'out_dir':run_modes.PROJECT_ROOT/'files/full_proof_graph'}])


class MplCacheTests(unittest.TestCase):
    def test_immutable_parse_and_alpha_results_match_uncached_functions(self):
        source = '(>[x](p[x,y])(>[z](q[x,z])(r[z,y])))'
        tree = mpl.parse_mpl(source)
        self.assertEqual(tree, mpl.parse_mpl.__wrapped__(source))
        self.assertIs(tree, mpl.parse_mpl(source))
        with self.assertRaises(FrozenInstanceError):
            tree.bound_variables = ('changed',)
        self.assertEqual(mpl.alpha_key(tree), mpl.alpha_key.__wrapped__(tree))
        renamed = mpl.parse_mpl('(>[a](p[a,y])(>[b](q[a,b])(r[b,y])))')
        self.assertEqual(mpl.alpha_key(tree), mpl.alpha_key(renamed))
        different_free = mpl.parse_mpl(source.replace(',y', ',other'))
        self.assertNotEqual(mpl.alpha_key(tree), mpl.alpha_key(different_free))

    def test_invalid_input_remains_an_error_on_repeated_calls(self):
        for _ in range(2):
            with self.assertRaises(AssertionError):
                mpl.parse_mpl('(p[x])trailing')


class ProcessedCacheTests(unittest.TestCase):
    def test_cached_and_uncached_runs_match_and_do_not_retain_configuration(self):
        config = configuration_reader(Path(__file__).resolve().parent.parent /
                                      'files/config/ConfigVisu.json')
        theorem = '(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7](in[7,1])(in[7,1])))'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / 'raw_proof_graph'
            raw.mkdir()
            (root / 'theorems').mkdir()
            (raw / 'global_theorem_list.txt').write_text(theorem+'\tsingle chapter proof\t-1\n', encoding='utf-8')
            cell = '(>[8](in[8,1])(in[8,1]))'
            rows = [f'{theorem}\tmain\ttask formulation',
                    f'{cell}\tmain\tdisintegration\t{theorem}\tmain',
                    f'{cell}\tmain\tequality1\t{cell}\tmain',
                    f'{theorem}\tmain_child\tzoom in\t{theorem}\tmain']
            (raw / '0_single_chapter_proof.txt').write_text('\n'.join(rows)+'\n', encoding='utf-8')
            with mock.patch.object(processed, 'cache', side_effect=lambda function: function):
                processed.create_processed_proof_graph(config, raw, root/'oracle')
            with mock.patch.object(processed, 'classify_args_by_defset',
                                   wraps=processed.classify_args_by_defset) as classify:
                processed.create_processed_proof_graph(config, raw, root/'cached')
                calls = classify.call_count
                processed.create_processed_proof_graph(config, raw, root/'again')
                self.assertEqual(classify.call_count, 2*calls)
            oracle = {p.name: p.read_bytes() for p in (root/'oracle').iterdir()}
            for label in ('cached', 'again'):
                self.assertEqual(oracle, {p.name:p.read_bytes() for p in (root/label).iterdir()})


if __name__ == '__main__':
    unittest.main()
