# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt)
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Tests for the live Lean export (``proof_export.live``): anchor blocks of a
theorem list, toolchain discovery, head-alias inference, the adaptation
search, and content-based resolution of external citations."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proof_export import live  # noqa: E402
from proof_export.certificate import (  # noqa: E402
    theorem_adaptation_solutions, _ordered_selected_theorems, _citation_source_indices,
)
from proof_export.graph import (  # noqa: E402
    Chapter,
    Dependency,
    ProofRow,
    ScopedExpression,
    TheoremRecord,
)
from proof_export.mpl import parse_mpl  # noqa: E402


PEANO = "(>[N,i0,s,+,*,i1](AnchorPeano[N,i0,s,+,*,i1])"
GAUSS = "(>[N,i0,s,+,*,i1,i2,id](AnchorGauss[N,i0,s,+,*,i1,i2,id])"
FTA = "(>[N,i0,s,+,*,i1,i2,id](AnchorFTA[N,i0,s,+,*,i1,i2,id])"


def _record(index: int, mpl: str, method: str = "single chapter proof") -> TheoremRecord:
    return TheoremRecord(index, parse_mpl(mpl), method, "-1", ())


def _scoped(mpl: str) -> ScopedExpression:
    return ScopedExpression(parse_mpl(mpl), mpl, "ordinary")


def _theorem_with_citation(index: int, mpl: str, cited_mpl: str) -> TheoremRecord:
    """A single-chapter theorem that cites one previously proved theorem."""

    rows = (
        ProofRow(1, _scoped(cited_mpl), "main", "theorem", ()),
        ProofRow(
            2, _scoped(mpl), "main", "implication",
            (Dependency(_scoped(cited_mpl), "main"),),
        ),
    )
    return TheoremRecord(
        index, parse_mpl(mpl), "single chapter proof", "-1",
        (Chapter(Path("0_single_chapter_proof.txt"), 0, "single_chapter_proof", rows),),
    )


class SameListDuplicateProofTests(unittest.TestCase):
    def setUp(self):
        loader = mock.patch("proof_export.certificate.find_theorem",
                            side_effect=lambda records, _expression, _method, index: records[index])
        loader.start()
        self.addCleanup(loader.stop)

    def records(self):
        prior = PEANO + "(in[i0,N]))"
        target = GAUSS + "(in[i2,N]))"
        return (_record(0, prior), _record(1, prior), _theorem_with_citation(2, target, prior))

    def selection(self, imported):
        return {"schema_version": 3, "theorem_id_prefix": "gauss_source",
                "theorem_name_prefix": "Gauss source", "source_range": {"first": 2, "last": 2},
                "excluded_sources": [], "certificate_dependencies": [{"required_source_indices": imported}]}

    def test_duplicate_prior_proofs_pin_the_first_source(self):
        records = self.records()
        self.assertEqual(live.same_list_imports(records, [records[2]], 2, 2), [0])
        _, edges = _ordered_selected_theorems(self.selection([0]), records)
        self.assertEqual(edges, {2: {0}})

    def test_two_authorized_copies_are_still_ambiguous(self):
        with self.assertRaisesRegex(AssertionError, "alpha-ambiguous"):
            _ordered_selected_theorems(self.selection([0, 1]), self.records())

    def test_an_unselected_prior_theorem_is_rejected(self):
        with self.assertRaisesRegex(AssertionError, "unselected theorem"):
            _ordered_selected_theorems(self.selection([]), self.records())


class SelectedCitationSourceTests(unittest.TestCase):
    def setUp(self):
        loader = mock.patch("proof_export.certificate.find_theorem",
                            side_effect=lambda records, _expression, _method, index: records[index])
        loader.start()
        self.addCleanup(loader.stop)

    def records(self, tag="compilation"):
        prior = PEANO + "(in[i0,N]))"
        target = PEANO + "(in[i1,N]))"
        row = ProofRow(1, _scoped("(implication0[])"), "main", tag,
                       (Dependency(_scoped(prior), "main"),))
        consumer = TheoremRecord(0, parse_mpl(target), "single chapter proof", "-1",
                                 (Chapter(Path("0_single_chapter_proof.txt"), 0, "single_chapter_proof", (row,)),))
        return (consumer, _record(1, prior), _record(2, prior))

    def selection(self, schema=2):
        return {"schema_version": schema, "source_range": {"first": 0, "last": 2},
                "excluded_sources": [], "citation_source_indices": [0, 1],
                "theorem_id_prefix": "test", "theorem_name_prefix": "Test",
                "certificate_dependencies": []}

    def test_canonical_sources_keep_one_index_per_statement_in_range(self):
        records = self.records()
        self.assertEqual(live.canonical_citation_sources(records, 0, 2), [0, 1])
        self.assertEqual(live.canonical_citation_sources(records, 2, 2), [2])
        for first, last in [(-1, 1), (2, 1), (0, 3)]:
            with self.subTest(first=first, last=last), self.assertRaises(AssertionError):
                live.canonical_citation_sources(records, first, last)

    def test_compilation_citation_orders_its_proof_and_keeps_both_copies(self):
        for schema in (2, 3):
            with self.subTest(schema=schema):
                ordered, edges = _ordered_selected_theorems(self.selection(schema), self.records())
                self.assertEqual([record.source_index for _, record in ordered], [1, 2, 0])
                self.assertEqual(edges, {0: {1}, 1: set(), 2: set()})

    def test_missing_explicit_choices_preserve_ambiguity_failure(self):
        selection = self.selection()
        del selection["citation_source_indices"]
        with self.assertRaisesRegex(AssertionError, "2 selected theorems"):
            _ordered_selected_theorems(selection, self.records())

    def test_invalid_choices_assert(self):
        for choices, message in [([0, 1, 1], "duplicate"), ([0, 3], "not selected"),
                                 ([0, 1, 2], "alpha-ambiguous"), ([1], "omit")]:
            selection = self.selection()
            selection["citation_source_indices"] = choices
            with self.subTest(choices=choices), self.assertRaisesRegex(AssertionError, message):
                _citation_source_indices(selection, list(self.records()))
        selection = self.selection()
        selection["citation_source_indices"] = [False, 1]
        with self.assertRaises(AssertionError):
            _citation_source_indices(selection, list(self.records()))

    def test_explicit_choices_do_not_hide_dependency_cycles(self):
        records = list(self.records())
        records[1] = _theorem_with_citation(1, PEANO + "(in[i0,N]))", PEANO + "(in[i1,N]))")
        with self.assertRaisesRegex(AssertionError, "cyclic"):
            _ordered_selected_theorems(self.selection(), tuple(records))

    def test_local_compilation_does_not_create_a_theorem_edge(self):
        records = list(self.records())
        row = ProofRow(1, _scoped("(implication0[])"), "main", "compilation",
                       (Dependency(_scoped("(in[i0,N])"), "main"),))
        records[0] = TheoremRecord(0, records[0].expression, "single chapter proof", "-1",
                                  (Chapter(Path("0_single_chapter_proof.txt"), 0, "single_chapter_proof", (row,)),))
        _, edges = _ordered_selected_theorems(self.selection(), tuple(records))
        self.assertEqual(edges, {0: set(), 1: set(), 2: set()})


class OrTheoremCompanionCitationTests(unittest.TestCase):
    """Schema 1/2 ordering of an or-theorem row whose head-switch companion
    is not a registered theorem (D-217)."""

    def setUp(self):
        loader = mock.patch("proof_export.certificate.find_theorem",
                            side_effect=lambda records, _expression, _method, index: records[index])
        loader.start()
        self.addCleanup(loader.stop)

    def records(self, with_grounding=True):
        grounding = PEANO + "(in[i0,N]))"
        companion = PEANO + "(in[i1,N]))"
        target = PEANO + "(in[s,N]))"
        row = ProofRow(1, _scoped(target), "main", "or theorem",
                       (Dependency(_scoped(grounding), "main"),
                        Dependency(_scoped(companion), "main")))
        or_theorem = TheoremRecord(0, parse_mpl(target), "or theorem", "-1",
                                   (Chapter(Path("0_or_theorem.txt"), 0, "or_theorem", (row,)),))
        return (or_theorem, _record(1, grounding)) if with_grounding else (or_theorem,)

    def selection(self, last):
        return {"schema_version": 2, "source_range": {"first": 0, "last": last},
                "excluded_sources": [], "theorem_id_prefix": "test",
                "theorem_name_prefix": "Test", "certificate_dependencies": []}

    def test_unregistered_companion_is_dropped_and_grounding_orders_first(self):
        ordered, edges = _ordered_selected_theorems(self.selection(1), self.records())
        self.assertEqual([record.source_index for _, record in ordered], [1, 0])
        self.assertEqual(edges, {0: {1}, 1: set()})

    def test_unregistered_grounding_still_asserts(self):
        with self.assertRaisesRegex(AssertionError, "resolves to 0 selected theorems"):
            _ordered_selected_theorems(self.selection(0), self.records(with_grounding=False))


class TheoremListBlockTests(unittest.TestCase):
    def test_contiguous_anchor_runs_become_blocks(self) -> None:
        records = (
            _record(0, PEANO + "(in[i0,N]))"),
            _record(1, PEANO + "(in[i1,N]))"),
            _record(2, GAUSS + "(in[i2,N]))"),
        )
        self.assertEqual(
            live.theorem_list_blocks(records),
            {"AnchorPeano": (0, 1), "AnchorGauss": (2, 2)},
        )

    def test_interleaved_anchors_assert(self) -> None:
        records = (
            _record(0, PEANO + "(in[i0,N]))"),
            _record(1, GAUSS + "(in[i2,N]))"),
            _record(2, PEANO + "(in[i1,N]))"),
        )
        with self.assertRaises(AssertionError):
            live.theorem_list_blocks(records)

    def test_unknown_anchor_asserts(self) -> None:
        with self.assertRaises(AssertionError):
            live.theorem_list_blocks((_record(0, "(>[N](AnchorX[N])(in[i0,N]))"),))


class LakeDiscoveryTests(unittest.TestCase):
    def test_path_wins_then_elan_then_none(self) -> None:
        # The default-home branch applies only when no ELAN_HOME is configured.
        environment = {key: value for key, value in os.environ.items() if key != "ELAN_HOME"}
        with tempfile.TemporaryDirectory() as temporary,                 mock.patch.dict(os.environ, environment, clear=True):
            root = Path(temporary)
            on_path = root / "bin" / ("lake.exe" if os.name == "nt" else "lake")
            on_path.parent.mkdir()
            on_path.write_bytes(b"")
            with mock.patch("shutil.which", return_value=str(on_path)):
                self.assertEqual(live.find_lake(), on_path)
            home = root / "home"
            elan = home / ".elan" / "bin" / "lake.exe"
            elan.parent.mkdir(parents=True)
            elan.write_bytes(b"")
            with mock.patch("shutil.which", return_value=None), \
                    mock.patch("pathlib.Path.home", return_value=home):
                self.assertEqual(live.find_lake(), elan)
            with mock.patch("shutil.which", return_value=None), \
                    mock.patch("pathlib.Path.home", return_value=root / "nowhere"):
                self.assertIsNone(live.find_lake())


    def test_configured_elan_home_is_used_without_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary) / "custom-elan"
            lake = home / "bin" / "lake.exe"
            lake.parent.mkdir(parents=True)
            lake.write_bytes(b"")
            with mock.patch.dict(os.environ, {"ELAN_HOME": str(home)}), \
                    mock.patch("shutil.which", return_value=None):
                self.assertEqual(live.find_lake(), lake)


class KernelLaunchTests(unittest.TestCase):
    def test_skipped_export_removes_stale_certificate_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stale = root / "lean_export"
            stale.mkdir()
            (stale / "manifest.json").write_text("stale")
            (root / "index.html").write_text("graph")
            live.skip_export(root, "Development shortcut")
            self.assertFalse(stale.exists())
            self.assertEqual((root / "index.html").read_text(), "graph")

    def test_default_elan_home_is_explicit_and_parent_environment_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lake = root / "lake.exe"
            with mock.patch.dict(os.environ, {"KEPT_SETTING": "present"}, clear=True), \
                    mock.patch("pathlib.Path.home", return_value=root / "home"), \
                    mock.patch("subprocess.Popen", return_value=mock.Mock(wait=mock.Mock(return_value=0))) as popen:
                live.kernel_check(lake, root, False)
                arguments, options = popen.call_args
                self.assertEqual(arguments[0], [str(lake), "build"])
                self.assertEqual(options["cwd"], root)
                self.assertEqual(options["env"]["ELAN_HOME"], str(root / "home" / ".elan"))
                self.assertEqual(options["env"]["KEPT_SETTING"], "present")
                self.assertNotIn("ELAN_HOME", os.environ)
                self.assertEqual(live._lake_environment()["ELAN_HOME"], str(root / "home" / ".elan"))
                self.assertNotIn("ELAN_HOME", os.environ)
            self.assertIn("exit 0", (root / "kernel_check.log").read_text(encoding="utf-8"))

    def test_explicit_home_is_preserved_for_build_and_fta_check(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            custom = str(root / "configured-elan")
            with mock.patch.dict(os.environ, {"ELAN_HOME": custom}), \
                    mock.patch("subprocess.Popen", return_value=mock.Mock(wait=mock.Mock(return_value=0))) as popen, \
                    mock.patch("subprocess.run", return_value=subprocess.CompletedProcess([], 0)) as run:
                live.kernel_check(root / "lake.exe", root, True)
                self.assertEqual((popen.call_count, run.call_count), (1, 1))
                for call in (popen.call_args, run.call_args):
                    self.assertEqual(call.kwargs["env"]["ELAN_HOME"], custom)
                    self.assertEqual(call.kwargs["cwd"], root)
                self.assertEqual(run.call_args.args[0][1:3], ["env", "lean"])
                self.assertEqual(live._lake_environment()["ELAN_HOME"], custom)
                self.assertEqual(os.environ["ELAN_HOME"], custom)
            log = (root / "kernel_check.log").read_text(encoding="utf-8")
            self.assertEqual([line for line in log.splitlines() if line.startswith(("$", "exit"))],
                             [f"$ {root / 'lake.exe'} build", "exit 0",
                              f"$ {root / 'lake.exe'} env lean {os.path.join('GLExport', 'Generated', 'FTA.lean')}",
                              "exit 0"])

    def test_kernel_failure_still_fails_the_run_and_is_logged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch("subprocess.Popen", return_value=mock.Mock(wait=mock.Mock(return_value=7))):
                with self.assertRaisesRegex(RuntimeError, "Lean kernel check failed"):
                    live.kernel_check(root / "lake.exe", root, False)
            self.assertIn("exit 7", (root / "kernel_check.log").read_text(encoding="utf-8"))

    def test_background_check_logs_its_output_then_its_exit_code(self) -> None:
        """A fake Lake: `python build` runs the project's `build` script."""
        for code, passes in ((0, True), (5, False)):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / "build").write_text(f"print('Build completed successfully')\nraise SystemExit({code})\n",
                                            encoding="utf-8")
                check = live.start_kernel_check(Path(sys.executable), root)
                self.assertEqual(check.log, root / live.KERNEL_LOG)
                with mock.patch("sys.stdout"):
                    if passes:
                        live.finish_kernel_check(check)
                    else:
                        with self.assertRaisesRegex(RuntimeError, "Lean kernel check failed"):
                            live.finish_kernel_check(check)
                lines = check.log.read_text(encoding="utf-8").splitlines()
                self.assertEqual(lines, [f"$ {Path(sys.executable)} build", "Build completed successfully", f"exit {code}"])

    def test_export_folder_keeps_lake_build_output_unless_fresh(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            export = out / live.EXPORT_FOLDER
            for fresh in (False, True):
                (export / live.LAKE_BUILD_FOLDER / "build").mkdir(parents=True, exist_ok=True)
                (export / live.LAKE_BUILD_FOLDER / "build" / "trace").write_text("kept", encoding="utf-8")
                (export / "certificates").mkdir(exist_ok=True)
                (export / "selection_fta.json").write_text("{}", encoding="utf-8")
                self.assertEqual(live._fresh_export_directory(out, fresh), export)
                self.assertEqual((export / live.LAKE_BUILD_FOLDER).exists(), not fresh)
                self.assertFalse((export / "certificates").exists())
                self.assertFalse((export / "selection_fta.json").exists())
                for relative in live.PROJECT_SKELETON:
                    self.assertTrue((export / relative).is_file(), relative)

    def test_main_export_is_written_without_running_the_kernel_check(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            out, proc = root / "full", root / "proc"
            proc.mkdir()
            blocks = {anchor: [index] for index, anchor in enumerate(sorted(live.MAIN_ANCHORS))}
            certificate = {"corpus": {"coverage": "complete"}}
            written = []
            with mock.patch.object(live, "find_lake", return_value=root / "lake.exe"), \
                    mock.patch.object(live, "load_theorem_records", return_value=[]), \
                    mock.patch.object(live, "theorem_anchor_indices", return_value=blocks), \
                    mock.patch.object(live, "sha256_file", return_value="0" * 64), \
                    mock.patch.object(live, "canonical_sources", return_value=[]), \
                    mock.patch.object(live, "parsed_sources", return_value={}), \
                    mock.patch.object(live, "imports_outside_sources", return_value=[]), \
                    mock.patch.object(live, "build_certificate", return_value=certificate), \
                    mock.patch.object(live, "write_certificate", side_effect=lambda _, path: written.append(path.parent.name)), \
                    mock.patch.object(live, "write_lean"), \
                    mock.patch("proof_export.nat.write_nat_model") as nat, \
                    mock.patch("subprocess.Popen") as popen, mock.patch("subprocess.run") as run, \
                    mock.patch("sys.stdout"):
                export = live.write_main_export(proc, out, root / "config", root / "binaries")
            self.assertEqual(export, out / live.EXPORT_FOLDER)
            self.assertEqual(written, ["peano", "gauss", "fta"])
            nat.assert_called_once()
            popen.assert_not_called()
            run.assert_not_called()
            selection = (export / "selection_fta.json").read_text(encoding="utf-8")
            self.assertTrue(selection.startswith('{"schema_version":3,"id":"fta_live_1",'))
            with mock.patch.object(live, "find_lake", return_value=None), mock.patch("sys.stdout"):
                self.assertIsNone(live.write_main_export(proc, out, root / "config", root / "binaries"))
            self.assertFalse(export.exists())

    def test_kernel_failure_fails_under_optimized_python(self) -> None:
        """`python -O` removes every `assert`; a failed build must still raise."""
        code = ("import subprocess, sys\n"
                "from pathlib import Path\n"
                "from unittest import mock\n"
                "from proof_export import live\n"
                "with mock.patch('subprocess.Popen', return_value=mock.Mock(wait=mock.Mock(return_value=7))):\n"
                "    live.kernel_check(Path('lake.exe'), Path(sys.argv[1]), False)\n")
        with tempfile.TemporaryDirectory() as temporary:
            completed = subprocess.run(
                [sys.executable, "-O", "-c", code, temporary],
                cwd=Path(__file__).resolve().parent.parent, capture_output=True, text=True)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("Lean kernel check failed", completed.stderr)
            self.assertNotIn("kernel check passed", completed.stdout)
            self.assertIn("exit 7", (Path(temporary) / "kernel_check.log").read_text(encoding="utf-8"))


SOURCE_DEFINITIONS = {
    "existence3": {"arity": 3, "category": "existence", "elements": ["!(>[y1](in[y1,x1])!(in2[x2,y1,x3]))"]},
    "or1": {"arity": 3, "category": "or", "elements": ["(in[x1,x2])", "(existence3[x2,x1,x3])"]},
    "in": {"arity": 2, "category": "atomic", "elements": []},
}
TARGET_DEFINITIONS = {
    "existence11": {"arity": 3, "category": "existence", "elements": ["!(>[y1](in[y1,x1])!(in2[x2,y1,x3]))"]},
    "or2": {"arity": 3, "category": "or", "elements": ["(in[x1,x2])", "(existence11[x2,x1,x3])"]},
    "or5": {"arity": 3, "category": "or", "elements": ["(in[x1,x2])", "(in[x2,x3])"]},
    "in": {"arity": 2, "category": "atomic", "elements": []},
}


class HeadAliasTests(unittest.TestCase):
    def test_nested_renamed_definitions_resolve_bottom_up(self) -> None:
        target = parse_mpl(PEANO + "(>[w1](in[w1,N])(or2[w1,i0,s])))")
        self.assertEqual(
            live.infer_head_aliases(target, SOURCE_DEFINITIONS, TARGET_DEFINITIONS),
            {"existence11": "existence3", "or2": "or1"},
        )

    def test_identical_names_need_no_alias(self) -> None:
        target = parse_mpl(PEANO + "(>[w1](in[w1,N])(existence3[w1,i0,s])))")
        definitions = dict(TARGET_DEFINITIONS, existence3=SOURCE_DEFINITIONS["existence3"])
        self.assertEqual(live.infer_head_aliases(target, SOURCE_DEFINITIONS, definitions), {})

    def test_head_without_twin_yields_none(self) -> None:
        target = parse_mpl(PEANO + "(>[w1](in[w1,N])(or5[w1,i0,s])))")
        self.assertIsNone(live.infer_head_aliases(target, SOURCE_DEFINITIONS, TARGET_DEFINITIONS))


class AdaptationSearchTests(unittest.TestCase):
    SOURCE = PEANO + "(>[w1,w2,w3](in3[w1,w2,w3,+])(>[w4,w5](in3[w4,w2,w5,+])(in3[w5,w1,w4,+]))))"
    PERMUTED = PEANO + "(>[v4,v5](in3[v4,v2,v5,+])(>[v1,v2,v3](in3[v1,v2,v3,+])(in3[v5,v1,v4,+]))))"
    DIFFERENT = PEANO + "(>[v4,v5](in3[v4,v2,v5,+])(>[v1,v2,v3](in3[v1,v2,v3,+])(in3[v5,v1,v4,*]))))"

    def test_premise_permutation_matches_and_a_different_head_does_not(self) -> None:
        source = parse_mpl(self.SOURCE)
        solutions = theorem_adaptation_solutions(
            source, parse_mpl(self.PERMUTED), SOURCE_DEFINITIONS, TARGET_DEFINITIONS, {},
        )
        self.assertEqual(len(solutions), 1)
        self.assertEqual(
            theorem_adaptation_solutions(
                source, parse_mpl(self.DIFFERENT), SOURCE_DEFINITIONS, TARGET_DEFINITIONS, {},
            ),
            [],
        )

    def test_premise_count_mismatch_is_a_negative_answer_not_an_assert(self) -> None:
        source = parse_mpl(self.SOURCE)
        shorter = parse_mpl(PEANO + "(>[v1,v2,v3](in3[v1,v2,v3,+])(in3[v3,v1,v2,+])))")
        self.assertEqual(
            theorem_adaptation_solutions(source, shorter, SOURCE_DEFINITIONS, TARGET_DEFINITIONS, {}),
            [],
        )


class BaseFormEquivalenceTests(unittest.TestCase):
    """The shortcut's or2[w1,i0,N,s] is Peano's or1[N,v1,s,i0]: permuted
    parameters, swapped disjuncts, renamed existence compact."""

    PEANO_DEFINITIONS = {
        "existence3": {"arity": 3, "category": "existence", "elements": ["(in[1,u_1])", "(in2[1,u_2,u_3])"]},
        "or1": {"arity": 4, "category": "or", "elements": ["(existence3[u_1,u_2,u_3])", "(=[u_2,u_4])"]},
        "AnchorPeano": {"arity": 6, "category": "and", "elements": ["(NaturalNumbers[u_1,u_2,u_3,u_4,u_5])", "(in2[u_2,u_6,u_3])"]},
    }
    FTA_DEFINITIONS = {
        "existence11": {"arity": 3, "category": "existence", "elements": ["(in[1,u_1])", "(in2[1,u_2,u_3])"]},
        "or2": {"arity": 4, "category": "or", "elements": ["(=[u_1,u_2])", "(existence11[u_3,u_1,u_4])"]},
        "or5": {"arity": 4, "category": "or", "elements": ["(=[u_1,u_2])", "(in[u_1,u_3])"]},
        "AnchorPeano": {"arity": 6, "category": "and", "elements": ["(NaturalNumbers[u_1,u_2,u_3,u_4,u_5])", "(in2[u_2,u_6,u_3])"]},
    }

    def test_renamed_reordered_disjunction_is_base_form_equivalent(self) -> None:
        from proof_export.certificate import base_form_equivalent
        source = parse_mpl(PEANO + "(>[v1](in[v1,N])(or1[N,v1,s,i0])))")
        target = parse_mpl(PEANO + "(>[w1](in[w1,N])(or2[w1,i0,N,s])))")
        self.assertTrue(base_form_equivalent(source, target, self.PEANO_DEFINITIONS, self.FTA_DEFINITIONS))
        other = parse_mpl(PEANO + "(>[w1](in[w1,N])(or5[w1,i0,N,s])))")
        self.assertFalse(base_form_equivalent(source, other, self.PEANO_DEFINITIONS, self.FTA_DEFINITIONS))

    def test_shared_heads_compare_by_name_and_premises_may_permute(self) -> None:
        from proof_export.certificate import base_form_equivalent
        source = parse_mpl(AdaptationSearchTests.SOURCE)
        self.assertTrue(base_form_equivalent(
            source, parse_mpl(AdaptationSearchTests.PERMUTED), self.PEANO_DEFINITIONS, self.FTA_DEFINITIONS))
        self.assertFalse(base_form_equivalent(
            source, parse_mpl(AdaptationSearchTests.DIFFERENT), self.PEANO_DEFINITIONS, self.FTA_DEFINITIONS))


class ExternalTheoremListTests(unittest.TestCase):
    """The externals snapshot becomes the FTA export's only dependency."""

    PEANO_ROW = "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7](in[7,1])!(&!(=[7,2])(>[8](in[8,1])!(in2[8,7,3])))))"
    GAUSS_ROW = "(>[1,2,3,4,5,6,7,8](AnchorGauss[1,2,3,4,5,6,7,8])(>[9,10](in2[9,10,3])(in3[9,10,6,4])))"

    def test_rows_split_by_anchor_into_certificate_shaped_lists(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            externals = root / "externally_provided_theorems.txt"
            externals.write_text(self.PEANO_ROW + "\n\n" + self.GAUSS_ROW + "\n", encoding="utf-8")
            export = root / "lean_export"
            documents = live.external_theorem_lists(externals, export)
            self.assertEqual([identifier for identifier, _ in documents], ["peano_externals_1", "gauss_externals_1"])
            peano = documents[0][1]
            self.assertEqual(peano["corpus"]["id"], "peano_externals_1")
            self.assertEqual(peano["theorems"][0]["id"], "peano_external_000")
            self.assertEqual(peano["theorems"][0]["theorem"]["mpl"], self.PEANO_ROW)
            self.assertEqual(peano["binary_definitions"], {})
            self.assertTrue((export / "certificates" / "peano_externals" / "certificate.json").is_file())
            self.assertTrue((export / "certificates" / "gauss_externals" / "certificate.json").is_file())

    def test_a_compact_citation_resolves_to_its_base_form_row(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            externals = root / "externally_provided_theorems.txt"
            externals.write_text(self.PEANO_ROW + "\n", encoding="utf-8")
            documents = live.external_theorem_lists(externals, root / "lean_export")
            cited = PEANO + "(>[w1](in[w1,N])(or2[w1,i0,N,s])))"
            citer = _theorem_with_citation(0, FTA + "(in[i1,N]))", cited)
            references = live.external_references(
                [citer], [citer], documents, BaseFormEquivalenceTests.FTA_DEFINITIONS,
            )
            self.assertEqual(references["peano_externals_1"]["required_source_indices"], [0])
            self.assertEqual(
                references["peano_externals_1"]["theorem_adaptations"],
                [{"source_index": 0, "target_theorem": cited, "head_aliases": {}, "base_form": True}],
            )
            self.assertEqual(references["gauss_externals_0"], {"required_source_indices": [], "theorem_adaptations": []})


class ExternalReferenceTests(unittest.TestCase):
    CITED_EXACT = PEANO + "(>[w1](in[w1,N])(in2[w1,w1,s])))"
    CITED_PERMUTED = AdaptationSearchTests.PERMUTED

    def _dependency(self) -> tuple[str, dict[str, object]]:
        return (
            "peano_live_2",
            {
                "theorems": [
                    {"id": "peano_source_000", "source_index": 0, "theorem": {"mpl": self.CITED_EXACT}},
                    {"id": "peano_source_001", "source_index": 1, "theorem": {"mpl": AdaptationSearchTests.SOURCE}},
                ],
                "binary_definitions": SOURCE_DEFINITIONS,
            },
        )

    def test_exact_then_adapted_citations_resolve_by_content(self) -> None:
        exact_citer = _theorem_with_citation(0, FTA + "(in[i1,N]))", self.CITED_EXACT)
        permuted_citer = _theorem_with_citation(1, FTA + "(in[i2,N]))", self.CITED_PERMUTED)
        references = live.external_references(
            [exact_citer, permuted_citer],
            [exact_citer, permuted_citer],
            [self._dependency()],
            TARGET_DEFINITIONS,
        )
        self.assertEqual(references["peano_live_2"]["required_source_indices"], [0, 1])
        self.assertEqual(
            references["peano_live_2"]["theorem_adaptations"],
            [{"source_index": 1, "target_theorem": self.CITED_PERMUTED, "head_aliases": {}}],
        )

    def test_local_citations_are_not_external(self) -> None:
        local = FTA + "(in[i1,N]))"
        citer = _theorem_with_citation(1, FTA + "(in[i2,N]))", local)
        references = live.external_references(
            [citer], [_record(0, local), citer], [self._dependency()], TARGET_DEFINITIONS,
        )
        self.assertEqual(references["peano_live_2"], {"required_source_indices": [], "theorem_adaptations": []})

    def test_unresolvable_citation_asserts(self) -> None:
        citer = _theorem_with_citation(0, FTA + "(in[i2,N]))", AdaptationSearchTests.DIFFERENT)
        with self.assertRaises(AssertionError):
            live.external_references([citer], [citer], [self._dependency()], TARGET_DEFINITIONS)

    def test_or_theorem_companion_citation_is_dropped_when_unresolvable(self) -> None:
        # D-217: the head-switched companion of a single-direction or theorem
        # may never have been proved on its own; the builder drops it.
        companion = FTA + "(>[w1](in[w1,N])(>[]!(in[w1,N])(in[w1,N]))))"
        rows = (
            ProofRow(
                1, _scoped(FTA + "(in[i2,N]))"), "main", "or theorem",
                (Dependency(_scoped(companion), "main"),),
            ),
        )
        citer = TheoremRecord(
            0, parse_mpl(FTA + "(in[i2,N]))"), "or theorem", "-1",
            (Chapter(Path("0_or_theorem.txt"), 0, "or_theorem", rows),),
        )
        references = live.external_references([citer], [citer], [self._dependency()], TARGET_DEFINITIONS)
        self.assertEqual(references["peano_live_2"], {"required_source_indices": [], "theorem_adaptations": []})


class ConstructedOrDefinitionTests(unittest.TestCase):
    """OR theorems sharing one compact head: one operator, per-theorem head arguments."""

    OR_A = "(>[a,b,c,d](in[a,S])(or0[a,b,c,d]))"
    OR_A_PARENTS = (
        "(>[a,b,c,d](in[a,S])(>[]!(=[a,b])!(=[c,d])))",
        "(>[a,b,c,d](in[a,S])(>[](=[c,d])(=[a,b])))",
    )
    # The same operator under different bound-variable names.
    OR_B = "(>[p,q,r,s](in[p,S])(or0[p,q,r,s]))"
    OR_B_PARENTS = (
        "(>[p,q,r,s](in[p,S])(>[]!(=[p,q])!(=[r,s])))",
        "(>[p,q,r,s](in[p,S])(>[](=[r,s])(=[p,q])))",
    )
    # The same head with a different operator: both disjuncts are equalities.
    OR_C = "(>[e,f,g,h](in[e,S])(or0[e,f,g,h]))"
    OR_C_PARENTS = (
        "(>[e,f,g,h](in[e,S])(>[]!(=[e,f])(=[g,h])))",
        "(>[e,f,g,h](in[e,S])(>[]!(=[g,h])(=[e,f])))",
    )

    @staticmethod
    def _or_record(directory: Path, index: int, theorem: str, parents: tuple[str, str]) -> TheoremRecord:
        path = directory / f"{index}_or_theorem.txt"
        path.write_text(
            "\t".join([theorem, "main", "or theorem", parents[0], "main", parents[1], "main"]) + "\n",
            encoding="utf-8",
        )
        return TheoremRecord(index, parse_mpl(theorem), "or theorem", "-1",
                             (Chapter(path, index, "or_theorem", ()),))

    def _overrides(self, theorems):
        from proof_export.certificate import binary_entries_with_overrides
        with tempfile.TemporaryDirectory(prefix="or_definitions_") as directory:
            root = Path(directory)
            records = tuple(
                self._or_record(root, index, theorem, parents)
                for index, (theorem, parents) in enumerate(theorems)
            )
            binary_path = root / "GL_binary.json"
            binary_path.write_text("{}", encoding="utf-8")
            return binary_entries_with_overrides(records, [], binary_path)

    def test_shared_head_with_different_variable_names_is_one_operator(self) -> None:
        constructed, overrides, binary_entries = self._overrides(
            [(self.OR_A, self.OR_A_PARENTS), (self.OR_B, self.OR_B_PARENTS)])
        self.assertEqual(set(constructed), {"or0"})
        self.assertEqual(constructed["or0"]["elements"], ["(=[u_1,u_2])", "!(=[u_3,u_4])"])
        self.assertEqual(constructed["or0"]["arguments"], ["a", "b", "c", "d"])
        self.assertEqual(binary_entries["or0"]["signature"], "(or0[u_1,u_2,u_3,u_4])")
        self.assertIs(overrides["or0"], constructed["or0"])

    def test_shared_head_with_different_operators_asserts(self) -> None:
        with self.assertRaises(AssertionError):
            self._overrides([(self.OR_A, self.OR_A_PARENTS), (self.OR_C, self.OR_C_PARENTS)])

    def test_or_theorem_step_renders_its_own_head_arguments(self) -> None:
        from proof_export.certificate import _chapter_certificate
        from proof_export.graph import parse_scoped_expression
        from proof_export.mpl import alpha_key
        constructed, _, _ = self._overrides(
            [(self.OR_A, self.OR_A_PARENTS), (self.OR_B, self.OR_B_PARENTS)])
        with tempfile.TemporaryDirectory(prefix="or_step_") as directory:
            path = Path(directory) / "1_or_theorem.txt"
            line = "\t".join([self.OR_B, "main", "or theorem",
                              self.OR_B_PARENTS[0], "main", self.OR_B_PARENTS[1], "main"])
            path.write_text(line + "\n", encoding="utf-8")
            row = ProofRow(1, parse_scoped_expression(self.OR_B), "main", "or theorem", tuple(
                Dependency(parse_scoped_expression(parent), "main") for parent in self.OR_B_PARENTS))
            selected = {
                alpha_key(parse_mpl(parent)): [{"source_index": index}]
                for index, parent in enumerate(self.OR_B_PARENTS, start=10)
            }
            certificate = _chapter_certificate(
                Chapter(path, 1, "or_theorem", (row,)),
                {"in": ("element", "set"), "or0": ("element", "element", "element", "element")},
                selected, constructed)
        definition = certificate["steps"][-1]["constructed_or_definition"]
        self.assertEqual(definition["arguments"], ["p", "q", "r", "s"])
        self.assertEqual(definition["elements"], constructed["or0"]["elements"])
        self.assertEqual(constructed["or0"]["arguments"], ["a", "b", "c", "d"])


if __name__ == "__main__":
    unittest.main()
