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
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proof_export import live  # noqa: E402
from proof_export.certificate import theorem_adaptation_solutions  # noqa: E402
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


def _record(index: int, mpl: str, method: str = "direct") -> TheoremRecord:
    return TheoremRecord(index, parse_mpl(mpl), method, "-1", ())


def _scoped(mpl: str) -> ScopedExpression:
    return ScopedExpression(parse_mpl(mpl), mpl, "ordinary")


def _theorem_with_citation(index: int, mpl: str, cited_mpl: str) -> TheoremRecord:
    """A direct theorem whose single chapter cites one previously proved theorem."""

    rows = (
        ProofRow(1, _scoped(cited_mpl), "main", "theorem", ()),
        ProofRow(
            2, _scoped(mpl), "main", "implication",
            (Dependency(_scoped(cited_mpl), "main"),),
        ),
    )
    return TheoremRecord(
        index, parse_mpl(mpl), "direct", "-1",
        (Chapter(Path("0_direct_proof.txt"), 0, "direct_proof", rows),),
    )


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
        with tempfile.TemporaryDirectory() as temporary:
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


if __name__ == "__main__":
    unittest.main()
