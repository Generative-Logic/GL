# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt)
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Direct unit tests for the ordinary-type GL-to-Lean proof exporter."""

from __future__ import annotations

import copy
import json
import re
import tempfile
import unittest
from pathlib import Path

from proof_export.certificate import (
    _constructed_or_definition,
    _defined_output_totality,
    _schema3_action_metadata,
    _validate_theorem_adaptation,
    build_certificate,
    write_certificate,
)
from proof_export.graph import (
    Chapter,
    Dependency,
    ProofRow,
    ScopedExpression,
    TheoremRecord,
    parse_scoped_expression,
)
from proof_export.lean import (
    _assert_manifest_coverage,
    _context_application,
    _context_signature,
    _definition_body,
    _definition_order,
    _external_reference_interfaces,
    _helper_free_tokens,
    _render_reformulated_theorem,
    _render_scoped_proposition,
    _scope_entry_and_specializations,
    _scoped_dependency,
    _scoped_expression,
    lean_name,
    render_definitions,
    render_expression,
    render_gauss_theory,
    render_peano_theory,
    write_lean,
)
from proof_export.mpl import parse_mpl, to_mpl
from proof_export.types import (
    BINARY_RELATION,
    ELEMENT,
    SET,
    TERNARY_RELATION,
    load_type_environment,
)


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
LEAN_FIXTURES = REPOSITORY_ROOT / "tests" / "fixtures" / "lean_certificates"
FULL_CERTIFICATE = LEAN_FIXTURES / "lean_full_65.json"
GAUSS_CERTIFICATE = LEAN_FIXTURES / "lean_gauss_main_29.json"
LIVE_MAIN_EXPORT = REPOSITORY_ROOT / "files" / "full_proof_graph" / "lean_export"
LIVE_SHORTCUT_EXPORT = (
    REPOSITORY_ROOT / "files" / "shortcut" / "full_proof_graph" / "lean_export"
)
FTA_CERTIFICATE = LIVE_SHORTCUT_EXPORT / "certificates" / "fta" / "certificate.json"
live_exports_present = unittest.skipUnless(
    FTA_CERTIFICATE.is_file()
    and (LIVE_MAIN_EXPORT / "GLExport" / "Generated" / "Peano.lean").is_file(),
    "no run Lean export on disk (run main.py and main.py --shortcut with Lean installed)",
)
PEANO_CONFIG = REPOSITORY_ROOT / "files" / "config" / "ConfigPeano.json"
PEANO_BINARY = (
    REPOSITORY_ROOT
    / "tests"
    / "fixtures"
    / "GL_binaries"
    / "GL_binary_Peano.json"
)


class NeutralFrontendTests(unittest.TestCase):
    """Exercise graph loading, typing, certification, and stable serialization."""

    @live_exports_present
    def test_synthetic_selection_builds_deterministically_without_gl(self) -> None:
        theorem = (
            "(>[N,i0,s,+,*,i1](AnchorPeano[N,i0,s,+,*,i1])"
            "(AnchorPeano[N,i0,s,+,*,i1]))"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            graph = root / "graph"
            graph.mkdir()
            (graph / "global_theorem_list.txt").write_text(
                f"{theorem}\tdirect\t-1\n",
                encoding="utf-8",
            )
            (graph / "0_direct_proof.txt").write_text(
                "(AnchorPeano[N,i0,s,+,*,i1])\tmain\ttask formulation\n",
                encoding="utf-8",
            )
            selection = root / "selection.json"
            selection.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "theorems": [
                            {
                                "id": "synthetic_anchor_identity",
                                "name": "Synthetic anchor identity",
                                "method": "direct",
                                "self_contained": True,
                                "theorem": theorem,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            first = build_certificate(selection, graph, PEANO_CONFIG, PEANO_BINARY)
            second = build_certificate(selection, graph, PEANO_CONFIG, PEANO_BINARY)
            self.assertEqual(first, second)
            self.assertEqual(first["schema_version"], 2)
            self.assertEqual(first["theorems"][0]["chapters"][0]["steps"][0]["action"], "assume")
            first_path = root / "first.json"
            second_path = root / "second.json"
            write_certificate(first, first_path)
            write_certificate(second, second_path)
            self.assertEqual(first_path.read_bytes(), second_path.read_bytes())


def _full_certificate() -> dict[str, object]:
    """
    @brief Load the frozen acyclic Peano fixture certificate.
    @details The fixture records the excluded-pair era of the Peano corpus and
    avoids any dependency on a live GL run.
    @return Parsed schema-2 certificate.
    """

    return json.loads(FULL_CERTIFICATE.read_text(encoding="utf-8"))


def _gauss_certificate() -> dict[str, object]:
    """
    @brief Load the schema-3 Gauss main fixture certificate.
    @details The fixture includes its pinned Peano dependency closure and needs
    no processed-proof regeneration or GL execution during renderer tests.
    @return Parsed schema-3 Gauss certificate.
    """

    return json.loads(GAUSS_CERTIFICATE.read_text(encoding="utf-8"))


def _fta_certificate() -> dict[str, object]:
    """
    @brief Load the FTA certificate the last shortcut run exported.
    @details The live export under `files/shortcut/full_proof_graph/lean_export/`
    is read as written by `proof_export.live`; tests using it are skipped when
    no shortcut run with a Lean toolchain has been made on this host.
    @return Parsed schema-3 FTA shortlist certificate.
    """

    return json.loads(FTA_CERTIFICATE.read_text(encoding="utf-8"))


class LeanOrdinaryTypeTests(unittest.TestCase):
    """Exercise the finite ordinary-type semantic encoding."""

    def test_nullary_compiled_fact_fixes_the_carrier_argument(self) -> None:
        rendered = render_expression(parse_mpl("(implication164[])"), {})
        self.assertEqual(rendered, "(gl_implication164 (α := α))")

    def test_peano_context_has_only_one_carrier_and_predicate_types(self) -> None:
        environment = load_type_environment(PEANO_CONFIG, PEANO_BINARY)
        self.assertEqual(
            environment.signatures["AnchorPeano"],
            (SET, ELEMENT, BINARY_RELATION, TERNARY_RELATION, TERNARY_RELATION, ELEMENT),
        )
        definitions = render_definitions(_full_certificate())
        self.assertIn("{α : Type u}", definitions)
        self.assertIn("(u_1 : GLSet α)", definitions)
        self.assertIn("(u_3 : GLBinaryRelation α)", definitions)
        self.assertNotIn("α → Type", definitions)
        self.assertNotIn("Sort", definitions)

    def test_typed_expression_rendering_uses_predicates_not_indexed_types(self) -> None:
        expression = parse_mpl("(>[v1](in[v1,N])(=[v1,i0]))")
        self.assertEqual(to_mpl(expression), "(>[v1](in[v1,N])(=[v1,i0]))")
        self.assertEqual(
            render_expression(
                expression,
                {"v1": ELEMENT, "N": SET, "i0": ELEMENT},
                {"N": "N", "i0": "zero"},
            ),
            "(∀ (v1 : α), ((N v1) → (v1 = zero)))",
        )

    def test_identifier_and_definition_rendering_are_deterministic(self) -> None:
        self.assertEqual(lean_name("+"), "add")
        self.assertEqual(lean_name("1"), "x_1")
        self.assertEqual(lean_name("theorem"), "glv_theorem")
        body = _definition_body(
            "sample",
            {
                "category": "implication",
                "signature": [SET],
                "elements": ["(in[1,u_1])", "(=[1,1])"],
            },
            {"in": [ELEMENT, SET], "=": [ELEMENT, ELEMENT]},
        )
        self.assertEqual(body, "(∀ (x_1 : α), ((u_1 x_1) → (x_1 = x_1)))")

    def test_definition_cycles_are_rejected(self) -> None:
        with self.assertRaisesRegex(AssertionError, "cyclic GL binary definition closure"):
            _definition_order(
                {
                    "left": {"elements": ["(right[u_1])"]},
                    "right": {"elements": ["(left[u_1])"]},
                }
            )

    def test_proof_support_existence_elimination_is_explicit(self) -> None:
        source = (
            REPOSITORY_ROOT / "lean_export" / "GLExport" / "ProofSupport.lean"
        ).read_text(encoding="utf-8")
        self.assertNotIn("grind", source)
        self.assertIn("apply Classical.byContradiction", source)
        self.assertIn("exact ⟨value, p_value, q_value⟩", source)

    def test_proof_support_or_bridge_is_explicit(self) -> None:
        source = (
            REPOSITORY_ROOT / "lean_export" / "GLExport" / "ProofSupport.lean"
        ).read_text(encoding="utf-8")
        self.assertIn("theorem orIffNotAndNot", source)
        self.assertIn("cases disjunction with", source)
        self.assertIn("no_disjunction (Or.inr right_value)", source)
        self.assertNotIn("grind", source)

    def test_helper_free_tokens_prefer_original_over_copy_alias(self) -> None:
        chapter = {
            "steps": [
                {
                    "action": "task_formulation",
                    "conclusion": {"mpl": "(=[v1,v1_copy])"},
                    "dependencies": [],
                }
            ],
            "variable_types": {"v1": ELEMENT, "v1_copy": ELEMENT},
        }
        self.assertEqual(
            _helper_free_tokens(chapter, {"v1_copy": "v1"}),
            ["v1"],
        )


class ConstructedOrDefinitionTests(unittest.TestCase):
    """Reconstruct OR operators from processed rows, negated disjuncts included."""

    @staticmethod
    def _or_theorem_record(head: str, parent_a: str, parent_b: str) -> TheoremRecord:
        """
        @brief Build a minimal one-row OR theorem record from MPL strings.
        @details Mirrors the processed-graph shape: one chapter, one
        ``or theorem`` row citing the two parent theorems.
        @param head Constructed OR theorem MPL.
        @param parent_a First cited parent theorem MPL.
        @param parent_b Second cited parent (companion) theorem MPL.
        @return Parsed theorem record with method ``or theorem``.
        """

        row = ProofRow(
            1,
            parse_scoped_expression(head),
            "main",
            "or theorem",
            (
                Dependency(parse_scoped_expression(parent_a), "main"),
                Dependency(parse_scoped_expression(parent_b), "main"),
            ),
        )
        chapter = Chapter(Path("0_or_theorem.txt"), 0, "or_theorem", (row,))
        return TheoremRecord(
            0,
            parse_scoped_expression(head).expression,
            "or theorem",
            "-1",
            (chapter,),
        )

    def test_negated_disjunct_keeps_polarity(self) -> None:
        # The live Peano or0 pair: second disjunct is NEGATED, so the
        # companion's premise is the double-negation-cancelled PLAIN atom.
        # A blind negation-node requirement rejects exactly this shape.
        head = (
            "(>[N,i0,s,+,*,i1](AnchorPeano[N,i0,s,+,*,i1])"
            "(>[v1,v2](in2[v1,v2,s])(>[v3,v4](in2[v3,v4,s])(or0[v2,v4,v1,v3]))))"
        )
        parent_a = (
            "(>[N,i0,s,+,*,i1](AnchorPeano[N,i0,s,+,*,i1])"
            "(>[w1,w2](in2[w1,w2,s])(>[w3,w4](in2[w3,w4,s])"
            "(>[]!(=[w2,w4])!(=[w1,w3])))))"
        )
        parent_b = (
            "(>[N,i0,s,+,*,i1](AnchorPeano[N,i0,s,+,*,i1])"
            "(>[w1,w2](in2[w1,w2,s])(>[w3,w4](in2[w3,w4,s])"
            "(>[](=[w1,w3])(=[w2,w4])))))"
        )
        definition = _constructed_or_definition(
            self._or_theorem_record(head, parent_a, parent_b)
        )
        self.assertEqual(definition["head"], "or0")
        self.assertEqual(definition["elements"], ["(=[u_1,u_2])", "!(=[u_3,u_4])"])
        self.assertEqual(definition["signature"], "(or0[u_1,u_2,u_3,u_4])")

    def test_positive_disjuncts_still_reconstruct(self) -> None:
        # All-positive binary OR: both parents carry literal negation premises.
        head = (
            "(>[N,i0,s,+,*,i1](AnchorPeano[N,i0,s,+,*,i1])"
            "(>[v1](in[v1,N])(or1[v1,i0,s])))"
        )
        parent_a = (
            "(>[N,i0,s,+,*,i1](AnchorPeano[N,i0,s,+,*,i1])"
            "(>[w1](in[w1,N])(>[]!(=[w1,i0])(in2[i0,w1,s]))))"
        )
        parent_b = (
            "(>[N,i0,s,+,*,i1](AnchorPeano[N,i0,s,+,*,i1])"
            "(>[w1](in[w1,N])(>[]!(in2[i0,w1,s])(=[w1,i0]))))"
        )
        definition = _constructed_or_definition(
            self._or_theorem_record(head, parent_a, parent_b)
        )
        self.assertEqual(definition["head"], "or1")
        self.assertEqual(
            definition["elements"], ["(=[u_1,u_2])", "(in2[u_2,u_1,u_3])"]
        )


class LeanFullCertificateTests(unittest.TestCase):
    """Exercise complete theorem rendering and the strict trust boundary."""

    def test_full_certificate_is_exactly_the_acyclic_peano_corpus(self) -> None:
        certificate = _full_certificate()
        self.assertEqual(certificate["schema_version"], 2)
        self.assertEqual(
            certificate["corpus"]["coverage"],
            {
                "actions": 3247,
                "chapters": 143,
                "methods": {"direct": 22, "induction": 39, "or theorem": 4},
                "rows": 3247,
                "theorems": 65,
            },
        )
        self.assertEqual(
            {theorem["source_index"] for theorem in certificate["theorems"]},
            set(range(67)) - {24, 25},
        )
        self.assertEqual(
            [item["source_index"] for item in certificate["excluded_sources"]],
            [24, 25],
        )

    def test_full_render_is_deterministic_row_total_and_admission_free(self) -> None:
        certificate = _full_certificate()
        first_source, first_manifest = render_peano_theory(certificate)
        second_source, second_manifest = render_peano_theory(certificate)
        self.assertEqual(first_source, second_source)
        self.assertEqual(first_manifest, second_manifest)
        self.assertEqual(first_source.count("GL tag "), 3247)
        self.assertEqual(first_source.count("\ntheorem peano_"), 65)
        self.assertIn("have step_induction_assumption_", first_source)
        self.assertIn("exact typingRule", first_source)
        self.assertIn("exact zeroRule", first_source)
        self.assertIn("exact induction_successor", first_source)
        self.assertIn("apply induction_hypothesis", first_source)
        self.assertIn("exact inductionProperty", first_source)
        self.assertNotIn("grind", first_source)
        ordered = _assert_manifest_coverage(certificate, first_manifest)
        self.assertEqual(len(ordered), 65)
        for forbidden in ("sorry", "admit", "axiom ", "opaque ", "unsafe "):
            self.assertNotIn(forbidden, first_source.lower())

    def test_writer_emits_exact_manifest_and_rejects_a_trust_shortcut(self) -> None:
        certificate = _full_certificate()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            write_lean(certificate, output)
            manifest = json.loads(
                (output / "lean_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["backend"], "Lean 4 ordinary predicates")
            self.assertEqual(manifest["coverage"]["theorems"], 65)
            self.assertEqual(manifest["coverage"]["rows"], 3247)
            self.assertEqual(
                [item["source_index"] for item in manifest["excluded_sources"]],
                [24, 25],
            )

        malformed = copy.deepcopy(certificate)
        malformed["theorems"][0]["id"] = "axiom forbidden"
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(AssertionError, "forbidden trust shortcut"):
                write_lean(malformed, Path(temporary))


class LeanGaussCertificateTests(unittest.TestCase):
    """Exercise schema-3 Gauss scopes, reformulations, and total rendering."""

    def test_gauss_certificate_has_exact_main_coverage_and_constructive_outputs(self) -> None:
        certificate = _gauss_certificate()
        self.assertEqual(certificate["schema_version"], 3)
        self.assertEqual(
            certificate["corpus"]["coverage"],
            {
                "actions": 1511,
                "chapters": 33,
                "methods": {
                    "direct": 19,
                    "induction": 2,
                    "reformulated statement": 8,
                },
                "rows": 1511,
                "theorems": 29,
            },
        )
        self.assertEqual(
            {theorem["source_index"] for theorem in certificate["theorems"]},
            set(range(64, 93)),
        )
        totality = _defined_output_totality(
            "limitSet",
            certificate["binary_definitions"],
            certificate["types"],
        )
        self.assertEqual(totality["output_type"], "set")
        self.assertTrue(totality["definition_dependency_acyclic"])
        reformulated = [
            theorem
            for theorem in certificate["theorems"]
            if theorem["method"] == "reformulated statement"
        ]
        self.assertEqual(len(reformulated), 8)
        self.assertTrue(
            all(
                theorem["chapters"][0]["steps"][0]["defined_output_totality"]
                ["definition_dependency_acyclic"]
                for theorem in reformulated
            )
        )

    def test_schema3_metadata_revalidates_a_tracked_scoped_chapter(self) -> None:
        certificate = _gauss_certificate()
        theorem = next(
            theorem
            for theorem in certificate["theorems"]
            if theorem["source_index"] == 69
        )
        record = next(
            chapter for chapter in theorem["chapters"] if chapter["scope_contexts"]
        )

        def scoped(payload: dict[str, object]) -> ScopedExpression:
            return ScopedExpression(
                parse_mpl(str(payload["mpl"])),
                str(payload.get("source_mpl", payload["mpl"])),
                str(payload.get("proof_scope", "ordinary")),
            )

        rows = []
        for step in sorted(record["steps"], key=lambda item: item["source_line"]):
            dependencies = tuple(
                Dependency(scoped(dependency["expression"]), str(dependency["namespace"]))
                for dependency in step["dependencies"]
            )
            rows.append(
                ProofRow(
                    int(step["source_line"]),
                    scoped(step["conclusion"]),
                    str(step["namespace"]),
                    str(step["source_tag"]),
                    dependencies,
                )
            )
        chapter = Chapter(
            Path(str(record["source_file"])),
            int(record["number"]),
            str(record["role"]),
            tuple(rows),
        )
        metadata, contexts = _schema3_action_metadata(
            chapter,
            certificate["binary_definitions"],
            {},
            certificate["types"],
        )
        self.assertEqual(contexts, record["scope_contexts"])
        self.assertEqual(
            set(metadata),
            {
                step["source_line"]
                for step in record["steps"]
                if step["source_tag"]
                in {
                    "premise element",
                    "reformulation for integration >[bound]",
                    "reformulation for integration >[]",
                    "reformulation for integration and",
                    "validity name",
                }
            },
        )

    def test_gauss_context_and_scope_render_as_ordinary_predicates(self) -> None:
        certificate = _gauss_certificate()
        theorem = next(
            theorem
            for theorem in certificate["theorems"]
            if theorem["source_index"] == 69
        )
        chapter = next(
            chapter for chapter in theorem["chapters"] if chapter["scope_contexts"]
        )
        context = chapter["scope_contexts"][0]
        scoped = _scoped_expression(
            parse_mpl("(=[v3,v3])"),
            context,
            ("V4",),
        )
        self.assertIn(">[V4", to_mpl(scoped))
        rendered = _render_scoped_proposition(
            "(∃ (v11 : α), N v11)",
            context,
            chapter["variable_types"],
            {"N": "N", "+": "add"},
            ("V4",),
        )
        self.assertIn("∀ (V4 : GLSet α)", rendered)
        self.assertIn("∃ (v11 : α)", rendered)
        signature = "\n".join(_context_signature("AnchorGauss"))
        self.assertIn("identity : GLBinaryRelation α", signature)
        self.assertNotIn("α → Type", signature)
        self.assertIn("anchorPeanoOfGauss", _context_application("AnchorGauss", "peano_full_65"))
        self.assertIn("anchorPeanoOfGauss", _context_application("AnchorGauss", "peano_full_66"))

    def test_gauss_render_is_deterministic_row_total_and_axiom_free(self) -> None:
        certificate = _gauss_certificate()
        first_source, first_manifest = render_gauss_theory(certificate)
        second_source, second_manifest = render_gauss_theory(certificate)
        self.assertEqual(first_source, second_source)
        self.assertEqual(first_manifest, second_manifest)
        self.assertEqual(first_source.count("GL tag "), 1511)
        self.assertEqual(first_source.count("\ntheorem gauss_"), 29)
        self.assertIn("exact anchor.1.1", first_source)
        self.assertIn("have defined_output_totality", first_source)
        self.assertIn("intro no_defined_output", first_source)
        self.assertIn("exact typingRule", first_source)
        self.assertIn("exact zeroRule", first_source)
        self.assertIn("exact induction_successor", first_source)
        self.assertIn("apply induction_hypothesis", first_source)
        self.assertIn("exact inductionProperty", first_source)
        self.assertNotIn("grind", first_source)
        self.assertNotIn("α → Type", first_source)
        for forbidden in ("sorry", "admit", "axiom ", "opaque ", "unsafe "):
            self.assertNotIn(forbidden, first_source.lower())
        self.assertEqual(len(_assert_manifest_coverage(certificate, first_manifest)), 29)

    def test_reformulated_renderer_constructs_the_defined_output_directly(self) -> None:
        certificate = _gauss_certificate()
        theorem = next(
            theorem
            for theorem in certificate["theorems"]
            if theorem["method"] == "reformulated statement"
        )
        source, manifest = _render_reformulated_theorem(theorem, certificate)
        rendered = "\n".join(source)
        self.assertIn("fun (x_1 : α) =>", rendered)
        self.assertIn("defined_output output_result", rendered)
        self.assertIn("exact reformulationSource", rendered)
        self.assertIn("exact ⟨member, ordering⟩", rendered)
        self.assertNotIn("grind", rendered)
        self.assertEqual(manifest["method"], "reformulated statement")


class LeanFTACertificateTests(unittest.TestCase):
    """Exercise FTA shortlist closure, external facts, adaptations, and isolated output."""

    @live_exports_present
    def test_fta_certificate_has_exact_internal_and_external_closure(self) -> None:
        certificate = _fta_certificate()
        self.assertEqual(
            certificate["corpus"]["coverage"],
            {
                "actions": 2782,
                "chapters": 109,
                "methods": {
                    "direct": 55,
                    "induction": 15,
                    "or elimination": 1,
                    "or theorem": 7,
                    "proved not broadcast": 1,
                },
                "rows": 2782,
                "theorems": 79,
            },
        )
        self.assertEqual(len(certificate["corpus"]["target_source_indices"]), 79)
        self.assertEqual(certificate["corpus"]["support_source_indices"], [])
        dependencies = {
            dependency["id"]: dependency
            for dependency in certificate["certificate_dependencies"]
        }
        self.assertTrue(
            all(
                "target_variable_types" in reference
                for dependency in certificate["certificate_dependencies"]
                for reference in dependency["required_theorems"]
            )
        )
        self.assertEqual(
            [item["source_index"] for item in dependencies["peano_externals_36"]["required_theorems"]],
            [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 17, 18, 19, 21, 22, 24, 25, 27, 28, 29, 31, 33, 34, 35],
        )
        self.assertEqual(
            [item["source_index"] for item in dependencies["gauss_externals_24"]["required_theorems"]],
            [18, 21],
        )
        actions = {
            step["action"]
            for theorem in certificate["theorems"]
            for chapter in theorem["chapters"]
            for step in chapter["steps"]
        }
        self.assertTrue(
            {"definition_compact", "or_branch", "or_eliminate", "inequality_symmetric"}
            .issubset(actions)
        )
        self.assertFalse(
            any(
                step["source_tag"] == "externally provided theorem"
                for theorem in certificate["theorems"]
                for chapter in theorem["chapters"]
                for step in chapter["steps"]
            )
        )

    @live_exports_present
    def test_external_adaptation_validator_checks_reordering_and_definition_aliases(self) -> None:
        certificate = _fta_certificate()
        peano_dependency = next(
            dependency
            for dependency in certificate["certificate_dependencies"]
            if dependency["id"] == "peano_externals_36"
        )
        source_certificate = json.loads(
            (REPOSITORY_ROOT / peano_dependency["certificate"]).read_text(encoding="utf-8")
        )
        adapted = {
            item["source_index"]: item
            for item in peano_dependency["required_theorems"]
            if "adaptation" in item
        }
        self.assertEqual(set(adapted), {2, 8, 9, 34, 35})
        for item in adapted.values():
            adaptation = item["adaptation"]
            renaming = _validate_theorem_adaptation(
                parse_mpl(str(adaptation["source_theorem"]["mpl"])),
                parse_mpl(str(item["theorem"]["mpl"])),
                source_certificate["binary_definitions"],
                certificate["binary_definitions"],
                adaptation["head_aliases"],
                base_form_adaptation=adaptation["base_form"],
            )
            self.assertEqual(renaming, adaptation["variable_renaming"])

        item = adapted[34]
        with self.assertRaises(AssertionError):
            _validate_theorem_adaptation(
                parse_mpl(str(item["adaptation"]["source_theorem"]["mpl"])),
                parse_mpl(str(item["theorem"]["mpl"])),
                source_certificate["binary_definitions"],
                certificate["binary_definitions"],
                item["adaptation"]["head_aliases"],
            )

    @live_exports_present
    def test_fta_render_is_deterministic_row_total_and_admission_free(self) -> None:
        certificate = _fta_certificate()
        first_source, first_manifest = render_gauss_theory(certificate)
        second_source, second_manifest = render_gauss_theory(certificate)
        self.assertEqual(first_source, second_source)
        self.assertEqual(first_manifest, second_manifest)
        self.assertEqual(first_source.count("GL tag "), 2782)
        self.assertEqual(first_source.count("\ntheorem fta_"), 79)
        self.assertIn("namespace GLExport.FTA", first_source)
        self.assertIn("anchorPeanoOfFTA", first_source)
        self.assertIn("anchorGaussOfFTA", first_source)
        self.assertEqual(first_source.count("\n  exact anchor.1.1\n"), 1)
        self.assertIn("external_peano_externals_36_000", first_source)
        self.assertNotIn("GLExport.peano_source_", first_source)
        self.assertNotIn("GLExport.gauss_source_", first_source)
        self.assertIn("\ntheorem fta_source_001\n", first_source)
        self.assertIn("  have row_9 : (add v3 v5 v2) := by", first_source)
        self.assertIn(
            "simpa only [gl_or2, GLExport.orIffNotAndNot]",
            first_source,
        )
        self.assertIn(
            "simpa only [gl_or0, GLExport.orIffNotAndNot]",
            first_source,
        )
        self.assertIn("gl_implication211 (α := α)", first_source)
        self.assertIn("∀ (induction_N : GLSet α)", first_source)
        self.assertIn(
            "exact relationalInduction N zero succ add mul one two identity anchor",
            first_source,
        )
        self.assertIn(
            "intro compiled_N compiled_zero compiled_succ compiled_add "
            "compiled_mul compiled_one compiled_two compiled_identity",
            first_source,
        )
        self.assertIn(
            "exact fta_source_056 compiled_N compiled_zero compiled_succ "
            "compiled_add compiled_mul compiled_one compiled_two "
            "compiled_identity compiled_anchor relationalInduction "
            "external_peano_externals_36_019",
            first_source,
        )
        self.assertNotIn("grind", first_source)
        self.assertEqual(len(_assert_manifest_coverage(certificate, first_manifest)), 79)
        for forbidden in ("sorry", "admit", "axiom ", "opaque ", "unsafe "):
            self.assertNotIn(forbidden, first_source.lower())

    @live_exports_present
    def test_fta_external_interfaces_render_validated_target_facts(self) -> None:
        certificate = _fta_certificate()
        theorem = next(
            item for item in certificate["theorems"] if item["id"] == "fta_source_003"
        )
        interfaces, facts = _external_reference_interfaces(theorem["chapters"])
        self.assertEqual(
            facts[("peano_externals_36", 8)],
            "external_peano_externals_36_008 N zero succ add mul one "
            "(anchorPeanoOfFTA N zero succ add mul one two identity anchor)",
        )
        proposition = dict(interfaces)["external_peano_externals_36_008"]
        self.assertIn("∀ (w1 : α)", proposition)
        self.assertIn("gl_AnchorPeano", proposition)

    @live_exports_present
    def test_fta_writer_emits_a_self_contained_project(self) -> None:
        certificate = _fta_certificate()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            write_lean(certificate, output)
            generated = output / "GLExport" / "Generated"
            self.assertEqual(
                sorted(path.name for path in generated.iterdir()),
                ["Definitions.lean", "FTA.lean"],
            )
            self.assertIn(
                "import GLExport.Generated.Definitions",
                (generated / "FTA.lean").read_text(encoding="utf-8"),
            )
            root = (output / "GLExport.lean").read_text(encoding="utf-8")
            self.assertIn("import GLExport.Generated.FTA", root)
            self.assertNotIn("Generated.Peano", root)
            manifest = json.loads(
                (output / "lean_fta_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["coverage"]["theorems"], 79)
            self.assertEqual(
                manifest["target_source_indices"],
                certificate["corpus"]["target_source_indices"],
            )
            self.assertEqual(manifest["support_source_indices"], [])


def _generated_module(name: str) -> Path:
    """
    @brief Return the run-export path of one generated theory module.
    @details Peano and Gauss modules live in the full run's export, the FTA
    module in the shortcut's; both are run artifacts, not tracked files.
    @param name Module file name (`Peano.lean`, `Gauss.lean`, `FTA.lean`).
    @return Path inside the owning export's `GLExport/Generated/`.
    """

    export = LIVE_SHORTCUT_EXPORT if name == "FTA.lean" else LIVE_MAIN_EXPORT
    return export / "GLExport" / "Generated" / name


def _generated_modules() -> list[Path]:
    """
    @brief List every generated `.lean` file of both run exports.
    @return Sorted module paths of the full run's and the shortcut's export.
    """

    return sorted(
        path
        for export in (LIVE_MAIN_EXPORT, LIVE_SHORTCUT_EXPORT)
        for path in (export / "GLExport" / "Generated").glob("*.lean")
    )


_LIVE_CERTIFICATE_FILES = {
    "Peano.lean": LIVE_MAIN_EXPORT / "certificates" / "peano" / "certificate.json",
    "Gauss.lean": LIVE_MAIN_EXPORT / "certificates" / "gauss" / "certificate.json",
    "FTA.lean": FTA_CERTIFICATE,
}
_LIVE_CERTIFICATES: dict[str, dict] = {}


def _expected_rows(tag_pattern: str, names, predicate=lambda step: True) -> int:
    """
    @brief Count the live certificate rows one module-scan test must find.
    @details The generated module of a corpus renders one `GL tag <tag>.`
    comment per certificate row, so the rows of the export's own certificate
    whose source tag matches `tag_pattern` are the exact number of proofs the
    scan has to see; counts follow the run instead of being pinned to one
    prover generation.
    @param tag_pattern Regex the row's `source_tag` must fully match.
    @param names Module file names whose certificates are counted.
    @param predicate Extra row filter (default: every row).
    @return Number of matching rows across the named corpora.
    """

    total = 0
    for name in names:
        if name not in _LIVE_CERTIFICATES:
            _LIVE_CERTIFICATES[name] = json.loads(
                _LIVE_CERTIFICATE_FILES[name].read_text(encoding="utf-8")
            )
        for theorem in _LIVE_CERTIFICATES[name]["theorems"]:
            for chapter in theorem["chapters"]:
                for step in chapter["steps"]:
                    if re.fullmatch(tag_pattern, step["source_tag"]) and predicate(step):
                        total += 1
    return total
SCOPE_TEMPLATE = "(>[v2,v5](in2[v2,v5,V2])(in[v2,V1]))"


def _scope_contexts() -> dict:
    """Return one validated scope-context record keyed by its namespace."""
    return {"boundary": {"namespace": "boundary", "template": {"mpl": SCOPE_TEMPLATE}}}


def _dependency(namespace: str) -> dict:
    """Return a minimal certificate dependency record in one namespace."""
    return {"namespace": namespace, "step": "chapter_1_line_9"}


class ScopeEntryTests(unittest.TestCase):
    """Exercise scope entry, cited-fact instantiation, and witness sharing."""

    def test_main_scope_row_needs_no_entry(self) -> None:
        self.assertEqual(
            _scope_entry_and_specializations("main", _scope_contexts(), (), (), [], {}),
            [],
        )

    def test_unknown_namespace_is_rejected(self) -> None:
        with self.assertRaises(AssertionError):
            _scope_entry_and_specializations(
                "absent", _scope_contexts(), (), (), [], {}
            )

    def test_entry_introduces_prefix_and_instantiates_each_cited_scope_fact(self) -> None:
        lines = _scope_entry_and_specializations(
            "boundary",
            _scope_contexts(),
            ("V4",),
            (),
            [
                (_dependency("main"), "row_9", (), (), (), False),
                (_dependency("boundary"), "row_7", (), (), (), False),
            ],
            {},
        )
        self.assertEqual(
            lines,
            [
                "    intro V4",
                "    intro v2",
                "    intro v5",
                "    intro scope_premise_1",
                "    have scoped_fact_2 := row_7 v2 v5 scope_premise_1",
            ],
        )

    def test_each_cited_fact_is_instantiated_at_its_own_prefix(self) -> None:
        lines = _scope_entry_and_specializations(
            "boundary",
            _scope_contexts(),
            (),
            (),
            [(_dependency("boundary"), "row_7", ("V4",), (), (), False)],
            {},
        )
        self.assertIn("    have scoped_fact_1 := row_7 V4 v2 v5 scope_premise_1", lines)

    def test_guarded_row_introduces_its_witness_and_passes_it_on(self) -> None:
        lines = _scope_entry_and_specializations(
            "boundary",
            _scope_contexts(),
            (),
            ("w",),
            [(_dependency("boundary"), "row_7", (), ("w",), (), False)],
            {},
        )
        self.assertEqual(lines[-3:], [
            "    intro w",
            "    intro witness_guard_1",
            "    have scoped_fact_1 := row_7 v2 v5 scope_premise_1 w witness_guard_1",
        ])

    def test_consumer_obtains_one_witness_and_shares_it_with_guarded_facts(self) -> None:
        lines = _scope_entry_and_specializations(
            "boundary",
            _scope_contexts(),
            (),
            (),
            [
                (_dependency("boundary"), "row_7", (), ("w",), (), False),
                (_dependency("boundary"), "row_8", (), ("w",), (), True),
            ],
            {},
        )
        self.assertEqual(
            lines[-2:],
            [
                "    obtain ⟨w, witness_guard_1⟩ := row_8 v2 v5 scope_premise_1",
                "    have scoped_fact_1 := row_7 v2 v5 scope_premise_1 w witness_guard_1",
            ],
        )

    def test_guarded_fact_without_any_establishing_projection_is_rejected(self) -> None:
        with self.assertRaises(AssertionError):
            _scope_entry_and_specializations(
                "boundary",
                _scope_contexts(),
                (),
                (),
                [(_dependency("boundary"), "row_7", (), ("w",), (), False)],
                {},
            )

    def test_dependency_record_carries_prefix_witness_and_projection_flag(self) -> None:
        record = _scoped_dependency(
            _dependency("boundary"),
            "row_7",
            {"row_7": ("V4",)},
            {"row_7": ("w",)},
            {"row_7"},
            {"row_7": ("v5",)},
        )
        self.assertEqual(
            record[1:], ("row_7", ("V4",), ("w",), ("v5",), True)
        )
        absent = _scoped_dependency(
            _dependency("main"), "row_9", {}, {}, set(), {}
        )
        self.assertEqual(absent[1:], ("row_9", (), (), (), False))

    def test_nested_projection_consumes_existing_guard_before_new_witness(self) -> None:
        lines = _scope_entry_and_specializations(
            "boundary",
            _scope_contexts(),
            (),
            ("outer",),
            [
                (
                    _dependency("boundary"),
                    "row_8",
                    (),
                    ("inner",),
                    ("outer",),
                    True,
                )
            ],
            {},
        )
        self.assertEqual(
            lines[-1],
            (
                "    obtain ⟨inner, witness_guard_2⟩ := "
                "row_8 v2 v5 scope_premise_1 outer witness_guard_1"
            ),
        )


class ExplicitRowReplayTests(unittest.TestCase):
    """Every certificate row must replay its recorded facts explicitly."""

    @live_exports_present
    def test_disintegration_rows_project_the_cited_compound(self) -> None:
        """Every disintegration must expose its projection or contradiction."""

        emitted = 0
        classical_implications = 0
        for name in ("Peano.lean", "Gauss.lean", "FTA.lean"):
            source = _generated_module(name).read_text(
                encoding="utf-8"
            ).splitlines()
            for index, line in enumerate(source):
                if not re.match(
                    r"\s*--\s*chapter_\d+_line_\d+: GL tag disintegration\.",
                    line,
                ):
                    continue
                emitted += 1
                body = []
                for part in source[index + 2 :]:
                    if not part.startswith("    "):
                        break
                    body.append(part)
                proof = "\n".join(body)
                self.assertNotIn("grind", proof, f"{name}: {line}")
                if "projection_premise" in proof:
                    classical_implications += 1

        self.assertEqual(
            emitted, _expected_rows(r"disintegration", ("Peano.lean", "Gauss.lean", "FTA.lean"))
        )
        # The direct-implication shape (a cited negated conjunction) must be
        # exercised; how many rows take it varies with the prover generation.
        self.assertGreaterEqual(classical_implications, 1)

    @live_exports_present
    def test_equality1_rows_eliminate_each_cited_equality(self) -> None:
        """Every equality1 row must eliminate named certificate equalities."""

        emitted = 0
        for name in ("Peano.lean", "Gauss.lean", "FTA.lean"):
            source = _generated_module(name).read_text(
                encoding="utf-8"
            ).splitlines()
            for index, line in enumerate(source):
                if not re.match(
                    r"\s*--\s*chapter_\d+_line_\d+: GL tag equality1\.",
                    line,
                ):
                    continue
                emitted += 1
                body = []
                for part in source[index + 2 :]:
                    if not part.startswith("    "):
                        break
                    body.append(part)
                proof = "\n".join(body)
                self.assertIn("have equality_source", proof, f"{name}: {line}")
                self.assertIn("cases equality_step_1", proof, f"{name}: {line}")
                self.assertNotIn("grind", proof, f"{name}: {line}")

        self.assertEqual(
            emitted, _expected_rows(r"equality1", ("Peano.lean", "Gauss.lean", "FTA.lean"))
        )

    @live_exports_present
    def test_theorem_or_elimination_splits_the_external_parent(self) -> None:
        """The FTA OR elimination must apply all three external interfaces."""

        source = _generated_module("FTA.lean").read_text(
            encoding="utf-8"
        ).splitlines()
        emitted = 0
        for index, line in enumerate(source):
            if not re.match(
                r"\s*--\s*chapter_\d+_line_\d+: GL tag or elimination\.",
                line,
            ):
                continue
            emitted += 1
            self.assertRegex(source[index + 1], r"have row_1 : \(∀ \(v1 : α\)")
            body = []
            for part in source[index + 2 :]:
                if not part.startswith("    "):
                    break
                body.append(part)
            proof = "\n".join(body)
            self.assertIn("multiplication_output_closed", proof, line)
            self.assertIn("rcases or_elim_cases", proof, line)
            for parent in range(1, 4):
                self.assertIn(f"or_elim_parent_{parent}", proof, line)
            self.assertNotIn("grind", proof, line)

        self.assertEqual(
            emitted, _expected_rows(r"or elimination", ("FTA.lean",))
        )

    @live_exports_present
    def test_reformulated_rows_apply_the_cited_source_theorem(self) -> None:
        """Every theorem reformulation must apply its one cited source directly."""

        source = _generated_module("Gauss.lean").read_text(
            encoding="utf-8"
        ).splitlines()
        emitted = 0
        for index, line in enumerate(source):
            if not re.match(
                r"\s*--\s*chapter_\d+_line_\d+: GL tag reformulated from\.",
                line,
            ):
                continue
            emitted += 1
            body = []
            for part in source[index + 2 :]:
                if not part.startswith("    "):
                    break
                body.append(part)
            proof = "\n".join(body)
            self.assertIn("exact reformulationSource", proof, line)
            self.assertNotIn("grind", proof, line)

        self.assertEqual(
            emitted, _expected_rows(r"reformulated from", ("Gauss.lean",))
        )

    @live_exports_present
    def test_or_convergence_rows_eliminate_each_cited_branch(self) -> None:
        """Every OR convergence must expose its two certificate branch cases."""

        emitted = 0
        witness_rows = 0
        for name in ("Peano.lean", "Gauss.lean", "FTA.lean"):
            source = _generated_module(name).read_text(encoding="utf-8").splitlines()
            for index, line in enumerate(source):
                if not re.match(
                    r"\s*--\s*chapter_\d+_line_\d+: GL tag or convergence\.",
                    line,
                ):
                    continue
                emitted += 1
                body = []
                for part in source[index + 2 :]:
                    if not part.startswith("    "):
                        break
                    body.append(part)
                proof = "\n".join(body)
                self.assertIn("rcases or_cases", proof, f"{name}: {line}")
                self.assertNotIn("grind", proof, f"{name}: {line}")
                if "existsAndOfNotForallImpNot" in proof:
                    witness_rows += 1

        self.assertEqual(
            emitted, _expected_rows(r"or convergence", ("Peano.lean", "Gauss.lean", "FTA.lean"))
        )
        self.assertEqual(witness_rows, 2)

    @live_exports_present
    def test_or_theorem_rows_construct_the_disjunction_explicitly(self) -> None:
        """Every constructed OR must expose its case split and parent call."""

        emitted = 0
        for name in ("Peano.lean", "Gauss.lean", "FTA.lean"):
            source = _generated_module(name).read_text(encoding="utf-8").splitlines()
            for index, line in enumerate(source):
                if not re.match(
                    r"\s*--\s*chapter_\d+_line_\d+: GL tag or theorem\.",
                    line,
                ):
                    continue
                emitted += 1
                body = []
                for part in source[index + 2 :]:
                    if not part.startswith("    "):
                        break
                    body.append(part)
                proof = "\n".join(body)
                self.assertIn("by_cases or_case_1", proof, f"{name}: {line}")
                self.assertIn("Or.in", proof, f"{name}: {line}")
                self.assertIn("or_parent_1", proof, f"{name}: {line}")
                self.assertNotIn("rename_i", proof, f"{name}: {line}")
                self.assertNotIn("grind", proof, f"{name}: {line}")

        self.assertEqual(
            emitted, _expected_rows(r"or theorem", ("Peano.lean", "Gauss.lean", "FTA.lean"))
        )

    @live_exports_present
    def test_or_disintegration_rows_return_the_branch_assumption(self) -> None:
        """Every OR branch entry must be the explicit identity implication."""

        emitted = 0
        for name in ("Peano.lean", "Gauss.lean", "FTA.lean"):
            source = _generated_module(name).read_text(encoding="utf-8").splitlines()
            for index, line in enumerate(source):
                if not re.match(
                    r"\s*--\s*chapter_\d+_line_\d+: GL tag or disintegration\.",
                    line,
                ):
                    continue
                emitted += 1
                body = []
                for part in source[index + 2 :]:
                    if not part.startswith("    "):
                        break
                    body.append(part)
                proof = "\n".join(body)
                self.assertIn("exact scope_premise_1", proof, f"{name}: {line}")
                self.assertNotIn("grind", proof, f"{name}: {line}")

        self.assertEqual(
            emitted, _expected_rows(r"or disintegration", ("Peano.lean", "Gauss.lean", "FTA.lean"), predicate=lambda step: "scope_namespace" in step)
        )

    @live_exports_present
    def test_bound_integration_reformulations_use_the_cited_equivalence(self) -> None:
        """Every bound reformulation must refute its universal directly."""

        emitted = 0
        for name in ("Peano.lean", "Gauss.lean", "FTA.lean"):
            source = _generated_module(name).read_text(encoding="utf-8").splitlines()
            for index, line in enumerate(source):
                if not re.match(
                    r"\s*--\s*chapter_\d+_line_\d+: "
                    r"GL tag reformulation for integration >\[bound\]\.",
                    line,
                ):
                    continue
                emitted += 1
                body = []
                for part in source[index + 2 :]:
                    if not part.startswith("    "):
                        break
                    body.append(part)
                proof = "\n".join(body)
                self.assertIn(").2", proof, f"{name}: {line}")
                self.assertIn("universal_counterexample", proof, f"{name}: {line}")
                self.assertNotIn("grind", proof, f"{name}: {line}")

        self.assertEqual(
            emitted, _expected_rows(r"reformulation for integration >\[bound\]", ("Peano.lean", "Gauss.lean", "FTA.lean"))
        )

    @live_exports_present
    def test_vacuous_truth_rows_eliminate_the_cited_contradiction(self) -> None:
        """Every vacuous truth must expose its cited complementary pair."""

        emitted = 0
        for name in ("Peano.lean", "Gauss.lean", "FTA.lean"):
            source = _generated_module(name).read_text(encoding="utf-8").splitlines()
            for index, line in enumerate(source):
                if not re.match(
                    r"\s*--\s*chapter_\d+_line_\d+: GL tag vacuous truth\.",
                    line,
                ):
                    continue
                emitted += 1
                body = []
                for part in source[index + 2 :]:
                    if not part.startswith("    "):
                        break
                    body.append(part)
                proof = "\n".join(body)
                self.assertIn("False.elim", proof, f"{name}: {line}")
                self.assertNotIn("grind", proof, f"{name}: {line}")

        self.assertEqual(
            emitted, _expected_rows(r"vacuous truth", ("Peano.lean", "Gauss.lean", "FTA.lean"))
        )

    @live_exports_present
    def test_empty_integration_reformulations_use_the_cited_equivalence(self) -> None:
        """Every empty-bound reformulation must refute its universal directly."""

        emitted = 0
        for name in ("Peano.lean", "Gauss.lean", "FTA.lean"):
            source = _generated_module(name).read_text(encoding="utf-8").splitlines()
            for index, line in enumerate(source):
                if not re.match(
                    r"\s*--\s*chapter_\d+_line_\d+: "
                    r"GL tag reformulation for integration >\[\]\.",
                    line,
                ):
                    continue
                emitted += 1
                body = []
                for part in source[index + 2 :]:
                    if not part.startswith("    "):
                        break
                    body.append(part)
                proof = "\n".join(body)
                self.assertIn(").2", proof, f"{name}: {line}")
                self.assertIn("universal_counterexample", proof, f"{name}: {line}")
                self.assertNotIn("grind", proof, f"{name}: {line}")

        self.assertEqual(
            emitted, _expected_rows(r"reformulation for integration >\[\]", ("Peano.lean", "Gauss.lean", "FTA.lean"))
        )

    @live_exports_present
    def test_implication_rows_apply_the_cited_rule_and_premises(self) -> None:
        """Every implication must replay its ordered certificate application."""

        emitted = 0
        for name in ("Peano.lean", "Gauss.lean", "FTA.lean"):
            source = _generated_module(name).read_text(encoding="utf-8").splitlines()
            for index, line in enumerate(source):
                if not re.match(
                    r"\s*--\s*chapter_\d+_line_\d+: GL tag implication\.",
                    line,
                ):
                    continue
                emitted += 1
                body = []
                for part in source[index + 2 :]:
                    if not part.startswith("    "):
                        break
                    body.append(part)
                proof = "\n".join(body)
                self.assertRegex(proof, r"(?m)^\s+apply \S+")
                self.assertNotIn("grind", proof, f"{name}: {line}")

        self.assertEqual(
            emitted, _expected_rows(r"implication", ("Peano.lean", "Gauss.lean", "FTA.lean"))
        )

    @live_exports_present
    def test_premise_element_rows_return_the_matching_scope_premise(self) -> None:
        """Every premise element must return its unique recorded scope layer."""

        emitted = 0
        for name in ("Peano.lean", "Gauss.lean", "FTA.lean"):
            source = _generated_module(name).read_text(encoding="utf-8").splitlines()
            for index, line in enumerate(source):
                if not re.match(
                    r"\s*--\s*chapter_\d+_line_\d+: GL tag premise element\.",
                    line,
                ):
                    continue
                emitted += 1
                body = []
                for part in source[index + 2 :]:
                    if not part.startswith("    "):
                        break
                    body.append(part)
                proof = "\n".join(body)
                self.assertRegex(proof, r"exact scope_premise_\d+")
                self.assertNotIn("grind", proof, f"{name}: {line}")

        self.assertEqual(
            emitted, _expected_rows(r"premise element", ("Peano.lean", "Gauss.lean", "FTA.lean"))
        )

    @live_exports_present
    def test_contradiction_rows_apply_the_complementary_citations(self) -> None:
        """Every contradiction row must expose its cited inconsistent pair."""

        emitted = 0
        for name in ("Peano.lean", "Gauss.lean", "FTA.lean"):
            source = _generated_module(name).read_text(encoding="utf-8").splitlines()
            for index, line in enumerate(source):
                if not re.match(
                    r"\s*--\s*chapter_\d+_line_\d+: GL tag contradiction\.",
                    line,
                ):
                    continue
                emitted += 1
                body = []
                for part in source[index + 2 :]:
                    if not part.startswith("    "):
                        break
                    body.append(part)
                proof = "\n".join(body)
                self.assertTrue(
                    "False.elim" in proof or "contradiction_assumption" in proof,
                    f"{name}: {line}",
                )
                self.assertNotIn("grind", proof, f"{name}: {line}")

        self.assertEqual(
            emitted, _expected_rows(r"contradiction", ("Peano.lean", "Gauss.lean", "FTA.lean"))
        )

    @live_exports_present
    def test_equality2_rows_chain_the_two_cited_equalities(self) -> None:
        """Every equality2 row must apply Eq.trans to its two cited facts."""

        emitted = 0
        for name in ("Peano.lean", "Gauss.lean", "FTA.lean"):
            source = _generated_module(name).read_text(encoding="utf-8").splitlines()
            for index, line in enumerate(source):
                if not re.match(
                    r"\s*--\s*chapter_\d+_line_\d+: GL tag equality2\.",
                    line,
                ):
                    continue
                emitted += 1
                body = []
                for part in source[index + 2 :]:
                    if not part.startswith("    "):
                        break
                    body.append(part)
                proof = "\n".join(body)
                self.assertIn("Eq.trans", proof, f"{name}: {line}")
                self.assertNotIn("grind", proof, f"{name}: {line}")

        self.assertEqual(
            emitted, _expected_rows(r"equality2", ("Peano.lean", "Gauss.lean", "FTA.lean"))
        )

    @live_exports_present
    def test_scoped_task_formulation_rows_return_the_scope_premise(self) -> None:
        """Every scoped task formulation must be the explicit identity proof."""

        emitted = 0
        for name in ("Peano.lean", "Gauss.lean", "FTA.lean"):
            source = _generated_module(name).read_text(encoding="utf-8").splitlines()
            for index, line in enumerate(source):
                if not re.match(
                    r"\s*--\s*chapter_\d+_line_\d+: GL tag task formulation\.",
                    line,
                ):
                    continue
                body = []
                for part in source[index + 2 :]:
                    if not part.startswith("    "):
                        break
                    body.append(part)
                if not any("scope_premise_1" in part for part in body):
                    continue
                emitted += 1
                proof = "\n".join(body)
                self.assertIn("exact scope_premise_1", proof, f"{name}: {line}")
                self.assertNotIn("grind", proof, f"{name}: {line}")

        self.assertEqual(
            emitted, _expected_rows(r"task formulation", ("Peano.lean", "Gauss.lean", "FTA.lean"), predicate=lambda step: "scope_namespace" in step)
        )

    @live_exports_present
    def test_inequality_symmetry_rows_use_the_cited_fact_explicitly(self) -> None:
        """Every inequality-symmetry row must apply Eq.symm without search."""

        emitted = 0
        for name in ("Peano.lean", "Gauss.lean", "FTA.lean"):
            source = _generated_module(name).read_text(encoding="utf-8").splitlines()
            for index, line in enumerate(source):
                if not re.match(
                    r"\s*--\s*chapter_\d+_line_\d+: "
                    r"GL tag symmetry of inequality\.",
                    line,
                ):
                    continue
                emitted += 1
                body = []
                for part in source[index + 2 :]:
                    if not part.startswith("    "):
                        break
                    body.append(part)
                self.assertTrue(body, f"{name}: empty inequality-symmetry proof")
                proof = "\n".join(body)
                self.assertIn("Eq.symm", proof, f"{name}: {line}")
                self.assertNotIn("grind", proof, f"{name}: {line}")

        self.assertEqual(
            emitted, _expected_rows(r"symmetry of inequality", ("Peano.lean", "Gauss.lean", "FTA.lean"))
        )

    @live_exports_present
    def test_generated_modules_contain_no_grind(self) -> None:
        """Every generated proof module of the run exports uses explicit proof terms only."""

        for name in ("Peano.lean", "Gauss.lean", "FTA.lean"):
            source = _generated_module(name).read_text(encoding="utf-8")
            self.assertNotIn("grind", source, name)

    @live_exports_present
    def test_schema3_integration_expansions_use_explicit_iff_terms(self) -> None:
        """All live Gauss and FTA integration expansions close explicitly."""

        emitted = 0
        for name in ("Gauss.lean", "FTA.lean"):
            source = _generated_module(name).read_text(
                encoding="utf-8"
            ).splitlines()
            for index, line in enumerate(source):
                if not re.match(
                    r"\s*--\s*chapter_\d+_line_\d+: "
                    r"GL tag expansion for integration\.",
                    line,
                ):
                    continue
                emitted += 1
                body = []
                for part in source[index + 2 :]:
                    if not part.startswith("    "):
                        break
                    body.append(part)
                proof = "\n".join(body)
                self.assertIn("exact Iff.rfl", proof, f"{name}: {line}")
                self.assertNotRegex(proof, r"(?m)\bsimp\s*$", f"{name}: {line}")

        self.assertEqual(
            emitted, _expected_rows(r"expansion for integration", ("Gauss.lean", "FTA.lean"))
        )

    @live_exports_present
    def test_generated_directory_contains_no_bare_simp(self) -> None:
        """No generated Lean line may end in an unrestricted simp tactic."""

        for path in _generated_modules():
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                self.assertIsNone(
                    re.search(r"\bsimp\s*$", line),
                    f"{path.name}:{line_number}: {line}",
                )


if __name__ == "__main__":
    unittest.main()
