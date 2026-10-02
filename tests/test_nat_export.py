# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt)
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""The concrete export must discharge its assumptions and reject assumed imports."""

import tempfile
import unittest
from pathlib import Path

from proof_export.lean import _context_signature, render_expression
from proof_export.mpl import parse_mpl
from proof_export.nat import write_nat_model


class NatExportTests(unittest.TestCase):
    def certificates(self):
        types = {"N": "set", "i0": "element", "s": "binary_relation",
                 "+": "ternary_relation", "*": "ternary_relation",
                 "i1": "element", "i2": "element", "id": "binary_relation"}
        definitions = {"Anchor" + kind: {"elements": ["(in[u_2,u_1])"]}
                       for kind in ("Peano", "Gauss", "FTA")}
        result = []
        for kind in ("Peano", "Gauss", "FTA"):
            arguments = "N,i0,s,+,*,i1" + ("" if kind == "Peano" else ",i2,id")
            result.append({"corpus": {"id": kind.lower() + "_test"},
                           "binary_definitions": definitions,
                           "theorems": [{"id": kind.lower() + "_zero",
                                         "variable_types": types,
                                         "theorem": {"mpl": f"(>[{arguments}](Anchor{kind}[{arguments}])(in[i0,N]))"}}]})
        return result

    def test_every_batch_gets_a_closed_nat_application(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / "GLExport").mkdir()
            write_nat_model(self.certificates(), project)
            source = (project / "GLExport/NatResults.lean").read_text(encoding="utf-8")
            self.assertEqual(source.count("theorem "), 3)
            self.assertNotIn("Type u", source)
            self.assertNotIn("external_", source)
            self.assertNotIn("(relationalInduction :", source)
            self.assertIn("exact GLExport.FTA.fta_zero", source)
            self.assertIn("NatModel.anchorFTA NatModel.induction", source)
            self.assertIn("import GLExport.NatResults", (project / "GLExport.lean").read_text())

    def test_concrete_nullary_fact_keeps_the_polymorphic_parameter_name(self):
        certificates = self.certificates()
        theorem = certificates[-1]["theorems"][0]
        theorem["theorem"]["mpl"] = "(>[N,i0,s,+,*,i1,i2,id](AnchorFTA[N,i0,s,+,*,i1,i2,id])(implication999[]))"
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / "GLExport").mkdir()
            write_nat_model(certificates, project)
            source = (project / "GLExport/NatResults.lean").read_text(encoding="utf-8")
            self.assertIn("(gl_implication999 (α := Nat))", source)
            self.assertNotIn("(Nat :=", source)

    def test_unsupported_carrier_is_rejected(self):
        with self.assertRaises(AssertionError):
            render_expression(parse_mpl("(=[x,x])"), {}, carrier="Unknown")

    def test_an_external_snapshot_cannot_be_certified_as_nat(self):
        certificates = self.certificates()
        certificates[-1]["certificate_dependencies"] = [{"theorem_list_scope": "external"}]
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(AssertionError):
            write_nat_model(certificates, Path(directory))

    def test_missing_anchor_definition_is_rejected(self):
        certificates = self.certificates()
        del certificates[-1]["binary_definitions"]["AnchorFTA"]
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(AssertionError, "definition missing"):
            write_nat_model(certificates, Path(directory))

    def test_global_anchor_quantification_keeps_induction_inside_its_scope(self):
        theorem = self.certificates()[-1]["theorems"][0]
        rendered = render_expression(parse_mpl(theorem["theorem"]["mpl"]), theorem["variable_types"])
        self.assertIn("→ (RelationalInduction N zero succ →", rendered)
        context = "\n".join(_context_signature("AnchorFTA"))
        self.assertNotIn("induction_N", context)
        self.assertIn("P zero", context)


if __name__ == "__main__":
    unittest.main()
