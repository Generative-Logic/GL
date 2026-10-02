# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Constructive reformulation across anchor and membership-clause shapes."""

import json
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from proof_export.certificate import _defined_output_totality, _definition_from_expansion, _membership_body_key
from proof_export.graph import Dependency, ProofRow, parse_scoped_expression
from proof_export.lean import _defined_output_proof_term, _membership_transport_term, _render_reformulated_theorem
from proof_export.membership_transport import membership_transport_plan
from proof_export.mpl import parse_mpl
from tests.lean_toolchain import LEAN_PROJECT, lean_environment, lean_executable


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/lean_certificates"


def singleton_certificate():
    return json.loads((FIXTURES / "fta_singleton_reformulation.json").read_text(encoding="utf-8"))


def singleton_with_membership_premise():
    certificate = singleton_certificate()
    theorem = certificate["theorems"][0]
    step = theorem["chapters"][0]["steps"][0]
    anchor = "(AnchorFTA[N,i0,s,+,*,i1,i2,id])"
    prefix = "(>[N,i0,s,+,*,i1,i2,id]" + anchor
    target = prefix + "(>[v1](in[v1,N])(existence21[v1,N,+])))"
    source = prefix + "(>[v1](in[v1,N])(>[V1](EnumerationSet1[v1,V1])(interval[N,+,v1,v1,V1]))))"
    theorem["theorem"]["mpl"] = target
    theorem["variable_types"]["v1"] = "element"
    step["conclusion"]["mpl"] = target
    step["dependencies"][0]["theorem_reference"]["theorem"]["mpl"] = source
    totality = step["defined_output_totality"]
    totality["defining_expression"]["mpl"] = "(EnumerationSet1[v1,1])"
    totality["witness_elements"][0]["mpl"] = "(=[1,v1])"
    return certificate


class ConstructiveReformulationTests(unittest.TestCase):
    def render(self, certificate):
        return "\n".join(_render_reformulated_theorem(certificate["theorems"][0], certificate)[0])

    def test_anchor_only_fta_constructs_the_singleton_and_applies_its_source(self):
        text = self.render(singleton_certificate())
        self.assertIn("anchor : gl_AnchorFTA", text)
        self.assertIn("fun (x_1 : α) => (x_1 = zero)", text)
        self.assertIn("exact reformulationSource (fun (x_1 : α) => (x_1 = zero)) defined_output", text)
        self.assertNotIn("reformulation_premise_", text)
        self.assertNotIn("AnchorGauss", text)
        for forbidden in ("sorry", "admit", "grind", "axiom "):
            self.assertNotIn(forbidden, text)

    def test_one_retained_premise_is_introduced_and_passed_to_the_source(self):
        text = self.render(singleton_with_membership_premise())
        self.assertIn("intro v1", text)
        self.assertIn("intro reformulation_premise_1", text)
        self.assertIn("exact reformulationSource _ reformulation_premise_1", text)
        self.assertIn("defined_output_totality v1", text)

    def test_unmatched_retained_premise_is_rejected(self):
        certificate = singleton_with_membership_premise()
        step = certificate["theorems"][0]["chapters"][0]["steps"][0]
        source = step["dependencies"][0]["theorem_reference"]["theorem"]
        source["mpl"] = source["mpl"].replace("(in[v1,N])", "!(in[v1,N])")
        with self.assertRaisesRegex(AssertionError, "logical orderings"):
            self.render(certificate)

    def test_source_anchor_cannot_change(self):
        certificate = singleton_certificate()
        step = certificate["theorems"][0]["chapters"][0]["steps"][0]
        source = step["dependencies"][0]["theorem_reference"]["theorem"]
        source["mpl"] = source["mpl"].replace("AnchorFTA", "AnchorGauss")
        with self.assertRaises(AssertionError):
            self.render(certificate)

    def test_uncertified_totality_and_changed_witness_are_rejected(self):
        for change in ("acyclicity", "witness"):
            with self.subTest(change=change):
                certificate = singleton_certificate()
                totality = certificate["theorems"][0]["chapters"][0]["steps"][0]["defined_output_totality"]
                if change == "acyclicity":
                    totality["definition_dependency_acyclic"] = False
                else:
                    totality["witness_elements"][0]["mpl"] = "!(=[1,i0])"
                with self.assertRaises(AssertionError):
                    self.render(certificate)

    def test_kernel_checks_singleton_set_binary_relation_and_three_conditions(self):
        singleton = singleton_certificate()
        totality = singleton["theorems"][0]["chapters"][0]["steps"][0]["defined_output_totality"]
        unary = _defined_output_proof_term(totality, singleton["binary_definitions"])

        gauss = json.loads((FIXTURES / "lean_gauss_main_29.json").read_text(encoding="utf-8"))
        relation = next(
            theorem["chapters"][0]["steps"][0]["defined_output_totality"]
            for theorem in gauss["theorems"]
            if theorem["method"] == "reformulated statement"
            and theorem["chapters"][0]["steps"][0]["defined_output_totality"]["output_type"] == "binary_relation"
        )
        binary = _defined_output_proof_term(relation, gauss["binary_definitions"])

        definitions = {
            "Triple": {"arity": 4, "category": "and", "elements": [
                "(Forward[u_4,u_1])", "(Forward[u_4,u_2])", "(Forward[u_4,u_3])",
                "(Reverse[u_1,u_2,u_3,u_4])"]},
            "Forward": {"arity": 2, "category": "implication", "elements": [
                "(in[1,u_1])", "(in[1,u_2])"]},
            "Reverse": {"arity": 4, "category": "implication", "elements": [
                "(in[1,u_1])", "(in[1,u_2])", "(in[1,u_3])", "(in[1,u_4])"]},
        }
        triple = _defined_output_totality("Triple", definitions, {"Triple": ["set"] * 4})
        triple["defining_expression"] = {"mpl": "(Triple[u_1,u_2,u_3,u_4])"}
        triple["witness_elements"] = [{"mpl": item} for item in triple["witness_elements"]]
        three = _defined_output_proof_term(triple, definitions)
        code = f"""
example {{α : Type}} (a : α) :
    (∀ x : α, x = a → x = a) ∧ (∀ x : α, x = a → x = a) := by
  exact {unary}
example {{α : Type}} (P : α → Prop) (R : α → α → Prop) :
    ((∀ x y : α, P x ∧ R x y → P x) ∧
     (∀ x y : α, P x ∧ R x y → R x y)) ∧
    (∀ x : α, P x → ∀ y : α, R x y → P x ∧ R x y) := by
  exact {binary}
example {{α : Type}} (A B C : α → Prop) :
    (((∀ x : α, A x ∧ B x ∧ C x → A x) ∧
      (∀ x : α, A x ∧ B x ∧ C x → B x)) ∧
      (∀ x : α, A x ∧ B x ∧ C x → C x)) ∧
    (∀ x : α, A x → B x → C x → A x ∧ B x ∧ C x) := by
  exact {three}
"""
        lean = lean_executable()
        with tempfile.TemporaryDirectory(prefix="fta_reformulation_") as directory:
            path = Path(directory) / "DefinitionWitnesses.lean"
            path.write_text(code, encoding="utf-8")
            result = subprocess.run(
                [str(lean), str(path)], cwd=LEAN_PROJECT,
                env=lean_environment(),
                text=True, encoding="utf-8", stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stdout)

    def test_connective_alias_keys_preserve_order_polarity_ports_and_recursion(self):
        definitions = {
            "A": {"arity": 3, "category": "or", "elements": ["(in[u_1,u_2])", "(=[u_1,u_3])"]},
            "B": {"arity": 3, "category": "or", "elements": ["(in[u_3,u_1])", "(=[u_3,u_2])"]},
        }
        key = _membership_body_key(parse_mpl("(A[x,M,a])"), definitions)
        self.assertEqual(key, _membership_body_key(parse_mpl("(B[M,a,x])"), definitions))
        self.assertNotEqual(key, _membership_body_key(parse_mpl("(B[M,x,a])"), definitions))
        definitions["Triple"] = {"arity": 3, "category": "and", "elements": ["(P[u_1])", "(P[u_2])", "(P[u_3])"]}
        triple = _membership_body_key(parse_mpl("(Triple[x,y,z])"), definitions)
        self.assertEqual(triple, _membership_body_key(parse_mpl("(&(&(P[x])(P[y]))(P[z]))"), definitions))
        self.assertNotEqual(triple, _membership_body_key(parse_mpl("(&(P[x])(P[y])(P[z]))"), definitions))
        for mode in ("order", "polarity", "connective"):
            changed = deepcopy(definitions)
            if mode == "order":
                changed["B"]["elements"].reverse()
            elif mode == "polarity":
                changed["B"]["elements"][0] = "!" + changed["B"]["elements"][0]
            else:
                changed["B"]["category"] = "and"
            self.assertNotEqual(key, _membership_body_key(parse_mpl("(B[M,a,x])"), changed))
        definitions["B"]["elements"][0] = "(B[u_1,u_2,u_3])"
        with self.assertRaisesRegex(AssertionError, "recursive membership"):
            _membership_body_key(parse_mpl("(B[M,a,x])"), definitions)

    def test_membership_transport_preserves_leaves_scopes_and_or_order(self):
        a = ("atom", "P", ("x", "fixed"))
        b = ("atom", "=", ("y", "value"))
        c = ("atom", "Q", ("x", "x"))
        source = ("or", c, ("and", b, a))
        target = ("or", c, ("and", a, b))
        plan = membership_transport_plan(source, target)
        self.assertEqual(plan["children"][1]["sources"], [1, 0])
        self.assertIn("Or.elim", _membership_transport_term(plan, "source"))
        self.assertEqual(_membership_transport_term({"kind": "identity"}, "source"), "source")
        nested = ("and", ("and", a, b), c)
        self.assertIsNotNone(membership_transport_plan(nested, ("and", c, ("and", b, a))))
        for altered in (
            ("or", ("and", a, b), c),
            ("or", c, ("and", a, a)),
            ("or", c, ("and", a, ("not", b))),
            ("or", ("atom", "Q", ("x", "y")), ("and", a, b)),
            ("or", c, ("and", ("atom", "P", ("x", "other")), b)),
            ("or", c, ("and", ("atom", "Other", ("x", "fixed")), b)),
        ):
            with self.subTest(altered=altered):
                self.assertIsNone(membership_transport_plan(source, altered))
        self.assertIsNone(membership_transport_plan(nested, ("and", a, ("and", b, c))))
        self.assertIsNone(membership_transport_plan(("not", source), ("not", target)))
        self.assertIsNone(membership_transport_plan(
            ("implies", ("x",), a, source), ("implies", ("x",), a, target)))
        self.assertIsNone(membership_transport_plan(
            ("implies", ("x",), a, a), ("implies", ("y",), a, a)))
        with self.assertRaises(AssertionError):
            _membership_transport_term({"kind": "and", "sources": [0, 0],
                "children": [{"kind": "identity"}] * 2}, "source")
        with self.assertRaises(AssertionError):
            _membership_transport_term({"kind": "or", "sources": [1, 0],
                "children": [{"kind": "identity"}] * 2}, "source")

    def test_binary_output_conjunction_swap_is_checked_and_kernel_proved(self):
        definitions = {
            "Append": {"arity": 4, "category": "and", "elements": [
                "(Forward[u_4,u_3,u_1,u_2])", "(Reverse[u_3,u_2,u_1,u_4])"]},
            "Forward": {"arity": 4, "category": "implication", "elements": [
                "(in2[1,2,u_1])", "(A[1,2,u_2,u_3,u_4])"]},
            "Reverse": {"arity": 4, "category": "implication", "elements": [
                "(B[1,2,u_1,u_2,u_3])", "(in2[1,2,u_4])"]},
            "A": {"arity": 5, "category": "or", "elements": [
                "(in2[u_1,u_2,u_3])", "(Pair[u_1,u_4,u_2,u_5])"]},
            "B": {"arity": 5, "category": "or", "elements": [
                "(in2[u_1,u_2,u_3])", "(Pair[u_2,u_4,u_1,u_5])"]},
            "Pair": {"arity": 4, "category": "and", "elements": [
                "(=[u_1,u_2])", "(=[u_3,u_4])"]},
        }
        signatures = {"Append": ["element", "element", "binary_relation", "binary_relation"]}
        totality = _defined_output_totality("Append", definitions, signatures)
        self.assertEqual(totality["output_argument"], 4)
        self.assertEqual(totality["reverse_membership_transport"][0]["children"][1]["sources"], [1, 0])
        totality["defining_expression"] = {"mpl": "(Append[u_1,u_2,u_3,u_4])"}
        totality["witness_elements"] = [{"mpl": item} for item in totality["witness_elements"]]
        term = _defined_output_proof_term(totality, definitions)
        source = ("def gl_A {α : Type} (x y : α) (f : α → α → Prop) (n a : α) : Prop := f x y ∨ (x = n ∧ y = a)\n"
                  "def gl_B {α : Type} (x y : α) (f : α → α → Prop) (a n : α) : Prop := f x y ∨ (y = a ∧ x = n)\n"
                  "example {α : Type} (f : α → α → Prop) (n a : α) :\n"
                  " (∀ x y, gl_A x y f n a → gl_A x y f n a) ∧ (∀ x y, gl_B x y f a n → gl_A x y f n a) := by\n"
                  f"  exact {term}\n")
        with tempfile.TemporaryDirectory(prefix="append_transport_") as directory:
            path = Path(directory) / "AppendTransport.lean"
            path.write_text(source, encoding="utf-8")
            result = subprocess.run([lean_executable(), str(path)],
                cwd=LEAN_PROJECT, env=lean_environment(),
                text=True, encoding="utf-8", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout)
        for mode in ("omitted", "altered"):
            corrupted = deepcopy(totality)
            if mode == "omitted":
                del corrupted["reverse_membership_transport"]
            else:
                corrupted["reverse_membership_transport"][0]["children"][1]["sources"] = [0, 1]
            with self.assertRaises(AssertionError):
                _defined_output_proof_term(corrupted, definitions)
        for changed_body in ("(Pair[u_2,u_4,u_1,u_4])", "!(Pair[u_2,u_4,u_1,u_5])"):
            changed = deepcopy(definitions)
            changed["B"]["elements"][1] = changed_body
            with self.assertRaisesRegex(AssertionError, "constructive defined output"):
                _defined_output_totality("Append", changed, signatures)
            with self.assertRaises(AssertionError):
                _defined_output_proof_term(totality, changed)

    def test_totality_and_kernel_accept_exact_or_aliases_but_reject_changed_body(self):
        definitions = {
            "Add": {"arity": 3, "category": "and", "elements": ["(Forward[u_3,u_1,u_2])", "(Reverse[u_1,u_2,u_3])"]},
            "Forward": {"arity": 3, "category": "implication", "elements": ["(in[1,u_1])", "(A[1,u_2,u_3])"]},
            "Reverse": {"arity": 3, "category": "implication", "elements": ["(B[u_1,u_2,1])", "(in[1,u_3])"]},
            "A": {"arity": 3, "category": "or", "elements": ["(in[u_1,u_2])", "(=[u_1,u_3])"]},
            "B": {"arity": 3, "category": "or", "elements": ["(in[u_3,u_1])", "(=[u_3,u_2])"]},
        }
        signatures = {"Add": ["set", "element", "set"]}
        totality = _defined_output_totality("Add", definitions, signatures)
        self.assertEqual(totality["output_argument"], 3)
        totality["defining_expression"] = {"mpl": "(Add[u_1,u_2,u_3])"}
        totality["witness_elements"] = [{"mpl": item} for item in totality["witness_elements"]]
        term = _defined_output_proof_term(totality, definitions)
        source = ("def gl_A {α : Type} (x : α) (M : α → Prop) (a : α) : Prop := M x ∨ x = a\n"
                  "def gl_B {α : Type} (M : α → Prop) (a x : α) : Prop := M x ∨ x = a\n"
                  "example {α : Type} (M : α → Prop) (a : α) :\n"
                  " (∀ x, gl_A x M a → gl_A x M a) ∧ (∀ x, gl_B M a x → gl_A x M a) := by\n"
                  f"  exact {term}\n")
        with tempfile.TemporaryDirectory(prefix="or_totality_") as directory:
            path = Path(directory) / "OrTotality.lean"
            path.write_text(source, encoding="utf-8")
            result = subprocess.run([lean_executable(), str(path)],
                cwd=LEAN_PROJECT, env=lean_environment(),
                text=True, encoding="utf-8", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout)
        definitions["B"]["elements"][1] = "!(=[u_3,u_2])"
        with self.assertRaisesRegex(AssertionError, "constructive defined output"):
            _defined_output_totality("Add", definitions, signatures)
        with self.assertRaises(AssertionError):
            _defined_output_proof_term(totality, definitions)


def reformulation_row(target, source):
    return ProofRow(1, parse_scoped_expression(target), "main", "reformulated from",
                    (Dependency(parse_scoped_expression(source), "main"),))


class ReformulationDefinitionTests(unittest.TestCase):
    ANCHOR = "(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])"

    def test_source_head_of_the_same_shape_as_a_premise_stays_in_the_body(self):
        # Restriction keeps the fold: the premise fold of x and the head fold
        # of the restriction y differ only in the sequence argument.
        source = (self.ANCHOR + "(>[9,10,11](limitSequence[1,4,9,10,11])"
                  "(>[12](fold[1,3,4,5,10,2,9,12])(fold[1,3,4,5,11,2,9,12]))))")
        target = (self.ANCHOR + "(>[10,9,12](fold[1,3,4,5,10,2,9,12])"
                  "(existence24[1,4,9,10,3,5,2,12])))")
        definition = _definition_from_expansion(reformulation_row(target, source))
        self.assertEqual(definition["head"], "existence24")
        self.assertEqual(definition["category"], "existence")
        self.assertEqual(definition["definition_source"], "reformulation_action")
        self.assertEqual(definition["elements"], [
            "(limitSequence[u_1,u_2,u_3,u_4,1])",
            "(fold[u_1,u_5,u_2,u_6,1,u_7,u_3,u_8])",
        ])

    def test_target_premise_matching_only_the_source_head_is_rejected(self):
        source = "(>[1,2](D[1,2])(H[2]))"
        target = "(>[2](H[2])(existence9[2]))"
        with self.assertRaises(AssertionError):
            _definition_from_expansion(reformulation_row(target, source))


if __name__ == "__main__":
    unittest.main()
