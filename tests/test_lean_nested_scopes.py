# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt)
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Nested case contexts must retain ancestor hypotheses and their shared witnesses."""

import unittest
import os
import subprocess
import tempfile
from pathlib import Path

from proof_export.certificate import _schema3_action_metadata
from proof_export.graph import Chapter, Dependency, ProofRow, ScopedExpression
from proof_export.lean import (
    _scope_layers, _scope_entry_and_specializations, _render_chapter_steps, ANCHOR_MPL,
    CONTEXT_NAMES, _negated_conjunction_projection, render_expression,
)
from proof_export.mpl import parse_mpl, map_arguments, to_mpl
from tests.lean_toolchain import LEAN_PROJECT, lean_environment, lean_executable


def scoped(text):
    return ScopedExpression(parse_mpl(text), text, "ordinary")


class NestedScopeTests(unittest.TestCase):
    def contexts(self):
        parent = "main_boundary_contradiction_parent"
        child = parent + "_boundary_ordis_child"
        return parent, child, {
            parent: {"template": {"mpl": "(>[](in[x,N])(in[x,N]))"}},
            child: {"template": {"mpl": "(>[](in[x,N])(>[](=[w,y])(=[w,y])))"}},
        }

    def test_scope_layers_preserve_ancestor_before_child(self):
        _, child, contexts = self.contexts()
        layers = _scope_layers(contexts[child])
        self.assertEqual([premise for _, premise in layers], [parse_mpl("(in[x,N])"), parse_mpl("(=[w,y])")])

    def test_a_non_implication_scope_is_rejected(self):
        with self.assertRaisesRegex(AssertionError, "requires an implication"):
            _scope_layers({"template": {"mpl": "(=[x,y])"}})

    def test_ancestor_fact_uses_only_its_prefix_and_the_same_witness(self):
        parent, child, contexts = self.contexts()
        lines = _scope_entry_and_specializations(
            child, contexts, ("w",), ("w",),
            [({"namespace": parent}, "parent_fact", (), ("w",), (), False)], {})
        self.assertEqual(lines.count("    intro w"), 1)
        self.assertIn("    intro scope_premise_2", lines)
        self.assertIn("    have scoped_fact_1 := parent_fact scope_premise_1 w witness_guard_1", lines)

    def metadata(self, missing_parent=False):
        parent = "main_boundary_contradiction_parent"
        source = "main_boundary_missing" if missing_parent else parent
        child = source + "_boundary_ordis_child"
        rows = (
            ProofRow(1, scoped("(in[x,N])"), parent, "task formulation", ()),
            ProofRow(2, scoped("(or99[x,y])"), parent, "implication", ()),
            ProofRow(3, scoped("(=[x,y])"), child, "or disintegration",
                     (Dependency(scoped("(or99[x,y])"), source),)),
            ProofRow(4, scoped("!(in[x,N])"), "main", "contradiction",
                     (Dependency(scoped("(=[x,y])"), parent), Dependency(scoped("!(=[x,y])"), parent))),
        )
        definitions = {"or99": {"arity": 2, "category": "or", "elements": ["(=[u_1,u_2])", "!(=[u_1,u_2])"]}}
        return child, _schema3_action_metadata(Chapter(Path("fixture.txt"), 0, "single_chapter_proof", rows), definitions, {}, {})

    def test_nested_or_records_its_established_parent_context(self):
        child, (_, contexts) = self.metadata()
        context = next(item for item in contexts if item["namespace"] == child)
        self.assertEqual(len(_scope_layers(context)), 2)
        self.assertEqual(context["parent_namespace"], "main_boundary_contradiction_parent")

    def test_nested_or_cannot_invent_a_parent_scope(self):
        with self.assertRaisesRegex(AssertionError, "parent scope is not established"):
            self.metadata(missing_parent=True)


class NegatedConjunctionReplayTests(unittest.TestCase):
    def exclusion(self, leaves, excluded, order=None):
        source = leaves[0]
        for leaf in leaves[1:]:
            source = "(&" + source + leaf + ")"
        head = leaves[excluded][1:] if leaves[excluded].startswith("!") else "!" + leaves[excluded]
        order = [index for index in range(len(leaves)) if index != excluded] if order is None else order
        target = head
        for index in reversed(order):
            target = "(>[]" + leaves[index] + target + ")"
        return "!" + source, target

    def test_five_member_exclusions_retain_every_position_and_polarity(self):
        leaves = ["(=[x,y])", "!(=[y,z])", "(=[z,t])", "(=[t,x])", "(=[x,z])"]
        for excluded in range(len(leaves)):
            with self.subTest(excluded=excluded):
                source, target = self.exclusion(leaves, excluded)
                proof = "\n".join(_negated_conjunction_projection(
                    parse_mpl(source), parse_mpl(target), "source_assumption"))
                self.assertEqual(proof.count("intro projection_premise"), 1)
                self.assertEqual("Classical.byContradiction" in proof, leaves[excluded].startswith("!"))
                self.assertEqual(proof.splitlines()[-1].count("projection_counterexample"), 1)
                for index in range(1, 5):
                    self.assertEqual(proof.splitlines()[-1].count(f"projection_premise_{index}"), 1)

    def test_exact_gauss_interval_failure_reconstructs_the_left_nested_source(self):
        leaves = ["(implication26[V2,N,+,i0])", "(implication27[V2,N,+,v2])",
                  "(implication28[N,+,i0,v2,V2])", "(in[i0,N])", "(in[v2,N])"]
        source, target = self.exclusion(leaves, 2)
        proof = _negated_conjunction_projection(parse_mpl(source), parse_mpl(target), "row_1")
        self.assertEqual(proof[-1], "    exact (row_1) ⟨⟨⟨⟨projection_premise_1, "
                         "projection_premise_2⟩, projection_counterexample⟩, "
                         "projection_premise_3⟩, projection_premise_4⟩")

    def test_reordered_repeated_and_grouped_premises_are_literal(self):
        a, b, c = "(=[x,y])", "(=[y,z])", "(=[z,t])"
        source, target = self.exclusion([a, b, a, c], 2, [3, 1, 0])
        proof = _negated_conjunction_projection(parse_mpl(source), parse_mpl(target), "row_1")
        self.assertIn("⟨⟨⟨projection_counterexample, projection_premise_2⟩, projection_premise_3⟩, projection_premise_1⟩", proof[-1])
        source, target = self.exclusion(["(&" + a + b + ")", c], 1)
        proof = _negated_conjunction_projection(parse_mpl(source), parse_mpl(target), "row_1")
        self.assertIn("⟨projection_premise, projection_counterexample⟩", proof[-1])

    def test_incomplete_extra_wrong_polarity_and_bound_chains_are_rejected(self):
        a, b, c = "(=[x,y])", "(=[y,z])", "(=[z,t])"
        source, target = self.exclusion([a, b, c], 2)
        for invalid in ("(>[]" + a + "!" + c + ")", "(>[]" + a + target + ")",
                        target.replace("!" + c, c), target.replace(">[]", ">[w]", 1)):
            with self.subTest(invalid=invalid), self.assertRaises(AssertionError):
                _negated_conjunction_projection(parse_mpl(source), parse_mpl(invalid), "row_1")

    def test_kernel_checks_all_five_exclusions(self):
        root = Path(__file__).resolve().parent.parent
        lean = lean_executable()
        leaves = ["(=[x,y])", "!(=[y,z])", "(=[z,t])", "(=[t,x])", "(=[x,z])"]
        types = {name: "element" for name in ("x", "y", "z", "t")}
        blocks = []
        for excluded in range(5):
            source, target = (parse_mpl(item) for item in self.exclusion(leaves, excluded))
            blocks.extend(["example {α : Type} (x y z t : α)",
                           f"    (source_assumption : {render_expression(source, types)}) :",
                           f"    {render_expression(target, types)} := by",
                           *_negated_conjunction_projection(source, target, "source_assumption"), ""])
        with tempfile.TemporaryDirectory(prefix="exclusion_projection_") as directory:
            path = Path(directory) / "ExclusionProjection.lean"
            path.write_text("\n".join(blocks), encoding="utf-8")
            result = subprocess.run([str(lean), str(path)], cwd=LEAN_PROJECT,
                env=lean_environment(), text=True,
                encoding="utf-8", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout)

    def test_residual_or_subsets_and_nested_alternatives_replay_in_kernel(self):
        root = Path(__file__).resolve().parent.parent
        leaves = ['!(=[x,y])', '(=[y,z])', '!(=[z,t])', '(=[t,x])']
        types = {name: 'element' for name in ('x', 'y', 'z', 't')}
        parameters = {name: f'u_{index + 1}' for index, name in enumerate(types)}
        negate = lambda text: text[1:] if text.startswith('!') else '!' + text
        conjunction = leaves[0]
        for leaf in leaves[1:]:
            conjunction = '(&' + conjunction + leaf + ')'
        source = parse_mpl('!' + conjunction)
        blocks = []
        for case, remaining in enumerate(((1, 3), (0, 2), (0, 1, 3))):
            alternatives = [negate(leaves[index]) for index in remaining]
            definitions = {}
            if len(alternatives) == 3:
                definitions['or98'] = {'category': 'or', 'arity': 4, 'elements': [
                    to_mpl(map_arguments(parse_mpl(item), parameters.__getitem__))
                    for item in alternatives[:2]]}
                outer = ['(or98[x,y,z,t])', alternatives[2]]
            else:
                outer = alternatives
            definitions['or99'] = {'category': 'or', 'arity': 4, 'elements': [
                to_mpl(map_arguments(parse_mpl(item), parameters.__getitem__)) for item in outer]}
            target = '(or99[x,y,z,t])'
            for index in reversed(range(len(leaves))):
                if index not in remaining:
                    target = '(>[]' + leaves[index] + target + ')'
            implication = parse_mpl(target)
            proof = _negated_conjunction_projection(source, implication, 'source_assumption', definitions)
            self.assertEqual(sum('have projection_residual_' in line for line in proof), len(remaining))
            for invalid in (target.replace('(or99[x,y,z,t])', '(or99[x,z,y,t])'),
                            target.replace('(or99[x,y,z,t])', '!(or99[x,y,z,t])')):
                with self.assertRaises(AssertionError):
                    _negated_conjunction_projection(source, parse_mpl(invalid), 'source_assumption', definitions)
            blocks.append(f'namespace residual_{case}')
            for name, entry in definitions.items():
                reverse = {value: key for key, value in parameters.items()}
                rendered = [render_expression(map_arguments(parse_mpl(item), reverse.__getitem__), types)
                            for item in entry['elements']]
                blocks.append(f'def gl_{name} {{α : Type}} (x y z t : α) : Prop := '
                              + '(' + ' ∨ '.join(rendered) + ')')
            blocks.extend(['example {α : Type} (x y z t : α)',
                           f'    (source_assumption : {render_expression(source, types)}) :',
                           f'    {render_expression(implication, types)} := by',
                           *proof, f'end residual_{case}', ''])
        with tempfile.TemporaryDirectory(prefix='residual_exclusion_') as directory:
            path = Path(directory) / 'Residual.lean'
            path.write_text('\n'.join(blocks), encoding='utf-8')
            result = subprocess.run([lean_executable(), str(path)],
                cwd=LEAN_PROJECT, env=lean_environment(),
                text=True, encoding='utf-8', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout)

    def render(self, left, right, premise, conclusion, compact=False):
        source = "!(&" + left + right + ")"
        premises = (premise,) if isinstance(premise, str) else premise
        target = conclusion
        for item in reversed(premises):
            target = "(>[]" + item + target + ")"
        definitions = {}
        if compact:
            target = "(implication99[x,y,z,t])"
            parameters = {name: f"u_{index}" for index, name in enumerate(("x", "y", "z", "t"), 1)}
            definitions = {"implication99": {"category": "implication", "elements": [
                to_mpl(map_arguments(parse_mpl(item), parameters.__getitem__))
                for item in (*premises, conclusion)]}}
        first = {"id": "chapter_1_line_1", "source_line": 1, "source_tag": "task formulation",
                 "action": "assume", "namespace": "main", "conclusion": {"mpl": source}, "dependencies": []}
        second = {"id": "chapter_1_line_2", "source_line": 2, "source_tag": "disintegration",
                  "action": "compound_project", "namespace": "main", "conclusion": {"mpl": target},
                  "dependencies": [{"step": first["id"], "namespace": "main", "expression": {"mpl": source}}]}
        chapter = {"steps": [first, second], "variable_types": {name: "element" for name in ("x", "y", "z", "t")}}
        lines, _ = _render_chapter_steps(chapter, "test", {source: "source_assumption"}, {},
                                        ambient_variables={"x", "y", "z", "t"}, binary_definitions=definitions)
        return "\n".join(lines)

    def test_both_projection_directions_and_both_conclusion_polarities(self):
        a, b = "(=[x,y])", "(=[z,t])"
        for left, right, premise, conclusion, classical, pair in (
            (a, b, a, "!" + b, False, "projection_premise, projection_counterexample"),
            (a, b, b, "!" + a, False, "projection_counterexample, projection_premise"),
            (a, "!" + b, a, b, True, "projection_premise, projection_counterexample"),
            ("!" + a, b, b, a, True, "projection_counterexample, projection_premise"),
        ):
            with self.subTest(left=left, right=right, premise=premise):
                text = self.render(left, right, premise, conclusion)
                self.assertEqual("apply Classical.byContradiction" in text, classical)
                self.assertIn("⟨" + pair + "⟩", text)

    def test_unlicensed_positive_conclusion_is_rejected(self):
        a, b = "(=[x,y])", "(=[z,t])"
        with self.assertRaisesRegex(AssertionError, "does not project"):
            self.render(a, b, a, b)

    def test_kernel_checks_compact_projection_with_reflexive_premise(self):
        root = Path(__file__).resolve().parent.parent
        lean = lean_executable()
        proof = self.render("!(=[z,t])", "(=[x,x])", "(=[x,x])", "(=[z,t])", compact=True)
        source = (
            "def gl_implication99 {α : Type} (x y z t : α) : Prop := (x = x) → (z = t)\n"
            "example {α : Type} (x y z t : α) (source_assumption : ¬ ((¬ (z = t)) ∧ (x = x))) "
            ": gl_implication99 x y z t := by\n" + proof + "\n  exact row_2\n")
        with tempfile.TemporaryDirectory(prefix="compact_projection_") as directory:
            path = Path(directory) / "CompactProjection.lean"
            path.write_text(source, encoding="utf-8")
            result = subprocess.run([str(lean), str(path)], cwd=LEAN_PROJECT,
                env=lean_environment(), text=True,
                encoding="utf-8", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout)

    def test_kernel_checks_compact_exclusion_with_complete_premise_chain(self):
        root = Path(__file__).resolve().parent.parent
        left = "(&!(=[x,y])!(=[y,z]))"
        right = "!(=[z,t])"
        proof = self.render(left, right, ("!(=[x,y])", right), "(=[y,z])", compact=True)
        source = (
            "def gl_implication99 {α : Type} (x y z t : α) : Prop := (¬ x = y) → (¬ z = t) → y = z\n"
            "example {α : Type} (x y z t : α) (source_assumption : ¬ ((¬ x = y ∧ ¬ y = z) ∧ ¬ z = t)) "
            ": gl_implication99 x y z t := by\n" + proof + "\n  exact row_2\n")
        with tempfile.TemporaryDirectory(prefix="compact_chain_") as directory:
            path = Path(directory) / "CompactChain.lean"
            path.write_text(source, encoding="utf-8")
            result = subprocess.run([lean_executable(), str(path)],
                cwd=LEAN_PROJECT, env=lean_environment(),
                text=True, encoding="utf-8", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout)
        for premises in (("!(=[x,y])",), ("!(=[x,y])", "(=[z,t])"),
                         ("!(=[x,y])", right, right)):
            with self.assertRaisesRegex(AssertionError, "does not project"):
                self.render(left, right, premises, "(=[y,z])", compact=True)


class NullaryTheoremCompactionTests(unittest.TestCase):
    def chapter(self, anchor_head, certificate_id=None):
        anchor = parse_mpl(ANCHOR_MPL[anchor_head])
        theorem = "(>[" + ",".join(anchor.arguments) + "]" + ANCHOR_MPL[anchor_head] + "(=[i0,i1]))"
        reference = {"theorem_id": "prior_proof", "source_index": 0, "theorem": {"mpl": theorem}}
        if certificate_id is not None:
            reference["certificate_id"] = certificate_id
        step = {"id": "chapter_1_line_1", "source_line": 1, "source_tag": "compilation",
                "action": "definition_compact", "namespace": "main",
                "conclusion": {"mpl": "(implication99[])"},
                "dependencies": [{"namespace": "main", "source_kind": "selected_theorem",
                    "expression": {"mpl": theorem}, "theorem_reference": reference}]}
        return {"steps": [step], "variable_types": {}}

    def test_theorem_of_a_preceding_certificate_binds_its_own_anchor_context(self):
        # A Peano theorem compiled into a zero-argument compact inside a Gauss
        # or FTA proof, and a Gauss theorem inside an FTA proof.
        for current, source, certificate in (("AnchorGauss", "AnchorPeano", "peano_live_67"),
                                              ("AnchorFTA", "AnchorPeano", "peano_live_67"),
                                              ("AnchorFTA", "AnchorGauss", "gauss_live_33")):
            with self.subTest(current=current, source=source):
                lines, _ = _render_chapter_steps(self.chapter(source, certificate), "test", {}, {},
                    anchor_head=current, binary_definitions={"implication99": {"category": "implication"}})
                text = "\n".join(lines)
                context = " ".join("compiled_" + CONTEXT_NAMES[a] for a in parse_mpl(ANCHOR_MPL[source]).arguments)
                self.assertIn("    simp only [gl_implication99]", text)
                self.assertIn("    intro " + context + "\n", text)
                self.assertIn("    intro compiled_anchor compiled_induction compiled_strong_induction", text)
                self.assertIn("    exact prior_proof " + context
                              + " compiled_anchor compiled_induction compiled_strong_induction", text)
                self.assertNotIn("have source_row_1", text)
                self.assertNotIn("anchorPeanoOf", text)

    def test_theorem_of_a_preceding_certificate_must_match_its_anchor(self):
        for chapter in (self.chapter("AnchorPeano", "gauss_live_33"),
                        self.chapter("AnchorGauss", "peano_live_67")):
            with self.subTest(chapter=chapter["steps"][0]["dependencies"][0]["theorem_reference"]["certificate_id"]):
                with self.assertRaises(AssertionError):
                    _render_chapter_steps(chapter, "test", {}, {}, anchor_head="AnchorFTA",
                        binary_definitions={"implication99": {"category": "implication"}})

    def test_adapted_theorem_of_a_preceding_certificate_is_rejected(self):
        chapter = self.chapter("AnchorPeano", "peano_live_67")
        chapter["steps"][0]["dependencies"][0]["theorem_reference"]["adaptation"] = {}
        with self.assertRaises(AssertionError):
            _render_chapter_steps(chapter, "test", {}, {}, anchor_head="AnchorGauss",
                binary_definitions={"implication99": {"category": "implication"}})

    def test_local_carriers_bind_the_complete_source_context_for_every_anchor(self):
        for anchor in ANCHOR_MPL:
            with self.subTest(anchor=anchor):
                lines, _ = _render_chapter_steps(self.chapter(anchor), "test", {}, {}, anchor_head=anchor,
                    binary_definitions={"implication99": {"category": "implication"}})
                source = "\n".join(lines)
                self.assertIn("intro compiled_N compiled_zero compiled_succ compiled_add compiled_mul compiled_one", source)
                self.assertIn("intro compiled_anchor compiled_induction", source)
                self.assertIn("exact prior_proof compiled_N compiled_zero compiled_succ", source)
                self.assertNotIn("have source_row_1", source)
                self.assertEqual("compiled_identity" in source, anchor != "AnchorPeano")

    def test_carrier_cannot_use_a_theorem_from_a_different_anchor(self):
        with self.assertRaises(AssertionError):
            _render_chapter_steps(self.chapter("AnchorGauss"), "test", {}, {}, anchor_head="AnchorPeano",
                binary_definitions={"implication99": {"category": "implication"}})


class OrExpansionPolarityTests(unittest.TestCase):
    def test_kernel_checks_two_to_four_alternative_expansions_for_every_polarity(self):
        root = Path(__file__).resolve().parent.parent
        lean = lean_executable()
        source = (root / "lean_export/GLExport/ProofSupport.lean").read_text(encoding="utf-8")
        atoms = ["(=[x,y])", "(=[z,t])", "(=[x,z])", "(=[y,t])"]
        types = {name: "element" for name in ("x", "y", "z", "t")}
        parameters = {name: f"u_{index + 1}" for index, name in enumerate(types)}
        negate = lambda text: text[1:] if text.startswith("!") else "!" + text
        for count in (2, 3, 4):
            for signs in range(1 << count):
                leaves = [("!" if signs & (1 << index) else "") + atom
                          for index, atom in enumerate(atoms[:count])]
                compact = "(or90[x,y,z,t])"
                conjunction = negate(leaves[0])
                for leaf in leaves[1:]:
                    conjunction = "(&" + conjunction + negate(leaf) + ")"
                expanded = "!" + conjunction
                first = {"id": "chapter_1_line_1", "source_line": 1,
                         "source_tag": "task formulation", "action": "assume",
                         "namespace": "main", "conclusion": {"mpl": compact}, "dependencies": []}
                second = {"id": "chapter_1_line_2", "source_line": 2,
                          "source_tag": "expansion", "action": "definition_expand",
                          "namespace": "main", "conclusion": {"mpl": expanded},
                          "dependencies": [{"step": first["id"], "namespace": "main",
                                            "expression": {"mpl": compact}}]}
                chapter = {"steps": [first, second], "variable_types": types}
                definitions = {"or90": {"category": "or", "elements": [
                    to_mpl(map_arguments(parse_mpl(leaf), parameters.__getitem__)) for leaf in leaves]}}
                lines, _ = _render_chapter_steps(chapter, "or_polarity",
                    {compact: "source_assumption"}, {}, ambient_variables=set(types),
                    binary_definitions=definitions)
                body = render_expression(parse_mpl(leaves[0]), types)
                for leaf in leaves[1:]:
                    body = "(" + body + " ∨ " + render_expression(parse_mpl(leaf), types) + ")"
                namespace = f"OrPolarity{count}_{signs}"
                source += (f"\nnamespace {namespace}\n"
                           f"def gl_or90 {{α : Type}} (x y z t : α) : Prop := {body}\n"
                           f"example {{α : Type}} (x y z t : α) (source_assumption : gl_or90 x y z t) : "
                           f"{render_expression(parse_mpl(expanded), types)} := by\n"
                           + "\n".join(lines) + "\n  exact row_2\n"
                           + f"end {namespace}\n")
        with tempfile.TemporaryDirectory(prefix="or_polarity_") as directory:
            path = Path(directory) / "OrPolarity.lean"
            path.write_text(source, encoding="utf-8")
            result = subprocess.run([str(lean), str(path)], cwd=LEAN_PROJECT,
                env=lean_environment(), text=True,
                encoding="utf-8", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout)


class OrConvergenceTests(unittest.TestCase):
    def fixture(self, count):
        variables = ["x", *[f"y{index}" for index in range(1, count + 1)]]
        leaves = [f"(=[x,y{index}])" for index in range(1, count + 1)]
        source = "(or99[" + ",".join(variables) + "])"
        scope = "main_boundary_ordis_last"
        steps = []

        def row(expression, action, namespace="main", dependencies=(), **metadata):
            index = len(steps) + 1
            step = {"id": f"chapter_1_line_{index}", "source_line": index,
                    "source_tag": "or convergence" if action == "or_eliminate" else action,
                    "action": action, "namespace": namespace,
                    "conclusion": {"mpl": expression}, "dependencies": list(dependencies),
                    **metadata}
            steps.append(step)
            return {"step": step["id"], "namespace": namespace,
                    "expression": {"mpl": expression}}

        source_dependency = row(source, "assume")
        assumptions = {source: "source_assumption"}
        refutations = []
        for index, leaf in enumerate(leaves[:-1], 1):
            refutations.append(row("!" + leaf, "assume"))
            assumptions["!" + leaf] = f"refuter_{index}"
        branch = row(leaves[-1], "or_branch", scope, [source_dependency],
                     scope_namespace=scope)
        # Certificate dependency order need not match definition case order.
        row(leaves[-1], "or_eliminate", dependencies=[
            source_dependency, branch, *reversed(refutations)], or_definition_head="or99")
        chapter = {"steps": steps,
                   "variable_types": {variable: "element" for variable in variables},
                   "scope_contexts": [{"namespace": scope, "compact": {"mpl": source},
                       "scope_kind": "or_branch", "parent_namespace": "main", "template": {
                       "mpl": "(>[]" + leaves[-1] + leaves[-1] + ")"}}]}
        definitions = {"or99": {"category": "or", "arity": len(variables),
            "elements": [f"(=[u_1,u_{index + 1}])" for index in range(1, count + 1)]}}
        return chapter, assumptions, variables, definitions

    def render(self, fixture):
        chapter, assumptions, variables, definitions = fixture
        return _render_chapter_steps(chapter, "or_convergence", assumptions, {},
            ambient_variables=set(variables), binary_definitions=definitions)

    def test_all_definition_cases_keep_left_grouping_and_exact_dependencies(self):
        for count in (2, 3, 4, 5):
            with self.subTest(count=count):
                lines, manifest = self.render(self.fixture(count))
                source = "\n".join(lines)
                pattern = "or_branch_1 | or_branch_2"
                for index in range(3, count + 1):
                    pattern = f"({pattern}) | or_branch_{index}"
                self.assertIn("rcases or_cases with " + pattern, source)
                self.assertEqual(source.count("· have branch_compact := or_branch_"), count)
                self.assertEqual(len(manifest), count + 2)
                for index in range(1, count):
                    self.assertIn(f"exact False.elim (row_{index + 1} branch_compact)", source)
                self.assertIn(f"exact row_{count + 1} branch_compact", source)

    def test_missing_extra_duplicate_and_wrong_case_dependencies_are_rejected(self):
        for mode in ("missing", "extra", "duplicate", "wrong_polarity"):
            with self.subTest(mode=mode):
                fixture = self.fixture(3)
                dependencies = fixture[0]["steps"][-1]["dependencies"]
                if mode == "missing":
                    dependencies.pop()
                elif mode == "extra":
                    dependencies.append(dependencies[-1])
                elif mode == "duplicate":
                    dependencies[-1] = dependencies[-2]
                else:
                    dependencies[-1]["expression"]["mpl"] = "(=[x,y1])"
                with self.assertRaises(AssertionError):
                    self.render(fixture)

    def test_kernel_checks_binary_ternary_and_larger_convergence(self):
        root = Path(__file__).resolve().parent.parent
        lean = lean_executable()
        source = (root / "lean_export/GLExport/ProofSupport.lean").read_text(encoding="utf-8")
        for count in (2, 3, 4, 5):
            fixture = self.fixture(count)
            lines, _ = self.render(fixture)
            variables = " ".join(fixture[2])
            body = "(x = y1)"
            for index in range(2, count + 1):
                body = f"({body} ∨ (x = y{index}))"
            refuters = " ".join(f"(refuter_{index} : ¬ (x = y{index}))"
                                for index in range(1, count))
            source += (f"\nnamespace OrConvergence{count}\n"
                f"def gl_or99 {{α : Type}} ({variables} : α) : Prop := {body}\n"
                f"example {{α : Type}} ({variables} : α) "
                f"(source_assumption : gl_or99 {variables}) {refuters} "
                f": x = y{count} := by\n" + "\n".join(lines)
                + f"\n  exact row_{count + 2}\nend OrConvergence{count}\n")
        with tempfile.TemporaryDirectory(prefix="or_convergence_") as directory:
            path = Path(directory) / "OrConvergence.lean"
            path.write_text(source, encoding="utf-8")
            result = subprocess.run([str(lean), str(path)], cwd=LEAN_PROJECT,
                env=lean_environment(), text=True,
                encoding="utf-8", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout)


if __name__ == "__main__":
    unittest.main()
