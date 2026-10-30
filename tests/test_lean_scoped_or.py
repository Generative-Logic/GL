# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt)
# Dual-licensed under AGPLv3 and commercial terms — https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Replay OR convergence under quantified implication premises."""

from copy import deepcopy
from dataclasses import replace
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from proof_export.certificate import ACTION_BY_TAG, _schema3_action_metadata
from proof_export.graph import Chapter, Dependency, ProofRow, ScopedExpression
from proof_export.lean import _chapter_replacements, _render_chapter_steps
from proof_export.mpl import parse_mpl, to_mpl
from tests.lean_toolchain import LEAN_PROJECT, lean_environment, lean_executable


PARENT = "main_boundary_implication99"
CHILD = PARENT + "_boundary_ordis_right"
SOURCE = "(or99[x,y,z])"
NEGATIVE = "!(=[x,y])"
RESULT = "(=[x,z])"
COMPACT = "(implication99[y,z])"
TEMPLATE = "(>[x]" + SOURCE + "(>[]" + NEGATIVE + RESULT + "))"
DEFINITIONS = {
    "or99": {"arity": 3, "category": "or", "elements": ["(=[u_1,u_2])", "(=[u_1,u_3])"]},
    "implication99": {"arity": 2, "category": "implication", "elements": ["(or99[1,u_1,u_2])", "!(=[1,u_1])", "(=[1,u_2])"]},
}


def fixture():
    parent = {"namespace": PARENT, "compact": {"mpl": COMPACT},
              "template": {"mpl": TEMPLATE}, "definition_head": "implication99"}
    child_template = "(>[x]" + SOURCE + "(>[]" + NEGATIVE + "(>[]" + RESULT + RESULT + ")))"
    child = {"namespace": CHILD, "parent_namespace": PARENT, "scope_kind": "or_branch",
             "compact": {"mpl": SOURCE}, "template": {"mpl": child_template}}
    steps = []

    def row(expression, action, namespace, dependencies, **metadata):
        number = len(steps) + 1
        step = {"id": f"chapter_1_line_{number}", "source_line": number,
                "source_tag": action, "action": action, "namespace": namespace,
                "conclusion": {"mpl": expression}, "dependencies": dependencies, **metadata}
        steps.append(step)
        return {"step": step["id"], "namespace": namespace, "expression": step["conclusion"]}

    template_dependency = {"namespace": "main", "source_kind": "integration_goal_template", "expression": {"mpl": TEMPLATE}}
    source = row(SOURCE, "integration_premise", PARENT, [template_dependency], scope_namespace=PARENT)
    refutation = row(NEGATIVE, "integration_premise", PARENT, [template_dependency], scope_namespace=PARENT)
    branch = row(RESULT, "or_branch", CHILD, [source], scope_namespace=CHILD)
    row(RESULT, "or_eliminate", PARENT, [source, branch, refutation], or_definition_head="or99")
    return {"steps": steps, "scope_contexts": [parent, child],
            "variable_types": {name: "element" for name in ("x", "y", "z")}}


def metadata_fixture():
    def scoped(text, goal=False):
        return ScopedExpression(parse_mpl(text), text + ("_integration_goal" if goal else ""),
                                "integration_goal" if goal else "ordinary")
    rows = (
        ProofRow(1, scoped(TEMPLATE, True), "main", "expansion for integration", (Dependency(scoped(COMPACT, True), "main"),)),
        ProofRow(2, scoped(SOURCE), PARENT, "premise element", (Dependency(scoped(TEMPLATE, True), "main"),)),
        ProofRow(3, scoped(NEGATIVE), PARENT, "premise element", (Dependency(scoped(TEMPLATE, True), "main"),)),
        ProofRow(4, scoped(RESULT), CHILD, "or disintegration", (Dependency(scoped(SOURCE), PARENT),)),
        ProofRow(5, scoped(RESULT), PARENT, "or convergence", (
            Dependency(scoped(SOURCE), PARENT), Dependency(scoped(RESULT), CHILD), Dependency(scoped(NEGATIVE), PARENT))),
        ProofRow(6, scoped(COMPACT), "main", "validity name", (Dependency(scoped(RESULT), PARENT),)),
    )
    return Chapter(Path("scoped_or_fixture.txt"), 1, "single_chapter_proof", rows)


class ScopedOrConvergenceTests(unittest.TestCase):
    def render(self, chapter):
        return _render_chapter_steps(chapter, "scoped_or", {}, {},
                                     ambient_variables={"y", "z"}, binary_definitions=DEFINITIONS)

    def test_kernel_retains_ancestor_goal_parameters_and_scopes_copy_reflexivity(self):
        chapter = fixture()
        template = "(>[x]" + SOURCE + "(>[]" + NEGATIVE + "(=[x,w])))"
        chapter["scope_contexts"][0]["template"]["mpl"] = template
        chapter["scope_contexts"][0]["compact"]["mpl"] = "(implication99[y,z,w])"
        chapter["variable_types"].update({"w": "element", "y_copy": "element"})
        for step in chapter["steps"]:
            for dependency in step["dependencies"]:
                if dependency.get("source_kind") == "integration_goal_template":
                    dependency["expression"]["mpl"] = template
        chapter["steps"].append({"id": "chapter_1_line_5", "source_line": 5,
            "source_tag": "variable copy", "action": "variable_alias", "namespace": CHILD,
            "conclusion": {"mpl": "(=[y,y_copy])"}, "dependencies": [],
            "variable_alias": {"source": "y", "alias": "y_copy"}})
        definitions = deepcopy(DEFINITIONS)
        definitions["implication99"] = {"arity": 3, "category": "implication", "elements": [
            "(or99[1,u_1,u_2])", "!(=[1,u_1])", "(=[1,u_3])"]}
        lines, _ = _render_chapter_steps(chapter, "scoped_copy", {}, _chapter_replacements(chapter),
            ambient_variables={"y", "z"}, binary_definitions=definitions)
        source = ("def gl_or99 {α : Type} (x y z : α) : Prop := x = y ∨ x = z\n"
            "example {α : Type} (y z : α) : ∀ (w x : α), gl_or99 x y z → ¬ x = y → x = z := by\n"
            + "\n".join(lines) + "\n  exact row_4\n")
        self.assertIn("row_2 w x", source)
        self.assertIn("have row_5 : (∀ (w : α) (x : α)", source)
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="inherited_scope_copy_") as directory:
            path = Path(directory) / "InheritedScopeCopy.lean"
            path.write_text(source, encoding="utf-8")
            result = subprocess.run([lean_executable(), str(path)],
                cwd=LEAN_PROJECT, env=lean_environment(),
                text=True, encoding="utf-8", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + "\n" + source)
        chapter["scope_contexts"][0]["parent_namespace"] = CHILD
        with self.assertRaises(AssertionError):
            _render_chapter_steps(chapter, "invalid_ancestry", {}, {},
                ambient_variables={"y", "z"}, binary_definitions=definitions)

    def test_metadata_retains_the_exact_parent_and_parent_refutation(self):
        metadata, contexts = _schema3_action_metadata(metadata_fixture(), DEFINITIONS, {}, {})
        self.assertEqual(metadata[5], {"or_definition_head": "or99"})
        self.assertEqual(next(context for context in contexts if context["namespace"] == CHILD)["parent_namespace"], PARENT)

    def test_metadata_rejects_a_compact_from_another_parent(self):
        chapter = metadata_fixture()
        rows = list(chapter.rows)
        row = rows[4]
        rows[4] = replace(row, dependencies=(replace(row.dependencies[0], namespace="main"), *row.dependencies[1:]))
        with self.assertRaisesRegex(AssertionError, "exact parent scope"):
            _schema3_action_metadata(replace(chapter, rows=tuple(rows)), DEFINITIONS, {}, {})

    def test_renderer_applies_parent_binders_and_premises_before_the_branch(self):
        lines, manifest = self.render(fixture())
        source = "\n".join(lines)
        self.assertIn("have or_cases := scoped_fact_1", source)
        self.assertIn("exact row_3 x scope_premise_1 scope_premise_2 branch_compact", source)
        self.assertIn("exact False.elim (scoped_fact_3 branch_compact)", source)
        self.assertEqual(len(manifest), 4)

    def test_renderer_rejects_scope_escape_and_invalid_case_evidence(self):
        for mode in ("escape", "wrong_parent", "missing", "duplicate"):
            with self.subTest(mode=mode):
                chapter = deepcopy(fixture())
                if mode == "escape":
                    chapter["steps"][-1]["namespace"] = "main"
                elif mode == "wrong_parent":
                    chapter["scope_contexts"][1]["parent_namespace"] = "main"
                elif mode == "missing":
                    chapter["steps"][-1]["dependencies"].pop()
                else:
                    chapter["steps"][-1]["dependencies"][-1] = chapter["steps"][-1]["dependencies"][1]
                with self.assertRaises(AssertionError):
                    self.render(chapter)

    def test_kernel_checks_quantified_parent_with_a_refuted_case(self):
        root = Path(__file__).resolve().parent.parent
        lines, _ = self.render(fixture())
        source = ("def gl_or99 {α : Type} (x y z : α) : Prop := x = y ∨ x = z\n"
                  "example {α : Type} (y z : α) : ∀ x, gl_or99 x y z → ¬ (x = y) → x = z := by\n"
                  + "\n".join(lines) + "\n  exact row_4\n")
        with tempfile.TemporaryDirectory(prefix="scoped_or_") as directory:
            path = Path(directory) / "ScopedOr.lean"
            path.write_text(source, encoding="utf-8")
            result = subprocess.run([lean_executable(), str(path)],
                cwd=LEAN_PROJECT, env=lean_environment(),
                text=True, encoding="utf-8", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout)


def introduction_fixture(root_scope=False, signed_peer=False):
    selected = "(=[x,y])"
    compact = "(or99[x,y,z])"
    parent = "main" if root_scope else PARENT
    child = parent + "_boundary_orint_" + compact + "_(" + selected + ")"
    definitions = deepcopy(DEFINITIONS)
    if root_scope:
        definitions["or99"]["elements"] = ["(=[u_1,u_2])", "!(=[u_1,u_2])"]
        peers = [selected]
    elif signed_peer:
        definitions["or99"]["elements"] = ["(=[u_1,u_2])", "!(=[u_1,u_3])", "(=[u_2,u_3])"]
        peers = ["(=[x,z])", "!(=[y,z])"]
    else:
        peers = ["!(=[x,z])"]
    template = "(>[x]" + selected + compact + ")"
    rows = []

    def scoped(text, goal=False):
        return ScopedExpression(parse_mpl(text), text + ("_integration_goal" if goal else ""),
                                "integration_goal" if goal else "ordinary")

    def row(text, namespace, tag, dependencies):
        rows.append(ProofRow(len(rows) + 1, scoped(text, tag == "expansion for integration"), namespace, tag, tuple(dependencies)))

    if not root_scope:
        row(template, "main", "expansion for integration", [Dependency(scoped(COMPACT, True), "main")])
        row(selected, parent, "premise element", [Dependency(scoped(template, True), "main")])
    for peer in peers:
        row(peer, child, "or branch assumption", [Dependency(scoped(compact, True), parent)])
    if not root_scope:
        row(selected, child, "zoom in", [Dependency(scoped(selected), parent)])
    row(compact, parent, "or branch proven", [Dependency(scoped(selected), child)])
    if not root_scope:
        row(COMPACT, "main", "validity name", [Dependency(scoped(compact), parent)])
    chapter = Chapter(Path("or_introduction_fixture.txt"), 1, "single_chapter_proof", tuple(rows))
    return chapter, definitions, template


class OrIntroductionTests(unittest.TestCase):
    def test_kernel_retains_projection_identity_and_discharges_branch_witness(self):
        compact = "(or99[a,N,b])"
        selected = "(in[a,N])"
        child = "main_boundary_orint_" + compact + "_(" + selected + ")"
        existence = "!(>[w](in[w,N])!(=[w,a]))"
        rule = "(>[w](=[w,b])(in[a,N]))"
        context = {"namespace": child, "parent_namespace": "main", "scope_kind": "or_integration",
                   "compact": {"mpl": compact}, "template": {"mpl": "(>[]!(=[a,b])" + selected + ")"}}
        steps = []

        def row(expression, action, namespace, dependencies, **metadata):
            step = {"id": f"chapter_1_line_{len(steps) + 1}", "source_line": len(steps) + 1,
                    "source_tag": action, "action": action, "namespace": namespace,
                    "conclusion": {"mpl": expression}, "dependencies": dependencies, **metadata}
            steps.append(step)
            return {"step": step["id"], "namespace": namespace, "expression": step["conclusion"], "source_kind": "local_step"}

        source = row(existence, "assume", "main", [])
        equality = row("(=[a,b])", "assume", "main", [])
        rule_row = row(rule, "assume", "main", [])
        row("!(=[a,b])", "or_integration_assume", child,
            [{"namespace": "main", "source_kind": "integration_goal_template", "expression": {"mpl": compact}}], scope_namespace=child)
        local_source = row(existence, "scope_inherit", child, [source])
        row("(in[w,N])", "compound_project", child, [local_source])
        projection = row("(=[w,a])", "compound_project", child, [local_source])
        rewritten = row("(=[w,b])", "equality_substitute", child, [projection, equality])
        branch = row(selected, "implication_apply", child, [rule_row, rewritten])
        row(compact, "or_introduce", "main", [branch], or_definition_head="or99", selected_case=0, branch_namespace=child)
        chapter = {"steps": steps, "scope_contexts": [context],
                   "variable_types": {"N": "set", "a": "element", "b": "element", "w": "element"}}
        definitions = {"or99": {"arity": 3, "category": "or", "elements": ["(in[u_1,u_2])", "(=[u_1,u_3])"]}}
        assumptions = {existence: "given_witness", "(=[a,b])": "given_equality", rule: "given_rule"}
        lines, _ = _render_chapter_steps(chapter, "or_witness", assumptions, {},
            ambient_variables={"a", "b"}, binary_definitions=definitions)
        rendered = "\n".join(lines)
        self.assertIn("have equality_source := witness_guard_1.2", rendered)
        self.assertIn("obtain ⟨w, witness_guard_1⟩ := row_6 missing_or.2", rendered)
        source = ("def gl_or99 {α : Type} (a : α) (N : α → Prop) (b : α) : Prop := N a ∨ a = b\n"
                  "example {α : Type} (N : α → Prop) (a b : α)\n"
                  " (given_witness : ¬ (∀ w, N w → ¬ (w = a))) (given_equality : a = b)\n"
                  " (given_rule : ∀ w, w = b → N a) : gl_or99 a N b := by\n"
                  + rendered + "\n  exact row_10\n")
        implication_chapter = deepcopy(chapter)
        implication_context = implication_chapter["scope_contexts"][0]
        implication_context["definition_head"] = "implication99"
        implication_context["compact"] = {"mpl": "(implication99[a,b,N])"}
        implication_context["template"] = {"mpl": "(>[](=[a,b])(in[a,N]))"}
        implication_steps = implication_chapter["steps"]
        implication_steps[3].update(action="integration_premise", conclusion={"mpl": "(=[a,b])"},
            dependencies=[{"namespace": "main", "source_kind": "integration_goal_template",
                           "expression": implication_context["template"]}])
        implication_steps[-1].update(action="integration_discharge", scope_context=implication_context,
            conclusion=implication_context["compact"])
        definitions["implication99"] = {"arity": 3, "category": "implication", "elements": ["(=[u_1,u_2])", "(in[u_1,u_3])"]}
        implication_lines, _ = _render_chapter_steps(implication_chapter, "integration_witness", assumptions, {},
            ambient_variables={"a", "b"}, binary_definitions=definitions)
        self.assertIn("    obtain ⟨w, witness_guard_1⟩ := row_6 scope_premise_1", implication_lines)
        source += ("def gl_implication99 {α : Type} (a b : α) (N : α → Prop) : Prop := a = b → N a\n"
                   "example {α : Type} (N : α → Prop) (a b : α)\n"
                   " (given_witness : ¬ (∀ w, N w → ¬ (w = a))) (given_equality : a = b)\n"
                   " (given_rule : ∀ w, w = b → N a) : gl_implication99 a b N := by\n"
                   + "\n".join(implication_lines) + "\n  exact row_10\n")
        root = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory(prefix="or_witness_") as directory:
            path = Path(directory) / "OrWitness.lean"
            path.write_text(source, encoding="utf-8")
            result = subprocess.run([lean_executable(), str(path)],
                cwd=LEAN_PROJECT, env=lean_environment(),
                text=True, encoding="utf-8", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + "\n" + source)
        steps[7]["conclusion"]["mpl"] = "(=[w,a])"
        steps[7]["dependencies"][1]["expression"] = {"mpl": "(=[b,N])"}
        with self.assertRaises(AssertionError):
            _render_chapter_steps(chapter, "or_witness", assumptions, {},
                ambient_variables={"a", "b"}, binary_definitions=definitions)

    def render(self, root_scope=False, signed_peer=False):
        chapter, definitions, template = introduction_fixture(root_scope, signed_peer)
        metadata, contexts = _schema3_action_metadata(chapter, definitions, {}, {})
        steps, references = [], {}
        for row in chapter.rows:
            if row.tag in {"expansion for integration", "validity name"}:
                continue
            dependencies = []
            for dependency in row.dependencies:
                item = {"namespace": dependency.namespace, "expression": {"mpl": to_mpl(dependency.expression)}}
                if dependency.scoped_expression.proof_scope == "integration_goal":
                    item["source_kind"] = "integration_goal_template"
                else:
                    item["step"] = references[(dependency.namespace, to_mpl(dependency.expression))]
                dependencies.append(item)
            step_id = f"chapter_1_line_{len(steps) + 1}"
            steps.append({"id": step_id, "source_line": row.source_line, "source_tag": row.tag,
                          "action": ACTION_BY_TAG[row.tag], "namespace": row.namespace,
                          "conclusion": {"mpl": to_mpl(row.conclusion.expression)},
                          "dependencies": dependencies, **metadata.get(row.source_line, {})})
            references[(row.namespace, to_mpl(row.conclusion.expression))] = step_id
        neutral = {"steps": steps, "scope_contexts": contexts,
                   "variable_types": {name: "element" for name in ("x", "y", "z")}}
        lines, manifest = _render_chapter_steps(neutral, "or_intro", {}, {},
            ambient_variables={"x", "y", "z"} if root_scope else {"y", "z"}, binary_definitions=definitions)
        return lines, manifest, len(steps)

    def test_metadata_rejects_wrong_peer_parent_and_selected_case(self):
        for mode in ("peer", "parent", "selected", "goal_scope"):
            chapter, definitions, _ = introduction_fixture()
            rows = list(chapter.rows)
            if mode == "peer":
                rows[2] = replace(rows[2], conclusion=rows[1].conclusion)
            elif mode == "parent":
                rows[2] = replace(rows[2], dependencies=(replace(rows[2].dependencies[0], namespace="main"),))
            elif mode == "selected":
                rows[4] = replace(rows[4], dependencies=(replace(rows[4].dependencies[0], scoped_expression=rows[2].conclusion),))
            else:
                dependency = rows[2].dependencies[0]
                rows[2] = replace(rows[2], dependencies=(replace(dependency,
                    scoped_expression=replace(dependency.scoped_expression, proof_scope="ordinary")),))
            with self.subTest(mode=mode), self.assertRaises(AssertionError):
                _schema3_action_metadata(replace(chapter, rows=tuple(rows)), definitions, {}, {})

    def test_kernel_checks_main_nested_and_signed_three_case_introduction(self):
        root = Path(__file__).resolve().parent.parent
        for root_scope, signed_peer in ((True, False), (False, False), (False, True)):
            with self.subTest(root_scope=root_scope, signed_peer=signed_peer):
                lines, manifest, count = self.render(root_scope, signed_peer)
                body = "x = y ∨ ¬ (x = y)" if root_scope else (
                    "(x = y ∨ ¬ (x = z)) ∨ y = z" if signed_peer else "x = y ∨ x = z")
                binders = "(x y z : α)" if root_scope else "(y z : α)"
                goal = "gl_or99 x y z" if root_scope else "∀ x, x = y → gl_or99 x y z"
                source = (f"def gl_or99 {{α : Type}} (x y z : α) : Prop := {body}\n"
                          f"example {{α : Type}} {binders} : {goal} := by\n"
                          + "\n".join(lines) + f"\n  exact row_{count}\n")
                with tempfile.TemporaryDirectory(prefix="or_intro_") as directory:
                    path = Path(directory) / "OrIntroduction.lean"
                    path.write_text(source, encoding="utf-8")
                    result = subprocess.run([lean_executable(), str(path)],
                        cwd=LEAN_PROJECT, env=lean_environment(),
                        text=True, encoding="utf-8", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
                    self.assertEqual(result.returncode, 0, result.stdout + "\n" + source)
                    self.assertEqual(len(manifest), count)


AND_SOURCE = "!(and98[x,y,z,w])"
AND_CASE = "!(=[x,y])"
AND_REFUTATION = "!(=[z,w])"
AND_COMPACT = "(implication98[y,z,w])"
AND_TEMPLATE = "(>[x]" + AND_SOURCE + "(>[]" + AND_REFUTATION + AND_CASE + "))"
AND_DEFINITIONS = {
    "and98": {"arity": 4, "category": "and", "elements": ["(=[u_1,u_2])", "!(=[u_3,u_4])"]},
    "implication98": {"arity": 3, "category": "implication", "elements": [
        "!(and98[1,u_1,u_2,u_3])", "!(=[u_2,u_3])", "!(=[1,u_1])"]},
}


def negated_and_fixture():
    """Steps of a negated-conjunction case split under an integration scope.

    The premise ``¬(x = y ∧ ¬ z = w)`` splits into the cases ``¬ x = y`` and
    ``z = w`` (D-460); the second case is
    refuted by the parent premise ``¬ z = w`` and the first case is the
    conclusion itself.
    """
    parent = {"namespace": PARENT, "compact": {"mpl": AND_COMPACT},
              "template": {"mpl": AND_TEMPLATE}, "definition_head": "implication98"}
    child_template = ("(>[x]" + AND_SOURCE + "(>[]" + AND_REFUTATION
                      + "(>[]" + AND_CASE + AND_CASE + ")))")
    child = {"namespace": CHILD, "parent_namespace": PARENT, "scope_kind": "or_branch",
             "compact": {"mpl": AND_SOURCE}, "template": {"mpl": child_template}}
    steps = []

    def row(expression, action, namespace, dependencies, **metadata):
        number = len(steps) + 1
        step = {"id": f"chapter_1_line_{number}", "source_line": number,
                "source_tag": action, "action": action, "namespace": namespace,
                "conclusion": {"mpl": expression}, "dependencies": dependencies, **metadata}
        steps.append(step)
        return {"step": step["id"], "namespace": namespace, "expression": step["conclusion"]}

    template_dependency = {"namespace": "main", "source_kind": "integration_goal_template",
                           "expression": {"mpl": AND_TEMPLATE}}
    source = row(AND_SOURCE, "integration_premise", PARENT, [template_dependency], scope_namespace=PARENT)
    refutation = row(AND_REFUTATION, "integration_premise", PARENT, [template_dependency],
                     scope_namespace=PARENT)
    branch = row(AND_CASE, "or_branch", CHILD, [source], scope_namespace=CHILD)
    row(AND_CASE, "or_eliminate", PARENT, [source, branch, refutation], or_definition_head="and98")
    return {"steps": steps, "scope_contexts": [parent, child],
            "variable_types": {name: "element" for name in ("x", "y", "z", "w")}}


class NegatedConjunctionCaseTests(unittest.TestCase):
    def test_case_split_cases_complements_conjuncts(self):
        from proof_export.mpl import Negation
        from proof_export.or_convergence import case_split_cases, convergence_cases
        atom, cases = case_split_cases(parse_mpl(AND_SOURCE), AND_DEFINITIONS)
        self.assertEqual(atom.head, "and98")
        self.assertEqual([to_mpl(case) for case in cases], ["!(=[x,y])", "(=[z,w])"])
        self.assertEqual([branch for _, branch in convergence_cases(parse_mpl(AND_SOURCE), PARENT, AND_DEFINITIONS)],
                         [PARENT + "_boundary_ordis_!(and98[x,y,z,w])_(!(=[x,y]))",
                          PARENT + "_boundary_ordis_!(and98[x,y,z,w])_((=[z,w]))"])
        _, or_cases = case_split_cases(parse_mpl(SOURCE), DEFINITIONS)
        self.assertEqual([to_mpl(case) for case in or_cases], ["(=[x,y])", "(=[x,z])"])
        with self.assertRaises(AssertionError):
            case_split_cases(Negation(parse_mpl(SOURCE)), DEFINITIONS)

    def test_metadata_accepts_a_negated_conjunction_source(self):
        def scoped(text, goal=False):
            return ScopedExpression(parse_mpl(text), text + ("_integration_goal" if goal else ""),
                                    "integration_goal" if goal else "ordinary")
        rows = (
            ProofRow(1, scoped(AND_TEMPLATE, True), "main", "expansion for integration",
                     (Dependency(scoped(AND_COMPACT, True), "main"),)),
            ProofRow(2, scoped(AND_SOURCE), PARENT, "premise element",
                     (Dependency(scoped(AND_TEMPLATE, True), "main"),)),
            ProofRow(3, scoped(AND_REFUTATION), PARENT, "premise element",
                     (Dependency(scoped(AND_TEMPLATE, True), "main"),)),
            ProofRow(4, scoped(AND_CASE), CHILD, "or disintegration",
                     (Dependency(scoped(AND_SOURCE), PARENT),)),
            ProofRow(5, scoped(AND_CASE), PARENT, "or convergence", (
                Dependency(scoped(AND_SOURCE), PARENT), Dependency(scoped(AND_CASE), CHILD),
                Dependency(scoped(AND_REFUTATION), PARENT))),
            ProofRow(6, scoped(AND_COMPACT), "main", "validity name",
                     (Dependency(scoped(AND_CASE), PARENT),)),
        )
        chapter = Chapter(Path("negated_and_fixture.txt"), 1, "single_chapter_proof", rows)
        metadata, contexts = _schema3_action_metadata(chapter, AND_DEFINITIONS, {}, {})
        self.assertEqual(metadata[5], {"or_definition_head": "and98"})
        branch = next(context for context in contexts if context["namespace"] == CHILD)
        self.assertEqual(branch["definition_head"], "and98")

    def test_kernel_checks_negated_conjunction_case_elimination(self):
        lines, _ = _render_chapter_steps(negated_and_fixture(), "negated_and", {}, {},
            ambient_variables={"y", "z", "w"}, binary_definitions=AND_DEFINITIONS)
        body = "\n".join(lines)
        self.assertIn("simp only [gl_and98, notAndIffNotOrNot, notNotIff] at or_cases", body)
        support = (LEAN_PROJECT / "GLExport" / "ProofSupport.lean").read_text(encoding="utf-8")
        source = (support + "\nopen GLExport\n"
                  "def gl_and98 {α : Type} (x y z w : α) : Prop := (x = y ∧ ¬ z = w)\n"
                  "example {α : Type} (y z w : α) : ∀ x, ¬ gl_and98 x y z w → ¬ (z = w) → ¬ (x = y) := by\n"
                  + body + "\n  exact row_4\n")
        with tempfile.TemporaryDirectory(prefix="negated_and_") as directory:
            path = Path(directory) / "NegatedAnd.lean"
            path.write_text(source, encoding="utf-8")
            result = subprocess.run([lean_executable(), str(path)],
                cwd=LEAN_PROJECT, env=lean_environment(),
                text=True, encoding="utf-8", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + "\n" + source)


BRANCH_WITNESS_DEFINITIONS = {
    "or99": {"arity": 3, "category": "or", "elements": ["(=[u_1,u_2])", "(=[u_1,u_3])"]},
    "ex1": {"arity": 2, "category": "existence", "elements": ["(in[1,u_1])", "(=[1,u_2])"]},
}


def branch_witness_fixture():
    """A case branch that mints two chained witnesses from its own rows: w from
    a replayed main existence, then v from a rule's existence built on w. The
    converged fact depends on both; the case assumption introduces neither."""
    source = "(or99[x,y,z])"
    case = "(=[x,y])"
    refutation = "!(=[x,z])"
    existence = "(ex1[N,x])"
    first_rule = "(>[w](=[w,x])(ex1[N,w]))"
    turn_first = "(>[w](=[w,x])(=[x,w]))"
    turn_second = "(>[v,w](=[v,w])(=[w,v]))"
    second_rule = "(>[v,w](=[w,v])(>[](=[x,w])(in[x,N])))"
    result = "(in[x,N])"
    child = "main_boundary_ordis_" + source + "_(" + case + ")"
    context = {"namespace": child, "parent_namespace": "main", "scope_kind": "or_branch",
               "compact": {"mpl": source}, "template": {"mpl": "(>[]" + case + case + ")"}}
    steps = []

    def row(expression, action, namespace, dependencies, **metadata):
        step = {"id": f"chapter_1_line_{len(steps) + 1}", "source_line": len(steps) + 1,
                "source_tag": action, "action": action, "namespace": namespace,
                "conclusion": {"mpl": expression}, "dependencies": dependencies, **metadata}
        steps.append(step)
        return {"step": step["id"], "namespace": namespace, "expression": step["conclusion"],
                "source_kind": "local_step"}

    or_source = row(source, "assume", "main", [])
    refuted = row(refutation, "assume", "main", [])
    given_existence = row(existence, "assume", "main", [])
    rule_1 = row(first_rule, "assume", "main", [])
    turn_1 = row(turn_first, "assume", "main", [])
    turn_2 = row(turn_second, "assume", "main", [])
    rule_2 = row(second_rule, "assume", "main", [])
    local_existence = row(existence, "scope_inherit", child, [given_existence])
    first_body = row("!(>[w](in[w,N])!(=[w,x]))", "definition_expand", child, [local_existence])
    first_link = row("(=[w,x])", "compound_project", child, [first_body])
    built = row("(ex1[N,w])", "implication_apply", child, [rule_1, first_link])
    second_body = row("!(>[v](in[v,N])!(=[v,w]))", "definition_expand", child, [built])
    second_link = row("(=[v,w])", "compound_project", child, [second_body])
    first_turned = row("(=[x,w])", "implication_apply", child, [turn_1, first_link])
    second_turned = row("(=[w,v])", "implication_apply", child, [turn_2, second_link])
    branch = row(result, "implication_apply", child, [rule_2, second_turned, first_turned])
    row(result, "or_eliminate", "main", [or_source, branch, refuted], or_definition_head="or99")
    chapter = {"steps": steps, "scope_contexts": [context],
               "variable_types": {"N": "set", "x": "element", "y": "element", "z": "element",
                                  "w": "element", "v": "element"}}
    assumptions = {source: "given_or", refutation: "given_refutation", existence: "given_witness",
                   first_rule: "given_rule_1", turn_first: "given_turn_1", turn_second: "given_turn_2",
                   second_rule: "given_rule_2"}
    return chapter, assumptions


class OrConvergenceBranchWitnessTests(unittest.TestCase):
    def test_kernel_obtains_chained_branch_witnesses_from_their_rows(self):
        chapter, assumptions = branch_witness_fixture()
        lines, _ = _render_chapter_steps(chapter, "branch_witnesses", assumptions, {},
            ambient_variables={"x", "y", "z"}, binary_definitions=BRANCH_WITNESS_DEFINITIONS)
        body = "\n".join(lines)
        self.assertIn("      obtain ⟨w, witness_guard_1⟩ := row_10 branch_compact\n"
                      "      obtain ⟨v, witness_guard_2⟩ := row_13 branch_compact w witness_guard_1\n"
                      "      exact row_16 branch_compact w witness_guard_1 v witness_guard_2", body)
        support = (LEAN_PROJECT / "GLExport" / "ProofSupport.lean").read_text(encoding="utf-8")
        source = (support + "\nopen GLExport\n"
                  "def gl_or99 {α : Type} (x y z : α) : Prop := x = y ∨ x = z\n"
                  "def gl_ex1 {α : Type} (N : α → Prop) (x : α) : Prop := ¬ (∀ w, N w → ¬ (w = x))\n"
                  "example {α : Type} (N : α → Prop) (x y z : α)\n"
                  " (given_or : gl_or99 x y z) (given_refutation : ¬ (x = z))\n"
                  " (given_witness : gl_ex1 N x)\n"
                  " (given_rule_1 : ∀ (w : α), w = x → gl_ex1 N w)\n"
                  " (given_turn_1 : ∀ (w : α), w = x → x = w)\n"
                  " (given_turn_2 : ∀ (v w : α), v = w → w = v)\n"
                  " (given_rule_2 : ∀ (v w : α), w = v → x = w → N x) : N x := by\n"
                  + body + "\n  exact row_17\n")
        with tempfile.TemporaryDirectory(prefix="branch_witnesses_") as directory:
            path = Path(directory) / "BranchWitnesses.lean"
            path.write_text(source, encoding="utf-8")
            result = subprocess.run([lean_executable(), str(path)],
                cwd=LEAN_PROJECT, env=lean_environment(),
                text=True, encoding="utf-8", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + "\n" + source)


def prerequisite_witness_fixture():
    """A case branch whose converged fact carries only its second witness v.

    v's existence is built on a fact about the first witness w, so v's row needs
    w's guard; the row that carries v into the converged fact obtained w from w's
    own row. The convergence therefore obtains w before it can obtain v (line 20's
    weak step: v15's row needs V11, which its branch fact does not carry)."""
    source = "(or99[x,y,z])"
    case = "(=[x,y])"
    refutation = "!(=[x,z])"
    existence = "(ex1[N,x])"
    first_turn = "(>[w](=[w,x])(=[x,w]))"
    first_rule = "(>[w](=[x,w])(ex1[N,y]))"
    second_turn = "(>[v](=[v,y])(=[y,v]))"
    join = "(>[w](=[w,x])(>[v](=[y,v])(in[v,N])))"
    last_rule = "(>[v](in[v,N])(in[x,N]))"
    result = "(in[x,N])"
    child = "main_boundary_ordis_" + source + "_(" + case + ")"
    context = {"namespace": child, "parent_namespace": "main", "scope_kind": "or_branch",
               "compact": {"mpl": source}, "template": {"mpl": "(>[]" + case + case + ")"}}
    steps = []

    def row(expression, action, namespace, dependencies, **metadata):
        step = {"id": f"chapter_1_line_{len(steps) + 1}", "source_line": len(steps) + 1,
                "source_tag": action, "action": action, "namespace": namespace,
                "conclusion": {"mpl": expression}, "dependencies": dependencies, **metadata}
        steps.append(step)
        return {"step": step["id"], "namespace": namespace, "expression": step["conclusion"],
                "source_kind": "local_step"}

    or_source = row(source, "assume", "main", [])
    refuted = row(refutation, "assume", "main", [])
    given_existence = row(existence, "assume", "main", [])
    turn_1 = row(first_turn, "assume", "main", [])
    rule_1 = row(first_rule, "assume", "main", [])
    turn_2 = row(second_turn, "assume", "main", [])
    joining = row(join, "assume", "main", [])
    rule_2 = row(last_rule, "assume", "main", [])
    local_existence = row(existence, "scope_inherit", child, [given_existence])
    first_body = row("!(>[w](in[w,N])!(=[w,x]))", "definition_expand", child, [local_existence])
    first_link = row("(=[w,x])", "compound_project", child, [first_body])
    first_turned = row("(=[x,w])", "implication_apply", child, [turn_1, first_link])
    built = row("(ex1[N,y])", "implication_apply", child, [rule_1, first_turned])
    second_body = row("!(>[v](in[v,N])!(=[v,y]))", "definition_expand", child, [built])
    second_link = row("(=[v,y])", "compound_project", child, [second_body])
    second_turned = row("(=[y,v])", "implication_apply", child, [turn_2, second_link])
    joined = row("(in[v,N])", "implication_apply", child, [joining, first_link, second_turned])
    branch = row(result, "implication_apply", child, [rule_2, joined])
    row(result, "or_eliminate", "main", [or_source, branch, refuted], or_definition_head="or99")
    chapter = {"steps": steps, "scope_contexts": [context],
               "variable_types": {"N": "set", "x": "element", "y": "element", "z": "element",
                                  "w": "element", "v": "element"}}
    assumptions = {source: "given_or", refutation: "given_refutation", existence: "given_witness",
                   first_turn: "given_turn_1", first_rule: "given_rule_1", second_turn: "given_turn_2",
                   join: "given_join", last_rule: "given_rule_2"}
    return chapter, assumptions


class OrConvergencePrerequisiteWitnessTests(unittest.TestCase):
    def render(self, chapter, assumptions):
        lines, _ = _render_chapter_steps(chapter, "prerequisite_witness", assumptions, {},
            ambient_variables={"x", "y", "z"}, binary_definitions=BRANCH_WITNESS_DEFINITIONS)
        return "\n".join(lines)

    def test_kernel_obtains_a_witness_the_branch_fact_does_not_carry_first(self):
        chapter, assumptions = prerequisite_witness_fixture()
        body = self.render(chapter, assumptions)
        # The converged fact (row 18) carries only v; v's row (row 15) needs w,
        # obtained first from w's own row (row 11).
        self.assertIn("      obtain ⟨w, witness_guard_1⟩ := row_11 branch_compact\n"
                      "      obtain ⟨v, witness_guard_2⟩ := row_15 branch_compact w witness_guard_1\n"
                      "      exact row_18 branch_compact v witness_guard_2", body)
        support = (LEAN_PROJECT / "GLExport" / "ProofSupport.lean").read_text(encoding="utf-8")
        source = (support + "\nopen GLExport\n"
                  "def gl_or99 {α : Type} (x y z : α) : Prop := x = y ∨ x = z\n"
                  "def gl_ex1 {α : Type} (N : α → Prop) (x : α) : Prop := ¬ (∀ w, N w → ¬ (w = x))\n"
                  "example {α : Type} (N : α → Prop) (x y z : α)\n"
                  " (given_or : gl_or99 x y z) (given_refutation : ¬ (x = z))\n"
                  " (given_witness : gl_ex1 N x)\n"
                  " (given_turn_1 : ∀ (w : α), w = x → x = w)\n"
                  " (given_rule_1 : ∀ (w : α), x = w → gl_ex1 N y)\n"
                  " (given_turn_2 : ∀ (v : α), v = y → y = v)\n"
                  " (given_join : ∀ (w : α), w = x → ∀ (v : α), y = v → N v)\n"
                  " (given_rule_2 : ∀ (v : α), N v → N x) : N x := by\n"
                  + body + "\n  exact row_19\n")
        with tempfile.TemporaryDirectory(prefix="prerequisite_witness_") as directory:
            path = Path(directory) / "PrerequisiteWitness.lean"
            path.write_text(source, encoding="utf-8")
            result = subprocess.run([lean_executable(), str(path)],
                cwd=LEAN_PROJECT, env=lean_environment(),
                text=True, encoding="utf-8", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + "\n" + source)


if __name__ == "__main__":
    unittest.main()
