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


if __name__ == "__main__":
    unittest.main()
