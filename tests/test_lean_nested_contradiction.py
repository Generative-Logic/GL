# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Contradiction discharge consumes exact ancestor prefixes, never sibling facts."""

from copy import deepcopy
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from proof_export.lean import _render_chapter_steps
from tests.lean_toolchain import LEAN_PROJECT, lean_environment, lean_executable


class NestedContradictionTests(unittest.TestCase):
    def fixture(self, negative_child=False, extra_parent=False, same_scope=False,
                reverse_dependencies=False):
        outer = "main_boundary_outer"
        parent = outer + "_boundary_parent" if extra_parent else outer
        child = parent + "_boundary_child"
        positive = "(=[x,y])"
        negative = "!(=[x,y])"
        ancestor_premise, child_premise = ((positive, negative) if negative_child
                                          else (negative, positive))
        outer_template = f"(>[x]{ancestor_premise}(=[x,z]))"
        parent_tail = "(>[](=[z,z])(=[x,z]))" if extra_parent else "(=[x,z])"
        parent_template = f"(>[x]{ancestor_premise}{parent_tail})"
        child_tail = f"(>[]{child_premise}(=[x,z]))"
        if extra_parent:
            child_tail = f"(>[](=[z,z]){child_tail})"
        child_template = f"(>[x]{ancestor_premise}{child_tail})"
        contexts = {outer: {"namespace": outer, "template": {"mpl": outer_template}},
                    parent: {"namespace": parent, "template": {"mpl": parent_template}},
                    child: {"namespace": child, "template": {"mpl": child_template}}}
        for index, context in enumerate(contexts.values()):
            context["compact"] = {"mpl": f"(implication{90 + index}[y,z])"}
        contexts[outer]["parent_namespace"] = "main"
        if extra_parent:
            contexts[parent]["parent_namespace"] = outer
        contexts[child]["parent_namespace"] = parent
        steps = []
        for expression, scope in ((ancestor_premise, child if same_scope else outer),
                                  (child_premise, child)):
            index = len(steps) + 1
            steps.append({"id": f"chapter_1_line_{index}", "source_line": index,
                "namespace": scope, "scope_namespace": scope,
                "action": "integration_premise", "source_tag": "premise element",
                "conclusion": {"mpl": expression}, "dependencies": [{
                    "namespace": "main", "source_kind": "integration_goal_template",
                    "expression": deepcopy(contexts[scope]["template"])}]})
        dependencies = [{"step": step["id"], "namespace": step["namespace"],
                         "expression": step["conclusion"]} for step in steps]
        if reverse_dependencies:
            dependencies.reverse()
        steps.append({"id": "chapter_1_line_3", "source_line": 3,
            "namespace": parent, "action": "ex_falso", "source_tag": "contradiction",
            "conclusion": {"mpl": "!" + child_premise}, "dependencies": dependencies})
        chapter = {"steps": steps, "scope_contexts": list(contexts.values()),
                   "variable_types": {name: "element" for name in ("x", "y", "z")}}
        return chapter

    def render(self, chapter):
        lines, manifest = _render_chapter_steps(chapter, "nested_contradiction", {}, {},
            ambient_variables={"y", "z"})
        self.assertEqual(len(manifest), 3)
        return "\n".join(lines)

    def test_each_dependency_consumes_only_its_own_prefix(self):
        for negative_child in (False, True):
            for extra_parent in (False, True):
                for reverse_dependencies in (False, True):
                    with self.subTest(negative_child=negative_child,
                                     extra_parent=extra_parent,
                                     reverse_dependencies=reverse_dependencies):
                        proof = self.render(self.fixture(negative_child, extra_parent,
                            reverse_dependencies=reverse_dependencies))
                        self.assertIn("row_1 x scope_premise_1", proof)
                        self.assertNotIn("row_1 x scope_premise_1 contradiction_assumption", proof)
                        middle = " scope_premise_2" if extra_parent else ""
                        self.assertIn("row_2 x scope_premise_1" + middle
                                      + " contradiction_assumption", proof)
                        self.assertEqual(proof.count("intro contradiction_assumption"), 1)

    def test_siblings_with_identical_templates_are_rejected(self):
        chapter = self.fixture(same_scope=True)
        source = chapter["steps"][0]
        sibling = source["namespace"] + "_sibling"
        context = deepcopy(chapter["scope_contexts"][-1])
        context["namespace"] = sibling
        chapter["scope_contexts"].append(context)
        source["namespace"] = source["scope_namespace"] = sibling
        chapter["steps"][-1]["dependencies"][0]["namespace"] = sibling
        with self.assertRaisesRegex(AssertionError, "unrelated scopes"):
            self.render(chapter)

    def test_ancestor_names_with_mismatched_assumptions_are_rejected(self):
        chapter = self.fixture()
        template = "(>[x](=[x,z])(>[](=[x,y])(=[x,z])))"
        chapter["scope_contexts"][-1]["template"]["mpl"] = template
        chapter["steps"][1]["dependencies"][0]["expression"]["mpl"] = template
        with self.assertRaisesRegex(AssertionError, "incompatible scope assumptions"):
            self.render(chapter)

    def test_wrong_discharge_skipped_layer_and_child_binders_are_rejected(self):
        chapter = self.fixture()
        chapter["steps"][-1]["conclusion"]["mpl"] = "!(=[x,z])"
        with self.assertRaises(AssertionError):
            self.render(chapter)
        chapter = self.fixture(extra_parent=True)
        chapter["steps"][-1]["namespace"] = "main_boundary_outer"
        with self.assertRaises(AssertionError):
            self.render(chapter)
        chapter = self.fixture()
        template = "(>[x]!(=[x,y])(>[z](=[x,y])(=[x,z])))"
        chapter["scope_contexts"][-1]["template"]["mpl"] = template
        chapter["steps"][1]["dependencies"][0]["expression"]["mpl"] = template
        with self.assertRaises(AssertionError):
            self.render(chapter)

    def test_kernel_checks_ancestor_and_same_scope_evidence_for_both_polarities(self):
        root = Path(__file__).resolve().parent.parent
        source = ""
        for negative_child in (False, True):
            for extra_parent in (False, True):
                for same_scope in (False, True):
                    for reverse_dependencies in (False, True):
                        proof = self.render(self.fixture(negative_child, extra_parent,
                            same_scope, reverse_dependencies))
                        ancestor = "x = y" if negative_child else "¬ (x = y)"
                        conclusion = "¬ (¬ (x = y))" if negative_child else "¬ (x = y)"
                        middle = "z = z → " if extra_parent else ""
                        source += ("example {α : Type} (y z : α) : ∀ (x : α), "
                            + ancestor + " → " + middle + conclusion + " := by\n"
                            + proof + "\n  exact row_3\n\n")
        with tempfile.TemporaryDirectory(prefix="nested_contradiction_") as directory:
            path = Path(directory) / "NestedContradiction.lean"
            path.write_text(source, encoding="utf-8")
            result = subprocess.run([lean_executable(), str(path)],
                cwd=LEAN_PROJECT, env=lean_environment(),
                text=True, encoding="utf-8", stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout)


if __name__ == "__main__":
    unittest.main()
