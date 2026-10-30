# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt)
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""A copy of a case witness, and a guarded subproof discharged into the case.

The successor case of a twin weak step block copies its witness q as `rec2`
and opens subproofs whose facts are guarded by witnesses of the case. This
chapter mirrors that in miniature: inside an OR case, the witness w is copied
as `rec2`, a second witness V is projected from an existence that names
`rec2`, and a subproof about V is discharged into the case. The Lean export
must read `rec2` as w when it guards rows (w is stated before V) and must take
the case's witnesses from the guards it introduced when it enters the case.
"""

from pathlib import Path
import subprocess
import tempfile
import unittest

from proof_export.certificate import _chapter_certificate
from proof_export.graph import Chapter, Dependency, ProofRow, parse_scoped_expression
from proof_export.lean import _chapter_copy_sources, _chapter_replacements, _render_chapter_steps
from proof_export.mpl import to_mpl
from tests.lean_toolchain import LEAN_PROJECT, lean_environment, lean_executable

CASE = "main_boundary_ordis_(or98[x,A])_((in[x,A]))"
SUBPROOF = CASE + "_boundary_(implication99[V,R])"
OR_FACT = "(or98[x,A])"
EXISTENCE = "!(>[w](in[w,A])!(in2[w,x,R]))"
RULE = "(>[u](in2[u,x,R])(ex98[u,A,R]))"
GOAL = "(>[y](in2[V,y,R])(in2[V,y,R]))"
DEFINITIONS = {
    "or98": {"arity": 2, "category": "or", "signature": "(or98[u_1,u_2])",
             "elements": ["(in[u_1,u_2])", "(=[u_1,u_1])"]},
    "ex98": {"arity": 3, "category": "existence", "signature": "(ex98[u_1,u_2,u_3])",
             "elements": ["(in[1,u_2])", "(in2[u_1,1,u_3])"]},
    "implication99": {"arity": 2, "category": "implication", "signature": "(implication99[u_1,u_2])",
                      "elements": ["(in2[u_1,1,u_2])", "(in2[u_1,1,u_2])"]},
}
SIGNATURES = {
    "in": ("element", "set"),
    "in2": ("element", "element", "binary_relation"),
    "or98": ("element", "set"),
    "ex98": ("element", "set", "binary_relation"),
    "implication99": ("element", "binary_relation"),
}
LEAN_DEFINITIONS = (
    "def gl_or98 {α : Type} (x : α) (A : GLSet α) : Prop := A x ∨ x = x\n"
    "def gl_ex98 {α : Type} (u : α) (A : GLSet α) (R : GLBinaryRelation α) : Prop :=\n"
    "  ¬ ∀ (V : α), A V → ¬ R u V\n"
    "def gl_implication99 {α : Type} (V : α) (R : GLBinaryRelation α) : Prop :=\n"
    "  ∀ (y : α), R V y → R V y\n"
)
ASSUMPTIONS = {OR_FACT: "given_or", EXISTENCE: "given_witness", RULE: "given_rule"}
# The subproof folded into the case is quantified over w before V.
FOLDED_IN_CASE = ("have row_1 : ((A x) → (∀ (w : α), ((R w x) → (∀ (V : α), ((R w V) → "
                  "(gl_implication99 V R))))))")


def proof_row(line, conclusion, namespace, tag, *dependencies):
    """
    @brief Build one GL proof row of the fixture chapter.
    @param line Source line of the row.
    @param conclusion Scoped MPL text of the row.
    @param namespace Namespace the row lives in.
    @param tag GL proof tag.
    @param dependencies Pairs of cited scoped MPL text and its namespace.
    @return The proof row.
    """
    return ProofRow(line, parse_scoped_expression(conclusion), namespace, tag,
                    tuple(Dependency(parse_scoped_expression(text), scope)
                          for text, scope in dependencies))


# As in a GL chapter file, line 1 comes first and a row's dependencies sit on
# later lines; the certificate replays the rows from the last line up.
ROWS = (
    proof_row(1, "(implication99[V,R])", CASE, "validity name", ("(in2[V,y,R])", SUBPROOF)),
    proof_row(2, "(in2[V,y,R])", SUBPROOF, "premise element", (GOAL + "_integration_goal", CASE)),
    proof_row(3, GOAL + "_integration_goal", CASE, "expansion for integration",
              ("(implication99[V,R])_integration_goal", CASE)),
    proof_row(4, "(in2[rec2,V,R])", CASE, "disintegration", ("!(>[V](in[V,A])!(in2[rec2,V,R]))", CASE)),
    proof_row(5, "!(>[V](in[V,A])!(in2[rec2,V,R]))", CASE, "expansion", ("(ex98[rec2,A,R])", CASE)),
    proof_row(6, "(ex98[rec2,A,R])", CASE, "implication", (RULE, "main"), ("(in2[rec2,x,R])", CASE)),
    proof_row(7, RULE, "main", "task formulation"),
    proof_row(8, "(in2[rec2,x,R])", CASE, "equality1", ("(in2[w,x,R])", CASE), ("(=[w,rec2])", CASE)),
    proof_row(9, "(=[w,rec2])", CASE, "variable copy"),
    proof_row(10, "(in2[w,x,R])", CASE, "disintegration", (EXISTENCE, CASE)),
    proof_row(11, EXISTENCE, CASE, "zoom in", (EXISTENCE, "main")),
    proof_row(12, EXISTENCE, "main", "task formulation"),
    proof_row(13, "(in[x,A])", CASE, "or disintegration", (OR_FACT, "main")),
    proof_row(14, OR_FACT, "main", "task formulation"),
)


class WitnessCopyInOrCaseTests(unittest.TestCase):
    def render(self):
        """
        @brief Certify the fixture chapter and render it as Lean facts.
        @return The rendered lines, joined, and the manifest dispositions.
        """
        with tempfile.TemporaryDirectory(prefix="witness_copy_") as directory:
            path = Path(directory) / "1_single_chapter_proof.txt"
            path.write_text("\n".join("\t".join([to_mpl(row.conclusion.expression), row.namespace, row.tag])
                                      for row in ROWS), encoding="utf-8")
            certificate = _chapter_certificate(Chapter(path, 1, "single_chapter_proof", ROWS), SIGNATURES,
                                               {}, {}, certificate_schema=3, binary_entries=DEFINITIONS)
        lines, manifest = _render_chapter_steps(certificate, "witness_copy_in_or_case", ASSUMPTIONS,
                                                _chapter_replacements(certificate),
                                                ambient_variables={"x", "A", "R"},
                                                binary_definitions=DEFINITIONS,
                                                copy_sources=_chapter_copy_sources(certificate))
        return "\n".join(lines), manifest

    def test_the_copy_is_guarded_by_its_source_witness_before_the_witness_it_names(self):
        rendered, manifest = self.render()
        self.assertEqual(len(manifest), len(ROWS))
        # V's guard names rec2, which is w: w and its guard come first.
        self.assertIn("have row_4 : ((A x) → (∀ (w : α), ((R w x) → (∃ (V : α), (R w V)))))", rendered)
        self.assertIn(FOLDED_IN_CASE, rendered)

    def test_the_discharge_takes_the_case_witnesses_from_the_case_entry(self):
        rendered, _ = self.render()
        discharge = rendered[rendered.index("  have row_1 :"):]
        self.assertIn("    intro w\n    intro witness_guard_1\n    intro V\n    intro witness_guard_2\n"
                      "    simp only [gl_implication99]\n    intro y\n    intro scope_premise_2\n"
                      "    have scoped_fact_1 := row_2 V scope_premise_1 y scope_premise_2 w "
                      "witness_guard_1 witness_guard_2\n    exact scoped_fact_1", discharge)

    def test_without_the_copy_map_the_copy_is_not_read_as_its_witness(self):
        # The copy map is what guards the folded row by w: without it V's
        # guard, which names rec2, is not seen to need w.
        with tempfile.TemporaryDirectory(prefix="witness_copy_") as directory:
            path = Path(directory) / "1_single_chapter_proof.txt"
            path.write_text("fixture", encoding="utf-8")
            certificate = _chapter_certificate(Chapter(path, 1, "single_chapter_proof", ROWS), SIGNATURES,
                                               {}, {}, certificate_schema=3, binary_entries=DEFINITIONS)
        lines, _ = _render_chapter_steps(certificate, "witness_copy_in_or_case", ASSUMPTIONS,
                                         _chapter_replacements(certificate),
                                         ambient_variables={"x", "A", "R"}, binary_definitions=DEFINITIONS)
        self.assertNotIn(FOLDED_IN_CASE, "\n".join(lines))

    def test_kernel_accepts_the_chapter(self):
        rendered, _ = self.render()
        support = (Path(__file__).resolve().parents[1] / "lean_export/GLExport/ProofSupport.lean"
                   ).read_text(encoding="utf-8")
        source = (support + "\nopen GLExport\n" + LEAN_DEFINITIONS
                  + "example {α : Type} (x : α) (A : GLSet α) (R : GLBinaryRelation α)\n"
                  + "    (given_or : gl_or98 x A) (given_witness : ¬ ∀ (w : α), A w → ¬ R w x)\n"
                  + "    (given_rule : ∀ (u : α), R u x → gl_ex98 u A R) : True := by\n"
                  + "  classical\n" + rendered + "\n  trivial\n")
        with tempfile.TemporaryDirectory(prefix="witness_copy_lean_") as directory:
            path = Path(directory) / "WitnessCopy.lean"
            path.write_text(source, encoding="utf-8")
            result = subprocess.run([lean_executable(), str(path)], cwd=LEAN_PROJECT, env=lean_environment(),
                                    text=True, encoding="utf-8", stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + "\n" + rendered)


LATE_CASE = "main_boundary_ordis_(or98[x,A])_((in[x,A]))"
LATE_EXISTENCE = "!(>[V](in[V,A])!(in2[x,V,R]))"
LATE_DEFINITIONS = {
    "ex97": {"arity": 3, "category": "existence", "elements": ["(in[1,u_2])", "(in2[u_1,1,u_3])"]},
}


def late_witness_chapter():
    """
    @brief Build a case whose integration rule names a witness projected later.
    @details
    Inside an OR case, the integration goal `ex97[x,A,R]` expands to its rule
    with the existence's local V free, and the empty-bound reformulation states
    that rule; only afterwards does the case project its own witness named V
    (line 20's weak step: `implication125[V12,…,v15,…]` before V12 and v15).
    @return Neutral certificate chapter for `_render_chapter_steps`.
    """
    steps = []

    def row(expression, action, namespace, dependencies, scope="ordinary", **metadata):
        step = {"id": f"chapter_1_line_{len(steps) + 1}", "source_line": len(steps) + 1,
                "source_tag": action, "action": action, "namespace": namespace,
                "conclusion": {"mpl": expression, "proof_scope": scope},
                "dependencies": dependencies, **metadata}
        steps.append(step)
        return {"step": step["id"], "namespace": namespace, "expression": step["conclusion"],
                "source_kind": "local_step"}

    given = row(LATE_EXISTENCE, "assume", "main", [])
    goal = row("(>[](in[V,A])(in2[x,V,R]))", "definition_expand", LATE_CASE,
               [{"namespace": LATE_CASE, "source_kind": "integration_goal_template",
                 "expression": {"mpl": "(ex97[x,A,R])", "proof_scope": "integration_goal"}}],
               scope="integration_goal")
    row("(>[](in[V,A])(>[](in2[x,V,R])(ex97[x,A,R])))", "classical_reformulate_empty", LATE_CASE,
        [goal], reformulation_definition="ex97")
    inherited = row(LATE_EXISTENCE, "scope_inherit", LATE_CASE, [given])
    row("(in2[x,V,R])", "compound_project", LATE_CASE, [inherited])
    context = {"namespace": LATE_CASE, "parent_namespace": "main", "scope_kind": "or_branch",
               "compact": {"mpl": "(or98[x,A])"}, "template": {"mpl": "(>[](in[x,A])(in[x,A]))"}}
    return {"steps": steps, "scope_contexts": [context],
            "variable_types": {"x": "element", "A": "set", "R": "binary_relation", "V": "element"}}


class IntegrationRuleNamingALaterWitnessTests(unittest.TestCase):
    def test_the_rule_is_quantified_over_the_name_and_enters_the_case(self):
        lines, manifest = _render_chapter_steps(late_witness_chapter(), "late_witness",
                                                {LATE_EXISTENCE: "given_witness"}, {},
                                                ambient_variables={"x", "A", "R"},
                                                binary_definitions=LATE_DEFINITIONS)
        rendered = "\n".join(lines)
        self.assertEqual(len(manifest), 5)
        self.assertIn("  have row_3 : (∀ (V : α), ((A x) → ((A V) → ((R x V) → (gl_ex97 x A R))))) := by\n"
                      "    intro V\n    intro scope_premise_1\n"
                      "    intro integration_premise_1\n    intro integration_premise_2\n"
                      "    apply (row_2).2\n", rendered)
        support = (Path(__file__).resolve().parents[1] / "lean_export/GLExport/ProofSupport.lean"
                   ).read_text(encoding="utf-8")
        source = (support + "\nopen GLExport\n"
                  "def gl_ex97 {α : Type} (x : α) (A : GLSet α) (R : GLBinaryRelation α) : Prop :=\n"
                  "  ¬ ∀ (V : α), A V → ¬ R x V\n"
                  "example {α : Type} (x : α) (A : GLSet α) (R : GLBinaryRelation α)\n"
                  "    (given_witness : ¬ ∀ (V : α), A V → ¬ R x V) : True := by\n"
                  + rendered + "\n  trivial\n")
        with tempfile.TemporaryDirectory(prefix="late_witness_") as directory:
            path = Path(directory) / "LateWitness.lean"
            path.write_text(source, encoding="utf-8")
            result = subprocess.run([lean_executable(), str(path)], cwd=LEAN_PROJECT, env=lean_environment(),
                                    text=True, encoding="utf-8", stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + "\n" + rendered)


if __name__ == "__main__":
    unittest.main()
