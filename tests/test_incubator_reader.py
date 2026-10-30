"""Incubator shares actual graph reshuffling and the complete proof reader."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

import generate_full_proof_graph as generator
import visu_helpers
from process_proof_graphs import _build_fname_list
from proof_convergence import process_to_fixed_point
from proof_hierarchy import is_reader_theorem, order_main_hierarchies, resolved_dependencies
from proof_reader import expanded_rows, reader_plan
from proof_reader_html import complete_caption, render_reader_proof
from tests.test_proof_reader import disintegration_rows


class IncubatorReaderTests(unittest.TestCase):
    def test_both_incubator_anchors_share_reshuffling_without_row_rewrites(self):
        for anchor in ("AnchorIncubator3", "AnchorIncubator8"):
            with self.subTest(anchor=anchor):
                source = f"(>[N]({anchor}[N])(in[a,N]))"
                consumer = f"(>[N]({anchor}[N])(in[b,N]))"
                registry = [[consumer,"single chapter proof","-1"], [source,"single chapter proof","-1"]]
                groups = _build_fname_list(registry)
                rows = disintegration_rows()
                stacks = {groups[0][0]: [*rows, [source,"main","theorem"]], groups[1][0]: [["(in[a,N])","main","task formulation"]]}
                original = deepcopy(stacks)
                self.assertTrue(is_reader_theorem(source))
                ordered, result = process_to_fixed_point(registry, stacks, _build_fname_list, (), Path("unused-for-reshuffling"))
                self.assertEqual(ordered, list(reversed(registry)))
                self.assertEqual(stacks, original)
                for before, after in zip(groups, reversed(_build_fname_list(ordered))):
                    self.assertEqual(Counter(map(tuple, original[before[0]])), Counter(map(tuple, result[after[0]])))
                nested = order_main_hierarchies(ordered, result, _build_fname_list(ordered), ())
                self.assertEqual(order_main_hierarchies(ordered, nested, _build_fname_list(ordered), ()), nested)
                known = [row[0] for row in ordered]
                for rows in nested.values():
                    self.assertEqual(expanded_rows(reader_plan(rows,resolved_dependencies(rows,known))), list(reversed(range(len(rows)))))
                self.assertEqual(process_to_fixed_point(ordered,result,_build_fname_list,(),Path("unused-for-reshuffling")),(ordered,result))

    def test_unknown_anchors_still_do_not_claim_reader_support(self):
        self.assertFalse(is_reader_theorem("(>[N](AnchorUnknown[N])(in[a,N]))"))
        self.assertFalse(is_reader_theorem("(in[a,N])"))

    def test_incubator_single_chapter_uses_the_shared_reader(self):
        rows = [["(in[a,N])","main","task formulation"]]
        readable = visu_helpers.ReadableForm({}, {}, structured_implications=True)
        with patch.object(generator,"read_stack",return_value=deepcopy(rows)), patch.object(generator,"rename_stack"), patch.object(generator,"_readable_form_for_render",readable):
            text = generator.single_chapter_proof("(>[N](AnchorIncubator8[N])(in[a,N]))", "fixture", "c3s0")
        self.assertIn('data-proof-steps="1"',text)
        self.assertIn('class="common-line"',text)
        self.assertIn('id="c3s0-entry0"',text)

    def test_integration_targets_keep_a_visible_to_prove_label(self):
        readable = visu_helpers.ReadableForm({}, {}, structured_implications=True)
        target = ["(in[a,N])_integration_goal","main","expansion for integration"]
        caption = complete_caption(target,readable)
        self.assertIn("rf-target-label",caption)
        self.assertIn("TO PROVE:",caption)
        self.assertNotIn("rf-reads-label",caption)
        self.assertNotIn("_integration_goal",caption)
        fact = complete_caption(["(in[a,N])","main","task formulation"],readable)
        self.assertNotIn("TO PROVE:",fact)
        self.assertIn("rf-reads-label",fact)

    def test_reader_expressions_preserve_definition_popup_parts(self):
        rows = disintegration_rows()
        rows = [rows[i] for i in reversed(expanded_rows(reader_plan(rows,resolved_dependencies(rows,()))))]
        definitions = {"and9":{"category":"and","signature":"(and9[u_1,u_2,u_3])","elements":["(in[u_1,u_3])","(in[u_2,u_3])"]}}
        readable = visu_helpers.ReadableForm(definitions,{},structured_implications=True)
        with patch.object(generator,"_readable_form_for_render",readable):
            text = render_reader_proof(rows,"c4s0",generator,())
        from html.parser import HTMLParser
        spans = []
        class Parser(HTMLParser):
            def handle_starttag(self,tag,attrs):
                data=dict(attrs)
                if "clickable" in data.get("class","").split():
                    spans.append(data)
        Parser().feed(text)
        self.assertTrue(spans)
        for span in spans:
            self.assertEqual(span["data-parts"],generator._extract_expr_parts(span["data-text"]))
        self.assertTrue(any(span["data-parts"]=="and9" for span in spans))

    def test_or_introduction_uses_negated_premise_and_restores_explanation(self):
        from unittest.mock import Mock
        ns = "main_boundary_orint_(or1[a,b])_((in[a,N]))"
        rows = [
            ["(or1[a,b])", "main", "or branch proven", "(in[a,N])", ns],
            ["(in[a,N])", ns, "task formulation"],
            ["!(in[b,N])", ns, "or branch assumption", "(or1[a,b])_integration_goal", "main"],
        ]
        _, scopes = generator._partition_stack_subproofs(rows)
        info = scopes[0]["title_info"]
        self.assertEqual(info["case_expressions"], ["!(in[b,N])"])
        self.assertEqual(info["goal_expr"], "(in[a,N])")
        readable = Mock()
        readable.formula.side_effect = lambda text: text
        with patch.object(generator,"_readable_form_for_render",readable):
            text = render_reader_proof(rows,"c212s0",generator,())
        header = text.split('data-closed-result="true"',1)[1].split('<div class="subproof-body">',1)[0]
        self.assertEqual(header.count("For this case, assume the negated premises:"),2)
        self.assertNotIn('<span class="reader-label">Case:</span>',header)
        self.assertIn("!(in[b,N])",header)
        self.assertIn("OR-introduction.</b>",header)
        self.assertIn("This branch proves the disjunct",header)
        self.assertNotIn("This branch asserts the disjunct",header)

    def test_refuted_or_case_has_contradiction_goal_and_explanation(self):
        from unittest.mock import Mock
        ns = "main_boundary_ordis_(or1[a,b])_((in[a,N]))"
        rows = [
            ["!(in[a,N])","main","contradiction","(in[a,N])",ns,"!(in[a,N])_integration_goal","main"],
            ["(in[a,N])",ns,"or disintegration","(or1[a,b])","main"],
            ["(or1[a,b])","main","task formulation"],
        ]
        _,scopes=generator._partition_stack_subproofs(rows)
        info=scopes[0]["title_info"]
        self.assertEqual(info["case_expressions"],["(in[a,N])"])
        self.assertIsNone(info["goal_expr"])
        self.assertEqual(info["goal_kind"],"contradiction")
        readable=Mock();readable.formula.side_effect=lambda text:text
        with patch.object(generator,"_readable_form_for_render",readable):
            text=render_reader_proof(rows,"c249s0",generator,())
        self.assertIn("Contradiction (case refuted)",text)
        self.assertIn("&#8869; (case refuted)",text)
        self.assertIn("OR-disintegration: refuted case.</b>",text)
        from html.parser import HTMLParser
        highlighted=[]
        class Highlights(HTMLParser):
            def __init__(self):
                super().__init__();self.stack=[]
            def handle_starttag(self,tag,attrs):
                data=dict(attrs)
                if "proof-line" in data.get("class","").split():
                    self.current=data["id"]
                if "goal-mpl" in data.get("class","").split():
                    highlighted.append(self.current)
        Highlights().feed(text)
        self.assertIn("c249s0-entry0",highlighted)
        self.assertNotIn("c249s0-entry1",highlighted)
        self.assertIn('href="#c249s0-entry0">step ',text)
        with self.assertRaisesRegex(AssertionError,"neither convergence nor refutation"):
            generator._partition_stack_subproofs(rows[1:])

    def test_prose_identifies_the_recorded_step_without_hiding_the_formula(self):
        readable=visu_helpers.ReadableForm({}, {}, structured_implications=True)
        for tag,phrase in (("task formulation","Given in this proof."),
                           ("or branch assumption","Assumed as a negated premise"),
                           ("equality1","Obtained by substitution using the cited equalities."),
                           ("expansion","Obtained by unfolding the definition."),
                           ("validity name","Obtained as a result of a subproof.")):
            caption=complete_caption(["(in[a,N])","main",tag],readable)
            self.assertIn('class="reader-prose"',caption)
            self.assertIn(phrase,caption)
            self.assertIn(readable.formula("(in[a,N])"),caption)
        with self.assertRaisesRegex(AssertionError,"Missing common-notation prose"):
            complete_caption(["(in[a,N])","main","unknown proof rule"],readable)

    def test_contradiction_seed_reads_as_an_assumption(self):
        from proof_reader_html import CONTRADICTION_SEED_PROSE, contradiction_seeds
        readable = visu_helpers.ReadableForm({}, {}, structured_implications=True)
        rows = [["!(=[a,b])","main","contradiction","!(in[a,N])","main","(in[a,N])","main","(=[a,b])","main"],
                ["(=[a,b])","main","task formulation"],
                ["(in[c,N])","main","task formulation"],
                ["!(in[d,N])","main","contradiction","(in[e,N])","main_boundary_ordis_or1_((in[d,N]))","!(in[e,N])","main"]]
        self.assertEqual(contradiction_seeds(rows),{("(=[a,b])","main")})
        seed = complete_caption(rows[1],readable,True)
        self.assertIn(CONTRADICTION_SEED_PROSE,seed)
        self.assertNotIn("Given in this proof.",seed)
        self.assertIn(readable.formula("(=[a,b])"),seed)
        self.assertIn("Given in this proof.",complete_caption(rows[2],readable))
        with self.assertRaisesRegex(AssertionError,"Contradiction seed with tag"):
            complete_caption(rows[0],readable,True)
        with self.assertRaisesRegex(AssertionError,"Contradiction seed outside main"):
            contradiction_seeds([["!(=[a,b])","main","contradiction","x","main","y","main","(=[a,b])","main_child"]])
        scoped = "main_boundary_contradiction_!(lt[a,b])"
        child = [["!(lt[a,b])","main","contradiction","!(=[a,b])",scoped,"(=[a,b])","main"],
                 ["(lt[a,b])",scoped,"task formulation"],
                 ["!(in[d,N])","main","contradiction","(in[e,N])","main_boundary_ordis_or3_((in[d,N]))","!(in[e,N])","main"]]
        self.assertEqual(contradiction_seeds(child),{("(lt[a,b])",scoped)})
        with self.assertRaisesRegex(AssertionError,"does not match its result"):
            contradiction_seeds([["!(lt[a,c])","main","contradiction","x",scoped,"!x","main"]])
        with self.assertRaisesRegex(AssertionError,"Unknown contradiction scope payload"):
            contradiction_seeds([["!(lt[a,b])","main","contradiction","x","main_boundary_other_y","!x","main"]])

    def test_chapter_controls_are_prominent_and_cover_every_section(self):
        from proof_reader_html import notation_toggle_html
        from html.parser import HTMLParser
        buttons=[]
        class Parser(HTMLParser):
            def handle_starttag(self,tag,attrs):
                if tag == 'button': buttons.append(dict(attrs))
        Parser().feed(notation_toggle_html(True, proof_controls=True))
        self.assertEqual([b['data-proof-action'] for b in buttons],['expand','collapse'])
        for button in buttons:
            self.assertEqual(button['type'],'button')
            self.assertIn('document.querySelectorAll',button['onclick'])
            self.assertIn('.reader-section .subproof-card',button['onclick'])
        self.assertIn('setProofCardOpen(c,true)',buttons[0]['onclick'])
        self.assertIn('setProofCardOpen(c,false)',buttons[1]['onclick'])
        self.assertNotIn('reader-page-controls',notation_toggle_html(True))
        self.assertEqual(notation_toggle_html(False, proof_controls=True),'')


if __name__ == "__main__":
    unittest.main()
