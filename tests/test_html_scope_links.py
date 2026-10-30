"""HTML identities must follow visible proof rows and induction sections."""
from html.parser import HTMLParser
import unittest
from unittest.mock import patch

import generate_full_proof_graph as gen


class Links(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.ids=[]
        self.links=[]
        self.feed(text)
    def handle_starttag(self, tag, attributes):
        fields=dict(attributes)
        if "id" in fields:self.ids.append(fields["id"])
        if tag=="a" and "href" in fields:self.links.append(fields["href"])


class ProofScopeLinksTests(unittest.TestCase):
    def test_induction_parts_have_distinct_nested_identifiers(self):
        child="main_boundary_subproof_(in[x,N])"
        nested=child+"_boundary_subproof_(in[y,N])"
        rows=[["(in[z,N])","main","task formulation"],
              ["(in[x,N])",child,"task formulation"],
              ["(in[y,N])",nested,"task formulation"]]
        html="".join(gen.render_stack_with_subproofs(rows,prefix=prefix) for prefix in ("sub1","sub2"))
        parsed=Links(html)
        self.assertEqual(len(parsed.ids),len(set(parsed.ids)))
        for prefix in ("sub1","sub2"):
            self.assertIn(prefix+"sp1-anchor",parsed.ids)
            self.assertIn(prefix+"sp1n1-anchor",parsed.ids)

    def test_hidden_theorem_import_links_to_its_chapter(self):
        theorem="(>[N,x](in[x,N])(=[x,x]))"
        rows=[["(=[x,x])","main","implication",theorem,"main","(in[x,N])","main"],
              ["(in[x,N])","main","task formulation"],
              [theorem,"main","theorem"]]
        with patch.dict(gen.theorem_to_file,{gen.rename_theorem(theorem):"chapter99.html"},clear=True):
            parsed=Links(gen.render_stack_with_subproofs(rows))
        self.assertIn("chapter99.html",parsed.links)
        self.assertNotIn("#m-entry0",parsed.links)
        self.assertNotIn("m-entry0",parsed.ids)
        self.assertTrue(all(link[1:] in parsed.ids for link in parsed.links if link.startswith("#")))

    def test_subproof_hidden_import_has_no_missing_local_anchor(self):
        theorem="(>[N,x](in[x,N])(=[x,x]))"
        child="main_boundary_subproof_(in[x,N])"
        rows=[["(in[z,N])","main","task formulation"],
              ["(=[x,x])",child,"implication",theorem,child,"(in[x,N])",child],
              ["(in[x,N])",child,"task formulation"],
              [theorem,child,"theorem"]]
        with patch.dict(gen.theorem_to_file,{gen.rename_theorem(theorem):"chapter99.html"},clear=True):
            parsed=Links(gen.render_stack_with_subproofs(rows,prefix="sub1"))
        self.assertIn("chapter99.html",parsed.links)
        self.assertTrue(all(link[1:] in parsed.ids for link in parsed.links if link.startswith("#")))


if __name__=="__main__":
    unittest.main()

