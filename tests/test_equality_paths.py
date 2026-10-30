# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Earlier equality support and irrelevant-argument projection shortcuts."""
import json,tempfile,unittest
from pathlib import Path
from proof_shortcuts import shorten_equality_paths,shortcut_main_equalities
from verifier import ProofLine,check_equality1,check_definition_projection,VerifierState

REGISTRY={'pair':{'category':'and','signature':'(pair[u_1,u_2,u_3])','arity':3,
                  'elements':['(P[u_1])','(Q[u_2,u_3])']}}

def equality_rows():
    return [['(P[c])','main','equality1','(P[b])','main','(=[b,c])','main'],
            ['(P[b])','main','equality1','(P[a])','main','(=[a,b])','main'],
            ['(=[a,c])','main','task formulation'],['(=[b,c])','main','task formulation'],
            ['(=[a,b])','main','task formulation'],['(P[a])','main','task formulation']]

def projection_rows():
    return [['(P[a])','main','definition projection','(pair[a,d,c])','main'],
            ['(pair[a,d,c])','main','equality1','(pair[a,b,c])','main','(=[b,d])','main'],
            ['(=[b,d])','main','task formulation'],['(pair[a,b,c])','main','task formulation']]

class EqualityPathTests(unittest.TestCase):
    def test_external_direct_pair_shortens_without_transitive_inference(self):
        rows=equality_rows();result=shorten_equality_paths(rows,{})
        self.assertEqual(len(result),5)
        self.assertEqual(result[0],['(P[c])','main','equality1','(P[a])','main','(=[a,c])','main'])
        r=result[0];self.assertTrue(check_equality1(ProofLine(*r[:3],r[3:],'',1),[],None))
        self.assertEqual(shorten_equality_paths(result,{}),result)

    def test_missing_reversed_or_later_support_keeps_chain(self):
        rows=equality_rows();rows[2][0]='(=[c,a])'
        self.assertEqual(shorten_equality_paths(rows,{}),rows)
        rows=equality_rows();rows.insert(0,rows.pop(2))
        self.assertEqual(shorten_equality_paths(rows,{}),rows)

    def test_irrelevant_definition_argument_rewrite_is_bypassed(self):
        rows=projection_rows();result=shorten_equality_paths(rows,REGISTRY)
        self.assertEqual(len(result),3);self.assertEqual(result[0][3],'(pair[a,b,c])')
        lines=[ProofLine(*r[:3],r[3:],'',i+1) for i,r in enumerate(result)]
        state=VerifierState();state.current_gl_binary=REGISTRY
        self.assertTrue(check_definition_projection(lines[0],lines,state))
        self.assertEqual(shorten_equality_paths(result,REGISTRY),result)

    def test_relevant_arguments_shared_sources_and_scope_edges_are_preserved(self):
        rows=projection_rows();rows[0][0]='(Q[d,c])'
        self.assertEqual(shorten_equality_paths(rows,REGISTRY),rows)
        rows=projection_rows();rows.append(['(R[a])','main','implication',rows[1][0],'main'])
        self.assertEqual(shorten_equality_paths(rows,REGISTRY),rows)
        rows=projection_rows();rows[1][-1]='main_boundary_other'
        self.assertEqual(shorten_equality_paths(rows,REGISTRY),rows)

    def test_duplicate_identity_and_bad_shape_assert(self):
        rows=projection_rows()
        with self.assertRaises(AssertionError):shorten_equality_paths(rows+[rows[-1]],REGISTRY)
        rows[1]=rows[1][:5]
        with self.assertRaises(AssertionError):shorten_equality_paths(rows,REGISTRY)
        with self.assertRaises(AssertionError):shorten_equality_paths([],REGISTRY)

    def test_main_wrapper_preserves_incubator(self):
        theorem='(>[N](AnchorPeano[N])(in[x,N]))';other=theorem.replace('AnchorPeano','AnchorIncubator')
        stacks={'a':projection_rows(),'b':projection_rows()}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with self.assertRaises(AssertionError):shortcut_main_equalities([[theorem]],{'a':stacks['a']},[['a']],root)
            (root/'GL_binary_Peano.json').write_text(json.dumps(REGISTRY),encoding='utf-8')
            result=shortcut_main_equalities([[theorem],[other]],stacks,[['a'],['b']],root)
            self.assertEqual(len(result['a']),3);self.assertEqual(result['b'],stacks['b'])

if __name__=='__main__':unittest.main()
