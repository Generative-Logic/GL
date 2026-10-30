# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Positive projection, capture safety, independent verification and Lean replay."""
import copy,json,subprocess,tempfile,unittest
from pathlib import Path
from proof_export.definition_projection import (projection_alpha_equal,instantiate_projection_part,
    projection_children,find_definition_projection,build_definition_projection_plan)
from proof_export.mpl import parse_mpl,to_mpl
from proof_processing import compact_definition_projections,compact_main_definitions
from tests.lean_toolchain import LEAN_PROJECT,lean_environment,lean_executable


def definitions():
    return {'pair':{'category':'and','signature':'(pair[u_1,u_2,u_3,u_4])','arity':4,
                    'elements':['(in[u_1,u_2])','(in[u_3,u_4])']},
            'outer':{'category':'and','signature':'(outer[u_1,u_2,u_3,u_4])','arity':4,
                     'elements':['(pair[u_1,u_2,u_3,u_4])','(in[u_1,u_4])','(in[u_3,u_2])']}}


def chapter_rows():
    return [['(in[p,N])','main','disintegration','(&(in[i,I])(in[p,N]))','main'],
            ['(&(in[i,I])(in[p,N]))','main','expansion','(pair[i,I,p,N])','main'],
            ['(pair[i,I,p,N])','main','disintegration','(&(&(pair[i,I,p,N])(in[i,N]))(in[p,I]))','main'],
            ['(&(&(pair[i,I,p,N])(in[i,N]))(in[p,I]))','main','expansion','(outer[i,I,p,N])','main'],
            ['(outer[i,I,p,N])','main','task formulation']]


class ProjectionRuleTests(unittest.TestCase):
    def plan(self,target,source,registry=None):
        return build_definition_projection_plan(parse_mpl(target),parse_mpl(source),definitions() if registry is None else registry)

    def test_nested_projection_and_typed_registry_agree(self):
        plan=self.plan('(in[p,N])','(outer[i,I,p,N])')
        self.assertEqual([p['element'] for p in plan['path']],[0,1])
        self.assertEqual(plan['definition_unfolds'],2);self.assertEqual(plan['conjunction_projections'],3)
        typed=definitions()
        for entry in typed.values():entry['signature']=['element','set','element','set']
        self.assertEqual(self.plan('(in[p,N])','(outer[i,I,p,N])',typed),plan)
        self.assertEqual(self.plan('(in[p,N])','(&(in[i,I])(in[p,N]))',{})['conjunction_projections'],1)

    def test_negative_and_implication_members_are_opaque(self):
        registry=definitions();registry['pair']['elements'][1]='!(in[u_3,u_4])'
        self.plan('!(in[p,N])','(pair[i,I,p,N])',registry)
        registry['pair']['elements'][1]='(>[x](in[x,u_2])(in[x,u_4]))'
        self.plan('(>[z](in[z,I])(in[z,N]))','(pair[i,I,p,N])',registry)
        for target,source in [('(in[p,N])','!(pair[i,I,p,N])'),
                              ('(in[p,N])','(>[](in[p,I])(in[p,N]))'),
                              ('(in[p,N])','(pair[i,I,p,N])')]:
            self.assertIsNone(find_definition_projection(parse_mpl(target),parse_mpl(source),registry))
        for category in ('or','existence','implication'):
            registry['pair']['category']=category
            self.assertEqual(projection_children(parse_mpl('(pair[i,I,p,N])'),registry),[])

    def test_fixed_ports_polarity_identity_and_invalid_definitions_reject(self):
        for target in ('(in[q,N])','!(in[p,N])','(in[N,p])','(outer[i,I,p,N])'):
            with self.assertRaises(AssertionError):self.plan(target,'(outer[i,I,p,N])')
        with self.assertRaises(AssertionError):self.plan('(in[p,N])','(pair[i,I,p])')
        registry=definitions();registry['pair']['elements']=['(pair[u_1,u_2,u_3,u_4])']
        with self.assertRaisesRegex(AssertionError,'recursive'):self.plan('(in[p,N])','(pair[i,I,p,N])',registry)
        with self.assertRaisesRegex(AssertionError,'unbound'):
            instantiate_projection_part(parse_mpl('(in[x,u_1])'),{'u_1':'N'})

    def test_bound_free_collisions_and_capture_are_not_equivalent(self):
        left=parse_mpl('(>[x](P[x,alpha_1])(Q[x]))')
        right=parse_mpl('(>[y](P[alpha_1,y])(Q[y]))')
        self.assertFalse(projection_alpha_equal(left,right))
        self.assertTrue(projection_alpha_equal(left,parse_mpl('(>[z](P[z,alpha_1])(Q[z]))')))
        body=parse_mpl('(>[x](P[x,u_1])(Q[u_1]))')
        result=instantiate_projection_part(body,{'u_1':'x'})
        self.assertTrue(projection_alpha_equal(result,parse_mpl('(>[z](P[z,x])(Q[x]))')))
        self.assertFalse(projection_alpha_equal(result,parse_mpl('(>[x](P[x,x])(Q[x]))')))

    def test_verifier_requires_one_existing_same_scope_source(self):
        from tests.test_harness import make_proof_line,make_state_minimal
        from verifier import check_definition_projection,TAG_CHECKERS,_ORIGIN_EXEMPT_TAGS
        state=make_state_minimal();state.current_gl_binary=definitions()
        self.assertIs(TAG_CHECKERS['definition projection'],check_definition_projection)
        self.assertNotIn('definition projection',_ORIGIN_EXEMPT_TAGS)
        for namespace in ('main','main_boundary_case'):
            source=make_proof_line('(outer[i,I,p,N])',namespace,'task formulation')
            row=make_proof_line('(in[p,N])',namespace,'definition projection',source.expression,namespace)
            self.assertTrue(check_definition_projection(row,[source,row],state))
            self.assertFalse(check_definition_projection(row,[row],state))
            malformed=make_proof_line(row.expression,namespace,'definition projection',source.expression,namespace,'(in[p,N])')
            self.assertFalse(check_definition_projection(malformed,[source,malformed],state))
        source=make_proof_line('(outer[i,I,p,N])','main','task formulation')
        row=make_proof_line('(in[p,N])','main_boundary_case','definition projection',source.expression,'main')
        self.assertFalse(check_definition_projection(row,[source,row],state))
        row=make_proof_line('(in[q,N])','main','definition projection',source.expression,'main')
        self.assertFalse(check_definition_projection(row,[source,row],state))

    def test_compaction_is_real_and_preserves_boundary_sources(self):
        rows=chapter_rows();result=compact_definition_projections(rows,definitions())
        self.assertEqual(result,[[rows[0][0],'main','definition projection',rows[-1][0],'main'],rows[-1]])
        self.assertEqual(compact_definition_projections(result,definitions()),result)
        self.assertEqual(len(rows),5)
        rows=chapter_rows();rows[0][1]='main_boundary_child'
        result=compact_definition_projections(rows,definitions())
        self.assertEqual(result[0],rows[0])
        self.assertIn(rows[1],result)
        with self.assertRaisesRegex(AssertionError,'ambiguous'):compact_definition_projections(rows+[rows[-1]],definitions())

    def test_main_wrapper_uses_only_exact_existing_anchor_registry(self):
        theorem='(>[N](AnchorPeano[N])(in[x,N]))';incubator=theorem.replace('AnchorPeano','AnchorIncubator')
        stacks={'a':chapter_rows(),'b':chapter_rows()}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with self.assertRaisesRegex(AssertionError,'missing exact'):compact_main_definitions([[theorem]],{'a':stacks['a']},[['a']],root)
            (root/'GL_binary_Peano.json').write_text(json.dumps(definitions()),encoding='utf-8')
            result=compact_main_definitions([[theorem],[incubator]],stacks,[['a'],['b']],root)
            self.assertEqual(len(result['a']),2);self.assertEqual(result['b'],stacks['b'])

    def test_certificate_and_lean_kernel_replay_direct_terms(self):
        from proof_export.certificate import _chapter_certificate
        from proof_export.graph import Chapter,ProofRow,Dependency,parse_scoped_expression
        from proof_export.lean import _render_chapter_steps
        examples=[]
        with tempfile.TemporaryDirectory(prefix='definition_projection_') as directory:
            for kind in ('positive','negative','quantified'):
                registry=definitions();source='(outer[i,I,p,N])';target='(in[p,N])'
                part='N p'
                if kind=='negative':registry['pair']['elements'][1]='!(in[u_3,u_4])';target='!(in[p,N])';part='¬ N p'
                if kind=='quantified':registry['pair']['elements'][1]='(>[x](in[x,u_2])(in[x,u_4]))';target='(>[z](in[z,I])(in[z,N]))';part='∀ x, I x → N x'
                scoped=parse_scoped_expression(source)
                rows=(ProofRow(1,parse_scoped_expression(target),'main','definition projection',(Dependency(scoped,'main'),)),
                      ProofRow(2,scoped,'main','task formulation',()))
                path=Path(directory)/'1_single_chapter_proof.txt'
                path.write_text(f'{target}\tmain\tdefinition projection\t{source}\tmain\n{source}\tmain\ttask formulation\n',encoding='utf-8')
                signatures={'in':('element','set'),'pair':('element','set','element','set'),'outer':('element','set','element','set')}
                certificate=_chapter_certificate(Chapter(path,1,'single_chapter_proof',rows),signatures,{}, {},binary_entries=registry)
                self.assertEqual(certificate['steps'][-1]['action'],'definition_project')
                lines,manifest=_render_chapter_steps(certificate,'projection',{source:'source'},
                    {name:name for name in ('i','I','p','N')},ambient_variables={'i','I','p','N'},binary_definitions=registry)
                self.assertEqual(len(manifest),2)
                examples.append(f'namespace {kind}\n'
                    'def gl_pair {α : Type} (i : α) (I : α → Prop) (p : α) (N : α → Prop) : Prop := '
                    f'I i ∧ ({part})\n'
                    'def gl_outer {α : Type} (i : α) (I : α → Prop) (p : α) (N : α → Prop) : Prop := '
                    '(gl_pair i I p N ∧ N i) ∧ I p\n'
                    'example {α : Type} (i p : α) (I N : α → Prop) (source : gl_outer i I p N) : '
                    f'({part}) := by\n'+'\n'.join(lines)+f'\n  exact row_1\nend {kind}\n')
                tampered=copy.deepcopy(certificate);tampered['steps'][-1]['projection']['path'][0]['element']=1
                with self.assertRaises(AssertionError):_render_chapter_steps(tampered,'bad',{source:'source'},
                    {name:name for name in ('i','I','p','N')},ambient_variables={'i','I','p','N'},binary_definitions=registry)
            path=Path(directory)/'Projection.lean';path.write_text('\n'.join(examples),encoding='utf-8')
            result=subprocess.run([lean_executable(),str(path)],cwd=LEAN_PROJECT,env=lean_environment(),
                text=True,encoding='utf-8',stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout)

if __name__=='__main__':unittest.main()
