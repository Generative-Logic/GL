# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
# Dual-licensed under AGPLv3 and commercial terms — https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Equality-supported case convergence without additional runtime facts."""

from copy import deepcopy
from dataclasses import replace
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from proof_export.mpl import parse_mpl, to_mpl
from proof_export.or_convergence import convergence_cases, match_equality_convergence
from proof_export.certificate import _schema3_action_metadata
from proof_export.graph import Chapter, Dependency, ProofRow, ScopedExpression
from proof_export.lean import _render_chapter_steps
from tests.test_harness import make_proof_line, make_state_minimal
from verifier import check_or_convergence
from tests.lean_toolchain import LEAN_PROJECT, lean_environment, lean_executable

PARENT = 'main_boundary_(implication99[N,n,g])'
OR = '(or99[i,n,N,g])'
TARGET = '(existence90[N,i,g])'
SOURCE = '(existence90[N,n,g])'
EQ = '(=[i,n])'
TEMPLATE = '(>[i]' + OR + '(>[]' + SOURCE + TARGET + '))'
DEFINITIONS = {
    'or99': {'arity': 4, 'category': 'or', 'signature': '(or99[u_1,u_2,u_3,u_4])',
             'elements': ['(existence90[u_3,u_1,u_4])', '(=[u_1,u_2])']},
    'existence90': {'arity': 3, 'category': 'existence', 'signature': '(existence90[u_1,u_2,u_3])',
                    'elements': ['(in[1,u_1])', '(in2[u_2,1,u_3])']},
    'implication99': {'arity': 3, 'category': 'implication', 'signature': '(implication99[u_1,u_2,u_3])',
                      'elements': ['(or99[1,u_2,u_1,u_3])', '(existence90[u_1,u_2,u_3])', '(existence90[u_1,1,u_3])']},
}
CASES = convergence_cases(parse_mpl(OR), PARENT, DEFINITIONS)
OLD, ENDPOINT = [scope for _, scope in CASES]


def support(ancestor=False):
    return [(parse_mpl(TARGET), OLD), (parse_mpl(SOURCE), PARENT if ancestor else ENDPOINT),
            (parse_mpl(EQ), ENDPOINT)]


def fixture(ancestor=False):
    def scoped(text, goal=False):
        return ScopedExpression(parse_mpl(text), text + ('_integration_goal' if goal else ''),
                                'integration_goal' if goal else 'ordinary')
    def dependency(text, scope, goal=False):
        return Dependency(scoped(text, goal), scope)
    rows = [
        ProofRow(1, scoped(TEMPLATE, True), 'main', 'expansion for integration',
                 (dependency('(implication99[N,n,g])', 'main', True),)),
        ProofRow(2, scoped(OR), PARENT, 'premise element', (dependency(TEMPLATE,'main',True),)),
        ProofRow(3, scoped(SOURCE), PARENT, 'premise element', (dependency(TEMPLATE,'main',True),)),
        ProofRow(4, scoped(TARGET), OLD, 'or disintegration', (dependency(OR,PARENT),)),
        ProofRow(5, scoped(EQ), ENDPOINT, 'or disintegration', (dependency(OR,PARENT),)),
    ]
    if not ancestor:
        rows.append(ProofRow(6,scoped(SOURCE),ENDPOINT,'zoom in',(dependency(SOURCE,PARENT),)))
    number = len(rows)+1
    rows.append(ProofRow(number,scoped(TARGET),PARENT,'or convergence',
        (dependency(OR,PARENT),*(dependency(to_mpl(expression),scope) for expression,scope in support(ancestor)))))
    rows.append(ProofRow(number+1,scoped('(implication99[N,n,g])'),'main','validity name',
        (dependency(TARGET,PARENT),)))
    chapter = Chapter(Path('or_equality_fixture.txt'),1,'single_chapter_proof',tuple(rows))
    metadata, contexts = _schema3_action_metadata(chapter,DEFINITIONS,{}, {})
    steps=[]
    by_pair={}
    actions={'premise element':'integration_premise','or disintegration':'or_branch',
             'zoom in':'scope_inherit','or convergence':'or_eliminate'}
    for row in rows[1:]:
        if row.tag == 'validity name':
            continue
        dependencies=[]
        for item in row.dependencies:
            entry={'namespace':item.namespace,'expression':{'mpl':to_mpl(item.expression)}}
            if item.scoped_expression.proof_scope == 'integration_goal':
                entry['source_kind']='integration_goal_template'
            else:
                entry['step']=by_pair[(to_mpl(item.expression),item.namespace)]
            dependencies.append(entry)
        step={'id':f'chapter_1_line_{row.source_line}','source_line':row.source_line,
              'source_tag':row.tag,'action':actions[row.tag],'namespace':row.namespace,
              'conclusion':{'mpl':to_mpl(row.conclusion.expression)},'dependencies':dependencies,
              **metadata.get(row.source_line,{})}
        steps.append(step)
        by_pair[(to_mpl(row.conclusion.expression),row.namespace)]=step['id']
    return chapter, {'steps':steps,'scope_contexts':contexts,
        'variable_types':{'N':'unary_relation','n':'element','g':'binary_relation','i':'element'}}


class EqualityConvergenceTests(unittest.TestCase):
    def test_exact_cases_and_both_orientations_and_ancestor_support(self):
        for ancestor in (False,True):
            for reverse in (False,True):
                evidence=support(ancestor)
                if reverse: evidence[-1]=(parse_mpl('(=[n,i])'),ENDPOINT)
                plan=match_equality_convergence(parse_mpl(TARGET),PARENT,CASES,evidence)
                self.assertEqual(plan,[{'source':0,'equalities':[],'retired':False},
                                       {'source':1,'equalities':[2],'retired':False}])

    def test_scope_sign_fixed_arguments_and_missing_evidence_rejected(self):
        for mode in ('missing','sibling_equality','descendant','other_parent','negative',
                     'fixed','repeated','predicate','unrelated','duplicate','binder'):
            evidence=support()
            if mode=='missing': evidence.pop()
            elif mode=='sibling_equality': evidence[-1]=(evidence[-1][0],OLD)
            elif mode=='descendant': evidence[-1]=(evidence[-1][0],ENDPOINT+'_boundary_deeper')
            elif mode=='other_parent': evidence[1]=(evidence[1][0],'main_boundary_other')
            elif mode=='negative': evidence[-1]=(parse_mpl('!'+EQ),ENDPOINT)
            elif mode=='fixed': evidence[1]=(parse_mpl('(existence90[M,n,g])'),ENDPOINT)
            elif mode=='repeated': evidence[1]=(parse_mpl('(existence90[N,n,n])'),ENDPOINT)
            elif mode=='predicate': evidence[1]=(parse_mpl('(existence91[N,n,g])'),ENDPOINT)
            elif mode=='unrelated': evidence.append((parse_mpl('(=[x,y])'),ENDPOINT))
            elif mode=='duplicate': evidence.append(evidence[-1])
            else: evidence[1]=(parse_mpl('(>[n](in[n,N])'+SOURCE+')'),ENDPOINT)
            with self.subTest(mode=mode):
                self.assertIsNone(match_equality_convergence(parse_mpl(TARGET),PARENT,CASES,evidence))

    def test_nested_cases_signed_conclusions_and_transitive_support(self):
        registry=deepcopy(DEFINITIONS)
        registry['or99']['elements'][0]='(or98[u_1,u_2,u_3,u_4])'
        registry['or98']={'category':'or','arity':4,'elements':[TARGET.replace('N','u_3').replace('i','u_1').replace('g','u_4'), '(r[u_1])']}
        # Exact atom substitutions, including the outer identity.
        registry['or98']['elements'][0]='(existence90[u_3,u_1,u_4])'
        cases=convergence_cases(parse_mpl(OR),PARENT,registry)
        self.assertEqual(len(cases),3)
        self.assertTrue(all('_ordis_'+OR in scope for _,scope in cases))
        target=parse_mpl('!'+TARGET)
        evidence=[(target,cases[0][1]),(target,cases[1][1]),(parse_mpl('!'+SOURCE),cases[2][1]),
                  (parse_mpl('(=[i,a])'),cases[2][1]),(parse_mpl('(=[a,n])'),PARENT)]
        self.assertIsNotNone(match_equality_convergence(target,PARENT,cases,evidence))

    def test_verifier_requires_every_actual_source_and_equality_row(self):
        state=make_state_minimal(); state.current_gl_binary=DEFINITIONS
        evidence=support()
        rest=[OR,PARENT,*[part for expression,scope in evidence for part in (to_mpl(expression),scope)]]
        row=make_proof_line(TARGET,PARENT,'or convergence',*rest)
        chapter=[make_proof_line(to_mpl(expression),scope,'task formulation') for expression,scope in evidence]
        self.assertTrue(check_or_convergence(row,chapter+[row],state))
        for missing in range(len(chapter)):
            self.assertFalse(check_or_convergence(row,chapter[:missing]+chapter[missing+1:]+[row],state))

    def test_certificate_and_kernel_keep_equalities_inside_the_case(self):
        root=Path(__file__).resolve().parents[1]
        for ancestor in (False,True):
            chapter,certificate=fixture(ancestor)
            lines,_=_render_chapter_steps(certificate,'or_equality',{}, {},
                ambient_variables={'N','n','g'},binary_definitions=DEFINITIONS)
            self.assertIn('      cases case_support_1',lines)
            bad=deepcopy(certificate);bad['steps'][-1]['or_equality_plan'][1]['equalities']=[]
            with self.assertRaises(AssertionError):
                _render_chapter_steps(bad,'invalid',{}, {},ambient_variables={'N','n','g'},binary_definitions=DEFINITIONS)
            source=('def gl_existence90 {α : Type} (N : α → Prop) (i : α) (g : α → α → Prop) : Prop := ∃ p, N p ∧ g i p\n'
                'def gl_or99 {α : Type} (i n : α) (N : α → Prop) (g : α → α → Prop) : Prop := gl_existence90 N i g ∨ i=n\n'
                'example {α : Type} (N : α → Prop) (n : α) (g : α → α → Prop) : ∀ i, gl_or99 i n N g → gl_existence90 N n g → gl_existence90 N i g := by\n'
                +'\n'.join(lines)+f'\n  exact row_{certificate["steps"][-1]["source_line"]}\n')
            with tempfile.TemporaryDirectory(prefix='or_equality_') as directory:
                path=Path(directory)/'Case.lean';path.write_text(source,encoding='utf-8')
                result=subprocess.run([lean_executable(),str(path)],cwd=LEAN_PROJECT,
                    env=lean_environment(),capture_output=True,text=True,encoding='utf-8',timeout=60)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)


if __name__=='__main__':
    unittest.main()
