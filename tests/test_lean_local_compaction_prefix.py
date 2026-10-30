# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt)
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Local definition folds instantiate the exact main-scope source prefix."""
import copy,subprocess,tempfile,unittest
from pathlib import Path
from proof_export.lean import _render_chapter_steps
from tests.lean_toolchain import LEAN_PROJECT,lean_environment,lean_executable


def chapter(with_witness):
    source='(=[I,I])'
    steps=[{'id':'chapter_1_line_4','source_line':4,'source_tag':'variable name','action':'variable_alias','namespace':'main','conclusion':{'mpl':source},'dependencies':[]}]
    assumption='!(>[I](in[w,I])!(in[w,I]))'
    if with_witness:
        steps.extend([{'id':'chapter_1_line_3','source_line':3,'source_tag':'task formulation','action':'assume','namespace':'main','conclusion':{'mpl':assumption},'dependencies':[]},
            {'id':'chapter_1_line_2','source_line':2,'source_tag':'disintegration','action':'compound_project','namespace':'main','conclusion':{'mpl':'(in[w,I])'},'dependencies':[{'step':'chapter_1_line_3','source_kind':'local_step','namespace':'main','expression':{'mpl':assumption}}]}])
    steps.append({'id':'chapter_1_line_1','source_line':1,'source_tag':'validity name','action':'definition_compact','namespace':'main','conclusion':{'mpl':'(same[I])'},'dependencies':[{'step':'chapter_1_line_4','source_kind':'local_step','namespace':'main','expression':{'mpl':source}}]})
    return {'steps':steps,'variable_types':{'I':'set','w':'element'}},assumption


class LocalDefinitionPrefixTests(unittest.TestCase):
    def test_kernel_accepts_generalized_bound_and_later_witness_cases(self):
        support=(Path(__file__).resolve().parents[1]/'lean_export/GLExport/ProofSupport.lean').read_text(encoding='utf-8')
        source=support+'\nopen GLExport\ndef gl_same {α : Type} (I : GLSet α) : Prop := I = I\n'
        for mode in ('generalized','ambient','witness'):
            data,assumption=chapter(mode=='witness')
            live={'w','I'} if mode=='ambient' else {'w'}
            lines,manifest=_render_chapter_steps(data,mode,{assumption:'witness_exists'}, {},
                ambient_variables=live,binary_definitions={'same':{'category':'and'}})
            rendered='\n'.join(lines)
            self.assertEqual(len(manifest),len(data['steps']))
            expected='using (row_4)' if mode=='ambient' else 'using (row_4 I)'
            self.assertIn(expected,rendered)
            if mode=='generalized':
                self.assertEqual(rendered.count('    intro I'),2)
                statement='example {α : Type} (w : α) : ∀ (I : GLSet α), gl_same I := by\n'
                ending='  exact row_1\n'
            elif mode=='ambient':
                statement='example {α : Type} (w : α) (I : GLSet α) : gl_same I := by\n';ending='  exact row_1\n'
            else:
                self.assertIn('obtain ⟨I, witness_row_3⟩',rendered)
                statement='example {α : Type} (w : α) (witness_exists : ¬ ∀ (I : GLSet α), I w → ¬ I w) : ∃ (I : GLSet α), gl_same I := by\n'
                ending='  exact ⟨I, row_1⟩\n'
            source+=statement+rendered+'\n'+ending
        with tempfile.TemporaryDirectory(prefix='local_fold_prefix_') as directory:
            path=Path(directory)/'Prefix.lean';path.write_text(source,encoding='utf-8')
            result=subprocess.run([lean_executable(),str(path)],cwd=LEAN_PROJECT,env=lean_environment(),text=True,encoding='utf-8',stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout)

    def test_local_compaction_rejects_foreign_or_missing_source_context(self):
        data,assumption=chapter(False);data['steps'][-1]['dependencies'][0]['namespace']='main_boundary_other'
        with self.assertRaises(AssertionError):_render_chapter_steps(data,'bad',{assumption:'h'},{},ambient_variables={'w'},binary_definitions={'same':{'category':'and'}})
        data,assumption=chapter(False);data['steps'][-1]['dependencies'][0]['step']='chapter_1_line_99'
        with self.assertRaises(AssertionError):_render_chapter_steps(data,'bad',{assumption:'h'},{},ambient_variables={'w'},binary_definitions={'same':{'category':'and'}})

    def test_kernel_accepts_scope_discharge_before_and_after_shared_witness(self):
        support=(Path(__file__).resolve().parents[1]/'lean_export/GLExport/ProofSupport.lean').read_text(encoding='utf-8')
        source=support+'\nopen GLExport\ndef gl_implication99 {α : Type} (I : GLSet α) : Prop := ∀ x, I x → I x\n'
        for mode in ('generalized','ambient','witness_before_source','witness_after_source'):
            has_witness=mode.startswith('witness')
            data,assumption=chapter(has_witness)
            scope='main_boundary_(implication99[I])'
            template='(>[x](in[x,I])(in[x,I]))'
            context={'namespace':scope,'parent_namespace':'main','definition_head':'implication99',
                     'compact':{'mpl':'(implication99[I])'},'template':{'mpl':template,'proof_scope':'integration_goal'}}
            data['scope_contexts']=[context];data['variable_types']['x']='element'
            first=data['steps'][0]
            first.update(namespace=scope,action='integration_premise',source_tag='premise element',
                conclusion={'mpl':'(in[x,I])'},scope_namespace=scope,
                dependencies=[{'namespace':'main','source_kind':'integration_goal_template','expression':{'mpl':template}}])
            data['steps'][-1].update(action='integration_discharge',conclusion={'mpl':'(implication99[I])'},scope_context=context,
                dependencies=[{'namespace':scope,'step':first['id'],'source_kind':'local_step','expression':first['conclusion']}])
            if mode=='witness_before_source':data['steps']=data['steps'][1:-1]+[first,data['steps'][-1]]
            live={'w','I'} if mode=='ambient' else {'w'}
            lines,_=_render_chapter_steps(data,mode,{assumption:'witness_exists'},{},ambient_variables=live,
                binary_definitions={'implication99':{'category':'implication'}})
            rendered='\n'.join(lines)
            expected='using (row_4)' if mode in ('ambient','witness_before_source') else 'using (row_4 I)'
            self.assertIn(expected,rendered)
            if mode=='generalized':
                statement='example {α : Type} (w : α) : ∀ (I : GLSet α), gl_implication99 I := by\n';ending='  exact row_1\n'
            elif mode=='ambient':
                statement='example {α : Type} (w : α) (I : GLSet α) : gl_implication99 I := by\n';ending='  exact row_1\n'
            else:
                statement='example {α : Type} (w : α) (witness_exists : ¬ ∀ (I : GLSet α), I w → ¬ I w) : ∃ (I : GLSet α), gl_implication99 I := by\n';ending='  exact ⟨I,row_1⟩\n'
            source+=statement+rendered+'\n'+ending
            invalid=copy.deepcopy(data);invalid['steps'][-1]['dependencies'][0]['namespace']='main_boundary_wrong'
            with self.assertRaises(AssertionError):_render_chapter_steps(invalid,'bad',{assumption:'witness_exists'},{},ambient_variables=live,binary_definitions={'implication99':{'category':'implication'}})
        with tempfile.TemporaryDirectory(prefix='scope_fold_prefix_') as directory:
            path=Path(directory)/'ScopePrefix.lean';path.write_text(source,encoding='utf-8')
            result=subprocess.run([lean_executable(),str(path)],cwd=LEAN_PROJECT,env=lean_environment(),text=True,encoding='utf-8',stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout)

if __name__=='__main__':unittest.main()
