"""Definition expansion applies each source at its recorded free-variable prefix."""
import copy,subprocess,tempfile,unittest
from pathlib import Path
from proof_export.lean import _render_chapter_steps
from tests.lean_toolchain import LEAN_PROJECT,lean_environment,lean_executable

ASSUMPTION='!(>[I](in[w,I])!(in[w,I]))'
def dependency(expression,number,namespace='main'):
 return {'step':f'chapter_1_line_{number}','source_kind':'local_step','namespace':namespace,'expression':{'mpl':expression}}
def row(number,action,conclusion,dependencies=(),namespace='main'):
 return {'id':f'chapter_1_line_{number}','source_line':number,'source_tag':action,'action':action,'namespace':namespace,'conclusion':{'mpl':conclusion},'dependencies':list(dependencies)}
def chapter(scoped,mode):
 scope='main_boundary_(implication99[I])' if scoped else 'main'
 source=[row(7,'variable_alias','(=[I,I])'),row(6,'definition_compact','(same[I])',[dependency('(=[I,I])',7)])]
 if scoped:source.append(row(5,'scope_inherit','(same[I])',[dependency('(same[I])',6)],scope))
 witness=[row(4,'assume',ASSUMPTION),row(3,'compound_project','(in[w,I])',[dependency(ASSUMPTION,4)])] if mode.startswith('witness') else []
 steps=witness+source if mode=='witness_before_source' else source+witness
 steps.append(row(1,'definition_expand','(=[I,I])',[dependency('(same[I])',5 if scoped else 6,scope)],scope))
 data={'steps':steps,'variable_types':{'I':'set','w':'element'}}
 if scoped:data['scope_contexts']=[{'namespace':scope,'parent_namespace':'main','definition_head':'implication99','compact':{'mpl':'(implication99[I])'},'template':{'mpl':'(>[](in[w,I])(in[w,I]))','proof_scope':'integration_goal'}}]
 return data

class DefinitionExpansionPrefixTests(unittest.TestCase):
 def test_kernel_checks_main_and_scoped_prefixes_around_shared_witness(self):
  support=(Path(__file__).resolve().parents[1]/'lean_export/GLExport/ProofSupport.lean').read_text(encoding='utf-8')
  source=support+'\nopen GLExport\ndef gl_same {α : Type} (I : GLSet α) : Prop := I = I\n'
  for scoped in (False,True):
   for mode in ('generalized','ambient','witness_before_source','witness_after_source'):
    data=chapter(scoped,mode);live={'w','I'} if mode=='ambient' else {'w'}
    lines,manifest=_render_chapter_steps(data,mode,{ASSUMPTION:'witness_exists'},{},ambient_variables=live,binary_definitions={'same':{'category':'and'}})
    self.assertEqual(len(manifest),len(data['steps']))
    term='row_5' if scoped else 'row_6';suffix='' if mode in ('ambient','witness_before_source') else ' I'
    self.assertIn('using ('+term+suffix+')','\n'.join(lines))
    goal='I w → I = I' if scoped else 'I = I'
    if mode=='generalized':statement=f'example {{α : Type}} (w : α) : ∀ (I : GLSet α), {goal} := by\n';ending='  exact row_1\n'
    elif mode=='ambient':statement=f'example {{α : Type}} (w : α) (I : GLSet α) : {goal} := by\n';ending='  exact row_1\n'
    else:statement=f'example {{α : Type}} (w : α) (witness_exists : ¬ ∀ (I : GLSet α), I w → ¬ I w) : ∃ (I : GLSet α), {goal} := by\n';ending='  exact ⟨I,row_1⟩\n'
    source+=statement+'\n'.join(lines)+'\n'+ending
  with tempfile.TemporaryDirectory(prefix='expansion_prefix_') as directory:
   path=Path(directory)/'Expansion.lean';path.write_text(source,encoding='utf-8')
   result=subprocess.run([lean_executable(),str(path)],cwd=LEAN_PROJECT,env=lean_environment(),text=True,encoding='utf-8',stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60)
   self.assertEqual(result.returncode,0,result.stdout)
 def test_foreign_and_missing_source_context_are_rejected(self):
  for scoped in (False,True):
   for change in ('foreign','missing'):
    data=chapter(scoped,'ambient')
    if change=='foreign':data['steps'][-1]['dependencies'][0]['namespace']='main_boundary_wrong'
    else:data['steps'][-1]['dependencies'][0]['step']='chapter_1_line_99'
    with self.assertRaises(AssertionError):_render_chapter_steps(data,'bad',{ASSUMPTION:'witness_exists'},{},ambient_variables={'w','I'},binary_definitions={'same':{'category':'and'}})

if __name__=='__main__':unittest.main()
