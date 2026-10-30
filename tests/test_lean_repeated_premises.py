"""Identical premise occurrences are not distinct logical proof orderings."""
from copy import deepcopy
import subprocess,tempfile,unittest
from pathlib import Path
from proof_export.lean import _implication_dependency_orderings,_ordered_implication_dependencies,_render_chapter_steps
from tests.lean_toolchain import LEAN_PROJECT,lean_environment,lean_executable

RULE='(>[x](in[x,I])(>[y](in[y,I])(=[x,y])))'
FACT='(in[w,I])'
def dependency(expression,number):
 return {'step':f'chapter_1_line_{number}','source_kind':'local_step','namespace':'main','expression':{'mpl':expression}}
def step(rule=RULE,conclusion='(=[w,w])'):
 return {'id':'chapter_1_line_1','source_line':1,'source_tag':'implication','action':'implication_apply','namespace':'main','conclusion':{'mpl':conclusion},'dependencies':[dependency(rule,3),dependency(FACT,2),dependency(FACT,2)]}

class RepeatedPremiseTests(unittest.TestCase):
 def test_identical_references_keep_both_required_occurrences(self):
  value=step();ordered=_ordered_implication_dependencies(value)
  self.assertEqual(ordered,value['dependencies'][1:]);self.assertEqual(len(ordered),2)
 def test_different_proof_sources_remain_ambiguous(self):
  value=step();value['dependencies'][2]['step']='chapter_1_line_4'
  with self.assertRaisesRegex(AssertionError,'2 logical orderings'):_ordered_implication_dependencies(value)
 def test_different_fact_assignments_remain_ambiguous(self):
  value=step('(>[x](in[x,I])(>[y](in[y,I])(=[a,a])))','(=[a,a])');value['dependencies'][2]=dependency('(in[z,I])',4)
  with self.assertRaisesRegex(AssertionError,'2 logical orderings'):_ordered_implication_dependencies(value)
 def test_orderings_list_the_single_ordering_and_stop_at_the_second(self):
  value=step();self.assertEqual(_implication_dependency_orderings(value),[value['dependencies'][1:]])
  value['dependencies'][2]['step']='chapter_1_line_4'
  self.assertEqual(len(_implication_dependency_orderings(value)),2)
 def test_conclusion_that_does_not_unify_has_no_ordering(self):
  value=step(conclusion='(in[w,I])')
  self.assertEqual(_implication_dependency_orderings(value),[])
  with self.assertRaisesRegex(AssertionError,'0 logical orderings'):_ordered_implication_dependencies(value)
 def test_missing_occurrence_and_unmatched_fact_are_rejected(self):
  for change in ('missing','wrong'):
   value=step()
   if change=='missing':value['dependencies'].pop()
   else:value['dependencies'][2]=dependency('!(in[w,I])',4)
   with self.assertRaisesRegex(AssertionError,'0 logical orderings'):_ordered_implication_dependencies(value)
 def test_kernel_reuses_the_same_fact_for_both_premises(self):
  data={'variable_types':{'w':'element','I':'set','x':'element','y':'element'},'steps':[
   {'id':'chapter_1_line_3','source_line':3,'source_tag':'task formulation','action':'assume','namespace':'main','conclusion':{'mpl':RULE},'dependencies':[]},
   {'id':'chapter_1_line_2','source_line':2,'source_tag':'task formulation','action':'assume','namespace':'main','conclusion':{'mpl':FACT},'dependencies':[]},step()]}
  lines,manifest=_render_chapter_steps(data,'repeated-premises',{RULE:'h',FACT:'hp'},{},ambient_variables={'w','I'})
  self.assertEqual(len(manifest),3)
  support=(Path(__file__).resolve().parents[1]/'lean_export/GLExport/ProofSupport.lean').read_text(encoding='utf-8')
  source=support+'\nopen GLExport\nexample {α : Type} (I : GLSet α) (w : α) (h : ∀ x, I x → ∀ y, I y → x = y) (hp : I w) : w = w := by\n'+'\n'.join(lines)+'\n  exact row_1\n'
  with tempfile.TemporaryDirectory(prefix='repeated_premises_') as directory:
   path=Path(directory)/'Repeated.lean';path.write_text(source,encoding='utf-8')
   result=subprocess.run([lean_executable(),str(path)],cwd=LEAN_PROJECT,env=lean_environment(),text=True,encoding='utf-8',stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60)
   self.assertEqual(result.returncode,0,result.stdout)

if __name__=='__main__':unittest.main()
