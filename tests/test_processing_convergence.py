"""The complete accepted transformation family reaches a checked fixed point."""
import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from proof_convergence import process_to_fixed_point
from process_proof_graphs import _build_fname_list
from tests.test_rewrite_placement import rows

P='(>[N](AnchorPeano[N])(R[b,d]))'
class ProcessingConvergenceTests(unittest.TestCase):
 def test_real_rewrite_is_shorter_idempotent_and_keeps_goal(self):
  theorems=[[P,'single chapter proof','-1']];name=_build_fname_list(theorems)[0][0];original=rows()
  with tempfile.TemporaryDirectory() as directory:
   binary=Path(directory);(binary/'GL_binary_Peano.json').write_text('{}',encoding='utf-8')
   ts,result=process_to_fixed_point(theorems,{name:original},_build_fname_list,set(),binary)
   self.assertEqual(len(result[name]),len(original)-1);self.assertEqual(result[name][0][:2],original[0][:2])
   self.assertEqual(process_to_fixed_point(ts,result,_build_fname_list,set(),binary),(ts,result))
 def test_unsupported_and_empty_book_remain_exact(self):
  theorems=[[P.replace('AnchorPeano','AnchorIncubator'),'single chapter proof','-1']];name=_build_fname_list(theorems)[0][0];stacks={name:rows()}
  self.assertEqual(process_to_fixed_point(theorems,stacks,_build_fname_list,set(),Path('missing-registry')),(theorems,stacks))
  self.assertEqual(process_to_fixed_point([],{},_build_fname_list,set(),Path('missing-registry')),([],{}))
 def test_missing_bundle_and_missing_registry_assert(self):
  theorems=[[P,'single chapter proof','-1']];name=_build_fname_list(theorems)[0][0]
  with self.assertRaisesRegex(AssertionError,'bundle coverage'):process_to_fixed_point(theorems,{},_build_fname_list,set(),Path('missing-registry'))
  with self.assertRaisesRegex(AssertionError,'missing exact'):process_to_fixed_point(theorems,{name:rows()},_build_fname_list,set(),Path('missing-registry'))
 def test_cycle_is_rejected_without_iteration_fallback(self):
  theorems=[[P,'single chapter proof','-1']];name=_build_fname_list(theorems)[0][0]
  def toggle(ts,ss,groups):
   row=list(ss[name][0]);row[2]='task formulation' if row[2]=='premise' else 'premise';return {name:[row]}
  with tempfile.TemporaryDirectory() as directory:
   binary=Path(directory);(binary/'GL_binary_Peano.json').write_text('{}',encoding='utf-8')
   with patch('proof_convergence.compact_main_equalities',side_effect=toggle):
    with self.assertRaisesRegex(AssertionError,'processing cycle'):process_to_fixed_point(theorems,{name:[[P,'main','premise']]},_build_fname_list,set(),binary)
 def test_row_growth_is_rejected(self):
  theorems=[[P,'single chapter proof','-1']];name=_build_fname_list(theorems)[0][0]
  def grow(ts,ss,groups):return {name:[*ss[name],['(U[a])','main','premise']]}
  with tempfile.TemporaryDirectory() as directory:
   binary=Path(directory);(binary/'GL_binary_Peano.json').write_text('{}',encoding='utf-8')
   with patch('proof_convergence.compact_main_equalities',side_effect=grow):
    with self.assertRaisesRegex(AssertionError,'row growth'):process_to_fixed_point(theorems,{name:[[P,'main','premise']]},_build_fname_list,set(),binary)

if __name__=='__main__':unittest.main()
