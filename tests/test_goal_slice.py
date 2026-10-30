import unittest
from proof_goal_slice import slice_unused_main_rows,apply_main_rows

class GoalSliceTests(unittest.TestCase):
    def test_unused_safe_chain_removed_to_fixed_point(self):
        rows=[['(G[a])','main','task formulation'],
              ['(P[c])','main','equality1','(P[b])','main','(=[b,c])','main'],
              ['(P[b])','main','equality1','(P[a])','main','(=[a,b])','main'],
              ['(P[a])','main','task formulation'],['(=[a,b])','main','task formulation'],['(=[b,c])','main','task formulation']]
        result=slice_unused_main_rows(rows)
        self.assertEqual(len(result),4);self.assertEqual(slice_unused_main_rows(result),result)
        self.assertEqual(result[0],rows[0])

    def test_subproof_uses_and_all_nonordinary_roots_preserved(self):
        rows=[['(G[a])','main','task formulation'],['(P[b])','main','equality1','(P[a])','main','(=[a,b])','main'],
              ['(P[b])','main_boundary_c','zoom in','(P[b])','main'],
              ['(Q[b])','main_boundary_c','equality1','(Q[a])','main_boundary_c','(=[a,b])','main_boundary_c'],
              ['(R[a])','main','expansion','(D[a])','main'],['(D[a])','main','task formulation'],
              ['(P[a])','main','task formulation'],['(=[a,b])','main','task formulation']]
        self.assertEqual(slice_unused_main_rows(rows),rows)

    def test_alpha_scoped_use_prevents_deletion_and_bad_shape_asserts(self):
        rows=[['(G[a])','main','implication','(>[y](P[y])(Q[y]))','main'],
              ['(>[x](P[x])(Q[x]))','main','implication']]
        self.assertEqual(slice_unused_main_rows(rows),rows)
        with self.assertRaises(AssertionError):slice_unused_main_rows([])
        with self.assertRaises(AssertionError):slice_unused_main_rows([['bad']])

    def test_main_wrapper_preserves_incubator(self):
        theorem='(>[N](AnchorPeano[N])(in[x,N]))'
        incubator=theorem.replace('AnchorPeano','AnchorIncubator')
        source=[['(G[a])','main','task formulation'],['(P[b])','main','equality1','(P[a])','main','(=[a,b])','main']]
        result=apply_main_rows([[theorem],[incubator]],{'a':source,'b':source},[['a'],['b']])
        self.assertLess(len(result['a']),len(source));self.assertEqual(result['b'],source)
        with self.assertRaises(AssertionError):apply_main_rows([[theorem]],{'a':source},[])

if __name__=='__main__':unittest.main()
