import unittest
from proof_rewrite_placement import place_conclusion_rewrite,apply_main_rows
from verifier import ProofLine,check_implication,check_equality1

RULE='(>[x](P[x])(>[y](Q[y])(R[x,y])))'

def rows():
    return [['(R[b,d])','main','implication',RULE,'main','(P[b])','main','(Q[d])','main'],
            ['(P[b])','main','equality1','(P[a])','main','(=[a,b])','main'],
            ['(Q[d])','main','equality1','(Q[c])','main','(=[c,d])','main'],
            ['(P[a])','main','task formulation'],['(Q[c])','main','task formulation'],
            ['(=[a,b])','main','task formulation'],['(=[c,d])','main','task formulation'],[RULE,'main','task formulation']]

class RewritePlacementTests(unittest.TestCase):
    def test_independent_candidate_predicates_reject_missing_or_wrong_support(self):
        from proof_rewrite_placement import implication_candidate, equality_candidate
        original = rows()
        self.assertTrue(implication_candidate(original[0]))
        self.assertTrue(equality_candidate(original[1]))
        for expression in ('(R[b,c])', '(R[b,d,e])', '!(R[b,d])'):
            candidate = list(original[0])
            candidate[0] = expression
            self.assertFalse(implication_candidate(candidate))
        candidate = list(original[1])
        candidate[5] = '(=[b,a])'
        self.assertFalse(equality_candidate(candidate))
        candidate[5] = '!(=[a,b])'
        self.assertFalse(equality_candidate(candidate))
        candidate = list(original[1])
        candidate[0] = '!(P[b])'
        self.assertFalse(equality_candidate(candidate))
        candidate = list(original[1])
        candidate[0] = '(Q[b])'
        self.assertFalse(equality_candidate(candidate))
        for predicate, row in ((implication_candidate, original[0]), (equality_candidate, original[1])):
            self.assertFalse(predicate(row[:3]))
            changed = list(row)
            changed[4] = 'main_boundary_other'
            self.assertFalse(predicate(changed))
            with self.assertRaises(AssertionError):
                predicate(row[:-1])

    def test_two_premise_rewrites_become_one_conclusion_rewrite(self):
        source=rows();result=place_conclusion_rewrite(source)
        self.assertEqual(len(result),len(source)-1)
        self.assertEqual(result[0][:3],['(R[b,d])','main','equality1'])
        self.assertEqual(result[1][:3],['(R[a,c])','main','implication'])
        for row in result[:2]:
            checker=check_implication if row[2]=='implication' else check_equality1
            self.assertTrue(checker(ProofLine(*row[:3],row[3:],'',0),[],None))
        self.assertEqual(place_conclusion_rewrite(result),result)

    def test_earlier_conclusion_reuse_avoids_added_step(self):
        source=rows();source.append(['(R[a,c])','main','task formulation'])
        result=place_conclusion_rewrite(source)
        self.assertEqual(len(result),len(source)-2)
        self.assertEqual(sum(r[0]=='(R[a,c])' for r in result),1)

    def test_ambiguous_inverse_shared_premise_and_cross_scope_are_ineligible(self):
        source=rows()
        source=[[cell.replace('d','b') for cell in row] for row in source]
        self.assertEqual(place_conclusion_rewrite(source),source)
        source=rows();source.append(['(T[a])','main','implication','(P[b])','main'])
        self.assertEqual(place_conclusion_rewrite(source),source)
        source=rows();source[0][4]='main_boundary_c'
        self.assertEqual(place_conclusion_rewrite(source),source)

    def test_invalid_candidate_and_protected_scope_boundary_are_preserved(self):
        source=rows();source[0][0]='(R[b,d,e])'
        self.assertEqual(place_conclusion_rewrite(source),source)
        source=rows();source[1][1]='main_boundary_c';source[0][6]='main_boundary_c'
        self.assertEqual(place_conclusion_rewrite(source),source)
        with self.assertRaises(AssertionError):place_conclusion_rewrite([])
        with self.assertRaises(AssertionError):place_conclusion_rewrite([['bad']])

    def test_main_wrapper_preserves_incubator(self):
        theorem='(>[N](AnchorPeano[N])(in[x,N]))'
        incubator=theorem.replace('AnchorPeano','AnchorIncubator')
        source=rows()
        result=apply_main_rows([[theorem],[incubator]],{'a':source,'b':source},[['a'],['b']])
        self.assertLess(len(result['a']),len(source));self.assertEqual(result['b'],source)
        with self.assertRaises(AssertionError):apply_main_rows([[theorem]],{'a':source},[])

    def test_new_result_reuses_a_removed_slot_and_orders_late_premises(self):
        rule='(>[x](P[x])(>[y](Q[y])(>[](S[e])(R[x,y]))))'
        source=rows();source[0][3]=rule;source[-1][0]=rule;source[0]+=['(S[e])','main']
        source.insert(1,['(S[e])','main','task formulation'])
        source.insert(2,['(T[e])','main_boundary_child','task formulation'])
        source.insert(4,['(U[e])','main_boundary_child','task formulation'])
        result=place_conclusion_rewrite(source)
        self.assertEqual(len(result),len(source)-1)
        remaining=iter(row[1] for row in source)
        for namespace in (row[1] for row in result):self.assertTrue(any(old==namespace for old in remaining))
        self.assertEqual([r for r in result if r[1]!='main'],[r for r in source if r[1]!='main'])
        positions={(r[0],r[1]):i for i,r in enumerate(result)}
        for i,row in enumerate(result):
            for j in range(3,len(row),2):self.assertGreater(positions[row[j],row[j+1]],i)
        self.assertEqual(place_conclusion_rewrite(result),result)

    def test_fixed_conclusion_needs_no_premise_substitution(self):
        source=rows();rule='(>[x](P[x])(>[y](Q[y])(R[b])))'
        source[0][0]='(R[b])';source[0][3]=rule;source[-1][0]=rule
        result=place_conclusion_rewrite(source)
        self.assertEqual(len(result),len(source)-2)
        self.assertEqual(result[0][:3],['(R[b])','main','implication'])
        self.assertEqual(result[0][5:],['(P[a])','main','(Q[c])','main'])
        self.assertTrue(check_implication(ProofLine(*result[0][:3],result[0][3:],'',0),[],None))
        self.assertEqual(place_conclusion_rewrite(result),result)

if __name__=='__main__':unittest.main()
