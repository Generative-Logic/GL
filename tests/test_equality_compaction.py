# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Direct substitution fusion, preservation barriers and verifier compatibility."""
import unittest
from proof_processing import fuse_equality_chains, compact_main_equalities
from verifier import ProofLine, check_equality1

def chain(namespace="main"):
    return [["(P[b,d])",namespace,"equality1","(P[b,c])",namespace,"(=[c,d])",namespace],
            ["(P[b,c])",namespace,"equality1","(P[a,c])",namespace,"(=[a,b])",namespace],
            ["(P[a,c])",namespace,"task formulation"],
            ["(=[a,b])",namespace,"task formulation"],
            ["(=[c,d])",namespace,"task formulation"]]

class EqualityFusionTests(unittest.TestCase):
    def test_direct_pairs_fuse_and_unchanged_verifier_accepts(self):
        rows=chain();result=fuse_equality_chains(rows)
        self.assertEqual(len(result),len(rows)-1)
        self.assertEqual(result[0],rows[0][:3]+["(P[a,c])","main","(=[a,b])","main","(=[c,d])","main"])
        fields=result[0]
        self.assertTrue(check_equality1(ProofLine(*fields[:3],fields[3:],"\t".join(fields),1),[],None))
        self.assertEqual(fuse_equality_chains(result),result)
        self.assertEqual(len(rows),5)

    def test_transitive_pairs_do_not_count_as_direct_evidence(self):
        rows=chain();rows[0]=["(P[d,c])","main","equality1","(P[b,c])","main","(=[b,d])","main"]
        rows[-1]=["(=[b,d])","main","task formulation"]
        self.assertEqual(fuse_equality_chains(rows),rows)

    def test_used_intermediate_and_first_scope_row_are_preserved(self):
        rows=chain();rows.append(["(Q[b,c])","main","implication","(P[b,c])","main"])
        self.assertEqual(fuse_equality_chains(rows),rows)
        rows=chain();rows[0],rows[1]=rows[1],rows[0]
        self.assertEqual(fuse_equality_chains(rows),rows)

    def test_cross_scope_chain_is_not_fused(self):
        rows=chain();rows[0][1]="main_boundary_child"
        self.assertEqual(fuse_equality_chains(rows),rows)

    def test_wrong_equality_direction_does_not_fuse(self):
        rows=chain();rows[0][-2]="(=[d,c])";rows[-1][0]="(=[d,c])"
        self.assertEqual(fuse_equality_chains(rows),rows)

    def test_negative_atom_and_ancestor_equality_are_supported(self):
        rows=chain("main_boundary_child")
        for row in rows:
            if row[0].startswith("(P["):row[0]="!"+row[0]
            if len(row)>3 and row[3].startswith("(P["):row[3]="!"+row[3]
        rows[1][-1]="main";rows[3][1]="main"
        result=fuse_equality_chains(rows)
        self.assertEqual(len(result),4)
        fields=result[0]
        self.assertTrue(check_equality1(ProofLine(*fields[:3],fields[3:],"",1),[],None))

    def test_duplicate_scoped_identity_asserts(self):
        rows=chain();rows.append(list(rows[-1]))
        with self.assertRaisesRegex(AssertionError,"ambiguous"):fuse_equality_chains(rows)

    def test_wrapper_preserves_incubator_and_bundles(self):
        main="(>[N](AnchorPeano[N])(in[x,N]))";other=main.replace("AnchorPeano","AnchorIncubator")
        stacks={"a":chain(),"b":chain()};theorems=[[main],[other]]
        result=compact_main_equalities(theorems,stacks,[["a"],["b"]])
        self.assertEqual(len(result['a']),4);self.assertEqual(result['b'],stacks['b'])
        with self.assertRaises(AssertionError):compact_main_equalities(theorems,stacks,[["a"]])

if __name__ == "__main__":unittest.main()
