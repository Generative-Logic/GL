# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt)
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Scoped implication contradiction elimination and scope isolation."""
import unittest
from tests.test_harness import register, make_proof_line, make_state_with_binaries
from verifier import check_vacuous_truth

class ScopedVacuityTests(unittest.TestCase):

    def make(self, payload='(implication0[f,I])', parent='main'):
        state = make_state_with_binaries(('Peano',))
        scope = parent + '_boundary_' + payload
        expanded = '(>[x,y](in2[x,y,f])(in[x,I]))_integration_goal'
        expansion = make_proof_line(expanded, parent, 'expansion for integration', '(implication0[f,I])_integration_goal', parent)
        premise = make_proof_line('(in2[x,y,f])', scope, 'premise element', expanded, parent)
        opposite = make_proof_line('!(in2[x,y,f])', parent, 'task formulation')
        result = make_proof_line('(in[x,I])', scope, 'vacuous truth', '(in2[x,y,f])', scope, '!(in2[x,y,f])', parent)
        return (state, [expansion, premise, opposite, result], result)

    def check(self, state, chapter, row):
        return check_vacuous_truth(row, chapter, state)

    def test_valid_exact_scope_and_ancestor(self):
        state, chapter, row = self.make()
        self.assertTrue(self.check(state, chapter, row))

    def test_valid_goal_owned_scope(self):
        state, chapter, row = self.make('(fold[N,s,add,mul,P,z,z,a])_subproof_(implication0[f,I])')
        self.assertTrue(self.check(state, chapter, row))

    def test_valid_nested_parent(self):
        state, chapter, row = self.make(parent='main_boundary_(implication9[f])')
        self.assertTrue(self.check(state, chapter, row))

    def test_reject_sibling_fact(self):
        state, chapter, row = self.make()
        row.rest[3] = 'main_boundary_sibling'
        self.assertFalse(self.check(state, chapter, row))

    def test_reject_descendant_fact(self):
        state, chapter, row = self.make()
        row.rest[3] = row.namespace + '_boundary_extra'
        self.assertFalse(self.check(state, chapter, row))

    def test_reject_only_ancestor_facts(self):
        state, chapter, row = self.make()
        row.rest[1] = 'main'
        self.assertFalse(self.check(state, chapter, row))

    def test_reject_wrong_goal(self):
        state, chapter, row = self.make()
        row.expression = '(in[y,I])'
        self.assertFalse(self.check(state, chapter, row))

    def test_reject_nonopposing_pair(self):
        state, chapter, row = self.make()
        row.rest[2] = '!(in2[y,x,f])'
        self.assertFalse(self.check(state, chapter, row))

    def test_reject_missing_dependency(self):
        state, chapter, row = self.make()
        self.assertFalse(self.check(state, [chapter[0], chapter[1], row], row))

    def test_reject_wrong_expansion(self):
        state, chapter, row = self.make()
        chapter[0].expression = '(>[x,y](in2[x,y,f])(in[y,I]))_integration_goal'
        self.assertFalse(self.check(state, chapter, row))

    def test_reject_unregistered_compact(self):
        state, chapter, row = self.make('(unknown[f,I])')
        self.assertFalse(self.check(state, chapter, row))

    def test_reject_missing_own_premise_trace(self):
        state, chapter, row = self.make()
        chapter[1].tag = 'task formulation'
        self.assertFalse(self.check(state, chapter, row))

    def test_legacy_main_unchanged(self):
        state = make_state_with_binaries(('Peano',))
        row = make_proof_line('(in[x,I])', 'main', 'vacuous truth', '(in2[x,y,f])', 'main', '!(in2[x,y,f])', 'main', '(in[x,I])', 'main')
        self.assertTrue(self.check(state, [row], row))
        row.rest[3] = 'main_boundary_other'
        self.assertFalse(self.check(state, [row], row))
for _name in unittest.defaultTestLoader.getTestCaseNames(ScopedVacuityTests):

    def _registered_case(name=_name):
        case = ScopedVacuityTests(name)
        case.setUp()
        try:
            getattr(case, name)()
        finally:
            case.tearDown()
    _registered_case.__name__ = 'test_scoped_vacuity_' + _name.removeprefix('test_')
    register(_registered_case)
if __name__ == '__main__':
    unittest.main()
