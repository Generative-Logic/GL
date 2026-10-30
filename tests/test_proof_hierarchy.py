"""Nested placement retains all scope boundaries and checks every dependency."""
from collections import Counter
from html.parser import HTMLParser
import re
import unittest
from unittest.mock import patch
from proof_hierarchy import (is_reader_theorem, resolved_dependencies, nested_order,
                             hierarchical_scope_items, order_main_hierarchies)
import generate_full_proof_graph as html

CHILD='main_boundary_subproof_(Q[x])'
NESTED=CHILD+'_boundary_subproof_(B[x])'
THEOREM='(>[N,x](AnchorPeano[N])(G[x]))'

def fixture():
    return [['(G[x])','main','implication','(Q[x])',CHILD,'(P[x])','main'],
            ['(Q[x])',CHILD,'implication','(B[x])',NESTED,'(A[x])','main'],
            ['(B[x])',NESTED,'implication','(X[x])',CHILD],
            ['(P[x])','main','task formulation'],
            ['(X[x])',CHILD,'task formulation'],
            ['(A[x])','main','task formulation']]

class Elements(HTMLParser):
    def __init__(self,text):
        super().__init__();self.ids=[];self.links=[];self.feed(text)
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'id' in a:self.ids.append(a['id'])
        if tag=='a' and 'href' in a:self.links.append(a['href'])

class HierarchyTests(unittest.TestCase):
    def test_supported_anchor_selection_and_unknown_book_identity(self):
        self.assertTrue(is_reader_theorem(THEOREM))
        other=THEOREM.replace('AnchorPeano','AnchorIncubator')
        self.assertFalse(is_reader_theorem(other));self.assertFalse(is_reader_theorem('(G[x])'))
        rows=fixture();ts=[[THEOREM,'single chapter proof','-1'],[other,'single chapter proof','-1']]
        result=order_main_hierarchies(ts,{'a':rows,'b':rows},[['a'],['b']],set())
        self.assertIs(result['b'],rows)
        self.assertEqual(Counter(map(tuple,result['a'])),Counter(map(tuple,rows)))
        with self.assertRaisesRegex(AssertionError,'missing processed'):
            order_main_hierarchies(ts,{},[['a'],['b']],set())

    def test_recursive_order_is_complete_goal_last_and_idempotent(self):
        rows=fixture();edges=resolved_dependencies(rows,set())
        self.assertEqual(edges,{(1,0),(3,0),(2,1),(5,1),(4,2)})
        result=nested_order(rows,edges)
        self.assertEqual(result[0],rows[0]);self.assertEqual(Counter(map(tuple,result)),Counter(map(tuple,rows)))
        self.assertEqual(nested_order(result,resolved_dependencies(result,set())),result)
        positions={tuple(r):i for i,r in enumerate(reversed(result))}
        self.assertTrue(all(positions[tuple(rows[a])]<positions[tuple(rows[b])] for a,b in edges))
        items=hierarchical_scope_items(result)
        self.assertEqual(sum(kind=='row' for seq in items.values() for kind,_ in seq),len(rows))
        self.assertIn(('scope',NESTED),items[CHILD])
        self.assertIn(('scope',CHILD),items['main'])

    def test_parent_rows_must_move_when_insertion_alone_cannot_work(self):
        rows=[['(G[x])','main','implication','(Q[x])',CHILD],
              ['(A[x])','main','task formulation'],
              ['(Q[x])',CHILD,'implication','(A[x])','main']]
        result=nested_order(rows,resolved_dependencies(rows,set()))
        self.assertEqual([r[0] for r in reversed(result)],['(A[x])','(Q[x])','(G[x])'])

    def test_contraction_cycle_is_rejected_without_flattening(self):
        rows=[['(G[x])','main','implication','(Q[x])',CHILD],
              ['(Q[x])',CHILD,'implication','(A[x])','main'],
              ['(A[x])','main','implication','(P[x])',CHILD],
              ['(P[x])',CHILD,'task formulation']]
        edges=resolved_dependencies(rows,set())
        with self.assertRaisesRegex(AssertionError,'cyclic nested'):
            nested_order(rows,edges)

    def test_unresolved_cycles_invalid_shapes_and_goal_consumers_assert(self):
        with self.assertRaisesRegex(AssertionError,'unresolved citation'):
            resolved_dependencies([['(G[x])','main','implication','(A[x])','main']],set())
        with self.assertRaises(AssertionError):resolved_dependencies([],set())
        with self.assertRaises(AssertionError):resolved_dependencies([['(G[x])','main','implication','bad']],set())
        rows=[['(G[x])','main','task formulation'],['(A[x])','main','implication','(G[x])','main']]
        with self.assertRaisesRegex(AssertionError,'cyclic nested'):nested_order(rows,{(0,1)})
        with self.assertRaises(AssertionError):nested_order([['(G[x])','other','task formulation']],set())

    def test_exact_alias_template_compilation_and_or_companion_resolution(self):
        theorem='(>[x](P[x])(Q[x]))';alias='(>[y](P[y])(Q[y]))'
        rows=[['(Q[a])','main','implication',alias,'main'],[theorem,'main','theorem']]
        self.assertEqual(resolved_dependencies(rows,{theorem}),{(1,0)})
        self.assertEqual(resolved_dependencies([['(Q[a])','main','compilation',alias,'main']],{theorem}),set())
        self.assertEqual(resolved_dependencies([['(Q[a])','main','premise element','(P[a])_integration_goal',CHILD]],set()),set())
        self.assertEqual(resolved_dependencies([['(or0[a])','main','or theorem',theorem,'main','(U[a])','main']],{theorem}),set())
        exact=[['(Q[a])','main','zoom in',alias,'main'],[theorem,'main','theorem']]
        with self.assertRaisesRegex(AssertionError,'unresolved citation'):resolved_dependencies(exact,set())
        with self.assertRaisesRegex(AssertionError,'duplicate alpha-equivalent'):
            resolved_dependencies([[theorem,'main','theorem'],[alias,'main','theorem']],set())

    def test_noncontiguous_scope_cannot_be_rendered_as_one_card(self):
        rows=[['(G[x])','main','task formulation'],['(Q[x])',CHILD,'task formulation'],
              ['(A[x])','main','task formulation'],['(B[x])',CHILD,'task formulation']]
        with self.assertRaisesRegex(AssertionError,'noncontiguous'):
            hierarchical_scope_items(rows)
        with self.assertRaises(AssertionError):hierarchical_scope_items([])

    @patch('proof_reader_html.complete_caption', return_value='')
    @patch.object(html, '_row_caption_html', return_value='')
    def test_inline_html_has_nested_cards_correct_anchors_and_continuous_numbers(self, caption, common_caption):
        original=fixture()
        for row in original:
            for i,cell in enumerate(row):
                for letter in 'GQBPXA':cell=cell.replace(f'({letter}[x])',f'(in[x,{letter}])')
                row[i]=cell
        rows=nested_order(original,resolved_dependencies(original,set()))
        text=html.render_stack_with_subproofs(rows,prefix='c1s0',hierarchical=True)
        parsed=Elements(text)
        self.assertEqual(len(parsed.ids),len(set(parsed.ids)))
        self.assertIn('c1s0-block-1-1-1-1',parsed.ids)
        self.assertLess(parsed.ids.index('c1s0-block-1-1-1'),parsed.ids.index('c1s0-entry0'))
        child_goal=next(i for i,r in enumerate(rows) if r[0]=='(in[x,Q])' and r[1].count('_boundary_')==1)
        self.assertLess(parsed.ids.index('c1s0-block-1-1-1-1'),parsed.ids.index(f'c1s0-entry{child_goal}'))
        self.assertEqual(re.findall(r'class="step-badge">\((\d+)/6\)',text),[str(i) for i in range(1,7)])
        self.assertTrue(all(link[1:] in parsed.ids for link in parsed.links if link.startswith('#')))
        self.assertNotIn("proof-section-title'>Subproofs",text)
        self.assertIn('Expand all',text);self.assertEqual(text.count('subproof-card collapsed'),2)
        legacy=html.render_stack_with_subproofs(original)
        self.assertIn("proof-section-title'>Subproofs",legacy)

    def test_scope_fragment_direct_call_and_invalid_child_reference(self):
        row=['(in[x,N])','main','task formulation']
        result=html._render_scope_contents([row],[],'main','m',1,
            {'main':[('row',0)]},{},[0],1,{}, {})
        self.assertIn('m-entry0',Elements(result).ids)
        with self.assertRaises(AssertionError):
            html._render_scope_contents([row],[],'main','m',1,
                {'main':[('scope','main_boundary_missing')]},{},[0],1,{}, {})

    def test_implicit_intermediate_scope_is_an_intact_card(self):
        rows=[['(G[x])','main','implication','(B[x])',NESTED],
              ['(B[x])',NESTED,'task formulation']]
        result=nested_order(rows,resolved_dependencies(rows,set()))
        items=hierarchical_scope_items(result)
        self.assertEqual(items[CHILD],[('scope',NESTED)])
        self.assertEqual(nested_order(result,resolved_dependencies(result,set())),result)

    def test_fragment_formatter_keeps_indices_and_highlight_selection(self):
        text=html.format_stack_entries([['(in[x,N])','main','task formulation']],prefix='p',
            reverse_entries=False,entry_indices=[7],highlight_position=-1,global_counter=[3],global_total=9)
        self.assertIn('p-entry7',Elements(text).ids);self.assertIn('(4/9)',text)
        with self.assertRaises(AssertionError):
            html.format_stack_entries([['(P[x])','main','task formulation']],entry_indices=[])

if __name__=='__main__':unittest.main()

