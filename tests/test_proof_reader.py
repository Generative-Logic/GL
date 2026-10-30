"""Operation grouping retains formal rows, scope ownership and public outputs."""

import unittest
from unittest.mock import patch
import visu_helpers
import generate_full_proof_graph as generator
from proof_reader import (
    row_node,
    block_node,
    expanded_rows,
    schedule_nodes,
    reader_plan,
    reader_metrics,
)
from proof_reader_html import (
    render_reader_proof,
    complete_caption,
    notation_toggle_html,
    notation_preference_script,
    induction_part_numbers,
    theorem_citation_label,
    disintegration_witnesses,
    _free_witness_arguments,
)
from proof_hierarchy import resolved_dependencies, resolved_citations, theorem_alpha_keys


def disintegration_rows():
    return [
        [
            "(=[a,a])",
            "main",
            "implication",
            "(>[](in[a,N])(=[a,a]))",
            "main",
            "(in[a,N])",
            "main",
        ],
        ["(>[](in[a,N])(=[a,a]))", "main", "task formulation"],
        ["(in[a,N])", "main", "disintegration", "(&(in[a,N])(in[b,N]))", "main"],
        ["(&(in[a,N])(in[b,N]))", "main", "expansion", "(and9[a,b,N])", "main"],
        ["(and9[a,b,N])", "main", "task formulation"],
    ]


class ReaderPlanTests(unittest.TestCase):
    def test_induction_parts_share_the_parent_chapter_and_start_at_one(self):
        self.assertEqual(induction_part_numbers(300), ("300.1", "300.2", "300.3"))
        self.assertEqual(induction_part_numbers(0), ("0.1", "0.2", "0.3"))
        with self.assertRaises(AssertionError):
            induction_part_numbers(-1)

    def test_theorem_citations_name_their_chapter(self):
        self.assertEqual(theorem_citation_label("chapter12.html"), "chapter 12")
        self.assertEqual(theorem_citation_label("chapter401.html"), "chapter 401")
        self.assertEqual(theorem_citation_label(None), "theorem")
        for url in ("chapter12.html#entry3", "../proofs/chapter12.html", "chapter.html"):
            with self.assertRaises(AssertionError):
                theorem_citation_label(url)
        rule = "(>[x](in[x,N])(=[x,x]))"
        rows = [
            ["(=[a,a])", "main", "implication", rule, "main", "(in[a,N])", "main"],
            ["(in[a,N])", "main", "task formulation"],
        ]
        readable = visu_helpers.ReadableForm({}, {}, structured_implications=True)
        with patch.object(generator, "_readable_form_for_render", readable), patch.object(
            generator, "_resolve_theorem_target", lambda expression: "chapter12.html"
        ):
            text = render_reader_proof(rows, "c20s0", generator, [rule])
        common = text.split('data-row-index="0"', 1)[1].split(
            'class="common-citations">', 1
        )[1].split("</span>", 1)[0]
        self.assertIn('href="chapter12.html"', common)
        self.assertIn(">(chapter 12)</a>", common)
        self.assertNotIn("(theorem)", text)

    def test_witness_arguments_respect_nested_lexical_binders(self):
        expression = visu_helpers.parse_mpl(
            "(&(in[v1,N])!(>[v2](in[v2,N])(in[v2,V1])))"
        )
        self.assertEqual(_free_witness_arguments(expression), {"v1", "N", "V1"})
        with self.assertRaises(AssertionError):
            _free_witness_arguments(object())

    def test_disintegration_witnesses_are_used_formal_binders_in_proof_order(self):
        source = "!(>[v7,V3,unused](in[v7,N])!(&(in[v7,V3])(in[V3,S])))"
        rows = [
            ["(in[V3,S])", "main", "disintegration", source, "main"],
            ["(in[v7,V3])", "main", "disintegration", source, "main"],
            [source, "main", "expansion", "(bundle[N,S])", "main"],
        ]
        block = {
            "kind": "disintegration",
            "members": frozenset(range(3)),
            "namespace": "main",
        }
        self.assertEqual(disintegration_witnesses(rows, block), ["v7", "V3"])
        self.assertNotIn("unused", disintegration_witnesses(rows, block))
        rebound = [
            ["(>[v7,V3](in[v7,N])(in[V3,S]))", "main", "disintegration", source, "main"]
        ]
        single = {
            "kind": "disintegration",
            "members": frozenset({0}),
            "namespace": "main",
        }
        self.assertEqual(disintegration_witnesses(rebound, single), [])
        universal = [
            [
                "(in[v7,N])",
                "main",
                "disintegration",
                "(>[v7](in[v7,N])(in[v7,S]))",
                "main",
            ]
        ]
        self.assertEqual(disintegration_witnesses(universal, single), [])
        with self.assertRaises(AssertionError):
            disintegration_witnesses(rows, dict(block, kind="integration"))
        with self.assertRaises(AssertionError):
            disintegration_witnesses(rows, dict(block, namespace="main_boundary_other"))
        with self.assertRaises(AssertionError):
            disintegration_witnesses([["(in[v7,N])", "main", "disintegration"]], single)

    def test_products_label_only_when_products_are_used(self):
        unused = [
            ["(=[a,a])", "main", "task formulation"],
            ["(in[a,N])", "main", "disintegration", "(&(in[a,N])(in[b,N]))", "main"],
            ["(&(in[a,N])(in[b,N]))", "main", "task formulation"],
        ]
        used = [["(=[b,b])", "main", "symmetry of equality", "(in[a,N])", "main"]] + unused[1:]
        readable = visu_helpers.ReadableForm({}, {}, structured_implications=True)
        with patch.object(generator, "_readable_form_for_render", readable):
            hidden = render_reader_proof(unused, "c1s0", generator, [])
            shown = render_reader_proof(used, "c1s0", generator, [])
        self.assertIn('data-block-kind="disintegration"', hidden)
        self.assertEqual(hidden.count('<span class="reader-label">From</span>'), 2)
        self.assertNotIn("Products of disintegration used in proof:", hidden)
        self.assertNotIn('class="block-products"', hidden)
        self.assertEqual(shown.count("Products of disintegration used in proof:"), 2)
        self.assertEqual(shown.count('class="block-products"'), 2)

    def test_supplied_registry_keys_render_like_computed_keys(self):
        source = "!(>[v7,V3](in[v7,N])!(in[V3,S]))"
        rows = [
            ["(&(in[v7,N])(in[V3,S]))", "main", "disintegration", source, "main"],
            [source, "main", "task formulation"],
        ]
        known = ["(>[x](P[x])(Q[x]))"]
        readable = visu_helpers.ReadableForm({}, {}, structured_implications=True)
        with patch.object(generator, "_readable_form_for_render", readable):
            computed = render_reader_proof(rows, "c1s0", generator, known)
            supplied = render_reader_proof(rows, "c1s0", generator, known, theorem_alpha_keys(known))
        self.assertEqual(computed, supplied)

    def test_witness_header_uses_one_comma_separated_row_per_notation(self):
        source = "!(>[v7,V3](in[v7,N])!(in[V3,S]))"
        rows = [
            ["(&(in[v7,N])(in[V3,S]))", "main", "disintegration", source, "main"],
            [source, "main", "task formulation"],
        ]
        readable = visu_helpers.ReadableForm({}, {}, structured_implications=True)
        with patch.object(generator, "_readable_form_for_render", readable):
            text = render_reader_proof(rows, "c359s0", generator, [])
        self.assertIn('data-block-witnesses="v7 V3"', text)
        self.assertEqual(text.count('class="reader-witnesses"'), 2)
        self.assertIn('data-witnesses="v7 V3">v7, V3</span>', text)
        self.assertIn(
            'data-witnesses="v7 V3">v<sub>7</sub>, V<sub>3</sub></span>', text
        )
        self.assertEqual(text.count("Witnesses used in proof:"), 2)

    def test_notation_script_is_main_only_and_owns_the_checkbox(self):
        script = notation_preference_script(True)
        self.assertIn('id="common-notation-navigation"', script)
        self.assertIn("replaceState", script)
        self.assertIn("pageshow", script)
        self.assertIn("popstate", script)
        self.assertEqual(notation_preference_script(False), "")
        self.assertNotIn("onchange=", notation_toggle_html(True))

    def test_notation_switch_keeps_the_first_line_that_begins_below_the_header(self):
        script = notation_preference_script(True)
        lines = script.split("const readingLines = ", 1)[1].split(";", 1)[0]
        for selector in (".reader-section .proof-line", ".reader-section .proof-line .common-line",
                         ".reader-section .subproof-title", ".reader-section .subproof-title > .readable-grey > *",
                         ".reader-section .subproof-title > .readable-grey .block-products > *",
                         "#theorem-list > li"):
            self.assertIn(selector, lines)
        self.assertNotIn("mpl", lines)
        handler = script.split("control.addEventListener('change'", 1)[1].split("window.addEventListener('resize'", 1)[0]
        self.assertIn("if (pinned && pinned.scrollY === window.scrollY", handler)
        below = handler.index("if (bounds.top >= toolbarBottom) {")
        covering = handler.index("covering = {node: node, top: bounds.top};")
        self.assertLess(below, covering)
        self.assertIn("if (reading === null) reading = covering;", handler)
        self.assertIn("if (bounds.height === 0 || bounds.bottom <= toolbarBottom) continue;", handler)
        self.assertIn("if (bounds.top >= window.innerHeight) break;", handler)
        self.assertIn("window.scrollBy(0, reading.node.getBoundingClientRect().top - reading.top)", handler)
        self.assertIn("pinned = reading ? {node: reading.node, scrollY: window.scrollY} : null;", handler)

    def test_notation_control_is_prominent_accessible_and_main_only(self):
        text = notation_toggle_html(True)
        self.assertIn('type="checkbox"', text)
        self.assertIn("<strong>Common notation</strong>", text)
        self.assertIn('aria-label="Common notation only"', text)
        self.assertIn("Click to read common notation only", text)
        self.assertEqual(notation_toggle_html(False), "")

    def test_disintegration_exports_only_public_products(self):
        rows = disintegration_rows()
        edges = resolved_dependencies(rows, [])
        citations = resolved_citations(rows, [])
        self.assertEqual(citations[2], [("local", 3)])
        tree = reader_plan(rows, edges)
        block = next(n for n in tree["children"] if n["kind"] == "disintegration")
        self.assertEqual(block["source"], ("(and9[a,b,N])", "main"))
        self.assertEqual(block["outputs"], [2])
        self.assertEqual(set(expanded_rows(tree)), set(range(5)))
        self.assertEqual(reader_metrics(tree, rows)["main_rows"], 4)
        self.assertEqual(reader_metrics(tree, rows)["proof_steps"], 5)
        reordered = [rows[i] for i in reversed(expanded_rows(tree))]
        self.assertEqual(
            expanded_rows(reader_plan(reordered, resolved_dependencies(reordered, []))),
            list(reversed(range(5))),
        )

    def test_integration_contains_complete_subproof(self):
        ns = "main_boundary_(>[](in[a,N])(in[a,N]))"
        rows = [
            [
                "(in[b,N])",
                "main",
                "implication",
                "(>[](in[a,N])(in[b,N]))",
                "main",
                "(in[a,N])",
                "main",
            ],
            ["(in[a,N])", "main", "validity name", "(in[a,N])", ns],
            ["(in[a,N])", ns, "premise element"],
            ["(>[](in[a,N])(in[b,N]))", "main", "reformulation for integration and"],
        ]
        tree = reader_plan(rows, {(3, 0), (1, 0), (2, 1)})
        block = tree["children"][0]
        self.assertEqual(block["kind"], "integration")
        self.assertTrue(any(n["kind"] == "subproof" for n in block["children"]))
        self.assertEqual(expanded_rows(tree)[-1], 0)
        self.assertEqual(reader_metrics(tree, rows)["main_rows"], 1)

    def test_compiled_integration_expansion_keeps_the_operation_block(self):
        rows = [
            ["(pair[a,b])", "main", "implication", "(>[](p[a])(pair[a,b]))", "main", "(p[a])", "main"],
            ["(>[](p[a])(pair[a,b]))", "main", "expansion", "(carrier[a,b])", "main"],
            ["(carrier[a,b])", "main", "reformulation for integration and"],
            ["(p[a])", "main", "task formulation"],
        ]
        tree = reader_plan(rows, resolved_dependencies(rows, []))
        block = next(node for node in tree["children"] if node["kind"] == "integration")
        self.assertEqual(block["members"], frozenset({0, 1, 2}))
        self.assertEqual(set(expanded_rows(tree)), set(range(4)))
        rows[2][2] = "task formulation"
        ordinary = reader_plan(rows, resolved_dependencies(rows, []))
        self.assertTrue(all(node["kind"] == "row" for node in ordinary["children"]))

    def test_or_siblings_are_consecutive_and_contract_cycles_assert(self):
        a = block_node("ordis", [row_node(3)], namespace="main_boundary_ordis_a")
        b = block_node("orint", [row_node(1)], namespace="main_boundary_orint_b")
        result = schedule_nodes([a, row_node(2), b, row_node(0)], {(3, 0), (1, 0)}, 0)
        kinds = [n["kind"] for n in result]
        self.assertEqual(abs(kinds.index("ordis") - kinds.index("orint")), 1)
        with self.assertRaisesRegex(AssertionError, "cyclic reader"):
            schedule_nodes([a, row_node(2), b], {(3, 2), (2, 1)})
        self.assertEqual(schedule_nodes([], set()), [])
        with self.assertRaises(AssertionError):
            block_node("subproof", [])
        with self.assertRaises(AssertionError):
            block_node("subproof", [row_node(1), row_node(1)])
        with self.assertRaises(AssertionError):
            reader_plan([], set())

    def test_or_header_names_target_not_the_parent_or_receipt(self):
        from unittest.mock import Mock

        ns = "main_boundary_orint_(or1[a,b])_((in[a,N]))"
        rows = [
            ["(or1[a,b])", "main", "or convergence", "(or1[a,b])", ns],
            [
                "(or1[a,b])",
                ns,
                "or branch proven",
                "(or1[a,b])_integration_goal",
                "main",
            ],
            ["(in[a,N])", ns, "task formulation"],
        ]
        readable = Mock()
        readable.formula.side_effect = lambda text: text
        with patch.object(generator, "_readable_form_for_render", readable):
            text = render_reader_proof(rows, "c12s0", generator, [])
        header = text.split('data-closed-result="true"', 1)[1].split(
            '<div class="subproof-body">', 1
        )[0]
        self.assertIn(
            '<span class="reader-label">Goal:</span> <span class="clickable" data-text="(in[a,N])"',
            header,
        )
        self.assertIn("No additional case assumption.", header)

    def test_html_has_every_caption_links_headers_and_counts(self):
        rows = disintegration_rows()
        tree = reader_plan(rows, resolved_dependencies(rows, []))
        rows = [rows[i] for i in reversed(expanded_rows(tree))]
        definitions = {
            "and9": {
                "category": "and",
                "signature": "(and9[u_1,u_2,u_3])",
                "elements": ["(in[u_1,u_3])", "(in[u_2,u_3])"],
            }
        }
        readable = visu_helpers.ReadableForm(
            definitions, {}, structured_implications=True
        )
        with patch.object(generator, "_readable_form_for_render", readable):
            text = render_reader_proof(rows, "c10s0", generator, [])
        self.assertEqual(text.count("data-row-index="), 5)
        self.assertEqual(text.count('class="common-line"'), 5)
        self.assertIn('data-block-kind="disintegration"', text)
        self.assertIn('data-closed-result="true"', text)
        self.assertIn("10.1.1", text)
        self.assertIn('data-main-rows="4"', text)
        self.assertIn('href="#c10s0-block-', text)
        self.assertIn("common-citations", text)
        common_line = text.split('class="common-line"', 1)[1].split(
            "</div></div></div>", 1
        )[0]
        self.assertLess(
            common_line.index("readable-grey"), common_line.index("common-citations")
        )
        self.assertIn('class="common-step-badge">(1/5)</span>', text)
        self.assertEqual(text.count('class="common-step-badge"'), 5)
        self.assertLess(
            common_line.index("common-step-badge"), common_line.index("readable-grey")
        )
        common_links = text.split('class="common-citations">')
        product_index = next(
            i for i, row in enumerate(rows) if row[2] == "disintegration"
        )
        consumer_index = next(
            i for i, row in enumerate(rows) if row[2] == "implication"
        )
        consumer_html = text.split(f'data-row-index="{consumer_index}"', 1)[1]
        consumer_common = consumer_html.split('class="common-citations">', 1)[1].split(
            "</span>", 1
        )[0]
        self.assertIn(f'href="#c10s0-entry{product_index}"', consumer_common)
        self.assertNotIn("-block-", consumer_common)
        self.assertTrue(
            all(
                "-block-" not in part.split("</span>", 1)[0]
                for part in common_links[1:]
            )
        )
        self.assertIn(
            "rf-label rf-reads-label",
            complete_caption(["(in[a,N])", "main", "task formulation"], readable),
        )
        self.assertIn('<span class="reader-block-kind">Disintegration</span>', text)
        self.assertEqual(
            text.count(
                '<span class="reader-label">Products of disintegration used in proof:</span>'
            ),
            2,
        )
        self.assertIn('<span class="reader-label">From</span>', text)
        self.assertIn("Products of disintegration used in proof:", text)
        self.assertNotIn("Products used outside:", text)
        self.assertIn(
            "READS:",
            complete_caption(["(in[a,N])", "main", "task formulation"], readable),
        )


if __name__ == "__main__":
    unittest.main()
