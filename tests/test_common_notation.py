"""Readable implication layout preserves premises, head and plain-text output."""

import unittest
import visu_helpers as vh


class CommonNotationTests(unittest.TestCase):
    def test_premise_columns_and_head_are_explicit(self):
        expression = "(>[x](in[x,N])(>[](=[x,a])(>[](=[x,b])(=[a,b]))))"
        renderer = vh.ReadableForm({}, {}, structured_implications=True)
        text = renderer.formula(expression)
        self.assertIn("class='rf-implication'", text)
        self.assertIn("class='rf-arrow'", text)
        self.assertIn("class='rf-conclusion'", text)
        self.assertEqual(
            renderer.formula(expression, False),
            vh.ReadableForm({}, {}).formula(expression, False),
        )

    def test_top_level_rule_has_each_premise(self):
        expression = "(>[](in[a,N])(>[](in[b,N])(=[a,b])))"
        renderer = vh.ReadableForm({}, {}, structured_implications=True)
        text = renderer.theorem(expression)
        self.assertEqual(text.count("class='rf-premise'"), 2)
        self.assertIn("a = b", text)
        self.assertNotIn("follows", text)

    def test_multiline_parentheses_stretch_and_preserve_plain_text(self):
        expression = "(&(>[](in[x,N])(>[](in[y,N])(=[x,y])))(in[x,N]))"
        renderer = vh.ReadableForm({}, {}, structured_implications=True)
        text = renderer.formula(expression)
        self.assertIn('class="rf-parentheses"', text)
        self.assertEqual(text.count('class="rf-fence"'), 6)
        self.assertIn('preserveAspectRatio="none"', text)
        self.assertEqual(
            renderer.formula(expression, False),
            vh.ReadableForm({}, {}).formula(expression, False),
        )
        self.assertEqual(vh._HTML.parentheses("x = y"), "(x = y)")
        self.assertEqual(vh._TEXT.parentheses("x ⇒ y"), "(x ⇒ y)")

    def test_recursive_conjunctions_and_implications_keep_all_members(self):
        renderer = vh.ReadableForm({}, {}, structured_implications=True)
        expressions = [
            "(&(in[a,N])(in[b,N])(>[](in[a,N])(>[](in[b,N])(=[a,b]))))",
            "(>[](in[a,N])(&(in[b,N])(in[c,N])))",
            "!(>[x](in[x,N])(>[](in[x,S])!(in[x,T])))",
            "(strictOrder[N,*,a,b])",
            "!(strictOrder[N,*,a,b])",
        ]
        for expression in expressions:
            with self.subTest(expression=expression):
                text = renderer.formula(expression)
                self.assertIn('class="rf-conjunction"', text)
                self.assertGreaterEqual(text.count('class="rf-conjunct"'), 2)
                self.assertEqual(
                    renderer.formula(expression, False),
                    vh.ReadableForm({}, {}).formula(expression, False),
                )
        nested = renderer.formula(expressions[0])
        self.assertEqual(nested.count('class="rf-conjunct"'), 3)
        self.assertIn("class='rf-premises'", nested)
        self.assertIn('class="rf-parentheses"', nested)
        for formula in ("a", "b", "a = b"):
            self.assertIn(formula, nested)
        self.assertEqual(renderer._conjunction(["A"], vh._HTML), "A")
        self.assertEqual(renderer._conjunction(["A", "B"], vh._TEXT), "A ∧ B")
        self.assertEqual(
            vh.ReadableForm({}, {})._conjunction(["A", "B"], vh._HTML), "A ∧ B"
        )
        self.assertEqual(
            renderer._conjunction(["A", "B"], vh._HTML).count('class="rf-conjunct"'), 2
        )
        with self.assertRaises(AssertionError):
            renderer._conjunction([], vh._HTML)
        with self.assertRaises(AssertionError):
            renderer._conjunction(["A", ""], vh._HTML)

    def test_centered_conclusion_preserves_inline_subscript_grouping(self):
        conclusion = "v<sub>1</sub> + v<sub>2</sub> = v<sub>3</sub>"
        text = vh._implication_columns(["A", "B"], conclusion)
        self.assertIn(
            "<span class='rf-conclusion-content'>" + conclusion + "</span>", text
        )
        self.assertEqual(text.count("class='rf-conclusion-content'"), 1)
        self.assertEqual(text.count("class='rf-premise'"), 2)

    def test_exact_duplicate_conjuncts_are_displayed_once_without_text_merging(self):
        renderer = vh.ReadableForm({}, {}, structured_implications=True)
        raw = "!(&(&(&(&(implication26[V1,N,+,i0])(implication27[V1,N,+,i0]))(implication28[N,+,i0,i0,V1]))(in[i0,N]))(in[i0,N]))"
        parts = vh._conjunction_parts(vh.parse_mpl(raw).inner)
        self.assertEqual(len(parts), 5)
        self.assertEqual(len(renderer._display_parts(parts, vh._HTML)), 4)
        self.assertEqual(len(renderer._display_parts(parts, vh._TEXT)), 5)
        self.assertEqual(
            len(vh.ReadableForm({}, {})._display_parts(parts, vh._HTML)), 5
        )
        self.assertEqual(renderer._display_parts([], vh._HTML), [])
        different_formulas = [vh.parse_mpl("(in[i0,N])"), vh.parse_mpl("(in[0,N])")]
        self.assertEqual(
            renderer._display_parts(different_formulas, vh._HTML), different_formulas
        )
        expression = "!(&(&(in[x,N])(in[i0,N]))(in[i0,N]))"
        self.assertEqual(renderer.formula(expression).count('class="rf-conjunct"'), 2)
        self.assertEqual(
            renderer.formula(expression, False),
            vh.ReadableForm({}, {}).formula(expression, False),
        )
        repeated_premise = "(>[](in[x,N])(>[](in[x,N])(=[x,x])))"
        self.assertEqual(
            renderer.formula(repeated_premise).count("class='rf-premise'"), 1
        )

    def test_multiline_groups_receive_one_matching_outer_pair(self):
        renderer = vh.ReadableForm({}, {}, structured_implications=True)
        group = renderer._conjunction(["A", "B"], vh._HTML)
        self.assertTrue(group.startswith('<span class="rf-parentheses">'))
        self.assertEqual(group.count('class="rf-fence"'), 2)
        self.assertEqual(vh._HTML.parentheses(group), group)
        combined = group + " ∨ " + group
        self.assertEqual(vh._HTML.parentheses(combined).count('class="rf-fence"'), 6)
        premises = vh._implication_columns(["A", "B"], "C")
        self.assertEqual(premises.count('class="rf-fence"'), 2)
        self.assertLess(
            premises.rindex('class="rf-fence"'), premises.index("class='rf-arrow'")
        )
        self.assertNotIn("rf-fence", vh._implication_columns(["A"], "C"))
        self.assertEqual(
            vh._implication_columns([group], "C").count('class="rf-fence"'), 2
        )
        with self.assertRaisesRegex(AssertionError, "Unclosed"):
            vh._HTML.parentheses('<span class="rf-parentheses"><span>unfinished')

    def test_column_contract_and_default_legacy(self):
        with self.assertRaises(AssertionError):
            vh._implication_columns([], "A")
        with self.assertRaises(AssertionError):
            vh._implication_columns(["A"], "")
        self.assertIn("rf-premises", vh._implication_columns(["A", "B"], "C"))
        self.assertNotIn(
            "rf-implication", vh.ReadableForm({}, {}).theorem("(>[](in[a,N])(=[a,a]))")
        )

    def test_narrow_layout_keeps_groups_whole_and_wraps_implications(self):
        import generate_full_proof_graph as gen
        import proof_reader_html as reader

        css = gen._READABLE_CAPTION_CSS
        rule = lambda selector: css.split("\n      " + selector + " {", 1)[1].split("}", 1)[0]
        self.assertNotIn("minmax(0,1fr)", css)
        self.assertIn("minmax(min-content,1fr)", rule(".rf-parentheses"))
        self.assertIn("minmax(min-content,1fr)", rule(".rf-conjunct"))
        self.assertIn("flex-wrap:wrap", rule(".rf-implication"))
        premises = rule(".rf-implication > :first-child")
        self.assertIn("flex:2 1 0%", premises)
        self.assertIn("min-width:min-content", premises)
        self.assertIn("max-width:max-content", premises)
        self.assertNotIn("overflow-wrap:anywhere", css)
        self.assertIn("overflow-wrap:break-word", rule(".rf-premise, .rf-conclusion"))
        self.assertIn("overflow-wrap:break-word", rule(".rf-conjunct-body"))
        self.assertIn("word-break: normal", rule(".readable-grey"))
        conclusion = rule(".rf-conclusion")
        self.assertIn("flex:1 1 0%", conclusion)
        self.assertIn("min-width:min-content", conclusion)
        self.assertIn("max-width:max-content", conclusion)
        phone = css.split("@media (max-width: 480px) {", 1)[1]
        self.assertIn(".rf-nb { white-space: normal; }", phone)
        self.assertIn("minmax(min-content,1fr)", phone)
        self.assertIn(".rf-implication > :first-child { flex-basis:max(40%, 6em); }", phone)
        self.assertIn(".rf-conclusion { flex-basis:max(50%, 6em); }", phone)
        self.assertLess(css.index(".rf-nb { white-space: nowrap; }"), css.index("@media (max-width: 480px)"))
        self.assertIn("white-space: nowrap", rule(".rf-unit"))
        self.assertNotIn(".rf-unit", phone)
        negation = rule(".rf-negation")
        self.assertIn("display:inline-flex", negation)
        self.assertIn("max-width:100%", negation)
        reader_phone = reader.STYLE.split("@media (max-width:480px) {", 1)[1].split("\n}\n", 1)[0]
        self.assertIn(".common-notation-only .readable-title { font-size:1.3em; }", reader_phone)
        self.assertIn(".common-line { flex-wrap:wrap; }", reader_phone)
        self.assertIn(".common-line .readable-grey { flex:1 0 100%; }", reader_phone)
        self.assertIn(".common-citations { flex:1 0 100%; }", reader_phone)
        self.assertIn(".reader-section .reader-block-kind { display:inline-block; }", reader_phone)
        self.assertIn("h2[id] { scroll-margin-top", reader.STYLE)
        self.assertIn("body:not(:has(.reader-section .subproof-card)) .reader-page-controls { display:none; }",
                      reader.STYLE)

    def test_chapter_page_keeps_header_and_flashes_the_cited_row(self):
        import inspect
        import generate_full_proof_graph as gen

        page = inspect.getsource(gen._generate_proof_graph_pages)
        self.assertNotIn("padding-bottom: 200em", page)
        self.assertIn('body::after { content: ""; display: block; height: 100vh; }', page)
        index_head = page.split('index_head = f"""', 1)[1].split('</style>', 1)[0]
        self.assertIn('body::after {{ content: ""; display: block; height: 100vh; }}', index_head)
        self.assertNotIn("scrollIntoView({block:'center'})", page)
        self.assertIn("function placeBelowHeader(target)", page)
        self.assertIn("placeBelowHeader(target);", page.split("function revealProofHash()", 1)[1].split("function placeBelowHeader", 1)[0])
        self.assertIn("targetEl.classList.add('jump-flash')", page)
        self.assertNotIn("targetCard.classList.add('jump-flash')", page)
        phone = page.split("@media (max-width: 480px) {", 1)[1].split("}\n     }", 1)[0]
        self.assertIn(".validity-tag { white-space: normal; max-width: 100%; }", phone)
        self.assertIn("max-width:600px; box-sizing:border-box;", page)


if __name__ == "__main__":
    unittest.main()
