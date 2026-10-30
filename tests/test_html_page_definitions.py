"""Per-page definition maps of the proof HTML and their shared-registry source."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import generate_full_proof_graph as generator


OR_ENTRY = {"category": "or", "signature": "(or1[u_1,u_2])",
            "elements": ["(in[u_1,u_2])", "!(=[u_1,u_2])"], "implications": []}
IMPLICATION_ENTRY = {"category": "implication", "signature": "(implication7[u_1,u_2])",
                     "elements": ["(in[3,u_1])", "(in2[3,u_2,u_1])"], "implications": []}
ATOMIC_ENTRY = {"category": "atomic", "signature": "(in[u_1,u_2])", "elements": [""]}


def _page(*parts):
    spans = "".join(f'<span class="clickable" data-parts="{p}">x</span>' for p in parts)
    return f"<script>const GL_BINARY_MAP = {generator._PAGE_DEFINITIONS_MARKER};</script>{spans}"


class EmbedPageDefinitionsTests(unittest.TestCase):
    MAP = {"or1": {"signature": "(or1[x1,x2])", "mpl": "!(&!(in[x1,x2])(=[x1,x2]))"},
           "implication7": {"signature": "(implication7[x1,x2])", "mpl": "(>[y1](in[y1,x1])(in2[y1,x2,x1]))"},
           "unused9": {"signature": "(unused9[x1])", "mpl": "(in[x1,x1])"}}

    def test_page_embeds_exactly_the_named_definitions_in_name_order(self):
        page = generator.embed_page_definitions(_page("or1,in", "implication7,or1", ""), self.MAP)
        embedded = page.split("const GL_BINARY_MAP = ", 1)[1].split(";</script>", 1)[0]
        self.assertEqual(list(json.loads(embedded)), ["implication7", "or1"])
        self.assertEqual(json.loads(embedded)["or1"], self.MAP["or1"])
        self.assertNotIn(generator._PAGE_DEFINITIONS_MARKER, page)
        self.assertNotIn("unused9", page)

    def test_page_without_named_definitions_embeds_an_empty_map(self):
        page = generator.embed_page_definitions(_page("in"), self.MAP)
        self.assertIn("const GL_BINARY_MAP = {};", page)

    def test_escaped_attribute_values_name_the_same_definitions(self):
        page = generator.embed_page_definitions(_page("&#x6f;r1"), self.MAP)
        self.assertIn('{"or1": {', page)

    def test_marker_must_occur_exactly_once(self):
        for page in ("<p>no marker</p>", _page("or1") + generator._PAGE_DEFINITIONS_MARKER):
            with self.assertRaisesRegex(AssertionError, "exactly once"):
                generator.embed_page_definitions(page, self.MAP)


class SharedRegistryMapTests(unittest.TestCase):
    def test_map_is_built_from_the_shared_registry_only(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "GL_binary_shared.json").write_text(json.dumps(
                {"or1": OR_ENTRY, "implication7": IMPLICATION_ENTRY, "in": ATOMIC_ENTRY}), encoding="utf-8")
            (root / "GL_binary_Peano.json").write_text(json.dumps(
                {"or9": OR_ENTRY, "in": ATOMIC_ENTRY}), encoding="utf-8")
            built = generator.build_gl_binary_map(root)
        self.assertEqual(set(built), {"or1", "implication7"})
        self.assertEqual(built["or1"], {"signature": "(or1[x1,x2])", "mpl": "!(&!(in[x1,x2])(=[x1,x2]))"})
        self.assertEqual(built["implication7"]["signature"], "(implication7[x1,x2])")
        self.assertEqual(built["implication7"]["mpl"], "(>[y1](in[y1,x1])(in2[y1,x2,x1]))")

    def test_missing_directory_has_no_definitions_and_missing_shared_file_asserts(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(generator.build_gl_binary_map(root / "missing"), {})
            (root / "GL_binary_Peano.json").write_text(json.dumps({"or1": OR_ENTRY}), encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "missing shared definition registry"):
                generator.build_gl_binary_map(root)


if __name__ == "__main__":
    unittest.main()
