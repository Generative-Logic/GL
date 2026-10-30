"""The proof books' theme: every color role a page uses is defined on both sides, no literal palette remains."""

import re
import unittest
from pathlib import Path

import proof_theme

ROOT = Path(__file__).resolve().parent.parent
PAGE_SOURCES = ("generate_full_proof_graph.py", "proof_reader_html.py", "lean_pages.py")


def _definitions(css: str) -> set:
    return set(re.findall(r"(--gl-[a-z0-9-]+)\s*:", css))


class ProofThemeTests(unittest.TestCase):
    def setUp(self):
        tokens = proof_theme.THEME_TOKENS_CSS
        self.day, self.night = tokens.split("@media (prefers-color-scheme: dark)", 1)
        self.day_names = _definitions(self.day)
        self.night_names = _definitions(self.night)

    def test_every_referenced_role_is_defined(self):
        sources = [proof_theme.THEME_SKIN_CSS] + [(ROOT / name).read_text(encoding="utf-8") for name in PAGE_SOURCES]
        used = set()
        for text in sources:
            used |= set(re.findall(r"var\((--gl-[a-z0-9-]+)\)", text))
        self.assertTrue(used, "no theme role referenced")
        self.assertEqual(sorted(used - self.day_names), [])

    def test_night_side_redefines_every_color_role(self):
        colors = {name for name in self.day_names if not name.startswith(("--gl-font-", "--gl-horizon"))}
        self.assertEqual(sorted(colors - self.night_names), [])
        self.assertEqual(sorted(self.night_names - self.day_names), [])

    def test_page_sources_carry_no_literal_palette(self):
        for name in PAGE_SOURCES:
            text = (ROOT / name).read_text(encoding="utf-8")
            self.assertEqual(re.findall(r"#[0-9A-Fa-f]{6}\b|rgba\(", text), [], name)

    def test_negative_undefined_role_is_reported(self):
        css = ":root { --gl-text: #000; }"
        self.assertEqual(sorted({"--gl-missing"} - _definitions(css)), ["--gl-missing"])


if __name__ == "__main__":
    unittest.main()
