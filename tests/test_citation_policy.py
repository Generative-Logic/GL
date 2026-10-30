"""Cross-consumer citation precedence and exact scope identity."""

import unittest
from proof_export.citation_policy import resolve_citation_source


class CitationPolicyTests(unittest.TestCase):
    def resolve(
        self,
        tag="implication",
        exact=None,
        aliases=None,
        produced=(),
        registered=False,
        template=False,
        optional=False,
        key=("fact", "main", "ordinary"),
        alias=("alpha", "main", "ordinary"),
    ):
        return resolve_citation_source(
            tag,
            key,
            alias,
            exact or {},
            aliases or {},
            set(produced),
            registered,
            template,
            optional,
        )

    def test_exact_zero_source_precedes_theorem_alias(self):
        exact = {("fact", "main", "ordinary"): 0}
        aliases = {("alpha", "main", "ordinary"): 4}
        self.assertEqual(
            self.resolve(exact=exact, aliases=aliases, registered=True), ("local", 0)
        )
        self.assertEqual(self.resolve(aliases=aliases), ("local", 4))
        self.assertEqual(self.resolve("zoom in", aliases=aliases), ("pending", None))

    def test_pending_producer_blocks_external_fallback(self):
        produced = [("fact", "main", "ordinary")]
        self.assertEqual(
            self.resolve(produced=produced, registered=True, template=True),
            ("pending", None),
        )
        self.assertEqual(
            self.resolve("or theorem", produced=produced, optional=True),
            ("pending", None),
        )
        self.assertEqual(
            self.resolve("compilation", produced=produced, registered=True),
            ("theorem", None),
        )

    def test_compilation_prefers_registered_theorem_to_local_copy(self):
        exact = {("fact", "main", "ordinary"): 0}
        self.assertEqual(
            self.resolve("compilation", exact=exact, registered=True), ("theorem", None)
        )
        self.assertEqual(self.resolve("compilation", exact=exact), ("local", 0))

    def test_external_template_and_optional_dispositions_remain_distinct(self):
        self.assertEqual(
            self.resolve(registered=True, template=True), ("theorem", None)
        )
        self.assertEqual(self.resolve(template=True), ("template", None))
        self.assertEqual(self.resolve("or theorem", optional=True), ("optional", None))
        self.assertEqual(self.resolve("or theorem"), ("pending", None))
        with self.assertRaises(AssertionError):
            self.resolve(optional=True)

    def test_namespace_and_proof_scope_do_not_alias(self):
        exact = {
            ("fact", "main_boundary_sibling", "ordinary"): 0,
            ("fact", "main", "integration_goal"): 1,
        }
        aliases = {("alpha", "main_boundary_sibling", "ordinary"): 2}
        self.assertEqual(self.resolve(exact=exact, aliases=aliases), ("pending", None))
        self.assertEqual(
            self.resolve(
                exact=exact, aliases=aliases, key=("fact", "main", "integration_goal")
            ),
            ("local", 1),
        )


if __name__ == "__main__":
    unittest.main()
