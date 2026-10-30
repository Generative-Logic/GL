# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Proof-book layout anchors and the separate connected certification corpora."""
MAIN_ANCHOR_CORPORA = {
    "AnchorPeano": "peano",
    "AnchorGauss": "gauss",
    "AnchorFTA": "fta",
}
MAIN_ANCHORS = frozenset(MAIN_ANCHOR_CORPORA)
INCUBATOR_ANCHORS = frozenset({"AnchorIncubator3", "AnchorIncubator8"})
ORDERED_BOOK_ANCHORS = MAIN_ANCHORS | INCUBATOR_ANCHORS
