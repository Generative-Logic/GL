# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# ------------------------------------------------------------------------------
#
# This software is also available under a commercial license. For details,
# see: https://generative-logic.com/license
#
# Contributions to this project must be made under the terms of the
# Contributor License Agreement (CLA). See the project's CONTRIBUTING.md file.

# !/usr/bin/env python3
"""
generate_full_proof_graph.py

Generates a set of HTML proof pages:
- An index page with a Table of Contents
- One HTML file per theorem (with navigation links)

Each theorem tuple is now (theorem_name:str, method:str, var_name:str).
Default output directory: full_proof_graph
"""
import copy
import json
import os
import html
from contextvars import ContextVar
from functools import lru_cache

import re
import shutil

import expression_utils
from proof_hierarchy import hierarchical_scope_items, is_reader_theorem, theorem_alpha_keys
from expression_utils import disintegrate_implication, replace_keys_in_string
from typing import Dict

from configuration_reader import configuration_reader
from parameters import debug

_RENDER_CACHE = ContextVar("gl_html_render_cache", default=None)

# =============================================================================
# License metadata injected into every generated HTML page.
# Kept as module-level constants so the three page templates (tags, index,
# chapter) stay in sync. GL is dual-licensed: AGPLv3 + commercial.
# =============================================================================
LICENSE_SOURCE_COMMENT = """<!--
  Generative Logic proof output
  Copyright © 2025-2026 Generative Logic UG
  Licensed under GNU AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
  Use outside AGPLv3 obligations, including AI model training,
  dataset construction, or redistribution, requires a commercial license:
  https://generative-logic.com/license/
-->"""

LICENSE_HEAD_META = """  <!-- License metadata -->
  <meta name="copyright" content="© 2025-2026 Generative Logic UG">
  <meta name="license" content="AGPL-3.0-or-later">
  <meta name="robots" content="noindex, follow, noai, noimageai">
  <link rel="license" href="https://www.gnu.org/licenses/agpl-3.0.html">
  <script type="application/ld+json">
  {
    "@context": "https://schema.org",
    "@type": "CreativeWork",
    "name": "Generative Logic Proof Graph",
    "copyrightHolder": {
      "@type": "Organization",
      "name": "Generative Logic UG",
      "url": "https://generative-logic.com/"
    },
    "copyrightYear": 2026,
    "license": "https://www.gnu.org/licenses/agpl-3.0.html",
    "isAccessibleForFree": true,
    "usageInfo": "https://generative-logic.com/license/"
  }
  </script>"""

LICENSE_FOOTER = """  <div style="margin-top:2em; padding-top:1em; border-top:1px solid var(--gl-rule); font-size:0.8em; color:var(--gl-text-muted); display:flex; align-items:center; gap:0.9em;">
    <a href="https://generative-logic.com" style="display:inline-flex; align-items:center; flex-shrink:0;" aria-label="Generative Logic homepage">
      <img src="gl-logo.png" alt="Generative Logic" width="40" height="40" style="display:block;" />
    </a>
    <span>
      Proof graph structure, presentation, and provenance chains
      © 2025-2026 Generative Logic UG. Licensed under
      <a href="https://www.gnu.org/licenses/agpl-3.0.html">AGPLv3</a>.
      Use outside AGPLv3 obligations, including AI model training,
      dataset construction, or redistribution, requires a
      <a href="https://generative-logic.com/license/">commercial license</a>.
    </span>
  </div>"""

import visu_helpers
import lean_pages
import proof_theme

# Style of the grey readable captions, shared by the chapter, index and Lean
# pages: the caption block, its labels (RULE: / APPLIED: / …), the sentence
# keywords (from / follows), the stacked sum and product, and the
# blackboard-bold ℕ. The Unicode ℕ glyph in Arial / system sans-serif is a
# fixed-weight stylistic character (font-weight has no effect: the font carries
# one weight for U+2115), so text-stroke plus text-shadow thicken it.
# No bracketed group, conjunct column, premise group or conclusion shrinks
# below its unbreakable content (min-content), or a bracket or the arrow would
# cross the text. Formulas break only between words: break-word, unlike
# anywhere, never lowers a box's narrowest width, so no column collapses to
# single characters. For the wrap decision an implication's premise group
# claims its natural width but at most half the line, and the conclusion its
# narrowest width; the premises then take the free space first, up to their
# natural width, and the conclusion the rest. So a long row narrows before it
# wraps, and short premises are never squeezed by a wide conclusion, which
# moves with its arrow onto the next line instead. On phone widths, formula
# pieces may wrap between words, brackets and indents tighten and the
# statement title shrinks, so a statement fits without sideways scrolling.
_READABLE_CAPTION_CSS = """
      .readable-grey { color: var(--gl-text-muted); font-weight: normal; font-size: 0.95em; line-height: 1.6; margin-top: 0.25em; overflow-wrap: normal; word-break: normal; }
      .readable-grey.readable-title { font-weight: bold; font-size: 2.2em; margin: 0.15em 0 1.1em 20px; line-height: 1.45; }
      .readable-grey.readable-index { font-size: 1.15em; margin: 0.1em 0 0.2em 20px; }
      .readable-grey.reads { color: var(--gl-text-quiet); }
      .readable-grey.subproof-reads { margin: 0.1em 0 0.4em 0; }
      .rf-implication { display:inline-flex; flex-wrap:wrap; align-items:stretch; gap:0.35em 0.65em; max-width:100%; vertical-align:middle; white-space:normal; }
      /* Premise group and conclusion share the line two to one, each at most
         its natural width (a premise group is a column of formulas, which
         wrapping makes tall); the conclusion moves below only when their
         narrowest forms do not fit beside the arrow. */
      .rf-implication > :first-child { flex:2 1 0%; min-width:min-content; max-width:max-content; }
      .rf-conjunction { display:inline-flex; flex-direction:column; gap:0.45em; padding-left:0.2em; max-width:100%; box-sizing:border-box; vertical-align:middle; }
      .rf-conjunct { display:grid; grid-template-columns:1em minmax(min-content,1fr); gap:0.35em; align-items:baseline; }
      .rf-conjunct-body { min-width:0; white-space:normal; overflow-wrap:break-word; }
      .rf-premises { display:flex; flex:0 1 auto; flex-direction:column; justify-content:center; padding-left:0.3em; gap:0.45em; min-width:0; }
      .rf-premise, .rf-conclusion { white-space:normal; overflow-wrap:break-word; min-width:0; }
      .rf-arrow { display:flex; align-items:center; flex:0 0 auto; line-height:1; font-size:2.2em; font-weight:700; padding:0 0.1em; }
      .rf-conclusion { display:flex; align-items:center; flex:1 1 0%; min-width:min-content; max-width:max-content; }
      .rf-conclusion-content { display:block; min-width:0; max-width:100%; }
      .rf-parentheses { display:inline-grid; grid-template-columns:0.6em minmax(min-content,1fr) 0.6em; align-items:stretch; gap:0.15em; max-width:100%; vertical-align:middle; }
      .rf-paren-content { align-self:center; min-width:0; }
      .rf-lines { display:inline-flex; flex-direction:column; gap:0.2em; max-width:100%; vertical-align:middle; }
      .rf-continued { padding-left:1.6em; }
      .rf-fence { display:block; position:relative; min-height:1.4em; align-self:stretch; }
      .rf-fence svg { position:absolute; inset:0; width:100%; height:100%; overflow:visible; }
      .rf-fence path { fill:none; stroke:currentColor; stroke-width:1.6; vector-effect:non-scaling-stroke; }
      .rf-fence-text { position:absolute; width:1px; height:1px; overflow:hidden; clip-path:inset(50%); }

      .readable-grey sub { font-size: 0.7em; }
      .rf-nb { white-space: nowrap; }
      .rf-unit { white-space: nowrap; }
      .rf-negation { display:inline-flex; align-items:center; max-width:100%; vertical-align:middle; }
      .rf-label { font-size: 0.72em; letter-spacing: 0.08em; color: var(--gl-text-faint); margin: 0 0.55em 0 1.4em; vertical-align: 0.08em; }
      .rf-kw { font-weight: normal; font-style: italic; color: var(--gl-text-quiet); margin: 0 0.4em; }
      .rf-label.rf-lead, .rf-kw.rf-lead { margin-left: 0; }
      .rf-big { display: inline-flex; flex-direction: column; align-items: center; vertical-align: middle; margin: 0 2px; line-height: 1; }
      .rf-big-hi, .rf-big-lo { font-size: 0.6em; }
      .rf-big-sym { font-size: 1.8em; line-height: 0.8; margin: 2px 0 6px 0; }
      .rf-big-lo { margin-top: 2px; }
      .rf-bigparen { font-size: 2em; vertical-align: middle; font-weight: normal; }
     .bb-N {
     font-weight: 900;
     font-size: 1.1em;
     letter-spacing: 0.02em;
     -webkit-text-stroke: 0.6px currentColor;
     text-shadow:
       0.4px 0 0 currentColor,
       -0.4px 0 0 currentColor,
       0 0.4px 0 currentColor,
       0 -0.4px 0 currentColor;
     }
      @media (max-width: 480px) {
        .rf-nb { white-space: normal; }
        /* Formula pieces now wrap between words, so a side squeezed beside the
           arrow could shrink to one word per line: each side claims at least
           its natural width or its share of the line (6em at least), and the
           conclusion moves below when both claims do not fit. */
        .rf-implication > :first-child { flex-basis:max(40%, 6em); }
        .rf-conclusion { flex-basis:max(50%, 6em); }
        .rf-parentheses { grid-template-columns:0.45em minmax(min-content,1fr) 0.45em; gap:0.05em; }
        .rf-conjunct { grid-template-columns:0.8em minmax(min-content,1fr); gap:0.2em; }
        .readable-grey.readable-title { font-size: 1.3em; margin-left: 4px; }
        .readable-grey.readable-index { margin-left: 4px; }
      }
"""
from visu_helpers import expand_expr
from pathlib import Path

# wherever this file lives, assume the project root is its parent folder
PROJECT_ROOT = Path(__file__).resolve().parent

# global mapping from displayed theorem title → its chapter file
theorem_to_file = {}
# alpha-normalized theorem shape -> file (only unique matches)
theorem_shape_to_file = {}

# GL-binary map populated by generate_proof_graph_pages. Used by
# _subproof_explanation to inline the expanded MPL form of an or<N>
# expression beside the symbolic name (so the reader sees the full
# !(&!E1!E2) De-Morgan form right next to the abbreviated `(orK[..])`).
_gl_binary_map_for_render: dict = {}

# Readable-caption renderer of the graph being rendered, created by
# _generate_proof_graph_pages from the same GL binaries; every grey
# "Readable version" line goes through it.
_readable_form_for_render: visu_helpers.ReadableForm = visu_helpers.ReadableForm({}, {})
_known_theorems_for_reader = ()
# Alpha keys of exactly `_known_theorems_for_reader`, computed once per render
# so a reader chapter adds only the keys of its own theorem rows.
_known_theorem_keys_for_reader = frozenset()

# Sibling-batch (cross-pipeline) lookup. Populated when
# generate_proof_graph_pages receives a `sibling_graphs` parameter so an
# incubator chapter that cites a Peano-batch external theorem can link
# directly into the main pipeline's HTML output. Targets carry relative
# URLs from the *current* output dir to the sibling's HTML output dir.
# Rendered links open in a new browser tab (target="_blank") so the
# sibling proof opens beside, not in place of, the current chapter.
sibling_theorem_to_file = {}
sibling_theorem_shape_to_file = {}

# Set of alpha-equivalent shapes for every external theorem that this
# pipeline registers (loaded from this run's processed proof graph
# external_theorems.txt). Used to detect cited rules that are "known
# external" but resolve to no chapter — neither locally nor in any
# sibling. Such cells render as orphan externals: no link, but the
# right-click popup explains the situation instead of just dumping the
# expression. The orphan path covers user-supplied externally-provided
# theorems whose proof graph is genuinely not part of any pipeline
# release.
external_orphan_shapes: set[str] = set()

# Tag descriptions for the proof graph reference page and right-click popups
TAG_DESCRIPTIONS = {
    "implication grouping": (
        "Implication conclusions grouped",
        "The cited implications have the same complete premise and compatible binders. "
        "Their conclusions are combined into the displayed conjunction, preserving every source proof."
    ),
    "contraposition": (
        "Contrapositive implication",
        "The cited implication is reversed with both sides negated. "
        "All bound variables, fixed parameters and the source scope are preserved."
    ),
    "implication composition": (
        "Implications composed",
        "Two one-premise implications are joined at their matching intermediate expression. "
        "The new implication retains both source proofs and holds at their deepest compatible scope."
    ),
    "implication": (
        "Hash-table inference",
        "A premise was matched against a universally quantified implication rule "
        "stored in hash memory, and the conclusion was emitted. "
        "The first dependency is the rule itself; the remaining dependencies are "
        "the expressions that matched the rule's premises."
    ),
    "definition projection": (
        "Property from a definition",
        "The stated property is a conjunct of the cited fact, after unfolding "
        "its named conjunction definitions. The source stays in this proof scope."
    ),
    "expansion": (
        "Named expression expanded",
        "A named compound expression (e.g. NaturalNumbers, fXY) was expanded "
        "into its compiled definition structure from the GL binary. "
        "The dependency is the named expression that was expanded."
    ),
    "compilation": (
        "Implication compiled to compact form",
        "An implication entering the mail broadcast channel was also "
        "compiled to its compact named form "
        "<code>(implication&lt;N&gt;[args])</code>. The row's expression is "
        "the compact form; the single dependency is the original expanded "
        "implication it was compiled from. The compact name's GL-binary "
        "definition, instantiated with its arguments, reconstructs the "
        "original. Preparatory provenance for the ASIC build; it does not "
        "change which theorems are proved."
    ),
    "disintegration": (
        "Compound expression decomposed",
        "A conjunction (&amp;), existence, or OR node was broken apart into its "
        "constituent sub-expressions. For conjunctions, each element is extracted. "
        "For existence nodes, the left element (with a fresh bound variable) and "
        "the right element are produced. "
        "For OR nodes, K mutual-exclusion sub-implications of the form "
        "<code>(!D_others &rarr; D_k)</code> are emitted &mdash; one per disjunct "
        "&mdash; so a two-disjunct OR <code>A &or; B</code> disintegrates into the "
        "rule pair <code>!A &rarr; B</code> and <code>!B &rarr; A</code>; the "
        "per-branch case-split that follows (one scope per disjunct) is tagged "
        "separately as <i>or disintegration</i>."
    ),
    "task formulation": (
        "Proof premise (root assumption)",
        "A root premise of the theorem under proof, asserted without justification "
        "at the top of the chapter. In single-chapter proofs, the row's left side is one of "
        "the theorem implication's chain premises (or the anchor application, which "
        "is always a premise by structure). In contradiction proofs &mdash; theorems "
        "whose head starts with <code>!</code> &mdash; the un-negated head is also "
        "valid as a task-formulation row, seeded into the contradiction LB as the "
        "hypothesis to be disproved. Symmetrically, for a positive-headed theorem "
        "proved by reductio, the negated head is valid as a task-formulation row "
        "&mdash; the complement contradiction LB's assumed hypothesis. "
        "Induction-hypothesis seeding uses the "
        "<i>recursion</i> tag, not task formulation."
    ),
    "equality1": (
        "Argument substitution via equality",
        "An expression's argument was replaced using an equality fact. "
        "If (=[a,b]) is known and f(...,a,...) exists, then f(...,b,...) is derived."
    ),
    "equality2": (
        "Transitivity of equality",
        "From (=[a,b]) and (=[b,c]), the equality (=[a,c]) is derived. "
        "This is the standard transitivity rule for the equality relation."
    ),
    "symmetry of equality": (
        "Symmetry of equality",
        "From (=[a,b]), the symmetric equality (=[b,a]) is derived. "
        "This is the standard symmetry rule for the equality relation."
    ),
    "symmetry of inequality": (
        "Symmetry of negated equality",
        "From !(=[a,b]), the symmetric inequality !(=[b,a]) is derived. "
        "The mirror of <i>symmetry of equality</i> for negated equality: "
        "equality is symmetric, and so is its negation."
    ),
    "recursion": (
        "Induction-hypothesis seeding",
        "Induction-hypothesis row at the top of an induction triad's check chapter. "
        "In <code>check_zero</code> chapters, the induction variable is identified "
        "with <code>i0</code> (the base case). In "
        "<code>check_induction_condition</code> chapters, the successor form is "
        "asserted as the inductive step's premise &mdash; <code>s(v_prev) = v_current</code> "
        "&mdash; identifying the current step variable as the successor of the "
        "previous-step variable to which the hypothesis applies. A weak induction's "
        "hypothesis is the theorem at that previous-step variable; a strong "
        "induction's hypothesis is the theorem at every value <code>m</code> below the "
        "current step variable <code>n</code>, bounded by <code>m + x = n</code> and "
        "<code>m &ne; n</code>."
    ),
    "theorem": (
        "Previously proved theorem",
        "A theorem that was proved in an earlier chapter is used as an inference "
        "rule. The dependency links to the chapter where the theorem was originally proved."
    ),
    "reformulation for integration and": (
        "Reformulated for reverse-disintegration (and)",
        "An expression of category <i>and</i> was reformulated into a chain of "
        "nested implications <code>(&gt;[]elem1(&gt;[]elem2...compact))</code> "
        "suitable for integration. The expanded <code>(&amp;...)</code> form is "
        "flattened element-by-element, then re-wrapped right-to-left around the "
        "compact head so the expression can be reassembled by the integration "
        "step."
    ),
    "reformulation for integration >[bound]": (
        "Reformulated for reverse-disintegration (existence, bound retained)",
        "An expression of category <i>existence</i> was reformulated into the "
        "negated-existence implication form for integration. The outermost "
        "<code>&gt;[...]</code> retains a bound variable (typically "
        "<code>pi_lev_&lt;N&gt;</code>) introduced by the reformulation; the "
        "verifier compares the converted form against the canonical existence "
        "shape derived from the GL binary entry."
    ),
    "reformulation for integration >[]": (
        "Reformulated for reverse-disintegration (existence, empty bound)",
        "An expression of category <i>existence</i> was reformulated into the "
        "negated-existence implication form for integration, with an empty "
        "outermost <code>&gt;[]</code>. The bound variable was stripped because "
        "the slot was already occupied; the verifier infers it from the GL "
        "binary entry and validates the existence-form match."
    ),
    "reformulation for integration flattened": (
        "Reformulated for reverse-disintegration (flat, multi-level)",
        "A flat integration rule for a goal: the goal's definition tree was "
        "unfolded through several levels in one implication — ad-hoc "
        "conjunction/existence products always opened, each user-defined "
        "compound either kept whole or opened one level, implications and "
        "disjunctions left whole — with the goal as head and every witness "
        "bound at its first premise. The verifier re-unfolds the goal from the "
        "GL binary and matches the premises modulo witness renaming."
    ),
    "expansion for integration": (
        "Expanded for reverse-disintegration",
        "A named expression was expanded specifically in preparation for integration. "
        "The expanded form provides the structural template needed for the "
        "reverse-disintegration step."
    ),
    "premise element": (
        "Implication premise consumed",
        "A premise of an implication was matched during the integration process. "
        "This marks one of the conditions that needed to be satisfied before "
        "the implication's conclusion could be assembled."
    ),
    "validity name": (
        "Implication scope identifier",
        "Ties an expression to the specific implication whose scope it belongs to "
        "during the integration process. Ensures that premises and conclusions "
        "are matched within the correct logical context."
    ),
    "anchor handling": (
        "Anchor slot rename",
        "Pin a raw bound-variable index in an anchor application to its "
        "anchor-slot name. The dependency is the original anchor expression; the "
        "row's left side is the renamed form (e.g. "
        "<code>(AnchorPeano[N,0_copy,s,+,*,i1])</code> &larr; "
        "<code>(AnchorPeano[N,i0,s,+,*,i1])</code> &mdash; the slot at position 2 "
        "becomes <code>0_copy</code>). At most one emission per chapter; subsequent "
        "occurrences of the same slot are tracked through <i>_copy</i> substitutions, "
        "not through additional anchor-handling rows."
    ),
    "reformulated from": (
        "Reformulation of source theorem",
        "This theorem is a reformulation of another theorem — the head (conclusion) "
        "has been rewritten using an existence-node expansion from the GL binary. "
        "The dependency links to the original source theorem."
    ),
    "variable copy": (
        "Variable copy axiom",
        "GL may introduce a free equality (=[Y, Y_copy]) at any point in any "
        "scope, where Y_copy is a fresh name manufactured by appending the "
        "suffix &quot;_copy&quot; to Y. Because Y_copy never occurs anywhere "
        "else in the model, this is a conservative extension — asserting that "
        "Y_copy equals Y cannot contradict anything. GL uses this mechanism "
        "to split duplicated argument positions (when a rule instantiation "
        "would force the same variable into two slots of the same expression) "
        "so that the disintegration and hash-burst machinery can treat the "
        "two positions independently. The equality is then reused via "
        "equality1 rewrites to glue them back together. In the weak induction "
        "step of a conjecture with two twin premises, GL also introduces "
        "(=[q, rec2]): q is the predecessor of the second premise's length, "
        "and rec2 stands beside rec, the predecessor of the induction "
        "variable; a chapter declares rec2 for one name only. The verifier "
        "enforces that every occurrence of the copy name elsewhere in the "
        "chapter has a provenance chain ending at the variable copy "
        "declaration for that exact equality, or at an integration goal that "
        "holds the name, since a goal is filed without a derivation."
    ),
    "externally provided theorem": (
        "External theorem (not proved by GL)",
        "A theorem injected from the externally_provided_theorems.txt file. "
        "This theorem was not proved by GL's own deduction engine but is "
        "accepted as a given fact for use in downstream proofs."
    ),
    "incubator back reformulation": (
        "Back-reformulated operator theorem",
        "An operator-equality theorem (e.g., a+b=c) was back-reformulated from "
        "the implication form into a direct operator statement. The dependency "
        "links to the source proof."
    ),
    "contradiction": (
        "Proved by contradiction",
        "Both an expression and its negation were derived within the same "
        "logic block, establishing a contradiction. The theorem is proved "
        "because the negation of the conclusion led to an inconsistency."
    ),
    "vacuous truth": (
        "Premise chain self-contradictory",
        "The premise chain leading to an implication was shown to be "
        "self-contradictory inside the implication's scope. The implication "
        "head is therefore trivially valid &mdash; <i>ex falso quodlibet</i>. "
        "Allows GL to close out branches whose premises lead to inconsistency "
        "without needing to derive the head separately. The record cites the "
        "opposing facts at their actual scopes. An ordinary implication closes "
        "under its scoped premises; the induction form also cites its hypothesis."
    ),
    "origin": (
        "Provenance tracking (non-checker)",
        "Non-checker chapter-meta tag. Tracks the dependency chain for an "
        "expression so the verifier can walk back to root assumptions or task "
        "formulations. Used most prominently in the contradiction-trace and "
        "vacuous-truth-trace meta-checks &mdash; the row's rest fields name "
        "the source derivation(s) that each chapter row's expression depends "
        "on, letting the verifier confirm a contradiction's grounding chain "
        "or a vacuous-truth's recursion-hypothesis route. Lives outside the "
        "<code>TAG_CHECKERS</code> dispatch table; counted as its own success/"
        "failure line in the verifier output."
    ),
    "multiplied from": (
        "Partition-based variable equalization",
        "A theorem produced by the multiplyImplication algorithm. Bell partitions "
        "of bound variables generate copies where variable groups are set equal, "
        "enabling cross-expression equalization."
    ),
    "or disintegration": (
        "Case analysis branch",
        "A disjunction (OR expression) was decomposed into individual cases. "
        "Each disjunct is processed under its own branch validity scope. "
        "The dependency is the expanded OR expression encoding all disjuncts."
    ),
    "or convergence": (
        "Case analysis convergence",
        "All branches of a case analysis (OR disintegration) have independently "
        "proved the same expression. The results converge back to the parent "
        "validity scope, completing the case analysis."
    ),
    "or theorem": (
        "OR-theorem chapter conclusion",
        "Marks a chapter whose theorem statement <em>is</em> a disjunction "
        "(an <code>or&lt;N&gt;[...]</code> node). Distinct from the per-branch "
        "bookkeeping rows <i>or branch proven</i> and <i>or branch assumption</i>: "
        "this is the chapter-conclusion record for theorems whose head is an OR. "
        "Emitted as a <code>&lt;N&gt;_or_theorem.txt</code> proof file with method "
        "label <code>or theorem</code>."
    ),
    "or elimination": (
        "Pre-split merge (or elimination)",
        "The theorem is <em>derived</em> from two previously proved guard "
        "variants — rows identical except one innermost guard premise — "
        "under a previously proved OR theorem whose two disjuncts are "
        "exactly those guards: classical or-elimination, with the OR "
        "theorem as the coverage certificate. The three dependencies cite "
        "the two variants and the licensing OR theorem. Emitted as a "
        "<code>&lt;N&gt;_or_elimination.txt</code> proof file with method "
        "label <code>or elimination</code>."
    ),
    "or branch proven": (
        "OR-introduction subproof seed",
        "Records that the parent-scope OR <code>(or&lt;N&gt;[...])</code> was "
        "opened into a per-subproof scope carrying one specific disjunct as "
        "its asserted target. Each disjunct gets its own <i>or branch proven</i> "
        "row pinning the parent OR to that subproof's namespace "
        "(<code>parent_boundary_orint_&lt;or&gt;_(&lt;disjunct&gt;)</code>). "
        "When any one subproof closes, the OR is emitted at the parent scope "
        "&mdash; this row IS the OR's derivation by design (no separate "
        "derivation row). The historical name &quot;branch&quot; refers to the "
        "OR-introduction sub-proof, not a case-split branch (those are recorded "
        "as <i>or disintegration</i>). See D-36 for the terminology note."
    ),
    "or branch assumption": (
        "OR-introduction subproof side-assumption",
        "Inside an OR-introduction subproof asserting disjunct <code>D_i</code> "
        "as the target, the negation <code>!D_j</code> of every other disjunct "
        "(<code>j &ne; i</code>) is seeded as a subproof-local assumption. "
        "This makes the OR-introduction sound &mdash; under "
        "<code>(!D_j → D_i)</code>, deriving <code>D_i</code> from "
        "<code>!D_j</code> witnesses one of the K mutual-exclusion sub-implications "
        "of the OR. Row layout: the negated-other-disjunct paired with the "
        "subproof namespace, with the parent-scope OR expression cited as "
        "<code>&lt;or-expr&gt;_integration_goal</code>."
    ),
    "zoom in": (
        "Inherited statement",
        "An unchanged expression is reused in a descendant scope. Its single dependency cites "
        "the identical expression at the exact strict-ancestor scope where it was established."
    ),
}


def _find_matching_paren_local(expr: str, start: int) -> int:
    """Return the index of the ')' matching '(' at position *start*, or -1."""
    depth = 0
    for i in range(start, len(expr)):
        if expr[i] == '(':
            depth += 1
        elif expr[i] == ')':
            depth -= 1
            if depth == 0:
                return i
    return -1


@lru_cache(maxsize=16384)
def _alpha_normalize_theorem_expr(expr: str) -> str:
    """Normalize theorem expressions up to variable renaming (alpha-equivalence).
    We replace every bracket-argument token with a stable placeholder by first occurrence.
    This lets instantiated theorem schemas (e.g. broadcasted versions) match the generic theorem page.
    """
    if not expr or not isinstance(expr, str):
        return expr
    s = expr.strip()
    if not (s.startswith('(>') or s.startswith('!(')):
        return s

    token_pattern = re.compile(r'(?<=[\[,])([^,\[\]]+)(?=[\],])')
    mapping = {}
    next_idx = 0

    def repl(m):
        nonlocal next_idx
        tok = m.group(1)
        base = _strip_u_prefixes(tok)
        if base not in mapping:
            mapping[base] = f"@{next_idx}"
            next_idx += 1
        return mapping[base]

    return token_pattern.sub(repl, s)


def _resolve_theorem_target(expr: str):
    target = theorem_to_file.get(expr)
    if target:
        return target
    shape = _alpha_normalize_theorem_expr(expr)
    return theorem_shape_to_file.get(shape)


def _resolve_sibling_theorem_target(expr: str):
    """Cross-batch lookup: if `expr` matches a theorem proven in a sibling
    pipeline (e.g. an incubator chapter citing a Peano-batch external),
    return the relative URL to the sibling's chapter HTML. Returns None
    when the expression is not a known sibling theorem.
    """
    target = sibling_theorem_to_file.get(expr)
    if target:
        return target
    shape = _alpha_normalize_theorem_expr(expr)
    return sibling_theorem_shape_to_file.get(shape)


def _is_orphan_external(expr: str) -> bool:
    """True iff `expr` is a known external (registered in this pipeline's
    external_theorems.txt) but resolves to no chapter — neither locally
    nor in any sibling pipeline. Such cells render as orphan externals
    (no link, but a right-click popup explains).
    """
    if not expr:
        return False
    shape = _alpha_normalize_theorem_expr(expr)
    if not shape:
        return False
    return shape in external_orphan_shapes


@lru_cache(maxsize=16384)
def _strip_i_prefix(expr: str) -> str:
    """Strip 'i' prefix from anchor element names (i0->0, i1->1) for display.
    Works on both bracketed args and readable math text."""
    return re.sub(r'\bi(\d+)\b', r'\1', expr)


# Extract top-level constituent expression names for GL binary display
_EXPR_NAME_RE = re.compile(r'\(([A-Za-z_][A-Za-z0-9_]*)\[')

def _extract_expr_parts(expr: str) -> str:
    """Extract unique expression names (e.g. 'in2', 'in3', 'AnchorPeano') from an expression.
    Returns comma-separated string of names that exist in the GL binary map."""
    names = list(dict.fromkeys(_EXPR_NAME_RE.findall(expr)))  # unique, order-preserving
    return ','.join(names) if names else ''


def _space_commas(s: str) -> str:
    """Add spaces after commas inside [...] brackets for display readability."""
    return re.sub(r',(?=[^\s])', ', ', s)


# Utility function to wrap clickable substrings starting with '(' or '!(' and ending at space or end-of-string
def wrap_clickable(text):
    render_cache = _RENDER_CACHE.get()
    if render_cache is not None and text in render_cache["clickable"]:
        return render_cache["clickable"][text]
    def repl(m):
        s = m.group(0)  # the raw token, e.g. "(v7*v8)=(v8*v7)"
        esc = html.escape(s, quote=True)
        parts = html.escape(_extract_expr_parts(s), quote=True)
        parts_attr = f' data-parts="{parts}"' if parts else ''
        target = _resolve_theorem_target(s)
        if target:
            # Display with w-variables, keep original v-expression for link target and data-text
            display = html.escape(_strip_i_prefix(s), quote=True)
            span = f'<span class="clickable" data-text="{esc}"{parts_attr}>{display}</span>'
            return f'<a href="{target}" class="theorem-link">{span}</a>'
        # Cross-batch (sibling) lookup: if the expression matches a theorem
        # proven in a sibling pipeline (e.g. an incubator chapter citing a
        # Peano-batch external), link to the sibling's HTML chapter. Open
        # the link in a new tab (target="_blank") so the sibling proof
        # opens beside the current chapter rather than replacing it.
        sibling_target = _resolve_sibling_theorem_target(s)
        if sibling_target:
            display = html.escape(_strip_i_prefix(s), quote=True)
            span = f'<span class="clickable" data-text="{esc}"{parts_attr}>{display}</span>'
            return (f'<a href="{sibling_target}" class="theorem-link external-link" '
                    f'target="_blank" rel="noopener">{span}</a>')
        # Orphan-external case: the expression is a registered external in
        # this pipeline's external_theorems.txt but resolves to no chapter
        # in any sibling. Mark it so the right-click popup explains the
        # situation instead of just dumping the expression. Rendered with
        # the same lavender colour as cross-batch links plus a dashed
        # underline (no ↗) to distinguish from clickable externals.
        if _is_orphan_external(s):
            display_esc = html.escape(_strip_i_prefix(s), quote=True)
            span = (f'<span class="clickable external-orphan" data-text="{esc}" '
                    f'data-external-orphan="1"{parts_attr}>{display_esc}</span>')
            return span
        # span for the normal "expand on right-click".
        # The per-implication local w/W rename for non-anchor bound vars
        # now lives in process_proof_graphs.py (ITERATION 4½), so the
        # processed proof graph already carries the rendered form. Just
        # render the cell as-is — both verifier and HTML see the same form.
        display_esc = html.escape(_strip_i_prefix(s), quote=True)
        span = f'<span class="clickable" data-text="{esc}"{parts_attr}>{display_esc}</span>'
        return span

    pattern = r"!?\([^ ]*?\)(?= |$)"
    result = re.sub(pattern, repl, text)
    # Strip i-prefix from plain text segments (outside HTML tags)
    result = re.sub(r'(?<![<"\w])i(\d+)(?!["\w>])', r'\1', result)
    # Wrap _integration_goal suffix as a right-clickable label
    result = result.replace(
        '_integration_goal',
        '<span class="integration-goal-label">_integration_goal</span>')
    if render_cache is not None:
        render_cache["clickable"][text] = result
    return result


def _anchor_signatures() -> dict[str, tuple[str, ...]]:
    """
    @brief Canonical slot names of every anchor in the current configuration.

    @details Read from each `Anchor…` entry's short MPL, e.g. `AnchorFTA` →
    `(N, i0, s, +, *, i1, i2, id)`. The readable captions give these names to
    the anchor slots of compiled theorems.

    @return Anchor name → slot names.
    """
    return {name: tuple(expression_utils.get_args(entry.short_mpl_raw))
            for name, entry in expression_utils.get_configuration_data().items()
            if name.startswith("Anchor")}


def _readable_theorem(theorem: str, as_html: bool = True) -> str:
    """
    @brief Grey readable caption of a theorem for the current graph.

    @details Delegates to the graph's `visu_helpers.ReadableForm`, created by
    `_generate_proof_graph_pages` from the graph's GL binaries: the chapter
    title, the index entry, the Lean twin page and the browser-tab title all
    show the same caption.

    @param theorem Theorem in MPL, as listed in `global_theorem_list.txt`.
    @param as_html True for page HTML, False for plain text (tab title).
    @return The caption.
    """
    return _readable_form_for_render.theorem(theorem, as_html)


def _row_caption_html(entry: list[str]) -> str:
    """
    @brief Grey readable caption of one chapter row, or the empty string.

    @details An `implication` row shows its rule, the instantiation that
    turns the rule into the row and, when the derived expression is not a
    plain relation, its reading; a `reformulated from` / `incubator back
    reformulation` row shows both forms; any other row shows "READS:" with
    its expression in mathematical notation (an integration goal without its
    `_integration_goal` marker) unless that expression is a plain relation.
    An `expansion` row shows no reading when it would repeat the reading of
    the expression it expands, or when it unfolds an anchor or the Peano
    axioms (a long constant readout). Subproof-title (`validity name`) and
    broadcast-theorem rows carry no caption; that is their defined rendering.

    @param entry Processed chapter row: expression, validity, tag, then
    (dependency, validity) pairs.
    @return `<div class='readable-grey …'>…</div>` or "".
    """
    readable = _readable_form_for_render
    tag = entry[2] if len(entry) > 2 else ""
    if tag in ("validity name", "theorem"):
        return ""
    if len(entry) <= 3 or tag not in ("implication", "reformulated from",
                                      "incubator back reformulation"):
        expression = entry[0]
        if expression.endswith("_integration_goal"):
            expression = expression[:-len("_integration_goal")]
        body = readable.reading(expression)
        if body and tag == "expansion" and len(entry) > 3:
            source = entry[3]
            if source.endswith("_integration_goal"):
                source = source[:-len("_integration_goal")]
            if (re.match(r'^!?\((Anchor\w+|NaturalNumbers)\[', source)
                    or readable.reading(source) == body):
                return ""
        return f"<div class='readable-grey reads'>{body}</div>" if body else ""
    if tag == "implication":
        premises = [entry[k] for k in range(5, len(entry), 2)]
        body = readable.rule_application(entry[0], entry[3], premises)
        derived = entry[0]
        if derived.endswith("_integration_goal"):
            derived = derived[:-len("_integration_goal")]
        reading = readable.reading(derived)
        if reading:
            body += f"<br>{reading}"
        return f"<div class='readable-grey implication'>{body}</div>"
    if tag == "reformulated from":
        body = readable.reformulation(entry[0], entry[3], "REFORMULATED FROM:")
        return f"<div class='readable-grey reformulated'>{body}</div>"
    if tag == "incubator back reformulation":
        body = readable.reformulation(entry[0], entry[3], "BACK-REFORMULATED FROM:")
        return f"<div class='readable-grey reformulated'>{body}</div>"
    return ""


def _subproof_title_reading(info: dict) -> str:
    """
    @brief Grey reading of a subproof card's title, or the empty string.

    @details An implication subproof reads as the implication it proves
    (theorem style); an OR-introduction subproof as the disjunct it asserts
    and the OR it belongs to; an OR-elimination branch as its case and the
    OR; any other scope carries no reading, which is its defined rendering.

    @param info Title metadata from `_scope_title_info`.
    @return HTML of the reading, or "".
    """
    readable = _readable_form_for_render
    kind = info["kind"]
    if kind == "subproof" and info.get("implication_expr"):
        return ("<span class='rf-label rf-lead'>PROVES:</span>"
                + readable.theorem(info["implication_expr"]))
    if kind in ("_orint_ subproof", "_ordis_ branch"):
        first = "ASSERTS:" if kind == "_orint_ subproof" else "CASE:"
        return (f"<span class='rf-label rf-lead'>{first}</span>"
                f"{readable.formula(info['disjunct'])}"
                f"<span class='rf-label'>OF:</span>{readable.formula(info['or_expr'])}")
    return ""


def _format_validity_tag(validity: str, namespace_anchor_map: dict | None = None) -> str:
    if not validity:
        return ""
    raw = validity.strip()
    # Strip the `i`-prefix from anchor element names for display
    # consistency with how regular expressions are rendered (e.g.
    # `i0`/`i1` -> `0`/`1`). Pre-fix the namespace strings showed
    # raw `i0`, `i1`, ... while the regular expressions rendered as
    # `0`, `1`, .... Same `_strip_i_prefix` regex used for both.
    display = _strip_i_prefix(raw)
    esc = html.escape(display)
    if display.startswith("(") and display.endswith(")"):
        body = f'<span class="validity-tag">{esc}</span>'
    else:
        body = f'<span class="validity-tag">({esc})</span>'
    # Every namespace renders unclickable, matching `(main)`. The
    # `namespace_anchor_map` parameter is accepted for callsite
    # compatibility but intentionally ignored — the prior subproof-
    # cross-link path turned non-main namespaces into `<a ns-jump>`
    # links, which the user wants gone.
    return f' {body}'


def format_stack_entries(stack, prefix='', cursor_index=None, reverse_entries=True, external_anchor_map=None, goal_key_norm=None, global_counter=None, global_total=None, namespace_anchor_map=None, chapter_ns_map=None, entry_indices=None, highlight_position=None, local_anchor_map=None):
    """
    @brief Render proof rows with stable scoped anchors and display numbering.

    @details Complete-scope calls derive their indices and goal highlight.
    Fragment calls retain caller-supplied scope indices, anchors and highlight
    while a shared counter follows the expanded nested proof.

    @param stack Proof rows belonging to one exact scope.
    @param prefix Stable anchor prefix for that scope and induction section.
    @param cursor_index Optional scope row to emphasize for navigation.
    @param reverse_entries Whether to reverse stored rows before rendering.
    @param external_anchor_map Existing nonlocal expression anchors.
    @param goal_key_norm Optional normalized scope conclusion.
    @param global_counter Shared mutable visible-row counter.
    @param global_total Expanded visible-row count for this proof section.
    @param namespace_anchor_map Existing scope-title anchor mapping.
    @param chapter_ns_map Complete scoped expression-to-row-anchor mapping.
    @param entry_indices Optional original scope indices for a row fragment.
    @param highlight_position Optional original scope index to highlight; -1 selects none.
    @param local_anchor_map Complete same-scope anchors when rendering fragments.
    @return Proof-row HTML, preserving references and continuous step numbers.
    @invariant Rendering changes no proof expression, premise or namespace.

    Convert a proof-stack (list of [key, validity, explanation, ing1, val1, ...]) into HTML.

    New Format Structure:
    0: Result Expression (Key)
    1: Result Validity
    2: Explanation (Method/Justification)
    3, 5, 7...: Ingredient Expressions
    4, 6, 8...: Ingredient Validities

    Step-numbering:
    - When `global_counter` (a one-element mutable list `[next_idx]`) and
      `global_total` (int) are provided, every visible row's step badge
      reads `(global_counter[0]/global_total)` and the counter advances
      across nested calls. This is how the chapter-wide common
      numbering threads through main stack + every subproof + nested
      subproofs in one continuous sequence.
    - When omitted (caller did not opt in), the function falls back to
      local-only `(local_idx/local_total)` numbering — the legacy
      behaviour preserved for any direct caller that wants per-section
      counts.
    """
    # Reverse so the earliest step is first in the output (legacy behavior)
    rev = list(stack)[::-1] if reverse_entries else list(stack)

    indices = list(range(len(rev))) if entry_indices is None else list(entry_indices)
    assert len(indices) == len(rev) and len(set(indices)) == len(indices)

    # Map each key (normalized) to a unique anchor ID
    key_map = dict(local_anchor_map or {})
    external_anchor_map = external_anchor_map or {}
    for idx, entry in zip(indices, rev):
        if not entry or 'theorem' in entry:
            continue
        key = entry[0]
        norm = re.sub(r'\s+', '', key).lower()
        key_map[norm] = f"{prefix}-entry{idx}" if prefix else f"entry{idx}"

    lines = []
    total = len(rev)
    visible_count = sum(1 for e in rev if e and 'theorem' not in e)
    use_global = global_counter is not None and global_total is not None
    step_num = 0

    # Highlight the *achieved* goal: the LAST non-theorem row in display
    # order whose expression matches goal_key_norm (when given) or the
    # last non-theorem row outright (when no goal is named — the
    # convention is that the latest derivation in display = the row that
    # closes the proof).
    #
    # Two reasons to scan from the end:
    #   (a) The same expression may appear multiple times in a stack
    #       (e.g. used as a premise earlier, then re-derived). Picking
    #       the EARLIEST match would highlight the row the proof
    #       *consumes*, not the row the proof *produces*. Latest match
    #       is the produced/achieved one.
    #   (b) A trailing 'theorem' row (broadcast-theorem reference) at
    #       on-disk position 0 (= total-1 in display) was previously
    #       skipped by the `'theorem' in entry` filter inside the render
    #       loop, leaving the subproof with NO orange highlight at all.
    #       Backward scan with the same theorem-filter skips those rows
    #       and finds the real conclusion.
    highlight_idx = None
    if total > 0:
        for i in range(total - 1, -1, -1):
            entry = rev[i]
            if not entry:
                continue
            if 'theorem' in entry:
                continue
            if goal_key_norm and _norm_expr(entry[0]) != goal_key_norm:
                continue
            highlight_idx = i
            break

    if highlight_position is not None:
        highlight_idx = highlight_position

    for idx, entry in zip(indices, rev):
        if 'theorem' in entry:
            continue
        if not entry:
            continue
        step_num += 1
        if use_global:
            global_counter[0] += 1
            step_badge = f"<span class='step-badge'>({global_counter[0]}/{global_total})</span>"
        else:
            step_badge = f"<span class='step-badge'>({step_num}/{visible_count})</span>"

        key_expr = entry[0]
        key_validity = entry[1] if len(entry) > 1 else ""
        explanation = entry[2] if len(entry) > 2 else ""

        norm_key = re.sub(r'\s+', '', key_expr).lower()
        anchor = key_map.get(norm_key, "")

        if highlight_idx is not None and idx == highlight_idx:
            first, *rest = key_expr.split(' ', 1)
            first_html = wrap_clickable(first)

            if first_html.startswith('<a '):
                first_html = first_html.replace(
                    '<span class="clickable"',
                    '<span class="clickable" style="color:var(--gl-goal) !important; font-weight:bold !important;"',
                    1
                )
            else:
                first_html = f'<span class="clickable" style="color:var(--gl-goal); font-weight:bold">{first_html}</span>'

            if rest:
                rest_html = wrap_clickable(rest[0])
                key_html = f"{first_html} {rest_html}"
            else:
                key_html = first_html
        else:
            key_html = wrap_clickable(key_expr)

        # Expression diff highlight for equality1 steps
        if explanation == "equality1" and len(entry) > 3:
            key_html = _diff_highlight_html(key_html, key_expr, entry[3])

        if key_validity:
            key_html += _format_validity_tag(key_validity, namespace_anchor_map)

        # Collect dependency anchor IDs for hover-highlighting
        dep_ids = []
        for di in range(3, len(entry), 2):
            d_expr = entry[di]
            d_norm = re.sub(r'\s+', '', d_expr).lower()
            if d_norm in key_map:
                dep_ids.append(key_map[d_norm])
            elif d_norm in external_anchor_map:
                dep_ids.append(external_anchor_map[d_norm])
        deps_attr = f" data-deps=\"{' '.join(dep_ids)}\"" if dep_ids else ""
        parts = [f"<span id='{anchor}'{deps_attr}>{key_html}</span>"]

        if explanation:
            tag_key = explanation.lower().strip()
            tag_anchor = tag_key.replace(" ", "-").replace("(", "").replace(")", "")
            escaped = html.escape(explanation)
            parts.append(
                f"<a href='tags.html#{tag_anchor}' class='proof-tag' "
                f"data-tag='{html.escape(tag_key, quote=True)}'>"
                f"<b>{escaped}</b></a>"
            )

        if explanation == "validity name":
            if len(entry) > 3:
                rhs_html = wrap_clickable(entry[3])
                # Goal-alias expression follows the unified rule too: it
                # links to the row where the goal is on the left side at
                # the matching namespace, via chapter_ns_map / key_map.
                # No subproof-anchor wrap.
                rhs_norm = re.sub(r'\s+', '', entry[3] or '').lower()
                rhs_ns_norm = re.sub(r'\s+', '', entry[4] if len(entry) > 4 else '').lower()
                rhs_key = (rhs_norm, rhs_ns_norm)
                if chapter_ns_map and rhs_key in chapter_ns_map:
                    rhs_html = (f"<a href='#{chapter_ns_map[rhs_key]}' "
                                f"style='text-decoration:none'>{rhs_html}</a>")
                elif rhs_norm in key_map:
                    rhs_html = (f"<a href='#{key_map[rhs_norm]}' "
                                f"style='text-decoration:none'>{rhs_html}</a>")
                if len(entry) > 4 and entry[4]:
                    rhs_html += _format_validity_tag(entry[4], namespace_anchor_map)
                parts.append(rhs_html)

            content_html = "&nbsp;&nbsp;".join(parts)
            if cursor_index is not None and idx == cursor_index:
                content_html = (
                    f"<span style='background-color:var(--gl-goal-bg); padding:4px; "
                    f"border-radius:4px'>{content_html}</span>"
                )
            if anchor:
                step_badge_html = (
                    f"<a href='#{anchor}' class='step-badge-link' "
                    f"title='Permalink to this step'>{step_badge}</a>"
                )
            else:
                step_badge_html = step_badge
            line_html = f"<div class='proof-line'>{step_badge_html}<span class='proof-line-content'>{content_html}</span></div>"
            lines.append(line_html)
            continue

        for i in range(3, len(entry), 2):
            ing_expr = entry[i]
            ing_val = entry[i + 1] if i + 1 < len(entry) else ""

            norm_ref = re.sub(r'\s+', '', ing_expr).lower()
            ref_html = wrap_clickable(ing_expr)
            # --- NEW: Highlight integration target ---
            is_integration_target = (explanation == "expansion for integration" and i == 3)
            if is_integration_target:
                # 1. Strip away the clickable <span> and data attributes to get plain text
                clean_text = re.sub(r'<[^>]+>', '', ref_html)

                # 2. Re-wrap in magenta span, right-clickable for integration goal explanation
                ref_html = f'<span class="integration-goal-label" style="color:var(--gl-target) !important; font-weight:bold !important;">"{clean_text}"</span>'
            # -----------------------------------------

            # Unified link rule: any expression citation jumps to the row
            # in this chapter where that expression appears as the LEFT
            # SIDE (key) under a matching namespace scope. Lookup priority:
            #   1. chapter_ns_map[(expr_norm, ns_norm)] — exact (expression,
            #      namespace) match across the whole chapter.
            #   2. key_map[expr_norm] — same-subproof local key map (back-
            #      compat fallback for entries not yet rolled into the
            #      chapter-wide map).
            #   3. external_anchor_map[expr_norm] — externally-provided
            #      theorem head minted as a chapter row.
            #   4. plain text — no in-page derivation row exists.
            # Subproof-anchor fallback (jump to subproof title) is gone:
            # the user's directive is "all links point not to subproofs
            # but to where expressions are on left side with their
            # namespace scope". Subproof-title jumps stay only for
            # namespace-tag chips (yellow validity tag) via
            # _format_validity_tag, which doesn't pass through this block.
            ing_ns_norm = re.sub(r'\s+', '', ing_val).lower() if ing_val else ''
            ns_key = (norm_ref, ing_ns_norm)

            if is_integration_target:
                linked_ref = ref_html
            elif chapter_ns_map and ns_key in chapter_ns_map:
                linked_ref = f"<a href='#{chapter_ns_map[ns_key]}' style='text-decoration:none'>{ref_html}</a>"
            elif norm_ref in key_map:
                linked_ref = f"<a href='#{key_map[norm_ref]}' style='text-decoration:none'>{ref_html}</a>"
            elif norm_ref in external_anchor_map:
                linked_ref = f"<a href='#{external_anchor_map[norm_ref]}' style='text-decoration:none'>{ref_html}</a>"
            else:
                linked_ref = ref_html

            if ing_val:
                linked_ref += _format_validity_tag(ing_val, namespace_anchor_map)

            parts.append(linked_ref)

        content_html = "&nbsp;&nbsp;".join(parts)
        if cursor_index is not None and idx == cursor_index:
            content_html = (
                f"<span style='background-color:var(--gl-goal-bg); padding:4px; "
                f"border-radius:4px'>{content_html}</span>"
            )

        # Wrap the step badge in an in-page permalink to the row's
        # own anchor when one exists. Click → URL hash updates to
        # `#<row_id>` so the reader can copy-link to a specific line
        # ('see line 47 of chapter 1209'). The cursor:pointer on
        # .step-badge-link makes the affordance obvious without
        # changing the badge's visual weight.
        if anchor:
            step_badge_html = (
                f"<a href='#{anchor}' class='step-badge-link' "
                f"title='Permalink to this step'>{step_badge}</a>"
            )
        else:
            step_badge_html = step_badge
        # The grey readable caption sits inside the row's content column,
        # directly under the row it explains.
        caption_html = _row_caption_html(entry)
        line_html = (f"<div class='proof-line'>{step_badge_html}<span class='proof-line-content'>"
                     f"{content_html}{caption_html}</span></div>")
        lines.append(line_html)

    return "\n".join(lines)

def extract_args(s: str) -> list[str]:
    # same pattern as before
    pattern = r'(?<=[\[,])([^,\[\]]+)(?=[\],])'
    all_subs = re.findall(pattern, s)
    # remove duplicates while preserving order
    return list(dict.fromkeys(all_subs))


def _diff_highlight_html(result_html: str, result_expr: str, source_expr: str) -> str:
    """Highlight arguments in result_html that differ from source_expr."""
    # Extract bracket structure: name[arg1,arg2,...] from both
    r_match = re.match(r'^!?\((\w+)\[([^\]]+)\]\)$', result_expr.strip())
    s_match = re.match(r'^!?\((\w+)\[([^\]]+)\]\)$', source_expr.strip())
    if not r_match or not s_match:
        return result_html
    if r_match.group(1) != s_match.group(1):
        return result_html  # different expression name
    r_args = r_match.group(2).split(',')
    s_args = s_match.group(2).split(',')
    if len(r_args) != len(s_args):
        return result_html
    # Find changed args
    changed = {r_args[i] for i in range(len(r_args)) if r_args[i] != s_args[i]}
    if not changed:
        return result_html
    # Wrap changed arg text in highlight spans — text segments only (skip HTML tags/attributes)
    # Search for both raw form (v2, i4) and i-stripped form (4) since display strips i-prefix
    for arg in changed:
        variants = [html.escape(arg)]
        if re.match(r'^i\d+$', arg):
            variants.append(html.escape(arg[1:]))  # stripped form: i4 -> 4
        for escaped in variants:
            pat = re.compile(r'(?<=[,\[])(' + re.escape(escaped) + r')(?=[,\]])')
            parts = re.split(r'(<[^>]+>)', result_html)
            result_html = ''.join(
                pat.sub(r'<span class="arg-changed">\1</span>', p) if not p.startswith('<') else p
                for p in parts
            )
    return result_html



def _strip_u_prefixes(token: str) -> str:
    out = token
    while out.startswith("u_"):
        out = out[2:]
    return out


def _looks_like_internal_var_token(token: str) -> bool:
    """
    Internal proof-engine variable-ish names we want to rename to v<number>.
    We intentionally do NOT rename theorem/operator symbols like implication26, in2, and0, id, s, etc.
    """
    if not token:
        return False

    base = _strip_u_prefixes(token)

    if base.isdigit():
        return True
    if re.fullmatch(r"x\d+", base):
        return True
    if re.fullmatch(r"(?:force|int_\d+|int|repl)_lev_\d+_\d+", base):
        return True
    if re.fullmatch(r"it_\d+_lev_\d+_\d+", base):
        return True
    if re.fullmatch(r"it_lev_\d+_\d+", base):
        return True
    if base == "rec":
        return True

    return False


def _normalize_local_expr_vars(expr: str):
    """
    Lightweight local normalization:
    Leaves u_ variables COMPLETELY ALONE (u_30 stays u_30).
    Only normalizes raw non-prefixed variables (7 -> v7, x7 -> vx7).
    """
    args = extract_args(expr)
    replacement_map = {}

    for arg in args:
        if arg.startswith("u_"):
            continue  # Do not touch u_ variables at all

        if arg.isdigit():
            replacement_map[arg] = "v" + arg
        elif arg.startswith("x") and arg[1:].isdigit():
            replacement_map[arg] = "v" + arg

    return replace_keys_in_string(expr, replacement_map)


def rename_expr_peano(expr: str):
    return _normalize_local_expr_vars(expr)


def rename_expr_gauss(expr: str):
    return _normalize_local_expr_vars(expr)


def rename_expr(expr: str):
    # generic fallback
    return _normalize_local_expr_vars(expr)



def extract_natural_numbers_expression(expression: str) -> str:
    """
    Extracts the substring starting at "(NaturalNumbers[" up to and including
    the first ']' that follows. Returns "" if not found or malformed.
    """
    needle = "(NaturalNumbers["
    start = expression.find(needle)
    if start == -1:
        return ""
    end = expression.find("]", start + len(needle))
    if end == -1:
        return ""
    return expression[start:end + 1]


def find_one_arg_name(zero_arg: str, s_arg: str, expr: str) -> str:
    """
    Find the middle argument name in a pattern like:
        (in2[<zero_arg>, <NAME>, <s_arg>])
    where NAME is [A-Za-z0-9_]+. Returns the first match or "" if none.
    """
    pattern = (
            r"\(in2\[\s*"
            + re.escape(zero_arg)
            + r"\s*,\s*([A-Za-z0-9_]+)\s*,\s*"
            + re.escape(s_arg)
            + r"\s*\]\)"
    )
    m = re.search(pattern, expr)
    return m.group(1) if m else ""


def find_identity_arg_name(n_arg: str, expr: str) -> str:
    """Extract id from (identity[<N>, <id>])."""
    pattern = (
        r"\(identity\[\s*"
        + re.escape(n_arg)
        + r"\s*,\s*([A-Za-z0-9_]+)\s*\]\)"
    )
    m = re.search(pattern, expr)
    return m.group(1) if m else ""


def _make_prefixed_token(token: str, mode: str) -> str:
    if not token:
        return token
    if mode == 'as_is':
        return token
    if mode == 'vprefix':
        return token if token.startswith('v') else f"v{token}"
    raise ValueError(f"Unknown prefix mode: {mode}")



def infer_anchor_kind_from_expr(expr: str) -> str:
    if not expr:
        return ""
    if "(AnchorGauss[" in expr:
        return "gauss"
    if "(AnchorPeano[" in expr:
        return "peano"
    return ""


def infer_anchor_kind_from_theorem(theorem_expr: str) -> str:
    kind = infer_anchor_kind_from_expr(theorem_expr)
    if kind:
        return kind

    temp_chain = []
    try:
        disintegrate_implication(theorem_expr, temp_chain)
    except Exception:
        return ""
    for element in temp_chain:
        k = infer_anchor_kind_from_expr(element[0])
        if k:
            return k
    return ""


def build_anchor_symbol_replacement_map(anchor_expr: str, prefix_mode: str = 'as_is', anchor_kind: str = 'auto') -> dict[str, str]:
    """Build replacement map for AnchorPeano / AnchorGauss symbols."""
    replacement_map: dict[str, str] = {}

    detected_kind = infer_anchor_kind_from_expr(anchor_expr)
    if anchor_kind == 'auto':
        anchor_kind = detected_kind

    if anchor_kind not in ('peano', 'gauss'):
        return replacement_map

    if not (anchor_expr.startswith('(AnchorPeano[') or anchor_expr.startswith('(AnchorGauss[')):
        return replacement_map

    expanded_anchor = expand_expr(anchor_expr)
    nn_expr = extract_natural_numbers_expression(expanded_anchor)
    if not nn_expr:
        return replacement_map

    args = expression_utils.get_args(nn_expr)
    if len(args) < 5:
        return replacement_map

    def put_symbol(token: str, symbol: str, include_vx_alias: bool = False):
        if not token:
            return
        key = _make_prefixed_token(token, prefix_mode)
        replacement_map[key] = symbol
        if include_vx_alias and key.startswith('v') and len(key) > 1:
            replacement_map['vx' + key[1:]] = symbol

    # NaturalNumbers[N,i0,s,+,*]
    put_symbol(args[0], 'N')
    put_symbol(args[1], '0', include_vx_alias=True)
    put_symbol(args[2], 's')
    put_symbol(args[3], '+')
    put_symbol(args[4], '*')

    zero_arg_name = args[1]
    s_arg_name = args[2]
    one_arg_name = find_one_arg_name(zero_arg_name, s_arg_name, expanded_anchor)
    put_symbol(one_arg_name, '1', include_vx_alias=True)

    if anchor_kind == 'gauss':
        two_arg_name = find_one_arg_name(one_arg_name, s_arg_name, expanded_anchor) if one_arg_name else ''
        put_symbol(two_arg_name, '2', include_vx_alias=True)

        id_arg_name = find_identity_arg_name(args[0], expanded_anchor)
        put_symbol(id_arg_name, 'id')

    return replacement_map


def _extract_anchor_subexpressions(expr: str) -> list[str]:
    """Return all nested (AnchorPeano[...]) / (AnchorGauss[...]) subexpressions found in expr."""
    if not expr:
        return []

    starts = ["(AnchorPeano[", "(AnchorGauss["]
    out = []
    i = 0
    n = len(expr)
    while i < n:
        start = -1
        start_token = ""
        for token in starts:
            j = expr.find(token, i)
            if j != -1 and (start == -1 or j < start):
                start = j
                start_token = token
        if start == -1:
            break

        # Scan until the matching ')' of the anchor expression.
        bracket_depth = 0
        k = start
        end = -1
        while k < n:
            ch = expr[k]
            if ch == '[':
                bracket_depth += 1
            elif ch == ']':
                bracket_depth = max(0, bracket_depth - 1)
            elif ch == ')' and bracket_depth == 0:
                end = k
                break
            k += 1

        if end != -1:
            out.append(expr[start:end + 1])
            i = end + 1
        else:
            # malformed; avoid infinite loop
            i = start + len(start_token)

    # preserve order, remove duplicates
    return list(dict.fromkeys(out))


def _contains_u_prefixed_token(expr: str) -> bool:
    for tok in extract_args(expr):
        if tok.startswith('u_'):
            return True
    return False


def _extract_arg_tokens_in_order(s: str) -> list[str]:
    """Return all bracket/comma-delimited argument tokens in order (with duplicates)."""
    return re.findall(r'(?<=[\[,])([^,\[\]]+)(?=[\],])', s)


def _collect_anchor_maps_from_expr(expr: str, prefix_mode: str, default_kind: str = "") -> dict[str, str]:
    out = {}
    if not expr:
        return out

    for anchor_expr in _extract_anchor_subexpressions(expr):
        k = infer_anchor_kind_from_expr(anchor_expr) or default_kind
        out.update(build_anchor_symbol_replacement_map(anchor_expr, prefix_mode=prefix_mode, anchor_kind=k or 'auto'))

    return out


def _infer_row_anchor_kind(row: list[str], default_kind: str = "") -> str:
    if not row:
        return default_kind
    expr_indices = [0] + list(range(3, len(row), 2))
    for idx in expr_indices:
        if idx < len(row):
            k = infer_anchor_kind_from_expr(row[idx])
            if k:
                return k
    return default_kind



def rename_theorem(theorem: str):
    """
    Renaming is now fully handled by process_proof_graphs.py.
    Just return the theorem exactly as it is.
    """
    return theorem


def clean_stack(stack: list[list[str]]):
    """
    Cleans a stack of strings by normalizing internal variable keys.
    Leaves u_ variables completely untouched.
    """
    # Avoid mutating while iterating
    stack[:] = [entry for entry in stack if not (len(entry) > 2 and entry[2] == "anchor handling")]

    token_pattern = re.compile(r'(?<=[\[,])([^,\[\]]+)(?=[\],])')

    def expr_cols(row):
        return [0] + list(range(3, len(row), 2))

    def validity_cols(row):
        cols = []
        if len(row) > 1:
            cols.append(1)
        cols.extend(range(4, len(row), 2))
        return cols

    def all_norm_cols(row):
        return expr_cols(row) + validity_cols(row)

    # Gather already-existing v/V<number> indices so numbering doesn't
    # collide. Index space is shared between v (digit) and V (set) so a
    # freshly-minted v{N} cannot be visually confused with an existing
    # V{N} in the same chapter.
    max_index = -1
    for row in stack:
        if not row:
            continue
        for idx in all_norm_cols(row):
            if idx >= len(row):
                continue
            for token in token_pattern.findall(row[idx]):
                m = re.fullmatch(r'[vV](\d+)', token)
                if m:
                    max_index = max(max_index, int(m.group(1)))

    next_idx = max_index + 1 if max_index >= 0 else 0
    replacement_map: dict[str, str] = {}

    for row in stack:
        if not row:
            continue
        for idx in all_norm_cols(row):
            if idx >= len(row):
                continue
            for token in token_pattern.findall(row[idx]):
                if token in replacement_map:
                    continue
                if token.startswith("u_"):
                    continue  # LEAVE u_ variables ALONE!

                if _looks_like_internal_var_token(token):
                    if token.isdigit():
                        replacement_map[token] = f"v{token}"
                    elif token.startswith("x") and token[1:].isdigit():
                        replacement_map[token] = f"v{token[1:]}"
                    else:
                        replacement_map[token] = f"v{next_idx}"
                        next_idx += 1

    # Apply replacements to all normalized columns (expr + validity)
    for i, row in enumerate(stack):
        if not row:
            continue
        for j in all_norm_cols(row):
            if j < len(row):
                stack[i][j] = replace_keys_in_string(row[j], replacement_map)


def rename_stack(stack: list[list[str]], theorem: str):
    """
    Renaming is now fully handled by process_proof_graphs.py.
    We do absolutely nothing here to preserve the processed variables.
    """
    pass


def read_stack(file_path: str, proof_part: str):
    """
    Load a raw stack written by generate_raw_proof_graph.

    - File is resolved under PROJECT_ROOT / 'files/processed_proof_graph'
    - Filename is the same sanitizer as in generate_raw_proof_graph:
        f"{_safe(theorem)}__{proof_part}.txt"
    - Each line corresponds to one list[str], items separated by tabs.
    - Empty line => empty list.
    """
    # same sanitizer used when writing

    render_cache = _RENDER_CACHE.get()
    cache_key = str(file_path)
    if render_cache is not None and cache_key in render_cache["chapters"]:
        return [list(row) for row in render_cache["chapters"][cache_key]]
    stack = []
    with open(file_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            # remove newline only (preserve all other characters)
            line = raw_line.rstrip("\n")
            # handle Windows CR if present
            if line.endswith("\r"):
                line = line[:-1]
            if line == "":
                stack.append([])  # empty list was written as a blank line
            else:
                stack.append(line.split("\t"))  # items were joined with tabs
    if render_cache is not None:
        render_cache["chapters"][cache_key] = tuple(tuple(row) for row in stack)
    return stack


def _norm_expr(expr: str) -> str:
    return re.sub(r"\s+", "", expr or "").lower()


def _row_validity_values(row: list[str]) -> list[str]:
    vals = []
    if not row:
        return vals
    if len(row) > 1 and row[1]:
        vals.append(row[1])
    for i in range(4, len(row), 2):
        if row[i]:
            vals.append(row[i])
    return vals





def _row_contains_rhs_expr(row: list[str], expr: str) -> bool:
    target = _norm_expr(expr)
    for i in range(3, len(row), 2):
        if _norm_expr(row[i]) == target:
            return True
    return False


def _strip_integration_goal(expr: str) -> str:
    s = (expr or "").strip()
    # Strip any accidental quotation marks so the suffix can be correctly removed
    if s.startswith('"') and s.endswith('"'):
        s = s[1:-1]
    suffix = "_integration_goal"
    return s[:-len(suffix)] if s.endswith(suffix) else s

def _structural_match(expr1: str, expr2: str) -> bool:
    """Check if two expressions match structurally, ignoring exact 'v' variable numbers."""
    def norm_vars(s: str) -> str:
        return re.sub(r'v\d+', 'vX', s)
    return norm_vars(expr1) == norm_vars(expr2)

def _row_contains_rhs_expr_integration_aware(row: list[str], expr: str) -> bool:
    target = _norm_expr(_strip_integration_goal(expr))
    for i in range(3, len(row), 2):
        cell_expr = _norm_expr(_strip_integration_goal(row[i]))
        # Check for exact match or structural match (v12 vs v17)
        if cell_expr == target or _structural_match(cell_expr, target):
            return True
    return False


def _scope_title_info(child_ns: str, parent_ns: str,
                      validity_name_row: list[str] | None,
                      or_convergence_map: dict | None = None) -> dict:
    """
    Derive title + metadata for a subproof scope.

    If a `validity name` row in the parent scope refers to this child_ns,
    use its row[0] (implication expression) and row[3] (goal alias) as
    the rich metadata.

    Otherwise infer from the namespace's payload pattern:
      - `_boundary_orint_<OR>_(<disjunct>)` \u2192 OR-introduction subproof.
        Goal alias is the disjunct itself \u2014 the branch's job is to
        prove that disjunct.
      - `_boundary_ordis_<OR>_(<disjunct>)` \u2192 OR-elimination branch.
        Goal alias is the convergence target (the conclusion every
        branch must derive), looked up by OR expression in
        or_convergence_map. Refuted cases receive their contradiction
        metadata from the actual parent proof in _partition_stack_subproofs.
      - anything else \u2192 fall back to raw payload as title.
    """
    sep = "_boundary_"
    info: dict = {"kind": "subproof"}

    payload = (child_ns[len(parent_ns) + len(sep):]
               if child_ns.startswith(parent_ns + sep)
               else child_ns)
    info["payload"] = payload

    if validity_name_row is not None:
        implication_expr = validity_name_row[0]
        goal_expr = validity_name_row[3] if len(validity_name_row) > 3 else None
        info["implication_expr"] = implication_expr
        info["implication_norm"] = _norm_expr(implication_expr)
        info["goal_expr"] = goal_expr
        info["goal_norm"] = _norm_expr(goal_expr) if goal_expr else None
        info["title_html"] = wrap_clickable(implication_expr)
        info["kind"] = "subproof"
        return info

    m = re.match(r'^orint_(\(or\d+\[[^\]]*\]\))_\((.+)\)$', payload)
    if m:
        info["or_expr"] = m.group(1)
        info["disjunct"] = m.group(2)
        # OR-introduction: the branch's goal IS the disjunct.
        info["goal_expr"] = m.group(2)
        info["goal_norm"] = _norm_expr(m.group(2))
        info["kind"] = "_orint_ subproof"
        info["title_html"] = (
            f"OR-introduction subproof \u2014 prove "
            f"{wrap_clickable(m.group(2))} "
            f"(of OR {wrap_clickable(m.group(1))})"
        )
        return info

    # The case-split source is a compiled OR, or a negated conjunction
    # integration premise split into its complemented conjuncts
    # (D-460).
    m = re.match(r'^ordis_(\(or\d+\[[^\]]*\]\)|!\([^\[\]()!&>]+\[[^\]]*\]\))_\((.+)\)$',
                 payload)
    if m:
        info["or_expr"] = m.group(1)
        info["disjunct"] = m.group(2)
        info["kind"] = "_ordis_ branch"
        # OR-elimination: the branch's goal is the convergence target
        # (the conclusion every branch must derive). Look it up by OR
        # expression in or_convergence_map (built from `or convergence`
        # rows in the parent scope). Refuted cases are identified separately
        # from their recorded parent-scope contradiction.
        or_norm = _norm_expr(m.group(1))
        converged = (or_convergence_map or {}).get(or_norm)
        info["goal_expr"] = converged
        info["goal_norm"] = _norm_expr(info["goal_expr"]) if info["goal_expr"] else None
        info["title_html"] = (
            f"OR-elimination branch \u2014 case "
            f"{wrap_clickable(m.group(2))} "
            f"(of OR {wrap_clickable(m.group(1))})"
        )
        return info

    info["title_html"] = html.escape(payload)
    return info


def _partition_stack_subproofs(stack: list[list[str]],
                               scope_ns: str = "main"):
    """
    Recursively partition a stack by primary namespace.

    Returns `(own_rows, subproofs)` where:

    - `own_rows` \u2014 rows whose primary namespace (`row[1]`) equals
      `scope_ns`.
    - `subproofs` \u2014 list of dicts, one per immediate-child namespace,
      each with:
        - `ns`             child namespace
        - `title_info`     derived from `_scope_title_info`
        - `display_stack`  rows directly at this child scope
        - `nested_subproofs`  recursive sub-sub-proofs
        - other legacy fields preserved for backward-compat callers

    Top-level scope is `"main"` by default. Recursion handles arbitrary
    depth \u2014 sub-sub-\u2026-proofs render as collapsed cards inside their
    parent's collapsed card (matryoshka).
    """
    sep = "_boundary_"
    parent_prefix = scope_ns + sep
    scope_norm = _norm_expr(scope_ns)

    # Validity-name rows AT THIS SCOPE provide rich metadata for child
    # subproofs (the row's row[4] is the child's namespace; row[0] is
    # the implication expression; row[3] is the goal alias).
    vn_lookup: dict[str, list[str]] = {}
    for row in stack:
        if not row or len(row) < 5:
            continue
        if _norm_expr(row[1]) != scope_norm:
            continue
        if row[2] != "validity name":
            continue
        vn_lookup[_norm_expr(row[4])] = row

    # `or convergence` rows AT THIS SCOPE map an OR expression to the
    # converged conclusion C: every ordis branch under that OR has C
    # as its goal. The row shape is
    #   [C, scope, "or convergence", or_expr, scope, ...].
    # Used by _scope_title_info to fill in the goal alias for ordis
    # branches.
    or_convergence_map: dict[str, str] = {}
    for row in stack:
        if not row or len(row) < 4:
            continue
        if _norm_expr(row[1]) != scope_norm:
            continue
        if row[2] != "or convergence":
            continue
        or_convergence_map[_norm_expr(row[3])] = row[0]

    # Group rows by immediate-child namespace; collect own_rows.
    own_rows: list[list[str]] = []
    child_groups: dict[str, list[list[str]]] = {}
    child_order: list[str] = []  # preserve first-appearance order
    for row in stack:
        if not row or len(row) < 2:
            continue
        primary_ns = row[1] or ""
        if _norm_expr(primary_ns) == scope_norm:
            own_rows.append(copy.deepcopy(row))
            continue
        if not primary_ns.startswith(parent_prefix):
            # Row is in an unrelated scope \u2014 happens when called on a
            # sub-stack that contains rows at scopes outside the parent
            # subtree. Skip silently.
            continue
        rest = primary_ns[len(parent_prefix):]
        next_boundary = rest.find(sep)
        child_segment = rest if next_boundary < 0 else rest[:next_boundary]
        child_ns = parent_prefix + child_segment
        if child_ns not in child_groups:
            child_groups[child_ns] = []
            child_order.append(child_ns)
        child_groups[child_ns].append(copy.deepcopy(row))

    # Recurse into each child scope.
    subproofs: list[dict] = []
    for child_ns in child_order:
        child_rows = child_groups[child_ns]
        nested_own, nested_subs = _partition_stack_subproofs(child_rows, child_ns)
        title_info = _scope_title_info(
            child_ns, scope_ns, vn_lookup.get(_norm_expr(child_ns)),
            or_convergence_map=or_convergence_map)
        if title_info["kind"] == "_orint_ subproof":
            title_info["case_expressions"] = list(dict.fromkeys(
                row[0] for row in reversed(nested_own)
                if row[2] == "or branch assumption"))
        elif title_info["kind"] == "_ordis_ branch":
            case = title_info["disjunct"]
            title_info["case_expressions"] = [case]
            if title_info["goal_expr"] is None:
                negated_case = case[1:] if case.startswith("!") else "!" + case
                refutations = [row for row in stack
                               if row[1] == scope_ns and row[2] == "contradiction"
                               and _norm_expr(row[0]) == _norm_expr(negated_case)
                               and child_ns in row[4::2]]
                assert refutations, f"OR case has neither convergence nor refutation: {child_ns}"
                title_info["goal_kind"] = "contradiction"
                title_info["refuted_case"] = refutations[0][0]
                title_info["refutation_namespace"] = scope_ns
        sp = {
            "ns": child_ns,
            "namespace_expr": child_ns,
            "namespace_norm": _norm_expr(child_ns),
            "title_info": title_info,
            "display_stack": nested_own,
            "nested_subproofs": nested_subs,
            # Backward-compat aliases used by the renderer
            "implication_expr": title_info.get("implication_expr"),
            "implication_norm": title_info.get("implication_norm"),
            "goal_expr": title_info.get("goal_expr"),
            "goal_norm": title_info.get("goal_norm"),
        }
        subproofs.append(sp)

    return own_rows, subproofs


def _flatten_conjunction(expr: str) -> list[str]:
    """Walk a right-associative MPL conjunction `(&E1(&E2 … (&E_{n-1}E_n)))`
    and return [E1, E2, …, E_{n-1}, E_n]. For a non-conjunctive
    expression, returns [expr] unchanged. Used to recover the
    individual premises of a multi-premise implication after
    disintegration peels its outer `(>[bound]premise body)` shell —
    the prover compiles multi-premise implications with a single
    nested `(&…)` premise (`build_gl_binary_map._gl_make_conjunction`),
    so disintegration alone yields one giant conjunctive premise that
    we want to display as a flat list.
    """
    if not expr.startswith('(&'):
        return [expr]
    out: list[str] = []
    s = expr
    while s.startswith('(&'):
        # Left operand of the conjunction starts at offset 2 (right
        # after '(&'); right operand follows it.
        if len(s) <= 2 or s[2] != '(':
            break
        left_end = _find_matching_paren_local(s, 2)
        if left_end < 0:
            break
        out.append(s[2:left_end + 1])
        right_start = left_end + 1
        if right_start >= len(s) or s[right_start] != '(':
            break
        right_end = _find_matching_paren_local(s, right_start)
        if right_end < 0:
            break
        s = s[right_start:right_end + 1]
        if not s.startswith('(&'):
            out.append(s)
            break
    return out


def _expand_or_to_mpl(or_expr: str) -> str:
    """Expand an `(or<N>[arg1,...,argK])` expression into its full
    De-Morgan MPL form using the global _gl_binary_map_for_render.
    Returns the original expression unchanged when the operator is
    not in the map.

    The map's `mpl` field stores the expanded form with placeholders
    `x1, x2, ...` (one per signature slot, ordered as in the GL
    binary's signature). We substitute those placeholders for the
    actual argument tokens of `or_expr`.
    """
    m = re.match(r'^\(([A-Za-z_][A-Za-z0-9_]*)\[([^\]]*)\]\)$', or_expr.strip())
    if not m:
        return or_expr
    core = m.group(1)
    actual_args = [a.strip() for a in m.group(2).split(',') if a.strip()]
    entry = _gl_binary_map_for_render.get(core)
    if not entry:
        return or_expr
    mpl_template = entry.get('mpl', '') or ''
    if not mpl_template:
        return or_expr
    var_map = {f'x{i + 1}': actual_args[i] for i in range(len(actual_args))}
    return _gl_rename_vars(mpl_template, var_map)


def _subproof_explanation(title_info: dict) -> str:
    """Return the textual explanation paragraph rendered under the
    title of a subproof card — implication-introduction, OR-introduction,
    or OR-elimination. For any other kind returns the empty string.

    The wording follows Hilbert-style classical-logic prose:
    declarative, passive, no narrative voice.
    """
    kind = title_info.get("kind", "")
    or_expr = title_info.get("or_expr")
    disjunct = title_info.get("disjunct")
    goal_expr = title_info.get("goal_expr")
    implication_expr = title_info.get("implication_expr")

    # Implication-introduction: the validity-name-row branch of
    # _scope_title_info sets implication_expr (the compact symbol of
    # the implication being introduced) but no or_expr / disjunct.
    if implication_expr and not or_expr:
        impl_expanded = _expand_or_to_mpl(implication_expr)
        # Disintegrate the expanded MPL to recover individual premises
        # and the head. Falls back to the compact form when expansion
        # or disintegration fails (operator missing from gl_binary_map,
        # malformed MPL).
        premises_list: list[str] = []
        head_expr: str | None = None
        if impl_expanded and impl_expanded != implication_expr:
            try:
                expr_iter = impl_expanded
                while expr_iter.startswith('(>['):
                    bracket_close = expr_iter.index(']', 3)
                    premise_start = bracket_close + 1
                    if premise_start >= len(expr_iter) or expr_iter[premise_start] != '(':
                        break
                    premise_end = _find_matching_paren_local(expr_iter, premise_start)
                    if premise_end < 0:
                        break
                    body_start = premise_end + 1
                    if body_start >= len(expr_iter) or expr_iter[body_start] != '(':
                        break
                    body_end = _find_matching_paren_local(expr_iter, body_start)
                    if body_end < 0:
                        break
                    premises_list.append(expr_iter[premise_start:premise_end + 1])
                    expr_iter = expr_iter[body_start:body_end + 1]
                head_expr = expr_iter if premises_list else None
            except (ValueError, IndexError):
                premises_list = []
                head_expr = None
        impl_expanded_html = (
            f" <span class='or-expansion'>[{wrap_clickable(impl_expanded)}]</span>"
            if impl_expanded and impl_expanded != implication_expr else ""
        )
        if premises_list and head_expr:
            # Flatten any single conjunctive premise into its individual
            # conjuncts so the reader sees each assumption separately
            # (`build_gl_binary_map` packs multi-premise implications
            # into one nested `(&…)` premise via _gl_make_conjunction).
            flat_premises: list[str] = []
            for p in premises_list:
                flat_premises.extend(_flatten_conjunction(p))
            premises_html = ", ".join(wrap_clickable(p) for p in flat_premises)
            head_html = wrap_clickable(head_expr)
            premise_clause = (
                f"the premises {premises_html} are assumed and the head "
                f"{head_html} is sought"
            )
        else:
            premise_clause = (
                "the premises (the conjunctive antecedent) are assumed "
                "and the head (the consequent) is sought"
            )
        # The symbolic clause uses sequent-calculus shorthand
        # `A, B ⊢ C` rather than the deduction-theorem prose form;
        # the underlying tautology equivalence (A ∧ B → C iff A ∧ B ⊢ C
        # in classical propositional logic) is the rule this branch
        # exercises but is left to the reader as background.
        return (
            "<div class='subproof-explanation'>"
            "<b>Implication-introduction.</b> An implication is "
            "established when its head has been derived under the "
            "assumption of its premises. "
            "<span class='symbolic-example'>Symbolically: "
            "<i>A, B &#x22A2; C</i>. Assume <i>A</i> and <i>B</i> and "
            "prove <i>C</i>.</span> "
            f"This subproof names the implication "
            f"{wrap_clickable(implication_expr)}{impl_expanded_html}; "
            f"within its scope {premise_clause}. On closure, "
            f"{wrap_clickable(implication_expr)} is emitted at the "
            "parent scope."
            "</div>"
        )

    if not or_expr or not disjunct:
        return ""
    or_expanded = _expand_or_to_mpl(or_expr)
    or_expanded_html = (
        f" <span class='or-expansion'>[{wrap_clickable(or_expanded)}]</span>"
        if or_expanded and or_expanded != or_expr else ""
    )
    if kind == "_orint_ subproof":
        assumptions = title_info.get("case_expressions", [])
        case_prose = (
            "The case assumes the negated premises "
            + ", ".join(wrap_clickable(expression) for expression in assumptions)
            + ". " if assumptions else "No additional branch assumption is recorded. ")
        return (
            "<div class='subproof-explanation'>"
            "<b>OR-introduction.</b> A disjunction is established when "
            "one of its disjuncts has been derived under the assumption "
            "of the falsity of the others. "
            "<span class='symbolic-example'>Symbolically: from "
            "<i>¬A &#x22A2; B</i> follows <i>A &#x2228; B</i>.</span> "
            + case_prose
            + "This branch proves the disjunct "
            f"{wrap_clickable(disjunct)} of "
            f"{wrap_clickable(or_expr)}{or_expanded_html}. On closure the "
            "disjunction is asserted at the parent scope."
            "</div>"
        )
    if kind == "_ordis_ branch" and title_info.get("goal_kind") == "contradiction":
        return (
            "<div class='subproof-explanation'><b>OR-disintegration: refuted case.</b> "
            f"Assume {wrap_clickable(disjunct)} from {wrap_clickable(or_expr)}. "
            "The proof derives a contradiction under that assumption. "
            f"This refutes the case and establishes {wrap_clickable(title_info['refuted_case'])} "
            "at the parent scope.</div>")
    if kind == "_ordis_ branch":
        goal_html = (wrap_clickable(goal_expr) if goal_expr
                     else "<em>convergence target</em>")
        return (
            "<div class='subproof-explanation'>"
            "<b>OR-elimination.</b> From a disjunction in force at a "
            "scope, a conclusion may be discharged at that scope "
            "provided it has been derived in every branch under the "
            "corresponding disjunct's assumption. "
            "<span class='symbolic-example'>Symbolically: from "
            "<i>A &#x2228; B</i>, <i>A &#x22A2; C</i>, and "
            "<i>B &#x22A2; C</i> follows <i>C</i> at the scope where "
            "<i>A &#x2228; B</i> holds.</span> "
            "This branch assumes the disjunct "
            f"{wrap_clickable(disjunct)} of "
            f"{wrap_clickable(or_expr)}{or_expanded_html}; its goal is "
            f"the convergence target {goal_html} required uniformly "
            "across the sibling branches. Once each branch has reached "
            "that target, the target is asserted at the parent scope."
            "</div>"
        )
    return ""


def _render_scope_contents(own_rows, subproofs, namespace, prefix, depth,
                           scope_items, external_anchor_map, global_counter,
                           global_total, namespace_anchor_map, chapter_ns_map,
                           goal_key_norm=None):
    """
    @brief Interleave ordinary rows with collapsed child cards in stored order.

    @details Row identifiers retain their scope-local indices while the common
    display counter advances through the recursively expanded proof. Highlight
    only the scope's achieved goal; a one-row rendering call is not a new proof.

    @param own_rows Reverse-stored rows at this exact namespace.
    @param subproofs Immediate child card records with assigned anchor prefixes.
    @param namespace Exact scope whose ordered items are being rendered.
    @param prefix Row-anchor prefix for this scope.
    @param depth Depth of child cards below the main scope.
    @param scope_items Validated namespace-to-row/card order from processed rows.
    @param external_anchor_map Existing expression-to-external-anchor map.
    @param global_counter Mutable expanded visible-row counter.
    @param global_total Total visible rows in this physical proof section.
    @param namespace_anchor_map Existing namespace title anchors.
    @param chapter_ns_map Complete scoped expression-to-row-anchor map.
    @param goal_key_norm Optional normalized scope conclusion for highlighting.
    @return HTML whose expanded row order equals the processed artifact.
    @invariant Collapse state changes visibility only, never scope or row order.
    """
    reader = list(reversed(own_rows))
    children = {sp['ns']:sp for sp in subproofs}
    assert len(children) == len(subproofs)
    local = {_norm_expr(row[0]):f'{prefix}-entry{i}' for i,row in enumerate(reader)
             if row and 'theorem' not in row}
    highlighted = next((i for i in range(len(reader)-1,-1,-1)
                        if reader[i] and 'theorem' not in reader[i]
                        and (not goal_key_norm or _norm_expr(reader[i][0]) == goal_key_norm)), -1)
    blocks = []
    for kind,key in scope_items[namespace]:
        if kind == 'row':
            blocks.append(format_stack_entries(
                [reader[key]], prefix=prefix, reverse_entries=False,
                entry_indices=[key], highlight_position=highlighted,
                local_anchor_map=local, external_anchor_map=external_anchor_map,
                global_counter=global_counter, global_total=global_total,
                namespace_anchor_map=namespace_anchor_map, chapter_ns_map=chapter_ns_map))
        else:
            assert kind == 'scope' and key in children
            child = children[key]
            blocks.append(_render_subproof_card(
                child, child['_anchor_prefix'], depth,
                external_anchor_map=external_anchor_map, global_counter=global_counter,
                global_total=global_total, namespace_anchor_map=namespace_anchor_map,
                chapter_ns_map=chapter_ns_map, scope_items=scope_items))
    return ''.join(blocks)


def _render_subproof_card(sp: dict, prefix: str, depth: int,
                          external_anchor_map: dict | None = None,
                          global_counter: list | None = None,
                          global_total: int | None = None,
                          namespace_anchor_map: dict | None = None,
                          chapter_ns_map: dict | None = None,
                          scope_items: dict | None = None) -> str:
    """
    @brief Render one intact collapsed subproof with recursive child cards.

    @details Main-book cards follow the expanded processed order. Other books
    retain separate nested sections. Shared row counters and anchor mappings
    remain valid through every recursion level.

    @param sp Parsed child-scope record and its own rows.
    @param prefix Unique card-anchor prefix.
    @param depth Nesting depth below main.
    @param external_anchor_map Nonlocal expression anchors.
    @param global_counter Shared expanded visible-row counter.
    @param global_total Expanded visible-row count for this proof section.
    @param namespace_anchor_map Existing scope-title anchors.
    @param chapter_ns_map Complete scoped expression-to-row-anchor mapping.
    @param scope_items Hierarchical stored order, or None for separate sections.
    @return One complete collapsed card including its nested proof bodies.
    @invariant No child assumption or proof row leaves its owning card.

    Render one subproof card (collapsed by default), recursing into
    nested sub-sub-proofs to produce the matryoshka structure.
    `depth` is 1 for top-level subproofs of main, 2 for sub-sub, etc.

    `global_counter` and `global_total` thread chapter-wide line
    numbering through the body and every nested card. See
    format_stack_entries for the contract.
    """
    blocks: list[str] = []
    anchor_id = f"{prefix}-anchor"
    title_html = sp["title_info"]["title_html"]
    kind_label = sp["title_info"]["kind"]

    blocks.append(
        f"<div class='subproof-card collapsed subproof-depth-{depth}'>")
    blocks.append(
        f"<div class='subproof-title'><span class='subproof-toggle'>\u25BC</span>"
        f"<span id='{anchor_id}'>{title_html}</span> "
        f"<span class='subproof-label'>{kind_label}</span></div>")
    blocks.append("<div class='subproof-body'>")
    title_reading = _subproof_title_reading(sp["title_info"])
    if title_reading:
        blocks.append(f"<div class='readable-grey subproof-reads'>{title_reading}</div>")

    # Textual explanation under the title \u2014 for OR-introduction and
    # OR-elimination branches only. Empty string for plain implication
    # subproofs and other kinds; falsy values are dropped from the
    # blocks list further down by the join.
    explanation_html = _subproof_explanation(sp["title_info"])
    if explanation_html:
        blocks.append(explanation_html)

    goal_expr = sp["title_info"].get("goal_expr")
    if goal_expr:
        goal_plain = (goal_expr[:-len("_integration_goal")]
                      if goal_expr.endswith("_integration_goal") else goal_expr)
        goal_reading = _readable_form_for_render.formula(goal_plain)
        blocks.append(
            f"<div class='subproof-meta'>Goal alias: "
            f"<span class='subproof-goal'>{html.escape(goal_expr)}</span>"
            f"{_format_validity_tag(sp['ns'])}"
            f"<span class='readable-grey'> &nbsp;{goal_reading}</span></div>")
    else:
        blocks.append(
            f"<div class='subproof-meta'>"
            f"{_format_validity_tag(sp['ns'])}</div>")

    if scope_items is not None:
        blocks.append(_render_scope_contents(
            sp['display_stack'], sp['nested_subproofs'], sp['ns'], f'{prefix}body',
            depth + 1, scope_items, external_anchor_map, global_counter, global_total,
            namespace_anchor_map, chapter_ns_map, sp['title_info'].get('goal_norm')))
    else:
        if sp["display_stack"]:
            blocks.append(
                format_stack_entries(
                    sp["display_stack"],
                    prefix=f"{prefix}body",
                    # reverse_entries defaults to True so subproof rows
                    # render in the same earliest->latest order as the
                    # main proof body. The earlier explicit `False` here
                    # produced latest->earliest order inside subproof
                    # cards, which read inverted from the surrounding
                    # main stack.
                    goal_key_norm=sp["title_info"].get("goal_norm"),
                    external_anchor_map=external_anchor_map,
                    global_counter=global_counter,
                    global_total=global_total,
                    namespace_anchor_map=namespace_anchor_map,
                    chapter_ns_map=chapter_ns_map,
                ))
        elif not sp["nested_subproofs"]:
            blocks.append("<div class='proof-empty'>No subproof steps detected.</div>")

        # Recurse: nested sub-sub-proofs render at the END of the body,
        # each as its own collapsed card. Depth increments per level. The
        # nested card's anchor prefix is sourced from sp["_anchor_prefix"]
        # (assigned by render_stack_with_subproofs's _walk_assign_anchors)
        # so the rendered id matches the namespace_anchor_map exactly.
        for nested in sp["nested_subproofs"]:
            nested_prefix = nested["_anchor_prefix"]
            blocks.append(
                _render_subproof_card(nested, nested_prefix, depth + 1,
                                      external_anchor_map=external_anchor_map,
                                      global_counter=global_counter,
                                      global_total=global_total,
                                      namespace_anchor_map=namespace_anchor_map,
                                      chapter_ns_map=chapter_ns_map))

    blocks.append("</div>")  # close subproof-body
    blocks.append("</div>")  # close subproof-card
    return "".join(blocks)


def _count_visible_in_subproof_tree(subproofs: list[dict]) -> int:
    """Recursively count visible (non-theorem, non-empty) rows across
    every subproof's display_stack and every nested sub-sub-proof.
    Used by render_stack_with_subproofs to compute the chapter-wide
    total for common line numbering.
    """
    total = 0
    for sp in subproofs:
        for entry in sp.get("display_stack", []):
            if entry and 'theorem' not in entry:
                total += 1
        total += _count_visible_in_subproof_tree(sp.get("nested_subproofs", []))
    return total


def render_stack_with_subproofs(stack: list[list[str]], prefix: str = "",
                                main_goal: str | None = None, hierarchical: bool = False) -> str:
    """
    @brief Render a complete proof section with preserved nested scope ownership.

    @details Supported-book hierarchical mode follows the reverse-stored expanded
    artifact exactly. Legacy mode retains separate subproof sections. Collapse
    controls affect visibility, never the recorded row order or total.

    @param stack Complete processed proof rows with the main goal first.
    @param prefix Unique induction-section prefix, or empty for a single proof.
    @param main_goal Optional theorem or typing target for the goal label.
    @param hierarchical Whether to interleave intact cards in processed order.
    @return Complete proof-section HTML with stable links and line numbering.
    @invariant Hierarchical input must contain each scope as one contiguous block.

    Render a proof with collapsed recursive subproof cards. Supported-book
    hierarchical mode interleaves cards according to processed row order;
    other books retain their separate subproof sections.

    `main_goal`, when given, is used to render a "Goal alias: ..."
    meta-line at the top of the main proof section — same format as
    the existing meta-line on every subproof card. The convention is
    chapter-wide: every section (main, every subproof, every nested
    sub-sub-proof) carries an explicit Goal alias line so the reader
    sees the section's target up front.
    """
    if hierarchical:
        import sys
        from proof_reader_html import render_reader_proof
        chapter_theorems = [row[0] for row in stack if row[2] == "theorem"]
        known = list(_known_theorems_for_reader) + chapter_theorems
        known_keys = _known_theorem_keys_for_reader | theorem_alpha_keys(chapter_theorems)
        return render_reader_proof(stack, prefix, sys.modules[__name__], known, known_keys)
    main_stack, subproofs = _partition_stack_subproofs(stack, scope_ns="main")
    scope_items = hierarchical_scope_items(stack) if hierarchical else None

    # Build cross-reference anchor map: each subproof (any depth) gets a
    # unique anchor ID. The map is keyed by normalised implication
    # expression (when available) so main-stack rows that mention the
    # implication can link to the subproof card.
    subproof_anchor_map: dict[str, str] = {}
    # Raw-namespace -> anchor map. Lets validity-tag rendering link a
    # `(<namespace>)` cell in any chapter row to the corresponding
    # subproof card (any depth). Click -> popup_script's ns-jump
    # handler uncollapses the target card + every collapsed ancestor,
    # then the browser scrolls the anchor to the top of the viewport.
    namespace_anchor_map: dict[str, str] = {}

    def _walk_assign_anchors(sps: list[dict], parent_prefix: str = "") -> None:
        # Mirrors the renderer's id-naming convention exactly:
        #   top-level subproof k        -> "<section>sp<k>"
        #   nested subproof k of <P>    -> "<P>n<k>"
        #   nested-nested subproof k    -> "<P>n<k>n<m>"  (etc.)
        # The maps below are kept for namespace-tag chip clicks (yellow
        # validity tags) which jump to the subproof title; expression
        # citations (ingredients, rules) use the unified (expression,
        # namespace) -> entry-id rule via chapter_ns_map below — see the
        # priority block in format_stack_entries.
        for j, sp in enumerate(sps, start=1):
            if parent_prefix == "":
                anchor_prefix = f"{prefix}sp{j}"
            else:
                anchor_prefix = f"{parent_prefix}n{j}"
            sp["_anchor_prefix"] = anchor_prefix
            anchor_target = f"{anchor_prefix}-anchor"
            impl_norm = sp["title_info"].get("implication_norm")
            if impl_norm:
                subproof_anchor_map[impl_norm] = anchor_target
            ns_raw = sp.get("ns")
            if ns_raw:
                namespace_anchor_map[ns_raw] = anchor_target
            _walk_assign_anchors(sp["nested_subproofs"], anchor_prefix)

    _walk_assign_anchors(subproofs)

    # Main key map for cross-references back to main rows from subproofs.
    main_prefix_str = f"{prefix}m"
    main_rev = list(main_stack)[::-1]
    main_key_map: dict[str, str] = {}
    for midx, mentry in enumerate(main_rev):
        if not mentry or "theorem" in mentry:
            continue
        main_key_map[_norm_expr(mentry[0])] = f"{main_prefix_str}-entry{midx}"

    subproof_external_map = {**main_key_map, **subproof_anchor_map}

    # Chapter-wide (expr, ns)-keyed anchor map. Mirrors the per-call
    # enumerate(rev) iteration inside format_stack_entries so the
    # assigned IDs match emission. Used by ingredient resolution to
    # disambiguate same-expression rows in different scopes (e.g. an
    # OR-convergence row at the parent scope and the per-branch
    # derivations of the same conclusion in branch-local scopes).
    chapter_ns_map: dict[tuple[str, str], str] = {}

    def _norm(s: str) -> str:
        return re.sub(r'\s+', '', s or '').lower()

    for midx, mentry in enumerate(main_rev):
        if not mentry or "theorem" in mentry:
            continue
        norm_expr_m = _norm(mentry[0])
        norm_ns_m = _norm(mentry[1] if len(mentry) > 1 else '')
        chapter_ns_map[(norm_expr_m, norm_ns_m)] = f"{main_prefix_str}-entry{midx}"

    def _walk_subproof_keys(sps: list[dict]) -> None:
        for sp in sps:
            sp_prefix = sp["_anchor_prefix"]
            body_prefix = f"{sp_prefix}body"
            sp_rev = list(sp["display_stack"])[::-1]
            for s_idx, sentry in enumerate(sp_rev):
                if not sentry or "theorem" in sentry:
                    continue
                norm_expr_s = _norm(sentry[0])
                norm_ns_s = _norm(sentry[1] if len(sentry) > 1 else '')
                chapter_ns_map[(norm_expr_s, norm_ns_s)] = f"{body_prefix}-entry{s_idx}"
            _walk_subproof_keys(sp["nested_subproofs"])

    _walk_subproof_keys(subproofs)

    # Chapter-wide common line numbering: every visible row across the
    # main stack + every subproof + every nested sub-sub-proof gets a
    # single sequential number. Lets the user (or future Claude) refer
    # unambiguously to "line N" of a chapter regardless of which
    # subproof card it lives in. Threaded through format_stack_entries
    # and _render_subproof_card via the global_counter / global_total
    # parameters; rendered as `(N/total)` in the step-badge.
    main_visible = sum(1 for e in main_stack if e and 'theorem' not in e)
    sub_visible = _count_visible_in_subproof_tree(subproofs)
    global_total = main_visible + sub_visible
    global_counter = [0]

    blocks: list[str] = []
    scope_class = " subproofs-section" if hierarchical else ""
    blocks.append(f"<div class='proof-section main-proof-section{scope_class}'>")
    blocks.append("<div class='proof-section-title'>Main stack</div>")
    # Mirror the per-subproof-card meta-line: "Goal alias: <expr> (ns)".
    # For the main section, ns is "main" and expr is the chapter's
    # theorem stripped to its HEAD — the inner-most non-implication
    # body — so the user sees what's actually being proved (e.g.
    # `(interval[N,+,i0,i1,V1])`) instead of the full anchor-bound
    # implication wrapper. The peel loop below strips every
    # `(>[...]premise body)` layer and continues with the body.
    if main_goal:
        head_display = main_goal
        try:
            while head_display.startswith('(>['):
                bracket_close = head_display.index(']', 3)
                premise_start = bracket_close + 1
                if premise_start >= len(head_display) or head_display[premise_start] != '(':
                    break
                premise_end = _find_matching_paren_local(head_display, premise_start)
                if premise_end < 0:
                    break
                body_start = premise_end + 1
                if body_start >= len(head_display) or head_display[body_start] != '(':
                    break
                body_end = _find_matching_paren_local(head_display, body_start)
                if body_end < 0:
                    break
                head_display = head_display[body_start:body_end + 1]
        except (ValueError, IndexError):
            head_display = main_goal
        # Goal-alias meta-line: namespace tag is plain text, NOT a
        # click target. The alias expression next to it is also dead
        # text, so the whole meta-line reads as a label rather than as
        # navigation. (Cross-scope clickability for namespaces is kept
        # intact on chapter-row validity tags via _format_validity_tag's
        # default namespace_anchor_map argument.)
        blocks.append(
            f"<div class='subproof-meta'>Goal alias: "
            f"<span class='subproof-goal'>{html.escape(head_display)}</span>"
            f"{_format_validity_tag('main')}</div>")
    else:
        blocks.append(
            f"<div class='subproof-meta'>"
            f"{_format_validity_tag('main')}</div>")
    if hierarchical:
        if subproofs:
            blocks.append(
                "<div class='subproof-controls'>"
                "<button type='button' onclick=\""
                "this.closest('.subproofs-section').querySelectorAll("
                "'.subproof-card.collapsed').forEach(function(c){"
                "c.classList.remove('collapsed');});\">"
                "Expand all</button>"
                "<button type='button' onclick=\""
                "this.closest('.subproofs-section').querySelectorAll("
                "'.subproof-card').forEach(function(c){"
                "c.classList.add('collapsed');});\">"
                "Collapse all</button>"
                "</div>")
        blocks.append(_render_scope_contents(
            main_stack, subproofs, 'main', main_prefix_str, 1, scope_items,
            subproof_external_map, global_counter, global_total,
            namespace_anchor_map, chapter_ns_map))
    elif main_stack:
        blocks.append(format_stack_entries(
            main_stack, prefix=main_prefix_str,
            external_anchor_map=subproof_anchor_map,
            global_counter=global_counter, global_total=global_total,
            namespace_anchor_map=namespace_anchor_map,
            chapter_ns_map=chapter_ns_map))
    else:
        blocks.append("<div class='proof-empty'>No main-stack entries.</div>")
    blocks.append("</div>")

    if subproofs and not hierarchical:
        blocks.append("<div class='proof-section subproofs-section'>")
        blocks.append("<div class='proof-section-title'>Subproofs</div>")
        # Expand-all / collapse-all toggles. The two buttons act on
        # every .subproof-card descendant of the subproofs-section,
        # at every nesting depth — so a single click expands the
        # whole matryoshka chain.
        blocks.append(
            "<div class='subproof-controls'>"
            "<button type='button' onclick=\""
            "this.closest('.subproofs-section').querySelectorAll("
            "'.subproof-card.collapsed').forEach(function(c){"
            "c.classList.remove('collapsed');});\">"
            "Expand all</button>"
            "<button type='button' onclick=\""
            "this.closest('.subproofs-section').querySelectorAll("
            "'.subproof-card').forEach(function(c){"
            "c.classList.add('collapsed');});\">"
            "Collapse all</button>"
            "</div>")
        for sp in subproofs:
            blocks.append(_render_subproof_card(
                sp, prefix=sp["_anchor_prefix"], depth=1,
                external_anchor_map=subproof_external_map,
                global_counter=global_counter, global_total=global_total,
                namespace_anchor_map=namespace_anchor_map,
                chapter_ns_map=chapter_ns_map))
        blocks.append("</div>")

    return "".join(blocks)

# Proof step functions returning HTML for subchapters
def check_variable_typing(theorem, file_path, induction_var, prefix=''):
    """Render the 'Induction variable typing' subsection.

    The goal of this subsection is the typing predicate
    ``(in[<induction_var>,N])`` — i.e. proof that the bound
    induction variable is a natural number — NOT the full theorem
    expression. Override main_goal accordingly so the goal-alias
    line at the head of the rendered stack reads ``in[v,N]`` rather
    than the entire theorem.
    """
    stack = read_stack(file_path, "check_variable_typing")
    rename_stack(stack, theorem)
    typing_goal = f"(in[{induction_var},N])"
    return render_stack_with_subproofs(stack, prefix, main_goal=typing_goal, hierarchical=is_reader_theorem(theorem))


def check_zero(theorem, file_path, induction_var, prefix=''):
    stack = read_stack(file_path, "check_zero")
    rename_stack(stack, theorem)
    return render_stack_with_subproofs(stack, prefix, main_goal=theorem, hierarchical=is_reader_theorem(theorem))


def check_induction_condition(theorem, file_path, induction_var, prefix=''):
    stack = read_stack(file_path, "check_induction_condition")
    rename_stack(stack, theorem)
    return render_stack_with_subproofs(stack, prefix, main_goal=theorem, hierarchical=is_reader_theorem(theorem))


def single_chapter_proof(theorem, file_path, prefix=''):
    stack = read_stack(file_path, "single chapter proof")
    rename_stack(stack, theorem)
    return render_stack_with_subproofs(stack, prefix, main_goal=theorem, hierarchical=is_reader_theorem(theorem))


def single_chapter_proof_heading(file_path):
    """
    @brief Select the human-facing HTML heading for one single-chapter proof.

    @details The first exported row is the chapter goal. A contradiction tag
    there means the chapter itself is a contradiction proof; contradiction
    rows later in the stack are dependencies and do not change the chapter's
    proof depiction.

    @param file_path Path to the processed single-chapter proof file.
    @return ``"Contradiction Proof"`` for a contradiction goal row, otherwise
    ``"Direct Proof"``.
    @invariant The chapter contains a first row with an expression, namespace,
    and tag.
    @see check_theorem_goal_reached in verifier.py for the same goal-row
    ownership contract.
    """
    proof_rows = read_stack(file_path, "single chapter proof heading")
    assert proof_rows and len(proof_rows[0]) >= 3, (
        "single chapter proof must contain a goal row with a tag"
    )
    return (
        "Contradiction Proof"
        if proof_rows[0][2] == "contradiction"
        else "Direct Proof"
    )


def split_at_plus(s: str) -> tuple[str, str]:
    left, sep, right = s.partition("+")
    if sep != "+":
        raise ValueError("String does not contain '+'")
    return left, right


def debugging(path_plus_end, file_path, prefix=''):
    stack = read_stack(file_path, "debugging")
    return render_stack_with_subproofs(stack, prefix, hierarchical=is_reader_theorem(path_plus_end))


def reformulated(theorem, file_path, prefix=''):
    stack = read_stack(file_path, "reformulated statement")
    rename_stack(stack, theorem)
    return render_stack_with_subproofs(stack, prefix, main_goal=theorem, hierarchical=is_reader_theorem(theorem))


def back_reformulated(theorem, file_path, prefix=''):
    stack = read_stack(file_path, "incubator back reformulation")
    rename_stack(stack, theorem)
    return render_stack_with_subproofs(stack, prefix, main_goal=theorem, hierarchical=is_reader_theorem(theorem))


def or_theorem(theorem, file_path, prefix=''):
    stack = read_stack(file_path, "or theorem")
    rename_stack(stack, theorem)
    return render_stack_with_subproofs(stack, prefix, main_goal=theorem, hierarchical=is_reader_theorem(theorem))


def or_elimination(theorem, file_path, prefix=''):
    stack = read_stack(file_path, "or elimination")
    rename_stack(stack, theorem)
    return render_stack_with_subproofs(stack, prefix, main_goal=theorem, hierarchical=is_reader_theorem(theorem))


def _compute_chapter_stats(*file_paths):
    """Compute stats across one or more proof stack files."""
    steps = 0
    tags = set()
    theorems_cited = 0
    for fp in file_paths:
        stack = read_stack(fp, None)
        for entry in stack:
            if not entry:
                continue
            tag = entry[2] if len(entry) > 2 else ""
            if 'theorem' in entry and tag != "theorem":
                continue
            steps += 1
            if tag:
                tags.add(tag)
            if tag == "theorem":
                theorems_cited += 1
    parts = [f"{steps} step{'s' if steps != 1 else ''}"]
    parts.append(f"{len(tags)} reasoning rule{'s' if len(tags) != 1 else ''}")
    if theorems_cited:
        parts.append(f"{theorems_cited} theorem{'s' if theorems_cited != 1 else ''} cited")
    return ", ".join(parts)


def read_theorem_list(map_path: str | Path | None = None):
    """
    Reads PROJECT_ROOT/files/processed_proof_graph/global_theorem_list.txt
    and returns a list of (theorem, method, var) tuples.

    Each line in the file must be tab-separated: theorem \t method \t var
    Blank or malformed lines are ignored.
    """
    if map_path is None:
        map_path = Path(PROJECT_ROOT) / "files" / "processed_proof_graph" / "global_theorem_list.txt"
    else:
        map_path = Path(map_path)

    theorem_list = []
    with open(map_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n").rstrip("\r")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            theorem, method, var = parts[0], parts[1], parts[2]
            theorem_list.append((theorem, method, var))
    return theorem_list


def makes_file_path_map(theorem_list, base_dir=None):
    """
    Build a map from theorem name -> list of indexed stack file paths.

    Indexing matches generate_raw_proof_graph/find_ends:
      - induction  -> two files:  <i>_check_zero.txt, <i+1>_check_induction_condition.txt
      - single chapter proof -> one file: <i>_single_chapter_proof.txt
      - debug      -> one file:   <i>_debug.txt
      - unknown method -> one file: <i>_unknown_<sanitized>.txt

    Args:
        theorem_list: list[tuple[str, str, str]] like [(theorem, method, var), ...]
        base_dir: optional Path/str; default PROJECT_ROOT/files/raw_proof_graph

    Returns:
        dict[str, list[Path]]
    """
    if base_dir is None:
        base_dir = Path(PROJECT_ROOT) / "files" / "processed_proof_graph"
    else:
        base_dir = Path(base_dir)

    result = {}
    idx = 0  # global file index (0-based)

    for name, method, var in theorem_list:
        m = (method or "").lower()
        files = []

        if m in ("induction", "strong induction"):
            files.append(base_dir / f"{idx}_induction_typing.txt")
            files.append(base_dir / f"{idx + 1}_check_zero.txt")
            files.append(base_dir / f"{idx + 2}_check_induction_condition.txt")
            idx += 3
        elif m == "single chapter proof":
            files.append(base_dir / f"{idx}_single_chapter_proof.txt")
            idx += 1
        elif m == "debug":
            files.append(base_dir / f"{idx}_debug.txt")
            idx += 1
        elif m == "reformulated statement":
            files.append(base_dir / f"{idx}_reformulated_statement.txt")
            idx += 1
        elif m == "incubator back reformulation":
            files.append(base_dir / f"{idx}_back_reformulated_statement.txt")
            idx += 1
        elif m == "or theorem":
            files.append(base_dir / f"{idx}_or_theorem.txt")
            idx += 1
        elif m == "or elimination":
            files.append(base_dir / f"{idx}_or_elimination.txt")
            idx += 1
        elif m == "proved not broadcast":
            files.append(base_dir / f"{idx}_proved_not_broadcast.txt")
            idx += 1
        else:
            safe = re.sub(r"[^A-Za-z0-9._\-+]+", "_", m)[:64] or "unknown"
            files.append(base_dir / f"{idx}_unknown_{safe}.txt")
            idx += 1

        # accumulate (support multiple entries of the same theorem name)
        if name in result:
            result[name].extend(files)
        else:
            result[name] = files

    return result


# ---------------------------------------------------------------------------
# GL Binary Map — read per-tag JSON from files/GL_binaries/
# ---------------------------------------------------------------------------

def _gl_split_elements(elements_raw):
    """Split elements string into individual elements, handling negation prefix."""
    elements = []
    current = ''
    depth = 0
    for ch in elements_raw:
        if ch == '(':
            depth += 1
            current += ch
        elif ch == ')':
            depth -= 1
            current += ch
            if depth == 0:
                elements.append(current.strip())
                current = ''
        elif depth == 0 and ch == '!':
            current += ch
        elif depth == 0 and ch == ' ':
            current = ''
        else:
            current += ch
    return elements


def _gl_get_bracket_tokens(expr):
    """Get all tokens from bracket args in order of first appearance."""
    tokens = []
    for m in re.finditer(r'\[([^\]]*)\]', expr):
        for tok in m.group(1).split(','):
            tok = tok.strip()
            if tok and tok not in tokens:
                tokens.append(tok)
    return tokens


def _gl_rename_vars(expr, var_map):
    """Replace variable tokens inside brackets using var_map."""
    def _repl(m):
        args = m.group(1).split(',')
        return '[' + ','.join(var_map.get(a.strip(), a.strip()) for a in args) + ']'
    return re.sub(r'\[([^\]]*)\]', _repl, expr)


def _gl_make_conjunction(elements):
    """Build nested right-associated conjunction: (&e1(&e2 e3))"""
    if len(elements) == 0:
        return ''
    if len(elements) == 1:
        return elements[0]
    if len(elements) == 2:
        return '(&' + elements[0] + elements[1] + ')'
    return '(&' + elements[0] + _gl_make_conjunction(elements[1:]) + ')'


def build_gl_binary_map(gl_binaries_dir):
    """
    @brief Build the right-click popup's definition map from the shared registry.

    @details Every non-atomic compiled name and user definition of the run is
    read from `GL_binary_shared.json` (`visu_helpers.read_shared_definition_registry`)
    and rendered as its signature over `x<k>` arguments and its expanded MPL over
    `y<k>` bound variables. Atomic relations and entries without elements have no
    expansion and are left out.

    @param gl_binaries_dir Directory holding the GL binary JSON files.
    @return Mapping from compiled name to its `signature` and `mpl` strings.
    @see embed_page_definitions
    """
    gl_binary_map = {}
    for core, entry in visu_helpers.read_shared_definition_registry(gl_binaries_dir).items():
        category = entry.get('category', '')
        signature = entry.get('signature', '')
        elements_list = entry.get('elements', [])

        if category == 'atomic' or not elements_list:
            continue

        # Parse signature args -> u_ to x mapping
        sig_args = []
        m = re.search(r'\[([^\]]*)\]', signature)
        if m:
            sig_args = [a.strip() for a in m.group(1).split(',')]
        u_to_x = {}
        xi = 1
        for arg in sig_args:
            if arg.startswith('u_') and arg not in u_to_x:
                u_to_x[arg] = f'x{xi}'
                xi += 1

        # Elements are already a list from JSON
        elements = elements_list

        # Find bound vars (non-u_ tokens) across all elements -> y mapping
        bound_to_y = {}
        yi = 1
        for elem in elements:
            for tok in _gl_get_bracket_tokens(elem):
                if tok not in u_to_x and tok not in bound_to_y:
                    bound_to_y[tok] = f'y{yi}'
                    yi += 1

        # Combined variable mapping
        var_map = {**u_to_x, **bound_to_y}

        # Rename signature and elements
        renamed_sig = _gl_rename_vars(signature, var_map)
        renamed_elems = [_gl_rename_vars(e, var_map) for e in elements]

        # Collect bound var names (yN) for quantifiers
        if category == 'existence':
            qvars = []
            for tok in _gl_get_bracket_tokens(elements[0]):
                if tok in bound_to_y and bound_to_y[tok] not in qvars:
                    qvars.append(bound_to_y[tok])
        else:
            qvars = list(bound_to_y.values())

        bound_str = ','.join(qvars)

        # Reconstruct MPL based on category
        if category == 'and':
            mpl = _gl_make_conjunction(renamed_elems)
        elif category == 'existence':
            first = renamed_elems[0]
            second = renamed_elems[1] if len(renamed_elems) > 1 else ''
            if second.startswith('!'):
                neg_second = second[1:]
            else:
                neg_second = '!' + second
            mpl = '!(>[' + bound_str + ']' + first + neg_second + ')'
        elif category == 'implication':
            premises = renamed_elems[:-1]
            conclusion = renamed_elems[-1]
            premises_conj = _gl_make_conjunction(premises)
            mpl = '(>[' + bound_str + ']' + premises_conj + conclusion + ')'
        elif category == 'or':
            # Mirror prover.cpp expandSignature CASE 4 (OR). Elements
            # carry each disjunct's true polarity; the conjunct form is
            # the element's negation WITH double-negation cancellation
            # (a negated disjunct contributes its bare positive core).
            def _neg(e):
                return e[1:] if e.startswith('!') else '!' + e
            if not renamed_elems:
                mpl = renamed_sig
            elif len(renamed_elems) == 1:
                mpl = renamed_elems[0]
            else:
                mpl = ('!(&' + _neg(renamed_elems[0])
                       + _neg(renamed_elems[1]) + ')')
                for i in range(2, len(renamed_elems)):
                    mpl = '!(&' + mpl + _neg(renamed_elems[i]) + ')'
        else:
            mpl = _gl_make_conjunction(renamed_elems)

        gl_binary_map[core] = {
            'signature': renamed_sig,
            'mpl': mpl
        }

    return gl_binary_map



_PAGE_DEFINITIONS_MARKER = "/*GL_PAGE_DEFINITIONS*/"
_DATA_PARTS = re.compile(r'data-parts="([^"]*)"')


def embed_page_definitions(page, gl_binary_map):
    """
    @brief Embed the definitions a page's expressions name into its popup script.

    @details The right-click popup expands only the compiled names listed in the
    clicked expression's `data-parts` attribute. A page therefore needs exactly
    the definitions its own `data-parts` attributes name; they replace the popup
    script's single marker as one JSON object in name order, so every page stays
    self-contained without carrying the whole registry. Names without a definition
    (atomic relations) are left out, as the popup skips them.

    @param page Complete page HTML holding the popup script's marker exactly once.
    @param gl_binary_map Mapping from compiled name to its popup entry.
    @return The page with the marker replaced by its definition map.
    @invariant The marker occurs exactly once in the page (asserted).
    @see build_gl_binary_map
    """
    assert page.count(_PAGE_DEFINITIONS_MARKER) == 1, "popup definition marker must occur exactly once"
    names = sorted({name for parts in _DATA_PARTS.findall(page)
                    for name in html.unescape(parts).split(",") if name in gl_binary_map})
    definitions = json.dumps({name: gl_binary_map[name] for name in names}, ensure_ascii=False)
    return page.replace(_PAGE_DEFINITIONS_MARKER, definitions)


_READABLE_NOTATION_ROWS = (
    ("RULE / APPLIED WITH", "The cited rule in notation, then the renaming of its variables that turns it into the row."),
    ("READS", "The row's own expression in notation."),
    ("REFORMULATED FROM", "Both forms of a reformulation, written out without substitution."),
    ("from A, B follows C", "An implication with premises A and B. A variable that a premise defines is replaced by its value when nothing is lost: values of s, + and · always, values of other functions only when another premise restricts the argument; quotient and remainder never."),
    ("f(a) = b, a + b = c", "The pair (a, b) belongs to the relation f; the operation + maps (a, b) to c."),
    ("F ⊆ G", "Every pair of the relation F is a pair of G."),
    ("∀ x ∈ S: A ⇒ B,  ∃ y ∈ S: …,  ∄", "Quantified statements inside a formula."),
    ("y₁, Y₁", "Names introduced when a compiled definition is written out: lowercase for elements, uppercase for sets and functions."),
    ("∑, ∏ with bounds", "Sum or product from the lower to the upper bound, both included."),
    ("≤, <, ∣ and ≰, ≮, ∤", "Order, strict order and divisibility, crossed when negated; a div d and a mod d are quotient and remainder."),
    ("[n, m], f: X → Y, f|≤n", "Interval, function type, restriction of f to arguments ≤ n."),
    ("g = f ∪ {n ↦ a}", "The sequence f extended by the value a at n."),
    ("∃ primes p₀ ≥ p₁ ≥ … ≥ pₖ: n = ∏ᵢ₌₀ᵏ pᵢ", "n is the product of the primes p₀, …, pₖ, listed in non-increasing order: a prime factorization of n."),
    ("gcd(a, b) = 1,  min M,  ∅", "Coprime numbers, the least element of a set, the empty set. Every defined notion reads in its own standard notation; letters it introduces itself (d, k, m, n, p) are never names of the proof. Right-click an expression for its definition."),
    ("x̃", "A fresh copy of x made for a subproof (anchor handling)."),
)

_READABLE_NOTATION_HTML = (
    '<h2 id="readable-notation" style="margin-top:0.5em;">Readable version</h2>\n'
    '  <p style="color:var(--gl-text-muted); margin-bottom:0.8em;">The grey lines translate each row into '
    'mathematical notation.</p>\n'
    '  <table class="notation-table" style="border-collapse:collapse; width:100%; margin-bottom:2em;">\n'
    + "\n".join(
        f'    <tr><td style="white-space:nowrap; vertical-align:top; padding:0.4em 1.2em 0.4em 0; '
        f'color:var(--gl-text-muted);">{html.escape(notation)}</td>'
        f'<td style="vertical-align:top; padding:0.4em 0;">{html.escape(meaning)}</td></tr>'
        for notation, meaning in _READABLE_NOTATION_ROWS)
    + '\n  </table>')


def _generate_tags_page(out_dir, common_style):
    """Generate a tags.html reference page listing all proof tags with descriptions."""
    rows = []
    for tag_key, (short_desc, long_desc) in TAG_DESCRIPTIONS.items():
        anchor = tag_key.replace(" ", "-").replace("(", "").replace(")", "")
        rows.append(
            f'<tr id="{html.escape(anchor)}">'
            f'<td style="white-space:nowrap; vertical-align:top; padding:0.6em 1.2em 0.6em 0;">'
            f'<b>{html.escape(tag_key)}</b></td>'
            f'<td style="vertical-align:top; padding:0.6em 1.2em 0.6em 0; color:var(--gl-link);">'
            f'{html.escape(short_desc)}</td>'
            f'<td style="vertical-align:top; padding:0.6em 0;">'
            f'{long_desc}</td></tr>'
        )
    table_html = "\n".join(rows)

    tags_page = f"""<!DOCTYPE html>
{LICENSE_SOURCE_COMMENT}
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Reasoning Rules Reference</title>
  <link rel="icon" type="image/png" href="favicon.png">
{LICENSE_HEAD_META}
  {common_style}
  <style>
    tr:target {{ background: var(--gl-goal-bg); }}
    tr:target td {{ padding-top: 0.8em; padding-bottom: 0.8em; }}
    /* phones: one entry per block instead of columns that cannot fit */
    @media (max-width: 640px) {{
      .rules-table, .rules-table tbody, .rules-table tr, .rules-table td,
      .notation-table, .notation-table tbody, .notation-table tr, .notation-table td {{ display: block; width: auto; }}
      .rules-table thead {{ display: none; }}
      .rules-table tr, .notation-table tr {{ padding: 0.6em 0; border-bottom: 1px solid var(--gl-rule); }}
      .rules-table td, .notation-table td {{ white-space: normal !important; padding: 0.15em 0 !important; overflow-wrap: anywhere; }}
    }}
  </style>
</head>
<body>
  <nav><a href="index.html">Index</a></nav>
  <h1>Reasoning Rules Reference</h1>
  <p style="color:var(--gl-text-muted); margin-bottom:1.5em;">
    Each step in a proof chapter is annotated with a reasoning rule that describes how the
    expression was derived. Click a rule name in any chapter to jump here; right-click
    for a quick popup.
  </p>
  {_READABLE_NOTATION_HTML}
  <table class="rules-table" style="border-collapse:collapse; width:100%;">
    <thead>
      <tr style="border-bottom:2px solid var(--gl-rule);">
        <th style="text-align:left; padding:0.5em 1.2em 0.5em 0;">Reasoning rule</th>
        <th style="text-align:left; padding:0.5em 1.2em 0.5em 0; color:var(--gl-link);">Short</th>
        <th style="text-align:left; padding:0.5em 0;">Description</th>
      </tr>
    </thead>
    <tbody>
      {table_html}
    </tbody>
  </table>
{LICENSE_FOOTER}
</body>
</html>"""

    with open(os.path.join(out_dir, "tags.html"), "w", encoding="utf-8") as f:
        f.write(tags_page)


def book_chapter_numbers(theorem_list):
    """@brief Number a book's chapter pages: one consecutive number per theorem.

    @details
    The processed proof graph stores an induction theorem as three chapter
    files (variable typing, zero case, induction condition), so its file
    numbers advance by three, while the book renders the theorem as one page
    whose parts read N.1 to N.3 (`induction_part_numbers`). A page is therefore
    numbered by its theorem's position in `global_theorem_list.txt`, from 0:
    the heading, the file name `chapter<N>.html`, the navigation, theorem
    citations, "used by" lists, sibling-book links and the Lean twin page all
    read this number, and consecutive theorems have consecutive chapters.
    The processed chapter files, the verifier and the Lean code keep their
    file numbers.

    @param theorem_list Records of `global_theorem_list.txt` in book order.
    @return The chapter number of each record, `0` to `len(theorem_list) - 1`.
    @invariant The numbers depend only on the record count, never on methods.
    @see induction_part_numbers
    """
    assert isinstance(theorem_list, list)
    return list(range(len(theorem_list)))


def _build_used_by(theorem_list, file_path_map, chapter_nums):
    """@brief Index the chapters that cite each local theorem.

    @details
    Renamed theorem identities are indexed once in original theorem order.
    Each citation selects the first matching chapter other than its own,
    preserving duplicate-identity handling and the historical self exclusion.

    @param theorem_list Ordered theorem, method, and variable records.
    @param file_path_map Theorem identity to ordered processed chapter paths.
    @param chapter_nums Book chapter number of each theorem record
           (`book_chapter_numbers`).
    @return Mapping from cited chapter number to its set of citing numbers.
    @invariant Indexing does not change first-match or chapter-order semantics.
    @see generate_proof_graph_pages
    """
    assert len(theorem_list) == len(chapter_nums)
    candidates = {}
    for position, (name, _, _) in enumerate(theorem_list):
        candidates.setdefault(rename_theorem(name), []).append(position)
    used_by = {}
    for citing_pos, (citing_name, _, _) in enumerate(theorem_list):
        citing_num = chapter_nums[citing_pos]
        for path in file_path_map.get(citing_name, []):
            if not os.path.exists(path):
                continue
            for entry in read_stack(path, None):
                if not entry or len(entry) < 3 or entry[2] != "theorem":
                    continue
                matches = candidates.get(rename_theorem(entry[0]), ())
                for source_pos in matches:
                    if source_pos != citing_pos:
                        used_by.setdefault(chapter_nums[source_pos], set()).add(citing_num)
                        break
    return used_by


def generate_proof_graph_pages(config: configuration_reader,
                               proc_dir=None, out_dir=None,
                               sibling_graphs=None, lean_kernel_check_running=False):
    """@brief Render a proof graph with caches confined to this invocation.

    @details
    Link identities and processed chapter files stay fixed during one render.
    Repeated cells share rendered strings; chapter reads return independent
    mutable rows. A nested render or an exception restores the caller's cache.

    @param config Visualizer configuration for the current graph.
    @param proc_dir Processed chapter directory, or the main graph default.
    @param out_dir HTML destination, or the main graph default.
    @param sibling_graphs Peer graph directories for cross-graph links.
    @param lean_kernel_check_running Whether the Lean export's kernel check is
           still running while the pages are rendered (the pipeline's main HTML).
    @return None.
    @invariant Cached links and chapter contents never survive a render call.
    @see _generate_proof_graph_pages
    """
    token = _RENDER_CACHE.set({"clickable": {}, "chapters": {}})
    try:
        return _generate_proof_graph_pages(config, proc_dir, out_dir, sibling_graphs,
                                           lean_kernel_check_running)
    finally:
        _RENDER_CACHE.reset(token)


def _generate_proof_graph_pages(config: configuration_reader,
                                proc_dir=None, out_dir=None,
                                sibling_graphs=None, lean_kernel_check_running=False):
    """@brief Generate all index, legend, proof, and Lean twin pages.

    @details
    Builds complete local and sibling link maps before rendering the ordered
    theorem pages. The public entry point supplies the optional per-call cache;
    direct calls execute the same rendering without caching mutable context.

    @param config Visualizer configuration for the current graph.
    @param proc_dir Processed chapter directory, or the main graph default.
    @param out_dir HTML destination, or the main graph default.
    @param sibling_graphs Peer graph directories for cross-graph links.
    @param lean_kernel_check_running Whether the Lean export's kernel check is
           still running (`lean_pages.load_lean_export`).
    @return None.
    @invariant Page order, chapter identities, and citation targets are stable.
    @see generate_proof_graph_pages
    """

    global theorem_to_file, theorem_shape_to_file
    global sibling_theorem_to_file, sibling_theorem_shape_to_file
    expression_utils.set_configuration(config)

    if proc_dir is None:
        proc_dir = PROJECT_ROOT / "files" / "processed_proof_graph"
    if out_dir is None:
        out_dir = PROJECT_ROOT / "files/full_proof_graph"

    # If the directory already exists, clear its contents. Use a per-
    # child unlink+rmtree pattern with ignore_errors so a chapter HTML
    # held open by a browser tab on Windows (which holds a directory
    # lock) doesn't abort the whole regen — files we can rewrite get
    # rewritten in place; locked ones are overwritten by the later
    # write step. shutil.rmtree on the directory itself fails on
    # Windows when ANY child is open, so we don't try.
    if os.path.isdir(out_dir):
        for child in os.listdir(out_dir):
            if child == lean_pages.EXPORT_FOLDER:
                # The run's Lean export lives here (proof_export.live writes it
                # before the HTML stage); the pages link into it.
                continue
            child_path = os.path.join(out_dir, child)
            try:
                if os.path.isfile(child_path) or os.path.islink(child_path):
                    os.unlink(child_path)
                elif os.path.isdir(child_path):
                    shutil.rmtree(child_path, ignore_errors=True)
            except OSError:
                pass

    os.makedirs(out_dir, exist_ok=True)

    # Copy favicon logo into output directory.
    # Historical name kept for backward compatibility; falls back to
    # gl-logo-small.png (the current 192x192 source) if absent.
    favicon_src = PROJECT_ROOT / "small_logo_amber_bl_bg_2.png"
    if not favicon_src.exists():
        favicon_src = PROJECT_ROOT / "gl-logo-small.png"
    if favicon_src.exists():
        shutil.copy2(favicon_src, Path(out_dir) / "favicon.png")

    # Copy the same 192x192 source into the output dir as gl-logo.png —
    # rendered at 40px in the chapter footer for click-back-home identity.
    logo_src = PROJECT_ROOT / "gl-logo-small.png"
    if logo_src.exists():
        shutil.copy2(logo_src, Path(out_dir) / "gl-logo.png")

    theorem_list = read_theorem_list(proc_dir / "global_theorem_list.txt")
    file_path_map = makes_file_path_map(theorem_list, base_dir=proc_dir)

    # One book chapter number per theorem (D-591):
    # an induction theorem's three processed files make one page.
    chapter_nums = book_chapter_numbers(theorem_list)

    # Build "used by" reverse-dependency map: chapter number -> citing chapter numbers
    used_by = _build_used_by(theorem_list, file_path_map, chapter_nums)

    # Build GL binary map from per-tag JSON files (check next to proc_dir first, then default)
    gl_binaries_dir = Path(proc_dir).parent / "GL_binaries"
    if not gl_binaries_dir.exists():
        gl_binaries_dir = PROJECT_ROOT / "files" / "GL_binaries"
    gl_binary_map = build_gl_binary_map(gl_binaries_dir)
    global _known_theorems_for_reader
    _known_theorems_for_reader = tuple(rename_theorem(t[0]) for t in theorem_list)
    external_path = proc_dir / "external_theorems.txt"
    if external_path.is_file():
        _known_theorems_for_reader += tuple(rename_theorem(t) for t in external_path.read_text(encoding="utf-8").splitlines() if t)
    global _known_theorem_keys_for_reader
    _known_theorem_keys_for_reader = theorem_alpha_keys(_known_theorems_for_reader)
    global _readable_form_for_render
    _readable_form_for_render = visu_helpers.ReadableForm(
        visu_helpers.load_compiled_definitions(gl_binaries_dir), _anchor_signatures(),
        structured_implications=any(is_reader_theorem(t[0]) for t in theorem_list))
    # Expose to module-level for _expand_or_to_mpl (used by
    # _subproof_explanation in _render_subproof_card).
    global _gl_binary_map_for_render
    _gl_binary_map_for_render = gl_binary_map

    # JavaScript for left-click/right-click expansion; popup named "Expression" with deep-navy styling
    popup_script = """
    <script>
    const GL_BINARY_MAP = """ + _PAGE_DEFINITIONS_MARKER + """;

    function processText(input) {
      const indentChar = "  ";
      let indent = 0, output = "", token = "";
      for (let i = 0; i < input.length; i++) {
        const char = input[i];
        if (char === "(") {
          if (token.trim()) { output += indentChar.repeat(indent) + token.trim() + "\\n"; token = ""; }
          // If the next char is `&` (n-ary conjunction operator that
          // sits flush against the opening paren in MPL: `(&E1 E2)`),
          // glue them: emit `(&` on one line so the operator stays at
          // a visible depth one shift past the parent's `(`.
          // Operands then read as siblings of `&`, indent+1 from `(`.
          if (input[i + 1] === "&") {
            output += indentChar.repeat(indent) + "(&\\n";
            indent++;
            i++;  // consumed the `&` along with the `(`
          } else {
            output += indentChar.repeat(indent) + "(\\n";
            indent++;
          }
        } else if (char === ")") {
          if (token.trim()) { output += indentChar.repeat(indent) + token.trim() + "\\n"; token = ""; }
          indent = Math.max(0, indent - 1);
          output += indentChar.repeat(indent) + ")" + "\\n";
        } else if (char === "!") {
          // Negation marker. Three cases:
          //   `!(&` -> triple-glue: De Morgan OR shape `!(&!E1!E2)`
          //            renders as `!(&\\n  !(...)  !(...)\\n)` so the
          //            outer-OR header occupies one compact line.
          //   `!(`  -> double-glue: negated parenthesised expression.
          //   `!`   -> alone (e.g. `!E` where E is a non-parenthesised
          //            token, or `&!` boundary) — emit on its own line
          //            so the negation never visually merges with a
          //            preceding `&`.
          // The accumulated token (e.g. the `&` of an outer
          // conjunction) is flushed first so it gets its own line.
          if (token.trim()) { output += indentChar.repeat(indent) + token.trim() + "\\n"; token = ""; }
          if (input[i + 1] === "(" && input[i + 2] === "&") {
            output += indentChar.repeat(indent) + "!(&\\n";
            indent++;
            i += 2;  // consumed the `(` and the `&`
          } else if (input[i + 1] === "(") {
            output += indentChar.repeat(indent) + "!(\\n";
            indent++;
            i++;  // consumed the `(`
          } else {
            output += indentChar.repeat(indent) + "!\\n";
          }
        } else {
          token += char;
        }
      }
      if (token.trim()) output += indentChar.repeat(indent) + token.trim() + "\\n";
      return output;
    }

    function escapeHtml(s) {
      return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    }

    document.addEventListener('DOMContentLoaded', function() {
      // Inject a reusable dialog into the body
      document.body.insertAdjacentHTML('beforeend', `
        <dialog id="expr-modal" style="border:1px solid var(--gl-rule); border-radius:6px; background:var(--gl-surface); color:var(--gl-text); padding:1em; max-width:80%; max-height:80vh; overflow:auto; box-shadow: 0 4px 12px var(--gl-shadow);">
          <pre id="expr-content" style="white-space:pre-wrap; font-family:var(--gl-font-mono); margin:0;"></pre>
        </dialog>
      `);
      
      const modal = document.getElementById('expr-modal');
      const content = document.getElementById('expr-content');

      // Close modal when clicking anywhere outside the box
      modal.addEventListener('click', (e) => {
        if (e.target === modal) modal.close();
      });

      document.body.addEventListener('contextmenu', function(e) {
        const el = e.target.closest('.clickable');
        if (!el) return;
        e.preventDefault();

        const rawText = el.getAttribute('data-text');
        let output = processText(rawText);
        let glBinary = '';

        // GL binary section: show definition for each constituent expression
        const parts = el.getAttribute('data-parts');
        if (parts) {
            const names = parts.split(',').filter(Boolean);
            const shown = [];
            for (const name of names) {
                const entry = GL_BINARY_MAP[name];
                if (entry && !shown.includes(name)) {
                    shown.push(name);
                    if (shown.length === 1) {
                        glBinary += '\\n\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\n\\n';
                        glBinary += '<b style="font-size:1.15em">GL binary:</b>\\n\\n';
                    } else {
                        glBinary += '\\n';
                    }
                    glBinary += escapeHtml(entry.signature + ' :=\\n\\n');
                    glBinary += escapeHtml(processText(entry.mpl));
                }
            }
        }

        // Orphan-external section: when the clicked element is marked
        // data-external-orphan="1" (i.e. registered as external in this
        // pipeline's external_theorems.txt but has no chapter — neither
        // locally nor in any sibling pipeline), append an explanation so
        // the reader knows why the link doesn't navigate.
        let orphanNote = '';
        if (el.getAttribute('data-external-orphan') === '1') {
          orphanNote += '\\n\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\n\\n';
          orphanNote += '<b style="font-size:1.15em">External theorem — no proof graph in this release</b>\\n\\n';
          orphanNote += 'This expression is a registered external theorem (it appears in this\\n';
          orphanNote += "pipeline's external_theorems.txt). The corresponding proof graph is not\\n";
          orphanNote += 'included in the current release — either because the source pipeline was\\n';
          orphanNote += 'not run, or because the theorem was supplied as an axiom via\\n';
          orphanNote += "files/theorems/externally_provided_theorems.txt without an accompanying\\n";
          orphanNote += 'proof.\\n\\n';
          orphanNote += "The verifier's origin check confirmed this expression is a known external\\n";
          orphanNote += "(state.external_theorems registry). It is treated as accepted and is not\\n";
          orphanNote += 're-checked here. The expression itself, the GL-binary expansion of every\\n';
          orphanNote += 'constituent operator, and the dual-license terms (AGPLv3 + commercial,\\n';
          orphanNote += 'see https://generative-logic.com/license) all apply unchanged.';
        }

        content.innerHTML = escapeHtml(output) + (glBinary ? glBinary : '') + (orphanNote ? orphanNote : '');
        modal.showModal();
      });

      // --- Tag popup (right-click on proof tags) ---
      const TAG_INFO = """ + json.dumps(
        {k: {"short": v[0], "long": v[1]} for k, v in TAG_DESCRIPTIONS.items()},
        ensure_ascii=False) + """;

      // Tag popup dialog
      document.body.insertAdjacentHTML('beforeend', `
        <dialog id="tag-modal" style="border:1px solid var(--gl-rule); border-radius:6px; background:var(--gl-surface); color:var(--gl-text); padding:1.2em 1.5em; max-width:500px; box-shadow: 0 4px 12px var(--gl-shadow);">
          <div id="tag-content"></div>
        </dialog>
      `);
      const tagModal = document.getElementById('tag-modal');
      const tagContent = document.getElementById('tag-content');
      tagModal.addEventListener('click', (e) => {
        if (e.target === tagModal) tagModal.close();
      });

      document.body.addEventListener('contextmenu', function(e) {
        const tagEl = e.target.closest('.proof-tag');
        if (!tagEl) return;
        e.preventDefault();
        e.stopPropagation();
        const tagKey = tagEl.getAttribute('data-tag') || '';
        const info = TAG_INFO[tagKey];
        if (info) {
          tagContent.innerHTML =
            '<b style="font-size:1.15em; color:var(--gl-link);">' + escapeHtml(tagKey) + '</b>' +
            '<div style="color:var(--gl-text-muted); margin:0.3em 0 0.6em 0; font-size:0.92em;">' + escapeHtml(info.short) + '</div>' +
            '<div style="line-height:1.5;">' + info.long + '</div>';
        } else {
          tagContent.innerHTML = '<b>' + escapeHtml(tagKey) + '</b>';
        }
        tagModal.showModal();
      });

      // --- Integration goal popup (right-click on _integration_goal labels) ---
      document.body.addEventListener('contextmenu', function(e) {
        const igEl = e.target.closest('.integration-goal-label');
        if (!igEl) return;
        e.preventDefault();
        e.stopPropagation();
        tagContent.innerHTML =
          '<b style="font-size:1.15em; color:var(--gl-target);">Integration goal</b>' +
          '<div style="color:var(--gl-text-muted); margin:0.3em 0 0.6em 0; font-size:0.92em;">Operational target, not a derived result</div>' +
          '<div style="line-height:1.5;">' +
          'An integration goal does not need to be justified &mdash; it only must make sense structurally. ' +
          'GL can, for various reasons, decide to try to integrate a compound logical structure. ' +
          'This expression is a <em>goal for the operation</em>, not a result of it.<br><br>' +
          'GL could equally well try to integrate the Schr\\u00F6dinger equation and it would still be ' +
          'a mathematically correct operation &mdash; although it would make zero sense in the current axiomatic context.' +
          '</div>';
        tagModal.showModal();
      });

      // --- Collapsible subproofs ---
      window.setProofCardOpen = function(card, open) {
        card.classList.toggle('collapsed', !open);
        var title = card.querySelector(':scope > .subproof-title');
        if (title && title.hasAttribute('aria-expanded')) title.setAttribute('aria-expanded', String(open));
      };
      document.querySelectorAll('.subproof-title').forEach(function(title) {
        title.style.cursor = 'pointer';
        title.addEventListener('click', function(e) {
          if (e.target.closest('a')) return;
          var card = title.closest('.subproof-card');
          window.setProofCardOpen(card, card.classList.contains('collapsed'));
        });
        title.addEventListener('keydown', function(e) {
          if (e.target !== title || (e.key !== 'Enter' && e.key !== ' ')) return;
          e.preventDefault();
          var card = title.closest('.subproof-card');
          window.setProofCardOpen(card, card.classList.contains('collapsed'));
        });
      });
      function revealProofHash() {
        if (!window.location.hash) return;
        var target = document.getElementById(decodeURIComponent(window.location.hash.slice(1)));
        if (!target) return;
        var node = target.hasAttribute('data-closed-result')
          ? target.closest('.subproof-card').parentElement : target;
        while (node) {
          var card = node.closest('.subproof-card');
          if (!card) break;
          window.setProofCardOpen(card, true);
          node = card.parentElement;
        }
        placeBelowHeader(target);
      }
      // A jump target sits below the floating reader header: a row or block
      // header that fits is centred in the space below it, anything taller
      // (or a section heading) starts just below it.
      function placeBelowHeader(target) {
        var toolbar = document.querySelector('.reader-toolbar');
        var header = toolbar ? toolbar.getBoundingClientRect().height : 0;
        var room = window.innerHeight - header;
        var rect = target.getBoundingClientRect();
        var centred = target.matches('.proof-line, .subproof-title') && rect.height < room;
        var offset = centred ? header + (room - rect.height) / 2 : header + 16;
        window.scrollTo(0, window.scrollY + rect.top - offset);
      }
      window.addEventListener('hashchange', revealProofHash);
      revealProofHash();

      // --- In-page anchor clicks (capture phase): when the link's
      //     target lives inside a collapsed subproof card, uncollapse
      //     that card AND every collapsed ancestor card up the
      //     matryoshka chain BEFORE the browser does its native
      //     scrollIntoView. Without the uncollapse, the browser
      //     scrolls to a display:none element and the user sees
      //     nothing change. The ns-jump class on namespace-tag links
      //     is no longer required — every in-page link
      //     (a[href^='#']) gets the same treatment so ingredient-cell
      //     cross-scope navigation works too.
      //
      //     Plus a brief flash on the target card after click so the
      //     destination is visually unambiguous when the scroll travel
      //     is short.
      document.body.addEventListener('click', function(e) {
        var link = e.target.closest('a[href^="#"]');
        if (!link) return;
        var hash = link.getAttribute('href');
        if (hash === '#' || hash.length < 2) return;
        var targetId = hash.slice(1);
        var targetEl = document.getElementById(targetId);
        if (!targetEl) return;
        // Skip jumps inside the visible main-proof-section — let the
        // browser handle those without modifying any subproof state.
        if (!targetEl.closest('.subproof-card')) return;
        // Walk up from the target through every ancestor subproof-card
        // and uncollapse it.
        var node = targetEl.hasAttribute('data-closed-result')
          ? targetEl.closest('.subproof-card').parentElement : targetEl;
        while (node) {
          var card = node.closest('.subproof-card');
          if (!card) break;
          window.setProofCardOpen(card, true);
          node = card.parentElement;
        }
        // Brief highlight flash on the cited row or header itself, so the
        // user can see WHERE the click landed.
        targetEl.classList.add('jump-flash');
        setTimeout(function() { targetEl.classList.remove('jump-flash'); }, 1100);
        // Don't preventDefault — let the browser do its native anchor
        // navigation now that the target is visible.
      }, true);

      // --- Hover-highlight dependency chain ---
      function collectAncestors(id, visited) {
        if (visited.has(id)) return;
        visited.add(id);
        var el = document.getElementById(id);
        if (!el) return;
        var deps = (el.getAttribute('data-deps') || '').split(' ').filter(Boolean);
        for (var i = 0; i < deps.length; i++) collectAncestors(deps[i], visited);
      }
      document.body.addEventListener('mouseover', function(e) {
        var link = e.target.closest('a[href^="#"]');
        if (!link) return;
        link.classList.add('dep-glow');
        var targetId = link.getAttribute('href').slice(1);
        var targetEl = document.getElementById(targetId);
        if (targetEl) targetEl.classList.add('dep-glow');
        var srcLine = link.closest('span[id]');
        if (srcLine) srcLine.classList.add('dep-glow');
        var ancestors = new Set();
        collectAncestors(targetId, ancestors);
        ancestors.forEach(function(id) {
          var el = document.getElementById(id);
          if (el) el.classList.add('dep-glow');
        });
      });
      document.body.addEventListener('mouseout', function(e) {
        var link = e.target.closest('a[href^="#"]');
        if (!link) return;
        document.querySelectorAll('.dep-glow').forEach(function(el) {
          el.classList.remove('dep-glow');
        });
      });
    });
    </script>
    """

    # Common CSS for chapter pages — dark theme matching generative-logic.com
    common_style = """
    <style>
""" + proof_theme.THEME_TOKENS_CSS + """
      body { font-family: var(--gl-font-ui); margin: 2rem; background: var(--gl-bg); color: var(--gl-text); }
      /* One viewport of tail space lets the last rows scroll to the top; it
         lies inside the body, so the floating reader header stays visible. */
      body::after { content: ""; display: block; height: 100vh; }
      nav a { text-decoration: none; color: var(--gl-link); margin-right: 1em; }
      /* enforce same color & no underline for all links, visited or not */
      a, a:link, a:visited {
        color: var(--gl-link);
        text-decoration: none;
      }
      .var-highlight { font-style: italic; margin-bottom: 1em; display: block; }
      .step-output { margin: 1em 0; padding: 0.5em; background: var(--gl-surface); border-radius: 4px; }
      .clickable    { cursor: pointer; }   /* no color overridden here */
      h1 .clickable { overflow-wrap: anywhere; }
      .integration-goal-label { cursor: pointer; }
      .proof-line { display: flex; align-items: flex-start; margin-bottom: 1.8em; }
      .step-badge { color: var(--gl-text-faint); font-size: 0.75em; white-space: nowrap; min-width: 5em; flex-shrink: 0; text-align: right; margin-right: 0.7em; padding-top: 0.15em; }
      .proof-line-content { flex: 1; min-width: 0; }
      .chapter-stats { margin-top: 2em; padding: 0.6em 1em; background: var(--gl-surface); border-top: 1px solid var(--gl-rule); color: var(--gl-text-muted); font-size: 0.9em; }
      .dep-glow { background: linear-gradient(90deg, rgb(var(--gl-target-rgb) / 0.15), rgb(var(--gl-goal-rgb) / 0.12) 60%, transparent); border-left: 2px solid rgb(var(--gl-target-rgb) / 0.6); padding-left: 6px; border-radius: 4px; box-shadow: 0 0 8px rgb(var(--gl-goal-rgb) / 0.15), inset 0 0 12px rgb(var(--gl-target-rgb) / 0.06); transition: all 0.2s ease; }
      .arg-changed { color: var(--gl-changed); text-decoration: underline; text-decoration-style: wavy; }
      .subproof-toggle { cursor: pointer; user-select: none; margin-right: 0.3em; display: inline-block; transition: transform 0.2s; }
      .subproof-card.collapsed .subproof-body { display: none; }
      .subproof-card.collapsed .subproof-toggle { transform: rotate(-90deg); }
      .goal-highlight { color: var(--gl-goal); font-weight: bold; }
       /* only kill underlines on our autogenerated theorem links */
      a.theorem-link,
      a.theorem-link .clickable {
      text-decoration: none;
      /* force link to inherit whatever color its parent has (e.g. the gold inline span) */
      color: inherit;
     }
     /* make inter-page theorem links mint green */
     a.theorem-link[href$=".html"] .clickable {
     color: var(--gl-link) !important;
     }
     /* cross-batch external-theorem links open in a new tab; render in
        a distinct lavender so the reader sees at a glance it goes to a
        sibling pipeline (e.g. incubator -> main, or main -> incubator) */
     a.theorem-link.external-link[href$=".html"] .clickable {
     color: var(--gl-external) !important;
     border-bottom: 1px dotted var(--gl-external);
     }
     a.theorem-link.external-link[href$=".html"]::after {
     content: " ↗";
     font-size: 0.85em;
     color: var(--gl-external);
     vertical-align: super;
     margin-left: 1px;
     }
     /* orphan external — registered as external in this pipeline but no
        chapter exists (here or in any sibling). Same lavender family as
        the cross-batch link, but dashed-underline + no ↗ icon to signal
        "click won't navigate; right-click for explanation". */
     .clickable.external-orphan {
     color: var(--gl-external) !important;
     border-bottom: 1px dashed var(--gl-external);
     cursor: help;
     }
     /* Namespace-tag jump links: explicit affordance so the user sees
        the yellow tag is clickable. Hover gives a subtle background
        for visual feedback before clicking. */
     a.ns-jump {
     cursor: pointer;
     border-radius: 3px;
     padding: 0 2px;
     transition: background 0.12s;
     }
     a.ns-jump:hover {
     background: rgb(var(--gl-namespace-rgb) / 0.18);
     }
     /* Brief flash on the directly-targeted subproof card right after
        an ns-jump click, to make the destination visually obvious. */
     @keyframes nsJumpFlash {
       0%   { box-shadow: 0 0 0 0 rgb(var(--gl-namespace-rgb) / 0.0); }
       12%  { box-shadow: 0 0 12px 2px rgb(var(--gl-namespace-rgb) / 0.55); }
       100% { box-shadow: 0 0 0 0 rgb(var(--gl-namespace-rgb) / 0.0); }
     }
     .jump-flash {
     animation: nsJumpFlash 1s ease-out;
     }
     /* Textual explanation paragraph under the title of an OR-
        introduction / OR-elimination subproof card. Subdued
        callout: pale background, left rule, slightly smaller text. */
     .subproof-explanation {
     margin: 0.4em 0 0.6em 0;
     padding: 0.55em 0.8em;
     background: var(--gl-tint);
     border-left: 2px solid var(--gl-text-faint);
     font-size: 0.92em;
     line-height: 1.45;
     color: var(--gl-text-soft);
     }
     .subproof-explanation b {
     color: var(--gl-namespace);
     }
     /* Inline symbolic example block — slightly tinted background +
        a hairline rule above and below to visually separate the
        formal-symbol example sentence from the surrounding prose. */
     .symbolic-example {
     display: inline-block;
     padding: 0.05em 0.45em;
     margin: 0 0.1em;
     background: rgb(var(--gl-namespace-rgb) / 0.07);
     border-radius: 3px;
     color: var(--gl-text-soft);
     }
     .symbolic-example i {
     font-style: normal;
     font-family: 'Cambria Math', 'STIX Two Math', 'Latin Modern Math', serif;
     font-weight: 600;
     letter-spacing: 0.02em;
     }
     /* The full-MPL expansion of an or<N> placeholder rendered in
        square brackets right after the symbolic name. Slightly
        de-emphasised vs the surrounding prose. */
     .or-expansion {
     color: var(--gl-text-quiet);
     font-size: 0.92em;
     }
     /* Step-badge as a permalink. The cursor change makes the
        affordance obvious; on hover the badge brightens slightly so
        the reader sees the click target before clicking. */
     a.step-badge-link {
     text-decoration: none;
     color: inherit;
     cursor: pointer;
     }
     a.step-badge-link:hover .step-badge {
     color: var(--gl-text-soft) !important;
     background: var(--gl-tint);
     border-radius: 3px;
     }
     /* Defensive wrapping for long expressions. The 13-conjunct
        disintegration sources can blow past the viewport width on
        narrow windows; allow break-word as a fallback so the line
        wraps inside the proof-line-content container instead of
        forcing horizontal scroll. */
     .proof-line-content {
     overflow-wrap: anywhere;
     word-break: break-word;
     }
     /* Expand-all / collapse-all subproof controls — small toggle
        bar above the Subproofs section title. */
     .subproof-controls {
     margin: 0.6em 0 0.4em 0;
     display: flex;
     gap: 0.5em;
     }
     .subproof-controls button {
     background: var(--gl-surface);
     color: var(--gl-text-soft);
     border: 1px solid var(--gl-rule);
     border-radius: 4px;
     padding: 0.25em 0.7em;
     cursor: pointer;
     font-family: inherit;
     font-size: 0.88em;
     }
     .subproof-controls button:hover {
     background: var(--gl-surface-hover);
     border-color: var(--gl-link);
     color: var(--gl-link);
     }
""" + _READABLE_CAPTION_CSS + """
     .validity-tag {
       color: var(--gl-namespace) !important;
       font-size: 0.78em;
       margin-left: 0.45em;
       white-space: nowrap;
       display: inline-block;
       font-weight: normal;
       line-height: 1;
       vertical-align: baseline;
     }
     a.proof-tag, a.proof-tag:link, a.proof-tag:visited {
       color: var(--gl-text);
       text-decoration: none;
       cursor: pointer;
       border-bottom: 1px dotted var(--gl-link);
     }
     a.proof-tag:hover {
       color: var(--gl-link);
     }

     .proof-section { margin-top: 0.8em; }
     .proof-section-title {
       font-size: 1.05em;
       font-weight: 700;
       color: var(--gl-text-muted);
       margin: 0.2em 0 0.8em 0;
       text-transform: uppercase;
       letter-spacing: 0.03em;
     }
     .main-proof-section { margin-bottom: 1.2em; }
     .subproofs-section { border-top: 1px solid var(--gl-rule); padding-top: 0.8em; }
     .subproof-card {
       margin: 0.8em 0 1.1em 0;
       padding: 0.7em 0.8em;
       background: var(--gl-surface);
       border: 1px solid var(--gl-rule);
       border-left: 4px solid var(--gl-link);
       border-radius: 6px;
     }
     /* Matryoshka depth — each nesting level shifts hue + indents.
        Depth 1 = direct subproof of main; deeper = sub-sub-…-proofs. */
     .subproof-depth-1 { border-left-color: var(--gl-link); background: var(--gl-surface); }
     .subproof-depth-2 { border-left-color: var(--gl-depth-2); background: var(--gl-depth-2-bg); margin-left: 0.6em; }
     .subproof-depth-3 { border-left-color: var(--gl-depth-3); background: var(--gl-depth-3-bg); margin-left: 1.2em; }
     .subproof-depth-4 { border-left-color: var(--gl-depth-4); background: var(--gl-depth-4-bg); margin-left: 1.8em; }
     .subproof-depth-5 { border-left-color: var(--gl-depth-5); background: var(--gl-depth-5-bg); margin-left: 2.4em; }
     .subproof-card[class*="subproof-depth-"]:not(.subproof-depth-1):not(.subproof-depth-2):not(.subproof-depth-3):not(.subproof-depth-4):not(.subproof-depth-5) {
       border-left-color: var(--gl-text-muted);
       background: var(--gl-depth-6-bg);
       margin-left: 3em;
     }
     .subproof-title {
       font-weight: 700;
       margin-bottom: 0.35em;
       color: var(--gl-text);
     }
     .subproof-label {
       color: var(--gl-text-muted);
       font-weight: 600;
       margin-left: 0.15em;
     }
     .subproof-meta {
       margin-bottom: 0.55em;
       color: var(--gl-text-muted);
       font-size: 0.95em;
     }
     .subproof-goal { color: var(--gl-link); }
     .proof-empty { color:var(--gl-text-faint); font-style: italic; }
     @media (max-width: 480px) {
       .validity-tag { white-space: normal; max-width: 100%; }
       .subproof-card { padding: 0.5em 0.35em; }
     }

    </style>
    """

    from proof_reader_html import STYLE as reader_style, notation_toggle_html, notation_preference_script, induction_part_numbers

    reader_navigation = notation_preference_script(any(is_reader_theorem(t[0]) for t in theorem_list))
    common_style += reader_navigation

    # --- Index page ---
    lean_export_view = lean_pages.load_lean_export(Path(out_dir), Path(proc_dir),
                                                   lean_kernel_check_running)
    lean_summary_html = lean_pages.render_index_summary(lean_export_view)
    index_head = f"""<!DOCTYPE html>
{LICENSE_SOURCE_COMMENT}
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Proof Graph – Index</title>
  <link rel="icon" type="image/png" href="favicon.png">
{LICENSE_HEAD_META}
  <style>
{proof_theme.THEME_TOKENS_CSS}
    body {{ font-family: var(--gl-font-ui); margin: 2rem; background: var(--gl-bg); color: var(--gl-text); }}
    /* One viewport of tail space, as on chapter pages: a notation switch at
       the end of the list keeps the reading position instead of clamping. */
    body::after {{ content: ""; display: block; height: 100vh; }}
    ul {{ list-style: none; padding: 0; }}
    ul ul {{ padding-left: 1.5rem; font-size: 0.9em; }}
    li {{ margin-bottom: 0.5em; }}
    a, a:link, a:visited {{ text-decoration: none; color: var(--gl-link); }}
    .clickable {{ cursor: pointer; }}
{_READABLE_CAPTION_CSS}
{reader_style if any(is_reader_theorem(t[0]) for t in theorem_list) else ''}
{proof_theme.THEME_SKIN_CSS}
  </style>
  {popup_script}
  {reader_navigation}
</head>
<body>
  {'<header class="reader-toolbar">' if any(is_reader_theorem(t[0]) for t in theorem_list) else ''}
  <nav><a href="tags.html">Reasoning rules</a></nav>
  <h1>Proof Graph</h1>
  {notation_toggle_html(any(is_reader_theorem(t[0]) for t in theorem_list))}
  {'</header>' if any(is_reader_theorem(t[0]) for t in theorem_list) else ''}
{lean_summary_html}
  <h2>Table of Contents</h2>
  <input type="text" id="theorem-search" placeholder="Filter theorems..."
         style="width:100%; max-width:600px; box-sizing:border-box; padding:0.5em; margin-bottom:1em; background:var(--gl-surface); color:var(--gl-text);
                border:1.5px solid var(--gl-rule-strong); border-radius:999px; padding:0.6em 1.1em; font-size:1em; font-family:inherit; outline:none;">
  <ul id="theorem-list">"""
    index_tail = f"""  </ul>
  <script>
  document.getElementById('theorem-search').addEventListener('input', function() {{
    const q = this.value.toLowerCase();
    document.querySelectorAll('#theorem-list > li').forEach(function(li) {{
      li.style.display = li.textContent.toLowerCase().includes(q) ? '' : 'none';
    }});
  }});
  </script>
{LICENSE_FOOTER}
</body>
</html>"""

    toc = []
    # build mapping from each displayed theorem title → its chapter file,
    # named by the book chapter number (book_chapter_numbers).
    theorem_to_file = {
        rename_theorem(name): f"chapter{chapter_nums[i]}.html"
        for i, (name, *_) in enumerate(theorem_list)
    }

    chapter_theorem_list_idx = [(name, method, var) for name, method, var in theorem_list]

    # Fuzzy theorem-link map for instantiated/broadcast theorem expressions (alpha-equivalent match).
    _shape_buckets = {}
    for i, (name, *_) in enumerate(chapter_theorem_list_idx):
        disp = rename_theorem(name)
        shape = _alpha_normalize_theorem_expr(disp)
        if not shape:
            continue
        _shape_buckets.setdefault(shape, set()).add(f"chapter{chapter_nums[i]}.html")
    theorem_shape_to_file = {
        shape: next(iter(files))
        for shape, files in _shape_buckets.items()
        if len(files) == 1
    }

    # Sibling-batch lookup map: build a per-sibling theorem -> URL map so a
    # cross-batch citation (e.g. an incubator chapter referencing a Peano-
    # batch external theorem) resolves to the sibling's chapter HTML page.
    # URLs are relative paths from this run's out_dir to the sibling's
    # out_dir. Entries are added to the global sibling_theorem_to_file /
    # sibling_theorem_shape_to_file maps so wrap_clickable's lookup can
    # find them. Same-pipeline theorems take precedence — sibling lookup
    # only fires when local lookup fails (see _resolve_sibling_theorem_target).
    sibling_theorem_to_file = {}
    sibling_theorem_shape_to_file = {}
    external_orphan_shapes.clear()

    # Load this pipeline's externals (the v/V + w/W forms saved by
    # process_proof_graphs.py to processed_proof_graph/external_theorems.txt).
    # Each entry's alpha-equivalent shape goes into external_orphan_shapes
    # so wrap_clickable can recognise an external citation that fails both
    # local and sibling lookup and render it as an orphan with a
    # dedicated right-click popup. The set is checked AFTER local and
    # sibling resolution, so externals that DO have a sibling chapter
    # never hit the orphan path.
    ext_file = Path(proc_dir) / "external_theorems.txt"
    if ext_file.exists():
        with open(ext_file, "r", encoding="utf-8") as f:
            for line in f:
                expr = line.strip()
                if not expr:
                    continue
                shape = _alpha_normalize_theorem_expr(expr)
                if shape:
                    external_orphan_shapes.add(shape)

    if sibling_graphs:
        sibling_shape_buckets: dict[str, set[str]] = {}
        for sib in sibling_graphs:
            sib_proc_dir = Path(sib["proc_dir"])
            sib_out_dir = Path(sib["out_dir"])
            sib_thms = read_theorem_list(sib_proc_dir / "global_theorem_list.txt")
            if not sib_thms:
                continue
            # The sibling book numbers its pages the same way.
            sib_chapter_nums = book_chapter_numbers(sib_thms)
            # Relative URL prefix from this run's out_dir to the sibling's
            # out_dir, e.g. ../../full_proof_graph from
            # files/incubator/full_proof_graph/.
            try:
                rel = os.path.relpath(sib_out_dir, out_dir).replace(os.sep, "/")
            except ValueError:
                rel = str(sib_out_dir).replace(os.sep, "/")
            for j, (sib_name, *_) in enumerate(sib_thms):
                sib_url = f"{rel}/chapter{sib_chapter_nums[j]}.html"
                disp = rename_theorem(sib_name)
                if disp not in sibling_theorem_to_file:
                    sibling_theorem_to_file[disp] = sib_url
                shape = _alpha_normalize_theorem_expr(disp)
                if shape:
                    sibling_shape_buckets.setdefault(shape, set()).add(sib_url)
        # Only keep alpha-shape entries that resolve to a unique sibling URL
        for shape, urls in sibling_shape_buckets.items():
            if len(urls) == 1 and shape not in theorem_shape_to_file:
                sibling_theorem_shape_to_file[shape] = next(iter(urls))

    for i, (name, method, _) in enumerate(chapter_theorem_list_idx):
        chapter_num = chapter_nums[i]
        filename = f"chapter{chapter_num}.html"
        theorem_display = rename_theorem(name)
        theorem_esc = html.escape(theorem_display, quote=True)
        display_stripped = html.escape(_strip_i_prefix(theorem_display), quote=True)
        # Populate data-parts so the right-click popup shows the GL
        # binary expansion of every constituent operator in the
        # theorem (AnchorXxx, in / in2 / in3, fXY / fXYZ, the
        # spontaneous compact-named operators, etc.). Same source-of-
        # truth as wrap_clickable's chapter-cell expansion.
        theorem_parts = html.escape(_extract_expr_parts(theorem_display), quote=True)
        parts_attr = f' data-parts="{theorem_parts}"' if theorem_parts else ''
        theorem_span = f'<span class="clickable mpl-only" data-text="{theorem_esc}"{parts_attr}>{display_stripped}</span>'
        toc.append(f"    <li>{chapter_num}. <a href='{filename}' style='text-decoration:none'>{theorem_span}</a>{lean_pages.toc_link(lean_export_view, i, chapter_num)}")
        # force a new line and style it
        if not debug:
            common_entry = _readable_theorem(rename_theorem(name))
            if is_reader_theorem(name):
                common_entry = f"<a class='common-index-link' href='{filename}'>{common_entry}</a>"
            toc.append(
                f"    <div class='readable-grey readable-index'>{common_entry}</div>"
            )

        if method.lower() in ("induction", "strong induction"):
            part_numbers = induction_part_numbers(chapter_num)
            toc.append("      <ul>")
            toc.append(f"        <li>{part_numbers[0]} <a href='{filename}#sub0'>Induction variable typing</a></li>")
            toc.append(f"        <li>{part_numbers[1]} <a href='{filename}#sub1'>Check for 0</a></li>")
            toc.append(f"        <li>{part_numbers[2]} <a href='{filename}#sub2'>Check induction condition</a></li>")
            toc.append("      </ul>")
        citing = sorted(used_by.get(chapter_num, set()))
        if citing:
            links = ", ".join(f"<a href='chapter{c}.html' style='color:var(--gl-goal); font-weight:bold;'>{c}</a>" for c in citing)
            toc.append(f"    <div style='margin-left:20px; color:var(--gl-text-faint); font-size:0.85em;'>Used by: {links}</div>")
        toc.append("    </li>")

    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(embed_page_definitions("\n".join([index_head] + toc + [index_tail]), gl_binary_map))

    # --- Tags reference page ---
    _generate_tags_page(out_dir, common_style + '<style>' + proof_theme.THEME_SKIN_CSS + '</style>')

    common_style += '<style>' + reader_style + '</style>'
    common_style += '<style>' + proof_theme.THEME_SKIN_CSS + '</style>'

    # --- Chapter pages ---
    chapter_theorem_list = [(name, method, var) for name, method, var in theorem_list]
    for i, (name, method, var) in enumerate(chapter_theorem_list):
        chapter_num = chapter_nums[i]
        filename = f"chapter{chapter_num}.html"
        prev_chapter_num = chapter_nums[i - 1] if i > 0 else None
        next_chapter_num = chapter_nums[i + 1] if i + 1 < len(chapter_theorem_list) else None
        prev_link = f"<a href='chapter{prev_chapter_num}.html'>Previous</a>" if prev_chapter_num is not None else ""
        next_link = f"<a href='chapter{next_chapter_num}.html'>Next</a>" if next_chapter_num is not None else ""
        lean_link = lean_pages.nav_link(lean_export_view, i, chapter_num)
        nav_links = ' '.join(link for link in (prev_link, next_link, lean_link) if link)

        head = f"""<!DOCTYPE html>
{LICENSE_SOURCE_COMMENT}
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{html.escape(_strip_i_prefix(rename_theorem(name)))}</title>
  {f'<title>{html.escape(_readable_theorem(rename_theorem(name), as_html=False))}</title>' if not debug else ''}
  <link rel="icon" type="image/png" href="favicon.png">
{LICENSE_HEAD_META}
  {common_style}
  {popup_script}
</head>
<body>
  {'<header class="reader-toolbar">' if is_reader_theorem(name) else ''}
  <div style="display:flex; justify-content:space-between; align-items:center;">
    <nav>
      <a href="index.html">Index</a> <a href="tags.html">Reasoning rules</a> {nav_links}
    </nav>
    <div style="display:grid; grid-template-columns:repeat(3, auto); column-gap:1.2em; row-gap:0.35em; justify-items:end; align-items:center;">
      <span style="color:var(--gl-goal); font-weight:bold;">Goal of the proof</span>
      <span style="color:var(--gl-link);">Has justification link</span>
      <span style="color:var(--gl-external); border-bottom:1px dotted var(--gl-external);">External (sibling pipeline) <span style="vertical-align:super; font-size:0.85em;">↗</span></span>
      <span style="color:var(--gl-external); border-bottom:1px dashed var(--gl-external);">External (no proof graph)</span>
      <a href="tags.html#readable-notation" style="color:var(--gl-text-muted); font-size:1.1em; text-decoration:none; border-bottom:1px dotted var(--gl-text-muted);">Common notation</a>
      <span style="color:var(--gl-namespace);">(Namespace)</span>
      <span style="color:var(--gl-target); font-weight:bold;">Integration goal</span>
      <span style="border-bottom:1px dotted var(--gl-link);"><b>Reasoning rule</b></span>
      <span>Right-click to expand</span>
    </div>
  </div>
  {notation_toggle_html(is_reader_theorem(name), proof_controls=True)}
  {'</header>' if is_reader_theorem(name) else ''}
  <h1>Chapter {chapter_num}: <span class="clickable mpl-only" data-text="{html.escape(rename_theorem(name), quote=True)}" data-parts="{html.escape(_extract_expr_parts(rename_theorem(name)), quote=True)}">{html.escape(_strip_i_prefix(rename_theorem(name)))}</span></h1>
  {f'''<div class="readable-grey readable-title">
    {_readable_theorem(rename_theorem(name))}
  </div>''' if not debug else ''}
"""

        body = [head]
        # "Used by" reverse-reference panel: list every other chapter
        # whose proof cites this chapter's theorem. Mirrors the
        # used_by line that already appears on the index TOC, but
        # rendered at the top of the chapter so a reader on chapter X
        # immediately sees who depends on X. Skip when no citers.
        citers = sorted(used_by.get(chapter_num, set()))
        if citers:
            citer_links = ", ".join(
                f"<a href='chapter{c}.html' style='color:var(--gl-goal); "
                f"font-weight:bold; text-decoration:none;'>{c}</a>"
                for c in citers
            )
            body.append(
                "  <div class='used-by-panel' style='margin: 0.4em 0 1em 0; "
                "padding: 0.55em 0.8em; background: rgb(var(--gl-goal-rgb) / 0.08); "
                "border-left: 2px solid var(--gl-goal); color: var(--gl-text-soft); "
                "font-size: 0.92em; line-height: 1.45;'>"
                f"<b style='color:var(--gl-goal);'>Used by:</b> {citer_links}"
                "</div>"
            )
        if method.lower() in ("induction", "strong induction"):
            part_prefixes = tuple(number + " " for number in induction_part_numbers(chapter_num)) if is_reader_theorem(name) else ("", "", "")
            body.extend([
                f"  <span class=\"var-highlight\">{'Strong induction' if method.lower() == 'strong induction' else 'Induction'} variable: {html.escape(var)}</span>",
                f"  <h2 id=\"sub0\">{part_prefixes[0]}Induction variable typing</h2>",
                f"  <div class=\"step-output\">{check_variable_typing(name, file_path_map[name][0], var, f'c{chapter_num}s0')}</div>",
                f"  <h2 id=\"sub1\">{part_prefixes[1]}Check for 0</h2>",
                f"  <div class=\"step-output\">{check_zero(name, file_path_map[name][1], var, f'c{chapter_num}s1')}</div>",
                f"  <h2 id=\"sub2\">{part_prefixes[2]}Check induction condition</h2>",
                f"  <div class=\"step-output\">{check_induction_condition(name, file_path_map[name][2], var, f'c{chapter_num}s2')}</div>",
            ])

        elif method.lower() == "single chapter proof":
            proof_heading = single_chapter_proof_heading(file_path_map[name][0])
            body.extend([
                f"  <h2>{proof_heading}</h2>",
                "  <div class='step-output'>", single_chapter_proof(name, file_path_map[name][0], f'c{chapter_num}s0' if is_reader_theorem(name) else ''), "  </div>",
            ])
        elif method.lower() == "debug":
            body.extend([
                "  <h2>Debugging</h2>",
                "  <div class='step-output'>", debugging(name, file_path_map[name][0], f'c{chapter_num}s0' if is_reader_theorem(name) else ''), "  </div>",
            ])
        elif method.lower() == "reformulated statement":
            body.extend([
                "  <h2>Reformulated</h2>",
                "  <div class='step-output'>",
                reformulated(name, file_path_map[name][0], f'c{chapter_num}s0' if is_reader_theorem(name) else ''),
                "  </div>",
            ])
        elif method.lower() == "incubator back reformulation":
            body.extend([
                "  <h2>Back-Reformulated</h2>",
                "  <div class='step-output'>",
                back_reformulated(name, file_path_map[name][0], f'c{chapter_num}s0' if is_reader_theorem(name) else ''),
                "  </div>",
            ])
        elif method.lower() == "or theorem":
            body.extend([
                "  <h2>OR Theorem</h2>",
                "  <div class='step-output'>",
                or_theorem(name, file_path_map[name][0], f'c{chapter_num}s0' if is_reader_theorem(name) else ''),
                "  </div>",
            ])
        elif method.lower() == "or elimination":
            body.extend([
                "  <h2>OR Elimination (pre-split merge)</h2>",
                "  <div class='step-output'>",
                or_elimination(name, file_path_map[name][0], f'c{chapter_num}s0' if is_reader_theorem(name) else ''),
                "  </div>",
            ])
        elif method.lower() == "proved not broadcast":
            body.extend([
                "  <h2>Proved, not broadcast (level-refused registration)</h2>",
                "  <div class='step-output'>",
                single_chapter_proof(name, file_path_map[name][0], f'c{chapter_num}s0' if is_reader_theorem(name) else ''),
                "  </div>",
            ])
        # Chapter statistics footer
        if is_reader_theorem(name):
            rendered = ''.join(body)
            reader_rows = sum(map(int, re.findall(r'data-main-rows="([0-9]+)"', rendered)))
            formal_rows = sum(map(int, re.findall(r'data-proof-steps="([0-9]+)"', rendered)))
            body.append(f'<div id="chapter-reader-counts" class="chapter-stats">Collapsed main rows: {reader_rows} · All proof steps: {formal_rows}</div>')
        stats_text = _compute_chapter_stats(*file_path_map[name])
        body.append(f"  <div class='chapter-stats'>{stats_text}</div>")

        body.append(LICENSE_FOOTER)
        body.append("</body>")
        body.append("</html>")

        with open(os.path.join(out_dir, filename), "w", encoding="utf-8") as f:
            f.write(embed_page_definitions("\n".join(body), gl_binary_map))

        # The chapter's Lean twin (only when the run exported this theorem).
        if lean_export_view is not None and i in lean_export_view.theorems:
            lean_pages.write_lean_page(
                Path(out_dir),
                lean_pages.render_lean_page(
                    lean_export_view.theorems[i], lean_export_view, chapter_num, filename,
                    html.escape(_strip_i_prefix(rename_theorem(name))),
                    _readable_theorem(rename_theorem(name)),
                    common_style, LICENSE_SOURCE_COMMENT, LICENSE_HEAD_META, LICENSE_FOOTER),
                chapter_num)
