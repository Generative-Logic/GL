# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Common-notation presentation of a lossless reader operation tree."""
import html
import re
from proof_reader import reader_plan, expanded_rows, reader_metrics
from proof_hierarchy import resolved_dependencies, resolved_citations, theorem_alpha_keys
from proof_export.mpl import Atom, Conjunction, Implication, Negation, parse_mpl
from visu_helpers import display_token, _HTML


STYLE = """
.reader-section .proof-line { align-items:flex-start; }
.readable-grey.readable-index { font-weight:700; }
.readable-index a.common-index-link, .readable-index a.common-index-link:link,
.readable-index a.common-index-link:visited { color:inherit; font-weight:inherit; }
.common-line { display:flex; align-items:baseline; gap:0.7em; width:100%; }
.common-citations { flex:0 1 auto; min-width:2.5em; max-width:100%; margin-left:auto; text-align:right; font-size:0.85em; }
.common-citations a { white-space:nowrap; }
.common-citations a { margin-right:0.35em; }
.common-step-badge { display:none; white-space:nowrap; font-size:0.75em; font-weight:600; flex:0 0 auto; }
.common-notation-only .common-step-badge { display:block; color:inherit; }
.common-notation-only .common-line { color:var(--gl-text); }
.common-notation-only .common-citations a { color:inherit; }
.common-notation-only .rf-reads-label { display:none; }
.rf-target-label { color:var(--gl-target); font-weight:800; }
.reader-prose { display:block; margin-bottom:0.2em; color:var(--gl-prose); font-weight:700; font-family:var(--gl-font-ui); }
.common-notation-only .reader-prose { color:var(--gl-label); }
.common-line .readable-grey { flex:1; min-width:0; }
.reader-section .subproof-title { display:block; }
.reader-section .reader-block-kind { color:var(--gl-link-strong); font-size:1.12em; font-weight:800; }
.reader-section .reader-label { display:block; margin:0.6em 0 0.25em; color:var(--gl-label); font-size:1em; font-weight:800; letter-spacing:0.015em; }
.reader-section .proof-line, .reader-section .subproof-title, h2[id] { scroll-margin-top:calc(var(--reader-toolbar-height, 0px) + 1rem); }
.reader-section .subproof-title .readable-grey { display:block; margin-left:1.6em; }
.reader-section .subproof-card.collapsed > .subproof-body { display:none; }
.reader-section .subproof-card:not(.collapsed) > .subproof-body { display:block; }
.reader-section .subproof-card:not(.collapsed) > .subproof-title .subproof-toggle { transform:none; }
.reader-section .subproof-title .mpl-line { font-weight:400; line-height:1.6; overflow-wrap:anywhere; white-space:normal; }
.reader-section .goal-mpl { color:var(--gl-goal); font-weight:bold; }
.reader-section .goal-common { font-weight:bold; }
.reader-section .reader-witnesses { display:block; white-space:nowrap; max-width:100%; overflow-x:auto; }
.reader-section .block-products { display:flex; flex-direction:column; gap:0.3em; }
.reader-section .block-number { color:var(--gl-prose); margin-right:0.7em; font-variant-numeric:tabular-nums; }
.common-notation-only .mpl-line, .common-notation-only .mpl-only,
.common-notation-only .step-badge-link, .common-notation-only .step-badge,
.common-notation-only .validity-tag { display:none !important; }
.common-notation-only .readable-grey { color:var(--gl-text-common); font-size:1.08em; font-weight:500; line-height:1.8; }
.common-notation-only .readable-grey.readable-index, .common-notation-only .readable-grey.readable-title { font-weight:600; }
.common-notation-only .readable-title { font-size:1.8em; }
.common-notation-only .goal-common { font-weight:700; }
.common-notation-only .proof-line { padding:0.65em 0; }
.common-notation-only .rf-label { color:var(--gl-prose); }
.common-notation-only .proof-line-content { width:100%; }
.reader-page-controls { display:flex; flex-wrap:wrap; gap:0.75rem; margin:0.55rem 0; }
body:not(:has(.reader-section .subproof-card)) .reader-page-controls { display:none; }
.reader-page-control, .subproof-controls button { border:2px solid var(--gl-link); border-radius:6px; padding:0.6rem 1rem; background:var(--gl-control-bg); color:var(--gl-text); font-size:1rem; font-weight:750; cursor:pointer; }
.reader-page-control:hover, .subproof-controls button:hover { background:var(--gl-control-bg-hover); }
.reader-page-control:focus-visible, .subproof-controls button:focus-visible { outline:2px solid var(--gl-goal); outline-offset:3px; }
.reader-toolbar { position:sticky; top:0; z-index:100; background:var(--gl-bg); padding:0.6rem 0; box-shadow:0 4px 12px #0006; }
.reader-toolbar > div:first-child { flex-wrap:wrap; gap:0.8em; }
.reader-toolbar .common-notation-control { margin:0.8em 0 0.25em; }
@media (max-width:760px) {
  .reader-toolbar > div:first-child > div { grid-template-columns:repeat(2, minmax(0,1fr)) !important; justify-items:start !important; font-size:0.8em; }
}
@media (max-width:480px) {
  .common-notation-only .readable-title { font-size:1.3em; }
  .common-line { flex-wrap:wrap; }
  .common-line .readable-grey { flex:1 0 100%; }
  .reader-section .proof-line { flex-wrap:wrap; }
  .reader-section .proof-line > .step-badge-link, .reader-section .proof-line > .proof-line-content { flex:1 0 100%; }
  .common-citations { flex:1 0 100%; }
  .reader-section .reader-block-kind { display:inline-block; }
}
.common-notation-control { display:flex; align-items:center; gap:0.9em; margin:1.2em 0; padding:0.9em 1.1em; max-width:44em; border:2px solid var(--gl-link); border-radius:9px; background:var(--gl-control-bg); color:var(--gl-text); font-size:1.12em; font-weight:700; cursor:pointer; }
.common-notation-control input { width:1.4em; height:1.4em; accent-color:var(--gl-link); cursor:pointer; }
.common-notation-control strong { display:block; font-size:1.15em; color:var(--gl-link-strong); }
.common-notation-control .notation-invitation { display:block; margin-top:0.2em; font-size:0.9em; font-weight:500; }
.common-notation-control:focus-within { outline:2px solid var(--gl-text); outline-offset:3px; }
.reader-section .reader-counts { color:var(--gl-prose); margin:0.7em 0; }
"""


def induction_part_numbers(chapter_number):
    """
    @brief Number induction parts beneath their reader chapter.

    @details Typing, the zero case and the induction condition use the same
    parent chapter and consecutive one-based parts, independent of file indices.

    @param chapter_number Nonnegative integer identifying the reader chapter.
    @return Three displayed numbers for typing, zero and induction condition.
    @invariant Physical proof filenames and fragment identifiers stay unchanged.
    """
    assert isinstance(chapter_number, int) and chapter_number >= 0
    return tuple(f"{chapter_number}.{part}" for part in (1, 2, 3))


_CHAPTER_PAGE = re.compile(r"chapter(\d+)\.html")


def theorem_citation_label(url):
    """
    @brief Name the chapter a theorem citation leads to.

    @details A citation of an earlier theorem reads as the chapter that proves
    it, for example "chapter 12", so the reader sees where the link goes
    before following it. A theorem without a chapter page in this book has no
    link; it keeps the word "theorem".

    @param url Chapter page of the cited theorem, ``chapterN.html``, or None
    when the book has no page for it.
    @return "chapter N" for a chapter page, "theorem" for None.
    @invariant Every theorem page of the book is named ``chapterN.html``.
    """
    if url is None:
        return "theorem"
    match = _CHAPTER_PAGE.fullmatch(url)
    assert match, f"theorem citation outside a chapter page: {url}"
    return f"chapter {match.group(1)}"


def notation_preference_script(enabled):
    """
    @brief Carry the selected notation through navigation within one proof book.

    @details The explicit common/mixed URL parameter travels with book links,
    including index, chapter, Lean and tag pages. It works for local files without
    assuming shared browser storage. Fragments, other query fields, downloads and
    links outside the book are preserved. Checkbox and body state are restored
    on load and browser history traversal. Mode changes retain the viewport
    position of the first reading line that begins below the floating header:
    a proof row, a block header, an index entry, or a common-notation line
    inside a row or block header, which both modes show; a line whose top lies
    behind the header would vanish behind it when the mode shortens it, and a
    tall header filling the screen still offers its inner lines. Only when no
    line begins in view does the innermost line reaching below the header's
    lower edge serve. A switch back without scrolling in between keeps the
    same line, so a round trip returns to the starting view.

    @param enabled Whether this generated book supports common notation.
    @return Initialization script, or no script for a legacy book.
    """
    if not enabled:
        return ""
    return r"""<script id="common-notation-navigation">
document.addEventListener('DOMContentLoaded', function() {
  const parameter = 'notation';
  const directory = new URL('.', window.location.href).href;
  const control = document.getElementById('common-notation-toggle');
  const toolbar = control ? control.closest('.reader-toolbar') : null;
  if (control && !toolbar) throw new Error('Notation control requires its reader toolbar');

  function updateToolbarSpacing() {
    if (toolbar) document.documentElement.style.setProperty('--reader-toolbar-height', toolbar.getBoundingClientRect().height + 'px');
  }

  function applyNotation(common, updateAddress) {
    document.body.classList.toggle('common-notation-only', common);
    if (control) control.checked = common;
    updateToolbarSpacing();
    const mode = common ? 'common' : 'mixed';
    if (updateAddress) {
      const address = new URL(window.location.href);
      address.searchParams.set(parameter, mode);
      window.history.replaceState(window.history.state, '', address.href);
    }
    document.querySelectorAll('a[href]').forEach(function(link) {
      const href = link.getAttribute('href');
      if (href.startsWith('#') || link.hasAttribute('download')) return;
      const target = new URL(href, document.baseURI);
      if (new URL('.', target).href !== directory) return;
      const filename = target.pathname.slice(target.pathname.lastIndexOf('/') + 1);
      if (!/^(?:index|tags|chapter\d+(?:_lean)?)\.html$/.test(filename)) return;
      target.searchParams.set(parameter, mode);
      link.setAttribute('href', target.href);
    });
  }

  function restoreNotation() {
    const mode = new URL(window.location.href).searchParams.get(parameter);
    if (mode !== null && mode !== 'common' && mode !== 'mixed') {
      throw new Error('Unknown proof notation mode: ' + mode);
    }
    applyNotation(mode === 'common', false);
  }

  const readingLines = '.reader-section .proof-line, .reader-section .proof-line .common-line, '
    + '.reader-section .subproof-title, .reader-section .subproof-title > .readable-grey > *, '
    + '.reader-section .subproof-title > .readable-grey .block-products > *, #theorem-list > li';
  let pinned = null;
  if (control) control.addEventListener('change', function() {
    const toolbarBottom = toolbar.getBoundingClientRect().bottom;
    let reading = null;
    if (pinned && pinned.scrollY === window.scrollY && pinned.node.getBoundingClientRect().height > 0) {
      reading = {node: pinned.node, top: pinned.node.getBoundingClientRect().top};
    } else {
      let covering = null;
      for (const node of document.querySelectorAll(readingLines)) {
        const bounds = node.getBoundingClientRect();
        if (bounds.height === 0 || bounds.bottom <= toolbarBottom) continue;
        if (bounds.top >= window.innerHeight) break;
        if (bounds.top >= toolbarBottom) {
          reading = {node: node, top: bounds.top};
          break;
        }
        covering = {node: node, top: bounds.top};
      }
      if (reading === null) reading = covering;
    }
    applyNotation(control.checked, true);
    if (reading) window.scrollBy(0, reading.node.getBoundingClientRect().top - reading.top);
    pinned = reading ? {node: reading.node, scrollY: window.scrollY} : null;
  });
  window.addEventListener('resize', updateToolbarSpacing);
  window.addEventListener('pageshow', restoreNotation);
  window.addEventListener('popstate', restoreNotation);
  restoreNotation();
});
</script>"""


def notation_toggle_html(enabled, proof_controls=False):
    """
    @brief Invite readers to switch the index or chapter to common notation.

    @details The complete label is clickable and keyboard focus is visible.
    Unsupported books deliberately retain their established controls.

    @param enabled Whether this page belongs to a supported proof book.
    @param proof_controls Whether to include whole-chapter expand/collapse buttons.
    @return Accessible checkbox and optional chapter actions, or no unsupported controls.
    """
    if not enabled:
        return ""
    toggle = (
        '<label class="common-notation-control">'
        '<input id="common-notation-toggle" type="checkbox" '
        'aria-label="Common notation only">'
        "<span><strong>Common notation</strong>"
        '<span class="notation-invitation">Click to read common notation only</span></span></label>'
    )
    if proof_controls:
        toggle += '<div class="reader-page-controls" role="group" aria-label="Chapter proof blocks">'
        for action, label, state in (("expand", "Expand all", "true"), ("collapse", "Collapse all", "false")):
            toggle += (
                f'<button type="button" class="reader-page-control" data-proof-action="{action}" '
                f'title="{label} proof blocks throughout this chapter" '
                f'onclick="document.querySelectorAll(&quot;.reader-section .subproof-card&quot;).forEach(c=>window.setProofCardOpen(c,{state}))">{label}</button>')
        toggle += '</div>'
    return toggle


CONTRADICTION_SEED_PROSE = "Assumed for the proof by contradiction: the negation of what is proved."

READER_PROSE = {
    "task formulation": "Given in this proof.",
    "recursion": "From the induction hypothesis.",
    "or branch assumption": "Assumed as a negated premise for this OR-introduction case.",
    "or disintegration": "Assumed as this case of the disjunction.",
    "implication": "Derived by applying the rule to the cited premises.",
    "expansion": "Obtained by unfolding the definition.",
    "disintegration": "Extracted from the cited expression.",
    "equality1": "Obtained by substitution using the cited equalities.",
    "equality2": "Obtained by combining the cited equalities.",
    "symmetry of equality": "The cited equality is reversed.",
    "symmetry of inequality": "The cited inequality is reversed.",
    "contradiction": "The cited expressions contradict each other.",
    "vacuous truth": "The implication follows because its premises are contradictory.",
    "contraposition": "Obtained by contraposition.",
    "implication composition": "Obtained by composing the cited implications.",
    "implication grouping": "The premises of the cited implication are grouped.",
    "compilation": "The expression is written using its defined name.",
    "multiplied from": "An instance of the cited theorem.",
    "variable copy": "Copied with the recorded variable substitution.",
    "zoom in": "The cited result is inherited in this scope.",
    "validity name": "Obtained as a result of a subproof.",
    "premise element": "A premise supplied for the construction.",
    "expansion for integration": "The definition is unfolded for the goal.",
    "reformulated from": "A reformulation of the cited result.",
    "incubator back reformulation": "The cited equality is expressed in operator notation.",
    "or branch proven": "The disjunction follows from the proved branch.",
    "or convergence": "The cases yield the same conclusion.",
    "or theorem": "The resulting disjunction.",
    "or elimination": "The conclusion follows by combining the cases.",
    "anchor handling": "The anchor is specialized as recorded.",
    "definition projection": "A defining property extracted from the cited expression.",
    "theorem": "Previously established theorem.",
    "externally provided theorem": "Supplied theorem."
}


def contradiction_seeds(rows):
    """
    @brief Collect the hypothesised seeds of the section's proofs by contradiction.

    @details Two row layouts exist (verifier ``check_contradiction``). A
    main-scope contradiction row proves the negation of its seed and cites the
    seed as its fifth field with its namespace as the sixth; the seed is a
    ``task formulation`` row at ``main``. A scope-carrying row has exactly four
    cited fields and refutes the assumption of the marked child scope named by
    its second field: for a ``contradiction_<!S>`` payload the row proves
    ``!S`` and the seed ``S`` is a ``task formulation`` row of that child
    scope; an ``ordis_`` payload refutes an OR case, whose assumption has its
    own case reading and is no seed. A seed is an assumption to be refuted, so
    its reading must not say that it is given.

    @param rows Complete processed proof section.
    @return Set of ``(expression, namespace)`` pairs of the seeds.
    @invariant A main-scope seed is cited with the ``main`` namespace; a
               scope-carrying row names a ``_boundary_`` child scope whose
               payload is ``contradiction_`` or ``ordis_``.
    """
    seeds = set()
    for row in rows:
        if row[2] != "contradiction":
            continue
        if len(row) >= 9:
            assert row[8] == "main", f"Contradiction seed outside main: {row}"
            seeds.add((row[7], row[8]))
            continue
        assert len(row) == 7, f"Contradiction row with {len(row) - 3} cited fields: {row}"
        scope = row[4]
        cut = scope.rfind("_boundary_")
        assert cut >= 0, f"Scope-carrying contradiction outside a boundary scope: {row}"
        payload = scope[cut + len("_boundary_"):]
        if payload.startswith("contradiction_"):
            assert payload == "contradiction_" + row[0], f"Contradiction scope does not match its result: {row}"
            seeds.add((row[0][1:] if row[0].startswith("!") else "!" + row[0], scope))
        else:
            assert payload.startswith("ordis_"), f"Unknown contradiction scope payload: {row}"
    return seeds


def complete_caption(row, readable, contradiction_seed=False):
    """
    @brief Render common notation for every displayed formal row.

    @details Rule applications retain their rule and substitutions as well as
    their result. Plain relations and structural scope rows also have a reading.
    Integration targets keep an explicit TO PROVE label in both notation modes.
    The seed of a proof by contradiction reads as an assumption, not as given.

    @param row One complete formal proof row.
    @param readable Graph-specific mathematical notation renderer.
    @param contradiction_seed Whether the row is a contradiction seed
           (``contradiction_seeds``); only a ``task formulation`` row can be one.
    @return Complete common-notation HTML, without a per-line number.
    """
    assert not contradiction_seed or row[2] == "task formulation", f"Contradiction seed with tag {row[2]}"
    expression = row[0].removesuffix("_integration_goal")
    reading = readable.formula(expression)
    if row[0].endswith("_integration_goal"):
        return f"<span class='rf-label rf-target-label reader-prose'>TO PROVE:</span>{reading}"
    result = f"<span class='rf-label rf-reads-label'>READS:</span>{reading}"
    if row[2] == "implication":
        result = readable.rule_application(row[0], row[3], row[5::2]) + "<br>" + result
    elif row[2] in ("reformulated from", "incubator back reformulation"):
        result += (
            "<br><span class='rf-label'>REFORMULATED FROM:</span>"
            + readable.formula(row[3].removesuffix("_integration_goal"))
        )
    if row[2].startswith("reformulation for integration"):
        prose = "The construction is established from the cited components."
    elif contradiction_seed:
        prose = CONTRADICTION_SEED_PROSE
    else:
        assert row[2] in READER_PROSE, f"Missing common-notation prose for {row[2]}"
        prose = READER_PROSE[row[2]]
    return f'<span class="reader-prose">{prose}</span>' + result


def _free_witness_arguments(expression):
    """
    @brief Collect free argument names with exact lexical binder scope.

    @details A binder covers its own implication premise and conclusion. Names
    used only inside a nested binder are not uses of an outer witness. The parser
    supplies immutable formula nodes; no printed-name comparison is involved.

    @param expression One parsed formal expression.
    @return Set of free argument tokens, possibly empty for a closed formula.
    """
    result = set()
    pending = [(expression, frozenset())]
    while pending:
        node, bound = pending.pop()
        if isinstance(node, Atom):
            result.update(
                argument for argument in node.arguments if argument not in bound
            )
        elif isinstance(node, Conjunction):
            pending.extend((part, bound) for part in node.parts)
        elif isinstance(node, Negation):
            pending.append((node.inner, bound))
        else:
            assert isinstance(node, Implication), "Unknown formal expression"
            nested = bound | frozenset(node.bound_variables)
            pending.extend(((node.premise, nested), (node.conclusion, nested)))
    return result


def disintegration_witnesses(rows, block):
    """
    @brief List the actual existential witnesses used by a disintegration block.

    @details A witness is a bound name of a negated implication source that
    becomes free in a retained projection result. Unused bound names and names
    rebound inside the result are excluded. Positive universal implications do
    not introduce witnesses. The complete operation is scanned in proof order,
    retaining each name once; internal as well as exported products are evidence.

    @param rows Complete reverse-stored processed proof section.
    @param block One disintegration reader node with its original members.
    @return Deterministically ordered witness names for a single header row.
    @invariant No witness is inferred from common notation or invented by a header.
    """
    assert block["kind"] == "disintegration"
    names = []
    seen = set()
    for index in sorted(block["members"], reverse=True):
        row = rows[index]
        assert row[1] == block["namespace"], "A disintegration block cannot mix scopes"
        if row[2] != "disintegration":
            continue
        assert len(row) == 5, "Disintegration requires its single formal source"
        source = parse_mpl(row[3])
        if not isinstance(source, Negation) or not isinstance(
            source.inner, Implication
        ):
            continue
        bound = []
        cursor = source.inner
        while isinstance(cursor, Implication):
            bound.extend(cursor.bound_variables)
            cursor = cursor.conclusion
        used = _free_witness_arguments(parse_mpl(row[0]))
        for name in bound:
            if name in used and name not in seen:
                names.append(name)
                seen.add(name)
    return names


def render_reader_proof(rows, prefix, generator, known_theorems, known_theorem_keys=None):
    """
    @brief Render the complete checked proof as nested collapsible operations.

    @details Public products have header anchors for formal citations.
    Common-notation citations always target the retained derivation row, allowing
    navigation to reveal it through every enclosing collapsed block.
    Common notation owns its right citation column and left row/total badge.

    @param rows Complete processed proof section in reverse storage order.
    @param prefix Unique chapter and section identifier.
    @param generator HTML module providing graph-specific notation and theorem links.
    @param known_theorems Complete theorem registry including external statements.
    @param known_theorem_keys Alpha keys of exactly that registry, computed once by a
           caller that renders many chapters; computed here when not supplied.
    @return HTML section with formal rows, common notation and reader-only headers.
    @invariant Every non-import proof row has exactly one rendered row and caption.
    """
    if known_theorem_keys is None:
        known_theorem_keys = theorem_alpha_keys(known_theorems)
    edges = resolved_dependencies(rows, known_theorems, known_theorem_keys)
    plan = reader_plan(rows, edges)
    order = expanded_rows(plan)
    assert order == list(
        range(len(rows) - 1, -1, -1)
    ), "HTML tree differs from processed order"
    readable = generator._readable_form_for_render
    citation_sources = resolved_citations(rows, known_theorems, known_theorem_keys)
    seeds = contradiction_seeds(rows)
    visible = [i for i in order if rows[i][2] != "theorem"]
    numbers = {i: j + 1 for j, i in enumerate(visible)}
    anchors = {i: f"{prefix}-entry{i}" for i in visible}
    public = {}
    scopes = {}
    scope_titles = {}
    _, parsed_scopes = generator._partition_stack_subproofs(rows)
    pending_scopes = list(parsed_scopes)
    while pending_scopes:
        scope = pending_scopes.pop()
        scope_titles[scope["ns"]] = scope["title_info"]
        pending_scopes.extend(scope["nested_subproofs"])
    pending = [plan]
    while pending:
        node = pending.pop()
        if node["kind"] == "row":
            continue
        if node["kind"] in ("root", "subproof", "orint", "ordis"):
            info = scope_titles.get(node["namespace"], {})
            goal_expression = info.get("goal_expr")
            if info.get("goal_kind") == "contradiction":
                scopes[node["namespace"]] = None
            elif goal_expression:
                scopes[node["namespace"]] = next(
                    (
                        i
                        for i in sorted(node["members"])
                        if rows[i][1] == node["namespace"]
                        and generator._norm_expr(rows[i][0])
                        == generator._norm_expr(goal_expression)
                    ),
                    None,
                )
            else:
                scopes[node["namespace"]] = node["goal"]
        pending.extend(node["children"])
    achieved_goals = {index for index in scopes.values() if index is not None}
    refutation_receipts = {}
    for namespace, info in scope_titles.items():
        if info.get("goal_kind") == "contradiction":
            receipts = [index for index, row in enumerate(rows)
                        if row[1] == info["refutation_namespace"]
                        and row[2] == "contradiction" and row[0] == info["refuted_case"]]
            assert len(receipts) == 1, "A refuted case must have one recorded closing receipt"
            refutation_receipts[namespace] = receipts[0]
            achieved_goals.add(receipts[0])
    match = re.fullmatch(r"c(\d+)s(\d+)", prefix)
    section = f"{match[1]}.{int(match[2]) + 1}" if match else prefix or "1"

    def assign(node, number):
        """
        @brief Assign hierarchical block numbers and exported-product anchors.

        @details Row identities remain independent of presentation numbering.

        @param node Reader node to annotate.
        @param number Hierarchical section number.
        @return None.
        """
        node["number"] = number
        node["anchor"] = f"{prefix}-block-{number.replace('.', '-')}"
        for index in node.get("outputs", []):
            public.setdefault(index, []).append(node)
        ordinal = 0
        for child in node.get("children", []):
            if child["kind"] != "row":
                ordinal += 1
                assign(child, f"{number}.{ordinal}")

    assign(plan, section)

    def target(index, consumer):
        """
        @brief Resolve a local citation to its row or closed public header.

        @details A consumer inside a block can inspect its own primitive rows.
        Consumers outside land on the outermost header exporting that product.

        @param index Producer row index.
        @param consumer Consumer row index, or None for a header input.
        @return Local fragment or external theorem URL.
        """
        if rows[index][2] == "theorem":
            return generator._resolve_theorem_target(rows[index][0])
        for block in public.get(index, []):
            if consumer not in block["members"]:
                return "#" + block["anchor"]
        return "#" + anchors[index]

    def citations(row, consumer):
        """
        @brief Collect the row's own proof citations with independent link labels.

        @details Integration templates are targets, not derived proof premises;
        unproved optional OR companions have no fabricated proof link.

        @param row Formal row whose dependency cells are displayed.
        @param consumer Consumer row index.
        @return Ordered tuples of expression, namespace, formal URL, label and common URL.
        """
        result = []
        for (expression, namespace), (kind, index) in zip(
            zip(row[3::2], row[4::2]), citation_sources[consumer], strict=True
        ):
            if kind == "local":
                url = target(index, consumer)
                label = str(numbers[index]) if index in numbers else theorem_citation_label(url)
            else:
                assert kind in ("theorem", "template", "optional")
                url = (
                    generator._resolve_theorem_target(expression)
                    if kind == "theorem"
                    else None
                )
                label = theorem_citation_label(url)
            common_url = (
                "#" + anchors[index] if kind == "local" and index in anchors else url
            )
            result.append((expression, namespace, url, label, common_url))
        return result

    def mpl(expression):
        """
        @brief Render an expandable MPL expression without an implicit outer link.

        @details All proof citation destinations are assigned explicitly.

        @param expression Exact formal expression.
        @return Escaped interactive MPL span.
        """
        escaped = html.escape(expression, quote=True)
        parts = html.escape(generator._extract_expr_parts(expression), quote=True)
        return (
            f'<span class="clickable" data-text="{escaped}" data-parts="{parts}">'
            f"{html.escape(generator._strip_i_prefix(expression))}</span>"
        )

    def leaf(index):
        """
        @brief Render one retained formal row and its common-notation counterpart.

        @details Imports are intentionally represented by external citations only.

        @param index Index of a non-import row in the complete graph.
        @return One proof-line element, or empty text for a theorem import.
        """
        row = rows[index]
        if row[2] == "theorem":
            return ""
        refs = citations(row, index)
        links = " ".join(
            f'<a href="{html.escape(common_url, quote=True)}" title="{html.escape(expr, quote=True)}">({label})</a>'
            for expr, ns, url, label, common_url in refs
            if common_url
        )
        ingredients = []
        for expr, ns, url, label, common_url in refs:
            text = mpl(expr)
            if url:
                text = f'<a href="{html.escape(url, quote=True)}">{text}</a>'
            ingredients.append(text + generator._format_validity_tag(ns))
        tag = row[2].lower().replace(" ", "-").replace("(", "").replace(")", "")
        goal = index in achieved_goals
        goal_class = " goal-mpl" if goal else ""
        common_class = " goal-common" if goal else ""
        deps = " ".join(
            url[1:] for _, _, url, _, _ in refs if url and url.startswith("#")
        )
        return (
            f'<div class="proof-line" id="{anchors[index]}" data-row-index="{index}" data-deps="{deps}">'
            f'<a class="step-badge-link" href="#{anchors[index]}"><span class="step-badge">({numbers[index]}/{len(visible)})</span></a>'
            '<div class="proof-line-content"><div class="mpl-line">'
            f'<span class="{goal_class.strip()}">{mpl(row[0])}</span>{generator._format_validity_tag(row[1])} '
            f'<a class="proof-tag" data-tag="{html.escape(row[2], quote=True)}" href="tags.html#{tag}"><b>{html.escape(row[2])}</b></a> '
            + " &nbsp; ".join(ingredients)
            + '</div><div class="common-line">'
            + f'<span class="common-step-badge">({numbers[index]}/{len(visible)})</span>'
            + f'<div class="readable-grey{common_class}">{complete_caption(row, readable, row[2] == "task formulation" and (row[0], row[1]) in seeds)}</div>'
            + f'<span class="common-citations">{links}</span></div></div></div>'
        )

    def render(node):
        """
        @brief Render a semantic node recursively without flattening nested scopes.

        @details Each summary exposes the goal or externally consumed products;
        its body retains every original primitive row in the planned order.

        @param node Reader node with assigned section number and anchor.
        @return Nested row or collapsed card HTML.
        """
        kind = node["kind"]
        if kind == "row":
            return leaf(node["index"])
        body = "".join(render(child) for child in node["children"])
        if kind == "root":
            return body
        label = {
            "integration": "Integration",
            "disintegration": "Disintegration",
            "orint": "OR introduction",
            "ordis": "OR case",
            "subproof": "Subproof",
        }[kind]
        outputs = node.get(
            "outputs", [node["goal"]] if node.get("goal") is not None else []
        )
        output_mpl, output_common = [], []
        for index in outputs:
            expression = rows[index][0].removesuffix("_integration_goal")
            goal = index in achieved_goals
            output_mpl.append(
                f'<span class="{"goal-mpl" if goal else ""}">{mpl(expression)}</span>'
            )
            output_common.append(
                f'<span class="{"goal-common" if goal else ""}">{readable.formula(expression)}</span>'
            )
        formal = '<span class="block-products">' + "".join(output_mpl) + "</span>"
        common = '<span class="block-products">' + "".join(output_common) + "</span>"
        info = scope_titles.get(node["namespace"], {})
        if kind in ("orint", "ordis", "subproof") and info.get("goal_expr"):
            goal_expression = info["goal_expr"].removesuffix("_integration_goal")
            formal = mpl(goal_expression)
            common = readable.formula(goal_expression)
        witnesses = (
            disintegration_witnesses(rows, node) if kind == "disintegration" else []
        )
        if kind == "disintegration":
            source, namespace = node["source"]
            witness_formal = witness_common = ""
            if witnesses:
                witness_data = html.escape(" ".join(witnesses), quote=True)
                witness_label = (
                    '<span class="reader-label">Witnesses used in proof:</span>'
                )
                witness_open = (
                    f'<span class="reader-witnesses" data-witnesses="{witness_data}">'
                )
                witness_formal = (
                    witness_label
                    + witness_open
                    + ", ".join(html.escape(name) for name in witnesses)
                    + "</span>"
                )
                witness_common = (
                    witness_label
                    + witness_open
                    + ", ".join(display_token(name, _HTML) for name in witnesses)
                    + "</span>"
                )
            # A block whose products nothing outside uses lists no products, and
            # shows no products label above an empty list.
            products_label = (
                '<span class="reader-label">Products of disintegration used in proof:</span>'
                if outputs else ""
            )
            formal = (
                '<span class="reader-label">From</span> '
                + mpl(source)
                + witness_formal
                + (products_label + formal if outputs else "")
            )
            common = (
                '<span class="reader-label">From</span> '
                + readable.formula(source)
                + witness_common
                + (products_label + common if outputs else "")
            )
        else:
            formal = '<span class="reader-label">Goal:</span> ' + formal
            common = '<span class="reader-label">Goal:</span> ' + common
            if kind in ("orint", "ordis"):
                assert "case_expressions" in info and "or_expr" in info
                if info.get("goal_kind") == "contradiction":
                    formal = '<span class="reader-label">Goal:</span> Contradiction (case refuted)'
                    common = '<span class="reader-label">Goal:</span> &#8869; (case refuted)'
                    receipt = refutation_receipts[node["namespace"]]
                    closing = (
                        '<span class="reader-prose">The contradiction closes this case at '
                        f'<a href="#{anchors[receipt]}">step {numbers[receipt]}</a>.</span>')
                    formal += closing
                    common += closing
                assumptions = info["case_expressions"]
                case_formal = ", ".join(mpl(expression) for expression in assumptions)
                case_common = ", ".join(readable.formula(expression) for expression in assumptions)
                case_prose = "For this case, assume the negated premises:" if kind == "orint" else "Assume this case:"
                if not assumptions:
                    case_prose = "No additional case assumption."
                formal = (
                    '<span class="reader-label">' + case_prose + '</span> '
                    + case_formal
                    + '<span class="reader-label">Disjunction:</span> ' + mpl(info["or_expr"])
                    + "<br>" + formal)
                common = (
                    '<span class="reader-label">' + case_prose + '</span> '
                    + case_common
                    + '<span class="reader-label">Disjunction:</span> ' + readable.formula(info["or_expr"])
                    + "<br>" + common)
            if kind in ("orint", "ordis", "subproof"):
                formal += generator._subproof_explanation(info)
        return (
            f'<div class="subproof-card collapsed reader-block" data-block-kind="{kind}" data-block-witnesses="{html.escape(" ".join(witnesses), quote=True)}" data-formal-rows="{len(node["members"])}">'
            f'<div class="subproof-title" id="{node["anchor"]}" data-closed-result="true" role="button" tabindex="0" aria-expanded="false">'
            f'<span class="subproof-toggle">▼</span><span class="block-number">{node["number"]}</span><span class="reader-block-kind">{label}</span>'
            f'<div class="mpl-line">{formal}</div><div class="readable-grey">{common}</div></div>'
            f'<div class="subproof-body">{body}</div></div>'
        )

    metrics = reader_metrics(plan, rows)
    controls = (
        '<div class="subproof-controls">'
        '<button type="button" onclick="this.closest(&quot;.reader-section&quot;).querySelectorAll(&quot;.subproof-card&quot;).forEach(c=>window.setProofCardOpen(c,true))">Expand all</button>'
        '<button type="button" onclick="this.closest(&quot;.reader-section&quot;).querySelectorAll(&quot;.subproof-card&quot;).forEach(c=>window.setProofCardOpen(c,false))">Collapse all</button></div>'
    )
    if all(child["kind"] == "row" for child in plan["children"]):
        controls = ""
    return (
        f'<section class="reader-section subproofs-section" data-main-rows="{metrics["main_rows"]}" data-proof-steps="{len(rows)}">'
        f'<h3><span class="block-number">{section}</span>Proof</h3>{controls}'
        f'<div class="reader-counts">Main rows: {metrics["main_rows"]} · All proof steps: {len(rows)}</div>'
        + render(plan)
        + "</section>"
    )
