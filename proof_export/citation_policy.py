# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Shared citation priority for processing, nested placement and certificates."""


def resolve_citation_source(
    tag,
    exact_key,
    alias_key,
    exact_sources,
    theorem_sources,
    produced_exact,
    registered,
    integration_template,
    optional_companion=False,
):
    """
    @brief Classify one citation under the shared exact-source priority policy.

    @details Compilation prefers a registered theorem. Other rows prefer an
    exact local source, then an alpha-equivalent theorem source except for
    zoom-in edges. A not-yet-available exact producer blocks external fallback.
    Otherwise registered theorems, goal templates and optional OR companions
    have their explicit nonlocal dispositions. Pending is a normal readiness
    result; the complete graph scheduler asserts if no pending row can advance.
    Callers own the key's namespace and proof-scope identity and determine
    whether their source maps cover an entire scope or already emitted rows.

    @param tag Proof tag of the citing row.
    @param exact_key Exact expression and scope identity in the caller's index.
    @param alias_key Bound-name-normalized identity in the theorem index.
    @param exact_sources Exact local identities mapped to source tokens.
    @param theorem_sources Theorem-import identities mapped to source tokens.
    @param produced_exact All exact local identities, including pending rows.
    @param registered Whether an explicit theorem registry resolves the citation.
    @param integration_template Whether the citation is a goal template.
    @param optional_companion Whether this OR citation may be absent.
    @return Disposition and a local source token, or None for a nonlocal result.
    @invariant Exact scope identities never acquire an unrelated theorem alias.
    """
    assert not optional_companion or tag == "or theorem"
    if tag == "compilation" and registered:
        return "theorem", None
    source = exact_sources.get(exact_key)
    if source is None and tag != "zoom in":
        source = theorem_sources.get(alias_key)
    if source is not None:
        return "local", source
    if exact_key in produced_exact:
        return "pending", None
    if registered:
        return "theorem", None
    if integration_template:
        return "template", None
    if optional_companion:
        return "optional", None
    return "pending", None
