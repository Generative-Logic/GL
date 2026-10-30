# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Conservative deletion of unused ordinary main-scope proof leaves."""
from proof_export.corpora import MAIN_ANCHORS
from proof_processing import citation_key

SAFE_TAGS = {
    "equality1",
    "equality2",
    "symmetry of equality",
    "symmetry of inequality",
    "implication",
    "definition projection",
}


def slice_unused_main_rows(rows):
    """
    @brief Delete ordinary main-scope leaves outside every explicit use closure.

    @details All non-main rows and all boundary, witness, import, assumption and
    definition tags are roots. The main goal is a root. Only an ordinary safe
    row with no alpha-equivalent scoped citation anywhere and only main-scope
    premises may be removed. Repeat until no further such leaf exists.

    @param rows Complete processed chapter rows.
    @return Rows after conservative deletion, preserving every subproof row.
    @invariant No cross-scope edge, witness obligation or scope boundary changes.
    """
    assert rows and all(len(r) >= 3 and (len(r) - 3) % 2 == 0 for r in rows)
    result = [list(r) for r in rows]
    while True:
        used = {
            (citation_key(r[j]), r[j + 1]) for r in result for j in range(3, len(r), 2)
        }
        keep = [
            r
            for i, r in enumerate(result)
            if i == 0
            or r[1] != "main"
            or r[2] not in SAFE_TAGS
            or any(r[j + 1] != "main" for j in range(3, len(r), 2))
            or (citation_key(r[0]), r[1]) in used
        ]
        if len(keep) == len(result):
            return result
        result = keep


def apply_main_rows(theorems, stacks, filename_groups):
    """
    @brief Apply this experiment only to the three certified main-book anchors.

    @details Each physical induction part is transformed separately; incubator
    and any other anchor retain their original rows and namespace ownership.

    @param theorems Complete processed theorem registry.
    @param stacks Mapping from physical chapter filenames to rows.
    @param filename_groups Parallel theorem-to-file bundles.
    @return Mapping with main-book rows transformed by this experiment.
    """
    from proof_export.mpl import Atom, Implication, parse_mpl

    assert len(theorems) == len(filename_groups)
    result = dict(stacks)
    for theorem, names in zip(theorems, filename_groups):
        expression = parse_mpl(theorem[0])
        assert isinstance(expression, Implication) and isinstance(
            expression.premise, Atom
        )
        if expression.premise.head in MAIN_ANCHORS:
            for name in names:
                result[name] = slice_unused_main_rows(stacks[name])
    return result
