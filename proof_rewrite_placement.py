# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Move repeated argument substitutions to an implication conclusion."""
from proof_export.corpora import MAIN_ANCHORS
from collections import Counter
from functools import cache
from proof_processing import order_scope_rows, citation_key
from proof_export.mpl import Atom, Negation, parse_mpl, to_mpl
from proof_export.rule_matching import _rule_binding


def implication_candidate(row):
    """
    @brief Decide whether one same-scope candidate instantiates its recorded rule.

    @details Match immutable parsed syntax with exact free ports and consistent
    bound-name assignments. This candidate selector is independent of verifier
    state; the completed graph must still pass both external proof checkers.

    @param row Candidate implication row with paired source citations.
    @return Whether the candidate has one consistent premise/rule matching.
    """
    assert len(row) >= 3 and (len(row) - 3) % 2 == 0
    if len(row) < 5 or any(row[j + 1] != row[1] for j in range(3, len(row), 2)):
        return False
    return (
        _rule_binding(
            parse_mpl(row[3]),
            [parse_mpl(row[j]) for j in range(5, len(row), 2)],
            parse_mpl(row[0]),
        )
        is not None
    )


def equality_candidate(row):
    """
    @brief Decide whether explicit oriented equalities cover a proposed rewrite.

    @details Source and target must be atoms with identical polarity, head and
    arity. Each changed argument requires its exact source-to-target equality;
    no transitive, symmetric or cross-scope evidence is inferred here.

    @param row Candidate equality1 row with paired source citations.
    @return Whether every changed position has explicit same-scope support.
    """
    assert len(row) >= 3 and (len(row) - 3) % 2 == 0
    if len(row) < 7 or any(row[j + 1] != row[1] for j in range(3, len(row), 2)):
        return False
    source, target = parse_mpl(row[3]), parse_mpl(row[0])
    if isinstance(source, Negation) != isinstance(target, Negation):
        return False
    if isinstance(source, Negation):
        source, target = source.inner, target.inner
    if not (
        isinstance(source, Atom)
        and isinstance(target, Atom)
        and source.head == target.head
        and len(source.arguments) == len(target.arguments)
    ):
        return False
    supports = set()
    for j in range(5, len(row), 2):
        equality = parse_mpl(row[j])
        if not (
            isinstance(equality, Atom)
            and equality.head == "="
            and len(equality.arguments) == 2
        ):
            return False
        supports.add(equality.arguments)
    return all(
        a == b or (a, b) in supports for a, b in zip(source.arguments, target.arguments)
    )


def place_conclusion_rewrite(rows, known_theorems=()):
    """
    @brief Replace multiple premise rewrites with at most one conclusion rewrite.

    @details Use only same-scope, single-consumer equality1 premises. Construct
    the pre-substitution conclusion using a consistent inverse argument map,
    then require independent structural matching and explicit oriented equalities
    to justify both candidate steps. If the original premises already yield the unchanged
    conclusion, no conclusion substitution is needed. An earlier identical
    conclusion may be reused. Every
    accepted move strictly reduces physical rows and preserves scope boundaries.

    @param rows Complete chapter in reversed dependency order.
    @param known_theorems Normalized registered external theorem citation keys.
    @return Shorter or unchanged rows under the existing inference vocabulary.
    @invariant No assumption, witness, scope or cross-scope interface is changed.
    """
    assert rows and all(len(r) >= 3 and (len(r) - 3) % 2 == 0 for r in rows)
    result = [list(r) for r in rows]
    key_for = cache(citation_key)
    known_theorems = set(known_theorems)
    while True:
        by = {(r[0], r[1]): i for i, r in enumerate(result)}
        assert len(by) == len(result)
        uses = Counter((r[j], r[j + 1]) for r in result for j in range(3, len(r), 2))
        first = {}
        for i, r in enumerate(result):
            first.setdefault(r[1], i)
        changed = False
        for index, row in enumerate(result):
            if row[2] != "implication" or any(
                row[j + 1] != row[1] for j in range(3, len(row), 2)
            ):
                continue
            target = parse_mpl(row[0])
            atom = target.inner if isinstance(target, Negation) else target
            parents = {}
            for j in range(5, len(row), 2):
                key = (row[j], row[j + 1])
                if key not in by:
                    continue
                position = by[key]
                parent = result[position]
                if (
                    parent[2] != "equality1"
                    or uses[key] != 1
                    or position == first[row[1]]
                ):
                    continue
                if any(parent[k + 1] != row[1] for k in range(3, len(parent), 2)):
                    continue
                assert position > index and len(parent) >= 7
                assert equality_candidate(parent)
                parents[key] = parent
            if len(parents) < 2:
                continue
            direct = row[:5]
            for j in range(5, len(row), 2):
                key = (row[j], row[j + 1])
                direct += parents[key][3:5] if key in parents else row[j : j + 2]
            if implication_candidate(direct):
                old_count = len(result)
                result = [
                    direct if i == index else r
                    for i, r in enumerate(result)
                    if i == index or (r[0], r[1]) not in parents
                ]
                assert len(result) < old_count
                result = order_scope_rows(result, known_theorems, key_for)
                changed = True
                break
            if not isinstance(atom, Atom):
                continue
            inverse = {}
            supports = {}
            consistent = True
            for parent in parents.values():
                old, new = parse_mpl(parent[3]), parse_mpl(parent[0])
                assert isinstance(old, Negation) == isinstance(new, Negation)
                if isinstance(old, Negation):
                    old, new = old.inner, new.inner
                assert (
                    isinstance(old, Atom)
                    and isinstance(new, Atom)
                    and old.head == new.head
                    and len(old.arguments) == len(new.arguments)
                )
                for a, b in zip(old.arguments, new.arguments):
                    if a == b:
                        continue
                    if b in inverse and inverse[b] != a:
                        consistent = False
                    inverse[b] = a
                for j in range(5, len(parent), 2):
                    equality = parse_mpl(parent[j])
                    assert (
                        isinstance(equality, Atom)
                        and equality.head == "="
                        and len(equality.arguments) == 2
                    )
                    supports[equality.arguments] = parent[j : j + 2]
            if not consistent:
                continue
            before_atom = Atom(
                atom.head, tuple(inverse.get(v, v) for v in atom.arguments)
            )
            before = to_mpl(
                Negation(before_atom) if isinstance(target, Negation) else before_atom
            )
            candidate = [before, row[1], "implication"] + row[3:5]
            for j in range(5, len(row), 2):
                key = (row[j], row[j + 1])
                candidate += parents[key][3:5] if key in parents else row[j : j + 2]
            if not implication_candidate(candidate):
                continue
            before_key = (before, row[1])
            extra = []
            if before == row[0]:
                replacement = candidate
            else:
                required = list(
                    dict.fromkeys(
                        (a, b)
                        for a, b in zip(before_atom.arguments, atom.arguments)
                        if a != b
                    )
                )
                if not all(pair in supports for pair in required):
                    continue
                replacement = (
                    row[:2]
                    + ["equality1", before, row[1]]
                    + [v for pair in required for v in supports[pair]]
                )
                if not equality_candidate(replacement):
                    continue
                if before_key in by:
                    if by[before_key] <= index or before_key in parents:
                        continue
                else:
                    extra = [candidate]
            if any(
                (replacement[j], replacement[j + 1]) in parents
                for j in range(3, len(replacement), 2)
            ):
                continue
            old_count = len(result)
            source_slot = min(by[key] for key in parents)
            result = [
                entry
                for i, r in enumerate(result)
                for entry in (
                    [replacement]
                    if i == index
                    else (
                        extra
                        if i == source_slot and extra
                        else [] if (r[0], r[1]) in parents else [r]
                    )
                )
            ]
            assert len(result) < old_count
            result = order_scope_rows(result, known_theorems, key_for)
            changed = True
            break
        if not changed:
            return result


def apply_main_rows(theorems, stacks, filename_groups, external_theorems=()):
    """
    @brief Apply this experiment only to the three certified main-book anchors.

    @details Each physical induction part is transformed separately; incubator
    and any other anchor retain their original rows and namespace ownership.

    @param theorems Complete processed theorem registry.
    @param stacks Mapping from physical chapter filenames to rows.
    @param filename_groups Parallel theorem-to-file bundles.
    @param external_theorems Exact registered external theorem spellings.
    @return Mapping with main-book rows transformed by this experiment.
    """
    from proof_export.mpl import Atom, Implication, parse_mpl

    assert len(theorems) == len(filename_groups)
    result = dict(stacks)
    known = {citation_key(theorem[0]) for theorem in theorems}
    known.update(citation_key(theorem) for theorem in external_theorems)
    for theorem, names in zip(theorems, filename_groups):
        expression = parse_mpl(theorem[0])
        assert isinstance(expression, Implication) and isinstance(
            expression.premise, Atom
        )
        if expression.premise.head in MAIN_ANCHORS:
            for name in names:
                result[name] = place_conclusion_rewrite(stacks[name], known)
    return result
