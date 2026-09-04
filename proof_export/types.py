# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt)
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Finite type inference for GL's element, set, and relation ports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .mpl import Expression, iter_atoms, parse_mpl


ELEMENT = "element"
SET = "set"
BINARY_RELATION = "binary_relation"
TERNARY_RELATION = "ternary_relation"

TYPE_LABELS = {
    "(1)": ELEMENT,
    "P(1)": SET,
    "P(x(1)(1))": BINARY_RELATION,
    "P(x(1)(x(1)(1)))": TERNARY_RELATION,
}


class _TypeUnion:
    """Union-find whose roots carry one asserted GL port type."""

    def __init__(self) -> None:
        """
        @brief Initialize an empty finite type-equivalence structure.
        @details Parent links and concrete root labels start empty.
        @return None.
        """

        self.parent: dict[tuple[object, ...], tuple[object, ...]] = {}
        self.value: dict[tuple[object, ...], str] = {}

    def _add(self, node: tuple[object, ...]) -> None:
        """
        @brief Register one type node as its own initial root.
        @details Existing nodes remain unchanged.
        @param node Stable structural type-node key.
        @return None.
        """

        if node not in self.parent:
            self.parent[node] = node

    def find(self, node: tuple[object, ...]) -> tuple[object, ...]:
        """
        @brief Find the canonical root for one type node.
        @details Missing nodes are registered and traversed paths are compressed.
        @param node Stable structural type-node key.
        @return Canonical root node.
        """

        self._add(node)
        parent = self.parent[node]
        if parent != node:
            self.parent[node] = self.find(parent)
        return self.parent[node]

    def unite(self, left: tuple[object, ...], right: tuple[object, ...]) -> None:
        """
        @brief Unify two type nodes.
        @details Existing concrete labels must agree and are retained on the
        chosen left root.
        @param left First type-node key.
        @param right Second type-node key.
        @return None.
        """

        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        left_value = self.value.get(left_root)
        right_value = self.value.get(right_root)
        assert left_value is None or right_value is None or left_value == right_value, (
            f"GL type conflict: {left_value!r} versus {right_value!r}"
        )
        self.parent[right_root] = left_root
        if left_value is None and right_value is not None:
            self.value[left_root] = right_value
        self.value.pop(right_root, None)

    def constrain(self, node: tuple[object, ...], kind: str) -> None:
        """
        @brief Constrain one equivalence class to a concrete GL port type.
        @details A conflicting existing constraint is rejected.
        @param node Type-node key.
        @param kind Finite GL type label.
        @return None.
        """

        root = self.find(node)
        existing = self.value.get(root)
        assert existing is None or existing == kind, (
            f"GL type conflict: {existing!r} versus {kind!r}"
        )
        self.value[root] = kind

    def resolved(self, node: tuple[object, ...]) -> str:
        """
        @brief Resolve one node to its concrete GL port type.
        @details An unconstrained root is an incomplete inference and asserts.
        @param node Type-node key.
        @return Resolved finite GL type label.
        """

        root = self.find(node)
        assert root in self.value, f"unresolved GL type node: {node!r}"
        return self.value[root]


@dataclass(frozen=True)
class TypeEnvironment:
    """Resolved signatures and GL-binary closure used by one export."""

    signatures: dict[str, tuple[str, ...]]
    binary_entries: dict[str, dict[str, object]]
    binary_closure: tuple[str, ...]


def _signature_node(name: str, index: int) -> tuple[object, ...]:
    """
    @brief Form the type node for one compiled signature port.
    @details The structural tuple separates signature ports from local variables.
    @param name Atom or compiled-definition head.
    @param index Zero-based argument index.
    @return Stable union-find node key.
    """

    return ("signature", name, index)


def _argument_node(definition: str, argument: str) -> tuple[object, ...]:
    """
    @brief Form the type node for one definition-body argument.
    @details `u_` parameters alias signature ports; other tokens are local nodes.
    @param definition Owning compiled definition.
    @param argument MPL argument token.
    @return Stable union-find node key.
    """

    if argument.startswith("u_") and argument[2:].isdigit():
        return _signature_node(definition, int(argument[2:]) - 1)
    return ("local", definition, argument)


def _binary_closure(
    binary_entries: dict[str, dict[str, object]],
    root: str,
) -> tuple[str, ...]:
    """
    @brief Compute the transitive compiled-definition closure of one root.
    @details Atomic entries terminate traversal and all returned names are sorted.
    @param binary_entries Parsed GL binary definition table.
    @param root Required compiled root name.
    @return Sorted closure names including the root.
    """

    assert root in binary_entries, f"GL binary has no {root!r} entry"
    visited: set[str] = set()
    pending = [root]
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        entry = binary_entries[name]
        elements = entry.get("elements")
        assert isinstance(elements, list), f"GL binary entry {name!r} has no elements list"
        referenced = {
            atom.head
            for element in elements
            for atom in iter_atoms(parse_mpl(str(element)))
            if (
                atom.head in binary_entries
                and binary_entries[atom.head].get("category") != "atomic"
            )
        }
        pending.extend(sorted(referenced, reverse=True))
    return tuple(sorted(visited))


def load_type_environment(
    config_path: Path,
    binary_path: Path,
    root: str = "AnchorPeano",
    extra_roots: tuple[str, ...] = (),
    compiled_overrides: dict[str, dict[str, object]] | None = None,
    extra_signature_names: tuple[str, ...] = (),
) -> TypeEnvironment:
    """
    @brief Infer finite port types for the selected compiled closure.
    @details
    Constructed OR definitions may override same-named stale fixture entries;
    their semantic source is the checked certificate action. All referenced
    signature ports are unified against configuration type labels.
    @param config_path Typed GL configuration JSON.
    @param binary_path GL binary definition JSON.
    @param root Primary compiled closure root.
    @param extra_roots Additional compiled roots required by selected rows.
    @param compiled_overrides Certificate-derived replacement definitions.
    @param extra_signature_names Referenced atomic signatures outside the closure.
    @return Fully resolved type environment and selected binary closure.
    """

    config = json.loads(config_path.read_text(encoding="utf-8"))
    binary_entries = json.loads(binary_path.read_text(encoding="utf-8"))
    assert isinstance(config, dict) and isinstance(binary_entries, dict)
    for name, entry in sorted((compiled_overrides or {}).items()):
        binary_entries[name] = entry
    closure_names: set[str] = set()
    for closure_root in (root, *extra_roots):
        closure_names.update(_binary_closure(binary_entries, closure_root))
    closure = tuple(sorted(closure_names))
    union = _TypeUnion()
    arities: dict[str, int] = {}

    for name, entry in config.items():
        if not isinstance(entry, dict) or "arity" not in entry:
            continue
        arity = int(entry["arity"])
        arities[name] = arity
        definition_sets = entry.get("definition_sets")
        assert isinstance(definition_sets, dict)
        for index in range(arity):
            raw = definition_sets[str(index + 1)]
            assert isinstance(raw, list) and raw
            label = str(raw[0])
            assert label in TYPE_LABELS, f"unsupported GL type label {label!r}"
            union.constrain(_signature_node(name, index), TYPE_LABELS[label])

    for name in closure:
        entry = binary_entries[name]
        arity = int(entry["arity"])
        arities[name] = arity
        for element_text in entry["elements"]:
            for atom in iter_atoms(parse_mpl(str(element_text))):
                assert atom.head in arities or atom.head in closure, (
                    f"no type signature for {atom.head!r} referenced by {name!r}"
                )
                callee_arity = arities.get(atom.head)
                if callee_arity is None:
                    callee_arity = int(binary_entries[atom.head]["arity"])
                    arities[atom.head] = callee_arity
                assert len(atom.arguments) == callee_arity, (
                    f"arity mismatch for {atom.head!r} in {name!r}"
                )
                for index, argument in enumerate(atom.arguments):
                    union.unite(
                        _argument_node(name, argument),
                        _signature_node(atom.head, index),
                    )

    needed_names = set(closure) | set(extra_signature_names)
    for name in closure:
        for element_text in binary_entries[name]["elements"]:
            needed_names.update(atom.head for atom in iter_atoms(parse_mpl(str(element_text))))
    signatures: dict[str, tuple[str, ...]] = {}
    for name in sorted(needed_names):
        assert name in arities, f"missing arity for {name!r}"
        signatures[name] = tuple(
            union.resolved(_signature_node(name, index))
            for index in range(arities[name])
        )

    selected_entries = {name: binary_entries[name] for name in closure}
    return TypeEnvironment(signatures, selected_entries, closure)


def infer_expression_variables(
    expressions: list[Expression],
    signatures: dict[str, tuple[str, ...]],
) -> dict[str, str]:
    """
    @brief Infer every argument token type across a group of expressions.
    @details Each atom occurrence constrains arguments by its resolved signature;
    a token joining unlike ports is rejected.
    @param expressions Parsed expressions sharing one variable environment.
    @param signatures Resolved atom signatures.
    @return Sorted token-to-type mapping.
    """

    variables: dict[str, str] = {}
    for expression in expressions:
        for atom in iter_atoms(expression):
            assert atom.head in signatures, f"unknown typed MPL head {atom.head!r}"
            signature = signatures[atom.head]
            assert len(atom.arguments) == len(signature), (
                f"arity mismatch for {atom.head!r}: {len(atom.arguments)} != {len(signature)}"
            )
            for argument, kind in zip(atom.arguments, signature, strict=True):
                previous = variables.get(argument)
                assert previous is None or previous == kind, (
                    f"variable {argument!r} joins {previous!r} and {kind!r} ports"
                )
                variables[argument] = kind
    return dict(sorted(variables.items()))
