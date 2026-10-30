# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Compiled-operator retirement and cross-batch reference migration."""

from copy import deepcopy
import json
from pathlib import Path
import re

CATEGORIES = ("implication", "existence", "and", "or")
ATOM = re.compile(r"\(([^()[\],\s]+)\[([^()[\]]*)\]\)")


def rewrite_references(text: str, replacements: dict) -> str:
    """@brief Replace retired MPL applications, including scope payloads.

    @details Changes only complete atom applications and their argument order.
    Surrounding bytes, binders, polarity, tabs and proof-reference pairs survive.
    Every application the pattern rewrites starts with `(<retired name>[`, so a
    text holding no such substring is returned as it is without the pattern scan.
    @param text Expression or proof-file text.
    @param replacements Retired names mapped to target and one-based arguments.
    @return Rewritten text; unrelated text is byte-identical.
    @invariant Invalid arities, permutations and cycles assert.
    """
    if not names_retired_operator(text, replacements):
        return text

    def substitute(match):
        name, argument_text = match.groups()
        if name not in replacements:
            return match.group(0)
        arguments = argument_text.split(",") if argument_text else []
        seen = set()
        while name in replacements:
            assert name not in seen, f"compiled replacement cycle: {name}"
            seen.add(name)
            replacement = replacements[name]
            positions = replacement["arguments"]
            assert len(positions) == len(arguments), f"replacement arity: {name}"
            assert sorted(positions) == list(range(1, len(arguments) + 1)), name
            arguments = [arguments[position - 1] for position in positions]
            name = replacement["target"]
        return f"({name}[{','.join(arguments)}])"

    result = ATOM.sub(substitute, text)
    assert all(f"({name}[" not in result for name in replacements), "malformed retired application"
    return result


def names_retired_operator(text: str, replacements: dict) -> bool:
    """@brief Whether a text applies any retired operator.

    @details A retired application always starts with `(<name>[`; the bracket
    after the name keeps a retired name from matching a longer name it
    prefixes. Used to skip the pattern rewrite and the defensive copies for
    texts and registry entries that name no retired operator.
    @param text Expression, proof-file text, or a serialized registry entry.
    @param replacements Retired names mapped to their replacement records.
    @return True when some retired name is applied in the text.
    """
    return any(f"({name}[" in text for name in replacements)


def validate_state(state: dict) -> None:
    """@brief Validate the complete compilation-state sidecar contract.

    @details Schema 1 records all next counters and argument-correct retirement
    records. Unknown fields, malformed names, cycles and invalid numbers assert.
    @param state Parsed sidecar object.
    @return None.
    """
    assert set(state) == {"schema", "next_counters", "replacements"}
    assert type(state["schema"]) is int and state["schema"] == 1
    assert set(state["next_counters"]) == set(CATEGORIES)
    assert all(type(value) is int and value >= 0
               for value in state["next_counters"].values())
    assert isinstance(state["replacements"], dict)
    for name, record in state["replacements"].items():
        assert re.fullmatch(r"(?:implication|existence|and|or)\d+", name), name
        assert set(record) == {"target", "arguments"}
        assert isinstance(record["target"], str) and record["target"]
        positions = record["arguments"]
        assert isinstance(positions, list) and all(type(i) is int for i in positions)
        assert sorted(positions) == list(range(1, len(positions) + 1)), name
        rewrite_references(f"({name}[{','.join(map(str, range(len(positions))))}])",
                           state["replacements"])


def read_state(directory: Path, tag: str) -> dict:
    """@brief Read a sidecar or derive state from an original-format registry.

    @details Sidecar absence identifies the supported legacy format. Its next
    counters are derived from every generated name; no retired state exists.
    @param directory Compiled-registry directory.
    @param tag Registry suffix, including shared or a batch name.
    @return Validated schema-1 state.
    """
    path = directory / f"GL_compilation_state_{tag}.json"
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
    else:
        registry_path = directory / f"GL_binary_{tag}.json"
        registry = (json.loads(registry_path.read_text(encoding="utf-8"))
                    if registry_path.exists() else {})
        counters = dict.fromkeys(CATEGORIES, 0)
        for name, entity in registry.items():
            category = entity["category"]
            if category in counters and re.fullmatch(re.escape(category) + r"\d+", name):
                counters[category] = max(counters[category], int(name[len(category):]) + 1)
        state = {"schema": 1, "next_counters": counters, "replacements": {}}
    validate_state(state)
    return state


def seed_state(root: Path, tag: str) -> None:
    """@brief Seed the native batch with shared monotone compilation state.

    @details Runs alongside compiled-registry seeding; each batch receives the
    cumulative retirement map and all issued-number high-water marks.
    @param root Project root.
    @param tag Batch registry suffix.
    @return None.
    """
    directory = root / "files/GL_binaries"
    state = read_state(directory, "shared")
    (directory / f"GL_compilation_state_{tag}.json").write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def rewrite_auxiliary_references(entity: dict, replacements: dict) -> list[str]:
    """@brief Rewrite every prepared auxiliary and return its dependency texts.

    @details Guarded support, direct conjunction refutations and contraposition
    metadata are part of the registry dependency graph. Preparation flags and
    source correspondence are preserved while operator names and ports migrate.
    @param entity Owned registry entry updated in place.
    @param replacements Complete retired-name substitutions.
    @return Rewritten auxiliary texts whose referenced definitions must be retained.
    @invariant Malformed auxiliary schemas assert before migration continues.
    """
    references = []
    for key in ("supporting_implications", "conjunction_refutations", "contraposition"):
        if key not in entity:
            continue
        auxiliary = entity[key]
        assert auxiliary["version"] in ((1, 2) if key == "supporting_implications" else (1,))
        assert isinstance(auxiliary["rules"], list)
        if key == "supporting_implications":
            auxiliary["rules"] = [rewrite_references(value, replacements) for value in auxiliary["rules"]]
            references.extend(auxiliary["rules"])
            if auxiliary["version"] == 2:
                composed = auxiliary["composed_rules"]
                assert isinstance(composed, list) and (not composed or len(composed) == len(auxiliary["rules"]))
                auxiliary["composed_rules"] = [rewrite_references(value, replacements) for value in composed]
                references.extend(auxiliary["composed_rules"])
        elif key == "conjunction_refutations":
            for pair in auxiliary["rules"]:
                assert len(pair) == 2
                pair[:] = [rewrite_references(value, replacements) for value in pair]
                references.extend(pair)
        else:
            for row in auxiliary["rules"]:
                assert row["sources"]
                row["sources"] = [rewrite_references(value, replacements) for value in row["sources"]]
                references.extend(row["sources"])
                for field in ("source", "reverse"):
                    row[field] = rewrite_references(row[field], replacements)
                    references.append(row[field])
    return references


def rewrite_registry(registry: dict, source: dict, replacements: dict) -> dict:
    """@brief Remove retired definitions and rewrite a registry's dependency graph.

    @details Imports retained public targets and their dependency closure from
    the completed native registry. No retired operator remains as an alias.
    An entry that applies no retired operator is carried over as the same
    object: every rewrite of it is the identity, so it needs no copy.
    @param registry Registry to migrate.
    @param source Completed batch registry containing retained public targets.
    @param replacements Complete retired-name substitutions.
    @return Migrated registry with all retained dependencies available.
    """
    result = {}
    required = {record["target"] for record in replacements.values()}
    for name, entity in registry.items():
        if name in replacements:
            required.add(replacements[name]["target"])
            continue
        mentions = names_retired_operator(json.dumps(entity, ensure_ascii=False), replacements)
        copied = deepcopy(entity) if mentions else entity
        for key in ("signature", "definedSet"):
            if key in copied:
                copied[key] = rewrite_references(copied[key], replacements)
        for key in ("elements", "implications"):
            if key in copied:
                copied[key] = [rewrite_references(value, replacements) for value in copied[key]]
                for value in copied[key]:
                    required.update(match.group(1) for match in ATOM.finditer(value))
        for record in copied.get("integration_implications", []):
            record["compact"] = rewrite_references(record["compact"], replacements)
            required.update(match.group(1) for match in ATOM.finditer(record["compact"]))
        for value in rewrite_auxiliary_references(copied, replacements):
            required.update(match.group(1) for match in ATOM.finditer(value))
        result[name] = copied
    while required:
        name = min(required)
        required.remove(name)
        if name in result:
            continue
        assert name not in replacements, f"uncomposed replacement target: {name}"
        assert name in source, f"missing retained compiler dependency: {name}"
        entity = deepcopy(source[name])
        result[name] = entity
        for value in rewrite_auxiliary_references(entity, replacements):
            required.update(match.group(1) for match in ATOM.finditer(value)
                            if match.group(1) not in result)
        for key in ("elements", "implications"):
            for value in entity.get(key, []):
                required.update(match.group(1) for match in ATOM.finditer(value)
                                if match.group(1) not in result)
        for record in entity.get("integration_implications", []):
            required.update(match.group(1) for match in ATOM.finditer(record["compact"])
                            if match.group(1) not in result)
    assert not (result.keys() & replacements.keys())
    return result


def migrate_batch(root: Path, tag: str, *, shortcut: bool = False) -> dict:
    """@brief Migrate active registries and proof references after native execution.

    @details Applies new retirements before the shared merge and subsequent
    proof readers. Historical archives and source definitions are untouched.
    Full runs migrate both theorem pools; replay migrates its isolated pool.
    @param root Project root.
    @param tag Completed native batch.
    @param shortcut Whether proof artifacts belong to the replay tree.
    @return Counts of newly retired operators and rewritten files.
    @invariant Shared state cannot resurrect retired operators or reuse numbers.
    """
    directory = root / "files/GL_binaries"
    state_path = directory / f"GL_compilation_state_{tag}.json"
    assert state_path.is_file(), f"native compiler omitted state: {tag}"
    state = read_state(directory, tag)
    prior = read_state(directory, "shared")
    for category in CATEGORIES:
        assert state["next_counters"][category] >= prior["next_counters"][category]
    replacements = state["replacements"]
    assert prior["replacements"].keys() <= replacements.keys()
    changed = {name: record for name, record in replacements.items()
               if prior["replacements"].get(name) != record}
    written = 0
    if changed:
        source = json.loads((directory / f"GL_binary_{tag}.json").read_text(encoding="utf-8"))
        assert not (source.keys() & replacements.keys())
        for path in sorted(directory.glob("GL_binary_*.json")):
            registry = json.loads(path.read_text(encoding="utf-8"))
            migrated = rewrite_registry(registry, source, replacements)
            if migrated != registry:
                path.write_text(json.dumps(migrated, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                written += 1
        prefixes = ("files/shortcut",) if shortcut else ("files", "files/incubator")
        for prefix in prefixes:
            for suffix in ("theorems", "raw_proof_graph", "processed_proof_graph"):
                for path in sorted((root / prefix / suffix).rglob("*.txt")):
                    original = path.read_bytes().decode("utf-8")
                    updated = rewrite_references(original, replacements)
                    if updated != original:
                        path.write_bytes(updated.encode("utf-8"))
                        written += 1
        # Every active registry can be loaded independently without recycling an
        # issued number. The old snapshots' metadata is migrated with its body.
        for path in sorted(directory.glob("GL_binary_*.json")):
            suffix = path.name[len("GL_binary_"):]
            (directory / f"GL_compilation_state_{suffix}").write_text(
                json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (directory / "GL_compilation_state_shared.json").write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"retired": len(changed), "rewritten_files": written}
