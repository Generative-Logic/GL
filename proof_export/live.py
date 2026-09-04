# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt)
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Lean export inside a pipeline run.

The tracked release corpora pin a theorem-list hash, a source range, the
expected coverage and every cited dependency theorem by index. A pipeline run
produces a fresh proof graph, so nothing can be pinned ahead of it: this module
derives the selection from the run itself — the anchor blocks of the theorem
list, the computed hash, the cited theorems by content — and writes the
certificates, the Lean project and the kernel-check log into the mode's own
proof-graph folder (`<full_proof_graph>/lean_export/`).

Policy (maintainer, 2026-09-04): when the Lean toolchain is absent the run
prints one notice and produces no Lean export; when it is present the run
exports and kernel-checks, and a kernel failure fails the run.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from .certificate import (
    base_form_equivalent,
    binary_entries_with_overrides,
    build_certificate,
    theorem_adaptation_solutions,
    theorem_citations,
    theorem_citations_with_tags,
    write_certificate,
)
from .graph import TheoremRecord, find_theorem, load_theorem_records, sha256_file
from .lean import write_lean
from .mpl import Atom, Expression, Implication, alpha_key, expression_json, iter_atoms, parse_mpl, to_mpl


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
TRACKED_LEAN_PROJECT = REPOSITORY_ROOT / "lean_export"
EXPORT_FOLDER = "lean_export"
PROJECT_SKELETON = (
    "lakefile.lean",
    "lean-toolchain",
    "lake-manifest.json",
    "GLExport/ProofSupport.lean",
)
ANCHOR_CORPORA = {"AnchorPeano": "peano", "AnchorGauss": "gauss", "AnchorFTA": "fta"}
INSTALL_HINT = (
    "install Lean with `winget install --id LeanProver.Elan` (Windows) or "
    "`curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh"
    " -sSf | sh` (Linux, macOS)"
)


def find_lake() -> Path | None:
    """
    @brief Locate the Lake build tool of the Lean toolchain.
    @details
    `lake` on the PATH wins; otherwise the elan default install directory
    `~/.elan/bin` is probed for `lake.exe` and `lake`. Absence is a defined
    answer, not a failure: the maintainer's policy makes the Lean export
    conditional on the toolchain and never crashes a customer run without it.
    @return Path of the Lake executable, or None when no toolchain is installed.
    """

    found = shutil.which("lake")
    if found is not None:
        return Path(found)
    for name in ("lake.exe", "lake"):
        candidate = Path.home() / ".elan" / "bin" / name
        if candidate.is_file():
            return candidate
    return None


def anchor_head(expression: Expression) -> str:
    """
    @brief Name the anchor a theorem is stated under.
    @details Every registered theorem is an implication whose outermost premise
    is the anchor atom (`AnchorPeano[...]`, `AnchorGauss[...]`, `AnchorFTA[...]`).
    @param expression Parsed theorem expression.
    @return The anchor atom's head.
    """

    assert isinstance(expression, Implication), "theorem is not an implication"
    premise = expression.premise
    assert isinstance(premise, Atom), "outermost premise is not an anchor atom"
    assert premise.head in ANCHOR_CORPORA, f"unknown anchor {premise.head!r}"
    return premise.head


def theorem_list_blocks(records: tuple[TheoremRecord, ...]) -> dict[str, tuple[int, int]]:
    """
    @brief Partition the theorem list into its contiguous anchor blocks.
    @details
    A full run writes the Peano theorems first and the Gauss theorems after
    them; the shortcut list is FTA only. Each anchor therefore owns one
    contiguous index run, which becomes the corpus' `source_range`. An
    interleaved list is a broken registry and asserts.
    @param records Loaded theorem registry (chapters need not be parsed).
    @return Anchor head → inclusive (first, last) source-index pair.
    """

    blocks: dict[str, tuple[int, int]] = {}
    for record in records:
        head = anchor_head(record.expression)
        if head not in blocks:
            blocks[head] = (record.source_index, record.source_index)
            continue
        first, last = blocks[head]
        assert last == record.source_index - 1, (
            f"theorem list interleaves anchors: {head} at {record.source_index} "
            f"after its block ended at {last}"
        )
        blocks[head] = (first, record.source_index)
    return blocks


def parsed_block(
    records: tuple[TheoremRecord, ...],
    first: int,
    last: int,
) -> list[TheoremRecord]:
    """
    @brief Parse the chapters of one anchor block.
    @param records Loaded theorem registry.
    @param first First source index of the block (inclusive).
    @param last Last source index of the block (inclusive).
    @return Theorem records of the block with their chapters parsed, in index order.
    """

    return [
        find_theorem(records, to_mpl(record.expression), record.method, record.source_index)
        for record in records[first:last + 1]
    ]


def same_list_imports(
    records: tuple[TheoremRecord, ...],
    parsed: list[TheoremRecord],
    first: int,
    last: int,
) -> list[int]:
    """
    @brief Find the theorems of the same list a block cites from outside itself.
    @details
    Gauss chapters cite Peano theorems of the same `global_theorem_list.txt`;
    the certificate builder requires exactly that set as the dependency's
    `required_source_indices`. Citations are resolved alpha-equivalently over
    the whole list; a citation matching no list theorem is left to the
    dependency certificates (the builder's own rule).
    @param records Loaded theorem registry.
    @param parsed Parsed theorem records of the block.
    @param first First source index of the block.
    @param last Last source index of the block.
    @return Sorted source indices outside the block that the block cites.
    """

    by_alpha: dict[str, list[int]] = {}
    for record in records:
        by_alpha.setdefault(alpha_key(record.expression), []).append(record.source_index)
    imported: set[int] = set()
    for theorem in parsed:
        for cited in theorem_citations(theorem):
            matches = by_alpha.get(alpha_key(cited), [])
            if not matches:
                continue
            assert len(matches) == 1, f"citation of source {theorem.source_index} is alpha-ambiguous"
            if not first <= matches[0] <= last:
                imported.add(matches[0])
    return sorted(imported)


def _rename_heads(expression: Expression, aliases: dict[str, str]) -> Expression:
    """
    @brief Rewrite compiled heads of an expression under an alias map.
    @param expression Expression in the target (shortcut) vocabulary.
    @param aliases Target head → source head.
    @return The expression in the source vocabulary; shape and arguments untouched.
    """

    if isinstance(expression, Atom):
        return Atom(aliases.get(expression.head, expression.head), expression.arguments)
    if isinstance(expression, Implication):
        return Implication(
            expression.bound_variables,
            _rename_heads(expression.premise, aliases),
            _rename_heads(expression.conclusion, aliases),
        )
    if hasattr(expression, "parts"):
        return type(expression)(tuple(_rename_heads(part, aliases) for part in expression.parts))
    assert hasattr(expression, "inner")
    return type(expression)(_rename_heads(expression.inner, aliases))


def _definition_matches(
    target_entry: dict[str, object],
    source_entry: dict[str, object],
    aliases: dict[str, str],
) -> bool:
    """
    @brief Decide whether two compiled definitions are identical under an alias map.
    @details Arity and category must agree, and every element of the target
    definition, with its heads renamed, must equal the source element text.
    @param target_entry Definition in the target vocabulary.
    @param source_entry Definition in the source vocabulary.
    @param aliases Target head → source head, applied inside the elements.
    @return True when the definitions are the same definition.
    """

    if int(target_entry["arity"]) != int(source_entry["arity"]):
        return False
    if str(target_entry["category"]) != str(source_entry["category"]):
        return False
    renamed = [
        to_mpl(_rename_heads(parse_mpl(str(element)), aliases))
        for element in target_entry["elements"]
    ]
    return renamed == [str(element) for element in source_entry["elements"]]


def infer_head_aliases(
    target: Expression,
    source_definitions: dict[str, dict[str, object]],
    target_definitions: dict[str, dict[str, object]],
) -> dict[str, str] | None:
    """
    @brief Find the source names of the compiled heads a cited theorem uses.
    @details
    A fresh compilation names identical compiled definitions differently
    (`existence11` in the shortcut is Peano's `existence3`). Every non-atomic
    head reachable from the theorem — through its atoms and, transitively,
    through the elements of its definitions — is either defined identically in
    the source (same name, same body under the aliases found so far) or must
    alias exactly one source definition with the same body. The map is closed
    by iteration so inner heads resolve before the definitions that mention
    them. Two source twins for one head is an ambiguity and asserts.
    @param target Cited theorem in the target vocabulary.
    @param source_definitions Definitions of the dependency certificate.
    @param target_definitions Completed binary entries of the citing corpus.
    @return Target head → source head for every renamed head, or None when some
        reachable head has no source twin (this certificate cannot supply the
        theorem).
    """

    reachable: list[str] = []
    pending = [atom.head for atom in iter_atoms(target)]
    while pending:
        head = pending.pop()
        if head in reachable or head not in target_definitions:
            continue
        entry = target_definitions[head]
        if str(entry.get("category", "")) == "atomic":
            continue
        reachable.append(head)
        for element in entry["elements"]:
            pending.extend(atom.head for atom in iter_atoms(parse_mpl(str(element))))

    aliases: dict[str, str] = {}
    unresolved = list(reachable)
    changed = True
    while changed and unresolved:
        changed = False
        for head in list(unresolved):
            entry = target_definitions[head]
            identical = source_definitions.get(head)
            if identical is not None and _definition_matches(entry, identical, aliases):
                unresolved.remove(head)
                changed = True
                continue
            twins = [
                name
                for name, source_entry in sorted(source_definitions.items())
                if _definition_matches(entry, source_entry, aliases)
            ]
            assert len(twins) <= 1, f"head {head!r} has several source twins: {twins}"
            if twins:
                aliases[head] = twins[0]
                unresolved.remove(head)
                changed = True
    if unresolved:
        return None
    return aliases


def certificate_theorems_by_alpha(certificate: dict[str, object]) -> dict[str, list[dict[str, object]]]:
    """
    @brief Index a certificate's theorems by their alpha-normalized statement.
    @param certificate Loaded dependency certificate.
    @return Alpha key → theorem records of the certificate.
    """

    by_alpha: dict[str, list[dict[str, object]]] = {}
    for theorem in certificate["theorems"]:
        key = alpha_key(parse_mpl(str(theorem["theorem"]["mpl"])))
        by_alpha.setdefault(key, []).append(theorem)
    return by_alpha


def external_references(
    parsed: list[TheoremRecord],
    local_records: list[TheoremRecord],
    dependencies: list[tuple[str, dict[str, object]]],
    target_definitions: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    """
    @brief Resolve every external citation of a block against dependency certificates.
    @details
    A citation that is no theorem of the block's own list must be a theorem of
    one dependency certificate: first by alpha-equivalence, else by the
    structural adaptation the validator accepts (inferred head aliases, one
    bijective renaming, one premise permutation). Exactly one dependency
    theorem may match; the adaptation is recorded in the selection format
    (`source_index`, `target_theorem`, `head_aliases`), one per source theorem.
    @param parsed Parsed theorem records of the block.
    @param local_records Theorem records of the block's own list.
    @param dependencies (certificate id, loaded certificate) pairs in priority order.
    @param target_definitions Completed binary entries of the citing corpus.
    @return Certificate id → {"required_source_indices": [...],
        "theorem_adaptations": [...]} in the selection format.
    """

    local_alpha = {alpha_key(record.expression) for record in local_records}
    by_alpha = {identifier: certificate_theorems_by_alpha(certificate) for identifier, certificate in dependencies}
    by_index = {
        identifier: {int(theorem["source_index"]): theorem for theorem in certificate["theorems"]}
        for identifier, certificate in dependencies
    }
    required: dict[str, set[int]] = {identifier: set() for identifier, _ in dependencies}
    adaptations: dict[str, dict[int, dict[str, object]]] = {identifier: {} for identifier, _ in dependencies}
    for theorem in parsed:
        for row_tag, cited in theorem_citations_with_tags(theorem):
            key = alpha_key(cited)
            if key in local_alpha:
                continue
            exact = [
                (identifier, matches)
                for identifier, _ in dependencies
                for matches in [by_alpha[identifier].get(key, [])]
                if matches
            ]
            if exact:
                identifier, matches = exact[0]
                # A statement proved twice in the dependency corpus (once as
                # an or theorem, once directly) is one proposition; the
                # lowest source index is its deterministic representative.
                required[identifier].add(min(int(match["source_index"]) for match in matches))
                continue
            found: list[tuple[str, int, dict[str, str], bool]] = []
            for identifier, certificate in dependencies:
                source_definitions = certificate["binary_definitions"]
                aliases = infer_head_aliases(cited, source_definitions, target_definitions)
                if aliases is None:
                    continue
                for candidate in certificate["theorems"]:
                    source = parse_mpl(str(candidate["theorem"]["mpl"]))
                    if theorem_adaptation_solutions(
                        source, cited, source_definitions, target_definitions, aliases,
                    ):
                        found.append((identifier, int(candidate["source_index"]), aliases, False))
            if not found:
                # No alias-and-permutation match: compare registry-independent
                # base forms (a compact renamed, re-parameterized or re-ordered
                # by the citing batch's own compilation).
                for identifier, certificate in dependencies:
                    source_definitions = certificate["binary_definitions"]
                    for candidate in certificate["theorems"]:
                        source = parse_mpl(str(candidate["theorem"]["mpl"]))
                        if base_form_equivalent(source, cited, source_definitions, target_definitions):
                            found.append((identifier, int(candidate["source_index"]), {}, True))
            if not found and row_tag == "or theorem":
                # The head-switched companion of a single-direction or theorem
                # (D-217): an alpha-permutation of the proved parent that may
                # never have been proved on its own; the chapter builder drops
                # the citation when it resolves nowhere, and so does this pass.
                continue
            assert found, (
                f"source {theorem.source_index} cites {to_mpl(cited)}, which matches "
                "no dependency theorem"
            )
            # Several matches are admissible only when they are one statement
            # proved more than once in one corpus; the lowest source index is
            # the deterministic representative.
            statements = {
                (identifier, alpha_key(parse_mpl(str(by_index[identifier][index]["theorem"]["mpl"]))))
                for identifier, index, _, _ in found
            }
            assert len(statements) == 1, (
                f"source {theorem.source_index} cites {to_mpl(cited)}, which matches "
                f"{len(found)} different dependency theorems"
            )
            identifier, index, aliases, by_base_form = min(found, key=lambda item: (item[0], item[1]))
            adaptation = {
                "source_index": index,
                "target_theorem": to_mpl(cited),
                "head_aliases": dict(sorted(aliases.items())),
            }
            if by_base_form:
                adaptation["base_form"] = True
            existing = adaptations[identifier].get(index)
            assert existing is None or existing == adaptation, (
                f"source theorem {index} of {identifier} is cited in two adapted spellings"
            )
            adaptations[identifier][index] = adaptation
            required[identifier].add(index)
    return {
        identifier: {
            "required_source_indices": sorted(required[identifier]),
            "theorem_adaptations": [
                adaptations[identifier][index] for index in sorted(adaptations[identifier])
            ],
        }
        for identifier, _ in dependencies
    }


def _write_json(path: Path, document: dict[str, object]) -> None:
    """
    @brief Write one selection or manifest document with stable formatting.
    @param path Output file; parent directories are created.
    @param document JSON-serializable document.
    @return None.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8", newline="\n")


def _fresh_export_directory(out_dir: Path) -> Path:
    """
    @brief Create the mode's empty `lean_export/` folder with the Lake skeleton.
    @details The previous export of the same mode is removed first; the project
    files that no certificate generates (lakefile, toolchain pin, manifest,
    proof support) are copied from the tracked project.
    @param out_dir The mode's `full_proof_graph` folder.
    @return The export folder.
    """

    export_dir = out_dir / EXPORT_FOLDER
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True)
    for relative in PROJECT_SKELETON:
        source = TRACKED_LEAN_PROJECT / relative
        assert source.is_file(), f"tracked Lean project file missing: {source}"
        target = export_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return export_dir


def _skip_without_toolchain(out_dir: Path) -> None:
    """
    @brief Apply the no-toolchain policy: one notice, no export, no stale folder.
    @details A previous run's export under `out_dir` would otherwise survive
    and be mistaken for this run's; it is removed so the HTML pages of this
    run carry no Lean links.
    @param out_dir The mode's `full_proof_graph` folder.
    @return None.
    """

    print(f"[lean-export] Lean toolchain not found — Lean export skipped; {INSTALL_HINT}")
    stale = out_dir / EXPORT_FOLDER
    if stale.exists():
        shutil.rmtree(stale)


def kernel_check(lake: Path, project_dir: Path, fta_module: bool) -> None:
    """
    @brief Kernel-check an exported Lean project and record the log.
    @details `lake build` checks the default target (Peano and Gauss); the FTA
    module is outside that target and is compiled separately. Output goes to
    `kernel_check.log` in the project; a non-zero exit fails the run.
    @param lake The Lake executable.
    @param project_dir Exported Lean project.
    @param fta_module Whether `GLExport/Generated/FTA.lean` must be compiled too.
    @return None.
    """

    commands: list[list[str]] = [[str(lake), "build"]]
    if fta_module:
        commands.append([str(lake), "env", "lean", os.path.join("GLExport", "Generated", "FTA.lean")])
    log = project_dir / "kernel_check.log"
    with open(log, "w", encoding="utf-8") as handle:
        for command in commands:
            handle.write("$ " + " ".join(command) + "\n")
            handle.flush()
            completed = subprocess.run(
                command, cwd=project_dir, stdout=handle, stderr=subprocess.STDOUT, check=False,
            )
            handle.write(f"exit {completed.returncode}\n")
            assert completed.returncode == 0, (
                f"Lean kernel check failed: {' '.join(command)} (see {log})"
            )
    print(f"[lean-export] kernel check passed: {project_dir}")


def export_main_graph(
    proc_dir: Path,
    out_dir: Path,
    config_dir: Path,
    binaries_dir: Path,
) -> Path | None:
    """
    @brief Export and kernel-check the full run's Peano and Gauss theorems.
    @details
    Writes `<out_dir>/lean_export/`: the two live selections, the Peano
    certificate, the Gauss certificate (depending on the Peano one by hash and
    by the Peano theorems Gauss cites), the Lean project with both generated
    modules and manifests, and the kernel-check log. Without a Lean toolchain
    nothing is written and one notice is printed.
    @param proc_dir The main `processed_proof_graph` folder.
    @param out_dir The main `full_proof_graph` folder.
    @param config_dir `files/config`.
    @param binaries_dir `files/GL_binaries` of this run.
    @return The export folder, or None when the toolchain is absent.
    """

    lake = find_lake()
    if lake is None:
        _skip_without_toolchain(out_dir)
        return None
    export_dir = _fresh_export_directory(out_dir)
    records = load_theorem_records(proc_dir)
    blocks = theorem_list_blocks(records)
    assert "AnchorPeano" in blocks and "AnchorGauss" in blocks, f"unexpected anchor blocks {blocks}"
    theorem_list_hash = sha256_file(proc_dir / "global_theorem_list.txt")

    peano_first, peano_last = blocks["AnchorPeano"]
    peano_id = f"peano_live_{peano_last - peano_first + 1}"
    peano_selection_path = export_dir / "selection_peano.json"
    _write_json(peano_selection_path, {
        "schema_version": 2,
        "id": peano_id,
        "theorem_list_sha256": theorem_list_hash,
        "source_range": {"first": peano_first, "last": peano_last},
        "excluded_sources": [],
    })
    peano_certificate_path = export_dir / "certificates" / "peano" / "certificate.json"
    peano_certificate = build_certificate(
        peano_selection_path, proc_dir, config_dir / "ConfigPeano.json", binaries_dir / "GL_binary_Peano.json",
    )
    write_certificate(peano_certificate, peano_certificate_path)
    write_lean(peano_certificate, export_dir)
    print(f"[lean-export] {peano_id}: {peano_certificate['corpus']['coverage']}")

    gauss_first, gauss_last = blocks["AnchorGauss"]
    gauss_id = f"gauss_live_{gauss_last - gauss_first + 1}"
    parsed = parsed_block(records, gauss_first, gauss_last)
    gauss_selection_path = export_dir / "selection_gauss.json"
    _write_json(gauss_selection_path, {
        "schema_version": 3,
        "id": gauss_id,
        "theorem_id_prefix": "gauss_source",
        "theorem_name_prefix": "Gauss source",
        "theorem_list_sha256": theorem_list_hash,
        "source_range": {"first": gauss_first, "last": gauss_last},
        "excluded_sources": [],
        "certificate_dependencies": [{
            "id": peano_id,
            "certificate": "certificates/peano/certificate.json",
            "certificate_sha256": sha256_file(peano_certificate_path),
            "required_source_indices": same_list_imports(records, parsed, gauss_first, gauss_last),
        }],
    })
    gauss_certificate = build_certificate(
        gauss_selection_path, proc_dir, config_dir / "ConfigGauss.json", binaries_dir / "GL_binary_Gauss.json",
    )
    write_certificate(gauss_certificate, export_dir / "certificates" / "gauss" / "certificate.json")
    write_lean(gauss_certificate, export_dir)
    print(f"[lean-export] {gauss_id}: {gauss_certificate['corpus']['coverage']}")
    kernel_check(lake, export_dir, fta_module=False)
    return export_dir


def external_theorem_lists(
    externals_path: Path,
    export_dir: Path,
) -> list[tuple[str, dict[str, object]]]:
    """
    @brief Turn the shortcut's externals snapshot into dependency theorem lists.
    @details
    The shortcut proves its conjectures against previously proved theorems
    supplied in expanded base form; that snapshot is the FTA export's ONLY
    dependency (the shortcut is independent of the main path). Its rows are
    split by anchor into a Peano-anchored and a Gauss-anchored list, each
    written as a certificate-shaped document under
    `certificates/<kind>_externals/certificate.json` — an id whose prefix
    names the kind, the theorem rows as `theorem` records with sequential
    source indices, the snapshot's hash as `source.theorem_list_sha256`, no
    compiled definitions (base form has none), no dependency edges — so the
    certificate builder loads it like any dependency certificate and the
    Lean writer passes every cited row to the FTA theorem as a proposition
    parameter. An empty list for one anchor is written as well (nothing to
    cite from it).
    @param externals_path The tracked `externally_provided_theorems.txt`.
    @param export_dir The shortcut's export folder.
    @return (certificate id, document) pairs in Peano, Gauss order.
    """

    rows = [line.strip() for line in externals_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows, f"{externals_path} holds no theorem"
    snapshot_hash = sha256_file(externals_path)
    by_kind: dict[str, list[Expression]] = {"peano": [], "gauss": []}
    for row in rows:
        expression = parse_mpl(row)
        kind = ANCHOR_CORPORA[anchor_head(expression)]
        assert kind in by_kind, f"externals row under {anchor_head(expression)} cannot be a shortcut dependency"
        by_kind[kind].append(expression)
    documents: list[tuple[str, dict[str, object]]] = []
    for kind, expressions in by_kind.items():
        identifier = f"{kind}_externals_{len(expressions)}"
        document: dict[str, object] = {
            "schema": "gl-external-theorem-list",
            "schema_version": 2,
            "source": {
                "theorem_list": _portable(externals_path),
                "theorem_list_sha256": snapshot_hash,
            },
            "corpus": {
                "id": identifier,
                "coverage": {"theorems": len(expressions)},
                "dependency_edges": [],
            },
            "types": {},
            "binary_definitions": {},
            "theorems": [
                {
                    "id": f"{kind}_external_{index:03d}",
                    "name": f"{kind} external {index}",
                    "method": "external",
                    "source_index": index,
                    "theorem": expression_json(expression),
                }
                for index, expression in enumerate(expressions)
            ],
        }
        _write_json(export_dir / "certificates" / f"{kind}_externals" / "certificate.json", document)
        documents.append((identifier, document))
    return documents


def _portable(path: Path) -> str:
    """
    @brief Repository-relative label of an input file.
    @param path Input file.
    @return The path from the `files` marker on, else the file name.
    """

    parts = path.resolve().parts
    if "files" in parts:
        return Path(*parts[parts.index("files"):]).as_posix()
    return path.name


def export_shortcut_graph(
    proc_dir: Path,
    out_dir: Path,
    externals_path: Path,
    config_dir: Path,
    binaries_dir: Path,
) -> Path | None:
    """
    @brief Export and kernel-check the shortcut run's FTA theorems.
    @details
    Independent of the main path: the FTA corpus cites only the externals
    snapshot the shortcut proved against, so that snapshot becomes the
    certificate's dependency (`external_theorem_lists`), every external
    citation is resolved by content against it — alpha-equivalently or by
    registry-independent base form — and enters Lean as a proposition
    parameter of the citing theorem. The Lean project is self-contained: its
    own definitions module, the FTA module, `lake build` as the check.
    Writes `<out_dir>/lean_export/`. Without a Lean toolchain one notice is
    printed and nothing is written.
    @param proc_dir The shortcut `processed_proof_graph` folder.
    @param out_dir The shortcut `full_proof_graph` folder.
    @param externals_path The tracked externals snapshot the shortcut used.
    @param config_dir `files/config`.
    @param binaries_dir `files/GL_binaries` of this run.
    @return The export folder, or None when the toolchain is absent.
    """

    lake = find_lake()
    if lake is None:
        _skip_without_toolchain(out_dir)
        return None
    export_dir = _fresh_export_directory(out_dir)
    dependencies = external_theorem_lists(externals_path, export_dir)

    records = load_theorem_records(proc_dir)
    blocks = theorem_list_blocks(records)
    assert set(blocks) == {"AnchorFTA"}, f"shortcut theorem list holds {sorted(blocks)}"
    first, last = blocks["AnchorFTA"]
    parsed = parsed_block(records, first, last)
    binary_path = binaries_dir / "GL_binary_FTA.json"
    _, _, target_definitions = binary_entries_with_overrides(
        records, [(None, theorem) for theorem in parsed], binary_path,
    )
    references = external_references(
        parsed, list(records[first:last + 1]), dependencies, target_definitions,
    )
    fta_id = f"fta_live_{last - first + 1}"
    selection_path = export_dir / "selection_fta.json"
    _write_json(selection_path, {
        "schema_version": 3,
        "id": fta_id,
        "theorem_id_prefix": "fta_source",
        "theorem_name_prefix": "FTA shortcut source",
        "theorem_list_sha256": sha256_file(proc_dir / "global_theorem_list.txt"),
        "source_range": {"first": first, "last": last},
        "excluded_sources": [],
        "target_source_indices": list(range(first, last + 1)),
        "certificate_dependencies": [
            {
                "id": identifier,
                "certificate": f"certificates/{identifier.split('_')[0]}_externals/certificate.json",
                "certificate_sha256": sha256_file(
                    export_dir / "certificates" / f"{identifier.split('_')[0]}_externals" / "certificate.json"
                ),
                "theorem_list_scope": "external",
                **references[identifier],
            }
            for identifier, _ in dependencies
        ],
    })
    certificate = build_certificate(selection_path, proc_dir, config_dir / "ConfigFTA.json", binary_path)
    write_certificate(certificate, export_dir / "certificates" / "fta" / "certificate.json")
    write_lean(certificate, export_dir)
    print(f"[lean-export] {fta_id}: {certificate['corpus']['coverage']}")
    kernel_check(lake, export_dir, fta_module=False)
    return export_dir
