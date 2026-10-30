# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt)
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Exact full-run FTA input capture and development replay."""

import hashlib
import json
import shutil
from pathlib import Path
from zipfile import ZipFile, ZIP_STORED


def source_fingerprints(root: Path) -> dict[str, str]:
    """@brief Identify the definitions, configurations and supplied FTA goals.

    @details
    A replay retains the compiler state produced by the preceding batches.
    Changing an input that produced that state requires a new full run.
    Production source may change between captures for development debugging.

    @param root Absolute project directory containing the pipeline inputs.
    @return Relative input paths mapped to their SHA-256 digests.
    @invariant The FTA configuration and its supplied goals exist.
    """
    paths = set()
    config_path = root / "files/config/ConfigFTA.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    goals = root / config["external_conjectures_file"]
    assert goals.is_file() and goals.read_text(encoding="utf-8").strip(), \
        f"FTA supplied goals are missing or empty: {goals}"
    paths.update({goals, config_path})
    for directory in ("files/config", "files/definitions"):
        paths.update(path for path in (root / directory).rglob("*") if path.is_file())
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def capture(root: Path) -> None:
    """@brief Save the exact native FTA boundary of a full pipeline run.

    @details
    Capture after conjecture preparation and compiler seeding, before native
    FTA mutates any output. Preserve proved and external theorem channels,
    compiler naming, auxiliary inputs and preceding proof chapters together.
    Each successful capture replaces the generated replay archive.

    @param root Absolute project directory containing the prepared FTA batch.
    @return None; writes files/shortcut/full_input.zip.
    @invariant The archive contains a nonempty seeded FTA compiler registry.
    @see restore
    """
    assert json.loads((root / "files/GL_binaries/GL_binary_FTA.json").read_text())
    destination = root / "files/shortcut/full_input.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": 1, "sources": source_fingerprints(root), "files": {}}
    with ZipFile(destination.with_suffix(".tmp"), "w", compression=ZIP_STORED) as archive:
        for directory in ("files/GL_binaries", "files/simple_facts",
                          "files/theorems", "files/raw_proof_graph"):
            assert (root / directory).is_dir(), directory
            for path in sorted((root / directory).rglob("*")):
                if path.is_file():
                    relative = path.relative_to(root).as_posix()
                    data = path.read_bytes()
                    archive.writestr(relative, data)
                    manifest["files"][relative] = hashlib.sha256(data).hexdigest()
        archive.writestr("manifest.json", json.dumps(manifest, sort_keys=True))
    destination.with_suffix(".tmp").replace(destination)


def restore(root: Path) -> None:
    """@brief Prepare shortcut output folders from the last full FTA boundary.

    @details
    Validate source and archive fingerprints before replacing generated files.
    Restore the shared compiler and simple facts byte for byte. Redirect the
    theorem pool and its prior proof chapters into the shortcut tree. The two
    historical durable input files remain intact; only the captured generated
    conjectures and compressed externals feed native replay.

    @param root Absolute project directory containing the replay archive.
    @return None; populates the generated shortcut and shared scratch folders.
    @invariant Missing, stale or damaged replay input fails before cleanup.
    @see capture
    """
    source = root / "files/shortcut/full_input.zip"
    assert source.is_file(), "FTA replay input missing; run a full main.py first"
    with ZipFile(source) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["schema"] == 1
        assert manifest["sources"] == source_fingerprints(root), (
            "FTA replay inputs changed; refresh with a full main.py run")
        assert set(archive.namelist()) == {*manifest["files"], "manifest.json"}
        for relative, digest in manifest["files"].items():
            assert hashlib.sha256(archive.read(relative)).hexdigest() == digest, relative
        required = {"files/GL_binaries/GL_binary_FTA.json",
                    "files/GL_binaries/GL_binary_shared.json",
                    "files/theorems/conjectures.txt", "files/theorems/theorems.txt",
                    "files/theorems/compressed_external_theorems.txt"}
        assert required <= manifest["files"].keys()
        durable = {"conjectures.txt", "externally_provided_theorems.txt"}
        for directory in ("files/GL_binaries", "files/simple_facts",
                          "files/shortcut/raw_proof_graph", "files/shortcut/theorems"):
            target = root / directory
            assert target.resolve().is_relative_to(root.resolve())
            target.mkdir(parents=True, exist_ok=True)
            for path in target.iterdir():
                if directory == "files/shortcut/theorems" and path.name in durable:
                    continue
                assert path.resolve().is_relative_to(target.resolve())
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
        for relative in manifest["files"]:
            if relative.startswith("files/theorems/"):
                name = relative.removeprefix("files/theorems/")
                if name == "externally_provided_theorems.txt":
                    continue
                if name == "conjectures.txt":
                    name = "replay_conjectures.txt"
                destination = root / "files/shortcut/theorems" / name
            elif relative.startswith("files/raw_proof_graph/"):
                destination = root / relative.replace("files/raw_proof_graph/",
                                                      "files/shortcut/raw_proof_graph/", 1)
            else:
                assert relative.startswith(("files/GL_binaries/", "files/simple_facts/"))
                destination = root / relative
            assert destination.resolve().is_relative_to(root.resolve())
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(relative))
