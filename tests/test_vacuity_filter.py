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

"""Tests for ``run_modes._filter_vacuous_tainted`` — the vacuity taint
closure over the raw proof-chapter citation graph."""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import run_modes  # noqa: E402


def _write(p: Path, text: str) -> None:
    p.write_text(text, encoding="utf-8")


def test_taint_closure_retires_dependents() -> None:
    """A theorem citing a vacuous seed, and a theorem citing THAT theorem,
    both leave the pool (direct + transitive taint); an uninvolved theorem
    stays; the paired pool files shrink in lockstep; the ``tainted`` label
    lands in the artifact."""
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        tdir = base / "theorems"
        rdir = base / "raw"
        tdir.mkdir()
        rdir.mkdir()
        seed = "(>[1](a[1])(b[1]))"
        dep = "(>[1](c[1])(d[1]))"
        dep2 = "(>[1](e[1])(f[1]))"
        clean = "(>[1](g[1])(h[1]))"
        _write(tdir / "vacuous_theorems.txt",
               seed + "\tpremise contradiction\n")
        _write(rdir / "global_theorem_list.txt",
               dep + "\tdirect\t-1\n"
               + dep2 + "\tdirect\t-1\n"
               + clean + "\tdirect\t-1\n")
        _write(rdir / "0_direct_proof.txt",
               dep + "\tmain\ttask formulation\n"
               + seed + "\tmain\ttheorem\n")
        _write(rdir / "1_direct_proof.txt",
               dep2 + "\tmain\ttask formulation\n"
               + dep + "\tmain\ttheorem\n")
        _write(rdir / "2_direct_proof.txt",
               clean + "\tmain\ttask formulation\n")
        _write(tdir / "compiled_theorems.txt",
               dep + "\n" + dep2 + "\n" + clean + "\n")
        _write(tdir / "theorems.txt",
               "expanded_dep\nexpanded_dep2\nexpanded_clean\n")

        run_modes._filter_vacuous_tainted(tdir, rdir)

        compiled = (tdir / "compiled_theorems.txt").read_text(
            encoding="utf-8").splitlines()
        raw = (tdir / "theorems.txt").read_text(
            encoding="utf-8").splitlines()
        assert compiled == [clean], compiled
        assert raw == ["expanded_clean"], raw
        vac = (tdir / "vacuous_theorems.txt").read_text(
            encoding="utf-8").splitlines()
        assert dep + "\ttainted" in vac, vac
        assert dep2 + "\ttainted" in vac, vac
        assert clean + "\ttainted" not in vac, vac


def test_no_seeds_is_noop() -> None:
    """Missing or empty seed artifact leaves the pool untouched."""
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        tdir = base / "theorems"
        rdir = base / "raw"
        tdir.mkdir()
        rdir.mkdir()
        _write(tdir / "theorems.txt", "x\n")
        _write(tdir / "compiled_theorems.txt", "x\n")
        run_modes._filter_vacuous_tainted(tdir, rdir)  # no seed file
        _write(tdir / "vacuous_theorems.txt", "")
        run_modes._filter_vacuous_tainted(tdir, rdir)  # empty seed set
        assert (tdir / "theorems.txt").read_text(encoding="utf-8") == "x\n"


def test_incubator_config_names_its_raw_proof_dir() -> None:
    """The vacuity closure must scan the batch family's OWN raw proof graph:
    every incubator config's top-level ``raw_proof_graph_folder`` resolves
    through the reader as an attribute, and a main config (which omits the
    key) falls back to the main directory. Guards against the expression-map
    ``get`` mis-read that silently pointed every incubator batch's taint
    closure at ``files/raw_proof_graph``."""
    from configuration_reader import configuration_reader
    config_dir = run_modes.PROJECT_ROOT / "files" / "config"
    incubator_configs = sorted(config_dir.glob("ConfigIncubator*.json"))
    assert incubator_configs, "no incubator configs found"
    for cfg_path in incubator_configs:
        cfg = configuration_reader(cfg_path)
        assert cfg.raw_proof_graph_folder == "files/incubator/raw_proof_graph", \
            cfg_path.name
    main_cfg = configuration_reader(config_dir / "ConfigPeano.json")
    assert main_cfg.raw_proof_graph_folder == "files/raw_proof_graph"


if __name__ == "__main__":
    test_taint_closure_retires_dependents()
    test_no_seeds_is_noop()
    test_incubator_config_names_its_raw_proof_dir()
    print("3/3 vacuity-filter tests passed")
