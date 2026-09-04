# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt)
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Command-line entry point for independent GL-to-Lean proof export."""

from __future__ import annotations

import argparse
from pathlib import Path

from .certificate import build_certificate, write_certificate
from .lean import write_lean


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


def _arguments() -> argparse.Namespace:
    """
    @brief Parse the explicit inputs defining one reproducible Lean export.
    @details
    The proof graph is an explicit read-only input; no GL executable or pipeline
    stage is invoked by this command.
    @return Parsed command-line namespace.
    """

    parser = argparse.ArgumentParser(description="Export a processed GL proof graph to Lean 4")
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--proof-graph", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "files" / "config" / "ConfigPeano.json",
    )
    parser.add_argument(
        "--gl-binary",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "tests"
            / "fixtures"
            / "GL_binaries"
            / "GL_binary_Peano.json"
        ),
    )
    parser.add_argument("--certificate-output", type=Path, required=True)
    parser.add_argument(
        "--lean-project",
        type=Path,
        default=REPOSITORY_ROOT / "lean_export",
    )
    return parser.parse_args()


def main() -> None:
    """
    @brief Build the neutral certificate and render the Lean project.
    @details
    Certificate construction independently parses copied proof artifacts. The
    Lean renderer then writes ordinary predicate definitions and named row facts.
    @return None.
    """

    arguments = _arguments()
    certificate = build_certificate(
        arguments.selection.resolve(),
        arguments.proof_graph.resolve(),
        arguments.config.resolve(),
        arguments.gl_binary.resolve(),
    )
    arguments.certificate_output.parent.mkdir(parents=True, exist_ok=True)
    write_certificate(certificate, arguments.certificate_output.resolve())
    write_lean(certificate, arguments.lean_project.resolve())


if __name__ == "__main__":
    main()
