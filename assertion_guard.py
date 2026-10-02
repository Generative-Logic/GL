# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt)
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Refuse to run GL with Python's assert statements removed."""

import sys


def require_assertions(program: str) -> None:
    """@brief Stop an entry point that runs with Python's assertions removed.

    @details
    GL states its contracts as `assert` statements: a broken contract stops
    the run where it breaks, and the verifier, the Lean export and the
    release tooling reject malformed input through them. `python -O`,
    `python -OO` and a non-empty `PYTHONOPTIMIZE` remove every `assert`
    statement, so a program started that way would accept what it exists to
    reject and still report success. Each entry point calls this first; with
    assertions removed it writes one line to stderr and exits with status 2
    before any work. The check cannot itself be an `assert`.

    @param program The entry point's name, shown in the message.
    @return None when assertions are enabled; otherwise the process exits
            with status 2.
    """
    if not __debug__:
        print(f"{program}: GL's checks are Python assert statements, which "
              "python -O and PYTHONOPTIMIZE remove; run without them.",
              file=sys.stderr)
        sys.exit(2)
