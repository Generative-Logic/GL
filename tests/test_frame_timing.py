# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt)
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Direct unit tests for the frame-timing ledger and report aggregation."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from frame_timing import (  # noqa: E402
    StageTimer, configure_frame_timing, record_frame_timing,
)


def load_report_module():
    path = REPO_ROOT / ".scripts" / "frame_timing_report.py"
    spec = importlib.util.spec_from_file_location("frame_timing_report", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_tests() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ledger = configure_frame_timing(Path(tmp) / "frame.jsonl")
        record_frame_timing(
            "python.child", 0.25, parent="python.parent", batch="Peano", count=2
        )
        with StageTimer("python.scope", parent="python.parent", batch="Peano"):
            pass
        record_frame_timing("python.parent", 1.0, parent="run.total", batch="Peano")
        record_frame_timing(
            "python.global_child", 0.1, parent="run.total", batch="Peano"
        )
        record_frame_timing("run.total", 1.5, parent="root")

        rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
        assert rows[0] == {
            "batch": "Peano", "count": 2, "excluded": False,
            "parent": "python.parent", "seconds": 0.25,
            "stage": "python.child",
        }
        assert rows[1]["stage"] == "python.scope"
        assert rows[1]["seconds"] >= 0.0

        report = load_report_module()
        aggregated = report.aggregate(report.load_records(ledger))
        assert sum(row["count"] for row in aggregated if row["stage"] == "python.child") == 2
        residuals = report.parent_residuals(aggregated)
        parent = next(row for row in residuals if row[0] == "python.parent")
        assert abs(parent[2] - 1.0) < 1e-12
        assert 0.0 <= parent[3] <= 0.75
        run_parent = next(row for row in residuals if row[0] == "run.total")
        assert run_parent[3] < 0.5

    os.environ.pop("GL_FRAME_TIMING_PATH", None)


if __name__ == "__main__":
    run_tests()
    print("frame timing tests passed")
