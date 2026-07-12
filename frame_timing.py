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

"""Cross-language-compatible wall-time records for the GL frame pipeline."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Optional, Type


TIMING_PATH_ENV = "GL_FRAME_TIMING_PATH"
_WRITE_LOCK = threading.Lock()


def configure_frame_timing(path: str | Path) -> Path:
    """@brief Start a fresh frame-timing JSON-lines ledger.

    @details
    Resolves ``path``, creates its parent directory, truncates any prior
    ledger, and publishes the absolute path through ``GL_FRAME_TIMING_PATH``
    so native subprocesses append to the same run. This is called exactly
    once by ``main.py`` before any measured stage starts.

    @param path Destination JSON-lines path for this ``main.py`` run.
    @return Absolute resolved ledger path.
    @invariant A configured run starts from an empty ledger.
    @see record_frame_timing
    """
    resolved = Path(path).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text("", encoding="utf-8")
    os.environ[TIMING_PATH_ENV] = str(resolved)
    return resolved


def record_frame_timing(stage: str, seconds: float, *, parent: str,
                        batch: str = "", excluded: bool = False,
                        count: int = 1) -> None:
    """@brief Append one deterministic frame-timing record.

    @details
    Writes a compact, key-sorted JSON object to the configured ledger. An
    absent ``GL_FRAME_TIMING_PATH`` deliberately disables instrumentation;
    this is the defined non-measurement mode used by standalone unit tests.

    @param stage Stable dotted stage name.
    @param seconds Elapsed wall time in seconds.
    @param parent Stable dotted parent stage used for residual accounting.
    @param batch Optional configuration tag or pipeline label.
    @param excluded Whether the measured stage is outside this campaign.
    @param count Number of homogeneous operations represented by the record.
    @return None.
    @invariant Stage times are finite and non-negative; counts are positive.
    @see StageTimer
    """
    assert stage and parent, "frame timing requires non-empty stage and parent"
    assert seconds >= 0.0, "frame timing seconds must be non-negative"
    assert count > 0, "frame timing count must be positive"
    path = os.environ.get(TIMING_PATH_ENV)
    if path is None:
        return
    payload = {
        "batch": batch,
        "count": count,
        "excluded": excluded,
        "parent": parent,
        "seconds": seconds,
        "stage": stage,
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    with _WRITE_LOCK:
        with open(path, "a", encoding="utf-8", newline="\n") as output:
            output.write(encoded)
            output.write("\n")


class StageTimer:
    """@brief Context manager that records one wall-time stage.

    @details
    Uses ``time.perf_counter_ns`` for monotonic wall time and records even when
    the measured body raises, preserving the last completed measurement in a
    crash log without swallowing or transforming the exception.

    @param stage Stable dotted stage name.
    @param parent Stable dotted parent stage.
    @param batch Optional configuration tag or pipeline label.
    @param excluded Whether the stage is outside this acceleration campaign.
    @param count Number of homogeneous operations represented by the scope.
    @invariant ``__exit__`` never suppresses an exception from the body.
    @see record_frame_timing
    """

    def __init__(self, stage: str, *, parent: str, batch: str = "",
                 excluded: bool = False, count: int = 1) -> None:
        self.stage = stage
        self.parent = parent
        self.batch = batch
        self.excluded = excluded
        self.count = count
        self._started_ns = 0

    def __enter__(self) -> "StageTimer":
        """@brief Start the monotonic stage clock.

        @details Stores the current ``perf_counter_ns`` reading on this scope.
        @return This timer instance.
        @invariant A scope has one start reading before it can exit.
        @see __exit__
        """
        assert self._started_ns == 0, "StageTimer cannot be entered twice"
        self._started_ns = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]],
                 exc_value: Optional[BaseException], traceback: object) -> bool:
        """@brief Stop the clock and append the elapsed record.

        @details Records elapsed wall time, then returns ``False`` so any body
        exception continues to its original caller unchanged.

        @param exc_type Exception type raised by the body, when present.
        @param exc_value Exception instance raised by the body, when present.
        @param traceback Exception traceback supplied by the context protocol.
        @return Always ``False``; exceptions are never suppressed.
        @invariant The elapsed duration is measured from this scope's start.
        @see __enter__
        """
        assert self._started_ns != 0, "StageTimer exited before entry"
        elapsed = (time.perf_counter_ns() - self._started_ns) / 1_000_000_000.0
        record_frame_timing(
            self.stage,
            elapsed,
            parent=self.parent,
            batch=self.batch,
            excluded=self.excluded,
            count=self.count,
        )
        return False
