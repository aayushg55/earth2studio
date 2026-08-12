# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from typing import ContextManager

CPU_TIMING_ENABLED = os.environ.get("EARTH2STUDIO_CPU_TIMING", "0") == "1"


class _TimingAccumulator:
    """Accumulate elapsed time by region."""

    def __init__(self, label: str) -> None:
        self._label = label
        self.total_ms: defaultdict[str, float] = defaultdict(float)
        self.calls: defaultdict[str, int] = defaultdict(int)

    def add(self, tag: str, ms: float) -> None:
        self.total_ms[tag] += ms
        self.calls[tag] += 1

    def reset(self) -> None:
        self.total_ms.clear()
        self.calls.clear()

    def report(self) -> str:
        """Format accumulated region times."""
        if not self.calls:
            return f"[{self._label}] no regions recorded"
        width = max(len(tag) for tag in self.calls)
        grand_total = sum(self.total_ms.values())
        lines = [f"[{self._label}] mean time per region:"]
        for tag in sorted(self.total_ms, key=self.total_ms.__getitem__, reverse=True):
            count = self.calls[tag]
            total = self.total_ms[tag]
            mean = total / count
            percent = 100.0 * total / grand_total if grand_total else 0.0
            lines.append(
                f"  {tag:<{width}}  {mean:8.3f} ms/call   "
                f"total {total:9.1f} ms  {percent:5.1f}%  (n={count})"
            )
        lines.append(
            f"  {'TOTAL':<{width}}  {'':>8}            total {grand_total:9.1f} ms"
        )
        return "\n".join(lines)


class _CpuTimer:
    """Measure wall time for named CPU regions."""

    def __init__(self) -> None:
        self._accumulator = _TimingAccumulator("cpu-timing")

    @contextmanager
    def range(self, tag: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self._accumulator.add(tag, (time.perf_counter() - start) * 1e3)

    def reset(self) -> None:
        self._accumulator.reset()

    def report(self) -> str:
        """Format accumulated region times."""
        return self._accumulator.report()

    def snapshot(self) -> dict[str, tuple[float, int]]:
        return {
            tag: (self._accumulator.total_ms[tag], count)
            for tag, count in self._accumulator.calls.items()
        }

    def merge(self, snapshot: dict[str, tuple[float, int]]) -> None:
        for tag, (total_ms, count) in snapshot.items():
            self._accumulator.total_ms[tag] += total_ms
            self._accumulator.calls[tag] += count


_CPU_TIMER = _CpuTimer()


def cpu_timing_range(tag: str, enabled: bool | None = None) -> ContextManager[None]:
    """Return a wall-clock timing context."""
    use_timing = CPU_TIMING_ENABLED if enabled is None else enabled
    if use_timing:
        return _CPU_TIMER.range(tag)
    return nullcontext()


def cpu_timing_report() -> str:
    """Format accumulated CPU region times."""
    return _CPU_TIMER.report()


def cpu_timing_reset() -> None:
    """Clear accumulated CPU region times."""
    _CPU_TIMER.reset()


def cpu_timing_snapshot() -> dict[str, tuple[float, int]]:
    """Accumulated region times as ``{tag: (total_ms, calls)}``."""
    return _CPU_TIMER.snapshot()


def cpu_timing_drain() -> dict[str, tuple[float, int]]:
    """Accumulated region times, clearing them.

    A worker returning one snapshot per unit of work must drain rather than
    snapshot; the registry is cumulative, so repeated snapshots of the same worker
    would each carry its earlier work.
    """
    snapshot = _CPU_TIMER.snapshot()
    _CPU_TIMER.reset()
    return snapshot


def cpu_timing_merge(snapshot: dict[str, tuple[float, int]]) -> None:
    """Add another process's snapshot to this process's totals.

    Regions merged from parallel workers sum CPU time across workers, so their
    totals exceed the wall time the parent spent waiting.
    """
    _CPU_TIMER.merge(snapshot)
