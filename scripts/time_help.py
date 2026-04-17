"""Measure `shapmonitor --help` startup time.

Used as a talk slide to demonstrate the impact of lazy imports.
Target: < 200ms cold start.
"""

from __future__ import annotations

import subprocess
import sys
import time

N_RUNS = 5


def main() -> None:
    times: list[float] = []

    for i in range(N_RUNS):
        start = time.perf_counter()
        subprocess.run(
            [sys.executable, "-m", "shapmonitor.cli", "--help"],
            capture_output=True,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)
        print(f"  Run {i + 1}: {elapsed_ms:.0f}ms")

    avg = sum(times) / len(times)
    best = min(times)
    worst = max(times)

    print(f"\n  Average: {avg:.0f}ms")
    print(f"  Best:    {best:.0f}ms")
    print(f"  Worst:   {worst:.0f}ms")

    if avg < 200:
        print("\n  ✓ PASS — under 200ms target")
    elif avg < 500:
        print("\n  ⚠ WARNING — over 200ms target but under 500ms")
    else:
        print("\n  ✗ FAIL — over 500ms, check for non-lazy imports")

    sys.exit(0 if avg < 500 else 1)


if __name__ == "__main__":
    main()
