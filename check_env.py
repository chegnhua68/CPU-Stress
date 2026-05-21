#!/usr/bin/env python3
"""
Environment checker for the OpenCV CPU benchmark.

This script uses only the Python standard library in the parent process and
loads NumPy/OpenCV in a child process.  That makes dependency crashes easier to
diagnose on unusual Python builds or embedded Linux images.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import textwrap


def main() -> int:
    print("Python executable:", sys.executable)
    print("Python version:", sys.version.replace("\n", " "))
    print("Platform:", platform.platform())
    print("Machine:", platform.machine())
    print("CPU count:", os.cpu_count())

    if sys.version_info[:2] != (3, 11):
        print("WARNING: this project is intended for Python 3.11.")

    probe = textwrap.dedent(
        """
        import cv2
        import numpy as np

        print("NumPy:", np.__version__)
        print("OpenCV:", cv2.__version__)
        print("OpenCV threads:", cv2.getNumThreads())
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(gray, 8, cv2.CV_32S)
        print("OpenCV smoke test: OK")
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", probe],
        check=False,
        capture_output=True,
        text=True,
    )

    print()
    print("Dependency probe")
    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr)

    if result.returncode:
        print(f"Dependency probe failed with exit code {result.returncode}.", file=sys.stderr)
        return result.returncode

    print("Environment looks ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
