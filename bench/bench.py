"""mojo-transforms3d against transforms3d on identical inputs."""

from __future__ import annotations

import math
import os
import platform
import subprocess
import sys
import time
from importlib.metadata import version

import numpy as np

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"
    ),
)

from mojotransforms3d import euler, quaternions  # noqa: E402
from transforms3d import euler as ref_euler  # noqa: E402
from transforms3d import quaternions as ref_quat  # noqa: E402


def timeit(function, repeat=3):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf8") as stream:
            for line in stream:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def upstream_euler2mat(angles):
    return np.array([ref_euler.euler2mat(*row) for row in angles])


def upstream_quat2mat(quats):
    return np.array([ref_quat.quat2mat(row) for row in quats])


def upstream_qmult(first, second):
    return np.array([ref_quat.qmult(a, b) for a, b in zip(first, second)])


def upstream_rotate(vectors, quats):
    return np.array(
        [
            ref_quat.rotate_vector(vector, quat, is_normalized=False)
            for vector, quat in zip(vectors, quats)
        ]
    )


def scalar_loop(function, count=20_000):
    for _ in range(count):
        function()


def main():
    rng = np.random.default_rng(42)
    n = 100_000
    angles = np.ascontiguousarray(rng.uniform(-np.pi, np.pi, size=(n, 3)))
    quats = np.ascontiguousarray(rng.normal(size=(n, 4)))
    other = np.ascontiguousarray(rng.normal(size=(n, 4)))
    vectors = np.ascontiguousarray(rng.normal(size=(n, 3)))

    cases = [
        (
            "euler2mat scalar (20k calls)",
            lambda: scalar_loop(lambda: euler.euler2mat(0.2, -0.4, 0.7)),
            lambda: scalar_loop(lambda: ref_euler.euler2mat(0.2, -0.4, 0.7)),
        ),
        (
            "euler2mat batch (100k)",
            lambda: euler.euler2mat_batch(angles[:, 0], angles[:, 1], angles[:, 2]),
            lambda: upstream_euler2mat(angles),
        ),
        (
            "quat2mat batch (100k)",
            lambda: quaternions.quat2mat_batch(quats),
            lambda: upstream_quat2mat(quats),
        ),
        (
            "qmult batch (100k)",
            lambda: quaternions.qmult_batch(quats, other),
            lambda: upstream_qmult(quats, other),
        ),
        (
            "rotate_vector batch (100k)",
            lambda: quaternions.rotate_vector_batch(
                vectors, quats, is_normalized=False
            ),
            lambda: upstream_rotate(vectors, quats),
        ),
    ]

    euler.euler2mat(0.2, -0.4, 0.7)
    mojo_version = subprocess.run(
        ["mojo", "--version"], check=True, capture_output=True, text=True
    ).stdout.strip()
    print(f"Machine: {cpu_name()}; {platform.system()} {platform.machine()}")
    print(
        f"Software: {mojo_version}; transforms3d {version('transforms3d')}; "
        f"NumPy {np.__version__}"
    )
    print()
    print("| case | Mojo | transforms3d | speedup |")
    print("| --- | ---: | ---: | ---: |")
    for name, mojo_fn, upstream_fn in cases:
        mojo_seconds = timeit(mojo_fn)
        upstream_seconds = timeit(upstream_fn)
        ratio = upstream_seconds / mojo_seconds
        print(
            f"| {name} | {mojo_seconds * 1e3:.2f} ms | "
            f"{upstream_seconds * 1e3:.2f} ms | {ratio:.2f}x |"
        )


if __name__ == "__main__":
    main()
