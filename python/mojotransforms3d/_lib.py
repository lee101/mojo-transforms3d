"""ctypes bridge to the Mojo rotation kernels."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src", "transforms3d.mojo")
LIB = os.environ.get("MOJO_TRANSFORMS3D_LIB") or os.path.join(
    ROOT, "dist", "libmojo-transforms3d.so"
)

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mt3_euler2mat": ([I, I, I, I, I, I, I, I, I], None),
    "mt3_euler2mat_scalar": ([F, F, F, I, I], None),
    "mt3_euler2mat_sxyz_scalar": ([F, F, F, I], None),
    "mt3_mat2euler": ([I, I, I, I, I, I, I], None),
    "mt3_euler2quat": ([I, I, I, I, I, I, I, I, I], None),
    "mt3_quat2mat": ([I, I, I], None),
    "mt3_axangle2mat": ([I, I, I, I, I], None),
    "mt3_axangle2quat": ([I, I, I, I, I], None),
    "mt3_quat2axangle": ([I, I, I, I, F], None),
    "mt3_qmult": ([I, I, I, I], None),
    "mt3_qconjugate": ([I, I, I], None),
    "mt3_qnorm": ([I, I, I], None),
    "mt3_qinverse": ([I, I, I], None),
    "mt3_qexp": ([I, I, I], None),
    "mt3_qlog": ([I, I, I], None),
    "mt3_qpow": ([I, I, I, F], None),
    "mt3_rotate_vector": ([I, I, I, I, I], None),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    if os.environ.get("MOJO_TRANSFORMS3D_LIB"):
        if os.path.exists(LIB):
            return LIB
        raise BuildError(f"MOJO_TRANSFORMS3D_LIB does not exist: {LIB}")
    if not force and os.path.exists(LIB) and os.path.getmtime(LIB) >= os.path.getmtime(SRC):
        return LIB
    mojo = shutil.which("mojo")
    if mojo is None:
        raise BuildError("mojo compiler not found; run inside `pixi run`")
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    command = [mojo, "build", "--emit", "shared-lib", SRC, "-o", LIB]
    result = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    if result.returncode or not os.path.exists(LIB):
        raise BuildError((result.stderr or result.stdout).strip()[:4000])
    return LIB


_LIBRARY: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _LIBRARY
    if _LIBRARY is None:
        _LIBRARY = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_LIBRARY, name)
            function.argtypes = argtypes
            function.restype = restype
    return _LIBRARY


def f64(value) -> np.ndarray:
    return np.ascontiguousarray(value, dtype=np.float64)


def addr(array: np.ndarray) -> int:
    if array.dtype != np.float64 or not array.flags.c_contiguous:
        raise TypeError("FFI buffers must be C-contiguous float64 arrays")
    pointer = int(array.ctypes.data)
    if pointer == 0:
        raise ValueError("FFI buffers must have a non-null data pointer")
    return pointer
