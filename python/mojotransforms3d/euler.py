"""Euler conversions compatible with :mod:`transforms3d.euler`."""

from __future__ import annotations

import threading

import numpy as np

from ._lib import addr, f64, lib
from .axangles import axangle2mat
from .quaternions import quat2axangle, quat2mat

_NEXT_AXIS = [1, 2, 0, 1]
_AXES2TUPLE = {
    "sxyz": (0, 0, 0, 0),
    "sxyx": (0, 0, 1, 0),
    "sxzy": (0, 1, 0, 0),
    "sxzx": (0, 1, 1, 0),
    "syzx": (1, 0, 0, 0),
    "syzy": (1, 0, 1, 0),
    "syxz": (1, 1, 0, 0),
    "syxy": (1, 1, 1, 0),
    "szxy": (2, 0, 0, 0),
    "szxz": (2, 0, 1, 0),
    "szyx": (2, 1, 0, 0),
    "szyz": (2, 1, 1, 0),
    "rzyx": (0, 0, 0, 1),
    "rxyx": (0, 0, 1, 1),
    "ryzx": (0, 1, 0, 1),
    "rxzx": (0, 1, 1, 1),
    "rxzy": (1, 0, 0, 1),
    "ryzy": (1, 0, 1, 1),
    "rzxy": (1, 1, 0, 1),
    "ryxy": (1, 1, 1, 1),
    "ryxz": (2, 0, 0, 1),
    "rzxz": (2, 0, 1, 1),
    "rxyz": (2, 1, 0, 1),
    "rzyz": (2, 1, 1, 1),
}
_TUPLE2AXES = {value: key for key, value in _AXES2TUPLE.items()}
_TUPLE2CODE = {
    spec: spec[0] | spec[1] << 2 | spec[2] << 3 | spec[3] << 4
    for spec in _TUPLE2AXES
}
_AXES2CODE = {name: _TUPLE2CODE[spec] for name, spec in _AXES2TUPLE.items()}
_SCALAR_STATE = threading.local()
_EULER2MAT_SCALAR = None


def _axes_tuple(axes) -> tuple[int, int, int, int]:
    if isinstance(axes, str):
        try:
            return _AXES2TUPLE[axes]
        except KeyError:
            try:
                return _AXES2TUPLE[axes.lower()]
            except KeyError:
                raise KeyError(axes) from None
    axes = tuple(axes)
    _TUPLE2AXES[axes]
    return axes


def _axes_code(axes) -> int:
    if isinstance(axes, str):
        try:
            return _AXES2CODE[axes]
        except KeyError:
            try:
                return _AXES2CODE[axes.lower()]
            except KeyError:
                raise KeyError(axes) from None
    axes = tuple(axes)
    _TUPLE2AXES[axes]
    return _TUPLE2CODE[axes]


def _scalar_matrix():
    try:
        return _SCALAR_STATE.matrix
    except AttributeError:
        result = np.empty((3, 3), dtype=np.float64)
        state = result, addr(result)
        _SCALAR_STATE.matrix = state
        return state


def _angles(ai, aj, ak):
    ai, aj, ak = np.broadcast_arrays(
        np.asarray(ai, dtype=np.float64),
        np.asarray(aj, dtype=np.float64),
        np.asarray(ak, dtype=np.float64),
    )
    shape = ai.shape
    return f64(ai.ravel()), f64(aj.ravel()), f64(ak.ravel()), shape


def euler2mat_batch(ai, aj, ak, axes="sxyz"):
    ai, aj, ak, shape = _angles(ai, aj, ak)
    result = np.empty((ai.size, 3, 3), dtype=np.float64)
    spec = _axes_tuple(axes)
    lib().mt3_euler2mat(
        addr(ai), addr(aj), addr(ak), addr(result), ai.size, *spec
    )
    return result.reshape(shape + (3, 3))


def euler2mat(ai, aj, ak, axes="sxyz"):
    global _EULER2MAT_SCALAR
    function = _EULER2MAT_SCALAR
    if function is None:
        function = lib().mt3_euler2mat_scalar
        _EULER2MAT_SCALAR = function
    result, pointer = _scalar_matrix()
    function(float(ai), float(aj), float(ak), pointer, _axes_code(axes))
    return result.copy()


def _matrix_rows(mat):
    matrix = np.asarray(mat, dtype=np.float64)
    if matrix.ndim < 2 or matrix.shape[-2:] not in ((3, 3), (4, 4)):
        raise ValueError("expected matrices with trailing shape (3, 3) or (4, 4)")
    shape = matrix.shape[:-2]
    matrix = f64(matrix[..., :3, :3]).reshape(-1, 9)
    return matrix, shape


def mat2euler_batch(mat, axes="sxyz"):
    matrix, shape = _matrix_rows(mat)
    result = np.empty((matrix.shape[0], 3), dtype=np.float64)
    spec = _axes_tuple(axes)
    lib().mt3_mat2euler(addr(matrix), addr(result), matrix.shape[0], *spec)
    return result.reshape(shape + (3,))


def mat2euler(mat, axes="sxyz"):
    return tuple(mat2euler_batch(mat, axes))


def euler2quat_batch(ai, aj, ak, axes="sxyz"):
    ai, aj, ak, shape = _angles(ai, aj, ak)
    result = np.empty((ai.size, 4), dtype=np.float64)
    spec = _axes_tuple(axes)
    lib().mt3_euler2quat(
        addr(ai), addr(aj), addr(ak), addr(result), ai.size, *spec
    )
    return result.reshape(shape + (4,))


def euler2quat(ai, aj, ak, axes="sxyz"):
    return euler2quat_batch(float(ai), float(aj), float(ak), axes)


def quat2euler(quaternion, axes="sxyz"):
    return mat2euler(quat2mat(quaternion), axes)


def euler2axangle(ai, aj, ak, axes="sxyz"):
    return quat2axangle(euler2quat(ai, aj, ak, axes))


def axangle2euler(vector, theta, axes="sxyz"):
    return mat2euler(axangle2mat(vector, theta), axes)


class EulerFuncs:
    def __init__(self, axes):
        _axes_tuple(axes)
        self.axes = axes

    def euler2mat(self, ai, aj, ak):
        return euler2mat(ai, aj, ak, self.axes)

    def mat2euler(self, mat):
        return mat2euler(mat, self.axes)

    def euler2quat(self, ai, aj, ak):
        return euler2quat(ai, aj, ak, self.axes)

    def quat2euler(self, quat):
        return quat2euler(quat, self.axes)

    def euler2axangle(self, ai, aj, ak):
        return euler2axangle(ai, aj, ak, self.axes)

    def axangle2euler(self, vector, theta):
        return axangle2euler(vector, theta, self.axes)


sxyz = EulerFuncs("sxyz")
rzxz = EulerFuncs("rzxz")
physics = rzxz


class TBZYX(EulerFuncs):
    def __init__(self):
        super().__init__("szyx")


szyx = TBZYX()
