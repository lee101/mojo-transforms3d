"""Quaternion conversions and algebra compatible with transforms3d."""

from __future__ import annotations

import numpy as np

from ._lib import addr, f64, lib

_FLOAT_EPS = np.finfo(np.float64).eps


def _rows(value, width: int, name: str):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim < 1 or array.shape[-1] != width:
        raise ValueError(f"{name} must have trailing shape ({width},)")
    return f64(array).reshape(-1, width), array.shape[:-1]


def _unary_batch(value, symbol: str):
    rows, shape = _rows(value, 4, "quaternion")
    result = np.empty_like(rows)
    getattr(lib(), symbol)(addr(rows), addr(result), rows.shape[0])
    return result.reshape(shape + (4,))


def fillpositive(xyz, w2_thresh=None):
    if len(xyz) != 3:
        raise ValueError("xyz should have length 3")
    if w2_thresh is None:
        try:
            w2_thresh = -np.finfo(xyz.dtype).eps * 3
        except (AttributeError, ValueError):
            w2_thresh = -_FLOAT_EPS * 3
    xyz = np.asarray(xyz, dtype=np.float64)
    w2 = 1.0 - np.dot(xyz, xyz)
    if w2 < w2_thresh:
        raise ValueError(f"w2 should be positive, but is {w2:e}")
    w = 0.0 if w2 < 0 else np.sqrt(w2)
    return np.r_[w, xyz]


def quat2mat_batch(q):
    rows, shape = _rows(q, 4, "q")
    result = np.empty((rows.shape[0], 3, 3), dtype=np.float64)
    lib().mt3_quat2mat(addr(rows), addr(result), rows.shape[0])
    return result.reshape(shape + (3, 3))


def quat2mat(q):
    return quat2mat_batch(q)


def mat2quat_batch(M):
    matrix = np.asarray(M, dtype=np.float64)
    if matrix.ndim < 2 or matrix.shape[-2:] != (3, 3):
        raise ValueError("M must have trailing shape (3, 3)")
    shape = matrix.shape[:-2]
    matrix = f64(matrix).reshape(-1, 3, 3)
    # Match upstream's Bar-Itzhack eigensystem method.  The more common
    # trace/diagonal formula is only equivalent for exact rotation matrices
    # and can return a noticeably different answer for noisy input.
    qxx = matrix[:, 0, 0]
    qyx = matrix[:, 0, 1]
    qzx = matrix[:, 0, 2]
    qxy = matrix[:, 1, 0]
    qyy = matrix[:, 1, 1]
    qzy = matrix[:, 1, 2]
    qxz = matrix[:, 2, 0]
    qyz = matrix[:, 2, 1]
    qzz = matrix[:, 2, 2]
    k = np.zeros((matrix.shape[0], 4, 4), dtype=np.float64)
    k[:, 0, 0] = qxx - qyy - qzz
    k[:, 1, 0] = k[:, 0, 1] = qyx + qxy
    k[:, 1, 1] = qyy - qxx - qzz
    k[:, 2, 0] = k[:, 0, 2] = qzx + qxz
    k[:, 2, 1] = k[:, 1, 2] = qzy + qyz
    k[:, 2, 2] = qzz - qxx - qyy
    k[:, 3, 0] = k[:, 0, 3] = qyz - qzy
    k[:, 3, 1] = k[:, 1, 3] = qzx - qxz
    k[:, 3, 2] = k[:, 2, 3] = qxy - qyx
    k[:, 3, 3] = qxx + qyy + qzz
    _, vectors = np.linalg.eigh(k / 3.0)
    result = vectors[:, [3, 0, 1, 2], -1]
    result[result[:, 0] < 0] *= -1
    return result.reshape(shape + (4,))


def mat2quat(M):
    return mat2quat_batch(M)


def qmult_batch(q1, q2):
    first = np.asarray(q1, dtype=np.float64)
    second = np.asarray(q2, dtype=np.float64)
    if first.shape[-1:] != (4,) or second.shape[-1:] != (4,):
        raise ValueError("q1 and q2 must have trailing shape (4,)")
    shape = np.broadcast_shapes(first.shape[:-1], second.shape[:-1])
    first = f64(np.broadcast_to(first, shape + (4,))).reshape(-1, 4)
    second = f64(np.broadcast_to(second, shape + (4,))).reshape(-1, 4)
    result = np.empty_like(first)
    lib().mt3_qmult(addr(first), addr(second), addr(result), first.shape[0])
    return result.reshape(shape + (4,))


def qmult(q1, q2):
    return qmult_batch(q1, q2)


def qconjugate_batch(q):
    return _unary_batch(q, "mt3_qconjugate")


def qconjugate(q):
    return qconjugate_batch(q)


def qnorm_batch(q):
    rows, shape = _rows(q, 4, "q")
    result = np.empty(rows.shape[0], dtype=np.float64)
    lib().mt3_qnorm(addr(rows), addr(result), rows.shape[0])
    return result.reshape(shape)


def qnorm(q):
    return float(qnorm_batch(q))


def qisunit(q):
    return np.allclose(qnorm(q), 1)


def qinverse_batch(q):
    return _unary_batch(q, "mt3_qinverse")


def qinverse(q):
    return qinverse_batch(q)


def qeye(dtype=np.float64):
    return np.array([1.0, 0, 0, 0], dtype=dtype)


def qexp_batch(q):
    return _unary_batch(q, "mt3_qexp")


def qexp(q):
    return qexp_batch(q)


def qlog_batch(q):
    return _unary_batch(q, "mt3_qlog")


def qlog(q):
    return qlog_batch(q)


def qpow_batch(q, n):
    rows, shape = _rows(q, 4, "q")
    result = np.empty_like(rows)
    lib().mt3_qpow(addr(rows), addr(result), rows.shape[0], float(n))
    return result.reshape(shape + (4,))


def qpow(q, n):
    return qpow_batch(q, n)


def rotate_vector_batch(v, q, is_normalized=True):
    vectors = np.asarray(v, dtype=np.float64)
    quats = np.asarray(q, dtype=np.float64)
    if vectors.shape[-1:] != (3,) or quats.shape[-1:] != (4,):
        raise ValueError("v and q must have trailing shapes (3,) and (4,)")
    shape = np.broadcast_shapes(vectors.shape[:-1], quats.shape[:-1])
    vectors = f64(np.broadcast_to(vectors, shape + (3,))).reshape(-1, 3)
    quats = f64(np.broadcast_to(quats, shape + (4,))).reshape(-1, 4)
    result = np.empty_like(vectors)
    lib().mt3_rotate_vector(
        addr(vectors), addr(quats), addr(result), vectors.shape[0], int(is_normalized)
    )
    return result.reshape(shape + (3,))


def rotate_vector(v, q, is_normalized=True):
    return rotate_vector_batch(v, q, is_normalized)


def nearly_equivalent(q1, q2, rtol=1e-5, atol=1e-8):
    q1 = np.array(q1)
    q2 = np.array(q2)
    return np.allclose(q1, q2, rtol, atol) or np.allclose(-q1, q2, rtol, atol)


def axangle2quat_batch(vector, theta, is_normalized=False):
    axes = np.asarray(vector, dtype=np.float64)
    angles = np.asarray(theta, dtype=np.float64)
    if axes.shape[-1:] != (3,):
        raise ValueError("vector must have trailing shape (3,)")
    shape = np.broadcast_shapes(axes.shape[:-1], angles.shape)
    axes = f64(np.broadcast_to(axes, shape + (3,))).reshape(-1, 3)
    angles = f64(np.broadcast_to(angles, shape)).reshape(-1)
    result = np.empty((axes.shape[0], 4), dtype=np.float64)
    lib().mt3_axangle2quat(
        addr(axes), addr(angles), addr(result), axes.shape[0], int(is_normalized)
    )
    return result.reshape(shape + (4,))


def axangle2quat(vector, theta, is_normalized=False):
    return axangle2quat_batch(vector, float(theta), is_normalized)


def quat2axangle_batch(quat, identity_thresh=None):
    quats, shape = _rows(quat, 4, "quat")
    axes = np.empty((quats.shape[0], 3), dtype=np.float64)
    angles = np.empty(quats.shape[0], dtype=np.float64)
    threshold = _FLOAT_EPS * 3 if identity_thresh is None else float(identity_thresh)
    lib().mt3_quat2axangle(
        addr(quats), addr(axes), addr(angles), quats.shape[0], threshold
    )
    finite = np.isfinite(quats).all(axis=1)
    axes[~finite] = (1.0, 0.0, 0.0)
    angles[~finite] = np.nan
    return axes.reshape(shape + (3,)), angles.reshape(shape)


def quat2axangle(quat, identity_thresh=None):
    axis, angle = quat2axangle_batch(quat, identity_thresh)
    return axis, float(angle)
