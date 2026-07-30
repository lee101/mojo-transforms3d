"""Axis-angle conversions compatible with :mod:`transforms3d.axangles`."""

from __future__ import annotations

import numpy as np

from ._lib import addr, f64, lib


def axangle2mat_batch(axis, angle, is_normalized=False):
    axes = np.asarray(axis, dtype=np.float64)
    angles = np.asarray(angle, dtype=np.float64)
    if axes.shape[-1:] != (3,):
        raise ValueError("axis must have trailing shape (3,)")
    shape = np.broadcast_shapes(axes.shape[:-1], angles.shape)
    axes = f64(np.broadcast_to(axes, shape + (3,))).reshape(-1, 3)
    angles = f64(np.broadcast_to(angles, shape)).reshape(-1)
    result = np.empty((axes.shape[0], 3, 3), dtype=np.float64)
    lib().mt3_axangle2mat(
        addr(axes), addr(angles), addr(result), axes.shape[0], int(is_normalized)
    )
    return result.reshape(shape + (3, 3))


def axangle2mat(axis, angle, is_normalized=False):
    return axangle2mat_batch(axis, float(angle), is_normalized)


def axangle2aff(axis, angle, point=None):
    affine = np.eye(4)
    rotation = axangle2mat(axis, angle)
    affine[:3, :3] = rotation
    if point is not None:
        point = np.array(point[:3], dtype=np.float64)
        affine[:3, 3] = point - rotation @ point
    return affine


def mat2axangle(mat, unit_thresh=1e-5):
    matrix = np.asarray(mat, dtype=np.float64)
    values, vectors = np.linalg.eig(matrix.T)
    candidates = np.where(np.abs(values - 1.0) < unit_thresh)[0]
    if not len(candidates):
        raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
    direction = np.real(vectors[:, candidates[-1]]).squeeze()
    cosa = (np.trace(matrix) - 1.0) / 2.0
    if abs(direction[2]) > 1e-8:
        sina = (
            matrix[1, 0] + (cosa - 1.0) * direction[0] * direction[1]
        ) / direction[2]
    elif abs(direction[1]) > 1e-8:
        sina = (
            matrix[0, 2] + (cosa - 1.0) * direction[0] * direction[2]
        ) / direction[1]
    else:
        sina = (
            matrix[2, 1] + (cosa - 1.0) * direction[1] * direction[2]
        ) / direction[0]
    return direction, float(np.arctan2(sina, cosa))


def aff2axangle(aff):
    affine = np.asarray(aff, dtype=np.float64)
    direction, angle = mat2axangle(affine[:3, :3])
    values, vectors = np.linalg.eig(affine)
    candidates = np.where(np.abs(np.real(values) - 1.0) < 1e-8)[0]
    if not len(candidates):
        raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
    point = np.real(vectors[:, candidates[-1]]).squeeze()
    point /= point[3]
    return direction, angle, point
