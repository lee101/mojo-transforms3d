"""Specialized extrinsic z-y-x Euler convention."""

from __future__ import annotations

import math

import numpy as np

from .axangles import axangle2mat
from .euler import (
    axangle2euler as _axangle2euler,
    euler2axangle as _euler2axangle,
    euler2mat as _euler2mat,
    euler2quat as _euler2quat,
    mat2euler as _mat2euler,
    quat2euler as _quat2euler,
)


def euler2mat(z, y, x):
    return _euler2mat(z, y, x, "szyx")


def mat2euler(M, cy_thresh=None):
    if cy_thresh is None:
        return _mat2euler(M, "szyx")
    matrix = np.asarray(M)
    r11, r12, r13, r21, r22, r23, _, _, r33 = matrix.flat
    cy = math.sqrt(r23 * r23 + r33 * r33)
    if cy > cy_thresh:
        return (
            math.atan2(-r12, r11),
            math.atan2(r13, cy),
            math.atan2(-r23, r33),
        )
    return math.atan2(r21, r22), math.atan2(r13, cy), 0.0


def euler2quat(z, y, x):
    return _euler2quat(z, y, x, "szyx")


def quat2euler(q):
    return _quat2euler(q, "szyx")


def euler2axangle(z, y, x):
    return _euler2axangle(z, y, x, "szyx")


def axangle2euler(vector, theta):
    return _axangle2euler(vector, theta, "szyx")
