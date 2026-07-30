import numpy as np
import pytest

from transforms3d import axangles as reference

from mojotransforms3d import axangles

RNG = np.random.default_rng(90210)


def test_axangle2mat_random_scalar_parity():
    for axis, angle in zip(RNG.normal(size=(100, 3)), RNG.uniform(-8, 8, 100)):
        np.testing.assert_allclose(
            axangles.axangle2mat(axis, angle),
            reference.axangle2mat(axis, angle),
            atol=2e-15,
        )


def test_axangle2mat_batch_and_normalized_paths():
    axes = RNG.normal(size=(4, 7, 3))
    axes /= np.linalg.norm(axes, axis=-1, keepdims=True)
    angles = RNG.uniform(-np.pi, np.pi, size=(4, 1))
    got = axangles.axangle2mat_batch(axes, angles, is_normalized=True)
    expected = np.array(
        [
            reference.axangle2mat(axis, angle, is_normalized=True)
            for axis, angle in zip(
                np.broadcast_to(axes, (4, 7, 3)).reshape(-1, 3),
                np.broadcast_to(angles, (4, 7)).ravel(),
            )
        ]
    ).reshape(4, 7, 3, 3)
    np.testing.assert_allclose(got, expected, atol=2e-15)


def test_published_principal_axis_rotations():
    np.testing.assert_allclose(
        axangles.axangle2mat([1, 0, 0], np.pi),
        np.diag([1, -1, -1]),
        atol=2e-15,
    )
    np.testing.assert_allclose(
        axangles.axangle2mat([0, 0, 1], np.pi / 2),
        [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
        atol=2e-15,
    )


def test_matrix_axis_angle_pair_matches_upstream_and_reconstructs():
    for axis, angle in zip(RNG.normal(size=(50, 3)), RNG.uniform(-3, 3, 50)):
        matrix = reference.axangle2mat(axis, angle)
        got_axis, got_angle = axangles.mat2axangle(matrix)
        ref_axis, ref_angle = reference.mat2axangle(matrix)
        np.testing.assert_allclose(got_axis, ref_axis, atol=2e-12)
        assert got_angle == pytest.approx(ref_angle, abs=2e-12)
        np.testing.assert_allclose(
            axangles.axangle2mat(got_axis, got_angle), matrix, atol=2e-12
        )


def test_affine_rotation_origin_and_arbitrary_point_parity():
    axis = np.array([0.2, -0.5, 0.7])
    point = np.array([2.0, 3.0, -1.0])
    angle = 1.2
    np.testing.assert_allclose(
        axangles.axangle2aff(axis, angle),
        reference.axangle2aff(axis, angle),
    )
    affine = axangles.axangle2aff(axis, angle, point)
    np.testing.assert_allclose(affine, reference.axangle2aff(axis, angle, point))
    np.testing.assert_allclose(affine @ np.r_[point, 1], np.r_[point, 1])


def test_affine_to_axis_angle_reconstructs_upstream_transform():
    axis = np.array([0.2, 0.8, -0.1])
    point = np.array([-4.0, 2.0, 5.0])
    affine = reference.axangle2aff(axis, -0.9, point)
    got_axis, got_angle, got_point = axangles.aff2axangle(affine)
    np.testing.assert_allclose(
        axangles.axangle2aff(got_axis, got_angle, got_point),
        affine,
        atol=2e-12,
    )
    assert got_point.shape == (4,)


def test_mat2axangle_rejects_non_rotation_without_unit_eigenvector():
    with pytest.raises(ValueError, match="no unit eigenvector"):
        axangles.mat2axangle(np.diag([2.0, 3.0, 4.0]))
