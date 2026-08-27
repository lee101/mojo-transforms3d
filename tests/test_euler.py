from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from transforms3d import euler as reference
from transforms3d import taitbryan as reference_tb

from mojotransforms3d import euler, taitbryan

AXES = tuple(reference._AXES2TUPLE)
RNG = np.random.default_rng(814)


@pytest.mark.parametrize("axes", AXES)
def test_euler_to_matrix_and_quaternion_all_conventions(axes):
    angles = RNG.uniform(-2 * np.pi, 2 * np.pi, size=(40, 3))
    for ai, aj, ak in angles:
        np.testing.assert_allclose(
            euler.euler2mat(ai, aj, ak, axes),
            reference.euler2mat(ai, aj, ak, axes),
            atol=2e-15,
        )
        np.testing.assert_allclose(
            euler.euler2quat(ai, aj, ak, axes),
            reference.euler2quat(ai, aj, ak, axes),
            atol=2e-15,
        )


@pytest.mark.parametrize("axes", AXES)
def test_matrix_to_euler_all_conventions_including_singularities(axes):
    spec = reference._AXES2TUPLE[axes]
    middle = (0.0, np.pi / 2, -np.pi / 2)
    if spec[2]:
        middle = (0.0, np.pi, -np.pi)
    angles = np.vstack(
        [
            RNG.uniform(-np.pi, np.pi, size=(40, 3)),
            np.column_stack(
                [
                    RNG.uniform(-np.pi, np.pi, 3),
                    middle,
                    RNG.uniform(-np.pi, np.pi, 3),
                ]
            ),
        ]
    )
    for values in angles:
        matrix = reference.euler2mat(*values, axes)
        got = euler.mat2euler(matrix, axes)
        expected = reference.mat2euler(matrix, axes)
        np.testing.assert_allclose(got, expected, atol=2e-15)
        np.testing.assert_allclose(euler.euler2mat(*got, axes), matrix, atol=2e-15)


def test_axis_tuple_and_case_handling_match_upstream():
    angles = (0.3, -1.2, 2.1)
    spec = (2, 1, 0, 1)
    np.testing.assert_allclose(
        euler.euler2mat(*angles, spec), reference.euler2mat(*angles, spec)
    )
    matrix = reference.euler2mat(*angles, "rxyz")
    np.testing.assert_allclose(
        euler.mat2euler(matrix, "RXYZ"), reference.mat2euler(matrix, "RXYZ")
    )
    with pytest.raises(KeyError):
        euler.euler2mat(*angles, "bad")


def test_euler_batch_conversions_match_upstream_loops():
    angles = RNG.normal(size=(7, 5, 3))
    matrices = euler.euler2mat_batch(
        angles[..., 0], angles[..., 1], angles[..., 2], "ryxy"
    )
    quats = euler.euler2quat_batch(
        angles[..., 0], angles[..., 1], angles[..., 2], "ryxy"
    )
    expected_m = np.array(
        [reference.euler2mat(*row, "ryxy") for row in angles.reshape(-1, 3)]
    ).reshape(7, 5, 3, 3)
    expected_q = np.array(
        [reference.euler2quat(*row, "ryxy") for row in angles.reshape(-1, 3)]
    ).reshape(7, 5, 4)
    np.testing.assert_allclose(matrices, expected_m, atol=2e-15)
    np.testing.assert_allclose(quats, expected_q, atol=2e-15)
    np.testing.assert_allclose(
        euler.mat2euler_batch(matrices, "ryxy"),
        np.array(
            [reference.mat2euler(row, "ryxy") for row in expected_m.reshape(-1, 3, 3)]
        ).reshape(7, 5, 3),
        atol=2e-15,
    )


@pytest.mark.parametrize("axes", AXES)
def test_euler2mat_scalar_fast_path_and_simd_tail_match_batch(axes):
    values = RNG.normal(size=3)
    np.testing.assert_array_equal(
        euler.euler2mat(*values, axes),
        euler.euler2mat_batch(*values, axes),
    )


def test_euler2mat_default_simd_path_owns_each_result():
    first = euler.euler2mat(0.2, -0.4, 0.7)
    second = euler.euler2mat(0.2, -0.4, 0.7)
    np.testing.assert_array_equal(first, second)
    first[0, 0] = 123.0
    assert second[0, 0] != 123.0


def test_euler2mat_scalar_buffers_are_owned_and_thread_local():
    expected = euler.euler2mat(0.2, -0.4, 0.7, "rzyz")

    def convert(_):
        return euler.euler2mat(0.2, -0.4, 0.7, "rzyz")

    with ThreadPoolExecutor(max_workers=4) as pool:
        matrices = list(pool.map(convert, range(64)))
    for matrix in matrices:
        np.testing.assert_array_equal(matrix, expected)
    matrices[0][0, 0] = 123.0
    assert matrices[1][0, 0] != 123.0


def test_matrix_input_may_be_affine():
    matrix = np.eye(4)
    matrix[:3, :3] = reference.euler2mat(0.2, 0.4, -0.7)
    assert euler.mat2euler(matrix) == pytest.approx(reference.mat2euler(matrix))


def test_euler_axis_angle_compositions_match_upstream():
    values = (0.7, -0.3, 1.8)
    axis, angle = euler.euler2axangle(*values, "rzxz")
    ref_axis, ref_angle = reference.euler2axangle(*values, "rzxz")
    np.testing.assert_allclose(axis, ref_axis, atol=2e-15)
    assert angle == pytest.approx(ref_angle, abs=2e-15)
    assert euler.axangle2euler(axis, angle, "rzxz") == pytest.approx(
        reference.axangle2euler(ref_axis, ref_angle, "rzxz"), abs=2e-15
    )


def test_euler_namespaces_bind_the_convention():
    values = (0.1, 0.2, 0.3)
    np.testing.assert_allclose(
        euler.physics.euler2mat(*values),
        reference.physics.euler2mat(*values),
    )
    np.testing.assert_allclose(
        euler.szyx.euler2quat(*values),
        reference.szyx.euler2quat(*values),
    )


def test_taitbryan_full_api_matches_specialized_upstream():
    values = (1.3, -0.1, 0.2)
    matrix = taitbryan.euler2mat(*values)
    np.testing.assert_allclose(matrix, reference_tb.euler2mat(*values))
    np.testing.assert_allclose(
        taitbryan.euler2quat(*values), reference_tb.euler2quat(*values)
    )
    assert taitbryan.mat2euler(matrix) == pytest.approx(
        reference_tb.mat2euler(matrix)
    )
    assert taitbryan.mat2euler(matrix, cy_thresh=2.0) == pytest.approx(
        reference_tb.mat2euler(matrix, cy_thresh=2.0)
    )
    axis, angle = taitbryan.euler2axangle(*values)
    ref_axis, ref_angle = reference_tb.euler2axangle(*values)
    np.testing.assert_allclose(axis, ref_axis)
    assert angle == pytest.approx(ref_angle)
    assert taitbryan.axangle2euler(axis, angle) == pytest.approx(
        reference_tb.axangle2euler(ref_axis, ref_angle)
    )
    assert taitbryan.quat2euler(taitbryan.euler2quat(*values)) == pytest.approx(
        reference_tb.quat2euler(reference_tb.euler2quat(*values))
    )
