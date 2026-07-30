import numpy as np
import pytest

from transforms3d import quaternions as reference

from mojotransforms3d import quaternions as quat
from mojotransforms3d._lib import addr

RNG = np.random.default_rng(1337)


def test_ffi_address_rejects_wrong_dtype_and_strides():
    with pytest.raises(TypeError, match="C-contiguous float64"):
        addr(np.ones(4, dtype=np.float32))
    with pytest.raises(TypeError, match="C-contiguous float64"):
        addr(np.ones(8, dtype=np.float64)[::2])


@pytest.mark.parametrize(
    "value",
    (
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [2.0, -3.0, 4.0, 1.0],
        [0.0, 0.0, 0.0, 0.0],
    ),
)
def test_quat2mat_published_and_edge_vectors(value):
    np.testing.assert_allclose(quat.quat2mat(value), reference.quat2mat(value))


def test_quat2mat_and_mat2quat_random_parity():
    quats = RNG.normal(size=(200, 4))
    matrices = quat.quat2mat_batch(quats)
    expected = np.array([reference.quat2mat(q) for q in quats])
    np.testing.assert_allclose(matrices, expected, atol=2e-15)
    recovered = quat.mat2quat_batch(matrices)
    reference_recovered = np.array([reference.mat2quat(m) for m in matrices])
    for got, expected_q, matrix in zip(recovered, reference_recovered, matrices):
        assert quat.nearly_equivalent(got, expected_q, atol=2e-14)
        np.testing.assert_allclose(quat.quat2mat(got), matrix, atol=2e-14)


def test_mat2quat_handles_nearly_pi_and_slight_matrix_noise():
    source = reference.axangle2quat([0.2, -0.7, 0.4], np.pi - 1e-10)
    matrix = reference.quat2mat(source)
    matrix += RNG.normal(scale=1e-10, size=(3, 3))
    got = quat.mat2quat(matrix)
    expected = reference.mat2quat(matrix)
    assert quat.nearly_equivalent(got, expected, atol=2e-9)


def test_hamilton_product_scalar_and_broadcast_batch():
    q1 = RNG.normal(size=(8, 1, 4))
    q2 = RNG.normal(size=(3, 4))
    got = quat.qmult_batch(q1, q2)
    expected = np.empty((8, 3, 4))
    for i in range(8):
        for j in range(3):
            expected[i, j] = reference.qmult(q1[i, 0], q2[j])
    np.testing.assert_allclose(got, expected, atol=2e-15)
    np.testing.assert_allclose(
        quat.qmult(q1[0, 0], q2[0]), reference.qmult(q1[0, 0], q2[0])
    )


def test_mat2quat_matches_upstream_for_materially_noisy_batch():
    matrices = []
    for _ in range(20):
        left, _, right = np.linalg.svd(RNG.normal(size=(3, 3)))
        rotation = left @ right
        if np.linalg.det(rotation) < 0:
            left[:, -1] *= -1
            rotation = left @ right
        matrices.append(rotation + 0.1 * RNG.normal(size=(3, 3)))
    matrices = np.asarray(matrices)
    got = quat.mat2quat_batch(matrices)
    expected = np.asarray([reference.mat2quat(matrix) for matrix in matrices])
    np.testing.assert_allclose(got, expected, atol=2e-15)


def test_all_unary_batch_extensions_and_strided_inputs():
    source = RNG.normal(size=(12, 8))[:, ::2]
    expected = {
        "qconjugate_batch": [reference.qconjugate(row) for row in source],
        "qinverse_batch": [reference.qinverse(row) for row in source],
        "qexp_batch": [reference.qexp(row) for row in source],
        "qlog_batch": [reference.qlog(row) for row in source],
    }
    for name, rows in expected.items():
        np.testing.assert_allclose(getattr(quat, name)(source), rows, atol=2e-15)
    np.testing.assert_allclose(
        quat.qnorm_batch(source),
        [reference.qnorm(row) for row in source],
        atol=2e-15,
    )
    np.testing.assert_allclose(
        quat.qpow_batch(source, 0.3),
        [reference.qpow(row, 0.3) for row in source],
        atol=2e-15,
    )
@pytest.mark.parametrize(
    "name",
    ("qconjugate", "qinverse", "qexp", "qlog"),
)
def test_quaternion_unary_algebra_matches_upstream(name):
    for value in RNG.normal(size=(30, 4)):
        got = getattr(quat, name)(value)
        expected = getattr(reference, name)(value)
        np.testing.assert_allclose(got, expected, atol=2e-15)


def test_quaternion_norm_unit_identity_and_power():
    for value in RNG.normal(size=(30, 4)):
        assert quat.qnorm(value) == pytest.approx(reference.qnorm(value))
        np.testing.assert_allclose(
            quat.qpow(value, -0.75), reference.qpow(value, -0.75), atol=3e-15
        )
    assert quat.qisunit([1, 0, 0, 0])
    assert not quat.qisunit([2, 0, 0, 0])
    assert np.array_equal(quat.qeye(np.float32), reference.qeye(np.float32))


def test_upstream_zero_quaternion_algebra_behavior():
    zero = np.zeros(4)
    np.testing.assert_array_equal(quat.qexp(zero), reference.qexp(zero))
    np.testing.assert_array_equal(quat.qlog(zero), reference.qlog(zero))
    np.testing.assert_array_equal(quat.qpow(zero, 2), reference.qpow(zero, 2))


def test_axis_angle_quaternion_scalar_batch_and_identity():
    axes = RNG.normal(size=(100, 3))
    angles = RNG.uniform(-4, 4, size=100)
    got = quat.axangle2quat_batch(axes, angles)
    expected = np.array(
        [reference.axangle2quat(axis, angle) for axis, angle in zip(axes, angles)]
    )
    np.testing.assert_allclose(got, expected, atol=2e-15)
    for q in got:
        got_axis, got_angle = quat.quat2axangle(q)
        ref_axis, ref_angle = reference.quat2axangle(q)
        np.testing.assert_allclose(got_axis, ref_axis, atol=2e-15)
        assert got_angle == pytest.approx(ref_angle, abs=2e-15)
    axis, angle = quat.quat2axangle([1, 0, 0, 0])
    np.testing.assert_array_equal(axis, [1, 0, 0])
    assert angle == 0.0


def test_quat2axangle_nonfinite_behavior():
    axis, angle = quat.quat2axangle([1, np.inf, 0, 0])
    np.testing.assert_array_equal(axis, [1, 0, 0])
    assert np.isnan(angle)


def test_fillpositive_matches_errors_threshold_and_values():
    np.testing.assert_array_equal(quat.fillpositive([0, 0, 0]), [1, 0, 0, 0])
    np.testing.assert_array_equal(quat.fillpositive([1, 0, 0]), [0, 1, 0, 0])
    with pytest.raises(ValueError, match="length 3"):
        quat.fillpositive([0, 0])
    with pytest.raises(ValueError, match="w2 should be positive"):
        quat.fillpositive([2, 0, 0])


def test_rotate_vector_scalar_and_broadcast_batch():
    vectors = RNG.normal(size=(4, 1, 3))
    quats = RNG.normal(size=(5, 4))
    got = quat.rotate_vector_batch(vectors, quats, is_normalized=False)
    expected = np.empty((4, 5, 3))
    for i in range(4):
        for j in range(5):
            expected[i, j] = reference.rotate_vector(
                vectors[i, 0], quats[j], is_normalized=False
            )
    np.testing.assert_allclose(got, expected, atol=2e-15)
    np.testing.assert_allclose(
        quat.rotate_vector(vectors[0, 0], quats[0], False),
        reference.rotate_vector(vectors[0, 0], quats[0], False),
        atol=2e-15,
    )


def test_nearly_equivalent_includes_quaternion_sign_ambiguity():
    assert quat.nearly_equivalent([1, 0, 0, 0], [-1, 0, 0, 0])
    assert not quat.nearly_equivalent([1, 0, 0, 0], [0, 1, 0, 0])
