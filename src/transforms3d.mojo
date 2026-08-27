"""Rotation conversion kernels and their C ABI."""

from std.math import acos, atan2, cos, exp, log, sin, sqrt
comptime Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]


def p(addr: Int) -> Ptr:
    return Ptr(unsafe_from_address=addr)


def next_axis(index: Int) -> Int:
    if index == 0:
        return 1
    if index == 1:
        return 2
    if index == 2:
        return 0
    return 1


def euler2mat_one(
    ai_in: Float64,
    aj_in: Float64,
    ak_in: Float64,
    dst: Ptr,
    firstaxis: Int,
    parity: Int,
    repetition: Int,
    frame: Int,
):
    var ai = ai_in
    var aj = aj_in
    var ak = ak_in
    var i = firstaxis
    var j = next_axis(i + parity)
    var k = next_axis(i - parity + 1)
    if frame != 0:
        var swap = ai
        ai = ak
        ak = swap
    if parity != 0:
        ai = -ai
        aj = -aj
        ak = -ak
    var si = sin(ai)
    var sj = sin(aj)
    var sk = sin(ak)
    var ci = cos(ai)
    var cj = cos(aj)
    var ck = cos(ak)
    var cc = ci * ck
    var cs = ci * sk
    var sc = si * ck
    var ss = si * sk
    if repetition != 0:
        dst[i * 3 + i] = cj
        dst[i * 3 + j] = sj * si
        dst[i * 3 + k] = sj * ci
        dst[j * 3 + i] = sj * sk
        dst[j * 3 + j] = -cj * ss + cc
        dst[j * 3 + k] = -cj * cs - sc
        dst[k * 3 + i] = -sj * ck
        dst[k * 3 + j] = cj * sc + cs
        dst[k * 3 + k] = cj * cc - ss
    else:
        dst[i * 3 + i] = cj * ck
        dst[i * 3 + j] = sj * sc - cs
        dst[i * 3 + k] = sj * cc + ss
        dst[j * 3 + i] = cj * sk
        dst[j * 3 + j] = sj * ss + cc
        dst[j * 3 + k] = sj * cs - sc
        dst[k * 3 + i] = -sj
        dst[k * 3 + j] = cj * si
        dst[k * 3 + k] = cj * ci


def mat2euler_one(
    mat: Ptr,
    dst: Ptr,
    firstaxis: Int,
    parity: Int,
    repetition: Int,
    frame: Int,
):
    var i = firstaxis
    var j = next_axis(i + parity)
    var k = next_axis(i - parity + 1)
    var ax = 0.0
    var ay = 0.0
    var az = 0.0
    var eps4 = 8.881784197001252e-16
    if repetition != 0:
        var sy = sqrt(
            mat[i * 3 + j] * mat[i * 3 + j]
            + mat[i * 3 + k] * mat[i * 3 + k]
        )
        if sy > eps4:
            ax = atan2(mat[i * 3 + j], mat[i * 3 + k])
            ay = atan2(sy, mat[i * 3 + i])
            az = atan2(mat[j * 3 + i], -mat[k * 3 + i])
        else:
            ax = atan2(-mat[j * 3 + k], mat[j * 3 + j])
            ay = atan2(sy, mat[i * 3 + i])
    else:
        var cy = sqrt(
            mat[i * 3 + i] * mat[i * 3 + i]
            + mat[j * 3 + i] * mat[j * 3 + i]
        )
        if cy > eps4:
            ax = atan2(mat[k * 3 + j], mat[k * 3 + k])
            ay = atan2(-mat[k * 3 + i], cy)
            az = atan2(mat[j * 3 + i], mat[i * 3 + i])
        else:
            ax = atan2(-mat[j * 3 + k], mat[j * 3 + j])
            ay = atan2(-mat[k * 3 + i], cy)
    if parity != 0:
        ax = -ax
        ay = -ay
        az = -az
    if frame != 0:
        var swap = ax
        ax = az
        az = swap
    dst[0] = ax
    dst[1] = ay
    dst[2] = az


def euler2quat_one(
    ai_in: Float64,
    aj_in: Float64,
    ak_in: Float64,
    dst: Ptr,
    firstaxis: Int,
    parity: Int,
    repetition: Int,
    frame: Int,
):
    var ai = ai_in
    var aj = aj_in
    var ak = ak_in
    var i = firstaxis + 1
    var j = next_axis(i + parity - 1) + 1
    var k = next_axis(i - parity) + 1
    if frame != 0:
        var swap = ai
        ai = ak
        ak = swap
    if parity != 0:
        aj = -aj
    ai *= 0.5
    aj *= 0.5
    ak *= 0.5
    var ci = cos(ai)
    var si = sin(ai)
    var cj = cos(aj)
    var sj = sin(aj)
    var ck = cos(ak)
    var sk = sin(ak)
    var cc = ci * ck
    var cs = ci * sk
    var sc = si * ck
    var ss = si * sk
    for n in range(4):
        dst[n] = 0.0
    if repetition != 0:
        dst[0] = cj * (cc - ss)
        dst[i] = cj * (cs + sc)
        dst[j] = sj * (cc + ss)
        dst[k] = sj * (cs - sc)
    else:
        dst[0] = cj * cc + sj * ss
        dst[i] = cj * sc - sj * cs
        dst[j] = cj * ss + sj * cc
        dst[k] = cj * cs - sj * sc
    if parity != 0:
        dst[j] *= -1.0


def quat2mat_one(q: Ptr, dst: Ptr):
    var w = q[0]
    var x = q[1]
    var y = q[2]
    var z = q[3]
    var nq = w * w + x * x + y * y + z * z
    if nq < 2.220446049250313e-16:
        for i in range(9):
            dst[i] = 0.0
        dst[0] = 1.0
        dst[4] = 1.0
        dst[8] = 1.0
        return
    var scale = 2.0 / nq
    var xs = x * scale
    var ys = y * scale
    var zs = z * scale
    var wx = w * xs
    var wy = w * ys
    var wz = w * zs
    var xx = x * xs
    var xy = x * ys
    var xz = x * zs
    var yy = y * ys
    var yz = y * zs
    var zz = z * zs
    dst[0] = 1.0 - (yy + zz)
    dst[1] = xy - wz
    dst[2] = xz + wy
    dst[3] = xy + wz
    dst[4] = 1.0 - (xx + zz)
    dst[5] = yz - wx
    dst[6] = xz - wy
    dst[7] = yz + wx
    dst[8] = 1.0 - (xx + yy)


def axangle2mat_one(axis: Ptr, angle: Float64, dst: Ptr, normalized: Int):
    var x = axis[0]
    var y = axis[1]
    var z = axis[2]
    if normalized == 0:
        var norm = sqrt(x * x + y * y + z * z)
        x /= norm
        y /= norm
        z /= norm
    var c = cos(angle)
    var s = sin(angle)
    var one_c = 1.0 - c
    var xs = x * s
    var ys = y * s
    var zs = z * s
    var xc = x * one_c
    var yc = y * one_c
    var zc = z * one_c
    var xyc = x * yc
    var yzc = y * zc
    var zxc = z * xc
    dst[0] = x * xc + c
    dst[1] = xyc - zs
    dst[2] = zxc + ys
    dst[3] = xyc + zs
    dst[4] = y * yc + c
    dst[5] = yzc - xs
    dst[6] = zxc - ys
    dst[7] = yzc + xs
    dst[8] = z * zc + c


def axangle2quat_one(axis: Ptr, angle: Float64, dst: Ptr, normalized: Int):
    var x = axis[0]
    var y = axis[1]
    var z = axis[2]
    if normalized == 0:
        var norm = sqrt(x * x + y * y + z * z)
        x /= norm
        y /= norm
        z /= norm
    var half = angle * 0.5
    var sh = sin(half)
    dst[0] = cos(half)
    dst[1] = x * sh
    dst[2] = y * sh
    dst[3] = z * sh


def quat2axangle_one(q: Ptr, axis: Ptr, angle: Ptr, identity_thresh: Float64):
    var nq = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]
    axis[0] = 1.0
    axis[1] = 0.0
    axis[2] = 0.0
    angle[0] = 0.0
    if nq < 4.930380657631324e-32:
        return
    var norm = sqrt(nq)
    var w = q[0] / norm
    var x = q[1] / norm
    var y = q[2] / norm
    var z = q[3] / norm
    var len2 = x * x + y * y + z * z
    if len2 < identity_thresh * identity_thresh:
        return
    var vnorm = sqrt(len2)
    axis[0] = x / vnorm
    axis[1] = y / vnorm
    axis[2] = z / vnorm
    w = min(max(w, -1.0), 1.0)
    angle[0] = 2.0 * acos(w)


@export("mt3_euler2mat")
def mt3_euler2mat(
    ai: Int,
    aj: Int,
    ak: Int,
    dst: Int,
    n: Int,
    firstaxis: Int,
    parity: Int,
    repetition: Int,
    frame: Int,
) abi("C"):
    var aip = p(ai)
    var ajp = p(aj)
    var akp = p(ak)
    var dstp = p(dst)
    for row in range(n):
        euler2mat_one(
            aip[row], ajp[row], akp[row], dstp + row * 9,
            firstaxis, parity, repetition, frame,
        )


@export("mt3_euler2mat_scalar")
def mt3_euler2mat_scalar(
    ai: Float64,
    aj: Float64,
    ak: Float64,
    dst: Int,
    axes: Int,
) abi("C"):
    var firstaxis = axes & 3
    var parity = (axes >> 2) & 1
    var repetition = (axes >> 3) & 1
    var frame = (axes >> 4) & 1
    euler2mat_one(
        ai, aj, ak, p(dst), firstaxis, parity, repetition, frame
    )


@export("mt3_mat2euler")
def mt3_mat2euler(
    mat: Int,
    dst: Int,
    n: Int,
    firstaxis: Int,
    parity: Int,
    repetition: Int,
    frame: Int,
) abi("C"):
    var matp = p(mat)
    var dstp = p(dst)
    for row in range(n):
        mat2euler_one(
            matp + row * 9, dstp + row * 3,
            firstaxis, parity, repetition, frame,
        )


@export("mt3_euler2quat")
def mt3_euler2quat(
    ai: Int,
    aj: Int,
    ak: Int,
    dst: Int,
    n: Int,
    firstaxis: Int,
    parity: Int,
    repetition: Int,
    frame: Int,
) abi("C"):
    var aip = p(ai)
    var ajp = p(aj)
    var akp = p(ak)
    var dstp = p(dst)
    for row in range(n):
        euler2quat_one(
            aip[row], ajp[row], akp[row], dstp + row * 4,
            firstaxis, parity, repetition, frame,
        )


@export("mt3_quat2mat")
def mt3_quat2mat(q: Int, dst: Int, n: Int) abi("C"):
    var qp = p(q)
    var dstp = p(dst)
    for row in range(n):
        quat2mat_one(qp + row * 4, dstp + row * 9)


@export("mt3_axangle2mat")
def mt3_axangle2mat(
    axes: Int, angles: Int, dst: Int, n: Int, normalized: Int
) abi("C"):
    var axesp = p(axes)
    var anglesp = p(angles)
    var dstp = p(dst)
    for row in range(n):
        axangle2mat_one(
            axesp + row * 3, anglesp[row], dstp + row * 9, normalized
        )


@export("mt3_axangle2quat")
def mt3_axangle2quat(
    axes: Int, angles: Int, dst: Int, n: Int, normalized: Int
) abi("C"):
    var axesp = p(axes)
    var anglesp = p(angles)
    var dstp = p(dst)
    for row in range(n):
        axangle2quat_one(
            axesp + row * 3, anglesp[row], dstp + row * 4, normalized
        )


@export("mt3_quat2axangle")
def mt3_quat2axangle(
    q: Int, axes: Int, angles: Int, n: Int, identity_thresh: Float64
) abi("C"):
    var qp = p(q)
    var axesp = p(axes)
    var anglesp = p(angles)
    for row in range(n):
        quat2axangle_one(
            qp + row * 4, axesp + row * 3, anglesp + row, identity_thresh
        )


@export("mt3_qmult")
def mt3_qmult(q1: Int, q2: Int, dst: Int, n: Int) abi("C"):
    var ap = p(q1)
    var bp = p(q2)
    var dstp = p(dst)
    for row in range(n):
        var a = ap + row * 4
        var b = bp + row * 4
        var r = dstp + row * 4
        var w1 = a[0]
        var x1 = a[1]
        var y1 = a[2]
        var z1 = a[3]
        var w2 = b[0]
        var x2 = b[1]
        var y2 = b[2]
        var z2 = b[3]
        r[0] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        r[1] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        r[2] = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
        r[3] = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2


@export("mt3_qconjugate")
def mt3_qconjugate(q: Int, dst: Int, n: Int) abi("C"):
    var qp = p(q)
    var dstp = p(dst)
    for row in range(n):
        var offset = row * 4
        dstp[offset] = qp[offset]
        dstp[offset + 1] = -qp[offset + 1]
        dstp[offset + 2] = -qp[offset + 2]
        dstp[offset + 3] = -qp[offset + 3]


@export("mt3_qnorm")
def mt3_qnorm(q: Int, dst: Int, n: Int) abi("C"):
    var qp = p(q)
    var dstp = p(dst)
    for row in range(n):
        var offset = row * 4
        dstp[row] = sqrt(
            qp[offset] * qp[offset]
            + qp[offset + 1] * qp[offset + 1]
            + qp[offset + 2] * qp[offset + 2]
            + qp[offset + 3] * qp[offset + 3]
        )


@export("mt3_qinverse")
def mt3_qinverse(q: Int, dst: Int, n: Int) abi("C"):
    var qp = p(q)
    var dstp = p(dst)
    for row in range(n):
        var offset = row * 4
        var norm = sqrt(
            qp[offset] * qp[offset]
            + qp[offset + 1] * qp[offset + 1]
            + qp[offset + 2] * qp[offset + 2]
            + qp[offset + 3] * qp[offset + 3]
        )
        dstp[offset] = qp[offset] / norm
        dstp[offset + 1] = -qp[offset + 1] / norm
        dstp[offset + 2] = -qp[offset + 2] / norm
        dstp[offset + 3] = -qp[offset + 3] / norm


@export("mt3_qexp")
def mt3_qexp(q: Int, dst: Int, n: Int) abi("C"):
    var qp = p(q)
    var dstp = p(dst)
    for row in range(n):
        var offset = row * 4
        var vnorm = sqrt(
            qp[offset + 1] * qp[offset + 1]
            + qp[offset + 2] * qp[offset + 2]
            + qp[offset + 3] * qp[offset + 3]
        )
        if vnorm == 0.0:
            dstp[offset] = 1.0
            dstp[offset + 1] = 0.0
            dstp[offset + 2] = 0.0
            dstp[offset + 3] = 0.0
        else:
            var ew = exp(qp[offset])
            var scale = ew * sin(vnorm) / vnorm
            dstp[offset] = ew * cos(vnorm)
            dstp[offset + 1] = scale * qp[offset + 1]
            dstp[offset + 2] = scale * qp[offset + 2]
            dstp[offset + 3] = scale * qp[offset + 3]


@export("mt3_qlog")
def mt3_qlog(q: Int, dst: Int, n: Int) abi("C"):
    var qp = p(q)
    var dstp = p(dst)
    for row in range(n):
        var offset = row * 4
        var qnorm = sqrt(
            qp[offset] * qp[offset]
            + qp[offset + 1] * qp[offset + 1]
            + qp[offset + 2] * qp[offset + 2]
            + qp[offset + 3] * qp[offset + 3]
        )
        var vnorm = sqrt(
            qp[offset + 1] * qp[offset + 1]
            + qp[offset + 2] * qp[offset + 2]
            + qp[offset + 3] * qp[offset + 3]
        )
        if qnorm == 0.0 or vnorm == 0.0:
            dstp[offset] = 1.0
            dstp[offset + 1] = 0.0
            dstp[offset + 2] = 0.0
            dstp[offset + 3] = 0.0
        else:
            var angle = acos(min(max(qp[offset] / qnorm, -1.0), 1.0))
            var scale = angle / vnorm
            dstp[offset] = log(qnorm)
            dstp[offset + 1] = scale * qp[offset + 1]
            dstp[offset + 2] = scale * qp[offset + 2]
            dstp[offset + 3] = scale * qp[offset + 3]


@export("mt3_qpow")
def mt3_qpow(q: Int, dst: Int, count: Int, exponent: Float64) abi("C"):
    var qp = p(q)
    var dstp = p(dst)
    for row in range(count):
        var offset = row * 4
        var qnorm = sqrt(
            qp[offset] * qp[offset]
            + qp[offset + 1] * qp[offset + 1]
            + qp[offset + 2] * qp[offset + 2]
            + qp[offset + 3] * qp[offset + 3]
        )
        var vnorm = sqrt(
            qp[offset + 1] * qp[offset + 1]
            + qp[offset + 2] * qp[offset + 2]
            + qp[offset + 3] * qp[offset + 3]
        )
        if qnorm == 0.0 or vnorm == 0.0:
            dstp[offset] = 1.0
            dstp[offset + 1] = 0.0
            dstp[offset + 2] = 0.0
            dstp[offset + 3] = 0.0
        else:
            var theta = acos(min(max(qp[offset] / qnorm, -1.0), 1.0))
            var magnitude = exp(exponent * log(qnorm))
            var scale = magnitude * sin(exponent * theta) / vnorm
            dstp[offset] = magnitude * cos(exponent * theta)
            dstp[offset + 1] = scale * qp[offset + 1]
            dstp[offset + 2] = scale * qp[offset + 2]
            dstp[offset + 3] = scale * qp[offset + 3]


@export("mt3_rotate_vector")
def mt3_rotate_vector(
    vectors: Int, quats: Int, dst: Int, n: Int, normalized: Int
) abi("C"):
    var vp = p(vectors)
    var qp = p(quats)
    var dstp = p(dst)
    for row in range(n):
        var vo = row * 3
        var qo = row * 4
        var w = qp[qo]
        var x = qp[qo + 1]
        var y = qp[qo + 2]
        var z = qp[qo + 3]
        if normalized == 0:
            var norm = sqrt(w * w + x * x + y * y + z * z)
            w /= norm
            x /= norm
            y /= norm
            z /= norm
        # q v q* simplified to two cross products.
        var vx = vp[vo]
        var vy = vp[vo + 1]
        var vz = vp[vo + 2]
        var tx = 2.0 * (y * vz - z * vy)
        var ty = 2.0 * (z * vx - x * vz)
        var tz = 2.0 * (x * vy - y * vx)
        dstp[vo] = vx + w * tx + (y * tz - z * ty)
        dstp[vo + 1] = vy + w * ty + (z * tx - x * tz)
        dstp[vo + 2] = vz + w * tz + (x * ty - y * tx)
