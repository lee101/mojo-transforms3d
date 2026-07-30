# mojo-transforms3d

The rotation representation conversions from
[`transforms3d`](https://matthew-brett.github.io/transforms3d/), implemented in
Mojo and exposed to Python through a compatible NumPy API.

```python
from mojotransforms3d import euler, quaternions

matrix = euler.euler2mat(0.2, -0.4, 0.7, axes="sxyz")
quat = euler.euler2quat(0.2, -0.4, 0.7, axes="sxyz")
vector = quaternions.rotate_vector([1, 0, 0], quat)
print(matrix)
print(vector)
```

For the covered subset, function names, argument order, defaults, quaternion
layout (`w, x, y, z`), and return shapes follow transforms3d 0.4.2. Changing
`from transforms3d import euler` to
`from mojotransforms3d import euler` is enough for scalar code using these
functions. The additional functions ending in `_batch` operate on arbitrary
leading dimensions and are the high-throughput API.

## Coverage

| module | compatible API | batch extensions |
| --- | --- | --- |
| `euler` | `euler2mat`, `mat2euler`, `euler2quat`, `quat2euler`, `euler2axangle`, `axangle2euler`, conversion re-exports, `EulerFuncs`, `TBZYX`, `sxyz`, `rzxz`, `physics`, `szyx`; all 24 intrinsic/extrinsic axis conventions | `euler2mat_batch`, `mat2euler_batch`, `euler2quat_batch` |
| `quaternions` | `fillpositive`, `quat2mat`, `mat2quat`, `qmult`, `qconjugate`, `qnorm`, `qisunit`, `qinverse`, `qeye`, `qexp`, `qlog`, `qpow`, `rotate_vector`, `nearly_equivalent`, `axangle2quat`, `quat2axangle` | batch forms for every conversion and algebra kernel |
| `axangles` | `axangle2mat`, `mat2axangle`, `axangle2aff`, `aff2axangle` | `axangle2mat_batch` |
| `taitbryan` | `euler2mat`, `mat2euler`, `euler2quat`, `quat2euler`, `euler2axangle`, `axangle2euler` | use the general Euler batch API with `axes="szyx"` |

This intentionally does not port the unrelated `transforms3d.affines`,
`reflections`, `shears`, `zooms`, or utility modules. It also does not attempt
symbolic inputs. Numerical inputs are converted to contiguous `float64`
arrays at the Python boundary.

## Install and verify

The repository carries its Mojo compiler and Python dependencies in the pixi
environment:

```bash
pixi install
pixi run build
pixi run test
pixi run bench
```

`pixi run build` emits
`dist/libmojo-transforms3d.so`. The Python wrapper also rebuilds a missing or
stale library on first use when the Mojo compiler is available. A prebuilt
library can be selected with `MOJO_TRANSFORMS3D_LIB=/absolute/path/library.so`.

Batch use keeps the same mathematical conventions:

```python
import numpy as np
from mojotransforms3d import euler

angles = np.array([[0.1, 0.2, 0.3], [-0.4, 0.5, 0.6]])
matrices = euler.euler2mat_batch(
    angles[:, 0], angles[:, 1], angles[:, 2], axes="rzyx"
)
assert matrices.shape == (2, 3, 3)
```

## Performance

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux x86-64, using Mojo `1.0.0b3.dev2026072406`, transforms3d 0.4.2, and
NumPy 2.5.1:

| case | Mojo | transforms3d | speedup |
| --- | ---: | ---: | ---: |
| `euler2mat` scalar (20k calls) | 50.52 ms | 76.85 ms | 1.52x |
| `euler2mat_batch` (100k) | 13.77 ms | 646.07 ms | 46.93x |
| `quat2mat_batch` (100k) | 1.30 ms | 632.01 ms | 486.60x |
| `qmult_batch` (100k) | 0.63 ms | 580.52 ms | 926.57x |
| `rotate_vector_batch` (100k) | 2.81 ms | 1772.61 ms | 631.25x |

The batch rows compare one Mojo call with repeated calls to the real upstream
API on the same inputs. transforms3d has no batch API, so the large speedups
include eliminating its Python loop and should not be read as per-operation
arithmetic speedups. Timings fluctuate on this shared machine; the table is
the output of the final run, without extrapolation. There is no GPU path.

## How it works

The hot elementwise kernels live in one Mojo compilation unit. The shared
library exports a small C ABI with `@export` and `abi("C")`; Python loads it
with `ctypes`. NumPy owns every allocation. Before a call, wrappers broadcast
inputs and make contiguous `float64` buffers, then keep those arrays and their
output alive until the synchronous native call returns. Mojo receives their
addresses and row counts, so it never owns or frees Python memory. Scalar
`euler2mat` assembles its results in a SIMD value and uses a thread-local
scratch matrix; callers receive an independently owned copy.

Operations whose upstream definitions rely on NumPy linear algebra remain in
Python. In particular, `mat2quat` uses upstream's robust symmetric-eigensystem
algorithm so noisy matrices produce compatible closest rotations.

Matrices are C-contiguous row-major arrays with trailing shape `(3, 3)`.
Quaternions have trailing shape `(4,)` in scalar-first order, axes and vectors
have trailing shape `(3,)`, and batch dimensions precede those fixed trailing
dimensions. Each batch operation performs one FFI call and loops over rows in
Mojo. The test suite compares every covered scalar operation and the batch
extensions with transforms3d on identical randomized and edge-case inputs.

## License

MIT
