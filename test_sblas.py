import numpy as np
from scipy.sparse import diags
import scipy.linalg.blas as sp_blas

#dsdot

# ==============================================================================
def symmetrize(a, lower=False):
    n = a.shape[0]
    if lower:
        for j in range(n):
            for i in range(j):
                a[i,j] = a[j,i]
    else:
        for i in range(n):
            for j in range(i):
                a[i,j] = a[j,i]

    return a

# ==============================================================================
def triangulize(a, lower=False):
    n = a.shape[0]
    if lower:
        for j in range(n):
            for i in range(j):
                a[i,j] = 0.
    else:
        for i in range(n):
            for j in range(i):
                a[i,j] = 0.

    return a

# ==============================================================================
def general_to_band(kl, ku, a):
    n = a.shape[1]
    ab = np.zeros((kl+ku+1, n))

    for j in range(n):
        k = ku - j
        i1 = max (j - ku, 0)
        i2 = min (j + kl + 1, n)
        for i in range(i1, i2):
            ab[k+i,j] = a[i,j]

    return ab

# ==============================================================================
def general_to_packed(a, lower=False):
    n = a.shape[0]
    ap = np.zeros(n*(n+1)//2)
    if lower:
        k = 0
        for j in range(n):
            for i in range(j,n):
                ap[k] = a[i,j]
                k += 1
    else:
        k = 0
        for j in range(n):
            for i in range(j+1):
                ap[k] = a[i,j]
                k += 1

    return ap

# ==============================================================================
#
#                                  LEVEL 1
#
# ==============================================================================

# ==============================================================================
def test_srotg_1():
    from sblas import blas_srotg

    a = b = np.float32(1.)
    c, s = blas_srotg (a, b)
    expected_c, expected_s = sp_blas.srotg (a, b)
    assert(np.abs(c - expected_c) < 1.e-10)
    assert(np.abs(s - expected_s) < 1.e-10)

# ==============================================================================
def test_srotmg_1():
    from sblas import blas_srotmg

    d1 = d2 = np.float32(1.)
    x1 = y1 = np.float32(.5)
    result = np.zeros(5, dtype=np.float32)
    blas_srotmg (d1, d2, x1, y1, result)
    expected = sp_blas.srotmg (d1, d2, x1, y1)
    assert(np.allclose(result, expected, 1.e-7))

# ==============================================================================
def test_srot_1():
    from sblas import blas_srot

    np.random.seed(2021)

    n = 10
    x = np.random.random(n)
    y = np.random.random(n)
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    expected_x = x.copy()
    expected_y = y.copy()

    # ...
    one = np.float32(1.)
    c, s = sp_blas.srotg (one, one)
    c = np.float32(c)
    s = np.float32(s)
    expected_x, expected_y = sp_blas.srot(x, y, c, s)
    blas_srot (x, y, c, s)
    assert(np.allclose(x, expected_x, 1.e-7))
    assert(np.allclose(y, expected_y, 1.e-7))
    # ...

# ==============================================================================
def test_srotm_1():
    from sblas import blas_srotm

    np.random.seed(2021)

    n = 10
    x = np.random.random(n)
    y = np.random.random(n)
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    expected_x = x.copy()
    expected_y = y.copy()

    # ...
    d1 = d2 = np.float32(1.)
    x1 = y1 = np.float32(.5)
    param = sp_blas.srotmg (d1, d2, x1, y1)
    param = np.array(param, dtype=np.float32)
    expected_x, expected_y = sp_blas.srotm(x, y, param)
    blas_srotm (x, y, param)
    assert(np.allclose(x, expected_x, 1.e-7))
    assert(np.allclose(y, expected_y, 1.e-7))
    # ...

# ==============================================================================
def test_scopy_1():
    from sblas import blas_scopy

    np.random.seed(2021)

    n = 10
    x = np.random.random(n)
    x = np.array(x, dtype=np.float32)
    y = np.zeros(n, dtype=np.float32)

    # ...
    expected  = np.zeros(n, dtype=np.float32)
    sp_blas.scopy(x, expected)
    blas_scopy (x, y)
    assert(np.allclose(y, expected, 1.e-7))
    # ...

# ==============================================================================
def test_sswap_1():
    from sblas import blas_sswap

    np.random.seed(2021)

    n = 10
    x = np.random.random(n)
    x = np.array(x, dtype=np.float32)
    y = 2* np.random.random(n) + 1.
    y = np.array(y, dtype=np.float32)

    # ... we swap two times to get back to the original arrays
    expected_x = x.copy()
    expected_y = y.copy()
    sp_blas.sswap (x, y)
    blas_sswap (x, y)
    assert(np.allclose(x, expected_x, 1.e-7))
    assert(np.allclose(y, expected_y, 1.e-7))
    # ...

# ==============================================================================
def test_sscal_1():
    from sblas import blas_sscal

    np.random.seed(2021)

    n = 10
    x = np.random.random(n)
    x = np.array(x, dtype=np.float32)

    # ... we scale two times to get back to the original arrays
    expected = x.copy()
    alpha = np.float32(2.5)
    sp_blas.sscal (alpha, x)
    blas_sscal (np.float32(1./alpha), x)
    assert(np.allclose(x, expected, 1.e-7))
    # ...

# ==============================================================================
def test_sdot_1():
    from sblas import blas_sdot

    np.random.seed(2021)

    n = 10
    x = np.random.random(n)
    y = np.random.random(n)
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # ...
    expected = sp_blas.sdot(x, y)
    result   = blas_sdot (x, y)
    assert(np.allclose(result, expected, 1.e-6))
    # ...

# ==============================================================================
def test_snrm2_1():
    from sblas import blas_snrm2

    np.random.seed(2021)

    n = 10
    x = np.random.random(n)
    x = np.array(x, dtype=np.float32)

    # ...
    expected = sp_blas.snrm2(x)
    result   = blas_snrm2 (x)
    assert(np.allclose(result, expected, 1.e-6))
    # ...

# ==============================================================================
def test_sasum_1():
    from sblas import blas_sasum

    np.random.seed(2021)

    n = 10
    x = np.random.random(n)
    x = np.array(x, dtype=np.float32)

    # ...
    expected = sp_blas.sasum(x)
    result   = blas_sasum (x)
    assert(np.allclose(result, expected, 1.e-7))
    # ...

# ==============================================================================
def test_isamax_1():
    from sblas import blas_isamax

    np.random.seed(2021)

    n = 10
    x = np.random.random(n)
    x = np.array(x, dtype=np.float32)

    # ...
    expected = sp_blas.isamax(x)
    result   = blas_isamax (x)
    assert(result == expected)
    # ...

# ==============================================================================
def test_saxpy_1():
    from sblas import blas_saxpy

    np.random.seed(2021)

    n = 10
    x = np.random.random(n)
    y = np.random.random(n)
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # ...
    alpha = np.float32(2.5)
    expected = y.copy()
    sp_blas.saxpy (x, expected, a=alpha)
    blas_saxpy (x, y, a=alpha )
    assert(np.allclose(y, expected, 1.e-7))
    # ...

# ==============================================================================
#
#                                  LEVEL 2
#
# ==============================================================================

# ==============================================================================
def test_sgemv_1():
    from sblas import blas_sgemv

    np.random.seed(2021)

    n = 10
    a = np.random.random((n,n)).copy(order='F')
    x = np.random.random(n)
    y = np.ones(n)

    a = np.array(a, dtype=np.float32)
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # ...
    alpha = np.float32(1.)
    beta = np.float32(0.5)
    expected = y.copy()
    expected = sp_blas.sgemv (alpha, a, x, beta=beta, y=expected)
    blas_sgemv (alpha, a, x, y, beta=beta)
    assert(np.allclose(y, expected, 1.e-7))
    # ...

# ==============================================================================
def test_sgbmv_1():
    from sblas import blas_sgbmv

    n = 5
    kl = np.int32(2)
    ku = np.int32(2)
    a = np.array([[11, 12,  0,  0,  0],
                  [21, 22, 23,  0,  0],
                  [31, 32, 33, 34,  0],
                  [ 0, 42, 43, 44, 45],
                  [ 0,  0, 53, 54, 55]
                 ], dtype=np.float64)

    ab = general_to_band(kl, ku, a).copy(order='F')

    np.random.seed(2021)

    x = np.random.random(n)
    y = np.random.random(n)

    ab = np.array(ab, dtype=np.float32)
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # ...
    alpha = np.float32(1.)
    beta = np.float32(0.5)
    expected = y.copy()
    expected = sp_blas.sgbmv (n, n, kl, ku, alpha, ab, x, beta=beta, y=expected)
    blas_sgbmv (kl, ku, alpha, ab, x, y, beta=beta)
    assert(np.allclose(y, expected, 1.e-7))
    # ...

# ==============================================================================
def test_ssymv_1():
    from sblas import blas_ssymv

    np.random.seed(2021)

    n = 4
    a = np.random.random((n,n)).copy(order='F')
    x = np.random.random(n)
    y = np.random.random(n)

    a = np.array(a, dtype=np.float32)
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # make a symmetric
    a = symmetrize(a)

    # ...
    alpha = np.float32(1.)
    beta = np.float32(0.5)
    expected = alpha * a @ x + beta * y

    # make a triangular
    a = triangulize(a)
    blas_ssymv (alpha, a, x, y, beta=beta)
    assert(np.allclose(y, expected, 1.e-6))
    # ...

# ==============================================================================
def test_ssbmv_1():
    from sblas import blas_ssbmv

    n = 5
    k = np.int32(2)
    a = np.array([[11, 12,  0,  0,  0],
                  [12, 22, 23,  0,  0],
                  [ 0, 32, 33, 34,  0],
                  [ 0,  0, 34, 44, 45],
                  [ 0,  0,  0, 45, 55]
                 ], dtype=np.float64)

    ab = general_to_band(k, k, a).copy(order='F')

    np.random.seed(2021)

    x = np.random.random(n)
    y = np.zeros(n, dtype=np.float32)

    ab = np.array(ab, dtype=np.float32)
    x = np.array(x, dtype=np.float32)

    # ...
    alpha = np.float32(1.)
    beta = np.float32(0.5)
    expected = y.copy()
    expected = sp_blas.ssbmv (k, alpha, ab, x, beta=beta, y=expected)
    blas_ssbmv (k, alpha, ab, x, y, beta=beta)
    assert(np.allclose(y, expected, 1.e-7))
    # ...

# ==============================================================================
def test_sspmv_1():
    from sblas import blas_sspmv

    np.random.seed(2021)

    n = 4
    a = np.random.random((n,n)).copy(order='F')
    x = np.random.random(n)
    y = np.random.random(n)

    # make a symmetric
    a = symmetrize(a)
    ap = general_to_packed(a)

    a = np.array(a, dtype=np.float32)
    ap = np.array(ap, dtype=np.float32)
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # ...
    alpha = np.float32(1.)
    beta = np.float32(0.5)
    expected = alpha * a @ x + beta * y

    # make a triangular
    a = triangulize(a)
    blas_sspmv (alpha, ap, x, y, beta=beta)
    assert(np.allclose(y, expected, 1.e-6))
    # ...

# ==============================================================================
def test_strmv_1():
    from sblas import blas_strmv

    np.random.seed(2021)

    n = 10
    a = np.random.random((n,n)).copy(order='F')
    x = np.random.random(n)

    a = np.array(a, dtype=np.float32)
    x = np.array(x, dtype=np.float32)

    # make a triangular
    a = triangulize(a)

    # ...
    expected = a @ x
    blas_strmv (a, x)
    assert(np.allclose(x, expected, 1.e-6))
    # ...

# ==============================================================================
def test_stbmv_1():
    from sblas import blas_stbmv

    n = 5
    k = np.int32(2)
    a = np.array([[11, 12,  0,  0,  0],
                  [12, 22, 23,  0,  0],
                  [ 0, 32, 33, 34,  0],
                  [ 0,  0, 34, 44, 45],
                  [ 0,  0,  0, 45, 55]
                 ], dtype=np.float64)

    ab = general_to_band(k, k, a).copy(order='F')

    np.random.seed(2021)

    x = np.random.random(n)

    ab = np.array(ab, dtype=np.float32)
    a = np.array(a, dtype=np.float32)
    x = np.array(x, dtype=np.float32)

    # make a triangular
    a = triangulize(a)

    # ...
    expected = a @ x
    blas_stbmv (k, ab, x)
    assert(np.allclose(x, expected, 1.e-6))
    # ...

# ==============================================================================
def test_stpmv_1():
    from sblas import blas_stpmv

    np.random.seed(2021)

    n = 10
    a = np.random.random((n,n)).copy(order='F')
    x = np.random.random(n)

    # make a triangular
    a = triangulize(a)
    ap = general_to_packed(a)

    ap = np.array(ap, dtype=np.float32)
    a = np.array(a, dtype=np.float32)
    x = np.array(x, dtype=np.float32)

    # ...
    expected = a @ x
    blas_stpmv (ap, x)
    assert(np.allclose(x, expected, 1.e-6))
    # ...

# ==============================================================================
def test_strsv_1():
    from sblas import blas_strsv

    np.random.seed(2021)

    n = 10
    a = np.random.random((n,n)).copy(order='F')
    x = np.random.random(n)

    # make a triangular
    a = triangulize(a)

    a = np.array(a, dtype=np.float32)
    x = np.array(x, dtype=np.float32)

    # ...
    expected = x.copy()
    x = a @ x
    blas_strsv (a, x)
    assert(np.allclose(x, expected, 1.e-5))
    # ...

# ==============================================================================
def test_stbsv_1():
    from sblas import blas_stbsv

    n = 5
    k = np.int32(2)
    a = np.array([[11, 12,  0,  0,  0],
                  [12, 22, 23,  0,  0],
                  [ 0, 32, 33, 34,  0],
                  [ 0,  0, 34, 44, 45],
                  [ 0,  0,  0, 45, 55]
                 ], dtype=np.float64)

    ab = general_to_band(k, k, a).copy(order='F')

    np.random.seed(2021)

    x = np.random.random(n)

    ab = np.array(ab, dtype=np.float32)
    x = np.array(x, dtype=np.float32)

    # ...
    expected = sp_blas.stbsv (k, ab, x)
    blas_stbsv (k, ab, x)
    assert(np.allclose(x, expected, 1.e-7))
    # ...

# ==============================================================================
def test_stpsv_1():
    from sblas import blas_stpsv

    np.random.seed(2021)

    n = 10
    a = np.random.random((n,n)).copy(order='F')
    x = np.random.random(n)

    # make a triangular
    a = triangulize(a)
    ap = general_to_packed(a)

    ap = np.array(ap, dtype=np.float32)
    a = np.array(a, dtype=np.float32)
    x = np.array(x, dtype=np.float32)

    # ...
    expected = x.copy()
    x = a @ x
    blas_stpsv (ap, x)
    assert(np.allclose(x, expected, 1.e-5))
    # ...

# ==============================================================================
def test_sger_1():
    from sblas import blas_sger

    np.random.seed(2021)

    n = 10
    a = np.random.random((n,n)).copy(order='F')

    x = np.ones(n)
    y = np.zeros(n)

    a = np.array(a, dtype=np.float32)
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # ...
    alpha = np.float32(1.)
    expected = alpha * np.outer(x,y) + a
    blas_sger (alpha, x, y, a)
    assert(np.allclose(a, expected, 1.e-7))
    # ...

# ==============================================================================
def test_ssyr_1():
    from sblas import blas_ssyr

    np.random.seed(2021)

    n = 4
    a = np.random.random((n,n)).copy(order='F')
    x = np.random.random(n)

    # syrketrize a
    a = symmetrize(a)

    a = np.array(a, dtype=np.float32)
    x = np.array(x, dtype=np.float32)

    # ...
    alpha = np.float32(1.)
    expected = alpha * np.outer(x.T, x) + a
    blas_ssyr (alpha, x, a, lower=False)
    a = symmetrize(a, lower=False)
    assert(np.allclose(a, expected, 1.e-7))
    # ...

# ==============================================================================
def test_sspr_1():
    from sblas import blas_sspr

    np.random.seed(2021)

    n = 10
    a = np.random.random((n,n)).copy(order='F')
    x = np.random.random(n)

    # syrketrize a
    a = symmetrize(a)
    ap = general_to_packed(a)

    ap = np.array(ap, dtype=np.float32)
    a = np.array(a, dtype=np.float32)
    x = np.array(x, dtype=np.float32)

    # ...
    alpha = np.float32(1.)
    expected = alpha * np.outer(x.T, x) + a
    expected = general_to_packed(expected)
    blas_sspr (alpha, x, ap, lower=False)
    assert(np.allclose(ap, expected, 1.e-7))
    # ...

# ==============================================================================
def test_ssyr2_1():
    from sblas import blas_ssyr2

    np.random.seed(2021)

    n = 4
    a = np.random.random((n,n)).copy(order='F')
    x = np.random.random(n)
    y = np.random.random(n)

    # syrketrize a
    a = symmetrize(a)

    a = np.array(a, dtype=np.float32)
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # ...
    alpha = np.float32(1.)
    expected = alpha * np.outer(y.T, x) + alpha * np.outer(x.T, y) + a
    blas_ssyr2 (alpha, x, y, a, lower=False)
    a = symmetrize(a, lower=False)
    assert(np.allclose(a, expected, 1.e-7))
    # ...

# ==============================================================================
def test_sspr2_1():
    from sblas import blas_sspr2

    np.random.seed(2021)

    n = 4
    a = np.random.random((n,n)).copy(order='F')
    x = np.random.random(n)
    y = np.random.random(n)

    # syrketrize a
    a = symmetrize(a)
    ap = general_to_packed(a)

    ap = np.array(ap, dtype=np.float32)
    a = np.array(a, dtype=np.float32)
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # ...
    alpha = np.float32(1.)
    expected = alpha * np.outer(y.T, x) + alpha * np.outer(x.T, y) + a
    expected = general_to_packed(expected)
    blas_sspr2 (alpha, x, y, ap, lower=False)
    assert(np.allclose(ap, expected, 1.e-7))
    # ...

# ==============================================================================
#
#                                  LEVEL 3
#
# ==============================================================================

# ==============================================================================
def test_sgemm_1():
    from sblas import blas_sgemm

    np.random.seed(2021)

    n = 4
    a = np.random.random((n,n)).copy(order='F')
    b = np.random.random((n,n)).copy(order='F')
    c = np.zeros((n,n), order='F')

    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)

    # ...
    alpha = np.float32(1.)
    beta = np.float32(0.)
    expected = alpha * a @ b + beta * c
    blas_sgemm (alpha, a, b, c, beta=beta)
    assert(np.allclose(c, expected, 1.e-7))
    # ...

# ==============================================================================
def test_sgemm_2():
    from sblas import blas_sgemm

    np.random.seed(2021)

    n = 4
    a = np.random.random((n,n)).copy(order='F')
    b = np.random.random((n,n)).copy(order='F')
    c = np.random.random((n,n)).copy(order='F')

    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)

    # ...
    alpha = np.float32(2.)
    beta = np.float32(1.)
    expected = alpha * a @ b + beta * c
    blas_sgemm (alpha, a, b, c, beta=beta)
    assert(np.allclose(c, expected, 1.e-7))
    # ...

# ==============================================================================
def test_ssymm_1():
    from sblas import blas_ssymm

    np.random.seed(2021)

    n = 4
    a = np.random.random((n,n)).copy(order='F')
    b = np.random.random((n,n)).copy(order='F')
    c = np.zeros((n,n), order='F')

    # symmetrize a & b
    a = symmetrize(a)
    b = symmetrize(b)

    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)

    # ...
    alpha = np.float32(1.)
    beta = np.float32(0.)
    expected = alpha * a @ b + beta * c
    blas_ssymm (alpha, a, b, c, beta=beta)
    assert(np.allclose(c, expected, 1.e-7))
    # ...

# ==============================================================================
def test_strmm_1():
    from sblas import blas_strmm

    np.random.seed(2021)

    n = 4
    a = np.random.random((n,n)).copy(order='F')
    b = np.random.random((n,n)).copy(order='F')

    # make a triangular
    a = triangulize(a)

    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)

    # ...
    alpha = np.float32(1.)
    expected = alpha * a @ b
    blas_strmm (alpha, a, b)
    assert(np.allclose(b, expected, 1.e-7))
    # ...

# ==============================================================================
def test_strsm_1():
    from sblas import blas_strsm

    np.random.seed(2021)

    n = 4
    a = np.random.random((n,n)).copy(order='F')
    b = np.random.random((n,n)).copy(order='F')

    # make a triangular
    a = triangulize(a)

    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)

    # ...
    alpha = np.float32(1.)
    expected = b.copy()
    b = alpha * a @ b
    b = b.copy(order='F')
    blas_strsm (alpha, a, b)
    assert(np.allclose(b, expected, 1.e-6))
    # ...

# ==============================================================================
def test_ssyrk_1():
    from sblas import blas_ssyrk

    np.random.seed(2021)

    n = 4
    a = np.random.random((n,n)).copy(order='F')
    c = np.zeros((n,n), order='F')

    # syrketrize a
    a = symmetrize(a)
    a_T = a.T.copy(order='F')

    a = np.array(a, dtype=np.float32)
    c = np.array(c, dtype=np.float32)

    # ...
    alpha = np.float32(1.)
    beta = np.float32(0.)
    expected = alpha * a @ a_T + beta * c
    blas_ssyrk (alpha, a, c, beta=beta)
    # we need to symmetrize the matrix
    c = symmetrize(c)
    assert(np.allclose(c, expected, 1.e-7))
    # ...

# ==============================================================================
def test_ssyr2k_1():
    from sblas import blas_ssyr2k

    np.random.seed(2021)

    n = 4
    a = np.random.random((n,n)).copy(order='F')
    b = np.random.random((n,n)).copy(order='F')
    c = np.zeros((n,n), order='F')

    # syr2ketrize a & b
    a = symmetrize(a)
    b = symmetrize(b)
    a_T = a.T.copy(order='F')
    b_T = b.T.copy(order='F')

    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)

    # ...
    alpha = np.float32(1.)
    beta = np.float32(0.)
    expected = alpha * a @ b_T + alpha * b @ a_T + beta * c
    blas_ssyr2k (alpha, a, b, c, beta=beta)
    # we need to symmetrize the matrix
    c = symmetrize(c)
    assert(np.allclose(c, expected, 1.e-7))
    # ...

# ******************************************************************************
if __name__ == '__main__':

    # ... LEVEL 1
    test_srotg_1()
    test_srotmg_1()
    test_srot_1()
    test_srotm_1()
    test_scopy_1()
    test_sswap_1()
    test_sscal_1()
    test_sdot_1()
    test_snrm2_1()
    test_sasum_1()
    test_isamax_1()
    test_saxpy_1()
    # ...

    # ... LEVEL 2
    test_sgemv_1()
    test_sgbmv_1()
    test_ssymv_1()
    test_ssbmv_1()
    test_sspmv_1()
    test_strmv_1()
    test_stbmv_1()
    test_stpmv_1()
    test_strsv_1()
    test_stbsv_1()
    test_stpsv_1()
    test_sger_1()
    test_ssyr_1()
    test_sspr_1()
    test_ssyr2_1()
    test_sspr2_1()
    # ...

    # ... LEVEL 3
    test_sgemm_1()
    test_sgemm_2()
    test_ssymm_1()
    test_strmm_1()
    test_strsm_1()
    test_ssyrk_1()
    test_ssyr2k_1()
    # ...
