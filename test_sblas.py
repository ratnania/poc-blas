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
def test_drotg_1():
    from sblas import blas_drotg

    a = b = 1.
    c, s = blas_drotg (a, b)
    expected_c, expected_s = sp_blas.drotg (a, b)
    assert(np.abs(c - expected_c) < 1.e-10)
    assert(np.abs(s - expected_s) < 1.e-10)

# ==============================================================================
def test_drotmg_1():
    from sblas import blas_drotmg

    d1 = d2 = 1.
    x1 = y1 = .5
    result = np.zeros(5)
    blas_drotmg (d1, d2, x1, y1, result)
    expected = sp_blas.drotmg (d1, d2, x1, y1)
    assert(np.allclose(result, expected, 1.e-14))

# ==============================================================================
def test_drot_1():
    from sblas import blas_drot

    np.random.seed(2021)

    n = 10
    x = np.random.random(n)
    y = np.random.random(n)
    expected_x = x.copy()
    expected_y = y.copy()

    # ...
    c, s = sp_blas.drotg (1., 1.)
    expected_x, expected_y = sp_blas.drot(x, y, c, s)
    blas_drot (x, y, c, s)
    assert(np.allclose(x, expected_x, 1.e-14))
    assert(np.allclose(y, expected_y, 1.e-14))
    # ...

# ==============================================================================
def test_drotm_1():
    from sblas import blas_drotm

    np.random.seed(2021)

    n = 10
    x = np.random.random(n)
    y = np.random.random(n)
    expected_x = x.copy()
    expected_y = y.copy()

    # ...
    d1 = d2 = 1.
    x1 = y1 = .5
    param = sp_blas.drotmg (d1, d2, x1, y1)
    expected_x, expected_y = sp_blas.drotm(x, y, param)
    blas_drotm (x, y, param)
    assert(np.allclose(x, expected_x, 1.e-14))
    assert(np.allclose(y, expected_y, 1.e-14))
    # ...

# ==============================================================================
def test_dcopy_1():
    from sblas import blas_dcopy

    np.random.seed(2021)

    n = 10
    x = np.random.random(n)
    y = np.zeros(n)

    # ...
    expected  = np.zeros(n)
    sp_blas.dcopy(x, expected)
    blas_dcopy (x, y)
    assert(np.allclose(y, expected, 1.e-14))
    # ...

# ==============================================================================
def test_dswap_1():
    from sblas import blas_dswap

    np.random.seed(2021)

    n = 10
    x = np.random.random(n)
    y = 2* np.random.random(n) + 1.

    # ... we swap two times to get back to the original arrays
    expected_x = x.copy()
    expected_y = y.copy()
    sp_blas.dswap (x, y)
    blas_dswap (x, y)
    assert(np.allclose(x, expected_x, 1.e-14))
    assert(np.allclose(y, expected_y, 1.e-14))
    # ...

# ==============================================================================
def test_dscal_1():
    from sblas import blas_dscal

    np.random.seed(2021)

    n = 10
    x = np.random.random(n)

    # ... we scale two times to get back to the original arrays
    expected = x.copy()
    alpha = 2.5
    sp_blas.dscal (alpha, x)
    blas_dscal (1./alpha, x)
    assert(np.allclose(x, expected, 1.e-14))
    # ...

# ==============================================================================
def test_ddot_1():
    from sblas import blas_ddot

    np.random.seed(2021)

    n = 10
    x = np.random.random(n)
    y = np.random.random(n)

    # ...
    expected = sp_blas.ddot(x, y)
    result   = blas_ddot (x, y)
    assert(np.allclose(result, expected, 1.e-14))
    # ...

# ==============================================================================
def test_dnrm2_1():
    from sblas import blas_dnrm2

    np.random.seed(2021)

    n = 10
    x = np.random.random(n)

    # ...
    expected = sp_blas.dnrm2(x)
    result   = blas_dnrm2 (x)
    assert(np.allclose(result, expected, 1.e-14))
    # ...

# ==============================================================================
def test_dasum_1():
    from sblas import blas_dasum

    np.random.seed(2021)

    n = 10
    x = np.random.random(n)

    # ...
    expected = sp_blas.dasum(x)
    result   = blas_dasum (x)
    assert(np.allclose(result, expected, 1.e-14))
    # ...

# ==============================================================================
def test_idamax_1():
    from sblas import blas_idamax

    np.random.seed(2021)

    n = 10
    x = np.random.random(n)

    # ...
    expected = sp_blas.idamax(x)
    result   = blas_idamax (x)
    assert(result == expected)
    # ...

# ==============================================================================
def test_daxpy_1():
    from sblas import blas_daxpy

    np.random.seed(2021)

    n = 10
    x = np.random.random(n)
    y = np.random.random(n)

    # ...
    alpha = 2.5
    expected = y.copy()
    sp_blas.daxpy (x, expected, a=alpha)
    blas_daxpy (x, y, a=alpha )
    assert(np.allclose(y, expected, 1.e-14))
    # ...

# ==============================================================================
#
#                                  LEVEL 2
#
# ==============================================================================

# ==============================================================================
def test_dgemv_1():
    from sblas import blas_dgemv

    np.random.seed(2021)

    n = 10
    a = np.random.random((n,n)).copy(order='F')
    x = np.random.random(n)
    y = np.ones(n)

    # ...
    alpha = 1.
    beta = 0.5
    expected = y.copy()
    expected = sp_blas.dgemv (alpha, a, x, beta=beta, y=expected)
    blas_dgemv (alpha, a, x, y, beta=beta)
    assert(np.allclose(y, expected, 1.e-14))
    # ...

# ==============================================================================
def test_dgbmv_1():
    from sblas import blas_dgbmv

#    n = 8
#    a = diags([1, -2, 1], [-1, 0, 1], shape=(n, n)).toarray()
#    ku = kl = np.int32(2)

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

    # ...
    alpha = 2.
    beta = 0.5
    expected = y.copy()
    expected = sp_blas.dgbmv (n, n, kl, ku, alpha, ab, x, beta=beta, y=expected)
    blas_dgbmv (kl, ku, alpha, ab, x, y, beta=beta)
    assert(np.allclose(y, expected, 1.e-14))
    # ...

# ==============================================================================
def test_dsymv_1():
    from sblas import blas_dsymv

    np.random.seed(2021)

    n = 4
    a = np.random.random((n,n)).copy(order='F')
    x = np.random.random(n)
    y = np.random.random(n)

    # make a symmetric
    a = symmetrize(a)

    # ...
    alpha = 1.
    beta = 0.5
    expected = alpha * a @ x + beta * y

    # make a triangular
    a = triangulize(a)
    blas_dsymv (alpha, a, x, y, beta=beta)
    assert(np.allclose(y, expected, 1.e-14))
    # ...

# ==============================================================================
def test_dsbmv_1():
    from sblas import blas_dsbmv

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
    y = np.zeros(n)

    # ...
    alpha = 2.
    beta = 0.5
    expected = y.copy()
    expected = sp_blas.dsbmv (k, alpha, ab, x, beta=beta, y=expected)
    blas_dsbmv (k, alpha, ab, x, y, beta=beta)
    assert(np.allclose(y, expected, 1.e-14))
    # ...

# ==============================================================================
def test_dspmv_1():
    from sblas import blas_dspmv

    np.random.seed(2021)

    n = 4
    a = np.random.random((n,n)).copy(order='F')
    x = np.random.random(n)
    y = np.random.random(n)

    # make a symmetric
    a = symmetrize(a)
    ap = general_to_packed(a)

    # ...
    alpha = 1.
    beta = 0.5
    expected = alpha * a @ x + beta * y

    # make a triangular
    a = triangulize(a)
    blas_dspmv (alpha, ap, x, y, beta=beta)
    assert(np.allclose(y, expected, 1.e-14))
    # ...

# ==============================================================================
def test_dtrmv_1():
    from sblas import blas_dtrmv

    np.random.seed(2021)

    n = 10
    a = np.random.random((n,n)).copy(order='F')
    x = np.random.random(n)

    # make a triangular
    a = triangulize(a)

    # ...
    expected = a @ x
    blas_dtrmv (a, x)
    assert(np.allclose(x, expected, 1.e-14))
    # ...

# ==============================================================================
def test_dtbmv_1():
    from sblas import blas_dtbmv

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

    # make a triangular
    a = triangulize(a)

    # ...
    expected = a @ x
    blas_dtbmv (k, ab, x)
    assert(np.allclose(x, expected, 1.e-14))
    # ...

# ==============================================================================
def test_dtpmv_1():
    from sblas import blas_dtpmv

    np.random.seed(2021)

    n = 10
    a = np.random.random((n,n)).copy(order='F')
    x = np.random.random(n)

    # make a triangular
    a = triangulize(a)
    ap = general_to_packed(a)

    # ...
    expected = a @ x
    blas_dtpmv (ap, x)
    assert(np.allclose(x, expected, 1.e-14))
    # ...

# ==============================================================================
def test_dtrsv_1():
    from sblas import blas_dtrsv

    np.random.seed(2021)

    n = 10
    a = np.random.random((n,n)).copy(order='F')
    x = np.random.random(n)

    # make a triangular
    a = triangulize(a)

    # ...
    expected = x.copy()
    x = a @ x
    blas_dtrsv (a, x)
    assert(np.allclose(x, expected, 1.e-14))
    # ...

# ==============================================================================
def test_dtbsv_1():
    from sblas import blas_dtbsv

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

    # ...
    expected = sp_blas.dtbsv (k, ab, x)
    blas_dtbsv (k, ab, x)
    assert(np.allclose(x, expected, 1.e-14))
    # ...

# ==============================================================================
def test_dtpsv_1():
    from sblas import blas_dtpsv

    np.random.seed(2021)

    n = 10
    a = np.random.random((n,n)).copy(order='F')
    x = np.random.random(n)

    # make a triangular
    a = triangulize(a)
    ap = general_to_packed(a)

    # ...
    expected = x.copy()
    x = a @ x
    blas_dtpsv (ap, x)
    assert(np.allclose(x, expected, 1.e-14))
    # ...

# ==============================================================================
def test_dger_1():
    from sblas import blas_dger

    np.random.seed(2021)

    n = 10
    a = np.random.random((n,n))

    x = np.ones(n)
    y = np.zeros(n)

    # ...
    alpha = 1.
    expected = alpha * np.outer(x,y) + a
    blas_dger (alpha, x, y, a)
    assert(np.allclose(a, expected, 1.e-14))
    # ...

# ==============================================================================
def test_dsyr_1():
    from sblas import blas_dsyr

    np.random.seed(2021)

    n = 4
    a = np.random.random((n,n)).copy(order='F')
    x = np.random.random(n)

    # syrketrize a
    a = symmetrize(a)

    # ...
    alpha = 1.
    expected = alpha * np.outer(x.T, x) + a
    blas_dsyr (alpha, x, a, lower=False)
    a = symmetrize(a, lower=False)
    assert(np.allclose(a, expected, 1.e-14))
    # ...

# ==============================================================================
def test_dspr_1():
    from sblas import blas_dspr

    np.random.seed(2021)

    n = 10
    a = np.random.random((n,n)).copy(order='F')
    x = np.random.random(n)

    # syrketrize a
    a = symmetrize(a)
    ap = general_to_packed(a)

    # ...
    alpha = 1.
    expected = alpha * np.outer(x.T, x) + a
    expected = general_to_packed(expected)
    blas_dspr (alpha, x, ap, lower=False)
    assert(np.allclose(ap, expected, 1.e-14))
    # ...

# ==============================================================================
def test_dsyr2_1():
    from sblas import blas_dsyr2

    np.random.seed(2021)

    n = 4
    a = np.random.random((n,n)).copy(order='F')
    x = np.random.random(n)
    y = np.random.random(n)

    # syrketrize a
    a = symmetrize(a)

    # ...
    alpha = 1.
    expected = alpha * np.outer(y.T, x) + alpha * np.outer(x.T, y) + a
    blas_dsyr2 (alpha, x, y, a, lower=False)
    a = symmetrize(a, lower=False)
    assert(np.allclose(a, expected, 1.e-14))
    # ...

# ==============================================================================
def test_dspr2_1():
    from sblas import blas_dspr2

    np.random.seed(2021)

    n = 4
    a = np.random.random((n,n)).copy(order='F')
    x = np.random.random(n)
    y = np.random.random(n)

    # syrketrize a
    a = symmetrize(a)
    ap = general_to_packed(a)

    # ...
    alpha = 1.
    expected = alpha * np.outer(y.T, x) + alpha * np.outer(x.T, y) + a
    expected = general_to_packed(expected)
    blas_dspr2 (alpha, x, y, ap, lower=False)
    assert(np.allclose(ap, expected, 1.e-14))
    # ...

# ==============================================================================
#
#                                  LEVEL 3
#
# ==============================================================================

# ==============================================================================
def test_dgemm_1():
    from sblas import blas_dgemm

    np.random.seed(2021)

    n = 4
    a = np.random.random((n,n)).copy(order='F')
    b = np.random.random((n,n)).copy(order='F')
    c = np.zeros((n,n), order='F')

    # ...
    alpha = 1.
    beta = 0.
    expected = alpha * a @ b + beta * c
    blas_dgemm (alpha, a, b, c, beta=beta)
    assert(np.allclose(c, expected, 1.e-14))
    # ...

# ==============================================================================
def test_dgemm_2():
    from sblas import blas_dgemm

    np.random.seed(2021)

    n = 4
    a = np.random.random((n,n)).copy(order='F')
    b = np.random.random((n,n)).copy(order='F')
    c = np.random.random((n,n)).copy(order='F')

    # ...
    alpha = 2.
    beta = 1.
    expected = alpha * a @ b + beta * c
    blas_dgemm (alpha, a, b, c, beta=beta)
    assert(np.allclose(c, expected, 1.e-14))
    # ...

# ==============================================================================
def test_dsymm_1():
    from sblas import blas_dsymm

    np.random.seed(2021)

    n = 4
    a = np.random.random((n,n)).copy(order='F')
    b = np.random.random((n,n)).copy(order='F')
    c = np.zeros((n,n), order='F')

    # symmetrize a & b
    a = symmetrize(a)
    b = symmetrize(b)

    # ...
    alpha = 1.
    beta = 0.
    expected = alpha * a @ b + beta * c
    blas_dsymm (alpha, a, b, c, beta=beta)
    assert(np.allclose(c, expected, 1.e-14))
    # ...

# ==============================================================================
def test_dtrmm_1():
    from sblas import blas_dtrmm

    np.random.seed(2021)

    n = 4
    a = np.random.random((n,n)).copy(order='F')
    b = np.random.random((n,n)).copy(order='F')

    # make a triangular
    a = triangulize(a)

    # ...
    alpha = 1.
    expected = alpha * a @ b
    blas_dtrmm (alpha, a, b)
    assert(np.allclose(b, expected, 1.e-14))
    # ...

# ==============================================================================
def test_dtrsm_1():
    from sblas import blas_dtrsm

    np.random.seed(2021)

    n = 4
    a = np.random.random((n,n)).copy(order='F')
    b = np.random.random((n,n)).copy(order='F')

    # make a triangular
    a = triangulize(a)

    # ...
    alpha = 1.
    expected = b.copy()
    b = alpha * a @ b
    b = b.copy(order='F')
    blas_dtrsm (alpha, a, b)
    assert(np.allclose(b, expected, 1.e-14))
    # ...

# ==============================================================================
def test_dsyrk_1():
    from sblas import blas_dsyrk

    np.random.seed(2021)

    n = 4
    a = np.random.random((n,n)).copy(order='F')
    c = np.zeros((n,n), order='F')

    # syrketrize a
    a = symmetrize(a)
    a_T = a.T.copy(order='F')

    # ...
    alpha = 1.
    beta = 0.
    expected = alpha * a @ a_T + beta * c
    blas_dsyrk (alpha, a, c, beta=beta)
    # we need to symmetrize the matrix
    c = symmetrize(c)
    assert(np.allclose(c, expected, 1.e-14))
    # ...

# ==============================================================================
def test_dsyr2k_1():
    from sblas import blas_dsyr2k

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

    # ...
    alpha = 1.
    beta = 0.
    expected = alpha * a @ b_T + alpha * b @ a_T + beta * c
    blas_dsyr2k (alpha, a, b, c, beta=beta)
    # we need to symmetrize the matrix
    c = symmetrize(c)
    assert(np.allclose(c, expected, 1.e-14))
    # ...

# ******************************************************************************
if __name__ == '__main__':

    # ... LEVEL 1
    test_drotg_1()
    test_drotmg_1()
    test_drot_1()
    test_drotm_1()
    test_dcopy_1()
    test_dswap_1()
    test_dscal_1()
    test_ddot_1()
    test_dnrm2_1()
    test_dasum_1()
    test_idamax_1()
    test_daxpy_1()
    # ...

    # ... LEVEL 2
    test_dgemv_1()
    test_dgbmv_1()
    test_dsymv_1()
    test_dsbmv_1()
    test_dspmv_1()
    test_dtrmv_1()
    test_dtbmv_1()
    test_dtpmv_1()
    test_dtrsv_1()
    test_dtbsv_1()
    test_dtpsv_1()
    test_dger_1()
    test_dsyr_1()
    test_dspr_1()
    test_dsyr2_1()
    test_dspr2_1()
    # ...

    # ... LEVEL 3
    test_dgemm_1()
    test_dgemm_2()
    test_dsymm_1()
    test_dtrmm_1()
    test_dtrsm_1()
    test_dsyrk_1()
    test_dsyr2k_1()
    # ...
