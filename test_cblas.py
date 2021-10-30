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
    ab = np.zeros((kl+ku+1, n), dtype=a.dtype)

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
    ap = np.zeros(n*(n+1)//2, dtype=a.dtype)
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
def test_ccopy_1():
    from cblas import blas_ccopy

    np.random.seed(20)

    n = 3
    x = np.random.random(n) + np.random.random(n) * 1j
    y = np.zeros(n) + 0. * 1j

    x = np.array(x, dtype=np.complex64)
    y = np.array(y, dtype=np.complex64)

    # ...
    expected = y.copy()
    sp_blas.ccopy(x, expected)
    blas_ccopy (x, y)
    assert(np.allclose(y, expected, 1.e-7))
    # ...

# ==============================================================================
def test_cswap_1():
    from cblas import blas_cswap

    np.random.seed(2021)

    n = 10
    x = np.random.random(n) + np.random.random(n) * 1j
    y = 2* np.random.random(n) + 1.*1j

    x = np.array(x, dtype=np.complex64)
    y = np.array(y, dtype=np.complex64)

    # ... we swap two times to get back to the original arrays
    expected_x = x.copy()
    expected_y = y.copy()
    sp_blas.cswap (x, y)
    blas_cswap (x, y)
    assert(np.allclose(x, expected_x, 1.e-7))
    assert(np.allclose(y, expected_y, 1.e-7))
    # ...

# ==============================================================================
def test_cscal_1():
    from cblas import blas_cscal

    np.random.seed(2021)

    n = 10
    x = np.random.random(n) + np.random.random(n) * 1j
    x = np.array(x, dtype=np.complex64)

    # ... we scale two times to get back to the original arrays
    expected = x.copy()
    alpha = np.complex64(2.5)
    inv_alpha = np.complex64(1./alpha)
    sp_blas.cscal (alpha, x)
    blas_cscal (inv_alpha, x)
    assert(np.allclose(x, expected, 1.e-7))
    # ...

# ==============================================================================
def test_scnrm2_1():
    from cblas import blas_scnrm2

    np.random.seed(2021)

    n = 10
    x = np.random.random(n) + np.random.random(n) * 1j
    x = np.array(x, dtype=np.complex64)

    # ...
    expected = sp_blas.scnrm2(x)
    result   = blas_scnrm2 (x)
    assert(np.allclose(result, expected, 1.e-6))
    # ...

# ==============================================================================
# TODO not working
def test_scasum_1():
    from cblas import blas_scasum

    np.random.seed(2021)

    n = 10
    x = np.random.random(n) + np.random.random(n) * 1j
    x = np.array(x, dtype=np.complex64)

    # ...
    expected = sp_blas.scasum(x)
    result   = blas_scasum (x)
#    assert(np.allclose(result, expected, 1.e-6))
    # ...

# ==============================================================================
def test_icamax_1():
    from cblas import blas_icamax

    np.random.seed(2021)

    n = 10
    x = np.random.random(n) + np.random.random(n) * 1j
    x = np.array(x, dtype=np.complex64)

    # ...
    expected = sp_blas.icamax(x)
    result   = blas_icamax (x)
    assert(result == expected)
    # ...

# ==============================================================================
def test_caxpy_1():
    from cblas import blas_caxpy

    np.random.seed(2021)

    n = 10
    x = np.random.random(n) + np.random.random(n) * 1j
    y = np.random.random(n) + np.random.random(n) * 1j
    x = np.array(x, dtype=np.complex64)
    y = np.array(y, dtype=np.complex64)

    # ...
    alpha = np.complex64(2.5)
    expected = y.copy()
    sp_blas.caxpy (x, expected, a=alpha)
    blas_caxpy (x, y, a=alpha )
    assert(np.allclose(y, expected, 1.e-7))
    # ...

# ==============================================================================
#
#                                  LEVEL 2
#
# ==============================================================================

# ==============================================================================
def test_cgemv_1():
    from cblas import blas_cgemv

    np.random.seed(2021)

    n = 10
    a = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    a = a.copy(order='F')
    x = np.random.random(n) + np.random.random(n) * 1j
    y = np.random.random(n) + np.random.random(n) * 1j

    a = np.array(a, dtype=np.complex64)
    x = np.array(x, dtype=np.complex64)
    y = np.array(y, dtype=np.complex64)

    # ...
    alpha = np.complex64(1.0)
    beta = np.complex64(0.5)
    expected = y.copy()
    expected = sp_blas.cgemv (alpha, a, x, beta=beta, y=expected)
    blas_cgemv (alpha, a, x, y, beta=beta)
    assert(np.allclose(y, expected, 1.e-7))
    # ...

# ==============================================================================
def test_cgbmv_1():
    from cblas import blas_cgbmv

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

    x = np.random.random(n) + np.random.random(n) * 1j
    y = np.random.random(n) + np.random.random(n) * 1j

    ab = np.array(ab, dtype=np.complex64)
    x = np.array(x, dtype=np.complex64)
    y = np.array(y, dtype=np.complex64)

    # ...
    alpha = np.complex64(1.0)
    beta = np.complex64(0.5)
    expected = y.copy()
    expected = sp_blas.cgbmv (n, n, kl, ku, alpha, ab, x, beta=beta, y=expected)
    blas_cgbmv (kl, ku, alpha, ab, x, y, beta=beta)
    assert(np.allclose(y, expected, 1.e-7))
    # ...

# ==============================================================================
def test_ctrmv_1():
    from cblas import blas_ctrmv

    np.random.seed(2021)

    n = 10
    a = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    a = a.copy(order='F')
    x = np.random.random(n) + np.random.random(n) * 1j

    # make a triangular
    a = triangulize(a)

    a = np.array(a, dtype=np.complex64)
    x = np.array(x, dtype=np.complex64)

    # ...
    expected = a @ x
    blas_ctrmv (a, x)
    assert(np.allclose(x, expected, 1.e-6))
    # ...

# ==============================================================================
def test_ctbmv_1():
    from cblas import blas_ctbmv

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

    x = np.random.random(n) + np.random.random(n) * 1j

    # make a triangular
    a = triangulize(a)

    ab = np.array(ab, dtype=np.complex64)
    a = np.array(a, dtype=np.complex64)
    x = np.array(x, dtype=np.complex64)

    # ...
    expected = a @ x
    blas_ctbmv (k, ab, x)
    assert(np.allclose(x, expected, 1.e-7))
    # ...

# ==============================================================================
def test_ctpmv_1():
    from cblas import blas_ctpmv

    np.random.seed(2021)

    n = 10
    a = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    a = a.copy(order='F')
    x = np.random.random(n) + np.random.random(n) * 1j

    # make a triangular
    a = triangulize(a)
    ap = general_to_packed(a)

    ap = np.array(ap, dtype=np.complex64)
    a = np.array(a, dtype=np.complex64)
    x = np.array(x, dtype=np.complex64)

    # ...
    expected = a @ x
    blas_ctpmv (ap, x)
    assert(np.allclose(x, expected, 1.e-6))
    # ...

# ==============================================================================
def test_ctrsv_1():
    from cblas import blas_ctrsv

    np.random.seed(2021)

    n = 10
    a = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    a = a.copy(order='F')
    x = np.random.random(n) + np.random.random(n) * 1j

    # make a triangular
    a = triangulize(a)

    a = np.array(a, dtype=np.complex64)
    x = np.array(x, dtype=np.complex64)

    # ...
    expected = x.copy()
    x = a @ x
    blas_ctrsv (a, x)
    assert(np.linalg.norm(x-expected) < 1.e-6)
    # ...

# ==============================================================================
def test_ctbsv_1():
    from cblas import blas_ctbsv

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

    x = np.random.random(n) + np.random.random(n) * 1j

    ab = np.array(ab, dtype=np.complex64)
    x = np.array(x, dtype=np.complex64)

    # ...
    expected = sp_blas.ctbsv (k, ab, x)
    blas_ctbsv (k, ab, x)
    assert(np.allclose(x, expected, 1.e-7))
    # ...

# ==============================================================================
def test_ctpsv_1():
    from cblas import blas_ctpsv

    np.random.seed(2021)

    n = 10
    a = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    a = a.copy(order='F')
    x = np.random.random(n) + np.random.random(n) * 1j

    # make a triangular
    a = triangulize(a)
    ap = general_to_packed(a)

    ap = np.array(ap, dtype=np.complex64)
    a = np.array(a, dtype=np.complex64)
    x = np.array(x, dtype=np.complex64)

    # ...
    expected = x.copy()
    x = a @ x
    blas_ctpsv (ap, x)
    assert(np.linalg.norm(x-expected) < 1.e-6)
    # ...

## ==============================================================================
#def test_cger_1():
#    from cblas import blas_cger
#
#    np.random.seed(2021)
#
#    n = 10
#    a = np.random.random((n,n))
#
#    x = np.ones(n)
#    y = np.zeros(n)
#
#    # ...
#    alpha = 1.
#    expected = alpha * np.outer(x,y) + a
#    blas_cger (alpha, x, y, a)
#    assert(np.allclose(a, expected, 1.e-7))
#    # ...
#
## ==============================================================================
#def test_csyr_1():
#    from cblas import blas_csyr
#
#    np.random.seed(2021)
#
#    n = 4
#    a = np.random.random((n,n)).copy(order='F')
#    x = np.random.random(n)
#
#    # syrketrize a
#    a = symmetrize(a)
#
#    # ...
#    alpha = 1.
#    expected = alpha * np.outer(x.T, x) + a
#    blas_csyr (alpha, x, a, lower=False)
#    a = symmetrize(a, lower=False)
#    assert(np.allclose(a, expected, 1.e-7))
#    # ...
#
## ==============================================================================
#def test_cspr_1():
#    from cblas import blas_cspr
#
#    np.random.seed(2021)
#
#    n = 10
#    a = np.random.random((n,n)).copy(order='F')
#    x = np.random.random(n)
#
#    # syrketrize a
#    a = symmetrize(a)
#    ap = general_to_packed(a)
#
#    # ...
#    alpha = 1.
#    expected = alpha * np.outer(x.T, x) + a
#    expected = general_to_packed(expected)
#    blas_cspr (alpha, x, ap, lower=False)
#    assert(np.allclose(ap, expected, 1.e-7))
#    # ...
#
## ==============================================================================
#def test_csyr2_1():
#    from cblas import blas_csyr2
#
#    np.random.seed(2021)
#
#    n = 4
#    a = np.random.random((n,n)).copy(order='F')
#    x = np.random.random(n)
#    y = np.random.random(n)
#
#    # syrketrize a
#    a = symmetrize(a)
#
#    # ...
#    alpha = 1.
#    expected = alpha * np.outer(y.T, x) + alpha * np.outer(x.T, y) + a
#    blas_csyr2 (alpha, x, y, a, lower=False)
#    a = symmetrize(a, lower=False)
#    assert(np.allclose(a, expected, 1.e-7))
#    # ...
#
## ==============================================================================
#def test_cspr2_1():
#    from cblas import blas_cspr2
#
#    np.random.seed(2021)
#
#    n = 4
#    a = np.random.random((n,n)).copy(order='F')
#    x = np.random.random(n)
#    y = np.random.random(n)
#
#    # syrketrize a
#    a = symmetrize(a)
#    ap = general_to_packed(a)
#
#    # ...
#    alpha = 1.
#    expected = alpha * np.outer(y.T, x) + alpha * np.outer(x.T, y) + a
#    expected = general_to_packed(expected)
#    blas_cspr2 (alpha, x, y, ap, lower=False)
#    assert(np.allclose(ap, expected, 1.e-7))
#    # ...
#
## ==============================================================================
##
##                                  LEVEL 3
##
## ==============================================================================
#
## ==============================================================================
#def test_cgemm_1():
#    from cblas import blas_cgemm
#
#    np.random.seed(2021)
#
#    n = 4
#    a = np.random.random((n,n)).copy(order='F')
#    b = np.random.random((n,n)).copy(order='F')
#    c = np.zeros((n,n), order='F')
#
#    # ...
#    alpha = 1.
#    beta = 0.
#    expected = alpha * a @ b + beta * c
#    blas_cgemm (alpha, a, b, c, beta=beta)
#    assert(np.allclose(c, expected, 1.e-7))
#    # ...
#
## ==============================================================================
#def test_cgemm_2():
#    from cblas import blas_cgemm
#
#    np.random.seed(2021)
#
#    n = 4
#    a = np.random.random((n,n)).copy(order='F')
#    b = np.random.random((n,n)).copy(order='F')
#    c = np.random.random((n,n)).copy(order='F')
#
#    # ...
#    alpha = 2.
#    beta = 1.
#    expected = alpha * a @ b + beta * c
#    blas_cgemm (alpha, a, b, c, beta=beta)
#    assert(np.allclose(c, expected, 1.e-7))
#    # ...
#
## ==============================================================================
#def test_csymm_1():
#    from cblas import blas_csymm
#
#    np.random.seed(2021)
#
#    n = 4
#    a = np.random.random((n,n)).copy(order='F')
#    b = np.random.random((n,n)).copy(order='F')
#    c = np.zeros((n,n), order='F')
#
#    # symmetrize a & b
#    a = symmetrize(a)
#    b = symmetrize(b)
#
#    # ...
#    alpha = 1.
#    beta = 0.
#    expected = alpha * a @ b + beta * c
#    blas_csymm (alpha, a, b, c, beta=beta)
#    assert(np.allclose(c, expected, 1.e-7))
#    # ...
#
## ==============================================================================
#def test_ctrmm_1():
#    from cblas import blas_ctrmm
#
#    np.random.seed(2021)
#
#    n = 4
#    a = np.random.random((n,n)).copy(order='F')
#    b = np.random.random((n,n)).copy(order='F')
#
#    # make a triangular
#    a = triangulize(a)
#
#    # ...
#    alpha = 1.
#    expected = alpha * a @ b
#    blas_ctrmm (alpha, a, b)
#    assert(np.allclose(b, expected, 1.e-7))
#    # ...
#
## ==============================================================================
#def test_ctrsm_1():
#    from cblas import blas_ctrsm
#
#    np.random.seed(2021)
#
#    n = 4
#    a = np.random.random((n,n)).copy(order='F')
#    b = np.random.random((n,n)).copy(order='F')
#
#    # make a triangular
#    a = triangulize(a)
#
#    # ...
#    alpha = 1.
#    expected = b.copy()
#    b = alpha * a @ b
#    b = b.copy(order='F')
#    blas_ctrsm (alpha, a, b)
#    assert(np.allclose(b, expected, 1.e-7))
#    # ...
#
## ==============================================================================
#def test_csyrk_1():
#    from cblas import blas_csyrk
#
#    np.random.seed(2021)
#
#    n = 4
#    a = np.random.random((n,n)).copy(order='F')
#    c = np.zeros((n,n), order='F')
#
#    # syrketrize a
#    a = symmetrize(a)
#    a_T = a.T.copy(order='F')
#
#    # ...
#    alpha = 1.
#    beta = 0.
#    expected = alpha * a @ a_T + beta * c
#    blas_csyrk (alpha, a, c, beta=beta)
#    # we need to symmetrize the matrix
#    c = symmetrize(c)
#    assert(np.allclose(c, expected, 1.e-7))
#    # ...
#
## ==============================================================================
#def test_csyr2k_1():
#    from cblas import blas_csyr2k
#
#    np.random.seed(2021)
#
#    n = 4
#    a = np.random.random((n,n)).copy(order='F')
#    b = np.random.random((n,n)).copy(order='F')
#    c = np.zeros((n,n), order='F')
#
#    # syr2ketrize a & b
#    a = symmetrize(a)
#    b = symmetrize(b)
#    a_T = a.T.copy(order='F')
#    b_T = b.T.copy(order='F')
#
#    # ...
#    alpha = 1.
#    beta = 0.
#    expected = alpha * a @ b_T + alpha * b @ a_T + beta * c
#    blas_csyr2k (alpha, a, b, c, beta=beta)
#    # we need to symmetrize the matrix
#    c = symmetrize(c)
#    assert(np.allclose(c, expected, 1.e-7))
#    # ...

# ******************************************************************************
if __name__ == '__main__':

    # ... LEVEL 1
    test_ccopy_1()
    test_cswap_1()
    test_cscal_1()
    test_scnrm2_1()
    test_scasum_1()
    test_icamax_1()
    test_caxpy_1()
    # ...

    # ... LEVEL 2
    test_cgemv_1()
    test_cgbmv_1()
    test_ctrmv_1()
    test_ctbmv_1()
    test_ctpmv_1()
    test_ctrsv_1()
    test_ctbsv_1()
    test_ctpsv_1()
    # ...

#    # ... LEVEL 3
#    test_cgemm_1()
#    test_cgemm_2()
#    test_csymm_1()
#    test_ctrmm_1()
#    test_ctrsm_1()
#    test_csyrk_1()
#    test_csyr2k_1()
#    # ...
