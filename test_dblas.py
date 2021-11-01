import numpy as np
from scipy.sparse import diags
import scipy.linalg.blas as sp_blas
from utilities import symmetrize, triangulize, general_to_band, general_to_packed

#TODO dsdot

TOL = 1.e-13

# ==============================================================================
#
#                                  LEVEL 1
#
# ==============================================================================

# ==============================================================================
def test_drotg_1():
    from dblas import blas_drotg

    a = b = np.float64(1.)
    c, s = blas_drotg (a, b)
    expected_c, expected_s = sp_blas.drotg (a, b)
    assert(np.abs(c - expected_c) < 1.e-10)
    assert(np.abs(s - expected_s) < 1.e-10)

# ==============================================================================
def test_drotmg_1():
    from dblas import blas_drotmg

    d1 = d2 = np.float64(1.)
    x1 = y1 = np.float64(.5)
    result = np.zeros(5, dtype=np.float64)
    blas_drotmg (d1, d2, x1, y1, result)
    expected = sp_blas.drotmg (d1, d2, x1, y1)
    assert(np.allclose(result, expected, TOL))

# ==============================================================================
def test_drot_1():
    from dblas import blas_drot

    np.random.seed(2021)

    n = 10
    x = np.random.random(n)
    y = np.random.random(n)
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    expected_x = x.copy()
    expected_y = y.copy()

    # ...
    one = np.float64(1.)
    c, s = sp_blas.drotg (one, one)
    c = np.float64(c)
    s = np.float64(s)
    expected_x, expected_y = sp_blas.drot(x, y, c, s)
    blas_drot (x, y, c, s)
    assert(np.allclose(x, expected_x, TOL))
    assert(np.allclose(y, expected_y, TOL))
    # ...

# ==============================================================================
def test_drotm_1():
    from dblas import blas_drotm

    np.random.seed(2021)

    n = 10
    x = np.random.random(n)
    y = np.random.random(n)
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    expected_x = x.copy()
    expected_y = y.copy()

    # ...
    d1 = d2 = np.float64(1.)
    x1 = y1 = np.float64(.5)
    param = sp_blas.drotmg (d1, d2, x1, y1)
    param = np.array(param, dtype=np.float64)
    expected_x, expected_y = sp_blas.drotm(x, y, param)
    blas_drotm (x, y, param)
    assert(np.allclose(x, expected_x, TOL))
    assert(np.allclose(y, expected_y, TOL))
    # ...

# ==============================================================================
def test_dcopy_1():
    from dblas import blas_dcopy

    np.random.seed(2021)

    n = 10
    x = np.random.random(n)
    x = np.array(x, dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)

    # ...
    expected  = np.zeros(n, dtype=np.float64)
    sp_blas.dcopy(x, expected)
    blas_dcopy (x, y)
    assert(np.allclose(y, expected, TOL))
    # ...

# ==============================================================================
def test_dswap_1():
    from dblas import blas_dswap

    np.random.seed(2021)

    n = 10
    x = np.random.random(n)
    x = np.array(x, dtype=np.float64)
    y = 2* np.random.random(n) + 1.
    y = np.array(y, dtype=np.float64)

    # ... we swap two times to get back to the original arrays
    expected_x = x.copy()
    expected_y = y.copy()
    sp_blas.dswap (x, y)
    blas_dswap (x, y)
    assert(np.allclose(x, expected_x, TOL))
    assert(np.allclose(y, expected_y, TOL))
    # ...

# ==============================================================================
def test_dscal_1():
    from dblas import blas_dscal

    np.random.seed(2021)

    n = 10
    x = np.random.random(n)
    x = np.array(x, dtype=np.float64)

    # ... we scale two times to get back to the original arrays
    expected = x.copy()
    alpha = np.float64(2.5)
    sp_blas.dscal (alpha, x)
    blas_dscal (np.float64(1./alpha), x)
    assert(np.allclose(x, expected, TOL))
    # ...

# ==============================================================================
def test_ddot_1():
    from dblas import blas_ddot

    np.random.seed(2021)

    n = 10
    x = np.random.random(n)
    y = np.random.random(n)
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    # ...
    expected = sp_blas.ddot(x, y)
    result   = blas_ddot (x, y)
    assert(np.allclose(result, expected, 1.e-6))
    # ...

# ==============================================================================
def test_dnrm2_1():
    from dblas import blas_dnrm2

    np.random.seed(2021)

    n = 10
    x = np.random.random(n)
    x = np.array(x, dtype=np.float64)

    # ...
    expected = sp_blas.dnrm2(x)
    result   = blas_dnrm2 (x)
    assert(np.allclose(result, expected, 1.e-6))
    # ...

# ==============================================================================
def test_dasum_1():
    from dblas import blas_dasum

    np.random.seed(2021)

    n = 10
    x = np.random.random(n)
    x = np.array(x, dtype=np.float64)

    # ...
    expected = sp_blas.dasum(x)
    result   = blas_dasum (x)
    assert(np.allclose(result, expected, TOL))
    # ...

# ==============================================================================
def test_idamax_1():
    from dblas import blas_idamax

    np.random.seed(2021)

    n = 10
    x = np.random.random(n)
    x = np.array(x, dtype=np.float64)

    # ...
    expected = sp_blas.idamax(x)
    result   = blas_idamax (x)
    assert(result == expected)
    # ...

# ==============================================================================
def test_daxpy_1():
    from dblas import blas_daxpy

    np.random.seed(2021)

    n = 10
    x = np.random.random(n)
    y = np.random.random(n)
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    # ...
    alpha = np.float64(2.5)
    expected = y.copy()
    sp_blas.daxpy (x, expected, a=alpha)
    blas_daxpy (x, y, a=alpha )
    assert(np.allclose(y, expected, TOL))
    # ...

# ==============================================================================
#
#                                  LEVEL 2
#
# ==============================================================================

# ==============================================================================
def test_dgemv_1():
    from dblas import blas_dgemv

    np.random.seed(2021)

    n = 10
    a = np.random.random((n,n)).copy(order='F')
    x = np.random.random(n)
    y = np.ones(n)

    a = np.array(a, dtype=np.float64)
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    # ...
    alpha = np.float64(1.)
    beta = np.float64(0.5)
    expected = y.copy()
    expected = sp_blas.dgemv (alpha, a, x, beta=beta, y=expected)
    blas_dgemv (alpha, a, x, y, beta=beta)
    assert(np.allclose(y, expected, TOL))
    # ...

# ==============================================================================
def test_dgbmv_1():
    from dblas import blas_dgbmv

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

    ab = np.array(ab, dtype=np.float64)
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    # ...
    alpha = np.float64(1.)
    beta = np.float64(0.5)
    expected = y.copy()
    expected = sp_blas.dgbmv (n, n, kl, ku, alpha, ab, x, beta=beta, y=expected)
    blas_dgbmv (kl, ku, alpha, ab, x, y, beta=beta)
    assert(np.allclose(y, expected, TOL))
    # ...

# ==============================================================================
def test_dsymv_1():
    from dblas import blas_dsymv

    np.random.seed(2021)

    n = 4
    a = np.random.random((n,n)).copy(order='F')
    x = np.random.random(n)
    y = np.random.random(n)

    a = np.array(a, dtype=np.float64)
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    # make a symmetric
    a = symmetrize(a)
    # make a triangular
    a = triangulize(a)

    # ...
    alpha = np.float64(1.)
    beta = np.float64(0.5)
    expected = sp_blas.dsymv (alpha, a, x, y=y, beta=beta)
    blas_dsymv (alpha, a, x, y, beta=beta)
    assert(np.allclose(y, expected, TOL))
    # ...

# ==============================================================================
def test_dsbmv_1():
    from dblas import blas_dsbmv

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
    y = np.zeros(n, dtype=np.float64)

    ab = np.array(ab, dtype=np.float64)
    x = np.array(x, dtype=np.float64)

    # ...
    alpha = np.float64(1.)
    beta = np.float64(0.5)
    expected = y.copy()
    expected = sp_blas.dsbmv (k, alpha, ab, x, beta=beta, y=expected)
    blas_dsbmv (k, alpha, ab, x, y, beta=beta)
    assert(np.allclose(y, expected, TOL))
    # ...

# ==============================================================================
def test_dspmv_1():
    from dblas import blas_dspmv

    np.random.seed(2021)

    n = 4
    a = np.random.random((n,n)).copy(order='F')
    x = np.random.random(n)
    y = np.random.random(n)

    # make a symmetric
    a = symmetrize(a)
    ap = general_to_packed(a)

    ap = np.array(ap, dtype=np.float64)
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    # ...
    alpha = np.float64(1.)
    beta = np.float64(0.5)
    expected = sp_blas.dspmv (n, alpha, ap, x, y=y, beta=beta)
    blas_dspmv (alpha, ap, x, y, beta=beta)
    assert(np.allclose(y, expected, TOL))
    # ...

# ==============================================================================
def test_dtrmv_1():
    from dblas import blas_dtrmv

    np.random.seed(2021)

    n = 10
    a = np.random.random((n,n)).copy(order='F')
    x = np.random.random(n)

    a = np.array(a, dtype=np.float64)
    x = np.array(x, dtype=np.float64)

    # make a triangular
    a = triangulize(a)

    # ...
    expected = sp_blas.dtrmv (a, x)
    blas_dtrmv (a, x)
    assert(np.allclose(x, expected, TOL))
    # ...

# ==============================================================================
def test_dtbmv_1():
    from dblas import blas_dtbmv

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

    ab = np.array(ab, dtype=np.float64)
    x = np.array(x, dtype=np.float64)

    # ...
    expected = sp_blas.dtbmv (k, ab, x)
    blas_dtbmv (k, ab, x)
    assert(np.allclose(x, expected, TOL))
    # ...

# ==============================================================================
def test_dtpmv_1():
    from dblas import blas_dtpmv

    np.random.seed(2021)

    n = 10
    a = np.random.random((n,n)).copy(order='F')
    x = np.random.random(n)

    # make a triangular
    a = triangulize(a)
    ap = general_to_packed(a)

    ap = np.array(ap, dtype=np.float64)
    x = np.array(x, dtype=np.float64)

    # ...
    expected = sp_blas.dtpmv (n, ap, x)
    blas_dtpmv (ap, x)
    assert(np.allclose(x, expected, TOL))
    # ...

# ==============================================================================
def test_dtrsv_1():
    from dblas import blas_dtrsv

    np.random.seed(2021)

    n = 10
    a = np.random.random((n,n)).copy(order='F')
    x = np.random.random(n)

    # make a triangular
    a = triangulize(a)

    a = np.array(a, dtype=np.float64)
    x = np.array(x, dtype=np.float64)

    # ...
    b = x.copy()
    expected = sp_blas.dtrsv (a, b)
    blas_dtrsv (a, x)
    assert(np.allclose(x, expected, TOL))
    # ...

# ==============================================================================
def test_dtbsv_1():
    from dblas import blas_dtbsv

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

    ab = np.array(ab, dtype=np.float64)
    x = np.array(x, dtype=np.float64)

    # ...
    expected = sp_blas.dtbsv (k, ab, x)
    blas_dtbsv (k, ab, x)
    assert(np.allclose(x, expected, TOL))
    # ...

# ==============================================================================
def test_dtpsv_1():
    from dblas import blas_dtpsv

    np.random.seed(2021)

    n = 10
    a = np.random.random((n,n)).copy(order='F')
    x = np.random.random(n)

    # make a triangular
    a = triangulize(a)
    ap = general_to_packed(a)

    ap = np.array(ap, dtype=np.float64)
    a = np.array(a, dtype=np.float64)
    x = np.array(x, dtype=np.float64)

    # ...
    b = x.copy()
    expected = sp_blas.dtpsv (n, ap, b)
    blas_dtpsv (ap, x)
    assert(np.allclose(x, expected, TOL))
    # ...

# ==============================================================================
def test_dger_1():
    from dblas import blas_dger

    np.random.seed(2021)

    n = 10
    a = np.random.random((n,n)).copy(order='F')

    x = np.ones(n)
    y = np.zeros(n)

    a = np.array(a, dtype=np.float64)
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    # ...
    alpha = np.float64(1.)
    expected = sp_blas.dger (alpha, x, y, a=a)
    blas_dger (alpha, x, y, a)
    assert(np.allclose(a, expected, TOL))
    # ...

# ==============================================================================
def test_dsyr_1():
    from dblas import blas_dsyr

    np.random.seed(2021)

    n = 4
    a = np.random.random((n,n)).copy(order='F')
    x = np.random.random(n)

    # syrketrize a
    a = symmetrize(a)

    a = np.array(a, dtype=np.float64)
    x = np.array(x, dtype=np.float64)

    # ...
    alpha = np.float64(1.)
    expected = sp_blas.dsyr (alpha, x, a=a)
    blas_dsyr (alpha, x, a)
    assert(np.allclose(a, expected, TOL))
    # ...

# ==============================================================================
def test_dspr_1():
    from dblas import blas_dspr

    np.random.seed(2021)

    n = 10
    a = np.random.random((n,n)).copy(order='F')
    x = np.random.random(n)

    # syrketrize a
    a = symmetrize(a)
    ap = general_to_packed(a)

    ap = np.array(ap, dtype=np.float64)
    a = np.array(a, dtype=np.float64)
    x = np.array(x, dtype=np.float64)

    # ...
    alpha = np.float64(1.)
    expected = sp_blas.dspr (n, alpha, x, ap)
    blas_dspr (alpha, x, ap)
    assert(np.allclose(ap, expected, TOL))
    # ...

# ==============================================================================
def test_dsyr2_1():
    from dblas import blas_dsyr2

    np.random.seed(2021)

    n = 4
    a = np.random.random((n,n)).copy(order='F')
    x = np.random.random(n)
    y = np.random.random(n)

    # syrketrize a
    a = symmetrize(a)

    a = np.array(a, dtype=np.float64)
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    # ...
    alpha = np.float64(1.)
    expected = sp_blas.dsyr2 (alpha, x, y, a=a)
    blas_dsyr2 (alpha, x, y, a=a)
    assert(np.allclose(a, expected, TOL))
    # ...

# ==============================================================================
def test_dspr2_1():
    from dblas import blas_dspr2

    np.random.seed(2021)

    n = 4
    a = np.random.random((n,n)).copy(order='F')
    x = np.random.random(n)
    y = np.random.random(n)

    # syrketrize a
    a = symmetrize(a)
    ap = general_to_packed(a)

    ap = np.array(ap, dtype=np.float64)
    a = np.array(a, dtype=np.float64)
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    # ...
    alpha = np.float64(1.)
    expected = sp_blas.dspr2 (n, alpha, x, y, ap)
    blas_dspr2 (alpha, x, y, ap)
    assert(np.allclose(ap, expected, TOL))
    # ...

# ==============================================================================
#
#                                  LEVEL 3
#
# ==============================================================================

# ==============================================================================
def test_dgemm_1():
    from dblas import blas_dgemm

    np.random.seed(2021)

    n = 4
    a = np.random.random((n,n)).copy(order='F')
    b = np.random.random((n,n)).copy(order='F')
    c = np.zeros((n,n), order='F')

    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    c = np.array(c, dtype=np.float64)

    # ...
    alpha = np.float64(1.)
    beta = np.float64(0.5)
    expected = sp_blas.dgemm (alpha, a, b, c=c, beta=beta)
    blas_dgemm (alpha, a, b, c, beta=beta)
    assert(np.allclose(c, expected, TOL))
    # ...

# ==============================================================================
def test_dsymm_1():
    from dblas import blas_dsymm

    np.random.seed(2021)

    n = 4
    a = np.random.random((n,n)).copy(order='F')
    b = np.random.random((n,n)).copy(order='F')
    c = np.zeros((n,n), order='F')

    # symmetrize a & b
    a = symmetrize(a)
    b = symmetrize(b)

    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    c = np.array(c, dtype=np.float64)

    # ...
    alpha = np.float64(1.)
    beta = np.float64(0.5)
    expected = sp_blas.dsymm (alpha, a, b, c=c, beta=beta)
    blas_dsymm (alpha, a, b, c, beta=beta)
    assert(np.allclose(c, expected, TOL))
    # ...

# ==============================================================================
def test_dtrmm_1():
    from dblas import blas_dtrmm

    np.random.seed(2021)

    n = 4
    a = np.random.random((n,n)).copy(order='F')
    b = np.random.random((n,n)).copy(order='F')

    # make a triangular
    a = triangulize(a)

    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)

    # ...
    alpha = np.float64(1.)
    expected = sp_blas.dtrmm (alpha, a, b)
    blas_dtrmm (alpha, a, b)
    assert(np.allclose(b, expected, TOL))
    # ...

# ==============================================================================
def test_dtrsm_1():
    from dblas import blas_dtrsm

    np.random.seed(2021)

    n = 4
    a = np.random.random((n,n)).copy(order='F')
    b = np.random.random((n,n)).copy(order='F')

    # make a triangular
    a = triangulize(a)

    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)

    # ...
    alpha = np.float64(1.)
    expected = sp_blas.dtrsm (alpha, a, b)
    blas_dtrsm (alpha, a, b)
    assert(np.allclose(b, expected, TOL))
    # ...

# ==============================================================================
def test_dsyrk_1():
    from dblas import blas_dsyrk

    np.random.seed(2021)

    n = 4
    a = np.random.random((n,n)).copy(order='F')
    c = np.zeros((n,n), order='F')

    # syrketrize a
    a = symmetrize(a)
    a_T = a.T.copy(order='F')

    a = np.array(a, dtype=np.float64)
    c = np.array(c, dtype=np.float64)

    # ...
    alpha = np.float64(1.)
    beta = np.float64(0.)
    expected = sp_blas.dsyrk (alpha, a, c=c, beta=beta)
    blas_dsyrk (alpha, a, c, beta=beta)
    assert(np.allclose(c, expected, TOL))
    # ...

# ==============================================================================
def test_dsyr2k_1():
    from dblas import blas_dsyr2k

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

    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    c = np.array(c, dtype=np.float64)

    # ...
    alpha = np.float64(1.)
    beta = np.float64(0.)
    expected = sp_blas.dsyr2k (alpha, a, b, c=c, beta=beta)
    blas_dsyr2k (alpha, a, b, c, beta=beta)
    assert(np.allclose(c, expected, TOL))
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
    test_dsymm_1()
    test_dtrmm_1()
    test_dtrsm_1()
    test_dsyrk_1()
    test_dsyr2k_1()
    # ...
