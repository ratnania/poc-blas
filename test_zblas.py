import numpy as np
from scipy.sparse import diags
import scipy.linalg.blas as sp_blas
from utilities import symmetrize, triangulize, general_to_band, general_to_packed

TOL = 1.e-12

# ==============================================================================
#
#                                  LEVEL 1
#
# ==============================================================================

# ==============================================================================
def test_zcopy_1():
    from zblas import blas_zcopy

    np.random.seed(20)

    n = 3
    x = np.random.random(n) + np.random.random(n) * 1j
    y = np.zeros(n) + 0. * 1j

    x = np.array(x, dtype=np.complex128)
    y = np.array(y, dtype=np.complex128)

    # ...
    expected = y.copy()
    sp_blas.zcopy(x, expected)
    blas_zcopy (x, y)
    assert(np.allclose(y, expected, TOL))
    # ...

# ==============================================================================
def test_zswap_1():
    from zblas import blas_zswap

    np.random.seed(2021)

    n = 10
    x = np.random.random(n) + np.random.random(n) * 1j
    y = 2* np.random.random(n) + 1.*1j

    x = np.array(x, dtype=np.complex128)
    y = np.array(y, dtype=np.complex128)

    # ... we swap two times to get back to the original arrays
    expected_x = x.copy()
    expected_y = y.copy()
    sp_blas.zswap (x, y)
    blas_zswap (x, y)
    assert(np.allclose(x, expected_x, TOL))
    assert(np.allclose(y, expected_y, TOL))
    # ...

# ==============================================================================
def test_zscal_1():
    from zblas import blas_zscal

    np.random.seed(2021)

    n = 10
    x = np.random.random(n) + np.random.random(n) * 1j
    x = np.array(x, dtype=np.complex128)

    # ... we scale two times to get back to the original arrays
    expected = x.copy()
    alpha = np.complex128(2.5)
    inv_alpha = np.complex128(1./alpha)
    sp_blas.zscal (alpha, x)
    blas_zscal (inv_alpha, x)
    assert(np.allclose(x, expected, TOL))
    # ...

# ==============================================================================
def test_dznrm2_1():
    from zblas import blas_dznrm2

    np.random.seed(2021)

    n = 10
    x = np.random.random(n) + np.random.random(n) * 1j
    x = np.array(x, dtype=np.complex128)

    # ...
    expected = sp_blas.dznrm2(x)
    result   = blas_dznrm2 (x)
    assert(np.allclose(result, expected, TOL))
    # ...

# ==============================================================================
def test_dzasum_1():
    from zblas import blas_dzasum

    np.random.seed(2021)

    n = 10
    x = np.random.random(n) + np.random.random(n) * 1j
    x = np.array(x, dtype=np.complex128)

    # ...
    expected = sp_blas.dzasum(x)
    result   = blas_dzasum (x)
    assert(np.allclose(result, expected, TOL))
    # ...

# ==============================================================================
def test_izamax_1():
    from zblas import blas_izamax

    np.random.seed(2021)

    n = 10
    x = np.random.random(n) + np.random.random(n) * 1j
    x = np.array(x, dtype=np.complex128)

    # ...
    expected = sp_blas.izamax(x)
    result   = blas_izamax (x)
    assert(result == expected)
    # ...

# ==============================================================================
def test_zaxpy_1():
    from zblas import blas_zaxpy

    np.random.seed(2021)

    n = 10
    x = np.random.random(n) + np.random.random(n) * 1j
    y = np.random.random(n) + np.random.random(n) * 1j
    x = np.array(x, dtype=np.complex128)
    y = np.array(y, dtype=np.complex128)

    # ...
    alpha = np.complex128(2.5)
    expected = y.copy()
    sp_blas.zaxpy (x, expected, a=alpha)
    blas_zaxpy (x, y, a=alpha )
    assert(np.allclose(y, expected, TOL))
    # ...

# ==============================================================================
def test_zdotc_1():
    from zblas import blas_zdotc

    np.random.seed(2021)

    n = 3
    x = np.random.random(n) + np.random.random(n) * 1j
    y = np.random.random(n) + np.random.random(n) * 1j
    x = np.array(x, dtype=np.complex128)
    y = np.array(y, dtype=np.complex128)

    # ...
    expected = sp_blas.zdotc(x, y)
    result   = blas_zdotc (x, y)
    assert(np.linalg.norm(result-expected) < TOL)
    # ...

# ==============================================================================
def test_zdotu_1():
    from zblas import blas_zdotu

    np.random.seed(2021)

    n = 10
    x = np.random.random(n) + np.random.random(n) * 1j
    y = np.random.random(n) + np.random.random(n) * 1j
    x = np.array(x, dtype=np.complex128)
    y = np.array(y, dtype=np.complex128)

    # ...
    expected = sp_blas.zdotu(x, y)
    result   = blas_zdotu (x, y)
    assert(np.linalg.norm(result-expected) < TOL)
    # ...

# ==============================================================================
#
#                                  LEVEL 2
#
# ==============================================================================

# ==============================================================================
def test_zgemv_1():
    from zblas import blas_zgemv

    np.random.seed(2021)

    n = 10
    a = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    a = a.copy(order='F')
    x = np.random.random(n) + np.random.random(n) * 1j
    y = np.random.random(n) + np.random.random(n) * 1j

    a = np.array(a, dtype=np.complex128)
    x = np.array(x, dtype=np.complex128)
    y = np.array(y, dtype=np.complex128)

    # ...
    alpha = np.complex128(1.0)
    beta = np.complex128(0.5)
    expected = y.copy()
    expected = sp_blas.zgemv (alpha, a, x, beta=beta, y=expected)
    blas_zgemv (alpha, a, x, y, beta=beta)
    assert(np.allclose(y, expected, TOL))
    # ...

# ==============================================================================
def test_zgbmv_1():
    from zblas import blas_zgbmv

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

    ab = np.array(ab, dtype=np.complex128)
    x = np.array(x, dtype=np.complex128)
    y = np.array(y, dtype=np.complex128)

    # ...
    alpha = np.complex128(1.0)
    beta = np.complex128(0.5)
    expected = y.copy()
    expected = sp_blas.zgbmv (n, n, kl, ku, alpha, ab, x, beta=beta, y=expected)
    blas_zgbmv (kl, ku, alpha, ab, x, y, beta=beta)
    assert(np.allclose(y, expected, TOL))
    # ...

# ==============================================================================
def test_zhemv_1():
    from zblas import blas_zhemv

    np.random.seed(2021)

    n = 10
    a = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    a = a.copy(order='F')
    x = np.random.random(n) + np.random.random(n) * 1j
    y = np.random.random(n) + np.random.random(n) * 1j

    a = np.array(a, dtype=np.complex128)
    x = np.array(x, dtype=np.complex128)
    y = np.array(y, dtype=np.complex128)

    # ...
    alpha = np.complex128(1.0)
    beta = np.complex128(0.5)
    expected = y.copy()
    expected = sp_blas.zhemv (alpha, a, x, beta=beta, y=expected)
    blas_zhemv (alpha, a, x, y, beta=beta)
    assert(np.allclose(y, expected, TOL))
    # ...

# ==============================================================================
def test_zhbmv_1():
    from zblas import blas_zhbmv

    n = 5
    k = np.int32(2)
    a = np.array([[11, 12,  0,  0,  0],
                  [21, 22, 23,  0,  0],
                  [31, 32, 33, 34,  0],
                  [ 0, 42, 43, 44, 45],
                  [ 0,  0, 53, 54, 55]
                 ], dtype=np.float64)

    ab = general_to_band(k, k, a).copy(order='F')

    np.random.seed(2021)

    x = np.random.random(n) + np.random.random(n) * 1j
    y = np.random.random(n) + np.random.random(n) * 1j

    ab = np.array(ab, dtype=np.complex128)
    x = np.array(x, dtype=np.complex128)
    y = np.array(y, dtype=np.complex128)

    # ...
    alpha = np.complex128(1.0)
    beta = np.complex128(0.5)
    expected = y.copy()
    expected = sp_blas.zhbmv (k, alpha, ab, x, beta=beta, y=expected)
    blas_zhbmv (k, alpha, ab, x, y, beta=beta)
    assert(np.allclose(y, expected, TOL))
    # ...

# ==============================================================================
def test_zhpmv_1():
    from zblas import blas_zhpmv

    np.random.seed(2021)

    n = 4
    a = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    a = a.copy(order='F')
    x = np.random.random(n) + np.random.random(n) * 1j
    y = np.random.random(n) + np.random.random(n) * 1j

    # make a symmetric
    a = symmetrize(a)
    ap = general_to_packed(a)

    a = np.array(a, dtype=np.complex128)
    ap = np.array(ap, dtype=np.complex128)
    x = np.array(x, dtype=np.complex128)
    y = np.array(y, dtype=np.complex128)

    # ...
    alpha = np.complex128(1.)
    beta = np.complex128(0.5)
    expected = sp_blas.zhpmv (n, alpha, ap, x, y=y, beta=beta)
    blas_zhpmv (alpha, ap, x, y, beta=beta)
    assert(np.allclose(y, expected, TOL))
    # ...

# ==============================================================================
def test_ztrmv_1():
    from zblas import blas_ztrmv

    np.random.seed(2021)

    n = 10
    a = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    a = a.copy(order='F')
    x = np.random.random(n) + np.random.random(n) * 1j

    # make a triangular
    a = triangulize(a)

    a = np.array(a, dtype=np.complex128)
    x = np.array(x, dtype=np.complex128)

    # ...
    expected = sp_blas.ztrmv (a, x)
    blas_ztrmv (a, x)
    assert(np.allclose(x, expected, TOL))
    # ...

# ==============================================================================
def test_ztbmv_1():
    from zblas import blas_ztbmv

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

    ab = np.array(ab, dtype=np.complex128)
    a = np.array(a, dtype=np.complex128)
    x = np.array(x, dtype=np.complex128)

    # ...
    expected = sp_blas.ztbmv (k, ab, x)
    blas_ztbmv (k, ab, x)
    assert(np.allclose(x, expected, TOL))
    # ...

# ==============================================================================
def test_ztpmv_1():
    from zblas import blas_ztpmv

    np.random.seed(2021)

    n = 10
    a = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    a = a.copy(order='F')
    x = np.random.random(n) + np.random.random(n) * 1j

    # make a triangular
    a = triangulize(a)
    ap = general_to_packed(a)

    ap = np.array(ap, dtype=np.complex128)
    a = np.array(a, dtype=np.complex128)
    x = np.array(x, dtype=np.complex128)

    # ...
    expected = sp_blas.ztpmv (n, ap, x)
    blas_ztpmv (ap, x)
    assert(np.allclose(x, expected, TOL))
    # ...

# ==============================================================================
def test_ztrsv_1():
    from zblas import blas_ztrsv

    np.random.seed(2021)

    n = 10
    a = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    a = a.copy(order='F')
    x = np.random.random(n) + np.random.random(n) * 1j

    # make a triangular
    a = triangulize(a)

    a = np.array(a, dtype=np.complex128)
    x = np.array(x, dtype=np.complex128)

    # ...
    b = x.copy()
    expected = sp_blas.ztrsv (a, b)
    blas_ztrsv (a, x)
    assert(np.linalg.norm(x-expected) < TOL)
    # ...

# ==============================================================================
def test_ztbsv_1():
    from zblas import blas_ztbsv

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

    ab = np.array(ab, dtype=np.complex128)
    x = np.array(x, dtype=np.complex128)

    # ...
    expected = sp_blas.ztbsv (k, ab, x)
    blas_ztbsv (k, ab, x)
    assert(np.allclose(x, expected, TOL))
    # ...

# ==============================================================================
def test_ztpsv_1():
    from zblas import blas_ztpsv

    np.random.seed(2021)

    n = 10
    a = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    a = a.copy(order='F')
    x = np.random.random(n) + np.random.random(n) * 1j

    # make a triangular
    a = triangulize(a)
    ap = general_to_packed(a)

    ap = np.array(ap, dtype=np.complex128)
    a = np.array(a, dtype=np.complex128)
    x = np.array(x, dtype=np.complex128)

    # ...
    x.copy()
    expected = sp_blas.ztpsv (n, ap, x)
    blas_ztpsv (ap, x)
    assert(np.linalg.norm(x-expected) < TOL)
    # ...

# ==============================================================================
def test_zgeru_1():
    from zblas import blas_zgeru

    np.random.seed(2021)

    n = 10
    a = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    a = a.copy(order='F')
    x = np.random.random(n) + np.random.random(n) * 1j
    y = np.random.random(n) + np.random.random(n) * 1j

    a = np.array(a, dtype=np.complex128)
    x = np.array(x, dtype=np.complex128)
    y = np.array(y, dtype=np.complex128)

    # ...
    alpha = np.complex128(1.)
    expected = sp_blas.zgeru (alpha, x, y, a=a)
    blas_zgeru (alpha, x, y, a)
    assert(np.allclose(a, expected, TOL))
    # ...

# ==============================================================================
def test_zgerc_1():
    from zblas import blas_zgerc

    np.random.seed(2021)

    n = 10
    a = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    a = a.copy(order='F')
    x = np.random.random(n) + np.random.random(n) * 1j
    y = np.random.random(n) + np.random.random(n) * 1j

    a = np.array(a, dtype=np.complex128)
    x = np.array(x, dtype=np.complex128)
    y = np.array(y, dtype=np.complex128)

    # ...
    alpha = np.complex128(1.)
    expected = sp_blas.zgerc (alpha, x, y, a=a)
    blas_zgerc (alpha, x, y, a)
    assert(np.allclose(a, expected, TOL))
    # ...

# ==============================================================================
def test_zher_1():
    from zblas import blas_zher

    np.random.seed(2021)

    n = 4
    a = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    a = a.copy(order='F')
    x = np.random.random(n) + np.random.random(n) * 1j

    # syrketrize a
    a = symmetrize(a)

    a = np.array(a, dtype=np.complex128)
    x = np.array(x, dtype=np.complex128)

    # ...
    alpha = np.float64(1.)
    expected = sp_blas.zher (alpha, x, a=a)
    blas_zher (alpha, x, a)
    assert(np.allclose(a, expected, TOL))
    # ...

# ==============================================================================
def test_zhpr_1():
    from zblas import blas_zhpr

    np.random.seed(2021)

    n = 10
    a = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    a = a.copy(order='F')
    x = np.random.random(n) + np.random.random(n) * 1j

    # syrketrize a
    a = symmetrize(a)
    ap = general_to_packed(a)

    ap = np.array(ap, dtype=np.complex128)
    a = np.array(a, dtype=np.complex128)
    x = np.array(x, dtype=np.complex128)

    # ...
    alpha = np.float64(1.)
    expected = sp_blas.zhpr (n, alpha, x, ap)
    blas_zhpr (alpha, x, ap)
    assert(np.allclose(ap, expected, TOL))
    # ...

# ==============================================================================
def test_zher2_1():
    from zblas import blas_zher2

    np.random.seed(2021)

    n = 4
    a = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    a = a.copy(order='F')
    x = np.random.random(n) + np.random.random(n) * 1j
    y = np.random.random(n) + np.random.random(n) * 1j

    # syrketrize a
    a = symmetrize(a)

    a = np.array(a, dtype=np.complex128)
    x = np.array(x, dtype=np.complex128)
    y = np.array(y, dtype=np.complex128)

    # ...
    alpha = np.complex128(1.)
    expected = sp_blas.zher2 (alpha, x, y, a=a)
    blas_zher2 (alpha, x, y, a)
    assert(np.allclose(a, expected, TOL))
    # ...

# ==============================================================================
def test_zhpr2_1():
    from zblas import blas_zhpr2

    np.random.seed(2021)

    n = 10
    a = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    a = a.copy(order='F')
    x = np.random.random(n) + np.random.random(n) * 1j
    y = np.random.random(n) + np.random.random(n) * 1j

    # syrketrize a
    a = symmetrize(a)
    ap = general_to_packed(a)

    ap = np.array(ap, dtype=np.complex128)
    a = np.array(a, dtype=np.complex128)
    x = np.array(x, dtype=np.complex128)
    y = np.array(x, dtype=np.complex128)

    # ...
    alpha = np.complex128(1.)
    expected = sp_blas.zhpr2 (n, alpha, x, y, ap)
    blas_zhpr2 (alpha, x, y, ap)
    assert(np.allclose(ap, expected, TOL))
    # ...

# ==============================================================================
#
#                                  LEVEL 3
#
# ==============================================================================

# ==============================================================================
def test_zgemm_1():
    from zblas import blas_zgemm

    np.random.seed(2021)

    n = 4
    a = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    b = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    c = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    a = a.copy(order='F')
    b = b.copy(order='F')
    c = c.copy(order='F')

    a = np.array(a, dtype=np.complex128)
    b = np.array(b, dtype=np.complex128)
    c = np.array(c, dtype=np.complex128)

    # ...
    alpha = np.complex128(1.)
    beta = np.complex128(.5)
    expected = sp_blas.zgemm (alpha, a, b, c=c, beta=beta)
    blas_zgemm (alpha, a, b, c, beta=beta)
    assert(np.allclose(c, expected, TOL))
    # ...

# ==============================================================================
def test_zsymm_1():
    from zblas import blas_zsymm

    np.random.seed(2021)

    n = 4
    a = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    b = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    c = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    a = a.copy(order='F')
    b = b.copy(order='F')
    c = c.copy(order='F')

    a = np.array(a, dtype=np.complex128)
    b = np.array(b, dtype=np.complex128)
    c = np.array(c, dtype=np.complex128)

    # symmetrize a & b
    a = symmetrize(a)
    b = symmetrize(b)

    # ...
    alpha = np.complex128(1.)
    beta = np.complex128(.5)
    expected = sp_blas.zsymm (alpha, a, b, c=c, beta=beta)
    blas_zsymm (alpha, a, b, c, beta=beta)
    assert(np.allclose(c, expected, TOL))
    # ...

# ==============================================================================
def test_zhemm_1():
    from zblas import blas_zhemm

    np.random.seed(2021)

    n = 4
    a = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    b = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    c = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    a = a.copy(order='F')
    b = b.copy(order='F')
    c = c.copy(order='F')

    a = np.array(a, dtype=np.complex128)
    b = np.array(b, dtype=np.complex128)
    c = np.array(c, dtype=np.complex128)

    # symmetrize a & b
    a = symmetrize(a)
    b = symmetrize(b)

    # ...
    alpha = np.complex128(1.)
    beta = np.complex128(.5)
    expected = sp_blas.zhemm (alpha, a, b, beta=beta, c=c)
    blas_zhemm (alpha, a, b, c, beta=beta)
    assert(np.allclose(c, expected, TOL))
    # ...

# ==============================================================================
def test_zsyrk_1():
    from zblas import blas_zsyrk

    np.random.seed(2021)

    n = 4
    a = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    c = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    a = a.copy(order='F')
    c = c.copy(order='F')

    a = np.array(a, dtype=np.complex128)
    c = np.array(c, dtype=np.complex128)

    # syrketrize a
    a = symmetrize(a)
    a_T = a.T.copy(order='F')

    # ...
    alpha = np.complex128(1.)
    beta = np.complex128(.5)
    expected = sp_blas.zsyrk (alpha, a, beta=beta, c=c)
    blas_zsyrk (alpha, a, c, beta=beta)
    assert(np.linalg.norm(c- expected) < TOL)
    # ...

# ==============================================================================
def test_zsyr2k_1():
    from zblas import blas_zsyr2k

    np.random.seed(2021)

    n = 4
    a = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    b = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    c = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    a = a.copy(order='F')
    b = b.copy(order='F')
    c = c.copy(order='F')

    a = np.array(a, dtype=np.complex128)
    b = np.array(b, dtype=np.complex128)
    c = np.array(c, dtype=np.complex128)

    # syr2ketrize a & b
    a = symmetrize(a)
    b = symmetrize(b)
    a_T = a.T.copy(order='F')
    b_T = b.T.copy(order='F')

    # ...
    alpha = np.complex128(1.)
    beta = np.complex128(.5)
    expected = sp_blas.zsyr2k (alpha, a, b, beta=beta, c=c)
    blas_zsyr2k (alpha, a, b, c, beta=beta)
    assert(np.allclose(c, expected, TOL))
    # ...

# ==============================================================================
def test_zherk_1():
    from zblas import blas_zherk

    np.random.seed(2021)

    n = 4
    a = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    c = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    a = a.copy(order='F')
    c = c.copy(order='F')

    a = np.array(a, dtype=np.complex128)
    c = np.array(c, dtype=np.complex128)

    # syrketrize a
    a = symmetrize(a)
    a_T = a.T.copy(order='F')

    # ...
    alpha = np.complex128(1.)
    beta = np.complex128(.5)
    expected = sp_blas.zherk (alpha, a, beta=beta, c=c)
    blas_zherk (alpha, a, c, beta=beta)
    assert(np.linalg.norm(c- expected) < TOL)
    # ...

# ==============================================================================
def test_zher2k_1():
    from zblas import blas_zher2k

    np.random.seed(2021)

    n = 4
    a = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    b = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    c = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    a = a.copy(order='F')
    b = b.copy(order='F')
    c = c.copy(order='F')

    a = np.array(a, dtype=np.complex128)
    b = np.array(b, dtype=np.complex128)
    c = np.array(c, dtype=np.complex128)

    # syr2ketrize a & b
    a = symmetrize(a)
    b = symmetrize(b)

    # ...
    alpha = np.complex128(1.)
    beta = np.complex128(.5)
    expected = sp_blas.zher2k (alpha, a, b, beta=beta, c=c)
    blas_zher2k (alpha, a, b, c, beta=beta)
    assert(np.allclose(c, expected, TOL))
    # ...

# ==============================================================================
def test_ztrmm_1():
    from zblas import blas_ztrmm

    np.random.seed(2021)

    n = 4
    a = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    b = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    a = a.copy(order='F')
    b = b.copy(order='F')

    a = np.array(a, dtype=np.complex128)
    b = np.array(b, dtype=np.complex128)

    # make a triangular
    a = triangulize(a)

    # ...
    alpha = np.complex128(1.)
    expected = sp_blas.ztrmm (alpha, a, b)
    blas_ztrmm (alpha, a, b)
    assert(np.allclose(b, expected, TOL))
    # ...

# ==============================================================================
def test_ztrsm_1():
    from zblas import blas_ztrsm

    np.random.seed(2021)

    n = 4
    a = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    b = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    a = a.copy(order='F')
    b = b.copy(order='F')

    a = np.array(a, dtype=np.complex128)
    b = np.array(b, dtype=np.complex128)

    # make a triangular
    a = triangulize(a)

    # ...
    alpha = np.complex128(1.)
    expected = sp_blas.ztrsm (alpha, a, b)
    blas_ztrsm (alpha, a, b)
    assert(np.linalg.norm(b- expected) < TOL)
    # ...

# ******************************************************************************
if __name__ == '__main__':

    # ... LEVEL 1
    test_zcopy_1()
    test_zswap_1()
    test_zscal_1()
    test_dznrm2_1()
    test_dzasum_1()
    test_izamax_1()
    test_zaxpy_1()
    test_zdotc_1()
    test_zdotu_1()
    # ...

    # ... LEVEL 2
    test_zgemv_1()
    test_zgbmv_1()
    test_zhemv_1()
    test_zhbmv_1()
    test_zhpmv_1()
    test_ztrmv_1()
    test_ztbmv_1()
    test_ztpmv_1()
    test_ztrsv_1()
    test_ztbsv_1()
    test_ztpsv_1()
    test_zgeru_1()
    test_zgerc_1()
    test_zher_1()
    test_zhpr_1()
    test_zher2_1()
    test_zhpr2_1()
    # ...

    # ... LEVEL 3
    test_zgemm_1()
    test_zsymm_1()
    test_zhemm_1()
    test_zsyrk_1()
    test_zsyr2k_1()
    test_zherk_1()
    test_zher2k_1()
    test_ztrmm_1()
    test_ztrsm_1()
    # ...
