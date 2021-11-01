import numpy as np
from scipy.sparse import diags
import scipy.linalg.blas as sp_blas
from utilities import symmetrize, triangulize, general_to_band, general_to_packed
from utilities import random_array

TOL = 1.e-12
DTYPE = np.complex128

# ==============================================================================
#
#                                  LEVEL 1
#
# ==============================================================================

# ==============================================================================
def test_zcopy_1():
    from zblas import blas_zcopy

    n = 3
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # ...
    expected = y.copy()
    sp_blas.zcopy(x, expected)
    blas_zcopy (x, y)
    assert(np.allclose(y, expected, TOL))
    # ...

# ==============================================================================
def test_zswap_1():
    from zblas import blas_zswap

    n = 10
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

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

    n = 10
    x = random_array(n, dtype=DTYPE)

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

    n = 10
    x = random_array(n, dtype=DTYPE)

    # ...
    expected = sp_blas.dznrm2(x)
    result   = blas_dznrm2 (x)
    assert(np.allclose(result, expected, TOL))
    # ...

# ==============================================================================
def test_dzasum_1():
    from zblas import blas_dzasum

    n = 10
    x = random_array(n, dtype=DTYPE)

    # ...
    expected = sp_blas.dzasum(x)
    result   = blas_dzasum (x)
    assert(np.allclose(result, expected, TOL))
    # ...

# ==============================================================================
def test_izamax_1():
    from zblas import blas_izamax

    n = 10
    x = random_array(n, dtype=DTYPE)

    # ...
    expected = sp_blas.izamax(x)
    result   = blas_izamax (x)
    assert(result == expected)
    # ...

# ==============================================================================
def test_zaxpy_1():
    from zblas import blas_zaxpy

    n = 10
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

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

    n = 3
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # ...
    expected = sp_blas.zdotc(x, y)
    result   = blas_zdotc (x, y)
    assert(np.linalg.norm(result-expected) < TOL)
    # ...

# ==============================================================================
def test_zdotu_1():
    from zblas import blas_zdotu

    n = 10
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

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

    n = 10
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

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
                 ], dtype=np.complex128)

    ab = general_to_band(kl, ku, a).copy(order='F')

    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

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

    n = 10
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

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
                 ], dtype=np.complex128)

    ab = general_to_band(k, k, a).copy(order='F')

    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

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

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # make a symmetric
    a = symmetrize(a)
    ap = general_to_packed(a)

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

    n = 10
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)

    # make a triangular
    a = triangulize(a)

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
                 ], dtype=np.complex128)

    ab = general_to_band(k, k, a).copy(order='F')

    x = random_array(n, dtype=DTYPE)

    # make a triangular
    a = triangulize(a)

    # ...
    expected = sp_blas.ztbmv (k, ab, x)
    blas_ztbmv (k, ab, x)
    assert(np.allclose(x, expected, TOL))
    # ...

# ==============================================================================
def test_ztpmv_1():
    from zblas import blas_ztpmv

    n = 10
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)

    # make a triangular
    a = triangulize(a)
    ap = general_to_packed(a)

    # ...
    expected = sp_blas.ztpmv (n, ap, x)
    blas_ztpmv (ap, x)
    assert(np.allclose(x, expected, TOL))
    # ...

# ==============================================================================
def test_ztrsv_1():
    from zblas import blas_ztrsv

    n = 10
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)

    # make a triangular
    a = triangulize(a)

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
                 ], dtype=np.complex128)

    ab = general_to_band(k, k, a).copy(order='F')

    x = random_array(n, dtype=DTYPE)

    # ...
    expected = sp_blas.ztbsv (k, ab, x)
    blas_ztbsv (k, ab, x)
    assert(np.allclose(x, expected, TOL))
    # ...

# ==============================================================================
def test_ztpsv_1():
    from zblas import blas_ztpsv

    n = 10
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)

    # make a triangular
    a = triangulize(a)
    ap = general_to_packed(a)

    # ...
    x.copy()
    expected = sp_blas.ztpsv (n, ap, x)
    blas_ztpsv (ap, x)
    assert(np.linalg.norm(x-expected) < TOL)
    # ...

# ==============================================================================
def test_zgeru_1():
    from zblas import blas_zgeru

    n = 10
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # ...
    alpha = np.complex128(1.)
    expected = sp_blas.zgeru (alpha, x, y, a=a)
    blas_zgeru (alpha, x, y, a)
    assert(np.allclose(a, expected, TOL))
    # ...

# ==============================================================================
def test_zgerc_1():
    from zblas import blas_zgerc

    n = 10
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # ...
    alpha = np.complex128(1.)
    expected = sp_blas.zgerc (alpha, x, y, a=a)
    blas_zgerc (alpha, x, y, a)
    assert(np.allclose(a, expected, TOL))
    # ...

# ==============================================================================
def test_zher_1():
    from zblas import blas_zher

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)

    # syrketrize a
    a = symmetrize(a)

    # ...
    alpha = np.float64(1.)
    expected = sp_blas.zher (alpha, x, a=a)
    blas_zher (alpha, x, a)
    assert(np.allclose(a, expected, TOL))
    # ...

# ==============================================================================
def test_zhpr_1():
    from zblas import blas_zhpr

    n = 10
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)

    # syrketrize a
    a = symmetrize(a)
    ap = general_to_packed(a)

    # ...
    alpha = np.float64(1.)
    expected = sp_blas.zhpr (n, alpha, x, ap)
    blas_zhpr (alpha, x, ap)
    assert(np.allclose(ap, expected, TOL))
    # ...

# ==============================================================================
def test_zher2_1():
    from zblas import blas_zher2

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # syrketrize a
    a = symmetrize(a)

    # ...
    alpha = np.complex128(1.)
    expected = sp_blas.zher2 (alpha, x, y, a=a)
    blas_zher2 (alpha, x, y, a)
    assert(np.allclose(a, expected, TOL))
    # ...

# ==============================================================================
def test_zhpr2_1():
    from zblas import blas_zhpr2

    n = 10
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # syrketrize a
    a = symmetrize(a)
    ap = general_to_packed(a)

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

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    b = random_array((n,n), dtype=DTYPE)
    c = random_array((n,n), dtype=DTYPE)

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

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    b = random_array((n,n), dtype=DTYPE)
    c = random_array((n,n), dtype=DTYPE)

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

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    b = random_array((n,n), dtype=DTYPE)
    c = random_array((n,n), dtype=DTYPE)

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

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    c = random_array((n,n), dtype=DTYPE)

    # syrketrize a
    a = symmetrize(a)

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

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    b = random_array((n,n), dtype=DTYPE)
    c = random_array((n,n), dtype=DTYPE)

    # syr2ketrize a & b
    a = symmetrize(a)
    b = symmetrize(b)

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

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    c = random_array((n,n), dtype=DTYPE)

    # syrketrize a
    a = symmetrize(a)

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

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    b = random_array((n,n), dtype=DTYPE)
    c = random_array((n,n), dtype=DTYPE)

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

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    b = random_array((n,n), dtype=DTYPE)

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

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    b = random_array((n,n), dtype=DTYPE)

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
