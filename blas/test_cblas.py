import numpy as np
from scipy.sparse import diags
import scipy.linalg.blas as sp_blas
from utilities import symmetrize, triangulize, general_to_band, general_to_packed

TOL = 1.e-7

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
    assert(np.allclose(y, expected, TOL))
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
    assert(np.allclose(x, expected_x, TOL))
    assert(np.allclose(y, expected_y, TOL))
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
    assert(np.allclose(x, expected, TOL))
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
def test_scasum_1():
    from cblas import blas_scasum

    np.random.seed(2021)

    n = 10
    x = np.random.random(n) + np.random.random(n) * 1j
    x = np.array(x, dtype=np.complex64)

    # ...
    expected = sp_blas.scasum(x)
    result   = blas_scasum (x)
    assert(np.allclose(result, expected, 1.e-6))
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
    assert(np.allclose(y, expected, TOL))
    # ...

# ==============================================================================
def test_cdotc_1():
    from cblas import blas_cdotc

    np.random.seed(2021)

    n = 3
    x = np.random.random(n) + np.random.random(n) * 1j
    y = np.random.random(n) + np.random.random(n) * 1j
    x = np.array(x, dtype=np.complex64)
    y = np.array(y, dtype=np.complex64)

    # ...
    expected = sp_blas.cdotc(x, y)
    result   = blas_cdotc (x, y)
    assert(np.linalg.norm(result-expected) < 1.e-6)
    # ...

# ==============================================================================
def test_cdotu_1():
    from cblas import blas_cdotu

    np.random.seed(2021)

    n = 10
    x = np.random.random(n) + np.random.random(n) * 1j
    y = np.random.random(n) + np.random.random(n) * 1j
    x = np.array(x, dtype=np.complex64)
    y = np.array(y, dtype=np.complex64)

    # ...
    expected = sp_blas.cdotu(x, y)
    result   = blas_cdotu (x, y)
    assert(np.linalg.norm(result-expected) < 1.e-6)
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
    assert(np.allclose(y, expected, TOL))
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
    assert(np.allclose(y, expected, TOL))
    # ...

# ==============================================================================
def test_chemv_1():
    from cblas import blas_chemv

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
    expected = sp_blas.chemv (alpha, a, x, beta=beta, y=expected)
    blas_chemv (alpha, a, x, y, beta=beta)
    assert(np.allclose(y, expected, TOL))
    # ...

# ==============================================================================
def test_chbmv_1():
    from cblas import blas_chbmv

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

    ab = np.array(ab, dtype=np.complex64)
    x = np.array(x, dtype=np.complex64)
    y = np.array(y, dtype=np.complex64)

    # ...
    alpha = np.complex64(1.0)
    beta = np.complex64(0.5)
    expected = y.copy()
    expected = sp_blas.chbmv (k, alpha, ab, x, beta=beta, y=expected)
    blas_chbmv (k, alpha, ab, x, y, beta=beta)
    assert(np.allclose(y, expected, TOL))
    # ...

# ==============================================================================
def test_chpmv_1():
    from cblas import blas_chpmv

    np.random.seed(2021)

    n = 4
    a = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    a = a.copy(order='F')
    x = np.random.random(n) + np.random.random(n) * 1j
    y = np.random.random(n) + np.random.random(n) * 1j

    # make a symmetric
    a = symmetrize(a)
    ap = general_to_packed(a)

    a = np.array(a, dtype=np.complex64)
    ap = np.array(ap, dtype=np.complex64)
    x = np.array(x, dtype=np.complex64)
    y = np.array(y, dtype=np.complex64)

    # ...
    alpha = np.complex64(1.)
    beta = np.complex64(0.5)
    expected = sp_blas.chpmv (n, alpha, ap, x, y=y, beta=beta)
    blas_chpmv (alpha, ap, x, y, beta=beta)
    assert(np.allclose(y, expected, TOL))
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
    expected = sp_blas.ctrmv (a, x)
    blas_ctrmv (a, x)
    assert(np.allclose(x, expected, TOL))
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
    expected = sp_blas.ctbmv (k, ab, x)
    blas_ctbmv (k, ab, x)
    assert(np.allclose(x, expected, TOL))
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
    expected = sp_blas.ctpmv (n, ap, x)
    blas_ctpmv (ap, x)
    assert(np.allclose(x, expected, TOL))
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
    b = x.copy()
    expected = sp_blas.ctrsv (a, b)
    blas_ctrsv (a, x)
    assert(np.linalg.norm(x-expected) < TOL)
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
    assert(np.allclose(x, expected, TOL))
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
    b = x.copy()
    expected = sp_blas.ctpsv (n, ap, b)
    blas_ctpsv (ap, x)
    assert(np.linalg.norm(x-expected) < TOL)
    # ...

# ==============================================================================
def test_cgeru_1():
    from cblas import blas_cgeru

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
    alpha = np.complex64(1.)
    expected = sp_blas.cgeru (alpha, x, y, a=a)
    blas_cgeru (alpha, x, y, a)
    assert(np.allclose(a, expected, TOL))
    # ...

# ==============================================================================
def test_cgerc_1():
    from cblas import blas_cgerc

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
    alpha = np.complex64(1.)
    expected = sp_blas.cgerc (alpha, x, y, a=a)
    blas_cgerc (alpha, x, y, a)
    assert(np.allclose(a, expected, TOL))
    # ...

# ==============================================================================
def test_cher_1():
    from cblas import blas_cher

    np.random.seed(2021)

    n = 4
    a = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    a = a.copy(order='F')
    x = np.random.random(n) + np.random.random(n) * 1j

    # syrketrize a
    a = symmetrize(a)

    a = np.array(a, dtype=np.complex64)
    x = np.array(x, dtype=np.complex64)

    # ...
    alpha = np.float32(1.)
    expected = sp_blas.cher (alpha, x, a=a)
    blas_cher (alpha, x, a)
    assert(np.allclose(a, expected, TOL))
    # ...

# ==============================================================================
def test_chpr_1():
    from cblas import blas_chpr

    np.random.seed(2021)

    n = 10
    a = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    a = a.copy(order='F')
    x = np.random.random(n) + np.random.random(n) * 1j

    # syrketrize a
    a = symmetrize(a)
    ap = general_to_packed(a)

    ap = np.array(ap, dtype=np.complex64)
    a = np.array(a, dtype=np.complex64)
    x = np.array(x, dtype=np.complex64)

    # ...
    alpha = np.float32(1.)
    expected = sp_blas.chpr (n, alpha, x, ap)
    blas_chpr (alpha, x, ap)
    assert(np.allclose(ap, expected, TOL))
    # ...

# ==============================================================================
def test_cher2_1():
    from cblas import blas_cher2

    np.random.seed(2021)

    n = 4
    a = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    a = a.copy(order='F')
    x = np.random.random(n) + np.random.random(n) * 1j
    y = np.random.random(n) + np.random.random(n) * 1j

    # syrketrize a
    a = symmetrize(a)

    a = np.array(a, dtype=np.complex64)
    x = np.array(x, dtype=np.complex64)
    y = np.array(y, dtype=np.complex64)

    # ...
    alpha = np.complex64(1.)
    expected = sp_blas.cher2 (alpha, x, y, a=a)
    blas_cher2 (alpha, x, y, a)
    assert(np.allclose(a, expected, TOL))
    # ...

# ==============================================================================
def test_chpr2_1():
    from cblas import blas_chpr2

    np.random.seed(2021)

    n = 10
    a = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    a = a.copy(order='F')
    x = np.random.random(n) + np.random.random(n) * 1j
    y = np.random.random(n) + np.random.random(n) * 1j

    # syrketrize a
    a = symmetrize(a)
    ap = general_to_packed(a)

    ap = np.array(ap, dtype=np.complex64)
    a = np.array(a, dtype=np.complex64)
    x = np.array(x, dtype=np.complex64)
    y = np.array(x, dtype=np.complex64)

    # ...
    alpha = np.complex64(1.)
    expected = sp_blas.chpr2 (n, alpha, x, y, ap)
    blas_chpr2 (alpha, x, y, ap)
    assert(np.allclose(ap, expected, TOL))
    # ...

# ==============================================================================
#
#                                  LEVEL 3
#
# ==============================================================================

# ==============================================================================
def test_cgemm_1():
    from cblas import blas_cgemm

    np.random.seed(2021)

    n = 4
    a = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    b = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    c = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    a = a.copy(order='F')
    b = b.copy(order='F')
    c = c.copy(order='F')

    a = np.array(a, dtype=np.complex64)
    b = np.array(b, dtype=np.complex64)
    c = np.array(c, dtype=np.complex64)

    # ...
    alpha = np.complex64(1.)
    beta = np.complex64(.5)
    expected = sp_blas.cgemm (alpha, a, b, c=c, beta=beta)
    blas_cgemm (alpha, a, b, c, beta=beta)
    assert(np.allclose(c, expected, TOL))
    # ...

# ==============================================================================
def test_csymm_1():
    from cblas import blas_csymm

    np.random.seed(2021)

    n = 4
    a = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    b = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    c = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    a = a.copy(order='F')
    b = b.copy(order='F')
    c = c.copy(order='F')

    a = np.array(a, dtype=np.complex64)
    b = np.array(b, dtype=np.complex64)
    c = np.array(c, dtype=np.complex64)

    # symmetrize a & b
    a = symmetrize(a)
    b = symmetrize(b)

    # ...
    alpha = np.complex64(1.)
    beta = np.complex64(.5)
    expected = sp_blas.csymm (alpha, a, b, c=c, beta=beta)
    blas_csymm (alpha, a, b, c, beta=beta)
    assert(np.allclose(c, expected, TOL))
    # ...

# ==============================================================================
def test_chemm_1():
    from cblas import blas_chemm

    np.random.seed(2021)

    n = 4
    a = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    b = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    c = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    a = a.copy(order='F')
    b = b.copy(order='F')
    c = c.copy(order='F')

    a = np.array(a, dtype=np.complex64)
    b = np.array(b, dtype=np.complex64)
    c = np.array(c, dtype=np.complex64)

    # symmetrize a & b
    a = symmetrize(a)
    b = symmetrize(b)

    # ...
    alpha = np.complex64(1.)
    beta = np.complex64(.5)
    expected = sp_blas.chemm (alpha, a, b, beta=beta, c=c)
    blas_chemm (alpha, a, b, c, beta=beta)
    assert(np.allclose(c, expected, TOL))
    # ...

# ==============================================================================
def test_csyrk_1():
    from cblas import blas_csyrk

    np.random.seed(2021)

    n = 4
    a = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    c = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    a = a.copy(order='F')
    c = c.copy(order='F')

    a = np.array(a, dtype=np.complex64)
    c = np.array(c, dtype=np.complex64)

    # syrketrize a
    a = symmetrize(a)
    a_T = a.T.copy(order='F')

    # ...
    alpha = np.complex64(1.)
    beta = np.complex64(.5)
    expected = sp_blas.csyrk (alpha, a, beta=beta, c=c)
    blas_csyrk (alpha, a, c, beta=beta)
    assert(np.linalg.norm(c- expected) < TOL)
    # ...

# ==============================================================================
def test_csyr2k_1():
    from cblas import blas_csyr2k

    np.random.seed(2021)

    n = 4
    a = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    b = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    c = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    a = a.copy(order='F')
    b = b.copy(order='F')
    c = c.copy(order='F')

    a = np.array(a, dtype=np.complex64)
    b = np.array(b, dtype=np.complex64)
    c = np.array(c, dtype=np.complex64)

    # syr2ketrize a & b
    a = symmetrize(a)
    b = symmetrize(b)
    a_T = a.T.copy(order='F')
    b_T = b.T.copy(order='F')

    # ...
    alpha = np.complex64(1.)
    beta = np.complex64(.5)
    expected = sp_blas.csyr2k (alpha, a, b, beta=beta, c=c)
    blas_csyr2k (alpha, a, b, c, beta=beta)
    assert(np.allclose(c, expected, TOL))
    # ...

# ==============================================================================
def test_cherk_1():
    from cblas import blas_cherk

    np.random.seed(2021)

    n = 4
    a = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    c = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    a = a.copy(order='F')
    c = c.copy(order='F')

    a = np.array(a, dtype=np.complex64)
    c = np.array(c, dtype=np.complex64)

    # syrketrize a
    a = symmetrize(a)

    # ...
    alpha = np.complex64(1.)
    beta = np.complex64(.5)
    expected = sp_blas.cherk (alpha, a, beta=beta, c=c)
    blas_cherk (alpha, a, c, beta=beta)
    assert(np.linalg.norm(c- expected) < TOL)
    # ...

# ==============================================================================
def test_cher2k_1():
    from cblas import blas_cher2k

    np.random.seed(2021)

    n = 4
    a = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    b = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    c = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    a = a.copy(order='F')
    b = b.copy(order='F')
    c = c.copy(order='F')

    a = np.array(a, dtype=np.complex64)
    b = np.array(b, dtype=np.complex64)
    c = np.array(c, dtype=np.complex64)

    # syr2ketrize a & b
    a = symmetrize(a)
    b = symmetrize(b)
    a_T = a.T.copy(order='F')
    b_T = b.T.copy(order='F')

    # ...
    alpha = np.complex64(1.)
    beta = np.complex64(.5)
    expected = sp_blas.cher2k (alpha, a, b, beta=beta, c=c)
    blas_cher2k (alpha, a, b, c, beta=beta)
    assert(np.allclose(c, expected, TOL))
    # ...

# ==============================================================================
def test_ctrmm_1():
    from cblas import blas_ctrmm

    np.random.seed(2021)

    n = 4
    a = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    b = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    a = a.copy(order='F')
    b = b.copy(order='F')

    a = np.array(a, dtype=np.complex64)
    b = np.array(b, dtype=np.complex64)

    # make a triangular
    a = triangulize(a)

    # ...
    alpha = np.complex64(1.)
    expected = sp_blas.ctrmm (alpha, a, b)
    blas_ctrmm (alpha, a, b)
    assert(np.allclose(b, expected, TOL))
    # ...

# ==============================================================================
def test_ctrsm_1():
    from cblas import blas_ctrsm

    np.random.seed(2021)

    n = 4
    a = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    b = np.random.random((n,n)) + np.random.random((n,n)) * 1j
    a = a.copy(order='F')
    b = b.copy(order='F')

    a = np.array(a, dtype=np.complex64)
    b = np.array(b, dtype=np.complex64)

    # make a triangular
    a = triangulize(a)

    # ...
    alpha = np.complex64(1.)
    expected = sp_blas.ctrsm (alpha, a, b)
    blas_ctrsm (alpha, a, b)
    assert(np.linalg.norm(b- expected) < TOL)
    # ...

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
    test_cdotc_1()
    test_cdotu_1()
    # ...

    # ... LEVEL 2
    test_cgemv_1()
    test_cgbmv_1()
    test_chemv_1()
    test_chbmv_1()
    test_chpmv_1()
    test_ctrmv_1()
    test_ctbmv_1()
    test_ctpmv_1()
    test_ctrsv_1()
    test_ctbsv_1()
    test_ctpsv_1()
    test_cgeru_1()
    test_cgerc_1()
    test_cher_1()
    test_chpr_1()
    test_cher2_1()
    test_chpr2_1()
    # ...

    # ... LEVEL 3
    test_cgemm_1()
    test_csymm_1()
    test_chemm_1()
    test_csyrk_1()
    test_csyr2k_1()
    test_cherk_1()
    test_cher2k_1()
    test_ctrmm_1()
    test_ctrsm_1()
    # ...