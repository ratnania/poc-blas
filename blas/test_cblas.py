import numpy as np
from scipy.sparse import diags
import scipy.linalg.blas as sp_blas
from utilities import symmetrize, triangulize, general_to_band, general_to_packed
from utilities import random_array

# ==============================================================================
#
#                                  LEVEL 1
#
# ==============================================================================

# ==============================================================================
def test_ccopy_1():
    from cblas import blas_ccopy

    TOL = 1.e-7
    DTYPE = np.complex64

    n = 3
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # ...
    expected = y.copy()
    sp_blas.ccopy(x, expected)
    blas_ccopy (x, y)
    assert(np.allclose(y, expected, TOL))
    # ...

# ==============================================================================
def test_cswap_1():
    from cblas import blas_cswap

    TOL = 1.e-7
    DTYPE = np.complex64

    n = 10
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

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

    TOL = 1.e-7
    DTYPE = np.complex64

    n = 10
    x = random_array(n, dtype=DTYPE)

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

    TOL = 1.e-7
    DTYPE = np.complex64

    n = 10
    x = random_array(n, dtype=DTYPE)

    # ...
    expected = sp_blas.scnrm2(x)
    result   = blas_scnrm2 (x)
    assert(np.allclose(result, expected, 1.e-6))
    # ...

# ==============================================================================
def test_scasum_1():
    from cblas import blas_scasum

    TOL = 1.e-7
    DTYPE = np.complex64

    n = 10
    x = random_array(n, dtype=DTYPE)

    # ...
    expected = sp_blas.scasum(x)
    result   = blas_scasum (x)
    assert(np.allclose(result, expected, 1.e-6))
    # ...

# ==============================================================================
def test_icamax_1():
    from cblas import blas_icamax

    TOL = 1.e-7
    DTYPE = np.complex64

    n = 10
    x = random_array(n, dtype=DTYPE)

    # ...
    expected = sp_blas.icamax(x)
    result   = blas_icamax (x)
    assert(result == expected)
    # ...

# ==============================================================================
def test_caxpy_1():
    from cblas import blas_caxpy

    TOL = 1.e-7
    DTYPE = np.complex64

    n = 10
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

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

    TOL = 1.e-7
    DTYPE = np.complex64

    n = 3
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # ...
    expected = sp_blas.cdotc(x, y)
    result   = blas_cdotc (x, y)
    assert(np.linalg.norm(result-expected) < 1.e-6)
    # ...

# ==============================================================================
def test_cdotu_1():
    from cblas import blas_cdotu

    TOL = 1.e-7
    DTYPE = np.complex64

    n = 10
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

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

    TOL = 1.e-7
    DTYPE = np.complex64

    n = 10
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

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

    TOL = 1.e-7
    DTYPE = np.complex64

    n = 5
    kl = np.int32(2)
    ku = np.int32(2)
    a = np.array([[11, 12,  0,  0,  0],
                  [21, 22, 23,  0,  0],
                  [31, 32, 33, 34,  0],
                  [ 0, 42, 43, 44, 45],
                  [ 0,  0, 53, 54, 55]
                 ], dtype=np.complex64)

    ab = general_to_band(kl, ku, a).copy(order='F')

    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

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

    TOL = 1.e-7
    DTYPE = np.complex64

    n = 10
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

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

    TOL = 1.e-7
    DTYPE = np.complex64

    n = 5
    k = np.int32(2)
    a = np.array([[11, 12,  0,  0,  0],
                  [21, 22, 23,  0,  0],
                  [31, 32, 33, 34,  0],
                  [ 0, 42, 43, 44, 45],
                  [ 0,  0, 53, 54, 55]
                 ], dtype=np.complex64)

    ab = general_to_band(k, k, a).copy(order='F')

    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

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

    TOL = 1.e-7
    DTYPE = np.complex64

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # make a symmetric
    a = symmetrize(a)
    ap = general_to_packed(a)

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

    TOL = 1.e-7
    DTYPE = np.complex64

    n = 10
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)

    # make a triangular
    a = triangulize(a)

    # ...
    expected = sp_blas.ctrmv (a, x)
    blas_ctrmv (a, x)
    assert(np.allclose(x, expected, TOL))
    # ...

# ==============================================================================
def test_ctbmv_1():
    from cblas import blas_ctbmv

    TOL = 1.e-7
    DTYPE = np.complex64

    n = 5
    k = np.int32(2)
    a = np.array([[11, 12,  0,  0,  0],
                  [12, 22, 23,  0,  0],
                  [ 0, 32, 33, 34,  0],
                  [ 0,  0, 34, 44, 45],
                  [ 0,  0,  0, 45, 55]
                 ], dtype=np.complex64)

    ab = general_to_band(k, k, a).copy(order='F')

    x = random_array(n, dtype=DTYPE)

    # make a triangular
    a = triangulize(a)

    # ...
    expected = sp_blas.ctbmv (k, ab, x)
    blas_ctbmv (k, ab, x)
    assert(np.allclose(x, expected, TOL))
    # ...

# ==============================================================================
def test_ctpmv_1():
    from cblas import blas_ctpmv

    TOL = 1.e-7
    DTYPE = np.complex64

    n = 10
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)

    # make a triangular
    a = triangulize(a)
    ap = general_to_packed(a)

    # ...
    expected = sp_blas.ctpmv (n, ap, x)
    blas_ctpmv (ap, x)
    assert(np.allclose(x, expected, TOL))
    # ...

# ==============================================================================
def test_ctrsv_1():
    from cblas import blas_ctrsv

    TOL = 1.e-7
    DTYPE = np.complex64

    n = 10
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)

    # make a triangular
    a = triangulize(a)

    # ...
    b = x.copy()
    expected = sp_blas.ctrsv (a, b)
    blas_ctrsv (a, x)
    assert(np.linalg.norm(x-expected) < TOL)
    # ...

# ==============================================================================
def test_ctbsv_1():
    from cblas import blas_ctbsv

    TOL = 1.e-7
    DTYPE = np.complex64

    n = 5
    k = np.int32(2)
    a = np.array([[11, 12,  0,  0,  0],
                  [12, 22, 23,  0,  0],
                  [ 0, 32, 33, 34,  0],
                  [ 0,  0, 34, 44, 45],
                  [ 0,  0,  0, 45, 55]
                 ], dtype=np.complex64)

    ab = general_to_band(k, k, a).copy(order='F')

    x = random_array(n, dtype=DTYPE)

    # ...
    expected = sp_blas.ctbsv (k, ab, x)
    blas_ctbsv (k, ab, x)
    assert(np.allclose(x, expected, TOL))
    # ...

# ==============================================================================
def test_ctpsv_1():
    from cblas import blas_ctpsv

    TOL = 1.e-7
    DTYPE = np.complex64

    n = 10
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)

    # make a triangular
    a = triangulize(a)
    ap = general_to_packed(a)

    # ...
    b = x.copy()
    expected = sp_blas.ctpsv (n, ap, b)
    blas_ctpsv (ap, x)
    assert(np.linalg.norm(x-expected) < TOL)
    # ...

# ==============================================================================
def test_cgeru_1():
    from cblas import blas_cgeru

    TOL = 1.e-7
    DTYPE = np.complex64

    n = 10
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # ...
    alpha = np.complex64(1.)
    expected = sp_blas.cgeru (alpha, x, y, a=a)
    blas_cgeru (alpha, x, y, a)
    assert(np.allclose(a, expected, TOL))
    # ...

# ==============================================================================
def test_cgerc_1():
    from cblas import blas_cgerc

    TOL = 1.e-7
    DTYPE = np.complex64

    n = 10
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # ...
    alpha = np.complex64(1.)
    expected = sp_blas.cgerc (alpha, x, y, a=a)
    blas_cgerc (alpha, x, y, a)
    assert(np.allclose(a, expected, TOL))
    # ...

# ==============================================================================
def test_cher_1():
    from cblas import blas_cher

    TOL = 1.e-7
    DTYPE = np.complex64

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)

    # syrketrize a
    a = symmetrize(a)

    # ...
    alpha = np.float32(1.)
    expected = sp_blas.cher (alpha, x, a=a)
    blas_cher (alpha, x, a)
    assert(np.allclose(a, expected, TOL))
    # ...

# ==============================================================================
def test_chpr_1():
    from cblas import blas_chpr

    TOL = 1.e-7
    DTYPE = np.complex64

    n = 10
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)

    # syrketrize a
    a = symmetrize(a)
    ap = general_to_packed(a)

    # ...
    alpha = np.float32(1.)
    expected = sp_blas.chpr (n, alpha, x, ap)
    blas_chpr (alpha, x, ap)
    assert(np.allclose(ap, expected, TOL))
    # ...

# ==============================================================================
def test_cher2_1():
    from cblas import blas_cher2

    TOL = 1.e-7
    DTYPE = np.complex64

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # syrketrize a
    a = symmetrize(a)

    # ...
    alpha = np.complex64(1.)
    expected = sp_blas.cher2 (alpha, x, y, a=a)
    blas_cher2 (alpha, x, y, a)
    assert(np.allclose(a, expected, TOL))
    # ...

# ==============================================================================
def test_chpr2_1():
    from cblas import blas_chpr2

    TOL = 1.e-7
    DTYPE = np.complex64

    n = 10
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # syrketrize a
    a = symmetrize(a)
    ap = general_to_packed(a)

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

    TOL = 1.e-7
    DTYPE = np.complex64

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    b = random_array((n,n), dtype=DTYPE)
    c = random_array((n,n), dtype=DTYPE)

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

    TOL = 1.e-7
    DTYPE = np.complex64

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    b = random_array((n,n), dtype=DTYPE)
    c = random_array((n,n), dtype=DTYPE)

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

    TOL = 1.e-7
    DTYPE = np.complex64

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    b = random_array((n,n), dtype=DTYPE)
    c = random_array((n,n), dtype=DTYPE)

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

    TOL = 1.e-7
    DTYPE = np.complex64

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    c = random_array((n,n), dtype=DTYPE)

    # syrketrize a
    a = symmetrize(a)

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

    TOL = 1.e-7
    DTYPE = np.complex64

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    b = random_array((n,n), dtype=DTYPE)
    c = random_array((n,n), dtype=DTYPE)

    # syr2ketrize a & b
    a = symmetrize(a)
    b = symmetrize(b)

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

    TOL = 1.e-7
    DTYPE = np.complex64

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    c = random_array((n,n), dtype=DTYPE)

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

    TOL = 1.e-7
    DTYPE = np.complex64

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    b = random_array((n,n), dtype=DTYPE)
    c = random_array((n,n), dtype=DTYPE)

    # syr2ketrize a & b
    a = symmetrize(a)
    b = symmetrize(b)

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

    TOL = 1.e-7
    DTYPE = np.complex64

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    b = random_array((n,n), dtype=DTYPE)

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

    TOL = 1.e-7
    DTYPE = np.complex64

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    b = random_array((n,n), dtype=DTYPE)

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
