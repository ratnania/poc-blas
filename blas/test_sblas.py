import numpy as np
from scipy.sparse import diags
import scipy.linalg.blas as sp_blas
from utilities import symmetrize, triangulize, general_to_band, general_to_packed
from utilities import random_array

#TODO dsdot

TOL = 1.e-7
DTYPE = np.float32

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
    assert(np.allclose(result, expected, TOL))

# ==============================================================================
def test_srot_1():
    from sblas import blas_srot

    n = 10
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    expected_x = x.copy()
    expected_y = y.copy()

    # ...
    one = np.float32(1.)
    c, s = sp_blas.srotg (one, one)
    c = np.float32(c)
    s = np.float32(s)
    expected_x, expected_y = sp_blas.srot(x, y, c, s)
    blas_srot (x, y, c, s)
    assert(np.allclose(x, expected_x, TOL))
    assert(np.allclose(y, expected_y, TOL))
    # ...

# ==============================================================================
def test_srotm_1():
    from sblas import blas_srotm

    n = 10
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)
    expected_x = x.copy()
    expected_y = y.copy()

    # ...
    d1 = d2 = np.float32(1.)
    x1 = y1 = np.float32(.5)
    param = sp_blas.srotmg (d1, d2, x1, y1)
    param = np.array(param, dtype=np.float32)
    expected_x, expected_y = sp_blas.srotm(x, y, param)
    blas_srotm (x, y, param)
    assert(np.allclose(x, expected_x, TOL))
    assert(np.allclose(y, expected_y, TOL))
    # ...

# ==============================================================================
def test_scopy_1():
    from sblas import blas_scopy

    n = 10
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # ...
    expected  = np.zeros(n, dtype=np.float32)
    sp_blas.scopy(x, expected)
    blas_scopy (x, y)
    assert(np.allclose(y, expected, TOL))
    # ...

# ==============================================================================
def test_sswap_1():
    from sblas import blas_sswap

    n = 10
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # ... we swap two times to get back to the original arrays
    expected_x = x.copy()
    expected_y = y.copy()
    sp_blas.sswap (x, y)
    blas_sswap (x, y)
    assert(np.allclose(x, expected_x, TOL))
    assert(np.allclose(y, expected_y, TOL))
    # ...

# ==============================================================================
def test_sscal_1():
    from sblas import blas_sscal

    n = 10
    x = random_array(n, dtype=DTYPE)

    # ... we scale two times to get back to the original arrays
    expected = x.copy()
    alpha = np.float32(2.5)
    sp_blas.sscal (alpha, x)
    blas_sscal (np.float32(1./alpha), x)
    assert(np.allclose(x, expected, TOL))
    # ...

# ==============================================================================
def test_sdot_1():
    from sblas import blas_sdot

    n = 10
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # ...
    expected = sp_blas.sdot(x, y)
    result   = blas_sdot (x, y)
    assert(np.allclose(result, expected, 1.e-6))
    # ...

# ==============================================================================
def test_snrm2_1():
    from sblas import blas_snrm2

    n = 10
    x = random_array(n, dtype=DTYPE)

    # ...
    expected = sp_blas.snrm2(x)
    result   = blas_snrm2 (x)
    assert(np.allclose(result, expected, 1.e-6))
    # ...

# ==============================================================================
def test_sasum_1():
    from sblas import blas_sasum

    n = 10
    x = random_array(n, dtype=DTYPE)

    # ...
    expected = sp_blas.sasum(x)
    result   = blas_sasum (x)
    assert(np.allclose(result, expected, TOL))
    # ...

# ==============================================================================
def test_isamax_1():
    from sblas import blas_isamax

    n = 10
    x = random_array(n, dtype=DTYPE)

    # ...
    expected = sp_blas.isamax(x)
    result   = blas_isamax (x)
    assert(result == expected)
    # ...

# ==============================================================================
def test_saxpy_1():
    from sblas import blas_saxpy

    n = 10
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # ...
    alpha = np.float32(2.5)
    expected = y.copy()
    sp_blas.saxpy (x, expected, a=alpha)
    blas_saxpy (x, y, a=alpha )
    assert(np.allclose(y, expected, TOL))
    # ...

# ==============================================================================
#
#                                  LEVEL 2
#
# ==============================================================================

# ==============================================================================
def test_sgemv_1():
    from sblas import blas_sgemv

    n = 10
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # ...
    alpha = np.float32(1.)
    beta = np.float32(0.5)
    expected = y.copy()
    expected = sp_blas.sgemv (alpha, a, x, beta=beta, y=expected)
    blas_sgemv (alpha, a, x, y, beta=beta)
    assert(np.allclose(y, expected, TOL))
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
                 ], dtype=np.float32)

    ab = general_to_band(kl, ku, a).copy(order='F')

    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # ...
    alpha = np.float32(1.)
    beta = np.float32(0.5)
    expected = y.copy()
    expected = sp_blas.sgbmv (n, n, kl, ku, alpha, ab, x, beta=beta, y=expected)
    blas_sgbmv (kl, ku, alpha, ab, x, y, beta=beta)
    assert(np.allclose(y, expected, TOL))
    # ...

# ==============================================================================
def test_ssymv_1():
    from sblas import blas_ssymv

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # make a symmetric
    a = symmetrize(a)
    # make a triangular
    a = triangulize(a)

    # ...
    alpha = np.float32(1.)
    beta = np.float32(0.5)
    expected = sp_blas.ssymv (alpha, a, x, y=y, beta=beta)
    blas_ssymv (alpha, a, x, y, beta=beta)
    assert(np.allclose(y, expected, TOL))
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
                 ], dtype=np.float32)

    ab = general_to_band(k, k, a).copy(order='F')

    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # ...
    alpha = np.float32(1.)
    beta = np.float32(0.5)
    expected = y.copy()
    expected = sp_blas.ssbmv (k, alpha, ab, x, beta=beta, y=expected)
    blas_ssbmv (k, alpha, ab, x, y, beta=beta)
    assert(np.allclose(y, expected, TOL))
    # ...

# ==============================================================================
def test_sspmv_1():
    from sblas import blas_sspmv

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # make a symmetric
    a = symmetrize(a)
    ap = general_to_packed(a)

    # ...
    alpha = np.float32(1.)
    beta = np.float32(0.5)
    expected = sp_blas.sspmv (n, alpha, ap, x, y=y, beta=beta)
    blas_sspmv (alpha, ap, x, y, beta=beta)
    assert(np.allclose(y, expected, TOL))
    # ...

# ==============================================================================
def test_strmv_1():
    from sblas import blas_strmv

    n = 10
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)

    # make a triangular
    a = triangulize(a)

    # ...
    expected = sp_blas.strmv (a, x)
    blas_strmv (a, x)
    assert(np.allclose(x, expected, TOL))
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
                 ], dtype=np.float32)

    ab = general_to_band(k, k, a).copy(order='F')

    x = random_array(n, dtype=DTYPE)

    # ...
    expected = sp_blas.stbmv (k, ab, x)
    blas_stbmv (k, ab, x)
    assert(np.allclose(x, expected, TOL))
    # ...

# ==============================================================================
def test_stpmv_1():
    from sblas import blas_stpmv

    n = 10
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)

    # make a triangular
    a = triangulize(a)
    ap = general_to_packed(a)

    # ...
    expected = sp_blas.stpmv (n, ap, x)
    blas_stpmv (ap, x)
    assert(np.allclose(x, expected, TOL))
    # ...

# ==============================================================================
def test_strsv_1():
    from sblas import blas_strsv

    n = 10
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)

    # make a triangular
    a = triangulize(a)

    # ...
    b = x.copy()
    expected = sp_blas.strsv (a, b)
    blas_strsv (a, x)
    assert(np.allclose(x, expected, TOL))
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
                 ], dtype=np.float32)

    ab = general_to_band(k, k, a).copy(order='F')

    x = random_array(n, dtype=DTYPE)

    # ...
    expected = sp_blas.stbsv (k, ab, x)
    blas_stbsv (k, ab, x)
    assert(np.allclose(x, expected, TOL))
    # ...

# ==============================================================================
def test_stpsv_1():
    from sblas import blas_stpsv

    n = 10
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # make a triangular
    a = triangulize(a)
    ap = general_to_packed(a)

    # ...
    b = x.copy()
    expected = sp_blas.stpsv (n, ap, b)
    blas_stpsv (ap, x)
    assert(np.allclose(x, expected, TOL))
    # ...

# ==============================================================================
def test_sger_1():
    from sblas import blas_sger

    n = 10
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # ...
    alpha = np.float32(1.)
    expected = sp_blas.sger (alpha, x, y, a=a)
    blas_sger (alpha, x, y, a)
    assert(np.allclose(a, expected, TOL))
    # ...

# ==============================================================================
def test_ssyr_1():
    from sblas import blas_ssyr

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)

    # syrketrize a
    a = symmetrize(a)

    # ...
    alpha = np.float32(1.)
    expected = sp_blas.ssyr (alpha, x, a=a)
    blas_ssyr (alpha, x, a)
    assert(np.allclose(a, expected, TOL))
    # ...

# ==============================================================================
def test_sspr_1():
    from sblas import blas_sspr

    n = 10
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)

    # syrketrize a
    a = symmetrize(a)
    ap = general_to_packed(a)

    # ...
    alpha = np.float32(1.)
    expected = sp_blas.sspr (n, alpha, x, ap)
    blas_sspr (alpha, x, ap)
    assert(np.allclose(ap, expected, TOL))
    # ...

# ==============================================================================
def test_ssyr2_1():
    from sblas import blas_ssyr2

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # syrketrize a
    a = symmetrize(a)

    # ...
    alpha = np.float32(1.)
    expected = sp_blas.ssyr2 (alpha, x, y, a=a)
    blas_ssyr2 (alpha, x, y, a=a)
    assert(np.allclose(a, expected, TOL))
    # ...

# ==============================================================================
def test_sspr2_1():
    from sblas import blas_sspr2

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    x = random_array(n, dtype=DTYPE)
    y = random_array(n, dtype=DTYPE)

    # syrketrize a
    a = symmetrize(a)
    ap = general_to_packed(a)

    # ...
    alpha = np.float32(1.)
    expected = sp_blas.sspr2 (n, alpha, x, y, ap)
    blas_sspr2 (alpha, x, y, ap)
    assert(np.allclose(ap, expected, TOL))
    # ...

# ==============================================================================
#
#                                  LEVEL 3
#
# ==============================================================================

# ==============================================================================
def test_sgemm_1():
    from sblas import blas_sgemm

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    b = random_array((n,n), dtype=DTYPE)
    c = random_array((n,n), dtype=DTYPE)

    # ...
    alpha = np.float32(1.)
    beta = np.float32(0.5)
    expected = sp_blas.sgemm (alpha, a, b, c=c, beta=beta)
    blas_sgemm (alpha, a, b, c, beta=beta)
    assert(np.allclose(c, expected, TOL))
    # ...

# ==============================================================================
def test_ssymm_1():
    from sblas import blas_ssymm

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    b = random_array((n,n), dtype=DTYPE)
    c = random_array((n,n), dtype=DTYPE)

    # symmetrize a & b
    a = symmetrize(a)
    b = symmetrize(b)

    # ...
    alpha = np.float32(1.)
    beta = np.float32(0.5)
    expected = sp_blas.ssymm (alpha, a, b, c=c, beta=beta)
    blas_ssymm (alpha, a, b, c, beta=beta)
    assert(np.allclose(c, expected, TOL))
    # ...

# ==============================================================================
def test_strmm_1():
    from sblas import blas_strmm

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    b = random_array((n,n), dtype=DTYPE)

    # make a triangular
    a = triangulize(a)

    # ...
    alpha = np.float32(1.)
    expected = sp_blas.strmm (alpha, a, b)
    blas_strmm (alpha, a, b)
    assert(np.allclose(b, expected, TOL))
    # ...

# ==============================================================================
def test_strsm_1():
    from sblas import blas_strsm

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    b = random_array((n,n), dtype=DTYPE)

    # make a triangular
    a = triangulize(a)

    # ...
    alpha = np.float32(1.)
    expected = sp_blas.strsm (alpha, a, b)
    blas_strsm (alpha, a, b)
    assert(np.allclose(b, expected, TOL))
    # ...

# ==============================================================================
def test_ssyrk_1():
    from sblas import blas_ssyrk

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    c = random_array((n,n), dtype=DTYPE)

    # syrketrize a
    a = symmetrize(a)

    # ...
    alpha = np.float32(1.)
    beta = np.float32(0.)
    expected = sp_blas.ssyrk (alpha, a, c=c, beta=beta)
    blas_ssyrk (alpha, a, c, beta=beta)
    assert(np.allclose(c, expected, TOL))
    # ...

# ==============================================================================
def test_ssyr2k_1():
    from sblas import blas_ssyr2k

    n = 4
    a = random_array((n,n), dtype=DTYPE)
    b = random_array((n,n), dtype=DTYPE)
    c = random_array((n,n), dtype=DTYPE)

    # syr2ketrize a & b
    a = symmetrize(a)
    b = symmetrize(b)

    # ...
    alpha = np.float32(1.)
    beta = np.float32(0.)
    expected = sp_blas.ssyr2k (alpha, a, b, c=c, beta=beta)
    blas_ssyr2k (alpha, a, b, c, beta=beta)
    assert(np.allclose(c, expected, TOL))
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
    test_ssymm_1()
    test_strmm_1()
    test_strsm_1()
    test_ssyrk_1()
    test_ssyr2k_1()
    # ...
