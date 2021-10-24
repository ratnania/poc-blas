import numpy as np
from scipy.sparse import diags
import scipy.linalg.blas as sp_blas

#dsdot

# ==============================================================================
def symmetrize(a):
    n = a.shape[0]
    for i in range(n):
        for j in range(i):
            a[i,j] = a[j,i]

    return a

# ==============================================================================
def general_to_band(kl, ku, a):
    n = a.shape[1]
    ab = np.zeros((kl+ku+1, n))

    nsub = 1
    nsuper = 2
    ndiag = nsub + 1 + nsuper
    for icol in range(n):
        i1 = max (1, icol - nsuper)
        i2 = min (n, icol + nsub)
        for irow in range(i1, i2+1):
            # we substruct 1 for Python indexing
            irowb = irow - icol + ndiag - 1
            ab[irowb,icol] = a[irow-1,icol]

    return ab

# ==============================================================================
#
#                                  LEVEL 1
#
# ==============================================================================

# ==============================================================================
def test_drotg_1():
    from blas import blas_drotg

    a = b = 1.
    c, s = blas_drotg (a, b)
    expected_c, expected_s = sp_blas.drotg (a, b)
    assert(np.abs(c - expected_c) < 1.e-10)
    assert(np.abs(s - expected_s) < 1.e-10)

# ==============================================================================
def test_drotmg_1():
    from blas import blas_drotmg

    d1 = d2 = 1.
    x1 = y1 = .5
    result = np.zeros(5)
    blas_drotmg (d1, d2, x1, y1, result)
    expected = sp_blas.drotmg (d1, d2, x1, y1)
    assert(np.allclose(result, expected, 1.e-14))

# ==============================================================================
def test_drot_1():
    from blas import blas_drot

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
    from blas import blas_drotm

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
    from blas import blas_dcopy

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
    from blas import blas_dswap

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
    from blas import blas_dscal

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
    from blas import blas_ddot

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
    from blas import blas_dnrm2

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
    from blas import blas_dasum

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
    from blas import blas_idamax

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
    from blas import blas_daxpy

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
    from blas import blas_dgemv

    n = 10
    a = np.ones((n,n))
    x = np.ones(n)
    y = np.zeros(n)

    # ...
    alpha = 1.
    beta = 0.
    expected = y.copy()
    expected = sp_blas.dgemv (alpha, a, x, beta=beta, y=expected)
    blas_dgemv (alpha, a, x, y, beta=beta)
    assert(np.allclose(y, expected, 1.e-14))
    # ...

# ==============================================================================
def test_dgemv_2():
    from blas import blas_dgemv

    n = 10
    a = np.ones((n,n))
    x = np.ones(n)
    y = np.ones(n)

    # ...
    alpha = 1.
    beta = 1.
    expected = y.copy()
    expected = sp_blas.dgemv (alpha, a, x, beta=beta, y=expected)
    blas_dgemv (alpha, a, x, y, beta=beta)
    assert(np.allclose(y, expected, 1.e-14))
    # ...

# ==============================================================================
# TODO not working yet
def test_dgbmv_1():
    from blas import blas_dgbmv

    n = 8
    ag = diags([1, -2, 1], [-1, 0, 1], shape=(n, n)).toarray()
    ku = 2 ; kl = 2
    a = general_to_band(kl, ku, ag)

    np.random.seed(2021)

    x = np.random.random(n)
    y = np.zeros(n)

    # ...
    alpha = 1.
    beta = 0.
    expected = alpha * ag @ x
    blas_dgbmv (kl, ku, alpha, a, x, y, beta=beta)
#    print('> expected = ', expected)
#    print('> y        = ', y)
#    assert(np.allclose(y, expected, 1.e-14))
    # ...

# ==============================================================================
def test_dsymv_1():
    from blas import blas_dsymv

    n = 10
    a = np.zeros((n,n))
    x = np.ones(n)
    y = np.zeros(n)

    # ...
    alpha = 1.
    beta = 0.
    expected = alpha * a @ x
    blas_dsymv (alpha, a, x, y, beta=beta)
    assert(np.allclose(y, expected, 1.e-14))
    # ...

# ==============================================================================
# TODO not working yet
def test_dsbmv_1():
    from blas import blas_dsbmv

    n = 8
    ag = diags([1, -2, 1], [-1, 0, 1], shape=(n, n)).toarray()
    k = 2
    a = general_to_band(k, k, ag)

    np.random.seed(2021)

    x = np.random.random(n)
    y = np.zeros(n)

    # ...
    alpha = 1.
    beta = 0.
    expected = alpha * ag @ x
    blas_dsbmv (k, alpha, a, x, y, beta=beta)
#    print('> expected = ', expected)
#    print('> y        = ', y)
#    assert(np.allclose(y, expected, 1.e-14))
    # ...

# ==============================================================================
def test_dger_1():
    from blas import blas_dger

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
#
#                                  LEVEL 3
#
# ==============================================================================

# ==============================================================================
def test_dgemm_1():
    from blas import blas_dgemm

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
    from blas import blas_dgemm

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
    from blas import blas_dsymm

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
def test_dsyrk_1():
    from blas import blas_dsyrk

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
    from blas import blas_dsyr2k

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
    test_dgemv_2()
    test_dgbmv_1()
    test_dsymv_1()
    test_dsbmv_1()
    test_dger_1()
    # ...

    # ... LEVEL 3
    test_dgemm_1()
    test_dgemm_2()
    test_dsymm_1()
    test_dsyrk_1()
    test_dsyr2k_1()
    # ...
