"""
The aim of this file is to provide functions that can be either:
    - called as they are
    - called as inlined statements
"""

# ==============================================================================
#
#                                  LEVEL 1
#
# ==============================================================================

# ==============================================================================
def blas_drotg(a: 'float64', b: 'float64',
               c: 'float64' = 0.,
               s: 'float64' = 0.,
              ):
    """
    """
    from pyccel.stdlib.internal.blas import drotg

    drotg (a, b, c, s)

    return c, s

# ==============================================================================
def blas_drotmg(d1: 'float64', d2: 'float64', x1: 'float64', y1: 'float64',
                param: 'float64[:]'):
    """
    """
    from pyccel.stdlib.internal.blas import drotmg

    drotmg (d1, d2, x1, y1, param)

# ==============================================================================
def blas_drot(x: 'float64[:]', y: 'float64[:]', c: 'float64', s: 'float64',
               incx: 'int64' = 1,
               incy: 'int64' = 1
              ):
    """
    """
    from pyccel.stdlib.internal.blas import drot

    n = x.shape[0]

    drot (n, x, incx, y, incy, c, s)

# ==============================================================================
def blas_drotm(x: 'float64[:]', y: 'float64[:]', param: 'float64[:]',
               incx: 'int64' = 1,
               incy: 'int64' = 1
              ):
    """
    """
    from pyccel.stdlib.internal.blas import drotm

    n = x.shape[0]

    drotm (n, x, incx, y, incy, param)

# ==============================================================================
def blas_dcopy(x: 'float64[:]', y: 'float64[:]',
               incx: 'int64' = 1,
               incy: 'int64' = 1
              ):
    """
    y ← x
    """
    from pyccel.stdlib.internal.blas import dcopy

    n = x.shape[0]

    dcopy (n, x, incx, y, incy)

# ==============================================================================
def blas_dswap(x: 'float64[:]', y: 'float64[:]',
               incx: 'int64' = 1,
               incy: 'int64' = 1
              ):
    """
    x ↔ y
    """
    from pyccel.stdlib.internal.blas import dswap

    n = x.shape[0]

    dswap (n, x, incx, y, incy)

# ==============================================================================
def blas_dscal(alpha: 'float64', x: 'float64[:]',
               incx: 'int64' = 1,
              ):
    """
    x ← αx
    """
    from pyccel.stdlib.internal.blas import dscal

    n = x.shape[0]

    dscal (n, alpha, x, incx)

# ==============================================================================
def blas_ddot(x: 'float64[:]', y: 'float64[:]',
               incx: 'int64' = 1,
               incy: 'int64' = 1
              ):
    """
    y ← x
    """
    from pyccel.stdlib.internal.blas import ddot

    n = x.shape[0]

    return ddot (n, x, incx, y, incy)

# ==============================================================================
def blas_daxpy(x: 'float64[:]', y: 'float64[:]',
               a: 'float64' = 1.,
               incx: 'int64' = 1,
               incy: 'int64' = 1
              ):
    """
    y ← αx + y
    """
    from pyccel.stdlib.internal.blas import daxpy

    n = x.shape[0]

    daxpy (n, a, x, incx, y, incy)

# ==============================================================================
def blas_dnrm2(x: 'float64[:]',
               incx: 'int64' = 1,
              ):
    """
    ||x||_2
    """
    from pyccel.stdlib.internal.blas import dnrm2

    n = x.shape[0]

    return dnrm2 (n, x, incx)

# ==============================================================================
def blas_dasum(x: 'float64[:]',
               incx: 'int64' = 1,
              ):
    """
    """
    from pyccel.stdlib.internal.blas import dasum

    n = x.shape[0]

    return dasum (n, x, incx)

# ==============================================================================
def blas_idamax(x: 'float64[:]',
               incx: 'int64' = 1,
              ):
    """
    """
    from pyccel.stdlib.internal.blas import idamax

    n = x.shape[0]

    i = idamax (n, x, incx)
    # we must substruct 1 because of the fortran indexing
    i = i-1
    return i

# ==============================================================================
#
#                                  LEVEL 2
#
# ==============================================================================

# ==============================================================================
def blas_dgemv(alpha: 'float64', a: 'float64[:,:]', x: 'float64[:]', y: 'float64[:]',
               beta: 'float64' = 1.,
               incx: 'int64' = 1,
               incy: 'int64' = 1,
               trans: 'bool' = False
              ):
    """
    y ← αAx + βy, y ← αA T x + βy, y ← αA H x + βy
    """
    from pyccel.stdlib.internal.blas import dgemv

    m,n = a.shape
    lda = m

    # ...
    flag = 'N'
    if trans: flag = 'T'
    # ...

    dgemv (flag, m, n, alpha, a, lda, x, incx, beta, y, incy)

# ==============================================================================
def blas_dgbmv(kl : 'int64', ku: 'int64',
               alpha: 'float64', a: 'float64[:,:]', x: 'float64[:]', y: 'float64[:]',
               beta: 'float64' = 1.,
               incx: 'int64' = 1,
               incy: 'int64' = 1,
               trans: 'bool' = False
              ):
    """
    y ← αAx + βy, y ← αA T x + βy, y ← αA H x + βy
    """
    from pyccel.stdlib.internal.blas import dgbmv

    m,n = a.shape
    # TODO which one?
    #      shall we put lda as optional kwarg?
    lda = m
#    lda = kl + ku + 1

    # ...
    flag = 'N'
    if trans: flag = 'T'
    # ...

    dgbmv (flag, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy)

# ==============================================================================
def blas_dsymv(alpha: 'float64', a: 'float64[:,:]', x: 'float64[:]', y: 'float64[:]',
               beta: 'float64' = 1.,
               incx: 'int64' = 1,
               incy: 'int64' = 1,
               lower: 'bool' = False
              ):
    """
    y ← αAx + βy
    """
    from pyccel.stdlib.internal.blas import dsymv

    m,n = a.shape
    lda = m

    # ...
    flag = 'U'
    if lower : flag = 'L'
    # ...

    dsymv (flag, n, alpha, a, lda, x, incx, beta, y, incy)

# ==============================================================================
def blas_dsbmv(k : 'int64',
               alpha: 'float64', a: 'float64[:,:]', x: 'float64[:]', y: 'float64[:]',
               beta: 'float64' = 1.,
               incx: 'int64' = 1,
               incy: 'int64' = 1,
               lower: 'bool' = False
              ):
    """
    y ← αAx + βy
    """
    from pyccel.stdlib.internal.blas import dsbmv

    m,n = a.shape
    lda = m

    # ...
    flag = 'U'
    if lower : flag = 'L'
    # ...

    dsbmv (flag, n, k, alpha, a, lda, x, incx, beta, y, incy)

# ==============================================================================
def blas_dger(alpha: 'float64', x: 'float64[:]', y: 'float64[:]', a: 'float64[:,:]',
              incx: 'int64' = 1,
              incy: 'int64' = 1,
              ):
    """
    A ← αxy^T + A
    """
    from pyccel.stdlib.internal.blas import dger

    m,n = a.shape
    lda = m

    dger (m, n, alpha, x, incx, y, incy, a, lda)

# ==============================================================================
#
#                                  LEVEL 3
#
# ==============================================================================

# ==============================================================================
def blas_dgemm(alpha: 'float64', a: 'float64[:,:](order=F)', b: 'float64[:,:](order=F)', c: 'float64[:,:](order=F)',
               beta: 'float64' = 0.,
               trans_a: 'bool' = False,
               trans_b: 'bool' = False
              ):
    """
    C ← αAB + βC
    C ← αAB^T + βC
    C ← αA^TB + βC
    C ← αA^TB^T + βC
    """
    from pyccel.stdlib.internal.blas import dgemm

    l,n = c.shape

    # ...
    flag_a = 'N'
    if trans_a: flag_a = 'T'

    flag_b = 'N'
    if trans_b: flag_b = 'T'
    # ...

    # ...
    if trans_a:
        m = a.shape[0]
    else:
        m = a.shape[1]
    # ...

    # TODO to be checked
    lda = m
    ldb = m
    ldc = l

    dgemm (flag_a, flag_b, l, n, m, alpha, a, lda, b, ldb, beta, c, ldc)

# ==============================================================================
def blas_dsymm(alpha: 'float64', a: 'float64[:,:](order=F)', b: 'float64[:,:](order=F)', c: 'float64[:,:](order=F)',
               beta: 'float64' = 0.,
               side: 'bool' = False,
               lower: 'bool' = False,
              ):
    """
    C←αAB+βC
    C←αBA+βC
    """
    from pyccel.stdlib.internal.blas import dsymm

    m,n = c.shape

    # ...
    # equation 1
    flag_side = 'L'
    lda = m
    # equation 2
    if side:
        flag_side = 'R'
        lda = n
    # ...

    # ...
    flag_lower = 'U'
    if lower : flag_lower = 'L'
    # ...

    ldb = m
    ldc = m

    dsymm (flag_side, flag_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc)

# ==============================================================================
def blas_dsyrk(alpha: 'float64', a: 'float64[:,:](order=F)', c: 'float64[:,:](order=F)',
               beta: 'float64' = 0.,
               lower: 'bool' = False,
               trans: 'bool' = False,
              ):
    """
    C ← αAA^T+βC
    C ← αA^TA+βC
    """
    from pyccel.stdlib.internal.blas import dsyrk

    n,k = c.shape

    # ...
    # equation 1
    flag_trans = 'N'
    lda = n
    # equation 2
    if trans:
        flag_trans = 'T'
        lda = k
    # ...

    # ...
    flag_lower = 'U'
    if lower : flag_lower = 'L'
    # ...

    ldc = n

    dsyrk (flag_lower, flag_trans, n, k, alpha, a, lda, beta, c, ldc)

# ==============================================================================
def blas_dsyr2k(alpha: 'float64', a: 'float64[:,:](order=F)', b: 'float64[:,:](order=F)', c: 'float64[:,:](order=F)',
               beta: 'float64' = 0.,
               lower: 'bool' = False,
               trans: 'bool' = False,
              ):
    """
    C ← αAB^T+αBA^T+βC
    C ← αA^TB+αB^TA+βC
    """
    from pyccel.stdlib.internal.blas import dsyr2k

    n = c.shape[0]

    # ...
    # equation 1
    flag_trans = 'N'
    k = a.shape[1]
    lda = n
    ldb = n
    # equation 2
    if trans:
        flag_trans = 'T'
        k = a.shape[0]
        lda = k
        ldb = k
    # ...

    # ...
    flag_lower = 'U'
    if lower : flag_lower = 'L'
    # ...

    ldc = n

    dsyr2k (flag_lower, flag_trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
