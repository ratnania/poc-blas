"""
The aim of this file is to provide functions that can be either:
    - called as they are
    - called as inlined statements
"""

import numpy as np

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
              incx: 'int32' = 1,
              incy: 'int32' = 1
              ):
    """
    """
    from pyccel.stdlib.internal.blas import drot

    n = np.int32(x.shape[0])

    drot (n, x, incx, y, incy, c, s)

# ==============================================================================
def blas_drotm(x: 'float64[:]', y: 'float64[:]', param: 'float64[:]',
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):
    """
    """
    from pyccel.stdlib.internal.blas import drotm

    n = np.int32(x.shape[0])

    drotm (n, x, incx, y, incy, param)

# ==============================================================================
def blas_dcopy(x: 'float64[:]', y: 'float64[:]',
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):
    """
    y ← x
    """
    from pyccel.stdlib.internal.blas import dcopy

    n = np.int32(x.shape[0])

    dcopy (n, x, incx, y, incy)

# ==============================================================================
def blas_dswap(x: 'float64[:]', y: 'float64[:]',
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):
    """
    x ↔ y
    """
    from pyccel.stdlib.internal.blas import dswap

    n = np.int32(x.shape[0])

    dswap (n, x, incx, y, incy)

# ==============================================================================
def blas_dscal(alpha: 'float64', x: 'float64[:]',
               incx: 'int32' = 1,
              ):
    """
    x ← αx
    """
    from pyccel.stdlib.internal.blas import dscal

    n = np.int32(x.shape[0])

    dscal (n, alpha, x, incx)

# ==============================================================================
def blas_ddot(x: 'float64[:]', y: 'float64[:]',
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):
    """
    y ← x
    """
    from pyccel.stdlib.internal.blas import ddot

    n = np.int32(x.shape[0])

    return ddot (n, x, incx, y, incy)

# ==============================================================================
def blas_daxpy(x: 'float64[:]', y: 'float64[:]',
               a: 'float64' = 1.,
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):
    """
    y ← αx + y
    """
    from pyccel.stdlib.internal.blas import daxpy

    n = np.int32(x.shape[0])

    daxpy (n, a, x, incx, y, incy)

# ==============================================================================
def blas_dnrm2(x: 'float64[:]',
               incx: 'int32' = 1,
              ):
    """
    ||x||_2
    """
    from pyccel.stdlib.internal.blas import dnrm2

    n = np.int32(x.shape[0])

    return dnrm2 (n, x, incx)

# ==============================================================================
def blas_dasum(x: 'float64[:]',
               incx: 'int32' = 1,
              ):
    """
    """
    from pyccel.stdlib.internal.blas import dasum

    n = np.int32(x.shape[0])

    return dasum (n, x, incx)

# ==============================================================================
def blas_idamax(x: 'float64[:]',
               incx: 'int32' = 1,
              ):
    """
    """
    from pyccel.stdlib.internal.blas import idamax

    n = np.int32(x.shape[0])

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
def blas_dgemv(alpha: 'float64', a: 'float64[:,:](order=F)', x: 'float64[:]', y: 'float64[:]',
               beta: 'float64' = 0.,
               incx: 'int32' = 1,
               incy: 'int32' = 1,
               trans: 'bool' = False
              ):
    """
    y ← αAx + βy, y ← αA T x + βy, y ← αA H x + βy
    """
    from pyccel.stdlib.internal.blas import dgemv

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    # ...
    flag_trans = 'N'
    if trans: flag_trans = 'T'
    # ...

    dgemv (flag_trans, m, n, alpha, a, lda, x, incx, beta, y, incy)

# ==============================================================================
def blas_dgbmv(kl : 'int32', ku: 'int32',
               alpha: 'float64', a: 'float64[:,:](order=F)', x: 'float64[:]', y: 'float64[:]',
               beta: 'float64' = 0.,
               incx: 'int32' = 1,
               incy: 'int32' = 1,
               trans: 'bool' = False
              ):
    """
    y ← αAx + βy, y ← αA T x + βy, y ← αA H x + βy
    """
    from pyccel.stdlib.internal.blas import dgbmv

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    # TODO which one?
    #      shall we put lda as optional kwarg?
    lda = m
#    lda = kl + ku + 1

    # ...
    flag_trans = 'N'
    if trans: flag_trans = 'T'
    # ...

    dgbmv (flag_trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy)

# ==============================================================================
def blas_dsymv(alpha: 'float64', a: 'float64[:,:](order=F)', x: 'float64[:]', y: 'float64[:]',
               beta: 'float64' = 0.,
               incx: 'int32' = 1,
               incy: 'int32' = 1,
               lower: 'bool' = False
              ):
    """
    DSYMV  performs the matrix-vector  operation

    y := alpha*A*x + beta*y,

    where alpha and beta are scalars, x and y are n element vectors and
    A is an n by n symmetric matrix.
    """
    from pyccel.stdlib.internal.blas import dsymv

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    dsymv (flag_uplo, n, alpha, a, lda, x, incx, beta, y, incy)

# ==============================================================================
def blas_dsbmv(k : 'int32',
               alpha: 'float64', a: 'float64[:,:]', x: 'float64[:]', y: 'float64[:]',
               beta: 'float64' = 0.,
               incx: 'int32' = 1,
               incy: 'int32' = 1,
               lower: 'bool' = False
              ):
    """
    y ← αAx + βy
    """
    from pyccel.stdlib.internal.blas import dsbmv

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    dsbmv (flag_uplo, n, k, alpha, a, lda, x, incx, beta, y, incy)

# ==============================================================================
def blas_dtrmv(a: 'float64[:,:](order=F)', x: 'float64[:]',
               incx: 'int32' = 1,
               lower: 'bool' = False,
               trans: 'bool' = False,
               diag: 'bool' = False
              ):
    """
    DTRMV  performs one of the matrix-vector operations

    x := A*x,   or   x := A**T*x,

    where x is an n element vector and  A is an n by n unit, or non-unit,
    upper or lower triangular matrix.
    """
    from pyccel.stdlib.internal.blas import dtrmv

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    # ...
    flag_trans = 'N'
    if trans: flag_trans = 'T'
    # ...

    # ...
    flag_diag = 'N'
    if diag: flag_diag = 'U'
    # ...

    dtrmv (flag_uplo, flag_trans, flag_diag, n, a, lda, x, incx)

# ==============================================================================
def blas_dtrsv(a: 'float64[:,:](order=F)', x: 'float64[:]',
               incx: 'int32' = 1,
               lower: 'bool' = False,
               trans: 'bool' = False,
               diag: 'bool' = False
              ):
    """
    DTRSV  solves one of the systems of equations

    A*x = b,   or   A**T*x = b,

    where b and x are n element vectors and A is an n by n unit, or
    non-unit, upper or lower triangular matrix.

    No test for singularity or near-singularity is included in this
    routine. Such tests must be performed before calling this routine.
    """

    from pyccel.stdlib.internal.blas import dtrsv

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    # ...
    flag_trans = 'N'
    if trans: flag_trans = 'T'
    # ...

    # ...
    flag_diag = 'N'
    if diag: flag_diag = 'U'
    # ...

    dtrsv (flag_uplo, flag_trans, flag_diag, n, a, lda, x, incx)

# ==============================================================================
def blas_dger(alpha: 'float64', x: 'float64[:]', y: 'float64[:]', a: 'float64[:,:]',
              incx: 'int32' = 1,
              incy: 'int32' = 1,
              ):
    """
    A ← αxy^T + A
    """
    from pyccel.stdlib.internal.blas import dger

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    dger (m, n, alpha, x, incx, y, incy, a, lda)

# ==============================================================================
def blas_dsyr(alpha: 'float64', x: 'float64[:]', a: 'float64[:,:](order=F)',
              incx: 'int32' = 1,
              lower: 'bool' = False
              ):
    """
    DSYR   performs the symmetric rank 1 operation

    A := alpha*x*x**T + A,

    where alpha is a real scalar, x is an n element vector and A is an
    n by n symmetric matrix.
    """
    from pyccel.stdlib.internal.blas import dsyr

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    dsyr (flag_uplo, n, alpha, x, incx, a, lda)

# ==============================================================================
def blas_dspr(alpha: 'float64', x: 'float64[:]', a: 'float64[:]',
              incx: 'int32' = 1,
              lower: 'bool' = False
              ):
    """
    DSPR    performs the symmetric rank 1 operation

    A := alpha*x*x**T + A,

    where alpha is a real scalar, x is an n element vector and A is an
    n by n symmetric matrix, supplied in packed form.
    """
    from pyccel.stdlib.internal.blas import dspr

    n = np.int32(x.shape[0])

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    dspr (flag_uplo, n, alpha, x, incx, a)

# ==============================================================================
def blas_dsyr2(alpha: 'float64', x: 'float64[:]', y: 'float64[:]', a: 'float64[:,:](order=F)',
              incx: 'int32' = 1,
              incy: 'int32' = 1,
              lower: 'bool' = False
              ):
    """
    DSYR2  performs the symmetric rank 2 operation

    A := alpha*x*y**T + alpha*y*x**T + A,

    where alpha is a scalar, x and y are n element vectors and A is an n
    by n symmetric matrix.
    """
    from pyccel.stdlib.internal.blas import dsyr2

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    dsyr2 (flag_uplo, n, alpha, x, incx, y, incy, a, lda)

# ==============================================================================
def blas_dspr2(alpha: 'float64', x: 'float64[:]', y: 'float64[:]', a: 'float64[:]',
              incx: 'int32' = 1,
              incy: 'int32' = 1,
              lower: 'bool' = False
              ):
    """
    DSPR2  performs the symmetric rank 2 operation

    A := alpha*x*y**T + alpha*y*x**T + A,

    where alpha is a scalar, x and y are n element vectors and A is an
    n by n symmetric matrix, supplied in packed form.
    """
    from pyccel.stdlib.internal.blas import dspr2

    n = np.int32(x.shape[0])

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    dspr2 (flag_uplo, n, alpha, x, incx, y, incy, a)

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

    l = np.int32(c.shape[0])
    n = np.int32(c.shape[1])

    # ...
    flag_trans_a = 'N'
    if trans_a: flag_trans_a = 'T'

    flag_trans_b = 'N'
    if trans_b: flag_trans_b = 'T'
    # ...

    # ...
    if trans_a:
        m = np.int32(a.shape[0])
    else:
        m = np.int32(a.shape[1])
    # ...

    # TODO to be checked
    lda = m
    ldb = m
    ldc = l

    dgemm (flag_trans_a, flag_trans_b, l, n, m, alpha, a, lda, b, ldb, beta, c, ldc)

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

    m = np.int32(c.shape[0])
    n = np.int32(c.shape[1])

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
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    ldb = m
    ldc = m

    dsymm (flag_side, flag_uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc)

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

    n = np.int32(c.shape[0])
    k = np.int32(c.shape[1])

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
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    ldc = n

    dsyrk (flag_uplo, flag_trans, n, k, alpha, a, lda, beta, c, ldc)

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

    n = np.int32(c.shape[0])

    # ...
    # equation 1
    flag_trans = 'N'
    k = np.int32(a.shape[1])
    lda = n
    ldb = n
    # equation 2
    if trans:
        flag_trans = 'T'
        k = np.int32(a.shape[0])
        lda = k
        ldb = k
    # ...

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    ldc = n

    dsyr2k (flag_uplo, flag_trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc)

# ==============================================================================
def blas_dtrmm(alpha: 'float64', a: 'float64[:,:](order=F)', b: 'float64[:,:](order=F)',
               side: 'bool' = False,
               lower: 'bool' = False,
               trans_a: 'bool' = False,
               diag: 'bool' = False
              ):
    """
    DTRMM  performs one of the matrix-matrix operations

    B := alpha*op( A )*B,   or   B := alpha*B*op( A ),

    where  alpha  is a scalar,  B  is an m by n matrix,  A  is a unit, or
    non-unit,  upper or lower triangular matrix  and  op( A )  is one  of

    op( A ) = A   or   op( A ) = A**T.
    """
    from pyccel.stdlib.internal.blas import dtrmm

    m = np.int32(b.shape[0])
    n = np.int32(b.shape[1])

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
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    # ...
    flag_trans_a = 'N'
    if trans_a: flag_trans_a = 'T'
    # ...

    # ...
    flag_diag = 'N'
    if diag: flag_diag = 'U'
    # ...

    ldb = m

    dtrmm (flag_side, flag_uplo, flag_trans_a, flag_diag, m, n, alpha, a, lda, b, ldb)

# ==============================================================================
def blas_dtrsm(alpha: 'float64', a: 'float64[:,:](order=F)', b: 'float64[:,:](order=F)',
               side: 'bool' = False,
               lower: 'bool' = False,
               trans_a: 'bool' = False,
               diag: 'bool' = False
              ):
    """
    DTRSM  solves one of the matrix equations

    op( A )*X = alpha*B,   or   X*op( A ) = alpha*B,

    where alpha is a scalar, X and B are m by n matrices, A is a unit, or
    non-unit,  upper or lower triangular matrix  and  op( A )  is one  of

    op( A ) = A   or   op( A ) = A**T.

    The matrix X is overwritten on B.
    """
    from pyccel.stdlib.internal.blas import dtrsm

    m = np.int32(b.shape[0])
    n = np.int32(b.shape[1])

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
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    # ...
    flag_trans_a = 'N'
    if trans_a: flag_trans_a = 'T'
    # ...

    # ...
    flag_diag = 'N'
    if diag: flag_diag = 'U'
    # ...

    ldb = m

    dtrsm (flag_side, flag_uplo, flag_trans_a, flag_diag, m, n, alpha, a, lda, b, ldb)
