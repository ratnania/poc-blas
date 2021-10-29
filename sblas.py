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
def blas_srotg(a: 'float32', b: 'float32',
               c: 'float32' = 0.,
               s: 'float32' = 0.,
              ):
    """
    Generate plane rotation
    """
    from pyccel.stdlib.internal.blas import srotg

    srotg (a, b, c, s)

    return c, s

# ==============================================================================
def blas_srotmg(d1: 'float32', d2: 'float32', x1: 'float32', y1: 'float32',
                param: 'float32[:]'):
    """
    Generate modified plane rotation
    """
    from pyccel.stdlib.internal.blas import srotmg

    srotmg (d1, d2, x1, y1, param)

# ==============================================================================
def blas_srot(x: 'float32[:]', y: 'float32[:]', c: 'float32', s: 'float32',
              incx: 'int32' = 1,
              incy: 'int32' = 1
              ):
    """
    Apply plane rotation
    """
    from pyccel.stdlib.internal.blas import srot

    n = np.int32(x.shape[0])

    srot (n, x, incx, y, incy, c, s)

# ==============================================================================
def blas_srotm(x: 'float32[:]', y: 'float32[:]', param: 'float32[:]',
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):
    """
    Apply modified plane rotation
    """
    from pyccel.stdlib.internal.blas import srotm

    n = np.int32(x.shape[0])

    srotm (n, x, incx, y, incy, param)

# ==============================================================================
def blas_scopy(x: 'float32[:]', y: 'float32[:]',
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):
    """
    DCOPY copies a vector, x, to a vector, y.
    uses unrolled loops for increments equal to 1.
    """
    from pyccel.stdlib.internal.blas import scopy

    n = np.int32(x.shape[0])

    scopy (n, x, incx, y, incy)

# ==============================================================================
def blas_sswap(x: 'float32[:]', y: 'float32[:]',
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):
    """
    DSWAP interchanges two vectors.
    uses unrolled loops for increments equal to 1.
    """
    from pyccel.stdlib.internal.blas import sswap

    n = np.int32(x.shape[0])

    sswap (n, x, incx, y, incy)

# ==============================================================================
def blas_sscal(alpha: 'float32', x: 'float32[:]',
               incx: 'int32' = 1,
              ):
    """
    DSCAL scales a vector by a constant.
    uses unrolled loops for increment equal to 1.
    """
    from pyccel.stdlib.internal.blas import sscal

    n = np.int32(x.shape[0])

    sscal (n, alpha, x, incx)

# ==============================================================================
def blas_sdot(x: 'float32[:]', y: 'float32[:]',
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):
    """
    DDOT forms the dot product of two vectors.
    uses unrolled loops for increments equal to one.
    """
    from pyccel.stdlib.internal.blas import sdot

    n = np.int32(x.shape[0])

    return sdot (n, x, incx, y, incy)

# ==============================================================================
def blas_saxpy(x: 'float32[:]', y: 'float32[:]',
               a: 'float32' = 1.,
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):
    """
    DAXPY constant times a vector plus a vector.
    uses unrolled loops for increments equal to one.
    """
    from pyccel.stdlib.internal.blas import saxpy

    n = np.int32(x.shape[0])

    saxpy (n, a, x, incx, y, incy)

# ==============================================================================
def blas_snrm2(x: 'float32[:]',
               incx: 'int32' = 1,
              ):
    """
    DNRM2 returns the euclidean norm of a vector via the function
    name, so that

    DNRM2 := sqrt( x'*x )
    """
    from pyccel.stdlib.internal.blas import snrm2

    n = np.int32(x.shape[0])

    return snrm2 (n, x, incx)

# ==============================================================================
def blas_sasum(x: 'float32[:]',
               incx: 'int32' = 1,
              ):
    """
    DASUM takes the sum of the absolute values.
    """
    from pyccel.stdlib.internal.blas import sasum

    n = np.int32(x.shape[0])

    return sasum (n, x, incx)

# ==============================================================================
def blas_isamax(x: 'float32[:]',
               incx: 'int32' = 1,
              ):
    """
    IDAMAX finds the index of the first element having maximum absolute value.
    """
    from pyccel.stdlib.internal.blas import isamax

    n = np.int32(x.shape[0])

    i = isamax (n, x, incx)
    # we must substruct 1 because of the fortran indexing
    i = i-1
    return i

# ==============================================================================
#
#                                  LEVEL 2
#
# ==============================================================================

# ==============================================================================
def blas_sgemv(alpha: 'float32', a: 'float32[:,:](order=F)', x: 'float32[:]', y: 'float32[:]',
               beta: 'float32' = 0.,
               incx: 'int32' = 1,
               incy: 'int32' = 1,
               trans: 'bool' = False
              ):
    """
    DGEMV  performs one of the matrix-vector operations

    y := alpha*A*x + beta*y,   or   y := alpha*A**T*x + beta*y,

    where alpha and beta are scalars, x and y are vectors and A is an
    m by n matrix.
    """
    from pyccel.stdlib.internal.blas import sgemv

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    # ...
    flag_trans = 'N'
    if trans: flag_trans = 'T'
    # ...

    sgemv (flag_trans, m, n, alpha, a, lda, x, incx, beta, y, incy)

# ==============================================================================
def blas_sgbmv(kl : 'int32', ku: 'int32', alpha: 'float32',
               a: 'float32[:,:](order=F)', x: 'float32[:]', y: 'float32[:]',
               beta: 'float32' = 0.,
               incx: 'int32' = 1,
               incy: 'int32' = 1,
               trans: 'bool' = False
              ):
    """
    DGBMV  performs one of the matrix-vector operations

    y := alpha*A*x + beta*y,   or   y := alpha*A**T*x + beta*y,

    where alpha and beta are scalars, x and y are vectors and A is an
    m by n band matrix, with kl sub-diagonals and ku super-diagonals.
    """
    from pyccel.stdlib.internal.blas import sgbmv

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])

    lda = m
#    lda = np.int32(1) + ku + kl

    # ...
    flag_trans = 'N'
    if trans: flag_trans = 'T'
    # ...

    sgbmv (flag_trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy)

# ==============================================================================
def blas_ssymv(alpha: 'float32', a: 'float32[:,:](order=F)', x: 'float32[:]', y: 'float32[:]',
               beta: 'float32' = 0.,
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
    from pyccel.stdlib.internal.blas import ssymv

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    ssymv (flag_uplo, n, alpha, a, lda, x, incx, beta, y, incy)

# ==============================================================================
def blas_ssbmv(k : 'int32', alpha: 'float32',
               a: 'float32[:,:](order=F)', x: 'float32[:]', y: 'float32[:]',
               beta: 'float32' = 0.,
               incx: 'int32' = 1,
               incy: 'int32' = 1,
               lower: 'bool' = False
              ):
    """
    DSBMV  performs the matrix-vector  operation

    y := alpha*A*x + beta*y,

    where alpha and beta are scalars, x and y are n element vectors and
    A is an n by n symmetric band matrix, with k super-diagonals.
    """
    from pyccel.stdlib.internal.blas import ssbmv

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    ssbmv (flag_uplo, n, k, alpha, a, lda, x, incx, beta, y, incy)

# ==============================================================================
def blas_sspmv(alpha: 'float32', a: 'float32[:]', x: 'float32[:]', y: 'float32[:]',
               beta: 'float32' = 0.,
               incx: 'int32' = 1,
               incy: 'int32' = 1,
               lower: 'bool' = False
              ):
    """
    DSPMV  performs the matrix-vector operation

    y := alpha*A*x + beta*y,

    where alpha and beta are scalars, x and y are n element vectors and
    A is an n by n symmetric matrix, supplied in packed form.
    """
    from pyccel.stdlib.internal.blas import sspmv

    n = np.int32(x.shape[0])

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    sspmv (flag_uplo, n, alpha, a, x, incx, beta, y, incy)

# ==============================================================================
def blas_strmv(a: 'float32[:,:](order=F)', x: 'float32[:]',
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
    from pyccel.stdlib.internal.blas import strmv

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

    strmv (flag_uplo, flag_trans, flag_diag, n, a, lda, x, incx)

# ==============================================================================
def blas_stbmv(k : 'int32', a: 'float32[:,:](order=F)', x: 'float32[:]',
               incx: 'int32' = 1,
               lower: 'bool' = False,
               trans: 'bool' = False,
               diag: 'bool' = False
              ):
    """
    DTBMV  performs one of the matrix-vector operations

    x := A*x,   or   x := A**T*x,

    where x is an n element vector and  A is an n by n unit, or non-unit,
    upper or lower triangular band matrix, with ( k + 1 ) diagonals.
    """
    from pyccel.stdlib.internal.blas import stbmv

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

    stbmv (flag_uplo, flag_trans, flag_diag, n, k, a, lda, x, incx)

# ==============================================================================
def blas_stpmv(a: 'float32[:]', x: 'float32[:]',
               incx: 'int32' = 1,
               lower: 'bool' = False,
               trans: 'bool' = False,
               diag: 'bool' = False
              ):
    """
    DTPMV  performs one of the matrix-vector operations

    x := A*x,   or   x := A**T*x,

    where x is an n element vector and  A is an n by n unit, or non-unit,
    upper or lower triangular matrix, supplied in packed form.
    """
    from pyccel.stdlib.internal.blas import stpmv

    n = np.int32(x.shape[0])

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

    stpmv (flag_uplo, flag_trans, flag_diag, n, a, x, incx)

# ==============================================================================
def blas_strsv(a: 'float32[:,:](order=F)', x: 'float32[:]',
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

    from pyccel.stdlib.internal.blas import strsv

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

    strsv (flag_uplo, flag_trans, flag_diag, n, a, lda, x, incx)

# ==============================================================================
def blas_stbsv(k: 'int32', a: 'float32[:,:](order=F)', x: 'float32[:]',
               incx: 'int32' = 1,
               lower: 'bool' = False,
               trans: 'bool' = False,
               diag: 'bool' = False
              ):
    """
    DTBSV  solves one of the systems of equations

    A*x = b,   or   A**T*x = b,

    where b and x are n element vectors and A is an n by n unit, or
    non-unit, upper or lower triangular band matrix, with ( k + 1 )
    diagonals.

    No test for singularity or near-singularity is included in this
    routine. Such tests must be performed before calling this routine.
    """

    from pyccel.stdlib.internal.blas import stbsv

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

    stbsv (flag_uplo, flag_trans, flag_diag, n, k, a, lda, x, incx)

# ==============================================================================
def blas_stpsv(a: 'float32[:]', x: 'float32[:]',
               incx: 'int32' = 1,
               lower: 'bool' = False,
               trans: 'bool' = False,
               diag: 'bool' = False
              ):
    """
    DTPSV  solves one of the systems of equations

    A*x = b,   or   A**T*x = b,

    where b and x are n element vectors and A is an n by n unit, or
    non-unit, upper or lower triangular matrix, supplied in packed form.

    No test for singularity or near-singularity is included in this
    routine. Such tests must be performed before calling this routine.
    """

    from pyccel.stdlib.internal.blas import stpsv

    n = np.int32(x.shape[0])

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

    stpsv (flag_uplo, flag_trans, flag_diag, n, a, x, incx)

# ==============================================================================
def blas_sger(alpha: 'float32', x: 'float32[:]', y: 'float32[:]', a: 'float32[:,:]',
              incx: 'int32' = 1,
              incy: 'int32' = 1,
              ):
    """
    DGER   performs the rank 1 operation

    A := alpha*x*y**T + A,

    where alpha is a scalar, x is an m element vector, y is an n element
    vector and A is an m by n matrix.
    """
    from pyccel.stdlib.internal.blas import sger

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    sger (m, n, alpha, x, incx, y, incy, a, lda)

# ==============================================================================
def blas_ssyr(alpha: 'float32', x: 'float32[:]', a: 'float32[:,:](order=F)',
              incx: 'int32' = 1,
              lower: 'bool' = False
              ):
    """
    DSYR   performs the symmetric rank 1 operation

    A := alpha*x*x**T + A,

    where alpha is a real scalar, x is an n element vector and A is an
    n by n symmetric matrix.
    """
    from pyccel.stdlib.internal.blas import ssyr

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    ssyr (flag_uplo, n, alpha, x, incx, a, lda)

# ==============================================================================
def blas_sspr(alpha: 'float32', x: 'float32[:]', a: 'float32[:]',
              incx: 'int32' = 1,
              lower: 'bool' = False
              ):
    """
    DSPR    performs the symmetric rank 1 operation

    A := alpha*x*x**T + A,

    where alpha is a real scalar, x is an n element vector and A is an
    n by n symmetric matrix, supplied in packed form.
    """
    from pyccel.stdlib.internal.blas import sspr

    n = np.int32(x.shape[0])

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    sspr (flag_uplo, n, alpha, x, incx, a)

# ==============================================================================
def blas_ssyr2(alpha: 'float32', x: 'float32[:]', y: 'float32[:]', a: 'float32[:,:](order=F)',
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
    from pyccel.stdlib.internal.blas import ssyr2

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    ssyr2 (flag_uplo, n, alpha, x, incx, y, incy, a, lda)

# ==============================================================================
def blas_sspr2(alpha: 'float32', x: 'float32[:]', y: 'float32[:]', a: 'float32[:]',
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
    from pyccel.stdlib.internal.blas import sspr2

    n = np.int32(x.shape[0])

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    sspr2 (flag_uplo, n, alpha, x, incx, y, incy, a)

# ==============================================================================
#
#                                  LEVEL 3
#
# ==============================================================================

# ==============================================================================
def blas_sgemm(alpha: 'float32', a: 'float32[:,:](order=F)', b: 'float32[:,:](order=F)', c: 'float32[:,:](order=F)',
               beta: 'float32' = 0.,
               trans_a: 'bool' = False,
               trans_b: 'bool' = False
              ):
    """
    DGEMM  performs one of the matrix-matrix operations

    C := alpha*op( A )*op( B ) + beta*C,

    where  op( X ) is one of

    op( X ) = X   or   op( X ) = X**T,

    alpha and beta are scalars, and A, B and C are matrices, with op( A )
    an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.
    """
    from pyccel.stdlib.internal.blas import sgemm

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

    sgemm (flag_trans_a, flag_trans_b, l, n, m, alpha, a, lda, b, ldb, beta, c, ldc)

# ==============================================================================
def blas_ssymm(alpha: 'float32', a: 'float32[:,:](order=F)', b: 'float32[:,:](order=F)', c: 'float32[:,:](order=F)',
               beta: 'float32' = 0.,
               side: 'bool' = False,
               lower: 'bool' = False,
              ):
    """
    DSYMM  performs one of the matrix-matrix operations

    C := alpha*A*B + beta*C,

    or

    C := alpha*B*A + beta*C,

    where alpha and beta are scalars,  A is a symmetric matrix and  B and
    C are  m by n matrices.
    """
    from pyccel.stdlib.internal.blas import ssymm

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

    ssymm (flag_side, flag_uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc)

# ==============================================================================
def blas_ssyrk(alpha: 'float32', a: 'float32[:,:](order=F)', c: 'float32[:,:](order=F)',
               beta: 'float32' = 0.,
               lower: 'bool' = False,
               trans: 'bool' = False,
              ):
    """
    DSYRK  performs one of the symmetric rank k operations

    C := alpha*A*A**T + beta*C,

    or

    C := alpha*A**T*A + beta*C,

    where  alpha and beta  are scalars, C is an  n by n  symmetric matrix
    and  A  is an  n by k  matrix in the first case and a  k by n  matrix
    in the second case.
    """
    from pyccel.stdlib.internal.blas import ssyrk

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

    ssyrk (flag_uplo, flag_trans, n, k, alpha, a, lda, beta, c, ldc)

# ==============================================================================
def blas_ssyr2k(alpha: 'float32', a: 'float32[:,:](order=F)', b: 'float32[:,:](order=F)', c: 'float32[:,:](order=F)',
               beta: 'float32' = 0.,
               lower: 'bool' = False,
               trans: 'bool' = False,
              ):
    """
    DSYR2K  performs one of the symmetric rank 2k operations

    C := alpha*A*B**T + alpha*B*A**T + beta*C,

    or

    C := alpha*A**T*B + alpha*B**T*A + beta*C,

    where  alpha and beta  are scalars, C is an  n by n  symmetric matrix
    and  A and B  are  n by k  matrices  in the  first  case  and  k by n
    matrices in the second case.
    """
    from pyccel.stdlib.internal.blas import ssyr2k

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

    ssyr2k (flag_uplo, flag_trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc)

# ==============================================================================
def blas_strmm(alpha: 'float32', a: 'float32[:,:](order=F)', b: 'float32[:,:](order=F)',
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
    from pyccel.stdlib.internal.blas import strmm

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

    strmm (flag_side, flag_uplo, flag_trans_a, flag_diag, m, n, alpha, a, lda, b, ldb)

# ==============================================================================
def blas_strsm(alpha: 'float32', a: 'float32[:,:](order=F)', b: 'float32[:,:](order=F)',
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
    from pyccel.stdlib.internal.blas import strsm

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

    strsm (flag_side, flag_uplo, flag_trans_a, flag_diag, m, n, alpha, a, lda, b, ldb)
