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
def blas_ccopy(x: 'complex64[:]', y: 'complex64[:]',
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):
    """
    DCOPY copies a vector, x, to a vector, y.
    uses unrolled loops for increments equal to 1.
    """
    from pyccel.stdlib.internal.blas import ccopy

    n = np.int32(x.shape[0])

    ccopy (n, x, incx, y, incy)

# ==============================================================================
def blas_cswap(x: 'complex64[:]', y: 'complex64[:]',
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):
    """
    DSWAP interchanges two vectors.
    uses unrolled loops for increments equal to 1.
    """
    from pyccel.stdlib.internal.blas import cswap

    n = np.int32(x.shape[0])

    cswap (n, x, incx, y, incy)

# ==============================================================================
def blas_cscal(alpha: 'complex64', x: 'complex64[:]',
               incx: 'int32' = 1,
              ):
    """
    DSCAL scales a vector by a constant.
    uses unrolled loops for increment equal to 1.
    """
    from pyccel.stdlib.internal.blas import cscal

    n = np.int32(x.shape[0])

    cscal (n, alpha, x, incx)

# ==============================================================================
def blas_cdotc(x: 'complex64[:]', y: 'complex64[:]',
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):
    """
    CDOTC forms the dot product of two complex vectors
      CDOTC = X^H * Y
    """
    from pyccel.stdlib.internal.blas import cdotc

    n = np.int32(x.shape[0])

    return cdotc (n, x, incx, y, incy)

# ==============================================================================
def blas_cdotu(x: 'complex64[:]', y: 'complex64[:]',
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):
    """
    CDOTU forms the dot product of two complex vectors
      CDOTU = X^T * Y
    """
    from pyccel.stdlib.internal.blas import cdotu

    n = np.int32(x.shape[0])

    return cdotu (n, x, incx, y, incy)

# ==============================================================================
def blas_caxpy(x: 'complex64[:]', y: 'complex64[:]',
               a: 'complex64' = 1.,
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):
    """
    DAXPY constant times a vector plus a vector.
    uses unrolled loops for increments equal to one.
    """
    from pyccel.stdlib.internal.blas import caxpy

    n = np.int32(x.shape[0])

    caxpy (n, a, x, incx, y, incy)

# ==============================================================================
def blas_scnrm2(x: 'complex64[:]',
               incx: 'int32' = 1,
              ):
    """
    DNRM2 returns the euclidean norm of a vector via the function
    name, so that

    DNRM2 := sqrt( x'*x )
    """
    from pyccel.stdlib.internal.blas import scnrm2

    n = np.int32(x.shape[0])

    return scnrm2 (n, x, incx)

# ==============================================================================
def blas_scasum(x: 'complex64[:]',
               incx: 'int32' = 1,
              ):
    """
    DASUM takes the sum of the absolute values.
    """
    from pyccel.stdlib.internal.blas import scasum

    n = np.int32(x.shape[0])

    return scasum (n, x, incx)

# ==============================================================================
def blas_icamax(x: 'complex64[:]',
               incx: 'int32' = 1,
              ):
    """
    IDAMAX finds the index of the first element having maximum absolute value.
    """
    from pyccel.stdlib.internal.blas import icamax

    n = np.int32(x.shape[0])

    i = icamax (n, x, incx)
    # we must substruct 1 because of the fortran indexing
    i = i-1
    return i

# ==============================================================================
#
#                                  LEVEL 2
#
# ==============================================================================

# ==============================================================================
def blas_cgemv(alpha: 'complex64', a: 'complex64[:,:](order=F)', x: 'complex64[:]', y: 'complex64[:]',
               beta: 'complex64' = 0.,
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
    from pyccel.stdlib.internal.blas import cgemv

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    # ...
    flag_trans = 'N'
    if trans: flag_trans = 'T'
    # ...

    cgemv (flag_trans, m, n, alpha, a, lda, x, incx, beta, y, incy)

# ==============================================================================
def blas_cgbmv(kl : 'int32', ku: 'int32', alpha: 'complex64',
               a: 'complex64[:,:](order=F)', x: 'complex64[:]', y: 'complex64[:]',
               beta: 'complex64' = 0.,
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
    from pyccel.stdlib.internal.blas import cgbmv

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])

    lda = m
#    lda = np.int32(1) + ku + kl

    # ...
    flag_trans = 'N'
    if trans: flag_trans = 'T'
    # ...

    cgbmv (flag_trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy)

# ==============================================================================
def blas_chemv(alpha: 'complex64', a: 'complex64[:,:](order=F)', x: 'complex64[:]', y: 'complex64[:]',
               beta: 'complex64' = 0.,
               incx: 'int32' = 1,
               incy: 'int32' = 1,
               lower: 'bool' = False
              ):
    """
    CHEMV  performs the matrix-vector  operation

    y := alpha*A*x + beta*y,

    where alpha and beta are scalars, x and y are n element vectors and
    A is an n by n hermitian matrix.
    """
    from pyccel.stdlib.internal.blas import chemv

    n = np.int32(a.shape[0])
    lda = n

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    chemv (flag_uplo, n, alpha, a, lda, x, incx, beta, y, incy)

# ==============================================================================
def blas_chbmv(k : 'int32', alpha: 'complex64',
               a: 'complex64[:,:](order=F)', x: 'complex64[:]', y: 'complex64[:]',
               beta: 'complex64' = 0.,
               incx: 'int32' = 1,
               incy: 'int32' = 1,
               lower: 'bool' = False
              ):
    """
    CHBMV  performs the matrix-vector  operation

    y := alpha*A*x + beta*y,

    where alpha and beta are scalars, x and y are n element vectors and
    A is an n by n hermitian band matrix, with k super-diagonals.
    """
    from pyccel.stdlib.internal.blas import chbmv

    n = np.int32(a.shape[0])
    lda = n

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    chbmv (flag_uplo, n, k, alpha, a, lda, x, incx, beta, y, incy)

# ==============================================================================
def blas_chpmv(alpha: 'complex64', a: 'complex64[:]', x: 'complex64[:]', y: 'complex64[:]',
               beta: 'complex64' = 0.,
               incx: 'int32' = 1,
               incy: 'int32' = 1,
               lower: 'bool' = False
              ):
    """
    CHPMV  performs the matrix-vector operation

    y := alpha*A*x + beta*y,

    where alpha and beta are scalars, x and y are n element vectors and
    A is an n by n hermitian matrix, supplied in packed form.
    """
    from pyccel.stdlib.internal.blas import chpmv

    n = np.int32(x.shape[0])

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    chpmv (flag_uplo, n, alpha, a, x, incx, beta, y, incy)

# ==============================================================================
def blas_ctrmv(a: 'complex64[:,:](order=F)', x: 'complex64[:]',
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
    from pyccel.stdlib.internal.blas import ctrmv

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

    ctrmv (flag_uplo, flag_trans, flag_diag, n, a, lda, x, incx)

# ==============================================================================
def blas_ctbmv(k : 'int32', a: 'complex64[:,:](order=F)', x: 'complex64[:]',
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
    from pyccel.stdlib.internal.blas import ctbmv

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

    ctbmv (flag_uplo, flag_trans, flag_diag, n, k, a, lda, x, incx)

# ==============================================================================
def blas_ctpmv(a: 'complex64[:]', x: 'complex64[:]',
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
    from pyccel.stdlib.internal.blas import ctpmv

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

    ctpmv (flag_uplo, flag_trans, flag_diag, n, a, x, incx)

# ==============================================================================
def blas_ctrsv(a: 'complex64[:,:](order=F)', x: 'complex64[:]',
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

    from pyccel.stdlib.internal.blas import ctrsv

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

    ctrsv (flag_uplo, flag_trans, flag_diag, n, a, lda, x, incx)

# ==============================================================================
def blas_ctbsv(k: 'int32', a: 'complex64[:,:](order=F)', x: 'complex64[:]',
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

    from pyccel.stdlib.internal.blas import ctbsv

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

    ctbsv (flag_uplo, flag_trans, flag_diag, n, k, a, lda, x, incx)

# ==============================================================================
def blas_ctpsv(a: 'complex64[:]', x: 'complex64[:]',
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

    from pyccel.stdlib.internal.blas import ctpsv

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

    ctpsv (flag_uplo, flag_trans, flag_diag, n, a, x, incx)

# ==============================================================================
def blas_cgeru(alpha: 'complex64', x: 'complex64[:]', y: 'complex64[:]', a: 'complex64[:,:](order=F)',
              incx: 'int32' = 1,
              incy: 'int32' = 1,
              ):
    """
    CGERU  performs the rank 1 operation

    A := alpha*x*y**T + A,

    where alpha is a scalar, x is an m element vector, y is an n element
    vector and A is an m by n matrix.
    """
    from pyccel.stdlib.internal.blas import cgeru

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    cgeru (m, n, alpha, x, incx, y, incy, a, lda)

# ==============================================================================
def blas_cgerc(alpha: 'complex64', x: 'complex64[:]', y: 'complex64[:]', a: 'complex64[:,:](order=F)',
              incx: 'int32' = 1,
              incy: 'int32' = 1,
              ):
    """
    CGERC  performs the rank 1 operation

    A := alpha*x*y**H + A,

    where alpha is a scalar, x is an m element vector, y is an n element
    vector and A is an m by n matrix.
    """
    from pyccel.stdlib.internal.blas import cgerc

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    cgerc (m, n, alpha, x, incx, y, incy, a, lda)

# ==============================================================================
def blas_cher(alpha: 'float32', x: 'complex64[:]', a: 'complex64[:,:](order=F)',
              incx: 'int32' = 1,
              lower: 'bool' = False
              ):
    """
    CHER   performs the hermitian rank 1 operation

    A := alpha*x*x**H + A,

    where alpha is a real scalar, x is an n element vector and A is an
    n by n hermitian matrix.
    """
    from pyccel.stdlib.internal.blas import cher

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    cher (flag_uplo, n, alpha, x, incx, a, lda)

# ==============================================================================
def blas_chpr(alpha: 'float32', x: 'complex64[:]', a: 'complex64[:]',
              incx: 'int32' = 1,
              lower: 'bool' = False
              ):
    """
    CHPR    performs the hermitian rank 1 operation

    A := alpha*x*x**H + A,

    where alpha is a real scalar, x is an n element vector and A is an
    n by n hermitian matrix, supplied in packed form.
    """
    from pyccel.stdlib.internal.blas import chpr

    n = np.int32(x.shape[0])

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    chpr (flag_uplo, n, alpha, x, incx, a)

# ==============================================================================
def blas_cher2(alpha: 'complex64', x: 'complex64[:]', y: 'complex64[:]', a: 'complex64[:,:](order=F)',
              incx: 'int32' = 1,
              incy: 'int32' = 1,
              lower: 'bool' = False
              ):
    """
    CHER2  performs the hermitian rank 2 operation

    A := alpha*x*y**H + conjg( alpha )*y*x**H + A,

    where alpha is a scalar, x and y are n element vectors and A is an n
    by n hermitian matrix.
    """
    from pyccel.stdlib.internal.blas import cher2

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    cher2 (flag_uplo, n, alpha, x, incx, y, incy, a, lda)

# ==============================================================================
def blas_chpr2(alpha: 'complex64', x: 'complex64[:]', y: 'complex64[:]', a: 'complex64[:]',
              incx: 'int32' = 1,
              incy: 'int32' = 1,
              lower: 'bool' = False
              ):
    """
    CHPR2  performs the hermitian rank 2 operation

    A := alpha*x*y**H + conjg( alpha )*y*x**H + A,

    where alpha is a scalar, x and y are n element vectors and A is an
    n by n hermitian matrix, supplied in packed form.
    """
    from pyccel.stdlib.internal.blas import chpr2

    n = np.int32(x.shape[0])

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    chpr2 (flag_uplo, n, alpha, x, incx, y, incy, a)

# ==============================================================================
#
#                                  LEVEL 3
#
# ==============================================================================

# ==============================================================================
def blas_cgemm(alpha: 'complex64', a: 'complex64[:,:](order=F)', b: 'complex64[:,:](order=F)', c: 'complex64[:,:](order=F)',
               beta: 'complex64' = 0.,
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
    from pyccel.stdlib.internal.blas import cgemm

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

    cgemm (flag_trans_a, flag_trans_b, l, n, m, alpha, a, lda, b, ldb, beta, c, ldc)

# ==============================================================================
def blas_csymm(alpha: 'complex64', a: 'complex64[:,:](order=F)', b: 'complex64[:,:](order=F)', c: 'complex64[:,:](order=F)',
               beta: 'complex64' = 0.,
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
    from pyccel.stdlib.internal.blas import csymm

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

    csymm (flag_side, flag_uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc)

# ==============================================================================
def blas_chemm(alpha: 'complex64', a: 'complex64[:,:](order=F)', b: 'complex64[:,:](order=F)', c: 'complex64[:,:](order=F)',
               beta: 'complex64' = 0.,
               side: 'bool' = False,
               lower: 'bool' = False,
              ):
    """
    CHEMM  performs one of the matrix-matrix operations

    C := alpha*A*B + beta*C,

    or

    C := alpha*B*A + beta*C,

    where alpha and beta are scalars, A is an hermitian matrix and  B and
    C are m by n matrices.
    """
    from pyccel.stdlib.internal.blas import chemm

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

    chemm (flag_side, flag_uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc)

# ==============================================================================
def blas_csyrk(alpha: 'complex64', a: 'complex64[:,:](order=F)', c: 'complex64[:,:](order=F)',
               beta: 'complex64' = 0.,
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
    from pyccel.stdlib.internal.blas import csyrk

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

    csyrk (flag_uplo, flag_trans, n, k, alpha, a, lda, beta, c, ldc)

# ==============================================================================
def blas_csyr2k(alpha: 'complex64', a: 'complex64[:,:](order=F)', b: 'complex64[:,:](order=F)', c: 'complex64[:,:](order=F)',
               beta: 'complex64' = 0.,
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
    from pyccel.stdlib.internal.blas import csyr2k

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

    csyr2k (flag_uplo, flag_trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc)

# ==============================================================================
def blas_cherk(alpha: 'complex64', a: 'complex64[:,:](order=F)', c: 'complex64[:,:](order=F)',
               beta: 'complex64' = 0.,
               lower: 'bool' = False,
               trans: 'bool' = False,
              ):
    """
    CHERK  performs one of the hermitian rank k operations

    C := alpha*A*A**H + beta*C,

    or

    C := alpha*A**H*A + beta*C,

    where  alpha and beta  are  real scalars,  C is an  n by n  hermitian
    matrix and  A  is an  n by k  matrix in the  first case and a  k by n
    matrix in the second case.
    """
    from pyccel.stdlib.internal.blas import cherk

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

    cherk (flag_uplo, flag_trans, n, k, alpha, a, lda, beta, c, ldc)

# ==============================================================================
def blas_cher2k(alpha: 'complex64', a: 'complex64[:,:](order=F)', b: 'complex64[:,:](order=F)', c: 'complex64[:,:](order=F)',
               beta: 'complex64' = 0.,
               lower: 'bool' = False,
               trans: 'bool' = False,
              ):
    """
    CHER2K  performs one of the hermitian rank 2k operations

    C := alpha*A*B**H + conjg( alpha )*B*A**H + beta*C,

    or

    C := alpha*A**H*B + conjg( alpha )*B**H*A + beta*C,

    where  alpha and beta  are scalars with  beta  real,  C is an  n by n
    hermitian matrix and  A and B  are  n by k matrices in the first case
    and  k by n  matrices in the second case.
    """
    from pyccel.stdlib.internal.blas import cher2k

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

    cher2k (flag_uplo, flag_trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc)

# ==============================================================================
def blas_ctrmm(alpha: 'complex64', a: 'complex64[:,:](order=F)', b: 'complex64[:,:](order=F)',
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
    from pyccel.stdlib.internal.blas import ctrmm

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

    ctrmm (flag_side, flag_uplo, flag_trans_a, flag_diag, m, n, alpha, a, lda, b, ldb)

# ==============================================================================
def blas_ctrsm(alpha: 'complex64', a: 'complex64[:,:](order=F)', b: 'complex64[:,:](order=F)',
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
    from pyccel.stdlib.internal.blas import ctrsm

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

    ctrsm (flag_side, flag_uplo, flag_trans_a, flag_diag, m, n, alpha, a, lda, b, ldb)
