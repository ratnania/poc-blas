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
def blas_zcopy(x: 'complex128[:]', y: 'complex128[:]',
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):
    """
    DCOPY copies a vector, x, to a vector, y.
    uses unrolled loops for increments equal to 1.
    """
    from pyccel.stdlib.internal.blas import zcopy

    n = np.int32(x.shape[0])

    zcopy (n, x, incx, y, incy)

# ==============================================================================
def blas_zswap(x: 'complex128[:]', y: 'complex128[:]',
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):
    """
    DSWAP interchanges two vectors.
    uses unrolled loops for increments equal to 1.
    """
    from pyccel.stdlib.internal.blas import zswap

    n = np.int32(x.shape[0])

    zswap (n, x, incx, y, incy)

# ==============================================================================
def blas_zscal(alpha: 'complex128', x: 'complex128[:]',
               incx: 'int32' = 1,
              ):
    """
    DSCAL scales a vector by a constant.
    uses unrolled loops for increment equal to 1.
    """
    from pyccel.stdlib.internal.blas import zscal

    n = np.int32(x.shape[0])

    zscal (n, alpha, x, incx)

# ==============================================================================
def blas_zdotc(x: 'complex128[:]', y: 'complex128[:]',
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):
    """
    CDOTC forms the dot product of two complex vectors
      CDOTC = X^H * Y
    """
    from pyccel.stdlib.internal.blas import zdotc

    n = np.int32(x.shape[0])

    return zdotc (n, x, incx, y, incy)

# ==============================================================================
def blas_zdotu(x: 'complex128[:]', y: 'complex128[:]',
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):
    """
    CDOTU forms the dot product of two complex vectors
      CDOTU = X^T * Y
    """
    from pyccel.stdlib.internal.blas import zdotu

    n = np.int32(x.shape[0])

    return zdotu (n, x, incx, y, incy)

# ==============================================================================
def blas_zaxpy(x: 'complex128[:]', y: 'complex128[:]',
               a: 'complex128' = 1.,
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):
    """
    DAXPY constant times a vector plus a vector.
    uses unrolled loops for increments equal to one.
    """
    from pyccel.stdlib.internal.blas import zaxpy

    n = np.int32(x.shape[0])

    zaxpy (n, a, x, incx, y, incy)

# ==============================================================================
def blas_dznrm2(x: 'complex128[:]',
               incx: 'int32' = 1,
              ):
    """
    DNRM2 returns the euclidean norm of a vector via the function
    name, so that

    DNRM2 := sqrt( x'*x )
    """
    from pyccel.stdlib.internal.blas import dznrm2

    n = np.int32(x.shape[0])

    return dznrm2 (n, x, incx)

# ==============================================================================
def blas_dzasum(x: 'complex128[:]',
               incx: 'int32' = 1,
              ):
    """
    DASUM takes the sum of the absolute values.
    """
    from pyccel.stdlib.internal.blas import dzasum

    n = np.int32(x.shape[0])

    return dzasum (n, x, incx)

# ==============================================================================
def blas_izamax(x: 'complex128[:]',
               incx: 'int32' = 1,
              ):
    """
    IDAMAX finds the index of the first element having maximum absolute value.
    """
    from pyccel.stdlib.internal.blas import izamax

    n = np.int32(x.shape[0])

    i = izamax (n, x, incx)
    # we must substruct 1 because of the fortran indexing
    i = i-1
    return i

# ==============================================================================
#
#                                  LEVEL 2
#
# ==============================================================================

# ==============================================================================
def blas_zgemv(alpha: 'complex128', a: 'complex128[:,:](order=F)', x: 'complex128[:]', y: 'complex128[:]',
               beta: 'complex128' = 0.,
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
    from pyccel.stdlib.internal.blas import zgemv

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    # ...
    flag_trans = 'N'
    if trans: flag_trans = 'T'
    # ...

    zgemv (flag_trans, m, n, alpha, a, lda, x, incx, beta, y, incy)

# ==============================================================================
def blas_zgbmv(kl : 'int32', ku: 'int32', alpha: 'complex128',
               a: 'complex128[:,:](order=F)', x: 'complex128[:]', y: 'complex128[:]',
               beta: 'complex128' = 0.,
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
    from pyccel.stdlib.internal.blas import zgbmv

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])

    lda = m
#    lda = np.int32(1) + ku + kl

    # ...
    flag_trans = 'N'
    if trans: flag_trans = 'T'
    # ...

    zgbmv (flag_trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy)

# ==============================================================================
def blas_zhemv(alpha: 'complex128', a: 'complex128[:,:](order=F)', x: 'complex128[:]', y: 'complex128[:]',
               beta: 'complex128' = 0.,
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
    from pyccel.stdlib.internal.blas import zhemv

    n = np.int32(a.shape[0])
    lda = n

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    zhemv (flag_uplo, n, alpha, a, lda, x, incx, beta, y, incy)

# ==============================================================================
def blas_zhbmv(k : 'int32', alpha: 'complex128',
               a: 'complex128[:,:](order=F)', x: 'complex128[:]', y: 'complex128[:]',
               beta: 'complex128' = 0.,
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
    from pyccel.stdlib.internal.blas import zhbmv

    n = np.int32(a.shape[0])
    lda = n

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    zhbmv (flag_uplo, n, k, alpha, a, lda, x, incx, beta, y, incy)

# ==============================================================================
def blas_zhpmv(alpha: 'complex128', a: 'complex128[:]', x: 'complex128[:]', y: 'complex128[:]',
               beta: 'complex128' = 0.,
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
    from pyccel.stdlib.internal.blas import zhpmv

    n = np.int32(x.shape[0])

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    zhpmv (flag_uplo, n, alpha, a, x, incx, beta, y, incy)

# ==============================================================================
def blas_ztrmv(a: 'complex128[:,:](order=F)', x: 'complex128[:]',
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
    from pyccel.stdlib.internal.blas import ztrmv

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

    ztrmv (flag_uplo, flag_trans, flag_diag, n, a, lda, x, incx)

# ==============================================================================
def blas_ztbmv(k : 'int32', a: 'complex128[:,:](order=F)', x: 'complex128[:]',
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
    from pyccel.stdlib.internal.blas import ztbmv

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

    ztbmv (flag_uplo, flag_trans, flag_diag, n, k, a, lda, x, incx)

# ==============================================================================
def blas_ztpmv(a: 'complex128[:]', x: 'complex128[:]',
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
    from pyccel.stdlib.internal.blas import ztpmv

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

    ztpmv (flag_uplo, flag_trans, flag_diag, n, a, x, incx)

# ==============================================================================
def blas_ztrsv(a: 'complex128[:,:](order=F)', x: 'complex128[:]',
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

    from pyccel.stdlib.internal.blas import ztrsv

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

    ztrsv (flag_uplo, flag_trans, flag_diag, n, a, lda, x, incx)

# ==============================================================================
def blas_ztbsv(k: 'int32', a: 'complex128[:,:](order=F)', x: 'complex128[:]',
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

    from pyccel.stdlib.internal.blas import ztbsv

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

    ztbsv (flag_uplo, flag_trans, flag_diag, n, k, a, lda, x, incx)

# ==============================================================================
def blas_ztpsv(a: 'complex128[:]', x: 'complex128[:]',
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

    from pyccel.stdlib.internal.blas import ztpsv

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

    ztpsv (flag_uplo, flag_trans, flag_diag, n, a, x, incx)

# ==============================================================================
def blas_zgeru(alpha: 'complex128', x: 'complex128[:]', y: 'complex128[:]', a: 'complex128[:,:](order=F)',
              incx: 'int32' = 1,
              incy: 'int32' = 1,
              ):
    """
    CGERU  performs the rank 1 operation

    A := alpha*x*y**T + A,

    where alpha is a scalar, x is an m element vector, y is an n element
    vector and A is an m by n matrix.
    """
    from pyccel.stdlib.internal.blas import zgeru

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    zgeru (m, n, alpha, x, incx, y, incy, a, lda)

# ==============================================================================
def blas_zgerc(alpha: 'complex128', x: 'complex128[:]', y: 'complex128[:]', a: 'complex128[:,:](order=F)',
              incx: 'int32' = 1,
              incy: 'int32' = 1,
              ):
    """
    CGERC  performs the rank 1 operation

    A := alpha*x*y**H + A,

    where alpha is a scalar, x is an m element vector, y is an n element
    vector and A is an m by n matrix.
    """
    from pyccel.stdlib.internal.blas import zgerc

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    zgerc (m, n, alpha, x, incx, y, incy, a, lda)

# ==============================================================================
def blas_zher(alpha: 'float64', x: 'complex128[:]', a: 'complex128[:,:](order=F)',
              incx: 'int32' = 1,
              lower: 'bool' = False
              ):
    """
    CHER   performs the hermitian rank 1 operation

    A := alpha*x*x**H + A,

    where alpha is a real scalar, x is an n element vector and A is an
    n by n hermitian matrix.
    """
    from pyccel.stdlib.internal.blas import zher

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    zher (flag_uplo, n, alpha, x, incx, a, lda)

# ==============================================================================
def blas_zhpr(alpha: 'float64', x: 'complex128[:]', a: 'complex128[:]',
              incx: 'int32' = 1,
              lower: 'bool' = False
              ):
    """
    CHPR    performs the hermitian rank 1 operation

    A := alpha*x*x**H + A,

    where alpha is a real scalar, x is an n element vector and A is an
    n by n hermitian matrix, supplied in packed form.
    """
    from pyccel.stdlib.internal.blas import zhpr

    n = np.int32(x.shape[0])

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    zhpr (flag_uplo, n, alpha, x, incx, a)

# ==============================================================================
def blas_zher2(alpha: 'complex128', x: 'complex128[:]', y: 'complex128[:]', a: 'complex128[:,:](order=F)',
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
    from pyccel.stdlib.internal.blas import zher2

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    zher2 (flag_uplo, n, alpha, x, incx, y, incy, a, lda)

# ==============================================================================
def blas_zhpr2(alpha: 'complex128', x: 'complex128[:]', y: 'complex128[:]', a: 'complex128[:]',
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
    from pyccel.stdlib.internal.blas import zhpr2

    n = np.int32(x.shape[0])

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    zhpr2 (flag_uplo, n, alpha, x, incx, y, incy, a)

# ==============================================================================
#
#                                  LEVEL 3
#
# ==============================================================================

# ==============================================================================
def blas_zgemm(alpha: 'complex128', a: 'complex128[:,:](order=F)', b: 'complex128[:,:](order=F)', c: 'complex128[:,:](order=F)',
               beta: 'complex128' = 0.,
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
    from pyccel.stdlib.internal.blas import zgemm

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

    zgemm (flag_trans_a, flag_trans_b, l, n, m, alpha, a, lda, b, ldb, beta, c, ldc)

# ==============================================================================
def blas_zsymm(alpha: 'complex128', a: 'complex128[:,:](order=F)', b: 'complex128[:,:](order=F)', c: 'complex128[:,:](order=F)',
               beta: 'complex128' = 0.,
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
    from pyccel.stdlib.internal.blas import zsymm

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

    zsymm (flag_side, flag_uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc)

# ==============================================================================
def blas_zhemm(alpha: 'complex128', a: 'complex128[:,:](order=F)', b: 'complex128[:,:](order=F)', c: 'complex128[:,:](order=F)',
               beta: 'complex128' = 0.,
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
    from pyccel.stdlib.internal.blas import zhemm

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

    zhemm (flag_side, flag_uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc)

# ==============================================================================
def blas_zsyrk(alpha: 'complex128', a: 'complex128[:,:](order=F)', c: 'complex128[:,:](order=F)',
               beta: 'complex128' = 0.,
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
    from pyccel.stdlib.internal.blas import zsyrk

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

    zsyrk (flag_uplo, flag_trans, n, k, alpha, a, lda, beta, c, ldc)

# ==============================================================================
def blas_zsyr2k(alpha: 'complex128', a: 'complex128[:,:](order=F)', b: 'complex128[:,:](order=F)', c: 'complex128[:,:](order=F)',
               beta: 'complex128' = 0.,
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
    from pyccel.stdlib.internal.blas import zsyr2k

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

    zsyr2k (flag_uplo, flag_trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc)

# ==============================================================================
def blas_zherk(alpha: 'complex128', a: 'complex128[:,:](order=F)', c: 'complex128[:,:](order=F)',
               beta: 'complex128' = 0.,
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
    from pyccel.stdlib.internal.blas import zherk

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

    zherk (flag_uplo, flag_trans, n, k, alpha, a, lda, beta, c, ldc)

# ==============================================================================
def blas_zher2k(alpha: 'complex128', a: 'complex128[:,:](order=F)', b: 'complex128[:,:](order=F)', c: 'complex128[:,:](order=F)',
               beta: 'complex128' = 0.,
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
    from pyccel.stdlib.internal.blas import zher2k

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

    zher2k (flag_uplo, flag_trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc)

# ==============================================================================
def blas_ztrmm(alpha: 'complex128', a: 'complex128[:,:](order=F)', b: 'complex128[:,:](order=F)',
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
    from pyccel.stdlib.internal.blas import ztrmm

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

    ztrmm (flag_side, flag_uplo, flag_trans_a, flag_diag, m, n, alpha, a, lda, b, ldb)

# ==============================================================================
def blas_ztrsm(alpha: 'complex128', a: 'complex128[:,:](order=F)', b: 'complex128[:,:](order=F)',
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
    from pyccel.stdlib.internal.blas import ztrsm

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

    ztrsm (flag_side, flag_uplo, flag_trans_a, flag_diag, m, n, alpha, a, lda, b, ldb)
