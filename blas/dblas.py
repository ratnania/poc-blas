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

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    mod_blas.drotg (a, b, c, s)

    return c, s

# ==============================================================================
def blas_drotmg(d1: 'float64', d2: 'float64', x1: 'float64', y1: 'float64',
                param: 'float64[:]'):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    mod_blas.drotmg (d1, d2, x1, y1, param)

# ==============================================================================
def blas_drot(x: 'float64[:]', y: 'float64[:]', c: 'float64', s: 'float64',
              incx: 'int32' = 1,
              incy: 'int32' = 1
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    mod_blas.drot (n, x, incx, y, incy, c, s)

# ==============================================================================
def blas_drotm(x: 'float64[:]', y: 'float64[:]', param: 'float64[:]',
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    mod_blas.drotm (n, x, incx, y, incy, param)

# ==============================================================================
def blas_dcopy(x: 'float64[:]', y: 'float64[:]',
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    mod_blas.dcopy (n, x, incx, y, incy)

# ==============================================================================
def blas_dswap(x: 'float64[:]', y: 'float64[:]',
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    mod_blas.dswap (n, x, incx, y, incy)

# ==============================================================================
def blas_dscal(alpha: 'float64', x: 'float64[:]',
               incx: 'int32' = 1,
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    mod_blas.dscal (n, alpha, x, incx)

# ==============================================================================
def blas_ddot(x: 'float64[:]', y: 'float64[:]',
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    return mod_blas.ddot (n, x, incx, y, incy)

# ==============================================================================
def blas_daxpy(x: 'float64[:]', y: 'float64[:]',
               a: 'float64' = 1.,
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    mod_blas.daxpy (n, a, x, incx, y, incy)

# ==============================================================================
def blas_dnrm2(x: 'float64[:]',
               incx: 'int32' = 1,
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    return mod_blas.dnrm2 (n, x, incx)

# ==============================================================================
def blas_dasum(x: 'float64[:]',
               incx: 'int32' = 1,
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    return mod_blas.dasum (n, x, incx)

# ==============================================================================
def blas_idamax(x: 'float64[:]',
               incx: 'int32' = 1,
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    i = mod_blas.idamax (n, x, incx)
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

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    # ...
    flag_trans = 'N'
    if trans: flag_trans = 'T'
    # ...

    mod_blas.dgemv (flag_trans, m, n, alpha, a, lda, x, incx, beta, y, incy)

# ==============================================================================
def blas_dgbmv(kl : 'int32', ku: 'int32', alpha: 'float64',
               a: 'float64[:,:](order=F)', x: 'float64[:]', y: 'float64[:]',
               beta: 'float64' = 0.,
               incx: 'int32' = 1,
               incy: 'int32' = 1,
               trans: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])

    lda = m
#    lda = np.int32(1) + ku + kl

    # ...
    flag_trans = 'N'
    if trans: flag_trans = 'T'
    # ...

    mod_blas.dgbmv (flag_trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy)

# ==============================================================================
def blas_dsymv(alpha: 'float64', a: 'float64[:,:](order=F)', x: 'float64[:]', y: 'float64[:]',
               beta: 'float64' = 0.,
               incx: 'int32' = 1,
               incy: 'int32' = 1,
               lower: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    mod_blas.dsymv (flag_uplo, n, alpha, a, lda, x, incx, beta, y, incy)

# ==============================================================================
def blas_dsbmv(k : 'int32', alpha: 'float64',
               a: 'float64[:,:](order=F)', x: 'float64[:]', y: 'float64[:]',
               beta: 'float64' = 0.,
               incx: 'int32' = 1,
               incy: 'int32' = 1,
               lower: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    mod_blas.dsbmv (flag_uplo, n, k, alpha, a, lda, x, incx, beta, y, incy)

# ==============================================================================
def blas_dspmv(alpha: 'float64', a: 'float64[:]', x: 'float64[:]', y: 'float64[:]',
               beta: 'float64' = 0.,
               incx: 'int32' = 1,
               incy: 'int32' = 1,
               lower: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    mod_blas.dspmv (flag_uplo, n, alpha, a, x, incx, beta, y, incy)

# ==============================================================================
def blas_dtrmv(a: 'float64[:,:](order=F)', x: 'float64[:]',
               incx: 'int32' = 1,
               lower: 'bool' = False,
               trans: 'bool' = False,
               diag: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

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

    mod_blas.dtrmv (flag_uplo, flag_trans, flag_diag, n, a, lda, x, incx)

# ==============================================================================
def blas_dtbmv(k : 'int32', a: 'float64[:,:](order=F)', x: 'float64[:]',
               incx: 'int32' = 1,
               lower: 'bool' = False,
               trans: 'bool' = False,
               diag: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

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

    mod_blas.dtbmv (flag_uplo, flag_trans, flag_diag, n, k, a, lda, x, incx)

# ==============================================================================
def blas_dtpmv(a: 'float64[:]', x: 'float64[:]',
               incx: 'int32' = 1,
               lower: 'bool' = False,
               trans: 'bool' = False,
               diag: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

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

    mod_blas.dtpmv (flag_uplo, flag_trans, flag_diag, n, a, x, incx)

# ==============================================================================
def blas_dtrsv(a: 'float64[:,:](order=F)', x: 'float64[:]',
               incx: 'int32' = 1,
               lower: 'bool' = False,
               trans: 'bool' = False,
               diag: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

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

    mod_blas.dtrsv (flag_uplo, flag_trans, flag_diag, n, a, lda, x, incx)

# ==============================================================================
def blas_dtbsv(k: 'int32', a: 'float64[:,:](order=F)', x: 'float64[:]',
               incx: 'int32' = 1,
               lower: 'bool' = False,
               trans: 'bool' = False,
               diag: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

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

    mod_blas.dtbsv (flag_uplo, flag_trans, flag_diag, n, k, a, lda, x, incx)

# ==============================================================================
def blas_dtpsv(a: 'float64[:]', x: 'float64[:]',
               incx: 'int32' = 1,
               lower: 'bool' = False,
               trans: 'bool' = False,
               diag: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

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

    mod_blas.dtpsv (flag_uplo, flag_trans, flag_diag, n, a, x, incx)

# ==============================================================================
def blas_dger(alpha: 'float64', x: 'float64[:]', y: 'float64[:]', a: 'float64[:,:](order=F)',
              incx: 'int32' = 1,
              incy: 'int32' = 1,
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    mod_blas.dger (m, n, alpha, x, incx, y, incy, a, lda)

# ==============================================================================
def blas_dsyr(alpha: 'float64', x: 'float64[:]', a: 'float64[:,:](order=F)',
              incx: 'int32' = 1,
              lower: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    mod_blas.dsyr (flag_uplo, n, alpha, x, incx, a, lda)

# ==============================================================================
def blas_dspr(alpha: 'float64', x: 'float64[:]', a: 'float64[:]',
              incx: 'int32' = 1,
              lower: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    mod_blas.dspr (flag_uplo, n, alpha, x, incx, a)

# ==============================================================================
def blas_dsyr2(alpha: 'float64', x: 'float64[:]', y: 'float64[:]', a: 'float64[:,:](order=F)',
              incx: 'int32' = 1,
              incy: 'int32' = 1,
              lower: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    mod_blas.dsyr2 (flag_uplo, n, alpha, x, incx, y, incy, a, lda)

# ==============================================================================
def blas_dspr2(alpha: 'float64', x: 'float64[:]', y: 'float64[:]', a: 'float64[:]',
              incx: 'int32' = 1,
              incy: 'int32' = 1,
              lower: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    mod_blas.dspr2 (flag_uplo, n, alpha, x, incx, y, incy, a)

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

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

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

    mod_blas.dgemm (flag_trans_a, flag_trans_b, l, n, m, alpha, a, lda, b, ldb, beta, c, ldc)

# ==============================================================================
def blas_dsymm(alpha: 'float64', a: 'float64[:,:](order=F)', b: 'float64[:,:](order=F)', c: 'float64[:,:](order=F)',
               beta: 'float64' = 0.,
               side: 'bool' = False,
               lower: 'bool' = False,
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

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

    mod_blas.dsymm (flag_side, flag_uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc)

# ==============================================================================
def blas_dsyrk(alpha: 'float64', a: 'float64[:,:](order=F)', c: 'float64[:,:](order=F)',
               beta: 'float64' = 0.,
               lower: 'bool' = False,
               trans: 'bool' = False,
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

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

    mod_blas.dsyrk (flag_uplo, flag_trans, n, k, alpha, a, lda, beta, c, ldc)

# ==============================================================================
def blas_dsyr2k(alpha: 'float64', a: 'float64[:,:](order=F)', b: 'float64[:,:](order=F)', c: 'float64[:,:](order=F)',
               beta: 'float64' = 0.,
               lower: 'bool' = False,
               trans: 'bool' = False,
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

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

    mod_blas.dsyr2k (flag_uplo, flag_trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc)

# ==============================================================================
def blas_dtrmm(alpha: 'float64', a: 'float64[:,:](order=F)', b: 'float64[:,:](order=F)',
               side: 'bool' = False,
               lower: 'bool' = False,
               trans_a: 'bool' = False,
               diag: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

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

    mod_blas.dtrmm (flag_side, flag_uplo, flag_trans_a, flag_diag, m, n, alpha, a, lda, b, ldb)

# ==============================================================================
def blas_dtrsm(alpha: 'float64', a: 'float64[:,:](order=F)', b: 'float64[:,:](order=F)',
               side: 'bool' = False,
               lower: 'bool' = False,
               trans_a: 'bool' = False,
               diag: 'bool' = False
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

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

    mod_blas.dtrsm (flag_side, flag_uplo, flag_trans_a, flag_diag, m, n, alpha, a, lda, b, ldb)
