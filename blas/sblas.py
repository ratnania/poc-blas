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
def blas_srotg(a: 'float32', b: 'float32',
               c: 'float32' = 0.,
               s: 'float32' = 0.,
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    mod_blas.srotg (a, b, c, s)

    return c, s

# ==============================================================================
def blas_srotmg(d1: 'float32', d2: 'float32', x1: 'float32', y1: 'float32',
                param: 'float32[:]'):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    mod_blas.srotmg (d1, d2, x1, y1, param)

# ==============================================================================
def blas_srot(x: 'float32[:]', y: 'float32[:]', c: 'float32', s: 'float32',
              incx: 'int32' = 1,
              incy: 'int32' = 1
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    mod_blas.srot (n, x, incx, y, incy, c, s)

# ==============================================================================
def blas_srotm(x: 'float32[:]', y: 'float32[:]', param: 'float32[:]',
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    mod_blas.srotm (n, x, incx, y, incy, param)

# ==============================================================================
def blas_scopy(x: 'float32[:]', y: 'float32[:]',
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    mod_blas.scopy (n, x, incx, y, incy)

# ==============================================================================
def blas_sswap(x: 'float32[:]', y: 'float32[:]',
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    mod_blas.sswap (n, x, incx, y, incy)

# ==============================================================================
def blas_sscal(alpha: 'float32', x: 'float32[:]',
               incx: 'int32' = 1,
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    mod_blas.sscal (n, alpha, x, incx)

# ==============================================================================
def blas_sdot(x: 'float32[:]', y: 'float32[:]',
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    return mod_blas.sdot (n, x, incx, y, incy)

# ==============================================================================
def blas_sdsdot(sb: 'float32', x: 'float32[:]', y: 'float32[:]',
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    return mod_blas.sdsdot (n, sb, x, incx, y, incy)

# ==============================================================================
def blas_dsdot(x: 'float32[:]', y: 'float32[:]',
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    return mod_blas.dsdot (n, x, incx, y, incy)

# ==============================================================================
def blas_saxpy(x: 'float32[:]', y: 'float32[:]',
               a: 'float32' = 1.,
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    mod_blas.saxpy (n, a, x, incx, y, incy)

# ==============================================================================
def blas_snrm2(x: 'float32[:]',
               incx: 'int32' = 1,
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    return mod_blas.snrm2 (n, x, incx)

# ==============================================================================
def blas_sasum(x: 'float32[:]',
               incx: 'int32' = 1,
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    return mod_blas.sasum (n, x, incx)

# ==============================================================================
def blas_isamax(x: 'float32[:]',
               incx: 'int32' = 1,
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    n = np.int32(x.shape[0])

    i = mod_blas.isamax (n, x, incx)
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

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    # ...
    flag_trans = 'N'
    if trans: flag_trans = 'T'
    # ...

    mod_blas.sgemv (flag_trans, m, n, alpha, a, lda, x, incx, beta, y, incy)

# ==============================================================================
def blas_sgbmv(kl : 'int32', ku: 'int32', alpha: 'float32',
               a: 'float32[:,:](order=F)', x: 'float32[:]', y: 'float32[:]',
               beta: 'float32' = 0.,
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

    mod_blas.sgbmv (flag_trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy)

# ==============================================================================
def blas_ssymv(alpha: 'float32', a: 'float32[:,:](order=F)', x: 'float32[:]', y: 'float32[:]',
               beta: 'float32' = 0.,
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

    mod_blas.ssymv (flag_uplo, n, alpha, a, lda, x, incx, beta, y, incy)

# ==============================================================================
def blas_ssbmv(k : 'int32', alpha: 'float32',
               a: 'float32[:,:](order=F)', x: 'float32[:]', y: 'float32[:]',
               beta: 'float32' = 0.,
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

    mod_blas.ssbmv (flag_uplo, n, k, alpha, a, lda, x, incx, beta, y, incy)

# ==============================================================================
def blas_sspmv(alpha: 'float32', a: 'float32[:]', x: 'float32[:]', y: 'float32[:]',
               beta: 'float32' = 0.,
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

    mod_blas.sspmv (flag_uplo, n, alpha, a, x, incx, beta, y, incy)

# ==============================================================================
def blas_strmv(a: 'float32[:,:](order=F)', x: 'float32[:]',
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

    mod_blas.strmv (flag_uplo, flag_trans, flag_diag, n, a, lda, x, incx)

# ==============================================================================
def blas_stbmv(k : 'int32', a: 'float32[:,:](order=F)', x: 'float32[:]',
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

    mod_blas.stbmv (flag_uplo, flag_trans, flag_diag, n, k, a, lda, x, incx)

# ==============================================================================
def blas_stpmv(a: 'float32[:]', x: 'float32[:]',
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

    mod_blas.stpmv (flag_uplo, flag_trans, flag_diag, n, a, x, incx)

# ==============================================================================
def blas_strsv(a: 'float32[:,:](order=F)', x: 'float32[:]',
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

    mod_blas.strsv (flag_uplo, flag_trans, flag_diag, n, a, lda, x, incx)

# ==============================================================================
def blas_stbsv(k: 'int32', a: 'float32[:,:](order=F)', x: 'float32[:]',
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

    mod_blas.stbsv (flag_uplo, flag_trans, flag_diag, n, k, a, lda, x, incx)

# ==============================================================================
def blas_stpsv(a: 'float32[:]', x: 'float32[:]',
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

    mod_blas.stpsv (flag_uplo, flag_trans, flag_diag, n, a, x, incx)

# ==============================================================================
def blas_sger(alpha: 'float32', x: 'float32[:]', y: 'float32[:]', a: 'float32[:,:](order=F)',
              incx: 'int32' = 1,
              incy: 'int32' = 1,
              ):

    import numpy as np
    import pyccel.stdlib.internal.blas as mod_blas

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    mod_blas.sger (m, n, alpha, x, incx, y, incy, a, lda)

# ==============================================================================
def blas_ssyr(alpha: 'float32', x: 'float32[:]', a: 'float32[:,:](order=F)',
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

    mod_blas.ssyr (flag_uplo, n, alpha, x, incx, a, lda)

# ==============================================================================
def blas_sspr(alpha: 'float32', x: 'float32[:]', a: 'float32[:]',
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

    mod_blas.sspr (flag_uplo, n, alpha, x, incx, a)

# ==============================================================================
def blas_ssyr2(alpha: 'float32', x: 'float32[:]', y: 'float32[:]', a: 'float32[:,:](order=F)',
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

    mod_blas.ssyr2 (flag_uplo, n, alpha, x, incx, y, incy, a, lda)

# ==============================================================================
def blas_sspr2(alpha: 'float32', x: 'float32[:]', y: 'float32[:]', a: 'float32[:]',
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

    mod_blas.sspr2 (flag_uplo, n, alpha, x, incx, y, incy, a)

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

    mod_blas.sgemm (flag_trans_a, flag_trans_b, l, n, m, alpha, a, lda, b, ldb, beta, c, ldc)

# ==============================================================================
def blas_ssymm(alpha: 'float32', a: 'float32[:,:](order=F)', b: 'float32[:,:](order=F)', c: 'float32[:,:](order=F)',
               beta: 'float32' = 0.,
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

    mod_blas.ssymm (flag_side, flag_uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc)

# ==============================================================================
def blas_ssyrk(alpha: 'float32', a: 'float32[:,:](order=F)', c: 'float32[:,:](order=F)',
               beta: 'float32' = 0.,
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

    mod_blas.ssyrk (flag_uplo, flag_trans, n, k, alpha, a, lda, beta, c, ldc)

# ==============================================================================
def blas_ssyr2k(alpha: 'float32', a: 'float32[:,:](order=F)', b: 'float32[:,:](order=F)', c: 'float32[:,:](order=F)',
               beta: 'float32' = 0.,
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

    mod_blas.ssyr2k (flag_uplo, flag_trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc)

# ==============================================================================
def blas_strmm(alpha: 'float32', a: 'float32[:,:](order=F)', b: 'float32[:,:](order=F)',
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

    mod_blas.strmm (flag_side, flag_uplo, flag_trans_a, flag_diag, m, n, alpha, a, lda, b, ldb)

# ==============================================================================
def blas_strsm(alpha: 'float32', a: 'float32[:,:](order=F)', b: 'float32[:,:](order=F)',
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

    mod_blas.strsm (flag_side, flag_uplo, flag_trans_a, flag_diag, m, n, alpha, a, lda, b, ldb)
