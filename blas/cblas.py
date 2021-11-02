"""
The aim of this file is to provide functions that can be either:
    - called as they are
    - called as inlined statements
"""

import numpy as np
import pyccel.stdlib.internal.blas as mod_blas

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

    n = np.int32(x.shape[0])

    mod_blas.ccopy (n, x, incx, y, incy)

# ==============================================================================
def blas_cswap(x: 'complex64[:]', y: 'complex64[:]',
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):

    n = np.int32(x.shape[0])

    mod_blas.cswap (n, x, incx, y, incy)

# ==============================================================================
def blas_cscal(alpha: 'complex64', x: 'complex64[:]',
               incx: 'int32' = 1,
              ):

    n = np.int32(x.shape[0])

    mod_blas.cscal (n, alpha, x, incx)

# ==============================================================================
def blas_cdotc(x: 'complex64[:]', y: 'complex64[:]',
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):

    n = np.int32(x.shape[0])

    return mod_blas.cdotc (n, x, incx, y, incy)

# ==============================================================================
def blas_cdotu(x: 'complex64[:]', y: 'complex64[:]',
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):

    n = np.int32(x.shape[0])

    return mod_blas.cdotu (n, x, incx, y, incy)

# ==============================================================================
def blas_caxpy(x: 'complex64[:]', y: 'complex64[:]',
               a: 'complex64' = 1.,
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):

    n = np.int32(x.shape[0])

    mod_blas.caxpy (n, a, x, incx, y, incy)

# ==============================================================================
def blas_scnrm2(x: 'complex64[:]',
               incx: 'int32' = 1,
              ):

    n = np.int32(x.shape[0])

    return mod_blas.scnrm2 (n, x, incx)

# ==============================================================================
def blas_scasum(x: 'complex64[:]',
               incx: 'int32' = 1,
              ):

    n = np.int32(x.shape[0])

    return mod_blas.scasum (n, x, incx)

# ==============================================================================
def blas_icamax(x: 'complex64[:]',
               incx: 'int32' = 1,
              ):

    n = np.int32(x.shape[0])

    i = mod_blas.icamax (n, x, incx)
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

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    # ...
    flag_trans = 'N'
    if trans: flag_trans = 'T'
    # ...

    mod_blas.cgemv (flag_trans, m, n, alpha, a, lda, x, incx, beta, y, incy)

# ==============================================================================
def blas_cgbmv(kl : 'int32', ku: 'int32', alpha: 'complex64',
               a: 'complex64[:,:](order=F)', x: 'complex64[:]', y: 'complex64[:]',
               beta: 'complex64' = 0.,
               incx: 'int32' = 1,
               incy: 'int32' = 1,
               trans: 'bool' = False
              ):

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])

    lda = m
#    lda = np.int32(1) + ku + kl

    # ...
    flag_trans = 'N'
    if trans: flag_trans = 'T'
    # ...

    mod_blas.cgbmv (flag_trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy)

# ==============================================================================
def blas_chemv(alpha: 'complex64', a: 'complex64[:,:](order=F)', x: 'complex64[:]', y: 'complex64[:]',
               beta: 'complex64' = 0.,
               incx: 'int32' = 1,
               incy: 'int32' = 1,
               lower: 'bool' = False
              ):

    n = np.int32(a.shape[0])
    lda = n

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    mod_blas.chemv (flag_uplo, n, alpha, a, lda, x, incx, beta, y, incy)

# ==============================================================================
def blas_chbmv(k : 'int32', alpha: 'complex64',
               a: 'complex64[:,:](order=F)', x: 'complex64[:]', y: 'complex64[:]',
               beta: 'complex64' = 0.,
               incx: 'int32' = 1,
               incy: 'int32' = 1,
               lower: 'bool' = False
              ):

    n = np.int32(a.shape[0])
    lda = n

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    mod_blas.chbmv (flag_uplo, n, k, alpha, a, lda, x, incx, beta, y, incy)

# ==============================================================================
def blas_chpmv(alpha: 'complex64', a: 'complex64[:]', x: 'complex64[:]', y: 'complex64[:]',
               beta: 'complex64' = 0.,
               incx: 'int32' = 1,
               incy: 'int32' = 1,
               lower: 'bool' = False
              ):

    n = np.int32(x.shape[0])

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    mod_blas.chpmv (flag_uplo, n, alpha, a, x, incx, beta, y, incy)

# ==============================================================================
def blas_ctrmv(a: 'complex64[:,:](order=F)', x: 'complex64[:]',
               incx: 'int32' = 1,
               lower: 'bool' = False,
               trans: 'bool' = False,
               diag: 'bool' = False
              ):

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

    mod_blas.ctrmv (flag_uplo, flag_trans, flag_diag, n, a, lda, x, incx)

# ==============================================================================
def blas_ctbmv(k : 'int32', a: 'complex64[:,:](order=F)', x: 'complex64[:]',
               incx: 'int32' = 1,
               lower: 'bool' = False,
               trans: 'bool' = False,
               diag: 'bool' = False
              ):

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

    mod_blas.ctbmv (flag_uplo, flag_trans, flag_diag, n, k, a, lda, x, incx)

# ==============================================================================
def blas_ctpmv(a: 'complex64[:]', x: 'complex64[:]',
               incx: 'int32' = 1,
               lower: 'bool' = False,
               trans: 'bool' = False,
               diag: 'bool' = False
              ):

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

    mod_blas.ctpmv (flag_uplo, flag_trans, flag_diag, n, a, x, incx)

# ==============================================================================
def blas_ctrsv(a: 'complex64[:,:](order=F)', x: 'complex64[:]',
               incx: 'int32' = 1,
               lower: 'bool' = False,
               trans: 'bool' = False,
               diag: 'bool' = False
              ):

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

    mod_blas.ctrsv (flag_uplo, flag_trans, flag_diag, n, a, lda, x, incx)

# ==============================================================================
def blas_ctbsv(k: 'int32', a: 'complex64[:,:](order=F)', x: 'complex64[:]',
               incx: 'int32' = 1,
               lower: 'bool' = False,
               trans: 'bool' = False,
               diag: 'bool' = False
              ):

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

    mod_blas.ctbsv (flag_uplo, flag_trans, flag_diag, n, k, a, lda, x, incx)

# ==============================================================================
def blas_ctpsv(a: 'complex64[:]', x: 'complex64[:]',
               incx: 'int32' = 1,
               lower: 'bool' = False,
               trans: 'bool' = False,
               diag: 'bool' = False
              ):

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

    mod_blas.ctpsv (flag_uplo, flag_trans, flag_diag, n, a, x, incx)

# ==============================================================================
def blas_cgeru(alpha: 'complex64', x: 'complex64[:]', y: 'complex64[:]', a: 'complex64[:,:](order=F)',
              incx: 'int32' = 1,
              incy: 'int32' = 1,
              ):

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    mod_blas.cgeru (m, n, alpha, x, incx, y, incy, a, lda)

# ==============================================================================
def blas_cgerc(alpha: 'complex64', x: 'complex64[:]', y: 'complex64[:]', a: 'complex64[:,:](order=F)',
              incx: 'int32' = 1,
              incy: 'int32' = 1,
              ):

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    mod_blas.cgerc (m, n, alpha, x, incx, y, incy, a, lda)

# ==============================================================================
def blas_cher(alpha: 'float32', x: 'complex64[:]', a: 'complex64[:,:](order=F)',
              incx: 'int32' = 1,
              lower: 'bool' = False
              ):

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    mod_blas.cher (flag_uplo, n, alpha, x, incx, a, lda)

# ==============================================================================
def blas_chpr(alpha: 'float32', x: 'complex64[:]', a: 'complex64[:]',
              incx: 'int32' = 1,
              lower: 'bool' = False
              ):

    n = np.int32(x.shape[0])

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    mod_blas.chpr (flag_uplo, n, alpha, x, incx, a)

# ==============================================================================
def blas_cher2(alpha: 'complex64', x: 'complex64[:]', y: 'complex64[:]', a: 'complex64[:,:](order=F)',
              incx: 'int32' = 1,
              incy: 'int32' = 1,
              lower: 'bool' = False
              ):

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    mod_blas.cher2 (flag_uplo, n, alpha, x, incx, y, incy, a, lda)

# ==============================================================================
def blas_chpr2(alpha: 'complex64', x: 'complex64[:]', y: 'complex64[:]', a: 'complex64[:]',
              incx: 'int32' = 1,
              incy: 'int32' = 1,
              lower: 'bool' = False
              ):

    n = np.int32(x.shape[0])

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    mod_blas.chpr2 (flag_uplo, n, alpha, x, incx, y, incy, a)

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

    mod_blas.cgemm (flag_trans_a, flag_trans_b, l, n, m, alpha, a, lda, b, ldb, beta, c, ldc)

# ==============================================================================
def blas_csymm(alpha: 'complex64', a: 'complex64[:,:](order=F)', b: 'complex64[:,:](order=F)', c: 'complex64[:,:](order=F)',
               beta: 'complex64' = 0.,
               side: 'bool' = False,
               lower: 'bool' = False,
              ):

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

    mod_blas.csymm (flag_side, flag_uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc)

# ==============================================================================
def blas_chemm(alpha: 'complex64', a: 'complex64[:,:](order=F)', b: 'complex64[:,:](order=F)', c: 'complex64[:,:](order=F)',
               beta: 'complex64' = 0.,
               side: 'bool' = False,
               lower: 'bool' = False,
              ):

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

    mod_blas.chemm (flag_side, flag_uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc)

# ==============================================================================
def blas_csyrk(alpha: 'complex64', a: 'complex64[:,:](order=F)', c: 'complex64[:,:](order=F)',
               beta: 'complex64' = 0.,
               lower: 'bool' = False,
               trans: 'bool' = False,
              ):

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

    mod_blas.csyrk (flag_uplo, flag_trans, n, k, alpha, a, lda, beta, c, ldc)

# ==============================================================================
def blas_csyr2k(alpha: 'complex64', a: 'complex64[:,:](order=F)', b: 'complex64[:,:](order=F)', c: 'complex64[:,:](order=F)',
               beta: 'complex64' = 0.,
               lower: 'bool' = False,
               trans: 'bool' = False,
              ):

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

    mod_blas.csyr2k (flag_uplo, flag_trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc)

# ==============================================================================
def blas_cherk(alpha: 'complex64', a: 'complex64[:,:](order=F)', c: 'complex64[:,:](order=F)',
               beta: 'complex64' = 0.,
               lower: 'bool' = False,
               trans: 'bool' = False,
              ):

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

    mod_blas.cherk (flag_uplo, flag_trans, n, k, alpha, a, lda, beta, c, ldc)

# ==============================================================================
def blas_cher2k(alpha: 'complex64', a: 'complex64[:,:](order=F)', b: 'complex64[:,:](order=F)', c: 'complex64[:,:](order=F)',
               beta: 'complex64' = 0.,
               lower: 'bool' = False,
               trans: 'bool' = False,
              ):

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

    mod_blas.cher2k (flag_uplo, flag_trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc)

# ==============================================================================
def blas_ctrmm(alpha: 'complex64', a: 'complex64[:,:](order=F)', b: 'complex64[:,:](order=F)',
               side: 'bool' = False,
               lower: 'bool' = False,
               trans_a: 'bool' = False,
               diag: 'bool' = False
              ):

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

    mod_blas.ctrmm (flag_side, flag_uplo, flag_trans_a, flag_diag, m, n, alpha, a, lda, b, ldb)

# ==============================================================================
def blas_ctrsm(alpha: 'complex64', a: 'complex64[:,:](order=F)', b: 'complex64[:,:](order=F)',
               side: 'bool' = False,
               lower: 'bool' = False,
               trans_a: 'bool' = False,
               diag: 'bool' = False
              ):

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

    mod_blas.ctrsm (flag_side, flag_uplo, flag_trans_a, flag_diag, m, n, alpha, a, lda, b, ldb)
