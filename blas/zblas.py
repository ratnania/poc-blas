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
def blas_zcopy(x: 'complex128[:]', y: 'complex128[:]',
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):

    n = np.int32(x.shape[0])

    mod_blas.zcopy (n, x, incx, y, incy)

# ==============================================================================
def blas_zswap(x: 'complex128[:]', y: 'complex128[:]',
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):

    n = np.int32(x.shape[0])

    mod_blas.zswap (n, x, incx, y, incy)

# ==============================================================================
def blas_zscal(alpha: 'complex128', x: 'complex128[:]',
               incx: 'int32' = 1,
              ):

    n = np.int32(x.shape[0])

    mod_blas.zscal (n, alpha, x, incx)

# ==============================================================================
def blas_zdotc(x: 'complex128[:]', y: 'complex128[:]',
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):

    n = np.int32(x.shape[0])

    return mod_blas.zdotc (n, x, incx, y, incy)

# ==============================================================================
def blas_zdotu(x: 'complex128[:]', y: 'complex128[:]',
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):

    n = np.int32(x.shape[0])

    return mod_blas.zdotu (n, x, incx, y, incy)

# ==============================================================================
def blas_zaxpy(x: 'complex128[:]', y: 'complex128[:]',
               a: 'complex128' = 1.,
               incx: 'int32' = 1,
               incy: 'int32' = 1
              ):

    n = np.int32(x.shape[0])

    mod_blas.zaxpy (n, a, x, incx, y, incy)

# ==============================================================================
def blas_dznrm2(x: 'complex128[:]',
               incx: 'int32' = 1,
              ):

    n = np.int32(x.shape[0])

    return mod_blas.dznrm2 (n, x, incx)

# ==============================================================================
def blas_dzasum(x: 'complex128[:]',
               incx: 'int32' = 1,
              ):

    n = np.int32(x.shape[0])

    return mod_blas.dzasum (n, x, incx)

# ==============================================================================
def blas_izamax(x: 'complex128[:]',
               incx: 'int32' = 1,
              ):

    n = np.int32(x.shape[0])

    i = mod_blas.izamax (n, x, incx)
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

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    # ...
    flag_trans = 'N'
    if trans: flag_trans = 'T'
    # ...

    mod_blas.zgemv (flag_trans, m, n, alpha, a, lda, x, incx, beta, y, incy)

# ==============================================================================
def blas_zgbmv(kl : 'int32', ku: 'int32', alpha: 'complex128',
               a: 'complex128[:,:](order=F)', x: 'complex128[:]', y: 'complex128[:]',
               beta: 'complex128' = 0.,
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

    mod_blas.zgbmv (flag_trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy)

# ==============================================================================
def blas_zhemv(alpha: 'complex128', a: 'complex128[:,:](order=F)', x: 'complex128[:]', y: 'complex128[:]',
               beta: 'complex128' = 0.,
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

    mod_blas.zhemv (flag_uplo, n, alpha, a, lda, x, incx, beta, y, incy)

# ==============================================================================
def blas_zhbmv(k : 'int32', alpha: 'complex128',
               a: 'complex128[:,:](order=F)', x: 'complex128[:]', y: 'complex128[:]',
               beta: 'complex128' = 0.,
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

    mod_blas.zhbmv (flag_uplo, n, k, alpha, a, lda, x, incx, beta, y, incy)

# ==============================================================================
def blas_zhpmv(alpha: 'complex128', a: 'complex128[:]', x: 'complex128[:]', y: 'complex128[:]',
               beta: 'complex128' = 0.,
               incx: 'int32' = 1,
               incy: 'int32' = 1,
               lower: 'bool' = False
              ):

    n = np.int32(x.shape[0])

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    mod_blas.zhpmv (flag_uplo, n, alpha, a, x, incx, beta, y, incy)

# ==============================================================================
def blas_ztrmv(a: 'complex128[:,:](order=F)', x: 'complex128[:]',
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

    mod_blas.ztrmv (flag_uplo, flag_trans, flag_diag, n, a, lda, x, incx)

# ==============================================================================
def blas_ztbmv(k : 'int32', a: 'complex128[:,:](order=F)', x: 'complex128[:]',
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

    mod_blas.ztbmv (flag_uplo, flag_trans, flag_diag, n, k, a, lda, x, incx)

# ==============================================================================
def blas_ztpmv(a: 'complex128[:]', x: 'complex128[:]',
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

    mod_blas.ztpmv (flag_uplo, flag_trans, flag_diag, n, a, x, incx)

# ==============================================================================
def blas_ztrsv(a: 'complex128[:,:](order=F)', x: 'complex128[:]',
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

    mod_blas.ztrsv (flag_uplo, flag_trans, flag_diag, n, a, lda, x, incx)

# ==============================================================================
def blas_ztbsv(k: 'int32', a: 'complex128[:,:](order=F)', x: 'complex128[:]',
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

    mod_blas.ztbsv (flag_uplo, flag_trans, flag_diag, n, k, a, lda, x, incx)

# ==============================================================================
def blas_ztpsv(a: 'complex128[:]', x: 'complex128[:]',
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

    mod_blas.ztpsv (flag_uplo, flag_trans, flag_diag, n, a, x, incx)

# ==============================================================================
def blas_zgeru(alpha: 'complex128', x: 'complex128[:]', y: 'complex128[:]', a: 'complex128[:,:](order=F)',
              incx: 'int32' = 1,
              incy: 'int32' = 1,
              ):

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    mod_blas.zgeru (m, n, alpha, x, incx, y, incy, a, lda)

# ==============================================================================
def blas_zgerc(alpha: 'complex128', x: 'complex128[:]', y: 'complex128[:]', a: 'complex128[:,:](order=F)',
              incx: 'int32' = 1,
              incy: 'int32' = 1,
              ):

    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    lda = m

    mod_blas.zgerc (m, n, alpha, x, incx, y, incy, a, lda)

# ==============================================================================
def blas_zher(alpha: 'float64', x: 'complex128[:]', a: 'complex128[:,:](order=F)',
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

    mod_blas.zher (flag_uplo, n, alpha, x, incx, a, lda)

# ==============================================================================
def blas_zhpr(alpha: 'float64', x: 'complex128[:]', a: 'complex128[:]',
              incx: 'int32' = 1,
              lower: 'bool' = False
              ):

    n = np.int32(x.shape[0])

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    mod_blas.zhpr (flag_uplo, n, alpha, x, incx, a)

# ==============================================================================
def blas_zher2(alpha: 'complex128', x: 'complex128[:]', y: 'complex128[:]', a: 'complex128[:,:](order=F)',
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

    mod_blas.zher2 (flag_uplo, n, alpha, x, incx, y, incy, a, lda)

# ==============================================================================
def blas_zhpr2(alpha: 'complex128', x: 'complex128[:]', y: 'complex128[:]', a: 'complex128[:]',
              incx: 'int32' = 1,
              incy: 'int32' = 1,
              lower: 'bool' = False
              ):

    n = np.int32(x.shape[0])

    # ...
    flag_uplo = 'U'
    if lower : flag_uplo = 'L'
    # ...

    mod_blas.zhpr2 (flag_uplo, n, alpha, x, incx, y, incy, a)

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

    mod_blas.zgemm (flag_trans_a, flag_trans_b, l, n, m, alpha, a, lda, b, ldb, beta, c, ldc)

# ==============================================================================
def blas_zsymm(alpha: 'complex128', a: 'complex128[:,:](order=F)', b: 'complex128[:,:](order=F)', c: 'complex128[:,:](order=F)',
               beta: 'complex128' = 0.,
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

    mod_blas.zsymm (flag_side, flag_uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc)

# ==============================================================================
def blas_zhemm(alpha: 'complex128', a: 'complex128[:,:](order=F)', b: 'complex128[:,:](order=F)', c: 'complex128[:,:](order=F)',
               beta: 'complex128' = 0.,
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

    mod_blas.zhemm (flag_side, flag_uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc)

# ==============================================================================
def blas_zsyrk(alpha: 'complex128', a: 'complex128[:,:](order=F)', c: 'complex128[:,:](order=F)',
               beta: 'complex128' = 0.,
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

    mod_blas.zsyrk (flag_uplo, flag_trans, n, k, alpha, a, lda, beta, c, ldc)

# ==============================================================================
def blas_zsyr2k(alpha: 'complex128', a: 'complex128[:,:](order=F)', b: 'complex128[:,:](order=F)', c: 'complex128[:,:](order=F)',
               beta: 'complex128' = 0.,
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

    mod_blas.zsyr2k (flag_uplo, flag_trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc)

# ==============================================================================
def blas_zherk(alpha: 'complex128', a: 'complex128[:,:](order=F)', c: 'complex128[:,:](order=F)',
               beta: 'complex128' = 0.,
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

    mod_blas.zherk (flag_uplo, flag_trans, n, k, alpha, a, lda, beta, c, ldc)

# ==============================================================================
def blas_zher2k(alpha: 'complex128', a: 'complex128[:,:](order=F)', b: 'complex128[:,:](order=F)', c: 'complex128[:,:](order=F)',
               beta: 'complex128' = 0.,
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

    mod_blas.zher2k (flag_uplo, flag_trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc)

# ==============================================================================
def blas_ztrmm(alpha: 'complex128', a: 'complex128[:,:](order=F)', b: 'complex128[:,:](order=F)',
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

    mod_blas.ztrmm (flag_side, flag_uplo, flag_trans_a, flag_diag, m, n, alpha, a, lda, b, ldb)

# ==============================================================================
def blas_ztrsm(alpha: 'complex128', a: 'complex128[:,:](order=F)', b: 'complex128[:,:](order=F)',
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

    mod_blas.ztrsm (flag_side, flag_uplo, flag_trans_a, flag_diag, m, n, alpha, a, lda, b, ldb)
