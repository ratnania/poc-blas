import numpy as np

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
def blas_dgemv(alpha: 'float64', a: 'float64[:,:]', x: 'float64[:]', y: 'float64[:]',
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
    flag = 'N'
    if trans: flag = 'T'
    # ...

    dgemv (flag, m, n, alpha, a, lda, x, incx, beta, y, incy)

# ======================================================================
def cg(A: 'float64[:,:]', b: 'float64[:]', x: 'float64[:]',
       maxiter: int = 100, tol:'float64' = 1.e-5):

    n = np.int32(x.shape[0])

    p   = np.zeros(n)
    r   = np.zeros(n)
    rpr = np.zeros(n)

    one_32 = np.int32(1)

    blas_dcopy (b, r, incx=one_32, incy=one_32)
    blas_dgemv(-1.0, A, x, r, beta=1., incx=one_32, incy=one_32)
    blas_dcopy (r, p, incx=one_32, incy=one_32)
    err_r = blas_dnrm2(r, incx=one_32)
    k = np.int32(0)
    while  err_r > tol and k < maxiter:
        blas_dgemv(1.0, A, p, rpr, incx=one_32, incy=one_32)
        aptp = blas_ddot(rpr, p, incx=one_32, incy=one_32)
        alpha =  err_r**2 / aptp
        blas_daxpy (p, x, a=alpha, incx=one_32, incy=one_32 )
        blas_dcopy (r, rpr, incx=one_32, incy=one_32)
        blas_dgemv(-alpha, A, p, r, beta=1., incx=one_32, incy=one_32)

        err_rpr = blas_dnrm2(rpr, incx=one_32)
        err_r = blas_dnrm2(r, incx=one_32)
        beta  = (err_r / err_rpr)**2
        blas_dscal (beta, p, incx=one_32)
        blas_daxpy (r, p, a=1., incx=one_32, incy=one_32 )

        k     = k + 1

    return err_r, k
