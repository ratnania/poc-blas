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
def blas_dgemv(alpha: 'float64', a: 'float64[:,:]', x: 'float64[:]', y: 'float64[:]',
               beta: 'float64' = 0.,
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
def cg(A: 'float64[:,:]', b: 'float64[:]', x: 'float64[:]',
       maxiter: int = 100, tol:'float64' = 1.e-5):

    import numpy as np

    n = x.shape[0]

    p   = np.zeros(n)
    r   = np.zeros(n)
    rpr = np.zeros(n)
    ap  = np.zeros(n)

    blas_dcopy (b, r)
    blas_dgemv(-1.0, A, x, r, beta=1.)
    blas_dcopy (r, p)
    err_r = blas_dnrm2(r)
    k    = 0
    while  err_r > tol and k < maxiter:
        blas_dgemv(1.0, A, p, ap)
        aptp = blas_ddot(ap, p)
        alpha =  err_r**2 / aptp
        blas_daxpy (p, x, a=alpha )
        blas_dcopy (r, rpr)
        blas_dgemv(-alpha, A, p, r, beta=1.)

        err_rpr = blas_dnrm2(rpr)
        err_r = blas_dnrm2(r)
        beta  = (err_r / err_rpr)**2
        blas_dscal (beta, p)
        blas_daxpy (r, p, a=1. )
#        p[:] = r[:] + beta*p[:]

        err_r = blas_dnrm2(r)
        k     = k + 1

    return err_r, k
