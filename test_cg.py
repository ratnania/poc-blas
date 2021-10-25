#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy.sparse.linalg import cg, gmres, bicgstab, minres
import numpy.linalg as lg
from scipy.sparse.linalg import cg as cg_scipy
from scipy.linalg import blas
from time import time
from tabulate import tabulate
from scipy.io import mmread
from cg_blas import cg as cg_pyccel

# ========================================================================
def cg_old(A,b,x,itermax,tol):
    r    = p = b- blas.dgemv(1.0, A, x)
    k    = 0
    while blas.dnrm2(r) > tol and k<itermax:
        alpha =  blas.dnrm2(r)**2/( blas.ddot(blas.dgemv(1.0, A, p), p))
        x     = x + alpha*p
        rpr   = r.copy()
        r     = rpr - blas.dgemv(alpha, A, p)
        beta  = (blas.dnrm2(r))**2/blas.dnrm2(rpr)**2
        p     = r + beta*p
        k     = k + 1
    return x, k

# ========================================================================
def solve_sparse_scipy(A, b, tol=1.e-5, maxiter=3000, M=None, atol=None, solver='cg', x0=None):

    # ...
    num_iters = 0
    def callback(xk):
        nonlocal num_iters
        num_iters+=1
    # ...

    t1 = time()

    if solver == 'cg':
        x, status = cg(A, b, tol=tol, maxiter=maxiter,
                       M=M, callback=callback, atol=atol, x0=x0)

    elif solver == 'bicgstab':
        x, status = bicgstab(A, b, tol=tol, maxiter=maxiter,
                             M=M, callback=callback, atol=atol, x0=x0)

    elif solver == 'minres':
        x, status = minres(A, b, tol=tol, maxiter=maxiter,
                           M=M, callback=callback, x0=x0, shift=-3.)

    elif solver == 'gmres':
        x, status = gmres(A, b, tol=tol, maxiter=maxiter,
                          M=M, callback=callback, atol=atol, restart=30, x0=x0)

    else:
        raise NotImplemented('solver not available')

    t2 = time()
    err = np.linalg.norm(A.dot(x) - b)
    info = {'niter': num_iters, 'res_norm': err, 'time': (t2-t1)*1000}

    return x, info

# ========================================================================
def solve_sparse_pyccel(A, b, tol=1.e-5, maxiter=3000, M=None, atol=None, solver='cg', x0=None):

    # TODO
    if not( M is None):
        raise ValueError('M must be None')

    # TODO
    if not( atol is None):
        raise ValueError('atol must be None')

    # ...
    num_iters = 0
    def callback(xk):
        nonlocal num_iters
        num_iters+=1
    # ...

    x = x0.copy()

    t1 = time()
    if solver == 'cg':
        err_r, num_iters = cg_pyccel(A, b, x, maxiter=maxiter, tol=tol)

    # TODO remove
    elif solver == 'cg_old':
        x, num_iters = cg_old(A, b, x, maxiter, tol)

    else:
        raise NotImplemented('solver not available')

    t2 = time()
    err = np.linalg.norm(A.dot(x) - b)
    info = {'niter': num_iters, 'res_norm': err, 'time': (t2-t1)*1000}

    return x, info

# ==============================================================================
def run(A, tol=1.e-5, maxiter=10):
    np.random.seed(2021)

    n  = A.shape[0]
    x  = np.random.random(n)
    b  = A @ x
    x0 = np.zeros(n)

    x_scipy, info_scipy = solve_sparse_scipy(A, b, tol=tol, maxiter=maxiter, x0=x0)
    time_scipy  = info_scipy['time']
    niter_scipy = info_scipy['niter']
    res_scipy   = info_scipy['res_norm']
    err_scipy   = np.linalg.norm(x_scipy - x)
    print(info_scipy)

    x_pyccel, info_pyccel = solve_sparse_pyccel(A, b, tol=tol, maxiter=maxiter, x0=x0)
    time_pyccel  = info_pyccel['time']
    niter_pyccel = info_pyccel['niter']
    res_pyccel   = info_pyccel['res_norm']
    err_pyccel   = np.linalg.norm(x_pyccel - x)
    print(info_pyccel)

#    l = []
#    l.append(A.shape)
#    l.append(lg.cond(A))
#    l.append(iterm)
#    l.append(res)
#    l.append(time_pyccel)
#    l.append(err)
#    l.append(niter_scipy)
#    l.append(res_scipy)
#    l.append(time_scipy)
#    l.append(err_scipy)
#    my_data.append(l)

B1=np.array(mmread('matrices/bcsstm21.mtx.gz').todense())
#B2=np.array(mmread('matrices/bcsstk16.mtx.gz').todense())

from scipy.sparse import diags
for n in [200, 500, 1000, 2000, 3000, 4000]:
    print('>>> n = ', n)
    A = diags([1, -2, 1], [-1, 0, 1], shape=(n, n)).toarray()

    run(A)

## create header
#head = ['matrix','condition number','Our_nbr_iter','res','time(ms)','Our_err','scipy_nbr_iter','res_scipy','time(ms)','scipy_err']

# display table
#print(tabulate(my_data, headers=head, tablefmt="grid"))
