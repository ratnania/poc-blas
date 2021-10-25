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
from cg_pure import cg as cg_pure

# ======================================================================
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

# ======================================================================
def solve_sparse_scipy(A, b, tol=1.e-5, maxiter=3000, M=None, atol=None,
                       solver='cg', x0=None, xref=None):

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
    res_norm = np.linalg.norm(A.dot(x) - b)
    err_norm = None
    if not( xref is None ):
        err_norm   = np.linalg.norm(x - xref)
    info = {'niter': num_iters, 'res_norm': res_norm, 'err_norm': err_norm, 'time': (t2-t1)*1000}

    return x, info

# ======================================================================
def solve_sparse_pyccel(A, b, tol=1.e-5, maxiter=3000, M=None, atol=None,
                        solver='cg', x0=None, xref=None):

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
    elif solver == 'cg_pure':
        num_iters = cg_pure(A, b, x, maxiter, tol)

    # TODO remove
    elif solver == 'cg_old':
        x, num_iters = cg_old(A, b, x, maxiter, tol)

    else:
        raise NotImplemented('solver not available')

    t2 = time()
    res_norm = np.linalg.norm(A.dot(x) - b)
    err_norm = None
    if not( xref is None ):
        err_norm   = np.linalg.norm(x - xref)
    info = {'niter': num_iters, 'res_norm': res_norm, 'err_norm': err_norm, 'time': (t2-t1)*1000}

    return x, info

# ======================================================================
def run(A, tol=1.e-5, maxiter=10):
    table = []
    np.random.seed(2021)

    n  = A.shape[0]
    x  = np.random.random(n)
    b  = A @ x
    x0 = np.zeros(n)

    x_scipy, info = solve_sparse_scipy(A, b, tol=tol, maxiter=maxiter, x0=x0, xref=x)
    line = ['scipy', info['niter'], info['res_norm'], info['err_norm'], info['time']] ; table.append(line)

    x_pyccel, info = solve_sparse_pyccel(A, b, tol=tol, maxiter=maxiter, x0=x0, xref=x)
    line = ['pyccel-blas', info['niter'], info['res_norm'], info['err_norm'], info['time']] ; table.append(line)

    x_pyccel, info = solve_sparse_pyccel(A, b, tol=tol, maxiter=maxiter, x0=x0, xref=x, solver='cg_pure')
    line = ['pyccel-pure', info['niter'], info['res_norm'], info['err_norm'], info['time']] ; table.append(line)

    return table

#B1=np.array(mmread('matrices/bcsstm21.mtx.gz').todense())
#B2=np.array(mmread('matrices/bcsstk16.mtx.gz').todense())

# create header
headers = ['implementation', 'num_iters','res','err','time(ms)']

from scipy.sparse import diags
for n in [200, 500, 1000, 2000, 3000, 4000, 6000]:
    print('>>> n = ', n)
    A = diags([1, -2, 1], [-1, 0, 1], shape=(n, n)).toarray()

    table = run(A)

    # display table
    print(tabulate(table, headers=headers, tablefmt="grid"))

