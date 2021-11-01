# poc-blas

POC for BLAS using Pyccel

## Compiling the code

```shell
pyccel sblas.py --libs blas
pyccel dblas.py --libs blas
pyccel cblas.py --libs blas
pyccel zblas.py --libs blas
```

## Running the tests

```shell
python3 test_sblas.py
python3 test_dblas.py
python3 test_cblas.py
python3 test_zblas.py
```
