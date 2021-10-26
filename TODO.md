# BUGS

* this import raises the following error
```python
from pyccel.stdlib.internal.blas import dgemv as core_dgemv

ERROR at annotation (semantic) stage
pyccel:
 |fatal [semantic]: utilities.py [14,4]| Undefined function (core_dgemv)
```

* **dnrm2** is not available, although it is declared in our header file
```shell
   51 |     r = dnrm2(n, x, incx)
      |        1
Error: Function ‘dnrm2’ at (1) has no IMPLICIT type;
```

* BLAS functions are not recognized, we are missing something like
```fortran
    real(f64), external :: ddot
```

# TODO

* band matrices
* other precisions than D

# Summary

## Level 1

| SUBROUTINE  | PRECISIONS  | PRECISIONS  | EXAMPLE     |
|             | (AVAILABLE) |    (DONE)   |             |
| ----------- | ----------- | ----------- | ----------- |
| xROTG       | S,D         | D           |             |
| xROTMG      | S,D         | D           |             |
| xROT        | S,D         | D           |             |
| xROTM       | S,D         | D           |             |
| xSWAP       | S,D,C,Z     | D           |             |
| xSCAL       | S,D,C,Z     | D           |             |
| xCOPY       | S,D,C,Z     | D           |             |
| xAXPY       | S,D,C,Z     | D           |             |
| xDOT        | S,D-DS      | D           |             |
| xDOTU       | C,Z         |             |             |
| xDOTC       | C,Z         |             |             |
| xxDOT       | SDS         |             |             |
| xNRM2       | S,D,SC,DZ   | D           |             |
| xASUM       | S,D,SC,DZ   | D           |             |
| IxAMAX      | S,D,C,Z     | D           |             |


## Level 2 

| SUBROUTINE  | PRECISIONS   | PRECISIONS  | EXAMPLE     |
|             | (AVAILABLE)  |   (DONE)    |             |
| ----------- | -----------  | ----------- | ----------- |
| xGEMV       | S,D,C,Z      | D           |             |
| xGBMV       | S,D,C,Z      | D           | NOT WORKING |
| xHEMV       | C,Z          |             |             |
| xHBMV       | C,Z          |             |             |
| xHPMV       | C,Z          |             |             |
| xSYMV       | S,D          | D           |             |
| xSBMV       | S,D          | D           | NOT WORKING |
| xSPMV       | S,D          |             |             |
| xTRMV       | S,D,C,Z      | D           |             |
| xTBMV       | S,D,C,Z      |             |             |
| xTPMV       | S,D,C,Z      |             |             |
| xTRSV       | S,D,C,Z      | D           |             |
| xTBSV       | S,D,C,Z      |             |             |
| xTPSV       | S,D,C,Z      |             |             |
| xGER        | S,D          | D           |             |
| xGERU       | C,Z          |             |             |
| xGERC       | C,Z          |             |             |
| xHER        | C,Z          |             |             |
| xHPR        | C,Z          |             |             |
| xHER2       | C,Z          |             |             |
| xHPR2       | C,Z          |             |             |
| xSYR        | S,D          | D           |             |
| xSPR        | S,D          |             |             |
| xSYR2       | S,D          | D           |             |
| xSPR2       | S,D          |             |             |

## Level 3 

| SUBROUTINE  | PRECISIONS   | PRECISIONS  | EXAMPLE     |
|             | (AVAILABLE)  |   (DONE)    |             |
| ----------- | -----------  | ----------- | ----------- |
| xGEMM       | S,D,C,Z      | D           |             |
| xSYMM       | S,D,C,Z      | D           |             |
| xHEMM       | C,Z          |             |             |
| xSYRK       | S,D,C,Z      | D           |             |
| xHERK       | C,Z          |             |             |
| xSYR2K      | S,D,C,Z      | D           |             |
| xHER2K      | C,Z          |             |             |
| xTRMM       | S,D,C,Z      | D           |             |
| xTRSM       | S,D,C,Z      | D           |             |
