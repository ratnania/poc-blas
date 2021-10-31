# BUGS

* this import raises the following error
```python
from pyccel.stdlib.internal.blas import dgemv as core_dgemv

ERROR at annotation (semantic) stage
pyccel:
 |fatal [semantic]: utilities.py [14,4]| Undefined function (core_dgemv)
```

# TODO

* Must check all **alpha** in the complex case; sometimes there must be float

# Summary

## Level 1

| SUBROUTINE  | PRECISIONS  | PRECISIONS  | EXAMPLE     |
|             | (AVAILABLE) |    (DONE)   |             |
| ----------- | ----------- | ----------- | ----------- |
| xROTG       | S,D         | SD          |             |
| xROTMG      | S,D         | SD          |             |
| xROT        | S,D         | SD          |             |
| xROTM       | S,D         | SD          |             |
| xSWAP       | S,D,C,Z     | SDC         |             |
| xSCAL       | S,D,C,Z     | SDC         |             |
| xCOPY       | S,D,C,Z     | SDC         |             |
| xAXPY       | S,D,C,Z     | SDC         |             |
| xDOT        | S,D-DS      | SD          |             |
| xDOTU       | C,Z         | C           | KO -> C     |
| xDOTC       | C,Z         | C           | KO -> C     |
| xxDOT       | SDS         |             |             |
| xNRM2       | S,D,SC,DZ   | SDC         |             |
| xASUM       | S,D,SC,DZ   | SDC         | KO -> S,C   |
| IxAMAX      | S,D,C,Z     | SDc         |             |


## Level 2 

| SUBROUTINE  | PRECISIONS   | PRECISIONS  | EXAMPLE     |
|             | (AVAILABLE)  |   (DONE)    |             |
| ----------- | -----------  | ----------- | ----------- |
| xGEMV       | S,D,C,Z      | SDC         |             |
| xGBMV       | S,D,C,Z      | SDC         |             |
| xHEMV       | C,Z          | C           |             |
| xHBMV       | C,Z          | C           |             |
| xHPMV       | C,Z          | C           |             |
| xSYMV       | S,D          | SD          |             |
| xSBMV       | S,D          | SD          |             |
| xSPMV       | S,D          | SD          |             |
| xTRMV       | S,D,C,Z      | SDC         |             |
| xTBMV       | S,D,C,Z      | SDC         |             |
| xTPMV       | S,D,C,Z      | SDC         |             |
| xTRSV       | S,D,C,Z      | SDC         |             |
| xTBSV       | S,D,C,Z      | SDC         |             |
| xTPSV       | S,D,C,Z      | SDC         |             |
| xGER        | S,D          | SD          |             |
| xGERU       | C,Z          | C           |             |
| xGERC       | C,Z          | C           |             |
| xHER        | C,Z          | C           |             |
| xHPR        | C,Z          | C           |             |
| xHER2       | C,Z          | C           |             |
| xHPR2       | C,Z          | C           |             |
| xSYR        | S,D          | SD          |             |
| xSPR        | S,D          | SD          |             |
| xSYR2       | S,D          | SD          |             |
| xSPR2       | S,D          | SD          |             |

## Level 3 

| SUBROUTINE  | PRECISIONS   | PRECISIONS  | EXAMPLE     |
|             | (AVAILABLE)  |   (DONE)    |             |
| ----------- | -----------  | ----------- | ----------- |
| xGEMM       | S,D,C,Z      | SDC         |             |
| xSYMM       | S,D,C,Z      | SDC         |             |
| xHEMM       | C,Z          | C           |             |
| xSYRK       | S,D,C,Z      | SDC         |             |
| xHERK       | C,Z          | C           |             |
| xSYR2K      | S,D,C,Z      | SDC         |             |
| xHER2K      | C,Z          | C           |             |
| xTRMM       | S,D,C,Z      | SDC         |             |
| xTRSM       | S,D,C,Z      | SDC         |             |
