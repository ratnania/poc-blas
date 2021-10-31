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
| xSWAP       | S,D,C,Z     | SDCZ        |             |
| xSCAL       | S,D,C,Z     | SDCZ        |             |
| xCOPY       | S,D,C,Z     | SDCZ        |             |
| xAXPY       | S,D,C,Z     | SDCZ        |             |
| xDOT        | S,D-DS      | SD          |             |
| xDOTU       | C,Z         | CZ          | KO -> C     |
| xDOTC       | C,Z         | CZ          | KO -> C     |
| xxDOT       | SDS         |             |             |
| xNRM2       | S,D,SC,DZ   | SDCZ        |             |
| xASUM       | S,D,SC,DZ   | SDCZ        | KO -> S,C   |
| IxAMAX      | S,D,C,Z     | SDCZ        |             |


## Level 2 

| SUBROUTINE  | PRECISIONS   | PRECISIONS  | EXAMPLE     |
|             | (AVAILABLE)  |   (DONE)    |             |
| ----------- | -----------  | ----------- | ----------- |
| xGEMV       | S,D,C,Z      | SDCZ        |             |
| xGBMV       | S,D,C,Z      | SDCZ        |             |
| xHEMV       | C,Z          | CZ          |             |
| xHBMV       | C,Z          | CZ          |             |
| xHPMV       | C,Z          | CZ          |             |
| xSYMV       | S,D          | SD          |             |
| xSBMV       | S,D          | SD          |             |
| xSPMV       | S,D          | SD          |             |
| xTRMV       | S,D,C,Z      | SDCZ        |             |
| xTBMV       | S,D,C,Z      | SDCZ        |             |
| xTPMV       | S,D,C,Z      | SDCZ        |             |
| xTRSV       | S,D,C,Z      | SDCZ        |             |
| xTBSV       | S,D,C,Z      | SDCZ        |             |
| xTPSV       | S,D,C,Z      | SDCZ        |             |
| xGER        | S,D          | SD          |             |
| xGERU       | C,Z          | CZ          |             |
| xGERC       | C,Z          | CZ          |             |
| xHER        | C,Z          | CZ          |             |
| xHPR        | C,Z          | CZ          |             |
| xHER2       | C,Z          | CZ          |             |
| xHPR2       | C,Z          | CZ          |             |
| xSYR        | S,D          | SD          |             |
| xSPR        | S,D          | SD          |             |
| xSYR2       | S,D          | SD          |             |
| xSPR2       | S,D          | SD          |             |

## Level 3 

| SUBROUTINE  | PRECISIONS   | PRECISIONS  | EXAMPLE     |
|             | (AVAILABLE)  |   (DONE)    |             |
| ----------- | -----------  | ----------- | ----------- |
| xGEMM       | S,D,C,Z      | SDCZ        |             |
| xSYMM       | S,D,C,Z      | SDCZ        |             |
| xHEMM       | C,Z          | CZ          |             |
| xSYRK       | S,D,C,Z      | SDCZ        |             |
| xHERK       | C,Z          | CZ          |             |
| xSYR2K      | S,D,C,Z      | SDCZ        |             |
| xHER2K      | C,Z          | CZ          |             |
| xTRMM       | S,D,C,Z      | SDCZ        |             |
| xTRSM       | S,D,C,Z      | SDCZ        |             |
