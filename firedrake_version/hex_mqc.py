from mqc import CQM, MeshQualityCalculator
from firedrake import *
from petsc4py import PETSc
import numpy as np

class HexahedronMeshQualityCalculator(MeshQualityCalculator):

    def __init__(self, mesh):
        super().__init__(mesh)


def test_main():
    print ("Firedrake successfully imported")
    # TODO: find out how to generate a hexahedral mesh
    # In theory, this shouldn't be hard to do, but I trust libraries a lot more than myself
    # mesh = UnitCubeMesh(2, 2, 2, )
    # hmqc = QuadrilateralMeshQualityCalculator(mesh)
    # print (hmqc.meshType)
    # cStart, cEnd = hmqc.getCellIndices()
    # print (cStart, cEnd)
    # for c in range(cStart, cEnd):
    #     print (qmqc.getCellQualityMeasures(c))

if __name__ == '__main__':
    test_main()
