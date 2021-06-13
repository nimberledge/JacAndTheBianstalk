from petsc4py import *
from petsc4py import PETSc

import numpy as np
from collections import namedtuple
from enum import Enum

CQM = namedtuple('CQM', 'measure, minAngle, aspectRatio, skewness, equiangleSkew, scaledJacobian')

class MeshQualityCalculator(object):

    VERTEX_DEPTH_STRATUM = 0
    EDGE_DEPTH_STRATUM = 1
    FACE_DEPTH_STRATUM = 2
    HEDRON_DEPTH_STRATUM = 3

    '''Enum to represent the different types of mesh.'''
    class MeshType(Enum):
        TRIANGLE_2D = 1
        QUADRILATERAL_2D = 2
        TETRAHEDRAL_3D = 3
        HEXAHEDRAL_3D = 4
        UNKNOWN = 5

    def __init__(self, plex, sec):
        '''Initialize with a mesh  object (plex), and the co-ordinate mapping given by the section object (sec)'''
        self.plex = plex
        self.sec = sec
        # Determine type of mesh
        self.meshType = MeshQualityCalculator.getMeshType(self.plex)
    
    @staticmethod
    def getMeshType(plex):
        '''Analyzes the plex object to classify a Mesh as one of the following types - 
        1. 2D Triangle Mesh
        2. 2D Quadrilateral Mesh
        3. 3D Tetrahedral Mesh
        4. 3D Hexahedral Mesh'''
        # Check for 3D vs 2D
        if plex.getDimension() == 2:     # 2D mesh
            # Now, count edges in the sample cell and determine if quad or tri
            sampleCell = plex.getDepthStratum(MeshQualityCalculator.FACE_DEPTH_STRATUM)[0]
            edges = plex.getCone(sampleCell)
            if len(edges) == 3:
                return MeshQualityCalculator.MeshType.TRIANGLE_2D
            else:
                return MeshQualityCalculator.MeshType.QUADRILATERAL_2D
        elif plex.getDimension() == 3:   # 3D mesh
            # Now, count faces in the sample cell and determine if tet or hex
            sampleCell = plex.getDepthStratum(MeshQualityCalculator.HEDRON_DEPTH_STRATUM)[0]
            faces = plex.getCone(sampleCell)
            if len(faces) == 4:
                return MeshQualityCalculator.MeshType.TETRAHEDRAL_3D
            else:
                return MeshQualityCalculator.MeshType.HEXAHEDRAL_3D
        else: # Only dealing with the above types for now
            return MeshQualityCalculator.MeshType.UNKNOWN

    def getCellQualityMeasures(self, index):
        '''Analyzes a given cell (identified by index) of the mesh and returns the following cell quality measures -
        1. Measure (Area for 2D / Volume for 3D)
        2. Minimum Angle
        3. Aspect Ratio
        4. Skewness
        5. Equiangle skew
        6. Scaled Jacobian
        '''
        if self.meshType == self.MeshType.TRIANGLE_2D:
            return self.getTriangleCellQualityMeasures(index)
        elif self.meshType == self.MeshType.QUADRILATERAL_2D:
            return self.getQuadrilateralCellQualityMeasures(index)
        elif self.meshType == self.MeshType.TETRAHEDRAL_3D:
            return self.getTetrahedralCellQualityMeasures(index)
        elif self.meshType == self.MeshType.HEXAHEDRAL_3D:
            return self.getHexahedralCellQualityMeasures(index)

        # Default return, just ensuring the typing is consistent
        return CQM(0, 0, 0, 0, 0, 0)
    
    def getTriangleCellQualityMeasures(self, index):
        return CQM(0, 0, 0, 0, 0, 0)
    
    def getQuadrilateralCellQualityMeasures(self, index):
        return CQM(0, 0, 0, 0, 0, 0)
    
    def getTetrahedralCellQualityMeasures(self, index):
        return CQM(0, 0, 0, 0, 0, 0)
    
    def getHexahedralCellQualityMeasures(self, index):
        return CQM(0, 0, 0, 0, 0, 0)
    
def main():
    dim = 2
    dim = 3 
    # Create DMPlex from cells and vertices
    plex = PETSc.DMPlex().createBoxMesh([4]*dim, simplex=True)

    # Actually still not sure 100% what this code is doing, but it creates a section
    numComponents = 1
    entityDofs = [dim, 0, 0]  # 2 entries for each vertex, 0 for each edge, 0 for each cell
    plex.setNumFields(1)
    sec = plex.createSection(numComponents, entityDofs)
    sec.setFieldName(0, 'Coordinates')
    sec.setUp()
    plex.setSection(sec)

    qualityCalculator = MeshQualityCalculator(plex, sec)
    print (qualityCalculator.meshType)

if __name__ == '__main__':
    main()