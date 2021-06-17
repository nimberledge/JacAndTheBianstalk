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
    def distance(p1, p2):
        if p1.shape[0] != p2.shape[0]:
            return ValueError
        
        return np.sqrt(np.sum([(p1[i] - p2[i])**2 for i in range(p1.shape[0])]))
    
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

    def getCellIndices(self):
        if self.meshType == self.MeshType.TRIANGLE_2D or self.meshType == self.MeshType.QUADRILATERAL_2D:
            return self.plex.getDepthStratum(self.FACE_DEPTH_STRATUM)
        elif self.meshType == self.MeshType.TETRAHEDRAL_3D or self.meshType == self.MeshType.HEXAHEDRAL_3D:
            return self.plex.getDepthStratum(self.HEDRON_DEPTH_STRATUM)
        else:
            return (-1, -1)

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
        cStart, cEnd = self.plex.getDepthStratum(self.FACE_DEPTH_STRATUM)
        assert cStart <= index < cEnd

        dim = self.plex.getDimension()
        plexCoords = self.plex.getCoordinates()
        edges = self.plex.getCone(index)
        vertices = set()
        minEdgeLength = np.inf
        maxEdgeLength = 0
        edgeLengths = []
        for e in edges:
            # Add vertices to a set (to avoid overcounting), use to calculate CQM
            verts = self.plex.getCone(e)
            v1, v2 = [([plexCoords.array[self.sec.getOffset(v) + j] for j in range(dim)]) for v in verts]
            vertices.add(tuple(v1))
            vertices.add(tuple(v2))
                
        v1, v2, v3 = [np.array(v) for v in vertices]
        # These guys are np arrays so i can do element-wise subtraction
        # Get edge vectors
        vec12 = v2 - v1
        vec23 = v3 - v2
        vec31 = v1 - v3

        # Calculate edge lengths        
        dist12 = self.distance(v1, v2)
        dist23 = self.distance(v2, v3)
        dist31 = self.distance(v3, v1)
        edgeLengths = [dist12, dist23, dist31]

        # Calculate the scaled Jacobian at each vertex
        # https://cubit.sandia.gov/15.5/help_manual/WebHelp/mesh_generation/mesh_quality_assessment/triangular_metrics.htm
        # https://www.osti.gov/biblio/5009 - shows how to calculate the jacobian
        scaledJacobian1 = np.absolute(np.linalg.det(np.array([vec12, -vec31]))) / (dist12 * dist31)
        scaledJacobian2 = np.absolute(np.linalg.det(np.array([vec23, -vec12]))) / (dist23 * dist12)
        scaledJacobian3 = np.absolute(np.linalg.det(np.array([vec31, -vec23]))) / (dist31 * dist23)
        scaledJacobian = min(scaledJacobian1, scaledJacobian2, scaledJacobian3)
        
        # Calculate angles at each vertex
        # Note: Orienting the vectors is important
        a1 = np.arccos (np.dot(vec12, -vec31) / (dist12 * dist31))
        a2 = np.arccos (np.dot(vec12, -vec23) / (dist12 * dist23))
        a3 = np.arccos (np.dot(vec31, -vec23) / (dist31 * dist23))
        anglesAtVertices = [a1, a2, a3]
        
        # maxEdgeLength = max(edgeLengths)
        # minEdgeLength = min(edgeLengths)
        minAngle = min(anglesAtVertices)
        maxAngle = max(anglesAtVertices)
        
        # Calculate area (Heron's formula), edge length ratio, and aspect ratio from edge lengths
        # edgeLengthRatio = maxEdgeLength / minEdgeLength
        semiPerimeter = sum(edgeLengths) / 2
        area = semiPerimeter
        miniProduct = 1             # We are reusing this value = (s-a)(s-b)(s-c)
        edgeLengthsProduct = 1      # Need this for aspect ratio calculation
        for i in range(len(edgeLengths)):
            miniProduct *= (semiPerimeter - edgeLengths[i])
            edgeLengthsProduct *= edgeLengths[i]
        area = np.sqrt(area * miniProduct)
        aspectRatio = edgeLengthsProduct / (8 * miniProduct)
        
        idealAngle = np.pi / 3      # There's gotta be a better way to write this stuff
        equiangleSkew = max( (maxAngle - idealAngle) / (np.pi - idealAngle), (idealAngle - minAngle) / idealAngle)

        # Calculating in accordance with https://www.engmorph.com/skewness-finite-elemnt
        # sideN -> side opposite vertex vN
        midPointSide1 = v2 + (v3 - v2) / 2
        midPointSide2 = v3 + (v1 - v3) / 2
        midPointSide3 = v1 + (v2 - v1) / 2

        lineNormalSide1 = midPointSide1 - v1
        lineOrthSide1 = midPointSide3 - midPointSide2
        theta1 = np.arccos (np.dot(lineNormalSide1, lineOrthSide1) / (self.distance(v1, midPointSide1) * self.distance(midPointSide2, midPointSide3)))
        theta2 = np.pi - theta1

        lineNormalSide2 = midPointSide2 - v2
        lineOrthSide2 = midPointSide1 - midPointSide3
        theta3 = np.arccos (np.dot(lineNormalSide2, lineOrthSide2) / (self.distance(v2, midPointSide2) * self.distance(midPointSide1, midPointSide3)))
        theta4 = np.pi - theta3

        lineNormalSide3 = midPointSide3 - v3
        lineOrthSide3 = midPointSide2 - midPointSide1
        theta5 = np.arccos (np.dot(lineNormalSide3, lineOrthSide3) / (self.distance(v3, midPointSide3) * self.distance(midPointSide2, midPointSide1)))
        theta6 = np.pi - theta5

        skewness = (np.pi / 2) - min(theta1, theta2, theta3, theta4, theta5, theta6)
        
        return CQM(area, minAngle, aspectRatio, skewness, equiangleSkew, scaledJacobian)
    
    def getQuadrilateralCellQualityMeasures(self, index):
        return CQM(0, 0, 0, 0, 0, 0)
    
    def getTetrahedralCellQualityMeasures(self, index):
        return CQM(0, 0, 0, 0, 0, 0)
    
    def getHexahedralCellQualityMeasures(self, index):
        return CQM(0, 0, 0, 0, 0, 0)


def formatCQM(cqm, dim=2):
    '''Just a pprint for CQM, default is {:.5f} for now'''
    fields = cqm._fields
    retval = '('
    for i, field in enumerate(fields):
        if i == 0:
            if dim == 2:
                retval += 'area: {:.5f} '.format(cqm[i])
            elif dim == 3:
                retval += 'volume: {:.5f} '.format(cqm[i])
            else:
                retval += field + ': {:.5f} '.format(cqm[i])
            continue
        retval += field + ': {:.5f} '.format(cqm[i])
    retval = retval.rstrip() + ')'
    return retval
    
def main():
    dim = 2
    coords = np.asarray([
        [0.0, 0.0], # 0
        [0.5, 0.0], # 1
        [1.0, 0.0], # 2
        [0.0, 0.5], # 3
        [0.5, 0.5], # 4
        [1.0, 0.5], # 5
        [0.0, 1.0], # 6
        [0.5, 1.0], # 7
        [1.0, 1.0], # 8
    ], dtype=float)
    
    cells = np.asarray([
        [0, 1, 3], # 0
        [1, 4, 3], # 1
        [1, 2, 4], # 2
        [2, 4, 5], # 3
        [3, 4, 6], # 4
        [4, 6, 7], # 5
        [5, 7, 8], # 6
        [4, 5, 7], # 7
    ], dtype=PETSc.IntType)
    # Create DMPlex from cells and vertices
    plex = PETSc.DMPlex().createFromCellList(dim, cells, coords, comm=PETSc.COMM_WORLD)
    # Alternative: Generate box mesh using PETSc 
    # dim = 3 
    # plex = PETSc.DMPlex().createBoxMesh([4]*dim, simplex=True)
    

    # Actually still not sure 100% what this code is doing, but it creates a section
    numComponents = 1
    entityDofs = [dim, 0, 0]  # 2 entries for each vertex, 0 for each edge, 0 for each cell
    plex.setNumFields(1)
    sec = plex.createSection(numComponents, entityDofs)
    sec.setFieldName(0, 'Coordinates')
    sec.setUp()
    plex.setSection(sec)

    qualityCalculator = MeshQualityCalculator(plex, sec)
    # print ("Mesh type: {}".format(qualityCalculator.meshType))
    cStart, cEnd = qualityCalculator.getCellIndices()
    for c in range(cStart, cEnd):
        print ("Cell: {} CQM: {}".format(c, formatCQM(qualityCalculator.getCellQualityMeasures(c), dim=dim)))

if __name__ == '__main__':
    main()