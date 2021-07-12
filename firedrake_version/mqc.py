import numpy as np
import pprint
from collections import namedtuple
from enum import Enum
from firedrake import *
from petsc4py import PETSc

# Cell quality measure data structure
CQM = namedtuple('CQM', 'measure, minAngle, aspectRatio, skewness, equiangleSkew, scaledJacobian')

class MeshQualityCalculator(object):

    '''Enum to represent the different types of mesh.'''
    class MeshType(Enum):
        TRIANGLE_2D = 1
        QUADRILATERAL_2D = 2
        TETRAHEDRAL_3D = 3
        HEXAHEDRAL_3D = 4
        UNKNOWN = 5

    def __init__(self, mesh):
        self.mesh = mesh
        self.dim = mesh._geometric_dimension
        self.plex = mesh._topology.topology_dm
        self.__makeCoordinateMap()          # sets self.sec
        self.coordArray = self.plex.getCoordinates().array
        self.meshType = self.__computeMeshType()

    def __makeCoordinateMap(self):
        '''Sets up a section for the DMPlex object so that we have a mapping from indices to coordinates'''
        # TODO: See if this needs to be changed for higher dimensions than 2
        numComponents = 1
        entityDofs = [self.dim]
        entityDofs.extend([0 for i in range(self.dim)])
        self.plex.setNumFields(numComponents)
        sec = self.plex.createSection(numComponents, entityDofs)
        sec.setFieldName(0, 'Coordinates')
        sec.setUp()
        self.plex.setSection(sec)

    def __computeMeshType(self):
        sampleCell = self.plex.getDepthStratum(self.dim)[0]
        if self.dim == 2:
            edges = self.plex.getCone(sampleCell)
            if len(edges) == 3:
                return self.MeshType.TRIANGLE_2D
            elif len(edges) == 4:
                return self.MeshType.QUADRILATERAL_2D

        elif self.dim == 3:
            faces = self.plex.getCone(sampleCell)
            if len(faces) == 4:
                return self.MeshType.TETRAHEDRAL_3D
            elif len(faces) == 6:
                return self.MeshType.HEXAHEDRAL_3D

        return self.MeshType.UNKNOWN

    def __repr__(self):
        return str(self.__dict__)


def test_main():
    print ("Firedrake successfully imported")
    pp = pprint.PrettyPrinter(indent=4)
    # mesh = UnitSquareMesh(2, 2)
    mesh = UnitTetrahedronMesh()
    # pp.pprint(mesh._ufl_coordinate_element._sub_element)
    mqc = MeshQualityCalculator(mesh)
    print (mqc.meshType)

if __name__ == '__main__':
    test_main()
