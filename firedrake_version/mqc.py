import numpy as np
import pprint
from collections import namedtuple
from enum import Enum
from firedrake import *
from petsc4py import PETSc

class MeshQualityCalculator(object):

    def __init__(self, mesh):
        self.mesh = mesh
        self.dim = mesh._geometric_dimension
        self.plex = mesh._topology.topology_dm
        self.__makeCoordinateMap()          # sets self.sec
        self.coordArray = self.plex.getCoordinates().array
    
    def __makeCoordinateMap(self):
        '''Sets up a section for the DMPlex object so that we have a mapping from indices to coordinates'''
        # TODO: See if this needs to be changed for higher dimensions than 2
        numComponents = 1
        entityDofs = [self.dim, 0, 0]
        self.plex.setNumFields(numComponents)
        sec = self.plex.createSection(numComponents, entityDofs)
        sec.setFieldName(0, 'Coordinates')
        sec.setUp()
        self.plex.setSection(sec)

    def __repr__(self):
        return str(self.__dict__)


def test_main():
    print ("Firedrake successfully imported")
    pp = pprint.PrettyPrinter(indent=4)
    mesh = UnitSquareMesh(2, 2)
    # pp.pprint(mesh._topology.__dict__)
    mqc = MeshQualityCalculator(mesh)
    print (mqc)

if __name__ == '__main__':
    test_main()