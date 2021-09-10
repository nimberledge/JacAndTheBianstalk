from firedrake import *
import time
import numpy as np
import os

def getCQM(mesh, include_dirs):
    '''Computes the CQM (Cell Quality Measures) of each element of a triangular 2D mesh using a C kernel.
    CQMs are as follows - 
    1. Area
    2. Minimum Angle
    3. Aspect Ratio
    4. Equiangle Skew
    5. Skewness
    6. Scaled Jacobian'''
    P0 = FunctionSpace(mesh, "Discontinuous Lagrange", 0)
    P0_ten = TensorFunctionSpace(mesh, "DG", 0)
    P0_vec = VectorFunctionSpace(mesh, "DG", 0)
    
    
    coords = mesh.coordinates
    areas = Function(P0)
    minAngles = Function(P0)
    aspectRatios = Function(P0)
    eSkews = Function(P0)
    skews = Function(P0)
    scaledJacobians = Function(P0)
    cqmKernel = """
    #include "meshquality.h"

    void doGetCQM(double *areas, double *minAngles, double *aspectRatios, double *eSkews,
                  double *skews, double *scaledJacobians, double *coords) {    
        getCQM(areas, minAngles, aspectRatios, eSkews, skews, scaledJacobians, coords);
    }
    """
    kernel = op2.Kernel(cqmKernel, "doGetCQM", cpp=True, include_dirs=include_dirs)
    op2.par_loop(kernel, mesh.cell_set, areas.dat(op2.WRITE, areas.cell_node_map()), minAngles.dat(op2.WRITE, minAngles.cell_node_map()),\
            aspectRatios.dat(op2.WRITE, aspectRatios.cell_node_map()), eSkews.dat(op2.WRITE, eSkews.cell_node_map()), skews.dat(op2.WRITE,\
            skews.cell_node_map()), scaledJacobians.dat(op2.WRITE, scaledJacobians.cell_node_map()), coords.dat(op2.READ, coords.cell_node_map()))
    return (areas, minAngles, aspectRatios, eSkews, skews, scaledJacobians)

def getMetric(mesh, include_dirs, M=None):
    '''Given a matrix M, a linear function in 2 dimensions, this function outputs
    the value of the Quality metric Q_M based on the transformation encoded in M.
    '''
    # x, y = SpatialCoordinate(mesh)
    # if M is None:
    #     M = [[1*x, 0], [-1*x, 1*y]]
    if M is None:
        M = [[1, 0], [0, 1]]
    
    coords = mesh.coordinates
    P0 = FunctionSpace(mesh, "Discontinuous Lagrange", 0)
    P1_ten = TensorFunctionSpace(mesh, "DG", 1)
    tensor = interpolate(as_matrix(M), P1_ten)
    metrics = Function(P0)
    cqmKernel = """
    #include "meshquality.h"

    void doGetMetric(double *metrics, const double *T_, double *coords) {
        getMetric(metrics, T_, coords);
    }
    """
    kernel = op2.Kernel(cqmKernel, "doGetMetric", cpp=True, include_dirs=include_dirs)
    op2.par_loop(kernel, mesh.cell_set, metrics.dat(op2.WRITE, metrics.cell_node_map()), tensor.dat(op2.READ, tensor.cell_node_map()), \
                 coords.dat(op2.READ, coords.cell_node_map()))
    return metrics

def main():
    try:
        from firedrake.slate.slac.compiler import PETSC_ARCH
    except ImportError:
        PETSC_ARCH = os.path.join(os.environ.get('PETSC_DIR'), os.environ.get('PETSC_ARCH'))
    include_dirs = ["%s/include/eigen3" % PETSC_ARCH]
    cwd = os.getcwd()

    cpp_include_dir = cwd + "/cpp_include/"
    include_dirs.append(cpp_include_dir)
    print ("Firedrake successfully imported")
    
    m,n = 500, 500
    mesh = UnitSquareMesh(m, n)
    print ("Mesh size: {} x {}".format(m, n))
    print ("Number of cells: {}".format(2 * m * n))
    start = time.time()
    areas, minAngles, aspectRatios, eSkews, skews, scaledJacobians = getCQM(mesh, include_dirs=include_dirs)
    metrics = getMetric(mesh, include_dirs)
    timeTaken = time.time() - start
    cqms = np.zeros((areas.dat.data.shape[0], 7))
    
    cqms[:, 0] = areas.dat.data
    cqms[:, 1] = minAngles.dat.data
    cqms[:, 2] = aspectRatios.dat.data
    cqms[:, 3] = skews.dat.data
    cqms[:, 4] = eSkews.dat.data
    cqms[:, 5] = scaledJacobians.dat.data
    cqms[:, 6] = metrics.dat.data
    
    
    print ("Area\t\tMin Angle\tAspect Ratio\tSkewness\tEq. skew\tS. Jacobian\tMetric")
    print ('\t'.join(["{:.6f}".format(k) for k in cqms[0, :]]))
    # for r in range(cqms.shape[0]):
    #     print ('\t'.join(["{:.6f}".format(k) for k in cqms[r, :]]))
    print ("Time taken: {}s".format(timeTaken))

if __name__ == '__main__':
    main()