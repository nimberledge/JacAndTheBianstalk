from firedrake import *
import time

def getMinAngle(mesh, include_dirs, P0=None, P0_ten=None, P0_vec=None):
    '''Computes the CQM (Cell Quality Measures) of each element of a triangular 2D mesh using a C kernel.
    CQMs are as follows - 
    1. Area
    2. Minimum Angle
    3. Aspect Ratio
    4. Equiangle Skew
    5. Skewness
    6. Scaled Jacobian'''
    if P0 is None:
        P0 = FunctionSpace(mesh, "Discontinuous Lagrange", 0)
    if P0_ten is None:
        P0_ten = TensorFunctionSpace(mesh, "DG", 0)
    if P0_vec is None:
        P0_vec = VectorFunctionSpace(mesh, "DG", 0)
    
    coords = mesh.coordinates
    print (type(coords))
    minAngles = Function(P0)
    kernel="""
    #include <Eigen/Dense>

    using namespace Eigen;

    void getMinAngles(double *coords ,double *minAngles) {
        Vector2d v1(coords[0], coords[1]);
        minAngles[0] = v1.dot(v1);
    }    
    """
    cppKernel = op2.Kernel(kernel, "getMinAngles", cpp=True, include_dirs=include_dirs)
    op2.par_loop(cppKernel, P0.node_set, coords.dat(op2.RW), minAngles.dat(op2.RW))
    return minAngles

def main():
    try:
        from firedrake.slate.slac.compiler import PETSC_ARCH
    except ImportError:
        import os
        PETSC_ARCH = os.path.join(os.environ.get('PETSC_DIR'), os.environ.get('PETSC_ARCH'))
    include_dirs = ["%s/include/eigen3" % PETSC_ARCH]
    print ("Firedrake successfully imported")
    m,n = 4, 4
    mesh = UnitSquareMesh(m, n)
    print ("Import successful")
    # areas = computeArea(mesh)
    # minAngles = computeMinAngle(mesh)
    # aspectRatios = computeAspectRatio(mesh)
    # equiangleSkews = computeEquiangleSkew(mesh)
    # scaledJacobians = computeScaledJacobian(mesh)
    # skews = computeSkewness(mesh)

    start = time.time()
    minAngles = getMinAngle(mesh, include_dirs=include_dirs)
    timeTaken = time.time() - start
    # cqms = np.zeros((areas.dat.data.shape[0], 6))
    
    # cqms[:, 0] = areas.dat.data
    # cqms[:, 1] = minAngles.dat.data
    # cqms[:, 2] = aspectRatios.dat.data
    # cqms[:, 3] = skews.dat.data
    # cqms[:, 4] = equiangleSkews.dat.data
    # cqms[:, 5] = scaledJacobians.dat.data
    
    # print ("Mesh size: {} x {}".format(m, n))
    # print ("Number of cells: {}".format(areas.dat.data.shape[0]))
    # print ("Area\t\tMin Angle\tAspect Ratio\tSkewness\tEq. skew\tScaled Jacobian")
    # for r in range(cqms.shape[0]):
    #     print ('\t'.join(["{:.6f}".format(k) for k in cqms[r, :]]))
    print (minAngles.dat.data)
    print ("Time taken: {}s".format(timeTaken))

if __name__ == '__main__':
    main()