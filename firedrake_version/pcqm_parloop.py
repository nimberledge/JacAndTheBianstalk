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
    P0 = FunctionSpace(mesh, "Discontinuous Lagrange", 0)
    V1 = Function(P0_vec)
    V2 = Function(P0_vec)
    V3 = Function(P0_vec)
    vertKernel = """
    V1[0] = coords[0];
    V1[1] = coords[1];
    V2[0] = coords[2];
    V2[1] = coords[3];
    V3[0] = coords[4];
    V3[1] = coords[5];
    """
    par_loop(vertKernel, dx, {'coords': (coords, READ), 'V1': (V1, RW), 'V2': (V2, RW), 'V3': (V3, RW)})
    minAngles = Function(P0)
    cqmKernel = """
    #include <Eigen/Dense>

    using namespace Eigen;

    double distance(Vector2d p1, Vector2d p2) {
        return sqrt ( pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2) );
    }

    void getMinAngles(const double V1_[2], const double V2_[2], const double V3_[2], double *minAngles) {
        // Map vertices as vectors
        Vector2d V1(V1_[0], V1_[1]);
        Vector2d V2(V2_[0], V2_[1]);
        Vector2d V3(V3_[0], V3_[1]);

        Vector2d V12 = V2 - V1;
        Vector2d V13 = V3 - V1;
        Vector2d V23 = V3 - V2;

        double a1 = acos (V12.dot(V13) / (distance(V1, V2) * distance(V1, V3)));
        double a2 = acos (-V12.dot(V23) / (distance(V1, V2) * distance(V2, V3)));
        double a3 = acos (V23.dot(V13) / (distance(V2, V3) * distance(V1, V3)));

        double aMin = std::min(a1, a2);
        minAngles[0] = std::min(aMin, a3);
    }
    """
    kernel = op2.Kernel(cqmKernel, "getMinAngles", cpp=True, include_dirs=include_dirs)
    op2.par_loop(kernel, P0.node_set, V1.dat(op2.READ), V2.dat(op2.READ), V3.dat(op2.READ), minAngles.dat(op2.RW))
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
