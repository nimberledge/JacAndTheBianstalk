from firedrake import *
import time

def getCQM(mesh, include_dirs, P0=None, P0_ten=None, P0_vec=None):
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
    areas = Function(P0)
    minAngles = Function(P0)
    aspectRatios = Function(P0)
    eSkews = Function(P0)
    skews = Function(P0)
    scaledJacobians = Function(P0)
    cqmKernel = """
    #include <Eigen/Dense>

    using namespace Eigen;

    double distance(Vector2d p1, Vector2d p2) {
        return sqrt ( pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2) );
    }

    void getCQM(const double V1_[2], const double V2_[2], const double V3_[2], 
                double *areas, double *minAngles, double *aspectRatios, double *eSkews,
                double *skews, double *scaledJacobians)
    {    
        // Map vertices as vectors
        Vector2d V1(V1_[0], V1_[1]);
        Vector2d V2(V2_[0], V2_[1]);
        Vector2d V3(V3_[0], V3_[1]);
        double pi = 3.14159265358979323846;

        // Precompute some vectors, and distances
        Vector2d V12 = V2 - V1;
        Vector2d V13 = V3 - V1;
        Vector2d V23 = V3 - V2;
        double d12 = distance(V1, V2);
        double d13 = distance(V1, V3);
        double d23 = distance(V2, V3);
        double s = (d12 + d23 + d13) / 2;

        // Scaled Jacobian
        // TODO: Maybe refactor this into having a determinant calculation...
        // Stuff that in another function so that I can use it for tets
        double sj1 = std::abs(V12[0] * V13[1] - V13[0]*V12[1]) / (d12 * d13);
        double sj2 = std::abs(V12[0] * V23[1] - V23[0]*V12[1]) / (d12 * d23);
        double sj3 = std::abs(V23[0] * V13[1] - V13[0]*V23[1]) / (d13 * d23);
        scaledJacobians[0] = std::min(sj1, sj2);
        scaledJacobians[0] = std::min(scaledJacobians[0], sj3);

        // Minimum angle
        double a1 = acos (V12.dot(V13) / (d12 * d13));
        double a2 = acos (-V12.dot(V23) / (d12 * d23));
        double a3 = acos (V23.dot(V13) / (d23 * d13));
        double aMin = std::min(a1, a2);
        minAngles[0] = std::min(aMin, a3);

        // eSkew
        double aMax = std::max(a1, a2);
        aMax = std::max(aMax, a3);
        double aIdeal = pi / 3;
        eSkews[0] = std::max((aMax - aIdeal / (pi - aIdeal)), (aIdeal - minAngles[0]) / aIdeal);

        // Area
        areas[0] = sqrt(s * (s-d12) * (s-d13) * (s-d23));

        // Aspect Ratio
        aspectRatios[0] = (d12 * d23 * d13) / (8 * (s - d12) * (s - d23) * (s - d13));

        // Skewness
        Vector2d midPoint1 = V2 + (V3 - V2) / 2;
        Vector2d midPoint2 = V3 + (V1 - V3) / 2;
        Vector2d midPoint3 = V1 + (V2 - V1) / 2;

        Vector2d lineNormal1 = midPoint1 - V1;
        Vector2d lineOrth1 = midPoint3 - midPoint2;
        double t1 = acos (lineNormal1.dot(lineOrth1) / (distance(V1, midPoint1) * distance(midPoint2, midPoint3)));
        double t2 = pi - t1;
        double tMin = std::min(t1, t2);

        Vector2d lineNormal2 = midPoint2 - V2;
        Vector2d lineOrth2 = midPoint1 - midPoint3;
        double t3 = acos (lineNormal2.dot(lineOrth2) / (distance(V2, midPoint2) * distance(midPoint1, midPoint3)));
        double t4 = std::min(t3, pi - t3);
        tMin = std::min(tMin, t4);

        Vector2d lineNormal3 = midPoint3 - V3;
        Vector2d lineOrth3 = midPoint2 - midPoint1;
        double t5 = acos (lineNormal3.dot(lineOrth3) / (distance(V3, midPoint3) * distance(midPoint1, midPoint2)));
        double t6 = std::min(t3, pi - t5);
        tMin = std::min(tMin, t6);

        skews[0] = pi/2 - tMin;
    }
    """
    kernel = op2.Kernel(cqmKernel, "getCQM", cpp=True, include_dirs=include_dirs)
    op2.par_loop(kernel, P0.node_set, V1.dat(op2.READ), V2.dat(op2.READ), V3.dat(op2.READ), \
        areas.dat(op2.RW), minAngles.dat(op2.RW), aspectRatios.dat(op2.RW), eSkews.dat(op2.RW), \
        skews.dat(op2.RW), scaledJacobians.dat(op2.RW))
    return (areas, minAngles, aspectRatios, eSkews, skews, scaledJacobians)

def main():
    try:
        from firedrake.slate.slac.compiler import PETSC_ARCH
    except ImportError:
        import os
        PETSC_ARCH = os.path.join(os.environ.get('PETSC_DIR'), os.environ.get('PETSC_ARCH'))
    include_dirs = ["%s/include/eigen3" % PETSC_ARCH]
    print ("Firedrake successfully imported")
    m,n = 2, 2
    mesh = UnitSquareMesh(m, n)
    print ("Import successful")
    # areas = computeArea(mesh)
    # minAngles = computeMinAngle(mesh)
    # aspectRatios = computeAspectRatio(mesh)
    # equiangleSkews = computeEquiangleSkew(mesh)
    # scaledJacobians = computeScaledJacobian(mesh)
    # skews = computeSkewness(mesh)

    start = time.time()
    areas, minAngles, aspectRatios, eSkews, skews, scaledJacobians = getCQM(mesh, include_dirs=include_dirs)
    timeTaken = time.time() - start
    cqms = np.zeros((areas.dat.data.shape[0], 6))
    
    cqms[:, 0] = areas.dat.data
    cqms[:, 1] = minAngles.dat.data
    cqms[:, 2] = aspectRatios.dat.data
    cqms[:, 3] = skews.dat.data
    cqms[:, 4] = eSkews.dat.data
    cqms[:, 5] = scaledJacobians.dat.data
    
    print ("Mesh size: {} x {}".format(m, n))
    print ("Number of cells: {}".format(areas.dat.data.shape[0]))
    print ("Area\t\tMin Angle\tAspect Ratio\tSkewness\tEq. skew\tScaled Jacobian")
    for r in range(cqms.shape[0]):
        print ('\t'.join(["{:.6f}".format(k) for k in cqms[r, :]]))
    print ("Time taken: {}s".format(timeTaken))

if __name__ == '__main__':
    main()
