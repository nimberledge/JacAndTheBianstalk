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
    #include <Eigen/Dense>
    #include <iostream>

    using namespace Eigen;

    double distance(Vector2d p1, Vector2d p2) {
        return sqrt ( pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2) );
    }

    void getCQM(double *areas, double *minAngles, double *aspectRatios, double *eSkews,
                double *skews, double *scaledJacobians, double *coords) {    
        
        double pi = 3.14159265358979323846;
        // Map vertices as vectors
        Map<Vector2d> V1((double *) &coords[0]);
        Map<Vector2d> V2((double *) &coords[2]);
        Map<Vector2d> V3((double *) &coords[4]);
        
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
    
    m,n = 5, 5
    mesh = UnitSquareMesh(m, n)
    print ("Mesh size: {} x {}".format(m, n))
    print ("Number of cells: {}".format(2 * m * n))
    start = time.time()
    # areas, minAngles, aspectRatios, eSkews, skews, scaledJacobians = getCQM(mesh, include_dirs=include_dirs)
    timeTaken = time.time() - start
    # cqms = np.zeros((areas.dat.data.shape[0], 7))
    metrics = getMetric(mesh, include_dirs)
    print (metrics.dat.data)
    
    # cqms[:, 0] = areas.dat.data
    # cqms[:, 1] = minAngles.dat.data
    # cqms[:, 2] = aspectRatios.dat.data
    # cqms[:, 3] = skews.dat.data
    # cqms[:, 4] = eSkews.dat.data
    # cqms[:, 5] = scaledJacobians.dat.data
    # cqms[:, 6] = metrics.dat.data
    
    
    # print ("Area\t\tMin Angle\tAspect Ratio\tSkewness\tEq. skew\tS. Jacobian\tMetric")
    # for r in range(cqms.shape[0]):
    #     print ('\t'.join(["{:.6f}".format(k) for k in cqms[r, :]]))
    # print ("Time taken: {}s".format(timeTaken))

if __name__ == '__main__':
    main()