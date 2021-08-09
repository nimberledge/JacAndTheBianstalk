from firedrake import *
import pprint

def main():
    try:
        from firedrake.slate.slac.compiler import PETSC_ARCH
    except ImportError:
        import os
        PETSC_ARCH = os.path.join(os.environ.get('PETSC_DIR'), os.environ.get('PETSC_ARCH'))
    include_dirs = ["%s/include/eigen3" % PETSC_ARCH]
    print ("Firedrake successfully imported")
    pp = pprint.PrettyPrinter(indent=4)
    m, n = 4, 4
    mesh = UnitSquareMesh(m, n)
    coords = mesh.coordinates
    P0 = VectorFunctionSpace(mesh, "Discontinuous Lagrange", 0)
    """Order 0 vector discontinuous Lagrange space. DOF is associated with element itself."""

    # Compute cell widths using a C kernel
    cell_widths = Function(P0)
    # pp.pprint(cell_widths.__dict__)
    kernel = """
    for (int i=0; i<coords.dofs; i++) {
        widths[0] = fmax(widths[0], fabs(coords[2*i] - coords[(2*i+2)%6]));
        widths[1] = fmax(widths[1], fabs(coords[2*i+1] - coords[(2*i+3)%6]));
    }
    """
    par_loop(kernel, dx, {'coords': (coords, READ), 'widths': (cell_widths, RW)})
    # pp.pprint(cell_widths.__dict__)
    # print (cell_widths.dat.data)
    assert np.allclose(cell_widths.dat.data, [1/m, 1/n])
    print ("Assertion fulfilled")
    # Compute eigendecomposition of a matrix using a C++ kernel
    P0_ten = TensorFunctionSpace(mesh, "DG", 0)
    P0_vec = VectorFunctionSpace(mesh, "DG", 0)
    M = interpolate(as_matrix([[2, -1], [-1, 2]]), P0_ten)
    evecs = Function(P0_ten, name="Eigenvectors")
    evals = Function(P0_vec, name="Eigenvalues")

    kernel = """
    #include <Eigen/Dense>

    using namespace Eigen;

    void get_eigendecomposition(double EVecs_[4], double EVals_[2], const double * M_) {

    // Map inputs and outputs onto Eigen objects
    Map<Matrix<double, 2, 2, RowMajor> > EVecs((double *)EVecs_);
    Map<Vector2d> EVals((double *)EVals_);
    Map<Matrix<double, 2, 2, RowMajor> > M((double *)M_);

    // Solve eigenvalue problem
    SelfAdjointEigenSolver<Matrix<double, 2, 2, RowMajor>> eigensolver(M);
    EVecs = eigensolver.eigenvectors();
    EVals = eigensolver.eigenvalues();
    }
    """
    kernel = op2.Kernel(kernel, "get_eigendecomposition", cpp=True, include_dirs=include_dirs)
    op2.par_loop(kernel, P0_ten.node_set, evecs.dat(op2.RW), evals.dat(op2.RW), M.dat(op2.READ))
    print(evecs.dat.data[0])
    print(evals.dat.data[0])

    # Check the eigendecomposition is valid
    Mv = interpolate(dot(M, evecs), P0_ten)
    vl = interpolate(dot(evecs, as_matrix([[evals[0], 0], [0, evals[1]]])), P0_ten)
    assert np.allclose(Mv.dat.data, vl.dat.data)



if __name__ == '__main__':
    main()