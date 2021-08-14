from firedrake import *
from tri_mqc import TriangleMeshQualityCalculator
from cqm_parloop import getCQM
from argparse import ArgumentParser
import time
import sys

def main():
    parser = ArgumentParser(description=\
        'Enter benchmark inputs (m, n, N) to test calculation of mesh quality measures using \
        native Python (with numpy) vs using a C-kernel and Firedrake par_loops. Meshes are \
        2D triangular meshes on the Unit Square with identical elements. \
        Values of CQM should be equal for all elements.')
    parser.add_argument('-m', default=50, type=int, help='Number of cells in the x-direction')
    parser.add_argument('-n', default=50, type=int, help='Number of cells in the y-direction')
    parser.add_argument('-N', default=20, type=int, help='Number of trials for benchmark')
    args = parser.parse_args()
    m, n, num_trials = args.m, args.n, args.N
    
    mesh = UnitSquareMesh(m, n)
    tmqc = TriangleMeshQualityCalculator(mesh)
    cStart, cEnd = tmqc.getCellIndices()
    print ("Mesh size (m x n): {} x {}".format(m, n))
    print ("Number of cells: {}".format(cEnd - cStart))
    print ("Number of trials: {}".format(num_trials))
    
    parLoopTimes = 0
    nativePyTimes = 0
    for t in range(num_trials):
        if t % (num_trials // min(20, num_trials)) == 0:
            print ('.', end='')
            sys.stdout.flush()
        
        start = time.time()
        _ = getCQM(mesh)
        parLoopTimes += time.time() - start
        
        start = time.time()
        for c in range(cStart, cEnd):
            _ = tmqc.getCellQualityMeasures(c)
        nativePyTimes += time.time() - start

    averageParLoopTime = parLoopTimes / num_trials
    averageNativePyTime = nativePyTimes / num_trials
    
    print ("\nAverage time with par_loop: {}s".format(averageParLoopTime))
    print ("Average time with native Python: {}s".format(averageNativePyTime))
    print ("Speedup: {:.1f}x".format(averageNativePyTime / averageParLoopTime))

        

if __name__ == '__main__':
    main()