from firedrake import *
from tri_mqc import TriangleMeshQualityCalculator
from cqm_parloop import getCQM
import time
import sys

def main():
    print ("Benchmark testing for par_loop vs Python implementation of CQM calculation.")
    num_trials = 50         # I guess I should make this configurable too
    if len(sys.argv) < 3:
        print ("Defaulting to a benchmark of 50 x 50 cells, with {} trials".format(num_trials))
        m, n = 50, 50
    elif len(sys.argv) == 3:
        m, n = [int(k) for k in sys.argv[1:]]
    elif len(sys.argv) == 4:
        m, n, num_trials = [int(k) for k in sys.argv[1:]]
    else:
        raise ValueError("Illegal number of arguments")

    # print ("m, n: {}, {}".format(m, n))
    mesh = UnitSquareMesh(m, n)
    tmqc = TriangleMeshQualityCalculator(mesh)
    cStart, cEnd = tmqc.getCellIndices()
    print ("Mesh size: {} x {}".format(m, n))
    print ("Number of cells: {}".format(cEnd - cStart))
    print ("Number of trials: {}".format(num_trials))
    
    parLoopTimes = 0
    nativePyTimes = 0
    for t in range(num_trials):
        if t % (num_trials//20) == 0:
            print ('.', end='')
            sys.stdout.flush()
        
        start = time.time()
        _, _, _, _, _, _ = getCQM(mesh)
        parLoopTimes += time.time() - start
        
        start = time.time()
        for c in range(cStart, cEnd):
            _ = tmqc.getCellQualityMeasures(c)
        nativePyTimes += time.time() - start

    averageParLoopTime = parLoopTimes / num_trials
    averageNativePyTime = nativePyTimes / num_trials
    
    print ("\nAverage time with par_loop: {}s".format(averageParLoopTime))
    print ("Average time with native Python: {}s".format(averageNativePyTime))

        

if __name__ == '__main__':
    main()