import sys
sys.path.append('../src')

from parareal import parareal
from impeuler import impeuler
from intexact import intexact
from trapezoidal import trapezoidal
from special_integrator import special_integrator
from solution_linear import solution_linear
import numpy as np
import scipy.sparse as sparse
from scipy.linalg import svdvals
import math
from numpy import linalg as LA
from scipy.optimize import NonlinearConstraint
from scipy.optimize import minimize

from pylab import rcParams

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from matplotlib.patches import Polygon
from subprocess import call
import sympy
from pylab import rcParams

fs = 8


if __name__ == "__main__":

    Tend     = 32.0
    nslices  = int(Tend) # Make sure each time slice has length 1
    ncoarse  = 5
    nfine    = 1
    u0_val     = np.array([[1.0]], dtype='complex')

    nproc    = Tend
    symb     = 0.0 + 3.0j

    epsilon = 1e-3
    '''
    NOTE: the lower bound is the sup over all epsilons... so finding a epsilon where the lower bound is large will imply non-monotonic convergence
    NOTE: generally, it seems that smaller epsilons represent converge later in the iteration where larger eps correspond to the first few iterations... can we substantiate this?
        --> for larger k, we will eventually recover asymyptotic convergence?
    NOTE: a PSR > 1 for large(ish) values of epsilon might indicate non-monotonic convergence, where the error increases before going down
    '''


    # Solution objects define the problem
    u0      = solution_linear(u0_val, np.array([[symb]],dtype='complex'))
    ucoarse = solution_linear(u0_val, np.array([[symb]],dtype='complex'))

    para = parareal(0.0, Tend, nslices, intexact, impeuler, nfine, ncoarse, 0.0, 1, u0)
    E, Mginv = para.get_parareal_matrix()
    E_norm = np.linalg.norm(E.todense(),2)

    # Find the pseudo spectral radius
    def constraint(x):
        z = x[0] + 1j*x[1]
        M = z*sparse.identity(np.shape(E)[0]) - E
        sv = svdvals(M.todense())
        return np.min(sv)

    def target(x):
      return 1.0/np.linalg.norm(x, 2)**2

    nlc   = NonlinearConstraint(constraint, epsilon-1e-9, epsilon+1e-9)
    # for a normal matrix, the epsilon isoline is a circle: therefore, use a point on the circle as starting value for the optimisation
    result = minimize(target, [np.sqrt(epsilon), np.sqrt(epsilon)], constraints=nlc, tol = 1e-10, method='trust-constr', options = {'xtol': 1e-10, 'gtol': 1e-10, 'maxiter': 1500})
    print(result.message)
    print("Constraint at solution: %5.3f" % constraint(result.x))
    #print("Target at solution:     %5.3f" % target(result.x))
    psr = np.linalg.norm(result.x, 2)
    print("Pseudospectralradius:   %5.3f" % psr)
    # Now compute powers of E
    power_norms = np.zeros((7, int(nproc)))
    for k in range(int(nproc)):
      E_power_k = LA.matrix_power(E.todense(), k+1)
      power_norms[0,k] = np.linalg.norm(E_power_k, 2)
      power_norms[1,k] = psr**(k+2)/epsilon
      power_norms[2,k] = (psr-1.0)/epsilon
      #power_norms[2,k] = psr**(k+1) - ( (E_norm + epsilon)**(k+1) - E_norm**(k+1) ) # (16.25) on p. 163
      power_norms[3,k] = E_norm**(k+1)
    #power_norms[4, :] = para.get_linear_bound(int(nproc)-1,mgritTerm=True)
    #power_norms[5, :] = para.get_superlinear_bound(int(nproc)-1,bruteForce=False)
    #power_norms[6, :] = para.get_superlinear_bound(int(nproc)-1,bruteForce=True)
    #power_norms[4:, :] *= power_norms[0,0]

    plt.figure(1)
    iters = range(int(nproc))
    plt.semilogy(iters, power_norms[0,:], 'bo--', label=r'|| E^k ||')
    plt.semilogy(iters, power_norms[1,:], 'r-', label='PSR')
    plt.semilogy(iters, power_norms[2,:], 'g-', label='Lower bound for sup|| E^k||')
    plt.semilogy(iters, power_norms[3,:], 'c--', label='Norm')
    #plt.semilogy(iters, power_norms[4,:], 'p:', label='Linear')
    #plt.semilogy(iters, power_norms[5,:], 's:', label='Superlinear (Gander)')
    #plt.semilogy(iters, power_norms[6,:], '>:', label='Superlinear (Tibo)')
    plt.ylim([1e-16, 1e5])
    plt.title((r'$\varepsilon$ = %5.3e' % epsilon), fontsize=fs)
    #print(power_norms[2,:])
    plt.legend()
    filename = 'parareal-psr-conv.pdf'
    plt.gcf().savefig(filename, bbox_inches='tight')
    call(["pdfcrop", filename, filename])
    plt.show()
