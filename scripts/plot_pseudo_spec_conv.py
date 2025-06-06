import sys
sys.path.append('../src')
sys.path.append('./spatial_coarsening')

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
from get_matrix import get_upwind, get_centered, get_diffusion
from pseudo_spectral_radius import pseudo_spectral_radius

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

    Tend     = 8.0
    nslices  = int(Tend) # Make sure each time slice has length 1
    ncoarse  = 1 # coarse time steps per slice
    nfine    = 20 # fine time steps per slice
    
    ndof_fine   = 16 # fine spatial resolution
    ndof_coarse = 8  # coarse spatial resolution
    
    # ... use something more interesting as initial value
    u0_val     = np.ones(ndof_fine, dtype='complex')
    u0_c_val   = np.ones(ndof_coarse, dtype='complex')
    
    nproc    = Tend
    
    # CAREFUL: rn, meshtransfer class assumes we operate on the unit interval
    xaxis_f = np.linspace(0.0, 1.0, ndof_fine, endpoint=True)
    h_f = xaxis_f[1] - xaxis_f[0]
    xaxis_c = np.linspace(0.0, 1.0, ndof_coarse, endpoint=True)      
    h_c = xaxis_c[1] - xaxis_c[0]
      
    symb   = get_centered(ndof_fine, h_f)
    symb_c = get_centered(ndof_coarse, h_c)
        
    epsilon = 0.1
    '''
    NOTE: the lower bound is the sup over all epsilons... so finding a epsilon where the lower bound is large will imply non-monotonic convergence
    NOTE: generally, it seems that smaller epsilons represent converge later in the iteration where larger eps correspond to the first few iterations... can we substantiate this?
        --> for larger k, we will eventually recover asymyptotic convergence?
    NOTE: a PSR > 1 for large(ish) values of epsilon might indicate non-monotonic convergence, where the error increases before going down
    '''


    # Solution objects define the problem
    u0      = solution_linear(u0_val, symb)
    ucoarse = solution_linear(u0_c_val, symb_c)

    para = parareal(0.0, Tend, nslices, impeuler, impeuler, nfine, ncoarse, 0.0, 1, u0)
    E, Mginv = para.get_parareal_matrix()
    E_norm = np.linalg.norm(E.todense(),2)

    # Find the pseudo spectral radius
    psr_e = pseudo_spectral_radius(E, epsilon)
    psr, x, target, constraint = psr_e.get_psr(verbose=True)
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
