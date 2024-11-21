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
from get_matrix import get_upwind, get_centered

from pylab import rcParams
from scipy.optimize import NonlinearConstraint
from scipy.optimize import minimize

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from matplotlib.patches import Polygon
from subprocess import call
import sympy
from pylab import rcParams

from pseudo_spectral_radius import pseudo_spectral_radius

'''
TODO: compute pseudo-spectral radius by finding point on isoline with maximum distance from origin
'''
if __name__ == "__main__":
  
    Tend    = 1.0
    nslices = 10
    tol     = 0.0
    maxiter = 9
    nfine   = 10
    ncoarse = 1
    
    ndof_f   = 32
    ndof_c   = 24

    xaxis_f = np.linspace(0.0, 2.0, ndof_f+1)[0:ndof_f]
    dx_f    = xaxis_f[1] - xaxis_f[0]

    xaxis_c = np.linspace(0.0, 2.0, ndof_c+1)[0:ndof_c]
    dx_c = xaxis_c[1] - xaxis_c[0]

    # 1 = advection with implicit Euler / upwind FD
    # 2 = advection with trapezoidal rule / centered FD
    problem      = 2

    if problem==1:
      A_f = get_upwind(ndof_f, dx_f)
      A_c = get_upwind(ndof_c, dx_c)
    elif problem==2:
      A_f = get_centered(ndof_f, dx_f)
      A_c = get_centered(ndof_c, dx_c)
    else:
      quit()
      
    ### Shape of plot looks similar but PSRs are > 1 for problem 2 and < 1 for problem 1
  
    D = A_f*A_f.H - A_f.H*A_f
    print("Normality number of the system matrix (this should be zero): %5.3f" % np.linalg.norm(D.todense()))


    u0fine     = solution_linear(np.zeros(ndof_f), A_f)
    u0coarse = solution_linear(np.zeros(ndof_c), A_c)

    if problem==1:
      para     = parareal(0.0, Tend, nslices, impeuler, impeuler, nfine, ncoarse, tol, maxiter, u0fine, u0coarse)
    elif problem==2:
      para     = parareal(0.0, Tend, nslices, trapezoidal, trapezoidal, nfine, ncoarse, tol, maxiter, u0fine, u0coarse)
    else:
      quit()

    E, Mginv = para.get_parareal_matrix()
    
    epsvec = np.linspace(0.1, 0.5, 6)
    nn     = np.size(epsvec)
    rhoeps = np.zeros(nn)
    for j in range(nn):
      psr_obj = pseudo_spectral_radius(E, epsvec[j])
      psr, x, tar, cons = psr_obj.get_psr()
      rhoeps[j] = np.max(0.0, (psr-1.0)/epsvec[j])
      print("Constraint at solution: %5.3f" % cons)
      print("Target at solution: %5.3f" % tar)
      print("Pseudo-spectral-radius: %5.3f" % psr)
     
    plt.figure(1)
    plt.plot(epsvec, rhoeps, 'bo-')
    
    plt.xlabel(r'$\epsilon$')
    plt.ylabel(r'$(\rho_{\epsilon}(E)-1)/\epsilon$')
    filename = 'parareal-rhoeps-conv.pdf'
    plt.gcf().savefig(filename, bbox_inches='tight')
    plt.show() 
