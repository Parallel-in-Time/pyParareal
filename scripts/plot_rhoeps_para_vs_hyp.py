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

if __name__ == "__main__":

    Tend     = 32.0
    nslices  = int(Tend) # Make sure each time slice has length 1
    ncoarse  = 2
    nfine    = 1
    u0_val     = np.array([[1.0]], dtype='complex')
    
    nproc    = Tend
    symb     = -0.0 + 2.0*1j
    symb2    = -1.0 + 0.0*1j
    
    # Solution objects define the problem
    u0      = solution_linear(u0_val, np.array([[symb]],dtype='complex'))
    u02      = solution_linear(u0_val, np.array([[symb2]],dtype='complex'))
    
    para = parareal(0.0, Tend, nslices, intexact, impeuler, nfine, ncoarse, 0.0, 1, u0)
    para2 = parareal(0.0, Tend, nslices, intexact, impeuler, nfine, ncoarse, 0.0, 1, u02)

    E, Mginv = para.get_parareal_matrix()
    E2, Mginv2 = para2.get_parareal_matrix()
    
    epsvec = [0.5, 0.1, 0.05, 0.001]
    nn     = np.size(epsvec)
    rhoeps = np.zeros((nn,2))
    lower_bound = 0.0
    for j in range(nn):
      psr_obj = pseudo_spectral_radius(E, epsvec[j])
      psr, x, tar, cons = psr_obj.get_psr()
      rhoeps[j,0] = psr
      psr_obj = pseudo_spectral_radius(E2, epsvec[j])
      psr, x, tar, cons = psr_obj.get_psr()
      rhoeps[j,1] = psr      

    plt.figure(1)
    plt.semilogx(epsvec, rhoeps[:,0], 'b+-', label=r'Hyperbolic')
    plt.semilogx(epsvec, rhoeps[:,1], 'ro-', label=r'Parabolic')
    plt.xlabel(r'$\varepsilon$')
    plt.ylabel(r'$\rho_{\varepsilon}$')
    plt.legend()
    plt.show()
