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

'''
TODO: compute pseudo-spectral radius by finding point on isoline with maximum distance from origin
'''
if __name__ == "__main__":

    Tend     = 16.0
    nslices  = int(Tend) # Make sure each time slice has length 1
    ncoarse  = 1
    nfine    = 1
    u0_val     = np.array([[1.0]], dtype='complex')
    
    nproc    = Tend
    symb     = -1.0 + 0.0*1j

    # Solution objects define the problem
    u0      = solution_linear(u0_val, np.array([[symb]],dtype='complex'))
    ucoarse = solution_linear(u0_val, np.array([[symb]],dtype='complex'))

    para = parareal(0.0, Tend, nslices, intexact, impeuler, nfine, ncoarse, 0.0, 1, u0)
    E, Mginv = para.get_parareal_matrix()
    
    epsvec = [2.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.1]
    nn = np.size(epsvec)
    rhoeps = np.zeros(nn)
    for j in range(nn):
      psr_obj = pseudo_spectral_radius(E, epsvec[j])
      psr, x, tar, cons = psr_obj.get_psr()
      rhoeps[j] = psr
      print("Constraint at solution: %5.3f" % cons)
      print("Target at solution: %5.3f" % tar)
      print("Pseudo-spectral-radius: %5.3f" % psr)

    plt.figure(1)
    plt.plot(epsvec, np.divide(rhoeps-1.0, epsvec), '+--')
    plt.xlabel(r'$\varepsilon$')
    plt.ylabel(r'$\rho_{\varepsilon}$')
    plt.show()
  
