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

    Tend     = 32.0
    nslices  = int(Tend) # Make sure each time slice has length 1
    ncoarse  = 2
    nfine    = 1
    u0_val     = np.array([[1.0]], dtype='complex')
    
    nproc    = Tend
    symb     = -1.0 + 0.0*1j

    # Solution objects define the problem
    u0      = solution_linear(u0_val, np.array([[symb]],dtype='complex'))
    ucoarse = solution_linear(u0_val, np.array([[symb]],dtype='complex'))

    para = parareal(0.0, Tend, nslices, intexact, impeuler, nfine, ncoarse, 0.0, 1, u0)
    E, Mginv = para.get_parareal_matrix()
    
    epsvec = [0.1, 0.05, 0.001]
    nn     = np.size(epsvec)
    rhoeps = np.zeros(nn)
    lower_bound = 0.0
    for j in range(nn):
      psr_obj = pseudo_spectral_radius(E, epsvec[j])
      psr, x, tar, cons = psr_obj.get_psr()
      rhoeps[j] = psr
      print("Constraint at solution: %5.3f" % cons)
      print("Target at solution: %5.3f" % tar)
      print("Pseudo-spectral-radius: %5.3f" % psr)
      lower_bound = max(lower_bound, (rhoeps[j]-1.0)/rhoeps[j])
    #plt.figure(1)
    #plt.semilogx(epsvec, rhoeps, '+--', label=r'$\rho_{\varepsilon}$')
    #plt.semilogx(epsvec, np.maximum(np.zeros(nn), np.divide(rhoeps - 1.0, epsvec)), 'rx-', label=r'$\max(0,(\rho_{\varepsilon}-1)/\varepsilon)$')
    #plt.xlabel(r'$\varepsilon$')
    #plt.ylabel(r'$\rho_{\varepsilon}$')
    #plt.legend()
    
    niter = 24
    bounds = np.zeros((nn+1,niter))
    E_power_k = E
    for j in range(niter):
      bounds[0,j] = np.linalg.norm( E_power_k.todense() )
      E_power_k = E@E_power_k
      for k in range(nn):
        bounds[k+1,j] = rhoeps[k]**(j+1)/epsvec[k]
     
    eps_index_1 = 1
    eps_index_2 = 2
    eps_index_3 = 3
    
    eps_iter_1 = 3
    eps_iter_2 = 9
    eps_iter_3 = 16

    plt.figure(2)
    plt.semilogy(range(eps_iter_3,eps_iter_3+4),  (bounds[0,eps_iter_3]/bounds[eps_index_3,eps_iter_3])*bounds[eps_index_3, eps_iter_3:eps_iter_3+4], 'b-', label = r'$\rho_{\varepsilon}^{k+1} / \varepsilon$ for $\rho_{\varepsilon} = $' + str(epsvec[eps_index_3-1]), linewidth=2.0)
    plt.semilogy(range(eps_iter_2,eps_iter_2+4),   (bounds[0,eps_iter_2]/bounds[eps_index_2,eps_iter_2])*bounds[eps_index_2, eps_iter_2:eps_iter_2+4],  'b--', label = r'$\rho_{\varepsilon}^{k+1} / \varepsilon$ for $\rho_{\varepsilon} = $' + str(epsvec[eps_index_2-1]), linewidth=2.0)
    plt.semilogy(range(eps_iter_1,eps_iter_1+4),   (bounds[0,eps_iter_1]/bounds[eps_index_1,eps_iter_1])*bounds[eps_index_1, eps_iter_1:eps_iter_1+4],  'b-.', label = r'$\rho_{\varepsilon}^{k+1} / \varepsilon$ for $\rho_{\varepsilon} = $' + str(epsvec[eps_index_1-1]), linewidth=2.0)
    plt.semilogy(range(niter), bounds[0,:], 'r', label = r'$\left\|| E^k \right\||_2$')
    plt.semilogy(range(niter), np.zeros(niter) + lower_bound, 'g', label = r'$(\rho_{\varepsilon} - 1 ) / \rho_{\varepsilon}$')
    plt.xlabel('Parareal Iteration')
    plt.ylabel(r'$\left|| A^k \right||$')
    #plt.ylim([1e-2, 1e1])
    plt.legend()
    
    
filename = 'parareal-rhoeps-conv.pdf'
plt.gcf().savefig(filename, bbox_inches='tight')
plt.show() 
