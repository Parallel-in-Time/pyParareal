import sys
sys.path.append('../src')
sys.path.append('./spatial_coarsening')

from parareal import parareal
from impeuler import impeuler
from intexact import intexact
from trapezoidal import trapezoidal
from special_integrator import special_integrator
from solution_linear import solution_linear
from get_matrix import get_upwind, get_centered, get_diffusion
import numpy as np
from numpy import linalg as LA
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

    case = 2

    Tend     = 16.0
    nslices  = int(Tend) # Make sure each time slice has length 1
    ncoarse  = 64 # spatial resolution for coarse propagator
    nfine    = 64 # spatial resolution for fine propagator
    u0_val     = np.ones(nfine, dtype='complex')
    u0_c_val   = np.ones(ncoarse, dtype='complex')
    
    nreal = 20
    nimag = 20
    
    sigma = 0.5
    
    if case==0:
        lambda_real = np.linspace(-8.0, 8.0,  nreal)
        lambda_imag = np.linspace(-3.0, 6.0, nimag)
        symb   = (0.0 + 3.*1j)*sparse.identity(nfine, dtype='complex')
        symb_c = (0.0 + 3.*1j)*sparse.identity(ncoarse, dtype='complex')
        
        filename = 'parareal-pseudospectrum-imag.pdf'        
    elif case==1:
        lambda_real = np.linspace(-1.25, 1.25,  nreal)
        lambda_imag = np.linspace(-1.25, 1.25, nimag)
        symb   = (-1.0 + 0.*1j)*sparse.identity(nfine, dtype='complex')
        symb_c = (-1.0 + 0.*1j)*sparse.identity(ncoarse, dtype='complex')        
        filename = 'parareal-pseudospectrum-real.pdf'    
    elif case==2:
        lambda_real = np.linspace(-3.0, 3.0,  nreal)
        lambda_imag = np.linspace(-3.0, 3.0, nimag)
        # CAREFUL: rn, meshtransfer class assumes we operate on the unit interval
        xaxis_f = np.linspace(0.0, 1.0, nfine, endpoint=True)
        h_f = xaxis_f[1] - xaxis_f[0]
        xaxis_c = np.linspace(0.0, 1.0, ncoarse, endpoint=True)      
        h_c = xaxis_c[1] - xaxis_c[0]
        symb = get_centered(nfine, h_f)
        symb_c = get_upwind(ncoarse, h_c)
        filename = 'parareal-pseudospectrum-advection.pdf'  
    elif case==3:
        lambda_real = np.linspace(-3.0, 3.0,  nreal)
        lambda_imag = np.linspace(-3.0, 3.0, nimag)      
        # CAREFUL: rn, meshtransfer class assumes we operate on the unit interval
        xaxis_f = np.linspace(0.0, 1.0, nfine, endpoint=True)
        h_f = xaxis_f[1] - xaxis_f[0]
        xaxis_c = np.linspace(0.0, 1.0, ncoarse, endpoint=True)      
        h_c = xaxis_c[1] - xaxis_c[0]
        symb = get_diffusion(nfine, h_f)
        symb_c = get_diffusion(ncoarse, h_c)
        filename = 'parareal-pseudospectrum-advection.pdf'  
        
    sigmin   = np.zeros((nimag,nreal))
    circs = np.zeros((nimag,nreal))
    nproc    = Tend

    # Solution objects define the problem
    u0      = solution_linear(u0_val, symb)
    ucoarse = solution_linear(u0_c_val, symb_c)

    para = parareal(0.0, Tend, nslices, impeuler, impeuler, 50, 5, 0.0, 1, u0)
    E, Mginv = para.get_parareal_matrix()
    D = E*E.H - E.H*E
    print("Normality number for E: %5.3e" % np.linalg.norm(D.todense()))
    
    sv = svdvals(E.todense())

    '''
    Diffusive problems have (i) very small normality number, (ii) a small pseudo spectral radius and (iii) a small norm
    Also, the eps-isolines are very much circles.
    QUESTION: can we work out the D matrix above and say something about how it looks like for diffusive/non-diffusive problems?
    '''
    eigs = np.linalg.eigvals(E.todense())
    print("Norm of E: %5.3f" % np.linalg.norm(E.todense(),2))
    for i in range(0,nreal):
      for j in range(0,nimag):
        z = lambda_real[i] + 1j*lambda_imag[j]
        # Use the algorithm on p. 371, Chapter 39 in the Trefethen book
        M = z*sparse.identity(np.shape(E)[0]) - E
        sv = svdvals(M.todense())
        sigmin[j,i] = np.min(sv)
        circs[j,i]  = np.sqrt(lambda_real[i]**2 + lambda_imag[j]**2)
        if np.min(sv) > abs(z):
          print("You were wrong!!!")
          print("sv-min: %5.3e" % np.min(sv))
          print("abs(z): %5.3e" % abs(z))
#rcParams['figure.figsize'] = 3.54, 3.54
fs = 8
fig, ax  = plt.subplots()
X, Y = np.meshgrid(lambda_real, lambda_imag)
#cset = ax.contour(X, Y, np.log10(sigmin))
lvls = np.linspace(0.0, 1.0, 11)
cset = ax.contour(X, Y, sigmin, levels=lvls)
ax.clabel(cset, fontsize=9, inline=True)
#cset2 = ax.contour(X, Y, circs, linestyles='dotted')
#ax.clabel(cset2, fontsize=9, inline=True)
plt.xlabel(r'Real part', fontsize=fs)
plt.ylabel(r'Imaginary part', fontsize=fs)
plt.title(r'$1/|| (z - E)^{-1} \||_2$')
ax.plot(0.0, 0.0, 'k+', markersize=fs)


'''
Check out Fig. 24.4 on p. 235: plot the difference between rho-eps and rho over eps (for us, this is rho_eps/eps)
'''

'''
NOW COMPUTE THE PSEUDO SPECTRAL RADIUS
'''
psr_obj = pseudo_spectral_radius(E, sigma)
psr, x, tar, cons = psr_obj.get_psr()
plt.plot(x[0], x[1], 'ko', markersize=fs)
print("Constraint at solution: %5.3f" % cons)
print("Target at solution: %5.3f" % tar)
print("Pseudo-spectral-radius: %5.3f" % psr)

plt.gcf().savefig(filename, bbox_inches='tight')
#call(["pdfcrop", filename, filename])
#fig.colorbar(surf, shrink=0.5, aspect=5)

'''
COMPUTE AND PLOT E^k
'''
power_norms = np.zeros(nslices)
for i in range(nslices):
  power_norms[i] = np.linalg.norm( LA.matrix_power(E.todense(), i+1), 2)
  
plt.figure(2)
plt.semilogy(range(nslices), power_norms, 'o')
plt.xlabel('k')
plt.ylabel(r'$||E^k||_2$')
plt.show()
