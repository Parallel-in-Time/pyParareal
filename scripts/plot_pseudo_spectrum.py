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

    nreal = 90
    nimag = 90
    lambda_real = np.linspace(-8.0, 8.0,  nreal)
    lambda_imag = np.linspace(-3.0, 6.0, nimag)

    sigmin   = np.zeros((nimag,nreal))
    circs = np.zeros((nimag,nreal))
    nproc    = Tend
    symb     = 0.0 + 3.0*1j

    # Solution objects define the problem
    u0      = solution_linear(u0_val, np.array([[symb]],dtype='complex'))
    ucoarse = solution_linear(u0_val, np.array([[symb]],dtype='complex'))

    para = parareal(0.0, Tend, nslices, intexact, trapezoidal, nfine, ncoarse, 0.0, 1, u0)
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
filename = 'parareal-pseudospectrum.pdf'
plt.gcf().savefig(filename, bbox_inches='tight')
call(["pdfcrop", filename, filename])
#fig.colorbar(surf, shrink=0.5, aspect=5)

'''
Check out Fig. 24.4 on p. 235: plot the difference between rho-eps and rho over eps (for us, this is rho_eps/eps)
'''

'''
NOW COMPUTE THE PSEUDO SPECTRAL RADIUS
'''
psr_obj = pseudo_spectral_radius(E, 0.1)
psr, x, tar, cons = psr_obj.get_psr()
plt.plot(x[0], x[1], 'ko', markersize=fs)
print("Constraint at solution: %5.3f" % cons)
print("Target at solution: %5.3f" % tar)
print("Pseudo-spectral-radius: %5.3f" % psr)
plt.show()
