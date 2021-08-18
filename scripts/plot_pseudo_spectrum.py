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

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from matplotlib.patches import Polygon
from subprocess import call
import sympy
from pylab import rcParams

'''
TODO: compute pseudo-spectral radius by finding point on isoline with maximum distance from origin
'''
if __name__ == "__main__":

    Tend     = 64.0
    nslices  = int(Tend) # Make sure each time slice has length 1
    ncoarse  = 1
    nfine    = 1
    u0_val     = np.array([[1.0]], dtype='complex')

    nreal = 40
    nimag = 40
    lambda_real = np.linspace(-2.5, 2.5,  nreal)
    lambda_imag = np.linspace(-2.5, 2.5, nimag)

    sigmin   = np.zeros((nimag,nreal))
    circs = np.zeros((nimag,nreal))
    nproc    = Tend
    symb     = 0.0 + 1.0*1j

    # Solution objects define the problem
    u0      = solution_linear(u0_val, np.array([[symb]],dtype='complex'))
    ucoarse = solution_linear(u0_val, np.array([[symb]],dtype='complex'))

    para = parareal(0.0, Tend, nslices, intexact, impeuler, nfine, ncoarse, 0.0, 1, u0)
    E, Mginv = para.get_parareal_matrix()
    eigs = np.linalg.eigvals(E.todense())
    print(np.linalg.norm(E.todense(),2))
    for i in range(0,nreal):
      for j in range(0,nimag):
        z = lambda_real[i] + 1j*lambda_imag[j]
        # Use the algorithm on p. 371, Chapter 39 in the Trefethen book
        M = z*sparse.identity(np.shape(E)[0]) - E
        sv = svdvals(M.todense())
        sigmin[j,i] = np.min(sv)
        circs[j,i]  = np.sqrt(lambda_real[i]**2 + lambda_imag[j]**2)
#rcParams['figure.figsize'] = 3.54, 3.54
fs = 8
fig, ax  = plt.subplots()
X, Y = np.meshgrid(lambda_real, lambda_imag)
#cset = ax.contour(X, Y, np.log10(sigmin))
cset = ax.contour(X, Y, sigmin)
ax.clabel(cset, fontsize=9, inline=True)
cset2 = ax.contour(X, Y, circs, linestyles='dotted')
ax.clabel(cset2, fontsize=9, inline=True)
plt.xlabel(r'Real part', fontsize=fs)
plt.ylabel(r'Imaginary part', fontsize=fs)
plt.title(r'$1/|| (z - E)^{-1} \||_2$')
ax.plot(0.0, 0.0, 'k+', markersize=fs)
#filename = 'parareal-sigma-vs-dt.pdf'
#plt.gcf().savefig(filename, bbox_inches='tight')
#call(["pdfcrop", filename, filename])
#fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

'''
Trefethen page 18: for a normal matrix, the eps-pseudospectrum is the union of eps-balls around the eigenvalues.
Since all eigenvalues of E are zero, if E were normal, the pseudo-spectrum would be the radius eps unit ball
'''

'''
Check Theorem 16.1 on p. 158: there is M, gamma such that

  || E^k || <= (rho_eps(E))^(k+1)/eps

for any (!) eps

Here, rho_eps(E) = sup{ |z| : z in sigma_eps(E) }.

That will be the point on some isoline that is furthest away from the origin

'''

'''
Theorem 16.6. on p. 164: E^N = 0 is equivalent to spectrum(E) = {0} and

  || (z - A)^(-1) || = O(1/|z|^{N})

as |z| -> 0 which implies

  1/||(z-A)^-1|| = O(|z|^N)

Therefore, we expect the norm of ||(z-A)^-1|| to grow faster as we move towards the origin if the number of time slices increases
'''

'''
Theorem 16.4 on p. 160 gives a lower bound (!)

sup || E^k || >= (rho_eps(E)-1)/eps
k>=0

This would give an idea when E^k grows before collapsing towards zero.

Plot || E^k || versus the upper and lower bound provided by rho_eps(E) for various eps
'''

'''
Check out Fig. 24.4 on p. 235: plot the difference between rho-eps and rho over eps (for us, this is rho_eps/eps)
'''
