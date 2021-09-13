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
    print("Norm of E: %5.3f" % np.linalg.norm(E.todense(),2))
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

'''
Check out Fig. 24.4 on p. 235: plot the difference between rho-eps and rho over eps (for us, this is rho_eps/eps)
'''

'''
NOW COMPUTE THE PSEUDO SPECTRAL RADIUS
'''
epsilon = 1.0

def constraint(x):
    z = x[0] + 1j*x[1]
    M = z*sparse.identity(np.shape(E)[0]) - E
    sv = svdvals(M.todense())
    return np.min(sv)
    
def target(x):
  return 1.0/np.linalg.norm(x, 2)**2

nlc   = NonlinearConstraint(constraint, epsilon-1e-9, epsilon+1e-9)
# for a normal matrix, the epsilon isoline is a circle: therefore, use a point on the circle as starting value for the optimisation
result = minimize(target, [np.sqrt(epsilon), np.sqrt(epsilon)], constraints=nlc, tol = 1e-10, options = {'method': 'BFGS', 'gtol': 1e-10, 'maxiter': 500})
print(result)
plt.plot(result.x[0], result.x[1], 'ko', markersize=fs)
print("Constraint at solution: %5.3f" % constraint(result.x))
print("Target at solution: %5.3f" % target(result.x))
print("Pseudo-spectral-radius: %5.3f" % np.linalg.norm(result.x,2))
plt.show()
