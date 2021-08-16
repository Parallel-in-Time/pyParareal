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

if __name__ == "__main__":

    Tend     = 8.0
    nslices  = int(Tend) # Make sure each time slice has length 1
    ncoarse  = 1
    nfine    = 1
    u0_val     = np.array([[1.0]], dtype='complex')

    nreal = 35
    nimag = 35
    lambda_real = np.linspace(-1.0, 1.0,  nreal)
    lambda_imag = np.linspace(-1.0, 1.0, nimag)

    sigmin   = np.zeros((nimag,nreal))
    nproc    = Tend
    symb     = -1.0 + 0.0*1j

    # Solution objects define the problem
    u0      = solution_linear(u0_val, np.array([[symb]],dtype='complex'))
    ucoarse = solution_linear(u0_val, np.array([[symb]],dtype='complex'))

    para = parareal(0.0, Tend, nslices, intexact, impeuler, nfine, ncoarse, 0.0, 1, u0)
    E, Mginv = para.get_parareal_matrix()
    print(np.linalg.norm(E.todense(),2))
    for i in range(0,nreal):
      for j in range(0,nimag):
        z = lambda_real[i] + 1j*lambda_imag[j]
        # Use the algorithm on p. 371, Chapter 39 in the Trefethen book
        M = z*sparse.identity(np.shape(E)[0]) - E
        sv = svdvals(M.todense())
        sigmin[j,i] = np.min(sv)
        
#rcParams['figure.figsize'] = 3.54, 3.54
fs = 8
fig, ax  = plt.subplots()
X, Y = np.meshgrid(lambda_real, lambda_imag)
#cset = ax.contour(X, Y, np.log10(sigmin))
cset = ax.contour(X, Y, sigmin)
ax.clabel(cset, fontsize=9, inline=True)
plt.xlabel(r'Real part', fontsize=fs)
plt.ylabel(r'Imaginary part', fontsize=fs)
#filename = 'parareal-sigma-vs-dt.pdf'
#plt.gcf().savefig(filename, bbox_inches='tight')
#call(["pdfcrop", filename, filename])
#fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
