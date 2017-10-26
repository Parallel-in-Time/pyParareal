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
    dx       = 1.0
    u0_val     = np.array([[1.0]], dtype='complex')

    nreal = 4
    nimag = 3
    lambda_real = np.linspace(-2.0, 0.0, nreal)
    lambda_imag = np.linspace(0.0,  5.0, nimag)

    svds = np.zeros((nimag,nreal))
    speedup = np.zeros((nimag,nreal))
    tolerance = 1e-2
    nproc     = Tend

    for i in range(0,nreal):
      for j in range(0,nimag):
        symb = lambda_real[i] + 1j*lambda_imag[j]
        symb_coarse = symb

        # Solution objects define the problem
        u0      = solution_linear(u0_val, np.array([[symb]],dtype='complex'))
        ucoarse = solution_linear(u0_val, np.array([[symb_coarse]],dtype='complex'))

        para = parareal(0.0, Tend, nslices, intexact, impeuler, nfine, ncoarse, 0.0, 1, u0)
        svds[j,i] = para.get_max_svd(ucoarse=ucoarse)

        kiter = np.floor( np.log(tolerance)/np.log(svds[j,i]) )
        coarse_to_fine = float(ncoarse)/float(nfine)
        speedup[j,i] = 1.0/( (1 + kiter/nproc)*coarse_to_fine + kiter/nproc )
  
rcParams['figure.figsize'] = 3.54, 3.54
fs = 8
fig  = plt.figure()
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(lambda_real, lambda_imag)
surf = ax.plot_surface(X, Y, svds, cmap=cm.coolwarm, linewidth=0, antialiased=False)
cset = ax.contour(X, Y, svds)
plt.xlabel(r'Real part', fontsize=fs)
plt.ylabel(r'Imaginary part', fontsize=fs)
#filename = 'parareal-sigma-vs-dt.pdf'
#plt.gcf().savefig(filename, bbox_inches='tight')
#call(["pdfcrop", filename, filename])
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
