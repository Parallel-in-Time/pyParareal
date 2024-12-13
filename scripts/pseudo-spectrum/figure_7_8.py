import sys
sys.path.append('../../src')

from parareal import parareal
from impeuler import impeuler
from intexact import intexact
from trapezoidal import trapezoidal
from special_integrator import special_integrator
from solution_linear import solution_linear
from get_matrix import get_upwind, get_centered, get_diffusion
import numpy as np
import scipy.sparse as sparse
from scipy.linalg import svdvals
import math

import matplotlib.pyplot as plt
from subprocess import call
from pylab import rcParams

from pseudo_spectral_radius import pseudo_spectral_radius


Tend    = 1.0
nslices = 10
tol     = 0.0
maxiter = 9
nfine   = 10
ncoarse = 1

ndof_f   = 32
ndof_c   = 24

epsilon = 0.1

xaxis_f = np.linspace(0.0, 2.0, ndof_f+1)[0:ndof_f]
dx_f    = xaxis_f[1] - xaxis_f[0]

xaxis_c = np.linspace(0.0, 2.0, ndof_c+1)[0:ndof_c]
dx_c = xaxis_c[1] - xaxis_c[0]

# 1 = advection with implicit Euler / upwind FD
# 2 = advection with trapezoidal rule / centered FD
try:
  figure      =  int(sys.argv[1]) # 7 generates figure_7, 8 generates figure_8
except:
  print("No or wrong command line argument provided, creating figure 7. Use 7 or 8 as command line argument.")
  figure = 7
  
if figure==7:
  A_f = get_upwind(ndof_f, dx_f)
  A_c = get_upwind(ndof_c, dx_c)
  
elif figure==8:
  A_f = get_centered(ndof_f, dx_f)
  A_c = get_centered(ndof_c, dx_c)
 
else:
  quit()
  
D = A_f*A_f.H - A_f.H*A_f
print("Normality number of the system matrix (this should be zero): %5.3f" % np.linalg.norm(D.todense()))
u0fine   = solution_linear(np.zeros(ndof_f), A_f)
u0coarse = solution_linear(np.zeros(ndof_c), A_c)

if figure==7:
  para     = parareal(0.0, Tend, nslices, impeuler, impeuler, nfine, ncoarse, tol, maxiter, u0fine, u0coarse)
elif figure==8:
  para     = parareal(0.0, Tend, nslices, trapezoidal, trapezoidal, nfine, ncoarse, tol, maxiter, u0fine, u0coarse)
Pmat, Bmat = para.get_parareal_matrix()

nreal = 20
nimag = 20
lambda_real = np.linspace(-3.0, 3.0,  nreal)
lambda_imag = np.linspace(-3.0, 3.0, nimag)
sigmin = np.zeros((nimag,nreal))
circs  = np.zeros((nimag,nreal))

   
'''
Diffusive problems have (i) very small normality number, (ii) a small pseudo spectral radius and (iii) a small norm
Also, the eps-isolines are very much circles.
QUESTION: can we work out the D matrix above and say something about how it looks like for diffusive/non-diffusive problems?
'''
print("Norm of E: %5.3f" % np.linalg.norm(Pmat.todense(),2))
for i in range(0,nreal):
  for j in range(0,nimag):
    z = lambda_real[i] + 1j*lambda_imag[j]
    # Use the algorithm on p. 371, Chapter 39 in the Trefethen book
    M = z*sparse.identity(np.shape(Pmat)[0]) - Pmat
    sv = svdvals(M.todense())
    sigmin[j,i] = np.min(sv)
    circs[j,i]  = np.sqrt(lambda_real[i]**2 + lambda_imag[j]**2)
    if np.min(sv) > abs(z):
      print("You were wrong!!!")
      print("sv-min: %5.3e" % np.min(sv))
      print("abs(z): %5.3e" % abs(z))
          
rcParams['figure.figsize'] = 2.5, 2.5
fs = 8
fig, ax  = plt.subplots()
X, Y = np.meshgrid(lambda_real, lambda_imag)
lvls = np.linspace(0.0, 1.0, 6)
cset = ax.contour(X, Y, sigmin, levels=lvls, colors='k')
ax.clabel(cset, lvls, fontsize=fs, inline=True)
plt.xlabel(r'Real part', fontsize=fs)
plt.ylabel(r'Imaginary part', fontsize=fs)
plt.title(r'$1/|| (z - E)^{-1} \||_2$')
ax.plot(0.0, 0.0, 'k+', markersize=fs)
if figure==7:
  filename = 'figure_7.pdf'
elif figure==8:
  filename = 'figure_8.pdf'
else:
  quit()
plt.gcf().savefig(filename, bbox_inches='tight')
call(["pdfcrop", filename, filename])
plt.show()
