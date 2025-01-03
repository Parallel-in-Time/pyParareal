import sys
sys.path.append('../../src')

from parareal import parareal
from impeuler import impeuler
from intexact import intexact
from trapezoidal import trapezoidal
from special_integrator import special_integrator
from solution_linear import solution_linear
from integrator_dedalus import integrator_dedalus
from solution_dedalus import solution_dedalus
from get_matrix import get_upwind, get_centered, get_diffusion
import numpy as np
import scipy.sparse as sparse
from scipy.linalg import svdvals
import math
from parameter import parameter

import matplotlib.pyplot as plt
from subprocess import call
from pylab import rcParams

from pseudo_spectral_radius import pseudo_spectral_radius

try:
  figure      =  int(sys.argv[1]) # 1 generates figure_1, 2 generates figure_2
except:
  print("No or wrong command line argument provided, creating figure 9. Use 9, 10, 11 or 12 as command line argument.")
  figure = 5
assert 9<= figure <= 12, "Figure should be 9, 10, 11 or 12"
  
if figure==9 or figure==10:
  par = parameter(dedalus = False)
  ndof_c   = 24
elif figure==11: 
  par = parameter(dedalus = True)
  ndof_c   = 24
elif figure==12:
  par = parameter(dedalus = True)
  ndof_c   = 30
else:
  sys.exit("This should have been caught above")
  
Tend, nslices, maxiter, nfine, ncoarse, tol, epsilon, ndof_f = par.getpar()

if figure==9:
  xaxis_f = np.linspace(0.0, 2.0, ndof_f+1)[0:ndof_f]
  dx_f    = xaxis_f[1] - xaxis_f[0]
  xaxis_c = np.linspace(0.0, 2.0, ndof_c+1)[0:ndof_c]
  dx_c = xaxis_c[1] - xaxis_c[0]  
  A_f = get_upwind(ndof_f, dx_f)
  A_c = get_upwind(ndof_c, dx_c)
  u0fine   = solution_linear(np.zeros(ndof_f), A_f)
  u0coarse = solution_linear(np.zeros(ndof_c), A_c)  
  para     = parareal(0.0, Tend, nslices, impeuler, impeuler, nfine, ncoarse, tol, maxiter, u0fine, u0coarse)  
  filename = 'figure_9.pdf'
  D = A_f*A_f.H - A_f.H*A_f
  print("Normality number of the system matrix (this should be zero): %5.3f" % np.linalg.norm(D.todense()))  
elif figure==10:
  xaxis_f = np.linspace(0.0, 2.0, ndof_f+1)[0:ndof_f]
  dx_f    = xaxis_f[1] - xaxis_f[0]
  xaxis_c = np.linspace(0.0, 2.0, ndof_c+1)[0:ndof_c]
  dx_c = xaxis_c[1] - xaxis_c[0]  
  A_f = get_centered(ndof_f, dx_f)
  A_c = get_centered(ndof_c, dx_c)
  u0fine   = solution_linear(np.zeros(ndof_f), A_f)
  u0coarse = solution_linear(np.zeros(ndof_c), A_c)  
  para     = parareal(0.0, Tend, nslices, trapezoidal, trapezoidal, nfine, ncoarse, tol, maxiter, u0fine, u0coarse)  
  filename = 'figure_10.pdf'
  D = A_f*A_f.H - A_f.H*A_f
  print("Normality number of the system matrix (this should be zero): %5.3f" % np.linalg.norm(D.todense()))
elif figure==11 or figure==12:
  u0fine     = solution_dedalus(np.zeros(ndof_f), ndof_f)
  u0coarse   = solution_dedalus(np.zeros(ndof_c), ndof_c)
  para       = parareal(0.0, Tend, nslices, integrator_dedalus, integrator_dedalus, nfine, ncoarse, tol, maxiter, u0fine, u0coarse)
  if figure==11:
   filename = 'figure_11.pdf'
  elif figure==12:
   filename = 'figure_12.pdf'        
else:
  sys.exit("Wrong value for figure")
  

Pmat, Bmat = para.get_parareal_matrix()
print("|| E ||_2 = %5.2f" % np.linalg.norm(Pmat.todense(), 2))
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
for i in range(0,nreal):
  for j in range(0,nimag):
    z = lambda_real[i] + 1j*lambda_imag[j]
    # Use the algorithm on p. 371, Chapter 39 in the Trefethen book
    M = z*sparse.identity(np.shape(Pmat)[0]) - Pmat
    sv = svdvals(M.todense())
    sigmin[j,i] = np.min(sv)
    circs[j,i]  = np.sqrt(lambda_real[i]**2 + lambda_imag[j]**2)
    if np.min(sv) > abs(z):
      print("This should not happen")
      print("sv-min: %5.3e" % np.min(sv))
      print("abs(z): %5.3e" % abs(z))
          
rcParams['figure.figsize'] = 2.5, 2.5
fs = 8
fig, ax  = plt.subplots()
X, Y = np.meshgrid(lambda_real, lambda_imag)
lvls = [0.3, 0.6, 1.0]
cset = ax.contour(X, Y, sigmin, levels=lvls, colors='k')
ax.clabel(cset, lvls, fontsize=fs, inline=True)
ax.contour(X, Y, X**2 + Y**2, levels=[1.0], linestyles='--', colors='k')
plt.xlabel(r'Real part', fontsize=fs)
plt.ylabel(r'Imaginary part', fontsize=fs)
#plt.title(r'$1/|| (z - E)^{-1} \||_2$')
ax.plot(0.0, 0.0, 'k+', markersize=fs)
plt.gcf().savefig(filename, bbox_inches='tight')
call(["pdfcrop", filename, filename])
plt.show()
