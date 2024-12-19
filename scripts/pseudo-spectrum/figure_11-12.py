import sys
sys.path.append('../../src')

from parareal import parareal
from integrator_dedalus import integrator_dedalus
from solution_dedalus import solution_dedalus
import numpy as np
import scipy.sparse as sparse
from scipy.linalg import svdvals

import matplotlib.pyplot as plt
from subprocess import call
from pylab import rcParams
from parameter import parameter

from pseudo_spectral_radius import pseudo_spectral_radius


par = parameter(dedalus = True)
Tend, nslices, maxiter, nfine, ncoarse, tol, epsilon, ndof_f = par.getpar()

# for 24 DoF, Parareal diverges, for 30 DoF, you get convergence. Of course, speedup would be impossible here.
try:
  figure      =  int(sys.argv[1]) 
except:
  print("No or wrong command line argument provided, creating figure 11. Use 11 or 12 as command line argument.")
  figure = 11

if figure==11:
  ndof_c   = 24
  filename = 'figure_11.pdf'
elif figure==12:
  ndof_c   = 30
  filename = 'figure_12.pdf'
else:
  sys.exit("Figure needs to be 11 or 12")

u0fine = solution_dedalus(np.zeros(ndof_f), ndof_f)
u0coarse = solution_dedalus(np.zeros(ndof_c), ndof_c)
para     = parareal(0.0, Tend, nslices, integrator_dedalus, integrator_dedalus, nfine, ncoarse, tol, maxiter, u0fine, u0coarse)
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
plt.gcf().savefig(filename, bbox_inches='tight')
call(["pdfcrop", filename, filename])
plt.show()
