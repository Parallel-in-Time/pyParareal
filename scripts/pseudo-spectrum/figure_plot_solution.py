import sys
sys.path.append('../../src')
import numpy as np
from numpy import linalg as LA
import scipy.sparse as sp
import scipy.linalg as spla
import scipy.sparse.linalg as linalg

from parareal import parareal
from integrator_dedalus import integrator_dedalus

from solution_dedalus import solution_dedalus
from solution_linear import solution_linear

from get_matrix import get_upwind, get_centered, get_diffusion
from impeuler import impeuler
from intexact import intexact
from trapezoidal import trapezoidal

from pseudo_spectral_radius import pseudo_spectral_radius
from parameter import parameter

from pylab import rcParams
import matplotlib.pyplot as plt
from subprocess import call
  
try:
  figure      =  int(sys.argv[1]) # 1 generates figure_1, 2 generates figure_2
except:
  print("No or wrong command line argument provided, creating figure 13. Use 13, 14, 15 or 16 as command line argument.")
  figure = 5
assert -4<= figure <= -1, "Figure should be -4, -3, -2 or -1"
  
if figure==-4 or figure==-3:
  par = parameter(dedalus = False)
  ndof_c   = 24
elif figure==-2: 
  par = parameter(dedalus = True)
  ndof_c   = 24
elif figure==-1:
  par = parameter(dedalus = True)
  ndof_c   = 30
else:
  sys.exit("This should have been caught above")
  
Tend, nslices, maxiter, nfine, ncoarse, tol, epsilon, ndof_f = par.getpar()

if figure==-4:
  xaxis_f = np.linspace(0.0, 1.0, ndof_f+1)[0:ndof_f]
  dx_f    = xaxis_f[1] - xaxis_f[0]
  xaxis_c = np.linspace(0.0, 1.0, ndof_c+1)[0:ndof_c]
  dx_c = xaxis_c[1] - xaxis_c[0]  
  A_f = get_upwind(ndof_f, dx_f)
  A_c = get_upwind(ndof_c, dx_c)
  u0fine   = solution_linear(np.zeros(ndof_f), A_f)
  u0coarse = solution_linear(np.zeros(ndof_c), A_c)  
  para     = parareal(0.0, Tend, nslices, impeuler, impeuler, nfine, ncoarse, tol, maxiter, u0fine, u0coarse)  
  filename = 'figure_minus4.pdf'
  mylabel = 'Conf. A'
  D = A_f*A_f.H - A_f.H*A_f
  print("Normality number of the system matrix (this should be zero): %5.3f" % np.linalg.norm(D.todense()))  
elif figure==-3:
  xaxis_f = np.linspace(0.0, 1.0, ndof_f+1)[0:ndof_f]
  dx_f    = xaxis_f[1] - xaxis_f[0]
  xaxis_c = np.linspace(0.0, 1.0, ndof_c+1)[0:ndof_c]
  dx_c = xaxis_c[1] - xaxis_c[0]  
  A_f = get_centered(ndof_f, dx_f)
  A_c = get_centered(ndof_c, dx_c)
  u0fine   = solution_linear(np.zeros(ndof_f), A_f)
  u0coarse = solution_linear(np.zeros(ndof_c), A_c)  
  para     = parareal(0.0, Tend, nslices, trapezoidal, trapezoidal, nfine, ncoarse, tol, maxiter, u0fine, u0coarse)  
  filename = 'figure_minus3.pdf'
  mylabel = 'Conf. B'  
  D = A_f*A_f.H - A_f.H*A_f
  print("Normality number of the system matrix (this should be zero): %5.3f" % np.linalg.norm(D.todense()))
elif figure==-2 or figure==-1:
  xaxis_f = np.linspace(0.0, 1.0, ndof_f+1)[0:ndof_f]
  dx_f    = xaxis_f[1] - xaxis_f[0]    
  u0fine     = solution_dedalus(np.zeros(ndof_f), ndof_f)
  u0coarse   = solution_dedalus(np.zeros(ndof_c), ndof_c)
  para       = parareal(0.0, Tend, nslices, integrator_dedalus, integrator_dedalus, nfine, ncoarse, tol, maxiter, u0fine, u0coarse)
  if figure==-2:
   filename = 'figure_minus2.pdf'
   mylabel = 'Conf. C'     
  elif figure==-1:
    filename = 'figure_minus1.pdf'
    mylabel = 'Conf. D'         
else:
  sys.exit("Wrong value for figure")  
  
Pmat, Bmat = para.get_parareal_matrix()
Fmat = para.timemesh.get_fine_matrix(u0fine)
bvec = np.zeros((ndof_f*(nslices+1),1))
bvec[0:ndof_f,:] = np.reshape(np.sin(2.0*np.pi*xaxis_f), (ndof_f, 1))
u_fine = linalg.spsolve(Fmat, bvec)
y = Bmat@bvec
for k in range(maxiter):
    y = Pmat@y + Bmat@bvec
  
yend = y[ndof_f*nslices:]

rcParams['figure.figsize'] = 2.5, 2.5
fs = 8
ms = 4
fig = plt.figure(1)
plt.plot(xaxis_f, yend, 'b', label=mylabel)
plt.plot(xaxis_f,  np.sin(2.0*np.pi*(xaxis_f-Tend)), 'r--', label='Exact')
plt.legend()
plt.xlabel('x')
plt.ylabel('u')
plt.gcf().savefig(filename, bbox_inches='tight')
call(["pdfcrop", filename, filename])
plt.show()
