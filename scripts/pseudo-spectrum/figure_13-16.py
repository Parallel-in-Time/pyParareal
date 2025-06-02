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

from get_matrix import get_upwind, get_centered, get_diffusion, get_desterck
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
assert 13<= figure <= 16 or figure==0 or figure==-1, "Figure should be 13, 14, 15 or 16"
  
if figure==13 or figure==14 or figure==0 or figure==-1:
  par = parameter(dedalus = False)
  ndof_c   = 24
elif figure==15: 
  par = parameter(dedalus = True)
  ndof_c   = 24
elif figure==16:
  par = parameter(dedalus = True)
  ndof_c   = 30
else:
  sys.exit("This should have been caught above")
  
Tend, nslices, maxiter, nfine, ncoarse, tol, epsilon, ndof_f = par.getpar()

if figure==13:
  xaxis_f = np.linspace(0.0, 1.0, ndof_f+1)[0:ndof_f]
  dx_f    = xaxis_f[1] - xaxis_f[0]
  xaxis_c = np.linspace(0.0, 1.0, ndof_c+1)[0:ndof_c]
  dx_c = xaxis_c[1] - xaxis_c[0]  
  A_f = get_upwind(ndof_f, dx_f)
  A_c = get_upwind(ndof_c, dx_c)
  u0fine   = solution_linear(np.zeros(ndof_f), A_f)
  u0coarse = solution_linear(np.zeros(ndof_c), A_c)  
  para     = parareal(0.0, Tend, nslices, impeuler, impeuler, nfine, ncoarse, tol, maxiter, u0fine, u0coarse)  
  filename = 'figure_13.pdf'
  D = A_f*A_f.H - A_f.H*A_f
  print("Normality number of the system matrix (this should be zero): %5.3f" % np.linalg.norm(D.todense()))  
elif figure==14:
  xaxis_f = np.linspace(0.0, 1.0, ndof_f+1)[0:ndof_f]
  dx_f    = xaxis_f[1] - xaxis_f[0]
  xaxis_c = np.linspace(0.0, 1.0, ndof_c+1)[0:ndof_c]
  dx_c = xaxis_c[1] - xaxis_c[0]  
  A_f = get_centered(ndof_f, dx_f)
  A_c = get_centered(ndof_c, dx_c)
  u0fine   = solution_linear(np.zeros(ndof_f), A_f)
  u0coarse = solution_linear(np.zeros(ndof_c), A_c)  
  para     = parareal(0.0, Tend, nslices, trapezoidal, trapezoidal, nfine, ncoarse, tol, maxiter, u0fine, u0coarse)  
  filename = 'figure_14.pdf'
  D = A_f*A_f.H - A_f.H*A_f
  print("Normality number of the system matrix (this should be zero): %5.3f" % np.linalg.norm(D.todense()))
elif figure==15 or figure==16:
  u0fine     = solution_dedalus(np.zeros(ndof_f), ndof_f)
  u0coarse   = solution_dedalus(np.zeros(ndof_c), ndof_c)
  para       = parareal(0.0, Tend, nslices, integrator_dedalus, integrator_dedalus, nfine, ncoarse, tol, maxiter, u0fine, u0coarse)
  if figure==15:
   filename = 'figure_15.pdf'
  elif figure==16:
   filename = 'figure_16.pdf'    
elif figure==0:
  xaxis_f = np.linspace(0.0, 1.0, ndof_f+1)[0:ndof_f]
  dx_f    = xaxis_f[1] - xaxis_f[0]
  xaxis_c = np.linspace(0.0, 1.0, ndof_c+1)[0:ndof_c]
  dx_c = xaxis_c[1] - xaxis_c[0]  
  A_f = get_diffusion(ndof_f, dx_f)
  A_c = get_diffusion(ndof_c, dx_c)
  u0fine   = solution_linear(np.zeros(ndof_f), A_f)
  u0coarse = solution_linear(np.zeros(ndof_c), A_c)  
  para     = parareal(0.0, Tend, nslices, impeuler, impeuler, nfine, ncoarse, tol, maxiter, u0fine, u0coarse)  
  filename = 'figure_00.pdf'    
  D = A_f*A_f.H - A_f.H*A_f
  print("Normality number of the system matrix (this should be zero): %5.3f" % np.linalg.norm(D.todense()))  
elif figure==-1:
  p = 5
  xaxis_f = np.linspace(0.0, 1.0, ndof_f+1)[0:ndof_f]
  dx_f    = xaxis_f[1] - xaxis_f[0]
  xaxis_c = np.linspace(0.0, 1.0, ndof_c+1)[0:ndof_c]
  dx_c = xaxis_c[1] - xaxis_c[0]  
  A_f = get_desterck(ndof_f, dx_f, p)
  A_c = get_desterck(ndof_c, dx_c, p)
  u0fine   = solution_linear(np.zeros(ndof_f), A_f)
  u0coarse = solution_linear(np.zeros(ndof_c), A_c)  
  para     = parareal(0.0, Tend, nslices, impeuler, impeuler, nfine, ncoarse, tol, maxiter, u0fine, u0coarse)  
  filename = ('figure_conv_desterck_%i.pdf' % p)  
else:
  sys.exit("Wrong value for figure")
  

Pmat, Bmat = para.get_parareal_matrix()
print("|| E ||_2 = %5.2f" % np.linalg.norm(Pmat.todense(),2))

defect_inf = np.zeros((1,maxiter))
defect_l2  = np.zeros((1,maxiter))
slopes     = np.zeros(1)
psr        = np.zeros(1)
  
### Parareal iteration: y^k+1 = Pmat*y^k + Bmat*b
E_norm = np.linalg.norm(Pmat.todense(),2)
  
Fmat = para.timemesh.get_fine_matrix(u0fine)
  
### Fine propagator: Fmat*y = b
b = np.zeros((ndof_f*(nslices+1),1))
b[0:ndof_f,:] = u0fine.y
  
# compute fine solution
u = linalg.spsolve(Fmat, b)
u = u.reshape((ndof_f*(nslices+1),1))
  
psr_pmat = pseudo_spectral_radius(Pmat, eps=epsilon)
psr, a, b, c = psr_pmat.get_psr(verbose=True)

# Now do Parareal iteration
for k in range(maxiter):
    P_power_k       = LA.matrix_power(Pmat.todense(), k+1)
    defect_l2[0,k]  = np.linalg.norm(P_power_k , 2)
  
print("Defect after maxiter iterations: %5.3e" % defect_l2[0,-1])  
  
rcParams['figure.figsize'] = 2.5, 2.5
fs = 8
ms = 4
fig = plt.figure(1)
plt.semilogy(range(1,maxiter+1), defect_l2[0,:], 'bo-', markersize=ms, label=r'$|| e^k ||$')
plt.semilogy(range(1,5), [E_norm**(val-1)*2.0*defect_l2[0,0] for val in range(1,5)], 'b-.', label=r'$|| E ||_2^k$', linewidth=2)
plt.semilogy(range(1,5), [psr**(val-1)*2.0*defect_l2[0,0] for val in range(1,5)], 'r--', label=r'$\rho_{\epsilon}(E)^k$', linewidth=2)
if figure==14 or figure==15:
    plt.legend(loc='lower right', fontsize=fs, prop={'size':fs-2}, handlelength=3)
else:
    plt.legend(loc='upper right', fontsize=fs, prop={'size':fs-2}, handlelength=3)
    
plt.xlim([1, maxiter+1])
plt.ylim([1e-5, 1e3])

plt.xlabel('Iteration $k$', fontsize=fs)

#plt.ylim([1e-15, 1e1])
plt.xlim([1, maxiter+1])
plt.xticks(range(2,maxiter,2))
plt.gcf().savefig(filename, bbox_inches='tight')
call(["pdfcrop", filename, filename])
plt.show()
