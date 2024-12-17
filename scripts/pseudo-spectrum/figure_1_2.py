import sys
sys.path.append('../../src')
import numpy as np
from numpy import linalg as LA
import scipy.sparse as sp
import scipy.linalg as spla
import scipy.sparse.linalg as linalg

from parareal import parareal
from impeuler import impeuler
from intexact import intexact
from trapezoidal import trapezoidal
from solution_linear import solution_linear
from get_matrix import get_upwind, get_centered
from pseudo_spectral_radius import pseudo_spectral_radius

from pylab import rcParams
import matplotlib.pyplot as plt
from subprocess import call
  
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
  figure      =  int(sys.argv[1]) # 1 generates figure_1, 2 generates figure_2
except:
  print("No or wrong command line argument provided, creating figure 1. Use 1 or 2 as command line argument.")
  figure = 1

if figure==1:
  A_f = get_upwind(ndof_f, dx_f)
  A_c = get_upwind(ndof_c, dx_c)
  
elif figure==2:
  A_f = get_centered(ndof_f, dx_f)
  A_c = get_centered(ndof_c, dx_c)
 
else:
  sys.exit("Figure should be set to 1 or 2")
  
D = A_f*A_f.H - A_f.H*A_f
print("Normality number of the system matrix (this should be zero): %5.3f" % np.linalg.norm(D.todense()))
u0fine     = solution_linear(np.zeros(ndof_f), A_f)
defect_inf = np.zeros((1,maxiter))
defect_l2  = np.zeros((1,maxiter))
slopes     = np.zeros(1)
psr        = np.zeros(1)
  
u0coarse = solution_linear(np.zeros(ndof_c), A_c)

if figure==1:
  para     = parareal(0.0, Tend, nslices, impeuler, impeuler, nfine, ncoarse, tol, maxiter, u0fine, u0coarse)
else:
  para     = parareal(0.0, Tend, nslices, trapezoidal, trapezoidal, nfine, ncoarse, tol, maxiter, u0fine, u0coarse)
Pmat, Bmat = para.get_parareal_matrix()

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
plt.semilogy(range(1,5), [E_norm**(val-1)*1.1*defect_l2[0,0] for val in range(1,5)], 'b--', label=r'$|| E ||_2^k$')
plt.semilogy(range(1,5), [psr**(val-1)*1.1*defect_l2[0,0] for val in range(1,5)], 'r-.', label=r'$\sigma_{\epsilon}(E)^k$')
plt.legend(loc='upper right', fontsize=fs, prop={'size':fs-2}, handlelength=3)
plt.xlim([1, maxiter+1])
plt.ylim([1e-5, 1e3])

plt.xlabel('Iteration $k$', fontsize=fs)

#plt.ylim([1e-15, 1e1])
plt.xlim([1, maxiter+1])
plt.xticks(range(2,maxiter,2))
if figure==1:
  filename = 'figure_1.pdf'
elif figure==2:
  filename = 'figure_2.pdf'
else:
  quit()
plt.gcf().savefig(filename, bbox_inches='tight')
call(["pdfcrop", filename, filename])
plt.show()
