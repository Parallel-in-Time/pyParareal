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

from pseudo_spectral_radius import pseudo_spectral_radius

from pylab import rcParams
import matplotlib.pyplot as plt
from subprocess import call
  
Tend    = 1.0
nslices = 10
tol     = 0.0
maxiter = 9
nfine   = 10
ncoarse = 10

ndof_f   = 32

# for 24 DoF, Parareal diverges, for 30 DoF, you get convergence. Of course, speedup would be impossible here.
try:
  figure      =  int(sys.argv[1]) # 3 generates figure_3, 4 generates figure_4
except:
  print("No or wrong command line argument provided, creating figure 3. Use 3 or 4 as command line argument.")
  figure = 3

if figure==3:
  ndof_c   = 24
elif figure==4:
  ndof_c   = 30   
else:
  sys.exit("Set figure to 1 or 2")
  
epsilon = 0.1

u0fine   = solution_dedalus(np.zeros(ndof_f), ndof_f)
u0coarse = solution_dedalus(np.zeros(ndof_c), ndof_c)
para     = parareal(0.0, Tend, nslices, integrator_dedalus, integrator_dedalus, nfine, ncoarse, tol, maxiter, u0fine, u0coarse)
Pmat, Bmat = para.get_parareal_matrix()

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
plt.semilogy(range(1,5), [E_norm**(val-1)*1.1*defect_l2[0,0] for val in range(1,5)], 'b--', label=r'$|| E ||_2^k$')
plt.semilogy(range(1,5), [psr**(val-1)*1.1*defect_l2[0,0] for val in range(1,5)], 'r-.', label=r'$\sigma_{\epsilon}(E)^k$')
if figure==3:
    plt.legend(loc='lower right', fontsize=fs, prop={'size':fs-2}, handlelength=3)
else:
    plt.legend(loc='upper right', fontsize=fs, prop={'size':fs-2}, handlelength=3)
    
plt.xlim([1, maxiter+1])
plt.ylim([1e-5, 1e3])

plt.xlabel('Iteration $k$', fontsize=fs)

#plt.ylim([1e-15, 1e1])
plt.xlim([1, maxiter+1])
plt.xticks(range(2,maxiter,2))
if figure==3:
    filename = 'figure_3.pdf'
elif figure==4:
    filename = 'figure_4.pdf'
else:
    quit()
plt.gcf().savefig(filename, bbox_inches='tight')
call(["pdfcrop", filename, filename])
plt.show()
