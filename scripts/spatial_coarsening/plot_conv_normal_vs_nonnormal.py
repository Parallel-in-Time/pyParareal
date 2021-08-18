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

from pylab import rcParams
import matplotlib.pyplot as plt
from subprocess import call

def uex(x,t):
  return np.exp(-(x-1.0-t)**2/0.1**2)
  
Tend    = 0.4
nslices = 8
tol     = 0.0
maxiter = 7
nfine   = 5
ncoarse = 5

ndof_f   = 64
ndof_c_v = [32, 48, 63, 64]
#ndof_f = 16
#ndof_c_v = [8, 12, 15, 16]

xaxis_f = np.linspace(0.0, 2.0, ndof_f+1)[0:ndof_f]
dx_f = xaxis_f[1] - xaxis_f[0]
u0_f   = uex(xaxis_f, 0.0)
col    = np.zeros(ndof_f)
# 1 = advection with implicit Euler / upwind FD
# 2 = advection with trapezoidal rule / centered FD
# 3 = diffusion with trapezoidal rule / centered second order FD
problem = 3

col[0] = 1.0
col[1] = -1.0
A_f_per    = -(1.0/dx_f)*spla.circulant(col)
  # non-periodic boundary conditions instead
A_f_np       = np.copy(A_f_per)
A_f_np[0,-1] = 0.0

  
A_f_per = sp.csc_matrix(A_f_per)
A_f_np  = sp.csc_matrix(A_f_np)
D = A_f_per*A_f_per.H - A_f_per.H*A_f_per
print("Normality number: %5.3f" % np.linalg.norm(D.todense()))
u0fine_per = solution_linear(u0_f, A_f_per)
u0fine_np  = solution_linear(u0_f, A_f_np)
defect_l2_per  = np.zeros((4,maxiter+1))
defect_l2_np   = np.zeros((4,maxiter+1))

for nn in range(4):

  ndof_c = ndof_c_v[nn]
  xaxis_c = np.linspace(0.0, 2.0, ndof_c+1)[0:ndof_c]
  dx_c = xaxis_c[1] - xaxis_c[0]
  u0_c   = uex(xaxis_c, 0.0)
  col    = np.zeros(ndof_c)
  
  # First order upwind
  col[0] = 1.0
  col[1] = -1.0
  A_c_per    = -(1.0/dx_c)*spla.circulant(col[0:ndof_c])
  # non-periodic BC instead
  A_c_np     = np.copy(A_c_per)
  A_c_np[0,-1] = 0.0

    
  u0coarse_per = solution_linear(u0_c, A_c_per)
  u0coarse_np  = solution_linear(u0_c, A_c_np)

  para_per     = parareal(0.0, Tend, nslices, impeuler, impeuler, nfine, ncoarse, tol, maxiter, u0fine_per, u0coarse_per)
  para_np      = parareal(0.0, Tend, nslices, impeuler, impeuler, nfine, ncoarse, tol, maxiter, u0fine_np, u0coarse_np)
  
  Pmat_per, Bmat_per = para_per.get_parareal_matrix()
  Pmat_np,  Bmat_np  = para_np.get_parareal_matrix()
  ### Parareal iteration: y^k+1 = Pmat*y^k + Bmat*b

  print("Normal matrix: 2-norm of E: %5.3f" % np.linalg.norm(Pmat_per.todense(),2))
  print("Non-Normal matrix: 2-norm of E: %5.3f" % np.linalg.norm(Pmat_np.todense(),2))

  Fmat_per = para_per.timemesh.get_fine_matrix(u0fine_per)
  Fmat_np = para_np.timemesh.get_fine_matrix(u0fine_np)
  ### Fine propagator: Fmat*y = b
  b_per = np.zeros((ndof_f*(nslices+1),1))
  b_per[0:ndof_f,:] = u0fine_per.y

  # compute fine solution
  u_per = linalg.spsolve(Fmat_per, b_per)
  u_per = u_per.reshape((ndof_f*(nslices+1),1))

  b_np  = np.zeros((ndof_f*(nslices+1),1))
  b_np[0:ndof_f,:] = u0fine_np.y
  
  #
  u_np = linalg.spsolve(Fmat_np, b_np)
  u_np = u_np.reshape((ndof_f*(nslices+1),1))
  
  # Now do Parareal iteration
  u_old_per = Bmat_per@b_per
  u_old_np  = Bmat_np@Bmat_np
  
  defect_l2_per[nn,0]  = np.linalg.norm(u_old_per - u_per, 2)
  defect_l2_np[nn,0]   = np.linalg.norm(u_old_np - u_np, 2)
  
  for k in range(maxiter):
    u_new_per = Pmat_per@u_old_per + Bmat_per@b_per
    u_new_np  = Pmat_np@u_old_np + Bmat_np@b_np
    defect_l2_per[nn,k+1]  = np.linalg.norm(u_new_per - u_per, 2)
    defect_l2_np[nn,k+1]   = np.linalg.norm(u_new_np - u_np, 2)
    u_old_per = np.copy(u_new_per)
    u_old_np  = np.copy(u_new_np)
    
fig = plt.figure(1)
plt.semilogy(range(maxiter+1), defect_l2_per[0,:], 'bo-', label='m='+str(ndof_c_v[0]))
plt.semilogy(range(maxiter+1), defect_l2_per[1,:], 'rx-', label='m='+str(ndof_c_v[1]))
plt.semilogy(range(maxiter+1), defect_l2_per[2,:], 'cd-', label='m='+str(ndof_c_v[2]))
plt.semilogy(range(maxiter+1), np.zeros(maxiter+1)+defect_l2_per[3,0], 'k+-', label='m=n='+str(ndof_c_v[3]))
plt.semilogy(range(maxiter+1), defect_l2_np[0,:],  'b--')
plt.semilogy(range(maxiter+1), defect_l2_np[1,:],  'r--')
plt.semilogy(range(maxiter+1), defect_l2_np[2,:],  'c--')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Error to fine solution')
plt.ylim([1e-16, 1e2])
plt.xlabel([0, maxiter])
plt.gca().text(0, 5e-3*defect_l2_per[2,0], 'Normal matrix', rotation=-20)
plt.gca().text(5, 110*defect_l2_np[2,5], 'Non-normal matrix', rotation=-15)
#plt.show()
filename='parareal-coarseing_normal_vs_nonnormal.pdf'
plt.gcf().savefig(filename, bbox_inches='tight')
call(["pdfcrop", filename, filename])
