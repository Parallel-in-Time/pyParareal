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

xaxis_f = np.linspace(0.0, 2.0, ndof_f+1)[0:ndof_f]
dx_f = xaxis_f[1] - xaxis_f[0]
u0_f   = uex(xaxis_f, 0.0)
col    = np.zeros(ndof_f)
do_upwind = False

if do_upwind:
  col[0] = 1.0
  col[1] = -1.0
  A_f    = -(1.0/dx_f)*spla.circulant(col)
else:
  col[1]  = -1.0
  col[-1] = 1.0
  A_f = -(1.0/(2.0*dx_f))*spla.circulant(col)
A_f = sp.csc_matrix(A_f)
D = A_f*A_f.H - A_f.H*A_f
print("Normality number: %5.3f" % np.linalg.norm(D.todense()))
u0fine   = solution_linear(u0_f, A_f)
defect_inf = np.zeros((4,maxiter+1))
defect_l2  = np.zeros((4,maxiter+1))

for nn in range(4):

  ndof_c = ndof_c_v[nn]
  xaxis_c = np.linspace(0.0, 2.0, ndof_c+1)[0:ndof_c]
  dx_c = xaxis_c[1] - xaxis_c[0]
  u0_c   = uex(xaxis_c, 0.0)
  col    = np.zeros(ndof_c)
  
  if do_upwind:
    # First order upwind
    col[0] = 1.0
    col[1] = -1.0
    A_c    = -(1.0/dx_c)*spla.circulant(col[0:ndof_c])
  # Second order centered
  else:

    col = np.zeros(ndof_c)
    col[1] = -1.0
    col[-1] = 1.0
    A_c = -(1.0/(2.0*dx_c))*spla.circulant(col)

  u0coarse = solution_linear(u0_c, A_c)

  if do_upwind:
    para     = parareal(0.0, Tend, nslices, impeuler, impeuler, nfine, ncoarse, tol, maxiter, u0fine, u0coarse)
  else:
    para     = parareal(0.0, Tend, nslices, trapezoidal, trapezoidal, nfine, ncoarse, tol, maxiter, u0fine, u0coarse)
  Pmat, Bmat = para.get_parareal_matrix()
  ### Parareal iteration: y^k+1 = Pmat*y^k + Bmat*b

  print("2-norm of E: %5.3f" % np.linalg.norm(Pmat.todense(),2))

  Fmat = para.timemesh.get_fine_matrix(u0fine)
  ### Fine propagator: Fmat*y = b
  b = np.zeros((ndof_f*(nslices+1),1))
  b[0:ndof_f,:] = u0fine.y

  # compute fine solution
  u = linalg.spsolve(Fmat, b)
  u = u.reshape((ndof_f*(nslices+1),1))

  # Now do Parareal iteration
  u_para_old = Bmat@b
  defect_inf[nn,0] = np.linalg.norm(u_para_old - u, np.inf)
  defect_l2[nn,0]  = np.linalg.norm(u_para_old - u, 2)
  for k in range(maxiter):
    u_para_new = Pmat@u_para_old + Bmat@b
    defect_inf[nn,k+1] = np.linalg.norm(u_para_new - u, np.inf)
    defect_l2[nn,k+1]  = np.linalg.norm(u_para_old - u, 2)
    u_para_old = np.copy(u_para_new)
  
fig = plt.figure(1)
plt.semilogy(range(maxiter+1), defect_l2[0,:], 'bo-', label='m='+str(ndof_c_v[0]))
plt.semilogy(range(maxiter+1), defect_l2[1,:], 'rx-', label='m='+str(ndof_c_v[1]))
plt.semilogy(range(maxiter+1), defect_l2[2,:], 'cd-', label='m='+str(ndof_c_v[2]))
plt.semilogy(range(maxiter+1), np.zeros(maxiter+1)+defect_l2[3,0], 'k+-', label='m=n='+str(ndof_c_v[3]))
#plt.semilogy(range(maxiter+1), defect_l2[0,:],  'b--')
#plt.semilogy(range(maxiter+1), defect_l2[1,:],  'r--')
#plt.semilogy(range(maxiter+1), defect_l2[2,:],  'c--')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Error to fine solution')
#plt.ylim([1e-15, 1e1])
plt.xlabel([0, maxiter])
if do_upwind:
  filename='parareal-coarsening-advection-upwind.pdf'
else:
  filename='parareal-coarsening-advection-centered.pdf'
plt.gcf().savefig(filename, bbox_inches='tight')
call(["pdfcrop", filename, filename])


#fig = plt.figure(2)
#plt.plot(xaxis_f, u[-ndof_f:,0], 'r+')
#plt.plot(xaxis_f, uex(xaxis_f, Tend), 'b--')
err = u[-ndof_f:,0] - uex(xaxis_f, Tend)
print("Discretisation error: %5.3f" % np.linalg.norm(err, np.inf))
#plt.show()
