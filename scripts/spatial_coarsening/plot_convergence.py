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

ndof_f = 64
ndof_c = 32

xaxis_f = np.linspace(0.0, 1.0, ndof_f+1)[0:ndof_f]
xaxis_c = np.linspace(0.0, 1.0, ndof_c+1)[0:ndof_c]
dx_f = xaxis_f[1] - xaxis_f[0]
dx_c = xaxis_c[1] - xaxis_c[0]

u0_f   = np.exp(-(xaxis_f - 0.5)**2/0.1**2)
u0_c   = np.exp(-(xaxis_c - 0.5)**2/0.1**2)
col    = np.zeros(ndof_f)
if True:
  # First order upwind
  col[0] = 1.0
  col[1] = -1.0
  A_f    = -(1.0/dx_f)*spla.circulant(col)
  A_c    = -(1.0/dx_c)*spla.circulant(col[0:ndof_c])
# Second order centered
else:
  col[1]  = -1.0
  col[-1] = 1.0
  A_f = (1.0/(2.0*dx_f))*spla.circulant(col)
  col = np.zeros(ndof_c)
  col[1] = -1.0
  col[-1] = 1.0
  A_c = (1.0/(2.0*dx_c))*spla.circulant(col)

u0fine   = solution_linear(u0_f, A_f)
u0coarse = solution_linear(u0_c, A_c)

Tend    = 0.2
nslices = 8
tol     = 0.0
maxiter = 7
nfine   = 2
ncoarse = 1

para     = parareal(0.0, Tend, nslices, impeuler, impeuler, nfine, ncoarse, tol, maxiter, u0fine, u0coarse)
Pmat, Bmat = para.get_parareal_matrix()
### Parareal iteration: y^k+1 = Pmat*y^k + Bmat*b

print("2-norm of E: %5.3e" % np.linalg.norm(Pmat.todense(),2))

Fmat = para.timemesh.get_fine_matrix(u0fine)
### Fine propagator: Fmat*y = b
b = np.zeros((ndof_f*(nslices+1),1))
b[0:ndof_f,:] = u0fine.y

# compute fine solution
u = linalg.spsolve(Fmat, b)
u = u.reshape((ndof_f*(nslices+1),1))

defect = np.zeros((1,maxiter+1))
# Now do Parareal iteration
u_para_old = Bmat@b
defect[0] = np.linalg.norm(u_para_old - u, np.inf)
for k in range(maxiter):
  u_para_new = Pmat@u_para_old + Bmat@b
  defect[0,k+1] = np.linalg.norm(u_para_new - u, np.inf)
  u_para_old = np.copy(u_para_new)
  
fig = plt.figure(1)
plt.semilogy(range(maxiter+1), defect[0,:], 'bo-')
plt.show()
