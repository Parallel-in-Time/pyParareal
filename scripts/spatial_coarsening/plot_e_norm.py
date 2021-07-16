import sys
sys.path.append('../../src')
import numpy as np
from numpy import linalg as LA
import scipy.sparse as sp
import scipy.linalg as spla

from parareal import parareal
from impeuler import impeuler
from intexact import intexact
from trapezoidal import trapezoidal
from solution_linear import solution_linear

from pylab import rcParams
import matplotlib.pyplot as plt

ndof_f = 16
ndof_c = 15

xaxis_f = np.linspace(0.0, 1.0, ndof_f+1)[0:ndof_f]
xaxis_c = np.linspace(0.0, 1.0, ndof_c+1)[0:ndof_c]
dx_f = xaxis_f[1] - xaxis_f[0]
dx_c = xaxis_c[1] - xaxis_c[0]

u0_f   = np.ones(ndof_f)
u0_c   = np.ones(ndof_c)
col    = np.zeros(ndof_f)
# First order upwind
col[0] = 1.0
col[1] = -1.0
A_f    = (1.0/dx_f)*spla.circulant(col)
A_c    = (1.0/dx_c)*spla.circulant(col[0:ndof_c])
# Second order centered
#col[1]  = -1.0
#col[-1] = 1.0
#A_f = (1.0/(2.0*dx_f))*spla.circulant(col)

u0fine   = solution_linear(u0_f, A_f)
u0coarse = solution_linear(u0_c, A_c)

Tend    = 1.0
nslices = 8
tol     = 0.0
maxiter = 4
nfine   = 1
ncoarse = 1



para     = parareal(0.0, Tend, nslices, trapezoidal, impeuler, nfine, ncoarse, tol, maxiter, u0fine, u0coarse)
E, Mginv = para.get_parareal_matrix()
F        = para.timemesh.slices[0].get_fine_update_matrix(u0fine)
G        = para.timemesh.slices[0].get_coarse_update_matrix(u0coarse)
rank_F   = LA.matrix_rank(F.todense())
if not rank_F==ndof_f:
  print("Rank of F is not equal to ndof_f")
rank_C = LA.matrix_rank(G)
if not rank_C==ndof_c:
  print("Rank of G is not equal to ndof_c")
rank_E = LA.matrix_rank(E.todense())
if not rank_E==nslices*ndof_f:
  print("Rank of E is not equal to nslices x ndof_f: rank(E) = %3i" % rank_E)
  
Enorm_2   = np.linalg.norm(E.todense(), 2)
Enorm_1   = np.linalg.norm(E.todense(), 1)
Enorm_inf = np.linalg.norm(E.todense(), np.inf)
print("Inf-Norm(E): %5.3e" % Enorm_inf)

# Compute SVD of F
u_f, s_f, vh_f = LA.svd(F.todense())
w, v           = LA.eig(F.todense())
#print("2-Norm of E:   %5.3e" % np.linalg.norm(E.todense(), 2))
#print("2-Norm of F-G: %5.3e" % np.linalg.norm(F-G, 2))
#print("m+1 SV of F:   %5.3e" % s_f[ndof_c])
#print("min SV of F:   %5.3e" % s_f[-1])

fig = plt.figure()
plt.semilogy(np.sort(s_f), 'bo', label='SV')
plt.plot(np.sort(np.abs(w)), 'rx', label='EV')
plt.plot(np.zeros(ndof_f) + Enorm_2, 'k--', label='Norm(E,2)')
#plt.ylim([0.9, 1.1])
plt.legend()
plt.show()

## The bound || E ||_2 >= || F - G ||_2 is reasonably sharp for small number of time slices.
## As the number of time slices increases, the norm of E becomes much larger than || F - G ||_2
## This is probably related to the B^k = G^k (F - G) matrices.
