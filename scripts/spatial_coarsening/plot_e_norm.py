import sys
sys.path.append('../../src')
import numpy as np
from numpy import linalg as LA
import scipy.sparse as sp

from parareal import parareal
from impeuler import impeuler
from intexact import intexact
from trapezoidal import trapezoidal
from solution_linear import solution_linear

from pylab import rcParams
import matplotlib.pyplot as plt

ndof_f = 16
ndof_c = 12
u0_f   = np.ones(ndof_f)
u0_c   = np.ones(ndof_c)
A_f    = (1.0/(ndof_f-1))*sp.diags([-np.ones(ndof_f), np.ones(ndof_f)], [-1,0], shape=(ndof_f, ndof_f))
A_c    = np.eye(ndof_c)

u0fine   = solution_linear(u0_f, A_f)
u0coarse = solution_linear(u0_c, A_c)

Tend    = 1.0
nslices = 8
tol     = 0.0
maxiter = 4
nfine   = 32
ncoarse = 2

para = parareal(0.0, Tend, nslices, trapezoidal, impeuler, nfine, ncoarse, tol, maxiter, u0fine, u0coarse)
E, Mginv = para.get_parareal_matrix()
F = para.timemesh.slices[0].get_fine_update_matrix(u0fine)
G = para.timemesh.slices[0].get_coarse_update_matrix(u0coarse)

print(np.linalg.norm(E.todense(), 2))

## DEFINE G BY MATRIX: compute low rank approximation for F and use this as coarse propagator ... what is the resulting E ? 
u, s, vh = LA.svd(E.todense())
print(np.max(np.abs(s)))


# confirm that the fine propagator matrix has full rank
rank_F = LA.matrix_rank(F.todense())
print("rank F: %5.3i" % rank_F)

# number of coarse DoF has no bearing on rank of E, it seems
# rank should be ndof_f*nslices
print("rank E: %5.3i" % LA.matrix_rank(E.todense()))
print(np.shape(E.todense()))
print("ndof_f x nslices: %5.3i" % (ndof_f*nslices))

print("==========")
print("1-norm of E: %5.3e" % np.linalg.norm(E.todense(), 1))
print("inf-norm of E: %5.3e" % np.linalg.norm(E.todense(), np.inf))

print("==========")
u_f, s_f, vh_f = LA.svd(F.todense())
w, v = LA.eig(F.todense())
print("Inf-Norm of E: %5.3e" % np.linalg.norm(E.todense(), np.inf))
print("Inf-Norm of F-G: %5.3e" % np.linalg.norm(F-G, np.inf))
print("m+1 SV of F: %5.3e" % s_f[ndof_c])
print("min SV of F: %5.3e" % s_f[-1])

fig = plt.figure()
plt.plot(s_f, 'bo', label='SV')
plt.plot(w, 'rx', label='EV')
plt.ylim([0.9, 1.1])
plt.show()

# NOTE: numerically it seems that the SV of F are all >= 1

print("1-norm of F-G: %5.3e" % np.linalg.norm(F-G, 1))
print("2-norm of F-G: %5.3e" % np.linalg.norm(F-G, 2))
print("inf-norm of F-G: %5.3e" % np.linalg.norm(F-G, np.inf))
