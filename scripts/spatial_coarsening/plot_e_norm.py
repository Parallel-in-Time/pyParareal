import sys
sys.path.append('../../src')
import numpy as np
from numpy import linalg as LA

from parareal import parareal
from impeuler import impeuler
from intexact import intexact
from trapezoidal import trapezoidal
from solution_linear import solution_linear

from pylab import rcParams
import matplotlib.pyplot as plt

ndof_f = 16
ndof_c = 15
u0_f   = np.ones(ndof_f)
u0_c   = np.ones(ndof_c)
A_f    = np.eye(ndof_f)
A_c    = np.eye(ndof_c)

u0fine   = solution_linear(u0_f, A_f)
u0coarse = solution_linear(u0_c, A_c)

Tend    = 1.0
nslices = 4
tol     = 0.0
maxiter = 4
nfine   = 32
ncoarse = 32

para = parareal(0.0, Tend, nslices, trapezoidal, impeuler, nfine, ncoarse, tol, maxiter, u0fine, u0coarse)
E, Mginv = para.get_parareal_matrix()
F = para.timemesh.get_fine_matrix(u0fine)
G = para.timemesh.get_coarse_matrix(u0coarse)

print(np.linalg.norm(E.todense(), 2))

## DEFINE G BY MATRIX: compute low rank approximation for F and use this as coarse propagator ... what is the resulting E ? 
u, s, vh = LA.svd(E.todense())
print(np.max(np.abs(s)))

u_f, s_f, vh_f = LA.svd((F-G).todense())
print(np.linalg.norm((F-G).todense()))
