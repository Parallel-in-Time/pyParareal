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
from get_matrix import get_upwind, get_centered, get_diffusion

from pylab import rcParams
import matplotlib.pyplot as plt
from subprocess import call

def uex(x,t):
  return np.exp(-(x-1.0-t)**2/0.1**2)
  
Tend    = 0.4
nslices = 8
tol     = 0.0
maxiter = 7
nsteps  = [1, 2, 4, 6, 8, 10]

ndof_f   = 64
ndof_c_v = [32, 48, 63, 64]

xaxis_f = np.linspace(0.0, 2.0, ndof_f+1)[0:ndof_f]
dx_f = xaxis_f[1] - xaxis_f[0]
u0_f   = uex(xaxis_f, 0.0)
col    = np.zeros(ndof_f)
# 1 = advection with implicit Euler / upwind FD
# 2 = advection with trapezoidal rule / centered FD
# 3 = diffusion with trapezoidal rule / centered second order FD
problem = 3

if problem==1:
  A_f = get_upwind(ndof_f, dx_f)
elif problem==2:
  A_f = get_centered(ndof_f, dx_f)
elif problem==3:
  A_f = get_diffusion(ndof_f, dx_f)
else:
  quit()

D = A_f*A_f.H - A_f.H*A_f
print("Normality number: %5.3f" % np.linalg.norm(D.todense()))
u0fine   = solution_linear(u0_f, A_f)
norm_l2  = np.zeros((4,np.size(nsteps)))
norm_inf = np.zeros((4,np.size(nsteps)))
for nn in range(4):

  ndof_c  = ndof_c_v[nn]
  xaxis_c = np.linspace(0.0, 2.0, ndof_c+1)[0:ndof_c]
  dx_c    = xaxis_c[1] - xaxis_c[0]
  u0_c    = uex(xaxis_c, 0.0)
  col     = np.zeros(ndof_c)
  
  if problem==1:
    # First order upwind
    col[0] = 1.0
    col[1] = -1.0
    A_c    = -(1.0/dx_c)*spla.circulant(col[0:ndof_c])
    # non-periodic BC instead
    #A_c[0,-1] = 0.0
  # Second order centered
  elif problem==2:

    col = np.zeros(ndof_c)
    col[1] = -1.0
    col[-1] = 1.0
    A_c = -(1.0/(2.0*dx_c))*spla.circulant(col)
  elif problem==3:
    col = np.zeros(ndof_c)
    col[0] = -2.0
    col[1] = 1.0
    col[-1] = 1.0
    A_c = (1/dx_c**2)*spla.circulant(col)
  u0coarse = solution_linear(u0_c, A_c)
  
  for mm in range(np.size(nsteps)):
  
    if problem==1:
      para     = parareal(0.0, Tend, nslices, impeuler, impeuler, nsteps[mm], nsteps[mm], tol, maxiter, u0fine, u0coarse)
    else:
      para     = parareal(0.0, Tend, nslices, trapezoidal, trapezoidal, nsteps[mm], nsteps[mm], tol, maxiter, u0fine, u0coarse)
    Pmat, Bmat = para.get_parareal_matrix()
    ### Parareal iteration: y^k+1 = Pmat*y^k + Bmat*b
    norm_l2[nn,mm] = np.linalg.norm(Pmat.todense(), 2)
    norm_inf[nn,mm] = np.linalg.norm(Pmat.todense(), np.inf)
fig = plt.figure(1)
plt.plot(nsteps, norm_l2[0,:], 'bo-', label='m='+str(ndof_c_v[0]))
plt.plot(nsteps, norm_l2[1,:], 'rx-', label='m='+str(ndof_c_v[1]))
plt.plot(nsteps, norm_l2[2,:], 'cd-', label='m='+str(ndof_c_v[2]))
plt.plot(nsteps, norm_l2[3,:], 'k+-', label='m='+str(ndof_c_v[3]))
#plt.plot(nsteps, norm_inf[0,:], 'b+--')
#plt.plot(nsteps, norm_inf[1,:], 'r+--')
#plt.plot(nsteps, norm_inf[2,:], 'c+--')

plt.legend()
plt.xlabel('Number of coarse/fine steps per slice')
plt.ylabel(r'$|| E ||_2$')
#plt.ylim([1e-15, 1e1])
#plt.xlabel([0, maxiter])
if problem==1:
  filename='parareal-coarsening-advection-upwind-norm.pdf'
elif problem==2:
  filename='parareal-coarsening-advection-centered-norm.pdf'
elif problem==3:
  filename = 'parareal-coarsening-heat-norm.pdf'
plt.gcf().savefig(filename, bbox_inches='tight')
call(["pdfcrop", filename, filename])
plt.show()
