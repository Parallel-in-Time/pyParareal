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
from integrator_dedalus import integrator_dedalus
from trapezoidal import trapezoidal
from solution_linear import solution_linear
from solution_dedalus import solution_dedalus
from get_matrix import get_upwind, get_centered, get_diffusion

from pylab import rcParams
import matplotlib.pyplot as plt
from subprocess import call

  
Tend    = 1.0
nslices = 10
tol     = 0.0
maxiter = 9
nsteps  = [1, 2, 4, 8, 12, 16, 20]

ndof_f   = 32
ndof_c_v = [16, 24, 30]
xaxis_f = np.linspace(0.0, 1.0, ndof_f+1)[0:ndof_f]
dx_f    = xaxis_f[1] - xaxis_f[0]

# 1 = advection with implicit Euler / upwind FD
# 2 = advection with trapezoidal rule / centered FD
# 3 = Dedalus
problem = 3

if problem==1:
  A_f = get_upwind(ndof_f, dx_f)
  u0fine   = solution_linear(np.zeros(ndof_f), A_f)
elif problem==2:
  A_f = get_centered(ndof_f, dx_f)
  u0fine   = solution_linear(np.zeros(ndof_f), A_f)
elif problem==3:
  u0fine = solution_dedalus(np.zeros(ndof_f), ndof_f)
else:
  sys.exit("Problem can only have values 1, 2 or 3")    

norm_l2  = np.zeros((3,np.size(nsteps)))
norm_inf = np.zeros((3,np.size(nsteps)))
dt_f_v   = np.zeros((1,np.size(nsteps)))
dt_c_v   = np.zeros((1,np.size(nsteps)))

for nn in range(3):

  ndof_c  = ndof_c_v[nn]
  xaxis_c = np.linspace(0.0, 1.0, ndof_c+1)[0:ndof_c]
  dx_c    = xaxis_c[1] - xaxis_c[0]

  if problem==1:
    A_c = get_upwind(ndof_c, dx_c)
    u0coarse = solution_linear(np.zeros(ndof_c), A_c)
  elif problem==2:
    A_c = get_centered(ndof_c, dx_c)
    u0coarse = solution_linear(np.zeros(ndof_c), A_c)
  elif problem==3:
    u0coarse = solution_dedalus(np.zeros(ndof_c), ndof_c)
  else:
    sys.exit("Problem can only have values 1, 2 or 3")    
      
  for mm in range(np.size(nsteps)):
  
    if problem==1:
      para     = parareal(0.0, Tend, nslices, impeuler, impeuler, nsteps[mm], nsteps[mm], tol, maxiter, u0fine, u0coarse)
    elif problem==2:
      para     = parareal(0.0, Tend, nslices, trapezoidal, trapezoidal, nsteps[mm], nsteps[mm], tol, maxiter, u0fine, u0coarse)
    elif problem==3:
     para     = parareal(0.0, Tend, nslices, integrator_dedalus, integrator_dedalus, nsteps[mm], nsteps[mm], tol, maxiter, u0fine, u0coarse)    
    else:
      quit()
    Pmat, Bmat = para.get_parareal_matrix()
    dt_f_v[0,mm] = para.timemesh.slices[0].int_fine.dt
    dt_c_v[0,mm] = para.timemesh.slices[0].int_coarse.dt
        
    ### Parareal iteration: y^k+1 = Pmat*y^k + Bmat*b
    norm_l2[nn,mm] = np.linalg.norm(Pmat.todense(), 2)

rcParams['figure.figsize'] = 2.5, 2.5
fs = 8
ms = 4
fig = plt.figure(1)
plt.plot(dt_f_v[0,:], norm_l2[0,:], 'bo-', label='m='+str(ndof_c_v[0]), markersize=ms)
plt.plot(dt_f_v[0,:], norm_l2[1,:], 'rx-', label='m='+str(ndof_c_v[1]), markersize=ms)
plt.plot(dt_f_v[0,:], norm_l2[2,:], 'cd-', label='m='+str(ndof_c_v[2]), markersize=ms)
plt.plot(dt_f_v[0,:], 1.0 + 0.0*dt_f_v[0,:], 'k:')
plt.legend(loc='best', bbox_to_anchor=(0.5, 0.5), fontsize=fs, prop={'size':fs-2}, handlelength=3)
plt.xlabel(r'$\delta t = \Delta t$', fontsize=fs)
plt.ylabel(r'$|| \mathbf{E} ||_2$', fontsize=fs)
plt.xlim([0.0, dt_f_v[0,0]])
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