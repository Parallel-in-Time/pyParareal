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
  k = 24
#  return np.exp(-(x-1.0-t)**2/0.25**2)
#  return np.sin(np.pi*(x-t)) + np.sin(k*np.pi*(x-t))
  return np.exp(-np.pi**2*t)*np.sin(np.pi*x) + np.exp(-np.pi**2*k**2*t)*np.sin(k*np.pi*x)
#  y = 0.0*x
#  for n in range(32):
#    y = y + np.sin(float(n)*np.pi*(x-t))
#  return y
  
Tend    = 1.0
nslices = 10
tol     = 0.0
maxiter = 9
nfine   = 10
ncoarse = 10

ndof_f   = 64
ndof_c_v = [32, 48, 63, 64]

#ndof_f   = 256
#ndof_c_v = [128, 196, 255, 256]

xaxis_f = np.linspace(0.0, 2.0, ndof_f+1)[0:ndof_f]
dx_f    = xaxis_f[1] - xaxis_f[0]
u0_f    = uex(xaxis_f, 0.0)
col     = np.zeros(ndof_f)
# 1 = advection with implicit Euler / upwind FD
# 2 = advection with trapezoidal rule / centered FD
# 3 = diffusion with trapezoidal rule / centered second order FD
problem      = 3
matrix_power = False # if False, do an actual Parareal iteration, if True, compute || E^k ||

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
u0fine     = solution_linear(u0_f, A_f)
defect_inf = np.zeros((4,maxiter))
defect_l2  = np.zeros((4,maxiter))
slopes     = np.zeros(4)

for nn in range(4):

  ndof_c = ndof_c_v[nn]
  xaxis_c = np.linspace(0.0, 2.0, ndof_c+1)[0:ndof_c]
  dx_c = xaxis_c[1] - xaxis_c[0]
  u0_c   = uex(xaxis_c, 0.0)
  col    = np.zeros(ndof_c)
  
  if problem==1:
    A_c = get_upwind(ndof_c, dx_c)
  elif problem==2:
    A_c = get_centered(ndof_c, dx_c)
  elif problem==3:
    A_c = get_diffusion(ndof_c, dx_c)
  else:
    quit()
    
  u0coarse = solution_linear(u0_c, A_c)

  if problem==1:
    para     = parareal(0.0, Tend, nslices, impeuler, impeuler, nfine, ncoarse, tol, maxiter, u0fine, u0coarse)
  else:
    para     = parareal(0.0, Tend, nslices, trapezoidal, trapezoidal, nfine, ncoarse, tol, maxiter, u0fine, u0coarse)
  Pmat, Bmat = para.get_parareal_matrix()
  ### Parareal iteration: y^k+1 = Pmat*y^k + Bmat*b

  slopes[nn] = np.linalg.norm(Pmat.todense(),2)
  
  Fmat = para.timemesh.get_fine_matrix(u0fine)
  
  ### Fine propagator: Fmat*y = b
  b = np.zeros((ndof_f*(nslices+1),1))
  b[0:ndof_f,:] = u0fine.y
  
  # compute fine solution
  u = linalg.spsolve(Fmat, b)
  u = u.reshape((ndof_f*(nslices+1),1))

  # Now do Parareal iteration
  u_para_old = Bmat@b
  #u_para_old = np.zeros((ndof_f*(nslices+1),1))
  for k in range(maxiter):
    
    # Compute actual Parareal iteration
    if not matrix_power:
      u_para_new = Pmat@u_para_old + Bmat@b
      defect_l2[nn,k] = np.linalg.norm(u_para_new - u, 2)
      u_para_old      = np.copy(u_para_new)
      
    # Compute norm of powers of E
    else:
      P_power_k        = LA.matrix_power(Pmat.todense(), k+1)
      defect_l2[nn,k]  = np.linalg.norm(P_power_k , 2)
  
  
fig = plt.figure(5)
plt.plot(xaxis_f, uex(xaxis_f, Tend), 'b--')
plt.plot(xaxis_f, u[-ndof_f:,0], 'r-')
err = np.linalg.norm(u[-ndof_f:,0] - uex(xaxis_f, Tend), 2)
print("Fine Method Discretisation error: %5.3e" % err)

rcParams['figure.figsize'] = 2.5, 2.5
fs = 8
ms = 4
fig = plt.figure(1)
plt.semilogy(range(1,maxiter+1), defect_l2[0,:], 'bo-', label='m='+str(ndof_c_v[0]), markersize=ms)
plt.semilogy(range(1,maxiter+1), defect_l2[1,:], 'rx-', label='m='+str(ndof_c_v[1]), markersize=ms)
plt.semilogy(range(1,maxiter+1), defect_l2[2,:], 'cd-', label='m='+str(ndof_c_v[2]), markersize=ms)
if matrix_power:
  plt.semilogy(range(1,maxiter+1), np.zeros(maxiter)+1e-16, 'k+-', label='m='+str(ndof_c_v[3]), markersize=ms)
else:
  plt.semilogy(range(1,maxiter+1), np.zeros(maxiter)+defect_l2[3,0], 'k+-', label='m='+str(ndof_c_v[3]), markersize=ms)

#plt.semilogy(range(1,maxiter+1), np.zeros(maxiter) + err, 'k--')
plt.legend(loc='best', bbox_to_anchor=(0.5, 0.5), fontsize=fs, prop={'size':fs-2}, handlelength=3)
#plt.legend(loc='upper right', fontsize=fs, prop={'size':fs-2}, handlelength=3)

plt.semilogy(range(1,5), [slopes[0]**(val-1)*1.1*defect_l2[0,0] for val in range(1,5)], 'b--')
plt.semilogy(range(1,5), [slopes[1]**(val-1)*1.1*defect_l2[1,0] for val in range(1,5)], 'r--')
plt.semilogy(range(1,5), [slopes[2]**(val-1)*1.1*defect_l2[2,0] for val in range(1,5)], 'c--')

#print([(slopes[0]/defect_l2[0,0])**val for val in range(1,3)])

plt.xlabel('$k$', fontsize=fs)
if not matrix_power:
  plt.ylabel('$||\mathbf{e}^k ||_2$', fontsize=fs)
else:
  plt.ylabel('$||\mathbf{E}^k ||_2$', fontsize=fs)

#plt.ylim([1e-15, 1e1])
plt.xlim([1, maxiter])
plt.xticks(range(1,maxiter,2))
if problem==1:
  if not matrix_power:
    filename='parareal-coarsening-advection-upwind-convergence.pdf'
  else:
    filename='parareal-coarsening-advection-upwind-matrix_power.pdf'

elif problem==2:
  if not matrix_power:
    filename='parareal-coarsening-advection-centered-convergence.pdf'
  else:
    filename='parareal-coarsening-advection-centered-matrix_power.pdf'

elif problem==3:
  if not matrix_power:
    filename='parareal-coarsening-heat-convergence.pdf'
  else:
    filename='parareal-coarsening-heat-matrix_power.pdf'

plt.gcf().savefig(filename, bbox_inches='tight')
call(["pdfcrop", filename, filename])
plt.show()

#fig = plt.figure(2)
#plt.plot(xaxis_f, u[-ndof_f:,0], 'r+')
#plt.plot(xaxis_f, uex(xaxis_f, Tend), 'b--')

#plt.show()
