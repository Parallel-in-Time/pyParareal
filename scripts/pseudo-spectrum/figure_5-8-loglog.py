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
from parameter import parameter

from pylab import rcParams
import matplotlib.pyplot as plt
from subprocess import call

def ie(z):
  return 1.0/(1.0 - z)

def trap(z):
  return (1.0 + 0.5*z)/(1.0 - 0.5*z)

try:
  figure      =  int(sys.argv[1]) # 1 generates figure_1, 2 generates figure_2
except:
  print("No or wrong command line argument provided, creating figure 5. Use 5, 6, 7 or 8 as command line argument.")
  figure = 5
assert 5<= figure <= 8, "Figure should be 5, 6, 7 or 8"
  
if figure==5 or figure==6:
  par = parameter(dedalus = False)
elif figure==7 or figure==8:
  par = parameter(dedalus = True)
else:
  sys.exit("This should have been caught above")

Tend, nslices, maxiter, nfine, ncoarse, tol, epsilon, ndof_f = par.getpar()

nsteps   = [1, 2, 4, 8, 12, 16, 20]
nsteps   = [1, 10, 50, 100, 1000, 10000]

ndof_c_v = [16, 24, 30]
xaxis_f  = np.linspace(0.0, 1.0, ndof_f+1)[0:ndof_f]
dx_f     = xaxis_f[1] - xaxis_f[0]

if figure==5:
  A_f = get_upwind(ndof_f, dx_f)
  u0fine   = solution_linear(np.zeros(ndof_f), A_f)
elif figure==6 or figure==7:
  A_f = get_centered(ndof_f, dx_f)
  u0fine   = solution_linear(np.zeros(ndof_f), A_f)
elif figure==8:
  u0fine = solution_dedalus(np.zeros(ndof_f), ndof_f)
else:
  sys.exit("This should have been caught above")

norm_l2  = np.zeros((3,np.size(nsteps)))
norm_inf = np.zeros((3,np.size(nsteps)))
dt_f_v   = np.zeros((1,np.size(nsteps)))
dt_c_v   = np.zeros((1,np.size(nsteps)))
A_f_eig  = np.zeros((3,np.size(nsteps)), dtype='complex')

for nn in range(3):

  ndof_c  = ndof_c_v[nn]
  xaxis_c = np.linspace(0.0, 1.0, ndof_c+1)[0:ndof_c]
  dx_c    = xaxis_c[1] - xaxis_c[0]

  ### Eigenvalues of A_f
  eig_val, eig_vec = LA.eig(A_f.todense())
    
  if figure==5:
    A_c = get_upwind(ndof_c, dx_c)
    u0coarse = solution_linear(np.zeros(ndof_c), A_c)
    filename = 'figure_5.pdf'
  elif figure==6:
    A_c = get_centered(ndof_c, dx_c)
    u0coarse = solution_linear(np.zeros(ndof_c), A_c)
    filename = 'figure_6.pdf'
  elif figure==7:
    A_c = get_centered(ndof_c, dx_c)
    u0coarse = solution_linear(np.zeros(ndof_c), A_c)
    filename = 'figure_7.pdf'    
  elif figure==8:
    u0coarse = solution_dedalus(np.zeros(ndof_c), ndof_c)
    filename = 'figure_8.pdf'
  else:
    sys.exit("Value of figure should be")    
      
  for mm in range(np.size(nsteps)):
  
    if figure==5 or figure==7:
      Rz = np.vectorize(ie)
      para     = parareal(0.0, Tend, nslices, impeuler, impeuler, nsteps[mm], nsteps[mm], tol, maxiter, u0fine, u0coarse)
    elif figure==6:
      Rz = np.vectorize(trap)      
      para     = parareal(0.0, Tend, nslices, trapezoidal, trapezoidal, nsteps[mm], nsteps[mm], tol, maxiter, u0fine, u0coarse)    
    elif figure==8:
     para     = parareal(0.0, Tend, nslices, integrator_dedalus, integrator_dedalus, nsteps[mm], nsteps[mm], tol, maxiter, u0fine, u0coarse)    
    else:
      sys.exit("Value of figure should be")    
    Pmat, Bmat = para.get_parareal_matrix()
    dt_f_v[0,mm] = para.timemesh.slices[0].int_fine.dt
    dt_c_v[0,mm] = para.timemesh.slices[0].int_coarse.dt
    
    # Sort according to absolute values of R(z) with z = lambda*dt_f
    sort_index = np.argsort(np.abs(Rz(eig_val*dt_f_v[0,mm])))
     
    # Store eigenvalue ndof_c+1. first "truncated" EV
    A_f_eig[nn,mm] = np.flip((eig_val[sort_index]))[ndof_c+1]
  
    ### Parareal iteration: y^k+1 = Pmat*y^k + Bmat*b
    norm_l2[nn,mm] = np.linalg.norm(Pmat.todense(), 2)
    

rcParams['figure.figsize'] = 2.5, 2.5
fs = 8
ms = 4
fig = plt.figure(1)
#plt.plot(dt_f_v[0,:], norm_l2[0,:], 'bo-', label='m='+str(ndof_c_v[0]), markersize=ms)
#plt.plot(dt_f_v[0,:], norm_l2[1,:], 'rx-', label='m='+str(ndof_c_v[1]), markersize=ms)
#plt.plot(dt_f_v[0,:], norm_l2[2,:], 'cd-', label='m='+str(ndof_c_v[2]), markersize=ms)
#plt.loglog(dt_f_v[0,:], norm_l2[0,:]-np.abs(np.exp(np.multiply(A_f_eig[0,:],dt_f_v[0,:]))), 'bo-', label='m='+str(ndof_c_v[0]), markersize=ms)
#plt.loglog(dt_f_v[0,:], norm_l2[1,:]-np.abs(np.exp(np.multiply(A_f_eig[1,:],dt_f_v[0,:]))), 'rx-', label='m='+str(ndof_c_v[1]), markersize=ms)
print(norm_l2[1,:])
print(A_f_eig[1,:])
print("\n")
print(norm_l2[2,:])
print(A_f_eig[2,:])

plt.loglog(dt_f_v[0,:], norm_l2[2,:]-0.0*np.abs(np.exp(np.multiply(A_f_eig[2,:],dt_f_v[0,:]))), 'cd-', label='m='+str(ndof_c_v[2]), markersize=ms)
#plt.plot(dt_f_v[0,:], 1.0 + 0.0*dt_f_v[0,:], 'k:')
plt.legend(loc='best', bbox_to_anchor=(0.5, 0.5), fontsize=fs, prop={'size':fs-2}, handlelength=3)
plt.xlabel(r'$\delta t = \Delta t$', fontsize=fs)
#plt.ylabel(r'$|| \mathbf{E} ||_2$', fontsize=fs)
plt.ylabel(r'$|| \mathbf{E} ||_2 - | \exp(\lambda_{m+1} \delta t) |$', fontsize=fs)

plt.xlim([0.0, dt_f_v[0,0]])
plt.ylim([1e-2, 1e1])
#plt.xlabel([0, maxiter])
plt.gcf().savefig(filename, bbox_inches='tight')
call(["pdfcrop", filename, filename])
plt.show()
