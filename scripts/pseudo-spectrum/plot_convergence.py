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
from integrator_dedalus import integrator_dedalus
from solution_linear import solution_linear
from solution_dedalus import solution_dedalus
from get_matrix import get_upwind, get_centered, get_diffusion
from pseudo_spectral_radius import pseudo_spectral_radius
from parameter import parameter

from pylab import rcParams
import matplotlib.pyplot as plt
from subprocess import call

#def uex(x):
#  return np.sin(2.0*np.pi*x)

# 1 = advection with implicit Euler / upwind FD
# 2 = advection with trapezoidal rule / centered FD
# 3 = Dedalus
try:
  figure      =  int(sys.argv[1]) # 1 generates figure_1, 2 generates figure_2
except:
  print("No or wrong command line argument provided, creating figure 11. Use 11, 12, 13 or 14 as command line argument.")
  figure = 11
assert 11<= figure <= 14, "Figure should be 11, 12, 13 or 14"
  
if figure==11 or figure==12:
  par = parameter(dedalus = False)
elif figure==13 or figure==14:
  par = parameter(dedalus = True)
else:
  sys.exit("Figure should be 11, 12, 13 or 14")

Tend, nslices, maxiter, nfine, ncoarse, tol, epsilon, ndof_f = par.getpar()

ndof_c_v = [24, 27, 30]

xaxis_f = np.linspace(0.0, 1.0, ndof_f+1)[0:ndof_f]
dx_f    = xaxis_f[1] - xaxis_f[0]
#u0_f = uex(xaxis_f)
u0_f = np.zeros(ndof_f)

if figure==11:
  A_f = get_upwind(ndof_f, dx_f)
  u0fine     = solution_linear(np.zeros(ndof_f), A_f)
  filename = 'figure_11.pdf'
elif figure==12:
  A_f = get_centered(ndof_f, dx_f)
  u0fine     = solution_linear(np.zeros(ndof_f), A_f)
  filename = 'figure_12.pdf'
elif figure==13:  
  u0fine = solution_dedalus(np.zeros(ndof_f), ndof_f)
  filename = 'figure_13.pdf'
elif figure==14:
  u0fine = solution_dedalus(np.zeros(ndof_f), ndof_f)
  filename = 'figure_14.pdf'
else:
  sys.exit("Figure should be 11, 12, 13 or 14")
  
defect_l2  = np.zeros((3,maxiter))
slopes     = np.zeros(3)

for nn in range(3):

  ndof_c = ndof_c_v[nn]
  xaxis_c = np.linspace(0.0, 1.0, ndof_c+1)[0:ndof_c]
  dx_c = xaxis_c[1] - xaxis_c[0]
  #u0_c   = uex(xaxis_c)
  u0_c = np.zeros(ndof_c)
  
  if figure==11:
    A_c = get_upwind(ndof_c, dx_c)
    u0coarse = solution_linear(np.zeros(ndof_c), A_c)
  elif figure==12:
    A_c = get_centered(ndof_c, dx_c)
    u0coarse = solution_linear(np.zeros(ndof_c), A_c)
  elif figure==13:
    u0coarse = solution_dedalus(np.zeros(ndof_c), ndof_c)    
  elif figure==14:
    u0coarse = solution_dedalus(np.zeros(ndof_c), ndof_c)  
  else:
    sys.exit("Wrong figure number")
    

  if figure==11:
    para     = parareal(0.0, Tend, nslices, impeuler, impeuler, nfine, ncoarse, tol, maxiter, u0fine, u0coarse)
  elif figure==12:
    para     = parareal(0.0, Tend, nslices, trapezoidal, trapezoidal, nfine, ncoarse, tol, maxiter, u0fine, u0coarse)
  elif (figure==13 or figure==14):
    para     = parareal(0.0, Tend, nslices, integrator_dedalus, integrator_dedalus, nfine, ncoarse, tol, maxiter, u0fine, u0coarse)    
  else:
    sys.exit("Wrong figure number")
  
  Pmat, Bmat = para.get_parareal_matrix()
  
  psr_pmat = pseudo_spectral_radius(Pmat, eps=epsilon)
  psr, a, b, c = psr_pmat.get_psr(verbose=True)

  ### Parareal iteration: y^k+1 = Pmat*y^k + Bmat*b

  #slopes[nn] = np.linalg.norm(Pmat.todense(),2)
  slopes[nn] = psr
  
  # Now do Parareal iteration
  P_power_k = Pmat.todense()
  for k in range(maxiter):    
    P_power_k        = Pmat@P_power_k
    defect_l2[nn,k]  = np.linalg.norm(P_power_k , 2)
  
rcParams['figure.figsize'] = 2.5, 2.5
fs = 8
ms = 4
fig = plt.figure(1)
plt.semilogy(range(1,maxiter+1), defect_l2[0,:], 'bo-', label='m='+str(ndof_c_v[0]), markersize=ms)
plt.semilogy(range(1,maxiter+1), defect_l2[1,:], 'rx-', label='m='+str(ndof_c_v[1]), markersize=ms)
plt.semilogy(range(1,maxiter+1), defect_l2[2,:], 'cd-', label='m='+str(ndof_c_v[2]), markersize=ms)


plt.legend(loc='best', bbox_to_anchor=(0.5, 0.5), fontsize=fs, prop={'size':fs-2}, handlelength=3)

plt.semilogy(range(1,5), [slopes[0]**(val-1)*1.1*defect_l2[0,0] for val in range(1,5)], 'b--')
plt.semilogy(range(1,5), [slopes[1]**(val-1)*1.1*defect_l2[1,0] for val in range(1,5)], 'r--')
plt.semilogy(range(1,5), [slopes[2]**(val-1)*1.1*defect_l2[2,0] for val in range(1,5)], 'c--')


plt.xlabel('$k$', fontsize=fs)
plt.ylabel('$||\mathbf{E}^k ||_2$', fontsize=fs)

#plt.ylim([1e-15, 1e1])
plt.xlim([1, maxiter])
plt.xticks(range(1,maxiter,2))
plt.gcf().savefig(filename, bbox_inches='tight')
call(["pdfcrop", filename, filename])
plt.show()

#fig = plt.figure(2)
#plt.plot(xaxis_f, u[-ndof_f:,0], 'r+')
#plt.plot(xaxis_f, uex(xaxis_f, Tend), 'b--')

#plt.show()
