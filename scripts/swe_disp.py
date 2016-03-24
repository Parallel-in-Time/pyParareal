import numpy as np
from scipy.sparse import linalg
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from subprocess import call
import sympy

def findomega(Z):
  assert np.array_equal(np.shape(Z),[3,3]), 'Not a 3x3 matrix...'
  omega = sympy.Symbol('omega')
  func = (sympy.exp(-1j*omega)-Z[0,0])*(sympy.exp(-1j*omega) - Z[1,1])*(sympy.exp(-1j*omega)-Z[2,2]) \
         - Z[0,1]*Z[1,2]*Z[2,0] - Z[0,2]*Z[1,0]*Z[2,1]                                               \
         - Z[0,2]*(sympy.exp(-1j*omega) - Z[1,1])*Z[2,0]                                             \
         - Z[0,1]*Z[1,0]*(sympy.exp(-1j*omega) - Z[2,2])                                             \
         - Z[1,2]*Z[2,1]*(sympy.exp(-1j*omega) - Z[0,0])
  solsym = sympy.solve(func, omega)
  sol0 = complex(solsym[0])
  sol1 = complex(solsym[1])
  sol2 = complex(solsym[2])
  return sol2

Tend = 1.0
Nsamples = 25
k_vec = np.linspace(0.0, np.pi, Nsamples+1, endpoint=False)
k_vec = k_vec[1:]

g = 0.1
H = 1.0
f = 0.25

phase      = np.zeros((2,Nsamples))
amp_factor = np.zeros((2,Nsamples))

for i in range(0,Nsamples):
  Lmat = -1.0*np.array([[0.0, -f, g*1j*k_vec[i] ],
                   [f, 0.0, 0.0],
                   [H*1j*k_vec[i], 0, 0]], dtype = 'complex')

  stab_ex         = linalg.expm(Lmat)
  omega           = findomega(stab_ex)
  phase[1,i]      = omega.real/k_vec[i]
  amp_factor[1,i] = np.exp(omega.imag)

  omega_ex        = np.sqrt( g*H*k_vec[i]**2 + f**2)
  phase[0,i]      = omega_ex.real/k_vec[i]
  amp_factor[0,i] = np.exp(omega_ex.imag)

#rcParams['figure.figsize'] = 1.5, 1.5
fs = 14
fig  = plt.figure()
plt.plot(k_vec, phase[0,:], '--', color='k', linewidth=1.5, label='Exact')
plt.plot(k_vec, phase[1,:], '-', color='g', linewidth=1.5, label='Solved')
plt.xlabel('Wave number', fontsize=fs, labelpad=0.25)
plt.ylabel('Phase speed', fontsize=fs, labelpad=0.5)
plt.xlim([k_vec[0], k_vec[-1:]])
#plt.ylim([0.0, 1.1])
fig.gca().tick_params(axis='both', labelsize=fs)
plt.legend(loc='lower left', fontsize=fs, prop={'size':fs-2})

fig  = plt.figure()
plt.plot(k_vec, amp_factor[0,:], '--', color='k', linewidth=1.5, label='Exact')
plt.plot(k_vec, amp_factor[1,:], '-',  color='g', linewidth=1.5, label='Solved')
plt.xlabel('Wave number', fontsize=fs, labelpad=0.25)
plt.ylabel('Amplification factor', fontsize=fs, labelpad=0.5)
fig.gca().tick_params(axis='both', labelsize=fs)
plt.xlim([k_vec[0], k_vec[-1:]])
#  plt.ylim([k_vec[0], k_vec[-1:]])
plt.legend(loc='lower left', fontsize=fs, prop={'size':fs-2})
plt.gca().set_ylim([0.0, 1.1])
#plt.xticks([0, 1, 2, 3], fontsize=fs)
plt.show()