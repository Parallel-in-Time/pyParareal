import numpy as np
from scipy.sparse import linalg
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from subprocess import call
import sympy
import warnings

def findomega(Z):
  omega = sympy.Symbol('omega')
  func  = sympy.exp(-1j*omega) - Z
  solsym = sympy.solve(func, omega)
  return complex(solsym[0])

def findroots(R, n):
  assert abs(n - float(int(n)))<1e-14, "n must be an integer or a float equal to an integer"
  p = np.zeros(n+1, dtype='complex')
  p[-1] = -R
  p[0]  = 1.0
  return np.roots(p)

def normalise(R, T, target):
  roots = findroots(R, T)
  for x in roots:
    assert abs(x**T-R)<1e-10, ("Element in roots not a proper root: err=%5.3e" % abs(x**T-R))
  minind = np.argmin(abs(np.angle(roots) - target))
  return roots[minind]

Tend     = 4.0
nslices  = int(Tend)
Nsamples = 30
k_vec    = np.linspace(0.0, np.pi, Nsamples+1, endpoint=False)
k_vec    = k_vec[1:]
U_speed  = 1.0
nu       = 0.0

phase      = np.zeros((3,Nsamples))
amp_factor = np.zeros((3,Nsamples))
target     = np.zeros(Nsamples)

for i in range(0, Nsamples):
  symb         = -(1j*U_speed*k_vec[i] + nu*k_vec[i]**2)
  stab_ex      = np.exp(symb*Tend)
  stab_ex_unit = np.exp(symb)
  # First make sure that stab_ex_unit**nslices = stab_ex
  assert abs(stab_ex_unit**nslices - stab_ex)<1e-14, "stab_ex_unit**nslices does not match stab_ex"

  if i==0:
    target[0] = np.angle(stab_ex_unit)
  
  # Now normalise stab_ex
  stab_norm = normalise(stab_ex, Tend, target[i])

  # nslices many applications of stab_norm give stab_ex
  assert abs(stab_norm**nslices - stab_ex)<1e-14, "stab_norm**nslices does not match stab_ex"

  # stab_norm is equal to stab_ex_unit
  assert abs(stab_norm - stab_ex_unit)<1e-14, "stab_norm does not match stab_unit"

  if i<Nsamples-1:
    target[i+1] = np.angle(stab_norm)

  omega_norm = findomega(stab_norm)
  omega_ex   = U_speed*k_vec[i] # ignore nu term for the moment

  phase[0,i] = omega_norm.real/k_vec[i]
  phase[1,i] = omega_ex.real/k_vec[i]

  amp_factor[0,i] = np.exp(omega_norm.imag)
  amp_factor[1,i] = np.exp(omega_ex.imag)


###
#rcParams['figure.figsize'] = 2.5, 2.5
fs = 8
fig  = plt.figure()
plt.plot(k_vec, phase[0,:], '--', color='k', linewidth=1.5, label='Exact')
plt.plot(k_vec, phase[1,:], '-o', color='g', linewidth=1.5, label='Normalised',   markevery=(1,5), markersize=fs/2)
plt.xlabel('Wave number', fontsize=fs, labelpad=0.25)
plt.ylabel('Phase speed', fontsize=fs, labelpad=0.5)
plt.xlim([k_vec[0], k_vec[-1:]])
plt.ylim([0.0, 1.1*U_speed])
fig.gca().tick_params(axis='both', labelsize=fs)
plt.legend(loc='lower left', fontsize=fs, prop={'size':fs-2})
plt.xticks([0, 1, 2, 3], fontsize=fs)
plt.show()

fig  = plt.figure()
plt.plot(k_vec, amp_factor[0,:], '--', color='k', linewidth=1.5, label='Exact')
plt.plot(k_vec, amp_factor[1,:], '-o', color='g', linewidth=1.5, label='Normalised',   markevery=(1,5), markersize=fs/2)
plt.xlabel('Wave number', fontsize=fs, labelpad=0.25)
plt.ylabel('Amplification factor', fontsize=fs, labelpad=0.5)
fig.gca().tick_params(axis='both', labelsize=fs)
plt.xlim([k_vec[0], k_vec[-1:]])
plt.ylim([0, 1.1*U_speed])
plt.legend(loc='lower left', fontsize=fs, prop={'size':fs-2})
plt.gca().set_ylim([0.0, 1.1])
plt.xticks([0, 1, 2, 3], fontsize=fs)
plt.show()
