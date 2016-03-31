import numpy as np
from scipy.sparse import linalg
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from subprocess import call
import sympy
import warnings

def findomega(Z):
  assert np.array_equal(np.shape(Z),[3,3]), 'Not a 3x3 matrix...'
  omega = sympy.Symbol('omega')
  func = (sympy.exp(-1j*omega)-Z[0,0])*(sympy.exp(-1j*omega) - Z[1,1])*(sympy.exp(-1j*omega)-Z[2,2]) \
         - Z[0,1]*Z[1,2]*Z[2,0] - Z[0,2]*Z[1,0]*Z[2,1]                                               \
         - Z[0,2]*(sympy.exp(-1j*omega) - Z[1,1])*Z[2,0]                                             \
         - Z[0,1]*Z[1,0]*(sympy.exp(-1j*omega) - Z[2,2])                                             \
         - Z[1,2]*Z[2,1]*(sympy.exp(-1j*omega) - Z[0,0])
  solsym = sympy.solve(func, omega)
#  np.set_printoptions(precision=4)
  print solsym
  sols = np.array([complex(solsym[0]), complex(solsym[1]), complex(solsym[2])], dtype='complex')
  return sols[2]

def findroots(R, n):
  assert abs(n - float(int(n)))<1e-14, "n must be an integer or a float equal to an integer"
  p = np.zeros(n+1, dtype='complex')
  p[-1] = -R
  p[0]  = 1.0
  return np.roots(p)

def normalise(R, T, target):
  roots = findroots(R, T)
  print "roots:"
  print np.angle(roots)
  print target
  print ""
  for x in roots:
    assert abs(x**T-R)<1e-10, ("Element in roots not a proper root: err=%5.3e" % abs(x**T-R))
  minind = np.argmin(abs(np.angle(roots) - target))
  return roots[minind]

Tend     = 4.0
nslices  = int(Tend)
Nsamples = 10
k_vec    = np.linspace(0.0, np.pi, Nsamples+1, endpoint=False)
k_vec    = k_vec[1:]

g = 1.0
H = 1.0
f = 0.0

phase      = np.zeros((2,Nsamples))
amp_factor = np.zeros((2,Nsamples))
targets    = np.zeros((3,Nsamples))

for i in range(0,3):
  Lmat = -1.0*np.array([[0.0, -f, g*1j*k_vec[i] ],
                   [f, 0.0, 0.0],
                   [H*1j*k_vec[i], 0, 0]], dtype = 'complex')

  stab_ex         = linalg.expm(Lmat*Tend)
  
  # if i==0, compute targets from contiuous stability function
  stab_ex_unit = linalg.expm(Lmat)
  if i==0:
    D, V = np.linalg.eig(stab_ex_unit)
    for j in range(0,3):
      targets[j,0] = np.angle(D[j])
    
  # normalize for Tend = 1.0
  D, V = np.linalg.eig(stab_ex)
  Vinv = np.linalg.inv(V)
  stab_test = V.dot(np.diag(D).dot(Vinv))
  assert np.linalg.norm(stab_test - stab_ex, np.inf)<1e-14, "Matrix reconstructed from EV decomposition does not match original matrix"

  S    = np.zeros(3, dtype = 'complex')
  for j in range(0,3):
    S[j] = normalise(D[j], Tend, targets[j,i])
    # Set targets for next step of loop
    if i<Nsamples-1:
      targets[j,i+1] = np.angle(S[j])

  assert np.linalg.norm(S**nslices - D, np.inf)<1e-14, "Entries in S to the power of P dont reproduce D"

  stab_normalise  = V.dot((np.diag(S)).dot(Vinv))

  assert np.linalg.norm( np.linalg.matrix_power(stab_normalise,nslices) - stab_ex, np.inf)<1e-10, "Power of normalised stability function not equal to non-normalised stability function"
  assert np.linalg.norm( np.linalg.matrix_power(stab_ex_unit,nslices) - stab_ex, np.inf)<1e-10, "Power of unit interval stability function ot equal to non-normalised stability function"
  err_norm_unit = np.linalg.norm( stab_normalise - stab_ex_unit, np.inf)
  if err_norm_unit>1e-14:
    warnings.warn("Normalised stability function not equal to stability function over unit interval -- error: %.3E" % err_norm_unit)

  #np.set_printoptions(precision=4)
  #print np.around(stab_ex_unit, decimals = 8)
  #print np.around(stab_normalise, decimals = 8)
  #print ""
  omega           = findomega(stab_ex_unit)
  omega           = findomega(stab_normalise)
  print "\n"
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
plt.ylim([0.0, 1.1*np.max(phase)])
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