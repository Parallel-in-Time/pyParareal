import numpy as np
from scipy.sparse import linalg
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from subprocess import call
import sympy
import warnings

def findomega(Z):
  assert np.size(Z)==3, 'Not a 3 vector of length 3...'
  omega = sympy.Symbol('omega')
  expf   = sympy.exp(-1j*omega)
  func   = (expf - Z[0])*(expf - Z[1])*(expf - Z[2])
  solsym = sympy.solve(func, omega)
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
  print roots
  for x in roots:
    assert abs(x**T-R)<1e-10, ("Element in roots not a proper root: err=%5.3e" % abs(x**T-R))
#  minind = np.argmin(abs(np.angle(roots) - target))
#  minind = np.argmin(abs( abs(np.angle(roots)) - abs(target)))
  minind = np.argmin( abs(roots.real - target.real) + abs(roots.imag - target.imag) )
  return roots[minind]

Tend     = 4.0
nslices  = int(Tend)
Nsamples = 25
k_vec    = np.linspace(0.0, np.pi, Nsamples+1, endpoint=False)
k_vec    = k_vec[1:]

g = 1.0
H = 1.0
f = 1.0

phase      = np.zeros((2,Nsamples))
amp_factor = np.zeros((2,Nsamples))
targets    = np.zeros((3,Nsamples), dtype = 'complex')

#for i in range(0,Nsamples):
for i in range(0,4):
  Lmat = -1.0*np.array([[0.0, -f, g*1j*k_vec[i] ],
                   [f, 0.0, 0.0],
                   [H*1j*k_vec[i], 0, 0]], dtype = 'complex')

  stab_ex         = linalg.expm(Lmat*Tend)
  
  # if i==0, compute targets from contiuous stability function
  stab_ex_unit = linalg.expm(Lmat)
  D_unit, V_unit = np.linalg.eig(stab_ex_unit)
  if i==0:
    for j in range(0,3):
      targets[j,0] = D_unit[j]

  # normalize for Tend = 1.0
  D, V = np.linalg.eig(stab_ex)
  Vinv = np.linalg.inv(V)
  err_commute = np.linalg.norm( D.dot(Vinv) - Vinv.dot(D) , np.inf )
  if err_commute>1e-14:
    warnings.warn("matrices D and Vinv do not commute: %.3E" % err_commute)
  S    = np.zeros(3, dtype = 'complex')
  for j in range(0,3):
    print ("correct value: %s" % D_unit[j])
    print ("target value: %s"  % targets[j,i])
    S[j] = normalise(D[j], Tend, targets[j,i])
    print ("selected value: %s" % S[j])
    print ("\n")
    # Set targets for next step of loop
    if i<Nsamples-1:
      targets[j,i+1] = S[j]

  assert np.linalg.norm(S**nslices - D, np.inf)<1e-10, "Entries in S to the power of P dont reproduce D"

  #
  # IN THIS TESTCASE, THE NORMALISATION PROCEDURE SHOULD RELIABLY PRODUCE THE STABILITY FUNCTION OVER THE UNIT INTERVAL
  #
  err_norm_unit = np.linalg.norm( V.dot(np.diag(S).dot(Vinv)) - stab_ex_unit, np.inf)
  if err_norm_unit>1e-10:
    warnings.warn("Normalised stability function not equal to stability function over unit interval -- error: %.3E" % err_norm_unit)

  omega_unit      = findomega(D_unit)
  omega           = findomega(S)
  omega_err = abs(omega - omega_unit)
  if omega_err>1e-10:
    warnings.warn("difference between omega and omega_unit: %.3E" % omega_err)

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