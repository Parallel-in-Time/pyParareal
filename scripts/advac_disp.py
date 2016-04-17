import numpy as np
from scipy.sparse import linalg
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from subprocess import call
import sympy
import warnings

def selectomega(omega):
  assert np.size(omega)==2, "Should have 2 entries..."
  return omega[1]

def findomegasystem(Z):
  assert np.array_equal(np.shape(Z), [2,2]), "Must be 2x2 matrix..."
  omega = sympy.Symbol('omega')
  expf  = sympy.exp(-1j*omega)
  func  = (expf - Z[0,0])*(expf - Z[1,1]) - Z[1,0]*Z[0,1]
  solsym = sympy.solve(func, omega)
  sols = np.array([complex(solsym[0]), complex(solsym[1])], dtype='complex')
  return sols

def findomega(Z):
  assert np.size(Z)==2, 'Not a vector of length 2...'
  omega = sympy.Symbol('omega')
  expf   = sympy.exp(-1j*omega)
  func   = (expf - Z[0])*(expf - Z[1])
  solsym = sympy.solve(func, omega)
  sols = np.array([complex(solsym[0]), complex(solsym[1])], dtype='complex')
  return sols

def findroots(R, n):
  assert abs(n - float(int(n)))<1e-14, "n must be an integer or a float equal to an integer"
  p = np.zeros(n+1, dtype='complex')
  p[-1] = -R
  p[0]  = 1.0
  return np.roots(p)

def normalise(R, T, targets, verbose=False, exact=None):
  roots = findroots(R, T)
  if verbose:
    print ""
    print "roots:"
    print roots
  for x in roots:
    assert abs(x**T-R)<1e-10, ("Element in roots not a proper root: err=%5.3e" % abs(x**T-R))

  resi  = np.zeros((np.size(roots),np.size(targets)))
  for i in range(0,np.size(targets)):
    for j in range(0,np.size(roots)):
      resi[j,i]  = abs( roots[j] - targets[i] )

  # Select root that generates smallest residual across all targets:
  # find row number of smallest element in resi
  minind = np.argmin(resi) / np.size(targets)
  if verbose:
    print ("Target values:   %s" % targets)
    print ("Residuals:       %s" % resi)
    print ("Selected row:    %s" % minind)
    print ("Selected value:  %s" % roots[minind])
    if not exact==None:
      print ("Exact values:    %s" % exact)
  return roots[minind]


Tend     = 4.0
nslices  = int(Tend)
Nsamples = 80
k_vec    = np.linspace(0.0, np.pi, Nsamples+1, endpoint=False)
k_vec    = k_vec[1:]

Uadv   = 0.1
cspeed = 1.0

phase      = np.zeros((2,Nsamples))
amp_factor = np.zeros((2,Nsamples))
targets    = np.zeros((2,Nsamples), dtype = 'complex')

imax = 21
#imax = Nsamples
for i in range(0,imax):
  print ("---- i = %2i ---- " % i)
  Lmat = -1j*k_vec[i]*np.array([ [Uadv, cspeed], [cspeed, Uadv] ], dtype = 'complex')

  stab         = linalg.expm(Lmat*Tend)
  
  # if i==0, compute targets from contiuous stability function
  stab_unit = linalg.expm(Lmat)

  # analytic frequencies match frequencies computed from unit interval system
  omegas_unit = findomegasystem(stab_unit)
  phase[0,i]  = Uadv + cspeed
  phase[1,i]  = selectomega(omegas_unit).real/k_vec[i]

  # diagonalise unit system and verify that frequencies do not change
  Dunit, Vunit = np.linalg.eig(stab_unit)
  #Dunit = np.sort(Dunit)
  if i==0:
    targets[:,i] = Dunit
  
  omegas_diag_unit = findomega(Dunit)

  err_omega = np.linalg.norm(omegas_unit - omegas_diag_unit, np.inf)
  print ("Defect between unit and unit-diagonalised frequencies: %5.3E" % err_omega)

  # normalise stab function
  D, V = np.linalg.eig(stab)
  omegas_diag_full = findomega(D)

  err_omega = np.linalg.norm(omegas_diag_full/Tend - omegas_diag_unit, np.inf)
  print ""
  print omegas_diag_unit
  print omegas_diag_full/Tend
  print (">>>> Defect between omegas from diagonalised unit system and diagonalised full system: %5.3E" % err_omega)

  D = np.sort(D)
  Dtilde = np.zeros(2, dtype='complex')
  for j in range(0,2):
    Dtilde[j] = normalise(D[j], Tend, targets=targets[:,i], verbose=True, exact=Dunit)
    #Dtilde[j] = normalise(D[j], Tend, targets=Dunit, verbose=False)
  #Dtilde = np.sort(Dtilde)
  err_eigv = np.linalg.norm(Dtilde - Dunit, np.inf)
  print ("Defect between normalised and unit stability function eigenvalues: %5.3E" % err_eigv)

  if (i<imax-1):
    targets[:,i+1] = Dtilde

  # normalised stability function matches unit interval stability function
  stab_unit_sorted = V.dot(np.diag(Dunit).dot(np.linalg.inv(V)))
  stab_normalised  = V.dot(np.diag(Dtilde).dot(np.linalg.inv(V)))
  err_stab = np.linalg.norm(stab_normalised - stab_unit_sorted, np.infty)
  print ("Defect between unit interval and normalised stability matrix: %5.3E" % err_stab)
  ### NOTE: The defects here are caused by different ordering of eigenvalues!

  # frequencies of unit interval system match frequencies of normalised system
  omegas_diag_normalised = findomega(Dtilde)
  err_omega = np.linalg.norm(omegas_diag_unit - omegas_diag_normalised, np.inf)
  print ("Defect between frequencies from diagonalised normalised system and diagonalised unit intervall: %5.3E" % err_omega)
  assert err_omega<1e-12, "Mismatch in frequencies..."
  # end of loop body over k_vec

  print "------"

#rcParams['figure.figsize'] = 1.5, 1.5
if True:
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