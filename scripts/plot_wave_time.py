import sys
sys.path.append('../src')

from parareal import parareal
from impeuler import impeuler
from intexact import intexact
from trapezoidal import trapezoidal
from special_integrator import special_integrator
from solution_linear import solution_linear
import numpy as np
import scipy.sparse as sparse
import math

from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from subprocess import call
import sympy
from pylab import rcParams

def solve_omega(R):
  return 1j*( np.log(abs(R)) + 1j*np.angle(R) )

def findroots(R, n):
  assert abs(n - float(int(n)))<1e-14, "n must be an integer or a float equal to an integer"
  p = np.zeros(int(n)+1, dtype='complex')
  p[-1] = -R
  p[0]  = 1.0
  return np.roots(p)

def normalise(R, T, target):
  roots = findroots(R, T)
  for x in roots:
    assert abs(x**T-R)<1e-5, ("Element in roots not a proper root: err=%5.3e" % abs(x**T-R))
  minind = np.argmin(abs(np.angle(roots) - target))
  return roots[minind]

if __name__ == "__main__":


    Tend     = 16.0
    nslices  = int(Tend) # Make sure each time slice has length 1
    U_speed  = 1.0
    nu       = 0.0
    ncoarse  = 5
    nfine    = 10
    taxis    = np.linspace(0.0, Tend, nfine*nslices)
    niter_v  = [3]

    k_vec = np.linspace(0.0, np.pi, 6, endpoint=False)
    k_vec = k_vec[1:]
    waveno = k_vec[-1]

    symb = -(1j*U_speed*waveno + nu*waveno**2)
    symb_coarse = symb
#    symb_coarse = -(1.0/dx)*(1.0 - np.exp(-1j*waveno*dx))

    # Solution objects define the problem
    u0_val     = np.array([[1.0]], dtype='complex')
    u0      = solution_linear(u0_val, np.array([[symb]],dtype='complex'))
    ucoarse = solution_linear(u0_val, np.array([[symb_coarse]],dtype='complex'))

    para = parareal(0.0, Tend, nslices, intexact, trapezoidal, nfine, ncoarse, 0.0, niter_v[0], u0)

    # get update matrix for imp Euler over one slice
    stab_fine   = para.timemesh.slices[0].get_fine_update_matrix(u0)    
    stab_coarse = para.timemesh.slices[0].get_coarse_update_matrix(ucoarse)
    stab_ex     = np.exp(symb)

    rcParams['figure.figsize'] = 7.5, 7.5
    fs = 8
    fig  = plt.figure()

    for k in range(0,nslices):
#    for k in range(0,2):
      plt.clf()

      stab_para_n0 = para.get_parareal_stab_function(k, ucoarse=ucoarse)
      stab_para_np1 = para.get_parareal_stab_function(k+1, ucoarse=ucoarse)

      stab_para_norm_n0  = normalise(stab_para_n0[0,0], Tend, np.angle(stab_ex))
      stab_para_norm_np1 = normalise(stab_para_np1[0,0], Tend, np.angle(stab_ex))

      sol_fine      = solve_omega(stab_fine[0,0])
      sol_ex        = solve_omega(stab_ex)
      sol_coarse    = solve_omega(stab_coarse[0,0])
      sol_para_n0   = solve_omega(stab_para_norm_n0)
      sol_para_np1  = solve_omega(stab_para_norm_np1)

      y_fine = np.exp(-1j*sol_fine*taxis)
      y_ex   = np.exp(-1j*sol_ex*taxis)
      y_coarse = np.exp(-1j*sol_coarse*taxis)
      y_para_n0 = np.exp(-1j*sol_para_n0*taxis)
      y_para_np1 = np.exp(-1j*sol_para_np1*taxis)

      update = y_para_np1 - y_para_n0


      plt.plot(taxis, y_fine.real, 'b')
  #    plt.plot(taxis, y_ex.real, 'g')
      plt.plot(taxis, y_para_n0.real, 'k')
#      plt.plot(taxis, (y_fine - y_para_n0).real, 'k')
      #plt.plot(taxis, (y_fine - y_coarse).real, 'r')
#      plt.plot(taxis, update.real, 'r')
      plt.ylim([-2, 2])
      if k<nslices-1:
        plt.show(block=False)
      else:
        plt.show()
      plt.pause(2)

