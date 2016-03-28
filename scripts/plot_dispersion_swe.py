import sys
sys.path.append('../src')

from parareal import parareal
from impeuler import impeuler
from intexact import intexact
from solution_linear import solution_linear
import numpy as np
from scipy.sparse import linalg
import math

from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from subprocess import call
import sympy

def solve_omega(Z):
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

def findroots(R, n):
  assert abs(n - float(int(n)))<1e-14, "n must be an integer or a float equal to an integer"
  p = np.zeros(n+1, dtype='complex')
  p[-1] = -R
  p[0]  = 1.0
  return np.roots(p)

def normalise(R, T, target):
  roots = findroots(R, T)
  for x in roots:
    assert abs(x**T-R)<1e-11, ("Element in roots not a proper root: err=%5.3e" % abs(x**T-R))
  minind = np.argmin(abs(np.angle(roots) - target))
  return roots[minind]


if __name__ == "__main__":


    Tend     = 4.0
    nslices  = 4
    U_speed  = 1.0
    nu       = 0.0
    ncoarse  = 1
    nfine    = 10
    niter_v  = [3]
    dx       = 1.0
    Nsamples = 5

    k_vec = np.linspace(0.0, np.pi, Nsamples+1, endpoint=False)
    k_vec = k_vec[1:]

    phase      = np.zeros((6,Nsamples))
    amp_factor = np.zeros((6,Nsamples))
    u0_val     = np.array([1.0, 1.0, 1.0], dtype='complex')
    targets    = np.zeros((3,Nsamples))

    for i in range(0,np.size(k_vec)):
      
      f = 0.1
      g = 1.0
      H = 1.0

      Lmat = -1.0*np.array([[0.0, -f, g*1j*k_vec[i] ],         \
                   [f, 0.0, 0.0],                              \
                   [H*1j*k_vec[i], 0, 0]], dtype = 'complex')
  
      
      u0   = solution_linear(u0_val, Lmat)
      para = parareal(0.0, Tend, nslices, impeuler, impeuler, nfine, ncoarse, 0.0, niter_v[0], u0)
      
      
      # get update matrix for imp Euler over one slice
      stab_fine   = para.timemesh.slices[0].get_fine_update_matrix(u0)
      
      stab_coarse = para.timemesh.slices[0].get_coarse_update_matrix(u0)
      
      stab_ex     = linalg.expm(Lmat)

      sol_fine   = solve_omega(stab_fine)
      sol_ex     = solve_omega(stab_ex)
      sol_coarse = solve_omega(stab_coarse)
      
      phase[0,i]      = sol_ex.real/k_vec[i]
      amp_factor[0,i] = np.exp(sol_ex.imag)
      
      phase[1,i]      = sol_fine.real/k_vec[i]
      amp_factor[1,i] = np.exp(sol_fine.imag)
      
      phase[2,i]      = sol_coarse.real/k_vec[i]
      amp_factor[2,i] = np.exp(sol_coarse.imag)
      
      # Compute Parareal phase velocity and amplification factor
      
      for jj in range(0,1):
        stab_para = para.get_parareal_stab_function(niter_v[jj])
        print solve_omega(stab_para)
        print "\n"
#
#        if i==0:
#          targets[jj,0] = np.angle(stab_ex)
#        
#        stab_para_norm = normalise(stab_para, Tend, targets[jj,i])
#        # Make sure that stab_norm*dt = stab
#        err = np.linalg.norm(stab_para_norm**Tend - stab_para, np.inf)
#        if err>1e-10:
#          print ("WARNING: power of normalised update does not match update over full length of time. error %5.3e" % err)
#        
#        if i<np.size(k_vec)-1:
#          targets[jj,i+1] = np.angle(stab_para_norm)
#        
#        #print ("k: %5.3f" % k_vec[i])
#        sol_para   = solve_omega(stab_para_norm)
#
#        # Now solve for discrete phase 
#        phase[3+jj,i]      = sol_para.real/k_vec[i]
#        amp_factor[3+jj,i] = np.exp(sol_para.imag)


    ###
    #rcParams['figure.figsize'] = 1.5, 1.5
    fs = 14
    fig  = plt.figure()
    plt.plot(k_vec, phase[0,:], '--', color='k', linewidth=1.5, label='Exact')
    plt.plot(k_vec, phase[1,:], '-',  color='g', linewidth=1.5, label='Fine')
    plt.plot(k_vec, phase[2,:], '-o', color='b', linewidth=1.5, label='Coarse', markevery=5, markersize=fs/2)
    #plt.plot(k_vec, phase[3,:], '-s', color='r', linewidth=1.5, label='Parareal k='+str(niter_v[0]), markevery=5, mew=1.0, markersize=fs/2)
    #plt.plot(k_vec, phase[4,:], '-d', color='r', linewidth=1.5, label='Parareal k='+str(niter_v[1]), markevery=5, mew=1.0, markersize=fs/2)
    #plt.plot(k_vec, phase[5,:], '-x', color='r', linewidth=1.5, label='Parareal k='+str(niter_v[2]), markevery=5, mew=1.0, markersize=fs/2)
    plt.xlabel('Wave number', fontsize=fs, labelpad=0.25)
    plt.ylabel('Phase speed', fontsize=fs, labelpad=0.5)
    plt.xlim([k_vec[0], k_vec[-1:]])
    plt.ylim([0.0, 1.1*U_speed])
    fig.gca().tick_params(axis='both', labelsize=fs)
    plt.legend(loc='lower left', fontsize=fs, prop={'size':fs-2})
    #plt.xticks([0, 1, 2, 3], fontsize=fs)
    #plt.show()
    #filename = 'sdc-fwsw-disprel-phase-K'+str(K)+'-M'+str(swparams['num_nodes'])+'.pdf'
    #plt.gcf().savefig(filename, bbox_inches='tight')
    #call(["pdfcrop", filename, filename])

    fig  = plt.figure()
    plt.plot(k_vec, amp_factor[0,:], '--', color='k', linewidth=1.5, label='Exact')
    plt.plot(k_vec, amp_factor[1,:], '-',  color='g', linewidth=1.5, label='Fine')
    plt.plot(k_vec, amp_factor[2,:], '-o', color='b', linewidth=1.5, label='Coarse', markevery=5, markersize=fs/2)
    plt.plot(k_vec, amp_factor[3,:], '-s', color='r', linewidth=1.5, label='Parareal k='+str(niter_v[0]), markevery=5, mew=1.0, markersize=fs/2)
    plt.plot(k_vec, amp_factor[4,:], '-d', color='r', linewidth=1.5, label='Parareal k='+str(niter_v[1]), markevery=5, mew=1.0, markersize=fs/2)
    plt.plot(k_vec, amp_factor[5,:], '-x', color='r', linewidth=1.5, label='Parareal k='+str(niter_v[2]), markevery=5, mew=1.0, markersize=fs/2)
    plt.xlabel('Wave number', fontsize=fs, labelpad=0.25)
    plt.ylabel('Amplification factor', fontsize=fs, labelpad=0.5)
    fig.gca().tick_params(axis='both', labelsize=fs)
    plt.xlim([k_vec[0], k_vec[-1:]])
  #  plt.ylim([k_vec[0], k_vec[-1:]])
    plt.legend(loc='lower left', fontsize=fs, prop={'size':fs-2})
    plt.gca().set_ylim([0.0, 1.1])
    #plt.xticks([0, 1, 2, 3], fontsize=fs)
    plt.show()
    #filename = 'sdc-fwsw-disprel-ampfac-K'+str(K)+'-M'+str(swparams['num_nodes'])+'.pdf'
    #plt.gcf().savefig(filename, bbox_inches='tight')
    #call(["pdfcrop", filename, filename])

