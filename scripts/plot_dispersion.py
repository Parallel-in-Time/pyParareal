import sys
sys.path.append('../src')

from parareal import parareal
from impeuler import impeuler
from intexact import intexact
from solution_linear import solution_linear
import numpy as np
#from scipy.sparse import linalg
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


if __name__ == "__main__":


    Tend     = 16.0
    nslices  = int(Tend) # Make sure each time slice has length 1
    U_speed  = 1.0
    nu       = 0.0
    ncoarse  = 1
    nfine    = 10
    niter_v  = [5, 10, 15]
    dx       = 0.1
    Nsamples = 30

    k_vec = np.linspace(0.0, np.pi, Nsamples+1, endpoint=False)
    k_vec = k_vec[1:]

    phase      = np.zeros((6,Nsamples))
    amp_factor = np.zeros((6,Nsamples))
    u0_val     = np.array([[1.0]], dtype='complex')
    targets    = np.zeros((3,Nsamples))

    for i in range(0,np.size(k_vec)):
      
      symb = -(1j*U_speed*k_vec[i] + nu*k_vec[i]**2)
#      symb_coarse = symb
      symb_coarse = -(1.0/dx)*(1.0 - np.exp(-1j*k_vec[i]*dx))

      u0      = solution_linear(u0_val, np.array([[symb]],dtype='complex'))
      ucoarse = solution_linear(u0_val, np.array([[symb_coarse]],dtype='complex'))
      para = parareal(0.0, Tend, nslices, intexact, impeuler, nfine, ncoarse, 0.0, niter_v[0], u0)
      
      
      # get update matrix for imp Euler over one slice
      stab_fine   = para.timemesh.slices[0].get_fine_update_matrix(u0)
      stab_fine   = stab_fine
      
      stab_coarse = para.timemesh.slices[0].get_coarse_update_matrix(u0)
      stab_coarse = stab_coarse
      
      stab_ex     = np.exp(symb)

      sol_fine   = solve_omega(stab_fine[0,0])
      sol_ex     = solve_omega(stab_ex)
      sol_coarse = solve_omega(stab_coarse[0,0])
      
      phase[0,i]      = sol_ex.real/k_vec[i]
      amp_factor[0,i] = np.exp(sol_ex.imag)
      
      phase[1,i]      = sol_fine.real/k_vec[i]
      amp_factor[1,i] = np.exp(sol_fine.imag)
      
      phase[2,i]      = sol_coarse.real/k_vec[i]
      amp_factor[2,i] = np.exp(sol_coarse.imag)
      
      # Compute Parareal phase velocity and amplification factor
      
      for jj in range(0,3):
        stab_para = para.get_parareal_stab_function(k=niter_v[jj], ucoarse=ucoarse)

        if i==0:
          targets[jj,0] = np.angle(stab_ex)
        
        stab_para_norm = normalise(stab_para[0,0], Tend, targets[jj,i])
        # Make sure that stab_norm*dt = stab
        err = abs(stab_para_norm**Tend - stab_para)
        if err>1e-10:
          print ("WARNING: power of normalised update does not match update over full length of time. error %5.3e" % err)
        
        if i<np.size(k_vec)-1:
          targets[jj,i+1] = np.angle(stab_para_norm)
        
        #print ("k: %5.3f" % k_vec[i])
        sol_para   = solve_omega(stab_para_norm)

        # Now solve for discrete phase 
        phase[3+jj,i]      = sol_para.real/k_vec[i]
        amp_factor[3+jj,i] = np.exp(sol_para.imag)


    ###
    rcParams['figure.figsize'] = 2.5, 2.5
    fs = 8
    fig  = plt.figure()
    plt.plot(k_vec, phase[0,:], '--', color='k', linewidth=1.5, label='Exact')
    plt.plot(k_vec, phase[1,:], '-o', color='g', linewidth=1.5, label='Fine',   markevery=(1,5), markersize=fs/2)
    plt.plot(k_vec, phase[2,:], '-o', color='b', linewidth=1.5, label='Coarse', markevery=(3,5), markersize=fs/2)
    plt.plot(k_vec, phase[3,:], '-s', color='r', linewidth=1.5, label='Parareal k='+str(niter_v[0]), markevery=(1,6), mew=1.0, markersize=fs/2)
    plt.plot(k_vec, phase[4,:], '-d', color='r', linewidth=1.5, label='Parareal k='+str(niter_v[1]), markevery=(3,6), mew=1.0, markersize=fs/2)
    plt.plot(k_vec, phase[5,:], '-x', color='r', linewidth=1.5, label='Parareal k='+str(niter_v[2]), markevery=(5,6), mew=1.0, markersize=fs/2)
    plt.xlabel('Wave number', fontsize=fs, labelpad=0.25)
    plt.ylabel('Phase speed', fontsize=fs, labelpad=0.5)
    plt.xlim([k_vec[0], k_vec[-1:]])
    plt.ylim([0.0, 1.1*U_speed])
    fig.gca().tick_params(axis='both', labelsize=fs)
    plt.legend(loc='lower left', fontsize=fs, prop={'size':fs-2})
    plt.xticks([0, 1, 2, 3], fontsize=fs)
    #plt.show()
    filename = 'parareal-dispersion-phase.pdf'
    plt.gcf().savefig(filename, bbox_inches='tight')
    call(["pdfcrop", filename, filename])

    fig  = plt.figure()
    plt.plot(k_vec, amp_factor[0,:], '--', color='k', linewidth=1.5, label='Exact')
    plt.plot(k_vec, amp_factor[1,:], '-o', color='g', linewidth=1.5, label='Fine',   markevery=(1,5), markersize=fs/2) 
    plt.plot(k_vec, amp_factor[2,:], '-o', color='b', linewidth=1.5, label='Coarse', markevery=(3,5), markersize=fs/2)
    plt.plot(k_vec, amp_factor[3,:], '-s', color='r', linewidth=1.5, label='Parareal k='+str(niter_v[0]), markevery=(1,6), mew=1.0, markersize=fs/2)
    plt.plot(k_vec, amp_factor[4,:], '-d', color='r', linewidth=1.5, label='Parareal k='+str(niter_v[1]), markevery=(3,6), mew=1.0, markersize=fs/2)
    plt.plot(k_vec, amp_factor[5,:], '-x', color='r', linewidth=1.5, label='Parareal k='+str(niter_v[2]), markevery=(5,6), mew=1.0, markersize=fs/2)
    plt.xlabel('Wave number', fontsize=fs, labelpad=0.25)
    plt.ylabel('Amplification factor', fontsize=fs, labelpad=0.5)
    fig.gca().tick_params(axis='both', labelsize=fs)
    plt.xlim([k_vec[0], k_vec[-1:]])
    plt.ylim([0, 1.1*U_speed])
    plt.legend(loc='lower left', fontsize=fs, prop={'size':fs-2})
    plt.gca().set_ylim([0.0, 1.1])
    plt.xticks([0, 1, 2, 3], fontsize=fs)
    #plt.show()
    filename = 'parareal-dispersion-ampf.pdf'
    plt.gcf().savefig(filename, bbox_inches='tight')
    call(["pdfcrop", filename, filename])

