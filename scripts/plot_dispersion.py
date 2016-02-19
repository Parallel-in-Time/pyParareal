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

def findomega(stab_fh, dt):
  omega = sympy.Symbol('omega')
  func = sympy.exp(-1j*omega)-stab_fh
  solsym = sympy.solve(func, omega)
  sol0 = complex(solsym[0])
  return sol0

def findomega_direct(stab_fh, dt):
  # Transform stab_fh into polar coordinates
  r = np.sqrt( stab_fh.real**2 + stab_fh.imag** 2)
  theta = np.angle(stab_fh)
  # log is log(r) + 1j*theta
  # return (1j*log - theta)/dt
  return (1j*np.log(r) - theta)/dt
#  return 1j*np.log(stab_fh)/dt


if __name__ == "__main__":


    Tend    = 8.0    
    nslices = 32
    U_speed = 1.0/Tend
    nu      = 0.0/Tend
    ncoarse = 1
    nfine   = 1
    niter   = 1

    Nsamples = 30

    k_vec = np.linspace(0.0, np.pi, Nsamples+1, endpoint=False)
    k_vec = k_vec[1:]

    phase      = np.zeros((4,Nsamples))
    amp_factor = np.zeros((4,Nsamples))
    u0_val     = np.array([[1.0]], dtype='complex')

    for i in range(0,np.size(k_vec)):            
      symb = -(1j*U_speed*k_vec[i] + nu*k_vec[i]**2)
      u0   = solution_linear(u0_val, np.array([[symb]],dtype='complex'))
      para = parareal(0.0, Tend, nslices, intexact, impeuler, nfine, ncoarse, 0.0, niter, u0)
      dt = para.timemesh.tend - para.timemesh.tstart

      stab_para = para.get_parareal_stab_function(niter)
      # get update matrix for imp Euler over one slice
      stab_fine   = para.timemesh.slices[0].get_fine_update_matrix(u0)
      stab_fine = stab_fine**nslices
      stab_coarse   = para.timemesh.slices[0].get_coarse_update_matrix(u0)
      stab_coarse = stab_coarse**nslices
      stab_ex   = np.exp(symb*dt)

      sol_para   = findomega_direct(stab_para[0,0], dt)
      sol_fine   = findomega_direct(stab_fine[0,0], dt)
      sol_ex     = findomega_direct(stab_ex, dt)
      sol_coarse = findomega_direct(stab_coarse[0,0], dt)

      # Now solve for discrete phase 
      phase[0,i]      = sol_para.real/k_vec[i]
      amp_factor[0,i] = np.exp(sol_para.imag)
      phase[1,i]      = sol_fine.real/k_vec[i]
      amp_factor[1,i] = np.exp(sol_fine.imag)
      phase[2,i]      = sol_ex.real/k_vec[i]
      amp_factor[2,i] = np.exp(sol_ex.imag)
      phase[3,i]      = sol_coarse.real/k_vec[i]
      amp_factor[3,i] = np.exp(sol_coarse.imag)

    ###
    #rcParams['figure.figsize'] = 1.5, 1.5
    fs = 14
    fig  = plt.figure()
    plt.plot(k_vec, Tend*phase[2,:], '--', color='k', linewidth=1.5, label='Exact')
    plt.plot(k_vec, Tend*phase[1,:], '-', color='g', linewidth=1.5, label='Fine')
    plt.plot(k_vec, Tend*phase[0,:], '-+', color='r', linewidth=1.5, label='Parareal', markevery=5, mew=1.0)
    plt.plot(k_vec, Tend*phase[3,:], '-o', color='b', linewidth=1.5, label='Coarse', markevery=5, markersize=fs/2)
    plt.xlabel('Wave number', fontsize=fs, labelpad=0.25)
    plt.ylabel('Phase speed', fontsize=fs, labelpad=0.5)
    plt.xlim([k_vec[0], k_vec[-1:]])
    #plt.ylim([0.0, 1.1*U_speed])
    fig.gca().tick_params(axis='both', labelsize=fs)
    plt.legend(loc='lower left', fontsize=fs, prop={'size':fs-2})
    plt.xticks([0, 1, 2, 3], fontsize=fs)
    plt.show()
    #filename = 'sdc-fwsw-disprel-phase-K'+str(K)+'-M'+str(swparams['num_nodes'])+'.pdf'
    #plt.gcf().savefig(filename, bbox_inches='tight')
    #call(["pdfcrop", filename, filename])

    fig  = plt.figure()
    plt.plot(k_vec, amp_factor[2,:], '--', color='k', linewidth=1.5, label='Exact')
    plt.plot(k_vec, amp_factor[1,:], '-', color='g', linewidth=1.5, label='Fine integrator')
    plt.plot(k_vec, amp_factor[0,:], '-+', color='r', linewidth=1.5, label='Parareal', markevery=5, mew=1.0)
    plt.plot(k_vec, amp_factor[3,:], '-o', color='b', linewidth=1.5, label='Coarse', markevery=5, markersize=fs/2)
    plt.xlabel('Wave number', fontsize=fs, labelpad=0.25)
    plt.ylabel('Amplification factor', fontsize=fs, labelpad=0.5)
    fig.gca().tick_params(axis='both', labelsize=fs)
    plt.xlim([k_vec[0], k_vec[-1:]])
  #  plt.ylim([k_vec[0], k_vec[-1:]])
    plt.legend(loc='lower left', fontsize=fs, prop={'size':fs-2})
    plt.gca().set_ylim([0.0, 1.1])
    plt.xticks([0, 1, 2, 3], fontsize=fs)
    plt.show()
    #filename = 'sdc-fwsw-disprel-ampfac-K'+str(K)+'-M'+str(swparams['num_nodes'])+'.pdf'
    #plt.gcf().savefig(filename, bbox_inches='tight')
    #call(["pdfcrop", filename, filename])

