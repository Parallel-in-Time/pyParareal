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

# computes the frequency omega = 1j*log(R)
def solve_omega(R):
  return 1j*( np.log(abs(R)) + 1j*np.angle(R) )

# finds all roots of the stability function
def findroots(R, n):
  assert abs(n - float(int(n)))<1e-14, "n must be an integer or a float equal to an integer"
  p = np.zeros(int(n)+1, dtype='complex')
  p[-1] = -R
  p[0]  = 1.0
  return np.roots(p)

# normalises the stability function of Parareal from [0,Tend] to [0,1]
# by computing all Tend=P roots and then selecting the one that is closest to the given
# target angle
def normalise(R, T, target):
  roots = findroots(R, T)
  
  # make sure all computed values are actually roots
  for x in roots:
    assert abs(x**T-R)<1e-3, ("Element in roots not a proper root: err=%5.3e" % abs(x**T-R))
  
  # find root that minimises distance to target angle
  minind = np.argmin(abs(np.angle(roots) - target))
  return roots[minind]

#
# Main script
#
if __name__ == "__main__":

    # Tend has to be an integer as we assume Tend = P with P being the number of processors
    Tend     = 16.0
    
    # number of time slices, equal to the number of processors P
    nslices  = int(Tend) # Make sure each time slice has length 1
    
    # advection speed
    U_speed  = 1.0
    
    # diffusivity parameter
    nu       = 0.1
    
    # select coarse integrator:
    # 0 = normal backward Euler method
    # 1 = artificially constructed method with phase error from backward Euler and exact amplification factor
    # 2 = artificially constructed method with exact phase and amplification factor from backward Euler
    artificial_coarse = 0
    
    # equivalently, artifical_fine==1 constructs a fine propagator with exact amplitude but the same coarse propagation characteristics as backward Euler
    artificial_fine = 0
    
    # Select finite difference stencil for coarse propagator
    # stencil = 0 : exact symbol, no approximation of spatial derivative
    # stencil = 1 : first order upwind  (only fur nu=0)
    # stencil = 2 : second order centred (only for nu=0)
    stencil = 0

    ncoarse  = 1
    nfine    = 10
    niter_v  = [5, 10, 15]
    dx       = 1.0 # only relevant if finite difference symbol is used instead of analytic symbol of spatial derivative operator
    
    
    # number of discrete values between kappa=0 and kappa=pi for which the dispersion relation is computed.
    # try to increase this value of the normalisation fails.
    Nsamples = 30

    k_vec = np.linspace(0.0, np.pi, Nsamples+1, endpoint=False)
    k_vec = k_vec[1:]

    phase      = np.zeros((6,Nsamples))
    amp_factor = np.zeros((6,Nsamples))
    u0_val     = np.array([[1.0]], dtype='complex')
    targets    = np.zeros((3,Nsamples))

    for i in range(0,np.size(k_vec)):
      
      symb = -(1j*U_speed*k_vec[i] + nu*k_vec[i]**2)
     
      if stencil==0:
        symb_coarse = symb
      
      elif stencil==1:
        assert nu==0, "Approximate coarse symbol currently only implemented for nu=0"
        symb_coarse = -U_speed*( 1.0 - np.exp(-1j*k_vec[i]*dx) )/dx

      elif stencil==2:
        assert nu==0, "Approximate coarse symbol currently only implemented for nu=0"
        symb_coarse = -U_speed*1j*np.sin(k_vec[i]*dx)/dx

      else:
        raise Exception("Not implemented")

      # Solution objects define the problem
      u0      = solution_linear(u0_val, np.array([[symb]],dtype='complex'))
      ucoarse = solution_linear(u0_val, np.array([[symb_coarse]],dtype='complex'))
      
      # create Parareal object; selects exact integrator (intexact) as fine propagator and backward Euler (impeuler) as coarse
      para = parareal(0.0, Tend, nslices, intexact, impeuler, nfine, ncoarse, 0.0, niter_v[0], u0)
           
      # get update matrix for imp Euler over one slice
      stab_fine   = para.timemesh.slices[0].get_fine_update_matrix(u0)    
      stab_coarse = para.timemesh.slices[0].get_coarse_update_matrix(ucoarse)
      
      # exact stability function is exponential
      stab_ex     = np.exp(symb)


      #
      # MODIFICATIONS FOR SPECIALLY TAILORED COARSE METHOD
      #

      if artificial_coarse==2:
      # for stab = r*exp(i*theta), r defines the amplitude factor and theta the phase speed
        stab_coarse = abs(stab_coarse[0,0])*np.exp(1j*np.angle(stab_ex)) # exact phase speed
      elif artificial_coarse==1:
        stab_coarse = abs(stab_ex)*np.exp(1j*np.angle(stab_coarse[0,0])) # exact amplification factor

      # if an artifical coarse method is used, need to reconstruct the Parareal object
      if not artificial_coarse==0:
        # Re-Create the parareal object to be used in the remainder
        stab_coarse = sparse.csc_matrix(np.array([stab_coarse], dtype='complex'))
      
        # Use tailored integrator as coarse method
        para = parareal(0.0, Tend, nslices, intexact, stab_coarse, nfine, ncoarse, 0.0, niter_v[0], u0)
      
      if artificial_fine==1:
        assert artificial_coarse==0, "Using artifical coarse and fine propagators together is not implemented and probably not working correctly"
        stab_fine = abs(stab_ex)*np.exp(1j*np.angle(stab_coarse[0,0]))
        stab_fine = sparse.csc_matrix(np.array([stab_fine], dtype='complex'))
        # Must use nfine=1 in this case
        para = parareal(0.0, Tend, nslices, stab_fine, impeuler, 1, ncoarse, 0.0, niter_v[0], u0)
      
      # compute frequency omeaga for fine propagator, exact propagator and coarse propagator
      sol_fine   = solve_omega(stab_fine[0,0])
      sol_ex     = solve_omega(stab_ex)
      sol_coarse = solve_omega(stab_coarse[0,0])
      
      # compute phase speed and amplification factor for fine, coarse and exact propagator
      phase[0,i]      = sol_ex.real/k_vec[i]
      amp_factor[0,i] = np.exp(sol_ex.imag)
      
      phase[1,i]      = sol_fine.real/k_vec[i]
      amp_factor[1,i] = np.exp(sol_fine.imag)
      
      phase[2,i]      = sol_coarse.real/k_vec[i]
      amp_factor[2,i] = np.exp(sol_coarse.imag)
      

      #################################################
      
      # Compute Parareal phase velocity and amplification factor for 3 different values of K (number of iterations)
      for jj in range(0,3):
      
        # stability function of Parareal over interval [0,Tend]
        stab_para = para.get_parareal_stab_function(k=niter_v[jj], ucoarse=ucoarse)

        # for very first wave number, target angle is computed by using angle of exact stability function
        if i==0:
          targets[jj,0] = np.angle(stab_ex)

        # normalise stability function to [0,1] by selecting the correct root (see functions defined above)
        stab_para_norm = normalise(stab_para[0,0], Tend, targets[jj,i])
        
        # Make sure that stab_norm*dt = stab
        err = abs(stab_para_norm**Tend - stab_para)
        if err>1e-6:
          print ("WARNING: power of normalised update does not match update over full length of time. error %5.3e" % err)
        
        if i<np.size(k_vec)-1:
          targets[jj,i+1] = np.angle(stab_para_norm)
        
        #print ("k: %5.3f" % k_vec[i])
        sol_para   = solve_omega(stab_para_norm)

        # Now solve for discrete phase 
        phase[3+jj,i]      = sol_para.real/k_vec[i]
        amp_factor[3+jj,i] = np.exp(sol_para.imag)

    ###
    rcParams['figure.figsize'] = 3.54, 3.54
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
    plt.legend(loc='lower left', fontsize=fs, prop={'size':fs-2})
    plt.gca().set_ylim([0.0, 1.2])
    plt.xticks([0, 1, 2, 3], fontsize=fs)
    #plt.show()
    filename = 'parareal-dispersion-ampf.pdf'
    plt.gcf().savefig(filename, bbox_inches='tight')
    call(["pdfcrop", filename, filename])
