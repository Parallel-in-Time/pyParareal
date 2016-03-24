import sys
sys.path.append('../src')

from parareal import parareal
from impeuler import impeuler
from intexact import intexact
from solution_linear import solution_linear
import numpy as np
from pylab import rcParams
import matplotlib.pyplot as plt
from subprocess import call
def solve_omega(R):
  return 1j*( np.log(abs(R)) + 1j*np.angle(R) )

if __name__ == "__main__":

    Nx = 200
    x = np.linspace(0,20,Nx+1,endpoint=False)
    x = x[0:Nx]

    Nk    = 4
    k_vec = np.linspace(0, np.pi, Nk+1, endpoint=False)
    k_vec = k_vec[1:]

    Tend    = 16.0    
    nslices = 16
    U_speed = 1.0
    nu      = 0.0
    ncoarse = 1
    nfine   = 1

    err       = np.zeros((np.size(k_vec),nslices))
    err_phase = np.zeros((np.size(k_vec),nslices))
    err_amp   = np.zeros((np.size(k_vec),nslices))

    for kk in range(0,np.size(k_vec)):
    
      symb      = -(1j*U_speed*k_vec[kk] + nu*k_vec[kk]**2)
      u0_val    = np.array([[1.0]], dtype='complex')
      u0        = solution_linear(u0_val, np.array([[symb]],dtype='complex'))
      para      = parareal(0.0, Tend, nslices, intexact, impeuler, nfine, ncoarse, 0.0, 0, u0)

      stab_ex   = np.exp(-1j*U_speed*k_vec[kk]*Tend)*np.exp(-nu*k_vec[kk]**2*Tend)

      stab_coarse   = para.timemesh.slices[0].get_coarse_update_matrix(u0)
      stab_coarse = stab_coarse**nslices

      stab_fine = para.timemesh.slices[0].get_fine_update_matrix(u0)
      stab_fine = stab_fine**nslices

      if abs(stab_fine[0,0]-stab_ex)>1e-14:
        print "WARNING: Fine method is not the exact integrator..."    

      y_start = np.exp(1j*k_vec[kk]*x)
      y_ex    = stab_ex*y_start

      y_coarse= stab_coarse[0,0]*y_start
      y_fine  = stab_fine[0,0]*y_start

      omega_fine = solve_omega(stab_fine[0,0])
      phase_fine = omega_fine.real/k_vec[kk]
      amp_fine   = np.exp(omega_fine.imag)

      for n in range(1,nslices+1):
        stab_para = para.get_parareal_stab_function(n)
        y_para  = stab_para[0,0]*y_start
        err[kk,n-1] = np.linalg.norm(y_para - y_fine, np.inf)/np.linalg.norm(y_fine, np.inf)
        omega_para = solve_omega(stab_para[0,0])
        phase_para = omega_para.real/k_vec[kk]
        amp_para = np.exp(omega_para.imag)
        err_phase[kk,n-1] = abs(phase_para - phase_fine)/abs(phase_fine)
        err_amp[kk,n-1]   = abs(amp_para - amp_fine)/abs(amp_fine)

    fs = 8
    rcParams['figure.figsize'] = 2.5, 2.5
    fig = plt.figure()
    iter_v = range(1,nslices)
    assert np.max(err[:,-1])<1e-14, "For at least one wavenumber, Parareal did not fully converge for niter=nslices"
    plt.semilogy(iter_v, err[0,0:-1], 'b-o', label=(r"$\kappa$=%4.2f" % k_vec[0]), markersize=fs/2)
    plt.semilogy(iter_v, err[1,0:-1], 'r-s', label=(r"$\kappa$=%4.2f" % k_vec[1]), markersize=fs/2)
    plt.semilogy(iter_v, err[2,0:-1], 'g-x', label=(r"$\kappa$=%4.2f" % k_vec[2]), markersize=fs/2)
    plt.semilogy(iter_v, err[3,0:-1], 'k-d', label=(r"$\kappa$=%4.2f" % k_vec[3]), markersize=fs/2)
    plt.legend(loc='lower left', fontsize=fs, prop={'size':fs-2}, handlelength=3)
    plt.xlabel('Parareal Iteration $k$', fontsize=fs)
    plt.ylabel('Parareal defect', fontsize=fs)    
    plt.xticks(np.arange(iter_v[0], iter_v[-1], 2), fontsize=fs)
    plt.yticks([1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2], fontsize=fs)
    filename = 'parareal-conv-waveno.pdf'
    plt.gcf().savefig(filename, bbox_inches='tight')
    call(["pdfcrop", filename, filename])

    #fig = plt.figure()
    #plt.semilogy(iter_v, err_phase[0,0:-1], 'b-o', label=(r"$\kappa$=%4.2f" % k_vec[0]), markersize=fs/2)
    #plt.semilogy(iter_v, err_phase[1,0:-1], 'r-s', label=(r"$\kappa$=%4.2f" % k_vec[1]), markersize=fs/2)
    #plt.semilogy(iter_v, err_phase[2,0:-1], 'g-x', label=(r"$\kappa$=%4.2f" % k_vec[2]), markersize=fs/2)
    #plt.semilogy(iter_v, err_phase[3,0:-1], 'k-d', label=(r"$\kappa$=%4.2f" % k_vec[3]), markersize=fs/2)
    #plt.legend(loc='lower left', fontsize=fs, prop={'size':fs}, handlelength=3)

    #fig = plt.figure()
    #plt.semilogy(iter_v, err_amp[0,0:-1], 'b-o', label=(r"$\kappa$=%4.2f" % k_vec[0]), markersize=fs/2)
    #plt.semilogy(iter_v, err_amp[1,0:-1], 'r-s', label=(r"$\kappa$=%4.2f" % k_vec[1]), markersize=fs/2)
    #plt.semilogy(iter_v, err_amp[2,0:-1], 'g-x', label=(r"$\kappa$=%4.2f" % k_vec[2]), markersize=fs/2)
    #plt.semilogy(iter_v, err_amp[3,0:-1], 'k-d', label=(r"$\kappa$=%4.2f" % k_vec[3]), markersize=fs/2)
    #plt.legend(loc='lower left', fontsize=fs, prop={'size':fs}, handlelength=3)

    #plt.show()
