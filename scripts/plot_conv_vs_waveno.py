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

    Nk    = 6
    k_vec = np.linspace(0, np.pi, Nk+1, endpoint=False)
    k_vec = k_vec[1:]
    k_vec = [k_vec[0], k_vec[1], k_vec[-1]]
    Tend    = 16.0    
    nslices = 16
    U_speed = 1.0
    nu      = 0.0
    ncoarse = 1
    nfine   = 1

    err       = np.zeros((np.size(k_vec),nslices))
    err_phase = np.zeros((np.size(k_vec),nslices))
    err_amp   = np.zeros((np.size(k_vec),nslices))
    svds      = np.zeros((np.size(k_vec),1))

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

      svds[kk,0] = para.get_max_svd()

      if abs(stab_fine[0,0]-stab_ex)>1e-14:
        print "WARNING: Fine method is not the exact integrator..."    

      omega_fine = solve_omega(stab_fine[0,0])
      phase_fine = omega_fine.real/k_vec[kk]
      amp_fine   = np.exp(omega_fine.imag)

      for n in range(1,nslices+1):
        stab_para = para.get_parareal_stab_function(n)
        err[kk,n-1] = abs(stab_para[0,0] - stab_fine[0,0])

    fs = 8
    rcParams['figure.figsize'] = 3.54, 3.54
    fig = plt.figure()
    iter_v = range(1,nslices)
    assert np.max(err[:,-1])<1e-10, ("For at least one wavenumber, Parareal did not fully converge for niter=nslices. Error: %5.3e" % np.max(err[:,-1]))
    plt.semilogy(iter_v, err[0,0:-1], 'b-o', label=(r"$\kappa$=%4.2f" % k_vec[0]), markersize=fs/2)
    #plt.semilogy(iter_v, err[0,0]*np.power(svds[0], iter_v), 'b--')

    plt.semilogy(iter_v, err[1,0:-1], 'r-s', label=(r"$\kappa$=%4.2f" % k_vec[1]), markersize=fs/2)
    #plt.semilogy(iter_v, err[1,0]*np.power(svds[1], iter_v), 'r--')

    plt.semilogy(iter_v, err[2,0:-1], 'g-x', label=(r"$\kappa$=%4.2f" % k_vec[2]), markersize=fs/2)
    #plt.semilogy(iter_v, err[2,0]*np.power(svds[2], iter_v), 'g--')

    plt.legend(loc='lower left', fontsize=fs, prop={'size':fs-2}, handlelength=3)
    plt.xlabel('Parareal Iteration $k$', fontsize=fs)
    plt.ylabel('Parareal defect', fontsize=fs)    
    plt.xticks(np.arange(iter_v[0], nslices, 2), fontsize=fs)
    plt.yticks([1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2], fontsize=fs)
    filename = 'parareal-conv-waveno.pdf'
    plt.gcf().savefig(filename, bbox_inches='tight')
    call(["pdfcrop", filename, filename])
    #plt.show()
