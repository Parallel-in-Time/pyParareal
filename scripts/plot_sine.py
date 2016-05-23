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

if __name__ == "__main__":

    Nx = 200
    x = np.linspace(0,20,Nx+1,endpoint=False)
    x = x[0:Nx]

    Nk    = 4
    k_vec = np.linspace(0, np.pi, Nk+1, endpoint=False)
    k_vec = k_vec[1:]
    # Select a wave number
    k_ind = 3
    k     = k_vec[k_ind]

    Tend    = 16.0    
    nslices = 16
    U_speed = 1.0
    nu      = 0.0
    ncoarse = 1
    nfine   = 1

    err = np.zeros(nslices)
    para_show = np.zeros((3,Nx))
    niter_show = [2, 4, 6]

    symb      = -(1j*U_speed*k + nu*k**2)
    u0_val    = np.array([[1.0]], dtype='complex')
    u0        = solution_linear(u0_val, np.array([[symb]],dtype='complex'))
    para      = parareal(0.0, Tend, nslices, intexact, impeuler, nfine, ncoarse, 0.0, 0, u0)

    stab_ex   = np.exp(-1j*U_speed*k*Tend)*np.exp(-nu*k**2*Tend)

    stab_coarse = para.timemesh.slices[0].get_coarse_update_matrix(u0)
    stab_coarse = stab_coarse**nslices

    stab_fine = para.timemesh.slices[0].get_fine_update_matrix(u0)
    stab_fine = stab_fine**nslices

    if abs(stab_fine[0,0]-stab_ex)>1e-14:
      print "WARNING: Fine method is not the exact integrator..."    

    y_start = np.exp(1j*k*x)
    y_ex    = stab_ex*y_start

    y_coarse= stab_coarse[0,0]*y_start
    y_fine  = stab_fine[0,0]*y_start

    for n in range(0,np.size(niter_show)):
      stab_para = para.get_parareal_stab_function(niter_show[n])
      para_show[n,:] = (stab_para[0,0]*y_start).real

    fs = 8
    rcParams['figure.figsize'] = 2.5, 2.5
    fig = plt.figure()
    plt.plot(x, y_coarse.real,  'b--', label='Coarse')
    plt.plot(x, para_show[0,:], 'r--+', label='Parareal k='+str(niter_show[0]), markevery=(5, 20), markersize=fs/2, mew=1)
    plt.plot(x, para_show[1,:], 'r:s', label='Parareal k='+str(niter_show[1]), markevery=(10,20),  markersize=fs/2, mew=1)
    plt.plot(x, para_show[2,:], 'r-o', label='Parareal k='+str(niter_show[2]), markevery=(15,20),  markersize=fs/2, mew=1)
    #plt.plot(x, y_ex.real,      'g--', label='Fine')
    plt.legend(loc='lower left', fontsize=fs, prop={'size':fs-2}, handlelength=3)
    plt.title((r'$\kappa$ = %4.2f' % k), fontsize=fs)
    plt.ylim([-2.5, 1.5])
    plt.xlim([x[0], x[-1]])
    plt.xlabel('x', fontsize=fs)
    plt.ylabel('y', fontsize=fs)
    filename = 'parareal-sine-'+str(k_ind)+'.pdf'
    plt.gcf().savefig(filename, bbox_inches='tight')
    call(["pdfcrop", filename, filename])
#    plt.show()
