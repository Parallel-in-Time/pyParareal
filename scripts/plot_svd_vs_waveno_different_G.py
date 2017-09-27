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

if __name__ == "__main__":

    Tend     = 16.0
    nslices  = int(Tend) # Make sure each time slice has length 1
    U_speed  = 1.0
    nu       = 0.0
    ncoarse  = 1
    nfine    = 10
    dx       = 1.0
    Nsamples = 80
    u0_val   = np.array([[1.0]], dtype='complex')

    k_vec = np.linspace(0.0, np.pi, Nsamples+1, endpoint=False)

    # compute sigma for three difference choices of coarse method: backward Euler,
    # propagator with exact phase and with exact amplification factor
    svds = np.zeros((3,np.size(k_vec)))

    for j in range(3):
      for i in range(0,np.size(k_vec)):
        waveno = k_vec[i]
        symb = -(1j*U_speed*waveno + nu*waveno**2)
        symb_coarse = symb
        #    symb_coarse = -(1.0/dx)*(1.0 - np.exp(-1j*waveno*dx))
        
        stab_ex = np.exp(symb)
        
        # Solution objects define the problem
        u0      = solution_linear(u0_val, np.array([[symb]],dtype='complex'))
        ucoarse = solution_linear(u0_val, np.array([[symb_coarse]],dtype='complex'))
        para = parareal(0.0, Tend, nslices, intexact, impeuler, nfine, ncoarse, 0.0, 0, u0)
        stab_coarse = para.timemesh.slices[0].get_coarse_update_matrix(ucoarse)

        if j==2:
          stab_tailor = abs(stab_ex)*np.exp(1j*np.angle(stab_coarse[0,0])) # exact amplification factor
        elif j==1:
          stab_tailor = abs(stab_coarse[0,0])*np.exp(1j*np.angle(stab_ex)) # exact phase speed
        
        if not j==0:
          # Re-Create the parareal object to be used in the remainder
          stab_tailor = sparse.csc_matrix(np.array([stab_tailor], dtype='complex'))
        
          # Use tailored integrator as coarse method
          para = parareal(0.0, Tend, nslices, intexact, stab_tailor, nfine, ncoarse, 0.0, 1, u0)

        svds[j,i]         = para.get_max_svd(ucoarse=ucoarse)

    rcParams['figure.figsize'] = 2.5, 2.5
    fs = 8
    fig  = plt.figure()
    plt.plot(k_vec, svds[0,:], 'b-o', label="Backward Euler", markersize=fs/2, markevery=(1,6))
    plt.plot(k_vec, svds[1,:], 'r-s', label=r"$R_1$", markersize=fs/2, markevery=(3,6))
    plt.plot(k_vec, svds[2,:], 'g-x', label=r"$R_2$", markersize=fs/2, markevery=(3,6))
    plt.plot(k_vec, 1.0+0.0*k_vec, 'k--')
    plt.gca().tick_params(axis='both', which='major', labelsize=fs-2)
    plt.gca().tick_params(axis='x', which='minor', bottom='off')
    plt.xlabel(r'Wave number $\kappa$', fontsize=fs, labelpad=1)
    plt.xlim([k_vec[0], k_vec[-1]])
    plt.ylim([0.0, 1.4])
    plt.ylabel(r'Maximum singular value $\sigma$', fontsize=fs, labelpad=0)
    plt.legend(loc='lower right', fontsize=fs, prop={'size':fs-2}, handlelength=3)
    filename='svd_vs_waveno_different_G.pdf'
    fig.savefig(filename, bbox_inches='tight')
    call(["pdfcrop", filename, filename])
#    plt.show()
