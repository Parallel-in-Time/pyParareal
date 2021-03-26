import sys
sys.path.append('../../src')

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
    nu_v     = [0.0, 1e-1, 0.5]
    ncoarse  = 1
    nfine    = 10
    dx       = 1.0
    Nsamples = 80
    u0_val   = np.array([[1.0]], dtype='complex')

    k_vec = np.linspace(0.0, np.pi, Nsamples+1, endpoint=False)

    svds = np.zeros((np.size(nu_v),np.size(k_vec)))

    for j in range(0,np.size(nu_v)):
      nu = nu_v[j]
      for i in range(0,np.size(k_vec)):
        waveno = k_vec[i]
        symb = -(1j*U_speed*waveno + nu*waveno**2)
        symb_coarse = symb
        #    symb_coarse = -(1.0/dx)*(1.0 - np.exp(-1j*waveno*dx))

        # Solution objects define the problem
        u0      = solution_linear(u0_val, np.array([[symb]],dtype='complex'))
        ucoarse = solution_linear(u0_val, np.array([[symb_coarse]],dtype='complex'))

        para = parareal(0.0, Tend, nslices, intexact, impeuler, nfine, ncoarse, 0.0, 0, u0)
        svds[j,i]         = para.get_max_svd(ucoarse=ucoarse)        

    rcParams['figure.figsize'] = 3.54, 3.54
    fs = 8
    fig  = plt.figure()
    plt.plot(k_vec, svds[0,:], 'b-o', label=(r'$\nu$=%3.2f' % nu_v[0]), markersize=fs/2, markevery=(1,6))
    plt.plot(k_vec, svds[1,:], 'r-s', label=(r'$\nu$=%3.2f' % nu_v[1]), markersize=fs/2, markevery=(3,6))
    plt.plot(k_vec, svds[2,:], 'g-x', label=(r'$\nu$=%3.2f' % nu_v[2]), markersize=fs/2, markevery=(5,6))
    plt.plot(k_vec, 1.0+0.0*k_vec, 'k--')
    plt.gca().tick_params(axis='both', which='major', labelsize=fs-2)
    plt.gca().tick_params(axis='x', which='minor', bottom='off')
    plt.xlabel(r'Wave number $\kappa$', fontsize=fs, labelpad=1)
    plt.xlim([k_vec[0], k_vec[-1]])
    plt.ylabel(r'Maximum singular value $\sigma$', fontsize=fs, labelpad=0)
    plt.legend(loc='upper left', fontsize=fs, prop={'size':fs-2}, handlelength=3)
    filename='svd_vs_waveno.pdf'
    fig.savefig(filename, bbox_inches='tight')
    call(["pdfcrop", filename, filename])
#    plt.show()
