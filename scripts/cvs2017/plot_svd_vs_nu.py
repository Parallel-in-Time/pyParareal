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
    ncoarse  = 1
    nfine    = 10
    dx       = 1.0
    Nsamples = 60
    u0_val     = np.array([[1.0]], dtype='complex')

    k_vec = np.linspace(0.0, np.pi, Nsamples+1, endpoint=False)
    k_vec = k_vec[1:]
    waveno = k_vec[-1]

    propagators = [impeuler, trapezoidal]

    nu_v = np.logspace(-16, 0, num=80, endpoint=True)
    # insert nu=0 as first value
    nu_v = np.insert(nu_v, 0, 0.0)
    svds = np.zeros((np.size(propagators),np.size(nu_v)))
	
    for i in range(0,np.size(nu_v)):
      symb = -(1j*U_speed*waveno + nu_v[i]*waveno**2)
      symb_coarse = symb

    # Solution objects define the problem
      u0      = solution_linear(u0_val, np.array([[symb]],dtype='complex'))
      ucoarse = solution_linear(u0_val, np.array([[symb_coarse]],dtype='complex'))

      for j in range(2):
        para = parareal(0.0, Tend, nslices, intexact, propagators[j], nfine, ncoarse, 0.0, 0, u0)
        svds[j,i]         = para.get_max_svd(ucoarse=ucoarse)        

    rcParams['figure.figsize'] = 2.5, 2.5
    fs = 8
    fig  = plt.figure()
    plt.semilogx(nu_v, svds[0,:], 'b', label='Implicit Euler')
    plt.semilogx(nu_v, svds[1,:], 'r', label='Trapezoidal')
    plt.semilogx(nu_v, 0.0*nu_v+1.0, 'k--')
    plt.gca().tick_params(axis='both', which='major', labelsize=fs-2)
    plt.gca().tick_params(axis='x', which='minor', bottom='off')
    plt.gca().set_xticks(np.logspace(-16, 0, num=5))
    plt.xlabel(r'Diffusion coefficient $\nu$', fontsize=fs, labelpad=1)
    plt.ylabel(r'Maximum singular value $\sigma$', fontsize=fs, labelpad=0)
    plt.legend(loc='center left', fontsize=fs, prop={'size':fs-2})
    filename='svd_vs_nu.pdf'
    fig.savefig(filename, bbox_inches='tight')
    call(["pdfcrop", filename, filename])
#    plt.show()
