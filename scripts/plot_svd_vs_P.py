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

    nslices_v = np.arange(2,32,2)

    U_speed  = 1.0
    nu       = 0.0
    ncoarse  = 1
    nfine    = 10
    u0_val     = np.array([[1.0]], dtype='complex')
    Nsamples = 40
    
    Nk    = 6
    k_vec = np.linspace(0, np.pi, Nk+1, endpoint=False)
    k_vec = k_vec[1:]
    waveno_v = [k_vec[0], k_vec[1], k_vec[-1]]

    svds = np.zeros((3, np.size(nslices_v)))


    for j in range(3):
      symb = -(1j*U_speed*waveno_v[j] + nu*waveno_v[j]**2)
      symb_coarse = symb
#    symb_coarse = -(1.0/dx)*(1.0 - np.exp(-1j*waveno*dx))

      # Solution objects define the problem
      u0      = solution_linear(u0_val, np.array([[symb]],dtype='complex'))
      ucoarse = solution_linear(u0_val, np.array([[symb_coarse]],dtype='complex'))
      for i in range(0,np.size(nslices_v)):
          para = parareal(0.0, float(nslices_v[i]), nslices_v[i], intexact, impeuler, nfine, ncoarse, 0.0, 1, u0)
          svds[j,i] = para.get_max_svd(ucoarse=ucoarse)

    rcParams['figure.figsize'] = 2.5, 2.5
    fs = 8
    fig  = plt.figure()
    plt.plot(nslices_v, svds[0,:], 'b-o', label=(r"$\kappa$=%4.2f" % waveno_v[0]), markersize=fs/2)
    plt.plot(nslices_v, svds[1,:], 'r-s', label=(r"$\kappa$=%4.2f" % waveno_v[1]), markersize=fs/2)
    plt.plot(nslices_v, svds[2,:], 'g-x', label=(r"$\kappa$=%4.2f" % waveno_v[2]), markersize=fs/2)
    plt.plot(nslices_v, 1.0+0.0*nslices_v, 'k--')
    plt.xlim([nslices_v[0], nslices_v[-1]])
    plt.xlabel('Number of processors', fontsize=fs)
    plt.ylabel(r'Maximum singular value $\sigma$', fontsize=fs)
    plt.legend(loc='lower right', fontsize=fs, prop={'size':fs-2}, handlelength=3)
    filename = 'parareal-svd-vs-p.pdf'
    plt.gcf().savefig(filename, bbox_inches='tight')
    call(["pdfcrop", filename, filename])