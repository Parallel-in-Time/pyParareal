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
    ncoarse_v = [1, 2, 4, 5, 10, 15, 20]
    nfine    = 20
    dx       = 1.0
    u0_val     = np.array([[1.0]], dtype='complex')

    Nk    = 6
    k_vec = np.linspace(0, np.pi, Nk+1, endpoint=False)
    k_vec = k_vec[1:]
    k_vec = [k_vec[0], k_vec[1], k_vec[-1]]


    svds = np.zeros((3,np.size(ncoarse_v)))
    dt_v = np.zeros((3,np.size(ncoarse_v)))

    for k in range(3):
      if k==0:
        waveno = k_vec[0]
      elif k==1:
        waveno = k_vec[1] 
      else:
        waveno = k_vec[2]
      symb = -(1j*U_speed*waveno + nu*waveno**2)
      symb_coarse = symb
  #    symb_coarse = -(1.0/dx)*(1.0 - np.exp(-1j*waveno*dx))

      # Solution objects define the problem
      u0      = solution_linear(u0_val, np.array([[symb]],dtype='complex'))
      ucoarse = solution_linear(u0_val, np.array([[symb_coarse]],dtype='complex'))

      for i in range(0,np.size(ncoarse_v)):
        para = parareal(0.0, Tend, nslices, intexact, impeuler, nfine, ncoarse_v[i], 0.0, 1, u0)
        dt_v[k,i] = Tend/float(ncoarse_v[i]*nslices)
        svds[k,i] = para.get_max_svd(ucoarse=ucoarse)        

    rcParams['figure.figsize'] = 2.5, 2.5
    fs = 8
    fig  = plt.figure()
    plt.plot(dt_v[0,:], svds[0,:], 'b-o', label=(r"$\kappa$=%4.2f" % k_vec[0]), markersize=fs/2)
    plt.plot(dt_v[1,:], svds[1,:], 'r-s', label=(r"$\kappa$=%4.2f" % k_vec[1]), markersize=fs/2)
    plt.plot(dt_v[2,:], svds[2,:], 'g-x', label=(r"$\kappa$=%4.2f" % k_vec[2]), markersize=fs/2)
    plt.legend(loc='upper left', fontsize=fs, prop={'size':fs-2}, handlelength=3)
    plt.xlabel(r'Coarse time step $\Delta t$', fontsize=fs)
    plt.ylabel(r'Maximum singular value $\sigma$', fontsize=fs)
    filename = 'parareal-sigma-vs-dt.pdf'
    plt.gcf().savefig(filename, bbox_inches='tight')
    call(["pdfcrop", filename, filename])
    plt.show()
