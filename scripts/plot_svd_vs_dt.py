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
    ncoarse_v = range(1,11)
    nfine    = 10
    niter_v  = [5, 10, 15]
    dx       = 1.0
    Nsamples = 60
    u0_val     = np.array([[1.0]], dtype='complex')

    k_vec = np.linspace(0.0, np.pi, Nsamples+1, endpoint=False)
    k_vec = k_vec[1:]
    waveno = k_vec[-1]

    svds = np.zeros((1,np.size(ncoarse_v)))
    dt_v = np.zeros((1,np.size(ncoarse_v)))

    symb = -(1j*U_speed*waveno + nu*waveno**2)
    symb_coarse = symb
#    symb_coarse = -(1.0/dx)*(1.0 - np.exp(-1j*waveno*dx))

    # Solution objects define the problem
    u0      = solution_linear(u0_val, np.array([[symb]],dtype='complex'))
    ucoarse = solution_linear(u0_val, np.array([[symb_coarse]],dtype='complex'))

    for i in range(0,np.size(ncoarse_v)):
      para = parareal(0.0, Tend, nslices, intexact, impeuler, nfine, ncoarse_v[i], 0.0, niter_v[2], u0)
      dt_v[0,i] = Tend/float(ncoarse_v[i]*nslices)
      svds[0,i]         = para.get_max_svd(ucoarse=ucoarse)        

    rcParams['figure.figsize'] = 7.5, 7.5
    fs = 8
    fig  = plt.figure()
    plt.plot(dt_v[0,:], svds[0,:])
    plt.show()
