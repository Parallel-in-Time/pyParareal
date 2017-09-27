import sys
sys.path.append('../src')

from parareal import parareal
from impeuler import impeuler
from intexact import intexact
from solution_linear import solution_linear
import numpy as np
#from scipy.sparse import linalg
import math

from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from subprocess import call
import sympy

if __name__ == "__main__":


    Tend    = 32.0
    nslices = 32
    U_speed = 1.0
    nu      = 0.0
    ncoarse = 400
    nfine   = 1
    niter   = 1
    dx      = 1.0
    Nsamples = 30
    k_vec = np.pi
    u0_val = np.array([[1.0]], dtype='complex')
    
    symb = -(1j*(U_speed*(k_vec)) + nu*(k_vec)**2)
    u0   = solution_linear(u0_val, np.array([[symb]],dtype='complex'))
    para = parareal(0.0, Tend, nslices, intexact, impeuler, nfine, ncoarse, 0.0, niter, u0)
    dt = para.timemesh.tend - para.timemesh.tstart

    stab_coarse = para.timemesh.slices[0].get_coarse_update_matrix(u0)
    stab_coarse = stab_coarse**nslices
    
    stab_fine   = para.timemesh.slices[0].get_fine_update_matrix(u0)
    stab_fine   = stab_fine**nslices
    
    u_ex = np.exp(symb*Tend)

    err_c = abs(stab_coarse[0,0] - u_ex)
    print ("Coarse error: %5.3e" % err_c)
    err_f = abs(stab_fine[0,0] - u_ex)
    print ("Fine error: %5.3e" % err_f)
