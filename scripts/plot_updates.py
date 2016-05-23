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
import time

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

    symb      = -(1j*U_speed*k + nu*k**2)
    u0_val    = np.array([[1.0]], dtype='complex')
    u0        = solution_linear(u0_val, np.array([[symb]],dtype='complex'))
    para      = parareal(0.0, Tend, nslices, intexact, impeuler, nfine, ncoarse, 0.0, 0, u0)

    stab_coarse = para.timemesh.slices[0].get_coarse_update_matrix(u0)
    stab_coarse = stab_coarse**nslices
    stab_ex   = np.exp(-1j*U_speed*k*Tend)*np.exp(-nu*k**2*Tend)

    y_start  = np.exp(1j*k*x)
    y_coarse = (stab_coarse[0,0]*y_start).real
    y_ex     = (stab_ex*y_start).real

    fs = 8
#    rcParams['figure.figsize'] = 2.5, 2.5
    rcParams['figure.figsize'] = 7.5, 7.5

    fig = plt.figure()
    y_old = 0.0*x  
    for k in range(3,4):
      stab_para = para.get_parareal_stab_function(k)
      y_new = (stab_para[0,0]*y_start).real
      update = y_new - y_old
      plt.plot(x, update, 'b')
      plt.plot(x, y_ex, 'r')
      plt.title(('k = %2i' % k), fontsize=fs)
      plt.xlim([x[0], x[-1]])
      #plt.ylim([-1, 1])
      plt.show()
      #plt.show(block=False)
      #plt.pause(1)
      #fig.clear()
      y_old = y_new
