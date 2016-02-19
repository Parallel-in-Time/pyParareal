import sys
sys.path.append('../src')

from parareal import parareal
from impeuler import impeuler
from intexact import intexact
from solution_linear import solution_linear
import numpy as np

# Assume that the wave is RIGHTWARD travelling
def findspeed(y0, y1, x, T):
  max_y0 = findmax(y0)
  max_y1 = findmax(y1)
  dist = np.zeros(np.size(max_y0)-1) 
  for i in range(0,np.size(max_y0)-1):
    i0 = max_y0[i]    
    i1 = max_y1[i]
    if x[i0]>x[i1]:
      i1 = max_y1[i+1]
    dist[i] = x[i1] - x[i0]
  return np.average(dist)/T

def findmax(y):
  max_ind = []
  for i in range(1,np.size(y)-1):
    if (y[i-1] < y[i] and y[i] > y[i+1]):
      max_ind.append(i)
  return max_ind

import matplotlib.pyplot as plt

if __name__ == "__main__":

    Nx = 1e5
    x = np.linspace(0,500,Nx+1,endpoint=False)
    x = x[0:Nx]

    Nk    = 4
    k_vec = np.linspace(0, np.pi, Nk+1, endpoint=False)
    k_vec = k_vec[1:]
    # Select a wave number
    k     = k_vec[1]

    Tend    = 8.0    
    nslices = 8
    U_speed = 1.0
    nu      = 0.0
    ncoarse = 1
    nfine   = 1

    err_amp   = np.zeros(nslices)
    err_speed = np.zeros(nslices)
    
    symb      = -(1j*U_speed*k + nu*k**2)
    u0_val    = np.array([[1.0]], dtype='complex')
    u0        = solution_linear(u0_val, np.array([[symb]],dtype='complex'))
    para      = parareal(0.0, Tend, nslices, intexact, impeuler, nfine, ncoarse, 0.0, 0, u0)

    stab_ex   = np.exp(-1j*U_speed*k*Tend)*np.exp(-nu*k**2*Tend)

    stab_coarse   = para.timemesh.slices[0].get_coarse_update_matrix(u0)
    stab_coarse = stab_coarse**nslices

    stab_fine = para.timemesh.slices[0].get_fine_update_matrix(u0)
    stab_fine = stab_fine**nslices

    if abs(stab_fine[0,0]-stab_ex)>1e-14:
      print "WARNING: Fine method is not the exact integrator..."    

    y_start = np.exp(1j*k*x)
    y_ex    = stab_ex*y_start

    y_coarse= stab_coarse[0,0]*y_start
    y_fine  = stab_fine[0,0]*y_start
    s_fine = findspeed(y_start, y_fine, x, Tend)

    for n in range(1,nslices+1):
      stab_para = para.get_parareal_stab_function(n)
      y_para  = stab_para[0,0]*y_start
      err_amp[n-1] = abs(np.max(abs(y_para))-np.max(abs(y_fine)))/np.max(abs(y_fine))
      s = findspeed(y_start.real, y_para.real, x, Tend)
      err_speed[n-1] = abs(s_fine - s)/abs(s_fine)

    fig = plt.figure()
    plt.semilogy(range(1,nslices+1), err_amp, 'bo', markersize=12, label="Amplitude")
    plt.semilogy(range(1,nslices+1), err_speed, 'rs', markersize=12, label="Speed")
    print err_speed

#    fig = plt.figure()
#    plt.plot(x, y_coarse.real, 'b', label='Coarse')
#    plt.plot(x, y_para.real, 'r', label='Parareal k='+str(niter))
#    plt.plot(x, y_ex.real,   'g', label='Exact')
#    plt.legend()
#    plt.ylim([-2, 2])
#    plt.xlim([x[0], x[-1]])
    plt.show()
