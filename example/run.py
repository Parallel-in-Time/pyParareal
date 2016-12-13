# Parts of this code are taken from the PseudoSpectralPython tutorial.
#
# Author: D. Ketcheson
# https://github.com/ketch/PseudoSpectralPython/blob/master/PSPython_01-linear-PDEs.ipynb
#
# The MIT License (MIT)
#
# Copyright (c) 2015 David Ketcheson
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys
sys.path.append('../src')
import numpy as np
import scipy.sparse as sp
import scipy.linalg as la

from parareal import parareal
from impeuler import impeuler
from intexact import intexact
from trapezoidal import trapezoidal
from special_integrator import special_integrator
from solution_linear import solution_linear

import matplotlib.pyplot as plt
from subprocess import call
from pylab import rcParams

from joblib import Parallel, delayed

# advection speed
uadv = 1.0

# diffusivity parameter
nu = 0.0

# Spatial grid
m = 64  # Number of grid points in space
L = 4.0  # Width of spatial domain

# Grid points
x = np.linspace(0, L, m, endpoint=False)
# Grid spacing
dx = x[1]-x[0]

# Temporal grid
tmax  = 16  # Final time
#N    = 100  # number grid points in time
#dt   = tmax/float(N)
nproc = int(tmax)

Kiter_v = [5, 10, 15]

xi = np.fft.fftfreq(m)*m*2*np.pi/L  # Wavenumber "grid"
# (this is the order in which numpy's FFT gives the frequencies)

# define spatial operator matrix
D = -1j*xi*uadv - nu*xi**2

# Initial data
sig   = 1.0
u     = np.exp(-(x-0.5*L)**2/sig**2)
uhat0 = np.fft.fft(u)

yend = np.zeros((3,m), dtype='complex')

### Because the exact integrator can only deal with scalar problems, we will need to solve
### to initial value problem for each wave number independently
### ...we use the JobLib module to speed up the computational
def run_parareal(uhat, D, k):
  sol = solution_linear(np.asarray([[uhat]]), np.asarray([[D]]))
  para = parareal(tstart=0.0, tend=tmax, nslices=nproc, fine=intexact, coarse=impeuler, nsteps_fine=1, nsteps_coarse=2, tolerance=0.0, iter_max=k, u0 = sol)
  para.run()
  temp = para.get_last_end_value()
  return temp.y[0,0]

for k in range(0,3):
  temp = Parallel(n_jobs=2, verbose=5)(delayed(run_parareal)(uhat0[n],D[n], Kiter_v[k]) for n in range(m))
  yend[k,:] = temp

rcParams['figure.figsize'] = 2.5, 2.5
fs = 8
fig1 = plt.figure()
plt.plot(x, np.fft.ifft(yend[0,:]).real,  '-s', color='r', linewidth=1.5, label='Parareal k='+str(Kiter_v[0]), markevery=(1,6), mew=1.0, markersize=fs/2)
plt.plot(x, np.fft.ifft(yend[1,:]).real,  '-d', color='r', linewidth=1.5, label='Parareal k='+str(Kiter_v[1]), markevery=(3,6), mew=1.0, markersize=fs/2)
plt.plot(x, np.fft.ifft(yend[2,:]).real,  '-x', color='r', linewidth=1.5, label='Parareal k='+str(Kiter_v[2]), markevery=(5,6), mew=1.0, markersize=fs/2)
plt.plot(x, np.fft.ifft(uhat0).real, '--', color='k', linewidth=1.5, label='Exact')
plt.xlim([x[0], x[-1]])
plt.ylim([-.2, 1.4])
plt.xlabel('x', fontsize=fs, labelpad=0.25)
plt.ylabel('u', fontsize=fs, labelpad=0.5)
fig1.gca().tick_params(axis='both', labelsize=fs)
plt.legend(loc='upper left', fontsize=fs, prop={'size':fs-2})
#plt.show()
filename = 'parareal-gauss-peak.pdf'
plt.gcf().savefig(filename, bbox_inches='tight')
call(["pdfcrop", filename, filename])

xi = np.fft.fftshift(xi)
xi = xi[m/2:m]/m
uhat0     = np.fft.fftshift(uhat0)
yend[0,:] = np.around(np.fft.fftshift(yend[0,:]), 10)
yend[1,:] = np.around(np.fft.fftshift(yend[1,:]), 10)
yend[2,:] = np.around(np.fft.fftshift(yend[2,:]), 10)

fig2 = plt.figure()
plt.semilogy(xi, np.absolute(uhat0[m/2:m])/m, '--', color='k', linewidth=1.5, label='Exact')
plt.semilogy(xi, np.absolute(yend[0,m/2:m])/m, '-s', color='r', linewidth=1.5, label='Parareal k='+str(Kiter_v[0]), markevery=(1,3), mew=1.0, markersize=fs/2)
plt.semilogy(xi, np.absolute(yend[1,m/2:m])/m, '-d', color='r', linewidth=1.5, label='Parareal k='+str(Kiter_v[1]), markevery=(2,3), mew=1.0, markersize=fs/2)
plt.semilogy(xi, np.absolute(yend[2,m/2:m])/m, '-x', color='r', linewidth=1.5, label='Parareal k='+str(Kiter_v[2]), markevery=(3,3), mew=1.0, markersize=fs/2)
plt.xlabel('Wave number', fontsize=fs, labelpad=0.25)
plt.ylabel(r'abs($\hat{u}$)')
plt.xticks([0.0, 0.2, 0.4, 0.6], fontsize=fs)
plt.yticks([1e0, 1e-3, 1e-6, 1e-9, 1e-12], fontsize=fs)
plt.legend(loc='upper right', fontsize=fs, prop={'size':fs-2})
filename = 'parareal-gauss-peak-spectrum.pdf'
plt.gcf().savefig(filename, bbox_inches='tight')
call(["pdfcrop", filename, filename])