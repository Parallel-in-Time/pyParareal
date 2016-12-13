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

from joblib import Parallel, delayed

# advection speed
uadv = 1.0

# diffusivity parameter
nu = 0.0

# Spatial grid
m = 64  # Number of grid points in space
L = 1.0  # Width of spatial domain

# Grid points
x = np.linspace(0, L, m, endpoint=False)
# Grid spacing
dx = x[1]-x[0]

# Temporal grid
tmax  = 16.0  # Final time
#N    = 100  # number grid points in time
#dt   = tmax/float(N)
nproc = int(tmax)
Kiter = 15

xi = np.fft.fftfreq(m)*m*2*np.pi/L  # Wavenumber "grid"
# (this is the order in which numpy's FFT gives the frequencies)

# define spatial operator matrix
D = -1j*xi*uadv - nu*xi**2

# Initial data
sig   = 0.15
u     = np.exp(-(x-0.5*L)**2/sig**2)
#u     = np.sin(2*np.pi*x)
uhat0 = np.fft.fft(u)

yend = np.zeros(m, dtype='complex')

### Because the exact integrator can only deal with scalar problems, we will need to solve
### to initial value problem for each wave number independently
### ...we use the JobLib module to speed up the computational
def run_parareal(uhat, D):
  sol = solution_linear(np.asarray([[uhat]]), np.asarray([[D]]))
  para = parareal(tstart=0.0, tend=tmax, nslices=nproc, fine=intexact, coarse=impeuler, nsteps_fine=1, nsteps_coarse=25, tolerance=0.0, iter_max=Kiter, u0 = sol)
  para.run()
  temp = para.get_last_end_value()
  return temp.y[0,0]

yend = Parallel(n_jobs=2, verbose=5)(delayed(run_parareal)(uhat0[n],D[n]) for n in range(m))

fig1 = plt.figure()
plt.plot(x, np.fft.ifft(uhat0).real, 'g')
plt.plot(x, np.fft.ifft(yend).real, 'b')
plt.xlim([x[0], x[-1]])
plt.show()
