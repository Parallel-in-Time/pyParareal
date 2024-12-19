import sys
sys.path.append('../../src')

import numpy as np
import scipy.linalg as LA
from get_matrix import get_upwind, get_centered, get_diffusion
from matplotlib import pyplot as plt
from subprocess import call
from pylab import rcParams

nx = 32
h  = 1.0
dt = 1.0

try:
  figure      =  int(sys.argv[1]) # 5 generates figure_5, 6 generates figure_6
except:
  print("No or wrong command line argument provided, creating figure 1. Use 1, 2, 3 or 4 as command line argument.")
  figure = 1


if figure==1:
    A    = get_upwind(nx, h)
    filename='figure_1.pdf'
elif figure==2:
    A    = get_centered(nx, h)
    filename='figure_2.pdf'
elif figure==3:
    A    = get_upwind(nx, h)
    filename='figure_3.pdf'
elif figure==4:
    A   = get_centered(nx, h)
    filename='figure_4.pdf'    
else:
    sys.exit("Figure needs to be 1, 2, 3 or 4")
    
eigs_space, dummy = LA.eig(A.todense())
svds        = LA.svdvals(A.todense())

if figure==1 or figure==2:
  A_exp                = LA.expm(A*dt)
  eigs_time, dummy      = LA.eig(A_exp.todense())
  mylabel = '$\exp(\Delta t A)$'
else:
  A_ie = LA.inv(np.eye(nx) - dt*A)
  eigs_time, dummy = LA.eig(A_ie)
  mylabel = '$(I - \Delta t A)^{-1}$'

rcParams['figure.figsize'] = 2.5, 2.5
fs = 8
ms = 4
fig = plt.figure(1)
plt.plot(0*eigs_space.real, np.linspace(-2.1,2.1,nx), 'k', linewidth=2)
plt.plot(eigs_space.real, eigs_space.imag, 'bo', markersize=ms, markerfacecolor='b', label='$A$')
plt.plot(eigs_time.real, eigs_time.imag, 'rs', markersize=ms, markerfacecolor='r', label=mylabel)
circle = plt.Circle((0.0,0.0), 1.0, color='k', fill=False)
fig.gca().add_patch(circle)
plt.xlim([-2.1, 2.1])
plt.ylim([-2.1, 2.1])
plt.xlabel('Re($\lambda$)', fontsize=fs)
plt.ylabel('Im($\lambda$)', fontsize=fs)
plt.legend(loc='lower right', fontsize=fs, prop={'size':fs-2}, handlelength=3)
plt.gcf().savefig(filename, bbox_inches='tight')
call(["pdfcrop", filename, filename])

D = A*A.H - A.H*A
print("Normality number (should be zero): %5.3e" % np.linalg.norm(D.todense()))
