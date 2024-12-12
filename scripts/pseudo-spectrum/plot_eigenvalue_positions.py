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
problem = 1
if problem==1:
    A    = get_upwind(nx, h)
    filename='upwind-eig.pdf'
elif problem==2:
    A    = get_centered(nx, h)
    filename='centered-eig.pdf'
else:
    sys.exit("Problem needs to be 1 or 2")
eigs, dummy = LA.eig(A.todense())
svds        = LA.svdvals(A.todense())

A_exp                = LA.expm(A*dt)
eigs_exp, dummy      = LA.eig(A_exp.todense())

rcParams['figure.figsize'] = 2.5, 2.5
fs = 8
ms = 4
fig = plt.figure(1)
plt.plot(0*eigs.real, np.linspace(-2.1,2.1,nx), 'k', linewidth=2)
plt.plot(eigs.real, eigs.imag, 'bo', markersize=ms, markerfacecolor='b', label='$\lambda(A)$')
plt.plot(eigs_exp.real, eigs_exp.imag, 'r+', markersize=ms, markerfacecolor='r', label='$\exp(\lambda(A))$')
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
