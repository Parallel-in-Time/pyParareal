import numpy as np
import scipy.linalg as LA
from get_matrix import get_upwind, get_centered, get_diffusion
from matplotlib import pyplot as plt
from subprocess import call

nx = 20
h  = 1.0
dt = 1.0
A_upwind = get_upwind(nx, h)
eigs, dummy = LA.eig(A_upwind.todense())
svds        = LA.svdvals(A_upwind.todense())

A_exp = LA.expm(A_upwind*dt)
eigs_exp, dummy = LA.eig(A_exp.todense())
A_exp_fine = LA.expm(A_upwind*0.1*dt)
eigs_exp_fine, dummy = LA.eig(A_exp_fine.todense())

fig = plt.figure(1)
plt.plot(0*eigs.real, np.linspace(-2.1,2.1,nx), 'k', linewidth=2)
plt.plot(eigs.real, eigs.imag, 'bo', markersize=8, markerfacecolor='b', label='$\lambda(A)$')
plt.plot(eigs_exp.real, eigs_exp.imag, 'ro', markersize=8, markerfacecolor='r', label='$\exp(\lambda(A))$')
plt.plot(eigs_exp_fine.real, eigs_exp_fine.imag, 'rx', markersize=4, markerfacecolor='r', label='$\exp(\lambda(A)/10)$')
circle = plt.Circle((0.0,0.0), 1.0, color='k', fill=False)
fig.gca().add_patch(circle)
plt.xlim([-2.1, 2.1])
plt.ylim([-2.1, 2.1])
plt.xlabel('Re($\lambda$)')
plt.ylabel('Im($\lambda$)')
plt.legend()
filename='upwind-eig.pdf'
plt.gcf().savefig(filename, bbox_inches='tight')
call(["pdfcrop", filename, filename])

D = A_upwind*A_upwind.H - A_upwind.H*A_upwind
print("Normality number: %5.3e" % np.linalg.norm(D.todense()))

'''
fig = plt.figure(1)
plt.plot(range(nx),np.sort(svds), 'bo', label=r'$\sigma$')
plt.plot(range(nx),np.sort(abs(eigs)), 'r+', markersize=12, label=r'$|\lambda|$')
plt.legend()
plt.xlabel('n')
'''

######

A_centered = get_centered(nx, h)
eigs, dummy = LA.eig(A_centered.todense())
svds        = LA.svdvals(A_centered.todense())

A_exp     = LA.expm(A_centered*dt)
eigs_exp, dummy = LA.eig(A_exp.todense())
A_exp_fine = LA.expm(A_centered*0.1*dt)
eigs_exp_fine, dummy = LA.eig(A_exp_fine.todense())

D = A_centered*A_centered.H - A_centered.H*A_centered
print("Normality number: %5.3e" % np.linalg.norm(D.todense()))

######

fig = plt.figure(2)
plt.plot(0*eigs.real, np.linspace(-2.1,2.1,nx), 'k', linewidth=2)
plt.plot(eigs.real, eigs.imag, 'bo', markersize=8, markerfacecolor='b', label='$\lambda(A)$')
plt.plot(eigs_exp.real, eigs_exp.imag, 'ro', markersize=8, markerfacecolor='r', label='$\exp(\lambda(A))$')
plt.plot(eigs_exp_fine.real, eigs_exp_fine.imag, 'rx', markersize=4, markerfacecolor='r', label='$\exp(\lambda(A)/10)$')
circle = plt.Circle((0.0,0.0), 1.0, color='k', fill=False)
fig.gca().add_patch(circle)
plt.xlim([-2.1, 2.1])
plt.ylim([-2.1, 2.1])
plt.xlabel('Re($\lambda$)')
plt.ylabel('Im($\lambda$)')
plt.legend()
filename='centered-eig.pdf'
plt.gcf().savefig(filename, bbox_inches='tight')
call(["pdfcrop", filename, filename])
#plt.show()

'''
fig = plt.figure(3)
plt.plot(range(nx),np.sort(svds), 'bo', label=r'$\sigma$')
plt.plot(range(nx),np.sort(abs(eigs)), 'r+', markersize=12, label=r'$|\lambda|$')
plt.legend()
plt.xlabel('n')
#plt.show()
filename='upwind-svdeig.pdf'
plt.gcf().savefig(filename, bbox_inches='tight')
#call(["pdfcrop", filename, filename])
'''
