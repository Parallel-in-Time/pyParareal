import numpy as np
from matplotlib import pyplot as plt
from subprocess import call
from pylab import rcParams

N     = 25
k_vec = np.linspace(0, np.pi, N+1)
k_vec = k_vec[1:]

phase = np.zeros((3,N))
ampfa = np.zeros((3,N))

dx = 1.0
dt = 1.0

# Select finite difference stencil
# stencil = 0 : first order upwind
# stencil = 1 : second order centred
stencil = 1

for k in range(0,np.size(k_vec)):
  
  # exact values
  phase[2,k] = 1.0
  ampfa[2,k] = 1.0
  
  # Implicit Euler only
  delta   = -1j*k_vec[k]
  stab_ie = 1.0/(1.0 - dt*delta)
  omega   = 1j*np.log(stab_ie)/dt
  
  phase[1,k] = omega.real/k_vec[k]
  ampfa[1,k] = np.exp(omega.imag)
  
  if stencil==0:
    delta_disc = -(1.0 - np.exp(-1j*k_vec[k]*dx))/dx
  
  elif stencil==1:
    delta_disc = -1j*np.sin(k_vec[k]*dx)/dx
  
  else:
    raise Exception("Not implemented")

  stab_disc  = 1.0/(1.0 - dt*delta_disc)
  omega_disc = 1j*np.log(stab_disc)/dt

  phase[0,k] = omega_disc.real/k_vec[k]
  ampfa[0,k] = np.exp(omega_disc.imag)

rcParams['figure.figsize'] = 2.5, 2.5
fs = 8

fig = plt.figure()
plt.plot(k_vec, phase[2,:], '-g', linewidth=1.5, label='Continuous', markevery=(1,5), markersize=fs/2)
plt.plot(k_vec, phase[1,:], 'k-o', linewidth=1.5, label='Implicit Euler', markevery=(3,5), markersize=fs/2)
plt.plot(k_vec, phase[0,:], 'b-s', linewidth=1.5, label='Implicit Euler + FD', markevery=(5,5), markersize=fs/2)
plt.xlim([k_vec[0], k_vec[-1]])
plt.ylim([0.0, 1.1])
plt.xlabel('Wave number', fontsize=fs, labelpad=0.25)
plt.ylabel('Phase speed', fontsize=fs, labelpad=0.5)
fig.gca().tick_params(axis='both', labelsize=fs)
plt.legend(loc='lower left', fontsize=fs, prop={'size':fs-2})
plt.xticks([0, 1, 2, 3], fontsize=fs)
filename = 'ieuler-dispersion-phase.pdf'
plt.gcf().savefig(filename, bbox_inches='tight')
call(["pdfcrop", filename, filename])

fig = plt.figure()
plt.plot(k_vec, ampfa[2,:], 'g-', linewidth=1.5, label='Continuous', markevery=(1,5), markersize=fs/2)
plt.plot(k_vec, ampfa[1,:], 'k-o', linewidth=1.5, label='Implicit Euler', markevery=(3,5), markersize=fs/2)
plt.plot(k_vec, ampfa[0,:], 'b-s', linewidth=1.5, label='Implicit Euler + FD', markevery=(5,5), markersize=fs/2)
plt.xlim([k_vec[0], k_vec[-1]])
plt.ylim([0.0, 1.1])
plt.xlabel('Wave number', fontsize=fs, labelpad=0.25)
plt.ylabel('Phase speed', fontsize=fs, labelpad=0.5)
fig.gca().tick_params(axis='both', labelsize=fs)
plt.legend(loc='lower left', fontsize=fs, prop={'size':fs-2})
plt.xticks([0, 1, 2, 3], fontsize=fs)
filename = 'ieuler-dispersion-ampf.pdf'
plt.gcf().savefig(filename, bbox_inches='tight')
call(["pdfcrop", filename, filename])