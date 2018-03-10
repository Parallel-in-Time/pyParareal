import sys
sys.path.append('../src')
import numpy as np

from expeuler import expeuler
from parareal import parareal
from solution_riemann import solution_riemann

import copy
import matplotlib.pyplot as plt

nx     = 200
tend   = 0.2
ncoarse = 1
nfine   = 40
nslices = 32
xaxis = np.linspace(0, 1, nx+1)
xaxis = xaxis[0:nx]
dx = xaxis[1] - xaxis[0]
y = np.sin(2.0*np.pi*xaxis)

u0 = solution_riemann(y, dx)
para = parareal(tstart=0.0, tend=tend, nslices=nslices, fine=expeuler, coarse=expeuler, nsteps_fine=nfine, nsteps_coarse=ncoarse, tolerance=0.0, iter_max=2, u0=u0)
para.run()
para_sol = para.get_last_end_value()

coarse = expeuler(0.0, tend, ncoarse*nslices)
fine = expeuler(0.0, tend, nfine*nslices)
exact = expeuler(0.0, tend, 10*nfine*nslices)

coarse_sol = copy.deepcopy(u0)
coarse.run(coarse_sol)
fine_sol = copy.deepcopy(u0)
fine.run(fine_sol)
exact_sol = copy.deepcopy(u0)
exact.run(exact_sol)


print ("Error coarse: %5.3e" % np.linalg.norm(coarse_sol.y - exact_sol.y, np.inf))
print ("Error fine: %5.3e" % np.linalg.norm(exact_sol.y - fine_sol.y, np.inf))
print ("Defect parareal: %5.3e" % np.linalg.norm(para_sol.y - fine_sol.y, np.inf))

fig = plt.figure()
plt.plot(xaxis, para_sol.y, 'b')
#plt.plot(xaxis, coarse_sol.y, 'r')
plt.plot(xaxis, fine_sol.y, 'g')
plt.show()
