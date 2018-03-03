import sys
sys.path.append('../src')
import numpy as np

from expeuler import expeuler
from solution_riemann import solution_riemann

import matplotlib.pyplot as plt

nx     = 50
tend   = 0.2
nsteps = 150
xaxis = np.linspace(0, 1, nx+1)
xaxis = xaxis[0:nx]
dx = xaxis[1] - xaxis[0]
y = np.sin(2.0*np.pi*xaxis)

sol = solution_riemann(y, dx)
expe = expeuler(0.0, tend, nsteps)
expe.run(sol)

fig = plt.figure()
plt.plot(xaxis, sol.y)
plt.show()
