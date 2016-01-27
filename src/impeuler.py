from integrator import integrator
from solution import solution

class impeuler(integrator):

  def __init__(self, tstart, tend, nsteps):
    super(impeuler, self).__init__(tstart, tend, nsteps)
    self.order = 1

  def run(self, u0):
    assert isinstance(u0, solution), "Initial value u0 must be an object of type solution"
    for i in range(0,self.nsteps):
      u0.applyM()
      u0.solve(self.dt)
