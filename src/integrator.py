from solution import solution
import numpy as np

class integrator(object):

  def __init__(self, tstart, tend, nsteps):
    assert tstart<tend, "tstart must be smaller than tend"
    assert (isinstance(nsteps, (np.integer, int)) and nsteps>0), "nsteps must be a positive integer"
    self.tstart = tstart
    self.tend   = tend
    self.nsteps = nsteps
    self.dt     = (tend - tstart)/float(nsteps)

  # Run integrator from tstart to tend using nstep many steps
  def run(self, u0):
    assert isinstance(u0, solution), "Initial value u0 must be an object of type solution"
    raise NotImplementedError("Function run in generic integrator not implemented: needs to be overloaded in derived class")
