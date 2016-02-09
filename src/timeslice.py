from integrator import integrator
from solution import solution
import numpy as np

class timeslice(object):

  def __init__(self, int_fine, int_coarse, tolerance, iter_max):
    assert (isinstance(tolerance, float) and tolerance>=0), "Parameter tolerance must be positive or zero"
    assert (isinstance(iter_max, int) and iter_max>=0), "Parameter iter_max must be a positive integer or zero"
    assert isinstance(int_fine, integrator), "Parameter int_fine has to be an object of type integrator"
    assert isinstance(int_coarse, integrator), "Parameter int_coarse has to be an object of type integrator"    
    assert np.isclose( int_fine.tstart, int_coarse.tstart, rtol = 1e-10, atol=1e-12 ), "Values tstart in coarse and fine integrator must be identical"
    assert np.isclose( int_fine.tend, int_coarse.tend, rtol = 1e-10, atol=1e-12 ), "Values tend in coarse and fine integrator must be identical"
    self.int_fine   = int_fine
    self.int_coarse = int_coarse
    self.tolerance  = tolerance
    self.iter_max   = iter_max
    self.iteration  = 0
    # Initialize residual such that first check for res < tol always fails
    self.residual   = tolerance + 1.0

  def update_fine(self):
    self.sol_fine = self.sol_start
    self.int_fine.run(self.sol_fine)

  def update_coarse(self):
    self.sol_coarse = self.sol_start
    self.int_coarse.run(self.sol_coarse)

  #
  # GET, SET and IS functions
  #

  def set_sol_start(self, sol):
    assert isinstance(sol, solution), "Parameter sol has to be of type solution"
    self.sol_start = sol

  def get_tstart(self):
    return self.int_fine.tstart

  def get_tend(self):
    return self.int_fine.tend

  def get_sol_fine(self):
    assert hasattr(self, 'sol_fine'), "Timeslice object does not have attribute sol_fine - may be function update_fine was never executed"
    return self.sol_fine

  def get_sol_coarse(self):
    assert hasattr(self, 'sol_coarse'), "Timeslice object does not have attribute sol_coarse - may be function update_coarse was never executed"
    return self.sol_coarse

  def is_converged(self):
    if ( (self.residual<self.tolerance) or (self.iteration>=self.iter_max) ):
      return True
    else:
      return False
