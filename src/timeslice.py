from integrator import integrator
from solution import solution
import numpy as np
import copy

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

  def update_fine(self):
    assert hasattr(self, 'sol_start'), "Timeslice object does not have attribute sol_start - may be function set_sol_start was never executed"   
    self.sol_fine = copy.deepcopy(self.sol_start)
    self.int_fine.run(self.sol_fine)

  def update_coarse(self):
    assert hasattr(self, 'sol_start'), "Timeslice object does not have attribute sol_start - may be function set_sol_start was never executed"
    self.sol_coarse = copy.deepcopy(self.sol_start)
    ### RESTRICT TO COARSE MESH
    
    # integrator does not require information about the size of the problem, this is stored in the solution object
    self.int_coarse.run(self.sol_coarse)
    ### INTERPOLATE RESULT TO FINE MESH AND STORE IN self.sol_coarse

  #
  # SET functions
  #

  def set_sol_start(self, sol):
    assert isinstance(sol, solution), "Parameter sol has to be of type solution"
    self.sol_start = sol

  def set_sol_end(self, sol):
    assert isinstance(sol, solution), "Parameter sol has to be of type solution"
    self.sol_end = sol

  def increase_iter(self):
    self.iteration += 1

  def set_residual(self):
    assert hasattr(self, 'sol_fine'), "Timeslice object does not have attribute sol_fine - may be function update_fine was never executed"
    assert hasattr(self, 'sol_end'), "Timeslice object does not have attribute sol_end - it has to be assigned using set_sol_end"
    # compute || F(y_n-1) - y_n ||
    res = self.sol_fine
    res.axpy(-1.0, self.sol_end)
    self.residual = res.norm()
    return self.residual

  #
  # GET functions
  #

  # For linear problems, returns a matrix that corresponds to running the fine method
  def get_fine_update_matrix(self, sol):
    return self.int_fine.get_update_matrix(sol)

  # For linear problems, returns a matrix that corresponds to running the coarse method
  def get_coarse_update_matrix(self, sol):
    ### GET SMALL COARSE LEVEL MATRIX, THEN APPLY RESTRICTION AND INTERPOLATION MATRIX TO BRING IT TO SIZE COMPATIBLE WITH FINE MESH
    return self.int_coarse.get_update_matrix(sol)

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

  def get_sol_end(self):
    assert hasattr(self, 'sol_end'), "Timeslice object does not have attribute sol_end - may be function set_sol_end was never executed"
    return self.sol_end

  def get_residual(self):
    self.set_residual()
    return self.residual

  #
  # IS functions
  #

  def is_converged(self):
    # update residual
    self.set_residual()
    if ( (self.get_residual()<self.tolerance) or (self.iteration>=self.iter_max) ):
      return True
    else:
      return False
