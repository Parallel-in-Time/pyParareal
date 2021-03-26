from integrator import integrator
from solution import solution
from meshtransfer import meshtransfer

import numpy as np
import copy

class timeslice(object):

  def __init__(self, int_fine, int_coarse, tolerance, iter_max, u0fine, u0coarse):
    assert (isinstance(tolerance, float) and tolerance>=0), "Parameter tolerance must be positive or zero"
    assert (isinstance(iter_max, int) and iter_max>=0), "Parameter iter_max must be a positive integer or zero"
    assert isinstance(int_fine, integrator), "Parameter int_fine has to be an object of type integrator"
    assert isinstance(int_coarse, integrator), "Parameter int_coarse has to be an object of type integrator"    
    assert np.isclose( int_fine.tstart, int_coarse.tstart, rtol = 1e-10, atol=1e-12 ), "Values tstart in coarse and fine integrator must be identical"
    assert np.isclose( int_fine.tend, int_coarse.tend, rtol = 1e-10, atol=1e-12 ), "Values tend in coarse and fine integrator must be identical"
    assert isinstance(u0fine, solution), "Parameter u0fine has to be an object of type solution"
    assert isinstance(u0coarse, solution), "Parameter u0coarse has to be an object of type solution"
    self.int_fine    = int_fine
    self.int_coarse  = int_coarse
    self.tolerance   = tolerance
    self.iter_max    = iter_max
    self.iteration   = 0
    # Note: self.coarse_temp is the only object that will have size ndof_c - all other solutions, even self.sol_coarse, have size ndof_f
    # Note also: the problem definition is fixed by these arguments; when functions later accept solution objects, this is only for the values
    self.coarse_temp = copy.deepcopy(u0coarse)
    self.sol_start   = copy.deepcopy(u0fine)
    self.sol_fine    = copy.deepcopy(u0fine)
    self.sol_coarse  = copy.deepcopy(u0fine)
    self.sol_end     = copy.deepcopy(u0fine)
    self.ndof_f      = u0fine.ndof
    self.ndof_c      = u0coarse.ndof
    self.meshtransfer = meshtransfer(self.ndof_f, self.ndof_c)

  def update_fine(self):
    # copy starting value to sol_fine and then overwrite sol_fine with result of fine integrator
    self.sol_fine.y = copy.deepcopy(self.sol_start.y)
    self.int_fine.run(self.sol_fine)

  def update_coarse(self):
    # restrict the starting value and write the restricted solution into coarse_temp; if no spatial coarsening is used, the meshtransfer operators are just the identity
    self.meshtransfer.restrict(self.sol_start, self.coarse_temp)
    # run the coarse propagator on coarse_temp
    self.int_coarse.run(self.coarse_temp)
    # then interpolate back the result to the full size sol_coarse solution; not that sol_coarse has size ndof_f
    self.meshtransfer.interpolate(self.sol_coarse, self.coarse_temp)
        
  #
  # SET functions
  #

  def set_sol_start(self, sol):
    assert isinstance(sol, solution), "Parameter sol has to be of type solution"
    assert type(sol)==type(self.sol_start), "Parameter sol must have the same type as the argument u0fine provided to the constructor"
    assert sol.ndof==self.ndof_f, "Argument sol must have same number of DoF as argument u0fine given to the constructor"
    # NOTE: doing this properly would require to add a copy function to the solution class
    self.sol_start.y = copy.deepcopy(sol.y)
    # For later use, also create the attribute sol_coarse - the values in it will be overwritten when update_coarse is called
    self.sol_coarse.y = copy.deepcopy(self.sol_start.y)
        
  def set_sol_end(self, sol):
    assert isinstance(sol, solution), "Parameter sol has to be of type solution"
    assert type(sol)==type(self.sol_end), "Parameter sol must have the same type as the argument u0fine provided to the constructor"
    self.sol_end = sol

  def increase_iter(self):
    self.iteration += 1

  def set_residual(self):
    # compute || F(y_n-1) - y_n ||
    res = self.sol_fine
    res.axpy(-1.0, self.sol_end)
    self.residual = res.norm()
    return self.residual

  #
  # GET functions
  #

  # For linear problems, returns a matrix that corresponds to running the fine method;
  # if
  def get_fine_update_matrix(self, sol):
    return self.int_fine.get_update_matrix(sol)

  # For linear problems, returns a matrix that corresponds to running the coarse method
  def get_coarse_update_matrix(self, sol):
    G = self.int_coarse.get_update_matrix(sol)
    return self.meshtransfer.Imat@(G@self.meshtransfer.Rmat)

  def get_tstart(self):
    return self.int_fine.tstart

  def get_tend(self):
    return self.int_fine.tend

  def get_sol_fine(self):
    return self.sol_fine

  def get_sol_coarse(self):
    return self.sol_coarse

  def get_sol_end(self):
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
