import sys
sys.path.append('./src')

from timeslice import timeslice
from integrator import integrator
from integrator_dedalus import integrator_dedalus
from impeuler import impeuler
from solution_linear import solution_linear
from solution_dedalus import solution_dedalus
import pytest
import numpy as np
import math

class TestClass:

  def setUp(self):
    self.t = np.sort( np.random.rand(2) )
    self.nsteps_c        = 1+np.random.randint(16)
    self.nsteps_f        = 1+np.random.randint(32)
    self.int_coarse = impeuler(self.t[0], self.t[1], self.nsteps_c)
    self.int_fine   = impeuler(self.t[0], self.t[1], self.nsteps_f)
    self.ndofs           = [np.random.randint(4,25), np.random.randint(4,25)]
    self.ndof_c          = np.min(self.ndofs)
    self.ndof_f          = np.max(self.ndofs)
    self.u0_f            = np.random.rand(self.ndof_f)
    self.A_f             = np.random.rand(self.ndof_f, self.ndof_f)
    self.u0fine          = solution_linear(self.u0_f, self.A_f)
    self.u0_c            = np.random.rand(self.ndof_c)
    self.A_c             = np.random.rand(self.ndof_c, self.ndof_c)
    self.u0coarse        = solution_linear(self.u0_f, self.A_f)
    # TODO: randomise tolerance and iteration number

  # Timeslice can be instantiated
  def test_can_instantiate(self):
    self.setUp()
    ts = timeslice(self.int_fine, self.int_coarse, 1e-10, 5, self.u0fine, self.u0coarse)


  # Timeslice can be instantiated with solution objects of type solution_dedalus
  def test_can_instantiate_with_dedalus(self):
    self.setUp()
    # Make sure number of degrees of freedom is even
    ndof_f = math.ceil(self.ndof_f/2)*2
    ndof_c = math.ceil(self.ndof_c/2)*2
    u0fine = solution_dedalus(np.zeros(ndof_f), ndof_f)
    u0coarse = solution_dedalus(np.zeros(ndof_c), ndof_c)
    int_coarse = integrator_dedalus(self.t[0], self.t[1], self.nsteps_c)
    int_fine   = integrator_dedalus(self.t[0], self.t[1], self.nsteps_f)  
    ts = timeslice(int_fine, int_coarse, 1e-10, 5, u0fine, u0coarse)
    
  # Negative tolerance throws exception
  def test_fails_negative_tol(self):
    self.setUp()
    with pytest.raises(AssertionError):
      ts = timeslice(self.int_fine, self.int_coarse, -1e-5, 5, self.u0fine, self.u0coarse)

  # Non-float tolerance throws exception
  def test_fails_integer_tol(self):
    self.setUp()
    with pytest.raises(AssertionError):
      ts = timeslice(self.int_fine, self.int_coarse, 1, 5, self.u0fine, self.u0coarse)

  # Non-int iter_max raises exception
  def test_fails_float_itermax(self):
    self.setUp()
    with pytest.raises(AssertionError):
      ts = timeslice(self.int_fine, self.int_coarse, 1e-10, 2.5, self.u0fine, self.u0coarse)

  # Negative iter_max raises exception
  def test_fails_negative_itermax(self):
    self.setUp()
    with pytest.raises(AssertionError):
      ts = timeslice(self.int_fine, self.int_coarse, 1e-10, -5, self.u0fine, self.u0coarse)

  # Different values for tstart in fine and coarse integrator raise exception
  def test_fails_different_tstart(self):
    self.setUp()
    int_c = integrator(1e-10+self.int_coarse.tstart, self.int_coarse.tend, self.int_coarse.nsteps)
    with pytest.raises(AssertionError):
      ts = timeslice(self.int_fine, int_c, 1e-10, 5, self.u0fine, self.u0coarse)

  # Different values for tend in fine and coarse integrator raise exception
  def test_fails_different_tend(self):
    self.setUp()
    int_c = integrator(self.int_coarse.tstart, 1e-8+self.int_coarse.tend, self.int_coarse.nsteps)
    with pytest.raises(AssertionError):
      ts = timeslice(self.int_fine, int_c, 1e-10, 5, self.u0fine, self.u0fine)

  # After running fine integrator and setting sol_end to the same value, is_converged returns True
  def test_is_converged(self):
    self.setUp()
    ts = timeslice(self.int_fine, self.int_coarse, 1e-14, 1+np.random.randint(5), self.u0fine, self.u0coarse)
    assert not ts.is_converged(), "After initialisation, timeslice should not be converged without further actions"
    sol = solution_linear(np.random.rand(self.ndof_f), np.random.rand(self.ndof_f, self.ndof_f))
    ts.set_sol_start(sol)
    ts.update_fine()
    ts.set_sol_end(ts.get_sol_fine())
    assert ts.is_converged(), "After running F and setting sol_end to the result, the residual should be zero and the time slice converged"  

  # get_tstart returns correct value
  def test_get_tstart(self):
    self.setUp()
    ts = timeslice(self.int_fine, self.int_coarse, 1e-10, 5, self.u0fine, self.u0coarse)
    assert abs(ts.get_tstart() - ts.int_fine.tstart)==0, "get_start returned wrong value"

  # get_tend returns correct value
  def test_get_tend(self):
    self.setUp()
    ts = timeslice(self.int_fine, self.int_coarse, 1e-10, 5, self.u0fine, self.u0coarse)
    assert abs(ts.get_tend() - ts.int_fine.tend)==0, "get_start returned wrong value"

  # set_sol_start with non-solution objects throws exception
  def test_sol_fine_no_solution_throws(self):
    self.setUp()
    ts = timeslice(self.int_fine, self.int_coarse, 1e-10, 5, self.u0fine, self.u0coarse)
    with pytest.raises(AssertionError):
      ts.set_sol_start(-1)

  # update_fine runs and returns value equal to what matrix provides
  def test_fine_equals_matrix(self):
    self.setUp()
    ts = timeslice(self.int_fine, self.int_coarse, 1e-10, 5, self.u0fine, self.u0coarse)
    ts.update_fine()
    sol_ts = ts.get_sol_fine()
    assert isinstance(sol_ts, solution_linear), "After running update_fine, object returned by get_sol_fine is of wrong type"
    Fmat = ts.get_fine_update_matrix(self.u0fine)
    fine = solution_linear( Fmat@self.u0_f, self.A_f)
    fine.axpy(-1.0, sol_ts)
    assert fine.norm()<1e-10, ("Solution generated with get_fine_update_matrix does not match the one generated by update_fine: defect = %5.3e" %  fine.norm())
    
  # update_fine runs and returns value equal to what matrix provides
  def test_fine_equals_matrix_with_dedalus_option(self):
    self.setUp()
    # Make sure number of degrees of freedom is even
    ndof_f = math.ceil(self.ndof_f/2)*2
    ndof_c = math.ceil(self.ndof_c/2)*2    
    u0fine = solution_dedalus(np.zeros(ndof_f), ndof_f)
    u0coarse = solution_dedalus(np.zeros(ndof_c), ndof_c)
    int_coarse = integrator_dedalus(self.t[0], self.t[1], self.nsteps_c)
    int_fine   = integrator_dedalus(self.t[0], self.t[1], self.nsteps_f)  
    ts = timeslice(int_fine, int_coarse, 1e-10, 5, u0fine, u0coarse)
    
    ### Code below will not work, because it is not yet possible to actually run Dedalus integrators. They can only provide matrices.
    
    #ts.update_fine()
    #sol_ts = ts.get_sol_fine()
    #assert isinstance(sol_ts, solution_linear), "After running update_fine, object returned by get_sol_fine is of wrong type"
    #Fmat = ts.get_fine_update_matrix(u0fine)
    #fine = solution_linear( Fmat@self.u0_f, self.A_f)
    #fine.axpy(-1.0, sol_ts)
    #assert fine.norm()<1e-10, ("Solution generated with get_fine_update_matrix does not match the one generated by update_fine: defect = %5.3e" %  fine.norm())    

  # update_coarse runs and returns value equal to what matrix provides
  def test_can_run_coarse(self):
    self.setUp()
    ts = timeslice(self.int_fine, self.int_coarse, 1e-10, 5, self.u0fine, self.u0coarse)
    ts.update_coarse()
    sol_ts = ts.get_sol_coarse()
    assert isinstance(sol_ts, solution_linear), "After running update_coarse, object returned by get_sol_coarse is of wrong type"
    Cmat = ts.get_coarse_update_matrix(self.u0fine)
    coarse = solution_linear( Cmat@self.u0_f, self.A_f)
    coarse.axpy(-1.0, sol_ts)
    assert coarse.norm()<1e-10, ("Solution generated with get_coarse_update_matrix does not match the one generated by update_coarse: defect = %5.3e" % coarse.norm())

  # If the length or type of the solution given to run_coarse is different to what was given to the constructor, throw and exception
  def test_coarse_fine_throws_if_solution_different(self):
    pass
