import sys
sys.path.append('../src')

from impeuler import impeuler
from timemesh import timemesh
from solution_linear import solution_linear
import unittest
import numpy as np
from scipy.sparse import linalg
from scipy import sparse


class TestTimemesh(unittest.TestCase):

  def setUp(self):
    times        = np.sort( np.random.rand(2) )
    self.tstart  = times[0]
    self.tend    = times[1]
    self.nslices = np.random.randint(1,128)
    steps        = np.sort( np.random.randint(low=1, high=128, size=2) )
    self.ncoarse = steps[0]
    self.nfine   = steps[1]
    self.u0      = solution_linear(np.array([1.0]), np.array([[-1.0]]))

  # timemesh class can be instantiated
  def test_caninstantiate(self):
    tm = timemesh(self.tstart, self.tend, self.nslices, impeuler, impeuler, self.nfine, self.ncoarse, 1e-10, 5)

  # initial value can be set for first time slice
  def test_cansetinitial(self):
    tm = timemesh(self.tstart, self.tend, self.nslices, impeuler, impeuler, self.nfine, self.ncoarse, 1e-10, 5)
    tm.set_initial_value(self.u0)

  # initial value can be set for first time slice
  def test_cansetinitialtimeslice(self):
    u0 = solution_linear(np.array([1.0]), np.array([[-1.0]]))
    tm = timemesh(self.tstart, self.tend, self.nslices, impeuler, impeuler, self.nfine, self.ncoarse, 1e-10, 5)
    tm.set_initial_value(self.u0, 3)

  # fails to set initial value for time slice with too large index
  def test_cansetinitialtimeslice(self):
    tm = timemesh(self.tstart, self.tend, self.nslices, impeuler, impeuler, self.nfine, self.ncoarse, 1e-10, 5)
    with self.assertRaises(AssertionError):
      tm.set_initial_value(self.u0, self.nslices+1)

  # all_converged throws if called directly after initialisation
  def test_allconvergedthrows(self):
    tm = timemesh(self.tstart, self.tend, self.nslices, impeuler, impeuler, self.nfine, self.ncoarse, 1e-10, 0)
    with self.assertRaises(AssertionError):
      tm.all_converged()

  def test_allconvergedzeromaxiter(self):
    tm = timemesh(self.tstart, self.tend, self.nslices, impeuler, impeuler, self.nfine, self.ncoarse, 1e-10, 0)
    # to allow computing residuals, set initial value, run fine and set end value
    sol_end     = solution_linear( np.ones(1), np.array([[-1.0]]) )
    sol_start   = solution_linear(np.zeros(1), np.array([[-1.0]]) )
    for i in range(0,self.nslices):
      tm.set_initial_value(sol_start, i)
      tm.set_end_value(sol_end, i)
      tm.slices[i].update_fine()
    # Since iter_max=0, all_converged should return True even though the residual is larger than 1e-10
    assert tm.get_max_residual()>1e-10, "Maximum residual is smaller than tolerance, even though set up not to be"    
    assert tm.all_converged, "For iter_max=0, all time slices should be considered converged"

  # raising iteration counter leads to convergence
  def test_raiseiterconvergence(self):
    tm = timemesh(self.tstart, self.tend, self.nslices, impeuler, impeuler, self.nfine, self.ncoarse, 1e-10, 2)
    sol_end     = solution_linear( np.ones(1), np.array([[-1.0]]) )
    sol_start   = solution_linear(np.zeros(1), np.array([[-1.0]]) )
    for i in range(0,self.nslices):
      tm.set_initial_value(sol_start, i)
      tm.set_end_value(sol_end, i)
      tm.slices[i].update_fine()
    # since iter_max=2, time slices should not yet have converged
    assert not tm.all_converged(), "All time slices have converged even though initialised not to be"
    # increase iteration counter 
    for i in range(0,self.nslices):
      tm.increase_iter(i)
    # For max_iter=2, timeslices should not yet have converged
    assert not tm.all_converged(), "Raising iteration counter once with iter_max should not lead to convergence"
    # increase iteration counter again
    for i in range(0,self.nslices):
      tm.increase_iter(i)
    assert tm.all_converged(), "Raising iteration counter through increase_iter did not lead to convergence"

  # get_fine_matrix is callable
  def test_finematrixcallable(self):
    tm = timemesh(self.tstart, self.tend, self.nslices, impeuler, impeuler, self.nfine, self.ncoarse, 1e-10, 5)
    Mat = tm.get_fine_matrix(self.u0)

  # get_coarse_matrix is callable
  def test_finematrixcallable(self):
    tm = timemesh(self.tstart, self.tend, self.nslices, impeuler, impeuler, self.nfine, self.ncoarse, 1e-10, 5)
    Mat = tm.get_coarse_matrix(self.u0)

  # run_coarse is callable and provides expected output at very end
  def test_runcoarse(self):
    tm = timemesh(self.tstart, self.tend, self.nslices, impeuler, impeuler, self.nfine, self.ncoarse, 1e-10, 5)
    tm.run_coarse(self.u0)
    u = np.array([1.0])
    Mat = tm.get_coarse_matrix(self.u0)
    b = np.zeros(self.nslices+1)
    b[0] = u[0]
    u = linalg.spsolve(Mat, b)
    err = abs(u[-1] - tm.get_coarse_value(self.nslices-1).y)
    assert err<1e-12, ("run_coarse and successive application of update matrix does not give identical results - error: %5.3e" % err)

  # run_fine is callable and provides expected output at very end
  def test_runfine(self):
    tm = timemesh(self.tstart, self.tend, self.nslices, impeuler, impeuler, self.nfine, self.ncoarse, 1e-10, 5)
    tm.run_fine(self.u0)
    u = np.array([1.0])
    Mat = tm.get_fine_matrix(self.u0)
    b = np.zeros(self.nslices+1)
    b[0] = u[0]
    u = linalg.spsolve(Mat, b)
    err = abs(u[-1] - tm.get_fine_value(self.nslices-1).y)
    assert err<1e-12, ("run_fine and successive application of update matrix does not give identical results - error: %5.3e" % err)

  # run_coarse provides expected intermediate values
  def test_runcoarseintermediate(self):
    tm = timemesh(self.tstart, self.tend, self.nslices, impeuler, impeuler, self.nfine, self.ncoarse, 1e-10, 5)
    tm.run_coarse(self.u0)
    u = np.array([1.0])
    Mat = tm.slices[0].int_coarse.get_update_matrix(self.u0)
    b = self.u0.y
    for i in range(0,3):
      # matrix update to end of slice
      b = Mat.dot(b)
      err = np.linalg.norm( tm.get_coarse_value(i).y - b, np.inf )
      assert err<2e-12, ("Successive application of update matrix does not reproduce intermediata coarse values generated by run_coarse. Error: %5.3e in slice %2i" % (err, i))

  # run_fine provides expected intermediate values
  def test_runfineintermediate(self):
    tm = timemesh(self.tstart, self.tend, self.nslices, impeuler, impeuler, self.nfine, self.ncoarse, 1e-10, 5)
    tm.run_fine(self.u0)
    u = np.array([1.0])
    Mat = tm.slices[0].int_fine.get_update_matrix(self.u0)
    b = self.u0.y
    for i in range(0,3):
      # matrix update to end of slice
      b = Mat.dot(b)
      err = np.linalg.norm( tm.get_fine_value(i).y - b, np.inf )
      assert err<2e-12, ("Successive application of update matrix does not reproduce intermediata coarse values generated by run_fine. Error: %5.3e in slice %2i" % (err, i))
