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

  # all_converged return True if all timelsices have max_iter = 0
  def test_allconvergedzeromaxiter(self):
    tm = timemesh(self.tstart, self.tend, self.nslices, impeuler, impeuler, self.nfine, self.ncoarse, 1e-10, 0)
    assert tm.all_converged(), "For iter_max=0, all timeslices should initially be converged and all_converged should return True"

  # all_converged return True if all timelsices have max_iter = 0
  def test_allconvergedinitiallyfalse(self):
    tm = timemesh(self.tstart, self.tend, self.nslices, impeuler, impeuler, self.nfine, self.ncoarse, 1e-10, 5)
    assert not tm.all_converged(), "For iter_max>0, all timeslices should initially not be converged and all_converged should return False"

  # get_fine_matrix is callable
  def test_finematrixcallable(self):
    tm = timemesh(self.tstart, self.tend, self.nslices, impeuler, impeuler, self.nfine, self.ncoarse, 1e-10, 5)
    Mat = tm.get_fine_matrix(self.u0)

  # get_coarse_matrix is callable
  def test_finematrixcallable(self):
    tm = timemesh(self.tstart, self.tend, self.nslices, impeuler, impeuler, self.nfine, self.ncoarse, 1e-10, 5)
    Mat = tm.get_coarse_matrix(self.u0)

  # run_coarse is callable and provides expected output
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

  # run_fine is callable and provides expected output
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
