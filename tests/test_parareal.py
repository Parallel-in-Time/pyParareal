import sys
sys.path.append('../src')

from parareal import parareal
from timemesh import timemesh
from impeuler import impeuler
from solution_linear import solution_linear
import unittest
import numpy as np

class TestParareal(unittest.TestCase):

  def setUp(self):
    times        = np.sort( np.random.rand(2) )
    self.tstart  = times[0]
    self.tend    = times[1]
    self.nslices = np.random.randint(1,128)
    steps        = np.sort( np.random.randint(low=1, high=128, size=2) )
    self.ncoarse = steps[0]
    self.nfine   = steps[1]
    self.u0      = solution_linear(np.array([1.0]), np.array([[-1.0]]))

  # Can instantiate object of type parareal
  def test_caninstantiate(self):
    para = parareal(self.tstart, self.tend, self.nslices, impeuler, impeuler, self.nfine, self.ncoarse, 1e-10, 5, self.u0)

  # Can execute run function
  def test_canrun(self):
    para = parareal(self.tstart, self.tend, self.nslices, impeuler, impeuler, self.nfine, self.ncoarse, 1e-10, 5, self.u0)
    para.run()  

  # Test matrix Parareal
  def test_pararealmatrix(self):
    para = parareal(self.tstart, self.tend, self.nslices, impeuler, impeuler, self.nfine, self.ncoarse, 1e-10, 5, self.u0)
    Pmat, Bmat = para.get_parareal_matrix()
