import sys
sys.path.append('../src')

import numpy as np
from scipy import sparse
from scipy import linalg
from integrator import integrator
from impeuler import impeuler
from solution_linear import solution_linear
from nose.tools import *
import unittest

class TestImpeuler(unittest.TestCase):
  
  def setUp(self):
    self.ndof = np.random.randint(255)
    self.A    = sparse.spdiags([ np.ones(self.ndof), -2.0*np.ones(self.ndof), np.ones(self.ndof)], [-1,0,1], self.ndof, self.ndof, format="csc")
    self.M    = sparse.spdiags([ np.random.rand(self.ndof) ], [0], self.ndof, self.ndof, format="csc")
    self.sol  = solution_linear(np.ones(self.ndof), self.A, self.M)

  # Can instantiate object
  def test_caninstantiate(self):
    ie = impeuler(0.0, 1.0, 10)

  # Throws if tend < tstart
  def test_tendbeforetstartthrows(self):
    with self.assertRaises(AssertionError):
      ie = impeuler(1.0, 0.0, 10)

  # See if run function can be called
  def test_cancallrun(self):
    ie = impeuler(0.0, 1.0, 10)
    ie.run(self.sol)

  # See if run does the same as the update matrix
  def test_callcorrect(self):
    u0 = solution_linear(np.ones(self.ndof), self.A, self.M)
    nsteps = 13
    ie = impeuler(0.0, 1.0, nsteps)
    ie.run(u0)
    # Compute output through update matrix and compare
    Rmat = ie.get_update_matrix(u0)
    Rmat = np.linalg.matrix_power(Rmat.todense(), nsteps)
    yend = Rmat.dot(np.ones((self.ndof,1)))
    sol_end = solution_linear( yend, self.A, self.M )
    sol_end.axpy(-1.0, u0)
    assert sol_end.norm()<1e-12, ("Output from implicit Euler integrator differs from result computed with power of update matrix -- norm of difference: %5.3e" % sol_end.norm())

