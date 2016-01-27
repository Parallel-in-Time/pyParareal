import sys
sys.path.append('../src')

from solution import solution
from solution_linear import solution_linear
from test_solution import TestSolution
from nose.tools import *
import unittest
import numpy as np
from scipy import sparse

class test_solution_linear(TestSolution):

  def setUp(self):
    super(test_solution_linear, self).setUp()
    self.A = sparse.spdiags([ np.ones(self.ndof), -2.0*np.ones(self.ndof), np.ones(self.ndof)], [-1,0,1], self.ndof, self.ndof)

  # Can instantiate
  def test_caninstantiate(self):
    sol_lin = solution_linear(self.y, self.A, self.M)

  # Throws if size of matrix A does not match y and M
  def test_wrongsizeA(self):
    A = np.random.rand(self.ndof+1, self.ndof+1)
    with self.assertRaises(AssertionError):
      sol_lin = solution_linear(self.y, A, self.M)

  # Solve works correctly
  def test_solve(self):
    u = np.random.rand(self.ndof)
    print np.shape(u)
    print np.shape(self.M)
    print np.shape(self.A)
    b = np.ravel( np.dot( self.M - 0.1*self.A, u) )
    sol_lin = solution_linear(b, self.A, self.M)
    sol_lin.solve(0.1)
    print np.linalg.norm(sol_lin.y - u, np.infty)
    assert np.allclose(sol_lin.y, u, rtol=1e-12, atol=1e-12)
