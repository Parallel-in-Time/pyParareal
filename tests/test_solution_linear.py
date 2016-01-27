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
    self.M = sparse.spdiags([ np.random.rand(self.ndof) ], [0], self.ndof, self.ndof)*sparse.identity(self.ndof)
    self.A = sparse.csc_matrix(self.A)
    u = np.random.rand(self.ndof)
    b = ( self.M - 0.1*self.A ).dot(u).T
    sol_lin = solution_linear(b, self.A, self.M)
    sol_lin.solve(0.1)
    assert np.allclose(sol_lin.y, u, rtol=1e-12, atol=1e-12), "Solution provided by solve seems wrong"
