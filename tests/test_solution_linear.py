import sys
sys.path.append('./src')

from solution import solution
from solution_linear import solution_linear
import pytest
import numpy as np
from scipy import sparse

class TestClass:

  def setUp(self):
    self.ndof = np.random.randint(255)
    self.M = np.random.rand(self.ndof, self.ndof)
    self.y = np.random.rand(self.ndof)
    self.x = np.random.rand(self.ndof)
    self.A = sparse.spdiags([ np.ones(self.ndof), -2.0*np.ones(self.ndof), np.ones(self.ndof)], [-1,0,1], self.ndof, self.ndof)

  # Can instantiate
  def test_caninstantiate(self):
    self.setUp()                   
    sol_lin = solution_linear(self.y, self.A, self.M)

  # Throws if size of matrix A does not match y and M
  def test_wrongsizeA(self):
    self.setUp()                   
    A = np.random.rand(self.ndof+1, self.ndof+1)
    with pytest.raises(AssertionError):
      sol_lin = solution_linear(self.y, A, self.M)

  # Solve works correctly
  def test_solve(self):
    self.setUp()                   
    self.M = sparse.spdiags([ np.random.rand(self.ndof) ], [0], self.ndof, self.ndof)*sparse.identity(self.ndof)
    self.A = sparse.csc_matrix(self.A)
    u = np.reshape(np.random.rand(self.ndof), (self.ndof,1))
    b = ( self.M - 0.1*self.A ).dot(u)
    sol_lin = solution_linear(b, self.A, self.M)
    sol_lin.solve(0.1)
    assert np.allclose(sol_lin.y, u, rtol=1e-12, atol=1e-12), "Solution provided by solve seems wrong"
