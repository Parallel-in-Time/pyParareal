import sys
sys.path.append('./src')

import numpy as np
from scipy import sparse
from scipy import linalg
from integrator import integrator
from impeuler import impeuler
from trapezoidal import trapezoidal
from special_integrator import special_integrator
from solution_linear import solution_linear
import copy
import pytest

class TestClass:

  def setUp(self):
    self.ndof = np.random.randint(255)
    self.A    = sparse.spdiags([ np.ones(self.ndof), -2.0*np.ones(self.ndof), np.ones(self.ndof)], [-1,0,1], self.ndof, self.ndof, format="csc")
    self.M    = sparse.spdiags([ np.random.rand(self.ndof) ], [0], self.ndof, self.ndof, format="csc")
    self.sol  = solution_linear(np.ones(self.ndof), self.A, self.M)

  # Can instantiate object
  def test_caninstantiate(self):
    self.setUp()            
    integ = special_integrator(0.0, 1.0, 10, sparse.eye(self.ndof))

  # Throws if tend < tstart
  def test_tendbeforetstartthrows(self):
    self.setUp()            
    with pytest.raises(AssertionError):
      integ = special_integrator(1.0, 0.0, 10, sparse.eye(self.ndof))

  # See if run function can be called
  def test_cancallrun(self):
    self.setUp()            
    integ = special_integrator(0.0, 1.0, 10, sparse.eye(self.ndof))
    integ.run(self.sol)

  # Make sure it provides identical solution to other integrator when stability function is given
  def test_reproducesimpeuler(self):
    self.setUp()            
    ie = impeuler(0.0, 1.0, 25)
    M = sparse.csc_matrix(self.sol.M)
    A = sparse.csc_matrix(self.sol.A)
    Rmat = sparse.linalg.inv(M - ie.dt*A)
    Rmat = Rmat.dot(M)
    integ = special_integrator(0.0, 1.0, 25, Rmat)
    sol2 = copy.deepcopy(self.sol)
    ie.run(sol2)
    integ.run(self.sol)
    self.sol.axpy(-1.0, sol2)
    assert (self.sol.norm()<1e-14), "special_integrator did produce output identical to backward Euler"

  def test_reproducetrapezoidal(self):
    self.setUp()            
    trap = trapezoidal(0.0, 1.0, 25)
    M = sparse.csc_matrix(self.sol.M)
    A = sparse.csc_matrix(self.sol.A)
    Rmat = sparse.linalg.inv(M - 0.5*trap.dt*A)
    Rmat = Rmat.dot(M+0.5*trap.dt*A)
    integ = special_integrator(0.0, 1.0, 25, Rmat)
    sol2 = copy.deepcopy(self.sol)
    trap.run(sol2)
    integ.run(self.sol)
    self.sol.axpy(-1.0, sol2)
    assert (self.sol.norm()<1e-14), "special_integrator did produce output identical to trapezoidal rule"
