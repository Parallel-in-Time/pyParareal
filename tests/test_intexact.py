import sys
sys.path.append('../src')

import numpy as np
from scipy import sparse
from scipy import linalg
from intexact import intexact
from solution_linear import solution_linear
import copy
from nose.tools import *

import unittest

class TestIntexact(unittest.TestCase):

  def setUp(self):
    #self.ndof = np.random.randint(255)
    self.ndof = 1
    self.A    = sparse.spdiags([ np.ones(self.ndof), -2.0*np.ones(self.ndof), np.ones(self.ndof)], [-1,0,1], self.ndof, self.ndof, format="csc")
    self.M    = sparse.spdiags([ np.random.rand(self.ndof) ], [0], self.ndof, self.ndof, format="csc")
    self.sol  = solution_linear(np.ones((self.ndof,1)), self.A, self.M)

  def test_caninstantiate(self):
    ex = intexact(0.0, 1.0, 10)

  def test_runs(self):
    tend = np.random.rand(1)*10.0
    tend = tend[0]
    ex   = intexact(0.0, tend, 10)
    ex.run(self.sol)
    yex = np.exp(tend*self.A[0,0]/self.M[0,0])*1.0
    uex = solution_linear(np.array([[yex]],dtype='complex'), self.A, self.M )
    uex.axpy(-1.0, self.sol)
    diff = uex.norm()
    assert diff<1e-14, ("intexact does not provide exact solution. Error: %5.3e" % diff)

  def test_runequalmatrix(self):
    tend   = np.random.rand(1)*10.0
    tend   = tend[0]
    nsteps = np.random.randint(2,30)
    ex     = intexact(0.0, tend, nsteps)
    ex.run(self.sol)
    M      = ex.get_update_matrix(self.sol)
    y_mat  = M.dot(np.ones((self.ndof,1)))
    diff = np.linalg.norm(y_mat - self.sol.y, np.inf)/np.linalg.norm(self.sol.y, np.inf)
    assert diff<1e-14, ("Update matrix of intexact does not provide same result as run. Error: %5.3e" % diff)

  # For the exact integrator the number of timesteps should not affect the result
  def test_invariantnsteps(self):
    tend   = np.random.rand(1)*10.0
    tend   = tend[0]
    nsteps = np.random.randint(2,30)
    ex     = intexact(0.0, tend, nsteps)
    ex_onestep = intexact(0.0, tend, 1)
    sol1   = copy.deepcopy(self.sol)
    ex.run(self.sol)
    ex_onestep.run(sol1)
    self.sol.axpy(-1.0, sol1)
    err = self.sol.norm()
    assert err<1e-14, ("Exact integrator not invariant in number of timesteps. Error: %5.3e" % err)
