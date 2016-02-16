import sys
sys.path.append('../src')

import numpy as np
from scipy import sparse
from scipy import linalg
from intexact import intexact
from solution_linear import solution_linear
from nose.tools import *
import unittest

class TestIntexact(unittest.TestCase):

  def setUp(self):
    self.ndof = np.random.randint(255)
    self.ndof = 4
    self.A    = sparse.spdiags([ np.ones(self.ndof), -2.0*np.ones(self.ndof), np.ones(self.ndof)], [-1,0,1], self.ndof, self.ndof, format="csc")
    self.M    = sparse.spdiags([ np.random.rand(self.ndof) ], [0], self.ndof, self.ndof, format="csc")
    self.sol  = solution_linear(np.ones((self.ndof,1)), self.A, self.M)

  def test_caninstantiate(self):
    ex = intexact(0.0, 1.0, 10)

  def test_runs(self):
    tend = np.random.rand(1)*10.0
    tend = tend[0]
    ex = intexact(0.0, tend, 10)
    ex.run(self.sol)
    Minv = sparse.linalg.inv(self.M)
    Mex  = sparse.linalg.expm(Minv.dot(self.A)*tend)
    yex = Mex.dot(np.ones((self.ndof,1)))
    uex = solution_linear(yex, self.A, self.M )
    uex.axpy(-1.0, self.sol)
    diff = uex.norm()
    assert diff<1e-14, ("intexact does not provide exact solution. Error: %5.3e" % diff)
