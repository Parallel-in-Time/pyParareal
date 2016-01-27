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
  
#  def setUp(self):

  # Can instantiate object
  def test_caninstantiate(self):
    ie = impeuler(0.0, 1.0, 10)

  # Throws if tend < tstart
  def test_tendbeforetstartthrows(self):
    with self.assertRaises(AssertionError):
      ie = impeuler(1.0, 0.0, 10)

  # See if run function can be called and returns correct value
  def test_canncallrun(self):
    ndof = np.random.randint(255)
    ndof = 5
#    M = sparse.spdiags([1.0+np.random.rand(ndof)], [0], ndof, ndof)
    M = sparse.identity(ndof)
    A = sparse.spdiags([ np.ones(ndof), -2.0*np.ones(ndof), np.ones(ndof)], [-1,0,1], ndof, ndof)
    u0 = solution_linear(np.ones(ndof), A, M)
    nsteps = 1
    ie = impeuler(0.0, 1.0, nsteps)
    ie.run(u0)
    # Compute output through update matrix and compare
    Rmat = sparse.linalg.inv(M - ie.dt*A)
    print Rmat
    print M
    Rmat = np.dot(M, Rmat)
    Rmat = np.linalg.matrix_power(Rmat, nsteps)
    uend = solution_linear( np.dot(Rmat, np.ones(ndof)), A, M)
    uend.axpy(-1.0, u0)
    print uend.norm()
