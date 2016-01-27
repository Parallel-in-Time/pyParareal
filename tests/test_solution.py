import sys
sys.path.append('../src')

from solution import solution
from nose.tools import *
import unittest
import numpy as np
import scipy as sp

class TestSolution(unittest.TestCase):

  def setUp(self):
    self.ndof = np.random.randint(255)
    self.M = np.random.rand(self.ndof, self.ndof)
    self.y = np.random.rand(self.ndof)
    self.x = np.random.rand(self.ndof)
    self.a = np.random.rand(1)

  # Make sure solution can be instantiated and that default M is identity
  def test_caninstantiate(self):
    sol = solution(self.y)
    assert np.array_equal(sol.M, sp.eye(self.ndof)), "Default M matrix is not identity"

  # Make sure exception is raised if y is not of type numpy.ndarray
  def test_failnoarray(self):
    with self.assertRaises(AssertionError):
      sol = solution(-1.0)

  # Make sure exception is raised if y is a 2D array
  def test_failsmatrixy(self):
    with self.assertRaises(AssertionError):
      sol = solution(np.array([[1, 1], [1, 1]]))

  # Make sure if matrix M is given as argument to constructor, it is used
  def test_fisused(self):
    sol = solution(self.y, self.M)
    assert np.array_equal(sol.M, self.M), "Stored M matrix is not identical to input"

  # Make sure exception is raised if size of y and M does not match
  def test_mismatchym(self):
    y = np.random.rand(self.ndof-1)
    with self.assertRaises(AssertionError):
      sol = solution(y, self.M)

  # Make sure exception is raised if M has more than two dimensions
  def test_mtoomanydim(self):
    M = np.random.rand(ndof, ndof, ndof)
    with self.assertRaises(AssertionError):
      sol = solution(self.y, M)

  # Make sure axpy performs expected operation
  def test_mtoomanydim(self):
    sol = solution(self.y)
    axpy = self.a*self.x+self.y
    sol_x = solution(self.x)
    sol.axpy(self.a, sol_x)
    assert np.array_equal(sol.y, axpy)

  # Make sure axpy throws exception if size of does not match y
  def test_yxmismatch(self):
      x = solution(np.random.rand(self.ndof+2))
      sol = solution(self.y)
      with self.assertRaises(AssertionError):
        sol.axpy(self.a, x)

  # Axpy correctly interprets a float as 1 entry numpy array
  def test_axpyfloat(self):
    a = 0.1
    sol = solution(self.y)
    sol2 = solution(np.ones(self.ndof))
    sol.axpy(0.1, sol2)
    assert np.array_equal(sol.y, a*np.ones(self.ndof) + self.y)

  # Make sure axpy throws exception if a is not a scalar
  def test_alphanotscalar(self):
      a = np.random.rand(3)
      sol = solution(self.y)
      with self.assertRaises(AssertionError):
        sol.axpy(a, self.x)  
