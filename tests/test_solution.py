import sys
sys.path.append('../src')

from solution import solution
from nose.tools import *
import unittest
import numpy as np
import scipy as sp

class TestSolution(unittest.TestCase):

  # Make sure solution can be instantiated and that default M is identity
  def test_caninstantiate(self):
    ndof = np.random.randint(255)
    y = np.random.rand(ndof)
    sol = solution(y)
    assert np.array_equal(sol.M, sp.eye(ndof))

  # Make sure exception is raised if y is not 1D array
  @raises(Exception)
  def test_failsmatrixy(self):
    sol = solution(1.0, np.matrix([[1, 1], [1, 1]]))

  # Make sure if matrix M is given as argument to constructor, it is used
  def test_fistused(self):
    ndof = np.random.randint(255)
    M = np.random.rand(ndof, ndof)
    y = np.random.rand(ndof)
    sol = solution(y, M)
    assert np.array_equal(sol.M, M)

  # Make sure exception is raised if size of y and M does not match
  @raises(Exception)
  def test_mismatchym(self):
    ndof = np.random.randint(255)
    M = np.random.rand(ndof, ndof)
    y = np.random.rand(ndof-1)
    sol = solution(y, M)

  # Make sure exception is raised if M has more than two dimensions
  @raises(Exception)
  def test_mtoomanydim(self):
    ndof = np.random.randint(255)
    M = np.random.rand(ndof, ndof, ndof)
    y = np.random.rand(ndof)
    sol = solution(y, M)

  # Make sure axpy performs expected operation
  def test_mtoomanydim(self):
    # perform ten tests with random size and values
    for i in range(0,10):
      ndof = np.random.randint(255)
      y = np.random.rand(ndof)
      x = np.random.rand(ndof)
      a = np.random.rand(1)
      sol = solution(y)
      axpy = a*x+y
      sol.axpy(a, x)
      assert np.array_equal(sol.y, axpy)

  # Make sure axpy throws exception if size of does not match y
  @raises(Exception)
  def test_yxmismatch(self):
      ndof = np.random.randint(255)
      y = np.random.rand(ndof)
      x = np.random.rand(ndof-2)
      a = np.random.rand(1)
      sol = solution(y)
      sol.axpy(a, x)

  # Make sure axpy throws exception if a is not a scalar
  @raises(Exception)
  def test_alphanotscalar(self):
      ndof = np.random.randint(255)
      y = np.random.rand(ndof)
      x = np.random.rand(ndof)
      a = np.random.rand(3)
      sol = solution(y)
      sol.axpy(a, x)  
