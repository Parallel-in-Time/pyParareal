import sys
sys.path.append('../src')

from solution import solution
from solution_riemann import solution_riemann
from test_solution import TestSolution
from nose.tools import *
import unittest
import numpy as np
from scipy import sparse

class test_solution_riemann(TestSolution):

  def setUp(self):
    super(test_solution_riemann, self).setUp()
    y = np.ones(15)

  # Can instantiate
  def test_caninstantiate(self):
    sol_rm = solution_riemann(self.y, dx=0.1)

  # Solve works correctly
  def test_solve_alpha_zero(self):
    sol_rm = solution_riemann(self.y, dx=0.1)
    sol_rm.solve(0.0)
    
  @raises(Exception)
  def test_solve_alpha_notzero(self):
    sol_rm = solution_riemann(self.y, dx=0.1)
    sol_rm.solve(1.0)

  def test_can_evaluate_f(self):
    sol_rm = solution_riemann(self.y, dx=0.1)
    sol_rm.f()
