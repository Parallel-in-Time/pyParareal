import sys
sys.path.append('./src')
sys.path.append('./tests')

from solution import solution
from solution_riemann import solution_riemann
import pytest
import numpy as np
from scipy import sparse

class TestClass:

  def setUp(self):
    self.ndof = np.random.randint(255)
    self.M = np.random.rand(self.ndof, self.ndof)
    self.y = np.random.rand(self.ndof)
    self.x = np.random.rand(self.ndof)
    y = np.ones(15)

  # Can instantiate
  def test_caninstantiate(self):
    self.setUp()               
    sol_rm = solution_riemann(self.y, dx=0.1)

  # Solve works correctly
  def test_solve_alpha_zero(self):
    self.setUp()                
    sol_rm = solution_riemann(self.y, dx=0.1)
    sol_rm.solve(0.0)
    
  def test_solve_alpha_notzero(self):
    self.setUp()                
    sol_rm = solution_riemann(self.y, dx=0.1)
    with pytest.raises(Exception):
      sol_rm.solve(1.0)

  def test_can_evaluate_f(self):
    self.setUp()                
    sol_rm = solution_riemann(self.y, dx=0.1)
    sol_rm.f()
