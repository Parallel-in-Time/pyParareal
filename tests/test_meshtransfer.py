import sys
sys.path.append('../src')

from meshtransfer import meshtransfer
from solution import solution
from solution_linear import solution_linear
from nose.tools import *
import unittest
import numpy as np

class TestMeshtransfer(unittest.TestCase):

  def setUp(self):
    ndofs = np.random.randint(5,64,size=2)
    self.ndof_coarse = np.min(ndofs)
    self.ndof_fine   = np.max(ndofs)
    
  def test_caninstantiate(self):
    mt = meshtransfer(self.ndof_fine, self.ndof_coarse)
    
  def test_samendofdoesnothing(self):
    self.ndof_fine = 5
    mt = meshtransfer(self.ndof_fine, self.ndof_fine)
    y = np.atleast_2d(np.random.rand(self.ndof_fine)).T
    sol_c = solution_linear(y, np.eye(self.ndof_fine))
    sol_f = solution_linear(y, np.eye(self.ndof_fine))
    mt.restrict(sol_f, sol_c)
    assert np.linalg.norm(sol_c.y - y, np.inf)<1e-15, "Restriction with same number of DoF on coarse and fine mesh is not the identity."
    mt.interpolate(sol_f, sol_c)
    assert np.linalg.norm(sol_f.y - y, np.inf)<1e-15, "Interpolation with same number of DoF on coarse and fine mesh is not the identity."
    
