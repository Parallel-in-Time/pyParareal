import sys
sys.path.append('../src')

from meshtransfer import meshtransfer
from solution import solution
from nose.tools import *
import unittest
import numpy as np

class TestMeshtransfer(unittest.TestCase):

  def setUp(self):
    ndofs = np.random.randint(2)
    self.ndof_coarse = np.min(ndofs)
    self.ndof_fine   = np.max(ndofs)
    
  def test_caninstantiate(self):
    mt = meshtransfer(self.ndof_fine, self.ndof_coarse)
