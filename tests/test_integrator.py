import sys
sys.path.append('../src')

from integrator import integrator
from nose.tools import *
import unittest
import numpy as np

class TestIntegrator(unittest.TestCase):
  
  def setUp(self):
    self.t = np.sort( np.random.rand(2) )
    self.nsteps = np.random.randint(25)

  # Make sure integrator can be instantiated
  def test_caninstantiate(self):
    integ = integrator(self.t[0], self.t[1], self.nsteps)

  # Make sure instantiation fails if tend is smaller than tstart
  def test_wrongboundsfail(self):
    with self.assertRaises(Exception):
      integ = integrator(0.0, -1.0, 10)

  # Make sure instantiation fails if nstep is negative
  def test_nstepnegativefail(self):
    with self.assertRaises(Exception):
      integ = integrator(0.0, 1.0, -10)

  # Make sure instantiation fails if nstep is float
  def test_nstepfloatfail(self):
    with self.assertRaises(Exception):
      integ = integrator(0.0, 1.0, 1.15)
