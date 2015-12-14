import sys
sys.path.append('../src')

from integrator import integrator
from nose.tools import *
import unittest
import numpy as np

class TestIntegrator(unittest.TestCase):
  
  # Make sure integrator can be instantiated
  def test_caninstantiate(self):
    t = np.sort( np.random.rand(2) )
    integ = integrator(t[0], t[1], 25)

  # Make sure instantiation fails if tend is smaller than tstart
  @raises(Exception)
  def test_wrongboundsfail(self):
    integ = integrator(0.0, -1.0, 10)

  # Make sure instantiation fails if nstep is negative
  @raises(Exception)
  def test_nstepnegativefail(self):
    integ = integrator(0.0, 1.0, -10)

  # Make sure instantiation fails if nstep is float
  @raises(Exception)
  def test_nstepfloatfail(self):
    integ = integrator(0.0, 1.0, 1.15)
