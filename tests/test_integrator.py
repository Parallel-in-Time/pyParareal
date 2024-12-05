import sys
sys.path.append('./src')

from integrator import integrator
from solution import solution
import pytest

import numpy as np

class TestClass:
  
  def setUp(self):
    self.t = np.sort( np.random.rand(2) )
    self.nsteps = 1 + np.random.randint(25)

  # Make sure integrator can be instantiated
  def test_caninstantiate(self):
    self.setUp()            
    integ = integrator(self.t[0], self.t[1], self.nsteps)

  # Make sure instantiation fails if tend is smaller than tstart
  def test_wrongboundsfail(self):
    self.setUp()            
    with pytest.raises(AssertionError):
      integ = integrator(0.0, -1.0, 10)

  # Make sure instantiation fails if nstep is negative
  def test_nstepnegativefail(self):
    self.setUp()                
    with pytest.raises(AssertionError):
      integ = integrator(0.0, 1.0, -10)

  # Make sure instantiation fails if nstep is float
  def test_nstepfloatfail(self):
    self.setUp()                
    with pytest.raises(AssertionError):
      integ = integrator(0.0, 1.0, 1.15)

  # run function of generic integrator should raise exception
  def test_failgenericrun(self):
    self.setUp()                
    integ = integrator(self.t[0], self.t[1], self.nsteps)
    sol   = solution(np.array([-1]))
    with pytest.raises(NotImplementedError):
      integ.run(sol)

  # run function of integrator with initial value not of solution type should raise exception
  def test_failwrongu0(self):
    self.setUp()                
    integ = integrator(self.t[0], self.t[1], self.nsteps)
    with pytest.raises(AssertionError):
      integ.run(-1)
