import sys
sys.path.append('../src')

from integrator import integrator
from impeuler import impeuler

class test_impeuler:
  
  #def setUp(self):
  
  # Can instantiate object
  def test_caninstantiate(self):
    ie = impeuler(0.0, 1.0, 10)
