import sys
sys.path.append('./src')

from integrator import integrator
from integrator_dedalus import integrator_dedalus
import numpy as np
import pytest

class TestClass:
    
    # Can instantiate
    def test_caninstantiate(self):     
        nsteps = 12
        sol = integrator_dedalus(0.0, 1.0, nsteps)
