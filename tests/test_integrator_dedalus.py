import sys
sys.path.append('./src')

from integrator import integrator
from integrator_dedalus import integrator_dedalus
from solution_dedalus import solution_dedalus
import numpy as np
import pytest

class TestClass:
    
    # Can instantiate
    def test_caninstantiate(self):     
        nsteps = 12
        integ = integrator_dedalus(0.0, 1.0, nsteps)
    
    # Returns a matrix
    def test_returnsmatrix(self):
        nsteps = 12
        ndof   = 16
        sol    = solution_dedalus(np.zeros(ndof), ndof)
        integ = integrator_dedalus(0.0, 1.0, nsteps)
        Rmat = integ.get_update_matrix(sol)
        assert isinstance(Rmat, np.ndarray), "The function get_update_matrix of integrator_dedalus returned an object that is not a numpy array"
