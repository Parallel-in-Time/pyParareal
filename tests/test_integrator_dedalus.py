import sys
sys.path.append('./src')

from integrator import integrator
from integrator_dedalus import integrator_dedalus
from special_integrator import special_integrator
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
        assert np.shape(Rmat)[0] == ndof, "The numpy array object returned by get_update_matrix has the wrong shape"
        assert np.shape(Rmat)[1] == ndof, "The numpy array object returned by get_update_matrix has the wrong shape"
        
    # Tests if the conversion to a special_integrator object works
    def test_can_convert_to_special_integrator(self):
        nsteps = 12
        ndof   = 16
        sol    = solution_dedalus(np.zeros(ndof), ndof)
        integ  = integrator_dedalus(0.0, 1.0, nsteps)        
        obj    = integ.convert_to_special_integrator(sol)
        assert isinstance(obj, special_integrator), "Function convert_to_special_integrator of integrator_dedalus did return an object of the wrong type"
