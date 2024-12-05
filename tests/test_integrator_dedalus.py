import sys
sys.path.append('./src')

from integrator import integrator
from integrator_dedalus import integrator_dedalus
from special_integrator import special_integrator
from solution_linear import solution_linear
from solution_dedalus import solution_dedalus
import numpy as np
import pytest

class TestClass:
    
    # Can instantiate
    def test_can_instantiate(self):     
        nsteps = 12
        integ = integrator_dedalus(0.0, 1.0, nsteps)
    
    # Returns a matrix
    def test_returns_matrix(self):
        nsteps = 12
        ndof   = 16
        sol    = solution_dedalus(np.zeros(ndof), ndof)
        integ = integrator_dedalus(0.0, 1.0, nsteps)
        Rmat = integ.get_update_matrix(sol)
        assert isinstance(Rmat, np.ndarray), "The function get_update_matrix of integrator_dedalus returned an object that is not a numpy array"
        assert np.shape(Rmat)[0] == ndof, "The numpy array object returned by get_update_matrix has the wrong shape"
        assert np.shape(Rmat)[1] == ndof, "The numpy array object returned by get_update_matrix has the wrong shape"
        
    # Checks that calling the run function is the same as applying the matrix from get_update_matrix
    def test_matrix_equals_run(self):        
        nsteps = 2
        ndof   = 18
        mesh   = np.linspace(0.0, 1.0, ndof, endpoint=False)
        y      = 0.0*np.sin(2.0*np.pi*mesh)   
        y[5]   = 1.0
        y      = np.reshape(y, (ndof,1))
        u0     = solution_dedalus(np.copy(y), ndof)
        integ  = integrator_dedalus(0.0, 1.0, nsteps)
        Rmat   = integ.get_update_matrix(u0)        
        integ.run(u0)
        y_run = u0.y 
        y_mat = Rmat@y
        assert np.linalg.norm(y_run - y_mat.flatten()) < 1e-14, "Run routine and multiplication with stability matrix do not deliver the same result for integrator_dedalus"
               
    # Tests if the conversion to a special_integrator object works
    def test_can_convert_to_special_integrator(self):
        nsteps = 12
        ndof   = 16
        sol    = solution_dedalus(np.zeros(ndof), ndof)
        integ  = integrator_dedalus(0.0, 1.0, nsteps)        
        obj    = integ.convert_to_special_integrator(sol)
        assert isinstance(obj, special_integrator), "Function convert_to_special_integrator of integrator_dedalus did return an object of the wrong type"
        
    # Checks that converting the dedalus integrator to special integrator and running that delivers the same answer as 
    # running the dedalus integrator
    def test_as_special_equals_run(self):
        nsteps = 1
        ndof   = 4
        mesh   = np.linspace(0.0, 1.0, ndof, endpoint=False)
        y      = np.sin(2.0*np.pi*mesh)        
        u0     = solution_dedalus(np.copy(y), ndof)
        u0_lin = solution_linear(np.copy(y), np.zeros((ndof,ndof)))
        integ  = integrator_dedalus(0.0, 1.0, nsteps)
        integ_special = integ.convert_to_special_integrator(u0)
        integ.run(u0)
        integ_special.run(u0_lin)
        assert np.linalg.norm(u0.y - u0_lin.y.flatten()) < 1e-14, "Running integrator_dedalus does not deliver the same output as converting it to special_integrator and running that"

    def test_can_convert_to_special_integrator_and_run(self):
        nsteps = 128
        ndof   = 32
        sol    = solution_dedalus(np.zeros(ndof), ndof)
        integ  = integrator_dedalus(0.0, 1.0, nsteps)        
        obj    = integ.convert_to_special_integrator(sol)
        # Because the A matrix of the linear solution does not matter for the special_integrator which only applies the provided stability matrix,
        # we can set it to zero
        mesh = np.linspace(0.0, 1.0, ndof, endpoint=False)
        u0     = solution_linear(np.sin(2.0*np.pi*mesh),np.zeros((ndof,ndof)))
        obj.run(u0)
        #print(np.linalg.norm(u0.y))
        #print(np.linalg.norm(np.sin(2.0*np.pi*mesh)))
        #print(np.linalg.norm(u0.y - np.sin(2.0*np.pi*mesh)))
        #assert np.linalg.norm(u0.y - np.sin(2.0*np.pi*mesh)) < 1e-8
