import sys
sys.path.append('./src')

from parareal import parareal
from timemesh import timemesh
from impeuler import impeuler
from integrator_dedalus import integrator_dedalus
from solution_linear import solution_linear
from solution_dedalus import solution_dedalus
import pytest
import numpy as np
from scipy import sparse
from scipy.sparse import linalg
import math

class TestClass:

  def setUp(self):
    times        = np.sort( np.random.rand(2) )
    self.tstart  = times[0]
    self.tend    = times[1]
    self.nslices = np.random.randint(2,32) 
    steps        = np.sort( np.random.randint(low=1, high=64, size=2) )
    self.ncoarse = steps[0]
    self.nfine   = steps[1]
    ndofs         = [ np.random.randint(4,32), np.random.randint(4,32)]

    self.ndof_f   = np.max(ndofs)
    self.A_f       = sparse.spdiags([ np.ones(self.ndof_f), -2.0*np.ones(self.ndof_f), np.ones(self.ndof_f)], [-1,0,1], self.ndof_f, self.ndof_f, format="csc")
    self.M_f       = sparse.spdiags([ 10.0+np.random.rand(self.ndof_f) ], [0], self.ndof_f, self.ndof_f, format="csc")
    self.u0      = solution_linear(np.ones((self.ndof_f,1)), self.A_f, self.M_f)

    self.ndof_c   = np.min(ndofs)
    self.A_c       = sparse.spdiags([ np.ones(self.ndof_c), -2.0*np.ones(self.ndof_c), np.ones(self.ndof_c)], [-1,0,1], self.ndof_c, self.ndof_c, format="csc")
    self.M_c       = sparse.spdiags([ 10.0+np.random.rand(self.ndof_c) ], [0], self.ndof_c, self.ndof_c, format="csc")
    self.u0coarse  = solution_linear(np.ones((self.ndof_c,1)), self.A_c, self.M_c)

  # Can instantiate object of type parareal
  def test_caninstantiate(self):
    self.setUp()        
    para = parareal(self.tstart, self.tend, self.nslices, impeuler, impeuler, self.nfine, self.ncoarse, 1e-10, 5, self.u0)

  # Can instantiate object of type parareal
  def test_caninstantiateWithu0coarse(self):
    self.setUp()        
    para = parareal(self.tstart, self.tend, self.nslices, impeuler, impeuler, self.nfine, self.ncoarse, 1e-10, 5, self.u0, self.u0coarse)
    
  # Can execute run function
  def test_canrun(self):
    self.setUp()        
    para = parareal(self.tstart, self.tend, self.nslices, impeuler, impeuler, self.nfine, self.ncoarse, 1e-10, 5, self.u0)
    para.run()
    
  # Can execute run function
  def test_canrunwithu0coarse(self):
    self.setUp()        
    para = parareal(self.tstart, self.tend, self.nslices, impeuler, impeuler, self.nfine, self.ncoarse, 1e-10, 5, self.u0, self.u0coarse)
    para.run()

  # Test matrix Parareal
  def test_pararealmatrix(self):
    self.setUp()        
    para = parareal(self.tstart, self.tend, self.nslices, impeuler, impeuler, self.nfine, self.ncoarse, 1e-10, 1, self.u0)
    Pmat, Bmat = para.get_parareal_matrix()
    bvec = np.zeros((self.ndof_f*(self.nslices+1),1))
    bvec[0:self.ndof_f,:] = self.u0.y
    # Perform one coarse step by matrix multiplication
    y0 = Bmat@bvec
    # Perform one Parareal step in matrix form
    y_mat = Pmat@y0 + Bmat@bvec
    para.run()
    y_para = np.zeros((self.ndof_f*(self.nslices+1),1))
    y_para[0:self.ndof_f,:] = self.u0.y
    for i in range(0,self.nslices):
      y_para[(i+1)*self.ndof_f:(i+2)*self.ndof_f,:] = para.get_end_value(i).y
    err = np.linalg.norm(y_para - y_mat, np.inf)
    assert err<1e-12, ("Parareal run and matrix form do not yield identical results for a single iteration. Error: %5.3e" % err)

  # Test matrix Parareal when u0coarse is provided
  def test_pararealmatrixwithu0coarse(self):
    self.setUp()        
    para = parareal(self.tstart, self.tend, self.nslices, impeuler, impeuler, self.nfine, self.ncoarse, 1e-10, 1, self.u0, self.u0coarse)
    Pmat, Bmat = para.get_parareal_matrix()
    bvec = np.zeros((self.ndof_f*(self.nslices+1),1))
    bvec[0:self.ndof_f,:] = self.u0.y
    # Perform one coarse step by matrix multiplication
    y0 = Bmat@bvec
    # Perform one Parareal step in matrix form
    y_mat = Pmat@y0 + Bmat@bvec
    para.run()
    y_para = np.zeros((self.ndof_f*(self.nslices+1),1))
    y_para[0:self.ndof_f,:] = self.u0.y
    for i in range(0,self.nslices):
      y_para[(i+1)*self.ndof_f:(i+2)*self.ndof_f,:] = para.get_end_value(i).y
    err = np.linalg.norm(y_para - y_mat, np.inf)
    assert err<1e-12, ("Parareal run and matrix form do not yield identical results for a single iteration. Error: %5.3e" % err)
    
  # Test matrix Parareal
  def test_pararealmatrixmultiple(self):
    self.setUp()        
    niter = np.random.randint(2,8) 
    para = parareal(self.tstart, self.tend, self.nslices, impeuler, impeuler, self.nfine, self.ncoarse, 0.0, niter, self.u0)
    Pmat, Bmat = para.get_parareal_matrix()
    bvec = np.zeros((self.ndof_f*(self.nslices+1),1))
    bvec[0:self.ndof_f,:] = self.u0.y
    # Perform one coarse step by matrix multiplication
    y_mat = Bmat@bvec
    # Perform niter Parareal step in matrix form
    for i in range(0,niter):
      y_mat = Pmat@y_mat + Bmat@bvec
    para.run()
    y_para = np.zeros((self.ndof_f*(self.nslices+1),1))
    y_para[0:self.ndof_f,:] = self.u0.y
    for i in range(0,self.nslices):
      y_para[(i+1)*self.ndof_f:(i+2)*self.ndof_f,:] = para.get_end_value(i).y
    err = np.linalg.norm(y_para - y_mat, np.inf)
    assert err<1e-12, ("Parareal run and matrix form do not yield identical results for multiple iterations. Error: %5.3e" % err)

  # Parareal reproduces fine solution after niter=nslice many iterations
  def test_reproduces_fine(self):
    self.setUp()        
    # Smaller number of slices to keep runtime short
    nslices = np.random.randint(2,12) 
    para = parareal(self.tstart, self.tend, nslices, impeuler, impeuler, self.nfine, self.ncoarse, 0.0, nslices, self.u0, self.u0coarse)
    Fmat = para.timemesh.get_fine_matrix(self.u0)
    b = np.zeros((self.ndof_f*(nslices+1),1))
    b[0:self.ndof_f,:] = self.u0.y
    # Solve system
    u = linalg.spsolve(Fmat, b)
    u = u.reshape((self.ndof_f*(nslices+1),1))
    # Run Parareal
    para.run()
    u_para = para.get_parareal_vector()
    diff = np.linalg.norm(u_para - u, np.inf)
    assert diff<1e-12, ("Parareal does not reproduce fine solution after nslice=niter many iterations. Error: %5.3e" % diff)
    
  @pytest.mark.skip(reason="Parareal cannot be run with solution_dedalus because the deepcopy commands throw and error because of some MPI object")    
  def test_reproduces_fine_with_dedalus(self):
    self.setUp()      
    # Make sure number of degrees of freedom is even
    ndof_f = math.ceil(self.ndof_f/2)*2
    ndof_c = math.ceil(self.ndof_c/2)*2       
    u0       = solution_dedalus(np.random.rand(ndof_f), ndof_f)
    u0coarse = solution_dedalus(np.random.rand(ndof_c), ndof_c)
    niter = np.random.randint(2,12) 
    para = parareal(self.tstart, self.tend, self.nslices, integrator_dedalus, integrator_dedalus, self.nfine, self.ncoarse, 0.0, niter, u0, u0coarse)    
    Fmat = para.timemesh.get_fine_matrix(u0)
    b = np.zeros((ndof_f*(self.nslices+1),1))
    b[0:ndof_f,:] = u0.y
    # Solve system
    u = linalg.spsolve(Fmat, b)
    u = u.reshape((ndof_f*(self.nslices+1),1))
    # Run Parareal
    para.run()
    u_para = para.get_parareal_vector()
    diff = np.linalg.norm(u_para - u, np.inf)
    assert diff<1e-12, ("Parareal does not reproduce fine solution after nslice=niter many iterations. Error: %5.3e" % diff)      
    
  def test_matrix_parareal_reproduces_fine_with_dedalus(self):
    self.setUp()      
    # Make sure number of degrees of freedom is even
    ndof_f = math.ceil(self.ndof_f/2)*2
    ndof_c = math.ceil(self.ndof_c/2)*2       
    u0       = solution_dedalus(np.random.rand(ndof_f), ndof_f)
    u0coarse = solution_dedalus(np.random.rand(ndof_c), ndof_c)
    niter = np.random.randint(2,12) 
    para = parareal(self.tstart, self.tend, self.nslices, integrator_dedalus, integrator_dedalus, self.nfine, self.ncoarse, 0.0, niter, u0, u0coarse)    
    Fmat = para.timemesh.get_fine_matrix(u0)
    Pmat, Bmat = para.get_parareal_matrix()  
    b = np.zeros((ndof_f*(self.nslices+1),1))
    b[0:ndof_f,:] = u0.y
    # Solve system
    u_fine = linalg.spsolve(Fmat, b)
    u_fine = u_fine.reshape((ndof_f*(self.nslices+1),1))
    # Solve Parareal matrix formulation
    u_para = np.copy(b)
    for n in range(self.nslices):
      # Apply matrix to fine solution
      u_para = Pmat@u_para + Bmat@b
    print(np.shape(u_para))
    print(np.shape(u_fine))
    diff = np.linalg.norm(u_para - u_fine, np.inf)
    assert diff<1e-13, ("Parareal does not reproduce fine solution after nslice=niter many iterations. Error: %5.3e" % diff)  
    
  # Fine solution is fixed point of Parareal iteration
  def test_fine_is_fixed_point_with_dedalus(self):
    self.setUp()      
    # Make sure number of degrees of freedom is even
    ndof_f = math.ceil(self.ndof_f/2)*2
    ndof_c = math.ceil(self.ndof_c/2)*2       
    u0       = solution_dedalus(np.random.rand(ndof_f), ndof_f)
    u0coarse = solution_dedalus(np.random.rand(ndof_c), ndof_c)
    niter = np.random.randint(2,8) 
    para = parareal(self.tstart, self.tend, self.nslices, integrator_dedalus, integrator_dedalus, self.nfine, self.ncoarse, 0.0, niter, u0, u0coarse)
    Fmat = para.timemesh.get_fine_matrix(u0)
    b = np.zeros((ndof_f*(self.nslices+1),1))
    b[0:ndof_f,:] = u0.y
    # Solve system
    u = linalg.spsolve(Fmat, b)
    u = u.reshape((ndof_f*(self.nslices+1),1))
    # Get Parareal iteration matrices
    Pmat, Bmat = para.get_parareal_matrix()
    # Apply matrix to fine solution
    u_para = Pmat@u + Bmat@b
    diff = np.linalg.norm( u_para - u, np.inf)
    assert diff<1e-14, ("Fine solution is not a fixed point of Parareal iteration - difference %5.3e" % diff)
    
  # Fine solution is fixed point of Parareal iteration
  def test_fine_is_fixed_point(self):
    self.setUp()        
    niter = np.random.randint(2,8) 
    para = parareal(self.tstart, self.tend, self.nslices, impeuler, impeuler, self.nfine, self.ncoarse, 0.0, niter, self.u0, self.u0coarse)
    Fmat = para.timemesh.get_fine_matrix(self.u0)
    b = np.zeros((self.ndof_f*(self.nslices+1),1))
    b[0:self.ndof_f,:] = self.u0.y
    # Solve system
    u = linalg.spsolve(Fmat, b)
    u = u.reshape((self.ndof_f*(self.nslices+1),1))
    # Get Parareal iteration matrices
    Pmat, Bmat = para.get_parareal_matrix()
    # Apply matrix to fine solution
    u_para = Pmat@u + Bmat@b
    diff = np.linalg.norm( u_para - u, np.inf)
    assert diff<1e-14, ("Fine solution is not a fixed point of Parareal iteration - difference %5.3e" % diff)    

  # Stability function is equivalent to full run of Parareal
  def test_stabfunction(self):
    self.setUp()        
    niter = np.random.randint(2,8)
    para  = parareal(self.tstart, self.tend, self.nslices, impeuler, impeuler, self.nfine, self.ncoarse, 0.0, niter, self.u0, self.u0coarse)
    Smat  = para.get_parareal_stab_function(niter)
    y_mat =  Smat@self.u0.y
    para.run()
    y_par = para.get_last_end_value().y
    diff  = np.linalg.norm(y_mat - y_par, np.inf)
    assert diff<1e-12, ("Generated Parareal stability matrix does not match result from run(). Error: %5.3e" % diff)

  # Fine solution is fixed point of Parareal iteration if Gmat is provided in call to get_parareal_matrix
  def test_fineisfixedpointGmatprovided(self):
    self.setUp()        
    niter = np.random.randint(2,8) 
    para = parareal(self.tstart, self.tend, self.nslices, impeuler, impeuler, self.nfine, self.ncoarse, 0.0, niter, self.u0, self.u0coarse)

    Fmat = para.timemesh.get_fine_matrix(self.u0)
    b = np.zeros((self.ndof_f*(self.nslices+1),1))
    b[0:self.ndof_f,:] = self.u0.y
    # Solve system
    u = linalg.spsolve(Fmat, b)
    u = u.reshape((self.ndof_f*(self.nslices+1),1))

    # Build coarse solution with matrix different from fine
    ucoarse = solution_linear(np.ones((self.ndof_c,1)), sparse.eye(self.ndof_c, format="csc"))

    # Get Parareal iteration matrices with ucoarse provided
    Pmat, Bmat = para.get_parareal_matrix(ucoarse=ucoarse)

    # For comparison also without ucoarse provided and check that both are different
    Pmat_ref, Bmat_ref = para.get_parareal_matrix()
    assert np.linalg.norm(Bmat_ref.todense() - Bmat.todense(), np.inf)>1e-4, "Parareal iteration matrix Bmat provided with and without ucoarse as argument do not seem to be different."
    assert np.linalg.norm(Pmat_ref.todense() - Pmat.todense(), np.inf)>1e-4, "Parareal iteration matrix Pmat provided with and without ucoarse as argument do not seem to be different."

    # Apply matrix to fine solution
    u_para = Pmat.dot(u) + Bmat.dot(b)
    diff = np.linalg.norm( u_para - u, np.inf)
    assert diff<1e-14, ("Fine solution is not a fixed point of Parareal iteration with provided Gmat matrix - difference %5.3e" % diff)

  # Fine solution is fixed point of Parareal iteration if u0coarse is provided when creating Parareal object
  def test_fineisfixedpointUcoarseprovided(self):
    self.setUp()        
    niter = np.random.randint(2,8)
    
    # Build coarse solution with matrix different from fine
    u0coarse = solution_linear(np.ones((self.ndof_c,1)), sparse.eye(self.ndof_c, format="csc"))
    
    para = parareal(self.tstart, self.tend, self.nslices, impeuler, impeuler, self.nfine, self.ncoarse, 0.0, niter, self.u0, u0coarse)

    '''
    Determine fine solution by solving M_f u = b
    '''
    Fmat = para.timemesh.get_fine_matrix(self.u0)
    b = np.zeros((self.ndof_f*(self.nslices+1),1))
    b[0:self.ndof_f,:] = self.u0.y
    # Solve system
    u = linalg.spsolve(Fmat, b)
    u = u.reshape((self.ndof_f*(self.nslices+1),1))

    # Get Parareal iteration matrices
    Pmat, Bmat = para.get_parareal_matrix()

    # Apply matrix to fine solution
    u_para = Pmat@u + Bmat@b
    diff = np.linalg.norm( u_para - u, np.inf)
    assert diff<1e-14, ("Fine solution is not a fixed point of Parareal iteration with provided Gmat matrix - difference %5.3e" % diff)
