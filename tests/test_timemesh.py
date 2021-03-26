import sys
sys.path.append('../src')

from impeuler import impeuler
from timemesh import timemesh
from solution_linear import solution_linear
import unittest
import numpy as np
from scipy.sparse import linalg
from scipy import sparse


class TestTimemesh(unittest.TestCase):

  def setUp(self):
    times        = np.sort( np.random.rand(2) )
    self.tstart  = times[0]
    self.tend    = times[1]
    self.nslices = np.random.randint(3,16)
    steps        = np.sort( np.random.randint(low=1, high=32, size=2) )
    self.ncoarse = steps[0]
    self.nfine   = steps[1]
    self.ndofs           = [np.random.randint(6,32), np.random.randint(6,32)]
    self.ndof_c          = np.min(self.ndofs)
    self.ndof_f          = np.max(self.ndofs)
    self.u0_f            = np.random.rand(self.ndof_f)
    self.A_f             = np.random.rand(self.ndof_f, self.ndof_f)
    self.u0fine          = solution_linear(self.u0_f, self.A_f)
    self.u0_c            = np.random.rand(self.ndof_c)
    self.A_c             = np.random.rand(self.ndof_c, self.ndof_c)
    self.u0coarse        = solution_linear(self.u0_c, self.A_c)
        
  # timemesh class can be instantiated
  def test_caninstantiate(self):
    tm = timemesh(self.tstart, self.tend, self.nslices, impeuler, impeuler, self.nfine, self.ncoarse, 1e-10, 5, self.u0fine, self.u0coarse)

  # timemesh class can be instantiated with matrix as argument for coarse
  def test_caninstantiatewithmatrix(self):
    mat = sparse.eye(1)
    tm = timemesh(self.tstart, self.tend, self.nslices, impeuler, mat, self.nfine, self.ncoarse, 1e-10, 5, self.u0fine, self.u0coarse)
     
  # initial value can be set for first time slice
  def test_cansetinitial(self):
    tm = timemesh(self.tstart, self.tend, self.nslices, impeuler, impeuler, self.nfine, self.ncoarse, 1e-10, 5, self.u0fine, self.u0coarse)
    tm.set_initial_value(self.u0fine)

  # initial value can be set for first time slice
  def test_cansetinitialtimeslice(self):
    u0 = solution_linear(np.array([1.0]), np.array([[-1.0]]))
    tm = timemesh(self.tstart, self.tend, self.nslices, impeuler, impeuler, self.nfine, self.ncoarse, 1e-10, 5, self.u0fine, self.u0coarse)
    tm.set_initial_value(self.u0fine, 3)

  # fails to set initial value for time slice with too large index
  def test_toohightimesliceindexthrows(self):
    tm = timemesh(self.tstart, self.tend, self.nslices, impeuler, impeuler, self.nfine, self.ncoarse, 1e-10, 5, self.u0fine, self.u0coarse)
    with self.assertRaises(AssertionError):
      tm.set_initial_value(self.u0fine, self.nslices+1)

  # all_converged gives false if called directly after initialisation unless the maximum number of iterations is zero
  def test_allconvergedthrows(self):
    tm = timemesh(self.tstart, self.tend, self.nslices, impeuler, impeuler, self.nfine, self.ncoarse, 1e-10, 5, self.u0fine, self.u0coarse)
    assert not tm.all_converged(), "all_converged should not be true directly after initialisation"

  def test_allconvergedzeromaxiter(self):
    tm = timemesh(self.tstart, self.tend, self.nslices, impeuler, impeuler, self.nfine, self.ncoarse, 1e-10, 0, self.u0fine, self.u0coarse)
  
    # to allow computing residuals, set initial value, run fine and set end value
    sol_start   = solution_linear( np.zeros(self.ndof_f), np.eye(self.ndof_f) )
    
    for i in range(0,self.nslices):
      tm.set_initial_value(sol_start, i)
      tm.slices[i].update_fine()
    # Since iter_max=0, all_converged should return True even though the residual is larger than 1e-10
    assert tm.get_max_residual()>1e-10, "Maximum residual is smaller than tolerance, even though set up not to be"    
    assert tm.all_converged, "For iter_max=0, all time slices should be considered converged"

  # raising iteration counter leads to convergence
  def test_raiseiterconvergence(self):
    tm = timemesh(self.tstart, self.tend, self.nslices, impeuler, impeuler, self.nfine, self.ncoarse, 1e-10, 2, self.u0fine, self.u0coarse)
    sol_start   = solution_linear( np.zeros(self.ndof_f), np.eye(self.ndof_f) )

    for i in range(0,self.nslices):
      tm.set_initial_value(sol_start, i)
      tm.slices[i].update_fine()
    # since iter_max=2, time slices should not yet have converged
    assert not tm.all_converged(), "All time slices have converged even though initialised not to be"
    # increase iteration counter 
    for i in range(0,self.nslices):
      tm.increase_iter(i)
    # For max_iter=2, timeslices should not yet have converged
    assert not tm.all_converged(), "Raising iteration counter once with iter_max should not lead to convergence"
    # increase iteration counter again
    for i in range(0,self.nslices):
      tm.increase_iter(i)
    assert tm.all_converged(), "Raising iteration counter through increase_iter did not lead to convergence"

  # get_fine_matrix is callable
  def test_finematrixcallable(self):
    tm = timemesh(self.tstart, self.tend, self.nslices, impeuler, impeuler, self.nfine, self.ncoarse, 1e-10, 5, self.u0fine, self.u0coarse)
    Mat = tm.get_fine_matrix(self.u0fine)

  # get_coarse_matrix is callable
  def test_coarsematrixcallable(self):
    tm = timemesh(self.tstart, self.tend, self.nslices, impeuler, impeuler, self.nfine, self.ncoarse, 1e-10, 5, self.u0fine, self.u0coarse)
    Mat = tm.get_coarse_matrix(self.u0coarse)
    
  # run_coarse is callable and provides expected output at very end
  def test_runcoarse(self):
    tm = timemesh(self.tstart, self.tend, self.nslices, impeuler, impeuler, self.nfine, self.ncoarse, 1e-10, 5, self.u0fine, self.u0coarse)
    
    # Run the coarse propagator but with the initial value given by u0fine
    tm.run_coarse(self.u0fine)

    # get the matrix that represents the coarse propagator but then apply it to propagate the initial value u0fine
    Mat = tm.get_coarse_matrix(self.u0coarse)
    b = np.zeros((self.nslices+1)*self.ndof_f)
    b[0:self.ndof_f] = self.u0_f
    u = linalg.spsolve(Mat, b)
    # get the last ndof_f elements; then convert into column vector
    uend = np.atleast_2d(u[-self.ndof_f:]).T
    err = np.linalg.norm(uend - tm.get_coarse_value(self.nslices-1).y, np.inf)
    assert err<1e-10, ("run_coarse and successive application of update matrix does not give identical results - error: %5.3e" % err)

    
  # with coarse method provided as matrix, run_coarse is callable and provides expected output at very end
  def test_runcoarsewithmatrix(self):
    dt = (self.tend - self.tstart)/(float(self.nslices)*float(self.ncoarse))
    mat = 1.0/(1.0 - dt)*sparse.eye(self.ndof_c, format="csc")
    
    # passing a matrix as coarse propagator creates a special_integrator that uses the matrix as stability function; in this case, the matrix that defines the problem is not used at all.
    tm = timemesh(self.tstart, self.tend, self.nslices, impeuler, mat, self.nfine, self.ncoarse, 1e-10, 5, self.u0fine, self.u0coarse)
    tm.run_coarse(self.u0fine)

    Mat = tm.get_coarse_matrix(self.u0coarse)
    b = np.zeros((self.nslices+1)*self.ndof_f)
    b[0:self.ndof_f] = self.u0_f
    u = linalg.spsolve(Mat, b)
    uend = np.atleast_2d(u[-self.ndof_f:]).T
    err = np.linalg.norm(uend - tm.get_coarse_value(self.nslices-1).y, np.inf)
    assert err<1e-10, ("for coarse provided as matrix, run_coarse and successive application of update matrix does not give identical results - error: %5.3e" % err)

  # run_fine is callable and provides expected output at very end
  def test_runfine(self):
    tm = timemesh(self.tstart, self.tend, self.nslices, impeuler, impeuler, self.nfine, self.ncoarse, 1e-10, 5, self.u0fine, self.u0coarse)
    
    tm.run_fine(self.u0fine)
    Mat = tm.get_fine_matrix(self.u0fine)
    
    b = np.zeros((self.nslices+1)*self.ndof_f)
    b[0:self.ndof_f] = self.u0_f
    u = linalg.spsolve(Mat, b)
    uend = np.atleast_2d(u[-self.ndof_f:]).T
    err = np.linalg.norm(uend - tm.get_fine_value(self.nslices-1).y, np.inf)
    assert err<1e-10, ("run_fine and successive application of update matrix does not give identical results - error: %5.3e" % err)

  # run_coarse provides expected intermediate values
  def test_runcoarseintermediate(self):
    tm = timemesh(self.tstart, self.tend, self.nslices, impeuler, impeuler, self.nfine, self.ncoarse, 1e-10, 5, self.u0fine, self.u0coarse)
    tm.run_coarse(self.u0fine)
    
    Mat = tm.slices[0].int_coarse.get_update_matrix(self.u0coarse)
    Mat = tm.slices[0].meshtransfer.Imat@Mat
    Mat = Mat@tm.slices[0].meshtransfer.Rmat
    
    b = self.u0_f
    for i in range(0,self.nslices):
      # matrix update to end of slice
      b = Mat@b
      uend = np.atleast_2d(tm.get_coarse_value(i).y).T
      err = np.linalg.norm( uend - b, np.inf )
      assert err<1e-10, ("Successive application of update matrix does not reproduce intermediata coarse values generated by run_coarse. Error: %5.3e in slice %2i" % (err, i))
    
  # with coarse provided as matrix, run_coarse provides expected intermediate values
  def test_runcoarseintermediatewithmatrix(self):
    # need at least three slices for this test
    self.nslices = min(self.nslices, 3)
    dt = (self.tend - self.tstart)/(float(self.nslices)*float(self.ncoarse))
    
    mat = 1.0/(1.0 - dt)*sparse.eye(self.ndof_c, format="csc")
    
    tm = timemesh(self.tstart, self.tend, self.nslices, impeuler, mat, self.nfine, self.ncoarse, 1e-10, 5, self.u0fine, self.u0coarse)
    tm.run_coarse(self.u0fine)
    
    Mat = tm.slices[0].int_coarse.get_update_matrix(self.u0coarse)
    Mat = tm.slices[0].meshtransfer.Imat@Mat
    Mat = Mat@tm.slices[0].meshtransfer.Rmat
    
    b = self.u0_f
    for i in range(0,self.nslices):
      # matrix update to end of slice
      b = Mat@b
      uend = np.atleast_2d(tm.get_coarse_value(i).y).T
      err = np.linalg.norm( uend - b, np.inf )
      assert err<2e-12, ("Successive application of update matrix does not reproduce intermediata coarse values generated by run_coarse. Error: %5.3e in slice %2i" % (err, i))

  # run_fine provides expected intermediate values
  def test_runfineintermediate(self):
    tm = timemesh(self.tstart, self.tend, self.nslices, impeuler, impeuler, self.nfine, self.ncoarse, 1e-10, 5, self.u0fine, self.u0coarse)
    tm.run_fine(self.u0fine)
    Mat = tm.slices[0].int_fine.get_update_matrix(self.u0fine)
    b = self.u0_f
    for i in range(0,self.nslices):
      # matrix update to end of slice
      b = Mat@b
      uend = np.atleast_2d(tm.get_fine_value(i).y).T
      err = np.linalg.norm( uend - b, np.inf )
      assert err<1e-10, ("Successive application of update matrix does not reproduce intermediata coarse values generated by run_fine. Error: %5.3e in slice %2i" % (err, i))
