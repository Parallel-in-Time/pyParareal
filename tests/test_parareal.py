import sys
sys.path.append('../src')

from parareal import parareal
from timemesh import timemesh
from impeuler import impeuler
from solution_linear import solution_linear
import unittest
import numpy as np
from scipy import sparse

class TestParareal(unittest.TestCase):

  def setUp(self):
    times        = np.sort( np.random.rand(2) )
    self.tstart  = times[0]
    self.tend    = times[1]
    self.nslices = np.random.randint(2,64) 
    steps        = np.sort( np.random.randint(low=1, high=128, size=2) )
    self.ncoarse = steps[0]
    self.nfine   = steps[1]
    self.ndof    = np.random.randint(1,16)
    #self.ndof  = 2
    #self.nslices = 2
    self.A       = sparse.spdiags([ np.ones(self.ndof), -2.0*np.ones(self.ndof), np.ones(self.ndof)], [-1,0,1], self.ndof, self.ndof, format="csc")
    self.M       = sparse.spdiags([ np.random.rand(self.ndof) ], [0], self.ndof, self.ndof, format="csc")
    self.u0      = solution_linear(np.ones(self.ndof), self.A, self.M)

  # Can instantiate object of type parareal
  def test_caninstantiate(self):
    para = parareal(self.tstart, self.tend, self.nslices, impeuler, impeuler, self.nfine, self.ncoarse, 1e-10, 5, self.u0)

  # Can execute run function
  def test_canrun(self):
    para = parareal(self.tstart, self.tend, self.nslices, impeuler, impeuler, self.nfine, self.ncoarse, 1e-10, 5, self.u0)
    para.run()  

  # Test matrix Parareal
  def test_pararealmatrix(self):
    para = parareal(self.tstart, self.tend, self.nslices, impeuler, impeuler, self.nfine, self.ncoarse, 1e-10, 1, self.u0)
    Pmat, Bmat = para.get_parareal_matrix()
    bvec = np.zeros((self.ndof*(self.nslices+1),1))
    bvec[0:self.ndof,:] = self.u0.y
    # Perform one coarse step by matrix multiplication
    y0 = Bmat.dot(bvec)
    # Perform one Parareal step in matrix form
    y_mat = Pmat.dot(y0) + Bmat.dot(bvec)
    para.run()
    y_para = np.zeros((self.ndof*(self.nslices+1),1))
    y_para[0:self.ndof,:] = self.u0.y
    for i in range(0,self.nslices):
      y_para[(i+1)*self.ndof:(i+2)*self.ndof,:] = para.get_end_value(i).y
    err = np.linalg.norm(y_para - y_mat, np.inf)
    assert err<1e-12, ("Parareal run and matrix form do not yield identical results for a single iteration. Error: %5.3e" % err)

  # Test matrix Parareal
  def test_pararealmatrixmultiple(self):
    niter = np.random.randint(2,8) 
    para = parareal(self.tstart, self.tend, self.nslices, impeuler, impeuler, self.nfine, self.ncoarse, 0.0, niter, self.u0)
    Pmat, Bmat = para.get_parareal_matrix()
    bvec = np.zeros((self.ndof*(self.nslices+1),1))
    bvec[0:self.ndof,:] = self.u0.y
    # Perform one coarse step by matrix multiplication
    y_mat = Bmat.dot(bvec)
    # Perform niter Parareal step in matrix form
    for i in range(0,niter):
      y_mat = Pmat.dot(y_mat) + Bmat.dot(bvec)
    para.run()
    y_para = np.zeros((self.ndof*(self.nslices+1),1))
    y_para[0:self.ndof,:] = self.u0.y
    for i in range(0,self.nslices):
      y_para[(i+1)*self.ndof:(i+2)*self.ndof,:] = para.get_end_value(i).y
    err = np.linalg.norm(y_para - y_mat, np.inf)
    assert err<1e-12, ("Parareal run and matrix form do not yield identical results for multiple iterations. Error: %5.3e" % err)
