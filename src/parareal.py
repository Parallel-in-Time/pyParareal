from timemesh import timemesh
from solution import solution
from scipy.sparse import linalg
from scipy import sparse
import copy
import numpy as np

class parareal(object):

    def __init__(self, tstart, tend, nslices, fine, coarse, nsteps_fine, nsteps_coarse, tolerance, iter_max, u0):
      assert isinstance(u0, solution), "Argument u0 must be an object of type solution"
      self.timemesh = timemesh(tstart, tend, nslices, fine, coarse, nsteps_fine, nsteps_coarse, tolerance, iter_max)
      self.u0 = u0

    def run(self):
      
      # Coarse predictor; need deepcopy to keep self.u0 unaltered
      self.timemesh.run_coarse(copy.deepcopy(self.u0))

      while True:
        
        # Run fine method
        self.timemesh.update_fine_all()

        for i in range(0,self.timemesh.nslices):

          # Compute difference F-G
          fine = copy.deepcopy( self.timemesh.get_fine_value(i) )
          fine.axpy(-1.0, self.timemesh.get_coarse_value(i))

          # Fetch update value from previous time slice
          if i==0:
            self.timemesh.set_initial_value(self.u0)
          else:
            self.timemesh.set_initial_value(copy.deepcopy(self.timemesh.get_end_value(i-1)), i)

          # Update coarse value
          self.timemesh.update_coarse(i)

          # Perform correction G_new + F_old - G_old
          fine.axpy(1.0, self.timemesh.get_coarse_value(i))

          # Set corrected value as new end value
          self.timemesh.set_end_value(fine, i)

        # increase iteration counter
        self.timemesh.increase_iter_all() 

        # stop loop if all slices have converged
        if self.timemesh.all_converged():
          break

    #
    # GET functions
    #

    # Returns matrices Pmat, Bmat such that a Parareal iteration is equivalent to
    # y_(k+1) = Pmat*y_k + Bmat*b
    # with b = (u0, 0, ..., 0) and u0 the initial value at the first time slice.
    def get_parareal_matrix(self, ucoarse=None):
      if ucoarse is None:
        Gmat = self.timemesh.get_coarse_matrix(self.u0)
      else:
        Gmat = self.timemesh.get_coarse_matrix(ucoarse)
      Fmat = self.timemesh.get_fine_matrix(self.u0)
      Bmat = sparse.linalg.inv(Gmat)
      # this is call is necessary because if Bmat has only 1 entry, it gets converted to a dense array here 
      Bmat = sparse.csc_matrix(Bmat)
      Pmat = Bmat.dot(Gmat-Fmat)
      return Pmat, Bmat

    # Returns the stability matrix for Parareal with fixed number of iterations
    def get_parareal_stab_function(self, k, ucoarse=None):
      e0 = np.zeros((self.timemesh.nslices+1,1))
      e0[0,:] = 1.0
      Mat = np.zeros((self.u0.ndof,self.u0.ndof), dtype='complex')
      Pmat, Bmat = self.get_parareal_matrix(ucoarse)
      Id = sparse.eye(self.u0.ndof*(self.timemesh.nslices+1), format="csc")

      # Selection matrix
      Zeros = np.zeros((self.u0.ndof,self.u0.ndof*self.timemesh.nslices))
      Idd   = sparse.eye(self.u0.ndof, format="csc")
      R = sparse.hstack((Zeros,Idd), format="csc")

      # Construct stability matrix from unit vectors
      for i in range(0,self.u0.ndof):
        y0 = np.zeros((self.u0.ndof,1))
        y0[i,0] = 1.0
        ee0 = np.kron(e0, y0)
        M = copy.deepcopy(Id)
        # Compute (sum(j=1,..n) Pmat^j) *Bmat
        for j in range(1,k+1):
          M += Pmat**j
        M = M.dot(Bmat)
        M = M.dot(ee0)
        Mat[:,i] = R.dot(M).flatten()
      return Mat

    # Returns the largest singular value of the error propagation matrix
    def get_max_svd(self, ucoarse=None):
      Pmat, Bmat = self.get_parareal_matrix(ucoarse)
      svds = linalg.svds(Pmat, k=1, tol=1e-6, return_singular_vectors=False)
      return svds[0]

    # Returns array containing all intermediate solutions
    def get_parareal_vector(self):
      b = np.zeros((self.u0.ndof*(self.timemesh.nslices+1),1))
      b[0:self.u0.ndof,:] = self.u0.y
      for i in range(0,self.timemesh.nslices):
        b[(i+1)*self.u0.ndof:(i+2)*self.u0.ndof,:] = self.timemesh.get_end_value(i).y
      return b

    # return end value of time slice i
    def get_end_value(self, i):
      return self.timemesh.get_end_value(i)

    # return end value of last time slice
    def get_last_end_value(self):
      return self.get_end_value(self.timemesh.nslices-1)
