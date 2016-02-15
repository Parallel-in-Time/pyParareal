from timemesh import timemesh
from solution import solution
from scipy.sparse import linalg
from scipy import sparse
import copy

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
    def get_parareal_matrix(self):
      Gmat = self.timemesh.get_coarse_matrix(self.u0)
      Fmat = self.timemesh.get_fine_matrix(self.u0)      
      Bmat = sparse.linalg.inv(Gmat)
      # this is call is necessary because if Bmat has only 1 entry, it gets converted to a dense array here 
      Bmat = sparse.csc_matrix(Bmat)
      Pmat = Bmat.dot(Gmat-Fmat)
      return Pmat, Bmat

    # return end value of time slice i
    def get_end_value(self, i):
      return self.timemesh.get_end_value(i)
