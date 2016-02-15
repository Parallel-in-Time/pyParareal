from timemesh import timemesh
from solution import solution
from scipy.sparse import linalg
from scipy import sparse

class parareal(object):

    def __init__(self, tstart, tend, nslices, fine, coarse, nsteps_fine, nsteps_coarse, tolerance, iter_max, u0):
      assert isinstance(u0, solution), "Argument u0 must be an object of type solution"
      self.timemesh = timemesh(tstart, tend, nslices, fine, coarse, nsteps_fine, nsteps_coarse, tolerance, iter_max)
      self.u0 = u0

    def run(self):

      # Coarse predictor
      self.timemesh.run_coarse(self.u0)
      while not self.timemesh.all_converged():
        self.timemesh.increase_iter_all() 

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

    # Return fine value of last time slices
    def get_final_value(self):
      return self.timemesh.get_fine_value(self.timemesh.nslices-1)
