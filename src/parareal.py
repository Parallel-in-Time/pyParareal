from timemesh import timemesh
from solution import solution
from scipy.sparse import linalg
from scipy import sparse
from scipy.special import factorial, binom
import copy
import numpy as np

class parareal(object):

    def __init__(self, tstart, tend, nslices, fine, coarse, nsteps_fine, nsteps_coarse, tolerance, iter_max, u0, u0coarse = None):
        assert isinstance(u0, solution), "Argument u0 must be an object of type solution"
        self.u0 = u0
        if (u0coarse is None):
            self.u0coarse = copy.deepcopy(u0)
        else:
            self.u0coarse = u0coarse
        self.timemesh = timemesh(tstart, tend, nslices, fine, coarse, nsteps_fine, nsteps_coarse, tolerance, iter_max, self.u0, self.u0coarse)

    '''
    Execute the Parareal iteration
    '''
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
    # Properties
    #

    @property
    def eigvals_fine(self):
        """Eigenvalue with largest abs. value for fine solver Jacobian matrix"""
        int_fine = self.timemesh.slices[0].int_fine
        jac = int_fine.get_update_matrix(self.u0).todense()
        return np.linalg.eigvals(jac)

    @property
    def eigvals_coarse(self):
        """Eigenvalue with largest abs. value for coarse solver Jacobian matrix"""
        int_coarse = self.timemesh.slices[0].int_coarse
        jac = int_coarse.get_update_matrix(self.u0).todense()
        return np.linalg.eigvals(jac)

    #
    # GET functions
    #

    # Returns matrices E, Mginv such that a Parareal iteration is equivalent to
    # y_(k+1) = E*y_k + Mginv*b
    # with b = (u0, 0, ..., 0) and u0 the initial value at the first time slice.
    def get_parareal_matrix(self, ucoarse=None):
      if ucoarse is None:
        Gmat = self.timemesh.get_coarse_matrix(self.u0coarse)
      else:
        Gmat = self.timemesh.get_coarse_matrix(ucoarse)
      Fmat = self.timemesh.get_fine_matrix(self.u0)
      Bmat = sparse.linalg.inv(Gmat)
      # this is call is necessary because if Bmat has only 1 entry, it gets converted to a dense array here
      Bmat = sparse.csc_matrix(Bmat)
      Emat = Bmat.dot(Gmat-Fmat)
      return Emat, Bmat

    # Returns the stability matrix for Parareal with fixed number of iterations
    def get_parareal_stab_function(self, k, ucoarse=None):
      e0         = np.zeros((self.timemesh.nslices+1,1))
      e0[0,:]    = 1.0
      Emat, Bmat = self.get_parareal_matrix(ucoarse)
      Id         = sparse.eye(self.u0.ndof*(self.timemesh.nslices+1), format="csc")

      # Selection matrices
      Zeros = np.zeros((self.u0.ndof,self.u0.ndof*self.timemesh.nslices))
      Idd   = sparse.eye(self.u0.ndof, format="csc")
      C1    = sparse.hstack((Zeros,Idd), format="csc")
      Zeros = np.zeros((self.u0.ndof*self.timemesh.nslices, self.u0.ndof))
      C2    = sparse.vstack((Idd, Zeros), format="csc")

      E_power_k = copy.deepcopy(Id)
      # Compute (sum(j=1,..n) Pmat^j) *Bmat
      for j in range(1,k+1):
        E_power_k += Emat**j

      Mat = C1.dot(E_power_k.dot(Bmat.dot(C2)))
      return Mat.todense()

    def get_linear_bound(self, nIter, mgritTerm=False):
        """Linear bound from M.J Gander"""
        lamF, lamG = self.eigvals_fine, self.eigvals_coarse
        gamma = np.abs(lamF - lamG)/(1-np.abs(lamG))
        if mgritTerm:
            N = self.timemesh.nslices
            gamma *= 1 - np.abs(lamG)**(N-1)
        gamma = np.max(gamma)
        bound = [gamma**k for k in range(nIter+1)]
        return np.array(bound)

    def get_superlinear_bound(self, nIter, bruteForce=False):
        """Superlinear bound from Generating Function Method"""
        lamF, lamG = self.eigvals_fine, self.eigvals_coarse
        alpha = np.max(np.abs(lamF - lamG))
        beta = np.max(np.abs(lamG))
        N = self.timemesh.nslices
        if bruteForce:
            bound = [alpha**k * sum([binom(i+k-1,i) * beta**i
                                     for i in range(N-k)])
                     for k in range(nIter+1)]
            # Simplification of the binom term, but there is still an error ...
            # bound = [1] + [alpha**k / factorial(k-1) * sum(
            #             [np.prod([(i+l)*beta**i for l in range(1, k)])
            #              for i in range(N-k)])
            #          for k in range(1, nIter+1)]
        else:
            # Bounding beta by 1, simplification used by M.J Gander
            bound = [alpha**k / factorial(k) * np.prod(
                        [N-j for j in range(1, k)])
                     for k in range(nIter+1)]
        return np.array(bound)


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
