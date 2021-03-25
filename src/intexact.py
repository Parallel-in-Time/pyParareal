from integrator import integrator
from solution_linear import solution_linear

# Used to compute update matrices
from solution_linear import solution_linear
from scipy import sparse
import numpy as np

class intexact(integrator):

  def __init__(self, tstart, tend, nsteps):
    super(intexact, self).__init__(tstart, tend, nsteps)

  def run(self, u0):
    assert isinstance(u0, solution_linear), "Exact integrator intexact can only be used for solution_linear type initial values"
    M = self.get_update_matrix(u0)
    u0.y = M.dot(u0.y)

  def get_update_matrix(self, sol):
    assert isinstance(sol, solution_linear), "Update function can only be computed for solutions of type solution_linear"
    # Fetch matrices from solution and make sure they are sparse
    M = sparse.csc_matrix(sol.getM())
    A = sparse.csc_matrix(sol.A)
    Minv = sparse.linalg.inv(M)
    # this is call is necessary because if Rmat has only 1 entry, it gets converted to a dense array here
    Minv = sparse.csc_matrix(Minv)
    # WEIRD BUG IN sparse.linalg.expm ... FOR NOW, USE SCALAR VERSION INSTEAD
  #    Mat = sparse.linalg.expm(Minv.dot(A)*self.dt)
    if sol.ndof>1:
      raise NotImplementedError("Because of a weird bug in scipys expm function, intexact is for the moment restricted to scalar problems")
    Mat = Minv.dot(A)*self.dt
    Mat = Mat[0,0]
    Mat = np.exp(Mat)**self.nsteps
    return sparse.csc_matrix(np.array([[Mat]], dtype='complex'))
