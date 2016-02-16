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
    for i in range(0,self.nsteps):
      u0.y = M.dot(u0.y)

  def get_update_matrix(self, sol):
    assert isinstance(sol, solution_linear), "Update function can only be computed for solutions of type solution_linear"
    # Fetch matrices from solution and make sure they are sparse
    M = sparse.csc_matrix(sol.M)
    A = sparse.csc_matrix(sol.A)
    Minv = sparse.linalg.inv(M)
    return sparse.linalg.expm(Minv.dot(A)*self.dt)
