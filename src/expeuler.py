from integrator import integrator
from solution import solution

# Used to compute update matrices
from solution_linear import solution_linear
from scipy import sparse
import numpy as np
import copy

class expeuler(integrator):

  def __init__(self, tstart, tend, nsteps):
    super(expeuler, self).__init__(tstart, tend, nsteps)
    self.order = 1

  def run(self, u0):
    assert isinstance(u0, solution), "Initial value u0 must be an object of type solution"
    for i in range(0,self.nsteps):
      fu0 = copy.deepcopy(u0)
      fu0.f()
      u0.applyM()
      u0.axpy(self.dt, fu0)
      u0.solve(0.0)
    return u0

  #
  # For linear problems My' = A', a backward Euler update corresponds
  # to 
  # u_n+1 = (M - dt*A)^(-1)*M*u_n
  # so that a full update is [(M-dt*A)^(-1)*M]^nsteps
  def get_update_matrix(self, sol):
    assert isinstance(sol, solution_linear), "Update function can only be computed for solutions of type solution_linear"
    # Fetch matrices from solution and make sure they are sparse
    M = sparse.csc_matrix(sol.getM())
    A = sparse.csc_matrix(sol.A)
    Minv = sparse.linalg.inv(M)
    Minv = sparse.csc_matrix(Minv)
    Rmat = M + self.dt*A
    Rmat = Minv.dot(Rmat)
    # this is call is necessary because if Rmat has only 1 entry, it gets converted to a dense array here
    Rmat = sparse.csc_matrix(Rmat)
    Rmat = Rmat**self.nsteps
    return Rmat
