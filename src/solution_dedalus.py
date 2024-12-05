from solution_linear import solution_linear
import dedalus.public as d3

import numpy as np
from scipy.sparse import linalg
from scipy import sparse

'''
While Dedalus can of course solve nonlinear problems, we currently only use it to produce the spectral derivative matrices that 
arise when solving the linear transport equation. Therefore, it derives from the solution_linear class so that we can use the 
get_update_matrix function in the integrator_dedalus.
'''
class solution_dedalus(solution_linear):

  # Attention: at the moment, the M=None default needs to match the default in te superclass and I don't have a mechanism to enforce this automatically
  def __init__(self, y, n):
    self.n = n
    xcoord = d3.Coordinate('x')
    dist = d3.Distributor(xcoord, dtype=np.complex128)
    # Warning: the code, also in the meshtransfer class, is hardcoded to operate on the unit interval [0,1]
    xbasis = d3.ComplexFourier(xcoord, size=self.n, bounds=(0.0, 1.0), dealias=3/2)
    self.x = dist.local_grid(xbasis)
    u = dist.Field(name='u', bases=xbasis)
    dx = lambda A: d3.Differentiate(A, xcoord)
    t_field = dist.Field()

    # Define advection problem with constant advection speed equal to 1
    self.problem = d3.IVP([u], time=t_field, namespace={"u": u, "dx": dx})
    # Note that since dx(u) appears on the left side of the equal sign, it will be integrated implicitly.
    self.problem.add_equation("dt(u) + dx(u) = 0")    
    
    ### To allow to run a solver on this solution, need to write y into a Dedalus solution object somehow

    # NOTE: this should pass the spectral differentiation matrix in space only to the superclass solution_linear! It does not because,
    # at the moment, the Dedalus spatial discretization is never mixed with a non-Dedalus timestepper. If this is required, this needs changing.
    super(solution_dedalus, self).__init__(y, np.zeros((self.n, self.n)))

  def f(self):
    raise NotImplementedError("solution_dedalus can only be used to produce the matrix form of an integrator but cannot be run normally")

  def solve(self, alpha):
    raise NotImplementedError("solution_dedalus can only be used to produce the matrix form of an integrator but cannot be run normally")
