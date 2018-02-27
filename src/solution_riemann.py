import numpy as np
import scipy as sp
from scipy import sparse

# Class representing the solution of an IVP
# M*y' = f(y)
# with y being an array.

class solution_riemann(object):

  def __init__(self, y, M=0): 
    assert isinstance(y, np.ndarray), "Argument y must be of type numpy.ndarray"
    assert np.shape(y)[0]==np.size(y), "Argument y must be a linear array"
    # If y is a purely 1D array, reshape it into a Nx1 2D array... if both types are mixed, horrible inconsistencies arise
    self.y    = np.reshape(y, (np.shape(y)[0], 1))
    self.ndof = np.size(y)
    if isinstance(M,int):
      self.M = sparse.eye(self.ndof, format="csc")
    else:
      self.M = M
      assert np.array_equal( np.shape(M), [self.ndof, self.ndof]), "Matrix M does not match size of argument y"
    self.A = np.array([1])

  def riemann_solver(self, u_left, u_right, jacobian):
    # find eigenvalue decomposition of Jacobian
    S, Q = np.linalg.eig(jacobian)
    
    # transform left and right value into eigencoordinates
    u_eigcoord_left  = (np.linalg.inv(Q)).dot(u_left)
    u_eigcoord_right = (np.linalg.inv(Q)).dot(u_right)

  # Overwrite y with a*x+y
  def axpy(self, a, x):
    assert (np.size(a)==1 or isinstance(a, float)), "Input a must be a scalar"
    # Ask for foregiveness instead of permission...
    try:
      self.y = a*x.y + self.y
    except:
      assert isinstance(x, solution), "Input x must be an object of type solution"
      assert x.ndof==self.ndof, "Number of degrees of freedom is different in x than in this solution object"
      raise Exception('Unknown error in solution.axpy')

  # Overwrite y with f(y)
  def f(self):
    fluxes = np.zeros(self.ndof, dtype=np.complex)
    for j in range(0,self.ndof+1):
      if j==0:
        pass
      elif j==self.ndof:
        pass
      else:
        u_interface = self.riemann_solver(self.y[j-1], self.y[j], self.A)
        fluxes[j] = self.A.dot(u_interface)
      
  # No mass matrix in FVM
  def applyM(self):
    pass

  # Overwrite y with solution of M*y-alpha*f(y) = y
  def solve(self, alpha):
    if not alpha==0.0:
      raise NotImplementedError("No implicit methods implemented for FVM/Riemann solver")
    else:
      # no mass matrix, so nothing to do
      pass

  # Return inf norm of y
  def norm(self):
    return np.linalg.norm(self.y, np.inf)
