import numpy as np
import scipy as sp
from scipy import sparse

# Class representing the solution of an IVP
# M*y' = f(y)
# with y being an array.

class solution(object):

  def __init__(self, y, M=0): 
    assert isinstance(y, np.ndarray), "Argument y must be of type numpy.ndarray"
    assert np.shape(y)[0]==np.size(y), "Argument y must be a linear array"
    # If y is a purely 1D array, reshape it into a Nx1 2D array... if both types are mixed, horrible inconsistencies arise
    self.y    = np.reshape(y, (np.shape(y)[0], 1))
    self.ndof = np.size(y)
    if isinstance(M,int):
      self.M = sp.eye(self.ndof)
    else:
      self.M = M
      assert np.array_equal( np.shape(M), [self.ndof, self.ndof]), "Matrix M does not match size of argument y"

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
    raise NotImplementedError("Function f in generic solution not implemented: needs to be overloaded in derived class")

  # Overwrite y with My
  def applyM(self):
    self.y = self.M.dot(self.y)

  # Overwrite y with solution of M*y-alpha*f(y) = y
  def solve(self, alpha):
    raise NotImplementedError("Function solve in generic solution not implemented: needs to be overloaded in derived class")

  # Return inf norm of y
  def norm(self):
    return np.linalg.norm(self.y, np.inf)
