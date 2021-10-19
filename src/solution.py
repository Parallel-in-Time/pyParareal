import numpy as np
import scipy as sp
from scipy import sparse

# Class representing the solution of an IVP
# M*y' = f(y)
# with y being an array.

class solution(object):

  def __init__(self, y, M=None):
    assert isinstance(y, np.ndarray), "Argument y must be of type numpy.ndarray"
    assert np.shape(y)[0]==np.size(y), "Argument y must be a linear array"
    # If y is a purely 1D array, reshape it into a Nx1 2D array... if both types are mixed, horrible inconsistencies arise
    self.y    = np.reshape(y, (np.shape(y)[0], 1))
    self.ndof = np.size(y)
    self.M = M
    if not (self.M is None):
      assert np.array_equal( np.shape(M), [self.ndof, self.ndof]), "Matrix M does not match size of argument y"

  # Overwrite y with a*x+y
  def axpy(self, a, x):
    assert (np.size(a)==1 or isinstance(a, float)), "Input a must be a scalar"
    # Ask for foregiveness instead of permission...
    try:
      self.y = a*x.y + self.y
      # BUG : the following lines (that should be equivalent since axpy
      # overwrites y values) do not pass the tests
      # self.y += a*x.y
      # np.copyto(self.y, a*x.y + self.y)
    except:
      assert isinstance(x, solution), "Input x must be an object of type solution"
      assert x.ndof==self.ndof, "Number of degrees of freedom is different in x than in this solution object"
      raise Exception('Unknown error in solution.axpy')

  # Overwrite y with f(y)
  def f(self):
    raise NotImplementedError("Function f in generic solution not implemented: needs to be overloaded in derived class")

  # Overwrite y with My
  def applyM(self):
    if not (self.M is None):
      np.copyto(self.y, self.M.dot(self.y))
    # else do nothing as this assume M is the identiy

  # Overwrite y with solution of M*y-alpha*f(y) = y
  def solve(self, alpha):
    raise NotImplementedError("Function solve in generic solution not implemented: needs to be overloaded in derived class")

  # Return inf norm of y
  def norm(self):
    return np.linalg.norm(self.y, np.inf)

  # Return mass matrix
  def getM(self):
    if self.M is None:
      return sparse.eye(self.ndof, format="csr")
    else:
      return self.M

  # Apply matrix
  def apply_matrix(self, A):
    assert np.shape(A)[1] == self.ndof, "Number of columns in argument matrix A does not match the number of DoF for this solution"
    self.y = A@self.y
