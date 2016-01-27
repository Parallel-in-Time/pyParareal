import numpy as np
import scipy as sp

# Class representing the solution of an IVP
# M*y' = f(y)
# with y being an array.

class solution:

  def __init__(self, y, M=0): 
    assert isinstance(y, np.ndarray)
    assert y.ndim==1, "Input y must be a one dimensional array"
    self.y    = y
    self.ndof = np.size(y)
    if isinstance(M,int):
      self.M = sp.eye(self.ndof)
    else:
      self.M = M
      assert np.array_equal( np.shape(M), [self.ndof, self.ndof]), ".."

  # Overwrite y with a*x+y
  def axpy(self, a, x):
    assert a.ndim==1 and np.size(a)==1, "Input a must be a scalar"
    assert x.ndim==1 and np.size(x)==self.ndof, "Input x must be a vector of same length as y"
    self.y = a*x + self.y

  # Overwrite y with f(y)
  def f(self):
    return 0.0

  # Overwrite y with solution of M*y-alpha*f(y) = b
  def solve(self, alpha, b):
    return 0.0

  # Return inf norm of y
  def norm(self):
    return np.linalg.norm(self.y, np.inf)
