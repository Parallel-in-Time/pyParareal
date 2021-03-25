from solution import solution
import numpy as np
from scipy.sparse import linalg
from scipy import sparse

class solution_linear(solution):

  # Attention: at the moment, the M=None default needs to match the default in te superclass and I don't have a mechanism to enforce this automatically
  def __init__(self, y, A, M=None):
    super(solution_linear, self).__init__(y, M)
    self.A = sparse.csc_matrix(A)
    assert np.array_equal( np.shape(A), [self.ndof, self.ndof]), "A must be a matrix of size ndof x ndof where ndof is the number of entries in argument y"

  def f(self):
    self.y = self.A.dot(self.y)

  def solve(self, alpha):
    self.y = linalg.spsolve( self.getM() - alpha*self.A, self.y)
    #self.y, info = linalg.gmres( self.M-alpha*self.A, self.y, x0=self.y, tol=1e-14, restart=10, maxiter=500)
    self.y = np.reshape(self.y, (self.ndof,1))
