from scipy.optimize import NonlinearConstraint
from scipy.optimize import minimize
import scipy.sparse as sparse
from scipy.linalg import svdvals
import numpy as np

'''
Computes the epsilon pseudo spectral radius of a nil-potent matrix E where this can be done by maximizing the distance from the origin while staying on an eps-isoline.
'''
class pseudo_spectral_radius(object):

  def __init__(self, E, eps):
    assert isinstance(eps, float) and eps > 0, "Parameter eps must be a positive real number"
    self.eps = eps
    self.E   = E
    self.n   = np.shape(E)[0]
    assert self.n==np.shape(E)[1], "Matrix E must be square"
    # Note that the simple way of computing the PSR used here only works for nil-potent matrices where epsilon isolines are around the origin. 
    # The matrix E is nil-potent if and only if its spectrum is {0}, so check that it is
    eigval, dummy = np.linalg.eig(E.todense())
    assert abs(min(eigval))<1e-15 and abs(max(eigval))<1e-15, ("Provided matrix is not nil-potent, min(eigval): %5.2e, max(eigval): %5.2e" % (min(eigval), max(eigval)))
    self.Id  = sparse.identity(self.n)
  
  '''
  The constraint makes sure we are searching along an epsilon isoline in the pseudo-spectrum
  '''
  def constraint(self, x):
      z = x[0] + 1j*x[1]
      # See algorithm on p. 371, Chapter 39 in the Trefethen book
      M = z*self.Id - self.E
      sv = svdvals(M.todense())
      return np.min(sv)
    
  '''
  Minimise 1/||x||_2 while the constraint keeps us on the isoline with maximise distance from the origin to yield the spectral radius.
  '''
  def target(self,x):
    return 1.0/np.linalg.norm(x, 2)**2

  '''
  Computes the epsilon-pseudospectral radius of the matrix E.
  '''
  def get_psr(self, verbose=False):
    # The constraint will keep us on an isoline where the minimum singular value of z I - E equals eps, meaning that || (z I - E)^(-1) ||_2 = 1\eps - we allow for a bit of variation (1e-9) to help with convergence of the optimiser
    nlc   = NonlinearConstraint(self.constraint, self.eps-1e-9, self.eps+1e-9)
    # Now run the minimiser to minimise 1/||x||_2^2 while keeping min(svd(zI-E))=eps
    result = minimize(self.target, [np.sqrt(self.eps), np.sqrt(self.eps)], constraints=nlc, tol = 1e-10, method='trust-constr', options = {'xtol': 1e-10, 'gtol': 1e-10, 'maxiter': 500})
    # The eps - pseudo spectral radius corresponds to the maximum distance from the origin on the eps isoline of min(svd(zI-E))
    if verbose:
      print("Message returned by minimize function: \n")
      print(result.message)
      print("Constraint at solution (should equal provided eps): %5.3f" % self.constraint(result.x))
      print("Target at solution:     %5.3f" % self.target(result.x))
    return np.linalg.norm(result.x,2), result.x, self.target(result.x), self.constraint(result.x)
