import numpy as np
import scipy as sp
from scipy import sparse
from solution import solution

# Class representing the solution of an IVP
# M*y' = f(y)
# with y being an array.


### PROBLEM DEFINITION ###
def flux_burgers(u):
  return u*u

def riemann_solver_burgers(u_left, u_right):
  if (u_left > u_right):
    # Shock: compute shock speed
    s = 0.5*(u_left + u_right)
    if (s>0):
      # rightward traveling shock
      return u_left
    else:
      # leftward traveling shock
      return u_right
  else:
    # Rarefaction wave
    if ((u_left > 0) and (u_right > 0)):
      return u_left
    elif ((u_left < 0) and (u_right < 0)):
      return u_right
    else:
      return 0.0

# ...can make this function static?
def riemann_solver(self, u_left, u_right, jacobian):
  # find eigenvalue decomposition of Jacobian
  S, Q = np.linalg.eig(jacobian)
  
  # transform left and right value into eigencoordinates
  u_eigcoord_left  = (np.linalg.inv(Q)).dot(u_left)
  u_eigcoord_right = (np.linalg.inv(Q)).dot(u_right)


class solution_riemann(solution):

  def __init__(self, y, dx):
    assert isinstance(y, np.ndarray), "Argument y must be of type numpy.ndarray"
    assert np.shape(y)[0]==np.size(y), "Argument y must be a linear array"
    # If y is a purely 1D array, reshape it into a Nx1 2D array... if both types are mixed, horrible inconsistencies arise
    self.y    = np.reshape(y, (np.shape(y)[0], 1))
    self.ndof = np.size(y)
    # No mass matrix in FVM
    self.M = sparse.eye(self.ndof, format="csc")
    self.A = np.array([1])
    self.ncomponents  = 1
    # number of entries per component... should have ncomponents*nx = ndof
    self.nx = self.ndof/self.ncomponents
    assert isinstance(self.nx, (np.integer, int)), "Mismatch in nx, ncomponents and ndof"
    # Now reshape into ((nx, ncomponents)) array
    self.y = np.reshape(self.y, ((self.nx, self.ncomponents)))
    self.dx = dx
    
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
    fluxes = np.zeros(self.ndof+1)
    for j in range(0,self.ndof+1):
      if j==0:
        u_interface = riemann_solver_burgers(self.y[-1], self.y[0])
      elif j==self.ndof:
        u_interface = riemann_solver_burgers(self.y[-1], self.y[0])
      else:
        u_interface = riemann_solver_burgers(self.y[j-1], self.y[j])
      fluxes[j] = flux_burgers(u_interface)
    for j in range(0,self.ndof):
      self.y[j] = -(1.0/self.dx)*(fluxes[j+1] - fluxes[j])
      
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
