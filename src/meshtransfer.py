from solution import solution
import numpy as np

class meshtransfer(object):

  def __init__(self, ndof_fine, ndof_coarse):
    assert ndof_fine >= ndof_coarse, "Numer of DoF for coarse level must be smaller or equal to number of DoF on fine level"
    self.restrict    = np.eye(ndof_fine, ndof_coarse)
    self.interpolate = np.eye(ndof_coarse, ndof_fine)
    self.ndof_fine   = ndof_fine
    self.ndof_coarse = ndof_coarse
    
  '''
  Receives a solution type object associated with a coarse mesh and returns a sol
  '''
  def restrict(self, sol_fine):
    assert sol_fine.ndof==self.ndof_fine, "Number of DoF in argument sol_fine does not match the number of fine DoF used to create this meshtransfer object"
    assert sol_fine.M is None, "Meshtransfer is currently only implemented for problems without a mass matrix"
    sol_coarse = solution(self.restrict@sol_fine.y)
    # Only correct if M = identity - otherwise, we would somehow have to recreate the mass matrix of the bigger fine problem for the smaller coarse problem
    return sol_coarse
    
  def interpolate(self, sol_coarse):
    assert sol_coarse.ndof==self.ndof_coarse, "Number of DoF in argument sol_coarse does not match the number of coarse DoF used to create this meshtransfer object"
    assert sol_coarse.M is None, "Meshtransfer is currently only implemented for problems without a mass matrix"
    sol_fine = solution(self.interpolate@sol_coarse.y)
    return sol_fine
