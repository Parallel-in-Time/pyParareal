from solution import solution
import numpy as np

class meshtransfer(object):

  def __init__(self, ndof_fine, ndof_coarse):
    assert ndof_fine >= ndof_coarse, "Numer of DoF for coarse level must be smaller or equal to number of DoF on fine level"
    self.Rmat    = np.eye(ndof_fine, ndof_coarse)
    self.Imat    = np.eye(ndof_coarse, ndof_fine)
    self.ndof_fine   = ndof_fine
    self.ndof_coarse = ndof_coarse
    
  '''
  Receives a solution type object associated with a coarse mesh and returns a sol
  '''
  def restrict(self, sol_fine, sol_coarse):
    assert sol_fine.ndof==self.ndof_fine, "Number of DoF in argument sol_fine does not match the number of fine DoF used to create this meshtransfer object"
    assert sol_coarse.ndof==self.ndof_coarse, "Number of DoF in argument sol_coarse does not match the number of coarse DoF used to create this meshtransfer object"
    sol_coarse.y = np.copy(self.Rmat@sol_fine.y)
    
  def interpolate(self, sol_fine, sol_coarse):
    assert sol_fine.ndof==self.ndof_fine, "Number of DoF in argument sol_fine does not match the number of fine DoF used to create this meshtransfer object"
    assert sol_coarse.ndof==self.ndof_coarse, "Number of DoF in argument sol_coarse does not match the number of coarse DoF used to create this meshtransfer object"
    sol_fine.y = np.copy(self.Imat@sol_coarse.y)
