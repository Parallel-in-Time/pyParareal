from solution import solution
import numpy as np
from scipy.interpolate import interp1d

class meshtransfer(object):

  def __init__(self, ndof_fine, ndof_coarse):
    assert ndof_fine >= ndof_coarse, "Numer of DoF for coarse level must be smaller or equal to number of DoF on fine level"

    self.ndof_fine   = ndof_fine
    self.ndof_coarse = ndof_coarse
    # For the time being, assume we are operating on the unit interval
    self.xaxis_f = np.linspace(0.0, 1.0, self.ndof_fine, endpoint=True)
    self.xaxis_c = np.linspace(0.0, 1.0, self.ndof_coarse, endpoint=True)
    
    self.Imat = np.zeros((self.ndof_fine, self.ndof_coarse))
    for n in range(self.ndof_coarse):
      e = np.zeros(self.ndof_coarse)
      e[n] = 1.0
      f = interp1d(self.xaxis_c, e, kind='cubic')
      self.Imat[:,n] = f(self.xaxis_f)
      
    self.Rmat = np.zeros((self.ndof_coarse, self.ndof_fine))
    for n in range(self.ndof_fine):
      e = np.zeros(self.ndof_fine)
      e[n] = 1.0
      f = interp1d(self.xaxis_f, e, kind='cubic')
      self.Rmat[:,n] = f(self.xaxis_c)
      
    #self.Imat = np.eye(self.ndof_fine,self.ndof_coarse)
    #self.Rmat = np.eye(self.ndof_coarse,self.ndof_fine)

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
