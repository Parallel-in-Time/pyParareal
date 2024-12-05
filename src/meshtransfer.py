from solution import solution
import numpy as np
from scipy import interpolate

class meshtransfer(object):

  def __init__(self, ndof_fine, ndof_coarse, dedalus = False):
    assert ndof_fine >= ndof_coarse, "Numer of DoF for coarse level must be smaller or equal to number of DoF on fine level"

    self.ndof_fine   = ndof_fine
    self.ndof_coarse = ndof_coarse
    
    # for a scalar problem, there is no interpolation or restriction; set corresponding operators to 1x1 identity matrix
    if self.ndof_fine==1 and self.ndof_coarse==1:
      self.Imat = np.eye(1)
      self.Rmat = np.eye(1)
    
    else:
      
      if not dedalus:
        
        # For the time being, assume we are operating on the unit interval
        self.xaxis_f = np.linspace(0.0, 1.0, self.ndof_fine, endpoint=True)
        self.xaxis_c = np.linspace(0.0, 1.0, self.ndof_coarse, endpoint=True)
        mykind = 'linear'
        
      else:
      
        # Because Dedalus uses spectral methods assuming periodic BC, the right endpoint is not included in the mesh
        self.xaxis_f = np.linspace(0.0, 1.0, self.ndof_fine, endpoint=False)
        self.xaxis_c = np.linspace(0.0, 1.0, self.ndof_coarse, endpoint=False)      
        mykind = 'cubic'
        
      self.Imat = np.zeros((self.ndof_fine, self.ndof_coarse))
      for n in range(self.ndof_coarse):
        e = np.zeros(self.ndof_coarse)
        e[n] = 1.0
        # Because the last right meshpoint is missing, the last fine mesh point is outside of the coarse mesh 
        # and we need to allow interp1d to extrapolate
        f = interpolate.interp1d(self.xaxis_c, e, kind=mykind, fill_value="extrapolate")
        self.Imat[:,n] = f(self.xaxis_f)
        
      self.Rmat = np.zeros((self.ndof_coarse, self.ndof_fine))
      for n in range(self.ndof_fine):
        e = np.zeros(self.ndof_fine)
        e[n] = 1.0
        f = interpolate.interp1d(self.xaxis_f, e, kind=mykind)
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
