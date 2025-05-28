import sys
sys.path.append('./src')

import pytest
import numpy as np
from get_matrix import get_upwind, get_centered, get_diffusion, get_desterck

class TestClass:
    
    def f(x):
      return np.sin(2.0*np.pi*x)
    
    def minus_fx(x):
      return -2.0*np.pi*np.cos(2.0*np.pi*x)
  
    def fxx(x):
      return -4.0*np.pi**2*np.sin(2.0*np.pi*x)  
    
    def setUp(self):
      self.ndof = 256
      self.xaxis = np.linspace(0.0, 1.0, self.ndof+1)[0:self.ndof]
      self.dx = self.xaxis[1] - self.xaxis[0]
      self.u  = TestClass.f(self.xaxis)
      self.ux = TestClass.minus_fx(self.xaxis)
      self.uxx = TestClass.fxx(self.xaxis)
      
    '''
    The following tests confirm that the finite difference provide an approximation that is within
    some fixed tolerance to the analytic solution
    '''
    
    def testUpwindIsAccurate(self):
      self.setUp()
      A = get_upwind(self.ndof, self.dx)    
      ux_fd = A*self.u
      err = np.linalg.norm(self.ux - ux_fd, np.inf)
      assert err < 1e-1, "Upwind finite difference seems inaccurate"
      
    def testCenteredIsAccurate(self):
      self.setUp()
      A = get_centered(self.ndof, self.dx)      
      ux_fd = A*self.u
      err = np.linalg.norm(self.ux - ux_fd, np.inf)
      assert err < 1e-2, "Centered finite difference seems inaccurate"
        
    def testDiffusionIsAccurate(self):
      self.setUp()
      A = get_diffusion(self.ndof, self.dx)      
      uxx_fd = A*self.u
      err = np.linalg.norm(self.uxx - uxx_fd, np.inf)
      assert err < 1e-2, "Centered finite difference for diffusion seems inaccurate"
        
    def testDeSterckIsAccurate(self):
      self.setUp()
      tols = [2e-2, 8e-6, 2e-7, 1e-9]
      for p in range(4):
          A = get_desterck(self.ndof, self.dx, p+2)
          ux_fd = A*self.u
          err = np.linalg.norm(self.ux - ux_fd, np.inf)
          assert err < tols[p], ("Order %i of De Sterck et al finite difference seems inaccurate" % (p+2))
      
            
      
    
