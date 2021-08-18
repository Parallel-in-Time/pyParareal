import sys
sys.path.append('../../src')
import numpy as np
from numpy import linalg as LA
import scipy.sparse as sp
import scipy.linalg as spla

def get_upwind(n, h, periodic = True):
  col = np.zeros(n)
  col[0] = 1.0
  col[1] = -1.0
  A      =  -(1.0/h)*spla.circulant(col)
  if not periodic:
    A[0,-1] = 0.0
  return sp.csc_matrix(A)
  
def get_centered(n, h):
    col = np.zeros(n)
    col[1]  = -1.0
    col[-1] = 1.0
    A       = -(1.0/(2.0*h))*spla.circulant(col)
    return sp.csc_matrix(A)
    
def get_diffusion(n,h):
    col = np.zeros(n)
    col[0]  = -2.0
    col[1]  = 1.0
    col[-1] = 1.0
    A       = (1.0/h**2)*spla.circulant(col)
    return sp.csc_matrix(A)
