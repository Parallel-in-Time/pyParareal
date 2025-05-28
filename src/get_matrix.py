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
  
def get_desterck(n, h, p):
  col = np.zeros(n)
  if p==2:
    col[0] = 3.0
    col[1] = -4.0
    col[2] = 1-0
    A = -1.0/(2.0*h)*spla.circulant(col)
  elif p==3:
    col[0] = 3.0
    col[1] = -6.0
    col[2] = 1.0
    col[-1] = 2.0
    A = -1.0/(6.0*h)*spla.circulant(col)
  elif p==4:
    col[0] = 10.0
    col[1] = -18.0
    col[2] = 6.0
    col[3] = -1.0
    col[-1] = 3.0
    A = -1.0/(12.0*h)*spla.circulant(col)
  elif p==5:
    col[0] = 20.0
    col[1] = -60.0
    col[2] = 15.0
    col[3] = -2.0
    col[-1] = 30.0
    col[-2] = -3.0
    A = -1.0/(60.0*h)*spla.circulant(col)
  else:
    sys.exit("Do not have coefficients for requested p")
  return sp.csc_matrix(A)
