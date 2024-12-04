from integrator import integrator
from solution import solution
from solution_dedalus import solution_dedalus
import copy

# Used to compute update matrices
from solution_linear import solution_linear
from scipy import sparse
from scipy import linalg
import dedalus.public as d3

import numpy as np

class dedalus(integrator):

  def __init__(self, tstart, tend, nsteps):
    super(dedalus, self).__init__(tstart, tend, nsteps)
    self.order = 1
    self.timestepper = d3.RK111

  def run(self, u0):
    assert isinstance(u0, solution_dedalus), "Initial value u0 must be an object of type solution_dedalus"
    raise NotImplementedError("The Dedalus time stepper can currently only return a matrix but cannot be run")

    #!/usr/bin/env python3
    # -*- coding: utf-8 -*-
    """
    Created on Thu Feb 22 13:53:07 2024.

    @author: Judith Angel
    """
  def get_update_matrix(self, sol):
    assert isinstance(u0, solution_dedalus), "Initial value u0 must be an object of type solution_dedalus"
    Mat = solver(self.nsteps, self.timestepper, sol.problem, sol.n, sol.x)

    
    
    return Rmat

  def findA(operator, M, timeslice):
    E = np.identity(M)
    A = np.zeros((M, M))
    for m in range(M):
       A[m] = operator.solve(E[m], timeslice)[-1]
    return A
