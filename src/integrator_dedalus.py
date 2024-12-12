from integrator import integrator
from solution import solution
from solution_dedalus import solution_dedalus
from special_integrator import special_integrator
import copy

# Used to compute update matrices
from solution_linear import solution_linear
from scipy import sparse
from scipy import linalg
import dedalus.public as d3
import logging

import numpy as np

class solver():
    """Class for coarse/fine propagator for forward problem (SWE)."""

    def __init__(self, N, problem, M, grid):

        self.N = N  # Number of timesteps in one time slice.
        self.timestepper = d3.RK443
        self.problem = problem
        self.M = M
        self.grid = grid

    def solve(self, ic, timegrid):
        """
        Solve IVP on the given time grid.

        Parameters
        ----------
        ic : numpy array
            initial condition for timegrid.
        timegrid : numpy array
            Time grid on which the problem should be solved.

        Returns
        -------
        numpy array
            Solution on timegrid.

        """
        # Time step
        dt = timegrid[1] - timegrid[0]
        
        for system in ['subsystems', 'solvers']:
          logging.getLogger(system).setLevel(logging.WARNING)

        # Build solver.
        solver = self.problem.build_solver(self.timestepper)
        solver.stop_iteration = timegrid.size - 1
        solver.stop_sim_time = timegrid[-1]
        solver.sim_time = timegrid[0]  # Set sim_time.

        # Initial conditions
        self.problem.namespace["u"].change_scales(1)
        self.problem.namespace["u"]["g"] = ic.flatten()
        self.problem.namespace["u"].change_scales(1)
        u_list = [np.copy(self.problem.namespace["u"]['g'])]
        t_list = [solver.sim_time]

        # Main loop
        while solver.proceed:
            solver.step(dt)
            self.problem.namespace["u"].change_scales(1)
            u_list.append(np.copy(self.problem.namespace["u"]['g']))
            t_list.append(solver.sim_time)
            if np.max(self.problem.namespace["u"]['g']) > 100:
                warnings.warn("Solution instable")
                break
        
        
        return np.copy(np.array(u_list)[-1])

class integrator_dedalus(integrator):

  def __init__(self, tstart, tend, nsteps):
    super(integrator_dedalus, self).__init__(tstart, tend, nsteps)
    self.timegrid = np.linspace(self.tstart, self.tend, self.nsteps+1)
    self.order = 1    
    
  def run(self, u0):
    assert isinstance(u0, solution_dedalus), "Initial value u0 must be an object of type solution_dedalus"
    mysolver = solver(self.nsteps, u0.problem, u0.n, u0.x)
    y_end = mysolver.solve(u0.y, self.timegrid)
    u0.y = y_end
    return u0
    
    #!/usr/bin/env python3
    # -*- coding: utf-8 -*-
    """
    Created on Thu Feb 22 13:53:07 2024.

    @author: Judith Angel
    """
  def get_update_matrix(self, u0):
    assert isinstance(u0, solution_dedalus), "Initial value u0 must be an object of type solution_dedalus"
    mysolver = solver(self.nsteps, u0.problem, u0.n, u0.x)
    E    = np.identity(u0.n)
    Rmat = np.zeros((u0.n, u0.n))
    for m in range(u0.n):
       Rmat[m] = mysolver.solve(E[m], self.timegrid)
    return Rmat.transpose()

  def convert_to_special_integrator(self, u0):
    assert isinstance(u0, solution_dedalus), "Initial value u0 must be an object of type solution_dedalus"
    return special_integrator(self.tstart, self.tend, 1, self.get_update_matrix(u0))
    
