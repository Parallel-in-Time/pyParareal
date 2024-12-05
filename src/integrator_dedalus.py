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

class solver():
    """Class for coarse/fine propagator for forward problem (SWE)."""

    def __init__(self, N, timestepper, problem, M, grid):

        self.N = N  # Number of timesteps in one time slice.
        self.timestepper = timestepper
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

        # Build solver.
        solver = self.problem.build_solver(self.timestepper)
        solver.stop_iteration = timegrid.size - 1
        solver.stop_sim_time = timegrid[-1]
        solver.sim_time = timegrid[0]  # Set sim_time.

        # Initial conditions
        self.problem.namespace["u"].change_scales(1)
        self.problem.namespace["u"]["g"] = ic
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

        return np.array(u_list)

class integrator_dedalus(integrator):

  def __init__(self, tstart, tend, nsteps):
    super(integrator_dedalus, self).__init__(tstart, tend, nsteps)
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
  def get_update_matrix(self, u0):
    assert isinstance(u0, solution_dedalus), "Initial value u0 must be an object of type solution_dedalus"
    mysolver = solver(self.nsteps, self.timestepper, u0.problem, u0.n, u0.x)
    Rmat = self.findA(mysolver, u0.n, np.linspace(self.tstart, self.tend, self.nsteps+1))   
    return Rmat

  def findA(self,operator, M, timeslice):
    E = np.identity(M)
    A = np.zeros((M, M))
    for m in range(M):
       A[m] = operator.solve(E[m], timeslice)[-1]
    return A
