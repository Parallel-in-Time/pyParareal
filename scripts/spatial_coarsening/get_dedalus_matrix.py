#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 13:53:07 2024.

@author: Judith Angel
"""
import numpy as np
from scipy import interpolate
import warnings
import dedalus.public as d3


def R(Uk):
    """
    Restriction operator.

    Parameters
    ----------
    Uk : numpy array
        Array of shape (M_F,2) that needs to be restricted to coarser grid.

    Returns
    -------
    numpy array
        Array on coarser grid.

    """
    gridG = G.grid
    gridF = F.grid
    Uint = interpolate.CubicSpline(gridF, Uk, axis=0)

    return Uint(gridG)


def L(Uk):
    """
    Lifting operator.

    Parameters
    ----------
    Uk : numpy array
        Array of shape (M_g,2) that needs to be lifted onto finer grid.

    Returns
    -------
    numpy array
        Lifted array.

    """
    gridG = G.grid
    gridF = F.grid
    Uint = interpolate.CubicSpline(gridG, Uk, axis=0)

    return Uint(gridF)


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


################################################
# ------------ SET PARAMETERS HERE -------------
xmin = -2
xmax = 2
mu   = 1e-4
M_G  = 50  # Number of grid points in space for the coarse propagator.
M_F  = 100  # Number of grid points in space for the fine propagator.
N_G  = 10  # Number of time steps within one time slice for the coarse prop.
N_F  = 10  # Number of time steps within one time slice for the fine prop.
P    = 10  # Number of time slices.
T_N  = 1
# ----------------------------------------------
################################################

# Define arrays of time slices.
c1, c2 = np.meshgrid(np.linspace(0, T_N/P, N_G+1),
                     np.linspace(0, T_N-T_N/P, P))
coarseGrids = c1 + c2
f1, f2 = np.meshgrid(np.linspace(0, T_N/P, N_F+1),
                     np.linspace(0, T_N-T_N/P, P))
fineGrids = f1 + f2

# Define time stepper.
timestepper = d3.RK111

# Coordinates, distributor bases, grid.
xcoord = d3.Coordinate('x')
dist = d3.Distributor(xcoord, dtype=np.complex128)
xbasisG = d3.ComplexFourier(xcoord, size=M_G, bounds=(xmin, xmax), dealias=3/2)
xbasisF = d3.ComplexFourier(xcoord, size=M_F, bounds=(xmin, xmax), dealias=3/2)
x_G = dist.local_grid(xbasisG)
x_F = dist.local_grid(xbasisF)

# Dedalus fields.
u_G = dist.Field(name='u', bases=xbasisG)
u_F = dist.Field(name='u', bases=xbasisF)
t_field = dist.Field()

dx = lambda A: d3.Differentiate(A, xcoord)

# Define advection diffusion problem.
problemG = d3.IVP([u_G], time=t_field, namespace={
    "u": u_G, "mu": mu, "dx": dx})
problemG.add_equation("dt(u) + dx(u) - mu*dx(dx(u)) = 0")
problemF = d3.IVP([u_F], time=t_field, namespace={
    "u": u_F, "mu": mu, "dx": dx})
problemF.add_equation("dt(u) + dx(u) - mu*dx(dx(u)) = 0")

# Define solvers.
G = solver(N_G, timestepper, problemG, M_G, x_G)
F = solver(N_F, timestepper, problemF, M_F, x_F)


def findA(operator, M, timeslice):
    E = np.identity(M)
    A = np.zeros((M, M))
    for m in range(M):
        A[m] = operator.solve(E[m], timeslice)[-1]
    return A


# Define identities.
id_F = np.identity(M_F)
id_G = np.identity(M_G)

# Initialise matrices.
A_R = np.zeros((M_F, M_G))  # For restriction.
A_L = np.zeros((M_G, M_F))  # For interpolation.

# Compute matrices for restriction and lifting operators.
for m in range(M_F):
    A_R[m] = R(id_F[m])
    if m < M_G:
        A_L[m] = L(id_G[m])

# Compute matrix representations for fine and coarse propagator.
A_G = findA(G, M_G, coarseGrids[0])  # For coarse propagator.
A_F = findA(F, M_F, fineGrids[0])  # For fine propagator.

print(np.shape(A_G))
print(no.shape(A_F)
