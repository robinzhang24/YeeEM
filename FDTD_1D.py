# Implementation of the Finite Difference Time Domain (Yee) method in 1D

import numpy as np

def field_evolve_vacuum(E_0 : np.ndarray, B_0 : np.ndarray, dx, dt, step):
    """Compute evolution of electromagnetic field in vacuum in 1D using the
    FDTD method. All quantities are assumed to be dimensionless.

    Inputs:
    - E_0 : Initial electric field at integer coordinates, t=-dt/2. Assume
      periodic boundary conditions
    - B_0 : Initial magnetic field at half-integer coordinates, t=0. 
      The length of B_0 has to be smaller than E_0 by 1.
    - dx : Standard grid size of the fields.
    - dt : Time step of field evolution.
    - step : Total number of time steps to be simulated. The system will generate
      a set of E and B field at each timestep, though the exact time at which the
      fields are calculated are different. The output arrays will have (step+1)
      entries.

    Returns:
    - E_history: Numpy array containing history of electric field evolution.
    - B_history: Numpy array containing history of magnetic field evolution.
    """
    # check field initialization
    if len(E_0) - len(B_0) != 1:
        raise ValueError('Invalid field initialization!')
    elif E_0[0] != E_0[-1]:
        raise ValueError('Initial E field does not satisfy periodic boundary condition!')
    
    grid_size = E_0.shape[0]
    E_history = np.zeros((step, grid_size))
    B_history = np.zeros((step, grid_size - 1))
    E_history[0] = E_0.copy()
    B_history[0] = B_0.copy()
    for i in range(step):
        for j in range(1, grid_size - 1):
            E_history[i+1][j] = E_history[i][j] + dt / dx * (B_history[i][j] - B_history[i][j-1])
        E_history[i+1][0] = E_history[i][0] + dt / dx * (B_history[i][0] - B_history[i][grid_size - 2])
        E_history[i+1][grid_size - 1] = E_history[i+1][0] # enforced periodic boundary condition
        for k in range(grid_size - 1):
            B_history[i+1][k] = B_history[i][k] + dt / dx * (E_history[i][k+1] - E_history[i][k])

    return E_history, B_history