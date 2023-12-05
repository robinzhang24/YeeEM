# Implementation of the Finite Difference Time Domain (Yee) method in 1D

import numpy as np

def advance_E_field(E : np.ndarray, B : np.ndarray, dx, dt):
    n = E.shape[0]
    for i in range(n):
        E[i] += dt / dx * (B[i] - B[i-1])

def advance_B_field(E : np.ndarray, B : np.ndarray, dx, dt):
    n = B.shape[0]
    for i in range(n-1):
        B[i] += dt / dx * (E[i+1] - E[i])
    B[n-1] += dt / dx * (E[0] - E[n-1])


def field_evolve_vacuum(E : np.ndarray, B : np.ndarray, dx, dt):
    """Compute evolution of electromagnetic field in vacuum in 1D using the
    FDTD method. All quantities are assumed to be dimensionless.

    Inputs:
    - E : Initial electric field at integer coordinates, t=-dt/2. Assume
      periodic boundary conditions
    - B : Initial magnetic field at half-integer coordinates, t=0. 
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
    
    advance_E_field(E, B, dx, dt)
    advance_B_field(E, B, dx, dt)
    return E, B
