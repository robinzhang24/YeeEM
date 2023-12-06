# Implementation of the Finite Difference Time Domain (Yee) method in 1D

import numpy as np


def advance_E_field_vacuum(E: np.ndarray, B: np.ndarray, dx, dt):
    n = E.shape[0]
    E[0] += dt / dx * (B[0] - B[n - 1])
    for i in range(1,n):
        E[i] += dt / dx * (B[i] - B[i - 1])


def advance_B_field_vacuum(E: np.ndarray, B: np.ndarray, dx, dt):#same in vacuum and medium
    n = B.shape[0]
    for i in range(n - 1):
        B[i] += -dt / dx * (E[i + 1] - E[i])
    B[n - 1] += -dt / dx * (E[0] - E[n - 1])


def field_evolve_vacuum(E: np.ndarray, B: np.ndarray, dx, dt):
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

    advance_E_field_vacuum(E, B, dx, dt)
    advance_B_field_vacuum(E, B, dx, dt)
    return E, B

def advance_JE_medium(E : np.ndarray, B : np.ndarray, J: np.ndarray, dx, dt, omega_p, nu):
    n = E.shape[0]
    J[0] += (omega_p ** 2 / 4 / np.pi * E[0] - nu * J[0]) / dt
    E[0] += dt / dx * (B[0] - B[n - 1]) - 4 * np.pi * J[0] / dt
    for i in range(1, n):
        J[i] += (omega_p ** 2/4/np.pi * E[i] - nu * J[i]) / dt
        E[i] += dt / dx * (B[i] - B[i - 1]) - 4 * np.pi * J[i] / dt

def field_evolve_medium(E: np.ndarray, B: np.ndarray, J: np.array, dx, dt, omega_p, nu):

    advance_JE_medium()(E, B, dx, dt, omega_p, nu)
    advance_B_field_vacuum(E, B, dx, dt)
    return E, B, J
