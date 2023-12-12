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
        B[i] += dt / dx * (E[i + 1] - E[i])
    B[n - 1] += dt / dx * (E[0] - E[n - 1])


def field_evolve_vacuum(E: np.ndarray, B: np.ndarray, dx, dt):
    """Compute evolution of electromagnetic field in vacuum in 1D using the
    FDTD method. All quantities are assumed to be dimensionless.

    Inputs:
    - E : Initial electric field at integer coordinates, t=-dt/2. Assume
      periodic boundary conditions.
    - B : Initial magnetic field at half-integer coordinates, t=0.
    - dx : Standard grid size of the fields.
    - dt : Time step of field evolution.

    Returns:
    - E : Numpy array containing electric field at the next time step.
    - B : Numpy array containing magnetic field at the next time step.
    """

    advance_E_field_vacuum(E, B, dx, dt)
    advance_B_field_vacuum(E, B, dx, dt)
    return E, B

def advance_JE_medium(E : np.ndarray, B : np.ndarray, J: np.ndarray, Omega_p: np.ndarray,
                       nu: np.ndarray, dx, dt):
    n = E.shape[0]
    
    for i in range(n):
        E[i] += dt / dx *(B[i] - B[i - 1]) - J[i]
        J[i] = ((1 - nu[i] * dt / 2) * J[i] + Omega_p[i] ** 2 * dt * E[i]) / (1 + nu[i] * dt / 2)

def field_evolve_medium(E: np.ndarray, B: np.ndarray, J: np.ndarray, Omega_p: np.ndarray,
                        nu: np.ndarray, dx, dt):
    """Compute evolution of electromagnetic field in medium in 1D using the
    FDTD method. All quantities are assumed to be dimensionless.

    Inputs:
    - E : Initial electric field at integer coordinates, t=-dt/2. Assume
      periodic boundary conditions.
    - B : Initial magnetic field at half-integer coordinates, t=0.
    - J : Initial current at integer coordinates, t=0.
    - Omega_p : Numpy array specifying plasma frequency over the 1D grid.
    - nu : Numpy array specifying collisional constant over the 1D grid.
      Will work properly only if taken nonzero value at exactly the same
      grid points as Omega_p.
    - dx : Standard grid size of the fields.
    - dt : Time step of field evolution.

    Returns:
    - E : Numpy array containing electric field at the next time step.
    - B : Numpy array containing magnetic field at the next time step.
    - J : Numpy array containing current at the next time step.
    """

    advance_JE_medium(E, B, J, Omega_p, nu, dx, dt)
    advance_B_field_vacuum(E, B, dx, dt)
    return E, B, J
