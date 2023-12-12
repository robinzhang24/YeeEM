import numpy as np

def advance_Ez(Ez, Bx, By, Jz, omega0, dx, dy, dt, step):
    nx, ny = Ez.shape
    for i in range(nx):
        for j in range(ny):
            # Store B-field at useful positions, use %nx and %ny to include the periodic boundary condition
            Bx_right = Bx[i, j]
            Bx_left = Bx[i, j - 1]
            By_top = By[i, j]
            By_bottom = By[i - 1, j]

            # Update E-field
            Ez[i, j] += dt * ((By_top - By_bottom) / dx - (Bx_right - Bx_left) / dy
                              - Jz(i, j, omega0, nx, ny, step, dt))
            #print(Jz(i, j, step, dt))

def advance_Bx(Bx, Ez, dy, dt):
    nx, ny = Bx.shape
    for i in range(nx):
        for j in range(ny):
            Ez_top = Ez[i, (j+1) % ny]
            Ez_bottom = Ez[i, j]
            Bx[i, j] -= dt * (Ez_top - Ez_bottom) / dy

def advance_By(By, Ez, dx, dt):
    nx, ny = By.shape
    for i in range(nx):
        for j in range(ny):
            Ez_right = Ez[(i+1) % nx, j]
            Ez_left = Ez[i, j]
            By[i, j] += dt * (Ez_right - Ez_left) / dx

def field_evolve_2D(Ez, Bx, By, Jz, omega0, dx, dy, dt, steps):
    Ez_history = [np.copy(Ez)]
    Bx_history = [np.copy(Bx)]
    By_history = [np.copy(By)]

    for step in range(steps):
        advance_Ez(Ez, Bx, By, Jz, omega0, dx, dy, dt, step)
        advance_Bx(Bx, Ez, dy, dt)
        advance_By(By, Ez, dx, dt)

        Ez_history.append(np.copy(Ez))
        Bx_history.append(np.copy(Bx))
        By_history.append(np.copy(By))

    return Ez_history, Bx_history, By_history

# Define source current function
def Jz_func(i, j, omega0, nx, ny, int_t, dt):
    if (i,j) == (nx//2, ny//2):
      t = int_t*dt
      return np.sin(t)**2 * np.cos(omega0*t)
    return 0