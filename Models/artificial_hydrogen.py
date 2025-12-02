"""
Artificial hydrogen model built on the single-junction RCSJ solver.

This module provides convenience helpers to:
- run a single-junction RCSJ simulation
- visualize phase vs time, energy partition, phase space (with direction),
  potential overlay with time-coded trajectory, and an optional phase-space
  animation.

All plotting routines expect a matplotlib Axes and return it so callers can
compose figures as needed.
"""

import pathlib
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


# Ensure repository root is on sys.path for direct script execution
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from RCSJ_Basis.single_junction import SingleRCSJSolve


# ------------------------------------------------------------
# 1. SETUP: create a single junction and solve its dynamics
# ------------------------------------------------------------

jj = SingleRCSJSolve(
    Ic=1e-6,
    C=1e-12,
    R=5e3,
    I_dc=0.3e-6,   # small tilt to keep the first well intact
    I_ac=0.0,
    omega_drive=0.0,
    phi_drive=0.0,
)

y0 = [0.0, 0.0]
tau_span = (0, 40)
t_eval = np.linspace(*tau_span, 2000)

sol = jj.solve(y0=y0, tau_span=tau_span, t_eval=t_eval)

phi = sol.y[0]
t = sol.t

# ------------------------------------------------------------
# 2. SIMPLE φ vs t PLOT
# ------------------------------------------------------------

plt.figure(figsize=(7,4))
plt.plot(t, phi, lw=1.5)
plt.xlabel("Dimensionless Time τ")
plt.ylabel("Phase φ")
plt.title("Phase vs Time")
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 3. ANIMATION: bead oscillating in the tilted washboard
# ------------------------------------------------------------

# Range that shows the first full well AND the next empty well
phi_min = -2.0
phi_max = 9.0
phi_grid = np.linspace(phi_min, phi_max, 800)
U_grid = jj.potential(phi_grid)

# Create the figure
fig, ax = plt.subplots(figsize=(7,4))
ax.set_xlim(phi_min, phi_max)
ax.set_ylim(min(U_grid)-0.5, max(U_grid)+0.5)

ax.set_xlabel("Phase φ")
ax.set_ylabel("Potential U(φ)")
ax.set_title("Bead Oscillating in the Tilted Washboard")

# Plot potential
ax.plot(phi_grid, U_grid, color="black", lw=2)

# Bead (the particle)
bead, = ax.plot([], [], "ro", ms=10)

# Animation update function
def update(frame):
    bead.set_data([phi[frame]], [jj.potential(phi[frame])])
    return bead,

# Create animation
anim = FuncAnimation(
    fig,
    update,
    frames=600,
    interval=1,
    blit=True
)

plt.show()
anim.save("washboard_potential.gif", writer=PillowWriter(fps=30))


# ------------------------------------------------------------
# 1. EFFECTIVE HYDROGEN RADIAL POTENTIAL (with angular momentum)
# ------------------------------------------------------------

L = 1.0  # angular momentum (in suitable units)

def V_eff(r):
    return -1.0 / r + 0.5 * L**2 / r**2

# ------------------------------------------------------------
# 2. CLASSICAL MOTION IN V_eff(r)  (symplectic integrator)
# ------------------------------------------------------------

def accel(r):
    # r'' = -dV_eff/dr = -1/r**2 + L**2 / r**3
    return -1.0 / (r**2) + L**2 / (r**3)

def integrate(r0, rdot0, t_max=40, dt=0.01):
    t = np.arange(0, t_max, dt)
    r = np.zeros_like(t)
    rd = np.zeros_like(t)
    r[0], rd[0] = r0, rdot0

    for i in range(len(t) - 1):
        # velocity Verlet / leapfrog (symplectic)
        a_n = accel(r[i])
        rd_half = rd[i] + 0.5 * a_n * dt
        r[i+1] = r[i] + rd_half * dt

        # avoid singularity
        if r[i+1] < 1e-3:
            r[i+1] = 1e-3

        a_np1 = accel(r[i+1])
        rd[i+1] = rd_half + 0.5 * a_np1 * dt

    return t, r

# Choose ICs near the minimum of V_eff for bound motion
t, r = integrate(r0=1.5, rdot0=0.0, t_max=40, dt=0.01)

# ------------------------------------------------------------
# 3. SIMPLE r vs t PLOT
# ------------------------------------------------------------

plt.figure(figsize=(7, 4))
plt.plot(t, r)
plt.xlabel("Time")
plt.ylabel("Radius r")
plt.title("Electron Radius vs Time in V_eff(r)")
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 4. ANIMATION OF THE ELECTRON IN V_eff(r)
# ------------------------------------------------------------

r_min = 0.1
r_max = 10.0
r_grid = np.linspace(r_min, r_max, 500)
V_grid = V_eff(r_grid)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(r_grid, V_grid, lw=2, color="black")
ax.set_xlim(r_min, r_max)
ax.set_ylim(-1, 4.0)
ax.set_xlabel("Radius r")
ax.set_ylabel("Effective Potential V_eff(r)")
ax.set_title("Electron Bead in Hydrogen Effective Potential")

bead, = ax.plot([], [], "ro", ms=10)

def update(frame):
    r_val = r[frame]
    bead.set_data([r_val], [V_eff(r_val)])
    return bead,

anim = FuncAnimation(
    fig,
    update,
    frames=1200,
    interval=0.5,
    blit=True
)

plt.show()
anim.save("H_potential.gif", writer=PillowWriter(fps=30))