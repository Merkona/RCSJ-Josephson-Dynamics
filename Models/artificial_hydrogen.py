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
from matplotlib import animation
from matplotlib.collections import LineCollection

# Ensure repository root is on sys.path for direct script execution
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from RCSJ_Basis.single_junction import SingleRCSJSolve


# --------------------------------------------------------------------------- #
# Simulation helper
# --------------------------------------------------------------------------- #
def run_sim(
    params,
    y0,
    tau_span,
    num_points=2000,
    **solve_kwargs,
):
    """
    Run a single-junction RCSJ simulation.

    Parameters
    ----------
    params : dict
        Keyword arguments passed to SingleRCSJSolve (e.g., Ic, C, R, I_dc, I_ac,
        omega_drive, phi_drive).
    y0 : sequence
        Initial conditions [phi0, phi_dot0].
    tau_span : tuple(float, float)
        Dimensionless time window (tau_start, tau_end).
    num_points : int, optional
        Number of evaluation points for t_eval (uniform grid).
    **solve_kwargs :
        Additional arguments forwarded to solve() (rtol, atol, max_step, etc.).

    Returns
    -------
    model : SingleRCSJSolve
    sol : OdeResult from scipy.integrate.solve_ivp
    """
    model = SingleRCSJSolve(**params)
    t_eval = np.linspace(*tau_span, num_points)
    sol = model.solve(y0=y0, tau_span=tau_span, t_eval=t_eval, **solve_kwargs)
    return model, sol


# --------------------------------------------------------------------------- #
# Plotting helpers
# --------------------------------------------------------------------------- #
def plot_phase_time(ax, sol):
    """Phase vs dimensionless time."""
    ax.plot(sol.t, sol.y[0], lw=1.5)
    ax.set_xlabel(r"Dimensionless time $\tau$")
    ax.set_ylabel(r"Phase $\phi$")
    ax.set_title("Phase vs Time")
    return ax


def plot_energy_split(ax, model, sol):
    """
    Plot kinetic, potential, and total energy vs time.
    """
    phi = sol.y[0]
    phi_dot = sol.y[1]

    kinetic = 0.5 * phi_dot**2
    potential = model.potential(phi)
    total = kinetic + potential

    ax.plot(sol.t, kinetic, label="Kinetic", lw=1.2)
    ax.plot(sol.t, potential, label="Potential", lw=1.2)
    ax.plot(sol.t, total, label="Total", lw=1.8)
    ax.set_xlabel(r"Dimensionless time $\tau$")
    ax.set_ylabel("Energy (dimensionless)")
    ax.set_title("Energy Partition vs Time")
    ax.legend()
    return ax


def plot_phase_space(ax, sol):
    """Phase-space trajectory: phi_dot vs phi with direction cues."""
    phi = sol.y[0]
    phi_dot = sol.y[1]

    ax.plot(phi, phi_dot, lw=1.0, color="C1")

    # Sparse, short arrows along the path to show direction without clutter
    n = len(phi)
    step = max(1, n // 30)
    idx = np.arange(0, n - 1, step)
    dphi = phi[idx + 1] - phi[idx]
    dphi_dot = phi_dot[idx + 1] - phi_dot[idx]
    norm = np.hypot(dphi, dphi_dot)
    norm[norm == 0] = 1.0
    ax.quiver(
        phi[idx],
        phi_dot[idx],
        dphi / norm,
        dphi_dot / norm,
        angles="xy",
        scale_units="xy",
        scale=12,
        width=0.003,
        color="C0",
        alpha=0.8,
    )

    ax.set_xlabel(r"Phase $\phi$")
    ax.set_ylabel(r"Phase velocity $\dot{\phi}$")
    ax.set_title("Phase Space")
    return ax


def plot_potential_overlay(ax, model, sol, n_grid=400):
    """
    Plot tilted washboard potential with a time-coded trajectory overlay.
    """
    phi_traj = sol.y[0]
    phi_min = phi_traj.min()
    phi_max = phi_traj.max()
    margin = 0.5 * (phi_max - phi_min + 1e-6)
    grid = np.linspace(phi_min - margin, phi_max + margin, n_grid)
    potential_curve = model.potential(grid)
    ax.plot(grid, potential_curve, color="0.4", lw=1.5, label="Potential")

    # Time-coded trajectory using a LineCollection
    phi = phi_traj
    U = model.potential(phi)
    points = np.array([phi, U]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap="plasma", linewidth=2.0)
    lc.set_array(sol.t[:-1])
    ax.add_collection(lc)
    cb = plt.colorbar(lc, ax=ax)
    cb.set_label(r"Time $\tau$")

    ax.set_xlabel(r"Phase $\phi$")
    ax.set_ylabel(r"Potential $U(\phi)$")
    ax.set_title("Potential Landscape with Trajectory")
    ax.legend(loc="upper left")
    return ax


def animate_phase_space(model, sol, interval=20):
    """
    Build a matplotlib FuncAnimation showing the phase-space trajectory.

    Parameters
    ----------
    model : SingleRCSJSolve (unused, kept for future extensions)
    sol : OdeResult
    interval : int
        Delay between frames in ms.

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
    fig, ax : Figure and Axes used.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    phi = sol.y[0]
    phi_dot = sol.y[1]

    ax.set_xlabel(r"Phase $\phi$")
    ax.set_ylabel(r"Phase velocity $\dot{\phi}$")
    ax.set_title("Phase Space (animated)")

    # Plot full path lightly for context
    ax.plot(phi, phi_dot, color="0.9", lw=0.8)
    point, = ax.plot([], [], "o", color="C1")
    trail_line, = ax.plot([], [], color="C0", lw=2.0)

    def init():
        point.set_data([], [])
        trail_line.set_data([], [])
        return point, trail_line

    def update(i):
        point.set_data(phi[i], phi_dot[i])
        trail_line.set_data(phi[: i + 1], phi_dot[: i + 1])
        return point, trail_line

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(phi),
        interval=interval,
        blit=True,
    )
    return anim, fig, ax


def animate_bead_on_potential(model, sol, interval=15):
    """
    Animate a bead sliding on the tilted washboard U(phi) following phi(t).

    Parameters
    ----------
    model : SingleRCSJSolve
    sol : OdeResult
    interval : int
        Delay between frames in ms.

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
    fig, ax : Figure and Axes used.
    """
    phi_traj = sol.y[0]
    phi_min = phi_traj.min()
    phi_max = phi_traj.max()
    margin = 0.5 * (phi_max - phi_min + 1e-6)
    grid = np.linspace(phi_min - margin, phi_max + margin, 400)
    U_grid = model.potential(grid)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(grid, U_grid, color="0.4", lw=1.5, label="Potential")
    bead, = ax.plot([], [], "o", color="C3", markersize=8, label="Bead")
    ax.set_xlabel(r"Phase $\phi$")
    ax.set_ylabel(r"Potential $U(\phi)$")
    ax.set_title("Bead on Tilted Washboard")
    ax.legend(loc="upper left")

    def init():
        bead.set_data([], [])
        return (bead,)

    def update(i):
        phi = phi_traj[i]
        U = model.potential(phi)
        bead.set_data(phi, U)
        return (bead,)

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(phi_traj),
        interval=interval,
        blit=True,
    )
    return anim, fig, ax


# --------------------------------------------------------------------------- #
# Example usage
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Example parameters for a driven single junction
    params = dict(
        Ic=1e-6,
        C=1e-12,
        R=1e3,
        I_dc=0.5e-6,
        I_ac=0.2e-6,
        omega_drive=2e9,
        phi_drive=0.0,
    )
    y0 = [0.0, 0.0]
    tau_span = (0.0, 50.0)

    model, sol = run_sim(params, y0, tau_span, num_points=2000)
    if not sol.success:
        raise RuntimeError(f"Solver failed: {sol.message}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plot_phase_time(axes[0, 0], sol)
    plot_energy_split(axes[0, 1], model, sol)
    plot_phase_space(axes[1, 0], sol)
    plot_potential_overlay(axes[1, 1], model, sol)
    fig.tight_layout()
    plt.show()

    # Optional: build and display an animated phase-space trajectory
    anim, fig_anim, ax_anim = animate_phase_space(model, sol)
    plt.show()

    # Optional: bead-on-track animation of the potential landscape
    anim_bead, fig_bead, ax_bead = animate_bead_on_potential(model, sol)
    plt.show()
