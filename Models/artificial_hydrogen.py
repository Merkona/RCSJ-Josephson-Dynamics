"""
Artificial hydrogen playground built on the single-junction RCSJ solver.

What this module provides:
- Thin wrapper around SingleRCSJSolve to run single-junction simulations.
- Plotting utilities for time traces, energy partition, phase space, and potential overlays.
- Animations for phase space and bead-on-washboard visuals.
- Convenience sweeps for IV curves.
"""

import pathlib
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import LineCollection
from matplotlib.animation import PillowWriter


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
    t_eval=None,
    **solve_kwargs,
):
    """
    Run a single-junction RCSJ simulation on a uniform time grid (or provided t_eval).

    Parameters
    ----------
    params : dict
        Keyword args for SingleRCSJSolve (Ic, C, R, I_dc, I_ac, omega_drive, phi_drive).
    y0 : sequence
        Initial conditions [phi0, phi_dot0].
    tau_span : tuple(float, float)
        Dimensionless time window (tau_start, tau_end).
    num_points : int, optional
        Number of evaluation points for t_eval (uniform grid).
    t_eval : array-like, optional
        Explicit time samples; overrides the uniform grid if provided.
    **solve_kwargs :
        Additional arguments forwarded to solve() (rtol, atol, max_step, etc.).

    Returns
    -------
    model : SingleRCSJSolve
        Solver object with parameters set from `params`.
    sol : OdeResult
        SciPy integrator output containing states and metadata.
    """
    tau_start, tau_end = map(float, tau_span)
    model = SingleRCSJSolve(**params)
    if t_eval is None:
        t_eval = np.linspace(tau_start, tau_end, num_points)
    sol = model.solve(
        y0=y0, tau_span=(tau_start, tau_end), t_eval=t_eval, **solve_kwargs
    )
    return model, sol


def _potential_grid(phi_traj, n_grid=400):
    """
    Build a grid around the trajectory range with padding to show neighboring wells.
    """
    phi_min = phi_traj.min()
    phi_max = phi_traj.max()
    span = phi_max - phi_min
    margin = 0.5 * (span + 1e-6) + 2 * np.pi
    grid = np.linspace(phi_min - margin, phi_max + margin, n_grid)
    return grid


# --------------------------------------------------------------------------- #
# Plotting helpers
# --------------------------------------------------------------------------- #
def plot_phase_time(ax, sol):
    """Plot junction phase versus dimensionless time on a supplied Axes."""
    ax.plot(sol.t, sol.y[0], lw=1.5)
    ax.set_xlabel(r"Dimensionless time $\tau$")
    ax.set_ylabel(r"Phase $\phi$")
    ax.set_title("Phase vs Time")
    return ax


def plot_energy_split(ax, model, sol):
    """
    Plot kinetic, potential, and total energy versus time.
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
    """
    Phase-space trajectory (phi, phi_dot) with direction cues.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which to draw the trajectory.
    sol : OdeResult
        Solution object from `run_sim`.
    """
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
    Tilted washboard potential with a time-coded trajectory overlay.
    """
    phi_traj = sol.y[0]
    grid = _potential_grid(phi_traj, n_grid)
    potential_curve = model.potential(grid)
    ax.plot(grid, potential_curve, color="0.4", lw=1.5, label="Potential")

    # Time-coded trajectory using a LineCollection
    phi = phi_traj
    if phi.size >= 2:
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


def compute_avg_voltage(sol, discard=0.5):
    """
    Compute normalized voltage proxy from average phase velocity.

    Parameters
    ----------
    sol : OdeResult
        Solution object from `run_sim`.
    discard : float, optional
        Fraction of initial samples to drop as transient (0 <= discard < 1).
    """
    n = sol.y.shape[1]
    start = int(discard * n)
    return np.mean(sol.y[1, start:])


def sweep_vi(params, i_dc_values, y0, tau_span=(0.0, 20.0), num_points=800):
    """
    Sweep DC bias and compute average normalized voltage (phi_dot).

    AC drive is turned off for a clean superconducting IV.
    """
    voltages = []
    for i_dc in i_dc_values:
        p = dict(params)
        p["I_dc"] = i_dc
        p["I_ac"] = 0.0
        model, sol = run_sim(p, y0, tau_span, num_points=num_points)
        if not sol.success:
            voltages.append(np.nan)
            continue
        voltages.append(compute_avg_voltage(sol))
    return np.array(voltages)


def plot_vi_curve(ax, currents, voltages, Ic):
    """
    Plot normalized IV curve (avg phi_dot vs I/Ic) for a DC sweep.
    """
    ax.plot(
        currents / Ic,
        voltages,
        marker="o",
        lw=1.2,
        label=r"$\langle \dot{\phi} \rangle$",
    )
    ax.set_xlabel(r"Normalized bias $I/I_c$")
    ax.set_ylabel(r"Normalized voltage $\langle \dot{\phi} \rangle$")
    ax.set_title("IV Curve (superconducting to running state)")
    ax.legend()
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
        Animation object for further saving or display.
    fig, ax : Figure, Axes
        Matplotlib figure and axes used to render the animation.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    phi = sol.y[0]
    phi_dot = sol.y[1]

    ax.set_xlabel(r"Phase $\phi$")
    ax.set_ylabel(r"Phase velocity $\dot{\phi}$")
    ax.set_title("Phase Space (animated)")

    # Plot full path lightly for context
    ax.plot(phi, phi_dot, color="0.9", lw=0.8)
    (point,) = ax.plot([], [], "o", color="C1")
    (trail_line,) = ax.plot([], [], color="C0", lw=2.0)

    def init():
        point.set_data([], [])
        trail_line.set_data([], [])
        return point, trail_line

    def update(i):
        point.set_data([phi[i]], [phi_dot[i]])
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

#Uncomment the ### lines to save animation to file directory

###writer = PillowWriter(fps=30)
###ani.save("animation.gif", writer=writer)


def animate_bead_on_potential(model, sol, interval=8):
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
        Animation object for further saving or display.
    fig, ax : Figure, Axes
        Matplotlib figure and axes used to render the animation.
    """
    phi_traj = sol.y[0]
    grid = _potential_grid(phi_traj, 400)
    U_grid = model.potential(grid)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(grid, U_grid, color="0.4", lw=1.5, label="Potential")
    (bead,) = ax.plot([], [], "o", color="C3", markersize=8, label="Bead")
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
        bead.set_data([phi], [U])
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

#Uncomment the ### lines to save animation to file directory

###writer = PillowWriter(fps=30)
###ani.save("animation.gif", writer=writer)


# GETTING STARTED - EXAMPLE USAGE
# READY TO RUN
# Feel free to change parameters of the below example as needed, we have inputted a reasonable case example
# Expect plots and animations outputted, no terminal output

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
    tau_span = (0.0, 100.0)

    model, sol = run_sim(params, y0, tau_span, num_points=4000)
    if not sol.success:
        raise RuntimeError(f"Solver failed: {sol.message}")

    # IV curve (DC sweep, no AC drive)
    i_dc_sweep = np.linspace(-1.5 * params["Ic"], 1.5 * params["Ic"], 35)
    vi_vals = sweep_vi(params, i_dc_sweep, y0, tau_span=(0.0, 40.0), num_points=1200)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plot_phase_time(axes[0, 0], sol)
    plot_energy_split(axes[0, 1], model, sol)
    plot_phase_space(axes[1, 0], sol)
    plot_potential_overlay(axes[1, 1], model, sol)
    fig.tight_layout()
    plt.show()

    # IV plot
    fig_iv, ax_iv = plt.subplots(figsize=(5, 4))
    plot_vi_curve(ax_iv, i_dc_sweep, vi_vals, params["Ic"])
    fig_iv.tight_layout()
    plt.show()

    # Optional: build and display an animated phase-space trajectory
    anim, fig_anim, ax_anim = animate_phase_space(model, sol)
    plt.show()

    # Optional: bead-on-track animation of the potential landscape
    anim_bead, fig_bead, ax_bead = animate_bead_on_potential(model, sol)
    plt.show()
