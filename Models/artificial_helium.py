"""
Artificial helium model built on the coupled-junction RCSJ solver.

Provides helpers to:
- run a coupled RCSJ simulation
- visualize phase trajectories, energy partitions, normal modes
- view the 2D potential landscape as heatmaps and 3D surfaces with trajectory overlays
"""

import pathlib
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib import animation

# Ensure repository root is on sys.path for direct script execution
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from RCSJ_Basis.coupled_junction import CoupledRCSJSolve


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
    Run a coupled-junction RCSJ simulation.

    Parameters
    ----------
    params : dict
        Keyword args for CoupledRCSJSolve (kappa, Ic, C, R, I_dc, I_ac, etc.).
    y0 : sequence
        Initial conditions [phi1, phi1_dot, phi2, phi2_dot].
    tau_span : tuple(float, float)
        Dimensionless time window (tau_start, tau_end).
    num_points : int, optional
        Number of evaluation points for t_eval (uniform grid).
    **solve_kwargs :
        Additional arguments forwarded to solve() (rtol, atol, max_step, etc.).

    Returns
    -------
    model : CoupledRCSJSolve
    sol : OdeResult from scipy.integrate.solve_ivp
    """
    model = CoupledRCSJSolve(**params)
    t_eval = np.linspace(*tau_span, num_points)
    sol = model.solve(y0=y0, tau_span=tau_span, t_eval=t_eval, **solve_kwargs)
    return model, sol


# --------------------------------------------------------------------------- #
# Plotting helpers
# --------------------------------------------------------------------------- #
def plot_phases_time(ax, sol):
    """Plot phi1 and phi2 vs time."""
    ax.plot(sol.t, sol.y[0], label=r"$\phi_1$", lw=1.4)
    ax.plot(sol.t, sol.y[2], label=r"$\phi_2$", lw=1.4)
    ax.set_xlabel(r"Dimensionless time $\tau$")
    ax.set_ylabel(r"Phase $\phi$")
    ax.set_title("Phases vs Time")
    ax.legend()
    return ax


def plot_delta_phase(ax, sol):
    """Plot phase difference vs time."""
    delta = sol.y[0] - sol.y[2]
    ax.plot(sol.t, delta, lw=1.4, color="C2")
    ax.set_xlabel(r"Dimensionless time $\tau$")
    ax.set_ylabel(r"$\Delta\phi = \phi_1 - \phi_2$")
    ax.set_title("Phase Difference vs Time")
    return ax


def plot_energy_split(ax, model, sol):
    """Plot kinetic, potential components, coupling, and total energy vs time."""
    phi1, phi1_dot, phi2, phi2_dot = sol.y
    kinetic = 0.5 * (phi1_dot**2 + phi2_dot**2)
    U1 = 1.0 - np.cos(phi1) - model.i_dc * phi1
    U2 = 1.0 - np.cos(phi2) - model.i_dc * phi2
    Uc = 0.5 * model.kappa * (phi1 - phi2) ** 2
    potential = U1 + U2 + Uc
    total = kinetic + potential

    ax.plot(sol.t, kinetic, label="Kinetic", lw=1.2)
    ax.plot(sol.t, U1 + U2, label="Single potentials", lw=1.2)
    ax.plot(sol.t, Uc, label="Coupling", lw=1.2)
    ax.plot(sol.t, total, label="Total", lw=1.8)
    ax.set_xlabel(r"Dimensionless time $\tau$")
    ax.set_ylabel("Energy (dimensionless)")
    ax.set_title("Energy Partition vs Time")
    ax.legend()
    return ax


def plot_phase_space(ax, sol, which="phi1"):
    """
    Phase-space trajectory with direction cues.

    which: "phi1", "phi2", or "delta" for (phi1-phi2, phi1_dot-phi2_dot).
    """
    if which == "phi1":
        phi = sol.y[0]
        phi_dot = sol.y[1]
        label = r"\phi_1"
    elif which == "phi2":
        phi = sol.y[2]
        phi_dot = sol.y[3]
        label = r"\phi_2"
    else:
        phi = sol.y[0] - sol.y[2]
        phi_dot = sol.y[1] - sol.y[3]
        label = r"\Delta\phi"

    ax.plot(phi, phi_dot, lw=1.0, color="C1")

    n = len(phi)
    step = max(1, n // 30)
    idx = np.arange(0, n - 1, step)
    dphi = phi[idx + 1] - phi[idx]
    dphi_dot = phi_dot[idx + 1] - phi_dot[idx]
    ax.quiver(
        phi[idx],
        phi_dot[idx],
        dphi,
        dphi_dot,
        angles="xy",
        scale_units="xy",
        scale=0.5,
        width=0.008,
        color="C0",
        alpha=0.95,
        headwidth=8,
        headlength=10,
        headaxislength=9,
    )

    ax.set_xlabel(rf"Phase ${label}$")
    ax.set_ylabel(rf"Phase velocity $\dot{{{label}}}$")
    ax.set_title(rf"Phase Space (${{{label}}}$)")
    return ax


def plot_normal_modes(ax, sol):
    """Plot symmetric and antisymmetric modes vs time."""
    phi1, _, phi2, _ = sol.y
    phi_plus = 0.5 * (phi1 + phi2)
    phi_minus = phi1 - phi2
    ax.plot(sol.t, phi_plus, label=r"$\phi_{+} = (\phi_1+\phi_2)/2$", lw=1.3)
    ax.plot(sol.t, phi_minus, label=r"$\phi_{-} = \phi_1-\phi_2$", lw=1.3)
    ax.set_xlabel(r"Dimensionless time $\tau$")
    ax.set_ylabel("Normal mode coordinate")
    ax.set_title("Normal Modes vs Time")
    ax.legend()
    return ax


def plot_potential_heatmap(ax, model, sol, n_grid=200):
    """
    Heatmap of U(phi1, phi2) with trajectory projection.
    """
    phi1_traj = sol.y[0]
    phi2_traj = sol.y[2]
    phi1_min, phi1_max = phi1_traj.min(), phi1_traj.max()
    phi2_min, phi2_max = phi2_traj.min(), phi2_traj.max()
    m1 = 0.5 * (phi1_max - phi1_min + 1e-6)
    m2 = 0.5 * (phi2_max - phi2_min + 1e-6)
    phi1_grid = np.linspace(phi1_min - m1, phi1_max + m1, n_grid)
    phi2_grid = np.linspace(phi2_min - m2, phi2_max + m2, n_grid)
    PHI1, PHI2 = np.meshgrid(phi1_grid, phi2_grid)
    U = model.potential(PHI1, PHI2)

    hm = ax.pcolormesh(PHI1, PHI2, U, shading="auto", cmap="viridis")
    cb = plt.colorbar(hm, ax=ax)
    cb.set_label(r"Potential $U(\phi_1,\phi_2)$")

    # Overlay trajectory
    ax.plot(phi1_traj, phi2_traj, color="C3", lw=1.2, label="Trajectory")
    ax.set_xlabel(r"$\phi_1$")
    ax.set_ylabel(r"$\phi_2$")
    ax.set_title("Potential Landscape (heatmap)")
    ax.legend()
    return ax


def plot_potential_surface(ax, model, sol, n_grid=80):
    """
    3D surface of U(phi1, phi2) with trajectory overlay.
    """
    phi1_traj = sol.y[0]
    phi2_traj = sol.y[2]
    phi1_min, phi1_max = phi1_traj.min(), phi1_traj.max()
    phi2_min, phi2_max = phi2_traj.min(), phi2_traj.max()
    m1 = 0.5 * (phi1_max - phi1_min + 1e-6)
    m2 = 0.5 * (phi2_max - phi2_min + 1e-6)
    phi1_grid = np.linspace(phi1_min - m1, phi1_max + m1, n_grid)
    phi2_grid = np.linspace(phi2_min - m2, phi2_max + m2, n_grid)
    PHI1, PHI2 = np.meshgrid(phi1_grid, phi2_grid)
    U = model.potential(PHI1, PHI2)

    surf = ax.plot_surface(PHI1, PHI2, U, cmap="viridis", alpha=0.8, linewidth=0)
    cb = plt.colorbar(surf, ax=ax, shrink=0.6, pad=0.1)
    cb.set_label(r"Potential $U(\phi_1,\phi_2)$")

    U_traj = model.potential(phi1_traj, phi2_traj)
    ax.plot(phi1_traj, phi2_traj, U_traj, color="C3", lw=2.0, label="Trajectory")

    ax.set_xlabel(r"$\phi_1$")
    ax.set_ylabel(r"$\phi_2$")
    ax.set_zlabel(r"Potential $U$")
    ax.set_title("Potential Landscape (3D)")
    ax.legend()
    return ax


def plot_potential_overlay_line(ax, model, sol):
    """
    Overlay potential along trajectory using time-coded line for intuition.
    """
    phi1 = sol.y[0]
    phi2 = sol.y[2]
    U = model.potential(phi1, phi2)
    points = np.array([phi1, phi2]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap="plasma", linewidth=2.0)
    lc.set_array(sol.t[:-1])
    ax.add_collection(lc)
    cb = plt.colorbar(lc, ax=ax)
    cb.set_label(r"Time $\tau$")
    ax.set_xlabel(r"$\phi_1$")
    ax.set_ylabel(r"$\phi_2$")
    ax.set_title("Trajectory in (phi1, phi2) with time coding")
    return ax


def animate_phase_space(sol, which="phi1", interval=12):
    """
    Animate a phase-space trajectory (phi, phi_dot) with accumulating path.
    """
    if which == "phi1":
        phi = sol.y[0]
        phi_dot = sol.y[1]
        label = r"\phi_1"
    elif which == "phi2":
        phi = sol.y[2]
        phi_dot = sol.y[3]
        label = r"\phi_2"
    else:
        phi = sol.y[0] - sol.y[2]
        phi_dot = sol.y[1] - sol.y[3]
        label = r"\Delta\phi"

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(phi, phi_dot, color="0.9", lw=0.8)
    (point,) = ax.plot([], [], "o", color="C1")
    (trail,) = ax.plot([], [], color="C0", lw=2.0)

    ax.set_xlabel(rf"Phase ${label}$")
    ax.set_ylabel(rf"Phase velocity $\dot{{{label}}}$")
    ax.set_title(rf"Phase Space (${{{label}}}$)")

    def init():
        point.set_data([], [])
        trail.set_data([], [])
        return point, trail

    def update(i):
        point.set_data([phi[i]], [phi_dot[i]])
        trail.set_data(phi[: i + 1], phi_dot[: i + 1])
        return point, trail

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(phi),
        interval=interval,
        blit=True,
    )
    return anim, fig, ax


def animate_heatmap_trajectory(model, sol, n_grid=200, interval=12):
    """
    Animate trajectory on the 2D potential heatmap with accumulating path.

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
    fig, ax : Figure and Axes used.
    """
    phi1_traj = sol.y[0]
    phi2_traj = sol.y[2]
    phi1_min, phi1_max = phi1_traj.min(), phi1_traj.max()
    phi2_min, phi2_max = phi2_traj.min(), phi2_traj.max()
    m1 = 0.5 * (phi1_max - phi1_min + 1e-6)
    m2 = 0.5 * (phi2_max - phi2_min + 1e-6)
    phi1_grid = np.linspace(phi1_min - m1, phi1_max + m1, n_grid)
    phi2_grid = np.linspace(phi2_min - m2, phi2_max + m2, n_grid)
    PHI1, PHI2 = np.meshgrid(phi1_grid, phi2_grid)
    U = model.potential(PHI1, PHI2)

    fig, ax = plt.subplots(figsize=(6, 5))
    hm = ax.pcolormesh(PHI1, PHI2, U, shading="auto", cmap="viridis")
    cb = plt.colorbar(hm, ax=ax)
    cb.set_label(r"Potential $U(\phi_1,\phi_2)$")

    (line,) = ax.plot([], [], color="C3", lw=2.0, label="Trajectory")
    (point,) = ax.plot([], [], "o", color="w", markersize=6)
    ax.set_xlabel(r"$\phi_1$")
    ax.set_ylabel(r"$\phi_2$")
    ax.set_title("Potential Landscape (animated trajectory)")
    ax.legend(loc="upper left")

    def init():
        line.set_data([], [])
        point.set_data([], [])
        return line, point

    def update(i):
        line.set_data(phi1_traj[: i + 1], phi2_traj[: i + 1])
        point.set_data([phi1_traj[i]], [phi2_traj[i]])
        return line, point

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(phi1_traj),
        interval=interval,
        blit=True,
    )
    return anim, fig, ax


# --------------------------------------------------------------------------- #
# Example usage
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    params = dict(
        kappa=0.2,
        Ic=1e-6,
        C=1e-12,
        R=1e3,
        I_dc=0.3e-6,
        I_ac=0.05e-6,
        omega_drive=2e9,
        phi_drive=0.0,
    )
    y0 = [0.0, 0.0, 0.1, 0.0]
    tau_span = (0.0, 40.0)

    model, sol = run_sim(params, y0, tau_span, num_points=2000)
    if not sol.success:
        raise RuntimeError(f"Solver failed: {sol.message}")

    # Time-domain views
    fig1, axes1 = plt.subplots(2, 2, figsize=(12, 8))
    plot_phases_time(axes1[0, 0], sol)
    plot_delta_phase(axes1[0, 1], sol)
    plot_energy_split(axes1[1, 0], model, sol)
    plot_normal_modes(axes1[1, 1], sol)
    fig1.tight_layout()

    # Phase-space views (static)
    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4))
    plot_phase_space(axes2[0], sol, which="phi1")
    plot_phase_space(axes2[1], sol, which="phi2")
    plot_phase_space(axes2[2], sol, which="delta")
    fig2.tight_layout()

    # Potential landscape visualizations
    fig3, ax3 = plt.subplots(figsize=(6, 5))
    plot_potential_heatmap(ax3, model, sol)
    fig3.tight_layout()

    fig4 = plt.figure(figsize=(7, 5))
    ax4 = fig4.add_subplot(111, projection="3d")
    plot_potential_surface(ax4, model, sol)
    fig4.tight_layout()

    # Animated phase-space trajectories
    anim_phi1, fig_phi1, ax_phi1 = animate_phase_space(sol, which="phi1", interval=10)
    anim_phi2, fig_phi2, ax_phi2 = animate_phase_space(sol, which="phi2", interval=10)
    anim_delta, fig_delta, ax_delta = animate_phase_space(
        sol, which="delta", interval=10
    )

    # Animated heatmap trajectory
    anim_heatmap, fig_anim, ax_anim = animate_heatmap_trajectory(model, sol)

    plt.show()
