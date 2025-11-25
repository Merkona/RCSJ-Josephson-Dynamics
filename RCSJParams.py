"""RCSJ parameters and model definitions."""

import numpy as np

from __future__ import annotations

class RCSJParams:
    """Stores RCSJ junction parameters in SI and computes dimensionless values."""

    # Physical parameters
    e_charge = 1.602176634e-19 
    hbar = 1.054571817e-34
    Ic: 
    C: 
    R: 
    I_dc = 0.0
    I_ac = 0.0
    omega_drive = 0.0
    phi_drive = 0.0

    # Derived dimensionless parameters (computed in __post_init__)
    # omega_p: float = field(init=False)
    # beta_c: float = field(init=False)
    # alpha: float = field(init=False)
    # i_dc: float = field(init=False)
    # i_ac: float = field(init=False)
    # Omega: float = field(init=False)

    def compute_dimensionless(self):
        """Compute dimensionless parameters derived from SI inputs."""
        if self.Ic <= 0:
            raise ValueError("Ic must be positive to compute dimensionless parameters.")
        if self.C <= 0:
            raise ValueError("C must be positive to compute dimensionless parameters.")
        if self.R <= 0:
            raise ValueError("R must be positive to compute dimensionless parameters.")

        # Plasma frequency
        self.omega_p = np.sqrt(2.0 * self.e_charge * self.Ic / (self.hbar * self.C))

        # Stewart-McCumber parameter
        self.beta_c = (2.0 * self.e_charge * self.Ic * (self.R ** 2) * self.C) / self.hbar

        # Damping coefficient
        self.alpha = 1.0 / np.sqrt(self.beta_c) if self.beta_c > 0 else float("inf")

        # Normalized currents
        self.i_dc = self.I_dc / self.Ic
        self.i_ac = self.I_ac / self.Ic

        # Dimensionless drive frequency
        self.Omega = self.omega_drive / self.omega_p if self.omega_p > 0 else 0.0


class RCSJModel(RCSJParams):
    """Implements the RCSJ ODE and related helper computations."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def drive_term(self, tau):
        """Compute the normalized drive current i_dc + i_ac * cos(Omega * tau + phi_drive)."""

        return self.i_dc + self.i_ac * np.cos(self.Omega * tau + self.phi_drive)

    def ode(self, tau, y):
        """RCSJ ODE right-hand side: returns [phi_dot, phi_ddot]."""

        phi, phi_dot = y
        phi_ddot = -self.alpha * phi_dot - np.sin(phi) + self.drive_term(tau)

        return [phi_dot, phi_ddot]

    def potential(self, phi):
        """Dimensionless tilted-washboard potential for given phase."""
        return 1.0 - np.cos(phi) - self.i_dc * phi
