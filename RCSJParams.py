"""RCSJ parameters and model definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, sin, sqrt
from typing import Dict, Sequence

ELEMENTARY_CHARGE = 1.602176634e-19  # Coulombs
HBAR = 1.054571817e-34  # Joule * seconds


@dataclass
class RCSJParams:
    """Stores RCSJ junction parameters in SI and computes dimensionless values."""

    # Physical parameters
    Ic: float
    C: float
    R: float
    I_dc: float = 0.0
    I_ac: float = 0.0
    omega_drive: float = 0.0
    phi_drive: float = 0.0

    # Derived dimensionless parameters (computed in __post_init__)
    omega_p: float = field(init=False)
    beta_c: float = field(init=False)
    alpha: float = field(init=False)
    i_dc: float = field(init=False)
    i_ac: float = field(init=False)
    Omega: float = field(init=False)

    def __post_init__(self) -> None:
        self.compute_dimensionless()

    def compute_dimensionless(self) -> None:
        """Compute dimensionless parameters derived from SI inputs."""
        if self.Ic <= 0:
            raise ValueError("Ic must be positive to compute dimensionless parameters.")
        if self.C <= 0:
            raise ValueError("C must be positive to compute dimensionless parameters.")
        if self.R <= 0:
            raise ValueError("R must be positive to compute dimensionless parameters.")

        # Plasma frequency
        self.omega_p = sqrt(2.0 * ELEMENTARY_CHARGE * self.Ic / (HBAR * self.C))
        # Stewart-McCumber parameter
        self.beta_c = (2.0 * ELEMENTARY_CHARGE * self.Ic * (self.R ** 2) * self.C) / HBAR
        # Damping coefficient
        self.alpha = 1.0 / sqrt(self.beta_c) if self.beta_c > 0 else float("inf")
        # Normalized currents
        self.i_dc = self.I_dc / self.Ic
        self.i_ac = self.I_ac / self.Ic
        # Dimensionless drive frequency
        self.Omega = self.omega_drive / self.omega_p if self.omega_p > 0 else 0.0

    def to_dict(self) -> Dict[str, float]:
        """Return a dictionary with both SI and dimensionless parameters."""
        return {
            "Ic": self.Ic,
            "C": self.C,
            "R": self.R,
            "I_dc": self.I_dc,
            "I_ac": self.I_ac,
            "omega_drive": self.omega_drive,
            "phi_drive": self.phi_drive,
            "omega_p": self.omega_p,
            "beta_c": self.beta_c,
            "alpha": self.alpha,
            "i_dc": self.i_dc,
            "i_ac": self.i_ac,
            "Omega": self.Omega,
        }

class RCSJModel(RCSJParams):
    """Implements the RCSJ ODE and related helper computations."""

    def drive_term(self, tau: float) -> float:
        """Compute the normalized drive current i_dc + i_ac * cos(Omega * tau + phi_drive)."""
        p = self.params
        return p.i_dc + p.i_ac * cos(p.Omega * tau + p.phi_drive)

    def ode(self, tau: float, y: Sequence[float]) -> list[float]:
        """RCSJ ODE right-hand side: returns [phi_dot, phi_ddot]."""
        phi, phi_dot = y
        p = self.params
        phi_ddot = -p.alpha * phi_dot - sin(phi) + self.drive_term(tau)
        return [phi_dot, phi_ddot]

    def potential(self, phi: float) -> float:
        """Dimensionless tilted-washboard potential for given phase."""
        return 1.0 - cos(phi) - self.params.i_dc * phi

class RCSJ:
    """Superclass for RCSJ-based models that owns the parameter set."""

    def __init__(self, params: RCSJParams) -> None:
        self.params = params

    def update_params(self, **kwargs: float) -> None:
        """Update parameter values and recompute derived quantities."""
        for name, value in kwargs.items():
            if not hasattr(self.params, name):
                raise AttributeError(f"Unknown parameter: {name}")
            setattr(self.params, name, value)
        self.params.compute_dimensionless()

print("test")