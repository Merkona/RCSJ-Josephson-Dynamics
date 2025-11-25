"""Parameter container for the RCSJ Josephson junction model."""

from __future__ import annotations
 
from dataclasses import dataclass, field
from math import sqrt
from typing import Dict

ELEMENTARY_CHARGE = 1.602176634e-19  # Coulombs
HBAR = 1.054571817e-34  # Joule * seconds


@dataclass
class RCSJParams:
    """Stores RCSJ junction parameters in SI and computes dimensionless values."""
#parameters
    Ic: float
    C: float
    R: float
    I_dc: float = 0.0
    I_ac: float = 0.0
    omega_drive: float = 0.0
    phi_drive: float = 0.0

#interesting things we calculate using compute_dimentionless later
    omega_p: float = field(init=False)
    beta_c: float = field(init=False)
    alpha: float = field(init=False)
    i_dc: float = field(init=False)
    i_ac: float = field(init=False)
    Omega: float = field(init=False)

    def __post_init__(self) -> None:
        self.compute_dimensionless()

    def compute_dimensionless(self) -> None:
        """
        Compute dimensionless parameters derived from SI inputs.
        This is mainly useful for ODE simulation
        """
        if self.Ic <= 0:
            raise ValueError("Ic must be positive to compute dimensionless parameters.")
        if self.C <= 0:
            raise ValueError("C must be positive to compute dimensionless parameters.")
        if self.R <= 0:
            raise ValueError("R must be positive to compute dimensionless parameters.")

#the aformentioned interesting things
    #plasma Frequency
        self.omega_p = sqrt(2.0 * ELEMENTARY_CHARGE * self.Ic / (HBAR * self.C))
    #Stewart-McCumber parameter    
        self.beta_c = (
            2.0 * ELEMENTARY_CHARGE * self.Ic * (self.R ** 2) * self.C
        ) / HBAR
    #Damping coeff    
        self.alpha = 1.0 / sqrt(self.beta_c) if self.beta_c > 0 else float("inf")
    #normalized currents    
        self.i_dc = self.I_dc / self.Ic
        self.i_ac = self.I_ac / self.Ic
    #dimensionless drive frequency (for shapiro steps and driven oscillations)
        self.Omega = self.omega_drive / self.omega_p if self.omega_p > 0 else 0.0

#converting to dictionary because why not
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
