"""
Coupled RCSJ model for two Josephson junctions with linear phase coupling.

This is a generic two-junction extension of the single-junction RCSJSolve
class defined in RCSJ_Core.py. It is used by higher-level "artificial atom"
models such as the helium toy model in artificial_helium.py.
"""

import numpy as np
from .RCSJ_Core import RCSJSolve


class CoupledRCSJSolve(RCSJSolve):
    """
    Two-junction coupled RCSJ model using the same dimensionless parameters
    as RCSJSolve, plus a coupling kappa between the phases.

    State vector:
        y = [phi1, phi1_dot, phi2, phi2_dot]

    Equations (dimensionless time tau):
        phi1_ddot = -alpha * phi1_dot - sin(phi1)
                    + drive(tau) - kappa * (phi1 - phi2)

        phi2_ddot = -alpha * phi2_dot - sin(phi2)
                    + drive(tau) + kappa * (phi1 - phi2)

    Both junctions share the same drive_term (same Ic, C, R, I_dc, I_ac, etc.).
    """

    def __init__(self, kappa, **kwargs):
        """
        Parameters
        ----------
        kappa : float
            Dimensionless coupling strength between the two junctions.
            kappa > 0 acts like a repulsive interaction between phi1 and phi2.
        **kwargs :
            Passed through to the usual RCSJParams/RCSJModel/RCSJSolve
            constructor: Ic, C, R, I_dc, I_ac, omega_drive, phi_drive, etc.
        """
        self.kappa = float(kappa)
        super().__init__(**kwargs)

    def ode(self, tau, y):
        """
        Coupled RCSJ ODE right-hand side.

        Input
        -----
        tau : float
            Dimensionless time.
        y : array-like
            State vector [phi1, phi1_dot, phi2, phi2_dot].

        Returns
        -------
        dydt : list
            Time derivatives [phi1_dot, phi1_ddot, phi2_dot, phi2_ddot].
        """
        phi1, phi1_dot, phi2, phi2_dot = y

        # Same normalized drive on both junctions for a symmetric system
        i_drive = self.drive_term(tau)

        # Second derivatives with coupling
        coupling = self.kappa * (phi1 - phi2)

        phi1_ddot = -self.alpha * phi1_dot - np.sin(phi1) + i_drive - coupling
        phi2_ddot = -self.alpha * phi2_dot - np.sin(phi2) + i_drive + coupling

        return [phi1_dot, phi1_ddot, phi2_dot, phi2_ddot]

    def potential(self, phi1, phi2):
        """
        Dimensionless potential energy surface for the coupled junctions:

            U = [1 - cos(phi1) - i_dc * phi1]
              + [1 - cos(phi2) - i_dc * phi2]
              + 0.5 * kappa * (phi1 - phi2)^2

        This is intended for analysis/plotting; it is not used by the solver.

        Parameters
        ----------
        phi1, phi2 : float or ndarray
            Junction phases.

        Returns
        -------
        U : float or ndarray
            Dimensionless potential.
        """
        # Individual tilted-washboard terms
        U1 = 1.0 - np.cos(phi1) - self.i_dc * phi1
        U2 = 1.0 - np.cos(phi2) - self.i_dc * phi2

        # Coupling term
        U_coup = 0.5 * self.kappa * (phi1 - phi2) ** 2

        return U1 + U2 + U_coup

    def energy(self, phi1, phi1_dot, phi2, phi2_dot):
        """
        Total dimensionless energy for a given state:

            E = T + U

        where
            T = 0.5 * (phi1_dot^2 + phi2_dot^2)
            U = potential(phi1, phi2)

        Parameters
        ----------
        phi1, phi2 : float or ndarray
            Phases.
        phi1_dot, phi2_dot : float or ndarray
            Phase velocities.

        Returns
        -------
        E : float or ndarray
            Dimensionless total energy.
        """
        T = 0.5 * (phi1_dot**2 + phi2_dot**2)
        U = self.potential(phi1, phi2)
        return T + U


if __name__ == "__main__":
    # Minimal smoke test for the coupled junction model
    cj = CoupledRCSJSolve(
        kappa=0.2,
        Ic=1e-6,
        C=1e-12,
        R=1e3,
        I_dc=0.3e-6,
        I_ac=0.0,
        omega_drive=2e9,
        phi_drive=0.0,
    )

    y0 = [0.0, 0.0, 0.1, 0.0]
    tau_span = (0.0, 50.0)

    sol = cj.solve(y0=y0, tau_span=tau_span, t_eval=np.linspace(*tau_span, 2000))
    print("CoupledJunction success:", sol.success)
    print("Final phi1, phi2:", sol.y[0, -1], sol.y[2, -1])
    print("Final phi1_dot, phi2_dot:", sol.y[1, -1], sol.y[3, -1])
    print("status:", sol.status)
    print("message:", sol.message)

    # Example: compute total 'energy' at final time
    E_final = cj.energy(
        sol.y[0, -1], sol.y[1, -1], sol.y[2, -1], sol.y[3, -1]
    )
    print("Final dimensionless energy:", E_final)
