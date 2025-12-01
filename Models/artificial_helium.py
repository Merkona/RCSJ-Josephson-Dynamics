"""
    Two-junction 'helium' toy model using the same dimensionless RCSJ
    parameters as RCSJSolve, plus a coupling kappa between the phases.
"""
import numpy as np
from RCSJ_Core import RCSJSolve   # import the solver

class HeliumRCSJSolve(RCSJSolve):

    """
    Two-junction 'helium' toy model using the same dimensionless RCSJ
    parameters as RCSJSolve, plus a coupling kappa between the phases.

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
            Passed through to the usual RCSJParams/RCSJModel constructor:
            Ic, C, R, I_dc, I_ac, omega_drive, phi_drive
        """
        self.kappa = float(kappa)
        super().__init__(**kwargs)

    def ode(self, tau, y):
        """
        Coupled RCSJ ODE right-hand side for the helium toy model.

        Input:
            tau : dimensionless time
            y   : [phi1, phi1_dot, phi2, phi2_dot]

        Output:
            [phi1_dot, phi1_ddot, phi2_dot, phi2_ddot]
        """
        phi1, phi1_dot, phi2, phi2_dot = y

        # Same normalized drive on both junctions for a symmetric "helium"
        i_drive = self.drive_term(tau)

        # Second derivatives with coupling
        coupling = self.kappa * (phi1 - phi2)

        phi1_ddot = -self.alpha * phi1_dot - np.sin(phi1) + i_drive - coupling
        phi2_ddot = -self.alpha * phi2_dot - np.sin(phi2) + i_drive + coupling

        return [phi1_dot, phi1_ddot, phi2_dot, phi2_ddot]

    def helium_potential(self, phi1, phi2):
        """
        Dimensionless 'helium' potential energy surface:

            U = [1 - cos(phi1) - i_dc * phi1]
              + [1 - cos(phi2) - i_dc * phi2]
              + 0.5 * kappa * (phi1 - phi2)^2

        This is just for analysis/plotting; it is not used by the solver.
        """
        # Individual tilted-washboard terms (reuse scalar potential logic)
        U1 = 1.0 - np.cos(phi1) - self.i_dc * phi1
        U2 = 1.0 - np.cos(phi2) - self.i_dc * phi2

        # "Electron-electron" repulsion term
        U_coup = 0.5 * self.kappa * (phi1 - phi2) ** 2

        return U1 + U2 + U_coup

    def helium_energy(self, phi1, phi1_dot, phi2, phi2_dot):
        """
        Total dimensionless 'energy' for a given state:
            E = T + U

        where
            T = 0.5 * (phi1_dot^2 + phi2_dot^2)
            U = helium_potential(phi1, phi2)
        """
        T = 0.5 * (phi1_dot**2 + phi2_dot**2)
        U = self.helium_potential(phi1, phi2)
        return T + U

if __name__ == "__main__":
    # Example: helium toy model with two identical JJs

    helium = HeliumRCSJSolve(
        kappa=0.2,        # coupling strength between the two "electrons"
        Ic=1e-6,          # 1 µA
        C=1e-12,          # 1 pF
        R=1e3,            # 1 kΩ
        I_dc=0.3e-6,      # shared DC bias
        I_ac=0.05e-6,     # shared AC drive (set to 0 for no drive)
        omega_drive=2e9,  # drive frequency (Hz)
        phi_drive=0.0,
    )

    # Initial conditions: both near same phase, small velocities
    y0 = [0.0, 0.0, 0.1, 0.0]  # [phi1, phi1_dot, phi2, phi2_dot]

    tau_span = (0.0, 200.0)
    t_eval = np.linspace(*tau_span, 5000)

    sol = helium.solve(y0=y0, tau_span=tau_span, t_eval=t_eval)

    print("Success:", sol.success)
    print("Final phi1, phi2:", sol.y[0, -1], sol.y[2, -1])
    print("Final phi1_dot, phi2_dot:", sol.y[1, -1], sol.y[3, -1])
    print("status:", sol.status)
    print("message:", sol.message)

    # Example: compute total 'energy' at final time
    E_final = helium.helium_energy(
        sol.y[0, -1], sol.y[1, -1], sol.y[2, -1], sol.y[3, -1]
    )
    print("Final dimensionless helium energy:", E_final)
