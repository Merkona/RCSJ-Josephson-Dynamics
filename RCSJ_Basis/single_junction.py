"""
Single-junction RCSJ model as a thin wrapper around RCSJSolve.

This module provides:
    - SingleRCSJSolve: a convenience class for a single Josephson junction
      with:
        * the standard RCSJ ODE
        * access to the tilted-washboard potential
        * a total (dimensionless) energy function
"""

import numpy as np
from RCSJ_Basis.RCSJ_Core import RCSJSolve


class SingleRCSJSolve(RCSJSolve):
    """
    Single-junction RCSJ model using the dimensionless parameters and ODE
    defined in RCSJ_Core.RCSJModel / RCSJSolve.
    """

    def potential(self, phi):
        """Dimensionless tilted-washboard potential for given phase."""
        return 1.0 - np.cos(phi) - self.i_dc * phi

    def energy(self, phi, phi_dot):
        """
        Compute the total dimensionless energy for a given phase and
        phase velocity:

            E = 0.5 * phi_dot^2 + U(phi)

        where U(phi) is the tilted-washboard potential defined in
        RCSJModel.potential().

        Parameters
        ----------
        phi : float or ndarray
            Phase difference across the junction.
        phi_dot : float or ndarray
            Dimensionless phase velocity.

        Returns
        -------
        E : float or ndarray
            Dimensionless total energy.
        """
        T = 0.5 * phi_dot**2
        U = self.potential(phi)  # inherited from RCSJModel
        return T + U
    
if __name__ == "__main__":
    # Example parameters
    jj = SingleRCSJSolve(
        Ic=1e-6,          # 1 µA
        C=1e-12,          # 1 pF
        R=1e3,            # 1 kΩ
        I_dc=0.5e-6,      # 0.5 µA
        I_ac=0.2e-6,      # 0.2 µA
        omega_drive=2e9,  # 2 GHz
        phi_drive=0.0,
    )

    # Initial conditions in dimensionless time
    # start at bottom of well, zero velocity
    y0 = [0.0, 0.0]

    # Arbitrary range of Tau
    tau_span = (0.0, 50.0)

    #5000 equivalent time steps that span tau
    t_eval = np.linspace(*tau_span, 2000)

    sol = jj.solve(y0=y0, tau_span=tau_span, t_eval=t_eval)

    # Print final phase and phase velocity, along with status and message
    print("Success:", sol.success)
    print("Final phi:", sol.y[0, -1])
    print("Final phi_dot:", sol.y[1, -1])

    # Status 0 means the solver ran smoothly,
    # 1 means it stopped upon a certian inputted condition
    # -1 means it failed
    print("status:", sol.status)
    print("message:", sol.message)

    # Compute final total energy
    E_final = jj.energy(sol.y[0, -1], sol.y[1, -1])
    print("Final dimensionless energy:", E_final)
