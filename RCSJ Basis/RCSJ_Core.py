"""RCSJ model parameters and solver utilities."""
import numpy as np
from scipy.integrate import solve_ivp


class RCSJParams:
    """
    Stores RCSJ junction parameters in SI and computes dimensionless values.
    All values are floats.
    
    Inputs: 
    Ic - Critical current of JJ
    C - Junction capacitance
    R - Shunt resistance
    I_dc - DC bias current
    I_ac - AC drive current
    omega_drive - AC drive frequency
    phi_drive - AC drive phase offset

    Calculated Values:
    omega_p - Characteristic plasma frequency
    beta_c - Stewart-McCumber Parameter (dimensionless measurement of damping)
    alpha - Dimensionless damping parameter
    i_dc - Normalized DC bias current
    i_ac - Normalized AC drive current
    Omega - Normalized AC drive frequency

    """

    def __init__(self, Ic, C, R, I_dc = 0.0, I_ac = 0.0, omega_drive = 0.0, phi_drive = 0.0):
        # Physical parameters
        self.e_charge = 1.602176634e-19 
        self.hbar = 1.054571817e-34
        self.Ic = Ic
        self.C = C
        self.R = R
        self.I_dc = I_dc
        self.I_ac = I_ac
        self.omega_drive = omega_drive
        self.phi_drive = phi_drive

        # self. is required to run functions defined within the class.
        self.compute_dimensionless()



    def compute_dimensionless(self):
        """Compute dimensionless parameters derived from SI inputs."""

        # Checks that inputs are physically meaningful and won't cause division by zero errors.
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
        self.alpha = 1.0 / np.sqrt(self.beta_c)

        # Normalized currents
        self.i_dc = self.I_dc / self.Ic
        self.i_ac = self.I_ac / self.Ic

        # Dimensionless drive frequency
        self.Omega = self.omega_drive / self.omega_p


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


    def solve(self, *args, **kwargs):
        #*args and **kwargs is included here so that the parent class' solve (this one)
        # allows the subclass' to define any input parameters for it's version of solve

        """Placeholder — overridden by solver subclasses."""
        raise NotImplementedError("Choose a solver subclass and call its solve()")


class RCSJSolve(RCSJModel):
    """Numerical solver for the RCSJ ODE using scipy.solve_ivp."""

    def solve(
        self,
        y0,
        tau_span,
        t_eval=None,
        max_step=np.inf,
        rtol=1e-6,
        atol=1e-9,
    ):
        """
        Solves the RCSJ ODE using the parameters defined in solve_ivp() below and 
        returns the sol object from scipy.integrate.solve_ivp, which contains the following:
        
        sol.t → array of times
        
        sol.y → array of states (phi and phi_dot)
        
        sol.success → True/False
        
        sol.status → what happened
        
        sol.message → summary
        
        sol.sol() (if dense_output=True) → interpolating function, probably won't need this
        """

        sol = solve_ivp(
            # The function that gives the time derivative of your system of equations. 
            # [phi_dot, phi_ddot])
            fun=self.ode,

            # (tau_start, tau_end) in dimensionless time. If None, solver chooses its own steps.
            t_span=tau_span,

            # Initial state [phi0, phi_dot0].
            y0=y0,

            # Times at which to store the computed solution (in dimensionless tau).
            t_eval=t_eval,

            # Maximum step size for the integrator (in tau).
            max_step=max_step,

            # Relative and absolute tolerances for solve_ivp.
            # The current values chosen are a starting point,
            # depending on runtime and accuracy we can tighten or loosen it later
            rtol=rtol,
            atol=atol,
        )
        return sol

if __name__ == "__main__":
    # Example parameters
    jj = RCSJSolve(
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
    tau_span = (0.0, 100.0)

    #5000 equivalent time steps that span tau
    t_eval = np.linspace(*tau_span, 5000)

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
