
import unittest
import numpy as np
from RCSJ_Basis.single_junction import SingleRCSJSolve

def make_test_solver():
    """Create a solver instance with simple test parameters."""
    return SingleRCSJSolve(
        Ic=1e-6,
        C=1e-12,
        R=1e3,
        I_dc=0.4e-6,
        I_ac=0.0,
        omega_drive=0.0,
        phi_drive=0.0,
    )

class TestSingleRCSJSolve(unittest.TestCase):,
    
    def test_potential(self):
        jj = make_test_solver()

        phi = 0
        expected = 1.0 - np.cos(phi) - jj.i_dc * phi
        
        self.assertAlmostEqual(expected, jj.potential(phi))

    def test_energy_scalar(self):
        jj = make_test_solver()

        phi = 0.3
        phi_dot = 0.5

        kinetic = 0.5 * phi_dot**2
        potential = jj.potential(phi)
        expected = kinetic + potential

        self.assertAlmostEqual(jj.energy(phi, phi_dot), expected)

    def test_energy_vector(self):
        jj = make_test_solver()

        phi = np.array([0.0, 0.2, 0.4])
        phi_dot = np.array([0.0, 0.3, 0.6])

        E = jj.energy(phi, phi_dot)

        # Ensure vectorized output has correct shape
        self.assertIsInstance(E, np.ndarray)
        self.assertEqual(E.shape, phi.shape)

    def test_inheritance(self):
        """Verify class inherits from RCSJSolve."""
        jj = make_test_solver()
        self.assertIsInstance(jj, RCSJSolve)

    def test_solve_runs(self):
        jj = make_test_solver()

        y0 = [0.0, 0.0]
        tau_span = (0.0, 5.0)
        t_eval = np.linspace(0.0, 5.0, 50)

        sol = jj.solve(y0=y0, tau_span=tau_span, t_eval=t_eval)

        # Basic solver success checks
        self.assertTrue(sol.success)
        self.assertEqual(sol.y.shape, (2, len(t_eval)))
        self.assertEqual(sol.t.size, len(t_eval))

if __name__ == "__main__":
    unittest.main()