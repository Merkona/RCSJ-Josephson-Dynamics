import unittest
import numpy as np
from RCSJ_Basis.coupled_junction import CoupledRCSJSolve


class TestCoupledRCSJSolve(unittest.TestCase):

    def setUp(self):
        self.model = CoupledRCSJSolve(
            kappa=0.2,
            Ic=1e-6,
            C=1e-12,
            R=1e3,
            I_dc=0.3e-6,
            I_ac=0.05e-6,
            omega_drive=2e9,
            phi_drive=0.0,
        )

    def test_init(self):
        """Model initialization should call the superclass to initialize parameters"""
        self.assertAlmostEqual(self.model.kappa, 0.2)
        self.assertAlmostEqual(self.model.Ic, 1e-6)
        self.assertAlmostEqual(self.model.C, 1e-12)
        self.assertAlmostEqual(self.model.R, 1e3)
        self.assertAlmostEqual(self.model.I_dc, 0.3e-6)
        self.assertAlmostEqual(self.model.I_ac, 0.05e-6)
        self.assertGreater(self.model.alpha, 0.0)
        self.assertGreater(self.model.Omega, 0.0)

    def test_ode1(self):
        """The result of the ode method should be finite and the first derivative of phi1 and phi2
        should be phi1_dot and phi_2 dot"""
        y = [0.0, 0.1, 0.2, -0.1]
        dy = self.model.ode(0.0, y)

        self.assertEqual(len(dy), 4)
        self.assertTrue(all(np.isfinite(dy)))

        self.assertAlmostEqual(dy[0], y[1])
        self.assertAlmostEqual(dy[2], y[3])

    def test_ode2(self):
        """Increasing kappa should increase the coupling term"""
        y = [1.0, 0.0, 0.0, 0.0]

        # Small kappa
        model_small = CoupledRCSJSolve(
            kappa=0.01, Ic=1e-6, C=1e-12, R=1e3
        )
        dy_small = model_small.ode(0.0, y)
        phi1_ddot_small = dy_small[1]
        phi2_ddot_small = dy_small[3]

        # Large kappa
        model_large = CoupledRCSJSolve(
            kappa=1.0, Ic=1e-6, C=1e-12, R=1e3
        )
        dy_large = model_large.ode(0.0, y)
        phi1_ddot_large = dy_large[1]
        phi2_ddot_large = dy_large[3]

        self.assertGreater(phi1_ddot_small, phi1_ddot_large)
        self.assertLess(phi2_ddot_small, phi2_ddot_large)

    def test_potential(self):
        phi1 = 1.0
        phi2 = -0.5

        U = self.model.potential(phi1, phi2)

        # Compute expected
        U1 = 1 - np.cos(phi1) - self.model.i_dc * phi1
        U2 = 1 - np.cos(phi2) - self.model.i_dc * phi2
        Uc = 0.5 * self.model.kappa * (phi1 - phi2) ** 2
        expected = U1 + U2 + Uc

        self.assertAlmostEqual(U, expected)

    def test_energy(self):
        E = self.model.energy(1.0, 0.3, -0.5, -0.1)

        T = 0.5 * (0.3**2 + (-0.1)**2)
        U = self.model.potential(1.0, -0.5)
        expected = T + U

        self.assertAlmostEqual(E, expected)

    def test_solver_1(self):
        y0 = [0.0, 0.0, 0.1, 0.0]
        tau_span = (0.0, 5.0)
        t_eval = np.linspace(0, 5, 200)

        sol = self.model.solve(y0=y0, tau_span=tau_span, t_eval=t_eval)

        self.assertTrue(sol.success)
        self.assertEqual(sol.y.shape, (4, 200))
        self.assertEqual(sol.t.shape, (200,))

    def test_solver_2(self):
        y0 = [0.0, 0.0, 0.1, 0.0]
        tau_span = (0.0, 1.0)

        sol = self.model.solve(y0=y0, tau_span=tau_span)

        self.assertTrue(sol.success)
        self.assertGreater(sol.t.size, 1)
        self.assertEqual(sol.y.shape[0], 4)


if __name__ == "__main__":
    unittest.main()