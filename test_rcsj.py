
import unittest
import numpy as np
from RCSJ_methods import RCSJModel, RCSJParams, RCSJSolve

class TestRCSJParams(unittest.TestCase):
    """Test the RCSJParams Class"""

    def test_dimensionless(self):

        params = RCSJParams(Ic=1e-6, C=1e-12, R=1e-3)

        self.assertAlmostEqual(params.omega_p, 5.512290719491326908e10)
        self.assertAlmostEqual(params.beta_c, 3.0385348976e-9)
        self.assertAlmostEqual(params.alpha, 1.81412782976780e4)
        self.assertAlmostEqual(params.i_dc, 0)
        self.assertAlmostEqual(params.i_ac, 0)
        self.assertAlmostEqual(params.Omega, 0)

    def test_dimensionless_invalid(self):

        with self.assertRaises(ValueError):
            RCSJParams(Ic=0, C=1e-12, R=1e3)
        with self.assertRaises(ValueError):
            RCSJParams(Ic=1e-6, C=0, R=1e3)
        with self.assertRaises(ValueError):
            RCSJParams(Ic=1e-6, C=1e-12, R=0)


class TestRCSJModel(unittest.TestCase):
    """Test the RCSJModel class"""

    def setUp(self):
        """Example parameters for testing"""
        self.model = RCSJModel(
            Ic=1e-6,
            C=1e-12,
            R=1e3,
            I_dc=0.5e-6,
            I_ac=0.2e-6,
            omega_drive=2e9,
            phi_drive=0.3,
        )

    def test_drive_term(self):
        
        tau = 0.5
        val = self.model.drive_term(tau)
        expected = self.model.i_dc + self.model.i_ac * np.cos(self.model.Omega * tau + self.model.phi_drive)

        self.assertAlmostEqual(val, expected)

    def test_ode_ouput(self):

        tau = 0.0
        y = [0.1, 0.2]
        dy = self.model.ode(tau, y)

        self.assertAlmostEqual(dy[0], y[1])
        self.assertAlmostEqual(dy[1], -self.model.alpha * y[1] - np.sin(y[0]) + self.model.drive_term(tau))

    def test_potential(self):

        phi = 1.0
        U = self.model.potential(phi)
        expected = 1.0 - np.cos(phi) - self.model.i_dc * phi

        self.assertAlmostEqual(U, expected)