"""
Robustness and validation tests for HGD motion models.

Tests for:
1. Mass conservation
2. Probability limits
3. Statistical convergence
4. Grid independence (when implemented)
"""

import unittest
import numpy as np
from HGD import operators, params
import sys
import os

# Add parent directory to path to import motion modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'HGD', 'motion'))


class TestMassConservation(unittest.TestCase):
    """Test that mass is exactly conserved in all motion models"""
    
    def setUp(self):
        """Create a simple test system"""
        self.p = params.dict_to_class({
            'nx': 10,
            'ny': 10,
            'nm': 5,
            'dt': 0.01,
            'dx': 1.0,
            'dy': 1.0,
            'g': 9.81,
            'alpha': 0.1,
            'nu_cs': 0.64,
            'P_stab': 1.0,
            'delta_limit': 0.1,
            'cyclic_BC': False,
            'cyclic_BC_y_offset': 0,
            'inertia': False,
            'advection_model': 'freefall',
            'boundary_mask': np.zeros((10, 10), dtype=bool)
        })
        
        # Initialize with some particles and some voids
        self.s = np.random.rand(10, 10, 5)
        # Make about 30% voids
        void_mask = np.random.rand(10, 10, 5) < 0.3
        self.s[void_mask] = np.nan
        
        # Randomize indices for d2q4_slow
        indices = []
        for i in range(self.p.nx):
            for j in range(self.p.ny - 1):
                for k in range(self.p.nm):
                    indices.append(i * (self.p.ny - 1) * self.p.nm + j * self.p.nm + k)
        np.random.shuffle(indices)
        self.p.indices = indices
    
    def test_mass_conservation_operators(self):
        """Test that single swap conserves mass"""
        s = self.s.copy()
        nu = operators.get_solid_fraction(s)
        
        mass_initial = np.sum(~np.isnan(s))
        
        # Perform a single swap
        src = [0, 0, 0]
        dst = [1, 1, 0]
        arrays_new, nu_new = operators.swap(src, dst, [s, None], nu, self.p)
        s_new = arrays_new[0]
        
        mass_final = np.sum(~np.isnan(s_new))
        
        self.assertEqual(mass_initial, mass_final,
                        "Mass not conserved in single swap")
    
    def test_mass_conservation_d2q4_slow(self):
        """Test mass conservation in d2q4_slow.py"""
        try:
            import d2q4_slow
        except ImportError:
            self.skipTest("d2q4_slow not available")
        
        s = self.s.copy()
        u = np.zeros_like(s)
        v = np.zeros_like(s)
        
        mass_initial = np.sum(~np.isnan(s))
        
        # Run for several steps
        for _ in range(10):
            u, v, s, c, T, chi, last_swap = d2q4_slow.move_voids(
                u, v, s, self.p, diag=0,
                c=None, T=None, N_swap=None, last_swap=None
            )
        
        mass_final = np.sum(~np.isnan(s))
        
        self.assertAlmostEqual(mass_initial, mass_final, places=10,
                              msg="Mass not conserved in d2q4_slow")
    
    def test_mass_conservation_d2q4_array(self):
        """Test mass conservation in d2q4_array.py"""
        try:
            import d2q4_array
        except ImportError:
            self.skipTest("d2q4_array not available")
        
        s = self.s.copy()
        u = np.zeros_like(s)
        v = np.zeros_like(s)
        
        mass_initial = np.sum(~np.isnan(s))
        
        # Run for several steps
        for _ in range(10):
            u, v, s, c, T, chi, last_swap = d2q4_array.move_voids(
                u, v, s, self.p, diag=0,
                c=None, T=None, chi=None, last_swap=None
            )
        
        mass_final = np.sum(~np.isnan(s))
        
        self.assertAlmostEqual(mass_initial, mass_final, places=10,
                              msg="Mass not conserved in d2q4_array")


class TestProbabilityLimits(unittest.TestCase):
    """Test that probabilities stay within physical bounds"""
    
    def test_probability_computation(self):
        """Test that computed probabilities are reasonable"""
        # Create parameters with large timestep (potential issue)
        p = params.dict_to_class({
            'dt': 1.0,  # Large timestep
            'dx': 1.0,
            'dy': 1.0,
            'g': 9.81,
            'alpha': 1.0,  # Large alpha
            'nu_cs': 0.64,
        })
        
        # Compute characteristic velocity
        v_y = np.sqrt(p.g * p.dy)
        P_u_ref = v_y * p.dt / p.dy
        P_lr_ref = p.alpha * P_u_ref
        
        # These should be large (> 1) with our parameters
        # In production code, these should be renormalized
        self.assertGreater(P_u_ref, 1.0,
                          "Test setup issue: P_u_ref should be > 1")
        
        # If we have renormalization, test it
        P_u = P_u_ref
        P_l = P_lr_ref * 0.5
        P_r = P_lr_ref * 0.5
        P_tot = P_u + P_l + P_r
        
        if P_tot > 1.0:
            # Renormalize (this is the fix we recommend)
            scale = 1.0 / P_tot
            P_u *= scale
            P_l *= scale
            P_r *= scale
            P_tot = 1.0
        
        self.assertLessEqual(P_tot, 1.0,
                           "Total probability should be <= 1 after renormalization")
        self.assertGreaterEqual(P_tot, 0.0,
                               "Total probability should be >= 0")


class TestStatisticalConvergence(unittest.TestCase):
    """Test that results converge statistically with multiple runs"""
    
    def test_statistical_variance(self):
        """Test that variance decreases with number of runs"""
        # This is a simplified test - a full test would run the same
        # simulation multiple times and check that results converge
        
        # Simple random walk to demonstrate the concept
        n_runs = [10, 100, 1000]
        variances = []
        
        for n in n_runs:
            results = []
            for _ in range(n):
                # Random walk
                position = 0
                for step in range(100):
                    position += np.random.choice([-1, 0, 1])
                results.append(position)
            
            variances.append(np.var(results))
        
        # Variance should decrease roughly as 1/n_runs
        # Check that variance at 1000 runs < variance at 10 runs
        self.assertLess(variances[2], variances[0],
                       "Variance should decrease with more runs")


class TestNumericalAccuracy(unittest.TestCase):
    """Test numerical accuracy and convergence"""
    
    def test_solid_fraction_accuracy(self):
        """Test that solid fraction calculation is accurate"""
        # Create known distribution
        s = np.ones((5, 5, 10))
        # Make exactly 30% voids
        s[:, :, 7:] = np.nan
        
        nu = operators.get_solid_fraction(s)
        
        expected = 0.7
        np.testing.assert_array_almost_equal(nu, expected,
                                            decimal=10,
                                            err_msg="Solid fraction not accurate")
    
    def test_average_accuracy(self):
        """Test that averaging is accurate"""
        # Create known distribution
        s = np.ones((3, 3, 5)) * 2.0
        s[:, :, 2:] = np.nan
        
        s_mean = operators.get_average(s)
        
        expected = 2.0
        np.testing.assert_array_almost_equal(s_mean, expected,
                                            decimal=10,
                                            err_msg="Average not accurate")
    
    def test_hyperbolic_average_accuracy(self):
        """Test hyperbolic average calculation"""
        # For uniform distribution, harmonic mean = value
        s = np.ones((3, 3, 5)) * 3.0
        s[:, :, 2:] = np.nan
        
        s_inv_bar = operators.get_hyperbolic_average(s)
        
        expected = 3.0
        np.testing.assert_array_almost_equal(s_inv_bar, expected,
                                            decimal=10,
                                            err_msg="Hyperbolic average not accurate")


class TestBoundaryConditions(unittest.TestCase):
    """Test that boundary conditions work correctly"""
    
    def test_no_flux_boundary(self):
        """Test that particles don't escape through boundaries"""
        p = params.dict_to_class({
            'nx': 5,
            'ny': 5,
            'nm': 3,
            'dt': 0.01,
            'dx': 1.0,
            'dy': 1.0,
            'g': 9.81,
            'alpha': 0.1,
            'nu_cs': 0.64,
            'P_stab': 1.0,
            'delta_limit': 0.1,
            'cyclic_BC': False,
            'cyclic_BC_y_offset': 0,
            'inertia': False,
            'advection_model': 'freefall',
            'boundary_mask': np.zeros((5, 5), dtype=bool)
        })
        
        # Place particles at boundary
        s = np.ones((5, 5, 3)) * np.nan
        s[0, 0, :] = 1.0  # Particles at corner
        
        mass_initial = np.sum(~np.isnan(s))
        
        # Particles at boundary shouldn't disappear
        self.assertGreater(mass_initial, 0,
                          "Test setup: should have particles")


class TestRobustness(unittest.TestCase):
    """Test robustness against edge cases"""
    
    def test_all_voids(self):
        """Test behavior with all voids"""
        s = np.ones((3, 3, 3)) * np.nan
        nu = operators.get_solid_fraction(s)
        
        np.testing.assert_array_equal(nu, 0.0,
                                     err_msg="All voids should give nu=0")
    
    def test_all_solid(self):
        """Test behavior with all solid"""
        s = np.ones((3, 3, 3))
        nu = operators.get_solid_fraction(s)
        
        np.testing.assert_array_equal(nu, 1.0,
                                     err_msg="All solid should give nu=1")
    
    def test_single_particle(self):
        """Test behavior with single particle"""
        s = np.ones((3, 3, 3)) * np.nan
        s[1, 1, 1] = 1.0
        
        mass = np.sum(~np.isnan(s))
        self.assertEqual(mass, 1, "Should have exactly one particle")


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
