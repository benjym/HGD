import unittest
from types import SimpleNamespace

import numpy as np

from HGD import fluid
from HGD.motion import hgfd


def parameters(**overrides):
    values = dict(
        nx=6,
        ny=8,
        nm=10,
        dx=0.01,
        dy=0.01,
        dt=1e-4,
        g=9.81,
        gas_density=1.225,
        gas_viscosity=1.8e-5,
        solid_density=2500.0,
        nu_fill=0.5,
        nu_cs=0.6,
        s_m=1e-3,
        alpha=0.3,
        fluid_alpha_max=0.3,
        fluid_alpha_epsilon=0.01,
        fluid_min_void_fraction=0.05,
        fluid_drag_min_particle_size=1e-6,
        fluid_inlet_velocity=0.4,
        fluid_probability_policy="error",
        fluid_coupling=True,
        P_stab=0.5,
        boundary_mask=np.zeros((6, 8), dtype=bool),
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class TestFluidCoupling(unittest.TestCase):
    def test_drag_is_zero_in_voids_and_stronger_for_small_grains(self):
        p = parameters(nx=1, ny=1, nm=2, boundary_mask=np.zeros((1, 1), dtype=bool))
        s = np.array([[[5e-4, 1e-3]]])
        zeros = np.zeros_like(s)
        fluid_v = np.ones((1, 1))
        layer_beta, cell_beta = fluid.drag_coefficients(s, zeros, zeros, np.zeros((1, 1)), fluid_v, p)
        self.assertGreater(layer_beta[0, 0, 0], layer_beta[0, 0, 1])
        self.assertGreater(cell_beta[0, 0], 0)

        s[0, 0, 0] = np.nan
        layer_beta, _ = fluid.drag_coefficients(s, zeros, zeros, np.zeros((1, 1)), fluid_v, p)
        self.assertEqual(layer_beta[0, 0, 0], 0)

    def test_upward_gas_accelerates_particle_upward(self):
        p = parameters(nx=1, ny=1, nm=1, dt=1e-3, boundary_mask=np.zeros((1, 1), dtype=bool))
        s = np.array([[[2e-4]]])
        state = fluid.initialize(p, s)
        state.v.fill(2.0)
        u, v = fluid.update_particle_velocities(np.zeros_like(s), np.zeros_like(s), s, state, p)
        self.assertGreater(v[0, 0, 0], 0)
        self.assertEqual(u[0, 0, 0], 0)

    def test_drag_reaction_opposes_upward_gas(self):
        p = parameters(nx=1, ny=1, nm=2, boundary_mask=np.zeros((1, 1), dtype=bool))
        s = np.full((1, 1, 2), p.s_m)
        state = fluid.initialize(p, s)
        state.v.fill(1.0)
        zeros = np.zeros_like(s)
        _, _, _, _, source_v = fluid.coupling_fields(s, zeros, zeros, state, p)
        self.assertLess(source_v[0, 0], 0)

    def test_minimum_fluidization_velocity_balances_dense_drag(self):
        p = parameters()
        superficial = fluid.minimum_fluidization_velocity(p)
        phi = p.nu_fill
        eps = 1.0 - phi
        relative = superficial / eps
        beta = (
            150 * p.gas_viscosity * phi**2 / (eps * p.s_m**2) + 1.75 * p.gas_density * phi * relative / p.s_m
        )
        drag = beta * relative
        weight = phi * (p.solid_density - p.gas_density) * p.g
        self.assertAlmostEqual(drag / weight, 1.0, places=10)

    def test_projection_reduces_weighted_divergence(self):
        p = parameters(fluid_inlet_velocity=0.0)
        rng = np.random.default_rng(4)
        void_fraction = np.full((p.nx, p.ny), 0.65)
        u = rng.normal(scale=0.05, size=(p.nx, p.ny))
        v = rng.normal(scale=0.05, size=(p.nx, p.ny))
        before = np.linalg.norm(fluid.weighted_divergence(u, v, void_fraction, p.dx, p.dy))
        u, v, _, divergence = fluid.project_velocity(u, v, np.zeros_like(u), void_fraction, p)
        self.assertTrue(np.all(np.isfinite(u)))
        self.assertTrue(np.all(np.isfinite(v)))
        self.assertLess(np.linalg.norm(divergence), before)
        self.assertLess(np.max(np.abs(divergence)), 1e-8)

    def test_coupled_step_conserves_solid_slots(self):
        p = parameters()
        s = np.full((p.nx, p.ny, p.nm), np.nan)
        s[:, :3, :5] = p.s_m
        p.fluid_state = fluid.initialize(p, s)
        u = np.zeros_like(s)
        v = np.zeros_like(s)
        last_swap = np.zeros_like(s)
        count_before = np.sum(np.isfinite(s))

        u, v, s, _, _, _, _ = hgfd.move_voids(u, v, s, p, c=None, T=None, chi=None, last_swap=last_swap)
        p.fluid_state = fluid.advance(p.fluid_state, s, u, v, p)
        self.assertEqual(np.sum(np.isfinite(s)), count_before)
        self.assertTrue(np.all(np.isfinite(p.fluid_state.v)))
        self.assertLessEqual(p.max_transition_probability, p.P_stab)


if __name__ == "__main__":
    unittest.main()
