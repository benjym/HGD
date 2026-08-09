import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from HGD import fluid, initial, params
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
    def test_paper_case_one_config_uses_cell_centred_mesh_and_one_particle(self):
        config = Path(__file__).parents[1] / "json" / "hgfd_paper_case1_0p5mm.json5"
        with config.open() as handle:
            _, p = params.load_file(handle)
        p.update_before_time_march(None)
        s = initial.IC(p)

        self.assertAlmostEqual(p.dx, 0.05)
        self.assertAlmostEqual(p.dy, 0.05)
        self.assertAlmostEqual(p.y[0], 0.025)
        self.assertAlmostEqual(p.y[-1], 0.625)
        self.assertAlmostEqual(p.W, 0.75)
        self.assertEqual(np.count_nonzero(np.isfinite(s)), 1)
        self.assertEqual(s[p.single_particle_i, p.single_particle_j, p.single_particle_k], p.s_m)

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

    def test_layer_drag_uses_paper_internal_coordinate_normalization(self):
        p = parameters(nx=1, ny=1, nm=4, boundary_mask=np.zeros((1, 1), dtype=bool))
        s = np.array([[[p.s_m, p.s_m, np.nan, np.nan]]])
        zeros = np.zeros_like(s)
        layer_beta, cell_beta = fluid.drag_coefficients(s, zeros, zeros, np.zeros((1, 1)), np.ones((1, 1)), p)
        self.assertAlmostEqual(layer_beta[0, 0, 0], layer_beta[0, 0, 1])
        self.assertAlmostEqual(np.sum(layer_beta[0, 0]), cell_beta[0, 0])
        self.assertAlmostEqual(layer_beta[0, 0, 0], cell_beta[0, 0] / 2.0)

    def test_single_particle_terminal_velocities_match_paper_case_one(self):
        # Fig. 4 model curves, evaluated from Eqs. (7)--(14) using the
        # per-coordinate mass rho_p/M implicit in the layer formulation.
        cases = (
            (0.5e-3, 2560.0, 0.07425),
            (1.5e-3, 2560.0, 0.21144),
            (2.0e-3, 2480.0, 0.26153),
        )
        for diameter, density, expected_terminal_speed in cases:
            p = parameters(
                nx=1,
                ny=1,
                nm=100,
                dt=1e-4,
                gas_density=997.0,
                gas_viscosity=1e-3,
                solid_density=density,
                fluid_inlet_velocity=0.0,
                boundary_mask=np.zeros((1, 1), dtype=bool),
            )
            s = np.full((1, 1, p.nm), np.nan)
            s[0, 0, 0] = diameter
            state = fluid.initialize(p, s)
            u = np.zeros_like(s)
            v = np.zeros_like(s)
            for _ in range(5000):
                u, v = fluid.update_particle_velocities(u, v, s, state, p)
            self.assertAlmostEqual(abs(v[0, 0, 0]), expected_terminal_speed, delta=5e-5)

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

    def test_exhausted_momentum_budget_is_excluded_from_solid_velocity_only(self):
        p = parameters(nx=1, ny=1, nm=2, boundary_mask=np.zeros((1, 1), dtype=bool))
        s = np.full((1, 1, 2), p.s_m)
        state = fluid.initialize(p, s)
        particle_u = np.zeros_like(s)
        particle_v = np.array([[[0.25, 1.0]]])
        p.hgfd_virtual_displacement = np.array([[[0.0, p.dx]]])
        beta, _, solid_v, _, _ = fluid.coupling_fields(s, particle_u, particle_v, state, p)
        self.assertAlmostEqual(solid_v[0, 0], 0.25)
        beta_without_budget, *_ = fluid.coupling_fields(
            s, particle_u, particle_v, state, parameters(nx=1, ny=1, nm=2)
        )
        self.assertAlmostEqual(beta[0, 0], beta_without_budget[0, 0])

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

    def test_uniform_flow_is_unchanged_by_conservative_transport(self):
        p = parameters(fluid_inlet_velocity=0.3)
        s = np.full((p.nx, p.ny, p.nm), np.nan)
        state = fluid.initialize(p, s)
        momentum_u, momentum_v = fluid._conservative_transport(state, state.void_fraction, p)
        np.testing.assert_allclose(momentum_u, state.void_fraction * state.u)
        np.testing.assert_allclose(momentum_v, state.void_fraction * state.v)

    def test_probability_limit_is_strictly_less_than_half(self):
        p = parameters(nx=1, ny=2, nm=1, dx=1.0, dy=1.0, dt=1.0, fluid_alpha_max=0.0)
        p.boundary_mask = np.zeros((1, 2), dtype=bool)
        p.fluid_state = fluid.initialize(p, np.array([[[np.nan], [p.s_m]]]))
        s = np.array([[[np.nan], [p.s_m]]])
        u = np.zeros_like(s)
        v = np.zeros_like(s)
        fixed_v = np.array([[[0.0], [-0.5]]])
        with patch.object(fluid, "update_particle_velocities", return_value=(u, fixed_v)):
            with self.assertRaisesRegex(ValueError, "not below P_stab"):
                hgfd.move_voids(u, v, s, p)

    def test_realised_swap_carries_virtual_displacement_deficit(self):
        p = parameters(
            nx=1,
            ny=2,
            nm=1,
            dx=1.0,
            dy=1.0,
            dt=0.1,
            fluid_alpha_max=0.0,
        )
        p.boundary_mask = np.zeros((1, 2), dtype=bool)
        s = np.array([[[np.nan], [p.s_m]]])
        p.fluid_state = fluid.initialize(p, s)
        p.hgfd_virtual_displacement = np.array([[[0.0], [1.2]]])
        u = np.zeros_like(s)
        fixed_v = np.array([[[0.0], [-0.2]]])
        with patch.object(fluid, "update_particle_velocities", return_value=(u, fixed_v)):
            with patch("numpy.random.random", return_value=np.zeros_like(s)):
                _, _, moved_s, *_ = hgfd.move_voids(u, np.zeros_like(s), s, p)

        self.assertTrue(np.isfinite(moved_s[0, 0, 0]))
        self.assertTrue(np.isnan(moved_s[0, 1, 0]))
        self.assertAlmostEqual(p.hgfd_virtual_displacement[0, 0, 0], 0.22)

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
