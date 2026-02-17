from types import SimpleNamespace

import numpy as np
import pytest

try:
    from HGD.motion import d2q4_cpp
except ImportError:  # pragma: no cover
    d2q4_cpp = None


pytestmark = pytest.mark.skipif(d2q4_cpp is None, reason="d2q4_cpp extension is not available")


def _make_particle_params(nx=5, ny=5, nm=10, nu_cs=0.5):
    return SimpleNamespace(
        g=9.81,
        dt=1.0,
        dx=1.0,
        dy=1.0,
        alpha=1.0,
        nu_cs=nu_cs,
        P_stab=0.5,
        delta_limit=0.1,
        seg_exponent=0.1,
        beta=6.0,
        cyclic_BC=False,
        inertia=True,
        cyclic_BC_y_offset=0,
        nx=nx,
        ny=ny,
        nm=nm,
        move_type="particle",
        max_threads=1,
        boundary_mask=np.zeros((nx, ny), dtype=bool),
        space_criterion="nu_cs",
        tau=0.0,
    )


def _run_move_particles(u, v, s, p):
    out = d2q4_cpp.move_voids(u, v, s, p, 0, None, None, None, None)
    return out[0], out[1], out[2]


def test_move_particles_core_only_updates_local_velocity_entries():
    p = _make_particle_params(nx=6, ny=6, nm=10, nu_cs=0.6)
    rng = np.random.default_rng(42)

    s = np.full((p.nx, p.ny, p.nm), np.nan, dtype=np.float64)
    # one particle only
    s[3, 4, 0] = 1.0

    # start with stale velocity everywhere
    u = rng.normal(size=(p.nx, p.ny, p.nm))
    v = rng.normal(size=(p.nx, p.ny, p.nm))

    u_out, v_out, s_out = _run_move_particles(u, v, s, p)

    # Particle motion should only touch velocity entries participating in a swap.
    # We intentionally do not enforce global void-velocity cleanup here.
    changed = (~np.isclose(u_out, u)) | (~np.isclose(v_out, v))
    assert np.count_nonzero(changed) <= 2


def test_move_particles_core_respects_nu_cs_cap():
    p = _make_particle_params(nx=5, ny=5, nm=10, nu_cs=0.5)

    s = np.full((p.nx, p.ny, p.nm), np.nan, dtype=np.float64)
    u = np.zeros((p.nx, p.ny, p.nm), dtype=np.float64)
    v = np.zeros((p.nx, p.ny, p.nm), dtype=np.float64)

    src = (2, 1, 0)
    dst = (2, 0)

    # Candidate moving particle (downward target only).
    s[src] = 1.0

    # Destination already at nu_cs = 5/10.
    for k in range(1, 6):
        s[dst[0], dst[1], k] = 1.0

    # Mask side neighbors so downward is the only possible move.
    p.boundary_mask[1, 1] = True
    p.boundary_mask[3, 1] = True

    _, _, s_out = _run_move_particles(u, v, s, p)

    nu_dest = np.mean(~np.isnan(s_out[dst[0], dst[1], :]))
    assert nu_dest <= p.nu_cs + 1e-12

    # The source particle should remain if destination is capped.
    assert not np.isnan(s_out[src])
    assert np.isnan(s_out[dst[0], dst[1], src[2]])


def test_single_particle_falls_at_gravity_rate_with_inertia():
    p = _make_particle_params(nx=3, ny=7, nm=1, nu_cs=1.0)
    p.g = 1.0
    p.dt = 1.0
    p.dx = 1.0
    p.dy = 1.0

    # Constrain to 1-D vertical motion so gravity-only physics is testable.
    p.boundary_mask[0, :] = True
    p.boundary_mask[2, :] = True

    s = np.full((p.nx, p.ny, p.nm), np.nan, dtype=np.float64)
    u = np.zeros((p.nx, p.ny, p.nm), dtype=np.float64)
    v = np.zeros((p.nx, p.ny, p.nm), dtype=np.float64)
    s[1, 6, 0] = 1.0

    expected_speed_increment = np.sqrt(p.g * p.dy)
    for step in range(1, 6):
        u, v, s = _run_move_particles(u, v, s, p)
        i, j, k = np.argwhere(~np.isnan(s))[0]
        assert (int(i), int(j), int(k)) == (1, 6 - step, 0)
        assert u[i, j, k] == pytest.approx(0.0, abs=1e-12)
        assert v[i, j, k] == pytest.approx(-step * expected_speed_increment, abs=1e-12)
