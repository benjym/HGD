from types import SimpleNamespace

import numpy as np
import pytest

try:
    from HGD.motion import d2q4_cpp
except ImportError:  # pragma: no cover
    d2q4_cpp = None


def _make_params(nx=9, ny=9, nm=10):
    return SimpleNamespace(
        g=9.81,
        dt=1.0,
        dx=1.0,
        dy=1.0,
        alpha=1.0,
        nu_cs=0.6,
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
        stream_model="lbm_zero_eq",
        space_criterion="nu_cs",
        tau=0.0,
    )


def _particle_pos(s):
    idx = np.argwhere(~np.isnan(s))
    if len(idx) != 1:
        raise AssertionError(f"Expected exactly one particle, got {len(idx)}")
    i, j, k = idx[0]
    return int(i), int(j), int(k)


def _single_particle_state(start=(4, 7, 0), vx=0.0, vy=0.0, nx=9, ny=9, nm=10):
    s = np.full((nx, ny, nm), np.nan, dtype=np.float64)
    u = np.zeros((nx, ny, nm), dtype=np.float64)
    v = np.zeros((nx, ny, nm), dtype=np.float64)
    s[start] = 1.0
    u[start] = vx
    v[start] = vy
    return u, v, s


pytestmark = pytest.mark.skipif(d2q4_cpp is None, reason="d2q4_cpp extension is not available")


def test_single_particle_falls_with_local_velocity():
    p = _make_params(nx=9, ny=9, nm=10)
    u, v, s = _single_particle_state(start=(4, 7, 0), vx=0.0, vy=-1.0, nx=p.nx, ny=p.ny, nm=p.nm)

    trajectory = []
    for _ in range(8):
        i, j, _ = _particle_pos(s)
        trajectory.append((i, j))
        u, v, s = d2q4_cpp.stream(u, v, s, p)

    assert trajectory == [(4, 7), (4, 6), (4, 5), (4, 4), (4, 3), (4, 2), (4, 1), (4, 0)]


def test_particle_velocity_is_streamed_with_particle():
    p = _make_params(nx=9, ny=9, nm=10)
    vx0, vy0 = 0.4, -0.3
    u, v, s = _single_particle_state(start=(4, 7, 0), vx=vx0, vy=vy0, nx=p.nx, ny=p.ny, nm=p.nm)

    for _ in range(8):
        i, j, k = _particle_pos(s)
        assert np.isclose(u[i, j, k], vx0)
        assert np.isclose(v[i, j, k], vy0)
        u, v, s = d2q4_cpp.stream(u, v, s, p)


@pytest.mark.parametrize(
    "vx, vy, expected",
    [
        (0.0, -0.75, {"down": 0.75, "up": 0.0, "left": 0.0, "right": 0.0, "stay": 0.25}),
        (0.0, -0.5, {"down": 0.5, "up": 0.0, "left": 0.0, "right": 0.0, "stay": 0.5}),
        (0.5, 0.0, {"down": 0.0, "up": 0.0, "left": 0.0, "right": 0.5, "stay": 0.5}),
        (-0.5, 0.0, {"down": 0.0, "up": 0.0, "left": 0.5, "right": 0.0, "stay": 0.5}),
        (0.4, -0.3, {"down": 0.24, "up": 0.0, "left": 0.0, "right": 0.34, "stay": 0.42}),
    ],
)
def test_one_step_streaming_probabilities(vx, vy, expected):
    p = _make_params(nx=9, ny=9, nm=10)
    start = (4, 7, 0)
    trials = 3000

    counts = {"down": 0, "up": 0, "left": 0, "right": 0, "stay": 0, "other": 0}

    for _ in range(trials):
        u, v, s = _single_particle_state(start=start, vx=vx, vy=vy, nx=p.nx, ny=p.ny, nm=p.nm)
        u, v, s = d2q4_cpp.stream(u, v, s, p)
        i, j, _ = _particle_pos(s)

        if (i, j) == (start[0], start[1] - 1):
            counts["down"] += 1
        elif (i, j) == (start[0], start[1] + 1):
            counts["up"] += 1
        elif (i, j) == (start[0] - 1, start[1]):
            counts["left"] += 1
        elif (i, j) == (start[0] + 1, start[1]):
            counts["right"] += 1
        elif (i, j) == (start[0], start[1]):
            counts["stay"] += 1
        else:
            counts["other"] += 1

    assert counts["other"] == 0
    observed = {k: counts[k] / trials for k in ["down", "up", "left", "right", "stay"]}
    for key, expected_value in expected.items():
        assert observed[key] == pytest.approx(expected_value, abs=0.035)


def test_destination_density_is_capped_by_nu_cs():
    p = _make_params(nx=5, ny=5, nm=10)
    p.nu_cs = 0.5

    s = np.full((p.nx, p.ny, p.nm), np.nan, dtype=np.float64)
    u = np.zeros((p.nx, p.ny, p.nm), dtype=np.float64)
    v = np.zeros((p.nx, p.ny, p.nm), dtype=np.float64)

    src = (2, 2, 0)
    dst = (3, 2)

    # Source particle wants to stream right with probability 1.
    s[src] = 1.0
    u[src] = 1.0

    # Destination is already at nu_cs: 5/10 occupied.
    for k in range(1, 6):
        s[dst[0], dst[1], k] = 1.0

    _, _, s_out = d2q4_cpp.stream(u, v, s, p)

    # Destination must not be overfilled.
    nu_dest = np.mean(~np.isnan(s_out[dst[0], dst[1], :]))
    assert nu_dest <= p.nu_cs + 1e-12

    # The moving particle should remain at source because right-move is blocked by nu_cs cap.
    assert not np.isnan(s_out[src])
    assert np.isnan(s_out[dst[0], dst[1], src[2]])


def test_void_cells_have_zero_velocity_after_streaming():
    p = _make_params(nx=9, ny=9, nm=10)
    rng = np.random.default_rng(123)

    s = np.full((p.nx, p.ny, p.nm), np.nan, dtype=np.float64)
    u = rng.normal(size=(p.nx, p.ny, p.nm))
    v = rng.normal(size=(p.nx, p.ny, p.nm))

    s[4, 7, 0] = 1.0
    u[4, 7, 0] = 0.4
    v[4, 7, 0] = -0.3

    u_out, v_out, s_out = d2q4_cpp.stream(u, v, s, p)
    void_mask = np.isnan(s_out)

    assert np.all(u_out[void_mask] == 0.0)
    assert np.all(v_out[void_mask] == 0.0)


def test_tau_does_not_relax_dilute_cells():
    p = _make_params(nx=7, ny=7, nm=10)
    p.tau = 2.0

    s = np.full((p.nx, p.ny, p.nm), np.nan, dtype=np.float64)
    u = np.zeros((p.nx, p.ny, p.nm), dtype=np.float64)
    v = np.zeros((p.nx, p.ny, p.nm), dtype=np.float64)

    src = (3, 3, 0)
    s[src] = 1.0
    u[src] = 1.0
    v[src] = -0.5

    # Block all neighboring moves so we isolate pure relaxation.
    p.boundary_mask[2, 3] = True
    p.boundary_mask[4, 3] = True
    p.boundary_mask[3, 2] = True
    p.boundary_mask[3, 4] = True

    u_out, v_out, s_out = d2q4_cpp.stream(u, v, s, p)
    i, j, k = _particle_pos(s_out)
    assert (i, j, k) == src

    # nu = 1/nm < nu_cs, so no relaxation should be applied.
    assert u_out[i, j, k] == pytest.approx(1.0, abs=1e-10)
    assert v_out[i, j, k] == pytest.approx(-0.5, abs=1e-10)


def test_tau_relaxes_velocity_when_nu_exceeds_nu_cs():
    p = _make_params(nx=7, ny=7, nm=10)
    p.tau = 2.0
    p.nu_cs = 0.5

    s = np.full((p.nx, p.ny, p.nm), np.nan, dtype=np.float64)
    u = np.zeros((p.nx, p.ny, p.nm), dtype=np.float64)
    v = np.zeros((p.nx, p.ny, p.nm), dtype=np.float64)

    i0, j0 = 3, 3
    # Dense local packing: 6/10 = 0.6 > nu_cs.
    for k in range(6):
        s[i0, j0, k] = 1.0
        u[i0, j0, k] = 1.0
        v[i0, j0, k] = -0.5

    # Block all neighboring moves so we isolate pure relaxation.
    p.boundary_mask[2, 3] = True
    p.boundary_mask[4, 3] = True
    p.boundary_mask[3, 2] = True
    p.boundary_mask[3, 4] = True

    u_out, v_out, s_out = d2q4_cpp.stream(u, v, s, p)
    relax_factor = np.exp(-p.dt / p.tau)

    for k in range(6):
        assert not np.isnan(s_out[i0, j0, k])
        assert u_out[i0, j0, k] == pytest.approx(1.0 * relax_factor, abs=1e-10)
        assert v_out[i0, j0, k] == pytest.approx(-0.5 * relax_factor, abs=1e-10)


def test_tau_does_not_relax_when_particle_fits_local_pore_size():
    p = _make_params(nx=7, ny=7, nm=10)
    p.tau = 2.0
    p.space_criterion = "pore_size"

    s = np.full((p.nx, p.ny, p.nm), np.nan, dtype=np.float64)
    u = np.zeros((p.nx, p.ny, p.nm), dtype=np.float64)
    v = np.zeros((p.nx, p.ny, p.nm), dtype=np.float64)

    src = (3, 3, 0)
    s[src] = 0.5
    u[src] = 1.0
    v[src] = -0.5

    # Block all neighboring moves so we isolate pure relaxation.
    p.boundary_mask[2, 3] = True
    p.boundary_mask[4, 3] = True
    p.boundary_mask[3, 2] = True
    p.boundary_mask[3, 4] = True

    u_out, v_out, s_out = d2q4_cpp.stream(u, v, s, p)
    i, j, k = _particle_pos(s_out)
    assert (i, j, k) == src

    # In dilute conditions, local d_pore is large and no relaxation is applied.
    assert u_out[i, j, k] == pytest.approx(1.0, abs=1e-10)
    assert v_out[i, j, k] == pytest.approx(-0.5, abs=1e-10)


def test_tau_relaxes_when_particle_exceeds_local_pore_size():
    p = _make_params(nx=7, ny=7, nm=10)
    p.tau = 2.0
    p.space_criterion = "pore_size"

    s = np.full((p.nx, p.ny, p.nm), np.nan, dtype=np.float64)
    u = np.zeros((p.nx, p.ny, p.nm), dtype=np.float64)
    v = np.zeros((p.nx, p.ny, p.nm), dtype=np.float64)

    i0, j0 = 3, 3
    # Dense local packing: d_pore becomes small enough that s > d_pore.
    for k in range(6):
        s[i0, j0, k] = 1.0
        u[i0, j0, k] = 1.0
        v[i0, j0, k] = -0.5

    # Block all neighboring moves so we isolate pure relaxation.
    p.boundary_mask[2, 3] = True
    p.boundary_mask[4, 3] = True
    p.boundary_mask[3, 2] = True
    p.boundary_mask[3, 4] = True

    u_out, v_out, s_out = d2q4_cpp.stream(u, v, s, p)
    relax_factor = np.exp(-p.dt / p.tau)

    for k in range(6):
        assert not np.isnan(s_out[i0, j0, k])
        assert u_out[i0, j0, k] == pytest.approx(1.0 * relax_factor, abs=1e-10)
        assert v_out[i0, j0, k] == pytest.approx(-0.5 * relax_factor, abs=1e-10)
