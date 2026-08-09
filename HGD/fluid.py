"""Two-way fluid coupling utilities for heterarchical granular dynamics.

The implementation follows the fluid-fraction-weighted formulation used by
Heterarchical Granular--Fluid Dynamics (HGFD).  It is intentionally a compact
2-D, isothermal, laminar solver: particle velocities are relaxed with the
Gidaspow drag law and the equal-and-opposite drag is applied to an
incompressible fluid using a variable-porosity pressure projection.
"""

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


@dataclass
class FluidState:
    """Eulerian fluid fields and coupling diagnostics on the HGD grid."""

    u: np.ndarray
    v: np.ndarray
    pressure: np.ndarray
    beta: np.ndarray
    solid_u: np.ndarray
    solid_v: np.ndarray
    momentum_source_u: np.ndarray
    momentum_source_v: np.ndarray
    divergence: np.ndarray
    fluidized: np.ndarray
    pressure_drop: float = 0.0
    fluidization_ratio: float = 0.0


def _void_fraction(solid_fraction, p):
    minimum = getattr(p, "fluid_min_void_fraction", 0.05)
    return np.clip(1.0 - solid_fraction, minimum, 1.0)


def initialize(p, s):
    """Initialise a co-located gas field from a prescribed superficial inlet velocity."""

    solid_fraction = 1.0 - np.mean(np.isnan(s), axis=2)
    void_fraction = _void_fraction(solid_fraction, p)
    superficial_velocity = getattr(p, "fluid_inlet_velocity", 0.0)
    shape = solid_fraction.shape

    # n * u_f is the superficial velocity.  This initial field therefore
    # satisfies the one-dimensional variable-porosity continuity equation.
    u = np.zeros(shape)
    v = superficial_velocity / void_fraction
    pressure = np.zeros(shape)
    zeros = np.zeros(shape)
    return FluidState(
        u=u,
        v=v,
        pressure=pressure,
        beta=zeros.copy(),
        solid_u=zeros.copy(),
        solid_v=zeros.copy(),
        momentum_source_u=zeros.copy(),
        momentum_source_v=zeros.copy(),
        divergence=zeros.copy(),
        fluidized=np.zeros(shape, dtype=bool),
    )


def drag_coefficients(s, particle_u, particle_v, fluid_u, fluid_v, p):
    """Return layer and cell Gidaspow momentum-exchange coefficients.

    ``layer_beta`` is the coefficient used for particle relaxation.  The
    cell-level coefficient contains the heterarchical ``1 / M`` weighting and
    is the quantity supplied to the fluid momentum equation.
    """

    occupied = np.isfinite(s)
    solid_fraction = np.mean(occupied, axis=2)
    void_fraction = _void_fraction(solid_fraction, p)
    phi = solid_fraction[:, :, np.newaxis]
    eps = void_fraction[:, :, np.newaxis]

    fluid_u_3d = fluid_u[:, :, np.newaxis]
    fluid_v_3d = fluid_v[:, :, np.newaxis]
    rel_u = fluid_u_3d - particle_u
    rel_v = fluid_v_3d - particle_v
    relative_speed = np.sqrt(rel_u**2 + rel_v**2)

    diameter_floor = getattr(p, "fluid_drag_min_particle_size", 1e-12)
    diameter = np.where(occupied, np.maximum(s, diameter_floor), 1.0)
    rho_f = p.gas_density
    mu_f = p.gas_viscosity
    reynolds = rho_f * relative_speed * diameter / mu_f

    # Written as Cd*|ur| to remain finite in the Stokes limit Re -> 0.
    beta_wen_yu = 18.0 * mu_f / diameter**2 * (1.0 + 0.15 * reynolds**0.687) * phi * eps ** (-2.65)
    beta_ergun = 150.0 * mu_f * phi**2 / (eps * diameter**2) + 1.75 * rho_f * phi * relative_speed / diameter
    layer_beta = np.where(phi < 0.2, beta_wen_yu, beta_ergun)
    layer_beta = np.where(occupied, layer_beta, 0.0)
    cell_beta = np.sum(layer_beta, axis=2) / p.nm
    return layer_beta, cell_beta


def update_particle_velocities(particle_u, particle_v, s, state, p):
    """Advance layer-wise particle velocity using the analytic drag relaxation."""

    layer_beta, _ = drag_coefficients(s, particle_u, particle_v, state.u, state.v, p)
    occupied = np.isfinite(s)
    relaxation_rate = layer_beta / p.solid_density
    decay = np.exp(-relaxation_rate * p.dt)

    buoyancy = 1.0 - p.gas_density / p.solid_density
    gravity_x = getattr(p, "fluid_gravity_x", 0.0) * buoyancy
    gravity_y = -abs(p.g) * buoyancy
    fluid_u = state.u[:, :, np.newaxis]
    fluid_v = state.v[:, :, np.newaxis]

    with np.errstate(divide="ignore", invalid="ignore"):
        terminal_u = fluid_u + np.divide(
            gravity_x,
            relaxation_rate,
            out=np.zeros_like(relaxation_rate),
            where=relaxation_rate > 0,
        )
        terminal_v = fluid_v + np.divide(
            gravity_y,
            relaxation_rate,
            out=np.zeros_like(relaxation_rate),
            where=relaxation_rate > 0,
        )

    new_u = terminal_u + (particle_u - terminal_u) * decay
    new_v = terminal_v + (particle_v - terminal_v) * decay

    # In cells where the drag coefficient is exactly zero, integrate gravity
    # directly instead of using the singular terminal-velocity expression.
    no_drag = relaxation_rate == 0
    new_u = np.where(no_drag, particle_u + gravity_x * p.dt, new_u)
    new_v = np.where(no_drag, particle_v + gravity_y * p.dt, new_v)
    return np.where(occupied, new_u, 0.0), np.where(occupied, new_v, 0.0)


def coupling_fields(s, particle_u, particle_v, state, p):
    """Aggregate heterarchical drag fields for the fluid momentum equation."""

    layer_beta, cell_beta = drag_coefficients(s, particle_u, particle_v, state.u, state.v, p)
    beta_sum = np.sum(layer_beta, axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        solid_u = np.divide(
            np.sum(layer_beta * particle_u, axis=2),
            beta_sum,
            out=np.zeros_like(cell_beta),
            where=beta_sum > 0,
        )
        solid_v = np.divide(
            np.sum(layer_beta * particle_v, axis=2),
            beta_sum,
            out=np.zeros_like(cell_beta),
            where=beta_sum > 0,
        )

    # Equal and opposite to beta * (u_f - u_p), the drag on particles.
    source_u = cell_beta * (solid_u - state.u)
    source_v = cell_beta * (solid_v - state.v)
    return cell_beta, solid_u, solid_v, source_u, source_v


def _laplacian(field, dx, dy):
    return np.gradient(np.gradient(field, dx, axis=0), dx, axis=0) + np.gradient(
        np.gradient(field, dy, axis=1), dy, axis=1
    )


def _upwind_derivative(field, velocity, spacing, axis):
    """First-order upwind derivative without periodic wraparound."""

    backward = (field - np.roll(field, 1, axis=axis)) / spacing
    forward = (np.roll(field, -1, axis=axis) - field) / spacing
    lower = [slice(None)] * field.ndim
    upper = [slice(None)] * field.ndim
    lower[axis] = 0
    upper[axis] = -1
    backward[tuple(lower)] = forward[tuple(lower)]
    forward[tuple(upper)] = backward[tuple(upper)]
    return np.where(velocity >= 0, backward, forward)


def _face_fluxes(u, v, void_fraction, inlet_velocity=0.0):
    """Interpolate superficial velocity to finite-volume cell faces."""

    nx, ny = void_fraction.shape
    qx = np.zeros((nx + 1, ny))
    qy = np.zeros((nx, ny + 1))
    qx[1:nx, :] = 0.5 * (void_fraction[:-1, :] * u[:-1, :] + void_fraction[1:, :] * u[1:, :])
    qy[:, 1:ny] = 0.5 * (void_fraction[:, :-1] * v[:, :-1] + void_fraction[:, 1:] * v[:, 1:])
    qy[:, 0] = inlet_velocity
    qy[:, -1] = void_fraction[:, -1] * v[:, -1]
    return qx, qy


def _flux_divergence(qx, qy, dx, dy):
    return (qx[1:, :] - qx[:-1, :]) / dx + (qy[:, 1:] - qy[:, :-1]) / dy


def weighted_divergence(u, v, void_fraction, dx, dy, inlet_velocity=0.0):
    """Calculate finite-volume ``div(n u_f)`` at cell centres."""

    qx, qy = _face_fluxes(u, v, void_fraction, inlet_velocity)
    return _flux_divergence(qx, qy, dx, dy)


def _pressure_matrix(void_fraction, dx, dy):
    """Build ``-div(n grad(.))`` with wall Neumann and open-top Dirichlet BCs."""

    nx, ny = void_fraction.shape
    matrix = sparse.lil_matrix((nx * ny, nx * ny))

    def flat(i, j):
        return i * ny + j

    for i in range(nx):
        for j in range(ny):
            row = flat(i, j)
            diagonal = 0.0
            for di, dj, spacing in ((-1, 0, dx), (1, 0, dx), (0, -1, dy), (0, 1, dy)):
                ii, jj = i + di, j + dj
                if ii < 0 or ii >= nx or jj < 0:
                    continue
                if jj >= ny:
                    continue
                face_void = 0.5 * (void_fraction[i, j] + void_fraction[ii, jj])
                coefficient = face_void / spacing**2
                diagonal += coefficient
                matrix[row, flat(ii, jj)] = -coefficient
            if j == ny - 1:
                # Fixed pressure at the open top face, half a cell away.
                diagonal += 2.0 * void_fraction[i, j] / dy**2
            matrix[row, row] = diagonal
    return matrix.tocsr()


def project_velocity(u, v, pressure, void_fraction, p):
    """Project a tentative velocity onto ``div(n u_f) = 0``."""

    del pressure
    inlet = getattr(p, "fluid_inlet_velocity", 0.0)
    qx, qy = _face_fluxes(u, v, void_fraction, inlet)
    divergence = _flux_divergence(qx, qy, p.dx, p.dy)
    rhs = -(p.gas_density / p.dt) * divergence.ravel()
    correction = spsolve(_pressure_matrix(void_fraction, p.dx, p.dy), rhs).reshape(void_fraction.shape)

    nx, ny = void_fraction.shape
    void_x = 0.5 * (void_fraction[:-1, :] + void_fraction[1:, :])
    void_y = 0.5 * (void_fraction[:, :-1] + void_fraction[:, 1:])
    qx[1:nx, :] -= p.dt / p.gas_density * void_x * (correction[1:, :] - correction[:-1, :]) / p.dx
    qy[:, 1:ny] -= p.dt / p.gas_density * void_y * (correction[:, 1:] - correction[:, :-1]) / p.dy
    qy[:, -1] -= p.dt / p.gas_density * void_fraction[:, -1] * (0.0 - correction[:, -1]) / (0.5 * p.dy)

    # Reconstruct intrinsic cell-centred velocities for HGD coupling.
    u = 0.5 * (qx[:-1, :] + qx[1:, :]) / void_fraction
    v = 0.5 * (qy[:, :-1] + qy[:, 1:]) / void_fraction
    # The tentative step does not include an old pressure gradient, so this is
    # the new pressure rather than an incremental pressure correction.
    pressure = correction
    return u, v, pressure, _flux_divergence(qx, qy, p.dx, p.dy)


def minimum_fluidization_velocity(p, solid_fraction=None, particle_size=None):
    """Estimate ``U_mf`` from the same dense Gidaspow/Ergun force balance."""

    phi = p.nu_fill if solid_fraction is None else solid_fraction
    diameter = p.s_m if particle_size is None else particle_size
    eps = max(1.0 - phi, getattr(p, "fluid_min_void_fraction", 0.05))
    weight = phi * (p.solid_density - p.gas_density) * p.g
    linear = 150.0 * p.gas_viscosity * phi**2 / (eps**2 * diameter**2)
    quadratic = 1.75 * p.gas_density * phi / (eps**2 * diameter)
    if quadratic == 0:
        return weight / linear
    return (-linear + np.sqrt(linear**2 + 4.0 * quadratic * weight)) / (2.0 * quadratic)


def _diagnostics(state, solid_fraction, p):
    drag_on_solid_y = state.beta * (state.v - state.solid_v)
    effective_weight = solid_fraction * (p.solid_density - p.gas_density) * p.g
    occupied = solid_fraction > 0
    state.fluidized = occupied & (drag_on_solid_y >= effective_weight)
    state.pressure_drop = float(np.mean(state.pressure[:, 0]) - np.mean(state.pressure[:, -1]))
    bed_weight = float(np.mean(np.sum(effective_weight, axis=1) * p.dy))
    state.fluidization_ratio = state.pressure_drop / bed_weight if bed_weight > 0 else 0.0


def advance(state, s, particle_u, particle_v, p):
    """Advance the fluid one staggered explicit HGD--CFD coupling step."""

    solid_fraction = 1.0 - np.mean(np.isnan(s), axis=2)
    void_fraction = _void_fraction(solid_fraction, p)
    beta, solid_u, solid_v, source_u, source_v = coupling_fields(s, particle_u, particle_v, state, p)

    adv_u = state.u * _upwind_derivative(state.u, state.u, p.dx, axis=0) + state.v * _upwind_derivative(
        state.u, state.v, p.dy, axis=1
    )
    adv_v = state.u * _upwind_derivative(state.v, state.u, p.dx, axis=0) + state.v * _upwind_derivative(
        state.v, state.v, p.dy, axis=1
    )
    viscosity = p.gas_viscosity / p.gas_density
    transport_u = state.u + p.dt * (-adv_u + viscosity * _laplacian(state.u, p.dx, p.dy))
    transport_v = state.v + p.dt * (-adv_v + viscosity * _laplacian(state.v, p.dx, p.dy))

    # Treat the stiff drag term analytically.  This is the fluid analogue of
    # the exponential particle relaxation and remains stable in a dense bed.
    drag_rate = beta / (p.gas_density * void_fraction)
    drag_decay = np.exp(-drag_rate * p.dt)
    tentative_u = solid_u + (transport_u - solid_u) * drag_decay
    tentative_v = solid_v + (transport_v - solid_v) * drag_decay

    state.u, state.v, state.pressure, state.divergence = project_velocity(
        tentative_u, tentative_v, state.pressure, void_fraction, p
    )
    state.beta = beta
    state.solid_u = solid_u
    state.solid_v = solid_v
    state.momentum_source_u = source_u
    state.momentum_source_v = source_v
    _diagnostics(state, solid_fraction, p)
    courant = np.max(np.abs(state.u) * p.dt / p.dx + np.abs(state.v) * p.dt / p.dy)
    p.fluid_courant = float(courant)
    if not np.all(np.isfinite(state.u)) or not np.all(np.isfinite(state.v)):
        raise FloatingPointError("Non-finite fluid velocity produced by the HGFD solver")
    if courant > getattr(p, "fluid_cfl_limit", 0.8):
        raise FloatingPointError(
            f"Fluid CFL={courant:.3g} exceeds fluid_cfl_limit; reduce defined_time_step_size"
        )
    return state
