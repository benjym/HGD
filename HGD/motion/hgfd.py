"""Inertial, four-direction HGD transport for two-way fluid coupling."""

import numpy as np

from HGD import fluid


def concentration_dependent_alpha(solid_fraction, p):
    """HGFD mixing coefficient that vanishes when dilute and saturates when dense."""

    alpha_max = getattr(p, "fluid_alpha_max", p.alpha)
    epsilon = getattr(p, "fluid_alpha_epsilon", 0.01)
    phi_c = float(np.min(p.nu_cs)) if isinstance(p.nu_cs, np.ndarray) else p.nu_cs
    phi_ref = phi_c - epsilon
    clipped = np.minimum(solid_fraction, phi_ref)
    numerator = np.maximum(phi_c - clipped, epsilon) ** -0.5 - phi_c**-0.5
    denominator = epsilon**-0.5 - phi_c**-0.5
    return np.clip(alpha_max * numerator / denominator, 0.0, alpha_max)


def _destination_masks(s, solid_fraction, boundary, axis, step, p):
    destination = np.roll(s, -step, axis=axis)
    destination_void = np.isnan(destination)
    destination_fraction = np.roll(solid_fraction, -step, axis=axis)[:, :, np.newaxis]
    destination_boundary = np.roll(boundary, -step, axis=axis)[:, :, np.newaxis]
    blocked = (destination_fraction >= p.nu_cs) | destination_boundary

    edge = [slice(None)] * 3
    edge[axis] = -1 if step > 0 else 0
    blocked[tuple(edge)] = True
    valid = destination_void & ~blocked
    # An event exists either when the matching destination layer is void or
    # when a wall/critically packed cell rejects the attempted event.  A
    # matching occupied layer in an under-packed cell simply offers no swap.
    candidate = destination_void | blocked
    return candidate, valid


def _conflict_free(sources, destinations):
    if len(sources) == 0:
        return sources, destinations
    order = np.random.permutation(len(sources))
    sources = sources[order]
    destinations = destinations[order]
    _, first = np.unique(destinations, axis=0, return_index=True)
    keep = np.sort(first)
    return sources[keep], destinations[keep]


def move_voids(u, v, s, p, diag=0, c=None, T=None, chi=None, last_swap=None):
    """Advance particles using force-derived velocities and stochastic swaps."""

    del diag
    if not getattr(p, "fluid_coupling", False):
        raise ValueError("The 'hgfd' motion model requires fluid_coupling=true")
    if last_swap is None:
        last_swap = np.zeros_like(s)
    if not hasattr(p, "hgfd_virtual_displacement") or p.hgfd_virtual_displacement.shape != s.shape:
        p.hgfd_virtual_displacement = np.zeros_like(s)

    u, v = fluid.update_particle_velocities(u, v, s, p.fluid_state, p)
    occupied = np.isfinite(s)
    virtual_displacement = p.hgfd_virtual_displacement
    virtual_displacement[occupied] += np.hypot(u[occupied], v[occupied]) * p.dt
    virtual_displacement[~occupied] = 0.0
    solid_fraction = np.mean(occupied, axis=2)
    boundary = getattr(p, "boundary_mask", np.zeros((p.nx, p.ny), dtype=bool))
    alpha = concentration_dependent_alpha(solid_fraction, p)[:, :, np.newaxis]
    diffusivity = alpha * np.nan_to_num(s) * np.abs(v)
    horizontal_diffusion = diffusivity * p.dt / p.dx**2

    probabilities = []
    valid_destinations = []
    directions = ((0, 1), (0, -1), (1, 1), (1, -1))
    for axis, step in directions:
        velocity = u if axis == 0 else v
        directional = np.maximum(step * velocity, 0.0) * p.dt / (p.dx if axis == 0 else p.dy)
        if axis == 0:
            directional = directional + horizontal_diffusion
        candidate, valid = _destination_masks(s, solid_fraction, boundary, axis, step, p)
        probabilities.append(np.where(occupied & candidate, directional, 0.0))
        valid_destinations.append(valid)

    total = np.sum(probabilities, axis=0)
    max_probability = float(np.max(total)) if total.size else 0.0
    p.max_transition_probability = max_probability
    limit = getattr(p, "P_stab", 0.5)
    if max_probability >= limit:
        policy = getattr(p, "fluid_probability_policy", "error")
        if policy == "scale":
            # Scaling is retained as an explicitly non-paper fallback.  Paper
            # reproduction configurations use ``error`` and select dt so that
            # the strict P_tot < 0.5 condition is satisfied.
            scale = np.minimum(1.0, np.nextafter(limit, 0.0) / np.maximum(total, 1e-30))
            probabilities = [probability * scale for probability in probabilities]
        else:
            raise ValueError(
                f"HGFD transition probability {max_probability:.3g} is not below P_stab={limit}; "
                "reduce defined_time_step_size or set fluid_probability_policy='scale'"
            )

    draw = np.random.random(s.shape)
    cumulative = np.zeros_like(s, dtype=float)
    source_batches = []
    destination_batches = []
    for probability, valid, (axis, step) in zip(probabilities, valid_destinations, directions):
        selected = (draw >= cumulative) & (draw < cumulative + probability)
        rejected = selected & ~valid
        u[rejected] = 0.0
        v[rejected] = 0.0
        sources = np.argwhere(selected & valid)
        if len(sources):
            destinations = sources.copy()
            destinations[:, axis] += step
            source_batches.append(sources)
            destination_batches.append(destinations)
        cumulative += probability

    if source_batches:
        sources = np.concatenate(source_batches)
        destinations = np.concatenate(destination_batches)
        sources, destinations = _conflict_free(sources, destinations)

        # Recreate direction markers after conflict filtering from displacement.
        displacement = destinations - sources
        swap_type = np.where(displacement[:, 1] != 0, 1, -1)
        source_index = tuple(sources.T)
        destination_index = tuple(destinations.T)
        source_displacement = virtual_displacement[source_index].copy()
        for array in (s, u, v, c, T):
            if array is not None:
                array[source_index], array[destination_index] = (
                    array[destination_index].copy(),
                    array[source_index].copy(),
                )

        # Eq. (26): carry any cell-traversal deficit with the solid element
        # and subtract exactly one lattice spacing after a realised swap.
        virtual_displacement[source_index] = 0.0
        virtual_displacement[destination_index] = np.maximum(source_displacement - p.dx, 0.0)

        last_swap[source_index] = np.nan
        last_swap[destination_index] = swap_type
        moved = np.zeros((p.nx, p.ny), dtype=float)
        np.add.at(moved, (sources[:, 0], sources[:, 1]), 1)
        np.add.at(moved, (destinations[:, 0], destinations[:, 1]), 1)
        chi = moved / (p.nm * limit)
    else:
        chi = np.zeros((p.nx, p.ny), dtype=float)

    u[np.isnan(s)] = 0.0
    v[np.isnan(s)] = 0.0
    virtual_displacement[np.isnan(s)] = 0.0
    last_swap[np.isnan(s)] = np.nan
    return u, v, s, c, T, chi, last_swap
