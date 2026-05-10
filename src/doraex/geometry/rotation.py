"""Rotating-body viewing geometry."""

import jax.numpy as jnp


def incline(theta0, phi0, alpha):
    """Rotate spherical coordinates around the y-axis.

    This is the coordinate transform used in Ureshino et al. Appendix A to
    move surface coordinates from the rotating-body frame into the observer
    frame. The rotation angle is typically ``pi / 2 - inclination``.

    Args:
        theta0: Colatitude in the input frame, in radians.
        phi0: Longitude in the input frame, in radians.
        alpha: Right-handed rotation angle around the y-axis, in radians.

    Returns:
        A tuple ``(theta, phi)`` with the rotated colatitude and longitude in
        radians. Inputs may be scalars or JAX arrays with broadcast-compatible
        shapes.
    """
    x0 = jnp.sin(theta0) * jnp.cos(phi0)
    y0 = jnp.sin(theta0) * jnp.sin(phi0)
    z0 = jnp.cos(theta0)

    x = jnp.cos(alpha) * x0 + jnp.sin(alpha) * z0
    y = y0
    z = -jnp.sin(alpha) * x0 + jnp.cos(alpha) * z0

    theta = jnp.arccos(jnp.clip(z, -1.0, 1.0))
    phi = jnp.arctan2(y, x)
    return theta, phi


def rotate_longitude(phi, phase):
    """Advance co-rotating longitudes by rotational phase.

    Implements the phase convention of Ureshino et al. Eq. (3), where one
    cycle corresponds to ``2 * pi`` radians and phase zero places the prime
    meridian toward the observer before the inclination rotation.

    Args:
        phi: Surface longitude in the co-rotating frame, in radians.
        phase: Rotational phase in cycles.

    Returns:
        Effective longitude at the requested phase, in radians.
    """
    return phi + 2.0 * jnp.pi * phase


def line_of_sight_velocity(vrot, inclination, theta, phi):
    """Compute the line-of-sight velocity for rigid rotation.

    This evaluates the radial velocity in Ureshino et al. Eq. (4), with the
    inclination convention ``i = pi / 2 - alpha``.

    Args:
        vrot: Equatorial rotation velocity, in km/s.
        inclination: Spin inclination angle ``i`` in radians.
        theta: Surface colatitude in the co-rotating frame, in radians.
        phi: Phase-advanced longitude in the co-rotating frame, in radians.

    Returns:
        Line-of-sight velocity in km/s. Positive values redshift the local
        rest-frame profile under :func:`doraex.operators.doppler.doppler_factor`.
    """
    return vrot * jnp.sin(inclination) * jnp.sin(theta) * jnp.sin(phi)


def projected_mu(theta_observer, phi_observer):
    """Compute the projected area factor in the observer frame.

    Args:
        theta_observer: Colatitude after rotating into the observer frame, in
            radians.
        phi_observer: Longitude after rotating into the observer frame, in
            radians.

    Returns:
        The direction cosine ``mu = sin(theta) * cos(phi)`` from Ureshino et
        al. Eq. (9). Visible pixels have positive ``mu``.
    """
    return jnp.sin(theta_observer) * jnp.cos(phi_observer)


def visible_mask(phi_observer):
    """Return a float visibility mask for the observer-facing hemisphere.

    Args:
        phi_observer: Longitude after rotating into the observer frame, in
            radians.

    Returns:
        A float array with value 1 for pixels on the observer-facing hemisphere
        and 0 otherwise. The boundary convention matches the strict longitude
        cut used by the original BayesianDI implementation.
    """
    return ((-jnp.pi / 2.0 < phi_observer) & (phi_observer < jnp.pi / 2.0)).astype(
        float
    )
