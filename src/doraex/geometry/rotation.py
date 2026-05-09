"""Rotating-body viewing geometry."""

import jax.numpy as jnp


def incline(theta0, phi0, alpha):
    """Rotate spherical coordinates around the y axis by ``alpha`` radians."""
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
    """Advance co-rotating longitudes by a rotational phase in cycles."""
    return phi + 2.0 * jnp.pi * phase


def line_of_sight_velocity(vrot, inclination, theta, phi):
    """Return the line-of-sight velocity for a rigidly rotating surface."""
    return vrot * jnp.sin(inclination) * jnp.sin(theta) * jnp.sin(phi)


def projected_mu(theta_observer, phi_observer):
    """Return the projected area factor in the observer frame."""
    return jnp.sin(theta_observer) * jnp.cos(phi_observer)


def visible_mask(phi_observer):
    """Return a float visibility mask for the observer-facing hemisphere."""
    return ((-jnp.pi / 2.0 < phi_observer) & (phi_observer < jnp.pi / 2.0)).astype(
        float
    )
