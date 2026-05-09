"""HEALPix grid helpers."""

from functools import partial

import jax
import jax.numpy as jnp


def angular_distance(theta1, phi1, theta2, phi2):
    """Return great-circle angular distances on a sphere."""
    cos_gamma = (
        jnp.cos(theta1) * jnp.cos(theta2)
        + jnp.sin(theta1) * jnp.sin(theta2) * jnp.cos(phi1 - phi2)
    )
    return jnp.arccos(jnp.clip(cos_gamma, -1.0, 1.0))


def healpix_pixel_angles(nside, order="ring"):
    """Return HEALPix pixel angles using healjax."""
    import healjax as hp

    npix = hp.nside2npix(nside)
    pix2ang = jax.vmap(partial(hp.pix2ang, order, nside))
    return pix2ang(jnp.arange(npix))


def angular_distance_matrix(theta, phi):
    """Return the pairwise great-circle distance matrix for pixels."""
    return angular_distance(theta[:, None], phi[:, None], theta[None, :], phi[None, :])
