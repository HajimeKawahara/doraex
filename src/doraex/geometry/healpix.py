"""HEALPix grid helpers."""

from functools import partial

import jax
import jax.numpy as jnp


def angular_distance(theta1, phi1, theta2, phi2):
    """Compute great-circle angular distances on a unit sphere.

    Args:
        theta1: Colatitude of the first point or points, in radians.
        phi1: Longitude of the first point or points, in radians.
        theta2: Colatitude of the second point or points, in radians.
        phi2: Longitude of the second point or points, in radians.

    Returns:
        Angular distance in radians, with shapes determined by JAX
        broadcasting. The result is clipped through the cosine argument for
        numerical stability.
    """
    cos_gamma = (
        jnp.cos(theta1) * jnp.cos(theta2)
        + jnp.sin(theta1) * jnp.sin(theta2) * jnp.cos(phi1 - phi2)
    )
    return jnp.arccos(jnp.clip(cos_gamma, -1.0, 1.0))


def healpix_pixel_angles(nside, order="ring"):
    """Return HEALPix pixel centers as spherical angles.

    Args:
        nside: HEALPix ``nside`` resolution parameter.
        order: HEALPix indexing order passed to ``healjax.pix2ang``. Common
            values are ``"ring"`` and ``"nested"``.

    Returns:
        A tuple ``(theta, phi)`` for all ``12 * nside**2`` pixels, in radians.
        These angles define the equal-area pixels used by the map vector in
        the Ureshino-style linear inverse problem.
    """
    import healjax as hp

    npix = hp.nside2npix(nside)
    pix2ang = jax.vmap(partial(hp.pix2ang, order, nside))
    return pix2ang(jnp.arange(npix))


def angular_distance_matrix(theta, phi):
    """Build the pairwise angular-distance matrix for map pixels.

    Args:
        theta: One-dimensional array of pixel colatitudes, in radians.
        phi: One-dimensional array of pixel longitudes, in radians.

    Returns:
        A square matrix whose ``(i, j)`` element is the great-circle distance
        between pixels ``i`` and ``j``. This is the ``d_jj'`` input to the
        spherical Gaussian-process covariance in Ureshino et al. Eq. (17).
    """
    return angular_distance(theta[:, None], phi[:, None], theta[None, :], phi[None, :])
