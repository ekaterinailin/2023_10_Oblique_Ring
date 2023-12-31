from .geometry import rotate_around_arb_axis, calculate_surface_element_velocities
import numpy as np

def numerical_spectral_line(alpha, x, y, z, z_rot, omega, Rstar, bins, dalpha=.1 * np.pi/180, normalize=True):
    """Calculate the broadened spectral line of the ring defined
    by x, y, z.

    Parameters
    ----------
    alpha : float
        The rotational phase of the star in rad.
    x : array
        The x positions of the ring in stellar radii.
    y : array
        The y positions of the ring in stellar radii.
    z : array
        The z positions of the ring in stellar radii.
    z_rot : float
        The rotational axis of the star in rad.
    omega : float
        The rotation rate of the star in rad / day.
    Rstar : float
        The radius of the star in solar radii.
    bins : array
        The velocity bins to use for the spectral line.
    dalpha : float
        The step size in alpha to use for velocity calculation.

    Returns
    -------
    flux : array
        The flux of the spectral line.
    """
    # rotate the ring
    xr, _, _ = rotate_around_arb_axis(alpha, np.array([x, y, z]), z_rot)

    # calculate the surface element velocities
    dxr_visible = calculate_surface_element_velocities(alpha, dalpha, x, y, z, z_rot, omega, Rstar)

    # define the visible part of the ring
    q = xr > 0

    # bin the flux
    flux, _ = np.histogram(dxr_visible, bins=bins, weights=np.cos(np.pi/2 - np.arcsin(xr[q])))

    # normalize the flux
    if normalize:
        if max(flux) != 0:
            flux = flux / np.max(flux)

    return flux