"""
UTF-8 
Python 3.10

Ekaterina Ilin 2023 -- MIT Licencse

Functions for the analytical model.
"""

import numpy as np

def x_params(alpha, i_rot, i_mag, latitude):
    """Calculate the parameters for the x-position of the ring.

    Parameters
    ----------
    alpha : float
        The rotational phase of the star in rad.
    i_rot : float
        The rotation inclination in rad.
    i_mag : float
        The magnetic obliquity in rad.
    latitude : float
        The pi/2 - latitude of the ring in rad.

    Returns
    -------
    A : float
        The A parameter of the ring position.
    B : float
        The B parameter of the ring position.
    C : float
        The C parameter of the ring position.
    """

    # define shorthands for various trig functions
    sa, ca = np.sin(alpha), np.cos(alpha)
    si, ci = np.sin(i_rot), np.cos(i_rot)
    st, ct = np.sin(latitude), np.cos(latitude)
    sip, cip = np.sin(i_rot + i_mag), np.cos(i_rot + i_mag)
    Ca = 1 - np.cos(alpha)

    # calculate the parameters
    A = ct * sip * (ca + si**2 * Ca) + ct * cip *ci * si * Ca
    B = cip * st  * (ca + si**2 * Ca) - sip * st * ci * si * Ca
    C = - sa * ci * st

    return A, B, C

def vx_params(alpha, i_rot, i_mag, latitude):
    """Calculate the parameters for the x-velocity of the ring.

    Parameters
    ----------
    alpha : float
        The rotational phase of the star in rad.
    i_rot : float
        The rotation inclination in rad.
    i_mag : float
        The magnetic obliquity in rad.
    latitude : float
        The pi/2 - latitude of the ring in rad.

    Returns
    -------
    X : float
        The X parameter of the ring velocity.
    Y : float
        The Y parameter of the ring velocity.
    Z : float
        The Z parameter of the ring velocity.
    """
    
    # define shorthands for various trig functions
    sa, ca = np.sin(alpha), np.cos(alpha)
    si, ci = np.sin(i_rot), np.cos(i_rot)
    st, ct = np.sin(latitude), np.cos(latitude)
    sip, cip = np.sin(i_rot + i_mag), np.cos(i_rot + i_mag)

    # calculate the parameters
    X = -sa * (sip * ct - ct * sip * si**2 - ct * cip * si * ci)
    Y = -sa * cip *st + sa * cip * st * si**2 - sa * sip * st * ci *si
    Z = -ca * ci * st

    return X, Y, Z
