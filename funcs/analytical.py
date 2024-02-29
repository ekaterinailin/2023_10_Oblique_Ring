"""
UTF-8 
Python 3.10

Ekaterina Ilin 2023 -- MIT Licencse

Functions for the analytical model.
"""

from numpy import pi, sin, cos, sqrt, abs, arcsin, max, digitize, bincount

import numpy as np

import matplotlib.pyplot as plt

def get_analytical_spectral_line(phi, i_rot, i_mag, latitude, alpha, bins, convert_to_kms):
    """Calculate the broadened spectral line of the infinitesimally narrow
    auroral ring.
    
    Parameters
    ----------
    phi : array
        The phase angles of the ring in rad. From 0 to 2 pi.
    i_rot : float
        The rotation inclination in rad.
    i_mag : float
        The magnetic obliquity in rad.
    latitude : float
        The latitude of the ring in rad.
    alpha : float
        The rotational phase of the star in rad.
    bins : int
        The number of velocity bins to use for the spectral line.
    convert_to_kms : float
        The conversion factor to convert from stellar radii / s to km / s.pyt
        
    Returns
    -------
    flux : array
        The flux of the spectral line.
    """

    # get the parameters for the positions
    A, B, C = x_params(alpha, i_rot=i_rot, i_mag=i_mag, latitude=pi/2 - latitude)

    # get the parameters for the velocities
    X, Y, Z = vx_params(alpha, i_rot=i_rot, i_mag=i_mag, latitude=pi/2 - latitude)

    # print("A, B, C", A, B, C)
    # print("X, Y, Z", X, Y, Z)

    # get the velocities and positions
    v = v_phi(phi, X, Y, Z)
    x = x_phi(phi, A, B, C)

    # get the flux
    flux = flux_at_x_vx(x)
    # print(flux.shape)

    # mask all the negative positions
    q = x > 0

    # convert to km/s
    v = v[q] * convert_to_kms 

    # flux_, bins = np.histogram(v, bins=bins, weights=flux[q])

    digitized = digitize(v, bins)

    flux_ = bincount(digitized, weights=flux[q], minlength=len(bins) - 1)

    # normalize the flux unless it is all zeros
    if max(flux_) != 0:
        flux_ = flux_ / max(flux_)
    
    return flux_


def x_phi(phi, A, B, C):
    """Calculate the x position of the ring at a given phase angle phi.
    
    Parameters
    ----------
    phi : array
        The phase angles of the ring in rad. From 0 to 2 pi.
    A : float
        The A parameter of the ring position.
    B : float
        The B parameter of the ring position.
    C : float       
        The C parameter of the ring position.
    """
    return B * sin(phi) + C * cos(phi) + A


def v_phi(phi, X, Y, Z):
    """Calculate the x-velocity of the ring at a given phase angle phi.
    
    Parameters
    ----------
    phi : array
        The phase angles of the ring in rad. From 0 to 2 pi.
    X : float
        The X parameter of the ring velocity.
    Y : float
        The Y parameter of the ring velocity.
    Z : float
        The Z parameter of the ring velocity.

    Returns
    -------
    v : array
        The x-velocity of the ring at the given phase angles.
    """
    return Y * sin(phi) + Z * cos(phi) + X


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
    sa, ca = sin(alpha), cos(alpha)
    si, ci = sin(i_rot), cos(i_rot)
    st, ct = sin(latitude), cos(latitude)
    sip, cip = sin(i_rot + i_mag), cos(i_rot + i_mag)
    Ca = 1 - cos(alpha)

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
    sa, ca = sin(alpha), cos(alpha)
    si, ci = sin(i_rot), cos(i_rot)
    st, ct = sin(latitude), cos(latitude)
    sip, cip = sin(i_rot + i_mag), cos(i_rot + i_mag)

    # calculate the parameters
    X = -sa * (sip * ct - ct * sip * si**2 - ct * cip * si * ci)
    Y = -sa * cip *st + sa * cip * st * si**2 - sa * sip * st * ci *si
    Z = -ca * ci * st

    return X, Y, Z


def flux_at_x_vx(x):
    """Calculate the flux of the ring at a given x position by the foreshortening factor.
    
    Parameters
    ----------
    x : array
        The x position of the ring.

    Returns
    -------
    flux : array
        The flux of the ring at the given positions.
    """
    return cos(pi/2 - arcsin(x)) 
    



