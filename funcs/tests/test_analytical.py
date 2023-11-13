"""
UTF-8
Python 3.10

Ekaterina Ilin 2023 -- MIT Licencse

Tests for the analytical model functions.
"""

from ..analytical import x_params, vx_params, x_phi, v_phi, flux_at_x_vx, get_analytical_spectral_line

import numpy as np
import pytest

def test_get_analytical_spectral_line():
    """Test the get_analytical_spectral_line function.
    """
    # define the parameters
    i_rot = 0
    i_mag = 0
    latitude = 0
    alpha = 0
    bins = np.linspace(-20, 20, 100)
    omega = 2 * np.pi / 3
    Rstar = 1

    # get the phase angles
    phi = np.linspace(0, 2*np.pi, 1000)

    # get the flux
    flux = get_analytical_spectral_line(phi, i_rot, i_mag, latitude, alpha, bins, omega, Rstar, norm=.1)

    # assert that the flux is a numpy array
    assert isinstance(flux, np.ndarray)

    # assert that the flux is between 0 and 1
    assert np.all(flux >= 0)
    assert np.all(flux <= 1)

    # assert that the flux is normalized
    assert np.isclose(np.max(flux), 1)

    # assert that the flux is not all zeros
    assert not np.allclose(flux, np.zeros_like(flux))

    # assert that the flux is finite
    assert np.all(np.isfinite(flux))


    # maximum flux should be at the edges where the velocity is also highest
    binmids = (bins[1:] + bins[:-1]) / 2
    maxv = 2 * np.pi * Rstar * 695700 / (3 * 86400)
    assert binmids[np.argmax(flux)] / maxv == pytest.approx(1, rel=1e-2)


def test_x_params():
    """Test the x_params function.
    """
    # define the parameters
    alpha = np.pi/2
    i_rot = np.pi/2
    i_mag = np.pi/2
    latitude = np.pi/2

    # get the parameters
    A, B, C = x_params(alpha, i_rot, i_mag, latitude)

    # define the expected values
    A_exp = 0
    B_exp = -1
    C_exp = 0

    # assert that the values are correct
    assert np.isclose(A, A_exp)
    assert np.isclose(B, B_exp)
    assert np.isclose(C, C_exp)

    # set all values to zero
    alpha = 0
    i_rot = 0
    i_mag = 0
    latitude = 0

    # get the parameters
    A, B, C = x_params(alpha, i_rot, i_mag, latitude)

    # define the expected values 
    A_exp = 1
    B_exp = 1
    C_exp = 0


# write a unit test function for vx_params
def test_vx_params():
    """Test the vx_params function.
    """
    # define the parameters
    alpha = np.pi/2
    i_rot = np.pi/2
    i_mag = np.pi/2
    latitude = np.pi/2

    # get the parameters
    X, Y, Z = vx_params(alpha, i_rot, i_mag, latitude)

    # define the expected values
    X_exp = 0
    Y_exp = 0
    Z_exp = 0

    # assert that the values are correct
    assert np.isclose(X, X_exp)
    assert np.isclose(Y, Y_exp)
    assert np.isclose(Z, Z_exp)

    # set all values to zero
    alpha = 0
    i_rot = 0
    i_mag = 0
    latitude = 0

    # get the parameters
    X, Y, Z = vx_params(alpha, i_rot, i_mag, latitude)

    # define the expected values 
    X_exp = 0
    Y_exp = 1
    Z_exp = 0


def test_x_phi():
    """Test the x_phi function."""
    phi = np.linspace(0, 2*np.pi, 1000)
    A = 1
    B = 0
    C = 0
    x = x_phi(phi, A, B, C)
    assert np.allclose(x, 1)

    A = 0
    B = 1
    C = 0
    x = x_phi(phi, A, B, C)
    assert np.allclose(x, np.sin(phi))

    A = 0
    B = 0
    C = 1
    x = x_phi(phi, A, B, C)
    assert np.allclose(x, np.cos(phi))

def test_v_phi():
    """Test the v_phi function."""
    phi = np.linspace(0, 2*np.pi, 1000)
    X = 1
    Y = 0
    Z = 0
    v = v_phi(phi, X, Y, Z)
    assert np.allclose(v, 1)

    X = 0
    Y = 1
    Z = 0
    v = v_phi(phi, X, Y, Z)
    assert np.allclose(v, np.sin(phi))

    X = 0
    Y = 0
    Z = 1
    v = v_phi(phi, X, Y, Z)
    assert np.allclose(v, np.cos(phi))



def test_flux_at_x_vx():
    """Test the flux_at_x_vx function.
    """
    vx = np.linspace(-10, 10, 100)
    x = np.linspace(0, 1, 100)
    X = 0
    Y = 0
    Z = 0
    
    flux = flux_at_x_vx(vx, x, X, Y, Z)
    
    # flux is inf at 10 and -10
    assert ~np.isfinite(flux[-1])
    assert ~np.isfinite(flux[0])

    # if you feed nan, you get nan back
    flux = flux_at_x_vx(np.nan, np.nan, X, Y, Z)

    assert np.isnan(flux)

    # if you feed x>1 you get nan back
    flux = flux_at_x_vx(20, 1.1, X, Y, Z) 

    assert np.isnan(flux)
