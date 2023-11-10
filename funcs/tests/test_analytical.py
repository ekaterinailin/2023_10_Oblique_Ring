"""
UTF-8
Python 3.10

Ekaterina Ilin 2023 -- MIT Licencse

Tests for the analytical model functions.
"""

from ..analytical import x_params, vx_params

import numpy as np

# write a unit test function for x_params
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
