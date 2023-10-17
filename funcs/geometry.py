import numpy as np


def create_spherical_grid(num_pts):
    """Method see:
    https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    answered by CR Drost

    Coversion to cartesian coordinates:
    x = np.cos(PHI) * np.sin(THETA)
    y = np.sin(PHI) * np.sin(THETA)
    z = np.cos(THETA);

    Parameters:
    -----------
    num_pts : int
        number of grid points on the full sphere

    Return:
    --------
    THETA, PHI - numpy arrays of latitude, longitude
    """

    # This is CR Drost's solution to the sunflower spiral:
    indices = np.arange(0, num_pts, dtype=float) + 0.5
    THETA = np.arccos(1 - 2 * indices/num_pts) #latitude
    PHI = np.pi * (1 + 5**0.5) * indices #longitude

    return THETA, PHI


def rotate_around_arb_axis(a, pos, axis):
      """Rotate pos around axis with angle.
      
      Parameters:
      -----------
      a : float
            angle in radians
      pos : array of the form: [x, y, z] or (3, N)
            position(s) to be rotated
      axis : array of the form: [x, y, z]
            axis around which pos is rotated

      Return:
      --------
      pos_rot : array of the form: [x, y, z] or (3, N)
            rotated position(s)
      """
      # some shortcts
      cosa = 1 - np.cos(a)
      ca = np.cos(a)
      sa = np.sin(a)
      ux, uy, uz = axis

      Rrotstar = np.array([[ca + ux**2 * cosa, ux * uy * cosa - uz * sa, ux * uz * cosa + uy * sa],
                              [uy * ux * cosa + uz * sa, ca + uy**2 * cosa, uy * uz * cosa - ux * sa],
                              [uz * ux * cosa - uy * sa, uz * uy * cosa + ux * sa, ca + uz**2 * cosa]])

      # rotate blue points
      x, y, z = pos
      xr, yr, zr = np.dot(Rrotstar, np.array([x, y, z]))

      return np.array([xr, yr, zr])