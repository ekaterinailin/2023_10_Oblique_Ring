import numpy as np
from astropy.constants import R_sun


def set_up_oblique_auroral_ring(THETA, PHI, PHI_max, PHI_min,
                                i_rot, i_mag):
     """Set up an oblique auroral ring around the magnetic axis.

     Parameters
     ----------
     THETA : array
           1D array of theta values in radians.
     PHI : array
           1D array of phi values in radians.
     PHI_max : float
           Upper latitude of ring around magnetic axis in radians.
     PHI_min : float
           Lower latitude of ring around magnetic axis in radians.
     i_rot : float
           Inclination of rotation axis in radians with the right convention.
     i_mag : float
           Inclination of magnetic axis in radians relative to rotation axis.

     Returns
     -------
     (x, y, z) : tuple
           Tuple of arrays of x, y, z coordinates of points on the sphere at phase 0.
     z_rot : array
           rotation axis after rotation around y axis with the i_rot angle.
     z_rot_mag : array
           magnetic axis after rotation around y axis with the i_rot + i_mag angle.
     """
     

     # select the points on a sphere that are within the ring
     # around the magnetic axis
     q = ((THETA > (np.pi/2 - PHI_max)) &
          (THETA < (np.pi/2 - PHI_min)))

     # 3D rotation matrix for rotation around y axis with the i_rot + i_mag angle
     Rrotmag = np.array([[np.cos(i_rot + i_mag), 0, np.sin(i_rot + i_mag)],
                    [0, 1, 0],
                    [-np.sin(i_rot + i_mag), 0, np.cos(i_rot + i_mag)]])

     # 3D rotation matrix for rotation around y axis with the i_rot angle
     Rrot = np.array([[np.cos(i_rot), 0, np.sin(i_rot)],
                    [0, 1, 0],
                    [-np.sin(i_rot), 0, np.cos(i_rot)]])


     # rotate the points on the sphere
     x, y, z = np.dot(Rrotmag, np.array([np.sin(THETA[q])*np.cos(PHI[q]),
                                   np.sin(THETA[q])*np.sin(PHI[q]),
                                   np.cos(THETA[q])]))
     
     print(x.shape)

     # rotate the z axis with Rrot
     z_rot = np.array([0, 0, 1])
     z_rot = np.dot(Rrot, z_rot)

     # rotate the z axis by i_rot + i_mag
     z_rot_mag = 1.5 * np.array([0, 0, 1])
     z_rot_mag = np.dot(Rrotmag, z_rot_mag)

     return (x, y, z), z_rot, z_rot_mag

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


def calculate_surface_element_velocities(alpha, dalpha, x, y, z, z_rot, omega, Rstar):
    """At a given phase angle alpha of the rotating star,
    calculate the radial velocity of the surface element from the
    x-component of the derivative
    
    ((x,y,z)(alpha + dalpha) - (x,y,z)(alpha)) / dalpha * omega

    where dalpha is a small angle in radians, x, y, z are the
    coordinates of the surface element in cartesian coordinates, and
    omega is the angular velocity of the star in radians per day.

    Convert the x-component of the derivative to km/s using Rstar and
    return their values for surface elements that are visible to the observer.

    Parameters
    ----------
    alpha : float
        phase angle in radians
    dalpha : float
        small angle in radians
    x : float
        x coordinate of surface element in cartesian coordinates
    y : float
        y coordinate of surface element in cartesian coordinates
    z : float
        z coordinate of surface element in cartesian coordinates
    z_rot : array
        rotation axis of the star in cartesian coordinates
    omega : float
        angular velocity of the star in radians per day
    Rstar : float
        stellar radius in solar radii

    Returns
    -------
    dxr_visible : array
        x-component of the derivative in km/s for surface elements
        that are visible to the observer.
    """

    # rotate the surface element around the rotation axis
    xr1, yr1, zr1 = rotate_around_arb_axis(alpha, np.array([x, y, z]), z_rot)

    # rotate the surface element around the rotation axis with an extra small angle
    xr2, yr2, zr2 = rotate_around_arb_axis(alpha + dalpha, np.array([x, y, z]), z_rot)

    # calculate the derivative of xr
    dxr = (xr2 - xr1) / dalpha * omega 

    # convert to stellar radii / s
    dxr = dxr / (24*3600)
     
    # convert to km/s
    dxr = (dxr * Rstar * R_sun).value / 1e3 

    # then select only the positive values of xr
    q = (xr1 > 0)

    # return visible surface element velocities
    return dxr[q]