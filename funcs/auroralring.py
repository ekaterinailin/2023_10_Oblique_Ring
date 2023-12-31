
from .analytical import get_analytical_spectral_line
from .numerical import numerical_spectral_line
from .geometry import create_spherical_grid, set_up_oblique_auroral_ring, rotate_around_arb_axis

import numpy as np
import matplotlib.pyplot as plt

class AuroralRing:
    """A class to represent an auroral ring on a star.

    Attributes
    ----------
    i_rot : float
        The rotation inclination in rad.
    i_mag : float
        The magnetic obliquity in rad.
    latitude : float
        The mid-latitude of the ring in rad.
    width : float
        The width of the ring in rad.
    Rstar : float
        The radius of the star in solar radii.
    P_rot : float
        The rotation period of the star in days.
    phi : array
        The phase angles of the ring in rad. From 0 to 2 pi of size N.
    omega : float
        The rotation rate of the star in rad / day.
    v_bins : array
        The velocity bins to use for the spectral line.
    lat_min : float
        The minimum latitude of the ring in rad.
    lat_max : float
        The maximum latitude of the ring in rad.
    v_mids : array
        The midpoints of the velocity bins.
    """

    # init function takes the parameters of the ring and sets up the phi array
    def __init__(self, i_rot, i_mag, latitude, width, Rstar, P_rot, 
                 N=1000, gridsize=int(1e5), norm=10, v_bins=None,
                 v_mids=None, phi=None, omega=None, convert_to_kms=None):
        """Initialize the AuroralRing class.

        Parameters
        ----------
        i_rot : float
            The rotation inclination in rad.
        i_mag : float
            The magnetic obliquity in rad.
        latitude : float
            The mid-latitude of the ring in rad.
        width : float
            The width of the ring in rad.
        Rstar : float
            The radius of the star in solar radii.
        P_rot : float
            The rotation period of the star in days.
        N : int 
            The number of phase angles to use for the ring.
            The same is used for the velocity bins.
        gridsize : int
            The number of grid points to use for the numerical
            calculation of the ring.
        norm : float
            The normalization factor to use for the flux.
        v_bins : array
            The velocity bins to use for the spectral line.
        v_mids : array
            The midpoints of the velocity bins.
        phi : array
            The phase angles of the ring in rad. From 0 to 2 pi.
        omega : float
            The rotation rate of the star in rad / day.
        convert_to_kms : float  
            The conversion factor to convert from stellar radii / s to km / s.
        """
        self.i_rot = i_rot
        self.i_mag = i_mag
        self.latitude = latitude
        self.Rstar = Rstar



        if omega is None:
            self.P_rot = P_rot
            self.omega = 2 * np.pi / self.P_rot
        else:   
            self.omega = omega

        if convert_to_kms is None:
            self.convert_to_kms = self.Rstar * 695700. / 86400.
        else:
            self.convert_to_kms = convert_to_kms

        # set up the phi array
        if phi is None:
            self.phi = np.linspace(0, 2*np.pi, N*30)
        else:
            self.phi = phi

        # set up velocity bins based on the highest possible velocity
        if v_bins is None:
            # calculate omega
            
            vmax = self.omega * self.Rstar * 695700. / 86400. # km/s
            self.v_bins = np.linspace(-vmax*1.02, vmax*1.02, N)
        else:
            self.v_bins = v_bins


        # calculate max and min latitude of the ring using width
        if gridsize > 0:
            self.width = width
            self.lat_min = latitude - width/2
            self.lat_max = latitude + width/2
            self.THETA, self.PHI = create_spherical_grid(int(gridsize))

        # define binmids for the velocity bins
        if v_mids is None:
            self.v_mids = (self.v_bins[1:] + self.v_bins[:-1]) / 2
        else:
            self.v_mids = v_mids

        # normalization factor for the analytical flux calculation
        self.norm = norm

    # define a method to get the flux of the ring
    def get_flux_analytically(self, alpha):
        """Calculate the flux of the ring at a given rotational phase.

        Parameters
        ----------
        alpha : float
            The rotational phase of the star in rad.

        Returns
        -------
        flux : array
            The flux of the ring at the given rotational phase.
        """
        return get_analytical_spectral_line(self.phi, self.i_rot, self.i_mag, self.latitude, 
                                            alpha, self.v_bins, self.convert_to_kms, norm=self.norm)
    
    # define a method to get the flux of the ring numerically
    def get_flux_numerically(self, alpha, normalize=True):
        """Calculate the flux of the ring at a given rotational phase.

        Parameters
        ----------
        alpha : float
            The rotational phase of the star in rad.

        Returns
        -------
        flux : array
            The flux of the ring at the given rotational phase.
        """
        # get the x, y, z positions of the ring
        (self.x, self.y, self.z), self.z_rot, self.z_rot_mag = set_up_oblique_auroral_ring(self.THETA, self.PHI, 
                                                                            self.lat_max, self.lat_min, 
                                                                            self.i_rot, self.i_mag)
        
        # calculate the flux
        return numerical_spectral_line(alpha, self.x, self.y, self.z, self.z_rot,
                                       self.omega, self.Rstar, self.v_bins, normalize=normalize)
    

    def plot_sphere_with_auroral_ring(self, ax, alpha):

        ax.scatter(np.sin(self.THETA)*np.cos(self.PHI),
              np.sin(self.THETA)*np.sin(self.PHI),
              np.cos(self.THETA), c='grey', alpha=0.01)

        # plot the x axis as a dashed line
        ax.plot([-1, 1], [0, 0], [0, 0], c='w', ls='--')

        z_mag_alpha = rotate_around_arb_axis(alpha, self.z_rot_mag, self.z_rot)

        xr, yr, zr = rotate_around_arb_axis(alpha, np.array([self.x, self.y, self.z]), self.z_rot)

        # plot z_rot
        ax.plot([0, 1.5 *self.z_rot[0]], [0, 1.5 *self.z_rot[1]], [0,1.5 * self.z_rot[2]], c='r')


        # plot z_rot_mag
        ax.plot([0, z_mag_alpha[0]], [0, z_mag_alpha[1]], [0, z_mag_alpha[2]], c='yellow')

        # THE RING ----------

        # plot the rotated blue points
        ax.scatter(xr, yr, zr, alpha=1)

    def plot_layout_sphere(self, ax, view="observer front"):
        # set figure limits
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_zlim(-1., 1.)

        # label axes
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # rotate the figure such that x-axis point towards me
        if view == "observer front":
            ax.view_init(0, 0)
        elif view == "observer left":
            ax.view_init(0, 90)

        # let axes disappear
        ax.set_axis_off()


    def plot_setup_sphere(self):

        fig = plt.figure(figsize=(10, 5))
        spec = fig.add_gridspec(ncols=1, nrows=1)

        ax = fig.add_subplot(spec[0, 0], projection='3d')
        ax.set_axis_off()

        return fig, ax