import os
import sys

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from funcs.analytical import get_analytical_spectral_line
from funcs.auroralring import AuroralRing

from scipy.optimize import minimize
from multiprocessing import Pool
import emcee
import corner

ALPHA = np.linspace(0, 2*np.pi, 100).reshape(100,1)

def get_full_rotation_line(ring: AuroralRing, alpha=ALPHA) -> np.array:

    full_flux_analytical = ring.get_flux_analytically(alpha)

    mf = np.max(full_flux_analytical)
    
    if mf != 0:
        full_flux_analytical /= mf

    return full_flux_analytical


def log_prior(theta: tuple) -> float:
    i, m, l, logf = theta
    if  (0<i<1) and (0 < m < 1) and (0 < l < 1) and (-10 < logf < 1):
        return 0.0
    return -np.inf


def log_probability(theta: tuple) -> float:
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)

# dark mode
plt.style.use('dark_background')

# start script
if __name__ == "__main__":


    # i_rots = [15, 30, 45, 60, 75, 90]
    # i_mags = [15, 30, 45, 60, 75, 90]


    # create a pandas data frame with all input, and output lines you need, and save it empty to results
    # name,i_rot_true,i_mag_true,latitude_true,log_f,n_datapoints,p_rot,r_star,max_vel_km_s,
    # i_rot_16,i_rot_50,i_rot_84,i_mag_16,i_mag_50,i_mag_84,lat_16,lat_50,lat_84,n_steps,n_walkers,
    # ml_i_rot,ml_i_mag,ml_logf,ml_latitude
    # results = pd.DataFrame(columns=['name','i_rot_true','i_mag_true','latitude_true','n_datapoints',
    #                                 'p_rot','r_star','max_vel_km_s','i_rot_16','i_rot_50','i_rot_84',
    #                                 'i_mag_16','i_mag_50','i_mag_84','lat_16','lat_50','lat_84',
    #                                 'n_steps','n_walkers','ml_i_rot','ml_i_mag','ml_logf','ml_latitude'])

    # results.to_csv('results/fit_fake_line_results.csv', index=False)

    N = 20000

    # read in i_rot_true and i_mag_true as arguments to the script
    i_rot_true = float(sys.argv[1])
    i_mag_true = float(sys.argv[2])

    name = f"2023_11_21_irot_{i_rot_true:.0f}_imag_{i_mag_true:.0f}"

    print("Fitting line for: ", name)
    

    # if plots/fit_fake_line/{name} does not exist, create it
    if not os.path.exists(f'plots/fit_fake_line/{name}/'):
        os.makedirs(f'plots/fit_fake_line/{name}')


    # set up the ring
    logf_true = np.log(0.5)
    i_rot_true = i_rot_true / 180 * np.pi
    i_mag_true = i_mag_true * np.pi / 180
    sin_i_mag_true = np.sin(i_mag_true)
    sin_i_rot_true = np.sin(i_rot_true)
    latitude_true = 88*np.pi/180
    sin_latitude_true = np.sin(latitude_true)

    # size of the line
    n=100

    # stellar parameters
    P_rot= 1.5 * np.pi
    omega = 2*np.pi/P_rot
    Rstar = 1
    vmax = omega * Rstar * 695700. / 86400.
    convert_to_kms = omega / 86400 * Rstar * 695700.

    # velocity bins and angle bins
    v_bins = np.linspace(-vmax*1.05, vmax*1.05, 101)
    v_mids = (v_bins[1:] + v_bins[:-1])/2
    phi = np.linspace(0, 2*np.pi, 1800)

    
    # set up the rin
    ring = AuroralRing(i_rot=i_rot_true, i_mag=i_mag_true, latitude=latitude_true,
                    width=3.9 * np.pi/180, Rstar=1, P_rot=1.5 * np.pi, N=60, 
                    norm=11, gridsize=int(4e5), v_bins=v_bins, v_mids=v_mids,
                    phi=phi, omega=omega, convert_to_kms=convert_to_kms)
    

    
    ffa_ = get_full_rotation_line(ring)

    # calculate the flux 
    # this serves as measurement
    full_flux_numerical = np.zeros_like(ring.v_mids)

    for alpha in np.linspace(0, 2*np.pi, n):
        full_flux_numerical += ring.get_flux_numerically(alpha)

    # use numerical flux as input data
    ffa_ = full_flux_numerical.copy()
    ffa_ = ffa_ / np.max(ffa_)

    # add some noise
    err = 0.02
    flux_err = np.ones_like(v_mids) * err
    ffa =  ffa_ - err/2 + np.random.randn(len(ffa_)) * err
    full_flux_numerical /= np.max(full_flux_numerical)


    # - write out a file with the input data using f-notation
    with open(f'plots/fit_fake_line/{name}/input.txt', 'w') as f:
        f.write(f'# inclination of rot. axis = {i_rot_true*180/np.pi:.3f} deg\n')
        f.write(f'# mag. obliquity = {i_mag_true*180/np.pi:.3f} deg\n')
        f.write(f'# latitude = {latitude_true*180/np.pi:.3f} deg\n')
        f.write(f'# logf = {logf_true:.3f}\n')
        f.write(f'# rel. err. = {err:.3f}\n')
        f.write(f'# spectral line size = {n}\n')
        f.write(f'# stellar rotation period = {P_rot:.3f} d\n')
        f.write(f'# stellar radius = {Rstar:.3f} solar radii\n')
        f.write(f'# maximum velocity = {vmax:.3f} km/s\n')
        f.write(f'# omega = {omega:.3f} rad/day\n')

    # - write out 

    # - write out the data and true line, i.e. vmids, ffa, and full_flux_numerical
    # make a pandas dataframe
    df = pd.DataFrame({'v_mids': v_mids, 'ffa': ffa, 'full_flux_numerical': full_flux_numerical})
    df.to_csv(f'plots/fit_fake_line/{name}/data.csv', index=False)


    # - SHOW THE INPUT FAKE LINE
    
    plt.figure(figsize=(7,6))
    plt.errorbar(ring.v_mids, ffa, yerr = flux_err, label='analytical')
    plt.plot(ring.v_mids, full_flux_numerical, label='numerical')
    plt.plot(ring.v_mids, ffa_, label='analytical true')
    plt.legend(frameon=False)
    plt.xlabel('v [km/s]')
    plt.ylabel('normalized flux')
    plt.savefig(f'plots/fit_fake_line/{name}/line.png', dpi=300)
    plt.close()

    # - SHOW THE GEOMETRICAL SETUP

    fig, ax = ring.plot_setup_sphere()

    ring.plot_sphere_with_auroral_ring(ax, alpha=alpha)
    ring.plot_layout_sphere(ax, view="observer left")
    plt.savefig(f'plots/fit_fake_line/{name}/setup.png', dpi=300)

    # - LOG-LIKELIHOOD ESTIMATE

    def log_likelihood(theta: tuple) -> np.array:  

        sin_i_rot, sin_i_mag, sin_latitude, log_f = theta
        
        model = get_analytical_spectral_line(phi, np.arcsin(sin_i_rot), 
                                                np.arcsin(sin_i_mag), np.arcsin(sin_latitude), 
                                                ALPHA, v_bins, convert_to_kms=convert_to_kms, 
                                                norm=11)

        mf = np.max(model)
        
        if mf != 0:
            model /= mf

        if np.isnan(model).any():
            return -np.inf
        else:
            sigma2 = flux_err**2 + model**2 * np.exp(2 * log_f)
            return -0.5 * np.sum((ffa - model) ** 2 / sigma2 + np.log(sigma2))

    nll = lambda *args: -log_likelihood(*args)
    initial = np.array([sin_i_rot_true, sin_i_mag_true, sin_latitude_true, logf_true]) + 0.01 * np.random.randn(4)
    soln = minimize(nll, initial)
    i, m, l, lf = soln.x

    print("Maximum likelihood estimates:")
    print("i_rot = {0:.3f}".format(np.arcsin(i)*180/np.pi))
    print("logf = {0:.3f}".format(lf))
    print("m = {0:.3f}".format(np.arcsin(m)*180/np.pi))
    print("l = {0:.3f}".format(np.arcsin(l)*180/np.pi))


    # - MCMC

    # initialize the walkers
    pos = soln.x + 1e-2 * np.random.randn(32, 4)
    nwalkers, ndim = pos.shape

    # parallelize the process
    with Pool(processes=5) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
                                        #args=(ffa, flux_err, v_bins, phi, convert_to_kms),
                                        pool=pool)
        # run MCMC
        sampler.run_mcmc(pos, N, progress=True)


    # - SAVE the MCMC chain

    samples = sampler.get_chain()[:,:, :-1]
    samples = samples.reshape(-1, ndim-1)
    df = pd.DataFrame(samples, columns=['sin_i_rot', 'sin_i_mag', 'sin_latitude'])
    df.to_csv(f'plots/fit_fake_line/{name}/chain.csv', index=False)

    # - WALKER PLOT

    fig, axes = plt.subplots(3, figsize=(10, 5), sharex=True)
    samples = sampler.get_chain()
    labels = ["i", "m", "l"]

    for i in range(ndim-1):
        ax = axes[i]
        ax.plot(np.arcsin(samples[:, :, i])*180/np.pi, "w", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")
    plt.savefig(f'plots/fit_fake_line/{name}/walkers.png', dpi=300)

    # - CORNER PLOT

    flat_samples = sampler.get_chain(discard=5000, thin=15, flat=True)[:,:-1]

    fs = np.arcsin(flat_samples)*180/np.pi

    try:
        fig = corner.corner(fs, labels=labels, 
                            truths=np.arcsin(np.array([sin_i_rot_true, 
                                                    sin_i_mag_true, 
                                                    sin_latitude_true]))* 180 /np.pi)
        plt.savefig(f'plots/fit_fake_line/{name}/corner.png', dpi=300)
    except:
        with open('plots/fit_fake_line/corner_failed.txt', 'a') as f:
            f.write(f'{name}\n')
        
    # calculate i_rot, i_mag, latitude at 16th, 50th and 84th percentiles
    print(fs.shape, fs[:, 0].shape)
    sin_i_rot_16, sin_i_rot_50, sin_i_rot_84 = np.percentile(samples[:,:, 0], [16, 50, 84])
    sin_i_mag_16, sin_i_mag_50, sin_i_mag_84 = np.percentile(samples[:,:, 1], [16, 50, 84])
    sin_lat_16, sin_lat_50, sin_lat_84 = np.percentile(samples[:,:, 2], [16, 50, 84])

    

    # add line to the results data with columns as introduced above
    with open('results/fit_fake_line_results.csv', 'a') as f:
        f.write(f'{name},{i_rot_true*180/np.pi},{i_mag_true*180/np.pi},'
                f'{latitude_true*180/np.pi},{n},'
                f'{P_rot},{Rstar},{vmax},'
                f'{sin_i_rot_16},{sin_i_rot_50},{sin_i_rot_84},'
                f'{sin_i_mag_16},{sin_i_mag_50},{sin_i_mag_84},'
                f'{sin_lat_16},{sin_lat_50},{sin_lat_84},'
                f'{N},{nwalkers},'
                f'{i},{m},{lf},{l}\n')











