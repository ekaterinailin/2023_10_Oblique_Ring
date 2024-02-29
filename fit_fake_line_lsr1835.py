import os

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
    l, logf = theta
    if  (0 < l < np.pi/2) and (-10 < logf < -1):
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

    name = "2024_02_23_5"

    # if plots/fit_fake_line/{name} does not exist, create it
    if not os.path.exists(f'plots/fit_fake_line/{name}/'):
        os.makedirs(f'plots/fit_fake_line/{name}')


    # set up the ring
    logf_true = np.log(0.5)
    i_rot_true = 140/180*np.pi
    i_mag_true = 40 * np.pi/180
    sin_i_mag_true = np.sin(i_mag_true)
    sin_i_rot_true = np.sin(i_rot_true)
    latitude_true = 25*np.pi/180
    sin_latitude_true = np.sin(latitude_true)

    # size of the line
    n=50

    # stellar parameters
    P_rot= 2.84 / 24. # 2.84 hours from Hallinan+2015
    omega = 2*np.pi/P_rot
    Rstar = 0.1 # 0.1 solar radii roughly because M9 type
    vmax = omega * Rstar * 695700. / 86400.
    convert_to_kms = omega / 86400 * Rstar * 695700.

    # velocity bins and angle bins
    v_bins = np.linspace(-vmax*1.05, vmax*1.05, 201)
    v_mids = (v_bins[1:] + v_bins[:-1])/2

    int_from = np.pi/4*6
    int_to = np.pi/4*7
    phi = np.linspace(0, 2*np.pi, 360)

    
    # set up the rin
    ring = AuroralRing(i_rot=i_rot_true, i_mag=i_mag_true, latitude=latitude_true,
                    width=1 * np.pi/180, Rstar=Rstar, P_rot=P_rot, N=60, 
                     gridsize=int(4e5), v_bins=v_bins, v_mids=v_mids,
                    phi=phi, omega=omega, convert_to_kms=convert_to_kms)
    

    
    ffa_ = get_full_rotation_line(ring)

    # calculate the flux 
    # this serves as measurement
    full_flux_numerical = np.zeros_like(ring.v_mids)

    for alpha in np.linspace(int_from, int_to, n):
        full_flux_numerical += ring.get_flux_numerically(alpha, normalize=False)


    # use numerical flux as input data
    ffa_ = full_flux_numerical.copy()
    ffa_ = ffa_ / np.max(ffa_)

    # add some noise
    err = 0.05
    flux_err = np.ones_like(v_mids) * err
    ffa =  ffa_ + np.random.normal(0, err, len(ffa_))
    full_flux_numerical /= np.max(full_flux_numerical)
    alpha = np.linspace(int_from, int_to, 100).reshape(100,1)

    model = get_analytical_spectral_line(phi, i_rot_true, i_mag_true, latitude_true,
                                        alpha, v_bins, convert_to_kms=convert_to_kms)
   




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

    # - write out the data and true line, i.e. vmids, ffa, and full_flux_numerical
    # make a pandas dataframe
    df = pd.DataFrame({'v_mids': v_mids, 'ffa': ffa, 'full_flux_numerical': full_flux_numerical})
    df.to_csv(f'plots/fit_fake_line/{name}/data.csv', index=False)


    # - SHOW THE INPUT FAKE LINE
    
    plt.figure(figsize=(7,6))
    # plt.errorbar(ring.v_mids, ffa, yerr = flux_err, label='numerical w/ error')
    plt.plot(ring.v_mids, full_flux_numerical, label='numerical')
    plt.plot(ring.v_mids, model, label='analytical', c="w", linestyle="--")
    plt.legend(frameon=False)
    plt.xlabel('v [km/s]')
    plt.ylabel('normalized flux')
    plt.savefig(f'plots/fit_fake_line/{name}/line.png', dpi=300)
    plt.close()


    # - SHOW THE GEOMETRICAL SETUP

    fig, ax = ring.plot_setup_sphere()

    ring.plot_sphere_with_auroral_ring(ax, alpha=0)
    ring.plot_layout_sphere(ax, view="observer left")
    plt.savefig(f'plots/fit_fake_line/{name}/setup.png', dpi=300)

    # - LOG-LIKELIHOOD ESTIMATE

    def log_likelihood(theta: tuple) -> np.array:  

        latitude, log_f = theta
        
        model = get_analytical_spectral_line(phi, i_rot_true, i_mag_true, latitude, 
                                            alpha, v_bins, convert_to_kms=convert_to_kms)

        if np.isnan(model).any():
            return -np.inf
        else:
            sigma2 = flux_err**2 + model**2 * np.exp(2 * log_f)
            return -0.5 * np.sum((ffa - model) ** 2 / sigma2 + np.log(sigma2))

    # nll = lambda *args: -log_likelihood(*args)
    # initial = np.array([sin_latitude_true, logf_true]) + 0.01 * np.random.randn(4)
    # soln = minimize(nll, initial)
    # i, m, l, lf = soln.x

    # print("Maximum likelihood estimates:")
    # print("i_rot = {0:.3f}".format(np.arcsin(i)*180/np.pi))
    # print("logf = {0:.3f}".format(lf))
    # print("m = {0:.3f}".format(np.arcsin(m)*180/np.pi))
    # print("l = {0:.3f}".format(np.arcsin(l)*180/np.pi))


    # - MCMC

    # initialize the walkers
    pos = np.array([1.5,-3]) + 0.01*np.random.randn(32, 2)
    nwalkers, ndim = pos.shape

    # parallelize the process
    with Pool(processes=5) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
                                        #args=(ffa, flux_err, v_bins, phi, convert_to_kms),
                                        pool=pool)
        # run MCMC
        sampler.run_mcmc(pos, 15000, progress=True)


    # - WALKER PLOT

    fig, axes = plt.subplots(2, figsize=(10, 5), sharex=True)
    samples = sampler.get_chain(discard=5000 )


    # save samples to csv file
    df = pd.DataFrame(samples.reshape(-1,2), columns=["l","f"])
    df.to_csv(f'plots/fit_fake_line/{name}/samples.csv', index=False)


    labels = ["l","f"]

    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "w", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")
    plt.savefig(f'plots/fit_fake_line/{name}/walkers.png', dpi=300)

    # - CORNER PLOT

    flat_samples = sampler.get_chain(discard=5000, thin=15, flat=True)

    fig = corner.corner(flat_samples, labels=labels, 
                        truths=np.arcsin(np.array([
                                                   sin_latitude_true,0.1]))* 180 /np.pi)
    plt.savefig(f'plots/fit_fake_line/{name}/corner.png', dpi=300)
        
    # make a figure with the best result for the spectral line
    latitude, log_f = np.median(flat_samples, axis=0)
    model = get_analytical_spectral_line(phi, i_rot_true, i_mag_true, latitude, 
                                            ALPHA, v_bins, convert_to_kms=convert_to_kms)
    
    mf = np.max(model)
    if mf != 0:
        model /= mf

    plt.figure(figsize=(7,6))
    plt.errorbar(ring.v_mids, ffa, yerr = flux_err, label='model')
    plt.plot(ring.v_mids, model, label='best fit')
    plt.legend(frameon=False)
    plt.xlabel('v [km/s]')
    plt.ylabel('normalized flux')
    plt.savefig(f'plots/fit_fake_line/{name}/line_fit.png', dpi=300)









