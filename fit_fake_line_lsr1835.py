import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from funcs.analytical import get_analytical_spectral_line
from funcs.auroralring import AuroralRing

from multiprocessing import Pool
import emcee
import corner

# dark mode
plt.style.use('dark_background')



def log_prior(theta: tuple) -> float:

    l = theta[0]

    if  (0 < l < np.pi/2):
        return 0.0

    return -np.inf


def log_probability(theta: tuple) -> float:

    lp = log_prior(theta)
    
    if not np.isfinite(lp):
        return -np.inf
    
    return lp + log_likelihood(theta)



# start script
if __name__ == "__main__":

    name = "2024_03_04_1"

    # if plots/fit_fake_line/{name} does not exist, create it
    if not os.path.exists(f'plots/fit_fake_line/{name}/'):
        os.makedirs(f'plots/fit_fake_line/{name}')


    # set up the ring
    logf_true = np.log(0.5)
    i_rot_true = 140/180*np.pi
    i_mag_true = 40 * np.pi/180
    sin_i_mag_true = np.sin(i_mag_true)
    sin_i_rot_true = np.sin(i_rot_true)
    latitude_true = 65*np.pi/180
    sin_latitude_true = np.sin(latitude_true)

    # stellar parameters
    P_rot= 2.84 / 24. # 2.84 hours from Hallinan+2015
    omega = 2 * np.pi / P_rot
    Rstar = 0.1 # 0.1 solar radii roughly because M9 type
    vmax = omega * Rstar * 695700. / 86400. # in km/s

    # velocity bins and angle bins
    v_bins = np.linspace(-vmax*1.05, vmax*1.05, 201)
    v_mids = (v_bins[1:] + v_bins[:-1])/2

    # set up the phi angle resolution
    phi = np.linspace(0, 2*np.pi, 180)

    
    # set up the ring
    ring = AuroralRing(i_rot=i_rot_true, i_mag=i_mag_true, latitude=latitude_true,
                    width=0.2 * np.pi/180, Rstar=Rstar, P_rot=P_rot, N=60, 
                     gridsize=int(4e5), v_bins=v_bins, v_mids=v_mids,
                    phi=phi, omega=omega, convert_to_kms=vmax)

    # phase coverage
    int_from = np.pi
    int_to = np.pi / 2 * 3
    alpha = np.linspace(int_from, int_to, 100)    


    # use numerical flux as input data
    numerical_ideal = ring.get_full_numerical_line(alpha)

    # add some noise
    err = 0.1
    flux_err = np.ones_like(v_mids) * err
    numerical_noisy =  numerical_ideal + np.random.normal(0, err, len(numerical_ideal))
    
    # get analytical line to compare to
    alpha = alpha.reshape(100,1) # reshape to make dimensionality match
    model = ring.get_flux_analytically(alpha)
   

    # - write out a file with the input data using f-notation
    with open(f'plots/fit_fake_line/{name}/input.txt', 'w') as f:
        f.write(f'# inclination of rot. axis = {i_rot_true*180/np.pi:.3f} deg\n')
        f.write(f'# mag. obliquity = {i_mag_true*180/np.pi:.3f} deg\n')
        f.write(f'# latitude = {latitude_true*180/np.pi:.3f} deg\n')
        f.write(f'# logf = {logf_true:.3f}\n')
        f.write(f'# rel. err. = {err:.3f}\n')
        f.write(f'# stellar rotation period = {P_rot:.3f} d\n')
        f.write(f'# stellar radius = {Rstar:.3f} solar radii\n')
        f.write(f'# maximum velocity = {vmax:.3f} km/s\n')
        f.write(f'# omega = {omega:.3f} rad/day\n')

    # WRITE THE INPUT DATA TO A CSV FILE
    df = pd.DataFrame({'v_mids': v_mids, 
                       'numerical_ideal': numerical_ideal, 
                       'numerical_noisy': numerical_noisy,
                       'analytical_ideal': model})
    
    df.to_csv(f'plots/fit_fake_line/{name}/data.csv', index=False)


    # - SHOW THE INPUT FAKE LINE
    
    plt.figure(figsize=(7,6))
    plt.errorbar(ring.v_mids, numerical_noisy, yerr = flux_err, label='numerical w/ noise')
    plt.plot(ring.v_mids, numerical_ideal, label='numerical w/o noise')
    plt.plot(ring.v_mids, model, label='analytical w/o noise', c="w", linestyle="--")
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

        # latitude, log_f = theta
        latitude = theta[0]
        
        model = get_analytical_spectral_line(phi, i_rot_true, i_mag_true, latitude, 
                                            alpha, v_bins, convert_to_kms=vmax)

        if np.isnan(model).any():
            return -np.inf
        elif np.isinf(model).any():
            return -np.inf
        else:
            sigma2 = flux_err**2
            
            return -0.5 * np.sum((numerical_noisy - model) ** 2 / sigma2 + np.log(sigma2))


    # - MCMC

    # initialize the walkers
    pos = np.array([.5]) + 0.01*np.random.randn(32, 1)
    nwalkers, ndim = pos.shape

    # parallelize the process
    with Pool(processes=5) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
                                        pool=pool)
        # run MCMC
        sampler.run_mcmc(pos, 15000, progress=True)


    # - WALKER PLOT

    fig, axes = plt.subplots(1, figsize=(10, 5), sharex=True)
    samples = sampler.get_chain(discard=5000 )


    # save samples to csv file
    df = pd.DataFrame(samples.reshape(-1,1), columns=["l"])
    df.to_csv(f'plots/fit_fake_line/{name}/samples.csv', index=False)


    labels = ["l"]

    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "w", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")
    plt.savefig(f'plots/fit_fake_line/{name}/walkers.png', dpi=300)

        
    # make a figure with the best result for the spectral line
    latitude = np.median(samples, axis=0)
    model = get_analytical_spectral_line(phi, i_rot_true, i_mag_true, latitude, 
                                            alpha, v_bins, convert_to_kms=vmax)
    
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









