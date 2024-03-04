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
    i_rot_true_sigma = 10/180*np.pi
    i_mag_true = 40 * np.pi/180
    i_mag_true_sigma = 10 * np.pi/180
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

    # produce list of alpha arrays, which each cover one quarter of the rotation
    phase_starts = [0,0.25,0.5,0.75]
    alphas = np.array([np.linspace(phase_start * np.pi * 2,
                                   (phase_start + 0.25) * np.pi * 2, 100) for phase_start in phase_starts])
    
    # use numerical flux as input data
    numerical_ideals = []
    for alpha in alphas:
        numerical_ideals.append(ring.get_phase_integrated_numerical_line(alpha))
    # convert to numpy array
    numerical_ideals = np.concatenate(numerical_ideals)

    # add some noise
    err = 0.1
    flux_err = np.ones_like(numerical_ideals) * err
    numerical_noisy =  numerical_ideals + np.random.normal(0, err, numerical_ideals.shape)

    # set negative values to zero
    numerical_noisy[numerical_noisy < 0] = 0
    

    # get analytical line to compare to
    models = []
    for alpha in alphas:
        alpha = alpha.reshape(100,1)# reshape to make dimensionality match
        models.append(ring.get_flux_analytically(alpha))
     
    # convert to numpy array
    models = np.concatenate(models)

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
        f.write(f'# number of lines = {len(phase_starts)}\n')
        f.write(f'# start phases = {phase_starts}\n')

    # WRITE THE INPUT DATA TO A CSV FILE including all spectral lines with one line per column
    # repeat vmids for each line, and flatten
    vmids_all = np.tile(ring.v_mids, len(phase_starts))
    # flatten the models
    modelsf = models.reshape(-1)
    # flatten the numerical_noisy
    numerical_noisyf = numerical_noisy.reshape(-1)
    # flatten the flux_err
    flux_errf = flux_err.reshape(-1)
    # add index to signify each line
    line_index = np.repeat(np.arange(len(phase_starts)), len(ring.v_mids))

    # create a dataframe
    df = pd.DataFrame({'v_mids': vmids_all,
                       'numerical_noisy': numerical_noisyf, 
                       'flux_err': flux_errf, 
                       'model': modelsf, 
                       'line_index': line_index})        
    


    df.to_csv(f'plots/fit_fake_line/{name}/data.csv', index=False)

    # - SHOW THE INPUT FAKE LINE FOR ALL LINES IN ONE PLOT offset by 1.2
    plt.figure(figsize=(7,3*len(phase_starts)))
    for i in range(len(phase_starts)):
        index = (df.line_index == i)
        plt.errorbar(df.v_mids[index].values, df.numerical_noisy[index].values - 1.2*i, 
                     yerr = df.flux_err[index], 
                     label=f'{phase_starts[i]:.2f}-{phase_starts[i]+0.25:.2f}', 
                     alpha=0.5)
        plt.plot(df.v_mids[index].values, df.model[index].values - 1.2*i, c="w", linestyle="--")

    plt.xlabel('v [km/s]')
    plt.ylabel('normalized flux')
    plt.legend(frameon=False)
    plt.xlim(-vmax*1.05, vmax*1.05)
    plt.savefig(f'plots/fit_fake_line/{name}/line.png', dpi=300)
    plt.close()


    # - SHOW THE GEOMETRICAL SETUP

    fig, ax = ring.plot_setup_sphere()
    ring.plot_sphere_with_auroral_ring(ax, alpha=0)
    ring.plot_layout_sphere(ax, view="observer left")
    plt.savefig(f'plots/fit_fake_line/{name}/setup.png', dpi=300)

    # - LOG-LIKELIHOOD ESTIMATE

    def log_prior(theta: tuple) -> float:

        l, i_rot, i_mag = theta

        if  (0 < l < np.pi/2) & (0 < i_rot < np.pi) & (0 < i_mag < np.pi/2):
            return -0.5 * (i_rot - i_rot_true) ** 2 / i_rot_true_sigma ** 2 - 0.5 * (i_mag - i_mag_true) ** 2 / i_mag_true_sigma ** 2

        return -np.inf

    def log_likelihood(theta: tuple) -> np.array:  

        # latitude, log_f = theta
        latitude, i_rot, i_mag = theta
        

        models = []
        for alpha in alphas:
            model = get_analytical_spectral_line(phi, i_rot, i_mag, latitude, 
                                                alpha.reshape((100,1)), v_bins, convert_to_kms=vmax)
            models.append(model)
        
        models = np.concatenate(models)

        if np.isnan(models).any():
            return -np.inf
        elif np.isinf(models).any():
            return -np.inf
        else:
            sigma2 = flux_err**2
            
            return -0.5 * np.sum(np.sum((numerical_noisy - models) ** 2 / sigma2 + np.log(sigma2)))


    # - MCMC

    # initialize the walkers
    pos = np.array([np.pi/4, 2.5, 0.5]) + 0.01*np.random.randn(32, 3)
    nwalkers, ndim = pos.shape
    print(nwalkers)


    # parallelize the process
    with Pool(processes=5) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
                                        pool=pool)
        # run MCMC
        sampler.run_mcmc(pos, 10000, progress=True)






    # # - WALKER PLOT

    labels = ["l", 'i_rot', 'i_mag']
    samples = sampler.get_chain()
    # set up a three panel plot
    fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
    for i in range(3):
        ax = axes[i]
        ax.plot(samples[i, :], "w", alpha=0.5, linewidth=0.5)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.set_xlabel("step number")
    
    plt.savefig(f'plots/fit_fake_line/{name}/walkers.png', dpi=300)



    samples = sampler.get_chain(discard=5000 ).reshape(-1,3)
    # save samples to csv file
    sdf = pd.DataFrame(samples, columns=["l",'i_rot', 'i_mag'])
    sdf.to_csv(f'plots/fit_fake_line/{name}/samples.csv', index=False)

        
    # make a figure with the best result for the spectral line
    latitude_fit, i_rot_fit, i_mag_fit = np.median(samples, axis=0)

    models = []
    for alpha in alphas:
        model = get_analytical_spectral_line(phi, i_rot_fit, i_mag_fit, latitude_fit, 
                                            alpha.reshape(100,1), v_bins, convert_to_kms=vmax)
        models.append(model)

    

    plt.figure(figsize=(7,6))
    for i in range(len(phase_starts)):
        index = (df.line_index == i)
        plt.errorbar(df.v_mids[index].values, df.numerical_noisy[index].values - 1.2*i, 
                     yerr = df.flux_err[index], 
                     label=f'{phase_starts[i]:.2f}-{phase_starts[i]+0.25:.2f}', 
                     alpha=0.5)
        plt.plot(df.v_mids[index].values, models[i] - 1.2*i, c="w", linestyle="--")
  
    plt.legend(frameon=False)
    plt.xlabel('v [km/s]')
    plt.ylabel('normalized flux')
    plt.savefig(f'plots/fit_fake_line/{name}/line_fit.png', dpi=300)









