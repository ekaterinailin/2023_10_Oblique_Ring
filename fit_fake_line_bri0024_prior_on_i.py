import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from funcs.analytical import get_analytical_spectral_line
from funcs.auroralring import AuroralRing

from scipy.ndimage import gaussian_filter1d

from multiprocessing import Pool
import emcee

# dark mode
# plt.style.use('dark_background')


def log_probability(theta: tuple) -> float:

    lp = log_prior(theta)
    
    if not np.isfinite(lp):
        return -np.inf
    
    return lp + log_likelihood(theta)



# start script
if __name__ == "__main__":

    name = "2024_03_07_bri2"

    # if plots/fit_fake_line/{name} does not exist, create it
    if not os.path.exists(f'plots/fit_fake_line/{name}/'):
        os.makedirs(f'plots/fit_fake_line/{name}')


    # set up the ring
    i_rot_true = 51.7/180*np.pi
    i_rot_true_sigma = 5/180*np.pi
    i_mag_true = 55 * np.pi/180

    sin_i_mag_true = np.sin(i_mag_true)
    sin_i_rot_true = np.sin(i_rot_true)
    latitude_true = 65*np.pi/180
    phi0 = 0.4 * np.pi * 2

    # stellar parameters
    P_rot= 3.052 / 24. # 2.84 hours from Hallinan+2015
    omega = 2 * np.pi / P_rot
    Rstar = 0.109 # 0.1 solar radii roughly because M9 type Fillipazzo et al. 2015
    vmax = omega * Rstar * 695700. / 86400. # in km/s


    # velocity bins and angle bins
    resol = 45
    v_bins = np.linspace(-vmax*1.2, vmax*1.2, resol + 1)
    v_mids = (v_bins[1:] + v_bins[:-1])/2

    # set up the phi angle resolution
    phi = np.linspace(0, 2*np.pi, 180)
    thermal_broadening = 20 / 2.355
    gaussbroadening = resol / 2.1 * thermal_broadening /vmax
    

    # set up the ring
    ring = AuroralRing(i_rot=i_rot_true, i_mag=i_mag_true, latitude=latitude_true,
                    width=0.2 * np.pi/180, Rstar=Rstar, P_rot=P_rot, N=60, 
                     gridsize=int(4e5), v_bins=v_bins, v_mids=v_mids,
                    phi=phi, omega=omega, convert_to_kms=vmax)
 

    # produce list of alpha arrays, which each cover one quarter of the rotation
    phase_starts = [0.,0.25,0.5,0.75]
    alphas = np.array([np.linspace(phase_start * np.pi * 2 + phi0,
                                   (phase_start + 0.25) * np.pi * 2 + phi0, 100)%(2*np.pi) for phase_start in phase_starts])
    
    # use numerical flux as input data
    numerical_ideals = []
    for alpha in alphas:
        numerical_ideals.append(ring.get_phase_integrated_numerical_line(alpha))
    # convert to numpy array
    numerical_ideals = np.concatenate(numerical_ideals)

    # add some noise
    err = 1/18.
    flux_err = np.ones_like(numerical_ideals) * err
    numerical_noisy =  numerical_ideals + np.random.normal(0, err, numerical_ideals.shape)

    # set negative values to zero
    numerical_noisy[numerical_noisy < 0] = 0
    

    # get analytical line to compare to
    models_ideal = []
    models_ideal_nogauss = []
    for alpha in alphas:
        alpha = alpha.reshape(100,1)# reshape to make dimensionality match
        model = ring.get_flux_analytically(alpha)
        models_ideal_nogauss.append(model)
        model = gaussian_filter1d(model, gaussbroadening, mode="constant", cval=0)
        models_ideal.append(model)
     
    # convert to numpy array
    models_ideal = np.concatenate(models_ideal)
    models_ideal_nogauss = np.concatenate(models_ideal_nogauss)

    # convolbe models_ideal with a gaussian with a width of 10km/s

    
    models_noisy = models_ideal + np.random.normal(0, err, models_ideal.shape)


    # convert 5km/s 


    # - write out a file with the input data using f-notation
    with open(f'plots/fit_fake_line/{name}/input.txt', 'w') as f:
        f.write(f'# inclination of rot. axis = {i_rot_true*180/np.pi:.3f} deg\n')
        f.write(f'# mag. obliquity = {i_mag_true*180/np.pi:.3f} deg\n')
        f.write(f'# latitude = {latitude_true*180/np.pi:.3f} deg\n')
        f.write(f'# rel. err. = {err:.3f}\n')
        f.write(f'# stellar rotation period = {P_rot:.3f} d\n')
        f.write(f'# stellar radius = {Rstar:.3f} solar radii\n')
        f.write(f'# maximum velocity = {vmax:.3f} km/s\n')
        f.write(f'# omega = {omega:.3f} rad/day\n')
        f.write(f'# number of lines = {len(phase_starts)}\n')
        f.write(f'# start phases = {phi0}\n')

    # WRITE THE INPUT DATA TO A CSV FILE including all spectral lines with one line per column
    # repeat vmids for each line, and flatten
    vmids_all = np.tile(ring.v_mids, len(phase_starts))
    # flatten the models
    models_idealf = models_ideal.reshape(-1)

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
                       'model': models_idealf, 
                       'line_index': line_index})        
    


    df.to_csv(f'plots/fit_fake_line/{name}/data.csv', index=False)

    # - SHOW THE INPUT FAKE LINE FOR ALL LINES IN ONE PLOT offset by 1.2
    plt.figure(figsize=(7,3*len(phase_starts)))
    for i in range(len(phase_starts)):
        index = (df.line_index == i)
        plt.errorbar(df.v_mids[index].values, models_noisy[index] - 1.2*i, 
                     yerr = df.flux_err[index], 
                     label=f'{phase_starts[i]:.2f}-{phase_starts[i]+0.25:.2f}', 
                     alpha=0.5)
        
        plt.plot(df.v_mids[index].values, models_ideal_nogauss[index] - 1.2*i, 
                     alpha=0.5, c="grey", linestyle=":")
        plt.plot(df.v_mids[index].values, df.model[index].values - 1.2*i, c="grey", linestyle="--")

    plt.xlabel('v [km/s]')
    plt.ylabel('normalized flux')
    plt.legend(frameon=False)
    plt.xlim(-vmax*1.2, vmax*1.2)
    plt.savefig(f'plots/fit_fake_line/{name}/line.png', dpi=300)
    plt.close()


    # - SHOW THE GEOMETRICAL SETUP

    fig, ax = ring.plot_setup_sphere()
    ring.plot_sphere_with_auroral_ring(ax, alpha=0)
    ring.plot_layout_sphere(ax, view="observer left")
    plt.savefig(f'plots/fit_fake_line/{name}/setup.png', dpi=300)

    # - LOG-LIKELIHOOD ESTIMATE

    def log_prior(theta: tuple) -> float:

        l, i_rot, i_mag, phi0 = theta

        if  (0 < l < np.pi/2) & (0 < i_rot < np.pi) & (0 < i_mag < np.pi/2) & (np.pi/2 < phi0 < np.pi):
            return -0.5 * (i_rot - i_rot_true) ** 2 / i_rot_true_sigma ** 2

        return -np.inf

    def log_likelihood(theta: tuple) -> np.array:  

        latitude, i_rot, i_mag, phi0 = theta
        alphas = np.array([np.linspace(phase_start * np.pi * 2 + phi0,
                                (phase_start + 0.25) * np.pi * 2 + phi0, 100).reshape((100,1)) %(2*np.pi) for phase_start in phase_starts])

        models = []
        for alpha in alphas:
            model = get_analytical_spectral_line(phi, i_rot, i_mag, latitude, 
                                                alpha, v_bins, convert_to_kms=vmax)
            model = gaussian_filter1d(model, gaussbroadening, mode="constant", cval=0)
            models.append(model)
        
        models = np.concatenate(models)

        if np.isnan(models).any():
            return -np.inf
        elif np.isinf(models).any():
            return -np.inf
        else:
            sigma2 = flux_err**2
            
            return -0.5 * np.sum(np.sum((models_noisy - models) ** 2 / sigma2 + np.log(sigma2)))

    # # run a minimization to get the best fit parameters
    # from scipy.optimize import minimize

    # # minimize the negative log likelihood
    # result = minimize(lambda *x: -log_probability(*x), x0=[1.1, i_rot_true, i_mag_true, 3])
    # print(result.x)

    # - MCMC

    # initialize the walkers with the minimization result
    pos = np.array([1.1, i_rot_true, i_mag_true, 3.]) + 0.3*np.random.randn(32, 4)
    pos[:,2] = np.random.uniform(0, np.pi/2, 32)
    pos[:,0] = np.random.uniform(0, np.pi/2, 32)
    pos[:,3] = np.random.uniform(np.pi/2, np.pi, 32)
    nwalkers, ndim = pos.shape


    # parallelize the process
    with Pool(processes=5) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
                                        pool=pool)
        # run MCMC
        sampler.run_mcmc(pos, 10000, progress=True)

    # - WALKER PLOT

    labels = ["l", 'i_rot', 'i_mag', 'phi0']
    samples = sampler.get_chain()
    # set up a four panel plot
    fig, axes = plt.subplots(4, figsize=(10, 10), sharex=True)
    for i in range(4):
        ax = axes[i]
        ax.plot(samples[:,:,i], "grey", alpha=0.5, linewidth=0.5)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.set_xlabel("step number")
    
    plt.savefig(f'plots/fit_fake_line/{name}/walkers.png', dpi=300)



    samples = sampler.get_chain(discard=5000 ).reshape(-1,4)
    # save samples to csv file
    sdf = pd.DataFrame(samples, columns=["l",'i_rot', 'i_mag', 'phi0'])
    sdf.to_csv(f'plots/fit_fake_line/{name}/samples.csv', index=False)

        
    # make a figure with the best result for the spectral line
    latitude_fit, i_rot_fit, i_mag_fit, phio_0_fit = np.median(samples, axis=0)

    alphas = np.array([np.linspace((phase_start * np.pi * 2 + phio_0_fit),
                                   ((phase_start + 0.25) * np.pi * 2 + phio_0_fit), 100)  % (2*np.pi)
                                   for phase_start in phase_starts])

    models = []
    for alpha in alphas:
        model = get_analytical_spectral_line(phi, i_rot_fit, i_mag_fit, latitude_fit, 
                                            alpha.reshape(100,1), v_bins, convert_to_kms=vmax)
        model = gaussian_filter1d(model, gaussbroadening, mode="constant", cval=0)
        models.append(model)
    models = np.concatenate(models)    

    plt.figure(figsize=(7,12))
    for i in range(len(phase_starts)):
        index = (df.line_index == i)
        plt.errorbar(df.v_mids[index].values, models_noisy[index] - 1.2*i + 4, 
                     yerr = df.flux_err[index], 
                     alpha=0.5)
        # add text to the plot with the phase range
        plt.text(29, 5 - 1.2*i, fr'synthetic obs. @ $\phi={phase_starts[i]:.2f}-{phase_starts[i]+0.25:.2f}$', 
                 fontsize=8, ha='center', va='center', color='grey')
        plt.plot(df.v_mids[index].values, models[index] - 1.2*i + 4, c="grey", linestyle="--")
        plt.plot(df.v_mids[index].values, models_ideal[index] - 1.2*i + 4, c="k", linestyle=":", alpha=0.5)
        plt.plot(df.v_mids[index].values, models_ideal_nogauss[index] - 1.2*i + 4, c="olive", linestyle=":", alpha=0.5)
    # make a legend that explains the different lines
    plt.plot([],[], c="k", linestyle=":", label="auroral model (with thermal broadening)")
    plt.plot([],[], c="olive", linestyle=":", label="auroral model (without thermal broadening)")
    plt.plot([],[], c="grey", linestyle="--", label="best-fit model")

    plt.legend(frameon=False, loc=2)
    plt.xlabel('v [km/s]')
    plt.ylabel('normalized flux')
    plt.xlim(-vmax*1.2, vmax*1.2)
    plt.title(r"BRI 0021-0214 @ $H\alpha$")
    plt.savefig(f'plots/fit_fake_line/{name}/line_fit.png', dpi=300)

    # write out all the data needed to reproduce the plot above to a csv file
    df = pd.DataFrame({'v_mids': df.v_mids.values,
                       'model_noisy': models_noisy, 
                       'flux_err': flux_err, 
                       'model': models_ideal, 
                       'model_fit': models, 
                       'line_index': df.line_index.values,
                       'model_nogauss': models_ideal_nogauss,
                       })

    df.to_csv(f'plots/fit_fake_line/{name}/data_fit.csv', index=False)

    # reproduce the figure above from the df
    plt.figure(figsize=(7,12))
    for i in range(len(phase_starts)):
        index = (df.line_index == i)
        plt.errorbar(df.v_mids[index].values, df.model_noisy[index].values - 1.2*i + 4, 
                     yerr = df.flux_err[index].values, 
                    #  label=fr'$\phi={phase_starts[i]:.2f}-{phase_starts[i]+0.25:.2f}$', 
                     alpha=0.5)
        # add text to the plot with the phase range
        plt.text(29, 5 - 1.2*i, fr'synthetic obs. @ $\phi={phase_starts[i]:.2f}-{phase_starts[i]+0.25:.2f}$', 
                 fontsize=8, ha='center', va='center', color='grey')
        plt.plot(df.v_mids[index].values, df.model_fit[index].values - 1.2*i + 4, c="grey", linestyle="--")
        plt.plot(df.v_mids[index].values, df.model[index].values - 1.2*i + 4, c="k", linestyle=":", alpha=0.5)
        plt.plot(df.v_mids[index].values, df.model_nogauss[index].values - 1.2*i + 4, c="olive", linestyle=":", alpha=0.5)
    # make a legend that explains the different lines
    plt.plot([],[], c="k", linestyle=":", label="auroral model (with thermal broadening)")
    plt.plot([],[], c="grey", linestyle="--", label="best-fit model")
    plt.plot([],[], c="olive", linestyle=":", label="auroral model (without thermal broadening)")

    plt.legend(frameon=False, loc=2)
    plt.xlabel('v [km/s]')
    plt.ylabel('normalized flux')
    plt.xlim(-vmax*1.2, vmax*1.2)
    plt.title(r"BRI 0021-0214 @ $H\alpha$")
    plt.savefig(f'plots/fit_fake_line/{name}/line_fit_from_df.png', dpi=300)










