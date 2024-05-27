import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import corner
import astropy.units as u
from astropy.constants import c

from funcs.auroralring import AuroralRing

if __name__ == "__main__":

    # READING THE RESULTS AND INPUTS --------------------------------------------

    # path to the data and mcmc samples
    path = "2024_05_24_bri"

    # read in the data (or synthetic data)
    df = pd.read_csv(f"plots/fit_fake_line/{path}/data.csv")

    # read in the input parameters
    with open(f"plots/fit_fake_line/{path}/input.txt", "r") as f:
        params = f.read().split("\n")
        print(params)
        
    name = params[0].split("= ")[1:]
    true_lat = float(params[4].split(" ")[-2])
    true_inc = float(params[1].split(" ")[-2])
    sig_true_inc = float(params[2].split(" ")[-2])
    true_obl = float(params[3].split(" ")[-2])
    vmax = float(params[9].split(" ")[-2])
    phi0 = float(params[12].split(" ")[-1])
    phases = params[11].split("= [")[-1][:-1]
    # split by , and convert to list
    phases = [float(p) for p in phases.split(", ")]
    Prot = float(params[6].split(" ")[-2])
    Rstar = float(params[7].split(" ")[-3])
    T_broaden = float(params[13].split(" ")[-2])

    # print input parameters
    print(f"True Latitude: {true_lat}")
    print(f"True Inclination: {true_inc}")
    print(f"True Obliquity: {true_obl}")
    print(f"Vmax: {vmax}")
    print(f"Phi0: {phi0 * 180 / np.pi:.0f} deg")
    print(f"Phases: {phases} rot. periods")
    print(f"Rotation Period: {Prot} days")

    # read in the mcmc chain
    chain = pd.read_csv(f"plots/fit_fake_line/{path}/samples.csv")
    chain[["l","i_rot","i_mag","phi0"]] *= 180/np.pi 
    chain = chain[chain["phi0"]< 150] 

    # print chain head
    print("Chain:")
    print(chain.head())
    print("...")

    # CORNER PLOT ------------------------------------------------------------------

    # make corner plot
    cn = chain.to_numpy()
    labels=[r"$\theta$ [deg]", r"$i_{rot}$ [deg]", r"$i_{mag}$ [deg]", r"$\phi_0$ [deg]"]
    truths=[true_lat,true_inc, true_obl, phi0 * 180 / np.pi]
    # produce figure
    fig = corner.corner(cn, labels=labels, truths=truths, fontsize=20, hspace=0)

    # get axes from figure
    axes = np.array(fig.axes).flatten()

    # pretty plot
    for ax in axes:
        ax.tick_params(axis='both', labelsize=11)
        # increase label size
        ax.set_xlabel(ax.get_xlabel(), fontsize=15)
        ax.set_ylabel(ax.get_ylabel(), fontsize=15)
    plt.tight_layout()

    # save figure
    plt.savefig(f'plots/fit_fake_line/{path}/corner.png', dpi=300)

    # RESULTS ----------------------------------------------------------------------

    # calculate 16th, 50th, and 84th percentiles
    params = chain.quantile([0.16,0.5,0.84]).to_numpy()

    # print the results
    print("Results:")
    print(params)

    # PLOT THE FIT -----------------------------------------------------------------

    # plot the fit to the data
    df = pd.read_csv(f"plots/fit_fake_line/{path}/data_fit.csv")
    print(df.head())

    # convert v_mids to AA to get actual spectrum
    df.lam = (df.v_mids.values * u.km / u.s / c * 6562.8 * u.AA).to(u.AA).value 
    lammax = df.lam.max()

    # reproduce the figure above from the df
    plt.figure(figsize=(7,13))

    # plot each phase integration separately
    for i, p in enumerate(phases):
        index = (df.line_index == i)

        # plot the (synthetic) data
        plt.errorbar(df.lam[index], df.model_noisy[index].values - 1.25*i + 4, 
                        yerr = df.flux_err[index].values, 
                        alpha=0.8, c="b", lw=3,zorder=-10)
        
        # add text to the plot with the phase range
        plt.text(0.73*lammax, 3.78 - 1.25*i, fr'$\phi: [{p:.2f}-{p+0.25:.2f}]$', 
                    fontsize=14, ha='center', va='center', color='grey')
        
        # plot the unbroadened auroral model
        plt.plot(df.lam[index], df.model_nogauss[index].values - 1.25*i + 4, c="r", linestyle=":", alpha=0.8, lw=3)

        # plot the best-fit model
        plt.plot(df.lam[index], df.model_fit[index].values - 1.25*i + 4, c="k", linestyle="-", lw=3)

        # plot the broadened auroral model
        plt.scatter(df.lam[index], df.model[index].values - 1.25*i + 4, c="magenta", marker=".", alpha=0.8, s=100, zorder=10)
        
    # make a legend that explains the different lines
    plt.plot([],[], c="red", linestyle=":", label="auroral model (without thermal broadening)", lw=6, zorder=-10)
    plt.scatter([],[], c="magenta", marker=".", label="auroral model (with thermal broadening)", s=400, zorder=-8)
    plt.plot([],[], c="blue", linestyle="-", label="synthetic observation", lw=6, zorder=-6)
    plt.plot([],[], c="k", linestyle="-", label="best-fit model", lw=6, zorder=-4)

    # make legend handle in the order of the above plots
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [0,3,1,2]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], frameon=False, loc=2, fontsize=14)

    # add labels and limits
    plt.xlabel(r'$\lambda [\AA]$', fontsize=14)
    plt.ylabel('normalized flux', fontsize=14)
    plt.xlim(-lammax , lammax)
    plt.ylim(-0.2, 5.5)

    # increase size of the ticklabels
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # add title
    plt.title(fr"{name} @ H$\alpha$", fontsize=15)
    plt.tight_layout()
    plt.savefig(f'plots/fit_fake_line/{path}/line_fit_from_df.png', dpi=300)


    # PLOT THE RINGS AT EACH PHASE (end of integration time) ------------------------------------------------

    # velocity bins and angle bins
    resol = 45
    v_bins = np.linspace(-vmax*1.2, vmax*1.2, resol + 1)
    v_mids = (v_bins[1:] + v_bins[:-1])/2

    # set up the phi angle resolution
    phi = np.linspace(0, 2*np.pi, 180)


    # set up the ring
    ring = AuroralRing(i_rot=true_inc/180*np.pi, i_mag=true_obl/180*np.pi, latitude=true_lat/180*np.pi,
                    width=2 * np.pi/180, Rstar=Rstar, P_rot=Prot, N=60, 
                        gridsize=int(3e4), v_bins=v_bins, v_mids=v_mids,
                    phi=phi, omega=2 * np.pi / Prot, convert_to_kms=vmax)
    ring.get_flux_numerically(0)

    # plot the ring at each phase
    for i in np.array(phases) * 2. * np.pi:

        # set up the new ring
        fig, ax = ring.plot_setup_sphere()

        # plot the ring
        ring.plot_sphere_with_auroral_ring(ax, alpha=i, c_sphere="silver", c_imag="k", sphere_alpha=0.01, ring_alpha=1, c_ring="k")

        # define the observer position in the figure
        ring.plot_layout_sphere(ax, view="observer left")
        
        # pretty
        plt.tight_layout()

        # save
        plt.savefig(f'plots/fit_fake_line/{path}/sphere_{i}.png', dpi=300)

