# Geometric Model of an Auroral Ring 

Get a local copy of the repository:

```
git clone https://github.com/ekaterinailin/2023_10_Oblique_Ring.git
```

**introduction\_to\_auroral\_ring.ipynb** contains an example of how to create visualizations of rings, instantaneous spectral lines, and integrated light curves from those spectral lines.

**bri0024_fit_fake_line_bri0024_prior_on_i.py** and **bri0024_result_figures.py** produce a series of synthetic spectral lines, fit a model, and assess the uncertainties using MCMC. Includes thermal broadening of the line and photon noise of the measurement, but no radiative transfer (yet).
