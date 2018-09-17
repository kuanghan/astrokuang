A new, object-oriented version for running GALFIT simulations with artificial
objects. Meant to replace run_sim_multiband_galfit, and part of it should be
branched out for SExtractor-only simulations.

Will contain the following files:
- prep_gfsim: prepare images for GAFLIT simulations if necessary (e.g., when
  differnt pixel scales are used for detection and measurement images)
- run_gfsim: driver script that reads in the parameter file and runs the 
  simulations

The scripts aim to do the following steps to simulate the detection/measurement
procedure of real galaxies:

1. Generate N (default N = 40) fake galaxies in an input image (usually an HST
image). Fake galaxies are generated using `iraf.mkobjects`.

2. Detect the fake galaxies using `SExtractor`.

3. Use `GALFIT` to fit the sizes, total magnitudes, and profile shapes of fake
galaxies. We use the measured properties of fake galaxies by `SExtractor` as
initial guesses. Fake galaxies that are not detected by `SExtractor` are not
measured by GALFIT.

We run these steps for a large number of times, and we collect statistics of
galaxy detection rate (completeness) and GALFIT measurement biases and errors
to construct "transfer functions" that we use to correct luminosity functions
or size-luminosity distributions.
