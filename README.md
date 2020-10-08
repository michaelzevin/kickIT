# kickIT
Codebase for the kickIT code

This repo houses code used for creating time-dependent models for the galactic hosts of GRBs, and kinematically evolving tracer particles through these potentials, 
as used in Zevin et al. 2020 (https://ui.adsabs.harvard.edu/abs/2019arXiv191003598Z/abstract). 

The main function is `run.py`. One can also create interpolations of the galactic potentials using `interpolate_potentials.py`, which will speed up the kinematic 
integration of tracer particles. 

The `examples/` directory contains the submission files and argument settings used in Zevin et al. 2020 (https://ui.adsabs.harvard.edu/abs/2019arXiv191003598Z/abstract). 
