#!/bin/bash

source activate sgrb-py36

python /Users/michaelzevin/research/github/sgrb/run.py \
--grb 070809 \
--t0 20 \
--Nsys 1 \
--Tsteps 100 \
--Rgrid 500 \
--Zgrid 300 \
--Rgrid-max 1000 \
--Zgrid-max 100 \
--Tint-max 180 \
--interp-dirpath /Users/michaelzevin/research/sgrb/interp_potentials \
--sgrb-path /Users/michaelzevin/research/sgrb/data/sgrb_hostprops_offsets.txt \
--samples-path /Users/michaelzevin/research/sgrb/data/example_bns.dat \
--output-dirpath /Users/michaelzevin/research/sgrb/output_files/test \
--verbose \
--multiproc max \
--save-traj \
--downsample 100 \
#--sample-progenitor-props \
