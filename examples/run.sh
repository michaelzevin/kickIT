#!/bin/bash

source activate sgrb-py36

python /Users/michaelzevin/research/github/sgrb/run.py \
--grb 070809 \
--Nsys 10 \
--smhm-relation Moster \
--smhm-sigma 0 \
--sgrb-path /Users/michaelzevin/research/sgrb/data/sgrb_hostprops_offsets.txt \
--samples-path /Users/michaelzevin/research/sgrb/data/example_bns.dat \
--output-dirpath /Users/michaelzevin/research/sgrb/output_files/test \
--verbose \
--save-traj \
--downsample 100 \
--sample-progenitor-props \
--interp-path /Users/michaelzevin/research/sgrb/interp_tests/interp.pkl \
--Tint-max 120
--multiproc max \
#--fixed-birth 50 \
#--fixed-potential 90 \
#--differential-prof \
