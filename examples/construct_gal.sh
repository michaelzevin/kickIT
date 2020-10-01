#!/bin/bash

source activate sgrb-py36

grb=$1
sigma=$2

python /projects/b1095/michaelzevin/github/sgrb/run.py \
--grb ${grb} \
--Nsys 1 \
--sgrb-path /projects/b1095/michaelzevin/sgrb/data/sgrb_prospector.txt \
--samples-path /projects/b1095/michaelzevin/sgrb/data/example_bns.dat \
--output-dirpath /projects/b1095/michaelzevin/sgrb/gal_files/${grb}/ \
--save-traj \
--downsample 100 \
--Tint-max 120 \
--disk-profile DoubleExponential \
--dm-profile NFW \
--z-scale 0.05 \
--differential-prof \
--smhm-relation Moster \
--smhm-sigma ${sigma} \
--gal-only \
--label ${grb}_sigma${sigma}_hostgal \
