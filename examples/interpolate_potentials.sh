#!/bin/bash

source activate sgrb-py36

python /Users/michaelzevin/research/github/sgrb/interpolate_potentials.py \
--gal-path /Users/michaelzevin/research/sgrb/gal.tmp \
--interp-path interp.pkl \
--Rgrid 100 \
--Zgrid 50 \
--Rgrid-max 1000 \
--Zgrid-max 100 \
--multiproc max \
