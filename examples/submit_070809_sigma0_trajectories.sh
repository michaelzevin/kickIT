#!/bin/bash

#SBATCH -A b1095
#SBATCH -p grail-std
#SBATCH -N 1
#SBATCH -n 28
#SBATCH -t 240:00:00
#SBATCH --output="070809_sigma0_trajectories.out"

module purge all
module load python/anaconda3.6
source activate sgrb-py36

mkdir -p /projects/b1095/michaelzevin/sgrb/output_files/070809/070809_sigma0/

python /projects/b1095/michaelzevin/github/sgrb/run.py --grb 070809 --Nsys 1000000 --output-dirpath /projects/b1095/michaelzevin/sgrb/output_files/070809/070809_sigma0/ --sgrb-path /projects/b1095/michaelzevin/sgrb/data/sgrb_prospector.txt --gal-path /projects/b1095/michaelzevin/sgrb/gal_files/070809/070809_sigma0_hostgal.pkl --interp-path /projects/b1095/michaelzevin/sgrb/interpolations/070809/070809_sigma0_interp.pkl --Tint-max 120 --resolution 1000 --multiproc 28

mv 070809_sigma0_trajectories.out /projects/b1095/michaelzevin/sgrb/submission_files/trajectories/output/

