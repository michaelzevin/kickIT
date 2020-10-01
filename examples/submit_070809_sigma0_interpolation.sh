#!/bin/bash

#SBATCH -A b1095
#SBATCH -p grail-ligo
#SBATCH -N 1
#SBATCH -n 28
#SBATCH -t 48:00:00
#SBATCH --mem-per-cpu=5G
#SBATCH --output="interpolation_070809_sigma0.out"

module purge all
module load python/anaconda3.6
source activate sgrb-py36

python /projects/b1095/michaelzevin/github/sgrb/interpolate_potentials.py --gal-path /projects/b1095/michaelzevin/sgrb/gal_files/070809/070809_sigma0_hostgal.pkl --multiproc 28 --interp-path /projects/b1095/michaelzevin/sgrb/interpolations/070809/070809_sigma0_interp.pkl --Rgrid 300 --Zgrid 100 --Rgrid-max 1000 --Zgrid-max 100

mv interpolation_070809_sigma0.out output/

