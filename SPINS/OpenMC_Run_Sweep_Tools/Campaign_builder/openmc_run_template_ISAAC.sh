#!/bin/bash
# This file is a submission script to request the ISAAC resources from Slurm 
#SBATCH -J RUNNAME			       #The name of the job
#SBATCH -A ISAAC-UTK0325              # The project account to be charged
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=8          # cpus per node 
#SBATCH --partition=campus            # If not specified then default is "campus"
#SBATCH --time=0-01:00:00             # Wall time (days-hh:mm:ss)
#SBATCH --error=job.o%J	       # The file where run time errors will be dumped
#SBATCH --output=job.o%J	       # The file where the output of the terminal will be dumped
#SBATCH --qos=campus            # Group account allows for QOS of campus, overflow, long-utk	

### Store starting dir
START_DIR=$(pwd)

### cd working directory
cd RUNDIR

### Executable Line

source /home/btaczak/.bashrc

conda activate openmc-env

# Set OPENMC_CROSS_SECTIONS by loading openMC-data module if not already done in bash script
# module load openMC-data/endfb-vii.1-cross
export HDF5_USE_FILE_LOCKING='FALSE'

python geometry_builder.py > geometry_build.log 2>&1
echo "GEOMETRY BUILD COMPLETE FOR RUNNAME"

python openmc_runner.py > output.log 2>&1
echo "STARTED RUN RUNNAME"

cd "$START_DIR"