#!/bin/bash
#PBS -V
#PBS -q fill
#PBS -l nodes=1:ppn=8
#PBS -N RUNNAME

### Store starting dir
START_DIR=$(pwd)

### cd working directory
cd RUNDIR

### Executable Line

source /home/btaczak/.bashrc

conda activate openmc-env

export OPENMC_CROSS_SECTIONS=/opt/DagMC/git/nuclear_data/hdf5_version3/tend_2017_hdf5/cross_sections.xml
export HDF5_USE_FILE_LOCKING='FALSE'

python geometry_builder.py > geometry_build.log 2>&1
echo "GEOMETRY BUILD COMPLETE FOR RUNNAME"

python openmc_runner.py > output.log 2>&1
echo "STARTED RUN RUNNAME"

cd "$START_DIR"