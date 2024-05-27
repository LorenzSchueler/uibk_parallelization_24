#!/bin/bash

# Execute job in the partition "lva" unless you have special requirements.
#SBATCH --partition=lva

# Name your job to be able to identify it later
#SBATCH --job-name mandelbrot_seq

#SBATCH --ntasks 1
#SBATCH --output mpi.log

# Enforce exclusive node allocation, do not share with other jobs
#SBATCH --exclusive

module load openmpi/3.1.6-gcc-12.2.0-d2gmn55

mpiexec --mca btl_openib_allow_ib true ./build/mandelbrot_seq