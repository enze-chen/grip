#!/bin/bash
#SBATCH -A account
#SBATCH -N 1
#SBATCH -n 24
#SBATCH -J GRIP
#SBATCH -t 12:00:00
#SBATCH -p pbatch
#SBATCH -o zjob.o%j

###########################

echo "Running GRIP on" $SLURM_NTASKS "cores on" $SLURM_CLUSTER_NAME
srun -N1 -n$SLURM_NTASKS --wait=0 --kill-on-bad-exit=0 python3 main.py

