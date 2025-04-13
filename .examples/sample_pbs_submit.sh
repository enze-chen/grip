#!/bin/bash -l
#PBS -A account
#PBS -l nodes=1:ppn=24
#PBS -N GRIP
#PBS -l walltime=12:00:00
#PBS -q prod
#PBS -j oe
#PBS -o zjob.out

# ------------------------------------------------

cd ${PBS_O_WORKDIR}

NNODES=`wc -l < $PBS_NODEFILE`
NRANKS=24

NTOTRANKS=$(( NNODES * NRANKS ))
echo "Running GRIP on" $NTOTRANKS "cores."

mpiexec -n ${NTOTRANKS} -ppn ${NRANKS} python3 main.py

