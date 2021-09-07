#!/bin/bash
#PBS -S /bin/bash

#PBS -l nodes=20:ppn=1,walltime=24:00:00 -q gigat

# Set the job name
#PBS -N test_networks

# Set the output file and merge it to the sterr
#PBS -o out-hostname-XyZ-N1x1-qsub.txt
#PBS -j oe
#PBS -e out-hostname-XyZ-N1x1.txt


cd ${PBS_O_WORKDIR}

export mkPrefix=/u/sw
source $mkPrefix/etc/profile
module load gcc-glibc/9
module load cgal
module load boost
module load pybind11

export CGAL_DIR=${mkCgalPrefix}
export PYTHON="/u/sw/toolchains/gcc-glibc/11/base/bin/python3"
export BOOST_DIR=${mkBoostPrefix}
export ARMADILLO_DIR="/u/archive/dott/beraha/usr/armadillo-10.6.2"

make generate_pybind -j10

python3 run_gnk.py
