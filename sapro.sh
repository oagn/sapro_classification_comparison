#!/bin/bash --login
#SBATCH --job-name=sapro_classification
#SBATCH --error=%x.e.%j
#SBATCH --output=%x.o.%j
#SBATCH --partition=gpu
#SBATCH --time=60:00:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --mem=148g
#SBATCH --account=scw1985

clush -w $SLURM_NODELIST "sudo /apps/slurm/gpuset_3_exclusive"

echo '----------------------------------------------------'
echo ' NODE USED = '$SLURM_NODELIST
echo ' SLURM_JOBID = '$SLURM_JOBID
echo ' OMP_NUM_THREADS = '$OMP_NUM_THREADS
echo ' ncores = '$NP
echo ' PPN = ' $PPN
echo '----------------------------------------------------'
#
echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo SLURM job ID is $SLURM_JOBID
#
echo Number of Processing Elements is $NP
echo Number of mpiprocs per node is $PPN
env

test=sapro_classification
input_dir=$HOME/workarea/sapro_classification_comparison
WDPATH=/scratch/$USER/${test}.$SLURM_JOBID
rm -rf ${WDPATH}
mkdir -p ${WDPATH}
cd ${WDPATH}

cp ${input_dir}/*.py ${WDPATH}

module purge
module load anaconda

source activate keras-jax

conda list

start="$(date +%s)"
time conda run -n keras-jax python ${test}.py 
stop="$(date +%s)"
finish=$(( $stop-$start ))
echo deep learning ${test} $SLURM_JOBID Job-Time $finish seconds
echo Keras End Time is `date`
