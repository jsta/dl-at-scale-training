#!/bin/bash 
#SBATCH -C gpu
#SBATCH -A ntrain4
#SBATCH -q regular
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 32
#SBATCH --gpus-per-node 4
#SBATCH --time=01:00:00
#SBATCH --image=nersc/pytorch:24.08.01
#SBATCH --module=gpu,nccl-plugin
#SBATCH --reservation=dlscale_training_2
#SBATCH -J vit-era5-mp
#SBATCH -o %x-%j.out

DATADIR=/pscratch/sd/s/shas1693/data/dl-at-scale-training-data
LOGDIR=${SCRATCH}/dl-at-scale-training/logs
mkdir -p ${LOGDIR}
args="${@}"

export HDF5_USE_FILE_LOCKING=FALSE

# Profiling
if [ "${ENABLE_PROFILING:-0}" -eq 1 ]; then
    echo "Enabling profiling..."
    NSYS_ARGS="--trace=cuda,cublas,nvtx --cuda-graph-trace=node --kill none -c cudaProfilerApi -f true"
    NSYS_OUTPUT=${LOGDIR}/${PROFILE_OUTPUT:-"profile"}
    export PROFILE_CMD="nsys profile $NSYS_ARGS -o $NSYS_OUTPUT"
fi

export MASTER_ADDR=$(hostname)

# Reversing order of GPUs to match default CPU affinities from Slurm
export CUDA_VISIBLE_DEVICES=3,2,1,0

# if cuda graphs, use train_mp_graphs.py
set -x
srun -u shifter -V ${DATADIR}:/data -V ${LOGDIR}:/logs \
    bash -c "
    source export_DDP_vars.sh
    ${PROFILE_CMD} python train_mp.py ${args}
    "
