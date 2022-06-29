#!/bin/bash                           # Batch script starts with shebang line

#SBATCH --partition=alpha
#SBATCH --ntasks=1                   # All #SBATCH lines have to follow uninterrupted
#SBATCH --time=01:00:00               # after the shebang line
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --mem-per-cpu=1700
#SBATCH --output=tensor-%j.out

module purge                          # Set up environment, e.g., clean modules environment
module load modenv/hiera  GCC/10.2.0  CUDA/11.1.1 OpenMPI/4.0.5 PyTorch/1.9.0

srun python torch_test_no_gpu.py
