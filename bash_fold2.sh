#!/bin/bash
#SBATCH --job-name=python_train_LN
#SBATCH --output=/blue/r.forghani/mdmahfuzalhasan/Lymph_Node_Segmentation/results/job.%J.out
#SBATCH --error=/blue/r.forghani/mdmahfuzalhasan/Lymph_Node_Segmentation/results/job.%J.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mdmahfuzalhasan@ufl.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40gb
#SBATCH --partition=hpg-ai
#SBATCH --gpus=a100:1
#SBATCH --time=15:00:00

module purge
module load conda
conda activate medical

# Navigate to the directory containing the script
cd /blue/r.forghani/mdmahfuzalhasan/Lymph_Node_Segmentation

# Execute the Python script
srun python train.py --fold 2