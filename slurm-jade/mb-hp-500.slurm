#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=1-1
#SBATCH --partition=small
#SBATCH --mem=16G 
#SBATCH --cpus-per-task=4

## output files
#SBATCH --output=/jmain02/home/J2AD008/wga37/mxr40-wga37/logs/%x.%j.out
#SBATCH --error=/jmain02/home/J2AD008/wga37/mxr40-wga37/logs/%x.%j.err

module purge
module load cuda/11.2
module load python/anaconda3

source ~/.bashrc
conda activate diffusion
module load pytorch/1.12.1

export PYTHONPATH="${PYTHONPATH}:/jmain02/home/J2AD008/wga37/mxr40-wga37/github/mbpo_pytorch"

GROUP=marc-probens-errors-nov9_datasets
SEED=1
DATASET=/jmain02/home/J2AD008/wga37/mxr40-wga37/github/diffusion-wm/datasets/test_datasets_nov9/Hopper
LOAD_STEP=1000000

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
export WANDB_API_KEY=833a88a1139a7f5523d4b39cfd80bbe8f6abb710
export WANDB_MODE=offline

cd /jmain02/home/J2AD008/wga37/mxr40-wga37/github/mbpo_pytorch
/jmain02/home/J2AD008/wga37/mxr40-wga37/.conda/envs/diffusion/bin/python3 mbpo_error_eval.py --group $GROUP --env_name Hopper-v3 --rollout_min_length 500 --rollout_max_length 500 --rollout_batch_size 1000 --load_path $DATASET --load_step $LOAD_STEP
