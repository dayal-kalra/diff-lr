#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=32:59:00
#SBATCH --job-name=mup_12345_0.001
#SBATCH --error=err/%A_%a.err
#SBATCH --output=out/%A_%a.out
#SBATCH --mem=16G
#SBATCH --partition=class
#SBATCH --account=class 
#SBATCH --qos=default 
#SBATCH --gres=gpu:rtxa5000:1

eval "$(micromamba shell hook --shell bash)"
micromamba activate nanogpt

cd $SLURM_SUBMIT_DIR

srun python3 train_lr_predictor_sgd_mup.py --rollout_steps 100 --num_meta_steps 200
