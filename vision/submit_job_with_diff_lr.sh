#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=2:59:00
#SBATCH --job-name=mup_12345_32
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

srun python3 train_cnn_sgd_mup_with_diff_lr.py --lr_predictor_ckpt_path lr_predictor_checkpoints/step_$1  --dataset_name cifar-10 --init_seed 65536 --width $2 --depth 5 --lr_peak 1.0 --lr_min_factor inf --augment True --warmup_exponent 1.0 --decay_schedule_name cosine --decay_exponent 1.0 --warmup_steps 2000 --num_steps 10_000 --batch_size 128 --momentum 0.0 --weight_decay 0.0 --save_ckpt True --rerun True
