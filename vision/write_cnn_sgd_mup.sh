echo "#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=16:59:00
#SBATCH --job-name=mup_$1_$2
#SBATCH --error=err/%A_%a.err
#SBATCH --output=out/%A_%a.out
#SBATCH --mem=16G
#SBATCH --partition=class
#SBATCH --account=class 
#SBATCH --qos=default 
#SBATCH --gres=gpu:rtxa5000:1

eval \"\$(micromamba shell hook --shell bash)\"
micromamba activate nanogpt

cd \$SLURM_SUBMIT_DIR

srun python3 train_cnn_sgd_mup.py --dataset_name cifar-10 --init_seed $1 --width 16 --depth 5 --lr_peak $2 --lr_min_factor inf --augment True --warmup_exponent 1.0 --decay_schedule_name cosine --decay_exponent 1.0 --warmup_steps 2000 --num_steps 10_000 --batch_size 128 --momentum 0.0 --weight_decay 0.0 --save_ckpt True --rerun True" > submit_job.sh
