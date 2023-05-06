#!/bin/bash

#SBATCH -N 1
#SBATCH -p gpu
#SBATCH -c 4
#SBATCH --mem=16GB
#SBATCH --output=./logs/out_%j.txt
#SBATCH --gres=gpu:1

source activate diffeq
module load cuda-10.0

cd ..

python run_grid.py --dataset multimodal_s --model_name $1 --train_config multimodal_grid --cond_width $2 --cond_depth $3 --nf_step $4 --inet_width $5 --inet_depth $6 --h_size $7 --lr $8 --scheduler $9
