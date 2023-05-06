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

python run_multimodal_exp.py --dataset multimodal_1 --n_samp 20000 --model_name $1 --nf_steps $2 --cond_width $3 --cond_depth $4 --batch_size 256 --lr $5 