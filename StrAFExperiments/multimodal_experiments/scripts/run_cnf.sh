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

python run_multimodal_cnf.py --dataset multimodal_s --model_name $1 \
    --nf_steps $2 --hidden_width $3 --hidden_depth $4 --batch_size 256 --lr $5
