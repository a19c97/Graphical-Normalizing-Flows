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

python run_sample_experiments.py --dataset multimodal_s --model_name $1 --train_config multimodal_multistep --random_seed $2
