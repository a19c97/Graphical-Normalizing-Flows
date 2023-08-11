#!/bin/bash

#SBATCH -N 1
#SBATCH --partition rtx6000,t4v2
#SBATCH -c 4
#SBATCH --mem=16GB
#SBATCH --output=./logs/out_%j.txt
#SBATCH --gres=gpu:1
#SBATCH --qos=normal

module load cuda-11.7
cd ..

python evaluate_best.py --model_tag $1
