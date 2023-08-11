model_tag=(best_daphne)

for mt in ${model_tag[@]}
do
    sbatch launch_eval_loss.sh $mt
done
