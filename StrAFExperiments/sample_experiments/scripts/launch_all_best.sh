model_tag=(best_daphne)
seed=(2541 2547 413 412 411 321 421 311)
for mt in ${model_tag[@]}
do
    for s in ${seed[@]}
    do
        sbatch launch_best.sh $mt $s
    done
done
