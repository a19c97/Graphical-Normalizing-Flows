model_name=(gnf_umnn_s_large)
seed=(2515)
for mn in ${model_name[@]}
do
    for s in ${seed[@]}
    do
        sbatch launch_exp.sh $mn $s
    done
done
