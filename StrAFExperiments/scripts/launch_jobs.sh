model_name=(straf_umnn gnf_umnn)
nf_steps=(5 10 15)
cond_width=(200 300)
cond_depth=(5 10)
lrs=(1e-3 1e-4)

for mn in ${model_name[@]}
do
    for nfs in ${nf_steps[@]}
    do
        for w in ${cond_width[@]}
        do
            for d in ${cond_depth[@]}
            do
                for lr in ${lrs[@]}
                do
                    sbatch run_exp.sh $mn $nfs $w $d $lr
                done
            done
        done
    done
done