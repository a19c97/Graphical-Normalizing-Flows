model_name=(gnf_umnn_grid straf_umnn_grid gnf_f_umnn_grid)
cond_width=(50 500)
cond_depth=(3 4)
nf_step=(5 10)
inet_width=(250 500)
inet_depth=(4 6)
h_size=(25 50)
lrs=(1e-3 1e-4)
scheduler=(ReduceLROnPlateau MultiStep None)

for mn in ${model_name[@]}
do
    for cw in ${cond_width[@]}
    do
        for cnd in ${cond_depth[@]}
        do
            for ns in ${nf_step[@]}
            do
                for iw in ${inet_width[@]}
                do
                    for id in ${inet_depth[@]}
                    do
                        for h in ${h_size[@]}
                        do
                            for lr in ${lrs[@]}
                            do
                                for s in ${scheduler[@]}
                                do
                                    sbatch launch_grid.sh $mn $cw $cnd $ns $iw $id $h $lr $s
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
