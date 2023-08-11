full_ar=(False)
lr=(3e-3)
hid=([5000,5000])
steps=(5)
for f in ${full_ar[@]}
do
    for l in ${lr[@]}
    do
        for h in ${hid[@]}
        do  
            for s in ${steps[@]}
            do
                sbatch dist_train.sh $f $l $h $s
            done
        done
    done
done
