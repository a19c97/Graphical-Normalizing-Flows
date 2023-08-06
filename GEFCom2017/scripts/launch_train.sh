full_ar=(False True)
lr=(5e-3)
hid=([500,500])
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
