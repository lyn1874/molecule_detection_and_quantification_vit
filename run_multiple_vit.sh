#!/bin/bash
trap "exit" INT
detection=${1?:Error: the number of gpus}
quantification=${2?:Error: batch_size default 512?}
dataset=${3?:Error: what is the name of the dataset?}
gpu_index=${4?:Error: the gpu index}


export CUDA_VISIBLE_DEVICES="$gpu_index"

if [ "$dataset" == TOMAS ]; then
    target_size=56
elif [ "$dataset" == DNP ]; then 
    target_size=44
elif [ "$dataset" == PA ]; then 
    target_size=40
else
    target_size=30
fi

if [ "$quantification" == true ]; then
    if [ "$dataset" == PA ]; then
        concentration_float=1e-5 
    else
        concentration_float=1e-6 
    fi
else
    concentration_float=0 
fi
skip_value=1 
bs=120
augment_seed=24907

if [ "$dataset" == TOMAS ]
then
    if [ "$quantification" == true ]
    then
        seed_group="6338 5353 12885 1746 8543"
        lr=0.08
        normalization=none 
        quantification_loss=mse
    elif [ "$quantification" == false ]
    then
        seed_group="30582 1066 30186 4731 2814"
        lr=0.008
        normalization=none 
        quantification_loss=none
    fi 
    for seed_use in $seed_group
    do
        ./run_vit.sh 1 "$bs" eetti16 300 "$seed_use" TOMAS "$lr" \
            false 2 2 dp '0' "$quantification" "$target_size" "$target_size" "$concentration_float" \
            "$detection" false 50 "$normalization" "$quantification_loss" \
            '0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24' \
            true "$skip_value" leave_one_chip "$augment_seed" "$seed_use" \
            "$gpu_index"
        if [ "$detection" == true ]; then 
            ./run_vit.sh 1 "$bs" eetti16 300 "$seed_use" TOMAS "$lr" \
                false 2 2 dp '0' "$quantification" "$target_size" "$target_size" "$concentration_float" \
                "$detection" false 50 "$normalization" "$quantification_loss" \
                '25 26 27 28 29' \
                true "$skip_value" leave_one_chip "$augment_seed" "$seed_use" \
                "$gpu_index"
        fi     
    done
elif [ "$dataset" == DNP ]
then 
    if [ "$quantification" == true ]
    then
        lr=0.006
        normalization=none 
        quantification_loss=mse
        seed_group="9860 19800 12266 13579 29055"
    elif [ "$quantification" == false ]
    then
        lr=0.005
        normalization=none 
        quantification_loss=none
        seed_group="32619 9093 31050 10637 31797"
    fi 
    for seed_use in $seed_group
    do
        ./run_vit.sh 1 "$bs" eetti16 300 "$seed_use" DNP "$lr" \
            false 2 2 dp '0' "$quantification" "$target_size" "$target_size" "$concentration_float" \
            "$detection" false 50 "$normalization" "$quantification_loss" \
            '0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24' \
            true "$skip_value" leave_one_chip "$augment_seed" "$seed_use" \
            "$gpu_index"
        if [ "$detection" == true ]; then 
            ./run_vit.sh 1 "$bs" eetti16 300 "$seed_use" DNP "$lr" \
                false 2 2 dp '0' "$quantification" "$target_size" "$target_size" "$concentration_float" \
                "$detection" false 50 "$normalization" "$quantification_loss" \
                '25 26 27 28 29' \
                true "$skip_value" leave_one_chip "$augment_seed" "$seed_use" \
                "$gpu_index"
        fi
    done 
elif [ "$dataset" == PA ]
then 
    if [ "$quantification" == true ]
    then
        lr=0.0006
        normalization=none 
        quantification_loss=mse
        seed_group="30541 989 31058 18197 30988"
    elif [ "$quantification" == false ]
    then
        lr=0.0006
        normalization=none
        quantification_loss=none
        seed_group="12759 5260 20202 20887 4209"
    fi 
    if [ "$quantification" == true ]
    then
        for seed_use in $seed_group
        do
            ./run_vit.sh 1 "$bs" eetti16 700 "$seed_use" "$dataset" "$lr" \
                false 2 2 dp '0' "$quantification" "$target_size" "$target_size" "$concentration_float" \
                "$detection" false 50 "$normalization" "$quantification_loss" \
                '0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19' \
                true "$skip_value" leave_one_chip "$augment_seed" "$seed_use" \
                "$gpu_index" 
        done 
    else
        for seed_use in $seed_group
        do
            ./run_vit.sh 1 "$bs" eetti16 500 "$seed_use" "$dataset" "$lr" \
                false 2 2 dp '0' "$quantification" "$target_size" "$target_size" "$concentration_float" \
                "$detection" false 50 "$normalization" "$quantification_loss" \
                "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24" \
                true "$skip_value" leave_one_chip "$augment_seed" "$seed_use" \
                "$gpu_index" 
        done
    fi 
else
# then     
    if [ "$quantification" == true ]; then
        lr=0.08
        quantification_loss=mae
        normalization=none 
        if [ "$dataset" == SIMU_TYPE_12 ]; then
            seed_group="1044 4879 8276 9567 23534"
        elif [ "$dataset" == SIMU_TYPE_13 ]; then
            seed_group="200 7707 11289 14152 32373"
        elif [ "$dataset" == SIMU_TYPE_14 ]; then
            seed_group="7106 7578 8400 12155 20624"
        fi
    elif [ "$quantification" == false ]; then
        lr=0.2
        quantification_loss=none
        normalization=none 
        if [ "$dataset" == SIMU_TYPE_2 ]; then
            seed_group="8799 17496 8810 16011 12964"
        elif [ "$dataset" == SIMU_TYPE_3 ]; then
            seed_group="4929 5192 16725 24962 15858"
        elif [ "$dataset" == SIMU_TYPE_4 ]; then
            seed_group="1905 6690 7508 22605 12031"
        fi
    fi
    for seed_use in $seed_group
    do
        ./run_vit.sh 1 "$bs" eetti16 300 "$seed_use" "$dataset" "$lr" \
            false 2 2 dp '0' "$quantification" "$target_size" "$target_size" "$concentration_float" \
            "$detection" false 50 "$normalization" "$quantification_loss" 0 \
            true 1 leave_one_chip "$augment_seed" "$seed_use" \
            "$gpu_index"
    done
fi




