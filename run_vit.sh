#!/bin/bash
trap "exit" INT
num_gpu=${1?:Error: the number of gpus}
batch_size=${2?:Error: batch_size default 512?}
model_type=${3?:Error: b16 or s16}
epoch=${4?:Error: the number of epochs}
version=${5?:Error: int}
dataset=${6?:Error: CIFAR10/SHAPE/Marlitt/DMMP}
enc_lr=${7?:Error: learning rate 0.03 default}
use_wave_as_batch=${8?:Error: true/false}
patch_height=${9?:Error: 16}
patch_width=${10?:Error: 1}
strategy=${11?:Error: dp/ddp when multiple gpu is used}
tr_limited_conc=${12?:Error: the concentration for training}
quantification=${13?:Error: using quantification or not}
target_h=${14?:Error: what is the height of the image}
target_w=${15?:Error: what is the width of the image}
concentration_float=${16?:Error: what is the concentration float }
detection=${17?:Error: true/false}
cast=${18?:Error: false}
warmup_epoch=${19?:Error: 50}
normalization=${20?:Error: none}
quantification_loss=${21?:Error: what is the quantification loss, mae/rae/rmae}
leave_index_g=${22?:Error: the leave index group}
eval_with_val=${23?:Error: evaluation with validation dataset}
skip_value=${24?:Error: the skip value}
leave_method=${25?:Error: the leave one method for Tomas dataset, leave_one_chip_per_conc/leave_one_chip}
augment_seed=${26?:Error: the random seed for augmentation}
exp_seed=${27?:Error: the initialisation seed for the experiment}
gpu_index=${28?:Error: the gpu index if use the same gpu}
lr_schedule=${29:-cosine}
model_init=${30:-xavier}

loc_use=scratch

if [ "$dataset" != TOMAS ] && [ "$dataset" != DNP ] && [ "$dataset" != PA ]
then
        echo "$dataset"
        python3 train_vit.py --batch_size "$batch_size" --model_type "$model_type" \
                --add_positional_encoding True --num_gpu "$num_gpu" \
                --version "$version" --dataset "$dataset" --enc_lr "$enc_lr" --patch_height "$patch_height" \
                --patch_width "$patch_width" --strategy "$strategy" --use_wave_as_batch "$use_wave_as_batch" \
                --tr_limited_conc $tr_limited_conc --epoch "$epoch" --quantification "$quantification" \
                --bg_method ar --target_h "$target_h" --target_w "$target_w" \
                --top_selection_method sers_maps --percentage 0 --concentration_float "$concentration_float"  \
                --detection "$detection" --cast_quantification_to_classification "$cast" --warmup_epochs "$warmup_epoch" \
                --normalization "$normalization" --quantification_loss "$quantification_loss" \
                --eval_with_val "$eval_with_val" --seed_use "$exp_seed" --augment_seed_use "$augment_seed" --loc "$loc_use" \
                --lr_schedule "$lr_schedule" --model_init "$model_init"
else
        for leave_index in $leave_index_g
        do 
                python3 train_vit.py --batch_size "$batch_size" --model_type "$model_type" \
                        --add_positional_encoding True --num_gpu "$num_gpu" \
                        --version "$version" --dataset "$dataset" --enc_lr "$enc_lr" --patch_height "$patch_height" \
                        --patch_width "$patch_width" --strategy "$strategy" --use_wave_as_batch "$use_wave_as_batch" \
                        --tr_limited_conc $tr_limited_conc --epoch "$epoch" --quantification "$quantification" \
                        --bg_method ar --target_h "$target_h" --target_w "$target_w" \
                        --top_selection_method sers_maps --percentage 0 --concentration_float "$concentration_float"  \
                        --detection "$detection" --cast_quantification_to_classification "$cast" --warmup_epochs "$warmup_epoch" \
                        --normalization "$normalization" --leave_index "$leave_index" --quantification_loss "$quantification_loss" \
                        --eval_with_val "$eval_with_val" --skip_value "$skip_value" --leave_method "$leave_method" --loc "$loc_use" \
                        --augment_seed_use "$augment_seed" --seed_use "$exp_seed" --lr_schedule "$lr_schedule" \
                        --model_init "$model_init"
        done
fi






