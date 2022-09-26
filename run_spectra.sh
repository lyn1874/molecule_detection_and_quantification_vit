#!/bin/bash
trap "exit" INT
num_gpu=${1?:Error: the number of gpus}
batch_size=${2?:Error: batch_size default 512?}
percentage=${3?:Error: the percentage of data that I am using}
version=${4?:Error: int}
dataset=${5?:Error: CIFAR10/SHAPE/Marlitt}
enc_lr=${6?:Error: learning rate 0.03 default}
epoch=${7?:Error: what is the number of epochs}
strategy=${8?:Error: dp/ddp when multiple gpu is used}
tr_limited_conc=${9?:Error: the concentration for training}
quantification=${10?:Error: using quantification or not}
target_h=${11?:Error: target height 45/40}
target_w=${12?:Error: target width 75/40}
top_selection_method=${13?:Error the selection method: top_mean, top_std, top_diff, all, avg_map_dim}
model_type=${14?:Error: xception/ unified_cnn}
use_map=${15?:Error: if use map or not}
avg_spectra=${16?:Error: if averaging the spectra}
detection=${17?:Error: whether detection or not}
concentration_float=${18?:Error: the concentration float}
cast_quantification_to_classification=${19?:Error: true/false}
normalization=${20?:Error: what is the normalization technique? max/none}
quantification_loss=${21?:Error: what is the quantification loss, mae}
seed_use=${22?:Error: the random seed}
gpu_index=${23?:Error: specify the gpu index}
loc_use=${24:-nobackup}
echo "$model_type"

if [ "$percentage" == 1.0 ]; then
  top_selection_method=avg_map_dim
fi

echo $top_selection_method

if [ "$dataset" != TOMAS ] && [ "$dataset" != DNP ] && [ "$dataset" != PA ]
then
  for s_percentage in $percentage
  do
    python3 train_spectra.py --batch_size "$batch_size" \
            --num_gpu "$num_gpu" \
            --version "$version" --dataset "$dataset" --enc_lr "$enc_lr" --strategy "$strategy" \
            --tr_limited_conc $tr_limited_conc --model_type "$model_type" \
            --quantification "$quantification" --bg_method ar --target_h "$target_h" \
            --target_w "$target_w" --use_wave_as_batch false --epoch "$epoch" \
            --top_selection_method "$top_selection_method" --percentage "$s_percentage" \
            --use_map "$use_map" --avg_spectra "$avg_spectra" --detection "$detection" --concentration_float "$concentration_float" \
            --cast_quantification_to_classification "$cast_quantification_to_classification" --warmup_epochs 50 --loc "$loc_use" \
            --normalization "$normalization" --quantification_loss "$quantification_loss" --seed_use "$seed_use" --gpu_index "$gpu_index"
  done
else
  for s_percentage in $percentage
  do
    if [ "$dataset" != PA ]; then
      for leave_index in {0..29} 
  # {0..29} 
      do
          python3 train_spectra.py --batch_size "$batch_size" \
                  --num_gpu "$num_gpu" \
                  --version "$version" --dataset "$dataset" --enc_lr "$enc_lr" --strategy "$strategy" \
                  --tr_limited_conc $tr_limited_conc --model_type "$model_type" \
                  --quantification "$quantification" --bg_method ar --target_h "$target_h" \
                  --target_w "$target_w" --use_wave_as_batch false --epoch "$epoch" \
                  --top_selection_method "$top_selection_method" --percentage "$s_percentage" \
                  --use_map "$use_map" --avg_spectra "$avg_spectra" --detection "$detection" --concentration_float "$concentration_float" \
                  --cast_quantification_to_classification "$cast_quantification_to_classification" --warmup_epochs 50 \
                  --normalization "$normalization" --leave_index "$leave_index" --quantification_loss "$quantification_loss" \
                  --seed_use "$seed_use" --loc "$loc_use" --gpu_index "$gpu_index"
      done 
    else
      for leave_index in {0..19}  
  # {0..29} 
      do
          python3 train_spectra.py --batch_size "$batch_size" \
                  --num_gpu "$num_gpu" \
                  --version "$version" --dataset "$dataset" --enc_lr "$enc_lr" --strategy "$strategy" \
                  --tr_limited_conc $tr_limited_conc --model_type "$model_type" \
                  --quantification "$quantification" --bg_method ar --target_h "$target_h" \
                  --target_w "$target_w" --use_wave_as_batch false --epoch "$epoch" \
                  --top_selection_method "$top_selection_method" --percentage "$s_percentage" \
                  --use_map "$use_map" --avg_spectra "$avg_spectra" --detection "$detection" --concentration_float "$concentration_float" \
                  --cast_quantification_to_classification "$cast_quantification_to_classification" --warmup_epochs 50 \
                  --normalization "$normalization" --leave_index "$leave_index" --quantification_loss "$quantification_loss" \
                  --seed_use "$seed_use" --loc "$loc_use" --gpu_index "$gpu_index"
      done 
    fi
  done 
fi    

# {0..29}



