#!/bin/bash
trap "exit" INT
dataset=${1?:Error: TOMAS/DNP/PA}
model_group=${2?:Error: xception/unified_cnn/resnet}
detection=${3?:Error: True/False}
quantification=${4?:Error: True/False}
version_g=${5?:Error: experiment version}
perc_use=${6?:Error: the percentage usage}
gpu_index=${7?:Error: the selected gpu index}


model_method=top_peak
bs=120
num_gpu=1
epoch=300
use_map=false 
avg_spectra=true 

if [ "$dataset" == PA ]; then
  concentration_float=1e-5
else
  concentration_float=1e-6
fi
cast_quantification_to_cls=false 

export CUDA_VISIBLE_DEVICES="$gpu_index"


if [ "$dataset" == TOMAS ] || [ "$dataset" == DNP ] || [ "$dataset" == PA ]; then
  if [ "$dataset" == TOMAS ]; then
    targ_h=56
  elif [ "$dataset" == DNP ]; then  
    targ_h=44
  elif [ "$dataset" == PA ]; then
    targ_h=40
  fi 
  normalization=none 
  if [ "$detection" == true ]; then 
    quantification_loss=none 
  else
    quantification_loss=mse 
  fi
else
  targ_h=30
  if [ "$detection" is true ]; then 
    quantification_loss=none 
  else
    quantification_loss=mae 
  fi 
  normalization=none 
fi 

for version in $version_g
do
  for model in $model_group
  do
    seed_use=$RANDOM
    if [ "$dataset" != TOMAS ] && [ "$dataset" != DNP ] && [ "$dataset" != PA ]; then 
      if [ "$detection" is true ]; then
        if [ "$model" == xception ]; then
          if [ "$dataset" == SIMU_TYPE_2 ]; then
            lr=0.0008
          else
            lr=0.0005
          fi
        elif [ "$model" == unified_cnn ]; then 
          if ["$dataset" == SIMU_TYPE_2 ] || [ "$dataset" == SIMU_TYPE_3 ]; then 
            lr=0.001
          else
            lr=0.0005
          fi 
        elif [ "$model" == resnet ]; then
          if [ "$dataset" == SIMU_TYPE_3 ]; then
            lr=0.01
          else
            lr=0.008
          fi
        fi 
      elif [ "$quantification" is true ]; then 
        if [ "$model" == xception ]; then
          lr=0.002
        elif [ "$model" == unified_cnn ]; then 
          lr=0.1
        elif [ "$model" == resnet ]; then 
          lr=0.008
        fi 
      fi
    elif [ "$dataset" == TOMAS ]
    then
      if [ "$detection" == true ]; then 
        if [ "$model" == resnet ] || [ "$model" == xception ]; then
          lr=0.0008
        else
          lr=0.008
        fi
      elif [ "$detection" == false ]; then 
        lr=0.08
      fi
    elif [ "$dataset" == DNP ]
    then
      if [ "$detection" == true ]; then 
        if [ "$model" == xception ]; then
          lr=0.002
        elif [ "$model" == resnet ]; then 
          lr=0.002
        else
          lr=0.002
        fi
      elif [ "$detection" == false ]; then 
        if [ "$model" == xception ] || [ "$model" == unified_cnn ]; then 
          lr=0.02
        elif [ "$model" == resnet ]; then 
          lr=0.02
        fi
      fi
    elif [ "$dataset" == PA ]
    then
      if [ "$detection" == true ]; then 
        if [ "$model" == resnet ]; then
          lr=0.06
        elif [ "$model" == xception ]; then
          lr=0.04
        else
          lr=0.06
        fi
      elif [ "$detection" == false ]; then 
        if [ "$model" == xception ]; then 
          lr=0.004
        elif [ "$model" == unified_cnn ]; then 
          lr=0.02 
        else
          lr=0.04
        fi
      fi
    fi
    for s_method in $model_method 
    do
      ./run_spectra.sh $num_gpu "$bs" $perc_use "$version" \
        "$dataset" "$lr" "$epoch" dp '0' "$quantification" "$targ_h" "$targ_h" $s_method "$model" \
        "$use_map" "$avg_spectra" "$detection" "$concentration_float" "$cast_quantification_to_cls" \
        "$normalization" "$quantification_loss" "$seed_use" "$gpu_index"
    done
  done
done

